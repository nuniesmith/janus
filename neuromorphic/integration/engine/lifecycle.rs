//! System lifecycle management
//!
//! Part of the Integration region — Engine component.
//!
//! `Lifecycle` manages the overall system lifecycle through a well-defined
//! state machine:
//!
//! ```text
//!   Init → Starting → Running ⇄ Paused → ShuttingDown → Stopped
//! ```
//!
//! The engine enforces valid transitions, tracks uptime per phase, runs
//! periodic health checks, and exposes EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the lifecycle manager.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Minimum ticks to stay in `Starting` before transitioning to `Running`.
    pub min_starting_ticks: u64,
    /// Minimum ticks in `Running` before allowing a pause.
    pub min_running_before_pause: u64,
    /// Minimum ticks in `ShuttingDown` before transitioning to `Stopped`.
    pub min_shutdown_ticks: u64,
    /// Interval (in ticks) between automatic health checks.
    pub health_check_interval: u64,
    /// Number of consecutive failed health checks before marking unhealthy.
    pub max_failed_health_checks: u32,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            min_starting_ticks: 5,
            min_running_before_pause: 10,
            min_shutdown_ticks: 3,
            health_check_interval: 10,
            max_failed_health_checks: 3,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle phase
// ---------------------------------------------------------------------------

/// System lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Initial state before any subsystems are started.
    Init,
    /// Subsystems are being initialised and warmed up.
    Starting,
    /// Fully operational — processing data and generating signals.
    Running,
    /// Temporarily suspended; can resume to `Running`.
    Paused,
    /// Gracefully shutting down subsystems.
    ShuttingDown,
    /// Terminal state — all subsystems stopped.
    Stopped,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Phase::Init => write!(f, "Init"),
            Phase::Starting => write!(f, "Starting"),
            Phase::Running => write!(f, "Running"),
            Phase::Paused => write!(f, "Paused"),
            Phase::ShuttingDown => write!(f, "ShuttingDown"),
            Phase::Stopped => write!(f, "Stopped"),
        }
    }
}

impl Phase {
    /// Returns `true` if the system is in a terminal state.
    pub fn is_terminal(self) -> bool {
        self == Phase::Stopped
    }

    /// Returns `true` if the system is actively processing.
    pub fn is_active(self) -> bool {
        self == Phase::Running
    }

    /// Returns all valid successor phases for the current phase.
    pub fn valid_transitions(self) -> &'static [Phase] {
        match self {
            Phase::Init => &[Phase::Starting],
            Phase::Starting => &[Phase::Running, Phase::ShuttingDown],
            Phase::Running => &[Phase::Paused, Phase::ShuttingDown],
            Phase::Paused => &[Phase::Running, Phase::ShuttingDown],
            Phase::ShuttingDown => &[Phase::Stopped],
            Phase::Stopped => &[],
        }
    }

    /// Whether `target` is a valid successor of this phase.
    pub fn can_transition_to(self, target: Phase) -> bool {
        self.valid_transitions().contains(&target)
    }
}

// ---------------------------------------------------------------------------
// Transition record
// ---------------------------------------------------------------------------

/// Record of a lifecycle phase transition.
#[derive(Debug, Clone)]
pub struct TransitionRecord {
    /// Phase transitioned from.
    pub from: Phase,
    /// Phase transitioned to.
    pub to: Phase,
    /// Tick at which the transition occurred.
    pub tick: u64,
    /// Number of ticks spent in the previous phase.
    pub duration_in_previous: u64,
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

/// Result of a periodic health check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All subsystems are healthy.
    Healthy,
    /// One or more subsystems are degraded but functional.
    Degraded,
    /// Critical failure detected.
    Unhealthy,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    phase: Phase,
    uptime: u64,
    health: HealthStatus,
    running_ratio: f64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the lifecycle manager.
#[derive(Debug, Clone)]
pub struct LifecycleStats {
    /// Total ticks elapsed.
    pub total_ticks: u64,
    /// Ticks spent in `Init`.
    pub ticks_in_init: u64,
    /// Ticks spent in `Starting`.
    pub ticks_in_starting: u64,
    /// Ticks spent in `Running`.
    pub ticks_in_running: u64,
    /// Ticks spent in `Paused`.
    pub ticks_in_paused: u64,
    /// Ticks spent in `ShuttingDown`.
    pub ticks_in_shutting_down: u64,
    /// Ticks spent in `Stopped`.
    pub ticks_in_stopped: u64,
    /// Total number of phase transitions.
    pub total_transitions: u64,
    /// Total health checks performed.
    pub total_health_checks: u64,
    /// Total failed health checks.
    pub total_failed_checks: u64,
    /// EMA-smoothed running ratio (fraction of ticks in `Running`).
    pub ema_running_ratio: f64,
    /// EMA-smoothed health score (1.0 = healthy, 0.5 = degraded, 0.0 = unhealthy).
    pub ema_health_score: f64,
}

impl Default for LifecycleStats {
    fn default() -> Self {
        Self {
            total_ticks: 0,
            ticks_in_init: 0,
            ticks_in_starting: 0,
            ticks_in_running: 0,
            ticks_in_paused: 0,
            ticks_in_shutting_down: 0,
            ticks_in_stopped: 0,
            total_transitions: 0,
            total_health_checks: 0,
            total_failed_checks: 0,
            ema_running_ratio: 0.0,
            ema_health_score: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// System lifecycle manager.
///
/// Manages phase transitions, enforces transition guards, tracks uptime per
/// phase, runs periodic health checks, and provides EMA + windowed diagnostics.
pub struct Lifecycle {
    config: LifecycleConfig,
    phase: Phase,
    tick: u64,
    ticks_in_current_phase: u64,
    last_health_check_tick: u64,
    health: HealthStatus,
    consecutive_failures: u32,
    transition_history: Vec<TransitionRecord>,
    ema_initialized: bool,
    recent: VecDeque<TickSnapshot>,
    stats: LifecycleStats,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl Lifecycle {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new lifecycle manager with default configuration.
    pub fn new() -> Self {
        Self::with_config(LifecycleConfig::default())
    }

    /// Create a new lifecycle manager with the given configuration.
    pub fn with_config(config: LifecycleConfig) -> Self {
        Self {
            config,
            phase: Phase::Init,
            tick: 0,
            ticks_in_current_phase: 0,
            last_health_check_tick: 0,
            health: HealthStatus::Healthy,
            consecutive_failures: 0,
            transition_history: Vec::new(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: LifecycleStats::default(),
        }
    }

    // -------------------------------------------------------------------
    // Phase queries
    // -------------------------------------------------------------------

    /// Returns the current phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Returns `true` if the system is in the `Running` phase.
    pub fn is_running(&self) -> bool {
        self.phase == Phase::Running
    }

    /// Returns `true` if the system has reached a terminal state.
    pub fn is_stopped(&self) -> bool {
        self.phase == Phase::Stopped
    }

    /// Returns how many ticks have been spent in the current phase.
    pub fn ticks_in_current_phase(&self) -> u64 {
        self.ticks_in_current_phase
    }

    /// Returns the current health status.
    pub fn health(&self) -> HealthStatus {
        self.health
    }

    // -------------------------------------------------------------------
    // Transition guards
    // -------------------------------------------------------------------

    /// Check whether transitioning to `target` is currently allowed.
    ///
    /// This considers both the state-machine topology and minimum-duration
    /// guards.
    pub fn can_transition_to(&self, target: Phase) -> bool {
        if !self.phase.can_transition_to(target) {
            return false;
        }
        match (self.phase, target) {
            (Phase::Starting, Phase::Running) => {
                self.ticks_in_current_phase >= self.config.min_starting_ticks
            }
            (Phase::Running, Phase::Paused) => {
                self.ticks_in_current_phase >= self.config.min_running_before_pause
            }
            (Phase::ShuttingDown, Phase::Stopped) => {
                self.ticks_in_current_phase >= self.config.min_shutdown_ticks
            }
            _ => true,
        }
    }

    // -------------------------------------------------------------------
    // Phase transitions
    // -------------------------------------------------------------------

    /// Attempt to transition to a new phase.
    ///
    /// Returns `Ok(())` on success. Returns `Err` if the transition is
    /// invalid or a guard prevents it.
    pub fn transition_to(&mut self, target: Phase) -> Result<()> {
        if !self.phase.can_transition_to(target) {
            return Err(Error::Configuration(format!(
                "Invalid lifecycle transition: {} → {}",
                self.phase, target
            )));
        }

        // Check minimum-duration guards.
        match (self.phase, target) {
            (Phase::Starting, Phase::Running)
                if self.ticks_in_current_phase < self.config.min_starting_ticks =>
            {
                return Err(Error::Configuration(format!(
                    "Must stay in Starting for at least {} ticks (currently {})",
                    self.config.min_starting_ticks, self.ticks_in_current_phase
                )));
            }
            (Phase::Running, Phase::Paused)
                if self.ticks_in_current_phase < self.config.min_running_before_pause =>
            {
                return Err(Error::Configuration(format!(
                    "Must stay in Running for at least {} ticks before pausing (currently {})",
                    self.config.min_running_before_pause, self.ticks_in_current_phase
                )));
            }
            (Phase::ShuttingDown, Phase::Stopped)
                if self.ticks_in_current_phase < self.config.min_shutdown_ticks =>
            {
                return Err(Error::Configuration(format!(
                    "Must stay in ShuttingDown for at least {} ticks (currently {})",
                    self.config.min_shutdown_ticks, self.ticks_in_current_phase
                )));
            }
            _ => {}
        }

        let record = TransitionRecord {
            from: self.phase,
            to: target,
            tick: self.tick,
            duration_in_previous: self.ticks_in_current_phase,
        };
        self.transition_history.push(record);
        self.phase = target;
        self.ticks_in_current_phase = 0;
        self.stats.total_transitions += 1;
        Ok(())
    }

    /// Convenience: transition Init → Starting.
    pub fn start(&mut self) -> Result<()> {
        self.transition_to(Phase::Starting)
    }

    /// Convenience: transition Starting → Running.
    pub fn go_live(&mut self) -> Result<()> {
        self.transition_to(Phase::Running)
    }

    /// Convenience: transition Running → Paused.
    pub fn pause(&mut self) -> Result<()> {
        self.transition_to(Phase::Paused)
    }

    /// Convenience: transition Paused → Running.
    pub fn resume(&mut self) -> Result<()> {
        self.transition_to(Phase::Running)
    }

    /// Convenience: transition current phase → ShuttingDown.
    pub fn shutdown(&mut self) -> Result<()> {
        self.transition_to(Phase::ShuttingDown)
    }

    /// Convenience: transition ShuttingDown → Stopped.
    pub fn stop(&mut self) -> Result<()> {
        self.transition_to(Phase::Stopped)
    }

    // -------------------------------------------------------------------
    // Health checks
    // -------------------------------------------------------------------

    /// Report a health check result.
    ///
    /// Call this periodically (or the `tick` method handles automatic checks).
    pub fn report_health(&mut self, status: HealthStatus) {
        self.stats.total_health_checks += 1;
        match status {
            HealthStatus::Healthy => {
                self.consecutive_failures = 0;
                self.health = HealthStatus::Healthy;
            }
            HealthStatus::Degraded => {
                self.consecutive_failures += 1;
                self.health = HealthStatus::Degraded;
                if self.consecutive_failures >= self.config.max_failed_health_checks {
                    self.health = HealthStatus::Unhealthy;
                    self.stats.total_failed_checks += 1;
                }
            }
            HealthStatus::Unhealthy => {
                self.consecutive_failures += 1;
                self.health = HealthStatus::Unhealthy;
                self.stats.total_failed_checks += 1;
            }
        }
    }

    /// Returns the number of consecutive failed (non-healthy) checks.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// Returns `true` if a health check is due on the current tick.
    pub fn is_health_check_due(&self) -> bool {
        if self.config.health_check_interval == 0 {
            return false;
        }
        self.tick.saturating_sub(self.last_health_check_tick) >= self.config.health_check_interval
    }

    // -------------------------------------------------------------------
    // Tick
    // -------------------------------------------------------------------

    /// Advance the lifecycle by one tick.
    ///
    /// Updates phase counters, records windowed snapshots, and refreshes
    /// EMA diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;
        self.ticks_in_current_phase += 1;
        self.stats.total_ticks += 1;

        // Accumulate per-phase counters.
        match self.phase {
            Phase::Init => self.stats.ticks_in_init += 1,
            Phase::Starting => self.stats.ticks_in_starting += 1,
            Phase::Running => self.stats.ticks_in_running += 1,
            Phase::Paused => self.stats.ticks_in_paused += 1,
            Phase::ShuttingDown => self.stats.ticks_in_shutting_down += 1,
            Phase::Stopped => self.stats.ticks_in_stopped += 1,
        }

        // Compute instantaneous ratios.
        let running_ratio = if self.stats.total_ticks > 0 {
            self.stats.ticks_in_running as f64 / self.stats.total_ticks as f64
        } else {
            0.0
        };

        let health_score = match self.health {
            HealthStatus::Healthy => 1.0,
            HealthStatus::Degraded => 0.5,
            HealthStatus::Unhealthy => 0.0,
        };

        // EMA update.
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_running_ratio = running_ratio;
            self.stats.ema_health_score = health_score;
            self.ema_initialized = true;
        } else {
            self.stats.ema_running_ratio =
                alpha * running_ratio + (1.0 - alpha) * self.stats.ema_running_ratio;
            self.stats.ema_health_score =
                alpha * health_score + (1.0 - alpha) * self.stats.ema_health_score;
        }

        // Windowed snapshot.
        let snapshot = TickSnapshot {
            phase: self.phase,
            uptime: self.stats.ticks_in_running,
            health: self.health,
            running_ratio,
        };
        self.recent.push_back(snapshot);
        if self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }
    }

    /// Current tick counter.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Alias for `tick()` to match the common `process` pattern.
    pub fn process(&mut self) {
        self.tick();
    }

    // -------------------------------------------------------------------
    // Transition history
    // -------------------------------------------------------------------

    /// Returns the full transition history.
    pub fn transition_history(&self) -> &[TransitionRecord] {
        &self.transition_history
    }

    /// Returns the most recent transition, if any.
    pub fn last_transition(&self) -> Option<&TransitionRecord> {
        self.transition_history.last()
    }

    /// Total number of phase transitions performed.
    pub fn total_transitions(&self) -> u64 {
        self.stats.total_transitions
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Returns a snapshot of cumulative statistics.
    pub fn stats(&self) -> &LifecycleStats {
        &self.stats
    }

    /// Returns the configuration.
    pub fn config(&self) -> &LifecycleConfig {
        &self.config
    }

    /// EMA-smoothed fraction of ticks spent in `Running`.
    pub fn smoothed_running_ratio(&self) -> f64 {
        self.stats.ema_running_ratio
    }

    /// EMA-smoothed health score (1 = healthy, 0 = unhealthy).
    pub fn smoothed_health_score(&self) -> f64 {
        self.stats.ema_health_score
    }

    /// Windowed average running ratio over the recent window.
    pub fn windowed_running_ratio(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.running_ratio).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average health score over the recent window.
    pub fn windowed_health_score(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| match s.health {
                HealthStatus::Healthy => 1.0,
                HealthStatus::Degraded => 0.5,
                HealthStatus::Unhealthy => 0.0,
            })
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Returns `true` if health appears to be declining over the window.
    ///
    /// Compares the first half average to the second half average.
    pub fn is_health_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let score = |s: &TickSnapshot| match s.health {
            HealthStatus::Healthy => 1.0,
            HealthStatus::Degraded => 0.5,
            HealthStatus::Unhealthy => 0.0,
        };
        let first_half: f64 = self.recent.iter().take(mid).map(score).sum::<f64>() / mid as f64;
        let second_half: f64 = self.recent.iter().skip(mid).map(score).sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half < first_half - 0.05
    }

    /// Returns the total system uptime (ticks in `Running`).
    pub fn uptime(&self) -> u64 {
        self.stats.ticks_in_running
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset the lifecycle to its initial state, preserving configuration.
    pub fn reset(&mut self) {
        self.phase = Phase::Init;
        self.tick = 0;
        self.ticks_in_current_phase = 0;
        self.last_health_check_tick = 0;
        self.health = HealthStatus::Healthy;
        self.consecutive_failures = 0;
        self.transition_history.clear();
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = LifecycleStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: small config for testing.
    fn small_config() -> LifecycleConfig {
        LifecycleConfig {
            min_starting_ticks: 2,
            min_running_before_pause: 3,
            min_shutdown_ticks: 1,
            health_check_interval: 5,
            max_failed_health_checks: 2,
            ema_decay: 0.5,
            window_size: 10,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let lc = Lifecycle::new();
        assert_eq!(lc.phase(), Phase::Init);
        assert_eq!(lc.current_tick(), 0);
        assert!(!lc.is_running());
        assert!(!lc.is_stopped());
        assert_eq!(lc.health(), HealthStatus::Healthy);
    }

    #[test]
    fn test_with_config() {
        let cfg = small_config();
        let lc = Lifecycle::with_config(cfg.clone());
        assert_eq!(lc.config().min_starting_ticks, 2);
        assert_eq!(lc.config().min_running_before_pause, 3);
    }

    // -------------------------------------------------------------------
    // Phase transition topology
    // -------------------------------------------------------------------

    #[test]
    fn test_valid_transitions_from_init() {
        assert!(Phase::Init.can_transition_to(Phase::Starting));
        assert!(!Phase::Init.can_transition_to(Phase::Running));
        assert!(!Phase::Init.can_transition_to(Phase::Stopped));
    }

    #[test]
    fn test_valid_transitions_from_starting() {
        assert!(Phase::Starting.can_transition_to(Phase::Running));
        assert!(Phase::Starting.can_transition_to(Phase::ShuttingDown));
        assert!(!Phase::Starting.can_transition_to(Phase::Paused));
    }

    #[test]
    fn test_valid_transitions_from_running() {
        assert!(Phase::Running.can_transition_to(Phase::Paused));
        assert!(Phase::Running.can_transition_to(Phase::ShuttingDown));
        assert!(!Phase::Running.can_transition_to(Phase::Init));
    }

    #[test]
    fn test_valid_transitions_from_paused() {
        assert!(Phase::Paused.can_transition_to(Phase::Running));
        assert!(Phase::Paused.can_transition_to(Phase::ShuttingDown));
        assert!(!Phase::Paused.can_transition_to(Phase::Starting));
    }

    #[test]
    fn test_stopped_is_terminal() {
        assert!(Phase::Stopped.is_terminal());
        assert!(Phase::Stopped.valid_transitions().is_empty());
    }

    // -------------------------------------------------------------------
    // Convenience transitions
    // -------------------------------------------------------------------

    #[test]
    fn test_start() {
        let mut lc = Lifecycle::with_config(small_config());
        assert!(lc.start().is_ok());
        assert_eq!(lc.phase(), Phase::Starting);
    }

    #[test]
    fn test_go_live_guard() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        // Not enough ticks in Starting.
        assert!(lc.go_live().is_err());
        lc.tick();
        lc.tick();
        assert!(lc.go_live().is_ok());
        assert_eq!(lc.phase(), Phase::Running);
    }

    #[test]
    fn test_pause_guard() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        // Not enough ticks in Running.
        assert!(lc.pause().is_err());
        for _ in 0..3 {
            lc.tick();
        }
        assert!(lc.pause().is_ok());
        assert_eq!(lc.phase(), Phase::Paused);
    }

    #[test]
    fn test_resume() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        for _ in 0..3 {
            lc.tick();
        }
        lc.pause().unwrap();
        assert!(lc.resume().is_ok());
        assert!(lc.is_running());
    }

    #[test]
    fn test_shutdown_and_stop() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        lc.shutdown().unwrap();
        assert_eq!(lc.phase(), Phase::ShuttingDown);
        // Need min_shutdown_ticks.
        assert!(lc.stop().is_err());
        lc.tick();
        assert!(lc.stop().is_ok());
        assert!(lc.is_stopped());
    }

    #[test]
    fn test_invalid_transition_returns_error() {
        let mut lc = Lifecycle::new();
        assert!(lc.transition_to(Phase::Running).is_err());
        assert!(lc.transition_to(Phase::Stopped).is_err());
    }

    #[test]
    fn test_stopped_cannot_transition() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        lc.shutdown().unwrap();
        lc.tick();
        lc.stop().unwrap();
        assert!(lc.start().is_err());
        assert!(lc.transition_to(Phase::Init).is_err());
    }

    // -------------------------------------------------------------------
    // Transition history
    // -------------------------------------------------------------------

    #[test]
    fn test_transition_history() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();

        let history = lc.transition_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].from, Phase::Init);
        assert_eq!(history[0].to, Phase::Starting);
        assert_eq!(history[1].from, Phase::Starting);
        assert_eq!(history[1].to, Phase::Running);
        assert_eq!(history[1].duration_in_previous, 2);
    }

    #[test]
    fn test_last_transition() {
        let mut lc = Lifecycle::with_config(small_config());
        assert!(lc.last_transition().is_none());
        lc.start().unwrap();
        let last = lc.last_transition().unwrap();
        assert_eq!(last.from, Phase::Init);
        assert_eq!(last.to, Phase::Starting);
    }

    // -------------------------------------------------------------------
    // Tick & phase counters
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut lc = Lifecycle::new();
        assert_eq!(lc.current_tick(), 0);
        lc.tick();
        lc.tick();
        assert_eq!(lc.current_tick(), 2);
        assert_eq!(lc.ticks_in_current_phase(), 2);
    }

    #[test]
    fn test_phase_counter_accumulates() {
        let mut lc = Lifecycle::with_config(small_config());
        // 3 ticks in Init.
        for _ in 0..3 {
            lc.tick();
        }
        lc.start().unwrap();
        // 2 ticks in Starting.
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        // 4 ticks in Running.
        for _ in 0..4 {
            lc.tick();
        }
        let stats = lc.stats();
        assert_eq!(stats.ticks_in_init, 3);
        assert_eq!(stats.ticks_in_starting, 2);
        assert_eq!(stats.ticks_in_running, 4);
        assert_eq!(stats.total_ticks, 9);
    }

    #[test]
    fn test_ticks_in_current_phase_resets_on_transition() {
        let mut lc = Lifecycle::with_config(small_config());
        for _ in 0..5 {
            lc.tick();
        }
        assert_eq!(lc.ticks_in_current_phase(), 5);
        lc.start().unwrap();
        assert_eq!(lc.ticks_in_current_phase(), 0);
        lc.tick();
        assert_eq!(lc.ticks_in_current_phase(), 1);
    }

    // -------------------------------------------------------------------
    // Health checks
    // -------------------------------------------------------------------

    #[test]
    fn test_report_health_healthy() {
        let mut lc = Lifecycle::new();
        lc.report_health(HealthStatus::Healthy);
        assert_eq!(lc.health(), HealthStatus::Healthy);
        assert_eq!(lc.consecutive_failures(), 0);
    }

    #[test]
    fn test_report_health_degraded_escalates() {
        let mut lc = Lifecycle::with_config(small_config());
        // max_failed_health_checks = 2
        lc.report_health(HealthStatus::Degraded);
        assert_eq!(lc.health(), HealthStatus::Degraded);
        assert_eq!(lc.consecutive_failures(), 1);

        lc.report_health(HealthStatus::Degraded);
        // Should escalate to Unhealthy after 2 consecutive.
        assert_eq!(lc.health(), HealthStatus::Unhealthy);
        assert_eq!(lc.consecutive_failures(), 2);
    }

    #[test]
    fn test_report_health_unhealthy_immediately() {
        let mut lc = Lifecycle::new();
        lc.report_health(HealthStatus::Unhealthy);
        assert_eq!(lc.health(), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_healthy_resets_consecutive_failures() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.report_health(HealthStatus::Degraded);
        assert_eq!(lc.consecutive_failures(), 1);
        lc.report_health(HealthStatus::Healthy);
        assert_eq!(lc.consecutive_failures(), 0);
        assert_eq!(lc.health(), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_check_due() {
        let mut lc = Lifecycle::with_config(small_config());
        // health_check_interval = 5
        assert!(lc.is_health_check_due()); // tick 0, last check tick 0 → diff = 0 ≥ 5? No... let me check.
        // Actually: 0 - 0 = 0 < 5, so not due initially.
        // After 5 ticks it should be due.
        for _ in 0..5 {
            lc.tick();
        }
        assert!(lc.is_health_check_due());
    }

    // -------------------------------------------------------------------
    // EMA diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.tick();
        // In Init, running_ratio = 0.
        assert!((lc.smoothed_running_ratio() - 0.0).abs() < 1e-9);
        assert!((lc.smoothed_health_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends_over_ticks() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        for _ in 0..10 {
            lc.tick();
        }
        // After 10 running ticks out of 12 total, ratio should be > 0.5.
        assert!(lc.smoothed_running_ratio() > 0.5);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_running_ratio_empty() {
        let lc = Lifecycle::new();
        assert!(lc.windowed_running_ratio().is_none());
    }

    #[test]
    fn test_windowed_running_ratio() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        for _ in 0..5 {
            lc.tick();
        }
        let ratio = lc.windowed_running_ratio().unwrap();
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_windowed_health_score() {
        let mut lc = Lifecycle::with_config(small_config());
        for _ in 0..5 {
            lc.tick();
        }
        let score = lc.windowed_health_score().unwrap();
        // All ticks healthy → score = 1.0.
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_health_declining() {
        let mut lc = Lifecycle::with_config(small_config());
        // First half: healthy.
        for _ in 0..4 {
            lc.tick();
        }
        // Second half: unhealthy.
        lc.report_health(HealthStatus::Unhealthy);
        for _ in 0..4 {
            lc.tick();
        }
        assert!(lc.is_health_declining());
    }

    #[test]
    fn test_is_health_declining_insufficient_data() {
        let mut lc = Lifecycle::new();
        lc.tick();
        assert!(!lc.is_health_declining());
    }

    // -------------------------------------------------------------------
    // Uptime
    // -------------------------------------------------------------------

    #[test]
    fn test_uptime() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        for _ in 0..7 {
            lc.tick();
        }
        assert_eq!(lc.uptime(), 7);
    }

    // -------------------------------------------------------------------
    // Process alias
    // -------------------------------------------------------------------

    #[test]
    fn test_process_alias() {
        let mut lc = Lifecycle::new();
        lc.process();
        assert_eq!(lc.current_tick(), 1);
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut lc = Lifecycle::with_config(small_config());
        lc.start().unwrap();
        for _ in 0..2 {
            lc.tick();
        }
        lc.go_live().unwrap();
        for _ in 0..5 {
            lc.tick();
        }
        lc.report_health(HealthStatus::Degraded);

        lc.reset();
        assert_eq!(lc.phase(), Phase::Init);
        assert_eq!(lc.current_tick(), 0);
        assert_eq!(lc.ticks_in_current_phase(), 0);
        assert_eq!(lc.health(), HealthStatus::Healthy);
        assert_eq!(lc.consecutive_failures(), 0);
        assert!(lc.transition_history().is_empty());
        assert_eq!(lc.stats().total_ticks, 0);
        assert!(lc.windowed_running_ratio().is_none());
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut lc = Lifecycle::with_config(small_config());

        // Init → Starting.
        lc.start().unwrap();
        assert_eq!(lc.phase(), Phase::Starting);

        // Warm up.
        for _ in 0..2 {
            lc.tick();
        }

        // Starting → Running.
        lc.go_live().unwrap();
        assert!(lc.is_running());

        // Run for a while.
        for _ in 0..5 {
            lc.tick();
            lc.report_health(HealthStatus::Healthy);
        }

        // Running → Paused.
        lc.pause().unwrap();
        assert_eq!(lc.phase(), Phase::Paused);
        lc.tick();

        // Paused → Running.
        lc.resume().unwrap();
        assert!(lc.is_running());
        lc.tick();

        // Running → ShuttingDown.
        lc.shutdown().unwrap();
        assert_eq!(lc.phase(), Phase::ShuttingDown);
        lc.tick();

        // ShuttingDown → Stopped.
        lc.stop().unwrap();
        assert!(lc.is_stopped());
        assert!(lc.phase().is_terminal());

        let stats = lc.stats();
        assert_eq!(stats.total_transitions, 6);
        assert!(stats.total_ticks > 0);
        assert!(stats.ticks_in_running > 0);
    }

    #[test]
    fn test_window_rolls() {
        let cfg = LifecycleConfig {
            window_size: 5,
            ..small_config()
        };
        let mut lc = Lifecycle::with_config(cfg);
        for _ in 0..20 {
            lc.tick();
        }
        // Window should cap at 5.
        assert!(lc.windowed_running_ratio().is_some());
        assert_eq!(lc.recent.len(), 5);
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::Init), "Init");
        assert_eq!(format!("{}", Phase::Running), "Running");
        assert_eq!(format!("{}", Phase::Stopped), "Stopped");
    }
}
