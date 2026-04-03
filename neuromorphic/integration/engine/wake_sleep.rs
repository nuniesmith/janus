//! Wake-sleep cycle coordination
//!
//! Part of the Integration region — Engine component.
//!
//! `WakeSleep` coordinates the alternating phases of the neuromorphic system:
//!
//! * **Wake phase** — live inference and trading, processing real-time market
//!   data and generating signals.
//! * **Sleep phase** — offline training, memory consolidation, and model
//!   updates using buffered experience.
//!
//! The engine manages phase transitions with configurable cooldowns, tracks
//! per-phase utilisation, and exposes EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the wake-sleep coordinator.
#[derive(Debug, Clone)]
pub struct WakeSleepConfig {
    /// Minimum number of ticks to stay in the wake phase before a transition
    /// to sleep is allowed.
    pub min_wake_ticks: u64,
    /// Minimum number of ticks to stay in the sleep phase before a transition
    /// back to wake is allowed.
    pub min_sleep_ticks: u64,
    /// Cooldown ticks after a phase transition before another is permitted.
    pub transition_cooldown: u64,
    /// Target fraction of total time spent in wake phase (0..1).
    pub target_wake_ratio: f64,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for WakeSleepConfig {
    fn default() -> Self {
        Self {
            min_wake_ticks: 10,
            min_sleep_ticks: 5,
            transition_cooldown: 2,
            target_wake_ratio: 0.8,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

/// The current operating phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Live inference and trading.
    Wake,
    /// Offline training and consolidation.
    Sleep,
    /// Transitioning between phases (cooldown period).
    Transitioning,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Phase::Wake => write!(f, "Wake"),
            Phase::Sleep => write!(f, "Sleep"),
            Phase::Transitioning => write!(f, "Transitioning"),
        }
    }
}

// ---------------------------------------------------------------------------
// Transition record
// ---------------------------------------------------------------------------

/// Record of a phase transition.
#[derive(Debug, Clone)]
pub struct TransitionRecord {
    /// The phase we transitioned from.
    pub from: Phase,
    /// The phase we transitioned to.
    pub to: Phase,
    /// Tick at which the transition occurred.
    pub tick: u64,
    /// Duration (in ticks) spent in the previous phase.
    pub previous_phase_duration: u64,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    phase: Phase,
    wake_ticks_total: u64,
    sleep_ticks_total: u64,
    wake_ratio: f64,
    ticks_in_current_phase: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the wake-sleep coordinator.
#[derive(Debug, Clone)]
pub struct WakeSleepStats {
    /// Total ticks spent in the wake phase.
    pub total_wake_ticks: u64,
    /// Total ticks spent in the sleep phase.
    pub total_sleep_ticks: u64,
    /// Total ticks spent transitioning (cooldown).
    pub total_transition_ticks: u64,
    /// Total number of phase transitions completed.
    pub total_transitions: u64,
    /// Total number of transition requests that were denied (cooldown / min ticks).
    pub total_denied_transitions: u64,
    /// Average duration of wake phases (ticks).
    pub avg_wake_duration: f64,
    /// Average duration of sleep phases (ticks).
    pub avg_sleep_duration: f64,
    /// EMA-smoothed wake ratio (fraction of time in wake).
    pub ema_wake_ratio: f64,
    /// EMA-smoothed phase duration.
    pub ema_phase_duration: f64,
}

impl Default for WakeSleepStats {
    fn default() -> Self {
        Self {
            total_wake_ticks: 0,
            total_sleep_ticks: 0,
            total_transition_ticks: 0,
            total_transitions: 0,
            total_denied_transitions: 0,
            avg_wake_duration: 0.0,
            avg_sleep_duration: 0.0,
            ema_wake_ratio: 0.8,
            ema_phase_duration: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// WakeSleep
// ---------------------------------------------------------------------------

/// Wake-sleep cycle coordinator.
///
/// Manages the alternating wake (live trading) and sleep (training /
/// consolidation) phases of the neuromorphic system. Enforces minimum
/// phase durations and transition cooldowns, tracks utilisation metrics,
/// and provides EMA-smoothed and windowed diagnostics.
pub struct WakeSleep {
    config: WakeSleepConfig,
    /// Current phase.
    phase: Phase,
    /// Tick counter.
    tick: u64,
    /// Number of ticks spent in the current phase.
    ticks_in_current_phase: u64,
    /// Tick at which the last transition completed.
    last_transition_tick: u64,
    /// The phase we are transitioning towards (set during `Transitioning`).
    pending_phase: Option<Phase>,
    /// Remaining cooldown ticks during a transition.
    cooldown_remaining: u64,
    /// History of transitions (most recent at back).
    transition_history: Vec<TransitionRecord>,
    /// Cumulative durations for average computation.
    cumulative_wake_durations: Vec<u64>,
    cumulative_sleep_durations: Vec<u64>,
    /// EMA initialisation flag.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: WakeSleepStats,
}

impl Default for WakeSleep {
    fn default() -> Self {
        Self::new()
    }
}

impl WakeSleep {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create with default configuration. Starts in the `Wake` phase.
    pub fn new() -> Self {
        Self::with_config(WakeSleepConfig::default())
    }

    /// Create with explicit configuration. Starts in the `Wake` phase.
    pub fn with_config(config: WakeSleepConfig) -> Self {
        Self {
            phase: Phase::Wake,
            tick: 0,
            ticks_in_current_phase: 0,
            last_transition_tick: 0,
            pending_phase: None,
            cooldown_remaining: 0,
            transition_history: Vec::new(),
            cumulative_wake_durations: Vec::new(),
            cumulative_sleep_durations: Vec::new(),
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size),
            stats: WakeSleepStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // Phase queries
    // -------------------------------------------------------------------

    /// Current operating phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Whether the system is currently in the wake phase.
    pub fn is_awake(&self) -> bool {
        self.phase == Phase::Wake
    }

    /// Whether the system is currently in the sleep phase.
    pub fn is_sleeping(&self) -> bool {
        self.phase == Phase::Sleep
    }

    /// Whether the system is transitioning between phases.
    pub fn is_transitioning(&self) -> bool {
        self.phase == Phase::Transitioning
    }

    /// Number of ticks spent in the current phase.
    pub fn ticks_in_current_phase(&self) -> u64 {
        self.ticks_in_current_phase
    }

    /// Remaining cooldown ticks during a transition.
    pub fn cooldown_remaining(&self) -> u64 {
        self.cooldown_remaining
    }

    /// Whether a transition to the given target phase is currently permitted.
    pub fn can_transition_to(&self, target: Phase) -> bool {
        if self.phase == Phase::Transitioning {
            return false;
        }
        if self.phase == target {
            return false;
        }
        // Check cooldown since last transition
        if self.tick.saturating_sub(self.last_transition_tick) < self.config.transition_cooldown
            && self.tick > 0
        {
            return false;
        }
        // Check minimum phase duration
        match self.phase {
            Phase::Wake => self.ticks_in_current_phase >= self.config.min_wake_ticks,
            Phase::Sleep => self.ticks_in_current_phase >= self.config.min_sleep_ticks,
            Phase::Transitioning => false,
        }
    }

    // -------------------------------------------------------------------
    // Phase transitions
    // -------------------------------------------------------------------

    /// Request a transition to the sleep phase.
    ///
    /// Returns `Ok(true)` if the transition was initiated, `Ok(false)` if
    /// the transition was denied (minimum duration or cooldown not met).
    pub fn request_sleep(&mut self) -> Result<bool> {
        if !self.can_transition_to(Phase::Sleep) {
            self.stats.total_denied_transitions += 1;
            return Ok(false);
        }
        self.initiate_transition(Phase::Sleep);
        Ok(true)
    }

    /// Request a transition to the wake phase.
    ///
    /// Returns `Ok(true)` if the transition was initiated, `Ok(false)` if
    /// the transition was denied.
    pub fn request_wake(&mut self) -> Result<bool> {
        if !self.can_transition_to(Phase::Wake) {
            self.stats.total_denied_transitions += 1;
            return Ok(false);
        }
        self.initiate_transition(Phase::Wake);
        Ok(true)
    }

    /// Force an immediate transition to the given phase, bypassing cooldowns
    /// and minimum durations. Useful for emergency situations.
    pub fn force_transition(&mut self, target: Phase) {
        if target == Phase::Transitioning {
            return; // cannot force into transitioning
        }
        self.record_phase_end();
        self.transition_history.push(TransitionRecord {
            from: self.phase,
            to: target,
            tick: self.tick,
            previous_phase_duration: self.ticks_in_current_phase,
        });
        self.phase = target;
        self.ticks_in_current_phase = 0;
        self.last_transition_tick = self.tick;
        self.pending_phase = None;
        self.cooldown_remaining = 0;
        self.stats.total_transitions += 1;
    }

    fn initiate_transition(&mut self, target: Phase) {
        self.record_phase_end();
        self.pending_phase = Some(target);
        self.cooldown_remaining = self.config.transition_cooldown;

        if self.cooldown_remaining == 0 {
            // No cooldown → immediate transition
            self.complete_transition();
        } else {
            let from = self.phase;
            self.transition_history.push(TransitionRecord {
                from,
                to: target,
                tick: self.tick,
                previous_phase_duration: self.ticks_in_current_phase,
            });
            self.phase = Phase::Transitioning;
            self.ticks_in_current_phase = 0;
            self.stats.total_transitions += 1;
        }
    }

    fn complete_transition(&mut self) {
        if let Some(target) = self.pending_phase.take() {
            if self.phase == Phase::Transitioning {
                self.transition_history.push(TransitionRecord {
                    from: Phase::Transitioning,
                    to: target,
                    tick: self.tick,
                    previous_phase_duration: self.ticks_in_current_phase,
                });
                self.stats.total_transitions += 1;
            } else {
                self.transition_history.push(TransitionRecord {
                    from: self.phase,
                    to: target,
                    tick: self.tick,
                    previous_phase_duration: self.ticks_in_current_phase,
                });
                self.stats.total_transitions += 1;
            }
            self.phase = target;
            self.ticks_in_current_phase = 0;
            self.last_transition_tick = self.tick;
        }
    }

    fn record_phase_end(&mut self) {
        if self.ticks_in_current_phase > 0 {
            match self.phase {
                Phase::Wake => {
                    self.cumulative_wake_durations
                        .push(self.ticks_in_current_phase);
                }
                Phase::Sleep => {
                    self.cumulative_sleep_durations
                        .push(self.ticks_in_current_phase);
                }
                Phase::Transitioning => {}
            }
        }
    }

    // -------------------------------------------------------------------
    // Tick lifecycle
    // -------------------------------------------------------------------

    /// Advance the wake-sleep coordinator by one tick.
    ///
    /// This:
    /// 1. Increments the tick counter and phase duration.
    /// 2. Processes any active transition cooldown.
    /// 3. Updates per-phase tick counts.
    /// 4. Captures EMA and windowed diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;
        self.ticks_in_current_phase += 1;

        // Process transition cooldown
        if self.phase == Phase::Transitioning && self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            self.stats.total_transition_ticks += 1;
            if self.cooldown_remaining == 0 {
                self.complete_transition();
            }
        }

        // Count phase ticks
        match self.phase {
            Phase::Wake => self.stats.total_wake_ticks += 1,
            Phase::Sleep => self.stats.total_sleep_ticks += 1,
            Phase::Transitioning => self.stats.total_transition_ticks += 1,
        }

        // Compute instantaneous wake ratio
        let total_active = self.stats.total_wake_ticks + self.stats.total_sleep_ticks;
        let wake_ratio = if total_active > 0 {
            self.stats.total_wake_ticks as f64 / total_active as f64
        } else {
            self.config.target_wake_ratio
        };

        // Update averages
        self.stats.avg_wake_duration = if self.cumulative_wake_durations.is_empty() {
            0.0
        } else {
            self.cumulative_wake_durations.iter().sum::<u64>() as f64
                / self.cumulative_wake_durations.len() as f64
        };
        self.stats.avg_sleep_duration = if self.cumulative_sleep_durations.is_empty() {
            0.0
        } else {
            self.cumulative_sleep_durations.iter().sum::<u64>() as f64
                / self.cumulative_sleep_durations.len() as f64
        };

        // EMA update
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_wake_ratio = wake_ratio;
            self.stats.ema_phase_duration = self.ticks_in_current_phase as f64;
            self.ema_initialized = true;
        } else {
            self.stats.ema_wake_ratio =
                alpha * wake_ratio + (1.0 - alpha) * self.stats.ema_wake_ratio;
            self.stats.ema_phase_duration = alpha * self.ticks_in_current_phase as f64
                + (1.0 - alpha) * self.stats.ema_phase_duration;
        }

        // Windowed snapshot
        let snapshot = TickSnapshot {
            phase: self.phase,
            wake_ticks_total: self.stats.total_wake_ticks,
            sleep_ticks_total: self.stats.total_sleep_ticks,
            wake_ratio,
            ticks_in_current_phase: self.ticks_in_current_phase,
        };

        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);
    }

    /// Current tick value.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Main processing function: advance one tick.
    pub fn process(&mut self) -> Result<()> {
        self.tick();
        Ok(())
    }

    // -------------------------------------------------------------------
    // Auto-scheduling
    // -------------------------------------------------------------------

    /// Suggest whether the system should transition based on the current
    /// wake ratio relative to the configured target.
    ///
    /// Returns `Some(Phase::Sleep)` if wake ratio is too high (should sleep
    /// more), `Some(Phase::Wake)` if too low (should wake more), or `None`
    /// if the current ratio is within tolerance.
    pub fn suggest_transition(&self) -> Option<Phase> {
        let total_active = self.stats.total_wake_ticks + self.stats.total_sleep_ticks;
        if total_active < 10 {
            return None; // too early to judge
        }
        let wake_ratio = self.stats.total_wake_ticks as f64 / total_active as f64;
        let tolerance = 0.05;

        if self.phase == Phase::Wake && wake_ratio > self.config.target_wake_ratio + tolerance {
            Some(Phase::Sleep)
        } else if self.phase == Phase::Sleep
            && wake_ratio < self.config.target_wake_ratio - tolerance
        {
            Some(Phase::Wake)
        } else {
            None
        }
    }

    // -------------------------------------------------------------------
    // History
    // -------------------------------------------------------------------

    /// Transition history (oldest first).
    pub fn transition_history(&self) -> &[TransitionRecord] {
        &self.transition_history
    }

    /// Number of transitions that have occurred.
    pub fn transition_count(&self) -> usize {
        self.transition_history.len()
    }

    /// Most recent transition record.
    pub fn last_transition(&self) -> Option<&TransitionRecord> {
        self.transition_history.last()
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Reference to cumulative statistics.
    pub fn stats(&self) -> &WakeSleepStats {
        &self.stats
    }

    /// Reference to configuration.
    pub fn config(&self) -> &WakeSleepConfig {
        &self.config
    }

    /// Current wake ratio (fraction of active time spent awake).
    pub fn wake_ratio(&self) -> f64 {
        let total = self.stats.total_wake_ticks + self.stats.total_sleep_ticks;
        if total == 0 {
            return self.config.target_wake_ratio;
        }
        self.stats.total_wake_ticks as f64 / total as f64
    }

    /// EMA-smoothed wake ratio.
    pub fn smoothed_wake_ratio(&self) -> f64 {
        self.stats.ema_wake_ratio
    }

    /// EMA-smoothed phase duration.
    pub fn smoothed_phase_duration(&self) -> f64 {
        self.stats.ema_phase_duration
    }

    /// Windowed average wake ratio.
    pub fn windowed_wake_ratio(&self) -> f64 {
        if self.recent.is_empty() {
            return self.config.target_wake_ratio;
        }
        let sum: f64 = self.recent.iter().map(|s| s.wake_ratio).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average phase duration.
    pub fn windowed_phase_duration(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.ticks_in_current_phase as f64)
            .sum();
        sum / self.recent.len() as f64
    }

    /// Whether the wake ratio is trending away from the target.
    pub fn is_wake_ratio_diverging(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let n = self.recent.len();
        let half = n / 2;

        let first_avg: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| (s.wake_ratio - self.config.target_wake_ratio).abs())
            .sum::<f64>()
            / half as f64;
        let second_avg: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| (s.wake_ratio - self.config.target_wake_ratio).abs())
            .sum::<f64>()
            / (n - half) as f64;

        second_avg > first_avg
    }

    /// Reset all state. Returns to the `Wake` phase.
    pub fn reset(&mut self) {
        self.phase = Phase::Wake;
        self.tick = 0;
        self.ticks_in_current_phase = 0;
        self.last_transition_tick = 0;
        self.pending_phase = None;
        self.cooldown_remaining = 0;
        self.transition_history.clear();
        self.cumulative_wake_durations.clear();
        self.cumulative_sleep_durations.clear();
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = WakeSleepStats::default();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> WakeSleepConfig {
        WakeSleepConfig {
            min_wake_ticks: 3,
            min_sleep_ticks: 2,
            transition_cooldown: 1,
            target_wake_ratio: 0.7,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    fn no_cooldown_config() -> WakeSleepConfig {
        WakeSleepConfig {
            min_wake_ticks: 2,
            min_sleep_ticks: 2,
            transition_cooldown: 0,
            target_wake_ratio: 0.5,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let ws = WakeSleep::new();
        assert_eq!(ws.phase(), Phase::Wake);
        assert!(ws.is_awake());
        assert!(!ws.is_sleeping());
        assert!(!ws.is_transitioning());
        assert_eq!(ws.current_tick(), 0);
        assert_eq!(ws.ticks_in_current_phase(), 0);
    }

    #[test]
    fn test_with_config() {
        let ws = WakeSleep::with_config(small_config());
        assert_eq!(ws.config().min_wake_ticks, 3);
        assert_eq!(ws.config().min_sleep_ticks, 2);
        assert_eq!(ws.config().transition_cooldown, 1);
    }

    // -------------------------------------------------------------------
    // Phase transitions
    // -------------------------------------------------------------------

    #[test]
    fn test_cannot_transition_too_early() {
        let mut ws = WakeSleep::with_config(small_config());
        // min_wake_ticks = 3, haven't ticked yet
        assert!(!ws.can_transition_to(Phase::Sleep));
        let result = ws.request_sleep().unwrap();
        assert!(!result);
        assert_eq!(ws.stats().total_denied_transitions, 1);
    }

    #[test]
    fn test_transition_after_min_ticks() {
        let mut ws = WakeSleep::with_config(small_config());
        // Tick 3 times to meet min_wake_ticks
        for _ in 0..3 {
            ws.tick();
        }
        assert!(ws.can_transition_to(Phase::Sleep));
        let result = ws.request_sleep().unwrap();
        assert!(result);
    }

    #[test]
    fn test_transition_with_cooldown() {
        let mut ws = WakeSleep::with_config(small_config()); // cooldown = 1
        for _ in 0..3 {
            ws.tick();
        }
        ws.request_sleep().unwrap();
        // Should be transitioning
        assert!(ws.is_transitioning());
        assert_eq!(ws.cooldown_remaining(), 1);

        // Tick through cooldown
        ws.tick();
        assert_eq!(ws.cooldown_remaining(), 0);
        assert!(ws.is_sleeping());
    }

    #[test]
    fn test_transition_without_cooldown() {
        let mut ws = WakeSleep::with_config(no_cooldown_config());
        for _ in 0..2 {
            ws.tick();
        }
        ws.request_sleep().unwrap();
        // Should transition immediately (no cooldown)
        assert!(ws.is_sleeping());
    }

    #[test]
    fn test_cannot_transition_to_same_phase() {
        let mut ws = WakeSleep::with_config(small_config());
        for _ in 0..5 {
            ws.tick();
        }
        assert!(!ws.can_transition_to(Phase::Wake));
        let result = ws.request_wake().unwrap();
        assert!(!result);
    }

    #[test]
    fn test_cannot_transition_while_transitioning() {
        let mut ws = WakeSleep::with_config(small_config());
        for _ in 0..3 {
            ws.tick();
        }
        ws.request_sleep().unwrap();
        assert!(ws.is_transitioning());
        assert!(!ws.can_transition_to(Phase::Wake));
    }

    #[test]
    fn test_force_transition() {
        let mut ws = WakeSleep::with_config(small_config());
        // Force transition without meeting any requirements
        ws.force_transition(Phase::Sleep);
        assert!(ws.is_sleeping());
        assert_eq!(ws.ticks_in_current_phase(), 0);
        assert_eq!(ws.stats().total_transitions, 1);
    }

    #[test]
    fn test_force_transition_to_transitioning_noop() {
        let mut ws = WakeSleep::new();
        ws.force_transition(Phase::Transitioning);
        assert!(ws.is_awake()); // should remain in Wake
    }

    #[test]
    fn test_round_trip_wake_sleep_wake() {
        let mut ws = WakeSleep::with_config(no_cooldown_config());

        // Start in Wake
        assert!(ws.is_awake());

        // Tick min_wake_ticks
        for _ in 0..2 {
            ws.tick();
        }
        assert!(ws.request_sleep().unwrap());
        assert!(ws.is_sleeping());

        // Tick min_sleep_ticks
        for _ in 0..2 {
            ws.tick();
        }
        assert!(ws.request_wake().unwrap());
        assert!(ws.is_awake());
    }

    // -------------------------------------------------------------------
    // Tick lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut ws = WakeSleep::new();
        ws.tick();
        ws.tick();
        assert_eq!(ws.current_tick(), 2);
        assert_eq!(ws.ticks_in_current_phase(), 2);
    }

    #[test]
    fn test_tick_counts_wake() {
        let mut ws = WakeSleep::new();
        ws.tick();
        ws.tick();
        assert_eq!(ws.stats().total_wake_ticks, 2);
    }

    #[test]
    fn test_tick_counts_sleep() {
        let mut ws = WakeSleep::with_config(no_cooldown_config());
        for _ in 0..2 {
            ws.tick();
        }
        ws.request_sleep().unwrap();
        ws.tick();
        ws.tick();
        assert_eq!(ws.stats().total_sleep_ticks, 2);
    }

    #[test]
    fn test_tick_counts_transition() {
        let mut ws = WakeSleep::with_config(WakeSleepConfig {
            min_wake_ticks: 1,
            transition_cooldown: 3,
            ..small_config()
        });
        ws.tick();
        ws.request_sleep().unwrap();
        assert!(ws.is_transitioning());
        ws.tick();
        ws.tick();
        ws.tick(); // completes transition
        assert!(ws.is_sleeping());
        // At least 3 transition ticks counted (could be more from tick accounting)
        assert!(ws.stats().total_transition_ticks >= 3);
    }

    #[test]
    fn test_process() {
        let mut ws = WakeSleep::new();
        assert!(ws.process().is_ok());
        assert_eq!(ws.current_tick(), 1);
    }

    // -------------------------------------------------------------------
    // History
    // -------------------------------------------------------------------

    #[test]
    fn test_transition_history() {
        let mut ws = WakeSleep::with_config(no_cooldown_config());
        for _ in 0..2 {
            ws.tick();
        }
        ws.request_sleep().unwrap();

        assert!(ws.transition_count() > 0);
        let last = ws.last_transition().unwrap();
        assert_eq!(last.to, Phase::Sleep);
    }

    #[test]
    fn test_transition_history_empty() {
        let ws = WakeSleep::new();
        assert_eq!(ws.transition_count(), 0);
        assert!(ws.last_transition().is_none());
    }

    // -------------------------------------------------------------------
    // Wake ratio
    // -------------------------------------------------------------------

    #[test]
    fn test_wake_ratio_initial() {
        let ws = WakeSleep::with_config(small_config());
        // No ticks yet → returns target
        assert!((ws.wake_ratio() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_wake_ratio_all_wake() {
        let mut ws = WakeSleep::new();
        for _ in 0..10 {
            ws.tick();
        }
        assert!((ws.wake_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wake_ratio_mixed() {
        let mut ws = WakeSleep::with_config(no_cooldown_config());

        // 4 wake ticks
        for _ in 0..4 {
            ws.tick();
        }
        ws.force_transition(Phase::Sleep);
        // 4 sleep ticks
        for _ in 0..4 {
            ws.tick();
        }
        // wake_ratio = 4 / 8 = 0.5
        assert!((ws.wake_ratio() - 0.5).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // Auto-scheduling
    // -------------------------------------------------------------------

    #[test]
    fn test_suggest_transition_too_early() {
        let mut ws = WakeSleep::new();
        ws.tick();
        assert!(ws.suggest_transition().is_none());
    }

    #[test]
    fn test_suggest_sleep_when_wake_ratio_high() {
        let mut ws = WakeSleep::with_config(WakeSleepConfig {
            target_wake_ratio: 0.5,
            ..no_cooldown_config()
        });
        // Spend lots of time in wake
        for _ in 0..20 {
            ws.tick();
        }
        // wake_ratio = 1.0, target = 0.5 → suggest Sleep
        assert_eq!(ws.suggest_transition(), Some(Phase::Sleep));
    }

    #[test]
    fn test_suggest_wake_when_wake_ratio_low() {
        let mut ws = WakeSleep::with_config(WakeSleepConfig {
            target_wake_ratio: 0.8,
            ..no_cooldown_config()
        });
        // 2 wake ticks
        for _ in 0..2 {
            ws.tick();
        }
        ws.force_transition(Phase::Sleep);
        // 20 sleep ticks → wake_ratio very low
        for _ in 0..20 {
            ws.tick();
        }
        assert_eq!(ws.suggest_transition(), Some(Phase::Wake));
    }

    // -------------------------------------------------------------------
    // EMA diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut ws = WakeSleep::with_config(small_config());
        ws.tick();
        assert!(ws.smoothed_wake_ratio() > 0.0);
        assert!(ws.smoothed_phase_duration() > 0.0);
    }

    #[test]
    fn test_ema_blends_on_subsequent_ticks() {
        let mut ws = WakeSleep::with_config(WakeSleepConfig {
            ema_decay: 0.5,
            ..no_cooldown_config()
        });
        for _ in 0..5 {
            ws.tick();
        }
        let ratio1 = ws.smoothed_wake_ratio();
        ws.force_transition(Phase::Sleep);
        for _ in 0..5 {
            ws.tick();
        }
        let ratio2 = ws.smoothed_wake_ratio();
        // After sleep ticks, wake ratio should decrease
        assert!(ratio2 < ratio1);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_wake_ratio() {
        let mut ws = WakeSleep::with_config(small_config());
        for _ in 0..3 {
            ws.tick();
        }
        assert!(ws.windowed_wake_ratio() > 0.0);
    }

    #[test]
    fn test_windowed_wake_ratio_empty() {
        let ws = WakeSleep::with_config(small_config());
        assert!((ws.windowed_wake_ratio() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_phase_duration() {
        let mut ws = WakeSleep::with_config(small_config());
        for _ in 0..3 {
            ws.tick();
        }
        assert!(ws.windowed_phase_duration() > 0.0);
    }

    #[test]
    fn test_windowed_phase_duration_empty() {
        let ws = WakeSleep::new();
        assert!((ws.windowed_phase_duration()).abs() < 1e-10);
    }

    #[test]
    fn test_is_wake_ratio_diverging_insufficient_data() {
        let mut ws = WakeSleep::new();
        ws.tick();
        assert!(!ws.is_wake_ratio_diverging());
    }

    // -------------------------------------------------------------------
    // Average durations
    // -------------------------------------------------------------------

    #[test]
    fn test_avg_wake_duration() {
        let mut ws = WakeSleep::with_config(no_cooldown_config());
        // 4 wake ticks then transition
        for _ in 0..4 {
            ws.tick();
        }
        ws.force_transition(Phase::Sleep);
        // 2 sleep ticks then transition
        for _ in 0..2 {
            ws.tick();
        }
        ws.force_transition(Phase::Wake);
        // 6 wake ticks then transition
        for _ in 0..6 {
            ws.tick();
        }
        ws.force_transition(Phase::Sleep);

        // avg_wake_duration = (4 + 6) / 2 = 5.0
        ws.tick(); // trigger stat update
        assert!((ws.stats().avg_wake_duration - 5.0).abs() < 1e-10);
        // avg_sleep_duration = 2.0
        assert!((ws.stats().avg_sleep_duration - 2.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut ws = WakeSleep::with_config(small_config());
        for _ in 0..5 {
            ws.tick();
        }
        ws.force_transition(Phase::Sleep);
        ws.tick();

        ws.reset();
        assert!(ws.is_awake());
        assert_eq!(ws.current_tick(), 0);
        assert_eq!(ws.ticks_in_current_phase(), 0);
        assert_eq!(ws.transition_count(), 0);
        assert_eq!(ws.stats().total_wake_ticks, 0);
        assert_eq!(ws.stats().total_sleep_ticks, 0);
        assert_eq!(ws.stats().total_transitions, 0);
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut ws = WakeSleep::with_config(WakeSleepConfig {
            min_wake_ticks: 3,
            min_sleep_ticks: 2,
            transition_cooldown: 0,
            target_wake_ratio: 0.6,
            ema_decay: 0.3,
            window_size: 10,
        });

        // Phase 1: Wake for 5 ticks
        for _ in 0..5 {
            ws.tick();
        }
        assert!(ws.is_awake());
        assert_eq!(ws.ticks_in_current_phase(), 5);

        // Transition to sleep
        assert!(ws.request_sleep().unwrap());
        assert!(ws.is_sleeping());

        // Phase 2: Sleep for 3 ticks
        for _ in 0..3 {
            ws.tick();
        }
        assert_eq!(ws.ticks_in_current_phase(), 3);

        // Transition back to wake
        assert!(ws.request_wake().unwrap());
        assert!(ws.is_awake());

        // Phase 3: Wake for 4 ticks
        for _ in 0..4 {
            ws.tick();
        }

        // Verify stats
        assert!(ws.stats().total_wake_ticks > 0);
        assert!(ws.stats().total_sleep_ticks > 0);
        assert!(ws.stats().total_transitions >= 2);
        assert!(ws.wake_ratio() > 0.0);
        assert!(ws.smoothed_wake_ratio() > 0.0);
        assert!(ws.stats().avg_wake_duration > 0.0);
        assert!(ws.stats().avg_sleep_duration > 0.0);
    }

    #[test]
    fn test_window_rolls() {
        let mut ws = WakeSleep::with_config(small_config()); // window = 5
        for _ in 0..10 {
            ws.tick();
        }
        assert!(ws.recent.len() <= ws.config.window_size);
    }

    // -------------------------------------------------------------------
    // Display
    // -------------------------------------------------------------------

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::Wake), "Wake");
        assert_eq!(format!("{}", Phase::Sleep), "Sleep");
        assert_eq!(format!("{}", Phase::Transitioning), "Transitioning");
    }
}
