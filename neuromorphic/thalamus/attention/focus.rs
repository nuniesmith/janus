//! Attentional focus control
//!
//! Part of the Thalamus region
//! Component: attention
//!
//! Manages a set of attentional focus targets, each with an activation level
//! that decays over time unless refreshed. This models the brain's ability
//! to selectively attend to a limited number of stimuli while ignoring others.
//!
//! ## Features
//!
//! - **Priority-weighted activation**: Each target has a priority that
//!   influences how quickly it gains and retains activation.
//! - **Exponential decay**: Activation levels decay exponentially when not
//!   refreshed, modelling the natural fading of attention.
//! - **Capacity management**: A configurable maximum number of active
//!   targets ensures the system doesn't spread attention too thin. When
//!   capacity is exceeded, the lowest-activation targets are evicted.
//! - **Inhibition of return**: Recently evicted targets receive a temporary
//!   penalty to prevent immediate re-activation (configurable).
//! - **Focus strength metric**: Aggregate metric indicating how concentrated
//!   vs. diffuse the current attentional state is.
//! - **EMA-smoothed focus strength**: Smoothed metric for downstream use.

use std::collections::{HashMap, VecDeque};

use crate::common::{Error, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the attentional focus controller.
#[derive(Debug, Clone)]
pub struct FocusConfig {
    /// Maximum number of concurrently active focus targets.
    pub max_targets: usize,
    /// Base decay rate per tick (fraction of activation lost per tick).
    /// Must be in (0, 1).
    pub decay_rate: f64,
    /// Minimum activation level below which a target is considered inactive
    /// and eligible for eviction.
    pub activation_threshold: f64,
    /// Duration (in ticks) for which inhibition-of-return applies to evicted
    /// targets. Set to 0 to disable.
    pub inhibition_duration: usize,
    /// Penalty multiplier applied to activation gains for targets under
    /// inhibition of return. Must be in [0, 1].
    pub inhibition_penalty: f64,
    /// EMA decay factor for smoothed focus strength (0, 1). Set to 0 to
    /// disable smoothing.
    pub ema_decay: f64,
    /// Maximum number of recent focus-strength values to retain for windowed
    /// statistics.
    pub window_size: usize,
    /// Whether to normalise activation levels so they sum to 1.0 after each
    /// tick (soft competition between targets).
    pub normalise: bool,
}

impl Default for FocusConfig {
    fn default() -> Self {
        Self {
            max_targets: 8,
            decay_rate: 0.1,
            activation_threshold: 0.01,
            inhibition_duration: 5,
            inhibition_penalty: 0.3,
            ema_decay: 0.8,
            window_size: 200,
            normalise: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// A request to focus on a particular target.
#[derive(Debug, Clone)]
pub struct FocusRequest {
    /// Identifier for the target (e.g. "BTC-USD", "ETH-USD", "volatility").
    pub target_id: String,
    /// Priority of this target in (0, ∞). Higher priority targets gain
    /// activation faster and decay slower.
    pub priority: f64,
    /// Activation boost to apply (added to current activation, scaled by
    /// priority).
    pub boost: f64,
}

/// Snapshot of the current attentional state.
#[derive(Debug, Clone)]
pub struct FocusSnapshot {
    /// Currently active targets sorted by activation (descending).
    pub targets: Vec<FocusTarget>,
    /// Focus strength: Herfindahl-Hirschman index of activation shares.
    /// 1.0 = perfectly focused on one target; 1/N = uniformly spread.
    pub strength: f64,
    /// EMA-smoothed focus strength.
    pub smoothed_strength: f64,
    /// Number of active targets.
    pub active_count: usize,
    /// Number of targets currently under inhibition of return.
    pub inhibited_count: usize,
}

/// A single focus target with its current state.
#[derive(Debug, Clone)]
pub struct FocusTarget {
    /// Target identifier.
    pub id: String,
    /// Current activation level in [0, 1] (or unbounded if not normalised).
    pub activation: f64,
    /// Priority weight.
    pub priority: f64,
    /// Number of ticks this target has been active.
    pub age: usize,
    /// Whether this target is under inhibition of return.
    pub inhibited: bool,
}

/// Aggregate statistics for the focus controller.
#[derive(Debug, Clone, Default)]
pub struct FocusStats {
    /// Total focus requests processed.
    pub total_requests: usize,
    /// Total ticks processed.
    pub total_ticks: usize,
    /// Total targets evicted due to capacity or threshold.
    pub total_evictions: usize,
    /// Total targets that entered inhibition of return.
    pub total_inhibitions: usize,
    /// Peak number of concurrent active targets observed.
    pub peak_active: usize,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TargetRecord {
    activation: f64,
    priority: f64,
    age: usize,
}

// ---------------------------------------------------------------------------
// Focus
// ---------------------------------------------------------------------------

/// Attentional focus controller.
///
/// Manages a set of focus targets with priority-weighted activation, decay,
/// capacity limits, and inhibition of return.
pub struct Focus {
    config: FocusConfig,
    /// Active targets.
    targets: HashMap<String, TargetRecord>,
    /// Inhibition-of-return timers: target_id → remaining ticks.
    inhibition: HashMap<String, usize>,
    /// EMA state for smoothed focus strength.
    ema_strength: f64,
    ema_initialized: bool,
    /// Windowed history of focus-strength values.
    history: VecDeque<f64>,
    /// Running statistics.
    stats: FocusStats,
}

impl Default for Focus {
    fn default() -> Self {
        Self::new()
    }
}

impl Focus {
    /// Create a new instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(FocusConfig::default())
    }

    /// Create a new instance with the given configuration.
    pub fn with_config(config: FocusConfig) -> Self {
        Self {
            targets: HashMap::new(),
            inhibition: HashMap::new(),
            ema_strength: 0.0,
            ema_initialized: false,
            history: VecDeque::with_capacity(config.window_size),
            stats: FocusStats::default(),
            config,
        }
    }

    /// Validate configuration parameters.
    pub fn process(&self) -> Result<()> {
        if self.config.max_targets == 0 {
            return Err(Error::InvalidInput("max_targets must be > 0".into()));
        }
        if self.config.decay_rate <= 0.0 || self.config.decay_rate >= 1.0 {
            return Err(Error::InvalidInput("decay_rate must be in (0, 1)".into()));
        }
        if self.config.activation_threshold < 0.0 {
            return Err(Error::InvalidInput(
                "activation_threshold must be >= 0".into(),
            ));
        }
        if self.config.inhibition_penalty < 0.0 || self.config.inhibition_penalty > 1.0 {
            return Err(Error::InvalidInput(
                "inhibition_penalty must be in [0, 1]".into(),
            ));
        }
        if self.config.ema_decay < 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in [0, 1)".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        Ok(())
    }

    // -- Focus requests ----------------------------------------------------

    /// Submit a focus request, boosting a target's activation.
    pub fn request_focus(&mut self, req: &FocusRequest) -> Result<()> {
        if req.priority <= 0.0 {
            return Err(Error::InvalidInput("priority must be > 0".into()));
        }
        if req.boost < 0.0 {
            return Err(Error::InvalidInput("boost must be >= 0".into()));
        }

        self.stats.total_requests += 1;

        let mut effective_boost = req.boost * req.priority;

        // Apply inhibition-of-return penalty.
        if self.inhibition.contains_key(&req.target_id) {
            effective_boost *= self.config.inhibition_penalty;
        }

        if let Some(record) = self.targets.get_mut(&req.target_id) {
            record.activation += effective_boost;
            record.priority = req.priority;
        } else {
            self.targets.insert(
                req.target_id.clone(),
                TargetRecord {
                    activation: effective_boost,
                    priority: req.priority,
                    age: 0,
                },
            );
        }

        // Enforce capacity: evict lowest-activation targets if over limit.
        self.enforce_capacity();

        Ok(())
    }

    // -- Tick (decay + maintenance) -----------------------------------------

    /// Advance one tick: decay activations, evict sub-threshold targets,
    /// decrement inhibition timers, and update statistics.
    pub fn tick(&mut self) -> FocusSnapshot {
        self.stats.total_ticks += 1;

        // Decay activations (priority modulates decay: higher priority → slower decay).
        for record in self.targets.values_mut() {
            let effective_decay = self.config.decay_rate / record.priority.max(0.1);
            record.activation *= 1.0 - effective_decay.min(0.99);
            record.age += 1;
        }

        // Normalise if configured.
        if self.config.normalise {
            let total: f64 = self.targets.values().map(|r| r.activation).sum();
            if total > 0.0 {
                for record in self.targets.values_mut() {
                    record.activation /= total;
                }
            }
        }

        // Evict sub-threshold targets.
        let threshold = self.config.activation_threshold;
        let evicted: Vec<String> = self
            .targets
            .iter()
            .filter(|(_, r)| r.activation < threshold)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &evicted {
            self.targets.remove(id);
            self.stats.total_evictions += 1;
            if self.config.inhibition_duration > 0 {
                self.inhibition
                    .insert(id.clone(), self.config.inhibition_duration);
                self.stats.total_inhibitions += 1;
            }
        }

        // Decrement inhibition timers.
        let expired: Vec<String> = self
            .inhibition
            .iter()
            .filter(|&(_, &t)| t <= 1)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            self.inhibition.remove(id);
        }
        for timer in self.inhibition.values_mut() {
            *timer = timer.saturating_sub(1);
        }

        // Update peak active count.
        if self.targets.len() > self.stats.peak_active {
            self.stats.peak_active = self.targets.len();
        }

        // Compute focus strength (HHI).
        let strength = self.compute_strength();

        // EMA smoothing.
        let smoothed = if self.config.ema_decay > 0.0 {
            if self.ema_initialized {
                let s = self.config.ema_decay * self.ema_strength
                    + (1.0 - self.config.ema_decay) * strength;
                self.ema_strength = s;
                s
            } else {
                self.ema_strength = strength;
                self.ema_initialized = true;
                strength
            }
        } else {
            strength
        };

        // History.
        self.history.push_back(strength);
        while self.history.len() > self.config.window_size {
            self.history.pop_front();
        }

        self.snapshot_with(strength, smoothed)
    }

    // -- Helpers -----------------------------------------------------------

    fn compute_strength(&self) -> f64 {
        let total: f64 = self.targets.values().map(|r| r.activation).sum();
        if total <= 0.0 || self.targets.is_empty() {
            return 0.0;
        }
        // HHI: sum of squared shares
        self.targets
            .values()
            .map(|r| {
                let share = r.activation / total;
                share * share
            })
            .sum()
    }

    fn enforce_capacity(&mut self) {
        while self.targets.len() > self.config.max_targets {
            // Find the target with the lowest activation.
            let min_id = self
                .targets
                .iter()
                .min_by(|(_, a), (_, b)| {
                    a.activation
                        .partial_cmp(&b.activation)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _)| id.clone());

            if let Some(id) = min_id {
                self.targets.remove(&id);
                self.stats.total_evictions += 1;
                if self.config.inhibition_duration > 0 {
                    self.inhibition.insert(id, self.config.inhibition_duration);
                    self.stats.total_inhibitions += 1;
                }
            } else {
                break;
            }
        }
    }

    fn snapshot_with(&self, strength: f64, smoothed: f64) -> FocusSnapshot {
        let mut targets: Vec<FocusTarget> = self
            .targets
            .iter()
            .map(|(id, r)| FocusTarget {
                id: id.clone(),
                activation: r.activation,
                priority: r.priority,
                age: r.age,
                inhibited: self.inhibition.contains_key(id),
            })
            .collect();
        targets.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        FocusSnapshot {
            active_count: targets.len(),
            inhibited_count: self.inhibition.len(),
            targets,
            strength,
            smoothed_strength: smoothed,
        }
    }

    // -- Accessors ---------------------------------------------------------

    /// Get the current snapshot without advancing a tick.
    pub fn snapshot(&self) -> FocusSnapshot {
        let strength = self.compute_strength();
        let smoothed = if self.ema_initialized {
            self.ema_strength
        } else {
            strength
        };
        self.snapshot_with(strength, smoothed)
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> &FocusStats {
        &self.stats
    }

    /// Number of currently active targets.
    pub fn active_count(&self) -> usize {
        self.targets.len()
    }

    /// Number of targets currently under inhibition of return.
    pub fn inhibited_count(&self) -> usize {
        self.inhibition.len()
    }

    /// Get the activation level for a specific target (0 if not active).
    pub fn activation(&self, target_id: &str) -> f64 {
        self.targets
            .get(target_id)
            .map(|r| r.activation)
            .unwrap_or(0.0)
    }

    /// Whether a target is currently under inhibition of return.
    pub fn is_inhibited(&self, target_id: &str) -> bool {
        self.inhibition.contains_key(target_id)
    }

    /// Windowed mean of recent focus-strength values.
    pub fn windowed_mean(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().sum();
        Some(sum / self.history.len() as f64)
    }

    /// Windowed standard deviation of recent focus-strength values.
    pub fn windowed_std(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let mean = self.windowed_mean().unwrap();
        let var: f64 = self.history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Remove a specific target from focus.
    pub fn remove_target(&mut self, target_id: &str) {
        self.targets.remove(target_id);
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.targets.clear();
        self.inhibition.clear();
        self.ema_strength = 0.0;
        self.ema_initialized = false;
        self.history.clear();
        self.stats = FocusStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn req(target: &str, priority: f64, boost: f64) -> FocusRequest {
        FocusRequest {
            target_id: target.to_string(),
            priority,
            boost,
        }
    }

    fn default_config() -> FocusConfig {
        FocusConfig {
            max_targets: 4,
            decay_rate: 0.2,
            activation_threshold: 0.01,
            inhibition_duration: 3,
            inhibition_penalty: 0.2,
            ema_decay: 0.5,
            window_size: 100,
            normalise: false,
        }
    }

    fn default_focus() -> Focus {
        Focus::with_config(default_config())
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = Focus::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_process_invalid_max_targets() {
        let f = Focus::with_config(FocusConfig {
            max_targets: 0,
            ..Default::default()
        });
        assert!(f.process().is_err());
    }

    #[test]
    fn test_process_invalid_decay_rate_zero() {
        let f = Focus::with_config(FocusConfig {
            decay_rate: 0.0,
            ..Default::default()
        });
        assert!(f.process().is_err());
    }

    #[test]
    fn test_process_invalid_decay_rate_one() {
        let f = Focus::with_config(FocusConfig {
            decay_rate: 1.0,
            ..Default::default()
        });
        assert!(f.process().is_err());
    }

    #[test]
    fn test_process_invalid_inhibition_penalty_negative() {
        let f = Focus::with_config(FocusConfig {
            inhibition_penalty: -0.1,
            ..Default::default()
        });
        assert!(f.process().is_err());
    }

    #[test]
    fn test_process_invalid_inhibition_penalty_above_one() {
        let f = Focus::with_config(FocusConfig {
            inhibition_penalty: 1.1,
            ..Default::default()
        });
        assert!(f.process().is_err());
    }

    #[test]
    fn test_process_invalid_ema_decay() {
        let f = Focus::with_config(FocusConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(f.process().is_err());
    }

    #[test]
    fn test_process_invalid_window_size() {
        let f = Focus::with_config(FocusConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(f.process().is_err());
    }

    #[test]
    fn test_process_invalid_activation_threshold() {
        let f = Focus::with_config(FocusConfig {
            activation_threshold: -1.0,
            ..Default::default()
        });
        assert!(f.process().is_err());
    }

    // -- Focus requests ----------------------------------------------------

    #[test]
    fn test_request_focus_creates_target() {
        let mut f = default_focus();
        f.request_focus(&req("BTC", 1.0, 1.0)).unwrap();
        assert_eq!(f.active_count(), 1);
        assert!(f.activation("BTC") > 0.0);
    }

    #[test]
    fn test_request_focus_boosts_existing() {
        let mut f = default_focus();
        f.request_focus(&req("BTC", 1.0, 0.5)).unwrap();
        let a1 = f.activation("BTC");
        f.request_focus(&req("BTC", 1.0, 0.5)).unwrap();
        let a2 = f.activation("BTC");
        assert!(
            a2 > a1,
            "repeated boost should increase activation: a1={}, a2={}",
            a1,
            a2
        );
    }

    #[test]
    fn test_request_focus_invalid_priority() {
        let mut f = default_focus();
        assert!(f.request_focus(&req("BTC", 0.0, 1.0)).is_err());
        assert!(f.request_focus(&req("BTC", -1.0, 1.0)).is_err());
    }

    #[test]
    fn test_request_focus_invalid_boost() {
        let mut f = default_focus();
        assert!(f.request_focus(&req("BTC", 1.0, -1.0)).is_err());
    }

    #[test]
    fn test_priority_scales_activation() {
        let mut f = default_focus();
        f.request_focus(&req("low", 1.0, 1.0)).unwrap();
        f.request_focus(&req("high", 3.0, 1.0)).unwrap();
        assert!(
            f.activation("high") > f.activation("low"),
            "higher priority should yield higher activation: high={}, low={}",
            f.activation("high"),
            f.activation("low")
        );
    }

    // -- Capacity management -----------------------------------------------

    #[test]
    fn test_capacity_eviction() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 2,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 2.0)).unwrap();
        f.request_focus(&req("C", 1.0, 3.0)).unwrap();
        // A should be evicted (lowest activation)
        assert_eq!(f.active_count(), 2);
        assert!(f.activation("A") < 0.01, "A should have been evicted");
        assert!(f.activation("B") > 0.0);
        assert!(f.activation("C") > 0.0);
    }

    #[test]
    fn test_capacity_eviction_increments_stats() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 1,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 2.0)).unwrap();
        assert!(f.stats().total_evictions >= 1);
    }

    // -- Decay -------------------------------------------------------------

    #[test]
    fn test_decay_reduces_activation() {
        let mut f = default_focus();
        f.request_focus(&req("BTC", 1.0, 1.0)).unwrap();
        let before = f.activation("BTC");
        f.tick();
        let after = f.activation("BTC");
        assert!(
            after < before,
            "activation should decay: before={}, after={}",
            before,
            after
        );
    }

    #[test]
    fn test_higher_priority_decays_slower() {
        let mut f = Focus::with_config(FocusConfig {
            decay_rate: 0.5,
            activation_threshold: 0.0001,
            max_targets: 10,
            ..default_config()
        });
        f.request_focus(&req("low", 1.0, 1.0)).unwrap();
        f.request_focus(&req("high", 5.0, 0.2)).unwrap(); // same effective boost = 1.0

        for _ in 0..5 {
            f.tick();
        }

        assert!(
            f.activation("high") > f.activation("low"),
            "higher priority should decay slower: high={}, low={}",
            f.activation("high"),
            f.activation("low")
        );
    }

    #[test]
    fn test_sub_threshold_eviction_on_tick() {
        let mut f = Focus::with_config(FocusConfig {
            decay_rate: 0.99, // very fast decay
            activation_threshold: 0.5,
            ..default_config()
        });
        f.request_focus(&req("BTC", 1.0, 1.0)).unwrap();
        f.tick();
        // After aggressive decay, activation should drop below threshold
        assert_eq!(
            f.active_count(),
            0,
            "target should be evicted when below threshold"
        );
    }

    // -- Inhibition of return ----------------------------------------------

    #[test]
    fn test_inhibition_after_eviction() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 1,
            inhibition_duration: 3,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 2.0)).unwrap(); // evicts A
        assert!(f.is_inhibited("A"), "A should be inhibited after eviction");
    }

    #[test]
    fn test_inhibition_reduces_boost() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 10,
            inhibition_duration: 10,
            inhibition_penalty: 0.5,
            ..default_config()
        });
        // Manually set inhibition on target "X"
        f.inhibition.insert("X".to_string(), 5);

        // Request focus on inhibited target
        f.request_focus(&req("X", 1.0, 2.0)).unwrap();
        // Without inhibition: activation = 2.0 * 1.0 = 2.0
        // With inhibition penalty 0.5: activation = 2.0 * 1.0 * 0.5 = 1.0
        assert!(
            (f.activation("X") - 1.0).abs() < 1e-10,
            "inhibition should reduce boost, got {}",
            f.activation("X")
        );
    }

    #[test]
    fn test_inhibition_expires() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 1,
            inhibition_duration: 2,
            activation_threshold: 0.0,
            decay_rate: 0.01,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 2.0)).unwrap(); // evicts A
        assert!(f.is_inhibited("A"));

        // Tick through the inhibition duration
        for _ in 0..3 {
            f.tick();
        }
        assert!(
            !f.is_inhibited("A"),
            "inhibition should have expired after {} ticks",
            3
        );
    }

    #[test]
    fn test_no_inhibition_when_duration_zero() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 1,
            inhibition_duration: 0,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 2.0)).unwrap(); // evicts A
        assert!(
            !f.is_inhibited("A"),
            "should not inhibit when duration is 0"
        );
    }

    // -- Focus strength ----------------------------------------------------

    #[test]
    fn test_focus_strength_single_target() {
        let mut f = default_focus();
        f.request_focus(&req("BTC", 1.0, 1.0)).unwrap();
        let snap = f.tick();
        assert!(
            (snap.strength - 1.0).abs() < 1e-10,
            "single target should have strength 1.0, got {}",
            snap.strength
        );
    }

    #[test]
    fn test_focus_strength_uniform_targets() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 4,
            decay_rate: 0.01,
            activation_threshold: 0.0,
            normalise: true,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 1.0)).unwrap();
        f.request_focus(&req("C", 1.0, 1.0)).unwrap();
        f.request_focus(&req("D", 1.0, 1.0)).unwrap();
        let snap = f.tick();
        // HHI of 4 equal shares = 4 * (0.25)^2 = 0.25
        assert!(
            (snap.strength - 0.25).abs() < 0.01,
            "4 uniform targets should have HHI ~0.25, got {}",
            snap.strength
        );
    }

    #[test]
    fn test_focus_strength_no_targets() {
        let mut f = default_focus();
        let snap = f.tick();
        assert!(
            snap.strength.abs() < 1e-10,
            "no targets should have strength 0"
        );
    }

    // -- Normalisation -----------------------------------------------------

    #[test]
    fn test_normalisation() {
        let mut f = Focus::with_config(FocusConfig {
            normalise: true,
            activation_threshold: 0.0,
            decay_rate: 0.01,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 3.0)).unwrap();
        f.request_focus(&req("B", 1.0, 7.0)).unwrap();
        f.tick();
        let total: f64 = f.targets.values().map(|r| r.activation).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "normalised activations should sum to 1.0, got {}",
            total
        );
    }

    // -- Snapshot -----------------------------------------------------------

    #[test]
    fn test_snapshot_sorted_by_activation() {
        let mut f = default_focus();
        f.request_focus(&req("low", 1.0, 0.1)).unwrap();
        f.request_focus(&req("high", 1.0, 10.0)).unwrap();
        f.request_focus(&req("mid", 1.0, 1.0)).unwrap();
        let snap = f.snapshot();
        assert_eq!(snap.targets[0].id, "high");
        assert_eq!(snap.targets[2].id, "low");
    }

    #[test]
    fn test_snapshot_active_count() {
        let mut f = default_focus();
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 1.0)).unwrap();
        let snap = f.snapshot();
        assert_eq!(snap.active_count, 2);
    }

    #[test]
    fn test_snapshot_inhibited_count() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 1,
            inhibition_duration: 5,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 2.0)).unwrap();
        let snap = f.snapshot();
        assert_eq!(snap.inhibited_count, 1);
    }

    // -- Target age --------------------------------------------------------

    #[test]
    fn test_target_age_increments() {
        let mut f = Focus::with_config(FocusConfig {
            decay_rate: 0.01,
            activation_threshold: 0.0,
            ..default_config()
        });
        f.request_focus(&req("BTC", 1.0, 1.0)).unwrap();
        f.tick();
        f.tick();
        f.tick();
        let snap = f.snapshot();
        assert_eq!(snap.targets[0].age, 3, "age should be 3 after 3 ticks");
    }

    // -- Remove target -----------------------------------------------------

    #[test]
    fn test_remove_target() {
        let mut f = default_focus();
        f.request_focus(&req("BTC", 1.0, 1.0)).unwrap();
        assert_eq!(f.active_count(), 1);
        f.remove_target("BTC");
        assert_eq!(f.active_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_target() {
        let mut f = default_focus();
        f.remove_target("NONEXISTENT"); // should not panic
        assert_eq!(f.active_count(), 0);
    }

    // -- EMA smoothing -----------------------------------------------------

    #[test]
    fn test_ema_smoothing() {
        let mut f = Focus::with_config(FocusConfig {
            ema_decay: 0.8,
            decay_rate: 0.01,
            activation_threshold: 0.0,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        let snap1 = f.tick();
        // First tick: EMA initialises to raw strength
        assert!((snap1.smoothed_strength - snap1.strength).abs() < 1e-10);

        // Add more targets to change strength
        f.request_focus(&req("B", 1.0, 1.0)).unwrap();
        f.request_focus(&req("C", 1.0, 1.0)).unwrap();
        let snap2 = f.tick();
        // Smoothed should lag
        if (snap2.strength - snap1.strength).abs() > 0.01 {
            assert!(
                (snap2.smoothed_strength - snap1.strength).abs()
                    < (snap2.strength - snap1.strength).abs(),
                "smoothed should be between previous and current"
            );
        }
    }

    #[test]
    fn test_ema_disabled_when_zero() {
        let mut f = Focus::with_config(FocusConfig {
            ema_decay: 0.0,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        let snap = f.tick();
        assert!(
            (snap.smoothed_strength - snap.strength).abs() < 1e-10,
            "with ema_decay=0, smoothed should equal raw"
        );
    }

    // -- Windowed statistics -----------------------------------------------

    #[test]
    fn test_windowed_mean() {
        let mut f = Focus::with_config(FocusConfig {
            decay_rate: 0.01,
            activation_threshold: 0.0,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        for _ in 0..5 {
            f.tick();
        }
        let mean = f.windowed_mean().unwrap();
        assert!(mean > 0.0);
    }

    #[test]
    fn test_windowed_mean_empty() {
        let f = default_focus();
        assert!(f.windowed_mean().is_none());
    }

    #[test]
    fn test_windowed_std_insufficient() {
        let mut f = default_focus();
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.tick();
        assert!(f.windowed_std().is_none());
    }

    // -- Reset -------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut f = default_focus();
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 1.0)).unwrap();
        f.tick();
        f.tick();

        assert!(f.active_count() > 0);
        assert!(f.stats().total_ticks > 0);

        f.reset();

        assert_eq!(f.active_count(), 0);
        assert_eq!(f.inhibited_count(), 0);
        assert_eq!(f.stats().total_ticks, 0);
        assert_eq!(f.stats().total_requests, 0);
        assert!(f.windowed_mean().is_none());
    }

    // -- Stats tracking ----------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut f = default_focus();
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 1.0)).unwrap();
        f.tick();
        f.tick();

        let s = f.stats();
        assert_eq!(s.total_requests, 2);
        assert_eq!(s.total_ticks, 2);
    }

    #[test]
    fn test_peak_active() {
        let mut f = Focus::with_config(FocusConfig {
            max_targets: 10,
            decay_rate: 0.01,
            activation_threshold: 0.0,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        f.request_focus(&req("B", 1.0, 1.0)).unwrap();
        f.request_focus(&req("C", 1.0, 1.0)).unwrap();
        f.tick();
        f.remove_target("C");
        f.tick();
        assert_eq!(f.stats().peak_active, 3);
    }

    // -- Window eviction ---------------------------------------------------

    #[test]
    fn test_window_eviction() {
        let mut f = Focus::with_config(FocusConfig {
            window_size: 3,
            decay_rate: 0.01,
            activation_threshold: 0.0,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        for _ in 0..10 {
            f.tick();
        }
        assert_eq!(f.history.len(), 3);
    }

    // -- Multiple ticks decay -----------------------------------------------

    #[test]
    fn test_multiple_ticks_full_decay() {
        let mut f = Focus::with_config(FocusConfig {
            decay_rate: 0.5,
            activation_threshold: 0.001,
            ..default_config()
        });
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        for _ in 0..50 {
            f.tick();
        }
        assert_eq!(
            f.active_count(),
            0,
            "target should eventually decay below threshold"
        );
    }

    // -- Zero boost --------------------------------------------------------

    #[test]
    fn test_zero_boost() {
        let mut f = default_focus();
        f.request_focus(&req("A", 1.0, 0.0)).unwrap();
        assert!(
            f.activation("A").abs() < 1e-10,
            "zero boost should create target with zero activation"
        );
    }

    // -- Activation accessor for missing target ----------------------------

    #[test]
    fn test_activation_missing_target() {
        let f = default_focus();
        assert!(f.activation("NONEXISTENT").abs() < 1e-10);
    }

    // -- Inhibited accessor for non-inhibited target -----------------------

    #[test]
    fn test_is_inhibited_false_for_active() {
        let mut f = default_focus();
        f.request_focus(&req("A", 1.0, 1.0)).unwrap();
        assert!(!f.is_inhibited("A"));
    }
}
