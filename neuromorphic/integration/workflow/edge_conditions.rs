//! Conditional transition evaluation
//!
//! Part of the Integration region — Workflow component.
//!
//! `EdgeConditions` evaluates typed predicates to determine whether a workflow
//! transition (edge) is permitted. Each condition carries a priority, an
//! optional cooldown, and tracks its evaluation history. The module provides
//! EMA-smoothed and windowed diagnostics over evaluation outcomes.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the edge-condition evaluator.
#[derive(Debug, Clone)]
pub struct EdgeConditionsConfig {
    /// Maximum number of registered conditions.
    pub max_conditions: usize,
    /// Maximum number of evaluation records retained per condition.
    pub max_history_per_condition: usize,
    /// Default cooldown ticks between consecutive *true* evaluations of any
    /// single condition (0 = no cooldown).
    pub default_cooldown_ticks: u64,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for EdgeConditionsConfig {
    fn default() -> Self {
        Self {
            max_conditions: 256,
            max_history_per_condition: 50,
            default_cooldown_ticks: 0,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Predicate kind
// ---------------------------------------------------------------------------

/// The kind of predicate a condition evaluates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PredicateKind {
    /// A threshold check: the supplied value must exceed the threshold.
    ThresholdAbove(OrderedF64),
    /// A threshold check: the supplied value must be below the threshold.
    ThresholdBelow(OrderedF64),
    /// The supplied value must fall within [low, high] inclusive.
    InRange(OrderedF64, OrderedF64),
    /// Boolean flag — the condition is simply true/false.
    Flag,
    /// Custom predicate identified by a string label.
    Custom(String),
}

/// Wrapper for `f64` that implements `Eq` and `Hash` by comparing the bit
/// pattern. Only used as a descriptor label — not for general-purpose
/// floating-point equality.
#[derive(Debug, Clone, Copy)]
pub struct OrderedF64(pub f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for OrderedF64 {}

impl std::hash::Hash for OrderedF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl std::fmt::Display for PredicateKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredicateKind::ThresholdAbove(t) => write!(f, "ThresholdAbove({})", t.0),
            PredicateKind::ThresholdBelow(t) => write!(f, "ThresholdBelow({})", t.0),
            PredicateKind::InRange(lo, hi) => write!(f, "InRange({}, {})", lo.0, hi.0),
            PredicateKind::Flag => write!(f, "Flag"),
            PredicateKind::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Condition priority
// ---------------------------------------------------------------------------

/// Priority tier for a condition. Higher-priority conditions are evaluated
/// first, and their results may short-circuit lower-priority evaluations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConditionPriority {
    /// Must be satisfied — a failure here blocks the transition.
    Critical = 0,
    /// Should be satisfied under normal circumstances.
    High = 1,
    /// Standard priority.
    Normal = 2,
    /// Advisory — failure does not block, only logged.
    Low = 3,
}

impl std::fmt::Display for ConditionPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConditionPriority::Critical => write!(f, "Critical"),
            ConditionPriority::High => write!(f, "High"),
            ConditionPriority::Normal => write!(f, "Normal"),
            ConditionPriority::Low => write!(f, "Low"),
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation record
// ---------------------------------------------------------------------------

/// Record of a single condition evaluation.
#[derive(Debug, Clone)]
pub struct EvalRecord {
    /// Tick at which the evaluation occurred.
    pub tick: u64,
    /// The value supplied for evaluation (if applicable).
    pub value: Option<f64>,
    /// Whether the condition evaluated to `true`.
    pub passed: bool,
}

// ---------------------------------------------------------------------------
// Condition descriptor
// ---------------------------------------------------------------------------

/// A registered condition.
#[derive(Debug, Clone)]
pub struct Condition {
    /// Unique name.
    pub name: String,
    /// Predicate kind.
    pub predicate: PredicateKind,
    /// Priority tier.
    pub priority: ConditionPriority,
    /// Cooldown ticks between consecutive *passed* evaluations.
    pub cooldown_ticks: u64,
    /// Tick at which the condition last passed.
    pub last_passed_tick: Option<u64>,
    /// Whether the condition is currently enabled.
    pub enabled: bool,
    /// Total evaluations.
    pub total_evals: u64,
    /// Total passes.
    pub total_passes: u64,
    /// Recent evaluation history.
    history: VecDeque<EvalRecord>,
    /// Maximum history length.
    max_history: usize,
}

impl Condition {
    fn new(
        name: impl Into<String>,
        predicate: PredicateKind,
        priority: ConditionPriority,
        cooldown_ticks: u64,
        max_history: usize,
    ) -> Self {
        Self {
            name: name.into(),
            predicate,
            priority,
            cooldown_ticks,
            last_passed_tick: None,
            enabled: true,
            total_evals: 0,
            total_passes: 0,
            history: VecDeque::with_capacity(max_history.min(128)),
            max_history,
        }
    }

    /// Pass rate over all evaluations.
    pub fn pass_rate(&self) -> f64 {
        if self.total_evals == 0 {
            return 0.0;
        }
        self.total_passes as f64 / self.total_evals as f64
    }

    /// Returns the recent evaluation history.
    pub fn history(&self) -> &VecDeque<EvalRecord> {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// Evaluation result (batch)
// ---------------------------------------------------------------------------

/// Result of evaluating all conditions for a transition.
#[derive(Debug, Clone)]
pub struct TransitionEvalResult {
    /// Whether the overall transition is allowed.
    pub allowed: bool,
    /// Per-condition results (name → passed).
    pub details: HashMap<String, bool>,
    /// Names of conditions that blocked the transition (critical/high that failed).
    pub blockers: Vec<String>,
    /// Names of conditions that were on cooldown and therefore skipped.
    pub on_cooldown: Vec<String>,
    /// Tick at which the evaluation was performed.
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    evals_performed: u64,
    evals_passed: u64,
    pass_rate: f64,
    conditions_on_cooldown: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the edge-condition evaluator.
#[derive(Debug, Clone)]
pub struct EdgeConditionsStats {
    /// Total individual condition evaluations.
    pub total_evals: u64,
    /// Total individual condition passes.
    pub total_passes: u64,
    /// Total transition-level evaluations.
    pub total_transition_evals: u64,
    /// Total transitions allowed.
    pub total_transitions_allowed: u64,
    /// Total transitions blocked.
    pub total_transitions_blocked: u64,
    /// EMA-smoothed per-evaluation pass rate.
    pub ema_pass_rate: f64,
    /// EMA-smoothed transition allow rate.
    pub ema_allow_rate: f64,
}

impl Default for EdgeConditionsStats {
    fn default() -> Self {
        Self {
            total_evals: 0,
            total_passes: 0,
            total_transition_evals: 0,
            total_transitions_allowed: 0,
            total_transitions_blocked: 0,
            ema_pass_rate: 0.0,
            ema_allow_rate: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// EdgeConditions
// ---------------------------------------------------------------------------

/// Conditional transition evaluator.
///
/// Manages a set of named, typed conditions. Conditions can be evaluated
/// individually or as a batch (for a transition). The evaluator enforces
/// cooldowns, tracks history, and provides EMA + windowed diagnostics.
pub struct EdgeConditions {
    config: EdgeConditionsConfig,
    /// Registered conditions keyed by name.
    conditions: HashMap<String, Condition>,
    /// Insertion order for deterministic iteration.
    condition_order: Vec<String>,
    /// Current tick counter.
    tick: u64,
    /// Evaluations performed in the current tick (for per-tick snapshot).
    evals_this_tick: u64,
    /// Passes in the current tick.
    passes_this_tick: u64,
    /// Whether EMA has been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: EdgeConditionsStats,
}

impl Default for EdgeConditions {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeConditions {
    // -- Construction -------------------------------------------------------

    /// Create a new evaluator with default configuration.
    pub fn new() -> Self {
        Self::with_config(EdgeConditionsConfig::default())
    }

    /// Create a new evaluator with the given configuration.
    pub fn with_config(config: EdgeConditionsConfig) -> Self {
        Self {
            conditions: HashMap::new(),
            condition_order: Vec::new(),
            tick: 0,
            evals_this_tick: 0,
            passes_this_tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: EdgeConditionsStats::default(),
            config,
        }
    }

    // -- Condition management -----------------------------------------------

    /// Register a new condition. Returns an error if a condition with the
    /// same name already exists or the maximum count is reached.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        predicate: PredicateKind,
        priority: ConditionPriority,
    ) -> Result<()> {
        let name = name.into();
        if self.conditions.contains_key(&name) {
            return Err(Error::Configuration(format!(
                "Condition '{}' already registered",
                name
            )));
        }
        if self.conditions.len() >= self.config.max_conditions {
            return Err(Error::Configuration(format!(
                "Maximum condition count ({}) reached",
                self.config.max_conditions
            )));
        }
        let cooldown = self.config.default_cooldown_ticks;
        let max_hist = self.config.max_history_per_condition;
        self.conditions
            .insert(name.clone(), Condition::new(&name, predicate, priority, cooldown, max_hist));
        self.condition_order.push(name);
        Ok(())
    }

    /// Register a condition with a custom cooldown.
    pub fn register_with_cooldown(
        &mut self,
        name: impl Into<String>,
        predicate: PredicateKind,
        priority: ConditionPriority,
        cooldown_ticks: u64,
    ) -> Result<()> {
        let name = name.into();
        if self.conditions.contains_key(&name) {
            return Err(Error::Configuration(format!(
                "Condition '{}' already registered",
                name
            )));
        }
        if self.conditions.len() >= self.config.max_conditions {
            return Err(Error::Configuration(format!(
                "Maximum condition count ({}) reached",
                self.config.max_conditions
            )));
        }
        let max_hist = self.config.max_history_per_condition;
        self.conditions
            .insert(name.clone(), Condition::new(&name, predicate, priority, cooldown_ticks, max_hist));
        self.condition_order.push(name);
        Ok(())
    }

    /// Enable a condition.
    pub fn enable(&mut self, name: &str) -> Result<()> {
        let cond = self
            .conditions
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown condition '{}'", name)))?;
        cond.enabled = true;
        Ok(())
    }

    /// Disable a condition (it will be skipped during evaluations).
    pub fn disable(&mut self, name: &str) -> Result<()> {
        let cond = self
            .conditions
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown condition '{}'", name)))?;
        cond.enabled = false;
        Ok(())
    }

    /// Look up a condition by name.
    pub fn condition(&self, name: &str) -> Option<&Condition> {
        self.conditions.get(name)
    }

    /// Number of registered conditions.
    pub fn condition_count(&self) -> usize {
        self.conditions.len()
    }

    /// Names of all registered conditions in insertion order.
    pub fn condition_names(&self) -> Vec<&str> {
        self.condition_order.iter().map(|s| s.as_str()).collect()
    }

    // -- Single-condition evaluation ----------------------------------------

    /// Check whether a condition is currently on cooldown.
    pub fn is_on_cooldown(&self, name: &str) -> bool {
        if let Some(cond) = self.conditions.get(name) {
            if cond.cooldown_ticks == 0 {
                return false;
            }
            if let Some(last) = cond.last_passed_tick {
                return self.tick.saturating_sub(last) < cond.cooldown_ticks;
            }
        }
        false
    }

    /// Evaluate a single condition with an optional numeric value.
    ///
    /// Returns `Ok(true)` if the predicate passes and the condition is not on
    /// cooldown. Returns `Ok(false)` otherwise. Returns `Err` if the condition
    /// is unknown or disabled.
    pub fn evaluate_one(&mut self, name: &str, value: Option<f64>) -> Result<bool> {
        let tick = self.tick;
        let cond = self
            .conditions
            .get_mut(name)
            .ok_or_else(|| Error::Configuration(format!("Unknown condition '{}'", name)))?;

        if !cond.enabled {
            return Err(Error::Configuration(format!(
                "Condition '{}' is disabled",
                name
            )));
        }

        // Cooldown check.
        if cond.cooldown_ticks > 0 {
            if let Some(last) = cond.last_passed_tick {
                if tick.saturating_sub(last) < cond.cooldown_ticks {
                    // On cooldown — record as a failed eval.
                    cond.total_evals += 1;
                    let record = EvalRecord {
                        tick,
                        value,
                        passed: false,
                    };
                    if cond.history.len() >= cond.max_history {
                        cond.history.pop_front();
                    }
                    cond.history.push_back(record);
                    self.stats.total_evals += 1;
                    self.evals_this_tick += 1;
                    return Ok(false);
                }
            }
        }

        // Predicate evaluation.
        let passed = match &cond.predicate {
            PredicateKind::ThresholdAbove(threshold) => {
                value.map_or(false, |v| v > threshold.0)
            }
            PredicateKind::ThresholdBelow(threshold) => {
                value.map_or(false, |v| v < threshold.0)
            }
            PredicateKind::InRange(lo, hi) => {
                value.map_or(false, |v| v >= lo.0 && v <= hi.0)
            }
            PredicateKind::Flag => {
                // For flags, a Some(non-zero) or absence of value means true/false.
                value.map_or(false, |v| v != 0.0)
            }
            PredicateKind::Custom(_) => {
                // Custom predicates are always evaluated as true if a non-zero
                // value is provided, allowing the caller to compute the
                // predicate externally.
                value.map_or(false, |v| v != 0.0)
            }
        };

        cond.total_evals += 1;
        if passed {
            cond.total_passes += 1;
            cond.last_passed_tick = Some(tick);
        }

        let record = EvalRecord {
            tick,
            value,
            passed,
        };
        if cond.history.len() >= cond.max_history {
            cond.history.pop_front();
        }
        cond.history.push_back(record);

        self.stats.total_evals += 1;
        if passed {
            self.stats.total_passes += 1;
            self.passes_this_tick += 1;
        }
        self.evals_this_tick += 1;

        Ok(passed)
    }

    // -- Batch / transition evaluation --------------------------------------

    /// Evaluate all enabled conditions for a transition.
    ///
    /// `values` maps condition names to the numeric value to supply. Conditions
    /// not present in `values` receive `None`.
    ///
    /// A transition is **allowed** when every `Critical` and `High`-priority
    /// condition passes (or is on cooldown and thus skipped). `Normal` and
    /// `Low` conditions are evaluated but do not block.
    pub fn evaluate_transition(
        &mut self,
        values: &HashMap<String, f64>,
    ) -> Result<TransitionEvalResult> {
        let tick = self.tick;
        let mut details: HashMap<String, bool> = HashMap::new();
        let mut blockers: Vec<String> = Vec::new();
        let mut on_cooldown: Vec<String> = Vec::new();

        // Collect names + priorities first to avoid borrow issues.
        let entries: Vec<(String, ConditionPriority, bool)> = self
            .condition_order
            .iter()
            .filter_map(|name| {
                let cond = self.conditions.get(name)?;
                if !cond.enabled {
                    return None;
                }
                Some((name.clone(), cond.priority, self.is_on_cooldown(name)))
            })
            .collect();

        for (name, priority, cooldown) in &entries {
            if *cooldown {
                on_cooldown.push(name.clone());
                details.insert(name.clone(), false);
                continue;
            }

            let val = values.get(name.as_str()).copied();
            let passed = self.evaluate_one(name, val)?;
            details.insert(name.clone(), passed);

            if !passed
                && (*priority == ConditionPriority::Critical
                    || *priority == ConditionPriority::High)
            {
                blockers.push(name.clone());
            }
        }

        let allowed = blockers.is_empty();

        self.stats.total_transition_evals += 1;
        if allowed {
            self.stats.total_transitions_allowed += 1;
        } else {
            self.stats.total_transitions_blocked += 1;
        }

        Ok(TransitionEvalResult {
            allowed,
            details,
            blockers,
            on_cooldown,
            tick,
        })
    }

    // -- Tick ---------------------------------------------------------------

    /// Advance the evaluator by one tick, updating EMA and windowed
    /// diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;

        let pass_rate = if self.evals_this_tick > 0 {
            self.passes_this_tick as f64 / self.evals_this_tick as f64
        } else {
            0.0
        };

        let cooldown_count = self
            .condition_order
            .iter()
            .filter(|n| self.is_on_cooldown(n))
            .count();

        let snapshot = TickSnapshot {
            evals_performed: self.evals_this_tick,
            evals_passed: self.passes_this_tick,
            pass_rate,
            conditions_on_cooldown: cooldown_count,
        };

        // EMA update.
        let allow_rate = if self.stats.total_transition_evals > 0 {
            self.stats.total_transitions_allowed as f64
                / self.stats.total_transition_evals as f64
        } else {
            1.0
        };

        if !self.ema_initialized && self.evals_this_tick > 0 {
            self.stats.ema_pass_rate = pass_rate;
            self.stats.ema_allow_rate = allow_rate;
            self.ema_initialized = true;
        } else if self.ema_initialized {
            let a = self.config.ema_decay;
            self.stats.ema_pass_rate = a * pass_rate + (1.0 - a) * self.stats.ema_pass_rate;
            self.stats.ema_allow_rate = a * allow_rate + (1.0 - a) * self.stats.ema_allow_rate;
        }

        // Window update.
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counters.
        self.evals_this_tick = 0;
        self.passes_this_tick = 0;
    }

    /// Current tick value.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Alias for `tick()`.
    pub fn process(&mut self) {
        self.tick();
    }

    // -- Diagnostics --------------------------------------------------------

    /// Returns a reference to cumulative statistics.
    pub fn stats(&self) -> &EdgeConditionsStats {
        &self.stats
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &EdgeConditionsConfig {
        &self.config
    }

    /// EMA-smoothed per-evaluation pass rate.
    pub fn smoothed_pass_rate(&self) -> f64 {
        self.stats.ema_pass_rate
    }

    /// EMA-smoothed transition allow rate.
    pub fn smoothed_allow_rate(&self) -> f64 {
        self.stats.ema_allow_rate
    }

    /// Windowed average pass rate.
    pub fn windowed_pass_rate(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.pass_rate).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average evaluations per tick.
    pub fn windowed_evals_per_tick(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.evals_performed as f64).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed average number of conditions on cooldown.
    pub fn windowed_cooldown_count(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.conditions_on_cooldown as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether the pass rate appears to be declining over the window.
    pub fn is_pass_rate_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(mid).map(|s| s.pass_rate).sum::<f64>() / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.pass_rate)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half < first_half - 0.05
    }

    // -- Reset --------------------------------------------------------------

    /// Reset all state, preserving configuration and registered conditions
    /// (which are also reset individually).
    pub fn reset(&mut self) {
        self.tick = 0;
        self.evals_this_tick = 0;
        self.passes_this_tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = EdgeConditionsStats::default();
        for cond in self.conditions.values_mut() {
            cond.total_evals = 0;
            cond.total_passes = 0;
            cond.last_passed_tick = None;
            cond.history.clear();
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> EdgeConditionsConfig {
        EdgeConditionsConfig {
            max_conditions: 8,
            max_history_per_condition: 5,
            default_cooldown_ticks: 0,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    // -- Construction -------------------------------------------------------

    #[test]
    fn test_new_default() {
        let ec = EdgeConditions::new();
        assert_eq!(ec.condition_count(), 0);
        assert_eq!(ec.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let ec = EdgeConditions::with_config(small_config());
        assert_eq!(ec.config().max_conditions, 8);
    }

    // -- Registration -------------------------------------------------------

    #[test]
    fn test_register_condition() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("vol_check", PredicateKind::ThresholdAbove(OrderedF64(0.5)), ConditionPriority::Critical)
            .unwrap();
        assert_eq!(ec.condition_count(), 1);
        assert!(ec.condition("vol_check").is_some());
    }

    #[test]
    fn test_register_duplicate_fails() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        assert!(ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).is_err());
    }

    #[test]
    fn test_register_at_capacity() {
        let mut ec = EdgeConditions::with_config(small_config());
        for i in 0..8 {
            ec.register(format!("c{}", i), PredicateKind::Flag, ConditionPriority::Normal)
                .unwrap();
        }
        assert!(ec.register("overflow", PredicateKind::Flag, ConditionPriority::Normal).is_err());
    }

    #[test]
    fn test_register_with_cooldown() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register_with_cooldown("cd", PredicateKind::Flag, ConditionPriority::Normal, 5)
            .unwrap();
        assert_eq!(ec.condition("cd").unwrap().cooldown_ticks, 5);
    }

    // -- Enable / Disable ---------------------------------------------------

    #[test]
    fn test_enable_disable() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        ec.disable("a").unwrap();
        assert!(!ec.condition("a").unwrap().enabled);
        ec.enable("a").unwrap();
        assert!(ec.condition("a").unwrap().enabled);
    }

    #[test]
    fn test_disable_unknown_fails() {
        let mut ec = EdgeConditions::with_config(small_config());
        assert!(ec.disable("nope").is_err());
    }

    #[test]
    fn test_condition_names_order() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("b", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        assert_eq!(ec.condition_names(), vec!["b", "a"]);
    }

    // -- Single evaluation: ThresholdAbove ----------------------------------

    #[test]
    fn test_threshold_above_passes() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("t", PredicateKind::ThresholdAbove(OrderedF64(10.0)), ConditionPriority::Normal)
            .unwrap();
        assert!(ec.evaluate_one("t", Some(15.0)).unwrap());
    }

    #[test]
    fn test_threshold_above_fails() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("t", PredicateKind::ThresholdAbove(OrderedF64(10.0)), ConditionPriority::Normal)
            .unwrap();
        assert!(!ec.evaluate_one("t", Some(5.0)).unwrap());
    }

    #[test]
    fn test_threshold_above_no_value() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("t", PredicateKind::ThresholdAbove(OrderedF64(10.0)), ConditionPriority::Normal)
            .unwrap();
        assert!(!ec.evaluate_one("t", None).unwrap());
    }

    // -- Single evaluation: ThresholdBelow ----------------------------------

    #[test]
    fn test_threshold_below_passes() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("t", PredicateKind::ThresholdBelow(OrderedF64(10.0)), ConditionPriority::Normal)
            .unwrap();
        assert!(ec.evaluate_one("t", Some(5.0)).unwrap());
    }

    #[test]
    fn test_threshold_below_fails() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("t", PredicateKind::ThresholdBelow(OrderedF64(10.0)), ConditionPriority::Normal)
            .unwrap();
        assert!(!ec.evaluate_one("t", Some(15.0)).unwrap());
    }

    // -- Single evaluation: InRange -----------------------------------------

    #[test]
    fn test_in_range_passes() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register(
            "r",
            PredicateKind::InRange(OrderedF64(1.0), OrderedF64(10.0)),
            ConditionPriority::Normal,
        )
        .unwrap();
        assert!(ec.evaluate_one("r", Some(5.0)).unwrap());
        assert!(ec.evaluate_one("r", Some(1.0)).unwrap()); // inclusive lower
        assert!(ec.evaluate_one("r", Some(10.0)).unwrap()); // inclusive upper
    }

    #[test]
    fn test_in_range_fails() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register(
            "r",
            PredicateKind::InRange(OrderedF64(1.0), OrderedF64(10.0)),
            ConditionPriority::Normal,
        )
        .unwrap();
        assert!(!ec.evaluate_one("r", Some(0.5)).unwrap());
        assert!(!ec.evaluate_one("r", Some(11.0)).unwrap());
    }

    // -- Single evaluation: Flag --------------------------------------------

    #[test]
    fn test_flag_passes() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("f", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        assert!(ec.evaluate_one("f", Some(1.0)).unwrap());
    }

    #[test]
    fn test_flag_fails_zero() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("f", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        assert!(!ec.evaluate_one("f", Some(0.0)).unwrap());
    }

    #[test]
    fn test_flag_fails_none() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("f", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        assert!(!ec.evaluate_one("f", None).unwrap());
    }

    // -- Single evaluation: Custom ------------------------------------------

    #[test]
    fn test_custom_passes() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register(
            "c",
            PredicateKind::Custom("my_check".into()),
            ConditionPriority::Normal,
        )
        .unwrap();
        assert!(ec.evaluate_one("c", Some(1.0)).unwrap());
    }

    // -- Unknown / disabled -------------------------------------------------

    #[test]
    fn test_evaluate_unknown() {
        let mut ec = EdgeConditions::with_config(small_config());
        assert!(ec.evaluate_one("nope", Some(1.0)).is_err());
    }

    #[test]
    fn test_evaluate_disabled() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        ec.disable("a").unwrap();
        assert!(ec.evaluate_one("a", Some(1.0)).is_err());
    }

    // -- Cooldown -----------------------------------------------------------

    #[test]
    fn test_cooldown() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register_with_cooldown("cd", PredicateKind::Flag, ConditionPriority::Normal, 3)
            .unwrap();

        // First evaluation passes.
        assert!(ec.evaluate_one("cd", Some(1.0)).unwrap());
        assert!(ec.is_on_cooldown("cd"));

        // Still on cooldown — returns false.
        ec.tick();
        assert!(!ec.evaluate_one("cd", Some(1.0)).unwrap());

        ec.tick();
        assert!(!ec.evaluate_one("cd", Some(1.0)).unwrap());

        // Cooldown expired after 3 ticks.
        ec.tick();
        assert!(!ec.is_on_cooldown("cd"));
        assert!(ec.evaluate_one("cd", Some(1.0)).unwrap());
    }

    #[test]
    fn test_is_on_cooldown_unknown() {
        let ec = EdgeConditions::with_config(small_config());
        assert!(!ec.is_on_cooldown("nope"));
    }

    #[test]
    fn test_is_on_cooldown_no_prior_pass() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register_with_cooldown("cd", PredicateKind::Flag, ConditionPriority::Normal, 5)
            .unwrap();
        assert!(!ec.is_on_cooldown("cd"));
    }

    // -- Evaluation history -------------------------------------------------

    #[test]
    fn test_history_tracking() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("h", PredicateKind::Flag, ConditionPriority::Normal).unwrap();

        ec.evaluate_one("h", Some(1.0)).unwrap(); // pass
        ec.evaluate_one("h", Some(0.0)).unwrap(); // fail
        ec.evaluate_one("h", Some(1.0)).unwrap(); // pass

        let cond = ec.condition("h").unwrap();
        assert_eq!(cond.history().len(), 3);
        assert_eq!(cond.total_evals, 3);
        assert_eq!(cond.total_passes, 2);
        assert!((cond.pass_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_history_bounded() {
        let mut ec = EdgeConditions::with_config(small_config()); // max_history_per_condition=5
        ec.register("h", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        for _ in 0..10 {
            ec.evaluate_one("h", Some(1.0)).unwrap();
        }
        assert_eq!(ec.condition("h").unwrap().history().len(), 5);
    }

    // -- Transition evaluation ----------------------------------------------

    #[test]
    fn test_transition_all_pass() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::ThresholdAbove(OrderedF64(5.0)), ConditionPriority::Critical)
            .unwrap();
        ec.register("b", PredicateKind::Flag, ConditionPriority::Normal).unwrap();

        let mut values = HashMap::new();
        values.insert("a".to_string(), 10.0);
        values.insert("b".to_string(), 1.0);

        let result = ec.evaluate_transition(&values).unwrap();
        assert!(result.allowed);
        assert!(result.blockers.is_empty());
    }

    #[test]
    fn test_transition_critical_blocks() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("crit", PredicateKind::ThresholdAbove(OrderedF64(100.0)), ConditionPriority::Critical)
            .unwrap();
        ec.register("norm", PredicateKind::Flag, ConditionPriority::Normal).unwrap();

        let mut values = HashMap::new();
        values.insert("crit".to_string(), 5.0); // fails
        values.insert("norm".to_string(), 1.0);

        let result = ec.evaluate_transition(&values).unwrap();
        assert!(!result.allowed);
        assert!(result.blockers.contains(&"crit".to_string()));
    }

    #[test]
    fn test_transition_high_blocks() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("hi", PredicateKind::Flag, ConditionPriority::High).unwrap();

        let values = HashMap::new(); // no value → Flag fails

        let result = ec.evaluate_transition(&values).unwrap();
        assert!(!result.allowed);
        assert!(result.blockers.contains(&"hi".to_string()));
    }

    #[test]
    fn test_transition_low_does_not_block() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("lo", PredicateKind::Flag, ConditionPriority::Low).unwrap();

        let values = HashMap::new(); // no value → Flag fails

        let result = ec.evaluate_transition(&values).unwrap();
        assert!(result.allowed); // Low does not block
        assert!(result.blockers.is_empty());
    }

    #[test]
    fn test_transition_normal_does_not_block() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("n", PredicateKind::Flag, ConditionPriority::Normal).unwrap();

        let values = HashMap::new();
        let result = ec.evaluate_transition(&values).unwrap();
        assert!(result.allowed);
    }

    #[test]
    fn test_transition_skips_disabled() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("crit", PredicateKind::ThresholdAbove(OrderedF64(100.0)), ConditionPriority::Critical)
            .unwrap();
        ec.disable("crit").unwrap();

        let values = HashMap::new();
        let result = ec.evaluate_transition(&values).unwrap();
        assert!(result.allowed);
        assert!(!result.details.contains_key("crit"));
    }

    #[test]
    fn test_transition_reports_cooldown() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register_with_cooldown("cd", PredicateKind::Flag, ConditionPriority::Normal, 5)
            .unwrap();

        // First pass triggers cooldown.
        let mut values = HashMap::new();
        values.insert("cd".to_string(), 1.0);
        ec.evaluate_transition(&values).unwrap();

        // Second evaluation — on cooldown.
        let result = ec.evaluate_transition(&values).unwrap();
        assert!(result.on_cooldown.contains(&"cd".to_string()));
    }

    #[test]
    fn test_transition_stats() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();

        let mut values = HashMap::new();
        values.insert("a".to_string(), 1.0);
        ec.evaluate_transition(&values).unwrap();

        values.insert("a".to_string(), 0.0);
        ec.evaluate_transition(&values).unwrap();

        assert_eq!(ec.stats().total_transition_evals, 2);
        assert_eq!(ec.stats().total_transitions_allowed, 2); // Normal doesn't block
    }

    // -- Tick & EMA ---------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.tick();
        ec.tick();
        assert_eq!(ec.current_tick(), 2);
    }

    #[test]
    fn test_process_alias() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.process();
        assert_eq!(ec.current_tick(), 1);
    }

    #[test]
    fn test_ema_initialises() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        ec.evaluate_one("a", Some(1.0)).unwrap();
        ec.tick();
        assert!((ec.smoothed_pass_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_blends() {
        let mut ec = EdgeConditions::with_config(small_config()); // ema_decay=0.5
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();

        // Tick 1: all pass → rate 1.0
        ec.evaluate_one("a", Some(1.0)).unwrap();
        ec.tick();

        // Tick 2: all fail → rate 0.0
        ec.evaluate_one("a", Some(0.0)).unwrap();
        ec.tick();

        // EMA = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert!((ec.smoothed_pass_rate() - 0.5).abs() < 1e-9);
    }

    // -- Windowed diagnostics -----------------------------------------------

    #[test]
    fn test_windowed_empty() {
        let ec = EdgeConditions::with_config(small_config());
        assert!(ec.windowed_pass_rate().is_none());
        assert!(ec.windowed_evals_per_tick().is_none());
        assert!(ec.windowed_cooldown_count().is_none());
    }

    #[test]
    fn test_windowed_pass_rate() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        ec.evaluate_one("a", Some(1.0)).unwrap();
        ec.tick();
        ec.evaluate_one("a", Some(0.0)).unwrap();
        ec.tick();
        let rate = ec.windowed_pass_rate().unwrap();
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_evals_per_tick() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        ec.evaluate_one("a", Some(1.0)).unwrap();
        ec.evaluate_one("a", Some(0.0)).unwrap();
        ec.tick(); // 2 evals this tick
        let avg = ec.windowed_evals_per_tick().unwrap();
        assert!((avg - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_pass_rate_declining() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();

        // First two ticks: all pass
        for _ in 0..2 {
            ec.evaluate_one("a", Some(1.0)).unwrap();
            ec.tick();
        }
        // Next two ticks: all fail
        for _ in 0..2 {
            ec.evaluate_one("a", Some(0.0)).unwrap();
            ec.tick();
        }
        assert!(ec.is_pass_rate_declining());
    }

    #[test]
    fn test_is_pass_rate_declining_insufficient() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.tick();
        assert!(!ec.is_pass_rate_declining());
    }

    #[test]
    fn test_window_rolls() {
        let mut ec = EdgeConditions::with_config(small_config()); // window_size=5
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        for _ in 0..20 {
            ec.evaluate_one("a", Some(1.0)).unwrap();
            ec.tick();
        }
        assert!(ec.recent.len() <= 5);
    }

    // -- Reset --------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register("a", PredicateKind::Flag, ConditionPriority::Normal).unwrap();
        ec.evaluate_one("a", Some(1.0)).unwrap();
        ec.tick();

        ec.reset();
        assert_eq!(ec.current_tick(), 0);
        assert_eq!(ec.stats().total_evals, 0);
        assert_eq!(ec.stats().total_passes, 0);
        assert_eq!(ec.condition("a").unwrap().total_evals, 0);
        assert!(ec.condition("a").unwrap().history().is_empty());
        // Condition is still registered.
        assert_eq!(ec.condition_count(), 1);
    }

    // -- Full lifecycle -----------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut ec = EdgeConditions::with_config(small_config());
        ec.register(
            "vol",
            PredicateKind::ThresholdAbove(OrderedF64(0.5)),
            ConditionPriority::Critical,
        )
        .unwrap();
        ec.register_with_cooldown(
            "rate_limit",
            PredicateKind::Flag,
            ConditionPriority::Normal,
            2,
        )
        .unwrap();
        ec.register(
            "spread",
            PredicateKind::InRange(OrderedF64(0.01), OrderedF64(0.10)),
            ConditionPriority::High,
        )
        .unwrap();

        // Round 1: all pass.
        let mut values = HashMap::new();
        values.insert("vol".to_string(), 1.0);
        values.insert("rate_limit".to_string(), 1.0);
        values.insert("spread".to_string(), 0.05);
        let result = ec.evaluate_transition(&values).unwrap();
        assert!(result.allowed);
        ec.tick();

        // Round 2: rate_limit on cooldown, vol still passes.
        let result = ec.evaluate_transition(&values).unwrap();
        assert!(result.allowed); // rate_limit is Normal, doesn't block
        assert!(result.on_cooldown.contains(&"rate_limit".to_string()));
        ec.tick();

        // Round 3: vol fails → blocked.
        values.insert("vol".to_string(), 0.2);
        let result = ec.evaluate_transition(&values).unwrap();
        assert!(!result.allowed);
        assert!(result.blockers.contains(&"vol".to_string()));
        ec.tick();

        assert_eq!(ec.stats().total_transition_evals, 3);
        assert!(ec.stats().total_transitions_blocked > 0);
        assert!(ec.smoothed_pass_rate() > 0.0);
    }

    // -- Display trait coverage ---------------------------------------------

    #[test]
    fn test_predicate_display() {
        assert_eq!(
            format!("{}", PredicateKind::ThresholdAbove(OrderedF64(5.0))),
            "ThresholdAbove(5)"
        );
        assert_eq!(format!("{}", PredicateKind::Flag), "Flag");
        assert_eq!(
            format!("{}", PredicateKind::Custom("x".into())),
            "Custom(x)"
        );
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(format!("{}", ConditionPriority::Critical), "Critical");
        assert_eq!(format!("{}", ConditionPriority::Low), "Low");
    }
}
