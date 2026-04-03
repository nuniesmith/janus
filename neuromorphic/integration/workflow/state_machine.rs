//! State machine orchestration
//!
//! Part of the Integration region — Workflow component.
//!
//! `StateMachine` provides a generic, configurable finite state machine for
//! orchestrating workflow execution. States are registered by name, transitions
//! are guarded by optional predicates, and entry/exit action hooks can be
//! attached. The module tracks transition history, time spent per state, and
//! exposes EMA-smoothed and windowed diagnostics.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the state machine.
#[derive(Debug, Clone)]
pub struct StateMachineConfig {
    /// Maximum number of states that can be registered.
    pub max_states: usize,
    /// Maximum number of transitions to retain in history.
    pub max_history: usize,
    /// EMA decay factor (0 < α < 1). Higher → faster adaptation.
    pub ema_decay: f64,
    /// Window size for rolling diagnostics.
    pub window_size: usize,
}

impl Default for StateMachineConfig {
    fn default() -> Self {
        Self {
            max_states: 64,
            max_history: 200,
            ema_decay: 0.1,
            window_size: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Guard result
// ---------------------------------------------------------------------------

/// Result of evaluating a transition guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardResult {
    /// Transition is allowed.
    Allow,
    /// Transition is denied.
    Deny,
}

// ---------------------------------------------------------------------------
// Transition definition
// ---------------------------------------------------------------------------

/// Definition of a transition between two states.
#[derive(Debug, Clone)]
pub struct TransitionDef {
    /// Source state name.
    pub from: String,
    /// Target state name.
    pub to: String,
    /// Optional human-readable label for this transition.
    pub label: Option<String>,
    /// Whether a guard predicate is attached.
    pub has_guard: bool,
    /// Number of times this transition has fired.
    pub fire_count: u64,
}

// ---------------------------------------------------------------------------
// State definition
// ---------------------------------------------------------------------------

/// Definition of a registered state.
#[derive(Debug, Clone)]
pub struct StateDef {
    /// Unique state name.
    pub name: String,
    /// Whether this is an accepting / terminal state.
    pub is_terminal: bool,
    /// Names of states reachable from this state.
    pub successors: Vec<String>,
    /// Cumulative ticks spent in this state.
    pub total_ticks: u64,
    /// Number of times this state has been entered.
    pub entry_count: u64,
}

// ---------------------------------------------------------------------------
// Transition record (history)
// ---------------------------------------------------------------------------

/// Record of a transition that was executed.
#[derive(Debug, Clone)]
pub struct TransitionRecord {
    /// Source state name.
    pub from: String,
    /// Target state name.
    pub to: String,
    /// Optional transition label.
    pub label: Option<String>,
    /// Tick at which the transition occurred.
    pub tick: u64,
    /// Number of ticks spent in the source state before leaving.
    pub ticks_in_source: u64,
}

// ---------------------------------------------------------------------------
// Tick snapshot (windowed diagnostics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickSnapshot {
    state_name: String,
    ticks_in_current: u64,
    transitions_total: u64,
    states_visited_total: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for the state machine.
#[derive(Debug, Clone)]
pub struct StateMachineStats {
    /// Total ticks elapsed.
    pub total_ticks: u64,
    /// Total transitions fired.
    pub total_transitions: u64,
    /// Total guard evaluations performed.
    pub total_guard_evals: u64,
    /// Total guard denials.
    pub total_guard_denials: u64,
    /// Distinct states visited at least once.
    pub states_visited: u64,
    /// EMA-smoothed ticks-per-state (average time in a state before leaving).
    pub ema_ticks_per_state: f64,
    /// EMA-smoothed transition rate (transitions per tick).
    pub ema_transition_rate: f64,
}

impl Default for StateMachineStats {
    fn default() -> Self {
        Self {
            total_ticks: 0,
            total_transitions: 0,
            total_guard_evals: 0,
            total_guard_denials: 0,
            states_visited: 0,
            ema_ticks_per_state: 0.0,
            ema_transition_rate: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal transition storage (with guards)
// ---------------------------------------------------------------------------

/// Internal representation of a transition with an optional guard closure.
struct TransitionEntry {
    def: TransitionDef,
    /// Optional guard function. Returns `GuardResult::Allow` if the transition
    /// is permitted.
    guard: Option<Box<dyn Fn() -> GuardResult + Send + Sync>>,
}

impl std::fmt::Debug for TransitionEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransitionEntry")
            .field("def", &self.def)
            .field("has_guard", &self.def.has_guard)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StateMachine
// ---------------------------------------------------------------------------

/// Workflow state machine.
///
/// Manages named states, guarded transitions, transition history, per-state
/// time tracking, and EMA + windowed diagnostics.
pub struct StateMachine {
    config: StateMachineConfig,
    /// Registered states keyed by name.
    states: HashMap<String, StateDef>,
    /// Insertion-ordered state names.
    state_order: Vec<String>,
    /// Transitions keyed by `(from, to)` tuple serialised as `"from->to"`.
    transitions: HashMap<String, TransitionEntry>,
    /// Current state name (`None` until the machine is started).
    current_state: Option<String>,
    /// Ticks elapsed in the current state.
    ticks_in_current: u64,
    /// Global tick counter.
    tick: u64,
    /// Transition history.
    history: VecDeque<TransitionRecord>,
    /// Transitions fired in the current tick (for rate EMA).
    transitions_this_tick: u64,
    /// Whether EMA values have been initialised.
    ema_initialized: bool,
    /// Rolling window of tick snapshots.
    recent: VecDeque<TickSnapshot>,
    /// Cumulative statistics.
    stats: StateMachineStats,
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl StateMachine {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new state machine with default configuration.
    pub fn new() -> Self {
        Self::with_config(StateMachineConfig::default())
    }

    /// Create a new state machine with the given configuration.
    pub fn with_config(config: StateMachineConfig) -> Self {
        Self {
            states: HashMap::new(),
            state_order: Vec::new(),
            transitions: HashMap::new(),
            current_state: None,
            ticks_in_current: 0,
            tick: 0,
            history: VecDeque::with_capacity(config.max_history.min(256)),
            transitions_this_tick: 0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size.min(256)),
            stats: StateMachineStats::default(),
            config,
        }
    }

    // -------------------------------------------------------------------
    // State registration
    // -------------------------------------------------------------------

    /// Register a new state. Returns `Err` if the state already exists or
    /// the maximum number of states has been reached.
    pub fn add_state(&mut self, name: impl Into<String>, is_terminal: bool) -> Result<()> {
        let name = name.into();
        if self.states.contains_key(&name) {
            return Err(Error::Configuration(format!(
                "State '{}' is already registered",
                name
            )));
        }
        if self.states.len() >= self.config.max_states {
            return Err(Error::Configuration(format!(
                "Maximum state count ({}) reached",
                self.config.max_states
            )));
        }
        self.states.insert(
            name.clone(),
            StateDef {
                name: name.clone(),
                is_terminal,
                successors: Vec::new(),
                total_ticks: 0,
                entry_count: 0,
            },
        );
        self.state_order.push(name);
        Ok(())
    }

    /// Returns `true` if a state with the given name is registered.
    pub fn has_state(&self, name: &str) -> bool {
        self.states.contains_key(name)
    }

    /// Returns the definition of a state by name.
    pub fn state(&self, name: &str) -> Option<&StateDef> {
        self.states.get(name)
    }

    /// Returns the number of registered states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Returns the names of all registered states in insertion order.
    pub fn state_names(&self) -> Vec<&str> {
        self.state_order.iter().map(|s| s.as_str()).collect()
    }

    // -------------------------------------------------------------------
    // Transition registration
    // -------------------------------------------------------------------

    fn transition_key(from: &str, to: &str) -> String {
        format!("{}->{}", from, to)
    }

    /// Register a transition between two states with no guard.
    pub fn add_transition(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        label: Option<String>,
    ) -> Result<()> {
        self.add_transition_inner(from.into(), to.into(), label, None)
    }

    /// Register a transition between two states with a guard closure.
    pub fn add_guarded_transition<F>(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        label: Option<String>,
        guard: F,
    ) -> Result<()>
    where
        F: Fn() -> GuardResult + Send + Sync + 'static,
    {
        self.add_transition_inner(from.into(), to.into(), label, Some(Box::new(guard)))
    }

    fn add_transition_inner(
        &mut self,
        from: String,
        to: String,
        label: Option<String>,
        guard: Option<Box<dyn Fn() -> GuardResult + Send + Sync>>,
    ) -> Result<()> {
        if !self.states.contains_key(&from) {
            return Err(Error::Configuration(format!(
                "Source state '{}' is not registered",
                from
            )));
        }
        if !self.states.contains_key(&to) {
            return Err(Error::Configuration(format!(
                "Target state '{}' is not registered",
                to
            )));
        }

        let key = Self::transition_key(&from, &to);
        if self.transitions.contains_key(&key) {
            return Err(Error::Configuration(format!(
                "Transition '{}' → '{}' already exists",
                from, to
            )));
        }

        let has_guard = guard.is_some();
        self.transitions.insert(
            key,
            TransitionEntry {
                def: TransitionDef {
                    from: from.clone(),
                    to: to.clone(),
                    label,
                    has_guard,
                    fire_count: 0,
                },
                guard,
            },
        );

        // Update successors list.
        if let Some(state) = self.states.get_mut(&from) {
            if !state.successors.contains(&to) {
                state.successors.push(to);
            }
        }

        Ok(())
    }

    /// Returns the number of registered transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Returns the transition definition for a `(from, to)` pair.
    pub fn transition_def(&self, from: &str, to: &str) -> Option<&TransitionDef> {
        let key = Self::transition_key(from, to);
        self.transitions.get(&key).map(|e| &e.def)
    }

    // -------------------------------------------------------------------
    // Machine lifecycle
    // -------------------------------------------------------------------

    /// Set the initial state and start the machine.
    ///
    /// Returns `Err` if the state is not registered or the machine is already
    /// started.
    pub fn start(&mut self, initial_state: &str) -> Result<()> {
        if self.current_state.is_some() {
            return Err(Error::Configuration(
                "State machine is already started".into(),
            ));
        }
        if !self.states.contains_key(initial_state) {
            return Err(Error::Configuration(format!(
                "Initial state '{}' is not registered",
                initial_state
            )));
        }
        self.current_state = Some(initial_state.to_string());
        self.ticks_in_current = 0;
        if let Some(state) = self.states.get_mut(initial_state) {
            state.entry_count += 1;
        }
        self.stats.states_visited += 1;
        Ok(())
    }

    /// Returns the name of the current state, if the machine has been started.
    pub fn current_state(&self) -> Option<&str> {
        self.current_state.as_deref()
    }

    /// Returns `true` if the machine is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        match &self.current_state {
            Some(name) => self
                .states
                .get(name)
                .map(|s| s.is_terminal)
                .unwrap_or(false),
            None => false,
        }
    }

    /// Returns `true` if the machine has been started.
    pub fn is_started(&self) -> bool {
        self.current_state.is_some()
    }

    /// Returns the ticks spent in the current state.
    pub fn ticks_in_current_state(&self) -> u64 {
        self.ticks_in_current
    }

    // -------------------------------------------------------------------
    // Transitions
    // -------------------------------------------------------------------

    /// Returns the list of valid successor state names from the current state.
    pub fn available_transitions(&self) -> Vec<&str> {
        match &self.current_state {
            Some(name) => self
                .states
                .get(name)
                .map(|s| s.successors.iter().map(|n| n.as_str()).collect())
                .unwrap_or_default(),
            None => Vec::new(),
        }
    }

    /// Check whether a transition to `target` is currently allowed, taking
    /// guards into account.
    pub fn can_transition_to(&self, target: &str) -> bool {
        let from = match &self.current_state {
            Some(name) => name.as_str(),
            None => return false,
        };

        let key = Self::transition_key(from, target);
        match self.transitions.get(&key) {
            Some(entry) => match &entry.guard {
                Some(guard) => guard() == GuardResult::Allow,
                None => true,
            },
            None => false,
        }
    }

    /// Attempt to transition to the given target state.
    ///
    /// Returns `Ok(())` on success. Returns `Err` if the transition is not
    /// registered, the machine is not started, the machine is in a terminal
    /// state, or a guard denies the transition.
    pub fn transition_to(&mut self, target: &str) -> Result<()> {
        let from = match &self.current_state {
            Some(name) => name.clone(),
            None => {
                return Err(Error::Configuration(
                    "State machine has not been started".into(),
                ))
            }
        };

        // Check terminal.
        if self
            .states
            .get(&from)
            .map(|s| s.is_terminal)
            .unwrap_or(false)
        {
            return Err(Error::Configuration(format!(
                "Cannot transition from terminal state '{}'",
                from
            )));
        }

        let key = Self::transition_key(&from, target);

        // Evaluate guard if present.
        let guard_result = match self.transitions.get(&key) {
            Some(entry) => {
                if let Some(guard) = &entry.guard {
                    self.stats.total_guard_evals += 1;
                    let result = guard();
                    if result == GuardResult::Deny {
                        self.stats.total_guard_denials += 1;
                        return Err(Error::Configuration(format!(
                            "Guard denied transition '{}' → '{}'",
                            from, target
                        )));
                    }
                    result
                } else {
                    GuardResult::Allow
                }
            }
            None => {
                return Err(Error::Configuration(format!(
                    "No transition registered from '{}' to '{}'",
                    from, target
                )));
            }
        };

        debug_assert_eq!(guard_result, GuardResult::Allow);

        // Record history.
        let label = self
            .transitions
            .get(&key)
            .and_then(|e| e.def.label.clone());
        let record = TransitionRecord {
            from: from.clone(),
            to: target.to_string(),
            label,
            tick: self.tick,
            ticks_in_source: self.ticks_in_current,
        };

        if self.history.len() >= self.config.max_history {
            self.history.pop_front();
        }
        self.history.push_back(record);

        // Update transition fire count.
        if let Some(entry) = self.transitions.get_mut(&key) {
            entry.def.fire_count += 1;
        }

        // Update source state total ticks.
        if let Some(state) = self.states.get_mut(&from) {
            state.total_ticks += self.ticks_in_current;
        }

        // Update target state entry count and track visited.
        let first_visit = if let Some(state) = self.states.get_mut(target) {
            let first = state.entry_count == 0;
            state.entry_count += 1;
            first
        } else {
            false
        };
        if first_visit {
            self.stats.states_visited += 1;
        }

        // Switch state.
        self.current_state = Some(target.to_string());
        self.stats.total_transitions += 1;
        self.ticks_in_current = 0;
        self.transitions_this_tick += 1;

        Ok(())
    }

    // -------------------------------------------------------------------
    // Tick
    // -------------------------------------------------------------------

    /// Advance the state machine by one tick, updating counters and
    /// diagnostics.
    pub fn tick(&mut self) {
        self.tick += 1;
        self.ticks_in_current += 1;
        self.stats.total_ticks += 1;

        // Accumulate ticks for current state.
        // (Note: total_ticks on the StateDef is only bumped on _exit_ via
        // transition; we keep a live counter via ticks_in_current.)

        // Compute instantaneous metrics.
        let transition_rate = self.transitions_this_tick as f64;
        let ticks_per_state = if self.stats.total_transitions > 0 {
            self.stats.total_ticks as f64 / self.stats.total_transitions as f64
        } else {
            self.ticks_in_current as f64
        };

        // EMA update.
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_ticks_per_state = ticks_per_state;
            self.stats.ema_transition_rate = transition_rate;
            self.ema_initialized = true;
        } else {
            self.stats.ema_ticks_per_state =
                alpha * ticks_per_state + (1.0 - alpha) * self.stats.ema_ticks_per_state;
            self.stats.ema_transition_rate =
                alpha * transition_rate + (1.0 - alpha) * self.stats.ema_transition_rate;
        }

        // Windowed snapshot.
        let snapshot = TickSnapshot {
            state_name: self
                .current_state
                .clone()
                .unwrap_or_else(|| "<none>".to_string()),
            ticks_in_current: self.ticks_in_current,
            transitions_total: self.stats.total_transitions,
            states_visited_total: self.stats.states_visited,
        };
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(snapshot);

        // Reset per-tick counter.
        self.transitions_this_tick = 0;
    }

    /// Current tick counter.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Alias for `tick()`.
    pub fn process(&mut self) {
        self.tick();
    }

    // -------------------------------------------------------------------
    // History
    // -------------------------------------------------------------------

    /// Returns the transition history.
    pub fn transition_history(&self) -> &VecDeque<TransitionRecord> {
        &self.history
    }

    /// Returns the most recent transition record, if any.
    pub fn last_transition(&self) -> Option<&TransitionRecord> {
        self.history.back()
    }

    // -------------------------------------------------------------------
    // Statistics & diagnostics
    // -------------------------------------------------------------------

    /// Returns a reference to cumulative statistics.
    pub fn stats(&self) -> &StateMachineStats {
        &self.stats
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &StateMachineConfig {
        &self.config
    }

    /// EMA-smoothed average ticks per state.
    pub fn smoothed_ticks_per_state(&self) -> f64 {
        self.stats.ema_ticks_per_state
    }

    /// EMA-smoothed transition rate (transitions per tick).
    pub fn smoothed_transition_rate(&self) -> f64 {
        self.stats.ema_transition_rate
    }

    /// Windowed average ticks-in-current-state across recent snapshots.
    pub fn windowed_ticks_in_current(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| s.ticks_in_current as f64)
            .sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed fraction of snapshots where the machine was in a given state.
    pub fn windowed_state_fraction(&self, state_name: &str) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let count = self
            .recent
            .iter()
            .filter(|s| s.state_name == state_name)
            .count();
        count as f64 / self.recent.len() as f64
    }

    /// Whether the transition rate appears to be declining (second half of
    /// window has fewer cumulative transitions than expected).
    pub fn is_throughput_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_last = self.recent[mid - 1].transitions_total as f64;
        let first_first = self.recent[0].transitions_total as f64;
        let second_last = self.recent[self.recent.len() - 1].transitions_total as f64;
        let second_first = self.recent[mid].transitions_total as f64;

        let first_half_delta = first_last - first_first;
        let second_half_delta = second_last - second_first;

        if first_half_delta <= 0.0 {
            return false;
        }
        second_half_delta < first_half_delta * 0.8
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    /// Reset the state machine to its initial (un-started) state, preserving
    /// registered states and transitions (but clearing their counters).
    pub fn reset(&mut self) {
        self.current_state = None;
        self.ticks_in_current = 0;
        self.tick = 0;
        self.transitions_this_tick = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.history.clear();
        self.stats = StateMachineStats::default();

        for state in self.states.values_mut() {
            state.total_ticks = 0;
            state.entry_count = 0;
        }
        for entry in self.transitions.values_mut() {
            entry.def.fire_count = 0;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> StateMachineConfig {
        StateMachineConfig {
            max_states: 8,
            max_history: 10,
            ema_decay: 0.5,
            window_size: 5,
        }
    }

    /// Helper: build a simple three-state machine (idle → active → done).
    fn build_simple_machine() -> StateMachine {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("idle", false).unwrap();
        sm.add_state("active", false).unwrap();
        sm.add_state("done", true).unwrap();
        sm.add_transition("idle", "active", Some("begin".into()))
            .unwrap();
        sm.add_transition("active", "done", Some("finish".into()))
            .unwrap();
        sm.add_transition("active", "idle", Some("reset".into()))
            .unwrap();
        sm
    }

    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_default() {
        let sm = StateMachine::new();
        assert_eq!(sm.state_count(), 0);
        assert_eq!(sm.transition_count(), 0);
        assert!(!sm.is_started());
        assert!(!sm.is_terminal());
        assert_eq!(sm.current_tick(), 0);
    }

    #[test]
    fn test_with_config() {
        let sm = StateMachine::with_config(small_config());
        assert_eq!(sm.config().max_states, 8);
        assert_eq!(sm.config().window_size, 5);
    }

    // -------------------------------------------------------------------
    // State registration
    // -------------------------------------------------------------------

    #[test]
    fn test_add_state() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("idle", false).unwrap();
        assert!(sm.has_state("idle"));
        assert_eq!(sm.state_count(), 1);
    }

    #[test]
    fn test_add_state_duplicate() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("idle", false).unwrap();
        assert!(sm.add_state("idle", false).is_err());
    }

    #[test]
    fn test_add_state_at_capacity() {
        let mut sm = StateMachine::with_config(small_config());
        for i in 0..8 {
            sm.add_state(format!("s{}", i), false).unwrap();
        }
        assert!(sm.add_state("overflow", false).is_err());
    }

    #[test]
    fn test_state_names() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("b", false).unwrap();
        sm.add_state("a", false).unwrap();
        assert_eq!(sm.state_names(), vec!["b", "a"]); // insertion order
    }

    #[test]
    fn test_state_query() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("idle", true).unwrap();
        let def = sm.state("idle").unwrap();
        assert!(def.is_terminal);
        assert_eq!(def.entry_count, 0);
        assert_eq!(def.total_ticks, 0);
    }

    // -------------------------------------------------------------------
    // Transition registration
    // -------------------------------------------------------------------

    #[test]
    fn test_add_transition() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_transition("a", "b", None).unwrap();
        assert_eq!(sm.transition_count(), 1);
        assert!(sm.transition_def("a", "b").is_some());
    }

    #[test]
    fn test_add_transition_missing_source() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("b", false).unwrap();
        assert!(sm.add_transition("a", "b", None).is_err());
    }

    #[test]
    fn test_add_transition_missing_target() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        assert!(sm.add_transition("a", "b", None).is_err());
    }

    #[test]
    fn test_add_transition_duplicate() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_transition("a", "b", None).unwrap();
        assert!(sm.add_transition("a", "b", None).is_err());
    }

    #[test]
    fn test_add_guarded_transition() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_guarded_transition("a", "b", None, || GuardResult::Allow)
            .unwrap();
        let def = sm.transition_def("a", "b").unwrap();
        assert!(def.has_guard);
    }

    #[test]
    fn test_successors_updated() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_state("c", false).unwrap();
        sm.add_transition("a", "b", None).unwrap();
        sm.add_transition("a", "c", None).unwrap();
        let def = sm.state("a").unwrap();
        assert!(def.successors.contains(&"b".to_string()));
        assert!(def.successors.contains(&"c".to_string()));
    }

    // -------------------------------------------------------------------
    // Start & current state
    // -------------------------------------------------------------------

    #[test]
    fn test_start() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        assert!(sm.is_started());
        assert_eq!(sm.current_state(), Some("idle"));
        assert!(!sm.is_terminal());
    }

    #[test]
    fn test_start_unknown_state() {
        let mut sm = StateMachine::with_config(small_config());
        assert!(sm.start("nonexistent").is_err());
    }

    #[test]
    fn test_start_twice() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        assert!(sm.start("active").is_err());
    }

    #[test]
    fn test_start_increments_entry_count() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        assert_eq!(sm.state("idle").unwrap().entry_count, 1);
    }

    // -------------------------------------------------------------------
    // Transitions
    // -------------------------------------------------------------------

    #[test]
    fn test_transition_basic() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        for _ in 0..3 {
            sm.tick();
        }
        sm.transition_to("active").unwrap();
        assert_eq!(sm.current_state(), Some("active"));
        assert_eq!(sm.ticks_in_current_state(), 0);
    }

    #[test]
    fn test_transition_to_terminal() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        sm.transition_to("active").unwrap();
        sm.transition_to("done").unwrap();
        assert!(sm.is_terminal());
    }

    #[test]
    fn test_transition_from_terminal_fails() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        sm.transition_to("active").unwrap();
        sm.transition_to("done").unwrap();
        assert!(sm.transition_to("idle").is_err());
    }

    #[test]
    fn test_transition_not_started() {
        let mut sm = build_simple_machine();
        assert!(sm.transition_to("active").is_err());
    }

    #[test]
    fn test_transition_unregistered() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        assert!(sm.transition_to("done").is_err()); // no direct idle→done
    }

    #[test]
    fn test_transition_guard_deny() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_guarded_transition("a", "b", None, || GuardResult::Deny)
            .unwrap();
        sm.start("a").unwrap();
        assert!(sm.transition_to("b").is_err());
        assert_eq!(sm.stats().total_guard_evals, 1);
        assert_eq!(sm.stats().total_guard_denials, 1);
    }

    #[test]
    fn test_transition_guard_allow() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_guarded_transition("a", "b", None, || GuardResult::Allow)
            .unwrap();
        sm.start("a").unwrap();
        sm.transition_to("b").unwrap();
        assert_eq!(sm.current_state(), Some("b"));
        assert_eq!(sm.stats().total_guard_evals, 1);
        assert_eq!(sm.stats().total_guard_denials, 0);
    }

    #[test]
    fn test_can_transition_to() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        assert!(sm.can_transition_to("active"));
        assert!(!sm.can_transition_to("done"));
        assert!(!sm.can_transition_to("nonexistent"));
    }

    #[test]
    fn test_available_transitions() {
        let mut sm = build_simple_machine();
        sm.start("active").unwrap();
        let avail = sm.available_transitions();
        assert!(avail.contains(&"done"));
        assert!(avail.contains(&"idle"));
    }

    #[test]
    fn test_available_transitions_not_started() {
        let sm = build_simple_machine();
        assert!(sm.available_transitions().is_empty());
    }

    // -------------------------------------------------------------------
    // Transition history
    // -------------------------------------------------------------------

    #[test]
    fn test_transition_history() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        for _ in 0..3 {
            sm.tick();
        }
        sm.transition_to("active").unwrap();
        sm.transition_to("done").unwrap();

        let history = sm.transition_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].from, "idle");
        assert_eq!(history[0].to, "active");
        assert_eq!(history[0].ticks_in_source, 3);
        assert_eq!(history[1].from, "active");
        assert_eq!(history[1].to, "done");
    }

    #[test]
    fn test_last_transition() {
        let mut sm = build_simple_machine();
        assert!(sm.last_transition().is_none());
        sm.start("idle").unwrap();
        sm.transition_to("active").unwrap();
        let last = sm.last_transition().unwrap();
        assert_eq!(last.from, "idle");
        assert_eq!(last.to, "active");
    }

    #[test]
    fn test_history_capped() {
        let mut sm = StateMachine::with_config(small_config()); // max_history=10
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_transition("a", "b", None).unwrap();
        sm.add_transition("b", "a", None).unwrap();
        sm.start("a").unwrap();
        for _ in 0..20 {
            sm.transition_to("b").unwrap();
            sm.transition_to("a").unwrap();
        }
        assert!(sm.transition_history().len() <= 10);
    }

    // -------------------------------------------------------------------
    // Fire count
    // -------------------------------------------------------------------

    #[test]
    fn test_fire_count() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_transition("a", "b", None).unwrap();
        sm.add_transition("b", "a", None).unwrap();
        sm.start("a").unwrap();
        sm.transition_to("b").unwrap();
        sm.transition_to("a").unwrap();
        sm.transition_to("b").unwrap();
        assert_eq!(sm.transition_def("a", "b").unwrap().fire_count, 2);
        assert_eq!(sm.transition_def("b", "a").unwrap().fire_count, 1);
    }

    // -------------------------------------------------------------------
    // Per-state counters
    // -------------------------------------------------------------------

    #[test]
    fn test_state_total_ticks() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        for _ in 0..5 {
            sm.tick();
        }
        sm.transition_to("active").unwrap();
        // idle's total_ticks should be 5 (accumulated on exit).
        assert_eq!(sm.state("idle").unwrap().total_ticks, 5);
    }

    #[test]
    fn test_state_entry_count() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_transition("a", "b", None).unwrap();
        sm.add_transition("b", "a", None).unwrap();
        sm.start("a").unwrap();
        sm.transition_to("b").unwrap();
        sm.transition_to("a").unwrap();
        sm.transition_to("b").unwrap();
        assert_eq!(sm.state("a").unwrap().entry_count, 2); // start + 1 re-entry
        assert_eq!(sm.state("b").unwrap().entry_count, 2);
    }

    // -------------------------------------------------------------------
    // Tick & EMA
    // -------------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut sm = StateMachine::new();
        sm.tick();
        sm.tick();
        assert_eq!(sm.current_tick(), 2);
    }

    #[test]
    fn test_ticks_in_current_state() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        sm.tick();
        sm.tick();
        assert_eq!(sm.ticks_in_current_state(), 2);
        sm.transition_to("active").unwrap();
        assert_eq!(sm.ticks_in_current_state(), 0);
        sm.tick();
        assert_eq!(sm.ticks_in_current_state(), 1);
    }

    #[test]
    fn test_process_alias() {
        let mut sm = StateMachine::new();
        sm.process();
        assert_eq!(sm.current_tick(), 1);
    }

    #[test]
    fn test_ema_initialises_on_first_tick() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        sm.tick();
        // With no transitions yet, ticks_per_state = 1.0
        assert!(sm.smoothed_ticks_per_state() >= 1.0);
    }

    #[test]
    fn test_ema_blends() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        for _ in 0..5 {
            sm.tick();
        }
        sm.transition_to("active").unwrap();
        for _ in 0..3 {
            sm.tick();
        }
        // EMA should be positive and > 0.
        assert!(sm.smoothed_ticks_per_state() > 0.0);
    }

    // -------------------------------------------------------------------
    // Windowed diagnostics
    // -------------------------------------------------------------------

    #[test]
    fn test_windowed_ticks_in_current_empty() {
        let sm = StateMachine::new();
        assert!(sm.windowed_ticks_in_current().is_none());
    }

    #[test]
    fn test_windowed_ticks_in_current() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        for _ in 0..5 {
            sm.tick();
        }
        let avg = sm.windowed_ticks_in_current().unwrap();
        // Ticks 1..5, average = 3.0
        assert!((avg - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_state_fraction() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        for _ in 0..3 {
            sm.tick();
        }
        sm.transition_to("active").unwrap();
        for _ in 0..2 {
            sm.tick();
        }
        // 3 ticks in idle, 2 in active → idle fraction = 3/5 = 0.6
        assert!((sm.windowed_state_fraction("idle") - 0.6).abs() < 1e-9);
        assert!((sm.windowed_state_fraction("active") - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_state_fraction_empty() {
        let sm = StateMachine::new();
        assert!((sm.windowed_state_fraction("idle") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_rolls() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        for _ in 0..20 {
            sm.tick();
        }
        assert!(sm.recent.len() <= 5);
    }

    // -------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------

    #[test]
    fn test_stats_total_transitions() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        sm.transition_to("active").unwrap();
        sm.transition_to("done").unwrap();
        assert_eq!(sm.stats().total_transitions, 2);
    }

    #[test]
    fn test_stats_states_visited() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap(); // visited: {idle}
        sm.transition_to("active").unwrap(); // visited: {idle, active}
        assert_eq!(sm.stats().states_visited, 2);
    }

    #[test]
    fn test_stats_states_visited_no_double_count() {
        let mut sm = StateMachine::with_config(small_config());
        sm.add_state("a", false).unwrap();
        sm.add_state("b", false).unwrap();
        sm.add_transition("a", "b", None).unwrap();
        sm.add_transition("b", "a", None).unwrap();
        sm.start("a").unwrap(); // visited: {a}
        sm.transition_to("b").unwrap(); // visited: {a, b}
        sm.transition_to("a").unwrap(); // still {a, b}
        assert_eq!(sm.stats().states_visited, 2);
    }

    // -------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        for _ in 0..5 {
            sm.tick();
        }
        sm.transition_to("active").unwrap();
        sm.tick();

        sm.reset();

        assert!(!sm.is_started());
        assert_eq!(sm.current_state(), None);
        assert_eq!(sm.current_tick(), 0);
        assert_eq!(sm.ticks_in_current_state(), 0);
        assert!(sm.transition_history().is_empty());
        assert!(sm.windowed_ticks_in_current().is_none());
        assert_eq!(sm.stats().total_transitions, 0);
        assert_eq!(sm.stats().total_ticks, 0);
        assert_eq!(sm.stats().states_visited, 0);

        // States and transitions still registered but counters reset.
        assert_eq!(sm.state_count(), 3);
        assert_eq!(sm.transition_count(), 3);
        assert_eq!(sm.state("idle").unwrap().entry_count, 0);
        assert_eq!(sm.state("idle").unwrap().total_ticks, 0);
        assert_eq!(sm.transition_def("idle", "active").unwrap().fire_count, 0);

        // Can re-start after reset.
        sm.start("idle").unwrap();
        assert!(sm.is_started());
    }

    // -------------------------------------------------------------------
    // Full lifecycle
    // -------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut sm = build_simple_machine();

        // Start in idle.
        sm.start("idle").unwrap();
        assert_eq!(sm.current_state(), Some("idle"));

        // Tick in idle.
        for _ in 0..4 {
            sm.tick();
        }

        // Transition to active.
        assert!(sm.can_transition_to("active"));
        sm.transition_to("active").unwrap();
        assert_eq!(sm.current_state(), Some("active"));

        // Tick in active.
        for _ in 0..3 {
            sm.tick();
        }

        // Go back to idle.
        sm.transition_to("idle").unwrap();
        sm.tick();

        // Back to active and then done.
        sm.transition_to("active").unwrap();
        sm.transition_to("done").unwrap();
        assert!(sm.is_terminal());

        // Cannot leave terminal.
        assert!(sm.transition_to("idle").is_err());

        // Verify stats.
        let stats = sm.stats();
        assert_eq!(stats.total_transitions, 4);
        assert!(stats.total_ticks > 0);
        assert_eq!(stats.states_visited, 3); // idle, active, done

        // Verify history.
        let history = sm.transition_history();
        assert_eq!(history.len(), 4);
        assert_eq!(history[0].from, "idle");
        assert_eq!(history[0].to, "active");
        assert_eq!(history[0].ticks_in_source, 4);

        // Verify per-state counts.
        assert_eq!(sm.state("idle").unwrap().entry_count, 2);
        assert_eq!(sm.state("active").unwrap().entry_count, 2);
        assert_eq!(sm.state("done").unwrap().entry_count, 1);

        // Diagnostics should be available.
        assert!(sm.smoothed_ticks_per_state() > 0.0);
    }

    #[test]
    fn test_transition_label() {
        let mut sm = build_simple_machine();
        sm.start("idle").unwrap();
        sm.transition_to("active").unwrap();
        let last = sm.last_transition().unwrap();
        assert_eq!(last.label.as_deref(), Some("begin"));
    }

    #[test]
    fn test_transition_def_query() {
        let sm = build_simple_machine();
        let def = sm.transition_def("idle", "active").unwrap();
        assert_eq!(def.from, "idle");
        assert_eq!(def.to, "active");
        assert_eq!(def.label.as_deref(), Some("begin"));
        assert!(!def.has_guard);
        assert_eq!(def.fire_count, 0);
    }

    #[test]
    fn test_transition_def_not_found() {
        let sm = build_simple_machine();
        assert!(sm.transition_def("idle", "done").is_none());
    }
}
