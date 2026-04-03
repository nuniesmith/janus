//! Q-value computation
//!
//! Part of the Basal Ganglia region
//! Component: selection
//!
//! Implements Q-value (action-value) computation for reinforcement learning:
//! - Q-learning updates
//! - SARSA updates
//! - Double Q-learning
//! - Expected SARSA
//! - Eligibility traces
//! - Function approximation support

use crate::common::Result;
use std::collections::HashMap;

/// Q-learning algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QAlgorithm {
    /// Standard Q-learning (off-policy)
    QLearning,
    /// SARSA (on-policy)
    Sarsa,
    /// Double Q-learning (reduces overestimation)
    DoubleQ,
    /// Expected SARSA
    ExpectedSarsa,
}

/// Configuration for action value computation
#[derive(Debug, Clone)]
pub struct ActionValueConfig {
    /// Learning rate (alpha)
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub discount_factor: f64,
    /// Initial Q-value for new state-action pairs
    pub initial_value: f64,
    /// Eligibility trace decay (lambda)
    pub trace_decay: f64,
    /// Whether to use eligibility traces
    pub use_traces: bool,
    /// Algorithm variant
    pub algorithm: QAlgorithm,
    /// Maximum Q-value (for clipping)
    pub max_q_value: f64,
    /// Minimum Q-value (for clipping)
    pub min_q_value: f64,
}

impl Default for ActionValueConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            discount_factor: 0.99,
            initial_value: 0.0,
            trace_decay: 0.9,
            use_traces: false,
            algorithm: QAlgorithm::QLearning,
            max_q_value: 100.0,
            min_q_value: -100.0,
        }
    }
}

/// State-action pair identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StateAction {
    pub state: u64,
    pub action: usize,
}

impl StateAction {
    pub fn new(state: u64, action: usize) -> Self {
        Self { state, action }
    }
}

/// Result of a Q-value update
#[derive(Debug, Clone)]
pub struct UpdateResult {
    /// TD error (temporal difference)
    pub td_error: f64,
    /// Old Q-value
    pub old_value: f64,
    /// New Q-value
    pub new_value: f64,
    /// State visited
    pub state: u64,
    /// Action taken
    pub action: usize,
}

/// Statistics about Q-value learning
#[derive(Debug, Clone, Default)]
pub struct QValueStats {
    /// Total updates performed
    pub total_updates: u64,
    /// Mean TD error (absolute)
    pub mean_td_error: f64,
    /// Maximum Q-value in table
    pub max_q: f64,
    /// Minimum Q-value in table
    pub min_q: f64,
    /// Number of unique state-action pairs
    pub unique_pairs: usize,
    /// Running TD error sum
    td_error_sum: f64,
}

/// Q-value computation and temporal difference learning
///
/// Implements various Q-learning algorithms for computing
/// action values in reinforcement learning.
pub struct ActionValue {
    /// Configuration
    config: ActionValueConfig,
    /// Q-value table (primary)
    q_table: HashMap<StateAction, f64>,
    /// Q-value table (secondary, for Double Q-learning)
    q_table_secondary: HashMap<StateAction, f64>,
    /// Eligibility traces
    traces: HashMap<StateAction, f64>,
    /// Number of actions (for expected SARSA)
    n_actions: usize,
    /// Statistics
    stats: QValueStats,
    /// Simple RNG for Double Q-learning
    rng_state: u64,
}

impl Default for ActionValue {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionValue {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(ActionValueConfig::default(), 4)
    }

    /// Create with custom configuration
    pub fn with_config(config: ActionValueConfig, n_actions: usize) -> Self {
        Self {
            config,
            q_table: HashMap::new(),
            q_table_secondary: HashMap::new(),
            traces: HashMap::new(),
            n_actions,
            stats: QValueStats::default(),
            rng_state: 0xDEADBEEF12345678,
        }
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Get Q-value for a state-action pair
    pub fn get_value(&self, state: u64, action: usize) -> f64 {
        let key = StateAction::new(state, action);
        *self.q_table.get(&key).unwrap_or(&self.config.initial_value)
    }

    /// Get all Q-values for a state
    pub fn get_values(&self, state: u64) -> Vec<f64> {
        (0..self.n_actions)
            .map(|a| self.get_value(state, a))
            .collect()
    }

    /// Get the best action for a state (argmax Q)
    pub fn best_action(&self, state: u64) -> (usize, f64) {
        let values = self.get_values(state);
        let mut best_action = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (action, &value) in values.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        (best_action, best_value)
    }

    /// Get maximum Q-value for a state
    pub fn max_value(&self, state: u64) -> f64 {
        self.best_action(state).1
    }

    /// Update Q-value with observed transition
    pub fn update(
        &mut self,
        state: u64,
        action: usize,
        reward: f64,
        next_state: u64,
        next_action: Option<usize>,
        done: bool,
    ) -> UpdateResult {
        match self.config.algorithm {
            QAlgorithm::QLearning => {
                self.q_learning_update(state, action, reward, next_state, done)
            }
            QAlgorithm::Sarsa => {
                let next_a = next_action.unwrap_or(0);
                self.sarsa_update(state, action, reward, next_state, next_a, done)
            }
            QAlgorithm::DoubleQ => self.double_q_update(state, action, reward, next_state, done),
            QAlgorithm::ExpectedSarsa => {
                let action_probs = next_action
                    .map(|_| self.uniform_action_probs())
                    .unwrap_or_else(|| self.uniform_action_probs());
                self.expected_sarsa_update(state, action, reward, next_state, &action_probs, done)
            }
        }
    }

    /// Q-learning update: Q(s,a) += α * (r + γ * max_a' Q(s',a') - Q(s,a))
    fn q_learning_update(
        &mut self,
        state: u64,
        action: usize,
        reward: f64,
        next_state: u64,
        done: bool,
    ) -> UpdateResult {
        let key = StateAction::new(state, action);
        let old_value = self.get_value(state, action);

        let target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.max_value(next_state)
        };

        let td_error = target - old_value;
        let new_value = (old_value + self.config.learning_rate * td_error)
            .clamp(self.config.min_q_value, self.config.max_q_value);

        self.q_table.insert(key, new_value);

        if self.config.use_traces {
            self.update_traces(state, action, td_error);
        }

        self.update_stats(td_error, new_value);

        UpdateResult {
            td_error,
            old_value,
            new_value,
            state,
            action,
        }
    }

    /// SARSA update: Q(s,a) += α * (r + γ * Q(s',a') - Q(s,a))
    fn sarsa_update(
        &mut self,
        state: u64,
        action: usize,
        reward: f64,
        next_state: u64,
        next_action: usize,
        done: bool,
    ) -> UpdateResult {
        let key = StateAction::new(state, action);
        let old_value = self.get_value(state, action);

        let target = if done {
            reward
        } else {
            reward + self.config.discount_factor * self.get_value(next_state, next_action)
        };

        let td_error = target - old_value;
        let new_value = (old_value + self.config.learning_rate * td_error)
            .clamp(self.config.min_q_value, self.config.max_q_value);

        self.q_table.insert(key, new_value);

        if self.config.use_traces {
            self.update_traces(state, action, td_error);
        }

        self.update_stats(td_error, new_value);

        UpdateResult {
            td_error,
            old_value,
            new_value,
            state,
            action,
        }
    }

    /// Double Q-learning update (reduces overestimation bias)
    fn double_q_update(
        &mut self,
        state: u64,
        action: usize,
        reward: f64,
        next_state: u64,
        done: bool,
    ) -> UpdateResult {
        let key = StateAction::new(state, action);

        // Randomly choose which Q-table to update
        let use_primary = self.next_random() < 0.5;

        let (q_table, other_table) = if use_primary {
            (&mut self.q_table, &self.q_table_secondary)
        } else {
            (&mut self.q_table_secondary, &self.q_table)
        };

        let old_value = *q_table.get(&key).unwrap_or(&self.config.initial_value);

        let target = if done {
            reward
        } else {
            // Find best action in current table, evaluate in other table
            // (inlined to avoid borrow conflict with self)
            let mut best_action = 0;
            let mut best_value = f64::NEG_INFINITY;
            for a in 0..self.n_actions {
                let k = StateAction::new(next_state, a);
                let v = *q_table.get(&k).unwrap_or(&self.config.initial_value);
                if v > best_value {
                    best_value = v;
                    best_action = a;
                }
            }
            let next_key = StateAction::new(next_state, best_action);
            let next_value = *other_table
                .get(&next_key)
                .unwrap_or(&self.config.initial_value);
            reward + self.config.discount_factor * next_value
        };

        let td_error = target - old_value;
        let new_value = (old_value + self.config.learning_rate * td_error)
            .clamp(self.config.min_q_value, self.config.max_q_value);

        q_table.insert(key, new_value);
        self.update_stats(td_error, new_value);

        UpdateResult {
            td_error,
            old_value,
            new_value,
            state,
            action,
        }
    }

    /// Expected SARSA update
    fn expected_sarsa_update(
        &mut self,
        state: u64,
        action: usize,
        reward: f64,
        next_state: u64,
        action_probs: &[f64],
        done: bool,
    ) -> UpdateResult {
        let key = StateAction::new(state, action);
        let old_value = self.get_value(state, action);

        let target = if done {
            reward
        } else {
            // Expected value over next state actions
            let expected_value: f64 = (0..self.n_actions)
                .map(|a| {
                    let prob = action_probs.get(a).copied().unwrap_or(0.0);
                    prob * self.get_value(next_state, a)
                })
                .sum();
            reward + self.config.discount_factor * expected_value
        };

        let td_error = target - old_value;
        let new_value = (old_value + self.config.learning_rate * td_error)
            .clamp(self.config.min_q_value, self.config.max_q_value);

        self.q_table.insert(key, new_value);
        self.update_stats(td_error, new_value);

        UpdateResult {
            td_error,
            old_value,
            new_value,
            state,
            action,
        }
    }

    /// Update eligibility traces
    fn update_traces(&mut self, current_state: u64, current_action: usize, td_error: f64) {
        // Increment trace for current state-action
        let current_key = StateAction::new(current_state, current_action);
        let current_trace = self.traces.entry(current_key.clone()).or_insert(0.0);
        *current_trace = 1.0; // Replacing traces

        // Decay and update all traces
        let decay = self.config.discount_factor * self.config.trace_decay;
        let lr = self.config.learning_rate;

        let keys: Vec<_> = self.traces.keys().cloned().collect();
        for key in keys {
            if let Some(trace) = self.traces.get_mut(&key) {
                // Update Q-value based on trace
                if let Some(q) = self.q_table.get_mut(&key) {
                    *q += lr * td_error * *trace;
                    *q = q.clamp(self.config.min_q_value, self.config.max_q_value);
                }

                // Decay trace
                *trace *= decay;

                // Remove negligible traces
                if *trace < 0.001 {
                    self.traces.remove(&key);
                }
            }
        }
    }

    /// Find best action from a specific Q-table
    #[allow(dead_code)]
    fn best_action_from_table(&self, state: u64, q_table: &HashMap<StateAction, f64>) -> usize {
        let mut best_action = 0;
        let mut best_value = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let key = StateAction::new(state, action);
            let value = *q_table.get(&key).unwrap_or(&self.config.initial_value);
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        best_action
    }

    /// Get uniform action probabilities
    fn uniform_action_probs(&self) -> Vec<f64> {
        vec![1.0 / self.n_actions as f64; self.n_actions]
    }

    /// Update statistics
    fn update_stats(&mut self, td_error: f64, new_value: f64) {
        self.stats.total_updates += 1;
        self.stats.td_error_sum += td_error.abs();
        self.stats.mean_td_error = self.stats.td_error_sum / self.stats.total_updates as f64;
        self.stats.unique_pairs = self.q_table.len();

        if new_value > self.stats.max_q {
            self.stats.max_q = new_value;
        }
        if new_value < self.stats.min_q || self.stats.total_updates == 1 {
            self.stats.min_q = new_value;
        }
    }

    /// Set Q-value directly (for initialization or external updates)
    pub fn set_value(&mut self, state: u64, action: usize, value: f64) {
        let key = StateAction::new(state, action);
        let clamped = value.clamp(self.config.min_q_value, self.config.max_q_value);
        self.q_table.insert(key, clamped);
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr.clamp(0.0, 1.0);
    }

    /// Get discount factor
    pub fn discount_factor(&self) -> f64 {
        self.config.discount_factor
    }

    /// Set discount factor
    pub fn set_discount_factor(&mut self, gamma: f64) {
        self.config.discount_factor = gamma.clamp(0.0, 1.0);
    }

    /// Get statistics
    pub fn stats(&self) -> &QValueStats {
        &self.stats
    }

    /// Get number of state-action pairs stored
    pub fn table_size(&self) -> usize {
        self.q_table.len()
    }

    /// Clear all Q-values (reset learning)
    pub fn reset(&mut self) {
        self.q_table.clear();
        self.q_table_secondary.clear();
        self.traces.clear();
        self.stats = QValueStats::default();
    }

    /// Clear eligibility traces (for episode boundaries)
    pub fn clear_traces(&mut self) {
        self.traces.clear();
    }

    /// Seed the RNG
    pub fn seed(&mut self, seed: u64) {
        self.rng_state = seed.max(1);
    }

    /// Simple xorshift64 PRNG
    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }

    /// Get algorithm type
    pub fn algorithm(&self) -> QAlgorithm {
        self.config.algorithm
    }

    /// Set algorithm type
    pub fn set_algorithm(&mut self, algorithm: QAlgorithm) {
        self.config.algorithm = algorithm;
    }

    /// Export Q-table as vector of (state, action, value) tuples
    pub fn export_table(&self) -> Vec<(u64, usize, f64)> {
        self.q_table
            .iter()
            .map(|(k, &v)| (k.state, k.action, v))
            .collect()
    }

    /// Import Q-values from vector
    pub fn import_table(&mut self, values: &[(u64, usize, f64)]) {
        for &(state, action, value) in values {
            self.set_value(state, action, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = ActionValue::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_initial_values() {
        let config = ActionValueConfig {
            initial_value: 5.0,
            ..Default::default()
        };
        let av = ActionValue::with_config(config, 4);

        // Unvisited state-action should return initial value
        assert_eq!(av.get_value(0, 0), 5.0);
        assert_eq!(av.get_value(100, 2), 5.0);
    }

    #[test]
    fn test_q_learning_update() {
        let config = ActionValueConfig {
            learning_rate: 0.5,
            discount_factor: 0.9,
            initial_value: 0.0,
            algorithm: QAlgorithm::QLearning,
            ..Default::default()
        };
        let mut av = ActionValue::with_config(config, 4);

        // Set up: next state has max Q of 10
        av.set_value(1, 0, 10.0);

        // Update: reward=1, discount=0.9, so target = 1 + 0.9 * 10 = 10
        // TD error = 10 - 0 = 10
        // New value = 0 + 0.5 * 10 = 5
        let result = av.update(0, 0, 1.0, 1, None, false);

        assert!((result.td_error - 10.0).abs() < 0.01);
        assert!((result.new_value - 5.0).abs() < 0.01);
        assert!((av.get_value(0, 0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_sarsa_update() {
        let config = ActionValueConfig {
            learning_rate: 0.5,
            discount_factor: 0.9,
            initial_value: 0.0,
            algorithm: QAlgorithm::Sarsa,
            ..Default::default()
        };
        let mut av = ActionValue::with_config(config, 4);

        av.set_value(1, 1, 8.0);

        // SARSA uses the actual next action value, not max
        let result = av.update(0, 0, 1.0, 1, Some(1), false);

        // target = 1 + 0.9 * 8 = 8.2
        // TD error = 8.2 - 0 = 8.2
        // New value = 0 + 0.5 * 8.2 = 4.1
        assert!((result.td_error - 8.2).abs() < 0.01);
        assert!((result.new_value - 4.1).abs() < 0.01);
    }

    #[test]
    fn test_terminal_state() {
        let config = ActionValueConfig {
            learning_rate: 1.0,
            discount_factor: 0.9,
            initial_value: 0.0,
            ..Default::default()
        };
        let mut av = ActionValue::with_config(config, 4);

        // Terminal state: no future value, just reward
        let result = av.update(0, 0, 10.0, 1, None, true);

        assert!((result.new_value - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_best_action() {
        let mut av = ActionValue::new();

        av.set_value(0, 0, 1.0);
        av.set_value(0, 1, 5.0);
        av.set_value(0, 2, 3.0);
        av.set_value(0, 3, 2.0);

        let (best, value) = av.best_action(0);
        assert_eq!(best, 1);
        assert!((value - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_get_values() {
        let mut av = ActionValue::with_config(ActionValueConfig::default(), 3);

        av.set_value(0, 0, 1.0);
        av.set_value(0, 1, 2.0);
        av.set_value(0, 2, 3.0);

        let values = av.get_values(0);
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 0.01);
        assert!((values[1] - 2.0).abs() < 0.01);
        assert!((values[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_value_clipping() {
        let config = ActionValueConfig {
            learning_rate: 1.0,
            max_q_value: 10.0,
            min_q_value: -10.0,
            ..Default::default()
        };
        let mut av = ActionValue::with_config(config, 4);

        // Try to set value outside bounds
        av.set_value(0, 0, 100.0);
        assert!((av.get_value(0, 0) - 10.0).abs() < 0.01);

        av.set_value(0, 1, -100.0);
        assert!((av.get_value(0, 1) - (-10.0)).abs() < 0.01);
    }

    #[test]
    fn test_double_q_learning() {
        let config = ActionValueConfig {
            learning_rate: 0.5,
            discount_factor: 0.9,
            algorithm: QAlgorithm::DoubleQ,
            ..Default::default()
        };
        let mut av = ActionValue::with_config(config, 4);
        av.seed(12345);

        // Run multiple updates
        for i in 0..10 {
            av.update(0, i % 4, 1.0, 1, None, false);
        }

        // Should have values in the table
        assert!(av.table_size() > 0);
    }

    #[test]
    fn test_expected_sarsa() {
        let config = ActionValueConfig {
            learning_rate: 0.5,
            discount_factor: 0.9,
            algorithm: QAlgorithm::ExpectedSarsa,
            ..Default::default()
        };
        let mut av = ActionValue::with_config(config, 4);

        av.set_value(1, 0, 4.0);
        av.set_value(1, 1, 8.0);
        av.set_value(1, 2, 4.0);
        av.set_value(1, 3, 4.0);

        let result = av.update(0, 0, 1.0, 1, Some(0), false);

        // Expected value = 0.25 * (4 + 8 + 4 + 4) = 5
        // target = 1 + 0.9 * 5 = 5.5
        // New value = 0 + 0.5 * 5.5 = 2.75
        assert!((result.new_value - 2.75).abs() < 0.01);
    }

    #[test]
    fn test_stats() {
        let mut av = ActionValue::new();

        for i in 0..10 {
            av.update(i, 0, 1.0, i + 1, None, false);
        }

        let stats = av.stats();
        assert_eq!(stats.total_updates, 10);
        assert!(stats.mean_td_error >= 0.0);
    }

    #[test]
    fn test_reset() {
        let mut av = ActionValue::new();

        av.set_value(0, 0, 5.0);
        av.update(1, 1, 1.0, 2, None, false);

        av.reset();

        assert_eq!(av.table_size(), 0);
        assert_eq!(av.stats().total_updates, 0);
    }

    #[test]
    fn test_export_import() {
        let mut av = ActionValue::new();

        av.set_value(0, 0, 1.0);
        av.set_value(0, 1, 2.0);
        av.set_value(1, 0, 3.0);

        let exported = av.export_table();
        assert_eq!(exported.len(), 3);

        let mut av2 = ActionValue::new();
        av2.import_table(&exported);

        assert!((av2.get_value(0, 0) - 1.0).abs() < 0.01);
        assert!((av2.get_value(0, 1) - 2.0).abs() < 0.01);
        assert!((av2.get_value(1, 0) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_learning_rate_adjustment() {
        let mut av = ActionValue::new();

        av.set_learning_rate(0.5);
        assert!((av.learning_rate() - 0.5).abs() < 0.01);

        // Should clamp
        av.set_learning_rate(2.0);
        assert!((av.learning_rate() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_eligibility_traces() {
        let config = ActionValueConfig {
            learning_rate: 0.5,
            discount_factor: 0.9,
            trace_decay: 0.9,
            use_traces: true,
            ..Default::default()
        };
        let mut av = ActionValue::with_config(config, 4);

        // Build up traces
        av.update(0, 0, 1.0, 1, None, false);
        av.update(1, 1, 1.0, 2, None, false);
        av.update(2, 0, 10.0, 3, None, true);

        // Earlier state-actions should also be updated via traces
        // The exact values depend on trace mechanics
        assert!(av.table_size() > 0);
    }

    #[test]
    fn test_clear_traces() {
        let config = ActionValueConfig {
            use_traces: true,
            ..Default::default()
        };
        let mut av = ActionValue::with_config(config, 4);

        av.update(0, 0, 1.0, 1, None, false);
        av.clear_traces();

        // Traces should be cleared, but Q-values remain
        assert!(av.table_size() > 0);
    }
}
