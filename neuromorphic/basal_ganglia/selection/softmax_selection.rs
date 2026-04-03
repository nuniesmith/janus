//! Softmax Action Selection
//!
//! Implements probabilistic action selection using softmax (Boltzmann) distribution:
//! - Temperature-controlled exploration/exploitation tradeoff
//! - Action value to probability conversion
//! - Entropy-based exploration monitoring
//!
//! Part of the Basal Ganglia region - Selection component

use crate::common::Result;
use rand::RngExt;

/// Configuration for softmax selection
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Temperature parameter (τ)
    /// - High temperature (τ >> 1): More random selection (exploration)
    /// - Low temperature (τ << 1): More greedy selection (exploitation)
    /// - τ = 1: Standard softmax
    pub temperature: f32,
    /// Minimum temperature (to prevent division by zero)
    pub min_temperature: f32,
    /// Temperature decay rate per step
    pub temperature_decay: f32,
    /// Target temperature after decay
    pub target_temperature: f32,
    /// Whether to use adaptive temperature based on value spread
    pub adaptive_temperature: bool,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            min_temperature: 0.01,
            temperature_decay: 0.9999,
            target_temperature: 0.1,
            adaptive_temperature: false,
        }
    }
}

/// Softmax (Boltzmann) action selection
pub struct SoftmaxSelection {
    /// Configuration
    config: SoftmaxConfig,
    /// Current temperature
    current_temperature: f32,
    /// Last computed probabilities
    last_probs: Vec<f32>,
    /// Selection history for analysis
    selection_history: SelectionHistory,
    /// Step counter for temperature annealing
    step_count: u64,
}

/// Result of action selection
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Selected action index
    pub action: usize,
    /// Probability of selected action
    pub probability: f32,
    /// All action probabilities
    pub probabilities: Vec<f32>,
    /// Entropy of the distribution
    pub entropy: f32,
    /// Current temperature used
    pub temperature: f32,
}

/// Selection statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct SelectionHistory {
    /// Total selections made
    pub total_selections: u64,
    /// Action selection counts
    pub action_counts: Vec<u64>,
    /// Running average entropy
    pub avg_entropy: f32,
    /// Running average max probability
    pub avg_max_prob: f32,
    /// EMA decay factor
    ema_decay: f32,
}

impl SelectionHistory {
    fn new(num_actions: usize) -> Self {
        Self {
            total_selections: 0,
            action_counts: vec![0; num_actions],
            avg_entropy: 0.0,
            avg_max_prob: 0.0,
            ema_decay: 0.99,
        }
    }

    fn record(&mut self, action: usize, entropy: f32, max_prob: f32) {
        self.total_selections += 1;
        if action < self.action_counts.len() {
            self.action_counts[action] += 1;
        }
        self.avg_entropy = self.ema_decay * self.avg_entropy + (1.0 - self.ema_decay) * entropy;
        self.avg_max_prob = self.ema_decay * self.avg_max_prob + (1.0 - self.ema_decay) * max_prob;
    }

    /// Get action selection frequencies
    pub fn get_frequencies(&self) -> Vec<f32> {
        if self.total_selections == 0 {
            return vec![0.0; self.action_counts.len()];
        }
        self.action_counts
            .iter()
            .map(|&c| c as f32 / self.total_selections as f32)
            .collect()
    }
}

impl Default for SoftmaxSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftmaxSelection {
    /// Create a new softmax selection instance with default config
    pub fn new() -> Self {
        Self::with_config(SoftmaxConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SoftmaxConfig) -> Self {
        Self {
            current_temperature: config.temperature,
            last_probs: Vec::new(),
            selection_history: SelectionHistory::new(10), // Default capacity
            step_count: 0,
            config,
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &SoftmaxConfig {
        &self.config
    }

    /// Get current temperature
    pub fn temperature(&self) -> f32 {
        self.current_temperature
    }

    /// Set temperature manually
    pub fn set_temperature(&mut self, temperature: f32) {
        self.current_temperature = temperature.max(self.config.min_temperature);
    }

    /// Convert action values to probabilities using softmax
    ///
    /// P(a) = exp(Q(a) / τ) / Σ exp(Q(a') / τ)
    pub fn compute_probabilities(&mut self, action_values: &[f32]) -> Vec<f32> {
        let temperature = if self.config.adaptive_temperature {
            self.compute_adaptive_temperature(action_values)
        } else {
            self.current_temperature
        };

        // Compute softmax with numerical stability
        let scaled: Vec<f32> = action_values.iter().map(|&v| v / temperature).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_values: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exp_values.iter().sum();

        let probs: Vec<f32> = exp_values.iter().map(|&e| e / sum.max(1e-8)).collect();

        self.last_probs = probs.clone();
        probs
    }

    /// Compute adaptive temperature based on value spread
    fn compute_adaptive_temperature(&self, action_values: &[f32]) -> f32 {
        if action_values.is_empty() {
            return self.current_temperature;
        }

        let max_val = action_values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_val = action_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let spread = max_val - min_val;

        // Scale temperature based on value spread
        // Larger spread -> lower temperature (more confident selection)
        // Smaller spread -> higher temperature (more exploration)
        let adaptive_temp = if spread > 0.0 {
            (self.current_temperature / spread.sqrt()).max(self.config.min_temperature)
        } else {
            self.current_temperature
        };

        adaptive_temp.clamp(self.config.min_temperature, self.config.temperature * 2.0)
    }

    /// Select an action based on action values
    pub fn select(&mut self, action_values: &[f32]) -> SelectionResult {
        // Resize history if needed
        if self.selection_history.action_counts.len() != action_values.len() {
            self.selection_history = SelectionHistory::new(action_values.len());
        }

        let probabilities = self.compute_probabilities(action_values);

        // Sample action from probability distribution
        let mut rng = rand::rng();
        let sample: f32 = rng.random();

        let mut cumsum = 0.0;
        let mut action = probabilities.len().saturating_sub(1);
        for (i, &p) in probabilities.iter().enumerate() {
            cumsum += p;
            if sample <= cumsum {
                action = i;
                break;
            }
        }

        let probability = probabilities[action];
        let entropy = self.compute_entropy(&probabilities);
        let max_prob = probabilities.iter().cloned().fold(0.0, f32::max);

        // Record in history
        self.selection_history.record(action, entropy, max_prob);

        // Decay temperature
        self.step_count += 1;
        self.decay_temperature();

        SelectionResult {
            action,
            probability,
            probabilities,
            entropy,
            temperature: self.current_temperature,
        }
    }

    /// Select action with exploration bonus
    pub fn select_with_bonus(
        &mut self,
        action_values: &[f32],
        exploration_bonus: &[f32],
    ) -> SelectionResult {
        assert_eq!(
            action_values.len(),
            exploration_bonus.len(),
            "Action values and exploration bonus must have same length"
        );

        let augmented_values: Vec<f32> = action_values
            .iter()
            .zip(exploration_bonus.iter())
            .map(|(&v, &b)| v + b)
            .collect();

        self.select(&augmented_values)
    }

    /// Compute entropy of probability distribution
    ///
    /// H(π) = -Σ P(a) * log(P(a))
    pub fn compute_entropy(&self, probabilities: &[f32]) -> f32 {
        -probabilities
            .iter()
            .map(|&p| if p > 1e-8 { p * p.ln() } else { 0.0 })
            .sum::<f32>()
    }

    /// Get maximum entropy for uniform distribution
    pub fn max_entropy(&self, num_actions: usize) -> f32 {
        if num_actions <= 1 {
            return 0.0;
        }
        (num_actions as f32).ln()
    }

    /// Get normalized entropy (0 = deterministic, 1 = uniform)
    pub fn normalized_entropy(&self, probabilities: &[f32]) -> f32 {
        let entropy = self.compute_entropy(probabilities);
        let max_ent = self.max_entropy(probabilities.len());
        if max_ent > 0.0 {
            entropy / max_ent
        } else {
            0.0
        }
    }

    /// Decay temperature according to schedule
    fn decay_temperature(&mut self) {
        if self.current_temperature > self.config.target_temperature {
            self.current_temperature *= self.config.temperature_decay;
            self.current_temperature = self
                .current_temperature
                .max(self.config.target_temperature)
                .max(self.config.min_temperature);
        }
    }

    /// Reset temperature to initial value
    pub fn reset_temperature(&mut self) {
        self.current_temperature = self.config.temperature;
        self.step_count = 0;
    }

    /// Get selection history statistics
    pub fn get_history(&self) -> &SelectionHistory {
        &self.selection_history
    }

    /// Get last computed probabilities
    pub fn get_last_probs(&self) -> &[f32] {
        &self.last_probs
    }

    /// Compute KL divergence from uniform distribution
    pub fn kl_from_uniform(&self, probabilities: &[f32]) -> f32 {
        let uniform_prob = 1.0 / probabilities.len() as f32;
        probabilities
            .iter()
            .map(|&p| {
                if p > 1e-8 {
                    p * (p / uniform_prob).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// UCB (Upper Confidence Bound) enhanced softmax selection
/// Combines softmax with exploration bonus based on action counts
pub struct UcbSoftmaxSelection {
    /// Base softmax selector
    softmax: SoftmaxSelection,
    /// UCB exploration constant
    exploration_constant: f32,
    /// Action selection counts for UCB
    action_counts: Vec<u64>,
    /// Total selections
    total_count: u64,
}

impl UcbSoftmaxSelection {
    pub fn new(num_actions: usize, exploration_constant: f32) -> Self {
        Self {
            softmax: SoftmaxSelection::new(),
            exploration_constant,
            action_counts: vec![0; num_actions],
            total_count: 0,
        }
    }

    /// Select action with UCB exploration bonus
    pub fn select(&mut self, action_values: &[f32]) -> SelectionResult {
        // Compute UCB exploration bonus
        let exploration_bonus: Vec<f32> = self
            .action_counts
            .iter()
            .map(|&count| {
                if count == 0 {
                    f32::INFINITY // Ensure unvisited actions are selected
                } else {
                    self.exploration_constant
                        * ((self.total_count as f32).ln() / count as f32).sqrt()
                }
            })
            .collect();

        // Handle infinity values by using very large number
        let exploration_bonus: Vec<f32> = exploration_bonus
            .iter()
            .map(|&b| if b.is_infinite() { 1000.0 } else { b })
            .collect();

        let result = self
            .softmax
            .select_with_bonus(action_values, &exploration_bonus);

        // Update counts
        self.action_counts[result.action] += 1;
        self.total_count += 1;

        result
    }

    /// Reset action counts
    pub fn reset(&mut self) {
        for count in &mut self.action_counts {
            *count = 0;
        }
        self.total_count = 0;
        self.softmax.reset_temperature();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_creation() {
        let selector = SoftmaxSelection::new();
        assert_eq!(selector.temperature(), 1.0);
    }

    #[test]
    fn test_probability_computation() {
        let mut selector = SoftmaxSelection::with_config(SoftmaxConfig {
            temperature: 1.0,
            ..Default::default()
        });

        let values = vec![1.0, 2.0, 3.0];
        let probs = selector.compute_probabilities(&values);

        assert_eq!(probs.len(), 3);

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Higher value should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_temperature_effect() {
        // Low temperature - more greedy
        let mut selector_low = SoftmaxSelection::with_config(SoftmaxConfig {
            temperature: 0.1,
            ..Default::default()
        });

        // High temperature - more uniform
        let mut selector_high = SoftmaxSelection::with_config(SoftmaxConfig {
            temperature: 10.0,
            ..Default::default()
        });

        let values = vec![1.0, 5.0, 2.0];

        let probs_low = selector_low.compute_probabilities(&values);
        let probs_high = selector_high.compute_probabilities(&values);

        // Max probability should be higher with lower temperature
        let max_low = probs_low.iter().cloned().fold(0.0, f32::max);
        let max_high = probs_high.iter().cloned().fold(0.0, f32::max);

        assert!(max_low > max_high);

        // Entropy should be higher with higher temperature
        let entropy_low = selector_low.compute_entropy(&probs_low);
        let entropy_high = selector_high.compute_entropy(&probs_high);

        assert!(entropy_high > entropy_low);
    }

    #[test]
    fn test_action_selection() {
        let mut selector = SoftmaxSelection::new();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = selector.select(&values);

        assert!(result.action < 5);
        assert!(result.probability > 0.0);
        assert!(result.entropy >= 0.0);
    }

    #[test]
    fn test_temperature_decay() {
        let mut selector = SoftmaxSelection::with_config(SoftmaxConfig {
            temperature: 1.0,
            temperature_decay: 0.9,
            target_temperature: 0.1,
            ..Default::default()
        });

        let initial_temp = selector.temperature();
        let values = vec![1.0, 2.0, 3.0];

        // Make several selections
        for _ in 0..10 {
            selector.select(&values);
        }

        // Temperature should have decreased
        assert!(selector.temperature() < initial_temp);
    }

    #[test]
    fn test_entropy_bounds() {
        let selector = SoftmaxSelection::new();

        // Uniform distribution
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let uniform_entropy = selector.compute_entropy(&uniform);
        let max_entropy = selector.max_entropy(4);

        assert!((uniform_entropy - max_entropy).abs() < 0.001);

        // Deterministic distribution
        let deterministic = vec![1.0, 0.0, 0.0, 0.0];
        let det_entropy = selector.compute_entropy(&deterministic);

        assert!(det_entropy < 0.001);
    }

    #[test]
    fn test_normalized_entropy() {
        let selector = SoftmaxSelection::new();

        let uniform = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let norm_entropy = selector.normalized_entropy(&uniform);

        assert!((norm_entropy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_selection_history() {
        let mut selector = SoftmaxSelection::new();
        let values = vec![1.0, 2.0, 3.0];

        for _ in 0..100 {
            selector.select(&values);
        }

        let history = selector.get_history();
        assert_eq!(history.total_selections, 100);

        // Higher value action should be selected more often
        let freqs = history.get_frequencies();
        assert!(freqs[2] >= freqs[0]); // Action 2 (highest value) should be >= action 0
    }

    #[test]
    fn test_ucb_softmax() {
        let mut selector = UcbSoftmaxSelection::new(3, 2.0);

        let values = vec![0.0, 0.0, 0.0]; // Equal values

        // First 3 selections should try all actions (UCB bonus for unvisited)
        let mut actions_seen = std::collections::HashSet::new();
        for _ in 0..3 {
            let result = selector.select(&values);
            actions_seen.insert(result.action);
        }

        assert_eq!(actions_seen.len(), 3);
    }

    #[test]
    fn test_kl_divergence() {
        let selector = SoftmaxSelection::new();

        // Uniform distribution - KL from uniform should be 0
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let kl = selector.kl_from_uniform(&uniform);
        assert!(kl.abs() < 0.001);

        // Peaked distribution - KL from uniform should be > 0
        let peaked = vec![0.7, 0.1, 0.1, 0.1];
        let kl_peaked = selector.kl_from_uniform(&peaked);
        assert!(kl_peaked > 0.0);
    }

    #[test]
    fn test_exploration_bonus() {
        let mut selector = SoftmaxSelection::new();

        let values = vec![1.0, 1.0, 1.0];
        let bonus = vec![0.0, 0.0, 10.0]; // Large bonus for action 2

        let result = selector.select_with_bonus(&values, &bonus);

        // Action 2 should have highest probability
        assert!(result.probabilities[2] > result.probabilities[0]);
        assert!(result.probabilities[2] > result.probabilities[1]);
    }

    #[test]
    fn test_equal_values() {
        let mut selector = SoftmaxSelection::new();

        let values = vec![1.0, 1.0, 1.0, 1.0];
        let probs = selector.compute_probabilities(&values);

        // All probabilities should be equal for equal values
        for i in 1..probs.len() {
            assert!((probs[i] - probs[0]).abs() < 0.001);
        }
    }

    #[test]
    fn test_negative_values() {
        let mut selector = SoftmaxSelection::new();

        let values = vec![-2.0, -1.0, 0.0, 1.0];
        let probs = selector.compute_probabilities(&values);

        // Should still work with negative values
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Order should be preserved
        assert!(probs[3] > probs[2]);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
}
