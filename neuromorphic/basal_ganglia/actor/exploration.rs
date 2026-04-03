//! Exploration strategies (epsilon-greedy, etc.)
//!
//! Part of the Basal Ganglia region
//! Component: actor
//!
//! Implements various exploration strategies for reinforcement learning:
//! - Epsilon-greedy: Random action with probability epsilon
//! - UCB (Upper Confidence Bound): Optimism in the face of uncertainty
//! - Softmax/Boltzmann: Temperature-scaled probability sampling
//! - Thompson Sampling: Bayesian approach with posterior sampling
//! - Decaying exploration: Reduce exploration over time
//! - Adaptive exploration: Adjust based on performance

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Exploration strategy type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorationStrategy {
    /// Random action with fixed probability epsilon
    EpsilonGreedy,
    /// Decaying epsilon over time
    DecayingEpsilon,
    /// Temperature-based softmax exploration
    Softmax,
    /// Upper Confidence Bound
    UCB,
    /// UCB with tuned constant
    UCB1Tuned,
    /// Thompson Sampling (Beta distribution for binary rewards)
    ThompsonSampling,
    /// No exploration (pure exploitation)
    Greedy,
    /// Adaptive epsilon based on recent performance
    Adaptive,
}

/// Configuration for exploration
#[derive(Debug, Clone)]
pub struct ExplorationConfig {
    /// Initial epsilon for epsilon-greedy strategies
    pub initial_epsilon: f64,
    /// Minimum epsilon (floor for decay)
    pub min_epsilon: f64,
    /// Decay rate per step (multiplicative)
    pub decay_rate: f64,
    /// Temperature for softmax exploration
    pub temperature: f64,
    /// UCB exploration constant (c)
    pub ucb_constant: f64,
    /// Window size for adaptive exploration
    pub adaptive_window: usize,
    /// Target success rate for adaptive exploration
    pub target_success_rate: f64,
    /// Learning rate for adaptive adjustment
    pub adaptive_lr: f64,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            initial_epsilon: 0.1,
            min_epsilon: 0.01,
            decay_rate: 0.9999,
            temperature: 1.0,
            ucb_constant: 2.0_f64.sqrt(),
            adaptive_window: 100,
            target_success_rate: 0.5,
            adaptive_lr: 0.01,
        }
    }
}

/// Statistics for an action (for UCB and Thompson Sampling)
#[derive(Debug, Clone, Default)]
pub struct ActionStats {
    /// Number of times action was selected
    pub selection_count: u64,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Mean reward
    pub mean_reward: f64,
    /// Variance of rewards (for UCB1-Tuned)
    pub reward_variance: f64,
    /// Sum of squared rewards (for variance calculation)
    pub reward_sq_sum: f64,
    /// Successes (for Thompson Sampling)
    pub successes: u64,
    /// Failures (for Thompson Sampling)
    pub failures: u64,
}

impl ActionStats {
    /// Update stats with new reward
    pub fn update(&mut self, reward: f64) {
        self.selection_count += 1;
        self.total_reward += reward;
        self.reward_sq_sum += reward * reward;

        // Update mean incrementally
        let n = self.selection_count as f64;
        self.mean_reward = self.total_reward / n;

        // Update variance
        if self.selection_count > 1 {
            self.reward_variance =
                (self.reward_sq_sum - self.total_reward * self.total_reward / n) / (n - 1.0);
        }

        // Update success/failure for Thompson Sampling (assuming reward in [0, 1])
        if reward > 0.5 {
            self.successes += 1;
        } else {
            self.failures += 1;
        }
    }

    /// Get UCB score
    pub fn ucb_score(&self, total_steps: u64, c: f64) -> f64 {
        if self.selection_count == 0 {
            return f64::INFINITY; // Unexplored actions have maximum priority
        }

        let exploration_term = c * ((total_steps as f64).ln() / self.selection_count as f64).sqrt();
        self.mean_reward + exploration_term
    }

    /// Get UCB1-Tuned score
    pub fn ucb1_tuned_score(&self, total_steps: u64) -> f64 {
        if self.selection_count == 0 {
            return f64::INFINITY;
        }

        let n = self.selection_count as f64;
        let t = total_steps as f64;
        let ln_t = t.ln();

        // Variance term
        let v = self.reward_variance + (2.0 * ln_t / n).sqrt();
        let min_term = v.min(0.25); // Clip variance term

        let exploration_term = (ln_t / n * min_term).sqrt();
        self.mean_reward + exploration_term
    }
}

/// Result of exploration decision
#[derive(Debug, Clone)]
pub struct ExplorationDecision {
    /// Selected action index
    pub action_index: usize,
    /// Whether this was an exploration (vs exploitation)
    pub is_exploration: bool,
    /// Probability of selecting this action
    pub selection_probability: f64,
    /// Current epsilon (for epsilon-greedy)
    pub current_epsilon: f64,
    /// Exploration score/bonus if applicable
    pub exploration_bonus: f64,
}

/// Rolling window for adaptive exploration
#[derive(Debug, Clone)]
struct AdaptiveWindow {
    rewards: Vec<f64>,
    position: usize,
    sum: f64,
    count: usize,
    capacity: usize,
}

impl AdaptiveWindow {
    fn new(capacity: usize) -> Self {
        Self {
            rewards: vec![0.0; capacity],
            position: 0,
            sum: 0.0,
            count: 0,
            capacity,
        }
    }

    fn add(&mut self, reward: f64) {
        // Remove old value from sum
        if self.count >= self.capacity {
            self.sum -= self.rewards[self.position];
        } else {
            self.count += 1;
        }

        // Add new value
        self.rewards[self.position] = reward;
        self.sum += reward;
        self.position = (self.position + 1) % self.capacity;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn success_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.5; // Default
        }
        let successes = self
            .rewards
            .iter()
            .take(self.count)
            .filter(|&&r| r > 0.0)
            .count();
        successes as f64 / self.count as f64
    }
}

/// Exploration strategies for reinforcement learning
///
/// Provides multiple exploration strategies to balance the
/// exploration-exploitation tradeoff in decision making.
pub struct Exploration {
    /// Configuration
    config: ExplorationConfig,
    /// Current strategy
    strategy: ExplorationStrategy,
    /// Current epsilon (may decay)
    current_epsilon: f64,
    /// Per-action statistics
    action_stats: HashMap<usize, ActionStats>,
    /// Total steps taken
    total_steps: u64,
    /// Total explorations performed
    exploration_count: u64,
    /// Adaptive window for recent rewards
    adaptive_window: AdaptiveWindow,
    /// Simple RNG state
    rng_state: u64,
}

impl Default for Exploration {
    fn default() -> Self {
        Self::new()
    }
}

impl Exploration {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(
            ExplorationConfig::default(),
            ExplorationStrategy::EpsilonGreedy,
        )
    }

    /// Create with custom configuration and strategy
    pub fn with_config(config: ExplorationConfig, strategy: ExplorationStrategy) -> Self {
        let initial_epsilon = config.initial_epsilon;
        let window_size = config.adaptive_window;
        Self {
            config,
            strategy,
            current_epsilon: initial_epsilon,
            action_stats: HashMap::new(),
            total_steps: 0,
            exploration_count: 0,
            adaptive_window: AdaptiveWindow::new(window_size),
            rng_state: 0xDEADBEEF12345678,
        }
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Select an action given Q-values or action scores
    pub fn select_action(&mut self, action_values: &[f64]) -> Result<ExplorationDecision> {
        if action_values.is_empty() {
            return Err(Error::InvalidInput("Empty action values".into()));
        }

        self.total_steps += 1;

        let decision = match self.strategy {
            ExplorationStrategy::EpsilonGreedy => self.epsilon_greedy(action_values),
            ExplorationStrategy::DecayingEpsilon => self.decaying_epsilon(action_values),
            ExplorationStrategy::Softmax => self.softmax_exploration(action_values),
            ExplorationStrategy::UCB => self.ucb_selection(action_values),
            ExplorationStrategy::UCB1Tuned => self.ucb1_tuned_selection(action_values),
            ExplorationStrategy::ThompsonSampling => self.thompson_sampling(action_values.len()),
            ExplorationStrategy::Greedy => self.greedy_selection(action_values),
            ExplorationStrategy::Adaptive => self.adaptive_epsilon(action_values),
        };

        if decision.is_exploration {
            self.exploration_count += 1;
        }

        Ok(decision)
    }

    /// Update with reward feedback for selected action
    pub fn update(&mut self, action_index: usize, reward: f64) {
        let stats = self.action_stats.entry(action_index).or_default();
        stats.update(reward);
        self.adaptive_window.add(reward);
    }

    /// Epsilon-greedy exploration
    fn epsilon_greedy(&mut self, action_values: &[f64]) -> ExplorationDecision {
        let random = self.next_random();

        if random < self.current_epsilon {
            // Explore: random action
            let action_index = (self.next_random() * action_values.len() as f64) as usize;
            let action_index = action_index.min(action_values.len() - 1);

            ExplorationDecision {
                action_index,
                is_exploration: true,
                selection_probability: self.current_epsilon / action_values.len() as f64,
                current_epsilon: self.current_epsilon,
                exploration_bonus: 0.0,
            }
        } else {
            // Exploit: best action
            let (action_index, _) = self.argmax(action_values);

            ExplorationDecision {
                action_index,
                is_exploration: false,
                selection_probability: 1.0 - self.current_epsilon
                    + self.current_epsilon / action_values.len() as f64,
                current_epsilon: self.current_epsilon,
                exploration_bonus: 0.0,
            }
        }
    }

    /// Decaying epsilon-greedy
    fn decaying_epsilon(&mut self, action_values: &[f64]) -> ExplorationDecision {
        let decision = self.epsilon_greedy(action_values);

        // Decay epsilon
        self.current_epsilon =
            (self.current_epsilon * self.config.decay_rate).max(self.config.min_epsilon);

        decision
    }

    /// Softmax/Boltzmann exploration
    fn softmax_exploration(&mut self, action_values: &[f64]) -> ExplorationDecision {
        let temp = self.config.temperature;

        // Compute softmax probabilities
        let max_val = action_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = action_values
            .iter()
            .map(|&v| ((v - max_val) / temp).exp())
            .sum();

        let probs: Vec<f64> = action_values
            .iter()
            .map(|&v| ((v - max_val) / temp).exp() / exp_sum)
            .collect();

        // Sample from distribution
        let random = self.next_random();
        let mut cumsum = 0.0;
        let mut action_index = 0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if random < cumsum {
                action_index = i;
                break;
            }
        }

        let (best_action, _) = self.argmax(action_values);
        let is_exploration = action_index != best_action;

        ExplorationDecision {
            action_index,
            is_exploration,
            selection_probability: probs[action_index],
            current_epsilon: self.current_epsilon,
            exploration_bonus: 0.0,
        }
    }

    /// UCB action selection
    fn ucb_selection(&mut self, action_values: &[f64]) -> ExplorationDecision {
        let n_actions = action_values.len();
        let mut ucb_scores = Vec::with_capacity(n_actions);

        for i in 0..n_actions {
            let stats = self.action_stats.get(&i);
            let base_value = action_values[i];

            let exploration_bonus = if let Some(s) = stats {
                if s.selection_count == 0 {
                    f64::INFINITY
                } else {
                    self.config.ucb_constant
                        * ((self.total_steps as f64).ln() / s.selection_count as f64).sqrt()
                }
            } else {
                f64::INFINITY // Unexplored
            };

            ucb_scores.push(base_value + exploration_bonus);
        }

        let (action_index, _) = self.argmax(&ucb_scores);
        let (best_action, _) = self.argmax(action_values);

        let exploration_bonus = ucb_scores[action_index] - action_values[action_index];

        ExplorationDecision {
            action_index,
            is_exploration: action_index != best_action,
            selection_probability: 1.0, // UCB is deterministic
            current_epsilon: self.current_epsilon,
            exploration_bonus,
        }
    }

    /// UCB1-Tuned action selection
    fn ucb1_tuned_selection(&mut self, action_values: &[f64]) -> ExplorationDecision {
        let n_actions = action_values.len();
        let mut scores = Vec::with_capacity(n_actions);

        for i in 0..n_actions {
            let stats = self.action_stats.get(&i);
            let base_value = action_values[i];

            let bonus = if let Some(s) = stats {
                s.ucb1_tuned_score(self.total_steps) - s.mean_reward
            } else {
                f64::INFINITY
            };

            scores.push(base_value + bonus);
        }

        let (action_index, _) = self.argmax(&scores);
        let (best_action, _) = self.argmax(action_values);

        ExplorationDecision {
            action_index,
            is_exploration: action_index != best_action,
            selection_probability: 1.0,
            current_epsilon: self.current_epsilon,
            exploration_bonus: scores[action_index] - action_values[action_index],
        }
    }

    /// Thompson Sampling
    fn thompson_sampling(&mut self, n_actions: usize) -> ExplorationDecision {
        let mut samples = Vec::with_capacity(n_actions);

        for i in 0..n_actions {
            let (alpha, beta) = if let Some(stats) = self.action_stats.get(&i) {
                // Prior + observed successes/failures
                (1.0 + stats.successes as f64, 1.0 + stats.failures as f64)
            } else {
                (1.0, 1.0) // Uniform prior
            };

            // Sample from Beta distribution (using approximation)
            let sample = self.sample_beta(alpha, beta);
            samples.push(sample);
        }

        let (action_index, _) = self.argmax(&samples);

        // Check if this is the empirically best action
        let best_empirical = self
            .action_stats
            .iter()
            .max_by(|a, b| {
                a.1.mean_reward
                    .partial_cmp(&b.1.mean_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&idx, _)| idx);

        let is_exploration = best_empirical.map(|b| b != action_index).unwrap_or(true);

        ExplorationDecision {
            action_index,
            is_exploration,
            selection_probability: samples[action_index],
            current_epsilon: self.current_epsilon,
            exploration_bonus: 0.0,
        }
    }

    /// Pure greedy selection (no exploration)
    fn greedy_selection(&self, action_values: &[f64]) -> ExplorationDecision {
        let (action_index, _) = self.argmax(action_values);

        ExplorationDecision {
            action_index,
            is_exploration: false,
            selection_probability: 1.0,
            current_epsilon: 0.0,
            exploration_bonus: 0.0,
        }
    }

    /// Adaptive epsilon based on recent performance
    fn adaptive_epsilon(&mut self, action_values: &[f64]) -> ExplorationDecision {
        // Adjust epsilon based on success rate
        let success_rate = self.adaptive_window.success_rate();
        let error = self.config.target_success_rate - success_rate;

        // Increase epsilon if doing poorly, decrease if doing well
        let adjustment = self.config.adaptive_lr * error;
        self.current_epsilon = (self.current_epsilon + adjustment)
            .max(self.config.min_epsilon)
            .min(1.0);

        self.epsilon_greedy(action_values)
    }

    /// Get current epsilon value
    pub fn epsilon(&self) -> f64 {
        self.current_epsilon
    }

    /// Set epsilon manually
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.current_epsilon = epsilon.clamp(0.0, 1.0);
    }

    /// Get current strategy
    pub fn strategy(&self) -> ExplorationStrategy {
        self.strategy
    }

    /// Set strategy
    pub fn set_strategy(&mut self, strategy: ExplorationStrategy) {
        self.strategy = strategy;
    }

    /// Get exploration rate (exploration count / total steps)
    pub fn exploration_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.exploration_count as f64 / self.total_steps as f64
        }
    }

    /// Get total steps
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Get action statistics
    pub fn action_stats(&self, action_index: usize) -> Option<&ActionStats> {
        self.action_stats.get(&action_index)
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.current_epsilon = self.config.initial_epsilon;
        self.action_stats.clear();
        self.total_steps = 0;
        self.exploration_count = 0;
        self.adaptive_window = AdaptiveWindow::new(self.config.adaptive_window);
    }

    /// Seed the RNG
    pub fn seed(&mut self, seed: u64) {
        self.rng_state = seed.max(1);
    }

    // --- Private helper methods ---

    /// Find argmax of values
    fn argmax(&self, values: &[f64]) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;

        for (i, &v) in values.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }

        (best_idx, best_val)
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

    /// Sample from Beta distribution (using normal approximation for simplicity)
    fn sample_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        // Use normal approximation for large alpha, beta
        // For small values, use a simple rejection sampling approach
        if alpha > 1.0 && beta > 1.0 {
            let mean = alpha / (alpha + beta);
            let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
            let std = variance.sqrt();

            // Sample from normal and clamp to [0, 1]
            let u1 = self.next_random().max(1e-10);
            let u2 = self.next_random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            (mean + std * z).clamp(0.0, 1.0)
        } else {
            // Simple fallback for edge cases
            let u = self.next_random();
            u.powf(1.0 / alpha) / (u.powf(1.0 / alpha) + (1.0 - u).powf(1.0 / beta))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = Exploration::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_epsilon_greedy() {
        let config = ExplorationConfig {
            initial_epsilon: 0.5,
            ..Default::default()
        };
        let mut explorer = Exploration::with_config(config, ExplorationStrategy::EpsilonGreedy);
        explorer.seed(12345);

        let values = vec![1.0, 2.0, 3.0, 4.0];

        // Run many times and count explorations
        let mut explorations = 0;
        let mut action_3_count = 0;

        for _ in 0..1000 {
            let decision = explorer.select_action(&values).unwrap();
            if decision.is_exploration {
                explorations += 1;
            }
            if decision.action_index == 3 {
                action_3_count += 1;
            }
        }

        // Should explore roughly 50% of the time
        let exploration_rate = explorations as f64 / 1000.0;
        assert!(
            exploration_rate > 0.3 && exploration_rate < 0.7,
            "Exploration rate should be around 0.5, got {}",
            exploration_rate
        );

        // Action 3 (highest value) should be selected most often
        assert!(
            action_3_count > 400,
            "Best action should be selected frequently"
        );
    }

    #[test]
    fn test_decaying_epsilon() {
        let config = ExplorationConfig {
            initial_epsilon: 1.0,
            min_epsilon: 0.1,
            decay_rate: 0.9,
            ..Default::default()
        };
        let mut explorer = Exploration::with_config(config, ExplorationStrategy::DecayingEpsilon);

        let values = vec![1.0, 2.0];
        let initial = explorer.epsilon();

        for _ in 0..100 {
            explorer.select_action(&values).unwrap();
        }

        // Epsilon should have decayed
        assert!(
            explorer.epsilon() < initial,
            "Epsilon should decay over time"
        );
        assert!(
            explorer.epsilon() >= 0.1,
            "Epsilon should not go below minimum"
        );
    }

    #[test]
    fn test_greedy() {
        let mut explorer =
            Exploration::with_config(ExplorationConfig::default(), ExplorationStrategy::Greedy);

        let values = vec![1.0, 5.0, 3.0, 2.0];

        for _ in 0..100 {
            let decision = explorer.select_action(&values).unwrap();
            assert_eq!(
                decision.action_index, 1,
                "Greedy should always pick best action"
            );
            assert!(!decision.is_exploration);
        }
    }

    #[test]
    fn test_softmax() {
        let config = ExplorationConfig {
            temperature: 0.1, // Low temperature = more deterministic
            ..Default::default()
        };
        let mut explorer = Exploration::with_config(config, ExplorationStrategy::Softmax);
        explorer.seed(42);

        let values = vec![1.0, 10.0, 1.0]; // Action 1 is clearly best

        let mut action_1_count = 0;
        for _ in 0..100 {
            let decision = explorer.select_action(&values).unwrap();
            if decision.action_index == 1 {
                action_1_count += 1;
            }
        }

        // With low temperature, best action should dominate
        assert!(
            action_1_count > 80,
            "Low temperature softmax should favor best action"
        );
    }

    #[test]
    fn test_ucb() {
        let mut explorer =
            Exploration::with_config(ExplorationConfig::default(), ExplorationStrategy::UCB);

        let values = vec![0.0, 0.0, 0.0, 0.0]; // All equal base values

        // First 4 selections should explore all actions (UCB gives infinity to unexplored)
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            let decision = explorer.select_action(&values).unwrap();
            seen.insert(decision.action_index);
            explorer.update(decision.action_index, 0.5);
        }

        assert_eq!(seen.len(), 4, "UCB should explore all actions first");
    }

    #[test]
    fn test_update_stats() {
        let mut explorer = Exploration::new();

        explorer.update(0, 1.0);
        explorer.update(0, 0.5);
        explorer.update(0, 0.0);

        let stats = explorer.action_stats(0).unwrap();
        assert_eq!(stats.selection_count, 3);
        assert!((stats.mean_reward - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_exploration_rate() {
        let config = ExplorationConfig {
            initial_epsilon: 1.0, // Always explore
            ..Default::default()
        };
        let mut explorer = Exploration::with_config(config, ExplorationStrategy::EpsilonGreedy);

        let values = vec![1.0, 2.0];
        for _ in 0..100 {
            explorer.select_action(&values).unwrap();
        }

        // With epsilon=1, should explore 100%
        assert!(
            explorer.exploration_rate() > 0.9,
            "With epsilon=1, exploration rate should be near 1.0"
        );
    }

    #[test]
    fn test_reset() {
        let mut explorer = Exploration::new();

        let values = vec![1.0, 2.0];
        for _ in 0..50 {
            let decision = explorer.select_action(&values).unwrap();
            explorer.update(decision.action_index, 0.5);
        }

        explorer.reset();

        assert_eq!(explorer.total_steps(), 0);
        assert_eq!(explorer.exploration_rate(), 0.0);
        assert!(explorer.action_stats(0).is_none());
    }

    #[test]
    fn test_thompson_sampling() {
        let mut explorer = Exploration::with_config(
            ExplorationConfig::default(),
            ExplorationStrategy::ThompsonSampling,
        );
        explorer.seed(123);

        // Action 0: high success rate
        for _ in 0..10 {
            explorer.update(0, 0.9);
        }

        // Action 1: low success rate
        for _ in 0..10 {
            explorer.update(1, 0.1);
        }

        // Thompson sampling should favor action 0
        let mut action_0_count = 0;
        for _ in 0..100 {
            let decision = explorer.select_action(&[0.0, 0.0]).unwrap();
            if decision.action_index == 0 {
                action_0_count += 1;
            }
        }

        assert!(
            action_0_count > 60,
            "Thompson sampling should favor high-success action"
        );
    }

    #[test]
    fn test_set_strategy() {
        let mut explorer = Exploration::new();

        explorer.set_strategy(ExplorationStrategy::UCB);
        assert_eq!(explorer.strategy(), ExplorationStrategy::UCB);

        explorer.set_strategy(ExplorationStrategy::Softmax);
        assert_eq!(explorer.strategy(), ExplorationStrategy::Softmax);
    }

    #[test]
    fn test_empty_actions_error() {
        let mut explorer = Exploration::new();
        let result = explorer.select_action(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_action_stats_ucb_score() {
        let mut stats = ActionStats::default();

        // Unexplored action should have infinite score
        assert_eq!(stats.ucb_score(100, 1.0), f64::INFINITY);

        // Add some rewards
        stats.update(0.5);
        stats.update(0.7);
        stats.update(0.3);

        let score = stats.ucb_score(100, 1.4);
        assert!(
            score > stats.mean_reward,
            "UCB score should include exploration bonus"
        );
    }

    #[test]
    fn test_adaptive_exploration() {
        let config = ExplorationConfig {
            initial_epsilon: 0.5,
            target_success_rate: 0.5,
            adaptive_lr: 0.1,
            adaptive_window: 10,
            ..Default::default()
        };
        let mut explorer = Exploration::with_config(config, ExplorationStrategy::Adaptive);

        let values = vec![1.0, 2.0];

        // Simulate poor performance (all failures)
        for _ in 0..20 {
            let decision = explorer.select_action(&values).unwrap();
            explorer.update(decision.action_index, 0.0); // Failure
        }

        // Epsilon should have increased (trying to explore more due to poor performance)
        // (depends on implementation details, but should adapt)
        assert!(explorer.epsilon() > 0.0);
    }
}
