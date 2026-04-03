//! Probabilistic action distribution
//!
//! Part of the Basal Ganglia region
//! Component: actor
//!
//! Implements probabilistic action selection for trading decisions:
//! - Categorical distribution over discrete actions
//! - Temperature-controlled exploration
//! - Action probability weighting by confidence
//! - Policy gradient compatible log probabilities
//! - Action masking for invalid actions

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Trading action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradingAction {
    /// Do nothing
    Hold,
    /// Open or increase long position
    Buy,
    /// Open or increase short position
    Sell,
    /// Close existing long position
    CloseLong,
    /// Close existing short position
    CloseShort,
    /// Reduce position size
    ReducePosition,
    /// Add to winning position
    ScaleIn,
    /// Take partial profits
    TakeProfit,
}

impl TradingAction {
    /// Get all possible actions
    pub fn all() -> &'static [TradingAction] {
        &[
            Self::Hold,
            Self::Buy,
            Self::Sell,
            Self::CloseLong,
            Self::CloseShort,
            Self::ReducePosition,
            Self::ScaleIn,
            Self::TakeProfit,
        ]
    }

    /// Get the action index (for array indexing)
    pub fn index(&self) -> usize {
        match self {
            Self::Hold => 0,
            Self::Buy => 1,
            Self::Sell => 2,
            Self::CloseLong => 3,
            Self::CloseShort => 4,
            Self::ReducePosition => 5,
            Self::ScaleIn => 6,
            Self::TakeProfit => 7,
        }
    }

    /// Get action from index
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Hold),
            1 => Some(Self::Buy),
            2 => Some(Self::Sell),
            3 => Some(Self::CloseLong),
            4 => Some(Self::CloseShort),
            5 => Some(Self::ReducePosition),
            6 => Some(Self::ScaleIn),
            7 => Some(Self::TakeProfit),
            _ => None,
        }
    }

    /// Number of possible actions
    pub fn count() -> usize {
        8
    }

    /// Whether this action opens a new position
    pub fn is_entry(&self) -> bool {
        matches!(self, Self::Buy | Self::Sell)
    }

    /// Whether this action closes or reduces position
    pub fn is_exit(&self) -> bool {
        matches!(
            self,
            Self::CloseLong | Self::CloseShort | Self::ReducePosition | Self::TakeProfit
        )
    }
}

/// Configuration for action distribution
#[derive(Debug, Clone)]
pub struct ActionDistributionConfig {
    /// Temperature for softmax (lower = more deterministic)
    pub temperature: f64,
    /// Minimum probability for any action (prevents zero probs)
    pub min_probability: f64,
    /// Maximum probability for any single action
    pub max_probability: f64,
    /// Default action when all others masked
    pub default_action: TradingAction,
    /// Enable entropy regularization
    pub entropy_regularization: bool,
    /// Target entropy for regularization
    pub target_entropy: f64,
}

impl Default for ActionDistributionConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            min_probability: 0.001,
            max_probability: 0.95,
            default_action: TradingAction::Hold,
            entropy_regularization: true,
            target_entropy: 1.5, // bits
        }
    }
}

/// Result of action sampling
#[derive(Debug, Clone)]
pub struct ActionSample {
    /// Selected action
    pub action: TradingAction,
    /// Probability of selected action
    pub probability: f64,
    /// Log probability (for policy gradients)
    pub log_probability: f64,
    /// Entropy of distribution
    pub entropy: f64,
    /// Whether action was sampled or deterministic
    pub is_stochastic: bool,
}

/// Statistics about action distribution
#[derive(Debug, Clone, Default)]
pub struct DistributionStats {
    /// Total samples drawn
    pub total_samples: u64,
    /// Samples per action
    pub action_counts: HashMap<TradingAction, u64>,
    /// Average entropy over recent samples
    pub avg_entropy: f64,
    /// Average temperature used
    pub avg_temperature: f64,
    /// Number of masked actions on average
    pub avg_masked_actions: f64,
}

/// Probabilistic action distribution for trading decisions
///
/// Implements a categorical distribution over discrete trading actions
/// with support for:
/// - Temperature-controlled exploration
/// - Action masking
/// - Log probabilities for policy gradients
/// - Entropy calculation for exploration bonuses
pub struct ActionDistribution {
    /// Configuration
    config: ActionDistributionConfig,
    /// Current action logits (unnormalized log probabilities)
    logits: Vec<f64>,
    /// Current probabilities (after softmax)
    probabilities: Vec<f64>,
    /// Action mask (true = allowed, false = masked)
    action_mask: Vec<bool>,
    /// Running statistics
    stats: DistributionStats,
    /// Entropy accumulator for averaging
    entropy_sum: f64,
    /// Temperature accumulator
    temperature_sum: f64,
    /// Masked action accumulator
    masked_sum: f64,
    /// Simple RNG state (xorshift)
    rng_state: u64,
}

impl Default for ActionDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionDistribution {
    /// Create a new action distribution with default config
    pub fn new() -> Self {
        Self::with_config(ActionDistributionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ActionDistributionConfig) -> Self {
        let n_actions = TradingAction::count();
        Self {
            config,
            logits: vec![0.0; n_actions],
            probabilities: vec![1.0 / n_actions as f64; n_actions],
            action_mask: vec![true; n_actions],
            stats: DistributionStats::default(),
            entropy_sum: 0.0,
            temperature_sum: 0.0,
            masked_sum: 0.0,
            rng_state: 0xDEADBEEF12345678,
        }
    }

    /// Set action logits (unnormalized scores from policy network)
    pub fn set_logits(&mut self, logits: &[f64]) -> Result<()> {
        if logits.len() != TradingAction::count() {
            return Err(Error::InvalidInput(format!(
                "Expected {} logits, got {}",
                TradingAction::count(),
                logits.len()
            )));
        }

        self.logits.copy_from_slice(logits);
        self.update_probabilities();
        Ok(())
    }

    /// Set action mask (true = allowed, false = masked)
    pub fn set_mask(&mut self, mask: &[bool]) -> Result<()> {
        if mask.len() != TradingAction::count() {
            return Err(Error::InvalidInput(format!(
                "Expected {} mask values, got {}",
                TradingAction::count(),
                mask.len()
            )));
        }

        self.action_mask.copy_from_slice(mask);
        self.update_probabilities();
        Ok(())
    }

    /// Mask a specific action
    pub fn mask_action(&mut self, action: TradingAction) {
        self.action_mask[action.index()] = false;
        self.update_probabilities();
    }

    /// Unmask a specific action
    pub fn unmask_action(&mut self, action: TradingAction) {
        self.action_mask[action.index()] = true;
        self.update_probabilities();
    }

    /// Clear all masks (allow all actions)
    pub fn clear_mask(&mut self) {
        self.action_mask.fill(true);
        self.update_probabilities();
    }

    /// Set temperature for exploration control
    pub fn set_temperature(&mut self, temperature: f64) {
        self.config.temperature = temperature.max(0.01);
        self.update_probabilities();
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.config.temperature
    }

    /// Sample an action from the distribution
    pub fn sample(&mut self) -> ActionSample {
        let random = self.next_random();
        self.sample_with_random(random, true)
    }

    /// Get the most probable action (greedy/deterministic)
    pub fn greedy(&self) -> ActionSample {
        let mut best_idx = 0;
        let mut best_prob = 0.0;

        for (i, &prob) in self.probabilities.iter().enumerate() {
            if self.action_mask[i] && prob > best_prob {
                best_prob = prob;
                best_idx = i;
            }
        }

        let action = TradingAction::from_index(best_idx).unwrap_or(self.config.default_action);
        let entropy = self.calculate_entropy();

        ActionSample {
            action,
            probability: best_prob,
            log_probability: best_prob.max(1e-10).ln(),
            entropy,
            is_stochastic: false,
        }
    }

    /// Sample with epsilon-greedy strategy
    pub fn epsilon_greedy(&mut self, epsilon: f64) -> ActionSample {
        let random = self.next_random();
        if random < epsilon {
            // Random action from allowed actions
            let allowed: Vec<_> = TradingAction::all()
                .iter()
                .filter(|a| self.action_mask[a.index()])
                .collect();

            if allowed.is_empty() {
                return self.greedy();
            }

            let idx = (random * allowed.len() as f64) as usize % allowed.len();
            let action = *allowed[idx];
            let prob = self.probabilities[action.index()];

            ActionSample {
                action,
                probability: prob,
                log_probability: prob.max(1e-10).ln(),
                entropy: self.calculate_entropy(),
                is_stochastic: true,
            }
        } else {
            self.greedy()
        }
    }

    /// Get probability of a specific action
    pub fn probability(&self, action: TradingAction) -> f64 {
        self.probabilities[action.index()]
    }

    /// Get log probability of a specific action
    pub fn log_probability(&self, action: TradingAction) -> f64 {
        self.probabilities[action.index()].max(1e-10).ln()
    }

    /// Get all probabilities
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Get all logits
    pub fn logits(&self) -> &[f64] {
        &self.logits
    }

    /// Calculate entropy of current distribution
    pub fn calculate_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        for (i, &prob) in self.probabilities.iter().enumerate() {
            if self.action_mask[i] && prob > 1e-10 {
                entropy -= prob * prob.ln();
            }
        }
        entropy
    }

    /// Get KL divergence from uniform distribution
    pub fn kl_from_uniform(&self) -> f64 {
        let n_allowed = self.action_mask.iter().filter(|&&m| m).count();
        if n_allowed == 0 {
            return 0.0;
        }

        let uniform_prob = 1.0 / n_allowed as f64;
        let mut kl = 0.0;

        for (i, &prob) in self.probabilities.iter().enumerate() {
            if self.action_mask[i] && prob > 1e-10 {
                kl += prob * (prob / uniform_prob).ln();
            }
        }

        kl
    }

    /// Get statistics
    pub fn stats(&self) -> &DistributionStats {
        &self.stats
    }

    /// Get action counts as a vector
    pub fn action_distribution_vector(&self) -> Vec<(TradingAction, f64)> {
        let total = self.stats.total_samples.max(1) as f64;
        TradingAction::all()
            .iter()
            .map(|&action| {
                let count = self.stats.action_counts.get(&action).copied().unwrap_or(0);
                (action, count as f64 / total)
            })
            .collect()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DistributionStats::default();
        self.entropy_sum = 0.0;
        self.temperature_sum = 0.0;
        self.masked_sum = 0.0;
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // --- Private methods ---

    /// Update probabilities from logits using softmax with temperature and masking
    fn update_probabilities(&mut self) {
        let temp = self.config.temperature;

        // Apply temperature scaling
        let scaled: Vec<f64> = self.logits.iter().map(|&l| l / temp).collect();

        // Find max for numerical stability (only among unmasked)
        let max_logit = scaled
            .iter()
            .enumerate()
            .filter(|(i, _)| self.action_mask[*i])
            .map(|(_, &l)| l)
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute exp and sum (masked actions get 0)
        let mut sum = 0.0;
        let mut exps = vec![0.0; TradingAction::count()];

        for (i, &logit) in scaled.iter().enumerate() {
            if self.action_mask[i] {
                let exp_val = (logit - max_logit).exp();
                exps[i] = exp_val;
                sum += exp_val;
            }
        }

        // Normalize to probabilities
        if sum > 0.0 {
            for (i, prob) in self.probabilities.iter_mut().enumerate() {
                if self.action_mask[i] {
                    *prob = (exps[i] / sum)
                        .max(self.config.min_probability)
                        .min(self.config.max_probability);
                } else {
                    *prob = 0.0;
                }
            }

            // Renormalize after clamping
            let new_sum: f64 = self.probabilities.iter().sum();
            if new_sum > 0.0 {
                for prob in &mut self.probabilities {
                    *prob /= new_sum;
                }
            }
        } else {
            // All masked - use default
            self.probabilities.fill(0.0);
            self.probabilities[self.config.default_action.index()] = 1.0;
        }
    }

    /// Sample action given a random value [0, 1)
    fn sample_with_random(&mut self, random: f64, is_stochastic: bool) -> ActionSample {
        // Cumulative sampling
        let mut cumsum = 0.0;
        let mut selected_idx = self.config.default_action.index();

        for (i, &prob) in self.probabilities.iter().enumerate() {
            cumsum += prob;
            if random < cumsum {
                selected_idx = i;
                break;
            }
        }

        let action = TradingAction::from_index(selected_idx).unwrap_or(self.config.default_action);
        let prob = self.probabilities[selected_idx];
        let entropy = self.calculate_entropy();

        // Update statistics
        self.stats.total_samples += 1;
        *self.stats.action_counts.entry(action).or_insert(0) += 1;

        self.entropy_sum += entropy;
        self.temperature_sum += self.config.temperature;
        self.masked_sum += self.action_mask.iter().filter(|&&m| !m).count() as f64;

        let n = self.stats.total_samples as f64;
        self.stats.avg_entropy = self.entropy_sum / n;
        self.stats.avg_temperature = self.temperature_sum / n;
        self.stats.avg_masked_actions = self.masked_sum / n;

        ActionSample {
            action,
            probability: prob,
            log_probability: prob.max(1e-10).ln(),
            entropy,
            is_stochastic,
        }
    }

    /// Simple xorshift64 PRNG
    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;

        // Convert to [0, 1)
        (x as f64) / (u64::MAX as f64)
    }

    /// Seed the RNG
    pub fn seed(&mut self, seed: u64) {
        self.rng_state = seed.max(1); // Avoid 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let dist = ActionDistribution::new();
        assert_eq!(dist.probabilities().len(), TradingAction::count());
        assert!(dist.calculate_entropy() > 0.0);
    }

    #[test]
    fn test_set_logits() {
        let mut dist = ActionDistribution::new();

        // Strong preference for Buy
        let mut logits = vec![0.0; TradingAction::count()];
        logits[TradingAction::Buy.index()] = 10.0;

        dist.set_logits(&logits).unwrap();

        assert!(dist.probability(TradingAction::Buy) > 0.9);
    }

    #[test]
    fn test_action_masking() {
        let mut dist = ActionDistribution::new();

        // Equal logits
        let logits = vec![1.0; TradingAction::count()];
        dist.set_logits(&logits).unwrap();

        // Mask all except Hold and Buy
        for action in TradingAction::all() {
            if *action != TradingAction::Hold && *action != TradingAction::Buy {
                dist.mask_action(*action);
            }
        }

        // Masked actions should have 0 probability
        assert_eq!(dist.probability(TradingAction::Sell), 0.0);
        assert_eq!(dist.probability(TradingAction::CloseLong), 0.0);

        // Unmasked should share probability
        let hold_prob = dist.probability(TradingAction::Hold);
        let buy_prob = dist.probability(TradingAction::Buy);
        assert!((hold_prob - buy_prob).abs() < 0.01);
        assert!((hold_prob + buy_prob - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sample() {
        let mut dist = ActionDistribution::new();

        // Strong preference for Hold
        let mut logits = vec![0.0; TradingAction::count()];
        logits[TradingAction::Hold.index()] = 100.0;
        dist.set_logits(&logits).unwrap();

        // Should almost always sample Hold
        let mut hold_count = 0;
        for _ in 0..100 {
            let sample = dist.sample();
            if sample.action == TradingAction::Hold {
                hold_count += 1;
            }
        }

        assert!(hold_count > 90, "Hold should be selected most of the time");
    }

    #[test]
    fn test_greedy() {
        let mut dist = ActionDistribution::new();

        let mut logits = vec![0.0; TradingAction::count()];
        logits[TradingAction::Sell.index()] = 5.0;
        dist.set_logits(&logits).unwrap();

        let sample = dist.greedy();
        assert_eq!(sample.action, TradingAction::Sell);
        assert!(!sample.is_stochastic);
    }

    #[test]
    fn test_temperature() {
        let mut dist = ActionDistribution::new();

        let mut logits = vec![0.0; TradingAction::count()];
        logits[TradingAction::Buy.index()] = 2.0;
        logits[TradingAction::Sell.index()] = 1.0;

        // Low temperature = more deterministic
        dist.set_temperature(0.1);
        dist.set_logits(&logits).unwrap();
        let low_temp_entropy = dist.calculate_entropy();

        // High temperature = more uniform
        dist.set_temperature(10.0);
        dist.set_logits(&logits).unwrap();
        let high_temp_entropy = dist.calculate_entropy();

        assert!(
            high_temp_entropy > low_temp_entropy,
            "Higher temperature should increase entropy"
        );
    }

    #[test]
    fn test_entropy() {
        let mut dist = ActionDistribution::new();

        // Uniform distribution has maximum entropy
        let uniform_logits = vec![1.0; TradingAction::count()];
        dist.set_logits(&uniform_logits).unwrap();
        let uniform_entropy = dist.calculate_entropy();

        // Peaked distribution has lower entropy
        let mut peaked_logits = vec![0.0; TradingAction::count()];
        peaked_logits[0] = 10.0;
        dist.set_logits(&peaked_logits).unwrap();
        let peaked_entropy = dist.calculate_entropy();

        assert!(uniform_entropy > peaked_entropy);
    }

    #[test]
    fn test_log_probability() {
        let mut dist = ActionDistribution::new();

        let logits = vec![1.0; TradingAction::count()];
        dist.set_logits(&logits).unwrap();

        for action in TradingAction::all() {
            let prob = dist.probability(*action);
            let log_prob = dist.log_probability(*action);
            assert!((log_prob - prob.ln()).abs() < 1e-6);
        }
    }

    #[test]
    fn test_epsilon_greedy() {
        let mut dist = ActionDistribution::new();

        let mut logits = vec![0.0; TradingAction::count()];
        logits[TradingAction::Buy.index()] = 100.0;
        dist.set_logits(&logits).unwrap();

        // With epsilon = 0, should always be greedy
        dist.seed(12345);
        let mut all_buy = true;
        for _ in 0..50 {
            let sample = dist.epsilon_greedy(0.0);
            if sample.action != TradingAction::Buy {
                all_buy = false;
            }
        }
        // Not strictly guaranteed but very likely
        assert!(all_buy, "epsilon=0 should mostly select greedy action");

        // With epsilon = 1, should explore
        dist.seed(12345);
        let mut actions_seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let sample = dist.epsilon_greedy(1.0);
            actions_seen.insert(sample.action);
        }
        assert!(
            actions_seen.len() > 1,
            "epsilon=1 should explore multiple actions"
        );
    }

    #[test]
    fn test_action_methods() {
        assert!(TradingAction::Buy.is_entry());
        assert!(TradingAction::Sell.is_entry());
        assert!(!TradingAction::Hold.is_entry());

        assert!(TradingAction::CloseLong.is_exit());
        assert!(TradingAction::CloseShort.is_exit());
        assert!(TradingAction::TakeProfit.is_exit());
        assert!(!TradingAction::Buy.is_exit());
    }

    #[test]
    fn test_action_indexing() {
        for action in TradingAction::all() {
            let idx = action.index();
            let recovered = TradingAction::from_index(idx);
            assert_eq!(Some(*action), recovered);
        }

        assert_eq!(TradingAction::from_index(100), None);
    }

    #[test]
    fn test_kl_divergence() {
        let mut dist = ActionDistribution::new();

        // Uniform should have KL = 0
        let uniform_logits = vec![1.0; TradingAction::count()];
        dist.set_logits(&uniform_logits).unwrap();
        let kl_uniform = dist.kl_from_uniform();
        assert!(kl_uniform.abs() < 0.01);

        // Peaked should have high KL
        let mut peaked_logits = vec![0.0; TradingAction::count()];
        peaked_logits[0] = 10.0;
        dist.set_logits(&peaked_logits).unwrap();
        let kl_peaked = dist.kl_from_uniform();
        assert!(kl_peaked > kl_uniform);
    }

    #[test]
    fn test_stats() {
        let mut dist = ActionDistribution::new();

        // Sample multiple times
        for _ in 0..100 {
            dist.sample();
        }

        let stats = dist.stats();
        assert_eq!(stats.total_samples, 100);
        assert!(stats.avg_entropy > 0.0);

        // Action counts should sum to total
        let total: u64 = stats.action_counts.values().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_reset_stats() {
        let mut dist = ActionDistribution::new();

        for _ in 0..50 {
            dist.sample();
        }

        dist.reset_stats();

        assert_eq!(dist.stats().total_samples, 0);
        assert!(dist.stats().action_counts.is_empty());
    }

    #[test]
    fn test_process_compatibility() {
        let dist = ActionDistribution::new();
        assert!(dist.process().is_ok());
    }

    #[test]
    fn test_invalid_logits_length() {
        let mut dist = ActionDistribution::new();
        let result = dist.set_logits(&[1.0, 2.0]); // Wrong length
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_mask_length() {
        let mut dist = ActionDistribution::new();
        let result = dist.set_mask(&[true, false]); // Wrong length
        assert!(result.is_err());
    }

    #[test]
    fn test_all_masked_fallback() {
        let mut dist = ActionDistribution::new();

        // Mask all actions
        for action in TradingAction::all() {
            dist.mask_action(*action);
        }

        // Should fall back to default action
        let sample = dist.greedy();
        assert_eq!(sample.action, TradingAction::Hold); // Default
    }

    #[test]
    fn test_clear_mask() {
        let mut dist = ActionDistribution::new();

        // Mask some actions
        dist.mask_action(TradingAction::Buy);
        dist.mask_action(TradingAction::Sell);

        assert_eq!(dist.probability(TradingAction::Buy), 0.0);

        // Clear mask
        dist.clear_mask();

        // All actions should be allowed now
        for action in TradingAction::all() {
            assert!(dist.probability(*action) > 0.0);
        }
    }
}
