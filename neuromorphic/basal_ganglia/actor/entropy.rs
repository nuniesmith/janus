//! Entropy bonus for exploration
//!
//! Part of the Basal Ganglia region
//! Component: actor
//!
//! Implements entropy-based exploration bonuses for reinforcement learning:
//! - Shannon entropy calculation for action distributions
//! - Entropy bonus scheduling (annealing)
//! - Maximum entropy regularization
//! - Adaptive entropy targeting
//! - Multi-dimensional entropy for complex action spaces

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for entropy bonus calculation
#[derive(Debug, Clone)]
pub struct EntropyConfig {
    /// Initial entropy coefficient
    pub initial_coefficient: f64,
    /// Minimum entropy coefficient (after annealing)
    pub min_coefficient: f64,
    /// Annealing rate per step (multiplicative decay)
    pub annealing_rate: f64,
    /// Target entropy (for adaptive entropy)
    pub target_entropy: Option<f64>,
    /// Learning rate for entropy coefficient adjustment
    pub entropy_lr: f64,
    /// Window size for entropy statistics
    pub stats_window: usize,
    /// Minimum probability for numerical stability
    pub min_prob: f64,
    /// Enable automatic entropy tuning
    pub auto_tune: bool,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            initial_coefficient: 0.01,
            min_coefficient: 0.001,
            annealing_rate: 0.9999,
            target_entropy: None,
            entropy_lr: 0.001,
            stats_window: 1000,
            min_prob: 1e-8,
            auto_tune: false,
        }
    }
}

/// Entropy calculation result
#[derive(Debug, Clone)]
pub struct EntropyResult {
    /// Raw entropy value (in nats)
    pub entropy: f64,
    /// Normalized entropy (0.0 - 1.0)
    pub normalized: f64,
    /// Entropy bonus (entropy * coefficient)
    pub bonus: f64,
    /// Current coefficient being used
    pub coefficient: f64,
    /// Maximum possible entropy for this distribution
    pub max_entropy: f64,
}

/// Statistics about entropy over time
#[derive(Debug, Clone, Default)]
pub struct EntropyStats {
    /// Mean entropy over window
    pub mean_entropy: f64,
    /// Standard deviation of entropy
    pub std_entropy: f64,
    /// Minimum entropy observed
    pub min_entropy: f64,
    /// Maximum entropy observed
    pub max_entropy: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Current coefficient
    pub current_coefficient: f64,
}

/// Entropy type for different scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyType {
    /// Shannon entropy (standard)
    Shannon,
    /// Renyi entropy with parameter alpha
    Renyi,
    /// Tsallis entropy with parameter q
    Tsallis,
    /// Gini-Simpson index
    GiniSimpson,
}

/// Entropy bonus calculator for exploration in reinforcement learning
///
/// Provides entropy bonuses to encourage exploration and prevent
/// premature convergence to deterministic policies.
pub struct Entropy {
    config: EntropyConfig,
    /// Current entropy coefficient (may change with annealing/adaptation)
    current_coefficient: f64,
    /// Log of current coefficient (for SAC-style auto-tuning)
    log_coefficient: f64,
    /// Rolling window of entropy values
    entropy_history: VecDeque<f64>,
    /// Total steps processed
    total_steps: u64,
    /// Running sum for mean calculation
    running_sum: f64,
    /// Running sum of squares for std calculation
    running_sum_sq: f64,
    /// Minimum observed entropy
    min_observed: f64,
    /// Maximum observed entropy
    max_observed: f64,
}

impl Default for Entropy {
    fn default() -> Self {
        Self::new()
    }
}

impl Entropy {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(EntropyConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EntropyConfig) -> Self {
        let initial = config.initial_coefficient;
        Self {
            current_coefficient: initial,
            log_coefficient: initial.ln(),
            entropy_history: VecDeque::with_capacity(config.stats_window),
            total_steps: 0,
            running_sum: 0.0,
            running_sum_sq: 0.0,
            min_observed: f64::INFINITY,
            max_observed: f64::NEG_INFINITY,
            config,
        }
    }

    /// Main processing function - for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Calculate entropy for a probability distribution
    pub fn calculate(&mut self, probabilities: &[f64]) -> Result<EntropyResult> {
        self.calculate_with_type(probabilities, EntropyType::Shannon)
    }

    /// Calculate entropy with specified type
    pub fn calculate_with_type(
        &mut self,
        probabilities: &[f64],
        entropy_type: EntropyType,
    ) -> Result<EntropyResult> {
        // Validate probabilities
        if probabilities.is_empty() {
            return Err(Error::InvalidInput("Empty probability distribution".into()));
        }

        let sum: f64 = probabilities.iter().sum();
        if (sum - 1.0).abs() > 0.01 {
            return Err(Error::InvalidInput(format!(
                "Probabilities must sum to 1.0, got {}",
                sum
            )));
        }

        // Calculate entropy based on type
        let entropy = match entropy_type {
            EntropyType::Shannon => self.shannon_entropy(probabilities),
            EntropyType::Renyi => self.renyi_entropy(probabilities, 2.0),
            EntropyType::Tsallis => self.tsallis_entropy(probabilities, 2.0),
            EntropyType::GiniSimpson => self.gini_simpson(probabilities),
        };

        // Maximum entropy for uniform distribution
        let n = probabilities.len() as f64;
        let max_entropy = n.ln();

        // Normalized entropy
        let normalized = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        // Calculate bonus
        let bonus = entropy * self.current_coefficient;

        // Update statistics
        self.update_stats(entropy);

        // Update coefficient
        self.update_coefficient(entropy);

        Ok(EntropyResult {
            entropy,
            normalized,
            bonus,
            coefficient: self.current_coefficient,
            max_entropy,
        })
    }

    /// Calculate Shannon entropy: H(p) = -sum(p * log(p))
    fn shannon_entropy(&self, probs: &[f64]) -> f64 {
        probs
            .iter()
            .filter(|&&p| p > self.config.min_prob)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Calculate Renyi entropy: H_alpha(p) = (1/(1-alpha)) * log(sum(p^alpha))
    fn renyi_entropy(&self, probs: &[f64], alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < 1e-10 {
            return self.shannon_entropy(probs);
        }

        let sum: f64 = probs.iter().map(|&p| p.powf(alpha)).sum();

        (1.0 / (1.0 - alpha)) * sum.ln()
    }

    /// Calculate Tsallis entropy: S_q(p) = (1/(q-1)) * (1 - sum(p^q))
    fn tsallis_entropy(&self, probs: &[f64], q: f64) -> f64 {
        if (q - 1.0).abs() < 1e-10 {
            return self.shannon_entropy(probs);
        }

        let sum: f64 = probs.iter().map(|&p| p.powf(q)).sum();

        (1.0 / (q - 1.0)) * (1.0 - sum)
    }

    /// Calculate Gini-Simpson index: 1 - sum(p^2)
    fn gini_simpson(&self, probs: &[f64]) -> f64 {
        let sum_sq: f64 = probs.iter().map(|&p| p * p).sum();
        1.0 - sum_sq
    }

    /// Update rolling statistics
    fn update_stats(&mut self, entropy: f64) {
        self.total_steps += 1;

        // Update min/max
        self.min_observed = self.min_observed.min(entropy);
        self.max_observed = self.max_observed.max(entropy);

        // Update running sums
        self.running_sum += entropy;
        self.running_sum_sq += entropy * entropy;

        // Add to history window
        self.entropy_history.push_back(entropy);
        if self.entropy_history.len() > self.config.stats_window {
            if let Some(old) = self.entropy_history.pop_front() {
                self.running_sum -= old;
                self.running_sum_sq -= old * old;
            }
        }
    }

    /// Update entropy coefficient (annealing or adaptive)
    fn update_coefficient(&mut self, entropy: f64) {
        if self.config.auto_tune {
            // SAC-style automatic entropy tuning
            if let Some(target) = self.config.target_entropy {
                let entropy_error = entropy - target;
                self.log_coefficient -= self.config.entropy_lr * entropy_error;
                self.current_coefficient = self.log_coefficient.exp();
            }
        } else {
            // Simple annealing
            self.current_coefficient = (self.current_coefficient * self.config.annealing_rate)
                .max(self.config.min_coefficient);
        }
    }

    /// Calculate entropy bonus for multiple action dimensions
    pub fn calculate_multidim(&mut self, distributions: &[Vec<f64>]) -> Result<Vec<EntropyResult>> {
        distributions
            .iter()
            .map(|d| self.calculate(d))
            .collect()
    }

    /// Calculate total entropy bonus across dimensions
    pub fn total_bonus(&mut self, distributions: &[Vec<f64>]) -> Result<f64> {
        let results = self.calculate_multidim(distributions)?;
        Ok(results.iter().map(|r| r.bonus).sum())
    }

    /// Get current entropy coefficient
    pub fn coefficient(&self) -> f64 {
        self.current_coefficient
    }

    /// Set entropy coefficient manually
    pub fn set_coefficient(&mut self, coeff: f64) {
        self.current_coefficient = coeff.max(self.config.min_coefficient);
        self.log_coefficient = self.current_coefficient.ln();
    }

    /// Get entropy statistics
    pub fn stats(&self) -> EntropyStats {
        let n = self.entropy_history.len();
        if n == 0 {
            return EntropyStats {
                current_coefficient: self.current_coefficient,
                ..Default::default()
            };
        }

        let mean = self.running_sum / n as f64;
        let variance = (self.running_sum_sq / n as f64) - (mean * mean);
        let std = variance.max(0.0).sqrt();

        EntropyStats {
            mean_entropy: mean,
            std_entropy: std,
            min_entropy: self.min_observed,
            max_entropy: self.max_observed,
            sample_count: self.total_steps as usize,
            current_coefficient: self.current_coefficient,
        }
    }

    /// Get total steps processed
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.current_coefficient = self.config.initial_coefficient;
        self.log_coefficient = self.current_coefficient.ln();
        self.entropy_history.clear();
        self.total_steps = 0;
        self.running_sum = 0.0;
        self.running_sum_sq = 0.0;
        self.min_observed = f64::INFINITY;
        self.max_observed = f64::NEG_INFINITY;
    }

    /// Calculate entropy of a softmax output from logits
    pub fn from_logits(&mut self, logits: &[f64]) -> Result<EntropyResult> {
        let probs = self.softmax(logits);
        self.calculate(&probs)
    }

    /// Apply softmax to convert logits to probabilities
    fn softmax(&self, logits: &[f64]) -> Vec<f64> {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&x| (x - max_logit).exp()).sum();

        logits
            .iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect()
    }

    /// Get recommended action given entropy-regularized values
    /// Higher entropy bonus encourages more random exploration
    pub fn entropy_regularized_values(&self, action_values: &[f64], temperature: f64) -> Vec<f64> {
        if temperature <= 0.0 || action_values.is_empty() {
            return action_values.to_vec();
        }

        // Scale by temperature and add entropy bonus consideration
        let scaled: Vec<f64> = action_values.iter().map(|&v| v / temperature).collect();

        // Apply softmax to get probabilities
        let probs = self.softmax(&scaled);

        // Calculate per-action entropy contribution
        probs
            .iter()
            .zip(action_values.iter())
            .map(|(&p, &v)| v + self.current_coefficient * (-p.max(self.config.min_prob).ln()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_basic() {
        let instance = Entropy::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_uniform_distribution_max_entropy() {
        let mut entropy = Entropy::new();

        // Uniform distribution should have maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let result = entropy.calculate(&uniform).unwrap();

        // For uniform distribution, entropy = ln(n)
        let expected_entropy = 4.0_f64.ln();
        assert!(
            approx_eq(result.entropy, expected_entropy, 0.001),
            "Expected entropy {}, got {}",
            expected_entropy,
            result.entropy
        );

        // Normalized should be 1.0
        assert!(
            approx_eq(result.normalized, 1.0, 0.001),
            "Normalized entropy should be 1.0"
        );
    }

    #[test]
    fn test_deterministic_distribution_zero_entropy() {
        let mut entropy = Entropy::new();

        // Deterministic distribution should have zero entropy
        let deterministic = vec![1.0, 0.0, 0.0, 0.0];
        let result = entropy.calculate(&deterministic).unwrap();

        assert!(
            result.entropy < 0.001,
            "Deterministic distribution should have near-zero entropy"
        );
    }

    #[test]
    fn test_binary_distribution() {
        let mut entropy = Entropy::new();

        // 50-50 binary should have entropy = ln(2)
        let binary = vec![0.5, 0.5];
        let result = entropy.calculate(&binary).unwrap();

        let expected = 2.0_f64.ln();
        assert!(
            approx_eq(result.entropy, expected, 0.001),
            "Binary uniform entropy should be ln(2)"
        );
    }

    #[test]
    fn test_entropy_bonus_calculation() {
        let config = EntropyConfig {
            initial_coefficient: 0.1,
            ..Default::default()
        };
        let mut entropy = Entropy::with_config(config);

        let probs = vec![0.5, 0.5];
        let result = entropy.calculate(&probs).unwrap();

        // Bonus should be entropy * coefficient
        let expected_bonus = result.entropy * 0.1;
        assert!(
            approx_eq(result.bonus, expected_bonus, 0.0001),
            "Bonus should be entropy * coefficient"
        );
    }

    #[test]
    fn test_coefficient_annealing() {
        let config = EntropyConfig {
            initial_coefficient: 0.1,
            annealing_rate: 0.9,
            min_coefficient: 0.01,
            auto_tune: false,
            ..Default::default()
        };
        let mut entropy = Entropy::with_config(config);

        let probs = vec![0.5, 0.5];
        let initial = entropy.coefficient();

        // Process several times
        for _ in 0..10 {
            entropy.calculate(&probs).unwrap();
        }

        // Coefficient should have decreased
        assert!(
            entropy.coefficient() < initial,
            "Coefficient should decrease with annealing"
        );

        // Should not go below minimum
        for _ in 0..1000 {
            entropy.calculate(&probs).unwrap();
        }
        assert!(
            entropy.coefficient() >= 0.01,
            "Coefficient should not go below minimum"
        );
    }

    #[test]
    fn test_renyi_entropy() {
        let mut entropy = Entropy::new();

        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = entropy.calculate_with_type(&probs, EntropyType::Renyi).unwrap();

        // For uniform distribution, Renyi entropy equals Shannon entropy
        let shannon = entropy.calculate_with_type(&probs, EntropyType::Shannon).unwrap();
        assert!(
            approx_eq(result.entropy, shannon.entropy, 0.01),
            "Renyi entropy for uniform should equal Shannon"
        );
    }

    #[test]
    fn test_gini_simpson() {
        let mut entropy = Entropy::new();

        // For uniform distribution, Gini-Simpson = 1 - 1/n
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = entropy.calculate_with_type(&probs, EntropyType::GiniSimpson).unwrap();

        let expected = 1.0 - 0.25; // 1 - 4 * 0.25^2 = 0.75
        assert!(
            approx_eq(result.entropy, expected, 0.001),
            "Gini-Simpson for uniform 4-way should be 0.75"
        );
    }

    #[test]
    fn test_invalid_probabilities() {
        let mut entropy = Entropy::new();

        // Empty distribution
        let empty: Vec<f64> = vec![];
        assert!(entropy.calculate(&empty).is_err());

        // Non-normalized distribution
        let non_normalized = vec![0.5, 0.3, 0.1];
        assert!(entropy.calculate(&non_normalized).is_err());
    }

    #[test]
    fn test_from_logits() {
        let mut entropy = Entropy::new();

        // Equal logits should give uniform distribution
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let result = entropy.from_logits(&logits).unwrap();

        // Should have maximum entropy
        assert!(
            result.normalized > 0.99,
            "Equal logits should give max entropy"
        );
    }

    #[test]
    fn test_multidim_entropy() {
        let mut entropy = Entropy::new();

        let distributions = vec![
            vec![0.5, 0.5],
            vec![0.25, 0.25, 0.25, 0.25],
        ];

        let results = entropy.calculate_multidim(&distributions).unwrap();
        assert_eq!(results.len(), 2);

        let total = entropy.total_bonus(&distributions).unwrap();
        assert!(total > 0.0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut entropy = Entropy::new();

        let probs = vec![0.5, 0.5];
        for _ in 0..50 {
            entropy.calculate(&probs).unwrap();
        }

        let stats = entropy.stats();
        assert_eq!(stats.sample_count, 50);
        assert!(stats.mean_entropy > 0.0);
        assert!(stats.max_entropy >= stats.min_entropy);
    }

    #[test]
    fn test_reset() {
        let mut entropy = Entropy::new();

        let probs = vec![0.5, 0.5];
        for _ in 0..10 {
            entropy.calculate(&probs).unwrap();
        }

        entropy.reset();

        assert_eq!(entropy.total_steps(), 0);
        let stats = entropy.stats();
        assert_eq!(stats.sample_count, 0);
    }

    #[test]
    fn test_set_coefficient() {
        let mut entropy = Entropy::new();

        entropy.set_coefficient(0.5);
        assert!(approx_eq(entropy.coefficient(), 0.5, 0.0001));

        // Should respect minimum
        entropy.set_coefficient(0.0001);
        assert!(entropy.coefficient() >= entropy.config.min_coefficient);
    }

    #[test]
    fn test_entropy_regularized_values() {
        let entropy = Entropy::new();

        let values = vec![1.0, 2.0, 3.0];
        let regularized = entropy.entropy_regularized_values(&values, 1.0);

        assert_eq!(regularized.len(), 3);
        // Higher values should still be preferred but with entropy bonus
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let entropy = Entropy::new();

        // Large logits that could cause overflow
        let large_logits = vec![1000.0, 1001.0, 999.0];
        let probs = entropy.softmax(&large_logits);

        // Should sum to 1.0
        let sum: f64 = probs.iter().sum();
        assert!(
            approx_eq(sum, 1.0, 0.001),
            "Softmax should sum to 1.0, got {}",
            sum
        );

        // Should not have NaN or Inf
        for &p in &probs {
            assert!(!p.is_nan() && !p.is_infinite());
        }
    }

    #[test]
    fn test_auto_tune() {
        let config = EntropyConfig {
            initial_coefficient: 0.1,
            target_entropy: Some(0.5),
            entropy_lr: 0.1,
            auto_tune: true,
            ..Default::default()
        };
        let mut entropy = Entropy::with_config(config);

        // Start with high entropy distribution
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let initial_coeff = entropy.coefficient();

        for _ in 0..50 {
            entropy.calculate(&probs).unwrap();
        }

        // Coefficient should have changed based on entropy vs target
        assert!(
            (entropy.coefficient() - initial_coeff).abs() > 0.001,
            "Auto-tune should adjust coefficient"
        );
    }
}
