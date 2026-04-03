//! TD-error based Priority Calculation for Prioritized Experience Replay
//!
//! Part of the Hippocampus region
//! Component: replay
//!
//! Implements priority calculation based on TD-errors for Prioritized
//! Experience Replay (PER). Higher TD-errors indicate more "surprising"
//! experiences that the model can learn more from.
//!
//! References:
//! - Schaul et al., "Prioritized Experience Replay" (2015)

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Priority calculation configuration
#[derive(Debug, Clone)]
pub struct PriorityConfig {
    /// Alpha: exponent for prioritization (0 = uniform, 1 = full prioritization)
    pub alpha: f64,
    /// Beta: importance sampling correction (starts low, anneals to 1)
    pub beta: f64,
    /// Beta increment per sampling
    pub beta_increment: f64,
    /// Maximum beta value
    pub beta_max: f64,
    /// Small constant to ensure non-zero priorities
    pub epsilon: f64,
    /// Maximum priority (for clipping)
    pub max_priority: f64,
    /// Initial priority for new experiences
    pub initial_priority: f64,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            alpha: 0.6, // Recommended in PER paper
            beta: 0.4,  // Start low, anneal to 1
            beta_increment: 1e-4,
            beta_max: 1.0,
            epsilon: 1e-6,
            max_priority: 100.0,
            initial_priority: 1.0,
        }
    }
}

impl PriorityConfig {
    /// Create config for proportional prioritization
    pub fn proportional() -> Self {
        Self {
            alpha: 0.6,
            ..Default::default()
        }
    }

    /// Create config for rank-based prioritization
    pub fn rank_based() -> Self {
        Self {
            alpha: 0.7, // Rank-based typically uses higher alpha
            ..Default::default()
        }
    }

    /// Create config for uniform sampling (no prioritization)
    pub fn uniform() -> Self {
        Self {
            alpha: 0.0,
            beta: 1.0, // No IS correction needed
            ..Default::default()
        }
    }
}

/// TD-error based priority calculation
#[derive(Debug, Clone)]
pub struct Priority {
    /// Configuration
    config: PriorityConfig,
    /// Current beta value (for importance sampling)
    current_beta: f64,
    /// Priority values indexed by experience ID
    priorities: HashMap<usize, f64>,
    /// TD-errors for tracking
    td_errors: HashMap<usize, f64>,
    /// Statistics
    stats: PriorityStats,
}

/// Statistics for priority tracking
#[derive(Debug, Clone, Default)]
pub struct PriorityStats {
    /// Total priority updates
    pub total_updates: u64,
    /// Average TD-error
    pub avg_td_error: f64,
    /// Maximum TD-error seen
    pub max_td_error: f64,
    /// Minimum TD-error seen
    pub min_td_error: f64,
    /// Number of experiences tracked
    pub experience_count: usize,
}

impl Default for Priority {
    fn default() -> Self {
        Self::new()
    }
}

impl Priority {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(PriorityConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: PriorityConfig) -> Self {
        Self {
            current_beta: config.beta,
            config,
            priorities: HashMap::new(),
            td_errors: HashMap::new(),
            stats: PriorityStats {
                min_td_error: f64::MAX,
                ..Default::default()
            },
        }
    }

    /// Calculate priority from TD-error
    ///
    /// priority = (|TD-error| + epsilon)^alpha
    pub fn calculate(&self, td_error: f64) -> f64 {
        let abs_error = td_error.abs();
        let priority = (abs_error + self.config.epsilon).powf(self.config.alpha);
        priority.min(self.config.max_priority)
    }

    /// Calculate priority from TD-error with rank-based approach
    ///
    /// priority = 1 / rank^alpha
    pub fn calculate_rank_based(&self, rank: usize) -> f64 {
        if rank == 0 {
            return self.config.max_priority;
        }
        (1.0 / rank as f64).powf(self.config.alpha)
    }

    /// Set priority for an experience based on its TD-error
    pub fn set_td_error(&mut self, experience_id: usize, td_error: f64) {
        let priority = self.calculate(td_error);
        self.priorities.insert(experience_id, priority);
        self.td_errors.insert(experience_id, td_error);
        self.update_stats(td_error);
    }

    /// Set priority directly
    pub fn set_priority(&mut self, experience_id: usize, priority: f64) {
        let clamped = priority.clamp(self.config.epsilon, self.config.max_priority);
        self.priorities.insert(experience_id, clamped);
    }

    /// Get priority for an experience
    pub fn get(&self, experience_id: usize) -> f64 {
        self.priorities
            .get(&experience_id)
            .copied()
            .unwrap_or(self.config.initial_priority)
    }

    /// Get TD-error for an experience
    pub fn get_td_error(&self, experience_id: usize) -> Option<f64> {
        self.td_errors.get(&experience_id).copied()
    }

    /// Calculate importance sampling weight
    ///
    /// weight = (N * P(i))^(-beta) / max_weight
    pub fn importance_weight(&self, experience_id: usize, total_priority: f64, n: usize) -> f64 {
        if total_priority <= 0.0 || n == 0 {
            return 1.0;
        }

        let priority = self.get(experience_id);
        let prob = priority / total_priority;

        // Calculate weight
        let weight = (n as f64 * prob).powf(-self.current_beta);

        // For normalization, we'd need the max weight (at min priority)
        // For simplicity, we cap at 1.0
        weight.min(1.0)
    }

    /// Calculate importance sampling weights for a batch
    pub fn importance_weights(&self, experience_ids: &[usize], total_priority: f64) -> Vec<f64> {
        let n = experience_ids.len();
        if total_priority <= 0.0 || n == 0 {
            return vec![1.0; n];
        }

        // Find minimum probability for max weight calculation
        let min_priority = experience_ids
            .iter()
            .map(|&id| self.get(id))
            .fold(f64::MAX, f64::min);
        let max_prob = min_priority / total_priority;
        let max_weight = if max_prob > 0.0 {
            (n as f64 * max_prob).powf(-self.current_beta)
        } else {
            1.0
        };

        experience_ids
            .iter()
            .map(|&id| {
                let priority = self.get(id);
                let prob = priority / total_priority;
                let weight = (n as f64 * prob).powf(-self.current_beta);
                // Normalize by max weight
                if max_weight > 0.0 {
                    weight / max_weight
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Anneal beta towards 1.0
    pub fn anneal_beta(&mut self) {
        self.current_beta =
            (self.current_beta + self.config.beta_increment).min(self.config.beta_max);
    }

    /// Set beta directly
    pub fn set_beta(&mut self, beta: f64) {
        self.current_beta = beta.clamp(0.0, self.config.beta_max);
    }

    /// Get current beta value
    pub fn beta(&self) -> f64 {
        self.current_beta
    }

    /// Get alpha value
    pub fn alpha(&self) -> f64 {
        self.config.alpha
    }

    /// Update statistics with new TD-error
    fn update_stats(&mut self, td_error: f64) {
        let abs_error = td_error.abs();
        self.stats.total_updates += 1;
        self.stats.experience_count = self.priorities.len();

        // Update max/min
        if abs_error > self.stats.max_td_error {
            self.stats.max_td_error = abs_error;
        }
        if abs_error < self.stats.min_td_error {
            self.stats.min_td_error = abs_error;
        }

        // Running average
        let n = self.stats.total_updates as f64;
        self.stats.avg_td_error = self.stats.avg_td_error * (n - 1.0) / n + abs_error / n;
    }

    /// Get statistics
    pub fn stats(&self) -> &PriorityStats {
        &self.stats
    }

    /// Remove priority for an experience
    pub fn remove(&mut self, experience_id: usize) {
        self.priorities.remove(&experience_id);
        self.td_errors.remove(&experience_id);
    }

    /// Clear all priorities
    pub fn clear(&mut self) {
        self.priorities.clear();
        self.td_errors.clear();
        self.stats = PriorityStats {
            min_td_error: f64::MAX,
            ..Default::default()
        };
    }

    /// Get all priorities as iterator
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &f64)> {
        self.priorities.iter()
    }

    /// Get count of tracked experiences
    pub fn len(&self) -> usize {
        self.priorities.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.priorities.is_empty()
    }

    /// Batch update priorities from TD-errors
    pub fn batch_update(&mut self, updates: &[(usize, f64)]) {
        for &(experience_id, td_error) in updates {
            self.set_td_error(experience_id, td_error);
        }
    }

    /// Get maximum priority value
    pub fn max_priority(&self) -> f64 {
        self.priorities
            .values()
            .copied()
            .fold(self.config.initial_priority, f64::max)
    }

    /// Get minimum priority value
    pub fn min_priority(&self) -> f64 {
        self.priorities.values().copied().fold(f64::MAX, f64::min)
    }

    /// Get sum of all priorities
    pub fn total_priority(&self) -> f64 {
        self.priorities.values().sum()
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.config.alpha < 0.0 || self.config.alpha > 1.0 {
            return Err(Error::InvalidInput("Alpha must be in [0, 1]".to_string()));
        }
        if self.config.beta < 0.0 || self.config.beta > 1.0 {
            return Err(Error::InvalidInput("Beta must be in [0, 1]".to_string()));
        }
        if self.config.epsilon <= 0.0 {
            return Err(Error::InvalidInput("Epsilon must be positive".to_string()));
        }
        Ok(())
    }
}

/// Builder for Priority configuration
pub struct PriorityBuilder {
    config: PriorityConfig,
}

impl PriorityBuilder {
    pub fn new() -> Self {
        Self {
            config: PriorityConfig::default(),
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn beta(mut self, beta: f64) -> Self {
        self.config.beta = beta;
        self
    }

    pub fn beta_increment(mut self, increment: f64) -> Self {
        self.config.beta_increment = increment;
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    pub fn max_priority(mut self, max: f64) -> Self {
        self.config.max_priority = max;
        self
    }

    pub fn initial_priority(mut self, initial: f64) -> Self {
        self.config.initial_priority = initial;
        self
    }

    pub fn build(self) -> Priority {
        Priority::with_config(self.config)
    }
}

impl Default for PriorityBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = Priority::new();
        assert!(instance.process().is_ok());
        assert!(instance.is_empty());
    }

    #[test]
    fn test_calculate_priority() {
        let priority = Priority::new();

        // Higher TD-error should give higher priority
        let p1 = priority.calculate(0.1);
        let p2 = priority.calculate(1.0);
        let p3 = priority.calculate(10.0);

        assert!(p1 < p2);
        assert!(p2 < p3);
    }

    #[test]
    fn test_set_and_get() {
        let mut priority = Priority::new();

        priority.set_td_error(0, 1.0);
        priority.set_td_error(1, 2.0);
        priority.set_td_error(2, 0.5);

        assert!(priority.get(0) > 0.0);
        assert!(priority.get(1) > priority.get(0));
        assert!(priority.get(2) < priority.get(0));
    }

    #[test]
    fn test_importance_weight() {
        let mut priority = Priority::new();

        priority.set_td_error(0, 1.0);
        priority.set_td_error(1, 2.0);
        priority.set_td_error(2, 3.0);

        let total = priority.total_priority();

        // Lower priority experiences should have higher IS weights
        let w0 = priority.importance_weight(0, total, 3);
        let w1 = priority.importance_weight(1, total, 3);
        let w2 = priority.importance_weight(2, total, 3);

        // All weights should be positive
        assert!(w0 > 0.0);
        assert!(w1 > 0.0);
        assert!(w2 > 0.0);
    }

    #[test]
    fn test_batch_importance_weights() {
        let mut priority = Priority::new();

        for i in 0..5 {
            priority.set_td_error(i, (i + 1) as f64);
        }

        let total = priority.total_priority();
        let ids: Vec<usize> = (0..5).collect();
        let weights = priority.importance_weights(&ids, total);

        assert_eq!(weights.len(), 5);

        // All weights should be in [0, 1] after normalization
        for w in &weights {
            assert!(*w >= 0.0);
            assert!(*w <= 1.0);
        }
    }

    #[test]
    fn test_beta_annealing() {
        let config = PriorityConfig {
            beta: 0.4,
            beta_increment: 0.1,
            beta_max: 1.0,
            ..Default::default()
        };
        let mut priority = Priority::with_config(config);

        assert!((priority.beta() - 0.4).abs() < 1e-10);

        priority.anneal_beta();
        assert!((priority.beta() - 0.5).abs() < 1e-10);

        priority.anneal_beta();
        assert!((priority.beta() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_beta_capped_at_max() {
        let config = PriorityConfig {
            beta: 0.95,
            beta_increment: 0.1,
            beta_max: 1.0,
            ..Default::default()
        };
        let mut priority = Priority::with_config(config);

        priority.anneal_beta();
        assert!((priority.beta() - 1.0).abs() < 1e-10);

        priority.anneal_beta();
        assert!((priority.beta() - 1.0).abs() < 1e-10); // Still capped
    }

    #[test]
    fn test_uniform_sampling() {
        let priority = Priority::with_config(PriorityConfig::uniform());

        // With alpha=0, all priorities should be equal (epsilon)
        let p1 = priority.calculate(0.1);
        let p2 = priority.calculate(10.0);

        assert!((p1 - p2).abs() < 1e-10);
    }

    #[test]
    fn test_stats() {
        let mut priority = Priority::new();

        priority.set_td_error(0, 1.0);
        priority.set_td_error(1, 2.0);
        priority.set_td_error(2, 3.0);

        let stats = priority.stats();
        assert_eq!(stats.total_updates, 3);
        assert_eq!(stats.experience_count, 3);
        assert!((stats.max_td_error - 3.0).abs() < 1e-10);
        assert!((stats.min_td_error - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clear() {
        let mut priority = Priority::new();

        priority.set_td_error(0, 1.0);
        priority.set_td_error(1, 2.0);
        assert_eq!(priority.len(), 2);

        priority.clear();
        assert!(priority.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut priority = Priority::new();

        priority.set_td_error(0, 1.0);
        priority.set_td_error(1, 2.0);
        assert_eq!(priority.len(), 2);

        priority.remove(0);
        assert_eq!(priority.len(), 1);
        assert!(priority.get_td_error(0).is_none());
        assert!(priority.get_td_error(1).is_some());
    }

    #[test]
    fn test_batch_update() {
        let mut priority = Priority::new();

        let updates = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        priority.batch_update(&updates);

        assert_eq!(priority.len(), 3);
        assert!(priority.get(2) > priority.get(1));
        assert!(priority.get(1) > priority.get(0));
    }

    #[test]
    fn test_max_min_priority() {
        let mut priority = Priority::new();

        priority.set_td_error(0, 1.0);
        priority.set_td_error(1, 5.0);
        priority.set_td_error(2, 0.1);

        let max = priority.max_priority();
        let min = priority.min_priority();

        assert!(max > min);
        assert_eq!(max, priority.get(1));
        assert_eq!(min, priority.get(2));
    }

    #[test]
    fn test_builder() {
        let priority = PriorityBuilder::new()
            .alpha(0.8)
            .beta(0.5)
            .epsilon(1e-5)
            .build();

        assert!((priority.alpha() - 0.8).abs() < 1e-10);
        assert!((priority.beta() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_validate() {
        let valid_priority = Priority::new();
        assert!(valid_priority.validate().is_ok());

        let config = PriorityConfig {
            alpha: 1.5, // Invalid
            ..Default::default()
        };
        let invalid_priority = Priority::with_config(config);
        assert!(invalid_priority.validate().is_err());
    }

    #[test]
    fn test_rank_based() {
        let priority = Priority::with_config(PriorityConfig::rank_based());

        // Higher rank (worse) should have lower priority
        let p1 = priority.calculate_rank_based(1);
        let p2 = priority.calculate_rank_based(10);
        let p3 = priority.calculate_rank_based(100);

        assert!(p1 > p2);
        assert!(p2 > p3);
    }

    #[test]
    fn test_priority_clamping() {
        let config = PriorityConfig {
            max_priority: 10.0,
            ..Default::default()
        };
        let priority = Priority::with_config(config);

        // Very large TD-error should be clamped
        let p = priority.calculate(1000.0);
        assert!(p <= 10.0);
    }

    #[test]
    fn test_iter() {
        let mut priority = Priority::new();

        priority.set_td_error(0, 1.0);
        priority.set_td_error(1, 2.0);

        let count = priority.iter().count();
        assert_eq!(count, 2);
    }
}
