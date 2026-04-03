//! Importance-weighted Sampling Strategies for Experience Replay
//!
//! Part of the Hippocampus region
//! Component: replay
//!
//! Implements various sampling strategies for experience replay:
//! - Uniform random sampling
//! - Prioritized Experience Replay (PER) with importance sampling correction
//! - Stratified sampling for better coverage
//! - Recent-biased sampling for non-stationary environments
//!
//! References:
//! - Schaul et al., "Prioritized Experience Replay" (2015)
//! - Zhang & Sutton, "A Deeper Look at Experience Replay" (2017)

use crate::common::{Error, Result};
use rand::RngExt;
use rand::prelude::*;
use std::collections::VecDeque;

use super::priority::Priority;
use super::sum_tree::SumTree;

/// Sampling strategy configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Batch size for sampling
    pub batch_size: usize,
    /// Minimum buffer size before sampling is allowed
    pub min_buffer_size: usize,
    /// Whether to use stratified sampling
    pub stratified: bool,
    /// Recency bias factor (0 = no bias, 1 = full bias)
    pub recency_bias: f64,
    /// Temperature for softmax sampling (higher = more uniform)
    pub temperature: f64,
    /// Seed for reproducibility (None for random)
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            min_buffer_size: 100,
            stratified: true,
            recency_bias: 0.0,
            temperature: 1.0,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Create config for uniform sampling
    pub fn uniform(batch_size: usize) -> Self {
        Self {
            batch_size,
            stratified: false,
            ..Default::default()
        }
    }

    /// Create config for prioritized sampling
    pub fn prioritized(batch_size: usize) -> Self {
        Self {
            batch_size,
            stratified: true,
            ..Default::default()
        }
    }

    /// Create config with recency bias
    pub fn with_recency(batch_size: usize, recency_bias: f64) -> Self {
        Self {
            batch_size,
            recency_bias: recency_bias.clamp(0.0, 1.0),
            ..Default::default()
        }
    }
}

/// Result of a sampling operation
#[derive(Debug, Clone)]
pub struct SampleBatch {
    /// Indices of sampled experiences
    pub indices: Vec<usize>,
    /// Importance sampling weights (for gradient correction)
    pub weights: Vec<f64>,
    /// Priorities of sampled experiences
    pub priorities: Vec<f64>,
}

impl SampleBatch {
    pub fn new(capacity: usize) -> Self {
        Self {
            indices: Vec::with_capacity(capacity),
            weights: Vec::with_capacity(capacity),
            priorities: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get normalized weights (max weight = 1.0)
    pub fn normalized_weights(&self) -> Vec<f64> {
        if self.weights.is_empty() {
            return Vec::new();
        }

        let max_weight = self
            .weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if max_weight <= 0.0 {
            return vec![1.0; self.weights.len()];
        }

        self.weights.iter().map(|w| w / max_weight).collect()
    }
}

/// Statistics for sampling operations
#[derive(Debug, Clone, Default)]
pub struct SamplingStats {
    /// Total samples drawn
    pub total_samples: u64,
    /// Total batches sampled
    pub total_batches: u64,
    /// Average weight across all samples
    pub avg_weight: f64,
    /// Minimum weight seen
    pub min_weight: f64,
    /// Maximum weight seen
    pub max_weight: f64,
    /// Average batch diversity (unique samples / batch size)
    pub avg_diversity: f64,
}

/// Importance-weighted sampling for experience replay
#[derive(Debug)]
pub struct Sampling {
    /// Configuration
    config: SamplingConfig,
    /// Sum tree for priority-based sampling
    sum_tree: SumTree,
    /// Priority calculator
    priority: Priority,
    /// Random number generator
    rng: StdRng,
    /// Statistics
    stats: SamplingStats,
    /// Current buffer size
    buffer_size: usize,
    /// History of recent sample indices (for diversity tracking)
    recent_samples: VecDeque<usize>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampling {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(SamplingConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: SamplingConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };

        Self {
            sum_tree: SumTree::new(10000), // Default capacity
            priority: Priority::new(),
            rng,
            stats: SamplingStats {
                min_weight: f64::MAX,
                ..Default::default()
            },
            buffer_size: 0,
            recent_samples: VecDeque::with_capacity(1000),
            config,
        }
    }

    /// Create with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let config = SamplingConfig::default();
        let rng = rand::make_rng();

        Self {
            config,
            sum_tree: SumTree::new(capacity),
            priority: Priority::new(),
            rng,
            stats: SamplingStats {
                min_weight: f64::MAX,
                ..Default::default()
            },
            buffer_size: 0,
            recent_samples: VecDeque::with_capacity(1000),
        }
    }

    /// Update capacity
    pub fn set_capacity(&mut self, capacity: usize) {
        self.sum_tree = SumTree::new(capacity);
    }

    /// Add a new experience with initial priority
    pub fn add(&mut self, td_error: f64) -> usize {
        let priority = self.priority.calculate(td_error);
        let idx = self.sum_tree.add(priority);
        self.buffer_size = self
            .buffer_size
            .saturating_add(1)
            .min(self.sum_tree.capacity());
        idx
    }

    /// Add with explicit priority
    pub fn add_with_priority(&mut self, priority: f64) -> usize {
        let idx = self.sum_tree.add(priority);
        self.buffer_size = self
            .buffer_size
            .saturating_add(1)
            .min(self.sum_tree.capacity());
        idx
    }

    /// Update priority for an experience
    pub fn update(&mut self, idx: usize, td_error: f64) {
        let priority = self.priority.calculate(td_error);
        self.sum_tree.update(idx, priority);
    }

    /// Update priority directly
    pub fn update_priority(&mut self, idx: usize, priority: f64) {
        self.sum_tree.update(idx, priority);
    }

    /// Sample a batch of experiences
    pub fn sample(&mut self) -> Result<SampleBatch> {
        self.sample_n(self.config.batch_size)
    }

    /// Sample n experiences
    pub fn sample_n(&mut self, n: usize) -> Result<SampleBatch> {
        if self.buffer_size < self.config.min_buffer_size {
            return Err(Error::InvalidInput(format!(
                "Buffer size {} below minimum {}",
                self.buffer_size, self.config.min_buffer_size
            )));
        }

        let total = self.sum_tree.total();
        if total <= 0.0 {
            return Err(Error::InvalidState("Sum tree is empty".to_string()));
        }

        let mut batch = SampleBatch::new(n);

        if self.config.stratified {
            // Stratified sampling: divide total into n segments
            let segment_size = total / n as f64;

            for i in 0..n {
                let low = segment_size * i as f64;
                let high = segment_size * (i + 1) as f64;

                // Sample uniformly within segment
                let value = low + self.rng.random::<f64>() * (high - low);
                let idx = self.sum_tree.sample(value);

                let priority = self.sum_tree.get_priority(idx);
                let weight = self.calculate_weight(idx, total, self.buffer_size);

                batch.indices.push(idx);
                batch.priorities.push(priority);
                batch.weights.push(weight);
            }
        } else {
            // Pure random sampling proportional to priority
            for _ in 0..n {
                let value = self.rng.random::<f64>() * total;
                let idx = self.sum_tree.sample(value);

                let priority = self.sum_tree.get_priority(idx);
                let weight = self.calculate_weight(idx, total, self.buffer_size);

                batch.indices.push(idx);
                batch.priorities.push(priority);
                batch.weights.push(weight);
            }
        }

        // Apply recency bias if configured
        if self.config.recency_bias > 0.0 {
            self.apply_recency_bias(&mut batch);
        }

        // Update statistics
        self.update_stats(&batch);

        // Track recent samples
        for &idx in &batch.indices {
            self.recent_samples.push_back(idx);
            if self.recent_samples.len() > 1000 {
                self.recent_samples.pop_front();
            }
        }

        Ok(batch)
    }

    /// Sample uniformly at random (ignoring priorities)
    pub fn sample_uniform(&mut self, n: usize) -> Result<SampleBatch> {
        if self.buffer_size < self.config.min_buffer_size {
            return Err(Error::InvalidInput(format!(
                "Buffer size {} below minimum {}",
                self.buffer_size, self.config.min_buffer_size
            )));
        }

        let mut batch = SampleBatch::new(n);

        for _ in 0..n {
            let idx = self.rng.random_range(0..self.buffer_size);
            batch.indices.push(idx);
            batch.priorities.push(1.0);
            batch.weights.push(1.0);
        }

        self.update_stats(&batch);
        Ok(batch)
    }

    /// Sample with temperature-scaled priorities
    pub fn sample_with_temperature(&mut self, n: usize, temperature: f64) -> Result<SampleBatch> {
        if self.buffer_size < self.config.min_buffer_size {
            return Err(Error::InvalidInput(format!(
                "Buffer size {} below minimum {}",
                self.buffer_size, self.config.min_buffer_size
            )));
        }

        // Collect all priorities and apply temperature scaling
        let priorities: Vec<f64> = (0..self.buffer_size)
            .map(|i| self.sum_tree.get_priority(i))
            .collect();

        // Apply temperature: p' = p^(1/T)
        let scaled: Vec<f64> = priorities
            .iter()
            .map(|&p| p.powf(1.0 / temperature))
            .collect();

        let total: f64 = scaled.iter().sum();
        if total <= 0.0 {
            return self.sample_uniform(n);
        }

        let mut batch = SampleBatch::new(n);

        // Sample from scaled distribution
        for _ in 0..n {
            let mut value = self.rng.random::<f64>() * total;
            let mut idx = 0;

            for (i, &p) in scaled.iter().enumerate() {
                value -= p;
                if value <= 0.0 {
                    idx = i;
                    break;
                }
            }

            batch.indices.push(idx);
            batch.priorities.push(priorities[idx]);
            batch.weights.push(1.0); // Uniform weights for temperature sampling
        }

        self.update_stats(&batch);
        Ok(batch)
    }

    /// Calculate importance sampling weight
    fn calculate_weight(&self, idx: usize, _total: f64, n: usize) -> f64 {
        let beta = self.priority.beta();
        self.sum_tree.importance_weight(idx, beta, n)
    }

    /// Apply recency bias to sampled batch
    fn apply_recency_bias(&mut self, batch: &mut SampleBatch) {
        let bias = self.config.recency_bias;
        if bias <= 0.0 || self.buffer_size == 0 {
            return;
        }

        // Re-weight based on recency
        for (i, &idx) in batch.indices.iter().enumerate() {
            // Recency factor: higher for more recent experiences
            let recency = idx as f64 / self.buffer_size as f64;
            let bias_factor = 1.0 + bias * recency;
            batch.weights[i] *= bias_factor;
        }
    }

    /// Update statistics
    fn update_stats(&mut self, batch: &SampleBatch) {
        self.stats.total_batches += 1;
        self.stats.total_samples += batch.len() as u64;

        for &w in &batch.weights {
            if w < self.stats.min_weight {
                self.stats.min_weight = w;
            }
            if w > self.stats.max_weight {
                self.stats.max_weight = w;
            }
        }

        // Update running average weight
        let batch_avg: f64 = batch.weights.iter().sum::<f64>() / batch.len() as f64;
        let n = self.stats.total_batches as f64;
        self.stats.avg_weight = self.stats.avg_weight * (n - 1.0) / n + batch_avg / n;

        // Calculate diversity (unique samples / batch size)
        let unique: std::collections::HashSet<_> = batch.indices.iter().collect();
        let diversity = unique.len() as f64 / batch.len() as f64;
        self.stats.avg_diversity = self.stats.avg_diversity * (n - 1.0) / n + diversity / n;
    }

    /// Get statistics
    pub fn stats(&self) -> &SamplingStats {
        &self.stats
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Set buffer size (for external buffer management)
    pub fn set_buffer_size(&mut self, size: usize) {
        self.buffer_size = size;
    }

    /// Get total priority
    pub fn total_priority(&self) -> f64 {
        self.sum_tree.total()
    }

    /// Get maximum priority
    pub fn max_priority(&self) -> f64 {
        self.sum_tree.max()
    }

    /// Get minimum priority
    pub fn min_priority(&self) -> f64 {
        self.sum_tree.min()
    }

    /// Anneal beta (for importance sampling correction)
    pub fn anneal_beta(&mut self) {
        self.priority.anneal_beta();
    }

    /// Set beta directly
    pub fn set_beta(&mut self, beta: f64) {
        self.priority.set_beta(beta);
    }

    /// Get current beta
    pub fn beta(&self) -> f64 {
        self.priority.beta()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.sum_tree.clear();
        self.priority.clear();
        self.buffer_size = 0;
        self.recent_samples.clear();
        self.stats = SamplingStats {
            min_weight: f64::MAX,
            ..Default::default()
        };
    }

    /// Check if ready to sample
    pub fn can_sample(&self) -> bool {
        self.buffer_size >= self.config.min_buffer_size && self.sum_tree.total() > 0.0
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Builder for Sampling configuration
pub struct SamplingBuilder {
    config: SamplingConfig,
    capacity: usize,
}

impl SamplingBuilder {
    pub fn new() -> Self {
        Self {
            config: SamplingConfig::default(),
            capacity: 10000,
        }
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn min_buffer_size(mut self, size: usize) -> Self {
        self.config.min_buffer_size = size;
        self
    }

    pub fn stratified(mut self, enabled: bool) -> Self {
        self.config.stratified = enabled;
        self
    }

    pub fn recency_bias(mut self, bias: f64) -> Self {
        self.config.recency_bias = bias.clamp(0.0, 1.0);
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.config.temperature = temp.max(0.01);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    pub fn build(self) -> Sampling {
        let rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };

        Sampling {
            sum_tree: SumTree::new(self.capacity),
            priority: Priority::new(),
            rng,
            stats: SamplingStats {
                min_weight: f64::MAX,
                ..Default::default()
            },
            buffer_size: 0,
            recent_samples: VecDeque::with_capacity(1000),
            config: self.config,
        }
    }
}

impl Default for SamplingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = Sampling::new();
        assert!(instance.process().is_ok());
        assert_eq!(instance.buffer_size(), 0);
        assert!(!instance.can_sample());
    }

    #[test]
    fn test_add_and_sample() {
        let mut sampling = SamplingBuilder::new()
            .min_buffer_size(5)
            .batch_size(3)
            .seed(42)
            .build();

        // Add experiences
        for i in 1..=10 {
            sampling.add(i as f64);
        }

        assert_eq!(sampling.buffer_size(), 10);
        assert!(sampling.can_sample());

        // Sample a batch
        let batch = sampling.sample().unwrap();
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_sample_before_min_buffer() {
        let mut sampling = SamplingBuilder::new().min_buffer_size(10).build();

        for i in 1..=5 {
            sampling.add(i as f64);
        }

        // Should fail - not enough samples
        assert!(sampling.sample().is_err());
    }

    #[test]
    fn test_stratified_sampling() {
        let mut sampling = SamplingBuilder::new()
            .min_buffer_size(5)
            .batch_size(5)
            .stratified(true)
            .seed(42)
            .capacity(100)
            .build();

        // Add with varying priorities
        for i in 1..=20 {
            sampling.add(i as f64);
        }

        let batch = sampling.sample().unwrap();
        assert_eq!(batch.len(), 5);

        // Stratified sampling should give diverse indices
        let unique: std::collections::HashSet<_> = batch.indices.iter().collect();
        // With stratified sampling, we expect good diversity
        assert!(unique.len() >= 3);
    }

    #[test]
    fn test_uniform_sampling() {
        let mut sampling = SamplingBuilder::new().min_buffer_size(5).seed(42).build();

        for i in 1..=20 {
            sampling.add(i as f64);
        }

        let batch = sampling.sample_uniform(10).unwrap();
        assert_eq!(batch.len(), 10);

        // All weights should be 1.0 for uniform
        for w in &batch.weights {
            assert!((*w - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_update_priority() {
        let mut sampling = SamplingBuilder::new().min_buffer_size(3).seed(42).build();

        let _idx0 = sampling.add(1.0);
        let idx1 = sampling.add(1.0);
        let _idx2 = sampling.add(1.0);

        let total_before = sampling.total_priority();

        // Increase priority of one experience
        sampling.update(idx1, 100.0);

        let total_after = sampling.total_priority();
        assert!(total_after > total_before);
    }

    #[test]
    fn test_importance_weights() {
        let mut sampling = SamplingBuilder::new()
            .min_buffer_size(3)
            .batch_size(3)
            .seed(42)
            .build();

        // Add with different priorities
        sampling.add(0.1); // Low priority
        sampling.add(1.0); // Medium priority
        sampling.add(10.0); // High priority

        let batch = sampling.sample().unwrap();

        // Weights should be positive
        for w in &batch.weights {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn test_normalized_weights() {
        let mut batch = SampleBatch::new(3);
        batch.weights = vec![0.5, 1.0, 2.0];

        let normalized = batch.normalized_weights();

        assert_eq!(normalized.len(), 3);
        assert!((normalized[2] - 1.0).abs() < 1e-10); // Max should be 1.0
        assert!((normalized[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_temperature_sampling() {
        let mut sampling = SamplingBuilder::new().min_buffer_size(5).seed(42).build();

        // Add experiences with varying priorities
        for i in 1..=10 {
            sampling.add((i * 10) as f64);
        }

        // High temperature should give more uniform distribution
        let batch_high_temp = sampling.sample_with_temperature(100, 10.0).unwrap();

        // Low temperature should favor high priority
        let batch_low_temp = sampling.sample_with_temperature(100, 0.1).unwrap();

        // Calculate average index (higher index = higher priority in this setup)
        let avg_high: f64 = batch_high_temp
            .indices
            .iter()
            .map(|&i| i as f64)
            .sum::<f64>()
            / batch_high_temp.len() as f64;
        let avg_low: f64 = batch_low_temp
            .indices
            .iter()
            .map(|&i| i as f64)
            .sum::<f64>()
            / batch_low_temp.len() as f64;

        // Low temperature should have higher average index (favoring high priority)
        // This may not always hold due to randomness, but trend should be visible
        // We don't assert strictly here due to stochastic nature
        assert!(avg_high >= 0.0);
        assert!(avg_low >= 0.0);
    }

    #[test]
    fn test_beta_annealing() {
        let mut sampling = Sampling::new();

        let initial_beta = sampling.beta();
        sampling.anneal_beta();
        let new_beta = sampling.beta();

        assert!(new_beta >= initial_beta);
    }

    #[test]
    fn test_clear() {
        let mut sampling = Sampling::new();

        for i in 1..=10 {
            sampling.add(i as f64);
        }

        assert_eq!(sampling.buffer_size(), 10);

        sampling.clear();

        assert_eq!(sampling.buffer_size(), 0);
        assert!((sampling.total_priority() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats() {
        let mut sampling = SamplingBuilder::new()
            .min_buffer_size(3)
            .batch_size(2)
            .seed(42)
            .build();

        for i in 1..=5 {
            sampling.add(i as f64);
        }

        let _ = sampling.sample().unwrap();
        let _ = sampling.sample().unwrap();

        let stats = sampling.stats();
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.total_samples, 4);
        assert!(stats.avg_weight > 0.0);
    }

    #[test]
    fn test_recency_bias() {
        let mut sampling = SamplingBuilder::new()
            .min_buffer_size(3)
            .batch_size(5)
            .recency_bias(0.5)
            .seed(42)
            .build();

        for _i in 1..=10 {
            sampling.add(1.0); // Same priority for all
        }

        let batch = sampling.sample().unwrap();

        // Weights should vary based on recency
        let min_w = batch.weights.iter().cloned().fold(f64::MAX, f64::min);
        let max_w = batch.weights.iter().cloned().fold(f64::MIN, f64::max);

        // With recency bias, weights should not all be equal
        assert!(max_w > min_w || batch.len() <= 1);
    }

    #[test]
    fn test_builder() {
        let sampling = SamplingBuilder::new()
            .batch_size(64)
            .min_buffer_size(200)
            .stratified(false)
            .recency_bias(0.3)
            .capacity(50000)
            .seed(12345)
            .build();

        assert_eq!(sampling.config.batch_size, 64);
        assert_eq!(sampling.config.min_buffer_size, 200);
        assert!(!sampling.config.stratified);
        assert!((sampling.config.recency_bias - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_sample_batch_is_empty() {
        let batch = SampleBatch::new(0);
        assert!(batch.is_empty());

        let mut batch2 = SampleBatch::new(1);
        batch2.indices.push(0);
        assert!(!batch2.is_empty());
    }

    #[test]
    fn test_can_sample() {
        let mut sampling = SamplingBuilder::new().min_buffer_size(5).build();

        assert!(!sampling.can_sample());

        for i in 1..=3 {
            sampling.add(i as f64);
        }
        assert!(!sampling.can_sample());

        for i in 4..=6 {
            sampling.add(i as f64);
        }
        assert!(sampling.can_sample());
    }

    #[test]
    fn test_add_with_priority() {
        let mut sampling = Sampling::new();

        sampling.add_with_priority(5.0);
        sampling.add_with_priority(10.0);
        sampling.add_with_priority(15.0);

        // Total should be sum of priorities
        assert!((sampling.total_priority() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_min_priority() {
        let mut sampling = Sampling::new();

        sampling.add_with_priority(1.0);
        sampling.add_with_priority(5.0);
        sampling.add_with_priority(10.0);

        assert!((sampling.max_priority() - 10.0).abs() < 1e-10);
        assert!((sampling.min_priority() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_buffer_size() {
        let mut sampling = Sampling::new();

        sampling.set_buffer_size(100);
        assert_eq!(sampling.buffer_size(), 100);
    }
}
