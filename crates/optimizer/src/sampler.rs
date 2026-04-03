//! Sampler Trait and Types
//!
//! This module defines the `Sampler` trait that all search algorithms must implement,
//! along with the `SamplerType` enum for selecting samplers dynamically.

use crate::constraints::{SampledParams, SearchSpace};
use crate::error::Result;
use crate::results::TrialResult;
use serde::{Deserialize, Serialize};

/// Enum of available sampler types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SamplerType {
    /// Random search - uniformly samples from the search space
    Random,

    /// Grid search - exhaustively searches a discretized grid
    Grid,

    /// TPE (Tree-structured Parzen Estimator) - Bayesian optimization
    #[default]
    Tpe,

    /// Evolutionary - uses genetic algorithm principles
    Evolutionary,
}

impl std::fmt::Display for SamplerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            SamplerType::Random => "Random",
            SamplerType::Grid => "Grid",
            SamplerType::Tpe => "TPE",
            SamplerType::Evolutionary => "Evolutionary",
        };
        write!(f, "{}", name)
    }
}

/// Trait that all parameter samplers must implement
pub trait Sampler: Default + Clone + Send + Sync {
    /// Sample a new set of parameters from the search space
    ///
    /// # Arguments
    /// * `space` - The search space defining parameter bounds
    /// * `history` - Previous trial results (for adaptive samplers)
    ///
    /// # Returns
    /// A new set of sampled parameters
    fn sample(&self, space: &SearchSpace, history: &[TrialResult]) -> Result<SampledParams>;

    /// Notify the sampler of a completed trial result
    ///
    /// This allows adaptive samplers (like TPE) to update their models.
    ///
    /// # Arguments
    /// * `params` - The parameters that were evaluated
    /// * `score` - The objective score achieved
    fn tell(&mut self, params: &SampledParams, score: f64) -> Result<()>;

    /// Reset the sampler's internal state
    fn reset(&mut self);

    /// Get the sampler type
    fn sampler_type(&self) -> SamplerType;

    /// Get the name of this sampler
    fn name(&self) -> &'static str {
        match self.sampler_type() {
            SamplerType::Random => "RandomSearch",
            SamplerType::Grid => "GridSearch",
            SamplerType::Tpe => "TPESampler",
            SamplerType::Evolutionary => "EvolutionarySampler",
        }
    }

    /// Check if this sampler is adaptive (uses history)
    fn is_adaptive(&self) -> bool {
        matches!(
            self.sampler_type(),
            SamplerType::Tpe | SamplerType::Evolutionary
        )
    }

    /// Get the minimum number of trials before adaptive behavior kicks in
    fn warmup_trials(&self) -> usize {
        match self.sampler_type() {
            SamplerType::Tpe => 10,
            SamplerType::Evolutionary => 20,
            _ => 0,
        }
    }
}

/// Configuration for samplers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Number of warmup trials (random sampling before adaptive)
    pub n_warmup: usize,

    /// Number of candidates to consider in TPE
    pub n_candidates: usize,

    /// Gamma parameter for TPE (fraction to consider as "good")
    pub gamma: f64,

    /// Population size for evolutionary sampler
    pub population_size: usize,

    /// Mutation rate for evolutionary sampler
    pub mutation_rate: f64,

    /// Crossover rate for evolutionary sampler
    pub crossover_rate: f64,

    /// Tournament size for evolutionary selection
    pub tournament_size: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            seed: None,
            n_warmup: 10,
            n_candidates: 24,
            gamma: 0.25, // Top 25% considered "good"
            population_size: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            tournament_size: 3,
        }
    }
}

impl SamplerConfig {
    /// Create new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set number of warmup trials
    pub fn with_warmup(mut self, n: usize) -> Self {
        self.n_warmup = n;
        self
    }

    /// Set number of TPE candidates
    pub fn with_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n;
        self
    }

    /// Set TPE gamma
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma.clamp(0.01, 0.5);
        self
    }

    /// Set evolutionary population size
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size.max(10);
        self
    }

    /// Set mutation rate
    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set crossover rate
    pub fn with_crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate.clamp(0.0, 1.0);
        self
    }
}

/// Helper struct for tracking sampler statistics
#[derive(Debug, Clone, Default)]
pub struct SamplerStats {
    /// Total samples generated
    pub total_samples: usize,

    /// Samples that violated constraints
    pub constraint_violations: usize,

    /// Samples that were resampled
    pub resamples: usize,

    /// Number of times tell() was called
    pub feedback_count: usize,

    /// Best score seen
    pub best_score: Option<f64>,

    /// Worst score seen
    pub worst_score: Option<f64>,

    /// Running sum of scores for computing the average
    score_sum: f64,
}

impl SamplerStats {
    /// Record a new sample
    pub fn record_sample(&mut self) {
        self.total_samples += 1;
    }

    /// Record a constraint violation
    pub fn record_violation(&mut self) {
        self.constraint_violations += 1;
        self.resamples += 1;
    }

    /// Record feedback from a completed trial
    pub fn record_feedback(&mut self, score: f64) {
        self.feedback_count += 1;
        self.score_sum += score;

        match self.best_score {
            Some(best) if score > best => self.best_score = Some(score),
            None => self.best_score = Some(score),
            _ => {}
        }

        match self.worst_score {
            Some(worst) if score < worst => self.worst_score = Some(score),
            None => self.worst_score = Some(score),
            _ => {}
        }
    }

    /// Get constraint violation rate
    pub fn violation_rate(&self) -> f64 {
        if self.total_samples > 0 {
            self.constraint_violations as f64 / self.total_samples as f64
        } else {
            0.0
        }
    }

    /// Get the average score across all feedback received
    pub fn average_score(&self) -> Option<f64> {
        if self.feedback_count > 0 {
            Some(self.score_sum / self.feedback_count as f64)
        } else {
            None
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Utility function to sample a uniform random value in a range
pub fn sample_uniform(rng: &mut impl rand::RngExt, min: f64, max: f64) -> f64 {
    rng.random_range(min..=max)
}

/// Utility function to sample a log-uniform value (for parameters that vary across orders of magnitude)
pub fn sample_log_uniform(rng: &mut impl rand::RngExt, min: f64, max: f64) -> f64 {
    let log_min = min.ln();
    let log_max = max.ln();
    rng.random_range(log_min..=log_max).exp()
}

/// Utility function to sample from a discrete set of values
pub fn sample_discrete<T: Clone>(rng: &mut impl rand::RngExt, values: &[T]) -> Option<T> {
    use rand::seq::IndexedRandom;
    values.choose(rng).cloned()
}

/// Utility function to snap a value to the nearest step
pub fn snap_to_step(value: f64, min: f64, step: f64) -> f64 {
    let steps = ((value - min) / step).round();
    min + steps * step
}

// ============================================================================
// Statistical Sampling Utilities (ported from Kraken optimizer)
// ============================================================================

/// Sample from a normal distribution using the Box-Muller transform.
///
/// This produces properly Gaussian-distributed samples, which is critical for
/// TPE candidate generation. The previous JANUS implementation used uniform
/// noise in `[-1, 1]` which biases exploration toward the edges of the
/// perturbation range rather than concentrating near the center.
///
/// Ported from `kraken::optimizer::sampler::sample_normal`.
pub fn sample_normal(rng: &mut impl rand::RngExt, mean: f64, std_dev: f64) -> f64 {
    let u1: f64 = rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();

    // Box-Muller transform
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + std_dev * z0
}

/// Sample from a truncated normal distribution (rejection sampling).
///
/// Draws from `N(mean, std_dev)` but rejects values outside `[min, max]`.
/// Falls back to uniform sampling if rejection sampling doesn't converge
/// within 100 attempts.
///
/// Ported from `kraken::optimizer::sampler::sample_truncated_normal`.
pub fn sample_truncated_normal(
    rng: &mut impl rand::RngExt,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
) -> f64 {
    for _ in 0..100 {
        let sample = sample_normal(rng, mean, std_dev);
        if sample >= min && sample <= max {
            return sample;
        }
    }
    // Fallback to uniform if rejection sampling fails
    sample_uniform(rng, min, max)
}

/// Calculate kernel bandwidth using Silverman's rule of thumb.
///
/// `h = 1.06 * σ * n^(-1/5)`
///
/// This data-adaptive bandwidth produces much better KDE estimates than a
/// fixed fraction of the parameter range. With few observations the bandwidth
/// is wide (exploratory); as observations accumulate it shrinks (exploitative).
///
/// Ported from `kraken::optimizer::sampler::silverman_bandwidth`.
pub fn silverman_bandwidth(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Silverman's rule: h = 1.06 * sigma * n^(-1/5)
    1.06 * std_dev * n.powf(-0.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_type_display() {
        assert_eq!(SamplerType::Random.to_string(), "Random");
        assert_eq!(SamplerType::Grid.to_string(), "Grid");
        assert_eq!(SamplerType::Tpe.to_string(), "TPE");
        assert_eq!(SamplerType::Evolutionary.to_string(), "Evolutionary");
    }

    #[test]
    fn test_sampler_type_default() {
        assert_eq!(SamplerType::default(), SamplerType::Tpe);
    }

    #[test]
    fn test_sampler_config_default() {
        let config = SamplerConfig::default();
        assert!(config.seed.is_none());
        assert_eq!(config.n_warmup, 10);
        assert_eq!(config.n_candidates, 24);
        assert_eq!(config.gamma, 0.25);
    }

    #[test]
    fn test_sampler_config_builder() {
        let config = SamplerConfig::new()
            .with_seed(42)
            .with_warmup(20)
            .with_candidates(50)
            .with_gamma(0.3);

        assert_eq!(config.seed, Some(42));
        assert_eq!(config.n_warmup, 20);
        assert_eq!(config.n_candidates, 50);
        assert_eq!(config.gamma, 0.3);
    }

    #[test]
    fn test_sampler_config_gamma_clamping() {
        let config = SamplerConfig::new().with_gamma(0.8);
        assert_eq!(config.gamma, 0.5); // Clamped to max

        let config = SamplerConfig::new().with_gamma(0.001);
        assert_eq!(config.gamma, 0.01); // Clamped to min
    }

    #[test]
    fn test_sampler_stats() {
        let mut stats = SamplerStats::default();

        stats.record_sample();
        stats.record_sample();
        stats.record_violation();
        stats.record_feedback(10.0);
        stats.record_feedback(15.0);
        stats.record_feedback(5.0);

        assert_eq!(stats.total_samples, 2);
        assert_eq!(stats.constraint_violations, 1);
        assert_eq!(stats.resamples, 1);
        assert_eq!(stats.feedback_count, 3);
        assert_eq!(stats.best_score, Some(15.0));
        assert_eq!(stats.worst_score, Some(5.0));
        assert_eq!(stats.violation_rate(), 0.5);
    }

    #[test]
    fn test_snap_to_step() {
        assert_eq!(snap_to_step(0.27, 0.0, 0.25), 0.25);
        assert_eq!(snap_to_step(0.13, 0.0, 0.25), 0.25);
        assert_eq!(snap_to_step(0.12, 0.0, 0.25), 0.0);
        assert_eq!(snap_to_step(10.3, 10.0, 0.5), 10.5);
    }

    #[test]
    fn test_sample_uniform() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let val = sample_uniform(&mut rng, 0.0, 1.0);
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_sample_log_uniform() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let val = sample_log_uniform(&mut rng, 0.01, 100.0);
            assert!((0.01..=100.0).contains(&val));
        }
    }

    #[test]
    fn test_sample_discrete() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let values = vec![15, 30, 60];
        for _ in 0..100 {
            let val = sample_discrete(&mut rng, &values);
            assert!(val.is_some());
            assert!(values.contains(&val.unwrap()));
        }

        let empty: Vec<i32> = vec![];
        assert!(sample_discrete(&mut rng, &empty).is_none());
    }

    #[test]
    fn test_sample_normal() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Sample many values and check the mean is approximately correct
        let samples: Vec<f64> = (0..1000)
            .map(|_| sample_normal(&mut rng, 0.0, 1.0))
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
    }

    #[test]
    fn test_sample_truncated_normal() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let val = sample_truncated_normal(&mut rng, 0.5, 0.2, 0.0, 1.0);
            assert!(
                (0.0..=1.0).contains(&val),
                "truncated normal out of bounds: {}",
                val
            );
        }
    }

    #[test]
    fn test_silverman_bandwidth_empty() {
        assert_eq!(silverman_bandwidth(&[]), 1.0);
    }

    #[test]
    fn test_silverman_bandwidth_single_value() {
        let bw = silverman_bandwidth(&[1.0]);
        assert!(bw.is_finite());
        // Single value → std_dev = 0 → bandwidth = 0
        assert_eq!(bw, 0.0);
    }

    #[test]
    fn test_silverman_bandwidth_spread() {
        let values: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let bw = silverman_bandwidth(&values);
        assert!(bw > 0.0, "bandwidth should be positive for spread data");
        assert!(bw < 1.0, "bandwidth should be < 1.0 for [0,1) data");
    }

    #[test]
    fn test_average_score() {
        let mut stats = SamplerStats::default();
        assert!(stats.average_score().is_none());

        stats.record_feedback(10.0);
        stats.record_feedback(20.0);
        assert!((stats.average_score().unwrap() - 15.0).abs() < 1e-10);
    }
}
