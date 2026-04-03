//! Search Algorithm Implementations
//!
//! This module provides concrete implementations of the `Sampler` trait:
//! - `RandomSearch` - Uniform random sampling from the search space
//! - `GridSearch` - Exhaustive grid search over discretized parameters
//! - `TpeSampler` - Tree-structured Parzen Estimator for Bayesian optimization

use crate::constraints::{SampledParams, SearchSpace};
use crate::error::{OptimizerError, Result};
use crate::results::TrialResult;
use crate::sampler::{
    Sampler, SamplerConfig, SamplerStats, SamplerType, sample_normal, sample_uniform,
    silverman_bandwidth, snap_to_step,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;

/// Maximum resampling attempts before giving up
const MAX_RESAMPLE_ATTEMPTS: usize = 100;

// ============================================================================
// RandomSearch
// ============================================================================

/// Random search sampler - uniformly samples from the search space
#[derive(Debug, Clone)]
pub struct RandomSearch {
    /// Configuration
    config: SamplerConfig,

    /// Random number generator
    rng: SmallRng,

    /// Statistics
    stats: SamplerStats,
}

impl Default for RandomSearch {
    fn default() -> Self {
        Self::new(SamplerConfig::default())
    }
}

impl RandomSearch {
    /// Create a new random search sampler
    pub fn new(config: SamplerConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };

        Self {
            config,
            rng,
            stats: SamplerStats::default(),
        }
    }

    /// Create with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self::new(SamplerConfig::default().with_seed(seed))
    }

    /// Get sampler statistics
    pub fn stats(&self) -> &SamplerStats {
        &self.stats
    }

    /// Sample a single parameter value
    fn sample_parameter(
        &mut self,
        _name: &str,
        bounds: &crate::constraints::ParameterBounds,
    ) -> f64 {
        let min = bounds.effective_min();
        let max = bounds.effective_max();

        let value = sample_uniform(&mut self.rng, min, max);

        // Snap to step if discrete
        if let Some(step) = bounds.step {
            snap_to_step(value, min, step)
        } else {
            value
        }
    }
}

impl Sampler for RandomSearch {
    fn sample(&self, space: &SearchSpace, _history: &[TrialResult]) -> Result<SampledParams> {
        // Clone self to get mutable access to rng
        let mut sampler = self.clone();
        sampler.sample_mut(space)
    }

    fn tell(&mut self, _params: &SampledParams, score: f64) -> Result<()> {
        self.stats.record_feedback(score);
        Ok(())
    }

    fn reset(&mut self) {
        self.stats.reset();
        // Reinitialize RNG
        self.rng = match self.config.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Random
    }
}

impl RandomSearch {
    /// Mutable version of sample for internal use
    fn sample_mut(&mut self, space: &SearchSpace) -> Result<SampledParams> {
        for attempt in 0..MAX_RESAMPLE_ATTEMPTS {
            self.stats.record_sample();

            let mut values = HashMap::new();

            // Sample each parameter
            for (name, bounds) in space.parameters() {
                let value = self.sample_parameter(name, bounds);
                values.insert(name.clone(), value);
            }

            let params = SampledParams::new(values);

            // Validate against constraints
            match space.validate(&params.values) {
                Ok(()) => return Ok(params),
                Err(_) => {
                    self.stats.record_violation();
                    if attempt == MAX_RESAMPLE_ATTEMPTS - 1 {
                        return Err(OptimizerError::SamplerError(
                            "Max resample attempts exceeded".to_string(),
                        ));
                    }
                }
            }
        }

        Err(OptimizerError::SamplerError(
            "Failed to sample valid parameters".to_string(),
        ))
    }
}

// ============================================================================
// GridSearch
// ============================================================================

/// Grid search sampler - exhaustively searches a discretized grid
#[derive(Debug, Clone)]
pub struct GridSearch {
    /// Configuration
    #[allow(dead_code)]
    config: SamplerConfig,

    /// Current grid index
    current_index: usize,

    /// Pre-computed grid points
    grid_points: Vec<HashMap<String, f64>>,

    /// Whether the grid has been initialized
    initialized: bool,

    /// Statistics
    stats: SamplerStats,
}

impl Default for GridSearch {
    fn default() -> Self {
        Self::new(SamplerConfig::default())
    }
}

impl GridSearch {
    /// Create a new grid search sampler
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            current_index: 0,
            grid_points: Vec::new(),
            initialized: false,
            stats: SamplerStats::default(),
        }
    }

    /// Get the total number of grid points
    pub fn grid_size(&self) -> usize {
        self.grid_points.len()
    }

    /// Get current progress through the grid
    pub fn progress(&self) -> (usize, usize) {
        (self.current_index, self.grid_points.len())
    }

    /// Initialize the grid from a search space
    fn initialize_grid(&mut self, space: &SearchSpace) {
        if self.initialized {
            return;
        }

        // Generate grid points for each parameter
        let mut param_grids: Vec<(String, Vec<f64>)> = Vec::new();

        for (name, bounds) in space.parameters() {
            let min = bounds.effective_min();
            let max = bounds.effective_max();

            let points: Vec<f64> = if let Some(step) = bounds.step {
                // Discrete parameter: use all valid steps
                let mut pts = Vec::new();
                let mut v = min;
                while v <= max {
                    pts.push(v);
                    v += step;
                }
                pts
            } else {
                // Continuous parameter: discretize into ~10 points
                let n_points = 10;
                (0..=n_points)
                    .map(|i| min + (max - min) * (i as f64 / n_points as f64))
                    .collect()
            };

            param_grids.push((name.clone(), points));
        }

        // Generate all combinations (Cartesian product)
        self.grid_points = Self::cartesian_product(&param_grids);
        self.initialized = true;

        tracing::debug!(
            grid_size = self.grid_points.len(),
            "Initialized grid search"
        );
    }

    /// Compute Cartesian product of parameter grids
    fn cartesian_product(grids: &[(String, Vec<f64>)]) -> Vec<HashMap<String, f64>> {
        if grids.is_empty() {
            return vec![HashMap::new()];
        }

        let (name, values) = &grids[0];
        let rest = Self::cartesian_product(&grids[1..]);

        let mut result = Vec::new();
        for value in values {
            for mut combo in rest.clone() {
                combo.insert(name.clone(), *value);
                result.push(combo);
            }
        }

        result
    }
}

impl Sampler for GridSearch {
    fn sample(&self, space: &SearchSpace, _history: &[TrialResult]) -> Result<SampledParams> {
        // Clone self to get mutable access
        let mut sampler = self.clone();
        sampler.sample_mut(space)
    }

    fn tell(&mut self, _params: &SampledParams, score: f64) -> Result<()> {
        self.stats.record_feedback(score);
        Ok(())
    }

    fn reset(&mut self) {
        self.current_index = 0;
        self.grid_points.clear();
        self.initialized = false;
        self.stats.reset();
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Grid
    }
}

impl GridSearch {
    fn sample_mut(&mut self, space: &SearchSpace) -> Result<SampledParams> {
        self.initialize_grid(space);
        self.stats.record_sample();

        if self.grid_points.is_empty() {
            return Err(OptimizerError::InvalidSearchSpace(
                "Grid is empty".to_string(),
            ));
        }

        // Get next grid point (wrap around if needed)
        let values = self.grid_points[self.current_index % self.grid_points.len()].clone();
        self.current_index += 1;

        Ok(SampledParams::new(values))
    }
}

// ============================================================================
// TpeSampler (Tree-structured Parzen Estimator)
// ============================================================================

/// TPE (Tree-structured Parzen Estimator) sampler for Bayesian optimization
///
/// TPE is an adaptive sampler that models the distribution of good and bad
/// parameters separately, then samples from regions more likely to be good.
///
/// Algorithm:
/// 1. Split observed trials into "good" (top gamma fraction) and "bad"
/// 2. Fit kernel density estimates (KDE) to each group
/// 3. Sample candidates and score them by l(x)/g(x) ratio
/// 4. Return the candidate with the highest ratio
#[derive(Debug, Clone)]
pub struct TpeSampler {
    /// Configuration
    config: SamplerConfig,

    /// Random number generator
    rng: SmallRng,

    /// Observed parameter-score pairs
    observations: Vec<(HashMap<String, f64>, f64)>,

    /// Statistics
    stats: SamplerStats,
}

impl Default for TpeSampler {
    fn default() -> Self {
        Self::new(SamplerConfig::default())
    }
}

impl TpeSampler {
    /// Create a new TPE sampler
    pub fn new(config: SamplerConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };

        Self {
            config,
            rng,
            observations: Vec::new(),
            stats: SamplerStats::default(),
        }
    }

    /// Create with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self::new(SamplerConfig::default().with_seed(seed))
    }

    /// Get the number of observations
    pub fn n_observations(&self) -> usize {
        self.observations.len()
    }

    /// Get sampler statistics
    pub fn stats(&self) -> &SamplerStats {
        &self.stats
    }

    /// Check if we're still in warmup phase
    fn in_warmup(&self) -> bool {
        self.observations.len() < self.config.n_warmup
    }

    /// Sample using random search (during warmup)
    fn sample_random(&mut self, space: &SearchSpace) -> Result<SampledParams> {
        let mut values = HashMap::new();

        for (name, bounds) in space.parameters() {
            let min = bounds.effective_min();
            let max = bounds.effective_max();
            let value = sample_uniform(&mut self.rng, min, max);

            let final_value = if let Some(step) = bounds.step {
                snap_to_step(value, min, step)
            } else {
                value
            };

            values.insert(name.clone(), final_value);
        }

        Ok(SampledParams::new(values))
    }

    /// Sample using TPE algorithm
    fn sample_tpe(&mut self, space: &SearchSpace) -> Result<SampledParams> {
        // Split observations into good and bad
        let n_good = ((self.observations.len() as f64) * self.config.gamma).max(1.0) as usize;

        // Sort by score (descending - higher is better)
        let mut sorted_obs = self.observations.clone();
        sorted_obs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let good_obs: Vec<_> = sorted_obs.iter().take(n_good).map(|(p, _)| p).collect();
        let bad_obs: Vec<_> = sorted_obs.iter().skip(n_good).map(|(p, _)| p).collect();

        // Generate candidates and score them
        let mut best_candidate: Option<HashMap<String, f64>> = None;
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..self.config.n_candidates {
            let candidate = self.generate_candidate(space, &good_obs)?;

            // Calculate EI (Expected Improvement) proxy: l(x) / g(x)
            let l_score = self.kde_score(&candidate, &good_obs, space);
            let g_score = self.kde_score(&candidate, &bad_obs, space);

            let ei_score = if g_score > 1e-10 {
                l_score / g_score
            } else {
                l_score
            };

            if ei_score > best_score {
                best_score = ei_score;
                best_candidate = Some(candidate);
            }
        }

        match best_candidate {
            Some(values) => Ok(SampledParams::new(values)),
            None => self.sample_random(space), // Fallback to random
        }
    }

    /// Generate a candidate by sampling around good observations.
    ///
    /// Uses proper Gaussian noise (Box-Muller) and Silverman's rule-of-thumb
    /// bandwidth instead of uniform noise with a fixed 10% range. This
    /// improvement was ported from Kraken's TPE sampler and produces better
    /// exploitation behavior — samples concentrate near good observations
    /// rather than spreading uniformly across the perturbation window.
    fn generate_candidate(
        &mut self,
        space: &SearchSpace,
        good_obs: &[&HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        let mut values = HashMap::new();

        for (name, bounds) in space.parameters() {
            let min = bounds.effective_min();
            let max = bounds.effective_max();

            // Get values from good observations for this parameter
            let good_values: Vec<f64> = good_obs
                .iter()
                .filter_map(|obs| obs.get(name).copied())
                .collect();

            let value = if good_values.is_empty() || self.rng.random_bool(0.1) {
                // 10% exploration: sample uniformly
                sample_uniform(&mut self.rng, min, max)
            } else {
                // 90% exploitation: sample around a good value with Gaussian noise
                let idx = self.rng.random_range(0..good_values.len());
                let center = good_values[idx];

                // Use Silverman's rule for data-adaptive bandwidth when we have
                // enough observations; fall back to 10% of range otherwise.
                // This narrows the search as observations accumulate (more
                // exploitative) while staying wide early on (more exploratory).
                let bandwidth = if good_values.len() > 1 {
                    silverman_bandwidth(&good_values).max((max - min) * 0.05)
                } else {
                    (max - min) * 0.1
                };

                // Gaussian perturbation (Box-Muller) instead of uniform noise.
                // Gaussian concentrates samples near `center` rather than
                // spreading them evenly across `[center - bw, center + bw]`.
                sample_normal(&mut self.rng, center, bandwidth).clamp(min, max)
            };

            let final_value = if let Some(step) = bounds.step {
                snap_to_step(value, min, step).clamp(min, max)
            } else {
                value
            };

            values.insert(name.clone(), final_value);
        }

        Ok(values)
    }

    /// Calculate KDE score for a candidate given observations.
    ///
    /// Uses Silverman's rule bandwidth when enough per-parameter observations
    /// exist, falling back to 10% of range otherwise. This makes the density
    /// estimate adapt to the actual spread of the observations rather than
    /// relying on a fixed fraction of the search range.
    fn kde_score(
        &self,
        candidate: &HashMap<String, f64>,
        observations: &[&HashMap<String, f64>],
        space: &SearchSpace,
    ) -> f64 {
        if observations.is_empty() {
            return 1.0;
        }

        let mut total_density = 0.0;

        for obs in observations {
            let mut param_density = 1.0;

            for (name, bounds) in space.parameters() {
                let range = bounds.effective_max() - bounds.effective_min();

                // Compute data-adaptive bandwidth per parameter via Silverman's
                // rule when we have > 1 observation; otherwise fall back to 10%.
                let obs_values: Vec<f64> = observations
                    .iter()
                    .filter_map(|o| o.get(name).copied())
                    .collect();

                let bandwidth = if obs_values.len() > 1 {
                    silverman_bandwidth(&obs_values).max(range * 0.05)
                } else {
                    range * 0.1
                };

                if let (Some(cand_val), Some(obs_val)) = (candidate.get(name), obs.get(name)) {
                    // Gaussian kernel
                    let diff = (cand_val - obs_val) / bandwidth;
                    let kernel = (-0.5 * diff * diff).exp();
                    param_density *= kernel;
                }
            }

            total_density += param_density;
        }

        total_density / observations.len() as f64
    }
}

impl Sampler for TpeSampler {
    fn sample(&self, space: &SearchSpace, _history: &[TrialResult]) -> Result<SampledParams> {
        let mut sampler = self.clone();
        sampler.sample_mut(space)
    }

    fn tell(&mut self, params: &SampledParams, score: f64) -> Result<()> {
        self.observations.push((params.values.clone(), score));
        self.stats.record_feedback(score);
        Ok(())
    }

    fn reset(&mut self) {
        self.observations.clear();
        self.stats.reset();
        self.rng = match self.config.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };
    }

    fn sampler_type(&self) -> SamplerType {
        SamplerType::Tpe
    }
}

impl TpeSampler {
    fn sample_mut(&mut self, space: &SearchSpace) -> Result<SampledParams> {
        self.stats.record_sample();

        // Use random sampling during warmup
        if self.in_warmup() {
            return self.sample_random(space);
        }

        // Use TPE after warmup
        for attempt in 0..MAX_RESAMPLE_ATTEMPTS {
            let params = self.sample_tpe(space)?;

            // Validate against constraints
            match space.validate(&params.values) {
                Ok(()) => return Ok(params),
                Err(_) => {
                    self.stats.record_violation();
                    if attempt == MAX_RESAMPLE_ATTEMPTS - 1 {
                        // Fall back to random
                        return self.sample_random(space);
                    }
                }
            }
        }

        self.sample_random(space)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::ParameterBounds;

    fn create_test_space() -> SearchSpace {
        let mut space = SearchSpace::new();
        space.add_parameter("x", ParameterBounds::continuous(0.0, 10.0));
        space.add_parameter("y", ParameterBounds::integer(1, 5));
        space.add_parameter("z", ParameterBounds::discrete(0.0, 1.0, 0.25));
        space
    }

    // RandomSearch tests
    #[test]
    fn test_random_search_creation() {
        let sampler = RandomSearch::default();
        assert_eq!(sampler.sampler_type(), SamplerType::Random);
    }

    #[test]
    fn test_random_search_with_seed() {
        let sampler1 = RandomSearch::with_seed(42);
        let sampler2 = RandomSearch::with_seed(42);

        let space = create_test_space();

        // Same seed should produce same results
        let params1 = sampler1.sample(&space, &[]).unwrap();
        let params2 = sampler2.sample(&space, &[]).unwrap();

        assert_eq!(params1.values, params2.values);
    }

    #[test]
    fn test_random_search_samples_in_bounds() {
        let mut sampler = RandomSearch::with_seed(42);
        let space = create_test_space();

        for _ in 0..100 {
            let params = sampler.sample_mut(&space).unwrap();

            let x = params.get("x").unwrap();
            let y = params.get("y").unwrap();
            let z = params.get("z").unwrap();

            assert!((0.0..=10.0).contains(&x), "x out of bounds: {}", x);
            assert!((1.0..=5.0).contains(&y), "y out of bounds: {}", y);
            assert!((0.0..=1.0).contains(&z), "z out of bounds: {}", z);

            // y should be integer
            assert_eq!(y, y.floor(), "y should be integer: {}", y);

            // z should be snapped to 0.25 steps
            let valid_z = [0.0, 0.25, 0.5, 0.75, 1.0];
            assert!(
                valid_z.iter().any(|&v| (z - v).abs() < 1e-10),
                "z should be discrete: {}",
                z
            );
        }
    }

    // GridSearch tests
    #[test]
    fn test_grid_search_creation() {
        let sampler = GridSearch::default();
        assert_eq!(sampler.sampler_type(), SamplerType::Grid);
    }

    #[test]
    fn test_grid_search_initialization() {
        let mut sampler = GridSearch::default();
        let space = create_test_space();

        // First sample initializes the grid
        let _ = sampler.sample_mut(&space).unwrap();

        assert!(sampler.initialized);
        assert!(sampler.grid_size() > 0);
    }

    #[test]
    fn test_grid_search_exhaustive() {
        let mut sampler = GridSearch::default();
        let mut space = SearchSpace::new();
        space.add_parameter("a", ParameterBounds::discrete(0.0, 1.0, 0.5)); // 3 values
        space.add_parameter("b", ParameterBounds::discrete(0.0, 1.0, 0.5)); // 3 values

        // Should have 9 grid points (3x3)
        let _ = sampler.sample_mut(&space).unwrap();
        assert_eq!(sampler.grid_size(), 9);
    }

    #[test]
    fn test_grid_search_progress() {
        let mut sampler = GridSearch::default();
        let mut space = SearchSpace::new();
        space.add_parameter("a", ParameterBounds::discrete(0.0, 1.0, 0.5));

        let _ = sampler.sample_mut(&space).unwrap();
        assert_eq!(sampler.progress(), (1, 3));

        let _ = sampler.sample_mut(&space).unwrap();
        assert_eq!(sampler.progress(), (2, 3));
    }

    // TpeSampler tests
    #[test]
    fn test_tpe_sampler_creation() {
        let sampler = TpeSampler::default();
        assert_eq!(sampler.sampler_type(), SamplerType::Tpe);
        assert!(sampler.is_adaptive());
    }

    #[test]
    fn test_tpe_sampler_warmup() {
        let mut sampler = TpeSampler::new(SamplerConfig::default().with_warmup(5));
        let space = create_test_space();

        // During warmup, should use random sampling
        assert!(sampler.in_warmup());

        for i in 0..5 {
            let params = sampler.sample_mut(&space).unwrap();
            sampler.tell(&params, i as f64 * 10.0).unwrap();
        }

        // After 5 observations, should no longer be in warmup
        assert!(!sampler.in_warmup());
    }

    #[test]
    fn test_tpe_sampler_observations() {
        let mut sampler = TpeSampler::default();
        let space = create_test_space();

        assert_eq!(sampler.n_observations(), 0);

        let params = sampler.sample_mut(&space).unwrap();
        sampler.tell(&params, 10.0).unwrap();

        assert_eq!(sampler.n_observations(), 1);
    }

    #[test]
    fn test_tpe_sampler_samples_in_bounds() {
        let mut sampler = TpeSampler::with_seed(42);
        let space = create_test_space();

        // Add some observations to exit warmup
        for i in 0..15 {
            let params = sampler.sample_mut(&space).unwrap();
            sampler.tell(&params, i as f64).unwrap();
        }

        // Now sample using TPE
        for _ in 0..50 {
            let params = sampler.sample_mut(&space).unwrap();

            let x = params.get("x").unwrap();
            let y = params.get("y").unwrap();
            let z = params.get("z").unwrap();

            assert!((0.0..=10.0).contains(&x), "x out of bounds: {}", x);
            assert!((1.0..=5.0).contains(&y), "y out of bounds: {}", y);
            assert!((0.0..=1.0).contains(&z), "z out of bounds: {}", z);
        }
    }

    #[test]
    fn test_tpe_sampler_reset() {
        let mut sampler = TpeSampler::with_seed(42);
        let space = create_test_space();

        let params = sampler.sample_mut(&space).unwrap();
        sampler.tell(&params, 10.0).unwrap();

        assert_eq!(sampler.n_observations(), 1);

        sampler.reset();

        assert_eq!(sampler.n_observations(), 0);
    }

    // Sampler trait tests
    #[test]
    fn test_sampler_names() {
        assert_eq!(RandomSearch::default().name(), "RandomSearch");
        assert_eq!(GridSearch::default().name(), "GridSearch");
        assert_eq!(TpeSampler::default().name(), "TPESampler");
    }

    #[test]
    fn test_sampler_warmup_trials() {
        assert_eq!(RandomSearch::default().warmup_trials(), 0);
        assert_eq!(GridSearch::default().warmup_trials(), 0);
        assert_eq!(TpeSampler::default().warmup_trials(), 10);
    }

    #[test]
    fn test_sampler_is_adaptive() {
        assert!(!RandomSearch::default().is_adaptive());
        assert!(!GridSearch::default().is_adaptive());
        assert!(TpeSampler::default().is_adaptive());
    }
}
