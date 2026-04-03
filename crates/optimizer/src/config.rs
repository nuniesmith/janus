//! Optimizer Configuration
//!
//! This module defines the configuration options for the JANUS optimizer,
//! including number of trials, timeout, objective weights, and more.

use crate::objective::{ObjectiveFunction, ScoringWeights};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default number of optimization trials
pub const DEFAULT_N_TRIALS: usize = 100;

/// Default timeout per trial in seconds
pub const DEFAULT_TRIAL_TIMEOUT_SECS: u64 = 60;

/// Default minimum data rows required for backtesting
pub const DEFAULT_MIN_DATA_ROWS: usize = 500;

/// Default minimum number of trades for valid backtest
pub const DEFAULT_MIN_TRADES: usize = 10;

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Number of optimization trials to run
    pub n_trials: usize,

    /// Number of parallel jobs (1 = sequential)
    pub n_jobs: usize,

    /// Timeout per trial
    pub trial_timeout: Duration,

    /// Total optimization timeout (None = no limit)
    pub total_timeout: Option<Duration>,

    /// Minimum data rows required
    pub min_data_rows: usize,

    /// Minimum trades required for valid backtest
    pub min_trades: usize,

    /// Objective function for scoring
    #[serde(skip)]
    pub objective: ObjectiveFunction,

    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,

    /// Whether to save intermediate results
    pub save_intermediate: bool,

    /// Directory for saving results
    pub output_dir: Option<String>,

    /// Whether to log verbose output
    pub verbose: bool,

    /// Early stopping: stop if no improvement for N trials
    pub early_stopping_rounds: Option<usize>,

    /// Minimum improvement threshold for early stopping
    pub min_improvement: f64,

    /// Whether to prune unpromising trials
    pub enable_pruning: bool,

    /// Pruning threshold (trials with score below this percentile are pruned)
    pub pruning_percentile: f64,

    /// Slippage in basis points for backtesting
    pub slippage_bps: f64,

    /// Commission in basis points for backtesting
    pub commission_bps: f64,

    /// Initial balance for backtesting
    pub initial_balance: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            n_trials: DEFAULT_N_TRIALS,
            n_jobs: 1,
            trial_timeout: Duration::from_secs(DEFAULT_TRIAL_TIMEOUT_SECS),
            total_timeout: None,
            min_data_rows: DEFAULT_MIN_DATA_ROWS,
            min_trades: DEFAULT_MIN_TRADES,
            objective: ObjectiveFunction::default(),
            seed: None,
            save_intermediate: false,
            output_dir: None,
            verbose: false,
            early_stopping_rounds: Some(20),
            min_improvement: 0.001,
            enable_pruning: false,
            pruning_percentile: 25.0,
            slippage_bps: 5.0,
            commission_bps: 6.0,
            initial_balance: 10_000.0,
        }
    }
}

impl OptimizerConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for constructing config
    pub fn builder() -> OptimizerConfigBuilder {
        OptimizerConfigBuilder::new()
    }

    /// Create config optimized for quick testing
    pub fn quick() -> Self {
        Self {
            n_trials: 20,
            trial_timeout: Duration::from_secs(30),
            early_stopping_rounds: Some(5),
            verbose: true,
            ..Default::default()
        }
    }

    /// Create config for thorough optimization
    pub fn thorough() -> Self {
        Self {
            n_trials: 500,
            n_jobs: num_cpus(),
            trial_timeout: Duration::from_secs(120),
            early_stopping_rounds: Some(50),
            enable_pruning: true,
            ..Default::default()
        }
    }

    /// Create config for production optimization
    pub fn production() -> Self {
        Self {
            n_trials: 200,
            n_jobs: num_cpus(),
            trial_timeout: Duration::from_secs(90),
            total_timeout: Some(Duration::from_secs(3600)), // 1 hour max
            early_stopping_rounds: Some(30),
            enable_pruning: true,
            save_intermediate: true,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.n_trials == 0 {
            return Err("n_trials must be > 0".to_string());
        }
        if self.n_jobs == 0 {
            return Err("n_jobs must be > 0".to_string());
        }
        if self.min_data_rows < 100 {
            return Err("min_data_rows must be >= 100".to_string());
        }
        if self.min_trades == 0 {
            return Err("min_trades must be > 0".to_string());
        }
        if self.pruning_percentile < 0.0 || self.pruning_percentile > 100.0 {
            return Err("pruning_percentile must be in [0, 100]".to_string());
        }
        if self.slippage_bps < 0.0 {
            return Err("slippage_bps must be >= 0".to_string());
        }
        if self.commission_bps < 0.0 {
            return Err("commission_bps must be >= 0".to_string());
        }
        if self.initial_balance <= 0.0 {
            return Err("initial_balance must be > 0".to_string());
        }
        Ok(())
    }
}

/// Builder for OptimizerConfig
#[derive(Debug, Clone)]
pub struct OptimizerConfigBuilder {
    config: OptimizerConfig,
}

impl OptimizerConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
        }
    }

    /// Set the number of trials
    pub fn n_trials(mut self, n: usize) -> Self {
        self.config.n_trials = n;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n: usize) -> Self {
        self.config.n_jobs = n;
        self
    }

    /// Set trial timeout
    pub fn trial_timeout(mut self, duration: Duration) -> Self {
        self.config.trial_timeout = duration;
        self
    }

    /// Set trial timeout in seconds
    pub fn trial_timeout_secs(mut self, secs: u64) -> Self {
        self.config.trial_timeout = Duration::from_secs(secs);
        self
    }

    /// Set total optimization timeout
    pub fn total_timeout(mut self, duration: Duration) -> Self {
        self.config.total_timeout = Some(duration);
        self
    }

    /// Set total timeout in seconds
    pub fn total_timeout_secs(mut self, secs: u64) -> Self {
        self.config.total_timeout = Some(Duration::from_secs(secs));
        self
    }

    /// Set minimum data rows required
    pub fn min_data_rows(mut self, n: usize) -> Self {
        self.config.min_data_rows = n;
        self
    }

    /// Set minimum trades required
    pub fn min_trades(mut self, n: usize) -> Self {
        self.config.min_trades = n;
        self
    }

    /// Set the objective function
    pub fn objective(mut self, objective: ObjectiveFunction) -> Self {
        self.config.objective = objective;
        self
    }

    /// Set scoring weights for the objective
    pub fn scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.config.objective = ObjectiveFunction::new(weights);
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Enable saving intermediate results
    pub fn save_intermediate(mut self, save: bool) -> Self {
        self.config.save_intermediate = save;
        self
    }

    /// Set output directory
    pub fn output_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.output_dir = Some(dir.into());
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set early stopping rounds
    pub fn early_stopping(mut self, rounds: usize) -> Self {
        self.config.early_stopping_rounds = Some(rounds);
        self
    }

    /// Disable early stopping
    pub fn no_early_stopping(mut self) -> Self {
        self.config.early_stopping_rounds = None;
        self
    }

    /// Set minimum improvement threshold
    pub fn min_improvement(mut self, threshold: f64) -> Self {
        self.config.min_improvement = threshold;
        self
    }

    /// Enable trial pruning
    pub fn enable_pruning(mut self, enable: bool) -> Self {
        self.config.enable_pruning = enable;
        self
    }

    /// Set pruning percentile
    pub fn pruning_percentile(mut self, percentile: f64) -> Self {
        self.config.pruning_percentile = percentile;
        self
    }

    /// Set slippage in basis points
    pub fn slippage_bps(mut self, bps: f64) -> Self {
        self.config.slippage_bps = bps;
        self
    }

    /// Set commission in basis points
    pub fn commission_bps(mut self, bps: f64) -> Self {
        self.config.commission_bps = bps;
        self
    }

    /// Set initial balance
    pub fn initial_balance(mut self, balance: f64) -> Self {
        self.config.initial_balance = balance;
        self
    }

    /// Build the config, returning an error if invalid
    pub fn build(self) -> Result<OptimizerConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build the config, panicking if invalid
    pub fn build_unchecked(self) -> OptimizerConfig {
        self.config
    }
}

impl Default for OptimizerConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Get number of CPU cores
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OptimizerConfig::default();
        assert_eq!(config.n_trials, DEFAULT_N_TRIALS);
        assert_eq!(config.n_jobs, 1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quick_config() {
        let config = OptimizerConfig::quick();
        assert_eq!(config.n_trials, 20);
        assert!(config.verbose);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_thorough_config() {
        let config = OptimizerConfig::thorough();
        assert_eq!(config.n_trials, 500);
        assert!(config.enable_pruning);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_production_config() {
        let config = OptimizerConfig::production();
        assert!(config.total_timeout.is_some());
        assert!(config.save_intermediate);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let config = OptimizerConfig::builder()
            .n_trials(50)
            .n_jobs(4)
            .seed(42)
            .verbose(true)
            .slippage_bps(10.0)
            .build()
            .unwrap();

        assert_eq!(config.n_trials, 50);
        assert_eq!(config.n_jobs, 4);
        assert_eq!(config.seed, Some(42));
        assert!(config.verbose);
        assert_eq!(config.slippage_bps, 10.0);
    }

    #[test]
    fn test_builder_validation_error() {
        let result = OptimizerConfig::builder()
            .n_trials(0) // Invalid
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("n_trials"));
    }

    #[test]
    fn test_builder_timeout() {
        let config = OptimizerConfig::builder()
            .trial_timeout_secs(120)
            .total_timeout_secs(3600)
            .build()
            .unwrap();

        assert_eq!(config.trial_timeout, Duration::from_secs(120));
        assert_eq!(config.total_timeout, Some(Duration::from_secs(3600)));
    }

    #[test]
    fn test_validation_errors() {
        let mut config = OptimizerConfig {
            n_trials: 0,
            ..Default::default()
        };

        assert!(config.validate().is_err());
        config.n_trials = 100;

        config.n_jobs = 0;
        assert!(config.validate().is_err());
        config.n_jobs = 1;

        config.min_data_rows = 50;
        assert!(config.validate().is_err());
        config.min_data_rows = 500;

        config.pruning_percentile = 150.0;
        assert!(config.validate().is_err());
        config.pruning_percentile = 25.0;

        config.slippage_bps = -5.0;
        assert!(config.validate().is_err());
        config.slippage_bps = 5.0;

        config.initial_balance = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_early_stopping_config() {
        let config = OptimizerConfig::builder()
            .early_stopping(10)
            .min_improvement(0.01)
            .build()
            .unwrap();

        assert_eq!(config.early_stopping_rounds, Some(10));
        assert_eq!(config.min_improvement, 0.01);

        let config = OptimizerConfig::builder()
            .no_early_stopping()
            .build()
            .unwrap();

        assert_eq!(config.early_stopping_rounds, None);
    }

    #[test]
    fn test_output_dir() {
        let config = OptimizerConfig::builder()
            .output_dir("/tmp/optimizer")
            .save_intermediate(true)
            .build()
            .unwrap();

        assert_eq!(config.output_dir, Some("/tmp/optimizer".to_string()));
        assert!(config.save_intermediate);
    }
}
