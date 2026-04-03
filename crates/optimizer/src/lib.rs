//! # JANUS Optimizer
//!
//! Hyperparameter optimization for JANUS trading strategies.
//!
//! This crate provides a Rust-native parameter optimization framework that:
//! - Uses the existing `janus-backtest` vectorized indicators for fast evaluation
//! - Implements TPE (Tree-structured Parzen Estimator) and other search algorithms
//! - Enforces per-asset parameter constraints (major/alt/meme coin floors)
//! - Publishes optimized parameters to Redis for hot-reload by the Forward service
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      JANUS Optimizer                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
//! │  │ OHLC Data    │───▶│ Backtester   │───▶│ Scoring         │   │
//! │  │ (Polars DF)  │    │ (Vectorized) │    │ Function        │   │
//! │  └──────────────┘    └──────────────┘    └────────┬────────┘   │
//! │                                                    │            │
//! │  ┌──────────────────────────────────────────────────▼────────┐  │
//! │  │                    Search Algorithm                       │  │
//! │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │  │
//! │  │  │ Random  │  │  Grid   │  │  TPE    │  │ Evolutionary│  │  │
//! │  │  │ Search  │  │ Search  │  │ Sampler │  │  Strategy   │  │  │
//! │  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘  │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! │                              │                                   │
//! │                              ▼                                   │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │ OptimizedParams → Redis → Forward Service (hot-reload)    │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_optimizer::{
//!     Optimizer, OptimizerConfig, SearchSpace, AssetClass,
//!     search::TpeSampler,
//! };
//! use polars::prelude::*;
//!
//! // Load historical data
//! let df = LazyFrame::scan_parquet("data/btc_1h.parquet", Default::default())?
//!     .collect()?;
//!
//! // Define search space with asset-specific constraints
//! let search_space = SearchSpace::for_asset("BTC", AssetClass::Major);
//!
//! // Configure optimizer
//! let config = OptimizerConfig::builder()
//!     .n_trials(100)
//!     .n_jobs(4)  // Parallel trials
//!     .build();
//!
//! // Run optimization
//! let optimizer = Optimizer::new(config, TpeSampler::default());
//! let result = optimizer.optimize("BTC", &df, &search_space).await?;
//!
//! println!("Best params: {:?}", result.best_params);
//! println!("Best score: {:.2}", result.best_score);
//!
//! // Publish to Redis for Forward service
//! result.publish_to_redis("redis://localhost:6379").await?;
//! ```
//!
//! ## Per-Asset Constraints
//!
//! Different asset classes have different volatility profiles and require
//! different parameter floors to prevent whipsaw trades:
//!
//! | Asset Class | Min EMA Spread | Min Hold Time | Examples |
//! |-------------|----------------|---------------|----------|
//! | Major       | 0.15%          | 15 min        | BTC, ETH, SOL |
//! | Altcoin     | 0.20%          | 20 min        | LINK, DOT, AVAX |
//! | Meme        | 0.30%          | 30 min        | DOGE, SHIB, PEPE |
//! | DeFi        | 0.20%          | 20 min        | UNI, AAVE, CRV |
//!
//! ## Features
//!
//! - `redis-publish` - Enable Redis publishing of optimized params
//! - `parallel` - Enable parallel optimization with Rayon

pub mod asset;
pub mod backtester;
pub mod config;
pub mod constraints;
pub mod error;
pub mod objective;
pub mod publisher;
pub mod results;
pub mod sampler;
pub mod search;

// Re-export main types
pub use asset::{AssetCategory, AssetConfig, AssetRegistry};
pub use backtester::{BacktestEngine, BacktestParams, BacktestResult};
pub use config::{OptimizerConfig, OptimizerConfigBuilder};
pub use constraints::{Constraint, ParameterBounds, SearchSpace};
pub use error::{OptimizerError, Result};
pub use objective::{ObjectiveFunction, ScoringWeights};
pub use publisher::{BatchPublishResult, ParamPublisher};
pub use results::{OptimizationResult, TrialResult};
pub use sampler::{Sampler, SamplerType};
pub use search::{GridSearch, RandomSearch, TpeSampler};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Version of the optimizer crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Main optimizer struct that coordinates the optimization process
pub struct Optimizer<S: Sampler> {
    /// Configuration
    config: OptimizerConfig,

    /// Parameter sampler (search algorithm)
    sampler: S,

    /// Asset registry for constraints
    asset_registry: AssetRegistry,

    /// Trial history for the current optimization
    trial_history: Arc<RwLock<Vec<TrialResult>>>,
}

impl<S: Sampler> Optimizer<S> {
    /// Create a new optimizer with the given configuration and sampler
    pub fn new(config: OptimizerConfig, sampler: S) -> Self {
        Self {
            config,
            sampler,
            asset_registry: AssetRegistry::default(),
            trial_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create optimizer with custom asset registry
    pub fn with_registry(mut self, registry: AssetRegistry) -> Self {
        self.asset_registry = registry;
        self
    }

    /// Run optimization for a single asset
    pub async fn optimize(
        &mut self,
        asset: &str,
        data: &polars::frame::DataFrame,
        search_space: &SearchSpace,
    ) -> Result<OptimizationResult> {
        info!(
            asset = asset,
            trials = self.config.n_trials,
            "Starting optimization"
        );

        let start_time = Utc::now();
        let mut best_result: Option<TrialResult> = None;
        let mut all_trials = Vec::with_capacity(self.config.n_trials);

        // Get asset-specific constraints
        let asset_config = self.asset_registry.get(asset);
        let constrained_space = search_space.clone().with_asset_constraints(&asset_config);

        for trial_num in 0..self.config.n_trials {
            // Sample parameters
            let params = self.sampler.sample(&constrained_space, &all_trials)?;

            // Run backtest
            let backtest_params = BacktestParams::from_sampled(&params);
            let engine = BacktestEngine::new(backtest_params.clone());
            let backtest_result = engine.run(data)?;

            // Calculate objective score
            let score = self.config.objective.score(&backtest_result);

            // Create trial result
            let trial = TrialResult {
                trial_number: trial_num,
                params: params.clone(),
                backtest_result: backtest_result.clone(),
                score,
                timestamp: Utc::now(),
            };

            debug!(
                trial = trial_num,
                score = score,
                trades = backtest_result.total_trades,
                pnl = backtest_result.total_pnl_pct,
                "Trial completed"
            );

            // Update best
            if best_result.as_ref().is_none_or(|b| score > b.score) {
                best_result = Some(trial.clone());
                info!(trial = trial_num, score = score, "New best found");
            }

            all_trials.push(trial);

            // Notify sampler of result (for adaptive samplers like TPE)
            self.sampler.tell(&params, score)?;
        }

        let best = best_result.ok_or(OptimizerError::NoTrialsCompleted)?;
        let duration = (Utc::now() - start_time).num_seconds() as f64;

        // Build final result
        let result = OptimizationResult {
            asset: asset.to_string(),
            best_params: best.params.clone(),
            best_score: best.score,
            best_backtest: best.backtest_result.clone(),
            all_trials,
            total_trials: self.config.n_trials,
            duration_seconds: duration,
            started_at: start_time,
            completed_at: Utc::now(),
            config: self.config.clone(),
        };

        info!(
            asset = asset,
            best_score = result.best_score,
            total_pnl = result.best_backtest.total_pnl_pct,
            win_rate = result.best_backtest.win_rate,
            trades = result.best_backtest.total_trades,
            duration_secs = duration,
            "Optimization completed"
        );

        Ok(result)
    }

    /// Run optimization for multiple assets in parallel
    #[cfg(feature = "parallel")]
    pub async fn optimize_multiple(
        &mut self,
        assets: &[&str],
        data_map: &HashMap<String, polars::frame::DataFrame>,
    ) -> Result<Vec<OptimizationResult>> {
        use rayon::prelude::*;

        let results: Vec<Result<OptimizationResult>> = assets
            .par_iter()
            .map(|asset| {
                let data = data_map
                    .get(*asset)
                    .ok_or_else(|| OptimizerError::MissingData(asset.to_string()))?;

                let search_space =
                    SearchSpace::for_asset(asset, self.asset_registry.get(asset).category);

                // Note: This is simplified - actual parallel impl needs more work
                // to handle async properly with rayon
                tokio::runtime::Handle::current().block_on(async {
                    let mut optimizer = Optimizer::new(self.config.clone(), S::default());
                    optimizer.optimize(asset, data, &search_space).await
                })
            })
            .collect();

        results.into_iter().collect()
    }

    /// Get trial history
    pub async fn trial_history(&self) -> Vec<TrialResult> {
        self.trial_history.read().await.clone()
    }

    /// Reset optimizer state for new optimization run
    pub async fn reset(&mut self) {
        self.trial_history.write().await.clear();
        self.sampler.reset();
    }
}

/// Optimizer metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerMetadata {
    /// Crate version
    pub version: String,

    /// Number of available CPU cores
    pub cpu_cores: usize,

    /// Whether parallel feature is enabled
    pub parallel_enabled: bool,

    /// Whether Redis publishing is enabled
    pub redis_enabled: bool,

    /// Build timestamp
    pub build_timestamp: DateTime<Utc>,
}

impl OptimizerMetadata {
    /// Create metadata for the current optimizer
    pub fn current() -> Self {
        Self {
            version: VERSION.to_string(),
            cpu_cores: num_cpus(),
            parallel_enabled: cfg!(feature = "parallel"),
            redis_enabled: cfg!(feature = "redis-publish"),
            build_timestamp: Utc::now(),
        }
    }
}

/// Get number of CPU cores
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Initialize the optimizer with default configuration
pub fn init() -> Optimizer<RandomSearch> {
    let config = OptimizerConfig::default();
    Optimizer::new(config, RandomSearch::default())
}

/// Initialize with TPE sampler (recommended for most use cases)
pub fn init_tpe() -> Optimizer<TpeSampler> {
    let config = OptimizerConfig::default();
    Optimizer::new(config, TpeSampler::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_metadata() {
        let metadata = OptimizerMetadata::current();
        assert_eq!(metadata.version, VERSION);
        assert!(metadata.cpu_cores > 0);
    }

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus > 0);
    }
}
