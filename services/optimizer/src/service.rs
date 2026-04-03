//! Optimizer Service
//!
//! Main service that orchestrates data collection, optimization runs,
//! and publishing results to Redis.
//!
//! # Responsibilities
//!
//! - Load OHLC data from SQLite and convert to Polars DataFrame
//! - Run optimization for each configured asset
//! - Apply asset-specific parameter constraints
//! - Publish optimized parameters to Redis
//! - Save results to JSON files
//!
//! # Usage
//!
//! ```rust,ignore
//! let service = OptimizerService::new(state).await?;
//!
//! // Run full optimization cycle
//! let results = service.run_optimization_cycle().await?;
//!
//! // Or optimize a single asset
//! let params = service.optimize_asset("BTC").await?;
//! ```

use anyhow::{Context, Result};
use chrono::Utc;
use janus_core::optimized_params::{BacktestResultSummary, OptimizedParams, ParamNotification};
use janus_optimizer::{
    AssetRegistry, OptimizationResult, Optimizer, OptimizerConfig, SearchSpace, TpeSampler,
};
use polars::prelude::*;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, warn};

use crate::collector::{CollectionStats, OhlcCandle, OhlcCollector};

use crate::{AppState, OptimizationStatus};

/// Results from an optimization cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCycleResult {
    pub successful: usize,
    pub failed: usize,
    pub assets: Vec<String>,
    pub results: HashMap<String, AssetOptimizationResult>,
    pub duration_secs: f64,
    pub started_at: chrono::DateTime<Utc>,
    pub completed_at: chrono::DateTime<Utc>,
}

/// Result for a single asset optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetOptimizationResult {
    pub asset: String,
    pub success: bool,
    pub score: Option<f64>,
    pub params: Option<OptimizedParams>,
    pub backtest: Option<BacktestResultSummary>,
    pub error: Option<String>,
    pub trials: usize,
    pub duration_secs: f64,
}

/// Data update statistics
#[derive(Debug, Clone, Default)]
pub struct DataUpdateStats {
    pub new_candles: usize,
    #[allow(dead_code)]
    pub assets_updated: Vec<String>,
}

/// Main optimizer service
pub struct OptimizerService {
    /// Application state
    state: Arc<AppState>,

    /// OHLC data collector
    collector: OhlcCollector,

    /// Redis client for publishing
    redis_client: Option<redis::Client>,

    /// Asset registry for constraints
    asset_registry: AssetRegistry,
}

impl OptimizerService {
    /// Create a new optimizer service
    pub async fn new(state: Arc<AppState>) -> Result<Self> {
        // Ensure directories exist
        state.config.ensure_directories()?;

        // Create OHLC collector
        let collector = OhlcCollector::new(state.config.clone())
            .await
            .context("Failed to create OHLC collector")?;

        // Create Redis client
        let redis_client = match redis::Client::open(state.config.redis_url.as_str()) {
            Ok(client) => {
                // Test connection
                match client.get_multiplexed_async_connection().await {
                    Ok(_) => {
                        info!("✅ Connected to Redis at {}", state.config.redis_url);
                        Some(client)
                    }
                    Err(e) => {
                        warn!("⚠️ Failed to connect to Redis: {}", e);
                        warn!("   Params will only be saved to files, not published to Redis");
                        None
                    }
                }
            }
            Err(e) => {
                warn!("⚠️ Invalid Redis URL: {}", e);
                None
            }
        };

        // Create asset registry with Kraken defaults
        let asset_registry = AssetRegistry::with_kraken_defaults();

        Ok(Self {
            state,
            collector,
            redis_client,
            asset_registry,
        })
    }

    /// Collect initial historical data
    pub async fn collect_initial_data(&mut self) -> Result<CollectionStats> {
        info!("Starting initial data collection...");
        let stats = self.collector.collect_all(true).await?;

        // Log summary
        info!(
            "Initial collection complete: {} candles from {:?}",
            stats.total_candles, stats.assets_collected
        );

        if !stats.errors.is_empty() {
            warn!("Collection errors: {:?}", stats.errors);
        }

        Ok(stats)
    }

    /// Update data (fetch latest candles)
    pub async fn update_data(&mut self) -> Result<DataUpdateStats> {
        let stats = self.collector.update().await?;

        Ok(DataUpdateStats {
            new_candles: stats.new_candles,
            assets_updated: stats.assets_collected,
        })
    }

    /// Load OHLC data for an asset as Polars DataFrame
    pub async fn load_ohlc_data(&self, asset: &str) -> Result<Option<DataFrame>> {
        let interval = self.state.config.preferred_interval_minutes;

        // Get candle count
        let count = self.collector.get_candle_count(asset, interval).await?;
        if count == 0 {
            return Ok(None);
        }

        // Calculate minimum required candles
        let min_candles = self.state.config.min_data_days as usize * 24; // Assuming hourly data
        if count < min_candles {
            warn!(
                "Insufficient data for {}: {} candles, need {}",
                asset, count, min_candles
            );
            return Ok(None);
        }

        // Fetch candles
        let candles = self
            .collector
            .get_candles(asset, interval, None, None)
            .await?;

        if candles.is_empty() {
            return Ok(None);
        }

        // Convert to Polars DataFrame
        let df = candles_to_dataframe(&candles)?;

        info!(
            "Loaded {} candles for {} interval={}",
            df.height(),
            asset,
            interval
        );

        Ok(Some(df))
    }

    /// Run optimization for a single asset
    pub async fn optimize_asset(&self, asset: &str) -> Result<AssetOptimizationResult> {
        let start = Instant::now();
        info!("Starting optimization for {}", asset);

        // Load data
        let df = match self.load_ohlc_data(asset).await? {
            Some(df) => df,
            None => {
                return Ok(AssetOptimizationResult {
                    asset: asset.to_string(),
                    success: false,
                    score: None,
                    params: None,
                    backtest: None,
                    error: Some("Insufficient data".to_string()),
                    trials: 0,
                    duration_secs: start.elapsed().as_secs_f64(),
                });
            }
        };

        // Get asset configuration
        let asset_config = self.asset_registry.get(asset);
        let category = asset_config.category;

        // Create search space with asset constraints
        let search_space = SearchSpace::for_asset(asset, category);

        // Configure optimizer
        let opt_config = OptimizerConfig::builder()
            .n_trials(self.state.config.n_trials)
            .n_jobs(1) // Single-threaded for now
            .seed(42)
            .verbose(true)
            .build()
            .map_err(|e| anyhow::anyhow!("Invalid optimizer config: {}", e))?;

        // Create optimizer with TPE sampler
        let mut optimizer = Optimizer::new(opt_config, TpeSampler::default())
            .with_registry(self.asset_registry.clone());

        // Run optimization
        let result = optimizer.optimize(asset, &df, &search_space).await;

        match result {
            Ok(opt_result) => {
                // Convert to OptimizedParams
                let params = self.convert_to_optimized_params(asset, &opt_result);

                // Save and publish
                self.save_params(&params).await?;

                if let Err(e) = self.publish_params(&params).await {
                    warn!("Failed to publish params to Redis: {}", e);
                }

                // Record metrics
                self.state.metrics.record_optimization(
                    asset,
                    opt_result.best_score,
                    opt_result.best_backtest.total_pnl_pct,
                    opt_result.best_backtest.win_rate,
                    opt_result.best_backtest.max_drawdown_pct,
                    opt_result.duration_seconds,
                );

                let backtest_summary = BacktestResultSummary {
                    total_trades: opt_result.best_backtest.total_trades as u32,
                    winning_trades: opt_result.best_backtest.winning_trades as u32,
                    losing_trades: opt_result.best_backtest.losing_trades as u32,
                    total_pnl_pct: opt_result.best_backtest.total_pnl_pct,
                    max_drawdown_pct: opt_result.best_backtest.max_drawdown_pct,
                    win_rate: opt_result.best_backtest.win_rate,
                    profit_factor: opt_result.best_backtest.profit_factor,
                    sharpe_ratio: opt_result.best_backtest.sharpe_ratio,
                    trades_per_day: opt_result.best_backtest.trades_per_day,
                };

                info!(
                    "Optimization complete for {}: score={:.2}, return={:.2}%, win_rate={:.1}%",
                    asset,
                    opt_result.best_score,
                    opt_result.best_backtest.total_pnl_pct,
                    opt_result.best_backtest.win_rate
                );

                Ok(AssetOptimizationResult {
                    asset: asset.to_string(),
                    success: true,
                    score: Some(opt_result.best_score),
                    params: Some(params),
                    backtest: Some(backtest_summary),
                    error: None,
                    trials: opt_result.total_trials,
                    duration_secs: start.elapsed().as_secs_f64(),
                })
            }
            Err(e) => {
                error!("Optimization failed for {}: {}", asset, e);
                Ok(AssetOptimizationResult {
                    asset: asset.to_string(),
                    success: false,
                    score: None,
                    params: None,
                    backtest: None,
                    error: Some(e.to_string()),
                    trials: 0,
                    duration_secs: start.elapsed().as_secs_f64(),
                })
            }
        }
    }

    /// Run optimization cycle for all assets
    pub async fn run_optimization_cycle(&mut self) -> Result<OptimizationCycleResult> {
        let start = Instant::now();
        let started_at = Utc::now();
        let assets = self.state.config.assets.clone();

        info!("═══════════════════════════════════════════════════════════");
        info!("Starting optimization cycle for {:?}", assets);
        info!("═══════════════════════════════════════════════════════════");

        // Update optimization status
        {
            let mut status = self.state.last_optimization.write().await;
            *status = Some(OptimizationStatus {
                started_at,
                completed_at: None,
                assets_optimized: vec![],
                assets_failed: vec![],
                total_duration_secs: None,
            });
        }

        // Publish optimization started notification
        self.publish_optimization_started(&assets).await;

        let mut results = HashMap::new();
        let mut successful = 0;
        let mut failed = 0;

        for asset in &assets {
            match self.optimize_asset(asset).await {
                Ok(result) => {
                    if result.success {
                        successful += 1;
                    } else {
                        failed += 1;
                        self.publish_optimization_failed(
                            asset,
                            result.error.as_deref().unwrap_or("Unknown error"),
                        )
                        .await;
                    }
                    results.insert(asset.clone(), result);
                }
                Err(e) => {
                    failed += 1;
                    error!("Optimization error for {}: {}", asset, e);
                    self.publish_optimization_failed(asset, &e.to_string())
                        .await;
                    results.insert(
                        asset.clone(),
                        AssetOptimizationResult {
                            asset: asset.clone(),
                            success: false,
                            score: None,
                            params: None,
                            backtest: None,
                            error: Some(e.to_string()),
                            trials: 0,
                            duration_secs: 0.0,
                        },
                    );
                }
            }
        }

        let completed_at = Utc::now();
        let duration_secs = start.elapsed().as_secs_f64();

        // Update optimization status
        {
            let mut status = self.state.last_optimization.write().await;
            if let Some(ref mut s) = *status {
                s.completed_at = Some(completed_at);
                s.total_duration_secs = Some(duration_secs);
                s.assets_optimized = results
                    .iter()
                    .filter(|(_, r)| r.success)
                    .map(|(a, _)| a.clone())
                    .collect();
                s.assets_failed = results
                    .iter()
                    .filter(|(_, r)| !r.success)
                    .map(|(a, _)| a.clone())
                    .collect();
            }
        }

        // Publish optimization complete notification
        self.publish_optimization_complete(successful, failed).await;

        info!("═══════════════════════════════════════════════════════════");
        info!(
            "Optimization cycle complete: {} successful, {} failed in {:.1}s",
            successful, failed, duration_secs
        );
        info!("═══════════════════════════════════════════════════════════");

        Ok(OptimizationCycleResult {
            successful,
            failed,
            assets,
            results,
            duration_secs,
            started_at,
            completed_at,
        })
    }

    /// Convert optimization result to OptimizedParams
    fn convert_to_optimized_params(
        &self,
        asset: &str,
        result: &OptimizationResult,
    ) -> OptimizedParams {
        let params = &result.best_params;
        let backtest = &result.best_backtest;

        OptimizedParams {
            asset: asset.to_string(),
            ema_fast_period: params.get_int("ema_fast_period").unwrap_or(9) as u32,
            ema_slow_period: params.get_int("ema_slow_period").unwrap_or(28) as u32,
            atr_length: params.get_int("atr_length").unwrap_or(14) as u32,
            atr_multiplier: params.get("atr_multiplier").unwrap_or(2.0),
            min_trailing_stop_pct: params.get("min_trailing_stop_pct").unwrap_or(0.5),
            min_ema_spread_pct: params.get("min_ema_spread_pct").unwrap_or(0.2),
            min_profit_pct: params.get("min_profit_pct").unwrap_or(0.4),
            take_profit_pct: params.get("take_profit_pct").unwrap_or(5.0),
            trade_cooldown_seconds: (params.get_int("cooldown_bars").unwrap_or(5) * 300) as u64,
            require_htf_alignment: params.get_bool("require_htf_alignment").unwrap_or(true),
            htf_timeframe_minutes: params.get_int("htf_period").unwrap_or(15) as u32,
            max_position_size_usd: 20.0,
            enabled: true,
            min_hold_minutes: self.get_min_hold_minutes(asset),
            prefer_trailing_stop_exit: true,
            optimized_at: Utc::now().to_rfc3339(),
            optimization_score: result.best_score,
            backtest_result: BacktestResultSummary {
                total_trades: backtest.total_trades as u32,
                winning_trades: backtest.winning_trades as u32,
                losing_trades: backtest.losing_trades as u32,
                total_pnl_pct: backtest.total_pnl_pct,
                max_drawdown_pct: backtest.max_drawdown_pct,
                win_rate: backtest.win_rate,
                profit_factor: backtest.profit_factor,
                sharpe_ratio: backtest.sharpe_ratio,
                trades_per_day: backtest.trades_per_day,
            },
        }
    }

    /// Get minimum hold minutes for an asset
    fn get_min_hold_minutes(&self, asset: &str) -> u32 {
        let config = self.asset_registry.get(asset);
        config.min_hold_minutes()
    }

    /// Save params to JSON file
    async fn save_params(&self, params: &OptimizedParams) -> Result<()> {
        let params_dir = self.state.config.params_dir();

        // Save individual asset params
        let filename = format!("{}_params.json", params.asset.to_lowercase());
        let path = params_dir.join(&filename);

        let json = serde_json::to_string_pretty(params)?;
        tokio::fs::write(&path, &json).await?;

        debug!("Saved params to {}", path.display());

        // Also save env format for easy copy-paste
        let env_filename = format!("{}_params.env", params.asset.to_lowercase());
        let env_path = params_dir.join(&env_filename);
        let env_content = format!(
            r#"# Optimized params for {} - {}
EMA_FAST_PERIOD={}
EMA_SLOW_PERIOD={}
ATR_LENGTH={}
ATR_MULTIPLIER={}
MIN_TRAILING_STOP_PCT={}
MIN_EMA_SPREAD_PCT={}
MIN_PROFIT_PCT={}
TAKE_PROFIT_PCT={}
TRADE_COOLDOWN_SECONDS={}
REQUIRE_HTF_ALIGNMENT={}
HTF_TIMEFRAME_MINUTES={}
MIN_HOLD_MINUTES={}
"#,
            params.asset,
            params.optimized_at,
            params.ema_fast_period,
            params.ema_slow_period,
            params.atr_length,
            params.atr_multiplier,
            params.min_trailing_stop_pct,
            params.min_ema_spread_pct,
            params.min_profit_pct,
            params.take_profit_pct,
            params.trade_cooldown_seconds,
            params.require_htf_alignment,
            params.htf_timeframe_minutes,
            params.min_hold_minutes,
        );

        tokio::fs::write(&env_path, &env_content).await?;

        Ok(())
    }

    /// Publish params to Redis
    async fn publish_params(&self, params: &OptimizedParams) -> Result<()> {
        let client = match &self.redis_client {
            Some(c) => c,
            None => return Ok(()),
        };

        let mut conn = client.get_multiplexed_async_connection().await?;

        // Store in hash
        let hash_key = self.state.config.redis_params_key();
        let json = serde_json::to_string(params)?;
        conn.hset::<_, _, _, ()>(&hash_key, &params.asset, &json)
            .await?;

        // Update timestamp
        conn.hset::<_, _, _, ()>(&hash_key, "_last_updated", Utc::now().to_rfc3339())
            .await?;

        // Publish notification
        let channel = self.state.config.redis_updates_channel();
        let notification = ParamNotification::ParamUpdate {
            asset: params.asset.clone(),
            timestamp: Utc::now().to_rfc3339(),
            params: params.clone(),
        };

        let notification_json = serde_json::to_string(&notification)?;
        let subscribers: i64 = conn.publish(&channel, &notification_json).await?;

        info!(
            "📡 Published {} params to Redis (subscribers: {})",
            params.asset, subscribers
        );

        Ok(())
    }

    /// Publish optimization started notification
    async fn publish_optimization_started(&self, assets: &[String]) {
        if let Some(client) = &self.redis_client
            && let Ok(mut conn) = client.get_multiplexed_async_connection().await
        {
            let notification = ParamNotification::OptimizationStarted {
                timestamp: Utc::now().to_rfc3339(),
                assets: assets.to_vec(),
            };

            if let Ok(json) = serde_json::to_string(&notification) {
                let channel = self.state.config.redis_updates_channel();
                let _ = conn.publish::<_, _, i64>(&channel, &json).await;
                info!("📡 Published optimization_started notification");
            }
        }
    }

    /// Publish optimization complete notification
    async fn publish_optimization_complete(&self, successful: usize, failed: usize) {
        if let Some(client) = &self.redis_client
            && let Ok(mut conn) = client.get_multiplexed_async_connection().await
        {
            let notification = ParamNotification::OptimizationComplete {
                timestamp: Utc::now().to_rfc3339(),
                successful: successful as u32,
                failed: failed as u32,
                assets: self.state.config.assets.clone(),
            };

            if let Ok(json) = serde_json::to_string(&notification) {
                let channel = self.state.config.redis_updates_channel();
                let _ = conn.publish::<_, _, i64>(&channel, &json).await;
                info!("📡 Published optimization_complete notification");
            }
        }
    }

    /// Publish optimization failed notification
    async fn publish_optimization_failed(&self, asset: &str, error: &str) {
        if let Some(client) = &self.redis_client
            && let Ok(mut conn) = client.get_multiplexed_async_connection().await
        {
            let notification = ParamNotification::OptimizationFailed {
                timestamp: Utc::now().to_rfc3339(),
                asset: asset.to_string(),
                error: error.to_string(),
            };

            if let Ok(json) = serde_json::to_string(&notification) {
                let channel = self.state.config.redis_updates_channel();
                let _ = conn.publish::<_, _, i64>(&channel, &json).await;
                info!(
                    "📡 Published optimization_failed notification for {}",
                    asset
                );
            }
        }
    }

    /// Get data collection status
    #[allow(dead_code)]
    pub async fn get_data_status(
        &self,
    ) -> Result<HashMap<String, HashMap<u32, crate::collector::CollectionMetadata>>> {
        self.collector.get_status().await
    }
}

/// Convert OHLC candles to Polars DataFrame
fn candles_to_dataframe(candles: &[OhlcCandle]) -> Result<DataFrame> {
    let timestamps: Vec<i64> = candles.iter().map(|c| c.timestamp).collect();
    let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let df = df!(
        "timestamp" => &timestamps,
        "open" => &opens,
        "high" => &highs,
        "low" => &lows,
        "close" => &closes,
        "volume" => &volumes
    )?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candles_to_dataframe() {
        let candles = vec![
            OhlcCandle {
                timestamp: 1700000000,
                open: 50000.0,
                high: 51000.0,
                low: 49000.0,
                close: 50500.0,
                volume: 100.0,
                vwap: 50250.0,
                count: 1000,
            },
            OhlcCandle {
                timestamp: 1700003600,
                open: 50500.0,
                high: 51500.0,
                low: 50000.0,
                close: 51000.0,
                volume: 150.0,
                vwap: 50750.0,
                count: 1200,
            },
        ];

        let df = candles_to_dataframe(&candles).unwrap();

        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 6);
        assert!(df.column("open").is_ok());
        assert!(df.column("high").is_ok());
        assert!(df.column("low").is_ok());
        assert!(df.column("close").is_ok());
        assert!(df.column("volume").is_ok());
    }
}
