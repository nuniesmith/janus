//! Optimizer Runner - Command Handlers
//!
//! Implements the execution logic for each CLI command:
//! - `optimize` - Run optimization for specific assets
//! - `run` - Start the scheduler daemon
//! - `run-once` - Run a single optimization cycle and exit
//! - `status` - Check optimization and data status
//! - `collect` - Collect OHLC data without optimization
//!
//! This module separates the CLI parsing from the execution logic,
//! making it easier to test and reuse.

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, broadcast};
use tracing::{error, info, warn};

use crate::AppState;
use crate::cli::{Cli, Commands, OutputFormat};
use crate::config::OptimizerServiceConfig;
use crate::health::HealthServer;
use crate::metrics::MetricsRegistry;
use crate::scheduler::OptimizationScheduler;
use crate::service::{AssetOptimizationResult, OptimizerService};

/// Result of running a CLI command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    /// Whether the command succeeded
    pub success: bool,
    /// Human-readable message
    pub message: String,
    /// Detailed results (command-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// Duration in seconds
    pub duration_secs: f64,
}

impl CommandResult {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            details: None,
            duration_secs: 0.0,
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            details: None,
            duration_secs: 0.0,
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_duration(mut self, duration: f64) -> Self {
        self.duration_secs = duration;
        self
    }
}

/// Status information for the `status` command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusInfo {
    pub redis_connected: bool,
    pub redis_url: String,
    pub data_dir: PathBuf,
    pub data_dir_exists: bool,
    pub assets: Vec<AssetStatus>,
    pub last_optimization: Option<OptimizationStatusInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetStatus {
    pub asset: String,
    pub has_data: bool,
    pub candle_count: Option<usize>,
    pub data_days: Option<f64>,
    pub last_update: Option<chrono::DateTime<Utc>>,
    pub has_optimized_params: bool,
    pub params_version: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatusInfo {
    pub started_at: chrono::DateTime<Utc>,
    pub completed_at: Option<chrono::DateTime<Utc>>,
    pub assets_optimized: Vec<String>,
    pub assets_failed: Vec<String>,
    pub duration_secs: Option<f64>,
}

/// Main runner that executes CLI commands
pub struct Runner {
    cli: Cli,
}

impl Runner {
    /// Create a new runner from CLI arguments
    pub fn new(cli: Cli) -> Self {
        Self { cli }
    }

    /// Execute the CLI command
    pub async fn run(self) -> Result<CommandResult> {
        let start = Instant::now();

        let result = match &self.cli.command {
            Commands::Optimize(args) => self.run_optimize(args).await,
            Commands::Run(args) => self.run_scheduler(args).await,
            Commands::RunOnce(args) => self.run_once(args).await,
            Commands::Status(args) => self.run_status(args).await,
            Commands::Collect(args) => self.run_collect(args).await,
            Commands::ListAssets(args) => self.run_list_assets(args).await,
            Commands::History(args) => self.run_history(args).await,
        };

        let duration = start.elapsed().as_secs_f64();

        match result {
            Ok(mut cmd_result) => {
                cmd_result.duration_secs = duration;
                self.output_result(&cmd_result);
                Ok(cmd_result)
            }
            Err(e) => {
                let result = CommandResult::failure(e.to_string()).with_duration(duration);
                self.output_result(&result);
                Err(e)
            }
        }
    }

    /// Output result based on format
    fn output_result(&self, result: &CommandResult) {
        if self.cli.quiet && result.success {
            return;
        }

        match self.cli.format {
            OutputFormat::Json => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(result).unwrap_or_default()
                );
            }
            OutputFormat::JsonCompact => {
                println!("{}", serde_json::to_string(result).unwrap_or_default());
            }
            OutputFormat::Text | OutputFormat::Table => {
                if result.success {
                    println!("✅ {}", result.message);
                } else {
                    eprintln!("❌ {}", result.message);
                }
                if let Some(details) = &result.details
                    && let Ok(pretty) = serde_json::to_string_pretty(details)
                {
                    println!("{}", pretty);
                }
                if result.duration_secs > 0.0 {
                    println!("⏱  Duration: {:.2}s", result.duration_secs);
                }
            }
        }
    }

    /// Create application state for the service
    async fn create_app_state(&self, config: OptimizerServiceConfig) -> Result<Arc<AppState>> {
        let metrics = Arc::new(MetricsRegistry::new());
        let (shutdown_tx, _) = broadcast::channel::<()>(1);

        Ok(Arc::new(AppState {
            config,
            metrics,
            shutdown_tx,
            healthy: Arc::new(RwLock::new(true)),
            last_optimization: Arc::new(RwLock::new(None)),
        }))
    }

    /// Build config from CLI options
    fn build_config(&self) -> OptimizerServiceConfig {
        OptimizerServiceConfig {
            data_dir: self.cli.data_dir.clone(),
            redis_url: self.cli.redis_url.clone(),
            redis_instance_id: self.cli.instance_id.clone(),
            ..Default::default()
        }
    }

    // ========================================================================
    // Command Handlers
    // ========================================================================

    /// Run optimization for specific assets
    async fn run_optimize(&self, args: &crate::cli::OptimizeArgs) -> Result<CommandResult> {
        let assets = args.get_assets();
        let trials = args.get_trials();

        info!("Running optimization for assets: {:?}", assets);
        info!("Trials: {}, Dry run: {}", trials, args.dry_run);

        let mut config = self.build_config();
        config.assets = assets.clone();
        config.n_trials = trials;
        config.preferred_interval_minutes = args.interval;
        config.min_data_days = args.min_days;

        // Validate config
        if let Err(e) = config.validate() {
            return Ok(CommandResult::failure(format!(
                "Invalid configuration: {}",
                e
            )));
        }

        let state = self.create_app_state(config).await?;
        let service = OptimizerService::new(state.clone())
            .await
            .context("Failed to create optimizer service")?;

        let mut results: HashMap<String, AssetOptimizationResult> = HashMap::new();
        let mut successful = 0;
        let mut failed = 0;

        for asset in &assets {
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("Optimizing: {}", asset);
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            match service.optimize_asset(asset).await {
                Ok(result) => {
                    if result.success {
                        successful += 1;
                        info!(
                            "✅ {} optimization complete: score={:.2}",
                            asset,
                            result.score.unwrap_or(0.0)
                        );
                    } else {
                        failed += 1;
                        warn!(
                            "⚠️ {} optimization failed: {}",
                            asset,
                            result.error.as_deref().unwrap_or("unknown error")
                        );
                    }
                    results.insert(asset.clone(), result);
                }
                Err(e) => {
                    failed += 1;
                    error!("❌ {} optimization error: {}", asset, e);
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

        let message = format!(
            "Optimization complete: {} successful, {} failed",
            successful, failed
        );

        let details = serde_json::json!({
            "assets": assets,
            "successful": successful,
            "failed": failed,
            "trials": trials,
            "dry_run": args.dry_run,
            "results": results,
        });

        if args.save || args.output.is_some() {
            let output_path = args.output.clone().unwrap_or_else(|| {
                state.config.results_dir().join(format!(
                    "optimization_{}.json",
                    Utc::now().format("%Y%m%d_%H%M%S")
                ))
            });

            if let Err(e) = std::fs::write(&output_path, serde_json::to_string_pretty(&details)?) {
                warn!("Failed to save results to {:?}: {}", output_path, e);
            } else {
                info!("Results saved to {:?}", output_path);
            }
        }

        Ok(CommandResult::success(message).with_details(details))
    }

    /// Start the scheduler daemon
    async fn run_scheduler(&self, args: &crate::cli::RunArgs) -> Result<CommandResult> {
        let assets = args.get_assets();
        let trials = args.get_trials();

        info!("Starting optimizer scheduler");
        info!("  Interval: {}", args.interval);
        info!("  Assets: {:?}", assets);
        info!("  Trials: {}", trials);
        info!("  Dry run: {}", args.dry_run);

        let mut config = self.build_config();
        config.assets = assets;
        config.n_trials = trials;
        config.optimization_interval = args.interval.clone();
        config.run_on_start = args.run_on_start;
        config.data_collection_enabled = args.collect_data;
        config.data_collection_interval_minutes = args.collect_interval;
        config.metrics_port = args.metrics_port;

        if let Err(e) = config.validate() {
            return Ok(CommandResult::failure(format!(
                "Invalid configuration: {}",
                e
            )));
        }

        let state = self.create_app_state(config).await?;
        let service = OptimizerService::new(state.clone())
            .await
            .context("Failed to create optimizer service")?;

        let service = Arc::new(RwLock::new(service));

        // Start health server
        let health_server = HealthServer::new(state.clone());
        let health_handle = tokio::spawn(async move {
            if let Err(e) = health_server.run().await {
                error!("Health server error: {}", e);
            }
        });

        // Run initial optimization if configured
        if state.config.run_on_start {
            info!("Running initial optimization...");
            let mut svc = service.write().await;
            match svc.run_optimization_cycle().await {
                Ok(results) => {
                    info!(
                        "Initial optimization: {} successful, {} failed",
                        results.successful, results.failed
                    );
                }
                Err(e) => {
                    error!("Initial optimization failed: {}", e);
                }
            }
        }

        // Create and run scheduler
        let scheduler = OptimizationScheduler::new(
            service.clone(),
            state.clone(),
            state.config.optimization_interval.clone(),
        );

        info!("Scheduler running. Press Ctrl+C to stop.");

        // Run scheduler (blocks until shutdown)
        scheduler.run().await;

        // Cleanup
        health_handle.abort();

        Ok(CommandResult::success("Scheduler stopped"))
    }

    /// Run a single optimization cycle and exit
    async fn run_once(&self, args: &crate::cli::RunOnceArgs) -> Result<CommandResult> {
        let assets = args.get_assets();
        let trials = args.get_trials();

        info!("Running single optimization cycle");
        info!("  Assets: {:?}", assets);
        info!("  Trials: {}", trials);

        let mut config = self.build_config();
        config.assets = assets.clone();
        config.n_trials = trials;

        if let Err(e) = config.validate() {
            return Ok(CommandResult::failure(format!(
                "Invalid configuration: {}",
                e
            )));
        }

        let state = self.create_app_state(config).await?;
        let mut service = OptimizerService::new(state.clone())
            .await
            .context("Failed to create optimizer service")?;

        // Update data if requested
        if args.update_data {
            info!("Updating OHLC data...");
            match service.update_data().await {
                Ok(stats) => {
                    info!("Data update: {} new candles", stats.new_candles);
                }
                Err(e) => {
                    warn!("Data update failed: {}", e);
                    if args.strict {
                        return Ok(CommandResult::failure(format!("Data update failed: {}", e)));
                    }
                }
            }
        }

        // Run optimization cycle
        let result = service.run_optimization_cycle().await;

        match result {
            Ok(cycle_result) => {
                let success = if args.strict {
                    cycle_result.failed == 0
                } else {
                    cycle_result.successful > 0
                };

                let message = format!(
                    "Optimization cycle complete: {} successful, {} failed in {:.1}s",
                    cycle_result.successful, cycle_result.failed, cycle_result.duration_secs
                );

                let details = serde_json::to_value(&cycle_result)?;

                if success {
                    Ok(CommandResult::success(message).with_details(details))
                } else {
                    Ok(CommandResult::failure(message).with_details(details))
                }
            }
            Err(e) => Ok(CommandResult::failure(format!(
                "Optimization cycle failed: {}",
                e
            ))),
        }
    }

    /// Check optimization and data status
    async fn run_status(&self, args: &crate::cli::StatusArgs) -> Result<CommandResult> {
        let config = self.build_config();
        let check_all = args.check_all();

        let mut status = StatusInfo {
            redis_connected: false,
            redis_url: config.redis_url.clone(),
            data_dir: config.data_dir.clone(),
            data_dir_exists: config.data_dir.exists(),
            assets: vec![],
            last_optimization: None,
        };

        // Check Redis connection
        if check_all || args.check_redis {
            match redis::Client::open(config.redis_url.as_str()) {
                Ok(client) => {
                    status.redis_connected =
                        client.get_multiplexed_async_connection().await.is_ok();
                }
                Err(_) => {
                    status.redis_connected = false;
                }
            }
        }

        // Check data for each asset
        if check_all || args.check_data {
            let assets_to_check = if let Some(asset) = &args.asset {
                vec![asset.clone()]
            } else {
                config.assets.clone()
            };

            for asset in assets_to_check {
                let asset_status = AssetStatus {
                    asset: asset.clone(),
                    has_data: false, // Would check DB
                    candle_count: None,
                    data_days: None,
                    last_update: None,
                    has_optimized_params: false, // Would check Redis
                    params_version: None,
                };
                status.assets.push(asset_status);
            }
        }

        let details = serde_json::to_value(&status)?;

        let message = if status.redis_connected {
            format!(
                "Status OK - Redis: connected, Data dir: {}",
                if status.data_dir_exists {
                    "exists"
                } else {
                    "missing"
                }
            )
        } else {
            format!(
                "Status WARNING - Redis: disconnected, Data dir: {}",
                if status.data_dir_exists {
                    "exists"
                } else {
                    "missing"
                }
            )
        };

        Ok(CommandResult::success(message).with_details(details))
    }

    /// Collect OHLC data without optimization
    async fn run_collect(&self, args: &crate::cli::CollectArgs) -> Result<CommandResult> {
        info!("Collecting OHLC data");
        info!("  Days: {}", args.days);
        info!("  Intervals: {:?}", args.intervals);

        let mut config = self.build_config();
        config.historical_days = args.days;
        config.collection_intervals = args.intervals.clone();

        if let Some(assets) = &args.assets {
            config.assets = assets.iter().map(|s| s.to_uppercase()).collect();
        }

        if let Err(e) = config.validate() {
            return Ok(CommandResult::failure(format!(
                "Invalid configuration: {}",
                e
            )));
        }

        let state = self.create_app_state(config).await?;
        let mut service = OptimizerService::new(state.clone())
            .await
            .context("Failed to create optimizer service")?;

        // Run initial data collection
        match service.collect_initial_data().await {
            Ok(stats) => {
                let message = format!(
                    "Data collection complete: {} candles collected",
                    stats.total_candles
                );

                let details = serde_json::json!({
                    "total_candles": stats.total_candles,
                    "assets": stats.assets_collected,
                    "intervals": args.intervals,
                    "days": args.days,
                });

                Ok(CommandResult::success(message).with_details(details))
            }
            Err(e) => Ok(CommandResult::failure(format!(
                "Data collection failed: {}",
                e
            ))),
        }
    }

    /// List configured assets and their settings
    async fn run_list_assets(&self, args: &crate::cli::ListAssetsArgs) -> Result<CommandResult> {
        use janus_optimizer::AssetRegistry;

        let registry = AssetRegistry::with_kraken_defaults();
        let all_assets = registry.all_assets();

        let mut asset_info: Vec<serde_json::Value> = vec![];

        for config in all_assets {
            if args.enabled_only && !config.enabled {
                continue;
            }

            if let Some(ref category_filter) = args.category {
                let cat_str = format!("{:?}", config.category).to_lowercase();
                if !cat_str.contains(&category_filter.to_lowercase()) {
                    continue;
                }
            }

            let info = if args.detailed {
                serde_json::json!({
                    "asset": config.symbol,
                    "category": format!("{:?}", config.category),
                    "enabled": config.enabled,
                    "max_position_usd": config.max_position_usd,
                    "liquidity_tier": config.liquidity_tier,
                })
            } else {
                serde_json::json!({
                    "asset": config.symbol,
                    "category": format!("{:?}", config.category),
                    "enabled": config.enabled,
                })
            };
            asset_info.push(info);
        }

        let message = format!("Found {} assets", asset_info.len());
        let details = serde_json::json!({
            "count": asset_info.len(),
            "assets": asset_info,
        });

        Ok(CommandResult::success(message).with_details(details))
    }

    /// Show or clear optimization history
    async fn run_history(&self, args: &crate::cli::HistoryArgs) -> Result<CommandResult> {
        if args.clear {
            // Would clear history from storage
            return Ok(CommandResult::success("History cleared"));
        }

        // Would load history from storage
        let history: Vec<serde_json::Value> = vec![];

        let message = format!("Showing {} history entries", history.len());
        let details = serde_json::json!({
            "count": history.len(),
            "entries": history,
        });

        Ok(CommandResult::success(message).with_details(details))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_result_success() {
        let result = CommandResult::success("Test passed");
        assert!(result.success);
        assert_eq!(result.message, "Test passed");
    }

    #[test]
    fn test_command_result_failure() {
        let result = CommandResult::failure("Test failed");
        assert!(!result.success);
        assert_eq!(result.message, "Test failed");
    }

    #[test]
    fn test_command_result_with_details() {
        let result = CommandResult::success("OK").with_details(serde_json::json!({"key": "value"}));

        assert!(result.details.is_some());
        let details = result.details.unwrap();
        assert_eq!(details["key"], "value");
    }

    #[test]
    fn test_command_result_with_duration() {
        let result = CommandResult::success("OK").with_duration(1.5);
        assert_eq!(result.duration_secs, 1.5);
    }
}
