//! # JANUS Optimizer Runner
//!
//! CLI binary for running scheduled parameter optimizations.
//!
//! ## Usage
//!
//! ```bash
//! # Run optimization for a single asset
//! optimizer-runner optimize --asset BTC --interval 15m
//!
//! # Run optimization for all configured assets
//! optimizer-runner optimize --all
//!
//! # Run with custom trials
//! optimizer-runner optimize --asset ETH --trials 200
//!
//! # Run continuously with schedule
//! optimizer-runner run --cron "0 */6 * * *"
//!
//! # Show current optimized params
//! optimizer-runner show --asset BTC
//!
//! # Export metrics
//! optimizer-runner metrics --port 9093
//! ```

use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand};
use janus_optimizer::{AssetRegistry, OptimizerConfig, ParamPublisher, RandomSearch, TpeSampler};
use polars::prelude::*;
use prometheus::{Encoder, TextEncoder};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{error, info, warn};

/// JANUS Optimizer Runner - Hyperparameter optimization for trading strategies
#[derive(Parser, Debug)]
#[command(name = "optimizer-runner")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Redis URL for publishing results
    #[arg(long, env = "REDIS_URL", default_value = "redis://localhost:6379")]
    redis_url: String,

    /// Instance ID for Redis key prefix
    #[arg(long, env = "OPTIMIZER_INSTANCE_ID", default_value = "default")]
    instance_id: String,

    /// Data directory for OHLC files
    #[arg(long, env = "OPTIMIZER_DATA_DIR", default_value = "./data")]
    data_dir: PathBuf,

    /// Log level
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run optimization for specified asset(s)
    Optimize {
        /// Asset to optimize (e.g., BTC, ETH, SOL)
        #[arg(long)]
        asset: Option<String>,

        /// Optimize all configured assets
        #[arg(long, default_value = "false")]
        all: bool,

        /// Assets to optimize (comma-separated)
        #[arg(long, env = "OPTIMIZE_ASSETS", default_value = "BTC,ETH,SOL")]
        assets: String,

        /// Number of optimization trials
        #[arg(long, env = "OPTIMIZE_TRIALS", default_value = "100")]
        trials: usize,

        /// Data timeframe/interval (e.g., 1m, 5m, 15m, 1h)
        #[arg(long, env = "OPTIMIZE_INTERVAL", default_value = "15m")]
        interval: String,

        /// Historical days to use for backtesting
        #[arg(long, env = "OPTIMIZE_HISTORICAL_DAYS", default_value = "30")]
        historical_days: u32,

        /// Use TPE sampler instead of random search
        #[arg(long, default_value = "true")]
        use_tpe: bool,

        /// Enable parallel trials
        #[arg(long, default_value = "false")]
        parallel: bool,

        /// Number of parallel jobs (0 = auto)
        #[arg(long, default_value = "0")]
        jobs: usize,
    },

    /// Run optimizer continuously with a schedule
    Run {
        /// Cron expression for scheduling (e.g., "0 */6 * * *" for every 6 hours)
        #[arg(long, env = "OPTIMIZE_CRON", default_value = "0 */6 * * *")]
        cron: String,

        /// Assets to optimize (comma-separated)
        #[arg(long, env = "OPTIMIZE_ASSETS", default_value = "BTC,ETH,SOL")]
        assets: String,

        /// Number of optimization trials
        #[arg(long, env = "OPTIMIZE_TRIALS", default_value = "100")]
        trials: usize,

        /// Data interval
        #[arg(long, env = "OPTIMIZE_INTERVAL", default_value = "15m")]
        interval: String,

        /// Metrics port
        #[arg(long, env = "OPTIMIZER_METRICS_PORT", default_value = "9093")]
        metrics_port: u16,
    },

    /// Show current optimized parameters from Redis
    Show {
        /// Asset to show (omit for all)
        #[arg(long)]
        asset: Option<String>,

        /// Output format (json, table)
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Start metrics server only
    Metrics {
        /// Metrics port
        #[arg(long, env = "OPTIMIZER_METRICS_PORT", default_value = "9093")]
        port: u16,
    },

    /// Health check
    Health,
}

/// Prometheus metrics for the optimizer
#[allow(dead_code)]
struct OptimizerMetrics {
    optimizations_total: prometheus::CounterVec,
    optimization_duration_seconds: prometheus::HistogramVec,
    trials_per_optimization: prometheus::HistogramVec,
    best_score: prometheus::GaugeVec,
    last_optimization_timestamp: prometheus::GaugeVec,
    optimization_errors_total: prometheus::CounterVec,
}

#[allow(dead_code)]
impl OptimizerMetrics {
    fn new() -> Result<Self> {
        Ok(Self {
            optimizations_total: prometheus::register_counter_vec!(
                "optimizer_optimizations_total",
                "Total number of optimizations run",
                &["asset", "status"]
            )?,
            optimization_duration_seconds: prometheus::register_histogram_vec!(
                "optimizer_optimization_duration_seconds",
                "Duration of optimization runs",
                &["asset"],
                vec![10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]
            )?,
            trials_per_optimization: prometheus::register_histogram_vec!(
                "optimizer_trials_per_optimization",
                "Number of trials per optimization",
                &["asset"],
                vec![10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
            )?,
            best_score: prometheus::register_gauge_vec!(
                "optimizer_best_score",
                "Best optimization score achieved",
                &["asset"]
            )?,
            last_optimization_timestamp: prometheus::register_gauge_vec!(
                "optimizer_last_optimization_timestamp_seconds",
                "Timestamp of last optimization",
                &["asset"]
            )?,
            optimization_errors_total: prometheus::register_counter_vec!(
                "optimizer_errors_total",
                "Total optimization errors",
                &["asset", "error_type"]
            )?,
        })
    }

    fn record_success(&self, asset: &str, duration_secs: f64, trials: usize, best_score: f64) {
        self.optimizations_total
            .with_label_values(&[asset, "success"])
            .inc();
        self.optimization_duration_seconds
            .with_label_values(&[asset])
            .observe(duration_secs);
        self.trials_per_optimization
            .with_label_values(&[asset])
            .observe(trials as f64);
        self.best_score.with_label_values(&[asset]).set(best_score);
        self.last_optimization_timestamp
            .with_label_values(&[asset])
            .set(Utc::now().timestamp() as f64);
    }

    fn record_error(&self, asset: &str, error_type: &str) {
        self.optimizations_total
            .with_label_values(&[asset, "error"])
            .inc();
        self.optimization_errors_total
            .with_label_values(&[asset, error_type])
            .inc();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(&cli.log_level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .init();

    info!(
        version = env!("CARGO_PKG_VERSION"),
        instance_id = cli.instance_id,
        "Starting JANUS Optimizer Runner"
    );

    match cli.command {
        Commands::Optimize {
            asset,
            all,
            assets,
            trials,
            interval,
            historical_days,
            use_tpe,
            parallel,
            jobs,
        } => {
            let asset_list = if all {
                assets.split(',').map(|s| s.trim().to_string()).collect()
            } else if let Some(a) = asset {
                vec![a]
            } else {
                assets.split(',').map(|s| s.trim().to_string()).collect()
            };

            run_optimization(
                &cli.redis_url,
                &cli.instance_id,
                &cli.data_dir,
                &asset_list,
                trials,
                &interval,
                historical_days,
                use_tpe,
                parallel,
                jobs,
            )
            .await
        }

        Commands::Run {
            cron,
            assets,
            trials,
            interval,
            metrics_port,
        } => {
            run_scheduled(
                &cli.redis_url,
                &cli.instance_id,
                &cli.data_dir,
                &cron,
                &assets,
                trials,
                &interval,
                metrics_port,
            )
            .await
        }

        Commands::Show { asset, format } => {
            show_params(&cli.redis_url, &cli.instance_id, asset, &format).await
        }

        Commands::Metrics { port } => run_metrics_server(port).await,

        Commands::Health => health_check(&cli.redis_url).await,
    }
}

/// Run optimization for specified assets
#[allow(clippy::too_many_arguments)]
async fn run_optimization(
    redis_url: &str,
    instance_id: &str,
    data_dir: &Path,
    assets: &[String],
    trials: usize,
    interval: &str,
    historical_days: u32,
    use_tpe: bool,
    _parallel: bool,
    _jobs: usize,
) -> Result<()> {
    info!(
        assets = ?assets,
        trials = trials,
        interval = interval,
        historical_days = historical_days,
        "Running optimization"
    );

    // Create publisher
    let publisher = ParamPublisher::new(redis_url, instance_id)
        .await
        .context("Failed to connect to Redis")?;

    // Create asset registry
    let registry = AssetRegistry::default();

    // Run optimization for each asset
    for asset in assets {
        info!(asset = asset, "Starting optimization for asset");

        match run_single_optimization(
            asset,
            data_dir,
            interval,
            historical_days,
            trials,
            use_tpe,
            &registry,
        )
        .await
        {
            Ok(result) => {
                info!(
                    asset = asset,
                    best_score = result.best_score,
                    total_pnl = result.best_backtest.total_pnl_pct,
                    win_rate = result.best_backtest.win_rate,
                    trades = result.best_backtest.total_trades,
                    "Optimization completed"
                );

                // Publish to Redis
                if let Err(e) = publisher.publish(&result).await {
                    error!(asset = asset, error = %e, "Failed to publish results to Redis");
                } else {
                    info!(asset = asset, "Published optimized params to Redis");
                }
            }
            Err(e) => {
                error!(asset = asset, error = %e, "Optimization failed");
            }
        }
    }

    Ok(())
}

/// Run optimization for a single asset
#[allow(clippy::too_many_arguments)]
async fn run_single_optimization(
    asset: &str,
    data_dir: &Path,
    interval: &str,
    historical_days: u32,
    trials: usize,
    use_tpe: bool,
    registry: &AssetRegistry,
) -> Result<janus_optimizer::OptimizationResult> {
    // Load data
    let data = load_ohlc_data(asset, data_dir, interval, historical_days).await?;

    info!(asset = asset, rows = data.height(), "Loaded OHLC data");

    // Get asset config
    let asset_config = registry.get(asset);

    // Create search space
    let search_space = janus_optimizer::SearchSpace::for_asset(asset, asset_config.category);

    // Create config
    let config = OptimizerConfig::builder()
        .n_trials(trials)
        .build()
        .map_err(|e| anyhow::anyhow!("Invalid config: {}", e))?;

    // Run optimization
    if use_tpe {
        let mut optimizer = janus_optimizer::Optimizer::new(config, TpeSampler::default());
        optimizer
            .optimize(asset, &data, &search_space)
            .await
            .map_err(|e| anyhow::anyhow!("Optimization failed: {}", e))
    } else {
        let config2 = OptimizerConfig::builder()
            .n_trials(trials)
            .build()
            .map_err(|e| anyhow::anyhow!("Invalid config: {}", e))?;
        let mut optimizer = janus_optimizer::Optimizer::new(config2, RandomSearch::default());
        optimizer
            .optimize(asset, &data, &search_space)
            .await
            .map_err(|e| anyhow::anyhow!("Optimization failed: {}", e))
    }
}

/// Load OHLC data for backtesting
async fn load_ohlc_data(
    asset: &str,
    data_dir: &Path,
    interval: &str,
    historical_days: u32,
) -> Result<DataFrame> {
    // Try multiple file patterns
    let patterns = vec![
        format!(
            "{}/{}_{}.parquet",
            data_dir.display(),
            asset.to_lowercase(),
            interval
        ),
        format!(
            "{}/{}_usdt_{}.parquet",
            data_dir.display(),
            asset.to_lowercase(),
            interval
        ),
        format!(
            "{}/{}/{}.parquet",
            data_dir.display(),
            asset.to_uppercase(),
            interval
        ),
        format!(
            "{}/ohlc_{}_{}.parquet",
            data_dir.display(),
            asset.to_lowercase(),
            interval
        ),
    ];

    for path in &patterns {
        let path = PathBuf::from(path);
        if path.exists() {
            info!(path = %path.display(), "Loading OHLC data from file");

            let df = LazyFrame::scan_parquet(&path, Default::default())
                .context("Failed to scan parquet file")?
                .collect()
                .context("Failed to collect dataframe")?;

            // Filter to historical_days if timestamp column exists
            let column_names: Vec<&str> =
                df.get_column_names().iter().map(|s| s.as_str()).collect();
            if column_names.contains(&"timestamp") {
                let cutoff = Utc::now() - chrono::Duration::days(historical_days as i64);
                let cutoff_ms = cutoff.timestamp_millis();

                let filtered = df
                    .lazy()
                    .filter(col("timestamp").gt_eq(lit(cutoff_ms)))
                    .collect()
                    .context("Failed to filter data")?;

                return Ok(filtered);
            }

            return Ok(df);
        }
    }

    // If no file found, create synthetic data for testing
    warn!(
        asset = asset,
        interval = interval,
        "No OHLC data file found, generating synthetic data for testing"
    );

    generate_synthetic_ohlc(asset, historical_days)
}

/// Generate synthetic OHLC data for testing
fn generate_synthetic_ohlc(asset: &str, days: u32) -> Result<DataFrame> {
    use rand::RngExt;

    let mut rng = rand::rng();

    // Base price based on asset
    let base_price: f64 = match asset.to_uppercase().as_str() {
        "BTC" => 45000.0,
        "ETH" => 2500.0,
        "SOL" => 100.0,
        _ => 100.0,
    };

    let rows = days as usize * 24 * 4; // 15-minute candles
    let start_time = Utc::now() - chrono::Duration::days(days as i64);

    let mut timestamps = Vec::with_capacity(rows);
    let mut opens = Vec::with_capacity(rows);
    let mut highs = Vec::with_capacity(rows);
    let mut lows = Vec::with_capacity(rows);
    let mut closes = Vec::with_capacity(rows);
    let mut volumes = Vec::with_capacity(rows);

    let mut price: f64 = base_price;

    for i in 0..rows {
        let timestamp = start_time + chrono::Duration::minutes(15 * i as i64);
        timestamps.push(timestamp.timestamp_millis());

        let open: f64 = price;
        let change_pct: f64 = rng.random_range(-0.02..0.02); // ±2% per candle
        let close: f64 = open * (1.0 + change_pct);
        let high: f64 = open.max(close) * (1.0 + rng.random_range(0.0..0.005));
        let low: f64 = open.min(close) * (1.0 - rng.random_range(0.0..0.005));
        let volume: f64 = rng.random_range(100.0..10000.0) * base_price / 100.0;

        opens.push(open);
        highs.push(high);
        lows.push(low);
        closes.push(close);
        volumes.push(volume);

        price = close;
    }

    let df = DataFrame::new(vec![
        Series::new("timestamp".into(), timestamps).into(),
        Series::new("open".into(), opens).into(),
        Series::new("high".into(), highs).into(),
        Series::new("low".into(), lows).into(),
        Series::new("close".into(), closes).into(),
        Series::new("volume".into(), volumes).into(),
    ])
    .context("Failed to create synthetic DataFrame")?;

    info!(
        asset = asset,
        rows = df.height(),
        "Generated synthetic OHLC data"
    );

    Ok(df)
}

/// Run optimizer on a schedule
#[allow(clippy::too_many_arguments)]
async fn run_scheduled(
    redis_url: &str,
    instance_id: &str,
    data_dir: &Path,
    cron_expr: &str,
    assets: &str,
    trials: usize,
    interval: &str,
    metrics_port: u16,
) -> Result<()> {
    info!(
        cron = cron_expr,
        assets = assets,
        metrics_port = metrics_port,
        "Starting scheduled optimizer"
    );

    // Initialize metrics
    let metrics = OptimizerMetrics::new()?;

    // Start metrics server
    let metrics_handle = tokio::spawn(async move {
        if let Err(e) = run_metrics_server(metrics_port).await {
            error!(error = %e, "Metrics server failed");
        }
    });

    // Parse assets
    let asset_list: Vec<String> = assets.split(',').map(|s| s.trim().to_string()).collect();

    // Create publisher
    let publisher = ParamPublisher::new(redis_url, instance_id)
        .await
        .context("Failed to connect to Redis")?;

    let registry = AssetRegistry::default();

    // Simple interval-based scheduling (cron parsing would require additional dependency)
    // For now, parse simple intervals like "6h", "12h", "24h"
    let interval_duration = parse_schedule_interval(cron_expr)?;

    info!(
        interval_minutes = interval_duration.as_secs() / 60,
        "Running optimization loop"
    );

    // Setup shutdown handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        info!("Received shutdown signal");
        r.store(false, Ordering::SeqCst);
    })
    .context("Failed to set Ctrl-C handler")?;

    // Run immediately, then on schedule
    loop {
        for asset in &asset_list {
            let start = std::time::Instant::now();

            match run_single_optimization(
                asset, data_dir, interval, 30, // default historical days
                trials, true, // use TPE
                &registry,
            )
            .await
            {
                Ok(result) => {
                    let duration = start.elapsed().as_secs_f64();

                    metrics.record_success(asset, duration, result.total_trials, result.best_score);

                    if let Err(e) = publisher.publish(&result).await {
                        error!(asset = asset, error = %e, "Failed to publish");
                        metrics.record_error(asset, "publish_failed");
                    }
                }
                Err(e) => {
                    error!(asset = asset, error = %e, "Optimization failed");
                    metrics.record_error(asset, "optimization_failed");
                }
            }
        }

        info!(
            next_run_minutes = interval_duration.as_secs() / 60,
            "Optimization cycle complete, sleeping until next run"
        );

        // Sleep until next run, checking for shutdown every second
        let mut remaining = interval_duration;
        while remaining > Duration::ZERO && running.load(Ordering::SeqCst) {
            let sleep_time = remaining.min(Duration::from_secs(1));
            tokio::time::sleep(sleep_time).await;
            remaining = remaining.saturating_sub(sleep_time);
        }

        if !running.load(Ordering::SeqCst) {
            info!("Shutting down optimizer");
            break;
        }
    }

    metrics_handle.abort();
    Ok(())
}

/// Parse schedule interval from cron-like expression or simple duration
fn parse_schedule_interval(expr: &str) -> Result<Duration> {
    // Try simple duration first (e.g., "6h", "30m", "1d")
    if let Some(hours) = expr.strip_suffix('h') {
        let h: u64 = hours.parse().context("Invalid hours")?;
        return Ok(Duration::from_secs(h * 3600));
    }
    if let Some(mins) = expr.strip_suffix('m') {
        let m: u64 = mins.parse().context("Invalid minutes")?;
        return Ok(Duration::from_secs(m * 60));
    }
    if let Some(days) = expr.strip_suffix('d') {
        let d: u64 = days.parse().context("Invalid days")?;
        return Ok(Duration::from_secs(d * 86400));
    }

    // Try to parse cron expression (simplified: extract hour interval)
    // "0 */6 * * *" -> every 6 hours
    if expr.contains("*/") {
        let parts: Vec<&str> = expr.split_whitespace().collect();
        if parts.len() >= 2
            && let Some(interval) = parts[1].strip_prefix("*/")
        {
            let hours: u64 = interval.parse().unwrap_or(6);
            return Ok(Duration::from_secs(hours * 3600));
        }
    }

    // Default to 6 hours
    warn!(
        expr = expr,
        "Could not parse schedule, defaulting to 6 hours"
    );
    Ok(Duration::from_secs(6 * 3600))
}

/// Show current optimized parameters
async fn show_params(
    redis_url: &str,
    instance_id: &str,
    asset: Option<String>,
    format: &str,
) -> Result<()> {
    use redis::AsyncCommands;

    let client = redis::Client::open(redis_url)?;
    let mut conn = client.get_multiplexed_async_connection().await?;

    let hash_key = format!("fks:{}:optimized_params", instance_id);

    if let Some(asset) = asset {
        // Get single asset
        let result: Option<String> = conn.hget(&hash_key, &asset).await?;

        match result {
            Some(json) => {
                if format == "json" {
                    println!("{}", json);
                } else {
                    let params: janus_core::OptimizedParams = serde_json::from_str(&json)?;
                    print_params_table(&params);
                }
            }
            None => {
                println!("No optimized params found for asset: {}", asset);
            }
        }
    } else {
        // Get all assets
        let all: HashMap<String, String> = conn.hgetall(&hash_key).await?;

        if all.is_empty() {
            println!("No optimized params found in Redis");
            return Ok(());
        }

        if format == "json" {
            println!("{}", serde_json::to_string_pretty(&all)?);
        } else {
            for (asset, json) in all {
                let params: janus_core::OptimizedParams = serde_json::from_str(&json)?;
                println!("\n═══ {} ═══", asset);
                print_params_table(&params);
            }
        }
    }

    Ok(())
}

/// Print parameters in a table format
fn print_params_table(params: &janus_core::OptimizedParams) {
    println!("Asset:              {}", params.asset);
    println!("Optimized At:       {}", params.optimized_at);
    println!("Score:              {:.4}", params.optimization_score);
    println!(
        "Enabled:            {}",
        if params.enabled { "Yes" } else { "No" }
    );
    println!();
    println!("Indicator Config:");
    println!("  EMA Fast:         {}", params.ema_fast_period);
    println!("  EMA Slow:         {}", params.ema_slow_period);
    println!("  ATR Length:       {}", params.atr_length);
    println!("  ATR Multiplier:   {:.2}", params.atr_multiplier);
    println!();
    println!("Risk Config:");
    println!("  Min EMA Spread:   {:.2}%", params.min_ema_spread_pct);
    println!("  Min Profit:       {:.2}%", params.min_profit_pct);
    println!("  Take Profit:      {:.2}%", params.take_profit_pct);
    println!("  Min Trailing Stop:{:.2}%", params.min_trailing_stop_pct);
    println!("  Max Position:     ${:.0}", params.max_position_size_usd);
    println!();
    println!("Strategy Config:");
    println!("  Min Hold Time:    {} min", params.min_hold_minutes);
    println!("  Trade Cooldown:   {} sec", params.trade_cooldown_seconds);
    println!(
        "  HTF Alignment:    {}",
        if params.require_htf_alignment {
            "Required"
        } else {
            "Not required"
        }
    );
    println!("  HTF Timeframe:    {} min", params.htf_timeframe_minutes);
    println!(
        "  Prefer Trailing:  {}",
        if params.prefer_trailing_stop_exit {
            "Yes"
        } else {
            "No"
        }
    );
    println!();
    println!("Backtest Results:");
    println!(
        "  Total Trades:     {}",
        params.backtest_result.total_trades
    );
    println!(
        "  Win Rate:         {:.2}%",
        params.backtest_result.win_rate
    );
    println!(
        "  Total PnL:        {:.2}%",
        params.backtest_result.total_pnl_pct
    );
    println!(
        "  Max Drawdown:     {:.2}%",
        params.backtest_result.max_drawdown_pct
    );
    println!(
        "  Profit Factor:    {:.2}",
        params.backtest_result.profit_factor
    );
    println!(
        "  Sharpe Ratio:     {:.2}",
        params.backtest_result.sharpe_ratio
    );
}

/// Run metrics server
async fn run_metrics_server(port: u16) -> Result<()> {
    use axum::{Router, routing::get};
    use std::net::SocketAddr;

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(|| async { "OK" }));

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(port = port, "Starting metrics server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn metrics_handler() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Health check
async fn health_check(redis_url: &str) -> Result<()> {
    println!("Checking Redis connection...");

    let client = redis::Client::open(redis_url)?;
    let mut conn = client.get_multiplexed_async_connection().await?;

    let pong: String = redis::cmd("PING").query_async(&mut conn).await?;

    if pong == "PONG" {
        println!("✅ Redis: Connected");
    } else {
        println!("❌ Redis: Unexpected response: {}", pong);
        return Err(anyhow::anyhow!("Redis health check failed"));
    }

    println!("✅ Health check passed");
    Ok(())
}
