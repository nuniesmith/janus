//! JANUS Optimizer Service
//!
//! A Rust-native parameter optimization service that:
//! - Collects OHLC data from Kraken via REST API
//! - Runs hyperparameter optimization using TPE sampler
//! - Publishes optimized parameters to Redis for Forward service hot-reload
//! - Exposes Prometheus metrics for monitoring
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    OPTIMIZER SERVICE                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
//! │  │ Data         │───▶│ Optimizer    │───▶│ Redis           │   │
//! │  │ Collector    │    │ Engine       │    │ Publisher       │   │
//! │  └──────────────┘    └──────────────┘    └─────────────────┘   │
//! │         │                   │                    │              │
//! │         ▼                   ▼                    ▼              │
//! │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
//! │  │ SQLite DB    │    │ Backtest     │    │ Forward Service │   │
//! │  │ (OHLC Data)  │    │ Engine       │    │ (Hot-Reload)    │   │
//! │  └──────────────┘    └──────────────┘    └─────────────────┘   │
//! │                                                                  │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │                Prometheus Metrics (:9092)                   │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tokio::sync::{RwLock, broadcast};
use tokio::time::interval;
use tracing::{error, info, warn};

mod cli;
mod collector;
mod config;
mod health;
mod metrics;
mod runner;
mod scheduler;
mod service;

use cli::Cli;
use config::OptimizerServiceConfig;
use health::HealthServer;
use metrics::MetricsRegistry;
use runner::Runner;
use scheduler::OptimizationScheduler;
use service::OptimizerService;

/// Application state shared across components
pub struct AppState {
    /// Service configuration
    pub config: OptimizerServiceConfig,

    /// Metrics registry
    pub metrics: Arc<MetricsRegistry>,

    /// Shutdown signal broadcaster
    pub shutdown_tx: broadcast::Sender<()>,

    /// Service health status
    pub healthy: Arc<RwLock<bool>>,

    /// Last optimization result
    pub last_optimization: Arc<RwLock<Option<OptimizationStatus>>>,
}

/// Status of the last optimization run
#[derive(Debug, Clone)]
pub struct OptimizationStatus {
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub assets_optimized: Vec<String>,
    pub assets_failed: Vec<String>,
    pub total_duration_secs: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Check if running in CLI mode
    let args: Vec<String> = std::env::args().collect();

    // CLI subcommands that trigger CLI mode
    let cli_subcommands = [
        "optimize",
        "run",
        "run-once",
        "status",
        "collect",
        "list-assets",
        "history",
        "help",
    ];

    // CLI mode is triggered by:
    // 1. Any subcommand appearing anywhere in args
    // 2. Help flags (-h, --help) anywhere in args
    // 3. Version flags (-V, --version) anywhere in args
    let is_cli_mode = args.iter().skip(1).any(|arg| {
        cli_subcommands.contains(&arg.as_str())
            || arg == "-h"
            || arg == "--help"
            || arg == "-V"
            || arg == "--version"
    });

    if is_cli_mode {
        return run_cli_mode().await;
    }

    // Otherwise, run in daemon mode (original behavior)
    run_daemon_mode().await
}

/// Run in CLI mode with subcommands
async fn run_cli_mode() -> Result<()> {
    use clap::Parser;

    let cli = Cli::parse();

    // Initialize logging based on CLI verbosity
    init_logging_with_level(cli.log_level());

    // Run the CLI command
    let runner = Runner::new(cli);
    match runner.run().await {
        Ok(result) => {
            if result.success {
                Ok(())
            } else {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run in daemon mode (original service behavior)
async fn run_daemon_mode() -> Result<()> {
    // Initialize logging
    init_logging();

    info!("═══════════════════════════════════════════════════════════");
    info!("         JANUS Optimizer Service Starting                   ");
    info!("═══════════════════════════════════════════════════════════");

    // Load configuration
    let config = OptimizerServiceConfig::from_env().context("Failed to load configuration")?;

    info!("Configuration loaded:");
    info!("  Assets: {:?}", config.assets);
    info!("  Data directory: {}", config.data_dir.display());
    info!("  Optimization interval: {}", config.optimization_interval);
    info!("  Trials per asset: {}", config.n_trials);
    info!("  Metrics port: {}", config.metrics_port);
    info!("  Redis URL: {}", config.redis_url);

    // Initialize metrics registry
    let metrics = Arc::new(MetricsRegistry::new());

    // Create shutdown channel
    let (shutdown_tx, _) = broadcast::channel::<()>(1);

    // Create application state
    let state = Arc::new(AppState {
        config: config.clone(),
        metrics: metrics.clone(),
        shutdown_tx: shutdown_tx.clone(),
        healthy: Arc::new(RwLock::new(true)),
        last_optimization: Arc::new(RwLock::new(None)),
    });

    // Start metrics and health server
    let health_server = HealthServer::new(state.clone());
    let health_handle = tokio::spawn(async move {
        if let Err(e) = health_server.run().await {
            error!("Health server error: {}", e);
        }
    });

    // Create optimizer service
    let service = OptimizerService::new(state.clone())
        .await
        .context("Failed to create optimizer service")?;

    let service = Arc::new(RwLock::new(service));

    // Run initial data collection if enabled
    if config.data_collection_enabled {
        info!("Starting initial data collection...");
        let mut svc = service.write().await;
        match svc.collect_initial_data().await {
            Ok(stats) => {
                info!(
                    "Initial data collection complete: {} candles collected",
                    stats.total_candles
                );
                metrics.record_collection_success();
            }
            Err(e) => {
                warn!("Initial data collection failed: {}", e);
                metrics.record_collection_failure();
            }
        }
    }

    // Run optimization on start if configured
    if config.run_on_start {
        info!("Running initial optimization...");
        let mut svc = service.write().await;
        match svc.run_optimization_cycle().await {
            Ok(results) => {
                info!(
                    "Initial optimization complete: {} successful, {} failed",
                    results.successful, results.failed
                );
            }
            Err(e) => {
                error!("Initial optimization failed: {}", e);
            }
        }
    }

    // Create scheduler
    let scheduler = OptimizationScheduler::new(
        service.clone(),
        state.clone(),
        config.optimization_interval.clone(),
    );

    // Start scheduler in background
    let scheduler_handle = tokio::spawn(async move {
        scheduler.run().await;
    });

    // Start data collection loop if enabled
    let data_collection_handle = if config.data_collection_enabled {
        let service_clone = service.clone();
        let state_clone = state.clone();
        let interval_mins = config.data_collection_interval_minutes;

        Some(tokio::spawn(async move {
            run_data_collection_loop(service_clone, state_clone, interval_mins).await;
        }))
    } else {
        None
    };

    // Wait for shutdown signal
    info!("Optimizer service is running. Press Ctrl+C to stop.");
    wait_for_shutdown().await;

    info!("Shutdown signal received, stopping services...");

    // Send shutdown signal to all components
    let _ = shutdown_tx.send(());

    // Mark service as unhealthy
    *state.healthy.write().await = false;

    // Wait for tasks to complete (with timeout)
    let shutdown_timeout = Duration::from_secs(30);

    tokio::select! {
        _ = async {
            let _ = scheduler_handle.await;
            if let Some(handle) = data_collection_handle {
                let _ = handle.await;
            }
            let _ = health_handle.await;
        } => {
            info!("All services stopped gracefully");
        }
        _ = tokio::time::sleep(shutdown_timeout) => {
            warn!("Shutdown timeout reached, forcing exit");
        }
    }

    info!("═══════════════════════════════════════════════════════════");
    info!("         JANUS Optimizer Service Stopped                    ");
    info!("═══════════════════════════════════════════════════════════");

    Ok(())
}

/// Initialize tracing/logging
fn init_logging() {
    init_logging_with_level("info");
}

/// Initialize tracing/logging with specific level
fn init_logging_with_level(level: &str) {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("{},janus_optimizer=debug", level)));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_thread_ids(true))
        .with(filter)
        .init();
}

/// Wait for shutdown signal (Ctrl+C or SIGTERM)
async fn wait_for_shutdown() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

/// Run periodic data collection
async fn run_data_collection_loop(
    service: Arc<RwLock<OptimizerService>>,
    state: Arc<AppState>,
    interval_minutes: u64,
) {
    let mut shutdown_rx = state.shutdown_tx.subscribe();
    let mut ticker = interval(Duration::from_secs(interval_minutes * 60));

    // Skip immediate tick
    ticker.tick().await;

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                info!("Running periodic data collection...");
                let mut svc = service.write().await;
                match svc.update_data().await {
                    Ok(stats) => {
                        info!("Data update complete: {} new candles", stats.new_candles);
                        state.metrics.record_collection_success();
                    }
                    Err(e) => {
                        warn!("Data update failed: {}", e);
                        state.metrics.record_collection_failure();
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                info!("Data collection loop shutting down");
                break;
            }
        }
    }
}
