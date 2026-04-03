//! # JANUS Data Factory
//!
//! High-performance market data ingestion service for cryptocurrency trading.
//!
//! This service is responsible for:
//! - Real-time WebSocket ingestion from multiple exchanges (Binance, Bybit, Kucoin)
//! - Alternative metrics polling (Fear & Greed, ETF Flows, Volatility)
//! - High-throughput persistence to QuestDB via Influx Line Protocol
//! - Automatic failover and gap detection
//! - Redis-backed rate limiting and deduplication
//! - Real-time technical indicator calculation (EMA, RSI, MACD, ATR)
//!
//! ## Architecture
//!
//! The Data Factory follows an Actor Model pattern using Tokio:
//! - Each exchange connection runs as an isolated actor
//! - A central Router dispatches messages to the Storage layer
//! - IndicatorActor receives candles and calculates technical indicators
//! - Metric pollers operate on scheduled intervals
//! - All actors communicate via MPSC channels
//!
//! ## Performance
//!
//! - Sub-millisecond latency for WebSocket message processing
//! - 100K+ ticks/second ingestion to QuestDB
//! - O(1) incremental indicator updates
//! - Zero-copy deserialization where possible
//! - Batched ILP writes with configurable flush intervals

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::signal;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

mod actors;
mod api;
pub mod backfill;
mod config;
mod connectors;
mod logging;
mod metrics;
mod self_healing;
mod storage;

use actors::signal::SignalActor;
use config::Config;

/// Main application state
pub struct DataFactory {
    config: Arc<Config>,
    shutdown_tx: broadcast::Sender<()>,
}

impl DataFactory {
    /// Create a new Data Factory instance
    pub fn new(config: Config) -> Result<Self> {
        let (shutdown_tx, _) = broadcast::channel(16);

        Ok(Self {
            config: Arc::new(config),
            shutdown_tx,
        })
    }

    /// Run the Data Factory service
    pub async fn run(&self) -> Result<()> {
        info!("╔═══════════════════════════════════════════════════════════╗");
        info!("║           JANUS DATA FACTORY - STARTING                  ║");
        info!("╚═══════════════════════════════════════════════════════════╝");

        self.print_config();

        // Initialize Prometheus metrics
        info!("📊 Initializing Prometheus metrics...");
        api::metrics::init_metrics().context("Failed to initialize metrics")?;
        info!("✅ Prometheus metrics initialized");

        // Initialize storage connections
        info!("🔌 Initializing storage connections...");
        let storage = self.init_storage().await?;
        info!("✅ Storage initialized");

        // Initialize exchange connectors
        info!("📡 Initializing exchange connectors...");
        let connectors = self.init_connectors().await?;
        info!(
            "✅ Connectors initialized for {} assets",
            self.config.assets.len()
        );

        // Initialize metrics pollers
        if self.should_init_metrics() {
            info!("📊 Initializing metrics pollers...");
            let _metrics = self.init_metrics().await?;
            info!("✅ Metrics pollers initialized");
        }

        // Start the central router
        info!("🚀 Starting message router...");
        let router = self.start_router(storage.clone()).await?;
        info!("✅ Router started");

        // Start indicator actor for real-time technical indicator calculation
        info!("📈 Starting indicator actor...");
        let indicator_actor = self.start_indicator_actor(storage.clone()).await?;

        // Wire indicator actor to router for candle forwarding
        router
            .set_indicator_sender(indicator_actor.get_sender())
            .await;
        info!("✅ Indicator actor started (EMA, RSI, MACD, ATR)");

        // Start signal actor for trading signal generation
        info!("🎯 Starting signal actor...");
        let signal_actor = self
            .start_signal_actor(storage.clone(), indicator_actor.clone())
            .await?;
        info!("✅ Signal actor started (EMA cross, RSI zones, MACD cross, confluence)");

        // Start HTTP/WebSocket API server
        info!("🌐 Starting HTTP/WebSocket API server...");
        let _api_handle = self
            .start_api_server(
                storage.clone(),
                router.broadcast_tx.clone(),
                indicator_actor.clone(),
                signal_actor.clone(),
            )
            .await?;
        info!("✅ API server started on port 8080");

        // Start WebSocket connections
        info!("📡 Starting exchange WebSocket connections...");
        self.start_websockets(connectors, router).await?;
        info!("✅ Exchange WebSocket connections established");

        // Warm up indicators from historical candle data
        info!("🔥 Warming up indicators from historical data...");
        self.warmup_indicators(storage.clone(), indicator_actor)
            .await;

        info!("╔═══════════════════════════════════════════════════════════╗");
        info!("║           DATA FACTORY RUNNING                           ║");
        info!("║                                                          ║");
        info!("║  HTTP API:       http://0.0.0.0:8080                     ║");
        info!("║  Health:         GET  /health                            ║");
        info!("║  Metrics:        GET  /metrics                           ║");
        info!("║  Gap Analysis:   GET  /api/v1/gaps                       ║");
        info!("║  Indicators:     GET  /api/v1/indicators                 ║");
        info!("║  Signals:        GET  /api/v1/signals                    ║");
        info!("║  WebSocket:      WS   /ws/stream                         ║");
        info!("║  Signal Stream:  WS   /ws/signals                        ║");
        info!("╚═══════════════════════════════════════════════════════════╝");

        // Wait for shutdown signal
        self.wait_for_shutdown().await;

        info!("🛑 Shutdown signal received, gracefully stopping...");
        self.shutdown().await?;

        Ok(())
    }

    fn print_config(&self) {
        info!("Configuration:");
        info!("  Assets: {:?}", self.config.assets);
        info!("  Primary Exchange: {}", self.config.exchanges.primary);
        info!("  Secondary Exchange: {}", self.config.exchanges.secondary);
        info!(
            "  QuestDB: {}:{}",
            self.config.questdb.host, self.config.questdb.ilp_port
        );
        info!("  Redis: {}", self.config.redis.url);
        info!("  Backfill: {}", self.config.operational.enable_backfill);
        info!("  Failover: {}", self.config.operational.enable_failover);
    }

    async fn init_storage(&self) -> Result<Arc<storage::StorageManager>> {
        storage::StorageManager::new(self.config.clone(), self.shutdown_tx.subscribe()).await
    }

    async fn init_connectors(&self) -> Result<connectors::ConnectorManager> {
        connectors::ConnectorManager::new(self.config.clone(), self.shutdown_tx.subscribe()).await
    }

    async fn init_metrics(&self) -> Result<metrics::MetricsManager> {
        metrics::MetricsManager::new(self.config.clone(), self.shutdown_tx.subscribe()).await
    }

    async fn start_router(
        &self,
        storage: Arc<storage::StorageManager>,
    ) -> Result<Arc<actors::Router>> {
        actors::Router::new(self.config.clone(), storage, self.shutdown_tx.subscribe()).await
    }

    async fn start_indicator_actor(
        &self,
        storage: Arc<storage::StorageManager>,
    ) -> Result<Arc<actors::IndicatorActor>> {
        actors::IndicatorActor::new(self.config.clone(), storage, self.shutdown_tx.subscribe())
            .await
    }

    async fn start_signal_actor(
        &self,
        storage: Arc<storage::StorageManager>,
        indicator_actor: Arc<actors::IndicatorActor>,
    ) -> Result<Arc<SignalActor>> {
        // Subscribe to indicator updates from the broadcast channel
        let indicator_rx = indicator_actor.indicator_tx.subscribe();

        // Create signal actor
        SignalActor::new(
            self.config.clone(),
            storage,
            indicator_rx,
            self.shutdown_tx.subscribe(),
        )
        .await
    }

    async fn start_api_server(
        &self,
        storage: Arc<storage::StorageManager>,
        broadcast_tx: broadcast::Sender<actors::router::NormalizedMessage>,
        indicator_actor: Arc<actors::IndicatorActor>,
        signal_actor: Arc<SignalActor>,
    ) -> Result<tokio::task::JoinHandle<()>> {
        // Create app state with indicator and signal actors
        let state =
            api::AppState::with_actors(storage, broadcast_tx, indicator_actor, signal_actor);

        // Build router
        let app = api::build_router(state);

        // Spawn server
        let handle = tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
                .await
                .expect("Failed to bind to port 8080");

            info!("API server listening on 0.0.0.0:8080");

            axum::serve(listener, app).await.expect("API server failed");
        });

        Ok(handle)
    }

    /// Warm up indicators from historical candle data.
    ///
    /// Symbols are passed as-is from the config (e.g. "MGC", "BTC").  The
    /// three-tier source chain in `IndicatorWarmup` handles normalisation:
    ///   1. Python data service  — authoritative for CME futures + crypto
    ///   2. QuestDB              — crypto already ingested via WebSocket
    ///   3. Binance REST         — crypto last resort
    async fn warmup_indicators(
        &self,
        storage: Arc<storage::StorageManager>,
        indicator_actor: Arc<actors::IndicatorActor>,
    ) {
        use crate::backfill::IndicatorWarmup;

        // Build list of symbol/timeframe pairs to warm up.
        // Pass symbols as configured — the PythonDataClient converts them to
        // the right format for the data service (e.g. "MGC" → "MGC=F").
        let pairs: Vec<(String, String)> = self
            .config
            .assets
            .iter()
            .map(|asset| (asset.to_uppercase(), "1m".to_string()))
            .collect();

        if pairs.is_empty() {
            info!("IndicatorWarmup: No pairs configured for warmup");
            return;
        }

        info!(
            "IndicatorWarmup: Warming up {} pairs (python-data → QuestDB → Binance fallback chain)",
            pairs.len()
        );

        let mut warmup = IndicatorWarmup::new(storage, indicator_actor);
        let results = warmup.warmup_all(&pairs).await;

        // Log results
        let total_candles: usize = results.iter().map(|r| r.candles_processed).sum();
        let ready_count = results.iter().filter(|r| r.all_indicators_ready).count();

        for result in &results {
            if result.candles_processed > 0 {
                info!(
                    "  ✅ {}:{} - {} candles, {}ms, ready={}",
                    result.symbol,
                    result.timeframe,
                    result.candles_processed,
                    result.duration_ms,
                    result.all_indicators_ready
                );
            } else {
                warn!(
                    "  ⚠️  {}:{} - No historical candles found",
                    result.symbol, result.timeframe
                );
            }
        }

        info!(
            "IndicatorWarmup: Complete - {} candles processed, {}/{} pairs fully warmed",
            total_candles,
            ready_count,
            results.len()
        );
    }

    async fn start_websockets(
        &self,
        connectors: connectors::ConnectorManager,
        router: Arc<actors::Router>,
    ) -> Result<()> {
        // This will spawn WebSocket actors for each asset/exchange pair
        connectors.start_all(router).await
    }

    fn should_init_metrics(&self) -> bool {
        self.config.metrics.enable_fear_greed
            || self.config.metrics.enable_etf_flows
            || self.config.metrics.enable_volatility
            || self.config.metrics.enable_altcoin_season
    }

    async fn wait_for_shutdown(&self) {
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("Received SIGINT (Ctrl+C)");
            }
            Err(err) => {
                error!("Unable to listen for shutdown signal: {}", err);
            }
        }
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Broadcasting shutdown signal to all actors...");

        // Send shutdown signal to all actors
        if let Err(e) = self.shutdown_tx.send(()) {
            warn!("Failed to send shutdown signal: {}", e);
        }

        // Give actors time to flush buffers and close connections gracefully
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        info!("✅ Data Factory shutdown complete");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing/logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,fks_ruby=debug".to_string()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .with_level(true)
        .with_ansi(true)
        .init();

    // ASCII Art Banner
    println!(
        r#"
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║     ██╗ █████╗ ███╗   ██╗██╗   ██╗███╗   ███╗                ║
    ║     ██║██╔══██╗████╗  ██║██║   ██║████╗ ████║                ║
    ║     ██║███████║██╔██╗ ██║██║   ██║██╔████╔██║                ║
    ║██   ██║██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║                ║
    ║╚█████╔╝██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║                ║
    ║ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝                ║
    ║                                                                ║
    ║              🏭 DATA FACTORY - Market Data Ingestion           ║
    ║              Version: 0.1.0                                    ║
    ║              Architecture: Actor Model (Tokio)                 ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    "#
    );

    // Load configuration
    info!("📋 Loading configuration from environment...");
    let config = Config::from_env().context("Failed to load configuration")?;

    // Validate configuration
    config
        .validate()
        .context("Configuration validation failed")?;
    info!("✅ Configuration loaded and validated");

    // Create and run the Data Factory
    let factory = DataFactory::new(config).context("Failed to initialize Data Factory")?;

    factory.run().await
}
