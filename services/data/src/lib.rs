//! FKS Data Service Library
//!
//! This library exposes core functionality for the FKS Data Service,
//! a standalone market data ingestion microservice extracted from JANUS.
//!
//! ## Unified Binary Integration
//!
//! When running inside the unified JANUS binary, the Data module's
//! [`start_module`] function checks the `DATA_SOURCE` environment variable:
//!
//! - `DATA_SOURCE=live` — connects to exchange WebSockets (Binance by default),
//!   ingests real-time kline/trade data, and publishes normalised
//!   [`MarketDataEvent`](janus_core::MarketDataEvent)s to the shared
//!   [`MarketDataBus`](janus_core::MarketDataBus) so that the Forward module
//!   can consume them for indicator calculation and strategy evaluation.
//!
//! - `DATA_SOURCE=standby` (default) — health-reporting only, no live
//!   ingestion. This preserves backward compatibility with the existing
//!   paper-trading soak tests that use synthetic signals.
//!
//! ## Environment Variables (live mode)
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `DATA_SOURCE` | `standby` | `live` to enable real-time ingestion |
//! | `DATA_EXCHANGE` | `binance` | Primary exchange connector |
//! | `DATA_WS_URL` | `wss://stream.binance.com:9443/ws` | WebSocket endpoint |
//! | `DATA_KLINE_INTERVALS` | `1m,5m` | Comma-separated kline intervals |
//! | `DATA_RECONNECT_DELAY_SECS` | `5` | Delay between reconnection attempts |
//! | `DATA_MAX_RECONNECT_ATTEMPTS` | `50` | Max consecutive reconnect failures |
//! | `DATA_HEALTH_POLL_SECS` | `10` | Health reporter tick interval |

pub mod actors;
pub mod backfill;
pub mod config;
pub mod connectors;
pub mod data_service_provider;
pub mod logging;
pub mod metrics;
pub mod self_healing;
pub mod storage;

// Re-export commonly used types
pub use backfill::{
    BackfillExecutor, BackfillLock, BackfillRequest, BackfillResult, LockConfig, LockGuard,
    LockMetrics,
};
pub use data_service_provider::DataServiceProvider;
pub use logging::CorrelationId;
pub use self_healing::{
    HealthStatus, RemediationResult, RemediationStats, RemediationType, SelfHealingConfig,
    SelfHealingEngine,
};

use std::sync::Arc;
use tracing::{debug, error, info, warn};

// Prometheus metrics — lazy statics registered in the global default registry.
// Importing them here ensures they get incremented on the live ingestion hot paths.
use crate::metrics::prometheus_exporter::{
    ACTIVE_SIGNALS, BACKFILL_DEDUP_HITS, BACKFILL_DEDUP_MISSES, BACKFILL_DEDUP_SET_SIZE,
    BACKFILL_DURATION, BACKFILL_LOCK_ACQUIRED, BACKFILL_LOCK_FAILED, BACKFILL_MAX_RETRIES_EXCEEDED,
    BACKFILL_QUEUE_SIZE, BACKFILL_RETRIES, BACKFILL_THROTTLE_REJECTIONS, BACKFILLS_COMPLETED,
    BACKFILLS_RUNNING, CIRCUIT_BREAKER_STATE, DATA_COMPLETENESS, GAP_DETECTION_ACCURACY,
    GAP_DETECTION_ACTIVE_GAPS, GAP_SIZE_TRADES, GAPS_DETECTED, INDICATOR_CALCULATION_DURATION,
    INDICATOR_PAIRS_TRACKED, INDICATOR_WARMUP_PROGRESS, INDICATORS_CALCULATED, INGESTION_LATENCY,
    QUESTDB_DISK_USAGE, QUESTDB_DISK_USAGE_BYTES, QUESTDB_WRITE_ERRORS, QUESTDB_WRITE_LATENCY,
    QUESTDB_WRITES, RATE_LIMITER_ACCEPTED, RATE_LIMITER_REJECTED, RATE_LIMITER_REQUESTS,
    RATE_LIMITER_TOKENS, SIGNAL_STRENGTH, SIGNALS_GENERATED, SYSTEM_UPTIME, TRADES_INGESTED,
    TRADES_PER_SECOND, WEBSOCKET_CONNECTED, WEBSOCKET_RECONNECTIONS,
};

// ═══════════════════════════════════════════════════════════════════════════
// Live-mode internal types & helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the live data ingestion mode inside the unified binary.
#[derive(Debug, Clone)]
struct LiveDataConfig {
    /// Exchange to connect to (e.g. "binance")
    exchange: String,
    /// WebSocket base URL
    ws_url: String,
    /// Kline intervals to subscribe to (e.g. ["1m", "5m"])
    kline_intervals: Vec<String>,
    /// Assets to subscribe (base symbols, e.g. ["BTC", "ETH", "SOL"])
    assets: Vec<String>,
    /// Default quote currency
    quote: String,
    /// Reconnection delay between attempts
    reconnect_delay_secs: u64,
    /// Maximum consecutive reconnection attempts before circuit-breaking
    max_reconnect_attempts: u32,
    /// Health poll interval
    health_poll_secs: u64,
}

impl LiveDataConfig {
    /// Build from environment variables + JanusState config.
    fn from_env(state: &janus_core::JanusState) -> Self {
        let exchange =
            std::env::var("DATA_EXCHANGE").unwrap_or_else(|_| state.config.market.exchange.clone());

        let ws_url = std::env::var("DATA_WS_URL").unwrap_or_else(|_| {
            match exchange.to_lowercase().as_str() {
                "binance" => "wss://stream.binance.com:9443/ws".to_string(),
                "bybit" => "wss://stream.bybit.com/v5/public/spot".to_string(),
                other => {
                    warn!(
                        "DATA_WS_URL not set and no default for exchange '{}', using binance",
                        other
                    );
                    "wss://stream.binance.com:9443/ws".to_string()
                }
            }
        });

        let kline_intervals: Vec<String> = std::env::var("DATA_KLINE_INTERVALS")
            .unwrap_or_else(|_| "1m,5m".to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let assets: Vec<String> = state.config.assets.enabled.clone();
        let quote = state.config.assets.default_quote.clone();

        let reconnect_delay_secs: u64 = std::env::var("DATA_RECONNECT_DELAY_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        let max_reconnect_attempts: u32 = std::env::var("DATA_MAX_RECONNECT_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);

        let health_poll_secs: u64 = std::env::var("DATA_HEALTH_POLL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        Self {
            exchange,
            ws_url,
            kline_intervals,
            assets,
            quote,
            reconnect_delay_secs,
            max_reconnect_attempts,
            health_poll_secs,
        }
    }

    /// Build the Binance combined-stream URL for one symbol.
    ///
    /// Uses the `/stream?streams=...` endpoint to subscribe to multiple kline
    /// intervals and optionally the trade stream in a single WebSocket.
    fn binance_stream_url(&self, asset: &str) -> String {
        let symbol_lower = format!("{}{}", asset, self.quote).to_lowercase();
        let base_url = self.ws_url.replace("/ws", "/stream");

        let mut streams: Vec<String> = Vec::new();

        // Trade stream for trade-level data
        streams.push(format!("{}@trade", symbol_lower));

        // Kline streams for each interval
        for interval in &self.kline_intervals {
            streams.push(format!("{}@kline_{}", symbol_lower, interval));
        }

        format!("{}?streams={}", base_url, streams.join("/"))
    }
}

/// Statistics for a single WebSocket ingestion task.
#[derive(Debug, Default)]
struct IngestionStats {
    trades_received: std::sync::atomic::AtomicU64,
    klines_received: std::sync::atomic::AtomicU64,
    klines_published: std::sync::atomic::AtomicU64,
    errors: std::sync::atomic::AtomicU64,
    reconnects: std::sync::atomic::AtomicU64,
}

// ═══════════════════════════════════════════════════════════════════════════
// start_module — unified JANUS binary entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Start the data service module as part of the unified JANUS system.
///
/// This function is called by the unified JANUS binary to start the data
/// ingestion module. Its behaviour depends on the `DATA_SOURCE` environment
/// variable:
///
/// - `live`    — connect to exchange WebSockets, ingest market data, publish
///               to [`MarketDataBus`](janus_core::MarketDataBus).
/// - `standby` — health-reporting only (default, backward-compatible).
#[tracing::instrument(name = "data::start_module", skip(state))]
pub async fn start_module(state: Arc<janus_core::JanusState>) -> janus_core::Result<()> {
    info!("Data module registered — waiting for start command...");

    state
        .register_module_health("data", true, Some("standby".to_string()))
        .await;

    // ── Wait for services to be started via API / web interface ──────
    if !state.wait_for_services_start().await {
        info!("Data module: shutdown requested before services started");
        state
            .register_module_health("data", false, Some("shutdown_before_start".to_string()))
            .await;
        return Ok(());
    }

    info!("Starting Data module...");
    state
        .register_module_health("data", true, Some("starting".to_string()))
        .await;

    // ── Check data source mode ───────────────────────────────────────
    let data_source = std::env::var("DATA_SOURCE")
        .unwrap_or_else(|_| "standby".to_string())
        .to_lowercase();

    match data_source.as_str() {
        "live" => {
            info!("📡 DATA_SOURCE=live — starting live market data ingestion");
            run_live_mode(state).await
        }
        "standby" | "" => {
            info!("⏸️  DATA_SOURCE=standby — health-reporting only (no live ingestion)");
            run_standby_mode(state).await
        }
        other => {
            warn!(
                "⚠️  Unknown DATA_SOURCE='{}' — falling back to standby mode",
                other
            );
            run_standby_mode(state).await
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Standby mode — original stub behaviour (health reporting only)
// ═══════════════════════════════════════════════════════════════════════════

async fn run_standby_mode(state: Arc<janus_core::JanusState>) -> janus_core::Result<()> {
    let health_poll_secs: u64 = std::env::var("DATA_HEALTH_POLL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    // Health-reporting background task
    let state_for_health = state.clone();
    let health_handle = tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(tokio::time::Duration::from_secs(health_poll_secs));
        let mut tick_count: u64 = 0;

        info!("Data health reporter started (standby mode)");

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    tick_count += 1;

                    let status_msg = format!("standby (ticks: {})", tick_count);
                    state_for_health
                        .register_module_health("data", true, Some(status_msg))
                        .await;

                    if tick_count.is_multiple_of(30) {
                        info!(tick_count, "Data module health reporter alive (standby)");
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(500)) => {
                    if state_for_health.is_shutdown_requested() {
                        break;
                    }
                }
            }
        }

        info!("Data health reporter stopped (standby)");
    });

    state
        .register_module_health("data", true, Some("running (standby)".to_string()))
        .await;

    info!("Data module running (standby mode — no live ingestion)");

    // Keep alive until shutdown
    while !state.is_shutdown_requested() {
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }

    info!("Data module shutting down (standby)...");
    health_handle.abort();

    state
        .register_module_health("data", false, Some("stopped".to_string()))
        .await;

    info!("Data module exited (standby)");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Live mode — real-time market data ingestion via WebSocket
// ═══════════════════════════════════════════════════════════════════════════

async fn run_live_mode(state: Arc<janus_core::JanusState>) -> janus_core::Result<()> {
    let config = LiveDataConfig::from_env(&state);

    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║         DATA MODULE — LIVE INGESTION MODE                ║");
    info!("╚═══════════════════════════════════════════════════════════╝");
    info!("  Exchange:        {}", config.exchange);
    info!("  WebSocket URL:   {}", config.ws_url);
    info!("  Kline intervals: {:?}", config.kline_intervals);
    info!("  Assets:          {:?}", config.assets);
    info!("  Quote:           {}", config.quote);
    info!(
        "  Reconnect:       {}s delay, max {} attempts",
        config.reconnect_delay_secs, config.max_reconnect_attempts
    );

    // ── Eagerly initialize all Prometheus lazy statics ────────────────
    // This ensures every metric appears in the /metrics output (as zero)
    // even before the first event, preventing "No data" in Grafana panels.
    {
        // Force Lazy initialization by touching each static.
        let _ = &*DATA_COMPLETENESS;
        let _ = &*GAPS_DETECTED;
        let _ = &*GAP_SIZE_TRADES;
        let _ = &*BACKFILL_QUEUE_SIZE;
        let _ = &*INGESTION_LATENCY;
        let _ = &*TRADES_INGESTED;
        let _ = &*TRADES_PER_SECOND;
        let _ = &*RATE_LIMITER_REQUESTS;
        let _ = &*RATE_LIMITER_ACCEPTED;
        let _ = &*RATE_LIMITER_REJECTED;
        let _ = &*RATE_LIMITER_TOKENS;
        let _ = &*CIRCUIT_BREAKER_STATE;
        let _ = &*WEBSOCKET_CONNECTED;
        let _ = &*WEBSOCKET_RECONNECTIONS;
        let _ = &*SYSTEM_UPTIME;
        let _ = &*QUESTDB_DISK_USAGE;
        let _ = &*QUESTDB_WRITES;
        let _ = &*QUESTDB_WRITE_ERRORS;
        let _ = &*BACKFILLS_COMPLETED;
        let _ = &*BACKFILL_DURATION;
        let _ = &*BACKFILLS_RUNNING;
        let _ = &*BACKFILL_RETRIES;
        let _ = &*BACKFILL_MAX_RETRIES_EXCEEDED;
        let _ = &*BACKFILL_DEDUP_HITS;
        let _ = &*BACKFILL_DEDUP_MISSES;
        let _ = &*BACKFILL_DEDUP_SET_SIZE;
        let _ = &*BACKFILL_LOCK_ACQUIRED;
        let _ = &*BACKFILL_LOCK_FAILED;
        let _ = &*BACKFILL_THROTTLE_REJECTIONS;
        let _ = &*QUESTDB_WRITE_LATENCY;
        let _ = &*QUESTDB_DISK_USAGE_BYTES;
        let _ = &*GAP_DETECTION_ACCURACY;
        let _ = &*GAP_DETECTION_ACTIVE_GAPS;
        let _ = &*INDICATORS_CALCULATED;
        let _ = &*INDICATOR_CALCULATION_DURATION;
        let _ = &*INDICATOR_WARMUP_PROGRESS;
        let _ = &*INDICATOR_PAIRS_TRACKED;
        let _ = &*SIGNALS_GENERATED;
        let _ = &*ACTIVE_SIGNALS;
        let _ = &*SIGNAL_STRENGTH;
        info!("📊 Prometheus metrics initialized (38 data-factory statics registered)");
    }

    if config.assets.is_empty() {
        warn!("No assets configured — live ingestion has nothing to subscribe to");
        state
            .register_module_health("data", false, Some("no assets configured".to_string()))
            .await;
        return run_standby_mode(state).await;
    }

    // Shared stats across all per-asset tasks
    let stats = Arc::new(IngestionStats::default());

    // ── Spawn one WebSocket task per asset ────────────────────────────
    let mut task_handles = Vec::new();

    for asset in &config.assets {
        let asset = asset.clone();
        let config = config.clone();
        let state = state.clone();
        let stats = stats.clone();

        let handle = tokio::spawn(async move {
            run_asset_ws(asset, config, state, stats).await;
        });
        task_handles.push(handle);

        // Stagger connections slightly to avoid burst
        tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
    }

    // ── Health reporter ──────────────────────────────────────────────
    let state_health = state.clone();
    let stats_health = stats.clone();
    let health_poll_secs = config.health_poll_secs;
    let asset_count = config.assets.len();

    let health_handle = tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(tokio::time::Duration::from_secs(health_poll_secs));
        let mut tick_count: u64 = 0;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    tick_count += 1;

                    let trades = stats_health.trades_received.load(std::sync::atomic::Ordering::Relaxed);
                    let klines = stats_health.klines_received.load(std::sync::atomic::Ordering::Relaxed);
                    let published = stats_health.klines_published.load(std::sync::atomic::Ordering::Relaxed);
                    let errors = stats_health.errors.load(std::sync::atomic::Ordering::Relaxed);
                    let reconnects = stats_health.reconnects.load(std::sync::atomic::Ordering::Relaxed);

                    // Update Prometheus system uptime gauge
                    SYSTEM_UPTIME.set((tick_count * health_poll_secs) as f64);

                    // Update Prometheus trade counters from atomic stats so
                    // dashboards see cumulative values even between scrapes.
                    BACKFILL_QUEUE_SIZE.set(0); // keep metric alive; real value set by scheduler

                    let status = format!(
                        "live: {} assets, {} trades, {} klines ({} published), {} errors, {} reconnects",
                        asset_count, trades, klines, published, errors, reconnects,
                    );

                    state_health
                        .register_module_health("data", true, Some(status.clone()))
                        .await;

                    if tick_count.is_multiple_of(6) {
                        // Log every ~60s at default 10s interval
                        info!("📊 Data ingestion stats: {}", status);
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(500)) => {
                    if state_health.is_shutdown_requested() {
                        break;
                    }
                }
            }
        }
    });

    state
        .register_module_health(
            "data",
            true,
            Some(format!("live: {} assets connected", config.assets.len())),
        )
        .await;

    info!(
        "Data module running (live mode — {} asset WebSocket tasks spawned)",
        task_handles.len()
    );

    // ── Wait for shutdown ────────────────────────────────────────────
    while !state.is_shutdown_requested() {
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    info!("Data module shutting down (live mode)...");

    // Cancel all tasks
    health_handle.abort();
    for handle in task_handles {
        handle.abort();
    }

    // Final stats
    let trades = stats
        .trades_received
        .load(std::sync::atomic::Ordering::Relaxed);
    let klines = stats
        .klines_received
        .load(std::sync::atomic::Ordering::Relaxed);
    let published = stats
        .klines_published
        .load(std::sync::atomic::Ordering::Relaxed);
    let errors = stats.errors.load(std::sync::atomic::Ordering::Relaxed);
    let reconnects = stats.reconnects.load(std::sync::atomic::Ordering::Relaxed);

    info!(
        "Data module final stats: {} trades, {} klines ({} published), {} errors, {} reconnects",
        trades, klines, published, errors, reconnects
    );

    state
        .register_module_health("data", false, Some("stopped".to_string()))
        .await;

    info!("Data module exited (live mode)");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-asset WebSocket ingestion loop
// ═══════════════════════════════════════════════════════════════════════════

/// Run a persistent WebSocket connection for a single asset, reconnecting
/// on failure up to `max_reconnect_attempts` consecutive times.
async fn run_asset_ws(
    asset: String,
    config: LiveDataConfig,
    state: Arc<janus_core::JanusState>,
    stats: Arc<IngestionStats>,
) {
    use futures_util::StreamExt;
    use tokio_tungstenite::tungstenite::Message;

    let stream_url = match config.exchange.to_lowercase().as_str() {
        "binance" => config.binance_stream_url(&asset),
        other => {
            // For now only Binance is fully supported in live mode.
            // Other exchanges can be added by implementing their own URL builder
            // and message parser.
            warn!(
                "Exchange '{}' is not yet supported for live ingestion — using Binance URL format",
                other
            );
            config.binance_stream_url(&asset)
        }
    };

    let symbol_display = format!("{}{}", asset.to_uppercase(), config.quote.to_uppercase());

    info!(
        "📡 [{}] Connecting to WebSocket: {}",
        symbol_display, stream_url
    );

    let mut consecutive_failures: u32 = 0;

    loop {
        if state.is_shutdown_requested() {
            info!(
                "[{}] Shutdown requested, stopping WebSocket task",
                symbol_display
            );
            break;
        }

        // ── Connect ──────────────────────────────────────────────────
        let ws_result = tokio::time::timeout(
            tokio::time::Duration::from_secs(30),
            tokio_tungstenite::connect_async(&stream_url),
        )
        .await;

        let (ws_stream, _response) = match ws_result {
            Ok(Ok((stream, response))) => {
                info!(
                    "✅ [{}] WebSocket connected (status: {})",
                    symbol_display,
                    response.status()
                );
                consecutive_failures = 0;
                // Update Prometheus websocket connection gauge
                WEBSOCKET_CONNECTED
                    .with_label_values(&[&config.exchange])
                    .set(1);
                (stream, response)
            }
            Ok(Err(e)) => {
                consecutive_failures += 1;
                stats
                    .errors
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                error!(
                    "❌ [{}] WebSocket connection failed ({}/{}): {}",
                    symbol_display, consecutive_failures, config.max_reconnect_attempts, e
                );
                if consecutive_failures >= config.max_reconnect_attempts {
                    error!(
                        "🚨 [{}] Max reconnect attempts reached — giving up",
                        symbol_display
                    );
                    WEBSOCKET_CONNECTED
                        .with_label_values(&[&config.exchange])
                        .set(0);
                    break;
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(
                    config.reconnect_delay_secs,
                ))
                .await;
                continue;
            }
            Err(_timeout) => {
                consecutive_failures += 1;
                stats
                    .errors
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                warn!(
                    "⏱️  [{}] WebSocket connection timed out ({}/{})",
                    symbol_display, consecutive_failures, config.max_reconnect_attempts
                );
                if consecutive_failures >= config.max_reconnect_attempts {
                    error!(
                        "🚨 [{}] Max reconnect attempts reached — giving up",
                        symbol_display
                    );
                    WEBSOCKET_CONNECTED
                        .with_label_values(&[&config.exchange])
                        .set(0);
                    break;
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(
                    config.reconnect_delay_secs,
                ))
                .await;
                continue;
            }
        };

        // ── Message processing loop ──────────────────────────────────
        let (mut _write, mut read) = ws_stream.split();

        loop {
            if state.is_shutdown_requested() {
                break;
            }

            let msg = tokio::select! {
                msg = read.next() => msg,
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(60)) => {
                    // No message in 60s — possible stale connection
                    warn!("[{}] No message received in 60s, reconnecting...", symbol_display);
                    stats.reconnects.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    WEBSOCKET_RECONNECTIONS
                        .with_label_values(&[&config.exchange])
                        .inc();
                    WEBSOCKET_CONNECTED
                        .with_label_values(&[&config.exchange])
                        .set(0);
                    break;
                }
            };

            match msg {
                Some(Ok(Message::Text(text))) => {
                    // Parse Binance combined-stream message
                    if let Err(e) =
                        process_binance_message(&text, &asset, &config.quote, &state, &stats).await
                    {
                        debug!("[{}] Failed to process message: {}", symbol_display, e);
                        // Don't count parse issues for unrecognised message types
                        // as errors (subscription confirmations, pong frames, etc.)
                    }
                }
                Some(Ok(Message::Ping(data))) => {
                    // Respond with pong — we need the write half for this
                    // tokio-tungstenite auto-responds to pings in most cases,
                    // but log for observability.
                    debug!("[{}] Received ping ({} bytes)", symbol_display, data.len());
                }
                Some(Ok(Message::Pong(_))) => {
                    debug!("[{}] Received pong", symbol_display);
                }
                Some(Ok(Message::Close(frame))) => {
                    warn!(
                        "[{}] WebSocket closed by server: {:?}",
                        symbol_display, frame
                    );
                    stats
                        .reconnects
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
                Some(Ok(Message::Binary(_))) => {
                    debug!("[{}] Received binary message (ignored)", symbol_display);
                }
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    warn!(
                        "[{}] WebSocket read error: {} — reconnecting",
                        symbol_display, e
                    );
                    stats
                        .errors
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    stats
                        .reconnects
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    WEBSOCKET_RECONNECTIONS
                        .with_label_values(&[&config.exchange])
                        .inc();
                    WEBSOCKET_CONNECTED
                        .with_label_values(&[&config.exchange])
                        .set(0);
                    break;
                }
                None => {
                    warn!("[{}] WebSocket stream ended — reconnecting", symbol_display);
                    stats
                        .reconnects
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    WEBSOCKET_RECONNECTIONS
                        .with_label_values(&[&config.exchange])
                        .inc();
                    WEBSOCKET_CONNECTED
                        .with_label_values(&[&config.exchange])
                        .set(0);
                    break;
                }
            }
        }

        // Reconnect after a delay
        if !state.is_shutdown_requested() {
            let delay = config.reconnect_delay_secs;
            info!("[{}] Reconnecting in {}s...", symbol_display, delay);
            tokio::time::sleep(tokio::time::Duration::from_secs(delay)).await;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Binance message parsing → MarketDataEvent publishing
// ═══════════════════════════════════════════════════════════════════════════

/// Binance combined-stream wrapper: `{ "stream": "...", "data": {...} }`
#[derive(serde::Deserialize)]
struct BinanceCombinedStream {
    stream: String,
    data: serde_json::Value,
}

/// Binance kline outer wrapper
#[derive(serde::Deserialize)]
struct BinanceKlineMsg {
    #[allow(dead_code)]
    e: String, // "kline"
    #[allow(dead_code)]
    #[serde(rename = "E")]
    event_time: i64,
    #[allow(dead_code)]
    s: String, // symbol
    k: BinanceKlineData,
}

/// Binance kline inner data
#[derive(serde::Deserialize)]
struct BinanceKlineData {
    /// Kline start time (ms)
    t: i64,
    /// Kline close time (ms)
    #[serde(rename = "T")]
    close_time: i64,
    /// Symbol
    #[allow(dead_code)]
    s: String,
    /// Interval
    i: String,
    /// Open price
    o: String,
    /// Close price
    c: String,
    /// High price
    h: String,
    /// Low price
    l: String,
    /// Volume
    v: String,
    /// Quote volume
    q: String,
    /// Number of trades
    n: u64,
    /// Is this kline closed?
    x: bool,
}

/// Binance trade message
#[derive(serde::Deserialize)]
struct BinanceTradeMsg {
    #[allow(dead_code)]
    e: String, // "trade"
    #[allow(dead_code)]
    s: String, // symbol
    /// Trade ID
    t: u64,
    /// Price (string)
    p: String,
    /// Quantity (string)
    q: String,
    /// Trade time (ms)
    #[serde(rename = "T")]
    trade_time: i64,
    /// Is buyer the market maker?
    m: bool,
}

/// Process a raw Binance WebSocket text message and publish to MarketDataBus.
async fn process_binance_message(
    raw: &str,
    asset: &str,
    quote: &str,
    state: &Arc<janus_core::JanusState>,
    stats: &Arc<IngestionStats>,
) -> Result<(), String> {
    // Try to parse as combined-stream wrapper
    if let Ok(wrapper) = serde_json::from_str::<BinanceCombinedStream>(raw) {
        if wrapper.stream.contains("@kline") {
            return process_kline_data(&wrapper.data, asset, quote, state, stats).await;
        } else if wrapper.stream.contains("@trade") {
            return process_trade_data(&wrapper.data, asset, quote, state, stats).await;
        }
        // Ignore other stream types (subscription confirmations etc.)
        return Ok(());
    }

    // Try direct kline message (non-combined)
    if raw.contains("\"e\":\"kline\"") {
        let data: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| format!("kline parse: {}", e))?;
        return process_kline_data(&data, asset, quote, state, stats).await;
    }

    // Try direct trade message
    if raw.contains("\"e\":\"trade\"") {
        let data: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| format!("trade parse: {}", e))?;
        return process_trade_data(&data, asset, quote, state, stats).await;
    }

    // Subscription confirmations, pong frames, etc. — silently ignore
    if raw.contains("\"result\":null") || raw.contains("\"id\":") {
        return Ok(());
    }

    Err("Unrecognised message type".to_string())
}

/// Parse and publish a kline/candle event.
async fn process_kline_data(
    data: &serde_json::Value,
    asset: &str,
    quote: &str,
    state: &Arc<janus_core::JanusState>,
    stats: &Arc<IngestionStats>,
) -> Result<(), String> {
    use rust_decimal::Decimal;
    use std::str::FromStr;

    let kline: BinanceKlineMsg =
        serde_json::from_value(data.clone()).map_err(|e| format!("kline deser: {}", e))?;

    let k = &kline.k;

    stats
        .klines_received
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Only publish closed klines to avoid noisy partial updates.
    // The Forward module will receive complete candles for indicator calculation.
    if !k.x {
        return Ok(());
    }

    let open = Decimal::from_str(&k.o).map_err(|e| format!("open: {}", e))?;
    let high = Decimal::from_str(&k.h).map_err(|e| format!("high: {}", e))?;
    let low = Decimal::from_str(&k.l).map_err(|e| format!("low: {}", e))?;
    let close = Decimal::from_str(&k.c).map_err(|e| format!("close: {}", e))?;
    let volume = Decimal::from_str(&k.v).map_err(|e| format!("volume: {}", e))?;
    let quote_volume = Decimal::from_str(&k.q).ok();

    let symbol = janus_core::Symbol::new(asset.to_uppercase(), quote.to_uppercase());

    let event = janus_core::MarketDataEvent::Kline(janus_core::KlineEvent {
        exchange: janus_core::Exchange::Binance,
        symbol,
        interval: k.i.clone(),
        open_time: k.t * 1000, // ms → µs
        close_time: k.close_time * 1000,
        open,
        high,
        low,
        close,
        volume,
        quote_volume,
        trades: Some(k.n),
        is_closed: true,
    });

    match state.market_data_bus.publish(event) {
        Ok(receivers) => {
            stats
                .klines_published
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            debug!(
                "📈 Published closed kline {}{} {} O={} H={} L={} C={} V={} → {} receivers",
                asset, quote, k.i, k.o, k.h, k.l, k.c, k.v, receivers
            );
        }
        Err(e) => {
            // No subscribers yet — not an error, just informational
            debug!(
                "Kline published but no subscribers: {} ({})",
                e,
                if state.market_data_bus.subscriber_count() == 0 {
                    "Forward module may not be running yet"
                } else {
                    "broadcast channel error"
                }
            );
        }
    }

    Ok(())
}

/// Parse and publish a trade event.
async fn process_trade_data(
    data: &serde_json::Value,
    asset: &str,
    quote: &str,
    state: &Arc<janus_core::JanusState>,
    stats: &Arc<IngestionStats>,
) -> Result<(), String> {
    use rust_decimal::Decimal;
    use std::str::FromStr;

    let trade: BinanceTradeMsg =
        serde_json::from_value(data.clone()).map_err(|e| format!("trade deser: {}", e))?;

    stats
        .trades_received
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Update Prometheus trade counter + ingestion latency
    let exchange_label = "binance";
    let symbol_label = format!("{}{}", asset.to_uppercase(), quote.to_uppercase());
    TRADES_INGESTED
        .with_label_values(&[exchange_label, &symbol_label])
        .inc();
    // Compute ingestion latency: now minus exchange trade time
    let latency_ms = {
        let now_ms = chrono::Utc::now().timestamp_millis();
        (now_ms - trade.trade_time).max(0) as f64
    };
    INGESTION_LATENCY
        .with_label_values(&[exchange_label, &symbol_label])
        .observe(latency_ms);

    let price = Decimal::from_str(&trade.p).map_err(|e| format!("price: {}", e))?;
    let quantity = Decimal::from_str(&trade.q).map_err(|e| format!("qty: {}", e))?;

    let symbol = janus_core::Symbol::new(asset.to_uppercase(), quote.to_uppercase());

    let side = if trade.m {
        janus_core::Side::Sell // buyer is maker → aggressive sell
    } else {
        janus_core::Side::Buy
    };

    let received_at = chrono::Utc::now().timestamp_micros();

    let event = janus_core::MarketDataEvent::Trade(janus_core::TradeEvent {
        exchange: janus_core::Exchange::Binance,
        symbol,
        timestamp: trade.trade_time * 1000, // ms → µs
        received_at,
        price,
        quantity,
        side,
        trade_id: trade.t.to_string(),
        buyer_is_maker: Some(trade.m),
    });

    // Trades are high-frequency — only publish if there are subscribers.
    // This avoids filling the broadcast buffer when Forward only cares
    // about klines.
    if state.market_data_bus.subscriber_count() > 0 {
        let _ = state.market_data_bus.publish(event);
    }

    Ok(())
}
