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
//! | `DATA_PERSIST_CANDLES` | `true` | Write closed klines (and backfill) to QuestDB |
//! | `JANUS_BACKFILL_DAYS` | `30` | One-shot deep backfill depth in days (`0` = off) |
//! | `JANUS_CANDLE_SCAN` | on | Periodic candle gap scan/repair (`0` = off; standalone binary keeps its opt-in default) |
//! | `QUESTDB_HTTP_URL` | from config | QuestDB HTTP endpoint for coverage/scan reads |
//!
//! The `JANUS_CANDLE_SCAN_*` tuning knobs (symbols, interval, lookback,
//! cadence, window caps) from the standalone data binary are honoured
//! unchanged; unset ones default to the live config (symbols/exchange) and
//! one scan loop per `DATA_KLINE_INTERVALS` entry. See
//! [`backfill::unified`].

pub mod actors;
pub mod backfill;
pub mod candle_sink;
pub mod config;
pub mod connectors;
pub mod logging;
pub mod metrics;
pub mod panic_guard;
pub mod self_healing;
pub mod storage;

// Re-export commonly used types
pub use backfill::{
    BackfillExecutor, BackfillLock, BackfillRequest, BackfillResult, LockConfig, LockGuard,
    LockMetrics,
};
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
    /// CLOSED klines seen, independent of whether the bus publish succeeded
    /// (publishing fails with zero subscribers, e.g. forward not started).
    /// This is the numerator of the data-completeness SLI.
    klines_closed: std::sync::atomic::AtomicU64,
    klines_published: std::sync::atomic::AtomicU64,
    errors: std::sync::atomic::AtomicU64,
    reconnects: std::sync::atomic::AtomicU64,
}

/// Parse a kline interval string ("1m", "5m", "1h", "1d", …) into seconds.
/// Returns `None` for unrecognised units (e.g. Binance's "1M" month).
fn interval_secs(interval: &str) -> Option<u64> {
    let (num, unit) = interval.split_at(interval.len().checked_sub(1)?);
    let n: u64 = num.parse().ok()?;
    if n == 0 {
        return None;
    }
    let mult = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3_600,
        "d" => 86_400,
        "w" => 604_800,
        _ => return None,
    };
    Some(n * mult)
}

/// Closed klines expected on ONE symbol × ONE interval between two instants:
/// the number of whole interval boundaries crossed, minus a grace period so
/// the most recent boundary's kline (typically in flight for a second or two)
/// is never counted as missing.
fn expected_closed_klines(
    anchor_epoch: u64,
    now_epoch: u64,
    grace_secs: u64,
    interval_secs: u64,
) -> u64 {
    let effective_now = now_epoch.saturating_sub(grace_secs);
    if effective_now <= anchor_epoch || interval_secs == 0 {
        return 0;
    }
    (effective_now / interval_secs).saturating_sub(anchor_epoch / interval_secs)
}

/// Data-completeness percentage over a window: closed klines received vs
/// expected across all assets/intervals, capped at 100 (reconnect replays can
/// briefly over-deliver).
fn completeness_pct(received: u64, expected: u64) -> f64 {
    if expected == 0 {
        return 0.0;
    }
    (100.0 * received as f64 / expected as f64).min(100.0)
}

/// Baseline rolling window for the data-completeness SLI (30 minutes).
const COMPLETENESS_WINDOW_MIN_SECS: u64 = 1800;

/// Grace period shielding the most recent kline boundary (typically in
/// flight for a second or two) from being counted as missing.
const COMPLETENESS_GRACE_SECS: u64 = 15;

/// Rolling-window length (seconds) for the data-completeness SLI.
///
/// The pruning in [`CompletenessWindow::observe`] caps the anchor snapshot's
/// age at the window length, while the warm-up gate only lets the gauge
/// compute once the anchor is at least `2 * max_interval` old. A fixed
/// window therefore silently pins the gauge at 0 forever whenever any
/// configured interval exceeds half the window (>15m at the 1800s baseline,
/// e.g. `1h`). Scale the baseline up so the window always spans at least two
/// of the longest configured interval (`1h` → 7200s, `4h` → 28800s).
fn completeness_window_secs(max_interval_secs: u64) -> u64 {
    COMPLETENESS_WINDOW_MIN_SECS.max(2 * max_interval_secs)
}

/// Rolling snapshot window driving the `data_completeness_percent` gauge.
///
/// Holds `(epoch secs, cumulative closed klines)` snapshots; each tick the
/// window's oldest entry anchors the expectation: completeness =
/// closed-klines-received / boundary-aligned-expected over the window.
struct CompletenessWindow {
    /// Parsed kline interval durations contributing to the expectation.
    interval_secs: Vec<u64>,
    /// Number of symbols each interval is ingested for.
    asset_count: u64,
    /// Longest configured interval; drives the warm-up gate.
    max_interval: u64,
    /// Rolling-window length; always ≥ `2 * max_interval` (see
    /// [`completeness_window_secs`]).
    window_secs: u64,
    /// `(epoch secs, cumulative closed klines)` snapshots, oldest first.
    snapshots: std::collections::VecDeque<(u64, u64)>,
}

impl CompletenessWindow {
    fn new(interval_secs: Vec<u64>, asset_count: u64) -> Self {
        let max_interval = interval_secs.iter().copied().max().unwrap_or(0);
        Self {
            interval_secs,
            asset_count,
            max_interval,
            window_secs: completeness_window_secs(max_interval),
            snapshots: std::collections::VecDeque::new(),
        }
    }

    /// Record a snapshot and return the completeness percentage, or `None`
    /// while the warm-up gate is closed (the window does not yet span two of
    /// the longest interval, or there is no expectation), so cold starts
    /// never report a bogus low completeness and the gauge is left untouched.
    fn observe(&mut self, now_epoch: u64, closed_total: u64) -> Option<f64> {
        self.snapshots.push_back((now_epoch, closed_total));
        while self.snapshots.len() > 1 && self.snapshots[1].0 + self.window_secs <= now_epoch {
            self.snapshots.pop_front();
        }
        let &(anchor_ts, anchor_closed) = self.snapshots.front()?;
        if self.max_interval == 0 || now_epoch.saturating_sub(anchor_ts) < 2 * self.max_interval {
            return None;
        }
        let expected: u64 = self
            .interval_secs
            .iter()
            .map(|&i| expected_closed_klines(anchor_ts, now_epoch, COMPLETENESS_GRACE_SECS, i))
            .sum::<u64>()
            * self.asset_count;
        if expected == 0 {
            return None;
        }
        let received = closed_total.saturating_sub(anchor_closed);
        Some(completeness_pct(received, expected))
    }
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

    // ── Arm lazy per-label alert series ───────────────────────────────
    // CounterVecs register no series until their first increment, so
    // rate()-based alerts (BackfillFailureRateHigh, GapDetectionRateAnomaly,
    // BackfillMaxRetriesExceeded) could never evaluate before a first event
    // ever occurred. Touch the known label combinations so the series exist
    // at 0 and the alerts arm immediately.
    {
        let ex = config.exchange.as_str();
        for asset in &config.assets {
            let symbol = format!("{}{}", asset.to_uppercase(), config.quote.to_uppercase());
            let _ = GAPS_DETECTED.with_label_values(&[ex, &symbol]);
            let _ = BACKFILL_MAX_RETRIES_EXCEEDED.with_label_values(&[ex, &symbol]);
            for status in ["success", "failed"] {
                let _ = BACKFILLS_COMPLETED.with_label_values(&[ex, &symbol, status]);
            }
        }
    }

    // Shared stats across all per-asset tasks
    let stats = Arc::new(IngestionStats::default());

    // ── QuestDB candle persistence ────────────────────────────────────
    // Closed klines published to the MarketDataBus are also written to the
    // QuestDB `candles_crypto` table (the WebUI chart history source).
    // Disable with DATA_PERSIST_CANDLES=false.
    let candle_sink_handle = if candle_sink::persist_enabled() {
        info!(
            "  Candle persistence: enabled → {}:{} (candles_crypto)",
            state.config.questdb.host, state.config.questdb.ilp_port
        );
        Some(tokio::spawn(candle_sink::run(state.clone())))
    } else {
        warn!("  Candle persistence: DISABLED (DATA_PERSIST_CANDLES=false)");
        None
    };

    // ── Spawn one WebSocket task per asset ────────────────────────────
    let mut task_handles = Vec::new();

    for asset in &config.assets {
        let asset = asset.clone();
        let config = config.clone();
        let state = state.clone();
        let stats = stats.clone();

        let handle = tokio::spawn(async move {
            // Supervise the per-asset ingestion loop: if run_asset_ws gives
            // up (max reconnect attempts) or panics before shutdown, re-spawn
            // it with backoff so a sustained exchange outage can never
            // permanently kill this symbol's ingestion.
            let sup_name = format!("data-ws-{asset}");
            let shutdown_state = state.clone();
            janus_core::supervisor::supervise(
                &sup_name,
                move || shutdown_state.is_shutdown_requested(),
                move || run_asset_ws(asset.clone(), config.clone(), state.clone(), stats.clone()),
            )
            .await;
        });
        task_handles.push(handle);

        // Stagger connections slightly to avoid burst
        tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
    }

    // ── Historical backfill + gap scan (roadmap P1) ──────────────────
    // With live ingestion up, run the one-shot deep backfill
    // (`JANUS_BACKFILL_DAYS`, default 30, 0 = off) and then the periodic
    // candle-scan → scheduler loop (`JANUS_CANDLE_SCAN`, default ON here).
    // Everything runs in its own tasks; errors are logged and counted on
    // the backfill metrics, never propagated into the WS ingestion path.
    // Gated on the same switch as the live sink: with persistence off there
    // is no candle store to deepen.
    let backfill_handle = if candle_sink::persist_enabled() {
        let symbols: Vec<String> = config
            .assets
            .iter()
            .map(|a| format!("{}{}", a.to_uppercase(), config.quote.to_uppercase()))
            .collect();
        let questdb_http_url = std::env::var("QUESTDB_HTTP_URL").unwrap_or_else(|_| {
            format!(
                "http://{}:{}",
                state.config.questdb.host, state.config.questdb.http_port
            )
        });
        backfill::unified::spawn_backfill_and_scan(backfill::unified::UnifiedBackfillParams {
            exchange: config.exchange.clone(),
            symbols,
            intervals: config.kline_intervals.clone(),
            questdb_host: state.config.questdb.host.clone(),
            ilp_port: state.config.questdb.ilp_port,
            questdb_http_url,
            redis_url: state.config.redis.url.clone(),
        })
    } else {
        info!("Backfill: skipped (DATA_PERSIST_CANDLES=false — no candle store)");
        None
    };

    // ── Health reporter ──────────────────────────────────────────────
    let state_health = state.clone();
    let stats_health = stats.clone();
    let health_poll_secs = config.health_poll_secs;
    let asset_count = config.assets.len();

    // Completeness SLI inputs: parsed kline interval durations. Unparseable
    // intervals are excluded from the expectation (warned once here).
    let kline_interval_secs: Vec<u64> = config
        .kline_intervals
        .iter()
        .filter_map(|s| {
            let parsed = interval_secs(s);
            if parsed.is_none() {
                warn!("completeness SLI: unrecognised kline interval '{s}' — excluded");
            }
            parsed
        })
        .collect();

    // ── data_completeness_percent (rolling window) ────────────────────
    // The gauge stays untouched (and the DataCompletenessLow alert stays
    // gated) until the window spans two of the longest interval, so cold
    // starts never report a bogus low completeness. The window scales with
    // the longest configured interval so that warm-up gate can always open.
    let mut completeness = CompletenessWindow::new(kline_interval_secs, config.assets.len() as u64);
    if completeness.window_secs > COMPLETENESS_WINDOW_MIN_SECS {
        info!(
            "completeness SLI: longest kline interval is {}s — rolling window scaled from {}s to {}s so the gauge can warm up",
            completeness.max_interval, COMPLETENESS_WINDOW_MIN_SECS, completeness.window_secs,
        );
    }

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
                    let closed = stats_health.klines_closed.load(std::sync::atomic::Ordering::Relaxed);
                    let published = stats_health.klines_published.load(std::sync::atomic::Ordering::Relaxed);
                    let errors = stats_health.errors.load(std::sync::atomic::Ordering::Relaxed);
                    let reconnects = stats_health.reconnects.load(std::sync::atomic::Ordering::Relaxed);

                    // Update Prometheus system uptime gauge
                    SYSTEM_UPTIME.set((tick_count * health_poll_secs) as f64);

                    // Update Prometheus trade counters from atomic stats so
                    // dashboards see cumulative values even between scrapes.
                    BACKFILL_QUEUE_SIZE.set(0); // keep metric alive; real value set by scheduler

                    // ── data_completeness_percent ─────────────────────
                    if let Ok(d) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                        && let Some(pct) = completeness.observe(d.as_secs(), closed)
                    {
                        DATA_COMPLETENESS.set(pct);
                    }

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

    // Cancel all tasks. (The backfill orchestrator's inner scan/scheduler
    // tasks are detached fire-and-forget loops — the process is exiting.)
    health_handle.abort();
    if let Some(handle) = candle_sink_handle {
        handle.abort();
    }
    if let Some(handle) = backfill_handle {
        handle.abort();
    }
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

    // Count every closed kline for the completeness SLI, regardless of
    // whether the bus publish below succeeds (it errors with 0 subscribers).
    stats
        .klines_closed
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

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

#[cfg(test)]
mod completeness_tests {
    use super::{
        COMPLETENESS_WINDOW_MIN_SECS, CompletenessWindow, completeness_pct,
        completeness_window_secs, expected_closed_klines, interval_secs,
    };

    #[test]
    fn interval_parsing() {
        assert_eq!(interval_secs("1m"), Some(60));
        assert_eq!(interval_secs("5m"), Some(300));
        assert_eq!(interval_secs("1h"), Some(3_600));
        assert_eq!(interval_secs("4h"), Some(14_400));
        assert_eq!(interval_secs("1d"), Some(86_400));
        assert_eq!(interval_secs("30s"), Some(30));
        assert_eq!(interval_secs("1w"), Some(604_800));
        // Unsupported / malformed → None (excluded from the expectation).
        assert_eq!(interval_secs("1M"), None); // month — unit not supported
        assert_eq!(interval_secs("0m"), None);
        assert_eq!(interval_secs("m"), None);
        assert_eq!(interval_secs(""), None);
        assert_eq!(interval_secs("abc"), None);
    }

    #[test]
    fn expected_counts_whole_boundaries() {
        // Anchor at t=0, now=600s, no grace: ten 1m closes, two 5m closes.
        assert_eq!(expected_closed_klines(0, 600, 0, 60), 10);
        assert_eq!(expected_closed_klines(0, 600, 0, 300), 2);
        // Grace shields the most recent boundary: at now=605 with 15s grace,
        // effective now=590 → boundary at 600 not yet expected.
        assert_eq!(expected_closed_klines(0, 605, 15, 60), 9);
        // Unaligned anchor: boundaries at 120..=580 → floor(590/60)-floor(70/60) = 9-1 = 8.
        assert_eq!(expected_closed_klines(70, 605, 15, 60), 8);
        // Degenerate ranges never underflow or divide by zero.
        assert_eq!(expected_closed_klines(600, 600, 0, 60), 0);
        assert_eq!(expected_closed_klines(600, 605, 15, 60), 0);
        assert_eq!(expected_closed_klines(0, 600, 0, 0), 0);
    }

    #[test]
    fn completeness_is_capped_and_zero_safe() {
        assert_eq!(completeness_pct(10, 10), 100.0);
        assert_eq!(completeness_pct(999, 10), 100.0); // reconnect replay burst
        assert_eq!(completeness_pct(0, 0), 0.0); // no expectation → stay gated
        let pct = completeness_pct(999, 1000);
        assert!((pct - 99.9).abs() < 1e-9);
    }

    #[test]
    fn window_scales_with_longest_interval() {
        // Default deploys (1m,5m) keep the 30-minute baseline.
        assert_eq!(completeness_window_secs(60), COMPLETENESS_WINDOW_MIN_SECS);
        assert_eq!(completeness_window_secs(300), COMPLETENESS_WINDOW_MIN_SECS);
        assert_eq!(completeness_window_secs(900), COMPLETENESS_WINDOW_MIN_SECS);
        // Anything longer than 15m needs the window to grow so the 2×max
        // warm-up gate can open: 1h → 7200s, 4h → 28800s.
        assert_eq!(completeness_window_secs(3_600), 7_200);
        assert_eq!(completeness_window_secs(14_400), 28_800);
        // No parseable intervals → baseline (the gate stays closed anyway).
        assert_eq!(completeness_window_secs(0), COMPLETENESS_WINDOW_MIN_SECS);
    }

    /// Regression test for the >15m dead-gauge bug: with a 1h interval the
    /// old fixed 1800s window pruned the anchor before it could ever become
    /// `2 * 3600s` old, so the warm-up gate never opened and the gauge pinned
    /// at 0 forever. With the scaled window the gauge must go live once the
    /// window spans two hours.
    #[test]
    fn one_hour_interval_produces_live_gauge() {
        let mut w = CompletenessWindow::new(vec![3_600], 2); // 1h × 2 assets
        assert_eq!(w.window_secs, 7_200);

        let poll = 10; // default health_poll_secs
        let mut live_at = None;
        // Simulate perfect ingestion: every 1h boundary delivers one closed
        // kline per asset. Snapshot the cumulative count each poll tick.
        for tick in 0..=(3 * 3_600 / poll) {
            let now = 1_000_000 + tick * poll; // unaligned start, like real life
            let closed_total = 2 * (now / 3_600 - 1_000_000 / 3_600);
            if let Some(pct) = w.observe(now, closed_total) {
                live_at.get_or_insert(now - 1_000_000);
                assert_eq!(pct, 100.0, "perfect ingestion must read 100%");
            }
        }
        // The gauge went live once the anchor was 2×3600s old — not never
        // (the old bug), and not before the warm-up gate allows.
        let live_at = live_at.expect("gauge never went live — warm-up gate never opened");
        assert!(
            (7_200..7_200 + 2 * 3_600).contains(&live_at),
            "gauge went live at +{live_at}s, expected shortly after 7200s"
        );

        // And a genuine gap is visible: two hours with no closed klines on
        // either asset drags the percentage down, not back to a gated 0.
        let frozen_total = 2 * ((1_000_000 + 3 * 3_600) / 3_600 - 1_000_000 / 3_600);
        let mut gap_pct = None;
        for tick in 1..=(2 * 3_600 / poll) {
            let now = 1_000_000 + 3 * 3_600 + tick * poll;
            gap_pct = w.observe(now, frozen_total).or(gap_pct);
        }
        let gap_pct = gap_pct.expect("gauge must stay live through a gap");
        assert!(
            gap_pct < 100.0,
            "a 2h silent gap must lower completeness, got {gap_pct}"
        );
    }

    /// The old behaviour, pinned forever: a fixed 1800s window with a 1h
    /// interval never opens the gate. Reproduced here by forcing the
    /// unscaled window, so the scaling in `CompletenessWindow::new` is
    /// provably what fixes it.
    #[test]
    fn unscaled_window_never_goes_live_for_one_hour_interval() {
        let mut w = CompletenessWindow::new(vec![3_600], 2);
        w.window_secs = COMPLETENESS_WINDOW_MIN_SECS; // pre-fix behaviour

        let poll = 10;
        for tick in 0..=(24 * 3_600 / poll) {
            let now = 1_000_000 + tick * poll;
            let closed_total = 2 * (now / 3_600 - 1_000_000 / 3_600);
            assert_eq!(
                w.observe(now, closed_total),
                None,
                "fixed 1800s window must keep the gate closed (the bug)"
            );
        }
    }

    /// Default 1m,5m deploys keep their behaviour: baseline window, gauge
    /// live after ten minutes (2 × 5m), 100% under perfect ingestion.
    #[test]
    fn default_intervals_unaffected() {
        let mut w = CompletenessWindow::new(vec![60, 300], 3);
        assert_eq!(w.window_secs, COMPLETENESS_WINDOW_MIN_SECS);

        let poll = 10;
        let mut live_at = None;
        for tick in 0..=(3_600 / poll) {
            let now = 2_000_000 + tick * poll;
            let closed_total = 3 * ((now / 60 - 2_000_000 / 60) + (now / 300 - 2_000_000 / 300));
            if let Some(pct) = w.observe(now, closed_total) {
                live_at.get_or_insert(now - 2_000_000);
                assert_eq!(pct, 100.0);
            }
        }
        assert_eq!(live_at, Some(600), "gauge must go live at 2 × 5m");
    }
}
