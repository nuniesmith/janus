//! Prometheus Metrics Exporter for Data Factory
//!
//! Implements P0 Item 5: Prometheus Metrics Export
//!
//! This module exports all critical SLIs (Service Level Indicators) defined in
//! the spike validation documentation. These metrics are essential for:
//! - Monitoring production health
//! - SLO tracking (99.9% data completeness)
//! - Alerting on violations
//! - Performance analysis

#![allow(dead_code)]
//!
//! ## Exported Metrics
//!
//! ### Data Completeness (SLI-Q1)
//! - `data_completeness_percent` - Percentage of complete data (target: 99.9%)
//! - `gaps_detected_total` - Total number of gaps detected
//! - `gap_size_trades` - Size of detected gaps in trades
//! - `backfill_queue_size` - Number of gaps waiting for backfill
//!
//! ### Ingestion Performance (SLI-I1, SLI-I2)
//! - `ingestion_latency_ms` - Time from exchange to QuestDB (P99 < 1000ms)
//! - `trades_ingested_total` - Total trades ingested
//! - `trades_per_second` - Current ingestion rate
//!
//! ### Rate Limiter (SLI-P1, SLI-P2)
//! - `rate_limiter_requests_total` - Total rate limit requests
//! - `rate_limiter_accepted_total` - Requests that passed immediately
//! - `rate_limiter_rejected_total` - Requests rejected (rate limit hit)
//! - `rate_limiter_tokens_available` - Current token bucket level
//! - `circuit_breaker_state` - Circuit breaker state (0=closed, 1=open, 2=half-open)
//!
//! ### System Health (SLI-A1, SLI-A2)
//! - `websocket_connected` - WebSocket connection status per exchange
//! - `websocket_reconnections_total` - Total reconnection count
//! - `system_uptime_seconds` - Service uptime
//!
//! ### Storage (SLI-S1, SLI-S2)
//! - `questdb_disk_usage_percent` - Disk usage percentage (alert at 80%)
//! - `questdb_writes_total` - Total writes to QuestDB
//! - `questdb_write_errors_total` - Failed writes
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_data::metrics::prometheus_exporter::PrometheusExporter;
//!
//! // Initialize metrics
//! let exporter = PrometheusExporter::new();
//!
//! // Record metrics
//! exporter.record_trade_ingested("binance", "BTCUSD", 150.5);
//! exporter.record_gap_detected("binance", "ETHUSDT", 42);
//!
//! // Expose /metrics endpoint
//! let metrics = exporter.export().unwrap();
//! ```

use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, HistogramVec, IntCounterVec, IntGauge,
    IntGaugeVec, TextEncoder, register_counter, register_counter_vec, register_gauge,
    register_gauge_vec, register_histogram_vec, register_int_counter_vec, register_int_gauge,
    register_int_gauge_vec,
};
use std::sync::LazyLock;
use std::time::Instant;

// ============================================================================
// Data Completeness Metrics (SLI-Q1)
// ============================================================================

/// Percentage of data completeness (target: 99.9%)
pub static DATA_COMPLETENESS: LazyLock<Gauge> = LazyLock::new(|| {
    register_gauge!("data_completeness_percent", "Data completeness percentage")
        .expect("Failed to create data_completeness_percent gauge")
});

/// Total number of gaps detected
pub static GAPS_DETECTED: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "gaps_detected_total",
        "Total number of data gaps detected",
        &["exchange", "symbol"]
    )
    .expect("Failed to create gaps_detected_total counter")
});

/// Size of detected gaps in trades
pub static GAP_SIZE_TRADES: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "gap_size_trades",
        "Size of detected gaps in number of trades",
        &["exchange", "symbol"],
        vec![1.0, 10.0, 100.0, 1000.0, 10000.0]
    )
    .expect("Failed to create gap_size_trades histogram")
});

/// Current backfill queue size
pub static BACKFILL_QUEUE_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!("backfill_queue_size", "Number of gaps waiting for backfill")
        .expect("Failed to create backfill_queue_size gauge")
});

// ============================================================================
// Ingestion Performance Metrics (SLI-I1, SLI-I2)
// ============================================================================

/// Ingestion latency from exchange to QuestDB (P99 target: <1000ms)
pub static INGESTION_LATENCY: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "ingestion_latency_ms",
        "Time from exchange timestamp to QuestDB write in milliseconds",
        &["exchange", "symbol"],
        vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    )
    .expect("Failed to create ingestion_latency_ms histogram")
});

/// Total trades ingested
pub static TRADES_INGESTED: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "trades_ingested_total",
        "Total number of trades ingested",
        &["exchange", "symbol"]
    )
    .expect("Failed to create trades_ingested_total counter")
});

/// Current ingestion rate (trades per second)
pub static TRADES_PER_SECOND: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "trades_per_second",
        "Current trade ingestion rate per second",
        &["exchange", "symbol"]
    )
    .expect("Failed to create trades_per_second gauge")
});

// ============================================================================
// Rate Limiter Metrics (SLI-P1, SLI-P2)
// ============================================================================

/// Total rate limiter requests
pub static RATE_LIMITER_REQUESTS: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "rate_limiter_requests_total",
        "Total rate limiter requests",
        &["exchange", "result"]
    )
    .expect("Failed to create rate_limiter_requests_total counter")
});

/// Requests accepted immediately (had tokens)
pub static RATE_LIMITER_ACCEPTED: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "rate_limiter_accepted_total",
        "Requests that passed rate limiter immediately",
        &["exchange"]
    )
    .expect("Failed to create rate_limiter_accepted_total counter")
});

/// Requests rejected (rate limit exceeded)
pub static RATE_LIMITER_REJECTED: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "rate_limiter_rejected_total",
        "Requests rejected due to rate limit",
        &["exchange"]
    )
    .expect("Failed to create rate_limiter_rejected_total counter")
});

/// Current token bucket level
pub static RATE_LIMITER_TOKENS: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "rate_limiter_tokens_available",
        "Current number of available tokens",
        &["exchange"]
    )
    .expect("Failed to create rate_limiter_tokens_available gauge")
});

/// Circuit breaker state (0=closed, 1=open, 2=half-open)
pub static CIRCUIT_BREAKER_STATE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    register_int_gauge_vec!(
        "circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        &["exchange"]
    )
    .expect("Failed to create circuit_breaker_state gauge")
});

// ============================================================================
// System Health Metrics (SLI-A1, SLI-A2)
// ============================================================================

/// WebSocket connection status (1=connected, 0=disconnected)
pub static WEBSOCKET_CONNECTED: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    register_int_gauge_vec!(
        "websocket_connected",
        "WebSocket connection status per exchange",
        &["exchange"]
    )
    .expect("Failed to create websocket_connected gauge")
});

/// Total WebSocket reconnections
pub static WEBSOCKET_RECONNECTIONS: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "websocket_reconnections_total",
        "Total WebSocket reconnection count",
        &["exchange"]
    )
    .expect("Failed to create websocket_reconnections_total counter")
});

/// System uptime in seconds
pub static SYSTEM_UPTIME: LazyLock<Gauge> = LazyLock::new(|| {
    register_gauge!("system_uptime_seconds", "Service uptime in seconds")
        .expect("Failed to create system_uptime_seconds gauge")
});

// ============================================================================
// Storage Metrics (SLI-S1, SLI-S2)
// ============================================================================

/// QuestDB disk usage percentage (alert at 80%, stop backfill at 90%)
pub static QUESTDB_DISK_USAGE: LazyLock<Gauge> = LazyLock::new(|| {
    register_gauge!(
        "questdb_disk_usage_percent",
        "QuestDB disk usage percentage"
    )
    .expect("Failed to create questdb_disk_usage_percent gauge")
});

/// Total writes to QuestDB
pub static QUESTDB_WRITES: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "questdb_writes_total",
        "Total writes to QuestDB",
        &["table"]
    )
    .expect("Failed to create questdb_writes_total counter")
});

/// Failed QuestDB writes
pub static QUESTDB_WRITE_ERRORS: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "questdb_write_errors_total",
        "Failed QuestDB writes",
        &["table", "error_type"]
    )
    .expect("Failed to create questdb_write_errors_total counter")
});

// ============================================================================
// Backfill Metrics
// ============================================================================

/// Total backfills completed
pub static BACKFILLS_COMPLETED: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "backfills_completed_total",
        "Total backfills completed (success or failed)",
        &["exchange", "symbol", "status"]
    )
    .expect("Failed to create backfills_completed_total counter")
});

/// Backfill duration in seconds
pub static BACKFILL_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "backfill_duration_seconds",
        "Duration of backfill operations in seconds",
        &["exchange", "symbol"],
        vec![1.0, 10.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]
    )
    .expect("Failed to create backfill_duration_seconds histogram")
});

/// Currently running backfills
pub static BACKFILLS_RUNNING: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!("backfills_running", "Number of currently running backfills")
        .expect("Failed to create backfills_running gauge")
});

/// Backfill retry attempts
pub static BACKFILL_RETRIES: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "backfill_retries_total",
        "Total number of backfill retry attempts",
        &["exchange", "symbol", "retry_count"]
    )
    .expect("Failed to create backfill_retries_total counter")
});

/// Backfills that exceeded max retries
pub static BACKFILL_MAX_RETRIES_EXCEEDED: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "backfill_max_retries_exceeded_total",
        "Backfills that exceeded maximum retry attempts",
        &["exchange", "symbol"]
    )
    .expect("Failed to create backfill_max_retries_exceeded_total counter")
});

// ============================================================================
// Deduplication Metrics
// ============================================================================

/// Deduplication hits (gap already in queue)
pub static BACKFILL_DEDUP_HITS: LazyLock<Counter> = LazyLock::new(|| {
    register_counter!(
        "backfill_dedup_hits_total",
        "Number of duplicate gaps detected and skipped"
    )
    .expect("Failed to create backfill_dedup_hits_total counter")
});

/// Deduplication misses (new gap)
pub static BACKFILL_DEDUP_MISSES: LazyLock<Counter> = LazyLock::new(|| {
    register_counter!(
        "backfill_dedup_misses_total",
        "Number of new gaps submitted to queue"
    )
    .expect("Failed to create backfill_dedup_misses_total counter")
});

/// Size of deduplication set
pub static BACKFILL_DEDUP_SET_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "backfill_dedup_set_size",
        "Current size of the deduplication set"
    )
    .expect("Failed to create backfill_dedup_set_size gauge")
});

// ============================================================================
// Lock Metrics
// ============================================================================

/// Successful lock acquisitions
pub static BACKFILL_LOCK_ACQUIRED: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "backfill_lock_acquired_total",
        "Number of successful distributed lock acquisitions",
        &["exchange", "symbol"]
    )
    .expect("Failed to create backfill_lock_acquired_total counter")
});

/// Failed lock acquisitions
pub static BACKFILL_LOCK_FAILED: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "backfill_lock_failed_total",
        "Number of failed distributed lock acquisitions",
        &["exchange", "symbol"]
    )
    .expect("Failed to create backfill_lock_failed_total counter")
});

// ============================================================================
// Throttle Metrics
// ============================================================================

/// Backfill throttle rejections
pub static BACKFILL_THROTTLE_REJECTIONS: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "backfill_throttle_rejections_total",
        "Number of backfills rejected by throttle",
        &["reason"]
    )
    .expect("Failed to create backfill_throttle_rejections_total counter")
});

// ============================================================================
// Enhanced Storage Metrics
// ============================================================================

/// QuestDB write latency
pub static QUESTDB_WRITE_LATENCY: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "questdb_write_latency_seconds",
        "QuestDB write latency in seconds",
        &["table"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    )
    .expect("Failed to create questdb_write_latency_seconds histogram")
});

/// QuestDB disk usage in bytes
pub static QUESTDB_DISK_USAGE_BYTES: LazyLock<Gauge> = LazyLock::new(|| {
    register_gauge!("questdb_disk_usage_bytes", "QuestDB disk usage in bytes")
        .expect("Failed to create questdb_disk_usage_bytes gauge")
});

// ============================================================================
// Gap Detection Metrics
// ============================================================================

/// Gap detection accuracy percentage
pub static GAP_DETECTION_ACCURACY: LazyLock<Gauge> = LazyLock::new(|| {
    register_gauge!(
        "gap_detection_accuracy_percent",
        "Gap detection accuracy percentage"
    )
    .expect("Failed to create gap_detection_accuracy_percent gauge")
});

/// Number of active gaps being tracked
pub static GAP_DETECTION_ACTIVE_GAPS: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "gap_detection_active_gaps",
        "Number of active gaps currently being tracked"
    )
    .expect("Failed to create gap_detection_active_gaps gauge")
});

// ============================================================================
// Technical Indicator Metrics
// ============================================================================

/// Total number of indicator calculations performed
pub static INDICATORS_CALCULATED: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "data_factory_indicators_calculated_total",
        "Total number of indicator calculations performed",
        &["symbol", "timeframe"]
    )
    .expect("Failed to create indicators_calculated counter")
});

/// Indicator calculation duration histogram
pub static INDICATOR_CALCULATION_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "data_factory_indicator_calculation_duration_seconds",
        "Time taken to calculate indicators for a candle",
        &["symbol", "timeframe"],
        vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    )
    .expect("Failed to create indicator_calculation_duration histogram")
});

/// Number of candles processed during indicator warmup
pub static INDICATOR_WARMUP_PROGRESS: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "data_factory_indicator_warmup_candles",
        "Number of candles processed during indicator warmup"
    )
    .expect("Failed to create indicator_warmup_candles gauge")
});

/// Number of symbol/timeframe pairs being tracked for indicators
pub static INDICATOR_PAIRS_TRACKED: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "data_factory_indicator_pairs_tracked",
        "Number of symbol/timeframe pairs being tracked for indicators"
    )
    .expect("Failed to create indicator_pairs_tracked gauge")
});

// ============================================================================
// Signal Generation Metrics
// ============================================================================

/// Total number of signals generated by type
pub static SIGNALS_GENERATED: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "data_factory_signals_generated_total",
        "Total number of trading signals generated",
        &["symbol", "timeframe", "signal_type", "direction"]
    )
    .expect("Failed to create signals_generated counter")
});

/// Active signals count (gauge)
pub static ACTIVE_SIGNALS: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    register_int_gauge_vec!(
        "data_factory_active_signals",
        "Number of currently active signals",
        &["symbol", "timeframe", "signal_type"]
    )
    .expect("Failed to create active_signals gauge")
});

/// Signal strength distribution
pub static SIGNAL_STRENGTH: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "data_factory_signal_strength",
        "Distribution of signal strength values",
        &["symbol", "timeframe"],
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    )
    .expect("Failed to create signal_strength histogram")
});

// ============================================================================
// Prometheus Exporter
// ============================================================================

/// Prometheus metrics exporter
pub struct PrometheusExporter {
    start_time: Instant,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    /// Record a trade ingestion with latency
    pub fn record_trade_ingested(&self, exchange: &str, symbol: &str, latency_ms: f64) {
        TRADES_INGESTED.with_label_values(&[exchange, symbol]).inc();

        INGESTION_LATENCY
            .with_label_values(&[exchange, symbol])
            .observe(latency_ms);
    }

    /// Record a gap detection
    pub fn record_gap_detected(&self, exchange: &str, symbol: &str, gap_size: u64) {
        GAPS_DETECTED.with_label_values(&[exchange, symbol]).inc();

        GAP_SIZE_TRADES
            .with_label_values(&[exchange, symbol])
            .observe(gap_size as f64);
    }

    /// Update data completeness percentage
    pub fn update_data_completeness(&self, percentage: f64) {
        DATA_COMPLETENESS.set(percentage);
    }

    /// Update backfill queue size
    pub fn update_backfill_queue_size(&self, size: i64) {
        BACKFILL_QUEUE_SIZE.set(size);
    }

    /// Record rate limiter request
    pub fn record_rate_limit_request(&self, exchange: &str, accepted: bool) {
        RATE_LIMITER_REQUESTS
            .with_label_values(&[exchange, if accepted { "pass" } else { "wait" }])
            .inc();

        if accepted {
            RATE_LIMITER_ACCEPTED.with_label_values(&[exchange]).inc();
        } else {
            RATE_LIMITER_REJECTED.with_label_values(&[exchange]).inc();
        }
    }

    /// Update rate limiter token count
    pub fn update_rate_limiter_tokens(&self, exchange: &str, tokens: f64) {
        RATE_LIMITER_TOKENS
            .with_label_values(&[exchange])
            .set(tokens);
    }

    /// Update circuit breaker state
    pub fn update_circuit_breaker_state(&self, exchange: &str, state: i64) {
        CIRCUIT_BREAKER_STATE
            .with_label_values(&[exchange])
            .set(state);
    }

    /// Update WebSocket connection status (SLI-W1)
    pub fn update_websocket_status(&self, exchange: &str, connected: bool) {
        WEBSOCKET_CONNECTED
            .with_label_values(&[exchange])
            .set(if connected { 1 } else { 0 });
    }

    /// Record WebSocket reconnection
    pub fn record_websocket_reconnection(&self, exchange: &str) {
        WEBSOCKET_RECONNECTIONS.with_label_values(&[exchange]).inc();
    }

    /// Update QuestDB disk usage
    pub fn update_questdb_disk_usage(&self, percentage: f64) {
        QUESTDB_DISK_USAGE.set(percentage);
    }

    /// Record QuestDB write
    pub fn record_questdb_write(&self, table: &str) {
        QUESTDB_WRITES.with_label_values(&[table]).inc();
    }

    /// Record QuestDB write error
    pub fn record_questdb_write_error(&self, table: &str, error_type: &str) {
        QUESTDB_WRITE_ERRORS
            .with_label_values(&[table, error_type])
            .inc();
    }

    /// Record backfill completion
    pub fn record_backfill_completed(&self, exchange: &str, symbol: &str, duration_secs: f64) {
        BACKFILLS_COMPLETED
            .with_label_values(&[exchange, symbol, "success"])
            .inc();

        BACKFILL_DURATION
            .with_label_values(&[exchange, symbol])
            .observe(duration_secs);
    }

    /// Increment running backfills
    pub fn backfill_started(&self) {
        BACKFILLS_RUNNING.inc();
    }

    /// Decrement running backfills
    pub fn backfill_finished(&self) {
        BACKFILLS_RUNNING.dec();
    }

    /// Record a backfill retry attempt
    pub fn record_backfill_retry(&self, exchange: &str, symbol: &str, retry_count: u32) {
        BACKFILL_RETRIES
            .with_label_values(&[exchange, symbol, &retry_count.to_string()])
            .inc();
    }

    /// Record a backfill exceeding max retries
    pub fn record_backfill_max_retries_exceeded(&self, exchange: &str, symbol: &str) {
        BACKFILL_MAX_RETRIES_EXCEEDED
            .with_label_values(&[exchange, symbol])
            .inc();
    }

    /// Record a deduplication hit (gap already in queue)
    pub fn record_dedup_hit(&self) {
        BACKFILL_DEDUP_HITS.inc();
    }

    /// Record a deduplication miss (new gap)
    pub fn record_dedup_miss(&self) {
        BACKFILL_DEDUP_MISSES.inc();
    }

    /// Update deduplication set size
    pub fn update_dedup_set_size(&self, size: i64) {
        BACKFILL_DEDUP_SET_SIZE.set(size);
    }

    /// Record a successful lock acquisition
    pub fn record_lock_acquired(&self, exchange: &str, symbol: &str) {
        BACKFILL_LOCK_ACQUIRED
            .with_label_values(&[exchange, symbol])
            .inc();
    }

    /// Record a failed lock acquisition
    pub fn record_lock_failed(&self, exchange: &str, symbol: &str) {
        BACKFILL_LOCK_FAILED
            .with_label_values(&[exchange, symbol])
            .inc();
    }

    /// Record a throttle rejection
    pub fn record_throttle_rejection(&self, reason: &str) {
        BACKFILL_THROTTLE_REJECTIONS
            .with_label_values(&[reason])
            .inc();
    }

    /// Record QuestDB write latency
    pub fn record_questdb_write_latency(&self, table: &str, latency_seconds: f64) {
        QUESTDB_WRITE_LATENCY
            .with_label_values(&[table])
            .observe(latency_seconds);
    }

    pub fn update_questdb_disk_usage_bytes(&self, bytes: u64) {
        QUESTDB_DISK_USAGE_BYTES.set(bytes as f64);
    }

    pub fn record_questdb_bytes_written(&self, bytes: usize) {
        // This could be tracked as a counter if needed
        // For now, we'll just log it or track as part of write metrics
        // Could extend QUESTDB_WRITES to include a bytes dimension
        let _ = bytes; // Placeholder
    }

    pub fn record_backfill_failed(&self, exchange: &str, symbol: &str) {
        // Increment a failure counter
        // We can track this using existing metrics or add a new counter
        BACKFILL_MAX_RETRIES_EXCEEDED
            .with_label_values(&[exchange, symbol])
            .inc();
    }

    /// Update gap detection accuracy
    pub fn update_gap_detection_accuracy(&self, accuracy_percent: f64) {
        GAP_DETECTION_ACCURACY.set(accuracy_percent);
    }

    /// Update active gaps count
    pub fn update_active_gaps(&self, count: i64) {
        GAP_DETECTION_ACTIVE_GAPS.set(count);
    }

    /// Update system uptime
    pub fn update_uptime(&self) {
        SYSTEM_UPTIME.set(self.start_time.elapsed().as_secs_f64());
    }

    /// Record indicator calculation
    pub fn record_indicator_calculated(&self, symbol: &str, timeframe: &str, duration_secs: f64) {
        INDICATORS_CALCULATED
            .with_label_values(&[symbol, timeframe])
            .inc();

        INDICATOR_CALCULATION_DURATION
            .with_label_values(&[symbol, timeframe])
            .observe(duration_secs);
    }

    /// Update indicator pairs tracked
    pub fn update_indicator_pairs_tracked(&self, count: i64) {
        INDICATOR_PAIRS_TRACKED.set(count);
    }

    /// Update indicator warmup progress
    pub fn update_indicator_warmup_progress(&self, candles: i64) {
        INDICATOR_WARMUP_PROGRESS.set(candles);
    }

    /// Record signal generated
    pub fn record_signal_generated(
        &self,
        symbol: &str,
        timeframe: &str,
        signal_type: &str,
        direction: &str,
    ) {
        SIGNALS_GENERATED
            .with_label_values(&[symbol, timeframe, signal_type, direction])
            .inc();
    }

    /// Export all metrics in Prometheus text format
    pub fn export(&self) -> Result<String, prometheus::Error> {
        // Update uptime before export
        self.update_uptime();

        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap_or_default())
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HTTP Handler (for Axum/Warp/etc)
// ============================================================================

/// HTTP handler for /metrics endpoint
///
/// Returns Prometheus metrics in text format
pub async fn metrics_handler() -> Result<String, String> {
    PrometheusExporter::new()
        .export()
        .map_err(|e| format!("Failed to export metrics: {}", e))
}

// ============================================================================
// Standalone Helper Functions (for use without PrometheusExporter instance)
// ============================================================================

/// Record a signal generation (standalone function)
pub fn record_signal_generated(symbol: &str, timeframe: &str, signal_type: &str, direction: &str) {
    SIGNALS_GENERATED
        .with_label_values(&[symbol, timeframe, signal_type, direction])
        .inc();
}

/// Record indicator calculation (standalone function)
pub fn record_indicator_calculated(symbol: &str, timeframe: &str, duration_secs: f64) {
    INDICATORS_CALCULATED
        .with_label_values(&[symbol, timeframe])
        .inc();

    INDICATOR_CALCULATION_DURATION
        .with_label_values(&[symbol, timeframe])
        .observe(duration_secs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exporter_creation() {
        let exporter = PrometheusExporter::new();
        assert!(exporter.start_time.elapsed().as_secs() < 1);
    }

    #[test]
    fn test_record_trade_ingested() {
        let exporter = PrometheusExporter::new();
        exporter.record_trade_ingested("binance", "BTCUSD", 150.5);

        // Verify counter incremented
        let value = TRADES_INGESTED
            .with_label_values(&["binance", "BTCUSD"])
            .get();
        assert!(value > 0.0);
    }

    #[test]
    fn test_record_gap_detected() {
        let exporter = PrometheusExporter::new();
        exporter.record_gap_detected("bybit", "ETHUSDT", 42);

        // Verify counter incremented
        let value = GAPS_DETECTED.with_label_values(&["bybit", "ETHUSDT"]).get();
        assert!(value > 0.0);
    }

    #[test]
    fn test_update_data_completeness() {
        let exporter = PrometheusExporter::new();
        exporter.update_data_completeness(99.95);

        assert_eq!(DATA_COMPLETENESS.get(), 99.95);
    }

    #[test]
    fn test_rate_limiter_metrics() {
        let exporter = PrometheusExporter::new();
        exporter.record_rate_limit_request("binance", true);
        exporter.record_rate_limit_request("binance", false);

        let accepted = RATE_LIMITER_ACCEPTED.with_label_values(&["binance"]).get();
        let rejected = RATE_LIMITER_REJECTED.with_label_values(&["binance"]).get();

        assert!(accepted > 0.0);
        assert!(rejected > 0.0);
    }

    #[test]
    fn test_circuit_breaker_state() {
        let exporter = PrometheusExporter::new();

        // Record initial state (may be non-zero due to other tests)
        let initial_state = CIRCUIT_BREAKER_STATE.with_label_values(&["binance"]).get();

        exporter.update_circuit_breaker_state("binance", 1); // Open

        let state = CIRCUIT_BREAKER_STATE.with_label_values(&["binance"]).get();
        assert_eq!(state, 1);

        // Reset to initial state for other tests
        exporter.update_circuit_breaker_state("binance", initial_state);
    }

    #[test]
    fn test_websocket_metrics() {
        let exporter = PrometheusExporter::new();
        exporter.update_websocket_status("binance", true);
        exporter.record_websocket_reconnection("binance");

        let connected = WEBSOCKET_CONNECTED.with_label_values(&["binance"]).get();
        assert_eq!(connected, 1);

        let reconnections = WEBSOCKET_RECONNECTIONS
            .with_label_values(&["binance"])
            .get();
        assert!(reconnections > 0.0);
    }

    #[test]
    fn test_export_metrics() {
        let exporter = PrometheusExporter::new();
        exporter.record_trade_ingested("binance", "BTCUSD", 100.0);

        let output = exporter.export().expect("Failed to export metrics");
        assert!(output.contains("trades_ingested_total"));
        assert!(output.contains("ingestion_latency_ms"));
        assert!(output.contains("system_uptime_seconds"));
    }

    #[test]
    fn test_backfill_metrics() {
        let exporter = PrometheusExporter::new();

        // Record initial state (may be non-zero due to other tests)
        let initial_running = BACKFILLS_RUNNING.get();
        let initial_completed = BACKFILLS_COMPLETED
            .with_label_values(&["binance", "BTCUSD", "success"])
            .get();

        exporter.backfill_started();
        // Should have increased by 1
        assert_eq!(BACKFILLS_RUNNING.get(), initial_running + 1);

        exporter.record_backfill_completed("binance", "BTCUSD", 45.5);
        exporter.backfill_finished();

        // Should be back to initial state
        let final_running = BACKFILLS_RUNNING.get();
        assert_eq!(final_running, initial_running);

        // Completed should have increased
        let final_completed = BACKFILLS_COMPLETED
            .with_label_values(&["binance", "BTCUSD", "success"])
            .get();
        assert!(final_completed > initial_completed);
    }
}
