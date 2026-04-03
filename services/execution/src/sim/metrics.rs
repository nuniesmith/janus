//! Prometheus Metrics for Simulation Components
//!
//! Exports RecorderStats and LiveFeedBridgeStats to Prometheus for monitoring.
//!
//! # Metrics Exported
//!
//! ## Data Recorder Metrics
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | sim_recorder_events_recorded_total | Counter | Total events recorded |
//! | sim_recorder_events_dropped_total | Counter | Events dropped due to buffer overflow |
//! | sim_recorder_write_errors_total | Counter | Write errors to QuestDB |
//! | sim_recorder_reconnections_total | Counter | Reconnection attempts |
//! | sim_recorder_flush_count_total | Counter | Successful flush operations |
//! | sim_recorder_bytes_written_total | Counter | Total bytes written |
//! | sim_recorder_buffer_depth | Gauge | Current buffer depth |
//! | sim_recorder_buffer_utilization_pct | Gauge | Buffer utilization percentage |
//! | sim_recorder_channel_depth | Gauge | Current channel depth |
//! | sim_recorder_channel_utilization_pct | Gauge | Channel utilization percentage |
//! | sim_recorder_connected | Gauge | Connection status (1=connected, 0=disconnected) |
//! | sim_recorder_events_per_second | Gauge | Events per second rate |
//! | sim_recorder_drop_rate_pct | Gauge | Drop rate percentage |
//! | sim_recorder_error_rate_pct | Gauge | Error rate percentage |
//! | sim_recorder_healthy | Gauge | Health status (1=healthy, 0=unhealthy) |
//! | sim_recorder_uptime_seconds | Gauge | Uptime in seconds |
//!
//! ## Live Feed Bridge Metrics
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | sim_bridge_events_received_total | Counter | Total events received from providers |
//! | sim_bridge_events_published_total | Counter | Events published to data feed |
//! | sim_bridge_events_dropped_total | Counter | Events dropped |
//! | sim_bridge_ticks_converted_total | Counter | Ticks converted |
//! | sim_bridge_trades_converted_total | Counter | Trades converted |
//! | sim_bridge_orderbooks_converted_total | Counter | Order books converted |
//! | sim_bridge_candles_converted_total | Counter | Candles converted |
//! | sim_bridge_events_per_second | Gauge | Events per second rate |
//! | sim_bridge_success_rate_pct | Gauge | Conversion success rate |
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_execution::sim::metrics::{SimMetricsExporter, global_sim_metrics};
//!
//! // Create exporter
//! let exporter = SimMetricsExporter::new();
//!
//! // Update from stats
//! exporter.update_recorder_stats(&recorder.stats());
//! exporter.update_bridge_stats(&bridge.stats());
//!
//! // Get Prometheus format output
//! let metrics = exporter.to_prometheus();
//!
//! // Or use global instance
//! global_sim_metrics().update_recorder_stats(&stats);
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::data_recorder::RecorderStats;
use super::live_feed_bridge::LiveFeedBridgeStats;

// ============================================================================
// Recorder Metrics
// ============================================================================

/// Prometheus metrics for the DataRecorder
#[derive(Debug)]
pub struct RecorderMetrics {
    /// Instance name for labeling
    instance: String,

    // Counters (monotonically increasing)
    events_recorded: AtomicU64,
    ticks_recorded: AtomicU64,
    trades_recorded: AtomicU64,
    orderbooks_recorded: AtomicU64,
    candles_recorded: AtomicU64,
    events_dropped: AtomicU64,
    write_errors: AtomicU64,
    reconnections: AtomicU64,
    bytes_written: AtomicU64,
    flush_count: AtomicU64,

    // Gauges (can go up and down)
    buffer_depth: AtomicU64,
    buffer_capacity: AtomicU64,
    channel_depth: AtomicU64,
    channel_capacity: AtomicU64,
    connected: AtomicU64,

    // Computed metrics stored atomically (multiplied by 100 for precision)
    events_per_second_x100: AtomicU64,
    buffer_utilization_pct_x100: AtomicU64,
    channel_utilization_pct_x100: AtomicU64,
    drop_rate_pct_x100: AtomicU64,
    error_rate_pct_x100: AtomicU64,
    uptime_seconds_x100: AtomicU64,
    healthy: AtomicU64,

    // Fallback metrics
    fallback_active: AtomicU64,
    fallback_events: AtomicU64,
    fallback_bytes: AtomicU64,
    fallback_errors: AtomicU64,

    /// Last update timestamp
    last_update: RwLock<Option<Instant>>,
}

impl RecorderMetrics {
    /// Create new recorder metrics
    pub fn new(instance: &str) -> Self {
        Self {
            instance: instance.to_string(),
            events_recorded: AtomicU64::new(0),
            ticks_recorded: AtomicU64::new(0),
            trades_recorded: AtomicU64::new(0),
            orderbooks_recorded: AtomicU64::new(0),
            candles_recorded: AtomicU64::new(0),
            events_dropped: AtomicU64::new(0),
            write_errors: AtomicU64::new(0),
            reconnections: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
            buffer_depth: AtomicU64::new(0),
            buffer_capacity: AtomicU64::new(0),
            channel_depth: AtomicU64::new(0),
            channel_capacity: AtomicU64::new(0),
            connected: AtomicU64::new(0),
            events_per_second_x100: AtomicU64::new(0),
            buffer_utilization_pct_x100: AtomicU64::new(0),
            channel_utilization_pct_x100: AtomicU64::new(0),
            drop_rate_pct_x100: AtomicU64::new(0),
            error_rate_pct_x100: AtomicU64::new(0),
            uptime_seconds_x100: AtomicU64::new(0),
            healthy: AtomicU64::new(0),
            fallback_active: AtomicU64::new(0),
            fallback_events: AtomicU64::new(0),
            fallback_bytes: AtomicU64::new(0),
            fallback_errors: AtomicU64::new(0),
            last_update: RwLock::new(None),
        }
    }

    /// Update from RecorderStats
    pub fn update(&self, stats: &RecorderStats) {
        // Update counters
        self.events_recorded
            .store(stats.events_recorded, Ordering::Relaxed);
        self.ticks_recorded
            .store(stats.ticks_recorded, Ordering::Relaxed);
        self.trades_recorded
            .store(stats.trades_recorded, Ordering::Relaxed);
        self.orderbooks_recorded
            .store(stats.orderbooks_recorded, Ordering::Relaxed);
        self.candles_recorded
            .store(stats.candles_recorded, Ordering::Relaxed);
        self.events_dropped
            .store(stats.events_dropped, Ordering::Relaxed);
        self.write_errors
            .store(stats.write_errors, Ordering::Relaxed);
        self.reconnections
            .store(stats.reconnections, Ordering::Relaxed);
        self.bytes_written
            .store(stats.bytes_written, Ordering::Relaxed);
        self.flush_count.store(stats.flush_count, Ordering::Relaxed);

        // Update gauges
        self.buffer_depth
            .store(stats.buffer_depth as u64, Ordering::Relaxed);
        self.buffer_capacity
            .store(stats.buffer_capacity as u64, Ordering::Relaxed);
        self.channel_depth
            .store(stats.channel_depth as u64, Ordering::Relaxed);
        self.channel_capacity
            .store(stats.channel_capacity as u64, Ordering::Relaxed);
        self.connected
            .store(if stats.connected { 1 } else { 0 }, Ordering::Relaxed);

        // Update computed metrics (store as x100 for precision)
        self.events_per_second_x100.store(
            (stats.events_per_second() * 100.0) as u64,
            Ordering::Relaxed,
        );
        self.buffer_utilization_pct_x100.store(
            (stats.buffer_utilization_pct() * 100.0) as u64,
            Ordering::Relaxed,
        );
        self.channel_utilization_pct_x100.store(
            (stats.channel_utilization_pct() * 100.0) as u64,
            Ordering::Relaxed,
        );
        self.drop_rate_pct_x100
            .store((stats.drop_rate_pct() * 100.0) as u64, Ordering::Relaxed);
        self.error_rate_pct_x100
            .store((stats.error_rate_pct() * 100.0) as u64, Ordering::Relaxed);
        self.uptime_seconds_x100
            .store((stats.uptime_seconds() * 100.0) as u64, Ordering::Relaxed);
        self.healthy
            .store(if stats.is_healthy() { 1 } else { 0 }, Ordering::Relaxed);

        // Update fallback metrics
        self.fallback_active
            .store(if stats.fallback_active { 1 } else { 0 }, Ordering::Relaxed);
        self.fallback_events
            .store(stats.fallback_events, Ordering::Relaxed);
        self.fallback_bytes
            .store(stats.fallback_bytes, Ordering::Relaxed);
        self.fallback_errors
            .store(stats.fallback_errors, Ordering::Relaxed);

        *self.last_update.write() = Some(Instant::now());
    }

    /// Generate Prometheus format output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::with_capacity(4096);
        let instance = &self.instance;

        // Counters
        output.push_str(&format!(
            "# HELP sim_recorder_events_recorded_total Total events recorded\n\
             # TYPE sim_recorder_events_recorded_total counter\n\
             sim_recorder_events_recorded_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.events_recorded.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_ticks_recorded_total Total ticks recorded\n\
             # TYPE sim_recorder_ticks_recorded_total counter\n\
             sim_recorder_ticks_recorded_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.ticks_recorded.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_trades_recorded_total Total trades recorded\n\
             # TYPE sim_recorder_trades_recorded_total counter\n\
             sim_recorder_trades_recorded_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.trades_recorded.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_orderbooks_recorded_total Total order book snapshots recorded\n\
             # TYPE sim_recorder_orderbooks_recorded_total counter\n\
             sim_recorder_orderbooks_recorded_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.orderbooks_recorded.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_candles_recorded_total Total candles recorded\n\
             # TYPE sim_recorder_candles_recorded_total counter\n\
             sim_recorder_candles_recorded_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.candles_recorded.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_events_dropped_total Events dropped due to buffer overflow\n\
             # TYPE sim_recorder_events_dropped_total counter\n\
             sim_recorder_events_dropped_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.events_dropped.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_write_errors_total Write errors to QuestDB\n\
             # TYPE sim_recorder_write_errors_total counter\n\
             sim_recorder_write_errors_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.write_errors.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_reconnections_total Reconnection attempts\n\
             # TYPE sim_recorder_reconnections_total counter\n\
             sim_recorder_reconnections_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.reconnections.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_bytes_written_total Total bytes written to QuestDB\n\
             # TYPE sim_recorder_bytes_written_total counter\n\
             sim_recorder_bytes_written_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.bytes_written.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_flush_count_total Successful flush operations\n\
             # TYPE sim_recorder_flush_count_total counter\n\
             sim_recorder_flush_count_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.flush_count.load(Ordering::Relaxed)
        ));

        // Gauges
        output.push_str(&format!(
            "# HELP sim_recorder_buffer_depth Current buffer depth\n\
             # TYPE sim_recorder_buffer_depth gauge\n\
             sim_recorder_buffer_depth{{instance=\"{}\"}} {}\n\n",
            instance,
            self.buffer_depth.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_buffer_capacity Buffer capacity\n\
             # TYPE sim_recorder_buffer_capacity gauge\n\
             sim_recorder_buffer_capacity{{instance=\"{}\"}} {}\n\n",
            instance,
            self.buffer_capacity.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_buffer_utilization_pct Buffer utilization percentage\n\
             # TYPE sim_recorder_buffer_utilization_pct gauge\n\
             sim_recorder_buffer_utilization_pct{{instance=\"{}\"}} {:.2}\n\n",
            instance,
            self.buffer_utilization_pct_x100.load(Ordering::Relaxed) as f64 / 100.0
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_channel_depth Current channel depth\n\
             # TYPE sim_recorder_channel_depth gauge\n\
             sim_recorder_channel_depth{{instance=\"{}\"}} {}\n\n",
            instance,
            self.channel_depth.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_channel_capacity Channel capacity\n\
             # TYPE sim_recorder_channel_capacity gauge\n\
             sim_recorder_channel_capacity{{instance=\"{}\"}} {}\n\n",
            instance,
            self.channel_capacity.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_channel_utilization_pct Channel utilization percentage\n\
             # TYPE sim_recorder_channel_utilization_pct gauge\n\
             sim_recorder_channel_utilization_pct{{instance=\"{}\"}} {:.2}\n\n",
            instance,
            self.channel_utilization_pct_x100.load(Ordering::Relaxed) as f64 / 100.0
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_connected Connection status (1=connected, 0=disconnected)\n\
             # TYPE sim_recorder_connected gauge\n\
             sim_recorder_connected{{instance=\"{}\"}} {}\n\n",
            instance,
            self.connected.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_events_per_second Events per second rate\n\
             # TYPE sim_recorder_events_per_second gauge\n\
             sim_recorder_events_per_second{{instance=\"{}\"}} {:.2}\n\n",
            instance,
            self.events_per_second_x100.load(Ordering::Relaxed) as f64 / 100.0
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_drop_rate_pct Drop rate percentage\n\
             # TYPE sim_recorder_drop_rate_pct gauge\n\
             sim_recorder_drop_rate_pct{{instance=\"{}\"}} {:.4}\n\n",
            instance,
            self.drop_rate_pct_x100.load(Ordering::Relaxed) as f64 / 100.0
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_error_rate_pct Error rate percentage\n\
             # TYPE sim_recorder_error_rate_pct gauge\n\
             sim_recorder_error_rate_pct{{instance=\"{}\"}} {:.4}\n\n",
            instance,
            self.error_rate_pct_x100.load(Ordering::Relaxed) as f64 / 100.0
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_healthy Health status (1=healthy, 0=unhealthy)\n\
             # TYPE sim_recorder_healthy gauge\n\
             sim_recorder_healthy{{instance=\"{}\"}} {}\n\n",
            instance,
            self.healthy.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_uptime_seconds Uptime in seconds\n\
             # TYPE sim_recorder_uptime_seconds gauge\n\
             sim_recorder_uptime_seconds{{instance=\"{}\"}} {:.2}\n\n",
            instance,
            self.uptime_seconds_x100.load(Ordering::Relaxed) as f64 / 100.0
        ));

        // Fallback metrics
        output.push_str(&format!(
            "# HELP sim_recorder_fallback_active Fallback storage active (1=active, 0=inactive)\n\
             # TYPE sim_recorder_fallback_active gauge\n\
             sim_recorder_fallback_active{{instance=\"{}\"}} {}\n\n",
            instance,
            self.fallback_active.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_fallback_events_total Events written to fallback storage\n\
             # TYPE sim_recorder_fallback_events_total counter\n\
             sim_recorder_fallback_events_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.fallback_events.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_fallback_bytes_total Bytes written to fallback storage\n\
             # TYPE sim_recorder_fallback_bytes_total counter\n\
             sim_recorder_fallback_bytes_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.fallback_bytes.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_recorder_fallback_errors_total Fallback write errors\n\
             # TYPE sim_recorder_fallback_errors_total counter\n\
             sim_recorder_fallback_errors_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.fallback_errors.load(Ordering::Relaxed)
        ));

        output
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.events_recorded.store(0, Ordering::Relaxed);
        self.ticks_recorded.store(0, Ordering::Relaxed);
        self.trades_recorded.store(0, Ordering::Relaxed);
        self.orderbooks_recorded.store(0, Ordering::Relaxed);
        self.candles_recorded.store(0, Ordering::Relaxed);
        self.events_dropped.store(0, Ordering::Relaxed);
        self.write_errors.store(0, Ordering::Relaxed);
        self.reconnections.store(0, Ordering::Relaxed);
        self.bytes_written.store(0, Ordering::Relaxed);
        self.flush_count.store(0, Ordering::Relaxed);
        self.buffer_depth.store(0, Ordering::Relaxed);
        self.fallback_active.store(0, Ordering::Relaxed);
        self.fallback_events.store(0, Ordering::Relaxed);
        self.fallback_bytes.store(0, Ordering::Relaxed);
        self.fallback_errors.store(0, Ordering::Relaxed);
        self.buffer_capacity.store(0, Ordering::Relaxed);
        self.channel_depth.store(0, Ordering::Relaxed);
        self.channel_capacity.store(0, Ordering::Relaxed);
        self.connected.store(0, Ordering::Relaxed);
        self.events_per_second_x100.store(0, Ordering::Relaxed);
        self.buffer_utilization_pct_x100.store(0, Ordering::Relaxed);
        self.channel_utilization_pct_x100
            .store(0, Ordering::Relaxed);
        self.drop_rate_pct_x100.store(0, Ordering::Relaxed);
        self.error_rate_pct_x100.store(0, Ordering::Relaxed);
        self.uptime_seconds_x100.store(0, Ordering::Relaxed);
        self.healthy.store(0, Ordering::Relaxed);
        *self.last_update.write() = None;
    }
}

impl Default for RecorderMetrics {
    fn default() -> Self {
        Self::new("default")
    }
}

// ============================================================================
// Live Feed Bridge Metrics
// ============================================================================

/// Prometheus metrics for the LiveFeedBridge
#[derive(Debug)]
pub struct BridgeMetrics {
    /// Instance name for labeling
    instance: String,

    // Counters
    events_received: AtomicU64,
    events_published: AtomicU64,
    events_dropped: AtomicU64,
    ticks_converted: AtomicU64,
    trades_converted: AtomicU64,
    orderbooks_converted: AtomicU64,
    candles_converted: AtomicU64,
    connection_events: AtomicU64,

    // Events by exchange
    events_by_exchange: RwLock<HashMap<String, u64>>,

    // Computed metrics
    events_per_second_x100: AtomicU64,
    success_rate_pct_x100: AtomicU64,

    /// Last update timestamp
    last_update: RwLock<Option<Instant>>,
}

impl BridgeMetrics {
    /// Create new bridge metrics
    pub fn new(instance: &str) -> Self {
        Self {
            instance: instance.to_string(),
            events_received: AtomicU64::new(0),
            events_published: AtomicU64::new(0),
            events_dropped: AtomicU64::new(0),
            ticks_converted: AtomicU64::new(0),
            trades_converted: AtomicU64::new(0),
            orderbooks_converted: AtomicU64::new(0),
            candles_converted: AtomicU64::new(0),
            connection_events: AtomicU64::new(0),
            events_by_exchange: RwLock::new(HashMap::new()),
            events_per_second_x100: AtomicU64::new(0),
            success_rate_pct_x100: AtomicU64::new(10000), // 100.00%
            last_update: RwLock::new(None),
        }
    }

    /// Update from LiveFeedBridgeStats
    pub fn update(&self, stats: &LiveFeedBridgeStats) {
        // Update counters
        self.events_received
            .store(stats.events_received, Ordering::Relaxed);
        self.events_published
            .store(stats.events_published, Ordering::Relaxed);
        self.events_dropped
            .store(stats.events_dropped, Ordering::Relaxed);
        self.ticks_converted
            .store(stats.ticks_converted, Ordering::Relaxed);
        self.trades_converted
            .store(stats.trades_converted, Ordering::Relaxed);
        self.orderbooks_converted
            .store(stats.orderbooks_converted, Ordering::Relaxed);
        self.candles_converted
            .store(stats.candles_converted, Ordering::Relaxed);
        self.connection_events
            .store(stats.connection_events, Ordering::Relaxed);

        // Update events by exchange
        {
            let mut by_exchange = self.events_by_exchange.write();
            by_exchange.clear();
            for (exchange, count) in &stats.events_by_exchange {
                by_exchange.insert(exchange.clone(), *count);
            }
        }

        // Update computed metrics
        self.events_per_second_x100.store(
            (stats.events_per_second() * 100.0) as u64,
            Ordering::Relaxed,
        );
        self.success_rate_pct_x100
            .store((stats.success_rate() * 100.0) as u64, Ordering::Relaxed);

        *self.last_update.write() = Some(Instant::now());
    }

    /// Generate Prometheus format output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::with_capacity(2048);
        let instance = &self.instance;

        // Counters
        output.push_str(&format!(
            "# HELP sim_bridge_events_received_total Total events received from providers\n\
             # TYPE sim_bridge_events_received_total counter\n\
             sim_bridge_events_received_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.events_received.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_bridge_events_published_total Events published to data feed\n\
             # TYPE sim_bridge_events_published_total counter\n\
             sim_bridge_events_published_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.events_published.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_bridge_events_dropped_total Events dropped\n\
             # TYPE sim_bridge_events_dropped_total counter\n\
             sim_bridge_events_dropped_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.events_dropped.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_bridge_ticks_converted_total Ticks converted\n\
             # TYPE sim_bridge_ticks_converted_total counter\n\
             sim_bridge_ticks_converted_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.ticks_converted.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_bridge_trades_converted_total Trades converted\n\
             # TYPE sim_bridge_trades_converted_total counter\n\
             sim_bridge_trades_converted_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.trades_converted.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_bridge_orderbooks_converted_total Order books converted\n\
             # TYPE sim_bridge_orderbooks_converted_total counter\n\
             sim_bridge_orderbooks_converted_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.orderbooks_converted.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_bridge_candles_converted_total Candles converted\n\
             # TYPE sim_bridge_candles_converted_total counter\n\
             sim_bridge_candles_converted_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.candles_converted.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP sim_bridge_connection_events_total Connection events\n\
             # TYPE sim_bridge_connection_events_total counter\n\
             sim_bridge_connection_events_total{{instance=\"{}\"}} {}\n\n",
            instance,
            self.connection_events.load(Ordering::Relaxed)
        ));

        // Events by exchange
        {
            let by_exchange = self.events_by_exchange.read();
            if !by_exchange.is_empty() {
                output.push_str(
                    "# HELP sim_bridge_events_by_exchange Events received by exchange\n\
                     # TYPE sim_bridge_events_by_exchange counter\n",
                );
                for (exchange, count) in by_exchange.iter() {
                    output.push_str(&format!(
                        "sim_bridge_events_by_exchange{{instance=\"{}\",exchange=\"{}\"}} {}\n",
                        instance, exchange, count
                    ));
                }
                output.push('\n');
            }
        }

        // Gauges
        output.push_str(&format!(
            "# HELP sim_bridge_events_per_second Events per second rate\n\
             # TYPE sim_bridge_events_per_second gauge\n\
             sim_bridge_events_per_second{{instance=\"{}\"}} {:.2}\n\n",
            instance,
            self.events_per_second_x100.load(Ordering::Relaxed) as f64 / 100.0
        ));

        output.push_str(&format!(
            "# HELP sim_bridge_success_rate_pct Conversion success rate percentage\n\
             # TYPE sim_bridge_success_rate_pct gauge\n\
             sim_bridge_success_rate_pct{{instance=\"{}\"}} {:.2}\n\n",
            instance,
            self.success_rate_pct_x100.load(Ordering::Relaxed) as f64 / 100.0
        ));

        output
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.events_received.store(0, Ordering::Relaxed);
        self.events_published.store(0, Ordering::Relaxed);
        self.events_dropped.store(0, Ordering::Relaxed);
        self.ticks_converted.store(0, Ordering::Relaxed);
        self.trades_converted.store(0, Ordering::Relaxed);
        self.orderbooks_converted.store(0, Ordering::Relaxed);
        self.candles_converted.store(0, Ordering::Relaxed);
        self.connection_events.store(0, Ordering::Relaxed);
        self.events_by_exchange.write().clear();
        self.events_per_second_x100.store(0, Ordering::Relaxed);
        self.success_rate_pct_x100.store(10000, Ordering::Relaxed);
        *self.last_update.write() = None;
    }
}

impl Default for BridgeMetrics {
    fn default() -> Self {
        Self::new("default")
    }
}

// ============================================================================
// Combined Metrics Exporter
// ============================================================================

/// Combined metrics exporter for all simulation components
#[derive(Debug)]
pub struct SimMetricsExporter {
    /// Recorder metrics by instance
    recorder_metrics: RwLock<HashMap<String, Arc<RecorderMetrics>>>,
    /// Bridge metrics by instance
    bridge_metrics: RwLock<HashMap<String, Arc<BridgeMetrics>>>,
}

impl SimMetricsExporter {
    /// Create a new metrics exporter
    pub fn new() -> Self {
        Self {
            recorder_metrics: RwLock::new(HashMap::new()),
            bridge_metrics: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create recorder metrics for an instance
    pub fn recorder(&self, instance: &str) -> Arc<RecorderMetrics> {
        let mut metrics = self.recorder_metrics.write();
        if let Some(m) = metrics.get(instance) {
            return m.clone();
        }
        let m = Arc::new(RecorderMetrics::new(instance));
        metrics.insert(instance.to_string(), m.clone());
        m
    }

    /// Get or create bridge metrics for an instance
    pub fn bridge(&self, instance: &str) -> Arc<BridgeMetrics> {
        let mut metrics = self.bridge_metrics.write();
        if let Some(m) = metrics.get(instance) {
            return m.clone();
        }
        let m = Arc::new(BridgeMetrics::new(instance));
        metrics.insert(instance.to_string(), m.clone());
        m
    }

    /// Update recorder stats for default instance
    pub fn update_recorder_stats(&self, stats: &RecorderStats) {
        self.recorder("default").update(stats);
    }

    /// Update bridge stats for default instance
    pub fn update_bridge_stats(&self, stats: &LiveFeedBridgeStats) {
        self.bridge("default").update(stats);
    }

    /// Update recorder stats for named instance
    pub fn update_recorder_stats_named(&self, instance: &str, stats: &RecorderStats) {
        self.recorder(instance).update(stats);
    }

    /// Update bridge stats for named instance
    pub fn update_bridge_stats_named(&self, instance: &str, stats: &LiveFeedBridgeStats) {
        self.bridge(instance).update(stats);
    }

    /// Generate combined Prometheus output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::with_capacity(8192);

        // Header
        output.push_str("# FKS Simulation Metrics\n\n");

        // Recorder metrics
        for (_, metrics) in self.recorder_metrics.read().iter() {
            output.push_str(&metrics.to_prometheus());
        }

        // Bridge metrics
        for (_, metrics) in self.bridge_metrics.read().iter() {
            output.push_str(&metrics.to_prometheus());
        }

        output
    }

    /// Reset all metrics
    pub fn reset_all(&self) {
        for (_, metrics) in self.recorder_metrics.read().iter() {
            metrics.reset();
        }
        for (_, metrics) in self.bridge_metrics.read().iter() {
            metrics.reset();
        }
    }

    /// Get list of registered recorder instances
    pub fn recorder_instances(&self) -> Vec<String> {
        self.recorder_metrics.read().keys().cloned().collect()
    }

    /// Get list of registered bridge instances
    pub fn bridge_instances(&self) -> Vec<String> {
        self.bridge_metrics.read().keys().cloned().collect()
    }
}

impl Default for SimMetricsExporter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Global Instance
// ============================================================================

/// Global simulation metrics exporter
static GLOBAL_SIM_METRICS: LazyLock<SimMetricsExporter> = LazyLock::new(SimMetricsExporter::new);

/// Get the global simulation metrics exporter
pub fn global_sim_metrics() -> &'static SimMetricsExporter {
    &GLOBAL_SIM_METRICS
}

/// Update recorder stats on global exporter
pub fn update_recorder_stats(stats: &RecorderStats) {
    GLOBAL_SIM_METRICS.update_recorder_stats(stats);
}

/// Update bridge stats on global exporter
pub fn update_bridge_stats(stats: &LiveFeedBridgeStats) {
    GLOBAL_SIM_METRICS.update_bridge_stats(stats);
}

/// Get combined Prometheus metrics from global exporter
pub fn sim_prometheus_metrics() -> String {
    GLOBAL_SIM_METRICS.to_prometheus()
}

// ============================================================================
// HTTP Handler (Axum compatible)
// ============================================================================

/// Axum handler for Prometheus metrics endpoint
///
/// # Example
///
/// ```rust,ignore
/// use axum::{Router, routing::get};
/// use janus_execution::sim::metrics::metrics_handler;
///
/// let app = Router::new()
///     .route("/metrics", get(metrics_handler));
/// ```
pub async fn metrics_handler() -> String {
    sim_prometheus_metrics()
}

// ============================================================================
// Periodic Update Task
// ============================================================================

use super::data_recorder::DataRecorder;
use super::live_feed_bridge::LiveFeedBridge;

/// Start a background task that periodically updates metrics
///
/// Returns a handle that can be used to stop the task
pub fn start_metrics_collector(
    recorder: Option<Arc<DataRecorder>>,
    bridge: Option<Arc<LiveFeedBridge>>,
    interval_ms: u64,
) -> tokio::task::JoinHandle<()> {
    let exporter = global_sim_metrics();

    tokio::spawn(async move {
        let interval = std::time::Duration::from_millis(interval_ms);

        loop {
            // Update recorder metrics
            if let Some(ref recorder) = recorder {
                exporter.update_recorder_stats(&recorder.stats());
            }

            // Update bridge metrics
            if let Some(ref bridge) = bridge {
                exporter.update_bridge_stats(&bridge.stats());
            }

            tokio::time::sleep(interval).await;
        }
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_recorder_metrics_creation() {
        let metrics = RecorderMetrics::new("test");
        assert_eq!(metrics.instance, "test");
        assert_eq!(metrics.events_recorded.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_recorder_metrics_update() {
        let metrics = RecorderMetrics::new("test");

        let mut stats = RecorderStats::default();
        stats.events_recorded = 1000;
        stats.events_dropped = 10;
        stats.write_errors = 5;
        stats.connected = true;
        stats.buffer_depth = 100;
        stats.buffer_capacity = 1000;
        stats.start_time = Some(Utc::now());

        metrics.update(&stats);

        assert_eq!(metrics.events_recorded.load(Ordering::Relaxed), 1000);
        assert_eq!(metrics.events_dropped.load(Ordering::Relaxed), 10);
        assert_eq!(metrics.write_errors.load(Ordering::Relaxed), 5);
        assert_eq!(metrics.connected.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.buffer_depth.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_recorder_metrics_prometheus() {
        let metrics = RecorderMetrics::new("test");

        let mut stats = RecorderStats::default();
        stats.events_recorded = 500;
        stats.connected = true;
        metrics.update(&stats);

        let output = metrics.to_prometheus();

        assert!(output.contains("sim_recorder_events_recorded_total"));
        assert!(output.contains("instance=\"test\""));
        assert!(output.contains("500"));
    }

    #[test]
    fn test_bridge_metrics_creation() {
        let metrics = BridgeMetrics::new("test");
        assert_eq!(metrics.instance, "test");
        assert_eq!(metrics.events_received.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_bridge_metrics_update() {
        let metrics = BridgeMetrics::new("test");

        let mut stats = LiveFeedBridgeStats::default();
        stats.events_received = 5000;
        stats.events_published = 4900;
        stats.events_dropped = 100;
        stats.ticks_converted = 3000;
        stats.events_by_exchange.insert("kraken".to_string(), 2000);
        stats.events_by_exchange.insert("binance".to_string(), 3000);

        metrics.update(&stats);

        assert_eq!(metrics.events_received.load(Ordering::Relaxed), 5000);
        assert_eq!(metrics.events_published.load(Ordering::Relaxed), 4900);
        assert_eq!(metrics.events_dropped.load(Ordering::Relaxed), 100);

        let by_exchange = metrics.events_by_exchange.read();
        assert_eq!(by_exchange.get("kraken"), Some(&2000));
        assert_eq!(by_exchange.get("binance"), Some(&3000));
    }

    #[test]
    fn test_bridge_metrics_prometheus() {
        let metrics = BridgeMetrics::new("test");

        let mut stats = LiveFeedBridgeStats::default();
        stats.events_received = 1000;
        stats.events_published = 980;
        stats.events_by_exchange.insert("kraken".to_string(), 500);
        metrics.update(&stats);

        let output = metrics.to_prometheus();

        assert!(output.contains("sim_bridge_events_received_total"));
        assert!(output.contains("sim_bridge_events_by_exchange"));
        assert!(output.contains("exchange=\"kraken\""));
    }

    #[test]
    fn test_exporter_creation() {
        let exporter = SimMetricsExporter::new();
        assert!(exporter.recorder_instances().is_empty());
        assert!(exporter.bridge_instances().is_empty());
    }

    #[test]
    fn test_exporter_get_or_create() {
        let exporter = SimMetricsExporter::new();

        let r1 = exporter.recorder("instance1");
        let r2 = exporter.recorder("instance1");
        let r3 = exporter.recorder("instance2");

        // Same instance should return same Arc
        assert!(Arc::ptr_eq(&r1, &r2));
        // Different instance should return different Arc
        assert!(!Arc::ptr_eq(&r1, &r3));

        assert_eq!(exporter.recorder_instances().len(), 2);
    }

    #[test]
    fn test_exporter_combined_prometheus() {
        let exporter = SimMetricsExporter::new();

        // Update some metrics
        let mut recorder_stats = RecorderStats::default();
        recorder_stats.events_recorded = 100;
        exporter.update_recorder_stats(&recorder_stats);

        let mut bridge_stats = LiveFeedBridgeStats::default();
        bridge_stats.events_received = 200;
        exporter.update_bridge_stats(&bridge_stats);

        let output = exporter.to_prometheus();

        assert!(output.contains("sim_recorder_events_recorded_total"));
        assert!(output.contains("sim_bridge_events_received_total"));
    }

    #[test]
    fn test_exporter_reset() {
        let exporter = SimMetricsExporter::new();

        let mut stats = RecorderStats::default();
        stats.events_recorded = 100;
        exporter.update_recorder_stats(&stats);

        exporter.reset_all();

        let recorder = exporter.recorder("default");
        assert_eq!(recorder.events_recorded.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_global_instance() {
        let _exporter = global_sim_metrics();

        let mut stats = RecorderStats::default();
        stats.events_recorded = 42;
        update_recorder_stats(&stats);

        let output = sim_prometheus_metrics();
        assert!(output.contains("42"));
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = RecorderMetrics::new("test");

        let mut stats = RecorderStats::default();
        stats.events_recorded = 1000;
        stats.connected = true;
        metrics.update(&stats);

        assert_eq!(metrics.events_recorded.load(Ordering::Relaxed), 1000);
        assert_eq!(metrics.connected.load(Ordering::Relaxed), 1);

        metrics.reset();

        assert_eq!(metrics.events_recorded.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.connected.load(Ordering::Relaxed), 0);
    }
}
