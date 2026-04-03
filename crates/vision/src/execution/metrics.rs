//! # Execution Metrics Module
//!
//! Provides Prometheus metrics for monitoring trade execution performance.
//! This module tracks execution quality, slippage, costs, and venue performance.

use prometheus::{
    Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, Opts,
    Registry,
};
use std::sync::{Arc, LazyLock};

use super::analytics::{ExecutionRecord, Side};

/// Global execution metrics registry
pub static EXECUTION_METRICS: LazyLock<Arc<ExecutionMetrics>> =
    LazyLock::new(|| Arc::new(ExecutionMetrics::new()));

/// Centralized metrics registry for trade execution
pub struct ExecutionMetrics {
    /// Prometheus registry
    pub registry: Registry,

    // === Execution Volume Metrics ===
    /// Total number of executions
    pub executions_total: IntCounterVec,

    /// Total quantity executed
    pub quantity_executed_total: GaugeVec,

    /// Total execution cost in dollars
    pub execution_cost_total: GaugeVec,

    // === Slippage Metrics ===
    /// Execution slippage in basis points
    pub slippage_bps: HistogramVec,

    /// Average slippage by venue
    pub average_slippage_bps: GaugeVec,

    /// VWAP slippage in basis points
    pub vwap_slippage_bps: Gauge,

    // === Cost Metrics ===
    /// Execution cost in basis points
    pub cost_bps: HistogramVec,

    /// Implementation shortfall percentage
    pub implementation_shortfall_pct: Gauge,

    // === Quality Metrics ===
    /// Execution quality score (0-100)
    pub quality_score: Gauge,

    /// Fill rate (percentage of orders filled)
    pub fill_rate_pct: GaugeVec,

    // === Venue Performance Metrics ===
    /// Executions per venue
    pub venue_executions_total: IntCounterVec,

    /// Average execution price by venue
    pub venue_avg_price: GaugeVec,

    /// Venue latency in microseconds
    pub venue_latency_us: HistogramVec,

    // === Order Type Metrics ===
    /// TWAP order executions
    pub twap_executions_total: IntCounter,

    /// VWAP order executions
    pub vwap_executions_total: IntCounter,

    /// Market order executions
    pub market_executions_total: IntCounter,

    /// Limit order executions
    pub limit_executions_total: IntCounter,

    // === Timing Metrics ===
    /// Time to fill orders (seconds)
    pub time_to_fill_seconds: Histogram,

    /// Execution latency (microseconds)
    pub execution_latency_us: Histogram,

    // === Error Metrics ===
    /// Failed executions
    pub executions_failed_total: IntCounterVec,

    /// Rejected orders
    pub orders_rejected_total: IntCounterVec,

    /// Partial fills
    pub partial_fills_total: IntCounter,
}

impl ExecutionMetrics {
    /// Create a new execution metrics registry
    pub fn new() -> Self {
        let registry = Registry::new();

        // Execution volume metrics
        let executions_total = IntCounterVec::new(
            Opts::new("vision_execution_total", "Total number of trade executions")
                .namespace("vision")
                .subsystem("execution"),
            &["side", "venue"],
        )
        .unwrap();

        let quantity_executed_total = GaugeVec::new(
            Opts::new("vision_execution_quantity_total", "Total quantity executed")
                .namespace("vision")
                .subsystem("execution"),
            &["side", "venue"],
        )
        .unwrap();

        let execution_cost_total = GaugeVec::new(
            Opts::new(
                "vision_execution_cost_total",
                "Total execution cost in dollars",
            )
            .namespace("vision")
            .subsystem("execution"),
            &["side", "venue"],
        )
        .unwrap();

        // Slippage metrics
        let slippage_bps = HistogramVec::new(
            HistogramOpts::new(
                "vision_execution_slippage_bps",
                "Execution slippage in basis points",
            )
            .namespace("vision")
            .subsystem("execution")
            .buckets(vec![
                -50.0, -25.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0,
            ]),
            &["side", "venue"],
        )
        .unwrap();

        let average_slippage_bps = GaugeVec::new(
            Opts::new(
                "vision_execution_avg_slippage_bps",
                "Average execution slippage by venue",
            )
            .namespace("vision")
            .subsystem("execution"),
            &["venue"],
        )
        .unwrap();

        let vwap_slippage_bps = Gauge::with_opts(
            Opts::new(
                "vision_execution_vwap_slippage_bps",
                "Volume-weighted average slippage",
            )
            .namespace("vision")
            .subsystem("execution"),
        )
        .unwrap();

        // Cost metrics
        let cost_bps = HistogramVec::new(
            HistogramOpts::new(
                "vision_execution_cost_bps",
                "Execution cost in basis points",
            )
            .namespace("vision")
            .subsystem("execution")
            .buckets(vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
            &["side", "venue"],
        )
        .unwrap();

        let implementation_shortfall_pct = Gauge::with_opts(
            Opts::new(
                "vision_execution_implementation_shortfall_pct",
                "Implementation shortfall as percentage",
            )
            .namespace("vision")
            .subsystem("execution"),
        )
        .unwrap();

        // Quality metrics
        let quality_score = Gauge::with_opts(
            Opts::new(
                "vision_execution_quality_score",
                "Overall execution quality score (0-100)",
            )
            .namespace("vision")
            .subsystem("execution"),
        )
        .unwrap();

        let fill_rate_pct = GaugeVec::new(
            Opts::new(
                "vision_execution_fill_rate_pct",
                "Order fill rate percentage",
            )
            .namespace("vision")
            .subsystem("execution"),
            &["order_type"],
        )
        .unwrap();

        // Venue performance metrics
        let venue_executions_total = IntCounterVec::new(
            Opts::new("vision_execution_venue_total", "Total executions per venue")
                .namespace("vision")
                .subsystem("execution"),
            &["venue"],
        )
        .unwrap();

        let venue_avg_price = GaugeVec::new(
            Opts::new(
                "vision_execution_venue_avg_price",
                "Average execution price by venue",
            )
            .namespace("vision")
            .subsystem("execution"),
            &["venue"],
        )
        .unwrap();

        let venue_latency_us = HistogramVec::new(
            HistogramOpts::new(
                "vision_execution_venue_latency_us",
                "Venue execution latency in microseconds",
            )
            .namespace("vision")
            .subsystem("execution")
            .buckets(vec![
                100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 50000.0, 100000.0,
            ]),
            &["venue"],
        )
        .unwrap();

        // Order type metrics
        let twap_executions_total = IntCounter::with_opts(
            Opts::new("vision_execution_twap_total", "Total TWAP order executions")
                .namespace("vision")
                .subsystem("execution"),
        )
        .unwrap();

        let vwap_executions_total = IntCounter::with_opts(
            Opts::new("vision_execution_vwap_total", "Total VWAP order executions")
                .namespace("vision")
                .subsystem("execution"),
        )
        .unwrap();

        let market_executions_total = IntCounter::with_opts(
            Opts::new(
                "vision_execution_market_total",
                "Total market order executions",
            )
            .namespace("vision")
            .subsystem("execution"),
        )
        .unwrap();

        let limit_executions_total = IntCounter::with_opts(
            Opts::new(
                "vision_execution_limit_total",
                "Total limit order executions",
            )
            .namespace("vision")
            .subsystem("execution"),
        )
        .unwrap();

        // Timing metrics
        let time_to_fill_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "vision_execution_time_to_fill_seconds",
                "Time to fill orders in seconds",
            )
            .namespace("vision")
            .subsystem("execution")
            .buckets(vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]),
        )
        .unwrap();

        let execution_latency_us = Histogram::with_opts(
            HistogramOpts::new(
                "vision_execution_latency_us",
                "Execution processing latency in microseconds",
            )
            .namespace("vision")
            .subsystem("execution")
            .buckets(vec![
                10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0,
            ]),
        )
        .unwrap();

        // Error metrics
        let executions_failed_total = IntCounterVec::new(
            Opts::new("vision_execution_failed_total", "Total failed executions")
                .namespace("vision")
                .subsystem("execution"),
            &["reason"],
        )
        .unwrap();

        let orders_rejected_total = IntCounterVec::new(
            Opts::new("vision_execution_rejected_total", "Total rejected orders")
                .namespace("vision")
                .subsystem("execution"),
            &["reason"],
        )
        .unwrap();

        let partial_fills_total = IntCounter::with_opts(
            Opts::new(
                "vision_execution_partial_fills_total",
                "Total partial fills",
            )
            .namespace("vision")
            .subsystem("execution"),
        )
        .unwrap();

        // Register all metrics
        registry
            .register(Box::new(executions_total.clone()))
            .unwrap();
        registry
            .register(Box::new(quantity_executed_total.clone()))
            .unwrap();
        registry
            .register(Box::new(execution_cost_total.clone()))
            .unwrap();
        registry.register(Box::new(slippage_bps.clone())).unwrap();
        registry
            .register(Box::new(average_slippage_bps.clone()))
            .unwrap();
        registry
            .register(Box::new(vwap_slippage_bps.clone()))
            .unwrap();
        registry.register(Box::new(cost_bps.clone())).unwrap();
        registry
            .register(Box::new(implementation_shortfall_pct.clone()))
            .unwrap();
        registry.register(Box::new(quality_score.clone())).unwrap();
        registry.register(Box::new(fill_rate_pct.clone())).unwrap();
        registry
            .register(Box::new(venue_executions_total.clone()))
            .unwrap();
        registry
            .register(Box::new(venue_avg_price.clone()))
            .unwrap();
        registry
            .register(Box::new(venue_latency_us.clone()))
            .unwrap();
        registry
            .register(Box::new(twap_executions_total.clone()))
            .unwrap();
        registry
            .register(Box::new(vwap_executions_total.clone()))
            .unwrap();
        registry
            .register(Box::new(market_executions_total.clone()))
            .unwrap();
        registry
            .register(Box::new(limit_executions_total.clone()))
            .unwrap();
        registry
            .register(Box::new(time_to_fill_seconds.clone()))
            .unwrap();
        registry
            .register(Box::new(execution_latency_us.clone()))
            .unwrap();
        registry
            .register(Box::new(executions_failed_total.clone()))
            .unwrap();
        registry
            .register(Box::new(orders_rejected_total.clone()))
            .unwrap();
        registry
            .register(Box::new(partial_fills_total.clone()))
            .unwrap();

        Self {
            registry,
            executions_total,
            quantity_executed_total,
            execution_cost_total,
            slippage_bps,
            average_slippage_bps,
            vwap_slippage_bps,
            cost_bps,
            implementation_shortfall_pct,
            quality_score,
            fill_rate_pct,
            venue_executions_total,
            venue_avg_price,
            venue_latency_us,
            twap_executions_total,
            vwap_executions_total,
            market_executions_total,
            limit_executions_total,
            time_to_fill_seconds,
            execution_latency_us,
            executions_failed_total,
            orders_rejected_total,
            partial_fills_total,
        }
    }

    /// Record an execution and update all relevant metrics
    pub fn record_execution(&self, record: &ExecutionRecord) {
        let side_str = match record.side {
            Side::Buy => "buy",
            Side::Sell => "sell",
        };

        // Update execution counts
        self.executions_total
            .with_label_values(&[side_str, &record.venue])
            .inc();

        self.venue_executions_total
            .with_label_values(&[&record.venue])
            .inc();

        // Update quantity
        self.quantity_executed_total
            .with_label_values(&[side_str, &record.venue])
            .set(record.quantity);

        // Update costs
        let cost = record.cost();
        self.execution_cost_total
            .with_label_values(&[side_str, &record.venue])
            .add(cost);

        // Record slippage
        let slippage = record.slippage_bps();
        self.slippage_bps
            .with_label_values(&[side_str, &record.venue])
            .observe(slippage);

        // Record cost in bps
        let cost_bps = record.cost_bps();
        self.cost_bps
            .with_label_values(&[side_str, &record.venue])
            .observe(cost_bps);
    }

    /// Update aggregate metrics from analytics
    pub fn update_from_analytics(&self, analytics: &super::analytics::ExecutionAnalytics) {
        // Update per-venue slippage
        for (venue, venue_stat) in analytics.venue_statistics() {
            self.average_slippage_bps
                .with_label_values(&[&venue])
                .set(venue_stat.average_slippage_bps);

            self.venue_avg_price
                .with_label_values(&[&venue])
                .set(venue_stat.total_cost / venue_stat.total_quantity.max(0.0001));
        }

        // Update VWAP slippage
        let vwap_slip = analytics.vwap_slippage_bps();
        if vwap_slip.is_finite() {
            self.vwap_slippage_bps.set(vwap_slip);
        }

        // Update implementation shortfall
        let is_pct = analytics.implementation_shortfall_pct();
        if is_pct.is_finite() {
            self.implementation_shortfall_pct.set(is_pct);
        }

        // Update quality score
        self.quality_score.set(analytics.quality_score());
    }

    /// Record a failed execution
    pub fn record_failure(&self, reason: &str) {
        self.executions_failed_total
            .with_label_values(&[reason])
            .inc();
    }

    /// Record a rejected order
    pub fn record_rejection(&self, reason: &str) {
        self.orders_rejected_total
            .with_label_values(&[reason])
            .inc();
    }

    /// Record a partial fill
    pub fn record_partial_fill(&self) {
        self.partial_fills_total.inc();
    }

    /// Record TWAP execution
    pub fn record_twap(&self) {
        self.twap_executions_total.inc();
    }

    /// Record VWAP execution
    pub fn record_vwap(&self) {
        self.vwap_executions_total.inc();
    }

    /// Record market order execution
    pub fn record_market(&self) {
        self.market_executions_total.inc();
    }

    /// Record limit order execution
    pub fn record_limit(&self) {
        self.limit_executions_total.inc();
    }

    /// Record execution latency
    pub fn record_latency(&self, latency_us: f64) {
        self.execution_latency_us.observe(latency_us);
    }

    /// Record time to fill
    pub fn record_time_to_fill(&self, seconds: f64) {
        self.time_to_fill_seconds.observe(seconds);
    }

    /// Record venue latency
    pub fn record_venue_latency(&self, venue: &str, latency_us: f64) {
        self.venue_latency_us
            .with_label_values(&[venue])
            .observe(latency_us);
    }

    /// Gather metrics for export
    pub fn gather(&self) -> Vec<prometheus::proto::MetricFamily> {
        self.registry.gather()
    }

    /// Encode metrics as Prometheus text format
    pub fn encode_text(&self) -> Result<String, Box<dyn std::error::Error>> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper struct for convenient metrics access
pub struct ExecutionMetricsCollector;

impl ExecutionMetricsCollector {
    /// Get the global metrics instance
    pub fn metrics() -> Arc<ExecutionMetrics> {
        EXECUTION_METRICS.clone()
    }

    /// Record an execution
    pub fn record(record: &ExecutionRecord) {
        EXECUTION_METRICS.record_execution(record);
    }

    /// Update from analytics
    pub fn update(analytics: &super::analytics::ExecutionAnalytics) {
        EXECUTION_METRICS.update_from_analytics(analytics);
    }

    /// Get metrics as text
    pub fn export() -> Result<String, Box<dyn std::error::Error>> {
        EXECUTION_METRICS.encode_text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn create_test_execution(
        order_id: &str,
        quantity: f64,
        execution_price: f64,
        benchmark_price: f64,
        side: Side,
        venue: &str,
    ) -> ExecutionRecord {
        ExecutionRecord {
            order_id: order_id.to_string(),
            quantity,
            execution_price,
            benchmark_price,
            timestamp: Instant::now(),
            side,
            venue: venue.to_string(),
        }
    }

    #[test]
    fn test_metrics_creation() {
        let metrics = ExecutionMetrics::new();
        assert!(!metrics.registry.gather().is_empty());
    }

    #[test]
    fn test_record_execution() {
        let metrics = ExecutionMetrics::new();
        let execution =
            create_test_execution("test-order-1", 100.0, 50.05, 50.0, Side::Buy, "exchange-a");

        metrics.record_execution(&execution);

        // Verify metrics were recorded (we can't easily assert on counter values
        // in tests, but we can verify no panics occurred)
    }

    #[test]
    fn test_record_failure() {
        let metrics = ExecutionMetrics::new();
        metrics.record_failure("timeout");
        // No panic = success
    }

    #[test]
    fn test_record_order_types() {
        let metrics = ExecutionMetrics::new();
        metrics.record_twap();
        metrics.record_vwap();
        metrics.record_market();
        metrics.record_limit();
        // No panic = success
    }

    #[test]
    fn test_encode_text() {
        let metrics = ExecutionMetrics::new();
        let text = metrics.encode_text().unwrap();
        assert!(text.contains("vision_execution"));
    }

    #[test]
    fn test_metrics_collector() {
        let metrics = ExecutionMetricsCollector::metrics();
        assert!(!metrics.registry.gather().is_empty());
    }

    #[test]
    fn test_export() {
        let result = ExecutionMetricsCollector::export();
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("vision_execution"));
    }
}
