//! Latency Histogram Metrics for Prometheus
//!
//! This module provides Prometheus-compatible histogram metrics for tracking
//! latency distributions across the execution subsystem.
//!
//! # Metrics Exported
//!
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | *_latency_bucket | Histogram | Latency distribution buckets |
//! | *_latency_sum | Counter | Sum of all observed latencies |
//! | *_latency_count | Counter | Count of observations |
//!
//! # Usage
//!
//! ```ignore
//! use execution::execution::histogram::LatencyHistogram;
//!
//! // Create histogram with default latency buckets (ms)
//! let histogram = LatencyHistogram::new_latency_ms("order_execution");
//!
//! // Record observations
//! histogram.observe(15.5);  // 15.5ms
//! histogram.observe(42.0);  // 42ms
//!
//! // Get Prometheus output
//! let output = histogram.to_prometheus();
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// Histogram Buckets
// ============================================================================

/// Default buckets for latency measurements in milliseconds
/// Optimized for trading system latencies (sub-ms to multi-second)
pub const DEFAULT_LATENCY_BUCKETS_MS: &[f64] = &[
    0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
];

/// High-resolution buckets for WebSocket message latencies
pub const HIGH_RES_LATENCY_BUCKETS_MS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
];

/// Buckets for order execution latencies (longer timeframes)
pub const ORDER_EXECUTION_BUCKETS_MS: &[f64] = &[
    10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0,
];

/// Buckets for API call latencies
pub const API_LATENCY_BUCKETS_MS: &[f64] = &[
    5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0,
];

// ============================================================================
// Latency Histogram
// ============================================================================

/// A Prometheus-compatible histogram for tracking latency distributions
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Name of the metric (without _bucket/_sum/_count suffix)
    name: String,

    /// Help text for the metric
    help: String,

    /// Labels for this histogram instance
    labels: HashMap<String, String>,

    /// Bucket boundaries (upper bounds, exclusive of +Inf)
    buckets: Vec<f64>,

    /// Counts per bucket (cumulative)
    /// bucket_counts[i] = count of observations <= buckets[i]
    bucket_counts: Vec<AtomicU64>,

    /// Total count of observations (same as +Inf bucket)
    count: AtomicU64,

    /// Sum of all observed values (stored as value * 1000 for precision)
    sum_x1000: AtomicU64,

    /// Creation time for uptime tracking
    created_at: Instant,
}

impl LatencyHistogram {
    /// Create a new histogram with custom buckets
    pub fn new(name: &str, help: &str, buckets: &[f64]) -> Self {
        let mut sorted_buckets = buckets.to_vec();
        sorted_buckets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let bucket_counts = sorted_buckets.iter().map(|_| AtomicU64::new(0)).collect();

        Self {
            name: name.to_string(),
            help: help.to_string(),
            labels: HashMap::new(),
            buckets: sorted_buckets,
            bucket_counts,
            count: AtomicU64::new(0),
            sum_x1000: AtomicU64::new(0),
            created_at: Instant::now(),
        }
    }

    /// Create a histogram with default latency buckets (milliseconds)
    pub fn new_latency_ms(name: &str) -> Self {
        Self::new(
            name,
            &format!("{} latency in milliseconds", name),
            DEFAULT_LATENCY_BUCKETS_MS,
        )
    }

    /// Create a histogram with high-resolution latency buckets
    pub fn new_high_res_latency_ms(name: &str) -> Self {
        Self::new(
            name,
            &format!("{} latency in milliseconds (high resolution)", name),
            HIGH_RES_LATENCY_BUCKETS_MS,
        )
    }

    /// Create a histogram for order execution latencies
    pub fn new_order_execution(name: &str) -> Self {
        Self::new(
            name,
            &format!("{} order execution latency in milliseconds", name),
            ORDER_EXECUTION_BUCKETS_MS,
        )
    }

    /// Create a histogram for API call latencies
    pub fn new_api_latency(name: &str) -> Self {
        Self::new(
            name,
            &format!("{} API call latency in milliseconds", name),
            API_LATENCY_BUCKETS_MS,
        )
    }

    /// Add a label to this histogram
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Observe a value
    pub fn observe(&self, value: f64) {
        // Increment count
        self.count.fetch_add(1, Ordering::Relaxed);

        // Add to sum (multiply by 1000 for precision)
        let value_x1000 = (value * 1000.0) as u64;
        self.sum_x1000.fetch_add(value_x1000, Ordering::Relaxed);

        // Increment appropriate bucket counters (cumulative)
        for (i, bucket_bound) in self.buckets.iter().enumerate() {
            if value <= *bucket_bound {
                self.bucket_counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Observe a duration from an Instant
    pub fn observe_duration(&self, start: Instant) {
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.observe(duration_ms);
    }

    /// Get total count of observations
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get sum of all observations
    pub fn sum(&self) -> f64 {
        self.sum_x1000.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Get mean of all observations
    pub fn mean(&self) -> Option<f64> {
        let count = self.count();
        if count == 0 {
            return None;
        }
        Some(self.sum() / count as f64)
    }

    /// Get approximate percentile (linear interpolation between buckets)
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if !(0.0..=100.0).contains(&p) {
            return None;
        }

        let total = self.count();
        if total == 0 {
            return None;
        }

        let target = (total as f64 * p / 100.0).ceil() as u64;

        // Find the bucket that contains the target count
        let mut prev_count = 0u64;
        let mut prev_bound = 0.0f64;

        for (i, bucket_bound) in self.buckets.iter().enumerate() {
            let bucket_count = self.bucket_counts[i].load(Ordering::Relaxed);
            if bucket_count >= target {
                // Linear interpolation within the bucket
                let bucket_range = bucket_bound - prev_bound;
                let count_in_bucket = bucket_count - prev_count;
                if count_in_bucket == 0 {
                    return Some(*bucket_bound);
                }
                let position_in_bucket = (target - prev_count) as f64 / count_in_bucket as f64;
                return Some(prev_bound + bucket_range * position_in_bucket);
            }
            prev_count = bucket_count;
            prev_bound = *bucket_bound;
        }

        // Target is beyond all buckets, return the last bucket bound
        self.buckets.last().copied()
    }

    /// Get P50 (median)
    pub fn p50(&self) -> Option<f64> {
        self.percentile(50.0)
    }

    /// Get P90
    pub fn p90(&self) -> Option<f64> {
        self.percentile(90.0)
    }

    /// Get P95
    pub fn p95(&self) -> Option<f64> {
        self.percentile(95.0)
    }

    /// Get P99
    pub fn p99(&self) -> Option<f64> {
        self.percentile(99.0)
    }

    /// Format labels for Prometheus output
    fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            String::new()
        } else {
            let label_str: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            label_str.join(",")
        }
    }

    /// Format labels with an additional le (less than or equal) label for buckets
    fn format_labels_with_le(&self, le: &str) -> String {
        let base_labels = self.format_labels();
        if base_labels.is_empty() {
            format!("le=\"{}\"", le)
        } else {
            format!("{},le=\"{}\"", base_labels, le)
        }
    }

    /// Generate Prometheus-compatible histogram output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // HELP and TYPE
        output.push_str(&format!("# HELP {}_latency_ms {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {}_latency_ms histogram\n", self.name));

        // Output counts for each bucket
        for (i, bucket_bound) in self.buckets.iter().enumerate() {
            // Get the count for this bucket (already cumulative in our implementation)
            let bucket_count = self.bucket_counts[i].load(Ordering::Relaxed);
            let le = format!("{}", bucket_bound);
            let labels = self.format_labels_with_le(&le);
            output.push_str(&format!(
                "{}_latency_ms_bucket{{{}}} {}\n",
                self.name, labels, bucket_count
            ));
        }

        // +Inf bucket
        let total_count = self.count();
        let inf_labels = self.format_labels_with_le("+Inf");
        output.push_str(&format!(
            "{}_latency_ms_bucket{{{}}} {}\n",
            self.name, inf_labels, total_count
        ));

        // Sum
        let sum = self.sum();
        let base_labels = self.format_labels();
        if base_labels.is_empty() {
            output.push_str(&format!("{}_latency_ms_sum {:.3}\n", self.name, sum));
        } else {
            output.push_str(&format!(
                "{}_latency_ms_sum{{{}}} {:.3}\n",
                self.name, base_labels, sum
            ));
        }

        // Count
        if base_labels.is_empty() {
            output.push_str(&format!("{}_latency_ms_count {}\n", self.name, total_count));
        } else {
            output.push_str(&format!(
                "{}_latency_ms_count{{{}}} {}\n",
                self.name, base_labels, total_count
            ));
        }

        output
    }

    /// Reset the histogram
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.sum_x1000.store(0, Ordering::Relaxed);
        for bucket in &self.bucket_counts {
            bucket.store(0, Ordering::Relaxed);
        }
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }
}

// ============================================================================
// Labeled Histogram Registry
// ============================================================================

/// A registry of histograms with dynamic labels
#[derive(Debug)]
pub struct LabeledHistogramRegistry {
    /// Name prefix for all histograms
    name: String,

    /// Help text
    help: String,

    /// Bucket configuration
    buckets: Vec<f64>,

    /// Histograms by label key (e.g., "operation=place_order")
    histograms: RwLock<HashMap<String, Arc<LatencyHistogram>>>,
}

impl LabeledHistogramRegistry {
    /// Create a new labeled histogram registry
    pub fn new(name: &str, help: &str, buckets: &[f64]) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            buckets: buckets.to_vec(),
            histograms: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default latency buckets
    pub fn new_latency(name: &str, help: &str) -> Self {
        Self::new(name, help, DEFAULT_LATENCY_BUCKETS_MS)
    }

    /// Create with high-resolution buckets
    pub fn new_high_res(name: &str, help: &str) -> Self {
        Self::new(name, help, HIGH_RES_LATENCY_BUCKETS_MS)
    }

    /// Create with order execution buckets
    pub fn new_order_execution(name: &str, help: &str) -> Self {
        Self::new(name, help, ORDER_EXECUTION_BUCKETS_MS)
    }

    /// Get or create a histogram for a specific label value
    pub fn with_label(&self, label_name: &str, label_value: &str) -> Arc<LatencyHistogram> {
        let key = format!("{}={}", label_name, label_value);

        // Fast path: check if already exists
        {
            let histograms = self.histograms.read();
            if let Some(h) = histograms.get(&key) {
                return h.clone();
            }
        }

        // Slow path: create new histogram
        let mut histograms = self.histograms.write();
        // Double-check after acquiring write lock
        if let Some(h) = histograms.get(&key) {
            return h.clone();
        }

        let histogram = Arc::new(
            LatencyHistogram::new(&self.name, &self.help, &self.buckets)
                .with_label(label_name, label_value),
        );
        histograms.insert(key, histogram.clone());
        histogram
    }

    /// Observe a value for a specific label
    pub fn observe(&self, label_name: &str, label_value: &str, value: f64) {
        self.with_label(label_name, label_value).observe(value);
    }

    /// Generate Prometheus output for all histograms
    pub fn to_prometheus(&self) -> String {
        let histograms = self.histograms.read();
        if histograms.is_empty() {
            return String::new();
        }

        let mut output = String::new();

        // HELP and TYPE (only once)
        output.push_str(&format!("# HELP {}_latency_ms {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {}_latency_ms histogram\n", self.name));

        // Output each histogram (skip the HELP/TYPE lines from individual histograms)
        for histogram in histograms.values() {
            let histogram_output = histogram.to_prometheus();
            // Skip the first two lines (HELP and TYPE) since we already output them
            for line in histogram_output.lines().skip(2) {
                output.push_str(line);
                output.push('\n');
            }
        }

        output
    }

    /// Reset all histograms
    pub fn reset(&self) {
        let histograms = self.histograms.read();
        for histogram in histograms.values() {
            histogram.reset();
        }
    }
}

// ============================================================================
// Execution Latency Histograms
// ============================================================================

/// Collection of latency histograms for the execution subsystem
#[derive(Debug)]
pub struct ExecutionLatencyHistograms {
    /// Order placement latency by exchange
    pub order_placement: LabeledHistogramRegistry,

    /// Order cancellation latency by exchange
    pub order_cancellation: LabeledHistogramRegistry,

    /// Order fill latency (time from submission to fill)
    pub order_fill: LabeledHistogramRegistry,

    /// Best execution analysis latency by symbol
    pub best_execution_analysis: LabeledHistogramRegistry,

    /// WebSocket message latency by exchange
    pub websocket_message: LabeledHistogramRegistry,

    /// REST API call latency by operation
    pub api_call: LabeledHistogramRegistry,

    /// Signal processing latency
    pub signal_processing: LatencyHistogram,

    /// End-to-end trade latency (signal to fill)
    pub trade_e2e: LatencyHistogram,
}

impl ExecutionLatencyHistograms {
    /// Create new execution latency histograms
    pub fn new() -> Self {
        Self {
            order_placement: LabeledHistogramRegistry::new_order_execution(
                "execution_order_placement",
                "Order placement latency by exchange",
            ),
            order_cancellation: LabeledHistogramRegistry::new_latency(
                "execution_order_cancellation",
                "Order cancellation latency by exchange",
            ),
            order_fill: LabeledHistogramRegistry::new_order_execution(
                "execution_order_fill",
                "Time from order submission to fill by exchange",
            ),
            best_execution_analysis: LabeledHistogramRegistry::new_latency(
                "execution_best_execution_analysis",
                "Best execution analysis latency by symbol",
            ),
            websocket_message: LabeledHistogramRegistry::new_high_res(
                "exchange_ws_message",
                "WebSocket message processing latency by exchange",
            ),
            api_call: LabeledHistogramRegistry::new_latency(
                "execution_api_call",
                "REST API call latency by operation",
            ),
            signal_processing: LatencyHistogram::new_latency_ms("execution_signal_processing"),
            trade_e2e: LatencyHistogram::new_order_execution("execution_trade_e2e"),
        }
    }

    /// Record order placement latency
    pub fn record_order_placement(&self, exchange: &str, latency_ms: f64) {
        self.order_placement
            .observe("exchange", exchange, latency_ms);
    }

    /// Record order cancellation latency
    pub fn record_order_cancellation(&self, exchange: &str, latency_ms: f64) {
        self.order_cancellation
            .observe("exchange", exchange, latency_ms);
    }

    /// Record order fill latency
    pub fn record_order_fill(&self, exchange: &str, latency_ms: f64) {
        self.order_fill.observe("exchange", exchange, latency_ms);
    }

    /// Record best execution analysis latency
    pub fn record_best_execution_analysis(&self, symbol: &str, latency_ms: f64) {
        self.best_execution_analysis
            .observe("symbol", symbol, latency_ms);
    }

    /// Record WebSocket message latency
    pub fn record_websocket_message(&self, exchange: &str, latency_ms: f64) {
        self.websocket_message
            .observe("exchange", exchange, latency_ms);
    }

    /// Record API call latency
    pub fn record_api_call(&self, operation: &str, latency_ms: f64) {
        self.api_call.observe("operation", operation, latency_ms);
    }

    /// Record signal processing latency
    pub fn record_signal_processing(&self, latency_ms: f64) {
        self.signal_processing.observe(latency_ms);
    }

    /// Record end-to-end trade latency
    pub fn record_trade_e2e(&self, latency_ms: f64) {
        self.trade_e2e.observe(latency_ms);
    }

    /// Generate Prometheus output
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str(&self.order_placement.to_prometheus());
        output.push_str(&self.order_cancellation.to_prometheus());
        output.push_str(&self.order_fill.to_prometheus());
        output.push_str(&self.best_execution_analysis.to_prometheus());
        output.push_str(&self.websocket_message.to_prometheus());
        output.push_str(&self.api_call.to_prometheus());
        output.push_str(&self.signal_processing.to_prometheus());
        output.push_str(&self.trade_e2e.to_prometheus());

        output
    }

    /// Reset all histograms
    pub fn reset(&self) {
        self.order_placement.reset();
        self.order_cancellation.reset();
        self.order_fill.reset();
        self.best_execution_analysis.reset();
        self.websocket_message.reset();
        self.api_call.reset();
        self.signal_processing.reset();
        self.trade_e2e.reset();
    }
}

impl Default for ExecutionLatencyHistograms {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Global Histogram Registry
// ============================================================================

use std::sync::LazyLock;

static GLOBAL_LATENCY_HISTOGRAMS: LazyLock<ExecutionLatencyHistograms> =
    LazyLock::new(ExecutionLatencyHistograms::new);

/// Get the global latency histograms
pub fn global_latency_histograms() -> &'static ExecutionLatencyHistograms {
    &GLOBAL_LATENCY_HISTOGRAMS
}

/// Get Prometheus output for all latency histograms
pub fn latency_prometheus_metrics() -> String {
    GLOBAL_LATENCY_HISTOGRAMS.to_prometheus()
}

// ============================================================================
// Timer Guard for automatic latency recording
// ============================================================================

/// A guard that records latency when dropped
pub struct LatencyTimer {
    histogram: Arc<LatencyHistogram>,
    start: Instant,
}

impl LatencyTimer {
    /// Create a new timer
    pub fn new(histogram: Arc<LatencyHistogram>) -> Self {
        Self {
            histogram,
            start: Instant::now(),
        }
    }

    /// Manually stop the timer and record the latency
    pub fn stop(self) -> f64 {
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.histogram.observe(duration_ms);
        duration_ms
    }
}

impl Drop for LatencyTimer {
    fn drop(&mut self) {
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.histogram.observe(duration_ms);
    }
}

/// Create a timer for a labeled histogram
pub fn start_timer(
    registry: &LabeledHistogramRegistry,
    label_name: &str,
    label_value: &str,
) -> LatencyTimer {
    LatencyTimer::new(registry.with_label(label_name, label_value))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let h = LatencyHistogram::new_latency_ms("test");
        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_histogram_observe() {
        let h = LatencyHistogram::new_latency_ms("test");
        h.observe(10.0);
        h.observe(20.0);
        h.observe(30.0);

        assert_eq!(h.count(), 3);
        assert!((h.sum() - 60.0).abs() < 0.001);
        assert!((h.mean().unwrap() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_histogram_percentiles() {
        let h = LatencyHistogram::new("test", "test histogram", &[10.0, 20.0, 50.0, 100.0, 200.0]);

        // Observe values
        for _ in 0..50 {
            h.observe(5.0); // 50 values <= 10
        }
        for _ in 0..30 {
            h.observe(15.0); // 30 values in (10, 20]
        }
        for _ in 0..15 {
            h.observe(35.0); // 15 values in (20, 50]
        }
        for _ in 0..5 {
            h.observe(75.0); // 5 values in (50, 100]
        }

        assert_eq!(h.count(), 100);

        // P50 should be around 10 (50th percentile, 50 values <= 10)
        let p50 = h.p50().unwrap();
        assert!(p50 <= 10.0, "P50 should be <= 10, got {}", p50);

        // P90 should be between 20 and 50
        let p90 = h.p90().unwrap();
        assert!(
            p90 > 10.0 && p90 <= 50.0,
            "P90 should be in (10, 50], got {}",
            p90
        );

        // P99 should be higher
        let p99 = h.p99().unwrap();
        assert!(p99 > 50.0, "P99 should be > 50, got {}", p99);
    }

    #[test]
    fn test_histogram_prometheus_output() {
        let h = LatencyHistogram::new("test", "Test histogram", &[10.0, 50.0, 100.0]);
        h.observe(5.0);
        h.observe(25.0);
        h.observe(75.0);

        let output = h.to_prometheus();

        assert!(output.contains("# HELP test_latency_ms"));
        assert!(output.contains("# TYPE test_latency_ms histogram"));
        assert!(output.contains("test_latency_ms_bucket{le=\"10\"}"));
        assert!(output.contains("test_latency_ms_bucket{le=\"+Inf\"}"));
        assert!(output.contains("test_latency_ms_sum"));
        assert!(output.contains("test_latency_ms_count 3"));
    }

    #[test]
    fn test_histogram_with_labels() {
        let h = LatencyHistogram::new("test", "Test histogram", &[10.0, 50.0, 100.0])
            .with_label("exchange", "kraken")
            .with_label("operation", "place_order");

        h.observe(25.0);

        let output = h.to_prometheus();
        assert!(output.contains("exchange=\"kraken\""));
        assert!(output.contains("operation=\"place_order\""));
    }

    #[test]
    fn test_labeled_registry() {
        let registry = LabeledHistogramRegistry::new_latency("api_latency", "API call latency");

        registry.observe("endpoint", "place_order", 50.0);
        registry.observe("endpoint", "cancel_order", 30.0);
        registry.observe("endpoint", "place_order", 60.0);

        let place_order = registry.with_label("endpoint", "place_order");
        assert_eq!(place_order.count(), 2);

        let cancel_order = registry.with_label("endpoint", "cancel_order");
        assert_eq!(cancel_order.count(), 1);

        let output = registry.to_prometheus();
        assert!(output.contains("endpoint=\"place_order\""));
        assert!(output.contains("endpoint=\"cancel_order\""));
    }

    #[test]
    fn test_execution_latency_histograms() {
        let histograms = ExecutionLatencyHistograms::new();

        histograms.record_order_placement("kraken", 100.0);
        histograms.record_order_placement("binance", 80.0);
        histograms.record_websocket_message("kraken", 5.0);
        histograms.record_signal_processing(15.0);

        let output = histograms.to_prometheus();

        assert!(output.contains("execution_order_placement_latency_ms"));
        assert!(output.contains("exchange=\"kraken\""));
        assert!(output.contains("exchange=\"binance\""));
        assert!(output.contains("exchange_ws_message_latency_ms"));
        assert!(output.contains("execution_signal_processing_latency_ms"));
    }

    #[test]
    fn test_histogram_reset() {
        let h = LatencyHistogram::new_latency_ms("test");
        h.observe(10.0);
        h.observe(20.0);

        assert_eq!(h.count(), 2);

        h.reset();

        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_global_histograms() {
        let histograms = global_latency_histograms();
        histograms.record_signal_processing(10.0);

        let output = latency_prometheus_metrics();
        assert!(output.contains("execution_signal_processing_latency_ms"));
    }

    #[test]
    fn test_timer_guard() {
        let h = Arc::new(LatencyHistogram::new_latency_ms("timer_test"));

        {
            let _timer = LatencyTimer::new(h.clone());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(h.count(), 1);
        assert!(h.sum() >= 10.0);
    }

    #[test]
    fn test_observe_duration() {
        let h = LatencyHistogram::new_latency_ms("duration_test");
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(5));
        h.observe_duration(start);

        assert_eq!(h.count(), 1);
        assert!(h.sum() >= 5.0);
    }

    #[test]
    fn test_empty_histogram_percentiles() {
        let h = LatencyHistogram::new_latency_ms("empty");
        assert!(h.p50().is_none());
        assert!(h.p99().is_none());
        assert!(h.mean().is_none());
    }

    #[test]
    fn test_bucket_boundaries() {
        // Test that values exactly on bucket boundaries are counted correctly
        let h = LatencyHistogram::new("test", "test", &[10.0, 20.0, 30.0]);

        h.observe(10.0); // Should be in bucket le=10
        h.observe(20.0); // Should be in bucket le=20
        h.observe(30.0); // Should be in bucket le=30

        let output = h.to_prometheus();

        // Check cumulative counts
        assert!(output.contains("le=\"10\"} 1"));
        assert!(output.contains("le=\"20\"} 2"));
        assert!(output.contains("le=\"30\"} 3"));
        assert!(output.contains("le=\"+Inf\"} 3"));
    }
}
