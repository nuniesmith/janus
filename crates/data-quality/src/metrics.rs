//! CNS Metrics Module for Data Quality Observability
//!
//! Provides Prometheus-compatible metrics for monitoring data quality pipeline
//! health, performance, and anomaly detection.
//!
//! ## Features
//!
//! - **Counters**: Track events, errors, validations
//! - **Gauges**: Current values, queue depths, connection states
//! - **Histograms**: Latency distributions, processing times
//! - **Labels**: Multi-dimensional metrics with exchange/symbol dimensions
//!
//! ## Example
//!
//! ```rust,ignore
//! use janus_data_quality::metrics::{MetricsRegistry, MetricsConfig};
//!
//! let config = MetricsConfig::default();
//! let registry = MetricsRegistry::new(config);
//!
//! // Record a validation
//! registry.record_validation("price_validator", true, 0.5);
//!
//! // Record an anomaly
//! registry.record_anomaly("Bybit", "BTCUSD", "statistical", "high");
//!
//! // Get Prometheus-format metrics
//! let output = registry.render();
//! ```

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Configuration for the metrics system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable CNS metrics reporting
    pub enabled: bool,
    /// Metrics prefix (e.g., "janus_data_quality")
    pub prefix: String,
    /// Histogram buckets for latency metrics (in seconds)
    pub latency_buckets: Vec<f64>,
    /// Histogram buckets for size metrics (in bytes)
    pub size_buckets: Vec<f64>,
    /// Maximum cardinality for label combinations
    pub max_cardinality: usize,
    /// Enable detailed per-symbol metrics
    pub per_symbol_metrics: bool,
    /// Enable detailed per-exchange metrics
    pub per_exchange_metrics: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prefix: "janus_data_quality".to_string(),
            latency_buckets: vec![
                0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                10.0,
            ],
            size_buckets: vec![
                64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0,
            ],
            max_cardinality: 10000,
            per_symbol_metrics: true,
            per_exchange_metrics: true,
        }
    }
}

/// A simple atomic counter
#[derive(Debug)]
pub struct Counter {
    value: AtomicU64,
    labels: HashMap<String, String>,
}

impl Counter {
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
            labels: HashMap::new(),
        }
    }

    pub fn with_labels(labels: HashMap<String, String>) -> Self {
        Self {
            value: AtomicU64::new(0),
            labels,
        }
    }

    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

/// A gauge that can go up and down
#[derive(Debug)]
pub struct Gauge {
    value: AtomicU64, // Stored as f64 bits
    labels: HashMap<String, String>,
}

impl Gauge {
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0f64.to_bits()),
            labels: HashMap::new(),
        }
    }

    pub fn with_labels(labels: HashMap<String, String>) -> Self {
        Self {
            value: AtomicU64::new(0f64.to_bits()),
            labels,
        }
    }

    pub fn set(&self, val: f64) {
        self.value.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn inc(&self) {
        self.add(1.0);
    }

    pub fn dec(&self) {
        self.sub(1.0);
    }

    pub fn add(&self, val: f64) {
        loop {
            let old = self.value.load(Ordering::Relaxed);
            let new = (f64::from_bits(old) + val).to_bits();
            if self
                .value
                .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    pub fn sub(&self, val: f64) {
        self.add(-val);
    }

    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

/// A histogram for tracking distributions
#[derive(Debug)]
pub struct Histogram {
    buckets: Vec<f64>,
    counts: Vec<AtomicU64>,
    sum: AtomicU64, // Stored as f64 bits
    count: AtomicU64,
    labels: HashMap<String, String>,
}

impl Histogram {
    pub fn new(buckets: Vec<f64>) -> Self {
        let len = buckets.len();
        Self {
            buckets,
            counts: (0..=len).map(|_| AtomicU64::new(0)).collect(),
            sum: AtomicU64::new(0f64.to_bits()),
            count: AtomicU64::new(0),
            labels: HashMap::new(),
        }
    }

    pub fn with_labels(buckets: Vec<f64>, labels: HashMap<String, String>) -> Self {
        let len = buckets.len();
        Self {
            buckets,
            counts: (0..=len).map(|_| AtomicU64::new(0)).collect(),
            sum: AtomicU64::new(0f64.to_bits()),
            count: AtomicU64::new(0),
            labels,
        }
    }

    pub fn observe(&self, val: f64) {
        // Find bucket
        let bucket_idx = self.buckets.iter().position(|&b| val <= b);
        let idx = bucket_idx.unwrap_or(self.buckets.len());

        // Increment bucket and all following
        for i in idx..self.counts.len() {
            self.counts[i].fetch_add(1, Ordering::Relaxed);
        }

        // Update sum
        loop {
            let old = self.sum.load(Ordering::Relaxed);
            let new = (f64::from_bits(old) + val).to_bits();
            if self
                .sum
                .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Increment count
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_duration(&self, start: Instant) {
        let duration = start.elapsed().as_secs_f64();
        self.observe(duration);
    }

    pub fn get_sum(&self) -> f64 {
        f64::from_bits(self.sum.load(Ordering::Relaxed))
    }

    pub fn get_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn get_bucket_counts(&self) -> Vec<u64> {
        self.counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect()
    }

    pub fn buckets(&self) -> &[f64] {
        &self.buckets
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }
}

/// Central metrics registry
#[derive(Debug)]
pub struct MetricsRegistry {
    config: MetricsConfig,

    // Validation metrics
    validations_total: Counter,
    validations_passed: Counter,
    validations_failed: Counter,
    validation_latency: Histogram,
    validators_by_type: RwLock<HashMap<String, Counter>>,

    // Anomaly metrics
    anomalies_detected: Counter,
    anomalies_by_severity: RwLock<HashMap<String, Counter>>,
    anomalies_by_type: RwLock<HashMap<String, Counter>>,

    // Pipeline metrics
    events_processed: Counter,
    events_by_type: RwLock<HashMap<String, Counter>>,
    processing_latency: Histogram,
    queue_depth: Gauge,
    batch_size: Histogram,

    // Export metrics
    exports_total: Counter,
    exports_failed: Counter,
    export_bytes: Counter,
    export_latency: Histogram,
    buffer_size: Gauge,

    // Connection metrics
    connections_active: Gauge,
    connections_by_exchange: RwLock<HashMap<String, Gauge>>,
    reconnections: Counter,

    // Data quality scores
    quality_score: Gauge,
    quality_by_exchange: RwLock<HashMap<String, Gauge>>,

    // Gap detection
    gaps_detected: Counter,
    gap_duration_seconds: Histogram,

    // Exchange-specific metrics
    exchange_latency: RwLock<HashMap<String, Histogram>>,
    exchange_events: RwLock<HashMap<String, Counter>>,

    // Symbol-specific metrics
    symbol_events: RwLock<HashMap<String, Counter>>,

    // Custom metrics
    custom_counters: RwLock<HashMap<String, Counter>>,
    custom_gauges: RwLock<HashMap<String, Gauge>>,
    custom_histograms: RwLock<HashMap<String, Histogram>>,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            validations_total: Counter::new(),
            validations_passed: Counter::new(),
            validations_failed: Counter::new(),
            validation_latency: Histogram::new(config.latency_buckets.clone()),
            validators_by_type: RwLock::new(HashMap::new()),

            anomalies_detected: Counter::new(),
            anomalies_by_severity: RwLock::new(HashMap::new()),
            anomalies_by_type: RwLock::new(HashMap::new()),

            events_processed: Counter::new(),
            events_by_type: RwLock::new(HashMap::new()),
            processing_latency: Histogram::new(config.latency_buckets.clone()),
            queue_depth: Gauge::new(),
            batch_size: Histogram::new(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),

            exports_total: Counter::new(),
            exports_failed: Counter::new(),
            export_bytes: Counter::new(),
            export_latency: Histogram::new(config.latency_buckets.clone()),
            buffer_size: Gauge::new(),

            connections_active: Gauge::new(),
            connections_by_exchange: RwLock::new(HashMap::new()),
            reconnections: Counter::new(),

            quality_score: Gauge::new(),
            quality_by_exchange: RwLock::new(HashMap::new()),

            gaps_detected: Counter::new(),
            gap_duration_seconds: Histogram::new(vec![
                1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0,
            ]),

            exchange_latency: RwLock::new(HashMap::new()),
            exchange_events: RwLock::new(HashMap::new()),

            symbol_events: RwLock::new(HashMap::new()),

            custom_counters: RwLock::new(HashMap::new()),
            custom_gauges: RwLock::new(HashMap::new()),
            custom_histograms: RwLock::new(HashMap::new()),

            config,
        }
    }

    // ==================== Validation Metrics ====================

    /// Record a validation event
    pub fn record_validation(&self, validator: &str, passed: bool, latency_secs: f64) {
        self.validations_total.inc();

        if passed {
            self.validations_passed.inc();
        } else {
            self.validations_failed.inc();
        }

        self.validation_latency.observe(latency_secs);

        // Track by validator type
        let mut validators = self.validators_by_type.write().unwrap();
        validators.entry(validator.to_string()).or_default().inc();
    }

    /// Get validation success rate
    pub fn validation_success_rate(&self) -> f64 {
        let total = self.validations_total.get();
        if total == 0 {
            return 1.0;
        }
        self.validations_passed.get() as f64 / total as f64
    }

    // ==================== Anomaly Metrics ====================

    /// Record an anomaly detection
    pub fn record_anomaly(
        &self,
        exchange: &str,
        _symbol: &str,
        anomaly_type: &str,
        severity: &str,
    ) {
        self.anomalies_detected.inc();

        // Track by severity
        let mut by_severity = self.anomalies_by_severity.write().unwrap();
        by_severity.entry(severity.to_string()).or_default().inc();

        // Track by type
        let mut by_type = self.anomalies_by_type.write().unwrap();
        by_type.entry(anomaly_type.to_string()).or_default().inc();

        // Track by exchange if enabled
        if self.config.per_exchange_metrics {
            let mut exchange_events = self.exchange_events.write().unwrap();
            let key = format!("{}_anomalies", exchange);
            exchange_events.entry(key).or_default().inc();
        }
    }

    /// Get anomaly count by severity
    pub fn anomaly_count(&self, severity: Option<&str>) -> u64 {
        match severity {
            Some(s) => {
                let by_severity = self.anomalies_by_severity.read().unwrap();
                by_severity.get(s).map(|c| c.get()).unwrap_or(0)
            }
            None => self.anomalies_detected.get(),
        }
    }

    // ==================== Pipeline Metrics ====================

    /// Record an event processed
    pub fn record_event(&self, event_type: &str, exchange: &str, symbol: &str) {
        self.events_processed.inc();

        // Track by type
        let mut by_type = self.events_by_type.write().unwrap();
        by_type.entry(event_type.to_string()).or_default().inc();

        // Track by exchange
        if self.config.per_exchange_metrics {
            let mut exchange_events = self.exchange_events.write().unwrap();
            exchange_events
                .entry(exchange.to_string())
                .or_default()
                .inc();
        }

        // Track by symbol
        if self.config.per_symbol_metrics {
            let mut symbol_events = self.symbol_events.write().unwrap();
            if symbol_events.len() < self.config.max_cardinality {
                symbol_events.entry(symbol.to_string()).or_default().inc();
            }
        }
    }

    /// Record processing latency
    pub fn record_processing_latency(&self, latency_secs: f64) {
        self.processing_latency.observe(latency_secs);
    }

    /// Record processing latency from a start time
    pub fn record_processing_duration(&self, start: Instant) {
        self.processing_latency.observe_duration(start);
    }

    /// Set current queue depth
    pub fn set_queue_depth(&self, depth: f64) {
        self.queue_depth.set(depth);
    }

    /// Record batch size
    pub fn record_batch_size(&self, size: usize) {
        self.batch_size.observe(size as f64);
    }

    // ==================== Export Metrics ====================

    /// Record an export operation
    pub fn record_export(&self, success: bool, bytes: u64, latency_secs: f64) {
        self.exports_total.inc();

        if !success {
            self.exports_failed.inc();
        }

        self.export_bytes.inc_by(bytes);
        self.export_latency.observe(latency_secs);
    }

    /// Set current buffer size
    pub fn set_buffer_size(&self, size: f64) {
        self.buffer_size.set(size);
    }

    /// Get export success rate
    pub fn export_success_rate(&self) -> f64 {
        let total = self.exports_total.get();
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.exports_failed.get() as f64 / total as f64)
    }

    // ==================== Connection Metrics ====================

    /// Record connection state change
    pub fn set_connection_active(&self, exchange: &str, active: bool) {
        if active {
            self.connections_active.inc();
        } else {
            self.connections_active.dec();
        }

        // Track by exchange
        let mut by_exchange = self.connections_by_exchange.write().unwrap();
        by_exchange
            .entry(exchange.to_string())
            .or_default()
            .set(if active { 1.0 } else { 0.0 });
    }

    /// Record a reconnection attempt
    pub fn record_reconnection(&self) {
        self.reconnections.inc();
    }

    // ==================== Quality Score Metrics ====================

    /// Set overall quality score
    pub fn set_quality_score(&self, score: f64) {
        self.quality_score.set(score);
    }

    /// Set quality score for an exchange
    pub fn set_exchange_quality(&self, exchange: &str, score: f64) {
        let mut by_exchange = self.quality_by_exchange.write().unwrap();
        by_exchange
            .entry(exchange.to_string())
            .or_default()
            .set(score);
    }

    // ==================== Gap Detection Metrics ====================

    /// Record a detected gap
    pub fn record_gap(&self, duration_secs: f64) {
        self.gaps_detected.inc();
        self.gap_duration_seconds.observe(duration_secs);
    }

    // ==================== Exchange-Specific Metrics ====================

    /// Record exchange latency
    pub fn record_exchange_latency(&self, exchange: &str, latency_secs: f64) {
        let mut latency_map = self.exchange_latency.write().unwrap();
        latency_map
            .entry(exchange.to_string())
            .or_insert_with(|| Histogram::new(self.config.latency_buckets.clone()))
            .observe(latency_secs);
    }

    // ==================== Custom Metrics ====================

    /// Register a custom counter
    pub fn register_counter(&self, name: &str) {
        let mut counters = self.custom_counters.write().unwrap();
        counters.entry(name.to_string()).or_default();
    }

    /// Increment a custom counter
    pub fn inc_counter(&self, name: &str) {
        let counters = self.custom_counters.read().unwrap();
        if let Some(counter) = counters.get(name) {
            counter.inc();
        }
    }

    /// Register a custom gauge
    pub fn register_gauge(&self, name: &str) {
        let mut gauges = self.custom_gauges.write().unwrap();
        gauges.entry(name.to_string()).or_default();
    }

    /// Set a custom gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        let gauges = self.custom_gauges.read().unwrap();
        if let Some(gauge) = gauges.get(name) {
            gauge.set(value);
        }
    }

    /// Register a custom histogram
    pub fn register_histogram(&self, name: &str, buckets: Vec<f64>) {
        let mut histograms = self.custom_histograms.write().unwrap();
        histograms
            .entry(name.to_string())
            .or_insert_with(|| Histogram::new(buckets));
    }

    /// Observe a value in a custom histogram
    pub fn observe_histogram(&self, name: &str, value: f64) {
        let histograms = self.custom_histograms.read().unwrap();
        if let Some(histogram) = histograms.get(name) {
            histogram.observe(value);
        }
    }

    // ==================== Rendering ====================

    /// Render all metrics in Prometheus format
    pub fn render(&self) -> String {
        let mut output = String::new();
        let prefix = &self.config.prefix;

        // Validation metrics
        output.push_str(&format!(
            "# HELP {prefix}_validations_total Total number of validations\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_validations_total counter\n"));
        output.push_str(&format!(
            "{prefix}_validations_total {}\n",
            self.validations_total.get()
        ));

        output.push_str(&format!(
            "{prefix}_validations_passed {}\n",
            self.validations_passed.get()
        ));
        output.push_str(&format!(
            "{prefix}_validations_failed {}\n",
            self.validations_failed.get()
        ));

        // Render validation latency histogram
        output.push_str(&self.render_histogram(
            &format!("{prefix}_validation_latency_seconds"),
            &self.validation_latency,
        ));

        // Anomaly metrics
        output.push_str(&format!(
            "# HELP {prefix}_anomalies_detected_total Total anomalies detected\n"
        ));
        output.push_str(&format!(
            "# TYPE {prefix}_anomalies_detected_total counter\n"
        ));
        output.push_str(&format!(
            "{prefix}_anomalies_detected_total {}\n",
            self.anomalies_detected.get()
        ));

        // Anomalies by severity
        let by_severity = self.anomalies_by_severity.read().unwrap();
        for (severity, counter) in by_severity.iter() {
            output.push_str(&format!(
                "{prefix}_anomalies_by_severity{{severity=\"{severity}\"}} {}\n",
                counter.get()
            ));
        }

        // Pipeline metrics
        output.push_str(&format!(
            "# HELP {prefix}_events_processed_total Total events processed\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_events_processed_total counter\n"));
        output.push_str(&format!(
            "{prefix}_events_processed_total {}\n",
            self.events_processed.get()
        ));

        output.push_str(&self.render_histogram(
            &format!("{prefix}_processing_latency_seconds"),
            &self.processing_latency,
        ));

        output.push_str(&format!(
            "# HELP {prefix}_queue_depth Current queue depth\n"
        ));
        output.push_str(&format!("# TYPE {prefix}_queue_depth gauge\n"));
        output.push_str(&format!(
            "{prefix}_queue_depth {}\n",
            self.queue_depth.get()
        ));

        // Export metrics
        output.push_str(&format!(
            "{prefix}_exports_total {}\n",
            self.exports_total.get()
        ));
        output.push_str(&format!(
            "{prefix}_exports_failed {}\n",
            self.exports_failed.get()
        ));
        output.push_str(&format!(
            "{prefix}_export_bytes_total {}\n",
            self.export_bytes.get()
        ));
        output.push_str(&format!(
            "{prefix}_buffer_size {}\n",
            self.buffer_size.get()
        ));

        // Connection metrics
        output.push_str(&format!(
            "{prefix}_connections_active {}\n",
            self.connections_active.get()
        ));
        output.push_str(&format!(
            "{prefix}_reconnections_total {}\n",
            self.reconnections.get()
        ));

        // Quality metrics
        output.push_str(&format!(
            "{prefix}_quality_score {}\n",
            self.quality_score.get()
        ));

        // Gap metrics
        output.push_str(&format!(
            "{prefix}_gaps_detected_total {}\n",
            self.gaps_detected.get()
        ));

        // Exchange-specific metrics
        let exchange_events = self.exchange_events.read().unwrap();
        for (exchange, counter) in exchange_events.iter() {
            output.push_str(&format!(
                "{prefix}_exchange_events{{exchange=\"{exchange}\"}} {}\n",
                counter.get()
            ));
        }

        // Custom metrics
        let custom_counters = self.custom_counters.read().unwrap();
        for (name, counter) in custom_counters.iter() {
            output.push_str(&format!("{prefix}_{name} {}\n", counter.get()));
        }

        let custom_gauges = self.custom_gauges.read().unwrap();
        for (name, gauge) in custom_gauges.iter() {
            output.push_str(&format!("{prefix}_{name} {}\n", gauge.get()));
        }

        output
    }

    /// Render a histogram in Prometheus format
    fn render_histogram(&self, name: &str, histogram: &Histogram) -> String {
        let mut output = String::new();
        let buckets = histogram.buckets();
        let counts = histogram.get_bucket_counts();

        output.push_str(&format!("# HELP {name} Histogram\n"));
        output.push_str(&format!("# TYPE {name} histogram\n"));

        for (i, bucket) in buckets.iter().enumerate() {
            output.push_str(&format!("{name}_bucket{{le=\"{bucket}\"}} {}\n", counts[i]));
        }

        output.push_str(&format!(
            "{name}_bucket{{le=\"+Inf\"}} {}\n",
            counts.last().unwrap_or(&0)
        ));
        output.push_str(&format!("{name}_sum {}\n", histogram.get_sum()));
        output.push_str(&format!("{name}_count {}\n", histogram.get_count()));

        output
    }

    /// Get a summary of current metrics
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            events_processed: self.events_processed.get(),
            validations_total: self.validations_total.get(),
            validation_success_rate: self.validation_success_rate(),
            anomalies_detected: self.anomalies_detected.get(),
            exports_total: self.exports_total.get(),
            export_success_rate: self.export_success_rate(),
            connections_active: self.connections_active.get() as u64,
            quality_score: self.quality_score.get(),
            gaps_detected: self.gaps_detected.get(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        // Note: This is a simplified reset. For atomic counters,
        // we'd need to store them differently to support reset.
        // For now, this serves as a documentation of the interface.
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new(MetricsConfig::default())
    }
}

/// Summary of key metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub events_processed: u64,
    pub validations_total: u64,
    pub validation_success_rate: f64,
    pub anomalies_detected: u64,
    pub exports_total: u64,
    pub export_success_rate: f64,
    pub connections_active: u64,
    pub quality_score: f64,
    pub gaps_detected: u64,
}

/// Timer utility for measuring durations
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get elapsed seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Record elapsed time to a histogram
    pub fn record_to(&self, histogram: &Histogram) {
        histogram.observe_duration(self.start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);

        counter.inc();
        assert_eq!(counter.get(), 1);

        counter.inc_by(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new();
        assert_eq!(gauge.get(), 0.0);

        gauge.set(10.0);
        assert_eq!(gauge.get(), 10.0);

        gauge.inc();
        assert_eq!(gauge.get(), 11.0);

        gauge.dec();
        assert_eq!(gauge.get(), 10.0);

        gauge.add(5.5);
        assert!((gauge.get() - 15.5).abs() < 1e-10);
    }

    #[test]
    fn test_histogram() {
        let histogram = Histogram::new(vec![1.0, 5.0, 10.0]);

        histogram.observe(0.5);
        histogram.observe(3.0);
        histogram.observe(7.0);
        histogram.observe(15.0);

        assert_eq!(histogram.get_count(), 4);
        assert!((histogram.get_sum() - 25.5).abs() < 1e-10);

        let counts = histogram.get_bucket_counts();
        assert_eq!(counts[0], 1); // <= 1.0
        assert_eq!(counts[1], 2); // <= 5.0
        assert_eq!(counts[2], 3); // <= 10.0
        assert_eq!(counts[3], 4); // +Inf
    }

    #[test]
    fn test_metrics_registry() {
        let registry = MetricsRegistry::default();

        // Record validations
        registry.record_validation("price", true, 0.001);
        registry.record_validation("volume", false, 0.002);

        assert_eq!(registry.validations_total.get(), 2);
        assert_eq!(registry.validations_passed.get(), 1);
        assert_eq!(registry.validations_failed.get(), 1);
        assert!((registry.validation_success_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_anomaly_recording() {
        let registry = MetricsRegistry::default();

        registry.record_anomaly("Bybit", "BTCUSD", "statistical", "high");
        registry.record_anomaly("Bybit", "ETHUSDT", "sequence", "medium");
        registry.record_anomaly("Coinbase", "BTCUSD", "statistical", "high");

        assert_eq!(registry.anomalies_detected.get(), 3);
        assert_eq!(registry.anomaly_count(Some("high")), 2);
        assert_eq!(registry.anomaly_count(Some("medium")), 1);
        assert_eq!(registry.anomaly_count(None), 3);
    }

    #[test]
    fn test_event_recording() {
        let config = MetricsConfig {
            per_exchange_metrics: true,
            per_symbol_metrics: true,
            ..Default::default()
        };
        let registry = MetricsRegistry::new(config);

        registry.record_event("trade", "Bybit", "BTCUSD");
        registry.record_event("trade", "Bybit", "ETHUSDT");
        registry.record_event("orderbook", "Coinbase", "BTCUSD");

        assert_eq!(registry.events_processed.get(), 3);
    }

    #[test]
    fn test_render() {
        let registry = MetricsRegistry::default();

        registry.record_validation("test", true, 0.001);
        registry.record_anomaly("Exchange", "Symbol", "type", "low");

        let output = registry.render();

        assert!(output.contains("validations_total"));
        assert!(output.contains("anomalies_detected"));
        assert!(output.contains("events_processed"));
    }

    #[test]
    fn test_summary() {
        let registry = MetricsRegistry::default();

        registry.record_validation("test", true, 0.001);
        registry.set_quality_score(0.95);

        let summary = registry.summary();

        assert_eq!(summary.validations_total, 1);
        assert_eq!(summary.validation_success_rate, 1.0);
        assert!((summary.quality_score - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let elapsed = timer.elapsed_secs();
        assert!(elapsed >= 0.01);
        assert!(elapsed < 0.1);
    }

    #[test]
    fn test_custom_metrics() {
        let registry = MetricsRegistry::default();

        // Register and use custom counter
        registry.register_counter("my_custom_counter");
        registry.inc_counter("my_custom_counter");
        registry.inc_counter("my_custom_counter");

        // Register and use custom gauge
        registry.register_gauge("my_custom_gauge");
        registry.set_gauge("my_custom_gauge", 42.0);

        // Register and use custom histogram
        registry.register_histogram("my_custom_histogram", vec![1.0, 10.0, 100.0]);
        registry.observe_histogram("my_custom_histogram", 5.0);
    }

    #[test]
    fn test_connection_metrics() {
        let registry = MetricsRegistry::default();

        registry.set_connection_active("Bybit", true);
        assert_eq!(registry.connections_active.get(), 1.0);

        registry.set_connection_active("Coinbase", true);
        assert_eq!(registry.connections_active.get(), 2.0);

        registry.set_connection_active("Bybit", false);
        assert_eq!(registry.connections_active.get(), 1.0);

        registry.record_reconnection();
        assert_eq!(registry.reconnections.get(), 1);
    }

    #[test]
    fn test_export_metrics() {
        let registry = MetricsRegistry::default();

        registry.record_export(true, 1024, 0.05);
        registry.record_export(true, 2048, 0.03);
        registry.record_export(false, 0, 0.1);

        assert_eq!(registry.exports_total.get(), 3);
        assert_eq!(registry.exports_failed.get(), 1);
        assert_eq!(registry.export_bytes.get(), 3072);
        assert!((registry.export_success_rate() - 0.666666).abs() < 0.01);
    }

    #[test]
    fn test_gap_detection() {
        let registry = MetricsRegistry::default();

        registry.record_gap(5.0);
        registry.record_gap(30.0);
        registry.record_gap(120.0);

        assert_eq!(registry.gaps_detected.get(), 3);
    }
}
