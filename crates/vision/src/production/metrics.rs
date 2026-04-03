//! Metrics collection and export for production monitoring.
//!
//! This module provides comprehensive metrics tracking:
//! - Counter, gauge, and histogram metrics
//! - Prometheus-compatible export format
//! - Time-series data aggregation
//! - Custom metrics registry
//! - Performance dashboards support

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Metric type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Counter - monotonically increasing value
    Counter,
    /// Gauge - value that can go up or down
    Gauge,
    /// Histogram - distribution of values
    Histogram,
    /// Summary - statistical summary of observations
    Summary,
}

impl MetricType {
    /// Convert to Prometheus type string.
    pub fn as_prometheus_type(&self) -> &str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
        }
    }
}

/// Counter metric - monotonically increasing value.
#[derive(Debug, Clone)]
pub struct Counter {
    name: String,
    help: String,
    value: Arc<RwLock<f64>>,
    labels: HashMap<String, String>,
}

impl Counter {
    /// Create a new counter.
    pub fn new(name: String, help: String) -> Self {
        Self {
            name,
            help,
            value: Arc::new(RwLock::new(0.0)),
            labels: HashMap::new(),
        }
    }

    /// Increment the counter by 1.
    pub fn inc(&self) {
        self.add(1.0);
    }

    /// Add a value to the counter.
    pub fn add(&self, value: f64) {
        if let Ok(mut v) = self.value.write() {
            *v += value;
        }
    }

    /// Get current value.
    pub fn get(&self) -> f64 {
        self.value.read().ok().map(|v| *v).unwrap_or(0.0)
    }

    /// Reset counter to zero.
    pub fn reset(&self) {
        if let Ok(mut v) = self.value.write() {
            *v = 0.0;
        }
    }

    /// Add labels to the counter.
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Export to Prometheus format.
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# HELP {} {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {} counter\n", self.name));

        if self.labels.is_empty() {
            output.push_str(&format!("{} {}\n", self.name, self.get()));
        } else {
            let labels_str = self.format_labels();
            output.push_str(&format!("{}{{{}}}\n", self.name, labels_str));
        }

        output
    }

    fn format_labels(&self) -> String {
        self.labels
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Gauge metric - value that can go up or down.
#[derive(Debug, Clone)]
pub struct Gauge {
    name: String,
    help: String,
    value: Arc<RwLock<f64>>,
    labels: HashMap<String, String>,
}

impl Gauge {
    /// Create a new gauge.
    pub fn new(name: String, help: String) -> Self {
        Self {
            name,
            help,
            value: Arc::new(RwLock::new(0.0)),
            labels: HashMap::new(),
        }
    }

    /// Set the gauge value.
    pub fn set(&self, value: f64) {
        if let Ok(mut v) = self.value.write() {
            *v = value;
        }
    }

    /// Increment the gauge by 1.
    pub fn inc(&self) {
        self.add(1.0);
    }

    /// Decrement the gauge by 1.
    pub fn dec(&self) {
        self.sub(1.0);
    }

    /// Add a value to the gauge.
    pub fn add(&self, value: f64) {
        if let Ok(mut v) = self.value.write() {
            *v += value;
        }
    }

    /// Subtract a value from the gauge.
    pub fn sub(&self, value: f64) {
        if let Ok(mut v) = self.value.write() {
            *v -= value;
        }
    }

    /// Get current value.
    pub fn get(&self) -> f64 {
        self.value.read().ok().map(|v| *v).unwrap_or(0.0)
    }

    /// Add labels to the gauge.
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Export to Prometheus format.
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# HELP {} {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {} gauge\n", self.name));
        output.push_str(&format!("{} {}\n", self.name, self.get()));
        output
    }
}

/// Histogram metric - distribution of values.
#[derive(Debug, Clone)]
pub struct Histogram {
    name: String,
    help: String,
    buckets: Vec<f64>,
    counts: Arc<RwLock<Vec<u64>>>,
    sum: Arc<RwLock<f64>>,
    count: Arc<RwLock<u64>>,
}

impl Histogram {
    /// Create a new histogram with default buckets.
    pub fn new(name: String, help: String) -> Self {
        Self::with_buckets(name, help, Self::default_buckets())
    }

    /// Create a histogram with custom buckets.
    pub fn with_buckets(name: String, help: String, buckets: Vec<f64>) -> Self {
        let counts = vec![0; buckets.len()];
        Self {
            name,
            help,
            buckets,
            counts: Arc::new(RwLock::new(counts)),
            sum: Arc::new(RwLock::new(0.0)),
            count: Arc::new(RwLock::new(0)),
        }
    }

    /// Default histogram buckets (milliseconds).
    pub fn default_buckets() -> Vec<f64> {
        vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ]
    }

    /// Observe a value.
    pub fn observe(&self, value: f64) {
        // Update sum
        if let Ok(mut sum) = self.sum.write() {
            *sum += value;
        }

        // Update count
        if let Ok(mut count) = self.count.write() {
            *count += 1;
        }

        // Update buckets
        if let Ok(mut counts) = self.counts.write() {
            for (i, &bucket) in self.buckets.iter().enumerate() {
                if value <= bucket {
                    counts[i] += 1;
                }
            }
        }
    }

    /// Get the sum of all observed values.
    pub fn sum(&self) -> f64 {
        self.sum.read().ok().map(|v| *v).unwrap_or(0.0)
    }

    /// Get the count of observations.
    pub fn count(&self) -> u64 {
        self.count.read().ok().map(|v| *v).unwrap_or(0)
    }

    /// Get the mean of observed values.
    pub fn mean(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            0.0
        } else {
            self.sum() / count as f64
        }
    }

    /// Export to Prometheus format.
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# HELP {} {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {} histogram\n", self.name));

        if let Ok(counts) = self.counts.read() {
            for (i, &bucket) in self.buckets.iter().enumerate() {
                output.push_str(&format!(
                    "{}_bucket{{le=\"{}\"}} {}\n",
                    self.name, bucket, counts[i]
                ));
            }
            output.push_str(&format!(
                "{}_bucket{{le=\"+Inf\"}} {}\n",
                self.name,
                self.count()
            ));
        }

        output.push_str(&format!("{}_sum {}\n", self.name, self.sum()));
        output.push_str(&format!("{}_count {}\n", self.name, self.count()));

        output
    }
}

/// Metrics registry for managing all metrics.
pub struct MetricsRegistry {
    counters: Arc<RwLock<HashMap<String, Counter>>>,
    gauges: Arc<RwLock<HashMap<String, Gauge>>>,
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
}

impl MetricsRegistry {
    /// Create a new metrics registry.
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a counter.
    pub fn register_counter(&self, name: String, help: String) -> Counter {
        let counter = Counter::new(name.clone(), help);
        if let Ok(mut counters) = self.counters.write() {
            counters.insert(name, counter.clone());
        }
        counter
    }

    /// Register a gauge.
    pub fn register_gauge(&self, name: String, help: String) -> Gauge {
        let gauge = Gauge::new(name.clone(), help);
        if let Ok(mut gauges) = self.gauges.write() {
            gauges.insert(name, gauge.clone());
        }
        gauge
    }

    /// Register a histogram.
    pub fn register_histogram(&self, name: String, help: String) -> Histogram {
        let histogram = Histogram::new(name.clone(), help);
        if let Ok(mut histograms) = self.histograms.write() {
            histograms.insert(name, histogram.clone());
        }
        histogram
    }

    /// Get a counter by name.
    pub fn get_counter(&self, name: &str) -> Option<Counter> {
        self.counters.read().ok().and_then(|c| c.get(name).cloned())
    }

    /// Get a gauge by name.
    pub fn get_gauge(&self, name: &str) -> Option<Gauge> {
        self.gauges.read().ok().and_then(|g| g.get(name).cloned())
    }

    /// Get a histogram by name.
    pub fn get_histogram(&self, name: &str) -> Option<Histogram> {
        self.histograms
            .read()
            .ok()
            .and_then(|h| h.get(name).cloned())
    }

    /// Export all metrics in Prometheus format.
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();

        // Export counters
        if let Ok(counters) = self.counters.read() {
            for counter in counters.values() {
                output.push_str(&counter.to_prometheus());
            }
        }

        // Export gauges
        if let Ok(gauges) = self.gauges.read() {
            for gauge in gauges.values() {
                output.push_str(&gauge.to_prometheus());
            }
        }

        // Export histograms
        if let Ok(histograms) = self.histograms.read() {
            for histogram in histograms.values() {
                output.push_str(&histogram.to_prometheus());
            }
        }

        output
    }

    /// Clear all metrics.
    pub fn clear(&self) {
        if let Ok(mut counters) = self.counters.write() {
            counters.clear();
        }
        if let Ok(mut gauges) = self.gauges.write() {
            gauges.clear();
        }
        if let Ok(mut histograms) = self.histograms.write() {
            histograms.clear();
        }
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-defined metrics for the vision pipeline.
pub struct PipelineMetrics {
    pub registry: MetricsRegistry,
    pub predictions_total: Counter,
    pub predictions_latency: Histogram,
    pub cache_hits: Counter,
    pub cache_misses: Counter,
    pub errors_total: Counter,
    pub active_requests: Gauge,
    pub memory_usage_bytes: Gauge,
}

impl PipelineMetrics {
    /// Create a new pipeline metrics collection.
    pub fn new() -> Self {
        let registry = MetricsRegistry::new();

        let predictions_total = registry.register_counter(
            "vision_predictions_total".to_string(),
            "Total number of predictions made".to_string(),
        );

        let predictions_latency = registry.register_histogram(
            "vision_predictions_latency_seconds".to_string(),
            "Prediction latency in seconds".to_string(),
        );

        let cache_hits = registry.register_counter(
            "vision_cache_hits_total".to_string(),
            "Total number of cache hits".to_string(),
        );

        let cache_misses = registry.register_counter(
            "vision_cache_misses_total".to_string(),
            "Total number of cache misses".to_string(),
        );

        let errors_total = registry.register_counter(
            "vision_errors_total".to_string(),
            "Total number of errors".to_string(),
        );

        let active_requests = registry.register_gauge(
            "vision_active_requests".to_string(),
            "Number of active prediction requests".to_string(),
        );

        let memory_usage_bytes = registry.register_gauge(
            "vision_memory_usage_bytes".to_string(),
            "Memory usage in bytes".to_string(),
        );

        Self {
            registry,
            predictions_total,
            predictions_latency,
            cache_hits,
            cache_misses,
            errors_total,
            active_requests,
            memory_usage_bytes,
        }
    }

    /// Record a prediction.
    pub fn record_prediction(&self, latency_seconds: f64) {
        self.predictions_total.inc();
        self.predictions_latency.observe(latency_seconds);
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&self) {
        self.cache_hits.inc();
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&self) {
        self.cache_misses.inc();
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.errors_total.inc();
    }

    /// Get cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.get();
        let misses = self.cache_misses.get();
        let total = hits + misses;
        if total == 0.0 { 0.0 } else { hits / total }
    }

    /// Export all metrics in Prometheus format.
    pub fn export_prometheus(&self) -> String {
        self.registry.export_prometheus()
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new("test".to_string(), "Test counter".to_string());
        assert_eq!(counter.get(), 0.0);

        counter.inc();
        assert_eq!(counter.get(), 1.0);

        counter.add(5.0);
        assert_eq!(counter.get(), 6.0);

        counter.reset();
        assert_eq!(counter.get(), 0.0);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new("test".to_string(), "Test gauge".to_string());
        assert_eq!(gauge.get(), 0.0);

        gauge.set(10.0);
        assert_eq!(gauge.get(), 10.0);

        gauge.inc();
        assert_eq!(gauge.get(), 11.0);

        gauge.dec();
        assert_eq!(gauge.get(), 10.0);

        gauge.add(5.0);
        assert_eq!(gauge.get(), 15.0);

        gauge.sub(3.0);
        assert_eq!(gauge.get(), 12.0);
    }

    #[test]
    fn test_histogram() {
        let histogram = Histogram::new("test".to_string(), "Test histogram".to_string());

        histogram.observe(0.5);
        histogram.observe(1.5);
        histogram.observe(2.5);

        assert_eq!(histogram.count(), 3);
        assert_eq!(histogram.sum(), 4.5);
        assert_eq!(histogram.mean(), 1.5);
    }

    #[test]
    fn test_metrics_registry() {
        let registry = MetricsRegistry::new();

        let counter = registry.register_counter("test_counter".to_string(), "Test".to_string());
        counter.inc();

        let retrieved = registry.get_counter("test_counter").unwrap();
        assert_eq!(retrieved.get(), 1.0);
    }

    #[test]
    fn test_pipeline_metrics() {
        let metrics = PipelineMetrics::new();

        metrics.record_prediction(0.001);
        assert_eq!(metrics.predictions_total.get(), 1.0);

        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        assert_eq!(metrics.cache_hit_rate(), 2.0 / 3.0);
    }

    #[test]
    fn test_prometheus_export() {
        let counter = Counter::new("test_counter".to_string(), "Test counter".to_string());
        counter.inc();

        let output = counter.to_prometheus();
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
        assert!(output.contains("test_counter"));
    }

    #[test]
    fn test_histogram_buckets() {
        let buckets = vec![1.0, 5.0, 10.0];
        let histogram =
            Histogram::with_buckets("test".to_string(), "Test".to_string(), buckets.clone());

        histogram.observe(0.5);
        histogram.observe(3.0);
        histogram.observe(7.0);
        histogram.observe(15.0);

        assert_eq!(histogram.count(), 4);
    }

    #[test]
    fn test_registry_clear() {
        let registry = MetricsRegistry::new();
        registry.register_counter("test".to_string(), "Test".to_string());

        registry.clear();
        assert!(registry.get_counter("test").is_none());
    }
}
