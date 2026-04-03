//! Prometheus Metrics for Optimizer Service
//!
//! Provides comprehensive metrics for monitoring optimization performance,
//! data collection, and service health.
//!
//! # Metrics Categories
//!
//! ## Optimization Metrics
//! - `optimizer_optimization_duration_seconds` - Duration of optimization runs
//! - `optimizer_best_score` - Best optimization score by asset
//! - `optimizer_best_return_pct` - Best backtest return percentage
//! - `optimizer_best_win_rate` - Best backtest win rate
//! - `optimizer_best_max_drawdown` - Best backtest max drawdown
//! - `optimizer_trials_total` - Total optimization trials run
//!
//! ## Data Collection Metrics
//! - `optimizer_collection_duration_seconds` - Duration of data collection runs
//! - `optimizer_collection_success_total` - Successful collection runs
//! - `optimizer_collection_failure_total` - Failed collection runs
//! - `optimizer_candles_collected_total` - Total candles collected
//!
//! ## Service Metrics
//! - `optimizer_healthy` - Service health status (1 = healthy, 0 = unhealthy)
//! - `optimizer_uptime_seconds` - Service uptime
//! - `optimizer_scheduled_runs_total` - Total scheduled optimization runs
//!
//! # Usage
//!
//! ```rust,ignore
//! let metrics = MetricsRegistry::new();
//!
//! // Record optimization
//! metrics.record_optimization("BTC", 42.5, 15.3, 65.0, 8.2, 120.5);
//!
//! // Get metrics text for Prometheus scraping
//! let text = metrics.get_metrics_text();
//! ```

use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, Opts,
    Registry, TextEncoder,
};

use std::time::Instant;
use tracing::{debug, error};

/// Metrics registry for the optimizer service
pub struct MetricsRegistry {
    /// Prometheus registry
    registry: Registry,

    /// Service start time for uptime calculation
    start_time: Instant,

    // === Optimization Metrics ===
    /// Duration of optimization runs (histogram)
    optimization_duration: HistogramVec,

    /// Best optimization score by asset
    best_score: GaugeVec,

    /// Best backtest return percentage by asset
    best_return_pct: GaugeVec,

    /// Best backtest win rate by asset
    best_win_rate: GaugeVec,

    /// Best backtest max drawdown by asset
    best_max_drawdown: GaugeVec,

    /// Total optimization trials run
    trials_total: CounterVec,

    // === Data Collection Metrics ===
    /// Duration of data collection runs
    collection_duration: Histogram,

    /// Successful collection runs
    collection_success: Counter,

    /// Failed collection runs
    collection_failure: Counter,

    /// Total candles collected
    candles_collected: CounterVec,

    // === Service Metrics ===
    /// Service health status
    healthy: Gauge,

    /// Service uptime
    uptime_seconds: Gauge,

    /// Total scheduled optimization runs
    scheduled_runs_total: Counter,

    /// Successful scheduled runs
    scheduled_runs_success: Counter,

    /// Failed scheduled runs
    scheduled_runs_failure: Counter,

    /// Last scheduled run duration
    last_scheduled_duration: Gauge,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        let registry = Registry::new();

        // Optimization metrics
        let optimization_duration = HistogramVec::new(
            HistogramOpts::new(
                "optimizer_optimization_duration_seconds",
                "Duration of optimization runs in seconds",
            )
            .buckets(vec![10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 3600.0]),
            &["asset"],
        )
        .expect("Failed to create optimization_duration metric");

        let best_score = GaugeVec::new(
            Opts::new("optimizer_best_score", "Best optimization score by asset"),
            &["asset"],
        )
        .expect("Failed to create best_score metric");

        let best_return_pct = GaugeVec::new(
            Opts::new(
                "optimizer_best_return_pct",
                "Best backtest return percentage by asset",
            ),
            &["asset"],
        )
        .expect("Failed to create best_return_pct metric");

        let best_win_rate = GaugeVec::new(
            Opts::new("optimizer_best_win_rate", "Best backtest win rate by asset"),
            &["asset"],
        )
        .expect("Failed to create best_win_rate metric");

        let best_max_drawdown = GaugeVec::new(
            Opts::new(
                "optimizer_best_max_drawdown",
                "Best backtest max drawdown by asset",
            ),
            &["asset"],
        )
        .expect("Failed to create best_max_drawdown metric");

        let trials_total = CounterVec::new(
            Opts::new("optimizer_trials_total", "Total optimization trials run"),
            &["asset"],
        )
        .expect("Failed to create trials_total metric");

        // Data collection metrics
        let collection_duration = Histogram::with_opts(
            HistogramOpts::new(
                "optimizer_collection_duration_seconds",
                "Duration of data collection runs in seconds",
            )
            .buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]),
        )
        .expect("Failed to create collection_duration metric");

        let collection_success = Counter::new(
            "optimizer_collection_success_total",
            "Total successful data collection runs",
        )
        .expect("Failed to create collection_success metric");

        let collection_failure = Counter::new(
            "optimizer_collection_failure_total",
            "Total failed data collection runs",
        )
        .expect("Failed to create collection_failure metric");

        let candles_collected = CounterVec::new(
            Opts::new(
                "optimizer_candles_collected_total",
                "Total candles collected by asset and interval",
            ),
            &["asset", "interval"],
        )
        .expect("Failed to create candles_collected metric");

        // Service metrics
        let healthy = Gauge::new(
            "optimizer_healthy",
            "Service health status (1=healthy, 0=unhealthy)",
        )
        .expect("Failed to create healthy metric");

        let uptime_seconds = Gauge::new("optimizer_uptime_seconds", "Service uptime in seconds")
            .expect("Failed to create uptime_seconds metric");

        let scheduled_runs_total = Counter::new(
            "optimizer_scheduled_runs_total",
            "Total scheduled optimization runs",
        )
        .expect("Failed to create scheduled_runs_total metric");

        let scheduled_runs_success = Counter::new(
            "optimizer_scheduled_runs_success_total",
            "Total successful scheduled optimization runs",
        )
        .expect("Failed to create scheduled_runs_success metric");

        let scheduled_runs_failure = Counter::new(
            "optimizer_scheduled_runs_failure_total",
            "Total failed scheduled optimization runs",
        )
        .expect("Failed to create scheduled_runs_failure metric");

        let last_scheduled_duration = Gauge::new(
            "optimizer_last_scheduled_duration_seconds",
            "Duration of the last scheduled optimization run",
        )
        .expect("Failed to create last_scheduled_duration metric");

        // Register all metrics
        let metrics = Self {
            registry,
            start_time: Instant::now(),
            optimization_duration,
            best_score,
            best_return_pct,
            best_win_rate,
            best_max_drawdown,
            trials_total,
            collection_duration,
            collection_success,
            collection_failure,
            candles_collected,
            healthy,
            uptime_seconds,
            scheduled_runs_total,
            scheduled_runs_success,
            scheduled_runs_failure,
            last_scheduled_duration,
        };

        // Register with the registry
        metrics.register_all();

        // Set initial health to healthy
        metrics.healthy.set(1.0);

        metrics
    }

    /// Register all metrics with the registry
    fn register_all(&self) {
        let _ = self
            .registry
            .register(Box::new(self.optimization_duration.clone()));
        let _ = self.registry.register(Box::new(self.best_score.clone()));
        let _ = self
            .registry
            .register(Box::new(self.best_return_pct.clone()));
        let _ = self.registry.register(Box::new(self.best_win_rate.clone()));
        let _ = self
            .registry
            .register(Box::new(self.best_max_drawdown.clone()));
        let _ = self.registry.register(Box::new(self.trials_total.clone()));
        let _ = self
            .registry
            .register(Box::new(self.collection_duration.clone()));
        let _ = self
            .registry
            .register(Box::new(self.collection_success.clone()));
        let _ = self
            .registry
            .register(Box::new(self.collection_failure.clone()));
        let _ = self
            .registry
            .register(Box::new(self.candles_collected.clone()));
        let _ = self.registry.register(Box::new(self.healthy.clone()));
        let _ = self
            .registry
            .register(Box::new(self.uptime_seconds.clone()));
        let _ = self
            .registry
            .register(Box::new(self.scheduled_runs_total.clone()));
        let _ = self
            .registry
            .register(Box::new(self.scheduled_runs_success.clone()));
        let _ = self
            .registry
            .register(Box::new(self.scheduled_runs_failure.clone()));
        let _ = self
            .registry
            .register(Box::new(self.last_scheduled_duration.clone()));
    }

    /// Record an optimization result
    pub fn record_optimization(
        &self,
        asset: &str,
        score: f64,
        return_pct: f64,
        win_rate: f64,
        max_drawdown: f64,
        duration_secs: f64,
    ) {
        self.optimization_duration
            .with_label_values(&[asset])
            .observe(duration_secs);

        self.best_score.with_label_values(&[asset]).set(score);
        self.best_return_pct
            .with_label_values(&[asset])
            .set(return_pct);
        self.best_win_rate.with_label_values(&[asset]).set(win_rate);
        self.best_max_drawdown
            .with_label_values(&[asset])
            .set(max_drawdown);

        debug!(
            "Recorded optimization metrics for {}: score={:.2}, return={:.2}%, duration={:.1}s",
            asset, score, return_pct, duration_secs
        );
    }

    /// Record optimization trials
    pub fn record_trials(&self, asset: &str, count: u64) {
        self.trials_total
            .with_label_values(&[asset])
            .inc_by(count as f64);
    }

    /// Record a successful data collection run
    pub fn record_collection_success(&self) {
        self.collection_success.inc();
    }

    /// Record a failed data collection run
    pub fn record_collection_failure(&self) {
        self.collection_failure.inc();
    }

    /// Record collection duration
    pub fn record_collection_duration(&self, duration_secs: f64) {
        self.collection_duration.observe(duration_secs);
    }

    /// Record candles collected
    pub fn record_candles_collected(&self, asset: &str, interval: u32, count: u64) {
        self.candles_collected
            .with_label_values(&[asset, &interval.to_string()])
            .inc_by(count as f64);
    }

    /// Record a scheduled run result
    pub fn record_scheduled_run(&self, success: bool, duration_secs: f64) {
        self.scheduled_runs_total.inc();

        if success {
            self.scheduled_runs_success.inc();
        } else {
            self.scheduled_runs_failure.inc();
        }

        self.last_scheduled_duration.set(duration_secs);
    }

    /// Set the service health status
    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.set(if healthy { 1.0 } else { 0.0 });
    }

    /// Update uptime metric
    pub fn update_uptime(&self) {
        self.uptime_seconds
            .set(self.start_time.elapsed().as_secs_f64());
    }

    /// Get metrics as Prometheus text format
    pub fn get_metrics_text(&self) -> String {
        // Update uptime before encoding
        self.update_uptime();

        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();

        let mut buffer = Vec::new();
        if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
            error!("Failed to encode metrics: {}", e);
            return String::new();
        }

        String::from_utf8(buffer).unwrap_or_default()
    }

    /// Get the uptime in seconds
    pub fn uptime(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registry_creation() {
        let metrics = MetricsRegistry::new();
        assert!(metrics.uptime() >= 0.0);
    }

    #[test]
    fn test_record_optimization() {
        let metrics = MetricsRegistry::new();
        metrics.record_optimization("BTC", 42.5, 15.3, 65.0, 8.2, 120.5);

        let text = metrics.get_metrics_text();
        assert!(text.contains("optimizer_best_score"));
        assert!(text.contains("BTC"));
    }

    #[test]
    fn test_record_collection() {
        let metrics = MetricsRegistry::new();
        metrics.record_collection_success();
        metrics.record_collection_success();
        metrics.record_collection_failure();

        let text = metrics.get_metrics_text();
        assert!(text.contains("optimizer_collection_success_total"));
        assert!(text.contains("optimizer_collection_failure_total"));
    }

    #[test]
    fn test_health_status() {
        let metrics = MetricsRegistry::new();

        // Should start healthy
        let text = metrics.get_metrics_text();
        assert!(text.contains("optimizer_healthy 1"));

        // Set unhealthy
        metrics.set_healthy(false);
        let text = metrics.get_metrics_text();
        assert!(text.contains("optimizer_healthy 0"));
    }

    #[test]
    fn test_uptime() {
        let metrics = MetricsRegistry::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(metrics.uptime() >= 0.01);
    }

    #[test]
    fn test_scheduled_runs() {
        let metrics = MetricsRegistry::new();
        metrics.record_scheduled_run(true, 60.0);
        metrics.record_scheduled_run(false, 30.0);

        let text = metrics.get_metrics_text();
        assert!(text.contains("optimizer_scheduled_runs_total 2"));
        assert!(text.contains("optimizer_scheduled_runs_success_total 1"));
        assert!(text.contains("optimizer_scheduled_runs_failure_total 1"));
    }

    #[test]
    fn test_candles_collected() {
        let metrics = MetricsRegistry::new();
        metrics.record_candles_collected("BTC", 60, 720);
        metrics.record_candles_collected("ETH", 60, 720);
        metrics.record_candles_collected("BTC", 15, 2880);

        let text = metrics.get_metrics_text();
        assert!(text.contains("optimizer_candles_collected_total"));
        assert!(text.contains("BTC"));
        assert!(text.contains("ETH"));
    }
}
