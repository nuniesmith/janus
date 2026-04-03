//! # Signal Metrics Collector
//!
//! Prometheus metrics for signal generation and quality tracking.
//!
//! ## Per-Strategy Metrics
//!
//! In addition to aggregate signal metrics, this module provides per-strategy
//! counters and gauges so operators can monitor the performance of each
//! individual strategy (EMA Flip, Mean Reversion, Squeeze Breakout, VWAP
//! Scalper, ORB, EMA Ribbon, Trend Pullback, Momentum Surge, Multi-TF Trend).
//!
//! Metrics are labelled by `strategy_name` so Grafana dashboards can filter
//! and compare strategies side-by-side.

use prometheus::{
    Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    Opts, Registry,
};
use std::sync::Arc;

/// Signal metrics collector
pub struct SignalMetricsCollector {
    // Signal generation metrics
    pub signals_generated_total: IntCounterVec,
    pub signals_filtered_total: IntCounter,
    pub signals_actionable_total: IntCounter,

    // Signal quality metrics
    pub signal_confidence_avg: Gauge,
    pub signal_strength_avg: Gauge,
    pub signal_confidence_histogram: Histogram,
    pub signal_strength_histogram: Histogram,

    // Signal type distribution
    pub signals_by_type: IntCounterVec,

    // Signal source metrics
    pub signals_by_source: IntCounterVec,

    // Timeframe metrics
    pub signals_by_timeframe: IntCounterVec,

    // Signal processing metrics
    pub signal_generation_duration: Histogram,
    pub signal_validation_duration: Histogram,

    // Cache metrics
    pub signal_cache_hits: IntCounter,
    pub signal_cache_misses: IntCounter,
    pub signal_cache_size: IntGauge,

    // Batch metrics
    pub signal_batch_size: Histogram,
    pub signal_batch_processing_duration: Histogram,

    // ML inference metrics (for signals)
    pub ml_signals_total: IntCounter,
    pub ml_inference_duration: Histogram,
    pub ml_confidence_avg: Gauge,

    // Strategy metrics
    pub strategy_signals_total: IntCounterVec,
    pub strategy_execution_duration: Histogram,

    // Error metrics
    pub signal_generation_errors: IntCounter,
    pub signal_validation_errors: IntCounter,

    // ========================================================================
    // Per-Strategy Metrics (Kraken integration)
    // ========================================================================
    /// Signals generated per strategy (labels: strategy_name, signal_type)
    pub per_strategy_signals: IntCounterVec,
    /// Signals approved by the PropFirm validator per strategy
    pub per_strategy_approved: IntCounterVec,
    /// Signals rejected by the PropFirm validator per strategy
    pub per_strategy_rejected: IntCounterVec,
    /// Positions opened per strategy
    pub per_strategy_positions_opened: IntCounterVec,
    /// Positions closed per strategy
    pub per_strategy_positions_closed: IntCounterVec,
    /// Cumulative realised P&L per strategy (can go negative)
    pub per_strategy_pnl: GaugeVec,
    /// Average confidence of signals per strategy (updated on each signal)
    pub per_strategy_confidence: GaugeVec,
    /// Histogram of signal confidence values per strategy
    pub per_strategy_confidence_hist: HistogramVec,
    /// Number of strategies currently active (regime-gated) — informational
    pub active_strategy_count: IntGauge,
}

impl SignalMetricsCollector {
    /// Create new signal metrics collector
    pub fn new(registry: Arc<Registry>) -> Result<Self, prometheus::Error> {
        // Signal generation metrics
        let signals_generated_total = IntCounterVec::new(
            Opts::new(
                "janus_signals_generated_total",
                "Total number of signals generated",
            ),
            &["symbol", "timeframe", "signal_type"],
        )?;
        registry.register(Box::new(signals_generated_total.clone()))?;

        let signals_filtered_total = IntCounter::with_opts(Opts::new(
            "janus_signals_filtered_total",
            "Total number of signals filtered out by quality threshold",
        ))?;
        registry.register(Box::new(signals_filtered_total.clone()))?;

        let signals_actionable_total = IntCounter::with_opts(Opts::new(
            "janus_signals_actionable_total",
            "Total number of actionable signals (not Hold)",
        ))?;
        registry.register(Box::new(signals_actionable_total.clone()))?;

        // Signal quality metrics
        let signal_confidence_avg = Gauge::with_opts(Opts::new(
            "janus_signal_confidence_avg",
            "Average signal confidence (0.0 to 1.0)",
        ))?;
        registry.register(Box::new(signal_confidence_avg.clone()))?;

        let signal_strength_avg = Gauge::with_opts(Opts::new(
            "janus_signal_strength_avg",
            "Average signal strength (0.0 to 1.0)",
        ))?;
        registry.register(Box::new(signal_strength_avg.clone()))?;

        let signal_confidence_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_signal_confidence",
            "Distribution of signal confidence values",
        ))?;
        registry.register(Box::new(signal_confidence_histogram.clone()))?;

        let signal_strength_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_signal_strength",
            "Distribution of signal strength values",
        ))?;
        registry.register(Box::new(signal_strength_histogram.clone()))?;

        // Signal type distribution
        let signals_by_type = IntCounterVec::new(
            Opts::new("janus_signals_by_type", "Signals grouped by type"),
            &["signal_type"],
        )?;
        registry.register(Box::new(signals_by_type.clone()))?;

        // Signal source metrics
        let signals_by_source = IntCounterVec::new(
            Opts::new("janus_signals_by_source", "Signals grouped by source"),
            &["source_type"],
        )?;
        registry.register(Box::new(signals_by_source.clone()))?;

        // Timeframe metrics
        let signals_by_timeframe = IntCounterVec::new(
            Opts::new("janus_signals_by_timeframe", "Signals grouped by timeframe"),
            &["timeframe"],
        )?;
        registry.register(Box::new(signals_by_timeframe.clone()))?;

        // Signal processing metrics
        let signal_generation_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_signal_generation_duration_seconds",
            "Time taken to generate a signal",
        ))?;
        registry.register(Box::new(signal_generation_duration.clone()))?;

        let signal_validation_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_signal_validation_duration_seconds",
            "Time taken to validate a signal",
        ))?;
        registry.register(Box::new(signal_validation_duration.clone()))?;

        // Cache metrics
        let signal_cache_hits = IntCounter::with_opts(Opts::new(
            "janus_signal_cache_hits_total",
            "Total number of signal cache hits",
        ))?;
        registry.register(Box::new(signal_cache_hits.clone()))?;

        let signal_cache_misses = IntCounter::with_opts(Opts::new(
            "janus_signal_cache_misses_total",
            "Total number of signal cache misses",
        ))?;
        registry.register(Box::new(signal_cache_misses.clone()))?;

        let signal_cache_size = IntGauge::with_opts(Opts::new(
            "janus_signal_cache_size",
            "Current number of signals in cache",
        ))?;
        registry.register(Box::new(signal_cache_size.clone()))?;

        // Batch metrics
        let signal_batch_size = Histogram::with_opts(HistogramOpts::new(
            "janus_signal_batch_size",
            "Number of signals in batch processing",
        ))?;
        registry.register(Box::new(signal_batch_size.clone()))?;

        let signal_batch_processing_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_signal_batch_processing_duration_seconds",
            "Time taken to process signal batch",
        ))?;
        registry.register(Box::new(signal_batch_processing_duration.clone()))?;

        // ML inference metrics
        let ml_signals_total = IntCounter::with_opts(Opts::new(
            "janus_ml_signals_total",
            "Total number of ML-generated signals",
        ))?;
        registry.register(Box::new(ml_signals_total.clone()))?;

        let ml_inference_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_ml_inference_duration_seconds",
            "Time taken for ML model inference",
        ))?;
        registry.register(Box::new(ml_inference_duration.clone()))?;

        let ml_confidence_avg = Gauge::with_opts(Opts::new(
            "janus_ml_confidence_avg",
            "Average confidence of ML predictions",
        ))?;
        registry.register(Box::new(ml_confidence_avg.clone()))?;

        // Strategy metrics
        let strategy_signals_total = IntCounterVec::new(
            Opts::new(
                "janus_strategy_signals_total",
                "Signals generated by strategy",
            ),
            &["strategy_name"],
        )?;
        registry.register(Box::new(strategy_signals_total.clone()))?;

        let strategy_execution_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_strategy_execution_duration_seconds",
            "Time taken to execute strategy",
        ))?;
        registry.register(Box::new(strategy_execution_duration.clone()))?;

        // Error metrics
        let signal_generation_errors = IntCounter::with_opts(Opts::new(
            "janus_signal_generation_errors_total",
            "Total number of signal generation errors",
        ))?;
        registry.register(Box::new(signal_generation_errors.clone()))?;

        let signal_validation_errors = IntCounter::with_opts(Opts::new(
            "janus_signal_validation_errors_total",
            "Total number of signal validation errors",
        ))?;
        registry.register(Box::new(signal_validation_errors.clone()))?;

        // ====================================================================
        // Per-Strategy Metrics
        // ====================================================================

        let per_strategy_signals = IntCounterVec::new(
            Opts::new(
                "janus_per_strategy_signals_total",
                "Total signals generated per strategy",
            ),
            &["strategy_name", "signal_type"],
        )?;
        registry.register(Box::new(per_strategy_signals.clone()))?;

        let per_strategy_approved = IntCounterVec::new(
            Opts::new(
                "janus_per_strategy_approved_total",
                "Signals approved by validator per strategy",
            ),
            &["strategy_name"],
        )?;
        registry.register(Box::new(per_strategy_approved.clone()))?;

        let per_strategy_rejected = IntCounterVec::new(
            Opts::new(
                "janus_per_strategy_rejected_total",
                "Signals rejected by validator per strategy",
            ),
            &["strategy_name"],
        )?;
        registry.register(Box::new(per_strategy_rejected.clone()))?;

        let per_strategy_positions_opened = IntCounterVec::new(
            Opts::new(
                "janus_per_strategy_positions_opened_total",
                "Positions opened per strategy",
            ),
            &["strategy_name"],
        )?;
        registry.register(Box::new(per_strategy_positions_opened.clone()))?;

        let per_strategy_positions_closed = IntCounterVec::new(
            Opts::new(
                "janus_per_strategy_positions_closed_total",
                "Positions closed per strategy",
            ),
            &["strategy_name"],
        )?;
        registry.register(Box::new(per_strategy_positions_closed.clone()))?;

        let per_strategy_pnl = GaugeVec::new(
            Opts::new(
                "janus_per_strategy_pnl_cumulative",
                "Cumulative realised P&L per strategy (USD)",
            ),
            &["strategy_name"],
        )?;
        registry.register(Box::new(per_strategy_pnl.clone()))?;

        let per_strategy_confidence = GaugeVec::new(
            Opts::new(
                "janus_per_strategy_confidence_avg",
                "Latest average signal confidence per strategy",
            ),
            &["strategy_name"],
        )?;
        registry.register(Box::new(per_strategy_confidence.clone()))?;

        let per_strategy_confidence_hist = HistogramVec::new(
            HistogramOpts::new(
                "janus_per_strategy_confidence",
                "Distribution of signal confidence per strategy",
            ),
            &["strategy_name"],
        )?;
        registry.register(Box::new(per_strategy_confidence_hist.clone()))?;

        let active_strategy_count = IntGauge::with_opts(Opts::new(
            "janus_active_strategy_count",
            "Number of strategies currently regime-gated as active",
        ))?;
        registry.register(Box::new(active_strategy_count.clone()))?;

        Ok(Self {
            signals_generated_total,
            signals_filtered_total,
            signals_actionable_total,
            signal_confidence_avg,
            signal_strength_avg,
            signal_confidence_histogram,
            signal_strength_histogram,
            signals_by_type,
            signals_by_source,
            signals_by_timeframe,
            signal_generation_duration,
            signal_validation_duration,
            signal_cache_hits,
            signal_cache_misses,
            signal_cache_size,
            signal_batch_size,
            signal_batch_processing_duration,
            ml_signals_total,
            ml_inference_duration,
            ml_confidence_avg,
            strategy_signals_total,
            strategy_execution_duration,
            signal_generation_errors,
            signal_validation_errors,
            per_strategy_signals,
            per_strategy_approved,
            per_strategy_rejected,
            per_strategy_positions_opened,
            per_strategy_positions_closed,
            per_strategy_pnl,
            per_strategy_confidence,
            per_strategy_confidence_hist,
            active_strategy_count,
        })
    }

    /// Record signal generation
    pub fn record_signal_generated(
        &self,
        symbol: &str,
        timeframe: &str,
        signal_type: &str,
        confidence: f64,
        strength: f64,
    ) {
        self.signals_generated_total
            .with_label_values(&[symbol, timeframe, signal_type])
            .inc();

        self.signal_confidence_histogram.observe(confidence);
        self.signal_strength_histogram.observe(strength);
        self.signals_by_type.with_label_values(&[signal_type]).inc();
        self.signals_by_timeframe
            .with_label_values(&[timeframe])
            .inc();
    }

    /// Record signal filtered out
    pub fn record_signal_filtered(&self) {
        self.signals_filtered_total.inc();
    }

    /// Record actionable signal
    pub fn record_signal_actionable(&self) {
        self.signals_actionable_total.inc();
    }

    /// Update average signal quality
    pub fn update_signal_quality(&self, avg_confidence: f64, avg_strength: f64) {
        self.signal_confidence_avg.set(avg_confidence);
        self.signal_strength_avg.set(avg_strength);
    }

    /// Record signal source
    pub fn record_signal_source(&self, source_type: &str) {
        self.signals_by_source
            .with_label_values(&[source_type])
            .inc();
    }

    /// Record signal generation duration
    pub fn record_generation_duration(&self, duration_secs: f64) {
        self.signal_generation_duration.observe(duration_secs);
    }

    /// Record signal validation duration
    pub fn record_validation_duration(&self, duration_secs: f64) {
        self.signal_validation_duration.observe(duration_secs);
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.signal_cache_hits.inc();
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.signal_cache_misses.inc();
    }

    /// Update cache size
    pub fn update_cache_size(&self, size: i64) {
        self.signal_cache_size.set(size);
    }

    /// Record batch processing
    pub fn record_batch_processing(&self, batch_size: usize, duration_secs: f64) {
        self.signal_batch_size.observe(batch_size as f64);
        self.signal_batch_processing_duration.observe(duration_secs);
    }

    /// Record ML signal
    pub fn record_ml_signal(&self, inference_duration_secs: f64, confidence: f64) {
        self.ml_signals_total.inc();
        self.ml_inference_duration.observe(inference_duration_secs);
        self.ml_confidence_avg.set(confidence);
    }

    /// Record strategy execution
    pub fn record_strategy_execution(&self, strategy_name: &str, duration_secs: f64) {
        self.strategy_signals_total
            .with_label_values(&[strategy_name])
            .inc();
        self.strategy_execution_duration.observe(duration_secs);
    }

    /// Record signal generation error
    pub fn record_generation_error(&self) {
        self.signal_generation_errors.inc();
    }

    /// Record signal validation error
    pub fn record_validation_error(&self) {
        self.signal_validation_errors.inc();
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.signal_cache_hits.get() as f64;
        let misses = self.signal_cache_misses.get() as f64;
        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }

    // ========================================================================
    // Per-Strategy Metric Helpers
    // ========================================================================

    /// Record that a strategy generated a signal (BUY, SELL, or HOLD).
    pub fn record_strategy_signal_generated(
        &self,
        strategy_name: &str,
        signal_type: &str,
        confidence: f64,
    ) {
        self.per_strategy_signals
            .with_label_values(&[strategy_name, signal_type])
            .inc();
        self.per_strategy_confidence
            .with_label_values(&[strategy_name])
            .set(confidence);
        self.per_strategy_confidence_hist
            .with_label_values(&[strategy_name])
            .observe(confidence);
    }

    /// Record that a strategy signal was approved by the PropFirm validator.
    pub fn record_strategy_signal_approved(&self, strategy_name: &str) {
        self.per_strategy_approved
            .with_label_values(&[strategy_name])
            .inc();
    }

    /// Record that a strategy signal was rejected by the PropFirm validator.
    pub fn record_strategy_signal_rejected(&self, strategy_name: &str) {
        self.per_strategy_rejected
            .with_label_values(&[strategy_name])
            .inc();
    }

    /// Record that a position was opened by a given strategy.
    pub fn record_strategy_position_opened(&self, strategy_name: &str) {
        self.per_strategy_positions_opened
            .with_label_values(&[strategy_name])
            .inc();
    }

    /// Record that a position was closed by a given strategy, with the P&L.
    pub fn record_strategy_position_closed(&self, strategy_name: &str, pnl: f64) {
        self.per_strategy_positions_closed
            .with_label_values(&[strategy_name])
            .inc();
        // Accumulate P&L (GaugeVec allows add for cumulative tracking)
        self.per_strategy_pnl
            .with_label_values(&[strategy_name])
            .add(pnl);
    }

    /// Set the number of currently regime-active strategies.
    pub fn set_active_strategy_count(&self, count: i64) {
        self.active_strategy_count.set(count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_metrics_creation() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry);
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_record_signal_generated() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_signal_generated("BTC/USD", "1h", "BUY", 0.85, 0.75);
        metrics.record_signal_generated("ETH/USD", "5m", "SELL", 0.70, 0.60);

        assert_eq!(
            metrics
                .signals_generated_total
                .with_label_values(&["BTC/USD", "1h", "BUY"])
                .get(),
            1
        );
    }

    #[test]
    fn test_cache_metrics() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        assert_eq!(metrics.signal_cache_hits.get(), 2);
        assert_eq!(metrics.signal_cache_misses.get(), 1);

        let hit_rate = metrics.cache_hit_rate();
        assert!((hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_batch_processing() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_batch_processing(10, 0.5);
        metrics.record_batch_processing(20, 0.8);

        // Metrics should be recorded without error
    }

    #[test]
    fn test_ml_signal_recording() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_ml_signal(0.05, 0.85);

        assert_eq!(metrics.ml_signals_total.get(), 1);
    }

    #[test]
    fn test_strategy_execution() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_strategy_execution("EMA_CROSSOVER", 0.01);
        metrics.record_strategy_execution("RSI_REVERSAL", 0.015);

        assert_eq!(
            metrics
                .strategy_signals_total
                .with_label_values(&["EMA_CROSSOVER"])
                .get(),
            1
        );
    }

    #[test]
    fn test_error_recording() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_generation_error();
        metrics.record_validation_error();

        assert_eq!(metrics.signal_generation_errors.get(), 1);
        assert_eq!(metrics.signal_validation_errors.get(), 1);
    }

    // ====================================================================
    // Per-Strategy Metric Tests
    // ====================================================================

    #[test]
    fn test_per_strategy_signal_generated() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_strategy_signal_generated("ema_ribbon", "BUY", 0.85);
        metrics.record_strategy_signal_generated("ema_ribbon", "SELL", 0.72);
        metrics.record_strategy_signal_generated("trend_pullback", "BUY", 0.90);

        assert_eq!(
            metrics
                .per_strategy_signals
                .with_label_values(&["ema_ribbon", "BUY"])
                .get(),
            1
        );
        assert_eq!(
            metrics
                .per_strategy_signals
                .with_label_values(&["ema_ribbon", "SELL"])
                .get(),
            1
        );
        assert_eq!(
            metrics
                .per_strategy_signals
                .with_label_values(&["trend_pullback", "BUY"])
                .get(),
            1
        );
    }

    #[test]
    fn test_per_strategy_approved_rejected() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_strategy_signal_approved("momentum_surge");
        metrics.record_strategy_signal_approved("momentum_surge");
        metrics.record_strategy_signal_rejected("momentum_surge");

        assert_eq!(
            metrics
                .per_strategy_approved
                .with_label_values(&["momentum_surge"])
                .get(),
            2
        );
        assert_eq!(
            metrics
                .per_strategy_rejected
                .with_label_values(&["momentum_surge"])
                .get(),
            1
        );
    }

    #[test]
    fn test_per_strategy_position_pnl() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_strategy_position_opened("multi_tf_trend");
        metrics.record_strategy_position_closed("multi_tf_trend", 150.0);
        metrics.record_strategy_position_opened("multi_tf_trend");
        metrics.record_strategy_position_closed("multi_tf_trend", -50.0);

        assert_eq!(
            metrics
                .per_strategy_positions_opened
                .with_label_values(&["multi_tf_trend"])
                .get(),
            2
        );
        assert_eq!(
            metrics
                .per_strategy_positions_closed
                .with_label_values(&["multi_tf_trend"])
                .get(),
            2
        );
        // Cumulative P&L should be 150 + (-50) = 100
        let pnl = metrics
            .per_strategy_pnl
            .with_label_values(&["multi_tf_trend"])
            .get();
        assert!((pnl - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_per_strategy_confidence() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.record_strategy_signal_generated("ema_flip", "BUY", 0.65);
        let conf = metrics
            .per_strategy_confidence
            .with_label_values(&["ema_flip"])
            .get();
        assert!((conf - 0.65).abs() < 0.01);

        // Update to a new value
        metrics.record_strategy_signal_generated("ema_flip", "SELL", 0.80);
        let conf = metrics
            .per_strategy_confidence
            .with_label_values(&["ema_flip"])
            .get();
        assert!((conf - 0.80).abs() < 0.01);
    }

    #[test]
    fn test_active_strategy_count() {
        let registry = Arc::new(Registry::new());
        let metrics = SignalMetricsCollector::new(registry).unwrap();

        metrics.set_active_strategy_count(5);
        assert_eq!(metrics.active_strategy_count.get(), 5);

        metrics.set_active_strategy_count(3);
        assert_eq!(metrics.active_strategy_count.get(), 3);
    }
}
