//! # Brain Pipeline Metrics Collector
//!
//! Prometheus metrics for the brain-inspired trading pipeline.
//! Tracks pipeline evaluations, trade actions (proceed/block/reduce-only),
//! block reasons by stage, evaluation latency, kill switch state, and
//! regime distribution.
//!
//! ## Metrics Exposed
//!
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | `janus_brain_pipeline_evaluations_total` | Counter | Total pipeline evaluations |
//! | `janus_brain_pipeline_proceeds_total` | Counter | Evaluations resulting in Proceed |
//! | `janus_brain_pipeline_blocks_total` | CounterVec | Blocks by stage |
//! | `janus_brain_pipeline_reduce_only_total` | Counter | Evaluations resulting in ReduceOnly |
//! | `janus_brain_pipeline_evaluation_duration_us` | Histogram | Evaluation latency in µs |
//! | `janus_brain_pipeline_scale` | Histogram | Position scale factors applied |
//! | `janus_brain_pipeline_kill_switch_active` | Gauge | 1 if kill switch is active |
//! | `janus_brain_pipeline_regime` | GaugeVec | Current regime distribution |
//! | `janus_brain_pipeline_confidence` | Histogram | Regime confidence values |
//! | `janus_brain_pipeline_amygdala_high_risk_total` | Counter | Amygdala high-risk detections |
//! | `janus_brain_pipeline_gate_rejections_total` | Counter | Strategy gate rejections |
//! | `janus_brain_pipeline_correlation_blocks_total` | Counter | Correlation filter blocks |
//! | `janus_brain_watchdog_components_total` | Gauge | Total registered components |
//! | `janus_brain_watchdog_alive_count` | Gauge | Components in Alive state |
//! | `janus_brain_watchdog_degraded_count` | Gauge | Components in Degraded state |
//! | `janus_brain_watchdog_dead_count` | Gauge | Components in Dead state |
//! | `janus_brain_boot_passed` | Gauge | 1 if last boot passed preflight |

use prometheus::{Gauge, Histogram, HistogramOpts, IntCounter, IntCounterVec, Opts, Registry};
use std::sync::Arc;

/// Prometheus metrics collector for the brain-inspired trading pipeline.
pub struct BrainPipelineMetricsCollector {
    // ── Pipeline evaluation counters ────────────────────────────────
    /// Total number of pipeline evaluations.
    pub evaluations_total: IntCounter,

    /// Total evaluations that resulted in `TradeAction::Proceed`.
    pub proceeds_total: IntCounter,

    /// Evaluations blocked, labelled by pipeline stage
    /// (RegimeBridge, Hypothalamus, Amygdala, StrategyGate, CorrelationFilter, KillSwitch).
    pub blocks_total: IntCounterVec,

    /// Total evaluations that resulted in `TradeAction::ReduceOnly`.
    pub reduce_only_total: IntCounter,

    // ── Latency ─────────────────────────────────────────────────────
    /// Pipeline evaluation duration in microseconds.
    pub evaluation_duration_us: Histogram,

    // ── Scale factor ────────────────────────────────────────────────
    /// Distribution of position scale factors returned by the pipeline.
    pub scale_histogram: Histogram,

    // ── Kill switch ─────────────────────────────────────────────────
    /// 1.0 when the pipeline kill switch is active, 0.0 otherwise.
    pub kill_switch_active: Gauge,

    // ── Regime ──────────────────────────────────────────────────────
    /// Current regime distribution gauge (labels: regime).
    /// Value is the count of evaluations in each regime since last reset.
    pub regime_evaluations: IntCounterVec,

    /// Regime confidence distribution.
    pub confidence_histogram: Histogram,

    // ── Sub-stage counters ──────────────────────────────────────────
    /// Amygdala high-risk detections.
    pub amygdala_high_risk_total: IntCounter,

    /// Strategy gate rejections.
    pub gate_rejections_total: IntCounter,

    /// Correlation filter blocks.
    pub correlation_blocks_total: IntCounter,

    // ── Watchdog metrics ────────────────────────────────────────────
    /// Total registered watchdog components.
    pub watchdog_components_total: Gauge,

    /// Components currently in Alive state.
    pub watchdog_alive_count: Gauge,

    /// Components currently in Degraded state.
    pub watchdog_degraded_count: Gauge,

    /// Components currently in Dead state.
    pub watchdog_dead_count: Gauge,

    // ── Boot ────────────────────────────────────────────────────────
    /// 1.0 if the last boot passed preflight checks, 0.0 otherwise.
    pub boot_passed: Gauge,
}

impl BrainPipelineMetricsCollector {
    /// Create a new collector and register all metrics with the given
    /// Prometheus `Registry`.
    pub fn new(registry: Arc<Registry>) -> Result<Self, prometheus::Error> {
        // ── Pipeline evaluation counters ────────────────────────────
        let evaluations_total = IntCounter::with_opts(Opts::new(
            "janus_brain_pipeline_evaluations_total",
            "Total number of brain pipeline evaluations",
        ))?;
        registry.register(Box::new(evaluations_total.clone()))?;

        let proceeds_total = IntCounter::with_opts(Opts::new(
            "janus_brain_pipeline_proceeds_total",
            "Pipeline evaluations resulting in Proceed",
        ))?;
        registry.register(Box::new(proceeds_total.clone()))?;

        let blocks_total = IntCounterVec::new(
            Opts::new(
                "janus_brain_pipeline_blocks_total",
                "Pipeline evaluations blocked, by stage",
            ),
            &["stage"],
        )?;
        registry.register(Box::new(blocks_total.clone()))?;

        let reduce_only_total = IntCounter::with_opts(Opts::new(
            "janus_brain_pipeline_reduce_only_total",
            "Pipeline evaluations resulting in ReduceOnly",
        ))?;
        registry.register(Box::new(reduce_only_total.clone()))?;

        // ── Latency ─────────────────────────────────────────────────
        let evaluation_duration_us = Histogram::with_opts(
            HistogramOpts::new(
                "janus_brain_pipeline_evaluation_duration_us",
                "Pipeline evaluation duration in microseconds",
            )
            .buckets(vec![
                5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0,
            ]),
        )?;
        registry.register(Box::new(evaluation_duration_us.clone()))?;

        // ── Scale ───────────────────────────────────────────────────
        let scale_histogram = Histogram::with_opts(
            HistogramOpts::new(
                "janus_brain_pipeline_scale",
                "Position scale factors from pipeline evaluations",
            )
            .buckets(vec![
                0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0,
            ]),
        )?;
        registry.register(Box::new(scale_histogram.clone()))?;

        // ── Kill switch ─────────────────────────────────────────────
        let kill_switch_active = Gauge::with_opts(Opts::new(
            "janus_brain_pipeline_kill_switch_active",
            "1 when the pipeline kill switch is active",
        ))?;
        registry.register(Box::new(kill_switch_active.clone()))?;

        // ── Regime ──────────────────────────────────────────────────
        let regime_evaluations = IntCounterVec::new(
            Opts::new(
                "janus_brain_pipeline_regime_evaluations_total",
                "Pipeline evaluations by regime type",
            ),
            &["regime"],
        )?;
        registry.register(Box::new(regime_evaluations.clone()))?;

        let confidence_histogram = Histogram::with_opts(
            HistogramOpts::new(
                "janus_brain_pipeline_confidence",
                "Regime confidence values observed during pipeline evaluation",
            )
            .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        )?;
        registry.register(Box::new(confidence_histogram.clone()))?;

        // ── Sub-stage counters ──────────────────────────────────────
        let amygdala_high_risk_total = IntCounter::with_opts(Opts::new(
            "janus_brain_pipeline_amygdala_high_risk_total",
            "Amygdala high-risk threat detections",
        ))?;
        registry.register(Box::new(amygdala_high_risk_total.clone()))?;

        let gate_rejections_total = IntCounter::with_opts(Opts::new(
            "janus_brain_pipeline_gate_rejections_total",
            "Strategy gate rejections",
        ))?;
        registry.register(Box::new(gate_rejections_total.clone()))?;

        let correlation_blocks_total = IntCounter::with_opts(Opts::new(
            "janus_brain_pipeline_correlation_blocks_total",
            "Correlation filter blocks",
        ))?;
        registry.register(Box::new(correlation_blocks_total.clone()))?;

        // ── Watchdog ────────────────────────────────────────────────
        let watchdog_components_total = Gauge::with_opts(Opts::new(
            "janus_brain_watchdog_components_total",
            "Total registered watchdog components",
        ))?;
        registry.register(Box::new(watchdog_components_total.clone()))?;

        let watchdog_alive_count = Gauge::with_opts(Opts::new(
            "janus_brain_watchdog_alive_count",
            "Watchdog components in Alive state",
        ))?;
        registry.register(Box::new(watchdog_alive_count.clone()))?;

        let watchdog_degraded_count = Gauge::with_opts(Opts::new(
            "janus_brain_watchdog_degraded_count",
            "Watchdog components in Degraded state",
        ))?;
        registry.register(Box::new(watchdog_degraded_count.clone()))?;

        let watchdog_dead_count = Gauge::with_opts(Opts::new(
            "janus_brain_watchdog_dead_count",
            "Watchdog components in Dead state",
        ))?;
        registry.register(Box::new(watchdog_dead_count.clone()))?;

        // ── Boot ────────────────────────────────────────────────────
        let boot_passed = Gauge::with_opts(Opts::new(
            "janus_brain_boot_passed",
            "1 if the last boot passed preflight checks",
        ))?;
        registry.register(Box::new(boot_passed.clone()))?;

        Ok(Self {
            evaluations_total,
            proceeds_total,
            blocks_total,
            reduce_only_total,
            evaluation_duration_us,
            scale_histogram,
            kill_switch_active,
            regime_evaluations,
            confidence_histogram,
            amygdala_high_risk_total,
            gate_rejections_total,
            correlation_blocks_total,
            watchdog_components_total,
            watchdog_alive_count,
            watchdog_degraded_count,
            watchdog_dead_count,
            boot_passed,
        })
    }

    // ────────────────────────────────────────────────────────────────
    // Convenience recording helpers
    // ────────────────────────────────────────────────────────────────

    /// Record the result of a single pipeline evaluation.
    ///
    /// * `action` — "proceed", "block", or "reduce_only"
    /// * `stage` — the pipeline stage label (only meaningful for blocks)
    /// * `scale` — the position scale factor (0.0 for blocks)
    /// * `evaluation_us` — wall-clock evaluation time in µs
    /// * `regime` — the regime label string (e.g. "Trending", "MeanReverting", "Volatile", "Uncertain")
    /// * `confidence` — the regime confidence [0.0, 1.0]
    /// * `amygdala_high_risk` — whether the amygdala flagged high risk
    /// * `gate_approved` — whether the strategy gate approved the trade
    /// * `correlation_passed` — whether the correlation filter passed
    #[allow(clippy::too_many_arguments)]
    pub fn record_evaluation(
        &self,
        action: &str,
        stage: &str,
        scale: f64,
        evaluation_us: u64,
        regime: &str,
        confidence: f64,
        amygdala_high_risk: bool,
        gate_approved: bool,
        correlation_passed: bool,
    ) {
        self.evaluations_total.inc();
        self.evaluation_duration_us.observe(evaluation_us as f64);
        self.confidence_histogram.observe(confidence);
        self.regime_evaluations.with_label_values(&[regime]).inc();

        match action {
            "proceed" => {
                self.proceeds_total.inc();
                self.scale_histogram.observe(scale);
            }
            "block" => {
                self.blocks_total.with_label_values(&[stage]).inc();
            }
            "reduce_only" => {
                self.reduce_only_total.inc();
                self.scale_histogram.observe(scale);
            }
            _ => {}
        }

        if amygdala_high_risk {
            self.amygdala_high_risk_total.inc();
        }

        if !gate_approved {
            self.gate_rejections_total.inc();
        }

        if !correlation_passed {
            self.correlation_blocks_total.inc();
        }
    }

    /// Record a pipeline evaluation directly from a `TradingDecision`.
    ///
    /// This is the preferred method — just pass the decision and we extract
    /// all the labels ourselves.
    pub fn record_decision(&self, decision: &crate::brain_wiring::TradingDecision) {
        let (action_str, stage_str, scale) = match &decision.action {
            crate::brain_wiring::TradeAction::Proceed { scale } => ("proceed", "", *scale),
            crate::brain_wiring::TradeAction::Block { stage, .. } => {
                ("block", stage_label(stage), 0.0)
            }
            crate::brain_wiring::TradeAction::ReduceOnly { scale, .. } => {
                ("reduce_only", "", *scale)
            }
        };

        let regime = regime_label(&decision.bridged_regime);

        self.record_evaluation(
            action_str,
            stage_str,
            scale,
            decision.evaluation_us,
            &regime,
            decision.bridged_regime.confidence,
            decision.amygdala_high_risk,
            decision.gate_approved,
            decision.correlation_passed,
        );
    }

    /// Update kill-switch gauge.
    pub fn set_kill_switch(&self, active: bool) {
        self.kill_switch_active.set(if active { 1.0 } else { 0.0 });
    }

    /// Update watchdog component gauges from a snapshot.
    pub fn update_watchdog_snapshot(
        &self,
        total: usize,
        alive: usize,
        degraded: usize,
        dead: usize,
    ) {
        self.watchdog_components_total.set(total as f64);
        self.watchdog_alive_count.set(alive as f64);
        self.watchdog_degraded_count.set(degraded as f64);
        self.watchdog_dead_count.set(dead as f64);
    }

    /// Update boot-passed gauge.
    pub fn set_boot_passed(&self, passed: bool) {
        self.boot_passed.set(if passed { 1.0 } else { 0.0 });
    }
}

// ════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════

/// Map a `PipelineStage` to a static label string for Prometheus.
fn stage_label(stage: &crate::brain_wiring::PipelineStage) -> &'static str {
    use crate::brain_wiring::PipelineStage;
    match stage {
        PipelineStage::RegimeBridge => "RegimeBridge",
        PipelineStage::Hypothalamus => "Hypothalamus",
        PipelineStage::Amygdala => "Amygdala",
        PipelineStage::StrategyGate => "StrategyGate",
        PipelineStage::CorrelationFilter => "CorrelationFilter",
        PipelineStage::KillSwitch => "KillSwitch",
    }
}

/// Extract a regime label string from a `BridgedRegimeState`.
///
/// Uses the `hypothalamus_regime` field since `BridgedRegimeState` does not
/// have a top-level `regime` field — the hypothalamus regime is the best
/// proxy for the overall market regime classification.
fn regime_label(bridged: &crate::regime_bridge::BridgedRegimeState) -> String {
    format!("{}", bridged.hypothalamus_regime)
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> Arc<Registry> {
        Arc::new(Registry::new())
    }

    #[test]
    fn test_collector_creation() {
        let registry = make_registry();
        let collector = BrainPipelineMetricsCollector::new(registry);
        assert!(collector.is_ok());
    }

    #[test]
    fn test_record_proceed() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        c.record_evaluation(
            "proceed", "", 0.85, 120, "Trending", 0.92, false, true, true,
        );

        assert_eq!(c.evaluations_total.get(), 1);
        assert_eq!(c.proceeds_total.get(), 1);
        assert_eq!(c.reduce_only_total.get(), 0);
    }

    #[test]
    fn test_record_block() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        c.record_evaluation(
            "block",
            "RegimeBridge",
            0.0,
            50,
            "Uncertain",
            0.3,
            false,
            true,
            true,
        );

        assert_eq!(c.evaluations_total.get(), 1);
        assert_eq!(c.proceeds_total.get(), 0);
        assert_eq!(c.blocks_total.with_label_values(&["RegimeBridge"]).get(), 1);
    }

    #[test]
    fn test_record_reduce_only() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        c.record_evaluation(
            "reduce_only",
            "",
            0.5,
            200,
            "Volatile",
            0.75,
            true,
            true,
            true,
        );

        assert_eq!(c.evaluations_total.get(), 1);
        assert_eq!(c.reduce_only_total.get(), 1);
        assert_eq!(c.amygdala_high_risk_total.get(), 1);
    }

    #[test]
    fn test_kill_switch_gauge() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        c.set_kill_switch(false);
        assert_eq!(c.kill_switch_active.get(), 0.0);

        c.set_kill_switch(true);
        assert_eq!(c.kill_switch_active.get(), 1.0);
    }

    #[test]
    fn test_watchdog_snapshot_update() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        c.update_watchdog_snapshot(5, 3, 1, 1);

        assert_eq!(c.watchdog_components_total.get(), 5.0);
        assert_eq!(c.watchdog_alive_count.get(), 3.0);
        assert_eq!(c.watchdog_degraded_count.get(), 1.0);
        assert_eq!(c.watchdog_dead_count.get(), 1.0);
    }

    #[test]
    fn test_boot_passed_gauge() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        c.set_boot_passed(true);
        assert_eq!(c.boot_passed.get(), 1.0);

        c.set_boot_passed(false);
        assert_eq!(c.boot_passed.get(), 0.0);
    }

    #[test]
    fn test_gate_rejection_counted() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        // gate_approved = false should increment gate_rejections_total
        c.record_evaluation(
            "block",
            "StrategyGate",
            0.0,
            80,
            "MeanReverting",
            0.65,
            false,
            false,
            true,
        );

        assert_eq!(c.gate_rejections_total.get(), 1);
    }

    #[test]
    fn test_correlation_block_counted() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        c.record_evaluation(
            "block",
            "CorrelationFilter",
            0.0,
            90,
            "Trending",
            0.80,
            false,
            true,
            false,
        );

        assert_eq!(c.correlation_blocks_total.get(), 1);
    }

    #[test]
    fn test_multiple_evaluations_accumulate() {
        let registry = make_registry();
        let c = BrainPipelineMetricsCollector::new(registry).unwrap();

        for _ in 0..10 {
            c.record_evaluation("proceed", "", 1.0, 100, "Trending", 0.95, false, true, true);
        }

        for _ in 0..5 {
            c.record_evaluation(
                "block",
                "KillSwitch",
                0.0,
                10,
                "Uncertain",
                0.2,
                false,
                true,
                true,
            );
        }

        assert_eq!(c.evaluations_total.get(), 15);
        assert_eq!(c.proceeds_total.get(), 10);
        assert_eq!(c.blocks_total.with_label_values(&["KillSwitch"]).get(), 5);
    }

    #[test]
    fn test_duplicate_registry_fails() {
        let registry = make_registry();
        let _first = BrainPipelineMetricsCollector::new(Arc::clone(&registry)).unwrap();
        // Registering the same metrics a second time must fail
        let second = BrainPipelineMetricsCollector::new(Arc::clone(&registry));
        assert!(second.is_err());
    }
}
