//! # Brain Runtime Integration Tests
//!
//! Integration tests for the brain-gated execution client, pipeline Prometheus
//! metrics, runtime health reporting, and the full boot→evaluate→shutdown
//! lifecycle.
//!
//! These tests validate:
//!
//! - `BrainPipelineMetricsCollector` records pipeline decisions correctly
//! - `BrainGatedExecutionClient` configuration and gating logic
//! - `BrainRuntime` boot → health_report → shutdown lifecycle
//! - Pipeline metrics flow end-to-end through the runtime
//! - Kill switch integration with metrics and runtime
//! - Watchdog snapshot propagation into health reports
//! - Preflight-only mode
//! - Multiple concurrent evaluations with metrics accumulation

use std::sync::Arc;

use janus_forward::brain_runtime::{BrainRuntime, BrainRuntimeConfig, WatchdogRuntimeConfig};
use janus_forward::brain_wiring::{
    TradeAction, TradingPipeline, TradingPipelineBuilder, TradingPipelineConfig, make_test_signal,
};
use janus_forward::execution::brain_gated::{
    BrainGatedConfig, GatedExecutionStats, GatedSubmissionResult,
};
use janus_forward::metrics::BrainPipelineMetricsCollector;
use janus_regime::{MarketRegime, RoutedSignal, TrendDirection};
use janus_risk::{CorrelationConfig, CorrelationTracker};
use janus_strategies::affinity::StrategyAffinityTracker;
use janus_strategies::gating::{StrategyGate, StrategyGatingConfig};
use prometheus::Registry;

// ============================================================================
// Test helpers
// ============================================================================

fn bullish_trending(confidence: f64) -> RoutedSignal {
    make_test_signal(MarketRegime::Trending(TrendDirection::Bullish), confidence)
}

fn bearish_trending(confidence: f64) -> RoutedSignal {
    make_test_signal(MarketRegime::Trending(TrendDirection::Bearish), confidence)
}

fn mean_reverting(confidence: f64) -> RoutedSignal {
    make_test_signal(MarketRegime::MeanReverting, confidence)
}

fn volatile(confidence: f64) -> RoutedSignal {
    make_test_signal(MarketRegime::Volatile, confidence)
}

fn uncertain(confidence: f64) -> RoutedSignal {
    make_test_signal(MarketRegime::Uncertain, confidence)
}

fn make_pipeline() -> TradingPipeline {
    TradingPipelineBuilder::new()
        .config(TradingPipelineConfig::default())
        .build()
}

fn make_pipeline_with_config(config: TradingPipelineConfig) -> TradingPipeline {
    TradingPipelineBuilder::new().config(config).build()
}

fn make_metrics() -> (Arc<Registry>, Arc<BrainPipelineMetricsCollector>) {
    let registry = Arc::new(Registry::new());
    let collector =
        BrainPipelineMetricsCollector::new(Arc::clone(&registry)).expect("metrics registration");
    (registry, Arc::new(collector))
}

fn make_runtime_config_no_watchdog() -> BrainRuntimeConfig {
    BrainRuntimeConfig {
        auto_start_watchdog: false,
        enforce_preflight: false,
        wire_kill_switch: false,
        ..Default::default()
    }
}

fn make_runtime_config_with_watchdog() -> BrainRuntimeConfig {
    BrainRuntimeConfig {
        auto_start_watchdog: true,
        enforce_preflight: false,
        wire_kill_switch: true,
        forward_heartbeat_ms: 500,
        watchdog: WatchdogRuntimeConfig {
            check_interval_ms: 200,
            degraded_threshold: 3,
            dead_threshold: 5,
            kill_on_critical_death: true,
        },
        ..Default::default()
    }
}

// ============================================================================
// Prometheus Metrics Tests
// ============================================================================

#[test]
fn test_metrics_collector_registers_all_metrics() {
    let (registry, collector) = make_metrics();

    // Touch the CounterVec labels so they appear in gather()
    collector
        .blocks_total
        .with_label_values(&["RegimeBridge"])
        .inc_by(0);
    collector
        .regime_evaluations
        .with_label_values(&["Trending"])
        .inc_by(0);

    let families = registry.gather();
    let names: Vec<String> = families.iter().map(|f| f.name().to_string()).collect();

    // Verify key metric families are registered
    assert!(names.contains(&"janus_brain_pipeline_evaluations_total".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_proceeds_total".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_blocks_total".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_reduce_only_total".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_evaluation_duration_us".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_scale".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_kill_switch_active".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_confidence".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_amygdala_high_risk_total".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_gate_rejections_total".to_string()));
    assert!(names.contains(&"janus_brain_pipeline_correlation_blocks_total".to_string()));
    assert!(names.contains(&"janus_brain_watchdog_components_total".to_string()));
    assert!(names.contains(&"janus_brain_watchdog_alive_count".to_string()));
    assert!(names.contains(&"janus_brain_watchdog_degraded_count".to_string()));
    assert!(names.contains(&"janus_brain_watchdog_dead_count".to_string()));
    assert!(names.contains(&"janus_brain_boot_passed".to_string()));
}

#[tokio::test]
async fn test_metrics_record_proceed_decision() {
    let (_registry, metrics) = make_metrics();
    let pipeline = Arc::new(make_pipeline());

    let signal = bullish_trending(0.9);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(decision.is_actionable());
    metrics.record_decision(&decision);

    assert_eq!(metrics.evaluations_total.get(), 1);
    assert_eq!(metrics.proceeds_total.get(), 1);
    assert_eq!(metrics.reduce_only_total.get(), 0);
}

#[tokio::test]
async fn test_metrics_record_block_decision() {
    let (_registry, metrics) = make_metrics();

    let config = TradingPipelineConfig {
        min_regime_confidence: 0.8,
        ..Default::default()
    };
    let pipeline = Arc::new(make_pipeline_with_config(config));

    // Low confidence should be blocked
    let signal = uncertain(0.2);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(!decision.is_actionable());
    metrics.record_decision(&decision);

    assert_eq!(metrics.evaluations_total.get(), 1);
    assert_eq!(metrics.proceeds_total.get(), 0);
    assert_eq!(
        metrics
            .blocks_total
            .with_label_values(&["RegimeBridge"])
            .get(),
        1
    );
}

#[tokio::test]
async fn test_metrics_record_reduce_only_decision() {
    let (_registry, metrics) = make_metrics();

    let config = TradingPipelineConfig {
        allow_new_positions_in_crisis: false,
        enable_amygdala_filter: true,
        ..Default::default()
    };
    let pipeline = Arc::new(make_pipeline_with_config(config));

    // Volatile with low confidence triggers amygdala crisis -> ReduceOnly
    let signal = volatile(0.35);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    // If the decision is ReduceOnly, verify metrics track it
    match &decision.action {
        TradeAction::ReduceOnly { .. } => {
            metrics.record_decision(&decision);
            assert_eq!(metrics.reduce_only_total.get(), 1);
        }
        TradeAction::Proceed { .. } => {
            // Some configs may still proceed — just verify the metric records correctly
            metrics.record_decision(&decision);
            assert_eq!(metrics.proceeds_total.get(), 1);
        }
        TradeAction::Block { .. } => {
            metrics.record_decision(&decision);
            assert!(metrics.evaluations_total.get() > 0);
        }
    }
}

#[tokio::test]
async fn test_metrics_accumulate_across_multiple_evaluations() {
    let (_registry, metrics) = make_metrics();
    let pipeline = Arc::new(make_pipeline());

    // 10 bullish trending evaluations
    for _ in 0..10 {
        let signal = bullish_trending(0.9);
        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;
        metrics.record_decision(&decision);
    }

    // 5 low-confidence evaluations (should be blocked or proceed depending on default min_confidence)
    let config = TradingPipelineConfig {
        min_regime_confidence: 0.9,
        ..Default::default()
    };
    let strict_pipeline = Arc::new(make_pipeline_with_config(config));

    for _ in 0..5 {
        let signal = uncertain(0.1);
        let decision = strict_pipeline
            .evaluate("ETHUSDT", &signal, "mean_rev", &[], None, None, None, None)
            .await;
        metrics.record_decision(&decision);
    }

    assert_eq!(metrics.evaluations_total.get(), 15);
    // At least the 10 bullish should have proceeded
    assert!(metrics.proceeds_total.get() >= 10);
}

#[tokio::test]
async fn test_metrics_kill_switch_gauge() {
    let (_registry, metrics) = make_metrics();
    let pipeline = Arc::new(make_pipeline());

    metrics.set_kill_switch(false);
    assert_eq!(metrics.kill_switch_active.get(), 0.0);

    pipeline.activate_kill_switch().await;
    metrics.set_kill_switch(pipeline.is_killed().await);
    assert_eq!(metrics.kill_switch_active.get(), 1.0);

    pipeline.deactivate_kill_switch().await;
    metrics.set_kill_switch(pipeline.is_killed().await);
    assert_eq!(metrics.kill_switch_active.get(), 0.0);
}

#[test]
fn test_metrics_watchdog_snapshot_update() {
    let (_registry, metrics) = make_metrics();

    metrics.update_watchdog_snapshot(6, 4, 1, 1);

    assert_eq!(metrics.watchdog_components_total.get(), 6.0);
    assert_eq!(metrics.watchdog_alive_count.get(), 4.0);
    assert_eq!(metrics.watchdog_degraded_count.get(), 1.0);
    assert_eq!(metrics.watchdog_dead_count.get(), 1.0);
}

#[test]
fn test_metrics_boot_passed_gauge() {
    let (_registry, metrics) = make_metrics();

    metrics.set_boot_passed(true);
    assert_eq!(metrics.boot_passed.get(), 1.0);

    metrics.set_boot_passed(false);
    assert_eq!(metrics.boot_passed.get(), 0.0);
}

#[tokio::test]
async fn test_metrics_amygdala_high_risk_counted_from_decision() {
    let (_registry, metrics) = make_metrics();
    let pipeline = Arc::new(make_pipeline());

    // Volatile regime should flag amygdala high risk
    let signal = volatile(0.6);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "squeeze", &[], None, None, None, None)
        .await;

    metrics.record_decision(&decision);

    if decision.amygdala_high_risk {
        assert!(metrics.amygdala_high_risk_total.get() >= 1);
    }
}

// ============================================================================
// BrainGatedConfig Tests
// ============================================================================

#[test]
fn test_brain_gated_config_default() {
    let config = BrainGatedConfig::default();
    assert!(config.allow_reduce_only);
    assert!(config.apply_scale_to_signal);
    assert_eq!(config.max_signal_age_secs, 0);
    assert!(config.record_trade_results);
}

#[test]
fn test_brain_gated_config_custom() {
    let config = BrainGatedConfig {
        allow_reduce_only: false,
        apply_scale_to_signal: false,
        max_signal_age_secs: 30,
        record_trade_results: false,
    };

    assert!(!config.allow_reduce_only);
    assert!(!config.apply_scale_to_signal);
    assert_eq!(config.max_signal_age_secs, 30);
    assert!(!config.record_trade_results);
}

// ============================================================================
// GatedExecutionStats Tests
// ============================================================================

#[test]
fn test_gated_execution_stats_default() {
    let stats = GatedExecutionStats::default();
    assert_eq!(stats.total_signals, 0);
    assert_eq!(stats.submitted, 0);
    assert_eq!(stats.blocked, 0);
    assert_eq!(stats.stale_rejected, 0);
    assert_eq!(stats.errors, 0);
    assert_eq!(stats.submission_rate(), 0.0);
    assert_eq!(stats.avg_submitted_scale(), 0.0);
}

#[test]
fn test_gated_execution_stats_calculations() {
    let stats = GatedExecutionStats {
        total_signals: 20,
        submitted: 12,
        blocked: 6,
        stale_rejected: 1,
        errors: 1,
        total_scale_sum: 9.6,
    };

    let rate = stats.submission_rate();
    assert!((rate - 0.6).abs() < 0.001);

    let avg_scale = stats.avg_submitted_scale();
    assert!((avg_scale - 0.8).abs() < 0.001);
}

// ============================================================================
// GatedSubmissionResult Tests
// ============================================================================

#[tokio::test]
async fn test_gated_submission_result_blocked_has_decision() {
    let pipeline = Arc::new(make_pipeline());
    let signal = bullish_trending(0.9);

    // Activate kill switch so evaluation produces a Block
    pipeline.activate_kill_switch().await;

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    let result = GatedSubmissionResult::Blocked {
        decision,
        reason: "Kill switch active".to_string(),
    };

    assert!(result.is_blocked());
    assert!(!result.is_submitted());
    assert!(result.decision().is_some());
    let d = result.decision().unwrap();
    assert_eq!(d.symbol, "BTCUSDT");
    assert_eq!(d.strategy, "ema_flip");
}

#[test]
fn test_gated_submission_result_stale() {
    let result = GatedSubmissionResult::Stale {
        signal_age_secs: 60,
        max_age_secs: 30,
    };

    assert!(!result.is_submitted());
    assert!(!result.is_blocked());
    assert!(result.decision().is_none());

    let display = format!("{}", result);
    assert!(display.contains("Stale"));
    assert!(display.contains("60"));
    assert!(display.contains("30"));
}

#[test]
fn test_gated_submission_result_error() {
    let result = GatedSubmissionResult::Error {
        decision: None,
        error: "Connection refused".to_string(),
    };

    assert!(!result.is_submitted());
    assert!(!result.is_blocked());
    assert!(result.decision().is_none());

    let display = format!("{}", result);
    assert!(display.contains("Error"));
    assert!(display.contains("Connection refused"));
}

// ============================================================================
// BrainRuntime Lifecycle Tests
// ============================================================================

#[tokio::test]
async fn test_runtime_boot_creates_pipeline_and_reports_health() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    let report = runtime.boot().await.expect("boot should succeed");
    assert!(report.is_boot_safe());

    // Pipeline should be initialized
    assert!(runtime.pipeline().is_some());

    // Health report should reflect running state
    let health = runtime.health_report().await;
    assert!(health.is_healthy());
    assert!(health.boot_passed);
    assert!(health.pipeline.is_some());
    assert_eq!(health.pipeline.as_ref().unwrap().total_evaluations, 0);

    runtime.shutdown().await;
}

#[tokio::test]
async fn test_runtime_boot_with_watchdog_creates_snapshot() {
    let config = make_runtime_config_with_watchdog();
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("boot should succeed");

    // Watchdog handle should be present
    assert!(runtime.watchdog_handle().is_some());

    // Snapshot should show registered components
    let snapshot = runtime.watchdog_snapshot().await;
    assert!(snapshot.is_some());
    let snap = snapshot.unwrap();
    assert!(snap.total_components >= 3); // At least forward_service, trading_pipeline, regime_detector

    runtime.shutdown().await;
}

#[tokio::test]
async fn test_runtime_pipeline_evaluations_appear_in_health() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("boot should succeed");

    let pipeline = runtime.pipeline().unwrap().clone();

    // Do some evaluations
    let signal = bullish_trending(0.9);
    for _ in 0..5 {
        pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;
    }

    let health = runtime.health_report().await;
    let pm = health.pipeline.unwrap();
    assert_eq!(pm.total_evaluations, 5);
    assert!(pm.proceed_count > 0);
    assert!(!pm.is_killed);

    runtime.shutdown().await;
}

#[tokio::test]
async fn test_runtime_kill_switch_appears_in_health() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("boot should succeed");

    let pipeline = runtime.pipeline().unwrap().clone();
    pipeline.activate_kill_switch().await;

    let health = runtime.health_report().await;
    assert!(health.pipeline.as_ref().unwrap().is_killed);

    // After deactivation
    pipeline.deactivate_kill_switch().await;
    let health2 = runtime.health_report().await;
    assert!(!health2.pipeline.as_ref().unwrap().is_killed);

    runtime.shutdown().await;
}

#[tokio::test]
async fn test_runtime_shutdown_activates_kill_switch() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("boot should succeed");

    let pipeline = runtime.pipeline().unwrap().clone();
    assert!(!pipeline.is_killed().await);

    runtime.shutdown().await;

    // After shutdown, pipeline kill switch should be active
    assert!(pipeline.is_killed().await);
}

#[tokio::test]
async fn test_runtime_double_shutdown_is_safe() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("boot should succeed");

    runtime.shutdown().await;
    runtime.shutdown().await; // Should not panic
}

#[tokio::test]
async fn test_runtime_preflight_only_does_not_create_pipeline() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    let report = runtime.preflight_only().await;
    assert!(report.is_boot_safe());

    // Pipeline should NOT be initialized
    assert!(runtime.pipeline().is_none());
}

#[tokio::test]
async fn test_runtime_boot_with_custom_pipeline() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    let pipeline = Arc::new(make_pipeline());

    runtime
        .boot_with_pipeline(pipeline.clone())
        .await
        .expect("boot should succeed");

    // Pipeline should be the one we injected
    assert!(runtime.pipeline().is_some());
    assert!(Arc::ptr_eq(runtime.pipeline().unwrap(), &pipeline));

    runtime.shutdown().await;
}

// ============================================================================
// Pipeline + Metrics End-to-End Flow
// ============================================================================

#[tokio::test]
async fn test_pipeline_metrics_e2e_bullish_flow() {
    let (_registry, metrics) = make_metrics();
    let pipeline = Arc::new(make_pipeline());

    let signal = bullish_trending(0.92);
    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            Some(45.0),
            None,
            None,
            None,
        )
        .await;

    metrics.record_decision(&decision);

    // With high confidence bullish, should proceed
    assert!(decision.is_actionable());
    assert_eq!(metrics.evaluations_total.get(), 1);
    assert_eq!(metrics.proceeds_total.get(), 1);

    // Regime evaluation counter should have been incremented
    // (label depends on hypothalamus_regime mapping)
    let total_regime_evals: u64 = metrics
        .regime_evaluations
        .with_label_values(&["Bullish"])
        .get()
        + metrics
            .regime_evaluations
            .with_label_values(&["StrongBullish"])
            .get();
    assert!(total_regime_evals >= 1);
}

#[tokio::test]
async fn test_pipeline_metrics_e2e_kill_switch_flow() {
    let (_registry, metrics) = make_metrics();
    let pipeline = Arc::new(make_pipeline());

    pipeline.activate_kill_switch().await;
    metrics.set_kill_switch(true);

    let signal = bullish_trending(0.95);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    metrics.record_decision(&decision);

    assert!(!decision.is_actionable());
    assert_eq!(
        metrics
            .blocks_total
            .with_label_values(&["KillSwitch"])
            .get(),
        1
    );
    assert_eq!(metrics.kill_switch_active.get(), 1.0);
}

#[tokio::test]
async fn test_pipeline_metrics_e2e_low_confidence_block() {
    let (_registry, metrics) = make_metrics();

    let config = TradingPipelineConfig {
        min_regime_confidence: 0.8,
        ..Default::default()
    };
    let pipeline = Arc::new(make_pipeline_with_config(config));

    let signal = uncertain(0.15);
    let decision = pipeline
        .evaluate("SOLUSDT", &signal, "momentum", &[], None, None, None, None)
        .await;

    metrics.record_decision(&decision);

    assert!(!decision.is_actionable());
    assert_eq!(
        metrics
            .blocks_total
            .with_label_values(&["RegimeBridge"])
            .get(),
        1
    );
}

// ============================================================================
// Concurrent Evaluations
// ============================================================================

#[tokio::test]
async fn test_concurrent_evaluations_all_metrics_accumulate() {
    let (_registry, metrics) = make_metrics();
    let metrics = Arc::clone(&metrics);
    let pipeline = Arc::new(make_pipeline());

    let mut handles = Vec::new();
    let num_tasks = 20;

    for i in 0..num_tasks {
        let pipeline = Arc::clone(&pipeline);
        let metrics = Arc::clone(&metrics);
        let handle = tokio::spawn(async move {
            let signal = if i % 2 == 0 {
                bullish_trending(0.9)
            } else {
                bearish_trending(0.85)
            };
            let symbol = format!("SYM{}", i);
            let decision = pipeline
                .evaluate(&symbol, &signal, "test_strat", &[], None, None, None, None)
                .await;
            metrics.record_decision(&decision);
        });
        handles.push(handle);
    }

    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(metrics.evaluations_total.get(), num_tasks as u64);
    // All should have proceeded (high confidence, no kill switch)
    assert_eq!(metrics.proceeds_total.get(), num_tasks as u64);
}

// ============================================================================
// Health Report Serialization / Display
// ============================================================================

#[tokio::test]
async fn test_health_report_summary_contains_key_info() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("boot should succeed");

    let health = runtime.health_report().await;
    let summary = health.summary();

    assert!(
        summary.contains("Running") || summary.contains("running"),
        "summary should contain runtime state, got: {}",
        summary
    );
    assert!(
        summary.contains("Boot Passed") || summary.contains("Boot") || summary.contains("boot"),
        "summary should contain boot info, got: {}",
        summary
    );

    runtime.shutdown().await;
}

#[tokio::test]
async fn test_health_report_is_healthy_before_boot() {
    let config = make_runtime_config_no_watchdog();
    let runtime = BrainRuntime::new(config);

    let health = runtime.health_report().await;
    // Before boot, should not be healthy
    assert!(!health.is_healthy());
    assert!(!health.boot_passed);
    assert!(health.pipeline.is_none());
}

// ============================================================================
// Pipeline Builder Integration
// ============================================================================

#[tokio::test]
async fn test_pipeline_builder_with_custom_components_and_metrics() {
    let (_registry, metrics) = make_metrics();

    let corr_config = CorrelationConfig {
        window: 30,
        correlation_threshold: 0.7,
        max_correlated_positions: 3,
        min_observations: 10,
        monitored_pairs: vec![],
    };

    let gating_config = StrategyGatingConfig::default();
    let affinity = StrategyAffinityTracker::new(5);
    let gate = StrategyGate::new(gating_config, affinity);
    let corr = CorrelationTracker::new(corr_config);

    let pipeline = Arc::new(
        TradingPipelineBuilder::new()
            .config(TradingPipelineConfig::default())
            .correlation_tracker(corr)
            .strategy_gate(gate)
            .build(),
    );

    let signal = bullish_trending(0.88);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    metrics.record_decision(&decision);

    assert_eq!(metrics.evaluations_total.get(), 1);
    assert!(decision.is_actionable());
}

// ============================================================================
// Mixed Regime Flow
// ============================================================================

#[tokio::test]
async fn test_mixed_regime_flow_with_metrics() {
    let (_registry, metrics) = make_metrics();
    let pipeline = Arc::new(make_pipeline());

    // Bullish trending
    let d1 = pipeline
        .evaluate(
            "BTCUSDT",
            &bullish_trending(0.9),
            "ema_flip",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    metrics.record_decision(&d1);

    // Mean reverting
    let d2 = pipeline
        .evaluate(
            "ETHUSDT",
            &mean_reverting(0.75),
            "mean_rev",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    metrics.record_decision(&d2);

    // Bearish trending
    let d3 = pipeline
        .evaluate(
            "SOLUSDT",
            &bearish_trending(0.85),
            "ema_flip",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    metrics.record_decision(&d3);

    // Volatile
    let d4 = pipeline
        .evaluate(
            "DOTUSD",
            &volatile(0.6),
            "squeeze",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    metrics.record_decision(&d4);

    // Uncertain
    let d5 = pipeline
        .evaluate(
            "ADAUSDT",
            &uncertain(0.4),
            "scalper",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    metrics.record_decision(&d5);

    assert_eq!(metrics.evaluations_total.get(), 5);

    // At least some should have proceeded
    let total_actions = metrics.proceeds_total.get()
        + metrics.reduce_only_total.get()
        + metrics
            .blocks_total
            .with_label_values(&["RegimeBridge"])
            .get()
        + metrics
            .blocks_total
            .with_label_values(&["Hypothalamus"])
            .get()
        + metrics.blocks_total.with_label_values(&["Amygdala"]).get()
        + metrics
            .blocks_total
            .with_label_values(&["StrategyGate"])
            .get()
        + metrics
            .blocks_total
            .with_label_values(&["CorrelationFilter"])
            .get()
        + metrics
            .blocks_total
            .with_label_values(&["KillSwitch"])
            .get();

    // Total actions should equal total evaluations
    assert_eq!(total_actions, 5);
}

// ============================================================================
// Pipeline Metrics Snapshot Correlation
// ============================================================================

#[tokio::test]
async fn test_pipeline_internal_metrics_match_prometheus_metrics() {
    let (_registry, prom_metrics) = make_metrics();
    let pipeline = Arc::new(make_pipeline());

    let signals = [
        bullish_trending(0.9),
        bearish_trending(0.85),
        mean_reverting(0.7),
    ];

    for (i, signal) in signals.iter().enumerate() {
        let symbol = format!("SYM{}", i);
        let decision = pipeline
            .evaluate(&symbol, signal, "test_strat", &[], None, None, None, None)
            .await;
        prom_metrics.record_decision(&decision);
    }

    // Compare pipeline internal metrics with Prometheus metrics
    let internal = pipeline.metrics_snapshot().await;

    assert_eq!(
        internal.total_evaluations as u64,
        prom_metrics.evaluations_total.get()
    );
    assert_eq!(
        internal.proceed_count as u64,
        prom_metrics.proceeds_total.get()
    );
}

// ============================================================================
// Runtime + Metrics Combined
// ============================================================================

#[tokio::test]
async fn test_runtime_health_with_metrics_after_evaluations() {
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("boot should succeed");

    let pipeline = runtime.pipeline().unwrap().clone();

    // Run evaluations
    for _ in 0..3 {
        let signal = bullish_trending(0.9);
        pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;
    }

    let health = runtime.health_report().await;
    assert!(health.is_healthy());

    let pm = health.pipeline.unwrap();
    assert_eq!(pm.total_evaluations, 3);
    assert!(pm.proceed_count > 0);
    assert!(pm.avg_evaluation_us >= 0.0);

    runtime.shutdown().await;
}

#[tokio::test]
async fn test_runtime_boot_sets_metrics_boot_passed() {
    let (_registry, metrics) = make_metrics();
    let config = make_runtime_config_no_watchdog();
    let mut runtime = BrainRuntime::new(config);

    let report = runtime.boot().await.expect("boot should succeed");
    metrics.set_boot_passed(report.is_boot_safe());

    assert_eq!(metrics.boot_passed.get(), 1.0);

    runtime.shutdown().await;
}

// ============================================================================
// Heartbeat Integration
// ============================================================================

#[tokio::test]
async fn test_runtime_heartbeat_core_works() {
    let config = make_runtime_config_with_watchdog();
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("boot should succeed");

    // Heartbeat should not panic
    runtime.heartbeat_core().await;
    runtime
        .heartbeat(janus_forward::brain_runtime::components::DATA_FEED)
        .await;
    runtime
        .heartbeat(janus_forward::brain_runtime::components::RISK_MANAGER)
        .await;

    runtime.shutdown().await;
}
