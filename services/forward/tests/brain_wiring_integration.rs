//! # Brain Wiring Integration Tests
//!
//! End-to-end tests for the Phase 5 brain-inspired trading pipeline:
//!
//! regime → bridge → hypothalamus → amygdala → gating → correlation → execution
//!
//! These tests validate the full pipeline behavior using in-memory components
//! (no external services required).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use janus_forward::brain_runtime::{BrainRuntime, BrainRuntimeConfig, WatchdogRuntimeConfig};
use janus_forward::brain_wiring::{
    PipelineStage, TradeAction, TradingDecision, TradingPipeline, TradingPipelineBuilder,
    TradingPipelineConfig, make_test_signal,
};
use janus_forward::regime_bridge::{AmygdalaRegime, HypothalamusRegime, bridge_regime_signal};

use janus_regime::{ActiveStrategy, DetectionMethod, MarketRegime, RoutedSignal, TrendDirection};
use janus_risk::{CorrelationConfig, CorrelationTracker};
use janus_strategies::affinity::StrategyAffinityTracker;
use janus_strategies::gating::{AssetStrategyConfig, StrategyGate, StrategyGatingConfig};

// ============================================================================
// Test helpers
// ============================================================================

fn bullish_trending(confidence: f64) -> RoutedSignal {
    RoutedSignal {
        regime: MarketRegime::Trending(TrendDirection::Bullish),
        strategy: ActiveStrategy::TrendFollowing,
        confidence,
        trend_direction: Some(TrendDirection::Bullish),
        position_factor: 1.0,
        expected_duration: Some(10.0),
        methods_agree: Some(confidence > 0.7),
        reason: String::new(),
        detection_method: DetectionMethod::Ensemble,
        state_probabilities: None,
    }
}

fn bearish_trending(confidence: f64) -> RoutedSignal {
    RoutedSignal {
        regime: MarketRegime::Trending(TrendDirection::Bearish),
        strategy: ActiveStrategy::TrendFollowing,
        confidence,
        trend_direction: Some(TrendDirection::Bearish),
        position_factor: 1.0,
        expected_duration: Some(10.0),
        methods_agree: Some(confidence > 0.7),
        reason: String::new(),
        detection_method: DetectionMethod::Ensemble,
        state_probabilities: None,
    }
}

fn mean_reverting(confidence: f64) -> RoutedSignal {
    RoutedSignal {
        regime: MarketRegime::MeanReverting,
        strategy: ActiveStrategy::MeanReversion,
        confidence,
        trend_direction: None,
        position_factor: 1.0,
        expected_duration: Some(8.0),
        methods_agree: Some(confidence > 0.7),
        reason: String::new(),
        detection_method: DetectionMethod::Ensemble,
        state_probabilities: None,
    }
}

fn volatile_crisis(confidence: f64) -> RoutedSignal {
    RoutedSignal {
        regime: MarketRegime::Volatile,
        strategy: ActiveStrategy::NoTrade,
        confidence,
        trend_direction: None,
        position_factor: 0.5,
        expected_duration: Some(5.0),
        methods_agree: Some(true),
        reason: String::new(),
        detection_method: DetectionMethod::Ensemble,
        state_probabilities: None,
    }
}

fn uncertain(confidence: f64) -> RoutedSignal {
    RoutedSignal {
        regime: MarketRegime::Uncertain,
        strategy: ActiveStrategy::NoTrade,
        confidence,
        trend_direction: None,
        position_factor: 0.3,
        expected_duration: Some(3.0),
        methods_agree: Some(false),
        reason: String::new(),
        detection_method: DetectionMethod::Ensemble,
        state_probabilities: None,
    }
}

fn default_pipeline_no_gate_no_corr() -> TradingPipelineConfig {
    TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    }
}

// ============================================================================
// 1. Regime → Bridge → Hypothalamus → Amygdala flow
// ============================================================================

#[tokio::test]
async fn test_strong_bullish_regime_scales_up() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());
    let signal = bullish_trending(0.85);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(decision.is_actionable());
    assert!(
        decision.scale() > 1.0,
        "StrongBullish should scale above 1.0, got {}",
        decision.scale()
    );
    assert!(!decision.amygdala_high_risk);
    assert!(matches!(decision.action, TradeAction::Proceed { .. }));
    assert_eq!(
        decision.bridged_regime.hypothalamus_regime,
        HypothalamusRegime::StrongBullish
    );
}

#[tokio::test]
async fn test_moderate_bullish_regime_scales_moderately() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());
    let signal = bullish_trending(0.5); // Lower confidence → Bullish not StrongBullish

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(decision.is_actionable());
    let scale = decision.scale();
    // Bullish (not Strong) should scale up but less aggressively
    assert!(
        (0.9..=1.5).contains(&scale),
        "Bullish should scale modestly, got {}",
        scale
    );
}

#[tokio::test]
async fn test_strong_bearish_regime_scales_down() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());
    let signal = bearish_trending(0.85);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(decision.is_actionable());
    assert!(
        decision.scale() < 1.0,
        "StrongBearish should scale below 1.0, got {}",
        decision.scale()
    );
    assert_eq!(
        decision.bridged_regime.hypothalamus_regime,
        HypothalamusRegime::StrongBearish
    );
}

#[tokio::test]
async fn test_mean_reverting_regime_neutral_scale() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());
    let signal = mean_reverting(0.7);

    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &signal,
            "mean_reversion",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(decision.is_actionable());
    assert!(
        (decision.scale() - 1.0).abs() < 0.01,
        "Neutral regime should be ~1.0 scale, got {}",
        decision.scale()
    );
    assert!(!decision.amygdala_high_risk);
}

#[tokio::test]
async fn test_crisis_regime_blocks_new_positions() {
    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: false,
        allow_new_positions_in_crisis: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = volatile_crisis(0.9); // High confidence volatile → Crisis

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(decision.amygdala_high_risk);
    assert!(
        matches!(decision.action, TradeAction::ReduceOnly { .. }),
        "Crisis should trigger ReduceOnly, got: {:?}",
        decision.action
    );

    // Verify the reduce-only scale is significantly reduced
    let scale = decision.scale();
    assert!(
        scale < 0.5,
        "Crisis scale should be heavily reduced, got {}",
        scale
    );
}

#[tokio::test]
async fn test_crisis_regime_allows_new_positions_when_configured() {
    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: false,
        allow_new_positions_in_crisis: true,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = volatile_crisis(0.9);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(decision.amygdala_high_risk);
    assert!(
        matches!(decision.action, TradeAction::Proceed { .. }),
        "Crisis with allow_new_positions should Proceed, got: {:?}",
        decision.action
    );
    assert!(
        decision.scale() < 1.0,
        "Crisis should still reduce scale even when allowed"
    );
}

#[tokio::test]
async fn test_high_risk_amygdala_reduces_scale() {
    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: false,
        high_risk_scale_factor: 0.3,
        allow_new_positions_in_crisis: true,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = volatile_crisis(0.6); // Moderate volatile → HighVol, not Crisis

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    if decision.amygdala_high_risk {
        // Scale should be multiplied by 0.3
        assert!(
            decision.scale() < decision.raw_hypothalamus_scale,
            "High-risk should reduce scale: {} < {}",
            decision.scale(),
            decision.raw_hypothalamus_scale
        );
    }
}

#[tokio::test]
async fn test_amygdala_disabled_does_not_reduce() {
    let config = TradingPipelineConfig {
        enable_amygdala_filter: false,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = volatile_crisis(0.9);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    // With amygdala disabled, crisis should NOT trigger ReduceOnly
    assert!(
        matches!(decision.action, TradeAction::Proceed { .. }),
        "Disabled amygdala should not block: {:?}",
        decision.action
    );
}

#[tokio::test]
async fn test_hypothalamus_disabled_uses_1x_scale() {
    let config = TradingPipelineConfig {
        enable_hypothalamus_scaling: false,
        enable_amygdala_filter: false,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = bullish_trending(0.9);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert_eq!(
        decision.scale(),
        1.0,
        "Disabled hypothalamus should use 1.0 scale"
    );
}

// ============================================================================
// 2. Confidence threshold filtering
// ============================================================================

#[tokio::test]
async fn test_low_confidence_blocked_by_minimum() {
    let config = TradingPipelineConfig {
        min_regime_confidence: 0.6,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = uncertain(0.3);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(!decision.is_actionable());
    assert!(matches!(
        decision.action,
        TradeAction::Block {
            stage: PipelineStage::RegimeBridge,
            ..
        }
    ));
}

#[tokio::test]
async fn test_confidence_exactly_at_threshold_passes() {
    let config = TradingPipelineConfig {
        min_regime_confidence: 0.5,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = bullish_trending(0.5);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(
        decision.is_actionable(),
        "Confidence at threshold should pass"
    );
}

#[tokio::test]
async fn test_zero_confidence_threshold_allows_everything() {
    let config = TradingPipelineConfig {
        min_regime_confidence: 0.0,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = uncertain(0.01);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(
        decision.is_actionable(),
        "Zero threshold should allow everything"
    );
}

// ============================================================================
// 3. Strategy gating integration
// ============================================================================

#[tokio::test]
async fn test_gating_blocks_disabled_strategy() {
    let mut assets = HashMap::new();
    assets.insert(
        "BTCUSDT".to_string(),
        AssetStrategyConfig {
            disabled_strategies: vec!["ema_flip".to_string()],
            ..Default::default()
        },
    );

    let gating_config = StrategyGatingConfig {
        assets,
        ..Default::default()
    };

    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: false,
        gating: gating_config.clone(),
        ..Default::default()
    };

    let tracker = StrategyAffinityTracker::new(10);
    let gate = StrategyGate::new(gating_config, tracker);

    let pipeline = TradingPipelineBuilder::new()
        .config(config)
        .strategy_gate(gate)
        .build();

    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(!decision.is_actionable());
    assert!(!decision.gate_approved);
    assert!(matches!(
        decision.action,
        TradeAction::Block {
            stage: PipelineStage::StrategyGate,
            ..
        }
    ));
}

#[tokio::test]
async fn test_gating_allows_enabled_strategy() {
    let mut assets = HashMap::new();
    assets.insert(
        "BTCUSDT".to_string(),
        AssetStrategyConfig {
            enabled_strategies: vec!["ema_flip".to_string(), "mean_reversion".to_string()],
            ..Default::default()
        },
    );

    let gating_config = StrategyGatingConfig {
        assets,
        ..Default::default()
    };

    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: false,
        gating: gating_config.clone(),
        ..Default::default()
    };

    let tracker = StrategyAffinityTracker::new(10);
    let gate = StrategyGate::new(gating_config, tracker);

    let pipeline = TradingPipelineBuilder::new()
        .config(config)
        .strategy_gate(gate)
        .build();

    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(decision.is_actionable());
    assert!(decision.gate_approved);
}

#[tokio::test]
async fn test_gating_blocks_unlisted_strategy_when_allowlist_set() {
    let mut assets = HashMap::new();
    assets.insert(
        "BTCUSDT".to_string(),
        AssetStrategyConfig {
            enabled_strategies: vec!["mean_reversion".to_string()],
            ..Default::default()
        },
    );

    let gating_config = StrategyGatingConfig {
        assets,
        ..Default::default()
    };

    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: false,
        gating: gating_config.clone(),
        ..Default::default()
    };

    let tracker = StrategyAffinityTracker::new(10);
    let gate = StrategyGate::new(gating_config, tracker);

    let pipeline = TradingPipelineBuilder::new()
        .config(config)
        .strategy_gate(gate)
        .build();

    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(!decision.is_actionable());
    assert!(!decision.gate_approved);
}

#[tokio::test]
async fn test_gating_untested_strategy_allowed_by_default() {
    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: false,
        gating: StrategyGatingConfig {
            allow_untested: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);

    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &signal,
            "never_seen_strategy",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        decision.is_actionable(),
        "Untested strategy should be allowed when allow_untested=true"
    );
}

#[tokio::test]
async fn test_gating_regime_preference_filters() {
    let mut regime_prefs = HashMap::new();
    regime_prefs.insert(
        "TrendFollowing".to_string(),
        vec!["ema_flip".to_string(), "ema_ribbon_scalper".to_string()],
    );
    regime_prefs.insert(
        "MeanReversion".to_string(),
        vec!["mean_reversion".to_string(), "vwap_scalper".to_string()],
    );

    let mut assets = HashMap::new();
    assets.insert(
        "BTCUSDT".to_string(),
        AssetStrategyConfig {
            preferred_regime_strategies: regime_prefs,
            ..Default::default()
        },
    );

    let gating_config = StrategyGatingConfig {
        assets,
        ..Default::default()
    };

    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: false,
        gating: gating_config.clone(),
        ..Default::default()
    };

    let tracker = StrategyAffinityTracker::new(10);
    let gate = StrategyGate::new(gating_config, tracker);

    let pipeline = TradingPipelineBuilder::new()
        .config(config)
        .strategy_gate(gate)
        .build();

    // Trending regime + mean_reversion strategy → should block
    let trend_signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &trend_signal,
            "mean_reversion",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        !decision.gate_approved,
        "Mean reversion should not be approved in trending regime when regime prefs are set"
    );

    // Trending regime + ema_flip strategy → should pass
    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &trend_signal,
            "ema_flip",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        decision.gate_approved,
        "ema_flip should be approved in trending regime"
    );
}

// ============================================================================
// 4. Affinity tracking + gating feedback loop
// ============================================================================

#[tokio::test]
async fn test_affinity_records_affect_gating() {
    let gating_config = StrategyGatingConfig {
        min_weight: 0.3,
        allow_untested: true,
        ..Default::default()
    };

    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: false,
        gating: gating_config,
        affinity_min_trades: 5,
        ..Default::default()
    };

    let pipeline = TradingPipeline::new(config);

    // Initially untested → allowed
    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &signal,
            "bad_strategy",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(
        decision.is_actionable(),
        "Untested strategy should pass initially"
    );

    // Record poor results for 'bad_strategy' on BTCUSDT
    for _ in 0..10 {
        pipeline
            .record_trade_result("bad_strategy", "BTCUSDT", -100.0, false)
            .await;
    }

    // After recording bad results, affinity weight should drop below min_weight
    let weight = pipeline.affinity_weight("bad_strategy", "BTCUSDT").await;
    assert!(
        weight < 0.3,
        "Bad strategy should have low affinity weight, got {}",
        weight
    );

    // Now gating should block it
    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &signal,
            "bad_strategy",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(
        !decision.is_actionable(),
        "Strategy with low affinity should be gated"
    );
}

#[tokio::test]
async fn test_affinity_good_strategy_passes_gate() {
    let gating_config = StrategyGatingConfig {
        min_weight: 0.3,
        allow_untested: true,
        ..Default::default()
    };

    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: false,
        gating: gating_config,
        affinity_min_trades: 5,
        ..Default::default()
    };

    let pipeline = TradingPipeline::new(config);

    // Record good results for 'good_strategy' on BTCUSDT
    for _ in 0..10 {
        pipeline
            .record_trade_result("good_strategy", "BTCUSDT", 150.0, true)
            .await;
    }

    let weight = pipeline.affinity_weight("good_strategy", "BTCUSDT").await;
    assert!(
        weight > 0.5,
        "Good strategy should have high affinity weight, got {}",
        weight
    );

    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &signal,
            "good_strategy",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(
        decision.is_actionable(),
        "Good strategy should pass through gate"
    );
}

// ============================================================================
// 5. Correlation-based position limits
// ============================================================================

#[tokio::test]
async fn test_correlation_blocks_excess_correlated_positions() {
    let corr_config = CorrelationConfig {
        window: 10,
        correlation_threshold: 0.5,
        max_correlated_positions: 2,
        min_observations: 5,
        monitored_pairs: Vec::new(),
    };

    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: true,
        correlation: corr_config.clone(),
        ..Default::default()
    };

    let mut ct = CorrelationTracker::new(corr_config);

    // Feed perfectly correlated prices
    for i in 0..20 {
        let base = 100.0 + i as f64;
        ct.update("BTCUSDT", base);
        ct.update("ETHUSDT", base * 2.0);
        ct.update("SOLUSDT", base * 0.5);
    }

    let pipeline = TradingPipelineBuilder::new()
        .config(config)
        .correlation_tracker(ct)
        .build();

    let signal = bullish_trending(0.85);

    // Holding 2 correlated positions: ETHUSDT + SOLUSDT
    let positions = vec!["ETHUSDT".to_string(), "SOLUSDT".to_string()];

    let decision = pipeline
        .evaluate(
            "BTCUSDT", &signal, "ema_flip", &positions, None, None, None, None,
        )
        .await;

    assert!(!decision.is_actionable());
    assert!(!decision.correlation_passed);
    assert!(matches!(
        decision.action,
        TradeAction::Block {
            stage: PipelineStage::CorrelationFilter,
            ..
        }
    ));
}

#[tokio::test]
async fn test_correlation_allows_uncorrelated_asset() {
    let corr_config = CorrelationConfig {
        window: 10,
        correlation_threshold: 0.5,
        max_correlated_positions: 2,
        min_observations: 5,
        monitored_pairs: Vec::new(),
    };

    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: true,
        correlation: corr_config.clone(),
        ..Default::default()
    };

    let mut ct = CorrelationTracker::new(corr_config);

    // Feed correlated BTC/ETH, but uncorrelated DOGE
    for i in 0..20 {
        let base = 100.0 + i as f64;
        ct.update("BTCUSDT", base);
        ct.update("ETHUSDT", base * 2.0);
        // DOGE has an inverse/random pattern
        ct.update(
            "DOGEUSDT",
            100.0 + (i as f64 * std::f64::consts::PI).sin() * 10.0,
        );
    }

    let pipeline = TradingPipelineBuilder::new()
        .config(config)
        .correlation_tracker(ct)
        .build();

    let signal = bullish_trending(0.85);

    // Holding BTCUSDT, try to add DOGEUSDT (which is uncorrelated)
    let positions = vec!["BTCUSDT".to_string()];

    let decision = pipeline
        .evaluate(
            "DOGEUSDT", &signal, "ema_flip", &positions, None, None, None, None,
        )
        .await;

    assert!(
        decision.is_actionable(),
        "Uncorrelated asset should pass correlation filter"
    );
    assert!(decision.correlation_passed);
}

#[tokio::test]
async fn test_correlation_price_updates_through_pipeline() {
    let corr_config = CorrelationConfig {
        window: 10,
        correlation_threshold: 0.5,
        max_correlated_positions: 2,
        min_observations: 5,
        monitored_pairs: Vec::new(),
    };

    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: true,
        correlation: corr_config,
        ..Default::default()
    };

    let pipeline = TradingPipeline::new(config);

    // Feed perfectly correlated prices through the pipeline API
    for i in 0..20 {
        let base = 100.0 + i as f64;
        pipeline.update_price("BTCUSDT", base).await;
        pipeline.update_price("ETHUSDT", base * 2.0).await;
    }

    let signal = bullish_trending(0.85);
    let positions = vec!["ETHUSDT".to_string()];

    let decision = pipeline
        .evaluate(
            "BTCUSDT", &signal, "ema_flip", &positions, None, None, None, None,
        )
        .await;

    // With max_correlated_positions=2, holding 1 correlated + adding 1 = 2, should pass
    assert!(
        decision.is_actionable(),
        "2 correlated positions should be within limit of 2"
    );

    // Add a second correlated position
    let positions = vec!["ETHUSDT".to_string(), "BTCUSDT".to_string()];

    // Now try to add a third correlated asset (feed correlated prices)
    for i in 0..20 {
        let base = 100.0 + i as f64;
        pipeline.update_price("BNBUSDT", base * 1.5).await;
    }

    let decision = pipeline
        .evaluate(
            "BNBUSDT", &signal, "ema_flip", &positions, None, None, None, None,
        )
        .await;

    // 2 existing + 1 new = 3, exceeds limit of 2
    assert!(
        !decision.is_actionable(),
        "3 correlated positions should exceed limit of 2"
    );
}

#[tokio::test]
async fn test_correlation_batch_price_updates() {
    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: true,
        correlation: CorrelationConfig {
            window: 10,
            min_observations: 5,
            ..Default::default()
        },
        ..Default::default()
    };

    let pipeline = TradingPipeline::new(config);

    // Initial prices
    pipeline.update_price("BTCUSDT", 50000.0).await;
    pipeline.update_price("ETHUSDT", 3000.0).await;

    // Batch updates
    for i in 1..20 {
        pipeline
            .update_prices_batch(&[
                ("BTCUSDT", 50000.0 + i as f64 * 100.0),
                ("ETHUSDT", 3000.0 + i as f64 * 6.0),
            ])
            .await;
    }

    let pairs = pipeline.highly_correlated_pairs().await;
    // Both assets are trending up, so they should be correlated
    assert!(
        !pairs.is_empty(),
        "Perfectly correlated assets should appear in highly_correlated_pairs"
    );
}

#[tokio::test]
async fn test_correlation_disabled_allows_everything() {
    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };

    let pipeline = TradingPipeline::new(config);

    // Even with tons of correlated positions, should pass
    let signal = bullish_trending(0.85);
    let positions: Vec<String> = (0..20).map(|i| format!("ASSET{i}")).collect();

    let decision = pipeline
        .evaluate(
            "BTCUSDT", &signal, "ema_flip", &positions, None, None, None, None,
        )
        .await;

    assert!(
        decision.is_actionable(),
        "Disabled correlation filter should not block"
    );
    assert!(decision.correlation_passed);
}

// ============================================================================
// 6. Kill switch
// ============================================================================

#[tokio::test]
async fn test_kill_switch_blocks_all_trading() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());

    // Initially should work
    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(decision.is_actionable());

    // Activate kill switch
    pipeline.activate_kill_switch().await;
    assert!(pipeline.is_killed().await);

    // All evaluations should be blocked
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(!decision.is_actionable());
    assert!(matches!(
        decision.action,
        TradeAction::Block {
            stage: PipelineStage::KillSwitch,
            ..
        }
    ));

    // Deactivate kill switch
    pipeline.deactivate_kill_switch().await;
    assert!(!pipeline.is_killed().await);

    // Should work again
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(decision.is_actionable());
}

#[tokio::test]
async fn test_kill_switch_idempotent() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());

    pipeline.activate_kill_switch().await;
    pipeline.activate_kill_switch().await; // double activate
    assert!(pipeline.is_killed().await);

    pipeline.deactivate_kill_switch().await;
    pipeline.deactivate_kill_switch().await; // double deactivate
    assert!(!pipeline.is_killed().await);
}

// ============================================================================
// 7. Pipeline metrics
// ============================================================================

#[tokio::test]
async fn test_pipeline_metrics_tracking() {
    let config = TradingPipelineConfig {
        min_regime_confidence: 0.5,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);

    // Proceed
    let signal = bullish_trending(0.85);
    pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    // ReduceOnly
    let signal = volatile_crisis(0.9);
    pipeline
        .evaluate("ETHUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    // Block (low confidence)
    let signal = uncertain(0.2);
    pipeline
        .evaluate("SOLUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    // Kill switch block
    pipeline.activate_kill_switch().await;
    let signal = bullish_trending(0.9);
    pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    let metrics = pipeline.metrics_snapshot().await;
    assert_eq!(metrics.total_evaluations, 4);
    assert_eq!(metrics.proceed_count, 1);
    assert_eq!(metrics.reduce_only_count, 1);
    assert_eq!(metrics.block_count, 2);
    assert!(metrics.avg_evaluation_us() > 0.0);
    assert!(metrics.block_rate_pct() > 0.0);
}

#[tokio::test]
async fn test_pipeline_metrics_blocks_by_stage() {
    let config = TradingPipelineConfig {
        min_regime_confidence: 0.5,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);

    // Block by regime bridge (low confidence)
    let signal = uncertain(0.2);
    pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    pipeline
        .evaluate("ETHUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    // Block by kill switch
    pipeline.activate_kill_switch().await;
    let signal = bullish_trending(0.9);
    pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    let metrics = pipeline.metrics_snapshot().await;
    assert_eq!(
        metrics.blocks_by_stage.get(&PipelineStage::RegimeBridge),
        Some(&2)
    );
    assert_eq!(
        metrics.blocks_by_stage.get(&PipelineStage::KillSwitch),
        Some(&1)
    );
}

// ============================================================================
// 8. Multi-asset pipeline consistency
// ============================================================================

#[tokio::test]
async fn test_multi_asset_evaluations_are_independent() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());

    let btc_signal = bullish_trending(0.85);
    let eth_signal = bearish_trending(0.8);
    let sol_signal = volatile_crisis(0.9);

    let btc_decision = pipeline
        .evaluate(
            "BTCUSDT",
            &btc_signal,
            "ema_flip",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    let eth_decision = pipeline
        .evaluate(
            "ETHUSDT",
            &eth_signal,
            "ema_flip",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    let sol_decision = pipeline
        .evaluate(
            "SOLUSDT",
            &sol_signal,
            "ema_flip",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;

    // BTC: Strong bullish → proceed with scale > 1
    assert!(btc_decision.is_actionable());
    assert!(btc_decision.scale() > 1.0);

    // ETH: Strong bearish → proceed with reduced scale
    assert!(eth_decision.is_actionable());
    assert!(eth_decision.scale() < 1.0);

    // SOL: Crisis → reduce only
    assert!(matches!(
        sol_decision.action,
        TradeAction::ReduceOnly { .. }
    ));
}

#[tokio::test]
async fn test_regime_transitions_produce_different_decisions() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());

    let regimes = [
        (bullish_trending(0.85), "StrongBullish"),
        (mean_reverting(0.7), "Neutral"),
        (volatile_crisis(0.9), "Crisis"),
        (bearish_trending(0.85), "StrongBearish"),
    ];

    let mut scales: Vec<f64> = Vec::new();

    for (signal, _label) in &regimes {
        let decision = pipeline
            .evaluate("BTCUSDT", signal, "ema_flip", &[], None, None, None, None)
            .await;
        scales.push(decision.scale());
    }

    // StrongBullish scale should be different from StrongBearish
    assert!(
        (scales[0] - scales[3]).abs() > 0.1,
        "Different regimes should produce different scales: {:?}",
        scales
    );
}

// ============================================================================
// 9. Sync evaluation parity
// ============================================================================

#[test]
fn test_sync_evaluation_matches_expected_behavior() {
    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let ct = CorrelationTracker::with_defaults();
    let gate = StrategyGate::new(
        StrategyGatingConfig::default(),
        StrategyAffinityTracker::new(10),
    );

    // Strong bullish
    let signal = bullish_trending(0.85);
    let decision = TradingPipeline::evaluate_sync(
        &config,
        "BTCUSDT",
        &signal,
        "ema_flip",
        &[],
        &ct,
        &gate,
        false,
        None,
        None,
        None,
        None,
    );
    assert!(decision.is_actionable());
    assert!(decision.scale() > 1.0);

    // Crisis
    let signal = volatile_crisis(0.9);
    let decision = TradingPipeline::evaluate_sync(
        &config,
        "BTCUSDT",
        &signal,
        "ema_flip",
        &[],
        &ct,
        &gate,
        false,
        None,
        None,
        None,
        None,
    );
    assert!(matches!(decision.action, TradeAction::ReduceOnly { .. }));

    // Kill switch
    let signal = bullish_trending(0.9);
    let decision = TradingPipeline::evaluate_sync(
        &config,
        "BTCUSDT",
        &signal,
        "ema_flip",
        &[],
        &ct,
        &gate,
        true,
        None,
        None,
        None,
        None,
    );
    assert!(!decision.is_actionable());
    assert!(matches!(
        decision.action,
        TradeAction::Block {
            stage: PipelineStage::KillSwitch,
            ..
        }
    ));
}

// ============================================================================
// 10. Scale clamping
// ============================================================================

#[tokio::test]
async fn test_scale_clamped_to_max() {
    let config = TradingPipelineConfig {
        max_position_scale: 1.5,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = bullish_trending(0.9);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(
        decision.scale() <= 1.5,
        "Scale should be clamped to max, got {}",
        decision.scale()
    );
}

#[tokio::test]
async fn test_scale_clamped_to_min() {
    let config = TradingPipelineConfig {
        min_position_scale: 0.1,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);
    let signal = bearish_trending(0.9); // Strong bearish scales down

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    if decision.is_actionable() {
        assert!(
            decision.scale() >= 0.1,
            "Scale should be clamped to min, got {}",
            decision.scale()
        );
    }
}

// ============================================================================
// 11. All features disabled passthrough
// ============================================================================

#[tokio::test]
async fn test_all_features_disabled_is_pure_passthrough() {
    let config = TradingPipelineConfig {
        enable_hypothalamus_scaling: false,
        enable_amygdala_filter: false,
        enable_gating: false,
        enable_correlation_filter: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);

    // Even crisis should pass through at 1.0 scale
    let signal = volatile_crisis(0.95);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(decision.is_actionable());
    assert_eq!(decision.scale(), 1.0);
    assert!(matches!(decision.action, TradeAction::Proceed { scale } if scale == 1.0));
}

// ============================================================================
// 12. Pipeline Builder
// ============================================================================

#[tokio::test]
async fn test_builder_with_custom_components() {
    let corr_config = CorrelationConfig {
        window: 5,
        min_observations: 3,
        ..Default::default()
    };
    let mut ct = CorrelationTracker::new(corr_config);

    // Pre-populate some data
    for i in 0..10 {
        ct.update("BTCUSDT", 50000.0 + i as f64 * 100.0);
        ct.update("ETHUSDT", 3000.0 + i as f64 * 6.0);
    }

    let mut tracker = StrategyAffinityTracker::new(5);
    for _ in 0..10 {
        tracker.record_trade_result("ema_flip", "BTCUSDT", 100.0, true);
    }

    let gate = StrategyGate::new(StrategyGatingConfig::default(), tracker);

    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: true,
        ..Default::default()
    };

    let pipeline = TradingPipelineBuilder::new()
        .config(config)
        .correlation_tracker(ct)
        .strategy_gate(gate)
        .build();

    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(
        decision.is_actionable(),
        "Custom components should be properly wired"
    );
}

// ============================================================================
// 13. make_test_signal helper
// ============================================================================

#[test]
fn test_make_test_signal_covers_all_regimes() {
    let regimes = [
        MarketRegime::Trending(TrendDirection::Bullish),
        MarketRegime::Trending(TrendDirection::Bearish),
        MarketRegime::MeanReverting,
        MarketRegime::Volatile,
        MarketRegime::Uncertain,
    ];

    for regime in &regimes {
        let signal = make_test_signal(*regime, 0.7);
        assert_eq!(signal.regime, *regime);
        assert_eq!(signal.confidence, 0.7);
    }
}

#[test]
fn test_make_test_signal_strategy_mapping() {
    assert_eq!(
        make_test_signal(MarketRegime::Trending(TrendDirection::Bullish), 0.8).strategy,
        ActiveStrategy::TrendFollowing
    );
    assert_eq!(
        make_test_signal(MarketRegime::Trending(TrendDirection::Bearish), 0.8).strategy,
        ActiveStrategy::TrendFollowing
    );
    assert_eq!(
        make_test_signal(MarketRegime::MeanReverting, 0.8).strategy,
        ActiveStrategy::MeanReversion
    );
    assert_eq!(
        make_test_signal(MarketRegime::Volatile, 0.8).strategy,
        ActiveStrategy::NoTrade
    );
    assert_eq!(
        make_test_signal(MarketRegime::Uncertain, 0.8).strategy,
        ActiveStrategy::NoTrade
    );
}

// ============================================================================
// 14. Brain Runtime integration
// ============================================================================

#[tokio::test]
async fn test_runtime_boot_and_evaluate() {
    let config = BrainRuntimeConfig {
        auto_start_watchdog: false,
        ..Default::default()
    };
    let mut runtime = BrainRuntime::new(config);

    // Boot
    let report = runtime.boot().await.expect("Boot should succeed");
    assert!(report.is_boot_safe());

    // Get pipeline and evaluate
    let pipeline = runtime.pipeline().expect("Pipeline should be initialized");
    let signal = bullish_trending(0.85);
    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(decision.is_actionable());
}

#[tokio::test]
async fn test_runtime_boot_fails_with_invalid_config() {
    let config = BrainRuntimeConfig {
        pipeline: TradingPipelineConfig {
            max_position_scale: -1.0, // Invalid
            ..Default::default()
        },
        enforce_preflight: true,
        auto_start_watchdog: false,
        ..Default::default()
    };
    let mut runtime = BrainRuntime::new(config);

    let result = runtime.boot().await;
    assert!(result.is_err(), "Boot should fail with invalid config");
}

#[tokio::test]
async fn test_runtime_shutdown_kills_pipeline() {
    let config = BrainRuntimeConfig {
        auto_start_watchdog: false,
        ..Default::default()
    };
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("Boot should succeed");
    let pipeline = runtime.pipeline().unwrap().clone();

    assert!(!pipeline.is_killed().await);

    runtime.shutdown().await;

    assert!(
        pipeline.is_killed().await,
        "Pipeline should be killed after shutdown"
    );
}

#[tokio::test]
async fn test_runtime_health_report_reflects_state() {
    let config = BrainRuntimeConfig {
        auto_start_watchdog: false,
        ..Default::default()
    };
    let mut runtime = BrainRuntime::new(config);

    // Before boot
    let report = runtime.health_report().await;
    assert!(!report.is_healthy());

    // After boot
    runtime.boot().await.expect("Boot should succeed");
    let report = runtime.health_report().await;
    assert!(report.is_healthy());

    // After kill switch
    let pipeline = runtime.pipeline().unwrap().clone();
    pipeline.activate_kill_switch().await;
    let report = runtime.health_report().await;
    assert!(!report.is_healthy());
}

#[tokio::test]
async fn test_runtime_preflight_only_does_not_start_pipeline() {
    let mut runtime = BrainRuntime::new(BrainRuntimeConfig::default());
    let report = runtime.preflight_only().await;

    assert!(report.is_boot_safe());
    assert!(
        runtime.pipeline().is_none(),
        "Pipeline should not be initialized in preflight-only mode"
    );
}

#[tokio::test]
async fn test_runtime_with_watchdog_and_heartbeat() {
    let config = BrainRuntimeConfig {
        auto_start_watchdog: true,
        forward_heartbeat_ms: 100,
        watchdog: WatchdogRuntimeConfig {
            check_interval_ms: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut runtime = BrainRuntime::new(config);

    runtime.boot().await.expect("Boot should succeed");
    assert!(runtime.watchdog_handle().is_some());

    // Give watchdog a moment
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Send heartbeats
    runtime.heartbeat_core().await;

    let snapshot = runtime.watchdog_snapshot().await;
    assert!(snapshot.is_some());
    assert!(snapshot.unwrap().total_components >= 4);

    runtime.shutdown().await;
}

// ============================================================================
// 15. End-to-end pipeline scenario tests
// ============================================================================

#[tokio::test]
async fn test_e2e_scenario_normal_trading_day() {
    // Simulates a sequence of regime changes during a trading day
    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: false,
        high_risk_scale_factor: 0.5,
        allow_new_positions_in_crisis: false,
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);

    // Morning: Trending bullish market
    let signal = bullish_trending(0.8);
    let d = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(d.is_actionable());
    assert!(d.scale() >= 1.0, "Bullish morning should allow trading");

    // Midday: Market becomes uncertain
    let signal = uncertain(0.4);
    let d = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(d.is_actionable()); // Still actionable but lower scale

    // Afternoon: Volatile spike (crisis)
    let signal = volatile_crisis(0.9);
    let d = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(
        matches!(d.action, TradeAction::ReduceOnly { .. }),
        "Crisis should trigger reduce-only"
    );

    // Evening: Mean reverting after volatility settles
    let signal = mean_reverting(0.7);
    let d = pipeline
        .evaluate(
            "BTCUSDT",
            &signal,
            "mean_reversion",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(d.is_actionable());
    assert!(
        (d.scale() - 1.0).abs() < 0.1,
        "Mean reverting should be near 1.0 scale"
    );
}

#[tokio::test]
async fn test_e2e_scenario_portfolio_concentration_risk() {
    // Simulates building up correlated positions until blocked
    let corr_config = CorrelationConfig {
        window: 10,
        correlation_threshold: 0.5,
        max_correlated_positions: 3,
        min_observations: 5,
        monitored_pairs: Vec::new(),
    };

    let config = TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: true,
        correlation: corr_config.clone(),
        ..Default::default()
    };

    let pipeline = TradingPipeline::new(config);

    // Feed correlated prices for 4 crypto assets
    for i in 0..20 {
        let base = 1000.0 + i as f64 * 10.0;
        pipeline.update_price("BTCUSDT", base).await;
        pipeline.update_price("ETHUSDT", base * 0.06).await;
        pipeline.update_price("BNBUSDT", base * 0.0005).await;
        pipeline.update_price("SOLUSDT", base * 0.003).await;
    }

    let signal = bullish_trending(0.85);

    // First position: BTCUSDT → should pass
    let d = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(d.is_actionable(), "First position should always pass");

    // Second position: ETHUSDT with BTCUSDT held → should pass (2 <= 3)
    let positions = vec!["BTCUSDT".to_string()];
    let d = pipeline
        .evaluate(
            "ETHUSDT", &signal, "ema_flip", &positions, None, None, None, None,
        )
        .await;
    assert!(d.is_actionable(), "Second correlated position should pass");

    // Third position: BNBUSDT with BTC+ETH held → should pass (3 <= 3)
    let positions = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
    let d = pipeline
        .evaluate(
            "BNBUSDT", &signal, "ema_flip", &positions, None, None, None, None,
        )
        .await;
    assert!(d.is_actionable(), "Third correlated position should pass");

    // Fourth position: SOLUSDT with BTC+ETH+BNB → should BLOCK (4 > 3)
    let positions = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "BNBUSDT".to_string(),
    ];
    let d = pipeline
        .evaluate(
            "SOLUSDT", &signal, "ema_flip", &positions, None, None, None, None,
        )
        .await;
    assert!(
        !d.is_actionable(),
        "Fourth correlated position should be blocked"
    );
    assert!(matches!(
        d.action,
        TradeAction::Block {
            stage: PipelineStage::CorrelationFilter,
            ..
        }
    ));
}

#[tokio::test]
async fn test_e2e_scenario_strategy_rotation() {
    // Simulates regime-driven strategy rotation over time
    let mut regime_prefs = HashMap::new();
    regime_prefs.insert(
        "TrendFollowing".to_string(),
        vec!["ema_flip".to_string(), "trend_pullback".to_string()],
    );
    regime_prefs.insert(
        "MeanReverting".to_string(),
        vec!["mean_reversion".to_string(), "vwap_scalper".to_string()],
    );

    let mut assets = HashMap::new();
    assets.insert(
        "BTCUSDT".to_string(),
        AssetStrategyConfig {
            preferred_regime_strategies: regime_prefs,
            ..Default::default()
        },
    );

    let gating_config = StrategyGatingConfig {
        assets,
        allow_untested: true,
        ..Default::default()
    };

    let config = TradingPipelineConfig {
        enable_gating: true,
        enable_correlation_filter: false,
        gating: gating_config.clone(),
        ..Default::default()
    };

    let tracker = StrategyAffinityTracker::new(10);
    let gate = StrategyGate::new(gating_config, tracker);

    let pipeline = TradingPipelineBuilder::new()
        .config(config)
        .strategy_gate(gate)
        .build();

    // Phase 1: Trending market → trend strategies should work, MR should not
    let trend_signal = bullish_trending(0.85);

    let d = pipeline
        .evaluate(
            "BTCUSDT",
            &trend_signal,
            "ema_flip",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(d.gate_approved, "ema_flip should work in trending");

    let d = pipeline
        .evaluate(
            "BTCUSDT",
            &trend_signal,
            "mean_reversion",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(
        !d.gate_approved,
        "mean_reversion should not work in trending"
    );

    // Phase 2: Mean reverting market → MR strategies should work, trend should not
    let mr_signal = mean_reverting(0.75);

    let d = pipeline
        .evaluate(
            "BTCUSDT",
            &mr_signal,
            "mean_reversion",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(
        d.gate_approved,
        "mean_reversion should work in mean reverting"
    );

    let d = pipeline
        .evaluate(
            "BTCUSDT",
            &mr_signal,
            "ema_flip",
            &[],
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(
        !d.gate_approved,
        "ema_flip should not work in mean reverting"
    );
}

#[tokio::test]
async fn test_e2e_scenario_kill_switch_emergency() {
    let config = BrainRuntimeConfig {
        auto_start_watchdog: false,
        ..Default::default()
    };
    let mut runtime = BrainRuntime::new(config);
    runtime.boot().await.expect("Boot should succeed");

    let pipeline = runtime.pipeline().unwrap().clone();
    let signal = bullish_trending(0.85);

    // Normal trading
    let d = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(d.is_actionable());

    // Emergency! Kill switch activated (simulating watchdog detecting critical failure)
    pipeline.activate_kill_switch().await;

    // All trading halted
    let d = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(!d.is_actionable());

    // Health report reflects kill state
    let health = runtime.health_report().await;
    assert!(!health.is_healthy());

    // Recovery
    pipeline.deactivate_kill_switch().await;
    let d = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(d.is_actionable());

    let health = runtime.health_report().await;
    assert!(health.is_healthy());

    runtime.shutdown().await;
}

// ============================================================================
// 16. Decision display / formatting
// ============================================================================

#[tokio::test]
async fn test_decision_display_contains_key_info() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());
    let signal = bullish_trending(0.85);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    let display = format!("{}", decision);
    assert!(display.contains("BTCUSDT"));
    assert!(display.contains("ema_flip"));
    assert!(display.contains("PROCEED") || display.contains("BLOCK") || display.contains("REDUCE"));
}

// ============================================================================
// 17. Enabled strategies query
// ============================================================================

#[tokio::test]
async fn test_enabled_strategies_query() {
    let config = TradingPipelineConfig {
        enable_gating: true,
        gating: StrategyGatingConfig {
            allow_untested: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let pipeline = TradingPipeline::new(config);

    let all_strategies = vec![
        "ema_flip".to_string(),
        "mean_reversion".to_string(),
        "vwap_scalper".to_string(),
        "trend_pullback".to_string(),
    ];

    // With default gating (allow_untested=true, no per-asset config), all should pass
    let enabled = pipeline
        .enabled_strategies_for(
            "BTCUSDT",
            &MarketRegime::Trending(TrendDirection::Bullish),
            &all_strategies,
        )
        .await;
    assert_eq!(enabled.len(), 4);
}

// ============================================================================
// 18. Concurrent evaluation safety
// ============================================================================

#[tokio::test]
async fn test_concurrent_evaluations_are_safe() {
    let pipeline = Arc::new(TradingPipeline::new(default_pipeline_no_gate_no_corr()));

    let mut handles = Vec::new();

    for i in 0..10 {
        let p = pipeline.clone();
        let symbol = format!("ASSET{}", i);
        let handle = tokio::spawn(async move {
            let signal = bullish_trending(0.85);
            let decision = p
                .evaluate(&symbol, &signal, "ema_flip", &[], None, None, None, None)
                .await;
            assert!(decision.is_actionable());
            decision
        });
        handles.push(handle);
    }

    let results: Vec<TradingDecision> = futures::future::try_join_all(handles)
        .await
        .expect("All tasks should complete");

    assert_eq!(results.len(), 10);

    let metrics = pipeline.metrics_snapshot().await;
    assert_eq!(metrics.total_evaluations, 10);
}

#[tokio::test]
async fn test_concurrent_price_updates_and_evaluations() {
    let pipeline = Arc::new(TradingPipeline::new(TradingPipelineConfig {
        enable_gating: false,
        enable_correlation_filter: true,
        ..Default::default()
    }));

    // Spawn price updaters
    let p1 = pipeline.clone();
    let updater = tokio::spawn(async move {
        for i in 0..50 {
            p1.update_price("BTCUSDT", 50000.0 + i as f64 * 10.0).await;
            p1.update_price("ETHUSDT", 3000.0 + i as f64 * 0.6).await;
        }
    });

    // Spawn evaluators
    let p2 = pipeline.clone();
    let evaluator = tokio::spawn(async move {
        let signal = bullish_trending(0.85);
        for _ in 0..20 {
            let _ = p2
                .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
                .await;
        }
    });

    // Both should complete without deadlock or panic
    let (r1, r2) = tokio::join!(updater, evaluator);
    r1.expect("Updater should complete");
    r2.expect("Evaluator should complete");

    let metrics = pipeline.metrics_snapshot().await;
    assert_eq!(metrics.total_evaluations, 20);
}

// ============================================================================
// 19. BridgedRegimeState coverage
// ============================================================================

#[test]
fn test_bridge_all_regime_types() {
    let test_cases: Vec<(RoutedSignal, HypothalamusRegime)> = vec![
        (bullish_trending(0.85), HypothalamusRegime::StrongBullish),
        (bullish_trending(0.5), HypothalamusRegime::Bullish),
        (bearish_trending(0.85), HypothalamusRegime::StrongBearish),
        (bearish_trending(0.5), HypothalamusRegime::Bearish),
        (mean_reverting(0.7), HypothalamusRegime::Neutral),
    ];

    for (signal, expected_regime) in &test_cases {
        let bridged = bridge_regime_signal("BTCUSDT", signal, None, None, None, None);
        assert_eq!(
            bridged.hypothalamus_regime, *expected_regime,
            "Signal {:?} should bridge to {:?}",
            signal.regime, expected_regime
        );
    }
}

#[test]
fn test_bridge_crisis_detection() {
    let signal = volatile_crisis(0.9);
    let bridged = bridge_regime_signal("BTCUSDT", &signal, None, None, None, None);

    assert!(bridged.is_high_risk, "Crisis should be high risk");
    assert_eq!(
        bridged.amygdala_regime,
        AmygdalaRegime::Crisis,
        "High confidence volatile should map to Crisis"
    );
    assert!(
        bridged.position_scale < 1.0,
        "Crisis scale should be below 1.0"
    );
}

// ============================================================================
// 20. Indicator enrichment passthrough
// ============================================================================

#[tokio::test]
async fn test_pipeline_passes_indicators_to_bridge() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());
    let signal = bullish_trending(0.85);

    let decision = pipeline
        .evaluate(
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            Some(35.0),  // ADX
            Some(0.6),   // BB width percentile
            Some(500.0), // ATR
            Some(1.5),   // Relative volume
        )
        .await;

    assert!(decision.is_actionable());
    // Verify indicators were populated
    let indicators = &decision.bridged_regime.indicators;
    assert!(
        indicators.trend_strength > 0.0,
        "ADX should set trend_strength"
    );
}

#[tokio::test]
async fn test_pipeline_works_without_indicators() {
    let pipeline = TradingPipeline::new(default_pipeline_no_gate_no_corr());
    let signal = bullish_trending(0.85);

    let decision = pipeline
        .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(
        decision.is_actionable(),
        "Pipeline should work without optional indicators"
    );
}
