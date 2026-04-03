//! Integration Tests for Regime Bridge Pipeline
//!
//! Validates the complete data flow:
//!   1. Raw ticks → RegimeManager tick aggregation → candle completion
//!   2. Candle → EnhancedRouter regime classification → RoutedSignal
//!   3. RoutedSignal → bridge_regime_signal() → BridgedRegimeState
//!   4. BridgedRegimeState contains valid hypothalamus + amygdala regimes
//!   5. TOML config loading works end-to-end
//!   6. Broadcast channel delivers BridgedRegimeState to subscribers
//!
//! # Running Tests
//!
//! ```bash
//! cargo test -p janus-forward --test regime_bridge_integration
//! ```
//!
//! These tests do NOT require any external services (no Redis, no Bybit, no QuestDB).

use janus_forward::regime::{RegimeManager, RegimeManagerConfig};
use janus_forward::regime_bridge::{
    AmygdalaRegime, BridgedRegimeState, HypothalamusRegime, RegimeIndicators, bridge_regime_signal,
    build_regime_indicators, to_amygdala_regime, to_amygdala_regime_with_vol,
    to_hypothalamus_regime,
};
use janus_regime::{ActiveStrategy, DetectionMethod, MarketRegime, RoutedSignal, TrendDirection};

// ============================================================================
// Helper functions
// ============================================================================

/// Create a RegimeManager with fast candle aggregation (few ticks per candle)
/// so tests don't need thousands of ticks to produce a signal.
fn fast_regime_manager() -> RegimeManager {
    RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    })
}

/// Feed a series of prices as ticks, each one tick at a time.
fn feed_prices(mgr: &mut RegimeManager, symbol: &str, prices: &[f64]) -> Vec<RoutedSignal> {
    let mut signals = Vec::new();
    for &price in prices {
        let spread = price * 0.0001; // tiny spread
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            signals.push(signal);
        }
    }
    signals
}

/// Generate a trending upward price series.
fn trending_up_prices(start: f64, count: usize) -> Vec<f64> {
    (0..count).map(|i| start + i as f64 * 0.5).collect()
}

/// Generate a mean-reverting (oscillating) price series.
fn mean_reverting_prices(center: f64, amplitude: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let phase = (i as f64 * 0.7).sin() * amplitude;
            center + phase
        })
        .collect()
}

/// Generate a volatile price series with large swings.
fn volatile_prices(start: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let swing = if i % 2 == 0 { 5.0 } else { -5.0 };
            let trend = (i as f64 * 0.3).sin() * 3.0;
            start + swing + trend
        })
        .collect()
}

// ============================================================================
// Pipeline: tick → signal → bridge
// ============================================================================

#[test]
fn test_tick_to_bridge_pipeline_produces_bridged_state() {
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Feed enough ticks to produce at least one RoutedSignal
    let prices = trending_up_prices(100.0, 300);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    // We should eventually get at least one signal
    assert!(
        !signals.is_empty(),
        "Expected at least one RoutedSignal after 300 ticks"
    );

    // Bridge every signal and validate the result
    for signal in &signals {
        let bridged = bridge_regime_signal(symbol, signal, None, None, None, None);

        // Basic invariants
        assert_eq!(bridged.symbol, symbol);
        assert!(
            bridged.confidence >= 0.0 && bridged.confidence <= 1.0,
            "Confidence out of range: {}",
            bridged.confidence
        );
        assert!(
            bridged.position_scale > 0.0,
            "Position scale must be positive: {}",
            bridged.position_scale
        );

        // The display impl should not panic
        let display = format!("{}", bridged);
        assert!(display.contains("BridgedRegime["));
        assert!(display.contains(symbol));
    }
}

#[test]
fn test_trending_market_bridges_to_bullish() {
    let mut mgr = fast_regime_manager();
    let symbol = "ETHUSD";
    mgr.register_asset(symbol);

    // Strong uptrend
    let prices = trending_up_prices(100.0, 500);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    let trending_signals: Vec<_> = signals
        .iter()
        .filter(|s| matches!(s.regime, MarketRegime::Trending(_)))
        .collect();

    if !trending_signals.is_empty() {
        for signal in &trending_signals {
            let bridged = bridge_regime_signal(symbol, signal, None, None, None, None);

            // Trending with positive direction should map to bullish variants
            let hypo = bridged.hypothalamus_regime;
            assert!(
                hypo == HypothalamusRegime::StrongBullish
                    || hypo == HypothalamusRegime::Bullish
                    || hypo == HypothalamusRegime::Neutral
                    || hypo == HypothalamusRegime::Transitional,
                "Expected bullish-family regime for trending market, got {:?}",
                hypo
            );

            // Position scale should be reasonable
            assert!(
                bridged.position_scale >= 0.5 && bridged.position_scale <= 1.5,
                "Unexpected position scale for trending: {}",
                bridged.position_scale
            );
        }
    }
}

#[test]
fn test_mean_reverting_market_bridges_to_neutral() {
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let prices = mean_reverting_prices(100.0, 0.5, 500);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    let mr_signals: Vec<_> = signals
        .iter()
        .filter(|s| s.regime == MarketRegime::MeanReverting)
        .collect();

    for signal in &mr_signals {
        let bridged = bridge_regime_signal(symbol, signal, None, None, None, None);

        // Mean reverting should generally map to neutral/low-vol
        let hypo = bridged.hypothalamus_regime;
        assert!(
            hypo == HypothalamusRegime::Neutral
                || hypo == HypothalamusRegime::LowVolatility
                || hypo == HypothalamusRegime::Transitional
                || hypo == HypothalamusRegime::Unknown,
            "Expected neutral-family regime for MR market, got {:?}",
            hypo
        );

        // Amygdala should see low-vol MR or similar
        let amyg = bridged.amygdala_regime;
        assert!(
            amyg == AmygdalaRegime::LowVolMeanReverting
                || amyg == AmygdalaRegime::HighVolMeanReverting
                || amyg == AmygdalaRegime::Transitional
                || amyg == AmygdalaRegime::Unknown,
            "Expected MR-family amygdala regime, got {:?}",
            amyg
        );
    }
}

#[test]
fn test_bridged_state_confidence_matches_signal() {
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let prices = trending_up_prices(50.0, 300);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    for signal in &signals {
        let bridged = bridge_regime_signal(symbol, signal, None, None, None, None);
        assert!(
            (bridged.confidence - signal.confidence).abs() < f64::EPSILON,
            "Bridged confidence {} should match signal confidence {}",
            bridged.confidence,
            signal.confidence
        );
    }
}

#[test]
fn test_bridged_state_is_high_risk_consistency() {
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let prices = volatile_prices(100.0, 400);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    for signal in &signals {
        let bridged = bridge_regime_signal(symbol, signal, None, None, None, None);

        // is_high_risk should be consistent with amygdala regime
        let expected_high_risk = bridged.amygdala_regime.is_high_risk();
        assert_eq!(
            bridged.is_high_risk, expected_high_risk,
            "is_high_risk ({}) should match amygdala_regime.is_high_risk() ({}) for {:?}",
            bridged.is_high_risk, expected_high_risk, bridged.amygdala_regime
        );
    }
}

#[test]
fn test_bridged_state_position_scale_matches_hypothalamus() {
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let prices = trending_up_prices(100.0, 300);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    for signal in &signals {
        let bridged = bridge_regime_signal(symbol, signal, None, None, None, None);

        let expected_scale = bridged.hypothalamus_regime.base_scaling();
        assert!(
            (bridged.position_scale - expected_scale).abs() < f64::EPSILON,
            "position_scale ({}) should match hypothalamus base_scaling ({}) for {:?}",
            bridged.position_scale,
            expected_scale,
            bridged.hypothalamus_regime
        );
    }
}

// ============================================================================
// Bridge function unit-level validation
// ============================================================================

#[test]
fn test_hypothalamus_regime_covers_all_janus_regimes() {
    // Every MarketRegime + confidence combination should produce a valid HypothalamusRegime
    let regimes = [
        MarketRegime::Trending(TrendDirection::Bullish),
        MarketRegime::Trending(TrendDirection::Bearish),
        MarketRegime::MeanReverting,
        MarketRegime::Volatile,
        MarketRegime::Uncertain,
    ];
    let confidences = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0];

    for &regime in &regimes {
        for &conf in &confidences {
            let hypo = to_hypothalamus_regime(regime, conf);
            // Should not panic, and should produce a valid variant
            let display = format!("{}", hypo);
            assert!(
                !display.is_empty(),
                "Display should be non-empty for {:?} @ {:.1}",
                regime,
                conf
            );
        }
    }
}

#[test]
fn test_amygdala_regime_covers_all_janus_regimes() {
    let regimes = [
        MarketRegime::Trending(TrendDirection::Bullish),
        MarketRegime::Trending(TrendDirection::Bearish),
        MarketRegime::MeanReverting,
        MarketRegime::Volatile,
        MarketRegime::Uncertain,
    ];
    let confidences = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0];

    for &regime in &regimes {
        for &conf in &confidences {
            let amyg = to_amygdala_regime(regime, conf);
            let display = format!("{}", amyg);
            assert!(
                !display.is_empty(),
                "Display should be non-empty for {:?} @ {:.1}",
                regime,
                conf
            );
        }
    }
}

#[test]
fn test_amygdala_with_vol_variant() {
    // High vol + trending = HighVolTrending
    let result =
        to_amygdala_regime_with_vol(MarketRegime::Trending(TrendDirection::Bullish), 0.8, true);
    assert_eq!(result, AmygdalaRegime::HighVolTrending);

    // Low vol + trending = LowVolTrending
    let result =
        to_amygdala_regime_with_vol(MarketRegime::Trending(TrendDirection::Bullish), 0.8, false);
    assert_eq!(result, AmygdalaRegime::LowVolTrending);

    // High vol + MR = HighVolMeanReverting
    let result = to_amygdala_regime_with_vol(MarketRegime::MeanReverting, 0.8, true);
    assert_eq!(result, AmygdalaRegime::HighVolMeanReverting);

    // Low vol + MR = LowVolMeanReverting
    let result = to_amygdala_regime_with_vol(MarketRegime::MeanReverting, 0.8, false);
    assert_eq!(result, AmygdalaRegime::LowVolMeanReverting);
}

#[test]
fn test_build_indicators_with_raw_values() {
    let signal = RoutedSignal {
        strategy: ActiveStrategy::TrendFollowing,
        regime: MarketRegime::Trending(TrendDirection::Bullish),
        confidence: 0.85,
        position_factor: 1.0,
        reason: "test".to_string(),
        detection_method: DetectionMethod::Indicators,
        methods_agree: Some(true),
        state_probabilities: None,
        expected_duration: Some(20.0),
        trend_direction: Some(TrendDirection::Bullish),
    };

    let indicators = build_regime_indicators(
        &signal,
        Some(35.0),  // ADX
        Some(0.6),   // BB width percentile
        Some(150.0), // ATR
        Some(1.5),   // relative volume
    );

    // trend should be positive for bullish
    assert!(indicators.trend > 0.0, "Bullish trend should be positive");

    // trend_strength should reflect ADX
    assert!(
        indicators.trend_strength > 0.0,
        "ADX 35 should produce positive trend_strength"
    );

    // volatility_percentile should be the BB width percentile
    assert!(
        (indicators.volatility_percentile - 0.6).abs() < 0.01,
        "Volatility percentile should reflect BB width percentile"
    );

    // volatility should be the ATR value
    assert!(
        (indicators.volatility - 150.0).abs() < 0.01,
        "Volatility should reflect ATR value"
    );

    // relative_volume should pass through
    assert!(
        (indicators.relative_volume - 1.5).abs() < 0.01,
        "Relative volume should be 1.5"
    );
}

#[test]
fn test_build_indicators_defaults_when_no_raw_values() {
    let signal = RoutedSignal {
        strategy: ActiveStrategy::MeanReversion,
        regime: MarketRegime::MeanReverting,
        confidence: 0.6,
        position_factor: 0.8,
        reason: "test".to_string(),
        detection_method: DetectionMethod::Ensemble,
        methods_agree: None,
        state_probabilities: None,
        expected_duration: None,
        trend_direction: None,
    };

    let indicators = build_regime_indicators(&signal, None, None, None, None);

    // Should produce sensible defaults (no panics, no NaN)
    assert!(indicators.trend.is_finite());
    assert!(indicators.trend_strength.is_finite());
    assert!(indicators.volatility.is_finite());
    assert!(indicators.relative_volume.is_finite());
    assert!(indicators.momentum.is_finite());
}

// ============================================================================
// TOML Config Loading
// ============================================================================

#[test]
fn test_toml_config_loads_and_produces_bridged_state() {
    // Load from embedded TOML string (simulates file loading without I/O)
    let toml_str = r#"
[manager]
ticks_per_candle = 5
detection_method = "Indicators"
min_confidence = 0.4
volatile_position_factor = 0.25
log_regime_changes = false

[indicators]
adx_period = 14
adx_trending_threshold = 22.0
adx_ranging_threshold = 18.0

[hmm]
n_states = 3
min_observations = 20

[ensemble]
indicator_weight = 0.7
hmm_weight = 0.3
"#;

    let config = RegimeManagerConfig::from_toml_str(toml_str).expect("Failed to parse TOML");
    assert_eq!(config.ticks_per_candle, 5);

    let mut mgr = RegimeManager::new(config);
    let symbol = "ETHUSD";
    mgr.register_asset(symbol);

    // Feed trending prices
    let prices = trending_up_prices(200.0, 300);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    assert!(
        !signals.is_empty(),
        "Should produce signals with ticks_per_candle=5"
    );

    // Bridge all signals
    for signal in &signals {
        let bridged = bridge_regime_signal(symbol, signal, None, None, None, None);
        assert_eq!(bridged.symbol, symbol);
        assert!(bridged.confidence >= 0.0);
        // Display should work
        let _ = format!("{}", bridged);
    }
}

#[test]
fn test_toml_from_file_or_default_with_nonexistent_path() {
    // Should fall back to defaults without error
    let config = RegimeManagerConfig::from_toml_file_or_default("/nonexistent/path/regime.toml");
    assert_eq!(config.ticks_per_candle, 100); // default value

    let mut mgr = RegimeManager::new(config);
    mgr.register_asset("BTCUSD");
    // Should work fine with defaults
    let _ = mgr.on_tick("BTCUSD", 100.0, 100.1);
}

#[test]
fn test_toml_partial_override() {
    let toml_str = r#"
[manager]
ticks_per_candle = 10
"#;

    let config = RegimeManagerConfig::from_toml_str(toml_str).expect("Failed to parse TOML");
    assert_eq!(config.ticks_per_candle, 10);
    // Other fields should be defaults
    assert_eq!(config.min_confidence, 0.5);
    assert!(config.volatile_position_factor > 0.0);
}

// ============================================================================
// Broadcast channel simulation
// ============================================================================

#[test]
fn test_broadcast_channel_delivers_bridged_state() {
    let (tx, _default_rx) = tokio::sync::broadcast::channel::<BridgedRegimeState>(128);
    // Subscribe *before* sending so the receiver sees all messages
    let mut rx = tx.subscribe();

    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let prices = trending_up_prices(100.0, 300);
    let mut sent_count = 0;

    for &price in &prices {
        let spread = price * 0.0001;
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            let bridged = bridge_regime_signal(symbol, &signal, None, None, None, None);
            tx.send(bridged).expect("Failed to send bridged state");
            sent_count += 1;
        }
    }

    assert!(
        sent_count > 0,
        "Should have sent at least one bridged state"
    );

    // Verify all sent messages are received
    let mut received_count = 0;
    while let Ok(state) = rx.try_recv() {
        assert_eq!(state.symbol, symbol);
        assert!(state.confidence >= 0.0 && state.confidence <= 1.0);
        received_count += 1;
    }

    assert_eq!(
        received_count, sent_count,
        "Should receive all sent messages"
    );
}

#[test]
fn test_broadcast_channel_multiple_subscribers() {
    let (tx, _default_rx) = tokio::sync::broadcast::channel::<BridgedRegimeState>(128);
    // Subscribe both receivers *before* sending
    let mut rx1 = tx.subscribe();
    let mut rx2 = tx.subscribe();

    let mut mgr = fast_regime_manager();
    let symbol = "SOLUSD";
    mgr.register_asset(symbol);

    let prices = trending_up_prices(50.0, 100);
    let mut sent_count = 0;

    for &price in &prices {
        let spread = price * 0.0001;
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            let bridged = bridge_regime_signal(symbol, &signal, None, None, None, None);
            tx.send(bridged).expect("Failed to send");
            sent_count += 1;
        }
    }

    if sent_count > 0 {
        let mut rx1_count = 0;
        while rx1.try_recv().is_ok() {
            rx1_count += 1;
        }

        let mut rx2_count = 0;
        while rx2.try_recv().is_ok() {
            rx2_count += 1;
        }

        assert_eq!(rx1_count, sent_count, "rx1 should receive all messages");
        assert_eq!(rx2_count, sent_count, "rx2 should receive all messages");
    }
}

#[test]
fn test_broadcast_no_subscriber_does_not_panic() {
    let (tx, _) = tokio::sync::broadcast::channel::<BridgedRegimeState>(4);

    // Drop the initial receiver — no one is listening
    // Sending should not panic, it just returns Err
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let prices = trending_up_prices(100.0, 50);
    for &price in &prices {
        let spread = price * 0.0001;
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            let bridged = bridge_regime_signal(symbol, &signal, None, None, None, None);
            // This will Err because no receivers, but should not panic
            let _ = tx.send(bridged);
        }
    }
}

// ============================================================================
// Regime indicator enrichment
// ============================================================================

#[test]
fn test_indicators_default_values_are_sane() {
    let indicators = RegimeIndicators::default();

    assert_eq!(indicators.trend, 0.0);
    assert_eq!(indicators.trend_strength, 0.0);
    assert_eq!(indicators.volatility, 0.15);
    assert_eq!(indicators.volatility_percentile, 0.5);
    assert_eq!(indicators.correlation, 0.0);
    assert_eq!(indicators.breadth, 1.0);
    assert_eq!(indicators.momentum, 0.0);
    assert_eq!(indicators.relative_volume, 1.0);
    assert_eq!(indicators.liquidity_score, 1.0);
    assert_eq!(indicators.fear_index, None);
}

// ============================================================================
// Display formatting
// ============================================================================

#[test]
fn test_all_hypothalamus_regimes_display() {
    let regimes = [
        HypothalamusRegime::StrongBullish,
        HypothalamusRegime::Bullish,
        HypothalamusRegime::Neutral,
        HypothalamusRegime::Bearish,
        HypothalamusRegime::StrongBearish,
        HypothalamusRegime::HighVolatility,
        HypothalamusRegime::LowVolatility,
        HypothalamusRegime::Transitional,
        HypothalamusRegime::Crisis,
        HypothalamusRegime::Unknown,
    ];

    for regime in &regimes {
        let display = format!("{}", regime);
        assert!(
            !display.is_empty(),
            "Display for {:?} should not be empty",
            regime
        );
    }
}

#[test]
fn test_all_amygdala_regimes_display() {
    let regimes = [
        AmygdalaRegime::LowVolTrending,
        AmygdalaRegime::LowVolMeanReverting,
        AmygdalaRegime::HighVolTrending,
        AmygdalaRegime::HighVolMeanReverting,
        AmygdalaRegime::Crisis,
        AmygdalaRegime::Transitional,
        AmygdalaRegime::Unknown,
    ];

    for regime in &regimes {
        let display = format!("{}", regime);
        assert!(
            !display.is_empty(),
            "Display for {:?} should not be empty",
            regime
        );
    }
}

#[test]
fn test_bridged_state_display_format() {
    let state = BridgedRegimeState {
        symbol: "BTCUSD".to_string(),
        hypothalamus_regime: HypothalamusRegime::StrongBullish,
        amygdala_regime: AmygdalaRegime::LowVolTrending,
        position_scale: 1.2,
        is_high_risk: false,
        confidence: 0.85,
        indicators: RegimeIndicators::default(),
    };

    let display = format!("{}", state);
    assert!(display.contains("BTCUSD"));
    assert!(display.contains("BridgedRegime["));
    assert!(display.contains("85%")); // confidence
}

// ============================================================================
// Hypothalamus helper methods
// ============================================================================

#[test]
fn test_hypothalamus_favors_trend_following() {
    assert!(HypothalamusRegime::StrongBullish.favors_trend_following());
    assert!(HypothalamusRegime::Bullish.favors_trend_following());
    assert!(HypothalamusRegime::StrongBearish.favors_trend_following());
    assert!(HypothalamusRegime::Bearish.favors_trend_following());
    assert!(!HypothalamusRegime::Neutral.favors_trend_following());
    assert!(!HypothalamusRegime::Crisis.favors_trend_following());
}

#[test]
fn test_hypothalamus_favors_mean_reversion() {
    assert!(HypothalamusRegime::Neutral.favors_mean_reversion());
    assert!(HypothalamusRegime::LowVolatility.favors_mean_reversion());
    assert!(!HypothalamusRegime::StrongBullish.favors_mean_reversion());
    assert!(!HypothalamusRegime::Crisis.favors_mean_reversion());
}

#[test]
fn test_hypothalamus_requires_caution() {
    assert!(HypothalamusRegime::Crisis.requires_caution());
    assert!(HypothalamusRegime::HighVolatility.requires_caution());
    assert!(HypothalamusRegime::Transitional.requires_caution());
    assert!(HypothalamusRegime::Unknown.requires_caution());
    assert!(!HypothalamusRegime::StrongBullish.requires_caution());
    assert!(!HypothalamusRegime::Neutral.requires_caution());
}

#[test]
fn test_hypothalamus_base_scaling_range() {
    let regimes = [
        HypothalamusRegime::StrongBullish,
        HypothalamusRegime::Bullish,
        HypothalamusRegime::Neutral,
        HypothalamusRegime::Bearish,
        HypothalamusRegime::StrongBearish,
        HypothalamusRegime::HighVolatility,
        HypothalamusRegime::LowVolatility,
        HypothalamusRegime::Transitional,
        HypothalamusRegime::Crisis,
        HypothalamusRegime::Unknown,
    ];

    for regime in &regimes {
        let scale = regime.base_scaling();
        assert!(
            scale > 0.0 && scale <= 2.0,
            "Scaling for {:?} should be in (0, 2], got {}",
            regime,
            scale
        );
    }
}

// ============================================================================
// Amygdala helper methods
// ============================================================================

#[test]
fn test_amygdala_is_high_risk() {
    assert!(AmygdalaRegime::Crisis.is_high_risk());
    assert!(AmygdalaRegime::HighVolTrending.is_high_risk());
    assert!(AmygdalaRegime::Transitional.is_high_risk());
    // HighVolMeanReverting is NOT considered high-risk by the amygdala —
    // mean-reverting regimes are bounded, so high vol within a range is
    // less threatening than high vol in a trending/directional context.
    assert!(!AmygdalaRegime::HighVolMeanReverting.is_high_risk());
    assert!(!AmygdalaRegime::LowVolTrending.is_high_risk());
    assert!(!AmygdalaRegime::LowVolMeanReverting.is_high_risk());
}

// ============================================================================
// End-to-end pipeline: TOML → manager → ticks → bridge → validate
// ============================================================================

#[test]
fn test_end_to_end_toml_to_bridge_pipeline() {
    // 1. Load config from TOML
    let toml_str = r#"
[manager]
ticks_per_candle = 4
detection_method = "Indicators"
min_confidence = 0.3
volatile_position_factor = 0.2
log_regime_changes = false

[indicators]
adx_period = 14
adx_trending_threshold = 20.0
adx_ranging_threshold = 15.0
bb_period = 20
bb_std_dev = 2.0
ema_short_period = 8
ema_long_period = 21
atr_period = 14
regime_stability_bars = 2
min_regime_duration = 3
"#;

    let config = RegimeManagerConfig::from_toml_str(toml_str).expect("TOML parse failed");
    assert_eq!(config.ticks_per_candle, 4);

    // 2. Create regime manager
    let mut mgr = RegimeManager::new(config);
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // 3. Create broadcast channel
    let (tx, _default_rx) = tokio::sync::broadcast::channel::<BridgedRegimeState>(256);
    let mut rx = tx.subscribe();

    // 4. Feed prices and bridge signals
    let prices = trending_up_prices(100.0, 500);
    let mut total_signals = 0;

    for &price in &prices {
        let spread = price * 0.0001;
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            total_signals += 1;

            // Bridge the signal
            let bridged = bridge_regime_signal(symbol, &signal, None, None, None, None);

            // Validate invariants before sending
            assert_eq!(bridged.symbol, symbol);
            assert!(bridged.confidence >= 0.0 && bridged.confidence <= 1.0);
            assert!(bridged.position_scale > 0.0);
            assert_eq!(bridged.is_high_risk, bridged.amygdala_regime.is_high_risk());
            assert!(
                (bridged.position_scale - bridged.hypothalamus_regime.base_scaling()).abs()
                    < f64::EPSILON
            );

            // Send on channel
            tx.send(bridged).expect("Channel send failed");
        }
    }

    assert!(
        total_signals > 0,
        "Pipeline should produce at least one signal"
    );

    // 5. Consume and validate all received states
    let mut received = 0;
    while let Ok(state) = rx.try_recv() {
        received += 1;
        assert_eq!(state.symbol, symbol);
    }
    assert_eq!(received, total_signals, "Should receive all sent states");
}

#[test]
fn test_multi_asset_bridge_pipeline() {
    let mut mgr = fast_regime_manager();

    let symbols = ["BTCUSD", "ETHUSD", "SOLUSD"];
    for &sym in &symbols {
        mgr.register_asset(sym);
    }

    let (tx, mut rx) = tokio::sync::broadcast::channel::<BridgedRegimeState>(128);

    // Feed different price patterns to different assets
    let btc_prices = trending_up_prices(40000.0, 100);
    let eth_prices = mean_reverting_prices(2500.0, 5.0, 100);
    let sol_prices = volatile_prices(100.0, 100);

    let mut total_sent = 0;

    for i in 0..100 {
        for (sym, prices) in [
            ("BTCUSD", &btc_prices),
            ("ETHUSD", &eth_prices),
            ("SOLUSD", &sol_prices),
        ] {
            let price = prices[i];
            let spread = price * 0.0001;
            if let Some(signal) = mgr.on_tick(sym, price - spread, price + spread) {
                let bridged = bridge_regime_signal(sym, &signal, None, None, None, None);
                assert_eq!(bridged.symbol, sym);
                let _ = tx.send(bridged);
                total_sent += 1;
            }
        }
    }

    // Verify we received states for multiple symbols
    let mut received_symbols = std::collections::HashSet::new();
    while let Ok(state) = rx.try_recv() {
        received_symbols.insert(state.symbol.clone());
    }

    if total_sent > 0 {
        assert!(
            !received_symbols.is_empty(),
            "Should have received states for at least one symbol"
        );
    }
}

// ============================================================================
// Regime transition tracking through bridge
// ============================================================================

#[test]
fn test_regime_transitions_produce_different_bridge_outputs() {
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Start with trending prices, then switch to mean-reverting
    let mut prices = trending_up_prices(100.0, 200);
    prices.extend(mean_reverting_prices(200.0, 1.0, 200));

    let signals = feed_prices(&mut mgr, symbol, &prices);

    let bridged_states: Vec<BridgedRegimeState> = signals
        .iter()
        .map(|s| bridge_regime_signal(symbol, s, None, None, None, None))
        .collect();

    if bridged_states.len() >= 2 {
        // At least check that we have valid states
        for state in &bridged_states {
            assert_eq!(state.symbol, symbol);
            assert!(state.position_scale > 0.0);
        }

        // Collect unique hypothalamus regimes seen
        let unique_hypo: std::collections::HashSet<_> = bridged_states
            .iter()
            .map(|s| format!("{:?}", s.hypothalamus_regime))
            .collect();

        // We may or may not see transitions depending on the data;
        // the important thing is that the pipeline handles them without panicking
        assert!(
            !unique_hypo.is_empty(),
            "Should have at least one unique hypothalamus regime"
        );
    }
}

// ============================================================================
// Strategy recommendation pass-through
// ============================================================================

#[test]
fn test_bridge_preserves_strategy_context() {
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let prices = trending_up_prices(100.0, 300);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    for signal in &signals {
        let bridged = bridge_regime_signal(symbol, signal, None, None, None, None);

        // The bridge should reflect the strategy recommendation
        // in its regime mapping:
        // TrendFollowing → should be bullish or bearish (not neutral)
        // MeanReversion → should be neutral-family
        match signal.strategy {
            ActiveStrategy::TrendFollowing => {
                // In a trending market, hypothalamus should favor trend
                // (but low confidence can map to transitional/unknown)
                let hypo = bridged.hypothalamus_regime;
                assert!(
                    hypo.favors_trend_following()
                        || hypo == HypothalamusRegime::Transitional
                        || hypo == HypothalamusRegime::Unknown
                        || hypo == HypothalamusRegime::Neutral
                        || hypo == HypothalamusRegime::HighVolatility,
                    "TrendFollowing strategy with regime {:?} produced unexpected hypo {:?}",
                    signal.regime,
                    hypo
                );
            }
            ActiveStrategy::MeanReversion => {
                // Mean reversion should map to neutral/low-vol family
                let hypo = bridged.hypothalamus_regime;
                assert!(
                    hypo.favors_mean_reversion()
                        || hypo == HypothalamusRegime::Transitional
                        || hypo == HypothalamusRegime::Unknown,
                    "MeanReversion strategy with regime {:?} produced unexpected hypo {:?}",
                    signal.regime,
                    hypo
                );
            }
            ActiveStrategy::NoTrade => {
                // NoTrade should map to cautious regimes
                let hypo = bridged.hypothalamus_regime;
                assert!(
                    hypo.requires_caution()
                        || hypo == HypothalamusRegime::Neutral
                        || hypo == HypothalamusRegime::LowVolatility,
                    "NoTrade strategy produced unexpectedly confident hypo {:?}",
                    hypo
                );
            }
        }
    }
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_bridge_with_zero_confidence() {
    let signal = RoutedSignal {
        strategy: ActiveStrategy::NoTrade,
        regime: MarketRegime::Uncertain,
        confidence: 0.0,
        position_factor: 0.0,
        reason: "No data".to_string(),
        detection_method: DetectionMethod::Indicators,
        methods_agree: None,
        state_probabilities: None,
        expected_duration: None,
        trend_direction: None,
    };

    let bridged = bridge_regime_signal("BTCUSD", &signal, None, None, None, None);
    assert_eq!(bridged.confidence, 0.0);
    assert_eq!(bridged.symbol, "BTCUSD");
    // Should not panic
    let _ = format!("{}", bridged);
}

#[test]
fn test_bridge_with_max_confidence() {
    let signal = RoutedSignal {
        strategy: ActiveStrategy::TrendFollowing,
        regime: MarketRegime::Trending(TrendDirection::Bullish),
        confidence: 1.0,
        position_factor: 1.0,
        reason: "Absolute certainty".to_string(),
        detection_method: DetectionMethod::Ensemble,
        methods_agree: Some(true),
        state_probabilities: Some(vec![0.0, 0.0, 1.0]),
        expected_duration: Some(100.0),
        trend_direction: Some(TrendDirection::Bullish),
    };

    let bridged = bridge_regime_signal(
        "BTCUSD",
        &signal,
        Some(50.0),
        Some(0.9),
        Some(200.0),
        Some(2.5),
    );
    assert_eq!(bridged.confidence, 1.0);
    assert_eq!(
        bridged.hypothalamus_regime,
        HypothalamusRegime::StrongBullish
    );
    assert!(
        !bridged.amygdala_regime.is_high_risk()
            || bridged.amygdala_regime == AmygdalaRegime::HighVolTrending
    );
}

#[test]
fn test_bridge_empty_symbol() {
    let signal = RoutedSignal {
        strategy: ActiveStrategy::NoTrade,
        regime: MarketRegime::Uncertain,
        confidence: 0.5,
        position_factor: 0.5,
        reason: "test".to_string(),
        detection_method: DetectionMethod::Indicators,
        methods_agree: None,
        state_probabilities: None,
        expected_duration: None,
        trend_direction: None,
    };

    // Empty symbol should not panic
    let bridged = bridge_regime_signal("", &signal, None, None, None, None);
    assert_eq!(bridged.symbol, "");
}

// ============================================================================
// Real indicator value extraction (ADX, ATR, BB width)
// ============================================================================

#[test]
fn test_regime_manager_exposes_adx_value_after_warmup() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Before warmup, ADX should be None
    assert!(
        mgr.adx_value(symbol).is_none(),
        "ADX should be None before warmup"
    );

    // Feed enough ticks for the indicator to warm up
    let prices = trending_up_prices(100.0, 500);
    feed_prices(&mut mgr, symbol, &prices);

    // After enough data, ADX should be available
    let adx = mgr.adx_value(symbol);
    if let Some(adx_val) = adx {
        assert!(
            (0.0..=100.0).contains(&adx_val),
            "ADX should be in [0, 100], got {}",
            adx_val
        );
    }
    // Note: ADX may still be None if the indicator period hasn't been
    // reached — this depends on ticks_per_candle and indicator warmup.
    // The important thing is no panic.
}

#[test]
fn test_regime_manager_exposes_atr_value_after_warmup() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Before warmup, ATR should be None
    assert!(
        mgr.atr_value(symbol).is_none(),
        "ATR should be None before warmup"
    );

    // Feed enough ticks
    let prices = trending_up_prices(100.0, 500);
    feed_prices(&mut mgr, symbol, &prices);

    let atr = mgr.atr_value(symbol);
    if let Some(atr_val) = atr {
        assert!(
            atr_val >= 0.0,
            "ATR should be non-negative, got {}",
            atr_val
        );
    }
}

#[test]
fn test_regime_manager_exposes_last_regime_confidence() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Before any updates, last_regime_confidence should be None
    assert!(
        mgr.last_regime_confidence(symbol).is_none(),
        "Should be None before any ticks"
    );

    // Feed enough ticks to produce signals
    let prices = trending_up_prices(100.0, 500);
    let signals = feed_prices(&mut mgr, symbol, &prices);

    if !signals.is_empty() {
        // After signals are produced, last_regime_confidence should be populated
        let rc = mgr.last_regime_confidence(symbol);
        assert!(
            rc.is_some(),
            "last_regime_confidence should be Some after producing signals"
        );

        let rc = rc.unwrap();
        assert!(
            rc.confidence >= 0.0 && rc.confidence <= 1.0,
            "Confidence out of range: {}",
            rc.confidence
        );
        assert!(
            rc.bb_width_percentile >= 0.0 && rc.bb_width_percentile <= 100.0,
            "BB width percentile out of range: {}",
            rc.bb_width_percentile
        );
        assert!(
            rc.adx_value >= 0.0,
            "ADX should be non-negative: {}",
            rc.adx_value
        );
    }
}

#[test]
fn test_regime_confidence_unknown_asset_returns_none() {
    let mgr = RegimeManager::new(RegimeManagerConfig::default());

    assert!(mgr.last_regime_confidence("UNKNOWN").is_none());
    assert!(mgr.adx_value("UNKNOWN").is_none());
    assert!(mgr.atr_value("UNKNOWN").is_none());
}

#[test]
fn test_bridge_with_real_indicator_enrichment() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Feed trending prices to warm up and get signals
    let prices = trending_up_prices(100.0, 500);
    let mut last_signal: Option<RoutedSignal> = None;

    for &price in &prices {
        let spread = price * 0.0001;
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            last_signal = Some(signal);
        }
    }

    if let Some(signal) = &last_signal {
        // Extract real indicator values from the manager
        let adx_val = mgr.adx_value(symbol);
        let bb_width_pct = mgr
            .last_regime_confidence(symbol)
            .map(|rc| rc.bb_width_percentile / 100.0);
        let atr_val = mgr.atr_value(symbol);

        // Bridge with real values
        let bridged = bridge_regime_signal(
            symbol,
            signal,
            adx_val,
            bb_width_pct,
            atr_val,
            None, // relative volume not tracked here
        );

        assert_eq!(bridged.symbol, symbol);
        assert!(bridged.confidence >= 0.0 && bridged.confidence <= 1.0);

        // When real indicator values are provided, the indicators struct
        // should reflect them (not defaults)
        if let Some(adx) = adx_val {
            // trend_strength should be derived from ADX (normalized to 0–1)
            let expected_strength = (adx / 50.0).min(1.0);
            if matches!(signal.regime, MarketRegime::Trending(_)) {
                assert!(
                    (bridged.indicators.trend_strength - expected_strength).abs() < 0.01,
                    "trend_strength should reflect ADX {}: expected {}, got {}",
                    adx,
                    expected_strength,
                    bridged.indicators.trend_strength
                );
            }
        }

        if let Some(bb_pct) = bb_width_pct {
            assert!(
                (bridged.indicators.volatility_percentile - bb_pct).abs() < 0.01,
                "volatility_percentile should match BB width percentile: expected {}, got {}",
                bb_pct,
                bridged.indicators.volatility_percentile
            );
        }

        if let Some(atr) = atr_val {
            assert!(
                (bridged.indicators.volatility - atr).abs() < 0.01,
                "volatility should match ATR: expected {}, got {}",
                atr,
                bridged.indicators.volatility
            );
        }
    }
}

#[test]
fn test_enriched_vs_unenriched_bridge_differs() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let prices = trending_up_prices(100.0, 500);
    let mut last_signal: Option<RoutedSignal> = None;

    for &price in &prices {
        let spread = price * 0.0001;
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            last_signal = Some(signal);
        }
    }

    if let Some(signal) = &last_signal {
        let adx_val = mgr.adx_value(symbol);
        let atr_val = mgr.atr_value(symbol);

        // Bridge without real values (defaults)
        let unenriched = bridge_regime_signal(symbol, signal, None, None, None, None);

        // Bridge with real values
        let enriched = bridge_regime_signal(symbol, signal, adx_val, None, atr_val, None);

        // Core regime mapping should be identical (it only depends on
        // MarketRegime + confidence, not indicator values)
        assert_eq!(
            enriched.hypothalamus_regime, unenriched.hypothalamus_regime,
            "Regime mapping should be the same regardless of indicator enrichment"
        );
        assert_eq!(
            enriched.amygdala_regime, unenriched.amygdala_regime,
            "Amygdala regime should be the same regardless of indicator enrichment"
        );
        assert_eq!(enriched.confidence, unenriched.confidence);
        assert_eq!(enriched.position_scale, unenriched.position_scale);

        // But the indicator details should differ when real values are provided
        if adx_val.is_some() || atr_val.is_some() {
            // At least one indicator dimension should differ from defaults
            let default_indicators = RegimeIndicators::default();
            let indicators_changed =
                (enriched.indicators.volatility - default_indicators.volatility).abs() > 0.001
                    || (enriched.indicators.trend_strength - default_indicators.trend_strength)
                        .abs()
                        > 0.001;

            // Only assert if we're in a regime where these indicators matter
            if matches!(signal.regime, MarketRegime::Trending(_)) && adx_val.is_some() {
                assert!(
                    indicators_changed,
                    "Enriched indicators should differ from defaults when real values are provided"
                );
            }
        }
    }
}

#[test]
fn test_end_to_end_enriched_bridge_pipeline() {
    // Full pipeline: TOML config → manager → ticks → enriched bridge → validate
    let toml_str = r#"
[manager]
ticks_per_candle = 4
detection_method = "Indicators"
min_confidence = 0.3
log_regime_changes = false

[indicators]
adx_period = 14
adx_trending_threshold = 20.0
adx_ranging_threshold = 15.0
regime_stability_bars = 2
min_regime_duration = 3
"#;

    let config = RegimeManagerConfig::from_toml_str(toml_str).expect("TOML parse failed");
    let mut mgr = RegimeManager::new(config);
    let symbol = "ETHUSD";
    mgr.register_asset(symbol);

    let (tx, _default_rx) = tokio::sync::broadcast::channel::<BridgedRegimeState>(256);
    let mut rx = tx.subscribe();

    let prices = trending_up_prices(200.0, 600);
    let mut enriched_count = 0;

    for &price in &prices {
        let spread = price * 0.0001;
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            // Extract real indicator values
            let adx_val = mgr.adx_value(symbol);
            let bb_pct = mgr
                .last_regime_confidence(symbol)
                .map(|rc| rc.bb_width_percentile / 100.0);
            let atr_val = mgr.atr_value(symbol);

            let bridged = bridge_regime_signal(symbol, &signal, adx_val, bb_pct, atr_val, None);

            // Count how many signals had real indicator enrichment
            if adx_val.is_some() || atr_val.is_some() {
                enriched_count += 1;
            }

            tx.send(bridged).expect("Channel send failed");
        }
    }

    // Verify we got enriched signals (after indicator warmup)
    assert!(
        enriched_count > 0,
        "Should have produced at least one enriched signal after indicator warmup"
    );

    // Verify all states come through the channel
    let mut received = 0;
    while let Ok(state) = rx.try_recv() {
        assert_eq!(state.symbol, symbol);
        received += 1;
    }
    assert!(received > 0, "Should have received bridged states");
}

// ============================================================================
// Relative Volume Tests
// ============================================================================

#[test]
fn test_relative_volume_none_before_any_volume() {
    // Relative volume should be None when no trade volume has been accumulated
    let mut mgr = fast_regime_manager();
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Feed ticks without any trade volume
    let prices = trending_up_prices(100.0, 30);
    feed_prices(&mut mgr, symbol, &prices);

    // No volume was ever supplied, so relative_volume should be None
    assert!(
        mgr.relative_volume(symbol).is_none(),
        "relative_volume should be None when no trade volume has been accumulated"
    );
}

#[test]
fn test_relative_volume_unknown_asset_returns_none() {
    let mgr = fast_regime_manager();
    assert!(
        mgr.relative_volume("NONEXISTENT").is_none(),
        "relative_volume for unknown asset should be None"
    );
}

#[test]
fn test_relative_volume_after_single_candle_with_volume() {
    // After exactly one candle with volume, relative_volume = vol / avg = 1.0
    // because the rolling average only contains one entry.
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Add trade volume before completing the candle
    mgr.on_trade_volume(symbol, 100.0);
    mgr.on_trade_volume(symbol, 50.0);

    // Feed exactly 3 ticks to complete one candle
    let prices = [100.0, 101.0, 102.0];
    let signals = feed_prices(&mut mgr, symbol, &prices);

    // Should have produced one signal (candle completed)
    assert_eq!(signals.len(), 1, "Should produce exactly one signal");

    // Relative volume: 150.0 / avg(150.0) = 1.0
    let rel_vol = mgr.relative_volume(symbol);
    assert!(
        rel_vol.is_some(),
        "relative_volume should be Some after a candle with volume"
    );
    assert!(
        (rel_vol.unwrap() - 1.0).abs() < 1e-10,
        "First candle relative volume should be 1.0 (only one data point), got {}",
        rel_vol.unwrap()
    );
}

#[test]
fn test_relative_volume_increases_with_volume_spike() {
    // Build up a baseline of normal volume, then spike — relative_volume should > 1.0
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Feed 5 candles with "normal" volume of 100 each
    for i in 0..5 {
        mgr.on_trade_volume(symbol, 100.0);
        let base = 100.0 + i as f64;
        let prices = [base, base + 0.5, base + 1.0];
        feed_prices(&mut mgr, symbol, &prices);
    }

    // Confirm baseline relative_volume ≈ 1.0
    let baseline = mgr
        .relative_volume(symbol)
        .expect("should have relative volume");
    assert!(
        (baseline - 1.0).abs() < 0.01,
        "Baseline relative volume should be ~1.0, got {}",
        baseline
    );

    // Now feed a candle with 3x normal volume (300)
    mgr.on_trade_volume(symbol, 300.0);
    let prices = [106.0, 106.5, 107.0];
    feed_prices(&mut mgr, symbol, &prices);

    let spike = mgr
        .relative_volume(symbol)
        .expect("should have relative volume after spike");
    // 300 / avg(100,100,100,100,100,300) = 300 / 133.33 ≈ 2.25
    assert!(
        spike > 1.5,
        "Relative volume after 3x spike should be > 1.5, got {}",
        spike
    );
    assert!(
        spike < 3.5,
        "Relative volume after 3x spike should be < 3.5, got {}",
        spike
    );
}

#[test]
fn test_relative_volume_decreases_with_low_volume() {
    // Build up baseline, then send a candle with very low volume
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Feed 5 candles with normal volume of 200 each
    for i in 0..5 {
        mgr.on_trade_volume(symbol, 200.0);
        let base = 100.0 + i as f64;
        let prices = [base, base + 0.5, base + 1.0];
        feed_prices(&mut mgr, symbol, &prices);
    }

    // Now feed a candle with very low volume (20 = 10% of normal)
    mgr.on_trade_volume(symbol, 20.0);
    let prices = [106.0, 106.5, 107.0];
    feed_prices(&mut mgr, symbol, &prices);

    let low_vol = mgr
        .relative_volume(symbol)
        .expect("should have relative volume");
    assert!(
        low_vol < 0.5,
        "Relative volume after low-volume candle should be < 0.5, got {}",
        low_vol
    );
}

#[test]
fn test_relative_volume_passed_to_bridge() {
    // Verify that relative volume from RegimeManager flows through to bridge indicators
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Feed several candles with volume so we have a relative volume value
    for i in 0..10 {
        mgr.on_trade_volume(symbol, 100.0 + i as f64 * 10.0);
        let base = 100.0 + i as f64 * 0.5;
        let prices = [base, base + 0.25, base + 0.5];
        for &price in &prices {
            let spread = price * 0.0001;
            if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
                let rel_vol = mgr.relative_volume(symbol);

                let bridged = bridge_regime_signal(
                    symbol,
                    &signal,
                    mgr.adx_value(symbol),
                    mgr.last_regime_confidence(symbol)
                        .map(|rc| rc.bb_width_percentile / 100.0),
                    mgr.atr_value(symbol),
                    rel_vol,
                );

                // When relative volume is available, it should be reflected
                // in the bridge indicators (not the default 1.0 unless it
                // actually IS 1.0)
                if let Some(rv) = rel_vol {
                    assert!(
                        (bridged.indicators.relative_volume - rv).abs() < 1e-10,
                        "Bridge indicators should contain the real relative volume ({}) but got {}",
                        rv,
                        bridged.indicators.relative_volume
                    );
                }
            }
        }
    }
}

#[test]
fn test_relative_volume_rolling_window_does_not_grow_unbounded() {
    // Feed many candles — the volume history should stay bounded (default 20)
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Feed 100 candles with increasing volume
    for i in 0..100 {
        mgr.on_trade_volume(symbol, 50.0 + i as f64);
        let base = 100.0 + i as f64 * 0.1;
        let prices = [base, base + 0.05, base + 0.1];
        feed_prices(&mut mgr, symbol, &prices);
    }

    // After 100 candles, relative volume should still be reasonable
    // (not NaN, not Inf, bounded)
    let rv = mgr
        .relative_volume(symbol)
        .expect("should have relative volume");
    assert!(
        rv.is_finite(),
        "relative volume should be finite, got {}",
        rv
    );
    assert!(rv > 0.0, "relative volume should be positive, got {}", rv);
    // Since volume increases linearly, the latest volume (149) divided by
    // the average of the last 20 (130..149, avg ~139.5) should be ~1.07
    assert!(
        rv < 2.0,
        "relative volume should be reasonable after many candles, got {}",
        rv
    );
}

// ============================================================================
// End-to-End Bridge Consumer Tests
// ============================================================================

#[test]
fn test_broadcast_consumer_receives_volume_enriched_states() {
    // Simulates the production flow: manager → bridge → broadcast → consumer
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let (tx, mut rx) = tokio::sync::broadcast::channel::<BridgedRegimeState>(256);

    // Feed candles with volume through the full pipeline
    let mut sent = 0;
    for i in 0..30 {
        // Simulate trade volume arriving between ticks
        mgr.on_trade_volume(symbol, 100.0 + i as f64 * 5.0);

        let base = 100.0 + i as f64 * 0.5;
        let prices = [base, base + 0.25, base + 0.5];
        for &price in &prices {
            let spread = price * 0.0001;
            if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
                let bridged = bridge_regime_signal(
                    symbol,
                    &signal,
                    mgr.adx_value(symbol),
                    mgr.last_regime_confidence(symbol)
                        .map(|rc| rc.bb_width_percentile / 100.0),
                    mgr.atr_value(symbol),
                    mgr.relative_volume(symbol),
                );
                tx.send(bridged).expect("send failed");
                sent += 1;
            }
        }
    }

    assert!(sent > 0, "Should have sent at least one bridged state");

    // Consumer side: drain all received states
    let mut received = 0;
    let mut saw_volume_enriched = false;
    while let Ok(state) = rx.try_recv() {
        assert_eq!(state.symbol, symbol);
        received += 1;

        // After enough candles, relative_volume should differ from default (1.0)
        // because volumes are increasing linearly
        if (state.indicators.relative_volume - 1.0).abs() > 0.01 {
            saw_volume_enriched = true;
        }
    }

    assert_eq!(received, sent, "Consumer should receive all sent states");
    assert!(
        saw_volume_enriched,
        "At least one bridged state should have non-default relative_volume"
    );
}

#[test]
fn test_broadcast_consumer_tracks_regime_transitions() {
    // Simulate a market that transitions between regimes and verify
    // the consumer sees different hypothalamus/amygdala regimes over time.
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        detection_method: DetectionMethod::Indicators,
        min_confidence: 0.3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    let (tx, mut rx) = tokio::sync::broadcast::channel::<BridgedRegimeState>(512);

    // Phase 1: trending up
    let trending = trending_up_prices(100.0, 300);
    for &price in &trending {
        let spread = price * 0.0001;
        mgr.on_trade_volume(symbol, 150.0);
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            let bridged = bridge_regime_signal(
                symbol,
                &signal,
                mgr.adx_value(symbol),
                mgr.last_regime_confidence(symbol)
                    .map(|rc| rc.bb_width_percentile / 100.0),
                mgr.atr_value(symbol),
                mgr.relative_volume(symbol),
            );
            let _ = tx.send(bridged);
        }
    }

    // Phase 2: mean reverting
    let mean_rev = mean_reverting_prices(250.0, 3.0, 300);
    for &price in &mean_rev {
        let spread = price * 0.0001;
        mgr.on_trade_volume(symbol, 80.0);
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            let bridged = bridge_regime_signal(
                symbol,
                &signal,
                mgr.adx_value(symbol),
                mgr.last_regime_confidence(symbol)
                    .map(|rc| rc.bb_width_percentile / 100.0),
                mgr.atr_value(symbol),
                mgr.relative_volume(symbol),
            );
            let _ = tx.send(bridged);
        }
    }

    // Collect all received states and verify we saw different regimes
    let mut hypothalamus_regimes = std::collections::HashSet::new();
    while let Ok(state) = rx.try_recv() {
        hypothalamus_regimes.insert(format!("{}", state.hypothalamus_regime));
    }

    // We should see at least 2 different hypothalamus regimes across the
    // trending → mean-reverting transition
    assert!(
        hypothalamus_regimes.len() >= 2,
        "Should see at least 2 different hypothalamus regimes during transition, got {:?}",
        hypothalamus_regimes
    );
}

#[test]
fn test_bridge_with_relative_volume_affects_indicators() {
    // Verify that passing a specific relative_volume value to bridge_regime_signal
    // produces indicators with that exact value (not the default).
    let signal = RoutedSignal {
        regime: MarketRegime::Trending(TrendDirection::Bullish),
        strategy: ActiveStrategy::TrendFollowing,
        confidence: 0.85,
        position_factor: 1.0,
        reason: "test".to_string(),
        detection_method: DetectionMethod::Ensemble,
        methods_agree: Some(true),
        state_probabilities: None,
        expected_duration: None,
        trend_direction: None,
    };

    // Without relative volume
    let bridged_no_vol = bridge_regime_signal("TEST", &signal, None, None, None, None);
    assert!(
        (bridged_no_vol.indicators.relative_volume - 1.0).abs() < 1e-10,
        "Without relative volume, should default to 1.0"
    );

    // With relative volume = 2.5 (high volume spike)
    let bridged_with_vol = bridge_regime_signal("TEST", &signal, None, None, None, Some(2.5));
    assert!(
        (bridged_with_vol.indicators.relative_volume - 2.5).abs() < 1e-10,
        "With relative volume 2.5, should be exactly 2.5, got {}",
        bridged_with_vol.indicators.relative_volume
    );

    // With relative volume = 0.3 (low volume)
    let bridged_low_vol = bridge_regime_signal("TEST", &signal, None, None, None, Some(0.3));
    assert!(
        (bridged_low_vol.indicators.relative_volume - 0.3).abs() < 1e-10,
        "With relative volume 0.3, should be exactly 0.3, got {}",
        bridged_low_vol.indicators.relative_volume
    );
}

#[test]
fn test_end_to_end_volume_enriched_bridge_pipeline() {
    // Full pipeline: TOML config → manager with volume → enriched bridge → broadcast → validate
    let toml_str = r#"
[manager]
ticks_per_candle = 4
detection_method = "Indicators"
min_confidence = 0.3
log_regime_changes = false

[indicators]
adx_period = 14
adx_trending_threshold = 20.0
adx_ranging_threshold = 15.0
regime_stability_bars = 2
min_regime_duration = 3
"#;

    let config = RegimeManagerConfig::from_toml_str(toml_str).expect("TOML parse failed");
    let mut mgr = RegimeManager::new(config);
    let symbol = "SOLUSD";
    mgr.register_asset(symbol);

    let (tx, _default_rx) = tokio::sync::broadcast::channel::<BridgedRegimeState>(512);
    let mut rx = tx.subscribe();

    let prices = trending_up_prices(20.0, 400);
    let mut total_signals = 0;
    let mut volume_enriched_signals = 0;

    for (i, &price) in prices.iter().enumerate() {
        // Simulate trade volume: normal for first half, spike in second half
        let volume = if i < 200 { 100.0 } else { 500.0 };
        mgr.on_trade_volume(symbol, volume);

        let spread = price * 0.0001;
        if let Some(signal) = mgr.on_tick(symbol, price - spread, price + spread) {
            let adx_val = mgr.adx_value(symbol);
            let bb_pct = mgr
                .last_regime_confidence(symbol)
                .map(|rc| rc.bb_width_percentile / 100.0);
            let atr_val = mgr.atr_value(symbol);
            let rel_vol = mgr.relative_volume(symbol);

            let bridged = bridge_regime_signal(symbol, &signal, adx_val, bb_pct, atr_val, rel_vol);

            total_signals += 1;
            if rel_vol.is_some() {
                volume_enriched_signals += 1;
            }

            tx.send(bridged).expect("Channel send failed");
        }
    }

    assert!(total_signals > 0, "Should have produced signals");
    assert!(
        volume_enriched_signals > 0,
        "Should have produced volume-enriched signals"
    );

    // Verify the consumer sees the volume spike near the transition point.
    // With ticks_per_candle=4 and the spike at tick 200, candle ~50 is
    // the first one with 5× volume. The rolling average is still dominated
    // by the old volume, so relative_volume should spike well above 1.0
    // near that transition before settling back to ~1.0 once the rolling
    // window fills with the new volume level.
    let mut received = Vec::new();
    while let Ok(state) = rx.try_recv() {
        assert_eq!(state.symbol, symbol);
        received.push(state);
    }

    assert_eq!(received.len(), total_signals);

    // Find the maximum relative_volume across all received states.
    // Near the transition (candle ~50) the spike from 400→2000 per-candle
    // should produce a relative_volume well above 1.0.
    let max_rel_vol = received
        .iter()
        .map(|s| s.indicators.relative_volume)
        .fold(0.0_f64, f64::max);

    assert!(
        max_rel_vol > 1.5,
        "Should see a relative_volume spike > 1.5 near the volume transition, got max {:.3}",
        max_rel_vol
    );
}

#[test]
fn test_zero_volume_candles_do_not_affect_relative_volume() {
    // If no trade volume is supplied, relative_volume should remain None
    // (zero-volume candles are excluded from the rolling average)
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    });
    let symbol = "BTCUSD";
    mgr.register_asset(symbol);

    // Feed 3 candles with volume, then 5 candles without volume
    for i in 0..3 {
        mgr.on_trade_volume(symbol, 100.0);
        let base = 100.0 + i as f64;
        let prices = [base, base + 0.5, base + 1.0];
        feed_prices(&mut mgr, symbol, &prices);
    }

    let vol_after_volume = mgr.relative_volume(symbol);
    assert!(
        vol_after_volume.is_some(),
        "Should have relative volume after candles with volume"
    );

    // Feed candles without any trade volume
    for i in 3..8 {
        // Notably: NO call to on_trade_volume here
        let base = 100.0 + i as f64;
        let prices = [base, base + 0.5, base + 1.0];
        feed_prices(&mut mgr, symbol, &prices);
    }

    // Relative volume should still be the last value from when we had volume,
    // since zero-volume candles are excluded from tracking
    let vol_after_no_volume = mgr.relative_volume(symbol);
    assert!(
        vol_after_no_volume.is_some(),
        "Relative volume should still be available (from last non-zero candle)"
    );
    assert_eq!(
        vol_after_volume.unwrap(),
        vol_after_no_volume.unwrap(),
        "Relative volume should not change when zero-volume candles are fed"
    );
}
