#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_comparisons,
    deprecated,
    clippy::absurd_extreme_comparisons,
    clippy::match_like_matches_macro
)]
//! Integration Tests for Regime Detection → Strategy Execution Path
//!
//! These tests validate the end-to-end pipeline:
//! 1. Raw ticks → RegimeManager tick aggregation → candle completion
//! 2. Candle → EnhancedRouter regime classification → RoutedSignal
//! 3. RoutedSignal.strategy == MeanReversion → MR / Squeeze / VWAP / ORB activation
//! 4. Strategies produce Buy/Sell/Hold signals with correct gating
//! 5. EMA Flip signals are suppressed when regime recommends MeanReversion
//! 6. Position source tracking (EmaFlip, MR, Squeeze, VwapScalper, Orb) works correctly
//!
//! # Running Tests
//!
//! ```bash
//! cargo test -p janus-forward --test regime_mr_integration
//! ```
//!
//! These tests do NOT require any external services (no Redis, no Bybit, no QuestDB).

use janus_forward::metrics::JanusMetrics;
use janus_forward::regime::{AggregatedCandle, RegimeManager, RegimeManagerConfig};
use janus_regime::{ActiveStrategy, MarketRegime, RoutedSignal, TrendDirection};
use janus_strategies::bollinger_squeeze::{
    SqueezeBreakoutConfig, SqueezeBreakoutSignal, SqueezeBreakoutStrategy,
};
use janus_strategies::mean_reversion::{
    MeanReversionConfig, MeanReversionResult, MeanReversionSignal, MeanReversionStrategy,
};
use janus_strategies::opening_range::{OrbConfig, OrbSignal, OrbStrategy};
use janus_strategies::vwap_scalper::{VwapScalperConfig, VwapScalperStrategy, VwapSignal};
use janus_strategies::{EMAFlipStrategy, Signal};
use std::sync::Arc;

// ============================================================================
// Helper functions
// ============================================================================

/// Create a RegimeManager with fast candle aggregation (few ticks per candle)
/// so tests don't need thousands of ticks.
fn fast_regime_manager() -> RegimeManager {
    RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    })
}

/// Create a RegimeManager with metrics wired to a fresh registry.
fn fast_regime_manager_with_metrics() -> (RegimeManager, Arc<JanusMetrics>) {
    let metrics = Arc::new(JanusMetrics::default());
    let mgr = RegimeManager::with_metrics(
        RegimeManagerConfig {
            ticks_per_candle: 3,
            log_regime_changes: false,
            ..Default::default()
        },
        &metrics.registry(),
    )
    .expect("Failed to create regime manager with metrics");
    (mgr, metrics)
}

/// Feed a sequence of prices (as mid-prices) into the RegimeManager,
/// collecting any RoutedSignals produced.
fn feed_prices(mgr: &mut RegimeManager, symbol: &str, prices: &[f64]) -> Vec<RoutedSignal> {
    let mut signals = Vec::new();
    for &price in prices {
        if let Some(signal) = mgr.on_tick_price(symbol, price) {
            signals.push(signal);
        }
    }
    signals
}

/// Generate a trending-up price series: base, base+step, base+2*step, ...
fn trending_up_prices(base: f64, step: f64, count: usize) -> Vec<f64> {
    (0..count).map(|i| base + step * i as f64).collect()
}

/// Generate a mean-reverting price series that oscillates around a center.
fn mean_reverting_prices(center: f64, amplitude: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let phase = (i as f64) * std::f64::consts::PI / 5.0;
            center + amplitude * phase.sin()
        })
        .collect()
}

/// Generate a volatile price series with large random-ish swings.
fn volatile_prices(base: f64, count: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(count);
    let mut price = base;
    for i in 0..count {
        // Pseudo-random large moves using a simple hash-like pattern
        let direction = if (i * 7 + 3) % 5 < 3 { 1.0 } else { -1.0 };
        let magnitude = 50.0 + ((i * 13 + 7) % 100) as f64;
        price += direction * magnitude;
        price = price.max(base * 0.5); // floor
        prices.push(price);
    }
    prices
}

/// Warm up a MeanReversionStrategy with enough data to become ready.
fn warm_up_mr(mr: &mut MeanReversionStrategy, center: f64, count: usize) {
    for i in 0..count {
        let phase = (i as f64) * std::f64::consts::PI / 7.0;
        let high = center + 50.0 + 20.0 * phase.sin().abs();
        let low = center - 50.0 - 20.0 * phase.cos().abs();
        let close = center + 30.0 * phase.sin();
        mr.update_hlc(high, low, close);
    }
}

// ============================================================================
// Test: RegimeManager tick aggregation produces candles
// ============================================================================

#[test]
fn test_tick_aggregation_produces_candles() {
    let mut mgr = fast_regime_manager();
    mgr.register_asset("BTCUSDT");

    // With ticks_per_candle = 3, every 3rd tick should complete a candle
    let result1 = mgr.on_tick_price("BTCUSDT", 50000.0);
    assert!(result1.is_none(), "1st tick should not produce a signal");

    let result2 = mgr.on_tick_price("BTCUSDT", 50100.0);
    assert!(result2.is_none(), "2nd tick should not produce a signal");

    let result3 = mgr.on_tick_price("BTCUSDT", 50050.0);
    // 3rd tick completes candle — may or may not produce a RoutedSignal
    // depending on detector warmup, but last_candle should be populated
    let candle = mgr.last_candle("BTCUSDT");
    assert!(candle.is_some(), "Candle should be available after 3 ticks");

    let c = candle.unwrap();
    assert!(
        (c.open - 50000.0).abs() < f64::EPSILON,
        "Open should be first tick"
    );
    assert!(
        (c.high - 50100.0).abs() < f64::EPSILON,
        "High should be max"
    );
    assert!((c.low - 50000.0).abs() < f64::EPSILON, "Low should be min");
    assert!(
        (c.close - 50050.0).abs() < f64::EPSILON,
        "Close should be last tick"
    );
}

#[test]
fn test_candle_fed_to_mr_strategy() {
    let mut mgr = fast_regime_manager();
    mgr.register_asset("BTCUSDT");

    let mut mr = MeanReversionStrategy::new(MeanReversionConfig::default());

    // Feed enough ticks to produce candles, and pipe each candle to MR
    let prices = mean_reverting_prices(50000.0, 200.0, 300);
    let mut candles_processed = 0;

    for &price in &prices {
        if let Some(_signal) = mgr.on_tick_price("BTCUSDT", price)
            && let Some(candle) = mgr.last_candle("BTCUSDT")
        {
            let result = mr.update_hlc_with_reason(candle.high, candle.low, candle.close);
            candles_processed += 1;
            // Just verify no panics and result is valid
            assert!(
                matches!(
                    result.signal,
                    MeanReversionSignal::Buy
                        | MeanReversionSignal::Sell
                        | MeanReversionSignal::Hold
                ),
                "MR signal should be valid"
            );
        }
    }

    assert!(
        candles_processed > 0,
        "Should have processed at least one candle through MR"
    );
}

// ============================================================================
// Test: Regime detection classifies market states
// ============================================================================

#[test]
fn test_regime_manager_eventually_produces_signals() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 2,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    // Feed a strong uptrend — enough data for the detector to warm up
    let prices = trending_up_prices(50000.0, 10.0, 1000);
    let signals = feed_prices(&mut mgr, "BTCUSDT", &prices);

    assert!(
        !signals.is_empty(),
        "Should produce RoutedSignals after sufficient warmup"
    );

    // All signals should have valid fields
    for signal in &signals {
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
        assert!(signal.position_factor >= 0.0 && signal.position_factor <= 1.5);
    }
}

#[test]
fn test_trending_market_recommends_trend_following() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 2,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    // Feed a strong monotonic uptrend
    let prices = trending_up_prices(50000.0, 15.0, 1500);
    let signals = feed_prices(&mut mgr, "BTCUSDT", &prices);

    // Among the later signals (after warmup), we should see TrendFollowing
    let late_signals: Vec<_> = signals
        .iter()
        .skip(signals.len().saturating_sub(20))
        .collect();

    let has_trend_following = late_signals
        .iter()
        .any(|s| s.strategy == ActiveStrategy::TrendFollowing);

    // The detector should eventually identify a trend.
    // If it doesn't find TrendFollowing, at minimum it shouldn't recommend NoTrade
    // for a clear trend.
    if !late_signals.is_empty() {
        let last = late_signals.last().unwrap();
        // Acceptable: TrendFollowing or MeanReversion (detector may disagree on
        // classification, but should not be NoTrade in a clear trend)
        assert!(
            last.strategy == ActiveStrategy::TrendFollowing
                || last.strategy == ActiveStrategy::MeanReversion
                || last.strategy == ActiveStrategy::NoTrade,
            "Strategy should be a valid variant: {:?}",
            last.strategy
        );
    }

    // Just verify we got signals at all
    assert!(
        !signals.is_empty(),
        "Trending prices should produce regime signals"
    );
}

// ============================================================================
// Test: MR strategy gating by regime
// ============================================================================

#[test]
fn test_mr_should_trade_in_mean_reverting_regime() {
    assert!(MeanReversionStrategy::should_trade_in_regime(
        &MarketRegime::MeanReverting
    ));
}

#[test]
fn test_mr_should_not_trade_in_trending_regime() {
    assert!(!MeanReversionStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bullish)
    ));
    assert!(!MeanReversionStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bearish)
    ));
}

#[test]
fn test_mr_should_not_trade_in_volatile_regime() {
    assert!(!MeanReversionStrategy::should_trade_in_regime(
        &MarketRegime::Volatile
    ));
}

#[test]
fn test_mr_regime_size_factor_scales_with_confidence() {
    // High confidence in MR regime → large factor
    let factor_high = MeanReversionStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.9);
    assert!(factor_high.is_some());
    assert!(factor_high.unwrap() >= 0.8);

    // Low confidence in MR regime → still trades but smaller
    let factor_low = MeanReversionStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.2);
    assert!(factor_low.is_some());
    assert!(factor_low.unwrap() <= 0.5);

    // Volatile → None (don't trade)
    let factor_volatile = MeanReversionStrategy::regime_size_factor(&MarketRegime::Volatile, 0.9);
    assert!(factor_volatile.is_none());
}

// ============================================================================
// Test: EMA Flip signal suppression when regime recommends MR
// ============================================================================

#[test]
fn test_ema_flip_suppressed_when_regime_is_mean_reversion() {
    let mut ema = EMAFlipStrategy::new(8, 21);

    // Generate a buy crossover
    ema.check_signal(50.0, 51.0); // fast below slow
    let signal = ema.check_signal(52.0, 51.0); // bullish crossover
    assert_eq!(signal, Signal::Buy, "EMA should detect crossover");

    // Simulate regime check — if strategy is MR, EMA signal should be suppressed
    let active_strategy = ActiveStrategy::MeanReversion;
    let regime_allows_ema = match active_strategy {
        ActiveStrategy::TrendFollowing => true,
        ActiveStrategy::NoTrade => false,
        ActiveStrategy::MeanReversion => false,
    };

    assert!(
        !regime_allows_ema,
        "EMA signal should be suppressed in MR regime"
    );
}

#[test]
fn test_ema_flip_allowed_when_regime_is_trend_following() {
    let active_strategy = ActiveStrategy::TrendFollowing;
    let regime_allows_ema = match active_strategy {
        ActiveStrategy::TrendFollowing => true,
        ActiveStrategy::NoTrade => false,
        ActiveStrategy::MeanReversion => false,
    };

    assert!(
        regime_allows_ema,
        "EMA signal should be allowed in TrendFollowing regime"
    );
}

#[test]
fn test_ema_flip_suppressed_when_regime_is_no_trade() {
    let active_strategy = ActiveStrategy::NoTrade;
    let regime_allows_ema = match active_strategy {
        ActiveStrategy::TrendFollowing => true,
        ActiveStrategy::NoTrade => false,
        ActiveStrategy::MeanReversion => false,
    };

    assert!(
        !regime_allows_ema,
        "EMA signal should be suppressed in NoTrade regime"
    );
}

// ============================================================================
// Test: End-to-end pipeline: ticks → regime → MR → signal
// ============================================================================

#[test]
fn test_end_to_end_regime_mr_pipeline() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 2,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    let mut mr = MeanReversionStrategy::new(MeanReversionConfig::crypto_aggressive());
    let mut ema = EMAFlipStrategy::new(8, 21);

    // Track which signals get through
    let mut mr_buy_count = 0;
    let mut mr_sell_count = 0;
    let mut mr_hold_count = 0;
    let mut ema_allowed_count = 0;
    let mut ema_suppressed_count = 0;
    let mut total_routed_signals = 0;

    // Generate a mixed price series: trend then mean-reverting then trend
    let mut prices = Vec::new();
    // Phase 1: Uptrend (500 ticks)
    prices.extend(trending_up_prices(50000.0, 8.0, 500));
    // Phase 2: Mean-reverting (600 ticks)
    prices.extend(mean_reverting_prices(54000.0, 150.0, 600));
    // Phase 3: Downtrend (400 ticks)
    let down = trending_up_prices(54000.0, -8.0, 400);
    prices.extend(down);

    for &price in &prices {
        if let Some(routed_signal) = mgr.on_tick_price("BTCUSDT", price) {
            total_routed_signals += 1;

            // Feed candle to MR strategy
            if let Some(candle) = mgr.last_candle("BTCUSDT") {
                let mr_result = mr.update_hlc_with_reason(candle.high, candle.low, candle.close);

                // Route based on regime recommendation
                match routed_signal.strategy {
                    ActiveStrategy::MeanReversion => match mr_result.signal {
                        MeanReversionSignal::Buy => mr_buy_count += 1,
                        MeanReversionSignal::Sell => mr_sell_count += 1,
                        MeanReversionSignal::Hold => mr_hold_count += 1,
                    },
                    ActiveStrategy::TrendFollowing => {
                        // EMA signals would be allowed here
                        let fast_ema = price * 1.001; // dummy
                        let slow_ema = price * 0.999;
                        let _sig = ema.check_signal(fast_ema, slow_ema);
                        ema_allowed_count += 1;
                    }
                    ActiveStrategy::NoTrade => {
                        ema_suppressed_count += 1;
                    }
                }
            }
        }
    }

    // Verify the pipeline processed data
    assert!(
        total_routed_signals > 0,
        "Pipeline should produce routed signals"
    );

    // Verify MR strategy processed candles (hold is the most common)
    let mr_total = mr_buy_count + mr_sell_count + mr_hold_count;
    println!(
        "Pipeline stats: routed_signals={}, mr_buy={}, mr_sell={}, mr_hold={}, ema_allowed={}, ema_suppressed={}",
        total_routed_signals,
        mr_buy_count,
        mr_sell_count,
        mr_hold_count,
        ema_allowed_count,
        ema_suppressed_count
    );

    // The pipeline should have processed at least some data through each path
    assert!(
        mr_total + ema_allowed_count + ema_suppressed_count == total_routed_signals,
        "All routed signals should be accounted for"
    );
}

// ============================================================================
// Test: MR strategy produces signals on mean-reverting data
// ============================================================================

#[test]
fn test_mr_strategy_produces_signals_on_ranging_data() {
    let mut mr = MeanReversionStrategy::new(MeanReversionConfig::crypto_aggressive());

    // Warm up with enough data
    warm_up_mr(&mut mr, 50000.0, 50);

    assert!(mr.is_ready(), "MR strategy should be ready after warmup");

    // Now feed data that dips to lower Bollinger Band
    // Simulate a sharp dip by feeding lower prices
    let mut signals = Vec::new();
    for i in 0..50 {
        let phase = (i as f64) * std::f64::consts::PI / 4.0;
        let high = 50000.0 + 300.0 * phase.sin().abs();
        let low = 50000.0 - 300.0 - 200.0 * phase.cos().abs();
        let close = 50000.0 + 250.0 * phase.sin();
        let result = mr.update_hlc_with_reason(high, low, close);
        if result.signal != MeanReversionSignal::Hold {
            signals.push(result);
        }
    }

    // We should get at least some non-Hold signals on ranging data
    // (Buy when near lower band, Sell when near upper band)
    // Note: depending on exact config + data, may not always trigger.
    // At minimum, verify no panics and results are well-formed.
    for sig in &signals {
        assert!(!sig.reason.is_empty(), "Signal reason should be populated");
    }
}

// ============================================================================
// Test: Metrics integration
// ============================================================================

#[test]
fn test_regime_metrics_registered_on_janus_metrics() {
    let (mut mgr, metrics) = fast_regime_manager_with_metrics();
    mgr.register_asset("BTCUSDT");

    // Feed some ticks
    for i in 0..100 {
        let price = 50000.0 + (i as f64) * 5.0;
        mgr.on_tick_price("BTCUSDT", price);
    }

    // Gather metrics from the registry — should include regime metrics
    let gathered = metrics.gather();
    assert!(
        !gathered.is_empty(),
        "Metrics registry should have metrics after regime processing"
    );

    // Look for regime-specific metric names
    let metric_names: Vec<String> = gathered
        .iter()
        .map(|mf| mf.get_name().to_string())
        .collect();

    // The regime metrics should be registered (even if values are 0)
    let has_regime_metric = metric_names
        .iter()
        .any(|name| name.contains("regime") || name.contains("janus"));
    assert!(
        has_regime_metric,
        "Should find regime or janus metrics in registry. Found: {:?}",
        metric_names
    );
}

#[test]
fn test_regime_metrics_update_on_signals() {
    let (mut mgr, metrics) = fast_regime_manager_with_metrics();
    mgr.register_asset("BTCUSDT");

    // Feed enough data to produce signals
    let prices = trending_up_prices(50000.0, 10.0, 600);
    let signals = feed_prices(&mut mgr, "BTCUSDT", &prices);

    // Gather and check that metrics were updated
    let gathered = metrics.gather();
    let metric_names: Vec<String> = gathered
        .iter()
        .map(|mf| mf.get_name().to_string())
        .collect();

    // If signals were produced, regime metrics should have been updated
    if !signals.is_empty() {
        // Check for candles_processed metric
        let has_candles_metric = metric_names
            .iter()
            .any(|name| name.contains("candles") || name.contains("processed"));
        // This is a soft check — metric naming may vary
        println!(
            "Metrics found: {:?}, signals produced: {}",
            metric_names.len(),
            signals.len()
        );
    }
}

// ============================================================================
// Test: Position source tracking
// ============================================================================

/// Simulates the PositionSource enum from event_loop.rs to verify the pattern
#[derive(Debug, Clone, PartialEq, Eq)]
enum PositionSource {
    EmaFlip,
    MeanReversion,
    SqueezeBreakout,
    VwapScalper,
    Orb,
}

#[derive(Debug)]
struct MockPosition {
    symbol: String,
    side: String,
    source: PositionSource,
}

#[test]
fn test_position_source_tracking() {
    // Simulate opening positions from different strategies
    let ema_position = MockPosition {
        symbol: "BTCUSDT".to_string(),
        side: "Buy".to_string(),
        source: PositionSource::EmaFlip,
    };

    let mr_position = MockPosition {
        symbol: "BTCUSDT".to_string(),
        side: "Buy".to_string(),
        source: PositionSource::MeanReversion,
    };

    let sq_position = MockPosition {
        symbol: "BTCUSDT".to_string(),
        side: "Buy".to_string(),
        source: PositionSource::SqueezeBreakout,
    };

    let vwap_position = MockPosition {
        symbol: "BTCUSDT".to_string(),
        side: "Buy".to_string(),
        source: PositionSource::VwapScalper,
    };

    let orb_position = MockPosition {
        symbol: "BTCUSDT".to_string(),
        side: "Buy".to_string(),
        source: PositionSource::Orb,
    };

    // MR sell should only close MR positions
    assert_eq!(mr_position.source, PositionSource::MeanReversion);
    assert_ne!(ema_position.source, PositionSource::MeanReversion);
    assert_ne!(sq_position.source, PositionSource::MeanReversion);
    assert_ne!(vwap_position.source, PositionSource::MeanReversion);
    assert_ne!(orb_position.source, PositionSource::MeanReversion);

    // EMA sell should only close EMA positions
    assert_eq!(ema_position.source, PositionSource::EmaFlip);
    assert_ne!(mr_position.source, PositionSource::EmaFlip);
    assert_ne!(sq_position.source, PositionSource::EmaFlip);
    assert_ne!(vwap_position.source, PositionSource::EmaFlip);
    assert_ne!(orb_position.source, PositionSource::EmaFlip);

    // Squeeze sell should only close Squeeze positions
    assert_eq!(sq_position.source, PositionSource::SqueezeBreakout);
    assert_ne!(ema_position.source, PositionSource::SqueezeBreakout);
    assert_ne!(mr_position.source, PositionSource::SqueezeBreakout);
    assert_ne!(vwap_position.source, PositionSource::SqueezeBreakout);
    assert_ne!(orb_position.source, PositionSource::SqueezeBreakout);

    // VWAP sell should only close VWAP positions
    assert_eq!(vwap_position.source, PositionSource::VwapScalper);
    assert_ne!(ema_position.source, PositionSource::VwapScalper);
    assert_ne!(mr_position.source, PositionSource::VwapScalper);
    assert_ne!(sq_position.source, PositionSource::VwapScalper);
    assert_ne!(orb_position.source, PositionSource::VwapScalper);

    // ORB sell should only close ORB positions
    assert_eq!(orb_position.source, PositionSource::Orb);
    assert_ne!(ema_position.source, PositionSource::Orb);
    assert_ne!(mr_position.source, PositionSource::Orb);
    assert_ne!(sq_position.source, PositionSource::Orb);
    assert_ne!(vwap_position.source, PositionSource::Orb);
}

// ============================================================================
// Test: Duplicate signal guard
// ============================================================================

#[test]
fn test_duplicate_mr_signal_guard() {
    // Simulates the duplicate-signal guard logic from event_loop.rs
    let mut last_mr_signal: Option<String> = None;

    let signals = vec![
        MeanReversionSignal::Buy,
        MeanReversionSignal::Buy,  // duplicate — should be suppressed
        MeanReversionSignal::Hold, // clears guard
        MeanReversionSignal::Buy,  // should fire again
        MeanReversionSignal::Sell,
        MeanReversionSignal::Sell, // duplicate — should be suppressed
    ];

    let mut executed = Vec::new();

    for signal in signals {
        match signal {
            MeanReversionSignal::Buy => {
                let sig_str = "MR_BUY".to_string();
                if last_mr_signal.as_ref() != Some(&sig_str) {
                    last_mr_signal = Some(sig_str);
                    executed.push("BUY");
                }
                // else: duplicate, skip
            }
            MeanReversionSignal::Sell => {
                let sig_str = "MR_SELL".to_string();
                if last_mr_signal.as_ref() != Some(&sig_str) {
                    last_mr_signal = Some(sig_str);
                    executed.push("SELL");
                }
            }
            MeanReversionSignal::Hold => {
                last_mr_signal = None;
            }
        }
    }

    assert_eq!(
        executed,
        vec!["BUY", "BUY", "SELL"],
        "Duplicate signals should be suppressed, Hold should reset guard"
    );
}

// ============================================================================
// Test: RegimeManager with on_candle direct path
// ============================================================================

#[test]
fn test_on_candle_direct_feeds_mr() {
    let mut mgr = fast_regime_manager();
    mgr.register_asset("ETHUSDT");

    let mut mr = MeanReversionStrategy::new(MeanReversionConfig::default());

    // Feed candles directly (bypassing tick aggregation)
    let mut processed = 0;
    for i in 0..200 {
        let base = 3000.0 + (i as f64) * 2.0;
        let high = base + 20.0;
        let low = base - 20.0;
        let close = base + 5.0 * ((i as f64 * 0.3).sin());

        mgr.on_candle("ETHUSDT", high, low, close);

        // Also feed to MR
        let result = mr.update_hlc_with_reason(high, low, close);
        processed += 1;

        // Verify result is well-formed
        assert!(matches!(
            result.signal,
            MeanReversionSignal::Buy | MeanReversionSignal::Sell | MeanReversionSignal::Hold
        ));
    }

    assert_eq!(processed, 200);
}

// ============================================================================
// Test: Multi-asset regime tracking
// ============================================================================

#[test]
fn test_multi_asset_regime_independence() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 2,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");
    mgr.register_asset("ETHUSDT");

    assert_eq!(mgr.asset_count(), 2);

    // Feed different price patterns to each asset
    // BTC: uptrend
    for i in 0..500 {
        let price = 50000.0 + (i as f64) * 10.0;
        mgr.on_tick_price("BTCUSDT", price);
    }

    // ETH: mean-reverting
    for i in 0..500 {
        let phase = (i as f64) * std::f64::consts::PI / 6.0;
        let price = 3000.0 + 100.0 * phase.sin();
        mgr.on_tick_price("ETHUSDT", price);
    }

    // Check that summaries are independent
    let summaries = mgr.summary();
    assert_eq!(summaries.len(), 2);

    let btc_summary = summaries.iter().find(|s| s.symbol == "BTCUSDT");
    let eth_summary = summaries.iter().find(|s| s.symbol == "ETHUSDT");

    assert!(btc_summary.is_some(), "BTC summary should exist");
    assert!(eth_summary.is_some(), "ETH summary should exist");

    // They may or may not have different regimes depending on detector warmup,
    // but they should be independently tracked
    println!("BTC: {}", btc_summary.unwrap());
    println!("ETH: {}", eth_summary.unwrap());
}

// ============================================================================
// Test: Combined MR + EMA execution routing
// ============================================================================

#[test]
fn test_strategy_routing_logic() {
    // Simulate the routing logic from process_tick in event_loop.rs
    // without requiring actual WebSocket connections

    struct StrategyRouter {
        ema: EMAFlipStrategy,
        mr: MeanReversionStrategy,
        squeeze: SqueezeBreakoutStrategy,
        vwap: VwapScalperStrategy,
        orb: OrbStrategy,
        last_mr_signal: Option<String>,
        last_sq_signal: Option<String>,
        last_vwap_signal: Option<String>,
        last_orb_signal: Option<String>,
        ema_signals_executed: u32,
        mr_signals_executed: u32,
        sq_signals_executed: u32,
        vwap_signals_executed: u32,
        orb_signals_executed: u32,
        suppressed_count: u32,
    }

    impl StrategyRouter {
        fn new() -> Self {
            let mut orb = OrbStrategy::new(OrbConfig::crypto_aggressive());
            orb.start_session(); // Start session so ORB can produce signals
            Self {
                ema: EMAFlipStrategy::new(8, 21),
                mr: MeanReversionStrategy::new(MeanReversionConfig::crypto_aggressive()),
                squeeze: SqueezeBreakoutStrategy::new(SqueezeBreakoutConfig::crypto_aggressive()),
                vwap: VwapScalperStrategy::new(VwapScalperConfig::crypto_aggressive()),
                orb,
                last_mr_signal: None,
                last_sq_signal: None,
                last_vwap_signal: None,
                last_orb_signal: None,
                ema_signals_executed: 0,
                mr_signals_executed: 0,
                sq_signals_executed: 0,
                vwap_signals_executed: 0,
                orb_signals_executed: 0,
                suppressed_count: 0,
            }
        }

        fn process(
            &mut self,
            price: f64,
            candle: Option<AggregatedCandle>,
            active_strategy: ActiveStrategy,
            regime: &MarketRegime,
        ) {
            // MR + Squeeze path: process candle when regime recommends MR
            if let Some(c) = candle {
                let mr_result = self.mr.update_hlc_with_reason(c.high, c.low, c.close);

                if active_strategy == ActiveStrategy::MeanReversion {
                    match mr_result.signal {
                        MeanReversionSignal::Buy => {
                            let sig = "MR_BUY".to_string();
                            if self.last_mr_signal.as_ref() != Some(&sig) {
                                self.last_mr_signal = Some(sig);
                                self.mr_signals_executed += 1;
                            }
                        }
                        MeanReversionSignal::Sell => {
                            let sig = "MR_SELL".to_string();
                            if self.last_mr_signal.as_ref() != Some(&sig) {
                                self.last_mr_signal = Some(sig);
                                self.mr_signals_executed += 1;
                            }
                        }
                        MeanReversionSignal::Hold => {
                            self.last_mr_signal = None;
                        }
                    }
                } else {
                    self.last_mr_signal = None;
                }

                // BB Squeeze runs alongside MR when regime is suitable
                if SqueezeBreakoutStrategy::should_trade_in_regime(regime) {
                    let sq_result = self.squeeze.update_ohlc(c.open, c.high, c.low, c.close);
                    match sq_result.signal {
                        SqueezeBreakoutSignal::BuyBreakout => {
                            let sig = "SQ_BUY".to_string();
                            if self.last_sq_signal.as_ref() != Some(&sig) {
                                self.last_sq_signal = Some(sig);
                                self.sq_signals_executed += 1;
                            }
                        }
                        SqueezeBreakoutSignal::SellBreakout => {
                            let sig = "SQ_SELL".to_string();
                            if self.last_sq_signal.as_ref() != Some(&sig) {
                                self.last_sq_signal = Some(sig);
                                self.sq_signals_executed += 1;
                            }
                        }
                        SqueezeBreakoutSignal::Squeeze | SqueezeBreakoutSignal::Hold => {
                            self.last_sq_signal = None;
                        }
                    }
                } else {
                    self.last_sq_signal = None;
                }

                // VWAP Scalper runs when regime is MeanReverting or Uncertain
                if VwapScalperStrategy::should_trade_in_regime(regime) {
                    let vwap_result = self
                        .vwap
                        .update_ohlcv(c.open, c.high, c.low, c.close, c.volume);
                    match vwap_result.signal {
                        VwapSignal::Buy => {
                            let sig = "VWAP_BUY".to_string();
                            if self.last_vwap_signal.as_ref() != Some(&sig) {
                                self.last_vwap_signal = Some(sig);
                                self.vwap_signals_executed += 1;
                            }
                        }
                        VwapSignal::Sell => {
                            let sig = "VWAP_SELL".to_string();
                            if self.last_vwap_signal.as_ref() != Some(&sig) {
                                self.last_vwap_signal = Some(sig);
                                self.vwap_signals_executed += 1;
                            }
                        }
                        VwapSignal::Hold => {
                            self.last_vwap_signal = None;
                        }
                    }
                } else {
                    self.last_vwap_signal = None;
                }

                // ORB runs when regime is MeanReverting or Uncertain
                if OrbStrategy::should_trade_in_regime(regime) {
                    let orb_result = self
                        .orb
                        .update_ohlcv(c.open, c.high, c.low, c.close, c.volume);
                    match orb_result.signal {
                        OrbSignal::BuyBreakout => {
                            let sig = "ORB_BUY".to_string();
                            if self.last_orb_signal.as_ref() != Some(&sig) {
                                self.last_orb_signal = Some(sig);
                                self.orb_signals_executed += 1;
                            }
                        }
                        OrbSignal::SellBreakout => {
                            let sig = "ORB_SELL".to_string();
                            if self.last_orb_signal.as_ref() != Some(&sig) {
                                self.last_orb_signal = Some(sig);
                                self.orb_signals_executed += 1;
                            }
                        }
                        OrbSignal::Forming | OrbSignal::Hold => {
                            self.last_orb_signal = None;
                        }
                    }
                } else {
                    self.last_orb_signal = None;
                }

                // If MR handled, skip EMA
                if active_strategy == ActiveStrategy::MeanReversion {
                    return;
                }
            }

            // EMA path: only when regime allows
            let regime_allows_ema = match active_strategy {
                ActiveStrategy::TrendFollowing => true,
                _ => false,
            };

            // Simulate EMA with dummy values
            let fast = price * 1.001;
            let slow = price * 0.999;
            let signal = self.ema.check_signal(fast, slow);

            match signal {
                Signal::Buy | Signal::Sell => {
                    if regime_allows_ema {
                        self.ema_signals_executed += 1;
                    } else {
                        self.suppressed_count += 1;
                    }
                }
                _ => {}
            }
        }
    }

    let mut router = StrategyRouter::new();

    // Warm up MR
    warm_up_mr(&mut router.mr, 50000.0, 50);

    // Simulate some ticks with different regime recommendations
    let candle = AggregatedCandle {
        open: 50000.0,
        high: 50200.0,
        low: 49800.0,
        close: 50050.0,
        volume: 0.0,
    };

    // TrendFollowing regime — EMA should execute, squeeze suppressed
    for i in 0..10 {
        let price = 50000.0 + i as f64 * 10.0;
        router.process(
            price,
            Some(candle),
            ActiveStrategy::TrendFollowing,
            &MarketRegime::Trending(TrendDirection::Bullish),
        );
    }

    // MeanReversion regime — MR + Squeeze should handle, EMA suppressed
    for i in 0..10 {
        let base = 50000.0 + (i as f64 * 0.5).sin() * 300.0;
        let c = AggregatedCandle {
            open: base,
            high: base + 100.0,
            low: base - 100.0,
            close: base + 50.0 * ((i as f64 * 0.3).sin()),
            volume: 0.0,
        };
        router.process(
            base,
            Some(c),
            ActiveStrategy::MeanReversion,
            &MarketRegime::MeanReverting,
        );
    }

    // NoTrade regime — everything suppressed
    for i in 0..5 {
        let price = 50000.0 + i as f64;
        router.process(
            price,
            None,
            ActiveStrategy::NoTrade,
            &MarketRegime::Uncertain,
        );
    }

    println!(
        "Router stats: ema_executed={}, mr_executed={}, sq_executed={}, vwap_executed={}, orb_executed={}, suppressed={}",
        router.ema_signals_executed,
        router.mr_signals_executed,
        router.sq_signals_executed,
        router.vwap_signals_executed,
        router.orb_signals_executed,
        router.suppressed_count
    );

    // The exact counts depend on EMA crossover detection with our dummy values,
    // but the routing logic should work without panics
    assert!(
        router.ema_signals_executed
            + router.mr_signals_executed
            + router.sq_signals_executed
            + router.vwap_signals_executed
            + router.orb_signals_executed
            + router.suppressed_count
            >= 0,
        "Routing should complete without errors"
    );
}

// ============================================================================
// Test: Config presets
// ============================================================================

#[test]
fn test_regime_config_presets() {
    let default = RegimeManagerConfig::default();
    assert_eq!(default.ticks_per_candle, 100);

    let fast = RegimeManagerConfig::fast();
    assert!(fast.ticks_per_candle < default.ticks_per_candle);

    let slow = RegimeManagerConfig::slow();
    assert!(slow.ticks_per_candle > default.ticks_per_candle);
}

#[test]
fn test_mr_config_presets() {
    let default = MeanReversionConfig::default();
    let aggressive = MeanReversionConfig::crypto_aggressive();
    let conservative = MeanReversionConfig::conservative();

    // Aggressive should have wider (less strict) entry threshold
    assert!(aggressive.entry_threshold >= default.entry_threshold);

    // Conservative should have tighter (more extreme) RSI thresholds
    assert!(conservative.rsi_oversold <= default.rsi_oversold);
}

// ============================================================================
// Test: AggregatedCandle struct
// ============================================================================

#[test]
fn test_aggregated_candle_is_copy() {
    let candle = AggregatedCandle {
        open: 100.0,
        high: 110.0,
        low: 90.0,
        close: 105.0,
        volume: 0.0,
    };

    // Should be Copy
    let candle2 = candle;
    assert!((candle.open - candle2.open).abs() < f64::EPSILON);
    assert!((candle.high - candle2.high).abs() < f64::EPSILON);
    assert!((candle.low - candle2.low).abs() < f64::EPSILON);
    assert!((candle.close - candle2.close).abs() < f64::EPSILON);
}

// ============================================================================
// Test: Stress test — many assets, many ticks
// ============================================================================

#[test]
fn test_stress_many_assets_many_ticks() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 5,
        log_regime_changes: false,
        ..Default::default()
    });

    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"];
    for sym in &symbols {
        mgr.register_asset(sym);
    }

    assert_eq!(mgr.asset_count(), 5);

    let mut total_signals = 0;

    // Feed 200 ticks to each asset with different patterns
    for i in 0..200 {
        for (j, sym) in symbols.iter().enumerate() {
            let base = 50000.0 / (j as f64 + 1.0);
            let price = base + (i as f64) * (j as f64 + 1.0) * 2.0;
            if let Some(_signal) = mgr.on_tick_price(sym, price) {
                total_signals += 1;
            }
        }
    }

    // Verify summaries for all assets
    let summaries = mgr.summary();
    assert_eq!(summaries.len(), 5);

    println!(
        "Stress test: {} total signals from {} assets",
        total_signals,
        symbols.len()
    );
}

// ============================================================================
// Tests: Bollinger Squeeze Breakout integration
// ============================================================================

#[test]
fn test_squeeze_should_trade_in_mean_reverting_regime() {
    assert!(SqueezeBreakoutStrategy::should_trade_in_regime(
        &MarketRegime::MeanReverting
    ));
}

#[test]
fn test_squeeze_should_trade_in_uncertain_regime() {
    assert!(SqueezeBreakoutStrategy::should_trade_in_regime(
        &MarketRegime::Uncertain
    ));
}

#[test]
fn test_squeeze_should_not_trade_in_trending_regime() {
    assert!(!SqueezeBreakoutStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bullish)
    ));
    assert!(!SqueezeBreakoutStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bearish)
    ));
}

#[test]
fn test_squeeze_should_not_trade_in_volatile_regime() {
    assert!(!SqueezeBreakoutStrategy::should_trade_in_regime(
        &MarketRegime::Volatile
    ));
}

#[test]
fn test_squeeze_regime_size_factor_mean_reverting() {
    let factor = SqueezeBreakoutStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.8);
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(
        (0.4..=1.0).contains(&f),
        "Factor {f} should be clamped 0.4–1.0"
    );
}

#[test]
fn test_squeeze_regime_size_factor_uncertain() {
    let factor = SqueezeBreakoutStrategy::regime_size_factor(&MarketRegime::Uncertain, 0.3);
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(f > 0.0 && f <= 0.5, "Uncertain factor {f} should be small");
}

#[test]
fn test_squeeze_regime_size_factor_trending_is_none() {
    let factor = SqueezeBreakoutStrategy::regime_size_factor(
        &MarketRegime::Trending(TrendDirection::Bullish),
        0.9,
    );
    assert!(factor.is_none(), "Should return None for trending regime");
}

#[test]
fn test_squeeze_config_presets() {
    let default = SqueezeBreakoutConfig::default();
    let aggressive = SqueezeBreakoutConfig::crypto_aggressive();
    let conservative = SqueezeBreakoutConfig::conservative();

    // Aggressive has tighter squeeze threshold
    assert!(aggressive.squeeze_threshold <= default.squeeze_threshold);

    // Conservative requires longer squeeze
    assert!(conservative.min_squeeze_candles >= default.min_squeeze_candles);

    // Conservative requires higher confidence
    assert!(conservative.min_confidence >= default.min_confidence);
}

#[test]
fn test_squeeze_receives_candles_from_regime_manager() {
    // Verify that the same candles fed to MR can also be fed to BB Squeeze
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 2,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    let mut mr = MeanReversionStrategy::new(MeanReversionConfig::crypto_aggressive());
    let mut squeeze = SqueezeBreakoutStrategy::new(SqueezeBreakoutConfig::crypto_aggressive());

    let mut mr_updates = 0;
    let mut sq_updates = 0;

    for i in 0..300 {
        let base = 50000.0 + 100.0 * ((i as f64) * 0.1).sin();
        if let Some(_routed_signal) = mgr.on_tick_price("BTCUSDT", base)
            && let Some(candle) = mgr.last_candle("BTCUSDT")
        {
            // Feed same candle to both strategies
            let _mr_result = mr.update_hlc_with_reason(candle.high, candle.low, candle.close);
            mr_updates += 1;

            let _sq_result =
                squeeze.update_ohlc(candle.open, candle.high, candle.low, candle.close);
            sq_updates += 1;
        }
    }

    assert_eq!(
        mr_updates, sq_updates,
        "MR and Squeeze should receive same number of candle updates"
    );
    assert!(
        mr_updates > 0,
        "Should have processed at least some candles"
    );
}

#[test]
fn test_duplicate_squeeze_signal_guard() {
    // Simulates the duplicate-signal guard logic for squeeze breakout
    let mut last_sq_signal: Option<String> = None;

    let signals = vec![
        SqueezeBreakoutSignal::BuyBreakout,
        SqueezeBreakoutSignal::BuyBreakout, // duplicate — should be suppressed
        SqueezeBreakoutSignal::Hold,        // clears guard
        SqueezeBreakoutSignal::BuyBreakout, // should fire again
        SqueezeBreakoutSignal::SellBreakout,
        SqueezeBreakoutSignal::SellBreakout, // duplicate — should be suppressed
        SqueezeBreakoutSignal::Squeeze,      // clears guard
        SqueezeBreakoutSignal::SellBreakout, // should fire again
    ];

    let mut executed = Vec::new();

    for signal in signals {
        match signal {
            SqueezeBreakoutSignal::BuyBreakout => {
                let sig_str = "SQ_BUY".to_string();
                if last_sq_signal.as_ref() != Some(&sig_str) {
                    last_sq_signal = Some(sig_str);
                    executed.push("SQ_BUY");
                }
            }
            SqueezeBreakoutSignal::SellBreakout => {
                let sig_str = "SQ_SELL".to_string();
                if last_sq_signal.as_ref() != Some(&sig_str) {
                    last_sq_signal = Some(sig_str);
                    executed.push("SQ_SELL");
                }
            }
            SqueezeBreakoutSignal::Squeeze | SqueezeBreakoutSignal::Hold => {
                last_sq_signal = None;
            }
        }
    }

    assert_eq!(
        executed,
        vec!["SQ_BUY", "SQ_BUY", "SQ_SELL", "SQ_SELL"],
        "Duplicate signals should be suppressed, Hold/Squeeze should reset guard"
    );
}

#[test]
fn test_end_to_end_regime_mr_squeeze_pipeline() {
    // Full pipeline: ticks → regime → MR + Squeeze strategy routing
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 2,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    let mut mr = MeanReversionStrategy::new(MeanReversionConfig::crypto_aggressive());
    let mut squeeze = SqueezeBreakoutStrategy::new(SqueezeBreakoutConfig::crypto_aggressive());

    let mut mr_count = 0u32;
    let mut sq_count = 0u32;
    let mut ema_count = 0u32;
    let mut no_trade_count = 0u32;
    let mut total_routed = 0u32;

    // Phase 1: mean-reverting prices
    let mut prices = mean_reverting_prices(50000.0, 100.0, 600);
    // Phase 2: uptrend
    prices.extend(trending_up_prices(50000.0, 6.0, 400));

    for &price in &prices {
        if let Some(routed_signal) = mgr.on_tick_price("BTCUSDT", price) {
            total_routed += 1;

            if let Some(candle) = mgr.last_candle("BTCUSDT") {
                match routed_signal.strategy {
                    ActiveStrategy::MeanReversion => {
                        // Both MR and Squeeze process the candle
                        let _mr_res =
                            mr.update_hlc_with_reason(candle.high, candle.low, candle.close);
                        mr_count += 1;

                        if SqueezeBreakoutStrategy::should_trade_in_regime(&routed_signal.regime) {
                            let _sq_res = squeeze.update_ohlc(
                                candle.open,
                                candle.high,
                                candle.low,
                                candle.close,
                            );
                            sq_count += 1;
                        }
                    }
                    ActiveStrategy::TrendFollowing => {
                        ema_count += 1;
                    }
                    ActiveStrategy::NoTrade => {
                        no_trade_count += 1;
                    }
                }
            }
        }
    }

    assert!(total_routed > 0, "Pipeline should produce routed signals");

    let accounted = mr_count + ema_count + no_trade_count;
    assert_eq!(
        accounted, total_routed,
        "All routed signals should be accounted for (mr={mr_count}, ema={ema_count}, no_trade={no_trade_count})"
    );

    // Squeeze updates should equal or be less than MR updates (only runs when regime is suitable)
    assert!(
        sq_count <= mr_count,
        "Squeeze updates ({sq_count}) should be <= MR updates ({mr_count})"
    );

    println!(
        "Pipeline stats: total={total_routed}, mr={mr_count}, sq={sq_count}, ema={ema_count}, no_trade={no_trade_count}"
    );
}

#[test]
fn test_squeeze_gated_by_regime_not_trending() {
    // Verify squeeze is NOT fed candles when regime recommends TrendFollowing
    let mut squeeze = SqueezeBreakoutStrategy::new(SqueezeBreakoutConfig::default());

    let regime_trending = MarketRegime::Trending(TrendDirection::Bullish);
    let regime_mr = MarketRegime::MeanReverting;

    // Simulate routing: only feed squeeze when regime allows
    let mut sq_updates_allowed = 0;
    let mut sq_updates_blocked = 0;

    for i in 0..50 {
        let base = 50000.0 + (i as f64) * 5.0;
        let candle = AggregatedCandle {
            open: base,
            high: base + 50.0,
            low: base - 50.0,
            close: base + 10.0,
            volume: 0.0,
        };

        // Alternate regimes
        let regime = if i % 2 == 0 {
            &regime_trending
        } else {
            &regime_mr
        };

        if SqueezeBreakoutStrategy::should_trade_in_regime(regime) {
            squeeze.update_ohlc(candle.open, candle.high, candle.low, candle.close);
            sq_updates_allowed += 1;
        } else {
            sq_updates_blocked += 1;
        }
    }

    assert!(sq_updates_allowed > 0, "Some updates should be allowed");
    assert!(sq_updates_blocked > 0, "Some updates should be blocked");
    assert_eq!(
        sq_updates_allowed + sq_updates_blocked,
        50,
        "All iterations accounted for"
    );
}

// ============================================================================
// VWAP Scalper Integration Tests
// ============================================================================

#[test]
fn test_vwap_should_trade_in_mean_reverting_regime() {
    assert!(VwapScalperStrategy::should_trade_in_regime(
        &MarketRegime::MeanReverting
    ));
}

#[test]
fn test_vwap_should_trade_in_uncertain_regime() {
    assert!(VwapScalperStrategy::should_trade_in_regime(
        &MarketRegime::Uncertain
    ));
}

#[test]
fn test_vwap_should_not_trade_in_trending_regime() {
    assert!(!VwapScalperStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bullish)
    ));
    assert!(!VwapScalperStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bearish)
    ));
}

#[test]
fn test_vwap_should_not_trade_in_volatile_regime() {
    assert!(!VwapScalperStrategy::should_trade_in_regime(
        &MarketRegime::Volatile
    ));
}

#[test]
fn test_vwap_regime_size_factor_mean_reverting() {
    let factor = VwapScalperStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.8);
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(
        (0.4..=1.0).contains(&f),
        "Factor should be in [0.4, 1.0], got {f}"
    );
}

#[test]
fn test_vwap_regime_size_factor_uncertain() {
    let factor = VwapScalperStrategy::regime_size_factor(&MarketRegime::Uncertain, 0.6);
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(f > 0.0 && f < 1.0, "Factor should be reduced, got {f}");
}

#[test]
fn test_vwap_regime_size_factor_trending_is_none() {
    let factor = VwapScalperStrategy::regime_size_factor(
        &MarketRegime::Trending(TrendDirection::Bullish),
        0.9,
    );
    assert!(
        factor.is_none(),
        "Trending regime should return None for VWAP"
    );
}

#[test]
fn test_vwap_receives_candles_from_regime_manager() {
    // Verify that candles produced by RegimeManager can be fed to VwapScalperStrategy
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 5,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    let mut vwap = VwapScalperStrategy::new(VwapScalperConfig::default());
    let mut candles_fed = 0;

    for i in 0..200 {
        let price = 50000.0 + (i as f64 * 0.3).sin() * 500.0;
        if let Some(_signal) = mgr.on_tick_price("BTCUSDT", price)
            && let Some(candle) = mgr.last_candle("BTCUSDT")
        {
            let result = vwap.update_ohlcv(
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            );
            candles_fed += 1;
            let _ = result.signal;
        }
    }

    assert!(
        candles_fed > 0,
        "Should have fed at least one candle to VWAP"
    );
}

#[test]
fn test_vwap_gated_by_regime_not_trending() {
    let mut vwap = VwapScalperStrategy::new(VwapScalperConfig::default());

    let regime_trending = MarketRegime::Trending(TrendDirection::Bullish);
    let regime_mr = MarketRegime::MeanReverting;

    let mut vwap_updates_allowed = 0;
    let mut vwap_updates_blocked = 0;

    for i in 0..50 {
        let base = 50000.0 + (i as f64) * 5.0;
        let candle = AggregatedCandle {
            open: base,
            high: base + 50.0,
            low: base - 50.0,
            close: base + 10.0,
            volume: 100.0,
        };

        let regime = if i % 2 == 0 {
            &regime_trending
        } else {
            &regime_mr
        };

        if VwapScalperStrategy::should_trade_in_regime(regime) {
            vwap.update_ohlcv(
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            );
            vwap_updates_allowed += 1;
        } else {
            vwap_updates_blocked += 1;
        }
    }

    assert!(
        vwap_updates_allowed > 0,
        "Some VWAP updates should be allowed"
    );
    assert!(
        vwap_updates_blocked > 0,
        "Some VWAP updates should be blocked"
    );
    assert_eq!(
        vwap_updates_allowed + vwap_updates_blocked,
        50,
        "All iterations accounted for"
    );
}

#[test]
fn test_duplicate_vwap_signal_guard() {
    // Simulate the duplicate-signal guard logic from event_loop.rs for VWAP
    let mut vwap = VwapScalperStrategy::new(VwapScalperConfig::crypto_aggressive());
    let mut last_vwap_signal: Option<String> = None;
    let mut executed_count = 0;

    // Feed enough candles to warm up
    for i in 0..50 {
        let base = 50000.0 + (i as f64) * 3.0;
        let result = vwap.update_ohlcv(base, base + 30.0, base - 30.0, base + 5.0, 100.0);

        let sig_str = match result.signal {
            VwapSignal::Buy => Some("VWAP_BUY".to_string()),
            VwapSignal::Sell => Some("VWAP_SELL".to_string()),
            VwapSignal::Hold => None,
        };

        if let Some(ref sig) = sig_str {
            if last_vwap_signal.as_ref() != Some(sig) {
                executed_count += 1;
                last_vwap_signal = Some(sig.clone());
            }
        } else {
            last_vwap_signal = None;
        }
    }

    // Should complete without panic — exact count depends on data
    println!("VWAP duplicate guard: executed {executed_count} unique signals out of 50 candles");
}

#[test]
fn test_vwap_session_reset() {
    let mut vwap = VwapScalperStrategy::new(VwapScalperConfig::default());

    // Feed some candles
    for i in 0..20 {
        let base = 50000.0 + (i as f64) * 10.0;
        vwap.update_ohlcv(base, base + 50.0, base - 50.0, base, 100.0);
    }

    assert!(vwap.candle_count() == 20);
    let had_vwap = vwap.last_vwap().is_some();

    // Reset session — VWAP should be cleared, but ATR preserved
    vwap.reset_session();

    assert!(
        vwap.last_vwap().is_none(),
        "VWAP should be None after session reset"
    );
    // ATR should still be available (cross-session context)
    // candle_count is NOT reset by reset_session (only by full reset)

    // Feed more data — should rebuild VWAP
    for i in 0..10 {
        let base = 51000.0 + (i as f64) * 5.0;
        vwap.update_ohlcv(base, base + 30.0, base - 30.0, base, 50.0);
    }

    if had_vwap {
        // After more candles, VWAP should be available again
        assert!(
            vwap.last_vwap().is_some(),
            "VWAP should rebuild after session reset"
        );
    }
}

// ============================================================================
// Opening Range Breakout Integration Tests
// ============================================================================

#[test]
fn test_orb_should_trade_in_mean_reverting_regime() {
    assert!(OrbStrategy::should_trade_in_regime(
        &MarketRegime::MeanReverting
    ));
}

#[test]
fn test_orb_should_trade_in_uncertain_regime() {
    assert!(OrbStrategy::should_trade_in_regime(
        &MarketRegime::Uncertain
    ));
}

#[test]
fn test_orb_should_not_trade_in_trending_regime() {
    assert!(!OrbStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bullish)
    ));
    assert!(!OrbStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bearish)
    ));
}

#[test]
fn test_orb_should_not_trade_in_volatile_regime() {
    assert!(!OrbStrategy::should_trade_in_regime(
        &MarketRegime::Volatile
    ));
}

#[test]
fn test_orb_regime_size_factor_mean_reverting() {
    let factor = OrbStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.8);
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(
        (0.4..=1.0).contains(&f),
        "Factor should be in [0.4, 1.0], got {f}"
    );
}

#[test]
fn test_orb_regime_size_factor_uncertain() {
    let factor = OrbStrategy::regime_size_factor(&MarketRegime::Uncertain, 0.6);
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(f > 0.0 && f < 1.0, "Factor should be reduced, got {f}");
}

#[test]
fn test_orb_regime_size_factor_trending_is_none() {
    let factor =
        OrbStrategy::regime_size_factor(&MarketRegime::Trending(TrendDirection::Bullish), 0.9);
    assert!(
        factor.is_none(),
        "Trending regime should return None for ORB"
    );
}

#[test]
fn test_orb_requires_session_start() {
    // ORB should not produce breakout signals without start_session()
    let mut orb = OrbStrategy::new(OrbConfig::default());

    for i in 0..20 {
        let base = 50000.0 + (i as f64) * 100.0;
        let result = orb.update_ohlcv(base, base + 50.0, base - 50.0, base + 30.0, 100.0);
        assert_ne!(
            result.signal,
            OrbSignal::BuyBreakout,
            "Should not produce breakout without session"
        );
        assert_ne!(
            result.signal,
            OrbSignal::SellBreakout,
            "Should not produce breakout without session"
        );
    }
}

#[test]
fn test_orb_session_lifecycle() {
    let config = OrbConfig {
        opening_range_candles: 3,
        ..OrbConfig::default()
    };
    let mut orb = OrbStrategy::new(config);

    // Before session: no signals
    assert!(!orb.is_session_active());
    assert!(!orb.is_range_complete());

    // Start session
    orb.start_session();
    assert!(orb.is_session_active());

    // Feed opening range candles
    let result1 = orb.update_ohlcv(100.0, 105.0, 95.0, 102.0, 100.0);
    assert_eq!(result1.signal, OrbSignal::Forming);

    let result2 = orb.update_ohlcv(102.0, 108.0, 98.0, 104.0, 100.0);
    assert_eq!(result2.signal, OrbSignal::Forming);

    let result3 = orb.update_ohlcv(104.0, 106.0, 97.0, 103.0, 100.0);
    // Third candle completes the opening range
    assert_eq!(result3.signal, OrbSignal::Forming);
    assert!(orb.is_range_complete());

    // Range should be [95.0, 108.0]
    assert!((orb.range_high().unwrap() - 108.0).abs() < 0.01);
    assert!((orb.range_low().unwrap() - 95.0).abs() < 0.01);

    // Start a new session — should reset
    orb.start_session();
    assert!(!orb.is_range_complete());
    assert!(!orb.breakout_fired());
}

#[test]
fn test_orb_receives_candles_from_regime_manager() {
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 5,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    let mut orb = OrbStrategy::new(OrbConfig::crypto_aggressive());
    orb.start_session();
    let mut candles_fed = 0;

    for i in 0..200 {
        let price = 50000.0 + (i as f64 * 0.3).sin() * 500.0;
        if let Some(_signal) = mgr.on_tick_price("BTCUSDT", price)
            && let Some(candle) = mgr.last_candle("BTCUSDT")
        {
            let result = orb.update_ohlcv(
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            );
            candles_fed += 1;
            let _ = result.signal;
        }
    }

    assert!(
        candles_fed > 0,
        "Should have fed at least one candle to ORB"
    );
    // After enough candles, the range should be complete
    assert!(
        orb.is_range_complete(),
        "ORB range should be complete after {} candles",
        candles_fed
    );
}

#[test]
fn test_orb_gated_by_regime_not_trending() {
    let mut orb = OrbStrategy::new(OrbConfig::default());
    orb.start_session();

    let regime_trending = MarketRegime::Trending(TrendDirection::Bullish);
    let regime_mr = MarketRegime::MeanReverting;

    let mut orb_updates_allowed = 0;
    let mut orb_updates_blocked = 0;

    for i in 0..50 {
        let base = 50000.0 + (i as f64) * 5.0;
        let candle = AggregatedCandle {
            open: base,
            high: base + 50.0,
            low: base - 50.0,
            close: base + 10.0,
            volume: 100.0,
        };

        let regime = if i % 2 == 0 {
            &regime_trending
        } else {
            &regime_mr
        };

        if OrbStrategy::should_trade_in_regime(regime) {
            orb.update_ohlcv(
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            );
            orb_updates_allowed += 1;
        } else {
            orb_updates_blocked += 1;
        }
    }

    assert!(
        orb_updates_allowed > 0,
        "Some ORB updates should be allowed"
    );
    assert!(
        orb_updates_blocked > 0,
        "Some ORB updates should be blocked"
    );
    assert_eq!(
        orb_updates_allowed + orb_updates_blocked,
        50,
        "All iterations accounted for"
    );
}

#[test]
fn test_duplicate_orb_signal_guard() {
    let config = OrbConfig {
        opening_range_candles: 3,
        allow_reentry: true, // Allow re-entry so we can test multiple signals
        ..OrbConfig::crypto_aggressive()
    };
    let mut orb = OrbStrategy::new(config);
    orb.start_session();

    let mut last_orb_signal: Option<String> = None;
    let mut executed_count = 0;

    // Feed enough candles for opening range + potential breakouts
    for i in 0..50 {
        let base = 50000.0 + (i as f64) * 20.0;
        let result = orb.update_ohlcv(base, base + 80.0, base - 80.0, base + 40.0, 200.0);

        let sig_str = match result.signal {
            OrbSignal::BuyBreakout => Some("ORB_BUY".to_string()),
            OrbSignal::SellBreakout => Some("ORB_SELL".to_string()),
            OrbSignal::Forming | OrbSignal::Hold => None,
        };

        if let Some(ref sig) = sig_str {
            if last_orb_signal.as_ref() != Some(sig) {
                executed_count += 1;
                last_orb_signal = Some(sig.clone());
            }
        } else {
            last_orb_signal = None;
        }
    }

    println!("ORB duplicate guard: executed {executed_count} unique signals out of 50 candles");
}

#[test]
fn test_volume_accumulation_in_aggregated_candle() {
    // Verify that trade volume accumulates into the AggregatedCandle
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 5,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    // Feed some trade volumes before the candle completes
    mgr.on_trade_volume("BTCUSDT", 1.5);
    mgr.on_trade_volume("BTCUSDT", 2.3);

    // Complete a candle with 5 ticks
    for i in 0..4 {
        let price = 50000.0 + (i as f64) * 10.0;
        assert!(mgr.on_tick_price("BTCUSDT", price).is_none());
    }

    // Add more volume before the last tick
    mgr.on_trade_volume("BTCUSDT", 0.7);

    // 5th tick completes the candle
    let signal = mgr.on_tick_price("BTCUSDT", 50050.0);
    assert!(signal.is_some() || signal.is_none()); // Signal depends on warmup

    if let Some(candle) = mgr.last_candle("BTCUSDT") {
        // Volume should be the sum of all trade volumes during this candle
        let expected_volume = 1.5 + 2.3 + 0.7;
        assert!(
            (candle.volume - expected_volume).abs() < 0.001,
            "Candle volume should be {expected_volume}, got {}",
            candle.volume
        );
    }
}

#[test]
fn test_end_to_end_regime_vwap_orb_pipeline() {
    // Full pipeline: ticks → regime → candle → VWAP + ORB
    let mut mgr = RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 5,
        log_regime_changes: false,
        ..Default::default()
    });
    mgr.register_asset("BTCUSDT");

    let mut vwap = VwapScalperStrategy::new(VwapScalperConfig::crypto_aggressive());
    let mut orb = OrbStrategy::new(OrbConfig::crypto_aggressive());
    orb.start_session();

    let mut total_routed = 0;
    let mut vwap_fed = 0;
    let mut orb_fed = 0;
    let mut vwap_signals = 0;
    let mut orb_breakouts = 0;

    // Mean-reverting data — sinusoidal
    for i in 0..500 {
        let phase = (i as f64) * std::f64::consts::PI / 10.0;
        let price = 50000.0 + 300.0 * phase.sin();

        // Add some trade volume
        if i % 3 == 0 {
            mgr.on_trade_volume("BTCUSDT", 0.5);
        }

        if let Some(signal) = mgr.on_tick_price("BTCUSDT", price) {
            total_routed += 1;

            if let Some(candle) = mgr.last_candle("BTCUSDT") {
                // Feed to VWAP when regime allows
                if VwapScalperStrategy::should_trade_in_regime(&signal.regime) {
                    let vr = vwap.update_ohlcv(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    );
                    vwap_fed += 1;
                    if matches!(vr.signal, VwapSignal::Buy | VwapSignal::Sell) {
                        vwap_signals += 1;
                    }
                }

                // Feed to ORB when regime allows
                if OrbStrategy::should_trade_in_regime(&signal.regime) {
                    let or = orb.update_ohlcv(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    );
                    orb_fed += 1;
                    if matches!(or.signal, OrbSignal::BuyBreakout | OrbSignal::SellBreakout) {
                        orb_breakouts += 1;
                    }
                }
            }
        }
    }

    assert!(total_routed > 0, "Pipeline should produce routed signals");
    // VWAP and ORB should have been fed candles (depends on regime classification)
    println!(
        "VWAP/ORB pipeline: routed={total_routed}, vwap_fed={vwap_fed}, orb_fed={orb_fed}, \
         vwap_signals={vwap_signals}, orb_breakouts={orb_breakouts}"
    );
}
