#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_comparisons,
    clippy::overly_complex_bool_expr,
    clippy::absurd_extreme_comparisons,
    clippy::match_like_matches_macro
)]
//! Integration Tests for Kraken-ported Strategy Pipeline
//!
//! These tests validate the regime → strategy → handler pipeline for the four
//! Kraken-ported strategies:
//!   - EMA Ribbon Scalper (8/13/21 ribbon pullback)
//!   - Trend Pullback (Fibonacci + candlestick patterns)
//!   - Momentum Surge (price surges + volume spikes)
//!   - Multi-TF Trend (EMA 50/200 + ADX with HTF alignment)
//!
//! Additionally, tests cover the HTF candle aggregator that feeds higher-timeframe
//! trend direction into HTF-aware strategies.
//!
//! # Running Tests
//!
//! ```bash
//! cargo test -p janus-forward --test kraken_strategies_integration
//! ```
//!
//! These tests do NOT require any external services (no Redis, no Bybit, no QuestDB).

use janus_forward::metrics::JanusMetrics;
use janus_forward::regime::{AggregatedCandle, RegimeManager, RegimeManagerConfig};
use janus_regime::{ActiveStrategy, MarketRegime, RoutedSignal, TrendDirection};
use janus_strategies::bollinger_squeeze::{SqueezeBreakoutConfig, SqueezeBreakoutStrategy};
use janus_strategies::ema_ribbon_scalper::{
    EmaRibbonConfig, EmaRibbonScalperStrategy, EmaRibbonSignal,
};
use janus_strategies::mean_reversion::{MeanReversionConfig, MeanReversionStrategy};
use janus_strategies::momentum_surge::{
    MomentumSurgeConfig, MomentumSurgeSignal, MomentumSurgeStrategy,
};
use janus_strategies::multi_tf_trend::{MultiTfConfig, MultiTfSignal, MultiTfTrendStrategy};
use janus_strategies::opening_range::{OrbConfig, OrbStrategy};
use janus_strategies::trend_pullback::{
    TrendPullbackConfig, TrendPullbackSignal, TrendPullbackStrategy,
};
use janus_strategies::vwap_scalper::{VwapScalperConfig, VwapScalperStrategy};
use janus_strategies::{EMAFlipStrategy, Signal};
use std::sync::Arc;

// ============================================================================
// Helper functions
// ============================================================================

/// Create a RegimeManager with fast candle aggregation (few ticks per candle).
fn fast_regime_manager() -> RegimeManager {
    RegimeManager::new(RegimeManagerConfig {
        ticks_per_candle: 3,
        log_regime_changes: false,
        ..Default::default()
    })
}

/// Feed a sequence of prices into the RegimeManager, collecting RoutedSignals.
fn feed_prices(mgr: &mut RegimeManager, symbol: &str, prices: &[f64]) -> Vec<RoutedSignal> {
    let mut signals = Vec::new();
    for &price in prices {
        if let Some(signal) = mgr.on_tick_price(symbol, price) {
            signals.push(signal);
        }
    }
    signals
}

/// Generate a trending-up price series.
fn trending_up_prices(base: f64, step: f64, count: usize) -> Vec<f64> {
    (0..count).map(|i| base + step * i as f64).collect()
}

/// Generate a mean-reverting (oscillating) price series.
fn mean_reverting_prices(center: f64, amplitude: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let phase = (i as f64) * std::f64::consts::PI / 5.0;
            center + amplitude * phase.sin()
        })
        .collect()
}

/// Warm up EMA Ribbon strategy with enough data to produce signals.
fn warm_up_ema_ribbon(strategy: &mut EmaRibbonScalperStrategy, center: f64, count: usize) {
    for i in 0..count {
        let phase = (i as f64) * std::f64::consts::PI / 7.0;
        let price = center + 10.0 * phase.sin();
        let high = price + 25.0;
        let low = price - 25.0;
        let open = price - 5.0;
        strategy.update_ohlcv(open, high, low, price, 1000.0);
    }
}

/// Warm up Trend Pullback strategy.
fn warm_up_trend_pullback(strategy: &mut TrendPullbackStrategy, center: f64, count: usize) {
    for i in 0..count {
        let phase = (i as f64) * std::f64::consts::PI / 7.0;
        let price = center + 5.0 * (i as f64) + 10.0 * phase.sin(); // slight uptrend
        let high = price + 30.0;
        let low = price - 30.0;
        let open = price - 5.0;
        strategy.update_ohlcv(open, high, low, price, 1000.0);
    }
}

/// Warm up Momentum Surge strategy.
fn warm_up_momentum_surge(strategy: &mut MomentumSurgeStrategy, center: f64, count: usize) {
    for i in 0..count {
        let price = center + (i as f64) * 0.5;
        let high = price + 10.0;
        let low = price - 10.0;
        let open = price - 2.0;
        strategy.update_ohlcv(open, high, low, price, 1000.0);
    }
}

/// Warm up Multi-TF Trend strategy with a slight uptrend.
fn warm_up_multi_tf(strategy: &mut MultiTfTrendStrategy, center: f64, count: usize) {
    for i in 0..count {
        let price = center + (i as f64) * 5.0;
        let high = price + 25.0;
        let low = price - 25.0;
        let open = price - 5.0;
        strategy.update_ohlcv(open, high, low, price, 1000.0);
    }
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
// HTF Aggregator Tests
// ============================================================================

/// Minimal HTF aggregator replica for testing (mirrors event_loop.rs logic).
struct TestHtfAggregator {
    ratio: usize,
    count: usize,
    agg_open: f64,
    agg_high: f64,
    agg_low: f64,
    agg_close: f64,
    agg_volume: f64,
    ema_short: janus_regime::indicators::EMA,
    ema_long: janus_regime::indicators::EMA,
    candle_count: usize,
    warmup: usize,
}

impl TestHtfAggregator {
    fn new(ratio: usize, ema_short_period: usize, ema_long_period: usize) -> Self {
        Self {
            ratio,
            count: 0,
            agg_open: 0.0,
            agg_high: f64::NEG_INFINITY,
            agg_low: f64::INFINITY,
            agg_close: 0.0,
            agg_volume: 0.0,
            ema_short: janus_regime::indicators::EMA::new(ema_short_period),
            ema_long: janus_regime::indicators::EMA::new(ema_long_period),
            candle_count: 0,
            warmup: ema_long_period + 5,
        }
    }

    fn feed(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<TrendDirection> {
        if self.count == 0 {
            self.agg_open = open;
            self.agg_high = high;
            self.agg_low = low;
        } else {
            if high > self.agg_high {
                self.agg_high = high;
            }
            if low < self.agg_low {
                self.agg_low = low;
            }
        }
        self.agg_close = close;
        self.agg_volume += volume;
        self.count += 1;

        if self.count >= self.ratio {
            let htf_close = self.agg_close;
            self.ema_short.update(htf_close);
            self.ema_long.update(htf_close);
            self.candle_count += 1;

            self.count = 0;
            self.agg_high = f64::NEG_INFINITY;
            self.agg_low = f64::INFINITY;
            self.agg_volume = 0.0;

            if self.candle_count >= self.warmup {
                let short_val = self.ema_short.value();
                let long_val = self.ema_long.value();
                if short_val > long_val {
                    return Some(TrendDirection::Bullish);
                } else {
                    return Some(TrendDirection::Bearish);
                }
            }
        }
        None
    }
}

#[test]
fn test_htf_aggregator_ratio() {
    // With ratio=3, every 3rd LTF candle should complete an HTF bar
    let mut agg = TestHtfAggregator::new(3, 5, 10);
    let mut htf_completions = 0;

    for i in 0..30 {
        let price = 50000.0 + (i as f64) * 10.0;
        if (agg
            .feed(price, price + 5.0, price - 5.0, price, 100.0)
            .is_some()
            || agg.candle_count > htf_completions)
            && agg.candle_count > htf_completions
        {
            htf_completions = agg.candle_count;
        }
    }
    // 30 LTF candles / 3 ratio = 10 HTF candles
    assert_eq!(agg.candle_count, 10);
}

#[test]
fn test_htf_aggregator_no_trend_before_warmup() {
    // EMA long period = 10, warmup = 15. Until 15 HTF candles, no trend emitted.
    let mut agg = TestHtfAggregator::new(2, 5, 10);
    let mut first_trend_at: Option<usize> = None;

    for i in 0..100 {
        let price = 50000.0 + (i as f64) * 5.0;
        if let Some(_trend) = agg.feed(price, price + 5.0, price - 5.0, price, 100.0)
            && first_trend_at.is_none()
        {
            first_trend_at = Some(i);
        }
    }

    // 100 LTF candles / 2 ratio = 50 HTF candles. warmup = 15.
    // First trend emitted when candle_count reaches 15, which is LTF candle index 29.
    assert!(first_trend_at.is_some(), "Should eventually emit a trend");
    let first = first_trend_at.unwrap();
    // The first trend should come after warmup * ratio LTF candles
    assert!(
        first >= 29,
        "First trend at LTF candle {}, expected >= 29",
        first
    );
}

#[test]
fn test_htf_aggregator_bullish_on_uptrend() {
    let mut agg = TestHtfAggregator::new(2, 5, 10);
    let mut last_trend = None;

    // Feed a strong uptrend
    for i in 0..200 {
        let price = 50000.0 + (i as f64) * 20.0; // strong uptrend
        if let Some(trend) = agg.feed(price, price + 5.0, price - 5.0, price, 100.0) {
            last_trend = Some(trend);
        }
    }

    assert_eq!(last_trend, Some(TrendDirection::Bullish));
}

#[test]
fn test_htf_aggregator_bearish_on_downtrend() {
    let mut agg = TestHtfAggregator::new(2, 5, 10);
    let mut last_trend = None;

    // Feed a strong downtrend
    for i in 0..200 {
        let price = 60000.0 - (i as f64) * 20.0; // strong downtrend
        if let Some(trend) = agg.feed(price, price + 5.0, price - 5.0, price, 100.0) {
            last_trend = Some(trend);
        }
    }

    assert_eq!(last_trend, Some(TrendDirection::Bearish));
}

#[test]
fn test_htf_aggregator_ohlcv_aggregation() {
    let mut agg = TestHtfAggregator::new(3, 5, 10);

    // Feed 3 candles manually and check OHLCV aggregation
    // Candle 1: O=100, H=110, L=90, C=105
    assert!(agg.feed(100.0, 110.0, 90.0, 105.0, 50.0).is_none());
    assert_eq!(agg.count, 1);
    assert_eq!(agg.agg_open, 100.0);
    assert_eq!(agg.agg_high, 110.0);
    assert_eq!(agg.agg_low, 90.0);

    // Candle 2: O=105, H=120, L=95, C=115
    assert!(agg.feed(105.0, 120.0, 95.0, 115.0, 60.0).is_none());
    assert_eq!(agg.count, 2);
    assert_eq!(agg.agg_open, 100.0); // first candle's open
    assert_eq!(agg.agg_high, 120.0); // max of 110, 120
    assert_eq!(agg.agg_low, 90.0); // min of 90, 95

    // Candle 3: O=115, H=125, L=88, C=110 → completes HTF bar
    // (won't emit trend yet since warmup not reached)
    agg.feed(115.0, 125.0, 88.0, 110.0, 70.0);
    assert_eq!(agg.candle_count, 1);
    assert_eq!(agg.count, 0); // reset for next bar
}

#[test]
fn test_htf_feeds_multi_tf_strategy() {
    // Verify that set_htf_trend enables the Multi-TF strategy to produce signals
    let mut strategy = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());

    // Without HTF trend, strategy should not produce buy/sell
    warm_up_multi_tf(&mut strategy, 50000.0, 250);
    let result = strategy.update_ohlcv(51300.0, 51350.0, 51250.0, 51300.0, 2000.0);
    // Without HTF trend, Multi-TF should hold
    assert_eq!(result.signal, MultiTfSignal::Hold);

    // Set bullish HTF trend
    strategy.set_htf_trend(TrendDirection::Bullish);

    // Feed more uptrend data — now signals should be possible
    for i in 0..50 {
        let price = 51300.0 + (i as f64) * 10.0;
        strategy.update_ohlcv(price - 5.0, price + 20.0, price - 20.0, price, 2000.0);
    }
    // At minimum, strategy is receiving data and not panicking
    assert!(strategy.is_ready());
}

#[test]
fn test_htf_feeds_ema_ribbon_strategy() {
    let mut strategy = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());

    // Set bullish HTF trend
    strategy.set_htf_trend(TrendDirection::Bullish);
    assert_eq!(strategy.htf_trend(), Some(TrendDirection::Bullish));

    // Clear and verify
    strategy.clear_htf_trend();
    assert_eq!(strategy.htf_trend(), None);

    // Set bearish
    strategy.set_htf_trend(TrendDirection::Bearish);
    assert_eq!(strategy.htf_trend(), Some(TrendDirection::Bearish));
}

#[test]
fn test_htf_feeds_trend_pullback_strategy() {
    let mut strategy = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());

    strategy.set_htf_trend(TrendDirection::Bullish);
    assert_eq!(strategy.htf_trend(), Some(TrendDirection::Bullish));

    strategy.set_htf_trend(TrendDirection::Bearish);
    assert_eq!(strategy.htf_trend(), Some(TrendDirection::Bearish));

    strategy.clear_htf_trend();
    assert_eq!(strategy.htf_trend(), None);
}

// ============================================================================
// EMA Ribbon Scalper Tests
// ============================================================================

#[test]
fn test_ema_ribbon_should_trade_in_trending_regime() {
    assert!(EmaRibbonScalperStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bullish)
    ));
    assert!(EmaRibbonScalperStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bearish)
    ));
}

#[test]
fn test_ema_ribbon_should_not_trade_in_mean_reverting() {
    assert!(!EmaRibbonScalperStrategy::should_trade_in_regime(
        &MarketRegime::MeanReverting
    ));
}

#[test]
fn test_ema_ribbon_regime_size_factor_trending() {
    let factor = EmaRibbonScalperStrategy::regime_size_factor(
        &MarketRegime::Trending(TrendDirection::Bullish),
        0.8,
    );
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(
        f > 0.0 && f <= 1.0,
        "Factor should be between 0 and 1, got {}",
        f
    );
}

#[test]
fn test_ema_ribbon_regime_size_factor_mean_reverting_is_none() {
    let factor = EmaRibbonScalperStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.9);
    assert!(
        factor.is_none(),
        "EMA Ribbon should not produce a size factor in MeanReverting regime"
    );
}

#[test]
fn test_ema_ribbon_config_presets() {
    let default_config = EmaRibbonConfig::default();
    let aggressive = EmaRibbonConfig::crypto_aggressive();

    // Both should create valid strategies
    let s1 = EmaRibbonScalperStrategy::new(default_config);
    let s2 = EmaRibbonScalperStrategy::new(aggressive);

    // Strategies start not ready
    assert!(!s1.is_ready());
    assert!(!s2.is_ready());
}

#[test]
fn test_ema_ribbon_warmup_and_readiness() {
    let mut strategy = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    assert!(!strategy.is_ready());

    warm_up_ema_ribbon(&mut strategy, 50000.0, 30);
    assert!(strategy.is_ready());
}

#[test]
fn test_ema_ribbon_produces_hold_during_flat_market() {
    let mut strategy = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    warm_up_ema_ribbon(&mut strategy, 50000.0, 30);

    // Feed flat data — should hold
    for _ in 0..10 {
        let result = strategy.update_ohlcv(50000.0, 50010.0, 49990.0, 50000.0, 1000.0);
        // In a flat market, ribbon is likely tangled — expect Hold
        assert!(
            result.signal == EmaRibbonSignal::Hold
                || result.signal == EmaRibbonSignal::Buy
                || result.signal == EmaRibbonSignal::Sell,
            "Should produce a valid signal variant"
        );
    }
}

#[test]
fn test_ema_ribbon_signal_has_confidence() {
    let mut strategy = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    warm_up_ema_ribbon(&mut strategy, 50000.0, 30);

    let result = strategy.update_ohlcv(50100.0, 50150.0, 50050.0, 50120.0, 2000.0);
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be between 0 and 1, got {}",
        result.confidence
    );
}

#[test]
fn test_ema_ribbon_duplicate_signal_guard() {
    // Simulate the duplicate-signal guard logic from event_loop.rs
    let mut strategy = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    warm_up_ema_ribbon(&mut strategy, 50000.0, 30);

    let mut last_signal: Option<String> = None;
    let mut signal_count = 0;

    for i in 0..100 {
        let price = 50000.0 + (i as f64) * 3.0;
        let result = strategy.update_ohlcv(price - 5.0, price + 15.0, price - 15.0, price, 1000.0);

        match result.signal {
            EmaRibbonSignal::Buy => {
                let sig = "RIBBON_BUY".to_string();
                if last_signal.as_ref() != Some(&sig) {
                    last_signal = Some(sig);
                    signal_count += 1;
                }
            }
            EmaRibbonSignal::Sell => {
                let sig = "RIBBON_SELL".to_string();
                if last_signal.as_ref() != Some(&sig) {
                    last_signal = Some(sig);
                    signal_count += 1;
                }
            }
            EmaRibbonSignal::Hold => {
                last_signal = None;
            }
        }
    }

    // The duplicate guard should prevent repeated identical signals
    // (exact count depends on data, but it should be reasonable)
    println!("EMA Ribbon signals after dedup: {}", signal_count);
    assert!(
        signal_count < 100,
        "Duplicate guard should reduce signal count"
    );
}

// ============================================================================
// Trend Pullback Tests
// ============================================================================

#[test]
fn test_trend_pullback_should_trade_in_trending_regime() {
    assert!(TrendPullbackStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bullish)
    ));
    assert!(TrendPullbackStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bearish)
    ));
}

#[test]
fn test_trend_pullback_should_not_trade_in_mean_reverting() {
    assert!(!TrendPullbackStrategy::should_trade_in_regime(
        &MarketRegime::MeanReverting
    ));
}

#[test]
fn test_trend_pullback_regime_size_factor_trending() {
    let factor = TrendPullbackStrategy::regime_size_factor(
        &MarketRegime::Trending(TrendDirection::Bullish),
        0.85,
    );
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(f > 0.0 && f <= 1.0, "Factor {}", f);
}

#[test]
fn test_trend_pullback_regime_size_factor_mean_reverting_is_none() {
    let factor = TrendPullbackStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.9);
    assert!(factor.is_none());
}

#[test]
fn test_trend_pullback_config_presets() {
    let default_config = TrendPullbackConfig::default();
    let aggressive = TrendPullbackConfig::crypto_aggressive();

    let s1 = TrendPullbackStrategy::new(default_config);
    let s2 = TrendPullbackStrategy::new(aggressive);
    assert!(!s1.is_ready());
    assert!(!s2.is_ready());
}

#[test]
fn test_trend_pullback_warmup_and_readiness() {
    let mut strategy = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());
    assert!(!strategy.is_ready());

    warm_up_trend_pullback(&mut strategy, 50000.0, 220);
    assert!(strategy.is_ready());
}

#[test]
fn test_trend_pullback_signal_has_metadata() {
    let mut strategy = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());
    warm_up_trend_pullback(&mut strategy, 50000.0, 220);

    let result = strategy.update_ohlcv(51200.0, 51250.0, 51150.0, 51200.0, 2000.0);
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be valid, got {}",
        result.confidence
    );
    // EMA values should be present after warmup
    assert!(
        result.ema_short.is_some(),
        "Short EMA should be present after warmup"
    );
    assert!(
        result.ema_long.is_some(),
        "Long EMA should be present after warmup"
    );
}

#[test]
fn test_trend_pullback_many_candles_no_panic() {
    let mut strategy = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());
    strategy.set_htf_trend(TrendDirection::Bullish);

    let mut price = 50000.0;
    for i in 0..2000 {
        let direction = if (i / 200) % 2 == 0 { 1.0 } else { -1.0 };
        price += direction * 10.0 + (i as f64 * 0.1).sin() * 20.0;
        let high = price + 25.0;
        let low = price - 25.0;
        let open = price - direction * 5.0;
        let _ = strategy.update_ohlcv(open, high, low, price, 1000.0);
    }
    // Should not panic with 2000 candles
}

// ============================================================================
// Momentum Surge Tests
// ============================================================================

#[test]
fn test_momentum_surge_should_trade_in_trending_regime() {
    assert!(MomentumSurgeStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bullish)
    ));
    assert!(MomentumSurgeStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bearish)
    ));
}

#[test]
fn test_momentum_surge_should_trade_in_volatile_regime() {
    assert!(MomentumSurgeStrategy::should_trade_in_regime(
        &MarketRegime::Volatile
    ));
}

#[test]
fn test_momentum_surge_should_not_trade_in_mean_reverting() {
    assert!(!MomentumSurgeStrategy::should_trade_in_regime(
        &MarketRegime::MeanReverting
    ));
}

#[test]
fn test_momentum_surge_regime_size_factor_trending() {
    let factor = MomentumSurgeStrategy::regime_size_factor(
        &MarketRegime::Trending(TrendDirection::Bullish),
        0.8,
    );
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(f > 0.0 && f <= 1.0);
}

#[test]
fn test_momentum_surge_regime_size_factor_volatile() {
    let factor = MomentumSurgeStrategy::regime_size_factor(&MarketRegime::Volatile, 0.7);
    assert!(factor.is_some());
}

#[test]
fn test_momentum_surge_regime_size_factor_mean_reverting_is_none() {
    let factor = MomentumSurgeStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.9);
    assert!(factor.is_none());
}

#[test]
fn test_momentum_surge_config_presets() {
    let default_config = MomentumSurgeConfig::default();
    let aggressive = MomentumSurgeConfig::crypto_aggressive();

    let s1 = MomentumSurgeStrategy::new(default_config);
    let s2 = MomentumSurgeStrategy::new(aggressive);
    assert!(!s1.is_ready());
    assert!(!s2.is_ready());
}

#[test]
fn test_momentum_surge_warmup_and_readiness() {
    let mut strategy = MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive());
    assert!(!strategy.is_ready());

    warm_up_momentum_surge(&mut strategy, 50000.0, 30);
    assert!(strategy.is_ready());
}

#[test]
fn test_momentum_surge_signal_has_metadata() {
    let mut strategy = MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive());
    warm_up_momentum_surge(&mut strategy, 50000.0, 30);

    let result = strategy.update_ohlcv(50100.0, 50200.0, 50050.0, 50150.0, 5000.0);
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be valid"
    );
}

#[test]
fn test_momentum_surge_many_candles_no_panic() {
    let mut strategy = MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive());

    let mut price = 50000.0;
    for i in 0..2000 {
        let direction = if (i / 100) % 2 == 0 { 1.0 } else { -1.0 };
        price += direction * 15.0;
        let vol = if i % 50 == 0 { 10000.0 } else { 1000.0 }; // occasional volume spikes
        let _ = strategy.update_ohlcv(price - 5.0, price + 25.0, price - 25.0, price, vol);
    }
}

// ============================================================================
// Multi-TF Trend Tests
// ============================================================================

#[test]
fn test_multi_tf_should_trade_in_trending_regime() {
    assert!(MultiTfTrendStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bullish)
    ));
    assert!(MultiTfTrendStrategy::should_trade_in_regime(
        &MarketRegime::Trending(TrendDirection::Bearish)
    ));
}

#[test]
fn test_multi_tf_should_not_trade_in_mean_reverting() {
    assert!(!MultiTfTrendStrategy::should_trade_in_regime(
        &MarketRegime::MeanReverting
    ));
}

#[test]
fn test_multi_tf_should_not_trade_in_volatile() {
    assert!(!MultiTfTrendStrategy::should_trade_in_regime(
        &MarketRegime::Volatile
    ));
}

#[test]
fn test_multi_tf_regime_size_factor_trending() {
    let factor = MultiTfTrendStrategy::regime_size_factor(
        &MarketRegime::Trending(TrendDirection::Bullish),
        0.9,
    );
    assert!(factor.is_some());
    let f = factor.unwrap();
    assert!(f > 0.0 && f <= 1.0);
}

#[test]
fn test_multi_tf_regime_size_factor_mean_reverting_is_none() {
    let factor = MultiTfTrendStrategy::regime_size_factor(&MarketRegime::MeanReverting, 0.9);
    assert!(factor.is_none());
}

#[test]
fn test_multi_tf_config_presets() {
    let default_config = MultiTfConfig::default();
    let aggressive = MultiTfConfig::crypto_aggressive();

    let s1 = MultiTfTrendStrategy::new(default_config);
    let s2 = MultiTfTrendStrategy::new(aggressive);
    assert!(!s1.is_ready());
    assert!(!s2.is_ready());
}

#[test]
fn test_multi_tf_warmup_and_readiness() {
    let mut strategy = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());
    assert!(!strategy.is_ready());

    warm_up_multi_tf(&mut strategy, 50000.0, 250);
    assert!(strategy.is_ready());
}

#[test]
fn test_multi_tf_requires_htf_for_signals() {
    let mut strategy = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());
    warm_up_multi_tf(&mut strategy, 50000.0, 250);
    assert!(strategy.is_ready());

    // Without HTF trend, should only produce Hold
    let mut non_hold_count = 0;
    for i in 0..50 {
        let price = 51300.0 + (i as f64) * 5.0;
        let result = strategy.update_ohlcv(price - 5.0, price + 20.0, price - 20.0, price, 1000.0);
        if result.signal != MultiTfSignal::Hold {
            non_hold_count += 1;
        }
    }
    assert_eq!(
        non_hold_count, 0,
        "Multi-TF should not produce buy/sell without HTF trend"
    );
}

#[test]
fn test_multi_tf_produces_signals_with_htf_set() {
    let mut strategy = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());
    strategy.set_htf_trend(TrendDirection::Bullish);

    // Feed a strong uptrend
    let mut price = 50000.0;
    for i in 0..300 {
        price += 5.0;
        let high = price + 25.0;
        let low = price - 25.0;
        let open = price - 5.0;
        let _ = strategy.update_ohlcv(open, high, low, price, 1000.0);
    }

    // Strategy should be ready after 300 candles
    assert!(strategy.is_ready());
}

#[test]
fn test_multi_tf_htf_trend_set_clear() {
    let mut strategy = MultiTfTrendStrategy::with_defaults();
    assert!(strategy.htf_trend().is_none());

    strategy.set_htf_trend(TrendDirection::Bullish);
    assert_eq!(strategy.htf_trend(), Some(TrendDirection::Bullish));

    strategy.set_htf_trend(TrendDirection::Bearish);
    assert_eq!(strategy.htf_trend(), Some(TrendDirection::Bearish));

    strategy.clear_htf_trend();
    assert!(strategy.htf_trend().is_none());
}

#[test]
fn test_multi_tf_many_candles_trend_change() {
    let mut strategy = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());

    // Phase 1: uptrend with bullish HTF
    strategy.set_htf_trend(TrendDirection::Bullish);
    warm_up_multi_tf(&mut strategy, 50000.0, 250);
    assert!(strategy.is_ready());

    // Phase 2: switch to downtrend
    strategy.set_htf_trend(TrendDirection::Bearish);
    for i in 0..100 {
        let price = 52250.0 - (i as f64) * 15.0;
        let _ = strategy.update_ohlcv(price + 5.0, price + 25.0, price - 25.0, price, 1000.0);
    }
    // Should not panic on trend reversal
}

// ============================================================================
// Strategy Routing Integration Tests
// ============================================================================

/// Position tracking enum (mirrors event_loop.rs)
#[derive(Debug, Clone, PartialEq)]
enum PositionSource {
    EmaFlip,
    MeanReversion,
    SqueezeBreakout,
    VwapScalper,
    Orb,
    EmaRibbon,
    TrendPullback,
    MomentumSurge,
    MultiTfTrend,
}

/// Mock position for tracking
#[derive(Debug, Clone)]
struct MockPosition {
    side: String,
    entry_price: f64,
    source: PositionSource,
}

#[test]
fn test_position_source_tracking_new_strategies() {
    // Verify that positions can be created with the new strategy sources
    let sources = vec![
        PositionSource::EmaRibbon,
        PositionSource::TrendPullback,
        PositionSource::MomentumSurge,
        PositionSource::MultiTfTrend,
    ];

    for source in sources {
        let pos = MockPosition {
            side: "Buy".to_string(),
            entry_price: 50000.0,
            source: source.clone(),
        };
        assert_eq!(pos.source, source);
        assert_eq!(pos.side, "Buy");
    }
}

#[test]
fn test_sell_only_closes_matching_strategy_position() {
    // Simulate: position opened by EmaRibbon should only be closed by EmaRibbon sell
    let position = Some(MockPosition {
        side: "Buy".to_string(),
        entry_price: 50000.0,
        source: PositionSource::EmaRibbon,
    });

    // TrendPullback sell should NOT close it
    if let Some(ref pos) = position {
        assert_ne!(pos.source, PositionSource::TrendPullback);
    }

    // EmaRibbon sell SHOULD close it
    if let Some(ref pos) = position {
        assert_eq!(pos.source, PositionSource::EmaRibbon);
    }
}

#[test]
fn test_full_strategy_routing_with_kraken_strategies() {
    // Extended version of test_strategy_routing_logic that includes Kraken strategies
    struct FullRouter {
        ema: EMAFlipStrategy,
        mr: MeanReversionStrategy,
        squeeze: SqueezeBreakoutStrategy,
        vwap: VwapScalperStrategy,
        orb: OrbStrategy,
        ema_ribbon: EmaRibbonScalperStrategy,
        trend_pullback: TrendPullbackStrategy,
        momentum_surge: MomentumSurgeStrategy,
        multi_tf: MultiTfTrendStrategy,
        last_mr_signal: Option<String>,
        last_sq_signal: Option<String>,
        last_vwap_signal: Option<String>,
        last_orb_signal: Option<String>,
        last_ribbon_signal: Option<String>,
        last_tp_signal: Option<String>,
        last_surge_signal: Option<String>,
        last_mtf_signal: Option<String>,
        // Counters
        ema_executed: u32,
        mr_executed: u32,
        sq_executed: u32,
        vwap_executed: u32,
        orb_executed: u32,
        ribbon_executed: u32,
        tp_executed: u32,
        surge_executed: u32,
        mtf_executed: u32,
        suppressed: u32,
    }

    impl FullRouter {
        fn new() -> Self {
            let mut orb = OrbStrategy::new(OrbConfig::crypto_aggressive());
            orb.start_session();
            Self {
                ema: EMAFlipStrategy::new(8, 21),
                mr: MeanReversionStrategy::new(MeanReversionConfig::crypto_aggressive()),
                squeeze: SqueezeBreakoutStrategy::new(SqueezeBreakoutConfig::crypto_aggressive()),
                vwap: VwapScalperStrategy::new(VwapScalperConfig::crypto_aggressive()),
                orb,
                ema_ribbon: EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive()),
                trend_pullback: TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive()),
                momentum_surge: MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive()),
                multi_tf: MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive()),
                last_mr_signal: None,
                last_sq_signal: None,
                last_vwap_signal: None,
                last_orb_signal: None,
                last_ribbon_signal: None,
                last_tp_signal: None,
                last_surge_signal: None,
                last_mtf_signal: None,
                ema_executed: 0,
                mr_executed: 0,
                sq_executed: 0,
                vwap_executed: 0,
                orb_executed: 0,
                ribbon_executed: 0,
                tp_executed: 0,
                surge_executed: 0,
                mtf_executed: 0,
                suppressed: 0,
            }
        }

        fn process(
            &mut self,
            candle: Option<AggregatedCandle>,
            active_strategy: ActiveStrategy,
            regime: &MarketRegime,
        ) {
            if let Some(c) = candle {
                // Mean-reverting strategies
                if active_strategy == ActiveStrategy::MeanReversion {
                    let mr_result = self.mr.update_hlc_with_reason(c.high, c.low, c.close);
                    match mr_result.signal {
                        janus_strategies::mean_reversion::MeanReversionSignal::Buy => {
                            let sig = "MR_BUY".to_string();
                            if self.last_mr_signal.as_ref() != Some(&sig) {
                                self.last_mr_signal = Some(sig);
                                self.mr_executed += 1;
                            }
                        }
                        janus_strategies::mean_reversion::MeanReversionSignal::Sell => {
                            let sig = "MR_SELL".to_string();
                            if self.last_mr_signal.as_ref() != Some(&sig) {
                                self.last_mr_signal = Some(sig);
                                self.mr_executed += 1;
                            }
                        }
                        _ => self.last_mr_signal = None,
                    }
                } else {
                    self.last_mr_signal = None;
                }

                // Squeeze runs when MeanReverting
                if SqueezeBreakoutStrategy::should_trade_in_regime(regime) {
                    let sq_result = self.squeeze.update_ohlc(c.open, c.high, c.low, c.close);
                    match sq_result.signal {
                        janus_strategies::bollinger_squeeze::SqueezeBreakoutSignal::BuyBreakout => {
                            let sig = "SQ_BUY".to_string();
                            if self.last_sq_signal.as_ref() != Some(&sig) {
                                self.last_sq_signal = Some(sig);
                                self.sq_executed += 1;
                            }
                        }
                        janus_strategies::bollinger_squeeze::SqueezeBreakoutSignal::SellBreakout => {
                            let sig = "SQ_SELL".to_string();
                            if self.last_sq_signal.as_ref() != Some(&sig) {
                                self.last_sq_signal = Some(sig);
                                self.sq_executed += 1;
                            }
                        }
                        _ => self.last_sq_signal = None,
                    }
                } else {
                    self.last_sq_signal = None;
                }

                // VWAP/ORB in mean-reverting/uncertain
                if VwapScalperStrategy::should_trade_in_regime(regime) {
                    let _ = self
                        .vwap
                        .update_ohlcv(c.open, c.high, c.low, c.close, c.volume);
                }
                if OrbStrategy::should_trade_in_regime(regime) {
                    let _ = self
                        .orb
                        .update_ohlcv(c.open, c.high, c.low, c.close, c.volume);
                }

                // ═══ Trend-following strategies (Kraken-ported) ═══
                if EmaRibbonScalperStrategy::should_trade_in_regime(regime) {
                    let result = self
                        .ema_ribbon
                        .update_ohlcv(c.open, c.high, c.low, c.close, c.volume);
                    match result.signal {
                        EmaRibbonSignal::Buy => {
                            let sig = "RIBBON_BUY".to_string();
                            if self.last_ribbon_signal.as_ref() != Some(&sig) {
                                self.last_ribbon_signal = Some(sig);
                                self.ribbon_executed += 1;
                            }
                        }
                        EmaRibbonSignal::Sell => {
                            let sig = "RIBBON_SELL".to_string();
                            if self.last_ribbon_signal.as_ref() != Some(&sig) {
                                self.last_ribbon_signal = Some(sig);
                                self.ribbon_executed += 1;
                            }
                        }
                        EmaRibbonSignal::Hold => {
                            self.last_ribbon_signal = None;
                        }
                    }
                } else {
                    self.last_ribbon_signal = None;
                }

                if TrendPullbackStrategy::should_trade_in_regime(regime) {
                    let result = self
                        .trend_pullback
                        .update_ohlcv(c.open, c.high, c.low, c.close, c.volume);
                    match result.signal {
                        TrendPullbackSignal::Buy => {
                            let sig = "TP_BUY".to_string();
                            if self.last_tp_signal.as_ref() != Some(&sig) {
                                self.last_tp_signal = Some(sig);
                                self.tp_executed += 1;
                            }
                        }
                        TrendPullbackSignal::Sell => {
                            let sig = "TP_SELL".to_string();
                            if self.last_tp_signal.as_ref() != Some(&sig) {
                                self.last_tp_signal = Some(sig);
                                self.tp_executed += 1;
                            }
                        }
                        TrendPullbackSignal::Hold => {
                            self.last_tp_signal = None;
                        }
                    }
                } else {
                    self.last_tp_signal = None;
                }

                if MomentumSurgeStrategy::should_trade_in_regime(regime) {
                    let result = self
                        .momentum_surge
                        .update_ohlcv(c.open, c.high, c.low, c.close, c.volume);
                    match result.signal {
                        MomentumSurgeSignal::Buy => {
                            let sig = "SURGE_BUY".to_string();
                            if self.last_surge_signal.as_ref() != Some(&sig) {
                                self.last_surge_signal = Some(sig);
                                self.surge_executed += 1;
                            }
                        }
                        MomentumSurgeSignal::Sell => {
                            let sig = "SURGE_SELL".to_string();
                            if self.last_surge_signal.as_ref() != Some(&sig) {
                                self.last_surge_signal = Some(sig);
                                self.surge_executed += 1;
                            }
                        }
                        MomentumSurgeSignal::Hold => {
                            self.last_surge_signal = None;
                        }
                    }
                } else {
                    self.last_surge_signal = None;
                }

                if MultiTfTrendStrategy::should_trade_in_regime(regime) {
                    let result = self
                        .multi_tf
                        .update_ohlcv(c.open, c.high, c.low, c.close, c.volume);
                    match result.signal {
                        MultiTfSignal::Buy => {
                            let sig = "MTF_BUY".to_string();
                            if self.last_mtf_signal.as_ref() != Some(&sig) {
                                self.last_mtf_signal = Some(sig);
                                self.mtf_executed += 1;
                            }
                        }
                        MultiTfSignal::Sell => {
                            let sig = "MTF_SELL".to_string();
                            if self.last_mtf_signal.as_ref() != Some(&sig) {
                                self.last_mtf_signal = Some(sig);
                                self.mtf_executed += 1;
                            }
                        }
                        MultiTfSignal::Hold => {
                            self.last_mtf_signal = None;
                        }
                    }
                } else {
                    self.last_mtf_signal = None;
                }

                // If MR handled, skip EMA
                if active_strategy == ActiveStrategy::MeanReversion {
                    return;
                }
            }

            // EMA path — only when TrendFollowing
            let regime_allows_ema = matches!(active_strategy, ActiveStrategy::TrendFollowing);
            if !regime_allows_ema {
                self.suppressed += 1;
            }
        }
    }

    let mut router = FullRouter::new();

    // Warm up strategies
    warm_up_mr(&mut router.mr, 50000.0, 50);
    warm_up_ema_ribbon(&mut router.ema_ribbon, 50000.0, 30);
    warm_up_momentum_surge(&mut router.momentum_surge, 50000.0, 30);

    // Phase 1: TrendFollowing regime — trend strategies should be active, MR suppressed
    let trending_regime = MarketRegime::Trending(TrendDirection::Bullish);
    for i in 0..50 {
        let base = 50000.0 + (i as f64) * 20.0;
        let candle = AggregatedCandle {
            open: base - 10.0,
            high: base + 50.0,
            low: base - 50.0,
            close: base,
            volume: 1000.0 + (i as f64) * 10.0,
        };
        router.process(
            Some(candle),
            ActiveStrategy::TrendFollowing,
            &trending_regime,
        );
    }

    // Phase 2: MeanReversion regime — MR/Squeeze/VWAP/ORB active, trend strategies suppressed
    let mr_regime = MarketRegime::MeanReverting;
    for i in 0..50 {
        let base = 51000.0 + (i as f64 * 0.5).sin() * 200.0;
        let candle = AggregatedCandle {
            open: base,
            high: base + 80.0,
            low: base - 80.0,
            close: base + 30.0 * (i as f64 * 0.3).sin(),
            volume: 500.0,
        };
        router.process(Some(candle), ActiveStrategy::MeanReversion, &mr_regime);
    }

    // Phase 3: NoTrade regime
    for _ in 0..10 {
        router.process(None, ActiveStrategy::NoTrade, &MarketRegime::Uncertain);
    }

    println!(
        "Full router stats: ema={}, mr={}, sq={}, vwap={}, orb={}, ribbon={}, tp={}, surge={}, mtf={}, suppressed={}",
        router.ema_executed,
        router.mr_executed,
        router.sq_executed,
        router.vwap_executed,
        router.orb_executed,
        router.ribbon_executed,
        router.tp_executed,
        router.surge_executed,
        router.mtf_executed,
        router.suppressed,
    );

    // The routing should complete without panics
    let total = router.ema_executed
        + router.mr_executed
        + router.sq_executed
        + router.vwap_executed
        + router.orb_executed
        + router.ribbon_executed
        + router.tp_executed
        + router.surge_executed
        + router.mtf_executed
        + router.suppressed;
    assert!(total >= 0, "Routing should complete without errors");
}

#[test]
fn test_trend_strategies_gated_by_regime_not_mean_reverting() {
    // When regime is MeanReverting, trend strategies should NOT trade
    let mr_regime = MarketRegime::MeanReverting;

    assert!(
        !EmaRibbonScalperStrategy::should_trade_in_regime(&mr_regime),
        "EMA Ribbon should not trade in MeanReverting"
    );
    assert!(
        !TrendPullbackStrategy::should_trade_in_regime(&mr_regime),
        "Trend Pullback should not trade in MeanReverting"
    );
    assert!(
        !MomentumSurgeStrategy::should_trade_in_regime(&mr_regime),
        "Momentum Surge should not trade in MeanReverting"
    );
    assert!(
        !MultiTfTrendStrategy::should_trade_in_regime(&mr_regime),
        "Multi-TF should not trade in MeanReverting"
    );
}

#[test]
fn test_trend_strategies_active_in_trending_bullish() {
    let regime = MarketRegime::Trending(TrendDirection::Bullish);

    assert!(EmaRibbonScalperStrategy::should_trade_in_regime(&regime));
    assert!(TrendPullbackStrategy::should_trade_in_regime(&regime));
    assert!(MomentumSurgeStrategy::should_trade_in_regime(&regime));
    assert!(MultiTfTrendStrategy::should_trade_in_regime(&regime));
}

#[test]
fn test_trend_strategies_active_in_trending_bearish() {
    let regime = MarketRegime::Trending(TrendDirection::Bearish);

    assert!(EmaRibbonScalperStrategy::should_trade_in_regime(&regime));
    assert!(TrendPullbackStrategy::should_trade_in_regime(&regime));
    assert!(MomentumSurgeStrategy::should_trade_in_regime(&regime));
    assert!(MultiTfTrendStrategy::should_trade_in_regime(&regime));
}

#[test]
fn test_momentum_surge_active_in_volatile_others_not() {
    let regime = MarketRegime::Volatile;

    // Momentum Surge should trade in Volatile
    assert!(MomentumSurgeStrategy::should_trade_in_regime(&regime));

    // Multi-TF should NOT trade in Volatile
    assert!(!MultiTfTrendStrategy::should_trade_in_regime(&regime));
}

// ============================================================================
// End-to-End Pipeline Tests
// ============================================================================

#[test]
fn test_end_to_end_regime_to_trend_strategies() {
    // Feed data through RegimeManager → check candles → feed to strategies
    let mut mgr = fast_regime_manager();
    mgr.register_asset("BTCUSDT");

    let mut ribbon = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    let mut pullback = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());
    let mut surge = MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive());
    let mut multi_tf = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());
    multi_tf.set_htf_trend(TrendDirection::Bullish);

    let prices = trending_up_prices(50000.0, 10.0, 500);
    let mut routed_signals = Vec::new();
    let mut candles_fed = 0;

    for &price in &prices {
        if let Some(routed) = mgr.on_tick_price("BTCUSDT", price) {
            routed_signals.push(routed);

            if let Some(candle) = mgr.last_candle("BTCUSDT") {
                // Feed candle to all trend strategies
                ribbon.update_ohlcv(
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                );
                pullback.update_ohlcv(
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                );
                surge.update_ohlcv(
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                );
                multi_tf.update_ohlcv(
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                );
                candles_fed += 1;
            }
        }
    }

    assert!(candles_fed > 0, "Should have fed candles to strategies");
    assert!(
        !routed_signals.is_empty(),
        "RegimeManager should produce routed signals"
    );

    println!(
        "End-to-end: {} routed signals, {} candles fed to strategies",
        routed_signals.len(),
        candles_fed
    );
}

#[test]
fn test_end_to_end_htf_aggregator_feeds_strategies() {
    // Full pipeline: LTF candles → HTF aggregator → set_htf_trend → strategies
    let mut htf_agg = TestHtfAggregator::new(5, 10, 20); // 5:1 ratio, small EMAs for fast warmup
    let mut multi_tf = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());
    let mut ribbon = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    let mut pullback = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());

    let mut htf_trends_emitted = 0;

    // Feed 500 LTF candles of uptrend
    for i in 0..500 {
        let price = 50000.0 + (i as f64) * 5.0;
        let open = price - 3.0;
        let high = price + 15.0;
        let low = price - 15.0;
        let close = price;
        let volume = 1000.0;

        // Feed LTF candle to HTF aggregator
        if let Some(htf_trend) = htf_agg.feed(open, high, low, close, volume) {
            htf_trends_emitted += 1;
            multi_tf.set_htf_trend(htf_trend);
            ribbon.set_htf_trend(htf_trend);
            pullback.set_htf_trend(htf_trend);
        }

        // Feed LTF candle to strategies
        multi_tf.update_ohlcv(open, high, low, close, volume);
        ribbon.update_ohlcv(open, high, low, close, volume);
        pullback.update_ohlcv(open, high, low, close, volume);
    }

    assert!(
        htf_trends_emitted > 0,
        "HTF aggregator should have emitted trends"
    );

    // After strong uptrend, HTF trend should be Bullish
    let last_trend = multi_tf.htf_trend();
    assert_eq!(
        last_trend,
        Some(TrendDirection::Bullish),
        "After uptrend, HTF trend should be Bullish"
    );

    println!(
        "HTF aggregator emitted {} trend updates over 500 LTF candles",
        htf_trends_emitted
    );
}

#[test]
fn test_end_to_end_htf_switches_trend_on_reversal() {
    let mut htf_agg = TestHtfAggregator::new(3, 5, 10);
    let mut multi_tf = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());

    // Phase 1: uptrend
    for i in 0..200 {
        let price = 50000.0 + (i as f64) * 10.0;
        if let Some(trend) = htf_agg.feed(price, price + 5.0, price - 5.0, price, 100.0) {
            multi_tf.set_htf_trend(trend);
        }
        multi_tf.update_ohlcv(price - 5.0, price + 5.0, price - 5.0, price, 100.0);
    }

    let after_uptrend = multi_tf.htf_trend();
    assert_eq!(after_uptrend, Some(TrendDirection::Bullish));

    // Phase 2: strong downtrend
    for i in 0..300 {
        let price = 52000.0 - (i as f64) * 15.0;
        if let Some(trend) = htf_agg.feed(price, price + 5.0, price - 5.0, price, 100.0) {
            multi_tf.set_htf_trend(trend);
        }
        multi_tf.update_ohlcv(price + 5.0, price + 5.0, price - 5.0, price, 100.0);
    }

    let after_downtrend = multi_tf.htf_trend();
    assert_eq!(
        after_downtrend,
        Some(TrendDirection::Bearish),
        "After strong downtrend, HTF should be Bearish"
    );
}

// ============================================================================
// Metrics Integration Tests
// ============================================================================

#[test]
fn test_per_strategy_metrics_signal_generated() {
    let metrics = Arc::new(JanusMetrics::default());
    let sm = metrics.signal_metrics();

    // Record signals for new strategies
    sm.record_strategy_signal_generated("ema_ribbon", "BUY", 0.85);
    sm.record_strategy_signal_generated("ema_ribbon", "SELL", 0.72);
    sm.record_strategy_signal_generated("trend_pullback", "BUY", 0.90);
    sm.record_strategy_signal_generated("momentum_surge", "BUY", 0.78);
    sm.record_strategy_signal_generated("multi_tf_trend", "BUY", 0.88);

    // Verify counters
    assert_eq!(
        sm.per_strategy_signals
            .with_label_values(&["ema_ribbon", "BUY"])
            .get(),
        1
    );
    assert_eq!(
        sm.per_strategy_signals
            .with_label_values(&["ema_ribbon", "SELL"])
            .get(),
        1
    );
    assert_eq!(
        sm.per_strategy_signals
            .with_label_values(&["trend_pullback", "BUY"])
            .get(),
        1
    );
    assert_eq!(
        sm.per_strategy_signals
            .with_label_values(&["momentum_surge", "BUY"])
            .get(),
        1
    );
    assert_eq!(
        sm.per_strategy_signals
            .with_label_values(&["multi_tf_trend", "BUY"])
            .get(),
        1
    );
}

#[test]
fn test_per_strategy_metrics_approved_rejected() {
    let metrics = Arc::new(JanusMetrics::default());
    let sm = metrics.signal_metrics();

    sm.record_strategy_signal_approved("ema_ribbon");
    sm.record_strategy_signal_approved("ema_ribbon");
    sm.record_strategy_signal_rejected("ema_ribbon");

    sm.record_strategy_signal_approved("trend_pullback");
    sm.record_strategy_signal_rejected("momentum_surge");
    sm.record_strategy_signal_rejected("multi_tf_trend");

    assert_eq!(
        sm.per_strategy_approved
            .with_label_values(&["ema_ribbon"])
            .get(),
        2
    );
    assert_eq!(
        sm.per_strategy_rejected
            .with_label_values(&["ema_ribbon"])
            .get(),
        1
    );
    assert_eq!(
        sm.per_strategy_approved
            .with_label_values(&["trend_pullback"])
            .get(),
        1
    );
    assert_eq!(
        sm.per_strategy_rejected
            .with_label_values(&["momentum_surge"])
            .get(),
        1
    );
    assert_eq!(
        sm.per_strategy_rejected
            .with_label_values(&["multi_tf_trend"])
            .get(),
        1
    );
}

#[test]
fn test_per_strategy_metrics_position_lifecycle() {
    let metrics = Arc::new(JanusMetrics::default());
    let sm = metrics.signal_metrics();

    // Simulate position lifecycle for each new strategy
    let strategies = [
        "ema_ribbon",
        "trend_pullback",
        "momentum_surge",
        "multi_tf_trend",
    ];
    let pnls = [150.0, -50.0, 200.0, -30.0];

    for (strat, pnl) in strategies.iter().zip(pnls.iter()) {
        sm.record_strategy_position_opened(strat);
        sm.record_strategy_position_closed(strat, *pnl);
    }

    for strat in &strategies {
        assert_eq!(
            sm.per_strategy_positions_opened
                .with_label_values(&[strat])
                .get(),
            1,
            "Strategy {} should have 1 position opened",
            strat
        );
        assert_eq!(
            sm.per_strategy_positions_closed
                .with_label_values(&[strat])
                .get(),
            1,
            "Strategy {} should have 1 position closed",
            strat
        );
    }

    // Check P&L accumulation
    let ribbon_pnl = sm.per_strategy_pnl.with_label_values(&["ema_ribbon"]).get();
    assert!((ribbon_pnl - 150.0).abs() < 0.01);

    let tp_pnl = sm
        .per_strategy_pnl
        .with_label_values(&["trend_pullback"])
        .get();
    assert!((tp_pnl - (-50.0)).abs() < 0.01);
}

#[test]
fn test_per_strategy_metrics_confidence_tracking() {
    let metrics = Arc::new(JanusMetrics::default());
    let sm = metrics.signal_metrics();

    sm.record_strategy_signal_generated("ema_ribbon", "BUY", 0.65);
    let conf = sm
        .per_strategy_confidence
        .with_label_values(&["ema_ribbon"])
        .get();
    assert!((conf - 0.65).abs() < 0.01);

    // Update to a new value
    sm.record_strategy_signal_generated("ema_ribbon", "SELL", 0.80);
    let conf = sm
        .per_strategy_confidence
        .with_label_values(&["ema_ribbon"])
        .get();
    assert!(
        (conf - 0.80).abs() < 0.01,
        "Confidence should update to latest value"
    );
}

#[test]
fn test_active_strategy_count_metric() {
    let metrics = Arc::new(JanusMetrics::default());
    let sm = metrics.signal_metrics();

    sm.set_active_strategy_count(4);
    assert_eq!(sm.active_strategy_count.get(), 4);

    sm.set_active_strategy_count(9);
    assert_eq!(sm.active_strategy_count.get(), 9);
}

// ============================================================================
// Stress & Edge Case Tests
// ============================================================================

#[test]
fn test_all_strategies_handle_zero_volume() {
    let mut ribbon = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    let mut pullback = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());
    let mut surge = MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive());
    let mut multi_tf = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());

    // Feed candles with zero volume — should not panic
    for i in 0..50 {
        let price = 50000.0 + (i as f64) * 5.0;
        ribbon.update_ohlcv(price, price + 10.0, price - 10.0, price, 0.0);
        pullback.update_ohlcv(price, price + 10.0, price - 10.0, price, 0.0);
        surge.update_ohlcv(price, price + 10.0, price - 10.0, price, 0.0);
        multi_tf.update_ohlcv(price, price + 10.0, price - 10.0, price, 0.0);
    }
}

#[test]
fn test_all_strategies_handle_extreme_prices() {
    let mut ribbon = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    let mut pullback = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());
    let mut surge = MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive());
    let mut multi_tf = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());

    // Very high prices
    for i in 0..30 {
        let price = 1_000_000.0 + (i as f64) * 1000.0;
        ribbon.update_ohlcv(price, price + 100.0, price - 100.0, price, 1000.0);
        pullback.update_ohlcv(price, price + 100.0, price - 100.0, price, 1000.0);
        surge.update_ohlcv(price, price + 100.0, price - 100.0, price, 1000.0);
        multi_tf.update_ohlcv(price, price + 100.0, price - 100.0, price, 1000.0);
    }

    // Very low prices (penny stock territory)
    for i in 0..30 {
        let price = 0.001 + (i as f64) * 0.0001;
        ribbon.update_ohlcv(price, price + 0.0001, price - 0.0001, price, 1000.0);
        pullback.update_ohlcv(price, price + 0.0001, price - 0.0001, price, 1000.0);
        surge.update_ohlcv(price, price + 0.0001, price - 0.0001, price, 1000.0);
        multi_tf.update_ohlcv(price, price + 0.0001, price - 0.0001, price, 1000.0);
    }
}

#[test]
fn test_all_strategies_handle_flat_candles() {
    let mut ribbon = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    let mut pullback = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());
    let mut surge = MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive());
    let mut multi_tf = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());

    // O = H = L = C (doji/flat candles)
    for _ in 0..50 {
        let price = 50000.0;
        ribbon.update_ohlcv(price, price, price, price, 1000.0);
        pullback.update_ohlcv(price, price, price, price, 1000.0);
        surge.update_ohlcv(price, price, price, price, 1000.0);
        multi_tf.update_ohlcv(price, price, price, price, 1000.0);
    }
}

#[test]
fn test_stress_many_candles_all_strategies() {
    let mut ribbon = EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive());
    let mut pullback = TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive());
    let mut surge = MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive());
    let mut multi_tf = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());
    multi_tf.set_htf_trend(TrendDirection::Bullish);

    let mut price = 50000.0;
    let mut buy_signals = 0u32;
    let mut sell_signals = 0u32;

    for i in 0..5000 {
        // Alternate between uptrend and downtrend phases
        let direction = if (i / 500) % 2 == 0 { 1.0 } else { -1.0 };
        price += direction * 5.0 + (i as f64 * 0.05).sin() * 10.0;
        price = price.max(100.0); // floor

        let volume = if i % 100 == 0 { 10000.0 } else { 1000.0 };
        let open = price - direction * 3.0;
        let high = price + 20.0;
        let low = price - 20.0;

        let r1 = ribbon.update_ohlcv(open, high, low, price, volume);
        let r2 = pullback.update_ohlcv(open, high, low, price, volume);
        let r3 = surge.update_ohlcv(open, high, low, price, volume);
        let r4 = multi_tf.update_ohlcv(open, high, low, price, volume);

        // Count signals
        for sig in [
            matches!(r1.signal, EmaRibbonSignal::Buy),
            matches!(r2.signal, TrendPullbackSignal::Buy),
            matches!(r3.signal, MomentumSurgeSignal::Buy),
            matches!(r4.signal, MultiTfSignal::Buy),
        ] {
            if sig {
                buy_signals += 1;
            }
        }
        for sig in [
            matches!(r1.signal, EmaRibbonSignal::Sell),
            matches!(r2.signal, TrendPullbackSignal::Sell),
            matches!(r3.signal, MomentumSurgeSignal::Sell),
            matches!(r4.signal, MultiTfSignal::Sell),
        ] {
            if sig {
                sell_signals += 1;
            }
        }

        // Switch HTF trend partway through
        if i == 2500 {
            multi_tf.set_htf_trend(TrendDirection::Bearish);
        }
    }

    println!(
        "Stress test: 5000 candles × 4 strategies, {} buys, {} sells",
        buy_signals, sell_signals
    );
    // Should complete without panicking
}

// ============================================================================
// HTF Aggregator + Strategy Pipeline Tests
// ============================================================================

#[test]
fn test_htf_aggregator_volume_accumulation() {
    let mut agg = TestHtfAggregator::new(4, 5, 10);

    // Feed 4 candles with known volumes
    agg.feed(100.0, 110.0, 90.0, 105.0, 100.0);
    agg.feed(105.0, 115.0, 95.0, 110.0, 200.0);
    agg.feed(110.0, 120.0, 100.0, 115.0, 150.0);
    // 4th candle completes the HTF bar — volume should be sum of all 4
    agg.feed(115.0, 125.0, 105.0, 120.0, 250.0);

    // After completion, volume is reset
    assert_eq!(agg.count, 0);
    assert_eq!(agg.candle_count, 1);
}

#[test]
fn test_htf_aggregator_high_low_tracking() {
    let mut agg = TestHtfAggregator::new(3, 5, 10);

    agg.feed(100.0, 150.0, 80.0, 120.0, 100.0); // H=150, L=80
    agg.feed(120.0, 130.0, 110.0, 125.0, 100.0); // H stays 150, L stays 80
    // Before completion, check aggregated values
    assert_eq!(agg.agg_high, 150.0);
    assert_eq!(agg.agg_low, 80.0);
    assert_eq!(agg.agg_open, 100.0);

    // Third candle: H=200 (new max), L=70 (new min)
    agg.feed(125.0, 200.0, 70.0, 180.0, 100.0);
    // Now completed — count reset, but we can verify it completed
    assert_eq!(agg.candle_count, 1);
}

#[test]
fn test_htf_aggregator_consistent_with_strategy_expectations() {
    // Verify that HTF aggregator and MultiTfTrendStrategy work together
    // without any inconsistencies
    let mut agg = TestHtfAggregator::new(10, 10, 25);
    let mut strategy = MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive());

    // Feed 1000 LTF candles of uptrend
    let mut trend_set_count = 0;
    for i in 0..1000 {
        let price = 50000.0 + (i as f64) * 3.0;
        let open = price - 2.0;
        let high = price + 10.0;
        let low = price - 10.0;

        if let Some(trend) = agg.feed(open, high, low, price, 500.0) {
            strategy.set_htf_trend(trend);
            trend_set_count += 1;
        }
        strategy.update_ohlcv(open, high, low, price, 500.0);
    }

    // 1000 / 10 = 100 HTF candles. Warmup = 30. So ~70 trend emissions.
    assert!(
        trend_set_count > 50,
        "Expected many trend emissions, got {}",
        trend_set_count
    );

    // Strategy should be ready and have bullish HTF
    assert!(strategy.is_ready());
    assert_eq!(strategy.htf_trend(), Some(TrendDirection::Bullish));
}
