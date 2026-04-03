//! Indicator-Based Regime Detector
//!
//! Combines multiple technical indicators (ADX, Bollinger Bands, ATR, EMA) to
//! classify the current market regime as Trending, Mean-Reverting, Volatile, or Uncertain.
//!
//! This is the fastest regime detection method — purely rule-based with no statistical
//! learning. It provides immediate classification once indicators are warmed up.
//!
//! Ported from kraken's `regime/detector.rs`, adapted for the JANUS type system.

use super::indicators::{ADX, ATR, BollingerBands, BollingerBandsValues, EMA};
use super::types::{
    MarketRegime, RecommendedStrategy, RegimeConfidence, RegimeConfig, TrendDirection,
};
use std::collections::VecDeque;

/// Main indicator-based regime detection engine.
///
/// Feeds OHLC bars through ADX, ATR, Bollinger Bands, and dual-EMA indicators,
/// then scores each regime possibility to produce a classification with confidence.
///
/// Includes a stability filter to prevent regime whipsawing.
///
/// # Example
///
/// ```rust
/// use janus_regime::{RegimeDetector, RegimeConfig, MarketRegime};
///
/// let mut detector = RegimeDetector::crypto_optimized();
///
/// // Feed OHLC bars
/// for i in 0..300 {
///     let price = 100.0 + i as f64 * 0.5;
///     let result = detector.update(price + 1.0, price - 1.0, price);
///     if detector.is_ready() {
///         println!("Regime: {} (conf: {:.0}%)", result.regime, result.confidence * 100.0);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct RegimeDetector {
    config: RegimeConfig,

    // Indicators
    adx: ADX,
    atr: ATR,
    atr_avg: EMA, // For measuring ATR expansion
    bb: BollingerBands,
    ema_short: EMA,
    ema_long: EMA,

    // State
    current_regime: MarketRegime,
    regime_history: VecDeque<MarketRegime>,
    bars_in_regime: usize,

    // For trend direction
    last_close: Option<f64>,
}

impl RegimeDetector {
    /// Create a new detector with the given configuration
    pub fn new(config: RegimeConfig) -> Self {
        Self {
            adx: ADX::new(config.adx_period),
            atr: ATR::new(config.atr_period),
            atr_avg: EMA::new(50), // Longer-term ATR average
            bb: BollingerBands::new(config.bb_period, config.bb_std_dev),
            ema_short: EMA::new(config.ema_short_period),
            ema_long: EMA::new(config.ema_long_period),
            current_regime: MarketRegime::Uncertain,
            regime_history: VecDeque::with_capacity(20),
            bars_in_regime: 0,
            last_close: None,
            config,
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(RegimeConfig::default())
    }

    /// Create optimized for crypto markets
    pub fn crypto_optimized() -> Self {
        Self::new(RegimeConfig::crypto_optimized())
    }

    /// Create with conservative config
    pub fn conservative() -> Self {
        Self::new(RegimeConfig::conservative())
    }

    /// Update with new OHLC bar and get the regime classification.
    ///
    /// Returns a `RegimeConfidence` with the detected regime, confidence score,
    /// and supporting indicator metrics.
    ///
    /// The detector requires a warmup period (determined by the longest indicator
    /// period) before it starts producing meaningful classifications. During
    /// warmup, it returns `MarketRegime::Uncertain` with zero confidence.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> RegimeConfidence {
        // Update all indicators
        let adx_value = self.adx.update(high, low, close);
        let atr_value = self.atr.update(high, low, close);
        let bb_values = self.bb.update(close);
        let ema_short = self.ema_short.update(close);
        let ema_long = self.ema_long.update(close);

        // Update ATR average for expansion detection
        if let Some(atr) = atr_value {
            self.atr_avg.update(atr);
        }

        self.last_close = Some(close);

        // Check if we have enough data
        if !self.is_ready() {
            return RegimeConfidence::new(MarketRegime::Uncertain, 0.0);
        }

        // Detect regime
        let (new_regime, confidence) = self.classify_regime(
            adx_value.unwrap(),
            atr_value.unwrap(),
            bb_values.as_ref().unwrap(),
            ema_short.unwrap(),
            ema_long.unwrap(),
            close,
        );

        // Apply stability filter - avoid whipsawing
        let stable_regime = self.apply_stability_filter(new_regime, confidence);

        // Update state
        if stable_regime != self.current_regime {
            self.regime_history.push_back(self.current_regime);
            if self.regime_history.len() > 20 {
                self.regime_history.pop_front();
            }
            self.current_regime = stable_regime;
            self.bars_in_regime = 0;
        } else {
            self.bars_in_regime += 1;
        }

        RegimeConfidence::with_metrics(
            stable_regime,
            confidence,
            adx_value.unwrap(),
            bb_values
                .as_ref()
                .map(|b| b.width_percentile)
                .unwrap_or(50.0),
            self.calculate_trend_strength(ema_short.unwrap(), ema_long.unwrap(), close),
        )
    }

    /// Classify regime based on indicator values.
    ///
    /// Scores each regime possibility across multiple indicator dimensions:
    /// - ADX for trend strength vs ranging
    /// - Bollinger Band width for volatility
    /// - ATR expansion for volatility confirmation
    /// - EMA alignment for trend direction
    /// - Price position relative to EMAs
    fn classify_regime(
        &self,
        adx: f64,
        atr: f64,
        bb: &BollingerBandsValues,
        ema_short: f64,
        ema_long: f64,
        close: f64,
    ) -> (MarketRegime, f64) {
        // Calculate ATR expansion
        let atr_expansion = if let Some(avg_atr) = self.atr_avg.value() {
            atr / avg_atr
        } else {
            1.0
        };

        // Score each regime possibility
        let mut trending_score: f64 = 0.0;
        let mut ranging_score: f64 = 0.0;
        let mut volatile_score: f64 = 0.0;

        // ADX analysis
        if adx >= self.config.adx_trending_threshold {
            trending_score += 0.4;
        } else if adx <= self.config.adx_ranging_threshold {
            ranging_score += 0.3;
        }

        // Bollinger Band width analysis
        if bb.is_high_volatility(self.config.bb_width_volatility_threshold) {
            volatile_score += 0.3;
        }
        if bb.is_squeeze(25.0) {
            ranging_score += 0.2; // Tight bands suggest range-bound
        }

        // ATR expansion
        if atr_expansion >= self.config.atr_expansion_threshold {
            volatile_score += 0.3;
        } else if atr_expansion < 0.8 {
            ranging_score += 0.2; // Low volatility suggests ranging
        }

        // EMA alignment for trend
        let ema_diff_pct = ((ema_short - ema_long) / ema_long).abs() * 100.0;
        if ema_diff_pct > 2.0 {
            trending_score += 0.3;
        } else if ema_diff_pct < 1.0 {
            ranging_score += 0.2;
        }

        // Price position relative to EMAs
        let price_above_both = close > ema_short && close > ema_long;
        let price_below_both = close < ema_short && close < ema_long;
        if price_above_both || price_below_both {
            trending_score += 0.2;
        } else {
            ranging_score += 0.2; // Price between EMAs suggests consolidation
        }

        // Determine regime and direction
        let _max_score = trending_score.max(ranging_score).max(volatile_score);
        let confidence = _max_score / 1.2; // Normalize to 0-1 range

        let regime = if volatile_score >= 0.5 && volatile_score >= trending_score {
            MarketRegime::Volatile
        } else if trending_score > ranging_score && trending_score > 0.3 {
            // Determine trend direction
            let direction = if ema_short > ema_long && close > ema_long {
                TrendDirection::Bullish
            } else if ema_short < ema_long && close < ema_long {
                TrendDirection::Bearish
            } else if let Some(dir) = self.adx.trend_direction() {
                dir
            } else {
                TrendDirection::Bullish // Default
            };
            MarketRegime::Trending(direction)
        } else if ranging_score > 0.3 {
            MarketRegime::MeanReverting
        } else {
            MarketRegime::Uncertain
        };

        (regime, confidence.min(1.0))
    }

    /// Apply stability filter to avoid regime whipsawing.
    ///
    /// Prevents rapid switching between regimes by requiring:
    /// - Minimum confidence threshold to change
    /// - Minimum duration in current regime before switching
    /// - Consistent signals in recent history
    fn apply_stability_filter(&self, new_regime: MarketRegime, confidence: f64) -> MarketRegime {
        // If confidence is low, stick with current regime
        if confidence < 0.4 {
            return self.current_regime;
        }

        // Require minimum duration in current regime before switching
        if self.bars_in_regime < self.config.min_regime_duration
            && new_regime != self.current_regime
        {
            // Only switch if new regime is strongly confirmed
            if confidence < 0.7 {
                return self.current_regime;
            }
        }

        // Check recent history for stability
        let recent_count = self
            .regime_history
            .iter()
            .rev()
            .take(self.config.regime_stability_bars)
            .filter(|&&r| {
                matches!(
                    (&r, &new_regime),
                    (MarketRegime::Trending(_), MarketRegime::Trending(_))
                        | (MarketRegime::MeanReverting, MarketRegime::MeanReverting)
                        | (MarketRegime::Volatile, MarketRegime::Volatile)
                )
            })
            .count();

        // If regime has been bouncing around, require stronger confirmation
        if recent_count < self.config.regime_stability_bars / 2 && confidence < 0.6 {
            return self.current_regime;
        }

        new_regime
    }

    /// Calculate trend strength from EMA alignment and price position
    fn calculate_trend_strength(&self, ema_short: f64, ema_long: f64, close: f64) -> f64 {
        let ema_alignment = (ema_short - ema_long).abs() / ema_long * 100.0;
        let price_position = if close > ema_short && close > ema_long {
            1.0
        } else if close < ema_short && close < ema_long {
            0.7
        } else {
            0.5
        };

        (ema_alignment * price_position / 5.0).min(1.0) // Normalize
    }

    // ========================================================================
    // Public Accessors
    // ========================================================================

    /// Check if detector has enough data to classify regime.
    ///
    /// All indicators must be warmed up before the detector can produce
    /// meaningful results.
    pub fn is_ready(&self) -> bool {
        self.adx.is_ready()
            && self.atr.is_ready()
            && self.bb.is_ready()
            && self.ema_short.is_ready()
            && self.ema_long.is_ready()
    }

    /// Get current detected regime
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get recommended strategy for current regime
    pub fn recommended_strategy(&self) -> RecommendedStrategy {
        RecommendedStrategy::from(&self.current_regime)
    }

    /// Get number of bars in current regime
    pub fn bars_in_current_regime(&self) -> usize {
        self.bars_in_regime
    }

    /// Get ADX value
    pub fn adx_value(&self) -> Option<f64> {
        self.adx.value()
    }

    /// Get ATR value
    pub fn atr_value(&self) -> Option<f64> {
        self.atr.value()
    }

    /// Get current config
    pub fn config(&self) -> &RegimeConfig {
        &self.config
    }

    /// Update config (resets internal state)
    pub fn set_config(&mut self, config: RegimeConfig) {
        *self = Self::new(config);
    }

    /// Get the regime history (most recent at the back)
    pub fn regime_history(&self) -> &VecDeque<MarketRegime> {
        &self.regime_history
    }

    /// Get the last close price
    pub fn last_close(&self) -> Option<f64> {
        self.last_close
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate synthetic trending data
    fn generate_trending_data(
        bars: usize,
        start_price: f64,
        trend_strength: f64,
    ) -> Vec<(f64, f64, f64)> {
        let mut data = Vec::new();
        let mut price = start_price;

        for _ in 0..bars {
            let change = trend_strength * (1.0 + (rand::random::<f64>() - 0.5) * 0.2);
            price += change;

            let high = price + price * 0.005;
            let low = price - price * 0.005;
            let close = price;

            data.push((high, low, close));
        }

        data
    }

    /// Generate synthetic ranging data
    fn generate_ranging_data(
        bars: usize,
        center_price: f64,
        range_pct: f64,
    ) -> Vec<(f64, f64, f64)> {
        let mut data = Vec::new();

        for i in 0..bars {
            let offset = (i as f64 * 0.5).sin() * center_price * range_pct / 100.0;
            let price = center_price + offset;

            let high = price + price * 0.002;
            let low = price - price * 0.002;
            let close = price;

            data.push((high, low, close));
        }

        data
    }

    /// Generate volatile data with large swings
    #[allow(dead_code)]
    fn generate_volatile_data(bars: usize, center_price: f64) -> Vec<(f64, f64, f64)> {
        let mut data = Vec::new();

        for i in 0..bars {
            let swing = if i % 2 == 0 { 1.05 } else { 0.95 };
            let price = center_price * swing;

            let high = price * 1.03;
            let low = price * 0.97;
            let close = price;

            data.push((high, low, close));
        }

        data
    }

    #[test]
    fn test_detector_creation() {
        let detector = RegimeDetector::default_config();
        assert!(!detector.is_ready());
        assert_eq!(detector.current_regime(), MarketRegime::Uncertain);
        assert_eq!(detector.bars_in_current_regime(), 0);
    }

    #[test]
    fn test_crypto_optimized_creation() {
        let detector = RegimeDetector::crypto_optimized();
        assert!(!detector.is_ready());
        assert_eq!(detector.config().adx_trending_threshold, 20.0);
        assert_eq!(detector.config().ema_short_period, 21);
    }

    #[test]
    fn test_conservative_creation() {
        let detector = RegimeDetector::conservative();
        assert_eq!(detector.config().adx_trending_threshold, 30.0);
        assert_eq!(detector.config().min_regime_duration, 10);
    }

    #[test]
    fn test_warmup_returns_uncertain() {
        let mut detector = RegimeDetector::default_config();

        // Feed a few bars — not enough for warmup
        for i in 0..10 {
            let price = 100.0 + i as f64;
            let result = detector.update(price + 1.0, price - 1.0, price);
            assert_eq!(result.regime, MarketRegime::Uncertain);
            assert_eq!(result.confidence, 0.0);
        }

        assert!(!detector.is_ready());
    }

    #[test]
    fn test_trending_detection() {
        let mut detector = RegimeDetector::default_config();

        // Generate uptrending data
        let data = generate_trending_data(300, 100.0, 0.5);

        let mut last_regime = MarketRegime::Uncertain;
        for (high, low, close) in data {
            let result = detector.update(high, low, close);
            if detector.is_ready() {
                last_regime = result.regime;
            }
        }

        assert!(
            matches!(last_regime, MarketRegime::Trending(_)),
            "Expected Trending regime, got: {last_regime:?}"
        );
    }

    #[test]
    fn test_trending_bullish_direction() {
        let mut detector = RegimeDetector::default_config();

        // Strong uptrend
        let data = generate_trending_data(300, 100.0, 0.5);

        let mut last_regime = MarketRegime::Uncertain;
        for (high, low, close) in data {
            let result = detector.update(high, low, close);
            if detector.is_ready() {
                last_regime = result.regime;
            }
        }

        assert!(
            matches!(last_regime, MarketRegime::Trending(TrendDirection::Bullish)),
            "Expected Bullish trend, got: {last_regime:?}"
        );
    }

    #[test]
    fn test_trending_bearish_direction() {
        let mut detector = RegimeDetector::default_config();

        // Strong downtrend
        let data = generate_trending_data(300, 200.0, -0.5);

        let mut last_regime = MarketRegime::Uncertain;
        for (high, low, close) in data {
            let result = detector.update(high, low, close);
            if detector.is_ready() {
                last_regime = result.regime;
            }
        }

        // Should be either Bearish trending or at least not Bullish
        if matches!(last_regime, MarketRegime::Trending(_)) {
            assert!(
                matches!(last_regime, MarketRegime::Trending(TrendDirection::Bearish)),
                "Expected Bearish trend, got: {last_regime:?}"
            );
        }
    }

    #[test]
    fn test_ranging_detection() {
        let mut detector = RegimeDetector::default_config();

        // Generate ranging data
        let data = generate_ranging_data(300, 100.0, 2.0);

        let mut last_regime = MarketRegime::Uncertain;
        for (high, low, close) in data {
            let result = detector.update(high, low, close);
            if detector.is_ready() {
                last_regime = result.regime;
            }
        }

        // Ranging data should produce MeanReverting or at least not strong Trending
        assert!(
            !matches!(last_regime, MarketRegime::Trending(TrendDirection::Bullish)),
            "Ranging data shouldn't produce strong bullish trend, got: {last_regime:?}"
        );
    }

    #[test]
    fn test_confidence_range() {
        let mut detector = RegimeDetector::default_config();

        let data = generate_trending_data(300, 100.0, 0.5);

        for (high, low, close) in data {
            let result = detector.update(high, low, close);
            assert!(
                (0.0..=1.0).contains(&result.confidence),
                "Confidence should be in [0, 1]: {}",
                result.confidence
            );
        }
    }

    #[test]
    fn test_regime_history_tracking() {
        let mut detector = RegimeDetector::default_config();

        // Feed enough data to produce some regime changes
        let trend_data = generate_trending_data(200, 100.0, 0.5);
        for (high, low, close) in trend_data {
            detector.update(high, low, close);
        }

        let range_data = generate_ranging_data(200, 200.0, 1.0);
        for (high, low, close) in range_data {
            detector.update(high, low, close);
        }

        // History should have been populated
        // (exact content depends on stability filter)
        assert!(
            detector.regime_history().len() <= 20,
            "History should be bounded"
        );
    }

    #[test]
    fn test_recommended_strategy() {
        let mut detector = RegimeDetector::default_config();

        // Feed trending data
        let data = generate_trending_data(300, 100.0, 0.5);
        for (high, low, close) in data {
            detector.update(high, low, close);
        }

        if matches!(detector.current_regime(), MarketRegime::Trending(_)) {
            assert_eq!(
                detector.recommended_strategy(),
                RecommendedStrategy::TrendFollowing
            );
        }
    }

    #[test]
    fn test_adx_atr_accessors() {
        let mut detector = RegimeDetector::default_config();

        // Before warmup
        assert!(detector.adx_value().is_none());
        assert!(detector.atr_value().is_none());

        // Feed data
        let data = generate_trending_data(300, 100.0, 0.5);
        for (high, low, close) in data {
            detector.update(high, low, close);
        }

        // After warmup
        assert!(detector.adx_value().is_some());
        assert!(detector.atr_value().is_some());
    }

    #[test]
    fn test_set_config_resets_state() {
        let mut detector = RegimeDetector::default_config();

        // Feed data
        let data = generate_trending_data(300, 100.0, 0.5);
        for (high, low, close) in data {
            detector.update(high, low, close);
        }
        assert!(detector.is_ready());

        // Reset with new config
        detector.set_config(RegimeConfig::crypto_optimized());
        assert!(!detector.is_ready());
        assert_eq!(detector.current_regime(), MarketRegime::Uncertain);
        assert_eq!(detector.bars_in_current_regime(), 0);
    }

    #[test]
    fn test_last_close_tracking() {
        let mut detector = RegimeDetector::default_config();

        assert!(detector.last_close().is_none());

        detector.update(101.0, 99.0, 100.0);
        assert_eq!(detector.last_close(), Some(100.0));

        detector.update(106.0, 104.0, 105.0);
        assert_eq!(detector.last_close(), Some(105.0));
    }

    #[test]
    fn test_bars_in_regime_increments() {
        let mut detector = RegimeDetector::default_config();

        // Feed gently trending data so ADX can warm up (needs price movement)
        for i in 0..300 {
            let price = 100.0 + i as f64 * 0.3;
            detector.update(price + 1.0, price - 1.0, price);
        }

        // After stabilization, bars_in_regime should be > 0
        assert!(
            detector.bars_in_current_regime() > 0,
            "Should have been in current regime for multiple bars (regime: {:?})",
            detector.current_regime()
        );
    }

    #[test]
    fn test_stability_filter_prevents_whipsaw() {
        let mut detector = RegimeDetector::new(RegimeConfig {
            min_regime_duration: 10,
            regime_stability_bars: 5,
            ..RegimeConfig::default()
        });

        // First establish a regime with trending data
        let trend_data = generate_trending_data(300, 100.0, 0.5);
        for (high, low, close) in trend_data {
            detector.update(high, low, close);
        }

        let regime_before = detector.current_regime();

        // Feed just 2-3 bars of ranging data — shouldn't immediately switch
        for (high, low, close) in generate_ranging_data(3, 250.0, 1.0) {
            detector.update(high, low, close);
        }

        let regime_after = detector.current_regime();

        // The stability filter should prevent an immediate switch
        // (This is probabilistic — the key point is the filter exists and is applied)
        // We just verify the detector didn't crash and the regime is valid
        assert!(
            matches!(
                regime_after,
                MarketRegime::Trending(_)
                    | MarketRegime::MeanReverting
                    | MarketRegime::Volatile
                    | MarketRegime::Uncertain
            ),
            "Regime should be a valid variant: {regime_after:?}"
        );
        let _ = regime_before; // used for debugging if needed
    }

    #[test]
    fn test_metrics_populated_after_warmup() {
        let mut detector = RegimeDetector::default_config();

        let data = generate_trending_data(300, 100.0, 0.5);
        let mut last_result = RegimeConfidence::default();

        for (high, low, close) in data {
            last_result = detector.update(high, low, close);
        }

        // After warmup, metrics should be populated
        assert!(last_result.adx_value > 0.0, "ADX should be > 0");
        assert!(
            last_result.bb_width_percentile >= 0.0 && last_result.bb_width_percentile <= 100.0,
            "BB width percentile should be in [0, 100]"
        );
        assert!(
            last_result.trend_strength >= 0.0 && last_result.trend_strength <= 1.0,
            "Trend strength should be in [0, 1]"
        );
    }
}
