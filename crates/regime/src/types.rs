//! Regime Types for JANUS
//!
//! Core types for market regime detection and classification.
//! These types are the shared vocabulary between the indicator-based detector,
//! HMM detector, ensemble detector, and the strategy router.

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Market Regime
// ============================================================================

/// Market regime classification.
///
/// Classifies the current market state to guide strategy selection.
/// Each regime maps to a recommended trading approach.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong directional movement — use trend-following strategies.
    /// Characteristics: High ADX (>25), price above/below MAs, clear momentum.
    Trending(TrendDirection),

    /// Price oscillating around a mean — use mean reversion strategies.
    /// Characteristics: Low ADX (<20), price within Bollinger Bands, range-bound.
    MeanReverting,

    /// High volatility, no clear direction — reduce exposure or stay cash.
    /// Characteristics: ATR expansion, wide Bollinger Bands, choppy price action.
    Volatile,

    /// Insufficient data or unclear signals — be cautious.
    #[default]
    Uncertain,
}

impl MarketRegime {
    /// Whether this regime favors entering new positions
    pub fn is_tradeable(&self) -> bool {
        matches!(
            self,
            MarketRegime::Trending(_) | MarketRegime::MeanReverting
        )
    }

    /// Suggested position size multiplier (1.0 = normal, 0.0 = no trade)
    pub fn size_multiplier(&self) -> f64 {
        match self {
            MarketRegime::Trending(_) => 1.0,
            MarketRegime::MeanReverting => 0.8,
            MarketRegime::Volatile => 0.3,
            MarketRegime::Uncertain => 0.0,
        }
    }

    /// Get the recommended strategy category
    pub fn recommended_strategy(&self) -> RecommendedStrategy {
        RecommendedStrategy::from(self)
    }
}

impl fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketRegime::Trending(TrendDirection::Bullish) => write!(f, "Trending (Bullish)"),
            MarketRegime::Trending(TrendDirection::Bearish) => write!(f, "Trending (Bearish)"),
            MarketRegime::MeanReverting => write!(f, "Mean-Reverting"),
            MarketRegime::Volatile => write!(f, "Volatile/Choppy"),
            MarketRegime::Uncertain => write!(f, "Uncertain"),
        }
    }
}

// ============================================================================
// Trend Direction
// ============================================================================

/// Direction of the trend when in Trending regime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrendDirection {
    Bullish,
    Bearish,
}

impl fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrendDirection::Bullish => write!(f, "Bullish"),
            TrendDirection::Bearish => write!(f, "Bearish"),
        }
    }
}

// ============================================================================
// Regime Confidence
// ============================================================================

/// Confidence level in regime classification.
///
/// Bundles the detected regime with a confidence score and supporting metrics
/// from the indicators that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegimeConfidence {
    /// Detected regime
    pub regime: MarketRegime,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// ADX value at detection time
    pub adx_value: f64,
    /// Bollinger Band width percentile (0–100)
    pub bb_width_percentile: f64,
    /// Trend strength metric
    pub trend_strength: f64,
}

impl RegimeConfidence {
    /// Create a basic regime confidence with just regime and confidence score
    pub fn new(regime: MarketRegime, confidence: f64) -> Self {
        Self {
            regime,
            confidence: confidence.clamp(0.0, 1.0),
            adx_value: 0.0,
            bb_width_percentile: 0.0,
            trend_strength: 0.0,
        }
    }

    /// Create with full indicator metrics
    pub fn with_metrics(
        regime: MarketRegime,
        confidence: f64,
        adx: f64,
        bb_width: f64,
        trend_strength: f64,
    ) -> Self {
        Self {
            regime,
            confidence: confidence.clamp(0.0, 1.0),
            adx_value: adx,
            bb_width_percentile: bb_width,
            trend_strength,
        }
    }

    /// Whether confidence is high enough to act on (>= 0.6)
    pub fn is_actionable(&self) -> bool {
        self.confidence >= 0.6
    }

    /// Whether confidence is strong (>= 0.75)
    pub fn is_strong(&self) -> bool {
        self.confidence >= 0.75
    }
}

impl Default for RegimeConfidence {
    fn default() -> Self {
        Self::new(MarketRegime::Uncertain, 0.0)
    }
}

impl fmt::Display for RegimeConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (conf: {:.0}%, ADX: {:.1}, BB%: {:.0}, trend: {:.2})",
            self.regime,
            self.confidence * 100.0,
            self.adx_value,
            self.bb_width_percentile,
            self.trend_strength,
        )
    }
}

// ============================================================================
// Recommended Strategy
// ============================================================================

/// Recommended strategy type based on regime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecommendedStrategy {
    /// Use trend-following (Golden Cross, EMA Pullback, Momentum)
    TrendFollowing,
    /// Use mean reversion (Bollinger Bands, VWAP)
    MeanReversion,
    /// Reduce position size, tight stops
    ReducedExposure,
    /// Stay in cash, wait for clarity
    StayCash,
}

impl From<&MarketRegime> for RecommendedStrategy {
    fn from(regime: &MarketRegime) -> Self {
        match regime {
            MarketRegime::Trending(_) => RecommendedStrategy::TrendFollowing,
            MarketRegime::MeanReverting => RecommendedStrategy::MeanReversion,
            MarketRegime::Volatile => RecommendedStrategy::ReducedExposure,
            MarketRegime::Uncertain => RecommendedStrategy::StayCash,
        }
    }
}

impl fmt::Display for RecommendedStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecommendedStrategy::TrendFollowing => write!(f, "Trend Following"),
            RecommendedStrategy::MeanReversion => write!(f, "Mean Reversion"),
            RecommendedStrategy::ReducedExposure => write!(f, "Reduced Exposure"),
            RecommendedStrategy::StayCash => write!(f, "Stay Cash"),
        }
    }
}

// ============================================================================
// Regime Config
// ============================================================================

/// Configuration for indicator-based regime detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// ADX period for trend strength
    pub adx_period: usize,
    /// ADX threshold above which market is considered trending
    pub adx_trending_threshold: f64,
    /// ADX threshold below which market is considered mean-reverting
    pub adx_ranging_threshold: f64,

    /// Bollinger Bands period
    pub bb_period: usize,
    /// Bollinger Bands standard deviation multiplier
    pub bb_std_dev: f64,
    /// BB width percentile threshold for high volatility
    pub bb_width_volatility_threshold: f64,

    /// EMA periods for trend direction
    pub ema_short_period: usize,
    pub ema_long_period: usize,

    /// ATR period for volatility measurement
    pub atr_period: usize,
    /// ATR expansion multiplier (current vs average) for volatile regime
    pub atr_expansion_threshold: f64,

    /// Lookback period for regime stability (avoid whipsaws)
    pub regime_stability_bars: usize,
    /// Minimum bars in current regime before switching
    pub min_regime_duration: usize,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            adx_period: 14,
            adx_trending_threshold: 25.0,
            adx_ranging_threshold: 20.0,
            bb_period: 20,
            bb_std_dev: 2.0,
            bb_width_volatility_threshold: 75.0, // percentile
            ema_short_period: 50,
            ema_long_period: 200,
            atr_period: 14,
            atr_expansion_threshold: 1.5,
            regime_stability_bars: 3,
            min_regime_duration: 5,
        }
    }
}

impl RegimeConfig {
    /// Configuration optimized for crypto markets (BTC, ETH, SOL)
    pub fn crypto_optimized() -> Self {
        Self {
            adx_period: 14,
            adx_trending_threshold: 20.0, // Lower threshold - crypto trends hard
            adx_ranging_threshold: 15.0,
            bb_period: 20,
            bb_std_dev: 2.0,
            bb_width_volatility_threshold: 70.0,
            ema_short_period: 21, // Faster for crypto
            ema_long_period: 50,
            atr_period: 14,
            atr_expansion_threshold: 1.3, // Crypto is naturally volatile
            regime_stability_bars: 2,
            min_regime_duration: 3,
        }
    }

    /// Conservative config - requires stronger signals
    pub fn conservative() -> Self {
        Self {
            adx_period: 14,
            adx_trending_threshold: 30.0,
            adx_ranging_threshold: 18.0,
            bb_period: 20,
            bb_std_dev: 2.0,
            bb_width_volatility_threshold: 80.0,
            ema_short_period: 50,
            ema_long_period: 200,
            atr_period: 14,
            atr_expansion_threshold: 2.0,
            regime_stability_bars: 5,
            min_regime_duration: 10,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_regime_display() {
        assert_eq!(
            format!("{}", MarketRegime::Trending(TrendDirection::Bullish)),
            "Trending (Bullish)"
        );
        assert_eq!(
            format!("{}", MarketRegime::Trending(TrendDirection::Bearish)),
            "Trending (Bearish)"
        );
        assert_eq!(format!("{}", MarketRegime::MeanReverting), "Mean-Reverting");
        assert_eq!(format!("{}", MarketRegime::Volatile), "Volatile/Choppy");
        assert_eq!(format!("{}", MarketRegime::Uncertain), "Uncertain");
    }

    #[test]
    fn test_market_regime_tradeable() {
        assert!(MarketRegime::Trending(TrendDirection::Bullish).is_tradeable());
        assert!(MarketRegime::Trending(TrendDirection::Bearish).is_tradeable());
        assert!(MarketRegime::MeanReverting.is_tradeable());
        assert!(!MarketRegime::Volatile.is_tradeable());
        assert!(!MarketRegime::Uncertain.is_tradeable());
    }

    #[test]
    fn test_size_multiplier() {
        assert_eq!(
            MarketRegime::Trending(TrendDirection::Bullish).size_multiplier(),
            1.0
        );
        assert_eq!(MarketRegime::MeanReverting.size_multiplier(), 0.8);
        assert_eq!(MarketRegime::Volatile.size_multiplier(), 0.3);
        assert_eq!(MarketRegime::Uncertain.size_multiplier(), 0.0);
    }

    #[test]
    fn test_recommended_strategy() {
        assert_eq!(
            MarketRegime::Trending(TrendDirection::Bullish).recommended_strategy(),
            RecommendedStrategy::TrendFollowing
        );
        assert_eq!(
            MarketRegime::MeanReverting.recommended_strategy(),
            RecommendedStrategy::MeanReversion
        );
        assert_eq!(
            MarketRegime::Volatile.recommended_strategy(),
            RecommendedStrategy::ReducedExposure
        );
        assert_eq!(
            MarketRegime::Uncertain.recommended_strategy(),
            RecommendedStrategy::StayCash
        );
    }

    #[test]
    fn test_regime_confidence_new() {
        let rc = RegimeConfidence::new(MarketRegime::Uncertain, 0.0);
        assert_eq!(rc.regime, MarketRegime::Uncertain);
        assert_eq!(rc.confidence, 0.0);
        assert!(!rc.is_actionable());
        assert!(!rc.is_strong());
    }

    #[test]
    fn test_regime_confidence_clamping() {
        let rc = RegimeConfidence::new(MarketRegime::Volatile, 1.5);
        assert_eq!(rc.confidence, 1.0);

        let rc2 = RegimeConfidence::new(MarketRegime::Volatile, -0.5);
        assert_eq!(rc2.confidence, 0.0);
    }

    #[test]
    fn test_regime_confidence_with_metrics() {
        let rc = RegimeConfidence::with_metrics(
            MarketRegime::Trending(TrendDirection::Bullish),
            0.85,
            32.5,
            60.0,
            0.72,
        );
        assert_eq!(rc.regime, MarketRegime::Trending(TrendDirection::Bullish));
        assert_eq!(rc.confidence, 0.85);
        assert_eq!(rc.adx_value, 32.5);
        assert_eq!(rc.bb_width_percentile, 60.0);
        assert_eq!(rc.trend_strength, 0.72);
        assert!(rc.is_actionable());
        assert!(rc.is_strong());
    }

    #[test]
    fn test_regime_confidence_actionable_threshold() {
        let below = RegimeConfidence::new(MarketRegime::MeanReverting, 0.59);
        assert!(!below.is_actionable());

        let at = RegimeConfidence::new(MarketRegime::MeanReverting, 0.6);
        assert!(at.is_actionable());

        let above = RegimeConfidence::new(MarketRegime::MeanReverting, 0.8);
        assert!(above.is_actionable());
    }

    #[test]
    fn test_trend_direction_display() {
        assert_eq!(format!("{}", TrendDirection::Bullish), "Bullish");
        assert_eq!(format!("{}", TrendDirection::Bearish), "Bearish");
    }

    #[test]
    fn test_regime_config_defaults() {
        let config = RegimeConfig::default();
        assert_eq!(config.adx_period, 14);
        assert_eq!(config.adx_trending_threshold, 25.0);
        assert_eq!(config.ema_short_period, 50);
        assert_eq!(config.ema_long_period, 200);
    }

    #[test]
    fn test_regime_config_crypto() {
        let config = RegimeConfig::crypto_optimized();
        assert_eq!(config.adx_trending_threshold, 20.0); // Lower for crypto
        assert_eq!(config.ema_short_period, 21); // Faster for crypto
        assert_eq!(config.ema_long_period, 50);
        assert_eq!(config.min_regime_duration, 3);
    }

    #[test]
    fn test_regime_config_conservative() {
        let config = RegimeConfig::conservative();
        assert_eq!(config.adx_trending_threshold, 30.0); // Higher for conservative
        assert_eq!(config.min_regime_duration, 10); // Longer hold
    }

    #[test]
    fn test_recommended_strategy_display() {
        assert_eq!(
            format!("{}", RecommendedStrategy::TrendFollowing),
            "Trend Following"
        );
        assert_eq!(
            format!("{}", RecommendedStrategy::MeanReversion),
            "Mean Reversion"
        );
        assert_eq!(
            format!("{}", RecommendedStrategy::ReducedExposure),
            "Reduced Exposure"
        );
        assert_eq!(format!("{}", RecommendedStrategy::StayCash), "Stay Cash");
    }

    #[test]
    fn test_regime_confidence_display() {
        let rc = RegimeConfidence::with_metrics(
            MarketRegime::Trending(TrendDirection::Bullish),
            0.85,
            32.5,
            60.0,
            0.72,
        );
        let display = format!("{}", rc);
        assert!(display.contains("Trending (Bullish)"));
        assert!(display.contains("85%"));
    }

    #[test]
    fn test_default_regime() {
        let regime = MarketRegime::default();
        assert_eq!(regime, MarketRegime::Uncertain);
    }

    #[test]
    fn test_default_regime_confidence() {
        let rc = RegimeConfidence::default();
        assert_eq!(rc.regime, MarketRegime::Uncertain);
        assert_eq!(rc.confidence, 0.0);
    }

    #[test]
    fn test_regime_equality() {
        assert_eq!(
            MarketRegime::Trending(TrendDirection::Bullish),
            MarketRegime::Trending(TrendDirection::Bullish)
        );
        assert_ne!(
            MarketRegime::Trending(TrendDirection::Bullish),
            MarketRegime::Trending(TrendDirection::Bearish)
        );
        assert_ne!(
            MarketRegime::Trending(TrendDirection::Bullish),
            MarketRegime::MeanReverting
        );
    }

    #[test]
    fn test_regime_serialization() {
        let regime = MarketRegime::Trending(TrendDirection::Bullish);
        let json = serde_json::to_string(&regime).unwrap();
        let deserialized: MarketRegime = serde_json::from_str(&json).unwrap();
        assert_eq!(regime, deserialized);
    }

    #[test]
    fn test_regime_confidence_serialization() {
        let rc = RegimeConfidence::with_metrics(MarketRegime::MeanReverting, 0.72, 18.5, 45.0, 0.3);
        let json = serde_json::to_string(&rc).unwrap();
        let deserialized: RegimeConfidence = serde_json::from_str(&json).unwrap();
        assert_eq!(rc.regime, deserialized.regime);
        assert!((rc.confidence - deserialized.confidence).abs() < f64::EPSILON);
    }

    #[test]
    fn test_regime_config_serialization() {
        let config = RegimeConfig::crypto_optimized();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RegimeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.adx_period, deserialized.adx_period);
        assert_eq!(
            config.adx_trending_threshold,
            deserialized.adx_trending_threshold
        );
    }
}
