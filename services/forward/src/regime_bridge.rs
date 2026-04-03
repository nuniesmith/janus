//! # Regime Bridge — Janus ↔ Neuromorphic Type Conversions
//!
//! This module bridges the regime detection system (`janus-regime`) with the
//! neuromorphic brain components (`janus-neuromorphic`), specifically:
//!
//! - **Hypothalamus** — Position sizing via `regime_scaling::MarketRegime` and
//!   `RegimeIndicators`
//! - **Amygdala** — Threat detection via `regime_shift::MarketRegime`
//!
//! The three systems use different `MarketRegime` enums because they serve
//! different purposes:
//!
//! | System | Enum | Purpose |
//! |--------|------|---------|
//! | `janus-regime` | 4 variants (Trending, MeanReverting, Volatile, Uncertain) | Strategy routing |
//! | `hypothalamus` | 10 variants (StrongBullish…Crisis) | Position sizing |
//! | `amygdala` | 7 variants (LowVolTrending…Crisis) | Threat detection |
//!
//! This bridge provides zero-cost conversions between them, using confidence
//! levels and indicator values to map the coarser janus-regime classification
//! into the finer-grained neuromorphic enums.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::regime_bridge::{
//!     to_hypothalamus_regime,
//!     to_amygdala_regime,
//!     build_regime_indicators,
//! };
//!
//! let routed = regime_mgr.on_tick("BTCUSDT", bid, ask).unwrap();
//!
//! // For position sizing (hypothalamus)
//! let hypo_regime = to_hypothalamus_regime(routed.regime, routed.confidence);
//! let indicators = build_regime_indicators(&routed, adx, bb_width, atr);
//! let scale = regime_scaling.update(&indicators);
//!
//! // For threat detection (amygdala)
//! let amyg_regime = to_amygdala_regime(routed.regime, routed.confidence);
//! ```

use janus_regime::{MarketRegime, RoutedSignal, TrendDirection};

// ============================================================================
// Hypothalamus Bridge (Position Sizing)
// ============================================================================

/// Market regime variants used by the hypothalamus position sizing system.
///
/// This is a mirror of `janus_neuromorphic::hypothalamus::position_sizing::
/// regime_scaling::MarketRegime`. We re-define it here to avoid coupling
/// the forward service's regime bridge to the neuromorphic crate at compile
/// time — the neuromorphic crate is large and pulls in CUDA/ML deps.
///
/// When wiring into the actual neuromorphic system, use the `From` impl
/// or call [`to_hypothalamus_regime`] and transmit the result via IPC/channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HypothalamusRegime {
    StrongBullish,
    Bullish,
    Neutral,
    Bearish,
    StrongBearish,
    HighVolatility,
    LowVolatility,
    Transitional,
    Crisis,
    #[default]
    Unknown,
}

impl std::fmt::Display for HypothalamusRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StrongBullish => write!(f, "StrongBullish"),
            Self::Bullish => write!(f, "Bullish"),
            Self::Neutral => write!(f, "Neutral"),
            Self::Bearish => write!(f, "Bearish"),
            Self::StrongBearish => write!(f, "StrongBearish"),
            Self::HighVolatility => write!(f, "HighVolatility"),
            Self::LowVolatility => write!(f, "LowVolatility"),
            Self::Transitional => write!(f, "Transitional"),
            Self::Crisis => write!(f, "Crisis"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl HypothalamusRegime {
    /// Get the base position scaling factor for this regime.
    ///
    /// Mirrors `hypothalamus::regime_scaling::MarketRegime::base_scaling()`.
    pub fn base_scaling(&self) -> f64 {
        match self {
            Self::StrongBullish => 1.25,
            Self::Bullish => 1.10,
            Self::Neutral => 1.00,
            Self::Bearish => 0.75,
            Self::StrongBearish => 0.50,
            Self::HighVolatility => 0.60,
            Self::LowVolatility => 0.90,
            Self::Transitional => 0.70,
            Self::Crisis => 0.20,
            Self::Unknown => 0.80,
        }
    }

    /// Check if this regime favors trend-following strategies.
    pub fn favors_trend_following(&self) -> bool {
        matches!(
            self,
            Self::StrongBullish | Self::Bullish | Self::StrongBearish | Self::Bearish
        )
    }

    /// Check if this regime favors mean-reversion strategies.
    pub fn favors_mean_reversion(&self) -> bool {
        matches!(self, Self::Neutral | Self::LowVolatility)
    }

    /// Check if this regime suggests caution.
    pub fn requires_caution(&self) -> bool {
        matches!(
            self,
            Self::HighVolatility | Self::Transitional | Self::Crisis | Self::Unknown
        )
    }

    /// Convert to an integer for Prometheus gauge encoding.
    ///
    /// Mapping:
    /// 0=Unknown, 1=StrongBullish, 2=Bullish, 3=Neutral, 4=Bearish,
    /// 5=StrongBearish, 6=HighVolatility, 7=LowVolatility, 8=Transitional, 9=Crisis
    pub fn to_prometheus_i64(self) -> i64 {
        match self {
            Self::Unknown => 0,
            Self::StrongBullish => 1,
            Self::Bullish => 2,
            Self::Neutral => 3,
            Self::Bearish => 4,
            Self::StrongBearish => 5,
            Self::HighVolatility => 6,
            Self::LowVolatility => 7,
            Self::Transitional => 8,
            Self::Crisis => 9,
        }
    }
}

/// Convert a `janus-regime` `MarketRegime` + confidence into the
/// hypothalamus position sizing regime.
///
/// The mapping uses confidence to distinguish between strong and moderate
/// variants:
///
/// | janus-regime | Confidence | Hypothalamus |
/// |-------------|------------|--------------|
/// | Trending(Bullish) | ≥ 0.7 | StrongBullish |
/// | Trending(Bullish) | < 0.7 | Bullish |
/// | Trending(Bearish) | ≥ 0.7 | StrongBearish |
/// | Trending(Bearish) | < 0.7 | Bearish |
/// | MeanReverting | any | Neutral |
/// | Volatile | ≥ 0.8 | Crisis |
/// | Volatile | < 0.8 | HighVolatility |
/// | Uncertain | ≤ 0.3 | Unknown |
/// | Uncertain | > 0.3 | Transitional |
pub fn to_hypothalamus_regime(regime: MarketRegime, confidence: f64) -> HypothalamusRegime {
    match regime {
        MarketRegime::Trending(TrendDirection::Bullish) => {
            if confidence >= 0.7 {
                HypothalamusRegime::StrongBullish
            } else {
                HypothalamusRegime::Bullish
            }
        }
        MarketRegime::Trending(TrendDirection::Bearish) => {
            if confidence >= 0.7 {
                HypothalamusRegime::StrongBearish
            } else {
                HypothalamusRegime::Bearish
            }
        }
        MarketRegime::MeanReverting => HypothalamusRegime::Neutral,
        MarketRegime::Volatile => {
            if confidence >= 0.8 {
                HypothalamusRegime::Crisis
            } else {
                HypothalamusRegime::HighVolatility
            }
        }
        MarketRegime::Uncertain => {
            if confidence <= 0.3 {
                HypothalamusRegime::Unknown
            } else {
                HypothalamusRegime::Transitional
            }
        }
    }
}

// ============================================================================
// Amygdala Bridge (Threat Detection)
// ============================================================================

/// Market regime variants used by the amygdala threat detection system.
///
/// Mirrors `janus_neuromorphic::amygdala::threat_detection::regime_shift::MarketRegime`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AmygdalaRegime {
    LowVolTrending,
    LowVolMeanReverting,
    HighVolTrending,
    HighVolMeanReverting,
    Crisis,
    Transitional,
    #[default]
    Unknown,
}

impl std::fmt::Display for AmygdalaRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LowVolTrending => write!(f, "LowVolTrending"),
            Self::LowVolMeanReverting => write!(f, "LowVolMeanReverting"),
            Self::HighVolTrending => write!(f, "HighVolTrending"),
            Self::HighVolMeanReverting => write!(f, "HighVolMeanReverting"),
            Self::Crisis => write!(f, "Crisis"),
            Self::Transitional => write!(f, "Transitional"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl AmygdalaRegime {
    /// Returns true if this is a high-risk regime that should trigger
    /// defensive measures (circuit breakers, position reduction, etc.).
    pub fn is_high_risk(&self) -> bool {
        matches!(
            self,
            Self::Crisis | Self::HighVolTrending | Self::Transitional
        )
    }

    /// Convert to an integer for Prometheus gauge encoding.
    ///
    /// Mapping:
    /// 0=Unknown, 1=LowVolTrending, 2=LowVolMeanReverting, 3=HighVolTrending,
    /// 4=HighVolMeanReverting, 5=Transitional, 6=Crisis
    pub fn to_prometheus_i64(self) -> i64 {
        match self {
            Self::Unknown => 0,
            Self::LowVolTrending => 1,
            Self::LowVolMeanReverting => 2,
            Self::HighVolTrending => 3,
            Self::HighVolMeanReverting => 4,
            Self::Transitional => 5,
            Self::Crisis => 6,
        }
    }
}

/// Convert a `janus-regime` `MarketRegime` + confidence into the
/// amygdala threat detection regime.
///
/// The amygdala uses a 2D classification (volatility × trend/reversion),
/// so we map:
///
/// | janus-regime | Confidence | Amygdala |
/// |-------------|------------|---------|
/// | Trending(_) | ≥ 0.5 | LowVolTrending |
/// | MeanReverting | ≥ 0.5 | LowVolMeanReverting |
/// | Volatile + Trending signals | any | HighVolTrending |
/// | Volatile + no trend | any | HighVolMeanReverting |
/// | Volatile | ≥ 0.8 | Crisis |
/// | Uncertain | any | Transitional or Unknown |
///
/// The `is_volatile` parameter can be derived from the router's state
/// probabilities or ATR expansion. When not available, we infer from
/// the regime + confidence.
pub fn to_amygdala_regime(regime: MarketRegime, confidence: f64) -> AmygdalaRegime {
    match regime {
        MarketRegime::Trending(_) => {
            // Trending with high confidence = established, low-vol trend
            // Trending with low confidence = possibly volatile trending
            if confidence >= 0.5 {
                AmygdalaRegime::LowVolTrending
            } else {
                AmygdalaRegime::HighVolTrending
            }
        }
        MarketRegime::MeanReverting => {
            if confidence >= 0.5 {
                AmygdalaRegime::LowVolMeanReverting
            } else {
                AmygdalaRegime::HighVolMeanReverting
            }
        }
        MarketRegime::Volatile => {
            if confidence >= 0.8 {
                AmygdalaRegime::Crisis
            } else {
                // Volatile without clear direction defaults to high-vol mean-reverting
                AmygdalaRegime::HighVolMeanReverting
            }
        }
        MarketRegime::Uncertain => {
            if confidence <= 0.3 {
                AmygdalaRegime::Unknown
            } else {
                AmygdalaRegime::Transitional
            }
        }
    }
}

/// Extended amygdala conversion that uses volatility information from the
/// router signal for more accurate classification.
pub fn to_amygdala_regime_with_vol(
    regime: MarketRegime,
    confidence: f64,
    is_volatile: bool,
) -> AmygdalaRegime {
    match regime {
        MarketRegime::Trending(_) => {
            if is_volatile {
                AmygdalaRegime::HighVolTrending
            } else {
                AmygdalaRegime::LowVolTrending
            }
        }
        MarketRegime::MeanReverting => {
            if is_volatile {
                AmygdalaRegime::HighVolMeanReverting
            } else {
                AmygdalaRegime::LowVolMeanReverting
            }
        }
        MarketRegime::Volatile => {
            if confidence >= 0.8 {
                AmygdalaRegime::Crisis
            } else {
                AmygdalaRegime::HighVolMeanReverting
            }
        }
        MarketRegime::Uncertain => {
            if confidence <= 0.3 {
                AmygdalaRegime::Unknown
            } else {
                AmygdalaRegime::Transitional
            }
        }
    }
}

// ============================================================================
// Regime Indicators Bridge
// ============================================================================

/// Indicators snapshot for the hypothalamus position sizing system.
///
/// Mirrors `janus_neuromorphic::hypothalamus::position_sizing::
/// regime_scaling::RegimeIndicators`.
#[derive(Debug, Clone)]
pub struct RegimeIndicators {
    /// Price trend (positive = bullish, negative = bearish)
    pub trend: f64,
    /// Trend strength (0.0–1.0)
    pub trend_strength: f64,
    /// Volatility level (annualized or normalized)
    pub volatility: f64,
    /// Volatility percentile (0.0–1.0)
    pub volatility_percentile: f64,
    /// Correlation with benchmark (not used in current regime system)
    pub correlation: f64,
    /// Market breadth (not used in current regime system)
    pub breadth: f64,
    /// Momentum (rate of change)
    pub momentum: f64,
    /// Volume relative to average
    pub relative_volume: f64,
    /// Spread/liquidity indicator
    pub liquidity_score: f64,
    /// VIX or fear index level
    pub fear_index: Option<f64>,
}

impl Default for RegimeIndicators {
    fn default() -> Self {
        Self {
            trend: 0.0,
            trend_strength: 0.0,
            volatility: 0.15,
            volatility_percentile: 0.5,
            correlation: 0.0,
            breadth: 1.0,
            momentum: 0.0,
            relative_volume: 1.0,
            liquidity_score: 1.0,
            fear_index: None,
        }
    }
}

/// Build a `RegimeIndicators` snapshot from a `RoutedSignal` and optional
/// raw indicator values.
///
/// # Arguments
///
/// * `signal` — The `RoutedSignal` from the regime router
/// * `adx_value` — Current ADX value (trend strength indicator)
/// * `bb_width_percentile` — Bollinger Band width as a percentile (0.0–1.0)
/// * `atr_value` — Current ATR value (volatility magnitude)
/// * `relative_volume` — Current volume relative to average (1.0 = average)
pub fn build_regime_indicators(
    signal: &RoutedSignal,
    adx_value: Option<f64>,
    bb_width_percentile: Option<f64>,
    atr_value: Option<f64>,
    relative_volume: Option<f64>,
) -> RegimeIndicators {
    // Derive trend direction from the regime signal
    let trend = match signal.regime {
        MarketRegime::Trending(TrendDirection::Bullish) => signal.confidence,
        MarketRegime::Trending(TrendDirection::Bearish) => -signal.confidence,
        _ => 0.0,
    };

    // Derive trend strength from ADX or confidence
    let trend_strength = adx_value
        .map(|adx| (adx / 50.0).min(1.0)) // Normalize ADX (50+ = max strength)
        .unwrap_or_else(|| {
            if matches!(signal.regime, MarketRegime::Trending(_)) {
                signal.confidence
            } else {
                0.0
            }
        });

    // Derive volatility from BB width percentile or ATR
    let volatility_percentile = bb_width_percentile.unwrap_or(0.5);
    let volatility = atr_value.unwrap_or(0.15);

    // Derive momentum from trend direction and confidence
    let momentum = match signal.regime {
        MarketRegime::Trending(TrendDirection::Bullish) => signal.confidence * 0.5,
        MarketRegime::Trending(TrendDirection::Bearish) => -signal.confidence * 0.5,
        MarketRegime::Volatile => 0.0,
        _ => 0.0,
    };

    RegimeIndicators {
        trend,
        trend_strength,
        volatility,
        volatility_percentile,
        correlation: 0.0, // Not available from current regime system
        breadth: 1.0,     // Not available (single-asset)
        momentum,
        relative_volume: relative_volume.unwrap_or(1.0),
        liquidity_score: 1.0, // Not available from current regime system
        fear_index: None,     // Could be derived from Volatile+high confidence
    }
}

// ============================================================================
// Composite Bridge Result
// ============================================================================

/// A complete bridge result containing all neuromorphic-compatible types
/// derived from a single `RoutedSignal`.
///
/// This is the recommended type to pass through channels or IPC to the
/// neuromorphic system.
#[derive(Debug, Clone)]
pub struct BridgedRegimeState {
    /// Symbol this state applies to
    pub symbol: String,
    /// Hypothalamus regime for position sizing
    pub hypothalamus_regime: HypothalamusRegime,
    /// Amygdala regime for threat detection
    pub amygdala_regime: AmygdalaRegime,
    /// Position scaling factor from hypothalamus regime
    pub position_scale: f64,
    /// Whether the amygdala considers this high-risk
    pub is_high_risk: bool,
    /// Original confidence from the regime detector
    pub confidence: f64,
    /// Regime indicators for hypothalamus consumption
    pub indicators: RegimeIndicators,
}

impl std::fmt::Display for BridgedRegimeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BridgedRegime[{}: hypo={}, amyg={}, scale={:.2}, risk={}, conf={:.0}%]",
            self.symbol,
            self.hypothalamus_regime,
            self.amygdala_regime,
            self.position_scale,
            if self.is_high_risk { "HIGH" } else { "low" },
            self.confidence * 100.0,
        )
    }
}

/// Build a complete `BridgedRegimeState` from a `RoutedSignal`.
///
/// This is the primary entry point for the bridge — call this on every
/// regime signal and forward the result to the neuromorphic system.
pub fn bridge_regime_signal(
    symbol: &str,
    signal: &RoutedSignal,
    adx_value: Option<f64>,
    bb_width_percentile: Option<f64>,
    atr_value: Option<f64>,
    relative_volume: Option<f64>,
) -> BridgedRegimeState {
    let hypothalamus_regime = to_hypothalamus_regime(signal.regime, signal.confidence);
    let amygdala_regime = to_amygdala_regime(signal.regime, signal.confidence);
    let indicators = build_regime_indicators(
        signal,
        adx_value,
        bb_width_percentile,
        atr_value,
        relative_volume,
    );

    BridgedRegimeState {
        symbol: symbol.to_string(),
        hypothalamus_regime,
        amygdala_regime,
        position_scale: hypothalamus_regime.base_scaling(),
        is_high_risk: amygdala_regime.is_high_risk(),
        confidence: signal.confidence,
        indicators,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use janus_regime::{ActiveStrategy, DetectionMethod};

    fn make_signal(regime: MarketRegime, confidence: f64) -> RoutedSignal {
        RoutedSignal {
            strategy: ActiveStrategy::TrendFollowing,
            regime,
            confidence,
            position_factor: 1.0,
            reason: "test".to_string(),
            detection_method: DetectionMethod::Ensemble,
            methods_agree: Some(true),
            state_probabilities: None,
            expected_duration: None,
            trend_direction: None,
        }
    }

    // ========================================================================
    // Hypothalamus regime conversion tests
    // ========================================================================

    #[test]
    fn test_trending_bullish_high_confidence_is_strong() {
        let regime = to_hypothalamus_regime(MarketRegime::Trending(TrendDirection::Bullish), 0.85);
        assert_eq!(regime, HypothalamusRegime::StrongBullish);
    }

    #[test]
    fn test_trending_bullish_low_confidence_is_moderate() {
        let regime = to_hypothalamus_regime(MarketRegime::Trending(TrendDirection::Bullish), 0.55);
        assert_eq!(regime, HypothalamusRegime::Bullish);
    }

    #[test]
    fn test_trending_bearish_high_confidence_is_strong() {
        let regime = to_hypothalamus_regime(MarketRegime::Trending(TrendDirection::Bearish), 0.75);
        assert_eq!(regime, HypothalamusRegime::StrongBearish);
    }

    #[test]
    fn test_trending_bearish_low_confidence_is_moderate() {
        let regime = to_hypothalamus_regime(MarketRegime::Trending(TrendDirection::Bearish), 0.5);
        assert_eq!(regime, HypothalamusRegime::Bearish);
    }

    #[test]
    fn test_mean_reverting_maps_to_neutral() {
        let regime = to_hypothalamus_regime(MarketRegime::MeanReverting, 0.8);
        assert_eq!(regime, HypothalamusRegime::Neutral);
    }

    #[test]
    fn test_volatile_high_confidence_maps_to_crisis() {
        let regime = to_hypothalamus_regime(MarketRegime::Volatile, 0.85);
        assert_eq!(regime, HypothalamusRegime::Crisis);
    }

    #[test]
    fn test_volatile_moderate_confidence_maps_to_high_vol() {
        let regime = to_hypothalamus_regime(MarketRegime::Volatile, 0.6);
        assert_eq!(regime, HypothalamusRegime::HighVolatility);
    }

    #[test]
    fn test_uncertain_low_confidence_maps_to_unknown() {
        let regime = to_hypothalamus_regime(MarketRegime::Uncertain, 0.2);
        assert_eq!(regime, HypothalamusRegime::Unknown);
    }

    #[test]
    fn test_uncertain_moderate_confidence_maps_to_transitional() {
        let regime = to_hypothalamus_regime(MarketRegime::Uncertain, 0.5);
        assert_eq!(regime, HypothalamusRegime::Transitional);
    }

    #[test]
    fn test_hypothalamus_boundary_at_0_7() {
        // Exactly 0.7 should be Strong
        let strong = to_hypothalamus_regime(MarketRegime::Trending(TrendDirection::Bullish), 0.7);
        assert_eq!(strong, HypothalamusRegime::StrongBullish);

        // Just below 0.7 should be moderate
        let moderate =
            to_hypothalamus_regime(MarketRegime::Trending(TrendDirection::Bullish), 0.699);
        assert_eq!(moderate, HypothalamusRegime::Bullish);
    }

    // ========================================================================
    // Amygdala regime conversion tests
    // ========================================================================

    #[test]
    fn test_trending_high_confidence_is_low_vol() {
        let regime = to_amygdala_regime(MarketRegime::Trending(TrendDirection::Bullish), 0.7);
        assert_eq!(regime, AmygdalaRegime::LowVolTrending);
    }

    #[test]
    fn test_trending_low_confidence_is_high_vol() {
        let regime = to_amygdala_regime(MarketRegime::Trending(TrendDirection::Bearish), 0.3);
        assert_eq!(regime, AmygdalaRegime::HighVolTrending);
    }

    #[test]
    fn test_mean_reverting_high_confidence_is_low_vol() {
        let regime = to_amygdala_regime(MarketRegime::MeanReverting, 0.8);
        assert_eq!(regime, AmygdalaRegime::LowVolMeanReverting);
    }

    #[test]
    fn test_mean_reverting_low_confidence_is_high_vol() {
        let regime = to_amygdala_regime(MarketRegime::MeanReverting, 0.4);
        assert_eq!(regime, AmygdalaRegime::HighVolMeanReverting);
    }

    #[test]
    fn test_volatile_extreme_confidence_is_crisis() {
        let regime = to_amygdala_regime(MarketRegime::Volatile, 0.9);
        assert_eq!(regime, AmygdalaRegime::Crisis);
    }

    #[test]
    fn test_volatile_moderate_is_high_vol_mr() {
        let regime = to_amygdala_regime(MarketRegime::Volatile, 0.6);
        assert_eq!(regime, AmygdalaRegime::HighVolMeanReverting);
    }

    #[test]
    fn test_uncertain_low_confidence_is_unknown() {
        let regime = to_amygdala_regime(MarketRegime::Uncertain, 0.1);
        assert_eq!(regime, AmygdalaRegime::Unknown);
    }

    #[test]
    fn test_uncertain_moderate_is_transitional() {
        let regime = to_amygdala_regime(MarketRegime::Uncertain, 0.5);
        assert_eq!(regime, AmygdalaRegime::Transitional);
    }

    // ========================================================================
    // Extended amygdala conversion with explicit volatility flag
    // ========================================================================

    #[test]
    fn test_with_vol_trending_volatile() {
        let regime =
            to_amygdala_regime_with_vol(MarketRegime::Trending(TrendDirection::Bullish), 0.8, true);
        assert_eq!(regime, AmygdalaRegime::HighVolTrending);
    }

    #[test]
    fn test_with_vol_trending_calm() {
        let regime = to_amygdala_regime_with_vol(
            MarketRegime::Trending(TrendDirection::Bullish),
            0.8,
            false,
        );
        assert_eq!(regime, AmygdalaRegime::LowVolTrending);
    }

    #[test]
    fn test_with_vol_mean_reverting_volatile() {
        let regime = to_amygdala_regime_with_vol(MarketRegime::MeanReverting, 0.8, true);
        assert_eq!(regime, AmygdalaRegime::HighVolMeanReverting);
    }

    #[test]
    fn test_with_vol_mean_reverting_calm() {
        let regime = to_amygdala_regime_with_vol(MarketRegime::MeanReverting, 0.8, false);
        assert_eq!(regime, AmygdalaRegime::LowVolMeanReverting);
    }

    // ========================================================================
    // High-risk classification tests
    // ========================================================================

    #[test]
    fn test_crisis_is_high_risk() {
        assert!(AmygdalaRegime::Crisis.is_high_risk());
    }

    #[test]
    fn test_high_vol_trending_is_high_risk() {
        assert!(AmygdalaRegime::HighVolTrending.is_high_risk());
    }

    #[test]
    fn test_transitional_is_high_risk() {
        assert!(AmygdalaRegime::Transitional.is_high_risk());
    }

    #[test]
    fn test_low_vol_trending_is_not_high_risk() {
        assert!(!AmygdalaRegime::LowVolTrending.is_high_risk());
    }

    #[test]
    fn test_low_vol_mean_reverting_is_not_high_risk() {
        assert!(!AmygdalaRegime::LowVolMeanReverting.is_high_risk());
    }

    // ========================================================================
    // Position scaling tests
    // ========================================================================

    #[test]
    fn test_strong_bullish_scales_up() {
        assert!(HypothalamusRegime::StrongBullish.base_scaling() > 1.0);
    }

    #[test]
    fn test_crisis_scales_down_aggressively() {
        assert!(HypothalamusRegime::Crisis.base_scaling() < 0.3);
    }

    #[test]
    fn test_neutral_is_baseline() {
        assert!((HypothalamusRegime::Neutral.base_scaling() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_favors_trend_following() {
        assert!(HypothalamusRegime::StrongBullish.favors_trend_following());
        assert!(HypothalamusRegime::Bearish.favors_trend_following());
        assert!(!HypothalamusRegime::Neutral.favors_trend_following());
        assert!(!HypothalamusRegime::HighVolatility.favors_trend_following());
    }

    #[test]
    fn test_favors_mean_reversion() {
        assert!(HypothalamusRegime::Neutral.favors_mean_reversion());
        assert!(HypothalamusRegime::LowVolatility.favors_mean_reversion());
        assert!(!HypothalamusRegime::StrongBullish.favors_mean_reversion());
    }

    #[test]
    fn test_requires_caution() {
        assert!(HypothalamusRegime::HighVolatility.requires_caution());
        assert!(HypothalamusRegime::Crisis.requires_caution());
        assert!(HypothalamusRegime::Unknown.requires_caution());
        assert!(!HypothalamusRegime::Bullish.requires_caution());
    }

    // ========================================================================
    // Regime indicators tests
    // ========================================================================

    #[test]
    fn test_build_indicators_trending_bullish() {
        let signal = make_signal(MarketRegime::Trending(TrendDirection::Bullish), 0.8);
        let indicators =
            build_regime_indicators(&signal, Some(35.0), Some(0.4), Some(0.02), Some(1.5));

        assert!(indicators.trend > 0.0, "Bullish trend should be positive");
        assert!(
            indicators.trend_strength > 0.5,
            "ADX 35 should give decent strength"
        );
        assert!(
            indicators.momentum > 0.0,
            "Bullish momentum should be positive"
        );
        assert!((indicators.relative_volume - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_indicators_trending_bearish() {
        let signal = make_signal(MarketRegime::Trending(TrendDirection::Bearish), 0.7);
        let indicators = build_regime_indicators(&signal, None, None, None, None);

        assert!(indicators.trend < 0.0, "Bearish trend should be negative");
        assert!(
            indicators.momentum < 0.0,
            "Bearish momentum should be negative"
        );
    }

    #[test]
    fn test_build_indicators_mean_reverting() {
        let signal = make_signal(MarketRegime::MeanReverting, 0.6);
        let indicators = build_regime_indicators(&signal, Some(15.0), Some(0.3), None, None);

        assert!(
            (indicators.trend - 0.0).abs() < f64::EPSILON,
            "MR should have zero trend"
        );
        assert!(
            indicators.trend_strength < 0.5,
            "Low ADX = low trend strength"
        );
    }

    #[test]
    fn test_build_indicators_defaults_without_raw_values() {
        let signal = make_signal(MarketRegime::Uncertain, 0.3);
        let indicators = build_regime_indicators(&signal, None, None, None, None);

        // Should use defaults
        assert!((indicators.volatility_percentile - 0.5).abs() < f64::EPSILON);
        assert!((indicators.relative_volume - 1.0).abs() < f64::EPSILON);
        assert!((indicators.volatility - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adx_normalization_capped_at_1() {
        let signal = make_signal(MarketRegime::Trending(TrendDirection::Bullish), 0.9);
        // ADX of 60 should normalize to 1.0 (capped at 50/50)
        let indicators = build_regime_indicators(&signal, Some(60.0), None, None, None);
        assert!((indicators.trend_strength - 1.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // Composite bridge tests
    // ========================================================================

    #[test]
    fn test_bridge_regime_signal_strong_bull() {
        let signal = make_signal(MarketRegime::Trending(TrendDirection::Bullish), 0.85);
        let bridged = bridge_regime_signal(
            "BTCUSDT",
            &signal,
            Some(40.0),
            Some(0.4),
            Some(0.03),
            Some(1.2),
        );

        assert_eq!(bridged.symbol, "BTCUSDT");
        assert_eq!(
            bridged.hypothalamus_regime,
            HypothalamusRegime::StrongBullish
        );
        assert_eq!(bridged.amygdala_regime, AmygdalaRegime::LowVolTrending);
        assert!(bridged.position_scale > 1.0);
        assert!(!bridged.is_high_risk);
        assert!((bridged.confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bridge_regime_signal_crisis() {
        let signal = make_signal(MarketRegime::Volatile, 0.9);
        let bridged = bridge_regime_signal("ETHUSDT", &signal, None, None, None, None);

        assert_eq!(bridged.hypothalamus_regime, HypothalamusRegime::Crisis);
        assert_eq!(bridged.amygdala_regime, AmygdalaRegime::Crisis);
        assert!(bridged.position_scale < 0.3);
        assert!(bridged.is_high_risk);
    }

    #[test]
    fn test_bridge_regime_signal_uncertain() {
        let signal = make_signal(MarketRegime::Uncertain, 0.2);
        let bridged = bridge_regime_signal("SOLUSDT", &signal, None, None, None, None);

        assert_eq!(bridged.hypothalamus_regime, HypothalamusRegime::Unknown);
        assert_eq!(bridged.amygdala_regime, AmygdalaRegime::Unknown);
        assert!(!bridged.is_high_risk);
    }

    // ========================================================================
    // Display tests
    // ========================================================================

    #[test]
    fn test_hypothalamus_regime_display() {
        assert_eq!(
            HypothalamusRegime::StrongBullish.to_string(),
            "StrongBullish"
        );
        assert_eq!(HypothalamusRegime::Crisis.to_string(), "Crisis");
        assert_eq!(HypothalamusRegime::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn test_amygdala_regime_display() {
        assert_eq!(AmygdalaRegime::LowVolTrending.to_string(), "LowVolTrending");
        assert_eq!(AmygdalaRegime::Crisis.to_string(), "Crisis");
        assert_eq!(AmygdalaRegime::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn test_bridged_state_display() {
        let signal = make_signal(MarketRegime::Trending(TrendDirection::Bullish), 0.75);
        let bridged = bridge_regime_signal("BTCUSDT", &signal, None, None, None, None);
        let display = bridged.to_string();
        assert!(display.contains("BTCUSDT"));
        assert!(display.contains("StrongBullish"));
        assert!(display.contains("75%"));
    }

    // ========================================================================
    // Default tests
    // ========================================================================

    #[test]
    fn test_hypothalamus_default() {
        assert_eq!(HypothalamusRegime::default(), HypothalamusRegime::Unknown);
    }

    #[test]
    fn test_amygdala_default() {
        assert_eq!(AmygdalaRegime::default(), AmygdalaRegime::Unknown);
    }

    #[test]
    fn test_regime_indicators_default() {
        let ind = RegimeIndicators::default();
        assert!((ind.trend - 0.0).abs() < f64::EPSILON);
        assert!((ind.volatility_percentile - 0.5).abs() < f64::EPSILON);
        assert!((ind.relative_volume - 1.0).abs() < f64::EPSILON);
        assert!(ind.fear_index.is_none());
    }

    // ========================================================================
    // Prometheus integer encoding tests
    // ========================================================================

    #[test]
    fn test_hypothalamus_to_prometheus_i64_all_variants() {
        assert_eq!(HypothalamusRegime::Unknown.to_prometheus_i64(), 0);
        assert_eq!(HypothalamusRegime::StrongBullish.to_prometheus_i64(), 1);
        assert_eq!(HypothalamusRegime::Bullish.to_prometheus_i64(), 2);
        assert_eq!(HypothalamusRegime::Neutral.to_prometheus_i64(), 3);
        assert_eq!(HypothalamusRegime::Bearish.to_prometheus_i64(), 4);
        assert_eq!(HypothalamusRegime::StrongBearish.to_prometheus_i64(), 5);
        assert_eq!(HypothalamusRegime::HighVolatility.to_prometheus_i64(), 6);
        assert_eq!(HypothalamusRegime::LowVolatility.to_prometheus_i64(), 7);
        assert_eq!(HypothalamusRegime::Transitional.to_prometheus_i64(), 8);
        assert_eq!(HypothalamusRegime::Crisis.to_prometheus_i64(), 9);
    }

    #[test]
    fn test_hypothalamus_to_prometheus_i64_unique_values() {
        let variants = [
            HypothalamusRegime::Unknown,
            HypothalamusRegime::StrongBullish,
            HypothalamusRegime::Bullish,
            HypothalamusRegime::Neutral,
            HypothalamusRegime::Bearish,
            HypothalamusRegime::StrongBearish,
            HypothalamusRegime::HighVolatility,
            HypothalamusRegime::LowVolatility,
            HypothalamusRegime::Transitional,
            HypothalamusRegime::Crisis,
        ];
        let values: Vec<i64> = variants.iter().map(|v| v.to_prometheus_i64()).collect();
        let unique: std::collections::HashSet<i64> = values.iter().cloned().collect();
        assert_eq!(
            values.len(),
            unique.len(),
            "All hypothalamus prometheus values must be unique"
        );
    }

    #[test]
    fn test_amygdala_to_prometheus_i64_all_variants() {
        assert_eq!(AmygdalaRegime::Unknown.to_prometheus_i64(), 0);
        assert_eq!(AmygdalaRegime::LowVolTrending.to_prometheus_i64(), 1);
        assert_eq!(AmygdalaRegime::LowVolMeanReverting.to_prometheus_i64(), 2);
        assert_eq!(AmygdalaRegime::HighVolTrending.to_prometheus_i64(), 3);
        assert_eq!(AmygdalaRegime::HighVolMeanReverting.to_prometheus_i64(), 4);
        assert_eq!(AmygdalaRegime::Transitional.to_prometheus_i64(), 5);
        assert_eq!(AmygdalaRegime::Crisis.to_prometheus_i64(), 6);
    }

    #[test]
    fn test_amygdala_to_prometheus_i64_unique_values() {
        let variants = [
            AmygdalaRegime::Unknown,
            AmygdalaRegime::LowVolTrending,
            AmygdalaRegime::LowVolMeanReverting,
            AmygdalaRegime::HighVolTrending,
            AmygdalaRegime::HighVolMeanReverting,
            AmygdalaRegime::Transitional,
            AmygdalaRegime::Crisis,
        ];
        let values: Vec<i64> = variants.iter().map(|v| v.to_prometheus_i64()).collect();
        let unique: std::collections::HashSet<i64> = values.iter().cloned().collect();
        assert_eq!(
            values.len(),
            unique.len(),
            "All amygdala prometheus values must be unique"
        );
    }

    #[test]
    fn test_hypothalamus_default_maps_to_zero() {
        assert_eq!(HypothalamusRegime::default().to_prometheus_i64(), 0);
    }

    #[test]
    fn test_amygdala_default_maps_to_zero() {
        assert_eq!(AmygdalaRegime::default().to_prometheus_i64(), 0);
    }
}
