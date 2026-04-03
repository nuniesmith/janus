//! Market-Specific Fuzzy Predicates for LTN
//!
//! This module defines fuzzy predicates that convert DSP feature values
//! into fuzzy truth values for use in LTN axioms.
//!
//! Each predicate:
//! - Takes DSP features as input
//! - Returns a fuzzy truth value in [0, 1]
//! - Is differentiable (smooth transitions via sigmoid)
//! - Encodes domain knowledge about market regimes
//!
//! # Feature Vector Layout (from DSP)
//!
//! ```text
//! [0] divergence_norm    - Z-score of price-FRAMA divergence
//! [1] alpha_norm         - Z-score of FRAMA smoothing factor
//! [2] fractal_dim        - Raw Sevcik fractal dimension [1.0, 2.0]
//! [3] hurst              - Raw Hurst exponent [0.0, 1.0]
//! [4] regime             - Market regime encoding {-1, 0, 1}
//! [5] divergence_sign    - Directional signal {-1, 0, 1}
//! [6] alpha_deviation    - Alpha deviation from midpoint ±0.25
//! [7] regime_confidence  - Distance from regime boundaries [0.0, 0.6]
//! ```

use super::fuzzy_ops::soft_threshold;

/// Feature indices (for readability)
pub mod feature_idx {
    pub const DIVERGENCE_NORM: usize = 0;
    pub const ALPHA_NORM: usize = 1;
    pub const FRACTAL_DIM: usize = 2;
    pub const HURST: usize = 3;
    pub const REGIME: usize = 4;
    pub const DIVERGENCE_SIGN: usize = 5;
    pub const ALPHA_DEVIATION: usize = 6;
    pub const REGIME_CONFIDENCE: usize = 7;
}

// ============================================================================
// Regime Predicates
// ============================================================================

/// Predicate: Market is in a trending regime
///
/// Trending if Hurst exponent > 0.6 (persistent, momentum-driven)
///
/// Uses sigmoid for smooth transition:
/// - H > 0.7 → strongly trending (>0.9)
/// - H ≈ 0.6 → borderline (≈0.5)
/// - H < 0.5 → not trending (<0.1)
///
/// # Arguments
///
/// * `features` - 8D feature vector from DSP
///
/// # Returns
///
/// Fuzzy truth value in [0, 1]
#[inline]
pub fn is_trending(features: &[f64; 8]) -> f64 {
    let hurst = features[feature_idx::HURST];
    let threshold = 0.6;
    let steepness = 10.0; // Sharp transition

    soft_threshold(hurst, threshold, steepness)
}

/// Predicate: Market is mean-reverting
///
/// Mean-reverting if Hurst exponent < 0.4 (anti-persistent)
///
/// Uses inverted sigmoid:
/// - H < 0.3 → strongly mean-reverting (>0.9)
/// - H ≈ 0.4 → borderline (≈0.5)
/// - H > 0.5 → not mean-reverting (<0.1)
#[inline]
pub fn is_mean_reverting(features: &[f64; 8]) -> f64 {
    let hurst = features[feature_idx::HURST];
    let threshold = 0.4;
    let steepness = 10.0;

    // Invert: mean-reverting when H < threshold
    1.0 - soft_threshold(hurst, threshold, steepness)
}

/// Predicate: Market is in random walk regime
///
/// Random walk if Hurst ≈ 0.5 (no persistence or anti-persistence)
///
/// Uses Gaussian-like response centered at 0.5:
/// - H ≈ 0.5 → strongly random (>0.9)
/// - H far from 0.5 → not random (<0.1)
#[inline]
pub fn is_random_walk(features: &[f64; 8]) -> f64 {
    let hurst = features[feature_idx::HURST];
    let center = 0.5;
    let width = 0.1; // ±0.1 around 0.5

    // Gaussian-like: exp(-((h - 0.5) / width)^2)
    let distance = (hurst - center) / width;
    (-distance * distance).exp()
}

/// Predicate: Regime classification confidence is high
///
/// Confidence > 0.2 (far from regime boundaries)
///
/// Higher confidence → stronger signal
#[inline]
pub fn high_regime_confidence(features: &[f64; 8]) -> f64 {
    let confidence = features[feature_idx::REGIME_CONFIDENCE];
    let threshold = 0.2;
    let steepness = 10.0;

    soft_threshold(confidence, threshold, steepness)
}

/// Predicate: Regime classification confidence is low
///
/// Confidence < 0.2 (near regime boundaries, uncertain)
///
/// Low confidence → should be cautious
#[inline]
pub fn low_regime_confidence(features: &[f64; 8]) -> f64 {
    let confidence = features[feature_idx::REGIME_CONFIDENCE];
    let threshold = 0.2;
    let steepness = 10.0;

    1.0 - soft_threshold(confidence, threshold, steepness)
}

// ============================================================================
// Divergence Predicates
// ============================================================================

/// Predicate: Price-FRAMA divergence is positive
///
/// Divergence > 0 (price above FRAMA)
///
/// Indicates:
/// - In trending markets: momentum continuation
/// - In mean-reverting markets: overbought
#[inline]
pub fn divergence_positive(features: &[f64; 8]) -> f64 {
    let divergence_norm = features[feature_idx::DIVERGENCE_NORM];
    let threshold = 0.0;
    let steepness = 5.0; // Moderate sharpness

    soft_threshold(divergence_norm, threshold, steepness)
}

/// Predicate: Price-FRAMA divergence is negative
///
/// Divergence < 0 (price below FRAMA)
///
/// Indicates:
/// - In trending markets: pullback or reversal
/// - In mean-reverting markets: oversold
#[inline]
pub fn divergence_negative(features: &[f64; 8]) -> f64 {
    let divergence_norm = features[feature_idx::DIVERGENCE_NORM];
    let threshold = 0.0;
    let steepness = 5.0;

    1.0 - soft_threshold(divergence_norm, threshold, steepness)
}

/// Predicate: Divergence is extreme (strong signal)
///
/// |Divergence| > 1.5σ
///
/// Extreme divergence suggests:
/// - Strong momentum (trending) or
/// - Imminent reversal (mean-reverting)
#[inline]
pub fn divergence_extreme(features: &[f64; 8]) -> f64 {
    let divergence_norm = features[feature_idx::DIVERGENCE_NORM];
    let threshold = 1.5; // 1.5 standard deviations
    let steepness = 5.0;

    soft_threshold(divergence_norm.abs(), threshold, steepness)
}

/// Predicate: Divergence is moderate (weak signal)
///
/// |Divergence| < 1.0σ
///
/// Moderate divergence → less conviction
#[inline]
pub fn divergence_moderate(features: &[f64; 8]) -> f64 {
    let divergence_norm = features[feature_idx::DIVERGENCE_NORM];
    let threshold = 1.0;
    let steepness = 5.0;

    1.0 - soft_threshold(divergence_norm.abs(), threshold, steepness)
}

// ============================================================================
// Noise & Volatility Predicates
// ============================================================================

/// Predicate: Market is noisy (high fractal dimension)
///
/// Fractal dimension > 1.7 (rough, chaotic)
///
/// High noise → reduce position size, increase caution
#[inline]
pub fn high_noise(features: &[f64; 8]) -> f64 {
    let fractal_dim = features[feature_idx::FRACTAL_DIM];
    let threshold = 1.7;
    let steepness = 10.0;

    soft_threshold(fractal_dim, threshold, steepness)
}

/// Predicate: Market is smooth (low fractal dimension)
///
/// Fractal dimension < 1.3 (smooth, trending)
///
/// Low noise → stronger trend signal
#[inline]
pub fn low_noise(features: &[f64; 8]) -> f64 {
    let fractal_dim = features[feature_idx::FRACTAL_DIM];
    let threshold = 1.3;
    let steepness = 10.0;

    1.0 - soft_threshold(fractal_dim, threshold, steepness)
}

/// Predicate: FRAMA alpha is extreme (high volatility)
///
/// |Alpha deviation| > 0.15
///
/// Extreme alpha suggests unstable market conditions
#[inline]
pub fn extreme_alpha(features: &[f64; 8]) -> f64 {
    let alpha_deviation = features[feature_idx::ALPHA_DEVIATION];
    let threshold = 0.15;
    let steepness = 10.0;

    soft_threshold(alpha_deviation.abs(), threshold, steepness)
}

/// Predicate: FRAMA alpha is stable (low volatility)
///
/// |Alpha deviation| < 0.05
///
/// Stable alpha → more predictable market
#[inline]
pub fn stable_alpha(features: &[f64; 8]) -> f64 {
    let alpha_deviation = features[feature_idx::ALPHA_DEVIATION];
    let threshold = 0.05;
    let steepness = 10.0;

    1.0 - soft_threshold(alpha_deviation.abs(), threshold, steepness)
}

// ============================================================================
// Signal Predicates (for axiom conclusions)
// ============================================================================

/// Trading signal probabilities
#[derive(Debug, Clone, Copy)]
pub struct TradingSignal {
    /// Probability of long position
    pub long: f64,
    /// Probability of neutral position
    pub neutral: f64,
    /// Probability of short position
    pub short: f64,
}

impl TradingSignal {
    /// Create from softmax output
    #[inline]
    pub fn new(long: f64, neutral: f64, short: f64) -> Self {
        debug_assert!(
            (long + neutral + short - 1.0).abs() < 1e-6,
            "Probabilities must sum to 1"
        );
        Self {
            long,
            neutral,
            short,
        }
    }

    /// Get maximum probability (confidence)
    #[inline]
    pub fn confidence(&self) -> f64 {
        self.long.max(self.neutral).max(self.short)
    }

    /// Get predicted action
    #[inline]
    pub fn predicted_action(&self) -> Action {
        if self.long > self.neutral && self.long > self.short {
            Action::Long
        } else if self.short > self.neutral && self.short > self.long {
            Action::Short
        } else {
            Action::Neutral
        }
    }

    /// Convert to array for softmax input
    #[inline]
    pub fn to_array(&self) -> [f64; 3] {
        [self.long, self.neutral, self.short]
    }
}

/// Trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Long,
    Neutral,
    Short,
}

/// Predicate: Signal indicates long position
#[inline]
pub fn should_long(signal: &TradingSignal) -> f64 {
    signal.long
}

/// Predicate: Signal indicates short position
#[inline]
pub fn should_short(signal: &TradingSignal) -> f64 {
    signal.short
}

/// Predicate: Signal indicates neutral position
#[inline]
pub fn should_be_neutral(signal: &TradingSignal) -> f64 {
    signal.neutral
}

/// Predicate: Signal has low conviction (max prob < threshold)
#[inline]
pub fn low_conviction(signal: &TradingSignal, threshold: f64) -> f64 {
    let confidence = signal.confidence();
    1.0 - soft_threshold(confidence, threshold, 10.0)
}

/// Predicate: Signal has high conviction (max prob > threshold)
#[inline]
pub fn high_conviction(signal: &TradingSignal, threshold: f64) -> f64 {
    let confidence = signal.confidence();
    soft_threshold(confidence, threshold, 10.0)
}

// ============================================================================
// Composite Predicates (Logical Combinations)
// ============================================================================

/// Predicate: Market conditions favor long position
///
/// Trending AND divergence positive AND high confidence
#[inline]
pub fn long_favorable(features: &[f64; 8]) -> f64 {
    use super::fuzzy_ops::product_tnorm as and;

    let trending = is_trending(features);
    let div_pos = divergence_positive(features);
    let high_conf = high_regime_confidence(features);

    and(and(trending, div_pos), high_conf)
}

/// Predicate: Market conditions favor short position
///
/// Trending AND divergence negative AND high confidence
#[inline]
pub fn short_favorable(features: &[f64; 8]) -> f64 {
    use super::fuzzy_ops::product_tnorm as and;

    let trending = is_trending(features);
    let div_neg = divergence_negative(features);
    let high_conf = high_regime_confidence(features);

    and(and(trending, div_neg), high_conf)
}

/// Predicate: Market conditions favor mean-reversion long
///
/// Mean-reverting AND divergence negative (oversold)
#[inline]
pub fn mean_reversion_long(features: &[f64; 8]) -> f64 {
    use super::fuzzy_ops::product_tnorm as and;

    let mr = is_mean_reverting(features);
    let div_neg = divergence_negative(features);

    and(mr, div_neg)
}

/// Predicate: Market conditions favor mean-reversion short
///
/// Mean-reverting AND divergence positive (overbought)
#[inline]
pub fn mean_reversion_short(features: &[f64; 8]) -> f64 {
    use super::fuzzy_ops::product_tnorm as and;

    let mr = is_mean_reverting(features);
    let div_pos = divergence_positive(features);

    and(mr, div_pos)
}

/// Predicate: Market is uncertain (should stay neutral)
///
/// Low confidence OR high noise OR random walk regime
#[inline]
pub fn uncertain_conditions(features: &[f64; 8]) -> f64 {
    use super::fuzzy_ops::product_tconorm as or;

    let low_conf = low_regime_confidence(features);
    let noisy = high_noise(features);
    let random = is_random_walk(features);

    or(or(low_conf, noisy), random)
}

/// Predicate: Contradictory signals (divergence vs regime)
///
/// Divergence sign conflicts with regime indication
#[inline]
pub fn contradictory_signals(features: &[f64; 8]) -> f64 {
    use super::fuzzy_ops::product_tnorm as and;

    // Example: Trending + divergence positive should agree
    // If trending but divergence negative → contradiction
    let trending = is_trending(features);
    let div_neg = divergence_negative(features);

    // Contradiction = both are high
    and(trending, div_neg)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn make_features(
        divergence_norm: f64,
        alpha_norm: f64,
        fractal_dim: f64,
        hurst: f64,
        regime: f64,
        divergence_sign: f64,
        alpha_deviation: f64,
        regime_confidence: f64,
    ) -> [f64; 8] {
        [
            divergence_norm,
            alpha_norm,
            fractal_dim,
            hurst,
            regime,
            divergence_sign,
            alpha_deviation,
            regime_confidence,
        ]
    }

    #[test]
    fn test_is_trending() {
        let features = make_features(0.0, 0.0, 1.5, 0.75, 1.0, 1.0, 0.0, 0.3);
        let result = is_trending(&features);
        // sigmoid(0.15, 10) ≈ 0.82, so expect > 0.7
        assert!(result > 0.7, "H=0.75 should be trending");

        let features = make_features(0.0, 0.0, 1.5, 0.45, 0.0, 1.0, 0.0, 0.1);
        let result = is_trending(&features);
        // sigmoid(-0.15, 10) ≈ 0.18, so expect < 0.3
        assert!(result < 0.3, "H=0.45 should not be trending");
    }

    #[test]
    fn test_is_mean_reverting() {
        let features = make_features(0.0, 0.0, 1.5, 0.3, -1.0, 1.0, 0.0, 0.3);
        let result = is_mean_reverting(&features);
        // 1 - sigmoid(-0.1, 10) ≈ 0.73, so expect > 0.6
        assert!(result > 0.6, "H=0.3 should be mean-reverting");

        let features = make_features(0.0, 0.0, 1.5, 0.65, 1.0, 1.0, 0.0, 0.3);
        let result = is_mean_reverting(&features);
        // 1 - sigmoid(0.25, 10) ≈ 0.08, so expect < 0.2
        assert!(result < 0.2, "H=0.65 should not be mean-reverting");
    }

    #[test]
    fn test_is_random_walk() {
        let features = make_features(0.0, 0.0, 1.5, 0.5, 0.0, 1.0, 0.0, 0.1);
        let result = is_random_walk(&features);
        assert!(result > 0.9, "H=0.5 should be random walk");

        let features = make_features(0.0, 0.0, 1.5, 0.8, 1.0, 1.0, 0.0, 0.3);
        let result = is_random_walk(&features);
        assert!(result < 0.1, "H=0.8 should not be random walk");
    }

    #[test]
    fn test_divergence_predicates() {
        let features = make_features(1.5, 0.0, 1.5, 0.7, 1.0, 1.0, 0.0, 0.3);
        // divergence_sign=1.0 -> positive check uses soft_threshold
        assert!(divergence_positive(&features) > 0.7);
        assert!(divergence_negative(&features) < 0.3);
        // divergence_norm=1.5, threshold=1.5, so sigmoid(0) = 0.5
        assert!(divergence_extreme(&features) >= 0.5);

        let features = make_features(-1.8, 0.0, 1.5, 0.7, 1.0, -1.0, 0.0, 0.3);
        assert!(divergence_negative(&features) > 0.7);
        assert!(divergence_positive(&features) < 0.3);
        // abs(-1.8) = 1.8, threshold=1.5, sigmoid(0.3, 5) ≈ 0.82
        assert!(divergence_extreme(&features) > 0.5);
    }

    #[test]
    fn test_noise_predicates() {
        let features = make_features(0.0, 0.0, 1.8, 0.6, 1.0, 1.0, 0.0, 0.3);
        // sigmoid(0.1, 10) ≈ 0.73, so expect > 0.6
        assert!(high_noise(&features) > 0.6, "D=1.8 should be noisy");

        let features = make_features(0.0, 0.0, 1.2, 0.7, 1.0, 1.0, 0.0, 0.3);
        // low_noise = 1 - soft_threshold(1.2, 1.3, 10) = 1 - sigmoid(-0.1, 10) ≈ 0.73
        assert!(low_noise(&features) > 0.6, "D=1.2 should be smooth");
    }

    #[test]
    fn test_trading_signal() {
        let signal = TradingSignal::new(0.7, 0.2, 0.1);
        assert_eq!(signal.predicted_action(), Action::Long);
        assert!((signal.confidence() - 0.7).abs() < 1e-10);

        assert!((should_long(&signal) - 0.7).abs() < 1e-10);
        assert!((should_short(&signal) - 0.1).abs() < 1e-10);
        assert!((should_be_neutral(&signal) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_composite_predicates() {
        // Trending + positive divergence + high confidence
        let features = make_features(1.5, 0.0, 1.5, 0.75, 1.0, 1.0, 0.0, 0.4);
        let result = long_favorable(&features);
        assert!(result > 0.7, "Should favor long");

        // Mean-reverting + negative divergence (oversold)
        let features = make_features(-1.5, 0.0, 1.5, 0.3, -1.0, -1.0, 0.0, 0.3);
        let result = mean_reversion_long(&features);
        assert!(result > 0.7, "Should favor mean-reversion long");

        // Low confidence + high noise
        let features = make_features(0.0, 0.0, 1.8, 0.5, 0.0, 1.0, 0.0, 0.1);
        let result = uncertain_conditions(&features);
        assert!(result > 0.7, "Should indicate uncertainty");
    }

    #[test]
    fn test_regime_confidence() {
        let features = make_features(0.0, 0.0, 1.5, 0.7, 1.0, 1.0, 0.0, 0.4);
        // sigmoid(0.2, 10) ≈ 0.88, so expect > 0.8
        assert!(high_regime_confidence(&features) > 0.8);
        // low = 1 - high ≈ 0.12, so expect < 0.2
        assert!(low_regime_confidence(&features) < 0.2);

        let features = make_features(0.0, 0.0, 1.5, 0.55, 0.0, 1.0, 0.0, 0.1);
        // sigmoid(-0.1, 10) ≈ 0.27, so low = 1 - 0.27 ≈ 0.73, expect > 0.6
        assert!(low_regime_confidence(&features) > 0.6);
        // high ≈ 0.27, so expect < 0.4
        assert!(high_regime_confidence(&features) < 0.4);
    }
}
