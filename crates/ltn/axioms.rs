//! Axiom Library for Logic Tensor Networks
//!
//! This module implements the complete axiom library for Project JANUS,
//! encoding market domain knowledge as differentiable logical rules.
//!
//! Each axiom:
//! - Combines fuzzy predicates into IF-THEN rules
//! - Returns a satisfaction score in [0, 1]
//! - Is fully differentiable for gradient descent
//! - Encodes expert knowledge about market behavior
//!
//! # Axiom Categories
//!
//! 1. **Regime-Based**: Link market regime to trading signals
//! 2. **Divergence-Based**: Link price-FRAMA divergence to direction
//! 3. **Confidence-Based**: Link uncertainty to neutral positions
//! 4. **Contradiction**: Prevent logically inconsistent signals
//! 5. **Risk Management**: Encode safety constraints
//!
//! # Example
//!
//! ```ignore
//! use janus_ltn::axioms::{AxiomLibrary, evaluate_all_axioms};
//! use janus_ltn::predicates::TradingSignal;
//!
//! let features = [0.5, 0.0, 1.5, 0.75, 1.0, 1.0, 0.0, 0.3]; // DSP output
//! let signal = TradingSignal::new(0.8, 0.15, 0.05); // Model prediction
//!
//! let axiom_lib = AxiomLibrary::default();
//! let results = axiom_lib.evaluate_all(&features, &signal);
//!
//! // Check individual axiom satisfaction
//! for result in &results {
//!     println!("Axiom {}: satisfaction = {:.3}",
//!              result.axiom_id, result.satisfaction);
//! }
//!
//! // Compute weighted semantic loss
//! let semantic_loss = axiom_lib.compute_semantic_loss(&results);
//! ```

use super::fuzzy_ops::{
    ImplicationType, TNormType, product_tconorm, product_tnorm, reichenbach_implication,
};
use super::predicates::{self, TradingSignal};

/// Result of evaluating a single axiom
#[derive(Debug, Clone, Copy)]
pub struct AxiomResult {
    /// Axiom identifier (0-9 for the 10 core axioms)
    pub axiom_id: usize,
    /// Satisfaction score in [0, 1]
    pub satisfaction: f64,
    /// Weight assigned to this axiom
    pub weight: f64,
    /// Weighted contribution to semantic loss
    pub weighted_satisfaction: f64,
}

impl AxiomResult {
    /// Create a new axiom result
    #[inline]
    pub fn new(axiom_id: usize, satisfaction: f64, weight: f64) -> Self {
        Self {
            axiom_id,
            satisfaction,
            weight,
            weighted_satisfaction: weight * satisfaction,
        }
    }
}

/// Axiom evaluation statistics
#[derive(Debug, Clone, Copy)]
pub struct AxiomStats {
    /// Mean satisfaction across all axioms
    pub mean_satisfaction: f64,
    /// Minimum satisfaction (weakest axiom)
    pub min_satisfaction: f64,
    /// Maximum satisfaction (strongest axiom)
    pub max_satisfaction: f64,
    /// Number of axioms with satisfaction > 0.8 (well satisfied)
    pub well_satisfied_count: usize,
    /// Number of axioms with satisfaction < 0.5 (violated)
    pub violated_count: usize,
}

// ============================================================================
// Core Axioms (10 market trading rules)
// ============================================================================

/// Axiom 1: Trending Market + Positive Divergence → Long Signal
///
/// **Logical Form**: `∀x: (is_trending(x) ∧ divergence_positive(x)) → should_long(x)`
///
/// **Intuition**: In trending markets (H > 0.6), if price is above FRAMA
/// (positive divergence), we should have a long signal (follow momentum).
///
/// **Weight**: 2.0 (core momentum strategy)
#[inline]
pub fn axiom_trending_long(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let is_trending = predicates::is_trending(features);
    let div_positive = predicates::divergence_positive(features);
    let should_long = predicates::should_long(signal);

    // Conjunction: both conditions must be true
    let premise = product_tnorm(is_trending, div_positive);

    // Implication: if premise is true, conclusion should be true
    reichenbach_implication(premise, should_long)
}

/// Axiom 2: Trending Market + Negative Divergence → Short Signal
///
/// **Logical Form**: `∀x: (is_trending(x) ∧ divergence_negative(x)) → should_short(x)`
///
/// **Intuition**: In trending markets, if price is below FRAMA
/// (negative divergence), we should have a short signal.
///
/// **Weight**: 2.0 (core momentum strategy)
#[inline]
pub fn axiom_trending_short(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let is_trending = predicates::is_trending(features);
    let div_negative = predicates::divergence_negative(features);
    let should_short = predicates::should_short(signal);

    let premise = product_tnorm(is_trending, div_negative);
    reichenbach_implication(premise, should_short)
}

/// Axiom 3: Mean-Reverting + Positive Divergence → Short Signal (Fade)
///
/// **Logical Form**: `∀x: (is_mean_reverting(x) ∧ divergence_positive(x)) → should_short(x)`
///
/// **Intuition**: In mean-reverting markets (H < 0.4), if price is too high
/// (positive divergence), fade it by going short (expect reversion).
///
/// **Weight**: 1.5 (mean-reversion secondary strategy)
#[inline]
pub fn axiom_mean_reverting_short(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let is_mr = predicates::is_mean_reverting(features);
    let div_positive = predicates::divergence_positive(features);
    let should_short = predicates::should_short(signal);

    let premise = product_tnorm(is_mr, div_positive);
    reichenbach_implication(premise, should_short)
}

/// Axiom 4: Mean-Reverting + Negative Divergence → Long Signal (Fade)
///
/// **Logical Form**: `∀x: (is_mean_reverting(x) ∧ divergence_negative(x)) → should_long(x)`
///
/// **Intuition**: In mean-reverting markets, if price is too low
/// (negative divergence), fade it by going long (expect bounce).
///
/// **Weight**: 1.5 (mean-reversion secondary strategy)
#[inline]
pub fn axiom_mean_reverting_long(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let is_mr = predicates::is_mean_reverting(features);
    let div_negative = predicates::divergence_negative(features);
    let should_long = predicates::should_long(signal);

    let premise = product_tnorm(is_mr, div_negative);
    reichenbach_implication(premise, should_long)
}

/// Axiom 5: Low Confidence → Neutral Position
///
/// **Logical Form**: `∀x: low_confidence(x) → should_be_neutral(x)`
///
/// **Intuition**: When regime confidence is low (< 0.2), we're uncertain
/// about market state, so stay neutral (critical risk control).
///
/// **Weight**: 3.0 (CRITICAL - risk management)
#[inline]
pub fn axiom_low_confidence_neutral(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let low_confidence = predicates::low_regime_confidence(features);
    let should_neutral = predicates::should_be_neutral(signal);

    reichenbach_implication(low_confidence, should_neutral)
}

/// Axiom 6: High Noise → Neutral or Low Conviction
///
/// **Logical Form**: `∀x: high_noise(x) → (should_be_neutral(x) ∨ low_conviction(x))`
///
/// **Intuition**: When fractal dimension is high (> 1.7, noisy market),
/// either stay neutral or have low conviction in directional signals.
///
/// **Weight**: 2.5 (risk management)
#[inline]
pub fn axiom_high_noise_caution(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let high_noise = predicates::high_noise(features);
    let should_neutral = predicates::should_be_neutral(signal);
    let low_conviction = predicates::low_conviction(signal, 0.6);

    // Disjunction: either neutral OR low conviction
    let conclusion = product_tconorm(should_neutral, low_conviction);

    reichenbach_implication(high_noise, conclusion)
}

/// Axiom 7: Contradictory Signals → Neutral
///
/// **Logical Form**: `∀x: contradictory(x) → should_be_neutral(x)`
///
/// **Intuition**: When divergence and regime give conflicting signals
/// (e.g., trending but negative divergence), stay neutral for safety.
///
/// **Weight**: 2.0 (logical consistency)
#[inline]
pub fn axiom_contradiction_neutral(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let contradictory = predicates::contradictory_signals(features);
    let should_neutral = predicates::should_be_neutral(signal);

    reichenbach_implication(contradictory, should_neutral)
}

/// Axiom 8: Extreme Alpha → Neutral (High Volatility Caution)
///
/// **Logical Form**: `∀x: extreme_alpha(x) → should_be_neutral(x)`
///
/// **Intuition**: Extreme FRAMA alpha (|deviation| > 0.15) indicates
/// unstable market conditions, so reduce exposure.
///
/// **Weight**: 1.0 (edge case handling)
#[inline]
pub fn axiom_extreme_alpha_neutral(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let extreme_alpha = predicates::extreme_alpha(features);
    let should_neutral = predicates::should_be_neutral(signal);

    reichenbach_implication(extreme_alpha, should_neutral)
}

/// Axiom 9: Probability Consistency (Sum to 1)
///
/// **Logical Form**: `∀x: |P_long + P_neutral + P_short - 1| ≈ 0`
///
/// **Intuition**: Probabilities must sum to 1 (enforced by softmax,
/// but axiom reinforces as a soft constraint).
///
/// **Weight**: 5.0 (HARD constraint - mathematical consistency)
#[inline]
pub fn axiom_probability_sum(signal: &TradingSignal) -> f64 {
    let sum = signal.long + signal.neutral + signal.short;
    let deviation = (sum - 1.0).abs();

    // Satisfaction = 1 - deviation (clamped to [0, 1])
    // Perfect sum → satisfaction = 1.0
    // Deviation of 0.1 → satisfaction = 0.9
    (1.0 - deviation).clamp(0.0, 1.0)
}

/// Axiom 10: Confidence Monotonicity
///
/// **Logical Form**: `∀x: high_regime_confidence(x) → high_conviction(signal)`
///
/// **Intuition**: Higher regime confidence should correlate with stronger
/// directional conviction (preference, not strict requirement).
///
/// **Weight**: 1.0 (soft preference)
#[inline]
pub fn axiom_confidence_monotonicity(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let high_confidence = predicates::high_regime_confidence(features);
    let high_conviction = predicates::high_conviction(signal, 0.6);

    reichenbach_implication(high_confidence, high_conviction)
}

// ============================================================================
// Axiom Library
// ============================================================================

/// Complete axiom library with weights and evaluation logic
#[derive(Debug, Clone)]
pub struct AxiomLibrary {
    /// Axiom weights (10 axioms)
    pub weights: [f64; 10],
    /// T-norm type for conjunctions
    pub tnorm_type: TNormType,
    /// Implication type for axioms
    pub impl_type: ImplicationType,
}

impl AxiomLibrary {
    /// Create a new axiom library with custom weights
    pub fn new(weights: [f64; 10]) -> Self {
        Self {
            weights,
            tnorm_type: TNormType::Product,
            impl_type: ImplicationType::Reichenbach,
        }
    }
}

impl Default for AxiomLibrary {
    /// Create library with default weights (from design document)
    fn default() -> Self {
        Self::new([
            2.0, // Axiom 1: Trending + pos div → long
            2.0, // Axiom 2: Trending + neg div → short
            1.5, // Axiom 3: MR + pos div → short
            1.5, // Axiom 4: MR + neg div → long
            3.0, // Axiom 5: Low confidence → neutral (CRITICAL)
            2.5, // Axiom 6: High noise → caution
            2.0, // Axiom 7: Contradiction → neutral
            1.0, // Axiom 8: Extreme alpha → neutral
            5.0, // Axiom 9: Probability sum (HARD constraint)
            1.0, // Axiom 10: Confidence monotonicity
        ])
    }
}

impl AxiomLibrary {
    /// Create library with equal weights (for ablation studies)
    pub fn equal_weights() -> Self {
        Self::new([1.0; 10])
    }

    /// Create library with only risk axioms enabled
    pub fn risk_only() -> Self {
        Self::new([
            0.0, // Axiom 1: disabled
            0.0, // Axiom 2: disabled
            0.0, // Axiom 3: disabled
            0.0, // Axiom 4: disabled
            5.0, // Axiom 5: Low confidence → neutral (CRITICAL)
            3.0, // Axiom 6: High noise → caution
            3.0, // Axiom 7: Contradiction → neutral
            2.0, // Axiom 8: Extreme alpha → neutral
            5.0, // Axiom 9: Probability sum
            0.0, // Axiom 10: disabled
        ])
    }

    /// Evaluate a single axiom by ID
    #[inline]
    pub fn evaluate_axiom(
        &self,
        axiom_id: usize,
        features: &[f64; 8],
        signal: &TradingSignal,
    ) -> f64 {
        match axiom_id {
            0 => axiom_trending_long(features, signal),
            1 => axiom_trending_short(features, signal),
            2 => axiom_mean_reverting_short(features, signal),
            3 => axiom_mean_reverting_long(features, signal),
            4 => axiom_low_confidence_neutral(features, signal),
            5 => axiom_high_noise_caution(features, signal),
            6 => axiom_contradiction_neutral(features, signal),
            7 => axiom_extreme_alpha_neutral(features, signal),
            8 => axiom_probability_sum(signal),
            9 => axiom_confidence_monotonicity(features, signal),
            _ => panic!("Invalid axiom ID: {}", axiom_id),
        }
    }

    /// Evaluate all axioms and return results
    pub fn evaluate_all(&self, features: &[f64; 8], signal: &TradingSignal) -> Vec<AxiomResult> {
        (0..10)
            .map(|axiom_id| {
                let satisfaction = self.evaluate_axiom(axiom_id, features, signal);
                let weight = self.weights[axiom_id];
                AxiomResult::new(axiom_id, satisfaction, weight)
            })
            .collect()
    }

    /// Compute semantic loss from axiom results
    ///
    /// Semantic loss = -Σᵢ (wᵢ × satisfaction(Axiom_i))
    ///
    /// Note: Negative because we want to MAXIMIZE satisfaction
    /// (gradient descent minimizes loss).
    pub fn compute_semantic_loss(&self, results: &[AxiomResult]) -> f64 {
        let total_weight: f64 = results.iter().map(|r| r.weight).sum();
        let weighted_sum: f64 = results.iter().map(|r| r.weighted_satisfaction).sum();

        // Normalize by total weight
        let normalized_satisfaction = weighted_sum / total_weight.max(1e-10);

        // Negate for loss (want to maximize satisfaction = minimize negative)
        -normalized_satisfaction
    }

    /// Compute axiom statistics
    pub fn compute_stats(&self, results: &[AxiomResult]) -> AxiomStats {
        let satisfactions: Vec<f64> = results.iter().map(|r| r.satisfaction).collect();

        let mean_satisfaction = satisfactions.iter().sum::<f64>() / satisfactions.len() as f64;
        let min_satisfaction = satisfactions.iter().copied().fold(f64::INFINITY, f64::min);
        let max_satisfaction = satisfactions
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let well_satisfied_count = satisfactions.iter().filter(|&&s| s > 0.8).count();
        let violated_count = satisfactions.iter().filter(|&&s| s < 0.5).count();

        AxiomStats {
            mean_satisfaction,
            min_satisfaction,
            max_satisfaction,
            well_satisfied_count,
            violated_count,
        }
    }

    /// Get total weight of all axioms
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Get axiom name by ID
    pub fn axiom_name(&self, axiom_id: usize) -> &'static str {
        match axiom_id {
            0 => "Trending + Positive Div → Long",
            1 => "Trending + Negative Div → Short",
            2 => "Mean-Reverting + Positive Div → Short",
            3 => "Mean-Reverting + Negative Div → Long",
            4 => "Low Confidence → Neutral",
            5 => "High Noise → Caution",
            6 => "Contradiction → Neutral",
            7 => "Extreme Alpha → Neutral",
            8 => "Probability Sum = 1",
            9 => "Confidence Monotonicity",
            _ => "Unknown",
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Evaluate all axioms with default weights
#[inline]
pub fn evaluate_all_axioms(features: &[f64; 8], signal: &TradingSignal) -> Vec<AxiomResult> {
    let lib = AxiomLibrary::default();
    lib.evaluate_all(features, signal)
}

/// Compute semantic loss with default weights
#[inline]
pub fn compute_semantic_loss(features: &[f64; 8], signal: &TradingSignal) -> f64 {
    let lib = AxiomLibrary::default();
    let results = lib.evaluate_all(features, signal);
    lib.compute_semantic_loss(&results)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(
        divergence_norm: f64,
        hurst: f64,
        regime_confidence: f64,
        fractal_dim: f64,
    ) -> [f64; 8] {
        [
            divergence_norm,
            0.0,
            fractal_dim,
            hurst,
            if hurst > 0.6 { 1.0 } else { -1.0 },
            if divergence_norm > 0.0 { 1.0 } else { -1.0 },
            0.0,
            regime_confidence,
        ]
    }

    #[test]
    fn test_axiom_trending_long() {
        // Trending market + positive divergence
        let features = make_features(1.5, 0.75, 0.4, 1.5);
        let signal = TradingSignal::new(0.9, 0.08, 0.02); // Strong long

        let satisfaction = axiom_trending_long(&features, &signal);
        assert!(satisfaction > 0.8, "Should be well satisfied");
    }

    #[test]
    fn test_axiom_trending_short() {
        // Trending market + negative divergence
        let features = make_features(-1.5, 0.75, 0.4, 1.5);
        let signal = TradingSignal::new(0.02, 0.08, 0.9); // Strong short

        let satisfaction = axiom_trending_short(&features, &signal);
        assert!(satisfaction > 0.8, "Should be well satisfied");
    }

    #[test]
    fn test_axiom_mean_reverting() {
        // Mean-reverting + positive divergence (overbought)
        let features = make_features(1.5, 0.3, 0.3, 1.5);
        let signal = TradingSignal::new(0.05, 0.05, 0.9); // Short to fade

        let satisfaction = axiom_mean_reverting_short(&features, &signal);
        assert!(satisfaction > 0.7, "Should favor short");
    }

    #[test]
    fn test_axiom_low_confidence() {
        // Low confidence (should be neutral)
        // regime_confidence=0.1, low_regime_confidence(0.1) ≈ 0.73
        // reichenbach_implication(0.73, 0.7) ≈ 0.78
        let features = make_features(0.5, 0.55, 0.1, 1.5);
        let signal = TradingSignal::new(0.2, 0.7, 0.1); // Neutral

        let satisfaction = axiom_low_confidence_neutral(&features, &signal);
        assert!(satisfaction > 0.7, "Low confidence should lead to neutral");
    }

    #[test]
    fn test_axiom_high_noise() {
        // High noise (should be cautious)
        let features = make_features(0.5, 0.6, 0.3, 1.85);
        let signal = TradingSignal::new(0.3, 0.5, 0.2); // Neutral-ish

        let satisfaction = axiom_high_noise_caution(&features, &signal);
        assert!(satisfaction > 0.5, "High noise should lead to caution");
    }

    #[test]
    fn test_axiom_probability_sum() {
        // Perfect sum
        let signal = TradingSignal::new(0.7, 0.2, 0.1);
        let satisfaction = axiom_probability_sum(&signal);
        assert!((satisfaction - 1.0).abs() < 1e-6, "Perfect sum = 1.0");

        // Imperfect sum (shouldn't happen with proper softmax, but test it)
        let signal_bad = TradingSignal {
            long: 0.7,
            neutral: 0.2,
            short: 0.05, // Sum = 0.95
        };
        let satisfaction = axiom_probability_sum(&signal_bad);
        assert!(
            satisfaction > 0.9,
            "Small deviation still high satisfaction"
        );
    }

    #[test]
    fn test_axiom_library_default() {
        let lib = AxiomLibrary::default();
        assert_eq!(lib.weights.len(), 10);
        assert_eq!(lib.weights[4], 3.0); // Low confidence weight
        assert_eq!(lib.weights[8], 5.0); // Probability sum weight
    }

    #[test]
    fn test_evaluate_all_axioms() {
        let features = make_features(1.0, 0.7, 0.3, 1.5);
        let signal = TradingSignal::new(0.8, 0.15, 0.05);

        let lib = AxiomLibrary::default();
        let results = lib.evaluate_all(&features, &signal);

        assert_eq!(results.len(), 10);

        // All satisfactions should be in [0, 1]
        for result in &results {
            assert!(result.satisfaction >= 0.0 && result.satisfaction <= 1.0);
        }
    }

    #[test]
    fn test_semantic_loss_computation() {
        let features = make_features(1.0, 0.7, 0.3, 1.5);
        let signal = TradingSignal::new(0.8, 0.15, 0.05);

        let lib = AxiomLibrary::default();
        let results = lib.evaluate_all(&features, &signal);
        let loss = lib.compute_semantic_loss(&results);

        // Loss should be negative (we want to maximize satisfaction)
        assert!(loss < 0.0);

        // Test that semantic loss is computed correctly
        // The loss is the negative of the weighted average satisfaction
        let total_weight: f64 = lib.total_weight();
        let weighted_sum: f64 = results
            .iter()
            .zip(lib.weights.iter())
            .map(|(r, &w)| r.satisfaction * w)
            .sum();
        let expected_loss = -weighted_sum / total_weight;
        assert!(
            (loss - expected_loss).abs() < 1e-10,
            "Loss should equal negative weighted average satisfaction"
        );
    }

    #[test]
    fn test_axiom_stats() {
        let features = make_features(1.0, 0.7, 0.3, 1.5);
        let signal = TradingSignal::new(0.8, 0.15, 0.05);

        let lib = AxiomLibrary::default();
        let results = lib.evaluate_all(&features, &signal);
        let stats = lib.compute_stats(&results);

        assert!(stats.mean_satisfaction >= 0.0 && stats.mean_satisfaction <= 1.0);
        assert!(stats.min_satisfaction >= 0.0);
        assert!(stats.max_satisfaction <= 1.0);
        assert!(stats.min_satisfaction <= stats.mean_satisfaction);
        assert!(stats.mean_satisfaction <= stats.max_satisfaction);
    }

    #[test]
    fn test_axiom_names() {
        let lib = AxiomLibrary::default();
        assert_eq!(lib.axiom_name(0), "Trending + Positive Div → Long");
        assert_eq!(lib.axiom_name(4), "Low Confidence → Neutral");
        assert_eq!(lib.axiom_name(8), "Probability Sum = 1");
    }

    #[test]
    fn test_risk_only_library() {
        let lib = AxiomLibrary::risk_only();

        // Trading axioms should be disabled (weight = 0)
        assert_eq!(lib.weights[0], 0.0);
        assert_eq!(lib.weights[1], 0.0);

        // Risk axioms should be enabled
        assert!(lib.weights[4] > 0.0); // Low confidence
        assert!(lib.weights[5] > 0.0); // High noise
    }

    #[test]
    fn test_violation_detection() {
        // Create a signal that violates low-confidence axiom
        let features = make_features(0.5, 0.55, 0.05, 1.5); // Very low confidence
        let signal = TradingSignal::new(0.9, 0.05, 0.05); // Strong directional (violates)

        let lib = AxiomLibrary::default();
        let results = lib.evaluate_all(&features, &signal);
        let stats = lib.compute_stats(&results);

        // Should have at least one violation
        assert!(stats.violated_count > 0);
    }

    #[test]
    fn test_contradiction_axiom() {
        // Trending market but negative divergence (contradiction)
        let features = make_features(-1.0, 0.75, 0.3, 1.5);
        let signal = TradingSignal::new(0.2, 0.6, 0.2); // Neutral

        let satisfaction = axiom_contradiction_neutral(&features, &signal);
        assert!(satisfaction > 0.5, "Contradiction should favor neutral");
    }
}
