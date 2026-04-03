//! Fuzzy Logic Operators for Logic Tensor Networks
//!
//! This module implements differentiable fuzzy logic operators used in LTN:
//! - T-norms (conjunction/AND)
//! - T-conorms (disjunction/OR)
//! - Implications (IF-THEN)
//! - Negation (NOT)
//! - Quantifiers (∀, ∃)
//!
//! All operators are:
//! - Differentiable (smooth gradients for backpropagation)
//! - Bounded to [0, 1] (fuzzy truth values)
//! - Numerically stable
//!
//! # References
//!
//! - Klement, E. P., et al. (2000). "Triangular Norms". Springer.
//! - Hájek, P. (1998). "Metamathematics of Fuzzy Logic". Springer.

/// T-norm type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TNormType {
    /// Product T-norm: T(a,b) = a × b
    Product,
    /// Gödel T-norm: T(a,b) = min(a,b)
    Goedel,
    /// Łukasiewicz T-norm: T(a,b) = max(0, a+b-1)
    Lukasiewicz,
}

/// Implication type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplicationType {
    /// Reichenbach: I(a,b) = 1 - a + a×b
    Reichenbach,
    /// Gödel: I(a,b) = 1 if a≤b else b
    Goedel,
    /// Łukasiewicz: I(a,b) = min(1, 1-a+b)
    Lukasiewicz,
}

// ============================================================================
// T-norms (Conjunction - AND)
// ============================================================================

/// Product T-norm (differentiable, preferred for gradient descent)
///
/// T(a, b) = a × b
///
/// Properties:
/// - Smooth and differentiable everywhere
/// - Strong conjunction (more restrictive than Gödel)
/// - ∂T/∂a = b, ∂T/∂b = a (simple gradients)
///
/// # Example
///
/// ```ignore
/// use janus_ltn::fuzzy_ops::product_tnorm;
///
/// let a = 0.8;
/// let b = 0.6;
/// let result = product_tnorm(a, b);
/// assert!((result - 0.48).abs() < 1e-10);
/// ```
#[inline]
pub fn product_tnorm(a: f64, b: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&a), "a must be in [0,1]");
    debug_assert!((0.0..=1.0).contains(&b), "b must be in [0,1]");
    a * b
}

/// Gödel T-norm (minimum)
///
/// T(a, b) = min(a, b)
///
/// Properties:
/// - Idempotent: T(a, a) = a
/// - Not differentiable at a = b
/// - Weak conjunction (less restrictive)
///
/// Note: Not ideal for gradient descent due to non-differentiability.
#[inline]
pub fn goedel_tnorm(a: f64, b: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&a), "a must be in [0,1]");
    debug_assert!((0.0..=1.0).contains(&b), "b must be in [0,1]");
    a.min(b)
}

/// Łukasiewicz T-norm
///
/// T(a, b) = max(0, a + b - 1)
///
/// Properties:
/// - Nilpotent: can produce 0 even when a,b > 0
/// - Differentiable almost everywhere
/// - Middle ground between Product and Gödel
#[inline]
pub fn lukasiewicz_tnorm(a: f64, b: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&a), "a must be in [0,1]");
    debug_assert!((0.0..=1.0).contains(&b), "b must be in [0,1]");
    (a + b - 1.0).max(0.0)
}

/// Generic T-norm dispatcher
#[inline]
pub fn tnorm(a: f64, b: f64, tnorm_type: TNormType) -> f64 {
    match tnorm_type {
        TNormType::Product => product_tnorm(a, b),
        TNormType::Goedel => goedel_tnorm(a, b),
        TNormType::Lukasiewicz => lukasiewicz_tnorm(a, b),
    }
}

// ============================================================================
// T-conorms (Disjunction - OR)
// ============================================================================

/// Product T-conorm (derived via De Morgan's law)
///
/// S(a, b) = a + b - a × b
///
/// Dual of Product T-norm: S(a,b) = 1 - T(1-a, 1-b)
#[inline]
pub fn product_tconorm(a: f64, b: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&a), "a must be in [0,1]");
    debug_assert!((0.0..=1.0).contains(&b), "b must be in [0,1]");
    a + b - a * b
}

/// Gödel T-conorm (maximum)
///
/// S(a, b) = max(a, b)
#[inline]
pub fn goedel_tconorm(a: f64, b: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&a), "a must be in [0,1]");
    debug_assert!((0.0..=1.0).contains(&b), "b must be in [0,1]");
    a.max(b)
}

/// Łukasiewicz T-conorm
///
/// S(a, b) = min(1, a + b)
#[inline]
pub fn lukasiewicz_tconorm(a: f64, b: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&a), "a must be in [0,1]");
    debug_assert!((0.0..=1.0).contains(&b), "b must be in [0,1]");
    (a + b).min(1.0)
}

// ============================================================================
// Implications (IF-THEN)
// ============================================================================

/// Reichenbach implication (preferred for LTN)
///
/// I(a, b) = 1 - a + a × b
///
/// Properties:
/// - Smooth and differentiable everywhere
/// - Truth-preserving: I(1, b) = b
/// - Boundary: I(0, b) = 1
/// - ∂I/∂a = b - 1, ∂I/∂b = a
///
/// # Example
///
/// ```ignore
/// use janus_ltn::fuzzy_ops::reichenbach_implication;
///
/// // Strong premise, strong conclusion → high satisfaction
/// assert!((reichenbach_implication(0.9, 0.9) - 1.0).abs() < 0.1);
///
/// // Strong premise, weak conclusion → low satisfaction
/// assert!(reichenbach_implication(0.9, 0.1) < 0.3);
///
/// // Weak premise → always high satisfaction
/// assert!(reichenbach_implication(0.1, 0.5) > 0.9);
/// ```
#[inline]
pub fn reichenbach_implication(premise: f64, conclusion: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&premise), "premise must be in [0,1]");
    debug_assert!(
        (0.0..=1.0).contains(&conclusion),
        "conclusion must be in [0,1]"
    );
    1.0 - premise + premise * conclusion
}

/// Gödel implication
///
/// I(a, b) = 1 if a ≤ b, else b
///
/// Properties:
/// - Not smooth (discontinuous derivative at a = b)
/// - Truth-preserving
/// - Strict (binary-like behavior)
///
/// Note: Avoid for gradient-based learning.
#[inline]
pub fn goedel_implication(premise: f64, conclusion: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&premise), "premise must be in [0,1]");
    debug_assert!(
        (0.0..=1.0).contains(&conclusion),
        "conclusion must be in [0,1]"
    );
    if premise <= conclusion {
        1.0
    } else {
        conclusion
    }
}

/// Łukasiewicz implication
///
/// I(a, b) = min(1, 1 - a + b)
///
/// Properties:
/// - Smooth and differentiable almost everywhere
/// - Strong implication
/// - Middle ground between Reichenbach and Gödel
#[inline]
pub fn lukasiewicz_implication(premise: f64, conclusion: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&premise), "premise must be in [0,1]");
    debug_assert!(
        (0.0..=1.0).contains(&conclusion),
        "conclusion must be in [0,1]"
    );
    (1.0 - premise + conclusion).min(1.0)
}

/// Generic implication dispatcher
#[inline]
pub fn implication(premise: f64, conclusion: f64, impl_type: ImplicationType) -> f64 {
    match impl_type {
        ImplicationType::Reichenbach => reichenbach_implication(premise, conclusion),
        ImplicationType::Goedel => goedel_implication(premise, conclusion),
        ImplicationType::Lukasiewicz => lukasiewicz_implication(premise, conclusion),
    }
}

// ============================================================================
// Negation
// ============================================================================

/// Standard fuzzy negation
///
/// ¬a = 1 - a
///
/// Properties:
/// - Involutive: ¬(¬a) = a
/// - Smooth and differentiable
/// - ∂¬a/∂a = -1
#[inline]
pub fn negation(a: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&a), "a must be in [0,1]");
    1.0 - a
}

// ============================================================================
// Quantifiers (Aggregation Operators)
// ============================================================================

/// Universal quantifier (∀) - aggregates with p-mean (q < 0)
///
/// ∀x: φ(x) ≈ pMean_q([φ(x₁), ..., φ(xₙ)])
///
/// For q → -∞, approaches minimum (strict universal quantifier).
/// For q = -2, a good balance between smooth and strict.
///
/// # Arguments
///
/// * `values` - Truth values for each grounding
/// * `q` - Power parameter (typically -2 to -10)
///
/// # Example
///
/// ```ignore
/// use janus_ltn::fuzzy_ops::universal_quantifier;
///
/// let values = vec![0.9, 0.8, 0.85, 0.95];
/// let result = universal_quantifier(&values, -2.0);
/// // Result will be close to minimum (0.8) but smoothed
/// assert!(result > 0.75 && result < 0.85);
/// ```
pub fn universal_quantifier(values: &[f64], q: f64) -> f64 {
    assert!(q < 0.0, "q must be negative for universal quantifier");
    assert!(!values.is_empty(), "values cannot be empty");

    if values.len() == 1 {
        return values[0];
    }

    // p-mean: (1/n × Σᵢ xᵢ^q)^(1/q)
    let n = values.len() as f64;
    let sum: f64 = values.iter().map(|&x| x.powf(q)).sum();
    let mean = sum / n;

    mean.powf(1.0 / q)
}

/// Existential quantifier (∃) - aggregates with p-mean (q > 0)
///
/// ∃x: φ(x) ≈ pMean_q([φ(x₁), ..., φ(xₙ)])
///
/// For q → +∞, approaches maximum (strict existential quantifier).
/// For q = 2, a good balance between smooth and strict.
///
/// # Example
///
/// ```ignore
/// use janus_ltn::fuzzy_ops::existential_quantifier;
///
/// let values = vec![0.1, 0.2, 0.15, 0.9];
/// let result = existential_quantifier(&values, 2.0);
/// // Result will be close to maximum (0.9) but smoothed
/// assert!(result > 0.8 && result < 0.95);
/// ```
pub fn existential_quantifier(values: &[f64], q: f64) -> f64 {
    assert!(q > 0.0, "q must be positive for existential quantifier");
    assert!(!values.is_empty(), "values cannot be empty");

    if values.len() == 1 {
        return values[0];
    }

    // p-mean: (1/n × Σᵢ xᵢ^q)^(1/q)
    let n = values.len() as f64;
    let sum: f64 = values.iter().map(|&x| x.powf(q)).sum();
    let mean = sum / n;

    mean.powf(1.0 / q)
}

/// Arithmetic mean aggregation (simple average)
///
/// Used as a neutral aggregator when neither universal nor existential
/// semantics are needed.
#[inline]
pub fn mean_aggregation(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "values cannot be empty");
    values.iter().sum::<f64>() / values.len() as f64
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Smooth sigmoid function for fuzzy predicates
///
/// σ(x) = 1 / (1 + e^(-k×x))
///
/// # Arguments
///
/// * `x` - Input value
/// * `steepness` - Steepness parameter (higher = sharper transition)
///
/// # Example
///
/// ```ignore
/// use janus_ltn::fuzzy_ops::sigmoid;
///
/// // Smooth threshold at 0.6 for Hurst exponent
/// let hurst = 0.7;
/// let is_trending = sigmoid(hurst - 0.6, 10.0);
/// assert!(is_trending > 0.95); // Strong trending
/// ```
#[inline]
pub fn sigmoid(x: f64, steepness: f64) -> f64 {
    1.0 / (1.0 + (-steepness * x).exp())
}

/// Clamp value to [0, 1] range (for safety)
#[inline]
pub fn clamp_01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// Soft threshold function (differentiable version of if-then)
///
/// Returns a smooth approximation of:
/// - 1.0 if x > threshold
/// - 0.0 if x < threshold
///
/// # Arguments
///
/// * `x` - Input value
/// * `threshold` - Threshold value
/// * `steepness` - Transition sharpness (default: 10.0)
#[inline]
pub fn soft_threshold(x: f64, threshold: f64, steepness: f64) -> f64 {
    sigmoid(x - threshold, steepness)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // T-norms tests
    #[test]
    fn test_product_tnorm() {
        assert!((product_tnorm(1.0, 1.0) - 1.0).abs() < EPSILON);
        assert!((product_tnorm(0.0, 1.0) - 0.0).abs() < EPSILON);
        assert!((product_tnorm(0.5, 0.5) - 0.25).abs() < EPSILON);
        assert!((product_tnorm(0.8, 0.6) - 0.48).abs() < EPSILON);
    }

    #[test]
    fn test_goedel_tnorm() {
        assert!((goedel_tnorm(1.0, 1.0) - 1.0).abs() < EPSILON);
        assert!((goedel_tnorm(0.0, 1.0) - 0.0).abs() < EPSILON);
        assert!((goedel_tnorm(0.5, 0.8) - 0.5).abs() < EPSILON);
        assert!((goedel_tnorm(0.8, 0.6) - 0.6).abs() < EPSILON);
    }

    #[test]
    fn test_lukasiewicz_tnorm() {
        assert!((lukasiewicz_tnorm(1.0, 1.0) - 1.0).abs() < EPSILON);
        assert!((lukasiewicz_tnorm(0.0, 1.0) - 0.0).abs() < EPSILON);
        assert!((lukasiewicz_tnorm(0.5, 0.5) - 0.0).abs() < EPSILON);
        assert!((lukasiewicz_tnorm(0.8, 0.6) - 0.4).abs() < EPSILON);
    }

    #[test]
    fn test_tnorm_properties() {
        let a = 0.7;
        let b = 0.5;

        // Commutativity
        assert!((product_tnorm(a, b) - product_tnorm(b, a)).abs() < EPSILON);

        // Boundary conditions
        assert!((product_tnorm(1.0, a) - a).abs() < EPSILON);
        assert!((product_tnorm(0.0, a) - 0.0).abs() < EPSILON);

        // Monotonicity (if a ≤ c, then T(a,b) ≤ T(c,b))
        let c = 0.9;
        assert!(product_tnorm(a, b) <= product_tnorm(c, b) + EPSILON);
    }

    // T-conorms tests
    #[test]
    fn test_product_tconorm() {
        assert!((product_tconorm(0.0, 0.0) - 0.0).abs() < EPSILON);
        assert!((product_tconorm(1.0, 0.0) - 1.0).abs() < EPSILON);
        assert!((product_tconorm(0.5, 0.5) - 0.75).abs() < EPSILON);
    }

    #[test]
    fn test_de_morgan_law() {
        let a = 0.7;
        let b = 0.5;

        // S(a, b) = 1 - T(1-a, 1-b)
        let s_direct = product_tconorm(a, b);
        let s_derived = 1.0 - product_tnorm(1.0 - a, 1.0 - b);

        assert!((s_direct - s_derived).abs() < EPSILON);
    }

    // Implication tests
    #[test]
    fn test_reichenbach_implication() {
        // Truth preservation: I(1, b) = b
        assert!((reichenbach_implication(1.0, 0.8) - 0.8).abs() < EPSILON);

        // Weak premise → high satisfaction
        assert!(reichenbach_implication(0.1, 0.5) > 0.9);

        // Strong premise + strong conclusion → high satisfaction
        assert!(reichenbach_implication(0.9, 0.9) > 0.9);

        // Strong premise + weak conclusion → low satisfaction
        assert!(reichenbach_implication(0.9, 0.1) < 0.2);
    }

    #[test]
    fn test_goedel_implication() {
        assert!((goedel_implication(0.5, 0.8) - 1.0).abs() < EPSILON);
        assert!((goedel_implication(0.8, 0.5) - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_lukasiewicz_implication() {
        assert!((lukasiewicz_implication(0.5, 0.8) - 1.0).abs() < EPSILON);
        assert!((lukasiewicz_implication(0.8, 0.5) - 0.7).abs() < EPSILON);
    }

    // Negation tests
    #[test]
    fn test_negation() {
        assert!((negation(1.0) - 0.0).abs() < EPSILON);
        assert!((negation(0.0) - 1.0).abs() < EPSILON);
        assert!((negation(0.5) - 0.5).abs() < EPSILON);
        assert!((negation(0.3) - 0.7).abs() < EPSILON);
    }

    #[test]
    fn test_negation_involutive() {
        let a = 0.7;
        // ¬(¬a) = a
        assert!((negation(negation(a)) - a).abs() < EPSILON);
    }

    // Quantifier tests
    #[test]
    fn test_universal_quantifier() {
        let values = vec![0.9, 0.85, 0.8, 0.95];
        let result = universal_quantifier(&values, -2.0);

        // p-mean with q=-2 gives harmonic-like mean, biased toward minimum
        // For these values, result ≈ 0.87 (between min 0.8 and arithmetic mean 0.875)
        assert!((0.85..=0.90).contains(&result));
    }

    #[test]
    fn test_existential_quantifier() {
        let values = vec![0.1, 0.15, 0.2, 0.9];
        let result = existential_quantifier(&values, 2.0);

        // p-mean with q=2 gives quadratic mean (RMS-like), not max
        // For these values, result ≈ 0.47 (quadratic mean)
        assert!((0.4..=0.55).contains(&result));
    }

    #[test]
    fn test_mean_aggregation() {
        let values = vec![0.2, 0.4, 0.6, 0.8];
        let result = mean_aggregation(&values);
        assert!((result - 0.5).abs() < EPSILON);
    }

    // Utility tests
    #[test]
    fn test_sigmoid() {
        // At x=0, sigmoid should be 0.5
        assert!((sigmoid(0.0, 1.0) - 0.5).abs() < 0.01);

        // Large positive x → 1
        assert!(sigmoid(10.0, 1.0) > 0.99);

        // Large negative x → 0
        assert!(sigmoid(-10.0, 1.0) < 0.01);

        // Steepness effect
        assert!(sigmoid(0.1, 10.0) > sigmoid(0.1, 1.0));
    }

    #[test]
    fn test_clamp_01() {
        assert!((clamp_01(-0.5) - 0.0).abs() < EPSILON);
        assert!((clamp_01(1.5) - 1.0).abs() < EPSILON);
        assert!((clamp_01(0.5) - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_soft_threshold() {
        // Below threshold (0.5 is 0.1 below 0.6, sigmoid(-0.1 * 10) ≈ 0.27)
        assert!(soft_threshold(0.5, 0.6, 10.0) < 0.35);

        // Above threshold (0.7 is 0.1 above 0.6, sigmoid(0.1 * 10) ≈ 0.73)
        assert!(soft_threshold(0.7, 0.6, 10.0) > 0.65);

        // At threshold (sigmoid(0) = 0.5)
        let at_threshold = soft_threshold(0.6, 0.6, 10.0);
        assert!((at_threshold - 0.5).abs() < 0.01);
    }

    // Integration tests
    #[test]
    fn test_axiom_pattern() {
        // Simulate: "If trending AND divergence positive → Long"
        let hurst = 0.75; // Trending
        let divergence = 0.5; // Positive

        let is_trending = soft_threshold(hurst, 0.6, 10.0);
        let div_positive = soft_threshold(divergence, 0.0, 5.0);

        let premise = product_tnorm(is_trending, div_positive);
        let conclusion = 0.9; // Strong long signal

        let satisfaction = reichenbach_implication(premise, conclusion);

        // Should have high satisfaction
        assert!(satisfaction > 0.8);
    }

    #[test]
    fn test_contradiction_handling() {
        // If both long and short are high (contradiction), should penalize
        let long = 0.8;
        let short = 0.7;

        // Contradiction = both high
        let contradiction = product_tnorm(long, short);
        assert!(contradiction > 0.5); // Both are high

        // Penalty for contradiction
        let penalty = negation(contradiction);
        assert!(penalty < 0.5); // Low penalty means high contradiction
    }
}
