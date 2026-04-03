//! Welford Online Normalization with Exponential Weighting
//!
//! This module implements streaming mean and variance calculation using
//! Welford's algorithm with exponential decay for non-stationary markets.
//!
//! # Algorithm
//!
//! Traditional Welford uses equal weights for all observations.
//! We extend it with exponential weighting to adapt to regime changes:
//!
//! - mean[t] = mean[t-1] + α * (x[t] - mean[t-1])
//! - M2[t] = (1-α) * M2[t-1] + α * (x[t] - mean[t-1]) * (x[t] - mean[t])
//! - variance[t] = M2[t]
//!
//! # Performance
//!
//! - Time complexity: O(1) per update
//! - Space complexity: O(1)
//! - Zero allocations
//! - Numerically stable (Welford's method avoids catastrophic cancellation)
//!
//! # References
//!
//! Welford, B. P. (1962). "Note on a method for calculating corrected sums
//! of squares and products". Technometrics, 4(3), 419-420.

/// Online normalization state
#[derive(Debug, Clone)]
pub struct WelfordNormalizer {
    /// Exponential decay factor (alpha)
    /// Higher = faster adaptation, lower = more stable
    alpha: f64,

    /// Running mean
    mean: f64,

    /// Running M2 (for variance calculation)
    m2: f64,

    /// Number of samples seen (capped at warmup)
    count: usize,

    /// Warmup period (minimum samples before normalizing)
    warmup: usize,

    /// Whether to clip outliers (Z-score threshold)
    clip_threshold: Option<f64>,
}

/// Normalization result
#[derive(Debug, Clone, Copy)]
pub struct NormalizedValue {
    /// Original value
    pub raw: f64,

    /// Normalized value (Z-score)
    pub normalized: f64,

    /// Current mean estimate
    pub mean: f64,

    /// Current standard deviation estimate
    pub std_dev: f64,

    /// Whether this value was clipped
    pub clipped: bool,

    /// Current sample count
    pub count: usize,
}

/// Error types for normalization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationError {
    /// Insufficient data (still warming up)
    InsufficientData,

    /// Invalid input (NaN or Inf)
    InvalidValue,

    /// Zero or negative variance (degenerate case)
    InvalidVariance,
}

impl std::fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData => write!(f, "Insufficient data for normalization"),
            Self::InvalidValue => write!(f, "NaN or Inf encountered"),
            Self::InvalidVariance => write!(f, "Variance is zero or negative"),
        }
    }
}

impl std::error::Error for NormalizationError {}

impl Default for WelfordNormalizer {
    fn default() -> Self {
        Self::new(0.05, 50, None)
    }
}

impl WelfordNormalizer {
    /// Create a new Welford normalizer
    ///
    /// # Arguments
    ///
    /// * `alpha` - Exponential decay factor (0.0 to 1.0)
    ///   - 0.01 = slow adaptation (100-period half-life)
    ///   - 0.1 = medium adaptation (7-period half-life)
    ///   - 0.5 = fast adaptation (1-period half-life)
    /// * `warmup` - Minimum samples before normalization (typically 30-100)
    /// * `clip_threshold` - Optional Z-score clipping threshold (e.g., 3.0 for ±3σ)
    ///
    /// # Panics
    ///
    /// Panics if alpha is not in (0, 1] or warmup < 2
    pub fn new(alpha: f64, warmup: usize, clip_threshold: Option<f64>) -> Self {
        assert!(
            0.0 < alpha && alpha <= 1.0,
            "Alpha must be in (0, 1], got {}",
            alpha
        );
        assert!(warmup >= 2, "Warmup must be at least 2, got {}", warmup);

        if let Some(thresh) = clip_threshold {
            assert!(
                thresh > 0.0,
                "Clip threshold must be positive, got {}",
                thresh
            );
        }

        Self {
            alpha,
            mean: 0.0,
            m2: 0.0,
            count: 0,
            warmup,
            clip_threshold,
        }
    }

    /// Create a fast-adapting normalizer (for high-frequency regime changes)
    #[inline]
    pub fn fast() -> Self {
        Self::new(0.2, 30, Some(3.0))
    }

    /// Create a slow-adapting normalizer (for stable regimes)
    #[inline]
    pub fn slow() -> Self {
        Self::new(0.01, 100, None)
    }

    /// Update with a new value and return normalized result
    ///
    /// # Arguments
    ///
    /// * `value` - New observation
    ///
    /// # Returns
    ///
    /// - `Ok(NormalizedValue)` with normalization result
    /// - `Err(NormalizationError)` if insufficient data or invalid value
    ///
    /// # Performance
    ///
    /// This method is on the critical hot path and must be:
    /// - Allocation-free
    /// - Branch-predictor friendly
    /// - SIMD-friendly (future optimization)
    #[inline]
    pub fn update(&mut self, value: f64) -> Result<NormalizedValue, NormalizationError> {
        // Validate input
        if !value.is_finite() {
            return Err(NormalizationError::InvalidValue);
        }

        // Update running statistics
        self.count += 1;

        if self.count == 1 {
            // First sample
            self.mean = value;
            self.m2 = 0.0;

            return Err(NormalizationError::InsufficientData);
        }

        // Welford update with exponential weighting
        let delta = value - self.mean;
        self.mean += self.alpha * delta;

        let delta2 = value - self.mean;
        self.m2 = (1.0 - self.alpha) * self.m2 + self.alpha * delta * delta2;

        // Need warmup period before normalizing
        if self.count < self.warmup {
            return Err(NormalizationError::InsufficientData);
        }

        // Calculate variance and standard deviation
        let variance = self.m2;

        // Protect against degenerate case (flat line)
        const MIN_VARIANCE: f64 = 1e-10;

        if variance < MIN_VARIANCE {
            // Return unnormalized value (or zero) when variance is too small
            return Ok(NormalizedValue {
                raw: value,
                normalized: 0.0,
                mean: self.mean,
                std_dev: MIN_VARIANCE.sqrt(),
                clipped: false,
                count: self.count,
            });
        }

        let std_dev = variance.sqrt();

        // Calculate Z-score
        let mut z_score = (value - self.mean) / std_dev;

        // Optional clipping
        let clipped = if let Some(threshold) = self.clip_threshold {
            let original_z = z_score;
            z_score = z_score.clamp(-threshold, threshold);
            (original_z - z_score).abs() > 1e-10
        } else {
            false
        };

        Ok(NormalizedValue {
            raw: value,
            normalized: z_score,
            mean: self.mean,
            std_dev,
            clipped,
            count: self.count,
        })
    }

    /// Get current mean estimate
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance estimate
    #[inline]
    pub fn variance(&self) -> f64 {
        self.m2
    }

    /// Get current standard deviation estimate
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.m2.sqrt()
    }

    /// Get sample count
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Check if warmed up (ready for normalization)
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.count >= self.warmup
    }

    /// Reset the normalizer state
    #[inline]
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
    }

    /// Get the alpha (decay factor)
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set a new alpha (for adaptive regime changes)
    ///
    /// # Panics
    ///
    /// Panics if alpha is not in (0, 1]
    #[inline]
    pub fn set_alpha(&mut self, alpha: f64) {
        assert!(
            0.0 < alpha && alpha <= 1.0,
            "Alpha must be in (0, 1], got {}",
            alpha
        );
        self.alpha = alpha;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_period() {
        let mut norm = WelfordNormalizer::new(0.1, 50, None);

        // First 49 updates should return InsufficientData
        for i in 0..49 {
            let result = norm.update(100.0 + i as f64);
            assert!(matches!(result, Err(NormalizationError::InsufficientData)));
        }

        // 50th update should succeed
        let result = norm.update(149.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mean_tracking() {
        let mut norm = WelfordNormalizer::new(0.1, 10, None);

        // Feed constant values
        for _ in 0..100 {
            let _ = norm.update(100.0);
        }

        // Mean should converge to 100.0
        assert!((norm.mean() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_variance_estimation() {
        let mut norm = WelfordNormalizer::new(0.05, 30, None);

        // Feed random values with known distribution
        // mean=100, std=10
        for i in 0..1000 {
            let value = 100.0 + 10.0 * ((i as f64 * 0.1).sin());
            let _ = norm.update(value);
        }

        let result = norm.update(100.0).unwrap();

        // Should estimate mean ≈ 100 and std ≈ 7 (sine wave has lower std than uniform)
        assert!((result.mean - 100.0).abs() < 5.0, "Mean: {}", result.mean);
        assert!(
            result.std_dev > 1.0 && result.std_dev < 20.0,
            "Std: {}",
            result.std_dev
        );
    }

    #[test]
    fn test_normalization() {
        let mut norm = WelfordNormalizer::new(0.1, 30, None);

        // Warmup with mean=100, std=10
        for i in 0..50 {
            let value = 100.0 + (i % 10) as f64 - 5.0;
            let _ = norm.update(value);
        }

        // Test value 1 std above mean
        let result = norm.update(110.0).unwrap();

        // Z-score should be positive
        assert!(result.normalized > 0.0);

        // Test value 1 std below mean
        let result = norm.update(90.0).unwrap();

        // Z-score should be negative
        assert!(result.normalized < 0.0);
    }

    #[test]
    fn test_clipping() {
        // Use smaller alpha (0.05) so variance adapts slower and extreme values trigger clipping
        let mut norm = WelfordNormalizer::new(0.05, 30, Some(3.0));

        // Warmup
        for i in 0..50 {
            let _ = norm.update(100.0 + (i % 10) as f64 - 5.0);
        }

        // Feed extreme outlier
        let result = norm.update(1000.0).unwrap();

        // Should be clipped to ±3σ
        assert!(result.clipped);
        assert!(result.normalized.abs() <= 3.0 + 1e-6);
    }

    #[test]
    fn test_flat_line() {
        let mut norm = WelfordNormalizer::new(0.1, 30, None);

        // Feed constant value
        for _ in 0..100 {
            let _ = norm.update(100.0);
        }

        let result = norm.update(100.0).unwrap();

        // Should handle zero variance gracefully
        assert!((result.normalized).abs() < 1e-6);
    }

    #[test]
    fn test_regime_change() {
        let mut norm = WelfordNormalizer::new(0.2, 30, None); // Fast adaptation

        // First regime: mean=100
        for _ in 0..100 {
            let _ = norm.update(100.0);
        }

        assert!((norm.mean() - 100.0).abs() < 1.0);

        // Regime shift to mean=200
        for _ in 0..100 {
            let _ = norm.update(200.0);
        }

        // Mean should adapt (not exactly 200 due to memory)
        assert!(norm.mean() > 150.0, "Mean: {}", norm.mean());
    }

    #[test]
    fn test_invalid_input() {
        let mut norm = WelfordNormalizer::new(0.1, 30, None);

        // Warmup
        for i in 0..50 {
            let _ = norm.update(100.0 + i as f64);
        }

        // NaN should be rejected
        let result = norm.update(f64::NAN);
        assert!(matches!(result, Err(NormalizationError::InvalidValue)));

        // Infinity should be rejected
        let result = norm.update(f64::INFINITY);
        assert!(matches!(result, Err(NormalizationError::InvalidValue)));

        // State should be unchanged
        assert_eq!(norm.count(), 50);
    }

    #[test]
    fn test_reset() {
        let mut norm = WelfordNormalizer::new(0.1, 30, None);

        // Warmup
        for i in 0..50 {
            let _ = norm.update(100.0 + i as f64);
        }

        assert!(norm.is_ready());

        // Reset
        norm.reset();

        assert_eq!(norm.count(), 0);
        assert!(!norm.is_ready());
        assert_eq!(norm.mean(), 0.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_alpha() {
        // Alpha > 1 should panic
        let _norm = WelfordNormalizer::new(1.5, 30, None);
    }

    #[test]
    #[should_panic]
    fn test_invalid_warmup() {
        // Warmup < 2 should panic
        let _norm = WelfordNormalizer::new(0.1, 1, None);
    }

    #[test]
    fn test_default_constructor() {
        let norm = WelfordNormalizer::default();
        assert_eq!(norm.alpha(), 0.05);
        assert_eq!(norm.count(), 0);
    }

    #[test]
    fn test_fast_slow_constructors() {
        let fast = WelfordNormalizer::fast();
        let slow = WelfordNormalizer::slow();

        assert!(fast.alpha() > slow.alpha());
    }

    #[test]
    fn test_set_alpha() {
        let mut norm = WelfordNormalizer::new(0.1, 30, None);

        norm.set_alpha(0.05);
        assert_eq!(norm.alpha(), 0.05);
    }

    #[test]
    #[should_panic]
    fn test_set_alpha_invalid() {
        let mut norm = WelfordNormalizer::new(0.1, 30, None);
        norm.set_alpha(0.0); // Should panic
    }
}
