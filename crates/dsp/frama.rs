//! Fractal Adaptive Moving Average (FRAMA)
//!
//! This module implements the FRAMA algorithm, which adapts its smoothing
//! factor based on the fractal dimension of the market.
//!
//! # Algorithm
//!
//! FRAMA adjusts its alpha parameter based on fractal dimension:
//! - High D (noisy) → Low alpha → Heavy smoothing
//! - Low D (trending) → High alpha → Fast response
//!
//! The adaptation formula is: α = exp(-4.6 * (D - 1))
//! With clamping to [alpha_min, alpha_max] to prevent extremes.
//!
//! # Performance
//!
//! - Time complexity: O(1) per update
//! - Space complexity: O(1)
//! - Zero allocations on hot path

use crate::sevcik::{FractalError, FractalResult, SevcikFractalDimension};

/// Market regime classification based on Hurst exponent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Strong trend (H > 0.6)
    Trending,
    /// Random walk (0.4 <= H <= 0.6)
    RandomWalk,
    /// Mean reverting (H < 0.4)
    MeanReverting,
    /// Unknown (insufficient data)
    Unknown,
}

impl MarketRegime {
    /// Classify regime from Hurst exponent
    #[inline]
    pub fn from_hurst(hurst: f64) -> Self {
        if !hurst.is_finite() {
            return Self::Unknown;
        }

        if hurst > 0.6 {
            Self::Trending
        } else if hurst < 0.4 {
            Self::MeanReverting
        } else {
            Self::RandomWalk
        }
    }
}

/// FRAMA diagnostic information
#[derive(Debug, Clone, Copy)]
pub struct FramaDiagnostics {
    /// Current FRAMA value
    pub frama: f64,
    /// Divergence (price - FRAMA)
    pub divergence: f64,
    /// Current alpha (smoothing factor)
    pub alpha: f64,
    /// Fractal dimension
    pub fractal_dim: f64,
    /// Hurst exponent
    pub hurst: f64,
    /// Market regime classification
    pub regime: MarketRegime,
}

/// Fractal Adaptive Moving Average calculator
///
/// # Example
///
/// ```rust,ignore
/// use janus_dsp::frama::Frama;
///
/// let mut frama = Frama::new(64, 0.01, 0.5, false);
///
/// // Feed prices
/// for i in 0..100 {
///     let price = 100.0 + (i as f64) * 0.1;
///
///     if let Ok(diag) = frama.update(price) {
///         println!("FRAMA: {:.2}, Alpha: {:.3}, Regime: {:?}",
///                  diag.frama, diag.alpha, diag.regime);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct Frama {
    /// Fractal dimension calculator
    fractal_calc: SevcikFractalDimension,

    /// Minimum alpha (maximum smoothing)
    alpha_min: f64,

    /// Maximum alpha (minimum smoothing)
    alpha_max: f64,

    /// Current FRAMA value
    frama_value: Option<f64>,

    /// Last computed alpha
    last_alpha: Option<f64>,

    /// Use Ehlers Super Smoother post-processing
    use_super_smoother: bool,

    /// Super Smoother coefficients (if enabled)
    ss_c1: f64,
    ss_c2: f64,
    ss_c3: f64,

    /// Super Smoother state
    ss_y1: Option<f64>, // y[t-1]
    ss_y2: Option<f64>, // y[t-2]
    ss_x1: Option<f64>, // x[t-1]
}

impl Frama {
    /// Create a new FRAMA calculator
    ///
    /// # Arguments
    ///
    /// * `window_size` - Window for fractal dimension calculation (typically 64)
    /// * `alpha_min` - Minimum smoothing factor (0.01 = heavy smoothing)
    /// * `alpha_max` - Maximum smoothing factor (0.5 = light smoothing)
    /// * `use_super_smoother` - Enable Ehlers Super Smoother post-processing
    ///
    /// # Panics
    ///
    /// Panics if alpha_min >= alpha_max or either is outside [0, 1]
    pub fn new(
        window_size: usize,
        alpha_min: f64,
        alpha_max: f64,
        use_super_smoother: bool,
    ) -> Self {
        assert!(
            0.0 < alpha_min && alpha_min < alpha_max && alpha_max <= 1.0,
            "Invalid alpha range: [{}, {}]",
            alpha_min,
            alpha_max
        );

        let fractal_calc = SevcikFractalDimension::new(window_size);

        // Initialize Super Smoother coefficients (if enabled)
        let (ss_c1, ss_c2, ss_c3) = if use_super_smoother {
            Self::init_super_smoother_coefficients(10.0)
        } else {
            (0.0, 0.0, 0.0)
        };

        Self {
            fractal_calc,
            alpha_min,
            alpha_max,
            frama_value: None,
            last_alpha: None,
            use_super_smoother,
            ss_c1,
            ss_c2,
            ss_c3,
            ss_y1: None,
            ss_y2: None,
            ss_x1: None,
        }
    }

    /// Initialize Ehlers Super Smoother coefficients
    ///
    /// This is a 2-pole Butterworth filter
    #[inline]
    fn init_super_smoother_coefficients(period: f64) -> (f64, f64, f64) {
        use std::f64::consts::PI;

        let a1 = (-std::f64::consts::SQRT_2 * PI / period).exp();

        let c2 = 2.0 * a1 * (std::f64::consts::SQRT_2 * PI / period).cos();
        let c3 = -a1 * a1;
        let c1 = (1.0 - c2 - c3) / 2.0;

        (c1, c2, c3)
    }

    /// Update FRAMA with new price
    ///
    /// # Arguments
    ///
    /// * `price` - New price observation
    ///
    /// # Returns
    ///
    /// - `Ok(FramaDiagnostics)` with complete diagnostic information
    /// - `Err(FractalError)` if insufficient data or invalid values
    ///
    /// # Performance
    ///
    /// This method is on the critical hot path. It must be:
    /// - Allocation-free
    /// - Branchless where possible (for SIMD)
    /// - Cache-friendly
    #[inline]
    pub fn update(&mut self, price: f64) -> Result<FramaDiagnostics, FractalError> {
        // Validate input
        if !price.is_finite() {
            return Err(FractalError::InvalidValue);
        }

        // Update fractal dimension
        let fractal_result = self.fractal_calc.update(price);

        // Handle initialization
        if self.frama_value.is_none() {
            if let Ok(result) = fractal_result {
                // Initialize FRAMA to current price
                self.frama_value = Some(price);

                if self.use_super_smoother {
                    self.ss_y1 = Some(price);
                    self.ss_y2 = Some(price);
                    self.ss_x1 = Some(price);
                }

                return self.make_diagnostics(price, result);
            } else {
                // Still warming up
                return Err(fractal_result.unwrap_err());
            }
        }

        // Compute alpha based on fractal dimension
        let (alpha, fractal_result) = match fractal_result {
            Ok(result) => {
                // Calculate raw alpha from fractal dimension
                let alpha_raw = (-4.6_f64 * (result.dimension - 1.0)).exp();

                // Clamp to prevent extreme behaviors
                let alpha = alpha_raw.clamp(self.alpha_min, self.alpha_max);

                self.last_alpha = Some(alpha);

                (alpha, result)
            }
            Err(e) => {
                // Use last known alpha or default
                let _alpha = self.last_alpha.unwrap_or(0.1);

                // Return error with partial diagnostics
                return Err(e);
            }
        };

        // FRAMA update: y[t] = α * x[t] + (1 - α) * y[t-1]
        let prev_frama = self.frama_value.unwrap();
        let new_frama = alpha * price + (1.0 - alpha) * prev_frama;

        self.frama_value = Some(new_frama);

        // Apply Super Smoother if enabled
        let output = if self.use_super_smoother {
            self.apply_super_smoother(new_frama)
        } else {
            new_frama
        };

        // Create diagnostics
        let mut diag = self.make_diagnostics(price, fractal_result)?;
        diag.frama = output;
        diag.divergence = price - output;

        Ok(diag)
    }

    /// Apply Ehlers Super Smoother filter
    ///
    /// Formula: y[t] = c1 * (x[t] + x[t-1]) + c2 * y[t-1] + c3 * y[t-2]
    #[inline]
    fn apply_super_smoother(&mut self, input: f64) -> f64 {
        let x1 = self.ss_x1.unwrap_or(input);
        let y1 = self.ss_y1.unwrap_or(input);
        let y2 = self.ss_y2.unwrap_or(input);

        let output = self.ss_c1 * (input + x1) + self.ss_c2 * y1 + self.ss_c3 * y2;

        // Update state
        self.ss_y2 = self.ss_y1;
        self.ss_y1 = Some(output);
        self.ss_x1 = Some(input);

        output
    }

    /// Create diagnostic information
    #[inline]
    fn make_diagnostics(
        &self,
        price: f64,
        fractal_result: FractalResult,
    ) -> Result<FramaDiagnostics, FractalError> {
        let frama = self.frama_value.ok_or(FractalError::InsufficientData)?;
        let alpha = self.last_alpha.unwrap_or(self.alpha_min);

        Ok(FramaDiagnostics {
            frama,
            divergence: price - frama,
            alpha,
            fractal_dim: fractal_result.dimension,
            hurst: fractal_result.hurst,
            regime: MarketRegime::from_hurst(fractal_result.hurst),
        })
    }

    /// Get current FRAMA value
    #[inline]
    pub fn value(&self) -> Option<f64> {
        self.frama_value
    }

    /// Get current alpha
    #[inline]
    pub fn alpha(&self) -> Option<f64> {
        self.last_alpha
    }

    /// Check if FRAMA is initialized
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.frama_value.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frama_initialization() {
        let mut frama = Frama::new(64, 0.01, 0.5, false);

        // First 63 updates should fail (warmup)
        for i in 0..63 {
            let result = frama.update(100.0 + i as f64 * 0.1);
            assert!(result.is_err());
        }

        // 64th update should succeed
        let result = frama.update(106.3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_frama_tracking() {
        let mut frama = Frama::new(64, 0.01, 0.5, false);

        // Feed trending data
        for i in 0..200 {
            let price = 100.0 + i as f64 * 0.1;
            let _ = frama.update(price);
        }

        let result = frama.update(120.0).unwrap();

        // FRAMA should lag behind price in uptrend
        assert!(result.frama < 120.0);

        // Divergence should be positive
        assert!(result.divergence > 0.0);
    }

    #[test]
    fn test_alpha_clamping() {
        let mut frama = Frama::new(64, 0.01, 0.5, false);

        // Generate various market conditions
        for i in 0..200 {
            let price = 100.0 + (i as f64 * 0.1);

            if let Ok(diag) = frama.update(price) {
                // Alpha must be within bounds
                assert!(diag.alpha >= 0.01);
                assert!(diag.alpha <= 0.5);
            }
        }
    }

    #[test]
    fn test_regime_classification() {
        let mut frama = Frama::new(64, 0.01, 0.5, false);

        // Strong trend
        for i in 0..100 {
            let _ = frama.update(100.0 + i as f64 * 0.5);
        }

        let result = frama.update(150.0).unwrap();

        // Should classify as trending (H > 0.6) or random walk
        assert!(
            matches!(
                result.regime,
                MarketRegime::Trending | MarketRegime::RandomWalk
            ),
            "Expected trending or random, got {:?}",
            result.regime
        );
    }

    #[test]
    fn test_super_smoother() {
        let mut frama_basic = Frama::new(64, 0.01, 0.5, false);
        let mut frama_smooth = Frama::new(64, 0.01, 0.5, true);

        for i in 0..200 {
            let price = 100.0 + i as f64 * 0.1 + ((i * 17) % 10) as f64 * 0.01;

            let _ = frama_basic.update(price);
            let _ = frama_smooth.update(price);
        }

        // Both should have valid values
        assert!(frama_basic.value().is_some());
        assert!(frama_smooth.value().is_some());
    }

    #[test]
    fn test_invalid_price() {
        let mut frama = Frama::new(64, 0.01, 0.5, false);

        // Warmup
        for i in 0..70 {
            let _ = frama.update(100.0 + i as f64 * 0.1);
        }

        // NaN should be rejected
        let result = frama.update(f64::NAN);
        assert!(matches!(result, Err(FractalError::InvalidValue)));

        // Infinity should be rejected
        let result = frama.update(f64::INFINITY);
        assert!(matches!(result, Err(FractalError::InvalidValue)));
    }

    #[test]
    #[should_panic]
    fn test_invalid_alpha_range() {
        // alpha_min >= alpha_max should panic
        let _frama = Frama::new(64, 0.5, 0.01, false);
    }
}
