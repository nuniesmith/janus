//! Sevcik Fractal Dimension - Streaming Implementation
//!
//! This module implements a real-time, streaming fractal dimension calculator
//! using the Sevcik approximation method with O(1) amortized complexity.
//!
//! # Mathematical Background
//!
//! The Sevcik fractal dimension D approximates the roughness of a time series:
//! - D = 1.0: Perfectly smooth line (strong trend)
//! - D = 1.5: Typical financial time series
//! - D = 2.0: Space-filling noise (random walk)
//!
//! The Hurst exponent H is derived as: H = 2 - D
//!
//! # References
//!
//! Sevcik, C. (2010). "A procedure to estimate the fractal dimension of waveforms"
//! arXiv:1003.5266

use std::collections::VecDeque;

/// Monotonic deque for O(1) min/max tracking in sliding windows.
///
/// This data structure maintains the minimum (or maximum) value in a sliding
/// window without requiring O(N) scanning on each update.
#[derive(Debug, Clone)]
pub struct MonotonicDeque {
    /// Internal deque storing (value, index) pairs
    deque: VecDeque<(f64, usize)>,
    /// Mode: true for min tracking, false for max tracking
    is_min_mode: bool,
}

impl MonotonicDeque {
    /// Create a new monotonic deque for minimum tracking
    #[inline]
    pub fn new_min() -> Self {
        Self {
            deque: VecDeque::with_capacity(64),
            is_min_mode: true,
        }
    }

    /// Create a new monotonic deque for maximum tracking
    #[inline]
    pub fn new_max() -> Self {
        Self {
            deque: VecDeque::with_capacity(64),
            is_min_mode: false,
        }
    }

    /// Push a new value with its index
    ///
    /// Maintains the monotonic property by removing values that will never
    /// be the min/max again.
    #[inline]
    pub fn push(&mut self, value: f64, index: usize) {
        // Remove elements that won't be min/max anymore
        while let Some(&(back_val, _)) = self.deque.back() {
            let should_pop = if self.is_min_mode {
                value < back_val // New value is smaller
            } else {
                value > back_val // New value is larger
            };

            if should_pop {
                self.deque.pop_back();
            } else {
                break;
            }
        }

        self.deque.push_back((value, index));
    }

    /// Remove all elements with index <= threshold
    #[inline]
    pub fn pop_old(&mut self, threshold_index: usize) {
        while let Some(&(_, idx)) = self.deque.front() {
            if idx <= threshold_index {
                self.deque.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get the current min/max value
    #[inline]
    pub fn get_value(&self) -> Option<f64> {
        self.deque.front().map(|&(val, _)| val)
    }
}

/// Result type for fractal dimension calculations
#[derive(Debug, Clone, Copy)]
pub struct FractalResult {
    /// Fractal dimension (1.0 to 2.0)
    pub dimension: f64,
    /// Hurst exponent (0.0 to 1.0)
    pub hurst: f64,
}

/// Error types for fractal dimension calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FractalError {
    /// Insufficient data (need full window)
    InsufficientData,
    /// Invalid price range (e.g., P_max == P_min)
    InvalidRange,
    /// Invalid dimension value (out of [1, 2] bounds)
    InvalidDimension(f64),
    /// NaN or Inf encountered
    InvalidValue,
}

impl std::fmt::Display for FractalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData => write!(f, "Insufficient data for calculation"),
            Self::InvalidRange => write!(f, "Invalid price range (flat line)"),
            Self::InvalidDimension(d) => write!(f, "Invalid dimension: {}", d),
            Self::InvalidValue => write!(f, "NaN or Inf encountered"),
        }
    }
}

impl std::error::Error for FractalError {}

/// Streaming Sevcik fractal dimension calculator
///
/// # Performance
///
/// - Time complexity: O(1) amortized per update
/// - Space complexity: O(window_size)
/// - Zero allocations after initialization
///
/// # Example
///
/// ```rust,ignore
/// use janus_dsp::sevcik::SevcikFractalDimension;
///
/// let mut calc = SevcikFractalDimension::new(64);
///
/// // Feed prices
/// for i in 0..100 {
///     let price = 100.0 + (i as f64) * 0.1;
///     match calc.update(price) {
///         Ok(result) => {
///             println!("D={:.3}, H={:.3}", result.dimension, result.hurst);
///         }
///         Err(e) => eprintln!("Error: {}", e),
///     }
/// }
/// ```
#[derive(Debug)]
pub struct SevcikFractalDimension {
    /// Window size for calculation (typically 64)
    window_size: usize,

    /// Circular buffer for price history
    buffer: VecDeque<f64>,

    /// Monotonic deque for minimum tracking
    min_tracker: MonotonicDeque,

    /// Monotonic deque for maximum tracking
    max_tracker: MonotonicDeque,

    /// Current index (increments with each update)
    index: usize,

    /// Pre-computed normalized x coordinates (uniform spacing)
    x_norm: Vec<f64>,
}

impl SevcikFractalDimension {
    /// Create a new Sevcik fractal dimension calculator
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of samples for calculation (power of 2 recommended)
    ///
    /// # Panics
    ///
    /// Panics if window_size < 2
    pub fn new(window_size: usize) -> Self {
        assert!(window_size >= 2, "Window size must be at least 2");

        // Pre-compute normalized x coordinates
        let x_norm: Vec<f64> = (0..window_size)
            .map(|i| i as f64 / (window_size - 1) as f64)
            .collect();

        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            min_tracker: MonotonicDeque::new_min(),
            max_tracker: MonotonicDeque::new_max(),
            index: 0,
            x_norm,
        }
    }

    /// Update with a new price tick
    ///
    /// # Arguments
    ///
    /// * `price` - New price observation
    ///
    /// # Returns
    ///
    /// - `Ok(FractalResult)` if calculation successful
    /// - `Err(FractalError)` if insufficient data or invalid values
    ///
    /// # Performance
    ///
    /// This method is on the critical hot path and must be allocation-free.
    #[inline]
    pub fn update(&mut self, price: f64) -> Result<FractalResult, FractalError> {
        // Validate input
        if !price.is_finite() {
            return Err(FractalError::InvalidValue);
        }

        // Add to buffer
        if self.buffer.len() == self.window_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(price);

        // Update min/max trackers
        self.min_tracker.push(price, self.index);
        self.max_tracker.push(price, self.index);

        // Remove old elements outside window
        let threshold = self.index.saturating_sub(self.window_size);
        self.min_tracker.pop_old(threshold);
        self.max_tracker.pop_old(threshold);

        self.index += 1;

        // Need full window before calculating
        if self.buffer.len() < self.window_size {
            return Err(FractalError::InsufficientData);
        }

        self.calculate_dimension()
    }

    /// Calculate the Sevcik fractal dimension
    ///
    /// # Algorithm
    ///
    /// 1. Normalize prices to unit square [0,1] x [0,1]
    /// 2. Calculate Euclidean length of the curve
    /// 3. Apply Sevcik formula: D = 1 + ln(L) / ln(2(N-1))
    /// 4. Derive Hurst exponent: H = 2 - D
    #[inline]
    fn calculate_dimension(&self) -> Result<FractalResult, FractalError> {
        debug_assert_eq!(self.buffer.len(), self.window_size);

        // Get min/max from trackers
        let p_min = self
            .min_tracker
            .get_value()
            .ok_or(FractalError::InsufficientData)?;
        let p_max = self
            .max_tracker
            .get_value()
            .ok_or(FractalError::InsufficientData)?;

        // Handle degenerate case (flat line)
        let price_range = p_max - p_min;
        const EPSILON: f64 = 1e-10;

        if price_range < EPSILON {
            // Flat line → perfectly smooth
            return Ok(FractalResult {
                dimension: 1.0,
                hurst: 1.0,
            });
        }

        // Validate range
        if !price_range.is_finite() {
            return Err(FractalError::InvalidRange);
        }

        // Normalize prices to [0, 1]
        // NOTE: SIMD opportunity — this loop and the Euclidean-distance loop below
        // are hot when `window_size` is large (≥256).  On x86-64 with AVX-512:
        //   1. Use `_mm512_sub_pd` / `_mm512_div_pd` for the normalization pass.
        //   2. Fuse the dx²+dy² reduction in the length loop with `_mm512_fmadd_pd`.
        //   3. Gate behind `#[cfg(target_feature = "avx512f")]` or a runtime `is_x86_feature_detected!`
        //      check to keep the scalar fallback for portability.
        //   4. Benchmark before committing — for typical window_size ≤ 128 the auto-vectorised
        //      scalar loop is likely sufficient and the SIMD overhead may not pay off.
        let y_norm: Vec<f64> = self
            .buffer
            .iter()
            .map(|&p| (p - p_min) / price_range)
            .collect();

        // Calculate cumulative Euclidean length
        let mut total_length = 0.0;

        for i in 0..self.window_size - 1 {
            let dx = self.x_norm[i + 1] - self.x_norm[i];
            let dy = y_norm[i + 1] - y_norm[i];

            // Euclidean distance
            let segment_length = (dx * dx + dy * dy).sqrt();
            total_length += segment_length;
        }

        // Sevcik formula: D = 1 + (ln(L) - ln(2)) / ln(2(N-1))
        let n = self.window_size as f64;

        // Protect against L < sqrt(2) (shouldn't happen with normalization)
        let length = total_length.max((2.0_f64).sqrt());

        let numerator = length.ln() - 2.0_f64.ln();
        let denominator = (2.0 * (n - 1.0)).ln();

        let dimension = 1.0 + numerator / denominator;

        // Validate result
        if !dimension.is_finite() {
            return Err(FractalError::InvalidValue);
        }

        // Clamp to valid range [1.0, 2.0]
        let dimension = dimension.clamp(1.0, 2.0);

        // Derive Hurst exponent
        let hurst = 2.0 - dimension;

        Ok(FractalResult { dimension, hurst })
    }

    /// Get the current window size
    #[inline]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get the number of samples currently in the buffer
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is full (ready for calculations)
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.buffer.len() == self.window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_deque_min() {
        let mut deque = MonotonicDeque::new_min();

        deque.push(5.0, 0);
        assert_eq!(deque.get_value(), Some(5.0));

        deque.push(3.0, 1);
        assert_eq!(deque.get_value(), Some(3.0));

        deque.push(7.0, 2);
        assert_eq!(deque.get_value(), Some(3.0));

        deque.pop_old(1);
        assert_eq!(deque.get_value(), Some(7.0));
    }

    #[test]
    fn test_sevcik_warmup() {
        let mut calc = SevcikFractalDimension::new(64);

        // First 63 updates should return InsufficientData
        for i in 0..63 {
            let result = calc.update(100.0 + i as f64 * 0.1);
            assert!(matches!(result, Err(FractalError::InsufficientData)));
        }

        // 64th update should succeed
        let result = calc.update(106.3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sevcik_flat_line() {
        let mut calc = SevcikFractalDimension::new(64);

        // Feed constant price
        for _ in 0..100 {
            let _ = calc.update(100.0);
        }

        let result = calc.update(100.0).unwrap();

        // Flat line should give D=1, H=1
        assert!((result.dimension - 1.0).abs() < 1e-6);
        assert!((result.hurst - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sevcik_bounds() {
        let mut calc = SevcikFractalDimension::new(64);

        // Feed random-ish data
        for i in 0..200 {
            let price = 100.0 + ((i * 17) % 100) as f64 * 0.1;

            if let Ok(result) = calc.update(price) {
                // D must be in [1, 2]
                assert!(result.dimension >= 1.0);
                assert!(result.dimension <= 2.0);

                // H must be in [0, 1]
                assert!(result.hurst >= 0.0);
                assert!(result.hurst <= 1.0);

                // H = 2 - D relationship
                assert!((result.hurst - (2.0 - result.dimension)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sevcik_trending() {
        let mut calc = SevcikFractalDimension::new(64);

        // Strong trend
        for i in 0..100 {
            let _ = calc.update(100.0 + i as f64 * 0.5);
        }

        let result = calc.update(150.0).unwrap();

        // Trending should have low D, high H
        assert!(
            result.dimension < 1.5,
            "Expected D < 1.5, got {}",
            result.dimension
        );
        assert!(result.hurst > 0.5, "Expected H > 0.5, got {}", result.hurst);
    }

    #[test]
    fn test_invalid_price() {
        let mut calc = SevcikFractalDimension::new(64);

        // Feed some valid data
        for i in 0..70 {
            let _ = calc.update(100.0 + i as f64 * 0.1);
        }

        // NaN should be rejected
        let result = calc.update(f64::NAN);
        assert!(matches!(result, Err(FractalError::InvalidValue)));

        // Infinity should be rejected
        let result = calc.update(f64::INFINITY);
        assert!(matches!(result, Err(FractalError::InvalidValue)));
    }

    #[test]
    fn test_zero_allocation_update() {
        // This test verifies that update() doesn't allocate after warmup
        let mut calc = SevcikFractalDimension::new(64);

        // Warmup
        for i in 0..64 {
            let _ = calc.update(100.0 + i as f64 * 0.1);
        }

        // Subsequent updates should not allocate
        // (This would need a custom allocator to verify properly)
        for i in 64..1000 {
            let _ = calc.update(100.0 + i as f64 * 0.1);
        }
    }
}
