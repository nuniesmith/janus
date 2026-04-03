//! Noise filtering
//!
//! Part of the Thalamus region - filters out noise and irrelevant signals
//! to improve signal-to-noise ratio in market data processing.

use crate::common::Result;
use std::collections::VecDeque;

/// Filter type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Simple moving average filter
    MovingAverage,
    /// Exponential moving average filter
    ExponentialMovingAverage,
    /// Median filter for outlier removal
    Median,
    /// Kalman filter for state estimation
    Kalman,
    /// Low-pass filter for high-frequency noise
    LowPass,
    /// High-pass filter for trend removal
    HighPass,
    /// Band-pass filter for specific frequency range
    BandPass,
    /// Adaptive filter that adjusts based on signal characteristics
    Adaptive,
}

impl Default for FilterType {
    fn default() -> Self {
        Self::ExponentialMovingAverage
    }
}

/// Filter configuration
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Filter type to use
    pub filter_type: FilterType,
    /// Window size for windowed filters
    pub window_size: usize,
    /// Smoothing factor for EMA (alpha)
    pub alpha: f64,
    /// Cutoff frequency for frequency-domain filters
    pub cutoff_frequency: f64,
    /// Minimum samples before filtering is active
    pub min_samples: usize,
    /// Enable adaptive adjustment
    pub adaptive: bool,
    /// Noise threshold for adaptive filtering
    pub noise_threshold: f64,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            filter_type: FilterType::ExponentialMovingAverage,
            window_size: 20,
            alpha: 0.1,
            cutoff_frequency: 0.5,
            min_samples: 5,
            adaptive: true,
            noise_threshold: 2.0, // Standard deviations
        }
    }
}

/// Kalman filter state
#[derive(Debug, Clone, Default)]
pub struct KalmanState {
    /// State estimate
    pub estimate: f64,
    /// Error covariance
    pub error_covariance: f64,
    /// Process noise
    pub process_noise: f64,
    /// Measurement noise
    pub measurement_noise: f64,
    /// Kalman gain
    pub gain: f64,
    /// Whether initialized
    pub initialized: bool,
}

impl KalmanState {
    /// Create a new Kalman state
    pub fn new(process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            estimate: 0.0,
            error_covariance: 1.0,
            process_noise,
            measurement_noise,
            gain: 0.0,
            initialized: false,
        }
    }

    /// Update with new measurement
    pub fn update(&mut self, measurement: f64) -> f64 {
        if !self.initialized {
            self.estimate = measurement;
            self.initialized = true;
            return measurement;
        }

        // Predict
        let predicted_estimate = self.estimate;
        let predicted_covariance = self.error_covariance + self.process_noise;

        // Update
        self.gain = predicted_covariance / (predicted_covariance + self.measurement_noise);
        self.estimate = predicted_estimate + self.gain * (measurement - predicted_estimate);
        self.error_covariance = (1.0 - self.gain) * predicted_covariance;

        self.estimate
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.estimate = 0.0;
        self.error_covariance = 1.0;
        self.gain = 0.0;
        self.initialized = false;
    }
}

/// Filter statistics
#[derive(Debug, Clone, Default)]
pub struct FilterStats {
    /// Total samples processed
    pub samples_processed: u64,
    /// Samples filtered out
    pub samples_filtered: u64,
    /// Current noise estimate
    pub noise_estimate: f64,
    /// Signal-to-noise ratio estimate
    pub snr_estimate: f64,
    /// Average filter latency in microseconds
    pub avg_latency_us: f64,
    /// Outliers detected
    pub outliers_detected: u64,
}

/// Filtered signal result
#[derive(Debug, Clone)]
pub struct FilteredSignal {
    /// Original value
    pub original: f64,
    /// Filtered value
    pub filtered: f64,
    /// Whether signal was flagged as outlier
    pub is_outlier: bool,
    /// Confidence in filtered value (0.0 - 1.0)
    pub confidence: f64,
    /// Noise estimate at this point
    pub noise_estimate: f64,
}

impl FilteredSignal {
    /// Get the absolute difference from filtering
    pub fn difference(&self) -> f64 {
        (self.original - self.filtered).abs()
    }

    /// Get relative difference as percentage
    pub fn relative_difference(&self) -> f64 {
        if self.original == 0.0 {
            return 0.0;
        }
        (self.difference() / self.original.abs()) * 100.0
    }
}

/// Noise filtering system
pub struct Filter {
    /// Configuration
    config: FilterConfig,
    /// Sample buffer for windowed operations
    buffer: VecDeque<f64>,
    /// Kalman filter state
    kalman_state: KalmanState,
    /// EMA state
    ema_value: Option<f64>,
    /// Running statistics for noise estimation
    mean: f64,
    variance: f64,
    sample_count: u64,
    /// Statistics
    stats: FilterStats,
}

impl Default for Filter {
    fn default() -> Self {
        Self::new()
    }
}

impl Filter {
    /// Create a new filter with default configuration
    pub fn new() -> Self {
        Self::with_config(FilterConfig::default())
    }

    /// Create a new filter with custom configuration
    pub fn with_config(config: FilterConfig) -> Self {
        Self {
            kalman_state: KalmanState::new(0.01, 0.1),
            buffer: VecDeque::with_capacity(config.window_size),
            ema_value: None,
            mean: 0.0,
            variance: 0.0,
            sample_count: 0,
            stats: FilterStats::default(),
            config,
        }
    }

    /// Set filter type
    pub fn with_filter_type(mut self, filter_type: FilterType) -> Self {
        self.config.filter_type = filter_type;
        self
    }

    /// Set window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self.buffer = VecDeque::with_capacity(size);
        self
    }

    /// Set EMA alpha
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Filter a single value
    pub fn filter(&mut self, value: f64) -> FilteredSignal {
        let start = std::time::Instant::now();

        // Update running statistics
        self.update_statistics(value);

        // Add to buffer
        self.buffer.push_back(value);
        if self.buffer.len() > self.config.window_size {
            self.buffer.pop_front();
        }

        // Check for outlier
        let is_outlier = self.is_outlier(value);
        if is_outlier {
            self.stats.outliers_detected += 1;
        }

        // Apply appropriate filter
        let filtered = match self.config.filter_type {
            FilterType::MovingAverage => self.apply_moving_average(),
            FilterType::ExponentialMovingAverage => self.apply_ema(value),
            FilterType::Median => self.apply_median(),
            FilterType::Kalman => self.apply_kalman(value),
            FilterType::LowPass => self.apply_low_pass(value),
            FilterType::HighPass => self.apply_high_pass(value),
            FilterType::BandPass => self.apply_band_pass(value),
            FilterType::Adaptive => self.apply_adaptive(value, is_outlier),
        };

        // Calculate confidence based on noise and sample count
        let confidence = self.calculate_confidence();

        // Update statistics
        self.stats.samples_processed += 1;
        if is_outlier {
            self.stats.samples_filtered += 1;
        }

        let elapsed = start.elapsed().as_micros() as f64;
        self.stats.avg_latency_us = self.stats.avg_latency_us * 0.9 + elapsed * 0.1;

        FilteredSignal {
            original: value,
            filtered,
            is_outlier,
            confidence,
            noise_estimate: self.stats.noise_estimate,
        }
    }

    /// Filter a batch of values
    pub fn filter_batch(&mut self, values: &[f64]) -> Vec<FilteredSignal> {
        values.iter().map(|&v| self.filter(v)).collect()
    }

    /// Update running statistics using Welford's algorithm
    fn update_statistics(&mut self, value: f64) {
        self.sample_count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.sample_count as f64;
        let delta2 = value - self.mean;
        self.variance += delta * delta2;

        // Update noise estimate
        if self.sample_count > 1 {
            let std_dev = (self.variance / (self.sample_count - 1) as f64).sqrt();
            self.stats.noise_estimate = std_dev;

            // Estimate SNR
            if std_dev > 0.0 {
                self.stats.snr_estimate = self.mean.abs() / std_dev;
            }
        }
    }

    /// Check if a value is an outlier using z-score
    fn is_outlier(&self, value: f64) -> bool {
        if self.sample_count < self.config.min_samples as u64 {
            return false;
        }

        let std_dev = self.stats.noise_estimate;
        if std_dev == 0.0 {
            return false;
        }

        let z_score = (value - self.mean).abs() / std_dev;
        z_score > self.config.noise_threshold
    }

    /// Apply simple moving average filter
    fn apply_moving_average(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.buffer.iter().sum();
        sum / self.buffer.len() as f64
    }

    /// Apply exponential moving average filter
    fn apply_ema(&mut self, value: f64) -> f64 {
        match self.ema_value {
            Some(ema) => {
                let new_ema = self.config.alpha * value + (1.0 - self.config.alpha) * ema;
                self.ema_value = Some(new_ema);
                new_ema
            }
            None => {
                self.ema_value = Some(value);
                value
            }
        }
    }

    /// Apply median filter
    fn apply_median(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = self.buffer.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Apply Kalman filter
    fn apply_kalman(&mut self, value: f64) -> f64 {
        self.kalman_state.update(value)
    }

    /// Apply low-pass filter (simple first-order IIR)
    fn apply_low_pass(&mut self, value: f64) -> f64 {
        // Use EMA as a simple low-pass filter
        let alpha = self.config.cutoff_frequency.min(1.0);
        match self.ema_value {
            Some(prev) => {
                let filtered = alpha * value + (1.0 - alpha) * prev;
                self.ema_value = Some(filtered);
                filtered
            }
            None => {
                self.ema_value = Some(value);
                value
            }
        }
    }

    /// Apply high-pass filter
    fn apply_high_pass(&mut self, value: f64) -> f64 {
        let low_pass = self.apply_low_pass(value);
        value - low_pass + self.mean // Add back mean to avoid negative values
    }

    /// Apply band-pass filter
    fn apply_band_pass(&mut self, value: f64) -> f64 {
        // High-pass then low-pass
        let high_passed = value - self.mean;
        let alpha = self.config.cutoff_frequency;
        alpha * high_passed
    }

    /// Apply adaptive filter
    fn apply_adaptive(&mut self, value: f64, is_outlier: bool) -> f64 {
        // Adjust alpha based on whether signal is outlier
        let adaptive_alpha = if is_outlier {
            self.config.alpha * 0.1 // Reduce response to outliers
        } else {
            self.config.alpha
        };

        match self.ema_value {
            Some(ema) => {
                let new_ema = adaptive_alpha * value + (1.0 - adaptive_alpha) * ema;
                self.ema_value = Some(new_ema);
                new_ema
            }
            None => {
                self.ema_value = Some(value);
                value
            }
        }
    }

    /// Calculate confidence in filtered value
    fn calculate_confidence(&self) -> f64 {
        // More samples = higher confidence
        let sample_confidence =
            (self.sample_count as f64 / self.config.min_samples as f64).min(1.0);

        // Lower noise = higher confidence
        let noise_confidence = if self.stats.noise_estimate > 0.0 {
            (1.0 / (1.0 + self.stats.noise_estimate)).min(1.0)
        } else {
            0.5
        };

        // Buffer fullness
        let buffer_confidence = self.buffer.len() as f64 / self.config.window_size as f64;

        (sample_confidence + noise_confidence + buffer_confidence) / 3.0
    }

    /// Get current filter statistics
    pub fn stats(&self) -> &FilterStats {
        &self.stats
    }

    /// Get current noise estimate
    pub fn noise_estimate(&self) -> f64 {
        self.stats.noise_estimate
    }

    /// Get current SNR estimate
    pub fn snr_estimate(&self) -> f64 {
        self.stats.snr_estimate
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current standard deviation
    pub fn std_dev(&self) -> f64 {
        if self.sample_count > 1 {
            (self.variance / (self.sample_count - 1) as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.kalman_state.reset();
        self.ema_value = None;
        self.mean = 0.0;
        self.variance = 0.0;
        self.sample_count = 0;
        self.stats = FilterStats::default();
    }

    /// Get filter configuration
    pub fn config(&self) -> &FilterConfig {
        &self.config
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_creation() {
        let filter = Filter::new();
        assert_eq!(
            filter.config.filter_type,
            FilterType::ExponentialMovingAverage
        );
    }

    #[test]
    fn test_ema_filter() {
        let mut filter = Filter::new().with_filter_type(FilterType::ExponentialMovingAverage);

        let result = filter.filter(100.0);
        assert_eq!(result.filtered, 100.0); // First value passes through

        let result = filter.filter(110.0);
        // EMA should be closer to 100 than 110 with default alpha
        assert!(result.filtered > 100.0 && result.filtered < 110.0);
    }

    #[test]
    fn test_moving_average_filter() {
        let mut filter = Filter::new()
            .with_filter_type(FilterType::MovingAverage)
            .with_window_size(3);

        filter.filter(10.0);
        filter.filter(20.0);
        let result = filter.filter(30.0);

        assert!((result.filtered - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_median_filter() {
        let mut filter = Filter::new()
            .with_filter_type(FilterType::Median)
            .with_window_size(5);

        // Add values including an outlier
        filter.filter(10.0);
        filter.filter(12.0);
        filter.filter(100.0); // Outlier
        filter.filter(11.0);
        let result = filter.filter(13.0);

        // Median should be robust to outlier
        assert!(result.filtered < 20.0);
    }

    #[test]
    fn test_kalman_filter() {
        let mut filter = Filter::new().with_filter_type(FilterType::Kalman);

        // Filter should smooth noisy signal
        let values = vec![10.0, 12.0, 9.0, 11.0, 10.5];
        let mut results: Vec<f64> = Vec::new();

        for v in values {
            let result = filter.filter(v);
            results.push(result.filtered);
        }

        // Kalman filtered values should be less variable
        let input_variance: f64 = [10.0_f64, 12.0, 9.0, 11.0, 10.5]
            .iter()
            .map(|&v| (v - 10.5_f64).powi(2))
            .sum::<f64>()
            / 5.0;

        let output_variance: f64 = results
            .iter()
            .map(|&v| (v - results.iter().sum::<f64>() / results.len() as f64).powi(2))
            .sum::<f64>()
            / results.len() as f64;

        // Output should have lower or similar variance
        assert!(output_variance <= input_variance * 1.5);
    }

    #[test]
    fn test_outlier_detection() {
        let mut filter = Filter::new();

        // Build up statistics
        for _ in 0..20 {
            filter.filter(100.0);
        }

        // Add an outlier
        let result = filter.filter(500.0);
        assert!(result.is_outlier);
    }

    #[test]
    fn test_adaptive_filter() {
        let mut filter = Filter::new().with_filter_type(FilterType::Adaptive);

        // Normal values
        for _ in 0..10 {
            filter.filter(100.0);
        }

        let _normal_result = filter.filter(101.0);
        let _ema_after_normal = filter.ema_value.unwrap();

        // Reset and test outlier handling
        let mut filter2 = Filter::new().with_filter_type(FilterType::Adaptive);

        for _ in 0..10 {
            filter2.filter(100.0);
        }

        let outlier_result = filter2.filter(500.0);

        // Adaptive filter should respond less to outlier
        assert!(outlier_result.filtered < 200.0); // Should be dampened
    }

    #[test]
    fn test_filter_batch() {
        let mut filter = Filter::new();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let results = filter.filter_batch(&values);

        assert_eq!(results.len(), 5);
        assert_eq!(filter.stats().samples_processed, 5);
    }

    #[test]
    fn test_statistics_tracking() {
        let mut filter = Filter::new();

        for i in 0..100 {
            filter.filter(i as f64);
        }

        let stats = filter.stats();
        assert_eq!(stats.samples_processed, 100);
        assert!(filter.mean() > 0.0);
        assert!(filter.std_dev() > 0.0);
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = Filter::new();

        for i in 0..50 {
            filter.filter(i as f64);
        }

        filter.reset();

        assert_eq!(filter.stats().samples_processed, 0);
        assert_eq!(filter.mean(), 0.0);
        assert!(filter.ema_value.is_none());
    }

    #[test]
    fn test_filtered_signal_helpers() {
        let signal = FilteredSignal {
            original: 100.0,
            filtered: 95.0,
            is_outlier: false,
            confidence: 0.9,
            noise_estimate: 2.0,
        };

        assert!((signal.difference() - 5.0).abs() < 0.001);
        assert!((signal.relative_difference() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_kalman_state() {
        let mut kalman = KalmanState::new(0.01, 0.1);

        let first = kalman.update(10.0);
        assert_eq!(first, 10.0);

        let second = kalman.update(12.0);
        assert!(second > 10.0 && second < 12.0);

        kalman.reset();
        assert!(!kalman.initialized);
    }

    #[test]
    fn test_process_compatibility() {
        let filter = Filter::new();
        assert!(filter.process().is_ok());
    }
}
