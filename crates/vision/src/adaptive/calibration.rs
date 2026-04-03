//! Confidence Calibration
//!
//! This module provides methods for calibrating model confidence scores to
//! actual probabilities using techniques like Platt scaling and isotonic regression.

use std::collections::VecDeque;

/// Configuration for confidence calibration
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of bins for calibration curve
    pub num_bins: usize,
    /// Minimum samples per bin
    pub min_samples_per_bin: usize,
    /// Lookback window for calibration data
    pub lookback_window: usize,
    /// Regularization parameter for Platt scaling
    pub regularization: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_bins: 10,
            min_samples_per_bin: 10,
            lookback_window: 1000,
            regularization: 0.01,
        }
    }
}

/// Calibration sample
#[derive(Debug, Clone)]
struct CalibrationSample {
    predicted_confidence: f64,
    actual_outcome: bool,
}

/// Platt scaling calibrator (logistic regression)
pub struct PlattScaling {
    /// Logistic regression parameter A
    a: f64,
    /// Logistic regression parameter B
    b: f64,
    /// Training samples
    samples: VecDeque<CalibrationSample>,
    config: CalibrationConfig,
    is_fitted: bool,
}

impl PlattScaling {
    /// Create a new Platt scaling calibrator
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            samples: VecDeque::with_capacity(config.lookback_window),
            config,
            is_fitted: false,
        }
    }

    /// Add a calibration sample
    pub fn add_sample(&mut self, predicted_confidence: f64, actual_outcome: bool) {
        self.samples.push_back(CalibrationSample {
            predicted_confidence,
            actual_outcome,
        });

        while self.samples.len() > self.config.lookback_window {
            self.samples.pop_front();
        }

        // Refit if we have enough samples
        if self.samples.len() >= self.config.min_samples_per_bin * 2 {
            self.fit();
        }
    }

    /// Fit the Platt scaling parameters
    fn fit(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        // Initialize parameters
        let mut a = 0.0;
        let mut b = 0.0;

        // Gradient descent to fit logistic regression
        let learning_rate = 0.01;
        let iterations = 100;

        for _ in 0..iterations {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;

            for sample in &self.samples {
                let z = a * sample.predicted_confidence + b;
                let predicted_prob = Self::sigmoid(z);
                let y = if sample.actual_outcome { 1.0 } else { 0.0 };
                let error = predicted_prob - y;

                grad_a += error * sample.predicted_confidence;
                grad_b += error;
            }

            // Add L2 regularization
            grad_a += self.config.regularization * a;
            grad_b += self.config.regularization * b;

            // Update parameters
            a -= learning_rate * grad_a / self.samples.len() as f64;
            b -= learning_rate * grad_b / self.samples.len() as f64;
        }

        self.a = a;
        self.b = b;
        self.is_fitted = true;
    }

    /// Calibrate a predicted confidence score
    pub fn calibrate(&self, predicted_confidence: f64) -> f64 {
        if !self.is_fitted {
            return predicted_confidence;
        }

        let z = self.a * predicted_confidence + self.b;
        Self::sigmoid(z)
    }

    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Check if the calibrator is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Reset the calibrator
    pub fn reset(&mut self) {
        self.a = 0.0;
        self.b = 0.0;
        self.samples.clear();
        self.is_fitted = false;
    }
}

/// Isotonic regression calibrator
pub struct IsotonicCalibration {
    /// Calibration bins (confidence, actual probability)
    bins: Vec<(f64, f64)>,
    /// Training samples
    samples: VecDeque<CalibrationSample>,
    config: CalibrationConfig,
    is_fitted: bool,
}

impl IsotonicCalibration {
    /// Create a new isotonic calibration
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            bins: Vec::new(),
            samples: VecDeque::with_capacity(config.lookback_window),
            config,
            is_fitted: false,
        }
    }

    /// Add a calibration sample
    pub fn add_sample(&mut self, predicted_confidence: f64, actual_outcome: bool) {
        self.samples.push_back(CalibrationSample {
            predicted_confidence,
            actual_outcome,
        });

        while self.samples.len() > self.config.lookback_window {
            self.samples.pop_front();
        }

        // Refit if we have enough samples
        if self.samples.len() >= self.config.min_samples_per_bin * self.config.num_bins {
            self.fit();
        }
    }

    /// Fit the isotonic regression
    fn fit(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        // Sort samples by predicted confidence
        let mut sorted_samples: Vec<CalibrationSample> = self.samples.iter().cloned().collect();
        sorted_samples.sort_by(|a, b| {
            a.predicted_confidence
                .partial_cmp(&b.predicted_confidence)
                .unwrap()
        });

        // Create bins
        self.bins.clear();
        let bin_size = sorted_samples.len() / self.config.num_bins;

        if bin_size < self.config.min_samples_per_bin {
            return;
        }

        for i in 0..self.config.num_bins {
            let start = i * bin_size;
            let end = if i == self.config.num_bins - 1 {
                sorted_samples.len()
            } else {
                (i + 1) * bin_size
            };

            let bin_samples = &sorted_samples[start..end];

            if bin_samples.is_empty() {
                continue;
            }

            // Calculate average predicted confidence
            let avg_confidence: f64 = bin_samples
                .iter()
                .map(|s| s.predicted_confidence)
                .sum::<f64>()
                / bin_samples.len() as f64;

            // Calculate actual positive rate
            let positive_count = bin_samples.iter().filter(|s| s.actual_outcome).count();
            let actual_prob = positive_count as f64 / bin_samples.len() as f64;

            self.bins.push((avg_confidence, actual_prob));
        }

        // Ensure monotonicity (isotonic regression)
        self.enforce_monotonicity();

        self.is_fitted = true;
    }

    /// Enforce monotonicity in the calibration curve
    fn enforce_monotonicity(&mut self) {
        if self.bins.len() < 2 {
            return;
        }

        // Simple pooling adjacent violators algorithm
        let mut i = 0;
        while i < self.bins.len() - 1 {
            if self.bins[i].1 > self.bins[i + 1].1 {
                // Violation found, pool the two bins
                let avg_conf = (self.bins[i].0 + self.bins[i + 1].0) / 2.0;
                let avg_prob = (self.bins[i].1 + self.bins[i + 1].1) / 2.0;

                self.bins[i] = (avg_conf, avg_prob);
                self.bins.remove(i + 1);

                // Check previous bin again
                if i > 0 {
                    i -= 1;
                }
            } else {
                i += 1;
            }
        }
    }

    /// Calibrate a predicted confidence score
    pub fn calibrate(&self, predicted_confidence: f64) -> f64 {
        if !self.is_fitted || self.bins.is_empty() {
            return predicted_confidence;
        }

        // Find the appropriate bin using linear interpolation
        if predicted_confidence <= self.bins[0].0 {
            return self.bins[0].1;
        }

        if predicted_confidence >= self.bins[self.bins.len() - 1].0 {
            return self.bins[self.bins.len() - 1].1;
        }

        // Linear interpolation between bins
        for i in 0..self.bins.len() - 1 {
            if predicted_confidence >= self.bins[i].0 && predicted_confidence <= self.bins[i + 1].0
            {
                let x0 = self.bins[i].0;
                let x1 = self.bins[i + 1].0;
                let y0 = self.bins[i].1;
                let y1 = self.bins[i + 1].1;

                if x1 == x0 {
                    return y0;
                }

                let t = (predicted_confidence - x0) / (x1 - x0);
                return y0 + t * (y1 - y0);
            }
        }

        predicted_confidence
    }

    /// Check if the calibrator is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Reset the calibrator
    pub fn reset(&mut self) {
        self.bins.clear();
        self.samples.clear();
        self.is_fitted = false;
    }

    /// Get calibration curve points
    pub fn get_calibration_curve(&self) -> Vec<(f64, f64)> {
        self.bins.clone()
    }
}

/// Combined calibrator using both Platt scaling and isotonic regression
pub struct CombinedCalibrator {
    platt: PlattScaling,
    isotonic: IsotonicCalibration,
    #[allow(dead_code)]
    config: CalibrationConfig,
    /// Weight for Platt scaling (1-weight for isotonic)
    platt_weight: f64,
}

impl CombinedCalibrator {
    /// Create a new combined calibrator
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            platt: PlattScaling::new(config.clone()),
            isotonic: IsotonicCalibration::new(config.clone()),
            config,
            platt_weight: 0.5,
        }
    }

    /// Add a calibration sample to both calibrators
    pub fn add_sample(&mut self, predicted_confidence: f64, actual_outcome: bool) {
        self.platt.add_sample(predicted_confidence, actual_outcome);
        self.isotonic
            .add_sample(predicted_confidence, actual_outcome);
    }

    /// Calibrate using weighted combination
    pub fn calibrate(&self, predicted_confidence: f64) -> f64 {
        let platt_cal = self.platt.calibrate(predicted_confidence);
        let isotonic_cal = self.isotonic.calibrate(predicted_confidence);

        self.platt_weight * platt_cal + (1.0 - self.platt_weight) * isotonic_cal
    }

    /// Set the weight for Platt scaling
    pub fn set_platt_weight(&mut self, weight: f64) {
        self.platt_weight = weight.clamp(0.0, 1.0);
    }

    /// Check if both calibrators are fitted
    pub fn is_fitted(&self) -> bool {
        self.platt.is_fitted() && self.isotonic.is_fitted()
    }

    /// Reset both calibrators
    pub fn reset(&mut self) {
        self.platt.reset();
        self.isotonic.reset();
    }

    /// Get calibration metrics
    pub fn get_metrics(&self) -> CalibrationMetrics {
        let samples: Vec<CalibrationSample> = self.platt.samples.iter().cloned().collect();

        if samples.is_empty() {
            return CalibrationMetrics::default();
        }

        let mut brier_score = 0.0;
        let mut log_loss = 0.0;

        for sample in &samples {
            let calibrated = self.calibrate(sample.predicted_confidence);
            let actual = if sample.actual_outcome { 1.0 } else { 0.0 };

            // Brier score
            brier_score += (calibrated - actual).powi(2);

            // Log loss (with epsilon to avoid log(0))
            let epsilon = 1e-15;
            let p = calibrated.clamp(epsilon, 1.0 - epsilon);
            log_loss += -actual * p.ln() - (1.0 - actual) * (1.0 - p).ln();
        }

        brier_score /= samples.len() as f64;
        log_loss /= samples.len() as f64;

        CalibrationMetrics {
            brier_score,
            log_loss,
            sample_count: samples.len(),
            is_fitted: self.is_fitted(),
        }
    }
}

/// Calibration quality metrics
#[derive(Debug, Clone, Default)]
pub struct CalibrationMetrics {
    /// Brier score (lower is better, 0-1)
    pub brier_score: f64,
    /// Log loss (lower is better)
    pub log_loss: f64,
    /// Number of calibration samples
    pub sample_count: usize,
    /// Whether the calibrator is fitted
    pub is_fitted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_scaling_initialization() {
        let config = CalibrationConfig::default();
        let platt = PlattScaling::new(config);
        assert!(!platt.is_fitted());
    }

    #[test]
    fn test_platt_scaling_sigmoid() {
        assert!((PlattScaling::sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(PlattScaling::sigmoid(10.0) > 0.99);
        assert!(PlattScaling::sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_platt_scaling_calibration() {
        let config = CalibrationConfig {
            min_samples_per_bin: 5,
            lookback_window: 100,
            ..Default::default()
        };
        let mut platt = PlattScaling::new(config);

        // Add samples: high confidence should correlate with positive outcomes
        for _ in 0..20 {
            platt.add_sample(0.9, true);
            platt.add_sample(0.2, false);
        }

        assert!(platt.is_fitted());

        // High confidence should calibrate to high probability
        let high_cal = platt.calibrate(0.9);
        let low_cal = platt.calibrate(0.2);

        assert!(high_cal > low_cal);
    }

    #[test]
    fn test_isotonic_calibration() {
        let config = CalibrationConfig {
            num_bins: 5,
            min_samples_per_bin: 5,
            lookback_window: 100,
            ..Default::default()
        };
        let mut isotonic = IsotonicCalibration::new(config);

        // Add calibration samples
        for i in 0..50 {
            let confidence = i as f64 / 50.0;
            let outcome = confidence > 0.5;
            isotonic.add_sample(confidence, outcome);
        }

        assert!(isotonic.is_fitted());

        // Should have monotonically increasing calibration
        let low = isotonic.calibrate(0.2);
        let mid = isotonic.calibrate(0.5);
        let high = isotonic.calibrate(0.8);

        assert!(low <= mid);
        assert!(mid <= high);
    }

    #[test]
    fn test_isotonic_monotonicity() {
        let config = CalibrationConfig {
            num_bins: 5,
            min_samples_per_bin: 5,
            lookback_window: 100,
            ..Default::default()
        };
        let mut isotonic = IsotonicCalibration::new(config);

        // Add samples
        for i in 0..50 {
            let confidence = i as f64 / 50.0;
            let outcome = i % 3 == 0; // Non-monotonic outcomes
            isotonic.add_sample(confidence, outcome);
        }

        if isotonic.is_fitted() {
            // Check monotonicity in bins
            for i in 0..isotonic.bins.len() - 1 {
                assert!(
                    isotonic.bins[i].1 <= isotonic.bins[i + 1].1,
                    "Bins should be monotonic"
                );
            }
        }
    }

    #[test]
    fn test_combined_calibrator() {
        let config = CalibrationConfig {
            min_samples_per_bin: 5,
            num_bins: 5,
            lookback_window: 100,
            ..Default::default()
        };
        let mut combined = CombinedCalibrator::new(config);

        // Add samples
        for i in 0..50 {
            let confidence = i as f64 / 50.0;
            let outcome = confidence > 0.5;
            combined.add_sample(confidence, outcome);
        }

        if combined.is_fitted() {
            let calibrated = combined.calibrate(0.7);
            assert!(calibrated >= 0.0 && calibrated <= 1.0);
        }
    }

    #[test]
    fn test_combined_weight_adjustment() {
        let config = CalibrationConfig::default();
        let mut combined = CombinedCalibrator::new(config);

        combined.set_platt_weight(0.8);
        assert!((combined.platt_weight - 0.8).abs() < 1e-6);

        combined.set_platt_weight(1.5); // Should clamp to 1.0
        assert!((combined.platt_weight - 1.0).abs() < 1e-6);

        combined.set_platt_weight(-0.5); // Should clamp to 0.0
        assert!(combined.platt_weight.abs() < 1e-6);
    }

    #[test]
    fn test_calibration_metrics() {
        let config = CalibrationConfig {
            min_samples_per_bin: 5,
            num_bins: 5,
            lookback_window: 100,
            ..Default::default()
        };
        let mut combined = CombinedCalibrator::new(config);

        // Add perfect predictions
        for _ in 0..30 {
            combined.add_sample(0.9, true);
            combined.add_sample(0.1, false);
        }

        let metrics = combined.get_metrics();

        assert_eq!(metrics.sample_count, 60);
        assert!(metrics.brier_score >= 0.0);
        assert!(metrics.log_loss >= 0.0);
    }

    #[test]
    fn test_reset() {
        let config = CalibrationConfig {
            min_samples_per_bin: 5,
            ..Default::default()
        };
        let mut platt = PlattScaling::new(config);

        for _ in 0..20 {
            platt.add_sample(0.8, true);
        }

        platt.reset();
        assert!(!platt.is_fitted());
        assert_eq!(platt.samples.len(), 0);
    }

    #[test]
    fn test_calibration_edge_cases() {
        let config = CalibrationConfig::default();
        let mut isotonic = IsotonicCalibration::new(config);

        // Before fitting, should return input
        let result = isotonic.calibrate(0.7);
        assert!((result - 0.7).abs() < 1e-6);

        // Add minimal samples
        isotonic.add_sample(0.5, true);

        // Still not enough to fit
        assert!(!isotonic.is_fitted());
    }

    #[test]
    fn test_platt_scaling_with_varied_data() {
        let config = CalibrationConfig {
            min_samples_per_bin: 5,
            lookback_window: 100,
            ..Default::default()
        };
        let mut platt = PlattScaling::new(config);

        // Add varied samples
        for i in 0..30 {
            let confidence = (i % 10) as f64 / 10.0;
            let outcome = i % 2 == 0;
            platt.add_sample(confidence, outcome);
        }

        if platt.is_fitted() {
            let result = platt.calibrate(0.5);
            assert!(result >= 0.0 && result <= 1.0);
        }
    }
}
