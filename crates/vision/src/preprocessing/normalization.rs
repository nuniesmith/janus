//! Normalization and scaling utilities for financial time-series data.
//!
//! This module provides various normalization strategies commonly used in
//! machine learning for financial data preprocessing.

use std::fmt;

/// Trait for scalers that can fit on data and transform it
pub trait Scaler: fmt::Debug + Clone {
    /// Fit the scaler to the data (compute statistics)
    fn fit(&mut self, data: &[f64]);

    /// Transform data using fitted parameters
    fn transform(&self, data: &[f64]) -> Vec<f64>;

    /// Fit and transform in one step
    fn fit_transform(&mut self, data: &[f64]) -> Vec<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform (if supported)
    fn inverse_transform(&self, data: &[f64]) -> Vec<f64>;

    /// Check if the scaler has been fitted
    fn is_fitted(&self) -> bool;

    /// Reset the scaler to unfitted state
    fn reset(&mut self);
}

/// Min-Max normalization: scales data to [0, 1] range
///
/// Formula: x_scaled = (x - min) / (max - min)
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    min: Option<f64>,
    max: Option<f64>,
    /// Optional custom range (default is [0, 1])
    pub feature_range: (f64, f64),
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxScaler {
    /// Create a new MinMaxScaler with default range [0, 1]
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            feature_range: (0.0, 1.0),
        }
    }

    /// Create a MinMaxScaler with custom range
    pub fn with_range(min: f64, max: f64) -> Self {
        Self {
            min: None,
            max: None,
            feature_range: (min, max),
        }
    }
}

impl Scaler for MinMaxScaler {
    fn fit(&mut self, data: &[f64]) {
        if data.is_empty() {
            return;
        }

        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        self.min = Some(min);
        self.max = Some(max);
    }

    fn transform(&self, data: &[f64]) -> Vec<f64> {
        let min = self.min.expect("Scaler must be fitted before transform");
        let max = self.max.expect("Scaler must be fitted before transform");

        let (target_min, target_max) = self.feature_range;
        let range = max - min;

        if range.abs() < 1e-10 {
            // All values are the same, return middle of target range
            return vec![(target_min + target_max) / 2.0; data.len()];
        }

        data.iter()
            .map(|&x| {
                let normalized = (x - min) / range;
                target_min + normalized * (target_max - target_min)
            })
            .collect()
    }

    fn inverse_transform(&self, data: &[f64]) -> Vec<f64> {
        let min = self
            .min
            .expect("Scaler must be fitted before inverse_transform");
        let max = self
            .max
            .expect("Scaler must be fitted before inverse_transform");

        let (target_min, target_max) = self.feature_range;
        let range = max - min;

        if range.abs() < 1e-10 {
            return vec![min; data.len()];
        }

        data.iter()
            .map(|&x| {
                let normalized = (x - target_min) / (target_max - target_min);
                min + normalized * range
            })
            .collect()
    }

    fn is_fitted(&self) -> bool {
        self.min.is_some() && self.max.is_some()
    }

    fn reset(&mut self) {
        self.min = None;
        self.max = None;
    }
}

/// Z-Score normalization: standardizes data to mean=0, std=1
///
/// Formula: x_scaled = (x - mean) / std
#[derive(Debug, Clone)]
pub struct ZScoreScaler {
    mean: Option<f64>,
    std: Option<f64>,
}

impl Default for ZScoreScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl ZScoreScaler {
    /// Create a new Z-Score scaler
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }

    /// Get the fitted mean
    pub fn mean(&self) -> Option<f64> {
        self.mean
    }

    /// Get the fitted standard deviation
    pub fn std(&self) -> Option<f64> {
        self.std
    }
}

impl Scaler for ZScoreScaler {
    fn fit(&mut self, data: &[f64]) {
        if data.is_empty() {
            return;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;

        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

        let std = variance.sqrt();

        self.mean = Some(mean);
        self.std = Some(std);
    }

    fn transform(&self, data: &[f64]) -> Vec<f64> {
        let mean = self.mean.expect("Scaler must be fitted before transform");
        let std = self.std.expect("Scaler must be fitted before transform");

        if std.abs() < 1e-10 {
            // All values are the same, return zeros
            return vec![0.0; data.len()];
        }

        data.iter().map(|&x| (x - mean) / std).collect()
    }

    fn inverse_transform(&self, data: &[f64]) -> Vec<f64> {
        let mean = self
            .mean
            .expect("Scaler must be fitted before inverse_transform");
        let std = self
            .std
            .expect("Scaler must be fitted before inverse_transform");

        data.iter().map(|&x| x * std + mean).collect()
    }

    fn is_fitted(&self) -> bool {
        self.mean.is_some() && self.std.is_some()
    }

    fn reset(&mut self) {
        self.mean = None;
        self.std = None;
    }
}

/// Robust scaler using median and IQR (Inter-Quartile Range)
///
/// More robust to outliers than Z-Score normalization.
/// Formula: x_scaled = (x - median) / IQR
#[derive(Debug, Clone)]
pub struct RobustScaler {
    median: Option<f64>,
    iqr: Option<f64>,
    q25: Option<f64>,
    q75: Option<f64>,
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustScaler {
    /// Create a new Robust scaler
    pub fn new() -> Self {
        Self {
            median: None,
            iqr: None,
            q25: None,
            q75: None,
        }
    }

    /// Get the fitted median
    pub fn median(&self) -> Option<f64> {
        self.median
    }

    /// Get the fitted IQR
    pub fn iqr(&self) -> Option<f64> {
        self.iqr
    }

    /// Calculate quantile from sorted data
    fn quantile(sorted_data: &[f64], q: f64) -> f64 {
        let n = sorted_data.len();
        let index = q * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let fraction = index - lower as f64;

        if lower == upper {
            sorted_data[lower]
        } else {
            sorted_data[lower] * (1.0 - fraction) + sorted_data[upper] * fraction
        }
    }
}

impl Scaler for RobustScaler {
    fn fit(&mut self, data: &[f64]) {
        if data.is_empty() {
            return;
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = Self::quantile(&sorted, 0.5);
        let q25 = Self::quantile(&sorted, 0.25);
        let q75 = Self::quantile(&sorted, 0.75);
        let iqr = q75 - q25;

        self.median = Some(median);
        self.q25 = Some(q25);
        self.q75 = Some(q75);
        self.iqr = Some(iqr);
    }

    fn transform(&self, data: &[f64]) -> Vec<f64> {
        let median = self.median.expect("Scaler must be fitted before transform");
        let iqr = self.iqr.expect("Scaler must be fitted before transform");

        if iqr.abs() < 1e-10 {
            // IQR is zero, return zeros
            return vec![0.0; data.len()];
        }

        data.iter().map(|&x| (x - median) / iqr).collect()
    }

    fn inverse_transform(&self, data: &[f64]) -> Vec<f64> {
        let median = self
            .median
            .expect("Scaler must be fitted before inverse_transform");
        let iqr = self
            .iqr
            .expect("Scaler must be fitted before inverse_transform");

        data.iter().map(|&x| x * iqr + median).collect()
    }

    fn is_fitted(&self) -> bool {
        self.median.is_some() && self.iqr.is_some()
    }

    fn reset(&mut self) {
        self.median = None;
        self.iqr = None;
        self.q25 = None;
        self.q75 = None;
    }
}

/// Multi-feature scaler that can scale multiple features independently
#[derive(Debug, Clone)]
pub struct MultiFeatureScaler<S: Scaler> {
    scalers: Vec<S>,
    num_features: usize,
}

impl<S: Scaler + Default> MultiFeatureScaler<S> {
    /// Create a new multi-feature scaler
    pub fn new(num_features: usize) -> Self {
        Self {
            scalers: vec![S::default(); num_features],
            num_features,
        }
    }

    /// Fit the scalers to multi-dimensional data
    /// Data shape: [num_samples, num_features]
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }

        // Transpose data to get feature columns
        for feature_idx in 0..self.num_features {
            let feature_data: Vec<f64> = data.iter().map(|sample| sample[feature_idx]).collect();

            self.scalers[feature_idx].fit(&feature_data);
        }
    }

    /// Transform multi-dimensional data
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(idx, &val)| self.scalers[idx].transform(&[val])[0])
                    .collect()
            })
            .collect()
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform
    pub fn inverse_transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(idx, &val)| self.scalers[idx].inverse_transform(&[val])[0])
                    .collect()
            })
            .collect()
    }

    /// Check if all scalers are fitted
    pub fn is_fitted(&self) -> bool {
        self.scalers.iter().all(|s| s.is_fitted())
    }

    /// Reset all scalers
    pub fn reset(&mut self) {
        for scaler in &mut self.scalers {
            scaler.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_scaler() {
        let mut scaler = MinMaxScaler::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(!scaler.is_fitted());

        let normalized = scaler.fit_transform(&data);

        assert!(scaler.is_fitted());
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[4], 1.0);
        assert!((normalized[2] - 0.5).abs() < 1e-10);

        let denormalized = scaler.inverse_transform(&normalized);
        for (orig, denorm) in data.iter().zip(denormalized.iter()) {
            assert!((orig - denorm).abs() < 1e-10);
        }
    }

    #[test]
    fn test_minmax_custom_range() {
        let mut scaler = MinMaxScaler::with_range(-1.0, 1.0);
        let data = vec![0.0, 5.0, 10.0];

        let normalized = scaler.fit_transform(&data);

        assert!((normalized[0] - (-1.0)).abs() < 1e-10);
        assert!((normalized[2] - 1.0).abs() < 1e-10);
        assert!((normalized[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_constant_data() {
        let mut scaler = MinMaxScaler::new();
        let data = vec![5.0, 5.0, 5.0];

        let normalized = scaler.fit_transform(&data);

        // Should return middle of range for constant data
        for &val in &normalized {
            assert!((val - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zscore_scaler() {
        let mut scaler = ZScoreScaler::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let normalized = scaler.fit_transform(&data);

        // Mean should be 3.0, std should be sqrt(2)
        assert!((scaler.mean().unwrap() - 3.0).abs() < 1e-10);

        // Normalized data should have mean ~0 and std ~1
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);

        let denormalized = scaler.inverse_transform(&normalized);
        for (orig, denorm) in data.iter().zip(denormalized.iter()) {
            assert!((orig - denorm).abs() < 1e-9);
        }
    }

    #[test]
    fn test_zscore_constant_data() {
        let mut scaler = ZScoreScaler::new();
        let data = vec![5.0, 5.0, 5.0];

        let normalized = scaler.fit_transform(&data);

        // Should return zeros for constant data
        for &val in &normalized {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_robust_scaler() {
        let mut scaler = RobustScaler::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // With outlier

        let normalized = scaler.fit_transform(&data);

        assert!(scaler.is_fitted());
        assert!(scaler.median().is_some());
        assert!(scaler.iqr().is_some());

        // Median should be 3.5
        assert!((scaler.median().unwrap() - 3.5).abs() < 1e-10);

        let denormalized = scaler.inverse_transform(&normalized);
        for (orig, denorm) in data.iter().zip(denormalized.iter()) {
            assert!((orig - denorm).abs() < 1e-9);
        }
    }

    #[test]
    fn test_scaler_reset() {
        let mut scaler = MinMaxScaler::new();
        let data = vec![1.0, 2.0, 3.0];

        scaler.fit(&data);
        assert!(scaler.is_fitted());

        scaler.reset();
        assert!(!scaler.is_fitted());
    }

    #[test]
    fn test_multi_feature_scaler() {
        let mut scaler = MultiFeatureScaler::<ZScoreScaler>::new(2);
        let data = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
        ];

        let normalized = scaler.fit_transform(&data);

        assert!(scaler.is_fitted());
        assert_eq!(normalized.len(), 4);
        assert_eq!(normalized[0].len(), 2);

        // Each feature should be independently normalized
        let feature0: Vec<f64> = normalized.iter().map(|s| s[0]).collect();
        let mean0: f64 = feature0.iter().sum::<f64>() / feature0.len() as f64;
        assert!(mean0.abs() < 1e-9);

        let denormalized = scaler.inverse_transform(&normalized);
        for (orig, denorm) in data.iter().zip(denormalized.iter()) {
            for (o, d) in orig.iter().zip(denorm.iter()) {
                assert!((o - d).abs() < 1e-9);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Scaler must be fitted before transform")]
    fn test_transform_before_fit() {
        let scaler = MinMaxScaler::new();
        let data = vec![1.0, 2.0, 3.0];
        scaler.transform(&data);
    }
}
