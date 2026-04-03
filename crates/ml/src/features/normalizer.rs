//! Feature normalization and scaling utilities
//!
//! This module provides various normalization methods for features:
//! - Min-Max scaling (normalize to [0, 1] or custom range)
//! - Z-score normalization (standardization)
//! - Log transformation
//! - Robust scaling (using median and IQR)
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::features::normalizer::{Normalizer, NormalizationMethod};
//!
//! let normalizer = Normalizer::new(NormalizationMethod::ZScore);
//! let normalized = normalizer.fit_transform(&features)?;
//! ```

use crate::error::{MLError, Result};
use burn_core::tensor::{Tensor, backend::Backend};

/// Normalization methods for features
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub enum NormalizationMethod {
    /// No normalization
    None,

    /// Min-Max scaling to [0, 1]
    MinMax,

    /// Min-Max scaling to custom range [min, max]
    MinMaxCustom { min: f64, max: f64 },

    /// Z-score normalization (mean=0, std=1)
    #[default]
    ZScore,

    /// Log transformation (log(1 + x))
    Log,

    /// Robust scaling using median and IQR
    Robust,
}

/// Statistics for normalization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalizationStats {
    /// Feature means
    pub means: Vec<f64>,

    /// Feature standard deviations
    pub stds: Vec<f64>,

    /// Feature minimums
    pub mins: Vec<f64>,

    /// Feature maximums
    pub maxs: Vec<f64>,

    /// Feature medians
    pub medians: Vec<f64>,

    /// Feature IQRs (interquartile ranges)
    pub iqrs: Vec<f64>,

    /// Number of features
    pub num_features: usize,
}

impl NormalizationStats {
    /// Create new normalization stats
    pub fn new(num_features: usize) -> Self {
        Self {
            means: vec![0.0; num_features],
            stds: vec![1.0; num_features],
            mins: vec![0.0; num_features],
            maxs: vec![1.0; num_features],
            medians: vec![0.0; num_features],
            iqrs: vec![1.0; num_features],
            num_features,
        }
    }

    /// Compute statistics from feature data
    pub fn from_features(features: &[Vec<f64>]) -> Result<Self> {
        if features.is_empty() {
            return Err(MLError::InsufficientData(
                "No features provided for normalization stats".to_string(),
            ));
        }

        let num_features = features[0].len();
        let mut stats = Self::new(num_features);

        for feat_idx in 0..num_features {
            let values: Vec<f64> = features.iter().map(|f| f[feat_idx]).collect();

            // Calculate mean
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            stats.means[feat_idx] = mean;

            // Calculate std
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            stats.stds[feat_idx] = variance.sqrt().max(1e-8); // Avoid division by zero

            // Calculate min and max
            stats.mins[feat_idx] = values.iter().copied().fold(f64::INFINITY, f64::min);
            stats.maxs[feat_idx] = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            // Calculate median and IQR
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median_idx = sorted.len() / 2;
            stats.medians[feat_idx] = if sorted.len().is_multiple_of(2) {
                (sorted[median_idx - 1] + sorted[median_idx]) / 2.0
            } else {
                sorted[median_idx]
            };

            let q1_idx = sorted.len() / 4;
            let q3_idx = (sorted.len() * 3) / 4;
            let q1 = sorted[q1_idx];
            let q3 = sorted[q3_idx];
            stats.iqrs[feat_idx] = (q3 - q1).max(1e-8); // Avoid division by zero
        }

        Ok(stats)
    }
}

/// Feature normalizer
#[derive(Debug, Clone)]
pub struct Normalizer {
    /// Normalization method
    pub method: NormalizationMethod,

    /// Computed statistics (fitted data)
    pub stats: Option<NormalizationStats>,
}

impl Normalizer {
    /// Create a new normalizer with the specified method
    pub fn new(method: NormalizationMethod) -> Self {
        Self {
            method,
            stats: None,
        }
    }

    /// Fit the normalizer to training data
    pub fn fit(&mut self, features: &[Vec<f64>]) -> Result<()> {
        if matches!(self.method, NormalizationMethod::None) {
            return Ok(());
        }

        self.stats = Some(NormalizationStats::from_features(features)?);
        Ok(())
    }

    /// Transform features using fitted statistics
    pub fn transform(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if matches!(self.method, NormalizationMethod::None) {
            return Ok(features.to_vec());
        }

        let stats = self.stats.as_ref().ok_or_else(|| {
            MLError::InvalidConfig(
                "Normalizer not fitted. Call fit() before transform()".to_string(),
            )
        })?;

        let mut normalized = Vec::new();

        for feature_vec in features {
            if feature_vec.len() != stats.num_features {
                return Err(MLError::InvalidShape {
                    expected: format!("[*, {}]", stats.num_features),
                    actual: format!("[*, {}]", feature_vec.len()),
                });
            }

            let mut norm_vec = Vec::new();

            for (idx, &value) in feature_vec.iter().enumerate() {
                let normalized_value = match self.method {
                    NormalizationMethod::None => value,
                    NormalizationMethod::MinMax => {
                        let min = stats.mins[idx];
                        let max = stats.maxs[idx];
                        if (max - min).abs() < 1e-8 {
                            0.5 // If range is zero, use middle value
                        } else {
                            (value - min) / (max - min)
                        }
                    }
                    NormalizationMethod::MinMaxCustom { min, max } => {
                        let data_min = stats.mins[idx];
                        let data_max = stats.maxs[idx];
                        if (data_max - data_min).abs() < 1e-8 {
                            (min + max) / 2.0
                        } else {
                            let normalized = (value - data_min) / (data_max - data_min);
                            min + normalized * (max - min)
                        }
                    }
                    NormalizationMethod::ZScore => (value - stats.means[idx]) / stats.stds[idx],
                    NormalizationMethod::Log => {
                        if value < 0.0 {
                            -((1.0 - value).ln())
                        } else {
                            (1.0 + value).ln()
                        }
                    }
                    NormalizationMethod::Robust => (value - stats.medians[idx]) / stats.iqrs[idx],
                };

                norm_vec.push(normalized_value);
            }

            normalized.push(norm_vec);
        }

        Ok(normalized)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        self.fit(features)?;
        self.transform(features)
    }

    /// Inverse transform (denormalize)
    pub fn inverse_transform(&self, normalized: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if matches!(self.method, NormalizationMethod::None) {
            return Ok(normalized.to_vec());
        }

        let stats = self.stats.as_ref().ok_or_else(|| {
            MLError::InvalidConfig("Normalizer not fitted. Cannot inverse transform.".to_string())
        })?;

        let mut denormalized = Vec::new();

        for norm_vec in normalized {
            if norm_vec.len() != stats.num_features {
                return Err(MLError::InvalidShape {
                    expected: format!("[*, {}]", stats.num_features),
                    actual: format!("[*, {}]", norm_vec.len()),
                });
            }

            let mut denorm_vec = Vec::new();

            for (idx, &value) in norm_vec.iter().enumerate() {
                let denormalized_value = match self.method {
                    NormalizationMethod::None => value,
                    NormalizationMethod::MinMax => {
                        let min = stats.mins[idx];
                        let max = stats.maxs[idx];
                        value * (max - min) + min
                    }
                    NormalizationMethod::MinMaxCustom { min, max } => {
                        let data_min = stats.mins[idx];
                        let data_max = stats.maxs[idx];
                        let normalized = (value - min) / (max - min);
                        normalized * (data_max - data_min) + data_min
                    }
                    NormalizationMethod::ZScore => value * stats.stds[idx] + stats.means[idx],
                    NormalizationMethod::Log => {
                        if value < 0.0 {
                            1.0 - (-value).exp()
                        } else {
                            value.exp() - 1.0
                        }
                    }
                    NormalizationMethod::Robust => value * stats.iqrs[idx] + stats.medians[idx],
                };

                denorm_vec.push(denormalized_value);
            }

            denormalized.push(denorm_vec);
        }

        Ok(denormalized)
    }

    /// Transform a tensor
    pub fn transform_tensor<B: Backend>(&self, tensor: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        if matches!(self.method, NormalizationMethod::None) {
            return Ok(tensor);
        }

        let stats = self
            .stats
            .as_ref()
            .ok_or_else(|| MLError::InvalidConfig("Normalizer not fitted".to_string()))?;

        let device = tensor.device();
        let [batch_size, num_features] = tensor.dims();

        if num_features != stats.num_features {
            return Err(MLError::InvalidShape {
                expected: format!("[*, {}]", stats.num_features),
                actual: format!("[*, {}]", num_features),
            });
        }

        match self.method {
            NormalizationMethod::None => Ok(tensor),
            NormalizationMethod::MinMax => {
                let mins = Tensor::<B, 1>::from_floats(stats.mins.as_slice(), &device);
                let maxs = Tensor::<B, 1>::from_floats(stats.maxs.as_slice(), &device);
                let ranges = maxs.clone() - mins.clone();

                // Broadcast mins and ranges to [batch_size, num_features]
                let mins_broadcast = mins.unsqueeze_dim(0).repeat(&[batch_size, 1]);
                let ranges_broadcast = ranges.unsqueeze_dim(0).repeat(&[batch_size, 1]);

                Ok((tensor - mins_broadcast) / ranges_broadcast)
            }
            NormalizationMethod::ZScore => {
                let means = Tensor::<B, 1>::from_floats(stats.means.as_slice(), &device);
                let stds = Tensor::<B, 1>::from_floats(stats.stds.as_slice(), &device);

                let means_broadcast = means.unsqueeze_dim(0).repeat(&[batch_size, 1]);
                let stds_broadcast = stds.unsqueeze_dim(0).repeat(&[batch_size, 1]);

                Ok((tensor - means_broadcast) / stds_broadcast)
            }
            NormalizationMethod::Robust => {
                let medians = Tensor::<B, 1>::from_floats(stats.medians.as_slice(), &device);
                let iqrs = Tensor::<B, 1>::from_floats(stats.iqrs.as_slice(), &device);

                let medians_broadcast = medians.unsqueeze_dim(0).repeat(&[batch_size, 1]);
                let iqrs_broadcast = iqrs.unsqueeze_dim(0).repeat(&[batch_size, 1]);

                Ok((tensor - medians_broadcast) / iqrs_broadcast)
            }
            _ => {
                // For Log and MinMaxCustom, fall back to CPU processing
                // This is less efficient but simpler for now
                Err(MLError::InvalidConfig(
                    "Log and MinMaxCustom normalization not yet supported for tensors".to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_stats() {
        let features = vec![
            vec![1.0, 10.0, 100.0],
            vec![2.0, 20.0, 200.0],
            vec![3.0, 30.0, 300.0],
            vec![4.0, 40.0, 400.0],
            vec![5.0, 50.0, 500.0],
        ];

        let stats = NormalizationStats::from_features(&features).unwrap();

        assert_eq!(stats.num_features, 3);
        assert_eq!(stats.means[0], 3.0);
        assert_eq!(stats.mins[0], 1.0);
        assert_eq!(stats.maxs[0], 5.0);
    }

    #[test]
    fn test_minmax_normalization() {
        let features = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];

        let mut normalizer = Normalizer::new(NormalizationMethod::MinMax);
        let normalized = normalizer.fit_transform(&features).unwrap();

        // First feature should be normalized to [0, 0.5, 1]
        assert!((normalized[0][0] - 0.0).abs() < 1e-6);
        assert!((normalized[1][0] - 0.5).abs() < 1e-6);
        assert!((normalized[2][0] - 1.0).abs() < 1e-6);

        // Inverse transform should restore original
        let denormalized = normalizer.inverse_transform(&normalized).unwrap();
        assert!((denormalized[0][0] - 1.0).abs() < 1e-6);
        assert!((denormalized[1][0] - 2.0).abs() < 1e-6);
        assert!((denormalized[2][0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_zscore_normalization() {
        let features = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];

        let mut normalizer = Normalizer::new(NormalizationMethod::ZScore);
        let normalized = normalizer.fit_transform(&features).unwrap();

        // Mean of normalized features should be ~0
        let mean = (normalized[0][0] + normalized[1][0] + normalized[2][0]) / 3.0;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_log_normalization() {
        let features = vec![vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0]];

        let mut normalizer = Normalizer::new(NormalizationMethod::Log);
        let normalized = normalizer.fit_transform(&features).unwrap();

        // log(1 + 0) = 0
        assert!((normalized[0][0] - 0.0).abs() < 1e-6);
        // log(1 + 1) = log(2) ≈ 0.693
        assert!((normalized[1][0] - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_robust_normalization() {
        let features = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![100.0], // Outlier
        ];

        let mut normalizer = Normalizer::new(NormalizationMethod::Robust);
        normalizer.fit(&features).unwrap();

        // Robust scaling should be less affected by outliers than z-score
        let stats = normalizer.stats.as_ref().unwrap();
        assert!(stats.medians[0] < 10.0); // Median should be around 3
    }

    #[test]
    fn test_no_normalization() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let mut normalizer = Normalizer::new(NormalizationMethod::None);
        let normalized = normalizer.fit_transform(&features).unwrap();

        assert_eq!(normalized, features);
    }

    #[test]
    fn test_custom_minmax() {
        let features = vec![vec![0.0], vec![50.0], vec![100.0]];

        let mut normalizer = Normalizer::new(NormalizationMethod::MinMaxCustom {
            min: -1.0,
            max: 1.0,
        });
        let normalized = normalizer.fit_transform(&features).unwrap();

        // Should be normalized to [-1, 0, 1]
        assert!((normalized[0][0] - (-1.0)).abs() < 1e-6);
        assert!((normalized[1][0] - 0.0).abs() < 1e-6);
        assert!((normalized[2][0] - 1.0).abs() < 1e-6);
    }
}
