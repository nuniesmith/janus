//! Feature extraction for ML models
//!
//! This module provides feature extraction capabilities for converting raw market data
//! into features suitable for ML models. It includes:
//!
//! - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
//! - Time-series features (returns, volatility, autocorrelation, etc.)
//! - Market microstructure features (order book depth, spreads, imbalance, etc.)
//! - Feature normalization and scaling
//!
//! # Architecture
//!
//! The module is organized into submodules:
//! - `technical` - Technical analysis indicators
//! - `timeseries` - Time-series statistical features
//! - `microstructure` - Market microstructure features
//! - `normalizer` - Feature normalization utilities
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::features::{FeatureExtractor, TechnicalIndicators};
//! use janus_core::MarketDataEvent;
//!
//! let extractor = TechnicalIndicators::new()
//!     .with_sma_periods(&[5, 10, 20])
//!     .with_rsi_period(14);
//!
//! let features = extractor.extract(&market_data)?;
//! ```

pub mod normalizer;
pub mod price;
pub mod technical;
pub mod volume;
// Future modules:
// pub mod timeseries;
// pub mod microstructure;

use crate::error::Result;
use burn_core::tensor::{Tensor, backend::Backend};
use janus_core::MarketDataEvent;

/// Trait for extracting features from market data
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from a single market data event
    ///
    /// Returns a vector of feature values in a consistent order.
    fn extract_single(&mut self, event: &MarketDataEvent) -> Result<Vec<f64>>;

    /// Extract features from a batch of market data events
    ///
    /// Returns a 2D tensor of shape [batch_size, num_features].
    fn extract_batch<B: Backend>(
        &mut self,
        events: &[MarketDataEvent],
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let features: Vec<Vec<f64>> = events
            .iter()
            .map(|event| self.extract_single(event))
            .collect::<Result<Vec<_>>>()?;

        if features.is_empty() {
            return Err(crate::error::MLError::InsufficientData(
                "No features provided for batch extraction".to_string(),
            ));
        }

        let batch_size = features.len();
        let num_features = features[0].len();

        // Flatten the 2D vector into a 1D vector
        let flat_features: Vec<f64> = features.into_iter().flatten().collect();

        // Convert to tensor
        let tensor = Tensor::<B, 1>::from_floats(flat_features.as_slice(), device);

        // Reshape to [batch_size, num_features]
        Ok(tensor.reshape([batch_size, num_features]))
    }

    /// Get the number of features this extractor produces
    fn num_features(&self) -> usize;

    /// Get the names of the features in order
    fn feature_names(&self) -> Vec<String>;

    /// Get the minimum number of historical data points required
    fn min_history_length(&self) -> usize {
        1 // Default: only need current event
    }
}

/// Feature metadata for tracking feature engineering
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureMetadata {
    /// Feature name
    pub name: String,

    /// Feature description
    pub description: String,

    /// Expected value range (min, max)
    pub value_range: Option<(f64, f64)>,

    /// Is this feature normalized?
    pub normalized: bool,

    /// Normalization method used
    pub normalization_method: Option<String>,
}

/// Container for extracted features with metadata
#[derive(Debug, Clone)]
pub struct ExtractedFeatures<B: Backend> {
    /// Feature tensor [batch_size, num_features]
    pub tensor: Tensor<B, 2>,

    /// Feature names in order
    pub names: Vec<String>,

    /// Feature metadata
    pub metadata: Vec<FeatureMetadata>,

    /// Timestamps corresponding to each sample
    pub timestamps: Vec<i64>,
}

impl<B: Backend> ExtractedFeatures<B> {
    /// Create new extracted features
    pub fn new(
        tensor: Tensor<B, 2>,
        names: Vec<String>,
        metadata: Vec<FeatureMetadata>,
        timestamps: Vec<i64>,
    ) -> Self {
        Self {
            tensor,
            names,
            metadata,
            timestamps,
        }
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.tensor.dims()[0]
    }

    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.tensor.dims()[1]
    }

    /// Get feature by name
    pub fn get_feature_index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
}

/// Helper function to calculate simple moving average
pub(crate) fn calculate_sma(values: &[f64], period: usize) -> Option<f64> {
    if values.len() < period {
        return None;
    }

    let sum: f64 = values[values.len() - period..].iter().sum();
    Some(sum / period as f64)
}

/// Helper function to calculate exponential moving average
pub(crate) fn calculate_ema(values: &[f64], period: usize) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    if values.len() < period {
        // Use SMA for initial value
        return calculate_sma(values, values.len());
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = calculate_sma(&values[..period], period)?;

    for &value in &values[period..] {
        ema = (value - ema) * multiplier + ema;
    }

    Some(ema)
}

/// Helper function to calculate standard deviation
pub(crate) fn calculate_std(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    Some(variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(calculate_sma(&values, 3), Some(4.0)); // (3+4+5)/3
        assert_eq!(calculate_sma(&values, 5), Some(3.0)); // (1+2+3+4+5)/5
        assert_eq!(calculate_sma(&values, 6), None); // Not enough data
    }

    #[test]
    fn test_calculate_ema() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let ema = calculate_ema(&values, 3);
        assert!(ema.is_some());
        assert!(ema.unwrap() >= 4.0); // EMA gives more weight to recent values
    }

    #[test]
    fn test_calculate_std() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        let std = calculate_std(&values);
        assert!(std.is_some());
        assert!((std.unwrap() - 2.0).abs() < 0.1); // Approximately 2.0
    }

    #[test]
    fn test_calculate_std_insufficient_data() {
        let values = vec![1.0];
        assert_eq!(calculate_std(&values), None);
    }
}
