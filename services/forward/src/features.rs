//! # Feature Engineering Module
//!
//! Transforms indicator analysis and market data into feature vectors for ML model inference.
//!
//! ## Overview
//!
//! The feature engineering pipeline converts raw technical indicators and market data
//! into normalized, standardized feature vectors suitable for machine learning models.
//!
//! ## Features
//!
//! - **Indicator-based Features**: Extract features from technical indicators (EMA, RSI, MACD, etc.)
//! - **Price-based Features**: Compute price momentum, volatility, and trend features
//! - **Normalization**: Z-score and min-max normalization for stable model inputs
//! - **Feature Validation**: Ensure features are within expected ranges
//! - **Extensible**: Easy to add new feature extractors
//!
//! ## Example
//!
//! ```rust,ignore
//! use janus_forward::features::{FeatureEngineering, FeatureConfig};
//! use janus::indicators::IndicatorAnalysis;
//!
//! let config = FeatureConfig {
//!     enable_validation: false,  // Disable for doctest
//!     ..Default::default()
//! };
//! let feature_eng = FeatureEngineering::new(config);
//!
//! let analysis = IndicatorAnalysis::default();
//! let features = feature_eng.extract_features(&analysis, 50000.0)?;
//!
//! println!("Feature vector: {:?}", features.values);
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::indicators::IndicatorAnalysis;
use anyhow::{Result, anyhow};
use chrono::{Datelike, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Enable normalization
    pub enable_normalization: bool,

    /// Normalization method
    pub normalization_method: NormalizationMethod,

    /// Minimum number of features required
    pub min_features: usize,

    /// Maximum allowed feature value (for outlier detection)
    pub max_feature_value: f64,

    /// Enable feature validation
    pub enable_validation: bool,

    /// Price lookback periods for momentum features
    pub momentum_periods: Vec<usize>,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            enable_normalization: true,
            normalization_method: NormalizationMethod::ZScore,
            min_features: 10,
            max_feature_value: 100.0,
            enable_validation: true,
            momentum_periods: vec![5, 10, 20, 50],
        }
    }
}

/// Normalization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Z-score normalization: (x - mean) / std
    ZScore,
    /// Min-max normalization: (x - min) / (max - min)
    MinMax,
    /// No normalization
    None,
}

/// Feature vector for ML model input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Feature names
    pub names: Vec<String>,

    /// Feature values
    pub values: Vec<f64>,

    /// Feature metadata
    pub metadata: FeatureMetadata,
}

impl FeatureVector {
    /// Create a new feature vector
    pub fn new(names: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            names,
            values,
            metadata: FeatureMetadata::default(),
        }
    }

    /// Get feature count
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get feature by name
    pub fn get(&self, name: &str) -> Option<f64> {
        self.names
            .iter()
            .position(|n| n == name)
            .and_then(|idx| self.values.get(idx).copied())
    }

    /// Convert to flat array (for model input)
    pub fn to_array(&self) -> Vec<f32> {
        self.values.iter().map(|&v| v as f32).collect()
    }

    /// Validate feature vector
    pub fn validate(&self, config: &FeatureConfig) -> Result<()> {
        if !config.enable_validation {
            return Ok(());
        }

        // Check minimum features
        if self.len() < config.min_features {
            return Err(anyhow!(
                "Insufficient features: {} < {}",
                self.len(),
                config.min_features
            ));
        }

        // Check for NaN or infinite values
        for (idx, &value) in self.values.iter().enumerate() {
            if !value.is_finite() {
                return Err(anyhow!("Invalid feature value at index {}: {}", idx, value));
            }

            if value.abs() > config.max_feature_value {
                return Err(anyhow!(
                    "Feature value out of range at index {}: {} (max: {})",
                    idx,
                    value,
                    config.max_feature_value
                ));
            }
        }

        Ok(())
    }
}

/// Feature metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureMetadata {
    /// Feature extraction timestamp
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,

    /// Feature statistics
    pub stats: FeatureStats,

    /// Additional context
    pub context: HashMap<String, String>,
}

/// Feature statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl FeatureStats {
    /// Calculate statistics from values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;

        let variance: f64 = values
            .iter()
            .map(|&v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        let std = variance.sqrt();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Self {
            mean,
            std,
            min,
            max,
        }
    }
}

/// Feature engineering pipeline
pub struct FeatureEngineering {
    config: FeatureConfig,
    price_history: Vec<f64>,
}

impl FeatureEngineering {
    /// Create a new feature engineering instance
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
            price_history: Vec::new(),
        }
    }

    /// Update price history
    pub fn update_price(&mut self, price: f64) {
        self.price_history.push(price);

        // Keep only the required history
        let max_period = self
            .config
            .momentum_periods
            .iter()
            .max()
            .copied()
            .unwrap_or(50);
        if self.price_history.len() > max_period + 10 {
            self.price_history.drain(0..10);
        }
    }

    /// Extract features from indicator analysis
    pub fn extract_features(
        &self,
        analysis: &IndicatorAnalysis,
        current_price: f64,
    ) -> Result<FeatureVector> {
        let mut names = Vec::new();
        let mut values = Vec::new();

        // 1. EMA Features
        if let Some(ema_fast) = analysis.ema_fast {
            names.push("ema_fast".to_string());
            values.push(ema_fast);

            if let Some(ema_slow) = analysis.ema_slow {
                names.push("ema_slow".to_string());
                values.push(ema_slow);

                names.push("ema_distance".to_string());
                values.push((ema_fast - ema_slow) / ema_slow * 100.0); // percentage distance
            }

            names.push("ema_cross".to_string());
            values.push(analysis.ema_cross);
        }

        // 2. RSI Features
        if let Some(rsi) = analysis.rsi {
            names.push("rsi".to_string());
            values.push(rsi);

            names.push("rsi_oversold".to_string());
            values.push(if rsi < 30.0 { 1.0 } else { 0.0 });

            names.push("rsi_overbought".to_string());
            values.push(if rsi > 70.0 { 1.0 } else { 0.0 });

            names.push("rsi_normalized".to_string());
            values.push(rsi / 100.0); // 0-1 range
        }

        // 3. MACD Features
        if let (Some(macd), Some(signal), Some(histogram)) = (
            analysis.macd_line,
            analysis.macd_signal,
            analysis.macd_histogram,
        ) {
            names.push("macd_line".to_string());
            values.push(macd);

            names.push("macd_signal".to_string());
            values.push(signal);

            names.push("macd_histogram".to_string());
            values.push(histogram);

            names.push("macd_crossover".to_string());
            values.push(if histogram > 0.0 { 1.0 } else { -1.0 });

            names.push("macd_strength".to_string());
            values.push(histogram.abs());

            names.push("macd_cross".to_string());
            values.push(analysis.macd_cross);
        }

        // 4. Bollinger Bands Features
        if let (Some(bb_upper), Some(bb_middle), Some(bb_lower)) =
            (analysis.bb_upper, analysis.bb_middle, analysis.bb_lower)
        {
            let bb_width = bb_upper - bb_lower;
            let bb_position = if bb_width > 0.0 {
                (current_price - bb_lower) / bb_width
            } else {
                0.5
            };

            names.push("bb_position".to_string());
            values.push(bb_position);

            names.push("bb_width".to_string());
            values.push(bb_width / bb_middle * 100.0); // percentage width

            names.push("bb_upper_distance".to_string());
            values.push((bb_upper - current_price) / current_price * 100.0);

            names.push("bb_lower_distance".to_string());
            values.push((current_price - bb_lower) / current_price * 100.0);
        }

        // 5. ATR Features (volatility)
        if let Some(atr) = analysis.atr {
            names.push("atr".to_string());
            values.push(atr);

            names.push("atr_pct".to_string());
            values.push(atr / current_price * 100.0);
        }

        // 6. Price-based Features
        names.push("price".to_string());
        values.push(current_price);

        // 7. Momentum Features (if price history available)
        if !self.price_history.is_empty() {
            for &period in &self.config.momentum_periods {
                if self.price_history.len() > period {
                    let past_price = self.price_history[self.price_history.len() - period - 1];
                    let momentum = (current_price - past_price) / past_price * 100.0;

                    names.push(format!("momentum_{}", period));
                    values.push(momentum);
                }
            }
        }

        // 8. Trend Features
        if let Some(ema_fast) = analysis.ema_fast {
            names.push("price_above_ema".to_string());
            values.push(if current_price > ema_fast { 1.0 } else { 0.0 });

            names.push("trend_strength".to_string());
            values.push(((current_price - ema_fast) / ema_fast * 100.0).abs());
        }

        // Create feature vector
        let mut feature_vec = FeatureVector::new(names, values);

        // Apply normalization if enabled
        if self.config.enable_normalization {
            self.normalize_features(&mut feature_vec)?;
        }

        // Calculate statistics
        feature_vec.metadata.stats = FeatureStats::from_values(&feature_vec.values);
        feature_vec.metadata.timestamp = Some(chrono::Utc::now());

        // Validate
        feature_vec.validate(&self.config)?;

        Ok(feature_vec)
    }

    /// Normalize feature vector
    fn normalize_features(&self, features: &mut FeatureVector) -> Result<()> {
        match self.config.normalization_method {
            NormalizationMethod::ZScore => self.zscore_normalize(features),
            NormalizationMethod::MinMax => self.minmax_normalize(features),
            NormalizationMethod::None => Ok(()),
        }
    }

    /// Apply z-score normalization
    fn zscore_normalize(&self, features: &mut FeatureVector) -> Result<()> {
        let stats = FeatureStats::from_values(&features.values);

        if stats.std > 0.0 {
            for value in &mut features.values {
                *value = (*value - stats.mean) / stats.std;
            }
        }

        Ok(())
    }

    /// Apply min-max normalization
    fn minmax_normalize(&self, features: &mut FeatureVector) -> Result<()> {
        let stats = FeatureStats::from_values(&features.values);

        let range = stats.max - stats.min;
        if range > 0.0 {
            for value in &mut features.values {
                *value = (*value - stats.min) / range;
            }
        }

        Ok(())
    }

    /// Extract features from raw market data
    pub fn extract_from_market_data(
        &self,
        symbol: &str,
        price: f64,
        volume: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<FeatureVector> {
        let names = vec![
            "price".to_string(),
            "volume".to_string(),
            "hour_of_day".to_string(),
            "day_of_week".to_string(),
        ];

        let values = vec![
            price,
            volume,
            timestamp.hour() as f64,
            timestamp.weekday().num_days_from_monday() as f64,
        ];

        let mut feature_vec = FeatureVector::new(names, values);
        feature_vec.metadata.timestamp = Some(timestamp);
        feature_vec
            .metadata
            .context
            .insert("symbol".to_string(), symbol.to_string());

        Ok(feature_vec)
    }

    /// Combine multiple feature vectors
    pub fn combine_features(&self, vectors: Vec<FeatureVector>) -> Result<FeatureVector> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot combine empty feature vectors"));
        }

        let mut names = Vec::new();
        let mut values = Vec::new();

        for (idx, vec) in vectors.iter().enumerate() {
            for (name, &value) in vec.names.iter().zip(&vec.values) {
                names.push(format!("{}_{}", name, idx));
                values.push(value);
            }
        }

        let combined = FeatureVector::new(names, values);
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_config_default() {
        let config = FeatureConfig::default();
        assert!(config.enable_normalization);
        assert_eq!(config.min_features, 10);
    }

    #[test]
    fn test_feature_vector_creation() {
        let names = vec!["feature1".to_string(), "feature2".to_string()];
        let values = vec![1.0, 2.0];
        let fv = FeatureVector::new(names, values);

        assert_eq!(fv.len(), 2);
        assert!(!fv.is_empty());
        assert_eq!(fv.get("feature1"), Some(1.0));
        assert_eq!(fv.get("feature2"), Some(2.0));
        assert_eq!(fv.get("feature3"), None);
    }

    #[test]
    fn test_feature_vector_to_array() {
        let names = vec!["a".to_string(), "b".to_string()];
        let values = vec![1.5, 2.5];
        let fv = FeatureVector::new(names, values);

        let array = fv.to_array();
        assert_eq!(array, vec![1.5f32, 2.5f32]);
    }

    #[test]
    fn test_feature_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FeatureStats::from_values(&values);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!(stats.std > 0.0);
    }

    #[test]
    fn test_feature_stats_empty() {
        let values = vec![];
        let stats = FeatureStats::from_values(&values);

        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_feature_engineering_creation() {
        let config = FeatureConfig::default();
        let fe = FeatureEngineering::new(config);
        assert_eq!(fe.price_history.len(), 0);
    }

    #[test]
    fn test_extract_features_basic() {
        let config = FeatureConfig {
            enable_normalization: false,
            enable_validation: false,
            ..Default::default()
        };
        let fe = FeatureEngineering::new(config);

        let analysis = IndicatorAnalysis {
            ema_fast: Some(50000.0),
            ema_slow: Some(49500.0),
            ema_cross: 1.0,
            rsi: Some(45.0),
            rsi_signal: -0.2,
            macd_line: Some(100.0),
            macd_signal: Some(90.0),
            macd_histogram: Some(10.0),
            macd_cross: 1.0,
            atr: Some(500.0),
            bb_upper: Some(51000.0),
            bb_middle: Some(50000.0),
            bb_lower: Some(49000.0),
            bb_position: 0.5,
            trend_strength: 0.6,
            volatility: 0.5,
        };

        let result = fe.extract_features(&analysis, 50500.0);
        assert!(result.is_ok());

        let features = result.unwrap();
        assert!(features.len() >= 10);
        assert!(features.get("rsi").is_some());
        assert!(features.get("macd_line").is_some());
    }

    #[test]
    fn test_extract_features_with_normalization() {
        let config = FeatureConfig {
            enable_normalization: true,
            normalization_method: NormalizationMethod::ZScore,
            enable_validation: false,
            ..Default::default()
        };
        let fe = FeatureEngineering::new(config);

        let analysis = IndicatorAnalysis {
            ema_fast: Some(50000.0),
            ema_slow: Some(49500.0),
            ema_cross: 1.0,
            rsi: Some(45.0),
            rsi_signal: -0.2,
            macd_line: Some(100.0),
            macd_signal: Some(90.0),
            macd_histogram: Some(10.0),
            macd_cross: 1.0,
            atr: Some(500.0),
            bb_upper: Some(51000.0),
            bb_middle: Some(50000.0),
            bb_lower: Some(49000.0),
            bb_position: 0.5,
            trend_strength: 0.6,
            volatility: 0.5,
        };

        let result = fe.extract_features(&analysis, 50500.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_update_price_history() {
        let config = FeatureConfig::default();
        let mut fe = FeatureEngineering::new(config);

        fe.update_price(100.0);
        fe.update_price(101.0);
        fe.update_price(102.0);

        assert_eq!(fe.price_history.len(), 3);
    }

    #[test]
    fn test_extract_from_market_data() {
        let config = FeatureConfig {
            enable_normalization: false,
            enable_validation: false,
            ..Default::default()
        };
        let fe = FeatureEngineering::new(config);

        let timestamp = chrono::Utc::now();
        let result = fe.extract_from_market_data("BTC/USD", 50000.0, 1000.0, timestamp);

        assert!(result.is_ok());
        let features = result.unwrap();
        assert_eq!(features.get("price"), Some(50000.0));
        assert_eq!(features.get("volume"), Some(1000.0));
    }

    #[test]
    fn test_feature_validation() {
        let config = FeatureConfig {
            enable_validation: true,
            min_features: 5,
            max_feature_value: 10.0,
            ..Default::default()
        };

        // Valid case
        let names = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fv = FeatureVector::new(names, values);
        assert!(fv.validate(&config).is_ok());

        // Too few features
        let names = vec!["a".to_string()];
        let values = vec![1.0];
        let fv = FeatureVector::new(names, values);
        assert!(fv.validate(&config).is_err());

        // Value out of range
        let names = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        let fv = FeatureVector::new(names, values);
        assert!(fv.validate(&config).is_err());
    }

    #[test]
    fn test_combine_features() {
        let config = FeatureConfig::default();
        let fe = FeatureEngineering::new(config);

        let fv1 = FeatureVector::new(vec!["a".to_string()], vec![1.0]);
        let fv2 = FeatureVector::new(vec!["b".to_string()], vec![2.0]);

        let result = fe.combine_features(vec![fv1, fv2]);
        assert!(result.is_ok());

        let combined = result.unwrap();
        assert_eq!(combined.len(), 2);
    }

    #[test]
    fn test_combine_empty_features() {
        let config = FeatureConfig::default();
        let fe = FeatureEngineering::new(config);

        let result = fe.combine_features(vec![]);
        assert!(result.is_err());
    }
}
