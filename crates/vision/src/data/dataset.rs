//! Dataset abstractions for time-series OHLCV data.
//!
//! This module provides utilities for creating sequences from OHLCV candles,
//! generating labels, and splitting data into train/validation/test sets.

use crate::data::csv_loader::OhlcvCandle;

/// Configuration for sequence generation
#[derive(Debug, Clone)]
pub struct SequenceConfig {
    /// Number of candles per sequence (time window)
    pub sequence_length: usize,
    /// Number of candles to skip between sequences (stride)
    pub stride: usize,
    /// Prediction horizon (candles ahead to predict)
    pub prediction_horizon: usize,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            sequence_length: 60,
            stride: 1,
            prediction_horizon: 1,
        }
    }
}

/// Train/validation/test split configuration
#[derive(Debug, Clone)]
pub struct TrainValSplit {
    /// Fraction of data for training (0.0 to 1.0)
    pub train_ratio: f32,
    /// Fraction of data for validation (0.0 to 1.0)
    pub val_ratio: f32,
    // test_ratio is implied: 1.0 - train_ratio - val_ratio
}

impl Default for TrainValSplit {
    fn default() -> Self {
        Self {
            train_ratio: 0.7,
            val_ratio: 0.15,
            // test_ratio = 0.15
        }
    }
}

impl TrainValSplit {
    /// Get the test ratio
    pub fn test_ratio(&self) -> f32 {
        1.0 - self.train_ratio - self.val_ratio
    }

    /// Validate that ratios sum to <= 1.0
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.train_ratio < 0.0 || self.train_ratio > 1.0 {
            return Err(anyhow::anyhow!("train_ratio must be between 0.0 and 1.0"));
        }
        if self.val_ratio < 0.0 || self.val_ratio > 1.0 {
            return Err(anyhow::anyhow!("val_ratio must be between 0.0 and 1.0"));
        }
        if self.train_ratio + self.val_ratio > 1.0 {
            return Err(anyhow::anyhow!(
                "train_ratio + val_ratio must be <= 1.0 (got {})",
                self.train_ratio + self.val_ratio
            ));
        }
        Ok(())
    }
}

/// OHLCV dataset with sequences for time-series modeling
pub struct OhlcvDataset {
    /// Sequences of candles
    pub sequences: Vec<Vec<OhlcvCandle>>,
    /// Labels for each sequence (e.g., future returns or classification labels)
    pub labels: Vec<f64>,
    /// Configuration used to create this dataset
    pub config: SequenceConfig,
}

impl OhlcvDataset {
    /// Create dataset from candles with sliding window
    pub fn from_candles(candles: Vec<OhlcvCandle>, config: SequenceConfig) -> anyhow::Result<Self> {
        if candles.is_empty() {
            return Err(anyhow::anyhow!("Cannot create dataset from empty candles"));
        }

        let total_length = config.sequence_length + config.prediction_horizon;
        if candles.len() < total_length {
            return Err(anyhow::anyhow!(
                "Not enough candles: need at least {}, got {}",
                total_length,
                candles.len()
            ));
        }

        let mut sequences = Vec::new();
        let mut labels = Vec::new();

        for i in (0..candles.len().saturating_sub(total_length)).step_by(config.stride) {
            // Extract sequence
            let sequence: Vec<OhlcvCandle> = candles
                .iter()
                .skip(i)
                .take(config.sequence_length)
                .cloned()
                .collect();

            // Calculate label: future return (percentage change)
            let current_close = candles[i + config.sequence_length - 1].close;
            let future_close =
                candles[i + config.sequence_length + config.prediction_horizon - 1].close;
            let return_pct = (future_close - current_close) / current_close;

            sequences.push(sequence);
            labels.push(return_pct);
        }

        Ok(Self {
            sequences,
            labels,
            config,
        })
    }

    /// Split dataset into train/val/test sets (temporal split, not random)
    pub fn split(&self, split_config: TrainValSplit) -> anyhow::Result<(Self, Self, Self)> {
        split_config.validate()?;

        let total = self.sequences.len();
        if total == 0 {
            return Err(anyhow::anyhow!("Cannot split empty dataset"));
        }

        let train_end = (total as f32 * split_config.train_ratio) as usize;
        let val_end = train_end + (total as f32 * split_config.val_ratio) as usize;

        let train = Self {
            sequences: self.sequences[..train_end].to_vec(),
            labels: self.labels[..train_end].to_vec(),
            config: self.config.clone(),
        };

        let val = Self {
            sequences: self.sequences[train_end..val_end].to_vec(),
            labels: self.labels[train_end..val_end].to_vec(),
            config: self.config.clone(),
        };

        let test = Self {
            sequences: self.sequences[val_end..].to_vec(),
            labels: self.labels[val_end..].to_vec(),
            config: self.config.clone(),
        };

        Ok((train, val, test))
    }

    /// Convert continuous labels to classification labels
    /// Returns class indices: 0 (down), 1 (neutral), 2 (up)
    pub fn to_classification_labels(&self, threshold: f64) -> Vec<usize> {
        self.labels
            .iter()
            .map(|&ret| {
                if ret > threshold {
                    2 // Up
                } else if ret < -threshold {
                    0 // Down
                } else {
                    1 // Neutral
                }
            })
            .collect()
    }

    /// Get feature matrix for all sequences
    /// Returns Vec<Vec<Vec<f64>>> with shape [num_sequences, sequence_length, 5]
    pub fn to_feature_matrix(&self) -> Vec<Vec<Vec<f64>>> {
        self.sequences
            .iter()
            .map(|seq| seq.iter().map(|candle| candle.to_features()).collect())
            .collect()
    }

    /// Number of sequences in the dataset
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Get a single sequence and label by index
    pub fn get(&self, idx: usize) -> Option<(&[OhlcvCandle], f64)> {
        if idx < self.len() {
            Some((&self.sequences[idx], self.labels[idx]))
        } else {
            None
        }
    }

    /// Get statistics about the dataset
    pub fn stats(&self) -> DatasetStats {
        if self.is_empty() {
            return DatasetStats::default();
        }

        let mean_return = self.labels.iter().sum::<f64>() / self.labels.len() as f64;

        let variance = self
            .labels
            .iter()
            .map(|&x| (x - mean_return).powi(2))
            .sum::<f64>()
            / self.labels.len() as f64;

        let std_return = variance.sqrt();

        let min_return = self.labels.iter().cloned().fold(f64::INFINITY, f64::min);

        let max_return = self
            .labels
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        DatasetStats {
            num_sequences: self.len(),
            mean_return,
            std_return,
            min_return,
            max_return,
        }
    }
}

/// Statistics about a dataset
#[derive(Debug, Clone, Default)]
pub struct DatasetStats {
    pub num_sequences: usize,
    pub mean_return: f64,
    pub std_return: f64,
    pub min_return: f64,
    pub max_return: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candles(n: usize) -> Vec<OhlcvCandle> {
        (0..n)
            .map(|i| {
                let price = 100.0 + i as f64;
                OhlcvCandle::new(
                    Utc::now(),
                    price,
                    price + 1.0,
                    price - 1.0,
                    price + 0.5,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_sequence_config_default() {
        let config = SequenceConfig::default();
        assert_eq!(config.sequence_length, 60);
        assert_eq!(config.stride, 1);
        assert_eq!(config.prediction_horizon, 1);
    }

    #[test]
    fn test_train_val_split_validation() {
        let valid = TrainValSplit {
            train_ratio: 0.7,
            val_ratio: 0.2,
        };
        assert!(valid.validate().is_ok());
        let test_ratio = valid.test_ratio();
        assert!((test_ratio - 0.1).abs() < 0.0001);

        let invalid = TrainValSplit {
            train_ratio: 0.8,
            val_ratio: 0.5, // Sum > 1.0
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_dataset_creation() {
        let candles = create_test_candles(100);
        let config = SequenceConfig {
            sequence_length: 10,
            stride: 5,
            prediction_horizon: 1,
        };

        let dataset = OhlcvDataset::from_candles(candles, config).unwrap();

        // With stride=5, we should get (100 - 11) / 5 + 1 = 18 sequences
        assert!(dataset.len() > 0);
        assert_eq!(dataset.sequences.len(), dataset.labels.len());
    }

    #[test]
    fn test_dataset_creation_insufficient_data() {
        let candles = create_test_candles(5);
        let config = SequenceConfig {
            sequence_length: 10,
            stride: 1,
            prediction_horizon: 1,
        };

        let result = OhlcvDataset::from_candles(candles, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_split() {
        let candles = create_test_candles(200);
        let config = SequenceConfig {
            sequence_length: 10,
            stride: 1,
            prediction_horizon: 1,
        };

        let dataset = OhlcvDataset::from_candles(candles, config).unwrap();
        let split = TrainValSplit::default();
        let (train, val, test) = dataset.split(split.clone()).unwrap();

        let total = dataset.len();
        assert_eq!(train.len(), (total as f32 * 0.7) as usize);
        assert!(val.len() > 0);
        assert!(test.len() > 0);

        // Verify no overlap and all data accounted for
        assert_eq!(train.len() + val.len() + test.len(), total);

        // Test ratio calculation with floating point tolerance
        let test_ratio = split.test_ratio();
        assert!((test_ratio - 0.15).abs() < 0.0001);
    }

    #[test]
    fn test_classification_labels() {
        let candles = create_test_candles(100);
        let config = SequenceConfig::default();
        let dataset = OhlcvDataset::from_candles(candles, config).unwrap();

        let labels = dataset.to_classification_labels(0.01); // 1% threshold

        // All labels should be 0, 1, or 2
        for &label in &labels {
            assert!(label <= 2);
        }

        assert_eq!(labels.len(), dataset.len());
    }

    #[test]
    fn test_feature_matrix() {
        let candles = create_test_candles(100);
        let config = SequenceConfig {
            sequence_length: 10,
            stride: 10,
            prediction_horizon: 1,
        };
        let dataset = OhlcvDataset::from_candles(candles, config).unwrap();

        let matrix = dataset.to_feature_matrix();

        // Check dimensions
        assert_eq!(matrix.len(), dataset.len());
        for seq in &matrix {
            assert_eq!(seq.len(), 10); // sequence_length
            for features in seq {
                assert_eq!(features.len(), 5); // OHLCV
            }
        }
    }

    #[test]
    fn test_dataset_stats() {
        let candles = create_test_candles(100);
        let config = SequenceConfig::default();
        let dataset = OhlcvDataset::from_candles(candles, config).unwrap();

        let stats = dataset.stats();
        assert_eq!(stats.num_sequences, dataset.len());
        assert!(stats.mean_return.is_finite());
        assert!(stats.std_return >= 0.0);
        assert!(stats.min_return <= stats.max_return);
    }

    #[test]
    fn test_dataset_get() {
        let candles = create_test_candles(100);
        let config = SequenceConfig::default();
        let dataset = OhlcvDataset::from_candles(candles, config.clone()).unwrap();

        // Valid index
        let (seq, label) = dataset.get(0).unwrap();
        assert_eq!(seq.len(), config.sequence_length);
        assert!(label.is_finite());

        // Invalid index
        assert!(dataset.get(dataset.len()).is_none());
    }
}
