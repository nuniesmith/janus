//! Tensor conversion utilities for Burn framework integration.
//!
//! This module provides utilities to convert OHLCV data and sequences
//! into Burn tensors for use with neural networks.

use burn::tensor::{Shape, Tensor, TensorData, backend::Backend};

use crate::data::csv_loader::OhlcvCandle;
use crate::data::dataset::OhlcvDataset;
use crate::preprocessing::normalization::{MultiFeatureScaler, Scaler};

/// Configuration for tensor conversion
#[derive(Debug, Clone)]
pub struct TensorConverterConfig {
    /// Whether to normalize features before conversion
    pub normalize: bool,
    /// Number of features per candle (default: 5 for OHLCV)
    pub num_features: usize,
}

impl Default for TensorConverterConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            num_features: 5, // OHLCV
        }
    }
}

/// Converts OHLCV data to Burn tensors
pub struct TensorConverter<S: Scaler + Default> {
    config: TensorConverterConfig,
    scaler: Option<MultiFeatureScaler<S>>,
}

impl<S: Scaler + Default> TensorConverter<S> {
    /// Create a new tensor converter
    pub fn new(config: TensorConverterConfig) -> Self {
        let scaler = if config.normalize {
            Some(MultiFeatureScaler::new(config.num_features))
        } else {
            None
        };

        Self { config, scaler }
    }

    /// Fit the scaler on training data
    pub fn fit(&mut self, dataset: &OhlcvDataset) {
        if self.config.normalize {
            let features = self.extract_all_features(dataset);
            if let Some(ref mut scaler) = self.scaler {
                scaler.fit(&features);
            }
        }
    }

    /// Fit the scaler on pre-engineered feature matrices
    ///
    /// # Arguments
    /// * `features` - Vec of feature matrices, where each matrix is [sequence_length, num_features]
    pub fn fit_with_features(&mut self, features: &[Vec<Vec<f64>>]) {
        if self.config.normalize {
            // Flatten all sequences into a single Vec<Vec<f64>> for fitting
            let all_features: Vec<Vec<f64>> = features.iter().flatten().cloned().collect();
            if let Some(ref mut scaler) = self.scaler {
                scaler.fit(&all_features);
            }
        }
    }

    /// Convert a single sequence to a tensor
    ///
    /// Returns a tensor of shape [sequence_length, num_features]
    pub fn sequence_to_tensor<B: Backend>(
        &self,
        sequence: &[OhlcvCandle],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let features = self.extract_sequence_features(sequence);
        let normalized = self.normalize_features(&[features]);

        // normalized is Vec<Vec<Vec<f64>>> with outer vec length 1
        // Extract the first (and only) element which is Vec<Vec<f64>>
        let seq_features = &normalized[0];
        let sequence_length = seq_features.len();
        let num_features = seq_features[0].len();

        let flat_data: Vec<f32> = seq_features.iter().flatten().map(|&x| x as f32).collect();

        let shape = Shape::new([sequence_length, num_features]);
        let data = TensorData::new(flat_data, shape);
        Tensor::<B, 2>::from_data(data.convert::<f32>(), device)
    }

    /// Convert an entire dataset to a batch tensor
    ///
    /// Returns a tensor of shape [batch_size, sequence_length, num_features]
    pub fn dataset_to_tensor<B: Backend>(
        &self,
        dataset: &OhlcvDataset,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let batch_size = dataset.len();
        if batch_size == 0 {
            panic!("Cannot convert empty dataset to tensor");
        }

        let sequence_length = dataset.config.sequence_length;
        let num_features = self.config.num_features;

        let mut all_features = Vec::with_capacity(batch_size);
        for seq in &dataset.sequences {
            let features = self.extract_sequence_features(seq);
            all_features.push(features);
        }

        let normalized = self.normalize_features(&all_features);

        // Flatten to 1D for tensor creation
        let flat_data: Vec<f32> = normalized
            .into_iter()
            .flatten()
            .flatten()
            .map(|x| x as f32)
            .collect();

        let shape = Shape::new([batch_size, sequence_length, num_features]);
        let data = TensorData::new(flat_data, shape);
        Tensor::<B, 3>::from_data(data.convert::<f32>(), device)
    }

    /// Convert labels to tensor
    ///
    /// Returns a tensor of shape [batch_size]
    pub fn labels_to_tensor<B: Backend>(&self, labels: &[f64], device: &B::Device) -> Tensor<B, 1> {
        let data: Vec<f32> = labels.iter().map(|&x| x as f32).collect();
        let shape = Shape::new([labels.len()]);
        let tensor_data = TensorData::new(data, shape);
        Tensor::<B, 1>::from_data(tensor_data.convert::<f32>(), device)
    }

    /// Convert classification labels to tensor
    ///
    /// Returns a tensor of shape [batch_size]
    pub fn class_labels_to_tensor<B: Backend>(
        &self,
        labels: &[usize],
        device: &B::Device,
    ) -> Tensor<B, 1, burn::tensor::Int> {
        let data: Vec<i64> = labels.iter().map(|&x| x as i64).collect();
        let shape = Shape::new([labels.len()]);
        let tensor_data = TensorData::new(data, shape);
        Tensor::<B, 1, burn::tensor::Int>::from_data(tensor_data.convert::<i64>(), device)
    }

    /// Convert pre-engineered feature matrix to tensor
    ///
    /// # Arguments
    /// * `features` - Feature matrix of shape [sequence_length, num_features]
    /// * `device` - Target device for the tensor
    ///
    /// Returns a tensor of shape [sequence_length, num_features]
    pub fn features_to_tensor<B: Backend>(
        &self,
        features: &[Vec<f64>],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let normalized = self.normalize_features(&[features.to_vec()]);

        // normalized is Vec<Vec<Vec<f64>>> with outer vec length 1
        let seq_features = &normalized[0];
        let sequence_length = seq_features.len();
        let num_features = seq_features[0].len();

        let flat_data: Vec<f32> = seq_features.iter().flatten().map(|&x| x as f32).collect();

        let shape = Shape::new([sequence_length, num_features]);
        let data = TensorData::new(flat_data, shape);
        Tensor::<B, 2>::from_data(data.convert::<f32>(), device)
    }

    /// Convert multiple pre-engineered feature matrices to a batch tensor
    ///
    /// # Arguments
    /// * `features` - Vec of feature matrices, each of shape [sequence_length, num_features]
    /// * `device` - Target device for the tensor
    ///
    /// Returns a tensor of shape [batch_size, sequence_length, num_features]
    pub fn features_batch_to_tensor<B: Backend>(
        &self,
        features: &[Vec<Vec<f64>>],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let batch_size = features.len();
        if batch_size == 0 {
            panic!("Cannot convert empty feature batch to tensor");
        }

        let sequence_length = features[0].len();
        let num_features = features[0][0].len();

        let normalized = self.normalize_features(features);

        // Flatten to 1D for tensor creation
        let flat_data: Vec<f32> = normalized
            .into_iter()
            .flatten()
            .flatten()
            .map(|x| x as f32)
            .collect();

        let shape = Shape::new([batch_size, sequence_length, num_features]);
        let data = TensorData::new(flat_data, shape);
        Tensor::<B, 3>::from_data(data.convert::<f32>(), device)
    }

    /// Extract features from a sequence
    fn extract_sequence_features(&self, sequence: &[OhlcvCandle]) -> Vec<Vec<f64>> {
        sequence.iter().map(|candle| candle.to_features()).collect()
    }

    /// Extract all features from dataset
    fn extract_all_features(&self, dataset: &OhlcvDataset) -> Vec<Vec<f64>> {
        dataset
            .sequences
            .iter()
            .flat_map(|seq| seq.iter().map(|candle| candle.to_features()))
            .collect()
    }

    /// Normalize features if scaler is present
    fn normalize_features(&self, features: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        if let Some(ref scaler) = self.scaler {
            features
                .iter()
                .map(|seq_features| scaler.transform(seq_features))
                .collect()
        } else {
            features.to_vec()
        }
    }

    /// Check if the converter is fitted (if normalization is enabled)
    pub fn is_fitted(&self) -> bool {
        if let Some(ref scaler) = self.scaler {
            scaler.is_fitted()
        } else {
            true // No normalization, always "fitted"
        }
    }
}

/// Helper functions for creating batches
pub struct BatchIterator<'a> {
    dataset: &'a OhlcvDataset,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> BatchIterator<'a> {
    /// Create a new batch iterator
    pub fn new(dataset: &'a OhlcvDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current_idx: 0,
        }
    }

    /// Get the next batch as indices
    pub fn next_batch(&mut self) -> Option<(Vec<usize>, bool)> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let indices: Vec<usize> = (self.current_idx..end_idx).collect();
        let is_last = end_idx >= self.dataset.len();

        self.current_idx = end_idx;

        Some((indices, is_last))
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }

    /// Get total number of batches
    pub fn num_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
}

/// Create a mini-batch from dataset indices
pub fn create_batch<B: Backend, S: Scaler + Default>(
    dataset: &OhlcvDataset,
    indices: &[usize],
    converter: &TensorConverter<S>,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 1>) {
    let batch_size = indices.len();
    let sequence_length = dataset.config.sequence_length;
    let num_features = converter.config.num_features;

    let mut batch_sequences = Vec::with_capacity(batch_size);
    let mut batch_labels = Vec::with_capacity(batch_size);

    for &idx in indices {
        if let Some((seq, label)) = dataset.get(idx) {
            batch_sequences.push(seq);
            batch_labels.push(label);
        }
    }

    // Convert sequences
    let mut all_features = Vec::with_capacity(batch_size);
    for seq in &batch_sequences {
        let features: Vec<Vec<f64>> = seq.iter().map(|c| c.to_features()).collect();
        all_features.push(features);
    }

    let normalized = if let Some(ref scaler) = converter.scaler {
        all_features
            .iter()
            .map(|seq_features| scaler.transform(seq_features))
            .collect()
    } else {
        all_features
    };

    let flat_data: Vec<f32> = normalized
        .into_iter()
        .flatten()
        .flatten()
        .map(|x| x as f32)
        .collect();

    let shape = Shape::new([batch_size, sequence_length, num_features]);
    let data = TensorData::new(flat_data, shape);
    let inputs = Tensor::<B, 3>::from_data(data.convert::<f32>(), device);

    // Convert labels
    let labels = converter.labels_to_tensor::<B>(&batch_labels, device);

    (inputs, labels)
}

/// Create a mini-batch from pre-engineered feature matrices
///
/// # Arguments
/// * `features` - Slice of feature matrices for the batch
/// * `labels` - Corresponding labels
/// * `converter` - Tensor converter (for normalization)
/// * `device` - Target device
///
/// Returns (inputs, labels) as tensors
pub fn create_batch_with_features<B: Backend, S: Scaler + Default>(
    features: &[Vec<Vec<f64>>],
    labels: &[f64],
    converter: &TensorConverter<S>,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 1>) {
    let inputs = converter.features_batch_to_tensor::<B>(features, device);
    let label_tensor = converter.labels_to_tensor::<B>(labels, device);
    (inputs, label_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::csv_loader::OhlcvCandle;
    use crate::data::dataset::SequenceConfig;
    use crate::preprocessing::normalization::ZScoreScaler;
    use burn::backend::NdArray;
    use chrono::Utc;

    type TestBackend = NdArray;

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
    fn test_tensor_converter_creation() {
        let config = TensorConverterConfig::default();
        let converter = TensorConverter::<ZScoreScaler>::new(config);

        assert!(converter.scaler.is_some());
        assert_eq!(converter.config.num_features, 5);
    }

    #[test]
    fn test_sequence_to_tensor() {
        let candles = create_test_candles(10);
        let config = TensorConverterConfig {
            normalize: false,
            num_features: 5,
        };

        let converter = TensorConverter::<ZScoreScaler>::new(config);
        let device = Default::default();
        let tensor: Tensor<TestBackend, 2> = converter.sequence_to_tensor(&candles, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims[0], 10); // sequence_length
        assert_eq!(shape.dims[1], 5); // num_features (OHLCV)
    }

    #[test]
    fn test_dataset_to_tensor() {
        let candles = create_test_candles(100);
        let seq_config = SequenceConfig {
            sequence_length: 10,
            stride: 10,
            prediction_horizon: 1,
        };

        let dataset = OhlcvDataset::from_candles(candles, seq_config).unwrap();

        let config = TensorConverterConfig {
            normalize: false,
            num_features: 5,
        };

        let converter = TensorConverter::<ZScoreScaler>::new(config);
        let device = Default::default();
        let tensor: Tensor<TestBackend, 3> = converter.dataset_to_tensor(&dataset, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims[0], dataset.len()); // batch_size
        assert_eq!(shape.dims[1], 10); // sequence_length
        assert_eq!(shape.dims[2], 5); // num_features
    }

    #[test]
    fn test_labels_to_tensor() {
        let labels = vec![0.1, -0.2, 0.3, -0.4];
        let config = TensorConverterConfig::default();
        let converter = TensorConverter::<ZScoreScaler>::new(config);
        let device = Default::default();

        let tensor: Tensor<TestBackend, 1> = converter.labels_to_tensor(&labels, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims[0], 4);
    }

    #[test]
    fn test_class_labels_to_tensor() {
        let labels = vec![0, 1, 2, 1, 0];
        let config = TensorConverterConfig::default();
        let converter = TensorConverter::<ZScoreScaler>::new(config);
        let device = Default::default();

        let tensor: Tensor<TestBackend, 1, burn::tensor::Int> =
            converter.class_labels_to_tensor(&labels, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims[0], 5);
    }

    #[test]
    fn test_tensor_converter_with_normalization() {
        let candles = create_test_candles(100);
        let seq_config = SequenceConfig {
            sequence_length: 10,
            stride: 10,
            prediction_horizon: 1,
        };

        let dataset = OhlcvDataset::from_candles(candles, seq_config).unwrap();

        let config = TensorConverterConfig {
            normalize: true,
            num_features: 5,
        };

        let mut converter = TensorConverter::<ZScoreScaler>::new(config);
        assert!(!converter.is_fitted());

        converter.fit(&dataset);
        assert!(converter.is_fitted());

        let device = Default::default();
        let tensor: Tensor<TestBackend, 3> = converter.dataset_to_tensor(&dataset, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims[0], dataset.len());
        assert_eq!(shape.dims[1], 10);
        assert_eq!(shape.dims[2], 5);
    }

    #[test]
    fn test_batch_iterator() {
        let candles = create_test_candles(100);
        let seq_config = SequenceConfig {
            sequence_length: 10,
            stride: 10,
            prediction_horizon: 1,
        };

        let dataset = OhlcvDataset::from_candles(candles, seq_config).unwrap();
        let mut iter = BatchIterator::new(&dataset, 3);

        let total_batches = iter.num_batches();
        assert!(total_batches > 0);

        let mut batch_count = 0;
        while let Some((indices, is_last)) = iter.next_batch() {
            assert!(!indices.is_empty());
            assert!(indices.len() <= 3);
            batch_count += 1;

            if is_last {
                break;
            }
        }

        assert_eq!(batch_count, total_batches);

        // Test reset
        iter.reset();
        let (first_batch, _) = iter.next_batch().unwrap();
        assert_eq!(first_batch[0], 0);
    }

    #[test]
    fn test_create_batch() {
        let candles = create_test_candles(100);
        let seq_config = SequenceConfig {
            sequence_length: 10,
            stride: 10,
            prediction_horizon: 1,
        };

        let dataset = OhlcvDataset::from_candles(candles, seq_config).unwrap();

        let config = TensorConverterConfig {
            normalize: false,
            num_features: 5,
        };

        let converter = TensorConverter::<ZScoreScaler>::new(config);
        let indices = vec![0, 1, 2];
        let device = Default::default();

        let (inputs, labels): (Tensor<TestBackend, 3>, Tensor<TestBackend, 1>) =
            create_batch(&dataset, &indices, &converter, &device);

        let input_shape = inputs.shape();
        assert_eq!(input_shape.dims[0], 3); // batch_size
        assert_eq!(input_shape.dims[1], 10); // sequence_length
        assert_eq!(input_shape.dims[2], 5); // num_features

        let label_shape = labels.shape();
        assert_eq!(label_shape.dims[0], 3);
    }

    #[test]
    fn test_fit_with_features() {
        let config = TensorConverterConfig {
            normalize: true,
            num_features: 10,
        };

        let mut converter = TensorConverter::<ZScoreScaler>::new(config);
        assert!(!converter.is_fitted());

        // Create sample feature matrices
        let features: Vec<Vec<Vec<f64>>> = vec![
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
            vec![
                vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            ],
        ];

        converter.fit_with_features(&features);
        assert!(converter.is_fitted());
    }

    #[test]
    fn test_features_to_tensor() {
        let config = TensorConverterConfig {
            normalize: false,
            num_features: 3,
        };

        let converter = TensorConverter::<ZScoreScaler>::new(config);

        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let device = Default::default();
        let tensor: Tensor<TestBackend, 2> = converter.features_to_tensor(&features, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims[0], 3); // sequence_length
        assert_eq!(shape.dims[1], 3); // num_features
    }

    #[test]
    fn test_features_batch_to_tensor() {
        let config = TensorConverterConfig {
            normalize: false,
            num_features: 4,
        };

        let converter = TensorConverter::<ZScoreScaler>::new(config);

        let features = vec![
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
            vec![vec![9.0, 10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0, 16.0]],
        ];

        let device = Default::default();
        let tensor: Tensor<TestBackend, 3> = converter.features_batch_to_tensor(&features, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims[0], 2); // batch_size
        assert_eq!(shape.dims[1], 2); // sequence_length
        assert_eq!(shape.dims[2], 4); // num_features
    }

    #[test]
    fn test_create_batch_with_features() {
        let config = TensorConverterConfig {
            normalize: false,
            num_features: 3,
        };

        let converter = TensorConverter::<ZScoreScaler>::new(config);

        let features = vec![
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]],
        ];

        let labels = vec![0.5, -0.3];
        let device = Default::default();

        let (inputs, label_tensor): (Tensor<TestBackend, 3>, Tensor<TestBackend, 1>) =
            create_batch_with_features(&features, &labels, &converter, &device);

        let input_shape = inputs.shape();
        assert_eq!(input_shape.dims[0], 2); // batch_size
        assert_eq!(input_shape.dims[1], 2); // sequence_length
        assert_eq!(input_shape.dims[2], 3); // num_features

        let label_shape = label_tensor.shape();
        assert_eq!(label_shape.dims[0], 2);
    }

    #[test]
    fn test_features_with_normalization() {
        let config = TensorConverterConfig {
            normalize: true,
            num_features: 2,
        };

        let mut converter = TensorConverter::<ZScoreScaler>::new(config);

        let train_features = vec![
            vec![vec![10.0, 20.0], vec![30.0, 40.0]],
            vec![vec![50.0, 60.0], vec![70.0, 80.0]],
        ];

        converter.fit_with_features(&train_features);
        assert!(converter.is_fitted());

        let test_features = vec![vec![90.0, 100.0], vec![110.0, 120.0]];

        let device = Default::default();
        let tensor: Tensor<TestBackend, 2> = converter.features_to_tensor(&test_features, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims[0], 2); // sequence_length
        assert_eq!(shape.dims[1], 2); // num_features
    }
}
