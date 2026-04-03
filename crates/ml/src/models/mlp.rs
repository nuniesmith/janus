//! MLP-based signal classifier for trading signals
//!
//! This module implements a multi-layer perceptron (MLP) network for
//! classifying trading signals (buy/sell/hold) based on market features.
//!
//! # Architecture
//!
//! The MLP classifier consists of:
//! - Input layer
//! - Multiple hidden layers with activation functions
//! - Dropout for regularization
//! - Batch normalization (optional)
//! - Output layer with softmax activation
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::models::{MlpConfig, MlpClassifier, Model};
//! use janus_ml::backend::BackendDevice;
//!
//! let config = MlpConfig::new(50, vec![128, 64], 3)
//!     .with_dropout(0.3)
//!     .with_batch_norm(true);
//!
//! let device = BackendDevice::cpu();
//! let model = MlpClassifier::new(config, device);
//! ```

use std::path::Path;

use burn_core::module::RunningState;
use burn_core::tensor::{Tensor, TensorData, activation, backend::Backend};
use burn_nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig};
use serde::{Deserialize, Serialize};

use crate::backend::{BackendDevice, CpuBackend};
use crate::error::{MLError, Result};

use super::{Model, ModelMetadata, ModelRecord, SerializedTensor, WeightMap};

/// Configuration for MLP classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpConfig {
    /// Input feature dimension
    pub input_size: usize,

    /// Hidden layer sizes
    pub hidden_sizes: Vec<usize>,

    /// Output dimension (number of classes)
    pub output_size: usize,

    /// Dropout probability (0.0 - 1.0)
    pub dropout: f64,

    /// Use batch normalization
    pub batch_norm: bool,

    /// Activation function type
    pub activation: ActivationType,
}

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationType {
    Relu,
    Tanh,
    Sigmoid,
    Gelu,
}

impl MlpConfig {
    /// Create a new MLP configuration
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_sizes,
            output_size,
            dropout: 0.3,
            batch_norm: true,
            activation: ActivationType::Relu,
        }
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set batch normalization
    pub fn with_batch_norm(mut self, batch_norm: bool) -> Self {
        self.batch_norm = batch_norm;
        self
    }

    /// Set activation function
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self::new(50, vec![128, 64], 3)
    }
}

// ---------------------------------------------------------------------------
// Weight serialisation helpers
// ---------------------------------------------------------------------------

/// Extract a 2-D tensor into a [`SerializedTensor`].
fn serialize_tensor_2d<B: Backend>(name: &str, t: &Tensor<B, 2>) -> SerializedTensor {
    let shape: Vec<usize> = t.dims().to_vec();
    let data: Vec<f32> = t.to_data().to_vec().unwrap();
    SerializedTensor {
        name: name.to_string(),
        shape,
        data,
    }
}

/// Extract a 1-D tensor into a [`SerializedTensor`].
fn serialize_tensor_1d<B: Backend>(name: &str, t: &Tensor<B, 1>) -> SerializedTensor {
    let shape: Vec<usize> = t.dims().to_vec();
    let data: Vec<f32> = t.to_data().to_vec().unwrap();
    SerializedTensor {
        name: name.to_string(),
        shape,
        data,
    }
}

/// Restore a 2-D tensor from a [`SerializedTensor`].
fn deserialize_tensor_2d(
    st: &SerializedTensor,
    device: &<CpuBackend as Backend>::Device,
) -> Tensor<CpuBackend, 2> {
    let shape: [usize; 2] = [st.shape[0], st.shape[1]];
    Tensor::<CpuBackend, 2>::from_data(TensorData::new(st.data.clone(), shape), device)
}

/// Restore a 1-D tensor from a [`SerializedTensor`].
fn deserialize_tensor_1d(
    st: &SerializedTensor,
    device: &<CpuBackend as Backend>::Device,
) -> Tensor<CpuBackend, 1> {
    let shape: [usize; 1] = [st.shape[0]];
    Tensor::<CpuBackend, 1>::from_data(TensorData::new(st.data.clone(), shape), device)
}

/// Extract all weights from a [`Linear`] layer.
fn extract_linear_weights<B: Backend>(prefix: &str, linear: &Linear<B>) -> Vec<SerializedTensor> {
    let mut out = Vec::new();
    out.push(serialize_tensor_2d(
        &format!("{}.weight", prefix),
        &linear.weight.val(),
    ));
    if let Some(ref bias) = linear.bias {
        out.push(serialize_tensor_1d(
            &format!("{}.bias", prefix),
            &bias.val(),
        ));
    }
    out
}

/// Restore weights into a [`Linear`] layer (CPU backend).
fn restore_linear_weights(
    prefix: &str,
    linear: &mut Linear<CpuBackend>,
    map: &std::collections::HashMap<String, &SerializedTensor>,
    device: &<CpuBackend as Backend>::Device,
) {
    let weight_key = format!("{}.weight", prefix);
    if let Some(st) = map.get(&weight_key) {
        linear.weight = burn_core::module::Param::initialized(
            linear.weight.id,
            deserialize_tensor_2d(st, device),
        );
    }
    let bias_key = format!("{}.bias", prefix);
    if let (Some(bias_param), Some(st)) = (&mut linear.bias, map.get(&bias_key)) {
        *bias_param =
            burn_core::module::Param::initialized(bias_param.id, deserialize_tensor_1d(st, device));
    }
}

/// Extract all weights from a [`BatchNorm`] layer.
fn extract_batchnorm_weights<B: Backend>(prefix: &str, bn: &BatchNorm<B>) -> Vec<SerializedTensor> {
    let mut out = Vec::new();
    out.push(serialize_tensor_1d(
        &format!("{}.gamma", prefix),
        &bn.gamma.val(),
    ));
    out.push(serialize_tensor_1d(
        &format!("{}.beta", prefix),
        &bn.beta.val(),
    ));
    // RunningState exposes its tensor via `.value()` (not `.val()`)
    out.push(serialize_tensor_1d(
        &format!("{}.running_mean", prefix),
        &bn.running_mean.value(),
    ));
    out.push(serialize_tensor_1d(
        &format!("{}.running_var", prefix),
        &bn.running_var.value(),
    ));
    out
}

/// Restore weights into a [`BatchNorm`] layer (CPU backend).
fn restore_batchnorm_weights(
    prefix: &str,
    bn: &mut BatchNorm<CpuBackend>,
    map: &std::collections::HashMap<String, &SerializedTensor>,
    device: &<CpuBackend as Backend>::Device,
) {
    let gamma_key = format!("{}.gamma", prefix);
    if let Some(st) = map.get(&gamma_key) {
        bn.gamma =
            burn_core::module::Param::initialized(bn.gamma.id, deserialize_tensor_1d(st, device));
    }
    let beta_key = format!("{}.beta", prefix);
    if let Some(st) = map.get(&beta_key) {
        bn.beta =
            burn_core::module::Param::initialized(bn.beta.id, deserialize_tensor_1d(st, device));
    }
    // RunningState fields are replaced via `RunningState::new()` since the
    // inner `id` / `value` fields are private.
    let rm_key = format!("{}.running_mean", prefix);
    if let Some(st) = map.get(&rm_key) {
        bn.running_mean = RunningState::new(deserialize_tensor_1d(st, device));
    }
    let rv_key = format!("{}.running_var", prefix);
    if let Some(st) = map.get(&rv_key) {
        bn.running_var = RunningState::new(deserialize_tensor_1d(st, device));
    }
}

// ---------------------------------------------------------------------------
// Hidden layer
// ---------------------------------------------------------------------------

/// Hidden layer of the MLP
#[derive(Debug)]
struct MlpLayer<B: Backend> {
    linear: Linear<B>,
    batch_norm: Option<BatchNorm<B>>,
    dropout: Dropout,
}

impl<B: Backend> MlpLayer<B> {
    fn new(
        input_size: usize,
        output_size: usize,
        dropout: f64,
        use_batch_norm: bool,
        device: &B::Device,
    ) -> Self {
        let linear = LinearConfig::new(input_size, output_size).init(device);

        let batch_norm = if use_batch_norm {
            Some(BatchNormConfig::new(output_size).init(device))
        } else {
            None
        };

        let dropout = DropoutConfig::new(dropout).init();

        Self {
            linear,
            batch_norm,
            dropout,
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.linear.forward(x);

        if let Some(ref bn) = self.batch_norm {
            x = bn.forward(x);
        }

        x = activation::relu(x);
        x = self.dropout.forward(x);

        x
    }
}

// ---------------------------------------------------------------------------
// MlpClassifier
// ---------------------------------------------------------------------------

/// MLP-based signal classifier
#[derive(Debug)]
pub struct MlpClassifier<B: Backend> {
    /// Hidden layers
    hidden_layers: Vec<MlpLayer<B>>,

    /// Output layer
    output: Linear<B>,

    /// Model configuration
    config: MlpConfig,

    /// Model metadata
    metadata: ModelMetadata,

    /// Backend device
    device: BackendDevice,
}

impl<B: Backend> MlpClassifier<B> {
    /// Create a new MLP classifier with the given configuration
    pub fn new_with_backend(config: MlpConfig, device: BackendDevice) -> Self {
        let mut hidden_layers = Vec::new();

        // Get device for initialization
        let device_ref = &B::Device::default();

        // Create hidden layers
        let mut prev_size = config.input_size;
        for &hidden_size in &config.hidden_sizes {
            let layer = MlpLayer::new(
                prev_size,
                hidden_size,
                config.dropout,
                config.batch_norm,
                device_ref,
            );
            hidden_layers.push(layer);
            prev_size = hidden_size;
        }

        // Create output layer
        let output = LinearConfig::new(prev_size, config.output_size).init(device_ref);

        // Create metadata
        let metadata = ModelMetadata::new(
            "mlp_classifier".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
            "mlp".to_string(),
            config.input_size,
            config.output_size,
            device.backend_type().to_string(),
        );

        Self {
            hidden_layers,
            output,
            config,
            metadata,
            device,
        }
    }

    /// Forward pass through the MLP on a **2-D** input.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `[batch_size, input_size]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, output_size]` with logits
    pub fn forward_2d(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        // Pass through hidden layers
        for layer in &self.hidden_layers {
            x = layer.forward(x);
        }

        // Output layer (logits, no activation)
        self.output.forward(x)
    }

    /// Forward pass through the MLP network on a **3-D** input.
    ///
    /// Takes the last timestep from the sequence dimension before feeding
    /// through the MLP.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `[batch_size, seq_len, input_size]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, output_size]` with logits
    pub fn forward_with_backend(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // Take the last timestep from sequence
        // input: [batch_size, seq_len, input_size]
        let [batch_size, seq_len, input_size] = input.dims();
        let x = input.slice([0..batch_size, (seq_len - 1)..seq_len, 0..input_size]);
        let x = x.reshape([batch_size, input_size]);

        self.forward_2d(x)
    }

    /// Forward pass with softmax activation
    ///
    /// Returns class probabilities instead of logits
    pub fn forward_with_softmax(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward_with_backend(input);
        activation::softmax(logits, 1)
    }

    /// Predict class labels
    ///
    /// Returns the class index with highest probability
    pub fn predict(&self, input: Tensor<B, 3>) -> Tensor<B, 1, burn_core::tensor::Int> {
        let probs = self.forward_with_softmax(input);
        probs.argmax(1).squeeze::<1>()
    }

    // -- backward-compatible convenience methods ---------------------------------

    /// Forward pass returning `Result` (compatible with `training.rs` API).
    ///
    /// Accepts a 3-D tensor `[batch, seq_len, features]`, takes the last
    /// timestep, and runs through the MLP.
    pub fn forward(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 2>> {
        Ok(self.forward_with_backend(input))
    }

    /// Get model name.
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get model configuration.
    pub fn config(&self) -> &MlpConfig {
        &self.config
    }

    /// Get mutable reference to metadata
    pub fn metadata_mut(&mut self) -> &mut ModelMetadata {
        &mut self.metadata
    }
}

// ---------------------------------------------------------------------------
// Weight extraction / restoration (CPU backend only)
// ---------------------------------------------------------------------------

impl MlpClassifier<CpuBackend> {
    /// Collect every learnable tensor into a flat [`WeightMap`].
    fn extract_weights(&self) -> WeightMap {
        let mut weights = Vec::new();

        for (i, layer) in self.hidden_layers.iter().enumerate() {
            let prefix = format!("hidden_layers.{}", i);

            // Linear
            weights.extend(extract_linear_weights(
                &format!("{}.linear", prefix),
                &layer.linear,
            ));

            // BatchNorm (optional)
            if let Some(ref bn) = layer.batch_norm {
                weights.extend(extract_batchnorm_weights(
                    &format!("{}.batch_norm", prefix),
                    bn,
                ));
            }
        }

        weights.extend(extract_linear_weights("output", &self.output));

        weights
    }

    /// Overwrite this model's parameters from a [`WeightMap`].
    fn restore_weights(&mut self, weight_map: &WeightMap) {
        let device = <CpuBackend as Backend>::Device::default();
        let map: std::collections::HashMap<String, &SerializedTensor> =
            weight_map.iter().map(|st| (st.name.clone(), st)).collect();

        for (i, layer) in self.hidden_layers.iter_mut().enumerate() {
            let prefix = format!("hidden_layers.{}", i);

            restore_linear_weights(
                &format!("{}.linear", prefix),
                &mut layer.linear,
                &map,
                &device,
            );

            if let Some(ref mut bn) = layer.batch_norm {
                restore_batchnorm_weights(&format!("{}.batch_norm", prefix), bn, &map, &device);
            }
        }

        restore_linear_weights("output", &mut self.output, &map, &device);
    }

    /// Public entry-point for restoring weights from a [`WeightMap`].
    ///
    /// This is used by the [`super::convert`] module to load weights
    /// extracted from a `TrainableMlp` into an inference `MlpClassifier`.
    pub fn apply_weight_map(&mut self, weight_map: &WeightMap) {
        self.restore_weights(weight_map);
    }

    /// Public read-only access to the model's current weights.
    ///
    /// This is used by [`super::convert::soft_update_mlp_target`] to read
    /// the target network's weights before blending with the online network.
    pub fn extract_weights_pub(&self) -> WeightMap {
        self.extract_weights()
    }
}

// ---------------------------------------------------------------------------
// Model trait implementation (CPU)
// ---------------------------------------------------------------------------

impl Model for MlpClassifier<CpuBackend> {
    type Config = MlpConfig;

    fn new(config: Self::Config, device: BackendDevice) -> Self {
        Self::new_with_backend(config, device)
    }

    fn forward(&self, input: Tensor<CpuBackend, 3>) -> Result<Tensor<CpuBackend, 2>> {
        Ok(self.forward_with_backend(input))
    }

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Serialise weights
        let weights = self.extract_weights();
        let weights_json =
            serde_json::to_vec(&weights).map_err(|e| MLError::io_error(e.to_string()))?;

        // Create model record
        let config_json = serde_json::to_string(&self.config)?;
        let record = ModelRecord::new(self.metadata.clone(), config_json, weights_json);

        // Save to file using postcard (migrated from bincode v2)
        let bytes = postcard::to_allocvec(&record)?;
        std::fs::write(path, bytes)
            .map_err(|e| MLError::io_error(format!("Failed to write file: {}", e)))?;

        tracing::info!(
            "Saved MLP model to {:?} ({} weight tensors)",
            path,
            weights.len()
        );
        Ok(())
    }

    fn load<P: AsRef<Path>>(path: P, device: BackendDevice) -> Result<Self> {
        let path = path.as_ref();

        // Load from file
        let bytes = std::fs::read(path)
            .map_err(|e| MLError::io_error(format!("Failed to read file: {}", e)))?;
        let record: ModelRecord<Vec<u8>> = postcard::from_bytes(&bytes)?;

        // Check compatibility
        if !record.is_compatible() {
            return Err(MLError::invalid_config(format!(
                "Incompatible model format version: {}",
                record.format_version
            )));
        }

        // Deserialise config
        let config: MlpConfig = serde_json::from_str(&record.config)?;

        // Create fresh model from config
        let mut model = Self::new(config, device);

        // Restore weights if the blob is non-empty
        if !record.weights.is_empty() {
            let weight_map: WeightMap = serde_json::from_slice(&record.weights).map_err(|e| {
                MLError::model_load(format!("Failed to deserialise weights: {}", e))
            })?;
            model.restore_weights(&weight_map);
            tracing::info!(
                "Loaded MLP model from {:?} ({} weight tensors restored)",
                path,
                weight_map.len()
            );
        } else {
            tracing::info!(
                "Loaded MLP model config from {:?} (no weights in file, using fresh init)",
                path
            );
        }

        Ok(model)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn device(&self) -> &BackendDevice {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::TensorData;

    #[test]
    fn test_mlp_config() {
        let config = MlpConfig::new(50, vec![128, 64], 3);
        assert_eq!(config.input_size, 50);
        assert_eq!(config.hidden_sizes, vec![128, 64]);
        assert_eq!(config.output_size, 3);
        assert_eq!(config.dropout, 0.3);
        assert!(config.batch_norm);
    }

    #[test]
    fn test_mlp_config_builder() {
        let config = MlpConfig::new(30, vec![64, 32], 2)
            .with_dropout(0.5)
            .with_batch_norm(false)
            .with_activation(ActivationType::Tanh);

        assert_eq!(config.dropout, 0.5);
        assert!(!config.batch_norm);
        assert_eq!(config.activation, ActivationType::Tanh);
    }

    #[test]
    fn test_mlp_creation() {
        let config = MlpConfig::new(50, vec![128, 64], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        assert_eq!(model.config.input_size, 50);
        assert_eq!(model.config.output_size, 3);
        assert_eq!(model.hidden_layers.len(), 2);
        assert_eq!(model.metadata.model_type, "mlp");
    }

    #[test]
    fn test_mlp_forward() {
        let config = MlpConfig::new(10, vec![16], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        // Create dummy input: [batch=2, seq_len=5, features=10]
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );

        let output = model.forward(input).unwrap();

        // Output should be [batch=2, classes=3]
        assert_eq!(output.dims(), [2, 3]);
    }

    #[test]
    fn test_mlp_forward_2d() {
        let config = MlpConfig::new(10, vec![16], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        // Direct 2-D input: [batch=4, features=10]
        let input_data: Vec<f32> = (0..40).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(input_data, [4, 10]),
            &Default::default(),
        );

        let output = model.forward_2d(input);
        assert_eq!(output.dims(), [4, 3]);
    }

    #[test]
    fn test_mlp_forward_with_softmax() {
        let config = MlpConfig::new(10, vec![16], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        // Create dummy input: [batch=2, seq_len=5, features=10]
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );

        let probs = model.forward_with_softmax(input);

        // Output should be [batch=2, classes=3]
        assert_eq!(probs.dims(), [2, 3]);

        // Probabilities should sum to ~1.0 for each sample
        let probs_data = probs.into_data().to_vec::<f32>().unwrap();
        let sample1_sum: f32 = probs_data[0..3].iter().sum();
        let sample2_sum: f32 = probs_data[3..6].iter().sum();

        assert!((sample1_sum - 1.0).abs() < 1e-5);
        assert!((sample2_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mlp_predict() {
        let config = MlpConfig::new(10, vec![16], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        // Create dummy input: [batch=2, seq_len=5, features=10]
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );

        let predictions = model.predict(input);

        // Predictions should be [batch=2]
        assert_eq!(predictions.dims(), [2]);

        // Predictions should be in range [0, num_classes)
        let pred_data = predictions.into_data().to_vec::<i64>().unwrap();
        for pred in pred_data {
            assert!(
                (0..3).contains(&pred),
                "Prediction {} out of range [0, 3)",
                pred
            );
        }
    }

    #[test]
    fn test_mlp_metadata() {
        let config = MlpConfig::new(50, vec![128, 64], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        let metadata = model.metadata();
        assert_eq!(metadata.name, "mlp_classifier");
        assert_eq!(metadata.model_type, "mlp");
        assert_eq!(metadata.input_size, 50);
        assert_eq!(metadata.output_size, 3);
        assert_eq!(metadata.backend, "cpu");
    }

    #[test]
    fn test_mlp_save_load_config_only() {
        // Verify round-trip of config when weights blob is empty (legacy compat)
        let config = MlpConfig::new(10, vec![16], 3);
        let device = BackendDevice::cpu();

        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_mlp_config_only.bin");

        // Build a record with empty weights
        let config_json = serde_json::to_string(&config).unwrap();
        let metadata = ModelMetadata::new(
            "mlp_classifier".into(),
            env!("CARGO_PKG_VERSION").into(),
            "mlp".into(),
            config.input_size,
            config.output_size,
            "cpu".into(),
        );
        let record = ModelRecord::new(metadata, config_json, Vec::<u8>::new());
        let bytes = postcard::to_allocvec(&record).unwrap();
        std::fs::write(&model_path, bytes).unwrap();

        let loaded = MlpClassifier::<CpuBackend>::load(&model_path, device).unwrap();
        assert_eq!(loaded.config.input_size, 10);
        assert_eq!(loaded.config.hidden_sizes, vec![16]);
        assert_eq!(loaded.config.output_size, 3);

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_mlp_save_load_with_weights() {
        let config = MlpConfig::new(10, vec![16], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config.clone(), device.clone());

        // Run a forward pass to make sure the model is initialised
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [2, 5, 10]),
            &Default::default(),
        );
        let original_output = model.forward_with_backend(input);
        let original_vals: Vec<f32> = original_output.to_data().to_vec().unwrap();

        // Save
        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_mlp_weights.bin");
        <MlpClassifier<CpuBackend> as Model>::save(&model, &model_path).unwrap();

        // Load
        let loaded = MlpClassifier::<CpuBackend>::load(&model_path, device).unwrap();

        // The loaded model should produce the same output
        let input2 = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );
        let loaded_output = loaded.forward_with_backend(input2);
        let loaded_vals: Vec<f32> = loaded_output.to_data().to_vec().unwrap();

        assert_eq!(original_vals.len(), loaded_vals.len());
        for (a, b) in original_vals.iter().zip(loaded_vals.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Weight restoration mismatch: {} vs {}",
                a,
                b
            );
        }

        // Verify configuration
        assert_eq!(loaded.config.input_size, config.input_size);
        assert_eq!(loaded.config.hidden_sizes, config.hidden_sizes);
        assert_eq!(loaded.config.output_size, config.output_size);

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_mlp_save_load_with_weights_no_batchnorm() {
        let config = MlpConfig::new(10, vec![16], 3).with_batch_norm(false);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config.clone(), device.clone());

        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [2, 5, 10]),
            &Default::default(),
        );
        let original_output = model.forward_with_backend(input);
        let original_vals: Vec<f32> = original_output.to_data().to_vec().unwrap();

        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_mlp_weights_nobn.bin");
        <MlpClassifier<CpuBackend> as Model>::save(&model, &model_path).unwrap();

        let loaded = MlpClassifier::<CpuBackend>::load(&model_path, device).unwrap();

        let input2 = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );
        let loaded_output = loaded.forward_with_backend(input2);
        let loaded_vals: Vec<f32> = loaded_output.to_data().to_vec().unwrap();

        assert_eq!(original_vals.len(), loaded_vals.len());
        for (a, b) in original_vals.iter().zip(loaded_vals.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Weight restoration mismatch: {} vs {}",
                a,
                b
            );
        }

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_mlp_model_trait() {
        let config = MlpConfig::new(20, vec![32], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        assert_eq!(model.input_size(), 20);
        assert_eq!(model.output_size(), 3);
        assert_eq!(model.name(), "mlp_classifier");
        assert!(!model.is_gpu());
    }

    #[test]
    fn test_mlp_no_batch_norm() {
        let config = MlpConfig::new(10, vec![16], 3).with_batch_norm(false);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        // Create dummy input
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );

        let output = model.forward(input).unwrap();
        assert_eq!(output.dims(), [2, 3]);
    }

    /// Helper to check that the `Model` trait default helpers delegate
    /// correctly through `ModelMetadata`.
    #[test]
    fn test_model_trait_defaults() {
        use super::super::Model as ModelTrait;

        let config = MlpConfig::new(10, vec![16], 3);
        let device = BackendDevice::cpu();
        let model = MlpClassifier::<CpuBackend>::new(config, device);

        assert_eq!(ModelTrait::input_size(&model), 10);
        assert_eq!(ModelTrait::output_size(&model), 3);
        assert_eq!(ModelTrait::name(&model), "mlp_classifier");
        assert!(!ModelTrait::is_gpu(&model));
    }
}
