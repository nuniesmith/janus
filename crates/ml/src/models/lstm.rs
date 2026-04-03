//! LSTM-based price predictor for time series forecasting
//!
//! This module implements a multi-layer LSTM network for predicting
//! future price movements based on historical market data features.
//!
//! # Architecture
//!
//! The LSTM predictor consists of:
//! - Multiple stacked LSTM layers (unidirectional or bidirectional)
//! - Dropout for regularization
//! - Output projection layer
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::models::{LstmConfig, LstmPredictor, Model};
//! use janus_ml::backend::BackendDevice;
//!
//! let config = LstmConfig::new(50, 64, 1)
//!     .with_num_layers(2)
//!     .with_dropout(0.2);
//!
//! let device = BackendDevice::cpu();
//! let model = LstmPredictor::new(config, device);
//! ```

use std::path::Path;

use burn_core::tensor::{Tensor, TensorData, backend::Backend};
use burn_nn::{
    Dropout, DropoutConfig, Linear, LinearConfig,
    lstm::{BiLstm, BiLstmConfig, Lstm, LstmConfig as BurnLstmConfig},
};
use serde::{Deserialize, Serialize};

use crate::backend::{BackendDevice, CpuBackend};
use crate::error::{MLError, Result};

use super::{Model, ModelMetadata, ModelRecord, SerializedTensor, WeightMap};

/// Configuration for LSTM predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmConfig {
    /// Input feature dimension
    pub input_size: usize,

    /// Hidden state dimension
    pub hidden_size: usize,

    /// Output dimension (typically 1 for price prediction)
    pub output_size: usize,

    /// Number of LSTM layers
    pub num_layers: usize,

    /// Dropout probability (0.0 - 1.0)
    pub dropout: f64,

    /// Use bidirectional LSTM
    pub bidirectional: bool,

    /// Batch first format
    pub batch_first: bool,
}

impl LstmConfig {
    /// Create a new LSTM configuration
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            num_layers: 2,
            dropout: 0.2,
            bidirectional: false,
            batch_first: true,
        }
    }

    /// Set the number of LSTM layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable bidirectional LSTM
    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Set batch_first format
    pub fn with_batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    /// The effective output size of the LSTM stack (accounts for bidirectional).
    pub fn lstm_output_size(&self) -> usize {
        self.hidden_size * if self.bidirectional { 2 } else { 1 }
    }
}

impl Default for LstmConfig {
    fn default() -> Self {
        Self::new(50, 64, 1)
    }
}

// ---------------------------------------------------------------------------
// LstmLayerKind — enum wrapping unidirectional / bidirectional LSTM
// ---------------------------------------------------------------------------

/// Wrapper enum so we can store both `Lstm<B>` and `BiLstm<B>` in the same
/// `Vec` and forward-pass through them uniformly.
#[derive(Debug)]
enum LstmLayerKind<B: Backend> {
    Unidirectional(Box<Lstm<B>>),
    Bidirectional(Box<BiLstm<B>>),
}

impl<B: Backend> LstmLayerKind<B> {
    /// Run the layer's forward pass.
    ///
    /// Returns the batched hidden states tensor of shape
    /// `[batch_size, seq_len, output_features]` where `output_features` is
    /// `hidden_size` for unidirectional and `hidden_size * 2` for bidirectional.
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Unidirectional(lstm) => lstm.forward(input, None).0,
            Self::Bidirectional(bilstm) => bilstm.forward(input, None).0,
        }
    }
}

// ---------------------------------------------------------------------------
// LstmPredictor
// ---------------------------------------------------------------------------

/// LSTM-based price predictor
#[derive(Debug)]
pub struct LstmPredictor<B: Backend> {
    /// LSTM layers (may be uni- or bidirectional)
    lstm_layers: Vec<LstmLayerKind<B>>,

    /// Dropout layer
    dropout: Dropout,

    /// Output projection layer
    output: Linear<B>,

    /// Model configuration
    config: LstmConfig,

    /// Model metadata
    metadata: ModelMetadata,

    /// Backend device
    device: BackendDevice,
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

/// Gate names in a Burn 0.19 [`Lstm`] layer, in canonical order.
const LSTM_GATE_NAMES: &[&str] = &["input_gate", "forget_gate", "output_gate", "cell_gate"];

/// Helper: extract weights from a single unidirectional `Lstm` core.
fn extract_uni_lstm_weights<B: Backend>(prefix: &str, lstm: &Lstm<B>) -> Vec<SerializedTensor> {
    let gates = [
        (&lstm.input_gate, LSTM_GATE_NAMES[0]),
        (&lstm.forget_gate, LSTM_GATE_NAMES[1]),
        (&lstm.output_gate, LSTM_GATE_NAMES[2]),
        (&lstm.cell_gate, LSTM_GATE_NAMES[3]),
    ];

    let mut out = Vec::new();
    for (gate, name) in &gates {
        let gate_prefix = format!("{}.{}", prefix, name);
        out.extend(extract_linear_weights(
            &format!("{}.input_transform", gate_prefix),
            &gate.input_transform,
        ));
        out.extend(extract_linear_weights(
            &format!("{}.hidden_transform", gate_prefix),
            &gate.hidden_transform,
        ));
    }
    out
}

/// Helper: restore weights into a single unidirectional `Lstm` core.
fn restore_uni_lstm_weights(
    prefix: &str,
    lstm: &mut Lstm<CpuBackend>,
    map: &std::collections::HashMap<String, &SerializedTensor>,
    device: &<CpuBackend as Backend>::Device,
) {
    let gates: Vec<(&mut burn_nn::GateController<CpuBackend>, &str)> = vec![
        (&mut lstm.input_gate, LSTM_GATE_NAMES[0]),
        (&mut lstm.forget_gate, LSTM_GATE_NAMES[1]),
        (&mut lstm.output_gate, LSTM_GATE_NAMES[2]),
        (&mut lstm.cell_gate, LSTM_GATE_NAMES[3]),
    ];

    for (gate, name) in gates {
        let gate_prefix = format!("{}.{}", prefix, name);
        restore_linear_weights(
            &format!("{}.input_transform", gate_prefix),
            &mut gate.input_transform,
            map,
            device,
        );
        restore_linear_weights(
            &format!("{}.hidden_transform", gate_prefix),
            &mut gate.hidden_transform,
            map,
            device,
        );
    }
}

/// Extract all weights from a [`LstmLayerKind`].
fn extract_layer_weights<B: Backend>(
    prefix: &str,
    layer: &LstmLayerKind<B>,
) -> Vec<SerializedTensor> {
    match layer {
        LstmLayerKind::Unidirectional(lstm) => extract_uni_lstm_weights(prefix, lstm),
        LstmLayerKind::Bidirectional(bilstm) => {
            let mut out = Vec::new();
            out.extend(extract_uni_lstm_weights(
                &format!("{}.forward", prefix),
                &bilstm.forward,
            ));
            out.extend(extract_uni_lstm_weights(
                &format!("{}.reverse", prefix),
                &bilstm.reverse,
            ));
            out
        }
    }
}

/// Restore all weights into a [`LstmLayerKind`] (CPU backend).
fn restore_layer_weights(
    prefix: &str,
    layer: &mut LstmLayerKind<CpuBackend>,
    map: &std::collections::HashMap<String, &SerializedTensor>,
    device: &<CpuBackend as Backend>::Device,
) {
    match layer {
        LstmLayerKind::Unidirectional(lstm) => {
            restore_uni_lstm_weights(prefix, lstm, map, device);
        }
        LstmLayerKind::Bidirectional(bilstm) => {
            restore_uni_lstm_weights(
                &format!("{}.forward", prefix),
                &mut bilstm.forward,
                map,
                device,
            );
            restore_uni_lstm_weights(
                &format!("{}.reverse", prefix),
                &mut bilstm.reverse,
                map,
                device,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// LstmPredictor implementation
// ---------------------------------------------------------------------------

impl<B: Backend> LstmPredictor<B> {
    /// Get number of parameters (approximate count of layer objects).
    pub fn num_params(&self) -> usize {
        self.lstm_layers.len() + 2 // LSTMs + dropout + output
    }

    /// Create a new LSTM predictor (inherent constructor, compatible with
    /// `training.rs` which calls `LstmPredictor::new(config, device)`).
    pub fn new(config: LstmConfig, device: BackendDevice) -> Self {
        Self::new_with_backend(config, device)
    }

    /// Create a new LSTM predictor with the given configuration.
    pub fn new_with_backend(config: LstmConfig, device: BackendDevice) -> Self {
        let mut lstm_layers = Vec::new();

        // Get device for initialization
        let device_ref = &B::Device::default();

        // Create LSTM layers
        for i in 0..config.num_layers {
            let input_size = if i == 0 {
                config.input_size
            } else {
                // After the first layer, the input size equals the output size
                // of the previous layer (hidden_size for uni, hidden_size*2 for bi).
                config.lstm_output_size()
            };

            let layer = if config.bidirectional {
                let bi_config = BiLstmConfig::new(input_size, config.hidden_size, true);
                LstmLayerKind::Bidirectional(Box::new(bi_config.init(device_ref)))
            } else {
                let uni_config = BurnLstmConfig::new(input_size, config.hidden_size, true);
                LstmLayerKind::Unidirectional(Box::new(uni_config.init(device_ref)))
            };
            lstm_layers.push(layer);
        }

        // Create dropout
        let dropout_config = DropoutConfig::new(config.dropout);
        let dropout = dropout_config.init();

        // Create output projection
        let output_config = LinearConfig::new(config.lstm_output_size(), config.output_size);
        let output = output_config.init(device_ref);

        // Create metadata
        let metadata = ModelMetadata::new(
            "lstm_predictor".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
            "lstm".to_string(),
            config.input_size,
            config.output_size,
            device.backend_type().to_string(),
        );

        Self {
            lstm_layers,
            dropout,
            output,
            config,
            metadata,
            device,
        }
    }

    /// Forward pass through the LSTM network (generic over backends).
    ///
    /// # Arguments
    /// * `input` – Tensor of shape `[batch_size, seq_len, input_size]`
    ///
    /// # Returns
    /// Tensor of shape `[batch_size, output_size]`
    pub fn forward_with_backend(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let mut x = input;

        // Pass through LSTM layers
        for (i, layer) in self.lstm_layers.iter().enumerate() {
            x = layer.forward(x);

            // Apply dropout between layers (except last)
            if i < self.lstm_layers.len() - 1 {
                x = self.dropout.forward(x);
            }
        }

        // Take the last timestep output
        // x: [batch_size, seq_len, hidden_size * directions]
        let [batch_size, seq_len, feature_dim] = x.dims();
        let last_output = x.slice([0..batch_size, (seq_len - 1)..seq_len, 0..feature_dim]);
        let last_output = last_output.reshape([batch_size, feature_dim]);

        // Project to output
        self.output.forward(last_output)
    }

    // -- backward-compatible convenience methods ---------------------------------

    /// Forward pass returning a `Result` (compatible with `training.rs` API).
    pub fn forward(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 2>> {
        Ok(self.forward_with_backend(input))
    }

    /// Get model name.
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get model configuration.
    pub fn config(&self) -> &LstmConfig {
        &self.config
    }

    /// Get mutable reference to metadata.
    pub fn metadata_mut(&mut self) -> &mut ModelMetadata {
        &mut self.metadata
    }
}

// ---------------------------------------------------------------------------
// Weight extraction / restoration + inherent save/load (CPU backend only)
// ---------------------------------------------------------------------------

impl LstmPredictor<CpuBackend> {
    /// Save the model (config + weights) to a file.
    ///
    /// This is an inherent method so that `training.rs` can call
    /// `self.model.save(path)` without importing the `Model` trait.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        <Self as Model>::save(self, path)
    }

    /// Load a model (config + weights) from a file.
    pub fn load<P: AsRef<Path>>(path: P, device: BackendDevice) -> Result<Self> {
        <Self as Model>::load(path, device)
    }

    /// Collect every learnable tensor into a flat [`WeightMap`].
    fn extract_weights(&self) -> WeightMap {
        let mut weights = Vec::new();

        for (i, layer) in self.lstm_layers.iter().enumerate() {
            weights.extend(extract_layer_weights(&format!("lstm_layers.{}", i), layer));
        }

        weights.extend(extract_linear_weights("output", &self.output));

        weights
    }

    /// Overwrite this model's parameters from a [`WeightMap`].
    fn restore_weights(&mut self, weight_map: &WeightMap) {
        let device = <CpuBackend as Backend>::Device::default();
        let map: std::collections::HashMap<String, &SerializedTensor> =
            weight_map.iter().map(|st| (st.name.clone(), st)).collect();

        for (i, layer) in self.lstm_layers.iter_mut().enumerate() {
            restore_layer_weights(&format!("lstm_layers.{}", i), layer, &map, &device);
        }

        restore_linear_weights("output", &mut self.output, &map, &device);
    }

    /// Public entry-point for restoring weights from a [`WeightMap`].
    ///
    /// This is used by the [`super::convert`] module to load weights
    /// extracted from a `TrainableLstm` into an inference `LstmPredictor`.
    pub fn apply_weight_map(&mut self, weight_map: &WeightMap) {
        self.restore_weights(weight_map);
    }

    /// Public read-only access to the model's current weights.
    ///
    /// This is used by [`super::convert::soft_update_lstm_target`] to read
    /// the target network's weights before blending with the online network.
    pub fn extract_weights_pub(&self) -> WeightMap {
        self.extract_weights()
    }
}

// ---------------------------------------------------------------------------
// Model trait implementation (CPU)
// ---------------------------------------------------------------------------

impl Model for LstmPredictor<CpuBackend> {
    type Config = LstmConfig;

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
            "Saved LSTM model to {:?} ({} weight tensors)",
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
        let config: LstmConfig = serde_json::from_str(&record.config)?;

        // Create fresh model from config
        let mut model = Self::new(config, device);

        // Restore weights if the blob is non-empty
        if !record.weights.is_empty() {
            let weight_map: WeightMap = serde_json::from_slice(&record.weights).map_err(|e| {
                MLError::model_load(format!("Failed to deserialise weights: {}", e))
            })?;
            model.restore_weights(&weight_map);
            tracing::info!(
                "Loaded LSTM model from {:?} ({} weight tensors restored)",
                path,
                weight_map.len()
            );
        } else {
            tracing::info!(
                "Loaded LSTM model config from {:?} (no weights in file, using fresh init)",
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
    fn test_lstm_config() {
        let config = LstmConfig::new(50, 64, 1);
        assert_eq!(config.input_size, 50);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.output_size, 1);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.dropout, 0.2);
    }

    #[test]
    fn test_lstm_config_builder() {
        let config = LstmConfig::new(30, 128, 1)
            .with_num_layers(3)
            .with_dropout(0.3)
            .with_bidirectional(true);

        assert_eq!(config.num_layers, 3);
        assert_eq!(config.dropout, 0.3);
        assert!(config.bidirectional);
    }

    #[test]
    fn test_lstm_output_size() {
        let uni = LstmConfig::new(10, 16, 1);
        assert_eq!(uni.lstm_output_size(), 16);

        let bi = LstmConfig::new(10, 16, 1).with_bidirectional(true);
        assert_eq!(bi.lstm_output_size(), 32);
    }

    #[test]
    fn test_lstm_creation() {
        let config = LstmConfig::new(50, 64, 1);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config, device);

        assert_eq!(model.config.input_size, 50);
        assert_eq!(model.config.hidden_size, 64);
        assert_eq!(model.config.output_size, 1);
        assert_eq!(model.metadata.model_type, "lstm");
    }

    #[test]
    fn test_lstm_forward() {
        let config = LstmConfig::new(10, 16, 1).with_num_layers(1);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config, device);

        // [batch=2, seq_len=5, features=10]
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );

        let output = model.forward(input).unwrap();
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_lstm_forward_bidirectional() {
        // Bidirectional LSTM doubles the effective hidden size so the output
        // layer receives hidden_size * 2 features. The model constructor
        // accounts for this automatically via `lstm_output_size()` and BiLstm.
        let config = LstmConfig::new(10, 8, 1)
            .with_num_layers(1)
            .with_bidirectional(true);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config, device);

        // [batch=2, seq_len=5, features=10]
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );

        let output = model.forward(input).unwrap();
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_lstm_forward_multi_layer_bidirectional() {
        let config = LstmConfig::new(10, 8, 1)
            .with_num_layers(2)
            .with_bidirectional(true);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config, device);

        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );

        let output = model.forward(input).unwrap();
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_lstm_forward_multi_layer_unidirectional() {
        let config = LstmConfig::new(10, 16, 1).with_num_layers(3);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config, device);

        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 5, 10]),
            &Default::default(),
        );

        let output = model.forward(input).unwrap();
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_lstm_metadata() {
        let config = LstmConfig::new(50, 64, 1);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config, device);

        let metadata = model.metadata();
        assert_eq!(metadata.name, "lstm_predictor");
        assert_eq!(metadata.model_type, "lstm");
        assert_eq!(metadata.input_size, 50);
        assert_eq!(metadata.output_size, 1);
        assert_eq!(metadata.backend, "cpu");
    }

    #[test]
    fn test_lstm_save_load_config_only() {
        // Verify round-trip of config when weights blob is empty (legacy compat)
        let config = LstmConfig::new(10, 16, 1).with_num_layers(1);
        let device = BackendDevice::cpu();

        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_lstm_config_only.bin");

        // Build a record with empty weights
        let config_json = serde_json::to_string(&config).unwrap();
        let metadata = ModelMetadata::new(
            "lstm_predictor".into(),
            env!("CARGO_PKG_VERSION").into(),
            "lstm".into(),
            config.input_size,
            config.output_size,
            "cpu".into(),
        );
        let record = ModelRecord::new(metadata, config_json, Vec::<u8>::new());
        let bytes = postcard::to_allocvec(&record).unwrap();
        std::fs::write(&model_path, bytes).unwrap();

        let loaded = LstmPredictor::<CpuBackend>::load(&model_path, device).unwrap();
        assert_eq!(loaded.config.input_size, 10);
        assert_eq!(loaded.config.hidden_size, 16);
        assert_eq!(loaded.config.output_size, 1);

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_lstm_save_load_with_weights() {
        let config = LstmConfig::new(10, 16, 1).with_num_layers(1);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config.clone(), device.clone());

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
        let model_path = temp_dir.join("test_lstm_weights.bin");
        <LstmPredictor<CpuBackend> as Model>::save(&model, &model_path).unwrap();

        // Load
        let loaded = LstmPredictor::<CpuBackend>::load(&model_path, device).unwrap();

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
        assert_eq!(loaded.config.hidden_size, config.hidden_size);
        assert_eq!(loaded.config.output_size, config.output_size);

        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_lstm_save_load_bidirectional_weights() {
        let config = LstmConfig::new(10, 8, 1)
            .with_num_layers(1)
            .with_bidirectional(true);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config.clone(), device.clone());

        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [2, 5, 10]),
            &Default::default(),
        );
        let original_output = model.forward_with_backend(input);
        let original_vals: Vec<f32> = original_output.to_data().to_vec().unwrap();

        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_lstm_bi_weights.bin");
        <LstmPredictor<CpuBackend> as Model>::save(&model, &model_path).unwrap();

        let loaded = LstmPredictor::<CpuBackend>::load(&model_path, device).unwrap();

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
    fn test_lstm_model_trait() {
        let config = LstmConfig::new(20, 32, 1);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config, device);

        assert_eq!(model.input_size(), 20);
        assert_eq!(model.output_size(), 1);
        assert_eq!(model.name(), "lstm_predictor");
        assert!(!model.is_gpu());
    }

    /// Helper to check that the `Model` trait default helpers delegate
    /// correctly through `ModelMetadata`.
    #[test]
    fn test_model_trait_defaults() {
        use super::super::Model as ModelTrait;

        let config = LstmConfig::new(10, 16, 3);
        let device = BackendDevice::cpu();
        let model = LstmPredictor::<CpuBackend>::new(config, device);

        assert_eq!(ModelTrait::input_size(&model), 10);
        assert_eq!(ModelTrait::output_size(&model), 3);
        assert_eq!(ModelTrait::name(&model), "lstm_predictor");
        assert!(!ModelTrait::is_gpu(&model));
    }
}
