//! Machine learning model implementations
//!
//! This module contains model architectures for training and inference.
//!
//! # Model Types
//!
//! ## Trainable Models (with autodiff)
//! - `TrainableLstm` - LSTM with gradient tracking for training
//! - `TrainableMlp` - MLP with gradient tracking for training
//!
//! ## Inference Models
//! - `LstmPredictor` - Optimized LSTM for inference (no gradients)
//! - `MlpClassifier` - Optimized MLP for inference (no gradients)
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::models::trainable::TrainableLstmConfig;
//! use janus_ml::backend::AutodiffCpuBackend;
//!
//! // Create trainable model for training
//! let config = TrainableLstmConfig::new(50, 64, 1);
//! let model = config.init::<AutodiffCpuBackend>(&device);
//!
//! // Train model...
//!
//! // Convert to inference model (future work)
//! let predictor = model.to_predictor();
//! ```

pub mod convert;
pub mod lstm;
pub mod mlp;
pub mod trainable;

// Re-export trainable models
pub use trainable::{TrainableLstm, TrainableLstmConfig, TrainableMlp, TrainableMlpConfig};

// Re-export inference models
pub use lstm::{LstmConfig, LstmPredictor};
pub use mlp::{ActivationType, MlpClassifier, MlpConfig};

// Re-export conversion utilities
pub use convert::{
    WeightMapValidation, extract_trainable_lstm_weights, extract_trainable_mlp_weights,
    soft_update_lstm_target, soft_update_mlp_target, soft_update_weight_maps,
    trainable_lstm_config_to_inference, trainable_lstm_to_predictor,
    trainable_lstm_to_predictor_auto, trainable_mlp_config_to_inference,
    trainable_mlp_to_classifier, trainable_mlp_to_classifier_auto, validate_lstm_weight_map,
    validate_mlp_weight_map,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

use burn_core::tensor::Tensor;

use crate::backend::BackendDevice;
use crate::error::Result;

/// Format version for model serialization.
/// Bump this when the serialization format changes incompatibly.
const MODEL_FORMAT_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

/// Trait for inference models that support save/load and forward pass.
///
/// This is the richer model trait used by the inference model implementations
/// (`LstmPredictor`, `MlpClassifier`). It provides configuration management,
/// metadata, serialization, and forward pass.
pub trait Model {
    /// The configuration type for this model.
    type Config: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Create a new model from a configuration and device.
    fn new(config: Self::Config, device: BackendDevice) -> Self;

    /// Run the forward pass.
    ///
    /// # Arguments
    /// * `input` - Tensor of shape `[batch_size, seq_len, input_size]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, output_size]`
    fn forward(
        &self,
        input: Tensor<crate::backend::CpuBackend, 3>,
    ) -> Result<Tensor<crate::backend::CpuBackend, 2>>;

    /// Save the model (config + weights) to a file.
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// Load a model (config + weights) from a file.
    fn load<P: AsRef<Path>>(path: P, device: BackendDevice) -> Result<Self>
    where
        Self: Sized;

    /// Get model metadata.
    fn metadata(&self) -> &ModelMetadata;

    /// Get model configuration.
    fn config(&self) -> &Self::Config;

    /// Get the device this model runs on.
    fn device(&self) -> &BackendDevice;

    // -- default helpers --

    /// Input feature dimension.
    fn input_size(&self) -> usize {
        self.metadata().input_size
    }

    /// Output dimension.
    fn output_size(&self) -> usize {
        self.metadata().output_size
    }

    /// Human-readable name.
    fn name(&self) -> &str {
        &self.metadata().name
    }

    /// Whether the model is running on a GPU backend.
    fn is_gpu(&self) -> bool {
        self.device().is_gpu()
    }
}

// ---------------------------------------------------------------------------
// ModelMetadata
// ---------------------------------------------------------------------------

/// Metadata associated with a persisted model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Human-readable model name (e.g. `"lstm_predictor"`).
    pub name: String,

    /// Crate / schema version that produced this model.
    pub version: String,

    /// Model architecture type (e.g. `"lstm"`, `"mlp"`).
    pub model_type: String,

    /// Input feature dimension the model was built with.
    pub input_size: usize,

    /// Output dimension the model was built with.
    pub output_size: usize,

    /// Backend string (e.g. `"cpu"`, `"gpu"`).
    pub backend: String,

    /// Number of training epochs completed (0 for freshly initialised).
    pub trained_epochs: usize,

    /// Best validation loss seen during training (`f64::INFINITY` if untrained).
    pub best_loss: f64,

    /// Unix timestamp (seconds) when this metadata was created.
    pub created_at: i64,
}

impl ModelMetadata {
    /// Create metadata for a freshly-initialised model.
    pub fn new(
        name: String,
        version: String,
        model_type: String,
        input_size: usize,
        output_size: usize,
        backend: String,
    ) -> Self {
        Self {
            name,
            version,
            model_type,
            input_size,
            output_size,
            backend,
            trained_epochs: 0,
            best_loss: f64::INFINITY,
            created_at: chrono::Utc::now().timestamp(),
        }
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: String::from("untrained"),
            version: String::from("0.1.0"),
            model_type: String::new(),
            input_size: 0,
            output_size: 0,
            backend: String::from("cpu"),
            trained_epochs: 0,
            best_loss: f64::INFINITY,
            created_at: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelRecord  –  on-disk envelope for config + weights
// ---------------------------------------------------------------------------

/// Serialisable envelope that pairs [`ModelMetadata`] with a config JSON string
/// and an opaque weights blob.
///
/// The weights blob (`W`) is typically `Vec<u8>` for binary serialisation or
/// `Vec<SerializedTensor>` when using the structured tensor format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord<W> {
    /// Format version – used for forward-compatibility checks.
    pub format_version: u32,

    /// Model metadata.
    pub metadata: ModelMetadata,

    /// JSON-encoded model configuration.
    pub config: String,

    /// Serialised weights payload.
    pub weights: W,
}

impl<W> ModelRecord<W> {
    /// Create a new model record.
    pub fn new(metadata: ModelMetadata, config: String, weights: W) -> Self {
        Self {
            format_version: MODEL_FORMAT_VERSION,
            metadata,
            config,
            weights,
        }
    }

    /// Returns `true` when the on-disk format version is compatible with the
    /// current build.
    pub fn is_compatible(&self) -> bool {
        self.format_version <= MODEL_FORMAT_VERSION
    }
}

// ---------------------------------------------------------------------------
// SerializedTensor – structured weight representation
// ---------------------------------------------------------------------------

/// A single named tensor serialised as flat `f32` data + shape metadata.
///
/// This is the unit of weight serialisation used by [`LstmPredictor`] and
/// [`MlpClassifier`] when persisting weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTensor {
    /// Dot-separated path that identifies the parameter
    /// (e.g. `"lstm_layers.0.input_transform.weight"`).
    pub name: String,

    /// Shape of the original tensor (e.g. `[64, 10]`).
    pub shape: Vec<usize>,

    /// Flattened `f32` data in row-major order.
    pub data: Vec<f32>,
}

/// A collection of named tensors that together represent a model's weights.
pub type WeightMap = Vec<SerializedTensor>;
