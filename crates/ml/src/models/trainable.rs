//! Trainable model wrappers with autodiff support
//!
//! This module provides wrappers around model architectures that enable
//! gradient-based training using Burn's autodiff backend.
//!
//! # Architecture
//!
//! The key insight is that Burn models need different backend types for:
//! - **Training**: `Autodiff<Backend>` - tracks gradients for backpropagation
//! - **Inference**: `Backend` - runs forward pass only, no gradient tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::models::trainable::{TrainableLstm, TrainableLstmConfig};
//! use janus_ml::backend::AutodiffCpuBackend;
//!
//! // Create trainable model
//! let config = TrainableLstmConfig::new(50, 64, 1);
//! let model: TrainableLstm<AutodiffCpuBackend> = config.init(&device);
//!
//! // Forward pass (with gradient tracking)
//! let predictions = model.forward(input);
//!
//! // Compute loss and gradients
//! let loss = loss_fn(predictions, targets);
//! let grads = loss.backward();
//!
//! // Update weights with optimizer
//! model = optimizer.step(lr, model, grads);
//! ```

use burn::module::Module;
use burn_core::tensor::{Tensor, backend::Backend};
use burn_nn::{
    Dropout, DropoutConfig, Linear, LinearConfig,
    lstm::{Lstm, LstmConfig as BurnLstmConfig},
};
use serde::{Deserialize, Serialize};

/// Configuration for trainable LSTM model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableLstmConfig {
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
}

impl TrainableLstmConfig {
    /// Create a new LSTM configuration
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            num_layers: 2,
            dropout: 0.2,
            bidirectional: false,
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

    /// Initialize the model with the given configuration
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrainableLstm<B> {
        let mut lstm_layers = Vec::new();

        // Create LSTM layers
        for i in 0..self.num_layers {
            let input_size = if i == 0 {
                self.input_size
            } else {
                self.hidden_size * if self.bidirectional { 2 } else { 1 }
            };

            let lstm_config = BurnLstmConfig::new(input_size, self.hidden_size, true);
            let lstm = lstm_config.init(device);
            lstm_layers.push(lstm);
        }

        // Create dropout
        let dropout_config = DropoutConfig::new(self.dropout);
        let dropout = dropout_config.init();

        // Create output projection
        let lstm_output_size = self.hidden_size * if self.bidirectional { 2 } else { 1 };
        let output_config = LinearConfig::new(lstm_output_size, self.output_size);
        let output = output_config.init(device);

        TrainableLstm {
            lstm_layers,
            dropout,
            output,
        }
    }
}

impl Default for TrainableLstmConfig {
    fn default() -> Self {
        Self::new(50, 64, 1)
    }
}

/// Trainable LSTM model with autodiff support
///
/// This model wraps Burn's LSTM layers and provides gradient tracking
/// for training via backpropagation.
#[derive(Module)]
pub struct TrainableLstm<B: Backend> {
    /// LSTM layers
    pub(crate) lstm_layers: Vec<Lstm<B>>,

    /// Dropout layer
    pub(crate) dropout: Dropout,

    /// Output projection layer
    pub(crate) output: Linear<B>,
}

impl<B: Backend> std::fmt::Debug for TrainableLstm<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainableLstm")
            .field("num_lstm_layers", &self.lstm_layers.len())
            .field("has_dropout", &true)
            .field("has_output_layer", &true)
            .finish()
    }
}

impl<B: Backend> TrainableLstm<B> {
    /// Forward pass through the LSTM network
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, seq_len, input_size]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, output_size]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let mut x = input;

        // Pass through LSTM layers
        for (i, lstm) in self.lstm_layers.iter().enumerate() {
            // LSTM forward
            let lstm_out = lstm.forward(x, None);
            x = lstm_out.0;

            // Apply dropout between layers (except last)
            if i < self.lstm_layers.len() - 1 {
                x = self.dropout.forward(x);
            }
        }

        // Take the last timestep output
        // x is [batch_size, seq_len, hidden_size * directions]
        let [batch_size, seq_len, feature_dim] = x.dims();
        let last_output = x.slice([0..batch_size, (seq_len - 1)..seq_len, 0..feature_dim]);
        let last_output = last_output.reshape([batch_size, feature_dim]);

        // Project to output

        self.output.forward(last_output)
    }
}

/// Configuration for trainable MLP model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableMlpConfig {
    /// Input feature dimension
    pub input_size: usize,

    /// Hidden layer dimensions
    pub hidden_sizes: Vec<usize>,

    /// Output dimension (number of classes or 1 for regression)
    pub output_size: usize,

    /// Dropout probability
    pub dropout: f64,

    /// Use batch normalization
    pub use_batch_norm: bool,
}

impl TrainableMlpConfig {
    /// Create a new MLP configuration
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_sizes,
            output_size,
            dropout: 0.2,
            use_batch_norm: false,
        }
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable batch normalization
    pub fn with_batch_norm(mut self) -> Self {
        self.use_batch_norm = true;
        self
    }

    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrainableMlp<B> {
        let mut layers = Vec::new();
        let mut prev_size = self.input_size;

        // Create hidden layers
        for &hidden_size in &self.hidden_sizes {
            let linear_config = LinearConfig::new(prev_size, hidden_size);
            let linear = linear_config.init(device);
            layers.push(linear);
            prev_size = hidden_size;
        }

        // Create output layer
        let output_config = LinearConfig::new(prev_size, self.output_size);
        let output = output_config.init(device);

        // Create dropout
        let dropout_config = DropoutConfig::new(self.dropout);
        let dropout = dropout_config.init();

        TrainableMlp {
            layers,
            output,
            dropout,
        }
    }
}

impl Default for TrainableMlpConfig {
    fn default() -> Self {
        Self::new(50, vec![128, 64], 1)
    }
}

/// Trainable MLP model with autodiff support
#[derive(Module)]
pub struct TrainableMlp<B: Backend> {
    /// Hidden layers
    pub(crate) layers: Vec<Linear<B>>,

    /// Output layer
    pub(crate) output: Linear<B>,

    /// Dropout layer
    pub(crate) dropout: Dropout,
}

impl<B: Backend> std::fmt::Debug for TrainableMlp<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainableMlp")
            .field("num_hidden_layers", &self.layers.len())
            .field("has_output_layer", &true)
            .field("has_dropout", &true)
            .finish()
    }
}

impl<B: Backend> TrainableMlp<B> {
    /// Forward pass through the MLP
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, input_size]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, output_size]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        // Pass through hidden layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            x = burn_core::tensor::activation::relu(x);

            // Apply dropout (except last hidden layer)
            if i < self.layers.len() - 1 {
                x = self.dropout.forward(x);
            }
        }

        // Output layer (no activation for regression, or will add softmax for classification)

        self.output.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_core::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_trainable_lstm_config() {
        let config = TrainableLstmConfig::new(50, 64, 1);
        assert_eq!(config.input_size, 50);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.output_size, 1);
        assert_eq!(config.num_layers, 2);
    }

    #[test]
    fn test_trainable_lstm_creation() {
        let config = TrainableLstmConfig::new(10, 16, 1).with_num_layers(1);
        let device = Default::default();
        let _model: TrainableLstm<TestBackend> = config.init(&device);

        // Model created successfully
    }

    #[test]
    fn test_trainable_lstm_forward() {
        let config = TrainableLstmConfig::new(10, 16, 1).with_num_layers(1);
        let device = Default::default();
        let model: TrainableLstm<TestBackend> = config.init(&device);

        // Create dummy input: [batch=2, seq_len=5, features=10]
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let input =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(input_data, [2, 5, 10]), &device);

        let output = model.forward(input);

        // Output should be [batch=2, output=1]
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_trainable_mlp_config() {
        let config = TrainableMlpConfig::new(50, vec![128, 64], 1);
        assert_eq!(config.input_size, 50);
        assert_eq!(config.hidden_sizes, vec![128, 64]);
        assert_eq!(config.output_size, 1);
    }

    #[test]
    fn test_trainable_mlp_creation() {
        let config = TrainableMlpConfig::new(20, vec![32, 16], 1);
        let device = Default::default();
        let _model: TrainableMlp<TestBackend> = config.init(&device);

        // Model created successfully
    }

    #[test]
    fn test_trainable_mlp_forward() {
        let config = TrainableMlpConfig::new(10, vec![16], 1);
        let device = Default::default();
        let model: TrainableMlp<TestBackend> = config.init(&device);

        // Create dummy input: [batch=4, features=10]
        let input_data: Vec<f32> = (0..40).map(|i| i as f32 * 0.01).collect();
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(input_data, [4, 10]), &device);

        let output = model.forward(input);

        // Output should be [batch=4, output=1]
        assert_eq!(output.dims(), [4, 1]);
    }

    #[test]
    fn test_backward_pass() {
        // Test that gradients can be computed
        let config = TrainableMlpConfig::new(5, vec![8], 1);
        let device = Default::default();
        let model: TrainableMlp<TestBackend> = config.init(&device);

        // Create input and target
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(input_data, [1, 5]), &device);

        let target =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![1.0], [1, 1]), &device);

        // Forward pass
        let output = model.forward(input);

        // Compute loss
        let loss = (output - target).powf_scalar(2.0).mean();

        // Backward pass - this should not panic
        let grads = loss.backward();

        // Verify we got gradients (grads object exists)
        // The actual gradient values don't matter for this test
        drop(grads);
    }
}
