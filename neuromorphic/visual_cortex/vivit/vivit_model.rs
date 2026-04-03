//! ViViT (Video Vision Transformer) Model
//!
//! Complete implementation of the ViViT architecture for spatiotemporal data processing.
//! This model processes video-like data (e.g., time series of market charts) through
//! a transformer-based architecture.
//!
//! # Architecture Overview
//!
//! ```text
//! Input Video: [B, T, H, W, C]
//!       ↓
//! ┌─────────────────────┐
//! │ Tubelet Embedding   │  Convert to patches + positional encoding
//! └─────────────────────┘
//!       ↓
//! [B, N+1, D]  (N patches + CLS token)
//!       ↓
//! ┌─────────────────────┐
//! │ Transformer Encoder │
//! │  ┌───────────────┐  │
//! │  │ Layer Norm    │  │
//! │  ├───────────────┤  │
//! │  │ Multi-Head    │  │  ×L layers
//! │  │ Attention     │  │
//! │  ├───────────────┤  │
//! │  │ Layer Norm    │  │
//! │  ├───────────────┤  │
//! │  │ Feed Forward  │  │
//! │  └───────────────┘  │
//! └─────────────────────┘
//!       ↓
//! ┌─────────────────────┐
//! │ Classification Head │  Extract CLS token
//! └─────────────────────┘
//!       ↓
//! Output: [B, num_classes]
//! ```
//!
//! # Usage for Trading
//!
//! The ViViT model processes sequences of market chart images (e.g., GAF-encoded
//! candlestick patterns) and learns to recognize complex spatiotemporal patterns
//! that predict market movements.

use super::{TemporalAttention, TemporalAttentionConfig, TubeletConfig, TubeletEmbedding};
use common::Result;
use ndarray::{Array1, Array2, Array3, Array5, s};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// ViViT model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VivitConfig {
    /// Number of frames in input video
    pub num_frames: usize,

    /// Height of input frames
    pub frame_height: usize,

    /// Width of input frames
    pub frame_width: usize,

    /// Number of input channels (e.g., 3 for RGB, 5 for OHLCV)
    pub num_channels: usize,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// Number of transformer encoder layers
    pub num_layers: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// MLP (feed-forward) hidden dimension ratio
    pub mlp_ratio: usize,

    /// Dropout rate
    pub dropout: f32,

    /// Attention dropout rate
    pub attention_dropout: f32,

    /// Number of output classes (e.g., up/down/neutral)
    pub num_classes: usize,

    /// Tubelet configuration
    pub tubelet_config: TubeletConfig,

    /// Whether to use layer normalization
    pub use_layer_norm: bool,

    /// Activation function type
    pub activation: ActivationType,
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
}

impl Default for VivitConfig {
    fn default() -> Self {
        Self {
            num_frames: 16,
            frame_height: 224,
            frame_width: 224,
            num_channels: 3,
            embedding_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4,
            dropout: 0.1,
            attention_dropout: 0.1,
            num_classes: 3, // up/down/neutral
            tubelet_config: TubeletConfig::default(),
            use_layer_norm: true,
            activation: ActivationType::GELU,
        }
    }
}

/// Complete ViViT model
pub struct VivitModel {
    /// Configuration
    config: VivitConfig,

    /// Tubelet embedding layer
    tubelet_embedding: TubeletEmbedding,

    /// Transformer encoder layers
    encoder_layers: Vec<TransformerEncoderLayer>,

    /// Final layer normalization
    final_norm: Option<LayerNorm>,

    /// Classification head (linear projection from embedding to classes)
    classifier: Array2<f32>,

    /// Classifier bias
    classifier_bias: Array1<f32>,

    /// Training mode flag
    training: bool,

    /// Cached intermediate activations for backprop
    cached_activations: Option<Vec<Array3<f32>>>,
}

/// Single transformer encoder layer
struct TransformerEncoderLayer {
    /// Multi-head self-attention
    attention: TemporalAttention,

    /// First layer normalization
    norm1: Option<LayerNorm>,

    /// Feed-forward network
    mlp: Mlp,

    /// Second layer normalization
    norm2: Option<LayerNorm>,

    /// Dropout rate (reserved for future dropout implementation)
    #[allow(dead_code)]
    dropout: f32,
}

/// Multi-layer perceptron (feed-forward network)
struct Mlp {
    /// First linear layer weights [hidden_dim, embedding_dim]
    fc1: Array2<f32>,

    /// First linear layer bias
    fc1_bias: Array1<f32>,

    /// Second linear layer weights [embedding_dim, hidden_dim]
    fc2: Array2<f32>,

    /// Second linear layer bias
    fc2_bias: Array1<f32>,

    /// Activation function
    activation: ActivationType,

    /// Dropout rate (reserved for future dropout implementation)
    #[allow(dead_code)]
    dropout: f32,
}

/// Layer normalization
struct LayerNorm {
    /// Learnable scale parameter
    gamma: Array1<f32>,

    /// Learnable shift parameter
    beta: Array1<f32>,

    /// Epsilon for numerical stability
    eps: f32,
}

impl VivitModel {
    /// Create a new ViViT model
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Example
    ///
    /// ```
    /// use janus_neuromorphic::visual_cortex::vivit::{VivitModel, VivitConfig};
    ///
    /// let config = VivitConfig::default();
    /// let model = VivitModel::new(config);
    /// ```
    pub fn new(config: VivitConfig) -> Self {
        info!("Initializing ViViT model with {} layers", config.num_layers);

        // Create tubelet embedding layer
        let tubelet_embedding = TubeletEmbedding::new(config.tubelet_config.clone());

        // Create transformer encoder layers
        let mut encoder_layers = Vec::new();
        for i in 0..config.num_layers {
            debug!("Creating encoder layer {}/{}", i + 1, config.num_layers);
            encoder_layers.push(TransformerEncoderLayer::new(&config));
        }

        // Create final layer normalization
        let final_norm = if config.use_layer_norm {
            Some(LayerNorm::new(config.embedding_dim))
        } else {
            None
        };

        // Initialize classification head
        let classifier = Self::xavier_init(config.num_classes, config.embedding_dim);
        let classifier_bias = Array1::zeros(config.num_classes);

        info!("ViViT model initialized successfully");

        Self {
            config,
            tubelet_embedding,
            encoder_layers,
            final_norm,
            classifier,
            classifier_bias,
            training: false,
            cached_activations: None,
        }
    }

    /// Forward pass through the ViViT model
    ///
    /// # Arguments
    ///
    /// * `input` - Input video tensor [B, T, H, W, C]
    ///
    /// # Returns
    ///
    /// Class logits [B, num_classes]
    pub fn forward(&mut self, input: &Array5<f32>) -> Result<Array2<f32>> {
        let batch_size = input.shape()[0];

        debug!("ViViT forward pass: input shape {:?}", input.shape());

        // 1. Tubelet embedding: [B, T, H, W, C] -> [B, N+1, D]
        let mut x = self.tubelet_embedding.forward(input)?;

        debug!("After tubelet embedding: {:?}", x.shape());

        // Cache activations if in training mode
        let mut activations = if self.training {
            Some(Vec::new())
        } else {
            None
        };

        // 2. Pass through transformer encoder layers
        let num_layers = self.encoder_layers.len();
        for (i, layer) in self.encoder_layers.iter_mut().enumerate() {
            debug!("Encoder layer {}/{}", i + 1, num_layers);
            x = layer.forward(&x)?;

            if let Some(ref mut cache) = activations {
                cache.push(x.clone());
            }
        }

        // 3. Final layer normalization
        if let Some(ref norm) = self.final_norm {
            x = norm.forward(&x)?;
        }

        // 4. Extract CLS token (first token)
        let mut cls_tokens = Array2::zeros((batch_size, self.config.embedding_dim));
        for b in 0..batch_size {
            cls_tokens
                .slice_mut(s![b, ..])
                .assign(&x.slice(s![b, 0, ..]));
        }

        debug!("CLS tokens extracted: {:?}", cls_tokens.shape());

        // 5. Classification head
        let logits = self.classify(&cls_tokens)?;

        debug!("Output logits: {:?}", logits.shape());

        // Cache activations for backprop
        self.cached_activations = activations;

        Ok(logits)
    }

    /// Extract CLS token embeddings without the classification head.
    ///
    /// Returns the raw embedding vector of shape `[B, embedding_dim]`,
    /// suitable for storage in a vector database.
    pub fn embed(&mut self, input: &Array5<f32>) -> Result<Array2<f32>> {
        let batch_size = input.shape()[0];

        debug!("ViViT embed pass: input shape {:?}", input.shape());

        // 1. Tubelet embedding: [B, T, H, W, C] -> [B, N+1, D]
        let mut x = self.tubelet_embedding.forward(input)?;

        // 2. Pass through transformer encoder layers
        for layer in self.encoder_layers.iter_mut() {
            x = layer.forward(&x)?;
        }

        // 3. Final layer normalization
        if let Some(ref norm) = self.final_norm {
            x = norm.forward(&x)?;
        }

        // 4. Extract CLS token (first token) -> [B, embedding_dim]
        let mut cls_tokens = Array2::zeros((batch_size, self.config.embedding_dim));
        for b in 0..batch_size {
            cls_tokens
                .slice_mut(s![b, ..])
                .assign(&x.slice(s![b, 0, ..]));
        }

        debug!("Embeddings extracted: {:?}", cls_tokens.shape());

        Ok(cls_tokens)
    }

    /// Classification head: project CLS token to class logits
    fn classify(&self, cls_tokens: &Array2<f32>) -> Result<Array2<f32>> {
        let batch_size = cls_tokens.shape()[0];
        let mut logits = Array2::zeros((batch_size, self.config.num_classes));

        for b in 0..batch_size {
            let x = cls_tokens.slice(s![b, ..]);
            let y = self.classifier.dot(&x) + &self.classifier_bias;
            logits.slice_mut(s![b, ..]).assign(&y);
        }

        Ok(logits)
    }

    /// Compute softmax probabilities from logits
    pub fn predict(&mut self, input: &Array5<f32>) -> Result<Array2<f32>> {
        let logits = self.forward(input)?;
        self.softmax(&logits)
    }

    /// Softmax activation
    fn softmax(&self, logits: &Array2<f32>) -> Result<Array2<f32>> {
        let (batch_size, num_classes) = logits.dim();
        let mut probs = Array2::zeros((batch_size, num_classes));

        for b in 0..batch_size {
            let logits_row = logits.slice(s![b, ..]);

            // Numerical stability: subtract max
            let max_logit = logits_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_values: Vec<f32> = logits_row.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            for c in 0..num_classes {
                probs[[b, c]] = exp_values[c] / sum_exp;
            }
        }

        Ok(probs)
    }

    /// Set model to training mode
    pub fn train(&mut self) {
        self.training = true;
        info!("ViViT model set to training mode");
    }

    /// Set model to evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
        self.cached_activations = None;
        info!("ViViT model set to evaluation mode");
    }

    /// Get model configuration
    pub fn config(&self) -> &VivitConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Tubelet embedding (estimated)
        total += self.config.embedding_dim
            * self.config.tubelet_config.temporal_patch_size
            * self.config.tubelet_config.spatial_patch_height
            * self.config.tubelet_config.spatial_patch_width
            * self.config.num_channels;

        // Encoder layers
        for _ in 0..self.config.num_layers {
            // Attention: 4 * (D * D) for Q, K, V, O projections
            total += 4 * self.config.embedding_dim * self.config.embedding_dim;

            // MLP: 2 * (D * mlp_hidden)
            let mlp_hidden = self.config.embedding_dim * self.config.mlp_ratio;
            total += self.config.embedding_dim * mlp_hidden;
            total += mlp_hidden * self.config.embedding_dim;

            // Layer norms: 2 * 2 * D (gamma + beta)
            if self.config.use_layer_norm {
                total += 4 * self.config.embedding_dim;
            }
        }

        // Classification head
        total += self.config.num_classes * self.config.embedding_dim;
        total += self.config.num_classes;

        total
    }

    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        use rand::RngExt;
        let mut rng = rand::rng();
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| rng.random_range(-limit..limit))
    }

    /// Get cached activations (for visualization/debugging)
    pub fn get_cached_activations(&self) -> Option<&Vec<Array3<f32>>> {
        self.cached_activations.as_ref()
    }
}

impl TransformerEncoderLayer {
    fn new(config: &VivitConfig) -> Self {
        let attention_config = TemporalAttentionConfig::new(config.embedding_dim, config.num_heads)
            .expect("Invalid attention config");

        let attention = TemporalAttention::new(attention_config);

        let norm1 = if config.use_layer_norm {
            Some(LayerNorm::new(config.embedding_dim))
        } else {
            None
        };

        let norm2 = if config.use_layer_norm {
            Some(LayerNorm::new(config.embedding_dim))
        } else {
            None
        };

        let mlp = Mlp::new(
            config.embedding_dim,
            config.embedding_dim * config.mlp_ratio,
            config.activation,
            config.dropout,
        );

        Self {
            attention,
            norm1,
            mlp,
            norm2,
            dropout: config.dropout,
        }
    }

    fn forward(&mut self, input: &Array3<f32>) -> Result<Array3<f32>> {
        // Pre-norm architecture:
        // x = x + Attention(LayerNorm(x))
        // x = x + MLP(LayerNorm(x))

        let mut x = input.clone();

        // 1. Self-attention with residual
        let normed = if let Some(ref norm) = self.norm1 {
            norm.forward(&x)?
        } else {
            x.clone()
        };

        let attn_output = self.attention.forward(&normed)?;
        x = x + attn_output;

        // 2. MLP with residual
        let normed = if let Some(ref norm) = self.norm2 {
            norm.forward(&x)?
        } else {
            x.clone()
        };

        let mlp_output = self.mlp.forward(&normed)?;
        x = x + mlp_output;

        Ok(x)
    }
}

impl Mlp {
    fn new(
        in_features: usize,
        hidden_features: usize,
        activation: ActivationType,
        dropout: f32,
    ) -> Self {
        let fc1 = Self::xavier_init(hidden_features, in_features);
        let fc1_bias = Array1::zeros(hidden_features);
        let fc2 = Self::xavier_init(in_features, hidden_features);
        let fc2_bias = Array1::zeros(in_features);

        Self {
            fc1,
            fc1_bias,
            fc2,
            fc2_bias,
            activation,
            dropout,
        }
    }

    fn forward(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, in_features) = input.dim();
        let hidden_features = self.fc1.shape()[0];

        // First linear layer
        let mut hidden = Array3::zeros((batch_size, seq_len, hidden_features));
        for b in 0..batch_size {
            for s in 0..seq_len {
                let x = input.slice(s![b, s, ..]);
                let h = self.fc1.dot(&x) + &self.fc1_bias;
                hidden.slice_mut(s![b, s, ..]).assign(&h);
            }
        }

        // Activation
        hidden = self.apply_activation(&hidden);

        // Second linear layer
        let mut output = Array3::zeros((batch_size, seq_len, in_features));
        for b in 0..batch_size {
            for s in 0..seq_len {
                let h = hidden.slice(s![b, s, ..]);
                let o = self.fc2.dot(&h) + &self.fc2_bias;
                output.slice_mut(s![b, s, ..]).assign(&o);
            }
        }

        Ok(output)
    }

    fn apply_activation(&self, x: &Array3<f32>) -> Array3<f32> {
        match self.activation {
            ActivationType::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationType::GELU => x.mapv(Self::gelu),
            ActivationType::Tanh => x.mapv(|v| v.tanh()),
            ActivationType::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
        }
    }

    fn gelu(x: f32) -> f32 {
        // GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of the standard normal
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        use rand::RngExt;
        let mut rng = rand::rng();
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| rng.random_range(-limit..limit))
    }
}

impl LayerNorm {
    fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: Array1::ones(normalized_shape),
            beta: Array1::zeros(normalized_shape),
            eps: 1e-5,
        }
    }

    fn forward(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, features) = input.dim();
        let mut output = Array3::zeros((batch_size, seq_len, features));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let x = input.slice(s![b, s, ..]);

                // Compute mean and variance
                let mean = x.mean().unwrap();
                let variance = x.mapv(|v| (v - mean).powi(2)).mean().unwrap();
                let std = (variance + self.eps).sqrt();

                // Normalize
                let normalized = x.mapv(|v| (v - mean) / std);

                // Scale and shift
                let scaled = &normalized * &self.gamma + &self.beta;

                output.slice_mut(s![b, s, ..]).assign(&scaled);
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vivit_creation() {
        let config = VivitConfig::default();
        let model = VivitModel::new(config);
        assert_eq!(model.config.num_layers, 12);
    }

    #[test]
    fn test_vivit_forward() {
        let mut config = VivitConfig::default();
        config.num_frames = 8;
        config.frame_height = 32;
        config.frame_width = 32;
        config.num_channels = 3;
        config.num_layers = 2;
        config.num_classes = 3;

        config.tubelet_config.temporal_patch_size = 2;
        config.tubelet_config.spatial_patch_height = 8;
        config.tubelet_config.spatial_patch_width = 8;

        let mut model = VivitModel::new(config);

        // Input: [B=1, T=8, H=32, W=32, C=3]
        let input = Array5::from_elem((1, 8, 32, 32, 3), 0.5);

        let result = model.forward(&input);
        assert!(result.is_ok());

        let logits = result.unwrap();
        assert_eq!(logits.shape(), &[1, 3]); // [batch_size, num_classes]
    }

    #[test]
    fn test_vivit_predict() {
        let mut config = VivitConfig::default();
        config.num_frames = 8;
        config.frame_height = 32;
        config.frame_width = 32;
        config.num_layers = 1;

        config.tubelet_config.temporal_patch_size = 2;
        config.tubelet_config.spatial_patch_height = 8;
        config.tubelet_config.spatial_patch_width = 8;

        let mut model = VivitModel::new(config);

        let input = Array5::from_elem((2, 8, 32, 32, 3), 0.5);

        let result = model.predict(&input);
        assert!(result.is_ok());

        let probs = result.unwrap();
        assert_eq!(probs.shape(), &[2, 3]);

        // Check probabilities sum to 1
        for b in 0..2 {
            let sum: f32 = probs.slice(s![b, ..]).iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "Probabilities sum: {}", sum);
        }
    }

    #[test]
    fn test_training_mode() {
        let config = VivitConfig::default();
        let mut model = VivitModel::new(config);

        assert!(!model.training);

        model.train();
        assert!(model.training);

        model.eval();
        assert!(!model.training);
    }

    #[test]
    fn test_num_parameters() {
        let config = VivitConfig::default();
        let model = VivitModel::new(config);

        let num_params = model.num_parameters();
        assert!(num_params > 0);

        // ViViT-Base typically has ~86M parameters
        // Our simplified version should have fewer
        println!("Model has {} parameters", num_params);
    }

    #[test]
    fn test_layer_norm() {
        let layer_norm = LayerNorm::new(128);

        // Use varied input values instead of constant
        let mut input = Array3::zeros((2, 10, 128));
        for i in 0..128 {
            input.slice_mut(s![.., .., i]).fill(i as f32 / 10.0);
        }

        let output = layer_norm.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 10, 128]);

        // After layer norm, mean should be ~0 and variance ~1
        for b in 0..2 {
            for s in 0..10 {
                let normalized = output.slice(s![b, s, ..]);
                let mean = normalized.mean().unwrap();
                let variance = normalized.mapv(|v| (v - mean).powi(2)).mean().unwrap();

                assert!(mean.abs() < 1e-5, "Mean: {}", mean);
                assert!((variance - 1.0).abs() < 1e-4, "Variance: {}", variance);
            }
        }
    }

    #[test]
    fn test_mlp() {
        let mlp = Mlp::new(128, 512, ActivationType::GELU, 0.1);

        let input = Array3::from_elem((2, 10, 128), 0.5);
        let output = mlp.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 10, 128]);
    }

    #[test]
    fn test_activation_functions() {
        let mlp_relu = Mlp::new(64, 256, ActivationType::ReLU, 0.0);
        let mlp_gelu = Mlp::new(64, 256, ActivationType::GELU, 0.0);
        let mlp_tanh = Mlp::new(64, 256, ActivationType::Tanh, 0.0);

        let input = Array3::from_elem((1, 5, 64), 1.0);

        assert!(mlp_relu.forward(&input).is_ok());
        assert!(mlp_gelu.forward(&input).is_ok());
        assert!(mlp_tanh.forward(&input).is_ok());
    }

    #[test]
    fn test_transformer_encoder_layer() {
        let mut config = VivitConfig::default();
        config.embedding_dim = 128;
        config.num_heads = 4;
        config.mlp_ratio = 4;

        let mut layer = TransformerEncoderLayer::new(&config);

        let input = Array3::from_elem((2, 10, 128), 0.5);
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 10, 128]);
    }
}
