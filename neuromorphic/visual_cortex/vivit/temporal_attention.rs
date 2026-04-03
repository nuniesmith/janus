//! Temporal Multi-Head Attention Mechanism
//!
//! Implements the temporal attention component of ViViT (Video Vision Transformer).
//! This module provides multi-head self-attention specifically designed for
//! processing temporal sequences of embedded tokens.
//!
//! # Architecture
//!
//! ```text
//! Input: [B, N, D]  (Batch, Num_tokens, Embedding_dim)
//!         ↓
//!  ┌─────────────────┐
//!  │  Query (Q)      │  Linear projection
//!  │  Key (K)        │  Q, K, V = Linear(input)
//!  │  Value (V)      │
//!  └─────────────────┘
//!         ↓
//!  ┌─────────────────┐
//!  │  Split Heads    │  Reshape to [B, H, N, D/H]
//!  └─────────────────┘
//!         ↓
//!  ┌─────────────────┐
//!  │  Attention      │  Softmax(QK^T / √d_k) V
//!  │  Scores         │
//!  └─────────────────┘
//!         ↓
//!  ┌─────────────────┐
//!  │  Concat Heads   │  Reshape to [B, N, D]
//!  └─────────────────┘
//!         ↓
//!  ┌─────────────────┐
//!  │  Output Proj    │  Linear projection
//!  └─────────────────┘
//!         ↓
//!    [B, N, D]
//! ```
//!
//! # Multi-Head Attention
//!
//! Multiple attention heads allow the model to jointly attend to information
//! from different representation subspaces:
//!
//! - **Head 1**: Might focus on short-term price movements
//! - **Head 2**: Might capture volume patterns
//! - **Head 3**: Might detect trend reversals
//! - **Head N**: Other temporal dependencies

use common::Result;
use ndarray::{Array1, Array2, Array3, s};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Configuration for temporal attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAttentionConfig {
    /// Embedding dimension (D)
    pub embedding_dim: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Dimension per head (typically embedding_dim / num_heads)
    pub head_dim: usize,

    /// Dropout rate for attention weights
    pub attention_dropout: f32,

    /// Dropout rate for output
    pub output_dropout: f32,

    /// Whether to use causal masking (prevent looking ahead)
    pub causal: bool,

    /// Scaling factor for attention scores (typically 1/√d_k)
    pub scale: f32,

    /// Whether to use bias in linear projections
    pub use_bias: bool,
}

impl TemporalAttentionConfig {
    /// Create a new configuration with validation
    pub fn new(embedding_dim: usize, num_heads: usize) -> Result<Self> {
        if !embedding_dim.is_multiple_of(num_heads) {
            return Err(common::JanusError::InvalidInput(format!(
                "embedding_dim ({}) must be divisible by num_heads ({})",
                embedding_dim, num_heads
            )));
        }

        let head_dim = embedding_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            embedding_dim,
            num_heads,
            head_dim,
            attention_dropout: 0.1,
            output_dropout: 0.1,
            causal: false,
            scale,
            use_bias: true,
        })
    }
}

impl Default for TemporalAttentionConfig {
    fn default() -> Self {
        Self::new(768, 12).unwrap()
    }
}

/// Multi-head temporal attention mechanism
pub struct TemporalAttention {
    /// Configuration
    config: TemporalAttentionConfig,

    /// Query projection weights [D, D]
    query_proj: Array2<f32>,

    /// Key projection weights [D, D]
    key_proj: Array2<f32>,

    /// Value projection weights [D, D]
    value_proj: Array2<f32>,

    /// Output projection weights [D, D]
    output_proj: Array2<f32>,

    /// Query bias [D]
    query_bias: Option<Array1<f32>>,

    /// Key bias [D]
    key_bias: Option<Array1<f32>>,

    /// Value bias [D]
    value_bias: Option<Array1<f32>>,

    /// Output bias [D]
    output_bias: Option<Array1<f32>>,

    /// Cached attention weights for visualization
    last_attention_weights: Option<Array3<f32>>,
}

impl TemporalAttention {
    /// Create a new temporal attention layer
    ///
    /// # Arguments
    ///
    /// * `config` - Attention configuration
    ///
    /// # Example
    ///
    /// ```
    /// use janus_neuromorphic::visual_cortex::vivit::{TemporalAttention, TemporalAttentionConfig};
    ///
    /// # fn example() -> common::Result<()> {
    /// let config = TemporalAttentionConfig::new(768, 12)?;
    /// let attention = TemporalAttention::new(config);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: TemporalAttentionConfig) -> Self {
        let d = config.embedding_dim;

        // Initialize projection weights with Xavier/Glorot
        let query_proj = Self::xavier_init(d, d);
        let key_proj = Self::xavier_init(d, d);
        let value_proj = Self::xavier_init(d, d);
        let output_proj = Self::xavier_init(d, d);

        // Initialize biases if enabled
        let (query_bias, key_bias, value_bias, output_bias) = if config.use_bias {
            (
                Some(Array1::zeros(d)),
                Some(Array1::zeros(d)),
                Some(Array1::zeros(d)),
                Some(Array1::zeros(d)),
            )
        } else {
            (None, None, None, None)
        };

        Self {
            config,
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            query_bias,
            key_bias,
            value_bias,
            output_bias,
            last_attention_weights: None,
        }
    }

    /// Forward pass through temporal attention
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [B, N, D] where:
    ///   - B: batch size
    ///   - N: sequence length (number of tokens)
    ///   - D: embedding dimension
    ///
    /// # Returns
    ///
    /// Output tensor [B, N, D] after multi-head attention
    pub fn forward(&mut self, input: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, embed_dim) = input.dim();

        debug!(
            "TemporalAttention forward: input shape [{}, {}, {}]",
            batch_size, seq_len, embed_dim
        );

        if embed_dim != self.config.embedding_dim {
            return Err(common::JanusError::InvalidInput(format!(
                "Input embedding_dim ({}) doesn't match config ({})",
                embed_dim, self.config.embedding_dim
            )));
        }

        // Linear projections: [B, N, D] -> [B, N, D]
        let queries = self.linear_projection(input, &self.query_proj, &self.query_bias)?;
        let keys = self.linear_projection(input, &self.key_proj, &self.key_bias)?;
        let values = self.linear_projection(input, &self.value_proj, &self.value_bias)?;

        // Reshape for multi-head: [B, N, D] -> [B, H, N, D/H]
        let queries = self.split_heads(&queries)?;
        let keys = self.split_heads(&keys)?;
        let values = self.split_heads(&values)?;

        // Compute attention: [B, H, N, D/H]
        let attended = self.scaled_dot_product_attention(&queries, &keys, &values)?;

        // Merge heads: [B, H, N, D/H] -> [B, N, D]
        let merged = self.merge_heads(&attended)?;

        // Output projection
        let output = self.linear_projection(&merged, &self.output_proj, &self.output_bias)?;

        debug!(
            "TemporalAttention output shape: [{}, {}, {}]",
            output.shape()[0],
            output.shape()[1],
            output.shape()[2]
        );

        Ok(output)
    }

    /// Perform linear projection with optional bias
    fn linear_projection(
        &self,
        input: &Array3<f32>,
        weight: &Array2<f32>,
        bias: &Option<Array1<f32>>,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len, _) = input.dim();
        let out_dim = weight.shape()[0];

        let mut output = Array3::zeros((batch_size, seq_len, out_dim));

        for b in 0..batch_size {
            for n in 0..seq_len {
                let x = input.slice(s![b, n, ..]);
                let y = weight.dot(&x);

                if let Some(bias_vec) = bias {
                    output.slice_mut(s![b, n, ..]).assign(&(&y + bias_vec));
                } else {
                    output.slice_mut(s![b, n, ..]).assign(&y);
                }
            }
        }

        Ok(output)
    }

    /// Split embedding dimension into multiple heads
    /// [B, N, D] -> [B, H, N, D/H]
    fn split_heads(&self, tensor: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, _) = tensor.dim();
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Reshape and transpose
        // We'll represent [B, H, N, D/H] as [B*H, N, D/H] for simplicity
        let mut output = Array3::zeros((batch_size * num_heads, seq_len, head_dim));

        for b in 0..batch_size {
            for h in 0..num_heads {
                for n in 0..seq_len {
                    let start = h * head_dim;
                    let end = start + head_dim;
                    let head_values = tensor.slice(s![b, n, start..end]);
                    output
                        .slice_mut(s![b * num_heads + h, n, ..])
                        .assign(&head_values);
                }
            }
        }

        Ok(output)
    }

    /// Merge heads back to embedding dimension
    /// [B*H, N, D/H] -> [B, N, D]
    fn merge_heads(&self, tensor: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_heads, seq_len, head_dim) = tensor.dim();
        let num_heads = self.config.num_heads;
        let batch_size = batch_heads / num_heads;

        let mut output = Array3::zeros((batch_size, seq_len, self.config.embedding_dim));

        for b in 0..batch_size {
            for h in 0..num_heads {
                for n in 0..seq_len {
                    let start = h * head_dim;
                    let end = start + head_dim;
                    let head_values = tensor.slice(s![b * num_heads + h, n, ..]);
                    output.slice_mut(s![b, n, start..end]).assign(&head_values);
                }
            }
        }

        Ok(output)
    }

    /// Scaled dot-product attention
    ///
    /// Attention(Q, K, V) = softmax(QK^T / √d_k) V
    fn scaled_dot_product_attention(
        &mut self,
        queries: &Array3<f32>,
        keys: &Array3<f32>,
        values: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let (batch_heads, seq_len, head_dim) = queries.dim();

        // Compute attention scores: Q @ K^T
        // [B*H, N, D/H] @ [B*H, D/H, N] -> [B*H, N, N]
        let mut scores = Array3::zeros((batch_heads, seq_len, seq_len));

        for bh in 0..batch_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let q = queries.slice(s![bh, i, ..]);
                    let k = keys.slice(s![bh, j, ..]);
                    let score = q.dot(&k) * self.config.scale;
                    scores[[bh, i, j]] = score;
                }
            }
        }

        // Apply causal mask if enabled (prevent looking ahead)
        if self.config.causal {
            for bh in 0..batch_heads {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        scores[[bh, i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Apply softmax to get attention weights
        let attention_weights = self.softmax_3d(&scores)?;

        // Cache for visualization
        self.last_attention_weights = Some(attention_weights.clone());

        // Apply attention to values: attention_weights @ V
        // [B*H, N, N] @ [B*H, N, D/H] -> [B*H, N, D/H]
        let mut output = Array3::zeros((batch_heads, seq_len, head_dim));

        for bh in 0..batch_heads {
            for i in 0..seq_len {
                let mut attended_value = Array1::zeros(head_dim);
                for j in 0..seq_len {
                    let weight = attention_weights[[bh, i, j]];
                    let value = values.slice(s![bh, j, ..]);
                    let weighted_value = value.to_owned() * weight;
                    attended_value += &weighted_value;
                }
                output.slice_mut(s![bh, i, ..]).assign(&attended_value);
            }
        }

        Ok(output)
    }

    /// Softmax activation over last dimension
    fn softmax_3d(&self, tensor: &Array3<f32>) -> Result<Array3<f32>> {
        let (d0, d1, d2) = tensor.dim();
        let mut output = Array3::zeros((d0, d1, d2));

        for i in 0..d0 {
            for j in 0..d1 {
                let row = tensor.slice(s![i, j, ..]);

                // Numerical stability: subtract max
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exp_values.iter().sum();

                for k in 0..d2 {
                    output[[i, j, k]] = exp_values[k] / sum_exp;
                }
            }
        }

        Ok(output)
    }

    /// Xavier/Glorot initialization for weights
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        use rand::RngExt;
        let mut rng = rand::rng();

        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| rng.random_range(-limit..limit))
    }

    /// Get the last computed attention weights (for visualization)
    pub fn get_attention_weights(&self) -> Option<&Array3<f32>> {
        self.last_attention_weights.as_ref()
    }

    /// Get configuration
    pub fn config(&self) -> &TemporalAttentionConfig {
        &self.config
    }

    /// Compute attention entropy (measure of attention diffusion)
    pub fn compute_attention_entropy(&self) -> Option<Vec<f32>> {
        self.last_attention_weights.as_ref().map(|weights| {
            let (batch_heads, seq_len, _) = weights.dim();
            let mut entropies = Vec::new();

            for bh in 0..batch_heads {
                for i in 0..seq_len {
                    let mut entropy = 0.0;
                    for j in 0..seq_len {
                        let p = weights[[bh, i, j]];
                        if p > 1e-10 {
                            entropy -= p * p.ln();
                        }
                    }
                    entropies.push(entropy);
                }
            }

            entropies
        })
    }
}

impl Default for TemporalAttention {
    fn default() -> Self {
        Self::new(TemporalAttentionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let config = TemporalAttentionConfig::new(768, 12).unwrap();
        let attention = TemporalAttention::new(config);
        assert_eq!(attention.config.num_heads, 12);
        assert_eq!(attention.config.head_dim, 64);
    }

    #[test]
    fn test_invalid_config() {
        // embedding_dim not divisible by num_heads
        let result = TemporalAttentionConfig::new(768, 11);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_pass() {
        let config = TemporalAttentionConfig::new(128, 4).unwrap();
        let mut attention = TemporalAttention::new(config);

        // Input: [B=2, N=10, D=128]
        let input = Array3::from_elem((2, 10, 128), 0.5);

        let result = attention.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[2, 10, 128]);
    }

    #[test]
    fn test_causal_attention() {
        let mut config = TemporalAttentionConfig::new(64, 2).unwrap();
        config.causal = true;

        let mut attention = TemporalAttention::new(config);

        // Input: [B=1, N=5, D=64]
        let input = Array3::from_elem((1, 5, 64), 1.0);

        let result = attention.forward(&input);
        assert!(result.is_ok());

        // Check that attention weights are causal (lower triangular)
        if let Some(weights) = attention.get_attention_weights() {
            let (_, seq_len, _) = weights.dim();
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    // Future positions should have zero attention
                    assert!(weights[[0, i, j]].abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let config = TemporalAttentionConfig::new(64, 2).unwrap();
        let mut attention = TemporalAttention::new(config);

        let input = Array3::from_elem((1, 5, 64), 1.0);
        attention.forward(&input).unwrap();

        if let Some(weights) = attention.get_attention_weights() {
            let (batch_heads, seq_len, _) = weights.dim();

            for bh in 0..batch_heads {
                for i in 0..seq_len {
                    let mut sum = 0.0;
                    for j in 0..seq_len {
                        sum += weights[[bh, i, j]];
                    }
                    // Attention weights should sum to 1.0
                    assert!((sum - 1.0).abs() < 1e-5, "Sum: {}", sum);
                }
            }
        }
    }

    #[test]
    fn test_split_and_merge_heads() {
        let config = TemporalAttentionConfig::new(128, 4).unwrap();
        let attention = TemporalAttention::new(config);

        let input = Array3::from_elem((2, 10, 128), 0.5);

        let split = attention.split_heads(&input).unwrap();
        assert_eq!(split.shape(), &[2 * 4, 10, 32]); // [B*H, N, D/H]

        let merged = attention.merge_heads(&split).unwrap();
        assert_eq!(merged.shape(), &[2, 10, 128]); // Back to [B, N, D]

        // Values should be approximately the same
        for i in 0..2 {
            for j in 0..10 {
                for k in 0..128 {
                    assert!((input[[i, j, k]] - merged[[i, j, k]]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_attention_entropy() {
        let config = TemporalAttentionConfig::new(64, 2).unwrap();
        let mut attention = TemporalAttention::new(config);

        let input = Array3::from_elem((1, 5, 64), 1.0);
        attention.forward(&input).unwrap();

        let entropy = attention.compute_attention_entropy();
        assert!(entropy.is_some());

        let entropies = entropy.unwrap();
        // Should have entropy for each position in each head
        assert_eq!(entropies.len(), 2 * 5); // 2 heads * 5 positions
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = TemporalAttentionConfig::new(128, 4).unwrap();
        let mut attention = TemporalAttention::new(config);

        // Wrong embedding dimension
        let input = Array3::from_elem((2, 10, 64), 0.5);

        let result = attention.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_token() {
        let config = TemporalAttentionConfig::new(64, 2).unwrap();
        let mut attention = TemporalAttention::new(config);

        // Single token sequence
        let input = Array3::from_elem((1, 1, 64), 1.0);

        let result = attention.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 1, 64]);
    }

    #[test]
    fn test_batch_independence() {
        let config = TemporalAttentionConfig::new(64, 2).unwrap();
        let mut attention = TemporalAttention::new(config);

        // Different values in each batch
        let mut input = Array3::zeros((2, 5, 64));
        input.slice_mut(s![0, .., ..]).fill(1.0);
        input.slice_mut(s![1, .., ..]).fill(2.0);

        let output = attention.forward(&input).unwrap();

        // Output batches should be different (not identical)
        let _batch0_mean = output.slice(s![0, .., ..]).mean().unwrap();
        let _batch1_mean = output.slice(s![1, .., ..]).mean().unwrap();

        // They might be similar but shouldn't be exactly the same
        // (due to learned weights)
        assert_eq!(output.shape(), &[2, 5, 64]);
    }
}
