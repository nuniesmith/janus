//! Multi-head cross-attention
//!
//! Part of the Thalamus region
//! Component: attention
//!
//! Implements a multi-head cross-attention mechanism that allows one signal
//! stream (queries) to attend to another stream (keys/values). This is used
//! in the thalamus to enable different market data modalities to selectively
//! focus on relevant features from other modalities.
//!
//! ## Features
//!
//! - **Multi-head attention**: Splits the embedding dimension into multiple
//!   heads so that different subspaces can attend to different aspects of
//!   the input.
//! - **Scaled dot-product**: Attention weights are computed as
//!   `softmax(Q·Kᵀ / √d_k)` following the standard transformer formulation.
//! - **Residual connections**: Optionally adds the query input back to the
//!   attention output for gradient-friendly learning.
//! - **Causal masking**: Optionally prevents positions from attending to
//!   future positions (useful for sequential/time-series data).
//! - **Attention statistics**: Tracks entropy of attention distributions,
//!   sparsity metrics, and per-head utilisation.
//! - **EMA-smoothed output**: The raw attention output can be smoothed with
//!   an exponential moving average to reduce jitter in streaming scenarios.

use std::collections::VecDeque;

use crate::common::{Error, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the cross-attention mechanism.
#[derive(Debug, Clone)]
pub struct CrossAttentionConfig {
    /// Dimension of query, key, and value vectors.
    pub d_model: usize,
    /// Number of attention heads. Must evenly divide `d_model`.
    pub num_heads: usize,
    /// Whether to add a residual connection (query + attention output).
    pub use_residual: bool,
    /// Whether to apply causal masking (prevent attending to future positions).
    pub causal_mask: bool,
    /// EMA decay factor for smoothed output (0, 1). Set to 0 to disable.
    pub ema_decay: f64,
    /// Temperature scaling factor applied to attention logits before softmax.
    /// Values > 1.0 sharpen the distribution; values < 1.0 flatten it.
    /// Default is 1.0 (standard scaling by √d_k only).
    pub temperature: f64,
    /// Maximum number of recent attention entropy values to retain for
    /// windowed statistics.
    pub window_size: usize,
    /// Dropout rate for attention weights (0.0 = no dropout). In inference
    /// mode this is ignored; it is provided for configuration parity with
    /// training pipelines.
    pub dropout_rate: f64,
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            num_heads: 4,
            use_residual: true,
            causal_mask: false,
            ema_decay: 0.0,
            temperature: 1.0,
            window_size: 200,
            dropout_rate: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// Input to the cross-attention mechanism.
///
/// `queries`, `keys`, and `values` are each a sequence of vectors.
/// - `queries`: shape conceptually `[seq_q, d_model]` — flattened row-major.
/// - `keys`:    shape conceptually `[seq_k, d_model]` — flattened row-major.
/// - `values`:  shape conceptually `[seq_k, d_model]` — flattened row-major.
///
/// `seq_k` and `seq_v` must be equal (keys and values come from the same
/// source). `seq_q` may differ (cross-attention).
#[derive(Debug, Clone)]
pub struct AttentionInput {
    /// Query vectors, flattened row-major `[seq_q × d_model]`.
    pub queries: Vec<f64>,
    /// Key vectors, flattened row-major `[seq_k × d_model]`.
    pub keys: Vec<f64>,
    /// Value vectors, flattened row-major `[seq_k × d_model]`.
    pub values: Vec<f64>,
    /// Number of query positions.
    pub seq_q: usize,
    /// Number of key/value positions.
    pub seq_k: usize,
}

/// Output of the cross-attention mechanism.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// The attended output vectors, flattened row-major `[seq_q × d_model]`.
    pub output: Vec<f64>,
    /// Per-head average attention entropy (one value per head).
    /// Higher entropy → more uniform attention; lower → more focused.
    pub head_entropies: Vec<f64>,
    /// Overall mean attention entropy across all heads and query positions.
    pub mean_entropy: f64,
    /// Sparsity metric: fraction of attention weights below 1/(2·seq_k).
    /// Higher → sparser (more focused) attention.
    pub sparsity: f64,
    /// EMA-smoothed output (same shape as `output`). `None` if EMA is
    /// disabled (`ema_decay == 0`).
    pub smoothed_output: Option<Vec<f64>>,
}

/// Aggregate statistics for the attention mechanism.
#[derive(Debug, Clone, Default)]
pub struct CrossAttentionStats {
    /// Total number of forward passes executed.
    pub total_forwards: usize,
    /// Cumulative sum of mean entropy values.
    pub sum_entropy: f64,
    /// Cumulative sum of sparsity values.
    pub sum_sparsity: f64,
    /// Maximum mean entropy observed.
    pub max_entropy: f64,
    /// Minimum mean entropy observed.
    pub min_entropy: f64,
}

impl CrossAttentionStats {
    /// Average mean entropy across all forward passes.
    pub fn avg_entropy(&self) -> f64 {
        if self.total_forwards == 0 {
            0.0
        } else {
            self.sum_entropy / self.total_forwards as f64
        }
    }

    /// Average sparsity across all forward passes.
    pub fn avg_sparsity(&self) -> f64 {
        if self.total_forwards == 0 {
            0.0
        } else {
            self.sum_sparsity / self.total_forwards as f64
        }
    }
}

// ---------------------------------------------------------------------------
// CrossAttention
// ---------------------------------------------------------------------------

/// Multi-head cross-attention mechanism.
///
/// Call [`forward`] with query, key, and value tensors to compute the
/// attended output. The mechanism splits the embedding dimension across
/// `num_heads` independent attention heads, computes scaled dot-product
/// attention in each head, and concatenates the results.
pub struct CrossAttention {
    config: CrossAttentionConfig,
    /// Per-head dimension (`d_model / num_heads`).
    d_k: usize,
    /// EMA state for smoothed output (flattened).
    ema_output: Vec<f64>,
    ema_initialized: bool,
    /// Windowed history of mean entropy values.
    entropy_history: VecDeque<f64>,
    /// Running statistics.
    stats: CrossAttentionStats,
}

impl Default for CrossAttention {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossAttention {
    /// Create a new instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(CrossAttentionConfig::default())
    }

    /// Create a new instance with the given configuration.
    pub fn with_config(config: CrossAttentionConfig) -> Self {
        let d_k = if config.num_heads > 0 {
            config.d_model / config.num_heads
        } else {
            config.d_model
        };
        Self {
            d_k,
            ema_output: Vec::new(),
            ema_initialized: false,
            entropy_history: VecDeque::with_capacity(config.window_size),
            stats: CrossAttentionStats::default(),
            config,
        }
    }

    /// Validate configuration parameters.
    pub fn process(&self) -> Result<()> {
        if self.config.d_model == 0 {
            return Err(Error::InvalidInput("d_model must be > 0".into()));
        }
        if self.config.num_heads == 0 {
            return Err(Error::InvalidInput("num_heads must be > 0".into()));
        }
        if self.config.d_model % self.config.num_heads != 0 {
            return Err(Error::InvalidInput(
                "d_model must be evenly divisible by num_heads".into(),
            ));
        }
        if self.config.temperature <= 0.0 {
            return Err(Error::InvalidInput("temperature must be > 0".into()));
        }
        if self.config.ema_decay < 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in [0, 1)".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.dropout_rate < 0.0 || self.config.dropout_rate > 1.0 {
            return Err(Error::InvalidInput("dropout_rate must be in [0, 1]".into()));
        }
        Ok(())
    }

    // -- Forward pass ------------------------------------------------------

    /// Compute the multi-head cross-attention forward pass.
    pub fn forward(&mut self, input: &AttentionInput) -> Result<AttentionOutput> {
        let d = self.config.d_model;
        let h = self.config.num_heads;
        let d_k = self.d_k;
        let seq_q = input.seq_q;
        let seq_k = input.seq_k;

        // Validate input dimensions.
        if input.queries.len() != seq_q * d {
            return Err(Error::InvalidInput(format!(
                "queries length {} != seq_q({}) * d_model({})",
                input.queries.len(),
                seq_q,
                d
            )));
        }
        if input.keys.len() != seq_k * d {
            return Err(Error::InvalidInput(format!(
                "keys length {} != seq_k({}) * d_model({})",
                input.keys.len(),
                seq_k,
                d
            )));
        }
        if input.values.len() != seq_k * d {
            return Err(Error::InvalidInput(format!(
                "values length {} != seq_k({}) * d_model({})",
                input.values.len(),
                seq_k,
                d
            )));
        }
        if seq_q == 0 || seq_k == 0 {
            return Err(Error::InvalidInput("seq_q and seq_k must be > 0".into()));
        }

        let scale = (d_k as f64).sqrt() / self.config.temperature;
        if scale <= 0.0 {
            return Err(Error::InvalidInput("effective scale must be > 0".into()));
        }
        let inv_scale = 1.0 / scale;

        // Output accumulator: [seq_q, d_model]
        let mut output = vec![0.0_f64; seq_q * d];

        // Per-head entropy accumulators
        let mut head_entropy_sums = vec![0.0_f64; h];
        let mut total_sparse_count = 0usize;
        let total_weight_count = seq_q * seq_k * h;
        let sparsity_threshold = if seq_k > 0 {
            1.0 / (2.0 * seq_k as f64)
        } else {
            0.0
        };

        // Process each head independently.
        for head in 0..h {
            let head_offset = head * d_k;

            for qi in 0..seq_q {
                // Extract query vector for this head: Q[qi, head_offset..head_offset+d_k]
                let q_start = qi * d + head_offset;

                // Compute attention scores: score[ki] = Q[qi] · K[ki] / scale
                let mut scores = vec![0.0_f64; seq_k];
                for ki in 0..seq_k {
                    let k_start = ki * d + head_offset;
                    let mut dot = 0.0_f64;
                    for j in 0..d_k {
                        dot += input.queries[q_start + j] * input.keys[k_start + j];
                    }
                    scores[ki] = dot * inv_scale;
                }

                // Apply causal mask: set future positions to -inf.
                if self.config.causal_mask {
                    for ki in (qi + 1)..seq_k {
                        scores[ki] = f64::NEG_INFINITY;
                    }
                }

                // Softmax.
                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_scores = vec![0.0_f64; seq_k];
                let mut sum_exp = 0.0_f64;
                for ki in 0..seq_k {
                    let e = (scores[ki] - max_score).exp();
                    exp_scores[ki] = e;
                    sum_exp += e;
                }
                if sum_exp > 0.0 {
                    for ki in 0..seq_k {
                        exp_scores[ki] /= sum_exp;
                    }
                }

                // Entropy of this attention distribution.
                let mut entropy = 0.0_f64;
                for &w in &exp_scores {
                    if w > 1e-30 {
                        entropy -= w * w.ln();
                    }
                }
                head_entropy_sums[head] += entropy;

                // Sparsity counting.
                for &w in &exp_scores {
                    if w < sparsity_threshold {
                        total_sparse_count += 1;
                    }
                }

                // Weighted sum of values: out[qi, head_offset..] = Σ_ki attn[ki] * V[ki, head_offset..]
                let o_start = qi * d + head_offset;
                for ki in 0..seq_k {
                    let v_start = ki * d + head_offset;
                    let w = exp_scores[ki];
                    for j in 0..d_k {
                        output[o_start + j] += w * input.values[v_start + j];
                    }
                }
            }
        }

        // Residual connection.
        if self.config.use_residual {
            for i in 0..output.len() {
                output[i] += input.queries[i];
            }
        }

        // Compute per-head average entropy.
        let head_entropies: Vec<f64> = head_entropy_sums
            .iter()
            .map(|&s| if seq_q > 0 { s / seq_q as f64 } else { 0.0 })
            .collect();
        let mean_entropy = if h > 0 {
            head_entropies.iter().sum::<f64>() / h as f64
        } else {
            0.0
        };

        let sparsity = if total_weight_count > 0 {
            total_sparse_count as f64 / total_weight_count as f64
        } else {
            0.0
        };

        // EMA smoothing of output.
        let smoothed_output = if self.config.ema_decay > 0.0 {
            if self.ema_initialized && self.ema_output.len() == output.len() {
                let alpha = self.config.ema_decay;
                for (ema, &val) in self.ema_output.iter_mut().zip(output.iter()) {
                    *ema = alpha * *ema + (1.0 - alpha) * val;
                }
            } else {
                self.ema_output = output.clone();
                self.ema_initialized = true;
            }
            Some(self.ema_output.clone())
        } else {
            None
        };

        // Update windowed history and stats.
        self.entropy_history.push_back(mean_entropy);
        while self.entropy_history.len() > self.config.window_size {
            self.entropy_history.pop_front();
        }

        self.stats.total_forwards += 1;
        self.stats.sum_entropy += mean_entropy;
        self.stats.sum_sparsity += sparsity;
        if self.stats.total_forwards == 1 {
            self.stats.max_entropy = mean_entropy;
            self.stats.min_entropy = mean_entropy;
        } else {
            if mean_entropy > self.stats.max_entropy {
                self.stats.max_entropy = mean_entropy;
            }
            if mean_entropy < self.stats.min_entropy {
                self.stats.min_entropy = mean_entropy;
            }
        }

        Ok(AttentionOutput {
            output,
            head_entropies,
            mean_entropy,
            sparsity,
            smoothed_output,
        })
    }

    // -- Accessors ---------------------------------------------------------

    /// Get aggregate statistics.
    pub fn stats(&self) -> &CrossAttentionStats {
        &self.stats
    }

    /// Windowed mean of recent mean-entropy values.
    pub fn windowed_mean_entropy(&self) -> Option<f64> {
        if self.entropy_history.is_empty() {
            return None;
        }
        let sum: f64 = self.entropy_history.iter().sum();
        Some(sum / self.entropy_history.len() as f64)
    }

    /// Windowed standard deviation of recent mean-entropy values.
    pub fn windowed_std_entropy(&self) -> Option<f64> {
        if self.entropy_history.len() < 2 {
            return None;
        }
        let mean = self.windowed_mean_entropy().unwrap();
        let var: f64 = self
            .entropy_history
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (self.entropy_history.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Get the per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.d_k
    }

    /// Reset all state (EMA, history, stats).
    pub fn reset(&mut self) {
        self.ema_output.clear();
        self.ema_initialized = false;
        self.entropy_history.clear();
        self.stats = CrossAttentionStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_config() -> CrossAttentionConfig {
        CrossAttentionConfig {
            d_model: 4,
            num_heads: 2,
            use_residual: false,
            causal_mask: false,
            ema_decay: 0.0,
            temperature: 1.0,
            window_size: 100,
            dropout_rate: 0.0,
        }
    }

    /// Build an identity-like input where Q=K=V are one-hot rows.
    fn identity_input(seq_len: usize, d_model: usize) -> AttentionInput {
        let mut data = vec![0.0_f64; seq_len * d_model];
        for i in 0..seq_len {
            let col = i % d_model;
            data[i * d_model + col] = 1.0;
        }
        AttentionInput {
            queries: data.clone(),
            keys: data.clone(),
            values: data,
            seq_q: seq_len,
            seq_k: seq_len,
        }
    }

    /// Build uniform input where all vectors are equal.
    fn uniform_input(seq_q: usize, seq_k: usize, d_model: usize, val: f64) -> AttentionInput {
        AttentionInput {
            queries: vec![val; seq_q * d_model],
            keys: vec![val; seq_k * d_model],
            values: vec![val; seq_k * d_model],
            seq_q,
            seq_k,
        }
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = CrossAttention::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_process_invalid_d_model_zero() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 0,
            ..Default::default()
        });
        assert!(ca.process().is_err());
    }

    #[test]
    fn test_process_invalid_num_heads_zero() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            num_heads: 0,
            ..Default::default()
        });
        assert!(ca.process().is_err());
    }

    #[test]
    fn test_process_invalid_d_model_not_divisible() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 7,
            num_heads: 3,
            ..Default::default()
        });
        assert!(ca.process().is_err());
    }

    #[test]
    fn test_process_invalid_temperature() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            temperature: 0.0,
            ..Default::default()
        });
        assert!(ca.process().is_err());
    }

    #[test]
    fn test_process_invalid_ema_decay() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(ca.process().is_err());
    }

    #[test]
    fn test_process_invalid_window_size() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(ca.process().is_err());
    }

    #[test]
    fn test_process_invalid_dropout_rate_negative() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            dropout_rate: -0.1,
            ..Default::default()
        });
        assert!(ca.process().is_err());
    }

    #[test]
    fn test_process_invalid_dropout_rate_above_one() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            dropout_rate: 1.1,
            ..Default::default()
        });
        assert!(ca.process().is_err());
    }

    #[test]
    fn test_process_valid_ema_decay_zero() {
        // ema_decay=0 should be valid (means EMA is disabled)
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(ca.process().is_ok());
    }

    // -- Input validation --------------------------------------------------

    #[test]
    fn test_forward_wrong_query_length() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = AttentionInput {
            queries: vec![1.0; 3], // should be 2*4=8
            keys: vec![1.0; 8],
            values: vec![1.0; 8],
            seq_q: 2,
            seq_k: 2,
        };
        assert!(ca.forward(&input).is_err());
    }

    #[test]
    fn test_forward_wrong_key_length() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = AttentionInput {
            queries: vec![1.0; 8],
            keys: vec![1.0; 3],
            values: vec![1.0; 8],
            seq_q: 2,
            seq_k: 2,
        };
        assert!(ca.forward(&input).is_err());
    }

    #[test]
    fn test_forward_wrong_value_length() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = AttentionInput {
            queries: vec![1.0; 8],
            keys: vec![1.0; 8],
            values: vec![1.0; 3],
            seq_q: 2,
            seq_k: 2,
        };
        assert!(ca.forward(&input).is_err());
    }

    #[test]
    fn test_forward_zero_seq_q() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = AttentionInput {
            queries: vec![],
            keys: vec![1.0; 4],
            values: vec![1.0; 4],
            seq_q: 0,
            seq_k: 1,
        };
        assert!(ca.forward(&input).is_err());
    }

    #[test]
    fn test_forward_zero_seq_k() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = AttentionInput {
            queries: vec![1.0; 4],
            keys: vec![],
            values: vec![],
            seq_q: 1,
            seq_k: 0,
        };
        assert!(ca.forward(&input).is_err());
    }

    // -- Forward pass basics -----------------------------------------------

    #[test]
    fn test_forward_output_shape() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = identity_input(3, 4);
        let out = ca.forward(&input).unwrap();
        assert_eq!(
            out.output.len(),
            3 * 4,
            "output should be [seq_q × d_model]"
        );
    }

    #[test]
    fn test_forward_uniform_input_uniform_output() {
        // When Q=K=V are all the same constant vector, attention weights are
        // uniform and the output should equal the value vectors (since the
        // weighted average of identical vectors is the same vector).
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 4,
            num_heads: 2,
            use_residual: false,
            ..simple_config()
        });
        let input = uniform_input(2, 3, 4, 1.0);
        let out = ca.forward(&input).unwrap();

        // Each output position should be the average of value vectors.
        // Since all value vectors are [1,1,1,1], the output should be [1,1,1,1].
        for i in 0..out.output.len() {
            assert!(
                (out.output[i] - 1.0).abs() < 1e-10,
                "uniform input should produce uniform output, got {} at index {}",
                out.output[i],
                i
            );
        }
    }

    #[test]
    fn test_forward_single_position() {
        // With seq_q=1, seq_k=1, attention weight is always 1.0, so
        // output = value (optionally + query for residual).
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            use_residual: false,
            ..simple_config()
        });
        let input = AttentionInput {
            queries: vec![1.0, 2.0, 3.0, 4.0],
            keys: vec![0.5, 0.5, 0.5, 0.5],
            values: vec![10.0, 20.0, 30.0, 40.0],
            seq_q: 1,
            seq_k: 1,
        };
        let out = ca.forward(&input).unwrap();
        // With a single key, softmax weight = 1.0, so output = value
        assert!(
            (out.output[0] - 10.0).abs() < 1e-10,
            "output[0] should be 10.0, got {}",
            out.output[0]
        );
        assert!(
            (out.output[1] - 20.0).abs() < 1e-10,
            "output[1] should be 20.0, got {}",
            out.output[1]
        );
        assert!(
            (out.output[2] - 30.0).abs() < 1e-10,
            "output[2] should be 30.0, got {}",
            out.output[2]
        );
        assert!(
            (out.output[3] - 40.0).abs() < 1e-10,
            "output[3] should be 40.0, got {}",
            out.output[3]
        );
    }

    // -- Residual connection -----------------------------------------------

    #[test]
    fn test_residual_connection() {
        let mut ca_no_res = CrossAttention::with_config(CrossAttentionConfig {
            use_residual: false,
            ..simple_config()
        });
        let mut ca_res = CrossAttention::with_config(CrossAttentionConfig {
            use_residual: true,
            ..simple_config()
        });

        let input = AttentionInput {
            queries: vec![1.0, 2.0, 3.0, 4.0],
            keys: vec![0.5, 0.5, 0.5, 0.5],
            values: vec![10.0, 20.0, 30.0, 40.0],
            seq_q: 1,
            seq_k: 1,
        };

        let out_no = ca_no_res.forward(&input).unwrap();
        let out_yes = ca_res.forward(&input).unwrap();

        // Residual = attention_output + query
        for i in 0..4 {
            let expected = out_no.output[i] + input.queries[i];
            assert!(
                (out_yes.output[i] - expected).abs() < 1e-10,
                "residual output[{}] should be {} + {} = {}, got {}",
                i,
                out_no.output[i],
                input.queries[i],
                expected,
                out_yes.output[i]
            );
        }
    }

    // -- Causal masking ----------------------------------------------------

    #[test]
    fn test_causal_mask_first_position_sees_only_first() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            causal_mask: true,
            use_residual: false,
            ..simple_config()
        });

        // seq_q=2, seq_k=2: position 0 should only attend to key 0
        let input = AttentionInput {
            queries: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            keys: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            values: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            seq_q: 2,
            seq_k: 2,
        };

        let out = ca.forward(&input).unwrap();
        // Position 0 can only see key 0 → output should be value[0] = [10, 20, 30, 40]
        assert!(
            (out.output[0] - 10.0).abs() < 1e-10,
            "causal: pos 0 should only see value 0, got {}",
            out.output[0]
        );
        assert!(
            (out.output[1] - 20.0).abs() < 1e-10,
            "causal: pos 0 output[1] should be 20.0, got {}",
            out.output[1]
        );
    }

    #[test]
    fn test_causal_mask_second_position_sees_both() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            causal_mask: true,
            use_residual: false,
            ..simple_config()
        });

        // Use uniform vectors so both keys have equal dot products with the query
        let input = uniform_input(2, 2, 4, 1.0);
        let out = ca.forward(&input).unwrap();

        // Position 1 can see keys 0 and 1 (both equal), so output should be
        // the average of values (which are all 1.0), giving 1.0.
        for j in 0..4 {
            assert!(
                (out.output[4 + j] - 1.0).abs() < 1e-10,
                "causal: pos 1 with uniform input should see average, got {} at dim {}",
                out.output[4 + j],
                j
            );
        }
    }

    #[test]
    fn test_no_causal_mask_all_positions_see_all() {
        let mut ca_causal = CrossAttention::with_config(CrossAttentionConfig {
            causal_mask: true,
            use_residual: false,
            ..simple_config()
        });
        let mut ca_full = CrossAttention::with_config(CrossAttentionConfig {
            causal_mask: false,
            use_residual: false,
            ..simple_config()
        });

        // With different values at each position, causal masking should
        // produce a different result for position 0 (restricted) vs full
        // attention (can see everything).
        let input = AttentionInput {
            queries: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            keys: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            seq_q: 2,
            seq_k: 2,
        };

        let out_causal = ca_causal.forward(&input).unwrap();
        let out_full = ca_full.forward(&input).unwrap();

        // Position 0 with causal mask: sees only value[0] = [1,0,0,0]
        // Position 0 without mask: sees average of value[0] and value[1]
        // These should differ
        let causal_pos0 = &out_causal.output[0..4];
        let full_pos0 = &out_full.output[0..4];

        let diff: f64 = causal_pos0
            .iter()
            .zip(full_pos0.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-10,
            "causal and full attention should differ for position 0"
        );
    }

    // -- Temperature scaling -----------------------------------------------

    #[test]
    fn test_high_temperature_sharpens() {
        // High temperature → sharper attention → lower entropy
        let mut ca_normal = CrossAttention::with_config(CrossAttentionConfig {
            temperature: 1.0,
            use_residual: false,
            ..simple_config()
        });
        let mut ca_sharp = CrossAttention::with_config(CrossAttentionConfig {
            temperature: 5.0,
            use_residual: false,
            ..simple_config()
        });

        // Use input where one key has a much higher dot product with the query
        let input = AttentionInput {
            queries: vec![1.0, 0.0, 0.0, 0.0],
            keys: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            values: vec![10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            seq_q: 1,
            seq_k: 2,
        };

        let out_normal = ca_normal.forward(&input).unwrap();
        let out_sharp = ca_sharp.forward(&input).unwrap();

        // Sharper attention should have lower entropy
        assert!(
            out_sharp.mean_entropy <= out_normal.mean_entropy + 1e-10,
            "higher temperature should sharpen attention: normal_entropy={}, sharp_entropy={}",
            out_normal.mean_entropy,
            out_sharp.mean_entropy
        );
    }

    // -- Entropy and sparsity ----------------------------------------------

    #[test]
    fn test_uniform_attention_max_entropy() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            use_residual: false,
            ..simple_config()
        });

        // Uniform input → uniform attention weights → max entropy
        let input = uniform_input(1, 4, 4, 1.0);
        let out = ca.forward(&input).unwrap();

        // Max entropy for 4 positions = ln(4) ≈ 1.386
        let max_entropy = (4.0_f64).ln();
        assert!(
            (out.mean_entropy - max_entropy).abs() < 0.01,
            "uniform attention should have max entropy ~{}, got {}",
            max_entropy,
            out.mean_entropy
        );
    }

    #[test]
    fn test_focused_attention_low_entropy() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            use_residual: false,
            d_model: 4,
            num_heads: 1,
            temperature: 1.0,
            ..simple_config()
        });

        // Query strongly matches only one key → focused attention → low entropy
        let input = AttentionInput {
            queries: vec![10.0, 0.0, 0.0, 0.0],
            keys: vec![
                10.0, 0.0, 0.0, 0.0, // strong match
                0.0, 0.0, 0.0, 0.0, // no match
                0.0, 0.0, 0.0, 0.0, // no match
            ],
            values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            seq_q: 1,
            seq_k: 3,
        };

        let out = ca.forward(&input).unwrap();
        let max_entropy = (3.0_f64).ln();
        assert!(
            out.mean_entropy < max_entropy * 0.5,
            "focused attention should have low entropy: got {}, max={}",
            out.mean_entropy,
            max_entropy
        );
    }

    #[test]
    fn test_sparsity_metric() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = identity_input(2, 4);
        let out = ca.forward(&input).unwrap();
        assert!(
            out.sparsity >= 0.0 && out.sparsity <= 1.0,
            "sparsity should be in [0, 1], got {}",
            out.sparsity
        );
    }

    #[test]
    fn test_head_entropies_count() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 8,
            num_heads: 4,
            use_residual: false,
            ..simple_config()
        });
        let input = identity_input(2, 8);
        let out = ca.forward(&input).unwrap();
        assert_eq!(
            out.head_entropies.len(),
            4,
            "should have one entropy per head"
        );
    }

    // -- EMA smoothing -----------------------------------------------------

    #[test]
    fn test_ema_disabled_when_decay_zero() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            ema_decay: 0.0,
            use_residual: false,
            ..simple_config()
        });
        let input = identity_input(2, 4);
        let out = ca.forward(&input).unwrap();
        assert!(
            out.smoothed_output.is_none(),
            "EMA should be disabled when decay=0"
        );
    }

    #[test]
    fn test_ema_enabled_when_decay_positive() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            ema_decay: 0.5,
            use_residual: false,
            ..simple_config()
        });
        let input = identity_input(2, 4);
        let out = ca.forward(&input).unwrap();
        assert!(
            out.smoothed_output.is_some(),
            "EMA should be enabled when decay > 0"
        );
    }

    #[test]
    fn test_ema_initializes_to_first_output() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            ema_decay: 0.8,
            use_residual: false,
            ..simple_config()
        });
        let input = identity_input(2, 4);
        let out = ca.forward(&input).unwrap();
        let smoothed = out.smoothed_output.unwrap();
        // First forward: EMA should equal raw output
        for (s, &o) in smoothed.iter().zip(out.output.iter()) {
            assert!((s - o).abs() < 1e-10, "first EMA should equal raw output");
        }
    }

    #[test]
    fn test_ema_lags_behind_changes() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            ema_decay: 0.9,
            use_residual: false,
            ..simple_config()
        });

        // First forward with one set of values
        let input1 = AttentionInput {
            queries: vec![1.0; 4],
            keys: vec![1.0; 4],
            values: vec![1.0; 4],
            seq_q: 1,
            seq_k: 1,
        };
        ca.forward(&input1).unwrap();

        // Second forward with very different values
        let input2 = AttentionInput {
            queries: vec![1.0; 4],
            keys: vec![1.0; 4],
            values: vec![100.0; 4],
            seq_q: 1,
            seq_k: 1,
        };
        let out2 = ca.forward(&input2).unwrap();
        let smoothed = out2.smoothed_output.unwrap();

        // Smoothed should lag behind: closer to 1.0 than to 100.0
        // EMA = 0.9 * 1.0 + 0.1 * 100.0 = 10.9
        for &s in &smoothed {
            assert!(
                (s - 10.9).abs() < 1e-8,
                "EMA should lag: expected ~10.9, got {}",
                s
            );
        }
    }

    // -- Multi-head behaviour ----------------------------------------------

    #[test]
    fn test_single_head_vs_multi_head() {
        // With uniform input, single-head and multi-head should give the same result
        let mut ca_single = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 4,
            num_heads: 1,
            use_residual: false,
            ..simple_config()
        });
        let mut ca_multi = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 4,
            num_heads: 4,
            use_residual: false,
            ..simple_config()
        });

        let input = uniform_input(2, 3, 4, 1.0);
        let out_single = ca_single.forward(&input).unwrap();
        let out_multi = ca_multi.forward(&input).unwrap();

        for (a, b) in out_single.output.iter().zip(out_multi.output.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "uniform input should give same result regardless of head count"
            );
        }
    }

    #[test]
    fn test_head_dim() {
        let ca = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 12,
            num_heads: 3,
            ..Default::default()
        });
        assert_eq!(ca.head_dim(), 4);
    }

    // -- Cross-attention (seq_q != seq_k) ----------------------------------

    #[test]
    fn test_cross_attention_different_lengths() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            use_residual: false,
            ..simple_config()
        });

        // seq_q=1, seq_k=3
        let input = AttentionInput {
            queries: vec![1.0, 0.0, 0.0, 0.0],
            keys: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            values: vec![
                10.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0,
            ],
            seq_q: 1,
            seq_k: 3,
        };

        let out = ca.forward(&input).unwrap();
        assert_eq!(out.output.len(), 4);
    }

    // -- Statistics --------------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = identity_input(2, 4);

        ca.forward(&input).unwrap();
        ca.forward(&input).unwrap();
        ca.forward(&input).unwrap();

        assert_eq!(ca.stats().total_forwards, 3);
        assert!(ca.stats().avg_entropy() > 0.0);
        assert!(ca.stats().avg_sparsity() >= 0.0);
    }

    #[test]
    fn test_stats_min_max_entropy() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 4,
            num_heads: 1,
            use_residual: false,
            ..simple_config()
        });

        // First: uniform → high entropy
        let input1 = uniform_input(1, 4, 4, 1.0);
        ca.forward(&input1).unwrap();

        // Second: focused → low entropy
        let input2 = AttentionInput {
            queries: vec![10.0, 0.0, 0.0, 0.0],
            keys: vec![
                10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            values: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            seq_q: 1,
            seq_k: 4,
        };
        ca.forward(&input2).unwrap();

        assert!(
            ca.stats().max_entropy > ca.stats().min_entropy,
            "max entropy should be > min entropy"
        );
    }

    // -- Windowed statistics -----------------------------------------------

    #[test]
    fn test_windowed_mean_entropy() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = uniform_input(1, 2, 4, 1.0);
        ca.forward(&input).unwrap();
        ca.forward(&input).unwrap();
        let mean = ca.windowed_mean_entropy().unwrap();
        assert!(mean > 0.0);
    }

    #[test]
    fn test_windowed_std_entropy_constant() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = uniform_input(1, 2, 4, 1.0);
        for _ in 0..5 {
            ca.forward(&input).unwrap();
        }
        let std = ca.windowed_std_entropy().unwrap();
        assert!(
            std < 1e-10,
            "constant entropy should have zero std, got {}",
            std
        );
    }

    #[test]
    fn test_windowed_mean_empty() {
        let ca = CrossAttention::with_config(simple_config());
        assert!(ca.windowed_mean_entropy().is_none());
    }

    #[test]
    fn test_windowed_std_insufficient() {
        let mut ca = CrossAttention::with_config(simple_config());
        let input = uniform_input(1, 2, 4, 1.0);
        ca.forward(&input).unwrap();
        assert!(ca.windowed_std_entropy().is_none());
    }

    // -- Reset -------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            ema_decay: 0.5,
            ..simple_config()
        });
        let input = identity_input(2, 4);
        for _ in 0..5 {
            ca.forward(&input).unwrap();
        }
        assert!(ca.stats().total_forwards > 0);

        ca.reset();

        assert_eq!(ca.stats().total_forwards, 0);
        assert!(ca.windowed_mean_entropy().is_none());
    }

    // -- Window eviction ---------------------------------------------------

    #[test]
    fn test_window_eviction() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            window_size: 3,
            ..simple_config()
        });
        let input = uniform_input(1, 2, 4, 1.0);
        for _ in 0..10 {
            ca.forward(&input).unwrap();
        }
        assert_eq!(ca.entropy_history.len(), 3);
    }

    // -- Large multi-head --------------------------------------------------

    #[test]
    fn test_large_multi_head() {
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            d_model: 16,
            num_heads: 8,
            use_residual: true,
            causal_mask: false,
            ..simple_config()
        });
        let input = identity_input(4, 16);
        let out = ca.forward(&input).unwrap();
        assert_eq!(out.output.len(), 4 * 16);
        assert_eq!(out.head_entropies.len(), 8);
    }

    // -- Attention weights sum to 1 ----------------------------------------

    #[test]
    fn test_attention_output_bounded() {
        // With values in [0, 1] and no residual, output should be in [0, 1]
        // since it's a convex combination of value vectors.
        let mut ca = CrossAttention::with_config(CrossAttentionConfig {
            use_residual: false,
            ..simple_config()
        });
        let input = AttentionInput {
            queries: vec![0.5, 0.3, 0.7, 0.1],
            keys: vec![0.5, 0.3, 0.7, 0.1, 0.2, 0.8, 0.4, 0.6],
            values: vec![0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0],
            seq_q: 1,
            seq_k: 2,
        };
        let out = ca.forward(&input).unwrap();
        for &v in &out.output {
            assert!(
                (-1e-10..=1.0 + 1e-10).contains(&v),
                "output should be bounded by value range, got {}",
                v
            );
        }
    }

    // -- Stats defaults ----------------------------------------------------

    #[test]
    fn test_stats_defaults() {
        let ca = CrossAttention::with_config(simple_config());
        assert_eq!(ca.stats().total_forwards, 0);
        assert_eq!(ca.stats().avg_entropy(), 0.0);
        assert_eq!(ca.stats().avg_sparsity(), 0.0);
    }
}
