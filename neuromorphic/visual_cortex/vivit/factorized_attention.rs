//! Factorized Space-Time Attention
//!
//! Part of the Visual Cortex region — ViViT component.
//!
//! Implements **Model 3** from the ViViT paper (Arnab et al., 2021):
//! factorized self-attention that decomposes full spatiotemporal attention
//! into separate spatial and temporal stages, reducing complexity from
//! O((n_s · n_t)²) to O(n_s² + n_t²).
//!
//! # Architecture
//!
//! ```text
//! Input: [B, T, S, D]  (batch, temporal tokens, spatial tokens, embedding dim)
//!         │
//!    ┌────▼────┐
//!    │ Spatial  │   Attend within each frame  → [B, T, S, D]
//!    │ Attention│   Q_s, K_s, V_s per frame
//!    └────┬────┘
//!         │
//!    ┌────▼────┐
//!    │Temporal  │   Attend across frames per spatial position → [B, T, S, D]
//!    │Attention │   Q_t, K_t, V_t per position
//!    └────┬────┘
//!         │
//!    Output: [B, T, S, D]
//! ```
//!
//! # Trading context
//!
//! - **Spatial dimension**: different market features (price, volume, spread, …)
//! - **Temporal dimension**: time steps / candle windows
//! - Factorisation lets the model first fuse features within a single time step
//!   (spatial attention) and then reason across time (temporal attention).

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the factorized attention module.
#[derive(Debug, Clone)]
pub struct FactorizedAttentionConfig {
    /// Embedding dimension (D). Must be divisible by `num_heads`.
    pub embedding_dim: usize,

    /// Number of attention heads for both spatial and temporal stages.
    pub num_heads: usize,

    /// Whether to use causal masking in the temporal stage (prevent look-ahead).
    pub causal_temporal: bool,

    /// Scaling factor override.  When `None`, defaults to `1 / sqrt(head_dim)`.
    pub scale: Option<f64>,

    /// EMA decay factor for throughput / quality tracking.  Must be in (0, 1).
    pub ema_decay: f64,

    /// Number of samples before EMA is considered warmed-up.
    pub min_samples: usize,

    /// Rolling window size for windowed diagnostics.
    pub window_size: usize,
}

impl Default for FactorizedAttentionConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            num_heads: 4,
            causal_temporal: false,
            scale: None,
            ema_decay: 0.05,
            min_samples: 10,
            window_size: 100,
        }
    }
}

// ─── Attention weight matrix ────────────────────────────────────────────────

/// A row-major dense matrix stored as a flat `Vec<f64>`.
#[derive(Debug, Clone)]
struct Mat {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn xavier(rows: usize, cols: usize) -> Self {
        // Deterministic Xavier-style init: alternating +/- with scale
        let scale = (2.0 / (rows + cols) as f64).sqrt();
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                // Simple deterministic pseudo-random using a hash-like mix
                let mix = ((i as f64 + 1.0).sin() * 43758.5453).fract();
                sign * mix.abs() * scale
            })
            .collect();
        Self { data, rows, cols }
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Matrix multiply: self [R, K] × other [K, C] → [R, C]
    fn matmul(&self, other: &Mat) -> Mat {
        debug_assert_eq!(self.cols, other.rows);
        let mut out = Mat::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[r * self.cols + k];
                if a == 0.0 {
                    continue;
                }
                for c in 0..other.cols {
                    out.data[r * other.cols + c] += a * other.data[k * other.cols + c];
                }
            }
        }
        out
    }

    /// Transpose → [cols, rows]
    fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Add bias vector (length == cols) to every row.
    fn add_bias(&mut self, bias: &[f64]) {
        debug_assert_eq!(bias.len(), self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                self.data[r * self.cols + c] += bias[c];
            }
        }
    }

    /// Extract row `r` as a slice.
    #[allow(dead_code)]
    fn row(&self, r: usize) -> &[f64] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Row-wise softmax in-place.
    fn softmax_rows(&mut self) {
        for r in 0..self.rows {
            let start = r * self.cols;
            let end = start + self.cols;
            let row = &mut self.data[start..end];
            let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }

    /// Apply a causal mask: set positions where col > row to -infinity *before*
    /// softmax.
    fn apply_causal_mask(&mut self) {
        for r in 0..self.rows {
            for c in (r + 1)..self.cols {
                self.set(r, c, f64::NEG_INFINITY);
            }
        }
    }
}

// ─── Projection weights for one attention stage ─────────────────────────────

#[derive(Debug, Clone)]
struct AttentionProjections {
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    bq: Vec<f64>,
    bk: Vec<f64>,
    bv: Vec<f64>,
    bo: Vec<f64>,
}

impl AttentionProjections {
    fn new(dim: usize) -> Self {
        Self {
            wq: Mat::xavier(dim, dim),
            wk: Mat::xavier(dim, dim),
            wv: Mat::xavier(dim, dim),
            wo: Mat::xavier(dim, dim),
            bq: vec![0.0; dim],
            bk: vec![0.0; dim],
            bv: vec![0.0; dim],
            bo: vec![0.0; dim],
        }
    }

    fn project(&self, input: &Mat) -> (Mat, Mat, Mat) {
        let mut q = input.matmul(&self.wq);
        q.add_bias(&self.bq);
        let mut k = input.matmul(&self.wk);
        k.add_bias(&self.bk);
        let mut v = input.matmul(&self.wv);
        v.add_bias(&self.bv);
        (q, k, v)
    }

    fn output_proj(&self, x: &Mat) -> Mat {
        let mut out = x.matmul(&self.wo);
        out.add_bias(&self.bo);
        out
    }
}

// ─── Single-stage multi-head attention ──────────────────────────────────────

/// Compute scaled dot-product multi-head attention.
///
/// * `input` – [N, D] token embeddings
/// * `projs` – Q/K/V/O projection weights
/// * `num_heads` – number of heads
/// * `scale` – scaling factor (1/√d_k)
/// * `causal` – whether to apply a causal mask
///
/// Returns `(output [N, D], attention_weights [N, N])`.
fn multi_head_attention(
    input: &Mat,
    projs: &AttentionProjections,
    num_heads: usize,
    scale: f64,
    causal: bool,
) -> (Mat, Mat) {
    let n = input.rows;
    let d = input.cols;
    let head_dim = d / num_heads;

    // Project Q, K, V
    let (q, k, v) = projs.project(input);

    // Accumulate output across heads
    let mut output = Mat::zeros(n, d);
    // Average attention weights across heads for diagnostics
    let mut avg_attn = Mat::zeros(n, n);

    for h in 0..num_heads {
        let offset = h * head_dim;

        // Extract head slice as [N, head_dim] matrices
        let mut q_h = Mat::zeros(n, head_dim);
        let mut k_h = Mat::zeros(n, head_dim);
        let mut v_h = Mat::zeros(n, head_dim);
        for i in 0..n {
            for j in 0..head_dim {
                q_h.set(i, j, q.get(i, offset + j));
                k_h.set(i, j, k.get(i, offset + j));
                v_h.set(i, j, v.get(i, offset + j));
            }
        }

        // Attention scores: Q_h · K_h^T / scale → [N, N]
        let k_t = k_h.transpose();
        let mut scores = q_h.matmul(&k_t);
        // Scale
        for val in scores.data.iter_mut() {
            *val *= scale;
        }

        // Optional causal mask
        if causal {
            scores.apply_causal_mask();
        }

        // Softmax
        scores.softmax_rows();

        // Accumulate average attention
        for i in 0..n * n {
            avg_attn.data[i] += scores.data[i];
        }

        // Weighted sum: scores · V_h → [N, head_dim]
        let context = scores.matmul(&v_h);

        // Write back into full-D output
        for i in 0..n {
            for j in 0..head_dim {
                let idx = i * d + offset + j;
                output.data[idx] += context.get(i, j);
            }
        }
    }

    // Average attention weights over heads
    let heads_f = num_heads as f64;
    for val in avg_attn.data.iter_mut() {
        *val /= heads_f;
    }

    // Output projection
    let projected = projs.output_proj(&output);

    (projected, avg_attn)
}

// ─── Attention entropy ──────────────────────────────────────────────────────

/// Compute the mean attention entropy across all query positions.
/// Higher entropy → more uniform attention; lower → more peaked.
fn attention_entropy(weights: &Mat) -> f64 {
    let mut total = 0.0;
    for r in 0..weights.rows {
        let mut h = 0.0;
        for c in 0..weights.cols {
            let p = weights.get(r, c);
            if p > 1e-12 {
                h -= p * p.ln();
            }
        }
        total += h;
    }
    total / weights.rows.max(1) as f64
}

// ─── Public output types ────────────────────────────────────────────────────

/// Result of a single forward pass through factorized attention.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Output tensor flattened as [B * T * S, D].
    pub output: Vec<f64>,

    /// Batch size that was processed.
    pub batch_size: usize,

    /// Temporal sequence length.
    pub num_temporal: usize,

    /// Spatial sequence length.
    pub num_spatial: usize,

    /// Embedding dimension.
    pub embedding_dim: usize,

    /// Mean spatial attention entropy (averaged over batch & time).
    pub spatial_entropy: f64,

    /// Mean temporal attention entropy (averaged over batch & spatial).
    pub temporal_entropy: f64,

    /// Total tokens processed: B * T * S.
    pub total_tokens: usize,
}

impl ForwardResult {
    /// Retrieve the embedding vector for the given batch / time / spatial index.
    pub fn get(&self, b: usize, t: usize, s: usize) -> Option<&[f64]> {
        if b >= self.batch_size || t >= self.num_temporal || s >= self.num_spatial {
            return None;
        }
        let idx = (b * self.num_temporal * self.num_spatial + t * self.num_spatial + s)
            * self.embedding_dim;
        Some(&self.output[idx..idx + self.embedding_dim])
    }
}

/// Windowed record for EMA / diagnostics.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WindowRecord {
    spatial_entropy: f64,
    temporal_entropy: f64,
    total_tokens: usize,
    tick: u64,
}

/// Running statistics.
#[derive(Debug, Clone, Default)]
pub struct FactorizedAttentionStats {
    /// Number of forward passes executed.
    pub total_forwards: u64,

    /// Total tokens processed across all forward passes.
    pub total_tokens: u64,

    /// Peak spatial attention entropy observed.
    pub peak_spatial_entropy: f64,

    /// Peak temporal attention entropy observed.
    pub peak_temporal_entropy: f64,

    /// Running sum of spatial entropy (for mean computation).
    pub sum_spatial_entropy: f64,

    /// Running sum of temporal entropy.
    pub sum_temporal_entropy: f64,

    /// Smallest batch size seen.
    pub min_batch_size: usize,

    /// Largest batch size seen.
    pub max_batch_size: usize,
}

impl FactorizedAttentionStats {
    /// Mean spatial attention entropy across all forward passes.
    pub fn mean_spatial_entropy(&self) -> f64 {
        if self.total_forwards == 0 {
            return 0.0;
        }
        self.sum_spatial_entropy / self.total_forwards as f64
    }

    /// Mean temporal attention entropy across all forward passes.
    pub fn mean_temporal_entropy(&self) -> f64 {
        if self.total_forwards == 0 {
            return 0.0;
        }
        self.sum_temporal_entropy / self.total_forwards as f64
    }
}

// ─── Main struct ────────────────────────────────────────────────────────────

/// Factorized space-time attention module.
///
/// Holds separate projection weights for spatial and temporal attention stages
/// and tracks attention quality metrics via EMA and a rolling window.
pub struct FactorizedAttention {
    config: FactorizedAttentionConfig,

    /// Projection weights for the spatial attention stage.
    spatial_projs: AttentionProjections,

    /// Projection weights for the temporal attention stage.
    temporal_projs: AttentionProjections,

    /// Computed scale factor: 1 / sqrt(head_dim).
    scale: f64,

    // EMA state
    ema_spatial_entropy: f64,
    ema_temporal_entropy: f64,
    ema_initialized: bool,

    // Rolling window
    recent: VecDeque<WindowRecord>,

    // Tick counter
    current_tick: u64,

    // Stats
    stats: FactorizedAttentionStats,
}

impl Default for FactorizedAttention {
    fn default() -> Self {
        Self::new()
    }
}

impl FactorizedAttention {
    /// Create an instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(FactorizedAttentionConfig::default())
            .expect("default config must be valid")
    }

    /// Create an instance from a validated configuration.
    pub fn with_config(config: FactorizedAttentionConfig) -> Result<Self> {
        if config.embedding_dim == 0 {
            return Err(Error::InvalidInput("embedding_dim must be > 0".into()));
        }
        if config.num_heads == 0 {
            return Err(Error::InvalidInput("num_heads must be > 0".into()));
        }
        if config.embedding_dim % config.num_heads != 0 {
            return Err(Error::InvalidInput(format!(
                "embedding_dim ({}) must be divisible by num_heads ({})",
                config.embedding_dim, config.num_heads,
            )));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }

        let head_dim = config.embedding_dim / config.num_heads;
        let scale = config
            .scale
            .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());

        let spatial_projs = AttentionProjections::new(config.embedding_dim);
        let temporal_projs = AttentionProjections::new(config.embedding_dim);

        Ok(Self {
            config,
            spatial_projs,
            temporal_projs,
            scale,
            ema_spatial_entropy: 0.0,
            ema_temporal_entropy: 0.0,
            ema_initialized: false,
            recent: VecDeque::new(),
            current_tick: 0,
            stats: FactorizedAttentionStats::default(),
        })
    }

    // ── Forward pass ────────────────────────────────────────────────────

    /// Run a forward pass through the factorized attention.
    ///
    /// # Arguments
    ///
    /// * `input` – flat f64 slice of shape `[B, T, S, D]` in row-major order.
    /// * `batch_size` – B
    /// * `num_temporal` – T (number of time steps / frames)
    /// * `num_spatial` – S (number of spatial tokens per frame)
    ///
    /// The `embedding_dim` (D) is taken from the config.
    ///
    /// # Errors
    ///
    /// Returns an error if the input length doesn't match `B * T * S * D`.
    pub fn forward(
        &mut self,
        input: &[f64],
        batch_size: usize,
        num_temporal: usize,
        num_spatial: usize,
    ) -> Result<ForwardResult> {
        let d = self.config.embedding_dim;
        let expected = batch_size * num_temporal * num_spatial * d;
        if input.len() != expected {
            return Err(Error::InvalidInput(format!(
                "expected input length {} ({}*{}*{}*{}), got {}",
                expected,
                batch_size,
                num_temporal,
                num_spatial,
                d,
                input.len(),
            )));
        }
        if batch_size == 0 || num_temporal == 0 || num_spatial == 0 {
            return Err(Error::InvalidInput(
                "batch_size, num_temporal, and num_spatial must all be > 0".into(),
            ));
        }

        let total_tokens = batch_size * num_temporal * num_spatial;

        // Allocate output buffer (same shape as input)
        let mut output = vec![0.0; expected];

        let mut spatial_entropy_sum = 0.0;
        let mut spatial_entropy_count = 0usize;
        let mut temporal_entropy_sum = 0.0;
        let mut temporal_entropy_count = 0usize;

        // ── Stage 1: Spatial attention ──
        // For each (b, t), apply MHA over the S spatial tokens.
        for b in 0..batch_size {
            for t in 0..num_temporal {
                // Build [S, D] matrix for this frame
                let mut mat = Mat::zeros(num_spatial, d);
                for s in 0..num_spatial {
                    let base = ((b * num_temporal + t) * num_spatial + s) * d;
                    for j in 0..d {
                        mat.set(s, j, input[base + j]);
                    }
                }

                let (attended, attn_w) = multi_head_attention(
                    &mat,
                    &self.spatial_projs,
                    self.config.num_heads,
                    self.scale,
                    false, // spatial attention is never causal
                );

                // Add residual connection
                for s in 0..num_spatial {
                    let base = ((b * num_temporal + t) * num_spatial + s) * d;
                    for j in 0..d {
                        output[base + j] = attended.get(s, j) + input[base + j];
                    }
                }

                spatial_entropy_sum += attention_entropy(&attn_w);
                spatial_entropy_count += 1;
            }
        }

        // ── Stage 2: Temporal attention ──
        // For each (b, s), apply MHA over the T temporal tokens.
        // We work on the intermediate `output` from Stage 1.
        let stage1 = output.clone();
        for b in 0..batch_size {
            for s in 0..num_spatial {
                // Build [T, D] matrix for this spatial position across time
                let mut mat = Mat::zeros(num_temporal, d);
                for t in 0..num_temporal {
                    let base = ((b * num_temporal + t) * num_spatial + s) * d;
                    for j in 0..d {
                        mat.set(t, j, stage1[base + j]);
                    }
                }

                let (attended, attn_w) = multi_head_attention(
                    &mat,
                    &self.temporal_projs,
                    self.config.num_heads,
                    self.scale,
                    self.config.causal_temporal,
                );

                // Add residual connection (from stage-1 output)
                for t in 0..num_temporal {
                    let base = ((b * num_temporal + t) * num_spatial + s) * d;
                    for j in 0..d {
                        output[base + j] = attended.get(t, j) + stage1[base + j];
                    }
                }

                temporal_entropy_sum += attention_entropy(&attn_w);
                temporal_entropy_count += 1;
            }
        }

        let spatial_entropy = if spatial_entropy_count > 0 {
            spatial_entropy_sum / spatial_entropy_count as f64
        } else {
            0.0
        };
        let temporal_entropy = if temporal_entropy_count > 0 {
            temporal_entropy_sum / temporal_entropy_count as f64
        } else {
            0.0
        };

        // Update EMA
        self.update_ema(spatial_entropy, temporal_entropy);

        // Update window
        self.current_tick += 1;
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(WindowRecord {
            spatial_entropy,
            temporal_entropy,
            total_tokens,
            tick: self.current_tick,
        });

        // Update stats
        self.stats.total_forwards += 1;
        self.stats.total_tokens += total_tokens as u64;
        self.stats.sum_spatial_entropy += spatial_entropy;
        self.stats.sum_temporal_entropy += temporal_entropy;
        if spatial_entropy > self.stats.peak_spatial_entropy {
            self.stats.peak_spatial_entropy = spatial_entropy;
        }
        if temporal_entropy > self.stats.peak_temporal_entropy {
            self.stats.peak_temporal_entropy = temporal_entropy;
        }
        if self.stats.total_forwards == 1 {
            self.stats.min_batch_size = batch_size;
            self.stats.max_batch_size = batch_size;
        } else {
            if batch_size < self.stats.min_batch_size {
                self.stats.min_batch_size = batch_size;
            }
            if batch_size > self.stats.max_batch_size {
                self.stats.max_batch_size = batch_size;
            }
        }

        Ok(ForwardResult {
            output,
            batch_size,
            num_temporal,
            num_spatial,
            embedding_dim: d,
            spatial_entropy,
            temporal_entropy,
            total_tokens,
        })
    }

    // ── EMA helpers ─────────────────────────────────────────────────────

    fn update_ema(&mut self, spatial_e: f64, temporal_e: f64) {
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.ema_spatial_entropy = spatial_e;
            self.ema_temporal_entropy = temporal_e;
            self.ema_initialized = true;
        } else {
            self.ema_spatial_entropy = alpha * spatial_e + (1.0 - alpha) * self.ema_spatial_entropy;
            self.ema_temporal_entropy =
                alpha * temporal_e + (1.0 - alpha) * self.ema_temporal_entropy;
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// Current EMA of spatial attention entropy.
    pub fn ema_spatial_entropy(&self) -> f64 {
        self.ema_spatial_entropy
    }

    /// Current EMA of temporal attention entropy.
    pub fn ema_temporal_entropy(&self) -> f64 {
        self.ema_temporal_entropy
    }

    /// Whether the EMA has received enough samples to be meaningful.
    pub fn is_warmed_up(&self) -> bool {
        self.stats.total_forwards as usize >= self.config.min_samples
    }

    /// Current tick counter.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Reference to running statistics.
    pub fn stats(&self) -> &FactorizedAttentionStats {
        &self.stats
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    /// Head dimension (embedding_dim / num_heads).
    pub fn head_dim(&self) -> usize {
        self.config.embedding_dim / self.config.num_heads
    }

    /// Whether causal masking is enabled for the temporal stage.
    pub fn causal_temporal(&self) -> bool {
        self.config.causal_temporal
    }

    /// Scale factor used in attention score computation.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    // ── Confidence & windowed diagnostics ───────────────────────────────

    /// Confidence score in [0, 1] based on the number of samples relative to
    /// `min_samples`.
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_forwards as f64;
        let min = self.config.min_samples as f64;
        (n / min).min(1.0)
    }

    /// Windowed mean spatial entropy over the rolling window.
    pub fn windowed_mean_spatial_entropy(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.spatial_entropy).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean temporal entropy over the rolling window.
    pub fn windowed_mean_temporal_entropy(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.temporal_entropy).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean total-tokens-per-forward.
    pub fn windowed_mean_tokens(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.total_tokens as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Returns `true` if spatial entropy has been trending upward over the
    /// most recent half of the window.  Requires at least 4 records.
    pub fn is_spatial_entropy_increasing(&self) -> bool {
        self.is_trend_increasing(|r| r.spatial_entropy)
    }

    /// Returns `true` if temporal entropy has been trending upward.
    pub fn is_temporal_entropy_increasing(&self) -> bool {
        self.is_trend_increasing(|r| r.temporal_entropy)
    }

    /// Returns `true` if spatial entropy has been trending downward.
    pub fn is_spatial_entropy_decreasing(&self) -> bool {
        self.is_trend_decreasing(|r| r.spatial_entropy)
    }

    /// Returns `true` if temporal entropy has been trending downward.
    pub fn is_temporal_entropy_decreasing(&self) -> bool {
        self.is_trend_decreasing(|r| r.temporal_entropy)
    }

    fn is_trend_increasing(&self, f: fn(&WindowRecord) -> f64) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 = self.recent.iter().take(mid).map(&f).sum::<f64>() / mid as f64;
        let second_half: f64 =
            self.recent.iter().skip(mid).map(f).sum::<f64>() / (self.recent.len() - mid) as f64;
        second_half > first_half
    }

    fn is_trend_decreasing(&self, f: fn(&WindowRecord) -> f64) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 = self.recent.iter().take(mid).map(&f).sum::<f64>() / mid as f64;
        let second_half: f64 =
            self.recent.iter().skip(mid).map(f).sum::<f64>() / (self.recent.len() - mid) as f64;
        second_half < first_half
    }

    // ── Reset ───────────────────────────────────────────────────────────

    /// Reset EMA and window state but keep weights.
    pub fn reset(&mut self) {
        self.ema_spatial_entropy = 0.0;
        self.ema_temporal_entropy = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = FactorizedAttentionStats::default();
    }

    /// Re-initialise projection weights (full reset).
    pub fn reset_all(&mut self) {
        self.spatial_projs = AttentionProjections::new(self.config.embedding_dim);
        self.temporal_projs = AttentionProjections::new(self.config.embedding_dim);
        self.reset();
    }

    /// Main processing function — no-op pass-through for trait compatibility.
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers -----------------------------------------------------------------

    fn default_fa() -> FactorizedAttention {
        FactorizedAttention::new()
    }

    fn make_input(b: usize, t: usize, s: usize, d: usize) -> Vec<f64> {
        (0..b * t * s * d)
            .map(|i| (i as f64 + 1.0).sin() * 0.5)
            .collect()
    }

    fn small_config() -> FactorizedAttentionConfig {
        FactorizedAttentionConfig {
            embedding_dim: 8,
            num_heads: 2,
            causal_temporal: false,
            scale: None,
            ema_decay: 0.1,
            min_samples: 3,
            window_size: 10,
        }
    }

    fn small_fa() -> FactorizedAttention {
        FactorizedAttention::with_config(small_config()).unwrap()
    }

    // Construction ------------------------------------------------------------

    #[test]
    fn test_basic() {
        let fa = default_fa();
        assert!(fa.process().is_ok());
    }

    #[test]
    fn test_default_config() {
        let fa = default_fa();
        assert_eq!(fa.embedding_dim(), 64);
        assert_eq!(fa.num_heads(), 4);
        assert_eq!(fa.head_dim(), 16);
        assert!(!fa.causal_temporal());
    }

    #[test]
    fn test_with_config() {
        let cfg = small_config();
        let fa = FactorizedAttention::with_config(cfg).unwrap();
        assert_eq!(fa.embedding_dim(), 8);
        assert_eq!(fa.num_heads(), 2);
        assert_eq!(fa.head_dim(), 4);
    }

    #[test]
    fn test_invalid_config_embedding_dim_zero() {
        let mut cfg = small_config();
        cfg.embedding_dim = 0;
        assert!(FactorizedAttention::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_num_heads_zero() {
        let mut cfg = small_config();
        cfg.num_heads = 0;
        assert!(FactorizedAttention::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_not_divisible() {
        let mut cfg = small_config();
        cfg.embedding_dim = 7;
        cfg.num_heads = 2;
        assert!(FactorizedAttention::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = small_config();
        cfg.ema_decay = 0.0;
        assert!(FactorizedAttention::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = small_config();
        cfg.ema_decay = 1.0;
        assert!(FactorizedAttention::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_window_size_zero() {
        let mut cfg = small_config();
        cfg.window_size = 0;
        assert!(FactorizedAttention::with_config(cfg).is_err());
    }

    // Forward pass ------------------------------------------------------------

    #[test]
    fn test_forward_basic() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        let result = fa.forward(&input, 1, 2, 3).unwrap();
        assert_eq!(result.batch_size, 1);
        assert_eq!(result.num_temporal, 2);
        assert_eq!(result.num_spatial, 3);
        assert_eq!(result.embedding_dim, 8);
        assert_eq!(result.total_tokens, 6);
        assert_eq!(result.output.len(), input.len());
    }

    #[test]
    fn test_forward_wrong_length() {
        let mut fa = small_fa();
        let input = vec![0.0; 10]; // wrong length
        assert!(fa.forward(&input, 1, 2, 3).is_err());
    }

    #[test]
    fn test_forward_zero_batch() {
        let mut fa = small_fa();
        assert!(fa.forward(&[], 0, 2, 3).is_err());
    }

    #[test]
    fn test_forward_zero_temporal() {
        let mut fa = small_fa();
        assert!(fa.forward(&[], 1, 0, 3).is_err());
    }

    #[test]
    fn test_forward_zero_spatial() {
        let mut fa = small_fa();
        assert!(fa.forward(&[], 1, 2, 0).is_err());
    }

    #[test]
    fn test_forward_output_finite() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        let result = fa.forward(&input, 1, 2, 3).unwrap();
        for v in &result.output {
            assert!(v.is_finite(), "output contains non-finite value: {}", v);
        }
    }

    #[test]
    fn test_forward_output_differs_from_input() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        let result = fa.forward(&input, 1, 2, 3).unwrap();
        // The output should differ due to attention + projections
        let same = result
            .output
            .iter()
            .zip(input.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12);
        assert!(!same, "output should differ from input after attention");
    }

    #[test]
    fn test_forward_batch_two() {
        let mut fa = small_fa();
        let input = make_input(2, 2, 3, 8);
        let result = fa.forward(&input, 2, 2, 3).unwrap();
        assert_eq!(result.batch_size, 2);
        assert_eq!(result.total_tokens, 12);
        assert_eq!(result.output.len(), 2 * 2 * 3 * 8);
    }

    #[test]
    fn test_forward_single_token() {
        let mut fa = small_fa();
        let input = make_input(1, 1, 1, 8);
        let result = fa.forward(&input, 1, 1, 1).unwrap();
        assert_eq!(result.total_tokens, 1);
        for v in &result.output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_forward_result_get() {
        let mut fa = small_fa();
        let input = make_input(2, 2, 3, 8);
        let result = fa.forward(&input, 2, 2, 3).unwrap();
        let embedding = result.get(0, 0, 0).unwrap();
        assert_eq!(embedding.len(), 8);
        assert!(result.get(2, 0, 0).is_none()); // out of bounds
        assert!(result.get(0, 2, 0).is_none());
        assert!(result.get(0, 0, 3).is_none());
    }

    // Residual connection -----------------------------------------------------

    #[test]
    fn test_residual_connection_present() {
        // With residual connections, a zero input should produce non-zero output
        // only if projections add something. But more importantly, output should
        // be correlated with input if we pass a known pattern.
        let mut fa = small_fa();
        let input_a = make_input(1, 2, 3, 8);
        let mut input_b = input_a.clone();
        // Perturb one value
        input_b[0] += 10.0;
        let res_a = fa.forward(&input_a, 1, 2, 3).unwrap();
        fa.reset_all(); // reset weights to same state
        let mut fa2 = small_fa();
        let res_b = fa2.forward(&input_b, 1, 2, 3).unwrap();
        // Outputs should differ due to the perturbed input
        let max_diff: f64 = res_a
            .output
            .iter()
            .zip(res_b.output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff > 0.0,
            "perturbation should propagate through attention"
        );
    }

    // Causal masking ----------------------------------------------------------

    #[test]
    fn test_causal_temporal() {
        let mut cfg = small_config();
        cfg.causal_temporal = true;
        let mut fa = FactorizedAttention::with_config(cfg).unwrap();
        assert!(fa.causal_temporal());
        let input = make_input(1, 4, 2, 8);
        let result = fa.forward(&input, 1, 4, 2).unwrap();
        assert_eq!(result.total_tokens, 8);
        for v in &result.output {
            assert!(v.is_finite());
        }
    }

    // Entropy -----------------------------------------------------------------

    #[test]
    fn test_entropy_non_negative() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        let result = fa.forward(&input, 1, 2, 3).unwrap();
        assert!(result.spatial_entropy >= 0.0);
        assert!(result.temporal_entropy >= 0.0);
    }

    #[test]
    fn test_entropy_uniform_distribution() {
        // A uniform attention distribution over N tokens should have entropy ≈ ln(N)
        let mut w = Mat::zeros(2, 4);
        for r in 0..2 {
            for c in 0..4 {
                w.set(r, c, 0.25);
            }
        }
        let e = attention_entropy(&w);
        let expected = (4.0_f64).ln();
        assert!(
            (e - expected).abs() < 1e-10,
            "expected entropy ≈ {}, got {}",
            expected,
            e
        );
    }

    #[test]
    fn test_entropy_peaked_distribution() {
        // A peaked distribution [1, 0, 0, 0] should have entropy ≈ 0
        let mut w = Mat::zeros(1, 4);
        w.set(0, 0, 1.0);
        let e = attention_entropy(&w);
        assert!(e.abs() < 1e-10, "peaked entropy should be ≈ 0, got {}", e);
    }

    // EMA ---------------------------------------------------------------------

    #[test]
    fn test_ema_initializes_on_first_forward() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        let result = fa.forward(&input, 1, 2, 3).unwrap();
        assert!(
            (fa.ema_spatial_entropy() - result.spatial_entropy).abs() < 1e-10,
            "first forward should initialize EMA to observed value"
        );
        assert!((fa.ema_temporal_entropy() - result.temporal_entropy).abs() < 1e-10);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        let ema1 = fa.ema_spatial_entropy();
        // Second forward with same input
        fa.forward(&input, 1, 2, 3).unwrap();
        let ema2 = fa.ema_spatial_entropy();
        // EMA should be smoothed (may or may not change much with same input)
        assert!(ema2.is_finite());
        // With identical input, entropy should be the same each time,
        // but EMA should converge. After two identical samples, EMA should
        // equal the sample value.
        let diff = (ema2 - ema1).abs();
        // Difference should be small since same input yields same entropy
        assert!(diff < 1.0);
    }

    // Warmed-up & confidence --------------------------------------------------

    #[test]
    fn test_not_warmed_up_initially() {
        let fa = small_fa();
        assert!(!fa.is_warmed_up());
        assert_eq!(fa.confidence(), 0.0);
    }

    #[test]
    fn test_warmed_up_after_min_samples() {
        let mut fa = small_fa(); // min_samples = 3
        let input = make_input(1, 2, 3, 8);
        for _ in 0..3 {
            fa.forward(&input, 1, 2, 3).unwrap();
        }
        assert!(fa.is_warmed_up());
        assert!((fa.confidence() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_increases() {
        let mut fa = small_fa(); // min_samples = 3
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        let c1 = fa.confidence();
        fa.forward(&input, 1, 2, 3).unwrap();
        let c2 = fa.confidence();
        assert!(c2 > c1);
    }

    // Windowed diagnostics ----------------------------------------------------

    #[test]
    fn test_windowed_mean_empty() {
        let fa = small_fa();
        assert_eq!(fa.windowed_mean_spatial_entropy(), 0.0);
        assert_eq!(fa.windowed_mean_temporal_entropy(), 0.0);
        assert_eq!(fa.windowed_mean_tokens(), 0.0);
    }

    #[test]
    fn test_windowed_mean_after_forwards() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        assert!(
            fa.windowed_mean_spatial_entropy() > 0.0 || fa.windowed_mean_spatial_entropy() == 0.0
        );
        assert_eq!(fa.windowed_mean_tokens(), 6.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut cfg = small_config();
        cfg.window_size = 3;
        let mut fa = FactorizedAttention::with_config(cfg).unwrap();
        let input = make_input(1, 2, 3, 8);
        for _ in 0..5 {
            fa.forward(&input, 1, 2, 3).unwrap();
        }
        assert_eq!(fa.recent.len(), 3);
    }

    #[test]
    fn test_trend_insufficient_data() {
        let mut fa = small_fa();
        assert!(!fa.is_spatial_entropy_increasing());
        assert!(!fa.is_temporal_entropy_increasing());
        assert!(!fa.is_spatial_entropy_decreasing());
        assert!(!fa.is_temporal_entropy_decreasing());
        // Even with 1-3 samples, should return false
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        assert!(!fa.is_spatial_entropy_increasing());
    }

    // Stats -------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let fa = small_fa();
        assert_eq!(fa.stats().total_forwards, 0);
        assert_eq!(fa.stats().total_tokens, 0);
        assert_eq!(fa.stats().peak_spatial_entropy, 0.0);
        assert_eq!(fa.stats().peak_temporal_entropy, 0.0);
        assert_eq!(fa.stats().mean_spatial_entropy(), 0.0);
        assert_eq!(fa.stats().mean_temporal_entropy(), 0.0);
    }

    #[test]
    fn test_stats_after_forward() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        assert_eq!(fa.stats().total_forwards, 1);
        assert_eq!(fa.stats().total_tokens, 6);
        assert_eq!(fa.stats().min_batch_size, 1);
        assert_eq!(fa.stats().max_batch_size, 1);
    }

    #[test]
    fn test_stats_batch_size_tracking() {
        let mut fa = small_fa();
        let input1 = make_input(1, 2, 3, 8);
        fa.forward(&input1, 1, 2, 3).unwrap();
        let input2 = make_input(3, 2, 3, 8);
        fa.forward(&input2, 3, 2, 3).unwrap();
        assert_eq!(fa.stats().min_batch_size, 1);
        assert_eq!(fa.stats().max_batch_size, 3);
    }

    #[test]
    fn test_stats_peak_entropy_tracked() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        let peak_s = fa.stats().peak_spatial_entropy;
        let peak_t = fa.stats().peak_temporal_entropy;
        // Peak should be >= 0
        assert!(peak_s >= 0.0);
        assert!(peak_t >= 0.0);
    }

    #[test]
    fn test_stats_mean_entropy() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        fa.forward(&input, 1, 2, 3).unwrap();
        let mean_s = fa.stats().mean_spatial_entropy();
        let mean_t = fa.stats().mean_temporal_entropy();
        // Mean should be finite and non-negative
        assert!(mean_s.is_finite() && mean_s >= 0.0);
        assert!(mean_t.is_finite() && mean_t >= 0.0);
    }

    // Tick counter -------------------------------------------------------------

    #[test]
    fn test_tick_increments() {
        let mut fa = small_fa();
        assert_eq!(fa.current_tick(), 0);
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        assert_eq!(fa.current_tick(), 1);
        fa.forward(&input, 1, 2, 3).unwrap();
        assert_eq!(fa.current_tick(), 2);
    }

    // Reset -------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        fa.forward(&input, 1, 2, 3).unwrap();
        assert_eq!(fa.stats().total_forwards, 2);
        fa.reset();
        assert_eq!(fa.stats().total_forwards, 0);
        assert_eq!(fa.current_tick(), 0);
        assert_eq!(fa.ema_spatial_entropy(), 0.0);
        assert!(fa.recent.is_empty());
    }

    #[test]
    fn test_reset_all() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        fa.forward(&input, 1, 2, 3).unwrap();
        fa.reset_all();
        assert_eq!(fa.stats().total_forwards, 0);
        assert_eq!(fa.current_tick(), 0);
    }

    // process() compat --------------------------------------------------------

    #[test]
    fn test_process() {
        let fa = small_fa();
        assert!(fa.process().is_ok());
    }

    // Scale -------------------------------------------------------------------

    #[test]
    fn test_default_scale() {
        let fa = small_fa(); // head_dim = 8/2 = 4
        let expected = 1.0 / (4.0_f64).sqrt();
        assert!((fa.scale() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_custom_scale() {
        let mut cfg = small_config();
        cfg.scale = Some(0.42);
        let fa = FactorizedAttention::with_config(cfg).unwrap();
        assert!((fa.scale() - 0.42).abs() < 1e-10);
    }

    // Mat helper tests --------------------------------------------------------

    #[test]
    fn test_mat_zeros() {
        let m = Mat::zeros(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        for v in &m.data {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_mat_matmul_identity() {
        // Multiply by identity-like matrix
        let mut a = Mat::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 3.0);
        a.set(1, 1, 4.0);

        let mut id = Mat::zeros(2, 2);
        id.set(0, 0, 1.0);
        id.set(1, 1, 1.0);

        let result = a.matmul(&id);
        assert!((result.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((result.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((result.get(1, 0) - 3.0).abs() < 1e-10);
        assert!((result.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_transpose() {
        let mut m = Mat::zeros(2, 3);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        m.set(1, 0, 4.0);
        m.set(1, 1, 5.0);
        m.set(1, 2, 6.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((t.get(2, 1) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_softmax_rows() {
        let mut m = Mat::zeros(1, 3);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        m.softmax_rows();
        let sum: f64 = m.row(0).iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax row should sum to 1");
        // Values should be monotonically increasing
        assert!(m.get(0, 0) < m.get(0, 1));
        assert!(m.get(0, 1) < m.get(0, 2));
    }

    #[test]
    fn test_mat_causal_mask() {
        let mut m = Mat::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                m.set(r, c, 1.0);
            }
        }
        m.apply_causal_mask();
        // Upper triangle should be -inf
        assert!(m.get(0, 1).is_infinite() && m.get(0, 1) < 0.0);
        assert!(m.get(0, 2).is_infinite());
        assert!(m.get(1, 2).is_infinite());
        // Diagonal and below should be 1.0
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 0), 1.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.get(2, 0), 1.0);
        assert_eq!(m.get(2, 1), 1.0);
        assert_eq!(m.get(2, 2), 1.0);
    }

    #[test]
    fn test_mat_add_bias() {
        let mut m = Mat::zeros(2, 3);
        m.add_bias(&[1.0, 2.0, 3.0]);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 1), 2.0);
    }

    // Larger dimension test ---------------------------------------------------

    #[test]
    fn test_forward_default_config() {
        let mut fa = default_fa(); // 64-dim, 4 heads
        let input = make_input(1, 4, 8, 64);
        let result = fa.forward(&input, 1, 4, 8).unwrap();
        assert_eq!(result.total_tokens, 32);
        assert_eq!(result.output.len(), 4 * 8 * 64);
        for v in &result.output {
            assert!(v.is_finite());
        }
    }

    // Determinism test --------------------------------------------------------

    #[test]
    fn test_deterministic() {
        let mut fa1 = small_fa();
        let mut fa2 = small_fa();
        let input = make_input(1, 2, 3, 8);
        let r1 = fa1.forward(&input, 1, 2, 3).unwrap();
        let r2 = fa2.forward(&input, 1, 2, 3).unwrap();
        for (a, b) in r1.output.iter().zip(r2.output.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "outputs should be deterministic: {} vs {}",
                a,
                b
            );
        }
    }

    // Xavier init test --------------------------------------------------------

    #[test]
    fn test_xavier_init_scale() {
        let m = Mat::xavier(64, 64);
        let mean: f64 = m.data.iter().sum::<f64>() / m.data.len() as f64;
        // Mean should be roughly around 0
        assert!(
            mean.abs() < 0.5,
            "Xavier init mean should be near 0, got {}",
            mean
        );
        // All values should be finite
        for v in &m.data {
            assert!(v.is_finite());
        }
    }

    // Attention weights sum to 1 test -----------------------------------------

    #[test]
    fn test_attention_weights_sum_to_one() {
        let projs = AttentionProjections::new(8);
        let mat = Mat {
            data: make_input(1, 1, 4, 8)[..4 * 8].to_vec(),
            rows: 4,
            cols: 8,
        };
        let (_, attn) = multi_head_attention(&mat, &projs, 2, 0.5, false);
        for r in 0..attn.rows {
            let sum: f64 = attn.row(r).iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "attention row {} sum = {}, expected 1.0",
                r,
                sum
            );
        }
    }

    #[test]
    fn test_causal_attention_weights() {
        let projs = AttentionProjections::new(8);
        let mat = Mat {
            data: make_input(1, 1, 4, 8)[..4 * 8].to_vec(),
            rows: 4,
            cols: 8,
        };
        let (_, attn) = multi_head_attention(&mat, &projs, 2, 0.5, true);
        // With causal masking, attn[0, 1..] should be 0 (or very close)
        for c in 1..4 {
            assert!(
                attn.get(0, c) < 1e-6,
                "causal mask: attn[0,{}] = {} should be ≈ 0",
                c,
                attn.get(0, c)
            );
        }
        // Row sums should still be 1
        for r in 0..attn.rows {
            let sum: f64 = attn.row(r).iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    // Batch independence test -------------------------------------------------

    #[test]
    fn test_batch_independence() {
        let mut fa = small_fa();
        // Process batch of 1
        let input_single = make_input(1, 2, 3, 8);
        let res_single = fa.forward(&input_single, 1, 2, 3).unwrap();
        fa.reset_all();
        let mut fa2 = small_fa();

        // Process batch of 2 where first item is the same
        let mut input_double = make_input(1, 2, 3, 8);
        // Append a second batch item (different data)
        let second: Vec<f64> = (0..2 * 3 * 8).map(|i| (i as f64 * 0.1).cos()).collect();
        input_double.extend_from_slice(&second);

        let res_double = fa2.forward(&input_double, 2, 2, 3).unwrap();

        // First batch item's output should match the single-batch result
        let single_first = &res_single.output[..2 * 3 * 8];
        let double_first = &res_double.output[..2 * 3 * 8];
        for (a, b) in single_first.iter().zip(double_first.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "batch independence violated: {} vs {}",
                a,
                b
            );
        }
    }

    // Multiple forwards accumulate stats properly -----------------------------

    #[test]
    fn test_multiple_forwards_stats() {
        let mut fa = small_fa();
        let input = make_input(1, 2, 3, 8);
        for _ in 0..5 {
            fa.forward(&input, 1, 2, 3).unwrap();
        }
        assert_eq!(fa.stats().total_forwards, 5);
        assert_eq!(fa.stats().total_tokens, 30); // 5 * 6
        assert_eq!(fa.current_tick(), 5);
    }
}
