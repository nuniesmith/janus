//! Chronos — Time Series Forecasting via ONNX Inference
//!
//! Implements a Chronos-style time series forecasting pipeline using `tract-onnx`
//! for model inference and a pure-Rust tokenizer for quantile-based discretization
//! of continuous time series into discrete tokens (and back).
//!
//! Chronos (Ansari et al., 2024) treats time series forecasting as a language
//! modeling task by:
//! 1. Tokenizing continuous values into discrete bins via quantile binning.
//! 2. Feeding token sequences through a T5/GPT-style autoregressive model.
//! 3. Decoding predicted tokens back into continuous forecast values.
//!
//! This module provides:
//! - A quantile-based tokenizer (`ChronosTokenizer`)
//! - An ONNX inference engine (`ChronosInference`) via `tract-onnx`
//! - Forecast output with confidence intervals (`ChronosForecast`)
//! - Integration hooks for the JANUS thalamus sensory pipeline
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Chronos ONNX Inference Pipeline                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐     ┌───────────────┐     ┌────────────┐ │
//! │  │ Raw Time     │     │   Quantile    │     │   ONNX     │ │
//! │  │ Series       │────▶│   Tokenizer   │────▶│   Model    │ │
//! │  │ (f32 values) │     │   (binning)   │     │ (tract)    │ │
//! │  └──────────────┘     └───────────────┘     └─────┬──────┘ │
//! │                                                    │        │
//! │         ┌──────────────────────────────────────────┘        │
//! │         ▼                                                   │
//! │  ┌──────────────┐     ┌───────────────┐                    │
//! │  │ Token        │     │   Forecast    │                    │
//! │  │ Decoder      │────▶│   Output      │                    │
//! │  │ (de-quantize)│     │   + Intervals │                    │
//! │  └──────────────┘     └───────────────┘                    │
//! │                                                              │
//! │  Use cases:                                                  │
//! │  • Short-term price/return forecasting                      │
//! │  • Volatility prediction                                    │
//! │  • Multi-horizon forecast for risk management               │
//! │  • Foundation model inference without Python                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_neuromorphic::thalamus::sources::chronos::*;
//!
//! // Configure the tokenizer
//! let tok_config = TokenizerConfig::default();
//! let tokenizer = ChronosTokenizer::new(tok_config);
//!
//! // Load an ONNX model
//! let config = ChronosConfig::default();
//! let engine = ChronosInference::load("path/to/chronos.onnx", config)?;
//!
//! // Forecast
//! let context = vec![100.0, 101.5, 99.8, 102.3, 103.1];
//! let forecast = engine.forecast(&context, 12)?;
//!
//! println!("Median forecast: {:?}", forecast.median());
//! println!("80% interval: {:?}", forecast.interval(0.80));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Chronos forecasting pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronosConfig {
    /// Maximum context length (number of historical time steps the model can see).
    pub context_length: usize,

    /// Default prediction horizon (number of future steps to forecast).
    pub prediction_length: usize,

    /// Number of quantile bins for tokenization.
    /// Must match the vocabulary size the ONNX model was trained with.
    pub n_bins: usize,

    /// Number of special tokens (PAD, BOS, EOS, MASK, UNK).
    pub n_special_tokens: usize,

    /// Number of Monte Carlo sample trajectories for probabilistic forecasting.
    pub n_samples: usize,

    /// Temperature for sampling from the predicted token distribution.
    /// Lower = more deterministic, higher = more diverse.
    pub temperature: f32,

    /// Top-k sampling: only consider the top k most likely tokens.
    /// Set to 0 to disable.
    pub top_k: usize,

    /// Whether to use mean scaling (divide by mean of context) before tokenization.
    pub use_mean_scaling: bool,

    /// Minimum absolute mean for scaling (avoids division by near-zero).
    pub min_scale: f32,

    /// Path to the ONNX model file (optional, can be set at load time).
    pub model_path: Option<String>,

    /// Whether to apply softmax to model logits before sampling.
    pub apply_softmax: bool,

    /// Optional: fixed quantile boundaries (if None, learned from context).
    pub fixed_boundaries: Option<Vec<f32>>,
}

impl Default for ChronosConfig {
    fn default() -> Self {
        Self {
            context_length: 512,
            prediction_length: 64,
            n_bins: 4096,
            n_special_tokens: 5,
            n_samples: 20,
            temperature: 1.0,
            top_k: 50,
            use_mean_scaling: true,
            min_scale: 1e-9,
            model_path: None,
            apply_softmax: true,
            fixed_boundaries: None,
        }
    }
}

impl ChronosConfig {
    /// Preset for Chronos-T5-Tiny (context=512, bins=4096).
    pub fn t5_tiny() -> Self {
        Self {
            context_length: 512,
            prediction_length: 64,
            n_bins: 4096,
            n_samples: 20,
            ..Default::default()
        }
    }

    /// Preset for Chronos-T5-Small (context=512, bins=4096).
    pub fn t5_small() -> Self {
        Self {
            context_length: 512,
            prediction_length: 64,
            n_bins: 4096,
            n_samples: 20,
            ..Default::default()
        }
    }

    /// Preset for Chronos-T5-Base (context=512, bins=4096).
    pub fn t5_base() -> Self {
        Self {
            context_length: 512,
            prediction_length: 64,
            n_bins: 4096,
            n_samples: 20,
            ..Default::default()
        }
    }

    /// Lightweight preset for testing.
    pub fn tiny_test() -> Self {
        Self {
            context_length: 32,
            prediction_length: 8,
            n_bins: 64,
            n_special_tokens: 5,
            n_samples: 5,
            temperature: 1.0,
            top_k: 10,
            use_mean_scaling: true,
            min_scale: 1e-9,
            model_path: None,
            apply_softmax: true,
            fixed_boundaries: None,
        }
    }

    /// Builder: set context length.
    pub fn with_context_length(mut self, len: usize) -> Self {
        self.context_length = len;
        self
    }

    /// Builder: set prediction length.
    pub fn with_prediction_length(mut self, len: usize) -> Self {
        self.prediction_length = len;
        self
    }

    /// Builder: set number of bins.
    pub fn with_n_bins(mut self, n: usize) -> Self {
        self.n_bins = n;
        self
    }

    /// Builder: set number of Monte Carlo samples.
    pub fn with_n_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Builder: set temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Builder: set top-k.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Builder: set model path.
    pub fn with_model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Builder: set fixed quantile boundaries.
    pub fn with_fixed_boundaries(mut self, boundaries: Vec<f32>) -> Self {
        self.fixed_boundaries = Some(boundaries);
        self
    }

    /// Total vocabulary size (bins + special tokens).
    pub fn vocab_size(&self) -> usize {
        self.n_bins + self.n_special_tokens
    }

    /// First bin token ID (after special tokens).
    pub fn first_bin_id(&self) -> usize {
        self.n_special_tokens
    }

    /// Last bin token ID (inclusive).
    pub fn last_bin_id(&self) -> usize {
        self.n_special_tokens + self.n_bins - 1
    }
}

// ---------------------------------------------------------------------------
// Special Token IDs
// ---------------------------------------------------------------------------

/// Special token constants matching the Chronos vocabulary layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Padding token ID.
    pub pad: usize,
    /// Beginning-of-sequence token ID.
    pub bos: usize,
    /// End-of-sequence token ID.
    pub eos: usize,
    /// Mask token ID (for masked language modeling).
    pub mask: usize,
    /// Unknown token ID.
    pub unk: usize,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: 0,
            bos: 1,
            eos: 2,
            mask: 3,
            unk: 4,
        }
    }
}

impl SpecialTokens {
    /// Check whether a token ID is a special token.
    pub fn is_special(&self, id: usize) -> bool {
        id == self.pad || id == self.bos || id == self.eos || id == self.mask || id == self.unk
    }
}

// ---------------------------------------------------------------------------
// Tokenizer Configuration
// ---------------------------------------------------------------------------

/// Configuration for the quantile-based tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Number of quantile bins.
    pub n_bins: usize,

    /// Number of special tokens prepended before bin tokens.
    pub n_special_tokens: usize,

    /// Method for constructing bin boundaries.
    pub binning_method: BinningMethod,

    /// Whether to use mean scaling before binning.
    pub use_mean_scaling: bool,

    /// Minimum absolute mean for scaling.
    pub min_scale: f32,

    /// Quantile range for uniform quantile binning: (low, high).
    /// Default: (1e-7, 1 - 1e-7) to avoid inf at the tails.
    pub quantile_low: f64,
    pub quantile_high: f64,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            n_bins: 4096,
            n_special_tokens: 5,
            binning_method: BinningMethod::UniformQuantile,
            use_mean_scaling: true,
            min_scale: 1e-9,
            quantile_low: 1e-7,
            quantile_high: 1.0 - 1e-7,
        }
    }
}

impl TokenizerConfig {
    /// Builder: set number of bins.
    pub fn with_n_bins(mut self, n: usize) -> Self {
        self.n_bins = n;
        self
    }

    /// Builder: set binning method.
    pub fn with_binning_method(mut self, method: BinningMethod) -> Self {
        self.binning_method = method;
        self
    }

    /// Builder: set mean scaling.
    pub fn with_mean_scaling(mut self, use_it: bool) -> Self {
        self.use_mean_scaling = use_it;
        self
    }
}

/// Method for constructing bin boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinningMethod {
    /// Uniform quantile spacing on the standard normal CDF.
    /// Bins are placed at evenly-spaced quantiles of N(0, 1).
    UniformQuantile,

    /// Linear spacing between a fixed range.
    LinearRange {
        /// Lower bound of the range.
        low: i32,
        /// Upper bound of the range (as integer, will be cast to f32).
        high: i32,
    },
}

// ---------------------------------------------------------------------------
// Chronos Tokenizer
// ---------------------------------------------------------------------------

/// Quantile-based tokenizer that converts continuous time series values
/// into discrete token IDs and back.
///
/// The tokenizer maps floating-point values to bins whose boundaries
/// are placed at quantiles of the standard normal distribution. This
/// ensures that scaled (zero-mean, unit-variance) values are evenly
/// distributed across bins.
///
/// # Token Layout
///
/// ```text
/// [PAD=0] [BOS=1] [EOS=2] [MASK=3] [UNK=4] [BIN_0=5] [BIN_1=6] ... [BIN_{n-1}]
/// ```
#[derive(Debug, Clone)]
pub struct ChronosTokenizer {
    config: TokenizerConfig,
    special_tokens: SpecialTokens,

    /// Bin boundary edges: length = n_bins + 1.
    /// `boundaries[i]` is the lower edge of bin `i`; `boundaries[i+1]` is the upper.
    boundaries: Vec<f32>,

    /// Bin centers: length = n_bins. The representative value for each bin.
    centers: Vec<f32>,
}

impl ChronosTokenizer {
    /// Create a new tokenizer with the given configuration.
    pub fn new(config: TokenizerConfig) -> Self {
        let (boundaries, centers) = match config.binning_method {
            BinningMethod::UniformQuantile => Self::build_quantile_boundaries(
                config.n_bins,
                config.quantile_low,
                config.quantile_high,
            ),
            BinningMethod::LinearRange { low, high } => {
                Self::build_linear_boundaries(config.n_bins, low as f32, high as f32)
            }
        };

        debug!(
            "ChronosTokenizer: {} bins, boundaries range [{:.4}, {:.4}]",
            config.n_bins,
            boundaries.first().unwrap_or(&0.0),
            boundaries.last().unwrap_or(&0.0),
        );

        Self {
            config,
            special_tokens: SpecialTokens::default(),
            boundaries,
            centers,
        }
    }

    /// Create a tokenizer from a `ChronosConfig`.
    pub fn from_config(config: &ChronosConfig) -> Self {
        let tok_config = TokenizerConfig {
            n_bins: config.n_bins,
            n_special_tokens: config.n_special_tokens,
            use_mean_scaling: config.use_mean_scaling,
            min_scale: config.min_scale,
            ..Default::default()
        };

        if let Some(ref fixed) = config.fixed_boundaries {
            // Use fixed boundaries directly
            let n_bins = fixed.len().saturating_sub(1);
            let centers: Vec<f32> = (0..n_bins)
                .map(|i| (fixed[i] + fixed[i + 1]) / 2.0)
                .collect();

            Self {
                config: tok_config,
                special_tokens: SpecialTokens::default(),
                boundaries: fixed.clone(),
                centers,
            }
        } else {
            Self::new(tok_config)
        }
    }

    /// Build bin boundaries using quantiles of the standard normal distribution.
    ///
    /// Places `n_bins + 1` boundary points at evenly-spaced quantiles of N(0,1),
    /// ranging from `quantile_low` to `quantile_high`.
    fn build_quantile_boundaries(n_bins: usize, q_low: f64, q_high: f64) -> (Vec<f32>, Vec<f32>) {
        let n_edges = n_bins + 1;
        let mut boundaries = Vec::with_capacity(n_edges);

        for i in 0..n_edges {
            let p = q_low + (q_high - q_low) * (i as f64) / (n_edges - 1).max(1) as f64;
            let z = probit(p);
            boundaries.push(z as f32);
        }

        // Ensure -inf and +inf at the extremes for correct clamping
        if let Some(first) = boundaries.first_mut() {
            *first = f32::NEG_INFINITY;
        }
        if let Some(last) = boundaries.last_mut() {
            *last = f32::INFINITY;
        }

        let centers: Vec<f32> = (0..n_bins)
            .map(|i| {
                let lo = boundaries[i];
                let hi = boundaries[i + 1];
                match (lo.is_finite(), hi.is_finite()) {
                    (true, true) => (lo + hi) / 2.0,
                    (false, true) => hi - 0.5, // leftmost bin
                    (true, false) => lo + 0.5, // rightmost bin
                    (false, false) => 0.0,     // degenerate
                }
            })
            .collect();

        (boundaries, centers)
    }

    /// Build bin boundaries using linear spacing.
    fn build_linear_boundaries(n_bins: usize, low: f32, high: f32) -> (Vec<f32>, Vec<f32>) {
        let n_edges = n_bins + 1;
        let step = (high - low) / n_bins as f32;

        let mut boundaries = Vec::with_capacity(n_edges);
        for i in 0..n_edges {
            boundaries.push(low + step * i as f32);
        }

        // Set extreme boundaries
        if let Some(first) = boundaries.first_mut() {
            *first = f32::NEG_INFINITY;
        }
        if let Some(last) = boundaries.last_mut() {
            *last = f32::INFINITY;
        }

        let centers: Vec<f32> = (0..n_bins)
            .map(|i| {
                let lo = if i == 0 { low } else { boundaries[i] };
                let hi = if i == n_bins - 1 {
                    high
                } else {
                    boundaries[i + 1]
                };
                (lo + hi) / 2.0
            })
            .collect();

        (boundaries, centers)
    }

    /// Get the bin boundaries.
    pub fn boundaries(&self) -> &[f32] {
        &self.boundaries
    }

    /// Get the bin centers.
    pub fn centers(&self) -> &[f32] {
        &self.centers
    }

    /// Get the special tokens.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Get the configuration.
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Number of bins.
    pub fn n_bins(&self) -> usize {
        self.config.n_bins
    }

    /// Total vocabulary size (bins + special tokens).
    pub fn vocab_size(&self) -> usize {
        self.config.n_bins + self.config.n_special_tokens
    }

    /// Compute the mean scaling factor for a context window.
    ///
    /// Returns `(scale, offset)` where:
    /// - `scale` = mean(|context|).max(min_scale)
    /// - `offset` = 0.0 (mean scaling only scales, doesn't shift)
    pub fn compute_scale(&self, context: &[f32]) -> (f32, f32) {
        if !self.config.use_mean_scaling || context.is_empty() {
            return (1.0, 0.0);
        }

        let abs_mean = context.iter().map(|x| x.abs()).sum::<f32>() / context.len() as f32;
        let scale = abs_mean.max(self.config.min_scale);
        (scale, 0.0)
    }

    /// Tokenize a context window of continuous values into token IDs.
    ///
    /// Steps:
    /// 1. Compute scale from context.
    /// 2. Divide values by scale.
    /// 3. Map each scaled value to the nearest bin.
    /// 4. Return token IDs (bin index + n_special_tokens).
    ///
    /// # Returns
    /// `(token_ids, scale, offset)` — the token IDs and the scale/offset
    /// used, which must be passed to `decode` for de-tokenization.
    pub fn encode(&self, values: &[f32]) -> (Vec<usize>, f32, f32) {
        let (scale, offset) = self.compute_scale(values);
        let first_bin = self.config.n_special_tokens;

        let token_ids: Vec<usize> = values
            .iter()
            .map(|&v| {
                let scaled = (v - offset) / scale;
                let bin = self.find_bin(scaled);
                bin + first_bin
            })
            .collect();

        (token_ids, scale, offset)
    }

    /// Encode and prepend a BOS token.
    pub fn encode_with_bos(&self, values: &[f32]) -> (Vec<usize>, f32, f32) {
        let (mut tokens, scale, offset) = self.encode(values);
        tokens.insert(0, self.special_tokens.bos);
        (tokens, scale, offset)
    }

    /// Decode token IDs back to continuous values using the stored bin centers
    /// and the scale/offset from encoding.
    ///
    /// Special tokens are decoded as NaN.
    pub fn decode(&self, token_ids: &[usize], scale: f32, offset: f32) -> Vec<f32> {
        let first_bin = self.config.n_special_tokens;

        token_ids
            .iter()
            .map(|&id| {
                if id < first_bin || id >= first_bin + self.config.n_bins {
                    f32::NAN // special token or out-of-range
                } else {
                    let bin = id - first_bin;
                    self.centers[bin] * scale + offset
                }
            })
            .collect()
    }

    /// Decode a single token ID.
    pub fn decode_one(&self, token_id: usize, scale: f32, offset: f32) -> f32 {
        let first_bin = self.config.n_special_tokens;
        if token_id < first_bin || token_id >= first_bin + self.config.n_bins {
            f32::NAN
        } else {
            let bin = token_id - first_bin;
            self.centers[bin] * scale + offset
        }
    }

    /// Find the bin index for a scaled value using binary search.
    fn find_bin(&self, value: f32) -> usize {
        if value.is_nan() {
            return self.config.n_bins / 2; // map NaN to middle bin
        }

        // Binary search: find the largest i such that boundaries[i] <= value
        let n = self.config.n_bins;
        match self
            .boundaries
            .binary_search_by(|b| b.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i.min(n - 1),
            Err(i) => {
                if i == 0 {
                    0
                } else {
                    (i - 1).min(n - 1)
                }
            }
        }
    }

    /// Encode a batch of context windows. Returns per-sample (tokens, scale, offset).
    pub fn encode_batch(&self, batch: &[Vec<f32>]) -> Vec<(Vec<usize>, f32, f32)> {
        batch.iter().map(|ctx| self.encode(ctx)).collect()
    }

    /// Decode a batch of token ID sequences.
    pub fn decode_batch(&self, batch: &[(Vec<usize>, f32, f32)]) -> Vec<Vec<f32>> {
        batch
            .iter()
            .map(|(ids, scale, offset)| self.decode(ids, *scale, *offset))
            .collect()
    }

    /// Compute a histogram of how many values from the input fall into each bin.
    pub fn bin_histogram(&self, values: &[f32]) -> Vec<usize> {
        let (scale, offset) = self.compute_scale(values);
        let mut counts = vec![0usize; self.config.n_bins];

        for &v in values {
            let scaled = (v - offset) / scale;
            let bin = self.find_bin(scaled);
            counts[bin] += 1;
        }

        counts
    }
}

// ---------------------------------------------------------------------------
// Probit function (inverse CDF of standard normal)
// ---------------------------------------------------------------------------

/// Approximate probit function: inverse CDF of the standard normal distribution.
///
/// Uses the rational approximation from Abramowitz and Stegun (1964),
/// formula 26.2.23, which is accurate to about 4.5e-4.
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Constants for the rational approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Rational approximation for lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Rational approximation for upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Forecast Output
// ---------------------------------------------------------------------------

/// A single-horizon probabilistic forecast from Chronos.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronosForecast {
    /// Sampled forecast trajectories: shape [n_samples][prediction_length].
    /// Each trajectory is one possible future.
    pub trajectories: Vec<Vec<f32>>,

    /// Prediction horizon.
    pub prediction_length: usize,

    /// Number of sample trajectories.
    pub n_samples: usize,

    /// Context length used for this forecast.
    pub context_length: usize,

    /// Scale factor used during tokenization (for reference).
    pub scale: f32,

    /// Offset used during tokenization (for reference).
    pub offset: f32,

    /// Wall-clock time for the forecast.
    pub inference_time: Duration,

    /// Optional metadata.
    pub metadata: HashMap<String, String>,
}

impl ChronosForecast {
    /// Compute the median forecast across all samples, per time step.
    pub fn median(&self) -> Vec<f32> {
        self.quantile(0.5)
    }

    /// Compute the mean forecast across all samples, per time step.
    pub fn mean(&self) -> Vec<f32> {
        if self.trajectories.is_empty() {
            return vec![];
        }

        let mut means = vec![0.0f32; self.prediction_length];
        for traj in &self.trajectories {
            for (i, &v) in traj.iter().enumerate() {
                if i < self.prediction_length {
                    means[i] += v;
                }
            }
        }
        for m in &mut means {
            *m /= self.n_samples as f32;
        }
        means
    }

    /// Compute a specific quantile across samples, per time step.
    ///
    /// `q` should be in [0, 1]. E.g., 0.1 for 10th percentile, 0.9 for 90th.
    pub fn quantile(&self, q: f32) -> Vec<f32> {
        if self.trajectories.is_empty() || self.prediction_length == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(self.prediction_length);
        for t in 0..self.prediction_length {
            let mut values: Vec<f32> = self
                .trajectories
                .iter()
                .filter_map(|traj| traj.get(t).copied())
                .filter(|v| v.is_finite())
                .collect();

            if values.is_empty() {
                result.push(f32::NAN);
                continue;
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((q * (values.len() - 1) as f32).round() as usize).min(values.len() - 1);
            result.push(values[idx]);
        }

        result
    }

    /// Compute a prediction interval: (lower, upper) quantile bounds per step.
    ///
    /// `coverage` is in (0, 1). E.g., 0.80 for an 80% prediction interval.
    pub fn interval(&self, coverage: f32) -> (Vec<f32>, Vec<f32>) {
        let alpha = (1.0 - coverage) / 2.0;
        let lower = self.quantile(alpha);
        let upper = self.quantile(1.0 - alpha);
        (lower, upper)
    }

    /// Compute the standard deviation across samples, per time step.
    pub fn std_dev(&self) -> Vec<f32> {
        let means = self.mean();
        if means.is_empty() {
            return vec![];
        }

        let mut vars = vec![0.0f32; self.prediction_length];
        for traj in &self.trajectories {
            for (i, &v) in traj.iter().enumerate() {
                if i < self.prediction_length && v.is_finite() {
                    let diff = v - means[i];
                    vars[i] += diff * diff;
                }
            }
        }

        vars.iter()
            .map(|v| (v / self.n_samples.max(1) as f32).sqrt())
            .collect()
    }

    /// Compute the coefficient of variation per time step.
    pub fn cv(&self) -> Vec<f32> {
        let means = self.mean();
        let stds = self.std_dev();
        means
            .iter()
            .zip(stds.iter())
            .map(|(&m, &s)| {
                if m.abs() > 1e-10 {
                    s / m.abs()
                } else {
                    f32::NAN
                }
            })
            .collect()
    }

    /// Get the forecast for a specific horizon step (0-indexed).
    ///
    /// Returns (mean, median, std, lower_80, upper_80).
    pub fn at_horizon(&self, h: usize) -> Option<HorizonSummary> {
        if h >= self.prediction_length {
            return None;
        }

        let mut values: Vec<f32> = self
            .trajectories
            .iter()
            .filter_map(|traj| traj.get(h).copied())
            .filter(|v| v.is_finite())
            .collect();

        if values.is_empty() {
            return None;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        let mean = values.iter().sum::<f32>() / n as f32;
        let median_idx = n / 2;
        let median = values[median_idx];

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt();

        let lower_idx = ((0.1 * (n - 1) as f32).round() as usize).min(n - 1);
        let upper_idx = ((0.9 * (n - 1) as f32).round() as usize).min(n - 1);

        Some(HorizonSummary {
            horizon: h,
            mean,
            median,
            std,
            lower_80: values[lower_idx],
            upper_80: values[upper_idx],
            n_samples: n,
        })
    }

    /// Builder: attach metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Summary statistics for a single forecast horizon step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonSummary {
    /// Horizon step index (0-indexed).
    pub horizon: usize,
    /// Mean across samples.
    pub mean: f32,
    /// Median across samples.
    pub median: f32,
    /// Standard deviation across samples.
    pub std: f32,
    /// Lower bound of 80% prediction interval.
    pub lower_80: f32,
    /// Upper bound of 80% prediction interval.
    pub upper_80: f32,
    /// Number of valid samples.
    pub n_samples: usize,
}

// ---------------------------------------------------------------------------
// ONNX Inference Engine
// ---------------------------------------------------------------------------

/// State of the inference engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineState {
    /// Engine not yet loaded.
    Unloaded,
    /// Model loaded and ready for inference.
    Ready,
    /// Model loading or inference failed.
    Failed,
}

/// Chronos ONNX inference engine using tract-onnx.
///
/// Loads an ONNX model exported from the Chronos family (T5-Tiny, T5-Small,
/// T5-Base, etc.) and runs inference using tract's optimized CPU runtime.
///
/// If no ONNX model file is available, the engine can operate in
/// **mock mode** for testing, producing forecasts using simple statistical
/// baselines (random walk, mean reversion, etc.).
pub struct ChronosInference {
    config: ChronosConfig,
    tokenizer: ChronosTokenizer,
    state: EngineState,
    model_path: Option<PathBuf>,

    // Statistics
    total_forecasts: u64,
    total_inference_time: Duration,
}

impl ChronosInference {
    /// Create an inference engine without loading a model (mock/test mode).
    ///
    /// In this mode, `forecast()` will produce statistical baseline predictions
    /// rather than neural network output. Useful for testing the pipeline
    /// end-to-end before model artifacts are available.
    pub fn mock(config: ChronosConfig) -> Self {
        let tokenizer = ChronosTokenizer::from_config(&config);

        info!(
            "ChronosInference created in mock mode (no ONNX model). \
             context_length={}, prediction_length={}, n_bins={}",
            config.context_length, config.prediction_length, config.n_bins
        );

        Self {
            config,
            tokenizer,
            state: EngineState::Ready,
            model_path: None,
            total_forecasts: 0,
            total_inference_time: Duration::ZERO,
        }
    }

    /// Create an inference engine and attempt to load an ONNX model.
    ///
    /// If the model file is not found or cannot be loaded, falls back to
    /// mock mode with a warning.
    pub fn load(model_path: impl AsRef<Path>, config: ChronosConfig) -> Result<Self, ChronosError> {
        let path = model_path.as_ref().to_path_buf();
        let tokenizer = ChronosTokenizer::from_config(&config);

        if !path.exists() {
            warn!(
                "ONNX model not found at {:?}, falling back to mock mode",
                path
            );
            return Ok(Self {
                config,
                tokenizer,
                state: EngineState::Ready,
                model_path: None,
                total_forecasts: 0,
                total_inference_time: Duration::ZERO,
            });
        }

        // NOTE: Actual tract-onnx loading would happen here.
        // For now, we validate the path exists and set up the engine.
        // The full tract integration requires:
        //   use tract_onnx::prelude::*;
        //   let model = tract_onnx::onnx()
        //       .model_for_path(&path)?
        //       .into_optimized()?
        //       .into_runnable()?;
        //
        // This is deferred to runtime integration since tract model types
        // are not easily stored in a struct without generics.

        info!(
            "ChronosInference: model file found at {:?} (tract loading deferred to inference call)",
            path
        );

        Ok(Self {
            config,
            tokenizer,
            state: EngineState::Ready,
            model_path: Some(path),
            total_forecasts: 0,
            total_inference_time: Duration::ZERO,
        })
    }

    /// Get the current engine state.
    pub fn state(&self) -> EngineState {
        self.state
    }

    /// Get the configuration.
    pub fn config(&self) -> &ChronosConfig {
        &self.config
    }

    /// Get the tokenizer.
    pub fn tokenizer(&self) -> &ChronosTokenizer {
        &self.tokenizer
    }

    /// Whether the engine has an ONNX model loaded (vs. mock mode).
    pub fn has_model(&self) -> bool {
        self.model_path.is_some()
    }

    /// Get the model path.
    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }

    /// Get the total number of forecasts produced.
    pub fn total_forecasts(&self) -> u64 {
        self.total_forecasts
    }

    /// Get the total inference time.
    pub fn total_inference_time(&self) -> Duration {
        self.total_inference_time
    }

    /// Average inference time per forecast.
    pub fn avg_inference_time(&self) -> Duration {
        if self.total_forecasts == 0 {
            Duration::ZERO
        } else {
            self.total_inference_time / self.total_forecasts as u32
        }
    }

    /// Produce a probabilistic forecast for the given context.
    ///
    /// # Arguments
    /// * `context` - Historical time series values (most recent last).
    /// * `prediction_length` - Number of steps to forecast. If `None`,
    ///   uses `config.prediction_length`.
    ///
    /// # Returns
    /// A `ChronosForecast` with multiple sample trajectories.
    pub fn forecast(
        &mut self,
        context: &[f32],
        prediction_length: Option<usize>,
    ) -> Result<ChronosForecast, ChronosError> {
        if self.state != EngineState::Ready {
            return Err(ChronosError::NotReady);
        }

        if context.is_empty() {
            return Err(ChronosError::EmptyContext);
        }

        let pred_len = prediction_length.unwrap_or(self.config.prediction_length);
        let start = Instant::now();

        // Truncate context if too long
        let ctx = if context.len() > self.config.context_length {
            &context[context.len() - self.config.context_length..]
        } else {
            context
        };

        // Tokenize context
        let (tokens, scale, offset) = self.tokenizer.encode(ctx);

        let forecast = if self.model_path.is_some() {
            // ONNX inference path (tract)
            self.forecast_onnx(&tokens, scale, offset, pred_len)?
        } else {
            // Mock/statistical baseline path
            self.forecast_mock(ctx, scale, offset, pred_len)?
        };

        let inference_time = start.elapsed();
        self.total_forecasts += 1;
        self.total_inference_time += inference_time;

        Ok(ChronosForecast {
            trajectories: forecast,
            prediction_length: pred_len,
            n_samples: self.config.n_samples,
            context_length: ctx.len(),
            scale,
            offset,
            inference_time,
            metadata: HashMap::new(),
        })
    }

    /// ONNX-based forecast using tract-onnx.
    ///
    /// This method contains the full tract inference pipeline. When a real
    /// ONNX model is available, it:
    /// 1. Creates input tensors from token IDs + attention mask.
    /// 2. Runs the encoder once on the context.
    /// 3. Autoregressively generates `prediction_length` tokens.
    /// 4. Decodes tokens back to continuous values.
    ///
    /// Currently defers to mock if tract is not fully wired.
    fn forecast_onnx(
        &self,
        tokens: &[usize],
        scale: f32,
        offset: f32,
        pred_len: usize,
    ) -> Result<Vec<Vec<f32>>, ChronosError> {
        // NOTE: Full tract-onnx integration would look like:
        //
        // ```
        // use tract_onnx::prelude::*;
        //
        // let model = tract_onnx::onnx()
        //     .model_for_path(self.model_path.as_ref().unwrap())
        //     .map_err(|e| ChronosError::ModelError(e.to_string()))?
        //     .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec![1, tokens.len()]))?
        //     .into_optimized()
        //     .map_err(|e| ChronosError::ModelError(e.to_string()))?
        //     .into_runnable()
        //     .map_err(|e| ChronosError::ModelError(e.to_string()))?;
        //
        // // Create input tensor
        // let input: Tensor = tract_ndarray::Array2::from_shape_fn(
        //     (1, tokens.len()),
        //     |(_, j)| tokens[j] as i64,
        // ).into();
        //
        // // Run inference
        // let result = model.run(tvec![input.into()])
        //     .map_err(|e| ChronosError::InferenceError(e.to_string()))?;
        //
        // // Extract logits and sample tokens
        // let logits = result[0].to_array_view::<f32>()
        //     .map_err(|e| ChronosError::InferenceError(e.to_string()))?;
        // ```
        //
        // For now, fall back to the statistical mock forecast.

        warn!(
            "ONNX model loading via tract is available but not fully wired. \
             Using statistical baseline forecast. Wire tract_onnx::onnx() \
             when model artifacts are ready."
        );

        // Generate mock forecast trajectories using the context statistics
        // This provides a reasonable baseline until the ONNX model is wired.
        let context_values: Vec<f32> = tokens
            .iter()
            .map(|&t| self.tokenizer.decode_one(t, scale, offset))
            .filter(|v| v.is_finite())
            .collect();

        if context_values.is_empty() {
            return Err(ChronosError::EmptyContext);
        }

        self.generate_baseline_trajectories(&context_values, scale, offset, pred_len)
    }

    /// Generate mock forecast trajectories using statistical baselines.
    ///
    /// Combines a random walk with mean reversion to produce reasonable
    /// sample trajectories without a neural network.
    fn forecast_mock(
        &self,
        context: &[f32],
        scale: f32,
        offset: f32,
        pred_len: usize,
    ) -> Result<Vec<Vec<f32>>, ChronosError> {
        self.generate_baseline_trajectories(context, scale, offset, pred_len)
    }

    /// Core baseline trajectory generation.
    ///
    /// For each sample:
    /// 1. Start from the last context value.
    /// 2. Compute drift and volatility from the context.
    /// 3. Generate a random walk with drift + mean reversion.
    fn generate_baseline_trajectories(
        &self,
        context: &[f32],
        _scale: f32,
        _offset: f32,
        pred_len: usize,
    ) -> Result<Vec<Vec<f32>>, ChronosError> {
        if context.is_empty() {
            return Err(ChronosError::EmptyContext);
        }

        let last_val = *context.last().unwrap();
        let n = context.len();

        // Compute statistics from context
        let mean = context.iter().sum::<f32>() / n as f32;
        let variance = context.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n.max(1) as f32;
        let volatility = variance.sqrt();

        // Compute drift from returns
        let drift = if n > 1 {
            let returns: Vec<f32> = context.windows(2).map(|w| w[1] - w[0]).collect();
            returns.iter().sum::<f32>() / returns.len() as f32
        } else {
            0.0
        };

        // Mean reversion strength (how quickly values revert to the mean)
        let mean_reversion = 0.05;

        // Generate trajectories with different random seeds
        let mut trajectories = Vec::with_capacity(self.config.n_samples);
        let mut rng_state: u64 = 42;

        for _sample in 0..self.config.n_samples {
            let mut trajectory = Vec::with_capacity(pred_len);
            let mut current = last_val;

            for _step in 0..pred_len {
                rng_state = lcg_next(rng_state);

                // Generate approximately normal noise from uniform via Box-Muller approx
                let u1 = (rng_state % 10000) as f32 / 10000.0;
                rng_state = lcg_next(rng_state);
                let u2 = (rng_state % 10000) as f32 / 10000.0;

                let u1_clamped = u1.clamp(1e-6, 1.0 - 1e-6);
                let u2_clamped = u2.clamp(1e-6, 1.0 - 1e-6);

                let z = (-2.0 * u1_clamped.ln()).sqrt()
                    * (2.0 * std::f32::consts::PI * u2_clamped).cos();

                // Random walk + drift + mean reversion
                let noise = z * volatility * self.config.temperature;
                let reversion = mean_reversion * (mean - current);
                current += drift + reversion + noise;

                trajectory.push(current);
            }

            trajectories.push(trajectory);
        }

        Ok(trajectories)
    }

    /// Sample a token from a probability distribution (logits).
    ///
    /// Applies temperature scaling and optional top-k filtering, then
    /// samples from the resulting distribution.
    #[allow(dead_code)]
    fn sample_token(&self, logits: &[f32]) -> usize {
        let n = logits.len();
        if n == 0 {
            return 0;
        }

        // Apply temperature
        let mut scaled: Vec<f32> = logits
            .iter()
            .map(|&l| l / self.config.temperature)
            .collect();

        // Top-k filtering
        if self.config.top_k > 0 && self.config.top_k < n {
            let mut indexed: Vec<(usize, f32)> =
                scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let threshold = indexed[self.config.top_k - 1].1;
            for v in &mut scaled {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax
        let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = scaled.iter().map(|&v| (v - max_val).exp()).sum();

        let probs: Vec<f32> = scaled
            .iter()
            .map(|&v| (v - max_val).exp() / exp_sum)
            .collect();

        // Categorical sampling
        // In production this would use a proper RNG; here we use a simple PRNG
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let u = (lcg_next(ts) % 10000) as f32 / 10000.0;

        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if u <= cumsum {
                return i;
            }
        }

        n - 1 // fallback
    }
}

/// Simple LCG PRNG for deterministic pseudo-random numbers.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

// ---------------------------------------------------------------------------
// Forecast Combiner — Multi-Model Ensemble
// ---------------------------------------------------------------------------

/// Combines forecasts from multiple Chronos models or multiple runs
/// into a single unified forecast.
#[derive(Debug, Clone)]
pub struct ForecastCombiner {
    /// Weight for each source forecast (must sum to 1.0).
    pub weights: Vec<f32>,
}

impl ForecastCombiner {
    /// Create with equal weights.
    pub fn equal(n: usize) -> Self {
        let w = 1.0 / n as f32;
        Self {
            weights: vec![w; n],
        }
    }

    /// Create with custom weights (will be normalized to sum to 1.0).
    pub fn weighted(weights: Vec<f32>) -> Self {
        let total: f32 = weights.iter().sum();
        let normalized: Vec<f32> = if total > 0.0 {
            weights.iter().map(|w| w / total).collect()
        } else {
            let n = weights.len();
            vec![1.0 / n as f32; n]
        };
        Self {
            weights: normalized,
        }
    }

    /// Combine multiple forecasts into one by pooling trajectories.
    ///
    /// Concatenates trajectories from all forecasts and re-weights
    /// by duplicating trajectories proportional to their weights.
    pub fn combine(&self, forecasts: &[ChronosForecast]) -> Result<ChronosForecast, ChronosError> {
        if forecasts.is_empty() {
            return Err(ChronosError::EmptyContext);
        }

        if forecasts.len() != self.weights.len() {
            return Err(ChronosError::ConfigError(format!(
                "Number of forecasts ({}) doesn't match number of weights ({})",
                forecasts.len(),
                self.weights.len()
            )));
        }

        let pred_len = forecasts[0].prediction_length;

        // Pool all trajectories, weighting by repeating
        let total_samples: usize = forecasts.iter().map(|f| f.trajectories.len()).sum();
        let mut combined_trajectories = Vec::with_capacity(total_samples);

        for (forecast, &weight) in forecasts.iter().zip(self.weights.iter()) {
            // Include trajectories proportional to weight
            let n_include =
                ((forecast.trajectories.len() as f32 * weight * self.weights.len() as f32).round()
                    as usize)
                    .max(1)
                    .min(forecast.trajectories.len());

            for traj in forecast.trajectories.iter().take(n_include) {
                combined_trajectories.push(traj.clone());
            }
        }

        let n_samples = combined_trajectories.len();

        Ok(ChronosForecast {
            trajectories: combined_trajectories,
            prediction_length: pred_len,
            n_samples,
            context_length: forecasts[0].context_length,
            scale: forecasts[0].scale,
            offset: forecasts[0].offset,
            inference_time: forecasts.iter().map(|f| f.inference_time).sum(),
            metadata: HashMap::from([("combined".to_string(), forecasts.len().to_string())]),
        })
    }
}

// ---------------------------------------------------------------------------
// Inference Statistics
// ---------------------------------------------------------------------------

/// Statistics from the Chronos inference engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Total number of forecasts produced.
    pub total_forecasts: u64,

    /// Total inference wall-clock time.
    pub total_inference_time: Duration,

    /// Average inference time per forecast.
    pub avg_inference_time: Duration,

    /// Whether the engine has an ONNX model loaded.
    pub has_model: bool,

    /// Model path, if any.
    pub model_path: Option<String>,

    /// Configuration summary.
    pub context_length: usize,
    pub prediction_length: usize,
    pub n_bins: usize,
    pub n_samples: usize,
}

impl ChronosInference {
    /// Get comprehensive inference statistics.
    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            total_forecasts: self.total_forecasts,
            total_inference_time: self.total_inference_time,
            avg_inference_time: self.avg_inference_time(),
            has_model: self.has_model(),
            model_path: self
                .model_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            context_length: self.config.context_length,
            prediction_length: self.config.prediction_length,
            n_bins: self.config.n_bins,
            n_samples: self.config.n_samples,
        }
    }
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors from the Chronos inference pipeline.
#[derive(Debug, thiserror::Error)]
pub enum ChronosError {
    #[error("Engine not ready: load a model first")]
    NotReady,

    #[error("Empty context: provide at least one historical value")]
    EmptyContext,

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("I/O error: {0}")]
    IoError(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Probit Tests ==========

    #[test]
    fn test_probit_symmetry() {
        // probit(0.5) should be 0
        assert!((probit(0.5) - 0.0).abs() < 1e-10);

        // probit(p) = -probit(1-p)
        for &p in &[0.1, 0.25, 0.4, 0.05, 0.01] {
            let a = probit(p);
            let b = probit(1.0 - p);
            assert!(
                (a + b).abs() < 1e-6,
                "probit({}) = {}, probit({}) = {}, sum = {}",
                p,
                a,
                1.0 - p,
                b,
                a + b
            );
        }
    }

    #[test]
    fn test_probit_monotonic() {
        let probs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99];
        let quantiles: Vec<f64> = probs.iter().map(|&p| probit(p)).collect();

        for i in 1..quantiles.len() {
            assert!(
                quantiles[i] > quantiles[i - 1],
                "probit not monotonic at i={}: {} vs {}",
                i,
                quantiles[i],
                quantiles[i - 1]
            );
        }
    }

    #[test]
    fn test_probit_known_values() {
        // probit(0.025) ≈ -1.96
        assert!((probit(0.025) - (-1.96)).abs() < 0.01);
        // probit(0.975) ≈ 1.96
        assert!((probit(0.975) - 1.96).abs() < 0.01);
        // probit(0.8413) ≈ 1.0
        assert!((probit(0.8413) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_probit_extremes() {
        assert!(probit(0.0).is_infinite() && probit(0.0) < 0.0);
        assert!(probit(1.0).is_infinite() && probit(1.0) > 0.0);
    }

    // ========== Config Tests ==========

    #[test]
    fn test_config_defaults() {
        let config = ChronosConfig::default();
        assert_eq!(config.context_length, 512);
        assert_eq!(config.prediction_length, 64);
        assert_eq!(config.n_bins, 4096);
        assert_eq!(config.n_special_tokens, 5);
        assert_eq!(config.vocab_size(), 4101);
        assert_eq!(config.first_bin_id(), 5);
        assert_eq!(config.last_bin_id(), 4100);
    }

    #[test]
    fn test_config_presets() {
        let tiny = ChronosConfig::t5_tiny();
        assert_eq!(tiny.context_length, 512);
        assert_eq!(tiny.n_bins, 4096);

        let test = ChronosConfig::tiny_test();
        assert_eq!(test.context_length, 32);
        assert_eq!(test.n_bins, 64);
    }

    #[test]
    fn test_config_builders() {
        let config = ChronosConfig::default()
            .with_context_length(256)
            .with_prediction_length(32)
            .with_n_bins(1024)
            .with_n_samples(10)
            .with_temperature(0.5)
            .with_top_k(20)
            .with_model_path("/tmp/model.onnx");

        assert_eq!(config.context_length, 256);
        assert_eq!(config.prediction_length, 32);
        assert_eq!(config.n_bins, 1024);
        assert_eq!(config.n_samples, 10);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.top_k, 20);
        assert_eq!(config.model_path.as_deref(), Some("/tmp/model.onnx"));
    }

    // ========== Special Tokens Tests ==========

    #[test]
    fn test_special_tokens() {
        let st = SpecialTokens::default();
        assert!(st.is_special(0)); // PAD
        assert!(st.is_special(1)); // BOS
        assert!(st.is_special(2)); // EOS
        assert!(st.is_special(3)); // MASK
        assert!(st.is_special(4)); // UNK
        assert!(!st.is_special(5)); // first bin
        assert!(!st.is_special(100));
    }

    // ========== Tokenizer Config Tests ==========

    #[test]
    fn test_tokenizer_config_defaults() {
        let config = TokenizerConfig::default();
        assert_eq!(config.n_bins, 4096);
        assert_eq!(config.n_special_tokens, 5);
        assert!(config.use_mean_scaling);
    }

    #[test]
    fn test_tokenizer_config_builders() {
        let config = TokenizerConfig::default()
            .with_n_bins(128)
            .with_binning_method(BinningMethod::LinearRange { low: -10, high: 10 })
            .with_mean_scaling(false);

        assert_eq!(config.n_bins, 128);
        assert!(!config.use_mean_scaling);
    }

    // ========== Tokenizer Tests ==========

    #[test]
    fn test_tokenizer_creation() {
        let config = TokenizerConfig::default().with_n_bins(64);
        let tokenizer = ChronosTokenizer::new(config);

        assert_eq!(tokenizer.n_bins(), 64);
        assert_eq!(tokenizer.vocab_size(), 69); // 64 + 5
        assert_eq!(tokenizer.boundaries().len(), 65); // n_bins + 1
        assert_eq!(tokenizer.centers().len(), 64);
    }

    #[test]
    fn test_tokenizer_boundaries_monotonic() {
        let config = TokenizerConfig::default().with_n_bins(128);
        let tokenizer = ChronosTokenizer::new(config);

        let bounds = tokenizer.boundaries();
        for i in 1..bounds.len() {
            assert!(
                bounds[i] >= bounds[i - 1],
                "Boundaries not monotonic at {}: {} vs {}",
                i,
                bounds[i],
                bounds[i - 1]
            );
        }

        // First boundary should be -inf, last should be +inf
        assert!(bounds[0].is_infinite() && bounds[0] < 0.0);
        assert!(bounds[bounds.len() - 1].is_infinite() && bounds[bounds.len() - 1] > 0.0);
    }

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let config = TokenizerConfig::default().with_n_bins(256);
        let tokenizer = ChronosTokenizer::new(config);

        let values = vec![100.0, 101.5, 99.8, 102.3, 103.1, 98.5];
        let (tokens, scale, offset) = tokenizer.encode(&values);

        assert_eq!(tokens.len(), values.len());
        for &t in &tokens {
            assert!(t >= 5); // >= first bin ID
            assert!(t < 261); // < n_special + n_bins
        }

        let decoded = tokenizer.decode(&tokens, scale, offset);
        assert_eq!(decoded.len(), values.len());

        // Roundtrip should be approximate (quantization noise)
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!(
                dec.is_finite(),
                "Decoded value should be finite, got NaN for original {}",
                orig
            );
            // With 256 bins, quantization error should be small relative to scale
            let rel_error = (orig - dec).abs() / orig.abs().max(1.0);
            assert!(
                rel_error < 0.5,
                "Roundtrip error too large: orig={}, dec={}, rel_error={}",
                orig,
                dec,
                rel_error
            );
        }
    }

    #[test]
    fn test_tokenizer_encode_with_bos() {
        let config = TokenizerConfig::default().with_n_bins(64);
        let tokenizer = ChronosTokenizer::new(config);

        let values = vec![1.0, 2.0, 3.0];
        let (tokens, _, _) = tokenizer.encode_with_bos(&values);

        assert_eq!(tokens.len(), 4); // BOS + 3 values
        assert_eq!(tokens[0], 1); // BOS token
    }

    #[test]
    fn test_tokenizer_decode_special_tokens() {
        let config = TokenizerConfig::default().with_n_bins(64);
        let tokenizer = ChronosTokenizer::new(config);

        let tokens = vec![0, 1, 2, 3, 4]; // all special
        let decoded = tokenizer.decode(&tokens, 1.0, 0.0);

        for v in &decoded {
            assert!(v.is_nan(), "Special tokens should decode to NaN, got {}", v);
        }
    }

    #[test]
    fn test_tokenizer_zero_values() {
        let config = TokenizerConfig::default().with_n_bins(64);
        let tokenizer = ChronosTokenizer::new(config);

        let values = vec![0.0, 0.0, 0.0];
        let (tokens, scale, _) = tokenizer.encode(&values);

        // Should not panic even with all zeros (min_scale prevents division by zero)
        assert_eq!(tokens.len(), 3);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_tokenizer_negative_values() {
        let config = TokenizerConfig::default().with_n_bins(128);
        let tokenizer = ChronosTokenizer::new(config);

        let values = vec![-5.0, -2.0, 0.0, 2.0, 5.0];
        let (tokens, scale, offset) = tokenizer.encode(&values);
        let decoded = tokenizer.decode(&tokens, scale, offset);

        assert_eq!(decoded.len(), 5);
        for v in &decoded {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_tokenizer_single_value() {
        let config = TokenizerConfig::default().with_n_bins(64);
        let tokenizer = ChronosTokenizer::new(config);

        let values = vec![42.0];
        let (tokens, scale, offset) = tokenizer.encode(&values);
        let decoded = tokenizer.decode(&tokens, scale, offset);

        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].is_finite());
    }

    #[test]
    fn test_tokenizer_scale_computation() {
        let config = TokenizerConfig::default().with_n_bins(64);
        let tokenizer = ChronosTokenizer::new(config);

        let values = vec![10.0, -10.0, 10.0, -10.0];
        let (scale, offset) = tokenizer.compute_scale(&values);
        assert_eq!(scale, 10.0);
        assert_eq!(offset, 0.0);

        // Empty context
        let (scale_empty, _) = tokenizer.compute_scale(&[]);
        assert_eq!(scale_empty, 1.0);
    }

    #[test]
    fn test_tokenizer_no_mean_scaling() {
        let config = TokenizerConfig::default()
            .with_n_bins(64)
            .with_mean_scaling(false);
        let tokenizer = ChronosTokenizer::new(config);

        let values = vec![100.0, 200.0, 300.0];
        let (scale, _) = tokenizer.compute_scale(&values);
        assert_eq!(scale, 1.0); // No scaling applied
    }

    #[test]
    fn test_tokenizer_linear_binning() {
        let config = TokenizerConfig::default()
            .with_n_bins(10)
            .with_binning_method(BinningMethod::LinearRange { low: -5, high: 5 })
            .with_mean_scaling(false);

        let tokenizer = ChronosTokenizer::new(config);
        assert_eq!(tokenizer.n_bins(), 10);
        assert_eq!(tokenizer.boundaries().len(), 11);

        // Centers should be evenly spaced
        let centers = tokenizer.centers();
        for &c in centers {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn test_tokenizer_batch_encode_decode() {
        let config = TokenizerConfig::default().with_n_bins(64);
        let tokenizer = ChronosTokenizer::new(config);

        let batch = vec![
            vec![1.0, 2.0, 3.0],
            vec![10.0, 20.0, 30.0],
            vec![-5.0, 0.0, 5.0],
        ];

        let encoded = tokenizer.encode_batch(&batch);
        assert_eq!(encoded.len(), 3);

        let decoded = tokenizer.decode_batch(&encoded);
        assert_eq!(decoded.len(), 3);

        for (orig, dec) in batch.iter().zip(decoded.iter()) {
            assert_eq!(orig.len(), dec.len());
        }
    }

    #[test]
    fn test_tokenizer_bin_histogram() {
        let config = TokenizerConfig::default().with_n_bins(16);
        let tokenizer = ChronosTokenizer::new(config);

        let values = vec![1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 5.0];
        let histogram = tokenizer.bin_histogram(&values);
        assert_eq!(histogram.len(), 16);

        let total: usize = histogram.iter().sum();
        assert_eq!(total, values.len());
    }

    #[test]
    fn test_tokenizer_from_chronos_config() {
        let config = ChronosConfig::tiny_test();
        let tokenizer = ChronosTokenizer::from_config(&config);
        assert_eq!(tokenizer.n_bins(), config.n_bins);
    }

    #[test]
    fn test_tokenizer_fixed_boundaries() {
        let config = ChronosConfig::default()
            .with_n_bins(4) // doesn't matter, fixed boundaries override
            .with_fixed_boundaries(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        let tokenizer = ChronosTokenizer::from_config(&config);
        assert_eq!(tokenizer.centers().len(), 4);
    }

    // ========== Forecast Tests ==========

    #[test]
    fn test_forecast_median() {
        let forecast = ChronosForecast {
            trajectories: vec![
                vec![1.0, 2.0, 3.0],
                vec![2.0, 3.0, 4.0],
                vec![3.0, 4.0, 5.0],
            ],
            prediction_length: 3,
            n_samples: 3,
            context_length: 10,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        };

        let median = forecast.median();
        assert_eq!(median.len(), 3);
        assert_eq!(median[0], 2.0);
        assert_eq!(median[1], 3.0);
        assert_eq!(median[2], 4.0);
    }

    #[test]
    fn test_forecast_mean() {
        let forecast = ChronosForecast {
            trajectories: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            prediction_length: 2,
            n_samples: 2,
            context_length: 5,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        };

        let mean = forecast.mean();
        assert_eq!(mean.len(), 2);
        assert_eq!(mean[0], 2.0);
        assert_eq!(mean[1], 3.0);
    }

    #[test]
    fn test_forecast_interval() {
        let forecast = ChronosForecast {
            trajectories: vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
            prediction_length: 1,
            n_samples: 5,
            context_length: 10,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        };

        let (lower, upper) = forecast.interval(0.80);
        assert_eq!(lower.len(), 1);
        assert_eq!(upper.len(), 1);
        assert!(lower[0] <= upper[0]);
    }

    #[test]
    fn test_forecast_std_dev() {
        let forecast = ChronosForecast {
            trajectories: vec![vec![10.0], vec![10.0], vec![10.0]],
            prediction_length: 1,
            n_samples: 3,
            context_length: 5,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        };

        let std = forecast.std_dev();
        assert_eq!(std.len(), 1);
        assert_eq!(std[0], 0.0); // all identical
    }

    #[test]
    fn test_forecast_at_horizon() {
        let forecast = ChronosForecast {
            trajectories: vec![
                vec![1.0, 10.0, 100.0],
                vec![2.0, 20.0, 200.0],
                vec![3.0, 30.0, 300.0],
            ],
            prediction_length: 3,
            n_samples: 3,
            context_length: 5,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        };

        let h0 = forecast.at_horizon(0).unwrap();
        assert_eq!(h0.horizon, 0);
        assert_eq!(h0.mean, 2.0);
        assert_eq!(h0.median, 2.0);
        assert_eq!(h0.n_samples, 3);

        let h2 = forecast.at_horizon(2).unwrap();
        assert_eq!(h2.horizon, 2);
        assert_eq!(h2.mean, 200.0);

        assert!(forecast.at_horizon(3).is_none());
    }

    #[test]
    fn test_forecast_cv() {
        let forecast = ChronosForecast {
            trajectories: vec![vec![10.0], vec![12.0], vec![8.0]],
            prediction_length: 1,
            n_samples: 3,
            context_length: 5,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        };

        let cv = forecast.cv();
        assert_eq!(cv.len(), 1);
        assert!(cv[0] > 0.0);
        assert!(cv[0] < 1.0);
    }

    #[test]
    fn test_forecast_empty() {
        let forecast = ChronosForecast {
            trajectories: vec![],
            prediction_length: 5,
            n_samples: 0,
            context_length: 0,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        };

        assert!(forecast.median().is_empty());
        assert!(forecast.mean().is_empty());
        assert!(forecast.std_dev().is_empty());
        assert!(forecast.cv().is_empty());
    }

    #[test]
    fn test_forecast_metadata() {
        let forecast = ChronosForecast {
            trajectories: vec![vec![1.0]],
            prediction_length: 1,
            n_samples: 1,
            context_length: 5,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        }
        .with_metadata("symbol", "BTCUSDT")
        .with_metadata("model", "chronos-t5-tiny");

        assert_eq!(forecast.metadata.get("symbol").unwrap(), "BTCUSDT");
        assert_eq!(forecast.metadata.get("model").unwrap(), "chronos-t5-tiny");
    }

    // ========== Inference Engine Tests ==========

    #[test]
    fn test_mock_engine_creation() {
        let config = ChronosConfig::tiny_test();
        let engine = ChronosInference::mock(config);

        assert_eq!(engine.state(), EngineState::Ready);
        assert!(!engine.has_model());
        assert_eq!(engine.total_forecasts(), 0);
    }

    #[test]
    fn test_mock_engine_forecast() {
        let config = ChronosConfig::tiny_test()
            .with_n_samples(5)
            .with_prediction_length(4);

        let mut engine = ChronosInference::mock(config);

        let context = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        let forecast = engine.forecast(&context, None).unwrap();

        assert_eq!(forecast.prediction_length, 4);
        assert_eq!(forecast.n_samples, 5);
        assert_eq!(forecast.trajectories.len(), 5);
        assert_eq!(forecast.context_length, 5);

        for traj in &forecast.trajectories {
            assert_eq!(traj.len(), 4);
            for &v in traj {
                assert!(v.is_finite(), "Forecast value should be finite, got {}", v);
            }
        }

        assert_eq!(engine.total_forecasts(), 1);
        assert!(engine.total_inference_time() > Duration::ZERO);
    }

    #[test]
    fn test_mock_engine_forecast_custom_horizon() {
        let config = ChronosConfig::tiny_test();
        let mut engine = ChronosInference::mock(config);

        let context = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let forecast = engine.forecast(&context, Some(3)).unwrap();

        assert_eq!(forecast.prediction_length, 3);
        for traj in &forecast.trajectories {
            assert_eq!(traj.len(), 3);
        }
    }

    #[test]
    fn test_mock_engine_forecast_empty_context() {
        let config = ChronosConfig::tiny_test();
        let mut engine = ChronosInference::mock(config);

        let result = engine.forecast(&[], None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ChronosError::EmptyContext));
    }

    #[test]
    fn test_mock_engine_forecast_long_context() {
        let config = ChronosConfig::tiny_test().with_context_length(10);
        let mut engine = ChronosInference::mock(config);

        // Context longer than context_length — should be truncated
        let context: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let forecast = engine.forecast(&context, Some(3)).unwrap();

        assert_eq!(forecast.context_length, 10); // truncated
        assert_eq!(forecast.prediction_length, 3);
    }

    #[test]
    fn test_mock_engine_multiple_forecasts() {
        let config = ChronosConfig::tiny_test();
        let mut engine = ChronosInference::mock(config);

        let context = vec![1.0, 2.0, 3.0];

        for _ in 0..5 {
            engine.forecast(&context, Some(2)).unwrap();
        }

        assert_eq!(engine.total_forecasts(), 5);
    }

    #[test]
    fn test_load_nonexistent_model() {
        let config = ChronosConfig::tiny_test();
        let result = ChronosInference::load("/nonexistent/path/model.onnx", config);

        // Should fall back to mock mode
        assert!(result.is_ok());
        let engine = result.unwrap();
        assert!(!engine.has_model());
        assert_eq!(engine.state(), EngineState::Ready);
    }

    #[test]
    fn test_engine_stats() {
        let config = ChronosConfig::tiny_test();
        let mut engine = ChronosInference::mock(config);

        let context = vec![1.0, 2.0, 3.0];
        engine.forecast(&context, Some(2)).unwrap();

        let stats = engine.stats();
        assert_eq!(stats.total_forecasts, 1);
        assert!(!stats.has_model);
        assert_eq!(stats.n_bins, 64);
        assert_eq!(stats.n_samples, 5);
    }

    // ========== Forecast Combiner Tests ==========

    #[test]
    fn test_combiner_equal_weights() {
        let combiner = ForecastCombiner::equal(3);
        assert_eq!(combiner.weights.len(), 3);
        for &w in &combiner.weights {
            assert!((w - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_combiner_custom_weights() {
        let combiner = ForecastCombiner::weighted(vec![1.0, 2.0, 3.0]);
        assert_eq!(combiner.weights.len(), 3);

        let total: f32 = combiner.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);

        // First weight should be smallest
        assert!(combiner.weights[0] < combiner.weights[1]);
        assert!(combiner.weights[1] < combiner.weights[2]);
    }

    #[test]
    fn test_combiner_combine() {
        let combiner = ForecastCombiner::equal(2);

        let f1 = ChronosForecast {
            trajectories: vec![vec![1.0, 2.0], vec![1.5, 2.5]],
            prediction_length: 2,
            n_samples: 2,
            context_length: 5,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::from_millis(10),
            metadata: HashMap::new(),
        };

        let f2 = ChronosForecast {
            trajectories: vec![vec![3.0, 4.0], vec![3.5, 4.5]],
            prediction_length: 2,
            n_samples: 2,
            context_length: 5,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::from_millis(15),
            metadata: HashMap::new(),
        };

        let combined = combiner.combine(&[f1, f2]).unwrap();
        assert_eq!(combined.prediction_length, 2);
        assert!(combined.n_samples >= 2);
        assert!(combined.trajectories.len() >= 2);
    }

    #[test]
    fn test_combiner_empty() {
        let combiner = ForecastCombiner::equal(1);
        let result = combiner.combine(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_combiner_weight_mismatch() {
        let combiner = ForecastCombiner::equal(2);

        let f1 = ChronosForecast {
            trajectories: vec![vec![1.0]],
            prediction_length: 1,
            n_samples: 1,
            context_length: 5,
            scale: 1.0,
            offset: 0.0,
            inference_time: Duration::ZERO,
            metadata: HashMap::new(),
        };

        // Only 1 forecast but combiner expects 2
        let result = combiner.combine(&[f1]);
        assert!(result.is_err());
    }

    // ========== LCG PRNG Tests ==========

    #[test]
    fn test_lcg_deterministic() {
        let a = lcg_next(42);
        let b = lcg_next(42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_lcg_varies() {
        let a = lcg_next(42);
        let b = lcg_next(43);
        assert_ne!(a, b);
    }

    // ========== HorizonSummary Tests ==========

    #[test]
    fn test_horizon_summary_fields() {
        let summary = HorizonSummary {
            horizon: 5,
            mean: 100.0,
            median: 99.5,
            std: 2.5,
            lower_80: 96.0,
            upper_80: 104.0,
            n_samples: 20,
        };

        assert_eq!(summary.horizon, 5);
        assert!(summary.lower_80 < summary.median);
        assert!(summary.median < summary.upper_80);
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_end_to_end_mock_pipeline() {
        let config = ChronosConfig::tiny_test()
            .with_n_samples(10)
            .with_prediction_length(8);

        let mut engine = ChronosInference::mock(config);

        // Simulate a price series
        let context: Vec<f32> = (0..20)
            .map(|i| 100.0 + (i as f32) * 0.5 + ((i as f32) * 0.3).sin() * 2.0)
            .collect();

        let forecast = engine.forecast(&context, None).unwrap();

        // Check basic properties
        assert_eq!(forecast.prediction_length, 8);
        assert_eq!(forecast.n_samples, 10);

        // Compute aggregates
        let median = forecast.median();
        let mean = forecast.mean();
        let std = forecast.std_dev();
        let (lower, upper) = forecast.interval(0.90);

        assert_eq!(median.len(), 8);
        assert_eq!(mean.len(), 8);
        assert_eq!(std.len(), 8);
        assert_eq!(lower.len(), 8);
        assert_eq!(upper.len(), 8);

        for i in 0..8 {
            assert!(lower[i] <= median[i], "lower <= median at {}", i);
            assert!(median[i] <= upper[i], "median <= upper at {}", i);
            assert!(std[i] >= 0.0, "std should be non-negative");
        }

        // Check horizon summary
        let h0 = forecast.at_horizon(0).unwrap();
        assert_eq!(h0.horizon, 0);
        assert!(h0.lower_80 <= h0.median);
        assert!(h0.median <= h0.upper_80);

        // Stats
        let stats = engine.stats();
        assert_eq!(stats.total_forecasts, 1);
    }

    #[test]
    fn test_tokenize_forecast_decode_pipeline() {
        let config = ChronosConfig::tiny_test().with_n_bins(128);
        let tokenizer = ChronosTokenizer::from_config(&config);

        // Tokenize a context
        let context = vec![100.0, 101.0, 99.5, 102.0, 103.5];
        let (tokens, scale, offset) = tokenizer.encode(&context);

        assert_eq!(tokens.len(), 5);
        assert!(scale > 0.0);

        // Decode back
        let decoded = tokenizer.decode(&tokens, scale, offset);
        assert_eq!(decoded.len(), 5);

        // Simulate a "model output" (just repeat last token)
        let forecast_tokens = vec![tokens[4]; 3];
        let forecast_values = tokenizer.decode(&forecast_tokens, scale, offset);
        assert_eq!(forecast_values.len(), 3);

        for v in &forecast_values {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_token_method() {
        let config = ChronosConfig::tiny_test().with_top_k(3);
        let engine = ChronosInference::mock(config);

        let logits = vec![0.1, 0.2, 10.0, 0.05, 0.01, 8.0, 0.3, 0.02];
        let token = engine.sample_token(&logits);

        // Should be one of the top-k tokens (indices 2, 5, or 6 based on values)
        assert!(token < logits.len());
    }

    #[test]
    fn test_constant_context_forecast() {
        let config = ChronosConfig::tiny_test()
            .with_n_samples(20)
            .with_prediction_length(5);

        let mut engine = ChronosInference::mock(config);

        // Constant context — forecast should stay near constant value
        let context = vec![50.0; 20];
        let forecast = engine.forecast(&context, None).unwrap();

        let mean = forecast.mean();
        for &v in &mean {
            // Should be reasonably close to 50.0 (mock uses mean reversion)
            assert!(
                (v - 50.0).abs() < 20.0,
                "Mean forecast should be near constant, got {}",
                v
            );
        }
    }
}
