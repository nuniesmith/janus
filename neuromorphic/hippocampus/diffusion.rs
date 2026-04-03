//! Denoising Diffusion Probabilistic Models (DDPM) for Synthetic Time Series
//!
//! Implements a regime-conditioned DDPM for generating synthetic financial
//! time series data to augment sparse regime training data. Based on
//! Ho et al. (2020) "Denoising Diffusion Probabilistic Models" adapted
//! for 1-D financial time series.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Diffusion Model Pipeline                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  Forward Process (fixed):                                    │
//! │  x_0 ──β₁──▶ x_1 ──β₂──▶ ... ──β_T──▶ x_T ~ N(0, I)     │
//! │                                                              │
//! │  Reverse Process (learned):                                  │
//! │  x_T ~ N(0, I) ──ε_θ──▶ x_{T-1} ──▶ ... ──▶ x_0          │
//! │                                                              │
//! │  ┌──────────────┐     ┌───────────────┐     ┌────────────┐ │
//! │  │ Noise        │     │ Denoising     │     │ Regime     │ │
//! │  │ Scheduler    │────▶│ Network       │◀────│ Condition  │ │
//! │  │ (β schedule) │     │ (MLP/ResNet)  │     │ Embedding  │ │
//! │  └──────────────┘     └───────┬───────┘     └────────────┘ │
//! │                               │                             │
//! │                        ┌──────▼──────┐                      │
//! │                        │  Synthetic  │                      │
//! │                        │  Time Series│                      │
//! │                        │  Samples    │                      │
//! │                        └─────────────┘                      │
//! │                                                              │
//! │  Use cases:                                                  │
//! │  • Augment sparse volatile regime training data             │
//! │  • Generate realistic market scenarios for stress testing   │
//! │  • Conditional generation per regime class                  │
//! │  • Replay buffer enrichment for RL training                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_neuromorphic::hippocampus::diffusion::*;
//!
//! // Configure the diffusion model
//! let config = DiffusionConfig::default()
//!     .with_seq_len(64)
//!     .with_n_features(5)
//!     .with_n_regimes(4);
//!
//! let mut ddpm = TimeSeriesDDPM::new(config)?;
//!
//! // Train on real time series data with regime labels
//! let data: Vec<TimeSeriesSample> = load_training_data();
//! let stats = ddpm.train(&data)?;
//!
//! // Generate synthetic samples conditioned on a regime
//! let synthetic = ddpm.sample(16, Some(RegimeLabel::Volatile))?;
//! ```

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, Optimizer, VarBuilder, VarMap, linear};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the diffusion model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionConfig {
    /// Length of time series sequences (number of time steps).
    pub seq_len: usize,

    /// Number of features per time step (e.g., OHLCV = 5).
    pub n_features: usize,

    /// Number of diffusion timesteps T. More steps = higher quality but slower.
    /// Typical range: 100–1000.
    pub n_timesteps: usize,

    /// Type of noise schedule.
    pub schedule: NoiseScheduleType,

    /// Beta range start for linear schedule.
    pub beta_start: f64,

    /// Beta range end for linear schedule.
    pub beta_end: f64,

    /// Number of distinct regime classes for conditional generation.
    /// Set to 0 for unconditional generation.
    pub n_regimes: usize,

    /// Dimension of the regime conditioning embedding.
    pub regime_embed_dim: usize,

    /// Dimension of the sinusoidal timestep embedding.
    pub time_embed_dim: usize,

    /// Hidden layer dimensions for the denoising network.
    pub hidden_dims: Vec<usize>,

    /// Learning rate for the optimizer.
    pub learning_rate: f64,

    /// Weight decay (L2 regularization).
    pub weight_decay: f64,

    /// Number of training epochs.
    pub n_epochs: usize,

    /// Mini-batch size.
    pub batch_size: usize,

    /// Whether to use residual connections in the denoising network.
    pub use_residual: bool,

    /// Dropout rate for the denoising network (0.0 = no dropout).
    pub dropout_rate: f32,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// EMA decay rate for model weights (0.0 = no EMA).
    pub ema_decay: f64,

    /// Number of epochs between logging training metrics.
    pub log_interval: usize,

    /// Whether to apply gradient clipping.
    pub clip_grad_norm: Option<f64>,

    /// Loss weighting type.
    pub loss_weighting: LossWeighting,

    /// Sampling variance type during reverse process.
    pub variance_type: VarianceType,
}

/// Type of noise schedule for the forward diffusion process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseScheduleType {
    /// Linear schedule: β_t increases linearly from beta_start to beta_end.
    Linear,
    /// Cosine schedule (Nichol & Dhariwal, 2021): smoother noise injection,
    /// better for small timestep counts.
    Cosine,
    /// Quadratic schedule: β_t = (β_start^0.5 + t/T * (β_end^0.5 - β_start^0.5))^2
    Quadratic,
}

/// How to weight the loss across diffusion timesteps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossWeighting {
    /// Uniform weighting across all timesteps.
    Uniform,
    /// SNR-based weighting (min-SNR-γ strategy).
    MinSnr { gamma: u32 },
    /// Importance sampling based on loss magnitude.
    ImportanceSampling,
}

/// Variance type for the reverse process sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarianceType {
    /// Fixed small variance: σ²_t = β_t
    FixedSmall,
    /// Fixed large variance: σ²_t = β̃_t = (1 - ᾱ_{t-1})/(1 - ᾱ_t) * β_t
    FixedLarge,
    /// Learned variance (model predicts log variance alongside noise).
    Learned,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            seq_len: 64,
            n_features: 5,
            n_timesteps: 200,
            schedule: NoiseScheduleType::Cosine,
            beta_start: 1e-4,
            beta_end: 0.02,
            n_regimes: 4,
            regime_embed_dim: 32,
            time_embed_dim: 64,
            hidden_dims: vec![256, 512, 512, 256],
            learning_rate: 1e-4,
            weight_decay: 1e-5,
            n_epochs: 100,
            batch_size: 32,
            use_residual: true,
            dropout_rate: 0.1,
            seed: 42,
            ema_decay: 0.9999,
            log_interval: 10,
            clip_grad_norm: Some(1.0),
            loss_weighting: LossWeighting::Uniform,
            variance_type: VarianceType::FixedSmall,
        }
    }
}

impl DiffusionConfig {
    /// Preset for OHLCV time series (5 features, 64 steps).
    pub fn ohlcv() -> Self {
        Self {
            seq_len: 64,
            n_features: 5,
            n_timesteps: 200,
            hidden_dims: vec![256, 512, 512, 256],
            ..Default::default()
        }
    }

    /// Preset for returns-only time series (1 feature, 128 steps).
    pub fn returns_only() -> Self {
        Self {
            seq_len: 128,
            n_features: 1,
            n_timesteps: 200,
            hidden_dims: vec![128, 256, 256, 128],
            ..Default::default()
        }
    }

    /// Preset for multi-asset features (many features, shorter sequence).
    pub fn multi_asset(n_features: usize) -> Self {
        Self {
            seq_len: 32,
            n_features,
            n_timesteps: 300,
            hidden_dims: vec![512, 1024, 1024, 512],
            learning_rate: 5e-5,
            ..Default::default()
        }
    }

    /// Lightweight config for testing.
    pub fn tiny() -> Self {
        Self {
            seq_len: 16,
            n_features: 2,
            n_timesteps: 50,
            hidden_dims: vec![32, 64, 32],
            n_epochs: 5,
            batch_size: 8,
            log_interval: 1,
            ..Default::default()
        }
    }

    /// Builder: set sequence length.
    pub fn with_seq_len(mut self, len: usize) -> Self {
        self.seq_len = len;
        self
    }

    /// Builder: set number of features.
    pub fn with_n_features(mut self, n: usize) -> Self {
        self.n_features = n;
        self
    }

    /// Builder: set number of diffusion timesteps.
    pub fn with_n_timesteps(mut self, n: usize) -> Self {
        self.n_timesteps = n;
        self
    }

    /// Builder: set number of regime classes.
    pub fn with_n_regimes(mut self, n: usize) -> Self {
        self.n_regimes = n;
        self
    }

    /// Builder: set noise schedule type.
    pub fn with_schedule(mut self, schedule: NoiseScheduleType) -> Self {
        self.schedule = schedule;
        self
    }

    /// Builder: set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder: set number of training epochs.
    pub fn with_n_epochs(mut self, n: usize) -> Self {
        self.n_epochs = n;
        self
    }

    /// Builder: set hidden layer dimensions.
    pub fn with_hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Builder: set batch size.
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Builder: set variance type.
    pub fn with_variance_type(mut self, vt: VarianceType) -> Self {
        self.variance_type = vt;
        self
    }

    /// Total flat dimension of a single sample: seq_len * n_features.
    pub fn flat_dim(&self) -> usize {
        self.seq_len * self.n_features
    }

    /// Whether the model is regime-conditioned.
    pub fn is_conditional(&self) -> bool {
        self.n_regimes > 0
    }
}

// ---------------------------------------------------------------------------
// Regime Labels
// ---------------------------------------------------------------------------

/// Regime label for conditional generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegimeLabel {
    /// Strong directional movement.
    Trending,
    /// Price oscillating around a mean.
    MeanReverting,
    /// High volatility, no clear direction.
    Volatile,
    /// Unclear or mixed signals.
    Uncertain,
    /// Custom regime with numeric ID (for extended regime sets).
    Custom(u32),
}

impl RegimeLabel {
    /// Convert to a zero-based index for embedding lookup.
    pub fn to_index(&self) -> usize {
        match self {
            RegimeLabel::Trending => 0,
            RegimeLabel::MeanReverting => 1,
            RegimeLabel::Volatile => 2,
            RegimeLabel::Uncertain => 3,
            RegimeLabel::Custom(id) => 4 + *id as usize,
        }
    }

    /// Create from a zero-based index.
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => RegimeLabel::Trending,
            1 => RegimeLabel::MeanReverting,
            2 => RegimeLabel::Volatile,
            3 => RegimeLabel::Uncertain,
            n => RegimeLabel::Custom((n - 4) as u32),
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            RegimeLabel::Trending => "trending",
            RegimeLabel::MeanReverting => "mean_reverting",
            RegimeLabel::Volatile => "volatile",
            RegimeLabel::Uncertain => "uncertain",
            RegimeLabel::Custom(_) => "custom",
        }
    }
}

impl std::fmt::Display for RegimeLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegimeLabel::Custom(id) => write!(f, "custom_{}", id),
            other => write!(f, "{}", other.label()),
        }
    }
}

// ---------------------------------------------------------------------------
// Training Sample
// ---------------------------------------------------------------------------

/// A single time series sample for training the diffusion model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesSample {
    /// Time series data, flattened: shape [seq_len * n_features].
    /// Row-major order: feature values for t=0, then t=1, etc.
    pub data: Vec<f32>,

    /// Original sequence length (before any padding).
    pub seq_len: usize,

    /// Number of features per time step.
    pub n_features: usize,

    /// Optional regime label for conditional training.
    pub regime: Option<RegimeLabel>,

    /// Optional symbol/asset identifier.
    pub symbol: Option<String>,

    /// Optional metadata.
    pub metadata: HashMap<String, String>,
}

impl TimeSeriesSample {
    /// Create a new sample from flat data.
    pub fn new(data: Vec<f32>, seq_len: usize, n_features: usize) -> Self {
        assert_eq!(
            data.len(),
            seq_len * n_features,
            "Data length {} != seq_len {} * n_features {}",
            data.len(),
            seq_len,
            n_features
        );
        Self {
            data,
            seq_len,
            n_features,
            regime: None,
            symbol: None,
            metadata: HashMap::new(),
        }
    }

    /// Create from a 2-D slice: outer = time steps, inner = features.
    pub fn from_2d(series: &[Vec<f32>]) -> Self {
        let seq_len = series.len();
        let n_features = if seq_len > 0 { series[0].len() } else { 0 };
        let data: Vec<f32> = series.iter().flatten().copied().collect();
        Self::new(data, seq_len, n_features)
    }

    /// Builder: attach a regime label.
    pub fn with_regime(mut self, regime: RegimeLabel) -> Self {
        self.regime = Some(regime);
        self
    }

    /// Builder: attach a symbol.
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    /// Builder: attach metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get value at (time_step, feature_index).
    pub fn get(&self, t: usize, f: usize) -> f32 {
        self.data[t * self.n_features + f]
    }

    /// Get a slice for a single time step.
    pub fn time_step(&self, t: usize) -> &[f32] {
        let start = t * self.n_features;
        &self.data[start..start + self.n_features]
    }

    /// Convert to 2-D: Vec of time steps, each a Vec of features.
    pub fn to_2d(&self) -> Vec<Vec<f32>> {
        self.data
            .chunks(self.n_features)
            .map(|c| c.to_vec())
            .collect()
    }

    /// Compute per-feature mean across time steps.
    pub fn feature_means(&self) -> Vec<f32> {
        let mut means = vec![0.0f32; self.n_features];
        for t in 0..self.seq_len {
            for f in 0..self.n_features {
                means[f] += self.get(t, f);
            }
        }
        for m in &mut means {
            *m /= self.seq_len as f32;
        }
        means
    }

    /// Compute per-feature standard deviation across time steps.
    pub fn feature_stds(&self) -> Vec<f32> {
        let means = self.feature_means();
        let mut vars = vec![0.0f32; self.n_features];
        for t in 0..self.seq_len {
            for f in 0..self.n_features {
                let diff = self.get(t, f) - means[f];
                vars[f] += diff * diff;
            }
        }
        vars.iter()
            .map(|v| (v / self.seq_len as f32).sqrt())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Noise Schedule
// ---------------------------------------------------------------------------

/// Precomputed noise schedule values for efficient forward/reverse passes.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    /// Beta values: β_t for t = 0..T
    pub betas: Vec<f64>,

    /// Alpha values: α_t = 1 - β_t
    pub alphas: Vec<f64>,

    /// Cumulative alpha products: ᾱ_t = Π_{s=0}^{t} α_s
    pub alphas_cumprod: Vec<f64>,

    /// ᾱ_{t-1} (shifted cumulative products, with ᾱ_{-1} = 1)
    pub alphas_cumprod_prev: Vec<f64>,

    /// √(ᾱ_t)
    pub sqrt_alphas_cumprod: Vec<f64>,

    /// √(1 - ᾱ_t)
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,

    /// 1 / √(α_t)
    pub sqrt_recip_alphas: Vec<f64>,

    /// Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
    pub posterior_variance: Vec<f64>,

    /// log(posterior_variance), clamped for stability
    pub posterior_log_variance_clipped: Vec<f64>,

    /// Posterior mean coefficient 1: β_t * √(ᾱ_{t-1}) / (1 - ᾱ_t)
    pub posterior_mean_coef1: Vec<f64>,

    /// Posterior mean coefficient 2: (1 - ᾱ_{t-1}) * √(α_t) / (1 - ᾱ_t)
    pub posterior_mean_coef2: Vec<f64>,

    /// Number of timesteps
    pub n_timesteps: usize,
}

impl NoiseSchedule {
    /// Build a noise schedule from configuration.
    pub fn new(config: &DiffusionConfig) -> Self {
        let n = config.n_timesteps;
        let betas = match config.schedule {
            NoiseScheduleType::Linear => {
                Self::linear_schedule(n, config.beta_start, config.beta_end)
            }
            NoiseScheduleType::Cosine => Self::cosine_schedule(n),
            NoiseScheduleType::Quadratic => {
                Self::quadratic_schedule(n, config.beta_start, config.beta_end)
            }
        };

        Self::from_betas(betas)
    }

    /// Build from raw beta values.
    pub fn from_betas(betas: Vec<f64>) -> Self {
        let n = betas.len();
        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut prod = 1.0;
        for &a in &alphas {
            prod *= a;
            alphas_cumprod.push(prod);
        }

        let mut alphas_cumprod_prev = vec![1.0];
        alphas_cumprod_prev.extend_from_slice(&alphas_cumprod[..n - 1]);

        let sqrt_alphas_cumprod: Vec<f64> = alphas_cumprod.iter().map(|a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f64> =
            alphas_cumprod.iter().map(|a| (1.0 - a).sqrt()).collect();
        let sqrt_recip_alphas: Vec<f64> = alphas.iter().map(|a| (1.0 / a).sqrt()).collect();

        let posterior_variance: Vec<f64> = (0..n)
            .map(|t| {
                betas[t] * (1.0 - alphas_cumprod_prev[t]) / (1.0 - alphas_cumprod[t]).max(1e-20)
            })
            .collect();

        let posterior_log_variance_clipped: Vec<f64> = posterior_variance
            .iter()
            .map(|v| v.max(1e-20).ln())
            .collect();

        let posterior_mean_coef1: Vec<f64> = (0..n)
            .map(|t| {
                betas[t] * alphas_cumprod_prev[t].sqrt() / (1.0 - alphas_cumprod[t]).max(1e-20)
            })
            .collect();

        let posterior_mean_coef2: Vec<f64> = (0..n)
            .map(|t| {
                (1.0 - alphas_cumprod_prev[t]) * alphas[t].sqrt()
                    / (1.0 - alphas_cumprod[t]).max(1e-20)
            })
            .collect();

        Self {
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
            n_timesteps: n,
        }
    }

    /// Linear beta schedule.
    fn linear_schedule(n: usize, beta_start: f64, beta_end: f64) -> Vec<f64> {
        (0..n)
            .map(|i| beta_start + (beta_end - beta_start) * (i as f64) / ((n - 1).max(1) as f64))
            .collect()
    }

    /// Cosine beta schedule (Nichol & Dhariwal, 2021).
    fn cosine_schedule(n: usize) -> Vec<f64> {
        let s = 0.008; // small offset to prevent β_0 = 0
        let f = |t: f64| -> f64 {
            let angle = (t + s) / (1.0 + s) * std::f64::consts::FRAC_PI_2;
            angle.cos().powi(2)
        };

        let mut betas = Vec::with_capacity(n);
        for i in 0..n {
            let t1 = i as f64 / n as f64;
            let t2 = (i + 1) as f64 / n as f64;
            let beta = 1.0 - f(t2) / f(t1);
            betas.push(beta.clamp(0.0001, 0.9999));
        }
        betas
    }

    /// Quadratic beta schedule.
    fn quadratic_schedule(n: usize, beta_start: f64, beta_end: f64) -> Vec<f64> {
        let sqrt_start = beta_start.sqrt();
        let sqrt_end = beta_end.sqrt();
        (0..n)
            .map(|i| {
                let frac = i as f64 / (n - 1).max(1) as f64;
                let s = sqrt_start + frac * (sqrt_end - sqrt_start);
                s * s
            })
            .collect()
    }

    /// Sample noisy data at timestep t: x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
    ///
    /// Returns (x_t, noise) as Tensors.
    pub fn q_sample(&self, x_0: &Tensor, t: usize, noise: &Tensor) -> CandleResult<Tensor> {
        let device = x_0.device();
        let sqrt_alpha = Tensor::new(self.sqrt_alphas_cumprod[t] as f32, device)?;
        let sqrt_one_minus = Tensor::new(self.sqrt_one_minus_alphas_cumprod[t] as f32, device)?;

        let signal = x_0.broadcast_mul(&sqrt_alpha)?;
        let noise_part = noise.broadcast_mul(&sqrt_one_minus)?;
        signal.add(&noise_part)
    }

    /// Compute the posterior mean: μ̃_t(x_t, x_0) = coef1 * x_0 + coef2 * x_t
    pub fn posterior_mean(&self, x_0: &Tensor, x_t: &Tensor, t: usize) -> CandleResult<Tensor> {
        let device = x_0.device();
        let coef1 = Tensor::new(self.posterior_mean_coef1[t] as f32, device)?;
        let coef2 = Tensor::new(self.posterior_mean_coef2[t] as f32, device)?;

        let term1 = x_0.broadcast_mul(&coef1)?;
        let term2 = x_t.broadcast_mul(&coef2)?;
        term1.add(&term2)
    }

    /// Signal-to-noise ratio at timestep t.
    pub fn snr(&self, t: usize) -> f64 {
        self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t]).max(1e-20)
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal Timestep Embedding
// ---------------------------------------------------------------------------

/// Create sinusoidal positional embedding for timestep t.
///
/// Returns a tensor of shape [embed_dim] with alternating sin/cos values.
#[allow(dead_code)]
fn timestep_embedding(t: usize, embed_dim: usize, device: &Device) -> CandleResult<Tensor> {
    let half = embed_dim / 2;
    let mut embedding = vec![0.0f32; embed_dim];

    for i in 0..half {
        let freq = (-((i as f64) * (10000.0f64).ln() / half as f64)).exp();
        let angle = t as f64 * freq;
        embedding[i] = angle.sin() as f32;
        embedding[i + half] = angle.cos() as f32;
    }

    Tensor::from_vec(embedding, (embed_dim,), device)
}

/// Create a batch of timestep embeddings.
fn batch_timestep_embeddings(
    timesteps: &[usize],
    embed_dim: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    let n = timesteps.len();
    let mut flat = Vec::with_capacity(n * embed_dim);
    for &t in timesteps {
        let half = embed_dim / 2;
        for i in 0..half {
            let freq = (-((i as f64) * (10000.0f64).ln() / half as f64)).exp();
            let angle = t as f64 * freq;
            flat.push(angle.sin() as f32);
        }
        for i in 0..half {
            let freq = (-((i as f64) * (10000.0f64).ln() / half as f64)).exp();
            let angle = t as f64 * freq;
            flat.push(angle.cos() as f32);
        }
    }
    Tensor::from_vec(flat, (n, embed_dim), device)
}

// ---------------------------------------------------------------------------
// Denoising Network
// ---------------------------------------------------------------------------

/// MLP-based denoising network with optional regime conditioning.
///
/// Takes as input:
/// - Noisy sample x_t (flattened: seq_len * n_features)
/// - Timestep embedding (sinusoidal)
/// - Optional regime embedding
///
/// Outputs predicted noise ε_θ (same shape as x_t).
struct DenoisingNetwork {
    /// Input projection: maps concatenated (x_t, time_embed, regime_embed) to hidden dim.
    input_proj: Linear,

    /// Hidden layers.
    hidden_layers: Vec<Linear>,

    /// Optional skip-connection projection layers (for residual connections).
    skip_projs: Vec<Option<Linear>>,

    /// Output projection: maps hidden dim back to flat_dim.
    output_proj: Linear,

    /// Regime embedding layer (lookup table).
    regime_embed: Option<Linear>,

    /// Time embedding projection (maps sinusoidal embedding to time_embed_dim).
    time_proj: Linear,

    /// Whether to use residual connections.
    use_residual: bool,

    /// Flat dimension of the input/output.
    #[allow(dead_code)]
    flat_dim: usize,
}

impl DenoisingNetwork {
    fn new(config: &DiffusionConfig, vb: VarBuilder<'_>) -> CandleResult<Self> {
        let flat_dim = config.flat_dim();
        let time_embed_dim = config.time_embed_dim;

        // Time embedding projection
        let time_proj = linear(time_embed_dim, time_embed_dim, vb.pp("time_proj"))?;

        // Regime embedding: small linear layer acting as an embedding table
        let regime_embed = if config.is_conditional() {
            Some(linear(
                config.n_regimes,
                config.regime_embed_dim,
                vb.pp("regime_embed"),
            )?)
        } else {
            None
        };

        // Input dimension: flat_dim + time_embed_dim + optional regime_embed_dim
        let cond_dim = if config.is_conditional() {
            config.regime_embed_dim
        } else {
            0
        };
        let input_dim = flat_dim + time_embed_dim + cond_dim;

        // Hidden layers
        let mut hidden_layers = Vec::new();
        let mut skip_projs = Vec::new();
        let first_hidden = *config.hidden_dims.first().unwrap_or(&256);

        // Input projection
        let input_proj = linear(input_dim, first_hidden, vb.pp("input_proj"))?;
        let mut prev_dim = first_hidden;

        for (i, &hidden_dim) in config.hidden_dims.iter().enumerate() {
            let layer = linear(prev_dim, hidden_dim, vb.pp(format!("hidden_{}", i)))?;
            hidden_layers.push(layer);

            // Skip/residual projection if dimensions change
            let skip = if config.use_residual && prev_dim != hidden_dim {
                Some(linear(prev_dim, hidden_dim, vb.pp(format!("skip_{}", i)))?)
            } else {
                None
            };
            skip_projs.push(skip);

            prev_dim = hidden_dim;
        }

        // Output projection
        let output_proj = linear(prev_dim, flat_dim, vb.pp("output_proj"))?;

        Ok(Self {
            input_proj,
            hidden_layers,
            skip_projs,
            output_proj,
            regime_embed,
            time_proj,
            use_residual: config.use_residual,
            flat_dim,
        })
    }

    /// Forward pass: predict noise given noisy input, timestep, and optional regime.
    ///
    /// # Arguments
    /// - `x_t`: [batch, flat_dim] — noisy sample
    /// - `t_embed`: [batch, time_embed_dim] — sinusoidal timestep embedding
    /// - `regime_onehot`: Optional [batch, n_regimes] — one-hot regime encoding
    fn forward(
        &self,
        x_t: &Tensor,
        t_embed: &Tensor,
        regime_onehot: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Project timestep embedding
        let t_proj = self.time_proj.forward(t_embed)?.relu()?;

        // Build conditioning vector: [x_t, t_proj, regime_embed?]
        let mut parts = vec![x_t.clone(), t_proj];

        if let (Some(embed_layer), Some(regime)) = (&self.regime_embed, regime_onehot) {
            let r_embed = embed_layer.forward(regime)?.relu()?;
            parts.push(r_embed);
        }

        // Concatenate along feature dimension
        let conditioned = Tensor::cat(&parts, 1)?;

        // Input projection
        let mut h = self.input_proj.forward(&conditioned)?.relu()?;

        // Hidden layers with optional residual connections
        for (i, layer) in self.hidden_layers.iter().enumerate() {
            let h_new = layer.forward(&h)?;

            if self.use_residual {
                // Apply residual: h_new + skip(h)
                let skip = if let Some(ref proj) = self.skip_projs[i] {
                    proj.forward(&h)?
                } else {
                    h.clone()
                };
                // Only apply residual if dimensions match (skip_proj handles mismatches)
                let can_add = h_new.dims() == skip.dims();
                h = if can_add {
                    h_new.add(&skip)?.relu()?
                } else {
                    h_new.relu()?
                };
            } else {
                h = h_new.relu()?;
            }
        }

        // Output projection (no activation — predicting noise)
        self.output_proj.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Training Statistics
// ---------------------------------------------------------------------------

/// Statistics from a diffusion model training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionTrainingStats {
    /// Number of epochs completed.
    pub epochs_completed: usize,

    /// Final training loss (MSE of noise prediction).
    pub final_loss: f32,

    /// Loss history (per epoch).
    pub loss_history: Vec<f32>,

    /// Total training wall-clock time.
    pub training_duration: Duration,

    /// Average time per epoch.
    pub avg_epoch_time: Duration,

    /// Number of training samples.
    pub n_samples: usize,

    /// Number of diffusion timesteps.
    pub n_timesteps: usize,

    /// Per-regime sample counts in training data.
    pub regime_counts: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Generated Output
// ---------------------------------------------------------------------------

/// A batch of generated synthetic time series samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedBatch {
    /// Generated samples.
    pub samples: Vec<TimeSeriesSample>,

    /// The regime condition used for generation (if any).
    pub regime: Option<RegimeLabel>,

    /// Number of reverse diffusion steps used.
    pub n_steps: usize,

    /// Wall-clock time for generation.
    pub generation_time: Duration,
}

impl GeneratedBatch {
    /// Compute per-feature statistics across all generated samples.
    pub fn feature_statistics(&self) -> Option<FeatureStatistics> {
        if self.samples.is_empty() {
            return None;
        }

        let n_features = self.samples[0].n_features;
        let n = self.samples.len();

        let mut means = vec![0.0f32; n_features];
        let mut all_means: Vec<Vec<f32>> = Vec::with_capacity(n);

        for sample in &self.samples {
            let m = sample.feature_means();
            for (j, &v) in m.iter().enumerate() {
                means[j] += v;
            }
            all_means.push(m);
        }

        for m in &mut means {
            *m /= n as f32;
        }

        let mut stds = vec![0.0f32; n_features];
        for m in &all_means {
            for (j, &v) in m.iter().enumerate() {
                let diff = v - means[j];
                stds[j] += diff * diff;
            }
        }
        for s in &mut stds {
            *s = (*s / n as f32).sqrt();
        }

        Some(FeatureStatistics {
            n_samples: n,
            n_features,
            feature_means: means,
            feature_stds: stds,
        })
    }
}

/// Per-feature statistics across a batch of generated samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    pub n_samples: usize,
    pub n_features: usize,
    pub feature_means: Vec<f32>,
    pub feature_stds: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Data Normalization
// ---------------------------------------------------------------------------

/// Per-feature normalization statistics computed from training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    /// Per-feature mean (across all samples and time steps).
    pub means: Vec<f64>,
    /// Per-feature standard deviation.
    pub stds: Vec<f64>,
    /// Number of features.
    pub n_features: usize,
}

impl NormalizationStats {
    /// Compute from a set of training samples.
    pub fn compute(samples: &[TimeSeriesSample]) -> Self {
        if samples.is_empty() {
            return Self {
                means: vec![],
                stds: vec![],
                n_features: 0,
            };
        }

        let n_features = samples[0].n_features;
        let mut sums = vec![0.0f64; n_features];
        let mut sq_sums = vec![0.0f64; n_features];
        let mut count = 0u64;

        for sample in samples {
            for t in 0..sample.seq_len {
                for f in 0..n_features {
                    let v = sample.get(t, f) as f64;
                    sums[f] += v;
                    sq_sums[f] += v * v;
                }
                count += 1;
            }
        }

        let means: Vec<f64> = sums.iter().map(|s| s / count as f64).collect();
        let stds: Vec<f64> = sq_sums
            .iter()
            .zip(means.iter())
            .map(|(&sq, &m)| ((sq / count as f64) - m * m).max(0.0).sqrt().max(1e-8))
            .collect();

        Self {
            means,
            stds,
            n_features,
        }
    }

    /// Normalize a sample in-place.
    pub fn normalize(&self, sample: &mut TimeSeriesSample) {
        for t in 0..sample.seq_len {
            for f in 0..self.n_features {
                let idx = t * sample.n_features + f;
                sample.data[idx] =
                    ((sample.data[idx] as f64 - self.means[f]) / self.stds[f]) as f32;
            }
        }
    }

    /// Denormalize a sample in-place.
    pub fn denormalize(&self, sample: &mut TimeSeriesSample) {
        for t in 0..sample.seq_len {
            for f in 0..self.n_features {
                let idx = t * sample.n_features + f;
                sample.data[idx] = (sample.data[idx] as f64 * self.stds[f] + self.means[f]) as f32;
            }
        }
    }

    /// Denormalize a raw flat vector.
    pub fn denormalize_flat(&self, data: &mut [f32], n_features: usize) {
        for (i, v) in data.iter_mut().enumerate() {
            let f = i % n_features;
            *v = (*v as f64 * self.stds[f] + self.means[f]) as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// TimeSeriesDDPM — Main Struct
// ---------------------------------------------------------------------------

/// State of the diffusion model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DdpmState {
    /// Model created but not trained.
    Initialized,
    /// Model has been trained.
    Trained,
    /// Training failed.
    Failed,
}

/// Time Series Denoising Diffusion Probabilistic Model.
///
/// Trains a parametric denoising network to reverse a noise diffusion
/// process, enabling generation of realistic synthetic financial time series
/// conditioned on market regime labels.
pub struct TimeSeriesDDPM {
    config: DiffusionConfig,
    schedule: NoiseSchedule,
    var_map: VarMap,
    device: Device,
    state: DdpmState,
    norm_stats: Option<NormalizationStats>,
    training_stats: Option<DiffusionTrainingStats>,
}

impl TimeSeriesDDPM {
    /// Create a new diffusion model.
    pub fn new(config: DiffusionConfig) -> Result<Self, DiffusionError> {
        let device = Device::Cpu;
        let schedule = NoiseSchedule::new(&config);
        let var_map = VarMap::new();

        info!(
            "Created TimeSeriesDDPM: seq_len={}, n_features={}, T={}, {} regimes",
            config.seq_len, config.n_features, config.n_timesteps, config.n_regimes
        );

        Ok(Self {
            config,
            schedule,
            var_map,
            device,
            state: DdpmState::Initialized,
            norm_stats: None,
            training_stats: None,
        })
    }

    /// Get the current state.
    pub fn state(&self) -> DdpmState {
        self.state
    }

    /// Get the configuration.
    pub fn config(&self) -> &DiffusionConfig {
        &self.config
    }

    /// Get the noise schedule.
    pub fn schedule(&self) -> &NoiseSchedule {
        &self.schedule
    }

    /// Get training statistics (available after training).
    pub fn training_stats(&self) -> Option<&DiffusionTrainingStats> {
        self.training_stats.as_ref()
    }

    /// Get normalization statistics (available after training).
    pub fn norm_stats(&self) -> Option<&NormalizationStats> {
        self.norm_stats.as_ref()
    }

    /// Train the diffusion model on a set of time series samples.
    ///
    /// # Arguments
    /// * `samples` - Training data. Each sample should have the same
    ///   seq_len and n_features as the config.
    pub fn train(
        &mut self,
        samples: &[TimeSeriesSample],
    ) -> Result<DiffusionTrainingStats, DiffusionError> {
        if samples.is_empty() {
            return Err(DiffusionError::InsufficientData { got: 0, min: 1 });
        }

        // Validate dimensions
        for (i, s) in samples.iter().enumerate() {
            if s.seq_len != self.config.seq_len || s.n_features != self.config.n_features {
                return Err(DiffusionError::DimensionMismatch {
                    expected_seq: self.config.seq_len,
                    expected_feat: self.config.n_features,
                    got_seq: s.seq_len,
                    got_feat: s.n_features,
                    index: i,
                });
            }
        }

        let start = Instant::now();
        let n = samples.len();
        let flat_dim = self.config.flat_dim();

        info!(
            "Starting DDPM training: {} samples, flat_dim={}, T={}",
            n, flat_dim, self.config.n_timesteps
        );

        // Compute normalization statistics
        let norm_stats = NormalizationStats::compute(samples);

        // Normalize training data
        let mut normalized_samples: Vec<TimeSeriesSample> = samples.to_vec();
        for s in &mut normalized_samples {
            norm_stats.normalize(s);
        }

        // Count regime occurrences
        let mut regime_counts: HashMap<String, usize> = HashMap::new();
        for s in samples {
            if let Some(ref regime) = s.regime {
                *regime_counts.entry(regime.to_string()).or_insert(0) += 1;
            }
        }

        // Pre-load all normalized data into a tensor [n, flat_dim]
        let flat_data: Vec<f32> = normalized_samples
            .iter()
            .flat_map(|s| s.data.iter())
            .copied()
            .collect();
        let data_tensor = Tensor::from_vec(flat_data, (n, flat_dim), &self.device)
            .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

        // Initialize fresh VarMap and network
        self.var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&self.var_map, DType::F32, &self.device);
        let network = DenoisingNetwork::new(&self.config, vb)
            .map_err(|e| DiffusionError::ModelError(e.to_string()))?;

        // Create optimizer
        let params = candle_nn::ParamsAdamW {
            lr: self.config.learning_rate,
            weight_decay: self.config.weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let mut optimizer = candle_nn::AdamW::new(self.var_map.all_vars(), params)
            .map_err(|e| DiffusionError::OptimizerError(e.to_string()))?;

        // Prepare regime one-hot encodings if conditional
        let regime_onehots: Option<Vec<Vec<f32>>> = if self.config.is_conditional() {
            Some(
                normalized_samples
                    .iter()
                    .map(|s| {
                        let mut oh = vec![0.0f32; self.config.n_regimes];
                        if let Some(ref regime) = s.regime {
                            let idx = regime.to_index();
                            if idx < self.config.n_regimes {
                                oh[idx] = 1.0;
                            }
                        }
                        oh
                    })
                    .collect(),
            )
        } else {
            None
        };

        // Training loop
        let mut loss_history = Vec::with_capacity(self.config.n_epochs);
        let mut rng_state: u64 = self.config.seed;

        for epoch in 0..self.config.n_epochs {
            let mut epoch_loss = 0.0f32;
            let mut n_batches = 0u32;

            // Shuffle indices
            let mut indices: Vec<usize> = (0..n).collect();
            // Fisher-Yates shuffle with simple PRNG
            for i in (1..indices.len()).rev() {
                rng_state = lcg_next(rng_state);
                let j = (rng_state as usize) % (i + 1);
                indices.swap(i, j);
            }

            for batch_start in (0..n).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(n);
                let batch_indices = &indices[batch_start..batch_end];
                let bs = batch_indices.len();

                // Gather batch data
                let idx_vec: Vec<u32> = batch_indices.iter().map(|&i| i as u32).collect();
                let idx_tensor = Tensor::from_vec(idx_vec, (bs,), &self.device)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                let x_0 = data_tensor
                    .index_select(&idx_tensor, 0)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                // Sample random timesteps for each item in the batch
                let timesteps: Vec<usize> = (0..bs)
                    .map(|_i| {
                        rng_state = lcg_next(rng_state);
                        (rng_state as usize) % self.config.n_timesteps
                    })
                    .collect();

                // Create timestep embeddings
                let t_embed =
                    batch_timestep_embeddings(&timesteps, self.config.time_embed_dim, &self.device)
                        .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                // Sample noise ε ~ N(0, I)
                let noise = Tensor::randn(0.0f32, 1.0f32, (bs, flat_dim), &self.device)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                // Compute x_t for each sample at its respective timestep
                // For simplicity, we use the mean timestep for the batch
                // (a more precise implementation would handle per-sample timesteps)
                let mut x_t_vecs = Vec::with_capacity(bs * flat_dim);
                let x_0_data: Vec<f32> = x_0
                    .flatten_all()
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?
                    .to_vec1::<f32>()
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;
                let noise_data: Vec<f32> = noise
                    .flatten_all()
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?
                    .to_vec1::<f32>()
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                for (sample_idx, &t) in timesteps.iter().enumerate() {
                    let sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t] as f32;
                    let sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[t] as f32;

                    for feat_idx in 0..flat_dim {
                        let flat_idx = sample_idx * flat_dim + feat_idx;
                        let x0_val = x_0_data[flat_idx];
                        let noise_val = noise_data[flat_idx];
                        x_t_vecs.push(sqrt_alpha * x0_val + sqrt_one_minus * noise_val);
                    }
                }

                let x_t = Tensor::from_vec(x_t_vecs, (bs, flat_dim), &self.device)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                // Regime conditioning
                let regime_tensor: Option<Tensor> = if let Some(ref onehots) = regime_onehots {
                    let mut batch_oh = Vec::with_capacity(bs * self.config.n_regimes);
                    for &idx in batch_indices {
                        batch_oh.extend_from_slice(&onehots[idx]);
                    }
                    Some(
                        Tensor::from_vec(batch_oh, (bs, self.config.n_regimes), &self.device)
                            .map_err(|e| DiffusionError::TensorError(e.to_string()))?,
                    )
                } else {
                    None
                };

                // Forward pass: predict noise
                let predicted_noise = network
                    .forward(&x_t, &t_embed, regime_tensor.as_ref())
                    .map_err(|e| DiffusionError::ModelError(e.to_string()))?;

                // MSE loss: ||ε - ε_θ(x_t, t)||²
                let diff = predicted_noise
                    .sub(&noise)
                    .map_err(|e| DiffusionError::LossError(e.to_string()))?;
                let loss = diff
                    .sqr()
                    .map_err(|e| DiffusionError::LossError(e.to_string()))?
                    .mean_all()
                    .map_err(|e| DiffusionError::LossError(e.to_string()))?;

                // Backward pass and optimizer step
                optimizer
                    .backward_step(&loss)
                    .map_err(|e| DiffusionError::OptimizerError(e.to_string()))?;

                let loss_val = loss
                    .to_scalar::<f32>()
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;
                epoch_loss += loss_val;
                n_batches += 1;
            }

            let avg_loss = if n_batches > 0 {
                epoch_loss / n_batches as f32
            } else {
                0.0
            };
            loss_history.push(avg_loss);

            if (epoch + 1) % self.config.log_interval == 0 || epoch == 0 {
                info!(
                    "Epoch {}/{}: loss={:.6}",
                    epoch + 1,
                    self.config.n_epochs,
                    avg_loss
                );
            }
        }

        let duration = start.elapsed();
        let avg_epoch = duration / self.config.n_epochs.max(1) as u32;

        let stats = DiffusionTrainingStats {
            epochs_completed: self.config.n_epochs,
            final_loss: *loss_history.last().unwrap_or(&0.0),
            loss_history,
            training_duration: duration,
            avg_epoch_time: avg_epoch,
            n_samples: n,
            n_timesteps: self.config.n_timesteps,
            regime_counts,
        };

        self.norm_stats = Some(norm_stats);
        self.training_stats = Some(stats.clone());
        self.state = DdpmState::Trained;

        info!(
            "DDPM training complete: final_loss={:.6}, duration={:.1}s",
            stats.final_loss,
            stats.training_duration.as_secs_f64()
        );

        Ok(stats)
    }

    /// Generate synthetic time series samples via the reverse diffusion process.
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples to generate.
    /// * `regime` - Optional regime label for conditional generation.
    ///
    /// # Returns
    /// A batch of synthetic time series samples (denormalized to original scale).
    pub fn sample(
        &self,
        n_samples: usize,
        regime: Option<RegimeLabel>,
    ) -> Result<GeneratedBatch, DiffusionError> {
        if self.state != DdpmState::Trained {
            return Err(DiffusionError::NotTrained);
        }

        if n_samples == 0 {
            return Ok(GeneratedBatch {
                samples: vec![],
                regime,
                n_steps: 0,
                generation_time: Duration::ZERO,
            });
        }

        let start = Instant::now();
        let flat_dim = self.config.flat_dim();

        // Rebuild network from VarMap
        let vb = VarBuilder::from_varmap(&self.var_map, DType::F32, &self.device);
        let network = DenoisingNetwork::new(&self.config, vb)
            .map_err(|e| DiffusionError::ModelError(e.to_string()))?;

        // Prepare regime conditioning
        let regime_tensor: Option<Tensor> = if self.config.is_conditional() {
            if let Some(ref r) = regime {
                let idx = r.to_index();
                let mut oh = vec![0.0f32; self.config.n_regimes * n_samples];
                for i in 0..n_samples {
                    if idx < self.config.n_regimes {
                        oh[i * self.config.n_regimes + idx] = 1.0;
                    }
                }
                Some(
                    Tensor::from_vec(oh, (n_samples, self.config.n_regimes), &self.device)
                        .map_err(|e| DiffusionError::TensorError(e.to_string()))?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // Start from pure noise: x_T ~ N(0, I)
        let mut x_t = Tensor::randn(0.0f32, 1.0f32, (n_samples, flat_dim), &self.device)
            .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

        // Reverse diffusion: t = T-1, T-2, ..., 0
        for t in (0..self.config.n_timesteps).rev() {
            // Create timestep embeddings (same t for all samples in batch)
            let timesteps = vec![t; n_samples];
            let t_embed =
                batch_timestep_embeddings(&timesteps, self.config.time_embed_dim, &self.device)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

            // Predict noise
            let predicted_noise = network
                .forward(&x_t, &t_embed, regime_tensor.as_ref())
                .map_err(|e| DiffusionError::ModelError(e.to_string()))?;

            // Compute x_{t-1} using the DDPM update rule:
            // x_{t-1} = 1/√(α_t) * (x_t - β_t/√(1-ᾱ_t) * ε_θ) + σ_t * z
            let sqrt_recip_alpha =
                Tensor::new(self.schedule.sqrt_recip_alphas[t] as f32, &self.device)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

            let beta = Tensor::new(self.schedule.betas[t] as f32, &self.device)
                .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

            let sqrt_one_minus_alpha_cumprod = Tensor::new(
                self.schedule.sqrt_one_minus_alphas_cumprod[t] as f32,
                &self.device,
            )
            .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

            // noise_coeff = β_t / √(1 - ᾱ_t)
            let noise_coeff = beta
                .broadcast_div(&sqrt_one_minus_alpha_cumprod)
                .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

            // mean = 1/√(α_t) * (x_t - noise_coeff * ε_θ)
            let scaled_noise = predicted_noise
                .broadcast_mul(&noise_coeff)
                .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

            let mean = x_t
                .sub(&scaled_noise)
                .map_err(|e| DiffusionError::TensorError(e.to_string()))?
                .broadcast_mul(&sqrt_recip_alpha)
                .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

            if t > 0 {
                // Add noise: x_{t-1} = mean + σ_t * z
                let sigma = match self.config.variance_type {
                    VarianceType::FixedSmall => self.schedule.betas[t].sqrt() as f32,
                    VarianceType::FixedLarge => self.schedule.posterior_variance[t].sqrt() as f32,
                    VarianceType::Learned => self.schedule.betas[t].sqrt() as f32,
                };

                let z = Tensor::randn(0.0f32, 1.0f32, (n_samples, flat_dim), &self.device)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                let sigma_tensor = Tensor::new(sigma, &self.device)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                let noise_term = z
                    .broadcast_mul(&sigma_tensor)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

                x_t = mean
                    .add(&noise_term)
                    .map_err(|e| DiffusionError::TensorError(e.to_string()))?;
            } else {
                // At t=0, no noise is added
                x_t = mean;
            }
        }

        // Extract generated data
        let generated_data: Vec<f32> = x_t
            .flatten_all()
            .map_err(|e| DiffusionError::TensorError(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| DiffusionError::TensorError(e.to_string()))?;

        // Create samples and denormalize
        let mut samples = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let start_idx = i * flat_dim;
            let end_idx = start_idx + flat_dim;
            let mut data = generated_data[start_idx..end_idx].to_vec();

            // Denormalize if normalization stats are available
            if let Some(ref norm) = self.norm_stats {
                norm.denormalize_flat(&mut data, self.config.n_features);
            }

            let mut sample =
                TimeSeriesSample::new(data, self.config.seq_len, self.config.n_features);
            if let Some(ref r) = regime {
                sample.regime = Some(*r);
            }
            samples.push(sample);
        }

        let generation_time = start.elapsed();

        info!(
            "Generated {} samples in {:.2}s (regime: {:?})",
            n_samples,
            generation_time.as_secs_f64(),
            regime
        );

        Ok(GeneratedBatch {
            samples,
            regime,
            n_steps: self.config.n_timesteps,
            generation_time,
        })
    }

    /// Save model weights.
    pub fn save_weights(&self, path: &str) -> Result<(), DiffusionError> {
        self.var_map
            .save(path)
            .map_err(|e| DiffusionError::IoError(e.to_string()))
    }

    /// Load model weights.
    pub fn load_weights(&mut self, path: &str) -> Result<(), DiffusionError> {
        self.var_map
            .load(path)
            .map_err(|e| DiffusionError::IoError(e.to_string()))?;
        self.state = DdpmState::Trained;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Synthetic Data Generator — High-Level API
// ---------------------------------------------------------------------------

/// High-level API for generating synthetic financial time series.
///
/// Wraps `TimeSeriesDDPM` with convenience methods for common tasks:
/// - Training from raw market data
/// - Generating augmentation data for sparse regimes
/// - Quality assessment of generated samples
pub struct SyntheticDataGenerator {
    ddpm: TimeSeriesDDPM,
    total_generated: u64,
}

impl SyntheticDataGenerator {
    /// Create a new generator.
    pub fn new(config: DiffusionConfig) -> Result<Self, DiffusionError> {
        let ddpm = TimeSeriesDDPM::new(config)?;
        Ok(Self {
            ddpm,
            total_generated: 0,
        })
    }

    /// Train the generator on a dataset.
    pub fn train(
        &mut self,
        samples: &[TimeSeriesSample],
    ) -> Result<DiffusionTrainingStats, DiffusionError> {
        self.ddpm.train(samples)
    }

    /// Generate unconditional samples.
    pub fn generate(&mut self, n: usize) -> Result<GeneratedBatch, DiffusionError> {
        let batch = self.ddpm.sample(n, None)?;
        self.total_generated += n as u64;
        Ok(batch)
    }

    /// Generate samples conditioned on a specific regime.
    pub fn generate_for_regime(
        &mut self,
        n: usize,
        regime: RegimeLabel,
    ) -> Result<GeneratedBatch, DiffusionError> {
        let batch = self.ddpm.sample(n, Some(regime))?;
        self.total_generated += n as u64;
        Ok(batch)
    }

    /// Augment a dataset by generating additional samples for under-represented regimes.
    ///
    /// Counts the number of samples per regime in the input data, then generates
    /// enough synthetic samples to bring all regimes up to `target_per_regime`.
    pub fn augment_sparse_regimes(
        &mut self,
        existing: &[TimeSeriesSample],
        target_per_regime: usize,
    ) -> Result<Vec<GeneratedBatch>, DiffusionError> {
        // Count existing samples per regime
        let mut counts: HashMap<RegimeLabel, usize> = HashMap::new();
        for s in existing {
            if let Some(regime) = s.regime {
                *counts.entry(regime).or_insert(0) += 1;
            }
        }

        let mut batches = Vec::new();

        // For each known regime, check if augmentation is needed
        let all_regimes = vec![
            RegimeLabel::Trending,
            RegimeLabel::MeanReverting,
            RegimeLabel::Volatile,
            RegimeLabel::Uncertain,
        ];

        for regime in all_regimes {
            let current = counts.get(&regime).copied().unwrap_or(0);
            if current < target_per_regime {
                let needed = target_per_regime - current;
                info!(
                    "Regime {:?}: have {}, need {}, generating {} synthetic samples",
                    regime, current, target_per_regime, needed
                );
                let batch = self.generate_for_regime(needed, regime)?;
                batches.push(batch);
            } else {
                debug!(
                    "Regime {:?}: have {} (>= target {}), no augmentation needed",
                    regime, current, target_per_regime
                );
            }
        }

        Ok(batches)
    }

    /// Compute basic quality metrics between real and synthetic samples.
    pub fn quality_assessment(
        &self,
        real: &[TimeSeriesSample],
        synthetic: &[TimeSeriesSample],
    ) -> QualityReport {
        if real.is_empty() || synthetic.is_empty() {
            return QualityReport::empty();
        }

        let n_features = real[0].n_features;

        // Per-feature mean comparison
        let real_means = Self::global_feature_means(real);
        let synth_means = Self::global_feature_means(synthetic);

        let real_stds = Self::global_feature_stds(real);
        let synth_stds = Self::global_feature_stds(synthetic);

        // Mean absolute error between means
        let mean_mae: f32 = real_means
            .iter()
            .zip(synth_means.iter())
            .map(|(r, s)| (r - s).abs())
            .sum::<f32>()
            / n_features as f32;

        // Mean absolute error between stds
        let std_mae: f32 = real_stds
            .iter()
            .zip(synth_stds.iter())
            .map(|(r, s)| (r - s).abs())
            .sum::<f32>()
            / n_features as f32;

        // Auto-correlation similarity (lag-1)
        let real_ac = Self::avg_autocorrelation(real, 1);
        let synth_ac = Self::avg_autocorrelation(synthetic, 1);
        let ac_diff = (real_ac - synth_ac).abs();

        // Compute a combined quality score (0 = identical, higher = worse)
        let quality_score = 1.0 / (1.0 + mean_mae + std_mae + ac_diff);

        QualityReport {
            mean_mae,
            std_mae,
            autocorrelation_diff: ac_diff,
            quality_score,
            real_feature_means: real_means,
            synthetic_feature_means: synth_means,
            real_feature_stds: real_stds,
            synthetic_feature_stds: synth_stds,
            n_real: real.len(),
            n_synthetic: synthetic.len(),
        }
    }

    /// Compute global per-feature means across all samples.
    fn global_feature_means(samples: &[TimeSeriesSample]) -> Vec<f32> {
        if samples.is_empty() {
            return vec![];
        }
        let nf = samples[0].n_features;
        let mut sums = vec![0.0f64; nf];
        let mut count = 0u64;

        for s in samples {
            for t in 0..s.seq_len {
                for f in 0..nf {
                    sums[f] += s.get(t, f) as f64;
                }
                count += 1;
            }
        }

        sums.iter().map(|s| (*s / count as f64) as f32).collect()
    }

    /// Compute global per-feature stds across all samples.
    fn global_feature_stds(samples: &[TimeSeriesSample]) -> Vec<f32> {
        let means = Self::global_feature_means(samples);
        if samples.is_empty() {
            return vec![];
        }
        let nf = samples[0].n_features;
        let mut sq_sums = vec![0.0f64; nf];
        let mut count = 0u64;

        for s in samples {
            for t in 0..s.seq_len {
                for f in 0..nf {
                    let diff = s.get(t, f) as f64 - means[f] as f64;
                    sq_sums[f] += diff * diff;
                }
                count += 1;
            }
        }

        sq_sums
            .iter()
            .map(|s| ((*s / count as f64).sqrt()) as f32)
            .collect()
    }

    /// Average lag-k autocorrelation across all samples and features.
    fn avg_autocorrelation(samples: &[TimeSeriesSample], lag: usize) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut total_ac = 0.0f64;
        let mut count = 0u64;

        for s in samples {
            if s.seq_len <= lag {
                continue;
            }
            for f in 0..s.n_features {
                let values: Vec<f64> = (0..s.seq_len).map(|t| s.get(t, f) as f64).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;

                let mut cov = 0.0f64;
                let mut var = 0.0f64;
                for t in 0..values.len() {
                    var += (values[t] - mean) * (values[t] - mean);
                    if t >= lag {
                        cov += (values[t] - mean) * (values[t - lag] - mean);
                    }
                }

                if var > 1e-12 {
                    total_ac += cov / var;
                    count += 1;
                }
            }
        }

        if count > 0 {
            (total_ac / count as f64) as f32
        } else {
            0.0
        }
    }

    /// Get the total number of samples generated so far.
    pub fn total_generated(&self) -> u64 {
        self.total_generated
    }

    /// Get the inner DDPM model state.
    pub fn model_state(&self) -> DdpmState {
        self.ddpm.state()
    }
}

/// Quality assessment report comparing real vs. synthetic samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Mean absolute error between real and synthetic per-feature means.
    pub mean_mae: f32,

    /// Mean absolute error between real and synthetic per-feature stds.
    pub std_mae: f32,

    /// Absolute difference in average lag-1 autocorrelation.
    pub autocorrelation_diff: f32,

    /// Combined quality score: 1.0 = perfect match, approaches 0 as quality degrades.
    pub quality_score: f32,

    /// Per-feature means for real data.
    pub real_feature_means: Vec<f32>,

    /// Per-feature means for synthetic data.
    pub synthetic_feature_means: Vec<f32>,

    /// Per-feature stds for real data.
    pub real_feature_stds: Vec<f32>,

    /// Per-feature stds for synthetic data.
    pub synthetic_feature_stds: Vec<f32>,

    /// Number of real samples.
    pub n_real: usize,

    /// Number of synthetic samples.
    pub n_synthetic: usize,
}

impl QualityReport {
    fn empty() -> Self {
        Self {
            mean_mae: f32::NAN,
            std_mae: f32::NAN,
            autocorrelation_diff: f32::NAN,
            quality_score: 0.0,
            real_feature_means: vec![],
            synthetic_feature_means: vec![],
            real_feature_stds: vec![],
            synthetic_feature_stds: vec![],
            n_real: 0,
            n_synthetic: 0,
        }
    }

    /// Whether the synthetic data passes basic quality thresholds.
    pub fn is_acceptable(&self) -> bool {
        self.quality_score > 0.3 && self.mean_mae.is_finite() && self.std_mae.is_finite()
    }
}

// ---------------------------------------------------------------------------
// Simple LCG PRNG (for reproducible shuffling without pulling in `rand`)
// ---------------------------------------------------------------------------

/// Linear congruential generator for deterministic pseudo-random numbers.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors from diffusion model operations.
#[derive(Debug, thiserror::Error)]
pub enum DiffusionError {
    #[error("Insufficient training data: got {got}, need at least {min}")]
    InsufficientData { got: usize, min: usize },

    #[error(
        "Dimension mismatch at index {index}: expected seq={expected_seq} feat={expected_feat}, got seq={got_seq} feat={got_feat}"
    )]
    DimensionMismatch {
        expected_seq: usize,
        expected_feat: usize,
        got_seq: usize,
        got_feat: usize,
        index: usize,
    },

    #[error("Model not trained: call train() before sample()")]
    NotTrained,

    #[error("Tensor error: {0}")]
    TensorError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Loss computation error: {0}")]
    LossError(String),

    #[error("Optimizer error: {0}")]
    OptimizerError(String),

    #[error("I/O error: {0}")]
    IoError(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple synthetic training dataset.
    fn make_training_data(
        n: usize,
        seq_len: usize,
        n_features: usize,
        regime: Option<RegimeLabel>,
    ) -> Vec<TimeSeriesSample> {
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let mut data = Vec::with_capacity(seq_len * n_features);
            let mut state = lcg_next(i as u64 + 42);
            for t in 0..seq_len {
                for f in 0..n_features {
                    state = lcg_next(state);
                    // Generate a simple mean-reverting process
                    let base = match f {
                        0 => 100.0, // "price"
                        1 => 50.0,  // "volume"
                        _ => 0.0,
                    };
                    let noise = ((state % 10000) as f32 / 10000.0 - 0.5) * 10.0;
                    data.push(base + noise + (t as f32) * 0.01);
                }
            }
            let mut sample = TimeSeriesSample::new(data, seq_len, n_features);
            sample.regime = regime;
            samples.push(sample);
        }
        samples
    }

    // ========== Config Tests ==========

    #[test]
    fn test_config_defaults() {
        let config = DiffusionConfig::default();
        assert_eq!(config.seq_len, 64);
        assert_eq!(config.n_features, 5);
        assert_eq!(config.n_timesteps, 200);
        assert_eq!(config.n_regimes, 4);
        assert!(config.is_conditional());
    }

    #[test]
    fn test_config_presets() {
        let ohlcv = DiffusionConfig::ohlcv();
        assert_eq!(ohlcv.seq_len, 64);
        assert_eq!(ohlcv.n_features, 5);

        let returns = DiffusionConfig::returns_only();
        assert_eq!(returns.seq_len, 128);
        assert_eq!(returns.n_features, 1);

        let multi = DiffusionConfig::multi_asset(10);
        assert_eq!(multi.n_features, 10);
        assert_eq!(multi.seq_len, 32);

        let tiny = DiffusionConfig::tiny();
        assert_eq!(tiny.seq_len, 16);
        assert_eq!(tiny.n_features, 2);
        assert_eq!(tiny.n_epochs, 5);
    }

    #[test]
    fn test_config_builders() {
        let config = DiffusionConfig::default()
            .with_seq_len(32)
            .with_n_features(3)
            .with_n_timesteps(100)
            .with_n_regimes(6)
            .with_learning_rate(5e-5)
            .with_n_epochs(50)
            .with_batch_size(16)
            .with_schedule(NoiseScheduleType::Linear)
            .with_hidden_dims(vec![64, 128, 64])
            .with_variance_type(VarianceType::FixedLarge);

        assert_eq!(config.seq_len, 32);
        assert_eq!(config.n_features, 3);
        assert_eq!(config.n_timesteps, 100);
        assert_eq!(config.n_regimes, 6);
        assert_eq!(config.n_epochs, 50);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.flat_dim(), 32 * 3);
    }

    #[test]
    fn test_config_unconditional() {
        let config = DiffusionConfig::default().with_n_regimes(0);
        assert!(!config.is_conditional());
    }

    // ========== Regime Label Tests ==========

    #[test]
    fn test_regime_label_roundtrip() {
        for idx in 0..8 {
            let label = RegimeLabel::from_index(idx);
            assert_eq!(label.to_index(), idx);
        }
    }

    #[test]
    fn test_regime_label_display() {
        assert_eq!(RegimeLabel::Trending.to_string(), "trending");
        assert_eq!(RegimeLabel::Volatile.to_string(), "volatile");
        assert_eq!(RegimeLabel::Custom(5).to_string(), "custom_5");
    }

    #[test]
    fn test_regime_label_names() {
        assert_eq!(RegimeLabel::Trending.label(), "trending");
        assert_eq!(RegimeLabel::MeanReverting.label(), "mean_reverting");
        assert_eq!(RegimeLabel::Volatile.label(), "volatile");
        assert_eq!(RegimeLabel::Uncertain.label(), "uncertain");
        assert_eq!(RegimeLabel::Custom(0).label(), "custom");
    }

    // ========== TimeSeriesSample Tests ==========

    #[test]
    fn test_sample_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sample = TimeSeriesSample::new(data, 3, 2);
        assert_eq!(sample.seq_len, 3);
        assert_eq!(sample.n_features, 2);
        assert_eq!(sample.get(0, 0), 1.0);
        assert_eq!(sample.get(0, 1), 2.0);
        assert_eq!(sample.get(1, 0), 3.0);
        assert_eq!(sample.get(2, 1), 6.0);
    }

    #[test]
    fn test_sample_from_2d() {
        let series = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let sample = TimeSeriesSample::from_2d(&series);
        assert_eq!(sample.seq_len, 3);
        assert_eq!(sample.n_features, 2);
        assert_eq!(sample.get(1, 1), 4.0);
    }

    #[test]
    fn test_sample_to_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sample = TimeSeriesSample::new(data, 3, 2);
        let twod = sample.to_2d();
        assert_eq!(twod.len(), 3);
        assert_eq!(twod[0], vec![1.0, 2.0]);
        assert_eq!(twod[2], vec![5.0, 6.0]);
    }

    #[test]
    fn test_sample_time_step() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sample = TimeSeriesSample::new(data, 3, 2);
        assert_eq!(sample.time_step(0), &[1.0, 2.0]);
        assert_eq!(sample.time_step(1), &[3.0, 4.0]);
    }

    #[test]
    fn test_sample_feature_means() {
        let data = vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0];
        let sample = TimeSeriesSample::new(data, 3, 2);
        let means = sample.feature_means();
        assert!((means[0] - 3.0).abs() < 1e-5);
        assert!((means[1] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_feature_stds() {
        let data = vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0];
        let sample = TimeSeriesSample::new(data, 3, 2);
        let stds = sample.feature_stds();
        assert!(stds[0] > 0.0);
        assert!(stds[1] > 0.0);
    }

    #[test]
    fn test_sample_with_builders() {
        let data = vec![1.0, 2.0];
        let sample = TimeSeriesSample::new(data, 1, 2)
            .with_regime(RegimeLabel::Volatile)
            .with_symbol("BTCUSDT")
            .with_metadata("source", "test");

        assert_eq!(sample.regime, Some(RegimeLabel::Volatile));
        assert_eq!(sample.symbol.as_deref(), Some("BTCUSDT"));
        assert_eq!(sample.metadata.get("source").unwrap(), "test");
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_sample_bad_dimensions() {
        let _sample = TimeSeriesSample::new(vec![1.0, 2.0, 3.0], 2, 2);
    }

    // ========== Noise Schedule Tests ==========

    #[test]
    fn test_linear_schedule() {
        let betas = NoiseSchedule::linear_schedule(10, 1e-4, 0.02);
        assert_eq!(betas.len(), 10);
        assert!((betas[0] - 1e-4).abs() < 1e-8);
        assert!((betas[9] - 0.02).abs() < 1e-8);
        // Should be monotonically increasing
        for i in 1..betas.len() {
            assert!(betas[i] >= betas[i - 1]);
        }
    }

    #[test]
    fn test_cosine_schedule() {
        let betas = NoiseSchedule::cosine_schedule(100);
        assert_eq!(betas.len(), 100);
        for &b in &betas {
            assert!(b > 0.0 && b < 1.0, "beta out of range: {}", b);
        }
    }

    #[test]
    fn test_quadratic_schedule() {
        let betas = NoiseSchedule::quadratic_schedule(50, 1e-4, 0.02);
        assert_eq!(betas.len(), 50);
        for &b in &betas {
            assert!((0.0..=1.0).contains(&b));
        }
    }

    #[test]
    fn test_noise_schedule_properties() {
        let config = DiffusionConfig::default().with_n_timesteps(100);
        let schedule = NoiseSchedule::new(&config);

        assert_eq!(schedule.n_timesteps, 100);
        assert_eq!(schedule.betas.len(), 100);
        assert_eq!(schedule.alphas.len(), 100);
        assert_eq!(schedule.alphas_cumprod.len(), 100);

        // ᾱ_t should be monotonically decreasing
        for i in 1..schedule.alphas_cumprod.len() {
            assert!(
                schedule.alphas_cumprod[i] <= schedule.alphas_cumprod[i - 1],
                "ᾱ not decreasing at t={}: {} > {}",
                i,
                schedule.alphas_cumprod[i],
                schedule.alphas_cumprod[i - 1]
            );
        }

        // ᾱ_0 should be close to 1, ᾱ_T should be close to 0
        assert!(schedule.alphas_cumprod[0] > 0.9);
        assert!(schedule.alphas_cumprod[99] < 0.5);

        // SNR should be monotonically decreasing
        for i in 1..schedule.n_timesteps {
            assert!(schedule.snr(i) <= schedule.snr(i - 1) + 1e-10);
        }
    }

    #[test]
    fn test_noise_schedule_q_sample() {
        let config = DiffusionConfig::default().with_n_timesteps(100);
        let schedule = NoiseSchedule::new(&config);
        let device = Device::Cpu;

        let x_0 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        let noise = Tensor::from_vec(vec![0.1f32, -0.2, 0.3], (1, 3), &device).unwrap();

        // At t=0, x_t should be very close to x_0
        let x_t0 = schedule.q_sample(&x_0, 0, &noise).unwrap();
        let vals: Vec<f32> = x_t0.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 1.0).abs() < 0.2);

        // At t=99 (near T), x_t should be dominated by noise
        let x_t99 = schedule.q_sample(&x_0, 99, &noise).unwrap();
        let vals99: Vec<f32> = x_t99.flatten_all().unwrap().to_vec1().unwrap();
        // The original signal should be significantly attenuated
        let original_norm: f32 = [1.0f32, 2.0, 3.0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let t99_norm: f32 = vals99.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            t99_norm < original_norm,
            "At t=99 the signal should be attenuated"
        );
    }

    #[test]
    fn test_noise_schedule_posterior_mean() {
        let config = DiffusionConfig::default().with_n_timesteps(50);
        let schedule = NoiseSchedule::new(&config);
        let device = Device::Cpu;

        let x_0 = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &device).unwrap();
        let x_t = Tensor::from_vec(vec![0.5f32, 1.5], (1, 2), &device).unwrap();

        let mean = schedule.posterior_mean(&x_0, &x_t, 10).unwrap();
        let vals: Vec<f32> = mean.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals.len(), 2);
        // The posterior mean should be between x_0 and x_t
        for v in &vals {
            assert!(v.is_finite());
        }
    }

    // ========== Timestep Embedding Tests ==========

    #[test]
    fn test_timestep_embedding() {
        let device = Device::Cpu;
        let embed = timestep_embedding(10, 64, &device).unwrap();
        assert_eq!(embed.dims(), &[64]);

        // Different timesteps should give different embeddings
        let embed2 = timestep_embedding(20, 64, &device).unwrap();
        let v1: Vec<f32> = embed.to_vec1().unwrap();
        let v2: Vec<f32> = embed2.to_vec1().unwrap();
        let diff: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.1,
            "Different timesteps should yield different embeddings"
        );
    }

    #[test]
    fn test_batch_timestep_embeddings() {
        let device = Device::Cpu;
        let timesteps = vec![0, 10, 50, 99];
        let embeds = batch_timestep_embeddings(&timesteps, 32, &device).unwrap();
        assert_eq!(embeds.dims(), &[4, 32]);
    }

    // ========== Normalization Tests ==========

    #[test]
    fn test_normalization_stats() {
        let samples = make_training_data(20, 8, 2, None);
        let stats = NormalizationStats::compute(&samples);

        assert_eq!(stats.n_features, 2);
        assert_eq!(stats.means.len(), 2);
        assert_eq!(stats.stds.len(), 2);

        // Stds should be positive
        for &s in &stats.stds {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let samples = make_training_data(10, 8, 2, None);
        let stats = NormalizationStats::compute(&samples);

        let original = samples[0].data.clone();
        let mut sample = samples[0].clone();

        stats.normalize(&mut sample);
        // Normalized data should have different values
        let normalized = sample.data.clone();
        assert_ne!(original, normalized);

        stats.denormalize(&mut sample);
        // Should be close to original
        for (orig, denorm) in original.iter().zip(sample.data.iter()) {
            assert!(
                (orig - denorm).abs() < 1e-3,
                "Roundtrip failed: {} vs {}",
                orig,
                denorm
            );
        }
    }

    #[test]
    fn test_empty_normalization() {
        let stats = NormalizationStats::compute(&[]);
        assert_eq!(stats.n_features, 0);
        assert!(stats.means.is_empty());
    }

    // ========== DDPM Model Tests ==========

    #[test]
    fn test_ddpm_creation() {
        let config = DiffusionConfig::tiny();
        let ddpm = TimeSeriesDDPM::new(config);
        assert!(ddpm.is_ok());

        let ddpm = ddpm.unwrap();
        assert_eq!(ddpm.state(), DdpmState::Initialized);
        assert!(ddpm.training_stats().is_none());
        assert!(ddpm.norm_stats().is_none());
    }

    #[test]
    fn test_ddpm_train_empty_data() {
        let config = DiffusionConfig::tiny();
        let mut ddpm = TimeSeriesDDPM::new(config).unwrap();

        let result = ddpm.train(&[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            DiffusionError::InsufficientData { got, min } => {
                assert_eq!(got, 0);
                assert_eq!(min, 1);
            }
            other => panic!("Expected InsufficientData, got: {:?}", other),
        }
    }

    #[test]
    fn test_ddpm_train_dimension_mismatch() {
        let config = DiffusionConfig::tiny();
        let mut ddpm = TimeSeriesDDPM::new(config).unwrap();

        // Wrong seq_len
        let bad_sample = TimeSeriesSample::new(vec![0.0; 100], 50, 2);
        let result = ddpm.train(&[bad_sample]);
        assert!(result.is_err());

        match result.unwrap_err() {
            DiffusionError::DimensionMismatch { .. } => {}
            other => panic!("Expected DimensionMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn test_ddpm_train_and_sample() {
        let config = DiffusionConfig::tiny()
            .with_n_regimes(0) // unconditional for simplicity
            .with_n_epochs(2)
            .with_n_timesteps(10);

        let mut ddpm = TimeSeriesDDPM::new(config.clone()).unwrap();
        let samples = make_training_data(10, config.seq_len, config.n_features, None);

        // Train
        let stats = ddpm.train(&samples).unwrap();
        assert_eq!(stats.epochs_completed, 2);
        assert_eq!(stats.n_samples, 10);
        assert_eq!(stats.n_timesteps, 10);
        assert_eq!(stats.loss_history.len(), 2);
        assert_eq!(ddpm.state(), DdpmState::Trained);

        // Sample
        let batch = ddpm.sample(3, None).unwrap();
        assert_eq!(batch.samples.len(), 3);
        assert_eq!(batch.n_steps, 10);

        for s in &batch.samples {
            assert_eq!(s.seq_len, config.seq_len);
            assert_eq!(s.n_features, config.n_features);
        }
    }

    #[test]
    fn test_ddpm_conditional_train_and_sample() {
        let config = DiffusionConfig::tiny()
            .with_n_regimes(4)
            .with_n_epochs(2)
            .with_n_timesteps(10);

        let mut ddpm = TimeSeriesDDPM::new(config.clone()).unwrap();

        // Create training data with regime labels
        let mut samples = Vec::new();
        for regime in [
            RegimeLabel::Trending,
            RegimeLabel::Volatile,
            RegimeLabel::MeanReverting,
            RegimeLabel::Uncertain,
        ] {
            let mut batch = make_training_data(5, config.seq_len, config.n_features, Some(regime));
            samples.append(&mut batch);
        }

        // Train
        let stats = ddpm.train(&samples).unwrap();
        assert_eq!(stats.epochs_completed, 2);
        assert!(!stats.regime_counts.is_empty());

        // Sample conditioned on Volatile
        let batch = ddpm.sample(4, Some(RegimeLabel::Volatile)).unwrap();
        assert_eq!(batch.samples.len(), 4);
        assert_eq!(batch.regime, Some(RegimeLabel::Volatile));

        for s in &batch.samples {
            assert_eq!(s.regime, Some(RegimeLabel::Volatile));
        }
    }

    #[test]
    fn test_ddpm_sample_not_trained() {
        let config = DiffusionConfig::tiny();
        let ddpm = TimeSeriesDDPM::new(config).unwrap();

        let result = ddpm.sample(3, None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DiffusionError::NotTrained));
    }

    #[test]
    fn test_ddpm_sample_zero() {
        let config = DiffusionConfig::tiny()
            .with_n_regimes(0)
            .with_n_epochs(2)
            .with_n_timesteps(10);

        let mut ddpm = TimeSeriesDDPM::new(config.clone()).unwrap();
        let samples = make_training_data(5, config.seq_len, config.n_features, None);
        ddpm.train(&samples).unwrap();

        let batch = ddpm.sample(0, None).unwrap();
        assert!(batch.samples.is_empty());
    }

    // ========== SyntheticDataGenerator Tests ==========

    #[test]
    fn test_generator_creation() {
        let config = DiffusionConfig::tiny();
        let generator = SyntheticDataGenerator::new(config);
        assert!(generator.is_ok());
        assert_eq!(generator.unwrap().total_generated(), 0);
    }

    #[test]
    fn test_generator_train_and_generate() {
        let config = DiffusionConfig::tiny()
            .with_n_regimes(0)
            .with_n_epochs(2)
            .with_n_timesteps(10);

        let mut generator = SyntheticDataGenerator::new(config.clone()).unwrap();
        let train_data = make_training_data(10, config.seq_len, config.n_features, None);

        generator.train(&train_data).unwrap();

        let batch = generator.generate(5).unwrap();
        assert_eq!(batch.samples.len(), 5);
        assert_eq!(generator.total_generated(), 5);

        // Generate more
        let batch2 = generator.generate(3).unwrap();
        assert_eq!(batch2.samples.len(), 3);
        assert_eq!(generator.total_generated(), 8);
    }

    #[test]
    fn test_generator_regime_generation() {
        let config = DiffusionConfig::tiny()
            .with_n_regimes(4)
            .with_n_epochs(2)
            .with_n_timesteps(10);

        let mut generator = SyntheticDataGenerator::new(config.clone()).unwrap();

        let mut train_data = Vec::new();
        for regime in [RegimeLabel::Trending, RegimeLabel::Volatile] {
            let mut batch = make_training_data(5, config.seq_len, config.n_features, Some(regime));
            train_data.append(&mut batch);
        }

        generator.train(&train_data).unwrap();

        let batch = generator
            .generate_for_regime(4, RegimeLabel::Volatile)
            .unwrap();
        assert_eq!(batch.samples.len(), 4);
        assert_eq!(batch.regime, Some(RegimeLabel::Volatile));
    }

    #[test]
    fn test_generator_augment_sparse_regimes() {
        let config = DiffusionConfig::tiny()
            .with_n_regimes(4)
            .with_n_epochs(2)
            .with_n_timesteps(10);

        let mut generator = SyntheticDataGenerator::new(config.clone()).unwrap();

        // Create unbalanced dataset
        let mut train_data = Vec::new();
        // 10 trending samples
        let mut trending = make_training_data(
            10,
            config.seq_len,
            config.n_features,
            Some(RegimeLabel::Trending),
        );
        train_data.append(&mut trending);
        // 2 volatile samples (sparse!)
        let mut volatile = make_training_data(
            2,
            config.seq_len,
            config.n_features,
            Some(RegimeLabel::Volatile),
        );
        train_data.append(&mut volatile);

        generator.train(&train_data).unwrap();

        // Augment to 10 per regime
        let batches = generator.augment_sparse_regimes(&train_data, 10).unwrap();

        // Should have generated batches for under-represented regimes
        assert!(!batches.is_empty());

        // Check that volatile was augmented (had 2, need 10 → generate 8)
        let volatile_batch = batches
            .iter()
            .find(|b| b.regime == Some(RegimeLabel::Volatile));
        assert!(volatile_batch.is_some());
        assert_eq!(volatile_batch.unwrap().samples.len(), 8);
    }

    // ========== Quality Assessment Tests ==========

    #[test]
    fn test_quality_assessment_identical() {
        let config = DiffusionConfig::tiny();
        let generator = SyntheticDataGenerator::new(config.clone()).unwrap();

        let samples = make_training_data(20, 16, 2, None);

        // Compare data with itself — should give high quality score
        let report = generator.quality_assessment(&samples, &samples);
        assert!(report.mean_mae < 1e-5);
        assert!(report.std_mae < 1e-5);
        assert!(report.quality_score > 0.9);
        assert!(report.is_acceptable());
    }

    #[test]
    fn test_quality_assessment_empty() {
        let config = DiffusionConfig::tiny();
        let generator = SyntheticDataGenerator::new(config).unwrap();

        let report = generator.quality_assessment(&[], &[]);
        assert_eq!(report.n_real, 0);
        assert_eq!(report.n_synthetic, 0);
        assert!(!report.is_acceptable());
    }

    #[test]
    fn test_quality_report_different_data() {
        let config = DiffusionConfig::tiny();
        let generator = SyntheticDataGenerator::new(config).unwrap();

        let real = make_training_data(20, 16, 2, None);

        // Create very different "synthetic" data
        let mut fake = Vec::new();
        for i in 0..20 {
            let mut data = Vec::with_capacity(16 * 2);
            for _ in 0..(16 * 2) {
                data.push(1000.0 + i as f32);
            }
            fake.push(TimeSeriesSample::new(data, 16, 2));
        }

        let report = generator.quality_assessment(&real, &fake);
        assert!(report.mean_mae > 1.0);
        assert_eq!(report.n_real, 20);
        assert_eq!(report.n_synthetic, 20);
    }

    // ========== Generated Batch Tests ==========

    #[test]
    fn test_generated_batch_feature_statistics() {
        let samples = make_training_data(10, 16, 2, None);
        let batch = GeneratedBatch {
            samples,
            regime: None,
            n_steps: 50,
            generation_time: Duration::from_millis(100),
        };

        let stats = batch.feature_statistics();
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.n_samples, 10);
        assert_eq!(stats.n_features, 2);
        assert_eq!(stats.feature_means.len(), 2);
        assert_eq!(stats.feature_stds.len(), 2);
    }

    #[test]
    fn test_generated_batch_empty() {
        let batch = GeneratedBatch {
            samples: vec![],
            regime: None,
            n_steps: 0,
            generation_time: Duration::ZERO,
        };

        assert!(batch.feature_statistics().is_none());
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

    // ========== Variance Type Tests ==========

    #[test]
    fn test_variance_types() {
        // Just ensure all variance types are constructible and comparable
        assert_eq!(VarianceType::FixedSmall, VarianceType::FixedSmall);
        assert_ne!(VarianceType::FixedSmall, VarianceType::FixedLarge);
        assert_ne!(VarianceType::FixedLarge, VarianceType::Learned);
    }

    // ========== Loss Weighting Tests ==========

    #[test]
    fn test_loss_weighting_types() {
        assert_eq!(LossWeighting::Uniform, LossWeighting::Uniform);
        assert_ne!(LossWeighting::Uniform, LossWeighting::ImportanceSampling);
    }

    // ========== Schedule Type Tests ==========

    #[test]
    fn test_all_schedule_types_build() {
        for schedule in [
            NoiseScheduleType::Linear,
            NoiseScheduleType::Cosine,
            NoiseScheduleType::Quadratic,
        ] {
            let config = DiffusionConfig::default()
                .with_schedule(schedule)
                .with_n_timesteps(50);
            let sched = NoiseSchedule::new(&config);
            assert_eq!(sched.n_timesteps, 50);
            assert_eq!(sched.betas.len(), 50);
        }
    }

    // ========== Integration: Train + Sample + Quality ==========

    #[test]
    fn test_end_to_end_pipeline() {
        let config = DiffusionConfig::tiny()
            .with_n_regimes(0)
            .with_n_epochs(3)
            .with_n_timesteps(10);

        let mut generator = SyntheticDataGenerator::new(config.clone()).unwrap();
        let train_data = make_training_data(20, config.seq_len, config.n_features, None);

        // Train
        let stats = generator.train(&train_data).unwrap();
        assert_eq!(generator.model_state(), DdpmState::Trained);
        assert_eq!(stats.epochs_completed, 3);

        // Generate
        let batch = generator.generate(10).unwrap();
        assert_eq!(batch.samples.len(), 10);

        // Quality assessment
        let report = generator.quality_assessment(&train_data, &batch.samples);
        assert!(report.mean_mae.is_finite());
        assert!(report.std_mae.is_finite());
        assert!(report.quality_score >= 0.0 && report.quality_score <= 1.0);

        // Feature statistics
        let feat_stats = batch.feature_statistics().unwrap();
        assert_eq!(feat_stats.n_features, config.n_features);
    }
}
