//! Differentiable GAF (DiffGAF)
//!
//! Part of the Visual Cortex region
//! Component: gaf
//!
//! Provides a differentiable Gramian Angular Field transform that preserves
//! gradient information for use in end-to-end learning pipelines. Unlike the
//! standard GAF encoding which discards derivative structure, DiffGAF retains
//! the Jacobian of the encoding w.r.t. the input time series, enabling
//! backpropagation through the image-formation step.
//!
//! ## Features
//!
//! - **Gradient-passable encoding**: Computes both the GAF image and its
//!   Jacobian w.r.t. the input, enabling gradient flow from downstream
//!   vision models back to raw price/volume features
//! - **Configurable GAF type**: Supports both GASF (summation) and GADF
//!   (difference) field types
//! - **Sensitivity analysis**: Per-pixel sensitivity map showing which
//!   input time steps most influence each output pixel
//! - **Batch encoding**: Encode multiple time series in one call with
//!   accumulated statistics
//! - **EMA-smoothed quality tracking**: Monitors encoding quality
//!   (dynamic range, sparsity) with exponential smoothing
//! - **Windowed diagnostics**: Recent encoding quality metrics for
//!   adaptive pipeline tuning
//! - **Running statistics**: Per-encoding min/max/mean of pixel values,
//!   gradient norms, and sensitivity magnitudes

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the differentiable GAF engine.
#[derive(Debug, Clone)]
pub struct DifferentiableConfig {
    /// Default image size (N×N output).
    pub image_size: usize,
    /// GAF type: summation (GASF) or difference (GADF).
    pub gaf_type: DiffGafType,
    /// EMA decay factor for quality tracking (0 < decay < 1).
    pub ema_decay: f64,
    /// Minimum samples before EMA is considered initialised.
    pub min_samples: usize,
    /// Sliding window size for windowed diagnostics.
    pub window_size: usize,
    /// Whether to compute the full Jacobian on every encoding.
    pub compute_jacobian: bool,
    /// Whether to compute sensitivity maps on every encoding.
    pub compute_sensitivity: bool,
    /// Epsilon for numerical stability in normalisation.
    pub epsilon: f64,
}

impl Default for DifferentiableConfig {
    fn default() -> Self {
        Self {
            image_size: 32,
            gaf_type: DiffGafType::GASF,
            ema_decay: 0.05,
            min_samples: 10,
            window_size: 64,
            compute_jacobian: true,
            compute_sensitivity: true,
            epsilon: 1e-10,
        }
    }
}

// ---------------------------------------------------------------------------
// GAF type
// ---------------------------------------------------------------------------

/// Type of Gramian Angular Field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffGafType {
    /// Gramian Angular Summation Field: cos(θ_i + θ_j)
    GASF,
    /// Gramian Angular Difference Field: sin(θ_i - θ_j)
    GADF,
}

impl DiffGafType {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::GASF => "GASF",
            Self::GADF => "GADF",
        }
    }
}

impl std::fmt::Display for DiffGafType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Encoding result
// ---------------------------------------------------------------------------

/// Result of a single differentiable GAF encoding.
#[derive(Debug, Clone)]
pub struct DiffGafEncoding {
    /// The GAF image as a flattened row-major N×N matrix.
    pub image: Vec<f64>,
    /// Image width/height.
    pub size: usize,
    /// The normalised input (after min-max to [-1, 1]).
    pub normalised: Vec<f64>,
    /// Polar angles (arccos of normalised values).
    pub angles: Vec<f64>,
    /// Jacobian of the GAF image w.r.t. the (rescaled) input.
    /// Shape: (size*size) × size, stored row-major.
    /// Only populated if `compute_jacobian` is enabled.
    pub jacobian: Option<Vec<f64>>,
    /// Per-pixel sensitivity: for each output pixel, the sum of
    /// absolute Jacobian entries across all input dimensions.
    /// Shape: size × size, stored row-major.
    /// Only populated if `compute_sensitivity` is enabled.
    pub sensitivity: Option<Vec<f64>>,
    /// Dynamic range of the image (max - min pixel value).
    pub dynamic_range: f64,
    /// Sparsity: fraction of near-zero pixels (|pixel| < epsilon).
    pub sparsity: f64,
    /// Mean absolute gradient norm (mean of |J| entries).
    pub mean_grad_norm: f64,
    /// GAF type used.
    pub gaf_type: DiffGafType,
}

impl DiffGafEncoding {
    /// Get pixel at (row, col).
    pub fn pixel(&self, row: usize, col: usize) -> f64 {
        self.image[row * self.size + col]
    }

    /// Get the image as a 2D vector.
    pub fn to_2d(&self) -> Vec<Vec<f64>> {
        self.image
            .chunks(self.size)
            .map(|row| row.to_vec())
            .collect()
    }

    /// Get the sensitivity map as a 2D vector.
    pub fn sensitivity_2d(&self) -> Option<Vec<Vec<f64>>> {
        self.sensitivity
            .as_ref()
            .map(|s| s.chunks(self.size).map(|row| row.to_vec()).collect())
    }

    /// Get the Jacobian entry dImage[pixel_idx] / dInput[input_idx].
    pub fn jacobian_entry(&self, pixel_idx: usize, input_idx: usize) -> Option<f64> {
        self.jacobian
            .as_ref()
            .map(|j| j[pixel_idx * self.size + input_idx])
    }

    /// Compute the gradient of a scalar loss w.r.t. the input,
    /// given the gradient of the loss w.r.t. the image pixels.
    ///
    /// `dloss_dimage` must have length size*size.
    /// Returns gradient w.r.t. the rescaled input of length `size`.
    pub fn backward(&self, dloss_dimage: &[f64]) -> Option<Vec<f64>> {
        let jac = self.jacobian.as_ref()?;
        let n = self.size;
        let n2 = n * n;
        if dloss_dimage.len() != n2 {
            return None;
        }
        let mut grad_input = vec![0.0; n];
        for pixel in 0..n2 {
            let dl = dloss_dimage[pixel];
            if dl.abs() < 1e-15 {
                continue;
            }
            for inp in 0..n {
                grad_input[inp] += dl * jac[pixel * n + inp];
            }
        }
        Some(grad_input)
    }
}

// ---------------------------------------------------------------------------
// Batch result
// ---------------------------------------------------------------------------

/// Result of encoding a batch of time series.
#[derive(Debug, Clone)]
pub struct BatchEncoding {
    /// Individual encoding results.
    pub encodings: Vec<DiffGafEncoding>,
    /// Mean dynamic range across the batch.
    pub mean_dynamic_range: f64,
    /// Mean sparsity across the batch.
    pub mean_sparsity: f64,
    /// Mean gradient norm across the batch.
    pub mean_grad_norm: f64,
    /// Number of encodings in the batch.
    pub count: usize,
}

// ---------------------------------------------------------------------------
// Window record
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct WindowRecord {
    dynamic_range: f64,
    sparsity: f64,
    mean_grad_norm: f64,
    input_length: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the differentiable GAF engine.
#[derive(Debug, Clone)]
pub struct DifferentiableStats {
    /// Total encodings performed.
    pub total_encodings: u64,
    /// Total batch encodings performed.
    pub total_batches: u64,
    /// Total backward passes computed.
    pub total_backwards: u64,
    /// Peak dynamic range seen.
    pub peak_dynamic_range: f64,
    /// Sum of dynamic ranges (for mean computation).
    pub sum_dynamic_range: f64,
    /// Sum of sparsities.
    pub sum_sparsity: f64,
    /// Sum of gradient norms.
    pub sum_grad_norm: f64,
    /// Minimum input length seen.
    pub min_input_length: usize,
    /// Maximum input length seen.
    pub max_input_length: usize,
}

impl Default for DifferentiableStats {
    fn default() -> Self {
        Self {
            total_encodings: 0,
            total_batches: 0,
            total_backwards: 0,
            peak_dynamic_range: 0.0,
            sum_dynamic_range: 0.0,
            sum_sparsity: 0.0,
            sum_grad_norm: 0.0,
            min_input_length: usize::MAX,
            max_input_length: 0,
        }
    }
}

impl DifferentiableStats {
    /// Mean dynamic range across all encodings.
    pub fn mean_dynamic_range(&self) -> f64 {
        if self.total_encodings == 0 {
            0.0
        } else {
            self.sum_dynamic_range / self.total_encodings as f64
        }
    }

    /// Mean sparsity across all encodings.
    pub fn mean_sparsity(&self) -> f64 {
        if self.total_encodings == 0 {
            0.0
        } else {
            self.sum_sparsity / self.total_encodings as f64
        }
    }

    /// Mean gradient norm across all encodings.
    pub fn mean_grad_norm(&self) -> f64 {
        if self.total_encodings == 0 {
            0.0
        } else {
            self.sum_grad_norm / self.total_encodings as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Differentiable
// ---------------------------------------------------------------------------

/// Differentiable GAF (DiffGAF) engine.
///
/// Computes Gramian Angular Field images from time series data while
/// preserving gradient information for end-to-end learning. Tracks
/// encoding quality with EMA smoothing and windowed diagnostics.
pub struct Differentiable {
    config: DifferentiableConfig,

    // EMA state
    ema_dynamic_range: f64,
    ema_sparsity: f64,
    ema_grad_norm: f64,
    ema_initialized: bool,

    // Windowed diagnostics
    recent: VecDeque<WindowRecord>,

    // Counters
    stats: DifferentiableStats,
    current_tick: u64,
}

impl Default for Differentiable {
    fn default() -> Self {
        Self::new()
    }
}

impl Differentiable {
    /// Create a new differentiable GAF engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(DifferentiableConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: DifferentiableConfig) -> Result<Self> {
        if config.image_size == 0 {
            return Err(Error::InvalidInput("image_size must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.epsilon <= 0.0 {
            return Err(Error::InvalidInput("epsilon must be > 0".into()));
        }
        Ok(Self {
            config,
            ema_dynamic_range: 0.0,
            ema_sparsity: 0.0,
            ema_grad_norm: 0.0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: DifferentiableStats::default(),
            current_tick: 0,
        })
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Encode a time series into a differentiable GAF image.
    pub fn encode(&mut self, time_series: &[f64]) -> Result<DiffGafEncoding> {
        self.encode_with_size(time_series, self.config.image_size)
    }

    /// Encode with a specific output image size.
    pub fn encode_with_size(
        &mut self,
        time_series: &[f64],
        image_size: usize,
    ) -> Result<DiffGafEncoding> {
        if time_series.is_empty() {
            return Err(Error::InvalidInput("time series cannot be empty".into()));
        }
        if image_size == 0 {
            return Err(Error::InvalidInput("image_size must be > 0".into()));
        }

        let n = image_size;

        // Step 1: Normalise to [-1, 1]
        let normalised = self.normalise(time_series);

        // Step 2: Rescale to image_size via piecewise aggregate approximation
        let rescaled = if normalised.len() != n {
            self.paa(&normalised, n)
        } else {
            normalised.clone()
        };

        // Step 3: Compute polar angles θ = arccos(x)
        let angles: Vec<f64> = rescaled
            .iter()
            .map(|&x| x.clamp(-1.0, 1.0).acos())
            .collect();

        // Step 4: Compute GAF image
        let image = self.compute_gaf(&angles, n);

        // Step 5: Compute Jacobian if requested
        let jacobian = if self.config.compute_jacobian {
            Some(self.compute_jacobian(&rescaled, &angles, n))
        } else {
            None
        };

        // Step 6: Compute sensitivity map if requested
        let sensitivity = if self.config.compute_sensitivity {
            jacobian
                .as_ref()
                .map(|jac| self.compute_sensitivity(jac, n))
        } else {
            None
        };

        // Step 7: Compute quality metrics
        let (dynamic_range, sparsity) = self.compute_quality_metrics(&image, n);
        let mean_grad_norm = jacobian
            .as_ref()
            .map(|jac| {
                let sum: f64 = jac.iter().map(|x| x.abs()).sum();
                sum / jac.len() as f64
            })
            .unwrap_or(0.0);

        // Step 8: Update EMA
        self.update_ema(dynamic_range, sparsity, mean_grad_norm);

        // Step 9: Update window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(WindowRecord {
            dynamic_range,
            sparsity,
            mean_grad_norm,
            input_length: time_series.len(),
        });

        // Step 10: Update stats
        self.stats.total_encodings += 1;
        self.stats.sum_dynamic_range += dynamic_range;
        self.stats.sum_sparsity += sparsity;
        self.stats.sum_grad_norm += mean_grad_norm;
        if dynamic_range > self.stats.peak_dynamic_range {
            self.stats.peak_dynamic_range = dynamic_range;
        }
        if time_series.len() < self.stats.min_input_length {
            self.stats.min_input_length = time_series.len();
        }
        if time_series.len() > self.stats.max_input_length {
            self.stats.max_input_length = time_series.len();
        }

        Ok(DiffGafEncoding {
            image,
            size: n,
            normalised: rescaled,
            angles,
            jacobian,
            sensitivity,
            dynamic_range,
            sparsity,
            mean_grad_norm,
            gaf_type: self.config.gaf_type,
        })
    }

    /// Encode a batch of time series.
    pub fn encode_batch(&mut self, batch: &[&[f64]]) -> Result<BatchEncoding> {
        if batch.is_empty() {
            return Err(Error::InvalidInput("batch cannot be empty".into()));
        }

        let mut encodings = Vec::with_capacity(batch.len());
        let mut sum_dr = 0.0;
        let mut sum_sp = 0.0;
        let mut sum_gn = 0.0;

        for ts in batch {
            let enc = self.encode(ts)?;
            sum_dr += enc.dynamic_range;
            sum_sp += enc.sparsity;
            sum_gn += enc.mean_grad_norm;
            encodings.push(enc);
        }

        let count = encodings.len();
        self.stats.total_batches += 1;

        Ok(BatchEncoding {
            encodings,
            mean_dynamic_range: sum_dr / count as f64,
            mean_sparsity: sum_sp / count as f64,
            mean_grad_norm: sum_gn / count as f64,
            count,
        })
    }

    // -----------------------------------------------------------------------
    // Internal computation
    // -----------------------------------------------------------------------

    /// Normalise time series to [-1, 1].
    fn normalise(&self, series: &[f64]) -> Vec<f64> {
        let min = series.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range.abs() < self.config.epsilon {
            return vec![0.0; series.len()];
        }

        series
            .iter()
            .map(|&x| 2.0 * (x - min) / range - 1.0)
            .collect()
    }

    /// Piecewise Aggregate Approximation — rescale to target length.
    fn paa(&self, series: &[f64], target: usize) -> Vec<f64> {
        let n = series.len();
        let seg = n as f64 / target as f64;

        (0..target)
            .map(|i| {
                let start = (i as f64 * seg) as usize;
                let end = (((i + 1) as f64 * seg).ceil() as usize).min(n);
                let sum: f64 = series[start..end].iter().sum();
                sum / (end - start) as f64
            })
            .collect()
    }

    /// Compute the GAF image from polar angles.
    fn compute_gaf(&self, angles: &[f64], n: usize) -> Vec<f64> {
        let mut image = vec![0.0; n * n];
        match self.config.gaf_type {
            DiffGafType::GASF => {
                for i in 0..n {
                    for j in 0..n {
                        image[i * n + j] = (angles[i] + angles[j]).cos();
                    }
                }
            }
            DiffGafType::GADF => {
                for i in 0..n {
                    for j in 0..n {
                        image[i * n + j] = (angles[i] - angles[j]).sin();
                    }
                }
            }
        }
        image
    }

    /// Compute the Jacobian of the GAF image w.r.t. the rescaled input.
    ///
    /// For GASF: image[i,j] = cos(θ_i + θ_j)
    ///   dimage[i,j]/dx_k = -sin(θ_i + θ_j) * (dθ_i/dx_k + dθ_j/dx_k)
    ///   where dθ_k/dx_k = -1/sqrt(1 - x_k^2)
    ///
    /// For GADF: image[i,j] = sin(θ_i - θ_j)
    ///   dimage[i,j]/dx_k = cos(θ_i - θ_j) * (dθ_i/dx_k - dθ_j/dx_k)
    fn compute_jacobian(&self, rescaled: &[f64], angles: &[f64], n: usize) -> Vec<f64> {
        // dθ_k / dx_k = -1 / sqrt(1 - x_k^2)
        let dtheta_dx: Vec<f64> = rescaled
            .iter()
            .map(|&x| {
                let denom = (1.0 - x * x).max(self.config.epsilon).sqrt();
                -1.0 / denom
            })
            .collect();

        let n2 = n * n;
        let mut jac = vec![0.0; n2 * n];

        match self.config.gaf_type {
            DiffGafType::GASF => {
                for i in 0..n {
                    for j in 0..n {
                        let pixel = i * n + j;
                        let neg_sin = -(angles[i] + angles[j]).sin();
                        // dimage[i,j]/dx_k is non-zero only when k==i or k==j
                        // k == i: neg_sin * dtheta_dx[i]
                        // k == j: neg_sin * dtheta_dx[j]
                        // If i == j, both contribute: neg_sin * 2 * dtheta_dx[i]
                        if i == j {
                            jac[pixel * n + i] = neg_sin * 2.0 * dtheta_dx[i];
                        } else {
                            jac[pixel * n + i] += neg_sin * dtheta_dx[i];
                            jac[pixel * n + j] += neg_sin * dtheta_dx[j];
                        }
                    }
                }
            }
            DiffGafType::GADF => {
                for i in 0..n {
                    for j in 0..n {
                        let pixel = i * n + j;
                        let cos_diff = (angles[i] - angles[j]).cos();
                        // dimage[i,j]/dx_k:
                        // k == i: cos_diff * dtheta_dx[i]
                        // k == j: cos_diff * (-dtheta_dx[j])
                        if i == j {
                            // cos_diff * (dtheta_dx[i] - dtheta_dx[j]) = 0 since i==j
                            // Actually: d/dx_k sin(θ_i - θ_j) at i==j gives sin(0)=0 image,
                            // derivative is cos(0) * (dθ_i/dx_k - dθ_j/dx_k)
                            // When k==i==j: cos(0) * (dtheta - dtheta) = 0
                            jac[pixel * n + i] = 0.0;
                        } else {
                            jac[pixel * n + i] += cos_diff * dtheta_dx[i];
                            jac[pixel * n + j] += cos_diff * (-dtheta_dx[j]);
                        }
                    }
                }
            }
        }

        jac
    }

    /// Compute per-pixel sensitivity from the Jacobian.
    fn compute_sensitivity(&self, jacobian: &[f64], n: usize) -> Vec<f64> {
        let n2 = n * n;
        let mut sensitivity = vec![0.0; n2];
        for pixel in 0..n2 {
            let mut sum = 0.0;
            for inp in 0..n {
                sum += jacobian[pixel * n + inp].abs();
            }
            sensitivity[pixel] = sum;
        }
        sensitivity
    }

    /// Compute quality metrics for an encoding.
    fn compute_quality_metrics(&self, image: &[f64], _n: usize) -> (f64, f64) {
        let min = image.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dynamic_range = max - min;

        let near_zero = image
            .iter()
            .filter(|&&x| x.abs() < self.config.epsilon)
            .count();
        let sparsity = near_zero as f64 / image.len() as f64;

        (dynamic_range, sparsity)
    }

    /// Update EMA tracking.
    fn update_ema(&mut self, dynamic_range: f64, sparsity: f64, grad_norm: f64) {
        let d = self.config.ema_decay;
        if !self.ema_initialized {
            self.ema_dynamic_range = dynamic_range;
            self.ema_sparsity = sparsity;
            self.ema_grad_norm = grad_norm;
            self.ema_initialized = true;
        } else {
            self.ema_dynamic_range = d * dynamic_range + (1.0 - d) * self.ema_dynamic_range;
            self.ema_sparsity = d * sparsity + (1.0 - d) * self.ema_sparsity;
            self.ema_grad_norm = d * grad_norm + (1.0 - d) * self.ema_grad_norm;
        }
    }

    // -----------------------------------------------------------------------
    // Configuration queries
    // -----------------------------------------------------------------------

    /// Get the current GAF type.
    pub fn gaf_type(&self) -> DiffGafType {
        self.config.gaf_type
    }

    /// Set the GAF type.
    pub fn set_gaf_type(&mut self, gaf_type: DiffGafType) {
        self.config.gaf_type = gaf_type;
    }

    /// Get the default image size.
    pub fn image_size(&self) -> usize {
        self.config.image_size
    }

    /// Set the default image size.
    pub fn set_image_size(&mut self, size: usize) -> Result<()> {
        if size == 0 {
            return Err(Error::InvalidInput("image_size must be > 0".into()));
        }
        self.config.image_size = size;
        Ok(())
    }

    /// Whether Jacobian computation is enabled.
    pub fn is_jacobian_enabled(&self) -> bool {
        self.config.compute_jacobian
    }

    /// Enable or disable Jacobian computation.
    pub fn set_compute_jacobian(&mut self, enabled: bool) {
        self.config.compute_jacobian = enabled;
    }

    /// Whether sensitivity computation is enabled.
    pub fn is_sensitivity_enabled(&self) -> bool {
        self.config.compute_sensitivity
    }

    /// Enable or disable sensitivity computation.
    pub fn set_compute_sensitivity(&mut self, enabled: bool) {
        self.config.compute_sensitivity = enabled;
    }

    // -----------------------------------------------------------------------
    // EMA queries
    // -----------------------------------------------------------------------

    /// EMA-smoothed dynamic range.
    pub fn ema_dynamic_range(&self) -> f64 {
        self.ema_dynamic_range
    }

    /// EMA-smoothed sparsity.
    pub fn ema_sparsity(&self) -> f64 {
        self.ema_sparsity
    }

    /// EMA-smoothed gradient norm.
    pub fn ema_grad_norm(&self) -> f64 {
        self.ema_grad_norm
    }

    /// Whether the EMA tracker has been initialised.
    pub fn is_warmed_up(&self) -> bool {
        self.stats.total_encodings >= self.config.min_samples as u64
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Windowed mean dynamic range.
    pub fn windowed_mean_dynamic_range(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.dynamic_range).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean sparsity.
    pub fn windowed_mean_sparsity(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.sparsity).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean gradient norm.
    pub fn windowed_mean_grad_norm(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.mean_grad_norm).sum();
        sum / self.recent.len() as f64
    }

    /// Whether dynamic range is increasing over the window.
    pub fn is_dynamic_range_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let n = self.recent.len();
        let half = n / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.dynamic_range)
            .sum::<f64>()
            / half as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.dynamic_range)
            .sum::<f64>()
            / (n - half) as f64;
        second_half > first_half
    }

    /// Confidence based on number of encodings vs min_samples.
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_encodings as f64;
        let req = self.config.min_samples as f64;
        (n / req).min(1.0)
    }

    // -----------------------------------------------------------------------
    // Stats / Tick / Process
    // -----------------------------------------------------------------------

    /// Get running statistics.
    pub fn stats(&self) -> &DifferentiableStats {
        &self.stats
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Advance one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Main processing function (tick alias).
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset EMA and windowed diagnostics, keeping stats.
    pub fn reset_ema(&mut self) {
        self.ema_dynamic_range = 0.0;
        self.ema_sparsity = 0.0;
        self.ema_grad_norm = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.reset_ema();
        self.stats = DifferentiableStats::default();
        self.current_tick = 0;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = Differentiable::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_default() {
        let d = Differentiable::default();
        assert_eq!(d.image_size(), 32);
        assert_eq!(d.gaf_type(), DiffGafType::GASF);
    }

    #[test]
    fn test_with_config() {
        let cfg = DifferentiableConfig {
            image_size: 8,
            gaf_type: DiffGafType::GADF,
            ..Default::default()
        };
        let d = Differentiable::with_config(cfg).unwrap();
        assert_eq!(d.image_size(), 8);
        assert_eq!(d.gaf_type(), DiffGafType::GADF);
    }

    #[test]
    fn test_invalid_config_image_size() {
        let mut cfg = DifferentiableConfig::default();
        cfg.image_size = 0;
        assert!(Differentiable::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = DifferentiableConfig::default();
        cfg.ema_decay = 0.0;
        assert!(Differentiable::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = DifferentiableConfig::default();
        cfg.ema_decay = 1.0;
        assert!(Differentiable::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_window_size() {
        let mut cfg = DifferentiableConfig::default();
        cfg.window_size = 0;
        assert!(Differentiable::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_epsilon() {
        let mut cfg = DifferentiableConfig::default();
        cfg.epsilon = 0.0;
        assert!(Differentiable::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // GAF type
    // -----------------------------------------------------------------------

    #[test]
    fn test_gaf_type_label() {
        assert_eq!(DiffGafType::GASF.label(), "GASF");
        assert_eq!(DiffGafType::GADF.label(), "GADF");
    }

    #[test]
    fn test_gaf_type_display() {
        assert_eq!(format!("{}", DiffGafType::GASF), "GASF");
        assert_eq!(format!("{}", DiffGafType::GADF), "GADF");
    }

    #[test]
    fn test_set_gaf_type() {
        let mut d = Differentiable::new();
        d.set_gaf_type(DiffGafType::GADF);
        assert_eq!(d.gaf_type(), DiffGafType::GADF);
    }

    // -----------------------------------------------------------------------
    // Encoding — GASF
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_gasf_basic() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GASF,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let enc = d.encode(&ts).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(enc.image.len(), 16);
        assert_eq!(enc.gaf_type, DiffGafType::GASF);
    }

    #[test]
    fn test_encode_gasf_symmetry() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GASF,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let enc = d.encode(&ts).unwrap();

        // GASF should be symmetric: image[i][j] == image[j][i]
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (enc.pixel(i, j) - enc.pixel(j, i)).abs() < 1e-10,
                    "GASF not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_encode_gasf_diagonal() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GASF,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let enc = d.encode(&ts).unwrap();

        // Diagonal: cos(2θ_i) — should be within [-1, 1]
        for i in 0..4 {
            assert!(enc.pixel(i, i) >= -1.0 - 1e-10);
            assert!(enc.pixel(i, i) <= 1.0 + 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Encoding — GADF
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_gadf_basic() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GADF,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let enc = d.encode(&ts).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(enc.gaf_type, DiffGafType::GADF);
    }

    #[test]
    fn test_encode_gadf_antisymmetry() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GADF,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let enc = d.encode(&ts).unwrap();

        // GADF should be antisymmetric: image[i][j] == -image[j][i]
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (enc.pixel(i, j) + enc.pixel(j, i)).abs() < 1e-10,
                    "GADF not antisymmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_encode_gadf_diagonal_zero() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GADF,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        let enc = d.encode(&ts).unwrap();

        // GADF diagonal: sin(θ_i - θ_i) = sin(0) = 0
        for i in 0..4 {
            assert!(enc.pixel(i, i).abs() < 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Encoding — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_empty_series() {
        let mut d = Differentiable::new();
        assert!(d.encode(&[]).is_err());
    }

    #[test]
    fn test_encode_single_value() {
        let cfg = DifferentiableConfig {
            image_size: 1,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[42.0]).unwrap();
        assert_eq!(enc.size, 1);
        assert_eq!(enc.image.len(), 1);
    }

    #[test]
    fn test_encode_constant_series() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GASF,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        // All values same → normalised to 0, θ = π/2
        let ts = vec![5.0, 5.0, 5.0, 5.0];
        let enc = d.encode(&ts).unwrap();
        // cos(π/2 + π/2) = cos(π) = -1
        for i in 0..4 {
            for j in 0..4 {
                assert!((enc.pixel(i, j) - (-1.0)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_encode_rescaling() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        // Input longer than image_size → PAA rescaling
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let enc = d.encode(&ts).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(enc.normalised.len(), 4);
    }

    #[test]
    fn test_encode_with_custom_size() {
        let mut d = Differentiable::new();
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let enc = d.encode_with_size(&ts, 3).unwrap();
        assert_eq!(enc.size, 3);
        assert_eq!(enc.image.len(), 9);
    }

    #[test]
    fn test_encode_with_size_zero() {
        let mut d = Differentiable::new();
        assert!(d.encode_with_size(&[1.0], 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Normalisation
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalised_range() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts = vec![10.0, 20.0, 30.0, 40.0];
        let enc = d.encode(&ts).unwrap();

        for &v in &enc.normalised {
            assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&v));
        }
        // First should be -1, last should be +1
        assert!((enc.normalised[0] - (-1.0)).abs() < 1e-10);
        assert!((enc.normalised[3] - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Jacobian
    // -----------------------------------------------------------------------

    #[test]
    fn test_jacobian_present_when_enabled() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            compute_jacobian: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(enc.jacobian.is_some());
        assert_eq!(enc.jacobian.as_ref().unwrap().len(), 16 * 4); // n2 * n
    }

    #[test]
    fn test_jacobian_absent_when_disabled() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            compute_jacobian: false,
            compute_sensitivity: false,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(enc.jacobian.is_none());
    }

    #[test]
    fn test_jacobian_entry() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            compute_jacobian: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0]).unwrap();
        let entry = enc.jacobian_entry(0, 0);
        assert!(entry.is_some());
    }

    #[test]
    fn test_jacobian_finite() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            compute_jacobian: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        for &v in enc.jacobian.as_ref().unwrap() {
            assert!(v.is_finite(), "Jacobian contains non-finite value: {}", v);
        }
    }

    // -----------------------------------------------------------------------
    // Sensitivity
    // -----------------------------------------------------------------------

    #[test]
    fn test_sensitivity_present_when_enabled() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            compute_jacobian: true,
            compute_sensitivity: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(enc.sensitivity.is_some());
        assert_eq!(enc.sensitivity.as_ref().unwrap().len(), 16);
    }

    #[test]
    fn test_sensitivity_absent_without_jacobian() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            compute_jacobian: false,
            compute_sensitivity: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        // sensitivity requires jacobian
        assert!(enc.sensitivity.is_none());
    }

    #[test]
    fn test_sensitivity_non_negative() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            compute_jacobian: true,
            compute_sensitivity: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        for &v in enc.sensitivity.as_ref().unwrap() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_sensitivity_2d() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            compute_jacobian: true,
            compute_sensitivity: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0]).unwrap();
        let s2d = enc.sensitivity_2d().unwrap();
        assert_eq!(s2d.len(), 3);
        assert_eq!(s2d[0].len(), 3);
    }

    // -----------------------------------------------------------------------
    // Backward pass
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            compute_jacobian: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0]).unwrap();

        // Uniform loss gradient
        let dloss = vec![1.0; 9];
        let grad = enc.backward(&dloss);
        assert!(grad.is_some());
        let grad = grad.unwrap();
        assert_eq!(grad.len(), 3);
        for &g in &grad {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_backward_no_jacobian() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            compute_jacobian: false,
            compute_sensitivity: false,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0]).unwrap();
        let dloss = vec![1.0; 9];
        assert!(enc.backward(&dloss).is_none());
    }

    #[test]
    fn test_backward_wrong_size() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            compute_jacobian: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0]).unwrap();
        // Wrong size gradient
        let dloss = vec![1.0; 5];
        assert!(enc.backward(&dloss).is_none());
    }

    #[test]
    fn test_backward_zero_gradient() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            compute_jacobian: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0]).unwrap();

        let dloss = vec![0.0; 9];
        let grad = enc.backward(&dloss).unwrap();
        for &g in &grad {
            assert!((g - 0.0).abs() < 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // to_2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_2d() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0]).unwrap();
        let img2d = enc.to_2d();
        assert_eq!(img2d.len(), 3);
        assert_eq!(img2d[0].len(), 3);
    }

    // -----------------------------------------------------------------------
    // Quality metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_dynamic_range() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(enc.dynamic_range >= 0.0);
    }

    #[test]
    fn test_sparsity_range() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(enc.sparsity >= 0.0 && enc.sparsity <= 1.0);
    }

    #[test]
    fn test_mean_grad_norm_non_negative() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            compute_jacobian: true,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(enc.mean_grad_norm >= 0.0);
    }

    #[test]
    fn test_mean_grad_norm_zero_without_jacobian() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            compute_jacobian: false,
            compute_sensitivity: false,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let enc = d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((enc.mean_grad_norm - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // EMA
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_initializes_on_first_encode() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        assert!(!d.ema_initialized);
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(d.ema_initialized);
    }

    #[test]
    fn test_ema_dynamic_range_updates() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ema_decay: 0.5,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let first = d.ema_dynamic_range();
        d.encode(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        // After encoding constant series (all pixels ≈ same), DR should decrease
        let second = d.ema_dynamic_range();
        assert!(second < first || (first - second).abs() < 1e-10);
    }

    #[test]
    fn test_is_warmed_up() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            min_samples: 3,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        assert!(!d.is_warmed_up());
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(!d.is_warmed_up());
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(d.is_warmed_up());
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    #[test]
    fn test_windowed_mean_dynamic_range_empty() {
        let d = Differentiable::new();
        assert!((d.windowed_mean_dynamic_range() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_dynamic_range() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            window_size: 10,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let wm = d.windowed_mean_dynamic_range();
        assert!(wm > 0.0);
    }

    #[test]
    fn test_windowed_mean_sparsity_empty() {
        let d = Differentiable::new();
        assert!((d.windowed_mean_sparsity() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_grad_norm_empty() {
        let d = Differentiable::new();
        assert!((d.windowed_mean_grad_norm() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_eviction() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            window_size: 3,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        for _ in 0..5 {
            d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        }
        assert_eq!(d.recent.len(), 3);
    }

    #[test]
    fn test_is_dynamic_range_increasing_insufficient() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(!d.is_dynamic_range_increasing()); // < 4 records
    }

    #[test]
    fn test_confidence_increases() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            min_samples: 10,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        assert!((d.confidence() - 0.0).abs() < 1e-10);
        for _ in 0..5 {
            d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        }
        assert!((d.confidence() - 0.5).abs() < 1e-10);
        for _ in 0..5 {
            d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        }
        assert!((d.confidence() - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Batch encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_batch() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts1 = vec![1.0, 2.0, 3.0, 4.0];
        let ts2 = vec![5.0, 6.0, 7.0, 8.0];
        let batch = d.encode_batch(&[&ts1, &ts2]).unwrap();
        assert_eq!(batch.count, 2);
        assert_eq!(batch.encodings.len(), 2);
        assert!(batch.mean_dynamic_range >= 0.0);
        assert!(batch.mean_sparsity >= 0.0);
    }

    #[test]
    fn test_encode_batch_empty() {
        let mut d = Differentiable::new();
        assert!(d.encode_batch(&[]).is_err());
    }

    #[test]
    fn test_encode_batch_updates_stats() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        let ts = vec![1.0, 2.0, 3.0, 4.0];
        d.encode_batch(&[&ts, &ts, &ts]).unwrap();
        assert_eq!(d.stats().total_encodings, 3);
        assert_eq!(d.stats().total_batches, 1);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let d = Differentiable::new();
        assert_eq!(d.stats().total_encodings, 0);
        assert_eq!(d.stats().total_batches, 0);
    }

    #[test]
    fn test_stats_after_encoding() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(d.stats().total_encodings, 1);
        assert!(d.stats().peak_dynamic_range > 0.0);
        assert_eq!(d.stats().min_input_length, 4);
        assert_eq!(d.stats().max_input_length, 4);
    }

    #[test]
    fn test_stats_mean_methods() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        d.encode(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert!(d.stats().mean_dynamic_range() > 0.0);
        assert!(d.stats().mean_sparsity() >= 0.0);
    }

    #[test]
    fn test_stats_mean_empty() {
        let d = Differentiable::new();
        assert!((d.stats().mean_dynamic_range() - 0.0).abs() < 1e-10);
        assert!((d.stats().mean_sparsity() - 0.0).abs() < 1e-10);
        assert!((d.stats().mean_grad_norm() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_input_length_tracking() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        d.encode(&[1.0, 2.0, 3.0]).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
        assert_eq!(d.stats().min_input_length, 3);
        assert_eq!(d.stats().max_input_length, 7);
    }

    // -----------------------------------------------------------------------
    // Configuration setters
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_image_size() {
        let mut d = Differentiable::new();
        d.set_image_size(16).unwrap();
        assert_eq!(d.image_size(), 16);
    }

    #[test]
    fn test_set_image_size_zero() {
        let mut d = Differentiable::new();
        assert!(d.set_image_size(0).is_err());
    }

    #[test]
    fn test_set_compute_jacobian() {
        let mut d = Differentiable::new();
        d.set_compute_jacobian(false);
        assert!(!d.is_jacobian_enabled());
        d.set_compute_jacobian(true);
        assert!(d.is_jacobian_enabled());
    }

    #[test]
    fn test_set_compute_sensitivity() {
        let mut d = Differentiable::new();
        d.set_compute_sensitivity(false);
        assert!(!d.is_sensitivity_enabled());
    }

    // -----------------------------------------------------------------------
    // Tick / Process
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick() {
        let mut d = Differentiable::new();
        assert_eq!(d.current_tick(), 0);
        d.tick();
        assert_eq!(d.current_tick(), 1);
    }

    #[test]
    fn test_process() {
        let d = Differentiable::new();
        assert!(d.process().is_ok());
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_ema() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        d.reset_ema();
        assert!(!d.ema_initialized);
        assert!((d.ema_dynamic_range() - 0.0).abs() < 1e-10);
        assert!(d.recent.is_empty());
        // Stats should be preserved
        assert_eq!(d.stats().total_encodings, 1);
    }

    #[test]
    fn test_reset() {
        let cfg = DifferentiableConfig {
            image_size: 4,
            ..Default::default()
        };
        let mut d = Differentiable::with_config(cfg).unwrap();
        d.encode(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        d.tick();
        d.reset();
        assert_eq!(d.stats().total_encodings, 0);
        assert_eq!(d.current_tick(), 0);
        assert!(!d.ema_initialized);
    }

    // -----------------------------------------------------------------------
    // GASF vs GADF produce different results
    // -----------------------------------------------------------------------

    #[test]
    fn test_gasf_vs_gadf_different() {
        let ts = vec![1.0, 2.0, 3.0, 4.0];

        let cfg_gasf = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GASF,
            ..Default::default()
        };
        let cfg_gadf = DifferentiableConfig {
            image_size: 4,
            gaf_type: DiffGafType::GADF,
            ..Default::default()
        };

        let mut d_gasf = Differentiable::with_config(cfg_gasf).unwrap();
        let mut d_gadf = Differentiable::with_config(cfg_gadf).unwrap();

        let enc_gasf = d_gasf.encode(&ts).unwrap();
        let enc_gadf = d_gadf.encode(&ts).unwrap();

        assert_ne!(enc_gasf.image, enc_gadf.image);
    }

    // -----------------------------------------------------------------------
    // Numerical gradient check (finite differences)
    // -----------------------------------------------------------------------

    #[test]
    fn test_jacobian_numerical_gasf() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            gaf_type: DiffGafType::GASF,
            compute_jacobian: true,
            compute_sensitivity: false,
            ..Default::default()
        };
        // Use 5 input values with image_size=3 so that PAA rescaling produces
        // interior normalised values (away from ±1 where the acos derivative
        // has a singularity, making both numerical and analytical Jacobians
        // unreliable).
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut d = Differentiable::with_config(cfg.clone()).unwrap();
        let enc = d.encode(&ts).unwrap();
        let jac = enc.jacobian.unwrap();

        // Numerical gradient by finite differences on the rescaled input
        let eps = 1e-6;
        let n = 3;
        for inp in 0..n {
            let mut ts_plus = enc.normalised.clone();
            ts_plus[inp] += eps;
            // Re-compute angles and image
            let angles_plus: Vec<f64> =
                ts_plus.iter().map(|&x| x.clamp(-1.0, 1.0).acos()).collect();
            let mut img_plus = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    img_plus[i * n + j] = (angles_plus[i] + angles_plus[j]).cos();
                }
            }

            let mut ts_minus = enc.normalised.clone();
            ts_minus[inp] -= eps;
            let angles_minus: Vec<f64> = ts_minus
                .iter()
                .map(|&x| x.clamp(-1.0, 1.0).acos())
                .collect();
            let mut img_minus = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    img_minus[i * n + j] = (angles_minus[i] + angles_minus[j]).cos();
                }
            }

            for pixel in 0..n * n {
                let numerical = (img_plus[pixel] - img_minus[pixel]) / (2.0 * eps);
                let analytical = jac[pixel * n + inp];
                assert!(
                    (numerical - analytical).abs() < 1e-4,
                    "GASF Jacobian mismatch at pixel={}, inp={}: numerical={}, analytical={}",
                    pixel,
                    inp,
                    numerical,
                    analytical
                );
            }
        }
    }

    #[test]
    fn test_jacobian_numerical_gadf() {
        let cfg = DifferentiableConfig {
            image_size: 3,
            gaf_type: DiffGafType::GADF,
            compute_jacobian: true,
            compute_sensitivity: false,
            ..Default::default()
        };
        // Use 5 values with image_size=3 so PAA rescaling produces interior
        // normalised values (away from ±1 where acos derivative is singular).
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut d = Differentiable::with_config(cfg.clone()).unwrap();
        let enc = d.encode(&ts).unwrap();
        let jac = enc.jacobian.unwrap();

        let eps = 1e-6;
        let n = 3;
        for inp in 0..n {
            let mut ts_plus = enc.normalised.clone();
            ts_plus[inp] += eps;
            let angles_plus: Vec<f64> =
                ts_plus.iter().map(|&x| x.clamp(-1.0, 1.0).acos()).collect();
            let mut img_plus = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    img_plus[i * n + j] = (angles_plus[i] - angles_plus[j]).sin();
                }
            }

            let mut ts_minus = enc.normalised.clone();
            ts_minus[inp] -= eps;
            let angles_minus: Vec<f64> = ts_minus
                .iter()
                .map(|&x| x.clamp(-1.0, 1.0).acos())
                .collect();
            let mut img_minus = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    img_minus[i * n + j] = (angles_minus[i] - angles_minus[j]).sin();
                }
            }

            for pixel in 0..n * n {
                let numerical = (img_plus[pixel] - img_minus[pixel]) / (2.0 * eps);
                let analytical = jac[pixel * n + inp];
                assert!(
                    (numerical - analytical).abs() < 1e-4,
                    "GADF Jacobian mismatch at pixel={}, inp={}: numerical={}, analytical={}",
                    pixel,
                    inp,
                    numerical,
                    analytical
                );
            }
        }
    }
}
