//! Saliency map generation
//!
//! Part of the Visual Cortex region - generates saliency maps that highlight
//! which input features most influenced model predictions using gradient-based methods.

use crate::common::Result;
use std::collections::HashMap;

/// Saliency method type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaliencyMethod {
    /// Vanilla gradient saliency
    VanillaGradient,
    /// Gradient times input
    GradientTimesInput,
    /// Integrated gradients
    IntegratedGradients,
    /// SmoothGrad (averaged gradients with noise)
    SmoothGrad,
    /// Guided backpropagation
    GuidedBackprop,
    /// DeconvNet style
    Deconvolution,
}

impl Default for SaliencyMethod {
    fn default() -> Self {
        Self::VanillaGradient
    }
}

/// Saliency configuration
#[derive(Debug, Clone)]
pub struct SaliencyConfig {
    /// Method to use for saliency computation
    pub method: SaliencyMethod,
    /// Number of samples for SmoothGrad
    pub smoothgrad_samples: usize,
    /// Noise standard deviation for SmoothGrad
    pub smoothgrad_noise_std: f64,
    /// Number of steps for integrated gradients
    pub integrated_steps: usize,
    /// Baseline type for integrated gradients
    pub baseline: BaselineType,
    /// Whether to take absolute value of gradients
    pub absolute_value: bool,
    /// Whether to normalize output
    pub normalize: bool,
    /// Aggregation method for multi-channel inputs
    pub channel_aggregation: ChannelAggregation,
    /// Clip percentile for outlier removal (0-100)
    pub clip_percentile: f64,
}

impl Default for SaliencyConfig {
    fn default() -> Self {
        Self {
            method: SaliencyMethod::VanillaGradient,
            smoothgrad_samples: 50,
            smoothgrad_noise_std: 0.1,
            integrated_steps: 50,
            baseline: BaselineType::Zero,
            absolute_value: true,
            normalize: true,
            channel_aggregation: ChannelAggregation::MaxAcrossChannels,
            clip_percentile: 99.0,
        }
    }
}

/// Baseline type for integrated gradients
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineType {
    /// Zero baseline (black image)
    Zero,
    /// Mean of input
    Mean,
    /// Random baseline
    Random,
    /// Gaussian blur of input
    Blurred,
    /// Uniform random baseline
    UniformRandom,
}

impl Default for BaselineType {
    fn default() -> Self {
        Self::Zero
    }
}

/// Channel aggregation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelAggregation {
    /// Sum across channels
    Sum,
    /// Mean across channels
    Mean,
    /// Max across channels
    MaxAcrossChannels,
    /// L2 norm across channels
    L2Norm,
    /// Keep all channels separate
    None,
}

impl Default for ChannelAggregation {
    fn default() -> Self {
        Self::MaxAcrossChannels
    }
}

/// Input tensor for saliency computation
#[derive(Debug, Clone)]
pub struct SaliencyInput {
    /// Input data (channels x height x width)
    pub data: Vec<Vec<Vec<f64>>>,
    /// Shape (channels, height, width)
    pub shape: (usize, usize, usize),
    /// Optional label for the input
    pub label: Option<String>,
}

impl SaliencyInput {
    /// Create from 3D data
    pub fn new(data: Vec<Vec<Vec<f64>>>) -> Self {
        let channels = data.len();
        let height = if channels > 0 { data[0].len() } else { 0 };
        let width = if height > 0 { data[0][0].len() } else { 0 };

        Self {
            data,
            shape: (channels, height, width),
            label: None,
        }
    }

    /// Create from 2D data (single channel)
    pub fn from_2d(data: Vec<Vec<f64>>) -> Self {
        let height = data.len();
        let width = if height > 0 { data[0].len() } else { 0 };

        Self {
            data: vec![data],
            shape: (1, height, width),
            label: None,
        }
    }

    /// Set label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get value at position
    pub fn get(&self, channel: usize, row: usize, col: usize) -> f64 {
        if channel < self.shape.0 && row < self.shape.1 && col < self.shape.2 {
            self.data[channel][row][col]
        } else {
            0.0
        }
    }

    /// Get mean value
    pub fn mean(&self) -> f64 {
        let total: f64 = self
            .data
            .iter()
            .flat_map(|c| c.iter())
            .flat_map(|r| r.iter())
            .sum();
        let count = self.shape.0 * self.shape.1 * self.shape.2;
        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Create zero baseline with same shape
    pub fn zero_baseline(&self) -> Self {
        let data = vec![vec![vec![0.0; self.shape.2]; self.shape.1]; self.shape.0];
        Self {
            data,
            shape: self.shape,
            label: None,
        }
    }

    /// Create mean baseline
    pub fn mean_baseline(&self) -> Self {
        let mean = self.mean();
        let data = vec![vec![vec![mean; self.shape.2]; self.shape.1]; self.shape.0];
        Self {
            data,
            shape: self.shape,
            label: None,
        }
    }

    /// Add Gaussian noise
    pub fn add_noise(&self, std: f64) -> Self {
        use std::f64::consts::PI;

        let mut data = self.data.clone();

        // Simple Box-Muller transform for Gaussian noise
        let mut seed = 42u64;
        for c in 0..self.shape.0 {
            for i in 0..self.shape.1 {
                for j in 0..self.shape.2 {
                    // Simple LCG random
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u1 = (seed as f64) / (u64::MAX as f64);
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u2 = (seed as f64) / (u64::MAX as f64);

                    let noise = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * PI * u2).cos() * std;
                    data[c][i][j] += noise;
                }
            }
        }

        Self {
            data,
            shape: self.shape,
            label: self.label.clone(),
        }
    }

    /// Interpolate between self and another input
    pub fn interpolate(&self, other: &Self, alpha: f64) -> Self {
        let mut data = vec![vec![vec![0.0; self.shape.2]; self.shape.1]; self.shape.0];

        for c in 0..self.shape.0.min(other.shape.0) {
            for i in 0..self.shape.1.min(other.shape.1) {
                for j in 0..self.shape.2.min(other.shape.2) {
                    data[c][i][j] =
                        other.data[c][i][j] + alpha * (self.data[c][i][j] - other.data[c][i][j]);
                }
            }
        }

        Self {
            data,
            shape: self.shape,
            label: None,
        }
    }
}

/// Gradient tensor from backward pass
#[derive(Debug, Clone)]
pub struct GradientTensor {
    /// Gradient data (channels x height x width)
    pub data: Vec<Vec<Vec<f64>>>,
    /// Shape (channels, height, width)
    pub shape: (usize, usize, usize),
    /// Target class for which gradient was computed
    pub target_class: Option<String>,
}

impl GradientTensor {
    /// Create new gradient tensor
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            data: vec![vec![vec![0.0; width]; height]; channels],
            shape: (channels, height, width),
            target_class: None,
        }
    }

    /// Create from 3D data
    pub fn from_data(data: Vec<Vec<Vec<f64>>>) -> Self {
        let channels = data.len();
        let height = if channels > 0 { data[0].len() } else { 0 };
        let width = if height > 0 { data[0][0].len() } else { 0 };

        Self {
            data,
            shape: (channels, height, width),
            target_class: None,
        }
    }

    /// Set target class
    pub fn with_target_class(mut self, class: impl Into<String>) -> Self {
        self.target_class = Some(class.into());
        self
    }

    /// Get value at position
    pub fn get(&self, channel: usize, row: usize, col: usize) -> f64 {
        if channel < self.shape.0 && row < self.shape.1 && col < self.shape.2 {
            self.data[channel][row][col]
        } else {
            0.0
        }
    }

    /// Set value at position
    pub fn set(&mut self, channel: usize, row: usize, col: usize, value: f64) {
        if channel < self.shape.0 && row < self.shape.1 && col < self.shape.2 {
            self.data[channel][row][col] = value;
        }
    }

    /// Add another gradient tensor
    pub fn add(&mut self, other: &GradientTensor) {
        for c in 0..self.shape.0.min(other.shape.0) {
            for i in 0..self.shape.1.min(other.shape.1) {
                for j in 0..self.shape.2.min(other.shape.2) {
                    self.data[c][i][j] += other.data[c][i][j];
                }
            }
        }
    }

    /// Scale by factor
    pub fn scale(&mut self, factor: f64) {
        for c in 0..self.shape.0 {
            for i in 0..self.shape.1 {
                for j in 0..self.shape.2 {
                    self.data[c][i][j] *= factor;
                }
            }
        }
    }

    /// Element-wise multiply with input
    pub fn multiply_with_input(&self, input: &SaliencyInput) -> GradientTensor {
        let mut result = GradientTensor::new(self.shape.0, self.shape.1, self.shape.2);

        for c in 0..self.shape.0.min(input.shape.0) {
            for i in 0..self.shape.1.min(input.shape.1) {
                for j in 0..self.shape.2.min(input.shape.2) {
                    result.data[c][i][j] = self.data[c][i][j] * input.data[c][i][j];
                }
            }
        }

        result
    }

    /// Apply ReLU (keep positive gradients only)
    pub fn apply_relu(&mut self) {
        for c in 0..self.shape.0 {
            for i in 0..self.shape.1 {
                for j in 0..self.shape.2 {
                    self.data[c][i][j] = self.data[c][i][j].max(0.0);
                }
            }
        }
    }

    /// Take absolute value
    pub fn abs(&mut self) {
        for c in 0..self.shape.0 {
            for i in 0..self.shape.1 {
                for j in 0..self.shape.2 {
                    self.data[c][i][j] = self.data[c][i][j].abs();
                }
            }
        }
    }
}

/// Saliency map result
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// Saliency values (height x width) - aggregated across channels
    pub map: Vec<Vec<f64>>,
    /// Shape (height, width)
    pub shape: (usize, usize),
    /// Method used to generate this map
    pub method: SaliencyMethod,
    /// Target class this saliency explains
    pub target_class: Option<String>,
    /// Normalization applied
    pub normalized: bool,
    /// Top salient regions
    pub salient_regions: Vec<SalientRegion>,
    /// Statistics about the map
    pub stats: SaliencyMapStats,
}

impl SaliencyMap {
    /// Get value at position
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.shape.0 && col < self.shape.1 {
            self.map[row][col]
        } else {
            0.0
        }
    }

    /// Get maximum saliency value
    pub fn max(&self) -> f64 {
        self.map
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get minimum saliency value
    pub fn min(&self) -> f64 {
        self.map
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get mean saliency value
    pub fn mean(&self) -> f64 {
        let total: f64 = self.map.iter().flat_map(|r| r.iter()).sum();
        total / (self.shape.0 * self.shape.1) as f64
    }

    /// Threshold the map
    pub fn threshold(&self, threshold: f64) -> SaliencyMap {
        let map: Vec<Vec<f64>> = self
            .map
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&v| if v >= threshold { v } else { 0.0 })
                    .collect()
            })
            .collect();

        SaliencyMap {
            map,
            shape: self.shape,
            method: self.method,
            target_class: self.target_class.clone(),
            normalized: self.normalized,
            salient_regions: Vec::new(),
            stats: self.stats.clone(),
        }
    }

    /// Resize map using bilinear interpolation
    pub fn resize(&self, target_height: usize, target_width: usize) -> SaliencyMap {
        let mut resized = vec![vec![0.0; target_width]; target_height];

        let h_scale = self.shape.0 as f64 / target_height as f64;
        let w_scale = self.shape.1 as f64 / target_width as f64;

        for i in 0..target_height {
            for j in 0..target_width {
                let src_y = i as f64 * h_scale;
                let src_x = j as f64 * w_scale;

                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(self.shape.0 - 1);
                let x1 = (x0 + 1).min(self.shape.1 - 1);

                let y_frac = src_y - y0 as f64;
                let x_frac = src_x - x0 as f64;

                let v00 = self.get(y0, x0);
                let v01 = self.get(y0, x1);
                let v10 = self.get(y1, x0);
                let v11 = self.get(y1, x1);

                resized[i][j] = v00 * (1.0 - y_frac) * (1.0 - x_frac)
                    + v01 * (1.0 - y_frac) * x_frac
                    + v10 * y_frac * (1.0 - x_frac)
                    + v11 * y_frac * x_frac;
            }
        }

        SaliencyMap {
            map: resized,
            shape: (target_height, target_width),
            method: self.method,
            target_class: self.target_class.clone(),
            normalized: self.normalized,
            salient_regions: Vec::new(),
            stats: self.stats.clone(),
        }
    }
}

/// Statistics about a saliency map
#[derive(Debug, Clone, Default)]
pub struct SaliencyMapStats {
    /// Maximum saliency value
    pub max_value: f64,
    /// Minimum saliency value
    pub min_value: f64,
    /// Mean saliency value
    pub mean_value: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Sparsity (fraction of near-zero values)
    pub sparsity: f64,
    /// Entropy of distribution
    pub entropy: f64,
}

/// A salient region identified in the map
#[derive(Debug, Clone)]
pub struct SalientRegion {
    /// Center position (row, col)
    pub center: (usize, usize),
    /// Bounding box (top, left, bottom, right)
    pub bbox: (usize, usize, usize, usize),
    /// Mean saliency in region
    pub mean_saliency: f64,
    /// Max saliency in region
    pub max_saliency: f64,
    /// Area in pixels
    pub area: usize,
    /// Optional label
    pub label: Option<String>,
}

impl SalientRegion {
    /// Check if point is inside region
    pub fn contains(&self, row: usize, col: usize) -> bool {
        row >= self.bbox.0 && row <= self.bbox.2 && col >= self.bbox.1 && col <= self.bbox.3
    }
}

/// Saliency generator statistics
#[derive(Debug, Clone, Default)]
pub struct SaliencyStats {
    /// Total maps generated
    pub maps_generated: u64,
    /// Maps by method
    pub by_method: HashMap<String, u64>,
    /// Average generation time in microseconds
    pub avg_generation_time_us: f64,
    /// Average map entropy
    pub avg_entropy: f64,
    /// Average sparsity
    pub avg_sparsity: f64,
}

/// Saliency map generation system
pub struct Saliency {
    /// Configuration
    config: SaliencyConfig,
    /// Statistics
    stats: SaliencyStats,
    /// Gradient function (simulated)
    gradient_fn: Option<Box<dyn Fn(&SaliencyInput) -> GradientTensor + Send + Sync>>,
}

impl Default for Saliency {
    fn default() -> Self {
        Self::new()
    }
}

impl Saliency {
    /// Create a new saliency generator
    pub fn new() -> Self {
        Self::with_config(SaliencyConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SaliencyConfig) -> Self {
        Self {
            config,
            stats: SaliencyStats::default(),
            gradient_fn: None,
        }
    }

    /// Set saliency method
    pub fn with_method(mut self, method: SaliencyMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Enable/disable absolute value
    pub fn with_absolute_value(mut self, abs: bool) -> Self {
        self.config.absolute_value = abs;
        self
    }

    /// Set channel aggregation method
    pub fn with_aggregation(mut self, aggregation: ChannelAggregation) -> Self {
        self.config.channel_aggregation = aggregation;
        self
    }

    /// Set gradient function (for real model integration)
    pub fn set_gradient_fn<F>(&mut self, f: F)
    where
        F: Fn(&SaliencyInput) -> GradientTensor + Send + Sync + 'static,
    {
        self.gradient_fn = Some(Box::new(f));
    }

    /// Compute saliency map using configured method
    pub fn compute(&mut self, input: &SaliencyInput) -> Result<SaliencyMap> {
        let start = std::time::Instant::now();

        let result = match self.config.method {
            SaliencyMethod::VanillaGradient => self.vanilla_gradient(input),
            SaliencyMethod::GradientTimesInput => self.gradient_times_input(input),
            SaliencyMethod::IntegratedGradients => self.integrated_gradients(input),
            SaliencyMethod::SmoothGrad => self.smoothgrad(input),
            SaliencyMethod::GuidedBackprop => self.guided_backprop(input),
            SaliencyMethod::Deconvolution => self.deconvolution(input),
        };

        // Update statistics
        let elapsed = start.elapsed().as_micros() as f64;
        self.stats.maps_generated += 1;
        self.stats.avg_generation_time_us = self.stats.avg_generation_time_us * 0.9 + elapsed * 0.1;

        let method_key = format!("{:?}", self.config.method);
        *self.stats.by_method.entry(method_key).or_insert(0) += 1;

        if let Ok(ref map) = result {
            self.stats.avg_entropy = self.stats.avg_entropy * 0.9 + map.stats.entropy * 0.1;
            self.stats.avg_sparsity = self.stats.avg_sparsity * 0.9 + map.stats.sparsity * 0.1;
        }

        result
    }

    /// Compute saliency from provided gradients
    pub fn compute_from_gradient(
        &mut self,
        input: &SaliencyInput,
        gradient: &GradientTensor,
    ) -> Result<SaliencyMap> {
        let mut grad = gradient.clone();

        // Apply method-specific processing
        match self.config.method {
            SaliencyMethod::GradientTimesInput => {
                grad = grad.multiply_with_input(input);
            }
            SaliencyMethod::GuidedBackprop => {
                grad.apply_relu();
            }
            _ => {}
        }

        // Apply absolute value if configured
        if self.config.absolute_value {
            grad.abs();
        }

        // Aggregate across channels
        let map = self.aggregate_channels(&grad);

        // Post-process
        let map = self.post_process(map);

        // Compute statistics
        let stats = self.compute_stats(&map);

        // Find salient regions
        let salient_regions = self.find_salient_regions(&map, 0.5);

        Ok(SaliencyMap {
            map,
            shape: (gradient.shape.1, gradient.shape.2),
            method: self.config.method,
            target_class: gradient.target_class.clone(),
            normalized: self.config.normalize,
            salient_regions,
            stats,
        })
    }

    /// Vanilla gradient saliency
    fn vanilla_gradient(&self, input: &SaliencyInput) -> Result<SaliencyMap> {
        let gradient = self.get_gradient(input);

        let mut grad = gradient.clone();
        if self.config.absolute_value {
            grad.abs();
        }

        let map = self.aggregate_channels(&grad);
        let map = self.post_process(map);
        let stats = self.compute_stats(&map);
        let salient_regions = self.find_salient_regions(&map, 0.5);

        Ok(SaliencyMap {
            map,
            shape: (input.shape.1, input.shape.2),
            method: SaliencyMethod::VanillaGradient,
            target_class: gradient.target_class,
            normalized: self.config.normalize,
            salient_regions,
            stats,
        })
    }

    /// Gradient times input
    fn gradient_times_input(&self, input: &SaliencyInput) -> Result<SaliencyMap> {
        let gradient = self.get_gradient(input);
        let mut grad = gradient.multiply_with_input(input);

        if self.config.absolute_value {
            grad.abs();
        }

        let map = self.aggregate_channels(&grad);
        let map = self.post_process(map);
        let stats = self.compute_stats(&map);
        let salient_regions = self.find_salient_regions(&map, 0.5);

        Ok(SaliencyMap {
            map,
            shape: (input.shape.1, input.shape.2),
            method: SaliencyMethod::GradientTimesInput,
            target_class: gradient.target_class,
            normalized: self.config.normalize,
            salient_regions,
            stats,
        })
    }

    /// Integrated gradients
    fn integrated_gradients(&self, input: &SaliencyInput) -> Result<SaliencyMap> {
        let baseline = match self.config.baseline {
            BaselineType::Zero => input.zero_baseline(),
            BaselineType::Mean => input.mean_baseline(),
            _ => input.zero_baseline(),
        };

        let mut accumulated = GradientTensor::new(input.shape.0, input.shape.1, input.shape.2);

        // Riemann sum approximation
        for step in 0..self.config.integrated_steps {
            let alpha = (step as f64 + 0.5) / self.config.integrated_steps as f64;
            let interpolated = baseline.interpolate(input, alpha);
            let grad = self.get_gradient(&interpolated);
            accumulated.add(&grad);
        }

        // Scale by step size and multiply by (input - baseline)
        accumulated.scale(1.0 / self.config.integrated_steps as f64);

        // Multiply by (input - baseline)
        let mut delta_input = input.clone();
        for c in 0..input.shape.0.min(baseline.shape.0) {
            for i in 0..input.shape.1.min(baseline.shape.1) {
                for j in 0..input.shape.2.min(baseline.shape.2) {
                    delta_input.data[c][i][j] -= baseline.data[c][i][j];
                }
            }
        }

        let delta_tensor = GradientTensor::from_data(delta_input.data);
        let mut grad = accumulated.multiply_with_input(&SaliencyInput::new(delta_tensor.data));

        if self.config.absolute_value {
            grad.abs();
        }

        let map = self.aggregate_channels(&grad);
        let map = self.post_process(map);
        let stats = self.compute_stats(&map);
        let salient_regions = self.find_salient_regions(&map, 0.5);

        Ok(SaliencyMap {
            map,
            shape: (input.shape.1, input.shape.2),
            method: SaliencyMethod::IntegratedGradients,
            target_class: None,
            normalized: self.config.normalize,
            salient_regions,
            stats,
        })
    }

    /// SmoothGrad
    fn smoothgrad(&self, input: &SaliencyInput) -> Result<SaliencyMap> {
        let mut accumulated = GradientTensor::new(input.shape.0, input.shape.1, input.shape.2);

        for _ in 0..self.config.smoothgrad_samples {
            let noisy = input.add_noise(self.config.smoothgrad_noise_std);
            let grad = self.get_gradient(&noisy);
            accumulated.add(&grad);
        }

        accumulated.scale(1.0 / self.config.smoothgrad_samples as f64);

        if self.config.absolute_value {
            accumulated.abs();
        }

        let map = self.aggregate_channels(&accumulated);
        let map = self.post_process(map);
        let stats = self.compute_stats(&map);
        let salient_regions = self.find_salient_regions(&map, 0.5);

        Ok(SaliencyMap {
            map,
            shape: (input.shape.1, input.shape.2),
            method: SaliencyMethod::SmoothGrad,
            target_class: None,
            normalized: self.config.normalize,
            salient_regions,
            stats,
        })
    }

    /// Guided backpropagation
    fn guided_backprop(&self, input: &SaliencyInput) -> Result<SaliencyMap> {
        let mut gradient = self.get_gradient(input);

        // Guided backprop: mask gradients where either input or gradient is negative
        for c in 0..gradient.shape.0.min(input.shape.0) {
            for i in 0..gradient.shape.1.min(input.shape.1) {
                for j in 0..gradient.shape.2.min(input.shape.2) {
                    if input.get(c, i, j) < 0.0 || gradient.get(c, i, j) < 0.0 {
                        gradient.set(c, i, j, 0.0);
                    }
                }
            }
        }

        if self.config.absolute_value {
            gradient.abs();
        }

        let map = self.aggregate_channels(&gradient);
        let map = self.post_process(map);
        let stats = self.compute_stats(&map);
        let salient_regions = self.find_salient_regions(&map, 0.5);

        Ok(SaliencyMap {
            map,
            shape: (input.shape.1, input.shape.2),
            method: SaliencyMethod::GuidedBackprop,
            target_class: gradient.target_class,
            normalized: self.config.normalize,
            salient_regions,
            stats,
        })
    }

    /// Deconvolution
    fn deconvolution(&self, input: &SaliencyInput) -> Result<SaliencyMap> {
        let mut gradient = self.get_gradient(input);

        // Deconv: only backprop positive gradients (ignore input)
        gradient.apply_relu();

        if self.config.absolute_value {
            gradient.abs();
        }

        let map = self.aggregate_channels(&gradient);
        let map = self.post_process(map);
        let stats = self.compute_stats(&map);
        let salient_regions = self.find_salient_regions(&map, 0.5);

        Ok(SaliencyMap {
            map,
            shape: (input.shape.1, input.shape.2),
            method: SaliencyMethod::Deconvolution,
            target_class: gradient.target_class,
            normalized: self.config.normalize,
            salient_regions,
            stats,
        })
    }

    /// Get gradient (uses gradient_fn if set, otherwise simulates)
    fn get_gradient(&self, input: &SaliencyInput) -> GradientTensor {
        if let Some(ref f) = self.gradient_fn {
            f(input)
        } else {
            // Simulate gradient based on input values
            self.simulate_gradient(input)
        }
    }

    /// Simulate gradient for testing
    fn simulate_gradient(&self, input: &SaliencyInput) -> GradientTensor {
        let mut gradient = GradientTensor::new(input.shape.0, input.shape.1, input.shape.2);

        // Simple simulation: gradient proportional to input magnitude with some variation
        let mean = input.mean();

        for c in 0..input.shape.0 {
            for i in 0..input.shape.1 {
                for j in 0..input.shape.2 {
                    let val = input.get(c, i, j);
                    // Gradient higher where input deviates from mean
                    let grad = (val - mean).abs() + 0.1 * val;
                    gradient.set(c, i, j, grad);
                }
            }
        }

        gradient
    }

    /// Aggregate gradients across channels
    fn aggregate_channels(&self, gradient: &GradientTensor) -> Vec<Vec<f64>> {
        let (channels, height, width) = gradient.shape;
        let mut result = vec![vec![0.0; width]; height];

        match self.config.channel_aggregation {
            ChannelAggregation::Sum => {
                for c in 0..channels {
                    for i in 0..height {
                        for j in 0..width {
                            result[i][j] += gradient.get(c, i, j);
                        }
                    }
                }
            }
            ChannelAggregation::Mean => {
                for c in 0..channels {
                    for i in 0..height {
                        for j in 0..width {
                            result[i][j] += gradient.get(c, i, j);
                        }
                    }
                }
                if channels > 0 {
                    for row in &mut result {
                        for val in row {
                            *val /= channels as f64;
                        }
                    }
                }
            }
            ChannelAggregation::MaxAcrossChannels => {
                for i in 0..height {
                    for j in 0..width {
                        let mut max_val = f64::NEG_INFINITY;
                        for c in 0..channels {
                            max_val = max_val.max(gradient.get(c, i, j));
                        }
                        result[i][j] = max_val;
                    }
                }
            }
            ChannelAggregation::L2Norm => {
                for i in 0..height {
                    for j in 0..width {
                        let mut sum_sq = 0.0;
                        for c in 0..channels {
                            let v = gradient.get(c, i, j);
                            sum_sq += v * v;
                        }
                        result[i][j] = sum_sq.sqrt();
                    }
                }
            }
            ChannelAggregation::None => {
                // Just use first channel
                for i in 0..height {
                    for j in 0..width {
                        result[i][j] = gradient.get(0, i, j);
                    }
                }
            }
        }

        result
    }

    /// Post-process map (clip, normalize)
    fn post_process(&self, mut map: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let height = map.len();
        let width = if height > 0 { map[0].len() } else { 0 };

        // Clip to percentile
        if self.config.clip_percentile < 100.0 {
            let mut all_values: Vec<f64> = map.iter().flat_map(|r| r.iter()).copied().collect();
            all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let clip_idx = ((all_values.len() as f64 * self.config.clip_percentile / 100.0)
                as usize)
                .min(all_values.len() - 1);
            let clip_val = all_values[clip_idx];

            for row in &mut map {
                for val in row {
                    *val = val.min(clip_val);
                }
            }
        }

        // Normalize to 0-1
        if self.config.normalize {
            let min_val = map
                .iter()
                .flat_map(|r| r.iter())
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max_val = map
                .iter()
                .flat_map(|r| r.iter())
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            let range = max_val - min_val;
            if range > 0.0 {
                for row in &mut map {
                    for val in row {
                        *val = (*val - min_val) / range;
                    }
                }
            }
        }

        map
    }

    /// Compute statistics for a map
    fn compute_stats(&self, map: &[Vec<f64>]) -> SaliencyMapStats {
        let all_values: Vec<f64> = map.iter().flat_map(|r| r.iter()).copied().collect();
        let n = all_values.len() as f64;

        if n == 0.0 {
            return SaliencyMapStats::default();
        }

        let min_value = all_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_value = all_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_value: f64 = all_values.iter().sum::<f64>() / n;

        let variance: f64 = all_values
            .iter()
            .map(|v| (v - mean_value).powi(2))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        // Sparsity: fraction of values near zero
        let threshold = 0.1;
        let sparse_count = all_values.iter().filter(|&&v| v < threshold).count();
        let sparsity = sparse_count as f64 / n;

        // Entropy (discretize to 100 bins)
        let mut histogram = vec![0u64; 100];
        for &v in &all_values {
            let bin = ((v * 99.0).floor() as usize).min(99);
            histogram[bin] += 1;
        }

        let mut entropy = 0.0;
        for count in histogram {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.log2();
            }
        }

        SaliencyMapStats {
            max_value,
            min_value,
            mean_value,
            std_dev,
            sparsity,
            entropy,
        }
    }

    /// Find salient regions above threshold
    fn find_salient_regions(&self, map: &[Vec<f64>], threshold: f64) -> Vec<SalientRegion> {
        let height = map.len();
        let width = if height > 0 { map[0].len() } else { 0 };

        let mut visited = vec![vec![false; width]; height];
        let mut regions = Vec::new();

        for i in 0..height {
            for j in 0..width {
                if !visited[i][j] && map[i][j] >= threshold {
                    // Flood fill to find connected region
                    let mut min_row = i;
                    let mut max_row = i;
                    let mut min_col = j;
                    let mut max_col = j;
                    let mut sum = 0.0;
                    let mut max_val = 0.0;
                    let mut count = 0;

                    let mut queue = vec![(i, j)];
                    visited[i][j] = true;

                    while let Some((ci, cj)) = queue.pop() {
                        min_row = min_row.min(ci);
                        max_row = max_row.max(ci);
                        min_col = min_col.min(cj);
                        max_col = max_col.max(cj);
                        sum += map[ci][cj];
                        max_val = max_val.max(map[ci][cj]);
                        count += 1;

                        for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                            let ni = (ci as i32 + di) as usize;
                            let nj = (cj as i32 + dj) as usize;

                            if ni < height
                                && nj < width
                                && !visited[ni][nj]
                                && map[ni][nj] >= threshold
                            {
                                visited[ni][nj] = true;
                                queue.push((ni, nj));
                            }
                        }
                    }

                    if count >= 4 {
                        regions.push(SalientRegion {
                            center: ((min_row + max_row) / 2, (min_col + max_col) / 2),
                            bbox: (min_row, min_col, max_row, max_col),
                            mean_saliency: sum / count as f64,
                            max_saliency: max_val,
                            area: count,
                            label: None,
                        });
                    }
                }
            }
        }

        regions.sort_by(|a, b| {
            b.mean_saliency
                .partial_cmp(&a.mean_saliency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        regions
    }

    /// Get statistics
    pub fn stats(&self) -> &SaliencyStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &SaliencyConfig {
        &self.config
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_input() -> SaliencyInput {
        let data = vec![vec![vec![0.5; 8]; 8]; 3];
        SaliencyInput::new(data)
    }

    #[test]
    fn test_saliency_creation() {
        let saliency = Saliency::new();
        assert_eq!(saliency.config.method, SaliencyMethod::VanillaGradient);
    }

    #[test]
    fn test_input_creation() {
        let input = create_test_input();
        assert_eq!(input.shape, (3, 8, 8));
    }

    #[test]
    fn test_input_mean() {
        let input = create_test_input();
        assert!((input.mean() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_input_interpolate() {
        let input1 = SaliencyInput::from_2d(vec![vec![0.0; 4]; 4]);
        let input2 = SaliencyInput::from_2d(vec![vec![1.0; 4]; 4]);

        let mid = input1.interpolate(&input2, 0.5);
        assert!((mid.get(0, 0, 0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_gradient_tensor() {
        let mut grad = GradientTensor::new(3, 4, 4);
        grad.set(0, 1, 1, 0.5);
        assert!((grad.get(0, 1, 1) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_gradient_abs() {
        let mut grad = GradientTensor::from_data(vec![vec![vec![-1.0, 1.0]]]);
        grad.abs();
        assert!((grad.get(0, 0, 0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_vanilla_gradient() {
        let mut saliency = Saliency::new().with_method(SaliencyMethod::VanillaGradient);
        let input = create_test_input();

        let result = saliency.compute(&input).unwrap();
        assert_eq!(result.shape, (8, 8));
    }

    #[test]
    fn test_gradient_times_input() {
        let mut saliency = Saliency::new().with_method(SaliencyMethod::GradientTimesInput);
        let input = create_test_input();

        let result = saliency.compute(&input).unwrap();
        assert_eq!(result.method, SaliencyMethod::GradientTimesInput);
    }

    #[test]
    fn test_smoothgrad() {
        let mut config = SaliencyConfig::default();
        config.method = SaliencyMethod::SmoothGrad;
        config.smoothgrad_samples = 5; // Fewer samples for testing
        let mut saliency = Saliency::with_config(config);

        let input = create_test_input();
        let result = saliency.compute(&input).unwrap();
        assert_eq!(result.method, SaliencyMethod::SmoothGrad);
    }

    #[test]
    fn test_integrated_gradients() {
        let mut config = SaliencyConfig::default();
        config.method = SaliencyMethod::IntegratedGradients;
        config.integrated_steps = 5; // Fewer steps for testing
        let mut saliency = Saliency::with_config(config);

        let input = create_test_input();
        let result = saliency.compute(&input).unwrap();
        assert_eq!(result.method, SaliencyMethod::IntegratedGradients);
    }

    #[test]
    fn test_saliency_map_stats() {
        let mut saliency = Saliency::new();
        let input = create_test_input();

        let result = saliency.compute(&input).unwrap();
        assert!(result.stats.max_value >= result.stats.min_value);
    }

    #[test]
    fn test_saliency_map_resize() {
        let mut saliency = Saliency::new();
        let input = create_test_input();

        let result = saliency.compute(&input).unwrap();
        let resized = result.resize(16, 16);
        assert_eq!(resized.shape, (16, 16));
    }

    #[test]
    fn test_saliency_map_threshold() {
        let mut saliency = Saliency::new();
        let input = create_test_input();

        let result = saliency.compute(&input).unwrap();
        let thresholded = result.threshold(0.5);
        assert_eq!(thresholded.shape, result.shape);
    }

    #[test]
    fn test_aggregation_methods() {
        let input = create_test_input();

        for aggregation in [
            ChannelAggregation::Sum,
            ChannelAggregation::Mean,
            ChannelAggregation::MaxAcrossChannels,
            ChannelAggregation::L2Norm,
        ] {
            let mut saliency = Saliency::new().with_aggregation(aggregation);
            let result = saliency.compute(&input).unwrap();
            assert_eq!(result.shape, (8, 8));
        }
    }

    #[test]
    fn test_statistics_tracking() {
        let mut saliency = Saliency::new();
        let input = create_test_input();

        for _ in 0..5 {
            saliency.compute(&input).unwrap();
        }

        assert_eq!(saliency.stats().maps_generated, 5);
    }

    #[test]
    fn test_salient_region() {
        let region = SalientRegion {
            center: (5, 5),
            bbox: (0, 0, 10, 10),
            mean_saliency: 0.8,
            max_saliency: 1.0,
            area: 100,
            label: None,
        };

        assert!(region.contains(5, 5));
        assert!(!region.contains(15, 15));
    }

    #[test]
    fn test_process_compatibility() {
        let saliency = Saliency::new();
        assert!(saliency.process().is_ok());
    }
}
