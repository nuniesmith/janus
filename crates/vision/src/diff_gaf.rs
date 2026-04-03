//! Differentiable Gramian Angular Field (DiffGAF) implementation.
//!
//! This module implements GAF as a differentiable neural network module using Candle,
//! enabling end-to-end gradient flow from the loss function back through the
//! time-series-to-image transformation.
//!
//! Key features:
//! - Learnable affine normalization parameters (γ, β) for adaptive scaling
//! - Full differentiability via Candle's autograd
//! - Support for both GASF (Gramian Angular Summation Field) and GADF (Gramian Angular Difference Field)
//! - Batch processing for efficient GPU utilization
//!
//! Mathematical formulation:
//! 1. Learnable normalization: x̃ = tanh(γ * (x - μ) / σ + β)
//! 2. Polar encoding: φ = arccos(x̃)
//! 3. GASF: G[i,j] = cos(φ_i + φ_j) = x̃_i * x̃_j - √(1-x̃_i²) * √(1-x̃_j²)
//! 4. GADF: G[i,j] = sin(φ_i - φ_j) = √(1-x̃_i²) * x̃_j - x̃_i * √(1-x̃_j²)

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Init, Module, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

/// Configuration for the DiffGAF module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffGafConfig {
    /// Number of input features (e.g., OHLCV = 5)
    pub num_features: usize,
    /// Output image size (GAF will be image_size x image_size)
    pub image_size: usize,
    /// GAF method to use
    pub method: DiffGafMethod,
    /// Whether to use learnable normalization parameters
    pub learnable_norm: bool,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl Default for DiffGafConfig {
    fn default() -> Self {
        Self {
            num_features: 5, // OHLCV
            image_size: 64,
            method: DiffGafMethod::Summation,
            learnable_norm: true,
            eps: 1e-7,
        }
    }
}

/// GAF method type - determines the angular operation used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffGafMethod {
    /// Gramian Angular Summation Field: cos(φ_i + φ_j)
    Summation,
    /// Gramian Angular Difference Field: sin(φ_i - φ_j)
    Difference,
}

/// Differentiable Gramian Angular Field module.
///
/// Transforms time series data into 2D images while maintaining full differentiability
/// for end-to-end training. The transformation encodes temporal correlations as
/// angular relationships in polar coordinates.
///
/// # Architecture
///
/// ```text
/// Input: (batch, time, features)
///    ↓
/// Learnable Normalization: x̃ = tanh(γ * normalize(x) + β)
///    ↓
/// Resize to image_size (if needed)
///    ↓
/// Polar Encoding: compute sin/cos values
///    ↓
/// Gramian Matrix: outer product in angular space
///    ↓
/// Output: (batch, features, image_size, image_size)
/// ```
pub struct DiffGaf {
    /// Learnable scale parameter γ (per feature)
    gamma: Option<Tensor>,
    /// Learnable shift parameter β (per feature)
    beta: Option<Tensor>,
    /// Configuration
    config: DiffGafConfig,
    /// Device for computation
    device: Device,
}

impl DiffGaf {
    /// Create a new DiffGAF module with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Configuration for the GAF transformation
    /// * `vb` - Variable builder for creating learnable parameters
    ///
    /// # Returns
    /// A new DiffGaf instance
    pub fn new(config: DiffGafConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        let (gamma, beta) = if config.learnable_norm {
            // Initialize gamma to 1.0 (identity scaling)
            let gamma = vb.get_with_hints(config.num_features, "gamma", Init::Const(1.0))?;

            // Initialize beta to 0.0 (no shift)
            let beta = vb.get_with_hints(config.num_features, "beta", Init::Const(0.0))?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        Ok(Self {
            gamma,
            beta,
            config,
            device,
        })
    }

    /// Create a DiffGAF module without learnable parameters.
    ///
    /// Useful for inference-only scenarios or when using pre-computed normalization.
    pub fn new_fixed(config: DiffGafConfig, device: &Device) -> Result<Self> {
        let mut config = config;
        config.learnable_norm = false;

        Ok(Self {
            gamma: None,
            beta: None,
            config,
            device: device.clone(),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &DiffGafConfig {
        &self.config
    }

    /// Normalize input tensor to [-1, 1] range with optional learnable parameters.
    ///
    /// Formula: x̃ = tanh(γ * (x - μ) / σ + β)
    ///
    /// The tanh ensures values are strictly bounded in [-1, 1], which is required
    /// for the subsequent arccos operation.
    fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: (batch, time, features)
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(candle_core::Error::Msg(format!(
                "Expected 3D tensor (batch, time, features), got {}D",
                dims.len()
            )));
        }

        // Compute mean and std along the time dimension (dim=1)
        let mean = x.mean_keepdim(1)?;
        let variance = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(1)?;
        let std = (variance + self.config.eps)?.sqrt()?;

        // Standardize: (x - mean) / std
        let standardized = x.broadcast_sub(&mean)?.broadcast_div(&std)?;

        // Apply learnable affine transformation if enabled
        let scaled = if let (Some(gamma), Some(beta)) = (&self.gamma, &self.beta) {
            // Reshape gamma and beta for broadcasting: (1, 1, features)
            let gamma = gamma.reshape((1, 1, self.config.num_features))?;
            let beta = beta.reshape((1, 1, self.config.num_features))?;

            standardized.broadcast_mul(&gamma)?.broadcast_add(&beta)?
        } else {
            standardized
        };

        // Apply tanh to bound values in [-1, 1]
        scaled.tanh()
    }

    /// Resize time series to target image size using linear interpolation.
    ///
    /// This is implemented as a differentiable operation using matrix multiplication
    /// with an interpolation weight matrix.
    fn resize(&self, x: &Tensor, target_size: usize) -> Result<Tensor> {
        let dims = x.dims();
        let time_len = dims[1];

        if time_len == target_size {
            return Ok(x.clone());
        }

        // Create interpolation weight matrix
        let mut weights = vec![0.0f32; time_len * target_size];

        for i in 0..target_size {
            let pos = (i as f64) * ((time_len - 1) as f64) / ((target_size - 1) as f64);
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;

            if idx + 1 < time_len {
                weights[i * time_len + idx] = 1.0 - frac;
                weights[i * time_len + idx + 1] = frac;
            } else {
                weights[i * time_len + idx] = 1.0;
            }
        }

        let weight_tensor = Tensor::from_vec(weights, (target_size, time_len), &self.device)?
            .to_dtype(x.dtype())?;

        // x shape: (batch, time, features)
        // weight shape: (target_size, time)
        // We need to apply: result[b, t_new, f] = sum_t(weight[t_new, t] * x[b, t, f])

        // Transpose x to (batch, features, time) for easier matmul
        let x_t = x.transpose(1, 2)?;

        // Apply interpolation: (batch, features, time) @ (time, target_size)^T
        let weight_t = weight_tensor.t()?;
        let resized = x_t.broadcast_matmul(&weight_t)?;

        // Transpose back to (batch, target_size, features)
        resized.transpose(1, 2)
    }

    /// Compute GASF: Gramian Angular Summation Field.
    ///
    /// Formula: GASF[i,j] = cos(φ_i + φ_j) = x̃_i * x̃_j - sin(φ_i) * sin(φ_j)
    /// where sin(φ) = √(1 - x̃²)
    fn compute_gasf(&self, normalized: &Tensor) -> Result<Tensor> {
        // normalized shape: (batch, image_size, features)
        let dims = normalized.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let num_features = dims[2];

        // Compute sin(φ) = sqrt(1 - x^2), clamped for numerical stability
        let x_squared = normalized.sqr()?;
        let one_minus_x_squared = (1.0 - self.config.eps - x_squared)?.clamp(0.0, 1.0)?;
        let sin_phi = one_minus_x_squared.sqrt()?;

        // Reshape for outer product computation
        // x_i shape: (batch, seq_len, 1, features)
        // x_j shape: (batch, 1, seq_len, features)
        let x_i = normalized.reshape((batch_size, seq_len, 1, num_features))?;
        let x_j = normalized.reshape((batch_size, 1, seq_len, num_features))?;

        let sin_i = sin_phi.reshape((batch_size, seq_len, 1, num_features))?;
        let sin_j = sin_phi.reshape((batch_size, 1, seq_len, num_features))?;

        // GASF = x_i * x_j - sin_i * sin_j
        // Result shape: (batch, seq_len, seq_len, features)
        let cos_sum = x_i.broadcast_mul(&x_j)?;
        let sin_prod = sin_i.broadcast_mul(&sin_j)?;
        let gasf = cos_sum.broadcast_sub(&sin_prod)?;

        // Permute to (batch, features, seq_len, seq_len) for image format
        gasf.permute((0, 3, 1, 2))
    }

    /// Compute GADF: Gramian Angular Difference Field.
    ///
    /// Formula: GADF[i,j] = sin(φ_i - φ_j) = sin(φ_i) * cos(φ_j) - cos(φ_i) * sin(φ_j)
    ///                    = √(1-x̃_i²) * x̃_j - x̃_i * √(1-x̃_j²)
    fn compute_gadf(&self, normalized: &Tensor) -> Result<Tensor> {
        // normalized shape: (batch, image_size, features)
        let dims = normalized.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let num_features = dims[2];

        // Compute sin(φ) = sqrt(1 - x^2)
        let x_squared = normalized.sqr()?;
        let one_minus_x_squared = (1.0 - self.config.eps - x_squared)?.clamp(0.0, 1.0)?;
        let sin_phi = one_minus_x_squared.sqrt()?;

        // Reshape for outer product computation
        let x_i = normalized.reshape((batch_size, seq_len, 1, num_features))?;
        let x_j = normalized.reshape((batch_size, 1, seq_len, num_features))?;

        let sin_i = sin_phi.reshape((batch_size, seq_len, 1, num_features))?;
        let sin_j = sin_phi.reshape((batch_size, 1, seq_len, num_features))?;

        // GADF = sin_i * x_j - x_i * sin_j
        // This computes sin(φ_i - φ_j) = sin(φ_i)cos(φ_j) - cos(φ_i)sin(φ_j)
        // Since cos(φ) = x̃ and sin(φ) = √(1-x̃²)
        let term1 = sin_i.broadcast_mul(&x_j)?;
        let term2 = x_i.broadcast_mul(&sin_j)?;
        let gadf = term1.broadcast_sub(&term2)?;

        // Permute to (batch, features, seq_len, seq_len) for image format
        gadf.permute((0, 3, 1, 2))
    }

    /// Forward pass: transform time series to GAF images.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, time, features)
    ///
    /// # Returns
    /// GAF images of shape (batch, features, image_size, image_size)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Step 1: Normalize with learnable parameters
        let normalized = self.normalize(x)?;

        // Step 2: Resize to target image size
        let resized = self.resize(&normalized, self.config.image_size)?;

        // Step 3: Compute Gramian matrix
        match self.config.method {
            DiffGafMethod::Summation => self.compute_gasf(&resized),
            DiffGafMethod::Difference => self.compute_gadf(&resized),
        }
    }

    /// Encode multiple time series with different GAF methods and concatenate.
    ///
    /// This produces a richer representation by combining both GASF and GADF.
    ///
    /// # Returns
    /// Tensor of shape (batch, features * 2, image_size, image_size)
    pub fn forward_dual(&self, x: &Tensor) -> Result<Tensor> {
        let normalized = self.normalize(x)?;
        let resized = self.resize(&normalized, self.config.image_size)?;

        let gasf = self.compute_gasf(&resized)?;
        let gadf = self.compute_gadf(&resized)?;

        // Concatenate along the channel dimension
        Tensor::cat(&[&gasf, &gadf], 1)
    }
}

impl Module for DiffGaf {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        DiffGaf::forward(self, x)
    }
}

/// Builder for creating DiffGAF with custom parameters.
pub struct DiffGafBuilder {
    config: DiffGafConfig,
}

impl DiffGafBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: DiffGafConfig::default(),
        }
    }

    /// Set the number of input features.
    pub fn num_features(mut self, n: usize) -> Self {
        self.config.num_features = n;
        self
    }

    /// Set the output image size.
    pub fn image_size(mut self, size: usize) -> Self {
        self.config.image_size = size;
        self
    }

    /// Set the GAF method.
    pub fn method(mut self, method: DiffGafMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Enable or disable learnable normalization.
    pub fn learnable_norm(mut self, learnable: bool) -> Self {
        self.config.learnable_norm = learnable;
        self
    }

    /// Build the DiffGAF module.
    pub fn build(self, vb: VarBuilder) -> Result<DiffGaf> {
        DiffGaf::new(self.config, vb)
    }

    /// Build without learnable parameters.
    pub fn build_fixed(self, device: &Device) -> Result<DiffGaf> {
        DiffGaf::new_fixed(self.config, device)
    }
}

impl Default for DiffGafBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create a VarMap and VarBuilder for DiffGAF.
pub fn create_diff_gaf(config: DiffGafConfig, device: &Device) -> Result<(DiffGaf, VarMap)> {
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
    let gaf = DiffGaf::new(config, vb)?;
    Ok((gaf, var_map))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_input(device: &Device) -> Result<Tensor> {
        // Create a simple test tensor: (batch=2, time=32, features=5)
        let data: Vec<f32> = (0..2 * 32 * 5).map(|i| (i as f32) / 320.0).collect();
        Tensor::from_vec(data, (2, 32, 5), device)
    }

    #[test]
    fn test_diff_gaf_creation() -> Result<()> {
        let device = Device::Cpu;
        let config = DiffGafConfig::default();
        let (gaf, _var_map) = create_diff_gaf(config, &device)?;

        assert_eq!(gaf.config().num_features, 5);
        assert_eq!(gaf.config().image_size, 64);
        assert!(gaf.gamma.is_some());
        assert!(gaf.beta.is_some());

        Ok(())
    }

    #[test]
    fn test_diff_gaf_fixed() -> Result<()> {
        let device = Device::Cpu;
        let config = DiffGafConfig::default();
        let gaf = DiffGaf::new_fixed(config, &device)?;

        assert!(gaf.gamma.is_none());
        assert!(gaf.beta.is_none());

        Ok(())
    }

    #[test]
    fn test_diff_gaf_forward_gasf() -> Result<()> {
        let device = Device::Cpu;
        let config = DiffGafConfig {
            num_features: 5,
            image_size: 16,
            method: DiffGafMethod::Summation,
            learnable_norm: false,
            eps: 1e-7,
        };

        let gaf = DiffGaf::new_fixed(config, &device)?;
        let input = create_test_input(&device)?;
        let output = gaf.forward(&input)?;

        // Output should be (batch=2, features=5, image_size=16, image_size=16)
        assert_eq!(output.dims(), &[2, 5, 16, 16]);

        // GASF values should be in [-1, 1] (cosine values)
        let max_val = output.max(0)?.max(0)?.max(0)?.max(0)?;
        let min_val = output.min(0)?.min(0)?.min(0)?.min(0)?;

        let max_f: f32 = max_val.to_scalar()?;
        let min_f: f32 = min_val.to_scalar()?;

        assert!(max_f <= 1.0 + 1e-5, "Max value {} > 1.0", max_f);
        assert!(min_f >= -1.0 - 1e-5, "Min value {} < -1.0", min_f);

        Ok(())
    }

    #[test]
    fn test_diff_gaf_forward_gadf() -> Result<()> {
        let device = Device::Cpu;
        let config = DiffGafConfig {
            num_features: 5,
            image_size: 16,
            method: DiffGafMethod::Difference,
            learnable_norm: false,
            eps: 1e-7,
        };

        let gaf = DiffGaf::new_fixed(config, &device)?;
        let input = create_test_input(&device)?;
        let output = gaf.forward(&input)?;

        assert_eq!(output.dims(), &[2, 5, 16, 16]);

        Ok(())
    }

    #[test]
    fn test_diff_gaf_dual() -> Result<()> {
        let device = Device::Cpu;
        let config = DiffGafConfig {
            num_features: 5,
            image_size: 16,
            method: DiffGafMethod::Summation,
            learnable_norm: false,
            eps: 1e-7,
        };

        let gaf = DiffGaf::new_fixed(config, &device)?;
        let input = create_test_input(&device)?;
        let output = gaf.forward_dual(&input)?;

        // Dual output has 2x features (GASF + GADF)
        assert_eq!(output.dims(), &[2, 10, 16, 16]);

        Ok(())
    }

    #[test]
    fn test_diff_gaf_with_learnable_params() -> Result<()> {
        let device = Device::Cpu;
        let config = DiffGafConfig {
            num_features: 5,
            image_size: 16,
            method: DiffGafMethod::Summation,
            learnable_norm: true,
            eps: 1e-7,
        };

        let (gaf, var_map) = create_diff_gaf(config, &device)?;
        let input = create_test_input(&device)?;
        let output = gaf.forward(&input)?;

        assert_eq!(output.dims(), &[2, 5, 16, 16]);

        // Check that we have learnable parameters
        let all_vars = var_map.all_vars();
        assert_eq!(all_vars.len(), 2); // gamma and beta

        Ok(())
    }

    #[test]
    fn test_diff_gaf_builder() -> Result<()> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let gaf = DiffGafBuilder::new()
            .num_features(3)
            .image_size(32)
            .method(DiffGafMethod::Difference)
            .learnable_norm(true)
            .build(vb)?;

        assert_eq!(gaf.config().num_features, 3);
        assert_eq!(gaf.config().image_size, 32);
        assert_eq!(gaf.config().method, DiffGafMethod::Difference);

        Ok(())
    }

    #[test]
    fn test_normalization_bounds() -> Result<()> {
        let device = Device::Cpu;
        let config = DiffGafConfig {
            num_features: 1,
            image_size: 8,
            method: DiffGafMethod::Summation,
            learnable_norm: false,
            eps: 1e-7,
        };

        let gaf = DiffGaf::new_fixed(config, &device)?;

        // Test with extreme values
        let extreme_data: Vec<f32> = vec![
            -1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, // batch 1
            1000.0, 500.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.1, // batch 2
        ];
        let input = Tensor::from_vec(extreme_data, (2, 8, 1), &device)?;

        let output = gaf.forward(&input)?;

        // Should still produce valid output
        assert_eq!(output.dims(), &[2, 1, 8, 8]);

        // Values should be bounded
        let max_val: f32 = output.max(0)?.max(0)?.max(0)?.max(0)?.to_scalar()?;
        let min_val: f32 = output.min(0)?.min(0)?.min(0)?.min(0)?.to_scalar()?;

        assert!(
            max_val.is_finite(),
            "Output contains non-finite max: {}",
            max_val
        );
        assert!(
            min_val.is_finite(),
            "Output contains non-finite min: {}",
            min_val
        );

        Ok(())
    }

    #[test]
    fn test_resize_preserves_features() -> Result<()> {
        let device = Device::Cpu;
        let config = DiffGafConfig {
            num_features: 3,
            image_size: 64,
            method: DiffGafMethod::Summation,
            learnable_norm: false,
            eps: 1e-7,
        };

        let gaf = DiffGaf::new_fixed(config, &device)?;

        // Input with different time length than image_size
        let data: Vec<f32> = (0..4 * 128 * 3).map(|i| (i as f32) / 1000.0).collect();
        let input = Tensor::from_vec(data, (4, 128, 3), &device)?;

        let output = gaf.forward(&input)?;

        // Should resize time dimension to image_size
        assert_eq!(output.dims(), &[4, 3, 64, 64]);

        Ok(())
    }
}
