//! DiffGAF transformation layers
//!
//! This module contains the core learnable transformations:
//! - LearnableNorm: Adaptive normalization
//! - PolarEncoder: Angle encoding
//! - GramianLayer: Gramian matrix computation (GASF, GADF, or Dual)
//!
//! # GAF Sign Ambiguity Warning
//!
//! **GASF (Gramian Angular Summation Field)** computes `cos(φ_i + φ_j)`, which is
//! **symmetric**: `G[i,j] = G[j,i]`. This means GASF alone **cannot distinguish**
//! upward price movements from downward ones — a rising series and its mirror
//! produce identical GASF matrices.
//!
//! **GADF (Gramian Angular Difference Field)** computes `sin(φ_i - φ_j)`, which is
//! **anti-symmetric**: `G[i,j] = -G[j,i]`. This preserves temporal direction and
//! resolves the sign ambiguity.
//!
//! **Recommendation**: Use `GramianMode::Dual` (the default) to concatenate both
//! GASF and GADF channels, giving the model both correlation magnitude (GASF) and
//! directional information (GADF). Using `GramianMode::Gasf` alone will lose
//! directional information.

use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::{Tensor, backend::Backend};

/// Configuration for learnable normalization
#[derive(Config, Debug)]
pub struct LearnableNormConfig {
    /// Number of features to normalize
    pub num_features: usize,
    /// Target minimum value after normalization
    #[config(default = "-1.0")]
    pub target_min: f32,
    /// Target maximum value after normalization
    #[config(default = "1.0")]
    pub target_max: f32,
    /// Epsilon for numerical stability
    #[config(default = "1e-7")]
    pub eps: f64,
}

impl LearnableNormConfig {
    /// Initialize the learnable normalization layer
    pub fn init<B: Backend>(&self, device: &B::Device) -> LearnableNorm<B> {
        LearnableNorm {
            min_bound: Param::from_tensor(Tensor::zeros([self.num_features], device)),
            max_bound: Param::from_tensor(Tensor::ones([self.num_features], device)),
            target_min: self.target_min,
            target_max: self.target_max,
            eps: self.eps,
        }
    }
}

/// Learnable normalization layer
///
/// Adaptively normalizes input tensors using learnable min/max bounds.
/// The bounds are trained via backpropagation to optimize for the task.
///
/// # Shape
/// - Input: `[batch, time, features]` or `[batch, features]`
/// - Output: Same as input
#[derive(Module, Debug)]
pub struct LearnableNorm<B: Backend> {
    /// Learnable minimum bound for each feature
    pub min_bound: Param<Tensor<B, 1>>,
    /// Learnable maximum bound for each feature
    pub max_bound: Param<Tensor<B, 1>>,
    /// Target minimum value after normalization
    pub target_min: f32,
    /// Target maximum value after normalization
    pub target_max: f32,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl<B: Backend> LearnableNorm<B> {
    /// Forward pass - normalize input tensor
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `[batch, time, features]`
    ///
    /// # Returns
    /// Normalized tensor in range [target_min, target_max]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _time, features] = input.dims();

        // Get learnable bounds and reshape for broadcasting
        let min = self.min_bound.val().reshape([1, 1, features]);
        let max = self.max_bound.val().reshape([1, 1, features]);

        // Clamp bounds to prevent gradient explosion
        let min_clamped = min.clamp(-100.0, 100.0);
        let max_clamped = max.clamp(-100.0, 100.0);

        // Ensure max > min by adding epsilon to range
        let range = (max_clamped.clone() - min_clamped.clone()).clamp(self.eps, 200.0);

        // Normalize: (x - min) / (max - min)
        let normalized = (input - min_clamped) / range;

        // Scale to target range: x' * (target_max - target_min) + target_min
        let scale = self.target_max - self.target_min;
        normalized * scale + self.target_min
    }
}

/// Configuration for polar encoder
#[derive(Config, Debug)]
pub struct PolarEncoderConfig {
    /// Number of features to encode
    pub num_features: usize,
    /// Use smooth arccos approximation for better gradients
    #[config(default = "true")]
    pub use_smooth: bool,
    /// Epsilon for numerical stability
    #[config(default = "1e-7")]
    pub eps: f64,
}

impl PolarEncoderConfig {
    /// Initialize the polar encoder layer
    pub fn init<B: Backend>(&self, device: &B::Device) -> PolarEncoder<B> {
        PolarEncoder {
            angle_scale: Param::from_tensor(Tensor::ones([self.num_features], device)),
            angle_offset: Param::from_tensor(Tensor::zeros([self.num_features], device)),
            use_smooth: self.use_smooth,
            eps: self.eps,
        }
    }
}

/// Polar encoding layer
///
/// Converts normalized values to angular representation using arccos.
/// Includes learnable scale and offset parameters.
///
/// # Shape
/// - Input: `[batch, time, features]` in range [-1, 1]
/// - Output: `[batch, time, features]` angles in radians
#[derive(Module, Debug)]
pub struct PolarEncoder<B: Backend> {
    /// Learnable scale factor for angles
    pub angle_scale: Param<Tensor<B, 1>>,
    /// Learnable offset for angles
    pub angle_offset: Param<Tensor<B, 1>>,
    /// Use smooth arccos approximation
    pub use_smooth: bool,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl<B: Backend> PolarEncoder<B> {
    /// Forward pass - encode to polar angles
    ///
    /// # Arguments
    /// * `input` - Normalized tensor in range [-1, 1]
    ///
    /// # Returns
    /// Angular representation
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _time, features] = input.dims();

        // Clamp input to valid arccos range with small margin
        let x_safe = input.clamp(-1.0 + self.eps, 1.0 - self.eps);

        // Compute arccos using element-wise operation
        // arccos(x) = π/2 - arcsin(x)
        // For now, use a polynomial approximation for arccos
        // arccos(x) ≈ π/2 - x - x³/6 (Taylor series approximation)
        let pi_half = std::f64::consts::PI / 2.0;
        let x_cubed = x_safe.clone().powf_scalar(3.0);
        let angles = Tensor::full(x_safe.shape(), pi_half, &x_safe.device())
            - x_safe.clone()
            - (x_cubed / 6.0);

        // Apply learnable transformation
        let scale = self.angle_scale.val().reshape([1, 1, features]);
        let offset = self.angle_offset.val().reshape([1, 1, features]);

        angles * scale + offset
    }
}

/// Mode for Gramian Angular Field computation.
///
/// # Sign Ambiguity
///
/// **`Gasf` is symmetric** — `G[i,j] = G[j,i]` — so it **cannot distinguish**
/// upward vs downward price movements. A rising series and its reverse produce
/// identical GASF matrices.
///
/// **`Gadf` is anti-symmetric** — `G[i,j] = -G[j,i]` — so it preserves temporal
/// direction.
///
/// **`Dual` concatenates both** along the feature/channel dimension, providing the
/// model with both correlation magnitude and directional information. This is the
/// recommended default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GramianMode {
    /// GASF only: `cos(φ_i + φ_j)` — symmetric, sign-ambiguous.
    /// ⚠️ Cannot distinguish upward from downward movements.
    Gasf,
    /// GADF only: `sin(φ_i - φ_j)` — anti-symmetric, preserves direction.
    Gadf,
    /// Dual: concatenates GASF + GADF along the channel dimension.
    /// Output has `2 * features` channels. **Recommended default.**
    Dual,
}

impl Default for GramianMode {
    fn default() -> Self {
        GramianMode::Dual
    }
}

/// Configuration for Gramian layer
#[derive(Config, Debug)]
pub struct GramianLayerConfig {
    /// Use memory-efficient computation
    #[config(default = "true")]
    pub use_efficient: bool,
    /// Gramian computation mode (GASF, GADF, or Dual).
    ///
    /// **Default: `Dual`** — concatenates GASF + GADF to resolve sign ambiguity.
    /// Using `Gasf` alone loses directional information (sign-ambiguous).
    #[config(default = "GramianMode::Dual")]
    pub mode: GramianMode,
}

impl GramianLayerConfig {
    /// Initialize the Gramian layer
    pub fn init<B: Backend>(&self) -> GramianLayer<B> {
        let (compute_gasf, compute_gadf) = match self.mode {
            GramianMode::Gasf => (true, false),
            GramianMode::Gadf => (false, true),
            GramianMode::Dual => (true, true),
        };
        GramianLayer {
            use_efficient: self.use_efficient,
            compute_gasf,
            compute_gadf,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Gramian matrix computation layer
///
/// Computes the Gramian Angular Field. Supports three modes:
///
/// - **GASF**: `G[i,j] = cos(φ_i + φ_j)` — symmetric, captures magnitude correlation
/// - **GADF**: `G[i,j] = sin(φ_i - φ_j)` — anti-symmetric, captures temporal direction
/// - **Dual**: concatenates GASF + GADF along channel dim (recommended)
///
/// # Sign Ambiguity Warning
///
/// GASF alone is **sign-ambiguous**: `cos(φ_i + φ_j) = cos(φ_j + φ_i)`, so the
/// model cannot tell if price went up or down. Use `GramianMode::Dual` (default)
/// or `GramianMode::Gadf` to preserve directional information.
///
/// # Shape
/// - Input: `[batch, time, features]` angles
/// - Output (GASF/GADF): `[batch, features, time, time]`
/// - Output (Dual): `[batch, features * 2, time, time]`
#[derive(Module, Debug)]
pub struct GramianLayer<B: Backend> {
    /// Use memory-efficient computation
    pub use_efficient: bool,
    /// Whether to compute GASF (Gramian Angular Summation Field).
    /// ⚠️ GASF alone is sign-ambiguous — enable `compute_gadf` too (Dual mode).
    pub compute_gasf: bool,
    /// Whether to compute GADF (Gramian Angular Difference Field).
    /// GADF preserves temporal direction and resolves GASF sign ambiguity.
    pub compute_gadf: bool,
    /// Phantom data for backend
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> GramianLayer<B> {
    /// Forward pass - compute Gramian Angular Field
    ///
    /// # Arguments
    /// * `angles` - Angular representation of shape `[batch, time, features]`
    ///
    /// # Returns
    /// - GASF/GADF mode: `[batch, features, time, time]`
    /// - Dual mode: `[batch, features * 2, time, time]` (GASF channels then GADF channels)
    pub fn forward(&self, angles: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, time, features] = angles.dims();

        // Compute cos and sin using Taylor series approximations
        // cos(x) ≈ 1 - x²/2 + x⁴/24
        // sin(x) ≈ x - x³/6 + x⁵/120
        let angles_sq = angles.clone().powf_scalar(2.0);
        let angles_cubed = angles.clone().powf_scalar(3.0);
        let angles_quad = angles.clone().powf_scalar(4.0);

        let cos_phi = Tensor::ones_like(&angles) - (angles_sq.clone() / 2.0) + (angles_quad / 24.0); // [B, T, F]

        let sin_phi = angles.clone() - (angles_cubed / 6.0); // [B, T, F]

        match (self.compute_gasf, self.compute_gadf) {
            (true, false) => self.compute_gasf_matrix(cos_phi, sin_phi, batch, time, features),
            (false, true) => self.compute_gadf_matrix(cos_phi, sin_phi, batch, time, features),
            (true, true) => {
                // Dual mode: concatenate GASF + GADF along channel dimension
                let gasf = self.compute_gasf_matrix(
                    cos_phi.clone(),
                    sin_phi.clone(),
                    batch,
                    time,
                    features,
                );
                let gadf = self.compute_gadf_matrix(cos_phi, sin_phi, batch, time, features);
                // [B, F, T, T] + [B, F, T, T] -> [B, 2F, T, T]
                Tensor::cat(vec![gasf, gadf], 1)
            }
            (false, false) => {
                // Fallback: at least compute GASF if nothing is selected
                self.compute_gasf_matrix(cos_phi, sin_phi, batch, time, features)
            }
        }
    }

    /// Compute GASF: Gramian Angular Summation Field.
    ///
    /// Formula: `GASF[i,j] = cos(φ_i + φ_j) = cos(φ_i)cos(φ_j) - sin(φ_i)sin(φ_j)`
    ///
    /// ⚠️ **Symmetric** — cannot distinguish temporal direction. Use Dual mode
    /// or GADF to preserve directional information.
    fn compute_gasf_matrix(
        &self,
        cos_phi: Tensor<B, 3>,
        sin_phi: Tensor<B, 3>,
        batch: usize,
        time: usize,
        features: usize,
    ) -> Tensor<B, 4> {
        // Reshape for broadcasting: [B, T, F] -> [B, F, T, 1] and [B, F, 1, T]
        let cos_i = cos_phi
            .clone()
            .swap_dims(1, 2)
            .reshape([batch, features, time, 1]);
        let cos_j = cos_phi.swap_dims(1, 2).reshape([batch, features, 1, time]);

        let sin_i = sin_phi
            .clone()
            .swap_dims(1, 2)
            .reshape([batch, features, time, 1]);
        let sin_j = sin_phi.swap_dims(1, 2).reshape([batch, features, 1, time]);

        // GASF[i,j] = cos(φ_i)cos(φ_j) - sin(φ_i)sin(φ_j)
        let cos_product = cos_i * cos_j;
        let sin_product = sin_i * sin_j;

        cos_product - sin_product
    }

    /// Compute GADF: Gramian Angular Difference Field.
    ///
    /// Formula: `GADF[i,j] = sin(φ_i - φ_j) = sin(φ_i)cos(φ_j) - cos(φ_i)sin(φ_j)`
    ///
    /// **Anti-symmetric** — `GADF[i,j] = -GADF[j,i]` — preserves temporal direction.
    fn compute_gadf_matrix(
        &self,
        cos_phi: Tensor<B, 3>,
        sin_phi: Tensor<B, 3>,
        batch: usize,
        time: usize,
        features: usize,
    ) -> Tensor<B, 4> {
        // Reshape for broadcasting: [B, T, F] -> [B, F, T, 1] and [B, F, 1, T]
        let cos_i = cos_phi
            .clone()
            .swap_dims(1, 2)
            .reshape([batch, features, time, 1]);
        let cos_j = cos_phi.swap_dims(1, 2).reshape([batch, features, 1, time]);

        let sin_i = sin_phi
            .clone()
            .swap_dims(1, 2)
            .reshape([batch, features, time, 1]);
        let sin_j = sin_phi.swap_dims(1, 2).reshape([batch, features, 1, time]);

        // GADF[i,j] = sin(φ_i)cos(φ_j) - cos(φ_i)sin(φ_j)
        let term1 = sin_i * cos_j;
        let term2 = cos_i * sin_j;

        term1 - term2
    }

    #[cfg(feature = "gpu")]
    /// GPU-optimized Gramian computation using batched matrix multiplication.
    /// Currently only supports GASF; Dual mode falls back to CPU path.
    fn forward_gpu_optimized(
        &self,
        cos_phi: Tensor<B, 3>,
        sin_phi: Tensor<B, 3>,
        batch: usize,
        time: usize,
        features: usize,
    ) -> Tensor<B, 4> {
        // Swap dims for efficient matmul: [B, T, F] -> [B, F, T]
        let cos_swapped = cos_phi.swap_dims(1, 2);
        let sin_swapped = sin_phi.swap_dims(1, 2);

        // Reshape for batched matmul: [B, F, T] -> [B*F, T, 1]
        let cos_col = cos_swapped.clone().reshape([batch * features, time, 1]);

        // [B, F, T] -> [B*F, 1, T]
        let cos_row = cos_swapped.reshape([batch * features, 1, time]);
        let sin_col = sin_swapped.clone().reshape([batch * features, time, 1]);
        let sin_row = sin_swapped.reshape([batch * features, 1, time]);

        // Batched outer product via matmul: [B*F, T, 1] x [B*F, 1, T] -> [B*F, T, T]
        let cos_outer = cos_col.matmul(cos_row);
        let sin_outer = sin_col.matmul(sin_row);

        // G[i,j] = cos(φ[i])cos(φ[j]) - sin(φ[i])sin(φ[j])
        let gramian_flat = cos_outer - sin_outer; // [B*F, T, T]

        // Reshape back: [B*F, T, T] -> [B, F, T, T]
        gramian_flat.reshape([batch, features, time, time])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_learnable_norm_config() {
        let config = LearnableNormConfig {
            num_features: 8,
            target_min: -1.0,
            target_max: 1.0,
            eps: 1e-7,
        };
        assert_eq!(config.num_features, 8);
        assert_eq!(config.target_min, -1.0);
        assert_eq!(config.target_max, 1.0);
    }

    #[test]
    fn test_learnable_norm_forward() {
        let device = Default::default();
        let config = LearnableNormConfig {
            num_features: 2,
            target_min: -1.0,
            target_max: 1.0,
            eps: 1e-7,
        };
        let norm = config.init::<TestBackend>(&device);

        // Create test input [batch=2, time=3, features=2]
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]],
                [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]],
            ],
            &device,
        );

        let output = norm.forward(input);

        // Check shape is preserved
        assert_eq!(output.dims(), [2, 3, 2]);
    }

    #[test]
    fn test_polar_encoder_config() {
        let config = PolarEncoderConfig {
            num_features: 8,
            use_smooth: true,
            eps: 1e-7,
        };
        assert_eq!(config.num_features, 8);
        assert!(config.use_smooth);
    }

    #[test]
    fn test_polar_encoder_forward() {
        let device = Default::default();
        let config = PolarEncoderConfig {
            num_features: 2,
            use_smooth: true,
            eps: 1e-7,
        };
        let encoder = config.init::<TestBackend>(&device);

        // Create test input in valid range [-1, 1]
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [[0.0, 0.5], [0.7, -0.3], [-0.5, 0.9]],
                [[-0.8, 0.2], [0.4, -0.6], [0.1, -0.1]],
            ],
            &device,
        );

        let output = encoder.forward(input);

        // Check shape is preserved
        assert_eq!(output.dims(), [2, 3, 2]);
    }

    #[test]
    fn test_gramian_layer_config() {
        let config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Dual,
        };
        assert!(config.use_efficient);
        assert_eq!(config.mode, GramianMode::Dual);
    }

    #[test]
    fn test_gramian_mode_default_is_dual() {
        // The default mode should be Dual to avoid sign ambiguity
        assert_eq!(GramianMode::default(), GramianMode::Dual);
    }

    #[test]
    fn test_gramian_layer_forward_gasf() {
        let device = Default::default();
        let config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Gasf,
        };
        let layer = config.init::<TestBackend>();

        // Create test angles [batch=2, time=4, features=3]
        let angles = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0.0, 0.5, 1.0],
                    [0.2, 0.7, 1.2],
                    [0.4, 0.9, 1.4],
                    [0.6, 1.1, 1.6],
                ],
                [
                    [0.1, 0.6, 1.1],
                    [0.3, 0.8, 1.3],
                    [0.5, 1.0, 1.5],
                    [0.7, 1.2, 1.7],
                ],
            ],
            &device,
        );

        let output = layer.forward(angles);

        // GASF output shape: [batch=2, features=3, time=4, time=4]
        assert_eq!(output.dims(), [2, 3, 4, 4]);
    }

    #[test]
    fn test_gramian_layer_forward_gadf() {
        let device = Default::default();
        let config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Gadf,
        };
        let layer = config.init::<TestBackend>();

        let angles = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0.0, 0.5, 1.0],
                    [0.2, 0.7, 1.2],
                    [0.4, 0.9, 1.4],
                    [0.6, 1.1, 1.6],
                ],
                [
                    [0.1, 0.6, 1.1],
                    [0.3, 0.8, 1.3],
                    [0.5, 1.0, 1.5],
                    [0.7, 1.2, 1.7],
                ],
            ],
            &device,
        );

        let output = layer.forward(angles);

        // GADF output shape: same as GASF [batch=2, features=3, time=4, time=4]
        assert_eq!(output.dims(), [2, 3, 4, 4]);
    }

    #[test]
    fn test_gramian_layer_forward_dual() {
        let device = Default::default();
        let config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Dual,
        };
        let layer = config.init::<TestBackend>();

        let angles = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0.0, 0.5, 1.0],
                    [0.2, 0.7, 1.2],
                    [0.4, 0.9, 1.4],
                    [0.6, 1.1, 1.6],
                ],
                [
                    [0.1, 0.6, 1.1],
                    [0.3, 0.8, 1.3],
                    [0.5, 1.0, 1.5],
                    [0.7, 1.2, 1.7],
                ],
            ],
            &device,
        );

        let output = layer.forward(angles);

        // Dual output: [batch=2, features*2=6, time=4, time=4]
        // First 3 channels are GASF, last 3 are GADF
        assert_eq!(output.dims(), [2, 6, 4, 4]);
    }

    #[test]
    fn test_gasf_symmetry() {
        // GASF should be symmetric: G[i,j] = G[j,i]
        let device = Default::default();
        let config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Gasf,
        };
        let layer = config.init::<TestBackend>();

        let angles =
            Tensor::<TestBackend, 3>::from_floats([[[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]]], &device);

        let gramian = layer.forward(angles);

        // Shape: [batch=1, features=2, time=3, time=3]
        assert_eq!(gramian.dims(), [1, 2, 3, 3]);

        // Check symmetry: G[0,1] should equal G[1,0] for each feature
        let data = gramian.to_data();
        let vals: Vec<f32> = data.to_vec().unwrap();
        // Layout: [batch][feature][row][col] -> flat index = b*2*3*3 + f*3*3 + i*3 + j
        for f in 0..2 {
            for i in 0..3 {
                for j in 0..3 {
                    let idx_ij = f * 9 + i * 3 + j;
                    let idx_ji = f * 9 + j * 3 + i;
                    let diff = (vals[idx_ij] - vals[idx_ji]).abs();
                    assert!(
                        diff < 1e-5,
                        "GASF not symmetric at f={}, i={}, j={}: {} vs {}",
                        f,
                        i,
                        j,
                        vals[idx_ij],
                        vals[idx_ji]
                    );
                }
            }
        }
    }

    #[test]
    fn test_gadf_anti_symmetry() {
        // GADF should be anti-symmetric: G[i,j] = -G[j,i]
        // This is the key property that resolves sign ambiguity
        let device = Default::default();
        let config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Gadf,
        };
        let layer = config.init::<TestBackend>();

        let angles =
            Tensor::<TestBackend, 3>::from_floats([[[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]]], &device);

        let gramian = layer.forward(angles);

        assert_eq!(gramian.dims(), [1, 2, 3, 3]);

        let data = gramian.to_data();
        let vals: Vec<f32> = data.to_vec().unwrap();
        for f in 0..2 {
            for i in 0..3 {
                for j in 0..3 {
                    let idx_ij = f * 9 + i * 3 + j;
                    let idx_ji = f * 9 + j * 3 + i;
                    // G[i,j] should equal -G[j,i]
                    let sum = vals[idx_ij] + vals[idx_ji];
                    assert!(
                        sum.abs() < 1e-5,
                        "GADF not anti-symmetric at f={}, i={}, j={}: {} + {} = {}",
                        f,
                        i,
                        j,
                        vals[idx_ij],
                        vals[idx_ji],
                        sum
                    );
                }
            }
            // Diagonal should be zero (sin(0) = 0)
            for i in 0..3 {
                let idx_ii = f * 9 + i * 3 + i;
                assert!(
                    vals[idx_ii].abs() < 1e-5,
                    "GADF diagonal not zero at f={}, i={}: {}",
                    f,
                    i,
                    vals[idx_ii]
                );
            }
        }
    }

    #[test]
    fn test_dual_contains_both_gasf_and_gadf() {
        // Dual mode should produce GASF in first F channels and GADF in last F channels
        let device = Default::default();

        let gasf_config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Gasf,
        };
        let gadf_config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Gadf,
        };
        let dual_config = GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Dual,
        };

        let gasf_layer = gasf_config.init::<TestBackend>();
        let gadf_layer = gadf_config.init::<TestBackend>();
        let dual_layer = dual_config.init::<TestBackend>();

        let angles = Tensor::<TestBackend, 3>::from_floats(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]],
            &device,
        );

        let gasf_out = gasf_layer.forward(angles.clone()); // [1, 2, 4, 4]
        let gadf_out = gadf_layer.forward(angles.clone()); // [1, 2, 4, 4]
        let dual_out = dual_layer.forward(angles); // [1, 4, 4, 4]

        assert_eq!(dual_out.dims(), [1, 4, 4, 4]);

        // Verify first 2 channels match GASF
        let dual_data: Vec<f32> = dual_out.to_data().to_vec().unwrap();
        let gasf_data: Vec<f32> = gasf_out.to_data().to_vec().unwrap();
        let gadf_data: Vec<f32> = gadf_out.to_data().to_vec().unwrap();

        // GASF channels (0..2) in dual should match standalone GASF
        for i in 0..gasf_data.len() {
            let diff = (dual_data[i] - gasf_data[i]).abs();
            assert!(
                diff < 1e-5,
                "Dual GASF channel mismatch at index {}: {} vs {}",
                i,
                dual_data[i],
                gasf_data[i]
            );
        }

        // GADF channels (2..4) in dual should match standalone GADF
        let gadf_offset = gasf_data.len(); // = 2 * 4 * 4 = 32
        for i in 0..gadf_data.len() {
            let diff = (dual_data[gadf_offset + i] - gadf_data[i]).abs();
            assert!(
                diff < 1e-5,
                "Dual GADF channel mismatch at index {}: {} vs {}",
                i,
                dual_data[gadf_offset + i],
                gadf_data[i]
            );
        }
    }
}
