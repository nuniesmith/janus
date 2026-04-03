//! DiffGAF main layer implementation

use burn::module::Module;
use burn::tensor::backend::Backend;

use super::transforms::{GramianLayer, GramianMode, LearnableNorm, PolarEncoder};

/// Differentiable Gramian Angular Field layer
///
/// Combines learnable normalization, polar encoding, and Gramian matrix
/// computation into a single end-to-end transformation.
///
/// # Architecture
/// ```text
/// Input → LearnableNorm → PolarEncoder → GramianLayer → Output
/// [B,T,F]      [B,T,F]         [B,T,F]        [B,F,T,T]
/// ```
///
/// # Shape
/// - Input: `[batch, time, features]` raw time series
/// - Output: `[batch, features, time, time]` GAF image
#[derive(Module, Debug)]
pub struct DiffGAF<B: Backend> {
    /// Learnable normalization layer
    normalizer: LearnableNorm<B>,
    /// Polar encoding layer
    encoder: PolarEncoder<B>,
    /// Gramian matrix computation layer
    gramian: GramianLayer<B>,
}

impl<B: Backend> DiffGAF<B> {
    /// Create a new DiffGAF layer from components
    pub fn new(
        normalizer: super::transforms::LearnableNorm<B>,
        encoder: super::transforms::PolarEncoder<B>,
        gramian: super::transforms::GramianLayer<B>,
    ) -> Self {
        Self {
            normalizer,
            encoder,
            gramian,
        }
    }

    /// Forward pass: time series → GAF image
    ///
    /// # Arguments
    /// * `input` - Time series tensor of shape `[batch, time, features]`
    ///
    /// # Returns
    /// GAF image tensor of shape `[batch, features, time, time]`
    pub fn forward(&self, input: burn::tensor::Tensor<B, 3>) -> burn::tensor::Tensor<B, 4> {
        // 1. Normalize to [-1, 1]
        let normalized = self.normalizer.forward(input);

        // 2. Encode to polar angles
        let angles = self.encoder.forward(normalized);

        // 3. Compute Gramian matrix
        let gramian = self.gramian.forward(angles);

        gramian
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_diffgaf_forward() {
        let device = Default::default();

        // Create config and initialize layers
        let norm_config = super::super::transforms::LearnableNormConfig {
            num_features: 2,
            target_min: -1.0,
            target_max: 1.0,
            eps: 1e-7,
        };
        let encoder_config = super::super::transforms::PolarEncoderConfig {
            num_features: 2,
            use_smooth: true,
            eps: 1e-7,
        };
        let gramian_config = super::super::transforms::GramianLayerConfig {
            use_efficient: true,
            mode: GramianMode::Dual,
        };

        let diffgaf = DiffGAF::new(
            norm_config.init(&device),
            encoder_config.init(&device),
            gramian_config.init(),
        );

        // Create test input [batch=2, time=4, features=2]
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
                [[0.5, 1.5], [1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
            ],
            &device,
        );

        let output = diffgaf.forward(input);

        // Check output shape: [batch=2, features*2=4, time=4, time=4] (Dual mode doubles channels)
        assert_eq!(output.dims(), [2, 4, 4, 4]);
    }
}
