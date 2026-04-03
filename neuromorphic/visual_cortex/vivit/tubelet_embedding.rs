//! 3D Tubelet Embeddings for Spatiotemporal Data
//!
//! Implements tubelet tokenization for ViViT (Video Vision Transformer).
//! Converts 3D spatiotemporal volumes (e.g., candlestick charts over time)
//! into patch embeddings suitable for transformer processing.
//!
//! # Architecture
//!
//! ```text
//! Input: [B, T, H, W, C]  (Batch, Time, Height, Width, Channels)
//!           ↓
//!    ┌──────────────┐
//!    │  3D Conv     │  Extract tubelets (t×h×w patches)
//!    │  Stride=S    │
//!    └──────────────┘
//!           ↓
//!    [B, N, D]  (Batch, Num_patches, Embedding_dim)
//!           ↓
//!    ┌──────────────┐
//!    │  Positional  │  Add learned position embeddings
//!    │  Encoding    │
//!    └──────────────┘
//!           ↓
//!    [B, N+1, D]  (with [CLS] token)
//! ```
//!
//! # Tubelet Structure
//!
//! A tubelet is a 3D patch extracted from the input:
//! - **Temporal extent**: `t` frames
//! - **Spatial extent**: `h × w` pixels
//! - Flattened to a single vector and projected to embedding dimension
//!
//! For trading:
//! - T = time steps (e.g., 60 bars)
//! - H = price range (e.g., 224 pixels for GAF image)
//! - W = features (e.g., OHLCV channels)

use common::{JanusError, Result};
use ndarray::{Array1, Array2, Array3, Array5};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Configuration for tubelet embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TubeletConfig {
    /// Number of temporal frames in each tubelet
    pub temporal_patch_size: usize,

    /// Height of spatial patch (pixels)
    pub spatial_patch_height: usize,

    /// Width of spatial patch (pixels)
    pub spatial_patch_width: usize,

    /// Stride for temporal dimension
    pub temporal_stride: usize,

    /// Stride for spatial height
    pub spatial_stride_h: usize,

    /// Stride for spatial width
    pub spatial_stride_w: usize,

    /// Embedding dimension for output
    pub embedding_dim: usize,

    /// Number of input channels
    pub in_channels: usize,

    /// Whether to add [CLS] token
    pub use_cls_token: bool,

    /// Dropout rate for embeddings
    pub dropout_rate: f32,
}

impl Default for TubeletConfig {
    fn default() -> Self {
        Self {
            temporal_patch_size: 4,   // 4 time steps
            spatial_patch_height: 16, // 16 pixels
            spatial_patch_width: 16,  // 16 pixels
            temporal_stride: 4,
            spatial_stride_h: 16,
            spatial_stride_w: 16,
            embedding_dim: 768, // Standard transformer dimension
            in_channels: 3,     // RGB or OHLCV features
            use_cls_token: true,
            dropout_rate: 0.1,
        }
    }
}

/// 3D tubelet embeddings for spatiotemporal data
pub struct TubeletEmbedding {
    /// Configuration
    config: TubeletConfig,

    /// Learned embedding projection weights [D, t*h*w*C]
    /// Where D = embedding_dim, and t*h*w*C = flattened tubelet size
    projection: Array2<f32>,

    /// Learned positional embeddings [N+1, D] (including CLS token)
    positional_embeddings: Array2<f32>,

    /// CLS token embedding [1, D]
    cls_token: Option<Array1<f32>>,

    /// Number of patches (computed from input dimensions)
    num_patches: Option<usize>,

    /// Cached statistics for normalization
    stats: Option<NormalizationStats>,
}

/// Statistics for input normalization
#[derive(Debug, Clone)]
struct NormalizationStats {
    mean: f32,
    std: f32,
}

impl TubeletEmbedding {
    /// Create a new tubelet embedding layer
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for tubelet parameters
    ///
    /// # Example
    ///
    /// ```
    /// use janus_neuromorphic::visual_cortex::vivit::{TubeletEmbedding, TubeletConfig};
    ///
    /// let config = TubeletConfig::default();
    /// let embedder = TubeletEmbedding::new(config);
    /// ```
    pub fn new(config: TubeletConfig) -> Self {
        let tubelet_size = config.temporal_patch_size
            * config.spatial_patch_height
            * config.spatial_patch_width
            * config.in_channels;

        // Initialize projection weights with Xavier/Glorot initialization
        let projection = Self::xavier_init(config.embedding_dim, tubelet_size);

        // CLS token if enabled
        let cls_token = if config.use_cls_token {
            Some(Self::random_init_1d(config.embedding_dim))
        } else {
            None
        };

        Self {
            config,
            projection,
            positional_embeddings: Array2::zeros((0, 0)), // Will be initialized on first forward
            cls_token,
            num_patches: None,
            stats: None,
        }
    }

    /// Forward pass: convert input video to tubelet embeddings
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [B, T, H, W, C]
    ///   - B: batch size
    ///   - T: temporal frames
    ///   - H: spatial height
    ///   - W: spatial width
    ///   - C: channels
    ///
    /// # Returns
    ///
    /// Embedded tubelets of shape [B, N+1, D] where:
    /// - N: number of tubelet patches
    /// - D: embedding dimension
    /// - +1: CLS token (if enabled)
    pub fn forward(&mut self, input: &Array5<f32>) -> Result<Array3<f32>> {
        let (batch_size, t, h, w, c) = input.dim();

        debug!(
            "Tubelet embedding forward: input shape [B={}, T={}, H={}, W={}, C={}]",
            batch_size, t, h, w, c
        );

        // Validate input dimensions
        if c != self.config.in_channels {
            return Err(JanusError::InvalidInput(format!(
                "Expected {} input channels, got {}",
                self.config.in_channels, c
            )));
        }

        // Extract tubelets
        let tubelets = self.extract_tubelets(input)?;
        let (batch, num_patches, tubelet_dim) = tubelets.dim();

        debug!(
            "Extracted {} tubelets of dimension {}",
            num_patches, tubelet_dim
        );

        // Initialize positional embeddings if needed
        if self.num_patches != Some(num_patches) {
            self.initialize_positional_embeddings(num_patches);
        }

        // Project tubelets to embedding space: [B, N, tubelet_dim] @ [tubelet_dim, D] -> [B, N, D]
        let mut embeddings = Array3::zeros((batch, num_patches, self.config.embedding_dim));

        for b in 0..batch {
            for n in 0..num_patches {
                let tubelet = tubelets.slice(s![b, n, ..]);
                let embedding = self.projection.dot(&tubelet);
                embeddings.slice_mut(s![b, n, ..]).assign(&embedding);
            }
        }

        // Add positional embeddings
        for b in 0..batch {
            for n in 0..num_patches {
                let pos_emb = self.positional_embeddings.slice(s![n, ..]);
                let mut emb = embeddings.slice_mut(s![b, n, ..]);
                emb += &pos_emb;
            }
        }

        // Prepend CLS token if enabled
        if let Some(ref cls_token) = self.cls_token {
            let mut with_cls = Array3::zeros((batch, num_patches + 1, self.config.embedding_dim));

            // Set CLS token for all batches
            for b in 0..batch {
                with_cls.slice_mut(s![b, 0, ..]).assign(cls_token);
            }

            // Copy embeddings
            for b in 0..batch {
                for n in 0..num_patches {
                    with_cls
                        .slice_mut(s![b, n + 1, ..])
                        .assign(&embeddings.slice(s![b, n, ..]));
                }
            }

            embeddings = with_cls;
        }

        debug!(
            "Tubelet embedding output shape: [{}, {}, {}]",
            embeddings.shape()[0],
            embeddings.shape()[1],
            embeddings.shape()[2]
        );

        Ok(embeddings)
    }

    /// Extract 3D tubelets from input volume
    fn extract_tubelets(&self, input: &Array5<f32>) -> Result<Array3<f32>> {
        let (batch_size, t, h, w, c) = input.dim();

        // Calculate number of patches
        let num_temporal_patches =
            (t - self.config.temporal_patch_size) / self.config.temporal_stride + 1;
        let num_height_patches =
            (h - self.config.spatial_patch_height) / self.config.spatial_stride_h + 1;
        let num_width_patches =
            (w - self.config.spatial_patch_width) / self.config.spatial_stride_w + 1;

        let num_patches = num_temporal_patches * num_height_patches * num_width_patches;
        let tubelet_size = self.config.temporal_patch_size
            * self.config.spatial_patch_height
            * self.config.spatial_patch_width
            * c;

        let mut tubelets = Array3::zeros((batch_size, num_patches, tubelet_size));

        for b in 0..batch_size {
            let mut patch_idx = 0;

            for t_idx in 0..num_temporal_patches {
                let t_start = t_idx * self.config.temporal_stride;
                let t_end = t_start + self.config.temporal_patch_size;

                for h_idx in 0..num_height_patches {
                    let h_start = h_idx * self.config.spatial_stride_h;
                    let h_end = h_start + self.config.spatial_patch_height;

                    for w_idx in 0..num_width_patches {
                        let w_start = w_idx * self.config.spatial_stride_w;
                        let w_end = w_start + self.config.spatial_patch_width;

                        // Extract tubelet
                        let tubelet =
                            input.slice(s![b, t_start..t_end, h_start..h_end, w_start..w_end, ..]);

                        // Flatten tubelet
                        let flat_tubelet: Vec<f32> = tubelet.iter().copied().collect();
                        tubelets
                            .slice_mut(s![b, patch_idx, ..])
                            .assign(&Array1::from(flat_tubelet));

                        patch_idx += 1;
                    }
                }
            }
        }

        Ok(tubelets)
    }

    /// Initialize positional embeddings for a given number of patches
    fn initialize_positional_embeddings(&mut self, num_patches: usize) {
        let total_positions = if self.config.use_cls_token {
            num_patches + 1
        } else {
            num_patches
        };

        info!(
            "Initializing positional embeddings for {} positions",
            total_positions
        );

        // Use learned positional embeddings (random initialization)
        self.positional_embeddings = Self::xavier_init(total_positions, self.config.embedding_dim);
        self.num_patches = Some(num_patches);
    }

    /// Xavier/Glorot initialization for weights
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        use rand::RngExt;
        let mut rng = rand::rng();

        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| rng.random_range(-limit..limit))
    }

    /// Random initialization for 1D array
    fn random_init_1d(size: usize) -> Array1<f32> {
        use rand::RngExt;
        let mut rng = rand::rng();

        Array1::from_shape_fn(size, |_| rng.random_range(-0.02..0.02))
    }

    /// Compute normalization statistics from a batch of inputs
    pub fn compute_stats(&mut self, inputs: &[Array5<f32>]) {
        let mut sum = 0.0;
        let mut count = 0;

        for input in inputs {
            sum += input.sum();
            count += input.len();
        }

        let mean = sum / count as f32;

        let mut var_sum = 0.0;
        for input in inputs {
            for &val in input.iter() {
                var_sum += (val - mean).powi(2);
            }
        }

        let variance = var_sum / count as f32;
        let std = variance.sqrt().max(1e-5); // Avoid division by zero

        self.stats = Some(NormalizationStats { mean, std });

        debug!("Computed normalization stats: mean={}, std={}", mean, std);
    }

    /// Normalize input using computed statistics
    pub fn normalize(&self, input: &Array5<f32>) -> Array5<f32> {
        if let Some(ref stats) = self.stats {
            (input - stats.mean) / stats.std
        } else {
            input.clone()
        }
    }

    /// Get the number of patches this embedder produces
    pub fn get_num_patches(&self, t: usize, h: usize, w: usize) -> usize {
        let num_temporal_patches =
            (t - self.config.temporal_patch_size) / self.config.temporal_stride + 1;
        let num_height_patches =
            (h - self.config.spatial_patch_height) / self.config.spatial_stride_h + 1;
        let num_width_patches =
            (w - self.config.spatial_patch_width) / self.config.spatial_stride_w + 1;

        num_temporal_patches * num_height_patches * num_width_patches
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get configuration
    pub fn config(&self) -> &TubeletConfig {
        &self.config
    }
}

// Use this for slicing (requires ndarray macros)
use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tubelet_creation() {
        let config = TubeletConfig {
            temporal_patch_size: 2,
            spatial_patch_height: 4,
            spatial_patch_width: 4,
            temporal_stride: 2,
            spatial_stride_h: 4,
            spatial_stride_w: 4,
            embedding_dim: 128,
            in_channels: 3,
            use_cls_token: true,
            dropout_rate: 0.1,
        };

        let embedder = TubeletEmbedding::new(config);
        assert_eq!(embedder.embedding_dim(), 128);
    }

    #[test]
    fn test_num_patches_calculation() {
        let config = TubeletConfig {
            temporal_patch_size: 4,
            spatial_patch_height: 16,
            spatial_patch_width: 16,
            temporal_stride: 4,
            spatial_stride_h: 16,
            spatial_stride_w: 16,
            embedding_dim: 768,
            in_channels: 3,
            use_cls_token: true,
            dropout_rate: 0.0,
        };

        let embedder = TubeletEmbedding::new(config);

        // Input: 16 frames, 64x64 spatial
        let num_patches = embedder.get_num_patches(16, 64, 64);

        // Expected: (16-4)/4+1 = 4 temporal, (64-16)/16+1 = 4 height, 4 width
        // Total: 4 * 4 * 4 = 64 patches
        assert_eq!(num_patches, 64);
    }

    #[test]
    fn test_forward_pass() {
        let config = TubeletConfig {
            temporal_patch_size: 2,
            spatial_patch_height: 4,
            spatial_patch_width: 4,
            temporal_stride: 2,
            spatial_stride_h: 4,
            spatial_stride_w: 4,
            embedding_dim: 128,
            in_channels: 3,
            use_cls_token: true,
            dropout_rate: 0.0,
        };

        let mut embedder = TubeletEmbedding::new(config);

        // Create dummy input: [B=2, T=4, H=8, W=8, C=3]
        let input = Array5::from_elem((2, 4, 8, 8, 3), 1.0);

        let result = embedder.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();

        // Expected patches: T:(4-2)/2+1=2, H:(8-4)/4+1=2, W:(8-4)/4+1=2 => 2*2*2=8 patches
        // With CLS token: 8+1=9 total tokens
        assert_eq!(output.shape(), &[2, 9, 128]);
    }

    #[test]
    fn test_forward_without_cls() {
        let config = TubeletConfig {
            temporal_patch_size: 2,
            spatial_patch_height: 4,
            spatial_patch_width: 4,
            temporal_stride: 2,
            spatial_stride_h: 4,
            spatial_stride_w: 4,
            embedding_dim: 128,
            in_channels: 3,
            use_cls_token: false,
            dropout_rate: 0.0,
        };

        let mut embedder = TubeletEmbedding::new(config);

        // Create dummy input: [B=2, T=4, H=8, W=8, C=3]
        let input = Array5::from_elem((2, 4, 8, 8, 3), 1.0);

        let result = embedder.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();

        // Without CLS token: just 8 patches
        assert_eq!(output.shape(), &[2, 8, 128]);
    }

    #[test]
    fn test_normalization() {
        let config = TubeletConfig::default();
        let mut embedder = TubeletEmbedding::new(config);

        let input1 = Array5::from_elem((1, 4, 8, 8, 3), 1.0);
        let input2 = Array5::from_elem((1, 4, 8, 8, 3), 2.0);

        embedder.compute_stats(&[input1.clone(), input2.clone()]);

        let normalized = embedder.normalize(&input1);

        // Mean should be 1.5, so normalized input1 (all 1.0) should be negative
        assert!(normalized.iter().all(|&x| x < 0.0));
    }

    #[test]
    fn test_invalid_channels() {
        let config = TubeletConfig {
            in_channels: 3,
            ..Default::default()
        };

        let mut embedder = TubeletEmbedding::new(config);

        // Create input with wrong number of channels
        let input = Array5::from_elem((1, 8, 16, 16, 5), 1.0);

        let result = embedder.forward(&input);
        assert!(result.is_err());
    }
}
