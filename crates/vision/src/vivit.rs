//! Video Vision Transformer (ViViT) implementation.
//!
//! This module implements the ViViT architecture for processing sequences of GAF images
//! (or any video-like input) using transformer-based spatiotemporal attention.
//!
//! Based on the paper: "ViViT: A Video Vision Transformer" (Arnab et al., 2021)
//!
//! # Architecture
//!
//! We implement the "Factorized Encoder" variant (Model 2 in the paper), which:
//! 1. Extracts non-overlapping spatiotemporal tubes (tubelets) from the input
//! 2. Projects tubelets to the embedding dimension
//! 3. Applies spatial attention within each frame
//! 4. Applies temporal attention across frames
//! 5. Produces a final embedding for downstream tasks
//!
//! This factorized approach is more efficient than full 3D attention while
//! maintaining good performance for temporal modeling.
//!
//! # Usage
//!
//! ```ignore
//! use vision::vivit::{ViViT, ViViTConfig};
//! use candle_core::Device;
//!
//! let device = Device::Cpu;
//! let config = ViViTConfig::default();
//! let (model, var_map) = ViViT::new(config, &device)?;
//!
//! // Input: (batch, frames, channels, height, width)
//! let input = Tensor::randn(0.0, 1.0, (2, 8, 5, 64, 64), &device)?;
//! let embedding = model.forward(&input)?;
//! // Output: (batch, embed_dim)
//! ```

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Dropout, LayerNorm, Linear, Module, VarBuilder, VarMap, layer_norm, linear};
use serde::{Deserialize, Serialize};

/// Configuration for the ViViT model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViViTConfig {
    /// Number of input frames (temporal dimension)
    pub num_frames: usize,
    /// Number of input channels (e.g., 5 for OHLCV GAF)
    pub in_channels: usize,
    /// Input image height
    pub image_height: usize,
    /// Input image width
    pub image_width: usize,
    /// Patch size for spatial dimension (height and width)
    pub patch_size: usize,
    /// Tubelet depth (number of frames per tube)
    pub tubelet_depth: usize,
    /// Hidden/embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of spatial transformer layers
    pub num_spatial_layers: usize,
    /// Number of temporal transformer layers
    pub num_temporal_layers: usize,
    /// MLP hidden dimension multiplier
    pub mlp_ratio: f32,
    /// Dropout probability
    pub dropout: f32,
    /// Attention dropout probability
    pub attention_dropout: f32,
    /// Whether to use a CLS token for pooling
    pub use_cls_token: bool,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}

impl Default for ViViTConfig {
    fn default() -> Self {
        Self {
            num_frames: 8,
            in_channels: 5, // OHLCV from GAF
            image_height: 64,
            image_width: 64,
            patch_size: 8,    // 8x8 patches
            tubelet_depth: 2, // 2 frames per tube
            embed_dim: 384,   // ViT-Small
            num_heads: 6,
            num_spatial_layers: 4,
            num_temporal_layers: 4,
            mlp_ratio: 4.0,
            dropout: 0.1,
            attention_dropout: 0.0,
            use_cls_token: true,
            layer_norm_eps: 1e-6,
        }
    }
}

impl ViViTConfig {
    /// Create a small configuration for testing/debugging.
    pub fn small() -> Self {
        Self {
            num_frames: 4,
            in_channels: 5,
            image_height: 32,
            image_width: 32,
            patch_size: 8,
            tubelet_depth: 2,
            embed_dim: 192,
            num_heads: 3,
            num_spatial_layers: 2,
            num_temporal_layers: 2,
            mlp_ratio: 4.0,
            dropout: 0.0,
            attention_dropout: 0.0,
            use_cls_token: true,
            layer_norm_eps: 1e-6,
        }
    }

    /// Create a tiny configuration for unit tests.
    pub fn tiny() -> Self {
        Self {
            num_frames: 2,
            in_channels: 3,
            image_height: 16,
            image_width: 16,
            patch_size: 4,
            tubelet_depth: 1,
            embed_dim: 64,
            num_heads: 2,
            num_spatial_layers: 1,
            num_temporal_layers: 1,
            mlp_ratio: 2.0,
            dropout: 0.0,
            attention_dropout: 0.0,
            use_cls_token: true,
            layer_norm_eps: 1e-6,
        }
    }

    /// Calculate the number of spatial patches per frame.
    pub fn num_patches_per_frame(&self) -> usize {
        let h_patches = self.image_height / self.patch_size;
        let w_patches = self.image_width / self.patch_size;
        h_patches * w_patches
    }

    /// Calculate the number of temporal tubes.
    pub fn num_temporal_tokens(&self) -> usize {
        self.num_frames / self.tubelet_depth
    }
}

/// Multi-Head Self-Attention module.
struct MultiHeadAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    dropout: Dropout,
}

impl MultiHeadAttention {
    fn new(embed_dim: usize, num_heads: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let qkv = linear(embed_dim, embed_dim * 3, vb.pp("qkv"))?;
        let proj = linear(embed_dim, embed_dim, vb.pp("proj"))?;
        let dropout = Dropout::new(dropout);

        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            scale,
            dropout,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let (batch, seq_len, _embed_dim) = x.dims3()?;

        // Compute Q, K, V
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, batch, heads, seq, head_dim)

        // Make tensors contiguous for matmul
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        // Scaled dot-product attention
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = (q.matmul(&k_t)? * self.scale)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let attn = self.dropout.forward(&attn, train)?;

        // Apply attention to values
        let out = attn.contiguous()?.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?; // (batch, seq, heads, head_dim)
        let out = out.reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.proj.forward(&out)
    }
}

/// MLP (Feed-Forward Network) module.
struct MLP {
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl MLP {
    fn new(embed_dim: usize, hidden_dim: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(embed_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = linear(hidden_dim, embed_dim, vb.pp("fc2"))?;
        let dropout = Dropout::new(dropout);

        Ok(Self { fc1, fc2, dropout })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        let x = self.dropout.forward(&x, train)?;
        let x = self.fc2.forward(&x)?;
        self.dropout.forward(&x, train)
    }
}

/// Transformer Encoder Block.
struct TransformerBlock {
    norm1: LayerNorm,
    attn: MultiHeadAttention,
    norm2: LayerNorm,
    mlp: MLP,
    dropout: Dropout,
}

impl TransformerBlock {
    fn new(config: &ViViTConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = layer_norm(config.embed_dim, config.layer_norm_eps, vb.pp("norm1"))?;
        let attn = MultiHeadAttention::new(
            config.embed_dim,
            config.num_heads,
            config.attention_dropout,
            vb.pp("attn"),
        )?;
        let norm2 = layer_norm(config.embed_dim, config.layer_norm_eps, vb.pp("norm2"))?;
        let mlp_hidden = (config.embed_dim as f32 * config.mlp_ratio) as usize;
        let mlp = MLP::new(config.embed_dim, mlp_hidden, config.dropout, vb.pp("mlp"))?;
        let dropout = Dropout::new(config.dropout);

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            dropout,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        // Pre-norm architecture (like in original ViT)
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x, train)?;
        let x = self.dropout.forward(&x, train)?;
        let x = (residual + x)?;

        let residual = &x;
        let out = self.norm2.forward(&x)?;
        let out = self.mlp.forward(&out, train)?;
        residual + out
    }
}

/// Tubelet Embedding module.
///
/// Extracts 3D patches (tubelets) from the video input and projects them
/// to the embedding dimension. This is equivalent to a 3D convolution
/// with non-overlapping patches.
struct TubeletEmbed {
    proj: Linear,
    patch_size: usize,
    tubelet_depth: usize,
    #[allow(dead_code)]
    in_channels: usize,
}

impl TubeletEmbed {
    fn new(config: &ViViTConfig, vb: VarBuilder) -> Result<Self> {
        // Each tubelet is flattened: tubelet_depth * patch_size * patch_size * in_channels
        let tubelet_dim =
            config.tubelet_depth * config.patch_size * config.patch_size * config.in_channels;
        let proj = linear(tubelet_dim, config.embed_dim, vb.pp("proj"))?;

        Ok(Self {
            proj,
            patch_size: config.patch_size,
            tubelet_depth: config.tubelet_depth,
            in_channels: config.in_channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: (batch, frames, channels, height, width)
        let (batch, frames, channels, height, width) = x.dims5()?;

        let h_patches = height / self.patch_size;
        let w_patches = width / self.patch_size;
        let t_tubes = frames / self.tubelet_depth;

        // Extract tubelets by processing each spatial patch location
        // We'll collect all tubelets and stack them
        //
        // Since Candle doesn't support 8D reshapes, we do this in steps:
        // 1. Reshape frames into temporal tubes
        // 2. For each spatial patch, extract and flatten

        // First, reshape to separate temporal tubes: (batch, t_tubes, tubelet_depth, channels, height, width)
        // But Candle doesn't support 6D either, so we flatten batch and t_tubes first

        // Flatten to (batch * frames, channels * height * width)
        let x_flat = x.reshape((batch * frames, channels * height * width))?;

        // Reshape to (batch, frames, channels, height, width) as contiguous
        let x = x_flat.reshape((batch, frames, channels, height, width))?;

        // Process by extracting patches manually
        // Strategy: for each tubelet position, gather the relevant elements
        let patch_dim = self.tubelet_depth * channels * self.patch_size * self.patch_size;
        let num_patches = t_tubes * h_patches * w_patches;

        let mut all_patches = Vec::with_capacity(num_patches);

        for t in 0..t_tubes {
            let t_start = t * self.tubelet_depth;
            // Extract temporal slice: (batch, tubelet_depth, channels, height, width)
            let temporal_slice = x.narrow(1, t_start, self.tubelet_depth)?;

            for h in 0..h_patches {
                let h_start = h * self.patch_size;
                // Extract height slice
                let h_slice = temporal_slice.narrow(3, h_start, self.patch_size)?;

                for w in 0..w_patches {
                    let w_start = w * self.patch_size;
                    // Extract width slice: (batch, tubelet_depth, channels, patch_h, patch_w)
                    let patch = h_slice.narrow(4, w_start, self.patch_size)?;

                    // Flatten to (batch, patch_dim)
                    let patch_flat = patch.reshape((batch, patch_dim))?;
                    all_patches.push(patch_flat);
                }
            }
        }

        // Stack all patches: (num_patches, batch, patch_dim) then transpose to (batch, num_patches, patch_dim)
        let stacked = Tensor::stack(&all_patches, 0)?;
        let x = stacked.permute((1, 0, 2))?;

        // Project to embedding dimension
        self.proj.forward(&x)
    }
}

/// Video Vision Transformer (ViViT).
///
/// A transformer-based model for processing video/sequence data.
/// Uses factorized spatial-temporal attention for efficiency.
pub struct ViViT {
    /// Configuration
    config: ViViTConfig,
    /// Tubelet embedding layer
    tubelet_embed: TubeletEmbed,
    /// Spatial positional embedding
    spatial_pos_embed: Tensor,
    /// Temporal positional embedding
    temporal_pos_embed: Tensor,
    /// CLS token for spatial attention
    spatial_cls_token: Option<Tensor>,
    /// CLS token for temporal attention
    temporal_cls_token: Option<Tensor>,
    /// Spatial transformer blocks
    spatial_blocks: Vec<TransformerBlock>,
    /// Temporal transformer blocks
    temporal_blocks: Vec<TransformerBlock>,
    /// Final layer norm
    norm: LayerNorm,
    /// Output projection head (optional)
    head: Option<Linear>,
    /// Dropout
    dropout: Dropout,
    /// Device
    #[allow(dead_code)]
    device: Device,
}

impl ViViT {
    /// Create a new ViViT model.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `device` - Computation device
    ///
    /// # Returns
    /// Tuple of (model, VarMap containing all parameters)
    pub fn new(config: ViViTConfig, device: &Device) -> Result<(Self, VarMap)> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let model = Self::from_vb(config, vb, device)?;
        Ok((model, var_map))
    }

    /// Create a ViViT model from a VarBuilder.
    pub fn from_vb(config: ViViTConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let tubelet_embed = TubeletEmbed::new(&config, vb.pp("tubelet_embed"))?;

        let num_patches = config.num_patches_per_frame();
        let num_temporal = config.num_temporal_tokens();

        // Spatial positional embedding (for patches within a frame)
        // +1 for CLS token if used
        let spatial_seq_len = if config.use_cls_token {
            num_patches + 1
        } else {
            num_patches
        };
        let spatial_pos_embed =
            vb.get((1, spatial_seq_len, config.embed_dim), "spatial_pos_embed")?;

        // Temporal positional embedding
        let temporal_seq_len = if config.use_cls_token {
            num_temporal + 1
        } else {
            num_temporal
        };
        let temporal_pos_embed = vb.get(
            (1, temporal_seq_len, config.embed_dim),
            "temporal_pos_embed",
        )?;

        // CLS tokens
        let (spatial_cls_token, temporal_cls_token) = if config.use_cls_token {
            let spatial_cls = vb.get((1, 1, config.embed_dim), "spatial_cls_token")?;
            let temporal_cls = vb.get((1, 1, config.embed_dim), "temporal_cls_token")?;
            (Some(spatial_cls), Some(temporal_cls))
        } else {
            (None, None)
        };

        // Spatial transformer blocks
        let mut spatial_blocks = Vec::with_capacity(config.num_spatial_layers);
        for i in 0..config.num_spatial_layers {
            let block = TransformerBlock::new(&config, vb.pp(format!("spatial_block_{}", i)))?;
            spatial_blocks.push(block);
        }

        // Temporal transformer blocks
        let mut temporal_blocks = Vec::with_capacity(config.num_temporal_layers);
        for i in 0..config.num_temporal_layers {
            let block = TransformerBlock::new(&config, vb.pp(format!("temporal_block_{}", i)))?;
            temporal_blocks.push(block);
        }

        // Final layer norm
        let norm = layer_norm(config.embed_dim, config.layer_norm_eps, vb.pp("norm"))?;

        let dropout = Dropout::new(config.dropout);

        Ok(Self {
            config,
            tubelet_embed,
            spatial_pos_embed,
            temporal_pos_embed,
            spatial_cls_token,
            temporal_cls_token,
            spatial_blocks,
            temporal_blocks,
            norm,
            head: None,
            dropout,
            device: device.clone(),
        })
    }

    /// Add a classification/projection head to the model.
    pub fn with_head(mut self, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        self.head = Some(linear(self.config.embed_dim, output_dim, vb.pp("head"))?);
        Ok(self)
    }

    /// Get the configuration.
    pub fn config(&self) -> &ViViTConfig {
        &self.config
    }

    /// Get the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, frames, channels, height, width)
    ///
    /// # Returns
    /// Output embedding of shape (batch, embed_dim) or (batch, output_dim) if head is attached
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_train(x, false)
    }

    /// Forward pass with training flag.
    pub fn forward_with_train(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let (batch, _frames, _channels, _height, _width) = x.dims5()?;

        // Extract tubelet embeddings
        // Shape: (batch, num_tubes, embed_dim) where num_tubes = t_tubes * h_patches * w_patches
        let x = self.tubelet_embed.forward(x)?;

        let num_patches = self.config.num_patches_per_frame();
        let num_temporal = self.config.num_temporal_tokens();

        // Reshape for spatial attention: (batch * num_temporal, num_patches, embed_dim)
        let x = x.reshape((batch * num_temporal, num_patches, self.config.embed_dim))?;

        // Add spatial CLS token and positional embedding
        let x = if let Some(ref cls_token) = self.spatial_cls_token {
            let cls_tokens =
                cls_token.broadcast_as((batch * num_temporal, 1, self.config.embed_dim))?;
            Tensor::cat(&[&cls_tokens, &x], 1)?
        } else {
            x
        };

        let x = x.broadcast_add(&self.spatial_pos_embed)?;
        let x = self.dropout.forward(&x, train)?;

        // Apply spatial transformer blocks
        let mut x = x;
        for block in &self.spatial_blocks {
            x = block.forward(&x, train)?;
        }

        // Extract spatial representation (CLS token or mean pool)
        let x = if self.config.use_cls_token {
            x.i((.., 0, ..))? // Take CLS token
        } else {
            x.mean(1)? // Mean pool
        };

        // Reshape for temporal attention: (batch, num_temporal, embed_dim)
        let x = x.reshape((batch, num_temporal, self.config.embed_dim))?;

        // Add temporal CLS token and positional embedding
        let x = if let Some(ref cls_token) = self.temporal_cls_token {
            let cls_tokens = cls_token.broadcast_as((batch, 1, self.config.embed_dim))?;
            Tensor::cat(&[&cls_tokens, &x], 1)?
        } else {
            x
        };

        let x = x.broadcast_add(&self.temporal_pos_embed)?;
        let x = self.dropout.forward(&x, train)?;

        // Apply temporal transformer blocks
        let mut x = x;
        for block in &self.temporal_blocks {
            x = block.forward(&x, train)?;
        }

        // Extract final representation
        let x = if self.config.use_cls_token {
            x.i((.., 0, ..))? // Take CLS token
        } else {
            x.mean(1)? // Mean pool
        };

        // Apply final layer norm
        let x = self.norm.forward(&x)?;

        // Apply head if present
        if let Some(ref head) = self.head {
            head.forward(&x)
        } else {
            Ok(x)
        }
    }

    /// Get intermediate features from spatial attention.
    ///
    /// Useful for visualization or multi-scale processing.
    pub fn get_spatial_features(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _frames, _channels, _height, _width) = x.dims5()?;

        let x = self.tubelet_embed.forward(x)?;

        let num_patches = self.config.num_patches_per_frame();
        let num_temporal = self.config.num_temporal_tokens();

        let x = x.reshape((batch * num_temporal, num_patches, self.config.embed_dim))?;

        let x = if let Some(ref cls_token) = self.spatial_cls_token {
            let cls_tokens =
                cls_token.broadcast_as((batch * num_temporal, 1, self.config.embed_dim))?;
            Tensor::cat(&[&cls_tokens, &x], 1)?
        } else {
            x
        };

        let x = x.broadcast_add(&self.spatial_pos_embed)?;

        let mut x = x;
        for block in &self.spatial_blocks {
            x = block.forward(&x, false)?;
        }

        // Return all spatial tokens (including CLS if present)
        Ok(x)
    }
}

/// Builder for creating ViViT models with custom configurations.
pub struct ViViTBuilder {
    config: ViViTConfig,
}

impl ViViTBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: ViViTConfig::default(),
        }
    }

    /// Set the number of input frames.
    pub fn num_frames(mut self, n: usize) -> Self {
        self.config.num_frames = n;
        self
    }

    /// Set the number of input channels.
    pub fn in_channels(mut self, n: usize) -> Self {
        self.config.in_channels = n;
        self
    }

    /// Set the input image dimensions.
    pub fn image_size(mut self, height: usize, width: usize) -> Self {
        self.config.image_height = height;
        self.config.image_width = width;
        self
    }

    /// Set the patch size.
    pub fn patch_size(mut self, size: usize) -> Self {
        self.config.patch_size = size;
        self
    }

    /// Set the tubelet depth.
    pub fn tubelet_depth(mut self, depth: usize) -> Self {
        self.config.tubelet_depth = depth;
        self
    }

    /// Set the embedding dimension.
    pub fn embed_dim(mut self, dim: usize) -> Self {
        self.config.embed_dim = dim;
        self
    }

    /// Set the number of attention heads.
    pub fn num_heads(mut self, n: usize) -> Self {
        self.config.num_heads = n;
        self
    }

    /// Set the number of spatial transformer layers.
    pub fn num_spatial_layers(mut self, n: usize) -> Self {
        self.config.num_spatial_layers = n;
        self
    }

    /// Set the number of temporal transformer layers.
    pub fn num_temporal_layers(mut self, n: usize) -> Self {
        self.config.num_temporal_layers = n;
        self
    }

    /// Set the dropout probability.
    pub fn dropout(mut self, p: f32) -> Self {
        self.config.dropout = p;
        self
    }

    /// Enable or disable CLS token.
    pub fn use_cls_token(mut self, use_cls: bool) -> Self {
        self.config.use_cls_token = use_cls;
        self
    }

    /// Build the ViViT model.
    pub fn build(self, device: &Device) -> Result<(ViViT, VarMap)> {
        ViViT::new(self.config, device)
    }

    /// Get the configuration.
    pub fn config(&self) -> &ViViTConfig {
        &self.config
    }
}

impl Default for ViViTBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    fn create_test_input(config: &ViViTConfig, device: &Device) -> Result<Tensor> {
        let shape = (
            2, // batch
            config.num_frames,
            config.in_channels,
            config.image_height,
            config.image_width,
        );
        let numel: usize = shape.0 * shape.1 * shape.2 * shape.3 * shape.4;
        let data: Vec<f32> = (0..numel).map(|i| (i as f32) / (numel as f32)).collect();
        Tensor::from_vec(data, shape, device)
    }

    #[test]
    fn test_vivit_creation() -> Result<()> {
        let device = Device::Cpu;
        let config = ViViTConfig::tiny();
        let (model, var_map) = ViViT::new(config.clone(), &device)?;

        assert_eq!(model.config().embed_dim, 64);
        assert_eq!(model.config().num_heads, 2);

        // Check that we have parameters
        let all_vars = var_map.all_vars();
        assert!(!all_vars.is_empty());

        Ok(())
    }

    #[test]
    fn test_vivit_forward() -> Result<()> {
        let device = Device::Cpu;
        let config = ViViTConfig::tiny();
        let (model, _var_map) = ViViT::new(config.clone(), &device)?;

        let input = create_test_input(&config, &device)?;
        let output = model.forward(&input)?;

        // Output should be (batch=2, embed_dim=64)
        assert_eq!(output.dims(), &[2, 64]);

        Ok(())
    }

    #[test]
    fn test_vivit_forward_small() -> Result<()> {
        let device = Device::Cpu;
        let config = ViViTConfig::small();
        let (model, _var_map) = ViViT::new(config.clone(), &device)?;

        let input = create_test_input(&config, &device)?;
        let output = model.forward(&input)?;

        // Output should be (batch=2, embed_dim=192)
        assert_eq!(output.dims(), &[2, 192]);

        Ok(())
    }

    #[test]
    fn test_vivit_without_cls_token() -> Result<()> {
        let device = Device::Cpu;
        let mut config = ViViTConfig::tiny();
        config.use_cls_token = false;

        let (model, _var_map) = ViViT::new(config.clone(), &device)?;
        let input = create_test_input(&config, &device)?;
        let output = model.forward(&input)?;

        assert_eq!(output.dims(), &[2, 64]);

        Ok(())
    }

    #[test]
    fn test_vivit_builder() -> Result<()> {
        let device = Device::Cpu;

        let (model, _var_map) = ViViTBuilder::new()
            .num_frames(4)
            .in_channels(3)
            .image_size(32, 32)
            .patch_size(8)
            .tubelet_depth(2)
            .embed_dim(128)
            .num_heads(4)
            .num_spatial_layers(2)
            .num_temporal_layers(2)
            .dropout(0.0)
            .build(&device)?;

        assert_eq!(model.config().embed_dim, 128);
        assert_eq!(model.config().num_heads, 4);

        // Test forward pass
        let input_shape = (2, 4, 3, 32, 32);
        let numel: usize =
            input_shape.0 * input_shape.1 * input_shape.2 * input_shape.3 * input_shape.4;
        let data: Vec<f32> = (0..numel).map(|i| (i as f32) / (numel as f32)).collect();
        let input = Tensor::from_vec(data, input_shape, &device)?;

        let output = model.forward(&input)?;
        assert_eq!(output.dims(), &[2, 128]);

        Ok(())
    }

    #[test]
    fn test_vivit_with_head() -> Result<()> {
        let device = Device::Cpu;
        let config = ViViTConfig::tiny();

        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let model = ViViT::from_vb(config.clone(), vb.pp("vivit"), &device)?;
        let model = model.with_head(3, vb.pp("vivit"))?; // 3 classes: buy, sell, hold

        let input = create_test_input(&config, &device)?;
        let output = model.forward(&input)?;

        // Output should be (batch=2, num_classes=3)
        assert_eq!(output.dims(), &[2, 3]);

        Ok(())
    }

    #[test]
    fn test_spatial_features() -> Result<()> {
        let device = Device::Cpu;
        let config = ViViTConfig::tiny();
        let (model, _var_map) = ViViT::new(config.clone(), &device)?;

        let input = create_test_input(&config, &device)?;
        let features = model.get_spatial_features(&input)?;

        // Should return all spatial tokens
        // num_patches = (16/4) * (16/4) = 16, +1 for CLS token = 17
        // num_temporal = 2/1 = 2
        // Shape: (batch * num_temporal, num_patches + 1, embed_dim) = (4, 17, 64)
        let expected_spatial_seq = config.num_patches_per_frame() + 1; // +1 for CLS
        let expected_batch = 2 * config.num_temporal_tokens();
        assert_eq!(features.dims(), &[expected_batch, expected_spatial_seq, 64]);

        Ok(())
    }

    #[test]
    fn test_config_calculations() {
        let config = ViViTConfig {
            num_frames: 8,
            in_channels: 5,
            image_height: 64,
            image_width: 64,
            patch_size: 8,
            tubelet_depth: 2,
            embed_dim: 384,
            num_heads: 6,
            num_spatial_layers: 4,
            num_temporal_layers: 4,
            mlp_ratio: 4.0,
            dropout: 0.1,
            attention_dropout: 0.0,
            use_cls_token: true,
            layer_norm_eps: 1e-6,
        };

        assert_eq!(config.num_patches_per_frame(), 64); // (64/8) * (64/8) = 8 * 8 = 64
        assert_eq!(config.num_temporal_tokens(), 4); // 8 / 2 = 4
    }

    #[test]
    fn test_tubelet_dimensions() -> Result<()> {
        // Verify that tubelet extraction produces correct dimensions
        let device = Device::Cpu;
        let config = ViViTConfig::tiny();
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let tubelet_embed = TubeletEmbed::new(&config, vb.pp("tubelet"))?;

        let input = create_test_input(&config, &device)?;
        let embedded = tubelet_embed.forward(&input)?;

        // Expected: (batch=2, num_tubes, embed_dim)
        // num_tubes = t_tubes * h_patches * w_patches
        //           = (2/1) * (16/4) * (16/4) = 2 * 4 * 4 = 32
        let t_tubes = config.num_frames / config.tubelet_depth;
        let h_patches = config.image_height / config.patch_size;
        let w_patches = config.image_width / config.patch_size;
        let num_tubes = t_tubes * h_patches * w_patches;

        assert_eq!(embedded.dims(), &[2, num_tubes, config.embed_dim]);

        Ok(())
    }
}
