//! Vision Pipeline — end-to-end orchestration
//!
//! Chains **DiffGAF** (Candle) → **ViViT** (Candle) into a single differentiable
//! pipeline that turns raw time-series windows into embedding vectors suitable
//! for downstream heads (DQN, classification, LTN grounding, …).
//!
//! # Data flow
//!
//! ```text
//! Input  : (batch, total_steps, features)
//!            │
//!            ▼  split into `num_frames` windows
//! Windows: (batch * num_frames, window_size, features)
//!            │
//!            ▼  DiffGaf  (learnable normalisation → polar → Gramian)
//! GAF img: (batch * num_frames, features, image_size, image_size)
//!            │
//!            ▼  reshape
//! Video  : (batch, num_frames, channels, image_size, image_size)
//!            │
//!            ▼  ViViT  (tubelet embed → spatial attn → temporal attn)
//! Embed  : (batch, embed_dim)
//!            │
//!            ▼  optional linear head
//! Output : (batch, output_dim)   — or  (batch, embed_dim) without head
//! ```
//!
//! # Quick start
//!
//! ```rust,ignore
//! use candle_core::Device;
//! use vision::pipeline::{VisionPipeline, VisionPipelineConfig};
//!
//! let device = Device::Cpu;
//! let config = VisionPipelineConfig::small();
//! let (pipeline, var_map) = VisionPipeline::new(config, &device)?;
//!
//! // (batch=4, 256 timesteps, 4 OHLC features)
//! let input = candle_core::Tensor::randn(0f32, 1.0, (4, 256, 4), &device)?;
//! let output = pipeline.forward(&input)?;   // (4, embed_dim)
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap, linear};
use serde::{Deserialize, Serialize};

use crate::diff_gaf::{DiffGaf, DiffGafConfig, DiffGafMethod};
use crate::diffgaf::DiffGAFConfig;
use crate::error::{Result, VisionError};
use crate::vivit::{ViViT, ViViTConfig as FullViViTConfig};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Configuration for the vision pipeline.
///
/// This is the *user-facing* config that gets translated internally into the
/// concrete [`DiffGafConfig`] and [`FullViViTConfig`] used by the Candle
/// modules.  It keeps backward compatibility with code that was written
/// against the earlier stub API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionPipelineConfig {
    /// DiffGAF image-generation settings.
    pub gaf: DiffGAFConfig,
    /// Number of temporal frames the input is split into.
    pub num_frames: usize,
    /// ViViT transformer settings (simplified surface).
    pub vivit: ViViTConfig,
    /// Whether the DiffGAF should output dual (GASF + GADF) channels.
    /// When `true` the number of image channels fed to ViViT is
    /// `2 × gaf.input_features`; otherwise `gaf.input_features`.
    #[serde(default)]
    pub dual_gaf: bool,
}

/// Simplified ViViT knobs exposed at the pipeline level.
///
/// The full [`FullViViTConfig`] is derived from these values together with
/// the GAF output geometry at construction time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViViTConfig {
    /// Embedding / hidden dimension.
    pub embed_dim: usize,
    /// Number of multi-head attention heads.
    pub num_heads: usize,
    /// Total number of transformer layers (split evenly between spatial and
    /// temporal encoders).
    pub num_layers: usize,
    /// Feed-forward / MLP hidden dimension.
    pub mlp_dim: usize,
    /// Dropout probability.
    pub dropout: f64,
}

impl Default for ViViTConfig {
    fn default() -> Self {
        Self {
            embed_dim: 256,
            num_heads: 8,
            num_layers: 6,
            mlp_dim: 512,
            dropout: 0.1,
        }
    }
}

impl VisionPipelineConfig {
    // ----- presets ----------------------------------------------------------

    /// A small config suitable for quick experiments and unit tests.
    pub fn small() -> Self {
        Self {
            gaf: DiffGAFConfig {
                input_features: 4,
                output_size: 32,
                norm_range: (-1.0, 1.0),
                use_smooth_arccos: true,
                use_aggregation_weights: false,
                eps: 1e-7,
            },
            num_frames: 16,
            vivit: ViViTConfig {
                embed_dim: 128,
                num_heads: 4,
                num_layers: 4,
                mlp_dim: 256,
                dropout: 0.1,
            },
            dual_gaf: false,
        }
    }

    /// A medium config for production use.
    pub fn medium() -> Self {
        Self {
            gaf: DiffGAFConfig {
                input_features: 8,
                output_size: 64,
                norm_range: (-1.0, 1.0),
                use_smooth_arccos: true,
                use_aggregation_weights: false,
                eps: 1e-7,
            },
            num_frames: 32,
            vivit: ViViTConfig {
                embed_dim: 256,
                num_heads: 8,
                num_layers: 6,
                mlp_dim: 512,
                dropout: 0.1,
            },
            dual_gaf: false,
        }
    }

    /// A large / maximum-capacity config.
    pub fn large() -> Self {
        Self {
            gaf: DiffGAFConfig {
                input_features: 16,
                output_size: 128,
                norm_range: (-1.0, 1.0),
                use_smooth_arccos: true,
                use_aggregation_weights: false,
                eps: 1e-7,
            },
            num_frames: 64,
            vivit: ViViTConfig {
                embed_dim: 512,
                num_heads: 16,
                num_layers: 12,
                mlp_dim: 1024,
                dropout: 0.1,
            },
            dual_gaf: false,
        }
    }

    // ----- internal helpers ------------------------------------------------

    /// Derive the Candle [`DiffGafConfig`] from this pipeline config.
    fn to_diff_gaf_config(&self) -> DiffGafConfig {
        DiffGafConfig {
            num_features: self.gaf.input_features,
            image_size: self.gaf.output_size,
            method: DiffGafMethod::Summation,
            learnable_norm: true,
            eps: self.gaf.eps as f64,
        }
    }

    /// Derive the full Candle [`FullViViTConfig`] from this pipeline config.
    fn to_vivit_config(&self) -> FullViViTConfig {
        let in_channels = if self.dual_gaf {
            self.gaf.input_features * 2
        } else {
            self.gaf.input_features
        };

        // Pick a patch size that evenly divides the image; fall back to 8.
        let patch_size = best_patch_size(self.gaf.output_size);

        // Pick a tubelet depth that evenly divides num_frames; fall back to 2.
        let tubelet_depth = best_tubelet_depth(self.num_frames);

        let spatial_layers = self.vivit.num_layers / 2;
        let temporal_layers = self.vivit.num_layers - spatial_layers;
        let mlp_ratio = if self.vivit.embed_dim > 0 {
            self.vivit.mlp_dim as f32 / self.vivit.embed_dim as f32
        } else {
            4.0
        };

        FullViViTConfig {
            num_frames: self.num_frames,
            in_channels,
            image_height: self.gaf.output_size,
            image_width: self.gaf.output_size,
            patch_size,
            tubelet_depth,
            embed_dim: self.vivit.embed_dim,
            num_heads: self.vivit.num_heads,
            num_spatial_layers: spatial_layers,
            num_temporal_layers: temporal_layers,
            mlp_ratio,
            dropout: self.vivit.dropout as f32,
            attention_dropout: 0.0,
            use_cls_token: true,
            layer_norm_eps: 1e-6,
        }
    }

    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<()> {
        if self.gaf.input_features == 0 {
            return Err(VisionError::InvalidConfig(
                "input_features must be > 0".into(),
            ));
        }
        if self.gaf.output_size == 0 {
            return Err(VisionError::InvalidConfig("output_size must be > 0".into()));
        }
        if self.num_frames == 0 {
            return Err(VisionError::InvalidConfig("num_frames must be > 0".into()));
        }
        if self.vivit.embed_dim == 0 {
            return Err(VisionError::InvalidConfig("embed_dim must be > 0".into()));
        }
        if self.vivit.num_heads == 0 || self.vivit.embed_dim % self.vivit.num_heads != 0 {
            return Err(VisionError::InvalidConfig(
                "embed_dim must be divisible by num_heads".into(),
            ));
        }
        if self.vivit.num_layers == 0 {
            return Err(VisionError::InvalidConfig("num_layers must be > 0".into()));
        }
        if self.gaf.output_size % best_patch_size(self.gaf.output_size) != 0 {
            return Err(VisionError::InvalidConfig(
                "output_size must be divisible by a reasonable patch size (4, 8, 16, …)".into(),
            ));
        }
        Ok(())
    }
}

impl Default for VisionPipelineConfig {
    fn default() -> Self {
        Self::medium()
    }
}

// ---------------------------------------------------------------------------
// Output wrapper
// ---------------------------------------------------------------------------

/// Structured output from a pipeline forward pass.
#[derive(Debug)]
pub struct VisionPipelineOutput {
    /// Final embedding (or projected logits if a head is attached).
    ///
    /// Shape: `(batch, embed_dim)` or `(batch, output_dim)`.
    pub embedding: Tensor,
    /// Intermediate GAF image tensor before ViViT.
    ///
    /// Shape: `(batch, num_frames, channels, H, W)`.
    /// Only populated when `return_intermediates = true` in the forward call.
    pub gaf_images: Option<Tensor>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// End-to-end vision pipeline:  **DiffGAF → ViViT (→ optional head)**.
///
/// The pipeline is fully differentiable and participates in Candle's autograd
/// graph, so gradients flow from a downstream loss all the way back through
/// the ViViT transformer, through the DiffGAF image encoder, and into its
/// learnable normalisation parameters.
pub struct VisionPipeline {
    /// User-facing configuration.
    pub config: VisionPipelineConfig,

    /// Candle DiffGaf encoder (time series → GAF images).
    diff_gaf: DiffGaf,

    /// Candle ViViT transformer (video → embedding).
    vivit: ViViT,

    /// Optional linear projection head (embedding → output_dim).
    head: Option<Linear>,

    /// Computation device.
    device: Device,
}

impl VisionPipeline {
    // ----- constructors ----------------------------------------------------

    /// Create a new pipeline together with a fresh [`VarMap`] that owns all
    /// learnable parameters.
    ///
    /// This is the primary entry point for stand-alone usage and unit tests.
    ///
    /// ```rust,ignore
    /// let (pipeline, var_map) = VisionPipeline::new(config, &device)?;
    /// ```
    pub fn new(config: VisionPipelineConfig, device: &Device) -> Result<(Self, VarMap)> {
        config.validate()?;

        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let pipeline = Self::from_vb(config, vb, device)?;
        Ok((pipeline, var_map))
    }

    /// Create a pipeline from an externally-owned [`VarBuilder`].
    ///
    /// Use this when the pipeline's parameters should be part of a larger
    /// model's parameter set (e.g. a training loop's shared `VarMap`).
    ///
    /// ```rust,ignore
    /// let vb = VarBuilder::from_varmap(training.var_map(), DType::F32, &device);
    /// let pipeline = VisionPipeline::from_vb(config, vb.pp("vision"), &device)?;
    /// ```
    pub fn from_vb(config: VisionPipelineConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        config.validate()?;

        // ---- DiffGaf -------------------------------------------------------
        let gaf_config = config.to_diff_gaf_config();
        let diff_gaf = DiffGaf::new(gaf_config, vb.pp("diff_gaf"))
            .map_err(|e| VisionError::InternalError(format!("Failed to create DiffGaf: {e}")))?;

        // ---- ViViT ---------------------------------------------------------
        let vivit_config = config.to_vivit_config();
        let vivit = ViViT::from_vb(vivit_config, vb.pp("vivit"), device)
            .map_err(|e| VisionError::InternalError(format!("Failed to create ViViT: {e}")))?;

        Ok(Self {
            config,
            diff_gaf,
            vivit,
            head: None,
            device: device.clone(),
        })
    }

    // ----- builder helpers --------------------------------------------------

    /// Attach a linear projection head that maps the ViViT embedding to
    /// `output_dim` logits.
    ///
    /// Must be called **before** the first forward pass.  Returns `self` for
    /// chaining.
    pub fn with_head(mut self, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let head = linear(self.config.vivit.embed_dim, output_dim, vb.pp("head"))
            .map_err(|e| VisionError::InternalError(format!("Failed to create head: {e}")))?;
        self.head = Some(head);
        Ok(self)
    }

    // ----- forward ----------------------------------------------------------

    /// Inference-mode forward pass (no dropout).
    ///
    /// # Arguments
    ///
    /// * `x` — time-series input, one of:
    ///   - `(batch, total_steps, features)` — will be split into
    ///     `num_frames` windows automatically.
    ///   - `(total_steps, features)` — treated as batch = 1.
    ///
    /// # Returns
    ///
    /// Embedding tensor of shape `(batch, embed_dim)`, or
    /// `(batch, output_dim)` when a head is attached.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.forward_impl(x, false, false)?;
        Ok(out.embedding)
    }

    /// Training-mode forward pass (dropout active).
    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.forward_impl(x, true, false)?;
        Ok(out.embedding)
    }

    /// Forward pass returning both the embedding and the intermediate GAF
    /// video tensor — useful for visualisation / debugging.
    pub fn forward_with_intermediates(
        &self,
        x: &Tensor,
        train: bool,
    ) -> Result<VisionPipelineOutput> {
        self.forward_impl(x, train, true)
    }

    /// Core implementation shared by all forward variants.
    fn forward_impl(
        &self,
        x: &Tensor,
        train: bool,
        keep_intermediates: bool,
    ) -> Result<VisionPipelineOutput> {
        // -- 0. Normalise input shape ----------------------------------------
        let (x, batch) = self.normalise_input(x)?;
        // x is now (batch, total_steps, features)

        let total_steps = x.dim(1).map_err(candle_err)?;
        let features = x.dim(2).map_err(candle_err)?;
        let num_frames = self.config.num_frames;

        // -- 1. Split into frames --------------------------------------------
        // Each frame covers `window_size` contiguous time steps.
        let window_size = total_steps / num_frames;
        if window_size == 0 {
            return Err(VisionError::InvalidInput(format!(
                "total_steps ({total_steps}) must be >= num_frames ({num_frames})"
            )));
        }
        // Truncate any remainder so the reshape is exact.
        let usable_steps = window_size * num_frames;
        let x = if usable_steps < total_steps {
            x.narrow(1, 0, usable_steps).map_err(candle_err)?
        } else {
            x
        };

        // (batch, num_frames, window_size, features) → (batch*num_frames, window_size, features)
        let x = x
            .reshape((batch * num_frames, window_size, features))
            .map_err(candle_err)?;

        // -- 2. DiffGAF per frame -------------------------------------------
        let gaf_images = if self.config.dual_gaf {
            self.diff_gaf.forward_dual(&x).map_err(candle_err)?
        } else {
            DiffGaf::forward(&self.diff_gaf, &x).map_err(candle_err)?
        };
        // gaf_images: (batch*num_frames, channels, image_size, image_size)

        let channels = gaf_images.dim(1).map_err(candle_err)?;
        let h = gaf_images.dim(2).map_err(candle_err)?;
        let w = gaf_images.dim(3).map_err(candle_err)?;

        // -- 3. Reshape into a video tensor for ViViT -----------------------
        let video = gaf_images
            .reshape((batch, num_frames, channels, h, w))
            .map_err(candle_err)?;

        // -- 4. ViViT --------------------------------------------------------
        let embedding = self
            .vivit
            .forward_with_train(&video, train)
            .map_err(candle_err)?;
        // embedding: (batch, embed_dim)

        // -- 5. Optional head ------------------------------------------------
        let output = match &self.head {
            Some(head) => head.forward(&embedding).map_err(candle_err)?,
            None => embedding,
        };

        Ok(VisionPipelineOutput {
            embedding: output,
            gaf_images: if keep_intermediates {
                Some(video)
            } else {
                None
            },
        })
    }

    // ----- accessors --------------------------------------------------------

    /// The embedding dimension produced by the ViViT backbone (before the
    /// optional head).
    pub fn embed_dim(&self) -> usize {
        self.config.vivit.embed_dim
    }

    /// Rough estimate of the total number of learnable scalar parameters.
    pub fn num_params(&self) -> usize {
        let cfg = &self.config;

        // DiffGaf: 2 × num_features (gamma + beta) when learnable
        let gaf_params = 2 * cfg.gaf.input_features;

        // ViViT —
        // Tubelet projection:
        let in_ch = if cfg.dual_gaf {
            cfg.gaf.input_features * 2
        } else {
            cfg.gaf.input_features
        };
        let patch = best_patch_size(cfg.gaf.output_size);
        let tube_d = best_tubelet_depth(cfg.num_frames);
        let tubelet_proj =
            in_ch * tube_d * patch * patch * cfg.vivit.embed_dim + cfg.vivit.embed_dim; // bias

        // Positional embeddings
        let patches_per_frame = (cfg.gaf.output_size / patch) * (cfg.gaf.output_size / patch);
        let num_temporal = cfg.num_frames / tube_d;
        let pos_params = (patches_per_frame + 1) * cfg.vivit.embed_dim   // spatial + CLS
                       + (num_temporal + 1) * cfg.vivit.embed_dim; // temporal + CLS

        // CLS tokens
        let cls_params = 2 * cfg.vivit.embed_dim;

        // Transformer blocks: each has QKV + proj + MLP (fc1 + fc2) + 2 LN
        let spatial_layers = cfg.vivit.num_layers / 2;
        let temporal_layers = cfg.vivit.num_layers - spatial_layers;
        let total_layers = spatial_layers + temporal_layers;

        let attn_params = cfg.vivit.embed_dim * 3 * cfg.vivit.embed_dim // QKV
                        + 3 * cfg.vivit.embed_dim                        // QKV bias
                        + cfg.vivit.embed_dim * cfg.vivit.embed_dim      // proj
                        + cfg.vivit.embed_dim; // proj bias
        let mlp_params = cfg.vivit.embed_dim * cfg.vivit.mlp_dim         // fc1
                       + cfg.vivit.mlp_dim                                // bias
                       + cfg.vivit.mlp_dim * cfg.vivit.embed_dim         // fc2
                       + cfg.vivit.embed_dim; // bias
        let ln_params = 4 * cfg.vivit.embed_dim; // 2 LN × (weight + bias)
        let block_params = attn_params + mlp_params + ln_params;

        // Final LN
        let final_ln = 2 * cfg.vivit.embed_dim;

        // Optional head
        let head_params = match &self.head {
            Some(_h) => 0, // already counted separately if needed
            None => 0,
        };

        gaf_params
            + tubelet_proj
            + pos_params
            + cls_params
            + total_layers * block_params
            + final_ln
            + head_params
    }

    /// The computation device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Reference to the inner DiffGaf module.
    pub fn diff_gaf(&self) -> &DiffGaf {
        &self.diff_gaf
    }

    /// Reference to the inner ViViT module.
    pub fn vivit(&self) -> &ViViT {
        &self.vivit
    }

    /// Reference to the pipeline config.
    pub fn config(&self) -> &VisionPipelineConfig {
        &self.config
    }

    /// The number of image channels produced by DiffGAF (and consumed by
    /// ViViT).
    pub fn gaf_channels(&self) -> usize {
        if self.config.dual_gaf {
            self.config.gaf.input_features * 2
        } else {
            self.config.gaf.input_features
        }
    }

    // ----- private helpers --------------------------------------------------

    /// Ensure the input is 3-D `(batch, steps, features)`.
    fn normalise_input(&self, x: &Tensor) -> Result<(Tensor, usize)> {
        let dims = x.dims();
        match dims.len() {
            2 => {
                // (steps, features) → (1, steps, features)
                let x = x.unsqueeze(0).map_err(candle_err)?;
                Ok((x, 1))
            }
            3 => Ok((x.clone(), dims[0])),
            other => Err(VisionError::InvalidShape {
                expected: "(batch, steps, features) or (steps, features)".into(),
                actual: format!("{other}-D tensor with dims {dims:?}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Pick the largest "nice" patch size that evenly divides `image_size`.
fn best_patch_size(image_size: usize) -> usize {
    for &p in &[16, 8, 4, 2] {
        if image_size % p == 0 && image_size / p >= 2 {
            return p;
        }
    }
    1
}

/// Pick a tubelet depth that evenly divides `num_frames`.
fn best_tubelet_depth(num_frames: usize) -> usize {
    for &d in &[4, 2] {
        if num_frames % d == 0 {
            return d;
        }
    }
    1
}

/// Convert a [`candle_core::Error`] into a [`VisionError`].
fn candle_err(e: candle_core::Error) -> VisionError {
    VisionError::TensorError(e.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- config presets -------------------------------------------------------

    #[test]
    fn test_config_presets() {
        let small = VisionPipelineConfig::small();
        assert_eq!(small.num_frames, 16);
        assert_eq!(small.vivit.embed_dim, 128);

        let medium = VisionPipelineConfig::medium();
        assert_eq!(medium.num_frames, 32);
        assert_eq!(medium.vivit.embed_dim, 256);

        let large = VisionPipelineConfig::large();
        assert_eq!(large.num_frames, 64);
        assert_eq!(large.vivit.embed_dim, 512);
    }

    #[test]
    fn test_vivit_config_default() {
        let config = ViViTConfig::default();
        assert_eq!(config.embed_dim, 256);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.mlp_dim, 512);
        assert!(config.dropout > 0.0);
    }

    #[test]
    fn test_config_validation_ok() {
        assert!(VisionPipelineConfig::small().validate().is_ok());
        assert!(VisionPipelineConfig::medium().validate().is_ok());
        assert!(VisionPipelineConfig::large().validate().is_ok());
    }

    #[test]
    fn test_config_validation_bad_features() {
        let mut cfg = VisionPipelineConfig::small();
        cfg.gaf.input_features = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_bad_heads() {
        let mut cfg = VisionPipelineConfig::small();
        cfg.vivit.num_heads = 3; // 128 not divisible by 3
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_best_patch_size() {
        assert_eq!(best_patch_size(32), 16);
        assert_eq!(best_patch_size(64), 16);
        assert_eq!(best_patch_size(128), 16);
        assert_eq!(best_patch_size(12), 4);
        assert_eq!(best_patch_size(6), 2);
    }

    #[test]
    fn test_best_tubelet_depth() {
        assert_eq!(best_tubelet_depth(16), 4);
        assert_eq!(best_tubelet_depth(32), 4);
        assert_eq!(best_tubelet_depth(64), 4);
        assert_eq!(best_tubelet_depth(6), 2);
        assert_eq!(best_tubelet_depth(3), 1);
    }

    // -- construction --------------------------------------------------------

    #[test]
    fn test_pipeline_creation() -> Result<()> {
        let config = VisionPipelineConfig::small();
        let device = Device::Cpu;

        let (pipeline, _var_map) = VisionPipeline::new(config.clone(), &device)?;

        assert_eq!(pipeline.embed_dim(), config.vivit.embed_dim);
        assert!(pipeline.num_params() > 0);
        Ok(())
    }

    #[test]
    fn test_pipeline_from_vb() -> Result<()> {
        let config = VisionPipelineConfig::small();
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let pipeline = VisionPipeline::from_vb(config.clone(), vb, &device)?;
        assert_eq!(pipeline.embed_dim(), 128);
        Ok(())
    }

    // -- forward pass --------------------------------------------------------

    #[test]
    fn test_pipeline_forward_small() -> Result<()> {
        let config = VisionPipelineConfig::small();
        let device = Device::Cpu;
        let (pipeline, _) = VisionPipeline::new(config.clone(), &device)?;

        // batch=2, total_steps = num_frames(16) * 8 = 128, features = 4
        let x = Tensor::randn(0f32, 1.0, (2, 128, 4), &device).map_err(candle_err)?;
        let out = pipeline.forward(&x)?;

        assert_eq!(out.dims(), &[2, config.vivit.embed_dim]);
        Ok(())
    }

    #[test]
    fn test_pipeline_forward_2d_input() -> Result<()> {
        let config = VisionPipelineConfig::small();
        let device = Device::Cpu;
        let (pipeline, _) = VisionPipeline::new(config.clone(), &device)?;

        // No batch dim — should be treated as batch=1
        let x = Tensor::randn(0f32, 1.0, (128, 4), &device).map_err(candle_err)?;
        let out = pipeline.forward(&x)?;

        assert_eq!(out.dims(), &[1, config.vivit.embed_dim]);
        Ok(())
    }

    #[test]
    fn test_pipeline_forward_truncates_remainder() -> Result<()> {
        let config = VisionPipelineConfig::small(); // num_frames = 16
        let device = Device::Cpu;
        let (pipeline, _) = VisionPipeline::new(config.clone(), &device)?;

        // 130 is not divisible by 16 — should still work (truncate to 128).
        let x = Tensor::randn(0f32, 1.0, (1, 130, 4), &device).map_err(candle_err)?;
        let out = pipeline.forward(&x)?;

        assert_eq!(out.dims(), &[1, 128]);
        Ok(())
    }

    #[test]
    fn test_pipeline_forward_train() -> Result<()> {
        let config = VisionPipelineConfig::small();
        let device = Device::Cpu;
        let (pipeline, _) = VisionPipeline::new(config.clone(), &device)?;

        let x = Tensor::randn(0f32, 1.0, (2, 128, 4), &device).map_err(candle_err)?;
        let out = pipeline.forward_train(&x)?;

        assert_eq!(out.dims(), &[2, 128]);
        Ok(())
    }

    #[test]
    fn test_pipeline_with_intermediates() -> Result<()> {
        let config = VisionPipelineConfig::small();
        let device = Device::Cpu;
        let (pipeline, _) = VisionPipeline::new(config.clone(), &device)?;

        let x = Tensor::randn(0f32, 1.0, (2, 128, 4), &device).map_err(candle_err)?;
        let out = pipeline.forward_with_intermediates(&x, false)?;

        assert_eq!(out.embedding.dims(), &[2, 128]);
        assert!(out.gaf_images.is_some());

        let gaf = out.gaf_images.unwrap();
        // (batch=2, num_frames=16, channels=4, 32, 32)
        assert_eq!(gaf.dims(), &[2, 16, 4, 32, 32]);
        Ok(())
    }

    #[test]
    fn test_pipeline_gaf_channels() -> Result<()> {
        let mut config = VisionPipelineConfig::small();
        assert_eq!(config.gaf.input_features, 4);

        config.dual_gaf = false;
        let device = Device::Cpu;
        let (p, _) = VisionPipeline::new(config.clone(), &device)?;
        assert_eq!(p.gaf_channels(), 4);

        // dual GAF doubles the channels
        config.dual_gaf = true;
        let (p2, _) = VisionPipeline::new(config, &device)?;
        assert_eq!(p2.gaf_channels(), 8);
        Ok(())
    }

    #[test]
    fn test_pipeline_num_params_positive() -> Result<()> {
        let config = VisionPipelineConfig::small();
        let device = Device::Cpu;
        let (pipeline, _) = VisionPipeline::new(config, &device)?;
        let n = pipeline.num_params();
        assert!(n > 1_000, "expected >1k params, got {n}");
        Ok(())
    }

    #[test]
    fn test_pipeline_error_on_too_few_steps() -> Result<()> {
        let config = VisionPipelineConfig::small(); // needs ≥ 16 steps
        let device = Device::Cpu;
        let (pipeline, _) = VisionPipeline::new(config, &device)?;

        let x = Tensor::randn(0f32, 1.0, (1, 4, 4), &device).map_err(candle_err)?;
        let result = pipeline.forward(&x);
        assert!(result.is_err());
        Ok(())
    }
}
