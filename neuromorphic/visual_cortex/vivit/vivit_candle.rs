//! GPU-Accelerated ViViT Model using Candle
//!
//! This module provides a GPU-accelerated implementation of ViViT using the candle framework.
//! It supports automatic differentiation, GPU/CPU execution, and model persistence.
//!
//! # Features
//!
//! - Automatic differentiation for training
//! - GPU acceleration (CUDA/Metal)
//! - Model checkpointing and loading
//! - Mixed precision training (fp16/fp32)
//! - Distributed training support (future)
//!
//! # Example
//!
//! ```ignore
//! use janus_neuromorphic::visual_cortex::vivit::ViviTCandle;
//! use candle_core::{Device, Tensor};
//!
//! # fn example() -> candle_core::Result<()> {
//! let device = Device::cuda_if_available(0)?;
//! let model = ViviTCandle::new(&device)?;
//!
//! // Forward pass
//! let input = Tensor::randn(0f32, 1f32, (1, 16, 224, 224, 3), &device)?;
//! let logits = model.forward(&input)?;
//! # Ok(())
//! # }
//! ```

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::optim::{Optimizer, SGD};
use candle_nn::{Dropout, LayerNorm, Linear, VarBuilder, VarMap, layer_norm, linear};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// Configuration for GPU-accelerated ViViT model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViviTCandleConfig {
    /// Number of frames in video
    pub num_frames: usize,
    /// Frame height
    pub frame_height: usize,
    /// Frame width
    pub frame_width: usize,
    /// Number of channels (RGB = 3)
    pub num_channels: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP expansion ratio
    pub mlp_ratio: f64,
    /// Dropout probability
    pub dropout: f32,
    /// Number of output classes
    pub num_classes: usize,
    /// Tubelet temporal size
    pub tubelet_t: usize,
    /// Tubelet height
    pub tubelet_h: usize,
    /// Tubelet width
    pub tubelet_w: usize,
    /// Use CLS token
    pub use_cls_token: bool,
    /// Data type (fp32 or fp16)
    pub dtype: String,
}

impl Default for ViviTCandleConfig {
    fn default() -> Self {
        Self {
            num_frames: 16,
            frame_height: 224,
            frame_width: 224,
            num_channels: 3,
            embedding_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            dropout: 0.1,
            num_classes: 10,
            tubelet_t: 2,
            tubelet_h: 16,
            tubelet_w: 16,
            use_cls_token: true,
            dtype: "fp32".to_string(),
        }
    }
}

impl ViviTCandleConfig {
    /// Get the DType for tensors
    pub fn dtype(&self) -> DType {
        match self.dtype.as_str() {
            "fp16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::F32,
        }
    }

    /// Calculate number of patches
    pub fn num_patches(&self) -> usize {
        let t_patches = self.num_frames / self.tubelet_t;
        let h_patches = self.frame_height / self.tubelet_h;
        let w_patches = self.frame_width / self.tubelet_w;
        t_patches * h_patches * w_patches
    }

    /// Calculate total sequence length (patches + optional CLS token)
    pub fn seq_len(&self) -> usize {
        let n = self.num_patches();
        if self.use_cls_token { n + 1 } else { n }
    }
}

/// Tubelet embedding layer
struct TubeletEmbedding {
    projection: Linear,
    cls_token: Option<Tensor>,
    positional_embedding: Tensor,
    config: ViviTCandleConfig,
}

impl TubeletEmbedding {
    fn new(vb: VarBuilder, config: &ViviTCandleConfig) -> Result<Self> {
        let patch_dim =
            config.tubelet_t * config.tubelet_h * config.tubelet_w * config.num_channels;

        // Linear projection
        let projection = linear(patch_dim, config.embedding_dim, vb.pp("projection"))?;

        // CLS token
        let cls_token = if config.use_cls_token {
            Some(vb.get((1, 1, config.embedding_dim), "cls_token")?)
        } else {
            None
        };

        // Positional embeddings
        let seq_len = config.seq_len();
        let positional_embedding = vb.get((1, seq_len, config.embedding_dim), "pos_embed")?;

        Ok(Self {
            projection,
            cls_token,
            positional_embedding,
            config: config.clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, _t, _h, _w, _c) = x.dims5()?;

        // Extract tubelets: [B, T, H, W, C] -> [B, N, P]
        let tubelets = self.extract_tubelets(x)?;

        // Project: [B, N, P] -> [B, N, D]
        let mut embeddings = tubelets.apply(&self.projection)?;

        // Add CLS token if used
        if let Some(cls) = &self.cls_token {
            let cls_tokens = cls.broadcast_as((batch_size, 1, self.config.embedding_dim))?;
            embeddings = Tensor::cat(&[&cls_tokens, &embeddings], 1)?;
        }

        // Add positional embeddings
        embeddings = embeddings.broadcast_add(&self.positional_embedding)?;

        Ok(embeddings)
    }

    fn extract_tubelets(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, t, h, w, c) = x.dims5()?;

        let t_patches = t / self.config.tubelet_t;
        let h_patches = h / self.config.tubelet_h;
        let w_patches = w / self.config.tubelet_w;

        let mut patches = Vec::new();

        for b in 0..batch_size {
            for tp in 0..t_patches {
                for hp in 0..h_patches {
                    for wp in 0..w_patches {
                        let t_start = tp * self.config.tubelet_t;
                        let h_start = hp * self.config.tubelet_h;
                        let w_start = wp * self.config.tubelet_w;

                        let patch = x
                            .narrow(0, b, 1)?
                            .narrow(1, t_start, self.config.tubelet_t)?
                            .narrow(2, h_start, self.config.tubelet_h)?
                            .narrow(3, w_start, self.config.tubelet_w)?;

                        patches.push(patch.flatten_all()?);
                    }
                }
            }
        }

        let num_patches = t_patches * h_patches * w_patches;
        let patch_dim = self.config.tubelet_t * self.config.tubelet_h * self.config.tubelet_w * c;

        Tensor::stack(&patches, 0)?.reshape((batch_size, num_patches, patch_dim))
    }
}

/// Multi-head self-attention
struct MultiHeadAttention {
    qkv: Linear,
    proj: Linear,
    dropout: Dropout,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadAttention {
    fn new(vb: VarBuilder, config: &ViviTCandleConfig) -> Result<Self> {
        let head_dim = config.embedding_dim / config.num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let qkv = linear(config.embedding_dim, config.embedding_dim * 3, vb.pp("qkv"))?;
        let proj = linear(config.embedding_dim, config.embedding_dim, vb.pp("proj"))?;
        let dropout = Dropout::new(config.dropout);

        Ok(Self {
            qkv,
            proj,
            dropout,
            num_heads: config.num_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let (batch_size, seq_len, embed_dim) = x.dims3()?;

        // QKV projection: [B, N, D] -> [B, N, 3*D]
        let qkv = x.apply(&self.qkv)?;

        // Reshape: [B, N, 3*D] -> [B, N, 3, H, D/H]
        let qkv = qkv.reshape((batch_size, seq_len, 3, self.num_heads, self.head_dim))?;

        // Permute: [B, N, 3, H, D/H] -> [3, B, H, N, D/H]
        // Ensure contiguous layout after permute to avoid matmul stride issues with batch > 1
        let qkv = qkv.permute((2, 0, 3, 1, 4))?.contiguous()?;

        // Split Q, K, V — make each contiguous for reliable batched matmul
        let q = qkv.narrow(0, 0, 1)?.squeeze(0)?.contiguous()?; // [B, H, N, D/H]
        let k = qkv.narrow(0, 1, 1)?.squeeze(0)?.contiguous()?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(0)?.contiguous()?;

        // Attention scores: Q @ K^T / sqrt(d)
        let k_t = k.transpose(D::Minus1, D::Minus2)?.contiguous()?;
        let attn = q.matmul(&k_t)?;
        let attn = (attn * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn = self.dropout.forward(&attn, train)?;

        // Apply attention to values: [B, H, N, D/H]
        let out = attn.matmul(&v)?;

        // Reshape: [B, H, N, D/H] -> [B, N, H, D/H] -> [B, N, D]
        // Ensure contiguous before reshape to avoid stride mismatch
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, embed_dim))?;

        // Output projection
        let out = out.apply(&self.proj)?;
        let out = self.dropout.forward(&out, train)?;

        Ok(out)
    }
}

/// Feed-forward MLP
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl Mlp {
    fn new(vb: VarBuilder, config: &ViviTCandleConfig) -> Result<Self> {
        let hidden_dim = (config.embedding_dim as f64 * config.mlp_ratio) as usize;

        let fc1 = linear(config.embedding_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = linear(hidden_dim, config.embedding_dim, vb.pp("fc2"))?;
        let dropout = Dropout::new(config.dropout);

        Ok(Self { fc1, fc2, dropout })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = x.apply(&self.fc1)?;
        let x = x.gelu()?;
        let x = self.dropout.forward(&x, train)?;
        let x = x.apply(&self.fc2)?;
        let x = self.dropout.forward(&x, train)?;
        Ok(x)
    }
}

/// Transformer encoder block
struct TransformerBlock {
    norm1: LayerNorm,
    attn: MultiHeadAttention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    fn new(vb: VarBuilder, config: &ViviTCandleConfig) -> Result<Self> {
        let norm1 = layer_norm(config.embedding_dim, 1e-6, vb.pp("norm1"))?;
        let attn = MultiHeadAttention::new(vb.pp("attn"), config)?;
        let norm2 = layer_norm(config.embedding_dim, 1e-6, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), config)?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        // Pre-norm architecture
        let normed = x.apply(&self.norm1)?;
        let attn_out = self.attn.forward(&normed, train)?;
        let x = (x + attn_out)?;

        let normed = x.apply(&self.norm2)?;
        let mlp_out = self.mlp.forward(&normed, train)?;
        let x = (x + mlp_out)?;

        Ok(x)
    }
}

/// GPU-accelerated ViViT model
pub struct ViviTCandle {
    tubelet_embedding: TubeletEmbedding,
    blocks: Vec<TransformerBlock>,
    norm: LayerNorm,
    head: Linear,
    varmap: Arc<Mutex<VarMap>>,
    config: ViviTCandleConfig,
    device: Device,
    train_mode: bool,
}

impl ViviTCandle {
    /// Create a new ViViT model
    pub fn new(device: &Device, config: ViviTCandleConfig) -> Result<Self> {
        info!("Initializing GPU-accelerated ViViT model");
        info!("  Device: {:?}", device);
        info!("  Embedding dim: {}", config.embedding_dim);
        info!("  Num layers: {}", config.num_layers);
        info!("  Num heads: {}", config.num_heads);
        info!("  DType: {:?}", config.dtype());

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, config.dtype(), device);

        // Tubelet embedding
        let tubelet_embedding = TubeletEmbedding::new(vb.pp("tubelet_embed"), &config)?;

        // Transformer blocks
        let mut blocks = Vec::new();
        for i in 0..config.num_layers {
            let block = TransformerBlock::new(vb.pp(format!("blocks.{}", i)), &config)?;
            blocks.push(block);
        }

        // Final layer norm
        let norm = layer_norm(config.embedding_dim, 1e-6, vb.pp("norm"))?;

        // Classification head
        let head = linear(config.embedding_dim, config.num_classes, vb.pp("head"))?;

        info!("ViViT model initialized successfully");

        Ok(Self {
            tubelet_embedding,
            blocks,
            norm,
            head,
            varmap: Arc::new(Mutex::new(varmap)),
            config,
            device: device.clone(),
            train_mode: false,
        })
    }

    /// Load model from checkpoint
    ///
    /// Creates the model architecture first (populating the VarMap with randomly
    /// initialized variables), then overwrites those variables with the saved
    /// checkpoint weights. This two-step process is required because
    /// `VarMap::load` only updates variables that already exist in the map.
    pub fn from_checkpoint<P: AsRef<Path>>(
        path: P,
        device: &Device,
        config: ViviTCandleConfig,
    ) -> Result<Self> {
        info!("Loading ViViT model from checkpoint: {:?}", path.as_ref());

        // Step 1: Create the model with random init (populates VarMap keys)
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, config.dtype(), device);

        let tubelet_embedding = TubeletEmbedding::new(vb.pp("tubelet_embed"), &config)?;

        let mut blocks = Vec::new();
        for i in 0..config.num_layers {
            let block = TransformerBlock::new(vb.pp(format!("blocks.{}", i)), &config)?;
            blocks.push(block);
        }

        let norm = layer_norm(config.embedding_dim, 1e-6, vb.pp("norm"))?;
        let head = linear(config.embedding_dim, config.num_classes, vb.pp("head"))?;

        // Step 2: Load saved weights into the now-populated VarMap.
        // VarMap::load iterates over existing vars and overwrites their values
        // from the safetensors file, so the keys must already be present.
        let mut varmap = varmap;
        varmap.load(path)?;

        info!(
            "Model loaded successfully ({} parameters restored)",
            varmap.all_vars().len()
        );

        Ok(Self {
            tubelet_embedding,
            blocks,
            norm,
            head,
            varmap: Arc::new(Mutex::new(varmap)),
            config,
            device: device.clone(),
            train_mode: false,
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        debug!("Forward pass: input shape {:?}", x.shape());

        // Tubelet embedding: [B, T, H, W, C] -> [B, N, D]
        let mut x = self.tubelet_embedding.forward(x)?;

        // Transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            debug!("Encoder block {}/{}", i + 1, self.blocks.len());
            x = block.forward(&x, self.train_mode)?;
        }

        // Final norm
        x = x.apply(&self.norm)?;

        // Extract CLS token or average pooling
        let x = if self.config.use_cls_token {
            x.narrow(1, 0, 1)?.squeeze(1)?
        } else {
            x.mean(1)?
        };

        // Classification head
        let logits = x.apply(&self.head)?;

        debug!("Output logits shape: {:?}", logits.shape());
        Ok(logits)
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.train_mode = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.train_mode = false;
    }

    /// Save model checkpoint
    ///
    /// Persists all model parameters (weights & biases) to a safetensors file.
    /// The checkpoint can later be loaded with [`from_checkpoint`].
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        info!("Saving checkpoint to: {:?}", path.as_ref());
        let varmap = self.varmap.lock().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to lock VarMap for saving: {}", e))
        })?;
        varmap.save(path.as_ref())?;
        info!(
            "Checkpoint saved successfully ({} tensors)",
            varmap.all_vars().len()
        );
        Ok(())
    }

    /// Get a reference to the VarMap (for optimizer integration)
    pub fn varmap(&self) -> &Arc<Mutex<VarMap>> {
        &self.varmap
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get config
    pub fn config(&self) -> &ViviTCandleConfig {
        &self.config
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        // Approximate parameter count
        let embed_params = self.config.embedding_dim
            * (self.config.tubelet_t
                * self.config.tubelet_h
                * self.config.tubelet_w
                * self.config.num_channels);

        let block_params = self.config.num_layers
            * (
                // Attention: QKV + projection
                4 * self.config.embedding_dim * self.config.embedding_dim +
            // MLP
            2 * self.config.embedding_dim * (self.config.embedding_dim as f64 * self.config.mlp_ratio) as usize +
            // LayerNorm (2x)
            4 * self.config.embedding_dim
            );

        let head_params = self.config.embedding_dim * self.config.num_classes;

        embed_params + block_params + head_params
    }
}

/// Training utilities with full backward pass and optimizer support.
///
/// Uses candle's autograd through `VarMap` variables and SGD optimization.
/// The VarMap is shared with the model so parameter updates are reflected
/// in subsequent forward passes.
pub struct ViviTTrainer {
    model: ViviTCandle,
    #[allow(dead_code)]
    device: Device,
    learning_rate: f64,
    optimizer: SGD,
}

impl ViviTTrainer {
    /// Create a new trainer with SGD optimizer
    pub fn new(model: ViviTCandle, learning_rate: f64) -> Result<Self> {
        let device = model.device().clone();

        let varmap = model.varmap().lock().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to lock VarMap for optimizer: {}", e))
        })?;
        let all_vars = varmap.all_vars();
        let optimizer = SGD::new(all_vars, learning_rate)?;
        drop(varmap);

        Ok(Self {
            model,
            device,
            learning_rate,
            optimizer,
        })
    }

    /// Training step with forward pass, loss computation, backward pass, and parameter update.
    ///
    /// Returns the scalar loss value for this step.
    pub fn train_step(&mut self, inputs: &Tensor, targets: &Tensor) -> Result<f32> {
        self.model.train();

        // Forward pass (through Var-backed parameters, so gradients are tracked)
        let logits = self.model.forward(inputs)?;

        // Compute cross-entropy loss
        let loss = self.cross_entropy_loss(&logits, targets)?;

        // Backward pass: compute gradients and update parameters via SGD
        self.optimizer.backward_step(&loss)?;

        let loss_val = loss.to_scalar::<f32>()?;
        debug!("Train step loss: {:.6}", loss_val);
        Ok(loss_val)
    }

    /// Validation step (no gradient computation or parameter updates)
    pub fn val_step(&mut self, inputs: &Tensor, targets: &Tensor) -> Result<(f32, f32)> {
        self.model.eval();

        // Forward pass (no gradients)
        let logits = self.model.forward(inputs)?;

        // Loss
        let loss = self.cross_entropy_loss(&logits, targets)?;
        let loss_val = loss.to_scalar::<f32>()?;

        // Accuracy
        let preds = logits.argmax(D::Minus1)?;
        let correct = preds.eq(targets)?.to_dtype(DType::F32)?.sum_all()?;
        let total = targets.elem_count() as f32;
        let accuracy = correct.to_scalar::<f32>()? / total;

        Ok((loss_val, accuracy))
    }

    /// Save the current model checkpoint
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.model.save_checkpoint(path)
    }

    /// Get the current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Cross-entropy loss using candle's built-in implementation.
    ///
    /// Expects `logits` of shape `[N, C]` (raw logits) and `targets` of shape `[N]` (u32 class indices).
    fn cross_entropy_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        candle_nn::loss::cross_entropy(logits, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ViviTCandleConfig::default();
        assert_eq!(config.num_frames, 16);
        assert_eq!(config.embedding_dim, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
    }

    #[test]
    fn test_num_patches() {
        let config = ViviTCandleConfig::default();
        let n = config.num_patches();
        // (16/2) * (224/16) * (224/16) = 8 * 14 * 14 = 1568
        assert_eq!(n, 1568);
    }

    #[test]
    fn test_seq_len() {
        let config = ViviTCandleConfig::default();
        assert_eq!(config.seq_len(), 1569); // 1568 patches + 1 CLS token
    }

    #[test]
    fn test_model_creation() {
        let device = Device::cuda_if_available(0).unwrap();
        let config = ViviTCandleConfig::default();
        let model = ViviTCandle::new(&device, config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_pass() {
        let device = Device::cuda_if_available(0).unwrap();
        if !matches!(device, Device::Cuda(_)) {
            eprintln!(
                "test_forward_pass: no CUDA device found — skipping full-size ViViT forward \
                 pass (>300 s on CPU). Use test_forward_pass_cpu for local CPU runs."
            );
            return;
        }
        let config = ViviTCandleConfig::default();
        let model = ViviTCandle::new(&device, config).unwrap();

        let input = Tensor::randn(0f32, 1f32, (1, 16, 224, 224, 3), &device).unwrap();
        let output = model.forward(&input);
        assert!(output.is_ok());

        let logits = output.unwrap();
        assert_eq!(logits.dims2().unwrap(), (1, 10)); // batch_size=1, num_classes=10
    }

    #[test]
    fn test_parameter_count() {
        let device = Device::Cpu;
        let config = ViviTCandleConfig::default();
        let model = ViviTCandle::new(&device, config).unwrap();
        let params = model.num_parameters();
        assert!(params > 1_000_000); // Should have millions of parameters
    }

    // CPU-based tests (no GPU required)

    #[test]
    fn test_model_creation_cpu() {
        let device = Device::Cpu;
        let config = ViviTCandleConfig {
            num_frames: 4,
            frame_height: 32,
            frame_width: 32,
            num_channels: 3,
            embedding_dim: 64,
            num_heads: 2,
            num_layers: 2,
            mlp_ratio: 2.0,
            num_classes: 5,
            dropout: 0.0,
            tubelet_t: 2,
            tubelet_h: 8,
            tubelet_w: 8,
            use_cls_token: true,
            dtype: "fp32".to_string(),
        };
        let model = ViviTCandle::new(&device, config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_pass_cpu() {
        let device = Device::Cpu;
        // Use smaller config for faster CPU testing
        let config = ViviTCandleConfig {
            num_frames: 4,
            frame_height: 32,
            frame_width: 32,
            num_channels: 3,
            embedding_dim: 64,
            num_heads: 2,
            num_layers: 2,
            mlp_ratio: 2.0,
            num_classes: 5,
            dropout: 0.0,
            tubelet_t: 2,
            tubelet_h: 8,
            tubelet_w: 8,
            use_cls_token: true,
            dtype: "fp32".to_string(),
        };
        let model = ViviTCandle::new(&device, config).unwrap();

        // Input shape: (batch, frames, height, width, channels)
        let input = Tensor::randn(0f32, 1f32, (1, 4, 32, 32, 3), &device).unwrap();
        let output = model.forward(&input);
        assert!(output.is_ok());

        let logits = output.unwrap();
        assert_eq!(logits.dims2().unwrap(), (1, 5)); // batch_size=1, num_classes=5
    }

    #[test]
    fn test_batch_forward_cpu() {
        // Batch forward with batch_size > 1 — previously failed due to non-contiguous
        // tensor striding in attention. Fixed by adding .contiguous() calls after
        // permute/transpose in MultiHeadAttention::forward.
        let device = Device::Cpu;
        let config = ViviTCandleConfig {
            num_frames: 4,
            frame_height: 32,
            frame_width: 32,
            num_channels: 3,
            embedding_dim: 64,
            num_heads: 2,
            num_layers: 2,
            mlp_ratio: 2.0,
            num_classes: 5,
            dropout: 0.0,
            tubelet_t: 2,
            tubelet_h: 8,
            tubelet_w: 8,
            use_cls_token: true,
            dtype: "fp32".to_string(),
        };
        let model = ViviTCandle::new(&device, config).unwrap();

        // batch_size = 3
        let input = Tensor::randn(0f32, 1f32, (3, 4, 32, 32, 3), &device).unwrap();
        let output = model.forward(&input);
        assert!(output.is_ok(), "Batch forward failed: {:?}", output.err());

        let logits = output.unwrap();
        assert_eq!(logits.dims2().unwrap(), (3, 5)); // batch_size=3, num_classes=5
    }

    #[test]
    fn test_checkpoint_save_load() {
        let device = Device::Cpu;
        let config = ViviTCandleConfig {
            num_frames: 4,
            frame_height: 32,
            frame_width: 32,
            num_channels: 3,
            embedding_dim: 64,
            num_heads: 2,
            num_layers: 2,
            mlp_ratio: 2.0,
            num_classes: 5,
            dropout: 0.0,
            tubelet_t: 2,
            tubelet_h: 8,
            tubelet_w: 8,
            use_cls_token: true,
            dtype: "fp32".to_string(),
        };
        let model = ViviTCandle::new(&device, config.clone()).unwrap();

        // Forward pass to get baseline output
        let input = Tensor::randn(0f32, 1f32, (1, 4, 32, 32, 3), &device).unwrap();
        let logits_before = model.forward(&input).unwrap();

        // Save checkpoint
        let tmp_dir = std::env::temp_dir();
        let checkpoint_path = tmp_dir.join("vivit_test_checkpoint.safetensors");
        model.save_checkpoint(&checkpoint_path).unwrap();
        assert!(checkpoint_path.exists());

        // Load checkpoint into new model
        let loaded = ViviTCandle::from_checkpoint(&checkpoint_path, &device, config).unwrap();
        let logits_after = loaded.forward(&input).unwrap();

        // Outputs should match
        let diff = (logits_before - logits_after)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap();
        let diff_val: f32 = diff.to_scalar().unwrap();
        assert!(
            diff_val < 1e-5,
            "Checkpoint load mismatch: diff = {}",
            diff_val
        );

        // Cleanup
        let _ = std::fs::remove_file(&checkpoint_path);
    }

    #[test]
    fn test_trainer_train_step() {
        let device = Device::Cpu;
        let config = ViviTCandleConfig {
            num_frames: 4,
            frame_height: 32,
            frame_width: 32,
            num_channels: 3,
            embedding_dim: 64,
            num_heads: 2,
            num_layers: 2,
            mlp_ratio: 2.0,
            num_classes: 5,
            dropout: 0.0,
            tubelet_t: 2,
            tubelet_h: 8,
            tubelet_w: 8,
            use_cls_token: true,
            dtype: "fp32".to_string(),
        };
        let model = ViviTCandle::new(&device, config).unwrap();
        let mut trainer = ViviTTrainer::new(model, 0.01).unwrap();

        let input = Tensor::randn(0f32, 1f32, (1, 4, 32, 32, 3), &device).unwrap();
        let targets = Tensor::new(&[2u32], &device).unwrap();

        // First training step should succeed and return a finite loss
        let loss = trainer.train_step(&input, &targets).unwrap();
        assert!(
            loss.is_finite(),
            "Training loss should be finite, got {}",
            loss
        );
        assert!(
            loss > 0.0,
            "Cross-entropy loss should be positive, got {}",
            loss
        );
    }
}
