//! ViViT (Video Vision Transformer) Component
//!
//! Implements the complete ViViT architecture for processing spatiotemporal data.
//! This includes tubelet embeddings, temporal attention mechanisms, and the
//! full transformer encoder stack.
//!
//! # Components
//!
//! - **Tubelet Embedding**: Converts 3D video volumes into patch embeddings
//! - **Temporal Attention**: Multi-head self-attention for temporal dependencies
//! - **ViViT Model**: Complete transformer architecture
//! - **Factorized Attention**: Efficient factorized spatiotemporal attention
//!
//! # Usage
//!
//! ```no_run
//! use janus_neuromorphic::visual_cortex::vivit::{VivitModel, VivitConfig};
//! use ndarray::Array5;
//!
//! # fn example() -> common::Result<()> {
//! // Create model
//! let config = VivitConfig::default();
//! let mut model = VivitModel::new(config);
//!
//! // Forward pass
//! let input = Array5::from_elem((1, 16, 224, 224, 3), 0.5);
//! let logits = model.forward(&input)?;
//! # Ok(())
//! # }
//! ```

pub mod factorized_attention;
pub mod persistence;
pub mod temporal_attention;
pub mod training;
pub mod tubelet_embedding;
pub mod vivit_candle;
pub mod vivit_model;

// Re-exports
pub use factorized_attention::FactorizedAttention;
pub use persistence::{
    CheckpointConfig, CheckpointFormat, CheckpointMetadata, ModelCheckpoint, ModelRegistry,
    ModelVersion, TrainingState,
};
pub use temporal_attention::{TemporalAttention, TemporalAttentionConfig};
pub use training::{
    AdamOptimizer, DataLoader, EpochMetrics, LRSchedule, Loss, LossType, OptimizerConfig,
    SGDOptimizer, Trainer, TrainingConfig,
};
pub use tubelet_embedding::{TubeletConfig, TubeletEmbedding};
pub use vivit_candle::{ViviTCandle, ViviTCandleConfig, ViviTTrainer};
pub use vivit_model::{ActivationType, VivitConfig, VivitModel};
