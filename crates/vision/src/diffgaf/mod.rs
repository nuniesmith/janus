//! Differentiable Gramian Angular Field implementation
//!
//! This module provides learnable GAF transformations for time series data.

pub mod combined;
pub mod config;
pub mod layers;
pub mod transforms;

pub use combined::{
    BestModelTracker, CheckpointMetadata, ClassificationBatch, DiffGafLstm, DiffGafLstmConfig,
};
pub use config::DiffGAFConfig;
pub use layers::DiffGAF;
pub use transforms::{GramianLayer, LearnableNorm, PolarEncoder};
