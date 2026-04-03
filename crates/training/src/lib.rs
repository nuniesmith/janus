//! # JANUS Training Infrastructure
//!
//! This crate provides the training infrastructure for Project JANUS, including:
//!
//! - **Optimizers**: Adam, AdamW, SGD with momentum
//! - **Learning Rate Schedulers**: Warmup, cosine annealing, step decay
//! - **Replay Buffers**: Prioritized experience replay with SWR sampling
//! - **Training Loop**: End-to-end training coordinator with Vision+LTN integration
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Training Infrastructure                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │  Optimizer   │  │   Scheduler  │  │    Replay    │     │
//! │  │ (AdamW/SGD)  │  │ (Warmup/Cos) │  │    Buffer    │     │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
//! │         │                  │                  │             │
//! │         └──────────────────┼──────────────────┘             │
//! │                            │                                │
//! │                   ┌────────▼────────┐                       │
//! │                   │ Training Loop   │                       │
//! │                   │  (Vision+LTN)   │                       │
//! │                   └─────────────────┘                       │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example: End-to-End Training
//!
//! ```ignore
//! use training::{
//!     TrainingLoop, TrainingConfig, OptimizerConfig,
//!     LRSchedulerConfig, ReplayBufferConfig, DefaultCallback
//! };
//! use candle_core::{Device, Tensor};
//!
//! // Configure training
//! let train_config = TrainingConfig::minimal()
//!     .device(Device::cuda_if_available(0)?);
//!
//! let opt_config = OptimizerConfig::adamw()
//!     .learning_rate(1e-4)
//!     .weight_decay(0.01)
//!     .build();
//!
//! let sched_config = LRSchedulerConfig::warmup_cosine()
//!     .warmup_steps(1000)
//!     .total_steps(100_000)
//!     .build();
//!
//! let replay_config = ReplayBufferConfig::default();
//!
//! // Create training loop
//! let mut training = TrainingLoop::new(
//!     train_config,
//!     opt_config,
//!     sched_config,
//!     replay_config,
//! )?;
//!
//! // Add logging callback
//! training.add_callback(Box::new(DefaultCallback));
//!
//! // Define loss functions
//! let task_loss_fn = |batch, var_map, device| {
//!     // Compute task-specific loss (e.g., MSE, CrossEntropy)
//!     // ... your forward pass here ...
//!     Ok(task_loss_tensor)
//! };
//!
//! let logic_loss_fn = |batch, var_map, device| {
//!     // Compute LTN satisfaction loss
//!     // ... your LTN rules here ...
//!     Ok(logic_loss_tensor)
//! };
//!
//! // Run training
//! let final_metrics = training.run(
//!     task_loss_fn,
//!     logic_loss_fn,
//!     None, // validation data (optional)
//!     None, // validation task loss fn
//!     None, // validation logic loss fn
//! )?;
//! ```

pub mod r#loop;
pub mod optimizer;
pub mod replay;
pub mod scheduler;

// Re-exports for convenience
pub use optimizer::{GradientClipper, OptimizerConfig, OptimizerType, OptimizerWrapper};

pub use scheduler::{
    LRScheduler, LRSchedulerConfig, SchedulerType, step_decay_schedule, warmup_cosine_schedule,
};

pub use replay::{
    Experience, ExperienceMetadata, PrioritizedReplayBuffer, ReplayBatch, ReplayBufferConfig,
    ReplayBufferStats, create_experience,
};

pub use r#loop::{
    CheckpointMetadata, DefaultCallback, StepMetrics, TrainingCallback, TrainingConfig,
    TrainingLoop, TrainingState, ValidationMetrics,
};

/// Training crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Training crate name
pub const CRATE_NAME: &str = "training";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_crate_name() {
        assert_eq!(CRATE_NAME, "training");
    }
}
