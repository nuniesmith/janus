//! # Training Loop Coordinator
//!
//! End-to-end training loop that integrates:
//! - Vision pipeline (DiffGAF + ViViT)
//! - Logic Tensor Networks (LTN) for neuro-symbolic constraints
//! - Prioritized replay buffer with SWR sampling
//! - Optimizer and learning rate scheduler
//! - Checkpointing and model versioning
//! - Metrics and telemetry
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Training Loop                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  1. Sample Batch ──→ Replay Buffer (Prioritized + SWR)      │
//! │         ↓                                                    │
//! │  2. Forward Pass ──→ Vision Pipeline (DiffGAF + ViViT)      │
//! │         ↓                                                    │
//! │  3. Task Loss ──→ Prediction Loss (MSE/CrossEntropy)        │
//! │         ↓                                                    │
//! │  4. Logic Loss ──→ LTN Satisfaction Loss                    │
//! │         ↓                                                    │
//! │  5. Combined Loss = task_loss + λ * logic_loss              │
//! │         ↓                                                    │
//! │  6. Backward Pass ──→ Compute Gradients                     │
//! │         ↓                                                    │
//! │  7. Optimizer Step ──→ Update Weights (AdamW/SGD)           │
//! │         ↓                                                    │
//! │  8. Update Priorities ──→ TD Errors to Replay Buffer        │
//! │         ↓                                                    │
//! │  9. LR Schedule ──→ Update Learning Rate                    │
//! │         ↓                                                    │
//! │  10. Checkpoint ──→ Save Model (every N steps)              │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::{
    optimizer::{OptimizerConfig, OptimizerWrapper},
    replay::{Experience, PrioritizedReplayBuffer, ReplayBatch, ReplayBufferConfig},
    scheduler::{LRScheduler, LRSchedulerConfig},
};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Training loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training steps
    pub num_steps: usize,

    /// Batch size for training
    pub batch_size: usize,

    /// Weight for combining logic loss with task loss
    /// total_loss = task_loss + logic_weight * logic_loss
    pub logic_weight: f64,

    /// Gradient clipping maximum norm (None = no clipping)
    pub max_grad_norm: Option<f64>,

    /// Checkpoint frequency (save every N steps)
    pub checkpoint_every: usize,

    /// Validation frequency (validate every N steps)
    pub validate_every: usize,

    /// Log metrics every N steps
    pub log_every: usize,

    /// Checkpoint directory
    pub checkpoint_dir: PathBuf,

    /// Maximum number of checkpoints to keep (oldest deleted)
    pub max_checkpoints: usize,

    /// Early stopping: stop if validation loss doesn't improve for N validations
    pub early_stopping_patience: Option<usize>,

    /// Device for training
    #[serde(skip, default = "default_device")]
    pub device: Device,
}

fn default_device() -> Device {
    Device::Cpu
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_steps: 100_000,
            batch_size: 32,
            logic_weight: 0.5,
            max_grad_norm: Some(1.0),
            checkpoint_every: 5_000,
            validate_every: 1_000,
            log_every: 100,
            checkpoint_dir: PathBuf::from("checkpoints"),
            max_checkpoints: 10,
            early_stopping_patience: Some(10),
            device: Device::Cpu,
        }
    }
}

impl TrainingConfig {
    /// Create a minimal config for quick testing
    pub fn minimal() -> Self {
        Self {
            num_steps: 1_000,
            batch_size: 8,
            checkpoint_every: 500,
            validate_every: 100,
            log_every: 10,
            ..Default::default()
        }
    }

    /// Create a production config for full training
    pub fn production() -> Self {
        Self {
            num_steps: 1_000_000,
            batch_size: 64,
            checkpoint_every: 10_000,
            validate_every: 2_000,
            log_every: 100,
            max_checkpoints: 20,
            ..Default::default()
        }
    }

    /// Set device
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

/// Training metrics for a single step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub step: usize,
    pub task_loss: f64,
    pub logic_loss: f64,
    pub total_loss: f64,
    pub learning_rate: f64,
    pub grad_norm: Option<f64>,
    pub step_duration_ms: u128,
    pub replay_buffer_size: usize,
    pub avg_priority: f64,
}

impl StepMetrics {
    /// Create a new step metrics
    pub fn new(step: usize) -> Self {
        Self {
            step,
            task_loss: 0.0,
            logic_loss: 0.0,
            total_loss: 0.0,
            learning_rate: 0.0,
            grad_norm: None,
            step_duration_ms: 0,
            replay_buffer_size: 0,
            avg_priority: 1.0,
        }
    }

    /// Format metrics for logging
    pub fn format(&self) -> String {
        format!(
            "Step {}: loss={:.4} (task={:.4}, logic={:.4}), lr={:.6}, grad_norm={:.4}, duration={}ms, buffer_size={}",
            self.step,
            self.total_loss,
            self.task_loss,
            self.logic_loss,
            self.learning_rate,
            self.grad_norm.unwrap_or(0.0),
            self.step_duration_ms,
            self.replay_buffer_size,
        )
    }

    /// Convert to Prometheus-style metrics
    pub fn to_prometheus_labels(&self) -> HashMap<String, String> {
        let mut labels = HashMap::new();
        labels.insert("step".to_string(), self.step.to_string());
        labels.insert("task_loss".to_string(), self.task_loss.to_string());
        labels.insert("logic_loss".to_string(), self.logic_loss.to_string());
        labels.insert("total_loss".to_string(), self.total_loss.to_string());
        labels.insert("learning_rate".to_string(), self.learning_rate.to_string());
        labels.insert(
            "replay_buffer_size".to_string(),
            self.replay_buffer_size.to_string(),
        );
        labels
    }
}

/// Validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub step: usize,
    pub val_task_loss: f64,
    pub val_logic_loss: f64,
    pub val_total_loss: f64,
    pub num_samples: usize,
}

impl ValidationMetrics {
    pub fn format(&self) -> String {
        format!(
            "Validation @ Step {}: val_loss={:.4} (task={:.4}, logic={:.4}), samples={}",
            self.step,
            self.val_total_loss,
            self.val_task_loss,
            self.val_logic_loss,
            self.num_samples
        )
    }
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub timestamp: String,
    pub metrics: StepMetrics,
    pub validation_metrics: Option<ValidationMetrics>,
    pub model_version: String,
}

/// Training state that can be saved/loaded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub step: usize,
    pub best_val_loss: f64,
    pub patience_counter: usize,
    pub metadata: CheckpointMetadata,
}

/// Callback trait for custom training logic
pub trait TrainingCallback: Send + Sync {
    /// Called at the start of training
    fn on_train_start(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called at the end of each step
    fn on_step_end(&mut self, _metrics: &StepMetrics) -> Result<()> {
        Ok(())
    }

    /// Called after validation
    fn on_validation_end(&mut self, _metrics: &ValidationMetrics) -> Result<()> {
        Ok(())
    }

    /// Called after checkpoint save
    fn on_checkpoint_saved(&mut self, _path: &Path, _metadata: &CheckpointMetadata) -> Result<()> {
        Ok(())
    }

    /// Called at the end of training
    fn on_train_end(&mut self, _final_metrics: &StepMetrics) -> Result<()> {
        Ok(())
    }

    /// Called if training stops early
    fn on_early_stopping(&mut self, _step: usize) -> Result<()> {
        Ok(())
    }
}

/// Default callback that logs to stdout
pub struct DefaultCallback;

impl TrainingCallback for DefaultCallback {
    fn on_step_end(&mut self, metrics: &StepMetrics) -> Result<()> {
        println!("{}", metrics.format());
        Ok(())
    }

    fn on_validation_end(&mut self, metrics: &ValidationMetrics) -> Result<()> {
        println!("{}", metrics.format());
        Ok(())
    }

    fn on_checkpoint_saved(&mut self, path: &Path, _metadata: &CheckpointMetadata) -> Result<()> {
        println!("✓ Checkpoint saved: {}", path.display());
        Ok(())
    }

    fn on_early_stopping(&mut self, step: usize) -> Result<()> {
        println!("⚠ Early stopping triggered at step {}", step);
        Ok(())
    }
}

/// Main training loop coordinator
pub struct TrainingLoop {
    config: TrainingConfig,
    var_map: VarMap,
    optimizer: OptimizerWrapper,
    scheduler: LRScheduler,
    replay_buffer: PrioritizedReplayBuffer<Tensor, Tensor>,
    state: TrainingState,
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl TrainingLoop {
    /// Create a new training loop
    pub fn new(
        config: TrainingConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: LRSchedulerConfig,
        replay_config: ReplayBufferConfig,
    ) -> Result<Self> {
        let var_map = VarMap::new();
        let optimizer = OptimizerWrapper::from_config(&optimizer_config, var_map.clone())
            .context("Failed to create optimizer")?;

        let base_lr = optimizer_config.learning_rate;
        let scheduler = LRScheduler::from_config(scheduler_config, base_lr);

        let replay_buffer = PrioritizedReplayBuffer::new(replay_config);

        // Initialize training state
        let metadata = CheckpointMetadata {
            step: 0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            metrics: StepMetrics::new(0),
            validation_metrics: None,
            model_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        let state = TrainingState {
            step: 0,
            best_val_loss: f64::INFINITY,
            patience_counter: 0,
            metadata,
        };

        Ok(Self {
            config,
            var_map,
            optimizer,
            scheduler,
            replay_buffer,
            state,
            callbacks: vec![],
        })
    }

    /// Add a callback
    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Get reference to VarMap for building models
    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }

    /// Get reference to replay buffer
    pub fn replay_buffer(&self) -> &PrioritizedReplayBuffer<Tensor, Tensor> {
        &self.replay_buffer
    }

    /// Get mutable reference to replay buffer
    pub fn replay_buffer_mut(&mut self) -> &mut PrioritizedReplayBuffer<Tensor, Tensor> {
        &mut self.replay_buffer
    }

    /// Add experience to replay buffer
    pub fn add_experience(&mut self, experience: Experience<Tensor, Tensor>) {
        self.replay_buffer.add(experience);
    }

    /// Run a single training step
    ///
    /// # Arguments
    ///
    /// * `task_loss_fn` - Function that computes task loss given a batch
    /// * `logic_loss_fn` - Function that computes logic loss given a batch
    ///
    /// Returns the step metrics
    pub fn train_step<F, G>(&mut self, task_loss_fn: F, logic_loss_fn: G) -> Result<StepMetrics>
    where
        F: FnOnce(&ReplayBatch<Tensor, Tensor>, &VarMap, &Device) -> Result<Tensor>,
        G: FnOnce(&ReplayBatch<Tensor, Tensor>, &VarMap, &Device) -> Result<Tensor>,
    {
        let step_start = Instant::now();
        let step = self.state.step;

        // 1. Sample batch from replay buffer
        let batch = self
            .replay_buffer
            .sample(self.config.batch_size)
            .context("Failed to sample batch from replay buffer")?;

        // 2. Compute task loss
        let task_loss = task_loss_fn(&batch, &self.var_map, &self.config.device)
            .context("Failed to compute task loss")?;

        // 3. Compute logic loss
        let logic_loss = logic_loss_fn(&batch, &self.var_map, &self.config.device)
            .context("Failed to compute logic loss")?;

        // 4. Combine losses
        let logic_weight = Tensor::new(self.config.logic_weight as f32, &self.config.device)?;
        let total_loss =
            (&task_loss + &logic_loss.mul(&logic_weight)?).context("Failed to combine losses")?;

        // 5. Backward pass
        let grads = total_loss
            .backward()
            .context("Failed to compute gradients")?;

        // 6. Compute gradient norm (for logging)
        // Note: Gradient clipping is handled by optimizer if configured
        let grad_norm = self.config.max_grad_norm.map(|_max_norm| {
            // Compute global L2 norm across all gradient tensors in the VarMap
            let mut total_norm_sq: f64 = 0.0;
            let all_vars = self.var_map.all_vars();
            for var in all_vars.iter() {
                // GradStore::get returns Option<Tensor>
                if let Some(grad) = grads.get(var)
                    && let Ok(flat) = grad.flatten_all()
                    && let Ok(sq) = flat.sqr()
                    && let Ok(sq_sum) = sq.sum_all()
                    && let Ok(val) = sq_sum.to_scalar::<f32>()
                {
                    total_norm_sq += val as f64;
                }
            }
            total_norm_sq.sqrt()
        });

        // 7. Optimizer step
        self.optimizer
            .step(&grads)
            .context("Failed to perform optimizer step")?;

        // 8. Update learning rate
        let lr = self.scheduler.get_lr(step);
        self.optimizer.set_learning_rate(lr);

        // 9. Update replay buffer priorities based on TD error
        // Use total loss as a proxy for TD error (same value for all experiences in batch)
        let loss_scalar = total_loss.to_scalar::<f32>()?;
        let td_errors: Vec<f32> = vec![loss_scalar.abs(); batch.experiences.len()];
        self.replay_buffer
            .update_priorities(&batch.indices, &td_errors);

        // 10. Collect metrics
        let stats = self.replay_buffer.stats();
        let mut metrics = StepMetrics::new(step);
        metrics.task_loss = task_loss.to_vec0::<f32>()? as f64;
        metrics.logic_loss = logic_loss.to_vec0::<f32>()? as f64;
        metrics.total_loss = total_loss.to_vec0::<f32>()? as f64;
        metrics.learning_rate = lr;
        metrics.grad_norm = grad_norm;
        metrics.step_duration_ms = step_start.elapsed().as_millis();
        metrics.replay_buffer_size = stats.size;
        metrics.avg_priority = stats.avg_priority as f64;

        // Update state
        self.state.step += 1;

        Ok(metrics)
    }

    /// Run validation on a validation dataset
    pub fn validate<F, G>(
        &self,
        val_experiences: &[Experience<Tensor, Tensor>],
        task_loss_fn: F,
        logic_loss_fn: G,
    ) -> Result<ValidationMetrics>
    where
        F: Fn(&[Experience<Tensor, Tensor>], &VarMap, &Device) -> Result<Tensor>,
        G: Fn(&[Experience<Tensor, Tensor>], &VarMap, &Device) -> Result<Tensor>,
    {
        let task_loss = task_loss_fn(val_experiences, &self.var_map, &self.config.device)?;
        let logic_loss = logic_loss_fn(val_experiences, &self.var_map, &self.config.device)?;

        let logic_weight = Tensor::new(self.config.logic_weight as f32, &self.config.device)?;
        let total_loss = task_loss.add(&logic_loss.mul(&logic_weight)?)?;

        Ok(ValidationMetrics {
            step: self.state.step,
            val_task_loss: task_loss.to_vec0::<f32>()? as f64,
            val_logic_loss: logic_loss.to_vec0::<f32>()? as f64,
            val_total_loss: total_loss.to_vec0::<f32>()? as f64,
            num_samples: val_experiences.len(),
        })
    }

    /// Save checkpoint
    pub fn save_checkpoint(&self, metrics: &StepMetrics) -> Result<PathBuf> {
        let checkpoint_dir = &self.config.checkpoint_dir;
        std::fs::create_dir_all(checkpoint_dir).context("Failed to create checkpoint directory")?;

        let checkpoint_name = format!("checkpoint_step_{:08}.safetensors", self.state.step);
        let checkpoint_path = checkpoint_dir.join(&checkpoint_name);

        // Save model weights
        self.var_map
            .save(&checkpoint_path)
            .context("Failed to save model weights")?;

        // Save metadata
        let metadata = CheckpointMetadata {
            step: self.state.step,
            timestamp: chrono::Utc::now().to_rfc3339(),
            metrics: metrics.clone(),
            validation_metrics: self.state.metadata.validation_metrics.clone(),
            model_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        let metadata_path =
            checkpoint_dir.join(format!("checkpoint_step_{:08}.json", self.state.step));
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(&metadata_path, metadata_json)
            .context("Failed to save checkpoint metadata")?;

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(checkpoint_path)
    }

    /// Load checkpoint from path
    pub fn load_checkpoint(&mut self, checkpoint_path: &Path) -> Result<()> {
        self.var_map
            .load(checkpoint_path)
            .context("Failed to load model weights")?;

        // Try to load metadata
        let metadata_path = checkpoint_path.with_extension("json");
        if metadata_path.exists() {
            let metadata_json = std::fs::read_to_string(&metadata_path)?;
            let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;
            self.state.step = metadata.step;
            self.state.metadata = metadata;
        }

        Ok(())
    }

    /// Cleanup old checkpoints (keep only max_checkpoints)
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let checkpoint_dir = &self.config.checkpoint_dir;
        let mut checkpoints: Vec<_> = std::fs::read_dir(checkpoint_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension().and_then(|s| s.to_str()) == Some("safetensors")
            })
            .collect();

        if checkpoints.len() <= self.config.max_checkpoints {
            return Ok(());
        }

        // Sort by modification time
        checkpoints.sort_by_key(|entry| {
            entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });

        // Delete oldest checkpoints
        let to_delete = checkpoints.len() - self.config.max_checkpoints;
        for entry in checkpoints.iter().take(to_delete) {
            let path = entry.path();
            std::fs::remove_file(&path)?;

            // Also remove metadata file
            let metadata_path = path.with_extension("json");
            if metadata_path.exists() {
                std::fs::remove_file(metadata_path)?;
            }
        }

        Ok(())
    }

    /// Main training loop
    ///
    /// # Arguments
    ///
    /// * `task_loss_fn` - Function that computes task loss for each batch
    /// * `logic_loss_fn` - Function that computes logic loss for each batch
    /// * `val_data` - Optional validation dataset
    /// * `val_task_loss_fn` - Validation task loss function
    /// * `val_logic_loss_fn` - Validation logic loss function
    pub fn run<F, G, VF, VG>(
        &mut self,
        mut task_loss_fn: F,
        mut logic_loss_fn: G,
        val_data: Option<&[Experience<Tensor, Tensor>]>,
        val_task_loss_fn: Option<VF>,
        val_logic_loss_fn: Option<VG>,
    ) -> Result<StepMetrics>
    where
        F: FnMut(&ReplayBatch<Tensor, Tensor>, &VarMap, &Device) -> Result<Tensor>,
        G: FnMut(&ReplayBatch<Tensor, Tensor>, &VarMap, &Device) -> Result<Tensor>,
        VF: Fn(&[Experience<Tensor, Tensor>], &VarMap, &Device) -> Result<Tensor>,
        VG: Fn(&[Experience<Tensor, Tensor>], &VarMap, &Device) -> Result<Tensor>,
    {
        // Trigger on_train_start callbacks
        for callback in &mut self.callbacks {
            callback.on_train_start()?;
        }

        let mut last_metrics = StepMetrics::new(0);

        for step in self.state.step..self.config.num_steps {
            // Training step
            let metrics = self.train_step(&mut task_loss_fn, &mut logic_loss_fn)?;
            last_metrics = metrics.clone();

            // Log metrics
            if step % self.config.log_every == 0 {
                for callback in &mut self.callbacks {
                    callback.on_step_end(&metrics)?;
                }
            }

            // Validation
            if step % self.config.validate_every == 0
                && let (Some(val_data), Some(vtf), Some(vlf)) =
                    (val_data, &val_task_loss_fn, &val_logic_loss_fn)
            {
                let val_metrics = self.validate(val_data, vtf, vlf)?;

                // Update state with validation metrics
                self.state.metadata.validation_metrics = Some(val_metrics.clone());

                // Check for improvement
                if val_metrics.val_total_loss < self.state.best_val_loss {
                    self.state.best_val_loss = val_metrics.val_total_loss;
                    self.state.patience_counter = 0;
                } else {
                    self.state.patience_counter += 1;
                }

                // Trigger callbacks
                for callback in &mut self.callbacks {
                    callback.on_validation_end(&val_metrics)?;
                }

                // Early stopping check
                if let Some(patience) = self.config.early_stopping_patience
                    && self.state.patience_counter >= patience
                {
                    for callback in &mut self.callbacks {
                        callback.on_early_stopping(step)?;
                    }
                    break;
                }
            }

            // Checkpointing
            if step > 0 && step % self.config.checkpoint_every == 0 {
                let checkpoint_path = self.save_checkpoint(&metrics)?;
                for callback in &mut self.callbacks {
                    callback.on_checkpoint_saved(&checkpoint_path, &self.state.metadata)?;
                }
            }
        }

        // Trigger on_train_end callbacks
        for callback in &mut self.callbacks {
            callback.on_train_end(&last_metrics)?;
        }

        Ok(last_metrics)
    }

    /// Get current training step
    pub fn current_step(&self) -> usize {
        self.state.step
    }

    /// Get training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::create_experience;

    fn dummy_task_loss(
        batch: &ReplayBatch<Tensor, Tensor>,
        _var_map: &VarMap,
        device: &Device,
    ) -> Result<Tensor> {
        // Return a tensor with shape matching the batch size
        let batch_size = batch.experiences.len();
        let losses = vec![0.5f32; batch_size];
        let loss_tensor = Tensor::from_vec(losses, (batch_size,), device)?;
        Ok(loss_tensor.mean_all()?)
    }

    fn dummy_logic_loss(
        batch: &ReplayBatch<Tensor, Tensor>,
        _var_map: &VarMap,
        device: &Device,
    ) -> Result<Tensor> {
        // Return a tensor with shape matching the batch size
        let batch_size = batch.experiences.len();
        let losses = vec![0.3f32; batch_size];
        let loss_tensor = Tensor::from_vec(losses, (batch_size,), device)?;
        Ok(loss_tensor.mean_all()?)
    }

    fn dummy_val_task_loss(
        experiences: &[Experience<Tensor, Tensor>],
        _var_map: &VarMap,
        device: &Device,
    ) -> Result<Tensor> {
        // Return a tensor with shape matching the validation set size
        let batch_size = experiences.len();
        let losses = vec![0.4f32; batch_size];
        let loss_tensor = Tensor::from_vec(losses, (batch_size,), device)?;
        Ok(loss_tensor.mean_all()?)
    }

    fn dummy_val_logic_loss(
        experiences: &[Experience<Tensor, Tensor>],
        _var_map: &VarMap,
        device: &Device,
    ) -> Result<Tensor> {
        // Return a tensor with shape matching the validation set size
        let batch_size = experiences.len();
        let losses = vec![0.2f32; batch_size];
        let loss_tensor = Tensor::from_vec(losses, (batch_size,), device)?;
        Ok(loss_tensor.mean_all()?)
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.num_steps, 100_000);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.logic_weight, 0.5);
    }

    #[test]
    fn test_training_config_minimal() {
        let config = TrainingConfig::minimal();
        assert_eq!(config.num_steps, 1_000);
        assert_eq!(config.batch_size, 8);
    }

    #[test]
    fn test_step_metrics_format() {
        let mut metrics = StepMetrics::new(100);
        metrics.task_loss = 0.5;
        metrics.logic_loss = 0.3;
        metrics.total_loss = 0.8;
        metrics.learning_rate = 1e-4;

        let formatted = metrics.format();
        assert!(formatted.contains("Step 100"));
        assert!(formatted.contains("0.5"));
        assert!(formatted.contains("0.3"));
    }

    #[test]
    fn test_training_loop_creation() -> Result<()> {
        let config = TrainingConfig::minimal();
        let opt_config = OptimizerConfig::adamw().learning_rate(1e-4).build();
        let sched_config = LRSchedulerConfig::constant();
        let replay_config = ReplayBufferConfig::default();

        let _training_loop = TrainingLoop::new(config, opt_config, sched_config, replay_config)?;
        Ok(())
    }

    #[test]
    fn test_training_step() -> Result<()> {
        let device = Device::Cpu;
        let config = TrainingConfig::minimal().device(device.clone());
        let opt_config = OptimizerConfig::adamw().learning_rate(1e-4).build();
        let sched_config = LRSchedulerConfig::constant();
        let replay_config = ReplayBufferConfig::default();

        let mut training_loop = TrainingLoop::new(config, opt_config, sched_config, replay_config)?;

        // Add some dummy experiences
        for _ in 0..100 {
            let state = Tensor::randn(0f32, 1.0, (10,), &device)?;
            let action = Tensor::randn(0f32, 1.0, (5,), &device)?;
            let next_state = Tensor::randn(0f32, 1.0, (10,), &device)?;
            let experience = create_experience(state, action, 1.0, next_state, false);
            training_loop.add_experience(experience);
        }

        // Run a single training step
        let metrics = training_loop.train_step(dummy_task_loss, dummy_logic_loss)?;

        assert_eq!(metrics.step, 0);
        assert!(metrics.total_loss > 0.0);
        assert_eq!(training_loop.current_step(), 1);

        Ok(())
    }

    #[test]
    fn test_validation() -> Result<()> {
        let device = Device::Cpu;
        let config = TrainingConfig::minimal().device(device.clone());
        let opt_config = OptimizerConfig::adamw().learning_rate(1e-4).build();
        let sched_config = LRSchedulerConfig::constant();
        let replay_config = ReplayBufferConfig::default();

        let training_loop = TrainingLoop::new(config, opt_config, sched_config, replay_config)?;

        // Create validation data
        let mut val_data = Vec::new();
        for _ in 0..10 {
            let state = Tensor::randn(0f32, 1.0, (10,), &device)?;
            let action = Tensor::randn(0f32, 1.0, (5,), &device)?;
            let next_state = Tensor::randn(0f32, 1.0, (10,), &device)?;
            let experience = create_experience(state, action, 1.0, next_state, false);
            val_data.push(experience);
        }

        let val_metrics =
            training_loop.validate(&val_data, dummy_val_task_loss, dummy_val_logic_loss)?;

        assert_eq!(val_metrics.num_samples, 10);
        assert!(val_metrics.val_total_loss > 0.0);

        Ok(())
    }

    #[test]
    fn test_checkpoint_metadata() -> Result<()> {
        let metrics = StepMetrics::new(100);
        let metadata = CheckpointMetadata {
            step: 100,
            timestamp: chrono::Utc::now().to_rfc3339(),
            metrics,
            validation_metrics: None,
            model_version: "0.1.0".to_string(),
        };

        let json = serde_json::to_string(&metadata)?;
        let deserialized: CheckpointMetadata = serde_json::from_str(&json)?;

        assert_eq!(deserialized.step, 100);
        assert_eq!(deserialized.model_version, "0.1.0");

        Ok(())
    }
}
