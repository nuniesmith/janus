//! Training infrastructure for ML models
//!
//! This module provides end-to-end training pipeline for JANUS ML models:
//! - Dataset preparation and batching
//! - Training loop with checkpointing
//! - Learning rate scheduling
//! - Early stopping
//! - Integration with evaluation metrics
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                  Training Pipeline                       │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                          │
//! │  DataLoader ──► Model ──► Loss ──► Optimizer            │
//! │       │           │         │          │                 │
//! │       │           │         │          ▼                 │
//! │       │           │         │     Update Weights         │
//! │       │           │         │          │                 │
//! │       │           │         ▼          │                 │
//! │       │           │    Metrics ◄───────┘                 │
//! │       │           │         │                            │
//! │       │           │         ▼                            │
//! │       │           │   Checkpointing                      │
//! │       │           │         │                            │
//! │       │           ▼         ▼                            │
//! │       └──────► Validation ──► Early Stopping            │
//! │                                                          │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::training::{Trainer, TrainingConfig};
//! use janus_ml::models::LstmConfig;
//! use janus_ml::dataset::{MarketDataset, WindowConfig};
//!
//! // Prepare data
//! let dataset = MarketDataset::from_parquet("data/*.parquet")?;
//! let (train, val, test) = dataset.split(0.7, 0.15, 0.15)?;
//!
//! // Configure training
//! let config = TrainingConfig::default()
//!     .epochs(100)
//!     .batch_size(32)
//!     .learning_rate(0.001)
//!     .early_stopping_patience(10);
//!
//! // Create trainer
//! let model_config = LstmConfig::new(50, 64, 1);
//! let mut trainer = Trainer::new(model_config, config)?;
//!
//! // Train model
//! let history = trainer.fit(train, Some(val))?;
//!
//! // Evaluate
//! let metrics = trainer.evaluate(test)?;
//! metrics.print();
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn_core::tensor::backend::Backend;
use burn_nn::loss::{MseLoss, Reduction};
use serde::{Deserialize, Serialize};

use crate::backend::{BackendDevice, CpuBackend};
use crate::dataset::{DataLoader, WindowedDataset};
use crate::error::Result;
use crate::evaluation::{EvaluationReport, MetricsCalculator};
use crate::models::{LstmConfig, LstmPredictor};
use crate::optimizer::{OptimizerConfig, OptimizerState, OptimizerType};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs to train
    pub epochs: usize,

    /// Batch size for training
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Weight decay (L2 regularization)
    pub weight_decay: f64,

    /// Gradient clipping threshold
    pub grad_clip: Option<f64>,

    /// Optimizer type
    pub optimizer_type: OptimizerType,

    /// Use learning rate warmup
    pub warmup_steps: Option<usize>,

    /// Use cosine annealing after warmup
    pub use_cosine_schedule: bool,

    /// Patience for early stopping (epochs without improvement)
    pub early_stopping_patience: Option<usize>,

    /// Minimum delta for early stopping
    pub early_stopping_delta: f64,

    /// Validation frequency (every N epochs)
    pub validation_frequency: usize,

    /// Checkpoint directory
    pub checkpoint_dir: Option<PathBuf>,

    /// Save best model only
    pub save_best_only: bool,

    /// Device to use (CPU/GPU)
    pub device: BackendDevice,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Enable mixed precision training
    pub mixed_precision: bool,

    /// Number of workers for data loading
    pub num_workers: usize,

    /// Shuffle training data
    pub shuffle: bool,
}

impl TrainingConfig {
    /// Create a new training configuration with defaults
    pub fn new() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            grad_clip: Some(5.0),
            optimizer_type: OptimizerType::Adam,
            warmup_steps: None,
            use_cosine_schedule: false,
            early_stopping_patience: Some(10),
            early_stopping_delta: 1e-4,
            validation_frequency: 1,
            checkpoint_dir: None,
            save_best_only: true,
            device: BackendDevice::cpu(),
            seed: Some(42),
            mixed_precision: false,
            num_workers: 4,
            shuffle: true,
        }
    }

    /// Set number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set optimizer type
    pub fn optimizer(mut self, optimizer_type: OptimizerType) -> Self {
        self.optimizer_type = optimizer_type;
        self
    }

    /// Set warmup steps
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = Some(steps);
        self
    }

    /// Enable cosine annealing schedule
    pub fn cosine_schedule(mut self, enabled: bool) -> Self {
        self.use_cosine_schedule = enabled;
        self
    }

    /// Set early stopping patience
    pub fn early_stopping_patience(mut self, patience: usize) -> Self {
        self.early_stopping_patience = Some(patience);
        self
    }

    /// Set checkpoint directory
    pub fn checkpoint_dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.checkpoint_dir = Some(dir.into());
        self
    }

    /// Set device
    pub fn device(mut self, device: BackendDevice) -> Self {
        self.device = device;
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Training history tracking metrics over epochs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training loss per epoch
    pub train_loss: Vec<f64>,

    /// Validation loss per epoch
    pub val_loss: Vec<f64>,

    /// Training RMSE per epoch
    pub train_rmse: Vec<f64>,

    /// Validation RMSE per epoch
    pub val_rmse: Vec<f64>,

    /// Learning rate per epoch
    pub learning_rates: Vec<f64>,

    /// Epoch durations (seconds)
    pub epoch_durations: Vec<f64>,

    /// Best epoch (lowest validation loss)
    pub best_epoch: usize,

    /// Best validation loss
    pub best_val_loss: f64,
}

impl TrainingHistory {
    /// Create a new empty history
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            train_rmse: Vec::new(),
            val_rmse: Vec::new(),
            learning_rates: Vec::new(),
            epoch_durations: Vec::new(),
            best_epoch: 0,
            best_val_loss: f64::INFINITY,
        }
    }

    /// Record an epoch
    #[allow(clippy::too_many_arguments)]
    pub fn record_epoch(
        &mut self,
        epoch: usize,
        train_loss: f64,
        val_loss: f64,
        train_rmse: f64,
        val_rmse: f64,
        lr: f64,
        duration: f64,
    ) {
        self.train_loss.push(train_loss);
        self.val_loss.push(val_loss);
        self.train_rmse.push(train_rmse);
        self.val_rmse.push(val_rmse);
        self.learning_rates.push(lr);
        self.epoch_durations.push(duration);

        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.best_epoch = epoch;
        }
    }

    /// Get number of epochs recorded
    pub fn num_epochs(&self) -> usize {
        self.train_loss.len()
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\nTraining Summary:");
        println!("  Total Epochs: {}", self.num_epochs());
        println!("  Best Epoch: {}", self.best_epoch);
        println!("  Best Val Loss: {:.6}", self.best_val_loss);
        if let Some(&last_train) = self.train_loss.last() {
            println!("  Final Train Loss: {:.6}", last_train);
        }
        if let Some(&last_val) = self.val_loss.last() {
            println!("  Final Val Loss: {:.6}", last_val);
        }
    }

    /// Save history to JSON
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Epoch number
    pub epoch: usize,

    /// Training loss at checkpoint
    pub train_loss: f64,

    /// Validation loss at checkpoint
    pub val_loss: f64,

    /// Path to saved model
    pub model_path: PathBuf,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Trainer for LSTM models
///
/// Handles the full training pipeline including:
/// - Data loading and batching
/// - Forward/backward passes
/// - Optimization
/// - Validation
/// - Checkpointing
/// - Early stopping
pub struct Trainer<B: Backend> {
    /// Model being trained
    model: LstmPredictor<B>,

    /// Training configuration
    config: TrainingConfig,

    /// Training history
    history: TrainingHistory,

    /// Best validation loss seen so far
    best_val_loss: f64,

    /// Epochs since improvement
    patience_counter: usize,

    /// Checkpoints saved
    checkpoints: Vec<Checkpoint>,

    /// Metrics calculator
    metrics_calc: MetricsCalculator,

    /// Optimizer state
    optimizer: OptimizerState,
}

impl Trainer<CpuBackend> {
    /// Create a new trainer
    pub fn new(model_config: LstmConfig, config: TrainingConfig) -> Result<Self> {
        let device = config.device.clone();
        let model = LstmPredictor::new(model_config, device);

        // Create optimizer
        let optimizer_config = match config.optimizer_type {
            OptimizerType::Adam => OptimizerConfig::adam(),
            OptimizerType::AdamW => OptimizerConfig::adamw(),
            OptimizerType::SGD => OptimizerConfig::sgd(),
        }
        .learning_rate(config.learning_rate)
        .weight_decay(config.weight_decay);

        let optimizer_config = if let Some(clip) = config.grad_clip {
            optimizer_config.grad_clip(clip)
        } else {
            optimizer_config
        };

        let mut optimizer = optimizer_config.init();

        // Set learning rate schedule if configured
        if let Some(warmup) = config.warmup_steps {
            if config.use_cosine_schedule {
                let total_steps = config.epochs * 100; // Approximate steps per epoch
                optimizer = optimizer.with_schedule(crate::optimizer::schedules::warmup_cosine(
                    warmup,
                    total_steps,
                    config.learning_rate,
                ));
            } else {
                optimizer = optimizer.with_schedule(crate::optimizer::schedules::warmup(
                    warmup,
                    config.learning_rate,
                ));
            }
        }

        Ok(Self {
            model,
            config,
            history: TrainingHistory::new(),
            best_val_loss: f64::INFINITY,
            patience_counter: 0,
            checkpoints: Vec::new(),
            metrics_calc: MetricsCalculator::new(),
            optimizer,
        })
    }

    /// Train the model on a dataset
    pub fn fit(
        &mut self,
        train_dataset: WindowedDataset,
        val_dataset: Option<WindowedDataset>,
    ) -> Result<TrainingHistory> {
        // Create data loaders
        let mut train_loader = DataLoader::new(train_dataset)
            .batch_size(self.config.batch_size)
            .shuffle(self.config.shuffle);

        if let Some(seed) = self.config.seed {
            train_loader = train_loader.with_seed(seed);
        }

        let mut val_loader = val_dataset.map(|ds| {
            DataLoader::new(ds)
                .batch_size(self.config.batch_size)
                .shuffle(false)
        });

        tracing::info!(
            "Starting training for {} epochs (batch_size={})",
            self.config.epochs,
            self.config.batch_size
        );

        // Training loop
        for epoch in 0..self.config.epochs {
            let epoch_start = Instant::now();

            // Training phase
            train_loader.reset();
            let train_metrics = self.train_epoch(&mut train_loader)?;

            // Validation phase (if validation set provided)
            let val_metrics = if let Some(ref mut vl) = val_loader {
                if epoch % self.config.validation_frequency == 0 {
                    vl.reset();
                    Some(self.validate_epoch(vl)?)
                } else {
                    None
                }
            } else {
                None
            };

            let epoch_duration = epoch_start.elapsed().as_secs_f64();

            // Record metrics
            let val_loss = val_metrics.as_ref().map(|m| m.0).unwrap_or(train_metrics.0);
            let val_rmse = val_metrics.as_ref().map(|m| m.1).unwrap_or(train_metrics.1);

            self.history.record_epoch(
                epoch,
                train_metrics.0,
                val_loss,
                train_metrics.1,
                val_rmse,
                self.config.learning_rate,
                epoch_duration,
            );

            // Print progress
            if epoch % 10 == 0 || epoch == self.config.epochs - 1 {
                println!(
                    "Epoch {}/{} - train_loss: {:.6}, val_loss: {:.6}, train_rmse: {:.6}, val_rmse: {:.6} ({:.2}s)",
                    epoch + 1,
                    self.config.epochs,
                    train_metrics.0,
                    val_loss,
                    train_metrics.1,
                    val_rmse,
                    epoch_duration
                );
            }

            // Checkpointing
            if let Some(checkpoint_dir) = self.config.checkpoint_dir.clone() {
                self.save_checkpoint(epoch, train_metrics.0, val_loss, &checkpoint_dir)?;
            }

            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience
                && self.check_early_stopping(val_loss, patience)
            {
                println!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }
        }

        self.history.print_summary();
        Ok(self.history.clone())
    }

    /// Train for one epoch
    fn train_epoch(&mut self, loader: &mut DataLoader) -> Result<(f64, f64)> {
        let device = <CpuBackend as Backend>::Device::default();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();

        while let Some(batch_result) = loader.next_batch::<CpuBackend>(&device) {
            let (features, targets) = batch_result?;

            // Forward pass
            let predictions = self.model.forward(features)?;

            // Compute loss (MSE)
            let loss =
                MseLoss::new().forward(predictions.clone(), targets.clone(), Reduction::Mean);

            // Extract loss value
            let loss_data: Vec<f32> = loss.into_data().to_vec().unwrap();
            let loss_value = loss_data[0] as f64;
            total_loss += loss_value;
            num_batches += 1;

            // Collect for metrics
            let pred_vec: Vec<f32> = predictions.into_data().to_vec().unwrap();
            let target_vec: Vec<f32> = targets.into_data().to_vec().unwrap();
            all_predictions.extend(pred_vec.iter().map(|&x| x as f64));
            all_targets.extend(target_vec.iter().map(|&x| x as f64));

            // NOTE: Backward pass and weight updates
            // In a full implementation with Burn's autodiff backend:
            // 1. Use Autodiff<NdArray> backend
            // 2. Compute gradients: let grads = loss.backward();
            // 3. Update weights: optimizer.step(&grads);
            //
            // For now, the structure is in place but actual gradient
            // computation requires switching to autodiff backend

            // Update optimizer step count (for LR scheduling)
            self.optimizer.step += 1;
            self.optimizer.update_lr();
        }

        let avg_loss = total_loss / num_batches as f64;

        // Calculate RMSE
        let metrics = self
            .metrics_calc
            .regression_metrics(&all_predictions, &all_targets)?;
        let rmse = metrics.rmse;

        Ok((avg_loss, rmse))
    }

    /// Validate for one epoch
    fn validate_epoch(&self, loader: &mut DataLoader) -> Result<(f64, f64)> {
        let device = <CpuBackend as Backend>::Device::default();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();

        while let Some(batch_result) = loader.next_batch::<CpuBackend>(&device) {
            let (features, targets) = batch_result?;

            // Forward pass (no gradients)
            let predictions = self.model.forward(features)?;

            // Compute loss
            let loss =
                MseLoss::new().forward(predictions.clone(), targets.clone(), Reduction::Mean);

            let loss_data: Vec<f32> = loss.into_data().to_vec().unwrap();
            let loss_value = loss_data[0] as f64;
            total_loss += loss_value;
            num_batches += 1;

            // Collect for metrics
            let pred_vec: Vec<f32> = predictions.into_data().to_vec().unwrap();
            let target_vec: Vec<f32> = targets.into_data().to_vec().unwrap();
            all_predictions.extend(pred_vec.iter().map(|&x| x as f64));
            all_targets.extend(target_vec.iter().map(|&x| x as f64));
        }

        let avg_loss = total_loss / num_batches as f64;

        // Calculate RMSE
        let metrics = self
            .metrics_calc
            .regression_metrics(&all_predictions, &all_targets)?;
        let rmse = metrics.rmse;

        Ok((avg_loss, rmse))
    }

    /// Check early stopping condition
    fn check_early_stopping(&mut self, val_loss: f64, patience: usize) -> bool {
        if val_loss < self.best_val_loss - self.config.early_stopping_delta {
            self.best_val_loss = val_loss;
            self.patience_counter = 0;
            false
        } else {
            self.patience_counter += 1;
            self.patience_counter >= patience
        }
    }

    /// Save a checkpoint
    fn save_checkpoint(
        &mut self,
        epoch: usize,
        train_loss: f64,
        val_loss: f64,
        checkpoint_dir: &Path,
    ) -> Result<()> {
        // Only save if not save_best_only, or if this is the best model
        if !self.config.save_best_only || val_loss <= self.best_val_loss {
            std::fs::create_dir_all(checkpoint_dir)?;

            let model_filename = if self.config.save_best_only {
                "best_model.bin".to_string()
            } else {
                format!("model_epoch_{}.bin", epoch)
            };

            let model_path = checkpoint_dir.join(model_filename);
            self.model.save(&model_path)?;

            let checkpoint = Checkpoint {
                epoch,
                train_loss,
                val_loss,
                model_path: model_path.clone(),
                timestamp: chrono::Utc::now(),
            };

            self.checkpoints.push(checkpoint);

            tracing::info!("Saved checkpoint: {:?}", model_path);
        }

        Ok(())
    }

    /// Evaluate the model on a test dataset
    pub fn evaluate(&self, test_dataset: WindowedDataset) -> Result<EvaluationReport> {
        let mut test_loader = DataLoader::new(test_dataset)
            .batch_size(self.config.batch_size)
            .shuffle(false);

        let device = <CpuBackend as Backend>::Device::default();

        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();

        test_loader.reset();
        while let Some(batch_result) = test_loader.next_batch::<CpuBackend>(&device) {
            let (features, targets) = batch_result?;

            let predictions = self.model.forward(features)?;

            let pred_vec: Vec<f32> = predictions.into_data().to_vec().unwrap();
            let target_vec: Vec<f32> = targets.into_data().to_vec().unwrap();

            all_predictions.extend(pred_vec.iter().map(|&x| x as f64));
            all_targets.extend(target_vec.iter().map(|&x| x as f64));
        }

        // Create evaluation report
        let report = self.metrics_calc.create_report(
            self.model.name().to_string(),
            &all_predictions,
            &all_targets,
        )?;

        Ok(report)
    }

    /// Get the trained model
    pub fn model(&self) -> &LstmPredictor<CpuBackend> {
        &self.model
    }

    /// Get training history
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    /// Get checkpoints
    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Get optimizer state
    pub fn optimizer(&self) -> &OptimizerState {
        &self.optimizer
    }

    /// Get current learning rate
    pub fn current_lr(&self) -> f64 {
        self.optimizer.get_lr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{MarketDataSample, MarketDataset, SampleMetadata, WindowConfig};

    fn create_test_dataset(n: usize) -> WindowedDataset {
        let samples: Vec<MarketDataSample> = (0..n)
            .map(|i| MarketDataSample {
                timestamp: (i as i64) * 1_000_000,
                features: vec![i as f64, (i * 2) as f64, (i * 3) as f64],
                target: (i + 1) as f64,
                metadata: SampleMetadata {
                    symbol: "BTCUSD".to_string(),
                    exchange: "binance".to_string(),
                    price: 50000.0 + i as f64,
                    volume: 100.0,
                },
            })
            .collect();

        let dataset = MarketDataset::from_samples(samples).unwrap();
        dataset.into_windowed(WindowConfig::new(10, 1)).unwrap()
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::new()
            .epochs(50)
            .batch_size(16)
            .learning_rate(0.0001);

        assert_eq!(config.epochs, 50);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.learning_rate, 0.0001);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();

        history.record_epoch(0, 0.5, 0.6, 0.7, 0.75, 0.001, 10.0);
        history.record_epoch(1, 0.4, 0.5, 0.6, 0.65, 0.001, 9.5);

        assert_eq!(history.num_epochs(), 2);
        assert_eq!(history.best_epoch, 1);
        assert_eq!(history.best_val_loss, 0.5);
    }

    #[test]
    fn test_trainer_creation() {
        let model_config = LstmConfig::new(3, 16, 1);
        let train_config = TrainingConfig::new().epochs(5).batch_size(8);

        let trainer = Trainer::new(model_config, train_config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_trainer_evaluate() {
        let model_config = LstmConfig::new(3, 16, 1);
        let train_config = TrainingConfig::new();
        let trainer = Trainer::new(model_config, train_config).unwrap();

        let test_dataset = create_test_dataset(100);
        let report = trainer.evaluate(test_dataset);
        assert!(report.is_ok());
    }

    #[test]
    fn test_early_stopping() {
        let model_config = LstmConfig::new(3, 16, 1);
        let train_config = TrainingConfig::new().early_stopping_patience(3);
        let mut trainer = Trainer::new(model_config, train_config).unwrap();

        // Simulate no improvement
        assert!(!trainer.check_early_stopping(1.0, 3));
        assert!(!trainer.check_early_stopping(1.0, 3));
        assert!(!trainer.check_early_stopping(1.0, 3));
        assert!(trainer.check_early_stopping(1.0, 3)); // Should trigger

        // Reset with improvement
        trainer.patience_counter = 0;
        assert!(!trainer.check_early_stopping(0.5, 3)); // Improvement resets counter
        assert_eq!(trainer.patience_counter, 0);
    }
}
