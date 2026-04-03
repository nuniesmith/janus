//! Training Module for ViViT
//!
//! Provides loss functions, optimizers, and training loop for ViViT model.
//! Uses manual backpropagation with ndarray for educational purposes.
//!
//! # Features
//!
//! - Cross-entropy loss for classification
//! - MSE loss for regression
//! - SGD and Adam optimizers
//! - Training loop with validation
//! - Learning rate scheduling
//! - Gradient clipping

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use tracing::info;

use common::{JanusError, Result};

/// Loss function type
#[derive(Debug, Clone, Copy)]
pub enum LossType {
    /// Cross-entropy loss for classification
    CrossEntropy,
    /// Mean squared error for regression
    MSE,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum (for SGD) or beta1 (for Adam)
    pub momentum: f32,
    /// Beta2 for Adam
    pub beta2: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Gradient clipping threshold
    pub grad_clip: Option<f32>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0001,
            grad_clip: Some(1.0),
        }
    }
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Loss function
    pub loss_type: LossType,
    /// Optimizer configuration
    pub optimizer: OptimizerConfig,
    /// Validation split (0.0 to 1.0)
    pub validation_split: f32,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: Option<usize>,
    /// Learning rate schedule
    pub lr_schedule: LRSchedule,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 8,
            loss_type: LossType::CrossEntropy,
            optimizer: OptimizerConfig::default(),
            validation_split: 0.2,
            early_stopping_patience: Some(5),
            lr_schedule: LRSchedule::Constant,
        }
    }
}

/// Learning rate schedule
#[derive(Debug, Clone)]
pub enum LRSchedule {
    /// Keep learning rate constant
    Constant,
    /// Step decay: multiply by gamma every step_size epochs
    StepDecay { step_size: usize, gamma: f32 },
    /// Exponential decay: lr = lr0 * gamma^epoch
    ExponentialDecay { gamma: f32 },
    /// Cosine annealing
    CosineAnnealing { t_max: usize, eta_min: f32 },
}

/// Training metrics for one epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub val_loss: Option<f32>,
    pub val_accuracy: Option<f32>,
    pub learning_rate: f32,
}

/// Loss computation
pub struct Loss;

impl Loss {
    /// Compute cross-entropy loss
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions [batch_size, num_classes]
    /// * `targets` - Target class indices [batch_size]
    ///
    /// # Returns
    ///
    /// Loss value and gradients
    pub fn cross_entropy(
        predictions: &Array2<f32>,
        targets: &Array1<usize>,
    ) -> Result<(f32, Array2<f32>)> {
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];

        // Compute softmax
        let softmax = Self::softmax(predictions);

        // Compute loss
        let mut loss = 0.0;
        for (i, &target) in targets.iter().enumerate() {
            if target >= num_classes {
                return Err(JanusError::Internal(format!(
                    "Target class {} out of bounds for {} classes",
                    target, num_classes
                )));
            }
            loss -= softmax[[i, target]].ln();
        }
        loss /= batch_size as f32;

        // Compute gradients
        let mut gradients = softmax.clone();
        for (i, &target) in targets.iter().enumerate() {
            gradients[[i, target]] -= 1.0;
        }
        gradients /= batch_size as f32;

        Ok((loss, gradients))
    }

    /// Compute mean squared error loss
    pub fn mse(predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<(f32, Array2<f32>)> {
        if predictions.shape() != targets.shape() {
            return Err(JanusError::Internal(
                "Predictions and targets must have same shape".to_string(),
            ));
        }

        let diff = predictions - targets;
        let loss = (&diff * &diff).mean().unwrap();

        let gradients = &diff * 2.0 / (predictions.len() as f32);

        Ok((loss, gradients))
    }

    /// Softmax activation
    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let mut result = x.clone();

        for mut row in result.axis_iter_mut(Axis(0)) {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row.mapv_inplace(|v| (v - max).exp());
            let sum: f32 = row.sum();
            row.mapv_inplace(|v| v / sum);
        }

        result
    }

    /// Compute accuracy for classification
    pub fn accuracy(predictions: &Array2<f32>, targets: &Array1<usize>) -> f32 {
        let batch_size = predictions.shape()[0];
        let mut correct = 0;

        for (i, &target) in targets.iter().enumerate() {
            let pred_class = predictions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if pred_class == target {
                correct += 1;
            }
        }

        correct as f32 / batch_size as f32
    }
}

/// SGD Optimizer with momentum
pub struct SGDOptimizer {
    config: OptimizerConfig,
    velocity: HashMap<String, Array2<f32>>,
}

impl SGDOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            velocity: HashMap::new(),
        }
    }

    /// Update parameters with gradients
    pub fn step(&mut self, param_name: &str, param: &mut Array2<f32>, grad: &Array2<f32>) {
        let mut grad = grad.clone();

        // Apply gradient clipping
        if let Some(clip) = self.config.grad_clip {
            Self::clip_gradient(&mut grad, clip);
        }

        // Apply weight decay
        if self.config.weight_decay > 0.0 {
            grad = &grad + &(param.mapv(|x| x * self.config.weight_decay));
        }

        // Get or initialize velocity
        let velocity = self
            .velocity
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(param.raw_dim()));

        // Update velocity with momentum
        *velocity = velocity.mapv(|v| v * self.config.momentum) - &grad * self.config.learning_rate;

        // Update parameters
        *param = &*param + &*velocity;
    }

    fn clip_gradient(grad: &mut Array2<f32>, max_norm: f32) {
        let norm = grad.mapv(|x| x * x).sum().sqrt();
        if norm > max_norm {
            grad.mapv_inplace(|x| x * max_norm / norm);
        }
    }
}

/// Adam Optimizer
pub struct AdamOptimizer {
    config: OptimizerConfig,
    m: HashMap<String, Array2<f32>>, // First moment
    v: HashMap<String, Array2<f32>>, // Second moment
    t: usize,                        // Timestep
}

impl AdamOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Update parameters with gradients
    pub fn step(&mut self, param_name: &str, param: &mut Array2<f32>, grad: &Array2<f32>) {
        self.t += 1;

        let mut grad = grad.clone();

        // Apply gradient clipping
        if let Some(clip) = self.config.grad_clip {
            Self::clip_gradient(&mut grad, clip);
        }

        // Apply weight decay
        if self.config.weight_decay > 0.0 {
            grad = &grad + &(param.mapv(|x| x * self.config.weight_decay));
        }

        // Get or initialize moments
        let m = self
            .m
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(param.raw_dim()));
        let v = self
            .v
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(param.raw_dim()));

        // Update biased first moment estimate
        *m = m.mapv(|val| val * self.config.momentum) + &grad * (1.0 - self.config.momentum);

        // Update biased second moment estimate
        *v = v.mapv(|val| val * self.config.beta2)
            + grad.mapv(|g| g * g) * (1.0 - self.config.beta2);

        // Bias correction
        let m_hat = m.mapv(|val| val / (1.0 - self.config.momentum.powi(self.t as i32)));
        let v_hat = v.mapv(|val| val / (1.0 - self.config.beta2.powi(self.t as i32)));

        // Update parameters
        *param = &*param
            - &m_hat.mapv(|m| m * self.config.learning_rate)
                / (v_hat.mapv(|v| v.sqrt() + self.config.epsilon));
    }

    fn clip_gradient(grad: &mut Array2<f32>, max_norm: f32) {
        let norm = grad.mapv(|x| x * x).sum().sqrt();
        if norm > max_norm {
            grad.mapv_inplace(|x| x * max_norm / norm);
        }
    }
}

/// Training loop
pub struct Trainer {
    config: TrainingConfig,
    history: Vec<EpochMetrics>,
    best_val_loss: f32,
    patience_counter: usize,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            best_val_loss: f32::INFINITY,
            patience_counter: 0,
        }
    }

    /// Get current learning rate based on schedule
    pub fn get_learning_rate(&self, epoch: usize, base_lr: f32) -> f32 {
        match &self.config.lr_schedule {
            LRSchedule::Constant => base_lr,
            LRSchedule::StepDecay { step_size, gamma } => {
                base_lr * gamma.powi((epoch / step_size) as i32)
            }
            LRSchedule::ExponentialDecay { gamma } => base_lr * gamma.powi(epoch as i32),
            LRSchedule::CosineAnnealing { t_max, eta_min } => {
                eta_min
                    + (base_lr - eta_min)
                        * 0.5
                        * (1.0 + ((epoch as f32 * std::f32::consts::PI) / *t_max as f32).cos())
            }
        }
    }

    /// Check if should stop early
    pub fn should_stop_early(&mut self, val_loss: f32) -> bool {
        if let Some(patience) = self.config.early_stopping_patience {
            if val_loss < self.best_val_loss {
                self.best_val_loss = val_loss;
                self.patience_counter = 0;
                false
            } else {
                self.patience_counter += 1;
                if self.patience_counter >= patience {
                    info!(
                        "Early stopping triggered after {} epochs without improvement",
                        patience
                    );
                    true
                } else {
                    false
                }
            }
        } else {
            false
        }
    }

    /// Record epoch metrics
    pub fn record_epoch(&mut self, metrics: EpochMetrics) {
        info!(
            "Epoch {}: train_loss={:.4}, train_acc={:.2}%, val_loss={:.4}, val_acc={:.2}%",
            metrics.epoch,
            metrics.train_loss,
            metrics.train_accuracy * 100.0,
            metrics.val_loss.unwrap_or(0.0),
            metrics.val_accuracy.unwrap_or(0.0) * 100.0
        );
        self.history.push(metrics);
    }

    /// Get training history
    pub fn history(&self) -> &[EpochMetrics] {
        &self.history
    }
}

/// Create batches from data
pub struct DataLoader {
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
    shuffle: bool,
}

impl DataLoader {
    pub fn new(data_size: usize, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..data_size).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }

        Self {
            indices,
            batch_size,
            current_idx: 0,
            shuffle,
        }
    }

    /// Get next batch of indices
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch = self.indices[self.current_idx..end_idx].to_vec();
        self.current_idx = end_idx;

        Some(batch)
    }

    /// Reset loader for new epoch
    pub fn reset(&mut self) {
        self.current_idx = 0;

        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Check if epoch is complete
    pub fn is_complete(&self) -> bool {
        self.current_idx >= self.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_loss() {
        let predictions =
            Array2::from_shape_vec((2, 3), vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1]).unwrap();

        let targets = Array1::from_vec(vec![0, 1]);

        let (loss, gradients) = Loss::cross_entropy(&predictions, &targets).unwrap();

        assert!(loss > 0.0);
        assert_eq!(gradients.shape(), predictions.shape());
    }

    #[test]
    fn test_mse_loss() {
        let predictions =
            Array2::from_shape_vec((2, 3), vec![0.5, 0.5, 0.5, 0.8, 0.8, 0.8]).unwrap();

        let targets = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        let (loss, gradients) = Loss::mse(&predictions, &targets).unwrap();

        assert!(loss > 0.0);
        assert_eq!(gradients.shape(), predictions.shape());
    }

    #[test]
    fn test_accuracy() {
        let predictions = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.8, 0.1, 0.1, // Correct: class 0
                0.1, 0.8, 0.1, // Correct: class 1
                0.1, 0.1, 0.8, // Correct: class 2
            ],
        )
        .unwrap();

        let targets = Array1::from_vec(vec![0, 1, 2]);

        let acc = Loss::accuracy(&predictions, &targets);
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_sgd_optimizer() {
        let config = OptimizerConfig::default();
        let mut optimizer = SGDOptimizer::new(config);

        let mut param = Array2::ones((2, 2));
        let grad = Array2::from_elem((2, 2), 0.1);

        optimizer.step("test_param", &mut param, &grad);

        // Parameters should have changed
        assert!(param[[0, 0]] < 1.0);
    }

    #[test]
    fn test_adam_optimizer() {
        let config = OptimizerConfig::default();
        let mut optimizer = AdamOptimizer::new(config);

        let mut param = Array2::ones((2, 2));
        let grad = Array2::from_elem((2, 2), 0.1);

        optimizer.step("test_param", &mut param, &grad);

        // Parameters should have changed
        assert!(param[[0, 0]] < 1.0);
    }

    #[test]
    fn test_lr_schedule_step_decay() {
        let config = TrainingConfig {
            lr_schedule: LRSchedule::StepDecay {
                step_size: 5,
                gamma: 0.1,
            },
            ..Default::default()
        };
        let trainer = Trainer::new(config);

        let lr0 = trainer.get_learning_rate(0, 0.1);
        let lr5 = trainer.get_learning_rate(5, 0.1);
        let lr10 = trainer.get_learning_rate(10, 0.1);

        assert_eq!(lr0, 0.1);
        assert!((lr5 - 0.01).abs() < 1e-6);
        assert!((lr10 - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_data_loader() {
        let mut loader = DataLoader::new(10, 3, false);

        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1.len(), 3);
        assert_eq!(batch1, vec![0, 1, 2]);

        let batch2 = loader.next_batch().unwrap();
        assert_eq!(batch2, vec![3, 4, 5]);

        loader.reset();
        let batch1_again = loader.next_batch().unwrap();
        assert_eq!(batch1_again, vec![0, 1, 2]);
    }

    #[test]
    fn test_early_stopping() {
        let config = TrainingConfig {
            early_stopping_patience: Some(3),
            ..Default::default()
        };
        let mut trainer = Trainer::new(config);

        assert!(!trainer.should_stop_early(1.0)); // First val_loss, becomes best
        assert!(!trainer.should_stop_early(1.1)); // Worse, counter=1
        assert!(!trainer.should_stop_early(1.2)); // Worse, counter=2
        assert!(trainer.should_stop_early(1.3)); // Worse, counter=3, should stop now
    }
}
