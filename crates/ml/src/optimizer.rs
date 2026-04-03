//! Optimizers for training ML models
//!
//! This module provides optimizers for gradient-based learning:
//! - Adam (Adaptive Moment Estimation)
//! - AdamW (Adam with decoupled weight decay)
//! - SGD (Stochastic Gradient Descent with momentum)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              Optimizer Module                        │
//! ├─────────────────────────────────────────────────────┤
//! │                                                      │
//! │  Model Parameters + Gradients                       │
//! │         │                                            │
//! │         ▼                                            │
//! │  Optimizer (Adam/AdamW/SGD)                         │
//! │         │                                            │
//! │         ├─► Compute adaptive learning rates         │
//! │         ├─► Apply momentum                          │
//! │         ├─► Apply weight decay                      │
//! │         │                                            │
//! │         ▼                                            │
//! │  Updated Parameters                                 │
//! │                                                      │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::optimizer::{AdamConfig, Optimizer};
//!
//! // Create optimizer
//! let config = AdamConfig::new()
//!     .learning_rate(0.001)
//!     .betas(0.9, 0.999)
//!     .epsilon(1e-8)
//!     .weight_decay(0.01);
//!
//! let mut optimizer = config.init();
//!
//! // Training step
//! optimizer.step(&grads, &mut params);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Learning rate schedule function type
pub type LRScheduleFn = Box<dyn Fn(usize) -> f64 + Send>;

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type
    pub optimizer_type: OptimizerType,

    /// Base learning rate
    pub learning_rate: f64,

    /// Weight decay (L2 regularization)
    pub weight_decay: f64,

    /// Gradient clipping threshold
    pub grad_clip: Option<f64>,

    /// Optimizer-specific parameters
    #[serde(skip)]
    pub params: HashMap<String, f64>,
}

/// Type of optimizer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// AdamW (Adam with decoupled weight decay)
    AdamW,
    /// SGD with momentum
    SGD,
}

impl OptimizerConfig {
    /// Create Adam optimizer config
    pub fn adam() -> Self {
        let mut params = HashMap::new();
        params.insert("beta1".to_string(), 0.9);
        params.insert("beta2".to_string(), 0.999);
        params.insert("epsilon".to_string(), 1e-8);

        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.001,
            weight_decay: 0.0,
            grad_clip: None,
            params,
        }
    }

    /// Create AdamW optimizer config
    pub fn adamw() -> Self {
        let mut params = HashMap::new();
        params.insert("beta1".to_string(), 0.9);
        params.insert("beta2".to_string(), 0.999);
        params.insert("epsilon".to_string(), 1e-8);

        Self {
            optimizer_type: OptimizerType::AdamW,
            learning_rate: 0.001,
            weight_decay: 0.01,
            grad_clip: None,
            params,
        }
    }

    /// Create SGD optimizer config
    pub fn sgd() -> Self {
        let mut params = HashMap::new();
        params.insert("momentum".to_string(), 0.9);
        params.insert("dampening".to_string(), 0.0);
        params.insert("nesterov".to_string(), 0.0);

        Self {
            optimizer_type: OptimizerType::SGD,
            learning_rate: 0.01,
            weight_decay: 0.0,
            grad_clip: None,
            params,
        }
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

    /// Set gradient clipping
    pub fn grad_clip(mut self, clip: f64) -> Self {
        self.grad_clip = Some(clip);
        self
    }

    /// Set Adam/AdamW beta parameters
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.params.insert("beta1".to_string(), beta1);
        self.params.insert("beta2".to_string(), beta2);
        self
    }

    /// Set Adam/AdamW epsilon
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.params.insert("epsilon".to_string(), eps);
        self
    }

    /// Set SGD momentum
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.params.insert("momentum".to_string(), momentum);
        self
    }

    /// Enable Nesterov momentum for SGD
    pub fn nesterov(mut self, enabled: bool) -> Self {
        self.params
            .insert("nesterov".to_string(), if enabled { 1.0 } else { 0.0 });
        self
    }

    /// Initialize the optimizer
    pub fn init(&self) -> OptimizerState {
        OptimizerState::new(self.clone())
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::adam()
    }
}

/// Optimizer state tracking
pub struct OptimizerState {
    /// Configuration
    pub config: OptimizerConfig,

    /// Current step number
    pub step: usize,

    /// Current learning rate (may be scheduled)
    pub current_lr: f64,

    /// First moment estimates (for Adam/AdamW)
    pub first_moments: HashMap<String, Vec<f64>>,

    /// Second moment estimates (for Adam/AdamW)
    pub second_moments: HashMap<String, Vec<f64>>,

    /// Velocity (for SGD with momentum)
    pub velocity: HashMap<String, Vec<f64>>,

    /// Learning rate schedule
    #[allow(clippy::type_complexity)]
    pub lr_schedule: Option<Box<dyn Fn(usize) -> f64 + Send>>,
}

impl OptimizerState {
    /// Create a new optimizer state
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            current_lr: config.learning_rate,
            config,
            step: 0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            velocity: HashMap::new(),
            lr_schedule: None,
        }
    }

    /// Set learning rate schedule
    pub fn with_schedule<F>(mut self, schedule: F) -> Self
    where
        F: Fn(usize) -> f64 + Send + 'static,
    {
        self.lr_schedule = Some(Box::new(schedule));
        self
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        if let Some(ref schedule) = self.lr_schedule {
            schedule(self.step)
        } else {
            self.current_lr
        }
    }

    /// Update learning rate based on schedule
    pub fn update_lr(&mut self) {
        if let Some(ref schedule) = self.lr_schedule {
            self.current_lr = schedule(self.step);
        }
    }

    /// Perform an optimization step (conceptual - actual implementation would work with tensors)
    ///
    /// This is a simplified version for demonstration. In practice, this would work with
    /// Burn's parameter groups and tensors.
    pub fn step_simple(&mut self, param_name: &str, gradient: &[f64], params: &mut [f64]) {
        assert_eq!(gradient.len(), params.len());

        self.step += 1;
        self.update_lr();
        let lr = self.get_lr();

        match self.config.optimizer_type {
            OptimizerType::Adam | OptimizerType::AdamW => {
                self.step_adam(param_name, gradient, params, lr);
            }
            OptimizerType::SGD => {
                self.step_sgd(param_name, gradient, params, lr);
            }
        }
    }

    /// Adam/AdamW step
    fn step_adam(&mut self, param_name: &str, gradient: &[f64], params: &mut [f64], lr: f64) {
        let beta1 = self.config.params.get("beta1").copied().unwrap_or(0.9);
        let beta2 = self.config.params.get("beta2").copied().unwrap_or(0.999);
        let epsilon = self.config.params.get("epsilon").copied().unwrap_or(1e-8);
        let wd = self.config.weight_decay;

        // Initialize moments if needed
        let m = self
            .first_moments
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);
        let v = self
            .second_moments
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);

        for i in 0..params.len() {
            let mut grad = gradient[i];

            // Gradient clipping
            if let Some(clip) = self.config.grad_clip {
                grad = grad.clamp(-clip, clip);
            }

            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad;

            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            let m_hat = m[i] / (1.0 - beta1.powi(self.step as i32));

            // Compute bias-corrected second raw moment estimate
            let v_hat = v[i] / (1.0 - beta2.powi(self.step as i32));

            // Update parameters
            if self.config.optimizer_type == OptimizerType::AdamW {
                // AdamW: decoupled weight decay
                params[i] -= lr * (m_hat / (v_hat.sqrt() + epsilon) + wd * params[i]);
            } else {
                // Adam: weight decay as L2 regularization
                let effective_grad = m_hat / (v_hat.sqrt() + epsilon);
                params[i] -= lr * (effective_grad + wd * params[i]);
            }
        }
    }

    /// SGD step
    fn step_sgd(&mut self, param_name: &str, gradient: &[f64], params: &mut [f64], lr: f64) {
        let momentum = self.config.params.get("momentum").copied().unwrap_or(0.0);
        let dampening = self.config.params.get("dampening").copied().unwrap_or(0.0);
        let nesterov = self.config.params.get("nesterov").copied().unwrap_or(0.0) > 0.5;
        let wd = self.config.weight_decay;

        // Initialize velocity if needed
        let vel = self
            .velocity
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; params.len()]);

        for i in 0..params.len() {
            let mut grad = gradient[i];

            // Gradient clipping
            if let Some(clip) = self.config.grad_clip {
                grad = grad.clamp(-clip, clip);
            }

            // Add weight decay
            if wd != 0.0 {
                grad += wd * params[i];
            }

            // Update velocity
            if momentum != 0.0 {
                if self.step > 1 {
                    vel[i] = momentum * vel[i] + (1.0 - dampening) * grad;
                } else {
                    vel[i] = grad;
                }

                // Apply Nesterov momentum if enabled
                if nesterov {
                    grad += momentum * vel[i];
                } else {
                    grad = vel[i];
                }
            }

            // Update parameters
            params[i] -= lr * grad;
        }
    }

    /// Zero the optimizer state (clear moments/velocity)
    pub fn zero_grad(&mut self) {
        // In practice, this would zero gradients in the computation graph
        // For now, we just track that we're ready for a new step
    }

    /// Get optimizer statistics
    pub fn stats(&self) -> OptimizerStats {
        OptimizerStats {
            step: self.step,
            learning_rate: self.current_lr,
            num_param_groups: self.first_moments.len().max(self.velocity.len()),
        }
    }
}

/// Optimizer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStats {
    /// Current step number
    pub step: usize,

    /// Current learning rate
    pub learning_rate: f64,

    /// Number of parameter groups
    pub num_param_groups: usize,
}

/// Learning rate schedules
pub mod schedules {
    use super::*;

    /// Constant learning rate
    pub fn constant(lr: f64) -> LRScheduleFn {
        Box::new(move |_step| lr)
    }

    /// Linear warmup followed by constant
    pub fn warmup(warmup_steps: usize, base_lr: f64) -> LRScheduleFn {
        Box::new(move |step| {
            if step < warmup_steps {
                base_lr * (step as f64 / warmup_steps as f64)
            } else {
                base_lr
            }
        })
    }

    /// Warmup followed by cosine annealing
    pub fn warmup_cosine(warmup_steps: usize, total_steps: usize, base_lr: f64) -> LRScheduleFn {
        Box::new(move |step| {
            if step < warmup_steps {
                // Linear warmup
                base_lr * (step as f64 / warmup_steps as f64)
            } else {
                // Cosine annealing
                let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                base_lr * cosine
            }
        })
    }

    /// Step decay (reduce LR at specific steps)
    pub fn step_decay(base_lr: f64, decay_factor: f64, decay_steps: Vec<usize>) -> LRScheduleFn {
        Box::new(move |step| {
            let num_decays = decay_steps.iter().filter(|&&s| step >= s).count();
            base_lr * decay_factor.powi(num_decays as i32)
        })
    }

    /// Exponential decay
    pub fn exponential_decay(base_lr: f64, decay_rate: f64) -> LRScheduleFn {
        Box::new(move |step| base_lr * decay_rate.powi(step as i32))
    }

    /// Polynomial decay
    pub fn polynomial_decay(
        base_lr: f64,
        end_lr: f64,
        total_steps: usize,
        power: f64,
    ) -> LRScheduleFn {
        Box::new(move |step| {
            let progress = (step.min(total_steps) as f64 / total_steps as f64).powf(power);
            base_lr - (base_lr - end_lr) * progress
        })
    }
}

impl std::fmt::Debug for OptimizerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptimizerState")
            .field("config", &self.config)
            .field("step", &self.step)
            .field("current_lr", &self.current_lr)
            .field("first_moments", &self.first_moments)
            .field("second_moments", &self.second_moments)
            .field("velocity", &self.velocity)
            .field("lr_schedule", &"<function>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_adam() {
        let config = OptimizerConfig::adam()
            .learning_rate(0.001)
            .betas(0.9, 0.999)
            .epsilon(1e-8);

        assert_eq!(config.optimizer_type, OptimizerType::Adam);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.params.get("beta1"), Some(&0.9));
        assert_eq!(config.params.get("beta2"), Some(&0.999));
    }

    #[test]
    fn test_optimizer_config_adamw() {
        let config = OptimizerConfig::adamw()
            .learning_rate(0.001)
            .weight_decay(0.01);

        assert_eq!(config.optimizer_type, OptimizerType::AdamW);
        assert_eq!(config.weight_decay, 0.01);
    }

    #[test]
    fn test_optimizer_config_sgd() {
        let config = OptimizerConfig::sgd()
            .learning_rate(0.1)
            .momentum(0.9)
            .nesterov(true);

        assert_eq!(config.optimizer_type, OptimizerType::SGD);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.params.get("momentum"), Some(&0.9));
        assert_eq!(config.params.get("nesterov"), Some(&1.0));
    }

    #[test]
    fn test_optimizer_state_creation() {
        let config = OptimizerConfig::adam();
        let state = config.init();

        assert_eq!(state.step, 0);
        assert_eq!(state.current_lr, 0.001);
        assert!(state.first_moments.is_empty());
        assert!(state.second_moments.is_empty());
    }

    #[test]
    fn test_adam_step() {
        let config = OptimizerConfig::adam().learning_rate(0.1);
        let mut state = config.init();

        let mut params = vec![1.0, 2.0, 3.0];
        let gradient = vec![0.1, 0.2, 0.3];

        let original_params = params.clone();

        state.step_simple("test_param", &gradient, &mut params);

        // Parameters should have been updated
        assert_ne!(params, original_params);
        assert_eq!(state.step, 1);

        // Moments should be initialized
        assert!(state.first_moments.contains_key("test_param"));
        assert!(state.second_moments.contains_key("test_param"));
    }

    #[test]
    fn test_sgd_step() {
        let config = OptimizerConfig::sgd().learning_rate(0.1).momentum(0.9);
        let mut state = config.init();

        let mut params = vec![1.0, 2.0, 3.0];
        let gradient = vec![0.1, 0.2, 0.3];

        let original_params = params.clone();

        state.step_simple("test_param", &gradient, &mut params);

        // Parameters should have been updated
        assert_ne!(params, original_params);
        assert_eq!(state.step, 1);

        // Velocity should be initialized
        assert!(state.velocity.contains_key("test_param"));
    }

    #[test]
    fn test_gradient_clipping() {
        let config = OptimizerConfig::sgd().learning_rate(0.1).grad_clip(0.5);
        let mut state = config.init();

        let mut params = vec![1.0];
        let gradient = vec![10.0]; // Large gradient

        state.step_simple("test_param", &gradient, &mut params);

        // Gradient should have been clipped, so change should be limited
        assert!(params[0] < 1.0);
        assert!(params[0] > 0.9); // Not too large of a change
    }

    #[test]
    fn test_lr_schedule_constant() {
        let schedule = schedules::constant(0.001);
        assert_eq!(schedule(0), 0.001);
        assert_eq!(schedule(100), 0.001);
        assert_eq!(schedule(1000), 0.001);
    }

    #[test]
    fn test_lr_schedule_warmup() {
        let schedule = schedules::warmup(100, 0.001);

        assert_eq!(schedule(0), 0.0);
        assert_eq!(schedule(50), 0.0005);
        assert_eq!(schedule(100), 0.001);
        assert_eq!(schedule(200), 0.001);
    }

    #[test]
    fn test_lr_schedule_warmup_cosine() {
        let schedule = schedules::warmup_cosine(100, 1000, 0.001);

        // Warmup phase
        assert!(schedule(0) < schedule(50));
        assert!(schedule(50) < schedule(100));

        // Cosine phase
        assert!(schedule(500) < schedule(100));
        assert!(schedule(1000) < schedule(500));
    }

    #[test]
    fn test_lr_schedule_step_decay() {
        let schedule = schedules::step_decay(0.1, 0.5, vec![100, 200, 300]);

        assert_eq!(schedule(0), 0.1);
        assert_eq!(schedule(50), 0.1);
        assert_eq!(schedule(100), 0.05);
        assert_eq!(schedule(200), 0.025);
        assert_eq!(schedule(300), 0.0125);
    }

    #[test]
    fn test_optimizer_with_schedule() {
        let config = OptimizerConfig::adam().learning_rate(0.001);
        let state = config.init().with_schedule(schedules::warmup(100, 0.001));

        assert_eq!(state.get_lr(), 0.0); // Step 0
    }

    #[test]
    fn test_optimizer_stats() {
        let config = OptimizerConfig::adam();
        let mut state = config.init();

        let mut params = vec![1.0, 2.0];
        let gradient = vec![0.1, 0.2];

        state.step_simple("param1", &gradient, &mut params);

        let stats = state.stats();
        assert_eq!(stats.step, 1);
        assert_eq!(stats.learning_rate, 0.001);
        assert_eq!(stats.num_param_groups, 1);
    }

    #[test]
    fn test_multiple_param_groups() {
        let config = OptimizerConfig::adam();
        let mut state = config.init();

        let mut params1 = vec![1.0];
        let mut params2 = vec![2.0];
        let gradient = vec![0.1];

        state.step_simple("param1", &gradient, &mut params1);
        state.step_simple("param2", &gradient, &mut params2);

        let stats = state.stats();
        assert_eq!(stats.num_param_groups, 2);
    }
}
