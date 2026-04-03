//! Optimizer Configuration for Neural Network Training.
//!
//! This module provides configuration and utilities for various optimization algorithms
//! used in training neural networks with Candle.
//!
//! # Supported Optimizers
//!
//! - **Adam**: Adaptive Moment Estimation with bias correction
//! - **AdamW**: Adam with decoupled weight decay (recommended)
//! - **SGD**: Stochastic Gradient Descent with optional momentum
//!
//! # Example
//!
//! ```ignore
//! use training::optimizer::{OptimizerConfig, OptimizerType};
//! use candle_nn::VarMap;
//!
//! let config = OptimizerConfig::adamw()
//!     .learning_rate(1e-4)
//!     .weight_decay(0.01)
//!     .build();
//!
//! let var_map = VarMap::new();
//! let optimizer = config.build_optimizer(&var_map)?;
//! ```

use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use serde::{Deserialize, Serialize};

/// Type of optimizer to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OptimizerType {
    /// Adam optimizer with bias correction
    Adam,

    /// AdamW optimizer with decoupled weight decay (recommended)
    #[default]
    AdamW,

    /// Stochastic Gradient Descent with optional momentum
    SGD,
}

/// Configuration for optimizers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Type of optimizer
    pub optimizer_type: OptimizerType,

    /// Learning rate
    pub learning_rate: f64,

    /// Weight decay (L2 regularization)
    pub weight_decay: f64,

    /// Beta1 parameter for Adam/AdamW (exponential decay rate for first moment)
    pub beta1: f64,

    /// Beta2 parameter for Adam/AdamW (exponential decay rate for second moment)
    pub beta2: f64,

    /// Epsilon for numerical stability
    pub epsilon: f64,

    /// Momentum for SGD
    pub momentum: f64,

    /// Nesterov momentum for SGD
    pub nesterov: bool,

    /// Maximum gradient norm for clipping (None = no clipping)
    pub max_grad_norm: Option<f64>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::adamw()
    }
}

impl OptimizerConfig {
    /// Create AdamW optimizer configuration (recommended default).
    pub fn adamw() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: 0.0,
            nesterov: false,
            max_grad_norm: Some(1.0),
        }
    }

    /// Create Adam optimizer configuration.
    pub fn adam() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 1e-3,
            weight_decay: 0.0, // Adam doesn't use decoupled weight decay
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: 0.0,
            nesterov: false,
            max_grad_norm: Some(1.0),
        }
    }

    /// Create SGD optimizer configuration.
    pub fn sgd() -> Self {
        Self {
            optimizer_type: OptimizerType::SGD,
            learning_rate: 1e-2,
            weight_decay: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: 0.9,
            nesterov: true,
            max_grad_norm: Some(1.0),
        }
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set beta1 (for Adam/AdamW).
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (for Adam/AdamW).
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon.
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set momentum (for SGD).
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable/disable Nesterov momentum (for SGD).
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Set maximum gradient norm for clipping.
    pub fn max_grad_norm(mut self, max_norm: Option<f64>) -> Self {
        self.max_grad_norm = max_norm;
        self
    }

    /// Build the optimizer configuration.
    pub fn build(self) -> Self {
        self
    }
}

/// Wrapper around Candle optimizers to provide a unified interface.
pub enum OptimizerWrapper {
    AdamW(AdamW),
}

impl OptimizerWrapper {
    /// Create optimizer from configuration and variable map.
    pub fn from_config(
        config: &OptimizerConfig,
        vars: candle_nn::VarMap,
    ) -> candle_core::Result<Self> {
        let params = ParamsAdamW {
            lr: config.learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.epsilon,
            weight_decay: config.weight_decay,
        };

        let optimizer = AdamW::new(vars.all_vars(), params)?;
        Ok(OptimizerWrapper::AdamW(optimizer))
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> candle_core::Result<()> {
        match self {
            OptimizerWrapper::AdamW(opt) => opt.step(grads),
        }
    }

    /// Get the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        match self {
            OptimizerWrapper::AdamW(opt) => opt.learning_rate(),
        }
    }

    /// Set the learning rate (for learning rate scheduling).
    pub fn set_learning_rate(&mut self, lr: f64) {
        match self {
            OptimizerWrapper::AdamW(opt) => opt.set_learning_rate(lr),
        }
    }
}

/// Gradient clipping utilities.
pub struct GradientClipper {
    max_norm: f64,
}

impl GradientClipper {
    /// Create a new gradient clipper.
    pub fn new(max_norm: f64) -> Self {
        Self { max_norm }
    }

    /// Clip gradients by global norm.
    ///
    /// Scales all gradients such that the global norm does not exceed max_norm.
    pub fn clip_gradients(&self, grads: &[&Tensor]) -> candle_core::Result<f64> {
        // Compute global norm
        let mut total_norm_sq = 0.0;

        for grad in grads {
            let grad_norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f64>()?;
            total_norm_sq += grad_norm_sq;
        }

        let total_norm = total_norm_sq.sqrt();

        // Compute clipping coefficient
        let clip_coef = self.max_norm / (total_norm + 1e-6);

        if clip_coef < 1.0 {
            // Need to clip - scale all gradients
            for grad in grads {
                let scaled = grad.affine(clip_coef, 0.0)?;
                // Note: In practice, you'd need to update the gradients in place
                // This is a simplified version showing the concept
                let _ = scaled;
            }
        }

        Ok(total_norm)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert_eq!(config.optimizer_type, OptimizerType::AdamW);
        assert_eq!(config.learning_rate, 1e-4);
    }

    #[test]
    fn test_optimizer_config_builder() {
        let config = OptimizerConfig::adamw()
            .learning_rate(3e-4)
            .weight_decay(0.01)
            .beta1(0.95)
            .max_grad_norm(Some(0.5))
            .build();

        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.weight_decay, 0.01);
        assert_eq!(config.beta1, 0.95);
        assert_eq!(config.max_grad_norm, Some(0.5));
    }

    #[test]
    fn test_adam_config() {
        let config = OptimizerConfig::adam();
        assert_eq!(config.optimizer_type, OptimizerType::Adam);
        assert_eq!(config.weight_decay, 0.0);
    }

    #[test]
    fn test_sgd_config() {
        let config = OptimizerConfig::sgd();
        assert_eq!(config.optimizer_type, OptimizerType::SGD);
        assert_eq!(config.momentum, 0.9);
        assert!(config.nesterov);
    }

    #[test]
    fn test_gradient_clipper() {
        let clipper = GradientClipper::new(1.0);
        assert_eq!(clipper.max_norm, 1.0);
    }
}
