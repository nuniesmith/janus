//! LTN Configuration Module
//!
//! This module defines configuration structures for the Logic Tensor Network,
//! including neural architecture hyperparameters, training settings, and
//! axiom weights.
//!
//! # Configuration Categories
//!
//! 1. **Model Architecture**: Layer sizes, activation functions, dropout
//! 2. **Training**: Learning rate, batch size, optimizer settings
//! 3. **Semantic Loss**: Axiom weights, semantic weight (α)
//! 4. **Inference**: Performance targets, batch processing
//!
//! # Example
//!
//! ```ignore
//! use janus_ltn::config::{LtnConfig, TrainingConfig};
//!
//! // Use default configuration
//! let config = LtnConfig::default();
//!
//! // Or customize for high-frequency trading
//! let config = LtnConfig::high_frequency();
//!
//! // Or build custom configuration
//! let config = LtnConfig::builder()
//!     .hidden_dims(vec![32, 64, 32])
//!     .dropout_rate(0.2)
//!     .semantic_weight(0.5)
//!     .learning_rate(0.001)
//!     .build();
//! ```

use serde::{Deserialize, Serialize};

/// Complete LTN configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LtnConfig {
    /// Model architecture configuration
    pub model: ModelConfig,

    /// Training configuration
    pub training: TrainingConfig,

    /// Axiom library configuration
    pub axioms: AxiomConfig,

    /// Inference configuration
    pub inference: InferenceConfig,
}

impl LtnConfig {
    /// High-frequency trading configuration (fast inference, conservative)
    pub fn high_frequency() -> Self {
        Self {
            model: ModelConfig {
                hidden_dims: vec![32, 32], // Smaller, faster
                dropout_rate: 0.1,
                ..ModelConfig::default()
            },
            training: TrainingConfig {
                learning_rate: 0.0005,
                batch_size: 64,
                ..TrainingConfig::default()
            },
            axioms: AxiomConfig::risk_focused(),
            inference: InferenceConfig {
                target_latency_us: 5,
                max_latency_us: 20,
                ..InferenceConfig::default()
            },
        }
    }

    /// Low-frequency trading configuration (larger model, more learning)
    pub fn low_frequency() -> Self {
        Self {
            model: ModelConfig {
                hidden_dims: vec![64, 128, 64],
                dropout_rate: 0.3,
                ..ModelConfig::default()
            },
            training: TrainingConfig {
                learning_rate: 0.002,
                batch_size: 256,
                num_epochs: 200,
                ..TrainingConfig::default()
            },
            axioms: AxiomConfig::balanced(),
            inference: InferenceConfig {
                target_latency_us: 50,
                max_latency_us: 500,
                ..InferenceConfig::default()
            },
        }
    }

    /// Research configuration (focus on interpretability)
    pub fn research() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig {
                early_stopping_patience: 20,
                ..TrainingConfig::default()
            },
            axioms: AxiomConfig::interpretable(),
            inference: InferenceConfig::default(),
        }
    }

    /// Create a builder for custom configuration
    pub fn builder() -> LtnConfigBuilder {
        LtnConfigBuilder::new()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        self.model.validate()?;
        self.training.validate()?;
        self.axioms.validate()?;
        self.inference.validate()?;
        Ok(())
    }
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Input dimension (fixed at 8 for DSP features)
    pub input_dim: usize,

    /// Hidden layer dimensions (e.g., [32, 64, 32])
    pub hidden_dims: Vec<usize>,

    /// Output dimension (fixed at 3 for long/neutral/short)
    pub output_dim: usize,

    /// Dropout rate for regularization
    pub dropout_rate: f64,

    /// L2 weight decay coefficient
    pub l2_weight_decay: f64,

    /// Activation function type
    pub activation: ActivationType,

    /// Use batch normalization
    pub use_batch_norm: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            hidden_dims: vec![32, 64, 32],
            output_dim: 3,
            dropout_rate: 0.2,
            l2_weight_decay: 1e-4,
            activation: ActivationType::ReLU,
            use_batch_norm: false,
        }
    }
}

impl ModelConfig {
    /// Count total parameters
    pub fn count_parameters(&self) -> usize {
        let mut total = 0;
        let mut prev_dim = self.input_dim;

        for &dim in &self.hidden_dims {
            total += prev_dim * dim + dim; // Weights + biases
            prev_dim = dim;
        }

        // Output layer
        total += prev_dim * self.output_dim + self.output_dim;

        total
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim != 8 {
            return Err(format!("input_dim must be 8, got {}", self.input_dim));
        }

        if self.output_dim != 3 {
            return Err(format!("output_dim must be 3, got {}", self.output_dim));
        }

        if self.hidden_dims.is_empty() {
            return Err("hidden_dims cannot be empty".to_string());
        }

        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err(format!(
                "dropout_rate must be in [0, 1), got {}",
                self.dropout_rate
            ));
        }

        if self.l2_weight_decay < 0.0 {
            return Err(format!(
                "l2_weight_decay must be non-negative, got {}",
                self.l2_weight_decay
            ));
        }

        Ok(())
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    LeakyReLU,
    Tanh,
    GELU,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,

    /// Learning rate schedule
    pub lr_schedule: LrSchedule,

    /// Batch size
    pub batch_size: usize,

    /// Number of training epochs
    pub num_epochs: usize,

    /// Semantic weight (α in hybrid loss)
    pub semantic_weight: f64,

    /// Semantic weight schedule
    pub semantic_schedule: SemanticSchedule,

    /// Optimizer type
    pub optimizer: OptimizerType,

    /// Gradient clipping max norm
    pub grad_clip_norm: f64,

    /// Early stopping patience (epochs)
    pub early_stopping_patience: usize,

    /// Validation split ratio
    pub val_split: f64,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            lr_schedule: LrSchedule::CosineAnnealing { min_lr: 1e-6 },
            batch_size: 128,
            num_epochs: 100,
            semantic_weight: 0.5,
            semantic_schedule: SemanticSchedule::Adaptive,
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            grad_clip_norm: 1.0,
            early_stopping_patience: 10,
            val_split: 0.2,
            seed: 42,
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err(format!(
                "learning_rate must be positive, got {}",
                self.learning_rate
            ));
        }

        if self.batch_size == 0 {
            return Err("batch_size must be positive".to_string());
        }

        if self.semantic_weight < 0.0 || self.semantic_weight > 1.0 {
            return Err(format!(
                "semantic_weight must be in [0, 1], got {}",
                self.semantic_weight
            ));
        }

        if self.val_split < 0.0 || self.val_split >= 1.0 {
            return Err(format!(
                "val_split must be in [0, 1), got {}",
                self.val_split
            ));
        }

        Ok(())
    }
}

/// Learning rate schedule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LrSchedule {
    /// Constant learning rate
    Constant,

    /// Step decay: lr *= gamma every step_size epochs
    StepDecay { step_size: usize, gamma: f64 },

    /// Exponential decay: lr *= gamma every epoch
    ExponentialDecay { gamma: f64 },

    /// Cosine annealing
    CosineAnnealing { min_lr: f64 },

    /// One-cycle policy
    OneCycle { max_lr: f64, pct_start: f64 },
}

/// Semantic weight schedule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticSchedule {
    /// Constant semantic weight
    Constant,

    /// Start high on supervised, gradually increase semantic
    /// Phase 1 (epochs 0-20): α = 0.8 (focus on data)
    /// Phase 2 (epochs 21-60): α = 0.5 (balance)
    /// Phase 3 (epochs 61+): α = 0.3 (focus on axioms)
    Adaptive,

    /// Linear decay from supervised to semantic
    LinearDecay,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD { momentum: f64, nesterov: bool },

    /// Adam optimizer
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },

    /// AdamW (Adam with decoupled weight decay)
    AdamW {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },

    /// RMSprop
    RMSprop { alpha: f64, epsilon: f64 },
}

/// Axiom configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomConfig {
    /// Weights for the 10 core axioms
    pub axiom_weights: [f64; 10],

    /// Enable/disable individual axioms
    pub axiom_enabled: [bool; 10],

    /// T-norm type for conjunctions
    pub tnorm_type: String, // "product", "goedel", "lukasiewicz"

    /// Implication type
    pub implication_type: String, // "reichenbach", "goedel", "lukasiewicz"
}

impl Default for AxiomConfig {
    fn default() -> Self {
        Self {
            axiom_weights: [2.0, 2.0, 1.5, 1.5, 3.0, 2.5, 2.0, 1.0, 5.0, 1.0],
            axiom_enabled: [true; 10],
            tnorm_type: "product".to_string(),
            implication_type: "reichenbach".to_string(),
        }
    }
}

impl AxiomConfig {
    /// Risk-focused configuration (emphasize safety axioms)
    pub fn risk_focused() -> Self {
        Self {
            axiom_weights: [1.0, 1.0, 0.5, 0.5, 5.0, 4.0, 3.0, 2.0, 5.0, 0.5],
            axiom_enabled: [true; 10],
            tnorm_type: "product".to_string(),
            implication_type: "reichenbach".to_string(),
        }
    }

    /// Balanced configuration
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Interpretable configuration (equal weights for analysis)
    pub fn interpretable() -> Self {
        Self {
            axiom_weights: [1.0; 10],
            axiom_enabled: [true; 10],
            tnorm_type: "product".to_string(),
            implication_type: "reichenbach".to_string(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.axiom_weights.len() != 10 {
            return Err("axiom_weights must have 10 elements".to_string());
        }

        for (i, &weight) in self.axiom_weights.iter().enumerate() {
            if weight < 0.0 {
                return Err(format!("axiom_weights[{}] must be non-negative", i));
            }
        }

        let valid_tnorms = ["product", "goedel", "lukasiewicz"];
        if !valid_tnorms.contains(&self.tnorm_type.as_str()) {
            return Err(format!("Invalid tnorm_type: {}", self.tnorm_type));
        }

        let valid_implications = ["reichenbach", "goedel", "lukasiewicz"];
        if !valid_implications.contains(&self.implication_type.as_str()) {
            return Err(format!(
                "Invalid implication_type: {}",
                self.implication_type
            ));
        }

        Ok(())
    }
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Target median latency (microseconds)
    pub target_latency_us: u64,

    /// Maximum acceptable P99 latency (microseconds)
    pub max_latency_us: u64,

    /// Batch inference size (if using batching)
    pub batch_size: Option<usize>,

    /// Device type for inference
    pub device: DeviceType,

    /// Number of threads for CPU inference
    pub num_threads: Option<usize>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 10,
            max_latency_us: 50,
            batch_size: None,
            device: DeviceType::Cpu,
            num_threads: Some(1), // Single-threaded for latency
        }
    }
}

impl InferenceConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.target_latency_us == 0 {
            return Err("target_latency_us must be positive".to_string());
        }

        if self.max_latency_us < self.target_latency_us {
            return Err("max_latency_us must be >= target_latency_us".to_string());
        }

        if let Some(batch_size) = self.batch_size
            && batch_size == 0
        {
            return Err("batch_size must be positive".to_string());
        }

        Ok(())
    }
}

/// Device type for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
}

/// Configuration builder
pub struct LtnConfigBuilder {
    model: ModelConfig,
    training: TrainingConfig,
    axioms: AxiomConfig,
    inference: InferenceConfig,
}

impl LtnConfigBuilder {
    pub fn new() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            axioms: AxiomConfig::default(),
            inference: InferenceConfig::default(),
        }
    }

    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.model.hidden_dims = dims;
        self
    }

    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.model.dropout_rate = rate;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.training.learning_rate = lr;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.training.batch_size = size;
        self
    }

    pub fn semantic_weight(mut self, weight: f64) -> Self {
        self.training.semantic_weight = weight;
        self
    }

    pub fn num_epochs(mut self, epochs: usize) -> Self {
        self.training.num_epochs = epochs;
        self
    }

    pub fn axiom_weights(mut self, weights: [f64; 10]) -> Self {
        self.axioms.axiom_weights = weights;
        self
    }

    pub fn target_latency_us(mut self, latency: u64) -> Self {
        self.inference.target_latency_us = latency;
        self
    }

    pub fn build(self) -> LtnConfig {
        LtnConfig {
            model: self.model,
            training: self.training,
            axioms: self.axioms,
            inference: self.inference,
        }
    }
}

impl Default for LtnConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LtnConfig::default();
        assert!(config.validate().is_ok());

        assert_eq!(config.model.input_dim, 8);
        assert_eq!(config.model.output_dim, 3);
        assert_eq!(config.model.hidden_dims, vec![32, 64, 32]);
    }

    #[test]
    fn test_high_frequency_config() {
        let config = LtnConfig::high_frequency();
        assert!(config.validate().is_ok());

        // Should have smaller model
        assert!(config.model.hidden_dims.len() <= 3);

        // Should have stricter latency requirements
        assert!(config.inference.target_latency_us <= 10);
    }

    #[test]
    fn test_low_frequency_config() {
        let config = LtnConfig::low_frequency();
        assert!(config.validate().is_ok());

        // Can have larger model
        assert!(config.model.hidden_dims.iter().sum::<usize>() > 100);
    }

    #[test]
    fn test_parameter_count() {
        let config = ModelConfig::default();
        let params = config.count_parameters();

        // 8→32: 8*32+32 = 288
        // 32→64: 32*64+64 = 2112
        // 64→32: 64*32+32 = 2080
        // 32→3: 32*3+3 = 99
        // Total: 4579
        assert_eq!(params, 4579);
    }

    #[test]
    fn test_config_builder() {
        let config = LtnConfig::builder()
            .hidden_dims(vec![16, 32, 16])
            .dropout_rate(0.3)
            .learning_rate(0.002)
            .semantic_weight(0.7)
            .build();

        assert!(config.validate().is_ok());
        assert_eq!(config.model.hidden_dims, vec![16, 32, 16]);
        assert_eq!(config.model.dropout_rate, 0.3);
        assert_eq!(config.training.learning_rate, 0.002);
        assert_eq!(config.training.semantic_weight, 0.7);
    }

    #[test]
    fn test_validation_errors() {
        let mut config = LtnConfig::default();

        // Invalid input dim
        config.model.input_dim = 10;
        assert!(config.validate().is_err());

        config.model.input_dim = 8;

        // Invalid dropout
        config.model.dropout_rate = 1.5;
        assert!(config.validate().is_err());

        config.model.dropout_rate = 0.2;

        // Invalid semantic weight
        config.training.semantic_weight = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_axiom_config() {
        let config = AxiomConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.axiom_weights.len(), 10);

        let risk = AxiomConfig::risk_focused();
        assert!(risk.axiom_weights[4] > risk.axiom_weights[0]); // Low conf > trending
    }

    #[test]
    fn test_serialization() {
        let config = LtnConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: LtnConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.model.input_dim, deserialized.model.input_dim);
        assert_eq!(
            config.training.learning_rate,
            deserialized.training.learning_rate
        );
    }
}
