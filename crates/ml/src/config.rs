//! Configuration types for ML models and training

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::backend::BackendConfig;

/// Main ML system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Backend configuration
    pub backend: BackendConfig,

    /// Model configurations
    pub models: Vec<ModelConfig>,

    /// Training configuration
    pub training: TrainingConfig,

    /// Inference configuration
    pub inference: InferenceConfig,

    /// Feature extraction configuration
    pub features: FeatureConfig,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            backend: BackendConfig::default(),
            models: vec![ModelConfig::default()],
            training: TrainingConfig::default(),
            inference: InferenceConfig::default(),
            features: FeatureConfig::default(),
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier
    pub name: String,

    /// Model type
    pub model_type: ModelType,

    /// Path to saved model file
    pub model_path: Option<PathBuf>,

    /// Model hyperparameters
    pub hyperparameters: ModelHyperparameters,

    /// Whether to enable this model
    pub enabled: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "default_model".to_string(),
            model_type: ModelType::Lstm,
            model_path: None,
            hyperparameters: ModelHyperparameters::default(),
            enabled: true,
        }
    }
}

/// Model type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// LSTM for time series prediction
    Lstm,

    /// MLP for classification
    Mlp,

    /// Ensemble of multiple models
    Ensemble,

    /// Custom model type
    Custom,
}

/// Model hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHyperparameters {
    /// Input feature dimension
    pub input_size: usize,

    /// Hidden layer sizes
    pub hidden_sizes: Vec<usize>,

    /// Output dimension
    pub output_size: usize,

    /// Dropout rate (0.0 - 1.0)
    pub dropout: f64,

    /// Use batch normalization
    pub batch_norm: bool,

    /// Activation function
    pub activation: ActivationType,

    /// Number of LSTM layers (for LSTM models)
    pub num_layers: Option<usize>,

    /// Bidirectional LSTM
    pub bidirectional: Option<bool>,
}

impl Default for ModelHyperparameters {
    fn default() -> Self {
        Self {
            input_size: 50,
            hidden_sizes: vec![64, 32],
            output_size: 1,
            dropout: 0.2,
            batch_norm: true,
            activation: ActivationType::Relu,
            num_layers: Some(2),
            bidirectional: Some(false),
        }
    }
}

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationType {
    Relu,
    Tanh,
    Sigmoid,
    Gelu,
    Swish,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Optimizer type
    pub optimizer: OptimizerType,

    /// Weight decay (L2 regularization)
    pub weight_decay: f64,

    /// Learning rate scheduler
    pub scheduler: SchedulerConfig,

    /// Early stopping patience (epochs)
    pub early_stopping_patience: Option<usize>,

    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,

    /// Validation split ratio (0.0 - 1.0)
    pub validation_split: f64,

    /// Checkpoint directory
    pub checkpoint_dir: PathBuf,

    /// Save checkpoint every N epochs
    pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 1e-3,
            optimizer: OptimizerType::AdamW,
            weight_decay: 1e-4,
            scheduler: SchedulerConfig::default(),
            early_stopping_patience: Some(10),
            gradient_clip: Some(1.0),
            validation_split: 0.2,
            checkpoint_dir: PathBuf::from("checkpoints"),
            checkpoint_interval: 5,
        }
    }
}

/// Optimizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerType {
    Adam,
    AdamW,
    Sgd,
    RmsProp,
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler type
    pub scheduler_type: SchedulerType,

    /// Step size for StepLR
    pub step_size: Option<usize>,

    /// Gamma for StepLR/ExponentialLR
    pub gamma: Option<f64>,

    /// T_max for CosineAnnealing
    pub t_max: Option<usize>,

    /// Minimum learning rate
    pub min_lr: Option<f64>,

    /// Factor for ReduceLROnPlateau
    pub factor: Option<f64>,

    /// Patience for ReduceLROnPlateau
    pub patience: Option<usize>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: SchedulerType::CosineAnnealing,
            step_size: None,
            gamma: None,
            t_max: Some(100),
            min_lr: Some(1e-6),
            factor: Some(0.5),
            patience: Some(5),
        }
    }
}

/// Learning rate scheduler type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerType {
    /// No scheduling
    None,

    /// Step decay
    StepLR,

    /// Exponential decay
    ExponentialLR,

    /// Cosine annealing
    CosineAnnealing,

    /// Reduce on plateau
    ReduceLROnPlateau,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference
    pub batch_size: usize,

    /// Cache models in memory
    pub cache_models: bool,

    /// Maximum number of cached models
    pub max_cached_models: usize,

    /// Model cache TTL (seconds)
    pub cache_ttl_secs: u64,

    /// Warmup iterations before timing
    pub warmup_iterations: usize,

    /// Enable mixed precision inference
    pub mixed_precision: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            cache_models: true,
            max_cached_models: 5,
            cache_ttl_secs: 3600, // 1 hour
            warmup_iterations: 10,
            mixed_precision: false,
        }
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Technical indicator configuration
    pub technical: TechnicalIndicatorConfig,

    /// Time series feature configuration
    pub timeseries: TimeSeriesConfig,

    /// Market microstructure configuration
    pub microstructure: MicrostructureConfig,

    /// Feature normalization method
    pub normalization: NormalizationType,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            technical: TechnicalIndicatorConfig::default(),
            timeseries: TimeSeriesConfig::default(),
            microstructure: MicrostructureConfig::default(),
            normalization: NormalizationType::ZScore,
        }
    }
}

/// Technical indicator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicatorConfig {
    /// SMA periods
    pub sma_periods: Vec<usize>,

    /// EMA periods
    pub ema_periods: Vec<usize>,

    /// RSI period
    pub rsi_period: usize,

    /// MACD parameters (fast, slow, signal)
    pub macd_params: (usize, usize, usize),

    /// Bollinger Bands period
    pub bb_period: usize,

    /// ATR period
    pub atr_period: usize,
}

impl Default for TechnicalIndicatorConfig {
    fn default() -> Self {
        Self {
            sma_periods: vec![5, 10, 20, 50, 200],
            ema_periods: vec![9, 21, 55],
            rsi_period: 14,
            macd_params: (12, 26, 9),
            bb_period: 20,
            atr_period: 14,
        }
    }
}

/// Time series feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    /// Return calculation windows (in periods)
    pub return_windows: Vec<usize>,

    /// ACF lag values
    pub acf_lags: Vec<usize>,

    /// Volatility windows
    pub volatility_windows: Vec<usize>,

    /// Include hour of day feature
    pub include_hour: bool,

    /// Include day of week feature
    pub include_day: bool,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            return_windows: vec![1, 5, 15, 60],
            acf_lags: vec![1, 5, 10, 20],
            volatility_windows: vec![5, 15, 60],
            include_hour: true,
            include_day: true,
        }
    }
}

/// Market microstructure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureConfig {
    /// Number of order book levels to use
    pub depth_levels: usize,

    /// Trade aggregation window (seconds)
    pub trade_window_secs: u64,

    /// Large trade threshold (in base currency)
    pub large_trade_threshold: f64,

    /// Calculate volume-weighted spread
    pub vw_spread: bool,
}

impl Default for MicrostructureConfig {
    fn default() -> Self {
        Self {
            depth_levels: 10,
            trade_window_secs: 60,
            large_trade_threshold: 10.0,
            vw_spread: true,
        }
    }
}

/// Feature normalization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationType {
    /// No normalization
    None,

    /// Min-max scaling to [0, 1]
    MinMax,

    /// Z-score normalization
    ZScore,

    /// Log transformation
    Log,

    /// Robust scaler (median and IQR)
    Robust,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs() {
        let ml_config = MLConfig::default();
        assert_eq!(ml_config.models.len(), 1);
        assert_eq!(ml_config.training.batch_size, 32);
        assert_eq!(ml_config.inference.batch_size, 1);
    }

    #[test]
    fn test_model_config() {
        let config = ModelConfig::default();
        assert_eq!(config.name, "default_model");
        assert_eq!(config.model_type, ModelType::Lstm);
        assert!(config.enabled);
    }

    #[test]
    fn test_hyperparameters() {
        let params = ModelHyperparameters::default();
        assert_eq!(params.input_size, 50);
        assert_eq!(params.hidden_sizes, vec![64, 32]);
        assert_eq!(params.output_size, 1);
        assert_eq!(params.dropout, 0.2);
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.optimizer, OptimizerType::AdamW);
        assert_eq!(config.validation_split, 0.2);
    }

    #[test]
    fn test_feature_config() {
        let config = FeatureConfig::default();
        assert_eq!(config.technical.sma_periods, vec![5, 10, 20, 50, 200]);
        assert_eq!(config.technical.rsi_period, 14);
        assert_eq!(config.normalization, NormalizationType::ZScore);
    }

    #[test]
    fn test_serialization() {
        let config = MLConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: MLConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.training.batch_size, config.training.batch_size);
        assert_eq!(deserialized.models.len(), config.models.len());
    }
}
