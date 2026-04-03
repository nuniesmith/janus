//! Machine Learning Infrastructure for JANUS Trading System
//!
//! This crate provides the ML foundation for JANUS, including:
//! - Feature extraction from market data
//! - Model training and inference
//! - Integration with data quality pipeline
//! - Support for multiple backends (CPU/GPU)
//!
//! # Architecture
//!
//! The ML crate is organized into several modules:
//! - `backend` - Backend configuration (CPU/GPU)
//! - `features` - Feature extraction from market data
//! - `models` - ML model implementations
//! - `training` - Training infrastructure
//! - `inference` - Inference engine and prediction
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use janus_ml::{
//!     backend::cpu_backend,
//!     features::TechnicalIndicators,
//!     models::LstmConfig,
//! };
//!
//! // Set up backend
//! let backend = cpu_backend();
//! backend.initialize().unwrap();
//!
//! // Extract features from market data
//! let features = TechnicalIndicators::default();
//! // let tensor = features.extract(&market_data).unwrap();
//!
//! // Train or load a model
//! // let model = LstmConfig::default().build(&backend.device());
//! ```
//!
//! # Integration with Data Quality
//!
//! The ML crate integrates seamlessly with the data quality pipeline:
//!
//! ```rust,ignore
//! use janus_ml::pipeline::MLPipeline;
//! use janus_data_quality::DataQualityPipeline;
//!
//! // Create integrated pipeline
//! // let dq_pipeline = DataQualityPipeline::from_config(config, state);
//! // let ml_pipeline = MLPipeline::new(predictor, dq_pipeline);
//!
//! // Process events with quality checks + ML inference
//! // let result = ml_pipeline.process(events).await?;
//! ```

pub mod backend;
pub mod config;
pub mod dataset;
pub mod dqn;
pub mod error;
pub mod evaluation;
pub mod features;
pub mod models;
pub mod optimizer;
pub mod training;
pub mod training_autodiff;

// Inference engine (to be implemented)
// pub mod inference;

// Re-export commonly used types
pub use backend::{
    AutodiffCpuBackend, BackendConfig, BackendDevice, CpuBackend, auto_backend, cpu_backend,
};
pub use config::{MLConfig, ModelConfig};
pub use dataset::{
    DataLoader, MarketDataSample, MarketDataset, SampleMetadata, WindowConfig, WindowedDataset,
};
pub use error::{MLError, Result};
pub use evaluation::{
    ClassificationMetrics, EvaluationReport, MetricsCalculator, RegressionMetrics, TradingMetrics,
};
pub use features::{
    ExtractedFeatures, FeatureExtractor, price::PriceFeatures, volume::VolumeFeatures,
};
pub use models::{
    ActivationType, LstmConfig, LstmPredictor, MlpClassifier, MlpConfig, Model, ModelMetadata,
    ModelRecord, SerializedTensor, TrainableLstm, TrainableLstmConfig, TrainableMlp,
    TrainableMlpConfig, WeightMap, WeightMapValidation, extract_trainable_lstm_weights,
    extract_trainable_mlp_weights, soft_update_lstm_target, soft_update_mlp_target,
    soft_update_weight_maps, trainable_lstm_config_to_inference, trainable_lstm_to_predictor,
    trainable_lstm_to_predictor_auto, trainable_mlp_config_to_inference,
    trainable_mlp_to_classifier, trainable_mlp_to_classifier_auto, validate_lstm_weight_map,
    validate_mlp_weight_map,
};
pub use optimizer::{OptimizerConfig, OptimizerState, OptimizerType};
pub use training::{Checkpoint, Trainer, TrainingConfig, TrainingHistory};

/// Re-export core Burn tensor types so downstream crates (e.g. `janus-backward`)
/// can construct tensors without adding `burn-core` as a direct dependency.
pub mod tensor {
    pub use burn_core::tensor::{Tensor, TensorData, backend::Backend};
}
pub use dqn::{DqnOnlineModel, DqnStepResult, compute_double_dqn_targets};
pub use training_autodiff::{AutodiffTrainer, AutodiffTrainingConfig, AutodiffTrainingHistory};

#[cfg(feature = "gpu")]
pub use backend::{AutodiffGpuBackend, GpuBackend, gpu_backend};

/// Version of the ML crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// ML system metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MLMetadata {
    /// Crate version
    pub version: String,

    /// Backend type (cpu/gpu)
    pub backend: String,

    /// Build timestamp
    pub build_timestamp: chrono::DateTime<chrono::Utc>,

    /// Enabled features
    pub features: Vec<String>,
}

impl MLMetadata {
    /// Create metadata for the current ML system
    pub fn current(backend_type: &str) -> Self {
        #[cfg(feature = "gpu")]
        let mut features = vec!["cpu".to_string()];
        #[cfg(not(feature = "gpu"))]
        let features = vec!["cpu".to_string()];

        #[cfg(feature = "gpu")]
        {
            features.push("gpu".to_string());
        }

        Self {
            version: VERSION.to_string(),
            backend: backend_type.to_string(),
            build_timestamp: chrono::Utc::now(),
            features,
        }
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.features.contains(&"gpu".to_string())
    }
}

/// Initialize the ML system with default configuration
pub fn init() -> Result<BackendConfig> {
    let config = auto_backend();
    config.initialize()?;
    tracing::info!(
        "JANUS ML v{} initialized with {} backend",
        VERSION,
        config.device().backend_type()
    );
    Ok(config)
}

/// Initialize the ML system with custom configuration
pub fn init_with_config(config: BackendConfig) -> Result<()> {
    config.initialize()?;
    tracing::info!(
        "JANUS ML v{} initialized with {} backend",
        VERSION,
        config.device().backend_type()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert!(VERSION.contains("."));
    }

    #[test]
    fn test_metadata() {
        let metadata = MLMetadata::current("cpu");
        assert_eq!(metadata.version, VERSION);
        assert_eq!(metadata.backend, "cpu");
        assert!(!metadata.features.is_empty());
        assert!(metadata.features.contains(&"cpu".to_string()));
    }

    #[test]
    fn test_init() {
        let result = init();
        assert!(result.is_ok());
        let config = result.unwrap();
        assert!(!config.device().is_gpu() || cfg!(feature = "gpu"));
    }

    #[test]
    fn test_init_with_config() {
        let config = cpu_backend().with_threads(2).with_seed(42);
        let result = init_with_config(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metadata_has_gpu() {
        let metadata = MLMetadata::current("cpu");

        #[cfg(feature = "gpu")]
        assert!(metadata.has_gpu());

        #[cfg(not(feature = "gpu"))]
        assert!(!metadata.has_gpu());
    }
}
