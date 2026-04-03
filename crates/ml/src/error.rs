//! Error types for the ML crate

use thiserror::Error;

/// Result type for ML operations
pub type Result<T> = std::result::Result<T, MLError>;

/// Error types for machine learning operations
#[derive(Debug, Error)]
pub enum MLError {
    /// Data-related errors
    #[error("Data loading error: {0}")]
    DataLoadingError(String),

    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Invalid data shape: expected {expected}, got {actual}")]
    InvalidShape { expected: String, actual: String },

    #[error("Missing features: {0}")]
    MissingFeatures(String),

    /// Model errors
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model load error: {0}")]
    ModelLoadError(String),

    #[error("Model save error: {0}")]
    ModelSaveError(String),

    #[error("Model inference error: {0}")]
    InferenceError(String),

    /// Training errors
    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Convergence failed: {0}")]
    ConvergenceError(String),

    /// Feature extraction errors
    #[error("Feature extraction error: {0}")]
    FeatureExtractionError(String),

    #[error("Normalization error: {0}")]
    NormalizationError(String),

    /// Configuration errors
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Missing configuration: {0}")]
    MissingConfig(String),

    /// Backend errors
    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Device not available: {0}")]
    DeviceNotAvailable(String),

    /// I/O errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Postcard error: {0}")]
    Postcard(#[from] postcard::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Data quality integration
    #[error("Data quality check failed: {0}")]
    DataQualityError(String),

    /// Other errors
    #[error("Internal error: {0}")]
    Internal(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl MLError {
    /// Create an insufficient data error
    pub fn insufficient_data(msg: impl Into<String>) -> Self {
        Self::InsufficientData(msg.into())
    }

    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create an invalid shape error
    pub fn invalid_shape(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::InvalidShape {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a feature extraction error
    pub fn feature_extraction(msg: impl Into<String>) -> Self {
        Self::FeatureExtractionError(msg.into())
    }

    /// Create a model load error
    pub fn model_load(msg: impl Into<String>) -> Self {
        Self::ModelLoadError(msg.into())
    }

    /// Create an inference error
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::InferenceError(msg.into())
    }

    /// Create a training error
    pub fn training(msg: impl Into<String>) -> Self {
        Self::TrainingError(msg.into())
    }

    /// Create an invalid config error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create an IO error
    pub fn io_error(msg: impl Into<String>) -> Self {
        Self::Io(std::io::Error::other(msg.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = MLError::insufficient_data("Need at least 100 samples");
        assert!(err.to_string().contains("Insufficient data"));

        let err = MLError::invalid_shape("(32, 10)", "(16, 5)");
        assert!(err.to_string().contains("expected (32, 10)"));

        let err = MLError::feature_extraction("RSI calculation failed");
        assert!(err.to_string().contains("Feature extraction"));
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let ml_err: MLError = io_err.into();
        assert!(matches!(ml_err, MLError::Io(_)));
    }
}
