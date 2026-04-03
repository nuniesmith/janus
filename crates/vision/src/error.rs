//! Error types for vision pipeline

use thiserror::Error;

/// Vision pipeline errors
#[derive(Debug, Error)]
pub enum VisionError {
    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidShape { expected: String, actual: String },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Tensor operation failed: {0}")]
    TensorError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for vision operations
pub type Result<T> = std::result::Result<T, VisionError>;
