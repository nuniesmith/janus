//! Error types for the data quality system

use thiserror::Error;

/// Result type for data quality operations
pub type Result<T> = std::result::Result<T, DataQualityError>;

/// Errors that can occur in the data quality system
#[derive(Debug, Error)]
pub enum DataQualityError {
    /// Validation error
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Anomaly detection error
    #[error("Anomaly detection failed: {0}")]
    AnomalyDetectionFailed(String),

    /// Gap detection error
    #[error("Gap detection failed: {0}")]
    GapDetectionFailed(String),

    /// Export error
    #[error("Export failed: {0}")]
    ExportFailed(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// State error
    #[error("State error: {0}")]
    StateError(String),

    /// Invalid data
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// IO error (from std::io::Error)
    #[error("IO error: {0}")]
    StdIoError(#[from] std::io::Error),

    /// IO error (from string description)
    #[error("IO error: {0}")]
    IoError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Parquet error
    #[error("Parquet error: {0}")]
    ParquetError(String),

    /// Arrow error
    #[error("Arrow error: {0}")]
    ArrowError(String),

    /// Database error
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Metrics error
    #[cfg(feature = "cns-metrics")]
    #[error("Metrics error: {0}")]
    MetricsError(String),

    /// Generic error
    #[error("Error: {0}")]
    Other(String),
}

impl From<parquet::errors::ParquetError> for DataQualityError {
    fn from(err: parquet::errors::ParquetError) -> Self {
        DataQualityError::ParquetError(err.to_string())
    }
}

impl From<arrow::error::ArrowError> for DataQualityError {
    fn from(err: arrow::error::ArrowError) -> Self {
        DataQualityError::ArrowError(err.to_string())
    }
}

impl From<String> for DataQualityError {
    fn from(err: String) -> Self {
        DataQualityError::Other(err)
    }
}

impl From<&str> for DataQualityError {
    fn from(err: &str) -> Self {
        DataQualityError::Other(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DataQualityError::ValidationFailed("price out of range".to_string());
        assert_eq!(err.to_string(), "Validation failed: price out of range");

        let err = DataQualityError::InvalidData("null value".to_string());
        assert_eq!(err.to_string(), "Invalid data: null value");
    }

    #[test]
    fn test_error_conversion() {
        let err: DataQualityError = "test error".into();
        assert!(matches!(err, DataQualityError::Other(_)));

        let err: DataQualityError = "test error".to_string().into();
        assert!(matches!(err, DataQualityError::Other(_)));
    }
}
