//! Error types for the JANUS Optimizer
//!
//! This module defines all error types that can occur during optimization.

use thiserror::Error;

/// Result type alias for optimizer operations
pub type Result<T> = std::result::Result<T, OptimizerError>;

/// Errors that can occur during optimization
#[derive(Error, Debug)]
pub enum OptimizerError {
    /// No trials completed during optimization
    #[error("No trials completed during optimization")]
    NoTrialsCompleted,

    /// Missing data for an asset
    #[error("Missing data for asset: {0}")]
    MissingData(String),

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Parameter out of bounds
    #[error("Parameter '{name}' value {value} is out of bounds [{min}, {max}]")]
    ParameterOutOfBounds {
        name: String,
        value: f64,
        min: f64,
        max: f64,
    },

    /// Insufficient data for backtesting
    #[error("Insufficient data: need at least {required} rows, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Backtest failed
    #[error("Backtest failed: {0}")]
    BacktestFailed(String),

    /// Search space is empty or invalid
    #[error("Invalid search space: {0}")]
    InvalidSearchSpace(String),

    /// Sampler error
    #[error("Sampler error: {0}")]
    SamplerError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Redis connection/publish error
    #[error("Redis error: {0}")]
    RedisError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Polars data processing error
    #[error("Data processing error: {0}")]
    PolarsError(String),

    /// Asset not found in registry
    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    /// Constraint violation
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    /// Timeout during optimization
    #[error("Optimization timed out after {0} seconds")]
    Timeout(u64),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<polars::error::PolarsError> for OptimizerError {
    fn from(err: polars::error::PolarsError) -> Self {
        OptimizerError::PolarsError(err.to_string())
    }
}

impl From<serde_json::Error> for OptimizerError {
    fn from(err: serde_json::Error) -> Self {
        OptimizerError::SerializationError(err.to_string())
    }
}

#[cfg(feature = "redis-publish")]
impl From<redis::RedisError> for OptimizerError {
    fn from(err: redis::RedisError) -> Self {
        OptimizerError::RedisError(err.to_string())
    }
}

impl OptimizerError {
    /// Create a new invalid parameter error
    pub fn invalid_param(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Create a new parameter out of bounds error
    pub fn out_of_bounds(name: impl Into<String>, value: f64, min: f64, max: f64) -> Self {
        Self::ParameterOutOfBounds {
            name: name.into(),
            value,
            min,
            max,
        }
    }

    /// Create a new insufficient data error
    pub fn insufficient_data(required: usize, actual: usize) -> Self {
        Self::InsufficientData { required, actual }
    }

    /// Check if this is a recoverable error (can retry)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            OptimizerError::BacktestFailed(_)
                | OptimizerError::RedisError(_)
                | OptimizerError::Timeout(_)
        )
    }

    /// Check if this is a configuration error
    pub fn is_config_error(&self) -> bool {
        matches!(
            self,
            OptimizerError::InvalidParameter { .. }
                | OptimizerError::ParameterOutOfBounds { .. }
                | OptimizerError::InvalidSearchSpace(_)
                | OptimizerError::ConfigError(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OptimizerError::NoTrialsCompleted;
        assert_eq!(err.to_string(), "No trials completed during optimization");

        let err = OptimizerError::MissingData("BTC".to_string());
        assert_eq!(err.to_string(), "Missing data for asset: BTC");

        let err = OptimizerError::invalid_param("ema_fast", "must be positive");
        assert_eq!(
            err.to_string(),
            "Invalid parameter 'ema_fast': must be positive"
        );

        let err = OptimizerError::out_of_bounds("ema_fast", 100.0, 5.0, 20.0);
        assert_eq!(
            err.to_string(),
            "Parameter 'ema_fast' value 100 is out of bounds [5, 20]"
        );
    }

    #[test]
    fn test_is_recoverable() {
        assert!(OptimizerError::BacktestFailed("test".into()).is_recoverable());
        assert!(OptimizerError::RedisError("connection".into()).is_recoverable());
        assert!(OptimizerError::Timeout(30).is_recoverable());
        assert!(!OptimizerError::NoTrialsCompleted.is_recoverable());
        assert!(!OptimizerError::ConfigError("bad".into()).is_recoverable());
    }

    #[test]
    fn test_is_config_error() {
        assert!(OptimizerError::invalid_param("x", "bad").is_config_error());
        assert!(OptimizerError::out_of_bounds("x", 1.0, 0.0, 0.5).is_config_error());
        assert!(OptimizerError::InvalidSearchSpace("empty".into()).is_config_error());
        assert!(!OptimizerError::NoTrialsCompleted.is_config_error());
        assert!(!OptimizerError::BacktestFailed("test".into()).is_config_error());
    }

    #[test]
    fn test_insufficient_data() {
        let err = OptimizerError::insufficient_data(1000, 500);
        assert_eq!(
            err.to_string(),
            "Insufficient data: need at least 1000 rows, got 500"
        );
    }
}
