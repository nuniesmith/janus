//! Error types for JANUS

use thiserror::Error;

/// Main error type for JANUS
#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Module error in {module}: {message}")]
    Module { module: String, message: String },

    #[error("Signal error: {0}")]
    Signal(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Channel send error: {0}")]
    ChannelSend(String),

    #[error("Channel receive error")]
    ChannelRecv,

    #[error("Health check failed: {0}")]
    HealthCheck(String),

    #[error("Shutdown error: {0}")]
    Shutdown(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Create a module error
    pub fn module(module: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Module {
            module: module.into(),
            message: message.into(),
        }
    }
}

/// Result type alias using our Error
pub type Result<T> = std::result::Result<T, Error>;

impl<T> From<tokio::sync::broadcast::error::SendError<T>> for Error {
    fn from(err: tokio::sync::broadcast::error::SendError<T>) -> Self {
        Self::ChannelSend(err.to_string())
    }
}

impl From<tokio::sync::broadcast::error::RecvError> for Error {
    fn from(_: tokio::sync::broadcast::error::RecvError) -> Self {
        Self::ChannelRecv
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_error() {
        let err = Error::module("forward", "failed to start");
        assert!(err.to_string().contains("forward"));
    }
}
