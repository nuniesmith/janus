//! Error types for Project JANUS.

use anyhow;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum JanusError {
    #[error("Market data error: {0}")]
    MarketData(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Risk violation: {0}")]
    RiskViolation(String),

    #[error("Logic constraint violation: {0}")]
    LogicViolation(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Insufficient funds: {0}")]
    InsufficientFunds(String),

    #[error("Exchange error: {0}")]
    ExchangeError(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Rate limited: {0}")]
    RateLimited(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Operation cancelled: {0}")]
    Cancelled(String),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<anyhow::Error> for JanusError {
    fn from(err: anyhow::Error) -> Self {
        JanusError::Internal(err.to_string())
    }
}

impl From<String> for JanusError {
    fn from(err: String) -> Self {
        JanusError::Other(err)
    }
}

impl From<&str> for JanusError {
    fn from(err: &str) -> Self {
        JanusError::Other(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, JanusError>;
