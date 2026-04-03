//! Error types for the FKS Execution Service

use thiserror::Error;

/// Main error type for the execution service
#[derive(Error, Debug)]
pub enum ExecutionError {
    /// Kill switch is active — all order submission is blocked.
    #[error("Kill switch active: {0}")]
    KillSwitchActive(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Exchange API errors
    #[error("Exchange error ({exchange}): {message}")]
    Exchange {
        exchange: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Order validation errors
    #[error("Order validation failed: {0}")]
    OrderValidation(String),

    /// Validation errors (alias for compatibility)
    #[error("Validation error: {0}")]
    Validation(String),

    /// Invalid order state
    #[error("Invalid order state: {0}")]
    InvalidOrderState(String),

    /// Risk management errors
    #[error("Risk check failed: {0}")]
    RiskViolation(String),

    /// Position management errors
    #[error("Position error: {0}")]
    Position(String),

    /// Database/storage errors
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    /// Network/connection errors
    #[error("Network error: {0}")]
    Network(String),

    /// Authentication/authorization errors
    #[error("Authentication error: {0}")]
    Auth(String),

    /// Authentication error (alias)
    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    /// Parse errors
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Order not found
    #[error("Order not found: {0}")]
    OrderNotFound(String),

    /// Position not found
    #[error("Position not found: {0}")]
    PositionNotFound(String),

    /// Insufficient balance
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance { required: f64, available: f64 },

    /// Rate limit exceeded
    #[error("Rate limit exceeded for exchange {0}")]
    RateLimitExceeded(String),

    /// Invalid state transition
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// gRPC errors
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    /// Internal service errors
    #[error("Internal error: {0}")]
    Internal(String),

    /// Timeout errors
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Compliance check failures
    #[error("Compliance check failed: {0}")]
    Compliance(String),

    /// Unknown exchange
    #[error("Unknown exchange: {0}")]
    UnknownExchange(String),

    /// Symbol not supported
    #[error("Symbol not supported: {0}")]
    UnsupportedSymbol(String),

    /// WebSocket errors
    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    /// Signature/authentication errors
    #[error("Signature error: {0}")]
    SignatureError(String),

    /// Deserialization errors
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Serialization errors (specific)
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid quantity
    #[error("Invalid quantity: {0}")]
    InvalidQuantity(String),

    /// Feature not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Storage-related errors
#[derive(Error, Debug)]
pub enum StorageError {
    /// Redis errors
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    /// QuestDB errors
    #[error("QuestDB error: {0}")]
    QuestDb(String),

    /// Serialization errors for storage
    #[error("Storage serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Connection pool errors
    #[error("Connection pool error: {0}")]
    Pool(String),

    /// Connection errors
    #[error("Connection error: {0}")]
    Connection(String),

    /// Write errors
    #[error("Write error: {0}")]
    Write(String),

    /// Data corruption
    #[error("Data corruption detected: {0}")]
    Corruption(String),
}

/// Result type alias for execution operations
pub type Result<T> = std::result::Result<T, ExecutionError>;

/// Result type alias for storage operations
pub type StorageResult<T> = std::result::Result<T, StorageError>;

impl ExecutionError {
    /// Create a new exchange error
    pub fn exchange(exchange: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Exchange {
            exchange: exchange.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a new exchange error with a source error
    pub fn exchange_with_source(
        exchange: impl Into<String>,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Exchange {
            exchange: exchange.into(),
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Check if this is a retryable error
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ExecutionError::Network(_)
                | ExecutionError::RateLimitExceeded(_)
                | ExecutionError::Timeout(_)
                | ExecutionError::Storage(_)
        )
    }

    /// Check if this is a kill switch error
    pub fn is_kill_switch(&self) -> bool {
        matches!(self, ExecutionError::KillSwitchActive(_))
    }

    /// Check if this is a rate limit error
    pub fn is_rate_limit(&self) -> bool {
        matches!(self, ExecutionError::RateLimitExceeded(_))
    }

    /// Convert to gRPC status
    pub fn to_grpc_status(&self) -> tonic::Status {
        use tonic::{Code, Status};

        match self {
            ExecutionError::OrderNotFound(_) | ExecutionError::PositionNotFound(_) => {
                Status::new(Code::NotFound, self.to_string())
            }
            ExecutionError::OrderValidation(_) | ExecutionError::RiskViolation(_) => {
                Status::new(Code::InvalidArgument, self.to_string())
            }
            ExecutionError::Auth(_) => Status::new(Code::Unauthenticated, self.to_string()),
            ExecutionError::KillSwitchActive(_) => {
                Status::new(Code::FailedPrecondition, self.to_string())
            }
            ExecutionError::InsufficientBalance { .. } => {
                Status::new(Code::FailedPrecondition, self.to_string())
            }
            ExecutionError::RateLimitExceeded(_) => {
                Status::new(Code::ResourceExhausted, self.to_string())
            }
            ExecutionError::Timeout(_) => Status::new(Code::DeadlineExceeded, self.to_string()),
            ExecutionError::UnknownExchange(_) | ExecutionError::UnsupportedSymbol(_) => {
                Status::new(Code::InvalidArgument, self.to_string())
            }
            ExecutionError::Grpc(status) => status.clone(),
            _ => Status::new(Code::Internal, self.to_string()),
        }
    }
}

impl From<reqwest::Error> for ExecutionError {
    fn from(err: reqwest::Error) -> Self {
        ExecutionError::Network(err.to_string())
    }
}

impl From<url::ParseError> for ExecutionError {
    fn from(err: url::ParseError) -> Self {
        ExecutionError::Config(format!("Invalid URL: {}", err))
    }
}

impl From<config::ConfigError> for ExecutionError {
    fn from(err: config::ConfigError) -> Self {
        ExecutionError::Config(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ExecutionError::OrderNotFound("order123".to_string());
        assert_eq!(err.to_string(), "Order not found: order123");
    }

    #[test]
    fn test_insufficient_balance() {
        let err = ExecutionError::InsufficientBalance {
            required: 1000.0,
            available: 500.0,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));
    }

    #[test]
    fn test_retryable_errors() {
        assert!(ExecutionError::Network("test".into()).is_retryable());
        assert!(ExecutionError::RateLimitExceeded("kraken".into()).is_retryable());
        assert!(ExecutionError::Timeout("test".into()).is_retryable());
        assert!(!ExecutionError::OrderNotFound("test".into()).is_retryable());
    }

    #[test]
    fn test_rate_limit_detection() {
        assert!(ExecutionError::RateLimitExceeded("kraken".into()).is_rate_limit());
        assert!(!ExecutionError::Network("test".into()).is_rate_limit());
    }

    #[test]
    fn test_grpc_status_conversion() {
        use tonic::Code;

        let err = ExecutionError::OrderNotFound("order123".into());
        let status = err.to_grpc_status();
        assert_eq!(status.code(), Code::NotFound);

        let err = ExecutionError::OrderValidation("invalid".into());
        let status = err.to_grpc_status();
        assert_eq!(status.code(), Code::InvalidArgument);

        let err = ExecutionError::RateLimitExceeded("kraken".into());
        let status = err.to_grpc_status();
        assert_eq!(status.code(), Code::ResourceExhausted);
    }
}
