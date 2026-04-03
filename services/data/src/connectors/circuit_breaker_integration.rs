//! Circuit Breaker Integration for Exchange Connectors
//!
//! This module provides circuit breaker wrappers for exchange API calls to prevent
//! cascading failures during rate limit events (429s) or exchange outages.
//!
//! ## Features
//! - Automatic circuit opening after consecutive failures
//! - Fast-fail behavior when circuit is open
//! - Gradual recovery testing via half-open state
//! - Per-exchange circuit breaker instances
//! - Prometheus metrics integration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   API Request                           │
//! └─────────────────────┬───────────────────────────────────┘
//!                       │
//!                       ▼
//!          ┌────────────────────────┐
//!          │   Circuit Breaker      │
//!          │   Check State          │
//!          └────────┬───────────────┘
//!                   │
//!        ┌──────────┼──────────┐
//!        │          │          │
//!    CLOSED      OPEN      HALF-OPEN
//!        │          │          │
//!        │          │          │
//!        ▼          ▼          ▼
//!   Execute    Fail Fast   Test Recovery
//!   Request                    │
//!        │                     │
//!        └─────────┬───────────┘
//!                  │
//!       ┌──────────┴──────────┐
//!       │                     │
//!    Success              Failure
//!       │                     │
//!       ▼                     ▼
//!  Reset Count        Increment Count
//!       │                     │
//!       │              ┌──────┴──────┐
//!       │              │             │
//!       │         Threshold?      < Threshold
//!       │              │             │
//!       │              ▼             │
//!       │         OPEN Circuit       │
//!       │              │             │
//!       └──────────────┴─────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_data::connectors::circuit_breaker_integration::ExchangeCircuitBreakers;
//! use fks_ruby::config::Exchange;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let breakers = ExchangeCircuitBreakers::new();
//!
//! // Wrap an API call with circuit breaker
//! let result = breakers.call(Exchange::Binance, || async {
//!     // Your exchange API call here
//!     Ok::<_, anyhow::Error>(())
//! }).await;
//!
//! match result {
//!     Ok(_) => println!("API call succeeded"),
//!     Err(e) => println!("API call failed or circuit open: {}", e),
//! }
//! # Ok(())
//! # }
//! ```

use crate::config::Exchange;
use crate::metrics::prometheus_exporter::CIRCUIT_BREAKER_STATE;
use janus_rate_limiter::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Default circuit breaker configuration for exchanges
pub fn default_circuit_config() -> CircuitBreakerConfig {
    CircuitBreakerConfig {
        failure_threshold: 5,             // Open after 5 consecutive failures
        success_threshold: 2,             // Close after 2 consecutive successes in half-open
        timeout: Duration::from_secs(60), // Wait 60s before testing recovery
    }
}

/// Aggressive circuit breaker config (for less reliable exchanges)
pub fn aggressive_circuit_config() -> CircuitBreakerConfig {
    CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 3,
        timeout: Duration::from_secs(120), // Longer cooldown
    }
}

// ============================================================================
// Exchange Circuit Breakers
// ============================================================================

/// Circuit breaker manager for all exchange connectors
pub struct ExchangeCircuitBreakers {
    /// Per-exchange circuit breakers
    breakers: Arc<RwLock<HashMap<Exchange, Arc<CircuitBreaker>>>>,

    /// Default configuration for new breakers
    default_config: CircuitBreakerConfig,
}

impl ExchangeCircuitBreakers {
    /// Create a new circuit breaker manager with default configuration
    pub fn new() -> Self {
        Self::with_config(default_circuit_config())
    }

    /// Create a new circuit breaker manager with custom configuration
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: Arc::new(RwLock::new(HashMap::new())),
            default_config: config,
        }
    }

    /// Get or create a circuit breaker for an exchange
    async fn get_breaker(&self, exchange: Exchange) -> Arc<CircuitBreaker> {
        // Try read lock first (fast path)
        {
            let breakers = self.breakers.read().await;
            if let Some(breaker) = breakers.get(&exchange) {
                return Arc::clone(breaker);
            }
        }

        // Need to create new breaker (slow path)
        let mut breakers = self.breakers.write().await;

        // Double-check in case another task created it
        if let Some(breaker) = breakers.get(&exchange) {
            return Arc::clone(breaker);
        }

        // Create new breaker (CircuitBreaker::new returns Arc<CircuitBreaker>)
        let breaker = CircuitBreaker::new(self.default_config.clone());

        info!(
            exchange = ?exchange,
            failure_threshold = self.default_config.failure_threshold,
            timeout_secs = self.default_config.timeout.as_secs(),
            "Circuit breaker initialized for exchange"
        );

        breakers.insert(exchange, Arc::clone(&breaker));
        breaker
    }

    /// Execute a function with circuit breaker protection
    ///
    /// If the circuit is open, this will fail fast without executing the function.
    /// If the function returns an error, it will be counted toward the failure threshold.
    pub async fn call<F, Fut, T>(&self, exchange: Exchange, f: F) -> anyhow::Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = anyhow::Result<T>>,
    {
        let breaker = self.get_breaker(exchange).await;

        // Update metrics before call
        self.update_metrics(exchange, breaker.state());

        // Execute with circuit breaker
        let result = breaker.call(f).await;

        // Update metrics after call
        self.update_metrics(exchange, breaker.state());

        // Log state transitions
        match breaker.state() {
            CircuitState::Open => {
                warn!(
                    exchange = ?exchange,
                    "Circuit breaker OPEN - failing fast to prevent cascading failures"
                );
            }
            CircuitState::HalfOpen => {
                info!(
                    exchange = ?exchange,
                    "Circuit breaker HALF-OPEN - testing recovery"
                );
            }
            CircuitState::Closed => {
                debug!(exchange = ?exchange, "Circuit breaker CLOSED - normal operation");
            }
        }

        result.map_err(|e| anyhow::anyhow!("Circuit breaker error for {}: {}", exchange, e))
    }

    /// Check if circuit is open for an exchange
    pub async fn is_circuit_open(&self, exchange: Exchange) -> bool {
        let breakers = self.breakers.read().await;
        if let Some(breaker) = breakers.get(&exchange) {
            breaker.state() == CircuitState::Open
        } else {
            false
        }
    }

    /// Get circuit state for an exchange
    pub async fn get_state(&self, exchange: Exchange) -> CircuitState {
        let breakers = self.breakers.read().await;
        if let Some(breaker) = breakers.get(&exchange) {
            breaker.state()
        } else {
            CircuitState::Closed
        }
    }

    /// Update Prometheus metrics for circuit breaker state
    fn update_metrics(&self, exchange: Exchange, state: CircuitState) {
        let state_value = match state {
            CircuitState::Closed => 0,
            CircuitState::Open => 1,
            CircuitState::HalfOpen => 2,
        };

        CIRCUIT_BREAKER_STATE
            .with_label_values(&[&exchange.to_string().to_lowercase()])
            .set(state_value);
    }

    /// Reset circuit breaker for an exchange (useful for testing or manual intervention)
    pub async fn reset(&self, exchange: Exchange) {
        let mut breakers = self.breakers.write().await;
        // Create a new breaker to reset state (CircuitBreaker::new returns Arc)
        breakers.insert(exchange, CircuitBreaker::new(self.default_config.clone()));

        info!(exchange = ?exchange, "Circuit breaker manually reset");
    }

    /// Get all circuit breaker states (useful for monitoring)
    pub async fn get_all_states(&self) -> HashMap<Exchange, CircuitState> {
        let breakers = self.breakers.read().await;
        breakers
            .iter()
            .map(|(exchange, breaker)| (*exchange, breaker.state()))
            .collect()
    }
}

impl Default for ExchangeCircuitBreakers {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if an error is a rate limit error (429)
pub fn is_rate_limit_error(error: &anyhow::Error) -> bool {
    let error_string = error.to_string().to_lowercase();
    error_string.contains("429")
        || error_string.contains("rate limit")
        || error_string.contains("too many requests")
}

/// Check if an error is a temporary network error that should trigger circuit breaker
pub fn is_temporary_error(error: &anyhow::Error) -> bool {
    let error_string = error.to_string().to_lowercase();
    error_string.contains("timeout")
        || error_string.contains("connection")
        || error_string.contains("503")
        || error_string.contains("502")
        || error_string.contains("504")
        || is_rate_limit_error(error)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_creation() {
        let breakers = ExchangeCircuitBreakers::new();

        // Initially no breakers exist
        let states = breakers.get_all_states().await;
        assert!(states.is_empty());
    }

    #[tokio::test]
    async fn test_get_state_for_new_exchange() {
        let breakers = ExchangeCircuitBreakers::new();

        // Getting state for non-existent breaker returns Closed
        let state = breakers.get_state(Exchange::Binance).await;
        assert_eq!(state, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_successful_call() {
        let breakers = ExchangeCircuitBreakers::new();

        let result = breakers
            .call(Exchange::Binance, || async { Ok::<_, anyhow::Error>(42) })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Circuit should still be closed
        let state = breakers.get_state(Exchange::Binance).await;
        assert_eq!(state, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_failed_calls_open_circuit() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
        };
        let breakers = ExchangeCircuitBreakers::with_config(config);

        // Make 3 failing calls with rate limit errors to open circuit
        for _ in 0..3 {
            let _ = breakers
                .call(Exchange::Binance, || async {
                    Err::<(), _>(anyhow::anyhow!("HTTP 429 Rate limit exceeded"))
                })
                .await;
        }

        // Circuit should be open
        let state = breakers.get_state(Exchange::Binance).await;
        assert_eq!(state, CircuitState::Open);

        // Next call should fail fast
        let result = breakers
            .call(Exchange::Binance, || async { Ok::<_, anyhow::Error>(42) })
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_is_rate_limit_error() {
        assert!(is_rate_limit_error(&anyhow::anyhow!(
            "HTTP 429 Too Many Requests"
        )));
        assert!(is_rate_limit_error(&anyhow::anyhow!("Rate limit exceeded")));
        assert!(is_rate_limit_error(&anyhow::anyhow!("too many requests")));
        assert!(!is_rate_limit_error(&anyhow::anyhow!("Connection timeout")));
    }

    #[tokio::test]
    async fn test_is_temporary_error() {
        assert!(is_temporary_error(&anyhow::anyhow!("Connection timeout")));
        assert!(is_temporary_error(&anyhow::anyhow!(
            "HTTP 503 Service Unavailable"
        )));
        assert!(is_temporary_error(&anyhow::anyhow!("429 Rate limit")));
        assert!(!is_temporary_error(&anyhow::anyhow!("Invalid API key")));
    }

    #[tokio::test]
    async fn test_reset_circuit() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(10),
        };
        let breakers = ExchangeCircuitBreakers::with_config(config);

        // Open the circuit with rate limit errors
        for _ in 0..2 {
            let _ = breakers
                .call(Exchange::Binance, || async {
                    Err::<(), _>(anyhow::anyhow!("429 Too Many Requests"))
                })
                .await;
        }

        assert!(breakers.is_circuit_open(Exchange::Binance).await);

        // Reset it
        breakers.reset(Exchange::Binance).await;

        // Should be closed now
        assert!(!breakers.is_circuit_open(Exchange::Binance).await);
    }

    #[tokio::test]
    async fn test_multiple_exchanges() {
        let breakers = ExchangeCircuitBreakers::new();

        // Make calls to different exchanges
        let _ = breakers
            .call(Exchange::Binance, || async { Ok::<_, anyhow::Error>(1) })
            .await;

        let _ = breakers
            .call(Exchange::Bybit, || async { Ok::<_, anyhow::Error>(2) })
            .await;

        // Both should have closed circuits
        let states = breakers.get_all_states().await;
        assert_eq!(states.len(), 2);
        assert_eq!(states.get(&Exchange::Binance), Some(&CircuitState::Closed));
        assert_eq!(states.get(&Exchange::Bybit), Some(&CircuitState::Closed));
    }
}
