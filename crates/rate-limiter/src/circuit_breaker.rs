//! Circuit Breaker for Rate Limiter
//!
//! Implements the circuit breaker pattern to prevent cascading failures when
//! rate limits are exceeded. After a threshold of consecutive failures (429s),
//! the circuit opens and requests fail fast without hitting the API.
//!
//! ## States
//! - **Closed**: Normal operation, requests pass through
//! - **Open**: Failing fast, no requests to API
//! - **HalfOpen**: Testing if service recovered
//!
//! ## Example
//! ```
//! use janus_rate_limiter::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
//! use std::time::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = CircuitBreakerConfig {
//!     failure_threshold: 5,
//!     success_threshold: 2,
//!     timeout: Duration::from_secs(60),
//! };
//!
//! let breaker = CircuitBreaker::new(config);
//!
//! // Wrap API calls
//! match breaker.call(|| async {
//!     // Your API call here
//!     Ok::<_, anyhow::Error>(())
//! }).await {
//!     Ok(result) => println!("Success: {:?}", result),
//!     Err(e) => println!("Failed or circuit open: {}", e),
//! }
//! # Ok(())
//! # }
//! ```

use parking_lot::RwLock;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Circuit breaker errors
#[derive(Error, Debug)]
pub enum CircuitBreakerError {
    /// Circuit is open, requests are failing fast
    #[error("Circuit breaker is OPEN - failing fast to prevent cascading failures")]
    CircuitOpen,

    /// The underlying operation failed
    #[error("Operation failed: {0}")]
    OperationFailed(#[from] anyhow::Error),

    /// Rate limit error (429)
    #[error("Rate limit exceeded (429)")]
    RateLimitExceeded,
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CircuitState {
    /// Normal operation - requests pass through
    Closed = 0,
    /// Circuit opened - requests fail immediately
    Open = 1,
    /// Testing recovery - limited requests allowed
    HalfOpen = 2,
}

impl From<u8> for CircuitState {
    fn from(value: u8) -> Self {
        match value {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed, // Default to closed for invalid values
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening circuit
    pub failure_threshold: u32,

    /// Number of consecutive successes in HalfOpen before closing
    pub success_threshold: u32,

    /// How long to wait before attempting recovery (Open -> HalfOpen)
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,             // Open after 5 consecutive 429s
            success_threshold: 2,             // Close after 2 successful requests
            timeout: Duration::from_secs(60), // Wait 60s before retry
        }
    }
}

/// Circuit breaker state machine
pub struct CircuitBreaker {
    /// Current state (0=Closed, 1=Open, 2=HalfOpen)
    state: AtomicU8,

    /// Consecutive failure count
    failures: AtomicU32,

    /// Consecutive success count (in HalfOpen state)
    successes: AtomicU32,

    /// Configuration
    config: CircuitBreakerConfig,

    /// Last state transition time
    last_transition: RwLock<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            failures: AtomicU32::new(0),
            successes: AtomicU32::new(0),
            config,
            last_transition: RwLock::new(Instant::now()),
        })
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        self.state.load(Ordering::Relaxed).into()
    }

    /// Get current failure count
    pub fn failure_count(&self) -> u32 {
        self.failures.load(Ordering::Relaxed)
    }

    /// Get current success count (only relevant in HalfOpen)
    pub fn success_count(&self) -> u32 {
        self.successes.load(Ordering::Relaxed)
    }

    /// Check if circuit should attempt recovery
    fn should_attempt_recovery(&self) -> bool {
        let elapsed = self.last_transition.read().elapsed();
        elapsed >= self.config.timeout
    }

    /// Transition to a new state
    fn transition_to(&self, new_state: CircuitState) {
        let old_state = self.state();
        self.state.store(new_state as u8, Ordering::Release);
        *self.last_transition.write() = Instant::now();

        tracing::info!(
            from = ?old_state,
            to = ?new_state,
            failures = self.failure_count(),
            successes = self.success_count(),
            "Circuit breaker state transition"
        );
    }

    /// Record a successful operation
    fn on_success(&self) {
        match self.state() {
            CircuitState::Closed => {
                // Reset failure counter on success
                self.failures.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                // Increment success counter
                let successes = self.successes.fetch_add(1, Ordering::SeqCst) + 1;

                if successes >= self.config.success_threshold {
                    // Enough successes - close the circuit
                    self.successes.store(0, Ordering::Relaxed);
                    self.failures.store(0, Ordering::Relaxed);
                    self.transition_to(CircuitState::Closed);

                    tracing::info!(
                        successes = successes,
                        "Circuit breaker recovered - closing circuit"
                    );
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but if it does, ignore
                tracing::warn!("Success recorded while circuit is open");
            }
        }
    }

    /// Record a failed operation (rate limit error)
    fn on_failure(&self) {
        match self.state() {
            CircuitState::Closed => {
                let failures = self.failures.fetch_add(1, Ordering::SeqCst) + 1;

                if failures >= self.config.failure_threshold {
                    // Too many failures - open the circuit
                    self.transition_to(CircuitState::Open);

                    tracing::error!(
                        failures = failures,
                        threshold = self.config.failure_threshold,
                        timeout_secs = self.config.timeout.as_secs(),
                        "Circuit breaker OPENED due to repeated rate limit failures"
                    );
                }
            }
            CircuitState::HalfOpen => {
                // Failed during recovery - back to Open
                self.successes.store(0, Ordering::Relaxed);
                self.transition_to(CircuitState::Open);

                tracing::warn!("Circuit breaker recovery failed - reopening circuit");
            }
            CircuitState::Open => {
                // Already open, just log
                tracing::debug!("Failure recorded while circuit is open");
            }
        }
    }

    /// Execute an operation with circuit breaker protection
    ///
    /// # Returns
    /// - `Ok(T)` - Operation succeeded
    /// - `Err(CircuitBreakerError::CircuitOpen)` - Circuit is open, failing fast
    /// - `Err(CircuitBreakerError::OperationFailed)` - Operation failed
    pub async fn call<F, Fut, T, E>(&self, f: F) -> Result<T, CircuitBreakerError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: Into<anyhow::Error>,
    {
        // Check current state
        match self.state() {
            CircuitState::Open => {
                // Check if we should attempt recovery
                if self.should_attempt_recovery() {
                    tracing::info!("Attempting circuit breaker recovery after timeout");
                    self.transition_to(CircuitState::HalfOpen);
                    // Continue to HalfOpen logic below
                } else {
                    // Still in timeout, fail fast
                    return Err(CircuitBreakerError::CircuitOpen);
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => {
                // Proceed with operation
            }
        }

        // Execute the operation
        match f().await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                let error = e.into();

                // Check if this is a rate limit error
                let is_rate_limit = error.to_string().to_lowercase().contains("429")
                    || error.to_string().to_lowercase().contains("rate limit");

                if is_rate_limit {
                    self.on_failure();
                    Err(CircuitBreakerError::RateLimitExceeded)
                } else {
                    // Other errors don't affect circuit state
                    Err(CircuitBreakerError::OperationFailed(error))
                }
            }
        }
    }

    /// Manually open the circuit (for testing or emergency)
    pub fn force_open(&self) {
        self.transition_to(CircuitState::Open);
    }

    /// Manually close the circuit (for testing or manual recovery)
    pub fn force_close(&self) {
        self.failures.store(0, Ordering::Relaxed);
        self.successes.store(0, Ordering::Relaxed);
        self.transition_to(CircuitState::Closed);
    }

    /// Reset all counters and close the circuit
    pub fn reset(&self) {
        self.force_close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[test]
    fn test_initial_state() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert_eq!(breaker.failure_count(), 0);
        assert_eq!(breaker.success_count(), 0);
    }

    #[tokio::test]
    async fn test_circuit_opens_after_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
        };
        let breaker = CircuitBreaker::new(config);

        // Simulate 3 rate limit failures
        for i in 0..3 {
            let _result = breaker
                .call(|| async { Err::<(), _>(anyhow::anyhow!("429 Rate limit exceeded")) })
                .await;

            if i < 2 {
                // First 2 failures should not open circuit
                assert_eq!(breaker.state(), CircuitState::Closed);
            }
        }

        // Circuit should be open now
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_fails_fast_when_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout: Duration::from_secs(10),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        let _ = breaker
            .call(|| async { Err::<(), _>(anyhow::anyhow!("429 Rate limit")) })
            .await;

        assert_eq!(breaker.state(), CircuitState::Open);

        // Next call should fail fast
        let result = breaker.call(|| async { Ok::<_, anyhow::Error>(()) }).await;

        assert!(matches!(result, Err(CircuitBreakerError::CircuitOpen)));
    }

    #[tokio::test]
    async fn test_circuit_recovery_halfopen() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        let _ = breaker
            .call(|| async { Err::<(), _>(anyhow::anyhow!("429")) })
            .await;

        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for timeout
        sleep(Duration::from_millis(150)).await;

        // Next call should transition to HalfOpen
        let _ = breaker.call(|| async { Ok::<_, anyhow::Error>(()) }).await;

        // Should be in HalfOpen or Closed (depending on success threshold)
        let state = breaker.state();
        assert!(state == CircuitState::HalfOpen || state == CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_closes_after_successes() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        let _ = breaker
            .call(|| async { Err::<(), _>(anyhow::anyhow!("429")) })
            .await;

        // Wait for timeout
        sleep(Duration::from_millis(150)).await;

        // 2 successful calls should close the circuit
        for _ in 0..2 {
            let _ = breaker.call(|| async { Ok::<_, anyhow::Error>(()) }).await;
        }

        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_non_rate_limit_errors_dont_open_circuit() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
        };
        let breaker = CircuitBreaker::new(config);

        // Non-rate-limit errors should not open circuit
        for _ in 0..5 {
            let _ = breaker
                .call(|| async { Err::<(), _>(anyhow::anyhow!("Network error")) })
                .await;
        }

        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_success_resets_failure_counter() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
        };
        let breaker = CircuitBreaker::new(config);

        // 2 failures
        for _ in 0..2 {
            let _ = breaker
                .call(|| async { Err::<(), _>(anyhow::anyhow!("429")) })
                .await;
        }

        assert_eq!(breaker.failure_count(), 2);

        // 1 success should reset counter
        let _ = breaker.call(|| async { Ok::<_, anyhow::Error>(()) }).await;

        assert_eq!(breaker.failure_count(), 0);
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_manual_operations() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());

        // Test force_open
        breaker.force_open();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Test force_close
        breaker.force_close();
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert_eq!(breaker.failure_count(), 0);

        // Test reset
        breaker.force_open();
        breaker.reset();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }
}
