//! Retry utilities for exchange operations
//!
//! This module provides robust retry logic with exponential backoff
//! for handling transient failures in exchange API calls.
//!
//! # Features
//!
//! - Exponential backoff with jitter
//! - Configurable retry attempts and delays
//! - Error classification (retryable vs non-retryable)
//! - Circuit breaker pattern support
//! - Per-operation timeout
//! - Metrics tracking for retry attempts
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::execution::retry::{RetryConfig, retry_with_backoff};
//!
//! let config = RetryConfig::default();
//! let result = retry_with_backoff(&config, || async {
//!     exchange.place_order(&order).await
//! }).await?;
//! ```

use crate::error::{ExecutionError, Result};

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries)
    pub max_attempts: u32,
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier (e.g., 2.0 = double delay each retry)
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0-1.0) - randomizes delay to prevent thundering herd
    pub jitter_factor: f64,
    /// Timeout for each individual operation attempt
    pub operation_timeout: Option<Duration>,
    /// Whether to retry on rate limit errors
    pub retry_on_rate_limit: bool,
    /// Special delay for rate limit errors
    pub rate_limit_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            operation_timeout: Some(Duration::from_secs(30)),
            retry_on_rate_limit: true,
            rate_limit_delay: Duration::from_secs(60),
        }
    }
}

impl RetryConfig {
    /// Create a configuration for aggressive retries (fast operations)
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
            operation_timeout: Some(Duration::from_secs(10)),
            retry_on_rate_limit: true,
            rate_limit_delay: Duration::from_secs(30),
        }
    }

    /// Create a configuration for conservative retries (important operations)
    pub fn conservative() -> Self {
        Self {
            max_attempts: 10,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(120),
            backoff_multiplier: 2.5,
            jitter_factor: 0.15,
            operation_timeout: Some(Duration::from_secs(60)),
            retry_on_rate_limit: true,
            rate_limit_delay: Duration::from_secs(120),
        }
    }

    /// Create a configuration for order placement (balance reliability vs speed)
    pub fn for_orders() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            operation_timeout: Some(Duration::from_secs(30)),
            retry_on_rate_limit: true,
            rate_limit_delay: Duration::from_secs(60),
        }
    }

    /// Create a configuration for market data (fast, less critical)
    pub fn for_market_data() -> Self {
        Self {
            max_attempts: 2,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(2),
            backoff_multiplier: 2.0,
            jitter_factor: 0.2,
            operation_timeout: Some(Duration::from_secs(5)),
            retry_on_rate_limit: false,
            rate_limit_delay: Duration::from_secs(30),
        }
    }

    /// Calculate delay for a given attempt number
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        // Calculate base exponential delay
        let base_delay = self.initial_delay.as_millis() as f64
            * self
                .backoff_multiplier
                .powi(attempt.saturating_sub(1) as i32);

        // Cap at max delay
        let capped_delay = base_delay.min(self.max_delay.as_millis() as f64);

        // Add jitter using simple random based on time
        let jitter = if self.jitter_factor > 0.0 {
            let jitter_range = capped_delay * self.jitter_factor;
            // Simple pseudo-random jitter based on current time nanoseconds
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0);
            let random_factor = (nanos % 1000) as f64 / 1000.0; // 0.0 to 1.0
            jitter_range * (random_factor * 2.0 - 1.0) // -range to +range
        } else {
            0.0
        };

        let final_delay = (capped_delay + jitter).max(0.0);
        Duration::from_millis(final_delay as u64)
    }
}

// ============================================================================
// Retry Metrics
// ============================================================================

/// Metrics for tracking retry behavior
#[derive(Debug, Default)]
pub struct RetryMetrics {
    /// Total operations attempted
    pub total_operations: AtomicU64,
    /// Operations that succeeded on first try
    pub first_try_successes: AtomicU64,
    /// Operations that succeeded after retry
    pub retry_successes: AtomicU64,
    /// Operations that failed after all retries
    pub total_failures: AtomicU64,
    /// Total retry attempts across all operations
    pub total_retries: AtomicU64,
    /// Rate limit errors encountered
    pub rate_limit_errors: AtomicU64,
    /// Network errors encountered
    pub network_errors: AtomicU64,
    /// Timeout errors encountered
    pub timeout_errors: AtomicU64,
}

impl RetryMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful operation
    pub fn record_success(&self, attempts: u32) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        if attempts == 1 {
            self.first_try_successes.fetch_add(1, Ordering::Relaxed);
        } else {
            self.retry_successes.fetch_add(1, Ordering::Relaxed);
            self.total_retries
                .fetch_add((attempts - 1) as u64, Ordering::Relaxed);
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self, attempts: u32) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        self.total_retries
            .fetch_add(attempts.saturating_sub(1) as u64, Ordering::Relaxed);
    }

    /// Record an error type
    pub fn record_error(&self, error: &ExecutionError) {
        if error.is_rate_limit() {
            self.rate_limit_errors.fetch_add(1, Ordering::Relaxed);
        } else if matches!(error, ExecutionError::Network(_)) {
            self.network_errors.fetch_add(1, Ordering::Relaxed);
        } else if matches!(error, ExecutionError::Timeout(_)) {
            self.timeout_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        let successes = self.first_try_successes.load(Ordering::Relaxed)
            + self.retry_successes.load(Ordering::Relaxed);
        successes as f64 / total as f64
    }

    /// Get first-try success rate
    pub fn first_try_rate(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        self.first_try_successes.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get average retries per operation
    pub fn avg_retries(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.total_retries.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get summary as string
    pub fn summary(&self) -> String {
        format!(
            "Operations: {}, Success Rate: {:.1}%, First-Try: {:.1}%, Avg Retries: {:.2}, \
             Rate Limits: {}, Network Errors: {}, Timeouts: {}",
            self.total_operations.load(Ordering::Relaxed),
            self.success_rate() * 100.0,
            self.first_try_rate() * 100.0,
            self.avg_retries(),
            self.rate_limit_errors.load(Ordering::Relaxed),
            self.network_errors.load(Ordering::Relaxed),
            self.timeout_errors.load(Ordering::Relaxed),
        )
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.total_operations.store(0, Ordering::Relaxed);
        self.first_try_successes.store(0, Ordering::Relaxed);
        self.retry_successes.store(0, Ordering::Relaxed);
        self.total_failures.store(0, Ordering::Relaxed);
        self.total_retries.store(0, Ordering::Relaxed);
        self.rate_limit_errors.store(0, Ordering::Relaxed);
        self.network_errors.store(0, Ordering::Relaxed);
        self.timeout_errors.store(0, Ordering::Relaxed);
    }
}

// ============================================================================
// Circuit Breaker
// ============================================================================

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected
    Open,
    /// Circuit is half-open, allowing test requests
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Duration to keep circuit open
    pub open_duration: Duration,
    /// Number of successes needed to close circuit from half-open
    pub success_threshold: u32,
    /// Time window for counting failures
    pub failure_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_duration: Duration::from_secs(60),
            success_threshold: 3,
            failure_window: Duration::from_secs(60),
        }
    }
}

/// Circuit breaker for preventing cascade failures
pub struct CircuitBreaker {
    /// Configuration
    config: CircuitBreakerConfig,
    /// Current state
    state: Arc<RwLock<CircuitState>>,
    /// Failure count in current window
    failure_count: AtomicU32,
    /// Success count (for half-open state)
    success_count: AtomicU32,
    /// Time when circuit was opened
    opened_at: Arc<RwLock<Option<Instant>>>,
    /// Time of first failure in window
    window_start: Arc<RwLock<Option<Instant>>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            opened_at: Arc::new(RwLock::new(None)),
            window_start: Arc::new(RwLock::new(None)),
        }
    }

    /// Get current circuit state
    pub async fn state(&self) -> CircuitState {
        let mut state = self.state.write().await;

        // Check if we should transition from Open to HalfOpen
        if *state == CircuitState::Open {
            if let Some(opened) = *self.opened_at.read().await {
                if opened.elapsed() >= self.config.open_duration {
                    *state = CircuitState::HalfOpen;
                    self.success_count.store(0, Ordering::Relaxed);
                    info!("Circuit breaker transitioning to half-open");
                }
            }
        }

        *state
    }

    /// Check if request is allowed
    pub async fn is_allowed(&self) -> bool {
        let state = self.state().await;
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => true, // Allow test requests
        }
    }

    /// Record a successful operation
    pub async fn record_success(&self) {
        let mut state = self.state.write().await;

        match *state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
                *self.window_start.write().await = None;
            }
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::Relaxed);
                    *self.opened_at.write().await = None;
                    *self.window_start.write().await = None;
                    info!("Circuit breaker closed after {} successes", count);
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but handle gracefully
            }
        }
    }

    /// Record a failed operation
    pub async fn record_failure(&self) {
        let mut state = self.state.write().await;

        match *state {
            CircuitState::Closed => {
                // Check if we need to start a new window
                let mut window_start = self.window_start.write().await;
                let now = Instant::now();

                if let Some(start) = *window_start {
                    if now.duration_since(start) > self.config.failure_window {
                        // Window expired, reset
                        self.failure_count.store(0, Ordering::Relaxed);
                        *window_start = Some(now);
                    }
                } else {
                    *window_start = Some(now);
                }
                drop(window_start);

                let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= self.config.failure_threshold {
                    *state = CircuitState::Open;
                    *self.opened_at.write().await = Some(now);
                    warn!("Circuit breaker opened after {} failures", count);
                }
            }
            CircuitState::HalfOpen => {
                // Failure in half-open state, go back to open
                *state = CircuitState::Open;
                *self.opened_at.write().await = Some(Instant::now());
                self.success_count.store(0, Ordering::Relaxed);
                warn!("Circuit breaker re-opened after failure in half-open state");
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }
    }

    /// Reset the circuit breaker
    pub async fn reset(&self) {
        *self.state.write().await = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        *self.opened_at.write().await = None;
        *self.window_start.write().await = None;
        info!("Circuit breaker reset");
    }
}

// ============================================================================
// Retry Functions
// ============================================================================

/// Execute an operation with retry logic
///
/// # Arguments
/// * `config` - Retry configuration
/// * `operation` - Async function to execute
///
/// # Returns
/// Result of the operation, or the last error after all retries exhausted
pub async fn retry_with_backoff<F, Fut, T>(config: &RetryConfig, operation: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    retry_with_backoff_and_metrics(config, None, operation).await
}

/// Execute an operation with retry logic and metrics tracking
///
/// # Arguments
/// * `config` - Retry configuration
/// * `metrics` - Optional metrics tracker
/// * `operation` - Async function to execute
///
/// # Returns
/// Result of the operation, or the last error after all retries exhausted
pub async fn retry_with_backoff_and_metrics<F, Fut, T>(
    config: &RetryConfig,
    metrics: Option<&RetryMetrics>,
    operation: F,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_error: Option<ExecutionError> = None;
    let max_attempts = config.max_attempts.max(1);

    for attempt in 1..=max_attempts {
        // Calculate and apply delay (except for first attempt)
        if attempt > 1 {
            let delay = if let Some(err) = &last_error {
                if err.is_rate_limit() && config.retry_on_rate_limit {
                    config.rate_limit_delay
                } else {
                    config.calculate_delay(attempt - 1)
                }
            } else {
                config.calculate_delay(attempt - 1)
            };

            debug!(
                "Retry attempt {}/{} after {:?}",
                attempt, max_attempts, delay
            );
            sleep(delay).await;
        }

        // Execute operation with optional timeout
        let result = if let Some(timeout) = config.operation_timeout {
            match tokio::time::timeout(timeout, operation()).await {
                Ok(result) => result,
                Err(_) => Err(ExecutionError::Timeout(format!(
                    "Operation timed out after {:?}",
                    timeout
                ))),
            }
        } else {
            operation().await
        };

        match result {
            Ok(value) => {
                if let Some(m) = metrics {
                    m.record_success(attempt);
                }
                if attempt > 1 {
                    info!(
                        "Operation succeeded on attempt {}/{}",
                        attempt, max_attempts
                    );
                }
                return Ok(value);
            }
            Err(err) => {
                if let Some(m) = metrics {
                    m.record_error(&err);
                }

                // Check if error is retryable
                let should_retry =
                    err.is_retryable() || (err.is_rate_limit() && config.retry_on_rate_limit);

                if !should_retry || attempt >= max_attempts {
                    if let Some(m) = metrics {
                        m.record_failure(attempt);
                    }

                    if attempt >= max_attempts {
                        error!("Operation failed after {} attempts: {}", attempt, err);
                    } else {
                        debug!("Non-retryable error: {}", err);
                    }
                    return Err(err);
                }

                warn!(
                    "Attempt {}/{} failed (retrying): {}",
                    attempt, max_attempts, err
                );
                last_error = Some(err);
            }
        }
    }

    // Should not reach here, but handle gracefully
    Err(last_error.unwrap_or_else(|| ExecutionError::Internal("Retry logic error".to_string())))
}

/// Execute an operation with circuit breaker and retry logic
pub async fn retry_with_circuit_breaker<F, Fut, T>(
    config: &RetryConfig,
    circuit_breaker: &CircuitBreaker,
    metrics: Option<&RetryMetrics>,
    operation: F,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    // Check circuit breaker
    if !circuit_breaker.is_allowed().await {
        return Err(ExecutionError::Internal(
            "Circuit breaker is open".to_string(),
        ));
    }

    let result = retry_with_backoff_and_metrics(config, metrics, operation).await;

    // Update circuit breaker based on result
    match &result {
        Ok(_) => circuit_breaker.record_success().await,
        Err(_) => circuit_breaker.record_failure().await,
    }

    result
}

// ============================================================================
// Retry Builder (Fluent API)
// ============================================================================

/// Builder for configuring and executing retryable operations
pub struct RetryBuilder<F, Fut, T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    operation: F,
    config: RetryConfig,
    metrics: Option<Arc<RetryMetrics>>,
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    operation_name: Option<String>,
}

impl<F, Fut, T> RetryBuilder<F, Fut, T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    /// Create a new retry builder
    pub fn new(operation: F) -> Self {
        Self {
            operation,
            config: RetryConfig::default(),
            metrics: None,
            circuit_breaker: None,
            operation_name: None,
        }
    }

    /// Set retry configuration
    pub fn with_config(mut self, config: RetryConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum attempts
    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.config.max_attempts = attempts;
        self
    }

    /// Set initial delay
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.config.initial_delay = delay;
        self
    }

    /// Set max delay
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.config.max_delay = delay;
        self
    }

    /// Set operation timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.operation_timeout = Some(timeout);
        self
    }

    /// Set metrics tracker
    pub fn with_metrics(mut self, metrics: Arc<RetryMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Set circuit breaker
    pub fn with_circuit_breaker(mut self, cb: Arc<CircuitBreaker>) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }

    /// Set operation name (for logging)
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.operation_name = Some(name.into());
        self
    }

    /// Execute the operation with configured retry logic
    pub async fn execute(self) -> Result<T> {
        if let Some(name) = &self.operation_name {
            debug!("Executing operation: {}", name);
        }

        let metrics_ref = self.metrics.as_ref().map(|m| m.as_ref());

        if let Some(cb) = &self.circuit_breaker {
            retry_with_circuit_breaker(&self.config, cb, metrics_ref, self.operation).await
        } else {
            retry_with_backoff_and_metrics(&self.config, metrics_ref, self.operation).await
        }
    }
}

/// Create a retry builder for an operation
pub fn retry<F, Fut, T>(operation: F) -> RetryBuilder<F, Fut, T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    RetryBuilder::new(operation)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert!(config.retry_on_rate_limit);
    }

    #[test]
    fn test_calculate_delay() {
        let config = RetryConfig {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter_factor: 0.0, // No jitter for deterministic test
            ..Default::default()
        };

        assert_eq!(config.calculate_delay(0), Duration::ZERO);
        assert_eq!(config.calculate_delay(1), Duration::from_millis(100));
        assert_eq!(config.calculate_delay(2), Duration::from_millis(200));
        assert_eq!(config.calculate_delay(3), Duration::from_millis(400));
    }

    #[test]
    fn test_calculate_delay_capped() {
        let config = RetryConfig {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 10.0,
            jitter_factor: 0.0,
            ..Default::default()
        };

        // Should be capped at 5 seconds
        assert_eq!(config.calculate_delay(5), Duration::from_secs(5));
    }

    #[test]
    fn test_retry_metrics() {
        let metrics = RetryMetrics::new();

        metrics.record_success(1);
        metrics.record_success(2);
        metrics.record_failure(3);

        assert_eq!(metrics.total_operations.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.first_try_successes.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.retry_successes.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_failures.load(Ordering::Relaxed), 1);

        // 1 retry from second success + 2 retries from failure
        assert_eq!(metrics.total_retries.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_retry_metrics_rates() {
        let metrics = RetryMetrics::new();

        // 80% success rate: 4 successes, 1 failure
        for _ in 0..4 {
            metrics.record_success(1);
        }
        metrics.record_failure(3);

        assert!((metrics.success_rate() - 0.8).abs() < 0.01);
        assert!((metrics.first_try_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_retry_metrics_summary() {
        let metrics = RetryMetrics::new();
        metrics.record_success(1);
        metrics.record_error(&ExecutionError::RateLimitExceeded("test".into()));

        let summary = metrics.summary();
        assert!(summary.contains("Operations: 1"));
        assert!(summary.contains("Rate Limits: 1"));
    }

    #[tokio::test]
    async fn test_circuit_breaker_states() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            open_duration: Duration::from_millis(100),
            success_threshold: 1,
            failure_window: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::new(config);

        assert_eq!(cb.state().await, CircuitState::Closed);
        assert!(cb.is_allowed().await);

        // Record failures to open circuit
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Closed);

        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);
        assert!(!cb.is_allowed().await);

        // Wait for circuit to transition to half-open
        sleep(Duration::from_millis(150)).await;
        assert_eq!(cb.state().await, CircuitState::HalfOpen);
        assert!(cb.is_allowed().await);

        // Success should close circuit
        cb.record_success().await;
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_retry_success_first_try() {
        let config = RetryConfig::default();
        let call_count = Arc::new(AtomicU32::new(0));

        let count = call_count.clone();
        let result = retry_with_backoff(&config, || {
            let c = count.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ExecutionError>(42)
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };
        let call_count = Arc::new(AtomicU32::new(0));

        let count = call_count.clone();
        let result = retry_with_backoff(&config, || {
            let c = count.clone();
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 3 {
                    Err(ExecutionError::Network("test".into()))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_all_failures() {
        let config = RetryConfig {
            max_attempts: 2,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };
        let call_count = Arc::new(AtomicU32::new(0));

        let count = call_count.clone();
        let result: Result<i32> = retry_with_backoff(&config, || {
            let c = count.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(ExecutionError::Network("test".into()))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_non_retryable_error() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };
        let call_count = Arc::new(AtomicU32::new(0));

        let count = call_count.clone();
        let result: Result<i32> = retry_with_backoff(&config, || {
            let c = count.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                // Non-retryable error
                Err(ExecutionError::OrderValidation("invalid".into()))
            }
        })
        .await;

        assert!(result.is_err());
        // Should only try once for non-retryable errors
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_builder_fluent_api() {
        let call_count = Arc::new(AtomicU32::new(0));

        let count = call_count.clone();
        let result = retry(|| {
            let c = count.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ExecutionError>(42)
            }
        })
        .max_attempts(5)
        .initial_delay(Duration::from_millis(10))
        .named("test_operation")
        .execute()
        .await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_with_metrics() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };
        let metrics = RetryMetrics::new();
        let call_count = Arc::new(AtomicU32::new(0));

        // Test successful operation
        let count = call_count.clone();
        let _ = retry_with_backoff_and_metrics(&config, Some(&metrics), || {
            let c = count.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ExecutionError>(42)
            }
        })
        .await;

        assert_eq!(metrics.total_operations.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.first_try_successes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_preset_configs() {
        let aggressive = RetryConfig::aggressive();
        let conservative = RetryConfig::conservative();

        assert!(aggressive.max_attempts < conservative.max_attempts);
        assert!(aggressive.initial_delay < conservative.initial_delay);
    }
}
