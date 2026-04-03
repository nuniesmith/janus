//! Error recovery and circuit breaker patterns for production resilience.
//!
//! This module provides fault tolerance mechanisms:
//! - Circuit breaker pattern for cascading failure prevention
//! - Automatic retry logic with exponential backoff
//! - Error rate tracking and alerting
//! - Graceful degradation strategies
//! - Fallback mechanisms

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

impl CircuitState {
    /// Check if the circuit allows requests.
    pub fn allows_requests(&self) -> bool {
        matches!(self, CircuitState::Closed | CircuitState::HalfOpen)
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &str {
        match self {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half_open",
        }
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening circuit
    pub failure_threshold: usize,
    /// Success threshold to close circuit from half-open
    pub success_threshold: usize,
    /// Timeout before attempting to close circuit
    pub timeout: Duration,
    /// Window size for tracking failures
    pub window_size: usize,
}

impl CircuitBreakerConfig {
    /// Create default configuration.
    pub fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            window_size: 10,
        }
    }

    /// Create strict configuration (fails faster).
    pub fn strict() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            window_size: 10,
        }
    }

    /// Create lenient configuration (more tolerant).
    pub fn lenient() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 2,
            timeout: Duration::from_secs(120),
            window_size: 20,
        }
    }
}

/// Circuit breaker statistics.
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub total_calls: usize,
    pub successful_calls: usize,
    pub failed_calls: usize,
    pub rejected_calls: usize,
    pub state_changes: usize,
    pub last_failure: Option<Instant>,
}

impl CircuitStats {
    fn new() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            rejected_calls: 0,
            state_changes: 0,
            last_failure: None,
        }
    }

    /// Get failure rate.
    pub fn failure_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.failed_calls as f64 / self.total_calls as f64
        }
    }

    /// Get success rate.
    pub fn success_rate(&self) -> f64 {
        1.0 - self.failure_rate()
    }
}

/// Circuit breaker for preventing cascading failures.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    stats: Arc<RwLock<CircuitStats>>,
    failure_window: Arc<RwLock<Vec<bool>>>,
    success_count: Arc<RwLock<usize>>,
    last_state_change: Arc<RwLock<Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            stats: Arc::new(RwLock::new(CircuitStats::new())),
            failure_window: Arc::new(RwLock::new(Vec::new())),
            success_count: Arc::new(RwLock::new(0)),
            last_state_change: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Create a circuit breaker with default configuration.
    pub fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /// Execute a call through the circuit breaker.
    pub fn call<F, T, E>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        // Check if circuit allows requests
        if !self.should_allow_request() {
            self.record_rejected();
            return Err(CircuitBreakerError::CircuitOpen);
        }

        // Execute the function
        let result = f();

        // Record the result
        match result {
            Ok(value) => {
                self.record_success();
                Ok(value)
            }
            Err(err) => {
                self.record_failure();
                Err(CircuitBreakerError::CallFailed(err))
            }
        }
    }

    /// Check if the circuit should allow a request.
    fn should_allow_request(&self) -> bool {
        let state = self
            .state
            .read()
            .ok()
            .map(|s| *s)
            .unwrap_or(CircuitState::Open);

        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                let last_change = self
                    .last_state_change
                    .read()
                    .ok()
                    .map(|t| *t)
                    .unwrap_or_else(Instant::now);
                if last_change.elapsed() >= self.config.timeout {
                    self.transition_to(CircuitState::HalfOpen);
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful call.
    fn record_success(&self) {
        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.total_calls += 1;
            stats.successful_calls += 1;
        }

        // Update failure window
        if let Ok(mut window) = self.failure_window.write() {
            window.push(false);
            if window.len() > self.config.window_size {
                window.remove(0);
            }
        }

        // Handle state transitions
        let state = self
            .state
            .read()
            .ok()
            .map(|s| *s)
            .unwrap_or(CircuitState::Closed);
        if state == CircuitState::HalfOpen {
            if let Ok(mut count) = self.success_count.write() {
                *count += 1;
                if *count >= self.config.success_threshold {
                    self.transition_to(CircuitState::Closed);
                    *count = 0;
                }
            }
        }
    }

    /// Record a failed call.
    fn record_failure(&self) {
        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.total_calls += 1;
            stats.failed_calls += 1;
            stats.last_failure = Some(Instant::now());
        }

        // Update failure window
        if let Ok(mut window) = self.failure_window.write() {
            window.push(true);
            if window.len() > self.config.window_size {
                window.remove(0);
            }
        }

        // Reset success count in half-open state
        if let Ok(mut count) = self.success_count.write() {
            *count = 0;
        }

        // Check if should open circuit
        self.check_failure_threshold();
    }

    /// Record a rejected call.
    fn record_rejected(&self) {
        if let Ok(mut stats) = self.stats.write() {
            stats.rejected_calls += 1;
        }
    }

    /// Check if failure threshold is exceeded.
    fn check_failure_threshold(&self) {
        if let Ok(window) = self.failure_window.read() {
            let failure_count = window.iter().filter(|&&failed| failed).count();
            if failure_count >= self.config.failure_threshold {
                self.transition_to(CircuitState::Open);
            }
        }
    }

    /// Transition to a new state.
    fn transition_to(&self, new_state: CircuitState) {
        if let Ok(mut state) = self.state.write() {
            if *state != new_state {
                *state = new_state;

                if let Ok(mut stats) = self.stats.write() {
                    stats.state_changes += 1;
                }

                if let Ok(mut last_change) = self.last_state_change.write() {
                    *last_change = Instant::now();
                }

                // Clear failure window on close
                if new_state == CircuitState::Closed {
                    if let Ok(mut window) = self.failure_window.write() {
                        window.clear();
                    }
                }
            }
        }
    }

    /// Get current circuit state.
    pub fn state(&self) -> CircuitState {
        self.state
            .read()
            .ok()
            .map(|s| *s)
            .unwrap_or(CircuitState::Open)
    }

    /// Get circuit statistics.
    pub fn stats(&self) -> CircuitStats {
        self.stats
            .read()
            .ok()
            .map(|s| s.clone())
            .unwrap_or_else(CircuitStats::new)
    }

    /// Reset the circuit breaker.
    pub fn reset(&self) {
        self.transition_to(CircuitState::Closed);
        if let Ok(mut stats) = self.stats.write() {
            *stats = CircuitStats::new();
        }
        if let Ok(mut window) = self.failure_window.write() {
            window.clear();
        }
        if let Ok(mut count) = self.success_count.write() {
            *count = 0;
        }
    }

    /// Force open the circuit.
    pub fn force_open(&self) {
        self.transition_to(CircuitState::Open);
    }

    /// Force close the circuit.
    pub fn force_close(&self) {
        self.transition_to(CircuitState::Closed);
    }
}

/// Circuit breaker error.
#[derive(Debug)]
pub enum CircuitBreakerError<E> {
    /// Circuit is open, call was rejected
    CircuitOpen,
    /// Call failed with the given error
    CallFailed(E),
}

/// Retry configuration with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub multiplier: f64,
}

impl RetryConfig {
    /// Create default retry configuration.
    pub fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
        }
    }

    /// Create aggressive retry configuration (more attempts).
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(10),
            multiplier: 1.5,
        }
    }

    /// Create conservative retry configuration (fewer attempts).
    pub fn conservative() -> Self {
        Self {
            max_attempts: 2,
            initial_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(60),
            multiplier: 3.0,
        }
    }

    /// Calculate delay for a given attempt.
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            return Duration::from_secs(0);
        }

        let delay_ms =
            self.initial_delay.as_millis() as f64 * self.multiplier.powi(attempt as i32 - 1);
        let delay = Duration::from_millis(delay_ms as u64);
        delay.min(self.max_delay)
    }
}

/// Retry executor with exponential backoff.
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    /// Create a new retry executor.
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create a retry executor with default configuration.
    pub fn default() -> Self {
        Self::new(RetryConfig::default())
    }

    /// Execute a function with retries.
    pub fn execute<F, T, E>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
    {
        let mut attempt = 0;
        loop {
            match f() {
                Ok(result) => return Ok(result),
                Err(err) => {
                    attempt += 1;
                    if attempt >= self.config.max_attempts {
                        return Err(err);
                    }

                    let delay = self.config.delay_for_attempt(attempt);
                    std::thread::sleep(delay);
                }
            }
        }
    }

    /// Execute with a predicate to determine if retry should happen.
    pub fn execute_if<F, T, E, P>(&self, mut f: F, mut should_retry: P) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
        P: FnMut(&E) -> bool,
    {
        let mut attempt = 0;
        loop {
            match f() {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if !should_retry(&err) || attempt >= self.config.max_attempts - 1 {
                        return Err(err);
                    }

                    attempt += 1;
                    let delay = self.config.delay_for_attempt(attempt);
                    std::thread::sleep(delay);
                }
            }
        }
    }
}

/// Error rate tracker for monitoring system health.
pub struct ErrorRateTracker {
    window: Arc<RwLock<Vec<(Instant, bool)>>>,
    window_duration: Duration,
}

impl ErrorRateTracker {
    /// Create a new error rate tracker.
    pub fn new(window_duration: Duration) -> Self {
        Self {
            window: Arc::new(RwLock::new(Vec::new())),
            window_duration,
        }
    }

    /// Record a success.
    pub fn record_success(&self) {
        self.record(false);
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.record(true);
    }

    /// Record an event.
    fn record(&self, is_error: bool) {
        if let Ok(mut window) = self.window.write() {
            window.push((Instant::now(), is_error));
            self.cleanup_old_entries(&mut window);
        }
    }

    /// Remove entries outside the time window.
    fn cleanup_old_entries(&self, window: &mut Vec<(Instant, bool)>) {
        let cutoff = Instant::now() - self.window_duration;
        window.retain(|(timestamp, _)| *timestamp > cutoff);
    }

    /// Get current error rate.
    pub fn error_rate(&self) -> f64 {
        if let Ok(mut window) = self.window.write() {
            self.cleanup_old_entries(&mut window);

            if window.is_empty() {
                return 0.0;
            }

            let error_count = window.iter().filter(|(_, is_error)| *is_error).count();
            error_count as f64 / window.len() as f64
        } else {
            0.0
        }
    }

    /// Check if error rate exceeds threshold.
    pub fn exceeds_threshold(&self, threshold: f64) -> bool {
        self.error_rate() > threshold
    }

    /// Get total event count in window.
    pub fn total_events(&self) -> usize {
        if let Ok(window) = self.window.read() {
            window.len()
        } else {
            0
        }
    }

    /// Get error count in window.
    pub fn error_count(&self) -> usize {
        if let Ok(window) = self.window.read() {
            window.iter().filter(|(_, is_error)| *is_error).count()
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed() {
        let breaker = CircuitBreaker::default();
        assert_eq!(breaker.state(), CircuitState::Closed);

        let result = breaker.call(|| Ok::<_, ()>(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            window_size: 10,
        };
        let breaker = CircuitBreaker::new(config);

        // Trigger failures
        for _ in 0..3 {
            let _ = breaker.call(|| Err::<(), _>("error"));
        }

        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_rejects_when_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            window_size: 10,
        };
        let breaker = CircuitBreaker::new(config);

        // Open circuit
        for _ in 0..2 {
            let _ = breaker.call(|| Err::<(), _>("error"));
        }

        // Should reject
        let result = breaker.call(|| Ok::<_, ()>(42));
        assert!(matches!(result, Err(CircuitBreakerError::CircuitOpen)));

        let stats = breaker.stats();
        assert!(stats.rejected_calls > 0);
    }

    #[test]
    fn test_circuit_breaker_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout: Duration::from_millis(50),
            window_size: 10,
        };
        let breaker = CircuitBreaker::new(config);

        // Open circuit
        for _ in 0..2 {
            let _ = breaker.call(|| Err::<(), _>("error"));
        }
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(60));

        // Next call should transition to half-open
        let _ = breaker.call(|| Ok::<_, ()>(42));
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_retry_executor() {
        let executor = RetryExecutor::new(RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            multiplier: 2.0,
        });

        let mut attempt = 0;
        let result = executor.execute(|| {
            attempt += 1;
            if attempt < 3 { Err("not yet") } else { Ok(42) }
        });

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt, 3);
    }

    #[test]
    fn test_retry_executor_gives_up() {
        let executor = RetryExecutor::new(RetryConfig {
            max_attempts: 2,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            multiplier: 2.0,
        });

        let result = executor.execute(|| Err::<(), _>("always fails"));
        assert!(result.is_err());
    }

    #[test]
    fn test_retry_delay_calculation() {
        let config = RetryConfig::default();

        let delay0 = config.delay_for_attempt(0);
        assert_eq!(delay0, Duration::from_secs(0));

        let delay1 = config.delay_for_attempt(1);
        assert_eq!(delay1, Duration::from_millis(100));

        let delay2 = config.delay_for_attempt(2);
        assert_eq!(delay2, Duration::from_millis(200));
    }

    #[test]
    fn test_error_rate_tracker() {
        let tracker = ErrorRateTracker::new(Duration::from_secs(60));

        tracker.record_success();
        tracker.record_success();
        tracker.record_error();

        assert_eq!(tracker.total_events(), 3);
        assert_eq!(tracker.error_count(), 1);
        assert!((tracker.error_rate() - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_error_rate_threshold() {
        let tracker = ErrorRateTracker::new(Duration::from_secs(60));

        tracker.record_success();
        tracker.record_error();
        tracker.record_error();

        assert!(tracker.exceeds_threshold(0.5));
        assert!(!tracker.exceeds_threshold(0.8));
    }

    #[test]
    fn test_circuit_breaker_stats() {
        let breaker = CircuitBreaker::default();

        let _ = breaker.call(|| Ok::<_, ()>(1));
        let _ = breaker.call(|| Err::<(), _>("error"));

        let stats = breaker.stats();
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.successful_calls, 1);
        assert_eq!(stats.failed_calls, 1);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let breaker = CircuitBreaker::default();

        for _ in 0..5 {
            let _ = breaker.call(|| Err::<(), _>("error"));
        }

        breaker.reset();
        assert_eq!(breaker.state(), CircuitState::Closed);

        let stats = breaker.stats();
        assert_eq!(stats.total_calls, 0);
    }

    #[test]
    fn test_circuit_state_allows_requests() {
        assert!(CircuitState::Closed.allows_requests());
        assert!(CircuitState::HalfOpen.allows_requests());
        assert!(!CircuitState::Open.allows_requests());
    }
}
