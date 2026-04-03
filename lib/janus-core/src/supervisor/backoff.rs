//! Exponential backoff with jitter for supervisor restart strategies.
//!
//! Implements the backoff algorithm described in the Janus Supervisor
//! Architecture Refactor document:
//!
//! ```text
//! delay = min(cap, base * 2^attempt) + jitter
//! ```
//!
//! Where `jitter` is a random value in `[0, base * 2^attempt)` to prevent
//! thundering-herd scenarios when multiple services restart simultaneously
//! after a shared outage.
//!
//! # Features
//!
//! - **Exponential growth** with configurable base and cap
//! - **Full jitter** to desynchronize concurrent restarts
//! - **Cooldown reset**: if a service runs successfully for a configurable
//!   period, the attempt counter resets to zero so occasional rare failures
//!   don't accumulate toward a long backoff
//! - **Circuit breaker**: if a service fails N times within a window T,
//!   the supervisor can escalate (e.g., shut down the node, fire an alert)

use std::time::{Duration, Instant};

use rand::RngExt;

/// Configuration for the exponential backoff strategy.
///
/// All fields have sensible defaults for a production trading system.
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Initial delay before the first retry (default: 100ms).
    pub base_delay: Duration,

    /// Maximum delay between retries (default: 60s).
    pub max_delay: Duration,

    /// If a service runs successfully for at least this long, the attempt
    /// counter resets to zero (default: 5 minutes).
    pub cooldown_period: Duration,

    /// Maximum consecutive failures before the circuit breaker trips.
    /// Set to `0` to disable the circuit breaker (default: 10).
    pub max_retries: u32,

    /// Window within which `max_retries` failures trigger the circuit
    /// breaker (default: 10 minutes). Only relevant when `max_retries > 0`.
    pub circuit_breaker_window: Duration,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            cooldown_period: Duration::from_secs(300), // 5 minutes
            max_retries: 10,
            circuit_breaker_window: Duration::from_secs(600), // 10 minutes
        }
    }
}

impl BackoffConfig {
    /// Create a new config with the given base and max delays.
    pub fn new(base_delay: Duration, max_delay: Duration) -> Self {
        Self {
            base_delay,
            max_delay,
            ..Default::default()
        }
    }

    /// Builder: set the cooldown period.
    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown_period = cooldown;
        self
    }

    /// Builder: set the circuit breaker parameters.
    pub fn with_circuit_breaker(mut self, max_retries: u32, window: Duration) -> Self {
        self.max_retries = max_retries;
        self.circuit_breaker_window = window;
        self
    }

    /// Builder: disable the circuit breaker entirely.
    pub fn without_circuit_breaker(mut self) -> Self {
        self.max_retries = 0;
        self
    }
}

/// Result of computing the next backoff delay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackoffAction {
    /// Wait for the given duration and then retry.
    Retry(Duration),

    /// The circuit breaker has tripped — too many failures within the
    /// configured window. The supervisor should escalate (alert, shut
    /// down, etc.) rather than retry.
    CircuitOpen {
        /// Total failures observed within the window.
        failures: u32,
        /// The configured maximum before tripping.
        max_retries: u32,
    },
}

/// Mutable state tracker for an individual service's backoff history.
///
/// One `BackoffState` instance is maintained per supervised service inside
/// the [`JanusSupervisor`](super::JanusSupervisor).
#[derive(Debug, Clone)]
pub struct BackoffState {
    config: BackoffConfig,

    /// Number of consecutive failures (resets on cooldown).
    attempt: u32,

    /// Timestamps of recent failures (within the circuit-breaker window).
    /// Older entries are pruned on each call to [`next_backoff`].
    failure_timestamps: Vec<Instant>,

    /// When the service last started successfully. Used for cooldown logic.
    last_start: Option<Instant>,
}

impl BackoffState {
    /// Create a fresh backoff state with the given configuration.
    pub fn new(config: BackoffConfig) -> Self {
        Self {
            config,
            attempt: 0,
            failure_timestamps: Vec::new(),
            last_start: None,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BackoffConfig::default())
    }

    /// Record that the service has started (or restarted) successfully.
    ///
    /// Called by the supervisor when a service's `run()` method is entered.
    pub fn record_start(&mut self) {
        self.last_start = Some(Instant::now());
    }

    /// Check if the cooldown period has elapsed since the last start,
    /// and if so, reset the attempt counter.
    ///
    /// Called by the supervisor when a service exits so that a service
    /// which ran healthily for a long time doesn't carry old failure
    /// history into its next restart.
    pub fn maybe_reset_on_cooldown(&mut self) {
        if let Some(start) = self.last_start
            && start.elapsed() >= self.config.cooldown_period
        {
            tracing::info!(
                cooldown_secs = self.config.cooldown_period.as_secs(),
                "Service ran for longer than cooldown period, resetting backoff"
            );
            self.attempt = 0;
            self.failure_timestamps.clear();
        }
    }

    /// Record a failure and compute the next action.
    ///
    /// Returns [`BackoffAction::Retry`] with the jittered delay, or
    /// [`BackoffAction::CircuitOpen`] if the circuit breaker has tripped.
    pub fn next_backoff(&mut self) -> BackoffAction {
        let now = Instant::now();

        // Record this failure
        self.failure_timestamps.push(now);
        self.attempt = self.attempt.saturating_add(1);

        // Prune old failure timestamps outside the circuit-breaker window
        if self.config.max_retries > 0 {
            let window_start = now - self.config.circuit_breaker_window;
            self.failure_timestamps.retain(|ts| *ts >= window_start);

            // Check circuit breaker
            if self.failure_timestamps.len() as u32 >= self.config.max_retries {
                return BackoffAction::CircuitOpen {
                    failures: self.failure_timestamps.len() as u32,
                    max_retries: self.config.max_retries,
                };
            }
        }

        // Compute exponential delay: base * 2^attempt, capped at max_delay
        let exp_delay = self.compute_exponential_delay();

        // Add full jitter: uniform random in [0, exp_delay)
        let jittered = self.add_jitter(exp_delay);

        BackoffAction::Retry(jittered)
    }

    /// Reset all state (attempt counter, failure history, start time).
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.failure_timestamps.clear();
        self.last_start = None;
    }

    /// Current attempt number (consecutive failures since last reset).
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Number of failures recorded within the circuit-breaker window.
    pub fn recent_failures(&self) -> usize {
        self.failure_timestamps.len()
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Compute `min(cap, base * 2^attempt)` without overflow.
    fn compute_exponential_delay(&self) -> Duration {
        let base_ms = self.config.base_delay.as_millis() as u64;
        let max_ms = self.config.max_delay.as_millis() as u64;

        // Prevent overflow: if attempt >= 63 the shift would overflow u64,
        // so we clamp early.
        let shift = self.attempt.min(62) as u64;

        // Saturating multiplication: base_ms * 2^shift, capped at max_ms
        let exp_ms = base_ms.saturating_mul(1u64.checked_shl(shift as u32).unwrap_or(u64::MAX));
        let capped_ms = exp_ms.min(max_ms);

        Duration::from_millis(capped_ms)
    }

    /// Add full jitter: returns a duration in `[0, delay + delay_jitter_range)`.
    ///
    /// Uses "full jitter" strategy as recommended in the architecture doc
    /// to desynchronize concurrent restarts after a shared outage.
    fn add_jitter(&self, base: Duration) -> Duration {
        if base.is_zero() {
            return base;
        }

        let base_ms = base.as_millis() as u64;
        let mut rng = rand::rng();

        // Full jitter: pick a random value in [0, base_ms]
        let jitter_ms = rng.random_range(0..=base_ms);

        // The actual delay is a random value in [0, 2*base_ms] effectively,
        // but we use the "full jitter" approach from AWS:
        // delay = random_between(0, min(cap, base * 2^attempt))
        // This gives better spread than "equal jitter" or "decorrelated jitter"
        // for our use case of independent services.
        Duration::from_millis(jitter_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = BackoffConfig::default();
        assert_eq!(cfg.base_delay, Duration::from_millis(100));
        assert_eq!(cfg.max_delay, Duration::from_secs(60));
        assert_eq!(cfg.cooldown_period, Duration::from_secs(300));
        assert_eq!(cfg.max_retries, 10);
        assert_eq!(cfg.circuit_breaker_window, Duration::from_secs(600));
    }

    #[test]
    fn test_builder_pattern() {
        let cfg = BackoffConfig::new(Duration::from_millis(200), Duration::from_secs(30))
            .with_cooldown(Duration::from_secs(120))
            .with_circuit_breaker(5, Duration::from_secs(60));

        assert_eq!(cfg.base_delay, Duration::from_millis(200));
        assert_eq!(cfg.max_delay, Duration::from_secs(30));
        assert_eq!(cfg.cooldown_period, Duration::from_secs(120));
        assert_eq!(cfg.max_retries, 5);
        assert_eq!(cfg.circuit_breaker_window, Duration::from_secs(60));
    }

    #[test]
    fn test_without_circuit_breaker() {
        let cfg = BackoffConfig::default().without_circuit_breaker();
        assert_eq!(cfg.max_retries, 0);
    }

    #[test]
    fn test_exponential_growth_is_capped() {
        let cfg = BackoffConfig::new(Duration::from_millis(100), Duration::from_secs(5))
            .without_circuit_breaker();

        let state = BackoffState::new(cfg.clone());

        // Attempt 0 => base * 2^0 = 100ms (before jitter)
        // Attempt 5 => base * 2^5 = 3200ms
        // Attempt 10 => base * 2^10 = 102400ms > 5000ms cap => 5000ms

        // Verify the internal exponential calculation (no jitter)
        let mut s = state.clone();
        s.attempt = 0;
        let d = s.compute_exponential_delay();
        assert_eq!(d, Duration::from_millis(100));

        s.attempt = 5;
        let d = s.compute_exponential_delay();
        assert_eq!(d, Duration::from_millis(3200));

        s.attempt = 10;
        let d = s.compute_exponential_delay();
        assert_eq!(d, Duration::from_secs(5)); // capped
    }

    #[test]
    fn test_exponential_no_overflow_at_high_attempts() {
        let cfg = BackoffConfig::default().without_circuit_breaker();
        let mut state = BackoffState::new(cfg);
        state.attempt = 100; // Would overflow without clamping
        let d = state.compute_exponential_delay();
        assert_eq!(d, Duration::from_secs(60)); // Should be capped at max_delay
    }

    #[test]
    fn test_jitter_stays_within_bounds() {
        let cfg = BackoffConfig::new(Duration::from_millis(100), Duration::from_secs(60))
            .without_circuit_breaker();

        let state = BackoffState::new(cfg);

        // Run many iterations to statistically verify jitter bounds
        for _ in 0..1000 {
            let jittered = state.add_jitter(Duration::from_millis(1000));
            assert!(jittered <= Duration::from_millis(1000));
        }
    }

    #[test]
    fn test_jitter_zero_base() {
        let cfg = BackoffConfig::default();
        let state = BackoffState::new(cfg);
        let jittered = state.add_jitter(Duration::ZERO);
        assert_eq!(jittered, Duration::ZERO);
    }

    #[test]
    fn test_next_backoff_increments_attempt() {
        let cfg = BackoffConfig::default().without_circuit_breaker();
        let mut state = BackoffState::new(cfg);

        assert_eq!(state.attempt(), 0);

        let action = state.next_backoff();
        assert!(matches!(action, BackoffAction::Retry(_)));
        assert_eq!(state.attempt(), 1);

        let action = state.next_backoff();
        assert!(matches!(action, BackoffAction::Retry(_)));
        assert_eq!(state.attempt(), 2);
    }

    #[test]
    fn test_circuit_breaker_trips() {
        let cfg = BackoffConfig::default().with_circuit_breaker(3, Duration::from_secs(600));

        let mut state = BackoffState::new(cfg);

        // First two failures should retry
        assert!(matches!(state.next_backoff(), BackoffAction::Retry(_)));
        assert!(matches!(state.next_backoff(), BackoffAction::Retry(_)));

        // Third failure should trip the circuit breaker
        let action = state.next_backoff();
        assert!(matches!(
            action,
            BackoffAction::CircuitOpen {
                failures: 3,
                max_retries: 3
            }
        ));
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = BackoffConfig::default().without_circuit_breaker();
        let mut state = BackoffState::new(cfg);

        state.next_backoff();
        state.next_backoff();
        assert_eq!(state.attempt(), 2);
        assert_eq!(state.recent_failures(), 2);

        state.reset();
        assert_eq!(state.attempt(), 0);
        assert_eq!(state.recent_failures(), 0);
    }

    #[test]
    fn test_cooldown_resets_attempt() {
        let cfg = BackoffConfig::default()
            .with_cooldown(Duration::from_millis(50))
            .without_circuit_breaker();

        let mut state = BackoffState::new(cfg);

        // Simulate some failures
        state.next_backoff();
        state.next_backoff();
        assert_eq!(state.attempt(), 2);

        // Record start and sleep past cooldown
        state.record_start();
        // We can't sleep in a non-async test, but we can manipulate last_start
        state.last_start = Some(Instant::now() - Duration::from_millis(100));

        state.maybe_reset_on_cooldown();
        assert_eq!(state.attempt(), 0);
        assert_eq!(state.recent_failures(), 0);
    }

    #[test]
    fn test_cooldown_does_not_reset_if_too_early() {
        let cfg = BackoffConfig::default()
            .with_cooldown(Duration::from_secs(300))
            .without_circuit_breaker();

        let mut state = BackoffState::new(cfg);

        state.next_backoff();
        state.next_backoff();
        state.record_start(); // Just started — nowhere near 5 min

        state.maybe_reset_on_cooldown();
        assert_eq!(state.attempt(), 2); // Not reset
    }

    #[test]
    fn test_backoff_delay_increases_monotonically_ignoring_jitter() {
        let cfg = BackoffConfig::new(Duration::from_millis(100), Duration::from_secs(60))
            .without_circuit_breaker();

        let mut state = BackoffState::new(cfg);

        // Verify the exponential base (before jitter) grows
        let delays: Vec<Duration> = (0..8)
            .map(|_| {
                let _ = state.next_backoff();
                state.compute_exponential_delay()
            })
            .collect();

        // Each delay should be >= the previous one (monotonic)
        for window in delays.windows(2) {
            assert!(window[1] >= window[0], "{:?} < {:?}", window[1], window[0]);
        }
    }

    #[test]
    fn test_attempt_saturates() {
        let cfg = BackoffConfig::default().without_circuit_breaker();
        let mut state = BackoffState::new(cfg);
        state.attempt = u32::MAX;

        let action = state.next_backoff();
        assert!(matches!(action, BackoffAction::Retry(_)));
        assert_eq!(state.attempt(), u32::MAX); // Saturated, didn't wrap
    }
}
