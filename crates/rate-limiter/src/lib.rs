//! # Rate Limiter Spike Prototype
//!
//! Comprehensive implementation of token bucket and sliding window rate limiters
//! designed for high-frequency crypto exchange API interactions.
//!
//! ## Key Features:
//! - Token bucket with configurable refill rates
//! - Sliding window for burst handling (Binance-style)
//! - Dynamic limit adjustment from response headers
//! - Thread-safe for concurrent actor usage
//! - Detailed metrics and observability
//!
//! ## Architecture:
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Rate Limiter Manager                     │
//! │  (Handles multiple exchanges with different algorithms)     │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!           ┌──────────────────┼──────────────────┐
//!           │                  │                  │
//!    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
//!    │  Binance    │   │   Bybit     │   │  Kucoin     │
//!    │ TokenBucket │   │ TokenBucket │   │ TokenBucket │
//!    │ + Sliding   │   │ (Per-second)│   │ (Per-min)   │
//!    └─────────────┘   └─────────────┘   └─────────────┘
//! ```

use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, instrument, warn};

// ============================================================================
// Modules
// ============================================================================

/// Circuit breaker for preventing cascading failures (P0 Item 3)
pub mod circuit_breaker;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug, Clone)]
pub enum RateLimitError {
    #[error("Rate limit exceeded. Retry after {retry_after:?}")]
    Exceeded { retry_after: Duration },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Capacity exhausted: {0}")]
    CapacityExhausted(String),
}

pub type Result<T> = std::result::Result<T, RateLimitError>;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a token bucket rate limiter
#[derive(Debug, Clone)]
pub struct TokenBucketConfig {
    /// Maximum tokens the bucket can hold (burst capacity)
    pub capacity: u32,

    /// Tokens added per second
    pub refill_rate: f64,

    /// Enable sliding window algorithm for precise burst control
    pub sliding_window: bool,

    /// Sliding window duration (typically 1 minute for Binance)
    pub window_duration: Duration,

    /// Safety margin (0.0 - 1.0). At 0.8, will throttle at 80% capacity
    pub safety_margin: f64,
}

impl Default for TokenBucketConfig {
    fn default() -> Self {
        Self {
            capacity: 6000,
            refill_rate: 100.0, // 6000 per minute = 100 per second
            sliding_window: true,
            window_duration: Duration::from_secs(60),
            safety_margin: 0.8,
        }
    }
}

impl TokenBucketConfig {
    /// Binance Spot API configuration (6000 weight/min)
    pub fn binance_spot() -> Self {
        Self {
            capacity: 6000,
            refill_rate: 100.0,
            sliding_window: true,
            window_duration: Duration::from_secs(60),
            safety_margin: 0.8,
        }
    }

    /// Bybit V5 API configuration (120 requests/second)
    pub fn bybit_v5() -> Self {
        Self {
            capacity: 120,
            refill_rate: 120.0,
            sliding_window: false, // Bybit uses simple per-second limits
            window_duration: Duration::from_secs(1),
            safety_margin: 0.85,
        }
    }

    /// Kucoin API configuration (varies by tier, conservative default)
    pub fn kucoin_public() -> Self {
        Self {
            capacity: 100,
            refill_rate: 1.67, // 100 requests per minute
            sliding_window: true,
            window_duration: Duration::from_secs(60),
            safety_margin: 0.75, // More conservative due to instability
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.capacity == 0 {
            return Err(RateLimitError::InvalidConfig("capacity must be > 0".into()));
        }
        if self.refill_rate <= 0.0 {
            return Err(RateLimitError::InvalidConfig(
                "refill_rate must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.safety_margin) {
            return Err(RateLimitError::InvalidConfig(
                "safety_margin must be between 0 and 1".into(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Token Bucket Implementation
// ============================================================================

/// Thread-safe token bucket rate limiter with sliding window support
pub struct TokenBucket {
    config: TokenBucketConfig,
    state: Arc<Mutex<BucketState>>,
    metrics: Arc<RwLock<Metrics>>,
}

struct BucketState {
    tokens: f64,
    last_refill: Instant,
    /// Sliding window: tracks (timestamp, weight) of recent requests
    request_history: VecDeque<(Instant, u32)>,
}

#[derive(Debug, Default, Clone)]
pub struct Metrics {
    pub total_requests: u64,
    pub accepted_requests: u64,
    pub rejected_requests: u64,
    pub total_wait_time_ms: u64,
    pub current_tokens: f64,
    pub window_usage: u32,
}

impl TokenBucket {
    /// Create a new token bucket with the given configuration
    pub fn new(config: TokenBucketConfig) -> Result<Self> {
        config.validate()?;

        let initial_tokens = config.capacity as f64;
        Ok(Self {
            config,
            state: Arc::new(Mutex::new(BucketState {
                tokens: initial_tokens,
                last_refill: Instant::now(),
                request_history: VecDeque::new(),
            })),
            metrics: Arc::new(RwLock::new(Metrics {
                current_tokens: initial_tokens,
                ..Default::default()
            })),
        })
    }

    /// Attempt to acquire tokens. Returns immediately if available,
    /// otherwise returns the duration to wait.
    #[instrument(skip(self), fields(weight = weight))]
    pub fn acquire(&self, weight: u32) -> Result<()> {
        let mut state = self.state.lock();
        let mut metrics = self.metrics.write();

        metrics.total_requests += 1;

        // Refill tokens based on elapsed time
        self.refill_tokens(&mut state);

        // If sliding window is enabled, check historical usage
        if self.config.sliding_window {
            self.cleanup_old_requests(&mut state);
            let window_usage = self.calculate_window_usage(&state);
            metrics.window_usage = window_usage;

            // Check if adding this request would exceed window capacity
            let effective_capacity =
                (self.config.capacity as f64 * self.config.safety_margin) as u32;
            if window_usage + weight > effective_capacity {
                metrics.rejected_requests += 1;
                let retry_after = self.calculate_retry_time(&state, weight);
                debug!(
                    window_usage = window_usage,
                    weight = weight,
                    capacity = effective_capacity,
                    retry_after_ms = retry_after.as_millis(),
                    "Rate limit exceeded (sliding window)"
                );
                return Err(RateLimitError::Exceeded { retry_after });
            }
        }

        // Check if we have enough tokens in the bucket (with safety margin applied)
        // Safety margin limits the effective capacity, not the remaining tokens
        let effective_capacity = self.config.capacity as f64 * self.config.safety_margin;
        let used_tokens = self.config.capacity as f64 - state.tokens;

        if used_tokens + weight as f64 > effective_capacity {
            metrics.rejected_requests += 1;
            let available = effective_capacity - used_tokens;
            let deficit = weight as f64 - available;
            let retry_after = Duration::from_secs_f64(deficit / self.config.refill_rate);
            debug!(
                current_tokens = state.tokens,
                used_tokens = used_tokens,
                effective_capacity = effective_capacity,
                weight = weight,
                retry_after_ms = retry_after.as_millis(),
                "Rate limit exceeded (token bucket)"
            );
            return Err(RateLimitError::Exceeded { retry_after });
        }

        // Deduct tokens and record request
        state.tokens -= weight as f64;
        metrics.accepted_requests += 1;
        metrics.current_tokens = state.tokens;

        if self.config.sliding_window {
            state.request_history.push_back((Instant::now(), weight));
        }

        debug!(
            remaining_tokens = state.tokens,
            weight = weight,
            "Request accepted"
        );

        Ok(())
    }

    /// Asynchronous version that waits if tokens are not available
    pub async fn acquire_async(&self, weight: u32) -> Result<()> {
        loop {
            match self.acquire(weight) {
                Ok(()) => return Ok(()),
                Err(RateLimitError::Exceeded { retry_after }) => {
                    warn!(
                        retry_after_ms = retry_after.as_millis(),
                        "Rate limited, waiting..."
                    );

                    // Explicit block scope to ensure guard is dropped before await
                    {
                        let mut metrics = self.metrics.write();
                        metrics.total_wait_time_ms += retry_after.as_millis() as u64;
                    }

                    tokio::time::sleep(retry_after).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Update limits dynamically based on API response headers
    ///
    /// # Example (Binance)
    /// ```ignore
    /// X-MBX-USED-WEIGHT-1M: 1500
    /// X-MBX-ORDER-COUNT-10S: 5
    /// ```
    pub fn update_from_headers(&self, used_weight: u32, limit: u32) {
        let mut state = self.state.lock();
        let mut metrics = self.metrics.write();

        // Calculate remaining capacity
        let remaining = limit.saturating_sub(used_weight);

        // Update tokens to match server's view
        state.tokens = remaining as f64;
        metrics.current_tokens = remaining as f64;

        debug!(
            used_weight = used_weight,
            limit = limit,
            remaining = remaining,
            "Updated limits from API response"
        );

        // If we're above 90% usage, trigger self-preservation mode
        let usage_ratio = used_weight as f64 / limit as f64;
        if usage_ratio > 0.9 {
            warn!(
                usage_percent = (usage_ratio * 100.0) as u32,
                "High rate limit usage detected"
            );
        }
    }

    /// Get current metrics (for observability)
    pub fn metrics(&self) -> Metrics {
        self.metrics.read().clone()
    }

    /// Reset the rate limiter (useful for testing)
    #[cfg(test)]
    pub fn reset(&self) {
        let mut state = self.state.lock();
        state.tokens = self.config.capacity as f64;
        state.last_refill = Instant::now();
        state.request_history.clear();

        let mut metrics = self.metrics.write();
        *metrics = Metrics {
            current_tokens: self.config.capacity as f64,
            ..Default::default()
        };
    }

    // ------------------------------------------------------------------------
    // Private Helper Methods
    // ------------------------------------------------------------------------

    fn refill_tokens(&self, state: &mut BucketState) {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill);
        let tokens_to_add = elapsed.as_secs_f64() * self.config.refill_rate;

        state.tokens = (state.tokens + tokens_to_add).min(self.config.capacity as f64);
        state.last_refill = now;
    }

    fn cleanup_old_requests(&self, state: &mut BucketState) {
        let cutoff = Instant::now() - self.config.window_duration;
        while let Some((timestamp, _)) = state.request_history.front() {
            if *timestamp < cutoff {
                state.request_history.pop_front();
            } else {
                break;
            }
        }
    }

    fn calculate_window_usage(&self, state: &BucketState) -> u32 {
        state.request_history.iter().map(|(_, weight)| weight).sum()
    }

    fn calculate_retry_time(&self, state: &BucketState, requested_weight: u32) -> Duration {
        // Find the oldest request that we need to wait for
        if let Some((oldest_timestamp, _oldest_weight)) = state.request_history.front() {
            let time_until_expiry = self
                .config
                .window_duration
                .saturating_sub(oldest_timestamp.elapsed());

            // Add a small buffer
            time_until_expiry + Duration::from_millis(100)
        } else {
            // Fallback: calculate based on refill rate
            let current_usage = self.calculate_window_usage(state);
            let deficit = (current_usage + requested_weight) as f64
                - (self.config.capacity as f64 * self.config.safety_margin);
            Duration::from_secs_f64((deficit / self.config.refill_rate).max(0.0))
        }
    }
}

// ============================================================================
// Multi-Exchange Rate Limiter Manager
// ============================================================================

use std::collections::HashMap;

/// Manages rate limiters for multiple exchanges
#[derive(Clone)]
pub struct RateLimiterManager {
    limiters: Arc<RwLock<HashMap<String, Arc<TokenBucket>>>>,
}

impl RateLimiterManager {
    pub fn new() -> Self {
        Self {
            limiters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a rate limiter for an exchange
    pub fn register(&self, exchange: String, config: TokenBucketConfig) -> Result<()> {
        let limiter = Arc::new(TokenBucket::new(config)?);
        self.limiters.write().insert(exchange, limiter);
        Ok(())
    }

    /// Get a rate limiter for an exchange
    pub fn get(&self, exchange: &str) -> Option<Arc<TokenBucket>> {
        self.limiters.read().get(exchange).cloned()
    }

    /// Acquire tokens from a specific exchange's rate limiter
    pub async fn acquire(&self, exchange: &str, weight: u32) -> Result<()> {
        let limiter = self.get(exchange).ok_or_else(|| {
            RateLimitError::InvalidConfig(format!("No limiter for exchange: {}", exchange))
        })?;
        limiter.acquire_async(weight).await
    }

    /// Get metrics for all exchanges
    pub fn all_metrics(&self) -> HashMap<String, Metrics> {
        self.limiters
            .read()
            .iter()
            .map(|(name, limiter)| (name.clone(), limiter.metrics()))
            .collect()
    }
}

impl Default for RateLimiterManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_config_validation() {
        let invalid = TokenBucketConfig {
            capacity: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let valid = TokenBucketConfig::default();
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn test_simple_token_acquisition() {
        let config = TokenBucketConfig {
            capacity: 100,
            refill_rate: 10.0,
            sliding_window: false,
            safety_margin: 1.0,
            ..Default::default()
        };

        let bucket = TokenBucket::new(config).unwrap();

        // Should succeed - we have 100 tokens
        assert!(bucket.acquire(50).is_ok());

        // Should succeed - we have 50 tokens left
        assert!(bucket.acquire(30).is_ok());

        // Should fail - we only have 20 tokens left but need 25
        assert!(bucket.acquire(25).is_err());
    }

    #[test]
    fn test_token_refill() {
        let config = TokenBucketConfig {
            capacity: 100,
            refill_rate: 100.0, // 100 tokens per second
            sliding_window: false,
            safety_margin: 1.0,
            ..Default::default()
        };

        let bucket = TokenBucket::new(config).unwrap();

        // Drain the bucket
        bucket.acquire(100).unwrap();

        // Should fail immediately
        assert!(bucket.acquire(10).is_err());

        // Wait for refill
        std::thread::sleep(Duration::from_millis(200));

        // Should succeed - we've refilled ~20 tokens
        assert!(bucket.acquire(15).is_ok());
    }

    #[test]
    fn test_sliding_window() {
        let config = TokenBucketConfig {
            capacity: 100,
            refill_rate: 100.0,
            sliding_window: true,
            window_duration: Duration::from_secs(1),
            safety_margin: 1.0,
        };

        let bucket = TokenBucket::new(config).unwrap();

        // Make 5 requests of 20 weight each
        for _ in 0..5 {
            assert!(bucket.acquire(20).is_ok());
        }

        // This should fail - we've used 100 in the window
        assert!(bucket.acquire(1).is_err());

        // Wait for the window to pass
        std::thread::sleep(Duration::from_millis(1100));

        // Should succeed now
        assert!(bucket.acquire(50).is_ok());
    }

    #[test]
    fn test_safety_margin() {
        let config = TokenBucketConfig {
            capacity: 100,
            refill_rate: 10.0,
            sliding_window: false,
            safety_margin: 0.8, // Only use 80% of capacity
            ..Default::default()
        };

        let bucket = TokenBucket::new(config).unwrap();

        // Should succeed - 80 <= 80% of 100
        assert!(bucket.acquire(80).is_ok());

        // Should fail - would exceed safety margin
        assert!(bucket.acquire(1).is_err());
    }

    #[test]
    fn test_header_update() {
        let config = TokenBucketConfig {
            capacity: 6000,
            refill_rate: 100.0,
            sliding_window: false,
            safety_margin: 1.0,
            ..Default::default()
        };

        let bucket = TokenBucket::new(config).unwrap();

        // Simulate server saying we've used 5000 of 6000
        bucket.update_from_headers(5000, 6000);

        let metrics = bucket.metrics();
        assert_eq!(metrics.current_tokens as u32, 1000);
    }

    #[tokio::test]
    async fn test_async_acquire_waits() {
        let config = TokenBucketConfig {
            capacity: 10,
            refill_rate: 10.0, // Refill fast for testing
            sliding_window: false,
            safety_margin: 1.0,
            ..Default::default()
        };

        let bucket = TokenBucket::new(config).unwrap();

        // Drain the bucket
        bucket.acquire(10).unwrap();

        let start = Instant::now();

        // This should wait ~0.5 seconds to acquire 5 tokens
        bucket.acquire_async(5).await.unwrap();

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(400));
        assert!(elapsed <= Duration::from_millis(700));
    }

    #[tokio::test]
    async fn test_manager() {
        let manager = RateLimiterManager::new();

        manager
            .register("binance".to_string(), TokenBucketConfig::binance_spot())
            .unwrap();
        manager
            .register("bybit".to_string(), TokenBucketConfig::bybit_v5())
            .unwrap();

        // Should succeed
        assert!(manager.acquire("binance", 100).await.is_ok());
        assert!(manager.acquire("bybit", 10).await.is_ok());

        // Check metrics
        let metrics = manager.all_metrics();
        assert_eq!(metrics.len(), 2);
        assert!(metrics.contains_key("binance"));
        assert!(metrics.contains_key("bybit"));
    }
}
