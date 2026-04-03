//! Rate limiting for exchange API calls
//!
//! This module implements a token bucket rate limiter to prevent exceeding
//! exchange API rate limits and getting banned or throttled.

use crate::error::{ExecutionError, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, warn};

/// Rate limiter using token bucket algorithm
pub struct RateLimiter {
    /// Rate limit configurations per endpoint
    limits: Arc<Mutex<HashMap<String, EndpointLimit>>>,

    /// Global rate limit
    global_limit: Arc<Mutex<GlobalLimit>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    ///
    /// # Arguments
    /// * `requests_per_second` - Global requests per second limit
    pub fn new(requests_per_second: u32) -> Self {
        Self {
            limits: Arc::new(Mutex::new(HashMap::new())),
            global_limit: Arc::new(Mutex::new(GlobalLimit::new(requests_per_second))),
        }
    }

    /// Add an endpoint-specific rate limit
    ///
    /// # Arguments
    /// * `endpoint` - The endpoint path (e.g., "/v5/order/create")
    /// * `requests_per_second` - Requests per second limit for this endpoint
    pub fn add_endpoint_limit(&self, endpoint: impl Into<String>, requests_per_second: u32) {
        let endpoint = endpoint.into();
        self.limits
            .lock()
            .insert(endpoint.clone(), EndpointLimit::new(requests_per_second));
        debug!(
            "Added rate limit for {}: {} req/s",
            endpoint, requests_per_second
        );
    }

    /// Acquire permission to make a request
    ///
    /// This method will block until a token is available, or return an error
    /// if rate limit is exceeded and waiting would take too long.
    ///
    /// # Arguments
    /// * `endpoint` - Optional endpoint path for endpoint-specific limiting
    ///
    /// # Returns
    /// * `Ok(())` - Permission granted
    /// * `Err(ExecutionError)` - Rate limit exceeded
    pub async fn acquire(&self, endpoint: Option<&str>) -> Result<()> {
        // Check global limit first
        let wait_time = {
            let mut global = self.global_limit.lock();
            global.wait_time()
        };

        if wait_time > Duration::from_secs(5) {
            warn!("Global rate limit exceeded, would wait {:?}", wait_time);
            return Err(ExecutionError::RateLimitExceeded("global".to_string()));
        }

        if wait_time > Duration::ZERO {
            debug!("Global rate limit: waiting {:?}", wait_time);
            sleep(wait_time).await;
        }

        // Consume global token
        self.global_limit.lock().consume();

        // Check endpoint-specific limit if provided
        if let Some(endpoint) = endpoint {
            let wait_time = {
                let mut limits = self.limits.lock();
                if let Some(limit) = limits.get_mut(endpoint) {
                    limit.wait_time()
                } else {
                    Duration::ZERO
                }
            };

            if wait_time > Duration::from_secs(5) {
                warn!(
                    "Endpoint rate limit exceeded for {}, would wait {:?}",
                    endpoint, wait_time
                );
                return Err(ExecutionError::RateLimitExceeded(endpoint.to_string()));
            }

            if wait_time > Duration::ZERO {
                debug!(
                    "Endpoint rate limit for {}: waiting {:?}",
                    endpoint, wait_time
                );
                sleep(wait_time).await;
            }

            // Consume endpoint token
            if let Some(limit) = self.limits.lock().get_mut(endpoint) {
                limit.consume();
            }
        }

        Ok(())
    }

    /// Check if a request would be rate limited without consuming tokens
    ///
    /// # Arguments
    /// * `endpoint` - Optional endpoint path
    ///
    /// # Returns
    /// * `Duration` - Time to wait before request would be allowed
    pub fn check(&self, endpoint: Option<&str>) -> Duration {
        let global_wait = self.global_limit.lock().wait_time();

        if let Some(endpoint) = endpoint {
            let endpoint_wait = self
                .limits
                .lock()
                .get_mut(endpoint)
                .map(|l| l.wait_time())
                .unwrap_or(Duration::ZERO);

            global_wait.max(endpoint_wait)
        } else {
            global_wait
        }
    }

    /// Reset all rate limits (useful for testing)
    pub fn reset(&self) {
        self.global_limit.lock().reset();
        for limit in self.limits.lock().values_mut() {
            limit.reset();
        }
    }

    /// Get statistics about rate limiting
    pub fn stats(&self) -> RateLimitStats {
        let global = self.global_limit.lock();
        let endpoints = self
            .limits
            .lock()
            .iter()
            .map(|(k, v)| (k.clone(), v.tokens))
            .collect();

        RateLimitStats {
            global_tokens: global.tokens,
            global_capacity: global.capacity,
            endpoint_tokens: endpoints,
        }
    }
}

/// Global rate limit state
struct GlobalLimit {
    /// Current number of tokens
    tokens: f64,

    /// Maximum tokens (capacity)
    capacity: f64,

    /// Tokens per second refill rate
    refill_rate: f64,

    /// Last refill time
    last_refill: Instant,
}

impl GlobalLimit {
    fn new(requests_per_second: u32) -> Self {
        let capacity = requests_per_second as f64;
        Self {
            tokens: capacity,
            capacity,
            refill_rate: requests_per_second as f64,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * self.refill_rate;

        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }

    fn wait_time(&mut self) -> Duration {
        self.refill();

        if self.tokens >= 1.0 {
            Duration::ZERO
        } else {
            let tokens_needed = 1.0 - self.tokens;
            let seconds = tokens_needed / self.refill_rate;
            Duration::from_secs_f64(seconds)
        }
    }

    fn consume(&mut self) {
        self.refill();
        self.tokens = (self.tokens - 1.0).max(0.0);
    }

    fn reset(&mut self) {
        self.tokens = self.capacity;
        self.last_refill = Instant::now();
    }
}

/// Endpoint-specific rate limit state
struct EndpointLimit {
    /// Current number of tokens
    tokens: f64,

    /// Maximum tokens (capacity)
    capacity: f64,

    /// Tokens per second refill rate
    refill_rate: f64,

    /// Last refill time
    last_refill: Instant,
}

impl EndpointLimit {
    fn new(requests_per_second: u32) -> Self {
        let capacity = requests_per_second as f64;
        Self {
            tokens: capacity,
            capacity,
            refill_rate: requests_per_second as f64,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * self.refill_rate;

        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }

    fn wait_time(&mut self) -> Duration {
        self.refill();

        if self.tokens >= 1.0 {
            Duration::ZERO
        } else {
            let tokens_needed = 1.0 - self.tokens;
            let seconds = tokens_needed / self.refill_rate;
            Duration::from_secs_f64(seconds)
        }
    }

    fn consume(&mut self) {
        self.refill();
        self.tokens = (self.tokens - 1.0).max(0.0);
    }

    fn reset(&mut self) {
        self.tokens = self.capacity;
        self.last_refill = Instant::now();
    }
}

/// Rate limit statistics
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    /// Global tokens available
    pub global_tokens: f64,

    /// Global capacity
    pub global_capacity: f64,

    /// Endpoint-specific tokens
    pub endpoint_tokens: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let limiter = RateLimiter::new(10); // 10 req/s

        // Should allow first request immediately
        assert!(limiter.acquire(None).await.is_ok());

        // Should allow burst up to capacity
        for _ in 0..9 {
            assert!(limiter.acquire(None).await.is_ok());
        }

        // Should require waiting for next token
        let start = Instant::now();
        assert!(limiter.acquire(None).await.is_ok());
        let elapsed = start.elapsed();

        // Should have waited at least 80ms (allowing some margin)
        assert!(elapsed.as_millis() >= 50);
    }

    #[tokio::test]
    async fn test_endpoint_specific_limit() {
        let limiter = RateLimiter::new(100); // High global limit
        limiter.add_endpoint_limit("/test", 2); // 2 req/s for specific endpoint

        // Should allow 2 requests immediately
        assert!(limiter.acquire(Some("/test")).await.is_ok());
        assert!(limiter.acquire(Some("/test")).await.is_ok());

        // Third request should wait
        let start = Instant::now();
        assert!(limiter.acquire(Some("/test")).await.is_ok());
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() >= 400); // Should wait ~500ms
    }

    #[test]
    fn test_check_without_consume() {
        let limiter = RateLimiter::new(10);

        // Check should not consume tokens
        let wait1 = limiter.check(None);
        let wait2 = limiter.check(None);

        assert_eq!(wait1, wait2);
        assert_eq!(wait1, Duration::ZERO);
    }

    #[test]
    fn test_reset() {
        let limiter = RateLimiter::new(5);

        // Consume all tokens
        for _ in 0..5 {
            limiter.global_limit.lock().consume();
        }

        // Should need to wait
        assert!(limiter.check(None) > Duration::ZERO);

        // Reset
        limiter.reset();

        // Should be able to proceed immediately
        assert_eq!(limiter.check(None), Duration::ZERO);
    }

    #[test]
    fn test_stats() {
        let limiter = RateLimiter::new(10);
        limiter.add_endpoint_limit("/test", 5);

        let stats = limiter.stats();
        assert_eq!(stats.global_capacity, 10.0);
        assert!(stats.endpoint_tokens.contains_key("/test"));
    }

    #[tokio::test]
    async fn test_rate_limit_exceeded_error() {
        let limiter = RateLimiter::new(1); // Very low limit

        // Consume token
        limiter.global_limit.lock().tokens = 0.0;

        // Manually set wait time to exceed threshold
        limiter.global_limit.lock().refill_rate = 0.1; // Very slow refill

        // Should return error instead of waiting too long
        let result = limiter.acquire(None).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutionError::RateLimitExceeded(_)
        ));
    }
}
