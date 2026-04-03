//! Rate limiting middleware for the JANUS Rust Gateway.
//!
//! This module provides configurable rate limiting using the governor crate,
//! supporting per-client and global rate limits.

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use governor::{
    Quota, RateLimiter,
    clock::{Clock, DefaultClock},
    middleware::NoOpMiddleware,
    state::{InMemoryState, NotKeyed},
};
use serde::Serialize;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

/// Rate limit exceeded response.
#[derive(Debug, Clone, Serialize)]
pub struct RateLimitExceeded {
    pub error: String,
    pub message: String,
    pub retry_after_secs: Option<u64>,
}

impl RateLimitExceeded {
    pub fn new(retry_after: Option<Duration>) -> Self {
        Self {
            error: "rate_limit_exceeded".to_string(),
            message: "Too many requests. Please slow down.".to_string(),
            retry_after_secs: retry_after.map(|d| d.as_secs()),
        }
    }
}

/// Global rate limiter (not keyed by client).
pub type GlobalRateLimiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>;

/// Rate limiter configuration.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second (global)
    pub requests_per_second: u32,

    /// Maximum burst size
    pub burst_size: u32,

    /// Enable rate limiting
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 100,
            burst_size: 50,
            enabled: true,
        }
    }
}

impl RateLimitConfig {
    /// Create a new rate limit config with custom values.
    pub fn new(requests_per_second: u32, burst_size: u32) -> Self {
        Self {
            requests_per_second,
            burst_size,
            enabled: true,
        }
    }

    /// Disable rate limiting.
    #[allow(dead_code)]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a rate limiter from this config.
    pub fn create_limiter(&self) -> Option<Arc<GlobalRateLimiter>> {
        if !self.enabled {
            return None;
        }

        let quota = Quota::per_second(
            NonZeroU32::new(self.requests_per_second).unwrap_or(NonZeroU32::new(100).unwrap()),
        )
        .allow_burst(NonZeroU32::new(self.burst_size).unwrap_or(NonZeroU32::new(50).unwrap()));

        Some(Arc::new(RateLimiter::direct(quota)))
    }
}

/// Shared state for rate limiting middleware.
#[derive(Clone)]
#[allow(dead_code)]
pub struct RateLimitState {
    /// Global rate limiter (optional - None means disabled)
    pub limiter: Option<Arc<GlobalRateLimiter>>,

    /// Configuration
    pub config: RateLimitConfig,
}

impl RateLimitState {
    /// Create a new rate limit state from config.
    pub fn new(config: RateLimitConfig) -> Self {
        let limiter = config.create_limiter();
        Self { limiter, config }
    }

    /// Create a disabled rate limit state.
    #[allow(dead_code)]
    pub fn disabled() -> Self {
        Self::new(RateLimitConfig::disabled())
    }

    /// Check if a request should be allowed.
    pub fn check(&self) -> Result<(), Duration> {
        match &self.limiter {
            Some(limiter) => match limiter.check() {
                Ok(_) => Ok(()),
                Err(not_until) => {
                    let clock = DefaultClock::default();
                    let wait_time = not_until.wait_time_from(clock.now());
                    Err(wait_time)
                }
            },
            None => Ok(()), // Rate limiting disabled
        }
    }
}

impl Default for RateLimitState {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

/// Rate limiting middleware function.
///
/// This middleware checks the global rate limiter before processing requests.
/// If the rate limit is exceeded, it returns a 429 Too Many Requests response.
///
/// # Usage
///
/// ```rust,ignore
/// use axum::{Router, middleware};
/// use janus_gateway::rate_limit::{rate_limit_middleware, RateLimitState, RateLimitConfig};
///
/// let rate_limit = RateLimitState::new(RateLimitConfig::new(100, 50));
///
/// let app: Router<()> = Router::new()
///     .layer(middleware::from_fn_with_state(rate_limit, rate_limit_middleware));
/// ```
pub async fn rate_limit_middleware(
    State(state): State<RateLimitState>,
    request: Request,
    next: Next,
) -> Response {
    // Check rate limit
    match state.check() {
        Ok(_) => {
            // Request allowed, proceed
            next.run(request).await
        }
        Err(wait_time) => {
            // Rate limit exceeded
            let path = request.uri().path().to_string();
            warn!(
                "Rate limit exceeded for path={}, retry_after={}s",
                path,
                wait_time.as_secs()
            );

            let response = RateLimitExceeded::new(Some(wait_time));

            (
                StatusCode::TOO_MANY_REQUESTS,
                [(
                    axum::http::header::RETRY_AFTER,
                    wait_time.as_secs().to_string(),
                )],
                Json(response),
            )
                .into_response()
        }
    }
}

/// Create a rate limit layer with default configuration.
#[allow(dead_code)]
pub fn default_rate_limit_state() -> RateLimitState {
    RateLimitState::default()
}

/// Rate limit specific endpoints more aggressively.
///
/// This is useful for expensive operations like signal dispatch.
#[derive(Clone)]
#[allow(dead_code)]
pub struct EndpointRateLimiter {
    /// Limiter for signal dispatch
    pub signal_dispatch: Option<Arc<GlobalRateLimiter>>,

    /// Limiter for signal generation
    pub signal_generate: Option<Arc<GlobalRateLimiter>>,
}

impl EndpointRateLimiter {
    /// Create endpoint-specific rate limiters.
    pub fn new() -> Self {
        // Signal dispatch: 10 per second, burst of 5
        let signal_dispatch_quota = Quota::per_second(NonZeroU32::new(10).unwrap())
            .allow_burst(NonZeroU32::new(5).unwrap());

        // Signal generation: 1 per second, burst of 2
        let signal_generate_quota =
            Quota::per_second(NonZeroU32::new(1).unwrap()).allow_burst(NonZeroU32::new(2).unwrap());

        Self {
            signal_dispatch: Some(Arc::new(RateLimiter::direct(signal_dispatch_quota))),
            signal_generate: Some(Arc::new(RateLimiter::direct(signal_generate_quota))),
        }
    }

    /// Check signal dispatch rate limit.
    #[allow(dead_code)]
    pub fn check_signal_dispatch(&self) -> Result<(), Duration> {
        match &self.signal_dispatch {
            Some(limiter) => match limiter.check() {
                Ok(_) => Ok(()),
                Err(not_until) => {
                    let clock = DefaultClock::default();
                    let wait_time = not_until.wait_time_from(clock.now());
                    Err(wait_time)
                }
            },
            None => Ok(()),
        }
    }

    /// Check signal generation rate limit.
    #[allow(dead_code)]
    pub fn check_signal_generate(&self) -> Result<(), Duration> {
        match &self.signal_generate {
            Some(limiter) => match limiter.check() {
                Ok(_) => Ok(()),
                Err(not_until) => {
                    let clock = DefaultClock::default();
                    let wait_time = not_until.wait_time_from(clock.now());
                    Err(wait_time)
                }
            },
            None => Ok(()),
        }
    }
}

impl Default for EndpointRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.requests_per_second, 100);
        assert_eq!(config.burst_size, 50);
        assert!(config.enabled);
    }

    #[test]
    fn test_rate_limit_config_disabled() {
        let config = RateLimitConfig::disabled();
        assert!(!config.enabled);
        assert!(config.create_limiter().is_none());
    }

    #[test]
    fn test_rate_limit_config_custom() {
        let config = RateLimitConfig::new(50, 25);
        assert_eq!(config.requests_per_second, 50);
        assert_eq!(config.burst_size, 25);
        assert!(config.enabled);
    }

    #[test]
    fn test_rate_limit_state_creation() {
        let state = RateLimitState::new(RateLimitConfig::new(10, 5));
        assert!(state.limiter.is_some());
    }

    #[test]
    fn test_rate_limit_check_allowed() {
        let state = RateLimitState::new(RateLimitConfig::new(100, 50));

        // First request should be allowed
        assert!(state.check().is_ok());
    }

    #[test]
    fn test_rate_limit_disabled_always_allows() {
        let state = RateLimitState::disabled();

        // All requests should be allowed when disabled
        for _ in 0..1000 {
            assert!(state.check().is_ok());
        }
    }

    #[test]
    fn test_rate_limit_exceeded_response() {
        let response = RateLimitExceeded::new(Some(Duration::from_secs(5)));
        assert_eq!(response.error, "rate_limit_exceeded");
        assert_eq!(response.retry_after_secs, Some(5));
    }

    #[test]
    fn test_endpoint_rate_limiter_creation() {
        let limiter = EndpointRateLimiter::new();
        assert!(limiter.signal_dispatch.is_some());
        assert!(limiter.signal_generate.is_some());
    }

    #[test]
    fn test_endpoint_rate_limiter_check() {
        let limiter = EndpointRateLimiter::new();

        // First requests should be allowed
        assert!(limiter.check_signal_dispatch().is_ok());
        assert!(limiter.check_signal_generate().is_ok());
    }

    #[test]
    fn test_burst_allows_multiple_requests() {
        // Create limiter with burst of 5
        let config = RateLimitConfig::new(1, 5);
        let state = RateLimitState::new(config);

        // Should allow burst of 5 requests
        for i in 0..5 {
            assert!(state.check().is_ok(), "Request {} should be allowed", i);
        }
    }
}
