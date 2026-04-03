//! API Clients for External Data Sources
//!
//! This module provides concrete API client implementations for various
//! external data sources including news, weather, and celestial data.
//!
//! ## Supported APIs
//!
//! ### News Sources
//! - CryptoCompare News API
//! - NewsAPI.org
//! - CryptoPanic
//!
//! ### Weather Sources
//! - OpenWeatherMap
//! - WeatherAPI
//!
//! ### Celestial Sources
//! - NOAA Space Weather Prediction Center
//! - Astronomy API (moon phases)

pub mod cryptocompare;
pub mod cryptopanic;
pub mod newsapi;
pub mod openweathermap;
pub mod spaceweather;

// Re-exports
pub use cryptocompare::CryptoCompareClient;
pub use cryptopanic::CryptoPanicClient;
pub use newsapi::NewsApiClient;
pub use openweathermap::OpenWeatherMapClient;
pub use spaceweather::SpaceWeatherClient;

use crate::common::Result;
use async_trait::async_trait;
use std::time::Duration;

/// Common configuration for API clients
#[derive(Debug, Clone)]
pub struct ApiClientConfig {
    /// API key for authentication
    pub api_key: Option<String>,

    /// Base URL for the API
    pub base_url: String,

    /// Request timeout
    pub timeout: Duration,

    /// Maximum retries on failure
    pub max_retries: u32,

    /// Delay between retries
    pub retry_delay: Duration,

    /// User agent string
    pub user_agent: String,
}

impl Default for ApiClientConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: String::new(),
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            user_agent: "JANUS-Trading-System/1.0".to_string(),
        }
    }
}

/// Trait for API clients with retry logic
#[async_trait]
pub trait ApiClient: Send + Sync {
    /// Get client name
    fn name(&self) -> &str;

    /// Check if client is configured (has API key if required)
    fn is_configured(&self) -> bool;

    /// Health check - verify API connectivity
    async fn health_check(&self) -> bool;
}

/// Helper function to build a reqwest client with common settings
pub fn build_http_client(config: &ApiClientConfig) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(config.timeout)
        .user_agent(&config.user_agent)
        .build()
        .map_err(|e| crate::common::Error::Other(format!("Failed to build HTTP client: {}", e)))
}

/// Rate limiter for API calls
pub struct RateLimiter {
    /// Minimum interval between requests
    interval: Duration,
    /// Last request timestamp
    last_request: std::sync::Mutex<Option<std::time::Instant>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(requests_per_second: f64) -> Self {
        let interval = Duration::from_secs_f64(1.0 / requests_per_second);
        Self {
            interval,
            last_request: std::sync::Mutex::new(None),
        }
    }

    /// Wait if necessary to respect rate limit
    pub async fn wait(&self) {
        let wait_time = {
            let last = self.last_request.lock().unwrap();
            let now = std::time::Instant::now();

            if let Some(last_time) = *last {
                let elapsed = now.duration_since(last_time);
                if elapsed < self.interval {
                    Some(self.interval - elapsed)
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(wait) = wait_time {
            tokio::time::sleep(wait).await;
        }

        // Update last request time
        let mut last = self.last_request.lock().unwrap();
        *last = Some(std::time::Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_client_config_default() {
        let config = ApiClientConfig::default();
        assert!(config.api_key.is_none());
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(10.0); // 10 requests per second

        let start = std::time::Instant::now();
        limiter.wait().await;
        limiter.wait().await;
        let elapsed = start.elapsed();

        // Should have waited at least 100ms (1/10 second) for the second request
        assert!(elapsed >= Duration::from_millis(90)); // Allow some tolerance
    }
}
