//! Integration test configuration
//!
//! Loads Kraken credentials and infrastructure endpoints from environment
//! variables for safe integration testing without hardcoding secrets.
//!
//! Kraken is the primary exchange for:
//! - WebSocket data streams (public market data — FREE, no key required)
//! - REST API for historical data and authenticated endpoints
//! - Order execution (requires API key + secret)

use std::env;
use std::time::Duration;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TestnetConfig {
    pub kraken: KrakenTestConfig,
    pub redis_url: String,
    pub questdb_host: String,
    pub questdb_port: u16,
    pub grpc_port: u16,
    pub http_port: u16,
    pub test_timeout: Duration,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct KrakenTestConfig {
    /// Kraken API key (for authenticated endpoints)
    pub api_key: String,
    /// Kraken API secret (for signing authenticated requests)
    pub api_secret: String,
    /// Kraken REST API URL
    pub rest_url: String,
    /// Kraken public WebSocket URL (v2)
    pub ws_public_url: String,
    /// Kraken private WebSocket URL (v2, requires auth)
    pub ws_private_url: String,
    /// Whether authenticated endpoints are available (key + secret set)
    pub authenticated: bool,
    /// Whether to use the demo/futures environment
    pub use_demo: bool,
}

impl TestnetConfig {
    /// Load configuration from environment variables
    ///
    /// Required env vars for authenticated Kraken tests:
    /// - KRAKEN_API_KEY
    /// - KRAKEN_API_SECRET
    ///
    /// Optional env vars:
    /// - KRAKEN_REST_URL (default: https://api.kraken.com)
    /// - KRAKEN_WS_PUBLIC_URL (default: wss://ws.kraken.com/v2)
    /// - KRAKEN_WS_PRIVATE_URL (default: wss://ws-auth.kraken.com/v2)
    /// - KRAKEN_USE_DEMO (default: false)
    /// - REDIS_URL (default: redis://localhost:6379)
    /// - QUESTDB_HOST (default: localhost)
    /// - QUESTDB_PORT (default: 9009)
    /// - GRPC_PORT (default: 50052)
    /// - HTTP_PORT (default: 8081)
    /// - TEST_TIMEOUT_SECS (default: 30)
    pub fn from_env() -> Result<Self, String> {
        let kraken_api_key = env::var("KRAKEN_API_KEY").ok();
        let kraken_api_secret = env::var("KRAKEN_API_SECRET").ok();

        let authenticated = kraken_api_key.is_some() && kraken_api_secret.is_some();

        let use_demo = env::var("KRAKEN_USE_DEMO")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);

        let kraken = KrakenTestConfig {
            api_key: kraken_api_key.unwrap_or_else(|| "MOCK_KEY".to_string()),
            api_secret: kraken_api_secret.unwrap_or_else(|| "MOCK_SECRET".to_string()),
            rest_url: env::var("KRAKEN_REST_URL")
                .unwrap_or_else(|_| "https://api.kraken.com".to_string()),
            ws_public_url: env::var("KRAKEN_WS_PUBLIC_URL")
                .unwrap_or_else(|_| "wss://ws.kraken.com/v2".to_string()),
            ws_private_url: env::var("KRAKEN_WS_PRIVATE_URL")
                .unwrap_or_else(|_| "wss://ws-auth.kraken.com/v2".to_string()),
            authenticated,
            use_demo,
        };

        let redis_url =
            env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());

        let questdb_host = env::var("QUESTDB_HOST").unwrap_or_else(|_| "localhost".to_string());

        let questdb_port = env::var("QUESTDB_PORT")
            .unwrap_or_else(|_| "9009".to_string())
            .parse::<u16>()
            .map_err(|e| format!("Invalid QUESTDB_PORT: {}", e))?;

        let grpc_port = env::var("GRPC_PORT")
            .unwrap_or_else(|_| "50052".to_string())
            .parse::<u16>()
            .map_err(|e| format!("Invalid GRPC_PORT: {}", e))?;

        let http_port = env::var("HTTP_PORT")
            .unwrap_or_else(|_| "8081".to_string())
            .parse::<u16>()
            .map_err(|e| format!("Invalid HTTP_PORT: {}", e))?;

        let test_timeout_secs = env::var("TEST_TIMEOUT_SECS")
            .unwrap_or_else(|_| "30".to_string())
            .parse::<u64>()
            .map_err(|e| format!("Invalid TEST_TIMEOUT_SECS: {}", e))?;

        Ok(TestnetConfig {
            kraken,
            redis_url,
            questdb_host,
            questdb_port,
            grpc_port,
            http_port,
            test_timeout: Duration::from_secs(test_timeout_secs),
        })
    }

    /// Create a mock configuration for offline testing
    pub fn mock() -> Self {
        TestnetConfig {
            kraken: KrakenTestConfig {
                api_key: "MOCK_API_KEY".to_string(),
                api_secret: "MOCK_API_SECRET".to_string(),
                rest_url: "https://api.kraken.com".to_string(),
                ws_public_url: "wss://ws.kraken.com/v2".to_string(),
                ws_private_url: "wss://ws-auth.kraken.com/v2".to_string(),
                authenticated: false,
                use_demo: false,
            },
            redis_url: "redis://localhost:6379".to_string(),
            questdb_host: "localhost".to_string(),
            questdb_port: 9009,
            grpc_port: 50052,
            http_port: 8081,
            test_timeout: Duration::from_secs(30),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_config_creation() {
        let config = TestnetConfig::mock();
        assert!(!config.kraken.authenticated);
        assert_eq!(config.kraken.api_key, "MOCK_API_KEY");
        assert_eq!(config.kraken.rest_url, "https://api.kraken.com");
        assert_eq!(config.kraken.ws_public_url, "wss://ws.kraken.com/v2");
        assert_eq!(config.grpc_port, 50052);
        assert_eq!(config.http_port, 8081);
    }

    #[test]
    fn test_from_env_without_credentials() {
        // Should succeed but with unauthenticated kraken
        let config = TestnetConfig::from_env().unwrap();
        // Kraken authenticated should be false if credentials not in env
        // (we can't assert this since env might have them)
        assert!(config.grpc_port > 0);
    }

    #[test]
    fn test_mock_config_defaults() {
        let config = TestnetConfig::mock();
        assert!(!config.kraken.use_demo);
        assert_eq!(config.questdb_port, 9009);
        assert_eq!(config.test_timeout, Duration::from_secs(30));
    }
}
