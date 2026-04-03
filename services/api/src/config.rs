//! Configuration module for the JANUS Rust Gateway.
//!
//! This module mirrors the Python gateway's configuration, enabling
//! environment-based settings for service URLs, Redis, and CORS.

use serde::Deserialize;
use std::env;

/// Gateway configuration settings.
///
/// Mirrors the Python `Settings` class from `config.py`.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Settings {
    /// Service name
    #[serde(default = "default_service_name")]
    pub service_name: String,

    /// HTTP server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// Environment (development, staging, production)
    #[serde(default = "default_environment")]
    pub environment: String,

    /// Janus Forward service gRPC URL (host:port)
    #[serde(default = "default_forward_url")]
    pub janus_forward_url: String,

    /// Janus Backward service gRPC URL (host:port)
    #[serde(default = "default_backward_url")]
    pub janus_backward_url: String,

    /// Redis URL for signal publishing (Pub/Sub with janus-forward)
    #[serde(default = "default_redis_signal_url")]
    pub redis_signal_url: String,

    /// Redis URL for Celery broker/backend (separate DB)
    #[serde(default = "default_redis_url")]
    pub redis_url: String,

    /// CORS origins (comma-separated or "*")
    #[serde(default = "default_cors_origins")]
    pub cors_origins: String,

    /// QuestDB host
    #[serde(default = "default_questdb_host")]
    pub questdb_host: String,

    /// QuestDB HTTP port
    #[serde(default = "default_questdb_port")]
    pub questdb_http_port: u16,

    /// Signal generation symbols (comma-separated)
    #[serde(default = "default_signal_symbols")]
    pub signal_generation_symbols: String,

    /// Signal generation interval in seconds
    #[serde(default = "default_signal_interval")]
    pub signal_generation_interval: u64,

    /// Directory for signal JSON files
    #[serde(default = "default_signals_dir")]
    pub signals_dir: String,

    /// Dead Man's Switch heartbeat interval in seconds
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u64,
}

// Default value functions
fn default_service_name() -> String {
    "janus-gateway".to_string()
}

fn default_port() -> u16 {
    8000
}

fn default_environment() -> String {
    "development".to_string()
}

fn default_forward_url() -> String {
    "localhost:50051".to_string()
}

fn default_backward_url() -> String {
    "localhost:50052".to_string()
}

fn default_redis_signal_url() -> String {
    "redis://localhost:6379/0".to_string()
}

fn default_redis_url() -> String {
    "redis://localhost:6379/1".to_string()
}

fn default_cors_origins() -> String {
    "*".to_string()
}

fn default_questdb_host() -> String {
    "localhost".to_string()
}

fn default_questdb_port() -> u16 {
    9000
}

fn default_signal_symbols() -> String {
    "BTC/USDT,ETH/USDT".to_string()
}

fn default_signal_interval() -> u64 {
    900 // 15 minutes
}

fn default_signals_dir() -> String {
    "/tmp/signals".to_string()
}

fn default_heartbeat_interval() -> u64 {
    2 // 2 seconds (Rust forward expects heartbeat within 5s)
}

impl Settings {
    /// Load settings from environment variables.
    ///
    /// Environment variables are mapped as follows:
    /// - `SERVICE_NAME` -> service_name
    /// - `PORT` -> port
    /// - `ENVIRONMENT` -> environment
    /// - `JANUS_FORWARD_URL` -> janus_forward_url
    /// - `JANUS_BACKWARD_URL` -> janus_backward_url
    /// - `REDIS_SIGNAL_URL` -> redis_signal_url
    /// - `REDIS_URL` -> redis_url
    /// - `CORS_ORIGINS` -> cors_origins
    /// - `QUESTDB_HOST` -> questdb_host
    /// - `QUESTDB_HTTP_PORT` -> questdb_http_port
    pub fn from_env() -> Self {
        // Load .env file if present
        let _ = dotenvy::dotenv();

        Self {
            service_name: env::var("SERVICE_NAME").unwrap_or_else(|_| default_service_name()),
            port: env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or_else(default_port),
            environment: env::var("ENVIRONMENT").unwrap_or_else(|_| default_environment()),
            janus_forward_url: env::var("JANUS_FORWARD_URL")
                .unwrap_or_else(|_| default_forward_url()),
            janus_backward_url: env::var("JANUS_BACKWARD_URL")
                .unwrap_or_else(|_| default_backward_url()),
            redis_signal_url: env::var("REDIS_SIGNAL_URL")
                .unwrap_or_else(|_| default_redis_signal_url()),
            redis_url: env::var("REDIS_URL").unwrap_or_else(|_| default_redis_url()),
            cors_origins: env::var("CORS_ORIGINS").unwrap_or_else(|_| default_cors_origins()),
            questdb_host: env::var("QUESTDB_HOST").unwrap_or_else(|_| default_questdb_host()),
            questdb_http_port: env::var("QUESTDB_HTTP_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or_else(default_questdb_port),
            signal_generation_symbols: env::var("SIGNAL_GENERATION_SYMBOLS")
                .unwrap_or_else(|_| default_signal_symbols()),
            signal_generation_interval: env::var("SIGNAL_GENERATION_INTERVAL")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or_else(default_signal_interval),
            signals_dir: env::var("SIGNALS_DIR").unwrap_or_else(|_| default_signals_dir()),
            heartbeat_interval_secs: env::var("HEARTBEAT_INTERVAL_SECS")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or_else(default_heartbeat_interval),
        }
    }

    /// Get CORS origins as a vector of strings.
    pub fn cors_origins_list(&self) -> Vec<String> {
        if self.cors_origins == "*" {
            vec!["*".to_string()]
        } else {
            self.cors_origins
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        }
    }

    /// Get signal generation symbols as a vector.
    #[allow(dead_code)]
    pub fn signal_symbols_list(&self) -> Vec<String> {
        self.signal_generation_symbols
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Check if running in production mode.
    pub fn is_production(&self) -> bool {
        self.environment.to_lowercase() == "production"
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = Settings {
            service_name: default_service_name(),
            port: default_port(),
            environment: default_environment(),
            janus_forward_url: default_forward_url(),
            janus_backward_url: default_backward_url(),
            redis_signal_url: default_redis_signal_url(),
            redis_url: default_redis_url(),
            cors_origins: default_cors_origins(),
            questdb_host: default_questdb_host(),
            questdb_http_port: default_questdb_port(),
            signal_generation_symbols: default_signal_symbols(),
            signal_generation_interval: default_signal_interval(),
            signals_dir: default_signals_dir(),
            heartbeat_interval_secs: default_heartbeat_interval(),
        };

        assert_eq!(settings.service_name, "janus-gateway");
        assert_eq!(settings.port, 8000);
        assert_eq!(settings.cors_origins_list(), vec!["*"]);
    }

    #[test]
    fn test_cors_origins_list() {
        let settings = Settings {
            cors_origins: "http://localhost:3000, http://localhost:8080".to_string(),
            ..Settings::from_env()
        };

        let origins = settings.cors_origins_list();
        assert_eq!(origins.len(), 2);
        assert_eq!(origins[0], "http://localhost:3000");
        assert_eq!(origins[1], "http://localhost:8080");
    }

    #[test]
    fn test_signal_symbols_list() {
        let settings = Settings {
            signal_generation_symbols: "BTC/USDT, ETH/USDT, SOL/USDT".to_string(),
            ..Settings::from_env()
        };

        let symbols = settings.signal_symbols_list();
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0], "BTC/USDT");
    }
}
