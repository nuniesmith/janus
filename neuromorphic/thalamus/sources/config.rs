//! Configuration Management Module
//!
//! This module provides configuration management for external data sources
//! and API keys. It supports loading configuration from:
//! - Environment variables
//! - Configuration files (TOML)
//! - Default values
//!
//! ## Environment Variables
//!
//! API Keys:
//! - `NEWSAPI_API_KEY` - NewsAPI.org API key
//! - `CRYPTOPANIC_API_KEY` - CryptoPanic API key
//! - `CRYPTOCOMPARE_API_KEY` - CryptoCompare API key
//! - `OPENWEATHERMAP_API_KEY` - OpenWeatherMap API key
//! - `SPACEWEATHER_API_KEY` - Space Weather API key (optional)
//!
//! Redis Configuration:
//! - `REDIS_URL` - Redis connection URL (default: redis://localhost:6379)
//! - `REDIS_CACHE_TTL` - Cache TTL in seconds (default: 300)
//!
//! Service Configuration:
//! - `AGGREGATOR_POLL_INTERVAL` - Poll interval in seconds (default: 60)
//! - `AGGREGATOR_TIMEOUT` - Request timeout in seconds (default: 30)
//! - `AGGREGATOR_MAX_RETRIES` - Maximum retry attempts (default: 3)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

/// Configuration errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing required configuration: {0}")]
    MissingRequired(String),

    #[error("Invalid configuration value for {key}: {message}")]
    InvalidValue { key: String, message: String },

    #[error("Failed to load configuration file: {0}")]
    FileLoadError(String),

    #[error("Failed to parse configuration: {0}")]
    ParseError(String),

    #[error("Environment variable error: {0}")]
    EnvError(String),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

/// API key configuration for all external data sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// NewsAPI.org API key
    pub newsapi: Option<String>,

    /// CryptoPanic API key
    pub cryptopanic: Option<String>,

    /// CryptoCompare API key
    pub cryptocompare: Option<String>,

    /// OpenWeatherMap API key
    pub openweathermap: Option<String>,

    /// Space Weather API key (NOAA doesn't require one, but for future use)
    pub spaceweather: Option<String>,

    /// Additional API keys stored by name
    #[serde(default)]
    pub custom: HashMap<String, String>,
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            newsapi: None,
            cryptopanic: None,
            cryptocompare: None,
            openweathermap: None,
            spaceweather: None,
            custom: HashMap::new(),
        }
    }
}

impl ApiKeyConfig {
    /// Load API keys from environment variables
    pub fn from_env() -> Self {
        Self {
            newsapi: env::var("NEWSAPI_API_KEY").ok(),
            cryptopanic: env::var("CRYPTOPANIC_API_KEY").ok(),
            cryptocompare: env::var("CRYPTOCOMPARE_API_KEY").ok(),
            openweathermap: env::var("OPENWEATHERMAP_API_KEY").ok(),
            spaceweather: env::var("SPACEWEATHER_API_KEY").ok(),
            custom: HashMap::new(),
        }
    }

    /// Check if a specific API key is configured
    pub fn has_key(&self, name: &str) -> bool {
        match name.to_lowercase().as_str() {
            "newsapi" => self.newsapi.is_some(),
            "cryptopanic" => self.cryptopanic.is_some(),
            "cryptocompare" => self.cryptocompare.is_some(),
            "openweathermap" => self.openweathermap.is_some(),
            "spaceweather" => self.spaceweather.is_some(),
            _ => self.custom.contains_key(name),
        }
    }

    /// Get a specific API key
    pub fn get_key(&self, name: &str) -> Option<&str> {
        match name.to_lowercase().as_str() {
            "newsapi" => self.newsapi.as_deref(),
            "cryptopanic" => self.cryptopanic.as_deref(),
            "cryptocompare" => self.cryptocompare.as_deref(),
            "openweathermap" => self.openweathermap.as_deref(),
            "spaceweather" => self.spaceweather.as_deref(),
            _ => self.custom.get(name).map(String::as_str),
        }
    }

    /// List all configured API keys (names only, not values)
    pub fn configured_keys(&self) -> Vec<&str> {
        let mut keys = Vec::new();
        if self.newsapi.is_some() {
            keys.push("newsapi");
        }
        if self.cryptopanic.is_some() {
            keys.push("cryptopanic");
        }
        if self.cryptocompare.is_some() {
            keys.push("cryptocompare");
        }
        if self.openweathermap.is_some() {
            keys.push("openweathermap");
        }
        if self.spaceweather.is_some() {
            keys.push("spaceweather");
        }
        keys
    }

    /// Merge with another config (other takes precedence)
    pub fn merge(self, other: Self) -> Self {
        let mut custom = self.custom;
        custom.extend(other.custom);

        Self {
            newsapi: other.newsapi.or(self.newsapi),
            cryptopanic: other.cryptopanic.or(self.cryptopanic),
            cryptocompare: other.cryptocompare.or(self.cryptocompare),
            openweathermap: other.openweathermap.or(self.openweathermap),
            spaceweather: other.spaceweather.or(self.spaceweather),
            custom,
        }
    }
}

/// Redis cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,

    /// Default cache TTL in seconds
    pub cache_ttl: u64,

    /// Connection pool size
    pub pool_size: u32,

    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,

    /// Enable Redis caching
    pub enabled: bool,

    /// Key prefix for all cache keys
    pub key_prefix: String,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".to_string(),
            cache_ttl: 300, // 5 minutes
            pool_size: 10,
            connection_timeout_ms: 5000,
            enabled: true,
            key_prefix: "janus:".to_string(),
        }
    }
}

impl RedisConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(url) = env::var("REDIS_URL") {
            config.url = url;
        }

        if let Ok(ttl) = env::var("REDIS_CACHE_TTL") {
            if let Ok(ttl) = ttl.parse() {
                config.cache_ttl = ttl;
            }
        }

        if let Ok(pool_size) = env::var("REDIS_POOL_SIZE") {
            if let Ok(size) = pool_size.parse() {
                config.pool_size = size;
            }
        }

        if let Ok(enabled) = env::var("REDIS_ENABLED") {
            config.enabled = enabled.to_lowercase() != "false" && enabled != "0";
        }

        if let Ok(prefix) = env::var("REDIS_KEY_PREFIX") {
            config.key_prefix = prefix;
        }

        config
    }

    /// Get the cache TTL as a Duration
    pub fn ttl_duration(&self) -> Duration {
        Duration::from_secs(self.cache_ttl)
    }

    /// Format a cache key with the configured prefix
    pub fn format_key(&self, key: &str) -> String {
        format!("{}{}", self.key_prefix, key)
    }
}

/// Aggregator service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorConfig {
    /// Poll interval for data sources (in seconds)
    pub poll_interval: u64,

    /// Request timeout (in seconds)
    pub timeout: u64,

    /// Maximum retry attempts
    pub max_retries: u32,

    /// Retry delay (in milliseconds)
    pub retry_delay_ms: u64,

    /// Enable circuit breaker
    pub circuit_breaker_enabled: bool,

    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: u32,

    /// Circuit breaker reset timeout (in seconds)
    pub circuit_breaker_reset_timeout: u64,

    /// Enable parallel fetching
    pub parallel_fetch: bool,

    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            poll_interval: 60, // 1 minute
            timeout: 30,       // 30 seconds
            max_retries: 3,
            retry_delay_ms: 1000, // 1 second
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_reset_timeout: 60, // 1 minute
            parallel_fetch: true,
            max_concurrent_requests: 5,
        }
    }
}

impl AggregatorConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(interval) = env::var("AGGREGATOR_POLL_INTERVAL") {
            if let Ok(interval) = interval.parse() {
                config.poll_interval = interval;
            }
        }

        if let Ok(timeout) = env::var("AGGREGATOR_TIMEOUT") {
            if let Ok(timeout) = timeout.parse() {
                config.timeout = timeout;
            }
        }

        if let Ok(retries) = env::var("AGGREGATOR_MAX_RETRIES") {
            if let Ok(retries) = retries.parse() {
                config.max_retries = retries;
            }
        }

        if let Ok(delay) = env::var("AGGREGATOR_RETRY_DELAY_MS") {
            if let Ok(delay) = delay.parse() {
                config.retry_delay_ms = delay;
            }
        }

        if let Ok(enabled) = env::var("AGGREGATOR_CIRCUIT_BREAKER") {
            config.circuit_breaker_enabled = enabled.to_lowercase() != "false" && enabled != "0";
        }

        if let Ok(parallel) = env::var("AGGREGATOR_PARALLEL_FETCH") {
            config.parallel_fetch = parallel.to_lowercase() != "false" && parallel != "0";
        }

        config
    }

    /// Get poll interval as Duration
    pub fn poll_duration(&self) -> Duration {
        Duration::from_secs(self.poll_interval)
    }

    /// Get timeout as Duration
    pub fn timeout_duration(&self) -> Duration {
        Duration::from_secs(self.timeout)
    }

    /// Get retry delay as Duration
    pub fn retry_delay_duration(&self) -> Duration {
        Duration::from_millis(self.retry_delay_ms)
    }
}

/// Data source specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// Enable this data source
    pub enabled: bool,

    /// Custom poll interval (overrides aggregator default)
    pub poll_interval: Option<u64>,

    /// Custom timeout (overrides aggregator default)
    pub timeout: Option<u64>,

    /// Rate limit (requests per second)
    pub rate_limit: f64,

    /// Custom base URL (for testing or proxies)
    pub base_url: Option<String>,

    /// Additional source-specific settings
    #[serde(default)]
    pub settings: HashMap<String, String>,
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval: None,
            timeout: None,
            rate_limit: 1.0,
            base_url: None,
            settings: HashMap::new(),
        }
    }
}

/// News source specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsSourceConfig {
    /// Base data source config
    #[serde(flatten)]
    pub base: DataSourceConfig,

    /// Keywords to search for
    pub keywords: Vec<String>,

    /// Categories to fetch
    pub categories: Vec<String>,

    /// Languages to filter
    pub languages: Vec<String>,

    /// Countries to filter
    pub countries: Vec<String>,

    /// Maximum articles per fetch
    pub max_articles: usize,
}

impl Default for NewsSourceConfig {
    fn default() -> Self {
        Self {
            base: DataSourceConfig::default(),
            keywords: vec![
                "bitcoin".to_string(),
                "ethereum".to_string(),
                "crypto".to_string(),
                "cryptocurrency".to_string(),
                "blockchain".to_string(),
            ],
            categories: vec!["business".to_string(), "technology".to_string()],
            languages: vec!["en".to_string()],
            countries: vec!["us".to_string()],
            max_articles: 100,
        }
    }
}

/// Weather source specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherSourceConfig {
    /// Base data source config
    #[serde(flatten)]
    pub base: DataSourceConfig,

    /// Locations to fetch weather for (city names or coordinates)
    pub locations: Vec<String>,

    /// Units (metric, imperial, standard)
    pub units: String,
}

impl Default for WeatherSourceConfig {
    fn default() -> Self {
        Self {
            base: DataSourceConfig::default(),
            locations: vec![
                "New York".to_string(),
                "London".to_string(),
                "Tokyo".to_string(),
                "Singapore".to_string(),
            ],
            units: "metric".to_string(),
        }
    }
}

/// Complete external data sources configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExternalDataConfig {
    /// API key configuration
    #[serde(default)]
    pub api_keys: ApiKeyConfig,

    /// Redis cache configuration
    #[serde(default)]
    pub redis: RedisConfig,

    /// Aggregator configuration
    #[serde(default)]
    pub aggregator: AggregatorConfig,

    /// NewsAPI configuration
    #[serde(default)]
    pub newsapi: NewsSourceConfig,

    /// CryptoPanic configuration
    #[serde(default)]
    pub cryptopanic: NewsSourceConfig,

    /// CryptoCompare configuration
    #[serde(default)]
    pub cryptocompare: NewsSourceConfig,

    /// OpenWeatherMap configuration
    #[serde(default)]
    pub openweathermap: WeatherSourceConfig,

    /// Space Weather configuration
    #[serde(default)]
    pub spaceweather: DataSourceConfig,
}

impl ExternalDataConfig {
    /// Create a new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            api_keys: ApiKeyConfig::from_env(),
            redis: RedisConfig::from_env(),
            aggregator: AggregatorConfig::from_env(),
            ..Self::default()
        }
    }

    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ConfigError::FileLoadError(format!("Failed to read {}: {}", path.as_ref().display(), e))
        })?;

        toml::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Load configuration from environment with file fallback
    pub fn load<P: AsRef<Path>>(config_path: Option<P>) -> Result<Self> {
        // Start with defaults
        let mut config = Self::default();

        // Try to load from file if provided
        if let Some(path) = config_path {
            if path.as_ref().exists() {
                let file_config = Self::from_file(path)?;
                config = config.merge(file_config);
            }
        }

        // Override with environment variables
        let env_config = Self::from_env();
        config = config.merge(env_config);

        Ok(config)
    }

    /// Merge with another config (other takes precedence for non-default values)
    pub fn merge(self, other: Self) -> Self {
        Self {
            api_keys: self.api_keys.merge(other.api_keys),
            redis: if other.redis.url != RedisConfig::default().url {
                other.redis
            } else {
                self.redis
            },
            aggregator: other.aggregator,
            newsapi: other.newsapi,
            cryptopanic: other.cryptopanic,
            cryptocompare: other.cryptocompare,
            openweathermap: other.openweathermap,
            spaceweather: other.spaceweather,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Check that at least one news source is configured
        let has_news_source = self.api_keys.has_key("newsapi")
            || self.api_keys.has_key("cryptopanic")
            || self.api_keys.has_key("cryptocompare");

        if !has_news_source {
            tracing::warn!("No news API keys configured - news sentiment will be unavailable");
        }

        // Validate Redis URL if enabled
        if self.redis.enabled && !self.redis.url.starts_with("redis://") {
            return Err(ConfigError::InvalidValue {
                key: "redis.url".to_string(),
                message: "Redis URL must start with redis://".to_string(),
            });
        }

        // Validate aggregator settings
        if self.aggregator.poll_interval < 10 {
            return Err(ConfigError::InvalidValue {
                key: "aggregator.poll_interval".to_string(),
                message: "Poll interval must be at least 10 seconds".to_string(),
            });
        }

        Ok(())
    }

    /// Get list of enabled data sources
    pub fn enabled_sources(&self) -> Vec<&str> {
        let mut sources = Vec::new();

        if self.newsapi.base.enabled && self.api_keys.has_key("newsapi") {
            sources.push("newsapi");
        }
        if self.cryptopanic.base.enabled && self.api_keys.has_key("cryptopanic") {
            sources.push("cryptopanic");
        }
        if self.cryptocompare.base.enabled && self.api_keys.has_key("cryptocompare") {
            sources.push("cryptocompare");
        }
        if self.openweathermap.base.enabled && self.api_keys.has_key("openweathermap") {
            sources.push("openweathermap");
        }
        // Space weather doesn't require API key
        if self.spaceweather.enabled {
            sources.push("spaceweather");
        }

        sources
    }
}

/// Configuration builder for fluent API
pub struct ConfigBuilder {
    config: ExternalDataConfig,
}

impl ConfigBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self {
            config: ExternalDataConfig::default(),
        }
    }

    /// Set NewsAPI key
    pub fn newsapi_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_keys.newsapi = Some(key.into());
        self
    }

    /// Set CryptoPanic key
    pub fn cryptopanic_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_keys.cryptopanic = Some(key.into());
        self
    }

    /// Set CryptoCompare key
    pub fn cryptocompare_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_keys.cryptocompare = Some(key.into());
        self
    }

    /// Set OpenWeatherMap key
    pub fn openweathermap_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_keys.openweathermap = Some(key.into());
        self
    }

    /// Set Redis URL
    pub fn redis_url(mut self, url: impl Into<String>) -> Self {
        self.config.redis.url = url.into();
        self
    }

    /// Set Redis cache TTL
    pub fn redis_ttl(mut self, ttl_seconds: u64) -> Self {
        self.config.redis.cache_ttl = ttl_seconds;
        self
    }

    /// Enable or disable Redis
    pub fn redis_enabled(mut self, enabled: bool) -> Self {
        self.config.redis.enabled = enabled;
        self
    }

    /// Set aggregator poll interval
    pub fn poll_interval(mut self, seconds: u64) -> Self {
        self.config.aggregator.poll_interval = seconds;
        self
    }

    /// Set request timeout
    pub fn timeout(mut self, seconds: u64) -> Self {
        self.config.aggregator.timeout = seconds;
        self
    }

    /// Set max retries
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.aggregator.max_retries = retries;
        self
    }

    /// Enable circuit breaker
    pub fn circuit_breaker(mut self, enabled: bool) -> Self {
        self.config.aggregator.circuit_breaker_enabled = enabled;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<ExternalDataConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation (for testing)
    pub fn build_unchecked(self) -> ExternalDataConfig {
        self.config
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_config_default() {
        let config = ApiKeyConfig::default();
        assert!(config.newsapi.is_none());
        assert!(config.configured_keys().is_empty());
    }

    #[test]
    fn test_api_key_config_has_key() {
        let mut config = ApiKeyConfig::default();
        config.newsapi = Some("test-key".to_string());
        assert!(config.has_key("newsapi"));
        assert!(config.has_key("NEWSAPI"));
        assert!(!config.has_key("cryptopanic"));
    }

    #[test]
    fn test_api_key_config_merge() {
        let config1 = ApiKeyConfig {
            newsapi: Some("key1".to_string()),
            cryptopanic: Some("key2".to_string()),
            ..Default::default()
        };

        let config2 = ApiKeyConfig {
            newsapi: Some("key1-override".to_string()),
            openweathermap: Some("key3".to_string()),
            ..Default::default()
        };

        let merged = config1.merge(config2);
        assert_eq!(merged.newsapi, Some("key1-override".to_string()));
        assert_eq!(merged.cryptopanic, Some("key2".to_string()));
        assert_eq!(merged.openweathermap, Some("key3".to_string()));
    }

    #[test]
    fn test_redis_config_default() {
        let config = RedisConfig::default();
        assert_eq!(config.url, "redis://localhost:6379");
        assert_eq!(config.cache_ttl, 300);
        assert!(config.enabled);
    }

    #[test]
    fn test_redis_config_format_key() {
        let config = RedisConfig::default();
        assert_eq!(config.format_key("test"), "janus:test");
    }

    #[test]
    fn test_aggregator_config_default() {
        let config = AggregatorConfig::default();
        assert_eq!(config.poll_interval, 60);
        assert_eq!(config.timeout, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .newsapi_key("test-key")
            .redis_url("redis://localhost:6380")
            .poll_interval(120)
            .build_unchecked();

        assert_eq!(config.api_keys.newsapi, Some("test-key".to_string()));
        assert_eq!(config.redis.url, "redis://localhost:6380");
        assert_eq!(config.aggregator.poll_interval, 120);
    }

    #[test]
    fn test_config_validation() {
        let config = ConfigBuilder::new()
            .poll_interval(5) // Too low
            .build();

        assert!(config.is_err());
    }

    #[test]
    fn test_enabled_sources() {
        let config = ConfigBuilder::new()
            .newsapi_key("test-key")
            .build_unchecked();

        let sources = config.enabled_sources();
        assert!(sources.contains(&"newsapi"));
        assert!(sources.contains(&"spaceweather")); // Doesn't require key
    }

    #[test]
    fn test_news_source_config_default() {
        let config = NewsSourceConfig::default();
        assert!(config.keywords.contains(&"bitcoin".to_string()));
        assert!(config.base.enabled);
    }

    #[test]
    fn test_weather_source_config_default() {
        let config = WeatherSourceConfig::default();
        assert!(config.locations.contains(&"New York".to_string()));
        assert_eq!(config.units, "metric");
    }
}
