//! Configuration management for the Data Factory
//!
//! Loads configuration from environment variables with sensible defaults.
//! Supports multiple exchanges, QuestDB, Redis, and metric sources.
//!
//! ## Security: Docker Secrets Support
//!
//! API keys are loaded from Docker Secrets files (mounted at /run/secrets/)
//! instead of environment variables for enhanced security.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;

// ============================================================================
// Secrets Management
// ============================================================================

/// Secure API credentials loaded from Docker Secrets
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ExchangeCredentials {
    /// Binance API credentials
    pub binance: Option<ApiKeyPair>,
    /// Bybit API credentials
    pub bybit: Option<ApiKeyPair>,
    /// Kucoin API credentials (includes passphrase)
    pub kucoin: Option<KucoinCredentials>,
    /// AlphaVantage API key
    pub alphavantage: Option<String>,
    /// CoinMarketCap API key
    pub coinmarketcap: Option<String>,
}

/// API key and secret pair
#[derive(Clone)]
#[allow(dead_code)]
pub struct ApiKeyPair {
    pub api_key: String,
    pub api_secret: String,
}

// Custom Debug to prevent secret leakage in logs
impl std::fmt::Debug for ApiKeyPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyPair")
            .field("api_key", &"***REDACTED***")
            .field("api_secret", &"***REDACTED***")
            .finish()
    }
}

/// Kucoin-specific credentials (includes passphrase)
#[derive(Clone)]
#[allow(dead_code)]
pub struct KucoinCredentials {
    pub api_key: String,
    pub api_secret: String,
    pub passphrase: String,
}

impl std::fmt::Debug for KucoinCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KucoinCredentials")
            .field("api_key", &"***REDACTED***")
            .field("api_secret", &"***REDACTED***")
            .field("passphrase", &"***REDACTED***")
            .finish()
    }
}

/// Read a secret from a file (Docker Secrets pattern)
///
/// Tries the file path first, then falls back to environment variable for development.
/// Returns None if neither exists (allowing optional secrets).
#[allow(dead_code)]
fn read_secret(file_env_var: &str, fallback_env_var: Option<&str>) -> Option<String> {
    // Try reading from file path specified in environment variable
    if let Ok(secret_path) = env::var(file_env_var) {
        if let Ok(content) = fs::read_to_string(&secret_path) {
            let trimmed = content.trim().to_string();
            if !trimmed.is_empty() {
                tracing::debug!(
                    file_path = %secret_path,
                    "Successfully loaded secret from file"
                );
                return Some(trimmed);
            }
        } else {
            tracing::warn!(
                env_var = file_env_var,
                path = %secret_path,
                "Secret file path specified but file not found"
            );
        }
    }

    // Fallback to direct environment variable (development mode)
    if let Some(env_var) = fallback_env_var
        && let Ok(value) = env::var(env_var)
        && !value.is_empty()
    {
        tracing::warn!(
            env_var = env_var,
            "Loading secret from environment variable (NOT RECOMMENDED for production)"
        );
        return Some(value.trim().to_string());
    }

    None
}

/// Load exchange credentials securely from Docker Secrets
///
/// # Security
///
/// - Reads from files mounted at /run/secrets/ (Docker Secrets)
/// - Falls back to environment variables ONLY in development
/// - Never logs actual secret values
/// - Returns None for optional secrets (e.g., read-only exchanges)
#[allow(dead_code)]
pub fn load_credentials() -> Result<ExchangeCredentials> {
    tracing::info!("Loading exchange credentials from Docker Secrets...");

    // Binance credentials
    let binance = match (
        read_secret("BINANCE_API_KEY_FILE", Some("BINANCE_API_KEY")),
        read_secret("BINANCE_API_SECRET_FILE", Some("BINANCE_API_SECRET")),
    ) {
        (Some(key), Some(secret)) => {
            tracing::info!("Binance credentials loaded successfully");
            Some(ApiKeyPair {
                api_key: key,
                api_secret: secret,
            })
        }
        _ => {
            tracing::warn!("Binance credentials not found - will use read-only access");
            None
        }
    };

    // Bybit credentials
    let bybit = match (
        read_secret("BYBIT_API_KEY_FILE", Some("BYBIT_API_KEY")),
        read_secret("BYBIT_API_SECRET_FILE", Some("BYBIT_API_SECRET")),
    ) {
        (Some(key), Some(secret)) => {
            tracing::info!("Bybit credentials loaded successfully");
            Some(ApiKeyPair {
                api_key: key,
                api_secret: secret,
            })
        }
        _ => {
            tracing::warn!("Bybit credentials not found - will use read-only access");
            None
        }
    };

    // Kucoin credentials (requires passphrase)
    let kucoin = match (
        read_secret("KUCOIN_API_KEY_FILE", Some("KUCOIN_API_KEY")),
        read_secret("KUCOIN_API_SECRET_FILE", Some("KUCOIN_API_SECRET")),
        read_secret("KUCOIN_API_PASSPHRASE_FILE", Some("KUCOIN_API_PASSPHRASE")),
    ) {
        (Some(key), Some(secret), Some(passphrase)) => {
            tracing::info!("Kucoin credentials loaded successfully");
            Some(KucoinCredentials {
                api_key: key,
                api_secret: secret,
                passphrase,
            })
        }
        _ => {
            tracing::warn!("Kucoin credentials not found - will use read-only access");
            None
        }
    };

    // AlphaVantage API key
    let alphavantage = read_secret("ALPHAVANTAGE_API_KEY_FILE", Some("ALPHAVANTAGE_API_KEY"));
    if alphavantage.is_some() {
        tracing::info!("AlphaVantage API key loaded successfully");
    }

    // CoinMarketCap API key
    let coinmarketcap = read_secret("COINMARKETCAP_API_KEY_FILE", Some("COINMARKETCAP_API_KEY"));
    if coinmarketcap.is_some() {
        tracing::info!("CoinMarketCap API key loaded successfully");
    }

    Ok(ExchangeCredentials {
        binance,
        bybit,
        kucoin,
        alphavantage,
        coinmarketcap,
    })
}

// ============================================================================
// Configuration Structures
// ============================================================================

/// Main configuration structure for the Data Factory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Assets to ingest (e.g., BTC, ETH, SOL)
    pub assets: Vec<String>,

    /// Exchange configuration
    pub exchanges: ExchangeConfig,

    /// QuestDB connection settings
    pub questdb: QuestDbConfig,

    /// Redis connection settings
    pub redis: RedisConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Operational settings
    pub operational: OperationalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    /// Primary exchange for market data
    pub primary: Exchange,

    /// Secondary exchange (failover)
    pub secondary: Exchange,

    /// Tertiary exchange (second failover)
    pub tertiary: Exchange,

    /// Binance WebSocket endpoint
    pub binance_ws_url: String,

    /// Bybit WebSocket endpoint
    pub bybit_ws_url: String,

    /// Kucoin REST API base URL
    pub kucoin_rest_url: String,

    /// Coinbase WebSocket endpoint
    pub coinbase_ws_url: String,

    /// Kraken WebSocket endpoint
    pub kraken_ws_url: String,

    /// OKX WebSocket endpoint
    pub okx_ws_url: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Exchange {
    Binance,
    Bybit,
    Kucoin,
    Coinbase,
    Kraken,
    Okx,
}

impl std::fmt::Display for Exchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exchange::Binance => write!(f, "binance"),
            Exchange::Bybit => write!(f, "bybit"),
            Exchange::Kucoin => write!(f, "kucoin"),
            Exchange::Coinbase => write!(f, "coinbase"),
            Exchange::Kraken => write!(f, "kraken"),
            Exchange::Okx => write!(f, "okx"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestDbConfig {
    /// QuestDB host
    pub host: String,

    /// QuestDB ILP port (TCP)
    pub ilp_port: u16,

    /// QuestDB HTTP port (for queries)
    pub http_port: u16,

    /// Buffer size for ILP batching (in lines)
    pub buffer_size: usize,

    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,

    /// Key prefix for namespacing
    pub key_prefix: String,

    /// Connection pool size
    pub pool_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable Fear & Greed Index ingestion
    pub enable_fear_greed: bool,

    /// Enable ETF Net Flows ingestion
    pub enable_etf_flows: bool,

    /// Enable Volatility (DVOL) ingestion
    pub enable_volatility: bool,

    /// Enable Altcoin Season Index
    pub enable_altcoin_season: bool,

    /// Fear & Greed API URL
    pub fear_greed_url: String,

    /// ETF Flows scraper URL (Farside Investors)
    pub etf_flows_url: String,

    /// Deribit DVOL API URL
    pub dvol_url: String,

    /// Altcoin Season Index URL
    pub altcoin_season_url: String,

    /// Polling interval for metrics (in seconds)
    pub poll_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalConfig {
    /// Enable backfilling on startup
    pub enable_backfill: bool,

    /// Maximum gap duration to backfill (in hours)
    pub max_backfill_hours: u64,

    /// Enable automatic exchange failover
    pub enable_failover: bool,

    /// Latency threshold for failover (in milliseconds)
    pub failover_latency_threshold_ms: u64,

    /// Number of consecutive errors before failover
    pub failover_error_count: u32,

    /// Health check interval (in seconds)
    pub health_check_interval_secs: u64,

    /// Log level
    pub log_level: String,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        // Load .env file if present (non-Docker environments)
        let _ = dotenvy::dotenv();

        let assets = env::var("ASSETS")
            .unwrap_or_else(|_| "BTC,ETH,SOL".to_string())
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect();

        let primary = Self::parse_exchange(
            &env::var("PRIMARY_EXCHANGE").unwrap_or_else(|_| "binance".to_string()),
        )?;
        let secondary = Self::parse_exchange(
            &env::var("SECONDARY_EXCHANGE").unwrap_or_else(|_| "bybit".to_string()),
        )?;
        let tertiary = Self::parse_exchange(
            &env::var("TERTIARY_EXCHANGE").unwrap_or_else(|_| "kucoin".to_string()),
        )?;

        Ok(Config {
            assets,
            exchanges: ExchangeConfig {
                primary,
                secondary,
                tertiary,
                binance_ws_url: env::var("BINANCE_WS_URL")
                    .unwrap_or_else(|_| "wss://stream.binance.com:9443/ws".to_string()),
                bybit_ws_url: env::var("BYBIT_WS_URL")
                    .unwrap_or_else(|_| "wss://stream.bybit.com/v5/public/spot".to_string()),
                kucoin_rest_url: env::var("KUCOIN_REST_URL")
                    .unwrap_or_else(|_| "https://api.kucoin.com".to_string()),
                coinbase_ws_url: env::var("COINBASE_WS_URL")
                    .unwrap_or_else(|_| "wss://advanced-trade-ws.coinbase.com".to_string()),
                kraken_ws_url: env::var("KRAKEN_WS_URL")
                    .unwrap_or_else(|_| "wss://ws.kraken.com/v2".to_string()),
                okx_ws_url: env::var("OKX_WS_URL")
                    .unwrap_or_else(|_| "wss://ws.okx.com:8443/ws/v5/public".to_string()),
            },
            questdb: QuestDbConfig {
                host: env::var("QUESTDB_HOST").unwrap_or_else(|_| "questdb".to_string()),
                ilp_port: env::var("QUESTDB_ILP_PORT")
                    .unwrap_or_else(|_| "9009".to_string())
                    .parse()
                    .context("Invalid QUESTDB_ILP_PORT")?,
                http_port: env::var("QUESTDB_HTTP_PORT")
                    .unwrap_or_else(|_| "9000".to_string())
                    .parse()
                    .context("Invalid QUESTDB_HTTP_PORT")?,
                buffer_size: env::var("QUESTDB_BUFFER_SIZE")
                    .unwrap_or_else(|_| "1000".to_string())
                    .parse()
                    .context("Invalid QUESTDB_BUFFER_SIZE")?,
                flush_interval_ms: env::var("QUESTDB_FLUSH_INTERVAL_MS")
                    .unwrap_or_else(|_| "100".to_string())
                    .parse()
                    .context("Invalid QUESTDB_FLUSH_INTERVAL_MS")?,
            },
            redis: RedisConfig {
                url: env::var("REDIS_URL").unwrap_or_else(|_| "redis://redis:6379".to_string()),
                key_prefix: env::var("REDIS_KEY_PREFIX")
                    .unwrap_or_else(|_| "data_factory".to_string()),
                pool_size: env::var("REDIS_POOL_SIZE")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .context("Invalid REDIS_POOL_SIZE")?,
            },
            metrics: MetricsConfig {
                enable_fear_greed: env::var("ENABLE_FEAR_GREED")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                enable_etf_flows: env::var("ENABLE_ETF_FLOWS")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                enable_volatility: env::var("ENABLE_VOLATILITY")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                enable_altcoin_season: env::var("ENABLE_ALTCOIN_SEASON")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(false),
                fear_greed_url: env::var("FEAR_GREED_URL")
                    .unwrap_or_else(|_| "https://api.alternative.me/fng/".to_string()),
                etf_flows_url: env::var("ETF_FLOWS_URL")
                    .unwrap_or_else(|_| "https://farside.co.uk/btc/".to_string()),
                dvol_url: env::var("DVOL_URL").unwrap_or_else(|_| {
                    "https://www.deribit.com/api/v2/public/get_volatility_index_data".to_string()
                }),
                altcoin_season_url: env::var("ALTCOIN_SEASON_URL").unwrap_or_else(|_| {
                    "https://www.blockchaincenter.net/altcoin-season-index/".to_string()
                }),
                poll_interval_secs: env::var("METRICS_POLL_INTERVAL_SECS")
                    .unwrap_or_else(|_| "300".to_string()) // 5 minutes default
                    .parse()
                    .context("Invalid METRICS_POLL_INTERVAL_SECS")?,
            },
            operational: OperationalConfig {
                enable_backfill: env::var("ENABLE_BACKFILL")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                max_backfill_hours: env::var("MAX_BACKFILL_HOURS")
                    .unwrap_or_else(|_| "24".to_string())
                    .parse()
                    .context("Invalid MAX_BACKFILL_HOURS")?,
                enable_failover: env::var("ENABLE_FAILOVER")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                failover_latency_threshold_ms: env::var("FAILOVER_LATENCY_THRESHOLD_MS")
                    .unwrap_or_else(|_| "500".to_string())
                    .parse()
                    .context("Invalid FAILOVER_LATENCY_THRESHOLD_MS")?,
                failover_error_count: env::var("FAILOVER_ERROR_COUNT")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .context("Invalid FAILOVER_ERROR_COUNT")?,
                health_check_interval_secs: env::var("HEALTH_CHECK_INTERVAL_SECS")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()
                    .context("Invalid HEALTH_CHECK_INTERVAL_SECS")?,
                log_level: env::var("RUST_LOG")
                    .unwrap_or_else(|_| "info,fks_ruby=debug".to_string()),
            },
        })
    }

    fn parse_exchange(s: &str) -> Result<Exchange> {
        match s.to_lowercase().as_str() {
            "binance" => Ok(Exchange::Binance),
            "bybit" => Ok(Exchange::Bybit),
            "kucoin" => Ok(Exchange::Kucoin),
            "coinbase" => Ok(Exchange::Coinbase),
            "kraken" => Ok(Exchange::Kraken),
            "okx" => Ok(Exchange::Okx),
            _ => Err(anyhow::anyhow!("Unknown exchange: {}", s)),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.assets.is_empty() {
            anyhow::bail!("At least one asset must be configured");
        }

        if self.exchanges.primary == self.exchanges.secondary {
            anyhow::bail!("Primary and secondary exchanges must be different");
        }

        if self.questdb.buffer_size == 0 {
            anyhow::bail!("QuestDB buffer size must be greater than 0");
        }

        if self.questdb.flush_interval_ms == 0 {
            anyhow::bail!("QuestDB flush interval must be greater than 0");
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_exchange_display() {
        assert_eq!(Exchange::Binance.to_string(), "binance");
        assert_eq!(Exchange::Bybit.to_string(), "bybit");
        assert_eq!(Exchange::Kucoin.to_string(), "kucoin");
    }

    #[test]
    fn test_parse_exchange() {
        assert!(matches!(
            Config::parse_exchange("binance").unwrap(),
            Exchange::Binance
        ));
        assert!(matches!(
            Config::parse_exchange("BYBIT").unwrap(),
            Exchange::Bybit
        ));
        assert!(Config::parse_exchange("unknown").is_err());
    }

    #[test]
    fn test_read_secret_from_env_fallback() {
        // Test fallback to environment variable
        // SAFETY: env var mutation is inherently racy in multi-threaded programs;
        // acceptable here because this test does not run concurrently with code
        // that reads these variables.
        unsafe { env::set_var("TEST_SECRET", "test_value") };
        let result = read_secret("TEST_SECRET_FILE", Some("TEST_SECRET"));
        assert_eq!(result, Some("test_value".to_string()));
        unsafe { env::remove_var("TEST_SECRET") };
    }

    #[test]
    fn test_read_secret_none_when_missing() {
        // Test None when neither file nor env var exists
        let result = read_secret("NONEXISTENT_FILE", Some("NONEXISTENT_VAR"));
        assert_eq!(result, None);
    }

    #[test]
    fn test_api_key_pair_debug_redacts_secrets() {
        let pair = ApiKeyPair {
            api_key: "secret_key".to_string(),
            api_secret: "secret_value".to_string(),
        };
        let debug_output = format!("{:?}", pair);
        assert!(!debug_output.contains("secret_key"));
        assert!(!debug_output.contains("secret_value"));
        assert!(debug_output.contains("REDACTED"));
    }

    #[test]
    fn test_kucoin_credentials_debug_redacts_secrets() {
        let creds = KucoinCredentials {
            api_key: "secret_key".to_string(),
            api_secret: "secret_value".to_string(),
            passphrase: "secret_pass".to_string(),
        };
        let debug_output = format!("{:?}", creds);
        assert!(!debug_output.contains("secret_key"));
        assert!(!debug_output.contains("secret_value"));
        assert!(!debug_output.contains("secret_pass"));
        assert!(debug_output.contains("REDACTED"));
    }
}
