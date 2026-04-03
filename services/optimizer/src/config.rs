//! Optimizer Service Configuration
//!
//! Configuration is loaded from environment variables with sensible defaults.
//! All settings can be overridden via environment variables.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Default values
pub const DEFAULT_DATA_DIR: &str = "/data";
#[allow(dead_code)]
pub const DEFAULT_ASSETS: &str = "BTC,ETH,SOL";
pub const DEFAULT_N_TRIALS: usize = 100;
pub const DEFAULT_OPTIMIZATION_INTERVAL: &str = "6h";
pub const DEFAULT_MIN_DATA_DAYS: u32 = 7;
pub const DEFAULT_PREFERRED_INTERVAL_MINUTES: u32 = 60;
pub const DEFAULT_DATA_COLLECTION_ENABLED: bool = true;
pub const DEFAULT_DATA_COLLECTION_INTERVAL_MINUTES: u64 = 5;
pub const DEFAULT_REDIS_URL: &str = "redis://localhost:6379";
pub const DEFAULT_REDIS_INSTANCE_ID: &str = "default";
pub const DEFAULT_METRICS_PORT: u16 = 9092;
pub const DEFAULT_RUN_ON_START: bool = true;
pub const DEFAULT_KRAKEN_API_URL: &str = "https://api.kraken.com";

/// Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerServiceConfig {
    /// Data directory for OHLC storage, params, and results
    pub data_dir: PathBuf,

    /// Assets to optimize (e.g., ["BTC", "ETH", "SOL"])
    pub assets: Vec<String>,

    /// Number of optimization trials per asset
    pub n_trials: usize,

    /// Optimization interval (e.g., "6h", "30m", "1d")
    pub optimization_interval: String,

    /// Minimum days of data required for optimization
    pub min_data_days: u32,

    /// Preferred OHLC interval for backtesting (minutes)
    pub preferred_interval_minutes: u32,

    /// Collection intervals to fetch (e.g., [1, 5, 15, 60, 240, 1440])
    pub collection_intervals: Vec<u32>,

    /// Enable automatic data collection
    pub data_collection_enabled: bool,

    /// Data collection interval in minutes
    pub data_collection_interval_minutes: u64,

    /// Redis URL for param publishing
    pub redis_url: String,

    /// Redis instance ID for key prefixes
    pub redis_instance_id: String,

    /// Prometheus metrics port
    pub metrics_port: u16,

    /// Run optimization immediately on startup
    pub run_on_start: bool,

    /// Kraken API base URL
    pub kraken_api_url: String,

    /// Number of parallel optimization jobs
    pub n_jobs: usize,

    /// Request timeout for API calls
    pub request_timeout: Duration,

    /// Rate limit: requests per second
    pub rate_limit_per_second: f64,

    /// Historical data lookback days for initial collection
    pub historical_days: u32,
}

impl Default for OptimizerServiceConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from(DEFAULT_DATA_DIR),
            assets: vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()],
            n_trials: DEFAULT_N_TRIALS,
            optimization_interval: DEFAULT_OPTIMIZATION_INTERVAL.to_string(),
            min_data_days: DEFAULT_MIN_DATA_DAYS,
            preferred_interval_minutes: DEFAULT_PREFERRED_INTERVAL_MINUTES,
            collection_intervals: vec![1, 5, 15, 60, 240, 1440],
            data_collection_enabled: DEFAULT_DATA_COLLECTION_ENABLED,
            data_collection_interval_minutes: DEFAULT_DATA_COLLECTION_INTERVAL_MINUTES,
            redis_url: DEFAULT_REDIS_URL.to_string(),
            redis_instance_id: DEFAULT_REDIS_INSTANCE_ID.to_string(),
            metrics_port: DEFAULT_METRICS_PORT,
            run_on_start: DEFAULT_RUN_ON_START,
            kraken_api_url: DEFAULT_KRAKEN_API_URL.to_string(),
            n_jobs: num_cpus(),
            request_timeout: Duration::from_secs(30),
            rate_limit_per_second: 1.0,
            historical_days: 30,
        }
    }
}

impl OptimizerServiceConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        // Load .env file if present
        let _ = dotenvy::dotenv();

        let mut config = Self::default();

        // Data directory
        if let Ok(dir) = std::env::var("DATA_DIR") {
            config.data_dir = PathBuf::from(dir);
        }

        // Assets to optimize
        if let Ok(assets) = std::env::var("OPTIMIZE_ASSETS") {
            config.assets = assets
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .filter(|s| !s.is_empty())
                .collect();
        }

        // Number of trials
        if let Ok(n) = std::env::var("OPTUNA_TRIALS") {
            config.n_trials = n.parse().context("Invalid OPTUNA_TRIALS value")?;
        }

        // Optimization interval
        if let Ok(interval) = std::env::var("OPTIMIZATION_INTERVAL") {
            config.optimization_interval = interval;
        }

        // Minimum data days
        if let Ok(days) = std::env::var("MIN_DATA_DAYS") {
            config.min_data_days = days.parse().context("Invalid MIN_DATA_DAYS value")?;
        }

        // Preferred interval
        if let Ok(interval) = std::env::var("PREFERRED_INTERVAL_MINUTES") {
            config.preferred_interval_minutes = interval
                .parse()
                .context("Invalid PREFERRED_INTERVAL_MINUTES value")?;
        }

        // Collection intervals
        if let Ok(intervals) = std::env::var("COLLECTION_INTERVALS") {
            config.collection_intervals = intervals
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
        }

        // Data collection enabled
        if let Ok(enabled) = std::env::var("DATA_COLLECTION_ENABLED") {
            config.data_collection_enabled = enabled.to_lowercase() == "true";
        }

        // Data collection interval
        if let Ok(interval) = std::env::var("DATA_COLLECTION_INTERVAL_MINUTES") {
            config.data_collection_interval_minutes = interval
                .parse()
                .context("Invalid DATA_COLLECTION_INTERVAL_MINUTES value")?;
        }

        // Redis URL
        if let Ok(url) = std::env::var("REDIS_URL") {
            config.redis_url = url;
        }

        // Redis instance ID
        if let Ok(id) = std::env::var("REDIS_INSTANCE_ID") {
            config.redis_instance_id = id;
        }

        // Metrics port
        if let Ok(port) = std::env::var("METRICS_PORT") {
            config.metrics_port = port.parse().context("Invalid METRICS_PORT value")?;
        }

        // Run on start
        if let Ok(run) = std::env::var("RUN_ON_START") {
            config.run_on_start = run.to_lowercase() == "true";
        }

        // Kraken API URL
        if let Ok(url) = std::env::var("KRAKEN_API_URL") {
            config.kraken_api_url = url;
        }

        // Number of parallel jobs
        if let Ok(n) = std::env::var("N_JOBS") {
            config.n_jobs = n.parse().context("Invalid N_JOBS value")?;
        }

        // Request timeout
        if let Ok(secs) = std::env::var("REQUEST_TIMEOUT_SECS") {
            let secs: u64 = secs.parse().context("Invalid REQUEST_TIMEOUT_SECS value")?;
            config.request_timeout = Duration::from_secs(secs);
        }

        // Rate limit
        if let Ok(rate) = std::env::var("RATE_LIMIT_PER_SECOND") {
            config.rate_limit_per_second = rate
                .parse()
                .context("Invalid RATE_LIMIT_PER_SECOND value")?;
        }

        // Historical days
        if let Ok(days) = std::env::var("HISTORICAL_DAYS") {
            config.historical_days = days.parse().context("Invalid HISTORICAL_DAYS value")?;
        }

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.assets.is_empty() {
            anyhow::bail!("At least one asset must be configured");
        }

        if self.n_trials == 0 {
            anyhow::bail!("n_trials must be > 0");
        }

        if self.min_data_days == 0 {
            anyhow::bail!("min_data_days must be > 0");
        }

        if self.collection_intervals.is_empty() {
            anyhow::bail!("At least one collection interval must be configured");
        }

        if !self
            .collection_intervals
            .contains(&self.preferred_interval_minutes)
        {
            anyhow::bail!(
                "preferred_interval_minutes ({}) must be in collection_intervals",
                self.preferred_interval_minutes
            );
        }

        // Parse and validate optimization interval
        parse_interval(&self.optimization_interval)?;

        Ok(())
    }

    /// Get the database path
    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("db").join("ohlc.db")
    }

    /// Get the params output directory
    pub fn params_dir(&self) -> PathBuf {
        self.data_dir.join("optimized_params")
    }

    /// Get the backtest results directory
    pub fn results_dir(&self) -> PathBuf {
        self.data_dir.join("backtest_results")
    }

    /// Get the logs directory
    pub fn logs_dir(&self) -> PathBuf {
        self.data_dir.join("logs")
    }

    /// Ensure all required directories exist
    pub fn ensure_directories(&self) -> Result<()> {
        let dirs = [
            self.data_dir.clone(),
            self.data_dir.join("db"),
            self.params_dir(),
            self.results_dir(),
            self.logs_dir(),
        ];

        for dir in dirs {
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create directory: {}", dir.display()))?;
        }

        Ok(())
    }

    /// Get Redis key for optimized params hash
    pub fn redis_params_key(&self) -> String {
        format!("fks:{}:optimized_params", self.redis_instance_id)
    }

    /// Get Redis channel for param updates
    pub fn redis_updates_channel(&self) -> String {
        format!("fks:{}:param_updates", self.redis_instance_id)
    }

    /// Get optimization interval as Duration
    pub fn optimization_duration(&self) -> Result<Duration> {
        parse_interval(&self.optimization_interval)
    }
}

/// Parse interval string (e.g., "6h", "30m", "1d") into Duration
pub fn parse_interval(interval: &str) -> Result<Duration> {
    let interval = interval.trim().to_lowercase();

    if interval.is_empty() {
        anyhow::bail!("Interval cannot be empty");
    }

    let (value_str, unit) = interval.split_at(interval.len() - 1);
    let value: u64 = value_str
        .parse()
        .with_context(|| format!("Invalid interval value: {}", value_str))?;

    let multiplier = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        "d" => 86400,
        _ => anyhow::bail!("Invalid interval unit '{}'. Use s, m, h, or d", unit),
    };

    Ok(Duration::from_secs(value * multiplier))
}

/// Get number of available CPU cores
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Kraken pair mappings for assets
pub fn get_kraken_pair(asset: &str) -> String {
    match asset.to_uppercase().as_str() {
        "BTC" => "XXBTZUSD".to_string(),
        "ETH" => "XETHZUSD".to_string(),
        "SOL" => "SOLUSD".to_string(),
        "XRP" => "XXRPZUSD".to_string(),
        "DOGE" => "XDGUSD".to_string(),
        "ADA" => "ADAUSD".to_string(),
        "DOT" => "DOTUSD".to_string(),
        "LINK" => "LINKUSD".to_string(),
        "AVAX" => "AVAXUSD".to_string(),
        "MATIC" => "MATICUSD".to_string(),
        "ATOM" => "ATOMUSD".to_string(),
        "UNI" => "UNIUSD".to_string(),
        "LTC" => "XLTCZUSD".to_string(),
        "SHIB" => "SHIBUSD".to_string(),
        "PEPE" => "PEPEUSD".to_string(),
        "ARB" => "ARBUSD".to_string(),
        "OP" => "OPUSD".to_string(),
        "APT" => "APTUSD".to_string(),
        "NEAR" => "NEARUSD".to_string(),
        "FIL" => "FILUSD".to_string(),
        _ => format!("{}USD", asset.to_uppercase()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OptimizerServiceConfig::default();
        assert_eq!(config.assets, vec!["BTC", "ETH", "SOL"]);
        assert_eq!(config.n_trials, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_parse_interval() {
        assert_eq!(parse_interval("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(parse_interval("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(parse_interval("6h").unwrap(), Duration::from_secs(21600));
        assert_eq!(parse_interval("1d").unwrap(), Duration::from_secs(86400));

        assert!(parse_interval("").is_err());
        assert!(parse_interval("5x").is_err());
        assert!(parse_interval("abc").is_err());
    }

    #[test]
    fn test_kraken_pairs() {
        assert_eq!(get_kraken_pair("BTC"), "XXBTZUSD");
        assert_eq!(get_kraken_pair("ETH"), "XETHZUSD");
        assert_eq!(get_kraken_pair("SOL"), "SOLUSD");
        assert_eq!(get_kraken_pair("UNKNOWN"), "UNKNOWNUSD");
    }

    #[test]
    fn test_redis_keys() {
        let config = OptimizerServiceConfig {
            redis_instance_id: "test".to_string(),
            ..Default::default()
        };

        assert_eq!(config.redis_params_key(), "fks:test:optimized_params");
        assert_eq!(config.redis_updates_channel(), "fks:test:param_updates");
    }

    #[test]
    fn test_paths() {
        let config = OptimizerServiceConfig {
            data_dir: PathBuf::from("/data"),
            ..Default::default()
        };

        assert_eq!(config.db_path(), PathBuf::from("/data/db/ohlc.db"));
        assert_eq!(config.params_dir(), PathBuf::from("/data/optimized_params"));
        assert_eq!(
            config.results_dir(),
            PathBuf::from("/data/backtest_results")
        );
    }

    #[test]
    fn test_validation_empty_assets() {
        let config = OptimizerServiceConfig {
            assets: vec![],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_zero_trials() {
        let config = OptimizerServiceConfig {
            n_trials: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_interval_mismatch() {
        let config = OptimizerServiceConfig {
            preferred_interval_minutes: 120,
            collection_intervals: vec![1, 5, 15, 60],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
