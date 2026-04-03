//! Simulation Environment Configuration
//!
//! Defines configuration for different simulation modes:
//! - Backtest: Historical data replay with zero-lookahead
//! - Forward Test: Live data with simulated execution (paper trading)
//! - Live: Real execution via exchange APIs

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Simulation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimMode {
    /// Historical data replay for backtesting
    Backtest,
    /// Live data with simulated execution (paper trading)
    ForwardTest,
    /// Real execution via exchange APIs
    Live,
}

impl Default for SimMode {
    fn default() -> Self {
        Self::Backtest
    }
}

impl std::fmt::Display for SimMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimMode::Backtest => write!(f, "backtest"),
            SimMode::ForwardTest => write!(f, "forward_test"),
            SimMode::Live => write!(f, "live"),
        }
    }
}

/// Data source for simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// Parquet file(s) with historical data
    Parquet(PathBuf),
    /// CSV file(s) with historical data
    Csv(PathBuf),
    /// QuestDB historical data query
    QuestDB {
        host: String,
        port: u16,
        query: String,
    },
    /// Live WebSocket feeds from exchanges
    Live {
        exchanges: Vec<String>,
        symbols: Vec<String>,
    },
    /// Recorded data replay from QuestDB
    Recorded {
        host: String,
        port: u16,
        start_time: chrono::DateTime<chrono::Utc>,
        end_time: chrono::DateTime<chrono::Utc>,
        symbols: Vec<String>,
    },
}

impl Default for DataSource {
    fn default() -> Self {
        Self::Live {
            exchanges: vec!["kraken".to_string()],
            symbols: vec!["BTC/USDT".to_string()],
        }
    }
}

/// Exchange credentials for live trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeCredentials {
    pub exchange: String,
    pub api_key: String,
    pub api_secret: String,
    pub testnet: bool,
}

/// Execution simulation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Slippage in basis points (1 bp = 0.01%)
    pub slippage_bps: f64,
    /// Commission/fee in basis points
    pub commission_bps: f64,
    /// Simulated fill delay in milliseconds
    pub fill_delay_ms: u64,
    /// Enable realistic slippage modeling
    pub enable_slippage: bool,
    /// Enable partial fills simulation
    pub enable_partial_fills: bool,
    /// Partial fill probability (0.0-1.0)
    pub partial_fill_probability: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            slippage_bps: 5.0,
            commission_bps: 6.0,
            fill_delay_ms: 50,
            enable_slippage: true,
            enable_partial_fills: false,
            partial_fill_probability: 0.1,
        }
    }
}

/// Risk management settings for simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum position size as percentage of account
    pub max_position_pct: f64,
    /// Maximum number of concurrent positions
    pub max_positions: usize,
    /// Daily loss limit as percentage of account
    pub daily_loss_limit_pct: f64,
    /// Maximum drawdown before stopping
    pub max_drawdown_pct: f64,
    /// Enable circuit breaker
    pub enable_circuit_breaker: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_pct: 10.0,
            max_positions: 5,
            daily_loss_limit_pct: 5.0,
            max_drawdown_pct: 20.0,
            enable_circuit_breaker: true,
        }
    }
}

/// Data recording configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingConfig {
    /// Enable data recording
    pub enabled: bool,
    /// QuestDB host for recording
    pub questdb_host: String,
    /// QuestDB port for recording
    pub questdb_port: u16,
    /// Record tick data
    pub record_ticks: bool,
    /// Record trade data
    pub record_trades: bool,
    /// Record order book snapshots
    pub record_orderbook: bool,
    /// Orderbook snapshot interval
    pub orderbook_interval: Duration,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            questdb_host: "localhost".to_string(),
            questdb_port: 9009,
            record_ticks: true,
            record_trades: true,
            record_orderbook: false,
            orderbook_interval: Duration::from_secs(1),
        }
    }
}

/// Metrics and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Metrics HTTP port
    pub port: u16,
    /// Metrics collection interval
    pub interval: Duration,
    /// Export to Prometheus
    pub prometheus_enabled: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8080,
            interval: Duration::from_secs(10),
            prometheus_enabled: true,
        }
    }
}

/// Main simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// Simulation mode
    pub mode: SimMode,
    /// Data source
    pub data_source: DataSource,
    /// Initial account balance (USDT)
    pub initial_balance: Decimal,
    /// Trading symbols
    pub symbols: Vec<String>,
    /// Exchanges to use
    pub exchanges: Vec<String>,
    /// Execution configuration
    pub execution: ExecutionConfig,
    /// Risk configuration
    pub risk: RiskConfig,
    /// Recording configuration
    pub recording: RecordingConfig,
    /// Metrics configuration
    pub metrics: MetricsConfig,
    /// Exchange credentials (for live mode)
    pub credentials: HashMap<String, ExchangeCredentials>,
    /// Backtest-specific: start time
    pub backtest_start: Option<chrono::DateTime<chrono::Utc>>,
    /// Backtest-specific: end time
    pub backtest_end: Option<chrono::DateTime<chrono::Utc>>,
    /// Forward test duration
    pub forward_test_duration: Option<Duration>,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            mode: SimMode::Backtest,
            data_source: DataSource::default(),
            initial_balance: Decimal::from(10_000),
            symbols: vec!["BTC/USDT".to_string()],
            exchanges: vec!["kraken".to_string()],
            execution: ExecutionConfig::default(),
            risk: RiskConfig::default(),
            recording: RecordingConfig::default(),
            metrics: MetricsConfig::default(),
            credentials: HashMap::new(),
            backtest_start: None,
            backtest_end: None,
            forward_test_duration: None,
            verbose: false,
        }
    }
}

impl SimConfig {
    /// Create a new backtest configuration
    pub fn backtest() -> SimConfigBuilder {
        SimConfigBuilder::new(SimMode::Backtest)
    }

    /// Create a new forward test configuration
    pub fn forward_test() -> SimConfigBuilder {
        SimConfigBuilder::new(SimMode::ForwardTest)
    }

    /// Create a new live trading configuration
    pub fn live() -> SimConfigBuilder {
        SimConfigBuilder::new(SimMode::Live)
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenvy::dotenv().ok();

        let mode = std::env::var("SIM_MODE")
            .unwrap_or_else(|_| "backtest".to_string())
            .parse::<SimMode>()
            .map_err(|e| ConfigError::InvalidValue("SIM_MODE".to_string(), e))?;

        let mut builder = match mode {
            SimMode::Backtest => Self::backtest(),
            SimMode::ForwardTest => Self::forward_test(),
            SimMode::Live => Self::live(),
        };

        // Parse symbols
        if let Ok(symbols) = std::env::var("SIM_SYMBOLS") {
            let symbols: Vec<String> = symbols.split(',').map(|s| s.trim().to_string()).collect();
            builder = builder.with_symbols(symbols);
        }

        // Parse exchanges
        if let Ok(exchanges) = std::env::var("SIM_EXCHANGES") {
            let exchanges: Vec<String> =
                exchanges.split(',').map(|s| s.trim().to_string()).collect();
            builder = builder.with_exchanges(exchanges);
        }

        // Parse initial balance
        if let Ok(balance) = std::env::var("SIM_INITIAL_BALANCE") {
            let balance: f64 = balance.parse().map_err(|_| {
                ConfigError::InvalidValue(
                    "SIM_INITIAL_BALANCE".to_string(),
                    "must be a number".to_string(),
                )
            })?;
            builder = builder.with_initial_balance(balance);
        }

        // Parse data source
        if let Ok(data_path) = std::env::var("SIM_DATA_PATH") {
            let path = PathBuf::from(data_path);
            if path.extension().map(|e| e == "parquet").unwrap_or(false) {
                builder = builder.with_data_source(DataSource::Parquet(path));
            } else {
                builder = builder.with_data_source(DataSource::Csv(path));
            }
        }

        // Parse execution config
        if let Ok(slippage) = std::env::var("SIM_SLIPPAGE_BPS") {
            let slippage: f64 = slippage.parse().map_err(|_| {
                ConfigError::InvalidValue(
                    "SIM_SLIPPAGE_BPS".to_string(),
                    "must be a number".to_string(),
                )
            })?;
            builder = builder.with_slippage_bps(slippage);
        }

        if let Ok(commission) = std::env::var("SIM_COMMISSION_BPS") {
            let commission: f64 = commission.parse().map_err(|_| {
                ConfigError::InvalidValue(
                    "SIM_COMMISSION_BPS".to_string(),
                    "must be a number".to_string(),
                )
            })?;
            builder = builder.with_commission_bps(commission);
        }

        // Parse recording config
        if let Ok(record) = std::env::var("SIM_RECORD_DATA") {
            let record: bool = record.parse().map_err(|_| {
                ConfigError::InvalidValue(
                    "SIM_RECORD_DATA".to_string(),
                    "must be true/false".to_string(),
                )
            })?;
            builder = builder.with_record_data(record);
        }

        // Parse credentials for live mode
        if mode == SimMode::Live {
            if let Ok(api_key) = std::env::var("KRAKEN_API_KEY") {
                let api_secret = std::env::var("KRAKEN_API_SECRET")
                    .map_err(|_| ConfigError::MissingRequired("KRAKEN_API_SECRET".to_string()))?;
                builder = builder.with_kraken_credentials(api_key, api_secret);
            }
        }

        // Parse verbose
        if let Ok(verbose) = std::env::var("SIM_VERBOSE") {
            let verbose: bool = verbose.parse().unwrap_or(false);
            builder = builder.with_verbose(verbose);
        }

        builder.build()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate mode-specific requirements
        match self.mode {
            SimMode::Backtest => {
                // Backtest requires historical data source
                match &self.data_source {
                    DataSource::Live { .. } => {
                        return Err(ConfigError::InvalidForMode(
                            "Live data source".to_string(),
                            "backtest".to_string(),
                        ));
                    }
                    _ => {}
                }
            }
            SimMode::ForwardTest => {
                // Forward test requires live or recorded data
                match &self.data_source {
                    DataSource::Parquet(_) | DataSource::Csv(_) => {
                        return Err(ConfigError::InvalidForMode(
                            "Static file data source".to_string(),
                            "forward_test".to_string(),
                        ));
                    }
                    _ => {}
                }
            }
            SimMode::Live => {
                // Live mode requires credentials
                if self.credentials.is_empty() {
                    return Err(ConfigError::MissingRequired(
                        "Exchange credentials for live mode".to_string(),
                    ));
                }
            }
        }

        // Validate symbols
        if self.symbols.is_empty() {
            return Err(ConfigError::MissingRequired("symbols".to_string()));
        }

        // Validate exchanges
        if self.exchanges.is_empty() {
            return Err(ConfigError::MissingRequired("exchanges".to_string()));
        }

        // Validate initial balance
        if self.initial_balance <= Decimal::ZERO {
            return Err(ConfigError::InvalidValue(
                "initial_balance".to_string(),
                "must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

impl std::str::FromStr for SimMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "backtest" | "bt" => Ok(SimMode::Backtest),
            "forward_test" | "forward" | "ft" | "paper" => Ok(SimMode::ForwardTest),
            "live" | "prod" | "production" => Ok(SimMode::Live),
            _ => Err(format!("Unknown simulation mode: {}", s)),
        }
    }
}

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing required configuration: {0}")]
    MissingRequired(String),

    #[error("Invalid value for {0}: {1}")]
    InvalidValue(String, String),

    #[error("{0} is not valid for {1} mode")]
    InvalidForMode(String, String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Builder for SimConfig
#[derive(Debug, Clone)]
pub struct SimConfigBuilder {
    config: SimConfig,
}

impl SimConfigBuilder {
    /// Create a new builder with the specified mode
    pub fn new(mode: SimMode) -> Self {
        let mut config = SimConfig::default();
        config.mode = mode;

        // Set mode-specific defaults
        match mode {
            SimMode::Backtest => {
                config.execution.fill_delay_ms = 0; // No delay in backtest
                config.recording.enabled = false;
            }
            SimMode::ForwardTest => {
                config.data_source = DataSource::Live {
                    exchanges: vec![
                        "kraken".to_string(),
                        "binance".to_string(),
                        "bybit".to_string(),
                    ],
                    symbols: vec!["BTC/USDT".to_string(), "ETH/USDT".to_string()],
                };
                config.recording.enabled = true;
            }
            SimMode::Live => {
                config.data_source = DataSource::Live {
                    exchanges: vec!["kraken".to_string()],
                    symbols: vec!["BTC/USDT".to_string()],
                };
                config.recording.enabled = true;
                config.execution.enable_slippage = false; // Real slippage from exchange
            }
        }

        Self { config }
    }

    /// Set data source
    pub fn with_data_source(mut self, source: DataSource) -> Self {
        self.config.data_source = source;
        self
    }

    /// Set initial balance
    pub fn with_initial_balance(mut self, balance: f64) -> Self {
        self.config.initial_balance = Decimal::try_from(balance).unwrap_or(Decimal::from(10_000));
        self
    }

    /// Set trading symbols
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.config.symbols = symbols.clone();
        // Update data source if it's Live
        if let DataSource::Live { exchanges, .. } = &self.config.data_source {
            self.config.data_source = DataSource::Live {
                exchanges: exchanges.clone(),
                symbols,
            };
        }
        self
    }

    /// Set exchanges
    pub fn with_exchanges(mut self, exchanges: Vec<String>) -> Self {
        self.config.exchanges = exchanges.clone();
        // Update data source if it's Live
        if let DataSource::Live { symbols, .. } = &self.config.data_source {
            self.config.data_source = DataSource::Live {
                exchanges,
                symbols: symbols.clone(),
            };
        }
        self
    }

    /// Set slippage in basis points
    pub fn with_slippage_bps(mut self, slippage: f64) -> Self {
        self.config.execution.slippage_bps = slippage;
        self
    }

    /// Set commission in basis points
    pub fn with_commission_bps(mut self, commission: f64) -> Self {
        self.config.execution.commission_bps = commission;
        self
    }

    /// Set fill delay
    pub fn with_fill_delay_ms(mut self, delay: u64) -> Self {
        self.config.execution.fill_delay_ms = delay;
        self
    }

    /// Enable or disable slippage simulation
    pub fn with_slippage_enabled(mut self, enabled: bool) -> Self {
        self.config.execution.enable_slippage = enabled;
        self
    }

    /// Enable or disable partial fills
    pub fn with_partial_fills(mut self, enabled: bool, probability: f64) -> Self {
        self.config.execution.enable_partial_fills = enabled;
        self.config.execution.partial_fill_probability = probability;
        self
    }

    /// Set execution configuration
    pub fn with_execution_config(mut self, config: ExecutionConfig) -> Self {
        self.config.execution = config;
        self
    }

    /// Set risk configuration
    pub fn with_risk_config(mut self, config: RiskConfig) -> Self {
        self.config.risk = config;
        self
    }

    /// Enable or disable data recording
    pub fn with_record_data(mut self, enabled: bool) -> Self {
        self.config.recording.enabled = enabled;
        self
    }

    /// Set recording configuration
    pub fn with_recording_config(mut self, config: RecordingConfig) -> Self {
        self.config.recording = config;
        self
    }

    /// Set QuestDB configuration for recording
    pub fn with_questdb(mut self, host: &str, port: u16) -> Self {
        self.config.recording.questdb_host = host.to_string();
        self.config.recording.questdb_port = port;
        self
    }

    /// Set metrics configuration
    pub fn with_metrics_config(mut self, config: MetricsConfig) -> Self {
        self.config.metrics = config;
        self
    }

    /// Set metrics port
    pub fn with_metrics_port(mut self, port: u16) -> Self {
        self.config.metrics.port = port;
        self
    }

    /// Add Kraken credentials
    pub fn with_kraken_credentials(mut self, api_key: String, api_secret: String) -> Self {
        self.config.credentials.insert(
            "kraken".to_string(),
            ExchangeCredentials {
                exchange: "kraken".to_string(),
                api_key,
                api_secret,
                testnet: false,
            },
        );
        self
    }

    /// Add Bybit credentials
    pub fn with_bybit_credentials(
        mut self,
        api_key: String,
        api_secret: String,
        testnet: bool,
    ) -> Self {
        self.config.credentials.insert(
            "bybit".to_string(),
            ExchangeCredentials {
                exchange: "bybit".to_string(),
                api_key,
                api_secret,
                testnet,
            },
        );
        self
    }

    /// Add Binance credentials
    pub fn with_binance_credentials(mut self, api_key: String, api_secret: String) -> Self {
        self.config.credentials.insert(
            "binance".to_string(),
            ExchangeCredentials {
                exchange: "binance".to_string(),
                api_key,
                api_secret,
                testnet: false,
            },
        );
        self
    }

    /// Set backtest time range
    pub fn with_backtest_range(
        mut self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        self.config.backtest_start = Some(start);
        self.config.backtest_end = Some(end);
        self
    }

    /// Set forward test duration
    pub fn with_forward_test_duration(mut self, duration: Duration) -> Self {
        self.config.forward_test_duration = Some(duration);
        self
    }

    /// Set verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<SimConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation (for testing)
    pub fn build_unchecked(self) -> SimConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_config() {
        let config = SimConfig::backtest()
            .with_data_source(DataSource::Parquet(PathBuf::from("data/test.parquet")))
            .with_initial_balance(50_000.0)
            .with_slippage_bps(3.0)
            .with_symbols(vec!["BTC/USDT".to_string()])
            .build_unchecked();

        assert_eq!(config.mode, SimMode::Backtest);
        assert_eq!(config.initial_balance, Decimal::from(50_000));
        assert_eq!(config.execution.slippage_bps, 3.0);
        assert!(!config.recording.enabled);
    }

    #[test]
    fn test_forward_test_config() {
        let config = SimConfig::forward_test()
            .with_exchanges(vec!["kraken".to_string(), "binance".to_string()])
            .with_symbols(vec!["BTC/USDT".to_string(), "ETH/USDT".to_string()])
            .with_record_data(true)
            .build_unchecked();

        assert_eq!(config.mode, SimMode::ForwardTest);
        assert!(config.recording.enabled);
        assert_eq!(config.exchanges.len(), 2);
    }

    #[test]
    fn test_live_config() {
        let config = SimConfig::live()
            .with_kraken_credentials("key".to_string(), "secret".to_string())
            .with_symbols(vec!["BTC/USDT".to_string()])
            .build_unchecked();

        assert_eq!(config.mode, SimMode::Live);
        assert!(config.credentials.contains_key("kraken"));
        assert!(!config.execution.enable_slippage); // Disabled for live
    }

    #[test]
    fn test_sim_mode_parsing() {
        assert_eq!("backtest".parse::<SimMode>().unwrap(), SimMode::Backtest);
        assert_eq!("bt".parse::<SimMode>().unwrap(), SimMode::Backtest);
        assert_eq!(
            "forward_test".parse::<SimMode>().unwrap(),
            SimMode::ForwardTest
        );
        assert_eq!("paper".parse::<SimMode>().unwrap(), SimMode::ForwardTest);
        assert_eq!("live".parse::<SimMode>().unwrap(), SimMode::Live);
        assert_eq!("prod".parse::<SimMode>().unwrap(), SimMode::Live);
    }

    #[test]
    fn test_validation_backtest_requires_historical_data() {
        let config = SimConfig::backtest()
            .with_data_source(DataSource::Live {
                exchanges: vec!["kraken".to_string()],
                symbols: vec!["BTC/USDT".to_string()],
            })
            .build_unchecked();

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_live_requires_credentials() {
        let config = SimConfig::live()
            .with_symbols(vec!["BTC/USDT".to_string()])
            .build_unchecked();

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_default_config() {
        let config = SimConfig::default();
        assert_eq!(config.mode, SimMode::Backtest);
        assert_eq!(config.initial_balance, Decimal::from(10_000));
        assert!(config.execution.enable_slippage);
    }
}
