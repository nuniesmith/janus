//! Unified configuration for all JANUS modules
//!
//! Supports loading from TOML files with environment variable overrides.
//! Configuration can be loaded from:
//! 1. TOML file (specified by JANUS_CONFIG_PATH or default locations)
//! 2. Environment variables (override file settings)
//!
//! Environment variable naming convention: JANUS_<SECTION>_<KEY>
//! Example: JANUS_PORTS_HTTP=8080, JANUS_MODULES_FORWARD=true

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

/// Default config file search paths
const CONFIG_PATHS: &[&str] = &[
    "config/janus.toml",
    "/etc/janus/janus.toml",
    "janus.toml",
    "infrastructure/config/janus/janus.toml",
];

// ============================================================================
// Main Configuration Structure
// ============================================================================

/// Main configuration for the unified JANUS service
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Service metadata
    pub service: ServiceConfig,

    /// Network ports configuration
    pub ports: PortsConfig,

    /// Host/bind configuration
    pub host: HostConfig,

    /// Module toggles
    pub modules: ModulesConfig,

    /// Redis configuration
    pub redis: RedisConfig,

    /// Database configuration
    pub database: DatabaseConfig,

    /// QuestDB configuration
    pub questdb: QuestDbConfig,

    /// Forward module settings
    pub forward: ForwardConfig,

    /// Risk management settings
    pub risk: RiskConfig,

    /// Backward module settings
    pub backward: BackwardConfig,

    /// CNS module settings
    pub cns: CnsConfig,

    /// Market data configuration
    pub market: MarketConfig,

    /// Assets configuration
    pub assets: AssetsConfig,

    /// Trading configuration
    pub trading: TradingConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Tracing configuration
    pub tracing: TracingConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Alerting configuration
    pub alerting: AlertingConfig,

    /// Parameter hot-reload configuration
    pub param_reload: ParamReloadConfig,

    /// Feature engineering configuration
    pub features: FeaturesConfig,

    /// Security configuration
    pub security: SecurityConfig,

    /// Advanced settings
    pub advanced: AdvancedConfig,

    // =========================================================================
    // Legacy fields for backward compatibility
    // =========================================================================
    /// HTTP/REST API port (legacy - use ports.http instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_port: Option<u16>,
    /// gRPC API port (legacy - use ports.grpc instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grpc_port: Option<u16>,
    /// WebSocket port (legacy - use ports.websocket instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websocket_port: Option<u16>,
    /// Metrics port (legacy - use ports.metrics instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics_port: Option<u16>,
    /// Enable forward module (legacy - use modules.forward instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_forward: Option<bool>,
    /// Enable backward module (legacy - use modules.backward instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_backward: Option<bool>,
    /// Enable CNS module (legacy - use modules.cns instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_cns: Option<bool>,
    /// Enable API module (legacy - use modules.api instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_api: Option<bool>,
    /// Enable Data module (legacy - use modules.data instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_data: Option<bool>,
    /// Redis URL (legacy - use redis.url instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redis_url: Option<String>,
    /// Database URL (legacy - use database.url instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub database_url: Option<String>,
    /// QuestDB host (legacy - use questdb.host instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub questdb_host: Option<String>,
    /// Environment (legacy - use service.environment instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    /// Service name (legacy - use service.name instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_name: Option<String>,
    /// CORS origins (legacy - use security.cors_origins instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cors_origins: Option<String>,

    // Legacy forward settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forward_signal_interval: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forward_ml_model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forward_risk_threshold: Option<f64>,

    // Legacy backward settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backward_persist_batch_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backward_analytics_interval: Option<u64>,

    // Legacy CNS settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cns_health_interval: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cns_auto_recovery: Option<bool>,

    // Legacy feature flags
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_websocket: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_grpc: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_metrics: Option<bool>,
}

// ============================================================================
// Configuration Sections
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServiceConfig {
    /// Service name
    pub name: String,
    /// Service version
    pub version: String,
    /// Environment: development | staging | production
    pub environment: String,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            name: "janus".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "development".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PortsConfig {
    /// HTTP/REST API port
    pub http: u16,
    /// gRPC API port
    pub grpc: u16,
    /// WebSocket port
    pub websocket: u16,
    /// Prometheus metrics port
    pub metrics: u16,
}

impl Default for PortsConfig {
    fn default() -> Self {
        Self {
            http: 8080,
            grpc: 50051,
            websocket: 8081,
            metrics: 9090,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HostConfig {
    /// Bind address for all services
    pub bind: String,
    /// Public hostname
    pub public: String,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0".to_string(),
            public: "localhost".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModulesConfig {
    /// Enable forward module
    pub forward: bool,
    /// Enable backward module
    pub backward: bool,
    /// Enable CNS module
    pub cns: bool,
    /// Enable API module
    pub api: bool,
    /// Enable Data module
    pub data: bool,
    /// Enable WebSocket streaming
    pub websocket: bool,
    /// Enable gRPC API
    pub grpc: bool,
    /// Enable Prometheus metrics
    pub metrics: bool,
}

impl Default for ModulesConfig {
    fn default() -> Self {
        Self {
            forward: true,
            backward: true,
            cns: true,
            api: true,
            data: true,
            websocket: true,
            grpc: true,
            metrics: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,
    /// Maximum connections in pool
    pub max_connections: u32,
    /// Minimum connections in pool
    pub min_connections: u32,
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    /// Pub/Sub channel for parameters
    pub param_channel: String,
    /// Pub/Sub channel for signals
    pub signal_channel: String,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379/0".to_string(),
            max_connections: 10,
            min_connections: 2,
            connect_timeout_secs: 10,
            param_channel: "fks:params".to_string(),
            signal_channel: "fks:signals".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL
    pub url: String,
    /// Maximum connections in pool
    pub max_connections: u32,
    /// Minimum connections in pool
    pub min_connections: u32,
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    /// Idle timeout in seconds
    pub idle_timeout_secs: u64,
    /// Maximum connection lifetime in seconds
    pub max_lifetime_secs: u64,
    /// Enable SQL query logging
    pub enable_logging: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgresql://postgres:postgres@localhost:5432/janus".to_string(),
            max_connections: 10,
            min_connections: 2,
            connect_timeout_secs: 30,
            idle_timeout_secs: 600,
            max_lifetime_secs: 1800,
            enable_logging: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QuestDbConfig {
    /// QuestDB host
    pub host: String,
    /// ILP port for line protocol
    pub ilp_port: u16,
    /// HTTP API port
    pub http_port: u16,
    /// PostgreSQL wire protocol port
    pub pg_port: u16,
}

impl Default for QuestDbConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            ilp_port: 9009,
            http_port: 9000,
            pg_port: 8812,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ForwardConfig {
    /// Signal generation interval in seconds
    pub signal_interval_secs: u64,
    /// Path to ML models
    pub ml_model_path: String,
    /// Enable ML inference
    pub enable_ml_inference: bool,
    /// Signal configuration
    pub signals: SignalConfig,
    /// Execution configuration
    pub execution: ExecutionConfig,
    /// Indicators configuration
    pub indicators: IndicatorsConfig,
    /// Strategies configuration
    pub strategies: StrategiesConfig,
}

impl Default for ForwardConfig {
    fn default() -> Self {
        Self {
            signal_interval_secs: 5,
            ml_model_path: "/models".to_string(),
            enable_ml_inference: false,
            signals: SignalConfig::default(),
            execution: ExecutionConfig::default(),
            indicators: IndicatorsConfig::default(),
            strategies: StrategiesConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SignalConfig {
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Minimum signal strength
    pub min_strength: f64,
    /// Maximum signal age in seconds
    pub max_age_secs: u64,
    /// Enable quality filtering
    pub enable_quality_filter: bool,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            min_strength: 0.5,
            max_age_secs: 300,
            enable_quality_filter: true,
            batch_size: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionConfig {
    /// Enable execution
    pub enabled: bool,
    /// Execution service endpoint
    pub endpoint: String,
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Enable TLS
    pub enable_tls: bool,
    /// Maximum retries
    pub max_retries: u32,
    /// Retry backoff in milliseconds
    pub retry_backoff_ms: u64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://execution:50052".to_string(),
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            enable_tls: false,
            max_retries: 3,
            retry_backoff_ms: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndicatorsConfig {
    /// EMA periods
    pub ema_periods: Vec<u32>,
    /// RSI period
    pub rsi_period: u32,
    /// RSI overbought threshold
    pub rsi_overbought: f64,
    /// RSI oversold threshold
    pub rsi_oversold: f64,
    /// MACD fast period
    pub macd_fast_period: u32,
    /// MACD slow period
    pub macd_slow_period: u32,
    /// MACD signal period
    pub macd_signal_period: u32,
    /// Bollinger period
    pub bollinger_period: u32,
    /// Bollinger standard deviation
    pub bollinger_std_dev: f64,
    /// ATR period
    pub atr_period: u32,
    /// Volume MA period
    pub volume_ma_period: u32,
}

impl Default for IndicatorsConfig {
    fn default() -> Self {
        Self {
            ema_periods: vec![9, 21, 50, 200],
            rsi_period: 14,
            rsi_overbought: 70.0,
            rsi_oversold: 30.0,
            macd_fast_period: 12,
            macd_slow_period: 26,
            macd_signal_period: 9,
            bollinger_period: 20,
            bollinger_std_dev: 2.0,
            atr_period: 14,
            volume_ma_period: 20,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StrategiesConfig {
    /// Strategy weights
    pub weights: StrategyWeights,
    /// Consensus settings
    pub consensus: ConsensusConfig,
    /// EMA crossover strategy
    pub ema_crossover: EmaCrossoverConfig,
    /// RSI reversal strategy
    pub rsi_reversal: RsiReversalConfig,
    /// MACD momentum strategy
    pub macd_momentum: MacdMomentumConfig,
    /// Bollinger breakout strategy
    pub bollinger_breakout: BollingerBreakoutConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StrategyWeights {
    pub ema_crossover: f64,
    pub rsi_reversal: f64,
    pub macd_momentum: f64,
    pub bollinger_breakout: f64,
}

impl Default for StrategyWeights {
    fn default() -> Self {
        Self {
            ema_crossover: 1.0,
            rsi_reversal: 1.0,
            macd_momentum: 1.0,
            bollinger_breakout: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsensusConfig {
    pub min_agreement: f64,
    pub min_strategies: u32,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            min_agreement: 0.6,
            min_strategies: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmaCrossoverConfig {
    pub enabled: bool,
    pub fast_period: u32,
    pub slow_period: u32,
    pub min_spread_pct: f64,
}

impl Default for EmaCrossoverConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fast_period: 9,
            slow_period: 21,
            min_spread_pct: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RsiReversalConfig {
    pub enabled: bool,
    pub period: u32,
    pub overbought_threshold: f64,
    pub oversold_threshold: f64,
    pub confirmation_candles: u32,
}

impl Default for RsiReversalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            period: 14,
            overbought_threshold: 70.0,
            oversold_threshold: 30.0,
            confirmation_candles: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MacdMomentumConfig {
    pub enabled: bool,
    pub fast_period: u32,
    pub slow_period: u32,
    pub signal_period: u32,
    pub histogram_threshold: f64,
}

impl Default for MacdMomentumConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
            histogram_threshold: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BollingerBreakoutConfig {
    pub enabled: bool,
    pub period: u32,
    pub std_dev: f64,
    pub require_close_outside: bool,
}

impl Default for BollingerBreakoutConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            period: 20,
            std_dev: 2.0,
            require_close_outside: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RiskConfig {
    /// Account balance for position sizing
    pub account_balance: f64,
    /// Maximum position size percentage
    pub max_position_size_pct: f64,
    /// Maximum portfolio risk percentage
    pub max_portfolio_risk_pct: f64,
    /// Maximum open positions
    pub max_open_positions: u32,
    /// Maximum daily loss
    pub max_daily_loss: f64,
    /// Maximum position hold time in hours
    pub max_position_hold_hours: u32,
    /// Stop loss configuration
    pub stop_loss: StopLossConfig,
    /// Take profit configuration
    pub take_profit: TakeProfitConfig,
    /// Position sizing configuration
    pub position_sizing: PositionSizingConfig,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            account_balance: 100000.0,
            max_position_size_pct: 0.02,
            max_portfolio_risk_pct: 0.10,
            max_open_positions: 10,
            max_daily_loss: 1000.0,
            max_position_hold_hours: 24,
            stop_loss: StopLossConfig::default(),
            take_profit: TakeProfitConfig::default(),
            position_sizing: PositionSizingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StopLossConfig {
    pub default_pct: f64,
    pub use_atr: bool,
    pub atr_multiplier: f64,
    pub min_distance_pct: f64,
    pub max_distance_pct: f64,
}

impl Default for StopLossConfig {
    fn default() -> Self {
        Self {
            default_pct: 0.02,
            use_atr: true,
            atr_multiplier: 2.0,
            min_distance_pct: 0.005,
            max_distance_pct: 0.10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TakeProfitConfig {
    pub risk_reward_ratio: f64,
    pub enable_trailing: bool,
    pub trailing_distance_pct: f64,
}

impl Default for TakeProfitConfig {
    fn default() -> Self {
        Self {
            risk_reward_ratio: 2.0,
            enable_trailing: false,
            trailing_distance_pct: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PositionSizingConfig {
    /// Method: fixed | risk_based | kelly | volatility
    pub method: String,
    pub fixed_size_usd: f64,
    pub risk_per_trade_pct: f64,
    pub kelly_fraction: f64,
}

impl Default for PositionSizingConfig {
    fn default() -> Self {
        Self {
            method: "risk_based".to_string(),
            fixed_size_usd: 1000.0,
            risk_per_trade_pct: 0.01,
            kelly_fraction: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackwardConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Enable scheduled jobs
    pub enable_scheduler: bool,
    /// Persistence configuration
    pub persistence: PersistenceConfig,
    /// Analytics configuration
    pub analytics: AnalyticsConfig,
    /// Data retention configuration
    pub retention: RetentionConfig,
}

impl Default for BackwardConfig {
    fn default() -> Self {
        Self {
            worker_threads: 4,
            enable_scheduler: true,
            persistence: PersistenceConfig::default(),
            analytics: AnalyticsConfig::default(),
            retention: RetentionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistenceConfig {
    pub batch_size: usize,
    pub flush_interval_secs: u64,
    pub enable_wal: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            flush_interval_secs: 5,
            enable_wal: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalyticsConfig {
    pub update_interval_secs: u64,
    pub performance_window_hours: u32,
    pub enable_trade_analysis: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            update_interval_secs: 60,
            performance_window_hours: 24,
            enable_trade_analysis: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetentionConfig {
    pub signals_days: u32,
    pub trades_days: u32,
    pub metrics_days: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            signals_days: 90,
            trades_days: 365,
            metrics_days: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CnsConfig {
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Enable automatic recovery reflexes
    pub enable_reflexes: bool,
    /// Verbose logging
    pub verbose_logging: bool,
    /// Startup grace period in seconds
    pub startup_grace_period_secs: u64,
    /// Maximum concurrent probes
    pub max_concurrent_probes: u32,
    /// Probe retry attempts
    pub probe_retry_attempts: u32,
    /// Endpoints configuration
    pub endpoints: CnsEndpointsConfig,
    /// Circuit breakers configuration
    pub circuit_breakers: HashMap<String, CircuitBreakerConfig>,
}

impl Default for CnsConfig {
    fn default() -> Self {
        Self {
            health_check_interval_secs: 10,
            enable_reflexes: true,
            verbose_logging: false,
            startup_grace_period_secs: 30,
            max_concurrent_probes: 10,
            probe_retry_attempts: 2,
            endpoints: CnsEndpointsConfig::default(),
            circuit_breakers: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CnsEndpointsConfig {
    pub forward_service: String,
    pub backward_service: String,
    pub gateway_service: String,
    pub redis: String,
    pub qdrant: String,
    pub shared_memory_path: String,
    pub neuromorphic: NeuromorphicConfig,
}

impl Default for CnsEndpointsConfig {
    fn default() -> Self {
        Self {
            forward_service: "http://localhost:8080/api/v1".to_string(),
            backward_service: "http://localhost:8082".to_string(),
            gateway_service: "http://localhost:8000".to_string(),
            redis: "redis://localhost:6379".to_string(),
            qdrant: String::new(),
            shared_memory_path: "/dev/shm/janus_forward_backward".to_string(),
            neuromorphic: NeuromorphicConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuromorphicConfig {
    pub enabled: bool,
    pub base_url: String,
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://localhost:8090".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub failure_window_secs: u64,
    pub recovery_timeout_secs: u64,
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            failure_window_secs: 60,
            recovery_timeout_secs: 30,
            success_threshold: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MarketConfig {
    /// Primary exchange
    pub exchange: String,
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Enable order book
    pub enable_orderbook: bool,
    /// Order book depth
    pub orderbook_depth: u32,
    /// Timeframes configuration
    pub timeframes: TimeframesConfig,
}

impl Default for MarketConfig {
    fn default() -> Self {
        Self {
            exchange: "kraken".to_string(),
            update_interval_ms: 1000,
            enable_orderbook: true,
            orderbook_depth: 10,
            timeframes: TimeframesConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AssetsConfig {
    /// List of enabled trading assets (base currency symbols)
    pub enabled: Vec<String>,
    /// Default quote currency
    pub default_quote: String,
    /// Assets to use for optimization runs
    pub optimize_assets: Vec<String>,
    /// High priority assets (receive more frequent updates)
    pub priority_assets: Vec<String>,
    /// Per-asset configurations
    #[serde(flatten)]
    pub configs: HashMap<String, AssetConfig>,
}

impl Default for AssetsConfig {
    fn default() -> Self {
        Self {
            enabled: vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()],
            default_quote: "USD".to_string(),
            optimize_assets: vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()],
            priority_assets: vec!["BTC".to_string(), "ETH".to_string()],
            configs: HashMap::new(),
        }
    }
}

impl AssetsConfig {
    /// Get list of enabled trading symbols (e.g., "BTC/USD")
    pub fn enabled_symbols(&self) -> Vec<String> {
        self.enabled
            .iter()
            .map(|asset| format!("{}/{}", asset, self.default_quote))
            .collect()
    }

    /// Get config for a specific asset
    pub fn get(&self, asset: &str) -> Option<&AssetConfig> {
        self.configs.get(asset)
    }

    /// Check if an asset is enabled
    pub fn is_enabled(&self, asset: &str) -> bool {
        self.enabled.contains(&asset.to_string())
    }

    /// Check if an asset is high priority
    pub fn is_priority(&self, asset: &str) -> bool {
        self.priority_assets.contains(&asset.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AssetConfig {
    /// Full trading symbol (e.g., "BTC/USD")
    pub symbol: String,
    /// Whether this asset is enabled for trading
    pub enabled: bool,
    /// Maximum position size as percentage of account
    pub max_position_size_pct: f64,
    /// Maximum leverage allowed
    pub max_leverage: f64,
    /// Minimum order size in base currency
    pub min_order_size: f64,
    /// Maximum order size in base currency
    pub max_order_size: f64,
    /// ATR multiplier for stop loss calculation
    pub atr_multiplier: f64,
    /// RSI overbought threshold
    pub rsi_overbought: f64,
    /// RSI oversold threshold
    pub rsi_oversold: f64,
    /// Exchange-specific configurations
    pub exchanges: HashMap<String, ExchangeAssetConfig>,
}

impl Default for AssetConfig {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            enabled: true,
            max_position_size_pct: 0.02,
            max_leverage: 2.0,
            min_order_size: 0.001,
            max_order_size: 100.0,
            atr_multiplier: 2.0,
            rsi_overbought: 70.0,
            rsi_oversold: 30.0,
            exchanges: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExchangeAssetConfig {
    /// Exchange-specific trading pair symbol
    pub pair: String,
    /// Minimum order size on this exchange
    pub min_order: f64,
    /// Fee tier (e.g., "maker", "taker")
    pub fee_tier: Option<String>,
    /// Category for derivatives (e.g., "linear", "inverse")
    pub category: Option<String>,
}

impl Default for ExchangeAssetConfig {
    fn default() -> Self {
        Self {
            pair: String::new(),
            min_order: 0.001,
            fee_tier: None,
            category: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TimeframesConfig {
    pub enabled: Vec<String>,
    pub primary: String,
}

impl Default for TimeframesConfig {
    fn default() -> Self {
        Self {
            enabled: vec![
                "1m".to_string(),
                "5m".to_string(),
                "15m".to_string(),
                "1h".to_string(),
                "4h".to_string(),
                "1d".to_string(),
            ],
            primary: "5m".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TradingConfig {
    /// Trading mode: simulation | paper | live
    pub mode: String,
    /// Enable real order execution
    pub real_orders_enabled: bool,
    /// Dry run mode
    pub dry_run: bool,
    /// Simulation settings
    pub simulation: SimulationConfig,
    /// Order settings
    pub orders: OrdersConfig,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            mode: "paper".to_string(),
            real_orders_enabled: false,
            dry_run: true,
            simulation: SimulationConfig::default(),
            orders: OrdersConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SimulationConfig {
    pub initial_balance: f64,
    pub slippage_bps: u32,
    pub fee_bps: u32,
    pub fill_delay_ms: u64,
    pub enable_slippage: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            initial_balance: 100000.0,
            slippage_bps: 5,
            fee_bps: 10,
            fill_delay_ms: 200,
            enable_slippage: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OrdersConfig {
    pub default_type: String,
    pub default_tif: String,
    pub min_size_usd: f64,
    pub max_size_usd: f64,
    pub max_slippage_bps: u32,
}

impl Default for OrdersConfig {
    fn default() -> Self {
        Self {
            default_type: "limit".to_string(),
            default_tif: "gtc".to_string(),
            min_size_usd: 10.0,
            max_size_usd: 100000.0,
            max_slippage_bps: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level: trace | debug | info | warn | error
    pub level: String,
    /// Log format: json | pretty
    pub format: String,
    /// Enable console output
    pub console: bool,
    /// Enable file logging
    pub file_enabled: bool,
    /// Log file path
    pub file_path: String,
    /// Enable SQL query logging
    pub sql_logging: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            console: true,
            file_enabled: false,
            file_path: "/var/log/janus/janus.log".to_string(),
            sql_logging: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TracingConfig {
    /// Enable distributed tracing
    pub enabled: bool,
    /// Jaeger endpoint
    pub jaeger_endpoint: String,
    /// Sampling rate
    pub sampling_rate: f64,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            jaeger_endpoint: "http://jaeger:14268/api/traces".to_string(),
            sampling_rate: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    /// Metrics endpoint path
    pub endpoint: String,
    /// Include detailed histograms
    pub detailed_histograms: bool,
    /// Custom labels
    pub labels: HashMap<String, String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        let mut labels = HashMap::new();
        labels.insert("service".to_string(), "janus".to_string());
        labels.insert("environment".to_string(), "development".to_string());
        Self {
            endpoint: "/metrics".to_string(),
            detailed_histograms: true,
            labels,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Discord configuration
    pub discord: DiscordConfig,
    /// Slack configuration
    pub slack: SlackConfig,
    /// Email configuration
    pub email: EmailConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DiscordConfig {
    pub webhook_url: String,
    pub enabled: bool,
    pub notify_on_signal: bool,
    pub notify_on_fill: bool,
    pub notify_on_error: bool,
}

impl Default for DiscordConfig {
    fn default() -> Self {
        Self {
            webhook_url: String::new(),
            enabled: false,
            notify_on_signal: true,
            notify_on_fill: true,
            notify_on_error: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SlackConfig {
    pub webhook_url: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmailConfig {
    pub enabled: bool,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub recipients: Vec<String>,
}

impl Default for EmailConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            smtp_host: "localhost".to_string(),
            smtp_port: 587,
            recipients: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ParamReloadConfig {
    /// Enable parameter hot-reload
    pub enabled: bool,
    /// Instance ID for namespacing
    pub instance_id: String,
    /// Reconnection delay in milliseconds
    pub reconnect_delay_ms: u64,
    /// Maximum reconnection attempts (0 = unlimited)
    pub max_retries: u32,
}

impl Default for ParamReloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            instance_id: "default".to_string(),
            reconnect_delay_ms: 5000,
            max_retries: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FeaturesConfig {
    /// Enable GAF image generation
    pub enable_gaf: bool,
    /// GAF image size
    pub gaf_image_size: u32,
    /// GAF method: summation | difference
    pub gaf_method: String,
    /// Lookback windows
    pub lookback_windows: Vec<u32>,
    /// Enable normalization
    pub normalize: bool,
    /// Normalization method: minmax | zscore | robust
    pub normalization_method: String,
}

impl Default for FeaturesConfig {
    fn default() -> Self {
        Self {
            enable_gaf: false,
            gaf_image_size: 32,
            gaf_method: "summation".to_string(),
            lookback_windows: vec![5, 10, 20, 50, 100],
            normalize: true,
            normalization_method: "zscore".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    /// CORS allowed origins
    pub cors_origins: String,
    /// Enable rate limiting
    pub enable_rate_limit: bool,
    /// Requests per second per IP
    pub rate_limit_rps: u32,
    /// API key header name
    pub api_key_header: String,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            cors_origins: "*".to_string(),
            enable_rate_limit: true,
            rate_limit_rps: 100,
            api_key_header: "X-API-Key".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdvancedConfig {
    /// Tokio worker thread count (0 = auto)
    pub tokio_worker_threads: usize,
    /// HTTP request timeout in seconds
    pub http_timeout_secs: u64,
    /// Database query timeout in seconds
    pub db_query_timeout_secs: u64,
    /// Signal buffer size
    pub signal_buffer_size: usize,
    /// Order buffer size
    pub order_buffer_size: usize,
    /// Enable experimental features
    pub experimental_features: bool,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            tokio_worker_threads: 0,
            http_timeout_secs: 30,
            db_query_timeout_secs: 30,
            signal_buffer_size: 1000,
            order_buffer_size: 500,
            experimental_features: false,
        }
    }
}

// ============================================================================
// Config Implementation
// ============================================================================

impl Config {
    /// Load configuration from TOML file with environment variable overrides
    ///
    /// Priority order (highest to lowest):
    /// 1. Environment variables
    /// 2. TOML config file
    /// 3. Default values
    pub fn load() -> anyhow::Result<Self> {
        // Try to load from TOML file
        let mut config = Self::load_from_file()?;

        // Apply environment variable overrides
        config.apply_env_overrides();

        // Normalize legacy fields
        config.normalize_legacy_fields();

        Ok(config)
    }

    /// Load configuration from environment variables only (legacy method)
    pub fn from_env() -> anyhow::Result<Self> {
        let mut config = Self::default();
        config.apply_env_overrides();
        config.normalize_legacy_fields();
        Ok(config)
    }

    /// Load configuration from a specific TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        let mut config: Config = toml::from_str(&contents)?;
        config.apply_env_overrides();
        config.normalize_legacy_fields();
        Ok(config)
    }

    /// Load from default config file locations
    fn load_from_file() -> anyhow::Result<Self> {
        // Check JANUS_CONFIG_PATH environment variable first
        if let Ok(config_path) = std::env::var("JANUS_CONFIG_PATH") {
            if Path::new(&config_path).exists() {
                info!(
                    "Loading configuration from JANUS_CONFIG_PATH: {}",
                    config_path
                );
                let contents = std::fs::read_to_string(&config_path)?;
                return Ok(toml::from_str(&contents)?);
            } else {
                warn!("JANUS_CONFIG_PATH set but file not found: {}", config_path);
            }
        }

        // Try default locations
        for path in CONFIG_PATHS {
            if Path::new(path).exists() {
                info!("Loading configuration from: {}", path);
                let contents = std::fs::read_to_string(path)?;
                return Ok(toml::from_str(&contents)?);
            }
        }

        debug!("No config file found, using defaults");
        Ok(Self::default())
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&mut self) {
        // Service
        if let Ok(v) = std::env::var("JANUS_SERVICE_NAME") {
            self.service.name = v;
        }
        if let Ok(v) = std::env::var("JANUS_ENVIRONMENT") {
            self.service.environment = v;
        }

        // Ports
        if let Ok(v) = std::env::var("JANUS_HTTP_PORT")
            && let Ok(port) = v.parse()
        {
            self.ports.http = port;
        }
        if let Ok(v) = std::env::var("JANUS_GRPC_PORT")
            && let Ok(port) = v.parse()
        {
            self.ports.grpc = port;
        }
        if let Ok(v) = std::env::var("JANUS_WS_PORT")
            && let Ok(port) = v.parse()
        {
            self.ports.websocket = port;
        }
        if let Ok(v) = std::env::var("JANUS_METRICS_PORT")
            && let Ok(port) = v.parse()
        {
            self.ports.metrics = port;
        }

        // Host
        if let Ok(v) = std::env::var("JANUS_HOST") {
            self.host.bind = v;
        }

        // Modules
        if let Ok(v) = std::env::var("JANUS_ENABLE_FORWARD") {
            self.modules.forward = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("JANUS_ENABLE_BACKWARD") {
            self.modules.backward = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("JANUS_ENABLE_CNS") {
            self.modules.cns = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("JANUS_ENABLE_API") {
            self.modules.api = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("JANUS_ENABLE_DATA") {
            self.modules.data = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("JANUS_ENABLE_WEBSOCKET") {
            self.modules.websocket = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("JANUS_ENABLE_GRPC") {
            self.modules.grpc = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("JANUS_ENABLE_METRICS") {
            self.modules.metrics = parse_bool(&v);
        }

        // External services
        if let Ok(v) = std::env::var("REDIS_URL") {
            self.redis.url = v;
        }
        if let Ok(v) = std::env::var("DATABASE_URL") {
            self.database.url = v;
        }
        if let Ok(v) = std::env::var("QUESTDB_HOST") {
            self.questdb.host = v;
        }

        // Forward settings
        if let Ok(v) = std::env::var("JANUS_FORWARD_SIGNAL_INTERVAL")
            && let Ok(interval) = v.parse()
        {
            self.forward.signal_interval_secs = interval;
        }
        if let Ok(v) = std::env::var("JANUS_FORWARD_ML_MODEL_PATH") {
            self.forward.ml_model_path = v;
        }

        // Risk settings
        if let Ok(v) = std::env::var("RISK_ACCOUNT_BALANCE")
            && let Ok(balance) = v.parse()
        {
            self.risk.account_balance = balance;
        }
        if let Ok(v) = std::env::var("RISK_MAX_POSITION_SIZE_PCT")
            && let Ok(pct) = v.parse()
        {
            self.risk.max_position_size_pct = pct;
        }

        // Backward settings
        if let Ok(v) = std::env::var("JANUS_BACKWARD_PERSIST_BATCH_SIZE")
            && let Ok(size) = v.parse()
        {
            self.backward.persistence.batch_size = size;
        }
        if let Ok(v) = std::env::var("JANUS_BACKWARD_ANALYTICS_INTERVAL")
            && let Ok(interval) = v.parse()
        {
            self.backward.analytics.update_interval_secs = interval;
        }

        // CNS settings
        if let Ok(v) = std::env::var("JANUS_CNS_HEALTH_INTERVAL")
            && let Ok(interval) = v.parse()
        {
            self.cns.health_check_interval_secs = interval;
        }
        if let Ok(v) = std::env::var("JANUS_CNS_AUTO_RECOVERY") {
            self.cns.enable_reflexes = parse_bool(&v);
        }

        // Assets settings
        if let Ok(v) = std::env::var("OPTIMIZE_ASSETS") {
            self.assets.optimize_assets = v.split(',').map(|s| s.trim().to_string()).collect();
        }
        if let Ok(v) = std::env::var("ENABLED_ASSETS") {
            self.assets.enabled = v.split(',').map(|s| s.trim().to_string()).collect();
        }
        if let Ok(v) = std::env::var("TRADING_ASSETS") {
            // Alias for ENABLED_ASSETS
            self.assets.enabled = v.split(',').map(|s| s.trim().to_string()).collect();
        }
        if let Ok(v) = std::env::var("PRIORITY_ASSETS") {
            self.assets.priority_assets = v.split(',').map(|s| s.trim().to_string()).collect();
        }
        if let Ok(v) = std::env::var("DEFAULT_QUOTE_CURRENCY") {
            self.assets.default_quote = v;
        }

        // Market/Exchange settings
        if let Ok(v) = std::env::var("PRIMARY_EXCHANGE") {
            self.market.exchange = v;
        }

        // Trading mode
        if let Ok(v) = std::env::var("TRADING_MODE") {
            self.trading.mode = v;
        }
        if let Ok(v) = std::env::var("REAL_ORDERS_ENABLED") {
            self.trading.real_orders_enabled = parse_bool(&v);
        }

        // Security
        if let Ok(v) = std::env::var("JANUS_CORS_ORIGINS") {
            self.security.cors_origins = v;
        }

        // Logging
        if let Ok(v) = std::env::var("RUST_LOG") {
            // Extract log level from RUST_LOG
            if v.contains("debug") {
                self.logging.level = "debug".to_string();
            } else if v.contains("trace") {
                self.logging.level = "trace".to_string();
            } else if v.contains("warn") {
                self.logging.level = "warn".to_string();
            } else if v.contains("error") {
                self.logging.level = "error".to_string();
            }
        }
        if let Ok(v) = std::env::var("LOG_FORMAT") {
            self.logging.format = v;
        }
    }

    /// Normalize legacy fields to new structure
    fn normalize_legacy_fields(&mut self) {
        // Ports
        if let Some(port) = self.http_port {
            self.ports.http = port;
        }
        if let Some(port) = self.grpc_port {
            self.ports.grpc = port;
        }
        if let Some(port) = self.websocket_port {
            self.ports.websocket = port;
        }
        if let Some(port) = self.metrics_port {
            self.ports.metrics = port;
        }

        // Modules
        if let Some(enabled) = self.enable_forward {
            self.modules.forward = enabled;
        }
        if let Some(enabled) = self.enable_backward {
            self.modules.backward = enabled;
        }
        if let Some(enabled) = self.enable_cns {
            self.modules.cns = enabled;
        }
        if let Some(enabled) = self.enable_api {
            self.modules.api = enabled;
        }
        if let Some(enabled) = self.enable_data {
            self.modules.data = enabled;
        }
        if let Some(enabled) = self.enable_websocket {
            self.modules.websocket = enabled;
        }
        if let Some(enabled) = self.enable_grpc {
            self.modules.grpc = enabled;
        }
        if let Some(enabled) = self.enable_metrics {
            self.modules.metrics = enabled;
        }

        // External services
        if let Some(ref url) = self.redis_url {
            self.redis.url = url.clone();
        }
        if let Some(ref url) = self.database_url {
            self.database.url = url.clone();
        }
        if let Some(ref host) = self.questdb_host {
            self.questdb.host = host.clone();
        }

        // Service
        if let Some(ref env) = self.environment {
            self.service.environment = env.clone();
        }
        if let Some(ref name) = self.service_name {
            self.service.name = name.clone();
        }

        // Security
        if let Some(ref origins) = self.cors_origins {
            self.security.cors_origins = origins.clone();
        }

        // Forward
        if let Some(interval) = self.forward_signal_interval {
            self.forward.signal_interval_secs = interval;
        }
        if let Some(ref path) = self.forward_ml_model_path {
            self.forward.ml_model_path = path.clone();
        }

        // Backward
        if let Some(size) = self.backward_persist_batch_size {
            self.backward.persistence.batch_size = size;
        }
        if let Some(interval) = self.backward_analytics_interval {
            self.backward.analytics.update_interval_secs = interval;
        }

        // CNS
        if let Some(interval) = self.cns_health_interval {
            self.cns.health_check_interval_secs = interval;
        }
        if let Some(enabled) = self.cns_auto_recovery {
            self.cns.enable_reflexes = enabled;
        }
    }

    /// Check if running in production
    pub fn is_production(&self) -> bool {
        self.service.environment == "production"
    }

    /// Get CORS origins as a list
    pub fn cors_origins_list(&self) -> Vec<String> {
        self.security
            .cors_origins
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    }

    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        // Ensure at least one module is enabled
        if !self.modules.forward
            && !self.modules.backward
            && !self.modules.cns
            && !self.modules.api
            && !self.modules.data
        {
            anyhow::bail!("At least one module must be enabled");
        }

        // Validate ports are unique
        let ports = [
            self.ports.http,
            self.ports.grpc,
            self.ports.websocket,
            self.ports.metrics,
        ];
        let unique: std::collections::HashSet<_> = ports.iter().collect();
        if unique.len() != ports.len() {
            anyhow::bail!("All ports must be unique");
        }

        // Validate trading mode
        let valid_modes = ["simulation", "paper", "live"];
        if !valid_modes.contains(&self.trading.mode.as_str()) {
            anyhow::bail!(
                "Invalid trading mode '{}'. Must be one of: {:?}",
                self.trading.mode,
                valid_modes
            );
        }

        // Warn about dangerous configurations
        if self.trading.mode == "live" && self.trading.real_orders_enabled {
            warn!("⚠️  LIVE TRADING ENABLED - Real orders will be executed!");
        }

        Ok(())
    }

    /// Export configuration to TOML string
    pub fn to_toml(&self) -> anyhow::Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let contents = self.to_toml()?;
        std::fs::write(path, contents)?;
        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_bool(s: &str) -> bool {
    matches!(s.to_lowercase().as_str(), "true" | "1" | "yes" | "on")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.ports.http, 8080);
        assert!(config.modules.forward);
    }

    #[test]
    fn test_validate_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cors_origins_list() {
        let mut config = Config::default();
        config.security.cors_origins = "http://localhost:3000, http://localhost:8080".to_string();
        let origins = config.cors_origins_list();
        assert_eq!(origins.len(), 2);
    }

    #[test]
    fn test_toml_serialization() {
        let config = Config::default();
        let toml_str = config.to_toml().unwrap();
        assert!(toml_str.contains("[service]"));
        assert!(toml_str.contains("[ports]"));
    }

    #[test]
    fn test_toml_deserialization() {
        let toml_str = r#"
            [service]
            name = "test-janus"
            environment = "testing"

            [ports]
            http = 9000
            grpc = 50052
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.service.name, "test-janus");
        assert_eq!(config.service.environment, "testing");
        assert_eq!(config.ports.http, 9000);
        assert_eq!(config.ports.grpc, 50052);
    }

    #[test]
    fn test_parse_bool() {
        assert!(parse_bool("true"));
        assert!(parse_bool("True"));
        assert!(parse_bool("1"));
        assert!(parse_bool("yes"));
        assert!(parse_bool("on"));
        assert!(!parse_bool("false"));
        assert!(!parse_bool("0"));
        assert!(!parse_bool("no"));
    }

    #[test]
    fn test_validate_unique_ports() {
        let mut config = Config::default();
        config.ports.http = 8080;
        config.ports.grpc = 8080; // Duplicate!
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_trading_mode() {
        let mut config = Config::default();
        config.trading.mode = "invalid".to_string();
        assert!(config.validate().is_err());
    }
}
