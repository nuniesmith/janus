//! Execution Service gRPC Client
//!
//! This module provides a gRPC client for submitting trading signals
//! from the forward service to the execution service.
//!
//! The client is exchange-agnostic and supports routing signals to any
//! configured exchange based on the signal's metadata or configuration.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::time::Duration;
use tonic::Request;
use tonic::transport::{Channel, Endpoint};
use tracing::{debug, error, info, warn};

use crate::signal::TradingSignal;

// Import protobuf types from centralized fks-proto crate
use fks_proto::common::{OrderType, Side, TimeInForce};
use fks_proto::execution::{
    ExecutionStrategy, SubmitSignalRequest, execution_service_client::ExecutionServiceClient,
};

/// Supported exchanges for order execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Exchange {
    /// Bybit exchange
    Bybit,
    /// Binance exchange
    Binance,
    /// Kraken exchange
    Kraken,
    /// Coinbase exchange
    Coinbase,
    /// OKX exchange
    Okx,
    /// Bitfinex exchange
    Bitfinex,
    /// KuCoin exchange
    KuCoin,
    /// Gate.io exchange
    GateIo,
    /// Simulated/Paper trading
    #[default]
    Paper,
}

impl Exchange {
    /// Get the exchange identifier string
    pub fn as_str(&self) -> &'static str {
        match self {
            Exchange::Bybit => "bybit",
            Exchange::Binance => "binance",
            Exchange::Kraken => "kraken",
            Exchange::Coinbase => "coinbase",
            Exchange::Okx => "okx",
            Exchange::Bitfinex => "bitfinex",
            Exchange::KuCoin => "kucoin",
            Exchange::GateIo => "gateio",
            Exchange::Paper => "paper",
        }
    }

    /// Parse exchange from string (case-insensitive)
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }

    /// Get all supported exchanges
    pub fn all() -> &'static [Exchange] {
        &[
            Exchange::Bybit,
            Exchange::Binance,
            Exchange::Kraken,
            Exchange::Coinbase,
            Exchange::Okx,
            Exchange::Bitfinex,
            Exchange::KuCoin,
            Exchange::GateIo,
            Exchange::Paper,
        ]
    }
}

impl std::fmt::Display for Exchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Exchange {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bybit" => Ok(Exchange::Bybit),
            "binance" => Ok(Exchange::Binance),
            "kraken" => Ok(Exchange::Kraken),
            "coinbase" => Ok(Exchange::Coinbase),
            "okx" => Ok(Exchange::Okx),
            "bitfinex" => Ok(Exchange::Bitfinex),
            "kucoin" => Ok(Exchange::KuCoin),
            "gateio" | "gate.io" | "gate" => Ok(Exchange::GateIo),
            "paper" | "sim" | "simulated" | "test" => Ok(Exchange::Paper),
            _ => Err(format!("Unknown exchange: {}", s)),
        }
    }
}

/// Exchange-specific configuration
#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    /// Whether this exchange is enabled
    pub enabled: bool,

    /// Default quantity for orders on this exchange
    pub default_quantity: f64,

    /// Minimum order quantity
    pub min_quantity: f64,

    /// Maximum order quantity
    pub max_quantity: f64,

    /// Symbol format transformer (e.g., "BTCUSD" -> "BTC/USDT" for some exchanges)
    pub symbol_format: SymbolFormat,

    /// Default time in force for this exchange
    pub default_time_in_force: TimeInForce,

    /// Whether to use market orders by default (vs limit)
    pub prefer_market_orders: bool,
}

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_quantity: 0.001,
            min_quantity: 0.0001,
            max_quantity: 100.0,
            symbol_format: SymbolFormat::Concatenated,
            default_time_in_force: TimeInForce::Gtc,
            prefer_market_orders: false,
        }
    }
}

/// Symbol format variants for different exchanges
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolFormat {
    /// Concatenated format: BTCUSD
    Concatenated,
    /// Slash-separated: BTC/USDT
    SlashSeparated,
    /// Dash-separated: BTC-USDT
    DashSeparated,
    /// Underscore-separated: BTC_USDT
    UnderscoreSeparated,
}

impl SymbolFormat {
    /// Transform a symbol to this format
    /// Assumes input is in concatenated format (e.g., "BTCUSD")
    pub fn transform(&self, symbol: &str) -> String {
        // Try to split the symbol into base and quote
        // Common quote currencies
        let quote_currencies = ["USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP", "BUSD"];

        for quote in quote_currencies {
            if symbol.ends_with(quote) && symbol.len() > quote.len() {
                let base = &symbol[..symbol.len() - quote.len()];
                return match self {
                    SymbolFormat::Concatenated => symbol.to_string(),
                    SymbolFormat::SlashSeparated => format!("{}/{}", base, quote),
                    SymbolFormat::DashSeparated => format!("{}-{}", base, quote),
                    SymbolFormat::UnderscoreSeparated => format!("{}_{}", base, quote),
                };
            }
        }

        // If we can't parse it, return as-is
        symbol.to_string()
    }
}

/// Configuration for the execution client
#[derive(Debug, Clone)]
pub struct ExecutionClientConfig {
    /// Execution service gRPC endpoint (e.g., "http://execution:50052")
    pub endpoint: String,

    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,

    /// Enable TLS
    pub enable_tls: bool,

    /// Number of retry attempts for failed requests
    pub max_retries: u32,

    /// Retry backoff delay in milliseconds
    pub retry_backoff_ms: u64,

    /// Default exchange to use when not specified in signal
    pub default_exchange: Exchange,

    /// Per-exchange configurations
    pub exchange_configs: HashMap<Exchange, ExchangeConfig>,

    /// Default execution strategy
    pub default_strategy: ExecutionStrategy,

    /// Whether to allow signals without explicit exchange specification
    pub require_explicit_exchange: bool,
}

impl Default for ExecutionClientConfig {
    fn default() -> Self {
        let mut exchange_configs = HashMap::new();

        // Configure common exchanges with sensible defaults
        exchange_configs.insert(
            Exchange::Bybit,
            ExchangeConfig {
                enabled: true,
                default_quantity: 0.001,
                min_quantity: 0.0001,
                max_quantity: 100.0,
                symbol_format: SymbolFormat::Concatenated,
                default_time_in_force: TimeInForce::Gtc,
                prefer_market_orders: false,
            },
        );

        exchange_configs.insert(
            Exchange::Binance,
            ExchangeConfig {
                enabled: true,
                default_quantity: 0.001,
                min_quantity: 0.0001,
                max_quantity: 100.0,
                symbol_format: SymbolFormat::Concatenated,
                default_time_in_force: TimeInForce::Gtc,
                prefer_market_orders: false,
            },
        );

        exchange_configs.insert(
            Exchange::Kraken,
            ExchangeConfig {
                enabled: true,
                default_quantity: 0.001,
                min_quantity: 0.0001,
                max_quantity: 50.0,
                symbol_format: SymbolFormat::SlashSeparated,
                default_time_in_force: TimeInForce::Gtc,
                prefer_market_orders: false,
            },
        );

        exchange_configs.insert(
            Exchange::Coinbase,
            ExchangeConfig {
                enabled: true,
                default_quantity: 0.001,
                min_quantity: 0.0001,
                max_quantity: 50.0,
                symbol_format: SymbolFormat::DashSeparated,
                default_time_in_force: TimeInForce::Gtc,
                prefer_market_orders: false,
            },
        );

        exchange_configs.insert(
            Exchange::Paper,
            ExchangeConfig {
                enabled: true,
                default_quantity: 1.0, // Larger quantities for paper trading
                min_quantity: 0.0001,
                max_quantity: 1000.0,
                symbol_format: SymbolFormat::Concatenated,
                default_time_in_force: TimeInForce::Gtc,
                prefer_market_orders: true, // Prefer market orders for paper trading
            },
        );

        Self {
            endpoint: "http://execution:50052".to_string(),
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            enable_tls: false,
            max_retries: 3,
            retry_backoff_ms: 100,
            default_exchange: Exchange::Paper,
            exchange_configs,
            default_strategy: ExecutionStrategy::Immediate,
            require_explicit_exchange: false,
        }
    }
}

impl ExecutionClientConfig {
    /// Create a new config builder
    pub fn builder() -> ExecutionClientConfigBuilder {
        ExecutionClientConfigBuilder::default()
    }

    /// Get the configuration for a specific exchange
    pub fn get_exchange_config(&self, exchange: Exchange) -> ExchangeConfig {
        self.exchange_configs
            .get(&exchange)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if an exchange is enabled
    pub fn is_exchange_enabled(&self, exchange: Exchange) -> bool {
        self.exchange_configs
            .get(&exchange)
            .map(|c| c.enabled)
            .unwrap_or(false)
    }
}

/// Builder for ExecutionClientConfig
#[derive(Debug, Default)]
pub struct ExecutionClientConfigBuilder {
    config: ExecutionClientConfig,
}

impl ExecutionClientConfigBuilder {
    /// Set the gRPC endpoint
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.endpoint = endpoint.into();
        self
    }

    /// Set connection timeout
    pub fn connect_timeout_secs(mut self, secs: u64) -> Self {
        self.config.connect_timeout_secs = secs;
        self
    }

    /// Set request timeout
    pub fn request_timeout_secs(mut self, secs: u64) -> Self {
        self.config.request_timeout_secs = secs;
        self
    }

    /// Enable or disable TLS
    pub fn enable_tls(mut self, enable: bool) -> Self {
        self.config.enable_tls = enable;
        self
    }

    /// Set max retries
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Set default exchange
    pub fn default_exchange(mut self, exchange: Exchange) -> Self {
        self.config.default_exchange = exchange;
        self
    }

    /// Add or update exchange configuration
    pub fn with_exchange(mut self, exchange: Exchange, config: ExchangeConfig) -> Self {
        self.config.exchange_configs.insert(exchange, config);
        self
    }

    /// Enable an exchange with default settings
    pub fn enable_exchange(mut self, exchange: Exchange) -> Self {
        self.config
            .exchange_configs
            .entry(exchange)
            .or_default()
            .enabled = true;
        self
    }

    /// Disable an exchange
    pub fn disable_exchange(mut self, exchange: Exchange) -> Self {
        if let Some(config) = self.config.exchange_configs.get_mut(&exchange) {
            config.enabled = false;
        }
        self
    }

    /// Require explicit exchange in signals
    pub fn require_explicit_exchange(mut self, require: bool) -> Self {
        self.config.require_explicit_exchange = require;
        self
    }

    /// Set default execution strategy
    pub fn default_strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.config.default_strategy = strategy;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ExecutionClientConfig {
        self.config
    }
}

/// Execution service gRPC client
pub struct ExecutionClient {
    client: ExecutionServiceClient<Channel>,
    config: ExecutionClientConfig,
}

impl ExecutionClient {
    /// Create a new execution client
    pub async fn new(config: ExecutionClientConfig) -> Result<Self> {
        info!("Connecting to execution service at {}", config.endpoint);

        let endpoint = Endpoint::from_shared(config.endpoint.clone())
            .context("Invalid execution service endpoint")?
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .timeout(Duration::from_secs(config.request_timeout_secs));

        let channel = endpoint
            .connect()
            .await
            .context("Failed to connect to execution service")?;

        let client = ExecutionServiceClient::new(channel);

        info!("✅ Connected to execution service");

        Ok(Self { client, config })
    }

    /// Submit a trading signal for execution
    ///
    /// This method converts a TradingSignal into an execution request
    /// and submits it to the execution service via gRPC.
    ///
    /// The exchange can be specified in the signal's metadata under the "exchange" key,
    /// or it will fall back to the default exchange in the configuration.
    pub async fn submit_signal(&mut self, signal: &TradingSignal) -> Result<SubmitSignalResponse> {
        debug!(
            "Submitting signal {} for {} ({:?})",
            signal.signal_id, signal.symbol, signal.signal_type
        );

        // Convert TradingSignal to SubmitSignalRequest
        let request = self.convert_signal_to_request(signal)?;

        // Submit with retries
        let mut attempt = 0;
        let mut last_error = None;

        while attempt < self.config.max_retries {
            attempt += 1;

            match self.submit_signal_once(request.clone()).await {
                Ok(response) => {
                    info!(
                        "✅ Signal {} submitted successfully (order_id: {}, exchange: {})",
                        signal.signal_id,
                        response.order_id.as_deref().unwrap_or("N/A"),
                        request.exchange
                    );
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.max_retries {
                        warn!(
                            "Signal submission failed (attempt {}/{}), retrying...",
                            attempt, self.config.max_retries
                        );
                        tokio::time::sleep(Duration::from_millis(
                            self.config.retry_backoff_ms * attempt as u64,
                        ))
                        .await;
                    }
                }
            }
        }

        error!(
            "❌ Failed to submit signal {} after {} attempts",
            signal.signal_id, self.config.max_retries
        );

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown error")))
    }

    /// Submit a signal to a specific exchange
    pub async fn submit_signal_to_exchange(
        &mut self,
        signal: &TradingSignal,
        exchange: Exchange,
    ) -> Result<SubmitSignalResponse> {
        // Create a modified signal with the exchange in metadata
        let mut modified_signal = signal.clone();
        modified_signal
            .metadata
            .insert("exchange".to_string(), exchange.as_str().to_string());

        self.submit_signal(&modified_signal).await
    }

    /// Submit signal once (internal method)
    async fn submit_signal_once(
        &mut self,
        request: SubmitSignalRequest,
    ) -> Result<SubmitSignalResponse> {
        let grpc_request = Request::new(request);

        let response = self
            .client
            .submit_signal(grpc_request)
            .await
            .context("gRPC call to execution service failed")?
            .into_inner();

        if !response.success {
            return Err(anyhow::anyhow!(
                "Execution service rejected signal: {}",
                response.message
            ));
        }

        Ok(SubmitSignalResponse {
            success: response.success,
            order_id: Some(response.order_id),
            internal_order_id: Some(response.internal_order_id),
            message: response.message,
            timestamp: response.timestamp,
        })
    }

    /// Determine which exchange to use for a signal
    fn determine_exchange(&self, signal: &TradingSignal) -> Result<Exchange> {
        // Check if exchange is specified in signal metadata
        if let Some(exchange_str) = signal.metadata.get("exchange") {
            if let Some(exchange) = Exchange::parse(exchange_str) {
                if !self.config.is_exchange_enabled(exchange) {
                    return Err(anyhow::anyhow!(
                        "Exchange '{}' is not enabled in configuration",
                        exchange_str
                    ));
                }
                return Ok(exchange);
            } else {
                return Err(anyhow::anyhow!(
                    "Unknown exchange '{}' specified in signal",
                    exchange_str
                ));
            }
        }

        // Check if explicit exchange is required
        if self.config.require_explicit_exchange {
            return Err(anyhow::anyhow!(
                "No exchange specified in signal and require_explicit_exchange is enabled"
            ));
        }

        // Fall back to default exchange
        Ok(self.config.default_exchange)
    }

    /// Convert TradingSignal to execution SubmitSignalRequest
    fn convert_signal_to_request(&self, signal: &TradingSignal) -> Result<SubmitSignalRequest> {
        // Determine which exchange to use
        let exchange = self.determine_exchange(signal)?;
        let exchange_config = self.config.get_exchange_config(exchange);

        // Determine side (Buy or Sell)
        let side = if signal.signal_type.is_bullish() {
            Side::Buy
        } else if signal.signal_type.is_bearish() {
            Side::Sell
        } else {
            warn!(
                "HOLD signal type, defaulting to BUY for signal {}",
                signal.signal_id
            );
            Side::Buy
        };

        // Calculate quantity based on confidence and exchange limits
        let base_quantity = exchange_config.default_quantity;
        let mut quantity = base_quantity * signal.confidence;

        // Apply exchange-specific limits
        quantity = quantity.max(exchange_config.min_quantity);
        quantity = quantity.min(exchange_config.max_quantity);

        if quantity <= 0.0 {
            return Err(anyhow::anyhow!(
                "Invalid quantity {} for signal {} (confidence: {})",
                quantity,
                signal.signal_id,
                signal.confidence
            ));
        }

        // Determine order type based on signal and exchange preferences
        let order_type = if exchange_config.prefer_market_orders {
            OrderType::Market
        } else if signal.entry_price.is_some() {
            OrderType::Limit
        } else {
            OrderType::Market
        };

        // Transform symbol for this exchange's format
        let symbol = exchange_config.symbol_format.transform(&signal.symbol);

        // Build metadata
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "janus-forward".to_string());
        metadata.insert("timeframe".to_string(), signal.timeframe.to_string());
        metadata.insert("confidence".to_string(), signal.confidence.to_string());
        metadata.insert("strength".to_string(), signal.strength.to_string());
        metadata.insert("original_symbol".to_string(), signal.symbol.clone());

        // Add signal source information
        metadata.insert("signal_source".to_string(), format!("{:?}", signal.source));

        if let Some(stop_loss) = signal.stop_loss {
            metadata.insert("stop_loss".to_string(), stop_loss.to_string());
        }

        if let Some(take_profit) = signal.take_profit {
            metadata.insert("take_profit".to_string(), take_profit.to_string());
        }

        // Copy over any additional metadata from the signal
        for (key, value) in &signal.metadata {
            if key != "exchange" {
                // Don't duplicate exchange
                metadata.insert(key.clone(), value.clone());
            }
        }

        // Create the request
        let request = SubmitSignalRequest {
            signal_id: signal.signal_id.clone(),
            symbol,
            exchange: exchange.as_str().to_string(),
            side: side as i32,
            quantity,
            order_type: order_type as i32,
            price: signal.entry_price,
            stop_price: signal.stop_loss,
            time_in_force: exchange_config.default_time_in_force as i32,
            strategy: self.config.default_strategy as i32,
            metadata,
        };

        Ok(request)
    }

    /// Check if the execution service is healthy
    pub async fn health_check(&mut self) -> Result<bool> {
        debug!("Performing health check on execution service");

        let request = Request::new(fks_proto::common::HealthCheckRequest {});

        match self.client.health_check(request).await {
            Ok(response) => {
                let health: fks_proto::common::HealthCheckResponse = response.into_inner();
                debug!(
                    "Execution service health: {} (status: {})",
                    health.healthy, health.status
                );
                Ok(health.healthy)
            }
            Err(e) => {
                warn!("Execution service health check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Get the execution service endpoint
    pub fn endpoint(&self) -> &str {
        &self.config.endpoint
    }

    /// Get the current configuration
    pub fn config(&self) -> &ExecutionClientConfig {
        &self.config
    }

    /// Get the default exchange
    pub fn default_exchange(&self) -> Exchange {
        self.config.default_exchange
    }

    /// List all enabled exchanges
    pub fn enabled_exchanges(&self) -> Vec<Exchange> {
        self.config
            .exchange_configs
            .iter()
            .filter(|(_, config)| config.enabled)
            .map(|(exchange, _)| *exchange)
            .collect()
    }
}

/// Response from signal submission
#[derive(Debug, Clone)]
pub struct SubmitSignalResponse {
    pub success: bool,
    pub order_id: Option<String>,
    pub internal_order_id: Option<String>,
    pub message: String,
    pub timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{SignalType, Timeframe};

    fn create_test_signal() -> TradingSignal {
        TradingSignal {
            signal_id: "test-signal-123".to_string(),
            symbol: "BTCUSD".to_string(),
            timeframe: Timeframe::M15,
            signal_type: SignalType::Buy,
            confidence: 0.85,
            strength: 0.75,
            entry_price: Some(50000.0),
            stop_loss: Some(49000.0),
            take_profit: Some(52000.0),
            metadata: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            source: crate::signal::SignalSource::Strategy {
                name: "TestStrategy".to_string(),
            },
            predicted_duration_seconds: None,
        }
    }

    #[test]
    fn test_exchange_parse() {
        assert_eq!(Exchange::parse("bybit"), Some(Exchange::Bybit));
        assert_eq!(Exchange::parse("BYBIT"), Some(Exchange::Bybit));
        assert_eq!(Exchange::parse("binance"), Some(Exchange::Binance));
        assert_eq!(Exchange::parse("kraken"), Some(Exchange::Kraken));
        assert_eq!(Exchange::parse("paper"), Some(Exchange::Paper));
        assert_eq!(Exchange::parse("sim"), Some(Exchange::Paper));
        assert_eq!(Exchange::parse("unknown"), None);
    }

    #[test]
    fn test_symbol_format_transform() {
        assert_eq!(SymbolFormat::Concatenated.transform("BTCUSD"), "BTCUSD");
        assert_eq!(SymbolFormat::SlashSeparated.transform("BTCUSD"), "BTC/USD");
        assert_eq!(SymbolFormat::DashSeparated.transform("BTCUSD"), "BTC-USD");
        assert_eq!(
            SymbolFormat::UnderscoreSeparated.transform("BTCUSD"),
            "BTC_USD"
        );
        // Test with USDT pairs
        assert_eq!(
            SymbolFormat::SlashSeparated.transform("BTCUSDT"),
            "BTC/USDT"
        );
        assert_eq!(SymbolFormat::SlashSeparated.transform("ETHBTC"), "ETH/BTC");
    }

    #[tokio::test]
    async fn test_convert_signal_to_request_default_exchange() {
        let config = ExecutionClientConfig::default();
        let client = ExecutionClient {
            client: ExecutionServiceClient::new(
                Channel::from_static("http://localhost:50052").connect_lazy(),
            ),
            config,
        };

        let signal = create_test_signal();
        let request = client.convert_signal_to_request(&signal).unwrap();

        assert_eq!(request.signal_id, "test-signal-123");
        assert_eq!(request.symbol, "BTCUSD");
        assert_eq!(request.exchange, "paper"); // Default is now paper
        assert_eq!(request.side, Side::Buy as i32);
    }

    #[tokio::test]
    async fn test_convert_signal_to_request_with_exchange_in_metadata() {
        let config = ExecutionClientConfig::default();
        let client = ExecutionClient {
            client: ExecutionServiceClient::new(
                Channel::from_static("http://localhost:50052").connect_lazy(),
            ),
            config,
        };

        let mut signal = create_test_signal();
        signal
            .metadata
            .insert("exchange".to_string(), "binance".to_string());

        let request = client.convert_signal_to_request(&signal).unwrap();

        assert_eq!(request.exchange, "binance");
    }

    #[tokio::test]
    async fn test_convert_signal_to_request_kraken_symbol_format() {
        let config = ExecutionClientConfig::builder()
            .default_exchange(Exchange::Kraken)
            .build();

        let client = ExecutionClient {
            client: ExecutionServiceClient::new(
                Channel::from_static("http://localhost:50052").connect_lazy(),
            ),
            config,
        };

        let signal = create_test_signal();
        let request = client.convert_signal_to_request(&signal).unwrap();

        assert_eq!(request.exchange, "kraken");
        assert_eq!(request.symbol, "BTC/USD"); // Kraken uses slash-separated format (USD from BTCUSD)
    }

    #[test]
    fn test_config_builder() {
        let config = ExecutionClientConfig::builder()
            .endpoint("http://localhost:9999")
            .default_exchange(Exchange::Binance)
            .max_retries(5)
            .require_explicit_exchange(true)
            .build();

        assert_eq!(config.endpoint, "http://localhost:9999");
        assert_eq!(config.default_exchange, Exchange::Binance);
        assert_eq!(config.max_retries, 5);
        assert!(config.require_explicit_exchange);
    }

    #[tokio::test]
    async fn test_enabled_exchanges() {
        let config = ExecutionClientConfig::default();
        let client = ExecutionClient {
            client: ExecutionServiceClient::new(
                Channel::from_static("http://localhost:50052").connect_lazy(),
            ),
            config,
        };

        let enabled = client.enabled_exchanges();
        assert!(enabled.contains(&Exchange::Paper));
        assert!(enabled.contains(&Exchange::Bybit));
        assert!(enabled.contains(&Exchange::Binance));
    }

    #[test]
    fn test_config_default() {
        let config = ExecutionClientConfig::default();
        assert_eq!(config.endpoint, "http://execution:50052");
        assert_eq!(config.connect_timeout_secs, 10);
        assert_eq!(config.request_timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.default_exchange, Exchange::Paper);
        assert!(!config.require_explicit_exchange);
    }
}
