//! Market Data Provider Trait for Multi-Exchange WebSocket Support
//!
//! This module defines the `MarketDataProvider` trait that all exchange
//! implementations must implement for streaming market data.
//!
//! # Supported Exchanges
//!
//! | Exchange | WebSocket URL | Free Data |
//! |----------|---------------|-----------|
//! | Bybit    | wss://stream.bybit.com/v5/public/spot | ✅ |
//! | Kraken   | wss://ws.kraken.com/v2 | ✅ |
//! | Binance  | wss://stream.binance.com:9443/ws | ✅ |
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::exchanges::{MarketDataProvider, ExchangeId};
//!
//! async fn stream_market_data(provider: impl MarketDataProvider) {
//!     // Subscribe to tickers
//!     provider.subscribe_ticker(&["BTC/USDT", "ETH/USDT"]).await.unwrap();
//!
//!     // Get the event stream
//!     let mut receiver = provider.subscribe_events();
//!
//!     while let Some(event) = receiver.recv().await {
//!         println!("Received: {:?}", event);
//!     }
//! }
//! ```

use crate::error::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::broadcast;

use super::market_data::{
    BestPrice, Candle, ExchangeId, MarketDataEvent, OrderBook, Subscription, Ticker, TradingPair,
};

// ============================================================================
// Market Data Provider Trait
// ============================================================================

/// Market data provider trait for WebSocket streaming
///
/// This trait provides a unified interface for receiving market data from
/// different exchanges. All market data streams are FREE and do not require
/// API keys.
///
/// # Implementation Notes
///
/// - Implementors must handle automatic reconnection
/// - All symbols should be normalized to "BASE/QUOTE" format internally
/// - Events should be broadcast to all subscribers
#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    // =========================================================================
    // Identification
    // =========================================================================

    /// Get the exchange identifier
    fn exchange_id(&self) -> ExchangeId;

    /// Get a human-readable name for this provider
    fn name(&self) -> &str {
        self.exchange_id().name()
    }

    // =========================================================================
    // Connection Management
    // =========================================================================

    /// Connect to the exchange WebSocket
    ///
    /// This should establish the WebSocket connection and start the
    /// message processing loop. Returns when connection is established.
    async fn connect(&self) -> Result<()>;

    /// Disconnect from the exchange WebSocket
    ///
    /// This should gracefully close the connection and stop message processing.
    async fn disconnect(&self) -> Result<()>;

    /// Check if currently connected
    fn is_connected(&self) -> bool;

    /// Get connection health/status
    async fn health_check(&self) -> Result<ConnectionHealth>;

    // =========================================================================
    // Subscriptions - Ticker (Level 1)
    // =========================================================================

    /// Subscribe to ticker updates for the given symbols
    ///
    /// Ticker data includes:
    /// - Best bid/ask prices and quantities
    /// - Last trade price
    /// - 24h volume, high, low, change
    ///
    /// # Arguments
    /// * `symbols` - Normalized symbols like "BTC/USDT", "ETH/USDT"
    ///
    /// # Example
    /// ```rust,ignore
    /// provider.subscribe_ticker(&["BTC/USDT", "ETH/USDT"]).await?;
    /// ```
    async fn subscribe_ticker(&self, symbols: &[&str]) -> Result<()>;

    /// Unsubscribe from ticker updates
    async fn unsubscribe_ticker(&self, symbols: &[&str]) -> Result<()>;

    /// Get current ticker for a symbol (from cache)
    fn get_ticker(&self, symbol: &str) -> Option<Ticker>;

    // =========================================================================
    // Subscriptions - Trades
    // =========================================================================

    /// Subscribe to trade stream for the given symbols
    ///
    /// Trade data includes:
    /// - Trade price and quantity
    /// - Trade side (buy/sell)
    /// - Trade timestamp
    ///
    /// # Arguments
    /// * `symbols` - Normalized symbols like "BTC/USDT"
    async fn subscribe_trades(&self, symbols: &[&str]) -> Result<()>;

    /// Unsubscribe from trade stream
    async fn unsubscribe_trades(&self, symbols: &[&str]) -> Result<()>;

    // =========================================================================
    // Subscriptions - Candles/OHLCV
    // =========================================================================

    /// Subscribe to candle/OHLCV updates
    ///
    /// # Arguments
    /// * `symbols` - Normalized symbols
    /// * `interval` - Candle interval in minutes (1, 5, 15, 60, 240, 1440)
    ///
    /// # Supported Intervals
    ///
    /// | Minutes | Human Readable |
    /// |---------|----------------|
    /// | 1       | 1 minute       |
    /// | 5       | 5 minutes      |
    /// | 15      | 15 minutes     |
    /// | 30      | 30 minutes     |
    /// | 60      | 1 hour         |
    /// | 240     | 4 hours        |
    /// | 1440    | 1 day          |
    async fn subscribe_candles(&self, symbols: &[&str], interval: u32) -> Result<()>;

    /// Unsubscribe from candle updates
    async fn unsubscribe_candles(&self, symbols: &[&str], interval: u32) -> Result<()>;

    /// Get latest candle for a symbol (from cache)
    fn get_candle(&self, symbol: &str, interval: u32) -> Option<Candle>;

    // =========================================================================
    // Subscriptions - Order Book
    // =========================================================================

    /// Subscribe to order book snapshots
    ///
    /// # Arguments
    /// * `symbols` - Normalized symbols
    /// * `depth` - Number of levels (5, 10, 20, 50)
    async fn subscribe_order_book(&self, symbols: &[&str], depth: u32) -> Result<()>;

    /// Unsubscribe from order book
    async fn unsubscribe_order_book(&self, symbols: &[&str]) -> Result<()>;

    /// Get current order book for a symbol (from cache)
    fn get_order_book(&self, symbol: &str) -> Option<OrderBook>;

    // =========================================================================
    // Event Stream
    // =========================================================================

    /// Subscribe to the market data event stream
    ///
    /// Returns a broadcast receiver that will receive all market data events.
    /// Multiple subscribers can receive the same events.
    fn subscribe_events(&self) -> broadcast::Receiver<MarketDataEvent>;

    // =========================================================================
    // Utilities
    // =========================================================================

    /// Convert a normalized symbol to exchange-specific format
    fn to_exchange_symbol(&self, normalized: &str) -> String {
        if let Some(pair) = TradingPair::from_normalized(normalized) {
            pair.to_exchange(self.exchange_id())
        } else {
            normalized.to_string()
        }
    }

    /// Get all currently subscribed symbols
    fn subscribed_symbols(&self) -> Vec<String>;

    /// Get all active subscriptions
    fn active_subscriptions(&self) -> Vec<Subscription>;
}

// ============================================================================
// Connection Health
// ============================================================================

/// Connection health information
#[derive(Debug, Clone)]
pub struct ConnectionHealth {
    /// Exchange identifier
    pub exchange: ExchangeId,
    /// Is currently connected
    pub connected: bool,
    /// Last successful message timestamp
    pub last_message: Option<chrono::DateTime<chrono::Utc>>,
    /// Number of reconnection attempts
    pub reconnect_count: u32,
    /// Current latency in milliseconds
    pub latency_ms: Option<u64>,
    /// Error message if unhealthy
    pub error: Option<String>,
}

impl ConnectionHealth {
    pub fn healthy(exchange: ExchangeId) -> Self {
        Self {
            exchange,
            connected: true,
            last_message: Some(chrono::Utc::now()),
            reconnect_count: 0,
            latency_ms: None,
            error: None,
        }
    }

    pub fn disconnected(exchange: ExchangeId, error: Option<String>) -> Self {
        Self {
            exchange,
            connected: false,
            last_message: None,
            reconnect_count: 0,
            latency_ms: None,
            error,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.connected && self.error.is_none()
    }
}

// ============================================================================
// Multi-Exchange Aggregator
// ============================================================================

/// Aggregates market data from multiple exchanges
///
/// This struct manages multiple `MarketDataProvider` instances and provides
/// unified access to market data across all exchanges.
///
/// # Example
///
/// ```rust,ignore
/// let mut aggregator = MarketDataAggregator::new();
/// aggregator.add_provider(Arc::new(BybitProvider::new()));
/// aggregator.add_provider(Arc::new(KrakenProvider::new()));
/// aggregator.add_provider(Arc::new(BinanceProvider::new()));
///
/// // Connect all
/// aggregator.connect_all().await?;
///
/// // Subscribe to tickers on all exchanges
/// aggregator.subscribe_ticker_all(&["BTC/USDT"]).await?;
///
/// // Get best price across all exchanges
/// let best = aggregator.get_best_price("BTC/USDT");
/// ```
pub struct MarketDataAggregator {
    providers: Vec<Arc<dyn MarketDataProvider>>,
    event_tx: broadcast::Sender<MarketDataEvent>,
    /// Handles for the event forwarder tasks
    forwarder_handles: std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl MarketDataAggregator {
    /// Create a new aggregator
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(10000);
        Self {
            providers: Vec::new(),
            event_tx,
            forwarder_handles: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Add a market data provider and start forwarding its events
    ///
    /// This spawns a background task that forwards all events from the provider
    /// to the aggregator's unified event channel, enabling the ArbitrageBridge
    /// and other consumers to receive events from all providers.
    pub fn add_provider(&mut self, provider: Arc<dyn MarketDataProvider>) {
        // Subscribe to the provider's events
        let mut provider_rx = provider.subscribe_events();
        let aggregator_tx = self.event_tx.clone();
        let exchange_id = provider.exchange_id();

        // Spawn a task to forward events from this provider to the aggregator
        let handle = tokio::spawn(async move {
            tracing::debug!("Started event forwarder for {:?}", exchange_id);
            loop {
                match provider_rx.recv().await {
                    Ok(event) => {
                        // Forward the event to the aggregator's channel
                        if let Err(e) = aggregator_tx.send(event) {
                            tracing::trace!(
                                "No subscribers for aggregator events from {:?}: {}",
                                exchange_id,
                                e
                            );
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            "Event forwarder for {:?} lagged by {} messages",
                            exchange_id,
                            n
                        );
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::debug!("Event channel closed for {:?}", exchange_id);
                        break;
                    }
                }
            }
            tracing::debug!("Event forwarder stopped for {:?}", exchange_id);
        });

        // Store the handle so we can clean up later if needed
        if let Ok(mut handles) = self.forwarder_handles.lock() {
            handles.push(handle);
        }

        self.providers.push(provider);
    }

    /// Get all provider exchange IDs
    pub fn exchanges(&self) -> Vec<ExchangeId> {
        self.providers.iter().map(|p| p.exchange_id()).collect()
    }

    /// Get a specific provider by exchange ID
    pub fn get_provider(&self, exchange: ExchangeId) -> Option<&Arc<dyn MarketDataProvider>> {
        self.providers.iter().find(|p| p.exchange_id() == exchange)
    }

    /// Connect all providers
    pub async fn connect_all(&self) -> Result<()> {
        for provider in &self.providers {
            provider.connect().await?;
        }
        Ok(())
    }

    /// Disconnect all providers
    pub async fn disconnect_all(&self) -> Result<()> {
        for provider in &self.providers {
            provider.disconnect().await?;
        }
        Ok(())
    }

    /// Subscribe to ticker on all exchanges
    pub async fn subscribe_ticker_all(&self, symbols: &[&str]) -> Result<()> {
        for provider in &self.providers {
            provider.subscribe_ticker(symbols).await?;
        }
        Ok(())
    }

    /// Subscribe to trades on all exchanges
    pub async fn subscribe_trades_all(&self, symbols: &[&str]) -> Result<()> {
        for provider in &self.providers {
            provider.subscribe_trades(symbols).await?;
        }
        Ok(())
    }

    /// Subscribe to candles on all exchanges
    pub async fn subscribe_candles_all(&self, symbols: &[&str], interval: u32) -> Result<()> {
        for provider in &self.providers {
            provider.subscribe_candles(symbols, interval).await?;
        }
        Ok(())
    }

    /// Get best price across all exchanges
    pub fn get_best_price(&self, symbol: &str) -> Option<BestPrice> {
        let mut best_bid = rust_decimal::Decimal::ZERO;
        let mut best_bid_exchange = None;
        let mut best_ask = rust_decimal::Decimal::MAX;
        let mut best_ask_exchange = None;
        let mut tickers = std::collections::HashMap::new();

        for provider in &self.providers {
            if let Some(ticker) = provider.get_ticker(symbol) {
                if ticker.bid > best_bid {
                    best_bid = ticker.bid;
                    best_bid_exchange = Some(provider.exchange_id());
                }
                if ticker.ask < best_ask {
                    best_ask = ticker.ask;
                    best_ask_exchange = Some(provider.exchange_id());
                }
                tickers.insert(provider.exchange_id(), ticker);
            }
        }

        match (best_bid_exchange, best_ask_exchange) {
            (Some(bid_ex), Some(ask_ex)) => Some(BestPrice {
                symbol: symbol.to_string(),
                best_bid,
                best_bid_exchange: bid_ex,
                best_ask,
                best_ask_exchange: ask_ex,
                spread: best_ask - best_bid,
                tickers,
                timestamp: chrono::Utc::now(),
            }),
            _ => None,
        }
    }

    /// Get health status of all providers
    pub async fn health_check_all(&self) -> Vec<ConnectionHealth> {
        let mut results = Vec::new();
        for provider in &self.providers {
            match provider.health_check().await {
                Ok(health) => results.push(health),
                Err(e) => results.push(ConnectionHealth::disconnected(
                    provider.exchange_id(),
                    Some(e.to_string()),
                )),
            }
        }
        results
    }

    /// Subscribe to aggregated event stream
    pub fn subscribe_events(&self) -> broadcast::Receiver<MarketDataEvent> {
        self.event_tx.subscribe()
    }

    /// Get event sender (for providers to broadcast events)
    pub fn event_sender(&self) -> broadcast::Sender<MarketDataEvent> {
        self.event_tx.clone()
    }
}

impl Default for MarketDataAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MarketDataAggregator {
    fn drop(&mut self) {
        // Abort all forwarder tasks when the aggregator is dropped
        if let Ok(handles) = self.forwarder_handles.lock() {
            for handle in handles.iter() {
                handle.abort();
            }
        }
    }
}

// ============================================================================
// Provider Configuration
// ============================================================================

/// Configuration for a market data provider
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Exchange identifier
    pub exchange: ExchangeId,
    /// Whether to use testnet
    pub testnet: bool,
    /// Custom WebSocket URL (overrides default)
    pub ws_url: Option<String>,
    /// Reconnection settings
    pub reconnect: ReconnectConfig,
    /// Subscription settings
    pub subscription: SubscriptionConfig,
}

impl ProviderConfig {
    pub fn new(exchange: ExchangeId) -> Self {
        Self {
            exchange,
            testnet: false,
            ws_url: None,
            reconnect: ReconnectConfig::default(),
            subscription: SubscriptionConfig::default(),
        }
    }

    pub fn testnet(mut self) -> Self {
        self.testnet = true;
        self
    }

    pub fn with_ws_url(mut self, url: impl Into<String>) -> Self {
        self.ws_url = Some(url.into());
        self
    }

    /// Get the WebSocket URL to use
    pub fn get_ws_url(&self) -> &str {
        if let Some(ref url) = self.ws_url {
            url
        } else if self.testnet {
            self.exchange
                .testnet_ws_url()
                .unwrap_or(self.exchange.market_data_ws_url())
        } else {
            self.exchange.market_data_ws_url()
        }
    }
}

/// Reconnection configuration
#[derive(Debug, Clone)]
pub struct ReconnectConfig {
    /// Enable automatic reconnection
    pub enabled: bool,
    /// Initial delay before reconnecting (milliseconds)
    pub initial_delay_ms: u64,
    /// Maximum delay between reconnection attempts (milliseconds)
    pub max_delay_ms: u64,
    /// Maximum number of reconnection attempts (None = infinite)
    pub max_attempts: Option<u32>,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            max_attempts: None,
            backoff_multiplier: 2.0,
        }
    }
}

/// Subscription configuration
#[derive(Debug, Clone)]
pub struct SubscriptionConfig {
    /// Default order book depth
    pub default_depth: u32,
    /// Default candle interval (minutes)
    pub default_candle_interval: u32,
    /// Auto-resubscribe on reconnection
    pub resubscribe_on_reconnect: bool,
}

impl Default for SubscriptionConfig {
    fn default() -> Self {
        Self {
            default_depth: 10,
            default_candle_interval: 5,
            resubscribe_on_reconnect: true,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_health_healthy() {
        let health = ConnectionHealth::healthy(ExchangeId::Bybit);
        assert!(health.is_healthy());
        assert!(health.connected);
        assert!(health.error.is_none());
    }

    #[test]
    fn test_connection_health_disconnected() {
        let health = ConnectionHealth::disconnected(
            ExchangeId::Kraken,
            Some("Connection timeout".to_string()),
        );
        assert!(!health.is_healthy());
        assert!(!health.connected);
        assert!(health.error.is_some());
    }

    #[test]
    fn test_provider_config() {
        let config = ProviderConfig::new(ExchangeId::Binance).testnet();
        assert!(config.testnet);
        assert_eq!(config.exchange, ExchangeId::Binance);
    }

    #[test]
    fn test_provider_config_ws_url() {
        let config = ProviderConfig::new(ExchangeId::Bybit);
        assert_eq!(config.get_ws_url(), "wss://stream.bybit.com/v5/public/spot");

        let config = ProviderConfig::new(ExchangeId::Bybit).testnet();
        assert_eq!(
            config.get_ws_url(),
            "wss://stream-testnet.bybit.com/v5/public/spot"
        );

        let config = ProviderConfig::new(ExchangeId::Bybit).with_ws_url("wss://custom.example.com");
        assert_eq!(config.get_ws_url(), "wss://custom.example.com");
    }

    #[test]
    fn test_reconnect_config_default() {
        let config = ReconnectConfig::default();
        assert!(config.enabled);
        assert_eq!(config.initial_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 30000);
        assert_eq!(config.backoff_multiplier, 2.0);
    }

    #[test]
    fn test_aggregator_creation() {
        let aggregator = MarketDataAggregator::new();
        assert!(aggregator.exchanges().is_empty());
    }
}
