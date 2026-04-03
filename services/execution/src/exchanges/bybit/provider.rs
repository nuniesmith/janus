//! Bybit Market Data Provider
//!
//! This module implements the `MarketDataProvider` trait for Bybit,
//! providing a unified interface for market data streaming using
//! the Bybit public WebSocket API.
//!
//! # Example
//!
//! ```ignore
//! use janus_execution::exchanges::bybit::BybitProvider;
//! use janus_execution::exchanges::provider::{MarketDataProvider, MarketDataAggregator};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = Arc::new(BybitProvider::new());
//!
//!     // Connect to Bybit
//!     provider.connect().await?;
//!
//!     // Subscribe to tickers
//!     provider.subscribe_ticker(&["BTC/USDT", "ETH/USDT"]).await?;
//!
//!     // Listen for events
//!     let mut rx = provider.subscribe_events();
//!     while let Ok(event) = rx.recv().await {
//!         println!("Event: {:?}", event);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{ExecutionError, Result};
use crate::exchanges::bybit::public_websocket::{
    BybitCategory, BybitPublicEvent, BybitPublicWebSocket, BybitPublicWsConfig,
};
use crate::exchanges::market_data::{
    Candle, ExchangeId, MarketDataEvent, OrderBook, Subscription, SubscriptionChannel, Ticker,
};
use crate::exchanges::provider::{ConnectionHealth, MarketDataProvider};
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use tokio::sync::RwLock;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Bybit public WebSocket URLs (re-exported for convenience)
pub use crate::exchanges::bybit::public_websocket::{
    WS_PUBLIC_LINEAR_MAINNET, WS_PUBLIC_LINEAR_TESTNET, WS_PUBLIC_SPOT_MAINNET,
    WS_PUBLIC_SPOT_TESTNET,
};

/// Shared state that can be safely passed to spawned tasks
struct SharedState {
    /// Connection state
    connected: AtomicBool,

    /// Reconnect count
    reconnect_count: AtomicU32,

    /// Last message timestamp
    last_message: RwLock<Option<chrono::DateTime<Utc>>>,

    /// Ticker cache (normalized symbol -> Ticker)
    ticker_cache: RwLock<HashMap<String, Ticker>>,

    /// Candle cache (symbol:interval -> Candle)
    candle_cache: RwLock<HashMap<String, Candle>>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            connected: AtomicBool::new(false),
            reconnect_count: AtomicU32::new(0),
            last_message: RwLock::new(None),
            ticker_cache: RwLock::new(HashMap::new()),
            candle_cache: RwLock::new(HashMap::new()),
        }
    }
}

/// Bybit provider configuration
#[derive(Debug, Clone)]
pub struct BybitProviderConfig {
    /// Use testnet instead of mainnet
    pub testnet: bool,
    /// Market category (Spot or Linear perpetuals)
    pub category: BybitCategory,
    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,
    /// Maximum reconnect attempts
    pub max_reconnect_attempts: u32,
    /// Reconnect delay in seconds
    pub reconnect_delay_secs: u64,
}

impl Default for BybitProviderConfig {
    fn default() -> Self {
        Self {
            testnet: false,
            category: BybitCategory::Spot,
            auto_reconnect: true,
            max_reconnect_attempts: 10,
            reconnect_delay_secs: 5,
        }
    }
}

/// Bybit market data provider implementing the `MarketDataProvider` trait
///
/// This provider uses the Bybit public WebSocket API to stream real-time
/// market data including tickers, trades, and candles.
pub struct BybitProvider {
    /// Configuration
    config: BybitProviderConfig,

    /// Shared state accessible from spawned tasks
    state: Arc<SharedState>,

    /// The underlying WebSocket client
    ws_client: Arc<RwLock<Option<Arc<BybitPublicWebSocket>>>>,

    /// Active subscriptions tracking
    subscribed_symbols: RwLock<Vec<String>>,

    /// Market data event sender for broadcasting to aggregator
    market_event_tx: broadcast::Sender<MarketDataEvent>,

    /// Subscription tracking for trades
    trade_subscriptions: RwLock<Vec<String>>,

    /// Subscription tracking for candles (interval -> symbols)
    candle_subscriptions: RwLock<HashMap<u32, Vec<String>>>,
}

impl BybitProvider {
    /// Create a new Bybit provider with default configuration
    pub fn new() -> Self {
        Self::with_config(BybitProviderConfig::default())
    }

    /// Create a new Bybit provider with custom configuration
    pub fn with_config(config: BybitProviderConfig) -> Self {
        let (market_event_tx, _) = broadcast::channel(10000);

        Self {
            config,
            state: Arc::new(SharedState::new()),
            ws_client: Arc::new(RwLock::new(None)),
            subscribed_symbols: RwLock::new(Vec::new()),
            market_event_tx,
            trade_subscriptions: RwLock::new(Vec::new()),
            candle_subscriptions: RwLock::new(HashMap::new()),
        }
    }

    /// Create a provider configured for testnet
    pub fn testnet() -> Self {
        Self::with_config(BybitProviderConfig {
            testnet: true,
            ..Default::default()
        })
    }

    /// Create a provider for linear perpetuals
    pub fn linear(testnet: bool) -> Self {
        Self::with_config(BybitProviderConfig {
            testnet,
            category: BybitCategory::Linear,
            ..Default::default()
        })
    }

    /// Get the market data event sender for external use
    pub fn event_sender(&self) -> broadcast::Sender<MarketDataEvent> {
        self.market_event_tx.clone()
    }

    /// Get the WebSocket URL based on configuration
    pub fn ws_url(&self) -> &str {
        match (self.config.testnet, self.config.category) {
            (false, BybitCategory::Spot) => WS_PUBLIC_SPOT_MAINNET,
            (true, BybitCategory::Spot) => WS_PUBLIC_SPOT_TESTNET,
            (false, BybitCategory::Linear) => WS_PUBLIC_LINEAR_MAINNET,
            (true, BybitCategory::Linear) => WS_PUBLIC_LINEAR_TESTNET,
        }
    }

    /// Manually update ticker cache (for use by external price feeds or testing)
    pub async fn update_ticker(&self, ticker: Ticker) {
        let symbol = ticker.symbol.clone();
        self.state
            .ticker_cache
            .write()
            .await
            .insert(symbol, ticker.clone());
        *self.state.last_message.write().await = Some(Utc::now());
        let _ = self.market_event_tx.send(MarketDataEvent::Ticker(ticker));
    }

    /// Manually update candle cache (for use by external price feeds or testing)
    pub async fn update_candle(&self, candle: Candle) {
        let cache_key = format!("{}:{}", candle.symbol, candle.interval);
        self.state
            .candle_cache
            .write()
            .await
            .insert(cache_key, candle.clone());
        *self.state.last_message.write().await = Some(Utc::now());
        let _ = self.market_event_tx.send(MarketDataEvent::Candle(candle));
    }

    /// Start the event forwarding task that converts Bybit events to unified events
    fn start_event_forwarder(
        state: Arc<SharedState>,
        market_event_tx: broadcast::Sender<MarketDataEvent>,
        mut ws_rx: broadcast::Receiver<BybitPublicEvent>,
    ) {
        tokio::spawn(async move {
            loop {
                match ws_rx.recv().await {
                    Ok(event) => {
                        // Update last message timestamp
                        *state.last_message.write().await = Some(Utc::now());

                        // Update connection state
                        match &event {
                            BybitPublicEvent::Connected => {
                                state.connected.store(true, Ordering::SeqCst);
                                state.reconnect_count.store(0, Ordering::SeqCst);
                            }
                            BybitPublicEvent::Disconnected => {
                                state.connected.store(false, Ordering::SeqCst);
                                state.reconnect_count.fetch_add(1, Ordering::SeqCst);
                            }
                            BybitPublicEvent::Ticker(ticker) => {
                                // Update ticker cache
                                let unified = ticker.to_ticker();
                                state
                                    .ticker_cache
                                    .write()
                                    .await
                                    .insert(unified.symbol.clone(), unified);
                            }
                            BybitPublicEvent::Candle(candle) => {
                                // Update candle cache
                                let unified = candle.to_candle();
                                let cache_key = format!("{}:{}", unified.symbol, unified.interval);
                                state.candle_cache.write().await.insert(cache_key, unified);
                            }
                            _ => {}
                        }

                        // Convert to unified event and broadcast
                        if let Some(market_event) = event.to_market_data_event() {
                            let _ = market_event_tx.send(market_event);
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("Bybit event forwarder lagged by {} messages", n);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("Bybit event channel closed");
                        break;
                    }
                }
            }
        });
    }
}

impl Default for BybitProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MarketDataProvider for BybitProvider {
    fn exchange_id(&self) -> ExchangeId {
        ExchangeId::Bybit
    }

    fn name(&self) -> &str {
        "Bybit"
    }

    async fn connect(&self) -> Result<()> {
        info!("Connecting Bybit provider to {}", self.ws_url());

        // Check if already connected
        if self.state.connected.load(Ordering::SeqCst) {
            warn!("Bybit provider already connected");
            return Ok(());
        }

        // Create WebSocket config from provider config
        let ws_config = BybitPublicWsConfig {
            testnet: self.config.testnet,
            category: self.config.category,
            auto_reconnect: self.config.auto_reconnect,
            max_reconnect_attempts: self.config.max_reconnect_attempts,
            reconnect_delay_secs: self.config.reconnect_delay_secs,
            ..Default::default()
        };

        // Create new WebSocket client
        let ws = Arc::new(BybitPublicWebSocket::new(ws_config));

        // Get pending subscriptions and apply them
        let ticker_syms = self.subscribed_symbols.read().await.clone();
        if !ticker_syms.is_empty() {
            let syms: Vec<&str> = ticker_syms.iter().map(|s| s.as_str()).collect();
            ws.subscribe_ticker(&syms).await?;
        }

        let trade_syms = self.trade_subscriptions.read().await.clone();
        if !trade_syms.is_empty() {
            let syms: Vec<&str> = trade_syms.iter().map(|s| s.as_str()).collect();
            ws.subscribe_trades(&syms).await?;
        }

        let candle_subs = self.candle_subscriptions.read().await.clone();
        for (interval, symbols) in candle_subs {
            if !symbols.is_empty() {
                let syms: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
                let interval_str = interval_to_bybit_string(interval);
                ws.subscribe_candles(&syms, &interval_str).await?;
            }
        }

        // Start event forwarder
        let ws_rx = ws.subscribe_events();
        Self::start_event_forwarder(self.state.clone(), self.market_event_tx.clone(), ws_rx);

        // Start WebSocket client
        ws.start().await?;

        // Store the client
        *self.ws_client.write().await = Some(ws);

        // Mark as connected (will be confirmed by Connected event)
        self.state.connected.store(true, Ordering::SeqCst);

        let _ = self
            .market_event_tx
            .send(MarketDataEvent::ConnectionStatus {
                exchange: ExchangeId::Bybit,
                connected: true,
                message: Some("Connected".to_string()),
                timestamp: Utc::now(),
            });

        info!("Bybit provider connected");
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        info!("Disconnecting Bybit provider...");

        // Stop the WebSocket client
        if let Some(ws) = self.ws_client.read().await.as_ref() {
            ws.stop().await;
        }

        // Clear the client
        *self.ws_client.write().await = None;

        self.state.connected.store(false, Ordering::SeqCst);

        let _ = self
            .market_event_tx
            .send(MarketDataEvent::ConnectionStatus {
                exchange: ExchangeId::Bybit,
                connected: false,
                message: Some("Disconnected".to_string()),
                timestamp: Utc::now(),
            });

        info!("Bybit provider disconnected");
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.state.connected.load(Ordering::SeqCst)
    }

    async fn health_check(&self) -> Result<ConnectionHealth> {
        let connected = self.is_connected();
        let last_message = *self.state.last_message.read().await;
        let reconnect_count = self.state.reconnect_count.load(Ordering::SeqCst);

        // Calculate latency if we have a last message
        let latency_ms = last_message.map(|t| {
            Utc::now()
                .signed_duration_since(t)
                .num_milliseconds()
                .max(0) as u64
        });

        if connected {
            Ok(ConnectionHealth {
                exchange: ExchangeId::Bybit,
                connected: true,
                last_message,
                reconnect_count,
                latency_ms,
                error: None,
            })
        } else {
            Ok(ConnectionHealth {
                exchange: ExchangeId::Bybit,
                connected: false,
                last_message,
                reconnect_count,
                latency_ms: None,
                error: Some("Not connected".to_string()),
            })
        }
    }

    async fn subscribe_ticker(&self, symbols: &[&str]) -> Result<()> {
        // Track subscribed symbols
        {
            let mut subscribed = self.subscribed_symbols.write().await;
            for symbol in symbols {
                if !subscribed.contains(&symbol.to_string()) {
                    subscribed.push(symbol.to_string());
                }
            }
        }

        // If connected, subscribe via WebSocket
        if let Some(ws) = self.ws_client.read().await.as_ref() {
            ws.subscribe_ticker(symbols).await?;
            debug!(
                "Bybit ticker subscription sent for {} symbols",
                symbols.len()
            );
        } else {
            debug!(
                "Bybit ticker subscription queued for {} symbols (not connected)",
                symbols.len()
            );
        }

        Ok(())
    }

    async fn unsubscribe_ticker(&self, symbols: &[&str]) -> Result<()> {
        let mut subscribed = self.subscribed_symbols.write().await;
        subscribed.retain(|s| !symbols.contains(&s.as_str()));
        // Note: Bybit doesn't support unsubscribe in v5 public WS
        // Symbols will be excluded on next reconnect
        Ok(())
    }

    fn get_ticker(&self, symbol: &str) -> Option<Ticker> {
        self.state
            .ticker_cache
            .try_read()
            .ok()
            .and_then(|cache| cache.get(symbol).cloned())
    }

    async fn subscribe_trades(&self, symbols: &[&str]) -> Result<()> {
        // Track subscribed symbols
        {
            let mut subscribed = self.trade_subscriptions.write().await;
            for symbol in symbols {
                if !subscribed.contains(&symbol.to_string()) {
                    subscribed.push(symbol.to_string());
                }
            }
        }

        // If connected, subscribe via WebSocket
        if let Some(ws) = self.ws_client.read().await.as_ref() {
            ws.subscribe_trades(symbols).await?;
            debug!(
                "Bybit trade subscription sent for {} symbols",
                symbols.len()
            );
        } else {
            debug!(
                "Bybit trade subscription queued for {} symbols (not connected)",
                symbols.len()
            );
        }

        Ok(())
    }

    async fn unsubscribe_trades(&self, symbols: &[&str]) -> Result<()> {
        let mut subscribed = self.trade_subscriptions.write().await;
        subscribed.retain(|s| !symbols.contains(&s.as_str()));
        Ok(())
    }

    async fn subscribe_candles(&self, symbols: &[&str], interval: u32) -> Result<()> {
        // Track subscribed symbols
        {
            let mut subscribed = self.candle_subscriptions.write().await;
            let entry = subscribed.entry(interval).or_default();
            for symbol in symbols {
                if !entry.contains(&symbol.to_string()) {
                    entry.push(symbol.to_string());
                }
            }
        }

        // If connected, subscribe via WebSocket
        if let Some(ws) = self.ws_client.read().await.as_ref() {
            let interval_str = interval_to_bybit_string(interval);
            ws.subscribe_candles(symbols, &interval_str).await?;
            debug!(
                "Bybit candle subscription sent for {} symbols at {}m interval",
                symbols.len(),
                interval
            );
        } else {
            debug!(
                "Bybit candle subscription queued for {} symbols (not connected)",
                symbols.len()
            );
        }

        Ok(())
    }

    async fn unsubscribe_candles(&self, symbols: &[&str], interval: u32) -> Result<()> {
        let mut subscribed = self.candle_subscriptions.write().await;
        if let Some(entry) = subscribed.get_mut(&interval) {
            entry.retain(|s| !symbols.contains(&s.as_str()));
        }
        Ok(())
    }

    fn get_candle(&self, symbol: &str, interval: u32) -> Option<Candle> {
        let cache_key = format!("{}:{}", symbol, interval);
        self.state
            .candle_cache
            .try_read()
            .ok()
            .and_then(|cache| cache.get(&cache_key).cloned())
    }

    async fn subscribe_order_book(&self, _symbols: &[&str], _depth: u32) -> Result<()> {
        Err(ExecutionError::NotImplemented(
            "Bybit order book subscription not yet implemented".to_string(),
        ))
    }

    async fn unsubscribe_order_book(&self, _symbols: &[&str]) -> Result<()> {
        Ok(())
    }

    fn get_order_book(&self, _symbol: &str) -> Option<OrderBook> {
        None
    }

    fn subscribe_events(&self) -> broadcast::Receiver<MarketDataEvent> {
        self.market_event_tx.subscribe()
    }

    fn subscribed_symbols(&self) -> Vec<String> {
        self.state
            .ticker_cache
            .try_read()
            .map(|cache| cache.keys().cloned().collect())
            .unwrap_or_default()
    }

    fn active_subscriptions(&self) -> Vec<Subscription> {
        let mut subscriptions = Vec::new();

        // Ticker subscriptions
        if let Ok(symbols) = self.subscribed_symbols.try_read() {
            if !symbols.is_empty() {
                subscriptions.push(Subscription {
                    exchange: ExchangeId::Bybit,
                    channel: SubscriptionChannel::Ticker,
                    symbols: symbols.clone(),
                });
            }
        }

        // Trade subscriptions
        if let Ok(symbols) = self.trade_subscriptions.try_read() {
            if !symbols.is_empty() {
                subscriptions.push(Subscription {
                    exchange: ExchangeId::Bybit,
                    channel: SubscriptionChannel::Trades,
                    symbols: symbols.clone(),
                });
            }
        }

        // Candle subscriptions
        if let Ok(candles) = self.candle_subscriptions.try_read() {
            for (interval, symbols) in candles.iter() {
                if !symbols.is_empty() {
                    subscriptions.push(Subscription {
                        exchange: ExchangeId::Bybit,
                        channel: SubscriptionChannel::Candles {
                            interval: *interval,
                        },
                        symbols: symbols.clone(),
                    });
                }
            }
        }

        subscriptions
    }
}

/// Convert interval in minutes to Bybit's interval string format
fn interval_to_bybit_string(interval: u32) -> String {
    match interval {
        1440 => "D".to_string(),
        10080 => "W".to_string(),
        43200 => "M".to_string(),
        _ => interval.to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = BybitProvider::new();
        assert_eq!(provider.exchange_id(), ExchangeId::Bybit);
        assert_eq!(provider.name(), "Bybit");
        assert!(!provider.is_connected());
    }

    #[test]
    fn test_provider_testnet() {
        let provider = BybitProvider::testnet();
        assert_eq!(provider.exchange_id(), ExchangeId::Bybit);
        assert_eq!(provider.ws_url(), WS_PUBLIC_SPOT_TESTNET);
    }

    #[test]
    fn test_provider_linear() {
        let provider = BybitProvider::linear(false);
        assert_eq!(provider.exchange_id(), ExchangeId::Bybit);
        assert_eq!(provider.ws_url(), WS_PUBLIC_LINEAR_MAINNET);
    }

    #[test]
    fn test_default_provider() {
        let provider = BybitProvider::default();
        assert_eq!(provider.exchange_id(), ExchangeId::Bybit);
        assert_eq!(provider.ws_url(), WS_PUBLIC_SPOT_MAINNET);
    }

    #[tokio::test]
    async fn test_health_check_disconnected() {
        let provider = BybitProvider::new();
        let health = provider.health_check().await.unwrap();

        assert!(!health.connected);
        assert_eq!(health.exchange, ExchangeId::Bybit);
        assert!(health.error.is_some());
    }

    #[tokio::test]
    async fn test_get_ticker_empty() {
        let provider = BybitProvider::new();
        assert!(provider.get_ticker("BTC/USDT").is_none());
    }

    #[tokio::test]
    async fn test_manual_ticker_update() {
        use rust_decimal::Decimal;

        let provider = BybitProvider::new();

        let ticker = Ticker {
            exchange: ExchangeId::Bybit,
            symbol: "BTC/USDT".to_string(),
            bid: Decimal::from(50000),
            bid_qty: Decimal::from(1),
            ask: Decimal::from(50010),
            ask_qty: Decimal::from(2),
            last: Decimal::from(50005),
            volume_24h: Decimal::from(1000),
            high_24h: Decimal::from(51000),
            low_24h: Decimal::from(49000),
            change_24h: Decimal::from(500),
            change_pct_24h: Decimal::from(1),
            vwap: None,
            timestamp: Utc::now(),
        };

        provider.update_ticker(ticker).await;

        let cached = provider.get_ticker("BTC/USDT");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().bid, Decimal::from(50000));
    }

    #[tokio::test]
    async fn test_subscribed_symbols_empty() {
        let provider = BybitProvider::new();
        assert!(provider.subscribed_symbols().is_empty());
    }

    #[tokio::test]
    async fn test_active_subscriptions_empty() {
        let provider = BybitProvider::new();
        assert!(provider.active_subscriptions().is_empty());
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let provider = BybitProvider::new();
        let _rx = provider.subscribe_events();
        // Just verify we can create a subscription
    }

    #[tokio::test]
    async fn test_subscribe_ticker() {
        let provider = BybitProvider::new();
        provider
            .subscribe_ticker(&["BTC/USDT", "ETH/USDT"])
            .await
            .unwrap();

        let subscribed = provider.subscribed_symbols.read().await;
        assert_eq!(subscribed.len(), 2);
        assert!(subscribed.contains(&"BTC/USDT".to_string()));
        assert!(subscribed.contains(&"ETH/USDT".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_trades() {
        let provider = BybitProvider::new();
        provider.subscribe_trades(&["BTC/USDT"]).await.unwrap();

        let subscribed = provider.trade_subscriptions.read().await;
        assert!(subscribed.contains(&"BTC/USDT".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_candles() {
        let provider = BybitProvider::new();
        provider.subscribe_candles(&["BTC/USDT"], 5).await.unwrap();

        let subscribed = provider.candle_subscriptions.read().await;
        let syms = subscribed.get(&5).unwrap();
        assert!(syms.contains(&"BTC/USDT".to_string()));
    }

    #[tokio::test]
    async fn test_active_subscriptions() {
        let provider = BybitProvider::new();
        provider.subscribe_ticker(&["BTC/USDT"]).await.unwrap();
        provider.subscribe_trades(&["ETH/USDT"]).await.unwrap();

        let subs = provider.active_subscriptions();
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_interval_to_bybit_string() {
        assert_eq!(interval_to_bybit_string(1), "1");
        assert_eq!(interval_to_bybit_string(5), "5");
        assert_eq!(interval_to_bybit_string(60), "60");
        assert_eq!(interval_to_bybit_string(1440), "D");
        assert_eq!(interval_to_bybit_string(10080), "W");
        assert_eq!(interval_to_bybit_string(43200), "M");
    }
}
