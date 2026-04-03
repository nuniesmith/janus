//! Binance Market Data Provider
//!
//! This module implements the `MarketDataProvider` trait for Binance,
//! wrapping the `BinanceWebSocket` to provide a unified interface for
//! market data streaming.
//!
//! # Example
//!
//! ```ignore
//! use janus_execution::exchanges::binance::BinanceProvider;
//! use janus_execution::exchanges::provider::{MarketDataProvider, MarketDataAggregator};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = Arc::new(BinanceProvider::new());
//!
//!     // Connect to Binance
//!     provider.connect().await?;
//!
//!     // Subscribe to tickers
//!     provider.subscribe_ticker(&["BTC/USDT", "ETH/USDT"]).await?;
//!
//!     // Add to aggregator
//!     let mut aggregator = MarketDataAggregator::new();
//!     aggregator.add_provider(provider);
//!
//!     Ok(())
//! }
//! ```

use crate::error::{ExecutionError, Result};
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
use tracing::{debug, error, info, warn};

use super::websocket::{BinanceEvent, BinanceWebSocket, BinanceWsConfig};

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

/// Binance market data provider implementing the `MarketDataProvider` trait
pub struct BinanceProvider {
    /// Underlying WebSocket client
    ws: Arc<BinanceWebSocket>,

    /// Shared state accessible from spawned tasks
    state: Arc<SharedState>,

    /// Active subscriptions
    subscribed_symbols: RwLock<Vec<String>>,

    /// Event forwarder task handle
    forwarder_handle: RwLock<Option<tokio::task::JoinHandle<()>>>,

    /// Market data event sender for broadcasting to aggregator
    market_event_tx: broadcast::Sender<MarketDataEvent>,
}

impl BinanceProvider {
    /// Create a new Binance provider with default configuration
    pub fn new() -> Self {
        Self::with_config(BinanceWsConfig::default())
    }

    /// Create a new Binance provider with custom configuration
    pub fn with_config(config: BinanceWsConfig) -> Self {
        let ws = Arc::new(BinanceWebSocket::new(config));
        let (market_event_tx, _) = broadcast::channel(10000);

        Self {
            ws,
            state: Arc::new(SharedState::new()),
            subscribed_symbols: RwLock::new(Vec::new()),
            forwarder_handle: RwLock::new(None),
            market_event_tx,
        }
    }

    /// Create a provider configured for testnet
    pub fn testnet() -> Self {
        Self::with_config(BinanceWsConfig {
            testnet: true,
            ..Default::default()
        })
    }

    /// Get the underlying WebSocket client
    pub fn websocket(&self) -> &Arc<BinanceWebSocket> {
        &self.ws
    }

    /// Get the market data event sender for external use
    pub fn event_sender(&self) -> broadcast::Sender<MarketDataEvent> {
        self.market_event_tx.clone()
    }

    /// Start the event forwarder that converts BinanceEvent to MarketDataEvent
    async fn start_event_forwarder(&self) {
        let mut rx = self.ws.subscribe_events();
        let market_tx = self.market_event_tx.clone();
        let state = self.state.clone();

        let handle = tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        // Update last message time
                        *state.last_message.write().await = Some(Utc::now());

                        // Convert and forward the event
                        match &event {
                            BinanceEvent::Ticker(ticker) => {
                                let market_ticker = ticker.to_ticker();
                                state
                                    .ticker_cache
                                    .write()
                                    .await
                                    .insert(market_ticker.symbol.clone(), market_ticker.clone());
                                let _ = market_tx.send(MarketDataEvent::Ticker(market_ticker));
                            }
                            BinanceEvent::Trade(trade) => {
                                let market_trade = trade.to_trade();
                                let _ = market_tx.send(MarketDataEvent::Trade(market_trade));
                            }
                            BinanceEvent::Candle(candle) => {
                                let market_candle = candle.to_candle();
                                let cache_key =
                                    format!("{}:{}", market_candle.symbol, market_candle.interval);
                                state
                                    .candle_cache
                                    .write()
                                    .await
                                    .insert(cache_key, market_candle.clone());
                                let _ = market_tx.send(MarketDataEvent::Candle(market_candle));
                            }
                            BinanceEvent::Connected => {
                                state.connected.store(true, Ordering::SeqCst);
                                info!("Binance WebSocket connected");
                                let _ = market_tx.send(MarketDataEvent::ConnectionStatus {
                                    exchange: ExchangeId::Binance,
                                    connected: true,
                                    message: Some("Connected".to_string()),
                                    timestamp: Utc::now(),
                                });
                            }
                            BinanceEvent::Disconnected => {
                                state.connected.store(false, Ordering::SeqCst);
                                state.reconnect_count.fetch_add(1, Ordering::SeqCst);
                                warn!("Binance WebSocket disconnected");
                                let _ = market_tx.send(MarketDataEvent::ConnectionStatus {
                                    exchange: ExchangeId::Binance,
                                    connected: false,
                                    message: Some("Disconnected".to_string()),
                                    timestamp: Utc::now(),
                                });
                            }
                            BinanceEvent::Error(msg) => {
                                error!("Binance WebSocket error: {}", msg);
                                let _ = market_tx.send(MarketDataEvent::Error {
                                    exchange: ExchangeId::Binance,
                                    code: None,
                                    message: msg.clone(),
                                    timestamp: Utc::now(),
                                });
                            }
                            BinanceEvent::Subscribed { streams } => {
                                debug!("Binance subscribed to {} streams", streams.len());
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        info!("Binance event channel closed");
                        break;
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("Binance event forwarder lagged by {} messages", n);
                    }
                }
            }
        });

        *self.forwarder_handle.write().await = Some(handle);
    }

    /// Stop the event forwarder
    async fn stop_event_forwarder(&self) {
        if let Some(handle) = self.forwarder_handle.write().await.take() {
            handle.abort();
        }
    }
}

impl Default for BinanceProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MarketDataProvider for BinanceProvider {
    fn exchange_id(&self) -> ExchangeId {
        ExchangeId::Binance
    }

    fn name(&self) -> &str {
        "Binance"
    }

    async fn connect(&self) -> Result<()> {
        info!("Connecting Binance provider...");

        // Apply any pending subscriptions BEFORE starting the WebSocket
        // This is required because Binance combined streams are set via URL params
        let subscribed = self.subscribed_symbols.read().await.clone();
        if !subscribed.is_empty() {
            let symbols_refs: Vec<&str> = subscribed.iter().map(|s| s.as_str()).collect();
            // Add to WebSocket's subscription list (will be included in URL)
            self.ws.subscribe_ticker(&symbols_refs).await?;
            debug!(
                "Binance: Pre-registered {} ticker subscriptions",
                subscribed.len()
            );
        }

        // Start the WebSocket (connects with subscriptions in URL)
        self.ws.start().await?;

        // Start the event forwarder
        self.start_event_forwarder().await;

        // Wait briefly for connection to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        self.state.connected.store(true, Ordering::SeqCst);
        info!("Binance provider connected");

        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        info!("Disconnecting Binance provider...");

        // Stop the event forwarder
        self.stop_event_forwarder().await;

        // Stop the WebSocket
        self.ws.stop().await;

        self.state.connected.store(false, Ordering::SeqCst);
        info!("Binance provider disconnected");

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
                exchange: ExchangeId::Binance,
                connected: true,
                last_message,
                reconnect_count,
                latency_ms,
                error: None,
            })
        } else {
            Ok(ConnectionHealth {
                exchange: ExchangeId::Binance,
                connected: false,
                last_message,
                reconnect_count,
                latency_ms: None,
                error: Some("Not connected".to_string()),
            })
        }
    }

    async fn subscribe_ticker(&self, symbols: &[&str]) -> Result<()> {
        // Track subscribed symbols (used when connect() is called)
        let mut subscribed = self.subscribed_symbols.write().await;
        for symbol in symbols {
            if !subscribed.contains(&symbol.to_string()) {
                subscribed.push(symbol.to_string());
            }
        }
        drop(subscribed);

        // If already connected, we need to reconnect to apply new subscriptions
        // (Binance combined streams require reconnection for new subscriptions)
        if self.is_connected() {
            debug!("Binance: Adding subscriptions requires reconnect for combined streams");
            // Add to WebSocket subscription list for next connection
            self.ws.subscribe_ticker(symbols).await?;
            // Note: For dynamic subscriptions, would need to implement SUBSCRIBE message
            // or reconnect. For now, subscriptions should be set before connect().
        } else {
            // Not connected yet - just track for later
            debug!(
                "Binance: Queued {} ticker subscriptions for connect",
                symbols.len()
            );
        }

        Ok(())
    }

    async fn unsubscribe_ticker(&self, symbols: &[&str]) -> Result<()> {
        // Remove from subscribed list
        let mut subscribed = self.subscribed_symbols.write().await;
        subscribed.retain(|s| !symbols.contains(&s.as_str()));

        // Note: Binance WebSocket doesn't have explicit unsubscribe in current implementation
        // Would need to add that to BinanceWebSocket

        Ok(())
    }

    fn get_ticker(&self, symbol: &str) -> Option<Ticker> {
        // Use try_read to avoid blocking - return None if lock is held
        self.state
            .ticker_cache
            .try_read()
            .ok()
            .and_then(|cache| cache.get(symbol).cloned())
    }

    async fn subscribe_trades(&self, symbols: &[&str]) -> Result<()> {
        self.ws.subscribe_trades(symbols).await
    }

    async fn unsubscribe_trades(&self, _symbols: &[&str]) -> Result<()> {
        // Note: Would need to implement unsubscribe in BinanceWebSocket
        Ok(())
    }

    async fn subscribe_candles(&self, symbols: &[&str], interval: u32) -> Result<()> {
        self.ws.subscribe_candles(symbols, interval).await
    }

    async fn unsubscribe_candles(&self, _symbols: &[&str], _interval: u32) -> Result<()> {
        // Note: Would need to implement unsubscribe in BinanceWebSocket
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
        // Order book subscription not yet implemented for Binance
        Err(ExecutionError::NotImplemented(
            "Binance order book subscription not yet implemented".to_string(),
        ))
    }

    async fn unsubscribe_order_book(&self, _symbols: &[&str]) -> Result<()> {
        Ok(())
    }

    fn get_order_book(&self, _symbol: &str) -> Option<OrderBook> {
        // Order book not yet implemented
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
        let symbols = self.subscribed_symbols();
        if symbols.is_empty() {
            return Vec::new();
        }

        vec![Subscription {
            exchange: ExchangeId::Binance,
            channel: SubscriptionChannel::Ticker,
            symbols,
        }]
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
        let provider = BinanceProvider::new();
        assert_eq!(provider.exchange_id(), ExchangeId::Binance);
        assert_eq!(provider.name(), "Binance");
        assert!(!provider.is_connected());
    }

    #[test]
    fn test_provider_testnet() {
        let provider = BinanceProvider::testnet();
        assert_eq!(provider.exchange_id(), ExchangeId::Binance);
    }

    #[test]
    fn test_default_provider() {
        let provider = BinanceProvider::default();
        assert_eq!(provider.exchange_id(), ExchangeId::Binance);
    }

    #[tokio::test]
    async fn test_health_check_disconnected() {
        let provider = BinanceProvider::new();
        let health = provider.health_check().await.unwrap();

        assert!(!health.connected);
        assert_eq!(health.exchange, ExchangeId::Binance);
        assert!(health.error.is_some());
    }

    #[tokio::test]
    async fn test_get_ticker_empty() {
        let provider = BinanceProvider::new();
        assert!(provider.get_ticker("BTC/USDT").is_none());
    }

    #[tokio::test]
    async fn test_subscribed_symbols_empty() {
        let provider = BinanceProvider::new();
        assert!(provider.subscribed_symbols().is_empty());
    }

    #[tokio::test]
    async fn test_active_subscriptions() {
        let provider = BinanceProvider::new();
        assert!(provider.active_subscriptions().is_empty());
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let provider = BinanceProvider::new();
        let _rx = provider.subscribe_events();
        // Just verify we can create a subscription
    }
}
