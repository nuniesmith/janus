//! Live Feed Bridge - Connects Exchange Providers to Simulation Data Feed
//!
//! This module bridges the gap between live exchange WebSocket providers
//! and the unified simulation data feed abstraction. It converts
//! `MarketDataEvent` from exchange providers to `MarketEvent` for the
//! simulation environment.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                       LiveFeedBridge                                     │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
//! │  │   Kraken    │  │   Binance   │  │   Bybit     │                      │
//! │  │  Provider   │  │  Provider   │  │  Provider   │                      │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                      │
//! │         │                │                │                              │
//! │         └────────────────┼────────────────┘                              │
//! │                          ▼                                               │
//! │              ┌───────────────────────┐                                   │
//! │              │  MarketDataAggregator │                                   │
//! │              │   (MarketDataEvent)   │                                   │
//! │              └───────────┬───────────┘                                   │
//! │                          │                                               │
//! │                          ▼                                               │
//! │              ┌───────────────────────┐                                   │
//! │              │    LiveFeedBridge     │                                   │
//! │              │  (Event Conversion)   │                                   │
//! │              └───────────┬───────────┘                                   │
//! │                          │                                               │
//! │                          ▼                                               │
//! │              ┌───────────────────────┐                                   │
//! │              │  AggregatedDataFeed   │                                   │
//! │              │    (MarketEvent)      │                                   │
//! │              └───────────────────────┘                                   │
//! │                          │                                               │
//! │         ┌────────────────┼────────────────┐                              │
//! │         ▼                ▼                ▼                              │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
//! │  │ SimEnviron- │  │   Data      │  │  Strategy   │                      │
//! │  │    ment     │  │  Recorder   │  │  (signals)  │                      │
//! │  └─────────────┘  └─────────────┘  └─────────────┘                      │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_execution::sim::{LiveFeedBridge, AggregatedDataFeed};
//! use janus_execution::exchanges::MarketDataAggregator;
//!
//! // Create the simulation data feed
//! let data_feed = Arc::new(AggregatedDataFeed::new("live"));
//!
//! // Create the bridge
//! let bridge = LiveFeedBridge::new(data_feed.clone());
//!
//! // Connect to the market data aggregator
//! bridge.connect_to_aggregator(&aggregator).await?;
//!
//! // Or connect to individual providers
//! bridge.connect_to_provider(kraken_provider).await?;
//!
//! // Now data_feed will receive MarketEvent from live exchanges
//! let mut rx = data_feed.subscribe();
//! while let Ok(event) = rx.recv().await {
//!     println!("Received: {:?}", event);
//! }
//! ```

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::exchanges::MarketDataAggregator;
use crate::exchanges::market_data::{
    MarketDataEvent, Ticker, Trade as ExchangeTrade, TradeSide as ExchangeTradeSide,
};
use crate::exchanges::provider::MarketDataProvider;

use super::data_feed::{
    AggregatedDataFeed, CandleData, MarketEvent, OrderBookData, OrderBookLevel, TickData,
    TradeData, TradeSide,
};

/// Errors that can occur in the live feed bridge
#[derive(Debug, Error)]
pub enum LiveFeedBridgeError {
    #[error("Bridge not started")]
    NotStarted,

    #[error("Bridge already running")]
    AlreadyRunning,

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Channel closed")]
    ChannelClosed,
}

/// Statistics for the live feed bridge
#[derive(Debug, Clone, Default)]
pub struct LiveFeedBridgeStats {
    /// Total events received from providers
    pub events_received: u64,
    /// Events successfully converted and published
    pub events_published: u64,
    /// Events dropped (conversion failed or channel full)
    pub events_dropped: u64,
    /// Events by source exchange
    pub events_by_exchange: HashMap<String, u64>,
    /// Ticks converted
    pub ticks_converted: u64,
    /// Trades converted
    pub trades_converted: u64,
    /// Order books converted
    pub orderbooks_converted: u64,
    /// Candles converted
    pub candles_converted: u64,
    /// Connection events
    pub connection_events: u64,
    /// Start time
    pub start_time: Option<DateTime<Utc>>,
    /// Last event time
    pub last_event_time: Option<DateTime<Utc>>,
}

impl LiveFeedBridgeStats {
    /// Get events per second
    pub fn events_per_second(&self) -> f64 {
        match (self.start_time, self.last_event_time) {
            (Some(start), Some(last)) => {
                let duration = (last - start).num_milliseconds() as f64 / 1000.0;
                if duration > 0.0 {
                    self.events_published as f64 / duration
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// Get conversion success rate
    pub fn success_rate(&self) -> f64 {
        if self.events_received > 0 {
            self.events_published as f64 / self.events_received as f64 * 100.0
        } else {
            100.0
        }
    }
}

/// Configuration for the live feed bridge
#[derive(Debug, Clone)]
pub struct LiveFeedBridgeConfig {
    /// Whether to normalize symbols to "BASE/QUOTE" format
    pub normalize_symbols: bool,
    /// Whether to filter out invalid ticks (zero prices, inverted spread)
    pub filter_invalid_ticks: bool,
    /// Minimum tick interval (debounce) in milliseconds (0 = no debounce)
    pub tick_debounce_ms: u64,
    /// Exchanges to include (empty = all)
    pub include_exchanges: Vec<String>,
    /// Symbols to include (empty = all)
    pub include_symbols: Vec<String>,
    /// Whether to emit connection status events
    pub emit_connection_events: bool,
}

impl Default for LiveFeedBridgeConfig {
    fn default() -> Self {
        Self {
            normalize_symbols: true,
            filter_invalid_ticks: true,
            tick_debounce_ms: 0,
            include_exchanges: Vec::new(),
            include_symbols: Vec::new(),
            emit_connection_events: true,
        }
    }
}

impl LiveFeedBridgeConfig {
    /// Create a config that passes through all events
    pub fn passthrough() -> Self {
        Self {
            normalize_symbols: false,
            filter_invalid_ticks: false,
            tick_debounce_ms: 0,
            include_exchanges: Vec::new(),
            include_symbols: Vec::new(),
            emit_connection_events: true,
        }
    }

    /// Builder: set symbol normalization
    pub fn with_normalize_symbols(mut self, normalize: bool) -> Self {
        self.normalize_symbols = normalize;
        self
    }

    /// Builder: set invalid tick filtering
    pub fn with_filter_invalid_ticks(mut self, filter: bool) -> Self {
        self.filter_invalid_ticks = filter;
        self
    }

    /// Builder: set tick debounce interval
    pub fn with_tick_debounce_ms(mut self, ms: u64) -> Self {
        self.tick_debounce_ms = ms;
        self
    }

    /// Builder: filter to specific exchanges
    pub fn with_exchanges(mut self, exchanges: Vec<String>) -> Self {
        self.include_exchanges = exchanges;
        self
    }

    /// Builder: filter to specific symbols
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.include_symbols = symbols;
        self
    }

    /// Check if an exchange should be included
    fn should_include_exchange(&self, exchange: &str) -> bool {
        self.include_exchanges.is_empty()
            || self
                .include_exchanges
                .iter()
                .any(|e| e.eq_ignore_ascii_case(exchange))
    }

    /// Check if a symbol should be included
    fn should_include_symbol(&self, symbol: &str) -> bool {
        self.include_symbols.is_empty()
            || self
                .include_symbols
                .iter()
                .any(|s| s.eq_ignore_ascii_case(symbol))
    }
}

/// Bridges live exchange data to the simulation data feed
pub struct LiveFeedBridge {
    /// Target data feed to publish events to
    data_feed: Arc<AggregatedDataFeed>,
    /// Configuration
    config: LiveFeedBridgeConfig,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<parking_lot::RwLock<LiveFeedBridgeStats>>,
    /// Event counters (atomic for fast updates)
    events_received: Arc<AtomicU64>,
    events_published: Arc<AtomicU64>,
    events_dropped: Arc<AtomicU64>,
    /// Task handles for cleanup
    task_handles: parking_lot::Mutex<Vec<JoinHandle<()>>>,
    /// Last tick times for debouncing (symbol -> timestamp)
    last_tick_times: Arc<parking_lot::RwLock<HashMap<String, DateTime<Utc>>>>,
}

impl LiveFeedBridge {
    /// Create a new live feed bridge
    pub fn new(data_feed: Arc<AggregatedDataFeed>) -> Self {
        Self::with_config(data_feed, LiveFeedBridgeConfig::default())
    }

    /// Create a new live feed bridge with custom configuration
    pub fn with_config(data_feed: Arc<AggregatedDataFeed>, config: LiveFeedBridgeConfig) -> Self {
        Self {
            data_feed,
            config,
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(parking_lot::RwLock::new(LiveFeedBridgeStats::default())),
            events_received: Arc::new(AtomicU64::new(0)),
            events_published: Arc::new(AtomicU64::new(0)),
            events_dropped: Arc::new(AtomicU64::new(0)),
            task_handles: parking_lot::Mutex::new(Vec::new()),
            last_tick_times: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }

    /// Connect to a MarketDataAggregator and start forwarding events
    pub async fn connect_to_aggregator(
        &self,
        aggregator: &MarketDataAggregator,
    ) -> Result<(), LiveFeedBridgeError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(LiveFeedBridgeError::AlreadyRunning);
        }

        self.running.store(true, Ordering::SeqCst);

        // Initialize stats
        {
            let mut stats = self.stats.write();
            stats.start_time = Some(Utc::now());
        }

        // Subscribe to aggregator events
        let mut rx = aggregator.subscribe_events();

        let data_feed = self.data_feed.clone();
        let config = self.config.clone();
        let running = self.running.clone();
        let stats = self.stats.clone();
        let events_received = self.events_received.clone();
        let events_published = self.events_published.clone();
        let events_dropped = self.events_dropped.clone();
        let last_tick_times = self.last_tick_times.clone();

        let handle = tokio::spawn(async move {
            info!("LiveFeedBridge: Started forwarding from MarketDataAggregator");

            while running.load(Ordering::SeqCst) {
                match rx.recv().await {
                    Ok(event) => {
                        events_received.fetch_add(1, Ordering::Relaxed);

                        if let Some(market_event) =
                            Self::convert_event(&event, &config, &stats, &last_tick_times)
                        {
                            data_feed.publish(market_event);
                            events_published.fetch_add(1, Ordering::Relaxed);

                            let mut s = stats.write();
                            s.last_event_time = Some(Utc::now());
                        } else {
                            events_dropped.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("LiveFeedBridge: Lagged by {} messages", n);
                        events_dropped.fetch_add(n, Ordering::Relaxed);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        info!("LiveFeedBridge: Aggregator channel closed");
                        break;
                    }
                }
            }

            info!("LiveFeedBridge: Stopped forwarding");
        });

        self.task_handles.lock().push(handle);
        Ok(())
    }

    /// Connect to a single MarketDataProvider and start forwarding events
    pub async fn connect_to_provider(
        &self,
        provider: Arc<dyn MarketDataProvider>,
    ) -> Result<(), LiveFeedBridgeError> {
        let exchange_name = provider.name().to_string();

        // Check if this exchange should be included
        if !self.config.should_include_exchange(&exchange_name) {
            debug!(
                "LiveFeedBridge: Skipping provider {} (not in include list)",
                exchange_name
            );
            return Ok(());
        }

        if !self.running.load(Ordering::SeqCst) {
            self.running.store(true, Ordering::SeqCst);
            let mut stats = self.stats.write();
            stats.start_time = Some(Utc::now());
        }

        // Subscribe to provider events
        let mut rx = provider.subscribe_events();

        let data_feed = self.data_feed.clone();
        let config = self.config.clone();
        let running = self.running.clone();
        let stats = self.stats.clone();
        let events_received = self.events_received.clone();
        let events_published = self.events_published.clone();
        let events_dropped = self.events_dropped.clone();
        let last_tick_times = self.last_tick_times.clone();

        let handle = tokio::spawn(async move {
            info!(
                "LiveFeedBridge: Started forwarding from provider {}",
                exchange_name
            );

            while running.load(Ordering::SeqCst) {
                match rx.recv().await {
                    Ok(event) => {
                        events_received.fetch_add(1, Ordering::Relaxed);

                        if let Some(market_event) =
                            Self::convert_event(&event, &config, &stats, &last_tick_times)
                        {
                            data_feed.publish(market_event);
                            events_published.fetch_add(1, Ordering::Relaxed);

                            let mut s = stats.write();
                            s.last_event_time = Some(Utc::now());
                        } else {
                            events_dropped.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(
                            "LiveFeedBridge: Lagged by {} messages from {}",
                            n, exchange_name
                        );
                        events_dropped.fetch_add(n, Ordering::Relaxed);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        info!("LiveFeedBridge: Provider {} channel closed", exchange_name);
                        break;
                    }
                }
            }

            info!("LiveFeedBridge: Stopped forwarding from {}", exchange_name);
        });

        self.task_handles.lock().push(handle);
        Ok(())
    }

    /// Stop the bridge and all forwarding tasks
    pub async fn stop(&self) {
        info!("LiveFeedBridge: Stopping...");
        self.running.store(false, Ordering::SeqCst);

        // Wait for all tasks to complete
        let handles: Vec<_> = {
            let mut guard = self.task_handles.lock();
            std::mem::take(&mut *guard)
        };

        for handle in handles {
            handle.abort();
        }

        // Update final stats
        {
            let mut stats = self.stats.write();
            stats.events_received = self.events_received.load(Ordering::Relaxed);
            stats.events_published = self.events_published.load(Ordering::Relaxed);
            stats.events_dropped = self.events_dropped.load(Ordering::Relaxed);
        }

        info!("LiveFeedBridge: Stopped");
    }

    /// Get current statistics
    pub fn stats(&self) -> LiveFeedBridgeStats {
        let mut stats = self.stats.read().clone();
        stats.events_received = self.events_received.load(Ordering::Relaxed);
        stats.events_published = self.events_published.load(Ordering::Relaxed);
        stats.events_dropped = self.events_dropped.load(Ordering::Relaxed);
        stats
    }

    /// Check if bridge is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Convert a MarketDataEvent to a MarketEvent
    fn convert_event(
        event: &MarketDataEvent,
        config: &LiveFeedBridgeConfig,
        stats: &Arc<parking_lot::RwLock<LiveFeedBridgeStats>>,
        last_tick_times: &Arc<parking_lot::RwLock<HashMap<String, DateTime<Utc>>>>,
    ) -> Option<MarketEvent> {
        match event {
            MarketDataEvent::Ticker(ticker) => {
                Self::convert_ticker(ticker, config, stats, last_tick_times)
            }
            MarketDataEvent::Trade(trade) => Self::convert_trade(trade, config, stats),
            MarketDataEvent::OrderBookSnapshot(orderbook) => {
                Self::convert_orderbook(orderbook, config, stats)
            }
            MarketDataEvent::OrderBookUpdate { .. } => {
                // Skip delta updates for now - would need more complex handling
                None
            }
            MarketDataEvent::Candle(candle) => Self::convert_candle(candle, config, stats),
            MarketDataEvent::ConnectionStatus {
                exchange,
                connected,
                message,
                ..
            } => {
                if config.emit_connection_events {
                    stats.write().connection_events += 1;
                    Some(MarketEvent::ConnectionStatus {
                        exchange: exchange.name().to_string(),
                        connected: *connected,
                        message: message.clone(),
                    })
                } else {
                    None
                }
            }
            MarketDataEvent::Error {
                exchange, message, ..
            } => {
                warn!("LiveFeedBridge: Error from {:?}: {}", exchange, message);
                None
            }
        }
    }

    /// Convert a Ticker to TickData
    fn convert_ticker(
        ticker: &Ticker,
        config: &LiveFeedBridgeConfig,
        stats: &Arc<parking_lot::RwLock<LiveFeedBridgeStats>>,
        last_tick_times: &Arc<parking_lot::RwLock<HashMap<String, DateTime<Utc>>>>,
    ) -> Option<MarketEvent> {
        let exchange = ticker.exchange.name();
        let symbol = if config.normalize_symbols {
            normalize_symbol(&ticker.symbol)
        } else {
            ticker.symbol.clone()
        };

        // Check filters
        if !config.should_include_exchange(exchange) {
            return None;
        }
        if !config.should_include_symbol(&symbol) {
            return None;
        }

        // Validate tick data
        if config.filter_invalid_ticks {
            if ticker.bid <= Decimal::ZERO || ticker.ask <= Decimal::ZERO || ticker.ask < ticker.bid
            {
                debug!(
                    "LiveFeedBridge: Filtering invalid tick for {}/{}: bid={}, ask={}",
                    symbol, exchange, ticker.bid, ticker.ask
                );
                return None;
            }
        }

        // Check debounce
        if config.tick_debounce_ms > 0 {
            let key = format!("{}:{}", symbol, exchange);
            let now = Utc::now();
            let mut last_times = last_tick_times.write();

            if let Some(last_time) = last_times.get(&key) {
                let elapsed = (now - *last_time).num_milliseconds() as u64;
                if elapsed < config.tick_debounce_ms {
                    return None;
                }
            }
            last_times.insert(key, now);
        }

        // Update stats
        {
            let mut s = stats.write();
            s.ticks_converted += 1;
            *s.events_by_exchange
                .entry(exchange.to_string())
                .or_insert(0) += 1;
        }

        Some(MarketEvent::Tick(TickData {
            symbol,
            exchange: exchange.to_string(),
            bid_price: ticker.bid,
            ask_price: ticker.ask,
            bid_size: ticker.bid_qty,
            ask_size: ticker.ask_qty,
            timestamp: ticker.timestamp,
            sequence: 0,
        }))
    }

    /// Convert an ExchangeTrade to TradeData
    fn convert_trade(
        trade: &ExchangeTrade,
        config: &LiveFeedBridgeConfig,
        stats: &Arc<parking_lot::RwLock<LiveFeedBridgeStats>>,
    ) -> Option<MarketEvent> {
        let exchange = trade.exchange.name();
        let symbol = if config.normalize_symbols {
            normalize_symbol(&trade.symbol)
        } else {
            trade.symbol.clone()
        };

        // Check filters
        if !config.should_include_exchange(exchange) {
            return None;
        }
        if !config.should_include_symbol(&symbol) {
            return None;
        }

        // Update stats
        {
            let mut s = stats.write();
            s.trades_converted += 1;
            *s.events_by_exchange
                .entry(exchange.to_string())
                .or_insert(0) += 1;
        }

        let side = match trade.side {
            ExchangeTradeSide::Buy => TradeSide::Buy,
            ExchangeTradeSide::Sell => TradeSide::Sell,
        };

        Some(MarketEvent::Trade(TradeData {
            symbol,
            exchange: exchange.to_string(),
            price: trade.price,
            size: trade.quantity,
            side,
            trade_id: trade.trade_id.clone(),
            timestamp: trade.timestamp,
        }))
    }

    /// Convert an OrderBook to OrderBookData
    fn convert_orderbook(
        orderbook: &crate::exchanges::market_data::OrderBook,
        config: &LiveFeedBridgeConfig,
        stats: &Arc<parking_lot::RwLock<LiveFeedBridgeStats>>,
    ) -> Option<MarketEvent> {
        let exchange = orderbook.exchange.name();
        let symbol = if config.normalize_symbols {
            normalize_symbol(&orderbook.symbol)
        } else {
            orderbook.symbol.clone()
        };

        // Check filters
        if !config.should_include_exchange(exchange) {
            return None;
        }
        if !config.should_include_symbol(&symbol) {
            return None;
        }

        // Update stats
        {
            let mut s = stats.write();
            s.orderbooks_converted += 1;
            *s.events_by_exchange
                .entry(exchange.to_string())
                .or_insert(0) += 1;
        }

        let bids: Vec<OrderBookLevel> = orderbook
            .bids
            .iter()
            .map(|level| OrderBookLevel {
                price: level.price,
                size: level.quantity,
            })
            .collect();

        let asks: Vec<OrderBookLevel> = orderbook
            .asks
            .iter()
            .map(|level| OrderBookLevel {
                price: level.price,
                size: level.quantity,
            })
            .collect();

        Some(MarketEvent::OrderBook(OrderBookData {
            symbol,
            exchange: exchange.to_string(),
            bids,
            asks,
            timestamp: orderbook.timestamp,
        }))
    }

    /// Convert a Candle to CandleData
    fn convert_candle(
        candle: &crate::exchanges::market_data::Candle,
        config: &LiveFeedBridgeConfig,
        stats: &Arc<parking_lot::RwLock<LiveFeedBridgeStats>>,
    ) -> Option<MarketEvent> {
        let exchange = candle.exchange.name();
        let symbol = if config.normalize_symbols {
            normalize_symbol(&candle.symbol)
        } else {
            candle.symbol.clone()
        };

        // Check filters
        if !config.should_include_exchange(exchange) {
            return None;
        }
        if !config.should_include_symbol(&symbol) {
            return None;
        }

        // Update stats
        {
            let mut s = stats.write();
            s.candles_converted += 1;
            *s.events_by_exchange
                .entry(exchange.to_string())
                .or_insert(0) += 1;
        }

        Some(MarketEvent::Candle(CandleData {
            symbol,
            exchange: exchange.to_string(),
            open_time: candle.open_time,
            close_time: candle.close_time,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.volume,
            trade_count: candle.trades.unwrap_or(0),
        }))
    }
}

impl Drop for LiveFeedBridge {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);

        // Abort all tasks
        let handles: Vec<_> = {
            let mut guard = self.task_handles.lock();
            std::mem::take(&mut *guard)
        };

        for handle in handles {
            handle.abort();
        }
    }
}

// ============================================================================
// Symbol Normalization
// ============================================================================

/// Normalize symbol to "BASE/QUOTE" format
///
/// Handles various exchange-specific formats:
/// - BTCUSD -> BTC/USDT
/// - btcusdt -> BTC/USDT
/// - BTC/USD -> BTC/USD
/// - BTC-USDT -> BTC/USDT
/// - XBTUSD -> XBT/USD
fn normalize_symbol(symbol: &str) -> String {
    let s = symbol.to_uppercase();

    // Already normalized
    if s.contains('/') {
        return s;
    }

    // Handle dash separator
    if s.contains('-') {
        return s.replace('-', "/");
    }

    // Try to split common quote currencies
    let quote_currencies = ["USDT", "USDC", "USD", "EUR", "GBP", "BTC", "ETH", "BUSD"];

    for quote in quote_currencies {
        if s.ends_with(quote) && s.len() > quote.len() {
            let base = &s[..s.len() - quote.len()];
            return format!("{}/{}", base, quote);
        }
    }

    // Fallback - return as-is
    s
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_normalize_symbol() {
        assert_eq!(normalize_symbol("BTCUSDT"), "BTC/USDT");
        assert_eq!(normalize_symbol("btcusdt"), "BTC/USDT");
        assert_eq!(normalize_symbol("BTC/USDT"), "BTC/USDT");
        assert_eq!(normalize_symbol("BTC-USDT"), "BTC/USDT");
        assert_eq!(normalize_symbol("ETHBTC"), "ETH/BTC");
        assert_eq!(normalize_symbol("SOLUSD"), "SOL/USD");
        assert_eq!(normalize_symbol("XBTUSD"), "XBT/USD");
    }

    #[test]
    fn test_config_filters() {
        let config = LiveFeedBridgeConfig::default()
            .with_exchanges(vec!["kraken".to_string(), "binance".to_string()])
            .with_symbols(vec!["BTC/USDT".to_string()]);

        assert!(config.should_include_exchange("kraken"));
        assert!(config.should_include_exchange("Kraken"));
        assert!(config.should_include_exchange("binance"));
        assert!(!config.should_include_exchange("bybit"));

        assert!(config.should_include_symbol("BTC/USDT"));
        assert!(config.should_include_symbol("btc/usdt"));
        assert!(!config.should_include_symbol("ETH/USDT"));
    }

    #[test]
    fn test_config_empty_filters() {
        let config = LiveFeedBridgeConfig::default();

        // Empty filters should include everything
        assert!(config.should_include_exchange("kraken"));
        assert!(config.should_include_exchange("binance"));
        assert!(config.should_include_exchange("bybit"));
        assert!(config.should_include_symbol("BTC/USDT"));
        assert!(config.should_include_symbol("ETH/USDT"));
    }

    #[test]
    fn test_stats() {
        let mut stats = LiveFeedBridgeStats::default();
        stats.events_received = 1000;
        stats.events_published = 950;
        stats.events_dropped = 50;

        assert!((stats.success_rate() - 95.0).abs() < 0.1);
    }

    #[test]
    fn test_tick_validation() {
        let config = LiveFeedBridgeConfig::default().with_filter_invalid_ticks(true);

        // Valid tick
        let valid_ticker = Ticker {
            symbol: "BTCUSD".to_string(),
            exchange: crate::exchanges::market_data::ExchangeId::Binance,
            bid: dec!(50000),
            ask: dec!(50001),
            bid_qty: dec!(1.0),
            ask_qty: dec!(1.0),
            last: dec!(50000.5),
            volume_24h: dec!(1000000),
            high_24h: dec!(51000),
            low_24h: dec!(49000),
            change_24h: dec!(100),
            change_pct_24h: dec!(0.2),
            vwap: None,
            timestamp: Utc::now(),
        };

        let stats = Arc::new(parking_lot::RwLock::new(LiveFeedBridgeStats::default()));
        let last_tick_times = Arc::new(parking_lot::RwLock::new(HashMap::new()));

        let result =
            LiveFeedBridge::convert_ticker(&valid_ticker, &config, &stats, &last_tick_times);
        assert!(result.is_some());

        // Invalid tick (zero bid)
        let mut invalid_ticker = valid_ticker.clone();
        invalid_ticker.bid = dec!(0);

        let result =
            LiveFeedBridge::convert_ticker(&invalid_ticker, &config, &stats, &last_tick_times);
        assert!(result.is_none());

        // Invalid tick (inverted spread)
        let mut inverted_ticker = valid_ticker.clone();
        inverted_ticker.bid = dec!(50001);
        inverted_ticker.ask = dec!(50000);

        let result =
            LiveFeedBridge::convert_ticker(&inverted_ticker, &config, &stats, &last_tick_times);
        assert!(result.is_none());
    }
}
