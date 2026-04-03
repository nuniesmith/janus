//! Unified Data Feed Abstraction
//!
//! Provides a common interface for consuming market data from various sources:
//! - Historical files (Parquet, CSV)
//! - Live WebSocket feeds (Kraken, Binance, Bybit)
//! - Recorded data from QuestDB
//!
//! This abstraction allows strategies to be tested against historical data,
//! validated with live data (paper trading), and deployed to production
//! without code changes.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::broadcast;

/// Errors that can occur in data feeds
#[derive(Debug, Error)]
pub enum DataFeedError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Data source not found: {0}")]
    SourceNotFound(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Channel closed")]
    ChannelClosed,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Query error: {0}")]
    QueryError(String),

    #[error("End of data")]
    EndOfData,
}

/// Tick data representing best bid/ask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    /// Trading symbol (normalized, e.g., "BTC/USDT")
    pub symbol: String,
    /// Source exchange
    pub exchange: String,
    /// Best bid price
    pub bid_price: Decimal,
    /// Best ask price
    pub ask_price: Decimal,
    /// Bid size
    pub bid_size: Decimal,
    /// Ask size
    pub ask_size: Decimal,
    /// Timestamp of the tick
    pub timestamp: DateTime<Utc>,
    /// Sequence number (for ordering)
    pub sequence: u64,
}

impl TickData {
    /// Create a new tick
    pub fn new(
        symbol: &str,
        exchange: &str,
        bid_price: Decimal,
        ask_price: Decimal,
        bid_size: Decimal,
        ask_size: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            exchange: exchange.to_string(),
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            timestamp,
            sequence: 0,
        }
    }

    /// Get the mid price
    pub fn mid_price(&self) -> Decimal {
        (self.bid_price + self.ask_price) / Decimal::from(2)
    }

    /// Get the spread in absolute terms
    pub fn spread(&self) -> Decimal {
        self.ask_price - self.bid_price
    }

    /// Get the spread in basis points
    pub fn spread_bps(&self) -> Decimal {
        if self.mid_price() == Decimal::ZERO {
            return Decimal::ZERO;
        }
        (self.spread() / self.mid_price()) * Decimal::from(10_000)
    }

    /// Check if tick data is valid
    pub fn is_valid(&self) -> bool {
        self.bid_price > Decimal::ZERO
            && self.ask_price > Decimal::ZERO
            && self.ask_price >= self.bid_price
    }
}

/// Trade data representing an executed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    /// Trading symbol (normalized)
    pub symbol: String,
    /// Source exchange
    pub exchange: String,
    /// Trade price
    pub price: Decimal,
    /// Trade size
    pub size: Decimal,
    /// Trade side (Buy or Sell - aggressor side)
    pub side: TradeSide,
    /// Trade ID
    pub trade_id: String,
    /// Timestamp of the trade
    pub timestamp: DateTime<Utc>,
}

/// Trade side (aggressor)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
    Unknown,
}

impl std::fmt::Display for TradeSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeSide::Buy => write!(f, "buy"),
            TradeSide::Sell => write!(f, "sell"),
            TradeSide::Unknown => write!(f, "unknown"),
        }
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookData {
    /// Trading symbol (normalized)
    pub symbol: String,
    /// Source exchange
    pub exchange: String,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
}

impl OrderBookData {
    /// Get best bid
    pub fn best_bid(&self) -> Option<&OrderBookLevel> {
        self.bids.first()
    }

    /// Get best ask
    pub fn best_ask(&self) -> Option<&OrderBookLevel> {
        self.asks.first()
    }

    /// Convert to tick data
    pub fn to_tick(&self) -> Option<TickData> {
        let bid = self.best_bid()?;
        let ask = self.best_ask()?;
        Some(TickData {
            symbol: self.symbol.clone(),
            exchange: self.exchange.clone(),
            bid_price: bid.price,
            ask_price: ask.price,
            bid_size: bid.size,
            ask_size: ask.size,
            timestamp: self.timestamp,
            sequence: 0,
        })
    }
}

/// Candle/OHLCV data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleData {
    /// Trading symbol (normalized)
    pub symbol: String,
    /// Source exchange
    pub exchange: String,
    /// Candle open time
    pub open_time: DateTime<Utc>,
    /// Candle close time
    pub close_time: DateTime<Utc>,
    /// Open price
    pub open: Decimal,
    /// High price
    pub high: Decimal,
    /// Low price
    pub low: Decimal,
    /// Close price
    pub close: Decimal,
    /// Volume
    pub volume: Decimal,
    /// Number of trades
    pub trade_count: u64,
}

/// Market event enum for unified processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEvent {
    /// Tick update (best bid/ask)
    Tick(TickData),
    /// Trade executed
    Trade(TradeData),
    /// Order book snapshot
    OrderBook(OrderBookData),
    /// Candle update
    Candle(CandleData),
    /// Connection status change
    ConnectionStatus {
        exchange: String,
        connected: bool,
        message: Option<String>,
    },
    /// End of data stream (for historical replay)
    EndOfData,
}

impl MarketEvent {
    /// Get the timestamp of the event
    pub fn timestamp(&self) -> Option<DateTime<Utc>> {
        match self {
            MarketEvent::Tick(t) => Some(t.timestamp),
            MarketEvent::Trade(t) => Some(t.timestamp),
            MarketEvent::OrderBook(ob) => Some(ob.timestamp),
            MarketEvent::Candle(c) => Some(c.close_time),
            MarketEvent::ConnectionStatus { .. } => Some(Utc::now()),
            MarketEvent::EndOfData => None,
        }
    }

    /// Get the symbol if applicable
    pub fn symbol(&self) -> Option<&str> {
        match self {
            MarketEvent::Tick(t) => Some(&t.symbol),
            MarketEvent::Trade(t) => Some(&t.symbol),
            MarketEvent::OrderBook(ob) => Some(&ob.symbol),
            MarketEvent::Candle(c) => Some(&c.symbol),
            MarketEvent::ConnectionStatus { .. } => None,
            MarketEvent::EndOfData => None,
        }
    }

    /// Get the exchange if applicable
    pub fn exchange(&self) -> Option<&str> {
        match self {
            MarketEvent::Tick(t) => Some(&t.exchange),
            MarketEvent::Trade(t) => Some(&t.exchange),
            MarketEvent::OrderBook(ob) => Some(&ob.exchange),
            MarketEvent::Candle(c) => Some(&c.exchange),
            MarketEvent::ConnectionStatus { exchange, .. } => Some(exchange),
            MarketEvent::EndOfData => None,
        }
    }

    /// Check if this is a tick event
    pub fn is_tick(&self) -> bool {
        matches!(self, MarketEvent::Tick(_))
    }

    /// Check if this is a trade event
    pub fn is_trade(&self) -> bool {
        matches!(self, MarketEvent::Trade(_))
    }
}

/// Data feed statistics
#[derive(Debug, Clone, Default)]
pub struct DataFeedStats {
    /// Total events received
    pub events_received: u64,
    /// Events by type
    pub events_by_type: HashMap<String, u64>,
    /// Events by exchange
    pub events_by_exchange: HashMap<String, u64>,
    /// Events by symbol
    pub events_by_symbol: HashMap<String, u64>,
    /// First event timestamp
    pub first_event_time: Option<DateTime<Utc>>,
    /// Last event timestamp
    pub last_event_time: Option<DateTime<Utc>>,
    /// Current replay position (for historical)
    pub replay_position: Option<DateTime<Utc>>,
}

impl DataFeedStats {
    /// Record an event
    pub fn record_event(&mut self, event: &MarketEvent) {
        self.events_received += 1;

        // Track by type
        let event_type = match event {
            MarketEvent::Tick(_) => "tick",
            MarketEvent::Trade(_) => "trade",
            MarketEvent::OrderBook(_) => "orderbook",
            MarketEvent::Candle(_) => "candle",
            MarketEvent::ConnectionStatus { .. } => "connection",
            MarketEvent::EndOfData => "end_of_data",
        };
        *self
            .events_by_type
            .entry(event_type.to_string())
            .or_insert(0) += 1;

        // Track by exchange
        if let Some(exchange) = event.exchange() {
            *self
                .events_by_exchange
                .entry(exchange.to_string())
                .or_insert(0) += 1;
        }

        // Track by symbol
        if let Some(symbol) = event.symbol() {
            *self.events_by_symbol.entry(symbol.to_string()).or_insert(0) += 1;
        }

        // Track timestamps
        if let Some(ts) = event.timestamp() {
            if self.first_event_time.is_none() {
                self.first_event_time = Some(ts);
            }
            self.last_event_time = Some(ts);
        }
    }

    /// Get events per second rate
    pub fn events_per_second(&self) -> f64 {
        match (self.first_event_time, self.last_event_time) {
            (Some(first), Some(last)) => {
                let duration = (last - first).num_milliseconds() as f64 / 1000.0;
                if duration > 0.0 {
                    self.events_received as f64 / duration
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

/// Trait for data feed implementations
pub trait DataFeed: Send + Sync {
    /// Start the data feed
    fn start(
        &mut self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), DataFeedError>> + Send + '_>>;

    /// Stop the data feed
    fn stop(
        &mut self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), DataFeedError>> + Send + '_>>;

    /// Subscribe to events
    fn subscribe(&self) -> broadcast::Receiver<MarketEvent>;

    /// Get current statistics
    fn stats(&self) -> DataFeedStats;

    /// Check if feed is running
    fn is_running(&self) -> bool;

    /// Get feed name/description
    fn name(&self) -> &str;
}

/// Multi-exchange aggregated data feed
pub struct AggregatedDataFeed {
    /// Name of this feed
    name: String,
    /// Event broadcaster
    event_tx: broadcast::Sender<MarketEvent>,
    /// Statistics
    stats: Arc<parking_lot::RwLock<DataFeedStats>>,
    /// Running flag
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Latest ticks by symbol and exchange
    latest_ticks: Arc<parking_lot::RwLock<HashMap<(String, String), TickData>>>,
}

impl AggregatedDataFeed {
    /// Create a new aggregated data feed
    pub fn new(name: &str) -> Self {
        let (event_tx, _) = broadcast::channel(10_000);
        Self {
            name: name.to_string(),
            event_tx,
            stats: Arc::new(parking_lot::RwLock::new(DataFeedStats::default())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            latest_ticks: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }

    /// Create a new aggregated data feed that shares an existing broadcast channel
    ///
    /// This allows multiple data feeds to publish to the same subscribers,
    /// which is useful for bridging live exchange data to the simulation environment.
    ///
    /// # Arguments
    ///
    /// * `sender` - An existing broadcast sender to share
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let original_feed = AggregatedDataFeed::new("main");
    /// let bridge_feed = AggregatedDataFeed::with_sender(original_feed.sender());
    ///
    /// // Events published to bridge_feed will be received by subscribers of original_feed
    /// bridge_feed.publish(MarketEvent::Tick(tick));
    /// ```
    pub fn with_sender(sender: broadcast::Sender<MarketEvent>) -> Self {
        Self {
            name: "bridge".to_string(),
            event_tx: sender,
            stats: Arc::new(parking_lot::RwLock::new(DataFeedStats::default())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            latest_ticks: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }

    /// Create a new aggregated data feed with a custom name that shares an existing broadcast channel
    pub fn with_sender_named(name: &str, sender: broadcast::Sender<MarketEvent>) -> Self {
        Self {
            name: name.to_string(),
            event_tx: sender,
            stats: Arc::new(parking_lot::RwLock::new(DataFeedStats::default())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            latest_ticks: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }

    /// Publish an event to subscribers
    pub fn publish(&self, event: MarketEvent) {
        // Update stats
        self.stats.write().record_event(&event);

        // Cache latest tick
        if let MarketEvent::Tick(ref tick) = event {
            let key = (tick.symbol.clone(), tick.exchange.clone());
            self.latest_ticks.write().insert(key, tick.clone());
        }

        // Broadcast to subscribers
        let _ = self.event_tx.send(event);
    }

    /// Get latest tick for a symbol/exchange pair
    pub fn get_latest_tick(&self, symbol: &str, exchange: &str) -> Option<TickData> {
        let key = (symbol.to_string(), exchange.to_string());
        self.latest_ticks.read().get(&key).cloned()
    }

    /// Get all latest ticks
    pub fn get_all_latest_ticks(&self) -> Vec<TickData> {
        self.latest_ticks.read().values().cloned().collect()
    }

    /// Get best price across exchanges for a symbol
    pub fn get_best_bid(&self, symbol: &str) -> Option<(String, Decimal)> {
        self.latest_ticks
            .read()
            .iter()
            .filter(|((s, _), tick)| s == symbol && tick.is_valid())
            .max_by(|(_, a), (_, b)| a.bid_price.cmp(&b.bid_price))
            .map(|((_, exchange), tick)| (exchange.clone(), tick.bid_price))
    }

    /// Get best ask across exchanges for a symbol
    pub fn get_best_ask(&self, symbol: &str) -> Option<(String, Decimal)> {
        self.latest_ticks
            .read()
            .iter()
            .filter(|((s, _), tick)| s == symbol && tick.is_valid())
            .min_by(|(_, a), (_, b)| a.ask_price.cmp(&b.ask_price))
            .map(|((_, exchange), tick)| (exchange.clone(), tick.ask_price))
    }

    /// Get cross-exchange spread for a symbol (best ask - best bid, which may be negative for arb)
    pub fn get_cross_spread(&self, symbol: &str) -> Option<Decimal> {
        let best_bid = self.get_best_bid(symbol)?;
        let best_ask = self.get_best_ask(symbol)?;
        Some(best_ask.1 - best_bid.1)
    }

    /// Set running state
    pub fn set_running(&self, running: bool) {
        self.running
            .store(running, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get event sender for external publishers
    pub fn sender(&self) -> broadcast::Sender<MarketEvent> {
        self.event_tx.clone()
    }
}

impl DataFeed for AggregatedDataFeed {
    fn start(
        &mut self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), DataFeedError>> + Send + '_>> {
        Box::pin(async move {
            self.running
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        })
    }

    fn stop(
        &mut self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), DataFeedError>> + Send + '_>> {
        Box::pin(async move {
            self.running
                .store(false, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        })
    }

    fn subscribe(&self) -> broadcast::Receiver<MarketEvent> {
        self.event_tx.subscribe()
    }

    fn stats(&self) -> DataFeedStats {
        self.stats.read().clone()
    }

    fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Default for AggregatedDataFeed {
    fn default() -> Self {
        Self::new("aggregated")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_tick_data() {
        let tick = TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(50000.0),
            dec!(50010.0),
            dec!(1.5),
            dec!(2.0),
            Utc::now(),
        );

        assert!(tick.is_valid());
        assert_eq!(tick.mid_price(), dec!(50005.0));
        assert_eq!(tick.spread(), dec!(10.0));
        // Spread bps = (10 / 50005) * 10000 ≈ 2.0
        assert!(tick.spread_bps() > dec!(1.9) && tick.spread_bps() < dec!(2.1));
    }

    #[test]
    fn test_invalid_tick() {
        let tick = TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(0.0), // Invalid bid
            dec!(50010.0),
            dec!(1.5),
            dec!(2.0),
            Utc::now(),
        );
        assert!(!tick.is_valid());

        let tick2 = TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(50010.0),
            dec!(50000.0), // Ask < Bid
            dec!(1.5),
            dec!(2.0),
            Utc::now(),
        );
        assert!(!tick2.is_valid());
    }

    #[test]
    fn test_market_event() {
        let tick = TickData::new(
            "ETH/USDT",
            "binance",
            dec!(3000.0),
            dec!(3001.0),
            dec!(10.0),
            dec!(10.0),
            Utc::now(),
        );

        let event = MarketEvent::Tick(tick);
        assert!(event.is_tick());
        assert!(!event.is_trade());
        assert_eq!(event.symbol(), Some("ETH/USDT"));
        assert_eq!(event.exchange(), Some("binance"));
    }

    #[test]
    fn test_orderbook_to_tick() {
        let ob = OrderBookData {
            symbol: "BTC/USDT".to_string(),
            exchange: "bybit".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: dec!(50000.0),
                    size: dec!(1.0),
                },
                OrderBookLevel {
                    price: dec!(49999.0),
                    size: dec!(2.0),
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: dec!(50001.0),
                    size: dec!(1.5),
                },
                OrderBookLevel {
                    price: dec!(50002.0),
                    size: dec!(2.5),
                },
            ],
            timestamp: Utc::now(),
        };

        let tick = ob.to_tick().unwrap();
        assert_eq!(tick.bid_price, dec!(50000.0));
        assert_eq!(tick.ask_price, dec!(50001.0));
    }

    #[test]
    fn test_data_feed_stats() {
        let mut stats = DataFeedStats::default();

        let tick = TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(50000.0),
            dec!(50010.0),
            dec!(1.5),
            dec!(2.0),
            Utc::now(),
        );

        stats.record_event(&MarketEvent::Tick(tick));
        assert_eq!(stats.events_received, 1);
        assert_eq!(stats.events_by_type.get("tick"), Some(&1));
        assert_eq!(stats.events_by_exchange.get("kraken"), Some(&1));
        assert_eq!(stats.events_by_symbol.get("BTC/USDT"), Some(&1));
    }

    #[tokio::test]
    async fn test_aggregated_data_feed() {
        let feed = AggregatedDataFeed::new("test");
        let mut rx = feed.subscribe();

        let tick = TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(50000.0),
            dec!(50010.0),
            dec!(1.5),
            dec!(2.0),
            Utc::now(),
        );

        feed.publish(MarketEvent::Tick(tick.clone()));

        let received = rx.recv().await.unwrap();
        assert!(received.is_tick());

        // Check cached tick
        let cached = feed.get_latest_tick("BTC/USDT", "kraken").unwrap();
        assert_eq!(cached.bid_price, dec!(50000.0));
    }

    #[tokio::test]
    async fn test_with_sender_shares_channel() {
        // Create original feed
        let original_feed = AggregatedDataFeed::new("original");
        let mut rx = original_feed.subscribe();

        // Create a second feed that shares the same channel
        let shared_feed = AggregatedDataFeed::with_sender(original_feed.sender());

        // Publish to the shared feed
        let tick = TickData::new(
            "ETH/USDT",
            "binance",
            dec!(3000.0),
            dec!(3001.0),
            dec!(10.0),
            dec!(10.0),
            Utc::now(),
        );
        shared_feed.publish(MarketEvent::Tick(tick.clone()));

        // Should receive on the original feed's subscriber
        let received = rx.recv().await.unwrap();
        assert!(received.is_tick());
        if let MarketEvent::Tick(t) = received {
            assert_eq!(t.symbol, "ETH/USDT");
            assert_eq!(t.exchange, "binance");
        }
    }

    #[tokio::test]
    async fn test_with_sender_named() {
        let original_feed = AggregatedDataFeed::new("original");
        let mut rx = original_feed.subscribe();

        // Create named shared feed
        let shared_feed = AggregatedDataFeed::with_sender_named("bridge", original_feed.sender());
        assert_eq!(shared_feed.name(), "bridge");

        // Verify channel sharing works
        let tick = TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(50000.0),
            dec!(50010.0),
            dec!(1.0),
            dec!(1.0),
            Utc::now(),
        );
        shared_feed.publish(MarketEvent::Tick(tick));

        let received = rx.recv().await.unwrap();
        assert!(received.is_tick());
    }

    #[test]
    fn test_cross_exchange_best_prices() {
        let feed = AggregatedDataFeed::new("test");

        // Kraken tick
        let tick1 = TickData::new(
            "BTC/USDT",
            "kraken",
            dec!(50000.0),
            dec!(50020.0),
            dec!(1.0),
            dec!(1.0),
            Utc::now(),
        );
        feed.publish(MarketEvent::Tick(tick1));

        // Binance tick with better prices
        let tick2 = TickData::new(
            "BTC/USDT",
            "binance",
            dec!(50005.0), // Higher bid
            dec!(50015.0), // Lower ask
            dec!(1.0),
            dec!(1.0),
            Utc::now(),
        );
        feed.publish(MarketEvent::Tick(tick2));

        // Best bid should be from Binance (50005)
        let (best_bid_exchange, best_bid) = feed.get_best_bid("BTC/USDT").unwrap();
        assert_eq!(best_bid_exchange, "binance");
        assert_eq!(best_bid, dec!(50005.0));

        // Best ask should be from Binance (50015)
        let (best_ask_exchange, best_ask) = feed.get_best_ask("BTC/USDT").unwrap();
        assert_eq!(best_ask_exchange, "binance");
        assert_eq!(best_ask, dec!(50015.0));

        // Cross spread (best ask - best bid)
        let spread = feed.get_cross_spread("BTC/USDT").unwrap();
        assert_eq!(spread, dec!(10.0));
    }
}
