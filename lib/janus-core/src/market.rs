//! Unified Market Data Types for JANUS
//!
//! This module defines normalized event types for market data across all exchanges.
//! All exchange-specific connectors convert their data into these unified types.
//!
//! ## MarketDataBus
//!
//! The [`MarketDataBus`] provides an in-process broadcast channel for streaming
//! live market data between JANUS modules. The Data module publishes
//! [`MarketDataEvent`]s and the Forward module subscribes to consume them for
//! indicator calculation and strategy-driven signal generation.

use chrono::Utc;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use tokio::sync::broadcast;

/// Unified market data event envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MarketDataEvent {
    /// Real-time trade execution
    Trade(TradeEvent),
    /// Order book snapshot or update
    OrderBook(OrderBookEvent),
    /// Ticker/24h statistics
    Ticker(TickerEvent),
    /// Liquidation event (futures)
    Liquidation(LiquidationEvent),
    /// Funding rate update (futures)
    FundingRate(FundingRateEvent),
    /// OHLCV candle/kline
    Kline(KlineEvent),
}

impl MarketDataEvent {
    /// Get the symbol for this event
    pub fn symbol(&self) -> &Symbol {
        match self {
            MarketDataEvent::Trade(e) => &e.symbol,
            MarketDataEvent::OrderBook(e) => &e.symbol,
            MarketDataEvent::Ticker(e) => &e.symbol,
            MarketDataEvent::Liquidation(e) => &e.symbol,
            MarketDataEvent::FundingRate(e) => &e.symbol,
            MarketDataEvent::Kline(e) => &e.symbol,
        }
    }

    /// Get the exchange for this event
    pub fn exchange(&self) -> Exchange {
        match self {
            MarketDataEvent::Trade(e) => e.exchange,
            MarketDataEvent::OrderBook(e) => e.exchange,
            MarketDataEvent::Ticker(e) => e.exchange,
            MarketDataEvent::Liquidation(e) => e.exchange,
            MarketDataEvent::FundingRate(e) => e.exchange,
            MarketDataEvent::Kline(e) => e.exchange,
        }
    }

    /// Get the timestamp for this event
    pub fn timestamp(&self) -> i64 {
        match self {
            MarketDataEvent::Trade(e) => e.timestamp,
            MarketDataEvent::OrderBook(e) => e.timestamp,
            MarketDataEvent::Ticker(e) => e.timestamp,
            MarketDataEvent::Liquidation(e) => e.timestamp,
            MarketDataEvent::FundingRate(e) => e.timestamp,
            MarketDataEvent::Kline(e) => e.close_time,
        }
    }
}

/// Normalized trade event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEvent {
    /// Exchange that produced this trade
    pub exchange: Exchange,
    /// Trading pair symbol
    pub symbol: Symbol,
    /// Trade execution timestamp (exchange time, Unix microseconds)
    pub timestamp: i64,
    /// Reception timestamp (our system time, Unix microseconds)
    pub received_at: i64,
    /// Execution price
    pub price: Decimal,
    /// Trade quantity/size
    pub quantity: Decimal,
    /// Trade side from taker perspective
    pub side: Side,
    /// Exchange-specific trade ID
    pub trade_id: String,
    /// Whether buyer was the market maker (if available)
    pub buyer_is_maker: Option<bool>,
}

impl TradeEvent {
    /// Create a new trade event with current reception time
    pub fn new(
        exchange: Exchange,
        symbol: Symbol,
        timestamp: i64,
        price: Decimal,
        quantity: Decimal,
        side: Side,
        trade_id: String,
    ) -> Self {
        Self {
            exchange,
            symbol,
            timestamp,
            received_at: Utc::now().timestamp_micros(),
            price,
            quantity,
            side,
            trade_id,
            buyer_is_maker: None,
        }
    }

    /// Calculate the notional value of this trade
    pub fn notional(&self) -> Decimal {
        self.price * self.quantity
    }

    /// Get the latency between exchange timestamp and reception
    pub fn latency_micros(&self) -> i64 {
        self.received_at - self.timestamp
    }
}

/// Order book snapshot or delta update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookEvent {
    /// Exchange that produced this order book
    pub exchange: Exchange,
    /// Trading pair symbol
    pub symbol: Symbol,
    /// Event timestamp (Unix microseconds)
    pub timestamp: i64,
    /// Sequence number for ordering updates
    pub sequence: u64,
    /// Whether this is a snapshot (true) or delta (false)
    pub is_snapshot: bool,
    /// Bid levels (price, quantity)
    pub bids: Vec<PriceLevel>,
    /// Ask levels (price, quantity)
    pub asks: Vec<PriceLevel>,
}

impl OrderBookEvent {
    /// Get the best bid price
    pub fn best_bid(&self) -> Option<&PriceLevel> {
        self.bids.first()
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<&PriceLevel> {
        self.asks.first()
    }

    /// Calculate the mid price
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / Decimal::from(2)),
            _ => None,
        }
    }

    /// Calculate the spread
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }

    /// Calculate the spread in basis points
    pub fn spread_bps(&self) -> Option<Decimal> {
        match (self.mid_price(), self.spread()) {
            (Some(mid), Some(spread)) if mid > Decimal::ZERO => {
                Some((spread / mid) * Decimal::from(10000))
            }
            _ => None,
        }
    }
}

/// Price level in order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    /// Price at this level
    pub price: Decimal,
    /// Total quantity available at this price
    pub quantity: Decimal,
}

impl PriceLevel {
    pub fn new(price: Decimal, quantity: Decimal) -> Self {
        Self { price, quantity }
    }

    /// Calculate notional value at this level
    pub fn notional(&self) -> Decimal {
        self.price * self.quantity
    }
}

/// 24-hour ticker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: i64,
    pub last_price: Decimal,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub volume_24h: Decimal,
    pub quote_volume_24h: Decimal,
    pub price_change_24h: Option<Decimal>,
    pub price_change_pct_24h: Option<Decimal>,
    pub high_24h: Option<Decimal>,
    pub low_24h: Option<Decimal>,
}

/// Liquidation event (futures markets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidationEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: i64,
    /// Side of the liquidated position
    pub side: Side,
    /// Liquidation price
    pub price: Decimal,
    /// Liquidated quantity
    pub quantity: Decimal,
    /// Order ID (if available)
    pub order_id: Option<String>,
}

impl LiquidationEvent {
    /// Calculate the notional value of the liquidation
    pub fn notional(&self) -> Decimal {
        self.price * self.quantity
    }
}

/// Funding rate event (perpetual futures)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRateEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub timestamp: i64,
    /// Current funding rate (as a percentage, e.g., 0.01 = 1%)
    pub rate: Decimal,
    /// Next funding time (Unix timestamp)
    pub next_funding_time: i64,
}

impl FundingRateEvent {
    /// Get annualized funding rate (assuming 8-hour funding intervals)
    pub fn annualized_rate(&self) -> Decimal {
        self.rate * Decimal::from(365 * 3) // 3 funding periods per day
    }
}

/// OHLCV candlestick/kline data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineEvent {
    pub exchange: Exchange,
    pub symbol: Symbol,
    /// Interval (e.g., "1m", "5m", "1h")
    pub interval: String,
    /// Candle open time (Unix microseconds)
    pub open_time: i64,
    /// Candle close time (Unix microseconds)
    pub close_time: i64,
    /// Opening price
    pub open: Decimal,
    /// Highest price
    pub high: Decimal,
    /// Lowest price
    pub low: Decimal,
    /// Closing price
    pub close: Decimal,
    /// Volume in base currency
    pub volume: Decimal,
    /// Quote volume (volume * price)
    pub quote_volume: Option<Decimal>,
    /// Number of trades in this candle
    pub trades: Option<u64>,
    /// Whether this candle is closed/complete
    pub is_closed: bool,
}

impl KlineEvent {
    /// Get the typical price (HLC/3)
    pub fn typical_price(&self) -> Decimal {
        (self.high + self.low + self.close) / Decimal::from(3)
    }

    /// Get the price change for this candle
    pub fn price_change(&self) -> Decimal {
        self.close - self.open
    }

    /// Get the price change percentage
    pub fn price_change_pct(&self) -> Decimal {
        if self.open > Decimal::ZERO {
            ((self.close - self.open) / self.open) * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }

    /// Get the candle range (high - low)
    pub fn range(&self) -> Decimal {
        self.high - self.low
    }
}

/// Trading pair symbol
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    /// Base currency (e.g., BTC)
    pub base: String,
    /// Quote currency (e.g., USDT)
    pub quote: String,
    /// Market type
    pub market_type: MarketType,
}

impl Symbol {
    pub fn new(base: impl Into<String>, quote: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            quote: quote.into(),
            market_type: MarketType::Spot,
        }
    }

    pub fn new_with_type(
        base: impl Into<String>,
        quote: impl Into<String>,
        market_type: MarketType,
    ) -> Self {
        Self {
            base: base.into(),
            quote: quote.into(),
            market_type,
        }
    }

    /// Parse symbol from exchange-specific format
    pub fn from_exchange_format(s: &str, exchange: Exchange) -> Option<Self> {
        match exchange {
            Exchange::Binance => {
                // Binance: BTCUSD
                if s.ends_with("USDT") {
                    let base = s.trim_end_matches("USDT");
                    Some(Symbol::new(base, "USDT"))
                } else if s.ends_with("BUSD") {
                    let base = s.trim_end_matches("BUSD");
                    Some(Symbol::new(base, "BUSD"))
                } else {
                    None
                }
            }
            Exchange::Bybit => {
                // Bybit: BTCUSD
                if s.ends_with("USDT") {
                    let base = s.trim_end_matches("USDT");
                    Some(Symbol::new(base, "USDT"))
                } else {
                    None
                }
            }
            Exchange::Coinbase => {
                // Coinbase: BTC-USD
                let parts: Vec<&str> = s.split('-').collect();
                if parts.len() == 2 {
                    Some(Symbol::new(parts[0], parts[1]))
                } else {
                    None
                }
            }
            Exchange::Kraken => {
                // Kraken: BTC/USD or XXBTZUSD
                if s.contains('/') {
                    let parts: Vec<&str> = s.split('/').collect();
                    if parts.len() == 2 {
                        Some(Symbol::new(parts[0], parts[1]))
                    } else {
                        None
                    }
                } else {
                    // Try to parse XXBTZUSD format
                    None // Complex, implement if needed
                }
            }
            Exchange::Okx => {
                // OKX: BTC-USDT
                let parts: Vec<&str> = s.split('-').collect();
                if parts.len() == 2 {
                    Some(Symbol::new(parts[0], parts[1]))
                } else {
                    None
                }
            }
            Exchange::Kucoin => {
                // Kucoin: BTC-USDT
                let parts: Vec<&str> = s.split('-').collect();
                if parts.len() == 2 {
                    Some(Symbol::new(parts[0], parts[1]))
                } else {
                    None
                }
            }
        }
    }

    /// Format symbol for specific exchange
    pub fn to_exchange_format(&self, exchange: Exchange) -> String {
        match exchange {
            Exchange::Binance => {
                format!("{}{}", self.base.to_uppercase(), self.quote.to_uppercase())
            }
            Exchange::Bybit => format!("{}{}", self.base.to_uppercase(), self.quote.to_uppercase()),
            Exchange::Coinbase => {
                format!("{}-{}", self.base.to_uppercase(), self.quote.to_uppercase())
            }
            Exchange::Kraken => {
                format!("{}/{}", self.base.to_uppercase(), self.quote.to_uppercase())
            }
            Exchange::Okx => format!("{}-{}", self.base.to_uppercase(), self.quote.to_uppercase()),
            Exchange::Kucoin => {
                format!("{}-{}", self.base.to_uppercase(), self.quote.to_uppercase())
            }
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.base, self.quote)
    }
}

/// Supported cryptocurrency exchanges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Exchange {
    Binance,
    Bybit,
    Coinbase,
    Kraken,
    Okx,
    Kucoin,
}

impl fmt::Display for Exchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exchange::Binance => write!(f, "binance"),
            Exchange::Bybit => write!(f, "bybit"),
            Exchange::Coinbase => write!(f, "coinbase"),
            Exchange::Kraken => write!(f, "kraken"),
            Exchange::Okx => write!(f, "okx"),
            Exchange::Kucoin => write!(f, "kucoin"),
        }
    }
}

impl std::str::FromStr for Exchange {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "binance" => Ok(Exchange::Binance),
            "bybit" => Ok(Exchange::Bybit),
            "coinbase" => Ok(Exchange::Coinbase),
            "kraken" => Ok(Exchange::Kraken),
            "okx" => Ok(Exchange::Okx),
            "kucoin" => Ok(Exchange::Kucoin),
            _ => Err(format!("Unknown exchange: {}", s)),
        }
    }
}

/// Market type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketType {
    /// Spot market
    Spot,
    /// Perpetual futures
    Perpetual,
    /// Dated futures
    Futures,
    /// Options
    Options,
}

impl fmt::Display for MarketType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketType::Spot => write!(f, "spot"),
            MarketType::Perpetual => write!(f, "perpetual"),
            MarketType::Futures => write!(f, "futures"),
            MarketType::Options => write!(f, "options"),
        }
    }
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    /// Buy/Long
    Buy,
    /// Sell/Short
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "buy"),
            Side::Sell => write!(f, "sell"),
        }
    }
}

impl std::str::FromStr for Side {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "buy" | "bid" | "long" => Ok(Side::Buy),
            "sell" | "ask" | "short" => Ok(Side::Sell),
            _ => Err(format!("Unknown side: {}", s)),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MarketDataBus — broadcast channel for live market data between modules
// ═══════════════════════════════════════════════════════════════════════════

/// Broadcast bus for streaming market data events between JANUS modules.
///
/// The Data module publishes [`MarketDataEvent`]s (trades, klines, order book
/// updates, etc.) and the Forward module subscribes to receive them for
/// real-time indicator calculation and strategy evaluation.
///
/// This is the in-process equivalent of the [`SignalBus`](crate::SignalBus)
/// but for raw/normalised market data rather than trading signals.
///
/// # Example
///
/// ```rust,no_run
/// use janus_core::market::{MarketDataBus, MarketDataEvent};
///
/// let bus = MarketDataBus::new(5000);
/// let mut rx = bus.subscribe();
///
/// // Publisher (Data module)
/// // bus.publish(some_event).unwrap();
///
/// // Consumer (Forward module)
/// // let event = rx.recv().await.unwrap();
/// ```
pub struct MarketDataBus {
    tx: broadcast::Sender<MarketDataEvent>,
    capacity: usize,
}

impl MarketDataBus {
    /// Create a new market data bus with the given channel capacity.
    ///
    /// A larger capacity reduces the chance of slow subscribers missing
    /// events but consumes more memory. 5 000 is a reasonable default for
    /// multi-asset 1-minute candle + trade ingestion.
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx, capacity }
    }

    /// Publish a market data event to all subscribers.
    ///
    /// Returns the number of active receivers that will see the event.
    pub fn publish(&self, event: MarketDataEvent) -> crate::Result<usize> {
        let receivers = self.tx.send(event)?;
        Ok(receivers)
    }

    /// Subscribe to the market data stream.
    pub fn subscribe(&self) -> broadcast::Receiver<MarketDataEvent> {
        self.tx.subscribe()
    }

    /// Current number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Channel capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for MarketDataBus {
    fn default() -> Self {
        Self::new(5000)
    }
}

impl Clone for MarketDataBus {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            capacity: self.capacity,
        }
    }
}

#[cfg(test)]
mod market_data_bus_tests {
    use super::*;

    #[test]
    fn test_market_data_bus() {
        let bus = MarketDataBus::new(100);
        assert_eq!(bus.subscriber_count(), 0);
        assert_eq!(bus.capacity(), 100);

        let _rx = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);

        // Clone preserves the same underlying channel
        let bus2 = bus.clone();
        assert_eq!(bus2.subscriber_count(), 1);

        let _rx2 = bus2.subscribe();
        assert_eq!(bus.subscriber_count(), 2);
    }

    #[test]
    fn test_market_data_bus_default() {
        let bus = MarketDataBus::default();
        assert_eq!(bus.capacity(), 5000);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_parsing() {
        let sym = Symbol::from_exchange_format("BTCUSDT", Exchange::Binance).unwrap();
        assert_eq!(sym.base, "BTC");
        assert_eq!(sym.quote, "USDT");

        let sym = Symbol::from_exchange_format("BTC-USD", Exchange::Coinbase).unwrap();
        assert_eq!(sym.base, "BTC");
        assert_eq!(sym.quote, "USD");
    }

    #[test]
    fn test_symbol_formatting() {
        let sym = Symbol::new("BTC", "USDT");
        assert_eq!(sym.to_exchange_format(Exchange::Binance), "BTCUSDT");
        assert_eq!(sym.to_exchange_format(Exchange::Coinbase), "BTC-USDT");
        assert_eq!(sym.to_exchange_format(Exchange::Kraken), "BTC/USDT");
    }

    #[test]
    fn test_spread_calculation() {
        let event = OrderBookEvent {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTC", "USDT"),
            timestamp: 0,
            sequence: 1,
            is_snapshot: true,
            bids: vec![PriceLevel::new(Decimal::from(50000), Decimal::from(1))],
            asks: vec![PriceLevel::new(Decimal::from(50010), Decimal::from(1))],
        };

        assert_eq!(event.spread(), Some(Decimal::from(10)));
        assert_eq!(event.mid_price(), Some(Decimal::from(50005)));
    }

    #[test]
    fn test_kline_calculations() {
        let kline = KlineEvent {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTC", "USDT"),
            interval: "1m".to_string(),
            open_time: 0,
            close_time: 60_000_000,
            open: Decimal::from(50000),
            high: Decimal::from(51000),
            low: Decimal::from(49000),
            close: Decimal::from(50500),
            volume: Decimal::from(100),
            quote_volume: None,
            trades: None,
            is_closed: true,
        };

        assert_eq!(kline.price_change(), Decimal::from(500));
        assert_eq!(kline.range(), Decimal::from(2000));
    }
}
