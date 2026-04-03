//! Market Data Types for Multi-Exchange Support
//!
//! This module provides unified types for market data across all supported exchanges.
//!
//! ## Exchange Priority
//!
//! | Priority | Exchange | Role                              |
//! |----------|----------|-----------------------------------|
//! | 1st      | Kraken   | **Primary** — default data & REST |
//! | 2nd      | Bybit    | Backup / alternate                |
//! | 3rd      | Binance  | Tertiary / additional liquidity   |
//!
//! All market data WebSocket streams are FREE and do not require API keys.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Exchange Identification
// ============================================================================

/// Supported exchanges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExchangeId {
    Bybit,
    Kraken,
    Binance,
}

impl ExchangeId {
    /// Get the display name
    pub fn name(&self) -> &'static str {
        match self {
            ExchangeId::Bybit => "bybit",
            ExchangeId::Kraken => "kraken",
            ExchangeId::Binance => "binance",
        }
    }

    /// Get the WebSocket URL for market data (FREE - no API key required)
    pub fn market_data_ws_url(&self) -> &'static str {
        match self {
            ExchangeId::Bybit => "wss://stream.bybit.com/v5/public/spot",
            ExchangeId::Kraken => "wss://ws.kraken.com/v2",
            ExchangeId::Binance => "wss://stream.binance.com:9443/ws",
        }
    }

    /// Get the testnet WebSocket URL (if available)
    pub fn testnet_ws_url(&self) -> Option<&'static str> {
        match self {
            ExchangeId::Bybit => Some("wss://stream-testnet.bybit.com/v5/public/spot"),
            ExchangeId::Kraken => None, // Kraken doesn't have public testnet WS
            ExchangeId::Binance => Some("wss://testnet.binance.vision/ws"),
        }
    }

    /// Get the REST API base URL
    pub fn rest_api_url(&self) -> &'static str {
        match self {
            ExchangeId::Bybit => "https://api.bybit.com",
            ExchangeId::Kraken => "https://api.kraken.com",
            ExchangeId::Binance => "https://api.binance.com",
        }
    }

    /// Get the testnet REST API URL (if available)
    pub fn testnet_rest_url(&self) -> Option<&'static str> {
        match self {
            ExchangeId::Bybit => Some("https://api-testnet.bybit.com"),
            ExchangeId::Kraken => None,
            ExchangeId::Binance => Some("https://testnet.binance.vision"),
        }
    }
}

impl std::fmt::Display for ExchangeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for ExchangeId {
    /// The default exchange is **Kraken** (primary for data websockets & REST).
    fn default() -> Self {
        Self::primary()
    }
}

impl ExchangeId {
    /// The primary exchange used for data websockets and REST APIs.
    ///
    /// Currently **Kraken**. All new connections should prefer this exchange
    /// unless a specific routing rule overrides it.
    pub fn primary() -> Self {
        ExchangeId::Kraken
    }

    /// The fallback exchange used when the primary is unavailable.
    ///
    /// Currently **Bybit**.
    pub fn fallback() -> Self {
        ExchangeId::Bybit
    }

    /// All supported exchanges ordered by priority (primary first).
    ///
    /// Use this when initialising a [`super::provider::MarketDataAggregator`]
    /// to ensure the highest-priority provider is registered first.
    ///
    /// Order: Kraken → Bybit → Binance
    pub fn all_by_priority() -> &'static [ExchangeId] {
        &[ExchangeId::Kraken, ExchangeId::Bybit, ExchangeId::Binance]
    }

    /// Whether this exchange is the primary (highest-priority) data source.
    pub fn is_primary(&self) -> bool {
        *self == Self::primary()
    }

    /// Whether this exchange is the fallback data source.
    pub fn is_fallback(&self) -> bool {
        *self == Self::fallback()
    }
}

impl std::str::FromStr for ExchangeId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bybit" => Ok(ExchangeId::Bybit),
            "kraken" => Ok(ExchangeId::Kraken),
            "binance" => Ok(ExchangeId::Binance),
            _ => Err(format!("Unknown exchange: {}", s)),
        }
    }
}

// ============================================================================
// Symbol Normalization
// ============================================================================

/// Normalized trading pair representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TradingPair {
    /// Base asset (e.g., "BTC")
    pub base: String,
    /// Quote asset (e.g., "USDT", "USD")
    pub quote: String,
}

impl TradingPair {
    pub fn new(base: impl Into<String>, quote: impl Into<String>) -> Self {
        Self {
            base: base.into().to_uppercase(),
            quote: quote.into().to_uppercase(),
        }
    }

    /// Create from normalized format "BTC/USDT"
    pub fn from_normalized(symbol: &str) -> Option<Self> {
        let parts: Vec<&str> = symbol.split('/').collect();
        if parts.len() == 2 {
            Some(Self::new(parts[0], parts[1]))
        } else {
            None
        }
    }

    /// Get normalized format "BTC/USDT"
    pub fn normalized(&self) -> String {
        format!("{}/{}", self.base, self.quote)
    }

    /// Convert to Bybit format "BTCUSD"
    pub fn to_bybit(&self) -> String {
        format!("{}{}", self.base, self.quote)
    }

    /// Convert to Kraken format "BTC/USD" (note: USDT -> USD for some pairs)
    pub fn to_kraken(&self) -> String {
        let quote = if self.quote == "USDT" {
            "USD"
        } else {
            &self.quote
        };
        format!("{}/{}", self.base, quote)
    }

    /// Convert to Binance format "btcusdt" (lowercase)
    pub fn to_binance(&self) -> String {
        format!("{}{}", self.base, self.quote).to_lowercase()
    }

    /// Convert to exchange-specific format
    pub fn to_exchange(&self, exchange: ExchangeId) -> String {
        match exchange {
            ExchangeId::Bybit => self.to_bybit(),
            ExchangeId::Kraken => self.to_kraken(),
            ExchangeId::Binance => self.to_binance(),
        }
    }
}

impl std::fmt::Display for TradingPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.normalized())
    }
}

/// Symbol normalizer for converting between exchange formats
pub struct SymbolNormalizer {
    /// Custom mappings for special cases
    custom_mappings: HashMap<(ExchangeId, String), TradingPair>,
}

impl SymbolNormalizer {
    pub fn new() -> Self {
        Self {
            custom_mappings: HashMap::new(),
        }
    }

    /// Add a custom mapping for edge cases
    pub fn add_mapping(
        &mut self,
        exchange: ExchangeId,
        exchange_symbol: String,
        pair: TradingPair,
    ) {
        self.custom_mappings
            .insert((exchange, exchange_symbol), pair);
    }

    /// Normalize an exchange-specific symbol to TradingPair
    pub fn normalize(&self, exchange: ExchangeId, exchange_symbol: &str) -> Option<TradingPair> {
        // Check custom mappings first
        if let Some(pair) = self
            .custom_mappings
            .get(&(exchange, exchange_symbol.to_string()))
        {
            return Some(pair.clone());
        }

        // Standard normalization
        match exchange {
            ExchangeId::Bybit => self.normalize_bybit(exchange_symbol),
            ExchangeId::Kraken => self.normalize_kraken(exchange_symbol),
            ExchangeId::Binance => self.normalize_binance(exchange_symbol),
        }
    }

    fn normalize_bybit(&self, symbol: &str) -> Option<TradingPair> {
        // Bybit format: "BTCUSD"
        let symbol = symbol.to_uppercase();
        for quote in &["USDT", "USDC", "BTC", "ETH"] {
            if symbol.ends_with(quote) {
                let base = &symbol[..symbol.len() - quote.len()];
                if !base.is_empty() {
                    return Some(TradingPair::new(base, *quote));
                }
            }
        }
        None
    }

    fn normalize_kraken(&self, symbol: &str) -> Option<TradingPair> {
        // Kraken format: "BTC/USD" or "XBT/USD"
        if let Some(pair) = TradingPair::from_normalized(symbol) {
            // Handle Kraken's XBT -> BTC mapping
            let base = if pair.base == "XBT" {
                "BTC".to_string()
            } else {
                pair.base
            };
            let quote = if pair.quote == "USD" {
                "USDT".to_string()
            } else {
                pair.quote
            };
            return Some(TradingPair::new(base, quote));
        }
        None
    }

    fn normalize_binance(&self, symbol: &str) -> Option<TradingPair> {
        // Binance format: "btcusdt" (lowercase)
        let symbol = symbol.to_uppercase();
        for quote in &["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"] {
            if symbol.ends_with(quote) {
                let base = &symbol[..symbol.len() - quote.len()];
                if !base.is_empty() {
                    return Some(TradingPair::new(base, *quote));
                }
            }
        }
        None
    }
}

impl Default for SymbolNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Market Data Types
// ============================================================================

/// Ticker data (Level 1 - Best Bid/Offer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Source exchange
    pub exchange: ExchangeId,
    /// Normalized trading pair
    pub symbol: String,
    /// Best bid price
    pub bid: Decimal,
    /// Best bid quantity
    pub bid_qty: Decimal,
    /// Best ask price
    pub ask: Decimal,
    /// Best ask quantity
    pub ask_qty: Decimal,
    /// Last traded price
    pub last: Decimal,
    /// 24h volume (base currency)
    pub volume_24h: Decimal,
    /// 24h high
    pub high_24h: Decimal,
    /// 24h low
    pub low_24h: Decimal,
    /// 24h price change
    pub change_24h: Decimal,
    /// 24h price change percentage
    pub change_pct_24h: Decimal,
    /// Volume weighted average price
    pub vwap: Option<Decimal>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Ticker {
    /// Calculate the spread
    pub fn spread(&self) -> Decimal {
        self.ask - self.bid
    }

    /// Calculate spread as percentage
    pub fn spread_pct(&self) -> Decimal {
        if self.bid > Decimal::ZERO {
            (self.ask - self.bid) / self.bid * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> Decimal {
        (self.bid + self.ask) / Decimal::from(2)
    }
}

/// Trade/execution data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Source exchange
    pub exchange: ExchangeId,
    /// Normalized trading pair
    pub symbol: String,
    /// Trade ID from exchange
    pub trade_id: String,
    /// Trade price
    pub price: Decimal,
    /// Trade quantity
    pub quantity: Decimal,
    /// Trade side (from taker's perspective)
    pub side: TradeSide,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Trade {
    /// Calculate trade value
    pub fn value(&self) -> Decimal {
        self.price * self.quantity
    }
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl std::fmt::Display for TradeSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeSide::Buy => write!(f, "BUY"),
            TradeSide::Sell => write!(f, "SELL"),
        }
    }
}

/// OHLCV Candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Source exchange
    pub exchange: ExchangeId,
    /// Normalized trading pair
    pub symbol: String,
    /// Candle interval in minutes
    pub interval: u32,
    /// Open price
    pub open: Decimal,
    /// High price
    pub high: Decimal,
    /// Low price
    pub low: Decimal,
    /// Close price
    pub close: Decimal,
    /// Volume (base currency)
    pub volume: Decimal,
    /// Quote volume
    pub quote_volume: Option<Decimal>,
    /// Number of trades
    pub trades: Option<u64>,
    /// Volume weighted average price
    pub vwap: Option<Decimal>,
    /// Candle open timestamp
    pub open_time: DateTime<Utc>,
    /// Candle close timestamp
    pub close_time: DateTime<Utc>,
    /// Is this candle closed/final?
    pub is_closed: bool,
}

impl Candle {
    /// Calculate candle range (high - low)
    pub fn range(&self) -> Decimal {
        self.high - self.low
    }

    /// Calculate candle body (close - open, absolute)
    pub fn body(&self) -> Decimal {
        (self.close - self.open).abs()
    }

    /// Is this a bullish candle?
    pub fn is_bullish(&self) -> bool {
        self.close >= self.open
    }

    /// Is this a bearish candle?
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate upper wick
    pub fn upper_wick(&self) -> Decimal {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower wick
    pub fn lower_wick(&self) -> Decimal {
        self.close.min(self.open) - self.low
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: Decimal,
    /// Quantity at this level
    pub quantity: Decimal,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Source exchange
    pub exchange: ExchangeId,
    /// Normalized trading pair
    pub symbol: String,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
    /// Update sequence number
    pub sequence: Option<u64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl OrderBook {
    /// Get best bid
    pub fn best_bid(&self) -> Option<&OrderBookLevel> {
        self.bids.first()
    }

    /// Get best ask
    pub fn best_ask(&self) -> Option<&OrderBookLevel> {
        self.asks.first()
    }

    /// Calculate spread
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / Decimal::from(2)),
            _ => None,
        }
    }

    /// Calculate total bid depth
    pub fn total_bid_depth(&self) -> Decimal {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Calculate total ask depth
    pub fn total_ask_depth(&self) -> Decimal {
        self.asks.iter().map(|l| l.quantity).sum()
    }

    /// Calculate bid/ask imbalance (-1 to +1, positive means more bids)
    pub fn imbalance(&self) -> Decimal {
        let bid_depth = self.total_bid_depth();
        let ask_depth = self.total_ask_depth();
        let total = bid_depth + ask_depth;
        if total > Decimal::ZERO {
            (bid_depth - ask_depth) / total
        } else {
            Decimal::ZERO
        }
    }
}

// ============================================================================
// Aggregated Market Data
// ============================================================================

/// Best price across all exchanges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPrice {
    /// Normalized symbol
    pub symbol: String,
    /// Best bid price (highest across exchanges)
    pub best_bid: Decimal,
    /// Exchange with best bid
    pub best_bid_exchange: ExchangeId,
    /// Best ask price (lowest across exchanges)
    pub best_ask: Decimal,
    /// Exchange with best ask
    pub best_ask_exchange: ExchangeId,
    /// Spread across best prices
    pub spread: Decimal,
    /// All tickers by exchange
    pub tickers: HashMap<ExchangeId, Ticker>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl BestPrice {
    /// Calculate mid price
    pub fn mid_price(&self) -> Decimal {
        (self.best_bid + self.best_ask) / Decimal::from(2)
    }

    /// Is there an arbitrage opportunity?
    pub fn has_arbitrage(&self) -> bool {
        self.best_bid > self.best_ask
    }

    /// Calculate arbitrage profit (if any)
    pub fn arbitrage_profit_pct(&self) -> Decimal {
        if self.best_bid > self.best_ask && self.best_ask > Decimal::ZERO {
            (self.best_bid - self.best_ask) / self.best_ask * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }
}

// ============================================================================
// Market Data Events (for streaming)
// ============================================================================

/// Market data event types for WebSocket streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MarketDataEvent {
    /// Ticker update
    Ticker(Ticker),
    /// Trade execution
    Trade(Trade),
    /// Candle update
    Candle(Candle),
    /// Order book snapshot
    OrderBookSnapshot(OrderBook),
    /// Order book update (delta)
    OrderBookUpdate {
        exchange: ExchangeId,
        symbol: String,
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
        sequence: Option<u64>,
        timestamp: DateTime<Utc>,
    },
    /// Connection status change
    ConnectionStatus {
        exchange: ExchangeId,
        connected: bool,
        message: Option<String>,
        timestamp: DateTime<Utc>,
    },
    /// Error event
    Error {
        exchange: ExchangeId,
        code: Option<String>,
        message: String,
        timestamp: DateTime<Utc>,
    },
}

impl MarketDataEvent {
    /// Get the exchange for this event
    pub fn exchange(&self) -> ExchangeId {
        match self {
            MarketDataEvent::Ticker(t) => t.exchange,
            MarketDataEvent::Trade(t) => t.exchange,
            MarketDataEvent::Candle(c) => c.exchange,
            MarketDataEvent::OrderBookSnapshot(ob) => ob.exchange,
            MarketDataEvent::OrderBookUpdate { exchange, .. } => *exchange,
            MarketDataEvent::ConnectionStatus { exchange, .. } => *exchange,
            MarketDataEvent::Error { exchange, .. } => *exchange,
        }
    }

    /// Get the symbol for this event (if applicable)
    pub fn symbol(&self) -> Option<&str> {
        match self {
            MarketDataEvent::Ticker(t) => Some(&t.symbol),
            MarketDataEvent::Trade(t) => Some(&t.symbol),
            MarketDataEvent::Candle(c) => Some(&c.symbol),
            MarketDataEvent::OrderBookSnapshot(ob) => Some(&ob.symbol),
            MarketDataEvent::OrderBookUpdate { symbol, .. } => Some(symbol),
            MarketDataEvent::ConnectionStatus { .. } => None,
            MarketDataEvent::Error { .. } => None,
        }
    }
}

// ============================================================================
// Subscription Types
// ============================================================================

/// Subscription request for market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    /// Exchange to subscribe to
    pub exchange: ExchangeId,
    /// Channel type
    pub channel: SubscriptionChannel,
    /// Symbols to subscribe
    pub symbols: Vec<String>,
}

/// Subscription channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionChannel {
    /// Ticker/BBO updates
    Ticker,
    /// Trade stream
    Trades,
    /// Candle/OHLC stream with interval in minutes
    Candles { interval: u32 },
    /// Order book snapshots with depth
    OrderBook { depth: u32 },
    /// Order book updates (deltas)
    OrderBookUpdates,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_id_display() {
        assert_eq!(ExchangeId::Bybit.name(), "bybit");
        assert_eq!(ExchangeId::Kraken.name(), "kraken");
        assert_eq!(ExchangeId::Binance.name(), "binance");
    }

    #[test]
    fn test_exchange_id_from_str() {
        assert_eq!("bybit".parse::<ExchangeId>().unwrap(), ExchangeId::Bybit);
        assert_eq!("KRAKEN".parse::<ExchangeId>().unwrap(), ExchangeId::Kraken);
        assert_eq!(
            "Binance".parse::<ExchangeId>().unwrap(),
            ExchangeId::Binance
        );
        assert!("unknown".parse::<ExchangeId>().is_err());
    }

    #[test]
    fn test_trading_pair_normalization() {
        let pair = TradingPair::new("BTC", "USDT");
        assert_eq!(pair.normalized(), "BTC/USDT");
        assert_eq!(pair.to_bybit(), "BTCUSDT");
        assert_eq!(pair.to_kraken(), "BTC/USD");
        assert_eq!(pair.to_binance(), "btcusdt");
    }

    #[test]
    fn test_trading_pair_from_normalized() {
        let pair = TradingPair::from_normalized("ETH/USDT").unwrap();
        assert_eq!(pair.base, "ETH");
        assert_eq!(pair.quote, "USDT");
    }

    #[test]
    fn test_symbol_normalizer_bybit() {
        let normalizer = SymbolNormalizer::new();
        let pair = normalizer.normalize(ExchangeId::Bybit, "BTCUSDT").unwrap();
        assert_eq!(pair.base, "BTC");
        assert_eq!(pair.quote, "USDT");
    }

    #[test]
    fn test_symbol_normalizer_kraken() {
        let normalizer = SymbolNormalizer::new();
        let pair = normalizer.normalize(ExchangeId::Kraken, "BTC/USD").unwrap();
        assert_eq!(pair.base, "BTC");
        assert_eq!(pair.quote, "USDT"); // USD normalized to USDT
    }

    #[test]
    fn test_symbol_normalizer_binance() {
        let normalizer = SymbolNormalizer::new();
        let pair = normalizer
            .normalize(ExchangeId::Binance, "btcusdt")
            .unwrap();
        assert_eq!(pair.base, "BTC");
        assert_eq!(pair.quote, "USDT");
    }

    #[test]
    fn test_ticker_calculations() {
        let ticker = Ticker {
            exchange: ExchangeId::Bybit,
            symbol: "BTC/USDT".to_string(),
            bid: Decimal::from(67000),
            bid_qty: Decimal::from(1),
            ask: Decimal::from(67010),
            ask_qty: Decimal::from(1),
            last: Decimal::from(67005),
            volume_24h: Decimal::from(1000),
            high_24h: Decimal::from(68000),
            low_24h: Decimal::from(66000),
            change_24h: Decimal::from(100),
            change_pct_24h: Decimal::new(15, 2), // 0.15%
            vwap: Some(Decimal::from(67100)),
            timestamp: Utc::now(),
        };

        assert_eq!(ticker.spread(), Decimal::from(10));
        assert_eq!(ticker.mid_price(), Decimal::from(67005));
    }

    #[test]
    fn test_candle_calculations() {
        let candle = Candle {
            exchange: ExchangeId::Binance,
            symbol: "BTC/USDT".to_string(),
            interval: 5,
            open: Decimal::from(67000),
            high: Decimal::from(67500),
            low: Decimal::from(66800),
            close: Decimal::from(67300),
            volume: Decimal::from(100),
            quote_volume: None,
            trades: Some(500),
            vwap: None,
            open_time: Utc::now(),
            close_time: Utc::now(),
            is_closed: true,
        };

        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert_eq!(candle.range(), Decimal::from(700));
        assert_eq!(candle.body(), Decimal::from(300));
    }

    #[test]
    fn test_order_book_calculations() {
        let order_book = OrderBook {
            exchange: ExchangeId::Kraken,
            symbol: "BTC/USDT".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: Decimal::from(67000),
                    quantity: Decimal::from(1),
                },
                OrderBookLevel {
                    price: Decimal::from(66990),
                    quantity: Decimal::from(2),
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: Decimal::from(67010),
                    quantity: Decimal::from(1),
                },
                OrderBookLevel {
                    price: Decimal::from(67020),
                    quantity: Decimal::from(2),
                },
            ],
            sequence: Some(12345),
            timestamp: Utc::now(),
        };

        assert_eq!(order_book.spread(), Some(Decimal::from(10)));
        assert_eq!(order_book.mid_price(), Some(Decimal::from(67005)));
        assert_eq!(order_book.total_bid_depth(), Decimal::from(3));
        assert_eq!(order_book.total_ask_depth(), Decimal::from(3));
        assert_eq!(order_book.imbalance(), Decimal::ZERO);
    }

    #[test]
    fn test_best_price_arbitrage() {
        let best = BestPrice {
            symbol: "BTC/USDT".to_string(),
            best_bid: Decimal::from(67050), // Higher bid on one exchange
            best_bid_exchange: ExchangeId::Bybit,
            best_ask: Decimal::from(67000), // Lower ask on another
            best_ask_exchange: ExchangeId::Binance,
            spread: Decimal::from(-50),
            tickers: HashMap::new(),
            timestamp: Utc::now(),
        };

        assert!(best.has_arbitrage());
        assert!(best.arbitrage_profit_pct() > Decimal::ZERO);
    }

    #[test]
    fn test_market_data_event_exchange() {
        let event = MarketDataEvent::Trade(Trade {
            exchange: ExchangeId::Kraken,
            symbol: "BTC/USDT".to_string(),
            trade_id: "123".to_string(),
            price: Decimal::from(67000),
            quantity: Decimal::from(1),
            side: TradeSide::Buy,
            timestamp: Utc::now(),
        });

        assert_eq!(event.exchange(), ExchangeId::Kraken);
        assert_eq!(event.symbol(), Some("BTC/USDT"));
    }

    #[test]
    fn test_exchange_id_primary_is_kraken() {
        assert_eq!(ExchangeId::primary(), ExchangeId::Kraken);
    }

    #[test]
    fn test_exchange_id_fallback_is_bybit() {
        assert_eq!(ExchangeId::fallback(), ExchangeId::Bybit);
    }

    #[test]
    fn test_exchange_id_default_is_primary() {
        assert_eq!(ExchangeId::default(), ExchangeId::primary());
        assert_eq!(ExchangeId::default(), ExchangeId::Kraken);
    }

    #[test]
    fn test_exchange_id_all_by_priority_order() {
        let priority = ExchangeId::all_by_priority();
        assert_eq!(priority.len(), 3);
        assert_eq!(
            priority[0],
            ExchangeId::Kraken,
            "Kraken should be first (primary)"
        );
        assert_eq!(
            priority[1],
            ExchangeId::Bybit,
            "Bybit should be second (backup)"
        );
        assert_eq!(
            priority[2],
            ExchangeId::Binance,
            "Binance should be third (tertiary)"
        );
    }

    #[test]
    fn test_exchange_id_is_primary() {
        assert!(ExchangeId::Kraken.is_primary());
        assert!(!ExchangeId::Bybit.is_primary());
        assert!(!ExchangeId::Binance.is_primary());
    }

    #[test]
    fn test_exchange_id_is_fallback() {
        assert!(!ExchangeId::Kraken.is_fallback());
        assert!(ExchangeId::Bybit.is_fallback());
        assert!(!ExchangeId::Binance.is_fallback());
    }
}
