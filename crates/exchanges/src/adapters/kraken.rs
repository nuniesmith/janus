//! Kraken Exchange Adapter
//!
//! Implements WebSocket connection to Kraken for real-time market data.
//!
//! ## API Documentation:
//! - WebSocket API: https://docs.kraken.com/websockets/
//! - Public Feeds: https://docs.kraken.com/websockets/#message-subscribe
//!
//! ## WebSocket Endpoint:
//! - Production: wss://ws.kraken.com
//! - Authentication: wss://ws-auth.kraken.com (for private feeds)
//!
//! ## Supported Channels:
//! - `trade`: Real-time trade executions
//! - `ticker`: 24hr rolling statistics
//! - `book`: Order book snapshots and updates (10, 25, 100, 500, 1000 levels)
//! - `spread`: Best bid/ask updates
//! - `ohlc`: OHLC data (1, 5, 15, 30, 60, 240, 1440, 10080, 21600 minutes)
//!
//! ## Symbol Format:
//! Kraken uses pairs like "XBT/USD" or "ETH/USD" in API responses.
//! Note: BTC is referred to as XBT on Kraken.
//!
//! ## Message Format (Trade):
//! ```json
//! [
//!   0,
//!   [
//!     ["50000.00", "0.001", "1609459200.123456", "b", "m", ""]
//!   ],
//!   "trade",
//!   "XBT/USD"
//! ]
//! ```
//! Trade array: [price, volume, time, side, orderType, misc]
//! - side: "b" = buy, "s" = sell
//! - orderType: "m" = market, "l" = limit
//!
//! ## Message Format (Ticker):
//! ```json
//! [
//!   0,
//!   {
//!     "a": ["50100.00", 1, "1.000"],
//!     "b": ["50000.00", 2, "2.000"],
//!     "c": ["50050.00", "0.001"],
//!     "v": ["1000.5", "5000.2"],
//!     "p": ["50025.00", "49950.00"],
//!     "t": [500, 2000],
//!     "l": ["49000.00", "48000.00"],
//!     "h": ["51000.00", "52000.00"],
//!     "o": ["50000.00", "49000.00"]
//!   },
//!   "ticker",
//!   "XBT/USD"
//! ]
//! ```
//! - a = ask [price, whole lot volume, lot volume]
//! - b = bid [price, whole lot volume, lot volume]
//! - c = close [price, lot volume]
//! - v = volume [today, last 24 hours]
//! - p = volume weighted average price [today, last 24 hours]
//! - t = number of trades [today, last 24 hours]
//! - l = low [today, last 24 hours]
//! - h = high [today, last 24 hours]
//! - o = open [today, last 24 hours]

use anyhow::{Context, Result};
use janus_core::{Exchange, MarketDataEvent, Side, Symbol, TradeEvent};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;
use tracing::{debug, trace, warn};

/// Kraken exchange adapter
pub struct KrakenAdapter {
    ws_url: String,
}

impl KrakenAdapter {
    /// Create a new Kraken adapter
    pub fn new() -> Self {
        Self {
            ws_url: "wss://ws.kraken.com".to_string(),
        }
    }

    /// Create adapter with custom WebSocket URL (for testing)
    pub fn with_url(ws_url: String) -> Self {
        Self { ws_url }
    }

    /// Get the WebSocket URL
    pub fn ws_url(&self) -> &str {
        &self.ws_url
    }

    /// Build subscription message for a symbol
    pub fn subscribe_message(&self, symbol: &Symbol, channels: &[KrakenChannel]) -> String {
        let pair = self.format_pair(symbol);
        let channel_names: Vec<&str> = channels.iter().map(|c| c.as_str()).collect();

        let msg = KrakenSubscribeMessage {
            event: "subscribe".to_string(),
            pair: vec![pair],
            subscription: KrakenSubscription {
                name: channel_names
                    .first()
                    .copied()
                    .unwrap_or("trade")
                    .to_string(),
            },
        };

        serde_json::to_string(&msg).unwrap_or_else(|_| "{}".to_string())
    }

    /// Build default subscription message (trade + ticker)
    pub fn default_subscribe(&self, symbol: &Symbol) -> String {
        self.subscribe_message(symbol, &[KrakenChannel::Trade])
    }

    /// Parse WebSocket message into market data events
    pub fn parse_message(&self, raw: &str) -> Result<Vec<MarketDataEvent>> {
        trace!("Kraken: Parsing message: {}", raw);

        // First, try to parse as JSON
        let value: Value =
            serde_json::from_str(raw).context("Failed to parse Kraken message as JSON")?;

        // Kraken messages can be:
        // 1. Arrays (data messages): [channelID, data, channelName, pair]
        // 2. Objects (control messages): {"event": "...", ...}
        match value {
            Value::Array(arr) => self.parse_data_message(&arr),
            Value::Object(obj) => self.parse_control_message(&obj),
            _ => {
                warn!("Kraken: Unexpected message format");
                Ok(vec![])
            }
        }
    }

    /// Parse data messages (array format)
    fn parse_data_message(&self, arr: &[Value]) -> Result<Vec<MarketDataEvent>> {
        if arr.len() < 4 {
            return Ok(vec![]);
        }

        let channel_name = arr[2].as_str().unwrap_or("");
        let pair = arr[3].as_str().unwrap_or("");

        match channel_name {
            "trade" => self.parse_trade_data(&arr[1], pair),
            "ticker" => self.parse_ticker_data(&arr[1], pair),
            "book-10" | "book-25" | "book-100" | "book-500" | "book-1000" => {
                self.parse_book_data(&arr[1], pair)
            }
            "ohlc" | "ohlc-1" | "ohlc-5" | "ohlc-15" | "ohlc-60" => {
                self.parse_ohlc_data(&arr[1], pair)
            }
            _ => {
                debug!("Kraken: Unknown channel: {}", channel_name);
                Ok(vec![])
            }
        }
    }

    /// Parse control messages (object format)
    fn parse_control_message(
        &self,
        obj: &serde_json::Map<String, Value>,
    ) -> Result<Vec<MarketDataEvent>> {
        let event = obj.get("event").and_then(|e| e.as_str()).unwrap_or("");

        match event {
            "systemStatus" => {
                let status = obj.get("status").and_then(|s| s.as_str()).unwrap_or("");
                debug!("Kraken: System status: {}", status);
                Ok(vec![])
            }
            "subscriptionStatus" => {
                let status = obj.get("status").and_then(|s| s.as_str()).unwrap_or("");
                debug!("Kraken: Subscription status: {}", status);
                Ok(vec![])
            }
            "heartbeat" => {
                trace!("Kraken: Heartbeat received");
                Ok(vec![])
            }
            "error" => {
                let error_msg = obj
                    .get("errorMessage")
                    .and_then(|e| e.as_str())
                    .unwrap_or("Unknown error");
                warn!("Kraken: Error: {}", error_msg);
                Ok(vec![])
            }
            _ => {
                debug!("Kraken: Unknown event: {}", event);
                Ok(vec![])
            }
        }
    }

    /// Parse trade data
    fn parse_trade_data(&self, data: &Value, pair: &str) -> Result<Vec<MarketDataEvent>> {
        let trades = data.as_array().context("Trade data is not an array")?;
        let mut events = Vec::new();

        let symbol = self.parse_pair(pair)?;

        for trade_array in trades {
            let trade = trade_array.as_array().context("Trade is not an array")?;

            if trade.len() < 6 {
                continue;
            }

            let price = Decimal::from_str(trade[0].as_str().unwrap_or("0"))
                .context("Failed to parse trade price")?;
            let quantity = Decimal::from_str(trade[1].as_str().unwrap_or("0"))
                .context("Failed to parse trade volume")?;
            let timestamp = trade[2]
                .as_str()
                .unwrap_or("0")
                .parse::<f64>()
                .context("Failed to parse trade timestamp")?;
            let side_str = trade[3].as_str().unwrap_or("");

            let side = match side_str {
                "b" => Side::Buy,
                "s" => Side::Sell,
                _ => {
                    warn!("Kraken: Unknown side: {}", side_str);
                    continue;
                }
            };

            // Convert timestamp to microseconds
            let timestamp_micros = (timestamp * 1_000_000.0) as i64;

            // Generate a trade ID from timestamp and price
            let trade_id = format!("{}-{}", timestamp_micros, price);

            let event = TradeEvent::new(
                Exchange::Kraken,
                symbol.clone(),
                timestamp_micros,
                price,
                quantity,
                side,
                trade_id,
            );

            events.push(MarketDataEvent::Trade(event));
        }

        Ok(events)
    }

    /// Parse ticker data
    fn parse_ticker_data(&self, data: &Value, pair: &str) -> Result<Vec<MarketDataEvent>> {
        let ticker = data.as_object().context("Ticker data is not an object")?;
        let symbol = self.parse_pair(pair)?;

        // Extract close price (last trade)
        let last_price = ticker
            .get("c")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|p| p.as_str())
            .and_then(|s| Decimal::from_str(s).ok())
            .unwrap_or(Decimal::ZERO);

        // Extract best bid
        let best_bid = ticker
            .get("b")
            .and_then(|b| b.as_array())
            .and_then(|arr| arr.first())
            .and_then(|p| p.as_str())
            .and_then(|s| Decimal::from_str(s).ok());

        // Extract best ask
        let best_ask = ticker
            .get("a")
            .and_then(|a| a.as_array())
            .and_then(|arr| arr.first())
            .and_then(|p| p.as_str())
            .and_then(|s| Decimal::from_str(s).ok());

        // Extract 24h volume (second element is last 24h)
        let volume_24h = ticker
            .get("v")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(1))
            .and_then(|v| v.as_str())
            .and_then(|s| Decimal::from_str(s).ok())
            .unwrap_or(Decimal::ZERO);

        // Extract 24h high
        let high_24h = ticker
            .get("h")
            .and_then(|h| h.as_array())
            .and_then(|arr| arr.get(1))
            .and_then(|h| h.as_str())
            .and_then(|s| Decimal::from_str(s).ok());

        // Extract 24h low
        let low_24h = ticker
            .get("l")
            .and_then(|l| l.as_array())
            .and_then(|arr| arr.get(1))
            .and_then(|l| l.as_str())
            .and_then(|s| Decimal::from_str(s).ok());

        // Extract opening price (today)
        let open_today = ticker
            .get("o")
            .and_then(|o| o.as_array())
            .and_then(|arr| arr.first())
            .and_then(|o| o.as_str())
            .and_then(|s| Decimal::from_str(s).ok());

        // Calculate price change
        let price_change_pct_24h = if let Some(open) = open_today {
            if open > Decimal::ZERO {
                Some(((last_price - open) / open) * Decimal::from(100))
            } else {
                None
            }
        } else {
            None
        };

        let ticker_event = janus_core::TickerEvent {
            exchange: Exchange::Kraken,
            symbol,
            timestamp: chrono::Utc::now().timestamp_micros(),
            last_price,
            best_bid,
            best_ask,
            volume_24h,
            quote_volume_24h: Decimal::ZERO,
            price_change_24h: None,
            price_change_pct_24h,
            high_24h,
            low_24h,
        };

        Ok(vec![MarketDataEvent::Ticker(ticker_event)])
    }

    /// Parse order book data
    fn parse_book_data(&self, data: &Value, pair: &str) -> Result<Vec<MarketDataEvent>> {
        let symbol = self.parse_pair(pair)?;

        // Kraken book data can be either a snapshot or an update
        // Snapshot format: {"as": [[price, volume, timestamp]], "bs": [[price, volume, timestamp]]}
        // Update format: {"a": [[price, volume, timestamp]], "b": [[price, volume, timestamp]]}

        let book_obj = data.as_object().context("Book data is not an object")?;

        let mut bids = Vec::new();
        let mut asks = Vec::new();
        let mut is_snapshot = false;

        // Check for snapshot format (as/bs keys)
        if let Some(asks_data) = book_obj.get("as").and_then(|a| a.as_array()) {
            is_snapshot = true;
            for ask_array in asks_data {
                if let Some(ask) = ask_array.as_array()
                    && ask.len() >= 2
                    && let (Some(price_str), Some(volume_str)) = (ask[0].as_str(), ask[1].as_str())
                    && let (Ok(price), Ok(volume)) =
                        (Decimal::from_str(price_str), Decimal::from_str(volume_str))
                    && volume > Decimal::ZERO
                {
                    asks.push(janus_core::PriceLevel::new(price, volume));
                }
            }
        }

        if let Some(bids_data) = book_obj.get("bs").and_then(|b| b.as_array()) {
            is_snapshot = true;
            for bid_array in bids_data {
                if let Some(bid) = bid_array.as_array()
                    && bid.len() >= 2
                    && let (Some(price_str), Some(volume_str)) = (bid[0].as_str(), bid[1].as_str())
                    && let (Ok(price), Ok(volume)) =
                        (Decimal::from_str(price_str), Decimal::from_str(volume_str))
                    && volume > Decimal::ZERO
                {
                    bids.push(janus_core::PriceLevel::new(price, volume));
                }
            }
        }

        // Check for update format (a/b keys)
        if let Some(asks_data) = book_obj.get("a").and_then(|a| a.as_array()) {
            for ask_array in asks_data {
                if let Some(ask) = ask_array.as_array()
                    && ask.len() >= 2
                    && let (Some(price_str), Some(volume_str)) = (ask[0].as_str(), ask[1].as_str())
                    && let (Ok(price), Ok(volume)) =
                        (Decimal::from_str(price_str), Decimal::from_str(volume_str))
                    && volume > Decimal::ZERO
                {
                    asks.push(janus_core::PriceLevel::new(price, volume));
                }
            }
        }

        if let Some(bids_data) = book_obj.get("b").and_then(|b| b.as_array()) {
            for bid_array in bids_data {
                if let Some(bid) = bid_array.as_array()
                    && bid.len() >= 2
                    && let (Some(price_str), Some(volume_str)) = (bid[0].as_str(), bid[1].as_str())
                    && let (Ok(price), Ok(volume)) =
                        (Decimal::from_str(price_str), Decimal::from_str(volume_str))
                    && volume > Decimal::ZERO
                {
                    bids.push(janus_core::PriceLevel::new(price, volume));
                }
            }
        }

        // Sort bids descending (highest first), asks ascending (lowest first)
        bids.sort_by(|a, b| b.price.cmp(&a.price));
        asks.sort_by(|a, b| a.price.cmp(&b.price));

        let orderbook_event = janus_core::OrderBookEvent {
            exchange: Exchange::Kraken,
            symbol,
            timestamp: chrono::Utc::now().timestamp_micros(),
            sequence: 0, // Kraken doesn't provide sequence numbers in the same way
            is_snapshot,
            bids,
            asks,
        };

        Ok(vec![MarketDataEvent::OrderBook(orderbook_event)])
    }

    /// Parse OHLC data
    ///
    /// Kraken OHLC message format:
    /// [channelID, [time, etime, open, high, low, close, vwap, volume, count], "ohlc-interval", "pair"]
    ///
    /// Where:
    /// - time: Start time of interval (Unix timestamp as string with decimal)
    /// - etime: End time of interval (Unix timestamp as string with decimal)
    /// - open: Opening price
    /// - high: Highest price
    /// - low: Lowest price
    /// - close: Closing price
    /// - vwap: Volume weighted average price
    /// - volume: Volume in base currency
    /// - count: Number of trades
    fn parse_ohlc_data(&self, data: &Value, pair: &str) -> Result<Vec<MarketDataEvent>> {
        let arr = data.as_array().context("OHLC data is not an array")?;

        if arr.len() < 9 {
            return Err(anyhow::anyhow!(
                "Invalid OHLC array length: expected 9, got {}",
                arr.len()
            ));
        }

        // Parse the symbol
        let symbol = self.parse_pair(pair)?;

        // Parse timestamp (Unix timestamp with decimal)
        let time_str = arr[0].as_str().context("OHLC time is not a string")?;
        let open_time = time_str
            .parse::<f64>()
            .context("Failed to parse OHLC time")?;
        let open_time_micros = (open_time * 1_000_000.0) as i64;

        // Parse end time
        let etime_str = arr[1].as_str().context("OHLC etime is not a string")?;
        let close_time = etime_str
            .parse::<f64>()
            .context("Failed to parse OHLC etime")?;
        let close_time_micros = (close_time * 1_000_000.0) as i64;

        // Parse OHLC prices
        let open = Decimal::from_str(arr[2].as_str().context("OHLC open is not a string")?)
            .context("Failed to parse OHLC open")?;

        let high = Decimal::from_str(arr[3].as_str().context("OHLC high is not a string")?)
            .context("Failed to parse OHLC high")?;

        let low = Decimal::from_str(arr[4].as_str().context("OHLC low is not a string")?)
            .context("Failed to parse OHLC low")?;

        let close = Decimal::from_str(arr[5].as_str().context("OHLC close is not a string")?)
            .context("Failed to parse OHLC close")?;

        // Parse vwap (volume weighted average price) - not used in KlineEvent but useful for logging
        let _vwap = Decimal::from_str(arr[6].as_str().context("OHLC vwap is not a string")?)
            .context("Failed to parse OHLC vwap")?;

        // Parse volume
        let volume = Decimal::from_str(arr[7].as_str().context("OHLC volume is not a string")?)
            .context("Failed to parse OHLC volume")?;

        // Parse trade count
        let trades = arr[8]
            .as_u64()
            .or_else(|| arr[8].as_str().and_then(|s| s.parse().ok()))
            .context("Failed to parse OHLC trade count")?;

        // Determine interval from channel name (this would need to be passed in or stored)
        // For now, default to "1m" - in practice this should come from the subscription
        let interval = "1m".to_string();

        let kline_event = janus_core::KlineEvent {
            exchange: Exchange::Kraken,
            symbol,
            interval,
            open_time: open_time_micros,
            close_time: close_time_micros,
            open,
            high,
            low,
            close,
            volume,
            quote_volume: None, // Kraken doesn't provide this directly
            trades: Some(trades),
            is_closed: false, // Kraken streams updates, final candle comes at interval end
        };

        trace!(
            "Kraken OHLC: {} O:{} H:{} L:{} C:{} V:{}",
            pair, open, high, low, close, volume
        );

        Ok(vec![MarketDataEvent::Kline(kline_event)])
    }

    /// Format symbol for Kraken API
    fn format_pair(&self, symbol: &Symbol) -> String {
        // Kraken uses XBT instead of BTC
        let base = if symbol.base.to_uppercase() == "BTC" {
            "XBT"
        } else {
            &symbol.base
        };

        // Kraken prefers "/" separator
        format!("{}/{}", base.to_uppercase(), symbol.quote.to_uppercase())
    }

    /// Parse Kraken pair to Symbol
    fn parse_pair(&self, pair: &str) -> Result<Symbol> {
        // Remove any "/" separators
        let clean_pair = pair.replace("/", "");

        // Common quote currencies
        let quote_currencies = ["USD", "USDT", "USDC", "EUR", "GBP", "JPY"];

        for quote in &quote_currencies {
            if clean_pair.ends_with(quote) {
                let base = clean_pair.trim_end_matches(quote);

                // Convert XBT back to BTC
                let base_normalized = if base == "XBT" { "BTC" } else { base };

                return Ok(Symbol::new(base_normalized, *quote));
            }
        }

        Err(anyhow::anyhow!("Failed to parse Kraken pair: {}", pair))
    }
}

impl Default for KrakenAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Kraken WebSocket channels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KrakenChannel {
    Trade,
    Ticker,
    Book10,
    Book25,
    Spread,
    Ohlc1,
    Ohlc5,
    Ohlc15,
    Ohlc60,
}

impl KrakenChannel {
    fn as_str(&self) -> &str {
        match self {
            KrakenChannel::Trade => "trade",
            KrakenChannel::Ticker => "ticker",
            KrakenChannel::Book10 => "book",
            KrakenChannel::Book25 => "book",
            KrakenChannel::Spread => "spread",
            KrakenChannel::Ohlc1 => "ohlc",
            KrakenChannel::Ohlc5 => "ohlc-5",
            KrakenChannel::Ohlc15 => "ohlc-15",
            KrakenChannel::Ohlc60 => "ohlc-60",
        }
    }
}

/// Kraken subscription message
#[derive(Debug, Serialize, Deserialize)]
struct KrakenSubscribeMessage {
    event: String,
    pair: Vec<String>,
    subscription: KrakenSubscription,
}

#[derive(Debug, Serialize, Deserialize)]
struct KrakenSubscription {
    name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_pair() {
        let adapter = KrakenAdapter::new();

        let btc_symbol = Symbol::new("BTC", "USD");
        assert_eq!(adapter.format_pair(&btc_symbol), "XBT/USD");

        let eth_symbol = Symbol::new("ETH", "USD");
        assert_eq!(adapter.format_pair(&eth_symbol), "ETH/USD");
    }

    #[test]
    fn test_parse_pair() {
        let adapter = KrakenAdapter::new();

        let symbol = adapter.parse_pair("XBT/USD").unwrap();
        assert_eq!(symbol.base, "BTC");
        assert_eq!(symbol.quote, "USD");

        let symbol = adapter.parse_pair("ETH/USDT").unwrap();
        assert_eq!(symbol.base, "ETH");
        assert_eq!(symbol.quote, "USDT");
    }

    #[test]
    fn test_parse_book_snapshot() {
        let adapter = KrakenAdapter::new();
        let raw = r#"[
            0,
            {
                "as": [
                    ["50001.00", "1.5", "1609459200.123456"],
                    ["50002.00", "2.0", "1609459200.123456"],
                    ["50003.00", "0.5", "1609459200.123456"]
                ],
                "bs": [
                    ["50000.00", "2.5", "1609459200.123456"],
                    ["49999.00", "1.0", "1609459200.123456"],
                    ["49998.00", "3.0", "1609459200.123456"]
                ]
            },
            "book-10",
            "XBT/USD"
        ]"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        if let MarketDataEvent::OrderBook(book) = &events[0] {
            assert_eq!(book.exchange, Exchange::Kraken);
            assert_eq!(book.symbol.base, "BTC");
            assert_eq!(book.symbol.quote, "USD");
            assert!(book.is_snapshot);
            assert_eq!(book.bids.len(), 3);
            assert_eq!(book.asks.len(), 3);

            // Verify bids are sorted descending
            assert!(book.bids[0].price > book.bids[1].price);
            assert_eq!(book.bids[0].price, Decimal::from_str("50000.00").unwrap());
            assert_eq!(book.bids[0].quantity, Decimal::from_str("2.5").unwrap());

            // Verify asks are sorted ascending
            assert!(book.asks[0].price < book.asks[1].price);
            assert_eq!(book.asks[0].price, Decimal::from_str("50001.00").unwrap());
            assert_eq!(book.asks[0].quantity, Decimal::from_str("1.5").unwrap());

            // Test spread calculation
            let spread = book.spread().unwrap();
            assert_eq!(spread, Decimal::from_str("1.00").unwrap());
        } else {
            panic!("Expected OrderBook event");
        }
    }

    #[test]
    fn test_parse_book_update() {
        let adapter = KrakenAdapter::new();
        let raw = r#"[
            0,
            {
                "a": [
                    ["50005.00", "5.5", "1609459201.123456"]
                ],
                "b": [
                    ["49995.00", "2.2", "1609459201.123456"],
                    ["49994.00", "0.0", "1609459201.123456"]
                ]
            },
            "book-10",
            "ETH/USDT"
        ]"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        if let MarketDataEvent::OrderBook(book) = &events[0] {
            assert_eq!(book.symbol.base, "ETH");
            assert_eq!(book.symbol.quote, "USDT");
            assert!(!book.is_snapshot);
            // Zero quantity updates should be filtered out
            assert_eq!(book.bids.len(), 1);
            assert_eq!(book.asks.len(), 1);
            assert_eq!(book.bids[0].price, Decimal::from_str("49995.00").unwrap());
            assert_eq!(book.bids[0].quantity, Decimal::from_str("2.2").unwrap());
        } else {
            panic!("Expected OrderBook event");
        }
    }

    #[test]
    fn test_parse_trade() {
        let adapter = KrakenAdapter::new();
        let raw = r#"[
            0,
            [
                ["50000.00", "0.001", "1609459200.123456", "b", "m", ""]
            ],
            "trade",
            "XBT/USD"
        ]"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            MarketDataEvent::Trade(trade) => {
                assert_eq!(trade.exchange, Exchange::Kraken);
                assert_eq!(trade.symbol.base, "BTC");
                assert_eq!(trade.symbol.quote, "USD");
                assert_eq!(trade.price, Decimal::from_str("50000.00").unwrap());
                assert_eq!(trade.quantity, Decimal::from_str("0.001").unwrap());
                assert_eq!(trade.side, Side::Buy);
            }
            _ => panic!("Expected TradeEvent"),
        }
    }

    #[test]
    fn test_parse_subscription_status() {
        let adapter = KrakenAdapter::new();
        let raw = r#"{
            "event": "subscriptionStatus",
            "status": "subscribed",
            "pair": "XBT/USD",
            "subscription": {"name": "trade"}
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_subscribe_message() {
        let adapter = KrakenAdapter::new();
        let symbol = Symbol::new("BTC", "USD");
        let msg = adapter.default_subscribe(&symbol);

        assert!(msg.contains("subscribe"));
        assert!(msg.contains("XBT/USD"));
        assert!(msg.contains("trade"));
    }
}
