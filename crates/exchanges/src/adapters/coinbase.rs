//! Coinbase Exchange Adapter
//!
//! Implements WebSocket connection to Coinbase Advanced Trade API for real-time market data.
//!
//! ## API Documentation:
//! - WebSocket API: https://docs.cloud.coinbase.com/advanced-trade-api/docs/ws-overview
//! - Channels: https://docs.cloud.coinbase.com/advanced-trade-api/docs/ws-channels
//!
//! ## WebSocket Endpoint:
//! - Production: wss://advanced-trade-ws.coinbase.com
//!
//! ## Supported Channels:
//! - `ticker`: Real-time price updates
//! - `ticker_batch`: Batch ticker updates
//! - `level2`: Full order book snapshots and updates
//! - `user`: User-specific data (requires authentication)
//! - `matches`: Trade executions
//!
//! ## Authentication:
//! Coinbase Advanced Trade uses API key authentication for WebSocket connections.
//! Public channels (ticker, level2, matches) don't require authentication.
//!
//! ## Message Format (Ticker):
//! ```json
//! {
//!   "channel": "ticker",
//!   "client_id": "",
//!   "timestamp": "2023-01-01T00:00:00.000000Z",
//!   "sequence_num": 0,
//!   "events": [{
//!     "type": "ticker",
//!     "tickers": [{
//!       "type": "ticker",
//!       "product_id": "BTC-USD",
//!       "price": "50000.00",
//!       "volume_24_h": "1000.5",
//!       "low_24_h": "49000.00",
//!       "high_24_h": "51000.00",
//!       "low_52_w": "20000.00",
//!       "high_52_w": "69000.00",
//!       "price_percent_chg_24_h": "2.5"
//!     }]
//!   }]
//! }
//! ```
//!
//! ## Message Format (Match/Trade):
//! ```json
//! {
//!   "channel": "matches",
//!   "client_id": "",
//!   "timestamp": "2023-01-01T00:00:00.000000Z",
//!   "sequence_num": 0,
//!   "events": [{
//!     "type": "match",
//!     "trades": [{
//!       "trade_id": "12345",
//!       "product_id": "BTC-USD",
//!       "price": "50000.00",
//!       "size": "0.001",
//!       "side": "BUY",
//!       "time": "2023-01-01T00:00:00.000000Z"
//!     }]
//!   }]
//! }
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use janus_core::{Exchange, MarketDataEvent, Side, Symbol, TradeEvent};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use tracing::{debug, trace, warn};

/// Coinbase exchange adapter
pub struct CoinbaseAdapter {
    ws_url: String,
}

impl CoinbaseAdapter {
    /// Create a new Coinbase adapter
    pub fn new() -> Self {
        Self {
            ws_url: "wss://advanced-trade-ws.coinbase.com".to_string(),
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
    pub fn subscribe_message(&self, symbol: &Symbol, channels: &[CoinbaseChannel]) -> String {
        let product_id = symbol.to_exchange_format(Exchange::Coinbase);
        let channel_names: Vec<String> = channels.iter().map(|c| c.to_string()).collect();

        let msg = CoinbaseSubscribeMessage {
            message_type: "subscribe".to_string(),
            product_ids: vec![product_id],
            channel: channel_names,
            signature: None,
            api_key: None,
            timestamp: None,
        };

        serde_json::to_string(&msg).unwrap_or_else(|_| "{}".to_string())
    }

    /// Build default subscription message (ticker + matches)
    pub fn default_subscribe(&self, symbol: &Symbol) -> String {
        self.subscribe_message(symbol, &[CoinbaseChannel::Ticker, CoinbaseChannel::Matches])
    }

    /// Parse WebSocket message into market data events
    pub fn parse_message(&self, raw: &str) -> Result<Vec<MarketDataEvent>> {
        trace!("Coinbase: Parsing message: {}", raw);

        // Try to parse as a generic message first to determine the channel
        let generic: serde_json::Value =
            serde_json::from_str(raw).context("Failed to parse Coinbase message as JSON")?;

        let channel = generic
            .get("channel")
            .and_then(|c| c.as_str())
            .unwrap_or("");

        match channel {
            "matches" => self.parse_matches(raw),
            "ticker" => self.parse_ticker(raw),
            "level2" => self.parse_level2(raw),
            "subscriptions" => {
                debug!("Coinbase: Subscription confirmed");
                Ok(vec![])
            }
            "heartbeats" => {
                trace!("Coinbase: Heartbeat received");
                Ok(vec![])
            }
            _ => {
                warn!("Coinbase: Unknown channel: {}", channel);
                Ok(vec![])
            }
        }
    }

    /// Parse match/trade messages
    fn parse_matches(&self, raw: &str) -> Result<Vec<MarketDataEvent>> {
        let msg: CoinbaseMatchMessage =
            serde_json::from_str(raw).context("Failed to parse Coinbase match message")?;

        let mut events = Vec::new();

        for event in msg.events {
            for trade in event.trades {
                let symbol = Symbol::from_exchange_format(&trade.product_id, Exchange::Coinbase)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Failed to parse symbol: {}", trade.product_id)
                    })?;

                let timestamp_dt = DateTime::parse_from_rfc3339(&trade.time)
                    .context("Failed to parse trade timestamp")?
                    .with_timezone(&Utc);

                let price =
                    Decimal::from_str(&trade.price).context("Failed to parse trade price")?;
                let quantity =
                    Decimal::from_str(&trade.size).context("Failed to parse trade size")?;

                let side = match trade.side.to_uppercase().as_str() {
                    "BUY" => Side::Buy,
                    "SELL" => Side::Sell,
                    _ => {
                        warn!("Coinbase: Unknown side: {}", trade.side);
                        continue;
                    }
                };

                let event = TradeEvent::new(
                    Exchange::Coinbase,
                    symbol,
                    timestamp_dt.timestamp_micros(),
                    price,
                    quantity,
                    side,
                    trade.trade_id.clone(),
                );

                events.push(MarketDataEvent::Trade(event));
            }
        }

        Ok(events)
    }

    /// Parse ticker messages
    fn parse_ticker(&self, raw: &str) -> Result<Vec<MarketDataEvent>> {
        let msg: CoinbaseTickerMessage =
            serde_json::from_str(raw).context("Failed to parse Coinbase ticker message")?;

        let mut events = Vec::new();

        for event in msg.events {
            for ticker in event.tickers {
                let symbol = Symbol::from_exchange_format(&ticker.product_id, Exchange::Coinbase)
                    .ok_or_else(|| {
                    anyhow::anyhow!("Failed to parse symbol: {}", ticker.product_id)
                })?;

                let timestamp_dt = DateTime::parse_from_rfc3339(&msg.timestamp)
                    .context("Failed to parse ticker timestamp")?
                    .with_timezone(&Utc);

                let last_price =
                    Decimal::from_str(&ticker.price).context("Failed to parse ticker price")?;

                let volume_24h = Decimal::from_str(&ticker.volume_24_h).unwrap_or(Decimal::ZERO);

                let high_24h = ticker
                    .high_24_h
                    .as_ref()
                    .and_then(|h| Decimal::from_str(h).ok());

                let low_24h = ticker
                    .low_24_h
                    .as_ref()
                    .and_then(|l| Decimal::from_str(l).ok());

                let price_change_pct_24h = ticker
                    .price_percent_chg_24_h
                    .as_ref()
                    .and_then(|p| Decimal::from_str(p).ok());

                let ticker_event = janus_core::TickerEvent {
                    exchange: Exchange::Coinbase,
                    symbol,
                    timestamp: timestamp_dt.timestamp_micros(),
                    last_price,
                    best_bid: None,
                    best_ask: None,
                    volume_24h,
                    quote_volume_24h: Decimal::ZERO,
                    price_change_24h: None,
                    price_change_pct_24h,
                    high_24h,
                    low_24h,
                };

                events.push(MarketDataEvent::Ticker(ticker_event));
            }
        }

        Ok(events)
    }

    /// Parse level2 (order book) messages
    fn parse_level2(&self, raw: &str) -> Result<Vec<MarketDataEvent>> {
        let msg: CoinbaseLevel2Message =
            serde_json::from_str(raw).context("Failed to parse Coinbase level2 message")?;

        let mut events = Vec::new();

        for event in msg.events {
            let symbol = Symbol::from_exchange_format(&event.product_id, Exchange::Coinbase)
                .ok_or_else(|| anyhow::anyhow!("Failed to parse symbol: {}", event.product_id))?;

            let timestamp_dt = DateTime::parse_from_rfc3339(&msg.timestamp)
                .context("Failed to parse level2 timestamp")?
                .with_timezone(&Utc);

            // Parse bids
            let mut bids = Vec::new();
            for update in &event.updates {
                if update.side == "bid" {
                    let price = Decimal::from_str(&update.price_level)
                        .context("Failed to parse bid price")?;
                    let quantity = Decimal::from_str(&update.new_quantity)
                        .context("Failed to parse bid quantity")?;

                    // Only include non-zero quantities
                    if quantity > Decimal::ZERO {
                        bids.push(janus_core::PriceLevel::new(price, quantity));
                    }
                }
            }

            // Parse asks
            let mut asks = Vec::new();
            for update in &event.updates {
                if update.side == "offer" {
                    let price = Decimal::from_str(&update.price_level)
                        .context("Failed to parse ask price")?;
                    let quantity = Decimal::from_str(&update.new_quantity)
                        .context("Failed to parse ask quantity")?;

                    // Only include non-zero quantities
                    if quantity > Decimal::ZERO {
                        asks.push(janus_core::PriceLevel::new(price, quantity));
                    }
                }
            }

            // Sort bids descending (highest first), asks ascending (lowest first)
            bids.sort_by(|a, b| b.price.cmp(&a.price));
            asks.sort_by(|a, b| a.price.cmp(&b.price));

            let is_snapshot = event.event_type == "snapshot";

            let orderbook_event = janus_core::OrderBookEvent {
                exchange: Exchange::Coinbase,
                symbol,
                timestamp: timestamp_dt.timestamp_micros(),
                sequence: msg.sequence_num,
                is_snapshot,
                bids,
                asks,
            };

            events.push(MarketDataEvent::OrderBook(orderbook_event));
        }

        Ok(events)
    }
}

impl Default for CoinbaseAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Coinbase WebSocket channels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoinbaseChannel {
    Ticker,
    TickerBatch,
    Level2,
    Matches,
    Heartbeats,
}

impl std::fmt::Display for CoinbaseChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoinbaseChannel::Ticker => write!(f, "ticker"),
            CoinbaseChannel::TickerBatch => write!(f, "ticker_batch"),
            CoinbaseChannel::Level2 => write!(f, "level2"),
            CoinbaseChannel::Matches => write!(f, "matches"),
            CoinbaseChannel::Heartbeats => write!(f, "heartbeats"),
        }
    }
}

/// Coinbase subscription message
#[derive(Debug, Serialize, Deserialize)]
struct CoinbaseSubscribeMessage {
    #[serde(rename = "type")]
    message_type: String,
    product_ids: Vec<String>,
    channel: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamp: Option<String>,
}

/// Coinbase match/trade message
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CoinbaseMatchMessage {
    channel: String,
    #[serde(default)]
    client_id: String,
    timestamp: String,
    sequence_num: u64,
    events: Vec<CoinbaseMatchEvent>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CoinbaseMatchEvent {
    #[serde(rename = "type")]
    event_type: String,
    trades: Vec<CoinbaseTrade>,
}

#[derive(Debug, Deserialize)]
struct CoinbaseTrade {
    trade_id: String,
    product_id: String,
    price: String,
    size: String,
    side: String,
    time: String,
}

/// Coinbase ticker message
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CoinbaseTickerMessage {
    channel: String,
    #[serde(default)]
    client_id: String,
    timestamp: String,
    sequence_num: u64,
    events: Vec<CoinbaseTickerEvent>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CoinbaseTickerEvent {
    #[serde(rename = "type")]
    event_type: String,
    tickers: Vec<CoinbaseTicker>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CoinbaseTicker {
    #[serde(rename = "type")]
    ticker_type: String,
    product_id: String,
    price: String,
    volume_24_h: String,
    low_24_h: Option<String>,
    high_24_h: Option<String>,
    #[serde(default)]
    low_52_w: Option<String>,
    #[serde(default)]
    high_52_w: Option<String>,
    price_percent_chg_24_h: Option<String>,
}

/// Coinbase Level2 order book message
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CoinbaseLevel2Message {
    channel: String,
    #[serde(default)]
    client_id: String,
    timestamp: String,
    sequence_num: u64,
    events: Vec<CoinbaseLevel2Event>,
}

#[derive(Debug, Deserialize)]
struct CoinbaseLevel2Event {
    #[serde(rename = "type")]
    event_type: String,
    product_id: String,
    updates: Vec<CoinbaseLevel2Update>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CoinbaseLevel2Update {
    side: String, // "bid" or "offer"
    #[serde(rename = "event_time")]
    event_time: String,
    price_level: String,
    new_quantity: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscribe_message() {
        let adapter = CoinbaseAdapter::new();
        let symbol = Symbol::new("BTC", "USD");
        let msg = adapter.default_subscribe(&symbol);

        assert!(msg.contains("subscribe"));
        assert!(msg.contains("BTC-USD"));
        assert!(msg.contains("ticker"));
        assert!(msg.contains("matches"));
    }

    #[test]
    fn test_parse_trade() {
        let adapter = CoinbaseAdapter::new();
        let raw = r#"{
            "channel": "matches",
            "client_id": "",
            "timestamp": "2023-01-01T00:00:00.000000Z",
            "sequence_num": 0,
            "events": [{
                "type": "match",
                "trades": [{
                    "trade_id": "12345",
                    "product_id": "BTC-USD",
                    "price": "50000.00",
                    "size": "0.001",
                    "side": "BUY",
                    "time": "2023-01-01T00:00:00.000000Z"
                }]
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            MarketDataEvent::Trade(trade) => {
                assert_eq!(trade.exchange, Exchange::Coinbase);
                assert_eq!(trade.symbol.base, "BTC");
                assert_eq!(trade.symbol.quote, "USD");
                assert_eq!(trade.price, Decimal::from_str("50000.00").unwrap());
                assert_eq!(trade.quantity, Decimal::from_str("0.001").unwrap());
                assert_eq!(trade.side, Side::Buy);
                assert_eq!(trade.trade_id, "12345");
            }
            _ => panic!("Expected TradeEvent"),
        }
    }

    #[test]
    fn test_parse_level2_snapshot() {
        let adapter = CoinbaseAdapter::new();
        let raw = r#"{
            "channel": "level2",
            "client_id": "",
            "timestamp": "2023-01-01T00:00:00.000000Z",
            "sequence_num": 1000,
            "events": [{
                "type": "snapshot",
                "product_id": "BTC-USD",
                "updates": [
                    {
                        "side": "bid",
                        "event_time": "2023-01-01T00:00:00.000000Z",
                        "price_level": "50000.00",
                        "new_quantity": "1.5"
                    },
                    {
                        "side": "bid",
                        "event_time": "2023-01-01T00:00:00.000000Z",
                        "price_level": "49999.00",
                        "new_quantity": "2.0"
                    },
                    {
                        "side": "offer",
                        "event_time": "2023-01-01T00:00:00.000000Z",
                        "price_level": "50001.00",
                        "new_quantity": "1.0"
                    },
                    {
                        "side": "offer",
                        "event_time": "2023-01-01T00:00:00.000000Z",
                        "price_level": "50002.00",
                        "new_quantity": "3.0"
                    }
                ]
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        if let MarketDataEvent::OrderBook(book) = &events[0] {
            assert_eq!(book.exchange, Exchange::Coinbase);
            assert_eq!(book.symbol.base, "BTC");
            assert_eq!(book.symbol.quote, "USD");
            assert!(book.is_snapshot);
            assert_eq!(book.bids.len(), 2);
            assert_eq!(book.asks.len(), 2);

            // Verify bids are sorted descending
            assert!(book.bids[0].price > book.bids[1].price);
            assert_eq!(book.bids[0].price, Decimal::from_str("50000.00").unwrap());

            // Verify asks are sorted ascending
            assert!(book.asks[0].price < book.asks[1].price);
            assert_eq!(book.asks[0].price, Decimal::from_str("50001.00").unwrap());

            // Test mid price calculation
            let mid = book.mid_price().unwrap();
            assert_eq!(mid, Decimal::from_str("50000.50").unwrap());
        } else {
            panic!("Expected OrderBook event");
        }
    }

    #[test]
    fn test_parse_level2_update() {
        let adapter = CoinbaseAdapter::new();
        let raw = r#"{
            "channel": "level2",
            "client_id": "",
            "timestamp": "2023-01-01T00:00:01.000000Z",
            "sequence_num": 1001,
            "events": [{
                "type": "update",
                "product_id": "ETH-USD",
                "updates": [
                    {
                        "side": "bid",
                        "event_time": "2023-01-01T00:00:01.000000Z",
                        "price_level": "3000.00",
                        "new_quantity": "0.0"
                    },
                    {
                        "side": "offer",
                        "event_time": "2023-01-01T00:00:01.000000Z",
                        "price_level": "3001.00",
                        "new_quantity": "5.5"
                    }
                ]
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        if let MarketDataEvent::OrderBook(book) = &events[0] {
            assert_eq!(book.symbol.base, "ETH");
            assert!(!book.is_snapshot);
            // Zero quantity updates should be filtered out
            assert_eq!(book.bids.len(), 0);
            assert_eq!(book.asks.len(), 1);
            assert_eq!(book.asks[0].quantity, Decimal::from_str("5.5").unwrap());
        } else {
            panic!("Expected OrderBook event");
        }
    }

    #[test]
    fn test_parse_ticker() {
        let adapter = CoinbaseAdapter::new();
        let raw = r#"{
            "channel": "ticker",
            "client_id": "",
            "timestamp": "2023-01-01T00:00:00.000000Z",
            "sequence_num": 0,
            "events": [{
                "type": "ticker",
                "tickers": [{
                    "type": "ticker",
                    "product_id": "BTC-USD",
                    "price": "50000.00",
                    "volume_24_h": "1000.5",
                    "low_24_h": "49000.00",
                    "high_24_h": "51000.00",
                    "price_percent_chg_24_h": "2.5"
                }]
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            MarketDataEvent::Ticker(ticker) => {
                assert_eq!(ticker.exchange, Exchange::Coinbase);
                assert_eq!(ticker.symbol.base, "BTC");
                assert_eq!(ticker.last_price, Decimal::from_str("50000.00").unwrap());
                assert_eq!(ticker.volume_24h, Decimal::from_str("1000.5").unwrap());
            }
            _ => panic!("Expected TickerEvent"),
        }
    }
}
