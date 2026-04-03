//! Binance Exchange Connector
//!
//! Implements WebSocket connection to Binance for real-time trade and kline data.
//!
//! ## API Documentation:
//! - WebSocket Streams: https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams
//! - Trade Streams: https://binance-docs.github.io/apidocs/spot/en/#trade-streams
//! - Kline Streams: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-streams
//!
//! ## Stream Format:
//! - Endpoint: wss://stream.binance.com:9443/ws
//! - Trade Stream: <symbol>@trade (e.g., btcusdt@trade)
//! - Kline Stream: <symbol>@kline_<interval> (e.g., btcusdt@kline_1m)
//! - Combined Stream: wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@kline_1m
//!
//! ## Message Format (Trade):
//! ```json
//! {
//!   "e": "trade",       // Event type
//!   "E": 123456789,     // Event time
//!   "s": "BTCUSD",     // Symbol
//!   "t": 12345,         // Trade ID
//!   "p": "50000.00",    // Price (string to avoid precision loss)
//!   "q": "0.001",       // Quantity
//!   "T": 123456785,     // Trade time
//!   "m": true,          // Is buyer the market maker?
//!   "M": true           // Ignore
//! }
//! ```
//!
//! ## Message Format (Kline):
//! ```json
//! {
//!   "e": "kline",
//!   "E": 123456789,     // Event time
//!   "s": "BTCUSD",     // Symbol
//!   "k": {
//!     "t": 123400000,   // Kline start time
//!     "T": 123460000,   // Kline close time
//!     "s": "BTCUSD",   // Symbol
//!     "i": "1m",        // Interval
//!     "o": "50000.00",  // Open price
//!     "c": "50100.00",  // Close price
//!     "h": "50200.00",  // High price
//!     "l": "49900.00",  // Low price
//!     "v": "100.5",     // Volume
//!     "x": false        // Is this kline closed?
//!   }
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::actors::{
    CandleData, DataMessage, ExchangeConnector, TradeData, TradeSide, WebSocketConfig,
};

/// Binance stream type configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Type of data stream to subscribe to
#[derive(Default)]
pub enum BinanceStreamType {
    /// Trade stream only (for gap detection)
    TradeOnly,
    /// Kline stream only (for indicator calculations)
    KlineOnly,
    /// Combined trade and kline streams
    #[default]
    TradeAndKline,
}

/// Binance connector implementation
pub struct BinanceConnector {
    ws_url: String,
    stream_type: BinanceStreamType,
    kline_intervals: Vec<String>,
}

impl BinanceConnector {
    /// Create a new Binance connector with default settings (trade + kline)
    pub fn new(ws_url: String) -> Self {
        Self {
            ws_url,
            stream_type: BinanceStreamType::TradeAndKline,
            kline_intervals: vec!["1m".to_string()],
        }
    }

    /// Create a new Binance connector with specific stream type
    pub fn with_stream_type(ws_url: String, stream_type: BinanceStreamType) -> Self {
        Self {
            ws_url,
            stream_type,
            kline_intervals: vec!["1m".to_string()],
        }
    }

    /// Set the kline interval (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
    pub fn with_kline_interval(mut self, interval: &str) -> Self {
        self.kline_intervals = vec![interval.to_string()];
        self
    }

    /// Set multiple kline intervals for multi-timeframe analysis
    pub fn with_kline_intervals(mut self, intervals: Vec<String>) -> Self {
        self.kline_intervals = intervals;
        self
    }

    /// Parse a Binance trade message
    fn parse_trade_message(&self, msg: &BinanceTrade) -> Result<TradeData> {
        let price: f64 = msg.p.parse().context("Failed to parse price")?;
        let amount: f64 = msg.q.parse().context("Failed to parse quantity")?;

        // If 'm' (is buyer market maker) is true, the seller is aggressive - SELL
        // If false, the buyer is aggressive - BUY
        let side = if msg.m {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        };

        let receipt_ts = chrono::Utc::now().timestamp_millis();

        Ok(TradeData {
            symbol: msg.s.clone(),
            exchange: "binance".to_string(),
            side,
            price,
            amount,
            exchange_ts: msg.T,
            receipt_ts,
            trade_id: msg.t.to_string(),
        })
    }

    /// Parse a Binance kline message
    fn parse_kline_message(&self, msg: &BinanceKline) -> Result<Option<CandleData>> {
        let kline = &msg.k;

        // Only emit candle data when the kline is closed
        if !kline.x {
            debug!(
                "Binance: Kline {} {} not closed yet, skipping",
                kline.s, kline.i
            );
            return Ok(None);
        }

        let open: f64 = kline.o.parse().context("Failed to parse open price")?;
        let high: f64 = kline.h.parse().context("Failed to parse high price")?;
        let low: f64 = kline.l.parse().context("Failed to parse low price")?;
        let close: f64 = kline.c.parse().context("Failed to parse close price")?;
        let volume: f64 = kline.v.parse().context("Failed to parse volume")?;

        info!(
            "Binance: Closed kline {} {} O={:.2} H={:.2} L={:.2} C={:.2} V={:.4}",
            kline.s, kline.i, open, high, low, close, volume
        );

        Ok(Some(CandleData {
            symbol: kline.s.clone(),
            exchange: "binance".to_string(),
            open_time: kline.t,
            close_time: kline.T,
            open,
            high,
            low,
            close,
            volume,
            interval: kline.i.clone(),
        }))
    }

    /// Build the stream URL for a symbol
    fn build_stream_url(&self, symbol: &str) -> String {
        let symbol_lower = symbol.to_lowercase();

        match self.stream_type {
            BinanceStreamType::TradeOnly => {
                format!("{}/{}@trade", self.ws_url, symbol_lower)
            }
            BinanceStreamType::KlineOnly => {
                // Support multiple kline intervals
                let base_url = self.ws_url.replace("/ws", "/stream");
                let kline_streams: Vec<String> = self
                    .kline_intervals
                    .iter()
                    .map(|interval| format!("{}@kline_{}", symbol_lower, interval))
                    .collect();
                format!("{}?streams={}", base_url, kline_streams.join("/"))
            }
            BinanceStreamType::TradeAndKline => {
                // Use combined streams endpoint with multiple kline intervals
                let base_url = self.ws_url.replace("/ws", "/stream");
                let mut streams = vec![format!("{}@trade", symbol_lower)];

                // Add all kline intervals
                for interval in &self.kline_intervals {
                    streams.push(format!("{}@kline_{}", symbol_lower, interval));
                }

                format!("{}?streams={}", base_url, streams.join("/"))
            }
        }
    }
}

impl ExchangeConnector for BinanceConnector {
    fn exchange_name(&self) -> &str {
        "binance"
    }

    fn ws_url(&self) -> &str {
        &self.ws_url
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        let stream_url = self.build_stream_url(symbol);

        info!(
            "Binance: Building WebSocket config for {} with stream type {:?}",
            symbol, self.stream_type
        );

        WebSocketConfig {
            url: stream_url,
            exchange: self.exchange_name().to_string(),
            symbol: symbol.to_string(),
            subscription_msg: None,
            ping_interval_secs: 180,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }

    fn subscription_message(&self, _symbol: &str) -> Option<String> {
        None
    }

    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>> {
        let mut messages = Vec::new();

        // For combined streams, the message is wrapped in { "stream": "...", "data": {...} }
        if let Ok(wrapper) = serde_json::from_str::<BinanceCombinedStreamWrapper>(raw) {
            // Parse the inner data based on stream type
            if wrapper.stream.contains("@trade") {
                if let Ok(trade) = serde_json::from_value::<BinanceTrade>(wrapper.data.clone())
                    && trade.e == "trade"
                {
                    let trade_data = self
                        .parse_trade_message(&trade)
                        .context("Failed to parse Binance trade from combined stream")?;
                    messages.push(DataMessage::Trade(trade_data));
                }
            } else if wrapper.stream.contains("@kline")
                && let Ok(kline) = serde_json::from_value::<BinanceKline>(wrapper.data.clone())
                && kline.e == "kline"
                && let Some(candle_data) = self
                    .parse_kline_message(&kline)
                    .context("Failed to parse Binance kline from combined stream")?
            {
                messages.push(DataMessage::Candle(candle_data));
            }
            return Ok(messages);
        }

        // Try to parse as a direct trade message
        if let Ok(trade) = serde_json::from_str::<BinanceTrade>(raw)
            && trade.e == "trade"
        {
            let trade_data = self
                .parse_trade_message(&trade)
                .context("Failed to parse Binance trade")?;
            messages.push(DataMessage::Trade(trade_data));
            return Ok(messages);
        }

        // Try to parse as a kline message
        if let Ok(kline) = serde_json::from_str::<BinanceKline>(raw)
            && kline.e == "kline"
        {
            if let Some(candle_data) = self
                .parse_kline_message(&kline)
                .context("Failed to parse Binance kline")?
            {
                messages.push(DataMessage::Candle(candle_data));
            }
            return Ok(messages);
        }

        // Log unrecognized message types
        if !raw.contains("\"result\":null") && !raw.contains("\"id\":") {
            debug!(
                "Binance: Unrecognized message type: {}",
                if raw.len() > 200 { &raw[..200] } else { raw }
            );
        }

        Ok(messages)
    }
}

/// Wrapper for combined stream messages
#[derive(Debug, Clone, Deserialize)]
struct BinanceCombinedStreamWrapper {
    /// Stream name (e.g., "btcusdt@trade" or "btcusdt@kline_1m")
    stream: String,
    /// Inner data
    data: serde_json::Value,
}

/// Binance trade message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
#[allow(non_snake_case)]
struct BinanceTrade {
    /// Event type (should be "trade")
    e: String,

    /// Event time (milliseconds)
    #[serde(rename = "E")]
    _event_time: i64,

    /// Symbol
    s: String,

    /// Trade ID
    t: u64,

    /// Price (as string to preserve precision)
    p: String,

    /// Quantity (as string to preserve precision)
    q: String,

    /// Buyer order ID (optional)
    #[serde(rename = "b", default)]
    _buyer_order_id: Option<u64>,

    /// Seller order ID (optional)
    #[serde(rename = "a", default)]
    _seller_order_id: Option<u64>,

    /// Trade time (milliseconds)
    #[serde(rename = "T")]
    T: i64,

    /// Is buyer the market maker?
    m: bool,

    /// Ignore
    #[serde(rename = "M", default)]
    _ignore: bool,
}

/// Binance kline/candlestick message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
struct BinanceKline {
    /// Event type (should be "kline")
    e: String,

    /// Event time
    #[serde(rename = "E")]
    event_time: i64,

    /// Symbol
    s: String,

    /// Kline data
    k: BinanceKlineData,
}

/// Inner kline data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
#[allow(non_snake_case)]
struct BinanceKlineData {
    /// Kline start time
    t: i64,

    /// Kline close time
    T: i64,

    /// Symbol
    s: String,

    /// Interval (e.g., "1m", "5m", "1h")
    i: String,

    /// First trade ID
    #[serde(default)]
    f: i64,

    /// Last trade ID
    #[serde(rename = "L", default)]
    last_trade_id: i64,

    /// Open price
    o: String,

    /// Close price
    c: String,

    /// High price
    h: String,

    /// Low price
    l: String,

    /// Base asset volume
    v: String,

    /// Number of trades
    n: i64,

    /// Is this kline closed?
    x: bool,

    /// Quote asset volume
    q: String,

    /// Taker buy base asset volume
    #[serde(rename = "V")]
    taker_buy_volume: String,

    /// Taker buy quote asset volume
    #[serde(rename = "Q")]
    taker_buy_quote_volume: String,

    /// Ignore
    #[serde(rename = "B")]
    _ignore: String,
}

/// Binance aggregate trade message (lower bandwidth alternative)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
struct BinanceAggTrade {
    /// Event type (should be "aggTrade")
    e: String,

    /// Event time
    #[serde(rename = "E")]
    event_time: i64,

    /// Symbol
    s: String,

    /// Aggregate trade ID
    a: u64,

    /// Price
    p: String,

    /// Quantity
    q: String,

    /// First trade ID
    f: u64,

    /// Last trade ID
    l: u64,

    /// Trade time
    #[serde(rename = "T")]
    trade_time: i64,

    /// Is buyer the market maker?
    m: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binance_connector_creation() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());
        assert_eq!(connector.exchange_name(), "binance");
        assert_eq!(connector.stream_type, BinanceStreamType::TradeAndKline);
    }

    #[test]
    fn test_stream_type_trade_only() {
        let connector = BinanceConnector::with_stream_type(
            "wss://stream.binance.com:9443/ws".to_string(),
            BinanceStreamType::TradeOnly,
        );
        let url = connector.build_stream_url("BTCUSDT");
        assert_eq!(url, "wss://stream.binance.com:9443/ws/btcusdt@trade");
    }

    #[test]
    fn test_stream_type_kline_only() {
        let connector = BinanceConnector::with_stream_type(
            "wss://stream.binance.com:9443/ws".to_string(),
            BinanceStreamType::KlineOnly,
        );
        let url = connector.build_stream_url("BTCUSDT");
        assert_eq!(
            url,
            "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m"
        );
    }

    #[test]
    fn test_stream_type_combined() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());
        let url = connector.build_stream_url("BTCUSDT");
        assert_eq!(
            url,
            "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@kline_1m"
        );
    }

    #[test]
    fn test_custom_kline_interval() {
        let connector = BinanceConnector::with_stream_type(
            "wss://stream.binance.com:9443/ws".to_string(),
            BinanceStreamType::KlineOnly,
        )
        .with_kline_interval("5m");

        let url = connector.build_stream_url("ETHUSDT");
        assert_eq!(
            url,
            "wss://stream.binance.com:9443/stream?streams=ethusdt@kline_5m"
        );
    }

    #[test]
    fn test_build_ws_config() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());
        let config = connector.build_ws_config("BTCUSDT");

        assert_eq!(config.exchange, "binance");
        assert_eq!(config.symbol, "BTCUSDT");
        assert!(config.url.contains("btcusdt@trade"));
        assert!(config.url.contains("btcusdt@kline_1m"));
    }

    #[test]
    fn test_parse_trade_message() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());

        let raw_msg = r#"{
            "e": "trade",
            "E": 1672531200000,
            "s": "BTCUSD",
            "t": 12345,
            "p": "50000.00",
            "q": "0.001",
            "b": 88,
            "a": 50,
            "T": 1672531200000,
            "m": true,
            "M": true
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();
        assert_eq!(messages.len(), 1);

        if let DataMessage::Trade(trade) = &messages[0] {
            assert_eq!(trade.symbol, "BTCUSD");
            assert_eq!(trade.exchange, "binance");
            assert_eq!(trade.price, 50000.00);
            assert_eq!(trade.amount, 0.001);
            assert_eq!(trade.side, TradeSide::Sell);
            assert_eq!(trade.trade_id, "12345");
        } else {
            panic!("Expected Trade message");
        }
    }

    #[test]
    fn test_parse_kline_message_closed() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());

        let raw_msg = r#"{
            "e": "kline",
            "E": 1672531260000,
            "s": "BTCUSD",
            "k": {
                "t": 1672531200000,
                "T": 1672531259999,
                "s": "BTCUSD",
                "i": "1m",
                "f": 100,
                "L": 200,
                "o": "50000.00",
                "c": "50100.00",
                "h": "50200.00",
                "l": "49900.00",
                "v": "10.5",
                "n": 100,
                "x": true,
                "q": "525000.00",
                "V": "5.0",
                "Q": "250000.00",
                "B": "0"
            }
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();
        assert_eq!(messages.len(), 1);

        if let DataMessage::Candle(candle) = &messages[0] {
            assert_eq!(candle.symbol, "BTCUSD");
            assert_eq!(candle.exchange, "binance");
            assert_eq!(candle.interval, "1m");
            assert_eq!(candle.open, 50000.00);
            assert_eq!(candle.high, 50200.00);
            assert_eq!(candle.low, 49900.00);
            assert_eq!(candle.close, 50100.00);
            assert_eq!(candle.volume, 10.5);
            assert_eq!(candle.open_time, 1672531200000);
            assert_eq!(candle.close_time, 1672531259999);
        } else {
            panic!("Expected Candle message");
        }
    }

    #[test]
    fn test_parse_kline_message_not_closed() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());

        let raw_msg = r#"{
            "e": "kline",
            "E": 1672531230000,
            "s": "BTCUSD",
            "k": {
                "t": 1672531200000,
                "T": 1672531259999,
                "s": "BTCUSD",
                "i": "1m",
                "f": 100,
                "L": 150,
                "o": "50000.00",
                "c": "50050.00",
                "h": "50100.00",
                "l": "49950.00",
                "v": "5.0",
                "n": 50,
                "x": false,
                "q": "250000.00",
                "V": "2.5",
                "Q": "125000.00",
                "B": "0"
            }
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();
        // Should return empty - kline not closed yet
        assert_eq!(messages.len(), 0);
    }

    #[test]
    fn test_parse_combined_stream_trade() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());

        let raw_msg = r#"{
            "stream": "btcusdt@trade",
            "data": {
                "e": "trade",
                "E": 1672531200000,
                "s": "BTCUSD",
                "t": 12345,
                "p": "50000.00",
                "q": "0.001",
                "T": 1672531200000,
                "m": false,
                "M": true
            }
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();
        assert_eq!(messages.len(), 1);

        if let DataMessage::Trade(trade) = &messages[0] {
            assert_eq!(trade.symbol, "BTCUSD");
            assert_eq!(trade.side, TradeSide::Buy);
        } else {
            panic!("Expected Trade message");
        }
    }

    #[test]
    fn test_parse_combined_stream_kline() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());

        let raw_msg = r#"{
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline",
                "E": 1672531260000,
                "s": "BTCUSD",
                "k": {
                    "t": 1672531200000,
                    "T": 1672531259999,
                    "s": "BTCUSD",
                    "i": "1m",
                    "f": 100,
                    "L": 200,
                    "o": "50000.00",
                    "c": "50100.00",
                    "h": "50200.00",
                    "l": "49900.00",
                    "v": "10.5",
                    "n": 100,
                    "x": true,
                    "q": "525000.00",
                    "V": "5.0",
                    "Q": "250000.00",
                    "B": "0"
                }
            }
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();
        assert_eq!(messages.len(), 1);

        if let DataMessage::Candle(candle) = &messages[0] {
            assert_eq!(candle.symbol, "BTCUSD");
            assert_eq!(candle.interval, "1m");
            assert_eq!(candle.close, 50100.00);
        } else {
            panic!("Expected Candle message");
        }
    }

    #[test]
    fn test_parse_buy_trade() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());

        let raw_msg = r#"{
            "e": "trade",
            "E": 1672531200000,
            "s": "ETHUSDT",
            "t": 67890,
            "p": "3000.50",
            "q": "0.5",
            "b": 100,
            "a": 200,
            "T": 1672531200000,
            "m": false,
            "M": false
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();

        if let DataMessage::Trade(trade) = &messages[0] {
            assert_eq!(trade.side, TradeSide::Buy);
            assert_eq!(trade.symbol, "ETHUSDT");
            assert_eq!(trade.price, 3000.50);
        } else {
            panic!("Expected Trade message");
        }
    }

    #[test]
    fn test_parse_invalid_message() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());

        let raw_msg = r#"{"invalid": "json", "structure": true}"#;
        let messages = connector.parse_message(raw_msg).unwrap();

        assert_eq!(messages.len(), 0);
    }

    #[test]
    fn test_parse_subscription_response() {
        let connector = BinanceConnector::new("wss://stream.binance.com:9443/ws".to_string());

        // Binance sends this response after subscription
        let raw_msg = r#"{"result":null,"id":1}"#;
        let messages = connector.parse_message(raw_msg).unwrap();

        assert_eq!(messages.len(), 0);
    }
}
