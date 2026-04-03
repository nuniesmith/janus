//! Bybit Exchange Connector (V5 API)
//!
//! Implements WebSocket connection to Bybit's V5 Unified API for real-time trade data.
//!
//! ## API Documentation:
//! - WebSocket API: https://bybit-exchange.github.io/docs/v5/ws/connect
//! - Public Trade Stream: https://bybit-exchange.github.io/docs/v5/websocket/public/trade
//!
//! ## Stream Format:
//! - Endpoint (Spot): wss://stream.bybit.com/v5/public/spot
//! - Endpoint (Linear): wss://stream.bybit.com/v5/public/linear
//! - Topic: publicTrade.{symbol} (e.g., publicTrade.BTCUSD)
//!
//! ## Subscription Message:
//! ```json
//! {
//!   "op": "subscribe",
//!   "args": ["publicTrade.BTCUSD"]
//! }
//! ```
//!
//! ## Message Format (Trade):
//! ```json
//! {
//!   "topic": "publicTrade.BTCUSD",
//!   "type": "snapshot",
//!   "ts": 1672531200000,
//!   "data": [
//!     {
//!       "T": 1672531200000,  // Trade timestamp
//!       "s": "BTCUSD",      // Symbol
//!       "S": "Buy",          // Side: Buy or Sell
//!       "v": "0.001",        // Volume
//!       "p": "50000.00",     // Price
//!       "L": "ZeroPlusTick", // Price change direction
//!       "i": "12345",        // Trade ID
//!       "BT": false          // Is block trade
//!     }
//!   ]
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::actors::{DataMessage, ExchangeConnector, TradeData, TradeSide, WebSocketConfig};

/// Bybit connector implementation
pub struct BybitConnector {
    ws_url: String,
}

impl BybitConnector {
    /// Create a new Bybit connector
    pub fn new(ws_url: String) -> Self {
        Self { ws_url }
    }

    /// Parse a Bybit trade message wrapper
    fn parse_trade_wrapper(&self, wrapper: &BybitTradeWrapper) -> Result<Vec<TradeData>> {
        let receipt_ts = chrono::Utc::now().timestamp_millis();
        let mut trades = Vec::new();

        for trade in &wrapper.data {
            // Parse side
            let side = match trade.S.as_str() {
                "Buy" => TradeSide::Buy,
                "Sell" => TradeSide::Sell,
                _ => {
                    warn!("Bybit: Unknown trade side: {}", trade.S);
                    continue;
                }
            };

            // Parse price and volume
            let price: f64 = trade.p.parse().context("Failed to parse price")?;

            let amount: f64 = trade.v.parse().context("Failed to parse volume")?;

            trades.push(TradeData {
                symbol: trade.s.clone(),
                exchange: "bybit".to_string(),
                side,
                price,
                amount,
                exchange_ts: trade.T,
                receipt_ts,
                trade_id: trade.i.clone(),
            });
        }

        Ok(trades)
    }
}

impl ExchangeConnector for BybitConnector {
    fn exchange_name(&self) -> &str {
        "bybit"
    }

    fn ws_url(&self) -> &str {
        &self.ws_url
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        // Bybit requires explicit subscription via JSON message
        let subscription = self.subscription_message(symbol);

        WebSocketConfig {
            url: self.ws_url.clone(),
            exchange: self.exchange_name().to_string(),
            symbol: symbol.to_string(),
            subscription_msg: subscription,
            ping_interval_secs: 20, // Bybit expects ping every 20s
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }

    fn subscription_message(&self, symbol: &str) -> Option<String> {
        // Bybit V5 subscription format
        let topic = format!("publicTrade.{}", symbol);
        let sub_msg = BybitSubscription {
            op: "subscribe".to_string(),
            args: vec![topic],
        };

        serde_json::to_string(&sub_msg).ok()
    }

    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>> {
        // Try to parse as subscription response first
        if let Ok(response) = serde_json::from_str::<BybitSubscriptionResponse>(raw) {
            if response.success {
                debug!("Bybit: Subscription successful: {:?}", response.op);
                return Ok(vec![]);
            } else {
                warn!("Bybit: Subscription failed: {:?}", response.ret_msg);
                return Ok(vec![]);
            }
        }

        // Try to parse as trade message
        match serde_json::from_str::<BybitTradeWrapper>(raw) {
            Ok(wrapper) => {
                // Verify this is a trade topic
                if !wrapper.topic.starts_with("publicTrade.") {
                    debug!("Bybit: Ignoring non-trade topic: {}", wrapper.topic);
                    return Ok(vec![]);
                }

                let trades = self
                    .parse_trade_wrapper(&wrapper)
                    .context("Failed to parse Bybit trade wrapper")?;

                Ok(trades.into_iter().map(DataMessage::Trade).collect())
            }
            Err(e) => {
                // Check if it's a ping/pong message
                if raw.contains("\"op\":\"pong\"") || raw.contains("\"op\":\"ping\"") {
                    debug!("Bybit: Received ping/pong");
                    return Ok(vec![]);
                }

                warn!(
                    "Bybit: Failed to parse message: {} - Raw: {}",
                    e,
                    if raw.len() > 200 { &raw[..200] } else { raw }
                );
                Ok(vec![])
            }
        }
    }
}

/// Bybit subscription message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BybitSubscription {
    op: String,
    args: Vec<String>,
}

/// Bybit subscription response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BybitSubscriptionResponse {
    success: bool,
    ret_msg: Option<String>,
    op: String,
    #[serde(rename = "conn_id")]
    conn_id: Option<String>,
}

/// Bybit trade message wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BybitTradeWrapper {
    /// Topic name (e.g., "publicTrade.BTCUSD")
    topic: String,

    /// Message type (snapshot or delta)
    #[serde(rename = "type")]
    msg_type: String,

    /// Timestamp
    ts: i64,

    /// Array of trades
    data: Vec<BybitTrade>,
}

/// Bybit individual trade
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
struct BybitTrade {
    /// Trade timestamp
    #[serde(rename = "T")]
    T: i64,

    /// Symbol
    s: String,

    /// Side: "Buy" or "Sell"
    #[serde(rename = "S")]
    S: String,

    /// Volume/Amount (string to preserve precision)
    v: String,

    /// Price (string to preserve precision)
    p: String,

    /// Price change direction ("PlusTick", "ZeroPlusTick", "MinusTick", "ZeroMinusTick")
    #[serde(rename = "L")]
    _direction: String,

    /// Trade ID
    i: String,

    /// Is block trade
    #[serde(rename = "BT")]
    _is_block_trade: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bybit_connector_creation() {
        let connector = BybitConnector::new("wss://stream.bybit.com/v5/public/spot".to_string());
        assert_eq!(connector.exchange_name(), "bybit");
    }

    #[test]
    fn test_subscription_message() {
        let connector = BybitConnector::new("wss://stream.bybit.com/v5/public/spot".to_string());
        let sub_msg = connector.subscription_message("BTCUSD");

        assert!(sub_msg.is_some());
        let msg = sub_msg.unwrap();
        assert!(msg.contains("\"op\":\"subscribe\""));
        assert!(msg.contains("publicTrade.BTCUSD"));
    }

    #[test]
    fn test_build_ws_config() {
        let connector = BybitConnector::new("wss://stream.bybit.com/v5/public/spot".to_string());
        let config = connector.build_ws_config("BTCUSD");

        assert_eq!(config.exchange, "bybit");
        assert_eq!(config.symbol, "BTCUSD");
        assert_eq!(config.url, "wss://stream.bybit.com/v5/public/spot");
        assert!(config.subscription_msg.is_some());
        assert_eq!(config.ping_interval_secs, 20);
    }

    #[test]
    fn test_parse_trade_message() {
        let connector = BybitConnector::new("wss://stream.bybit.com/v5/public/spot".to_string());

        let raw_msg = r#"{
            "topic": "publicTrade.BTCUSD",
            "type": "snapshot",
            "ts": 1672531200000,
            "data": [
                {
                    "T": 1672531200000,
                    "s": "BTCUSD",
                    "S": "Buy",
                    "v": "0.001",
                    "p": "50000.00",
                    "L": "ZeroPlusTick",
                    "i": "12345",
                    "BT": false
                }
            ]
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();
        assert_eq!(messages.len(), 1);

        if let DataMessage::Trade(trade) = &messages[0] {
            assert_eq!(trade.symbol, "BTCUSD");
            assert_eq!(trade.exchange, "bybit");
            assert_eq!(trade.price, 50000.00);
            assert_eq!(trade.amount, 0.001);
            assert_eq!(trade.side, TradeSide::Buy);
            assert_eq!(trade.trade_id, "12345");
        } else {
            panic!("Expected Trade message");
        }
    }

    #[test]
    fn test_parse_multiple_trades() {
        let connector = BybitConnector::new("wss://stream.bybit.com/v5/public/spot".to_string());

        let raw_msg = r#"{
            "topic": "publicTrade.ETHUSDT",
            "type": "snapshot",
            "ts": 1672531200000,
            "data": [
                {
                    "T": 1672531200000,
                    "s": "ETHUSDT",
                    "S": "Sell",
                    "v": "0.5",
                    "p": "3000.00",
                    "L": "MinusTick",
                    "i": "11111",
                    "BT": false
                },
                {
                    "T": 1672531201000,
                    "s": "ETHUSDT",
                    "S": "Buy",
                    "v": "1.0",
                    "p": "3001.00",
                    "L": "PlusTick",
                    "i": "22222",
                    "BT": false
                }
            ]
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();
        assert_eq!(messages.len(), 2);

        if let DataMessage::Trade(trade1) = &messages[0] {
            assert_eq!(trade1.side, TradeSide::Sell);
            assert_eq!(trade1.trade_id, "11111");
        }

        if let DataMessage::Trade(trade2) = &messages[1] {
            assert_eq!(trade2.side, TradeSide::Buy);
            assert_eq!(trade2.trade_id, "22222");
        }
    }

    #[test]
    fn test_parse_subscription_response() {
        let connector = BybitConnector::new("wss://stream.bybit.com/v5/public/spot".to_string());

        let raw_msg = r#"{
            "success": true,
            "ret_msg": "",
            "op": "subscribe",
            "conn_id": "abc123"
        }"#;

        let messages = connector.parse_message(raw_msg).unwrap();
        // Should return empty vec for subscription responses
        assert_eq!(messages.len(), 0);
    }

    #[test]
    fn test_parse_pong_message() {
        let connector = BybitConnector::new("wss://stream.bybit.com/v5/public/spot".to_string());

        let raw_msg = r#"{"op":"pong"}"#;
        let messages = connector.parse_message(raw_msg).unwrap();

        // Should return empty vec for pong messages
        assert_eq!(messages.len(), 0);
    }
}
