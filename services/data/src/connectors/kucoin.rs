//! Kucoin Exchange Connector
//!
//! Implements WebSocket connection to Kucoin with token-based authentication.
//!
//! ## API Documentation:
//! - WebSocket API: https://docs.kucoin.com/#websocket-feed
//! - Public Token: https://docs.kucoin.com/#apply-connect-token
//! - Market Data: https://docs.kucoin.com/#match-execution-data
//!
//! ## Authentication Flow:
//! 1. POST to /api/v1/bullet-public to get a token and instance servers
//! 2. Connect to WebSocket with token appended to URL
//! 3. Send client-side ping every 18 seconds (Kucoin doesn't use WebSocket ping/pong)
//!
//! ## Token Response:
//! ```json
//! {
//!   "code": "200000",
//!   "data": {
//!     "token": "2neAiuYvAU61ZDXANAGAsiL4-iAExhsBXZxftpOeh_55i3Ysy2q2LEsEWU64mdzUOPusi34M_wGoSf7iNyEWJ1UQy47YbpY4zVdzilNP-Bj3iXzrjjGlWtiYB9J6i9GjsxUuhPw3BlrzazF6ghq4L2MqQnCL...",
//!     "instanceServers": [
//!       {
//!         "endpoint": "wss://ws-api-spot.kucoin.com/",
//!         "encrypt": true,
//!         "protocol": "websocket",
//!         "pingInterval": 18000,
//!         "pingTimeout": 10000
//!       }
//!     ]
//!   }
//! }
//! ```
//!
//! ## Subscription Message:
//! ```json
//! {
//!   "id": "1545910660739",
//!   "type": "subscribe",
//!   "topic": "/market/match:BTC-USDT",
//!   "privateChannel": false,
//!   "response": true
//! }
//! ```
//!
//! ## Trade Message Format:
//! ```json
//! {
//!   "type": "message",
//!   "topic": "/market/match:BTC-USDT",
//!   "subject": "trade.l3match",
//!   "data": {
//!     "sequence": "1545896669145",
//!     "symbol": "BTC-USDT",
//!     "side": "buy",
//!     "size": "0.00001",
//!     "price": "50000.0",
//!     "takerOrderId": "5c24c5da03aa673885cd67aa",
//!     "time": "1545913818099033203",
//!     "type": "match",
//!     "makerOrderId": "5c24c5da03aa673885cd67ab",
//!     "tradeId": "5c24c5da03aa673885cd67ac"
//!   }
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::actors::{DataMessage, ExchangeConnector, TradeData, TradeSide, WebSocketConfig};

/// Kucoin connector implementation
pub struct KucoinConnector {
    rest_url: String,
    ws_endpoint: String,
    token: String,
    ping_interval: u64,
}

impl KucoinConnector {
    /// Create a new Kucoin connector with token authentication
    pub async fn new(rest_url: String) -> Result<Self> {
        info!(
            "KucoinConnector: Requesting WebSocket token from {}",
            rest_url
        );

        // Request token from Kucoin
        let token_url = format!("{}/api/v1/bullet-public", rest_url);
        let client = reqwest::Client::new();

        let response = client
            .post(&token_url)
            .send()
            .await
            .context("Failed to request Kucoin token")?;

        let token_response: KucoinTokenResponse = response
            .json()
            .await
            .context("Failed to parse Kucoin token response")?;

        if token_response.code != "200000" {
            anyhow::bail!(
                "Kucoin token request failed with code: {}",
                token_response.code
            );
        }

        // Extract the first instance server
        let instance = token_response
            .data
            .instance_servers
            .first()
            .context("No instance servers returned")?;

        let ws_endpoint = instance.endpoint.clone();
        let token = token_response.data.token.clone();
        let ping_interval = instance.ping_interval / 1000; // Convert ms to seconds

        info!(
            "KucoinConnector: Token acquired, endpoint: {}, ping interval: {}s",
            ws_endpoint, ping_interval
        );

        Ok(Self {
            rest_url,
            ws_endpoint,
            token,
            ping_interval,
        })
    }

    /// Parse a Kucoin trade message
    fn parse_trade_message(&self, msg: &KucoinTradeMessage) -> Result<TradeData> {
        let data = &msg.data;

        // Parse side
        let side = match data.side.as_str() {
            "buy" => TradeSide::Buy,
            "sell" => TradeSide::Sell,
            _ => {
                warn!("Kucoin: Unknown trade side: {}", data.side);
                return Err(anyhow::anyhow!("Unknown side: {}", data.side));
            }
        };

        // Parse price and size
        let price: f64 = data.price.parse().context("Failed to parse Kucoin price")?;

        let amount: f64 = data.size.parse().context("Failed to parse Kucoin size")?;

        // Parse timestamp (nanoseconds to milliseconds)
        let exchange_ts = data
            .time
            .parse::<i64>()
            .context("Failed to parse timestamp")?
            / 1_000_000;

        let receipt_ts = chrono::Utc::now().timestamp_millis();

        Ok(TradeData {
            symbol: data.symbol.clone(),
            exchange: "kucoin".to_string(),
            side,
            price,
            amount,
            exchange_ts,
            receipt_ts,
            trade_id: data.trade_id.clone(),
        })
    }

    /// Generate a unique message ID for subscriptions
    fn generate_msg_id() -> String {
        chrono::Utc::now().timestamp_millis().to_string()
    }
}

impl ExchangeConnector for KucoinConnector {
    fn exchange_name(&self) -> &str {
        "kucoin"
    }

    fn ws_url(&self) -> &str {
        // Return the full URL with token
        // Note: This is a bit of a hack since we can't modify the signature
        // In production, you might want to cache this in a field
        &self.ws_endpoint
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        // Build WebSocket URL with token parameter
        let ws_url = if self.ws_endpoint.ends_with('/') {
            format!(
                "{}?token={}&connectId={}",
                self.ws_endpoint,
                self.token,
                Self::generate_msg_id()
            )
        } else {
            format!(
                "{}/?token={}&connectId={}",
                self.ws_endpoint,
                self.token,
                Self::generate_msg_id()
            )
        };

        // Build subscription message
        let subscription = self.subscription_message(symbol);

        WebSocketConfig {
            url: ws_url,
            exchange: self.exchange_name().to_string(),
            symbol: symbol.to_string(),
            subscription_msg: subscription,
            ping_interval_secs: self.ping_interval,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }

    fn subscription_message(&self, symbol: &str) -> Option<String> {
        // Kucoin subscription format
        let topic = format!("/market/match:{}", symbol);

        let sub_msg = KucoinSubscription {
            id: Self::generate_msg_id(),
            msg_type: "subscribe".to_string(),
            topic,
            private_channel: false,
            response: true,
        };

        serde_json::to_string(&sub_msg).ok()
    }

    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>> {
        // Try to parse as welcome message
        if let Ok(welcome) = serde_json::from_str::<KucoinWelcome>(raw)
            && welcome.msg_type == "welcome"
        {
            info!("Kucoin: Connected - {}", welcome.id);
            return Ok(vec![]);
        }

        // Try to parse as subscription ACK
        if let Ok(ack) = serde_json::from_str::<KucoinAck>(raw)
            && ack.msg_type == "ack"
        {
            debug!("Kucoin: Subscription acknowledged - ID: {}", ack.id);
            return Ok(vec![]);
        }

        // Try to parse as pong response
        if let Ok(pong) = serde_json::from_str::<KucoinPong>(raw)
            && pong.msg_type == "pong"
        {
            debug!("Kucoin: Received pong");
            return Ok(vec![]);
        }

        // Try to parse as trade message
        match serde_json::from_str::<KucoinTradeMessage>(raw) {
            Ok(trade_msg) => {
                // Verify this is a trade message
                if trade_msg.msg_type != "message" {
                    debug!("Kucoin: Ignoring non-message type: {}", trade_msg.msg_type);
                    return Ok(vec![]);
                }

                if !trade_msg.topic.starts_with("/market/match:") {
                    debug!("Kucoin: Ignoring non-trade topic: {}", trade_msg.topic);
                    return Ok(vec![]);
                }

                let trade_data = self
                    .parse_trade_message(&trade_msg)
                    .context("Failed to parse Kucoin trade")?;

                Ok(vec![DataMessage::Trade(trade_data)])
            }
            Err(e) => {
                warn!(
                    "Kucoin: Failed to parse message: {} - Raw: {}",
                    e,
                    if raw.len() > 200 { &raw[..200] } else { raw }
                );
                Ok(vec![])
            }
        }
    }
}

/// Kucoin token response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinTokenResponse {
    code: String,
    data: KucoinTokenData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinTokenData {
    token: String,
    #[serde(rename = "instanceServers")]
    instance_servers: Vec<KucoinInstanceServer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinInstanceServer {
    endpoint: String,
    encrypt: bool,
    protocol: String,
    #[serde(rename = "pingInterval")]
    ping_interval: u64,
    #[serde(rename = "pingTimeout")]
    ping_timeout: u64,
}

/// Kucoin subscription message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinSubscription {
    id: String,
    #[serde(rename = "type")]
    msg_type: String,
    topic: String,
    #[serde(rename = "privateChannel")]
    private_channel: bool,
    response: bool,
}

/// Kucoin welcome message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinWelcome {
    id: String,
    #[serde(rename = "type")]
    msg_type: String,
}

/// Kucoin ACK message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinAck {
    id: String,
    #[serde(rename = "type")]
    msg_type: String,
}

/// Kucoin pong message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinPong {
    id: String,
    #[serde(rename = "type")]
    msg_type: String,
}

/// Kucoin trade message wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinTradeMessage {
    #[serde(rename = "type")]
    msg_type: String,
    topic: String,
    subject: String,
    data: KucoinTrade,
}

/// Kucoin individual trade
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KucoinTrade {
    sequence: String,
    symbol: String,
    side: String,
    size: String,
    price: String,
    #[serde(rename = "takerOrderId")]
    taker_order_id: String,
    time: String, // Nanoseconds as string
    #[serde(rename = "type")]
    trade_type: String,
    #[serde(rename = "makerOrderId")]
    maker_order_id: String,
    #[serde(rename = "tradeId")]
    trade_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscription_message() {
        // Can't easily test the constructor without hitting the API
        // So we'll test the subscription message generation logic directly
        let sub_msg = KucoinSubscription {
            id: "123456".to_string(),
            msg_type: "subscribe".to_string(),
            topic: "/market/match:BTC-USDT".to_string(),
            private_channel: false,
            response: true,
        };

        let json = serde_json::to_string(&sub_msg).unwrap();
        assert!(json.contains("\"type\":\"subscribe\""));
        assert!(json.contains("/market/match:BTC-USDT"));
    }

    #[test]
    fn test_parse_welcome_message() {
        let raw_msg = r#"{
            "id": "hQvf8jkno",
            "type": "welcome"
        }"#;

        let welcome: KucoinWelcome = serde_json::from_str(raw_msg).unwrap();
        assert_eq!(welcome.msg_type, "welcome");
        assert_eq!(welcome.id, "hQvf8jkno");
    }

    #[test]
    fn test_parse_trade_data() {
        let raw_msg = r#"{
            "type": "message",
            "topic": "/market/match:BTC-USDT",
            "subject": "trade.l3match",
            "data": {
                "sequence": "1545896669145",
                "symbol": "BTC-USDT",
                "side": "buy",
                "size": "0.00001",
                "price": "50000.0",
                "takerOrderId": "5c24c5da03aa673885cd67aa",
                "time": "1545913818099033203",
                "type": "match",
                "makerOrderId": "5c24c5da03aa673885cd67ab",
                "tradeId": "5c24c5da03aa673885cd67ac"
            }
        }"#;

        let trade_msg: KucoinTradeMessage = serde_json::from_str(raw_msg).unwrap();
        assert_eq!(trade_msg.msg_type, "message");
        assert_eq!(trade_msg.data.symbol, "BTC-USDT");
        assert_eq!(trade_msg.data.side, "buy");
        assert_eq!(trade_msg.data.price, "50000.0");
        assert_eq!(trade_msg.data.size, "0.00001");
    }

    #[test]
    fn test_generate_msg_id() {
        let id1 = KucoinConnector::generate_msg_id();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let id2 = KucoinConnector::generate_msg_id();

        // IDs should be different (timestamp-based)
        assert_ne!(id1, id2);

        // Should be parseable as i64
        assert!(id1.parse::<i64>().is_ok());
    }
}
