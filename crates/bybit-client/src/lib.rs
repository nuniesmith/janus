//! Bybit V5 API Client
//!
//! This module provides WebSocket and REST API clients for Bybit V5.
//! Supports both public (market data) and private (trading) streams.

use anyhow::{Result, anyhow};
use futures_util::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::Sha256;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::time::{Duration, interval};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

type HmacSha256 = Hmac<Sha256>;

// ============================================================================
// Constants
// ============================================================================

const BYBIT_WS_PUBLIC_MAINNET: &str = "wss://stream.bybit.com/v5/public/linear";
const BYBIT_WS_PRIVATE_MAINNET: &str = "wss://stream.bybit.com/v5/private";
const BYBIT_WS_PUBLIC_TESTNET: &str = "wss://stream-testnet.bybit.com/v5/public/linear";
const BYBIT_WS_PRIVATE_TESTNET: &str = "wss://stream-testnet.bybit.com/v5/private";

const BYBIT_REST_MAINNET: &str = "https://api.bybit.com";
const BYBIT_REST_TESTNET: &str = "https://api-testnet.bybit.com";

const HEARTBEAT_INTERVAL_SECS: u64 = 20;
const RECONNECT_DELAY_SECS: u64 = 5;

// ============================================================================
// Credentials
// ============================================================================

/// Bybit API credentials
#[derive(Clone)]
pub struct BybitCredentials {
    pub api_key: String,
    pub api_secret: String,
}

impl BybitCredentials {
    pub fn new(api_key: String, api_secret: String) -> Self {
        Self {
            api_key,
            api_secret,
        }
    }

    /// Generate HMAC-SHA256 signature for REST API
    pub fn sign(&self, timestamp: u64, params: &str) -> String {
        let sign_str = format!("{}{}{}", timestamp, &self.api_key, params);
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(sign_str.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Generate WebSocket authentication signature
    pub fn sign_ws(&self, expires: u64) -> String {
        let sign_str = format!("GET/realtime{}", expires);
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(sign_str.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }
}

// ============================================================================
// Market Data Types
// ============================================================================

/// Bybit market tick (Level 1 orderbook data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitTick {
    pub symbol: String,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub timestamp: u64,
}

/// Bybit kline/candlestick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitKline {
    pub symbol: String,
    pub interval: String,
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitTrade {
    pub symbol: String,
    pub side: String,
    pub size: f64,
    pub price: f64,
    pub timestamp: u64,
}

// ============================================================================
// Order Types
// ============================================================================

/// Order side
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "Buy"),
            OrderSide::Sell => write!(f, "Sell"),
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum OrderType {
    Limit,
    Market,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Limit => write!(f, "Limit"),
            OrderType::Market => write!(f, "Market"),
        }
    }
}

/// Time in force
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC, // Good Till Cancel
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
}

/// Order status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Order request
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderRequest {
    pub category: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub qty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_in_force: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_link_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reduce_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub close_on_trigger: Option<bool>,
}

/// Order response
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderResponse {
    pub order_id: String,
    pub order_link_id: String,
}

// ============================================================================
// WebSocket Messages
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct WsSubscribe {
    op: String,
    args: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WsAuth {
    op: String,
    args: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WsResponse {
    success: Option<bool>,
    ret_msg: Option<String>,
    op: Option<String>,
    topic: Option<String>,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    msg_type: Option<String>,
    data: Option<serde_json::Value>,
}

// ============================================================================
// WebSocket Client
// ============================================================================

/// WebSocket message types
#[derive(Debug, Clone)]
pub enum WsMessage {
    Tick(BybitTick),
    Kline(BybitKline),
    Trade(BybitTrade),
    OrderUpdate(serde_json::Value),
    PositionUpdate(serde_json::Value),
    ExecutionUpdate(serde_json::Value),
}

/// Bybit WebSocket client
pub struct BybitWebSocket {
    url: String,
    credentials: Option<Arc<BybitCredentials>>,
    message_tx: mpsc::UnboundedSender<WsMessage>,
    is_private: bool,
}

impl BybitWebSocket {
    /// Create new public WebSocket client
    pub fn new_public(testnet: bool) -> (Self, mpsc::UnboundedReceiver<WsMessage>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let url = if testnet {
            BYBIT_WS_PUBLIC_TESTNET.to_string()
        } else {
            BYBIT_WS_PUBLIC_MAINNET.to_string()
        };

        (
            Self {
                url,
                credentials: None,
                message_tx: tx,
                is_private: false,
            },
            rx,
        )
    }

    /// Create new private WebSocket client
    pub fn new_private(
        credentials: BybitCredentials,
        testnet: bool,
    ) -> (Self, mpsc::UnboundedReceiver<WsMessage>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let url = if testnet {
            BYBIT_WS_PRIVATE_TESTNET.to_string()
        } else {
            BYBIT_WS_PRIVATE_MAINNET.to_string()
        };

        (
            Self {
                url,
                credentials: Some(Arc::new(credentials)),
                message_tx: tx,
                is_private: true,
            },
            rx,
        )
    }

    /// Connect and start message loop
    pub async fn connect(self, subscriptions: Vec<String>) -> Result<()> {
        loop {
            match self.connect_internal(&subscriptions).await {
                Ok(_) => {
                    warn!(
                        "WebSocket connection closed, reconnecting in {}s",
                        RECONNECT_DELAY_SECS
                    );
                }
                Err(e) => {
                    error!(
                        "WebSocket error: {}, reconnecting in {}s",
                        e, RECONNECT_DELAY_SECS
                    );
                }
            }
            tokio::time::sleep(Duration::from_secs(RECONNECT_DELAY_SECS)).await;
        }
    }

    async fn connect_internal(&self, subscriptions: &[String]) -> Result<()> {
        info!("Connecting to Bybit WebSocket: {}", self.url);

        let (ws_stream, _) = connect_async(&self.url)
            .await
            .map_err(|e| anyhow!("Failed to connect: {}", e))?;

        info!("WebSocket connected");

        let (mut write, mut read) = ws_stream.split();

        // Authenticate if private stream
        if self.is_private
            && let Some(creds) = &self.credentials
        {
            let expires = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64 + 10000;
            let signature = creds.sign_ws(expires);

            let auth_msg = WsAuth {
                op: "auth".to_string(),
                args: vec![creds.api_key.clone(), expires.to_string(), signature],
            };

            write
                .send(Message::Text(serde_json::to_string(&auth_msg)?.into()))
                .await?;

            info!("Sent authentication message");
        }

        // Subscribe to topics
        if !subscriptions.is_empty() {
            let sub_msg = WsSubscribe {
                op: "subscribe".to_string(),
                args: subscriptions.to_vec(),
            };

            write
                .send(Message::Text(serde_json::to_string(&sub_msg)?.into()))
                .await?;

            info!("Subscribed to topics: {:?}", subscriptions);
        }

        // Start heartbeat task
        let mut heartbeat = interval(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));
        let mut write_half = write;

        // Message processing loop
        loop {
            tokio::select! {
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Err(e) = self.handle_message(&text) {
                                warn!("Failed to handle message: {}", e);
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            debug!("Received ping, sending pong");
                            write_half.send(Message::Pong(data)).await?;
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("WebSocket closed by server");
                            break;
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                        None => {
                            warn!("WebSocket stream ended");
                            break;
                        }
                        _ => {}
                    }
                }
                _ = heartbeat.tick() => {
                    debug!("Sending heartbeat");
                    let ping = json!({"op": "ping"});
                    write_half.send(Message::Text(ping.to_string().into())).await?;
                }
            }
        }

        Ok(())
    }

    fn handle_message(&self, text: &str) -> Result<()> {
        let response: WsResponse = serde_json::from_str(text)?;

        // Handle subscription/auth responses
        if let Some(op) = &response.op
            && (op == "subscribe" || op == "auth")
        {
            if let Some(success) = response.success {
                if success {
                    debug!("Operation {} succeeded", op);
                } else {
                    warn!("Operation {} failed: {:?}", op, response.ret_msg);
                }
            }
            return Ok(());
        }

        // Handle data messages
        if let Some(topic) = &response.topic
            && let Some(data) = &response.data
        {
            self.parse_topic_data(topic, data)?;
        }

        Ok(())
    }

    fn parse_topic_data(&self, topic: &str, data: &serde_json::Value) -> Result<()> {
        if topic.starts_with("orderbook") {
            // Parse orderbook snapshot/delta into tick
            if let Some(obj) = data.as_object()
                && let (Some(symbol), Some(b), Some(a)) = (
                    obj.get("s").and_then(|s| s.as_str()),
                    obj.get("b").and_then(|v| v.as_array()),
                    obj.get("a").and_then(|v| v.as_array()),
                )
                && let (Some(bid), Some(ask)) = (b.first(), a.first())
            {
                let tick = BybitTick {
                    symbol: symbol.to_string(),
                    bid_price: bid[0].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    bid_size: bid[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    ask_price: ask[0].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    ask_size: ask[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    last_price: 0.0,
                    timestamp: obj.get("ts").and_then(|v| v.as_u64()).unwrap_or(0),
                };
                let _ = self.message_tx.send(WsMessage::Tick(tick));
            }
        } else if topic.starts_with("publicTrade") {
            // Parse trade data
            if let Some(arr) = data.as_array() {
                for trade_data in arr {
                    if let Some(obj) = trade_data.as_object() {
                        let trade = BybitTrade {
                            symbol: obj
                                .get("s")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            side: obj
                                .get("S")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            price: obj
                                .get("p")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0),
                            size: obj
                                .get("v")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0),
                            timestamp: obj.get("T").and_then(|v| v.as_u64()).unwrap_or(0),
                        };
                        let _ = self.message_tx.send(WsMessage::Trade(trade));
                    }
                }
            }
        } else if topic.starts_with("kline") {
            // Parse kline data
            if let Some(arr) = data.as_array() {
                for kline_data in arr {
                    if let Some(obj) = kline_data.as_object() {
                        let kline = BybitKline {
                            symbol: obj
                                .get("symbol")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            interval: obj
                                .get("interval")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            open_time: obj.get("start").and_then(|v| v.as_u64()).unwrap_or(0),
                            open: obj
                                .get("open")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0),
                            high: obj
                                .get("high")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0),
                            low: obj
                                .get("low")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0),
                            close: obj
                                .get("close")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0),
                            volume: obj
                                .get("volume")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0),
                            turnover: obj
                                .get("turnover")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0),
                        };
                        let _ = self.message_tx.send(WsMessage::Kline(kline));
                    }
                }
            }
        } else if topic == "order" {
            let _ = self.message_tx.send(WsMessage::OrderUpdate(data.clone()));
        } else if topic == "position" {
            let _ = self
                .message_tx
                .send(WsMessage::PositionUpdate(data.clone()));
        } else if topic == "execution" {
            let _ = self
                .message_tx
                .send(WsMessage::ExecutionUpdate(data.clone()));
        }

        Ok(())
    }
}

// ============================================================================
// REST Client
// ============================================================================

/// Bybit REST API client
pub struct BybitRestClient {
    base_url: String,
    credentials: Arc<BybitCredentials>,
    client: Client,
}

impl BybitRestClient {
    /// Create new REST client
    pub fn new(credentials: BybitCredentials, testnet: bool) -> Self {
        let base_url = if testnet {
            BYBIT_REST_TESTNET.to_string()
        } else {
            BYBIT_REST_MAINNET.to_string()
        };

        Self {
            base_url,
            credentials: Arc::new(credentials),
            client: Client::new(),
        }
    }

    /// Place a new order
    pub async fn place_order(&self, order: OrderRequest) -> Result<OrderResponse> {
        let endpoint = "/v5/order/create";
        let body = serde_json::to_string(&order)?;

        let response = self.post(endpoint, &body).await?;
        let parsed: serde_json::Value = response.json().await?;

        if let Some(result) = parsed.get("result") {
            let order_resp: OrderResponse = serde_json::from_value(result.clone())?;
            Ok(order_resp)
        } else {
            Err(anyhow!("Invalid response: {:?}", parsed))
        }
    }

    /// Cancel an order
    pub async fn cancel_order(&self, category: &str, symbol: &str, order_id: &str) -> Result<()> {
        let endpoint = "/v5/order/cancel";
        let body = json!({
            "category": category,
            "symbol": symbol,
            "orderId": order_id,
        })
        .to_string();

        let response = self.post(endpoint, &body).await?;
        let parsed: serde_json::Value = response.json().await?;

        if parsed.get("retCode").and_then(|v| v.as_i64()) == Some(0) {
            Ok(())
        } else {
            Err(anyhow!("Cancel failed: {:?}", parsed))
        }
    }

    /// Get account balance
    pub async fn get_balance(&self, account_type: &str) -> Result<serde_json::Value> {
        let endpoint = format!("/v5/account/wallet-balance?accountType={}", account_type);
        let response = self.get(&endpoint).await?;
        let parsed: serde_json::Value = response.json().await?;
        Ok(parsed)
    }

    /// Get positions
    pub async fn get_positions(
        &self,
        category: &str,
        symbol: Option<&str>,
    ) -> Result<serde_json::Value> {
        let mut endpoint = format!("/v5/position/list?category={}", category);
        if let Some(sym) = symbol {
            endpoint.push_str(&format!("&symbol={}", sym));
        }
        let response = self.get(&endpoint).await?;
        let parsed: serde_json::Value = response.json().await?;
        Ok(parsed)
    }

    // Private helper methods

    async fn get(&self, endpoint: &str) -> Result<reqwest::Response> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;

        let query_string = endpoint.split('?').nth(1).unwrap_or("");
        let signature = self.credentials.sign(timestamp, query_string);

        let url = format!("{}{}", self.base_url, endpoint);

        let response = self
            .client
            .get(&url)
            .header("X-BAPI-API-KEY", &self.credentials.api_key)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-SIGN", signature)
            .send()
            .await?;

        Ok(response)
    }

    async fn post(&self, endpoint: &str, body: &str) -> Result<reqwest::Response> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;

        let signature = self.credentials.sign(timestamp, body);

        let url = format!("{}{}", self.base_url, endpoint);

        let response = self
            .client
            .post(&url)
            .header("X-BAPI-API-KEY", &self.credentials.api_key)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-SIGN", signature)
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await?;

        Ok(response)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials() {
        let creds = BybitCredentials::new("test_key".to_string(), "test_secret".to_string());
        assert_eq!(creds.api_key, "test_key");
        assert_eq!(creds.api_secret, "test_secret");
    }

    #[test]
    fn test_signature() {
        let creds = BybitCredentials::new("test_key".to_string(), "test_secret".to_string());
        let sig = creds.sign(1234567890, "param1=value1");
        assert!(!sig.is_empty());
        assert_eq!(sig.len(), 64); // HMAC-SHA256 produces 64 hex chars
    }

    #[test]
    fn test_order_request_serialization() {
        let order = OrderRequest {
            category: "linear".to_string(),
            symbol: "BTCUSD".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            qty: "0.001".to_string(),
            price: Some("50000.0".to_string()),
            time_in_force: Some("GTC".to_string()),
            order_link_id: None,
            reduce_only: None,
            close_on_trigger: None,
        };

        let json = serde_json::to_string(&order).unwrap();
        assert!(json.contains("Buy"));
        assert!(json.contains("Limit"));
        assert!(json.contains("BTCUSD"));
    }

    #[tokio::test]
    async fn test_websocket_creation() {
        let (ws, _rx) = BybitWebSocket::new_public(true);
        assert!(!ws.is_private);
        assert!(ws.url.contains("testnet"));
    }

    #[test]
    fn test_rest_client_creation() {
        let creds = BybitCredentials::new("key".to_string(), "secret".to_string());
        let client = BybitRestClient::new(creds, true);
        assert!(client.base_url.contains("testnet"));
    }
}
