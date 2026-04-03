//! Bybit WebSocket Client
//!
//! Real-time streaming of order updates, position updates, and account balance changes
//! from Bybit's private WebSocket channels.

use crate::error::{ExecutionError, Result};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tokio::time::{Duration, sleep};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};

type HmacSha256 = Hmac<Sha256>;

const WS_MAINNET: &str = "wss://stream.bybit.com/v5/private";
const WS_TESTNET: &str = "wss://stream-testnet.bybit.com/v5/private";
const PING_INTERVAL_SECS: u64 = 20;
const RECONNECT_DELAY_SECS: u64 = 5;
const MAX_RECONNECT_ATTEMPTS: u32 = 10;

/// WebSocket message types from Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
#[serde(rename_all = "lowercase")]
enum WsRequest {
    Auth {
        args: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        req_id: Option<String>,
    },
    Subscribe {
        args: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        req_id: Option<String>,
    },
    Unsubscribe {
        args: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        req_id: Option<String>,
    },
    #[serde(rename = "ping")]
    Ping {
        #[serde(skip_serializing_if = "Option::is_none")]
        req_id: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum WsResponse {
    Success {
        success: bool,
        ret_msg: String,
        #[serde(default)]
        req_id: Option<String>,
        conn_id: String,
        op: String,
    },
    Pong {
        success: bool,
        ret_msg: String,
        conn_id: String,
        op: String,
    },
    Data {
        topic: String,
        #[serde(rename = "type")]
        update_type: String,
        data: serde_json::Value,
        #[serde(default)]
        ts: i64,
        #[serde(default)]
        cts: i64,
    },
}

/// Order update from WebSocket
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderUpdate {
    pub order_id: String,
    pub order_link_id: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub price: String,
    pub qty: String,
    pub cum_exec_qty: String,
    pub cum_exec_value: String,
    pub cum_exec_fee: String,
    pub order_status: String,
    pub avg_price: String,
    pub created_time: String,
    pub updated_time: String,
    #[serde(default)]
    pub reject_reason: String,
}

/// Position update from WebSocket
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionUpdate {
    pub symbol: String,
    pub side: String,
    pub size: String,
    pub position_value: String,
    pub entry_price: String,
    pub leverage: String,
    pub mark_price: String,
    pub liq_price: String,
    pub unrealised_pnl: String,
    pub cum_realised_pnl: String,
    pub position_status: String,
    pub updated_time: String,
}

/// Wallet (balance) update from WebSocket
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WalletUpdate {
    pub account_type: String,
    pub account_im: String,
    pub account_mm: String,
    pub total_equity: String,
    pub total_wallet_balance: String,
    pub total_margin_balance: String,
    pub total_available_balance: String,
    pub total_perp_upl: String,
    pub total_initial_margin: String,
    pub total_maintenance_margin: String,
    pub coin: Vec<CoinBalance>,
    pub updated_time: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoinBalance {
    pub coin: String,
    pub equity: String,
    pub wallet_balance: String,
    pub available_balance: String,
    pub locked: String,
}

/// Event types broadcast by the WebSocket client
#[derive(Debug, Clone)]
pub enum BybitEvent {
    OrderUpdate(OrderUpdate),
    PositionUpdate(PositionUpdate),
    WalletUpdate(WalletUpdate),
    Connected,
    Disconnected,
    Error(String),
}

/// Configuration for Bybit WebSocket client
#[derive(Debug, Clone)]
pub struct BybitWsConfig {
    pub api_key: String,
    pub api_secret: String,
    pub testnet: bool,
    pub subscribe_orders: bool,
    pub subscribe_positions: bool,
    pub subscribe_wallet: bool,
}

/// Bybit WebSocket client for real-time updates
pub struct BybitWebSocket {
    config: BybitWsConfig,
    event_tx: broadcast::Sender<BybitEvent>,
    subscriptions: Arc<RwLock<Vec<String>>>,
    is_running: Arc<RwLock<bool>>,
    reconnect_count: Arc<RwLock<u32>>,
}

impl BybitWebSocket {
    /// Create a new Bybit WebSocket client
    pub fn new(config: BybitWsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(1000);
        Self {
            config,
            event_tx,
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            is_running: Arc::new(RwLock::new(false)),
            reconnect_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<BybitEvent> {
        self.event_tx.subscribe()
    }

    /// Start the WebSocket client
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(ExecutionError::WebSocketError(
                "WebSocket client already running".to_string(),
            ));
        }
        *is_running = true;
        drop(is_running);

        let config = self.config.clone();
        let event_tx = self.event_tx.clone();
        let subscriptions = self.subscriptions.clone();
        let is_running = self.is_running.clone();
        let reconnect_count = self.reconnect_count.clone();

        tokio::spawn(async move {
            Self::run_loop(config, event_tx, subscriptions, is_running, reconnect_count).await;
        });

        Ok(())
    }

    /// Stop the WebSocket client
    pub async fn stop(&self) {
        let mut is_running = self.is_running.write().await;
        *is_running = false;
        info!("Bybit WebSocket client stopped");
    }

    /// Main run loop with reconnection logic
    async fn run_loop(
        config: BybitWsConfig,
        event_tx: broadcast::Sender<BybitEvent>,
        subscriptions: Arc<RwLock<Vec<String>>>,
        is_running: Arc<RwLock<bool>>,
        reconnect_count: Arc<RwLock<u32>>,
    ) {
        while *is_running.read().await {
            let count = *reconnect_count.read().await;
            if count > 0 {
                let delay = std::cmp::min(RECONNECT_DELAY_SECS * count as u64, 60);
                info!("Reconnecting in {} seconds (attempt {})", delay, count);
                sleep(Duration::from_secs(delay)).await;
            }

            if count >= MAX_RECONNECT_ATTEMPTS {
                error!("Max reconnection attempts reached, stopping WebSocket");
                let _ = event_tx.send(BybitEvent::Error(
                    "Max reconnection attempts exceeded".to_string(),
                ));
                break;
            }

            match Self::connect_and_run(&config, &event_tx, &subscriptions).await {
                Ok(_) => {
                    info!("WebSocket connection closed normally");
                    *reconnect_count.write().await = 0;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    let _ = event_tx.send(BybitEvent::Error(e.to_string()));
                    let _ = event_tx.send(BybitEvent::Disconnected);
                    *reconnect_count.write().await += 1;
                }
            }

            if !*is_running.read().await {
                break;
            }
        }
    }

    /// Connect to WebSocket and run until disconnect
    async fn connect_and_run(
        config: &BybitWsConfig,
        event_tx: &broadcast::Sender<BybitEvent>,
        subscriptions: &Arc<RwLock<Vec<String>>>,
    ) -> Result<()> {
        let url = if config.testnet {
            WS_TESTNET
        } else {
            WS_MAINNET
        };

        info!("Connecting to Bybit WebSocket: {}", url);
        let (ws_stream, _) = connect_async(url)
            .await
            .map_err(|e| ExecutionError::WebSocketError(format!("Connection failed: {}", e)))?;

        info!("WebSocket connected");
        let (mut write, mut read) = ws_stream.split();

        // Authenticate
        let auth_msg = Self::create_auth_message(&config.api_key, &config.api_secret)?;
        write
            .send(Message::Text(auth_msg.into()))
            .await
            .map_err(|e| ExecutionError::WebSocketError(format!("Auth send failed: {}", e)))?;

        // Wait for auth confirmation
        if let Some(msg) = read.next().await {
            let msg = msg.map_err(|e| {
                ExecutionError::WebSocketError(format!("Auth response read failed: {}", e))
            })?;
            if let Message::Text(text) = msg {
                debug!("Auth response: {}", text);
                let response: WsResponse = serde_json::from_str(&text).map_err(|e| {
                    ExecutionError::WebSocketError(format!("Auth parse failed: {}", e))
                })?;

                if let WsResponse::Success {
                    success, ret_msg, ..
                } = response
                {
                    if !success {
                        return Err(ExecutionError::WebSocketError(format!(
                            "Auth failed: {}",
                            ret_msg
                        )));
                    }
                    info!("WebSocket authenticated");
                }
            }
        }

        // Subscribe to channels
        let mut subs = Vec::new();
        if config.subscribe_orders {
            subs.push("order".to_string());
        }
        if config.subscribe_positions {
            subs.push("position".to_string());
        }
        if config.subscribe_wallet {
            subs.push("wallet".to_string());
        }

        if !subs.is_empty() {
            *subscriptions.write().await = subs.clone();
            let sub_msg = Self::create_subscribe_message(subs)?;
            write
                .send(Message::Text(sub_msg.into()))
                .await
                .map_err(|e| ExecutionError::WebSocketError(format!("Subscribe failed: {}", e)))?;
        }

        let _ = event_tx.send(BybitEvent::Connected);

        // Ping task
        let ping_task = {
            let (ping_tx, mut ping_rx) = tokio::sync::mpsc::channel::<()>(1);
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(PING_INTERVAL_SECS));
                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            if ping_tx.send(()).await.is_err() {
                                break;
                            }
                        }
                        _ = ping_rx.recv() => break,
                    }
                }
            })
        };

        // Read messages
        loop {
            tokio::select! {
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Err(e) = Self::handle_message(&text, event_tx) {
                                warn!("Failed to handle message: {}", e);
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("WebSocket close frame received");
                            break;
                        }
                        Some(Ok(Message::Ping(payload))) => {
                            // Send pong manually
                            if write.send(Message::Pong(payload)).await.is_err() {
                                break;
                            }
                        }
                        Some(Ok(Message::Pong(_))) => {
                            debug!("Pong received");
                        }
                        Some(Err(e)) => {
                            error!("WebSocket read error: {}", e);
                            break;
                        }
                        None => {
                            info!("WebSocket stream ended");
                            break;
                        }
                        _ => {}
                    }
                }
                _ = tokio::time::sleep(Duration::from_secs(PING_INTERVAL_SECS)) => {
                    // Send ping
                    let ping = serde_json::to_string(&WsRequest::Ping { req_id: None })
                        .unwrap_or_else(|_| r#"{"op":"ping"}"#.to_string());
                    if write.send(Message::Text(ping.into())).await.is_err() {
                        break;
                    }
                }
            }
        }

        ping_task.abort();
        Ok(())
    }

    /// Create authentication message
    fn create_auth_message(api_key: &str, api_secret: &str) -> Result<String> {
        let expires = (Utc::now().timestamp_millis() + 10000).to_string();
        let sign_str = format!("GET/realtime{}", expires);

        let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
            .map_err(|e| ExecutionError::SignatureError(e.to_string()))?;
        mac.update(sign_str.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let auth_req = WsRequest::Auth {
            args: vec![api_key.to_string(), expires, signature],
            req_id: Some(uuid::Uuid::new_v4().to_string()),
        };

        serde_json::to_string(&auth_req)
            .map_err(|e| ExecutionError::SerializationError(e.to_string()))
    }

    /// Create subscribe message
    fn create_subscribe_message(topics: Vec<String>) -> Result<String> {
        let sub_req = WsRequest::Subscribe {
            args: topics,
            req_id: Some(uuid::Uuid::new_v4().to_string()),
        };

        serde_json::to_string(&sub_req)
            .map_err(|e| ExecutionError::SerializationError(e.to_string()))
    }

    /// Handle incoming WebSocket message
    fn handle_message(text: &str, event_tx: &broadcast::Sender<BybitEvent>) -> Result<()> {
        let response: WsResponse = serde_json::from_str(text)
            .map_err(|e| ExecutionError::DeserializationError(e.to_string()))?;

        match response {
            WsResponse::Success {
                success,
                ret_msg,
                op,
                ..
            } => {
                debug!("Operation {} success: {} - {}", op, success, ret_msg);
            }
            WsResponse::Pong { .. } => {
                debug!("Pong received");
            }
            WsResponse::Data { topic, data, .. } => {
                if topic.starts_with("order") {
                    Self::handle_order_update(data, event_tx)?;
                } else if topic.starts_with("position") {
                    Self::handle_position_update(data, event_tx)?;
                } else if topic.starts_with("wallet") {
                    Self::handle_wallet_update(data, event_tx)?;
                } else {
                    debug!("Unknown topic: {}", topic);
                }
            }
        }

        Ok(())
    }

    /// Handle order update
    fn handle_order_update(
        data: serde_json::Value,
        event_tx: &broadcast::Sender<BybitEvent>,
    ) -> Result<()> {
        // Data can be an array or single object
        let updates: Vec<OrderUpdate> = if data.is_array() {
            serde_json::from_value(data)
                .map_err(|e| ExecutionError::DeserializationError(e.to_string()))?
        } else {
            vec![
                serde_json::from_value(data)
                    .map_err(|e| ExecutionError::DeserializationError(e.to_string()))?,
            ]
        };

        for update in updates {
            debug!("Order update: {:?}", update);
            let _ = event_tx.send(BybitEvent::OrderUpdate(update));
        }

        Ok(())
    }

    /// Handle position update
    fn handle_position_update(
        data: serde_json::Value,
        event_tx: &broadcast::Sender<BybitEvent>,
    ) -> Result<()> {
        let updates: Vec<PositionUpdate> = if data.is_array() {
            serde_json::from_value(data)
                .map_err(|e| ExecutionError::DeserializationError(e.to_string()))?
        } else {
            vec![
                serde_json::from_value(data)
                    .map_err(|e| ExecutionError::DeserializationError(e.to_string()))?,
            ]
        };

        for update in updates {
            debug!("Position update: {:?}", update);
            let _ = event_tx.send(BybitEvent::PositionUpdate(update));
        }

        Ok(())
    }

    /// Handle wallet update
    fn handle_wallet_update(
        data: serde_json::Value,
        event_tx: &broadcast::Sender<BybitEvent>,
    ) -> Result<()> {
        let updates: Vec<WalletUpdate> = if data.is_array() {
            serde_json::from_value(data)
                .map_err(|e| ExecutionError::DeserializationError(e.to_string()))?
        } else {
            vec![
                serde_json::from_value(data)
                    .map_err(|e| ExecutionError::DeserializationError(e.to_string()))?,
            ]
        };

        for update in updates {
            debug!("Wallet update: {:?}", update);
            let _ = event_tx.send(BybitEvent::WalletUpdate(update));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_config_creation() {
        let config = BybitWsConfig {
            api_key: "test_key".to_string(),
            api_secret: "test_secret".to_string(),
            testnet: true,
            subscribe_orders: true,
            subscribe_positions: true,
            subscribe_wallet: false,
        };

        assert!(config.testnet);
        assert!(config.subscribe_orders);
        assert!(!config.subscribe_wallet);
    }

    #[test]
    fn test_create_auth_message() {
        let msg = BybitWebSocket::create_auth_message("test_key", "test_secret");
        assert!(msg.is_ok());
        let msg_str = msg.unwrap();
        assert!(msg_str.contains("auth"));
        assert!(msg_str.contains("test_key"));
    }

    #[test]
    fn test_create_subscribe_message() {
        let topics = vec!["order".to_string(), "position".to_string()];
        let msg = BybitWebSocket::create_subscribe_message(topics);
        assert!(msg.is_ok());
        let msg_str = msg.unwrap();
        assert!(msg_str.contains("subscribe"));
        assert!(msg_str.contains("order"));
        assert!(msg_str.contains("position"));
    }

    #[tokio::test]
    async fn test_websocket_creation() {
        let config = BybitWsConfig {
            api_key: "test".to_string(),
            api_secret: "test".to_string(),
            testnet: true,
            subscribe_orders: true,
            subscribe_positions: false,
            subscribe_wallet: false,
        };

        let ws = BybitWebSocket::new(config);
        let mut rx = ws.subscribe();

        // Should be able to subscribe
        assert!(rx.try_recv().is_err()); // No events yet
    }
}
