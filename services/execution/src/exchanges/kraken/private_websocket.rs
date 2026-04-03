//! Kraken Private WebSocket Client
//!
//! Authenticated WebSocket connection to Kraken for real-time execution updates.
//! This module handles order fill notifications, order status changes, and account updates.
//!
//! # WebSocket API v2 (Authenticated)
//!
//! Kraken's private WebSocket API v2 requires authentication via a WebSocket token
//! obtained from the REST API. The token is used to subscribe to private channels.
//!
//! ## Supported Channels
//!
//! - `executions`: Real-time fill notifications
//! - `balances`: Account balance updates
//! - `openOrders`: Open order status changes
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::exchanges::kraken::private_websocket::{
//!     KrakenPrivateWebSocket, KrakenPrivateWsConfig, PrivateWsEvent,
//! };
//!
//! let config = KrakenPrivateWsConfig::from_env();
//! let ws = KrakenPrivateWebSocket::new(config);
//!
//! // Get WebSocket token from REST API first
//! ws.authenticate().await?;
//!
//! // Start the client
//! ws.start().await?;
//!
//! // Subscribe to execution updates
//! ws.subscribe_executions().await?;
//!
//! // Receive events
//! let mut rx = ws.subscribe_events();
//! while let Ok(event) = rx.recv().await {
//!     match event {
//!         PrivateWsEvent::Execution(exec) => {
//!             println!("Fill: {} {} @ {}", exec.order_id, exec.quantity, exec.price);
//!         }
//!         PrivateWsEvent::OrderStatus(status) => {
//!             println!("Order status: {} -> {:?}", status.order_id, status.status);
//!         }
//!         _ => {}
//!     }
//! }
//! ```

use crate::error::{ExecutionError, Result};
use crate::types::{Fill, OrderSide, OrderStatusEnum};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tokio::time::{Duration, sleep};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Kraken Private WebSocket URL
pub const WS_PRIVATE_URL: &str = "wss://ws-auth.kraken.com/v2";

/// Connection settings
const PING_INTERVAL_SECS: u64 = 30;
const RECONNECT_DELAY_SECS: u64 = 5;
const MAX_RECONNECT_ATTEMPTS: u32 = 10;
const TOKEN_REFRESH_SECS: u64 = 840; // Refresh token every 14 minutes (token valid for 15 min)

// ============================================================================
// WebSocket Message Types
// ============================================================================

/// Private WebSocket request
#[derive(Debug, Clone, Serialize)]
struct PrivateWsRequest {
    method: String,
    params: PrivateWsParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    req_id: Option<u64>,
}

/// Private request parameters
#[derive(Debug, Clone, Serialize)]
struct PrivateWsParams {
    channel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    snap_orders: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    snap_trades: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rate_counter: Option<bool>,
}

/// Private WebSocket response
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum PrivateWsResponse {
    /// Subscription confirmation
    Subscribe {
        method: String,
        #[serde(default)]
        success: bool,
        #[serde(default)]
        result: Option<SubscribeResult>,
        #[serde(default)]
        error: Option<String>,
        #[serde(default)]
        req_id: Option<u64>,
    },
    /// Channel data
    ChannelData {
        channel: String,
        #[serde(rename = "type")]
        update_type: Option<String>,
        data: serde_json::Value,
    },
    /// Heartbeat
    Heartbeat { channel: String },
    /// System status
    Status {
        channel: String,
        data: serde_json::Value,
        #[serde(rename = "type")]
        update_type: Option<String>,
    },
}

/// Subscribe result
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct SubscribeResult {
    channel: String,
    #[serde(default)]
    snapshot: Option<bool>,
}

// ============================================================================
// Execution Data Types
// ============================================================================

/// Raw execution data from Kraken WebSocket
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenExecutionData {
    /// Order ID (Kraken TXID)
    pub order_id: String,
    /// Trade ID
    #[serde(default)]
    pub exec_id: Option<String>,
    /// Execution type (filled, partial, etc.)
    #[serde(default)]
    pub exec_type: Option<String>,
    /// Trading pair (e.g., "BTC/USD")
    #[serde(default)]
    pub symbol: Option<String>,
    /// Side (buy/sell)
    #[serde(default)]
    pub side: Option<String>,
    /// Order type
    #[serde(default)]
    pub order_type: Option<String>,
    /// Fill price (avg_price in Kraken)
    #[serde(default)]
    pub avg_price: Option<String>,
    /// Fill quantity (last_qty for this fill)
    #[serde(default)]
    pub last_qty: Option<String>,
    /// Cumulative filled quantity
    #[serde(default)]
    pub cum_qty: Option<String>,
    /// Order quantity
    #[serde(default)]
    pub order_qty: Option<String>,
    /// Remaining quantity
    #[serde(default)]
    pub leaves_qty: Option<String>,
    /// Fill cost
    #[serde(default)]
    pub cum_cost: Option<String>,
    /// Fee amount
    #[serde(default)]
    pub fee: Option<String>,
    /// Fee currency
    #[serde(default)]
    pub fee_ccy: Option<String>,
    /// Order status
    #[serde(default)]
    pub order_status: Option<String>,
    /// Timestamp
    #[serde(default)]
    pub timestamp: Option<String>,
    /// Limit price
    #[serde(default)]
    pub limit_price: Option<String>,
    /// Stop price
    #[serde(default)]
    pub stop_price: Option<String>,
    /// Time in force
    #[serde(default)]
    pub time_in_force: Option<String>,
    /// Is maker
    #[serde(default)]
    pub is_maker: Option<bool>,
}

/// Parsed execution/fill event
#[derive(Debug, Clone)]
pub struct ExecutionEvent {
    /// Unique execution ID
    pub exec_id: String,
    /// Order ID (Kraken TXID)
    pub order_id: String,
    /// Trading symbol
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Execution type
    pub exec_type: ExecutionType,
    /// This fill's quantity
    pub fill_quantity: Decimal,
    /// This fill's price
    pub fill_price: Decimal,
    /// Cumulative filled quantity
    pub cumulative_quantity: Decimal,
    /// Remaining quantity
    pub remaining_quantity: Decimal,
    /// Fee amount
    pub fee: Decimal,
    /// Fee currency
    pub fee_currency: String,
    /// Is maker order
    pub is_maker: bool,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl ExecutionEvent {
    /// Convert to internal Fill type
    pub fn to_fill(&self) -> Fill {
        Fill {
            id: self.exec_id.clone(),
            order_id: self.order_id.clone(),
            quantity: self.fill_quantity,
            price: self.fill_price,
            fee: self.fee,
            fee_currency: self.fee_currency.clone(),
            side: self.side,
            timestamp: self.timestamp,
            is_maker: self.is_maker,
        }
    }
}

/// Execution type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionType {
    /// New order acknowledged
    New,
    /// Partial fill
    PartialFill,
    /// Full fill
    Fill,
    /// Order cancelled
    Cancelled,
    /// Order expired
    Expired,
    /// Order rejected
    Rejected,
    /// Trade (fill notification)
    Trade,
    /// Unknown
    Unknown,
}

impl FromStr for ExecutionType {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "new" | "pending_new" => Ok(ExecutionType::New),
            "partial" | "partially_filled" => Ok(ExecutionType::PartialFill),
            "filled" | "fill" | "filled_fully" => Ok(ExecutionType::Fill),
            "canceled" | "cancelled" => Ok(ExecutionType::Cancelled),
            "expired" => Ok(ExecutionType::Expired),
            "rejected" => Ok(ExecutionType::Rejected),
            "trade" => Ok(ExecutionType::Trade),
            _ => Ok(ExecutionType::Unknown),
        }
    }
}

/// Order status update event
#[derive(Debug, Clone)]
pub struct OrderStatusEvent {
    /// Order ID (Kraken TXID)
    pub order_id: String,
    /// Trading symbol
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// New order status
    pub status: OrderStatusEnum,
    /// Order quantity
    pub quantity: Decimal,
    /// Filled quantity
    pub filled_quantity: Decimal,
    /// Remaining quantity
    pub remaining_quantity: Decimal,
    /// Average fill price
    pub avg_fill_price: Option<Decimal>,
    /// Limit price
    pub limit_price: Option<Decimal>,
    /// Stop price
    pub stop_price: Option<Decimal>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Balance update event
#[derive(Debug, Clone)]
pub struct BalanceUpdateEvent {
    /// Currency
    pub currency: String,
    /// Available balance
    pub available: Decimal,
    /// Total balance (including held)
    pub total: Decimal,
    /// Held balance
    pub held: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// Events
// ============================================================================

/// Private WebSocket events
#[derive(Debug, Clone)]
pub enum PrivateWsEvent {
    /// Execution/fill event
    Execution(ExecutionEvent),
    /// Order status change
    OrderStatus(OrderStatusEvent),
    /// Balance update
    BalanceUpdate(BalanceUpdateEvent),
    /// Connected to private WebSocket
    Connected,
    /// Disconnected from private WebSocket
    Disconnected,
    /// Authenticated successfully
    Authenticated,
    /// Authentication failed
    AuthenticationFailed(String),
    /// Subscription confirmed
    Subscribed { channel: String },
    /// Error occurred
    Error(String),
}

// ============================================================================
// Configuration
// ============================================================================

/// Kraken Private WebSocket configuration
#[derive(Debug, Clone)]
pub struct KrakenPrivateWsConfig {
    /// API key for REST token retrieval
    pub api_key: String,
    /// API secret for REST token retrieval
    pub api_secret: String,
    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,
    /// Maximum reconnect attempts
    pub max_reconnect_attempts: u32,
    /// Reconnect delay in seconds
    pub reconnect_delay_secs: u64,
    /// Ping interval in seconds
    pub ping_interval_secs: u64,
    /// Token refresh interval in seconds
    pub token_refresh_secs: u64,
    /// Subscribe to order snapshots on connect
    pub snap_orders: bool,
    /// Subscribe to trade history on connect
    pub snap_trades: bool,
}

impl Default for KrakenPrivateWsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_secret: String::new(),
            auto_reconnect: true,
            max_reconnect_attempts: MAX_RECONNECT_ATTEMPTS,
            reconnect_delay_secs: RECONNECT_DELAY_SECS,
            ping_interval_secs: PING_INTERVAL_SECS,
            token_refresh_secs: TOKEN_REFRESH_SECS,
            snap_orders: true,
            snap_trades: false,
        }
    }
}

impl KrakenPrivateWsConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let api_key = std::env::var("KRAKEN_API_KEY").unwrap_or_default();
        let api_secret = std::env::var("KRAKEN_API_SECRET").unwrap_or_default();

        let auto_reconnect = std::env::var("KRAKEN_WS_AUTO_RECONNECT")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let max_reconnect_attempts = std::env::var("KRAKEN_WS_MAX_RECONNECTS")
            .unwrap_or_else(|_| MAX_RECONNECT_ATTEMPTS.to_string())
            .parse()
            .unwrap_or(MAX_RECONNECT_ATTEMPTS);

        let snap_orders = std::env::var("KRAKEN_WS_SNAP_ORDERS")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        Self {
            api_key,
            api_secret,
            auto_reconnect,
            max_reconnect_attempts,
            snap_orders,
            ..Default::default()
        }
    }

    /// Check if credentials are configured
    pub fn has_credentials(&self) -> bool {
        !self.api_key.is_empty() && !self.api_secret.is_empty()
    }
}

// ============================================================================
// WebSocket Client
// ============================================================================

/// Kraken Private WebSocket client for authenticated execution updates
pub struct KrakenPrivateWebSocket {
    config: KrakenPrivateWsConfig,
    event_tx: broadcast::Sender<PrivateWsEvent>,
    /// WebSocket authentication token
    token: Arc<RwLock<Option<String>>>,
    /// Token expiration time
    token_expires_at: Arc<RwLock<Option<DateTime<Utc>>>>,
    /// Is the client running
    is_running: Arc<RwLock<bool>>,
    /// Is the client authenticated
    is_authenticated: Arc<RwLock<bool>>,
    /// Reconnection count
    reconnect_count: Arc<RwLock<u32>>,
    /// Request ID counter
    request_id: Arc<RwLock<u64>>,
    /// Subscribed channels
    subscribed_channels: Arc<RwLock<Vec<String>>>,
    /// Pending order updates (order_id -> latest event)
    pending_orders: Arc<RwLock<HashMap<String, OrderStatusEvent>>>,
}

impl KrakenPrivateWebSocket {
    /// Create a new private WebSocket client
    pub fn new(config: KrakenPrivateWsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(10000);

        Self {
            config,
            event_tx,
            token: Arc::new(RwLock::new(None)),
            token_expires_at: Arc::new(RwLock::new(None)),
            is_running: Arc::new(RwLock::new(false)),
            is_authenticated: Arc::new(RwLock::new(false)),
            reconnect_count: Arc::new(RwLock::new(0)),
            request_id: Arc::new(RwLock::new(0)),
            subscribed_channels: Arc::new(RwLock::new(Vec::new())),
            pending_orders: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Subscribe to events from this client
    pub fn subscribe_events(&self) -> broadcast::Receiver<PrivateWsEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender for forwarding
    pub fn event_sender(&self) -> broadcast::Sender<PrivateWsEvent> {
        self.event_tx.clone()
    }

    /// Check if the client is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Check if the client is authenticated
    pub async fn is_authenticated(&self) -> bool {
        *self.is_authenticated.read().await
    }

    /// Authenticate and obtain WebSocket token
    ///
    /// This must be called before starting the WebSocket connection.
    /// The token is obtained from the Kraken REST API.
    pub async fn authenticate(&self) -> Result<()> {
        if !self.config.has_credentials() {
            return Err(ExecutionError::AuthenticationError(
                "Kraken API credentials not configured".to_string(),
            ));
        }

        info!("Authenticating with Kraken for WebSocket token...");

        // Use the REST client to get a WebSocket token
        let rest_config = super::rest::KrakenRestConfig {
            api_key: self.config.api_key.clone(),
            api_secret: self.config.api_secret.clone(),
            testnet: false,
            custom_url: None,
            timeout_secs: 30,
        };

        let rest_client = super::rest::KrakenRestClient::new(rest_config);
        let token = rest_client.get_websocket_token().await?;

        // Store token with expiration (tokens are valid for 15 minutes)
        let expires_at = Utc::now() + chrono::Duration::seconds(900);
        *self.token.write().await = Some(token);
        *self.token_expires_at.write().await = Some(expires_at);
        *self.is_authenticated.write().await = true;

        info!("WebSocket token obtained, expires at {}", expires_at);
        let _ = self.event_tx.send(PrivateWsEvent::Authenticated);

        Ok(())
    }

    /// Check if the token needs refresh
    #[allow(dead_code)]
    async fn needs_token_refresh(&self) -> bool {
        match *self.token_expires_at.read().await {
            Some(expires_at) => {
                let refresh_threshold =
                    Utc::now() + chrono::Duration::seconds(self.config.token_refresh_secs as i64);
                refresh_threshold >= expires_at
            }
            _ => true,
        }
    }

    /// Refresh the authentication token
    #[allow(dead_code)]
    async fn refresh_token(&self) -> Result<()> {
        info!("Refreshing WebSocket authentication token...");
        self.authenticate().await
    }

    /// Start the WebSocket client
    pub async fn start(&self) -> Result<()> {
        // Check if already running
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(ExecutionError::WebSocketError(
                "Kraken private WebSocket already running".to_string(),
            ));
        }

        // Ensure we have a token
        if self.token.read().await.is_none() {
            drop(is_running);
            self.authenticate().await?;
            is_running = self.is_running.write().await;
        }

        *is_running = true;
        drop(is_running);

        // Clone for the spawned task
        let config = self.config.clone();
        let event_tx = self.event_tx.clone();
        let token = self.token.clone();
        let token_expires_at = self.token_expires_at.clone();
        let is_running = self.is_running.clone();
        let is_authenticated = self.is_authenticated.clone();
        let reconnect_count = self.reconnect_count.clone();
        let request_id = self.request_id.clone();
        let subscribed_channels = self.subscribed_channels.clone();
        let pending_orders = self.pending_orders.clone();

        // Spawn the WebSocket loop
        tokio::spawn(async move {
            Self::run_loop(
                config,
                event_tx,
                token,
                token_expires_at,
                is_running,
                is_authenticated,
                reconnect_count,
                request_id,
                subscribed_channels,
                pending_orders,
            )
            .await;
        });

        info!("Kraken private WebSocket client started");
        Ok(())
    }

    /// Stop the WebSocket client
    pub async fn stop(&self) {
        *self.is_running.write().await = false;
        let _ = self.event_tx.send(PrivateWsEvent::Disconnected);
        info!("Kraken private WebSocket client stopped");
    }

    /// Subscribe to execution updates
    pub async fn subscribe_executions(&self) -> Result<()> {
        let mut channels = self.subscribed_channels.write().await;
        if !channels.contains(&"executions".to_string()) {
            channels.push("executions".to_string());
        }
        Ok(())
    }

    /// Subscribe to balance updates
    pub async fn subscribe_balances(&self) -> Result<()> {
        let mut channels = self.subscribed_channels.write().await;
        if !channels.contains(&"balances".to_string()) {
            channels.push("balances".to_string());
        }
        Ok(())
    }

    /// Get pending order by ID
    pub async fn get_pending_order(&self, order_id: &str) -> Option<OrderStatusEvent> {
        self.pending_orders.read().await.get(order_id).cloned()
    }

    /// Get all pending orders
    pub async fn get_all_pending_orders(&self) -> Vec<OrderStatusEvent> {
        self.pending_orders.read().await.values().cloned().collect()
    }

    /// Main WebSocket loop
    #[allow(clippy::too_many_arguments)]
    async fn run_loop(
        config: KrakenPrivateWsConfig,
        event_tx: broadcast::Sender<PrivateWsEvent>,
        token: Arc<RwLock<Option<String>>>,
        token_expires_at: Arc<RwLock<Option<DateTime<Utc>>>>,
        is_running: Arc<RwLock<bool>>,
        is_authenticated: Arc<RwLock<bool>>,
        reconnect_count: Arc<RwLock<u32>>,
        request_id: Arc<RwLock<u64>>,
        subscribed_channels: Arc<RwLock<Vec<String>>>,
        pending_orders: Arc<RwLock<HashMap<String, OrderStatusEvent>>>,
    ) {
        while *is_running.read().await {
            let current_reconnect = *reconnect_count.read().await;

            if current_reconnect >= config.max_reconnect_attempts {
                error!(
                    "Max reconnection attempts ({}) reached, stopping",
                    config.max_reconnect_attempts
                );
                *is_running.write().await = false;
                let _ = event_tx.send(PrivateWsEvent::Error(
                    "Max reconnection attempts reached".to_string(),
                ));
                break;
            }

            // Check if token needs refresh
            let needs_refresh = {
                match *token_expires_at.read().await {
                    Some(expires_at) => {
                        let refresh_threshold = Utc::now()
                            + chrono::Duration::seconds(config.token_refresh_secs as i64);
                        refresh_threshold >= expires_at
                    }
                    _ => true,
                }
            };

            if needs_refresh {
                info!("Token needs refresh, re-authenticating...");
                // Would need to call authenticate again here
                // For now, just signal the error
                let _ = event_tx.send(PrivateWsEvent::Error(
                    "Token expired, needs re-authentication".to_string(),
                ));
            }

            // Get current token
            let current_token = token.read().await.clone();
            if current_token.is_none() {
                error!("No authentication token available");
                let _ = event_tx.send(PrivateWsEvent::AuthenticationFailed(
                    "No token available".to_string(),
                ));
                break;
            }
            let current_token = current_token.unwrap();

            // Get channels to subscribe
            let channels = subscribed_channels.read().await.clone();

            match Self::connect_and_run(
                &config,
                &event_tx,
                &current_token,
                &channels,
                &request_id,
                &pending_orders,
            )
            .await
            {
                Ok(()) => {
                    // Normal disconnect
                    *reconnect_count.write().await = 0;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    let _ = event_tx.send(PrivateWsEvent::Error(e.to_string()));

                    if config.auto_reconnect {
                        *reconnect_count.write().await += 1;
                        let delay = config.reconnect_delay_secs * (current_reconnect + 1) as u64;
                        warn!("Reconnecting in {} seconds...", delay);
                        sleep(Duration::from_secs(delay)).await;
                    } else {
                        break;
                    }
                }
            }
        }

        *is_running.write().await = false;
        *is_authenticated.write().await = false;
        let _ = event_tx.send(PrivateWsEvent::Disconnected);
    }

    /// Connect and run the WebSocket
    async fn connect_and_run(
        config: &KrakenPrivateWsConfig,
        event_tx: &broadcast::Sender<PrivateWsEvent>,
        token: &str,
        channels: &[String],
        request_id: &Arc<RwLock<u64>>,
        pending_orders: &Arc<RwLock<HashMap<String, OrderStatusEvent>>>,
    ) -> Result<()> {
        info!("Connecting to Kraken private WebSocket...");

        let (ws_stream, _response) = connect_async(WS_PRIVATE_URL)
            .await
            .map_err(|e| ExecutionError::WebSocketError(format!("Connection failed: {}", e)))?;

        info!("Connected to Kraken private WebSocket");
        let _ = event_tx.send(PrivateWsEvent::Connected);

        let (mut write, mut read) = ws_stream.split();

        // Subscribe to channels
        for channel in channels {
            let mut rid = request_id.write().await;
            *rid += 1;
            let req_id = *rid;

            let request = PrivateWsRequest {
                method: "subscribe".to_string(),
                params: PrivateWsParams {
                    channel: channel.clone(),
                    token: Some(token.to_string()),
                    snap_orders: if channel == "executions" {
                        Some(config.snap_orders)
                    } else {
                        None
                    },
                    snap_trades: if channel == "executions" {
                        Some(config.snap_trades)
                    } else {
                        None
                    },
                    rate_counter: None,
                },
                req_id: Some(req_id),
            };

            let msg = serde_json::to_string(&request)
                .map_err(|e| ExecutionError::WebSocketError(format!("Serialize error: {}", e)))?;

            debug!("Subscribing to channel: {}", channel);
            write
                .send(Message::Text(msg.into()))
                .await
                .map_err(|e| ExecutionError::WebSocketError(format!("Send error: {}", e)))?;
        }

        // Setup ping interval
        let ping_interval = Duration::from_secs(config.ping_interval_secs);
        let mut last_ping = std::time::Instant::now();

        // Message loop
        loop {
            // Check if ping is needed
            if last_ping.elapsed() >= ping_interval {
                write
                    .send(Message::Ping(vec![].into()))
                    .await
                    .map_err(|e| ExecutionError::WebSocketError(format!("Ping error: {}", e)))?;
                last_ping = std::time::Instant::now();
            }

            // Wait for next message with timeout
            let msg = tokio::time::timeout(Duration::from_secs(60), read.next()).await;

            match msg {
                Ok(Some(Ok(message))) => match message {
                    Message::Text(text) => {
                        if let Err(e) = Self::handle_message(&text, event_tx, pending_orders).await
                        {
                            warn!("Error handling message: {}", e);
                        }
                    }
                    Message::Ping(data) => {
                        write.send(Message::Pong(data)).await.map_err(|e| {
                            ExecutionError::WebSocketError(format!("Pong error: {}", e))
                        })?;
                    }
                    Message::Pong(_) => {
                        debug!("Received pong");
                    }
                    Message::Close(frame) => {
                        info!("WebSocket closed: {:?}", frame);
                        return Ok(());
                    }
                    _ => {}
                },
                Ok(Some(Err(e))) => {
                    return Err(ExecutionError::WebSocketError(format!(
                        "WebSocket error: {}",
                        e
                    )));
                }
                Ok(None) => {
                    info!("WebSocket stream ended");
                    return Ok(());
                }
                Err(_) => {
                    // Timeout - send ping
                    write
                        .send(Message::Ping(vec![].into()))
                        .await
                        .map_err(|e| {
                            ExecutionError::WebSocketError(format!("Ping error: {}", e))
                        })?;
                    last_ping = std::time::Instant::now();
                }
            }
        }
    }

    /// Handle a WebSocket message
    async fn handle_message(
        text: &str,
        event_tx: &broadcast::Sender<PrivateWsEvent>,
        pending_orders: &Arc<RwLock<HashMap<String, OrderStatusEvent>>>,
    ) -> Result<()> {
        debug!("Received message: {}", text);

        let response: PrivateWsResponse = serde_json::from_str(text)
            .map_err(|e| ExecutionError::ParseError(format!("JSON parse error: {}", e)))?;

        match response {
            PrivateWsResponse::Subscribe {
                method,
                success,
                result,
                error,
                req_id,
            } => {
                if method == "subscribe" {
                    if success {
                        if let Some(result) = result {
                            info!("Subscribed to channel: {}", result.channel);
                            let _ = event_tx.send(PrivateWsEvent::Subscribed {
                                channel: result.channel,
                            });
                        }
                    } else {
                        let err_msg = error.unwrap_or_else(|| "Unknown error".to_string());
                        error!("Subscription failed (req_id: {:?}): {}", req_id, err_msg);
                        let _ = event_tx.send(PrivateWsEvent::Error(format!(
                            "Subscription failed: {}",
                            err_msg
                        )));
                    }
                }
            }
            PrivateWsResponse::ChannelData {
                channel,
                update_type,
                data,
            } => match channel.as_str() {
                "executions" => {
                    Self::handle_executions(data, update_type, event_tx, pending_orders).await?;
                }
                "balances" => {
                    Self::handle_balances(data, event_tx).await?;
                }
                _ => {
                    debug!("Unhandled channel: {}", channel);
                }
            },
            PrivateWsResponse::Heartbeat { channel } => {
                debug!("Heartbeat for channel: {}", channel);
            }
            PrivateWsResponse::Status {
                channel,
                data,
                update_type,
            } => {
                debug!(
                    "Status update for {}: {:?} (type: {:?})",
                    channel, data, update_type
                );
            }
        }

        Ok(())
    }

    /// Handle execution messages
    async fn handle_executions(
        data: serde_json::Value,
        update_type: Option<String>,
        event_tx: &broadcast::Sender<PrivateWsEvent>,
        pending_orders: &Arc<RwLock<HashMap<String, OrderStatusEvent>>>,
    ) -> Result<()> {
        // Executions come as an array
        let executions: Vec<KrakenExecutionData> = serde_json::from_value(data)
            .map_err(|e| ExecutionError::ParseError(format!("Execution parse error: {}", e)))?;

        let is_snapshot = update_type.as_deref() == Some("snapshot");
        if is_snapshot {
            info!(
                "Received execution snapshot with {} items",
                executions.len()
            );
        }

        for exec_data in executions {
            // Parse execution event
            if let Some(event) = Self::parse_execution(&exec_data) {
                // Update pending orders
                let order_status = Self::execution_to_order_status(&exec_data, &event);
                pending_orders
                    .write()
                    .await
                    .insert(event.order_id.clone(), order_status.clone());

                // Emit events
                if event.fill_quantity > Decimal::ZERO {
                    let _ = event_tx.send(PrivateWsEvent::Execution(event));
                }
                let _ = event_tx.send(PrivateWsEvent::OrderStatus(order_status));
            }
        }

        Ok(())
    }

    /// Parse execution data into event
    fn parse_execution(data: &KrakenExecutionData) -> Option<ExecutionEvent> {
        let order_id = data.order_id.clone();
        let symbol = data.symbol.clone().unwrap_or_default();

        let side = match data.side.as_deref() {
            Some("buy") => OrderSide::Buy,
            Some("sell") => OrderSide::Sell,
            _ => OrderSide::Buy, // Default
        };

        let exec_type = data
            .exec_type
            .as_deref()
            .and_then(|s| s.parse().ok())
            .or_else(|| data.order_status.as_deref().and_then(|s| s.parse().ok()))
            .unwrap_or(ExecutionType::Unknown);

        let fill_quantity = data
            .last_qty
            .as_ref()
            .and_then(|s| Decimal::from_str(s).ok())
            .unwrap_or(Decimal::ZERO);

        let fill_price = data
            .avg_price
            .as_ref()
            .and_then(|s| Decimal::from_str(s).ok())
            .unwrap_or(Decimal::ZERO);

        let cumulative_quantity = data
            .cum_qty
            .as_ref()
            .and_then(|s| Decimal::from_str(s).ok())
            .unwrap_or(Decimal::ZERO);

        let remaining_quantity = data
            .leaves_qty
            .as_ref()
            .and_then(|s| Decimal::from_str(s).ok())
            .unwrap_or(Decimal::ZERO);

        let fee = data
            .fee
            .as_ref()
            .and_then(|s| Decimal::from_str(s).ok())
            .unwrap_or(Decimal::ZERO);

        let fee_currency = data.fee_ccy.clone().unwrap_or_else(|| "USD".to_string());

        let timestamp = data
            .timestamp
            .as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let exec_id = data
            .exec_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let is_maker = data.is_maker.unwrap_or(false);

        Some(ExecutionEvent {
            exec_id,
            order_id,
            symbol,
            side,
            exec_type,
            fill_quantity,
            fill_price,
            cumulative_quantity,
            remaining_quantity,
            fee,
            fee_currency,
            is_maker,
            timestamp,
        })
    }

    /// Convert execution to order status event
    fn execution_to_order_status(
        data: &KrakenExecutionData,
        exec: &ExecutionEvent,
    ) -> OrderStatusEvent {
        let status = match data.order_status.as_deref() {
            Some("new") | Some("pending_new") => OrderStatusEnum::New,
            Some("open") => OrderStatusEnum::Submitted,
            Some("partially_filled") => OrderStatusEnum::PartiallyFilled,
            Some("filled") => OrderStatusEnum::Filled,
            Some("canceled") | Some("cancelled") => OrderStatusEnum::Cancelled,
            Some("expired") => OrderStatusEnum::Expired,
            Some("rejected") => OrderStatusEnum::Rejected,
            _ => OrderStatusEnum::Submitted,
        };

        let quantity = data
            .order_qty
            .as_ref()
            .and_then(|s| Decimal::from_str(s).ok())
            .unwrap_or(Decimal::ZERO);

        let limit_price = data
            .limit_price
            .as_ref()
            .and_then(|s| Decimal::from_str(s).ok());

        let stop_price = data
            .stop_price
            .as_ref()
            .and_then(|s| Decimal::from_str(s).ok());

        let avg_fill_price = if exec.cumulative_quantity > Decimal::ZERO {
            Some(exec.fill_price)
        } else {
            None
        };

        OrderStatusEvent {
            order_id: exec.order_id.clone(),
            symbol: exec.symbol.clone(),
            side: exec.side,
            status,
            quantity,
            filled_quantity: exec.cumulative_quantity,
            remaining_quantity: exec.remaining_quantity,
            avg_fill_price,
            limit_price,
            stop_price,
            timestamp: exec.timestamp,
        }
    }

    /// Handle balance messages
    async fn handle_balances(
        data: serde_json::Value,
        event_tx: &broadcast::Sender<PrivateWsEvent>,
    ) -> Result<()> {
        // Balances come as an object with currency keys
        if let Some(obj) = data.as_object() {
            for (currency, balance_data) in obj {
                if let Some(balance) = balance_data.as_object() {
                    let available = balance
                        .get("available")
                        .and_then(|v| v.as_str())
                        .and_then(|s| Decimal::from_str(s).ok())
                        .unwrap_or(Decimal::ZERO);

                    let total = balance
                        .get("balance")
                        .and_then(|v| v.as_str())
                        .and_then(|s| Decimal::from_str(s).ok())
                        .unwrap_or(Decimal::ZERO);

                    let held = total - available;

                    let event = BalanceUpdateEvent {
                        currency: currency.clone(),
                        available,
                        total,
                        held,
                        timestamp: Utc::now(),
                    };

                    let _ = event_tx.send(PrivateWsEvent::BalanceUpdate(event));
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = KrakenPrivateWsConfig::default();
        assert!(config.auto_reconnect);
        assert_eq!(config.max_reconnect_attempts, MAX_RECONNECT_ATTEMPTS);
        assert!(config.snap_orders);
    }

    #[test]
    fn test_config_has_credentials() {
        let mut config = KrakenPrivateWsConfig::default();
        assert!(!config.has_credentials());

        config.api_key = "test_key".to_string();
        config.api_secret = "test_secret".to_string();
        assert!(config.has_credentials());
    }

    #[test]
    fn test_execution_type_parsing() {
        assert_eq!("new".parse::<ExecutionType>().unwrap(), ExecutionType::New);
        assert_eq!(
            "filled".parse::<ExecutionType>().unwrap(),
            ExecutionType::Fill
        );
        assert_eq!(
            "partial".parse::<ExecutionType>().unwrap(),
            ExecutionType::PartialFill
        );
        assert_eq!(
            "canceled".parse::<ExecutionType>().unwrap(),
            ExecutionType::Cancelled
        );
    }

    #[tokio::test]
    async fn test_websocket_creation() {
        let config = KrakenPrivateWsConfig::default();
        let ws = KrakenPrivateWebSocket::new(config);

        assert!(!ws.is_running().await);
        assert!(!ws.is_authenticated().await);
    }

    #[tokio::test]
    async fn test_subscribe_channels() {
        let config = KrakenPrivateWsConfig::default();
        let ws = KrakenPrivateWebSocket::new(config);

        ws.subscribe_executions().await.unwrap();
        ws.subscribe_balances().await.unwrap();

        let channels = ws.subscribed_channels.read().await;
        assert!(channels.contains(&"executions".to_string()));
        assert!(channels.contains(&"balances".to_string()));
    }

    #[test]
    fn test_parse_execution() {
        let data = KrakenExecutionData {
            order_id: "OABCDE-12345-FGHIJK".to_string(),
            exec_id: Some("EXEC123".to_string()),
            exec_type: Some("filled".to_string()),
            symbol: Some("BTC/USD".to_string()),
            side: Some("buy".to_string()),
            order_type: Some("limit".to_string()),
            avg_price: Some("50000.00".to_string()),
            last_qty: Some("0.1".to_string()),
            cum_qty: Some("0.1".to_string()),
            order_qty: Some("0.1".to_string()),
            leaves_qty: Some("0".to_string()),
            cum_cost: Some("5000.00".to_string()),
            fee: Some("5.00".to_string()),
            fee_ccy: Some("USD".to_string()),
            order_status: Some("filled".to_string()),
            timestamp: None,
            limit_price: Some("50000.00".to_string()),
            stop_price: None,
            time_in_force: Some("GTC".to_string()),
            is_maker: Some(false),
        };

        let event = KrakenPrivateWebSocket::parse_execution(&data).unwrap();

        assert_eq!(event.order_id, "OABCDE-12345-FGHIJK");
        assert_eq!(event.symbol, "BTC/USD");
        assert_eq!(event.side, OrderSide::Buy);
        assert_eq!(event.exec_type, ExecutionType::Fill);
        assert_eq!(event.fill_quantity, Decimal::from_str("0.1").unwrap());
        assert_eq!(event.fill_price, Decimal::from_str("50000.00").unwrap());
        assert_eq!(event.fee, Decimal::from_str("5.00").unwrap());
    }

    #[test]
    fn test_execution_to_fill() {
        let event = ExecutionEvent {
            exec_id: "EXEC123".to_string(),
            order_id: "ORDER456".to_string(),
            symbol: "ETH/USD".to_string(),
            side: OrderSide::Sell,
            exec_type: ExecutionType::Fill,
            fill_quantity: Decimal::from_str("1.5").unwrap(),
            fill_price: Decimal::from_str("3000.00").unwrap(),
            cumulative_quantity: Decimal::from_str("1.5").unwrap(),
            remaining_quantity: Decimal::ZERO,
            fee: Decimal::from_str("4.50").unwrap(),
            fee_currency: "USD".to_string(),
            is_maker: true,
            timestamp: Utc::now(),
        };

        let fill = event.to_fill();

        assert_eq!(fill.id, "EXEC123");
        assert_eq!(fill.order_id, "ORDER456");
        assert_eq!(fill.quantity, Decimal::from_str("1.5").unwrap());
        assert_eq!(fill.price, Decimal::from_str("3000.00").unwrap());
        assert_eq!(fill.fee, Decimal::from_str("4.50").unwrap());
        assert_eq!(fill.side, OrderSide::Sell);
        assert!(fill.is_maker);
    }
}
