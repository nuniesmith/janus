//! Binance Private WebSocket Client (User Data Stream)
//!
//! Authenticated WebSocket connection to Binance for real-time execution updates.
//! This module handles order fill notifications, order status changes, and account updates.
//!
//! # User Data Stream
//!
//! Binance's User Data Stream requires a listen key obtained via REST API.
//! The listen key must be kept alive by sending a PUT request every 30 minutes.
//!
//! ## Supported Events
//!
//! - `executionReport`: Order updates and fills
//! - `outboundAccountPosition`: Account balance updates
//! - `balanceUpdate`: Balance changes (deposits/withdrawals)
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::exchanges::binance::private_websocket::{
//!     BinancePrivateWebSocket, BinancePrivateWsConfig, PrivateWsEvent,
//! };
//!
//! let config = BinancePrivateWsConfig::from_env();
//! let ws = BinancePrivateWebSocket::new(config);
//!
//! // Start the client (automatically obtains listen key)
//! ws.start().await?;
//!
//! // Receive events
//! let mut rx = ws.subscribe_events();
//! while let Ok(event) = rx.recv().await {
//!     match event {
//!         PrivateWsEvent::OrderUpdate(update) => {
//!             println!("Order: {} {} @ {}", update.order_id, update.quantity, update.price);
//!         }
//!         PrivateWsEvent::Fill(fill) => {
//!             println!("Fill: {} {} @ {}", fill.order_id, fill.quantity, fill.price);
//!         }
//!         _ => {}
//!     }
//! }
//! ```

use crate::error::{ExecutionError, Result};
use crate::execution::histogram::global_latency_histograms;
use crate::types::{Fill, OrderSide, OrderStatusEnum};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tokio::time::{Duration, Instant, sleep};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};

/// Binance User Data Stream WebSocket URLs
pub const WS_MAINNET_URL: &str = "wss://stream.binance.com:9443/ws";
pub const WS_TESTNET_URL: &str = "wss://testnet.binance.vision/ws";

/// Binance REST API URLs for listen key management
pub const REST_MAINNET_URL: &str = "https://api.binance.com";
pub const REST_TESTNET_URL: &str = "https://testnet.binance.vision";

/// Listen key endpoint
const LISTEN_KEY_ENDPOINT: &str = "/api/v3/userDataStream";

/// Connection settings
const PING_INTERVAL_SECS: u64 = 30;
const RECONNECT_DELAY_SECS: u64 = 5;
const MAX_RECONNECT_ATTEMPTS: u32 = 10;
const LISTEN_KEY_REFRESH_SECS: u64 = 1800; // 30 minutes

// ============================================================================
// WebSocket Message Types
// ============================================================================

/// Execution report from Binance (order updates)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionReport {
    /// Event type ("executionReport")
    #[serde(rename = "e")]
    pub event_type: String,

    /// Event time
    #[serde(rename = "E")]
    pub event_time: i64,

    /// Symbol
    #[serde(rename = "s")]
    pub symbol: String,

    /// Client order ID
    #[serde(rename = "c")]
    pub client_order_id: String,

    /// Side (BUY/SELL)
    #[serde(rename = "S")]
    pub side: String,

    /// Order type (LIMIT, MARKET, etc.)
    #[serde(rename = "o")]
    pub order_type: String,

    /// Time in force
    #[serde(rename = "f")]
    pub time_in_force: String,

    /// Order quantity
    #[serde(rename = "q")]
    pub quantity: String,

    /// Order price
    #[serde(rename = "p")]
    pub price: String,

    /// Stop price
    #[serde(rename = "P")]
    pub stop_price: String,

    /// Iceberg quantity
    #[serde(rename = "F")]
    pub iceberg_qty: String,

    /// Order list ID
    #[serde(rename = "g")]
    pub order_list_id: i64,

    /// Original client order ID (for cancels)
    #[serde(rename = "C")]
    pub orig_client_order_id: String,

    /// Current execution type (NEW, TRADE, CANCELED, etc.)
    #[serde(rename = "x")]
    pub execution_type: String,

    /// Current order status
    #[serde(rename = "X")]
    pub order_status: String,

    /// Order reject reason
    #[serde(rename = "r")]
    pub reject_reason: String,

    /// Order ID
    #[serde(rename = "i")]
    pub order_id: i64,

    /// Last executed quantity (fill qty)
    #[serde(rename = "l")]
    pub last_executed_qty: String,

    /// Cumulative filled quantity
    #[serde(rename = "z")]
    pub cumulative_qty: String,

    /// Last executed price (fill price)
    #[serde(rename = "L")]
    pub last_executed_price: String,

    /// Commission amount
    #[serde(rename = "n")]
    pub commission: String,

    /// Commission asset
    #[serde(rename = "N")]
    pub commission_asset: Option<String>,

    /// Transaction time
    #[serde(rename = "T")]
    pub transaction_time: i64,

    /// Trade ID
    #[serde(rename = "t")]
    pub trade_id: i64,

    /// Is the order on the book?
    #[serde(rename = "w")]
    pub is_on_book: bool,

    /// Is this a maker trade?
    #[serde(rename = "m")]
    pub is_maker: bool,

    /// Order creation time
    #[serde(rename = "O")]
    pub order_creation_time: i64,

    /// Cumulative quote asset quantity
    #[serde(rename = "Z")]
    pub cumulative_quote_qty: String,

    /// Last quote asset transacted quantity
    #[serde(rename = "Y")]
    pub last_quote_qty: String,

    /// Quote order quantity
    #[serde(rename = "Q")]
    pub quote_order_qty: String,
}

/// Account position update
#[derive(Debug, Clone, Deserialize)]
pub struct AccountPositionUpdate {
    /// Event type ("outboundAccountPosition")
    #[serde(rename = "e")]
    pub event_type: String,

    /// Event time
    #[serde(rename = "E")]
    pub event_time: i64,

    /// Last update time
    #[serde(rename = "u")]
    pub last_update_time: i64,

    /// Balances
    #[serde(rename = "B")]
    pub balances: Vec<BalanceInfo>,
}

/// Balance information
#[derive(Debug, Clone, Deserialize)]
pub struct BalanceInfo {
    /// Asset
    #[serde(rename = "a")]
    pub asset: String,

    /// Free balance
    #[serde(rename = "f")]
    pub free: String,

    /// Locked balance
    #[serde(rename = "l")]
    pub locked: String,
}

/// Balance update (deposits/withdrawals)
#[derive(Debug, Clone, Deserialize)]
pub struct BalanceUpdateEvent {
    /// Event type ("balanceUpdate")
    #[serde(rename = "e")]
    pub event_type: String,

    /// Event time
    #[serde(rename = "E")]
    pub event_time: i64,

    /// Asset
    #[serde(rename = "a")]
    pub asset: String,

    /// Balance delta
    #[serde(rename = "d")]
    pub delta: String,

    /// Clear time
    #[serde(rename = "T")]
    pub clear_time: i64,
}

/// Listen key response from REST API
#[derive(Debug, Clone, Deserialize)]
pub struct ListenKeyResponse {
    #[serde(rename = "listenKey")]
    pub listen_key: String,
}

// ============================================================================
// Public Event Types
// ============================================================================

/// Order update event (parsed from execution report)
#[derive(Debug, Clone)]
pub struct OrderUpdateEvent {
    /// Symbol
    pub symbol: String,
    /// Order ID (exchange)
    pub order_id: String,
    /// Client order ID
    pub client_order_id: String,
    /// Order status
    pub status: OrderStatusEnum,
    /// Side
    pub side: OrderSide,
    /// Original quantity
    pub quantity: Decimal,
    /// Filled quantity
    pub filled_quantity: Decimal,
    /// Average fill price
    pub average_price: Option<Decimal>,
    /// Order price (for limit orders)
    pub price: Option<Decimal>,
    /// Execution type (NEW, TRADE, CANCELED, etc.)
    pub execution_type: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Balance update event
#[derive(Debug, Clone)]
pub struct AccountBalanceUpdate {
    /// Asset
    pub asset: String,
    /// Free balance
    pub free: Decimal,
    /// Locked balance
    pub locked: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Events emitted by the private WebSocket
#[derive(Debug, Clone)]
pub enum PrivateWsEvent {
    /// Order status update
    OrderUpdate(OrderUpdateEvent),
    /// Fill/trade event
    Fill(Fill),
    /// Account balance update
    BalanceUpdate(AccountBalanceUpdate),
    /// Connected to WebSocket
    Connected,
    /// Disconnected from WebSocket
    Disconnected,
    /// Listen key obtained
    Authenticated,
    /// Authentication failed
    AuthenticationFailed(String),
    /// Error occurred
    Error(String),
}

// ============================================================================
// Configuration
// ============================================================================

/// Private WebSocket configuration
#[derive(Debug, Clone)]
pub struct BinancePrivateWsConfig {
    /// API key
    pub api_key: String,
    /// API secret (needed for listen key refresh)
    pub api_secret: String,
    /// Use testnet
    pub testnet: bool,
    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,
    /// Event buffer size
    pub buffer_size: usize,
}

impl Default for BinancePrivateWsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_secret: String::new(),
            testnet: false,
            auto_reconnect: true,
            buffer_size: 1000,
        }
    }
}

impl BinancePrivateWsConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let api_key = std::env::var("BINANCE_API_KEY").unwrap_or_default();
        let api_secret = std::env::var("BINANCE_API_SECRET").unwrap_or_default();
        let testnet = std::env::var("BINANCE_TESTNET")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);

        Self {
            api_key,
            api_secret,
            testnet,
            ..Default::default()
        }
    }

    /// Check if credentials are configured
    pub fn has_credentials(&self) -> bool {
        !self.api_key.is_empty()
    }

    /// Get REST API URL
    pub fn rest_url(&self) -> &'static str {
        if self.testnet {
            REST_TESTNET_URL
        } else {
            REST_MAINNET_URL
        }
    }

    /// Get WebSocket URL
    pub fn ws_url(&self) -> &'static str {
        if self.testnet {
            WS_TESTNET_URL
        } else {
            WS_MAINNET_URL
        }
    }
}

// ============================================================================
// Private WebSocket Client
// ============================================================================

/// Binance Private WebSocket client for user data stream
pub struct BinancePrivateWebSocket {
    /// Configuration
    config: BinancePrivateWsConfig,
    /// HTTP client for listen key management
    http_client: Client,
    /// Current listen key
    listen_key: Arc<RwLock<Option<String>>>,
    /// Event broadcaster
    event_tx: broadcast::Sender<PrivateWsEvent>,
    /// Is the client running
    is_running: Arc<RwLock<bool>>,
    /// Reconnect counter
    reconnect_count: Arc<RwLock<u32>>,
    /// Last listen key refresh time
    last_key_refresh: Arc<RwLock<Option<Instant>>>,
}

impl BinancePrivateWebSocket {
    /// Create a new private WebSocket client
    pub fn new(config: BinancePrivateWsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.buffer_size);

        Self {
            config,
            http_client: Client::new(),
            listen_key: Arc::new(RwLock::new(None)),
            event_tx,
            is_running: Arc::new(RwLock::new(false)),
            reconnect_count: Arc::new(RwLock::new(0)),
            last_key_refresh: Arc::new(RwLock::new(None)),
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self::new(BinancePrivateWsConfig::from_env())
    }

    /// Subscribe to events
    pub fn subscribe_events(&self) -> broadcast::Receiver<PrivateWsEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender (for external broadcast)
    pub fn event_sender(&self) -> broadcast::Sender<PrivateWsEvent> {
        self.event_tx.clone()
    }

    /// Check if the client is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get current listen key
    pub async fn get_listen_key(&self) -> Option<String> {
        self.listen_key.read().await.clone()
    }

    /// Obtain a new listen key from Binance REST API
    pub async fn create_listen_key(&self) -> Result<String> {
        if !self.config.has_credentials() {
            return Err(ExecutionError::AuthenticationError(
                "API key not configured".to_string(),
            ));
        }

        let url = format!("{}{}", self.config.rest_url(), LISTEN_KEY_ENDPOINT);

        let response = self
            .http_client
            .post(&url)
            .header("X-MBX-APIKEY", &self.config.api_key)
            .send()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        if !status.is_success() {
            return Err(ExecutionError::AuthenticationError(format!(
                "Failed to create listen key: {} - {}",
                status, body
            )));
        }

        let key_response: ListenKeyResponse = serde_json::from_str(&body)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        info!("Obtained new listen key");
        Ok(key_response.listen_key)
    }

    /// Keep the listen key alive (must be called every 30 minutes)
    pub async fn keepalive_listen_key(&self) -> Result<()> {
        let listen_key = self.listen_key.read().await.clone();

        let listen_key = listen_key
            .ok_or_else(|| ExecutionError::Internal("No listen key to keep alive".to_string()))?;

        let url = format!("{}{}", self.config.rest_url(), LISTEN_KEY_ENDPOINT);

        let response = self
            .http_client
            .put(&url)
            .header("X-MBX-APIKEY", &self.config.api_key)
            .query(&[("listenKey", &listen_key)])
            .send()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ExecutionError::Network(format!(
                "Failed to keepalive listen key: {}",
                body
            )));
        }

        *self.last_key_refresh.write().await = Some(Instant::now());
        debug!("Listen key keepalive successful");
        Ok(())
    }

    /// Close/delete the listen key
    pub async fn close_listen_key(&self) -> Result<()> {
        let listen_key = self.listen_key.read().await.clone();

        if let Some(key) = listen_key {
            let url = format!("{}{}", self.config.rest_url(), LISTEN_KEY_ENDPOINT);

            let _ = self
                .http_client
                .delete(&url)
                .header("X-MBX-APIKEY", &self.config.api_key)
                .query(&[("listenKey", &key)])
                .send()
                .await;

            *self.listen_key.write().await = None;
            info!("Listen key closed");
        }

        Ok(())
    }

    /// Start the WebSocket client
    pub async fn start(&self) -> Result<()> {
        if !self.config.has_credentials() {
            return Err(ExecutionError::AuthenticationError(
                "API key not configured for private WebSocket".to_string(),
            ));
        }

        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(ExecutionError::Internal(
                "Private WebSocket already running".to_string(),
            ));
        }

        info!("Starting Binance private WebSocket...");

        // Obtain listen key
        let listen_key = self.create_listen_key().await?;
        *self.listen_key.write().await = Some(listen_key.clone());
        *self.last_key_refresh.write().await = Some(Instant::now());

        *is_running = true;
        drop(is_running);

        let _ = self.event_tx.send(PrivateWsEvent::Authenticated);

        // Spawn the main WebSocket loop
        let config = self.config.clone();
        let listen_key_arc = self.listen_key.clone();
        let event_tx = self.event_tx.clone();
        let is_running = self.is_running.clone();
        let reconnect_count = self.reconnect_count.clone();
        let last_key_refresh = self.last_key_refresh.clone();
        let http_client = self.http_client.clone();

        tokio::spawn(async move {
            Self::run_loop(
                config,
                listen_key_arc,
                event_tx,
                is_running,
                reconnect_count,
                last_key_refresh,
                http_client,
            )
            .await;
        });

        info!("Binance private WebSocket started");
        Ok(())
    }

    /// Stop the WebSocket client
    pub async fn stop(&self) {
        info!("Stopping Binance private WebSocket...");
        *self.is_running.write().await = false;
        let _ = self.close_listen_key().await;
        info!("Binance private WebSocket stopped");
    }

    /// Main WebSocket loop with reconnection
    async fn run_loop(
        config: BinancePrivateWsConfig,
        listen_key: Arc<RwLock<Option<String>>>,
        event_tx: broadcast::Sender<PrivateWsEvent>,
        is_running: Arc<RwLock<bool>>,
        reconnect_count: Arc<RwLock<u32>>,
        last_key_refresh: Arc<RwLock<Option<Instant>>>,
        http_client: Client,
    ) {
        while *is_running.read().await {
            let current_key = listen_key.read().await.clone();

            if current_key.is_none() {
                warn!("No listen key available, stopping...");
                break;
            }

            let ws_url = format!("{}/{}", config.ws_url(), current_key.unwrap());

            match Self::connect_and_run(
                &ws_url,
                &config,
                &listen_key,
                &event_tx,
                &is_running,
                &last_key_refresh,
                &http_client,
            )
            .await
            {
                Ok(_) => {
                    info!("WebSocket connection closed normally");
                    *reconnect_count.write().await = 0;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    let _ = event_tx.send(PrivateWsEvent::Error(e.to_string()));
                }
            }

            let _ = event_tx.send(PrivateWsEvent::Disconnected);

            // Reconnect logic
            if *is_running.read().await && config.auto_reconnect {
                let mut count = reconnect_count.write().await;
                *count += 1;

                if *count > MAX_RECONNECT_ATTEMPTS {
                    error!("Max reconnect attempts reached, stopping");
                    *is_running.write().await = false;
                    break;
                }

                let delay = RECONNECT_DELAY_SECS * (*count as u64).min(6);
                info!(
                    "Reconnecting in {} seconds (attempt {}/{})",
                    delay, *count, MAX_RECONNECT_ATTEMPTS
                );
                drop(count);

                sleep(Duration::from_secs(delay)).await;

                // Refresh listen key before reconnecting
                if let Ok(new_key) = Self::refresh_listen_key(&config, &http_client).await {
                    *listen_key.write().await = Some(new_key);
                    *last_key_refresh.write().await = Some(Instant::now());
                }
            } else {
                break;
            }
        }

        info!("Private WebSocket run loop ended");
    }

    /// Connect and run the WebSocket
    async fn connect_and_run(
        url: &str,
        config: &BinancePrivateWsConfig,
        listen_key: &Arc<RwLock<Option<String>>>,
        event_tx: &broadcast::Sender<PrivateWsEvent>,
        is_running: &Arc<RwLock<bool>>,
        last_key_refresh: &Arc<RwLock<Option<Instant>>>,
        http_client: &Client,
    ) -> Result<()> {
        info!("Connecting to Binance private WebSocket...");

        let (ws_stream, _) = connect_async(url)
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let (mut write, mut read) = ws_stream.split();

        info!("Connected to Binance private WebSocket");
        let _ = event_tx.send(PrivateWsEvent::Connected);

        let mut ping_interval = tokio::time::interval(Duration::from_secs(PING_INTERVAL_SECS));
        let mut keepalive_interval =
            tokio::time::interval(Duration::from_secs(LISTEN_KEY_REFRESH_SECS));

        loop {
            tokio::select! {
                // Check if we should stop
                _ = async {
                    while *is_running.read().await {
                        sleep(Duration::from_millis(100)).await;
                    }
                } => {
                    info!("Stop signal received");
                    break;
                }

                // Handle incoming messages
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            Self::handle_message(&text, event_tx);
                        }
                        Some(Ok(Message::Ping(data))) => {
                            if let Err(e) = write.send(Message::Pong(data)).await {
                                error!("Failed to send pong: {}", e);
                                break;
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("Received close frame");
                            break;
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                        None => {
                            info!("WebSocket stream ended");
                            break;
                        }
                        _ => {}
                    }
                }

                // Send periodic ping
                _ = ping_interval.tick() => {
                    if let Err(e) = write.send(Message::Ping(Bytes::new())).await {
                        error!("Failed to send ping: {}", e);
                        break;
                    }
                }

                // Keep listen key alive
                _ = keepalive_interval.tick() => {
                    if let Err(e) = Self::do_keepalive(config, listen_key, last_key_refresh, http_client).await {
                        warn!("Listen key keepalive failed: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Refresh listen key
    async fn refresh_listen_key(
        config: &BinancePrivateWsConfig,
        http_client: &Client,
    ) -> Result<String> {
        let url = format!("{}{}", config.rest_url(), LISTEN_KEY_ENDPOINT);

        let response = http_client
            .post(&url)
            .header("X-MBX-APIKEY", &config.api_key)
            .send()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let body = response
            .text()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let key_response: ListenKeyResponse = serde_json::from_str(&body)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        info!("Listen key refreshed");
        Ok(key_response.listen_key)
    }

    /// Do keepalive for listen key
    async fn do_keepalive(
        config: &BinancePrivateWsConfig,
        listen_key: &Arc<RwLock<Option<String>>>,
        last_refresh: &Arc<RwLock<Option<Instant>>>,
        http_client: &Client,
    ) -> Result<()> {
        let key = listen_key.read().await.clone();

        if let Some(key) = key {
            let url = format!("{}{}", config.rest_url(), LISTEN_KEY_ENDPOINT);

            let response = http_client
                .put(&url)
                .header("X-MBX-APIKEY", &config.api_key)
                .query(&[("listenKey", &key)])
                .send()
                .await
                .map_err(|e| ExecutionError::Network(e.to_string()))?;

            if !response.status().is_success() {
                return Err(ExecutionError::Network("Keepalive failed".to_string()));
            }

            *last_refresh.write().await = Some(Instant::now());
            debug!("Listen key keepalive successful");
        }

        Ok(())
    }

    /// Handle incoming WebSocket message
    fn handle_message(text: &str, event_tx: &broadcast::Sender<PrivateWsEvent>) {
        let start = std::time::Instant::now();
        let histograms = global_latency_histograms();

        // Try to parse the event type first
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
            let event_type = json.get("e").and_then(|v| v.as_str()).unwrap_or("");

            match event_type {
                "executionReport" => {
                    if let Ok(report) = serde_json::from_str::<ExecutionReport>(text) {
                        // Always emit order update
                        let order_update = Self::parse_order_update(&report);
                        let _ = event_tx.send(PrivateWsEvent::OrderUpdate(order_update));

                        // Emit fill if this is a trade execution
                        if report.execution_type == "TRADE" {
                            if let Some(fill) = Self::parse_fill(&report) {
                                let _ = event_tx.send(PrivateWsEvent::Fill(fill));
                            }
                        }

                        // Record WebSocket message processing latency
                        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                        histograms.record_websocket_message("binance_private", duration_ms);
                    }
                }
                "outboundAccountPosition" => {
                    if let Ok(account) = serde_json::from_str::<AccountPositionUpdate>(text) {
                        for balance in &account.balances {
                            let update = AccountBalanceUpdate {
                                asset: balance.asset.clone(),
                                free: Decimal::from_str(&balance.free).unwrap_or_default(),
                                locked: Decimal::from_str(&balance.locked).unwrap_or_default(),
                                timestamp: DateTime::from_timestamp_millis(account.event_time)
                                    .unwrap_or_else(Utc::now),
                            };
                            let _ = event_tx.send(PrivateWsEvent::BalanceUpdate(update));
                        }
                    }
                }
                "balanceUpdate" => {
                    // Balance update from deposits/withdrawals
                    debug!("Balance update received: {}", text);
                    // Record WebSocket message processing latency
                    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                    histograms.record_websocket_message("binance_private", duration_ms);
                }
                _ => {
                    debug!("Unknown event type: {}", event_type);
                }
            }
        }
    }

    /// Parse order update from execution report
    fn parse_order_update(report: &ExecutionReport) -> OrderUpdateEvent {
        let status = Self::parse_order_status(&report.order_status);
        let side = Self::parse_side(&report.side);

        let quantity = Decimal::from_str(&report.quantity).unwrap_or_default();
        let filled_quantity = Decimal::from_str(&report.cumulative_qty).unwrap_or_default();
        let price = Decimal::from_str(&report.price)
            .ok()
            .filter(|p| *p > Decimal::ZERO);

        // Calculate average price from cumulative quote qty
        let average_price = if filled_quantity > Decimal::ZERO {
            Decimal::from_str(&report.cumulative_quote_qty)
                .ok()
                .map(|q| q / filled_quantity)
        } else {
            None
        };

        OrderUpdateEvent {
            symbol: Self::normalize_symbol(&report.symbol),
            order_id: report.order_id.to_string(),
            client_order_id: report.client_order_id.clone(),
            status,
            side,
            quantity,
            filled_quantity,
            average_price,
            price,
            execution_type: report.execution_type.clone(),
            timestamp: DateTime::from_timestamp_millis(report.event_time).unwrap_or_else(Utc::now),
        }
    }

    /// Parse fill from execution report
    fn parse_fill(report: &ExecutionReport) -> Option<Fill> {
        let fill_qty = Decimal::from_str(&report.last_executed_qty).ok()?;
        let fill_price = Decimal::from_str(&report.last_executed_price).ok()?;

        if fill_qty <= Decimal::ZERO {
            return None;
        }

        let fee = Decimal::from_str(&report.commission).unwrap_or_default();
        let fee_currency = report
            .commission_asset
            .clone()
            .unwrap_or_else(|| "USDT".to_string());

        Some(Fill {
            id: report.trade_id.to_string(),
            order_id: report.order_id.to_string(),
            quantity: fill_qty,
            price: fill_price,
            fee,
            fee_currency,
            side: Self::parse_side(&report.side),
            timestamp: DateTime::from_timestamp_millis(report.transaction_time)
                .unwrap_or_else(Utc::now),
            is_maker: report.is_maker,
        })
    }

    /// Parse order status
    fn parse_order_status(status: &str) -> OrderStatusEnum {
        match status {
            "NEW" => OrderStatusEnum::New,
            "PARTIALLY_FILLED" => OrderStatusEnum::PartiallyFilled,
            "FILLED" => OrderStatusEnum::Filled,
            "CANCELED" => OrderStatusEnum::Cancelled,
            "PENDING_CANCEL" => OrderStatusEnum::PendingCancel,
            "REJECTED" => OrderStatusEnum::Rejected,
            "EXPIRED" => OrderStatusEnum::Expired,
            _ => OrderStatusEnum::New,
        }
    }

    /// Parse side
    fn parse_side(side: &str) -> OrderSide {
        match side {
            "BUY" => OrderSide::Buy,
            "SELL" => OrderSide::Sell,
            _ => OrderSide::Buy,
        }
    }

    /// Normalize Binance symbol to standard format
    fn normalize_symbol(binance_symbol: &str) -> String {
        let upper = binance_symbol.to_uppercase();
        let quotes = ["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"];

        for quote in quotes {
            if upper.ends_with(quote) {
                let base = &upper[..upper.len() - quote.len()];
                if !base.is_empty() {
                    return format!("{}/{}", base, quote);
                }
            }
        }

        upper
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
        let config = BinancePrivateWsConfig::default();
        assert!(config.api_key.is_empty());
        assert!(config.api_secret.is_empty());
        assert!(!config.testnet);
        assert!(config.auto_reconnect);
    }

    #[test]
    fn test_config_has_credentials() {
        let mut config = BinancePrivateWsConfig::default();
        assert!(!config.has_credentials());

        config.api_key = "test_key".to_string();
        assert!(config.has_credentials());
    }

    #[test]
    fn test_config_urls() {
        let mut config = BinancePrivateWsConfig::default();
        assert_eq!(config.rest_url(), REST_MAINNET_URL);
        assert_eq!(config.ws_url(), WS_MAINNET_URL);

        config.testnet = true;
        assert_eq!(config.rest_url(), REST_TESTNET_URL);
        assert_eq!(config.ws_url(), WS_TESTNET_URL);
    }

    #[tokio::test]
    async fn test_websocket_creation() {
        let config = BinancePrivateWsConfig::default();
        let ws = BinancePrivateWebSocket::new(config);

        assert!(!ws.is_running().await);
        assert!(ws.get_listen_key().await.is_none());
    }

    #[test]
    fn test_parse_order_status() {
        assert_eq!(
            BinancePrivateWebSocket::parse_order_status("NEW"),
            OrderStatusEnum::New
        );
        assert_eq!(
            BinancePrivateWebSocket::parse_order_status("FILLED"),
            OrderStatusEnum::Filled
        );
        assert_eq!(
            BinancePrivateWebSocket::parse_order_status("CANCELED"),
            OrderStatusEnum::Cancelled
        );
        assert_eq!(
            BinancePrivateWebSocket::parse_order_status("PARTIALLY_FILLED"),
            OrderStatusEnum::PartiallyFilled
        );
    }

    #[test]
    fn test_parse_side() {
        assert_eq!(BinancePrivateWebSocket::parse_side("BUY"), OrderSide::Buy);
        assert_eq!(BinancePrivateWebSocket::parse_side("SELL"), OrderSide::Sell);
    }

    #[test]
    fn test_normalize_symbol() {
        assert_eq!(
            BinancePrivateWebSocket::normalize_symbol("BTCUSDT"),
            "BTC/USDT"
        );
        assert_eq!(
            BinancePrivateWebSocket::normalize_symbol("ETHUSDC"),
            "ETH/USDC"
        );
        assert_eq!(
            BinancePrivateWebSocket::normalize_symbol("btcusdt"),
            "BTC/USDT"
        );
    }

    #[tokio::test]
    async fn test_subscribe_events() {
        let config = BinancePrivateWsConfig::default();
        let ws = BinancePrivateWebSocket::new(config);

        let _rx = ws.subscribe_events();
        // Just verify we can create a subscriber
    }

    #[test]
    fn test_parse_execution_report() {
        let report = ExecutionReport {
            event_type: "executionReport".to_string(),
            event_time: 1699999999999,
            symbol: "BTCUSDT".to_string(),
            client_order_id: "client123".to_string(),
            side: "BUY".to_string(),
            order_type: "LIMIT".to_string(),
            time_in_force: "GTC".to_string(),
            quantity: "1.0".to_string(),
            price: "50000".to_string(),
            stop_price: "0".to_string(),
            iceberg_qty: "0".to_string(),
            order_list_id: -1,
            orig_client_order_id: "".to_string(),
            execution_type: "NEW".to_string(),
            order_status: "NEW".to_string(),
            reject_reason: "NONE".to_string(),
            order_id: 123456,
            last_executed_qty: "0".to_string(),
            cumulative_qty: "0".to_string(),
            last_executed_price: "0".to_string(),
            commission: "0".to_string(),
            commission_asset: Some("USDT".to_string()),
            transaction_time: 1699999999999,
            trade_id: -1,
            is_on_book: true,
            is_maker: false,
            order_creation_time: 1699999999999,
            cumulative_quote_qty: "0".to_string(),
            last_quote_qty: "0".to_string(),
            quote_order_qty: "0".to_string(),
        };

        let update = BinancePrivateWebSocket::parse_order_update(&report);
        assert_eq!(update.symbol, "BTC/USDT");
        assert_eq!(update.order_id, "123456");
        assert_eq!(update.status, OrderStatusEnum::New);
        assert_eq!(update.side, OrderSide::Buy);
    }

    #[test]
    fn test_parse_fill_from_trade() {
        let report = ExecutionReport {
            event_type: "executionReport".to_string(),
            event_time: 1699999999999,
            symbol: "BTCUSDT".to_string(),
            client_order_id: "client123".to_string(),
            side: "BUY".to_string(),
            order_type: "LIMIT".to_string(),
            time_in_force: "GTC".to_string(),
            quantity: "1.0".to_string(),
            price: "50000".to_string(),
            stop_price: "0".to_string(),
            iceberg_qty: "0".to_string(),
            order_list_id: -1,
            orig_client_order_id: "".to_string(),
            execution_type: "TRADE".to_string(),
            order_status: "FILLED".to_string(),
            reject_reason: "NONE".to_string(),
            order_id: 123456,
            last_executed_qty: "0.5".to_string(),
            cumulative_qty: "0.5".to_string(),
            last_executed_price: "49999".to_string(),
            commission: "0.025".to_string(),
            commission_asset: Some("USDT".to_string()),
            transaction_time: 1699999999999,
            trade_id: 789,
            is_on_book: false,
            is_maker: true,
            order_creation_time: 1699999999999,
            cumulative_quote_qty: "24999.5".to_string(),
            last_quote_qty: "24999.5".to_string(),
            quote_order_qty: "0".to_string(),
        };

        let fill = BinancePrivateWebSocket::parse_fill(&report);
        assert!(fill.is_some());

        let fill = fill.unwrap();
        assert_eq!(fill.order_id, "123456");
        assert_eq!(fill.quantity, Decimal::from_str("0.5").unwrap());
        assert_eq!(fill.price, Decimal::from(49999));
        assert_eq!(fill.fee, Decimal::from_str("0.025").unwrap());
        assert_eq!(fill.fee_currency, "USDT");
        assert!(fill.is_maker);
    }
}
