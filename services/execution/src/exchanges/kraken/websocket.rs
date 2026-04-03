//! Kraken WebSocket Client
//!
//! Real-time market data streaming from Kraken's WebSocket API v2.
//! This is FREE and does NOT require any API keys for public market data.
//!
//! # WebSocket API v2
//!
//! Kraken's v2 WebSocket API uses JSON-RPC style messages:
//!
//! ## Subscribe Request
//! ```json
//! {
//!     "method": "subscribe",
//!     "params": {
//!         "channel": "ticker",
//!         "symbol": ["BTC/USD", "ETH/USD"]
//!     }
//! }
//! ```
//!
//! ## Response Format
//! ```json
//! {
//!     "channel": "ticker",
//!     "type": "update",
//!     "data": [{
//!         "symbol": "BTC/USD",
//!         "bid": 67250.0,
//!         "ask": 67255.0,
//!         ...
//!     }]
//! }
//! ```

use crate::error::{ExecutionError, Result};
use crate::exchanges::market_data::{
    Candle, ExchangeId, MarketDataEvent, Ticker, Trade, TradeSide,
};
use crate::exchanges::metrics::{ExchangeMetrics, register_exchange};
use crate::execution::histogram::global_latency_histograms;
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, broadcast};
use tokio::time::{Duration, sleep};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};

/// Kraken WebSocket URLs
pub const WS_PUBLIC_URL: &str = "wss://ws.kraken.com/v2";
pub const WS_PRIVATE_URL: &str = "wss://ws-auth.kraken.com/v2";

/// Connection settings
const PING_INTERVAL_SECS: u64 = 30;
const RECONNECT_DELAY_SECS: u64 = 5;
const MAX_RECONNECT_ATTEMPTS: u32 = 10;

// ============================================================================
// WebSocket Message Types (Kraken v2 API)
// ============================================================================

/// Kraken v2 WebSocket request
#[derive(Debug, Clone, Serialize)]
struct WsRequest {
    method: String,
    params: WsParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    req_id: Option<u64>,
}

/// Request parameters
#[derive(Debug, Clone, Serialize)]
struct WsParams {
    channel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depth: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interval: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<bool>,
}

/// Kraken v2 WebSocket response (generic)
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum WsResponse {
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
    /// Channel data update
    ChannelData {
        channel: String,
        #[serde(rename = "type")]
        update_type: String,
        data: serde_json::Value,
    },
    /// Heartbeat
    Heartbeat { channel: String },
    /// Status message
    Status {
        channel: String,
        data: Vec<StatusData>,
        #[serde(rename = "type")]
        update_type: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct SubscribeResult {
    channel: String,
    #[serde(default)]
    symbol: Option<String>,
    #[serde(default)]
    snapshot: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct StatusData {
    #[serde(default)]
    api_version: Option<String>,
    #[serde(default)]
    connection_id: Option<u64>,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    version: Option<String>,
}

// ============================================================================
// Ticker Data
// ============================================================================

/// Raw ticker data from Kraken
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenTickerData {
    pub symbol: String,
    pub bid: f64,
    pub bid_qty: f64,
    pub ask: f64,
    pub ask_qty: f64,
    pub last: f64,
    pub volume: f64,
    pub vwap: f64,
    pub low: f64,
    pub high: f64,
    pub change: f64,
    pub change_pct: f64,
}

/// Parsed Kraken ticker
#[derive(Debug, Clone)]
pub struct KrakenTicker {
    pub symbol: String,
    pub bid: Decimal,
    pub bid_qty: Decimal,
    pub ask: Decimal,
    pub ask_qty: Decimal,
    pub last: Decimal,
    pub volume_24h: Decimal,
    pub vwap: Decimal,
    pub low_24h: Decimal,
    pub high_24h: Decimal,
    pub change_24h: Decimal,
    pub change_pct_24h: Decimal,
    pub timestamp: DateTime<Utc>,
}

impl From<KrakenTickerData> for KrakenTicker {
    fn from(data: KrakenTickerData) -> Self {
        Self {
            symbol: data.symbol,
            bid: Decimal::try_from(data.bid).unwrap_or_default(),
            bid_qty: Decimal::try_from(data.bid_qty).unwrap_or_default(),
            ask: Decimal::try_from(data.ask).unwrap_or_default(),
            ask_qty: Decimal::try_from(data.ask_qty).unwrap_or_default(),
            last: Decimal::try_from(data.last).unwrap_or_default(),
            volume_24h: Decimal::try_from(data.volume).unwrap_or_default(),
            vwap: Decimal::try_from(data.vwap).unwrap_or_default(),
            low_24h: Decimal::try_from(data.low).unwrap_or_default(),
            high_24h: Decimal::try_from(data.high).unwrap_or_default(),
            change_24h: Decimal::try_from(data.change).unwrap_or_default(),
            change_pct_24h: Decimal::try_from(data.change_pct).unwrap_or_default(),
            timestamp: Utc::now(),
        }
    }
}

impl KrakenTicker {
    /// Convert to normalized Ticker type
    pub fn to_ticker(&self) -> Ticker {
        Ticker {
            exchange: ExchangeId::Kraken,
            symbol: normalize_symbol(&self.symbol),
            bid: self.bid,
            bid_qty: self.bid_qty,
            ask: self.ask,
            ask_qty: self.ask_qty,
            last: self.last,
            volume_24h: self.volume_24h,
            high_24h: self.high_24h,
            low_24h: self.low_24h,
            change_24h: self.change_24h,
            change_pct_24h: self.change_pct_24h,
            vwap: Some(self.vwap),
            timestamp: self.timestamp,
        }
    }
}

// ============================================================================
// Trade Data
// ============================================================================

/// Raw trade data from Kraken
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenTradeData {
    pub symbol: String,
    pub side: String,
    pub price: f64,
    pub qty: f64,
    pub ord_type: String,
    pub trade_id: i64,
    pub timestamp: String,
}

/// Parsed Kraken trade
#[derive(Debug, Clone)]
pub struct KrakenTrade {
    pub symbol: String,
    pub side: TradeSide,
    pub price: Decimal,
    pub quantity: Decimal,
    pub trade_id: String,
    pub timestamp: DateTime<Utc>,
}

impl From<KrakenTradeData> for KrakenTrade {
    fn from(data: KrakenTradeData) -> Self {
        let side = if data.side == "buy" {
            TradeSide::Buy
        } else {
            TradeSide::Sell
        };

        let timestamp = DateTime::parse_from_rfc3339(&data.timestamp)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Self {
            symbol: data.symbol,
            side,
            price: Decimal::try_from(data.price).unwrap_or_default(),
            quantity: Decimal::try_from(data.qty).unwrap_or_default(),
            trade_id: data.trade_id.to_string(),
            timestamp,
        }
    }
}

impl KrakenTrade {
    /// Convert to normalized Trade type
    pub fn to_trade(&self) -> Trade {
        Trade {
            exchange: ExchangeId::Kraken,
            symbol: normalize_symbol(&self.symbol),
            trade_id: self.trade_id.clone(),
            price: self.price,
            quantity: self.quantity,
            side: self.side,
            timestamp: self.timestamp,
        }
    }
}

// ============================================================================
// OHLC/Candle Data
// ============================================================================

/// Raw OHLC data from Kraken
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOhlcData {
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub vwap: f64,
    pub volume: f64,
    pub trades: i64,
    pub interval_begin: String,
    pub interval: i32,
}

/// Parsed Kraken candle
#[derive(Debug, Clone)]
pub struct KrakenCandle {
    pub symbol: String,
    pub interval: u32,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub vwap: Decimal,
    pub trades: u64,
    pub open_time: DateTime<Utc>,
}

impl From<KrakenOhlcData> for KrakenCandle {
    fn from(data: KrakenOhlcData) -> Self {
        let open_time = DateTime::parse_from_rfc3339(&data.interval_begin)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Self {
            symbol: data.symbol,
            interval: data.interval as u32,
            open: Decimal::try_from(data.open).unwrap_or_default(),
            high: Decimal::try_from(data.high).unwrap_or_default(),
            low: Decimal::try_from(data.low).unwrap_or_default(),
            close: Decimal::try_from(data.close).unwrap_or_default(),
            volume: Decimal::try_from(data.volume).unwrap_or_default(),
            vwap: Decimal::try_from(data.vwap).unwrap_or_default(),
            trades: data.trades as u64,
            open_time,
        }
    }
}

impl KrakenCandle {
    /// Convert to normalized Candle type
    pub fn to_candle(&self) -> Candle {
        // Calculate close time based on interval
        let close_time = self.open_time + chrono::Duration::minutes(self.interval as i64);

        Candle {
            exchange: ExchangeId::Kraken,
            symbol: normalize_symbol(&self.symbol),
            interval: self.interval,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            quote_volume: None,
            trades: Some(self.trades),
            vwap: Some(self.vwap),
            open_time: self.open_time,
            close_time,
            is_closed: false, // Kraken sends updates for current candle
        }
    }
}

// ============================================================================
// Event Types
// ============================================================================

/// Events broadcast by the Kraken WebSocket client
#[derive(Debug, Clone)]
pub enum KrakenEvent {
    /// Ticker update
    Ticker(KrakenTicker),
    /// Trade update
    Trade(KrakenTrade),
    /// Candle/OHLC update
    Candle(KrakenCandle),
    /// Connection established
    Connected,
    /// Connection lost
    Disconnected,
    /// Error occurred
    Error(String),
    /// Subscription confirmed
    Subscribed {
        channel: String,
        symbol: Option<String>,
    },
}

impl KrakenEvent {
    /// Convert to MarketDataEvent if applicable
    pub fn to_market_data_event(&self) -> Option<MarketDataEvent> {
        match self {
            KrakenEvent::Ticker(t) => Some(MarketDataEvent::Ticker(t.to_ticker())),
            KrakenEvent::Trade(t) => Some(MarketDataEvent::Trade(t.to_trade())),
            KrakenEvent::Candle(c) => Some(MarketDataEvent::Candle(c.to_candle())),
            KrakenEvent::Connected => Some(MarketDataEvent::ConnectionStatus {
                exchange: ExchangeId::Kraken,
                connected: true,
                message: Some("Connected to Kraken WebSocket".to_string()),
                timestamp: Utc::now(),
            }),
            KrakenEvent::Disconnected => Some(MarketDataEvent::ConnectionStatus {
                exchange: ExchangeId::Kraken,
                connected: false,
                message: Some("Disconnected from Kraken WebSocket".to_string()),
                timestamp: Utc::now(),
            }),
            KrakenEvent::Error(msg) => Some(MarketDataEvent::Error {
                exchange: ExchangeId::Kraken,
                code: None,
                message: msg.clone(),
                timestamp: Utc::now(),
            }),
            KrakenEvent::Subscribed { .. } => None,
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Kraken WebSocket client
#[derive(Debug, Clone)]
pub struct KrakenWsConfig {
    /// Use testnet (Kraken doesn't have public testnet, ignored)
    pub testnet: bool,
    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,
    /// Maximum reconnection attempts
    pub max_reconnect_attempts: u32,
    /// Initial reconnect delay in seconds
    pub reconnect_delay_secs: u64,
    /// Ping interval in seconds
    pub ping_interval_secs: u64,
}

impl Default for KrakenWsConfig {
    fn default() -> Self {
        Self {
            testnet: false,
            auto_reconnect: true,
            max_reconnect_attempts: MAX_RECONNECT_ATTEMPTS,
            reconnect_delay_secs: RECONNECT_DELAY_SECS,
            ping_interval_secs: PING_INTERVAL_SECS,
        }
    }
}

// ============================================================================
// WebSocket Client
// ============================================================================

/// Kraken WebSocket client for real-time market data
///
/// This client connects to Kraken's public WebSocket API v2 and streams
/// market data without requiring any API keys.
///
/// # Example
///
/// ```rust,ignore
/// let config = KrakenWsConfig::default();
/// let mut ws = KrakenWebSocket::new(config);
///
/// // Start the client
/// ws.start().await?;
///
/// // Subscribe to tickers
/// ws.subscribe_ticker(&["BTC/USD", "ETH/USD"]).await?;
///
/// // Receive events
/// let mut rx = ws.subscribe_events();
/// while let Ok(event) = rx.recv().await {
///     match event {
///         KrakenEvent::Ticker(ticker) => println!("Ticker: {:?}", ticker),
///         KrakenEvent::Trade(trade) => println!("Trade: {:?}", trade),
///         _ => {}
///     }
/// }
/// ```
pub struct KrakenWebSocket {
    config: KrakenWsConfig,
    event_tx: broadcast::Sender<KrakenEvent>,
    subscriptions: Arc<RwLock<Subscriptions>>,
    is_running: Arc<RwLock<bool>>,
    reconnect_count: Arc<RwLock<u32>>,
    ticker_cache: Arc<RwLock<HashMap<String, KrakenTicker>>>,
    request_id: Arc<RwLock<u64>>,
    /// Metrics for WebSocket connection health
    metrics: Arc<ExchangeMetrics>,
}

/// Active subscriptions
#[derive(Debug, Default)]
struct Subscriptions {
    tickers: Vec<String>,
    trades: Vec<String>,
    candles: HashMap<u32, Vec<String>>, // interval -> symbols
}

impl KrakenWebSocket {
    /// Create a new Kraken WebSocket client
    pub fn new(config: KrakenWsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(10000);
        // Register with global metrics registry
        let metrics = register_exchange(ExchangeId::Kraken);
        Self {
            config,
            event_tx,
            subscriptions: Arc::new(RwLock::new(Subscriptions::default())),
            is_running: Arc::new(RwLock::new(false)),
            reconnect_count: Arc::new(RwLock::new(0)),
            ticker_cache: Arc::new(RwLock::new(HashMap::new())),
            request_id: Arc::new(RwLock::new(0)),
            metrics,
        }
    }

    /// Get metrics for this WebSocket connection
    pub fn metrics(&self) -> &Arc<ExchangeMetrics> {
        &self.metrics
    }

    /// Subscribe to events from this client
    pub fn subscribe_events(&self) -> broadcast::Receiver<KrakenEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender for forwarding to aggregators
    pub fn event_sender(&self) -> broadcast::Sender<KrakenEvent> {
        self.event_tx.clone()
    }

    /// Check if the client is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get a cached ticker
    pub async fn get_ticker(&self, symbol: &str) -> Option<KrakenTicker> {
        self.ticker_cache.read().await.get(symbol).cloned()
    }

    /// Start the WebSocket client
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(ExecutionError::WebSocketError(
                "Kraken WebSocket already running".to_string(),
            ));
        }
        *is_running = true;
        drop(is_running);

        let config = self.config.clone();
        let event_tx = self.event_tx.clone();
        let subscriptions = self.subscriptions.clone();
        let is_running = self.is_running.clone();
        let reconnect_count = self.reconnect_count.clone();
        let ticker_cache = self.ticker_cache.clone();
        let request_id = self.request_id.clone();
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            Self::run_loop(
                config,
                event_tx,
                subscriptions,
                is_running,
                reconnect_count,
                ticker_cache,
                request_id,
                metrics,
            )
            .await;
        });

        Ok(())
    }

    /// Stop the WebSocket client
    pub async fn stop(&self) {
        let mut is_running = self.is_running.write().await;
        *is_running = false;
        info!("Kraken WebSocket client stopped");
    }

    /// Subscribe to ticker updates for the given symbols
    pub async fn subscribe_ticker(&self, symbols: &[&str]) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            let kraken_symbol = to_kraken_symbol(symbol);
            if !subs.tickers.contains(&kraken_symbol) {
                subs.tickers.push(kraken_symbol);
            }
        }
        Ok(())
    }

    /// Subscribe to trade updates for the given symbols
    pub async fn subscribe_trades(&self, symbols: &[&str]) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            let kraken_symbol = to_kraken_symbol(symbol);
            if !subs.trades.contains(&kraken_symbol) {
                subs.trades.push(kraken_symbol);
            }
        }
        Ok(())
    }

    /// Subscribe to candle/OHLC updates for the given symbols
    pub async fn subscribe_candles(&self, symbols: &[&str], interval: u32) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        let entry = subs.candles.entry(interval).or_default();
        for symbol in symbols {
            let kraken_symbol = to_kraken_symbol(symbol);
            if !entry.contains(&kraken_symbol) {
                entry.push(kraken_symbol);
            }
        }
        Ok(())
    }

    /// Get all subscribed ticker symbols
    pub async fn subscribed_tickers(&self) -> Vec<String> {
        self.subscriptions.read().await.tickers.clone()
    }

    /// Main run loop with reconnection logic
    #[allow(clippy::too_many_arguments)]
    async fn run_loop(
        config: KrakenWsConfig,
        event_tx: broadcast::Sender<KrakenEvent>,
        subscriptions: Arc<RwLock<Subscriptions>>,
        is_running: Arc<RwLock<bool>>,
        reconnect_count: Arc<RwLock<u32>>,
        ticker_cache: Arc<RwLock<HashMap<String, KrakenTicker>>>,
        request_id: Arc<RwLock<u64>>,
        metrics: Arc<ExchangeMetrics>,
    ) {
        while *is_running.read().await {
            let count = *reconnect_count.read().await;
            if count > 0 && config.auto_reconnect {
                let delay =
                    std::cmp::min(config.reconnect_delay_secs * (2_u64.pow(count.min(6))), 60);
                info!(
                    "Kraken: Reconnecting in {} seconds (attempt {})",
                    delay, count
                );
                // Record reconnect attempt in metrics
                metrics.record_reconnect();
                sleep(Duration::from_secs(delay)).await;
            }

            if count >= config.max_reconnect_attempts {
                error!("Kraken: Max reconnection attempts reached");
                metrics.record_error();
                let _ = event_tx.send(KrakenEvent::Error(
                    "Max reconnection attempts exceeded".to_string(),
                ));
                break;
            }

            match Self::connect_and_run(
                &config,
                &event_tx,
                &subscriptions,
                &ticker_cache,
                &request_id,
                &metrics,
            )
            .await
            {
                Ok(_) => {
                    info!("Kraken: WebSocket connection closed normally");
                    metrics.record_disconnected();
                    *reconnect_count.write().await = 0;
                }
                Err(e) => {
                    error!("Kraken: WebSocket error: {}", e);
                    metrics.record_error();
                    metrics.record_disconnected();
                    let _ = event_tx.send(KrakenEvent::Error(e.to_string()));
                    let _ = event_tx.send(KrakenEvent::Disconnected);
                    *reconnect_count.write().await += 1;
                }
            }

            if !config.auto_reconnect || !*is_running.read().await {
                break;
            }
        }
    }

    /// Connect to WebSocket and run until disconnect
    async fn connect_and_run(
        config: &KrakenWsConfig,
        event_tx: &broadcast::Sender<KrakenEvent>,
        subscriptions: &Arc<RwLock<Subscriptions>>,
        ticker_cache: &Arc<RwLock<HashMap<String, KrakenTicker>>>,
        request_id: &Arc<RwLock<u64>>,
        metrics: &Arc<ExchangeMetrics>,
    ) -> Result<()> {
        info!("Kraken: Connecting to {}", WS_PUBLIC_URL);

        let (ws_stream, _) = connect_async(WS_PUBLIC_URL).await.map_err(|e| {
            metrics.record_error();
            ExecutionError::WebSocketError(format!("Connection failed: {}", e))
        })?;

        info!("Kraken: WebSocket connected");
        metrics.record_connected();
        let (mut write, mut read) = ws_stream.split();

        let _ = event_tx.send(KrakenEvent::Connected);

        // Wait for status message
        if let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    debug!("Kraken: Initial message: {}", text);
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(ExecutionError::WebSocketError(format!(
                        "Failed to receive initial message: {}",
                        e
                    )));
                }
            }
        }

        // Subscribe to channels
        let subs = subscriptions.read().await;

        // Subscribe to tickers
        if !subs.tickers.is_empty() {
            let mut id = request_id.write().await;
            *id += 1;
            let req = WsRequest {
                method: "subscribe".to_string(),
                params: WsParams {
                    channel: "ticker".to_string(),
                    symbol: Some(subs.tickers.clone()),
                    depth: None,
                    interval: None,
                    snapshot: Some(true),
                },
                req_id: Some(*id),
            };
            let msg = serde_json::to_string(&req)
                .map_err(|e| ExecutionError::WebSocketError(format!("Serialize failed: {}", e)))?;
            debug!("Kraken: Subscribing to tickers: {}", msg);
            write
                .send(Message::Text(msg.into()))
                .await
                .map_err(|e| ExecutionError::WebSocketError(format!("Send failed: {}", e)))?;
        }

        // Subscribe to trades
        if !subs.trades.is_empty() {
            let mut id = request_id.write().await;
            *id += 1;
            let req = WsRequest {
                method: "subscribe".to_string(),
                params: WsParams {
                    channel: "trade".to_string(),
                    symbol: Some(subs.trades.clone()),
                    depth: None,
                    interval: None,
                    snapshot: Some(false),
                },
                req_id: Some(*id),
            };
            let msg = serde_json::to_string(&req)
                .map_err(|e| ExecutionError::WebSocketError(format!("Serialize failed: {}", e)))?;
            debug!("Kraken: Subscribing to trades: {}", msg);
            write
                .send(Message::Text(msg.into()))
                .await
                .map_err(|e| ExecutionError::WebSocketError(format!("Send failed: {}", e)))?;
        }

        // Subscribe to candles
        for (interval, symbols) in &subs.candles {
            if !symbols.is_empty() {
                let mut id = request_id.write().await;
                *id += 1;
                let req = WsRequest {
                    method: "subscribe".to_string(),
                    params: WsParams {
                        channel: "ohlc".to_string(),
                        symbol: Some(symbols.clone()),
                        depth: None,
                        interval: Some(*interval),
                        snapshot: Some(true),
                    },
                    req_id: Some(*id),
                };
                let msg = serde_json::to_string(&req).map_err(|e| {
                    ExecutionError::WebSocketError(format!("Serialize failed: {}", e))
                })?;
                debug!("Kraken: Subscribing to candles: {}", msg);
                write
                    .send(Message::Text(msg.into()))
                    .await
                    .map_err(|e| ExecutionError::WebSocketError(format!("Send failed: {}", e)))?;
            }
        }
        drop(subs);

        // Update subscription count in metrics
        let sub_count = subs_count(subscriptions).await;
        metrics.set_subscription_count(sub_count);

        // Message processing loop
        let mut ping_interval =
            tokio::time::interval(Duration::from_secs(config.ping_interval_secs));

        loop {
            tokio::select! {
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            // Record message received
                            metrics.record_message();
                            if let Err(e) = Self::handle_message(&text, event_tx, ticker_cache).await {
                                warn!("Kraken: Failed to handle message: {}", e);
                                metrics.record_error();
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("Kraken: Close frame received");
                            break;
                        }
                        Some(Ok(Message::Ping(payload))) => {
                            metrics.record_message();
                            if write.send(Message::Pong(payload)).await.is_err() {
                                break;
                            }
                        }
                        Some(Ok(Message::Pong(_))) => {
                            debug!("Kraken: Pong received");
                            metrics.record_message();
                        }
                        Some(Err(e)) => {
                            error!("Kraken: Read error: {}", e);
                            metrics.record_error();
                            break;
                        }
                        None => {
                            info!("Kraken: Stream ended");
                            break;
                        }
                        _ => {}
                    }
                }
                _ = ping_interval.tick() => {
                    // Kraken v2 uses heartbeat channel, but we can still send ping frames
                    if write.send(Message::Ping(vec![].into())).await.is_err() {
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle incoming WebSocket message
    async fn handle_message(
        text: &str,
        event_tx: &broadcast::Sender<KrakenEvent>,
        ticker_cache: &Arc<RwLock<HashMap<String, KrakenTicker>>>,
    ) -> Result<()> {
        let start = Instant::now();
        let response: WsResponse = serde_json::from_str(text).map_err(|e| {
            ExecutionError::WebSocketError(format!("Parse failed: {} - {}", e, text))
        })?;

        match response {
            WsResponse::Subscribe {
                method,
                success,
                result,
                error,
                ..
            } => {
                if method == "subscribe" {
                    if success {
                        if let Some(res) = result {
                            info!("Kraken: Subscribed to {} for {:?}", res.channel, res.symbol);
                            let _ = event_tx.send(KrakenEvent::Subscribed {
                                channel: res.channel,
                                symbol: res.symbol,
                            });
                        }
                    } else if let Some(err) = error {
                        warn!("Kraken: Subscription failed: {}", err);
                        let _ = event_tx.send(KrakenEvent::Error(err));
                    }
                }
            }
            WsResponse::ChannelData {
                channel,
                update_type,
                data,
            } => {
                match channel.as_str() {
                    "ticker" => {
                        if let Ok(tickers) = serde_json::from_value::<Vec<KrakenTickerData>>(data) {
                            for ticker_data in tickers {
                                let ticker = KrakenTicker::from(ticker_data);
                                // Update cache
                                ticker_cache
                                    .write()
                                    .await
                                    .insert(ticker.symbol.clone(), ticker.clone());
                                let _ = event_tx.send(KrakenEvent::Ticker(ticker));
                            }
                        }
                    }
                    "trade" => {
                        if let Ok(trades) = serde_json::from_value::<Vec<KrakenTradeData>>(data) {
                            for trade_data in trades {
                                let trade = KrakenTrade::from(trade_data);
                                let _ = event_tx.send(KrakenEvent::Trade(trade));
                            }
                        }
                    }
                    "ohlc" => {
                        if let Ok(candles) = serde_json::from_value::<Vec<KrakenOhlcData>>(data) {
                            for candle_data in candles {
                                let candle = KrakenCandle::from(candle_data);
                                let _ = event_tx.send(KrakenEvent::Candle(candle));
                            }
                        }
                    }
                    _ => {
                        debug!("Kraken: Unknown channel {} type {}", channel, update_type);
                    }
                }

                // Record WebSocket message processing latency
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                global_latency_histograms().record_websocket_message("kraken", duration_ms);
            }
            WsResponse::Heartbeat { .. } => {
                debug!("Kraken: Heartbeat received");
            }
            WsResponse::Status { data, .. } => {
                for status in data {
                    if let Some(version) = status.api_version {
                        info!("Kraken: API version {}", version);
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Count total subscriptions
async fn subs_count(subscriptions: &Arc<RwLock<Subscriptions>>) -> u64 {
    let subs = subscriptions.read().await;
    let ticker_count = subs.tickers.len();
    let trade_count = subs.trades.len();
    let candle_count: usize = subs.candles.values().map(|v| v.len()).sum();
    (ticker_count + trade_count + candle_count) as u64
}

/// Convert normalized symbol to Kraken format
fn to_kraken_symbol(symbol: &str) -> String {
    // If it contains USDT, convert to USD
    if symbol.contains("USDT") {
        symbol.replace("USDT", "USD")
    } else {
        symbol.to_string()
    }
}

/// Convert Kraken symbol to normalized format
fn normalize_symbol(kraken_symbol: &str) -> String {
    // Convert USD -> USDT for consistency
    if kraken_symbol.ends_with("/USD") && !kraken_symbol.ends_with("/USDT") {
        kraken_symbol.replace("/USD", "/USDT")
    } else {
        kraken_symbol.to_string()
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
        let config = KrakenWsConfig::default();
        assert!(!config.testnet);
        assert!(config.auto_reconnect);
        assert_eq!(config.max_reconnect_attempts, 10);
    }

    #[test]
    fn test_to_kraken_symbol() {
        assert_eq!(to_kraken_symbol("BTC/USDT"), "BTC/USD");
        assert_eq!(to_kraken_symbol("ETH/USDT"), "ETH/USD");
        assert_eq!(to_kraken_symbol("BTC/USD"), "BTC/USD");
    }

    #[test]
    fn test_normalize_symbol() {
        assert_eq!(normalize_symbol("BTC/USD"), "BTC/USDT");
        assert_eq!(normalize_symbol("ETH/USD"), "ETH/USDT");
        assert_eq!(normalize_symbol("BTC/USDT"), "BTC/USDT");
    }

    #[tokio::test]
    async fn test_websocket_creation() {
        let config = KrakenWsConfig::default();
        let ws = KrakenWebSocket::new(config);

        assert!(!ws.is_running().await);
    }

    #[tokio::test]
    async fn test_subscribe_ticker() {
        let config = KrakenWsConfig::default();
        let ws = KrakenWebSocket::new(config);

        ws.subscribe_ticker(&["BTC/USDT", "ETH/USDT"])
            .await
            .unwrap();

        let subs = ws.subscriptions.read().await;
        assert!(subs.tickers.contains(&"BTC/USD".to_string()));
        assert!(subs.tickers.contains(&"ETH/USD".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_trades() {
        let config = KrakenWsConfig::default();
        let ws = KrakenWebSocket::new(config);

        ws.subscribe_trades(&["BTC/USDT"]).await.unwrap();

        let subs = ws.subscriptions.read().await;
        assert!(subs.trades.contains(&"BTC/USD".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_candles() {
        let config = KrakenWsConfig::default();
        let ws = KrakenWebSocket::new(config);

        ws.subscribe_candles(&["BTC/USDT"], 5).await.unwrap();

        let subs = ws.subscriptions.read().await;
        assert!(
            subs.candles
                .get(&5)
                .unwrap()
                .contains(&"BTC/USD".to_string())
        );
    }

    #[test]
    fn test_kraken_ticker_conversion() {
        let data = KrakenTickerData {
            symbol: "BTC/USD".to_string(),
            bid: 67250.0,
            bid_qty: 1.5,
            ask: 67255.0,
            ask_qty: 2.3,
            last: 67252.0,
            volume: 1523.45,
            vwap: 67100.0,
            low: 66500.0,
            high: 68000.0,
            change: 150.0,
            change_pct: 0.22,
        };

        let ticker = KrakenTicker::from(data);
        assert_eq!(ticker.symbol, "BTC/USD");
        assert_eq!(ticker.bid, Decimal::try_from(67250.0).unwrap());

        let normalized = ticker.to_ticker();
        assert_eq!(normalized.symbol, "BTC/USDT");
        assert_eq!(normalized.exchange, ExchangeId::Kraken);
    }

    #[test]
    fn test_kraken_trade_conversion() {
        let data = KrakenTradeData {
            symbol: "ETH/USD".to_string(),
            side: "buy".to_string(),
            price: 3500.0,
            qty: 2.5,
            ord_type: "market".to_string(),
            trade_id: 12345,
            timestamp: "2024-01-15T10:30:00Z".to_string(),
        };

        let trade = KrakenTrade::from(data);
        assert_eq!(trade.symbol, "ETH/USD");
        assert!(matches!(trade.side, TradeSide::Buy));

        let normalized = trade.to_trade();
        assert_eq!(normalized.symbol, "ETH/USDT");
    }

    #[test]
    fn test_event_to_market_data() {
        let ticker = KrakenTicker {
            symbol: "BTC/USD".to_string(),
            bid: Decimal::from(67250),
            bid_qty: Decimal::from(1),
            ask: Decimal::from(67255),
            ask_qty: Decimal::from(2),
            last: Decimal::from(67252),
            volume_24h: Decimal::from(1523),
            vwap: Decimal::from(67100),
            low_24h: Decimal::from(66500),
            high_24h: Decimal::from(68000),
            change_24h: Decimal::from(150),
            change_pct_24h: Decimal::try_from(0.22).unwrap(),
            timestamp: Utc::now(),
        };

        let event = KrakenEvent::Ticker(ticker);
        let market_event = event.to_market_data_event();
        assert!(market_event.is_some());

        if let Some(MarketDataEvent::Ticker(t)) = market_event {
            assert_eq!(t.exchange, ExchangeId::Kraken);
            assert_eq!(t.symbol, "BTC/USDT");
        }
    }
}
