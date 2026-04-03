//! Binance WebSocket Client
//!
//! Real-time market data streaming from Binance's WebSocket API.
//! This is FREE and does NOT require any API keys for public market data.
//!
//! # WebSocket API
//!
//! Binance uses stream-based subscriptions:
//!
//! ## Subscribe Request
//! ```json
//! {
//!     "method": "SUBSCRIBE",
//!     "params": ["btcusdt@trade", "ethusdt@ticker"],
//!     "id": 1
//! }
//! ```
//!
//! ## Trade Stream Response
//! ```json
//! {
//!     "e": "trade",
//!     "E": 1672515782136,
//!     "s": "BTCUSD",
//!     "t": 12345,
//!     "p": "67252.00",
//!     "q": "0.5",
//!     "T": 1672515782136,
//!     "m": true
//! }
//! ```

use crate::error::{ExecutionError, Result};
use crate::exchanges::market_data::{
    Candle, ExchangeId, MarketDataEvent, Ticker, Trade, TradeSide,
};
use crate::exchanges::metrics::{ExchangeMetrics, register_exchange};
use crate::execution::histogram::global_latency_histograms;
use chrono::{DateTime, LocalResult, TimeZone, Utc};
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, broadcast};
use tokio::time::{Duration, sleep};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};

/// Binance WebSocket URLs
pub const WS_SPOT_URL: &str = "wss://stream.binance.com:9443/ws";
pub const WS_COMBINED_URL: &str = "wss://stream.binance.com:9443/stream";
pub const WS_TESTNET_URL: &str = "wss://testnet.binance.vision/ws";

/// Connection settings
const PING_INTERVAL_SECS: u64 = 180; // Binance requires ping every 3 minutes
const RECONNECT_DELAY_SECS: u64 = 5;
const MAX_RECONNECT_ATTEMPTS: u32 = 10;

// ============================================================================
// WebSocket Message Types
// ============================================================================

/// Binance WebSocket subscription request
#[derive(Debug, Clone, Serialize)]
struct WsRequest {
    method: String,
    params: Vec<String>,
    id: u64,
}

/// Binance WebSocket response
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum WsResponse {
    /// Subscription confirmation
    Subscribe {
        result: Option<serde_json::Value>,
        id: u64,
    },
    /// Trade event
    Trade {
        #[serde(rename = "e")]
        event_type: String,
        #[serde(rename = "E")]
        event_time: i64,
        #[serde(rename = "s")]
        symbol: String,
        #[serde(rename = "t")]
        trade_id: i64,
        #[serde(rename = "p")]
        price: String,
        #[serde(rename = "q")]
        quantity: String,
        #[serde(rename = "T")]
        trade_time: i64,
        #[serde(rename = "m")]
        is_buyer_maker: bool,
    },
    /// 24hr Ticker event
    Ticker24hr {
        #[serde(rename = "e")]
        event_type: String,
        #[serde(rename = "E")]
        event_time: i64,
        #[serde(rename = "s")]
        symbol: String,
        #[serde(rename = "p")]
        price_change: String,
        #[serde(rename = "P")]
        price_change_percent: String,
        #[serde(rename = "w")]
        weighted_avg_price: String,
        #[serde(rename = "c")]
        last_price: String,
        #[serde(rename = "Q")]
        last_qty: String,
        #[serde(rename = "b")]
        best_bid_price: String,
        #[serde(rename = "B")]
        best_bid_qty: String,
        #[serde(rename = "a")]
        best_ask_price: String,
        #[serde(rename = "A")]
        best_ask_qty: String,
        #[serde(rename = "o")]
        open_price: String,
        #[serde(rename = "h")]
        high_price: String,
        #[serde(rename = "l")]
        low_price: String,
        #[serde(rename = "v")]
        volume: String,
        #[serde(rename = "q")]
        quote_volume: String,
    },
    /// Kline/Candlestick event
    Kline {
        #[serde(rename = "e")]
        event_type: String,
        #[serde(rename = "E")]
        event_time: i64,
        #[serde(rename = "s")]
        symbol: String,
        #[serde(rename = "k")]
        kline: KlineData,
    },
    /// Combined stream wrapper
    Combined {
        stream: String,
        data: Box<WsResponse>,
    },
}

/// Kline data structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct KlineData {
    #[serde(rename = "t")]
    open_time: i64,
    #[serde(rename = "T")]
    close_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "i")]
    interval: String,
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "v")]
    volume: String,
    #[serde(rename = "n")]
    trades: i64,
    #[serde(rename = "x")]
    is_closed: bool,
    #[serde(rename = "q")]
    quote_volume: String,
}

// ============================================================================
// Binance Data Types
// ============================================================================

/// Parsed Binance ticker
#[derive(Debug, Clone)]
pub struct BinanceTicker {
    pub symbol: String,
    pub bid: Decimal,
    pub bid_qty: Decimal,
    pub ask: Decimal,
    pub ask_qty: Decimal,
    pub last: Decimal,
    pub volume_24h: Decimal,
    pub quote_volume_24h: Decimal,
    pub open_24h: Decimal,
    pub high_24h: Decimal,
    pub low_24h: Decimal,
    pub change_24h: Decimal,
    pub change_pct_24h: Decimal,
    pub vwap: Decimal,
    pub timestamp: DateTime<Utc>,
}

impl BinanceTicker {
    /// Convert to normalized Ticker type
    pub fn to_ticker(&self) -> Ticker {
        Ticker {
            exchange: ExchangeId::Binance,
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

/// Parsed Binance trade
#[derive(Debug, Clone)]
pub struct BinanceTrade {
    pub symbol: String,
    pub trade_id: String,
    pub price: Decimal,
    pub quantity: Decimal,
    pub side: TradeSide,
    pub timestamp: DateTime<Utc>,
}

impl BinanceTrade {
    /// Convert to normalized Trade type
    pub fn to_trade(&self) -> Trade {
        Trade {
            exchange: ExchangeId::Binance,
            symbol: normalize_symbol(&self.symbol),
            trade_id: self.trade_id.clone(),
            price: self.price,
            quantity: self.quantity,
            side: self.side,
            timestamp: self.timestamp,
        }
    }
}

/// Parsed Binance candle
#[derive(Debug, Clone)]
pub struct BinanceCandle {
    pub symbol: String,
    pub interval: String,
    pub interval_minutes: u32,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub quote_volume: Decimal,
    pub trades: u64,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub is_closed: bool,
}

impl BinanceCandle {
    /// Convert to normalized Candle type
    pub fn to_candle(&self) -> Candle {
        Candle {
            exchange: ExchangeId::Binance,
            symbol: normalize_symbol(&self.symbol),
            interval: self.interval_minutes,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            quote_volume: Some(self.quote_volume),
            trades: Some(self.trades),
            vwap: None,
            open_time: self.open_time,
            close_time: self.close_time,
            is_closed: self.is_closed,
        }
    }
}

// ============================================================================
// Event Types
// ============================================================================

/// Events broadcast by the Binance WebSocket client
#[derive(Debug, Clone)]
pub enum BinanceEvent {
    /// Ticker update
    Ticker(BinanceTicker),
    /// Trade update
    Trade(BinanceTrade),
    /// Candle/Kline update
    Candle(BinanceCandle),
    /// Connection established
    Connected,
    /// Connection lost
    Disconnected,
    /// Error occurred
    Error(String),
    /// Subscription confirmed
    Subscribed { streams: Vec<String> },
}

impl BinanceEvent {
    /// Convert to MarketDataEvent if applicable
    pub fn to_market_data_event(&self) -> Option<MarketDataEvent> {
        match self {
            BinanceEvent::Ticker(t) => Some(MarketDataEvent::Ticker(t.to_ticker())),
            BinanceEvent::Trade(t) => Some(MarketDataEvent::Trade(t.to_trade())),
            BinanceEvent::Candle(c) => Some(MarketDataEvent::Candle(c.to_candle())),
            BinanceEvent::Connected => Some(MarketDataEvent::ConnectionStatus {
                exchange: ExchangeId::Binance,
                connected: true,
                message: Some("Connected to Binance WebSocket".to_string()),
                timestamp: Utc::now(),
            }),
            BinanceEvent::Disconnected => Some(MarketDataEvent::ConnectionStatus {
                exchange: ExchangeId::Binance,
                connected: false,
                message: Some("Disconnected from Binance WebSocket".to_string()),
                timestamp: Utc::now(),
            }),
            BinanceEvent::Error(msg) => Some(MarketDataEvent::Error {
                exchange: ExchangeId::Binance,
                code: None,
                message: msg.clone(),
                timestamp: Utc::now(),
            }),
            BinanceEvent::Subscribed { .. } => None,
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Binance WebSocket client
#[derive(Debug, Clone)]
pub struct BinanceWsConfig {
    /// Use testnet
    pub testnet: bool,
    /// Use combined streams (recommended for multiple subscriptions)
    pub use_combined_stream: bool,
    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,
    /// Maximum reconnection attempts
    pub max_reconnect_attempts: u32,
    /// Initial reconnect delay in seconds
    pub reconnect_delay_secs: u64,
    /// Ping interval in seconds (Binance requires ping every 3 minutes)
    pub ping_interval_secs: u64,
}

impl Default for BinanceWsConfig {
    fn default() -> Self {
        Self {
            testnet: false,
            use_combined_stream: true,
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

/// Binance WebSocket client for real-time market data
///
/// This client connects to Binance's public WebSocket API and streams
/// market data without requiring any API keys.
///
/// # Example
///
/// ```rust,ignore
/// let config = BinanceWsConfig::default();
/// let mut ws = BinanceWebSocket::new(config);
///
/// // Subscribe before starting
/// ws.subscribe_ticker(&["BTC/USDT", "ETH/USDT"]).await?;
///
/// // Start the client
/// ws.start().await?;
///
/// // Receive events
/// let mut rx = ws.subscribe_events();
/// while let Ok(event) = rx.recv().await {
///     match event {
///         BinanceEvent::Ticker(ticker) => println!("Ticker: {:?}", ticker),
///         BinanceEvent::Trade(trade) => println!("Trade: {:?}", trade),
///         _ => {}
///     }
/// }
/// ```
pub struct BinanceWebSocket {
    config: BinanceWsConfig,
    event_tx: broadcast::Sender<BinanceEvent>,
    subscriptions: Arc<RwLock<Subscriptions>>,
    is_running: Arc<RwLock<bool>>,
    reconnect_count: Arc<RwLock<u32>>,
    ticker_cache: Arc<RwLock<HashMap<String, BinanceTicker>>>,
    request_id: Arc<RwLock<u64>>,
    /// Metrics for WebSocket connection health
    metrics: Arc<ExchangeMetrics>,
}

/// Active subscriptions
#[derive(Debug, Default)]
struct Subscriptions {
    streams: Vec<String>,
}

impl BinanceWebSocket {
    /// Create a new Binance WebSocket client
    pub fn new(config: BinanceWsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(10000);
        // Register with global metrics registry
        let metrics = register_exchange(ExchangeId::Binance);
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
    pub fn subscribe_events(&self) -> broadcast::Receiver<BinanceEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender for forwarding to aggregators
    pub fn event_sender(&self) -> broadcast::Sender<BinanceEvent> {
        self.event_tx.clone()
    }

    /// Check if the client is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get a cached ticker
    pub async fn get_ticker(&self, symbol: &str) -> Option<BinanceTicker> {
        self.ticker_cache.read().await.get(symbol).cloned()
    }

    /// Start the WebSocket client
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(ExecutionError::WebSocketError(
                "Binance WebSocket already running".to_string(),
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
        info!("Binance WebSocket client stopped");
    }

    /// Subscribe to ticker updates for the given symbols
    ///
    /// # Arguments
    /// * `symbols` - Normalized symbols like "BTC/USDT", "ETH/USDT"
    pub async fn subscribe_ticker(&self, symbols: &[&str]) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            let stream = format!("{}@ticker", to_binance_symbol(symbol));
            if !subs.streams.contains(&stream) {
                subs.streams.push(stream);
            }
        }
        Ok(())
    }

    /// Subscribe to trade updates for the given symbols
    pub async fn subscribe_trades(&self, symbols: &[&str]) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            let stream = format!("{}@trade", to_binance_symbol(symbol));
            if !subs.streams.contains(&stream) {
                subs.streams.push(stream);
            }
        }
        Ok(())
    }

    /// Subscribe to candle/kline updates for the given symbols
    ///
    /// # Arguments
    /// * `symbols` - Normalized symbols like "BTC/USDT"
    /// * `interval` - Interval in minutes (1, 5, 15, 60, 240, 1440, etc.)
    pub async fn subscribe_candles(&self, symbols: &[&str], interval: u32) -> Result<()> {
        let interval_str = interval_to_binance(interval);
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            let stream = format!("{}@kline_{}", to_binance_symbol(symbol), interval_str);
            if !subs.streams.contains(&stream) {
                subs.streams.push(stream);
            }
        }
        Ok(())
    }

    /// Subscribe to aggregate trades for the given symbols
    pub async fn subscribe_agg_trades(&self, symbols: &[&str]) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            let stream = format!("{}@aggTrade", to_binance_symbol(symbol));
            if !subs.streams.contains(&stream) {
                subs.streams.push(stream);
            }
        }
        Ok(())
    }

    /// Get all subscribed streams
    pub async fn subscribed_streams(&self) -> Vec<String> {
        self.subscriptions.read().await.streams.clone()
    }

    /// Main run loop with reconnection logic
    #[allow(clippy::too_many_arguments)]
    async fn run_loop(
        config: BinanceWsConfig,
        event_tx: broadcast::Sender<BinanceEvent>,
        subscriptions: Arc<RwLock<Subscriptions>>,
        is_running: Arc<RwLock<bool>>,
        reconnect_count: Arc<RwLock<u32>>,
        ticker_cache: Arc<RwLock<HashMap<String, BinanceTicker>>>,
        request_id: Arc<RwLock<u64>>,
        metrics: Arc<ExchangeMetrics>,
    ) {
        while *is_running.read().await {
            let count = *reconnect_count.read().await;
            if count > 0 && config.auto_reconnect {
                let delay =
                    std::cmp::min(config.reconnect_delay_secs * (2_u64.pow(count.min(6))), 60);
                info!(
                    "Binance: Reconnecting in {} seconds (attempt {})",
                    delay, count
                );
                // Record reconnect attempt in metrics
                metrics.record_reconnect();
                sleep(Duration::from_secs(delay)).await;
            }

            if count >= config.max_reconnect_attempts {
                error!("Binance: Max reconnection attempts reached");
                metrics.record_error();
                let _ = event_tx.send(BinanceEvent::Error(
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
                    info!("Binance: WebSocket connection closed normally");
                    metrics.record_disconnected();
                    *reconnect_count.write().await = 0;
                }
                Err(e) => {
                    error!("Binance: WebSocket error: {}", e);
                    metrics.record_error();
                    metrics.record_disconnected();
                    let _ = event_tx.send(BinanceEvent::Error(e.to_string()));
                    let _ = event_tx.send(BinanceEvent::Disconnected);
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
        config: &BinanceWsConfig,
        event_tx: &broadcast::Sender<BinanceEvent>,
        subscriptions: &Arc<RwLock<Subscriptions>>,
        ticker_cache: &Arc<RwLock<HashMap<String, BinanceTicker>>>,
        request_id: &Arc<RwLock<u64>>,
        metrics: &Arc<ExchangeMetrics>,
    ) -> Result<()> {
        let subs = subscriptions.read().await;
        let streams = subs.streams.clone();
        drop(subs);

        // Update subscription count in metrics
        metrics.set_subscription_count(streams.len() as u64);

        // Build URL based on config
        let url = if config.testnet {
            if config.use_combined_stream && !streams.is_empty() {
                format!(
                    "{}/stream?streams={}",
                    WS_TESTNET_URL.trim_end_matches("/ws"),
                    streams.join("/")
                )
            } else {
                WS_TESTNET_URL.to_string()
            }
        } else if config.use_combined_stream && !streams.is_empty() {
            format!(
                "wss://stream.binance.com:9443/stream?streams={}",
                streams.join("/")
            )
        } else {
            WS_SPOT_URL.to_string()
        };

        info!("Binance: Connecting to {}", url);

        let (ws_stream, _) = connect_async(&url).await.map_err(|e| {
            metrics.record_error();
            ExecutionError::WebSocketError(format!("Connection failed: {}", e))
        })?;

        info!("Binance: WebSocket connected");
        metrics.record_connected();
        let (mut write, mut read) = ws_stream.split();

        let _ = event_tx.send(BinanceEvent::Connected);

        // If not using combined streams URL, subscribe via message
        if !config.use_combined_stream && !streams.is_empty() {
            let mut id = request_id.write().await;
            *id += 1;
            let req = WsRequest {
                method: "SUBSCRIBE".to_string(),
                params: streams.clone(),
                id: *id,
            };
            let msg = serde_json::to_string(&req)
                .map_err(|e| ExecutionError::WebSocketError(format!("Serialize failed: {}", e)))?;
            debug!("Binance: Subscribing: {}", msg);
            write
                .send(Message::Text(msg.into()))
                .await
                .map_err(|e| ExecutionError::WebSocketError(format!("Send failed: {}", e)))?;

            let _ = event_tx.send(BinanceEvent::Subscribed { streams });
        } else if !streams.is_empty() {
            let _ = event_tx.send(BinanceEvent::Subscribed { streams });
        }

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
                            if let Err(e) = Self::handle_message(&text, event_tx, ticker_cache, config.use_combined_stream).await {
                                warn!("Binance: Failed to handle message: {}", e);
                                metrics.record_error();
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("Binance: Close frame received");
                            break;
                        }
                        Some(Ok(Message::Ping(payload))) => {
                            debug!("Binance: Ping received");
                            metrics.record_message();
                            if write.send(Message::Pong(payload)).await.is_err() {
                                break;
                            }
                        }
                        Some(Ok(Message::Pong(_))) => {
                            debug!("Binance: Pong received");
                            metrics.record_message();
                        }
                        Some(Err(e)) => {
                            error!("Binance: Read error: {}", e);
                            metrics.record_error();
                            break;
                        }
                        None => {
                            info!("Binance: Stream ended");
                            break;
                        }
                        _ => {}
                    }
                }
                _ = ping_interval.tick() => {
                    // Binance requires pong response within 10 minutes
                    // We send ping every 3 minutes to keep connection alive
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
        event_tx: &broadcast::Sender<BinanceEvent>,
        ticker_cache: &Arc<RwLock<HashMap<String, BinanceTicker>>>,
        use_combined_stream: bool,
    ) -> Result<()> {
        let start = Instant::now();
        let response: WsResponse = serde_json::from_str(text).map_err(|e| {
            ExecutionError::WebSocketError(format!("Parse failed: {} - text: {}", e, text))
        })?;

        // Handle combined stream wrapper
        let response = if use_combined_stream {
            if let WsResponse::Combined { data, .. } = response {
                *data
            } else {
                response
            }
        } else {
            response
        };

        match response {
            WsResponse::Subscribe { id, .. } => {
                debug!("Binance: Subscription confirmed (id: {})", id);
            }
            WsResponse::Trade {
                event_type,
                symbol,
                trade_id,
                price,
                quantity,
                trade_time,
                is_buyer_maker,
                ..
            } => {
                if event_type == "trade" {
                    let trade = BinanceTrade {
                        symbol: symbol.clone(),
                        trade_id: trade_id.to_string(),
                        price: Decimal::from_str(&price).unwrap_or_default(),
                        quantity: Decimal::from_str(&quantity).unwrap_or_default(),
                        side: if is_buyer_maker {
                            TradeSide::Sell
                        } else {
                            TradeSide::Buy
                        },
                        timestamp: match Utc.timestamp_millis_opt(trade_time) {
                            LocalResult::Single(dt) => dt,
                            _ => Utc::now(),
                        },
                    };
                    let _ = event_tx.send(BinanceEvent::Trade(trade));

                    // Record WebSocket message processing latency
                    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                    global_latency_histograms().record_websocket_message("binance", duration_ms);
                }
            }
            WsResponse::Ticker24hr {
                event_type,
                event_time,
                symbol,
                price_change,
                price_change_percent,
                weighted_avg_price,
                last_price,
                best_bid_price,
                best_bid_qty,
                best_ask_price,
                best_ask_qty,
                open_price,
                high_price,
                low_price,
                volume,
                quote_volume,
                ..
            } => {
                if event_type == "24hrTicker" {
                    let ticker = BinanceTicker {
                        symbol: symbol.clone(),
                        bid: Decimal::from_str(&best_bid_price).unwrap_or_default(),
                        bid_qty: Decimal::from_str(&best_bid_qty).unwrap_or_default(),
                        ask: Decimal::from_str(&best_ask_price).unwrap_or_default(),
                        ask_qty: Decimal::from_str(&best_ask_qty).unwrap_or_default(),
                        last: Decimal::from_str(&last_price).unwrap_or_default(),
                        volume_24h: Decimal::from_str(&volume).unwrap_or_default(),
                        quote_volume_24h: Decimal::from_str(&quote_volume).unwrap_or_default(),
                        open_24h: Decimal::from_str(&open_price).unwrap_or_default(),
                        high_24h: Decimal::from_str(&high_price).unwrap_or_default(),
                        low_24h: Decimal::from_str(&low_price).unwrap_or_default(),
                        change_24h: Decimal::from_str(&price_change).unwrap_or_default(),
                        change_pct_24h: Decimal::from_str(&price_change_percent)
                            .unwrap_or_default(),
                        vwap: Decimal::from_str(&weighted_avg_price).unwrap_or_default(),
                        timestamp: match Utc.timestamp_millis_opt(event_time) {
                            LocalResult::Single(dt) => dt,
                            _ => Utc::now(),
                        },
                    };
                    // Update cache
                    ticker_cache.write().await.insert(symbol, ticker.clone());
                    let _ = event_tx.send(BinanceEvent::Ticker(ticker));

                    // Record WebSocket message processing latency
                    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                    global_latency_histograms().record_websocket_message("binance", duration_ms);
                }
            }
            WsResponse::Kline {
                event_type,
                symbol,
                kline,
                ..
            } => {
                if event_type == "kline" {
                    let interval_minutes = interval_from_binance(&kline.interval);
                    let candle = BinanceCandle {
                        symbol: symbol.clone(),
                        interval: kline.interval,
                        interval_minutes,
                        open: Decimal::from_str(&kline.open).unwrap_or_default(),
                        high: Decimal::from_str(&kline.high).unwrap_or_default(),
                        low: Decimal::from_str(&kline.low).unwrap_or_default(),
                        close: Decimal::from_str(&kline.close).unwrap_or_default(),
                        volume: Decimal::from_str(&kline.volume).unwrap_or_default(),
                        quote_volume: Decimal::from_str(&kline.quote_volume).unwrap_or_default(),
                        trades: kline.trades as u64,
                        open_time: match Utc.timestamp_millis_opt(kline.open_time) {
                            LocalResult::Single(dt) => dt,
                            _ => Utc::now(),
                        },
                        close_time: match Utc.timestamp_millis_opt(kline.close_time) {
                            LocalResult::Single(dt) => dt,
                            _ => Utc::now(),
                        },
                        is_closed: kline.is_closed,
                    };
                    let _ = event_tx.send(BinanceEvent::Candle(candle));

                    // Record WebSocket message processing latency
                    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                    global_latency_histograms().record_websocket_message("binance", duration_ms);
                }
            }
            WsResponse::Combined { .. } => {
                // Already unwrapped above
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert normalized symbol to Binance format (lowercase, no separator)
fn to_binance_symbol(symbol: &str) -> String {
    symbol.to_lowercase().replace("/", "")
}

/// Convert Binance symbol to normalized format
fn normalize_symbol(binance_symbol: &str) -> String {
    let upper = binance_symbol.to_uppercase();

    // Common quote currencies to check
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

/// Convert interval minutes to Binance interval string
fn interval_to_binance(minutes: u32) -> &'static str {
    match minutes {
        1 => "1m",
        3 => "3m",
        5 => "5m",
        15 => "15m",
        30 => "30m",
        60 => "1h",
        120 => "2h",
        240 => "4h",
        360 => "6h",
        480 => "8h",
        720 => "12h",
        1440 => "1d",
        4320 => "3d",
        10080 => "1w",
        43200 => "1M",
        _ => "1m",
    }
}

/// Convert Binance interval string to minutes
fn interval_from_binance(interval: &str) -> u32 {
    match interval {
        "1m" => 1,
        "3m" => 3,
        "5m" => 5,
        "15m" => 15,
        "30m" => 30,
        "1h" => 60,
        "2h" => 120,
        "4h" => 240,
        "6h" => 360,
        "8h" => 480,
        "12h" => 720,
        "1d" => 1440,
        "3d" => 4320,
        "1w" => 10080,
        "1M" => 43200,
        _ => 1,
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
        let config = BinanceWsConfig::default();
        assert!(!config.testnet);
        assert!(config.use_combined_stream);
        assert!(config.auto_reconnect);
        assert_eq!(config.max_reconnect_attempts, 10);
        assert_eq!(config.ping_interval_secs, 180);
    }

    #[test]
    fn test_to_binance_symbol() {
        assert_eq!(to_binance_symbol("BTC/USDT"), "btcusdt");
        assert_eq!(to_binance_symbol("ETH/USDT"), "ethusdt");
        assert_eq!(to_binance_symbol("BTCUSDT"), "btcusdt");
    }

    #[test]
    fn test_normalize_symbol() {
        assert_eq!(normalize_symbol("BTCUSDT"), "BTC/USDT");
        assert_eq!(normalize_symbol("ETHUSDT"), "ETH/USDT");
        assert_eq!(normalize_symbol("btcusdt"), "BTC/USDT");
        assert_eq!(normalize_symbol("SOLUSDC"), "SOL/USDC");
    }

    #[test]
    fn test_interval_conversion() {
        assert_eq!(interval_to_binance(1), "1m");
        assert_eq!(interval_to_binance(5), "5m");
        assert_eq!(interval_to_binance(15), "15m");
        assert_eq!(interval_to_binance(60), "1h");
        assert_eq!(interval_to_binance(240), "4h");
        assert_eq!(interval_to_binance(1440), "1d");

        assert_eq!(interval_from_binance("1m"), 1);
        assert_eq!(interval_from_binance("5m"), 5);
        assert_eq!(interval_from_binance("1h"), 60);
        assert_eq!(interval_from_binance("4h"), 240);
        assert_eq!(interval_from_binance("1d"), 1440);
    }

    #[tokio::test]
    async fn test_websocket_creation() {
        let config = BinanceWsConfig::default();
        let ws = BinanceWebSocket::new(config);

        assert!(!ws.is_running().await);
    }

    #[tokio::test]
    async fn test_subscribe_ticker() {
        let config = BinanceWsConfig::default();
        let ws = BinanceWebSocket::new(config);

        ws.subscribe_ticker(&["BTC/USDT", "ETH/USDT"])
            .await
            .unwrap();

        let streams = ws.subscribed_streams().await;
        assert!(streams.contains(&"btcusdt@ticker".to_string()));
        assert!(streams.contains(&"ethusdt@ticker".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_trades() {
        let config = BinanceWsConfig::default();
        let ws = BinanceWebSocket::new(config);

        ws.subscribe_trades(&["BTC/USDT"]).await.unwrap();

        let streams = ws.subscribed_streams().await;
        assert!(streams.contains(&"btcusdt@trade".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_candles() {
        let config = BinanceWsConfig::default();
        let ws = BinanceWebSocket::new(config);

        ws.subscribe_candles(&["BTC/USDT"], 5).await.unwrap();

        let streams = ws.subscribed_streams().await;
        assert!(streams.contains(&"btcusdt@kline_5m".to_string()));
    }

    #[test]
    fn test_binance_ticker_conversion() {
        let ticker = BinanceTicker {
            symbol: "BTCUSDT".to_string(),
            bid: Decimal::from(67250),
            bid_qty: Decimal::from(1),
            ask: Decimal::from(67255),
            ask_qty: Decimal::from(2),
            last: Decimal::from(67252),
            volume_24h: Decimal::from(1523),
            quote_volume_24h: Decimal::from(102000000),
            open_24h: Decimal::from(67000),
            high_24h: Decimal::from(68000),
            low_24h: Decimal::from(66500),
            change_24h: Decimal::from(252),
            change_pct_24h: Decimal::try_from(0.38).unwrap(),
            vwap: Decimal::from(67100),
            timestamp: Utc::now(),
        };

        let normalized = ticker.to_ticker();
        assert_eq!(normalized.symbol, "BTC/USDT");
        assert_eq!(normalized.exchange, ExchangeId::Binance);
    }

    #[test]
    fn test_binance_trade_conversion() {
        let trade = BinanceTrade {
            symbol: "ETHUSDT".to_string(),
            trade_id: "12345".to_string(),
            price: Decimal::from(3500),
            quantity: Decimal::try_from(2.5).unwrap(),
            side: TradeSide::Buy,
            timestamp: Utc::now(),
        };

        let normalized = trade.to_trade();
        assert_eq!(normalized.symbol, "ETH/USDT");
        assert_eq!(normalized.exchange, ExchangeId::Binance);
    }

    #[test]
    fn test_binance_candle_conversion() {
        let candle = BinanceCandle {
            symbol: "BTCUSDT".to_string(),
            interval: "5m".to_string(),
            interval_minutes: 5,
            open: Decimal::from(67000),
            high: Decimal::from(67500),
            low: Decimal::from(66900),
            close: Decimal::from(67250),
            volume: Decimal::from(100),
            quote_volume: Decimal::from(6700000),
            trades: 1500,
            open_time: Utc::now(),
            close_time: Utc::now(),
            is_closed: true,
        };

        let normalized = candle.to_candle();
        assert_eq!(normalized.symbol, "BTC/USDT");
        assert_eq!(normalized.exchange, ExchangeId::Binance);
        assert_eq!(normalized.interval, 5);
        assert!(normalized.is_closed);
    }

    #[test]
    fn test_event_to_market_data() {
        let ticker = BinanceTicker {
            symbol: "BTCUSDT".to_string(),
            bid: Decimal::from(67250),
            bid_qty: Decimal::from(1),
            ask: Decimal::from(67255),
            ask_qty: Decimal::from(2),
            last: Decimal::from(67252),
            volume_24h: Decimal::from(1523),
            quote_volume_24h: Decimal::from(102000000),
            open_24h: Decimal::from(67000),
            high_24h: Decimal::from(68000),
            low_24h: Decimal::from(66500),
            change_24h: Decimal::from(252),
            change_pct_24h: Decimal::try_from(0.38).unwrap(),
            vwap: Decimal::from(67100),
            timestamp: Utc::now(),
        };

        let event = BinanceEvent::Ticker(ticker);
        let market_event = event.to_market_data_event();
        assert!(market_event.is_some());

        if let Some(MarketDataEvent::Ticker(t)) = market_event {
            assert_eq!(t.exchange, ExchangeId::Binance);
            assert_eq!(t.symbol, "BTC/USDT");
        }
    }

    #[test]
    fn test_connected_event() {
        let event = BinanceEvent::Connected;
        let market_event = event.to_market_data_event();
        assert!(market_event.is_some());

        if let Some(MarketDataEvent::ConnectionStatus {
            exchange,
            connected,
            ..
        }) = market_event
        {
            assert_eq!(exchange, ExchangeId::Binance);
            assert!(connected);
        }
    }
}
