//! Bybit Public WebSocket Client
//!
//! Real-time streaming of public market data from Bybit's public WebSocket channels.
//! This is FREE and does not require API keys.
//!
//! # Supported Channels
//!
//! - Tickers: Real-time ticker updates (best bid/ask, last price, 24h stats)
//! - Trades: Real-time trade stream
//! - Kline: Candlestick/OHLC data
//!
//! # Example
//!
//! ```ignore
//! use janus_execution::exchanges::bybit::BybitPublicWebSocket;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let ws = BybitPublicWebSocket::new(Default::default());
//!
//!     // Subscribe to tickers before starting
//!     ws.subscribe_ticker(&["BTCUSD", "ETHUSDT"]).await?;
//!
//!     // Start the WebSocket connection
//!     ws.start().await?;
//!
//!     // Listen for events
//!     let mut rx = ws.subscribe_events();
//!     while let Ok(event) = rx.recv().await {
//!         println!("Event: {:?}", event);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{ExecutionError, Result};
use crate::exchanges::market_data::{
    Candle, ExchangeId, MarketDataEvent, Ticker, Trade, TradeSide,
};
use crate::exchanges::metrics::{ExchangeMetrics, register_exchange};
use crate::execution::histogram::global_latency_histograms;
use chrono::{DateTime, TimeZone, Utc};
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

// ============================================================================
// Constants
// ============================================================================

/// Bybit public WebSocket URL for spot market (mainnet)
pub const WS_PUBLIC_SPOT_MAINNET: &str = "wss://stream.bybit.com/v5/public/spot";
/// Bybit public WebSocket URL for spot market (testnet)
pub const WS_PUBLIC_SPOT_TESTNET: &str = "wss://stream-testnet.bybit.com/v5/public/spot";
/// Bybit public WebSocket URL for linear (USDT perpetuals) market
pub const WS_PUBLIC_LINEAR_MAINNET: &str = "wss://stream.bybit.com/v5/public/linear";
/// Bybit public WebSocket URL for linear market (testnet)
pub const WS_PUBLIC_LINEAR_TESTNET: &str = "wss://stream-testnet.bybit.com/v5/public/linear";

/// Ping interval in seconds
const PING_INTERVAL_SECS: u64 = 20;
/// Reconnect delay base in seconds
const RECONNECT_DELAY_SECS: u64 = 5;
/// Maximum reconnect attempts before giving up
const MAX_RECONNECT_ATTEMPTS: u32 = 10;

// ============================================================================
// WebSocket Message Types (Bybit v5 API)
// ============================================================================

/// Bybit v5 WebSocket request
#[derive(Debug, Clone, Serialize)]
struct WsRequest {
    op: String,
    args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    req_id: Option<String>,
}

/// Bybit v5 WebSocket response (generic envelope)
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum WsResponse {
    /// Subscription confirmation
    Subscribe {
        success: bool,
        ret_msg: String,
        #[serde(default)]
        conn_id: Option<String>,
        #[serde(default)]
        req_id: Option<String>,
        op: String,
    },
    /// Pong response
    Pong {
        success: bool,
        ret_msg: String,
        #[serde(default)]
        conn_id: Option<String>,
        op: String,
    },
    /// Channel data update
    ChannelData {
        topic: String,
        #[serde(rename = "type")]
        update_type: String,
        data: serde_json::Value,
        #[serde(default)]
        ts: i64,
        #[serde(default)]
        cs: Option<i64>,
    },
}

// ============================================================================
// Ticker Data Structures
// ============================================================================

/// Raw ticker data from Bybit spot WebSocket
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitTickerData {
    pub symbol: String,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub last_price: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub high_price24h: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub low_price24h: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub prev_price24h: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub volume24h: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub turnover24h: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub price24h_pcnt: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub bid1_price: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub bid1_size: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub ask1_price: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_decimal_opt")]
    pub ask1_size: Option<Decimal>,
}

/// Parsed Bybit ticker with Decimal values
#[derive(Debug, Clone)]
pub struct BybitTicker {
    pub symbol: String,
    pub last: Decimal,
    pub high_24h: Decimal,
    pub low_24h: Decimal,
    pub prev_24h: Decimal,
    pub volume_24h: Decimal,
    pub turnover_24h: Decimal,
    pub change_pct_24h: Decimal,
    pub bid: Decimal,
    pub bid_qty: Decimal,
    pub ask: Decimal,
    pub ask_qty: Decimal,
    pub timestamp: DateTime<Utc>,
}

impl From<BybitTickerData> for BybitTicker {
    fn from(data: BybitTickerData) -> Self {
        let last = data.last_price.unwrap_or_default();
        let prev = data.prev_price24h.unwrap_or_default();

        Self {
            symbol: data.symbol,
            last,
            high_24h: data.high_price24h.unwrap_or_default(),
            low_24h: data.low_price24h.unwrap_or_default(),
            prev_24h: prev,
            volume_24h: data.volume24h.unwrap_or_default(),
            turnover_24h: data.turnover24h.unwrap_or_default(),
            change_pct_24h: data.price24h_pcnt.unwrap_or_default() * Decimal::from(100),
            bid: data.bid1_price.unwrap_or_default(),
            bid_qty: data.bid1_size.unwrap_or_default(),
            ask: data.ask1_price.unwrap_or_default(),
            ask_qty: data.ask1_size.unwrap_or_default(),
            timestamp: Utc::now(),
        }
    }
}

impl BybitTicker {
    /// Convert to unified Ticker type
    pub fn to_ticker(&self) -> Ticker {
        let change_24h = self.last - self.prev_24h;

        Ticker {
            exchange: ExchangeId::Bybit,
            symbol: normalize_symbol(&self.symbol),
            bid: self.bid,
            bid_qty: self.bid_qty,
            ask: self.ask,
            ask_qty: self.ask_qty,
            last: self.last,
            volume_24h: self.volume_24h,
            high_24h: self.high_24h,
            low_24h: self.low_24h,
            change_24h,
            change_pct_24h: self.change_pct_24h,
            vwap: None, // Bybit doesn't provide VWAP in ticker
            timestamp: self.timestamp,
        }
    }

    /// Merge a delta update into this ticker
    ///
    /// Bybit sends delta updates that only contain changed fields.
    /// This method merges those partial updates with the existing cached ticker
    /// to avoid losing bid/ask data when only one side is updated.
    pub fn merge_delta(&mut self, delta: &BybitTickerData) {
        // Only update fields that are present in the delta (Some values)
        if let Some(last) = delta.last_price {
            self.last = last;
        }
        if let Some(high) = delta.high_price24h {
            self.high_24h = high;
        }
        if let Some(low) = delta.low_price24h {
            self.low_24h = low;
        }
        if let Some(prev) = delta.prev_price24h {
            self.prev_24h = prev;
        }
        if let Some(vol) = delta.volume24h {
            self.volume_24h = vol;
        }
        if let Some(turnover) = delta.turnover24h {
            self.turnover_24h = turnover;
        }
        if let Some(pct) = delta.price24h_pcnt {
            self.change_pct_24h = pct * Decimal::from(100);
        }
        if let Some(bid) = delta.bid1_price {
            self.bid = bid;
        }
        if let Some(bid_qty) = delta.bid1_size {
            self.bid_qty = bid_qty;
        }
        if let Some(ask) = delta.ask1_price {
            self.ask = ask;
        }
        if let Some(ask_qty) = delta.ask1_size {
            self.ask_qty = ask_qty;
        }
        // Always update timestamp
        self.timestamp = Utc::now();
    }
}

// ============================================================================
// Trade Data Structures
// ============================================================================

/// Raw trade data from Bybit WebSocket
#[derive(Debug, Clone, Deserialize)]
pub struct BybitTradeData {
    #[serde(rename = "T")]
    pub timestamp: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "S")]
    pub side: String,
    #[serde(rename = "v", deserialize_with = "deserialize_decimal")]
    pub volume: Decimal,
    #[serde(rename = "p", deserialize_with = "deserialize_decimal")]
    pub price: Decimal,
    #[serde(rename = "i")]
    pub trade_id: String,
    #[serde(rename = "BT")]
    pub is_block_trade: bool,
}

/// Parsed Bybit trade
#[derive(Debug, Clone)]
pub struct BybitTrade {
    pub symbol: String,
    pub side: TradeSide,
    pub price: Decimal,
    pub quantity: Decimal,
    pub trade_id: String,
    pub timestamp: DateTime<Utc>,
}

impl From<BybitTradeData> for BybitTrade {
    fn from(data: BybitTradeData) -> Self {
        let side = if data.side == "Buy" {
            TradeSide::Buy
        } else {
            TradeSide::Sell
        };

        let timestamp = Utc
            .timestamp_millis_opt(data.timestamp)
            .single()
            .unwrap_or_else(Utc::now);

        Self {
            symbol: data.symbol,
            side,
            price: data.price,
            quantity: data.volume,
            trade_id: data.trade_id,
            timestamp,
        }
    }
}

impl BybitTrade {
    /// Convert to unified Trade type
    pub fn to_trade(&self) -> Trade {
        Trade {
            exchange: ExchangeId::Bybit,
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
// Kline (Candle) Data Structures
// ============================================================================

/// Raw kline data from Bybit WebSocket
#[derive(Debug, Clone, Deserialize)]
pub struct BybitKlineData {
    pub start: i64,
    pub end: i64,
    pub interval: String,
    #[serde(deserialize_with = "deserialize_decimal")]
    pub open: Decimal,
    #[serde(deserialize_with = "deserialize_decimal")]
    pub close: Decimal,
    #[serde(deserialize_with = "deserialize_decimal")]
    pub high: Decimal,
    #[serde(deserialize_with = "deserialize_decimal")]
    pub low: Decimal,
    #[serde(deserialize_with = "deserialize_decimal")]
    pub volume: Decimal,
    #[serde(deserialize_with = "deserialize_decimal")]
    pub turnover: Decimal,
    pub confirm: bool,
    pub timestamp: i64,
}

/// Parsed Bybit candle
#[derive(Debug, Clone)]
pub struct BybitCandle {
    pub symbol: String,
    pub interval: u32,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub turnover: Decimal,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub is_closed: bool,
}

impl BybitCandle {
    /// Create from raw data with symbol
    pub fn from_data(data: BybitKlineData, symbol: &str) -> Self {
        let interval = parse_interval(&data.interval);
        let open_time = Utc
            .timestamp_millis_opt(data.start)
            .single()
            .unwrap_or_else(Utc::now);
        let close_time = Utc
            .timestamp_millis_opt(data.end)
            .single()
            .unwrap_or_else(Utc::now);

        Self {
            symbol: symbol.to_string(),
            interval,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            volume: data.volume,
            turnover: data.turnover,
            open_time,
            close_time,
            is_closed: data.confirm,
        }
    }

    /// Convert to unified Candle type
    pub fn to_candle(&self) -> Candle {
        Candle {
            exchange: ExchangeId::Bybit,
            symbol: normalize_symbol(&self.symbol),
            interval: self.interval,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            quote_volume: Some(self.turnover),
            trades: None,
            vwap: None,
            open_time: self.open_time,
            close_time: self.close_time,
            is_closed: self.is_closed,
        }
    }
}

// ============================================================================
// Events
// ============================================================================

/// Events emitted by the Bybit public WebSocket client
#[derive(Debug, Clone)]
pub enum BybitPublicEvent {
    /// Ticker update
    Ticker(BybitTicker),
    /// Trade update
    Trade(BybitTrade),
    /// Candle update
    Candle(BybitCandle),
    /// WebSocket connected
    Connected,
    /// WebSocket disconnected
    Disconnected,
    /// Subscription confirmed
    Subscribed { channel: String, symbol: String },
    /// Error occurred
    Error(String),
}

impl BybitPublicEvent {
    /// Convert to unified MarketDataEvent
    pub fn to_market_data_event(&self) -> Option<MarketDataEvent> {
        match self {
            BybitPublicEvent::Ticker(t) => Some(MarketDataEvent::Ticker(t.to_ticker())),
            BybitPublicEvent::Trade(t) => Some(MarketDataEvent::Trade(t.to_trade())),
            BybitPublicEvent::Candle(c) => Some(MarketDataEvent::Candle(c.to_candle())),
            BybitPublicEvent::Connected => Some(MarketDataEvent::ConnectionStatus {
                exchange: ExchangeId::Bybit,
                connected: true,
                message: Some("Connected".to_string()),
                timestamp: Utc::now(),
            }),
            BybitPublicEvent::Disconnected => Some(MarketDataEvent::ConnectionStatus {
                exchange: ExchangeId::Bybit,
                connected: false,
                message: Some("Disconnected".to_string()),
                timestamp: Utc::now(),
            }),
            BybitPublicEvent::Error(msg) => Some(MarketDataEvent::Error {
                exchange: ExchangeId::Bybit,
                code: None,
                message: msg.clone(),
                timestamp: Utc::now(),
            }),
            BybitPublicEvent::Subscribed { .. } => None,
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Market category for Bybit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BybitCategory {
    /// Spot trading
    #[default]
    Spot,
    /// Linear perpetuals (USDT settled)
    Linear,
}

/// Configuration for the Bybit public WebSocket client
#[derive(Debug, Clone)]
pub struct BybitPublicWsConfig {
    /// Use testnet instead of mainnet
    pub testnet: bool,
    /// Market category
    pub category: BybitCategory,
    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,
    /// Maximum reconnect attempts
    pub max_reconnect_attempts: u32,
    /// Base reconnect delay in seconds
    pub reconnect_delay_secs: u64,
    /// Ping interval in seconds
    pub ping_interval_secs: u64,
}

impl Default for BybitPublicWsConfig {
    fn default() -> Self {
        Self {
            testnet: false,
            category: BybitCategory::Spot,
            auto_reconnect: true,
            max_reconnect_attempts: MAX_RECONNECT_ATTEMPTS,
            reconnect_delay_secs: RECONNECT_DELAY_SECS,
            ping_interval_secs: PING_INTERVAL_SECS,
        }
    }
}

impl BybitPublicWsConfig {
    /// Get the WebSocket URL based on configuration
    pub fn ws_url(&self) -> &'static str {
        match (self.testnet, self.category) {
            (false, BybitCategory::Spot) => WS_PUBLIC_SPOT_MAINNET,
            (true, BybitCategory::Spot) => WS_PUBLIC_SPOT_TESTNET,
            (false, BybitCategory::Linear) => WS_PUBLIC_LINEAR_MAINNET,
            (true, BybitCategory::Linear) => WS_PUBLIC_LINEAR_TESTNET,
        }
    }
}

// ============================================================================
// WebSocket Client
// ============================================================================

/// Subscription tracking
#[derive(Default)]
struct Subscriptions {
    tickers: Vec<String>,
    trades: Vec<String>,
    candles: HashMap<String, Vec<String>>, // interval -> symbols
}

/// Bybit public WebSocket client for market data streaming
pub struct BybitPublicWebSocket {
    /// Configuration
    config: BybitPublicWsConfig,
    /// Event broadcast sender
    event_tx: broadcast::Sender<BybitPublicEvent>,
    /// Active subscriptions
    subscriptions: Arc<RwLock<Subscriptions>>,
    /// Whether the client is running
    is_running: Arc<RwLock<bool>>,
    /// Reconnect count
    reconnect_count: Arc<RwLock<u32>>,
    /// Ticker cache
    ticker_cache: Arc<RwLock<HashMap<String, BybitTicker>>>,
    /// Candle cache (symbol:interval -> candle)
    candle_cache: Arc<RwLock<HashMap<String, BybitCandle>>>,
    /// Request ID counter
    request_id: Arc<RwLock<u64>>,
    /// Metrics for WebSocket connection health
    metrics: Arc<ExchangeMetrics>,
}

impl BybitPublicWebSocket {
    /// Create a new Bybit public WebSocket client
    pub fn new(config: BybitPublicWsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(10000);
        // Register with global metrics registry
        let metrics = register_exchange(ExchangeId::Bybit);

        Self {
            config,
            event_tx,
            subscriptions: Arc::new(RwLock::new(Subscriptions::default())),
            is_running: Arc::new(RwLock::new(false)),
            reconnect_count: Arc::new(RwLock::new(0)),
            ticker_cache: Arc::new(RwLock::new(HashMap::new())),
            candle_cache: Arc::new(RwLock::new(HashMap::new())),
            request_id: Arc::new(RwLock::new(0)),
            metrics,
        }
    }

    /// Get metrics for this WebSocket connection
    pub fn metrics(&self) -> &Arc<ExchangeMetrics> {
        &self.metrics
    }

    /// Create a client configured for testnet
    pub fn testnet() -> Self {
        Self::new(BybitPublicWsConfig {
            testnet: true,
            ..Default::default()
        })
    }

    /// Create a client for linear perpetuals
    pub fn linear(testnet: bool) -> Self {
        Self::new(BybitPublicWsConfig {
            testnet,
            category: BybitCategory::Linear,
            ..Default::default()
        })
    }

    /// Subscribe to events
    pub fn subscribe_events(&self) -> broadcast::Receiver<BybitPublicEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender for external use
    pub fn event_sender(&self) -> broadcast::Sender<BybitPublicEvent> {
        self.event_tx.clone()
    }

    /// Check if the client is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get cached ticker for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Option<BybitTicker> {
        self.ticker_cache.read().await.get(symbol).cloned()
    }

    /// Get all cached tickers
    pub async fn get_all_tickers(&self) -> HashMap<String, BybitTicker> {
        self.ticker_cache.read().await.clone()
    }

    /// Get cached candle for a symbol and interval
    pub async fn get_candle(&self, symbol: &str, interval: &str) -> Option<BybitCandle> {
        let key = format!("{}:{}", symbol, interval);
        self.candle_cache.read().await.get(&key).cloned()
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
        let ticker_cache = self.ticker_cache.clone();
        let candle_cache = self.candle_cache.clone();
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
                candle_cache,
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
        info!("Bybit public WebSocket client stopped");
    }

    /// Subscribe to ticker updates for the given symbols
    ///
    /// Symbols should be in Bybit format (e.g., "BTCUSD", "ETHUSDT")
    pub async fn subscribe_ticker(&self, symbols: &[&str]) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            let sym = to_bybit_symbol(symbol);
            if !subs.tickers.contains(&sym) {
                subs.tickers.push(sym);
            }
        }
        Ok(())
    }

    /// Subscribe to trade updates for the given symbols
    pub async fn subscribe_trades(&self, symbols: &[&str]) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            let sym = to_bybit_symbol(symbol);
            if !subs.trades.contains(&sym) {
                subs.trades.push(sym);
            }
        }
        Ok(())
    }

    /// Subscribe to candle/kline updates for the given symbols
    ///
    /// Interval format: "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"
    pub async fn subscribe_candles(&self, symbols: &[&str], interval: &str) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        let entry = subs.candles.entry(interval.to_string()).or_default();
        for symbol in symbols {
            let sym = to_bybit_symbol(symbol);
            if !entry.contains(&sym) {
                entry.push(sym);
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
        config: BybitPublicWsConfig,
        event_tx: broadcast::Sender<BybitPublicEvent>,
        subscriptions: Arc<RwLock<Subscriptions>>,
        is_running: Arc<RwLock<bool>>,
        reconnect_count: Arc<RwLock<u32>>,
        ticker_cache: Arc<RwLock<HashMap<String, BybitTicker>>>,
        candle_cache: Arc<RwLock<HashMap<String, BybitCandle>>>,
        request_id: Arc<RwLock<u64>>,
        metrics: Arc<ExchangeMetrics>,
    ) {
        while *is_running.read().await {
            let count = *reconnect_count.read().await;
            if count > 0 && config.auto_reconnect {
                // Exponential backoff with max of 60 seconds
                let delay =
                    std::cmp::min(config.reconnect_delay_secs * (2_u64.pow(count.min(6))), 60);
                info!(
                    "Bybit: Reconnecting in {} seconds (attempt {})",
                    delay, count
                );
                // Record reconnect attempt in metrics
                metrics.record_reconnect();
                sleep(Duration::from_secs(delay)).await;
            }

            if count >= config.max_reconnect_attempts {
                error!("Bybit: Max reconnection attempts reached");
                metrics.record_error();
                let _ = event_tx.send(BybitPublicEvent::Error(
                    "Max reconnection attempts exceeded".to_string(),
                ));
                break;
            }

            match Self::connect_and_run(
                &config,
                &event_tx,
                &subscriptions,
                &ticker_cache,
                &candle_cache,
                &request_id,
                &metrics,
            )
            .await
            {
                Ok(_) => {
                    info!("Bybit: WebSocket connection closed normally");
                    metrics.record_disconnected();
                    *reconnect_count.write().await = 0;
                }
                Err(e) => {
                    error!("Bybit: WebSocket error: {}", e);
                    metrics.record_error();
                    metrics.record_disconnected();
                    let _ = event_tx.send(BybitPublicEvent::Error(e.to_string()));
                    let _ = event_tx.send(BybitPublicEvent::Disconnected);
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
        config: &BybitPublicWsConfig,
        event_tx: &broadcast::Sender<BybitPublicEvent>,
        subscriptions: &Arc<RwLock<Subscriptions>>,
        ticker_cache: &Arc<RwLock<HashMap<String, BybitTicker>>>,
        candle_cache: &Arc<RwLock<HashMap<String, BybitCandle>>>,
        request_id: &Arc<RwLock<u64>>,
        metrics: &Arc<ExchangeMetrics>,
    ) -> Result<()> {
        let url = config.ws_url();
        info!("Bybit: Connecting to {}", url);

        let (ws_stream, _) = connect_async(url).await.map_err(|e| {
            metrics.record_error();
            ExecutionError::WebSocketError(format!("Connection failed: {}", e))
        })?;

        info!("Bybit: WebSocket connected");
        metrics.record_connected();
        let (mut write, mut read) = ws_stream.split();

        let _ = event_tx.send(BybitPublicEvent::Connected);

        // Build subscription messages
        let subs = subscriptions.read().await;

        // Subscribe to tickers
        if !subs.tickers.is_empty() {
            let args: Vec<String> = subs
                .tickers
                .iter()
                .map(|s| format!("tickers.{}", s))
                .collect();

            let mut id = request_id.write().await;
            *id += 1;
            let req = WsRequest {
                op: "subscribe".to_string(),
                args,
                req_id: Some(id.to_string()),
            };

            let msg = serde_json::to_string(&req)
                .map_err(|e| ExecutionError::Serialization(e.to_string()))?;
            debug!("Bybit: Sending subscription: {}", msg);
            write
                .send(Message::Text(msg.into()))
                .await
                .map_err(|e| ExecutionError::WebSocketError(format!("Subscribe failed: {}", e)))?;
        }

        // Subscribe to trades
        if !subs.trades.is_empty() {
            let args: Vec<String> = subs
                .trades
                .iter()
                .map(|s| format!("publicTrade.{}", s))
                .collect();

            let mut id = request_id.write().await;
            *id += 1;
            let req = WsRequest {
                op: "subscribe".to_string(),
                args,
                req_id: Some(id.to_string()),
            };

            let msg = serde_json::to_string(&req)
                .map_err(|e| ExecutionError::Serialization(e.to_string()))?;
            debug!("Bybit: Sending trade subscription: {}", msg);
            write
                .send(Message::Text(msg.into()))
                .await
                .map_err(|e| ExecutionError::WebSocketError(format!("Subscribe failed: {}", e)))?;
        }

        // Subscribe to candles
        for (interval, symbols) in subs.candles.iter() {
            if !symbols.is_empty() {
                let args: Vec<String> = symbols
                    .iter()
                    .map(|s| format!("kline.{}.{}", interval, s))
                    .collect();

                let mut id = request_id.write().await;
                *id += 1;
                let req = WsRequest {
                    op: "subscribe".to_string(),
                    args,
                    req_id: Some(id.to_string()),
                };

                let msg = serde_json::to_string(&req)
                    .map_err(|e| ExecutionError::Serialization(e.to_string()))?;
                debug!("Bybit: Sending kline subscription: {}", msg);
                write.send(Message::Text(msg.into())).await.map_err(|e| {
                    ExecutionError::WebSocketError(format!("Subscribe failed: {}", e))
                })?;
            }
        }

        // Update subscription count in metrics before dropping subs
        let sub_count = subs.tickers.len()
            + subs.trades.len()
            + subs.candles.values().map(|v| v.len()).sum::<usize>();
        metrics.set_subscription_count(sub_count as u64);

        drop(subs);

        // Spawn ping task
        let ping_interval = config.ping_interval_secs;
        let (ping_tx, mut ping_rx) = tokio::sync::mpsc::channel::<()>(1);

        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(ping_interval)).await;
                if ping_tx.send(()).await.is_err() {
                    break;
                }
            }
        });

        // Message loop
        loop {
            tokio::select! {
                // Handle incoming messages
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            // Record message received
                            metrics.record_message();
                            Self::handle_message(&text, event_tx, ticker_cache, candle_cache).await;
                        }
                        Some(Ok(Message::Ping(data))) => {
                            debug!("Bybit: Received ping, sending pong");
                            metrics.record_message();
                            if let Err(e) = write.send(Message::Pong(data)).await {
                                warn!("Bybit: Failed to send pong: {}", e);
                            }
                        }
                        Some(Ok(Message::Pong(_))) => {
                            debug!("Bybit: Received pong");
                            metrics.record_message();
                        }
                        Some(Ok(Message::Close(frame))) => {
                            info!("Bybit: WebSocket closed: {:?}", frame);
                            let _ = event_tx.send(BybitPublicEvent::Disconnected);
                            break;
                        }
                        Some(Ok(Message::Binary(_))) => {
                            debug!("Bybit: Received binary message (ignored)");
                            metrics.record_message();
                        }
                        Some(Ok(Message::Frame(_))) => {
                            // Raw frame, ignore
                        }
                        Some(Err(e)) => {
                            error!("Bybit: WebSocket error: {}", e);
                            metrics.record_error();
                            return Err(ExecutionError::WebSocketError(e.to_string()));
                        }
                        None => {
                            info!("Bybit: WebSocket stream ended");
                            break;
                        }
                    }
                }

                // Send ping
                _ = ping_rx.recv() => {
                    let req = WsRequest {
                        op: "ping".to_string(),
                        args: Vec::new(),
                        req_id: None,
                    };
                    if let Ok(msg) = serde_json::to_string(&req) {
                        debug!("Bybit: Sending ping");
                        if let Err(e) = write.send(Message::Text(msg.into())).await {
                            warn!("Bybit: Failed to send ping: {}", e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle an incoming WebSocket message
    async fn handle_message(
        text: &str,
        event_tx: &broadcast::Sender<BybitPublicEvent>,
        ticker_cache: &Arc<RwLock<HashMap<String, BybitTicker>>>,
        candle_cache: &Arc<RwLock<HashMap<String, BybitCandle>>>,
    ) {
        let start = Instant::now();
        debug!("Bybit: Received: {}", text);

        // Try to parse as WsResponse
        match serde_json::from_str::<WsResponse>(text) {
            Ok(WsResponse::Subscribe {
                success,
                ret_msg,
                op,
                ..
            }) => {
                if success {
                    debug!("Bybit: Subscription confirmed: {}", ret_msg);
                    let _ = event_tx.send(BybitPublicEvent::Subscribed {
                        channel: op,
                        symbol: ret_msg,
                    });
                } else {
                    warn!("Bybit: Subscription failed: {}", ret_msg);
                    let _ = event_tx.send(BybitPublicEvent::Error(format!(
                        "Subscription failed: {}",
                        ret_msg
                    )));
                }
            }
            Ok(WsResponse::Pong {
                success, ret_msg, ..
            }) => {
                if success {
                    debug!("Bybit: Pong received");
                } else {
                    warn!("Bybit: Pong error: {}", ret_msg);
                }
            }
            Ok(WsResponse::ChannelData {
                topic,
                data,
                update_type,
                ..
            }) => {
                // Parse based on topic prefix
                if topic.starts_with("tickers.") {
                    Self::handle_ticker_data(&topic, &data, event_tx, ticker_cache).await;
                } else if topic.starts_with("publicTrade.") {
                    Self::handle_trade_data(&data, event_tx).await;
                } else if topic.starts_with("kline.") {
                    Self::handle_kline_data(&topic, &data, event_tx, candle_cache).await;
                } else {
                    debug!("Bybit: Unknown topic {} (type: {})", topic, update_type);
                }

                // Record WebSocket message processing latency
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                global_latency_histograms().record_websocket_message("bybit", duration_ms);
            }
            Err(e) => {
                // Check if it's a pong response (different format)
                if text.contains("\"op\":\"pong\"") {
                    debug!("Bybit: Pong received (alt format)");
                } else {
                    warn!("Bybit: Failed to parse message: {} - {}", e, text);
                }
            }
        }
    }

    /// Handle ticker data
    async fn handle_ticker_data(
        topic: &str,
        data: &serde_json::Value,
        event_tx: &broadcast::Sender<BybitPublicEvent>,
        ticker_cache: &Arc<RwLock<HashMap<String, BybitTicker>>>,
    ) {
        // Extract symbol from topic (e.g., "tickers.BTCUSD")
        let symbol = topic.strip_prefix("tickers.").unwrap_or("");
        let normalized_symbol = normalize_symbol(symbol);

        match serde_json::from_value::<BybitTickerData>(data.clone()) {
            Ok(ticker_data) => {
                // Get or create ticker, then merge delta update
                let mut cache = ticker_cache.write().await;
                let ticker = cache
                    .entry(normalized_symbol)
                    .or_insert_with(|| BybitTicker::from(ticker_data.clone()));

                // Merge delta update to preserve existing bid/ask when only one side updates
                ticker.merge_delta(&ticker_data);
                let ticker_clone = ticker.clone();
                drop(cache);

                // Send event with merged ticker
                let _ = event_tx.send(BybitPublicEvent::Ticker(ticker_clone));
            }
            Err(e) => {
                warn!("Bybit: Failed to parse ticker data: {} - {:?}", e, data);
            }
        }
    }

    /// Handle trade data
    async fn handle_trade_data(
        data: &serde_json::Value,
        event_tx: &broadcast::Sender<BybitPublicEvent>,
    ) {
        // Trades come as an array
        if let Some(trades) = data.as_array() {
            for trade_data in trades {
                match serde_json::from_value::<BybitTradeData>(trade_data.clone()) {
                    Ok(trade_data) => {
                        let trade: BybitTrade = trade_data.into();
                        let _ = event_tx.send(BybitPublicEvent::Trade(trade));
                    }
                    Err(e) => {
                        warn!(
                            "Bybit: Failed to parse trade data: {} - {:?}",
                            e, trade_data
                        );
                    }
                }
            }
        }
    }

    /// Handle kline/candle data
    async fn handle_kline_data(
        topic: &str,
        data: &serde_json::Value,
        event_tx: &broadcast::Sender<BybitPublicEvent>,
        candle_cache: &Arc<RwLock<HashMap<String, BybitCandle>>>,
    ) {
        // Extract interval and symbol from topic (e.g., "kline.1.BTCUSD")
        let parts: Vec<&str> = topic.split('.').collect();
        if parts.len() < 3 {
            warn!("Bybit: Invalid kline topic format: {}", topic);
            return;
        }
        let interval = parts[1];
        let symbol = parts[2];

        // Klines come as an array
        if let Some(klines) = data.as_array() {
            for kline_data in klines {
                match serde_json::from_value::<BybitKlineData>(kline_data.clone()) {
                    Ok(kline_data) => {
                        let candle = BybitCandle::from_data(kline_data, symbol);
                        let cache_key = format!("{}:{}", symbol, interval);

                        // Update cache
                        candle_cache.write().await.insert(cache_key, candle.clone());

                        // Send event
                        let _ = event_tx.send(BybitPublicEvent::Candle(candle));
                    }
                    Err(e) => {
                        warn!(
                            "Bybit: Failed to parse kline data: {} - {:?}",
                            e, kline_data
                        );
                    }
                }
            }
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Deserialize a string to Decimal
fn deserialize_decimal<'de, D>(deserializer: D) -> std::result::Result<Decimal, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    Decimal::from_str(&s).map_err(serde::de::Error::custom)
}

/// Deserialize an optional string to Option<Decimal>
fn deserialize_decimal_opt<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Decimal>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Deserialize::deserialize(deserializer)?;
    match opt {
        Some(s) if !s.is_empty() => Decimal::from_str(&s)
            .map(Some)
            .map_err(serde::de::Error::custom),
        _ => Ok(None),
    }
}

/// Convert a normalized symbol (e.g., "BTC/USDT") to Bybit format ("BTCUSD")
fn to_bybit_symbol(symbol: &str) -> String {
    symbol.replace("/", "").replace("-", "").to_uppercase()
}

/// Normalize a Bybit symbol (e.g., "BTCUSD") to standard format ("BTC/USDT")
fn normalize_symbol(symbol: &str) -> String {
    let symbol = symbol.to_uppercase();

    // Common quote currencies (order matters - check longer ones first)
    let quotes = ["USDT", "USDC", "USD", "BTC", "ETH"];

    for quote in quotes {
        if symbol.ends_with(quote) {
            let base = &symbol[..symbol.len() - quote.len()];
            if !base.is_empty() {
                return format!("{}/{}", base, quote);
            }
        }
    }

    // Return as-is if no match
    symbol
}

/// Parse interval string to minutes
fn parse_interval(interval: &str) -> u32 {
    match interval {
        "1" => 1,
        "3" => 3,
        "5" => 5,
        "15" => 15,
        "30" => 30,
        "60" => 60,
        "120" => 120,
        "240" => 240,
        "360" => 360,
        "720" => 720,
        "D" => 1440,  // 1 day = 1440 minutes
        "W" => 10080, // 1 week = 10080 minutes
        "M" => 43200, // ~1 month = 43200 minutes
        _ => interval.parse().unwrap_or(1),
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
        let config = BybitPublicWsConfig::default();
        assert!(!config.testnet);
        assert_eq!(config.category, BybitCategory::Spot);
        assert!(config.auto_reconnect);
        assert_eq!(config.ws_url(), WS_PUBLIC_SPOT_MAINNET);
    }

    #[test]
    fn test_config_testnet() {
        let config = BybitPublicWsConfig {
            testnet: true,
            ..Default::default()
        };
        assert_eq!(config.ws_url(), WS_PUBLIC_SPOT_TESTNET);
    }

    #[test]
    fn test_config_linear() {
        let config = BybitPublicWsConfig {
            category: BybitCategory::Linear,
            ..Default::default()
        };
        assert_eq!(config.ws_url(), WS_PUBLIC_LINEAR_MAINNET);
    }

    #[test]
    fn test_to_bybit_symbol() {
        assert_eq!(to_bybit_symbol("BTC/USDT"), "BTCUSDT");
        assert_eq!(to_bybit_symbol("eth/usdt"), "ETHUSDT");
        assert_eq!(to_bybit_symbol("BTC-USDT"), "BTCUSDT");
        assert_eq!(to_bybit_symbol("BTCUSDT"), "BTCUSDT");
    }

    #[test]
    fn test_normalize_symbol() {
        assert_eq!(normalize_symbol("BTCUSDT"), "BTC/USDT");
        assert_eq!(normalize_symbol("ETHUSDT"), "ETH/USDT");
        assert_eq!(normalize_symbol("BTCUSDC"), "BTC/USDC");
        assert_eq!(normalize_symbol("btcusdt"), "BTC/USDT");
    }

    #[test]
    fn test_parse_interval() {
        assert_eq!(parse_interval("1"), 1);
        assert_eq!(parse_interval("5"), 5);
        assert_eq!(parse_interval("60"), 60);
        assert_eq!(parse_interval("D"), 1440);
        assert_eq!(parse_interval("W"), 10080);
    }

    #[tokio::test]
    async fn test_websocket_creation() {
        let ws = BybitPublicWebSocket::new(Default::default());
        assert!(!ws.is_running().await);
    }

    #[tokio::test]
    async fn test_subscribe_ticker() {
        let ws = BybitPublicWebSocket::new(Default::default());
        ws.subscribe_ticker(&["BTC/USDT", "ETH/USDT"])
            .await
            .unwrap();

        let tickers = ws.subscribed_tickers().await;
        assert_eq!(tickers.len(), 2);
        assert!(tickers.contains(&"BTCUSDT".to_string()));
        assert!(tickers.contains(&"ETHUSDT".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_trades() {
        let ws = BybitPublicWebSocket::new(Default::default());
        ws.subscribe_trades(&["BTCUSDT"]).await.unwrap();

        let subs = ws.subscriptions.read().await;
        assert!(subs.trades.contains(&"BTCUSDT".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_candles() {
        let ws = BybitPublicWebSocket::new(Default::default());
        ws.subscribe_candles(&["BTCUSDT"], "5").await.unwrap();

        let subs = ws.subscriptions.read().await;
        let candle_syms = subs.candles.get("5").unwrap();
        assert!(candle_syms.contains(&"BTCUSDT".to_string()));
    }

    #[test]
    fn test_bybit_ticker_conversion() {
        let ticker_data = BybitTickerData {
            symbol: "BTCUSDT".to_string(),
            last_price: Some(Decimal::from(50000)),
            high_price24h: Some(Decimal::from(51000)),
            low_price24h: Some(Decimal::from(49000)),
            prev_price24h: Some(Decimal::from(49500)),
            volume24h: Some(Decimal::from(1000)),
            turnover24h: Some(Decimal::from(50000000)),
            price24h_pcnt: Some(Decimal::from_str("0.0101").unwrap()),
            bid1_price: Some(Decimal::from(49990)),
            bid1_size: Some(Decimal::from(5)),
            ask1_price: Some(Decimal::from(50010)),
            ask1_size: Some(Decimal::from(3)),
        };

        let ticker: BybitTicker = ticker_data.into();
        assert_eq!(ticker.symbol, "BTCUSDT");
        assert_eq!(ticker.last, Decimal::from(50000));
        assert_eq!(ticker.bid, Decimal::from(49990));
        assert_eq!(ticker.ask, Decimal::from(50010));

        // Test conversion to unified type
        let unified = ticker.to_ticker();
        assert_eq!(unified.exchange, ExchangeId::Bybit);
        assert_eq!(unified.symbol, "BTC/USDT");
        assert_eq!(unified.last, Decimal::from(50000));
    }

    #[test]
    fn test_bybit_trade_conversion() {
        let trade_data = BybitTradeData {
            timestamp: 1700000000000,
            symbol: "BTCUSDT".to_string(),
            side: "Buy".to_string(),
            volume: Decimal::from_str("0.5").unwrap(),
            price: Decimal::from(50000),
            trade_id: "12345".to_string(),
            is_block_trade: false,
        };

        let trade: BybitTrade = trade_data.into();
        assert_eq!(trade.symbol, "BTCUSDT");
        assert!(matches!(trade.side, TradeSide::Buy));
        assert_eq!(trade.price, Decimal::from(50000));

        // Test conversion to unified type
        let unified = trade.to_trade();
        assert_eq!(unified.exchange, ExchangeId::Bybit);
        assert_eq!(unified.symbol, "BTC/USDT");
    }

    #[test]
    fn test_event_to_market_data() {
        let ticker = BybitTicker {
            symbol: "BTCUSDT".to_string(),
            last: Decimal::from(50000),
            high_24h: Decimal::from(51000),
            low_24h: Decimal::from(49000),
            prev_24h: Decimal::from(49500),
            volume_24h: Decimal::from(1000),
            turnover_24h: Decimal::from(50000000),
            change_pct_24h: Decimal::from_str("1.01").unwrap(),
            bid: Decimal::from(49990),
            bid_qty: Decimal::from(5),
            ask: Decimal::from(50010),
            ask_qty: Decimal::from(3),
            timestamp: Utc::now(),
        };

        let event = BybitPublicEvent::Ticker(ticker);
        let market_event = event.to_market_data_event();
        assert!(market_event.is_some());

        if let Some(MarketDataEvent::Ticker(t)) = market_event {
            assert_eq!(t.exchange, ExchangeId::Bybit);
            assert_eq!(t.symbol, "BTC/USDT");
        }
    }

    #[test]
    fn test_connected_event() {
        let event = BybitPublicEvent::Connected;
        let market_event = event.to_market_data_event();
        assert!(market_event.is_some());

        if let Some(MarketDataEvent::ConnectionStatus {
            exchange,
            connected,
            ..
        }) = market_event
        {
            assert_eq!(exchange, ExchangeId::Bybit);
            assert!(connected);
        }
    }

    #[test]
    fn test_ticker_merge_delta() {
        // Create initial ticker with full data
        let mut ticker = BybitTicker {
            symbol: "BTCUSDT".to_string(),
            last: Decimal::from(50000),
            high_24h: Decimal::from(51000),
            low_24h: Decimal::from(49000),
            prev_24h: Decimal::from(49500),
            volume_24h: Decimal::from(1000),
            turnover_24h: Decimal::from(50000000),
            change_pct_24h: Decimal::from_str("1.01").unwrap(),
            bid: Decimal::from(49990),
            bid_qty: Decimal::from(5),
            ask: Decimal::from(50010),
            ask_qty: Decimal::from(3),
            timestamp: Utc::now(),
        };

        // Delta update with only bid price changed (simulates Bybit partial update)
        let delta = BybitTickerData {
            symbol: "BTCUSDT".to_string(),
            last_price: None,
            high_price24h: None,
            low_price24h: None,
            prev_price24h: None,
            volume24h: None,
            turnover24h: None,
            price24h_pcnt: None,
            bid1_price: Some(Decimal::from(49995)), // Only bid changed
            bid1_size: None,
            ask1_price: None, // Ask not in update
            ask1_size: None,
        };

        ticker.merge_delta(&delta);

        // Bid should be updated
        assert_eq!(ticker.bid, Decimal::from(49995));
        // Ask should be preserved from original
        assert_eq!(ticker.ask, Decimal::from(50010));
        // Other fields should be preserved
        assert_eq!(ticker.last, Decimal::from(50000));
        assert_eq!(ticker.bid_qty, Decimal::from(5));

        // Another delta with only ask price changed
        let delta2 = BybitTickerData {
            symbol: "BTCUSDT".to_string(),
            last_price: None,
            high_price24h: None,
            low_price24h: None,
            prev_price24h: None,
            volume24h: None,
            turnover24h: None,
            price24h_pcnt: None,
            bid1_price: None,
            bid1_size: None,
            ask1_price: Some(Decimal::from(50015)), // Only ask changed
            ask1_size: Some(Decimal::from(10)),
        };

        ticker.merge_delta(&delta2);

        // Ask should be updated
        assert_eq!(ticker.ask, Decimal::from(50015));
        assert_eq!(ticker.ask_qty, Decimal::from(10));
        // Bid should still be preserved from previous merge
        assert_eq!(ticker.bid, Decimal::from(49995));
    }
}
