//! Actor-based message processing system
//!
//! This module implements the Actor Model pattern for concurrent data processing.
//! Each actor runs in its own Tokio task and communicates via MPSC channels.
//!
//! ## Actors:
//! - `Router`: Central message dispatcher that routes data to storage
//! - `WebSocketActor`: Manages WebSocket connections with auto-reconnection
//! - `PollerActor`: Handles REST API polling with rate limiting

#![allow(dead_code)]

use anyhow::Result;

pub mod indicator;
mod poller;
pub mod router;
pub mod signal;
mod websocket;

pub use indicator::IndicatorActor;
pub use poller::PollerBuilder;
pub use router::Router;
pub use websocket::{WebSocketActor, WebSocketConfig};

/// Trait for exchange-specific connector implementations
///
/// This trait is defined here to avoid circular dependencies between
/// the actors and connectors modules.
pub trait ExchangeConnector: Send + Sync {
    /// Get the exchange name
    fn exchange_name(&self) -> &str;

    /// Build WebSocket configuration for a specific symbol
    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig;

    /// Parse a raw WebSocket message into DataMessage(s)
    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>>;

    /// Get the WebSocket URL for this exchange
    fn ws_url(&self) -> &str;

    /// Build subscription message for a symbol
    fn subscription_message(&self, symbol: &str) -> Option<String>;
}

/// Messages that flow through the system
#[derive(Debug, Clone)]
pub enum DataMessage {
    /// Real-time trade data from exchanges
    Trade(TradeData),

    /// Aggregated candle/OHLCV data
    Candle(CandleData),

    /// Alternative metrics (Fear & Greed, ETF flows, etc.)
    Metric(MetricData),

    /// Health/status messages
    Health(HealthData),
}

/// Trade tick data
#[derive(Debug, Clone)]
pub struct TradeData {
    /// Symbol (e.g., "BTC-USDT")
    pub symbol: String,

    /// Exchange that produced this trade
    pub exchange: String,

    /// Trade side (buy/sell)
    pub side: TradeSide,

    /// Execution price
    pub price: f64,

    /// Trade size/amount
    pub amount: f64,

    /// Exchange timestamp (milliseconds)
    pub exchange_ts: i64,

    /// Receipt timestamp (milliseconds) - when we received it
    pub receipt_ts: i64,

    /// Unique trade ID from exchange
    pub trade_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl std::fmt::Display for TradeSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeSide::Buy => write!(f, "buy"),
            TradeSide::Sell => write!(f, "sell"),
        }
    }
}

/// OHLCV candle data
#[derive(Debug, Clone)]
pub struct CandleData {
    pub symbol: String,
    pub exchange: String,
    pub open_time: i64,
    pub close_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub interval: String, // "1m", "5m", "1h", etc.
}

/// Alternative metric data
#[derive(Debug, Clone)]
pub struct MetricData {
    /// Metric type (e.g., "fear_greed", "etf_inflow", "dvol")
    pub metric_type: String,

    /// Asset this metric applies to (or "GLOBAL")
    pub asset: String,

    /// Data source
    pub source: String,

    /// Numeric value
    pub value: f64,

    /// Optional metadata (JSON string)
    pub meta: Option<String>,

    /// Timestamp (milliseconds)
    pub timestamp: i64,
}

/// Health check data
#[derive(Debug, Clone)]
pub struct HealthData {
    pub component: String,
    pub status: HealthStatus,
    pub message: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Actor statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ActorStats {
    pub messages_processed: u64,
    pub messages_failed: u64,
    pub last_activity_ts: i64,
    pub uptime_secs: u64,
}

impl ActorStats {
    pub fn new() -> Self {
        Self {
            messages_processed: 0,
            messages_failed: 0,
            last_activity_ts: chrono::Utc::now().timestamp_millis(),
            uptime_secs: 0,
        }
    }

    pub fn record_success(&mut self) {
        self.messages_processed += 1;
        self.last_activity_ts = chrono::Utc::now().timestamp_millis();
    }

    pub fn record_failure(&mut self) {
        self.messages_failed += 1;
        self.last_activity_ts = chrono::Utc::now().timestamp_millis();
    }
}
