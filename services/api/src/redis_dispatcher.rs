//! Redis signal dispatcher for publishing signals to janus-forward service.
//!
//! This module handles publishing signals and heartbeats via Redis Pub/Sub
//! to communicate with the Rust forward service.
//!
//! This is the Rust port of `gateway/src/clients/redis_publisher.py`.

use chrono::Utc;
use redis::AsyncCommands;
use redis::aio::MultiplexedConnection;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Redis channel names (must match Python/Rust constants)
pub const SIGNAL_CHANNEL: &str = "janus:signals";
pub const HEARTBEAT_CHANNEL: &str = "janus:heartbeat";

/// Errors that can occur during signal dispatch
#[derive(Error, Debug)]
pub enum DispatcherError {
    #[error("Redis connection error: {0}")]
    ConnectionError(#[from] redis::RedisError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Dispatcher not connected")]
    NotConnected,
}

/// Signal data structure for dispatch to Rust forward service.
///
/// This matches the expected format in janus-forward's SignalReceiver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Trading symbol (e.g., "BTCUSD")
    pub symbol: String,

    /// Order side: "Buy" or "Sell" (matches Rust OrderSide enum)
    pub side: String,

    /// Signal strength (0.0 to 1.0)
    pub strength: f64,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Predicted trade duration in seconds (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_duration_seconds: Option<i64>,

    /// Suggested entry price (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_price: Option<f64>,

    /// Stop loss price (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_loss: Option<f64>,

    /// Take profit price (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub take_profit: Option<f64>,
}

impl Signal {
    /// Create a new signal with required fields.
    pub fn new(symbol: impl Into<String>, side: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            side: side.into(),
            strength: 0.5,
            confidence: 0.5,
            predicted_duration_seconds: None,
            entry_price: None,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Set signal strength.
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set confidence level.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set price levels.
    pub fn with_prices(
        mut self,
        entry: Option<f64>,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
    ) -> Self {
        self.entry_price = entry;
        self.stop_loss = stop_loss;
        self.take_profit = take_profit;
        self
    }

    /// Normalize side string to match Rust OrderSide enum.
    #[allow(dead_code)]
    fn normalize_side(side: &str) -> String {
        let upper = side.to_uppercase();
        if upper == "SELL" || upper == "SHORT" {
            "Sell".to_string()
        } else {
            "Buy".to_string()
        }
    }

    /// Create signal from a raw dictionary/JSON value.
    #[allow(dead_code)]
    pub fn from_raw(raw: serde_json::Value) -> Result<Self, serde_json::Error> {
        let mut signal: Signal = serde_json::from_value(raw)?;
        signal.side = Self::normalize_side(&signal.side);
        Ok(signal)
    }
}

/// Heartbeat data sent to forward service (Dead Man's Switch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    /// ISO 8601 timestamp
    pub timestamp: String,

    /// Gateway identifier
    pub gateway_id: String,
}

impl Heartbeat {
    /// Create a new heartbeat with current timestamp.
    pub fn new(gateway_id: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now().to_rfc3339(),
            gateway_id: gateway_id.into(),
        }
    }
}

/// Redis signal dispatcher.
///
/// Dispatches signals from the gateway to the Rust forward service via Redis Pub/Sub.
/// This is the Rust counterpart to the Python SignalDispatcher.
pub struct SignalDispatcher {
    /// Redis connection URL
    redis_url: String,

    /// Multiplexed connection (thread-safe)
    connection: Arc<RwLock<Option<MultiplexedConnection>>>,

    /// Gateway identifier for heartbeats
    gateway_id: String,
}

impl SignalDispatcher {
    /// Create a new signal dispatcher.
    ///
    /// # Arguments
    /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379/0")
    pub fn new(redis_url: impl Into<String>) -> Self {
        Self {
            redis_url: redis_url.into(),
            connection: Arc::new(RwLock::new(None)),
            gateway_id: "janus-gateway-rs".to_string(),
        }
    }

    /// Set custom gateway ID for heartbeats.
    #[allow(dead_code)]
    pub fn with_gateway_id(mut self, id: impl Into<String>) -> Self {
        self.gateway_id = id.into();
        self
    }

    /// Connect to Redis.
    pub async fn connect(&self) -> Result<(), DispatcherError> {
        let client = redis::Client::open(self.redis_url.as_str())?;
        let conn = client.get_multiplexed_async_connection().await?;

        // Test connection with PING
        let mut test_conn = conn.clone();
        let _: String = redis::cmd("PING").query_async(&mut test_conn).await?;

        let mut guard = self.connection.write().await;
        *guard = Some(conn);

        info!("Signal dispatcher connected to Redis: {}", self.redis_url);
        Ok(())
    }

    /// Close the Redis connection.
    pub async fn close(&self) {
        let mut guard = self.connection.write().await;
        *guard = None;
        info!("Signal dispatcher disconnected from Redis");
    }

    /// Check if connected to Redis.
    pub async fn is_connected(&self) -> bool {
        self.connection.read().await.is_some()
    }

    /// Dispatch a signal to the Rust forward service via Redis.
    ///
    /// # Arguments
    /// * `signal` - The signal to dispatch
    ///
    /// # Returns
    /// Number of subscribers that received the message, or error
    pub async fn dispatch_signal(&self, signal: &Signal) -> Result<i64, DispatcherError> {
        let guard = self.connection.read().await;
        let conn = guard.as_ref().ok_or(DispatcherError::NotConnected)?;
        let mut conn = conn.clone();

        // Serialize signal to JSON
        let signal_json = serde_json::to_string(signal)?;

        // Publish to Redis channel
        let subscribers: i64 = conn.publish(SIGNAL_CHANNEL, &signal_json).await?;

        info!(
            "Dispatched signal: {} {} (strength: {:.2}, confidence: {:.2}) to {} subscribers",
            signal.symbol, signal.side, signal.strength, signal.confidence, subscribers
        );

        Ok(subscribers)
    }

    /// Dispatch a signal from raw JSON/dictionary data.
    #[allow(dead_code)]
    pub async fn dispatch_raw(&self, raw: serde_json::Value) -> Result<i64, DispatcherError> {
        let signal = Signal::from_raw(raw)?;
        self.dispatch_signal(&signal).await
    }

    /// Send heartbeat to Rust forward service (Dead Man's Switch).
    ///
    /// # Returns
    /// Number of subscribers that received the heartbeat, or error
    pub async fn send_heartbeat(&self) -> Result<i64, DispatcherError> {
        let guard = self.connection.read().await;
        let conn = match guard.as_ref() {
            Some(c) => c.clone(),
            None => {
                debug!("Signal dispatcher not connected, skipping heartbeat");
                return Ok(0);
            }
        };
        let mut conn = conn;

        let heartbeat = Heartbeat::new(&self.gateway_id);
        let heartbeat_json = serde_json::to_string(&heartbeat)?;

        let subscribers: i64 = conn.publish(HEARTBEAT_CHANNEL, &heartbeat_json).await?;

        debug!(
            "Heartbeat sent to forward service ({} subscribers)",
            subscribers
        );

        Ok(subscribers)
    }

    /// Start a background heartbeat task.
    ///
    /// # Arguments
    /// * `interval_secs` - Heartbeat interval in seconds
    ///
    /// # Returns
    /// A handle to the spawned task
    #[allow(dead_code)]
    pub fn start_heartbeat_task(
        self: Arc<Self>,
        interval_secs: u64,
    ) -> tokio::task::JoinHandle<()> {
        let dispatcher = self;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;

                if let Err(e) = dispatcher.send_heartbeat().await {
                    warn!("Failed to send heartbeat: {}", e);
                }
            }
        })
    }
}

/// Dispatch result for API responses.
#[derive(Debug, Clone, Serialize)]
pub struct DispatchResult {
    pub success: bool,
    pub message: String,
    pub channel: String,
    pub subscribers: i64,
}

impl DispatchResult {
    pub fn success(symbol: &str, subscribers: i64) -> Self {
        Self {
            success: true,
            message: format!("Signal dispatched: {}", symbol),
            channel: SIGNAL_CHANNEL.to_string(),
            subscribers,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            channel: SIGNAL_CHANNEL.to_string(),
            subscribers: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::DateTime;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new("BTCUSD", "Buy")
            .with_strength(0.8)
            .with_confidence(0.9)
            .with_prices(Some(50000.0), Some(49000.0), Some(52000.0));

        assert_eq!(signal.symbol, "BTCUSD");
        assert_eq!(signal.side, "Buy");
        assert_eq!(signal.strength, 0.8);
        assert_eq!(signal.confidence, 0.9);
        assert_eq!(signal.entry_price, Some(50000.0));
        assert_eq!(signal.stop_loss, Some(49000.0));
        assert_eq!(signal.take_profit, Some(52000.0));
    }

    #[test]
    fn test_signal_side_normalization() {
        // Test various side formats
        assert_eq!(Signal::normalize_side("buy"), "Buy");
        assert_eq!(Signal::normalize_side("BUY"), "Buy");
        assert_eq!(Signal::normalize_side("sell"), "Sell");
        assert_eq!(Signal::normalize_side("SELL"), "Sell");
        assert_eq!(Signal::normalize_side("short"), "Sell");
        assert_eq!(Signal::normalize_side("SHORT"), "Sell");
        assert_eq!(Signal::normalize_side("long"), "Buy");
    }

    #[test]
    fn test_signal_serialization() {
        let signal = Signal::new("ETHUSDT", "Sell")
            .with_strength(0.7)
            .with_confidence(0.85);

        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("ETHUSDT"));
        assert!(json.contains("Sell"));
        assert!(json.contains("0.7"));

        // Deserialize back
        let deserialized: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.symbol, "ETHUSDT");
        assert_eq!(deserialized.side, "Sell");
    }

    #[test]
    fn test_signal_from_raw() {
        let raw = serde_json::json!({
            "symbol": "BTCUSD",
            "side": "sell",  // lowercase
            "strength": 0.6,
            "confidence": 0.8,
            "entry_price": 45000.0
        });

        let signal = Signal::from_raw(raw).unwrap();
        assert_eq!(signal.symbol, "BTCUSD");
        assert_eq!(signal.side, "Sell"); // normalized
        assert_eq!(signal.strength, 0.6);
        assert_eq!(signal.entry_price, Some(45000.0));
    }

    #[test]
    fn test_heartbeat_creation() {
        let heartbeat = Heartbeat::new("test-gateway");
        assert_eq!(heartbeat.gateway_id, "test-gateway");
        assert!(!heartbeat.timestamp.is_empty());

        // Verify timestamp is valid RFC 3339
        let parsed: Result<DateTime<Utc>, _> = heartbeat.timestamp.parse();
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_dispatch_result() {
        let success = DispatchResult::success("BTCUSD", 2);
        assert!(success.success);
        assert_eq!(success.subscribers, 2);
        assert!(success.message.contains("BTCUSD"));

        let error = DispatchResult::error("Connection failed");
        assert!(!error.success);
        assert_eq!(error.message, "Connection failed");
    }

    #[test]
    fn test_strength_clamping() {
        let signal = Signal::new("TEST", "Buy")
            .with_strength(1.5) // Over 1.0
            .with_confidence(-0.5); // Below 0.0

        assert_eq!(signal.strength, 1.0);
        assert_eq!(signal.confidence, 0.0);
    }
}
