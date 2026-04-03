//! # Data Service WebSocket Client
//!
//! WebSocket client for connecting to external Rust data service for real-time market data.
//! Supports automatic reconnection, subscription management, and historical data fetching.

use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

/// Data service client configuration
#[derive(Debug, Clone)]
pub struct DataServiceConfig {
    /// WebSocket URL
    pub ws_url: String,

    /// HTTP API URL for historical data
    pub http_url: String,

    /// API key for authentication
    pub api_key: Option<String>,

    /// Reconnection policy
    pub reconnect_policy: ReconnectPolicy,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,
}

impl Default for DataServiceConfig {
    fn default() -> Self {
        Self {
            ws_url: "ws://localhost:8080/stream".to_string(),
            http_url: "http://localhost:8080".to_string(),
            api_key: None,
            reconnect_policy: ReconnectPolicy::default(),
            connection_timeout: Duration::from_secs(10),
            heartbeat_interval: Duration::from_secs(30),
        }
    }
}

/// Reconnection policy
#[derive(Debug, Clone)]
pub struct ReconnectPolicy {
    /// Maximum number of retries (None = infinite)
    pub max_retries: Option<usize>,

    /// Initial delay between retries
    pub initial_delay: Duration,

    /// Maximum delay between retries
    pub max_delay: Duration,

    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        Self {
            max_retries: Some(10),
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
        }
    }
}

/// Market data types from data service
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DataServiceMessage {
    #[serde(rename = "tick")]
    Tick(TickData),

    #[serde(rename = "candle")]
    Candle(CandleData),

    #[serde(rename = "trade")]
    Trade(TradeData),

    #[serde(rename = "orderbook")]
    OrderBook(OrderBookData),

    #[serde(rename = "error")]
    Error(ErrorData),

    #[serde(rename = "subscribed")]
    Subscribed(SubscriptionConfirmation),

    #[serde(rename = "unsubscribed")]
    Unsubscribed(SubscriptionConfirmation),
}

/// Tick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
}

/// Candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleData {
    pub symbol: String,
    pub interval: String,
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub quantity: f64,
    pub side: String,
    pub trade_id: String,
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>,
}

/// Error data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorData {
    pub code: String,
    pub message: String,
}

/// Subscription confirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionConfirmation {
    pub symbols: Vec<String>,
}

/// Message handler trait
#[async_trait::async_trait]
pub trait MessageHandler: Send + Sync {
    async fn on_tick(&self, tick: TickData);
    async fn on_candle(&self, candle: CandleData);
    async fn on_trade(&self, trade: TradeData);
    async fn on_orderbook(&self, book: OrderBookData);
    async fn on_error(&self, error: ErrorData);
    async fn on_reconnect(&self);
}

/// Data service WebSocket client
pub struct DataServiceClient {
    config: DataServiceConfig,
    connection: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    subscriptions: Arc<RwLock<HashSet<String>>>,
    message_handler: Arc<dyn MessageHandler>,
    running: Arc<RwLock<bool>>,
}

impl DataServiceClient {
    /// Create a new data service client
    pub fn new(config: DataServiceConfig, message_handler: Arc<dyn MessageHandler>) -> Self {
        Self {
            config,
            connection: Arc::new(RwLock::new(None)),
            subscriptions: Arc::new(RwLock::new(HashSet::new())),
            message_handler,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Connect to the data service
    pub async fn connect(&self) -> Result<(), DataServiceError> {
        info!("Connecting to data service: {}", self.config.ws_url);

        // Build connection URL with API key if present
        let url = if let Some(api_key) = &self.config.api_key {
            format!("{}?api_key={}", self.config.ws_url, api_key)
        } else {
            self.config.ws_url.clone()
        };

        // Connect with timeout
        let connect_future = connect_async(&url);
        let (ws_stream, _) = tokio::time::timeout(self.config.connection_timeout, connect_future)
            .await
            .map_err(|_| DataServiceError::ConnectionTimeout)?
            .map_err(|e| DataServiceError::ConnectionFailed(e.to_string()))?;

        // Store connection
        let mut conn = self.connection.write().await;
        *conn = Some(ws_stream);

        info!("Successfully connected to data service");
        Ok(())
    }

    /// Start receiving messages
    pub async fn start(&self) -> Result<(), DataServiceError> {
        let mut running = self.running.write().await;
        if *running {
            return Err(DataServiceError::AlreadyRunning);
        }
        *running = true;
        drop(running);

        // Ensure we're connected
        if self.connection.read().await.is_none() {
            self.connect().await?;
        }

        // Spawn message receiver task
        let connection = self.connection.clone();
        let handler = self.message_handler.clone();
        let running = self.running.clone();

        tokio::spawn(async move {
            loop {
                // Check if we should stop
                if !*running.read().await {
                    info!("Data service client stopped");
                    break;
                }

                // Get the websocket stream
                let mut ws_stream = {
                    let mut conn = connection.write().await;
                    match conn.take() {
                        Some(stream) => stream,
                        None => {
                            error!("No active connection");
                            sleep(Duration::from_secs(1)).await;
                            continue;
                        }
                    }
                };

                // Receive messages
                while let Some(msg_result) = ws_stream.next().await {
                    match msg_result {
                        Ok(Message::Text(text)) => {
                            if let Err(e) = Self::handle_message(&text, handler.clone()).await {
                                error!("Error handling message: {}", e);
                            }
                        }
                        Ok(Message::Ping(data)) => {
                            if let Err(e) = ws_stream.send(Message::Pong(data)).await {
                                error!("Error sending pong: {}", e);
                            }
                        }
                        Ok(Message::Close(_)) => {
                            info!("Connection closed by server");
                            break;
                        }
                        Err(e) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                        _ => {}
                    }
                }

                // Put the stream back
                let mut conn = connection.write().await;
                *conn = Some(ws_stream);
                drop(conn);

                // Connection lost, trigger reconnect
                sleep(Duration::from_secs(1)).await;
            }
        });

        Ok(())
    }

    /// Stop receiving messages
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("Stopping data service client");
    }

    /// Subscribe to symbols
    pub async fn subscribe_symbols(&self, symbols: Vec<String>) -> Result<(), DataServiceError> {
        debug!("Subscribing to symbols: {:?}", symbols);

        // Add to subscription list
        {
            let mut subs = self.subscriptions.write().await;
            for symbol in &symbols {
                subs.insert(symbol.clone());
            }
        }

        // Send subscription message
        let subscribe_msg = serde_json::json!({
            "action": "subscribe",
            "symbols": symbols
        });

        self.send_message(&subscribe_msg.to_string()).await?;

        Ok(())
    }

    /// Unsubscribe from symbols
    pub async fn unsubscribe_symbols(&self, symbols: Vec<String>) -> Result<(), DataServiceError> {
        debug!("Unsubscribing from symbols: {:?}", symbols);

        // Remove from subscription list
        {
            let mut subs = self.subscriptions.write().await;
            for symbol in &symbols {
                subs.remove(symbol);
            }
        }

        // Send unsubscription message
        let unsubscribe_msg = serde_json::json!({
            "action": "unsubscribe",
            "symbols": symbols
        });

        self.send_message(&unsubscribe_msg.to_string()).await?;

        Ok(())
    }

    /// Get historical candle data
    pub async fn get_historical_candles(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        interval: &str,
    ) -> Result<Vec<CandleData>, DataServiceError> {
        let url = format!(
            "{}/api/v1/historical/candles?symbol={}&start={}&end={}&interval={}",
            self.config.http_url,
            symbol,
            start.to_rfc3339(),
            end.to_rfc3339(),
            interval
        );

        let client = reqwest::Client::new();
        let mut request = client.get(&url);

        // Add API key if present
        if let Some(api_key) = &self.config.api_key {
            request = request.header("X-API-Key", api_key);
        }

        let response = request
            .send()
            .await
            .map_err(|e| DataServiceError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DataServiceError::HttpError(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let candles = response
            .json()
            .await
            .map_err(|e| DataServiceError::ParseError(e.to_string()))?;

        Ok(candles)
    }

    /// Reconnect to data service
    pub async fn reconnect(&self) -> Result<(), DataServiceError> {
        let policy = &self.config.reconnect_policy;
        let mut retry_count = 0;
        let mut delay = policy.initial_delay;

        loop {
            // Check retry limit
            if let Some(max) = policy.max_retries
                && retry_count >= max
            {
                return Err(DataServiceError::MaxRetriesExceeded);
            }

            retry_count += 1;
            info!("Reconnection attempt {} (delay: {:?})", retry_count, delay);

            // Attempt to connect
            match self.connect().await {
                Ok(_) => {
                    info!("Reconnection successful");

                    // Resubscribe to all symbols
                    let symbols: Vec<String> = {
                        let subs = self.subscriptions.read().await;
                        subs.iter().cloned().collect()
                    };

                    if !symbols.is_empty()
                        && let Err(e) = self.subscribe_symbols(symbols).await
                    {
                        warn!("Failed to resubscribe after reconnect: {}", e);
                    }

                    // Notify handler
                    self.message_handler.on_reconnect().await;

                    return Ok(());
                }
                Err(e) => {
                    warn!("Reconnection attempt {} failed: {}", retry_count, e);
                }
            }

            // Wait before next retry
            sleep(delay).await;

            // Calculate next delay with exponential backoff
            delay = Duration::from_secs_f64(
                (delay.as_secs_f64() * policy.backoff_multiplier)
                    .min(policy.max_delay.as_secs_f64()),
            );
        }
    }

    /// Send a message to the data service
    async fn send_message(&self, message: &str) -> Result<(), DataServiceError> {
        let mut conn = self.connection.write().await;
        if let Some(ws) = conn.as_mut() {
            ws.send(Message::Text(message.to_string().into()))
                .await
                .map_err(|e| DataServiceError::SendError(e.to_string()))?;
            Ok(())
        } else {
            Err(DataServiceError::NotConnected)
        }
    }

    /// Handle incoming message
    async fn handle_message(
        text: &str,
        handler: Arc<dyn MessageHandler>,
    ) -> Result<(), DataServiceError> {
        let message: DataServiceMessage =
            serde_json::from_str(text).map_err(|e| DataServiceError::ParseError(e.to_string()))?;

        match message {
            DataServiceMessage::Tick(tick) => handler.on_tick(tick).await,
            DataServiceMessage::Candle(candle) => handler.on_candle(candle).await,
            DataServiceMessage::Trade(trade) => handler.on_trade(trade).await,
            DataServiceMessage::OrderBook(book) => handler.on_orderbook(book).await,
            DataServiceMessage::Error(error) => handler.on_error(error).await,
            DataServiceMessage::Subscribed(conf) => {
                debug!("Subscribed to: {:?}", conf.symbols);
            }
            DataServiceMessage::Unsubscribed(conf) => {
                debug!("Unsubscribed from: {:?}", conf.symbols);
            }
        }

        Ok(())
    }

    /// Get current subscriptions
    pub async fn get_subscriptions(&self) -> Vec<String> {
        let subs = self.subscriptions.read().await;
        subs.iter().cloned().collect()
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        self.connection.read().await.is_some() && *self.running.read().await
    }
}

/// Data service errors
#[derive(Debug, thiserror::Error)]
pub enum DataServiceError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Connection timeout")]
    ConnectionTimeout,

    #[error("Not connected")]
    NotConnected,

    #[error("Already running")]
    AlreadyRunning,

    #[error("Send error: {0}")]
    SendError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("HTTP error: {0}")]
    HttpError(String),

    #[error("Max retries exceeded")]
    MaxRetriesExceeded,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHandler;

    #[async_trait::async_trait]
    impl MessageHandler for TestHandler {
        async fn on_tick(&self, _tick: TickData) {}
        async fn on_candle(&self, _candle: CandleData) {}
        async fn on_trade(&self, _trade: TradeData) {}
        async fn on_orderbook(&self, _book: OrderBookData) {}
        async fn on_error(&self, _error: ErrorData) {}
        async fn on_reconnect(&self) {}
    }

    #[test]
    fn test_config_default() {
        let config = DataServiceConfig::default();
        assert_eq!(config.ws_url, "ws://localhost:8080/stream");
        assert_eq!(config.http_url, "http://localhost:8080");
    }

    #[test]
    fn test_reconnect_policy_default() {
        let policy = ReconnectPolicy::default();
        assert_eq!(policy.max_retries, Some(10));
        assert_eq!(policy.initial_delay, Duration::from_secs(1));
        assert_eq!(policy.max_delay, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_client_creation() {
        let config = DataServiceConfig::default();
        let handler = Arc::new(TestHandler);
        let client = DataServiceClient::new(config, handler);

        assert!(!client.is_connected().await);
        assert_eq!(client.get_subscriptions().await.len(), 0);
    }
}
