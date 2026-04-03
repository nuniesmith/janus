//! # WebSocket Client Connection
//!
//! Manages individual WebSocket client connections with subscription filtering,
//! heartbeat monitoring, and message routing.

use crate::websocket::{SubscriptionFilter, WebSocketMessage};
use axum::extract::ws::Message;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;

/// Individual client connection
#[derive(Clone)]
pub struct ClientConnection {
    /// Unique client identifier
    id: Uuid,

    /// Message sender channel
    sender: mpsc::UnboundedSender<Message>,

    /// Subscription filter
    subscription_filter: Arc<RwLock<SubscriptionFilter>>,

    /// Connection timestamp
    connected_at: DateTime<Utc>,

    /// Last activity timestamp
    last_activity: Arc<RwLock<DateTime<Utc>>>,

    /// Client metadata (user_id, session_id, etc.)
    metadata: Arc<RwLock<HashMap<String, String>>>,
}

impl ClientConnection {
    /// Create a new client connection
    pub fn new(sender: mpsc::UnboundedSender<Message>) -> Self {
        Self {
            id: Uuid::new_v4(),
            sender,
            subscription_filter: Arc::new(RwLock::new(SubscriptionFilter::default())),
            connected_at: Utc::now(),
            last_activity: Arc::new(RwLock::new(Utc::now())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new client connection with ID
    pub fn with_id(id: Uuid, sender: mpsc::UnboundedSender<Message>) -> Self {
        Self {
            id,
            sender,
            subscription_filter: Arc::new(RwLock::new(SubscriptionFilter::default())),
            connected_at: Utc::now(),
            last_activity: Arc::new(RwLock::new(Utc::now())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get client ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get connection timestamp
    pub fn connected_at(&self) -> DateTime<Utc> {
        self.connected_at
    }

    /// Send a message to the client
    pub async fn send_message(&self, message: WebSocketMessage) -> Result<(), ClientError> {
        // Update activity
        self.update_activity().await;

        // Serialize message to JSON
        let json = serde_json::to_string(&message)
            .map_err(|e| ClientError::SerializationError(e.to_string()))?;

        // Send as text message
        self.sender
            .send(Message::Text(json.into()))
            .map_err(|e| ClientError::SendError(e.to_string()))
    }

    /// Send raw text message
    pub async fn send_text(&self, text: String) -> Result<(), ClientError> {
        self.update_activity().await;
        self.sender
            .send(Message::Text(text.into()))
            .map_err(|e| ClientError::SendError(e.to_string()))
    }

    /// Send ping
    pub async fn send_ping(&self) -> Result<(), ClientError> {
        self.sender
            .send(Message::Ping(vec![].into()))
            .map_err(|e| ClientError::SendError(e.to_string()))
    }

    /// Send pong
    pub async fn send_pong(&self) -> Result<(), ClientError> {
        self.sender
            .send(Message::Pong(vec![].into()))
            .map_err(|e| ClientError::SendError(e.to_string()))
    }

    /// Check if client is subscribed to a symbol
    pub fn is_subscribed_to_symbol(&self, symbol: &str) -> bool {
        // This is a synchronous check, so we'll use try_read
        // In production, you might want to use a different approach
        match self.subscription_filter.try_read() {
            Ok(filter) => {
                if let Some(symbols) = &filter.symbols {
                    symbols.iter().any(|s| s == symbol)
                } else {
                    // None means subscribed to all symbols
                    true
                }
            }
            _ => {
                // If we can't acquire the lock, assume not subscribed
                false
            }
        }
    }

    /// Check if message matches subscription filter
    pub async fn matches_filter(&self, message: &WebSocketMessage) -> bool {
        let filter = self.subscription_filter.read().await;

        match message {
            WebSocketMessage::SignalUpdate(signal) => {
                // Check symbol filter
                if let Some(symbols) = &filter.symbols
                    && !symbols.iter().any(|s| s == &signal.symbol)
                {
                    return false;
                }

                // Check confidence filter
                if let Some(min_confidence) = filter.min_confidence
                    && signal.confidence < min_confidence
                {
                    return false;
                }

                // Check signal type filter
                if let Some(signal_types) = &filter.signal_types {
                    let signal_type_str = format!("{:?}", signal.signal_type);
                    if !signal_types.iter().any(|t| t == &signal_type_str) {
                        return false;
                    }
                }

                true
            }
            WebSocketMessage::PortfolioUpdate(_) => filter.portfolio_updates,
            WebSocketMessage::RiskAlert(_) => filter.risk_alerts,
            WebSocketMessage::PerformanceUpdate(_) => filter.portfolio_updates,
            WebSocketMessage::MarketData(data) => {
                // Check symbol filter for market data
                if let Some(symbols) = &filter.symbols {
                    symbols.iter().any(|s| s == &data.symbol)
                } else {
                    true
                }
            }
            // System messages always pass through
            _ => true,
        }
    }

    /// Update subscription filter
    pub async fn update_subscription(&self, filter: SubscriptionFilter) {
        let mut current_filter = self.subscription_filter.write().await;
        *current_filter = filter;
        self.update_activity().await;
    }

    /// Get current subscription filter
    pub async fn get_subscription(&self) -> SubscriptionFilter {
        self.subscription_filter.read().await.clone()
    }

    /// Update last activity timestamp
    pub async fn update_activity(&self) {
        let mut activity = self.last_activity.write().await;
        *activity = Utc::now();
    }

    /// Get last activity timestamp
    pub async fn last_activity(&self) -> DateTime<Utc> {
        *self.last_activity.read().await
    }

    /// Check if client connection is stale (no activity for timeout duration)
    pub async fn is_stale(&self, timeout: Duration) -> bool {
        let last = self.last_activity().await;
        let now = Utc::now();
        let elapsed = now.signed_duration_since(last);

        elapsed.num_milliseconds() > timeout.as_millis() as i64
    }

    /// Set client metadata
    pub async fn set_metadata(&self, key: String, value: String) {
        let mut metadata = self.metadata.write().await;
        metadata.insert(key, value);
    }

    /// Get client metadata
    pub async fn get_metadata(&self, key: &str) -> Option<String> {
        let metadata = self.metadata.read().await;
        metadata.get(key).cloned()
    }

    /// Get all metadata
    pub async fn get_all_metadata(&self) -> HashMap<String, String> {
        self.metadata.read().await.clone()
    }

    /// Get connection duration
    pub fn connection_duration(&self) -> chrono::Duration {
        Utc::now().signed_duration_since(self.connected_at)
    }

    /// Close the connection
    pub async fn close(&self) -> Result<(), ClientError> {
        self.sender
            .send(Message::Close(None))
            .map_err(|e| ClientError::SendError(e.to_string()))
    }
}

/// Client connection errors
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Send error: {0}")]
    SendError(String),

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Invalid message: {0}")]
    InvalidMessage(String),
}

/// Client statistics
#[derive(Debug, Clone)]
pub struct ClientStats {
    pub id: Uuid,
    pub connected_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub connection_duration_seconds: i64,
    pub subscribed_symbols: Option<Vec<String>>,
    pub metadata: HashMap<String, String>,
}

impl ClientConnection {
    /// Get client statistics
    pub async fn get_stats(&self) -> ClientStats {
        let last_activity = self.last_activity().await;
        let filter = self.subscription_filter.read().await;
        let metadata = self.metadata.read().await;

        ClientStats {
            id: self.id,
            connected_at: self.connected_at,
            last_activity,
            connection_duration_seconds: self.connection_duration().num_seconds(),
            subscribed_symbols: filter.symbols.clone(),
            metadata: metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    fn create_test_client() -> (ClientConnection, mpsc::UnboundedReceiver<Message>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let client = ClientConnection::new(tx);
        (client, rx)
    }

    #[tokio::test]
    async fn test_client_creation() {
        let (client, _rx) = create_test_client();
        assert!(client.id() != Uuid::nil());
    }

    #[tokio::test]
    async fn test_send_ping() {
        let (client, mut rx) = create_test_client();
        client.send_ping().await.unwrap();

        match rx.recv().await {
            Some(msg) => {
                assert!(matches!(msg, Message::Ping(_)));
            }
            _ => {
                panic!("Expected ping message");
            }
        }
    }

    #[tokio::test]
    async fn test_subscription_filter_default() {
        let (client, _rx) = create_test_client();
        let filter = client.get_subscription().await;

        assert!(filter.symbols.is_none());
        assert!(filter.portfolio_updates);
        assert!(filter.risk_alerts);
    }

    #[tokio::test]
    async fn test_subscription_update() {
        let (client, _rx) = create_test_client();

        let filter = SubscriptionFilter {
            symbols: Some(vec!["BTCUSD".to_string(), "ETHUSDT".to_string()]),
            min_confidence: Some(0.7),
            ..Default::default()
        };

        client.update_subscription(filter.clone()).await;

        let updated = client.get_subscription().await;
        assert_eq!(updated.symbols, filter.symbols);
        assert_eq!(updated.min_confidence, filter.min_confidence);
    }

    #[tokio::test]
    async fn test_is_subscribed_to_symbol() {
        let (client, _rx) = create_test_client();

        // Default: subscribed to all
        assert!(client.is_subscribed_to_symbol("BTCUSD"));

        // Set specific symbols
        let filter = SubscriptionFilter {
            symbols: Some(vec!["BTCUSD".to_string()]),
            ..Default::default()
        };
        client.update_subscription(filter).await;

        // Give it a moment for the lock
        sleep(Duration::from_millis(10)).await;

        assert!(client.is_subscribed_to_symbol("BTCUSD"));
        assert!(!client.is_subscribed_to_symbol("ETHUSDT"));
    }

    #[tokio::test]
    async fn test_metadata() {
        let (client, _rx) = create_test_client();

        client
            .set_metadata("user_id".to_string(), "user123".to_string())
            .await;
        client
            .set_metadata("session_id".to_string(), "session456".to_string())
            .await;

        assert_eq!(
            client.get_metadata("user_id").await,
            Some("user123".to_string())
        );
        assert_eq!(
            client.get_metadata("session_id").await,
            Some("session456".to_string())
        );
        assert_eq!(client.get_metadata("nonexistent").await, None);

        let all_metadata = client.get_all_metadata().await;
        assert_eq!(all_metadata.len(), 2);
    }

    #[tokio::test]
    async fn test_activity_tracking() {
        let (client, _rx) = create_test_client();

        let initial_activity = client.last_activity().await;
        sleep(Duration::from_millis(100)).await;

        client.update_activity().await;
        let updated_activity = client.last_activity().await;

        assert!(updated_activity > initial_activity);
    }

    #[tokio::test]
    async fn test_stale_detection() {
        let (client, _rx) = create_test_client();

        // Should not be stale immediately
        assert!(!client.is_stale(Duration::from_secs(1)).await);

        // Simulate stale connection (activity in the past)
        sleep(Duration::from_millis(100)).await;
        assert!(client.is_stale(Duration::from_millis(50)).await);
    }

    #[tokio::test]
    async fn test_connection_duration() {
        let (client, _rx) = create_test_client();

        sleep(Duration::from_millis(100)).await;
        let duration = client.connection_duration();

        assert!(duration.num_milliseconds() >= 100);
    }

    #[tokio::test]
    async fn test_client_stats() {
        let (client, _rx) = create_test_client();

        client
            .set_metadata("user_id".to_string(), "test_user".to_string())
            .await;

        let stats = client.get_stats().await;

        assert_eq!(stats.id, client.id());
        assert_eq!(stats.connected_at, client.connected_at());
        assert!(stats.metadata.contains_key("user_id"));
    }
}
