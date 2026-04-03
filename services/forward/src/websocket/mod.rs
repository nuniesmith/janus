//! # WebSocket Module
//!
//! Real-time streaming capabilities for JANUS service including:
//! - Signal updates streaming
//! - Portfolio change notifications
//! - Risk alerts
//! - Performance updates
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::websocket::{WebSocketServer, WebSocketConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = WebSocketConfig::default();
//!     let ws_server = WebSocketServer::new(config);
//!
//!     ws_server.start().await?;
//!
//!     Ok(())
//! }
//! ```

pub mod broadcaster;
pub mod client;
pub mod data_client;
pub mod heartbeat;
pub mod message;
pub mod server;

pub use broadcaster::{BroadcasterError, SignalBroadcaster};
pub use client::{ClientConnection, ClientError, ClientStats};
pub use data_client::{
    CandleData, DataServiceClient, DataServiceConfig, DataServiceError, DataServiceMessage,
    ErrorData, MessageHandler, OrderBookData, ReconnectPolicy, SubscriptionConfirmation, TickData,
    TradeData,
};
pub use heartbeat::{HeartbeatConfig, HeartbeatError, HeartbeatManager, HeartbeatStats};
pub use message::{
    AlertSeverity, ErrorMessage, GoodbyeMessage, MarketDataType, MarketDataUpdate,
    PerformanceUpdate, PortfolioUpdate, PositionSnapshot, RiskAlert, RiskAlertType, SignalType,
    SignalUpdate, SubscribeRequest, UnsubscribeRequest, WebSocketMessage, WelcomeMessage,
};
pub use server::{WebSocketConfig, WebSocketServer};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// WebSocket client manager
pub struct ClientManager {
    clients: Arc<RwLock<HashMap<Uuid, ClientConnection>>>,
}

impl ClientManager {
    /// Create a new client manager
    pub fn new() -> Self {
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a new client
    pub async fn add_client(&self, client_id: Uuid, client: ClientConnection) {
        let mut clients = self.clients.write().await;
        clients.insert(client_id, client);
    }

    /// Remove a client
    pub async fn remove_client(&self, client_id: &Uuid) {
        let mut clients = self.clients.write().await;
        clients.remove(client_id);
    }

    /// Get client count
    pub async fn client_count(&self) -> usize {
        let clients = self.clients.read().await;
        clients.len()
    }

    /// Get all connected client IDs (for graceful shutdown enumeration)
    pub async fn client_ids(&self) -> Vec<Uuid> {
        let clients = self.clients.read().await;
        clients.keys().copied().collect()
    }

    /// Broadcast message to all clients
    pub async fn broadcast(&self, message: WebSocketMessage) {
        let clients = self.clients.read().await;
        for client in clients.values() {
            let _ = client.send_message(message.clone()).await;
        }
    }

    /// Send message to specific client
    pub async fn send_to_client(
        &self,
        client_id: &Uuid,
        message: WebSocketMessage,
    ) -> Result<(), String> {
        let clients = self.clients.read().await;
        if let Some(client) = clients.get(client_id) {
            client
                .send_message(message)
                .await
                .map_err(|e| e.to_string())
        } else {
            Err("Client not found".to_string())
        }
    }

    /// Send message to clients subscribed to a symbol
    pub async fn send_to_symbol_subscribers(&self, symbol: &str, message: WebSocketMessage) {
        let clients = self.clients.read().await;
        for client in clients.values() {
            if client.is_subscribed_to_symbol(symbol) {
                let _ = client.send_message(message.clone()).await;
            }
        }
    }

    /// Get all clients
    pub async fn get_all_clients(&self) -> HashMap<Uuid, ClientConnection> {
        let clients = self.clients.read().await;
        clients.clone()
    }

    /// Get stale clients
    pub async fn get_stale_clients(&self, timeout: std::time::Duration) -> Vec<Uuid> {
        let clients = self.clients.read().await;
        let mut stale = Vec::new();

        for (id, client) in clients.iter() {
            if client.is_stale(timeout).await {
                stale.push(*id);
            }
        }

        stale
    }

    /// Get client by ID
    pub async fn get_client(&self, client_id: &Uuid) -> Option<ClientConnection> {
        let clients = self.clients.read().await;
        clients.get(client_id).cloned()
    }
}

impl Default for ClientManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Subscription filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionFilter {
    /// Symbols to subscribe to (None = all)
    pub symbols: Option<Vec<String>>,

    /// Minimum signal confidence
    pub min_confidence: Option<f64>,

    /// Signal types to include
    pub signal_types: Option<Vec<String>>,

    /// Subscribe to portfolio updates
    pub portfolio_updates: bool,

    /// Subscribe to risk alerts
    pub risk_alerts: bool,
}

impl Default for SubscriptionFilter {
    fn default() -> Self {
        Self {
            symbols: None,
            min_confidence: None,
            signal_types: None,
            portfolio_updates: true,
            risk_alerts: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_manager_creation() {
        let manager = ClientManager::new();
        assert_eq!(manager.client_count().await, 0);
    }

    #[test]
    fn test_subscription_filter_default() {
        let filter = SubscriptionFilter::default();
        assert!(filter.portfolio_updates);
        assert!(filter.risk_alerts);
        assert!(filter.symbols.is_none());
    }
}
