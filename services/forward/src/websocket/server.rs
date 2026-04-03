//! # WebSocket Server
//!
//! Production-grade WebSocket server with Axum integration.
//! Handles client connections, message routing, and graceful shutdown.

use crate::websocket::{
    ClientConnection, ClientManager, HeartbeatManager, SignalBroadcaster, WebSocketMessage,
    WelcomeMessage,
};
use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::IntoResponse,
    routing::get,
};
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// WebSocket server configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// Bind address
    pub bind_address: String,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// Client timeout duration
    pub client_timeout: Duration,

    /// Maximum message size in bytes
    pub max_message_size: usize,

    /// Enable compression
    pub enable_compression: bool,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:8081".to_string(),
            max_connections: 10000,
            heartbeat_interval: Duration::from_secs(30),
            client_timeout: Duration::from_secs(90),
            max_message_size: 1024 * 1024, // 1MB
            enable_compression: true,
        }
    }
}

/// WebSocket server state
#[derive(Clone)]
pub struct WebSocketState {
    client_manager: Arc<ClientManager>,
    #[allow(dead_code)]
    broadcaster: Arc<SignalBroadcaster>,
    config: WebSocketConfig,
    total_connections: Arc<AtomicU64>,
}

impl WebSocketState {
    /// Create new WebSocket state
    pub fn new(
        client_manager: Arc<ClientManager>,
        broadcaster: Arc<SignalBroadcaster>,
        config: WebSocketConfig,
        total_connections: Arc<AtomicU64>,
    ) -> Self {
        Self {
            client_manager,
            broadcaster,
            config,
            total_connections,
        }
    }
}

/// WebSocket server
pub struct WebSocketServer {
    config: WebSocketConfig,
    client_manager: Arc<ClientManager>,
    broadcaster: Arc<SignalBroadcaster>,
    heartbeat_manager: Arc<HeartbeatManager>,
    running: Arc<RwLock<bool>>,
    total_connections: Arc<AtomicU64>,
}

impl WebSocketServer {
    /// Create a new WebSocket server
    pub fn new(
        config: WebSocketConfig,
        client_manager: Arc<ClientManager>,
        broadcaster: Arc<SignalBroadcaster>,
        heartbeat_manager: Arc<HeartbeatManager>,
    ) -> Self {
        Self {
            config,
            client_manager,
            broadcaster,
            heartbeat_manager,
            running: Arc::new(RwLock::new(false)),
            total_connections: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start the WebSocket server
    pub async fn start(&self) -> Result<(), WebSocketError> {
        let mut running = self.running.write().await;
        if *running {
            return Err(WebSocketError::AlreadyRunning);
        }
        *running = true;
        drop(running);

        info!("Starting WebSocket server on {}", self.config.bind_address);

        // Start heartbeat manager
        self.heartbeat_manager.start().await?;

        // Create Axum app
        let state = WebSocketState::new(
            self.client_manager.clone(),
            self.broadcaster.clone(),
            self.config.clone(),
            self.total_connections.clone(),
        );

        let app = Router::new()
            .route("/ws", get(ws_handler))
            .with_state(state);

        // Parse bind address
        let addr: SocketAddr = self
            .config
            .bind_address
            .parse()
            .map_err(|e| WebSocketError::ConfigError(format!("Invalid bind address: {}", e)))?;

        // Start server
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| WebSocketError::BindError(e.to_string()))?;

        info!("✅ WebSocket server listening on {}", addr);

        axum::serve(listener, app)
            .await
            .map_err(|e| WebSocketError::ServerError(e.to_string()))?;

        Ok(())
    }

    /// Stop the WebSocket server
    pub async fn stop(&self) -> Result<(), WebSocketError> {
        let mut running = self.running.write().await;
        *running = false;

        info!("Stopping WebSocket server");

        // Stop heartbeat manager
        self.heartbeat_manager.stop().await;

        // Disconnect all connected clients gracefully
        let client_ids = self.client_manager.client_ids().await;
        let count = client_ids.len();
        for id in client_ids {
            self.client_manager.remove_client(&id).await;
        }
        if count > 0 {
            info!("Disconnected {} client(s) during shutdown", count);
        }

        Ok(())
    }

    /// Check if server is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Get server statistics
    pub async fn get_stats(&self) -> ServerStats {
        ServerStats {
            active_connections: self.client_manager.client_count().await,
            total_connections: self.total_connections.load(Ordering::Relaxed),
            running: self.is_running().await,
        }
    }
}

/// Server statistics
#[derive(Debug, Clone)]
pub struct ServerStats {
    pub active_connections: usize,
    pub total_connections: u64,
    pub running: bool,
}

/// WebSocket handler for Axum
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<WebSocketState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle a WebSocket connection
async fn handle_socket(socket: WebSocket, state: WebSocketState) {
    let client_id = Uuid::new_v4();
    info!("New WebSocket connection: {}", client_id);

    // Increment lifetime connection counter
    state.total_connections.fetch_add(1, Ordering::Relaxed);

    // Check connection limit
    let current_connections = state.client_manager.client_count().await;
    if current_connections >= state.config.max_connections {
        warn!(
            "Connection limit reached ({}/{}), rejecting client {}",
            current_connections, state.config.max_connections, client_id
        );
        // Send error and close
        let _ = send_error_and_close(socket, "Server at capacity").await;
        return;
    }

    // Split socket into sender and receiver
    let (mut sender, mut receiver) = socket.split();

    // Create channel for sending messages to this client
    let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

    // Create client connection
    let client = ClientConnection::with_id(client_id, tx);

    // Add client to manager
    state
        .client_manager
        .add_client(client_id, client.clone())
        .await;

    // Send welcome message
    let welcome = WebSocketMessage::Welcome(WelcomeMessage::default());
    if let Err(e) = client.send_message(welcome).await {
        error!("Failed to send welcome message to {}: {}", client_id, e);
        state.client_manager.remove_client(&client_id).await;
        return;
    }

    // Spawn task to send messages from the channel to the WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if sender.send(message).await.is_err() {
                break;
            }
        }
    });

    // Spawn task to receive messages from the WebSocket
    let client_manager = state.client_manager.clone();
    let client_clone = client.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    debug!("Received text message from {}: {}", client_id, text);
                    if let Err(e) = handle_client_message(&client_clone, &text).await {
                        error!("Error handling message from {}: {}", client_id, e);
                    }
                }
                Message::Binary(_) => {
                    debug!("Received binary message from {} (ignoring)", client_id);
                }
                Message::Ping(_data) => {
                    debug!("Received ping from {}", client_id);
                    if let Err(e) = client_clone.send_pong().await {
                        error!("Failed to send pong to {}: {}", client_id, e);
                        break;
                    }
                }
                Message::Pong(_) => {
                    debug!("Received pong from {}", client_id);
                    client_clone.update_activity().await;
                }
                Message::Close(_) => {
                    info!("Client {} closed connection", client_id);
                    break;
                }
            }
        }

        // Remove client when connection closes
        client_manager.remove_client(&client_id).await;
        info!("Client {} disconnected", client_id);
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {
            debug!("Send task completed for {}", client_id);
        }
        _ = recv_task => {
            debug!("Receive task completed for {}", client_id);
        }
    }

    // Cleanup
    state.client_manager.remove_client(&client_id).await;
    info!("Connection cleanup completed for {}", client_id);
}

/// Handle a client message
async fn handle_client_message(
    client: &ClientConnection,
    text: &str,
) -> Result<(), WebSocketError> {
    // Parse message
    let message: WebSocketMessage =
        serde_json::from_str(text).map_err(|e| WebSocketError::ParseError(e.to_string()))?;

    match message {
        WebSocketMessage::Subscribe(request) => {
            debug!("Client {} subscribing: {:?}", client.id(), request);

            // Update client subscription filter
            let filter = crate::websocket::SubscriptionFilter {
                symbols: request.symbols,
                min_confidence: request.min_confidence,
                signal_types: request.signal_types,
                portfolio_updates: request.portfolio_updates,
                risk_alerts: request.risk_alerts,
            };

            client.update_subscription(filter).await;

            // Send acknowledgment
            let ack = WebSocketMessage::Welcome(WelcomeMessage {
                session_id: client.id(),
                server_version: "1.0.0".to_string(),
                timestamp: chrono::Utc::now(),
                capabilities: vec!["signals".to_string(), "risk_alerts".to_string()],
            });
            client.send_message(ack).await?;
        }
        WebSocketMessage::Unsubscribe(request) => {
            debug!("Client {} unsubscribing: {:?}", client.id(), request);

            // Update subscription filter to remove symbols
            let mut current_filter = client.get_subscription().await;
            if let Some(unsub_symbols) = request.symbols
                && let Some(current_symbols) = &mut current_filter.symbols
            {
                current_symbols.retain(|s| !unsub_symbols.contains(s));
            }
            client.update_subscription(current_filter).await;
        }
        WebSocketMessage::Ping => {
            debug!("Client {} sent ping", client.id());
            client.send_message(WebSocketMessage::Pong).await?;
        }
        _ => {
            warn!("Unexpected message type from client {}", client.id());
        }
    }

    Ok(())
}

/// Send error message and close connection
async fn send_error_and_close(
    mut socket: WebSocket,
    error_msg: &str,
) -> Result<(), WebSocketError> {
    let error = WebSocketMessage::Error(crate::websocket::ErrorMessage {
        code: "ERROR".to_string(),
        message: error_msg.to_string(),
        timestamp: chrono::Utc::now(),
    });

    let text = serde_json::to_string(&error)
        .map_err(|e| WebSocketError::SerializationError(e.to_string()))?;

    socket
        .send(Message::Text(text.into()))
        .await
        .map_err(|e| WebSocketError::SendError(e.to_string()))?;

    socket
        .close()
        .await
        .map_err(|e| WebSocketError::CloseError(e.to_string()))?;

    Ok(())
}

/// WebSocket server errors
#[derive(Debug, thiserror::Error)]
pub enum WebSocketError {
    #[error("Server is already running")]
    AlreadyRunning,

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Bind error: {0}")]
    BindError(String),

    #[error("Server error: {0}")]
    ServerError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Send error: {0}")]
    SendError(String),

    #[error("Close error: {0}")]
    CloseError(String),

    #[error("Heartbeat error: {0}")]
    HeartbeatError(#[from] crate::websocket::HeartbeatError),

    #[error("Client error: {0}")]
    ClientError(#[from] crate::websocket::ClientError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::JanusMetrics;
    use crate::websocket::HeartbeatConfig;

    #[test]
    fn test_config_default() {
        let config = WebSocketConfig::default();
        assert_eq!(config.bind_address, "0.0.0.0:8081");
        assert_eq!(config.max_connections, 10000);
        assert_eq!(config.heartbeat_interval, Duration::from_secs(30));
        assert_eq!(config.client_timeout, Duration::from_secs(90));
    }

    #[tokio::test]
    async fn test_server_creation() {
        let config = WebSocketConfig::default();
        let client_manager = Arc::new(ClientManager::new());
        let metrics = Arc::new(JanusMetrics::new().unwrap());
        let broadcaster = Arc::new(SignalBroadcaster::new(
            client_manager.clone(),
            metrics.clone(),
        ));
        let heartbeat_config = HeartbeatConfig::default();
        let heartbeat = Arc::new(HeartbeatManager::new(
            client_manager.clone(),
            heartbeat_config,
        ));

        let server = WebSocketServer::new(config, client_manager.clone(), broadcaster, heartbeat);
        assert!(!server.is_running().await);
    }

    #[tokio::test]
    async fn test_server_stats() {
        let config = WebSocketConfig::default();
        let client_manager = Arc::new(ClientManager::new());
        let metrics = Arc::new(JanusMetrics::new().unwrap());
        let broadcaster = Arc::new(SignalBroadcaster::new(
            client_manager.clone(),
            metrics.clone(),
        ));
        let heartbeat_config = HeartbeatConfig::default();
        let heartbeat = Arc::new(HeartbeatManager::new(
            client_manager.clone(),
            heartbeat_config,
        ));

        let server = WebSocketServer::new(config, client_manager.clone(), broadcaster, heartbeat);
        let stats = server.get_stats().await;

        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_connections, 0);
        assert!(!stats.running);
    }
}
