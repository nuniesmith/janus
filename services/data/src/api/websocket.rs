// WebSocket streaming handler
//
// Provides real-time streaming of market data to connected clients
// Supports subscription filtering by symbol, exchange, and message type

use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tokio::sync::broadcast;

use crate::actors::router::NormalizedMessage;
use crate::api::AppState;
use crate::api::metrics::{ACTIVE_SUBSCRIPTIONS, WEBSOCKET_CLIENTS};

/// WebSocket subscription request from client
#[derive(Debug, Deserialize)]
struct SubscribeRequest {
    /// Symbols to subscribe to (e.g., ["BTC-USDT", "ETH-USDT"])
    #[serde(default)]
    symbols: Vec<String>,

    /// Exchanges to filter by (empty = all)
    #[serde(default)]
    exchanges: Vec<String>,

    /// Message types to receive ("trade", "metric", "umap")
    #[serde(default = "default_message_types")]
    types: Vec<String>,
}

fn default_message_types() -> Vec<String> {
    vec!["trade".to_string(), "metric".to_string()]
}

/// WebSocket subscription response
#[derive(Debug, Serialize)]
struct SubscribeResponse {
    success: bool,
    message: String,
    subscribed_count: usize,
}

/// Client session state
struct ClientSession {
    /// WebSocket sender
    sender: futures_util::stream::SplitSink<WebSocket, Message>,

    /// Broadcast receiver for messages from Router
    broadcast_rx: broadcast::Receiver<NormalizedMessage>,

    /// Subscribed symbols (empty = all symbols)
    symbols: HashSet<String>,

    /// Subscribed exchanges (empty = all exchanges)
    exchanges: HashSet<String>,

    /// Subscribed message types
    types: HashSet<String>,

    /// Client ID for logging
    client_id: String,
}

/// WebSocket upgrade handler
pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle WebSocket connection
async fn handle_socket(socket: WebSocket, state: AppState) {
    let client_id = uuid::Uuid::new_v4().to_string();
    tracing::info!("WebSocket client connected: {}", client_id);

    // Increment connected clients metric
    WEBSOCKET_CLIENTS.inc();

    let (sender, mut receiver) = socket.split();

    // Subscribe to broadcast channel
    let broadcast_rx = state.broadcast_tx.subscribe();

    let mut session = ClientSession {
        sender,
        broadcast_rx,
        symbols: HashSet::new(),
        exchanges: HashSet::new(),
        types: default_message_types().into_iter().collect(),
        client_id: client_id.clone(),
    };

    // Main event loop
    loop {
        tokio::select! {
            // Handle incoming messages from client
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(e) = handle_client_message(&mut session, &text).await {
                            tracing::error!("Error handling client message: {}", e);
                        }
                    }
                    Some(Ok(Message::Binary(_))) => {
                        tracing::warn!("Received unexpected binary message from client");
                    }
                    Some(Ok(Message::Close(_))) => {
                        tracing::info!("Client {} closed connection", client_id);
                        break;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        if let Err(e) = session.sender.send(Message::Pong(data)).await {
                            tracing::error!("Error sending pong: {}", e);
                            break;
                        }
                    }
                    Some(Ok(Message::Pong(_))) => {
                        // Pong received, connection is alive
                    }
                    Some(Err(e)) => {
                        tracing::error!("WebSocket error: {}", e);
                        break;
                    }
                    None => {
                        tracing::info!("Client {} disconnected", client_id);
                        break;
                    }
                }
            }

            // Handle broadcast messages from Router
            msg = session.broadcast_rx.recv() => {
                match msg {
                    Ok(normalized_msg) => {
                        if session.should_send(&normalized_msg)
                            && let Err(e) = send_to_client(&mut session, &normalized_msg).await {
                                tracing::error!("Error sending to client: {}", e);
                                break;
                            }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!("Client {} lagged, skipped {} messages", client_id, skipped);
                        // Send warning to client
                        let warning = serde_json::json!({
                            "type": "warning",
                            "message": format!("Client lagging, skipped {} messages", skipped)
                        });
                        if let Ok(text) = serde_json::to_string(&warning) {
                            let _ = session.sender.send(Message::Text(text.into())).await;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::info!("Broadcast channel closed");
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
    WEBSOCKET_CLIENTS.dec();
    ACTIVE_SUBSCRIPTIONS.sub(session.symbols.len() as i64);
    tracing::info!("WebSocket client disconnected: {}", client_id);
}

/// Handle incoming message from client (subscription requests)
async fn handle_client_message(
    session: &mut ClientSession,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let request: SubscribeRequest = serde_json::from_str(text)?;

    // Update subscriptions
    let old_count = session.symbols.len();

    session.symbols = request.symbols.into_iter().collect();
    session.exchanges = request.exchanges.into_iter().collect();
    session.types = request.types.into_iter().collect();

    let new_count = session.symbols.len();

    // Update metrics
    ACTIVE_SUBSCRIPTIONS.add((new_count as i64) - (old_count as i64));

    // Send acknowledgment
    let response = SubscribeResponse {
        success: true,
        message: format!("Subscribed to {} symbols", new_count),
        subscribed_count: new_count,
    };

    let response_text = serde_json::to_string(&response)?;
    session
        .sender
        .send(Message::Text(response_text.into()))
        .await?;

    tracing::info!(
        "Client {} subscribed to {} symbols",
        session.client_id,
        new_count
    );

    Ok(())
}

/// Send normalized message to client as JSON
async fn send_to_client(
    session: &mut ClientSession,
    msg: &NormalizedMessage,
) -> Result<(), Box<dyn std::error::Error>> {
    // Convert to JSON
    let json = serde_json::json!({
        "type": "trade",
        "timestamp": msg.timestamp,
        "symbol": msg.symbol,
        "exchange": msg.exchange,
        "side": msg.side,
        "price": msg.price,
        "amount": msg.amount,
        "trade_id": msg.trade_id,
    });

    let text = serde_json::to_string(&json)?;
    session.sender.send(Message::Text(text.into())).await?;

    Ok(())
}

impl ClientSession {
    /// Check if message should be sent to this client based on subscriptions
    fn should_send(&self, msg: &NormalizedMessage) -> bool {
        // Check symbol filter (empty = all symbols)
        let symbol_match = self.symbols.is_empty() || self.symbols.contains(&msg.symbol);

        // Check exchange filter (empty = all exchanges)
        let exchange_match = self.exchanges.is_empty() || self.exchanges.contains(&msg.exchange);

        // Check type filter
        let type_match = self.types.contains("trade");

        symbol_match && exchange_match && type_match
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscribe_request_deserialization() {
        let json = r#"{
            "symbols": ["BTC-USDT", "ETH-USDT"],
            "exchanges": ["binance"],
            "types": ["trade"]
        }"#;

        let request: SubscribeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.symbols.len(), 2);
        assert_eq!(request.exchanges.len(), 1);
        assert_eq!(request.types.len(), 1);
    }

    #[test]
    fn test_subscribe_response_serialization() {
        let response = SubscribeResponse {
            success: true,
            message: "Subscribed to 2 symbols".to_string(),
            subscribed_count: 2,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("true"));
        assert!(json.contains("Subscribed to 2 symbols"));
    }

    #[test]
    fn test_default_message_types() {
        let types = default_message_types();
        assert_eq!(types.len(), 2);
        assert!(types.contains(&"trade".to_string()));
        assert!(types.contains(&"metric".to_string()));
    }
}
