//! WebSocket Signal Streaming Handler
//!
//! Provides real-time streaming of trading signals to connected clients.
//! Supports subscription filtering by symbol, timeframe, signal_type, and direction.
//!
//! ## Protocol
//!
//! Client → Server (Subscribe):
//! ```json
//! {
//!   "symbols": ["BTCUSD", "ETHUSDT"],
//!   "timeframes": ["1m", "5m"],
//!   "signal_types": ["ema_golden_cross", "bullish_confluence"],
//!   "directions": ["bullish"]
//! }
//! ```
//!
//! Server → Client (Signal):
//! ```json
//! {
//!   "type": "signal",
//!   "symbol": "BTCUSD",
//!   "exchange": "binance",
//!   "timeframe": "1m",
//!   "signal_type": "ema_golden_cross",
//!   "direction": "bullish",
//!   "strength": 3,
//!   "timestamp": 1737384600000,
//!   "price": 42350.50,
//!   "trigger_value": 42300.25,
//!   "trigger_value_2": 42280.75,
//!   "description": "EMA-8 crossed above EMA-21"
//! }
//! ```

use anyhow::Result;
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
use tracing::{debug, error, info, warn};

use crate::actors::signal::Signal;
use crate::api::AppState;
use crate::api::metrics::{ACTIVE_SUBSCRIPTIONS, WEBSOCKET_CLIENTS};

/// WebSocket subscription request from client
#[derive(Debug, Deserialize)]
struct SignalSubscribeRequest {
    /// Symbols to subscribe to (empty = all symbols)
    #[serde(default)]
    symbols: Vec<String>,

    /// Timeframes to filter by (empty = all timeframes)
    #[serde(default)]
    timeframes: Vec<String>,

    /// Signal types to receive (empty = all types)
    /// Examples: "ema_golden_cross", "rsi_overbought", "bullish_confluence"
    #[serde(default)]
    signal_types: Vec<String>,

    /// Directions to filter by (empty = all directions)
    /// Values: "bullish", "bearish"
    #[serde(default)]
    directions: Vec<String>,
}

/// WebSocket subscription response
#[derive(Debug, Serialize)]
struct SignalSubscribeResponse {
    success: bool,
    message: String,
    filters: FilterSummary,
}

#[derive(Debug, Serialize)]
struct FilterSummary {
    symbols: usize,
    timeframes: usize,
    signal_types: usize,
    directions: usize,
}

/// Signal message sent to client
#[derive(Debug, Serialize)]
struct SignalMessage {
    #[serde(rename = "type")]
    message_type: String,
    symbol: String,
    exchange: String,
    timeframe: String,
    signal_type: String,
    direction: String,
    strength: u8,
    timestamp: i64,
    price: f64,
    trigger_value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    trigger_value_2: Option<f64>,
    description: String,
}

/// Error message sent to client
#[derive(Debug, Serialize)]
struct ErrorMessage {
    #[serde(rename = "type")]
    message_type: String,
    error: String,
}

/// Warning message sent to client
#[derive(Debug, Serialize)]
struct WarningMessage {
    #[serde(rename = "type")]
    message_type: String,
    message: String,
}

/// Client session state
struct SignalClientSession {
    /// WebSocket sender
    sender: futures_util::stream::SplitSink<WebSocket, Message>,

    /// Broadcast receiver for signals from SignalActor
    signal_rx: broadcast::Receiver<Signal>,

    /// Subscribed symbols (empty = all symbols)
    symbols: HashSet<String>,

    /// Subscribed timeframes (empty = all timeframes)
    timeframes: HashSet<String>,

    /// Subscribed signal types (empty = all types)
    signal_types: HashSet<String>,

    /// Subscribed directions (empty = all directions)
    directions: HashSet<String>,

    /// Client ID for logging
    client_id: String,

    /// Signal counter for this session
    signals_sent: u64,
}

/// WebSocket upgrade handler for signal streaming
pub async fn ws_signals_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_signal_socket(socket, state))
}

/// Handle WebSocket connection for signal streaming
async fn handle_signal_socket(socket: WebSocket, state: AppState) {
    let client_id = uuid::Uuid::new_v4().to_string();
    info!("Signal WebSocket client connected: {}", client_id);

    // Check if SignalActor is available
    let signal_actor = match &state.signal_actor {
        Some(actor) => actor,
        None => {
            error!("SignalActor not available, rejecting WebSocket connection");
            let mut socket = socket;
            let _ = socket
                .close()
                .await
                .map_err(|e| error!("Failed to close socket: {}", e));
            return;
        }
    };

    // Increment connected clients metric
    WEBSOCKET_CLIENTS.inc();

    let (sender, mut receiver) = socket.split();

    // Subscribe to signal broadcast channel
    let signal_rx = signal_actor.signal_tx.subscribe();

    let mut session = SignalClientSession {
        sender,
        signal_rx,
        symbols: HashSet::new(),
        timeframes: HashSet::new(),
        signal_types: HashSet::new(),
        directions: HashSet::new(),
        client_id: client_id.clone(),
        signals_sent: 0,
    };

    // Send welcome message
    if let Err(e) = send_welcome_message(&mut session).await {
        error!("Failed to send welcome message: {}", e);
        WEBSOCKET_CLIENTS.dec();
        return;
    }

    // Main event loop
    loop {
        tokio::select! {
            // Handle incoming messages from client (subscription updates)
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(e) = handle_signal_subscription(&mut session, &text).await {
                            error!("Error handling subscription: {}", e);
                            let _ = send_error(&mut session, &format!("Subscription error: {}", e)).await;
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        info!("Signal client {} closed connection (sent {} signals)", client_id, session.signals_sent);
                        break;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        if let Err(e) = session.sender.send(Message::Pong(data)).await {
                            error!("Error sending pong: {}", e);
                            break;
                        }
                    }
                    Some(Ok(Message::Pong(_))) => {
                        // Pong received, connection alive
                    }
                    Some(Ok(_)) => {
                        warn!("Received unexpected message type from signal client");
                    }
                    Some(Err(e)) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                    None => {
                        info!("Signal client {} disconnected", client_id);
                        break;
                    }
                }
            }

            // Handle broadcast signals from SignalActor
            signal = session.signal_rx.recv() => {
                match signal {
                    Ok(signal) => {
                        if session.should_send(&signal)
                            && let Err(e) = send_signal_to_client(&mut session, &signal).await {
                                error!("Error sending signal to client: {}", e);
                                break;
                            }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!("Signal client {} lagged, skipped {} signals", client_id, skipped);
                        let _ = send_warning(&mut session, &format!("Client lagging, skipped {} signals", skipped)).await;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        info!("Signal broadcast channel closed");
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
    WEBSOCKET_CLIENTS.dec();
    let total_filters = session.symbols.len()
        + session.timeframes.len()
        + session.signal_types.len()
        + session.directions.len();
    ACTIVE_SUBSCRIPTIONS.sub(total_filters as i64);

    info!(
        "Signal WebSocket client disconnected: {} (sent {} signals)",
        client_id, session.signals_sent
    );
}

/// Send welcome message to newly connected client
async fn send_welcome_message(session: &mut SignalClientSession) -> Result<()> {
    let welcome = serde_json::json!({
        "type": "welcome",
        "message": "Connected to FKS Signal Stream",
        "client_id": session.client_id,
        "info": {
            "description": "Subscribe to real-time trading signals",
            "available_signal_types": [
                "ema_golden_cross",
                "ema_death_cross",
                "rsi_overbought",
                "rsi_oversold",
                "rsi_exit_overbought",
                "rsi_exit_oversold",
                "macd_bullish_cross",
                "macd_bearish_cross",
                "bullish_confluence",
                "bearish_confluence"
            ],
            "example_subscription": {
                "symbols": ["BTCUSD"],
                "timeframes": ["1m"],
                "signal_types": ["ema_golden_cross", "bullish_confluence"],
                "directions": ["bullish"]
            }
        }
    });

    let text = serde_json::to_string(&welcome)?;
    session.sender.send(Message::Text(text.into())).await?;
    Ok(())
}

/// Handle subscription request from client
async fn handle_signal_subscription(session: &mut SignalClientSession, text: &str) -> Result<()> {
    let request: SignalSubscribeRequest = serde_json::from_str(text)?;

    // Update subscription metrics (remove old counts)
    let old_count = session.symbols.len()
        + session.timeframes.len()
        + session.signal_types.len()
        + session.directions.len();

    // Update subscriptions
    session.symbols = request.symbols.into_iter().collect();
    session.timeframes = request.timeframes.into_iter().collect();
    session.signal_types = request.signal_types.into_iter().collect();
    session.directions = request.directions.into_iter().collect();

    let new_count = session.symbols.len()
        + session.timeframes.len()
        + session.signal_types.len()
        + session.directions.len();

    // Update metrics
    ACTIVE_SUBSCRIPTIONS.add((new_count as i64) - (old_count as i64));

    // Build filter summary
    let filter_summary = FilterSummary {
        symbols: session.symbols.len(),
        timeframes: session.timeframes.len(),
        signal_types: session.signal_types.len(),
        directions: session.directions.len(),
    };

    // Build response message
    let message = if new_count == 0 {
        "Subscribed to ALL signals (no filters)".to_string()
    } else {
        format!(
            "Filters: {} symbols, {} timeframes, {} types, {} directions",
            filter_summary.symbols,
            filter_summary.timeframes,
            filter_summary.signal_types,
            filter_summary.directions
        )
    };

    // Send acknowledgment
    let response = SignalSubscribeResponse {
        success: true,
        message,
        filters: filter_summary,
    };

    let response_text = serde_json::to_string(&response)?;
    session
        .sender
        .send(Message::Text(response_text.into()))
        .await?;

    debug!(
        "Signal client {} updated subscription: {} total filters",
        session.client_id, new_count
    );

    Ok(())
}

/// Send signal to client
async fn send_signal_to_client(session: &mut SignalClientSession, signal: &Signal) -> Result<()> {
    let message = SignalMessage {
        message_type: "signal".to_string(),
        symbol: signal.symbol.clone(),
        exchange: signal.exchange.clone(),
        timeframe: signal.timeframe.clone(),
        signal_type: signal.signal_type.as_str().to_string(),
        direction: signal.direction.as_str().to_string(),
        strength: signal.strength,
        timestamp: signal.timestamp,
        price: signal.price,
        trigger_value: signal.trigger_value,
        trigger_value_2: signal.trigger_value_2,
        description: signal.description.clone(),
    };

    let text = serde_json::to_string(&message)?;
    session.sender.send(Message::Text(text.into())).await?;

    session.signals_sent += 1;

    debug!(
        "Sent {} signal to client {}: {}:{}",
        signal.signal_type.as_str(),
        session.client_id,
        signal.symbol,
        signal.timeframe
    );

    Ok(())
}

/// Send error message to client
async fn send_error(session: &mut SignalClientSession, error_msg: &str) -> Result<()> {
    let error = ErrorMessage {
        message_type: "error".to_string(),
        error: error_msg.to_string(),
    };

    let text = serde_json::to_string(&error)?;
    session.sender.send(Message::Text(text.into())).await?;
    Ok(())
}

/// Send warning message to client
async fn send_warning(session: &mut SignalClientSession, warning_msg: &str) -> Result<()> {
    let warning = WarningMessage {
        message_type: "warning".to_string(),
        message: warning_msg.to_string(),
    };

    let text = serde_json::to_string(&warning)?;
    session.sender.send(Message::Text(text.into())).await?;
    Ok(())
}

impl SignalClientSession {
    /// Check if signal should be sent to this client based on subscriptions
    fn should_send(&self, signal: &Signal) -> bool {
        // Check symbol filter (empty = all symbols)
        let symbol_match = self.symbols.is_empty() || self.symbols.contains(&signal.symbol);

        // Check timeframe filter (empty = all timeframes)
        let timeframe_match =
            self.timeframes.is_empty() || self.timeframes.contains(&signal.timeframe);

        // Check signal type filter (empty = all types)
        let signal_type_match =
            self.signal_types.is_empty() || self.signal_types.contains(signal.signal_type.as_str());

        // Check direction filter (empty = all directions)
        let direction_match =
            self.directions.is_empty() || self.directions.contains(signal.direction.as_str());

        symbol_match && timeframe_match && signal_type_match && direction_match
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscribe_request_deserialization() {
        let json = r#"{
            "symbols": ["BTCUSD", "ETHUSDT"],
            "timeframes": ["1m", "5m"],
            "signal_types": ["ema_golden_cross"],
            "directions": ["bullish"]
        }"#;

        let request: SignalSubscribeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.symbols.len(), 2);
        assert_eq!(request.timeframes.len(), 2);
        assert_eq!(request.signal_types.len(), 1);
        assert_eq!(request.directions.len(), 1);
    }

    #[test]
    fn test_subscribe_response_serialization() {
        let response = SignalSubscribeResponse {
            success: true,
            message: "Subscribed to 2 symbols".to_string(),
            filters: FilterSummary {
                symbols: 2,
                timeframes: 1,
                signal_types: 1,
                directions: 1,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("true"));
        assert!(json.contains("Subscribed to 2 symbols"));
    }

    #[test]
    fn test_signal_message_serialization() {
        let message = SignalMessage {
            message_type: "signal".to_string(),
            symbol: "BTCUSD".to_string(),
            exchange: "binance".to_string(),
            timeframe: "1m".to_string(),
            signal_type: "ema_golden_cross".to_string(),
            direction: "bullish".to_string(),
            strength: 3,
            timestamp: 1737384600000,
            price: 42350.50,
            trigger_value: 42300.25,
            trigger_value_2: Some(42280.75),
            description: "EMA-8 crossed above EMA-21".to_string(),
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("signal"));
        assert!(json.contains("BTCUSD"));
        assert!(json.contains("ema_golden_cross"));
    }

    // Tests removed - they require constructing futures::stream::SplitSink which is not possible
    // Integration tests will cover WebSocket functionality
}
