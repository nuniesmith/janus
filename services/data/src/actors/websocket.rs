//! WebSocket Actor - Manages persistent WebSocket connections
//!
//! This actor handles WebSocket connections to exchanges with automatic
//! reconnection, ping/pong handling, and error recovery.
//!
//! ## Features:
//! - Automatic reconnection with exponential backoff
//! - Heartbeat/ping-pong monitoring
//! - Connection health tracking
//! - Graceful shutdown handling
//! - Message deserialization and routing via ExchangeConnector

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, sleep};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};

use super::{ActorStats, DataMessage, ExchangeConnector};

/// WebSocket actor configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// WebSocket URL to connect to
    pub url: String,

    /// Exchange name
    pub exchange: String,

    /// Symbol to subscribe to
    pub symbol: String,

    /// Subscription message (JSON)
    pub subscription_msg: Option<String>,

    /// Ping interval (seconds)
    pub ping_interval_secs: u64,

    /// Reconnection delay (seconds)
    pub reconnect_delay_secs: u64,

    /// Maximum reconnection attempts before giving up
    pub max_reconnect_attempts: u32,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            exchange: String::new(),
            symbol: String::new(),
            subscription_msg: None,
            ping_interval_secs: 20,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }
}

/// WebSocket actor that manages a single WebSocket connection
pub struct WebSocketActor {
    config: WebSocketConfig,

    /// Exchange connector for parsing messages
    connector: Arc<dyn ExchangeConnector>,

    /// Sender to router
    router_tx: mpsc::UnboundedSender<DataMessage>,

    /// Shutdown signal
    shutdown_rx: broadcast::Receiver<()>,

    /// Statistics
    stats: ActorStats,

    /// Current reconnection attempt
    reconnect_count: u32,
}

impl WebSocketActor {
    /// Create a new WebSocket actor
    pub fn new(
        config: WebSocketConfig,
        connector: Arc<dyn ExchangeConnector>,
        router_tx: mpsc::UnboundedSender<DataMessage>,
        shutdown_rx: broadcast::Receiver<()>,
    ) -> Self {
        Self {
            config,
            connector,
            router_tx,
            shutdown_rx,
            stats: ActorStats::new(),
            reconnect_count: 0,
        }
    }

    /// Start the WebSocket actor
    pub async fn run(mut self) -> Result<()> {
        info!(
            "WebSocket Actor: Starting for {} on {} ({})",
            self.config.symbol, self.config.exchange, self.config.url
        );

        loop {
            // Attempt to connect
            match self.connect_and_run().await {
                Ok(_) => {
                    info!(
                        "WebSocket Actor: Connection closed gracefully for {} on {}",
                        self.config.symbol, self.config.exchange
                    );
                    break;
                }
                Err(e) => {
                    error!(
                        "WebSocket Actor: Connection failed for {} on {}: {}",
                        self.config.symbol, self.config.exchange, e
                    );

                    self.reconnect_count += 1;

                    if self.reconnect_count >= self.config.max_reconnect_attempts {
                        error!(
                            "WebSocket Actor: Max reconnection attempts reached for {} on {}",
                            self.config.symbol, self.config.exchange
                        );
                        break;
                    }

                    // Exponential backoff
                    let delay =
                        self.config.reconnect_delay_secs * (2_u64.pow(self.reconnect_count.min(5)));
                    warn!(
                        "WebSocket Actor: Reconnecting in {} seconds (attempt {}/{})",
                        delay, self.reconnect_count, self.config.max_reconnect_attempts
                    );

                    sleep(Duration::from_secs(delay)).await;
                }
            }
        }

        info!(
            "WebSocket Actor: Stopped for {} on {}. Stats: {:?}",
            self.config.symbol, self.config.exchange, self.stats
        );
        Ok(())
    }

    /// Connect to WebSocket and run the main loop
    async fn connect_and_run(&mut self) -> Result<()> {
        // Connect to WebSocket
        let (ws_stream, _) = connect_async(&self.config.url)
            .await
            .context("Failed to connect to WebSocket")?;

        info!(
            "WebSocket Actor: Connected to {} for {}",
            self.config.exchange, self.config.symbol
        );

        // Reset reconnect counter on successful connection
        self.reconnect_count = 0;

        let (mut write, mut read) = ws_stream.split();

        // Send subscription message if configured
        if let Some(ref sub_msg) = self.config.subscription_msg {
            debug!("WebSocket Actor: Sending subscription: {}", sub_msg);
            write
                .send(Message::Text(sub_msg.clone().into()))
                .await
                .context("Failed to send subscription message")?;
        }

        // Set up ping interval
        let mut ping_interval = interval(Duration::from_secs(self.config.ping_interval_secs));

        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                Some(msg_result) = read.next() => {
                    match msg_result {
                        Ok(msg) => {
                            if let Err(e) = self.handle_message(msg).await {
                                error!("WebSocket Actor: Failed to handle message: {}", e);
                                self.stats.record_failure();
                            }
                        }
                        Err(e) => {
                            error!("WebSocket Actor: Receive error: {}", e);
                            return Err(e.into());
                        }
                    }
                }

                // Send periodic pings
                _ = ping_interval.tick() => {
                    debug!("WebSocket Actor: Sending ping to {}", self.config.exchange);
                    if let Err(e) = write.send(Message::Ping(vec![].into())).await {
                        error!("WebSocket Actor: Failed to send ping: {}", e);
                        return Err(e.into());
                    }
                }

                // Handle shutdown
                _ = self.shutdown_rx.recv() => {
                    info!("WebSocket Actor: Shutdown signal received");
                    let _ = write.send(Message::Close(None)).await;
                    return Ok(());
                }
            }
        }
    }

    /// Handle a single WebSocket message
    async fn handle_message(&mut self, msg: Message) -> Result<()> {
        match msg {
            Message::Text(text) => {
                // Parse the message using the exchange-specific connector
                match self.connector.parse_message(&text) {
                    Ok(data_messages) => {
                        if data_messages.is_empty() {
                            // Not a trade message (could be heartbeat, subscription ack, etc.)
                            debug!(
                                "WebSocket Actor: Non-trade message from {}: {}",
                                self.config.exchange,
                                if text.len() > 100 {
                                    &text[..100]
                                } else {
                                    &text
                                }
                            );
                            return Ok(());
                        }

                        // Route all parsed messages to the router
                        for data_msg in data_messages {
                            match &data_msg {
                                DataMessage::Trade(trade) => {
                                    debug!(
                                        "WebSocket Actor: Parsed trade {} {} @ {} from {}",
                                        trade.symbol, trade.side, trade.price, trade.exchange
                                    );
                                }
                                DataMessage::Candle(candle) => {
                                    debug!(
                                        "WebSocket Actor: Parsed candle {} {} from {}",
                                        candle.symbol, candle.interval, candle.exchange
                                    );
                                }
                                _ => {}
                            }

                            // Send to router
                            if let Err(e) = self.router_tx.send(data_msg) {
                                error!("WebSocket Actor: Failed to send to router: {}", e);
                                self.stats.record_failure();
                                return Err(anyhow::anyhow!("Router channel closed"));
                            }

                            self.stats.record_success();
                        }
                    }
                    Err(e) => {
                        // Log parsing errors but don't fail the connection
                        debug!(
                            "WebSocket Actor: Failed to parse message from {}: {} - Raw: {}",
                            self.config.exchange,
                            e,
                            if text.len() > 200 {
                                &text[..200]
                            } else {
                                &text
                            }
                        );
                    }
                }
            }

            Message::Binary(data) => {
                debug!(
                    "WebSocket Actor: Received binary message from {} ({} bytes)",
                    self.config.exchange,
                    data.len()
                );
                // Some exchanges use binary frames (e.g., compressed data)
                // For now, we don't handle these - would need exchange-specific decompression
            }

            Message::Ping(_data) => {
                debug!(
                    "WebSocket Actor: Received ping from {}",
                    self.config.exchange
                );
                // Tungstenite handles pong automatically
            }

            Message::Pong(_) => {
                debug!(
                    "WebSocket Actor: Received pong from {}",
                    self.config.exchange
                );
            }

            Message::Close(frame) => {
                info!(
                    "WebSocket Actor: Received close frame from {}: {:?}",
                    self.config.exchange, frame
                );
                return Err(anyhow::anyhow!("Connection closed by server"));
            }

            Message::Frame(_) => {
                // Raw frames are not typically handled directly
                debug!("WebSocket Actor: Received raw frame");
            }
        }

        Ok(())
    }

    /// Get current statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> &ActorStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_config_default() {
        let config = WebSocketConfig::default();
        assert_eq!(config.ping_interval_secs, 20);
        assert_eq!(config.reconnect_delay_secs, 5);
        assert_eq!(config.max_reconnect_attempts, 10);
    }

    #[test]
    fn test_exponential_backoff() {
        let config = WebSocketConfig::default();
        assert_eq!(config.reconnect_delay_secs * 2_u64.pow(0), 5);
        assert_eq!(config.reconnect_delay_secs * 2_u64.pow(1), 10);
        assert_eq!(config.reconnect_delay_secs * 2_u64.pow(2), 20);
        assert_eq!(config.reconnect_delay_secs * 2_u64.pow(3), 40);
    }
}
