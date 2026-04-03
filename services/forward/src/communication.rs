//! Communication module for signal dispatch via Redis Pub/Sub.
//!
//! This module handles receiving signals from the gateway (janus-gateway)
//! via Redis Pub/Sub channels. This is part of the hot path and must
//! maintain <50ms latency target.

use anyhow::{Context, Result};
use common::traits::Signal;
use serde::{Deserialize, Serialize};

use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

/// Redis channel names
pub const SIGNAL_CHANNEL: &str = "janus:signals";
pub const HEARTBEAT_CHANNEL: &str = "janus:heartbeat";

/// Signal message format for Redis Pub/Sub
/// This extends the base Signal with additional fields needed for transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalMessage {
    pub symbol: String,
    pub side: common::OrderSide,
    pub strength: f64,
    pub confidence: f64,
    pub predicted_duration_seconds: Option<u64>,
    pub entry_price: Option<common::Price>,
    pub stop_loss: Option<common::Price>,
    pub take_profit: Option<common::Price>,
}

impl From<SignalMessage> for Signal {
    fn from(msg: SignalMessage) -> Self {
        Signal {
            symbol: msg.symbol,
            side: msg.side,
            strength: msg.strength,
            confidence: msg.confidence,
            predicted_duration_seconds: msg.predicted_duration_seconds,
        }
    }
}

/// Signal dispatcher for receiving signals from the gateway
pub struct SignalReceiver {
    redis_url: String,
    signal_tx: mpsc::Sender<Signal>,
}

impl SignalReceiver {
    /// Create a new signal receiver
    ///
    /// # Arguments
    ///
    /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379/0")
    /// * `signal_tx` - Channel sender for forwarding signals to the trading engine
    pub fn new(redis_url: String, signal_tx: mpsc::Sender<Signal>) -> Self {
        Self {
            redis_url,
            signal_tx,
        }
    }

    /// Start listening for signals on Redis Pub/Sub
    ///
    /// This spawns a task that subscribes to the signal channel and forwards
    /// received signals to the trading engine via the channel.
    pub async fn start(&self) -> Result<()> {
        info!("Starting signal receiver on channel: {}", SIGNAL_CHANNEL);

        let redis_url = self.redis_url.clone();
        let mut signal_tx = self.signal_tx.clone();

        tokio::spawn(async move {
            loop {
                match Self::receive_loop(&redis_url, &mut signal_tx).await {
                    Ok(_) => {
                        warn!("Receive loop exited, restarting...");
                    }
                    Err(e) => {
                        error!("Error in receive loop: {}, restarting...", e);
                        // Wait before retrying to avoid tight loop
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Main receive loop for Redis Pub/Sub
    async fn receive_loop(redis_url: &str, signal_tx: &mut mpsc::Sender<Signal>) -> Result<()> {
        let client = redis::Client::open(redis_url).context("Failed to create Redis client")?;
        let mut pubsub = client
            .get_async_pubsub()
            .await
            .context("Failed to create pubsub connection")?;

        // Subscribe to the signal channel
        pubsub
            .subscribe(SIGNAL_CHANNEL)
            .await
            .context("Failed to subscribe to signal channel")?;

        info!("Subscribed to Redis channel: {}", SIGNAL_CHANNEL);

        // Process messages from the pubsub stream
        use futures_util::StreamExt;
        let mut stream = pubsub.on_message();
        while let Some(msg) = stream.next().await {
            match Self::handle_message(msg, signal_tx).await {
                Ok(_) => {
                    // Signal processed successfully
                }
                Err(e) => {
                    error!("Error handling signal message: {}", e);
                    // Continue processing other messages
                }
            }
        }

        warn!("Redis pubsub stream ended, reconnecting...");

        Ok(())
    }

    /// Handle a message from Redis Pub/Sub
    async fn handle_message(msg: redis::Msg, signal_tx: &mut mpsc::Sender<Signal>) -> Result<()> {
        let payload: Vec<u8> = msg.get_payload()?;
        let json_str = String::from_utf8(payload).context("Failed to decode message as UTF-8")?;

        // Deserialize signal message from JSON
        let signal_msg: SignalMessage =
            serde_json::from_str(&json_str).context("Failed to deserialize signal message")?;

        info!(
            "Received signal: {} {:?} (strength: {:.2}, confidence: {:.2})",
            signal_msg.symbol, signal_msg.side, signal_msg.strength, signal_msg.confidence
        );

        // Convert to core Signal type
        let signal: Signal = signal_msg.into();

        // Forward to trading engine
        signal_tx
            .send(signal)
            .await
            .context("Failed to send signal to trading engine")?;

        Ok(())
    }
}

/// Heartbeat monitor for Dead Man's Switch
///
/// Monitors heartbeats from the gateway and can trigger shutdown
/// if heartbeats stop (indicating gateway is down).
pub struct HeartbeatMonitor {
    redis_url: String,
    heartbeat_timeout: Duration,
    on_timeout: Arc<dyn Fn() + Send + Sync>,
}

impl HeartbeatMonitor {
    /// Create a new heartbeat monitor
    ///
    /// # Arguments
    ///
    /// * `redis_url` - Redis connection URL
    /// * `heartbeat_timeout` - Maximum time to wait for heartbeat before triggering timeout
    /// * `on_timeout` - Callback to execute when heartbeat times out (wrapped in Arc)
    pub fn new(
        redis_url: String,
        heartbeat_timeout: Duration,
        on_timeout: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        Self {
            redis_url,
            heartbeat_timeout,
            on_timeout,
        }
    }

    /// Start monitoring heartbeats
    ///
    /// This spawns a task that subscribes to the heartbeat channel and
    /// monitors for incoming heartbeats. If no heartbeat is received within
    /// the timeout period, the `on_timeout` callback is triggered.
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting heartbeat monitor on channel: {}",
            HEARTBEAT_CHANNEL
        );

        let redis_url = self.redis_url.clone();
        let heartbeat_timeout = self.heartbeat_timeout;
        let on_timeout = Arc::clone(&self.on_timeout);

        tokio::spawn(async move {
            loop {
                match Self::monitor_loop(&redis_url, heartbeat_timeout, &on_timeout).await {
                    Ok(_) => {
                        warn!("Heartbeat monitor loop exited, restarting...");
                    }
                    Err(e) => {
                        error!("Error in heartbeat monitor: {}, restarting...", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Main monitoring loop
    async fn monitor_loop(
        redis_url: &str,
        heartbeat_timeout: Duration,
        on_timeout: &Arc<dyn Fn() + Send + Sync>,
    ) -> Result<()> {
        let client = redis::Client::open(redis_url).context("Failed to create Redis client")?;
        let mut pubsub = client
            .get_async_pubsub()
            .await
            .context("Failed to create pubsub connection")?;

        pubsub
            .subscribe(HEARTBEAT_CHANNEL)
            .await
            .context("Failed to subscribe to heartbeat channel")?;

        info!("Subscribed to Redis channel: {}", HEARTBEAT_CHANNEL);

        let mut last_heartbeat = std::time::Instant::now();
        let mut check_interval = interval(Duration::from_secs(1));
        use futures_util::StreamExt;
        let mut stream = pubsub.on_message();

        loop {
            tokio::select! {
                // Check for heartbeat timeout
                _ = check_interval.tick() => {
                    if last_heartbeat.elapsed() > heartbeat_timeout {
                        warn!(
                            "Heartbeat timeout: {} seconds since last heartbeat",
                            last_heartbeat.elapsed().as_secs()
                        );
                        on_timeout();
                        // Optionally break or continue based on requirements
                    }
                }
                // Receive heartbeat message
                msg_opt = stream.next() => {
                    match msg_opt {
                        Some(msg) => {
                            let _payload: Vec<u8> = msg.get_payload()?;
                            last_heartbeat = std::time::Instant::now();
                            info!("Received heartbeat from gateway");
                        }
                        None => {
                            warn!("Heartbeat stream ended, reconnecting...");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
