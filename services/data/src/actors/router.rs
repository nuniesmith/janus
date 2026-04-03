//! Router Actor - Central message dispatcher
//!
//! The Router is the central hub of the Data Factory. It receives messages from
//! all WebSocket and Poller actors and routes them to the appropriate storage
//! destination (QuestDB, Redis, etc.).
//!
//! ## Responsibilities:
//! - Message validation and filtering
//! - Routing to appropriate storage backends
//! - Message queueing and backpressure handling
//! - Statistics tracking and health monitoring
//! - Forwarding candle data to IndicatorActor for real-time indicator calculation

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast, mpsc};
use tracing::{debug, error, info, warn};

use super::DataMessage;
use super::indicator::{CandleInput, IndicatorMessage};
use crate::config::Config;
use crate::storage::StorageManager;

/// Normalized message for broadcasting to clients
#[derive(Debug, Clone)]
pub struct NormalizedMessage {
    pub timestamp: i64,
    pub symbol: String,
    pub exchange: String,
    pub side: String,
    pub price: f64,
    pub amount: f64,
    pub trade_id: String,
}

/// Router actor that dispatches messages to storage
pub struct Router {
    #[allow(dead_code)]
    config: Arc<Config>,
    #[allow(dead_code)]
    storage: Arc<StorageManager>,

    /// Sender for incoming messages (cloned and given to producers)
    tx: mpsc::UnboundedSender<DataMessage>,

    /// Broadcast sender for real-time client streaming (public for API server)
    pub broadcast_tx: broadcast::Sender<NormalizedMessage>,

    /// Optional indicator actor sender for candle forwarding
    indicator_tx: Arc<RwLock<Option<mpsc::UnboundedSender<IndicatorMessage>>>>,

    /// Shutdown signal sender (for cloning)
    #[allow(dead_code)]
    shutdown_tx: broadcast::Sender<()>,
}

impl Router {
    /// Create a new Router instance
    pub async fn new(
        config: Arc<Config>,
        storage: Arc<StorageManager>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> Result<Arc<Self>> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (shutdown_tx, _) = broadcast::channel(16);
        let (broadcast_tx, _) = broadcast::channel(1000); // Buffer for 1000 messages
        let indicator_tx = Arc::new(RwLock::new(None));

        let router = Arc::new(Self {
            config: config.clone(),
            storage: storage.clone(),
            tx: tx.clone(),
            broadcast_tx: broadcast_tx.clone(),
            indicator_tx: indicator_tx.clone(),
            shutdown_tx: shutdown_tx.clone(),
        });

        // Spawn the router task
        let storage_clone = storage.clone();
        let broadcast_clone = broadcast_tx.clone();
        let indicator_tx_clone = indicator_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::run_loop(
                storage_clone,
                broadcast_clone,
                indicator_tx_clone,
                &mut rx,
                &mut shutdown_rx,
            )
            .await
            {
                error!("Router task failed: {}", e);
            }
        });

        Ok(router)
    }

    /// Get a sender handle for this router
    pub fn get_sender(&self) -> mpsc::UnboundedSender<DataMessage> {
        self.tx.clone()
    }

    /// Set the indicator actor sender for candle forwarding
    pub async fn set_indicator_sender(&self, sender: mpsc::UnboundedSender<IndicatorMessage>) {
        let mut indicator_tx = self.indicator_tx.write().await;
        *indicator_tx = Some(sender);
        info!("Router: Indicator actor connected for candle forwarding");
    }

    /// Main router loop (internal)
    async fn run_loop(
        storage: Arc<StorageManager>,
        broadcast_tx: broadcast::Sender<NormalizedMessage>,
        indicator_tx: Arc<RwLock<Option<mpsc::UnboundedSender<IndicatorMessage>>>>,
        rx: &mut mpsc::UnboundedReceiver<DataMessage>,
        shutdown_rx: &mut broadcast::Receiver<()>,
    ) -> Result<()> {
        info!("Router: Starting message processing loop");

        let mut stats_interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
        let mut message_count = 0u64;
        let mut candles_forwarded = 0u64;

        loop {
            tokio::select! {
                // Process incoming messages
                Some(msg) = rx.recv() => {
                    match Self::handle_message(
                        &storage,
                        &broadcast_tx,
                        &indicator_tx,
                        msg,
                        &mut candles_forwarded,
                    ).await { Err(e) => {
                        error!("Router: Failed to handle message: {}", e);
                    } _ => {
                        message_count += 1;
                    }}
                }

                // Print statistics periodically
                _ = stats_interval.tick() => {
                    info!(
                        "Router Stats - Processed: {} messages, Candles forwarded to indicators: {}",
                        message_count, candles_forwarded
                    );
                }

                // Handle shutdown
                _ = shutdown_rx.recv() => {
                    info!("Router: Shutdown signal received");
                    break;
                }
            }
        }

        info!("Router: Stopped");
        Ok(())
    }

    /// Handle a single message
    async fn handle_message(
        storage: &Arc<StorageManager>,
        broadcast_tx: &broadcast::Sender<NormalizedMessage>,
        indicator_tx: &Arc<RwLock<Option<mpsc::UnboundedSender<IndicatorMessage>>>>,
        msg: DataMessage,
        candles_forwarded: &mut u64,
    ) -> Result<()> {
        match msg {
            DataMessage::Trade(trade) => {
                debug!(
                    "Router: Trade {} {} @ {} on {}",
                    trade.symbol, trade.side, trade.price, trade.exchange
                );

                // Broadcast to connected clients
                let normalized = NormalizedMessage {
                    timestamp: trade.exchange_ts,
                    symbol: trade.symbol.clone(),
                    exchange: trade.exchange.clone(),
                    side: trade.side.to_string(),
                    price: trade.price,
                    amount: trade.amount,
                    trade_id: trade.trade_id.clone(),
                };
                let _ = broadcast_tx.send(normalized); // Ignore if no receivers

                storage.store_trade(trade).await?;
            }

            DataMessage::Candle(candle) => {
                debug!(
                    "Router: Candle {} {} [{}-{}] on {}",
                    candle.symbol, candle.interval, candle.open, candle.close, candle.exchange
                );

                // Store to QuestDB
                storage.store_candle(candle.clone()).await?;

                // Forward to indicator actor if connected
                let indicator_sender = indicator_tx.read().await;
                if let Some(ref sender) = *indicator_sender {
                    let candle_input = CandleInput {
                        symbol: candle.symbol.clone(),
                        exchange: candle.exchange.clone(),
                        timeframe: candle.interval.clone(),
                        timestamp: candle.open_time,
                        open: candle.open,
                        high: candle.high,
                        low: candle.low,
                        close: candle.close,
                        volume: candle.volume,
                    };

                    match sender.send(IndicatorMessage::Candle(candle_input)) {
                        Err(e) => {
                            warn!("Router: Failed to forward candle to indicator actor: {}", e);
                        }
                        _ => {
                            *candles_forwarded += 1;
                        }
                    }
                }
            }

            DataMessage::Metric(metric) => {
                debug!(
                    "Router: Metric {} = {} for {} from {}",
                    metric.metric_type, metric.value, metric.asset, metric.source
                );
                storage.store_metric(metric).await?;
            }

            DataMessage::Health(health) => {
                debug!(
                    "Router: Health check from {} - {}",
                    health.component, health.status
                );
                storage.store_health(health).await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_router_creation() {
        // This is a placeholder test - would need mocked storage
        // let config = Arc::new(Config::from_env().unwrap());
        // let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        // Add actual test implementation
    }
}
