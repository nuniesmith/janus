//! Kraken Fill Tracker
//!
//! This module integrates the Kraken private WebSocket with the execution engine
//! to provide real-time fill tracking and order status updates.
//!
//! # Overview
//!
//! The fill tracker:
//! - Connects to Kraken's authenticated WebSocket
//! - Subscribes to execution updates
//! - Processes fills and order status changes
//! - Broadcasts updates via channels for consumers
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::exchanges::kraken::fill_tracker::{KrakenFillTracker, FillTrackerConfig};
//!
//! let config = FillTrackerConfig::from_env();
//! let tracker = KrakenFillTracker::new(config);
//!
//! // Start tracking
//! tracker.start().await?;
//!
//! // Subscribe to fill events
//! let mut rx = tracker.subscribe_fills();
//! while let Ok(fill) = rx.recv().await {
//!     println!("Fill received: {} @ {}", fill.quantity, fill.price);
//! }
//! ```

use crate::error::{ExecutionError, Result};
use crate::exchanges::kraken::private_websocket::{
    KrakenPrivateWebSocket, KrakenPrivateWsConfig, OrderStatusEvent, PrivateWsEvent,
};
use crate::execution::histogram::global_latency_histograms;
use crate::types::{Fill, OrderStatusEnum};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, error, info, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Fill tracker configuration
#[derive(Debug, Clone)]
pub struct FillTrackerConfig {
    /// Kraken API key
    pub api_key: String,
    /// Kraken API secret
    pub api_secret: String,
    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,
    /// Request order snapshot on connect
    pub snap_orders: bool,
    /// Buffer size for fill events
    pub fill_buffer_size: usize,
    /// Buffer size for order status events
    pub order_status_buffer_size: usize,
}

impl Default for FillTrackerConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_secret: String::new(),
            auto_reconnect: true,
            snap_orders: true,
            fill_buffer_size: 1000,
            order_status_buffer_size: 1000,
        }
    }
}

impl FillTrackerConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let api_key = std::env::var("KRAKEN_API_KEY").unwrap_or_default();
        let api_secret = std::env::var("KRAKEN_API_SECRET").unwrap_or_default();

        let auto_reconnect = std::env::var("KRAKEN_FILL_TRACKER_AUTO_RECONNECT")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let snap_orders = std::env::var("KRAKEN_FILL_TRACKER_SNAP_ORDERS")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        Self {
            api_key,
            api_secret,
            auto_reconnect,
            snap_orders,
            ..Default::default()
        }
    }

    /// Check if credentials are configured
    pub fn has_credentials(&self) -> bool {
        !self.api_key.is_empty() && !self.api_secret.is_empty()
    }
}

// ============================================================================
// Fill Callback
// ============================================================================

/// Callback type for fill events
pub type FillCallback = Box<dyn Fn(Fill) + Send + Sync>;

/// Callback type for order status events
pub type OrderStatusCallback = Box<dyn Fn(String, OrderStatusEnum) + Send + Sync>;

// ============================================================================
// Fill Tracker
// ============================================================================

/// Kraken fill tracker for real-time execution updates
pub struct KrakenFillTracker {
    /// Configuration
    config: FillTrackerConfig,
    /// Private WebSocket client
    ws_client: Arc<KrakenPrivateWebSocket>,
    /// Fill event broadcaster
    fill_tx: broadcast::Sender<Fill>,
    /// Order status event broadcaster
    order_status_tx: broadcast::Sender<(String, OrderStatusEnum)>,
    /// Is the tracker running
    is_running: Arc<RwLock<bool>>,
    /// Order ID mappings (internal_id -> kraken_txid)
    order_id_map: Arc<RwLock<HashMap<String, String>>>,
    /// Reverse order ID mappings (kraken_txid -> internal_id)
    reverse_id_map: Arc<RwLock<HashMap<String, String>>>,
    /// Fill callbacks
    fill_callbacks: Arc<RwLock<Vec<FillCallback>>>,
    /// Order status callbacks
    status_callbacks: Arc<RwLock<Vec<OrderStatusCallback>>>,
    /// Order submission timestamps for fill latency tracking
    order_submission_times: Arc<RwLock<HashMap<String, Instant>>>,
}

impl KrakenFillTracker {
    /// Create a new fill tracker
    pub fn new(config: FillTrackerConfig) -> Self {
        let ws_config = KrakenPrivateWsConfig {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
            auto_reconnect: config.auto_reconnect,
            snap_orders: config.snap_orders,
            ..Default::default()
        };

        let ws_client = Arc::new(KrakenPrivateWebSocket::new(ws_config));
        let (fill_tx, _) = broadcast::channel(config.fill_buffer_size);
        let (order_status_tx, _) = broadcast::channel(config.order_status_buffer_size);

        Self {
            config,
            ws_client,
            fill_tx,
            order_status_tx,
            is_running: Arc::new(RwLock::new(false)),
            order_id_map: Arc::new(RwLock::new(HashMap::new())),
            reverse_id_map: Arc::new(RwLock::new(HashMap::new())),
            fill_callbacks: Arc::new(RwLock::new(Vec::new())),
            status_callbacks: Arc::new(RwLock::new(Vec::new())),
            order_submission_times: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self::new(FillTrackerConfig::from_env())
    }

    /// Subscribe to fill events
    pub fn subscribe_fills(&self) -> broadcast::Receiver<Fill> {
        self.fill_tx.subscribe()
    }

    /// Subscribe to order status events
    pub fn subscribe_order_status(&self) -> broadcast::Receiver<(String, OrderStatusEnum)> {
        self.order_status_tx.subscribe()
    }

    /// Register a fill callback
    pub async fn on_fill(&self, callback: FillCallback) {
        self.fill_callbacks.write().await.push(callback);
    }

    /// Register an order status callback
    pub async fn on_order_status(&self, callback: OrderStatusCallback) {
        self.status_callbacks.write().await.push(callback);
    }

    /// Register an order ID mapping
    pub async fn register_order(&self, internal_id: &str, kraken_txid: &str) {
        self.order_id_map
            .write()
            .await
            .insert(internal_id.to_string(), kraken_txid.to_string());
        self.reverse_id_map
            .write()
            .await
            .insert(kraken_txid.to_string(), internal_id.to_string());

        // Record submission time for fill latency tracking
        self.order_submission_times
            .write()
            .await
            .insert(internal_id.to_string(), Instant::now());

        debug!(
            "Registered order mapping: {} -> {}",
            internal_id, kraken_txid
        );
    }

    /// Record order submission time for latency tracking
    /// Call this when placing an order to track time until fill
    pub async fn record_order_submission(&self, internal_id: &str) {
        self.order_submission_times
            .write()
            .await
            .insert(internal_id.to_string(), Instant::now());
        debug!("Recorded submission time for order {}", internal_id);
    }

    /// Remove an order ID mapping
    pub async fn unregister_order(&self, internal_id: &str) {
        if let Some(kraken_txid) = self.order_id_map.write().await.remove(internal_id) {
            self.reverse_id_map.write().await.remove(&kraken_txid);
        }
        // Also remove submission time tracking
        self.order_submission_times
            .write()
            .await
            .remove(internal_id);
    }

    /// Get internal order ID from Kraken TXID
    pub async fn get_internal_id(&self, kraken_txid: &str) -> Option<String> {
        self.reverse_id_map.read().await.get(kraken_txid).cloned()
    }

    /// Get Kraken TXID from internal order ID
    pub async fn get_kraken_txid(&self, internal_id: &str) -> Option<String> {
        self.order_id_map.read().await.get(internal_id).cloned()
    }

    /// Check if the tracker is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Start the fill tracker
    pub async fn start(&self) -> Result<()> {
        if !self.config.has_credentials() {
            return Err(ExecutionError::AuthenticationError(
                "Kraken API credentials not configured for fill tracking".to_string(),
            ));
        }

        // Check if already running
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(ExecutionError::Internal(
                "Fill tracker already running".to_string(),
            ));
        }

        info!("Starting Kraken fill tracker...");

        // Authenticate WebSocket
        self.ws_client.authenticate().await?;

        // Subscribe to executions channel
        self.ws_client.subscribe_executions().await?;

        // Start WebSocket
        self.ws_client.start().await?;

        *is_running = true;
        drop(is_running);

        // Spawn event processing task
        let ws_client = self.ws_client.clone();
        let fill_tx = self.fill_tx.clone();
        let order_status_tx = self.order_status_tx.clone();
        let is_running = self.is_running.clone();
        let reverse_id_map = self.reverse_id_map.clone();
        let fill_callbacks = self.fill_callbacks.clone();
        let status_callbacks = self.status_callbacks.clone();
        let order_submission_times = self.order_submission_times.clone();

        tokio::spawn(async move {
            Self::process_events(
                ws_client,
                fill_tx,
                order_status_tx,
                is_running,
                reverse_id_map,
                fill_callbacks,
                status_callbacks,
                order_submission_times,
            )
            .await;
        });

        info!("Kraken fill tracker started successfully");
        Ok(())
    }

    /// Stop the fill tracker
    pub async fn stop(&self) {
        info!("Stopping Kraken fill tracker...");
        *self.is_running.write().await = false;
        self.ws_client.stop().await;
        info!("Kraken fill tracker stopped");
    }

    /// Process WebSocket events
    #[allow(clippy::too_many_arguments)]
    async fn process_events(
        ws_client: Arc<KrakenPrivateWebSocket>,
        fill_tx: broadcast::Sender<Fill>,
        order_status_tx: broadcast::Sender<(String, OrderStatusEnum)>,
        is_running: Arc<RwLock<bool>>,
        reverse_id_map: Arc<RwLock<HashMap<String, String>>>,
        fill_callbacks: Arc<RwLock<Vec<FillCallback>>>,
        status_callbacks: Arc<RwLock<Vec<OrderStatusCallback>>>,
        order_submission_times: Arc<RwLock<HashMap<String, Instant>>>,
    ) {
        let mut event_rx = ws_client.subscribe_events();

        while *is_running.read().await {
            match event_rx.recv().await {
                Ok(event) => {
                    match event {
                        PrivateWsEvent::Execution(exec) => {
                            // Convert execution to fill
                            let fill = exec.to_fill();

                            // Try to resolve internal order ID
                            let order_id = {
                                let map = reverse_id_map.read().await;
                                map.get(&exec.order_id).cloned()
                            };

                            let mut resolved_fill = fill.clone();
                            if let Some(internal_id) = &order_id {
                                resolved_fill.order_id = internal_id.clone();
                            }

                            info!(
                                "Fill received: order={} qty={} price={} fee={}",
                                resolved_fill.order_id,
                                resolved_fill.quantity,
                                resolved_fill.price,
                                resolved_fill.fee
                            );

                            // Record fill latency in histogram
                            if let Some(internal_id) = &order_id {
                                let submission_times = order_submission_times.read().await;
                                if let Some(submission_time) = submission_times.get(internal_id) {
                                    let fill_latency_ms =
                                        submission_time.elapsed().as_secs_f64() * 1000.0;
                                    global_latency_histograms()
                                        .record_order_fill("kraken", fill_latency_ms);
                                    debug!(
                                        "Order {} fill latency: {:.2}ms",
                                        internal_id, fill_latency_ms
                                    );
                                }
                            }

                            // Broadcast fill
                            let _ = fill_tx.send(resolved_fill.clone());

                            // Call fill callbacks
                            let callbacks = fill_callbacks.read().await;
                            for callback in callbacks.iter() {
                                callback(resolved_fill.clone());
                            }
                        }
                        PrivateWsEvent::OrderStatus(status) => {
                            // Try to resolve internal order ID
                            let order_id = {
                                let map = reverse_id_map.read().await;
                                map.get(&status.order_id)
                                    .cloned()
                                    .unwrap_or_else(|| status.order_id.clone())
                            };

                            debug!("Order status update: {} -> {:?}", order_id, status.status);

                            // Broadcast status
                            let _ = order_status_tx.send((order_id.clone(), status.status));

                            // Call status callbacks
                            let callbacks = status_callbacks.read().await;
                            for callback in callbacks.iter() {
                                callback(order_id.clone(), status.status);
                            }
                        }
                        PrivateWsEvent::Connected => {
                            info!("Fill tracker WebSocket connected");
                        }
                        PrivateWsEvent::Disconnected => {
                            warn!("Fill tracker WebSocket disconnected");
                        }
                        PrivateWsEvent::Authenticated => {
                            info!("Fill tracker authenticated");
                        }
                        PrivateWsEvent::AuthenticationFailed(err) => {
                            error!("Fill tracker authentication failed: {}", err);
                        }
                        PrivateWsEvent::Subscribed { channel } => {
                            info!("Fill tracker subscribed to channel: {}", channel);
                        }
                        PrivateWsEvent::BalanceUpdate(balance) => {
                            debug!(
                                "Balance update: {} available={} total={}",
                                balance.currency, balance.available, balance.total
                            );
                        }
                        PrivateWsEvent::Error(err) => {
                            error!("Fill tracker WebSocket error: {}", err);
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Fill tracker lagged {} events", n);
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("Fill tracker event channel closed");
                    break;
                }
            }
        }

        info!("Fill tracker event processing stopped");
    }

    /// Query order status from exchange
    ///
    /// This performs a REST API call to get the current order status.
    /// Use this for reconciliation when the WebSocket might have missed updates.
    pub async fn query_order_status(&self, kraken_txid: &str) -> Result<OrderStatusEvent> {
        let order = self.ws_client.get_pending_order(kraken_txid).await;

        order.ok_or_else(|| {
            ExecutionError::OrderNotFound(format!(
                "Order {} not found in pending orders",
                kraken_txid
            ))
        })
    }

    /// Get all pending orders from WebSocket state
    pub async fn get_pending_orders(&self) -> Vec<OrderStatusEvent> {
        self.ws_client.get_all_pending_orders().await
    }

    /// Reconcile orders with exchange state
    ///
    /// This is useful after reconnection to ensure local state matches exchange state.
    pub async fn reconcile_orders(&self) -> Result<ReconciliationResult> {
        info!("Reconciling orders with Kraken...");

        let pending_orders = self.get_pending_orders().await;
        let registered_orders: Vec<_> = self.order_id_map.read().await.keys().cloned().collect();

        let mut matched = 0;
        let mut missing_local = 0;
        let mut missing_exchange = 0;
        let status_mismatches = Vec::new();

        // Check each pending order from exchange
        let pending_txids: std::collections::HashSet<_> =
            pending_orders.iter().map(|o| o.order_id.clone()).collect();

        for order in &pending_orders {
            if let Some(_internal_id) = self.get_internal_id(&order.order_id).await {
                matched += 1;
            } else {
                missing_local += 1;
                debug!(
                    "Order {} exists on exchange but not tracked locally",
                    order.order_id
                );
            }
        }

        // Check each registered order
        for internal_id in &registered_orders {
            if let Some(txid) = self.get_kraken_txid(internal_id).await {
                if !pending_txids.contains(&txid) {
                    // Order not in pending - might be filled/cancelled
                    missing_exchange += 1;
                    debug!(
                        "Local order {} (txid: {}) not in exchange pending orders",
                        internal_id, txid
                    );
                }
            }
        }

        let result = ReconciliationResult {
            total_registered: registered_orders.len(),
            total_exchange: pending_orders.len(),
            matched,
            missing_local,
            missing_exchange,
            status_mismatches,
        };

        info!(
            "Reconciliation complete: {} matched, {} missing local, {} missing exchange",
            result.matched, result.missing_local, result.missing_exchange
        );

        Ok(result)
    }
}

/// Result of order reconciliation
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    /// Total orders registered locally
    pub total_registered: usize,
    /// Total orders on exchange
    pub total_exchange: usize,
    /// Orders matched between local and exchange
    pub matched: usize,
    /// Orders on exchange but not tracked locally
    pub missing_local: usize,
    /// Orders tracked locally but not on exchange (likely filled/cancelled)
    pub missing_exchange: usize,
    /// Orders with mismatched status
    pub status_mismatches: Vec<StatusMismatch>,
}

/// Order status mismatch
#[derive(Debug, Clone)]
pub struct StatusMismatch {
    /// Order ID
    pub order_id: String,
    /// Local status
    pub local_status: OrderStatusEnum,
    /// Exchange status
    pub exchange_status: OrderStatusEnum,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FillTrackerConfig::default();
        assert!(config.auto_reconnect);
        assert!(config.snap_orders);
        assert_eq!(config.fill_buffer_size, 1000);
    }

    #[test]
    fn test_config_has_credentials() {
        let mut config = FillTrackerConfig::default();
        assert!(!config.has_credentials());

        config.api_key = "test_key".to_string();
        config.api_secret = "test_secret".to_string();
        assert!(config.has_credentials());
    }

    #[tokio::test]
    async fn test_fill_tracker_creation() {
        let config = FillTrackerConfig::default();
        let tracker = KrakenFillTracker::new(config);

        assert!(!tracker.is_running().await);
    }

    #[tokio::test]
    async fn test_order_id_mapping() {
        let config = FillTrackerConfig::default();
        let tracker = KrakenFillTracker::new(config);

        tracker
            .register_order("internal-123", "OABCDE-12345-FGHIJK")
            .await;

        assert_eq!(
            tracker.get_kraken_txid("internal-123").await,
            Some("OABCDE-12345-FGHIJK".to_string())
        );
        assert_eq!(
            tracker.get_internal_id("OABCDE-12345-FGHIJK").await,
            Some("internal-123".to_string())
        );

        tracker.unregister_order("internal-123").await;

        assert_eq!(tracker.get_kraken_txid("internal-123").await, None);
        assert_eq!(tracker.get_internal_id("OABCDE-12345-FGHIJK").await, None);
    }

    #[tokio::test]
    async fn test_subscribe_channels() {
        let config = FillTrackerConfig::default();
        let tracker = KrakenFillTracker::new(config);

        // Subscribe to fills
        let _fill_rx = tracker.subscribe_fills();

        // Subscribe to order status
        let _status_rx = tracker.subscribe_order_status();
    }

    #[test]
    fn test_reconciliation_result() {
        let result = ReconciliationResult {
            total_registered: 10,
            total_exchange: 8,
            matched: 7,
            missing_local: 1,
            missing_exchange: 3,
            status_mismatches: vec![],
        };

        assert_eq!(result.matched, 7);
        assert_eq!(result.missing_local, 1);
        assert_eq!(result.missing_exchange, 3);
    }
}
