//! Bybit Fill Tracker
//!
//! This module integrates the Bybit private WebSocket with the execution engine
//! to provide real-time fill tracking and order status updates.
//!
//! # Overview
//!
//! The fill tracker:
//! - Connects to Bybit's authenticated WebSocket
//! - Subscribes to order and execution updates
//! - Processes fills and order status changes
//! - Broadcasts updates via channels for consumers
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::exchanges::bybit::fill_tracker::{BybitFillTracker, FillTrackerConfig};
//!
//! let config = FillTrackerConfig::from_env();
//! let tracker = BybitFillTracker::new(config);
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
use crate::exchanges::bybit::websocket::{
    BybitEvent, BybitWebSocket, BybitWsConfig, OrderUpdate as BybitOrderUpdate,
};
use crate::execution::histogram::global_latency_histograms;
use crate::types::{Fill, OrderSide, OrderStatusEnum};
use chrono::Utc;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ============================================================================
// Configuration
// ============================================================================

/// Fill tracker configuration
#[derive(Debug, Clone)]
pub struct FillTrackerConfig {
    /// Bybit API key
    pub api_key: String,
    /// Bybit API secret
    pub api_secret: String,
    /// Use testnet
    pub testnet: bool,
    /// Auto-reconnect on disconnect
    pub auto_reconnect: bool,
    /// Subscribe to order updates
    pub subscribe_orders: bool,
    /// Subscribe to position updates
    pub subscribe_positions: bool,
    /// Subscribe to wallet updates
    pub subscribe_wallet: bool,
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
            testnet: false,
            auto_reconnect: true,
            subscribe_orders: true,
            subscribe_positions: false,
            subscribe_wallet: false,
            fill_buffer_size: 1000,
            order_status_buffer_size: 1000,
        }
    }
}

impl FillTrackerConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let api_key = std::env::var("BYBIT_API_KEY").unwrap_or_default();
        let api_secret = std::env::var("BYBIT_API_SECRET").unwrap_or_default();

        let testnet = std::env::var("BYBIT_TESTNET")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);

        let auto_reconnect = std::env::var("BYBIT_FILL_TRACKER_AUTO_RECONNECT")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let subscribe_orders = std::env::var("BYBIT_FILL_TRACKER_SUBSCRIBE_ORDERS")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let subscribe_positions = std::env::var("BYBIT_FILL_TRACKER_SUBSCRIBE_POSITIONS")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);

        Self {
            api_key,
            api_secret,
            testnet,
            auto_reconnect,
            subscribe_orders,
            subscribe_positions,
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

/// Bybit fill tracker for real-time execution updates
pub struct BybitFillTracker {
    /// Configuration
    config: FillTrackerConfig,
    /// Private WebSocket client
    ws_client: Arc<BybitWebSocket>,
    /// Fill event broadcaster
    fill_tx: broadcast::Sender<Fill>,
    /// Order status event broadcaster
    order_status_tx: broadcast::Sender<(String, OrderStatusEnum)>,
    /// Is the tracker running
    is_running: Arc<RwLock<bool>>,
    /// Order ID mapping: internal ID -> Bybit order ID
    order_id_map: Arc<RwLock<HashMap<String, String>>>,
    /// Reverse mapping: Bybit order ID -> internal ID
    reverse_id_map: Arc<RwLock<HashMap<String, String>>>,
    /// Client order ID mapping: internal ID -> client order ID
    client_order_map: Arc<RwLock<HashMap<String, String>>>,
    /// Reverse client order mapping: client order ID -> internal ID
    reverse_client_order_map: Arc<RwLock<HashMap<String, String>>>,
    /// Fill callbacks
    fill_callbacks: Arc<RwLock<Vec<FillCallback>>>,
    /// Order status callbacks
    status_callbacks: Arc<RwLock<Vec<OrderStatusCallback>>>,
    /// Last known order states for fill detection
    last_order_states: Arc<RwLock<HashMap<String, BybitOrderUpdate>>>,
    /// Order submission timestamps for fill latency tracking
    order_submission_times: Arc<RwLock<HashMap<String, Instant>>>,
}

impl BybitFillTracker {
    /// Create a new fill tracker
    pub fn new(config: FillTrackerConfig) -> Self {
        let ws_config = BybitWsConfig {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
            testnet: config.testnet,
            subscribe_orders: config.subscribe_orders,
            subscribe_positions: config.subscribe_positions,
            subscribe_wallet: config.subscribe_wallet,
        };

        let ws_client = Arc::new(BybitWebSocket::new(ws_config));

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
            client_order_map: Arc::new(RwLock::new(HashMap::new())),
            reverse_client_order_map: Arc::new(RwLock::new(HashMap::new())),
            fill_callbacks: Arc::new(RwLock::new(Vec::new())),
            status_callbacks: Arc::new(RwLock::new(Vec::new())),
            last_order_states: Arc::new(RwLock::new(HashMap::new())),
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
    ///
    /// # Arguments
    /// * `internal_id` - Internal order ID used by the execution engine
    /// * `bybit_order_id` - Bybit's order ID
    /// * `client_order_id` - Optional client order ID (orderLinkId in Bybit)
    pub async fn register_order(
        &self,
        internal_id: &str,
        bybit_order_id: &str,
        client_order_id: Option<&str>,
    ) {
        self.order_id_map
            .write()
            .await
            .insert(internal_id.to_string(), bybit_order_id.to_string());
        self.reverse_id_map
            .write()
            .await
            .insert(bybit_order_id.to_string(), internal_id.to_string());

        if let Some(client_id) = client_order_id {
            self.client_order_map
                .write()
                .await
                .insert(internal_id.to_string(), client_id.to_string());
            self.reverse_client_order_map
                .write()
                .await
                .insert(client_id.to_string(), internal_id.to_string());
        }

        // Record submission time for fill latency tracking
        self.order_submission_times
            .write()
            .await
            .insert(internal_id.to_string(), Instant::now());

        debug!(
            "Registered order mapping: {} -> {} (client: {:?})",
            internal_id, bybit_order_id, client_order_id
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
        if let Some(bybit_order_id) = self.order_id_map.write().await.remove(internal_id) {
            self.reverse_id_map.write().await.remove(&bybit_order_id);
            self.last_order_states.write().await.remove(&bybit_order_id);
        }
        if let Some(client_order_id) = self.client_order_map.write().await.remove(internal_id) {
            self.reverse_client_order_map
                .write()
                .await
                .remove(&client_order_id);
        }
        // Also remove submission time tracking
        self.order_submission_times
            .write()
            .await
            .remove(internal_id);
    }

    /// Get internal order ID from Bybit order ID
    pub async fn get_internal_id(&self, bybit_order_id: &str) -> Option<String> {
        self.reverse_id_map
            .read()
            .await
            .get(bybit_order_id)
            .cloned()
    }

    /// Get internal order ID from client order ID
    pub async fn get_internal_id_by_client_id(&self, client_order_id: &str) -> Option<String> {
        self.reverse_client_order_map
            .read()
            .await
            .get(client_order_id)
            .cloned()
    }

    /// Get Bybit order ID from internal order ID
    pub async fn get_bybit_order_id(&self, internal_id: &str) -> Option<String> {
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
                "Bybit API credentials not configured for fill tracking".to_string(),
            ));
        }

        // Check if already running
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(ExecutionError::Internal(
                "Fill tracker already running".to_string(),
            ));
        }

        info!("Starting Bybit fill tracker...");

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
        let reverse_client_order_map = self.reverse_client_order_map.clone();
        let fill_callbacks = self.fill_callbacks.clone();
        let status_callbacks = self.status_callbacks.clone();
        let last_order_states = self.last_order_states.clone();
        let order_submission_times = self.order_submission_times.clone();

        tokio::spawn(async move {
            Self::process_events(
                ws_client,
                fill_tx,
                order_status_tx,
                is_running,
                reverse_id_map,
                reverse_client_order_map,
                fill_callbacks,
                status_callbacks,
                last_order_states,
                order_submission_times,
            )
            .await;
        });

        info!("Bybit fill tracker started successfully");
        Ok(())
    }

    /// Stop the fill tracker
    pub async fn stop(&self) {
        info!("Stopping Bybit fill tracker...");
        *self.is_running.write().await = false;
        self.ws_client.stop().await;
        info!("Bybit fill tracker stopped");
    }

    /// Process WebSocket events
    #[allow(clippy::too_many_arguments)]
    async fn process_events(
        ws_client: Arc<BybitWebSocket>,
        fill_tx: broadcast::Sender<Fill>,
        order_status_tx: broadcast::Sender<(String, OrderStatusEnum)>,
        is_running: Arc<RwLock<bool>>,
        reverse_id_map: Arc<RwLock<HashMap<String, String>>>,
        reverse_client_order_map: Arc<RwLock<HashMap<String, String>>>,
        fill_callbacks: Arc<RwLock<Vec<FillCallback>>>,
        status_callbacks: Arc<RwLock<Vec<OrderStatusCallback>>>,
        last_order_states: Arc<RwLock<HashMap<String, BybitOrderUpdate>>>,
        order_submission_times: Arc<RwLock<HashMap<String, Instant>>>,
    ) {
        let mut event_rx = ws_client.subscribe();

        while *is_running.read().await {
            match event_rx.recv().await {
                Ok(event) => {
                    match event {
                        BybitEvent::OrderUpdate(order_update) => {
                            // Try to resolve internal order ID from bybit order ID
                            let mut internal_id = {
                                let map = reverse_id_map.read().await;
                                map.get(&order_update.order_id).cloned()
                            };

                            // Try client order ID if bybit order ID not found
                            if internal_id.is_none() && !order_update.order_link_id.is_empty() {
                                let map = reverse_client_order_map.read().await;
                                internal_id = map.get(&order_update.order_link_id).cloned();
                            }

                            let order_id = internal_id
                                .clone()
                                .unwrap_or_else(|| order_update.order_id.clone());

                            // Parse order status
                            let status = Self::parse_order_status(&order_update.order_status);

                            debug!(
                                "Order update: {} status={:?} filled={}",
                                order_id, status, order_update.cum_exec_qty
                            );

                            // Check if there's a new fill by comparing with last known state
                            let fill = {
                                let mut states = last_order_states.write().await;
                                let last_state = states.get(&order_update.order_id);

                                let fill = Self::extract_fill_from_update(
                                    &order_update,
                                    last_state,
                                    &order_id,
                                );

                                // Update last known state
                                states.insert(order_update.order_id.clone(), order_update.clone());

                                fill
                            };

                            // Broadcast fill if detected
                            if let Some(fill) = fill {
                                info!(
                                    "Fill detected: order={} qty={} price={} fee={}",
                                    fill.order_id, fill.quantity, fill.price, fill.fee
                                );

                                // Record fill latency in histogram
                                let submission_times = order_submission_times.read().await;
                                if let Some(submission_time) = submission_times.get(&order_id) {
                                    let fill_latency_ms =
                                        submission_time.elapsed().as_secs_f64() * 1000.0;
                                    global_latency_histograms()
                                        .record_order_fill("bybit", fill_latency_ms);
                                    debug!(
                                        "Order {} fill latency: {:.2}ms",
                                        order_id, fill_latency_ms
                                    );
                                }
                                drop(submission_times);

                                let _ = fill_tx.send(fill.clone());

                                // Call fill callbacks
                                let callbacks = fill_callbacks.read().await;
                                for callback in callbacks.iter() {
                                    callback(fill.clone());
                                }
                            }

                            // Broadcast status update
                            let _ = order_status_tx.send((order_id.clone(), status));

                            // Call status callbacks
                            let callbacks = status_callbacks.read().await;
                            for callback in callbacks.iter() {
                                callback(order_id.clone(), status);
                            }
                        }
                        BybitEvent::PositionUpdate(pos) => {
                            debug!(
                                "Position update: {} side={} size={} entry={}",
                                pos.symbol, pos.side, pos.size, pos.entry_price
                            );
                        }
                        BybitEvent::WalletUpdate(wallet) => {
                            debug!(
                                "Wallet update: {} equity={} available={}",
                                wallet.account_type,
                                wallet.total_equity,
                                wallet.total_available_balance
                            );
                        }
                        BybitEvent::Connected => {
                            info!("Bybit fill tracker WebSocket connected");
                        }
                        BybitEvent::Disconnected => {
                            warn!("Bybit fill tracker WebSocket disconnected");
                        }
                        BybitEvent::Error(err) => {
                            error!("Bybit fill tracker WebSocket error: {}", err);
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Bybit fill tracker lagged {} events", n);
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("Bybit fill tracker event channel closed");
                    break;
                }
            }
        }

        info!("Bybit fill tracker event processing stopped");
    }

    /// Parse Bybit order status string to internal enum
    fn parse_order_status(status: &str) -> OrderStatusEnum {
        match status {
            "New" => OrderStatusEnum::New,
            "PartiallyFilled" => OrderStatusEnum::PartiallyFilled,
            "Filled" => OrderStatusEnum::Filled,
            "Cancelled" => OrderStatusEnum::Cancelled,
            "Rejected" => OrderStatusEnum::Rejected,
            "PendingCancel" => OrderStatusEnum::PendingCancel,
            "Untriggered" => OrderStatusEnum::New, // Stop order not triggered yet
            "Triggered" => OrderStatusEnum::Submitted,
            "Deactivated" => OrderStatusEnum::Cancelled,
            _ => {
                warn!("Unknown Bybit order status: {}", status);
                OrderStatusEnum::New
            }
        }
    }

    /// Parse Bybit side string to internal enum
    fn parse_side(side: &str) -> OrderSide {
        match side {
            "Buy" => OrderSide::Buy,
            "Sell" => OrderSide::Sell,
            _ => {
                warn!("Unknown Bybit side: {}, defaulting to Buy", side);
                OrderSide::Buy
            }
        }
    }

    /// Extract fill information from order update by comparing with last state
    fn extract_fill_from_update(
        current: &BybitOrderUpdate,
        last_state: Option<&BybitOrderUpdate>,
        internal_order_id: &str,
    ) -> Option<Fill> {
        let current_filled = Decimal::from_str(&current.cum_exec_qty).unwrap_or(Decimal::ZERO);

        let last_filled = last_state
            .map(|s| Decimal::from_str(&s.cum_exec_qty).unwrap_or(Decimal::ZERO))
            .unwrap_or(Decimal::ZERO);

        // Check if there's new fill quantity
        let fill_qty = current_filled - last_filled;

        if fill_qty <= Decimal::ZERO {
            return None;
        }

        // Calculate fill price
        // Bybit provides average price, so we need to calculate incremental fill price
        let current_value = Decimal::from_str(&current.cum_exec_value).unwrap_or(Decimal::ZERO);
        let last_value = last_state
            .map(|s| Decimal::from_str(&s.cum_exec_value).unwrap_or(Decimal::ZERO))
            .unwrap_or(Decimal::ZERO);

        let fill_value = current_value - last_value;
        let fill_price = if fill_qty > Decimal::ZERO {
            fill_value / fill_qty
        } else {
            Decimal::from_str(&current.avg_price).unwrap_or(Decimal::ZERO)
        };

        // Calculate fee for this fill
        let current_fee = Decimal::from_str(&current.cum_exec_fee).unwrap_or(Decimal::ZERO);
        let last_fee = last_state
            .map(|s| Decimal::from_str(&s.cum_exec_fee).unwrap_or(Decimal::ZERO))
            .unwrap_or(Decimal::ZERO);
        let fill_fee = current_fee - last_fee;

        Some(Fill {
            id: Uuid::new_v4().to_string(),
            order_id: internal_order_id.to_string(),
            quantity: fill_qty,
            price: fill_price,
            fee: fill_fee.abs(),
            fee_currency: "USDT".to_string(), // Bybit typically uses USDT for fees
            side: Self::parse_side(&current.side),
            timestamp: Utc::now(),
            is_maker: false, // Bybit doesn't provide this in order updates
        })
    }

    /// Get count of registered orders
    pub async fn registered_order_count(&self) -> usize {
        self.order_id_map.read().await.len()
    }

    /// Get all registered order IDs
    pub async fn get_registered_orders(&self) -> Vec<String> {
        self.order_id_map.read().await.keys().cloned().collect()
    }

    /// Clear all order mappings
    pub async fn clear_all_mappings(&self) {
        self.order_id_map.write().await.clear();
        self.reverse_id_map.write().await.clear();
        self.client_order_map.write().await.clear();
        self.reverse_client_order_map.write().await.clear();
        self.last_order_states.write().await.clear();
        self.order_submission_times.write().await.clear();
        info!("Cleared all order mappings");
    }
}

// ============================================================================
// Reconciliation Support
// ============================================================================

/// Result of order reconciliation
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    /// Total orders registered locally
    pub total_registered: usize,
    /// Total orders found on exchange
    pub total_exchange: usize,
    /// Orders matched between local and exchange
    pub matched: usize,
    /// Orders registered locally but not on exchange
    pub missing_exchange: Vec<String>,
    /// Orders on exchange but not registered locally
    pub missing_local: Vec<String>,
    /// Status mismatches between local and exchange
    pub status_mismatches: Vec<StatusMismatch>,
}

/// Status mismatch between local and exchange
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
        assert!(config.api_key.is_empty());
        assert!(config.api_secret.is_empty());
        assert!(!config.testnet);
        assert!(config.auto_reconnect);
        assert!(config.subscribe_orders);
    }

    #[test]
    fn test_config_has_credentials() {
        let mut config = FillTrackerConfig::default();
        assert!(!config.has_credentials());

        config.api_key = "test_key".to_string();
        assert!(!config.has_credentials());

        config.api_secret = "test_secret".to_string();
        assert!(config.has_credentials());
    }

    #[tokio::test]
    async fn test_fill_tracker_creation() {
        let config = FillTrackerConfig::default();
        let tracker = BybitFillTracker::new(config);

        assert!(!tracker.is_running().await);
    }

    #[tokio::test]
    async fn test_order_id_mapping() {
        let config = FillTrackerConfig::default();
        let tracker = BybitFillTracker::new(config);

        // Register an order
        tracker
            .register_order("internal-1", "bybit-123", Some("client-456"))
            .await;

        // Verify mappings
        assert_eq!(
            tracker.get_bybit_order_id("internal-1").await,
            Some("bybit-123".to_string())
        );
        assert_eq!(
            tracker.get_internal_id("bybit-123").await,
            Some("internal-1".to_string())
        );
        assert_eq!(
            tracker.get_internal_id_by_client_id("client-456").await,
            Some("internal-1".to_string())
        );

        // Unregister
        tracker.unregister_order("internal-1").await;
        assert_eq!(tracker.get_bybit_order_id("internal-1").await, None);
        assert_eq!(tracker.get_internal_id("bybit-123").await, None);
    }

    #[tokio::test]
    async fn test_subscribe_channels() {
        let config = FillTrackerConfig::default();
        let tracker = BybitFillTracker::new(config);

        // Just verify we can create subscribers
        let _fill_rx = tracker.subscribe_fills();
        let _status_rx = tracker.subscribe_order_status();
    }

    #[test]
    fn test_parse_order_status() {
        assert_eq!(
            BybitFillTracker::parse_order_status("New"),
            OrderStatusEnum::New
        );
        assert_eq!(
            BybitFillTracker::parse_order_status("PartiallyFilled"),
            OrderStatusEnum::PartiallyFilled
        );
        assert_eq!(
            BybitFillTracker::parse_order_status("Filled"),
            OrderStatusEnum::Filled
        );
        assert_eq!(
            BybitFillTracker::parse_order_status("Cancelled"),
            OrderStatusEnum::Cancelled
        );
        assert_eq!(
            BybitFillTracker::parse_order_status("Rejected"),
            OrderStatusEnum::Rejected
        );
    }

    #[test]
    fn test_parse_side() {
        assert_eq!(BybitFillTracker::parse_side("Buy"), OrderSide::Buy);
        assert_eq!(BybitFillTracker::parse_side("Sell"), OrderSide::Sell);
    }

    #[test]
    fn test_reconciliation_result() {
        let result = ReconciliationResult {
            total_registered: 10,
            total_exchange: 8,
            matched: 7,
            missing_exchange: vec!["order-1".to_string()],
            missing_local: vec![],
            status_mismatches: vec![StatusMismatch {
                order_id: "order-2".to_string(),
                local_status: OrderStatusEnum::New,
                exchange_status: OrderStatusEnum::Filled,
            }],
        };

        assert_eq!(result.total_registered, 10);
        assert_eq!(result.missing_exchange.len(), 1);
        assert_eq!(result.status_mismatches.len(), 1);
    }

    #[tokio::test]
    async fn test_clear_all_mappings() {
        let config = FillTrackerConfig::default();
        let tracker = BybitFillTracker::new(config);

        tracker
            .register_order("internal-1", "bybit-123", Some("client-456"))
            .await;
        tracker
            .register_order("internal-2", "bybit-789", None)
            .await;

        assert_eq!(tracker.registered_order_count().await, 2);

        tracker.clear_all_mappings().await;

        assert_eq!(tracker.registered_order_count().await, 0);
    }

    #[test]
    fn test_extract_fill_no_change() {
        let current = BybitOrderUpdate {
            order_id: "order-1".to_string(),
            order_link_id: String::new(),
            symbol: "BTCUSD".to_string(),
            side: "Buy".to_string(),
            order_type: "Limit".to_string(),
            price: "50000".to_string(),
            qty: "1".to_string(),
            cum_exec_qty: "0.5".to_string(),
            cum_exec_value: "25000".to_string(),
            cum_exec_fee: "25".to_string(),
            order_status: "PartiallyFilled".to_string(),
            avg_price: "50000".to_string(),
            created_time: "0".to_string(),
            updated_time: "0".to_string(),
            reject_reason: String::new(),
        };

        let last = current.clone();

        let fill = BybitFillTracker::extract_fill_from_update(&current, Some(&last), "internal-1");
        assert!(fill.is_none());
    }

    #[test]
    fn test_extract_fill_with_change() {
        let last = BybitOrderUpdate {
            order_id: "order-1".to_string(),
            order_link_id: String::new(),
            symbol: "BTCUSD".to_string(),
            side: "Buy".to_string(),
            order_type: "Limit".to_string(),
            price: "50000".to_string(),
            qty: "1".to_string(),
            cum_exec_qty: "0".to_string(),
            cum_exec_value: "0".to_string(),
            cum_exec_fee: "0".to_string(),
            order_status: "New".to_string(),
            avg_price: "0".to_string(),
            created_time: "0".to_string(),
            updated_time: "0".to_string(),
            reject_reason: String::new(),
        };

        let current = BybitOrderUpdate {
            order_id: "order-1".to_string(),
            order_link_id: String::new(),
            symbol: "BTCUSD".to_string(),
            side: "Buy".to_string(),
            order_type: "Limit".to_string(),
            price: "50000".to_string(),
            qty: "1".to_string(),
            cum_exec_qty: "0.5".to_string(),
            cum_exec_value: "25000".to_string(),
            cum_exec_fee: "25".to_string(),
            order_status: "PartiallyFilled".to_string(),
            avg_price: "50000".to_string(),
            created_time: "0".to_string(),
            updated_time: "0".to_string(),
            reject_reason: String::new(),
        };

        let fill = BybitFillTracker::extract_fill_from_update(&current, Some(&last), "internal-1");
        assert!(fill.is_some());

        let fill = fill.unwrap();
        assert_eq!(fill.order_id, "internal-1");
        assert_eq!(fill.quantity, Decimal::from_str("0.5").unwrap());
        assert_eq!(fill.price, Decimal::from(50000));
        assert_eq!(fill.fee, Decimal::from(25));
        assert_eq!(fill.side, OrderSide::Buy);
    }
}
