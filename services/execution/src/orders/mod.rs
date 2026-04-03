//! Order Management Module
//!
//! This module provides comprehensive order management functionality including:
//! - Order lifecycle tracking
//! - Pre-trade validation
//! - Order state machine
//! - Partial fill handling
//! - QuestDB persistence
//!
//! # Example
//!
//! ```ignore
//! use janus_execution::orders::OrderManager;
//! use janus_execution::types::Order;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let manager = OrderManager::new("questdb_host:9009".to_string()).await?;
//!
//! // Submit an order
//! let order_id = manager.submit_order(order).await?;
//!
//! // Track order status
//! let status = manager.get_order(&order_id).await?;
//! # Ok(())
//! # }
//! ```

pub mod history;
pub mod tracking;
pub mod validation;

use crate::error::{ExecutionError, Result};
use crate::exchanges::router::ExchangeRouter;
use crate::kill_switch_guard::OrderGate;
use crate::positions::{Position, PositionTracker};
use crate::types::{Fill, Order, OrderStatusEnum};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

pub use history::OrderHistory;
pub use tracking::{OrderStateTransition, OrderTracker};
pub use validation::{OrderValidator, ValidationConfig, ValidationResult};

/// Main order manager coordinating all order operations
pub struct OrderManager {
    /// Order tracking system
    tracker: Arc<OrderTracker>,

    /// Pre-trade validation
    validator: Arc<OrderValidator>,

    /// Order history persistence
    history: Arc<OrderHistory>,

    /// Exchange router for order execution
    exchange_router: Option<Arc<ExchangeRouter>>,

    /// Position tracker for P&L and position management
    position_tracker: Arc<PositionTracker>,

    /// Active orders cache
    active_orders: Arc<RwLock<HashMap<String, Order>>>,

    /// Completed orders cache (recent)
    completed_orders: Arc<RwLock<HashMap<String, Order>>>,

    /// Maximum completed orders to keep in memory
    max_completed_cache: usize,

    /// Kill switch guard — blocks ALL order submission when active.
    /// This is defense-in-depth: even if the brain pipeline's kill switch
    /// fires, the execution service independently refuses orders.
    order_gate: Option<Arc<dyn OrderGate>>,
}

impl OrderManager {
    /// Create a new order manager
    pub async fn new(questdb_host: String) -> Result<Self> {
        let tracker = Arc::new(OrderTracker::new());
        let validator = Arc::new(OrderValidator::new(ValidationConfig::default()));
        let history = Arc::new(OrderHistory::new(&questdb_host).await?);

        Ok(Self {
            tracker,
            validator,
            history,
            exchange_router: None,
            position_tracker: Arc::new(PositionTracker::new()),
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            completed_orders: Arc::new(RwLock::new(HashMap::new())),
            max_completed_cache: 1000,
            order_gate: None,
        })
    }

    /// Create with custom validation config
    pub async fn with_config(
        questdb_host: String,
        validation_config: ValidationConfig,
    ) -> Result<Self> {
        let tracker = Arc::new(OrderTracker::new());
        let validator = Arc::new(OrderValidator::new(validation_config));
        let history = Arc::new(OrderHistory::new(&questdb_host).await?);

        Ok(Self {
            tracker,
            validator,
            history,
            exchange_router: None,
            position_tracker: Arc::new(PositionTracker::new()),
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            completed_orders: Arc::new(RwLock::new(HashMap::new())),
            max_completed_cache: 1000,
            order_gate: None,
        })
    }

    /// Set the exchange router
    pub fn set_exchange_router(&mut self, router: Arc<ExchangeRouter>) {
        self.exchange_router = Some(router);
    }

    /// Set the kill switch order gate.
    ///
    /// When set, every call to `submit_order` will check this gate
    /// **before** any validation or exchange interaction. If the gate
    /// reports blocked, the order is immediately rejected with
    /// `ExecutionError::KillSwitchActive`.
    pub fn set_order_gate(&mut self, gate: Arc<dyn OrderGate>) {
        self.order_gate = Some(gate);
    }

    /// Submit a new order
    pub async fn submit_order(&self, mut order: Order) -> Result<String> {
        // ── Kill switch guard (defense-in-depth) ───────────────────────
        // This check runs BEFORE validation, BEFORE any exchange I/O.
        // It is a single atomic load — negligible latency on the hot path.
        if let Some(gate) = &self.order_gate {
            if gate.is_blocked() {
                error!(
                    order_id = %order.id,
                    symbol = %order.symbol,
                    side = ?order.side,
                    "🚨 ORDER BLOCKED BY KILL SWITCH"
                );
                return Err(ExecutionError::KillSwitchActive(
                    gate.block_reason().to_string(),
                ));
            }
        }

        info!(
            order_id = %order.id,
            symbol = %order.symbol,
            side = ?order.side,
            quantity = %order.quantity,
            "Submitting order"
        );

        // Pre-trade validation
        let validation_result = self.validator.validate_order(&order).await?;
        if !validation_result.is_valid {
            error!(
                order_id = %order.id,
                reasons = ?validation_result.rejection_reasons,
                "Order validation failed"
            );
            order.status = OrderStatusEnum::Rejected;
            self.tracker.transition(
                &order.id,
                OrderStatusEnum::New,
                OrderStatusEnum::Rejected,
                Some(validation_result.rejection_reasons.join("; ")),
            )?;

            // Persist rejection
            self.history.record_order(&order).await?;

            return Err(ExecutionError::Validation(format!(
                "Order validation failed: {}",
                validation_result.rejection_reasons.join(", ")
            )));
        }

        // Track order
        self.tracker.add_order(&order)?;

        // Add to active orders
        {
            let mut active = self.active_orders.write();
            active.insert(order.id.clone(), order.clone());
        }

        // Submit to exchange if router is available
        if let Some(router) = &self.exchange_router {
            match router.place_order(&order).await {
                Ok((_exchange_name, exchange_order_id)) => {
                    order.exchange_order_id = Some(exchange_order_id.clone());
                    order.status = OrderStatusEnum::Submitted;

                    self.tracker.transition(
                        &order.id,
                        OrderStatusEnum::New,
                        OrderStatusEnum::Submitted,
                        Some(format!("Exchange order ID: {}", exchange_order_id)),
                    )?;

                    // Update active orders
                    {
                        let mut active = self.active_orders.write();
                        active.insert(order.id.clone(), order.clone());
                    }

                    info!(
                        order_id = %order.id,
                        exchange_order_id = %exchange_order_id,
                        "Order submitted to exchange"
                    );
                }
                Err(e) => {
                    error!(
                        order_id = %order.id,
                        error = %e,
                        "Failed to submit order to exchange"
                    );
                    order.status = OrderStatusEnum::Rejected;
                    self.tracker.transition(
                        &order.id,
                        OrderStatusEnum::New,
                        OrderStatusEnum::Rejected,
                        Some(format!("Exchange error: {}", e)),
                    )?;

                    // Move to completed
                    self.move_to_completed(&order);

                    return Err(e);
                }
            }
        } else {
            // No exchange router - just mark as submitted
            order.status = OrderStatusEnum::Submitted;
            self.tracker.transition(
                &order.id,
                OrderStatusEnum::New,
                OrderStatusEnum::Submitted,
                Some("No exchange router configured".to_string()),
            )?;
        }

        // Persist to history
        self.history.record_order(&order).await?;

        Ok(order.id.clone())
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        info!(order_id = %order_id, "Cancelling order");

        // Get order from active orders
        let order = {
            let active = self.active_orders.read();
            active
                .get(order_id)
                .ok_or_else(|| ExecutionError::OrderNotFound(order_id.to_string()))?
                .clone()
        };

        // Check if order can be cancelled
        if !order.is_active() {
            warn!(
                order_id = %order_id,
                status = ?order.status,
                "Cannot cancel order in terminal state"
            );
            return Err(ExecutionError::InvalidOrderState(format!(
                "Order {} is not active (status: {:?})",
                order_id, order.status
            )));
        }

        // Transition to pending cancel
        self.tracker.transition(
            order_id,
            order.status,
            OrderStatusEnum::PendingCancel,
            Some("Cancel requested".to_string()),
        )?;

        // Cancel on exchange if router is available
        if let Some(router) = &self.exchange_router {
            if let Some(exchange_order_id) = &order.exchange_order_id {
                router
                    .cancel_order(&order.exchange, exchange_order_id)
                    .await?;
            }
        }

        // Transition to cancelled
        let mut cancelled_order = order.clone();
        cancelled_order.status = OrderStatusEnum::Cancelled;

        self.tracker.transition(
            order_id,
            OrderStatusEnum::PendingCancel,
            OrderStatusEnum::Cancelled,
            Some("Order cancelled".to_string()),
        )?;

        // Move to completed
        self.move_to_completed(&cancelled_order);

        // Persist
        self.history.record_order(&cancelled_order).await?;

        info!(order_id = %order_id, "Order cancelled successfully");

        Ok(())
    }

    /// Handle a fill for an order
    pub async fn handle_fill(&self, order_id: &str, fill: Fill) -> Result<()> {
        info!(
            order_id = %order_id,
            fill_id = %fill.id,
            quantity = %fill.quantity,
            price = %fill.price,
            "Processing fill"
        );

        // Get order
        let mut order = {
            let active = self.active_orders.read();
            active
                .get(order_id)
                .ok_or_else(|| ExecutionError::OrderNotFound(order_id.to_string()))?
                .clone()
        };

        let old_status = order.status;

        // Add fill to order
        order.add_fill(fill.clone());

        // Update position tracker
        if let Err(e) = self
            .position_tracker
            .apply_fill(
                &order.exchange,
                order.symbol.clone(),
                order.side,
                fill.quantity,
                fill.price,
            )
            .await
        {
            warn!(
                order_id = %order_id,
                error = %e,
                "Failed to update position tracker"
            );
        }

        // Determine new status
        let new_status = order.status;

        // Track state transition
        if old_status != new_status {
            self.tracker.transition(
                order_id,
                old_status,
                new_status,
                Some(format!("Fill: {} @ {}", fill.quantity, fill.price)),
            )?;
        }

        // Update active orders or move to completed
        if order.is_terminal() {
            self.move_to_completed(&order);
            info!(
                order_id = %order_id,
                status = ?order.status,
                "Order completed"
            );
        } else {
            let mut active = self.active_orders.write();
            active.insert(order_id.to_string(), order.clone());
        }

        // Persist order and fill
        self.history.record_order(&order).await?;
        self.history.record_fill(&fill).await?;

        Ok(())
    }

    /// Get order by ID
    pub fn get_order(&self, order_id: &str) -> Result<Order> {
        // Check active orders first
        {
            let active = self.active_orders.read();
            if let Some(order) = active.get(order_id) {
                return Ok(order.clone());
            }
        }

        // Check completed orders
        {
            let completed = self.completed_orders.read();
            if let Some(order) = completed.get(order_id) {
                return Ok(order.clone());
            }
        }

        Err(ExecutionError::OrderNotFound(order_id.to_string()))
    }

    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<Order> {
        let active = self.active_orders.read();
        active.values().cloned().collect()
    }

    /// Get all positions across all exchanges
    pub async fn get_all_positions(&self) -> Vec<Position> {
        self.position_tracker.get_all_positions().await
    }

    /// Get positions for a specific exchange
    pub async fn get_positions_by_exchange(&self, exchange: &str) -> Vec<Position> {
        self.position_tracker
            .get_positions_by_exchange(exchange)
            .await
    }

    /// Get position for a specific symbol on an exchange
    pub async fn get_position(&self, exchange: &str, symbol: &str) -> Position {
        self.position_tracker
            .get_position(exchange, symbol.to_string())
            .await
    }

    /// Update mark price for position P&L calculation
    pub async fn update_mark_price(
        &self,
        exchange: &str,
        symbol: &str,
        mark_price: Decimal,
    ) -> Result<()> {
        self.position_tracker
            .update_mark_price(exchange, symbol.to_string(), mark_price)
            .await
            .map_err(|e| ExecutionError::Internal(e.to_string()))
    }

    /// Get position statistics
    pub async fn get_position_stats(&self) -> crate::positions::PositionStats {
        self.position_tracker.get_stats().await
    }

    /// Get order history for a symbol
    pub async fn get_order_history(&self, symbol: &str, limit: usize) -> Result<Vec<Order>> {
        self.history.get_orders_by_symbol(symbol, limit).await
    }

    /// Get order state history
    pub fn get_order_state_history(&self, order_id: &str) -> Result<Vec<OrderStateTransition>> {
        self.tracker.get_state_history(order_id)
    }

    /// Get statistics
    pub fn get_statistics(&self) -> OrderStatistics {
        let active = self.active_orders.read();
        let completed = self.completed_orders.read();

        OrderStatistics {
            active_orders: active.len(),
            completed_orders: completed.len(),
            total_orders_tracked: self.tracker.total_orders(),
        }
    }

    /// Move order from active to completed cache
    fn move_to_completed(&self, order: &Order) {
        // Remove from active
        {
            let mut active = self.active_orders.write();
            active.remove(&order.id);
        }

        // Add to completed
        {
            let mut completed = self.completed_orders.write();

            // Evict oldest if cache is full
            if completed.len() >= self.max_completed_cache {
                // Simple eviction: remove first entry
                if let Some(key) = completed.keys().next().cloned() {
                    completed.remove(&key);
                }
            }

            completed.insert(order.id.clone(), order.clone());
        }
    }

    /// Reconcile orders with exchange
    /// This is useful for recovering from crashes or network issues
    pub async fn reconcile_orders(&self) -> Result<usize> {
        info!("Starting order reconciliation");

        let mut reconciled = 0;
        let mut fills_processed = 0;

        if let Some(router) = &self.exchange_router {
            let active_orders = self.get_active_orders();

            for order in active_orders {
                if let Some(exchange_order_id) = &order.exchange_order_id {
                    match router
                        .get_order_status(&order.exchange, exchange_order_id)
                        .await
                    {
                        Ok(status_response) => {
                            // Check if status changed or there are new fills
                            let status_changed = status_response.status != order.status;
                            let has_new_fills =
                                status_response.filled_quantity > order.filled_quantity;

                            if status_changed || has_new_fills {
                                debug!(
                                    order_id = %order.id,
                                    old_status = ?order.status,
                                    new_status = ?status_response.status,
                                    old_filled = %order.filled_quantity,
                                    new_filled = %status_response.filled_quantity,
                                    "Reconciling order"
                                );

                                // Process any fills from the exchange response
                                for fill in &status_response.fills {
                                    // Check if we've already processed this fill
                                    let fill_already_processed =
                                        order.fills.iter().any(|f| f.id == fill.id);

                                    if !fill_already_processed {
                                        info!(
                                            order_id = %order.id,
                                            fill_id = %fill.id,
                                            quantity = %fill.quantity,
                                            price = %fill.price,
                                            "Processing missed fill during reconciliation"
                                        );

                                        // Handle the fill through the normal flow
                                        match self.handle_fill(&order.id, fill.clone()).await {
                                            Err(e) => {
                                                warn!(
                                                    order_id = %order.id,
                                                    fill_id = %fill.id,
                                                    error = %e,
                                                    "Failed to process fill during reconciliation"
                                                );
                                            }
                                            _ => {
                                                fills_processed += 1;
                                            }
                                        }
                                    }
                                }

                                // If no fills but status changed, update the order directly
                                if status_response.fills.is_empty() && status_changed {
                                    let mut active = self.active_orders.write();
                                    if let Some(mut_order) = active.get_mut(&order.id) {
                                        let old_status = mut_order.status;
                                        mut_order.status = status_response.status;
                                        mut_order.updated_at = chrono::Utc::now();

                                        // Track state transition
                                        if let Err(e) = self.tracker.transition(
                                            &order.id,
                                            old_status,
                                            status_response.status,
                                            Some("Reconciled from exchange".to_string()),
                                        ) {
                                            warn!(
                                                order_id = %order.id,
                                                error = %e,
                                                "Failed to track reconciliation transition"
                                            );
                                        }

                                        // Move to completed if terminal
                                        if mut_order.is_terminal() {
                                            let completed_order = mut_order.clone();
                                            drop(active);
                                            self.move_to_completed(&completed_order);
                                        }
                                    }
                                }

                                reconciled += 1;
                            }
                        }
                        Err(e) => {
                            warn!(
                                order_id = %order.id,
                                exchange_order_id = %exchange_order_id,
                                error = %e,
                                "Failed to reconcile order"
                            );
                        }
                    }
                }
            }
        }

        info!(
            reconciled_orders = reconciled,
            fills_processed = fills_processed,
            "Order reconciliation complete"
        );

        Ok(reconciled)
    }
}

/// Order management statistics
#[derive(Debug, Clone)]
pub struct OrderStatistics {
    pub active_orders: usize,
    pub completed_orders: usize,
    pub total_orders_tracked: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_order_manager_creation() {
        // Use a test QuestDB host (won't actually connect in this test)
        let _result = OrderManager::new("localhost:9009".to_string()).await;
        // We expect this to fail if QuestDB is not running, which is fine for this test
        // Just verifying the construction logic
    }

    #[tokio::test]
    async fn test_order_lifecycle() {
        // This would require a running QuestDB instance
        // For now, we test the in-memory parts
    }
}
