//! Order Tracking Module
//!
//! Manages order state transitions and lifecycle tracking with a robust state machine.
//! Provides audit trail of all order state changes for compliance and debugging.

use crate::error::{ExecutionError, Result};
use crate::types::{Order, OrderStatusEnum};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

/// Order state transition record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderStateTransition {
    /// Order ID
    pub order_id: String,

    /// Previous state
    pub from_state: OrderStatusEnum,

    /// New state
    pub to_state: OrderStatusEnum,

    /// Timestamp of transition
    pub timestamp: DateTime<Utc>,

    /// Reason or context for the transition
    pub reason: Option<String>,

    /// Transition sequence number
    pub sequence: u64,
}

/// Order state machine and tracker
pub struct OrderTracker {
    /// Order state history
    state_history: RwLock<HashMap<String, Vec<OrderStateTransition>>>,

    /// Current order states
    current_states: RwLock<HashMap<String, OrderStatusEnum>>,

    /// Transition counters per order
    transition_counters: RwLock<HashMap<String, u64>>,

    /// Total number of orders tracked
    total_orders: RwLock<usize>,
}

impl OrderTracker {
    /// Create a new order tracker
    pub fn new() -> Self {
        Self {
            state_history: RwLock::new(HashMap::new()),
            current_states: RwLock::new(HashMap::new()),
            transition_counters: RwLock::new(HashMap::new()),
            total_orders: RwLock::new(0),
        }
    }

    /// Add a new order to tracking
    pub fn add_order(&self, order: &Order) -> Result<()> {
        let order_id = &order.id;
        let initial_state = order.status;

        debug!(
            order_id = %order_id,
            initial_state = ?initial_state,
            "Adding order to tracking"
        );

        // Check if order already exists
        {
            let states = self.current_states.read();
            if states.contains_key(order_id) {
                return Err(ExecutionError::InvalidOrderState(format!(
                    "Order {} already exists in tracker",
                    order_id
                )));
            }
        }

        // Initialize tracking
        {
            let mut states = self.current_states.write();
            states.insert(order_id.clone(), initial_state);
        }

        {
            let mut history = self.state_history.write();
            history.insert(order_id.clone(), Vec::new());
        }

        {
            let mut counters = self.transition_counters.write();
            counters.insert(order_id.clone(), 0);
        }

        {
            let mut total = self.total_orders.write();
            *total += 1;
        }

        // Record initial state
        self.record_transition(
            order_id,
            initial_state,
            initial_state,
            Some("Order created".to_string()),
        )?;

        Ok(())
    }

    /// Transition order to a new state
    pub fn transition(
        &self,
        order_id: &str,
        from_state: OrderStatusEnum,
        to_state: OrderStatusEnum,
        reason: Option<String>,
    ) -> Result<()> {
        debug!(
            order_id = %order_id,
            from_state = ?from_state,
            to_state = ?to_state,
            reason = ?reason,
            "Order state transition"
        );

        // Validate state transition
        if !self.is_valid_transition(from_state, to_state) {
            warn!(
                order_id = %order_id,
                from_state = ?from_state,
                to_state = ?to_state,
                "Invalid state transition attempted"
            );
            return Err(ExecutionError::InvalidOrderState(format!(
                "Invalid transition from {:?} to {:?}",
                from_state, to_state
            )));
        }

        // Verify current state matches expected from_state
        {
            let states = self.current_states.read();
            if let Some(&current_state) = states.get(order_id) {
                if current_state != from_state {
                    return Err(ExecutionError::InvalidOrderState(format!(
                        "Order {} is in state {:?}, cannot transition from {:?}",
                        order_id, current_state, from_state
                    )));
                }
            } else {
                return Err(ExecutionError::OrderNotFound(order_id.to_string()));
            }
        }

        // Update current state
        {
            let mut states = self.current_states.write();
            states.insert(order_id.to_string(), to_state);
        }

        // Record transition
        self.record_transition(order_id, from_state, to_state, reason)?;

        Ok(())
    }

    /// Record a state transition
    fn record_transition(
        &self,
        order_id: &str,
        from_state: OrderStatusEnum,
        to_state: OrderStatusEnum,
        reason: Option<String>,
    ) -> Result<()> {
        let sequence = {
            let mut counters = self.transition_counters.write();
            let counter = counters
                .get_mut(order_id)
                .ok_or_else(|| ExecutionError::OrderNotFound(order_id.to_string()))?;
            *counter += 1;
            *counter
        };

        let transition = OrderStateTransition {
            order_id: order_id.to_string(),
            from_state,
            to_state,
            timestamp: Utc::now(),
            reason,
            sequence,
        };

        {
            let mut history = self.state_history.write();
            let order_history = history
                .get_mut(order_id)
                .ok_or_else(|| ExecutionError::OrderNotFound(order_id.to_string()))?;
            order_history.push(transition);
        }

        Ok(())
    }

    /// Check if a state transition is valid
    pub fn is_valid_transition(&self, from: OrderStatusEnum, to: OrderStatusEnum) -> bool {
        use OrderStatusEnum::*;

        match (from, to) {
            // Same state is always valid (no-op)
            (s1, s2) if s1 == s2 => true,

            // From New
            (New, Submitted) => true,
            (New, Rejected) => true,
            (New, Cancelled) => true,

            // From Submitted
            (Submitted, PartiallyFilled) => true,
            (Submitted, Filled) => true,
            (Submitted, PendingCancel) => true,
            (Submitted, Cancelled) => true,
            (Submitted, Rejected) => true,

            // From PartiallyFilled
            (PartiallyFilled, Filled) => true,
            (PartiallyFilled, PendingCancel) => true,
            (PartiallyFilled, Cancelled) => true,

            // From PendingCancel
            (PendingCancel, Cancelled) => true,
            (PendingCancel, Filled) => true, // Can fill while cancel is pending
            (PendingCancel, PartiallyFilled) => true,

            // Terminal states cannot transition
            (Filled, _) => false,
            (Cancelled, _) => false,
            (Rejected, _) => false,
            (Expired, _) => false,

            // Any other transition is invalid
            _ => false,
        }
    }

    /// Get current state of an order
    pub fn get_current_state(&self, order_id: &str) -> Result<OrderStatusEnum> {
        let states = self.current_states.read();
        states
            .get(order_id)
            .copied()
            .ok_or_else(|| ExecutionError::OrderNotFound(order_id.to_string()))
    }

    /// Get state history for an order
    pub fn get_state_history(&self, order_id: &str) -> Result<Vec<OrderStateTransition>> {
        let history = self.state_history.read();
        history
            .get(order_id)
            .cloned()
            .ok_or_else(|| ExecutionError::OrderNotFound(order_id.to_string()))
    }

    /// Get all orders in a specific state
    pub fn get_orders_in_state(&self, state: OrderStatusEnum) -> Vec<String> {
        let states = self.current_states.read();
        states
            .iter()
            .filter(|&(_, &s)| s == state)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get all active orders (not in terminal state)
    pub fn get_active_orders(&self) -> Vec<String> {
        let states = self.current_states.read();
        states
            .iter()
            .filter(|&(_, &state)| !Self::is_terminal_state(state))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Check if a state is terminal
    pub fn is_terminal_state(state: OrderStatusEnum) -> bool {
        matches!(
            state,
            OrderStatusEnum::Filled
                | OrderStatusEnum::Cancelled
                | OrderStatusEnum::Rejected
                | OrderStatusEnum::Expired
        )
    }

    /// Get total number of orders tracked
    pub fn total_orders(&self) -> usize {
        *self.total_orders.read()
    }

    /// Get number of transitions for an order
    pub fn get_transition_count(&self, order_id: &str) -> Option<u64> {
        let counters = self.transition_counters.read();
        counters.get(order_id).copied()
    }

    /// Remove an order from tracking (for cleanup)
    pub fn remove_order(&self, order_id: &str) -> Result<()> {
        {
            let mut states = self.current_states.write();
            states.remove(order_id);
        }

        {
            let mut history = self.state_history.write();
            history.remove(order_id);
        }

        {
            let mut counters = self.transition_counters.write();
            counters.remove(order_id);
        }

        Ok(())
    }

    /// Get statistics
    pub fn get_statistics(&self) -> TrackerStatistics {
        let states = self.current_states.read();
        let total = *self.total_orders.read();

        let mut stats = TrackerStatistics {
            total_orders: total,
            active_orders: 0,
            new_orders: 0,
            submitted_orders: 0,
            partially_filled_orders: 0,
            filled_orders: 0,
            cancelled_orders: 0,
            rejected_orders: 0,
            pending_cancel_orders: 0,
        };

        for &state in states.values() {
            if !Self::is_terminal_state(state) {
                stats.active_orders += 1;
            }

            match state {
                OrderStatusEnum::New => stats.new_orders += 1,
                OrderStatusEnum::Submitted => stats.submitted_orders += 1,
                OrderStatusEnum::PartiallyFilled => stats.partially_filled_orders += 1,
                OrderStatusEnum::Filled => stats.filled_orders += 1,
                OrderStatusEnum::Cancelled => stats.cancelled_orders += 1,
                OrderStatusEnum::Rejected => stats.rejected_orders += 1,
                OrderStatusEnum::PendingCancel => stats.pending_cancel_orders += 1,
                OrderStatusEnum::Expired => {}
            }
        }

        stats
    }
}

impl Default for OrderTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerStatistics {
    pub total_orders: usize,
    pub active_orders: usize,
    pub new_orders: usize,
    pub submitted_orders: usize,
    pub partially_filled_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub rejected_orders: usize,
    pub pending_cancel_orders: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExecutionStrategyEnum, OrderSide, OrderTypeEnum, TimeInForceEnum};
    use rust_decimal::Decimal;
    use std::collections::HashMap;

    fn create_test_order(id: &str) -> Order {
        Order {
            id: id.to_string(),
            exchange_order_id: None,
            client_order_id: None,
            signal_id: "signal-1".to_string(),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            order_type: OrderTypeEnum::Market,
            quantity: Decimal::from(1),
            filled_quantity: Decimal::ZERO,
            remaining_quantity: Decimal::from(1),
            price: None,
            stop_price: None,
            average_fill_price: None,
            time_in_force: TimeInForceEnum::Gtc,
            strategy: ExecutionStrategyEnum::Immediate,
            status: OrderStatusEnum::New,
            fills: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_add_order() {
        let tracker = OrderTracker::new();
        let order = create_test_order("order-1");

        tracker.add_order(&order).unwrap();

        assert_eq!(tracker.total_orders(), 1);
        assert_eq!(
            tracker.get_current_state("order-1").unwrap(),
            OrderStatusEnum::New
        );
    }

    #[test]
    fn test_valid_transitions() {
        let tracker = OrderTracker::new();

        assert!(tracker.is_valid_transition(OrderStatusEnum::New, OrderStatusEnum::Submitted));

        assert!(
            tracker
                .is_valid_transition(OrderStatusEnum::Submitted, OrderStatusEnum::PartiallyFilled)
        );

        assert!(
            tracker.is_valid_transition(OrderStatusEnum::PartiallyFilled, OrderStatusEnum::Filled)
        );
    }

    #[test]
    fn test_invalid_transitions() {
        let tracker = OrderTracker::new();

        // Cannot go from Filled to anything
        assert!(!tracker.is_valid_transition(OrderStatusEnum::Filled, OrderStatusEnum::Submitted));

        // Cannot go from Cancelled to anything
        assert!(
            !tracker.is_valid_transition(OrderStatusEnum::Cancelled, OrderStatusEnum::Submitted)
        );

        // Cannot go from New to Filled directly
        assert!(!tracker.is_valid_transition(OrderStatusEnum::New, OrderStatusEnum::Filled));
    }

    #[test]
    fn test_state_transition() {
        let tracker = OrderTracker::new();
        let order = create_test_order("order-1");

        tracker.add_order(&order).unwrap();

        tracker
            .transition(
                "order-1",
                OrderStatusEnum::New,
                OrderStatusEnum::Submitted,
                Some("Sent to exchange".to_string()),
            )
            .unwrap();

        assert_eq!(
            tracker.get_current_state("order-1").unwrap(),
            OrderStatusEnum::Submitted
        );

        let history = tracker.get_state_history("order-1").unwrap();
        assert_eq!(history.len(), 2); // Initial + transition
    }

    #[test]
    fn test_invalid_state_transition() {
        let tracker = OrderTracker::new();
        let order = create_test_order("order-1");

        tracker.add_order(&order).unwrap();

        let result = tracker.transition(
            "order-1",
            OrderStatusEnum::New,
            OrderStatusEnum::Filled,
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_get_orders_in_state() {
        let tracker = OrderTracker::new();

        let order1 = create_test_order("order-1");
        let order2 = create_test_order("order-2");
        let mut order3 = create_test_order("order-3");
        order3.status = OrderStatusEnum::Submitted;

        tracker.add_order(&order1).unwrap();
        tracker.add_order(&order2).unwrap();
        tracker.add_order(&order3).unwrap();

        let new_orders = tracker.get_orders_in_state(OrderStatusEnum::New);
        assert_eq!(new_orders.len(), 2);

        let submitted_orders = tracker.get_orders_in_state(OrderStatusEnum::Submitted);
        assert_eq!(submitted_orders.len(), 1);
    }

    #[test]
    fn test_get_active_orders() {
        let tracker = OrderTracker::new();

        let order1 = create_test_order("order-1");
        let mut order2 = create_test_order("order-2");
        order2.status = OrderStatusEnum::Filled;

        tracker.add_order(&order1).unwrap();
        tracker.add_order(&order2).unwrap();

        let active_orders = tracker.get_active_orders();
        assert_eq!(active_orders.len(), 1);
        assert!(active_orders.contains(&"order-1".to_string()));
    }

    #[test]
    fn test_transition_count() {
        let tracker = OrderTracker::new();
        let order = create_test_order("order-1");

        tracker.add_order(&order).unwrap();

        tracker
            .transition(
                "order-1",
                OrderStatusEnum::New,
                OrderStatusEnum::Submitted,
                None,
            )
            .unwrap();

        tracker
            .transition(
                "order-1",
                OrderStatusEnum::Submitted,
                OrderStatusEnum::Filled,
                None,
            )
            .unwrap();

        assert_eq!(tracker.get_transition_count("order-1"), Some(3)); // Initial + 2 transitions
    }

    #[test]
    fn test_statistics() {
        let tracker = OrderTracker::new();

        let order1 = create_test_order("order-1");
        let mut order2 = create_test_order("order-2");
        order2.status = OrderStatusEnum::Submitted;
        let mut order3 = create_test_order("order-3");
        order3.status = OrderStatusEnum::Filled;

        tracker.add_order(&order1).unwrap();
        tracker.add_order(&order2).unwrap();
        tracker.add_order(&order3).unwrap();

        let stats = tracker.get_statistics();
        assert_eq!(stats.total_orders, 3);
        assert_eq!(stats.active_orders, 2); // New + Submitted
        assert_eq!(stats.new_orders, 1);
        assert_eq!(stats.submitted_orders, 1);
        assert_eq!(stats.filled_orders, 1);
    }

    #[test]
    fn test_remove_order() {
        let tracker = OrderTracker::new();
        let order = create_test_order("order-1");

        tracker.add_order(&order).unwrap();
        assert_eq!(tracker.total_orders(), 1);

        tracker.remove_order("order-1").unwrap();
        assert!(tracker.get_current_state("order-1").is_err());
    }

    #[test]
    fn test_is_terminal_state() {
        assert!(OrderTracker::is_terminal_state(OrderStatusEnum::Filled));
        assert!(OrderTracker::is_terminal_state(OrderStatusEnum::Cancelled));
        assert!(OrderTracker::is_terminal_state(OrderStatusEnum::Rejected));
        assert!(OrderTracker::is_terminal_state(OrderStatusEnum::Expired));

        assert!(!OrderTracker::is_terminal_state(OrderStatusEnum::New));
        assert!(!OrderTracker::is_terminal_state(OrderStatusEnum::Submitted));
        assert!(!OrderTracker::is_terminal_state(
            OrderStatusEnum::PartiallyFilled
        ));
    }
}
