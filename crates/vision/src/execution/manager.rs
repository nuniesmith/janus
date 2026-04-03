//! Execution Manager
//!
//! This module provides a comprehensive execution manager that orchestrates
//! TWAP/VWAP execution, tracks order lifecycle, and provides smart routing capabilities.
//!
//! # Example
//!
//! ```rust
//! use vision::execution::manager::{ExecutionManager, OrderRequest, ExecutionStrategy};
//! use std::time::Duration;
//!
//! let mut manager = ExecutionManager::new();
//!
//! // Submit an order with TWAP strategy
//! let order_id = manager.submit_order(OrderRequest {
//!     symbol: "AAPL".to_string(),
//!     quantity: 10000.0,
//!     side: vision::execution::analytics::Side::Buy,
//!     strategy: ExecutionStrategy::TWAP {
//!         duration: Duration::from_secs(300),
//!         num_slices: 10,
//!     },
//!     limit_price: Some(150.0),
//! });
//!
//! // Process execution
//! manager.process();
//!
//! // Get order status
//! if let Some(status) = manager.get_order_status(&order_id) {
//!     println!("Order {} is {:?}", order_id, status.state);
//! }
//! ```

use crate::execution::analytics::{ExecutionAnalytics, ExecutionRecord, Side};
use crate::execution::twap::{TWAPConfig, TWAPExecutor};
use crate::execution::vwap::{VWAPConfig, VWAPExecutor, VolumeProfile};

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Unique order identifier
pub type OrderId = String;

/// Execution venue
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Venue {
    NYSE,
    NASDAQ,
    BATS,
    IEX,
    Dark,
    Custom(String),
}

impl Venue {
    pub fn as_str(&self) -> &str {
        match self {
            Venue::NYSE => "NYSE",
            Venue::NASDAQ => "NASDAQ",
            Venue::BATS => "BATS",
            Venue::IEX => "IEX",
            Venue::Dark => "DARK",
            Venue::Custom(s) => s.as_str(),
        }
    }
}

impl ToString for Venue {
    fn to_string(&self) -> String {
        self.as_str().to_string()
    }
}

/// Execution strategy
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Immediate market order
    Market,
    /// Limit order at specified price
    Limit { price: f64 },
    /// Time-weighted average price
    TWAP {
        duration: Duration,
        num_slices: usize,
    },
    /// Volume-weighted average price
    VWAP {
        duration: Duration,
        volume_profile: VolumeProfile,
    },
    /// Percentage of volume
    POV {
        duration: Duration,
        participation_rate: f64,
    },
}

/// Order request
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub quantity: f64,
    pub side: Side,
    pub strategy: ExecutionStrategy,
    pub limit_price: Option<f64>,
    pub venues: Option<Vec<Venue>>,
}

/// Order state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderState {
    Pending,
    Active,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Failed,
}

/// Order status
#[derive(Debug, Clone)]
pub struct OrderStatus {
    pub order_id: OrderId,
    pub symbol: String,
    pub state: OrderState,
    pub total_quantity: f64,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub average_price: Option<f64>,
    pub submitted_at: Instant,
    pub last_updated: Instant,
    pub fills: Vec<Fill>,
}

impl OrderStatus {
    pub fn fill_rate(&self) -> f64 {
        if self.total_quantity > 0.0 {
            self.filled_quantity / self.total_quantity
        } else {
            0.0
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(
            self.state,
            OrderState::Filled | OrderState::Cancelled | OrderState::Rejected | OrderState::Failed
        )
    }
}

/// Order fill
#[derive(Debug, Clone)]
pub struct Fill {
    pub fill_id: String,
    pub quantity: f64,
    pub price: f64,
    pub venue: Venue,
    pub timestamp: Instant,
}

/// Active order being executed
struct ActiveOrder {
    request: OrderRequest,
    status: OrderStatus,
    twap_executor: Option<TWAPExecutor>,
    vwap_executor: Option<VWAPExecutor>,
}

/// Execution manager
pub struct ExecutionManager {
    orders: HashMap<OrderId, ActiveOrder>,
    analytics: ExecutionAnalytics,
    next_order_id: u64,
    benchmark_prices: HashMap<String, f64>,
}

impl ExecutionManager {
    /// Create a new execution manager
    pub fn new() -> Self {
        Self {
            orders: HashMap::new(),
            analytics: ExecutionAnalytics::new(),
            next_order_id: 1,
            benchmark_prices: HashMap::new(),
        }
    }

    /// Submit an order for execution
    pub fn submit_order(&mut self, request: OrderRequest) -> OrderId {
        let order_id = format!("ORD{:08}", self.next_order_id);
        self.next_order_id += 1;

        let now = Instant::now();
        let total_quantity = request.quantity;

        // Store benchmark price (for analytics)
        if !self.benchmark_prices.contains_key(&request.symbol) {
            // In production, this would be the arrival price
            self.benchmark_prices.insert(request.symbol.clone(), 100.0);
        }

        let mut status = OrderStatus {
            order_id: order_id.clone(),
            symbol: request.symbol.clone(),
            state: OrderState::Pending,
            total_quantity,
            filled_quantity: 0.0,
            remaining_quantity: total_quantity,
            average_price: None,
            submitted_at: now,
            last_updated: now,
            fills: Vec::new(),
        };

        // Initialize strategy executor
        let mut twap_executor = None;
        let mut vwap_executor = None;

        match &request.strategy {
            ExecutionStrategy::TWAP {
                duration,
                num_slices,
            } => {
                let config = TWAPConfig {
                    total_quantity,
                    duration: *duration,
                    num_slices: *num_slices,
                    ..Default::default()
                };
                twap_executor = Some(TWAPExecutor::new(config));
                status.state = OrderState::Active;
            }
            ExecutionStrategy::VWAP {
                duration,
                volume_profile,
            } => {
                let config = VWAPConfig {
                    total_quantity,
                    duration: *duration,
                    volume_profile: volume_profile.clone(),
                    ..Default::default()
                };
                vwap_executor = Some(VWAPExecutor::new(config));
                status.state = OrderState::Active;
            }
            ExecutionStrategy::Market | ExecutionStrategy::Limit { .. } => {
                status.state = OrderState::Active;
            }
            ExecutionStrategy::POV { .. } => {
                // POV would use VWAP with adaptive participation
                status.state = OrderState::Active;
            }
        }

        let active_order = ActiveOrder {
            request,
            status,
            twap_executor,
            vwap_executor,
        };

        self.orders.insert(order_id.clone(), active_order);
        order_id
    }

    /// Process all active orders
    pub fn process(&mut self) {
        let order_ids: Vec<OrderId> = self.orders.keys().cloned().collect();

        for order_id in order_ids {
            self.process_order(&order_id);
        }
    }

    /// Process a specific order
    fn process_order(&mut self, order_id: &OrderId) {
        // Collect data before mutable borrow
        let (is_complete, strategy, remaining_quantity) = {
            let order = match self.orders.get(order_id) {
                Some(o) => o,
                None => return,
            };

            if order.status.is_complete() {
                return;
            }

            (
                order.status.is_complete(),
                order.request.strategy.clone(),
                order.status.remaining_quantity,
            )
        };

        if is_complete {
            return;
        }

        // Process based on strategy
        match strategy {
            ExecutionStrategy::Market => {
                // Execute entire order immediately
                self.execute_slice(order_id, remaining_quantity);
            }
            ExecutionStrategy::Limit { price } => {
                // In production, check market price vs limit
                // For now, execute if "market price" is favorable
                if self.should_execute_limit(order_id, price) {
                    self.execute_slice(order_id, remaining_quantity);
                }
            }
            ExecutionStrategy::TWAP { .. } => {
                self.process_twap_order(order_id);
            }
            ExecutionStrategy::VWAP { .. } => {
                self.process_vwap_order(order_id);
            }
            ExecutionStrategy::POV { .. } => {
                self.process_pov_order(order_id);
            }
        }
    }

    /// Process TWAP order
    fn process_twap_order(&mut self, order_id: &OrderId) {
        // Check if next slice is ready and get quantity
        let quantity = {
            let order = match self.orders.get_mut(order_id) {
                Some(o) => o,
                None => return,
            };

            let executor = match &mut order.twap_executor {
                Some(e) => e,
                None => return,
            };

            match executor.next_slice() {
                Some(slice) => slice.quantity,
                None => return,
            }
        };

        // Execute the slice
        self.execute_slice(order_id, quantity);

        // Get price before mutable borrow
        let price = self.get_execution_price(order_id);

        // Mark slice as executed in TWAP executor
        if let Some(order) = self.orders.get_mut(order_id) {
            if let Some(executor) = &mut order.twap_executor {
                executor.mark_executed(quantity, price);

                if executor.is_complete() {
                    order.status.state = OrderState::Filled;
                }
            }
        }
    }

    /// Process VWAP order
    fn process_vwap_order(&mut self, order_id: &OrderId) {
        // Check if next slice is ready and get quantity
        let quantity = {
            let order = match self.orders.get_mut(order_id) {
                Some(o) => o,
                None => return,
            };

            let executor = match &mut order.vwap_executor {
                Some(e) => e,
                None => return,
            };

            match executor.next_slice() {
                Some(slice) => slice.quantity,
                None => return,
            }
        };

        // Execute the slice
        self.execute_slice(order_id, quantity);

        // Get price before mutable borrow
        let price = self.get_execution_price(order_id);

        // Mark slice as executed in VWAP executor
        if let Some(order) = self.orders.get_mut(order_id) {
            if let Some(executor) = &mut order.vwap_executor {
                // In production, get actual market volume
                let market_volume = Some(quantity * 10.0);
                executor.mark_executed(quantity, price, market_volume);

                if executor.is_complete() {
                    order.status.state = OrderState::Filled;
                }
            }
        }
    }

    /// Process POV (Percentage of Volume) order
    fn process_pov_order(&mut self, order_id: &OrderId) {
        // POV would monitor market volume and execute proportionally
        // Simplified implementation
        let order = match self.orders.get_mut(order_id) {
            Some(o) => o,
            None => return,
        };

        if order.status.remaining_quantity > 0.0 {
            let quantity = (order.status.remaining_quantity * 0.1).min(100.0);
            self.execute_slice(order_id, quantity);
        }
    }

    /// Execute a slice of an order
    fn execute_slice(&mut self, order_id: &OrderId, quantity: f64) {
        // Collect data we need before mutable borrow
        let (execute_qty, symbol, side) = {
            let order = match self.orders.get(order_id) {
                Some(o) => o,
                None => return,
            };

            if quantity <= 0.0 || order.status.remaining_quantity <= 0.0 {
                return;
            }

            let execute_qty = quantity.min(order.status.remaining_quantity);
            (
                execute_qty,
                order.request.symbol.clone(),
                order.request.side,
            )
        };

        let execution_price = self.get_execution_price(order_id);
        let venue = self.select_venue_for_quantity(execute_qty);

        // Now get mutable borrow
        let order = match self.orders.get_mut(order_id) {
            Some(o) => o,
            None => return,
        };

        // Create fill
        let fill = Fill {
            fill_id: format!("FILL{:08}", order.status.fills.len() + 1),
            quantity: execute_qty,
            price: execution_price,
            venue: venue.clone(),
            timestamp: Instant::now(),
        };

        // Update order status
        order.status.fills.push(fill);
        order.status.filled_quantity += execute_qty;
        order.status.remaining_quantity -= execute_qty;
        order.status.last_updated = Instant::now();

        // Update state
        if order.status.remaining_quantity <= 0.01 {
            order.status.state = OrderState::Filled;
            order.status.remaining_quantity = 0.0;
        } else if order.status.filled_quantity > 0.0 {
            order.status.state = OrderState::PartiallyFilled;
        }

        // Calculate average price
        let total_value: f64 = order
            .status
            .fills
            .iter()
            .map(|f| f.quantity * f.price)
            .sum();
        order.status.average_price = Some(total_value / order.status.filled_quantity);

        // Record in analytics (outside of order borrow)
        let benchmark_price = self
            .benchmark_prices
            .get(&symbol)
            .copied()
            .unwrap_or(execution_price);

        let record = ExecutionRecord {
            order_id: order_id.clone(),
            quantity: execute_qty,
            execution_price,
            benchmark_price,
            timestamp: Instant::now(),
            side,
            venue: venue.to_string(),
        };

        self.analytics.record_execution(record);
    }

    /// Get execution price (simulated)
    fn get_execution_price(&self, order_id: &OrderId) -> f64 {
        let order = match self.orders.get(order_id) {
            Some(o) => o,
            None => return 100.0,
        };

        // Simulate price with slight randomness
        let base_price = 100.0;
        let noise = ((order_id.len() as f64 * 0.1) % 1.0 - 0.5) * 0.2;

        match order.request.side {
            Side::Buy => base_price + noise.abs(),
            Side::Sell => base_price - noise.abs(),
        }
    }

    /// Check if limit order should execute
    fn should_execute_limit(&self, _order_id: &OrderId, limit_price: f64) -> bool {
        // In production, compare with market price
        // Simplified: execute if limit is reasonable
        limit_price > 0.0
    }

    /// Select best venue for execution based on quantity
    fn select_venue_for_quantity(&self, quantity: f64) -> Venue {
        // Default routing logic based on size
        match quantity {
            q if q > 10000.0 => Venue::Dark, // Large orders to dark pool
            q if q > 1000.0 => Venue::NYSE,  // Medium orders to NYSE
            _ => Venue::NASDAQ,              // Small orders to NASDAQ
        }
    }

    /// Get order status
    pub fn get_order_status(&self, order_id: &OrderId) -> Option<&OrderStatus> {
        self.orders.get(order_id).map(|o| &o.status)
    }

    /// Cancel an order
    pub fn cancel_order(&mut self, order_id: &OrderId) -> bool {
        if let Some(order) = self.orders.get_mut(order_id) {
            if !order.status.is_complete() {
                order.status.state = OrderState::Cancelled;
                order.status.last_updated = Instant::now();
                return true;
            }
        }
        false
    }

    /// Get all active orders
    pub fn active_orders(&self) -> Vec<&OrderStatus> {
        self.orders
            .values()
            .filter(|o| !o.status.is_complete())
            .map(|o| &o.status)
            .collect()
    }

    /// Get analytics
    pub fn analytics(&self) -> &ExecutionAnalytics {
        &self.analytics
    }

    /// Get execution report
    pub fn execution_report(&self) -> crate::execution::analytics::ExecutionReport {
        self.analytics.generate_report()
    }

    /// Set benchmark price for a symbol
    pub fn set_benchmark_price(&mut self, symbol: String, price: f64) {
        self.benchmark_prices.insert(symbol, price);
    }

    /// Get total orders
    pub fn total_orders(&self) -> usize {
        self.orders.len()
    }

    /// Clear completed orders
    pub fn clear_completed(&mut self) {
        self.orders.retain(|_, order| !order.status.is_complete());
    }
}

impl Default for ExecutionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = ExecutionManager::new();
        assert_eq!(manager.total_orders(), 0);
    }

    #[test]
    fn test_submit_market_order() {
        let mut manager = ExecutionManager::new();

        let order_id = manager.submit_order(OrderRequest {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        assert!(manager.get_order_status(&order_id).is_some());
    }

    #[test]
    fn test_market_order_execution() {
        let mut manager = ExecutionManager::new();

        let order_id = manager.submit_order(OrderRequest {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        manager.process();

        let status = manager.get_order_status(&order_id).unwrap();
        assert_eq!(status.state, OrderState::Filled);
        assert_eq!(status.filled_quantity, 100.0);
    }

    #[test]
    fn test_twap_order() {
        let mut manager = ExecutionManager::new();

        let order_id = manager.submit_order(OrderRequest {
            symbol: "AAPL".to_string(),
            quantity: 300.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::TWAP {
                duration: Duration::from_secs(3),
                num_slices: 3,
            },
            limit_price: None,
            venues: None,
        });

        // Process over time
        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(1100));
            manager.process();
        }

        let status = manager.get_order_status(&order_id).unwrap();
        assert!(status.filled_quantity > 0.0);
    }

    #[test]
    fn test_cancel_order() {
        let mut manager = ExecutionManager::new();

        let order_id = manager.submit_order(OrderRequest {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Limit { price: 150.0 },
            limit_price: Some(150.0),
            venues: None,
        });

        assert!(manager.cancel_order(&order_id));

        let status = manager.get_order_status(&order_id).unwrap();
        assert_eq!(status.state, OrderState::Cancelled);
    }

    #[test]
    fn test_active_orders() {
        let mut manager = ExecutionManager::new();

        manager.submit_order(OrderRequest {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Limit { price: 150.0 },
            limit_price: Some(150.0),
            venues: None,
        });

        let active = manager.active_orders();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_venue_selection() {
        let manager = ExecutionManager::new();

        // Large order -> Dark pool
        assert_eq!(manager.select_venue_for_quantity(15000.0), Venue::Dark);

        // Medium order -> NYSE
        assert_eq!(manager.select_venue_for_quantity(5000.0), Venue::NYSE);

        // Small order -> NASDAQ
        assert_eq!(manager.select_venue_for_quantity(100.0), Venue::NASDAQ);
    }

    #[test]
    fn test_execution_analytics() {
        let mut manager = ExecutionManager::new();

        manager.submit_order(OrderRequest {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        manager.process();

        let analytics = manager.analytics();
        assert_eq!(analytics.executions().len(), 1);
    }

    #[test]
    fn test_clear_completed() {
        let mut manager = ExecutionManager::new();

        let _order_id = manager.submit_order(OrderRequest {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        manager.process();
        assert_eq!(manager.total_orders(), 1);

        manager.clear_completed();
        assert_eq!(manager.total_orders(), 0);
    }

    #[test]
    fn test_vwap_order() {
        let mut manager = ExecutionManager::new();

        let _order_id = manager.submit_order(OrderRequest {
            symbol: "AAPL".to_string(),
            quantity: 600.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::VWAP {
                duration: Duration::from_secs(3),
                volume_profile: VolumeProfile::uniform(3),
            },
            limit_price: None,
            venues: None,
        });

        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(1100));
            manager.process();
        }

        if let Some(status) = manager.get_order_status(&_order_id) {
            assert!(status.filled_quantity > 0.0);
        }
    }
}
