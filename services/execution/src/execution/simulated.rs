//! Simulated execution engine for backtesting and paper trading

use crate::config::SimulationConfig;
use crate::error::{ExecutionError, Result};
use crate::types::{Fill, Order, OrderSide, OrderStatusEnum, Position, PositionSide};
use chrono::Utc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{Duration, sleep};
use tracing::{debug, info};
use uuid::Uuid;

/// Simulated execution engine
pub struct SimulatedExecutor {
    /// Simulation configuration
    config: SimulationConfig,

    /// Current account balance
    balance: Arc<RwLock<Decimal>>,

    /// Active orders
    orders: Arc<RwLock<HashMap<String, Order>>>,

    /// Active positions
    positions: Arc<RwLock<HashMap<String, Position>>>,

    /// Market prices (symbol -> price)
    market_prices: Arc<RwLock<HashMap<String, Decimal>>>,

    /// Daily P&L tracker
    daily_pnl: Arc<RwLock<Decimal>>,
}

impl SimulatedExecutor {
    /// Create a new simulated executor
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            balance: Arc::new(RwLock::new(
                Decimal::try_from(config.initial_balance).unwrap(),
            )),
            config,
            orders: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            market_prices: Arc::new(RwLock::new(HashMap::new())),
            daily_pnl: Arc::new(RwLock::new(Decimal::ZERO)),
        }
    }

    /// Submit an order for simulated execution
    pub async fn submit_order(&self, mut order: Order) -> Result<String> {
        info!(
            "Simulated execution: {} {} {} @ {:?}",
            order.side, order.quantity, order.symbol, order.price
        );

        // Validate order
        self.validate_order(&order)?;

        // Update order status
        order.status = OrderStatusEnum::Submitted;
        order.updated_at = Utc::now();

        let order_id = order.id.clone();

        // Store order
        self.orders.write().insert(order_id.clone(), order.clone());

        // Simulate order processing delay
        if self.config.fill_delay_ms > 0 {
            let delay = Duration::from_millis(self.config.fill_delay_ms);
            sleep(delay).await;
        }

        // Execute the order
        self.execute_order(order_id.clone()).await?;

        Ok(order_id)
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let mut orders = self.orders.write();

        if let Some(order) = orders.get_mut(order_id) {
            if order.is_terminal() {
                return Err(ExecutionError::InvalidStateTransition(
                    "Cannot cancel terminal order".to_string(),
                ));
            }

            order.status = OrderStatusEnum::Cancelled;
            order.updated_at = Utc::now();
            info!("Cancelled order: {}", order_id);
            Ok(())
        } else {
            Err(ExecutionError::OrderNotFound(order_id.to_string()))
        }
    }

    /// Get order status
    pub fn get_order(&self, order_id: &str) -> Result<Order> {
        self.orders
            .read()
            .get(order_id)
            .cloned()
            .ok_or_else(|| ExecutionError::OrderNotFound(order_id.to_string()))
    }

    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<Order> {
        self.orders
            .read()
            .values()
            .filter(|o| o.is_active())
            .cloned()
            .collect()
    }

    /// Get position for a symbol
    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        self.positions.read().get(symbol).cloned()
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> Vec<Position> {
        self.positions.read().values().cloned().collect()
    }

    /// Get current account balance
    pub fn get_balance(&self) -> Decimal {
        *self.balance.read()
    }

    /// Get unrealized P&L
    pub fn get_unrealized_pnl(&self) -> Decimal {
        let positions = self.positions.read();
        positions.values().map(|p| p.unrealized_pnl).sum()
    }

    /// Get daily realized P&L
    pub fn get_daily_pnl(&self) -> Decimal {
        *self.daily_pnl.read()
    }

    /// Update market price for a symbol
    pub fn update_market_price(&self, symbol: String, price: Decimal) {
        self.market_prices.write().insert(symbol.clone(), price);

        // Update position marks
        if let Some(position) = self.positions.write().get_mut(&symbol) {
            position.current_price = price;
            position.calculate_unrealized_pnl();
        }
    }

    /// Execute an order (internal)
    async fn execute_order(&self, order_id: String) -> Result<()> {
        let order = {
            let orders = self.orders.read();
            orders
                .get(&order_id)
                .cloned()
                .ok_or_else(|| ExecutionError::OrderNotFound(order_id.clone()))?
        };

        // Get market price
        let market_price = self
            .market_prices
            .read()
            .get(&order.symbol)
            .cloned()
            .unwrap_or_else(|| {
                // Use order price if available, otherwise a default
                order.price.unwrap_or(Decimal::from(50000))
            });

        // Calculate fill price with slippage
        let is_buy = matches!(order.side, OrderSide::Buy);
        let fill_price = self.config.calculate_slippage(market_price, is_buy);

        // Check if we have sufficient balance
        let trade_value = fill_price * order.quantity;
        let fee = self.config.calculate_fee(trade_value);
        let total_cost = trade_value + fee;

        if is_buy {
            let balance = *self.balance.read();
            if total_cost > balance {
                // Reject order
                let mut orders = self.orders.write();
                if let Some(order) = orders.get_mut(&order_id) {
                    order.status = OrderStatusEnum::Rejected;
                    order.updated_at = Utc::now();
                }
                return Err(ExecutionError::InsufficientBalance {
                    required: total_cost.to_f64().unwrap_or(0.0),
                    available: balance.to_f64().unwrap_or(0.0),
                });
            }
        }

        // Create fill
        let fill = Fill {
            id: Uuid::new_v4().to_string(),
            order_id: order_id.clone(),
            quantity: order.quantity,
            price: fill_price,
            fee,
            fee_currency: "USDT".to_string(),
            side: order.side,
            timestamp: Utc::now(),
            is_maker: false, // Simulated fills are always taker
        };

        debug!(
            "Simulated fill: {} {} @ {} (fee: {})",
            fill.quantity, order.symbol, fill.price, fill.fee
        );

        // Update order with fill
        {
            let mut orders = self.orders.write();
            if let Some(order) = orders.get_mut(&order_id) {
                order.add_fill(fill.clone());
            }
        }

        // Update position
        self.update_position(&order.symbol, &fill, &order.exchange)?;

        // Update balance
        match fill.side {
            OrderSide::Buy => {
                *self.balance.write() -= total_cost;
            }
            OrderSide::Sell => {
                *self.balance.write() += trade_value - fee;
            }
        }

        info!(
            "Order {} filled: {} {} @ {}",
            order_id, fill.quantity, order.symbol, fill.price
        );

        Ok(())
    }

    /// Update position after a fill
    fn update_position(&self, symbol: &str, fill: &Fill, exchange: &str) -> Result<()> {
        let mut positions = self.positions.write();

        let position = positions.entry(symbol.to_string()).or_insert_with(|| {
            let side = match fill.side {
                OrderSide::Buy => PositionSide::Long,
                OrderSide::Sell => PositionSide::Short,
            };

            Position {
                symbol: symbol.to_string(),
                exchange: exchange.to_string(),
                side,
                quantity: Decimal::ZERO,
                average_entry_price: Decimal::ZERO,
                current_price: fill.price,
                unrealized_pnl: Decimal::ZERO,
                realized_pnl: Decimal::ZERO,
                margin_used: Decimal::ZERO,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            }
        });

        // Store previous realized P&L
        let prev_realized_pnl = position.realized_pnl;

        // Update position with fill
        position.update_with_fill(fill);

        // Track realized P&L
        let realized_pnl_change = position.realized_pnl - prev_realized_pnl;
        if realized_pnl_change != Decimal::ZERO {
            *self.daily_pnl.write() += realized_pnl_change;
            info!(
                "Realized P&L from {}: {} (total daily: {})",
                symbol,
                realized_pnl_change,
                *self.daily_pnl.read()
            );
        }

        // Remove position if closed
        if position.quantity == Decimal::ZERO {
            positions.remove(symbol);
            info!("Position {} closed", symbol);
        }

        Ok(())
    }

    /// Validate order before execution
    fn validate_order(&self, order: &Order) -> Result<()> {
        // Check quantity
        if order.quantity <= Decimal::ZERO {
            return Err(ExecutionError::OrderValidation(
                "Order quantity must be positive".to_string(),
            ));
        }

        // Check price for limit orders
        if let Some(price) = order.price {
            if price <= Decimal::ZERO {
                return Err(ExecutionError::OrderValidation(
                    "Order price must be positive".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Reset daily P&L (call at start of each trading day)
    pub fn reset_daily_pnl(&self) {
        *self.daily_pnl.write() = Decimal::ZERO;
        info!("Daily P&L reset");
    }

    /// Get statistics
    pub fn get_stats(&self) -> SimulatorStats {
        let balance = *self.balance.read();
        let positions = self.positions.read();
        let orders = self.orders.read();

        let unrealized_pnl: Decimal = positions.values().map(|p| p.unrealized_pnl).sum();
        let realized_pnl: Decimal = positions.values().map(|p| p.realized_pnl).sum();

        let total_orders = orders.len();
        let filled_orders = orders
            .values()
            .filter(|o| o.status == OrderStatusEnum::Filled)
            .count();
        let cancelled_orders = orders
            .values()
            .filter(|o| o.status == OrderStatusEnum::Cancelled)
            .count();

        SimulatorStats {
            balance,
            unrealized_pnl,
            realized_pnl,
            daily_pnl: *self.daily_pnl.read(),
            total_value: balance + unrealized_pnl,
            open_positions: positions.len(),
            total_orders,
            filled_orders,
            cancelled_orders,
        }
    }
}

/// Simulator statistics
#[derive(Debug, Clone)]
pub struct SimulatorStats {
    pub balance: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub daily_pnl: Decimal,
    pub total_value: Decimal,
    pub open_positions: usize,
    pub total_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OrderTypeEnum;

    fn create_test_executor() -> SimulatedExecutor {
        let config = SimulationConfig {
            initial_balance: 100000.0,
            slippage_bps: 5,
            fee_bps: 10,
            fill_delay_ms: 0, // No delay in tests
            enable_slippage: true,
        };

        SimulatedExecutor::new(config)
    }

    fn create_test_order(symbol: &str, side: OrderSide, quantity: f64) -> Order {
        Order::new(
            "signal123".to_string(),
            symbol.to_string(),
            "simulated".to_string(),
            side,
            OrderTypeEnum::Market,
            Decimal::try_from(quantity).unwrap(),
        )
    }

    #[tokio::test]
    async fn test_submit_buy_order() {
        let executor = create_test_executor();
        executor.update_market_price("BTCUSD".to_string(), Decimal::from(50000));

        let order = create_test_order("BTCUSD", OrderSide::Buy, 1.0);
        let order_id = executor.submit_order(order).await.unwrap();

        let filled_order = executor.get_order(&order_id).unwrap();
        assert_eq!(filled_order.status, OrderStatusEnum::Filled);
        assert_eq!(filled_order.filled_quantity, Decimal::from(1));

        // Check position created
        let position = executor.get_position("BTCUSD").unwrap();
        assert_eq!(position.side, PositionSide::Long);
        assert_eq!(position.quantity, Decimal::from(1));
    }

    #[tokio::test]
    async fn test_submit_sell_order() {
        let executor = create_test_executor();
        executor.update_market_price("BTCUSD".to_string(), Decimal::from(50000));

        let order = create_test_order("BTCUSD", OrderSide::Sell, 1.0);
        let order_id = executor.submit_order(order).await.unwrap();

        let filled_order = executor.get_order(&order_id).unwrap();
        assert_eq!(filled_order.status, OrderStatusEnum::Filled);

        let position = executor.get_position("BTCUSD").unwrap();
        assert_eq!(position.side, PositionSide::Short);
    }

    #[tokio::test]
    async fn test_insufficient_balance() {
        let executor = create_test_executor();
        executor.update_market_price("BTCUSD".to_string(), Decimal::from(50000));

        // Try to buy more than balance allows
        let order = create_test_order("BTCUSD", OrderSide::Buy, 3.0);
        let result = executor.submit_order(order).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutionError::InsufficientBalance { .. }
        ));
    }

    #[tokio::test]
    async fn test_cancel_order() {
        let executor = create_test_executor();

        let mut order = create_test_order("BTCUSD", OrderSide::Buy, 1.0);
        order.status = OrderStatusEnum::Submitted;
        let order_id = order.id.clone();
        executor.orders.write().insert(order_id.clone(), order);

        executor.cancel_order(&order_id).await.unwrap();

        let cancelled_order = executor.get_order(&order_id).unwrap();
        assert_eq!(cancelled_order.status, OrderStatusEnum::Cancelled);
    }

    #[tokio::test]
    async fn test_position_pnl() {
        let executor = create_test_executor();
        executor.update_market_price("BTCUSD".to_string(), Decimal::from(50000));

        // Buy at 50000
        let buy_order = create_test_order("BTCUSD", OrderSide::Buy, 1.0);
        executor.submit_order(buy_order).await.unwrap();

        // Update price to 51000
        executor.update_market_price("BTCUSD".to_string(), Decimal::from(51000));

        let position = executor.get_position("BTCUSD").unwrap();
        assert!(position.unrealized_pnl > Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_close_position() {
        let executor = create_test_executor();
        executor.update_market_price("BTCUSD".to_string(), Decimal::from(50000));

        // Buy
        let buy_order = create_test_order("BTCUSD", OrderSide::Buy, 1.0);
        executor.submit_order(buy_order).await.unwrap();

        // Sell (close)
        let sell_order = create_test_order("BTCUSD", OrderSide::Sell, 1.0);
        executor.submit_order(sell_order).await.unwrap();

        // Position should be closed
        assert!(executor.get_position("BTCUSD").is_none());
    }

    #[tokio::test]
    async fn test_stats() {
        let executor = create_test_executor();
        executor.update_market_price("BTCUSD".to_string(), Decimal::from(50000));

        let order = create_test_order("BTCUSD", OrderSide::Buy, 1.0);
        executor.submit_order(order).await.unwrap();

        let stats = executor.get_stats();
        assert_eq!(stats.filled_orders, 1);
        assert_eq!(stats.open_positions, 1);
        assert!(stats.balance < Decimal::from(100000)); // Reduced by purchase
    }
}
