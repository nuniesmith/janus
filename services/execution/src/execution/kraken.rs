//! Kraken Live Execution Engine
//!
//! This module provides live trading execution on Kraken exchange
//! using the authenticated REST API.
//!
//! # Features
//!
//! - Market orders
//! - Limit orders
//! - Stop-loss orders
//! - Take-profit orders
//! - Order cancellation
//! - Position tracking
//! - Balance management
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::execution::kraken::{KrakenExecutionEngine, KrakenExecutionConfig};
//!
//! let config = KrakenExecutionConfig::from_env();
//! let engine = KrakenExecutionEngine::new(config).await?;
//!
//! // Submit an order
//! let order_id = engine.submit_order(order).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::config::ExecutionMode;
use crate::error::{ExecutionError, Result};
use crate::exchanges::kraken::rest::{KrakenRestClient, KrakenRestConfig};
use crate::types::{
    Fill, Order, OrderSide, OrderStatusEnum, OrderTypeEnum, Position, PositionSide,
};

/// Kraken execution engine configuration
#[derive(Debug, Clone)]
pub struct KrakenExecutionConfig {
    /// Kraken REST client configuration
    pub rest_config: KrakenRestConfig,
    /// Enable dry run mode (validate orders but don't execute)
    pub dry_run: bool,
    /// Default slippage tolerance in basis points
    pub slippage_tolerance_bps: u32,
    /// Fee rate in basis points (for P&L calculation)
    pub fee_rate_bps: u32,
}

impl Default for KrakenExecutionConfig {
    fn default() -> Self {
        Self {
            rest_config: KrakenRestConfig::default(),
            dry_run: true,
            slippage_tolerance_bps: 10,
            fee_rate_bps: 26, // Kraken taker fee
        }
    }
}

impl KrakenExecutionConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let rest_config = KrakenRestConfig::from_env();

        let dry_run = std::env::var("KRAKEN_DRY_RUN")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let slippage_tolerance_bps = std::env::var("KRAKEN_SLIPPAGE_TOLERANCE_BPS")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10);

        let fee_rate_bps = std::env::var("KRAKEN_FEE_RATE_BPS")
            .unwrap_or_else(|_| "26".to_string())
            .parse()
            .unwrap_or(26);

        Self {
            rest_config,
            dry_run,
            slippage_tolerance_bps,
            fee_rate_bps,
        }
    }
}

/// Kraken live execution engine
pub struct KrakenExecutionEngine {
    /// Configuration
    config: KrakenExecutionConfig,
    /// REST API client
    client: KrakenRestClient,
    /// Active orders (internal ID -> Order)
    orders: Arc<RwLock<HashMap<String, Order>>>,
    /// Order ID mapping (internal ID -> Kraken TXID)
    order_id_map: Arc<RwLock<HashMap<String, String>>>,
    /// Positions by symbol
    positions: Arc<RwLock<HashMap<String, Position>>>,
    /// USD balance
    balance: Arc<RwLock<Decimal>>,
    /// Daily realized P&L
    daily_pnl: Arc<RwLock<Decimal>>,
}

impl KrakenExecutionEngine {
    /// Create a new Kraken execution engine
    pub fn new(config: KrakenExecutionConfig) -> Self {
        let client = KrakenRestClient::new(config.rest_config.clone());

        info!(
            "Kraken execution engine created (dry_run: {}, testnet: {})",
            config.dry_run, config.rest_config.testnet
        );

        Self {
            config,
            client,
            orders: Arc::new(RwLock::new(HashMap::new())),
            order_id_map: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            balance: Arc::new(RwLock::new(Decimal::ZERO)),
            daily_pnl: Arc::new(RwLock::new(Decimal::ZERO)),
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self::new(KrakenExecutionConfig::from_env())
    }

    /// Initialize the engine by fetching account balance
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing Kraken execution engine...");

        // Verify connectivity
        match self.client.get_server_time().await {
            Ok(time) => {
                info!("Connected to Kraken API, server time: {}", time);
            }
            Err(e) => {
                error!("Failed to connect to Kraken API: {}", e);
                return Err(e);
            }
        }

        // Fetch account balance if we have credentials
        if self.client.has_credentials() {
            match self.client.get_usd_balance().await {
                Ok(usd_balance) => {
                    *self.balance.write() = usd_balance;
                    info!("Account USD balance: ${}", usd_balance);
                }
                Err(e) => {
                    warn!("Failed to fetch balance: {}", e);
                }
            }

            // Sync open orders
            match self.client.get_open_orders().await {
                Ok(open_orders) => {
                    info!("Found {} open orders on Kraken", open_orders.len());
                    // Note: We could sync these to our internal state if needed
                }
                Err(e) => {
                    warn!("Failed to fetch open orders: {}", e);
                }
            }
        } else {
            warn!("No Kraken API credentials configured, balance sync disabled");
        }

        info!("Kraken execution engine initialized");
        Ok(())
    }

    /// Submit an order for execution
    pub async fn submit_order(&self, mut order: Order) -> Result<String> {
        let order_id = order.id.clone();
        info!(
            "Submitting {} {} order for {} {} on Kraken",
            order.order_type, order.side, order.quantity, order.symbol
        );

        // Validate order
        self.validate_order(&order)?;

        // Convert to Kraken format
        let kraken_symbol = to_kraken_symbol(&order.symbol);
        let side = match order.side {
            OrderSide::Buy => crate::types::OrderSide::Buy,
            OrderSide::Sell => crate::types::OrderSide::Sell,
        };

        // Update order status
        order.status = OrderStatusEnum::Submitted;
        order.updated_at = Utc::now();

        // Store order
        self.orders.write().insert(order_id.clone(), order.clone());

        // Execute based on order type
        let result = match order.order_type {
            OrderTypeEnum::Market => {
                self.execute_market_order(&kraken_symbol, side, order.quantity, &order_id)
                    .await
            }
            OrderTypeEnum::Limit => {
                let price = order.price.ok_or_else(|| {
                    ExecutionError::OrderValidation("Limit order requires price".to_string())
                })?;
                self.execute_limit_order(&kraken_symbol, side, price, order.quantity, &order_id)
                    .await
            }
            OrderTypeEnum::StopMarket => {
                let stop_price = order.stop_price.ok_or_else(|| {
                    ExecutionError::OrderValidation("Stop order requires stop price".to_string())
                })?;
                self.execute_stop_loss_order(
                    &kraken_symbol,
                    side,
                    stop_price,
                    order.quantity,
                    &order_id,
                )
                .await
            }
            _ => Err(ExecutionError::OrderValidation(format!(
                "Order type {:?} not supported on Kraken",
                order.order_type
            ))),
        };

        match result {
            Ok(txid) => {
                // Store mapping
                self.order_id_map
                    .write()
                    .insert(order_id.clone(), txid.clone());

                // Update order with exchange ID
                if let Some(order) = self.orders.write().get_mut(&order_id) {
                    order.exchange_order_id = Some(txid.clone());
                    order.updated_at = Utc::now();
                }

                info!("Order {} submitted successfully, txid: {}", order_id, txid);
                Ok(order_id)
            }
            Err(e) => {
                // Update order status to rejected
                if let Some(order) = self.orders.write().get_mut(&order_id) {
                    order.status = OrderStatusEnum::Rejected;
                    order.updated_at = Utc::now();
                }
                error!("Order {} rejected: {}", order_id, e);
                Err(e)
            }
        }
    }

    /// Execute a market order
    async fn execute_market_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        order_id: &str,
    ) -> Result<String> {
        debug!("Executing market order: {} {} {}", side, quantity, symbol);

        let result = self
            .client
            .place_market_order(symbol, side, quantity, self.config.dry_run)
            .await?;

        let txid = result.txid.first().cloned().unwrap_or_default();

        // For market orders, simulate immediate fill (in production, we'd poll for status)
        if !self.config.dry_run {
            // In a full implementation, we would:
            // 1. Poll order status until filled
            // 2. Get fill details
            // 3. Update position
            // For now, we assume immediate fill for market orders
            self.simulate_market_fill(order_id, symbol, side, quantity)
                .await?;
        }

        Ok(txid)
    }

    /// Execute a limit order
    async fn execute_limit_order(
        &self,
        symbol: &str,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
        _order_id: &str,
    ) -> Result<String> {
        debug!(
            "Executing limit order: {} {} {} @ {}",
            side, quantity, symbol, price
        );

        let result = self
            .client
            .place_limit_order(symbol, side, price, quantity, self.config.dry_run)
            .await?;

        let txid = result.txid.first().cloned().unwrap_or_default();
        Ok(txid)
    }

    /// Execute a stop-loss order
    async fn execute_stop_loss_order(
        &self,
        symbol: &str,
        side: OrderSide,
        stop_price: Decimal,
        quantity: Decimal,
        _order_id: &str,
    ) -> Result<String> {
        debug!(
            "Executing stop-loss order: {} {} {} @ {}",
            side, quantity, symbol, stop_price
        );

        let result = self
            .client
            .place_stop_loss_order(symbol, side, stop_price, quantity, self.config.dry_run)
            .await?;

        let txid = result.txid.first().cloned().unwrap_or_default();
        Ok(txid)
    }

    /// Simulate a market fill (for immediate execution)
    async fn simulate_market_fill(
        &self,
        order_id: &str,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
    ) -> Result<()> {
        // In production, we'd get the actual fill price from the exchange
        // For now, we just mark the order as filled
        let mut orders = self.orders.write();
        if let Some(order) = orders.get_mut(order_id) {
            // Create a fill (in production, get actual fill price)
            let fill = Fill {
                id: Uuid::new_v4().to_string(),
                order_id: order_id.to_string(),
                quantity,
                price: order.price.unwrap_or(Decimal::ZERO), // Would come from exchange
                fee: self.calculate_fee(quantity, order.price.unwrap_or(Decimal::ZERO)),
                fee_currency: "USD".to_string(),
                side,
                timestamp: Utc::now(),
                is_maker: false,
            };

            order.add_fill(fill.clone());
            order.status = OrderStatusEnum::Filled;
            order.updated_at = Utc::now();

            // Update position
            drop(orders);
            self.update_position(symbol, &fill)?;
        }

        Ok(())
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        info!("Cancelling order: {}", order_id);

        // Get the Kraken TXID
        let txid = {
            let map = self.order_id_map.read();
            map.get(order_id).cloned()
        };

        if let Some(txid) = txid {
            // Cancel on exchange
            self.client.cancel_order(&txid).await?;

            // Update internal state
            if let Some(order) = self.orders.write().get_mut(order_id) {
                order.status = OrderStatusEnum::Cancelled;
                order.updated_at = Utc::now();
            }

            info!("Order {} cancelled successfully", order_id);
            Ok(())
        } else {
            // Order might be local-only (rejected before submission)
            if let Some(order) = self.orders.write().get_mut(order_id) {
                order.status = OrderStatusEnum::Cancelled;
                order.updated_at = Utc::now();
                Ok(())
            } else {
                Err(ExecutionError::OrderNotFound(order_id.to_string()))
            }
        }
    }

    /// Get an order by ID
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

    /// Get execution mode
    pub fn mode(&self) -> ExecutionMode {
        ExecutionMode::Live
    }

    /// Get a position by symbol
    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        self.positions.read().get(symbol).cloned()
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> Vec<Position> {
        self.positions.read().values().cloned().collect()
    }

    /// Get current USD balance
    pub fn get_balance(&self) -> Decimal {
        *self.balance.read()
    }

    /// Get daily P&L
    pub fn get_daily_pnl(&self) -> Decimal {
        *self.daily_pnl.read()
    }

    /// Refresh balance from exchange
    pub async fn refresh_balance(&self) -> Result<Decimal> {
        let balance = self.client.get_usd_balance().await?;
        *self.balance.write() = balance;
        Ok(balance)
    }

    /// Validate an order before submission
    fn validate_order(&self, order: &Order) -> Result<()> {
        // Check quantity
        if order.quantity <= Decimal::ZERO {
            return Err(ExecutionError::OrderValidation(
                "Order quantity must be positive".to_string(),
            ));
        }

        // Check symbol format
        if order.symbol.is_empty() {
            return Err(ExecutionError::OrderValidation(
                "Order symbol is required".to_string(),
            ));
        }

        // Check limit order has price
        if order.order_type == OrderTypeEnum::Limit && order.price.is_none() {
            return Err(ExecutionError::OrderValidation(
                "Limit order requires price".to_string(),
            ));
        }

        // Check stop order has stop price
        if order.order_type == OrderTypeEnum::StopMarket && order.stop_price.is_none() {
            return Err(ExecutionError::OrderValidation(
                "Stop order requires stop price".to_string(),
            ));
        }

        // Check minimum order size (Kraken minimum is ~$10)
        let min_order_value = Decimal::from(10);
        let order_value = order.quantity * order.price.unwrap_or(Decimal::from(1));
        if order_value < min_order_value && order.order_type == OrderTypeEnum::Market {
            // For market orders without price, we can't validate value
            // This would be caught by the exchange
        }

        Ok(())
    }

    /// Update position after a fill
    fn update_position(&self, symbol: &str, fill: &Fill) -> Result<()> {
        let mut positions = self.positions.write();

        if let Some(position) = positions.get_mut(symbol) {
            // Existing position
            let old_qty = position.quantity;
            position.update_with_fill(fill);

            if position.quantity == Decimal::ZERO {
                // Position closed
                let realized = position.realized_pnl;
                *self.daily_pnl.write() += realized;
                positions.remove(symbol);
                info!("Position {} closed, realized P&L: ${}", symbol, realized);
            } else {
                info!(
                    "Position {} updated: {} -> {} @ {}",
                    symbol, old_qty, position.quantity, position.average_entry_price
                );
            }
        } else {
            // New position
            let side = match fill.side {
                OrderSide::Buy => PositionSide::Long,
                OrderSide::Sell => PositionSide::Short,
            };

            let position = Position {
                symbol: symbol.to_string(),
                exchange: "kraken".to_string(),
                side,
                quantity: fill.quantity,
                average_entry_price: fill.price,
                current_price: fill.price,
                unrealized_pnl: Decimal::ZERO,
                realized_pnl: Decimal::ZERO,
                margin_used: Decimal::ZERO,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };

            info!(
                "New {} position opened: {} {} @ {}",
                side, fill.quantity, symbol, fill.price
            );

            positions.insert(symbol.to_string(), position);
        }

        Ok(())
    }

    /// Calculate fee for a trade
    fn calculate_fee(&self, quantity: Decimal, price: Decimal) -> Decimal {
        let trade_value = quantity * price;
        let fee_rate = Decimal::from(self.config.fee_rate_bps) / Decimal::from(10000);
        trade_value * fee_rate
    }

    /// Reset daily P&L (call at end of day)
    pub fn reset_daily_pnl(&self) {
        *self.daily_pnl.write() = Decimal::ZERO;
        info!("Daily P&L reset");
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> KrakenExecutionStats {
        let orders = self.orders.read();
        let positions = self.positions.read();

        let total_orders = orders.len();
        let filled_orders = orders
            .values()
            .filter(|o| o.status == OrderStatusEnum::Filled)
            .count();
        let cancelled_orders = orders
            .values()
            .filter(|o| o.status == OrderStatusEnum::Cancelled)
            .count();
        let rejected_orders = orders
            .values()
            .filter(|o| o.status == OrderStatusEnum::Rejected)
            .count();

        let unrealized_pnl: Decimal = positions.values().map(|p| p.unrealized_pnl).sum();
        let realized_pnl: Decimal = positions.values().map(|p| p.realized_pnl).sum();

        KrakenExecutionStats {
            balance: *self.balance.read(),
            unrealized_pnl,
            realized_pnl,
            daily_pnl: *self.daily_pnl.read(),
            total_value: *self.balance.read() + unrealized_pnl,
            open_positions: positions.len(),
            total_orders,
            filled_orders,
            cancelled_orders,
            rejected_orders,
            dry_run: self.config.dry_run,
        }
    }
}

/// Kraken execution statistics
#[derive(Debug, Clone)]
pub struct KrakenExecutionStats {
    pub balance: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub daily_pnl: Decimal,
    pub total_value: Decimal,
    pub open_positions: usize,
    pub total_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub rejected_orders: usize,
    pub dry_run: bool,
}

// ==================== Helper Functions ====================

/// Convert normalized symbol to Kraken format
fn to_kraken_symbol(symbol: &str) -> String {
    // Convert BTC/USDT -> BTC/USD
    // Kraken uses USD instead of USDT for major pairs
    symbol.replace("USDT", "USD").replace("BTCUSD", "BTC/USD")
}

// ==================== ExecutionEngine Implementation ====================

#[async_trait::async_trait]
impl super::ExecutionEngine for KrakenExecutionEngine {
    async fn submit_order(&self, order: Order) -> Result<String> {
        self.submit_order(order).await
    }

    async fn cancel_order(&self, order_id: &str) -> Result<()> {
        self.cancel_order(order_id).await
    }

    fn get_order(&self, order_id: &str) -> Result<Order> {
        self.get_order(order_id)
    }

    fn get_active_orders(&self) -> Vec<Order> {
        self.get_active_orders()
    }

    fn mode(&self) -> ExecutionMode {
        self.mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> KrakenExecutionConfig {
        KrakenExecutionConfig {
            rest_config: KrakenRestConfig::default(),
            dry_run: true,
            slippage_tolerance_bps: 10,
            fee_rate_bps: 26,
        }
    }

    fn create_test_order() -> Order {
        Order::new(
            "sig123".to_string(),
            "BTC/USD".to_string(),
            "kraken".to_string(),
            OrderSide::Buy,
            OrderTypeEnum::Market,
            Decimal::from_str_exact("0.001").unwrap(),
        )
    }

    #[test]
    fn test_engine_creation() {
        let config = create_test_config();
        let engine = KrakenExecutionEngine::new(config);
        assert_eq!(engine.mode(), ExecutionMode::Live);
        assert!(engine.config.dry_run);
    }

    #[test]
    fn test_validate_order_valid() {
        let config = create_test_config();
        let engine = KrakenExecutionEngine::new(config);
        let order = create_test_order();

        assert!(engine.validate_order(&order).is_ok());
    }

    #[test]
    fn test_validate_order_zero_quantity() {
        let config = create_test_config();
        let engine = KrakenExecutionEngine::new(config);
        let mut order = create_test_order();
        order.quantity = Decimal::ZERO;

        assert!(engine.validate_order(&order).is_err());
    }

    #[test]
    fn test_validate_limit_order_no_price() {
        let config = create_test_config();
        let engine = KrakenExecutionEngine::new(config);
        let mut order = create_test_order();
        order.order_type = OrderTypeEnum::Limit;
        order.price = None;

        assert!(engine.validate_order(&order).is_err());
    }

    #[test]
    fn test_to_kraken_symbol() {
        assert_eq!(to_kraken_symbol("BTC/USDT"), "BTC/USD");
        assert_eq!(to_kraken_symbol("ETH/USDT"), "ETH/USD");
        assert_eq!(to_kraken_symbol("BTC/USD"), "BTC/USD");
    }

    #[test]
    fn test_calculate_fee() {
        let config = create_test_config();
        let engine = KrakenExecutionEngine::new(config);

        let quantity = Decimal::from(1);
        let price = Decimal::from(50000);
        let fee = engine.calculate_fee(quantity, price);

        // 26 bps = 0.26% = $130 on $50,000
        assert_eq!(fee, Decimal::from(130));
    }

    #[test]
    fn test_stats() {
        let config = create_test_config();
        let engine = KrakenExecutionEngine::new(config);

        let stats = engine.get_stats();
        assert_eq!(stats.total_orders, 0);
        assert_eq!(stats.open_positions, 0);
        assert!(stats.dry_run);
    }

    #[test]
    fn test_get_order_not_found() {
        let config = create_test_config();
        let engine = KrakenExecutionEngine::new(config);

        let result = engine.get_order("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_active_orders_empty() {
        let config = create_test_config();
        let engine = KrakenExecutionEngine::new(config);

        let active = engine.get_active_orders();
        assert!(active.is_empty());
    }
}
