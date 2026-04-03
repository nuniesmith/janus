//! Exchange router for multi-exchange order routing and aggregation
//!
//! This module provides a router that can manage multiple exchange connections,
//! route orders to the appropriate exchange, and aggregate positions across exchanges.
//!
//! ## Exchange Priority
//!
//! | Priority | Exchange | Role                              |
//! |----------|----------|-----------------------------------|
//! | 1st      | Kraken   | **Primary** — default data & REST |
//! | 2nd      | Bybit    | Backup / alternate                |
//! | 3rd      | Binance  | Tertiary / additional liquidity   |
//!
//! Use [`ExchangeRouter::new_with_kraken_default`] to create a router that
//! automatically sets Kraken as the default exchange once it is added.

use crate::error::{ExecutionError, Result};
use crate::exchanges::{Balance, Exchange, OrderStatusResponse};
use crate::types::{Order, Position};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Exchange router that manages multiple exchanges
pub struct ExchangeRouter {
    /// Map of exchange name to exchange instance
    exchanges: Arc<RwLock<HashMap<String, Arc<dyn Exchange>>>>,

    /// Default exchange for routing
    default_exchange: Arc<RwLock<Option<String>>>,

    /// Routing rules (symbol -> exchange)
    routing_rules: Arc<RwLock<HashMap<String, String>>>,
}

impl ExchangeRouter {
    /// Create a new exchange router with no default.
    ///
    /// The first exchange added via [`add_exchange`] becomes the default.
    /// If you want Kraken to be the default regardless of insertion order,
    /// use [`new_with_kraken_default`] instead.
    pub fn new() -> Self {
        Self {
            exchanges: Arc::new(RwLock::new(HashMap::new())),
            default_exchange: Arc::new(RwLock::new(None)),
            routing_rules: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new exchange router that prefers **Kraken** as the default.
    ///
    /// When an exchange named `"kraken"` is added it will automatically be
    /// promoted to the default, even if other exchanges were added first.
    /// Bybit and Binance serve as backup/alternate routes.
    pub fn new_with_kraken_default() -> Self {
        Self {
            exchanges: Arc::new(RwLock::new(HashMap::new())),
            default_exchange: Arc::new(RwLock::new(Some("kraken".to_string()))),
            routing_rules: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add an exchange to the router
    ///
    /// # Arguments
    /// * `name` - Exchange name (e.g., "kraken", "bybit", "binance")
    /// * `exchange` - Exchange implementation
    ///
    /// If the router was created via [`new_with_kraken_default`] and `name`
    /// matches `"kraken"`, it will remain the default. Otherwise the first
    /// exchange added becomes the default.
    pub async fn add_exchange(&self, name: impl Into<String>, exchange: Box<dyn Exchange>) {
        let name = name.into();
        let exchange: Arc<dyn Exchange> = Arc::from(exchange);
        info!("Adding exchange: {}", name);

        // Step 1: Insert the exchange and snapshot the state we need.
        //         Drop the exchanges lock before touching the default lock.
        let (exchange_count, current_default_snapshot) = {
            let mut exchanges = self.exchanges.write().await;
            exchanges.insert(name.clone(), exchange);
            let count = exchanges.len();
            // Snapshot the current default so we can reason about it after
            // releasing the exchanges lock.
            let default_snapshot = self.default_exchange.read().await.clone();
            (count, default_snapshot)
        };
        // Both locks are now released.

        // Step 2: Decide whether to update the default exchange.
        let needs_update = match current_default_snapshot.as_deref() {
            // No default configured yet — use this exchange.
            None => true,
            // This *is* the preferred default — already set.
            Some(preferred) if preferred == name => false,
            Some(preferred) => {
                // Preferred default hasn't been added yet and this is the
                // only exchange available — use it as interim default.
                let exchanges = self.exchanges.read().await;
                !exchanges.contains_key(preferred) && exchange_count == 1
            }
        };

        if needs_update {
            *self.default_exchange.write().await = Some(name);
        }
    }

    /// Remove an exchange from the router
    ///
    /// # Arguments
    /// * `name` - Exchange name to remove
    pub async fn remove_exchange(&self, name: &str) -> Option<Arc<dyn Exchange>> {
        info!("Removing exchange: {}", name);
        let exchange = self.exchanges.write().await.remove(name);

        // Update default if we removed it
        if self.default_exchange.read().await.as_deref() == Some(name) {
            *self.default_exchange.write().await =
                self.exchanges.read().await.keys().next().cloned();
        }

        exchange
    }

    /// Set the default exchange for routing
    ///
    /// # Arguments
    /// * `name` - Exchange name to use as default
    pub async fn set_default_exchange(&self, name: impl Into<String>) -> Result<()> {
        let name = name.into();

        if !self.exchanges.read().await.contains_key(&name) {
            return Err(ExecutionError::UnknownExchange(name));
        }

        info!("Setting default exchange: {}", name);
        *self.default_exchange.write().await = Some(name);
        Ok(())
    }

    /// Add a routing rule for a specific symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol (e.g., "BTCUSD")
    /// * `exchange` - Exchange name to route this symbol to
    pub async fn add_routing_rule(
        &self,
        symbol: impl Into<String>,
        exchange: impl Into<String>,
    ) -> Result<()> {
        let symbol = symbol.into();
        let exchange = exchange.into();

        if !self.exchanges.read().await.contains_key(&exchange) {
            return Err(ExecutionError::UnknownExchange(exchange));
        }

        debug!("Adding routing rule: {} -> {}", symbol, exchange);
        self.routing_rules.write().await.insert(symbol, exchange);
        Ok(())
    }

    /// Get the exchange for a given symbol or order
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol
    /// * `preferred_exchange` - Optional preferred exchange
    ///
    /// # Returns
    /// * Exchange name to use
    async fn get_exchange_for_symbol(
        &self,
        symbol: &str,
        preferred_exchange: Option<&str>,
    ) -> Result<String> {
        // 1. Use preferred exchange if specified and available
        if let Some(exchange) = preferred_exchange {
            if self.exchanges.read().await.contains_key(exchange) {
                return Ok(exchange.to_string());
            }
        }

        // 2. Check routing rules
        if let Some(exchange) = self.routing_rules.read().await.get(symbol) {
            return Ok(exchange.clone());
        }

        // 3. Use default exchange
        self.default_exchange
            .read()
            .await
            .clone()
            .ok_or_else(|| ExecutionError::Config("No exchanges configured".to_string()))
    }

    /// Get a reference to an exchange by name
    pub async fn get_exchange(&self, name: &str) -> Result<Arc<dyn Exchange>> {
        self.exchanges
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| ExecutionError::UnknownExchange(name.to_string()))
    }

    /// Place an order on the appropriate exchange
    ///
    /// # Arguments
    /// * `order` - Order to place
    ///
    /// # Returns
    /// * `Ok((exchange_name, order_id))` - Exchange used and order ID
    pub async fn place_order(&self, order: &Order) -> Result<(String, String)> {
        let exchange_name = self
            .get_exchange_for_symbol(&order.symbol, Some(&order.exchange))
            .await?;

        let order_id = {
            let exchanges = self.exchanges.read().await;
            let exchange = exchanges
                .get(&exchange_name)
                .ok_or_else(|| ExecutionError::UnknownExchange(exchange_name.clone()))?;

            exchange.place_order(order).await?
        };

        info!(
            "Order placed on {}: {} {} {}",
            exchange_name, order.side, order.quantity, order.symbol
        );

        Ok((exchange_name, order_id))
    }

    /// Cancel an order on a specific exchange
    ///
    /// # Arguments
    /// * `exchange_name` - Exchange where the order was placed
    /// * `order_id` - Order ID to cancel
    pub async fn cancel_order(&self, exchange_name: &str, order_id: &str) -> Result<()> {
        let exchanges = self.exchanges.read().await;
        let exchange = exchanges
            .get(exchange_name)
            .ok_or_else(|| ExecutionError::UnknownExchange(exchange_name.to_string()))?;

        exchange.cancel_order(order_id).await
    }

    /// Get order status from a specific exchange
    ///
    /// # Arguments
    /// * `exchange_name` - Exchange where the order was placed
    /// * `order_id` - Order ID to query
    pub async fn get_order_status(
        &self,
        exchange_name: &str,
        order_id: &str,
    ) -> Result<OrderStatusResponse> {
        let exchanges = self.exchanges.read().await;
        let exchange = exchanges
            .get(exchange_name)
            .ok_or_else(|| ExecutionError::UnknownExchange(exchange_name.to_string()))?;

        exchange.get_order_status(order_id).await
    }

    /// Get all active orders across all exchanges
    ///
    /// # Arguments
    /// * `symbol` - Optional symbol filter
    ///
    /// # Returns
    /// * `Ok(Vec<(exchange_name, orders)>)` - List of orders by exchange
    pub async fn get_all_active_orders(
        &self,
        symbol: Option<&str>,
    ) -> Result<Vec<(String, Vec<OrderStatusResponse>)>> {
        let mut all_orders = Vec::new();

        let exchanges = self.exchanges.read().await;
        for (name, exchange) in exchanges.iter() {
            match exchange.get_active_orders(symbol).await {
                Ok(orders) => {
                    if !orders.is_empty() {
                        all_orders.push((name.clone(), orders));
                    }
                }
                Err(e) => {
                    warn!("Failed to get orders from {}: {}", name, e);
                }
            }
        }

        Ok(all_orders)
    }

    /// Get aggregated positions across all exchanges
    ///
    /// # Arguments
    /// * `symbol` - Optional symbol filter
    ///
    /// # Returns
    /// * `Ok(Vec<Position>)` - Aggregated positions
    pub async fn get_aggregated_positions(&self, symbol: Option<&str>) -> Result<Vec<Position>> {
        let mut positions_by_symbol: HashMap<String, Position> = HashMap::new();

        let exchanges = self.exchanges.read().await;
        for (name, exchange) in exchanges.iter() {
            match exchange.get_positions(symbol).await {
                Ok(positions) => {
                    for mut pos in positions {
                        positions_by_symbol
                            .entry(pos.symbol.clone())
                            .and_modify(|existing| {
                                // Aggregate positions for the same symbol
                                // This is a simplified aggregation - production would need more sophisticated logic
                                existing.quantity += pos.quantity;
                                existing.unrealized_pnl += pos.unrealized_pnl;
                                existing.realized_pnl += pos.realized_pnl;
                            })
                            .or_insert_with(|| {
                                // Mark which exchange this came from
                                pos.exchange = name.clone();
                                pos
                            });
                    }
                }
                Err(e) => {
                    warn!("Failed to get positions from {}: {}", name, e);
                }
            }
        }

        Ok(positions_by_symbol.into_values().collect())
    }

    /// Get total balance across all exchanges
    ///
    /// # Returns
    /// * `Ok(Balance)` - Aggregated balance
    pub async fn get_total_balance(&self) -> Result<Balance> {
        use rust_decimal::Decimal;

        let mut total = Decimal::ZERO;
        let mut available = Decimal::ZERO;
        let mut used = Decimal::ZERO;

        let exchanges = self.exchanges.read().await;
        for (name, exchange) in exchanges.iter() {
            match exchange.get_balance().await {
                Ok(balance) => {
                    total += balance.total;
                    available += balance.available;
                    used += balance.used;
                }
                Err(e) => {
                    warn!("Failed to get balance from {}: {}", name, e);
                }
            }
        }

        Ok(Balance {
            total,
            available,
            used,
            currency: "USDT".to_string(), // Simplified
            timestamp: chrono::Utc::now(),
        })
    }

    /// Health check all exchanges
    ///
    /// # Returns
    /// * `Ok(Vec<(exchange_name, is_healthy)>)` - Health status of all exchanges
    pub async fn health_check_all(&self) -> Vec<(String, bool)> {
        let mut results = Vec::new();

        let exchanges = self.exchanges.read().await;
        for (name, exchange) in exchanges.iter() {
            let is_healthy = exchange.health_check().await.is_ok();
            results.push((name.clone(), is_healthy));
        }

        results
    }

    /// Get list of available exchanges
    pub async fn list_exchanges(&self) -> Vec<String> {
        self.exchanges.read().await.keys().cloned().collect()
    }

    /// Get the default exchange name
    pub async fn get_default_exchange(&self) -> Option<String> {
        self.default_exchange.read().await.clone()
    }

    /// Replace routing rules with a map obtained from the Registry service.
    ///
    /// Only rules whose exchange name matches a currently-registered exchange
    /// are inserted. Returns the number of rules actually added.
    pub async fn sync_routing_from_registry(&self, routes: HashMap<String, String>) -> usize {
        let exchanges = self.exchanges.read().await;
        let mut rules = self.routing_rules.write().await;

        rules.clear();

        let mut count = 0usize;
        for (symbol, exchange) in routes {
            if exchanges.contains_key(&exchange) {
                rules.insert(symbol, exchange);
                count += 1;
            } else {
                warn!(
                    symbol = %symbol,
                    exchange = %exchange,
                    "Skipping routing rule — exchange not registered in router"
                );
            }
        }

        info!(rules_added = count, "Synced routing rules from registry");
        count
    }
}

impl Default for ExchangeRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exchanges::{OrderUpdateReceiver, PositionUpdateReceiver};
    use crate::types::{OrderSide, OrderStatusEnum, OrderTypeEnum};
    use async_trait::async_trait;
    use rust_decimal::Decimal;

    // Mock exchange for testing
    struct MockExchange {
        name: String,
    }

    #[async_trait]
    impl Exchange for MockExchange {
        fn name(&self) -> &str {
            &self.name
        }

        fn is_testnet(&self) -> bool {
            true
        }

        async fn place_order(&self, _order: &Order) -> Result<String> {
            Ok(format!("{}_order_123", self.name))
        }

        async fn cancel_order(&self, _order_id: &str) -> Result<()> {
            Ok(())
        }

        async fn cancel_all_orders(&self, _symbol: Option<&str>) -> Result<Vec<String>> {
            Ok(vec![])
        }

        async fn get_order_status(&self, order_id: &str) -> Result<OrderStatusResponse> {
            Ok(OrderStatusResponse {
                order_id: order_id.to_string(),
                client_order_id: None,
                symbol: "BTCUSD".to_string(),
                status: OrderStatusEnum::Filled,
                quantity: Decimal::from(1),
                filled_quantity: Decimal::from(1),
                remaining_quantity: Decimal::ZERO,
                price: Some(Decimal::from(50000)),
                average_fill_price: Some(Decimal::from(50000)),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                fills: vec![],
            })
        }

        async fn get_active_orders(
            &self,
            _symbol: Option<&str>,
        ) -> Result<Vec<OrderStatusResponse>> {
            Ok(vec![])
        }

        async fn get_balance(&self) -> Result<Balance> {
            Ok(Balance {
                total: Decimal::from(10000),
                available: Decimal::from(5000),
                used: Decimal::from(5000),
                currency: "USDT".to_string(),
                timestamp: chrono::Utc::now(),
            })
        }

        async fn get_positions(&self, _symbol: Option<&str>) -> Result<Vec<Position>> {
            Ok(vec![])
        }

        async fn subscribe_order_updates(&self) -> Result<OrderUpdateReceiver> {
            let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
            Ok(rx)
        }

        async fn subscribe_position_updates(&self) -> Result<PositionUpdateReceiver> {
            let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
            Ok(rx)
        }

        async fn health_check(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_router_creation() {
        let router = ExchangeRouter::new();
        assert!(router.list_exchanges().await.is_empty());
        assert!(router.get_default_exchange().await.is_none());
    }

    #[tokio::test]
    async fn test_add_exchange() {
        let router = ExchangeRouter::new();
        let mock = MockExchange {
            name: "test".to_string(),
        };

        router.add_exchange("test", Box::new(mock)).await;

        assert_eq!(router.list_exchanges().await.len(), 1);
        assert_eq!(
            router.get_default_exchange().await,
            Some("test".to_string())
        );
    }

    #[tokio::test]
    async fn test_routing_rules() {
        let router = ExchangeRouter::new();
        let mock1 = MockExchange {
            name: "exchange1".to_string(),
        };
        let mock2 = MockExchange {
            name: "exchange2".to_string(),
        };

        router.add_exchange("exchange1", Box::new(mock1)).await;
        router.add_exchange("exchange2", Box::new(mock2)).await;

        router
            .add_routing_rule("BTCUSD", "exchange1")
            .await
            .unwrap();
        router
            .add_routing_rule("ETHUSDT", "exchange2")
            .await
            .unwrap();

        assert_eq!(
            router
                .get_exchange_for_symbol("BTCUSD", None)
                .await
                .unwrap(),
            "exchange1"
        );
        assert_eq!(
            router
                .get_exchange_for_symbol("ETHUSDT", None)
                .await
                .unwrap(),
            "exchange2"
        );
    }

    #[tokio::test]
    async fn test_place_order() {
        let router = ExchangeRouter::new();
        let mock = MockExchange {
            name: "test".to_string(),
        };

        router.add_exchange("test", Box::new(mock)).await;

        let order = Order::new(
            "signal123".to_string(),
            "BTCUSD".to_string(),
            "test".to_string(),
            OrderSide::Buy,
            OrderTypeEnum::Market,
            Decimal::from(1),
        );

        let result = router.place_order(&order).await;
        assert!(result.is_ok());

        let (exchange, order_id) = result.unwrap();
        assert_eq!(exchange, "test");
        assert!(order_id.contains("test"));
    }

    #[tokio::test]
    async fn test_health_check_all() {
        let router = ExchangeRouter::new();
        let mock1 = MockExchange {
            name: "exchange1".to_string(),
        };
        let mock2 = MockExchange {
            name: "exchange2".to_string(),
        };

        router.add_exchange("exchange1", Box::new(mock1)).await;
        router.add_exchange("exchange2", Box::new(mock2)).await;

        let results = router.health_check_all().await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(_, healthy)| *healthy));
    }
}
