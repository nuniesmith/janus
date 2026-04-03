//! # CancelAll Circuit Breaker
//!
//! Emergency circuit breaker that cancels all open orders across all exchanges.
//! Part of the Amygdala's threat response system.
//!
//! ## Use Cases
//!
//! - Market crash detected
//! - System malfunction
//! - Regulatory halt
//! - Excessive losses
//! - Manual emergency stop

use crate::common::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Result of cancel all operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAllResult {
    /// When the operation was triggered
    pub triggered_at: DateTime<Utc>,

    /// Total orders attempted to cancel
    pub total_orders: usize,

    /// Successfully cancelled orders
    pub cancelled_count: usize,

    /// Failed cancellations
    pub failed_count: usize,

    /// Cancellation duration
    pub duration_ms: u64,

    /// Per-exchange results
    pub exchange_results: HashMap<String, ExchangeResult>,

    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Result for a specific exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeResult {
    pub exchange: String,
    pub orders_cancelled: usize,
    pub orders_failed: usize,
    pub duration_ms: u64,
}

/// Order information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub exchange: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub price: Option<f64>,
    pub status: OrderStatus,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Open,
    Filled,
    Cancelled,
    PartiallyFilled,
    Rejected,
    Expired,
}

/// Exchange client trait for cancellation
#[async_trait::async_trait]
pub trait ExchangeClient: Send + Sync {
    /// Get all open orders
    async fn get_open_orders(&self) -> Result<Vec<Order>>;

    /// Cancel a specific order
    async fn cancel_order(&self, order_id: &str, symbol: &str) -> Result<()>;

    /// Bulk cancel all orders (if supported)
    async fn cancel_all_orders(&self) -> Result<usize>;

    /// Exchange identifier
    fn exchange_name(&self) -> &str;
}

/// CancelAll circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAllConfig {
    /// Maximum time to wait for all cancellations
    pub timeout_secs: u64,

    /// Whether to use bulk cancel if available
    pub use_bulk_cancel: bool,

    /// Retry failed cancellations
    pub retry_failed: bool,

    /// Maximum retry attempts
    pub max_retries: u32,

    /// Delay between retries (ms)
    pub retry_delay_ms: u64,

    /// Send alerts on trigger
    pub send_alerts: bool,
}

impl Default for CancelAllConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            use_bulk_cancel: true,
            retry_failed: true,
            max_retries: 3,
            retry_delay_ms: 500,
            send_alerts: true,
        }
    }
}

/// CancelAll circuit breaker
pub struct CancelAll {
    config: CancelAllConfig,
    exchange_clients: Arc<RwLock<HashMap<String, Arc<dyn ExchangeClient>>>>,
    last_trigger: Arc<RwLock<Option<DateTime<Utc>>>>,
    trigger_count: Arc<RwLock<u64>>,
}

impl Default for CancelAll {
    fn default() -> Self {
        Self::new(CancelAllConfig::default())
    }
}

impl CancelAll {
    /// Create a new CancelAll circuit breaker
    pub fn new(config: CancelAllConfig) -> Self {
        Self {
            config,
            exchange_clients: Arc::new(RwLock::new(HashMap::new())),
            last_trigger: Arc::new(RwLock::new(None)),
            trigger_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Register an exchange client
    pub async fn register_exchange(&self, client: Arc<dyn ExchangeClient>) {
        let exchange_name = client.exchange_name().to_string();
        let mut clients = self.exchange_clients.write().await;
        clients.insert(exchange_name.clone(), client);
        info!("Registered exchange client: {}", exchange_name);
    }

    /// Trigger cancel all orders
    pub async fn trigger(&self, reason: &str) -> Result<CancelAllResult> {
        let start = std::time::Instant::now();

        info!("🚨 CANCEL ALL TRIGGERED: {}", reason);

        // Update trigger tracking
        {
            let mut last_trigger = self.last_trigger.write().await;
            *last_trigger = Some(Utc::now());

            let mut count = self.trigger_count.write().await;
            *count += 1;
        }

        let mut total_orders = 0;
        let mut cancelled_count = 0;
        let mut failed_count = 0;
        let mut exchange_results = HashMap::new();
        let mut errors = Vec::new();

        let clients = self.exchange_clients.read().await;

        // Cancel orders on each exchange
        for (exchange_name, client) in clients.iter() {
            let exchange_start = std::time::Instant::now();

            match self.cancel_exchange_orders(client.clone()).await {
                Ok((cancelled, failed)) => {
                    total_orders += cancelled + failed;
                    cancelled_count += cancelled;
                    failed_count += failed;

                    exchange_results.insert(
                        exchange_name.clone(),
                        ExchangeResult {
                            exchange: exchange_name.clone(),
                            orders_cancelled: cancelled,
                            orders_failed: failed,
                            duration_ms: exchange_start.elapsed().as_millis() as u64,
                        },
                    );

                    info!(
                        "Exchange {}: {} cancelled, {} failed",
                        exchange_name, cancelled, failed
                    );
                }
                Err(e) => {
                    let error_msg = format!("Exchange {} error: {}", exchange_name, e);
                    error!("{}", error_msg);
                    errors.push(error_msg);
                }
            }
        }

        let result = CancelAllResult {
            triggered_at: Utc::now(),
            total_orders,
            cancelled_count,
            failed_count,
            duration_ms: start.elapsed().as_millis() as u64,
            exchange_results,
            errors,
        };

        info!(
            "✅ CANCEL ALL COMPLETE: {}/{} cancelled in {}ms",
            result.cancelled_count, result.total_orders, result.duration_ms
        );

        Ok(result)
    }

    /// Cancel all orders on a specific exchange
    async fn cancel_exchange_orders(
        &self,
        client: Arc<dyn ExchangeClient>,
    ) -> Result<(usize, usize)> {
        let mut cancelled = 0;
        let mut failed = 0;

        // Try bulk cancel first if enabled
        if self.config.use_bulk_cancel {
            match client.cancel_all_orders().await {
                Ok(count) => {
                    cancelled = count;
                    return Ok((cancelled, failed));
                }
                Err(e) => {
                    warn!(
                        "Bulk cancel failed for {}, falling back to individual: {}",
                        client.exchange_name(),
                        e
                    );
                }
            }
        }

        // Fall back to individual cancellations
        let orders = client.get_open_orders().await?;
        info!(
            "Found {} open orders on {}",
            orders.len(),
            client.exchange_name()
        );

        for order in orders {
            if !matches!(
                order.status,
                OrderStatus::Open | OrderStatus::PartiallyFilled
            ) {
                continue;
            }

            match self
                .cancel_order_with_retry(client.clone(), &order.id, &order.symbol)
                .await
            {
                Ok(_) => {
                    cancelled += 1;
                    info!("Cancelled order {} on {}", order.id, order.symbol);
                }
                Err(e) => {
                    failed += 1;
                    error!("Failed to cancel order {}: {}", order.id, e);
                }
            }
        }

        Ok((cancelled, failed))
    }

    /// Cancel a single order with retry logic
    async fn cancel_order_with_retry(
        &self,
        client: Arc<dyn ExchangeClient>,
        order_id: &str,
        symbol: &str,
    ) -> Result<()> {
        let mut last_error = None;

        for attempt in 1..=self.config.max_retries {
            match client.cancel_order(order_id, symbol).await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.max_retries {
                        tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Get statistics
    pub async fn stats(&self) -> CancelAllStats {
        let trigger_count = *self.trigger_count.read().await;
        let last_trigger = *self.last_trigger.read().await;
        let registered_exchanges = self.exchange_clients.read().await.len();

        CancelAllStats {
            trigger_count,
            last_trigger,
            registered_exchanges,
        }
    }

    /// Check if recently triggered (within last N seconds)
    pub async fn is_recently_triggered(&self, within_secs: u64) -> bool {
        match *self.last_trigger.read().await {
            Some(last) => {
                let elapsed = Utc::now().signed_duration_since(last);
                elapsed.num_seconds() < within_secs as i64
            }
            _ => false,
        }
    }

    /// Reset trigger tracking
    pub async fn reset(&self) {
        let mut last_trigger = self.last_trigger.write().await;
        *last_trigger = None;

        let mut count = self.trigger_count.write().await;
        *count = 0;
    }
}

/// CancelAll statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAllStats {
    pub trigger_count: u64,
    pub last_trigger: Option<DateTime<Utc>>,
    pub registered_exchanges: usize,
}

// ============================================================================
// Mock Exchange Client for Testing
// ============================================================================

#[cfg(test)]
mod mock_exchange {
    use super::*;

    pub struct MockExchange {
        name: String,
        orders: Arc<RwLock<Vec<Order>>>,
        fail_cancel: bool,
    }

    impl MockExchange {
        pub fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                orders: Arc::new(RwLock::new(Vec::new())),
                fail_cancel: false,
            }
        }

        pub fn with_fail_cancel(mut self) -> Self {
            self.fail_cancel = true;
            self
        }

        pub async fn add_order(&self, order: Order) {
            self.orders.write().await.push(order);
        }
    }

    #[async_trait::async_trait]
    impl ExchangeClient for MockExchange {
        async fn get_open_orders(&self) -> Result<Vec<Order>> {
            let orders = self.orders.read().await;
            Ok(orders
                .iter()
                .filter(|o| matches!(o.status, OrderStatus::Open))
                .cloned()
                .collect())
        }

        async fn cancel_order(&self, order_id: &str, _symbol: &str) -> Result<()> {
            if self.fail_cancel {
                return Err(anyhow::anyhow!("Mock cancel failure").into());
            }

            let mut orders = self.orders.write().await;
            if let Some(order) = orders.iter_mut().find(|o| o.id == order_id) {
                order.status = OrderStatus::Cancelled;
                Ok(())
            } else {
                Err(anyhow::anyhow!("Order not found").into())
            }
        }

        async fn cancel_all_orders(&self) -> Result<usize> {
            if self.fail_cancel {
                return Err(anyhow::anyhow!("Mock bulk cancel failure").into());
            }

            let mut orders = self.orders.write().await;
            let mut count = 0;
            for order in orders.iter_mut() {
                if matches!(order.status, OrderStatus::Open) {
                    order.status = OrderStatus::Cancelled;
                    count += 1;
                }
            }
            Ok(count)
        }

        fn exchange_name(&self) -> &str {
            &self.name
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mock_exchange::MockExchange;

    fn create_order(id: &str, symbol: &str) -> Order {
        Order {
            id: id.to_string(),
            exchange: "mock".to_string(),
            symbol: symbol.to_string(),
            side: "BUY".to_string(),
            quantity: 100.0,
            price: Some(50.0),
            status: OrderStatus::Open,
        }
    }

    #[tokio::test]
    async fn test_cancel_all_basic() {
        let cancel_all = CancelAll::default();
        let exchange = Arc::new(MockExchange::new("binance"));

        // Add some orders
        exchange.add_order(create_order("O1", "BTCUSD")).await;
        exchange.add_order(create_order("O2", "ETHUSDT")).await;
        exchange.add_order(create_order("O3", "SOLUSDT")).await;

        cancel_all.register_exchange(exchange).await;

        let result = cancel_all.trigger("Test trigger").await.unwrap();

        assert_eq!(result.cancelled_count, 3);
        assert_eq!(result.failed_count, 0);
        assert_eq!(result.total_orders, 3);
    }

    #[tokio::test]
    async fn test_cancel_all_multiple_exchanges() {
        let cancel_all = CancelAll::default();

        let exchange1 = Arc::new(MockExchange::new("binance"));
        exchange1.add_order(create_order("O1", "BTCUSD")).await;
        exchange1.add_order(create_order("O2", "ETHUSDT")).await;

        let exchange2 = Arc::new(MockExchange::new("coinbase"));
        exchange2.add_order(create_order("O3", "BTCUSD")).await;

        cancel_all.register_exchange(exchange1).await;
        cancel_all.register_exchange(exchange2).await;

        let result = cancel_all.trigger("Multi-exchange test").await.unwrap();

        assert_eq!(result.cancelled_count, 3);
        assert_eq!(result.exchange_results.len(), 2);
    }

    #[tokio::test]
    async fn test_cancel_all_with_failures() {
        let cancel_all = CancelAll::new(CancelAllConfig {
            retry_failed: false,
            max_retries: 1,
            ..Default::default()
        });

        let exchange = Arc::new(MockExchange::new("test").with_fail_cancel());
        exchange.add_order(create_order("O1", "BTCUSD")).await;

        cancel_all.register_exchange(exchange).await;

        let result = cancel_all.trigger("Failure test").await.unwrap();

        assert_eq!(result.cancelled_count, 0);
        assert!(result.failed_count > 0 || !result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_trigger_tracking() {
        let cancel_all = CancelAll::default();
        let exchange = Arc::new(MockExchange::new("test"));

        cancel_all.register_exchange(exchange).await;

        // Initial state
        let stats = cancel_all.stats().await;
        assert_eq!(stats.trigger_count, 0);
        assert!(stats.last_trigger.is_none());

        // Trigger once
        cancel_all.trigger("First").await.unwrap();

        let stats = cancel_all.stats().await;
        assert_eq!(stats.trigger_count, 1);
        assert!(stats.last_trigger.is_some());

        // Check recently triggered
        assert!(cancel_all.is_recently_triggered(10).await);
    }

    #[tokio::test]
    async fn test_reset() {
        let cancel_all = CancelAll::default();
        let exchange = Arc::new(MockExchange::new("test"));

        cancel_all.register_exchange(exchange).await;
        cancel_all.trigger("Test").await.unwrap();

        let stats = cancel_all.stats().await;
        assert_eq!(stats.trigger_count, 1);

        cancel_all.reset().await;

        let stats = cancel_all.stats().await;
        assert_eq!(stats.trigger_count, 0);
        assert!(stats.last_trigger.is_none());
    }
}
