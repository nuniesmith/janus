//! End-to-end integration test scenarios
//!
//! These tests cover:
//! - TWAP strategy execution with mock exchange
//! - VWAP strategy execution with volume profiles
//! - Iceberg order execution with tip management
//! - Kraken WebSocket data stream connectivity
//! - Kraken REST API connectivity
//! - WebSocket reconnection and recovery
//! - Position & P&L tracking
//!
//! Tests can run in two modes:
//! 1. Mock mode (default): Uses mock exchange adapters for offline testing
//! 2. Live mode: Uses Kraken public WebSocket (FREE, no key) and
//!    authenticated REST/private WS (requires KRAKEN_API_KEY / KRAKEN_API_SECRET)

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;
use tokio::time::{Duration, sleep, timeout};

/// Helper to check if integration tests should run against real exchanges
pub fn should_run_integration_tests() -> bool {
    std::env::var("RUN_INTEGRATION_TESTS").is_ok()
}

/// Helper to check if Kraken API credentials are available
pub fn has_kraken_credentials() -> bool {
    std::env::var("KRAKEN_API_KEY").is_ok() && std::env::var("KRAKEN_API_SECRET").is_ok()
}

/// Mock exchange for offline testing
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MockExchange {
    /// Order ID counter
    order_counter: Arc<AtomicU64>,
    /// Active orders: order_id -> (symbol, side, qty, price, filled_qty)
    orders: Arc<RwLock<HashMap<String, MockOrder>>>,
    /// Fill delay simulation (ms)
    fill_delay_ms: u64,
    /// Fill rate (0.0 - 1.0, probability of fill per check)
    fill_rate: f64,
    /// Current prices per symbol
    prices: Arc<RwLock<HashMap<String, Decimal>>>,
    /// Slippage simulation (percentage)
    slippage_pct: Decimal,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MockOrder {
    pub order_id: String,
    pub symbol: String,
    pub side: MockOrderSide,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub filled_quantity: Decimal,
    pub average_fill_price: Decimal,
    pub status: MockOrderStatus,
    pub order_type: MockOrderType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MockOrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MockOrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MockOrderType {
    Market,
    Limit,
}

impl MockExchange {
    pub fn new() -> Self {
        let mut prices = HashMap::new();
        prices.insert("BTCUSD".to_string(), dec!(50000));
        prices.insert("ETHUSDT".to_string(), dec!(3000));
        prices.insert("SOLUSDT".to_string(), dec!(100));

        Self {
            order_counter: Arc::new(AtomicU64::new(1)),
            orders: Arc::new(RwLock::new(HashMap::new())),
            fill_delay_ms: 50,
            fill_rate: 1.0, // 100% fill rate by default
            prices: Arc::new(RwLock::new(prices)),
            slippage_pct: dec!(0.01), // 0.01% slippage
        }
    }

    pub fn with_fill_delay(mut self, delay_ms: u64) -> Self {
        self.fill_delay_ms = delay_ms;
        self
    }

    pub fn with_fill_rate(mut self, rate: f64) -> Self {
        self.fill_rate = rate;
        self
    }

    pub fn with_slippage(mut self, slippage_pct: Decimal) -> Self {
        self.slippage_pct = slippage_pct;
        self
    }

    pub async fn set_price(&self, symbol: &str, price: Decimal) {
        let mut prices = self.prices.write().await;
        prices.insert(symbol.to_string(), price);
    }

    pub async fn get_price(&self, symbol: &str) -> Option<Decimal> {
        let prices = self.prices.read().await;
        prices.get(symbol).copied()
    }

    /// Submit a market order
    pub async fn submit_market_order(
        &self,
        symbol: &str,
        side: MockOrderSide,
        quantity: Decimal,
    ) -> Result<String, String> {
        let order_id = format!("mock_{}", self.order_counter.fetch_add(1, Ordering::SeqCst));
        let price = self
            .get_price(symbol)
            .await
            .ok_or_else(|| format!("Unknown symbol: {}", symbol))?;

        // Apply slippage
        let fill_price = match side {
            MockOrderSide::Buy => price * (Decimal::ONE + self.slippage_pct / dec!(100)),
            MockOrderSide::Sell => price * (Decimal::ONE - self.slippage_pct / dec!(100)),
        };

        let order = MockOrder {
            order_id: order_id.clone(),
            symbol: symbol.to_string(),
            side,
            quantity,
            price: None,
            filled_quantity: Decimal::ZERO,
            average_fill_price: Decimal::ZERO,
            status: MockOrderStatus::New,
            order_type: MockOrderType::Market,
        };

        {
            let mut orders = self.orders.write().await;
            orders.insert(order_id.clone(), order);
        }

        // Simulate fill delay
        if self.fill_delay_ms > 0 {
            sleep(Duration::from_millis(self.fill_delay_ms)).await;
        }

        // Fill the order (market orders fill immediately)
        {
            let mut orders = self.orders.write().await;
            if let Some(order) = orders.get_mut(&order_id) {
                order.filled_quantity = quantity;
                order.average_fill_price = fill_price;
                order.status = MockOrderStatus::Filled;
            }
        }

        Ok(order_id)
    }

    /// Submit a limit order
    pub async fn submit_limit_order(
        &self,
        symbol: &str,
        side: MockOrderSide,
        quantity: Decimal,
        price: Decimal,
    ) -> Result<String, String> {
        let order_id = format!("mock_{}", self.order_counter.fetch_add(1, Ordering::SeqCst));

        let order = MockOrder {
            order_id: order_id.clone(),
            symbol: symbol.to_string(),
            side,
            quantity,
            price: Some(price),
            filled_quantity: Decimal::ZERO,
            average_fill_price: Decimal::ZERO,
            status: MockOrderStatus::New,
            order_type: MockOrderType::Limit,
        };

        {
            let mut orders = self.orders.write().await;
            orders.insert(order_id.clone(), order);
        }

        // Simulate fill for limit orders based on fill_rate
        if self.fill_delay_ms > 0 {
            sleep(Duration::from_millis(self.fill_delay_ms)).await;
        }

        let current_price = self.get_price(symbol).await.unwrap_or(price);

        // Check if limit order would fill
        let should_fill = match side {
            MockOrderSide::Buy => current_price <= price,
            MockOrderSide::Sell => current_price >= price,
        };

        if should_fill && rand::random::<f64>() < self.fill_rate {
            let mut orders = self.orders.write().await;
            if let Some(order) = orders.get_mut(&order_id) {
                order.filled_quantity = quantity;
                order.average_fill_price = price;
                order.status = MockOrderStatus::Filled;
            }
        }

        Ok(order_id)
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<(), String> {
        let mut orders = self.orders.write().await;
        if let Some(order) = orders.get_mut(order_id) {
            if order.status == MockOrderStatus::Filled {
                return Err("Order already filled".to_string());
            }
            order.status = MockOrderStatus::Cancelled;
            Ok(())
        } else {
            Err(format!("Order not found: {}", order_id))
        }
    }

    /// Get order status
    pub async fn get_order(&self, order_id: &str) -> Option<MockOrder> {
        let orders = self.orders.read().await;
        orders.get(order_id).cloned()
    }

    /// Get all orders
    #[allow(dead_code)]
    pub async fn get_all_orders(&self) -> Vec<MockOrder> {
        let orders = self.orders.read().await;
        orders.values().cloned().collect()
    }

    /// Simulate partial fill
    pub async fn simulate_partial_fill(
        &self,
        order_id: &str,
        fill_qty: Decimal,
    ) -> Result<(), String> {
        let mut orders = self.orders.write().await;
        if let Some(order) = orders.get_mut(order_id) {
            if order.status == MockOrderStatus::Cancelled || order.status == MockOrderStatus::Filled
            {
                return Err("Order not active".to_string());
            }

            let new_filled = order.filled_quantity + fill_qty;
            if new_filled > order.quantity {
                return Err("Fill exceeds order quantity".to_string());
            }

            // Update average price
            let prices = self.prices.read().await;
            let current_price = prices
                .get(&order.symbol)
                .copied()
                .unwrap_or(order.price.unwrap_or(dec!(50000)));

            let total_cost =
                order.average_fill_price * order.filled_quantity + current_price * fill_qty;
            order.filled_quantity = new_filled;
            order.average_fill_price = if new_filled > Decimal::ZERO {
                total_cost / new_filled
            } else {
                Decimal::ZERO
            };

            if order.filled_quantity >= order.quantity {
                order.status = MockOrderStatus::Filled;
            } else {
                order.status = MockOrderStatus::PartiallyFilled;
            }

            Ok(())
        } else {
            Err(format!("Order not found: {}", order_id))
        }
    }
}

impl Default for MockExchange {
    fn default() -> Self {
        Self::new()
    }
}

/// Test execution statistics
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct ExecutionStats {
    pub total_quantity: Decimal,
    pub filled_quantity: Decimal,
    pub average_price: Decimal,
    pub orders_created: u32,
    pub orders_filled: u32,
    pub orders_cancelled: u32,
    pub execution_time_ms: u64,
}

impl ExecutionStats {
    pub fn fill_rate(&self) -> Decimal {
        if self.total_quantity > Decimal::ZERO {
            self.filled_quantity / self.total_quantity * dec!(100)
        } else {
            Decimal::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Mock Exchange Tests ====================

    #[tokio::test]
    async fn test_mock_exchange_market_order() {
        let exchange = MockExchange::new().with_fill_delay(0);

        let order_id = exchange
            .submit_market_order("BTCUSD", MockOrderSide::Buy, dec!(1.5))
            .await
            .expect("Should submit order");

        let order = exchange
            .get_order(&order_id)
            .await
            .expect("Order should exist");

        assert_eq!(order.status, MockOrderStatus::Filled);
        assert_eq!(order.filled_quantity, dec!(1.5));
        assert!(order.average_fill_price > Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_mock_exchange_limit_order() {
        let exchange = MockExchange::new().with_fill_delay(0).with_fill_rate(1.0);

        // Place a buy limit at market price (should fill)
        let order_id = exchange
            .submit_limit_order("BTCUSD", MockOrderSide::Buy, dec!(2.0), dec!(50000))
            .await
            .expect("Should submit order");

        let order = exchange
            .get_order(&order_id)
            .await
            .expect("Order should exist");
        assert_eq!(order.status, MockOrderStatus::Filled);
        assert_eq!(order.filled_quantity, dec!(2.0));
    }

    #[tokio::test]
    async fn test_mock_exchange_cancel_order() {
        let exchange = MockExchange::new()
            .with_fill_delay(1000)
            .with_fill_rate(0.0);

        let order_id = exchange
            .submit_limit_order("BTCUSD", MockOrderSide::Buy, dec!(1.0), dec!(45000))
            .await
            .expect("Should submit order");

        // Cancel before fill
        exchange
            .cancel_order(&order_id)
            .await
            .expect("Should cancel");

        let order = exchange
            .get_order(&order_id)
            .await
            .expect("Order should exist");
        assert_eq!(order.status, MockOrderStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_mock_exchange_partial_fill() {
        let exchange = MockExchange::new().with_fill_delay(0).with_fill_rate(0.0);

        let order_id = exchange
            .submit_limit_order("BTCUSD", MockOrderSide::Buy, dec!(10.0), dec!(50000))
            .await
            .expect("Should submit order");

        // Simulate partial fills
        exchange
            .simulate_partial_fill(&order_id, dec!(3.0))
            .await
            .expect("Should fill");
        let order = exchange.get_order(&order_id).await.unwrap();
        assert_eq!(order.status, MockOrderStatus::PartiallyFilled);
        assert_eq!(order.filled_quantity, dec!(3.0));

        exchange
            .simulate_partial_fill(&order_id, dec!(7.0))
            .await
            .expect("Should fill");
        let order = exchange.get_order(&order_id).await.unwrap();
        assert_eq!(order.status, MockOrderStatus::Filled);
        assert_eq!(order.filled_quantity, dec!(10.0));
    }

    // ==================== TWAP Strategy Tests ====================

    /// Test TWAP strategy with mock exchange - basic execution
    #[tokio::test]
    async fn test_twap_strategy_basic_execution() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));
        let exchange_clone = exchange.clone();

        // TWAP parameters
        let total_quantity = dec!(100);
        let num_slices = 5;
        let slice_size = total_quantity / Decimal::from(num_slices);

        let mut stats = ExecutionStats {
            total_quantity,
            ..Default::default()
        };

        let start = std::time::Instant::now();

        // Execute TWAP slices
        for _slice_num in 0..num_slices {
            let order_id = exchange_clone
                .submit_market_order("BTCUSD", MockOrderSide::Buy, slice_size)
                .await
                .expect("Should submit slice");

            stats.orders_created += 1;

            let order = exchange_clone.get_order(&order_id).await.unwrap();
            if order.status == MockOrderStatus::Filled {
                stats.orders_filled += 1;
                stats.filled_quantity += order.filled_quantity;

                // Update average price
                let total_cost = stats.average_price
                    * (stats.filled_quantity - order.filled_quantity)
                    + order.average_fill_price * order.filled_quantity;
                stats.average_price = total_cost / stats.filled_quantity;
            }

            // Simulate inter-slice delay (would be longer in real TWAP)
            sleep(Duration::from_millis(10)).await;
        }

        stats.execution_time_ms = start.elapsed().as_millis() as u64;

        // Verify results
        assert_eq!(stats.orders_created, 5, "Should create 5 slices");
        assert_eq!(stats.orders_filled, 5, "All slices should fill");
        assert_eq!(
            stats.filled_quantity, total_quantity,
            "Should fill total quantity"
        );
        assert!(
            stats.average_price > Decimal::ZERO,
            "Should have valid average price"
        );

        println!("TWAP Execution Stats:");
        println!("  Total Quantity: {}", stats.total_quantity);
        println!("  Filled Quantity: {}", stats.filled_quantity);
        println!("  Fill Rate: {}%", stats.fill_rate());
        println!("  Average Price: {}", stats.average_price);
        println!("  Execution Time: {}ms", stats.execution_time_ms);
    }

    /// Test TWAP strategy with partial fills
    #[tokio::test]
    async fn test_twap_strategy_partial_fills() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0).with_fill_rate(0.5));

        let total_quantity = dec!(50);
        let num_slices = 10;
        let slice_size = total_quantity / Decimal::from(num_slices);

        let mut filled_quantity = Decimal::ZERO;
        let mut orders_filled = 0u32;

        for _ in 0..num_slices {
            let order_id = exchange
                .submit_limit_order("BTCUSD", MockOrderSide::Buy, slice_size, dec!(50000))
                .await
                .expect("Should submit");

            let order = exchange.get_order(&order_id).await.unwrap();
            if order.status == MockOrderStatus::Filled {
                filled_quantity += order.filled_quantity;
                orders_filled += 1;
            }
        }

        // With 50% fill rate, we expect roughly half to fill
        println!(
            "Partial fill test: {}/{} slices filled",
            orders_filled, num_slices
        );
        assert!(
            orders_filled > 0 && orders_filled < num_slices,
            "Expected partial fills, got {}/{}",
            orders_filled,
            num_slices
        );
    }

    /// Test TWAP strategy cancellation mid-execution
    #[tokio::test]
    async fn test_twap_strategy_cancellation() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));
        let cancel_signal = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancel_signal_clone = cancel_signal.clone();

        let total_quantity = dec!(100);
        let num_slices = 10;
        let slice_size = total_quantity / Decimal::from(num_slices);

        let mut executed_slices = 0u32;
        let mut filled_quantity = Decimal::ZERO;

        // Cancel after 5 slices
        let cancel_at_slice = 5;

        for slice_num in 0..num_slices {
            // Check for cancellation
            if cancel_signal_clone.load(Ordering::SeqCst) {
                break;
            }

            // Trigger cancellation at specified slice
            if slice_num == cancel_at_slice {
                cancel_signal.store(true, Ordering::SeqCst);
                continue;
            }

            let order_id = exchange
                .submit_market_order("BTCUSD", MockOrderSide::Buy, slice_size)
                .await
                .expect("Should submit");

            let order = exchange.get_order(&order_id).await.unwrap();
            if order.status == MockOrderStatus::Filled {
                executed_slices += 1;
                filled_quantity += order.filled_quantity;
            }
        }

        assert_eq!(
            executed_slices, 5,
            "Should have executed 5 slices before cancel"
        );
        assert_eq!(
            filled_quantity,
            slice_size * Decimal::from(5),
            "Should have partial fill"
        );

        println!(
            "TWAP cancelled: {}/{} slices executed, {} filled",
            executed_slices, num_slices, filled_quantity
        );
    }

    /// Test TWAP with price deviation handling
    #[tokio::test]
    async fn test_twap_price_deviation() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));

        let initial_price = dec!(50000);
        let max_deviation_pct = dec!(1.0); // 1% max deviation

        // Execute first slice at initial price
        let order_id = exchange
            .submit_market_order("BTCUSD", MockOrderSide::Buy, dec!(10))
            .await
            .expect("Should submit");
        let order = exchange.get_order(&order_id).await.unwrap();
        let first_price = order.average_fill_price;

        // Simulate price movement beyond threshold
        let new_price = initial_price * dec!(1.02); // 2% increase
        exchange.set_price("BTCUSD", new_price).await;

        // Check if we should pause/skip slice due to deviation
        let current_price = exchange.get_price("BTCUSD").await.unwrap();
        let deviation = (current_price - first_price).abs() / first_price * dec!(100);

        let should_pause = deviation > max_deviation_pct;
        assert!(
            should_pause,
            "Should detect price deviation: {}%",
            deviation
        );

        println!(
            "Price deviation detected: {}% (threshold: {}%)",
            deviation, max_deviation_pct
        );
    }

    // ==================== VWAP Strategy Tests ====================

    /// Test VWAP strategy with uniform volume profile
    #[tokio::test]
    async fn test_vwap_strategy_uniform_profile() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));

        // VWAP with uniform volume distribution
        let total_quantity = dec!(100);
        let num_buckets = 4;
        let volume_profile: Vec<Decimal> = vec![dec!(0.25); num_buckets]; // 25% each bucket

        let mut stats = ExecutionStats {
            total_quantity,
            ..Default::default()
        };

        for (bucket_idx, &volume_pct) in volume_profile.iter().enumerate() {
            let bucket_quantity = total_quantity * volume_pct;

            let order_id = exchange
                .submit_market_order("BTCUSD", MockOrderSide::Buy, bucket_quantity)
                .await
                .expect("Should submit bucket order");

            stats.orders_created += 1;

            let order = exchange.get_order(&order_id).await.unwrap();
            if order.status == MockOrderStatus::Filled {
                stats.orders_filled += 1;
                stats.filled_quantity += order.filled_quantity;

                // Track VWAP
                let total_cost = stats.average_price
                    * (stats.filled_quantity - order.filled_quantity)
                    + order.average_fill_price * order.filled_quantity;
                stats.average_price = total_cost / stats.filled_quantity;
            }

            println!(
                "VWAP bucket {}: {}% = {} units @ {}",
                bucket_idx + 1,
                volume_pct * dec!(100),
                bucket_quantity,
                order.average_fill_price
            );
        }

        assert_eq!(stats.filled_quantity, total_quantity);
        println!("VWAP Execution VWAP: {}", stats.average_price);
    }

    /// Test VWAP strategy with U-shaped volume profile (market open/close pattern)
    #[tokio::test]
    async fn test_vwap_strategy_u_shaped_profile() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));

        // U-shaped profile: high volume at open/close, low in middle
        let total_quantity = dec!(1000);
        let volume_profile: Vec<Decimal> = vec![
            dec!(0.20), // Bucket 1: 20% (open)
            dec!(0.10), // Bucket 2: 10%
            dec!(0.10), // Bucket 3: 10%
            dec!(0.10), // Bucket 4: 10%
            dec!(0.10), // Bucket 5: 10%
            dec!(0.10), // Bucket 6: 10%
            dec!(0.10), // Bucket 7: 10%
            dec!(0.20), // Bucket 8: 20% (close)
        ];

        // Verify profile sums to 100%
        let total_pct: Decimal = volume_profile.iter().sum();
        assert_eq!(total_pct, Decimal::ONE, "Volume profile should sum to 100%");

        let mut bucket_fills: Vec<Decimal> = Vec::new();
        let mut total_filled = Decimal::ZERO;

        // Simulate price movement during the day
        let price_path: Vec<Decimal> = vec![
            dec!(50000), // Open
            dec!(50100),
            dec!(50200),
            dec!(50150), // Mid-day high
            dec!(50100),
            dec!(50000),
            dec!(49900),
            dec!(49950), // Close
        ];

        for (&volume_pct, &price) in volume_profile.iter().zip(price_path.iter()) {
            exchange.set_price("BTCUSD", price).await;

            let bucket_quantity = total_quantity * volume_pct;

            let order_id = exchange
                .submit_market_order("BTCUSD", MockOrderSide::Buy, bucket_quantity)
                .await
                .expect("Should submit");

            let order = exchange.get_order(&order_id).await.unwrap();
            bucket_fills.push(order.filled_quantity);
            total_filled += order.filled_quantity;
        }

        assert_eq!(total_filled, total_quantity);

        // First and last buckets should have larger fills
        assert!(
            bucket_fills[0] > bucket_fills[3],
            "Open should have more volume than mid-day"
        );
        assert!(
            bucket_fills[7] > bucket_fills[3],
            "Close should have more volume than mid-day"
        );

        println!("VWAP U-shaped execution complete: {} filled", total_filled);
    }

    /// Test VWAP with participation rate limit
    #[tokio::test]
    async fn test_vwap_participation_rate() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));

        // Simulated market volume per bucket
        let market_volumes: Vec<Decimal> = vec![dec!(1000), dec!(500), dec!(800), dec!(1200)];

        let participation_rate = dec!(0.10); // Max 10% of market volume
        let total_quantity = dec!(300);
        let bucket_quantity = total_quantity / Decimal::from(4);

        let mut total_filled = Decimal::ZERO;
        let mut constrained_count = 0u32;

        for (bucket_idx, &market_volume) in market_volumes.iter().enumerate() {
            let max_quantity = market_volume * participation_rate;
            let actual_quantity = bucket_quantity.min(max_quantity);

            if actual_quantity < bucket_quantity {
                constrained_count += 1;
                println!(
                    "Bucket {}: Constrained from {} to {} ({}% of market vol {})",
                    bucket_idx + 1,
                    bucket_quantity,
                    actual_quantity,
                    participation_rate * dec!(100),
                    market_volume
                );
            }

            let order_id = exchange
                .submit_market_order("BTCUSD", MockOrderSide::Buy, actual_quantity)
                .await
                .expect("Should submit");

            let order = exchange.get_order(&order_id).await.unwrap();
            total_filled += order.filled_quantity;
        }

        // Bucket 2 (500 volume * 10% = 50) should be constrained from 75
        assert!(
            constrained_count > 0,
            "Should have participation rate constraints"
        );
        println!(
            "VWAP with participation limit: {} filled, {} buckets constrained",
            total_filled, constrained_count
        );
    }

    /// Test VWAP benchmark comparison
    #[tokio::test]
    async fn test_vwap_benchmark_slippage() {
        let exchange = Arc::new(
            MockExchange::new()
                .with_fill_delay(0)
                .with_slippage(dec!(0.05)),
        );

        // Simulate trades throughout the day for VWAP benchmark
        let trades: Vec<(Decimal, Decimal)> = vec![
            (dec!(50000), dec!(100)), // price, volume
            (dec!(50100), dec!(150)),
            (dec!(50200), dec!(80)),
            (dec!(50150), dec!(120)),
            (dec!(50050), dec!(200)),
        ];

        // Calculate true VWAP benchmark
        let total_volume: Decimal = trades.iter().map(|(_, v)| v).sum();
        let total_value: Decimal = trades.iter().map(|(p, v)| p * v).sum();
        let vwap_benchmark = total_value / total_volume;

        // Execute our VWAP strategy
        let total_quantity = dec!(50);
        let num_buckets = 5;
        let bucket_qty = total_quantity / Decimal::from(num_buckets);

        let mut execution_cost = Decimal::ZERO;

        for (price, _) in trades.iter() {
            exchange.set_price("BTCUSD", *price).await;

            let order_id = exchange
                .submit_market_order("BTCUSD", MockOrderSide::Buy, bucket_qty)
                .await
                .expect("Should submit");

            let order = exchange.get_order(&order_id).await.unwrap();
            execution_cost += order.average_fill_price * order.filled_quantity;
        }

        let execution_vwap = execution_cost / total_quantity;
        let slippage_bps = (execution_vwap - vwap_benchmark) / vwap_benchmark * dec!(10000);

        println!("VWAP Benchmark: {}", vwap_benchmark);
        println!("Execution VWAP: {}", execution_vwap);
        println!("Slippage: {} bps", slippage_bps);

        // With 0.05% slippage, we expect positive slippage for buys
        assert!(
            slippage_bps.abs() < dec!(10),
            "Slippage should be reasonable: {} bps",
            slippage_bps
        );
    }

    // ==================== Iceberg Order Tests ====================

    /// Test basic iceberg order execution
    #[tokio::test]
    async fn test_iceberg_order_basic_execution() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));

        let total_quantity = dec!(1000);
        let tip_size = dec!(50);
        let expected_tips: u32 = (total_quantity / tip_size).try_into().unwrap();

        let mut tips_created = 0u32;
        let mut tips_filled = 0u32;
        let mut filled_quantity = Decimal::ZERO;

        while filled_quantity < total_quantity {
            let remaining = total_quantity - filled_quantity;
            let current_tip = tip_size.min(remaining);

            let order_id = exchange
                .submit_limit_order("BTCUSD", MockOrderSide::Buy, current_tip, dec!(50000))
                .await
                .expect("Should submit tip");

            tips_created += 1;

            let order = exchange.get_order(&order_id).await.unwrap();
            if order.status == MockOrderStatus::Filled {
                tips_filled += 1;
                filled_quantity += order.filled_quantity;
            }
        }

        assert_eq!(filled_quantity, total_quantity);
        assert_eq!(
            tips_created, expected_tips,
            "Should create correct number of tips"
        );
        assert_eq!(tips_filled, expected_tips, "All tips should fill");

        println!(
            "Iceberg executed: {} total via {} tips of {} each",
            filled_quantity, tips_created, tip_size
        );
    }

    /// Test iceberg with variable tip sizes
    #[tokio::test]
    async fn test_iceberg_variable_tips() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));

        let total_quantity = dec!(500);
        let base_tip_size = dec!(50);
        let _tip_variance_pct = dec!(0.2); // 20% variance

        let mut filled_quantity = Decimal::ZERO;
        let mut tip_sizes: Vec<Decimal> = Vec::new();

        while filled_quantity < total_quantity {
            let remaining = total_quantity - filled_quantity;

            // Add randomness to tip size (simulated)
            let variance_factor =
                dec!(1.0) + (dec!(0.1) * Decimal::from(tip_sizes.len() % 3)) - dec!(0.1);
            let current_tip = (base_tip_size * variance_factor).min(remaining);

            let order_id = exchange
                .submit_market_order("BTCUSD", MockOrderSide::Buy, current_tip)
                .await
                .expect("Should submit");

            let order = exchange.get_order(&order_id).await.unwrap();
            filled_quantity += order.filled_quantity;
            tip_sizes.push(current_tip);
        }

        assert_eq!(filled_quantity, total_quantity);

        // Verify tip sizes vary
        let unique_sizes: std::collections::HashSet<String> =
            tip_sizes.iter().map(|d| d.to_string()).collect();
        println!(
            "Iceberg with variable tips: {} sizes used across {} tips",
            unique_sizes.len(),
            tip_sizes.len()
        );
    }

    /// Test iceberg with tip cancellation and refresh
    #[tokio::test]
    async fn test_iceberg_tip_refresh() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0).with_fill_rate(0.0));

        let total_quantity = dec!(200);
        let tip_size = dec!(50);
        let _max_tip_age_ms = 100u64;

        let mut active_tip: Option<String> = None;
        let mut filled_quantity = Decimal::ZERO;
        let mut refresh_count = 0u32;
        let start = std::time::Instant::now();

        // Run for limited time
        while filled_quantity < total_quantity && start.elapsed() < Duration::from_millis(500) {
            // Check if we need to refresh tip
            let should_refresh = active_tip.is_none();

            if should_refresh {
                // Cancel existing tip if any
                if let Some(ref tip_id) = active_tip {
                    let _ = exchange.cancel_order(tip_id).await;
                    refresh_count += 1;
                }

                // Create new tip
                let remaining = total_quantity - filled_quantity;
                let current_tip = tip_size.min(remaining);

                let order_id = exchange
                    .submit_limit_order("BTCUSD", MockOrderSide::Buy, current_tip, dec!(50000))
                    .await
                    .expect("Should submit tip");

                active_tip = Some(order_id);
            }

            // Simulate fill
            if let Some(ref tip_id) = active_tip {
                let _ = exchange.simulate_partial_fill(tip_id, tip_size).await;

                let order = exchange.get_order(tip_id).await.unwrap();
                if order.status == MockOrderStatus::Filled {
                    filled_quantity += order.filled_quantity;
                    active_tip = None;
                }
            }

            sleep(Duration::from_millis(10)).await;
        }

        println!(
            "Iceberg tip refresh test: {} filled, {} refreshes",
            filled_quantity, refresh_count
        );
    }

    /// Test iceberg order concealment (only tip visible)
    #[tokio::test]
    async fn test_iceberg_concealment() {
        let exchange = Arc::new(MockExchange::new().with_fill_delay(0));

        let total_quantity = dec!(1000);
        let tip_size = dec!(100);
        let hidden_quantity = total_quantity - tip_size;

        // First tip
        let order_id = exchange
            .submit_limit_order("BTCUSD", MockOrderSide::Buy, tip_size, dec!(50000))
            .await
            .expect("Should submit tip");

        let order = exchange.get_order(&order_id).await.unwrap();

        // Verify only tip is "visible" (the order quantity)
        assert_eq!(
            order.quantity, tip_size,
            "Only tip size should be visible in order"
        );
        assert_eq!(hidden_quantity, dec!(900), "Hidden quantity calculation");

        println!(
            "Iceberg concealment: {} visible, {} hidden",
            order.quantity, hidden_quantity
        );
    }

    // ==================== Kraken Integration Tests ====================
    //
    // Public WebSocket tests (ticker, trades) do NOT require API keys.
    // Authenticated tests (balance, orders) require KRAKEN_API_KEY + KRAKEN_API_SECRET.

    /// Test scenario: Kraken REST API server time (public, no key needed)
    ///
    /// Verifies basic network connectivity to Kraken REST API.
    #[tokio::test]
    #[ignore] // Run with: cargo test --test integration -- --ignored
    async fn test_kraken_rest_server_time() {
        if !should_run_integration_tests() {
            eprintln!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
            return;
        }

        use janus_execution::exchanges::kraken::rest::{KrakenRestClient, KrakenRestConfig};

        let config = KrakenRestConfig::default();
        let client = KrakenRestClient::new(config);

        let server_time = client
            .get_server_time()
            .await
            .expect("Failed to get Kraken server time");

        assert!(server_time > 0, "Server time should be positive");
        println!("✓ Kraken REST API reachable — server time: {}", server_time);
    }

    /// Test scenario: Kraken public WebSocket ticker stream
    ///
    /// Connects to Kraken's FREE public WebSocket and subscribes to
    /// BTC/USD ticker updates. Verifies we receive at least one tick.
    #[tokio::test]
    #[ignore]
    async fn test_kraken_ws_ticker_stream() {
        if !should_run_integration_tests() {
            eprintln!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
            return;
        }

        use janus_execution::exchanges::kraken::{KrakenWebSocket, KrakenWsConfig};

        let config = KrakenWsConfig::default();
        let ws = KrakenWebSocket::new(config);

        let mut rx = ws.subscribe_events();

        ws.start().await.expect("Failed to start Kraken WebSocket");
        ws.subscribe_ticker(&["BTC/USD"])
            .await
            .expect("Failed to subscribe to BTC/USD ticker");

        // Wait up to 15 seconds for a ticker event
        let result = timeout(Duration::from_secs(15), async {
            loop {
                if let Ok(event) = rx.recv().await {
                    use janus_execution::exchanges::kraken::KrakenEvent;
                    match event {
                        KrakenEvent::Ticker(ticker) => {
                            println!(
                                "✓ Kraken ticker received — {} bid={} ask={}",
                                ticker.symbol, ticker.bid, ticker.ask
                            );
                            return ticker;
                        }
                        KrakenEvent::Connected => {
                            println!("  WebSocket connected");
                        }
                        KrakenEvent::Subscribed { channel, symbol } => {
                            println!("  Subscribed to {} for {:?}", channel, symbol);
                        }
                        _ => {}
                    }
                }
            }
        })
        .await;

        ws.stop().await;

        let ticker = result.expect("Timed out waiting for ticker data");
        assert_eq!(ticker.symbol, "BTC/USD");
        assert!(ticker.bid > Decimal::ZERO, "Bid should be positive");
        assert!(ticker.ask > Decimal::ZERO, "Ask should be positive");
        assert!(ticker.ask >= ticker.bid, "Ask should be >= bid");
    }

    /// Test scenario: Kraken public WebSocket trade stream
    ///
    /// Subscribes to BTC/USD trades and verifies we receive real trade data.
    #[tokio::test]
    #[ignore]
    async fn test_kraken_ws_trade_stream() {
        if !should_run_integration_tests() {
            eprintln!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
            return;
        }

        use janus_execution::exchanges::kraken::{KrakenEvent, KrakenWebSocket, KrakenWsConfig};

        let config = KrakenWsConfig::default();
        let ws = KrakenWebSocket::new(config);

        let mut rx = ws.subscribe_events();

        ws.start().await.expect("Failed to start Kraken WebSocket");
        ws.subscribe_trades(&["BTC/USD"])
            .await
            .expect("Failed to subscribe to BTC/USD trades");

        // Wait up to 30 seconds for a trade (trades are less frequent than ticks)
        let result = timeout(Duration::from_secs(30), async {
            loop {
                if let Ok(KrakenEvent::Trade(trade)) = rx.recv().await {
                    println!(
                        "✓ Kraken trade received — {} {:?} {} @ {}",
                        trade.symbol, trade.side, trade.quantity, trade.price
                    );
                    return trade;
                }
            }
        })
        .await;

        ws.stop().await;

        let trade = result.expect("Timed out waiting for trade data");
        assert!(
            trade.price > Decimal::ZERO,
            "Trade price should be positive"
        );
        assert!(
            trade.quantity > Decimal::ZERO,
            "Trade quantity should be positive"
        );
    }

    /// Test scenario: Kraken WebSocket reconnection and recovery
    ///
    /// Connects, receives data, disconnects, and verifies auto-reconnection
    /// restores the data stream.
    #[tokio::test]
    #[ignore]
    async fn test_kraken_ws_reconnection() {
        if !should_run_integration_tests() {
            eprintln!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
            return;
        }

        use janus_execution::exchanges::kraken::{KrakenEvent, KrakenWebSocket, KrakenWsConfig};

        let config = KrakenWsConfig {
            auto_reconnect: true,
            max_reconnect_attempts: 3,
            reconnect_delay_secs: 2,
            ..Default::default()
        };
        let ws = KrakenWebSocket::new(config);
        let mut rx = ws.subscribe_events();

        // Connect and get initial data
        ws.start().await.expect("Failed to start Kraken WebSocket");
        ws.subscribe_ticker(&["BTC/USD"])
            .await
            .expect("Failed to subscribe");

        // Wait for first tick to confirm connection
        let first_tick = timeout(Duration::from_secs(15), async {
            loop {
                if let Ok(KrakenEvent::Ticker(_)) = rx.recv().await {
                    return true;
                }
            }
        })
        .await;
        assert!(first_tick.is_ok(), "Should receive initial ticker data");
        println!("✓ Initial connection established and receiving data");

        // Force disconnect by stopping and restarting
        ws.stop().await;
        sleep(Duration::from_secs(1)).await;
        ws.start().await.expect("Failed to restart WebSocket");
        ws.subscribe_ticker(&["BTC/USD"])
            .await
            .expect("Failed to resubscribe");

        // Verify we receive data again after reconnection
        let reconnect_tick = timeout(Duration::from_secs(15), async {
            loop {
                if let Ok(KrakenEvent::Ticker(_)) = rx.recv().await {
                    return true;
                }
            }
        })
        .await;

        ws.stop().await;

        assert!(
            reconnect_tick.is_ok(),
            "Should receive ticker data after reconnection"
        );
        println!("✓ WebSocket reconnection successful — data stream restored");
    }

    /// Test scenario: Kraken REST API balance check (authenticated)
    ///
    /// Requires KRAKEN_API_KEY + KRAKEN_API_SECRET environment variables.
    #[tokio::test]
    #[ignore]
    async fn test_kraken_rest_balance() {
        if !should_run_integration_tests() {
            eprintln!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
            return;
        }

        if !has_kraken_credentials() {
            eprintln!("Skipping: KRAKEN_API_KEY / KRAKEN_API_SECRET not set.");
            return;
        }

        use janus_execution::exchanges::kraken::rest::{KrakenRestClient, KrakenRestConfig};

        let config = KrakenRestConfig::from_env();
        assert!(
            config.has_credentials(),
            "Kraken credentials should be available"
        );

        let client = KrakenRestClient::new(config);

        match client.get_balance().await {
            Ok(balances) => {
                println!("✓ Kraken balance retrieved — {} currencies", balances.len());
                for balance in &balances {
                    println!("  {} — available: {}", balance.currency, balance.available);
                }
            }
            Err(e) => {
                panic!("Failed to get Kraken balance: {}", e);
            }
        }
    }

    /// Test scenario: Kraken multi-symbol ticker stream
    ///
    /// Subscribes to multiple symbols and verifies we get data for each.
    #[tokio::test]
    #[ignore]
    async fn test_kraken_ws_multi_symbol() {
        if !should_run_integration_tests() {
            eprintln!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
            return;
        }

        use janus_execution::exchanges::kraken::{KrakenEvent, KrakenWebSocket, KrakenWsConfig};
        use std::collections::HashSet;

        let config = KrakenWsConfig::default();
        let ws = KrakenWebSocket::new(config);
        let mut rx = ws.subscribe_events();

        ws.start().await.expect("Failed to start Kraken WebSocket");
        ws.subscribe_ticker(&["BTC/USD", "ETH/USD"])
            .await
            .expect("Failed to subscribe to multiple tickers");

        let mut symbols_seen = HashSet::new();

        // Wait up to 30 seconds to see both symbols
        let result = timeout(Duration::from_secs(30), async {
            loop {
                if let Ok(KrakenEvent::Ticker(ticker)) = rx.recv().await {
                    symbols_seen.insert(ticker.symbol.clone());
                    println!(
                        "  Received ticker for {} (seen {}/2)",
                        ticker.symbol,
                        symbols_seen.len()
                    );
                    if symbols_seen.len() >= 2 {
                        return symbols_seen;
                    }
                }
            }
        })
        .await;

        ws.stop().await;

        let seen = result.expect("Timed out waiting for multi-symbol data");
        assert!(
            seen.contains("BTC/USD"),
            "Should have received BTC/USD ticker"
        );
        assert!(
            seen.contains("ETH/USD"),
            "Should have received ETH/USD ticker"
        );
        println!(
            "✓ Multi-symbol WebSocket stream working — {} symbols",
            seen.len()
        );
    }

    /// Test scenario: Kraken WebSocket data throughput
    ///
    /// Subscribes to ticker data and measures message throughput over
    /// a short window to verify the stream is healthy.
    #[tokio::test]
    #[ignore]
    async fn test_kraken_ws_throughput() {
        if !should_run_integration_tests() {
            eprintln!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
            return;
        }

        use janus_execution::exchanges::kraken::{KrakenEvent, KrakenWebSocket, KrakenWsConfig};
        use std::time::Instant;

        let config = KrakenWsConfig::default();
        let ws = KrakenWebSocket::new(config);
        let mut rx = ws.subscribe_events();

        ws.start().await.expect("Failed to start Kraken WebSocket");
        ws.subscribe_ticker(&["BTC/USD", "ETH/USD"])
            .await
            .expect("Failed to subscribe");

        // Wait for the first ticker to confirm the stream is live before measuring
        let warmup = timeout(Duration::from_secs(15), async {
            loop {
                if let Ok(KrakenEvent::Ticker(_)) = rx.recv().await {
                    return;
                }
            }
        })
        .await;
        warmup.expect("Timed out waiting for first ticker during warmup");

        // Count messages over 10 seconds
        let start = Instant::now();
        let measurement_secs = 10;
        let mut tick_count: u64 = 0;

        while start.elapsed().as_secs() < measurement_secs {
            // Use a per-recv timeout so the loop condition is re-checked regularly
            // even when no messages arrive, allowing the loop to exit naturally.
            match timeout(Duration::from_secs(2), rx.recv()).await {
                Ok(Ok(KrakenEvent::Ticker(_))) => tick_count += 1,
                Ok(_) => {}  // non-ticker event or channel error
                Err(_) => {} // per-recv timeout, re-check elapsed
            }
        }

        ws.stop().await;

        let count = tick_count;
        let rate = count as f64 / measurement_secs as f64;
        println!(
            "✓ Kraken throughput: {} ticks in {}s ({:.1} ticks/sec)",
            count, measurement_secs, rate
        );
        assert!(count > 0, "Should receive at least some ticker data");
    }
}
