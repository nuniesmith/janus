//! # Instrumented Execution Manager
//!
//! Provides an execution manager with integrated Prometheus metrics collection.
//! This wraps the standard ExecutionManager and automatically records all
//! execution events to Prometheus for monitoring and alerting.

#[cfg_attr(not(test), allow(unused_imports))]
use super::analytics::Side;
use super::analytics::{ExecutionAnalytics, ExecutionRecord, ExecutionReport};
use super::manager::{ExecutionManager, ExecutionStrategy, OrderId, OrderRequest, OrderStatus};
use super::metrics::{EXECUTION_METRICS, ExecutionMetrics};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
#[cfg_attr(not(test), allow(unused_imports))]
use std::time::Duration;
use std::time::Instant;

/// Instrumented execution manager with Prometheus metrics
pub struct InstrumentedExecutionManager {
    /// Underlying execution manager
    manager: ExecutionManager,

    /// Execution analytics
    analytics: ExecutionAnalytics,

    /// Metrics registry
    metrics: Arc<ExecutionMetrics>,

    /// Order start times (for time-to-fill tracking)
    order_start_times: HashMap<OrderId, Instant>,

    /// Strategy counters (for tracking order types)
    strategy_counters: StrategyCounters,

    /// Health status
    health: Arc<RwLock<HealthStatus>>,
}

/// Strategy usage counters
#[derive(Default)]
struct StrategyCounters {
    market: u64,
    limit: u64,
    twap: u64,
    vwap: u64,
}

/// Health status for monitoring
#[derive(Clone)]
pub struct HealthStatus {
    /// Is the execution manager healthy?
    pub healthy: bool,

    /// Total orders submitted
    pub total_orders: u64,

    /// Total orders completed
    pub completed_orders: u64,

    /// Total orders failed
    pub failed_orders: u64,

    /// Average execution quality score
    pub avg_quality_score: f64,

    /// Recent error messages
    pub recent_errors: Vec<String>,

    /// Last update timestamp
    pub last_update: Instant,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            healthy: true,
            total_orders: 0,
            completed_orders: 0,
            failed_orders: 0,
            avg_quality_score: 100.0,
            recent_errors: Vec::new(),
            last_update: Instant::now(),
        }
    }
}

impl InstrumentedExecutionManager {
    /// Create a new instrumented execution manager
    pub fn new() -> Self {
        Self {
            manager: ExecutionManager::new(),
            analytics: ExecutionAnalytics::new(),
            metrics: EXECUTION_METRICS.clone(),
            order_start_times: HashMap::new(),
            strategy_counters: StrategyCounters::default(),
            health: Arc::new(RwLock::new(HealthStatus::default())),
        }
    }

    /// Submit an order and record metrics
    pub fn submit_order(&mut self, request: OrderRequest) -> OrderId {
        let start = Instant::now();
        let order_id = self.manager.submit_order(request.clone());

        // Record order start time
        self.order_start_times.insert(order_id.clone(), start);

        // Update strategy counters and metrics
        match &request.strategy {
            ExecutionStrategy::Market => {
                self.strategy_counters.market += 1;
                self.metrics.record_market();
            }
            ExecutionStrategy::Limit { .. } => {
                self.strategy_counters.limit += 1;
                self.metrics.record_limit();
            }
            ExecutionStrategy::TWAP { .. } => {
                self.strategy_counters.twap += 1;
                self.metrics.record_twap();
            }
            ExecutionStrategy::VWAP { .. } => {
                self.strategy_counters.vwap += 1;
                self.metrics.record_vwap();
            }
            ExecutionStrategy::POV { .. } => {
                // POV treated similarly to VWAP for metrics purposes
                self.strategy_counters.vwap += 1;
                self.metrics.record_vwap();
            }
        }

        // Update health status
        if let Ok(mut health) = self.health.write() {
            health.total_orders += 1;
            health.last_update = Instant::now();
        }

        order_id
    }

    /// Process pending orders and update metrics
    pub fn process(&mut self) {
        let start = Instant::now();

        // Get order statuses before processing
        let orders_before: Vec<_> = self
            .manager
            .active_orders()
            .iter()
            .map(|s| s.order_id.clone())
            .collect();

        // Process orders
        self.manager.process();

        // Check for completed orders
        for order_id in orders_before {
            if let Some(status) = self.manager.get_order_status(&order_id) {
                // Clone status to avoid borrow checker issues
                let status_clone = status.clone();

                // Check if order completed
                if status_clone.is_complete() && status_clone.filled_quantity > 0.0 {
                    self.on_order_complete(&order_id, &status_clone);
                }

                // Check if order failed
                if matches!(status_clone.state, super::manager::OrderState::Failed) {
                    self.on_order_failed(&order_id, &status_clone);
                }
            }
        }

        // Record processing latency
        let latency_us = start.elapsed().as_micros() as f64;
        self.metrics.record_latency(latency_us);

        // Update aggregate metrics from analytics
        self.metrics.update_from_analytics(&self.analytics);
    }

    /// Handle order completion
    fn on_order_complete(&mut self, order_id: &OrderId, status: &OrderStatus) {
        // Record time to fill
        if let Some(start_time) = self.order_start_times.get(order_id) {
            let time_to_fill = start_time.elapsed().as_secs_f64();
            self.metrics.record_time_to_fill(time_to_fill);
            self.order_start_times.remove(order_id);
        }

        // Update health status
        if let Ok(mut health) = self.health.write() {
            health.completed_orders += 1;
            health.avg_quality_score = self.analytics.quality_score();
            health.last_update = Instant::now();
        }

        // Check for partial fills
        if status.filled_quantity < status.total_quantity * 0.99 {
            self.metrics.record_partial_fill();
        }
    }

    /// Handle order failure
    fn on_order_failed(&mut self, order_id: &OrderId, status: &OrderStatus) {
        let reason = format!("{:?}", status.state);

        // Record failure metric
        self.metrics.record_failure(&reason);

        // Update health status
        if let Ok(mut health) = self.health.write() {
            health.failed_orders += 1;
            health.recent_errors.push(reason);

            // Keep only last 10 errors
            if health.recent_errors.len() > 10 {
                health.recent_errors.remove(0);
            }

            // Mark unhealthy if too many failures
            let failure_rate = health.failed_orders as f64 / health.total_orders as f64;
            health.healthy = failure_rate < 0.1; // Less than 10% failure rate
            health.last_update = Instant::now();
        }

        // Clean up
        self.order_start_times.remove(order_id);
    }

    /// Record an execution manually (for custom integrations)
    pub fn record_execution(&mut self, record: ExecutionRecord) {
        // Record to analytics
        self.analytics.record_execution(record.clone());

        // Record to metrics
        self.metrics.record_execution(&record);
    }

    /// Get order status
    pub fn get_order_status(&self, order_id: &OrderId) -> Option<OrderStatus> {
        self.manager.get_order_status(order_id).cloned()
    }

    /// Cancel an order
    pub fn cancel_order(&mut self, order_id: &OrderId) -> bool {
        let result = self.manager.cancel_order(order_id);

        if result {
            // Clean up tracking
            self.order_start_times.remove(order_id);
        }

        result
    }

    /// Get execution analytics
    pub fn analytics(&self) -> &ExecutionAnalytics {
        &self.analytics
    }

    /// Get execution report
    pub fn execution_report(&self) -> ExecutionReport {
        self.analytics.generate_report()
    }

    /// Get active orders
    pub fn active_orders(&self) -> Vec<OrderId> {
        self.manager
            .active_orders()
            .iter()
            .map(|s| s.order_id.clone())
            .collect()
    }

    /// Get health status
    pub fn health_status(&self) -> HealthStatus {
        self.health.read().unwrap().clone()
    }

    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        self.health.read().unwrap().healthy
    }

    /// Get metrics registry
    pub fn metrics(&self) -> Arc<ExecutionMetrics> {
        self.metrics.clone()
    }

    /// Export metrics as Prometheus text format
    pub fn export_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        self.metrics.encode_text()
    }

    /// Get strategy statistics
    pub fn strategy_stats(&self) -> StrategyStats {
        StrategyStats {
            market_orders: self.strategy_counters.market,
            limit_orders: self.strategy_counters.limit,
            twap_orders: self.strategy_counters.twap,
            vwap_orders: self.strategy_counters.vwap,
        }
    }

    /// Reset all metrics and analytics
    pub fn reset(&mut self) {
        self.analytics.reset();
        self.order_start_times.clear();
        self.strategy_counters = StrategyCounters::default();

        if let Ok(mut health) = self.health.write() {
            *health = HealthStatus::default();
        }
    }

    /// Print comprehensive status report
    pub fn print_status(&self) {
        println!("\n=== Instrumented Execution Manager Status ===");

        let health = self.health_status();
        println!("\n## Health Status");
        println!(
            "  Status: {}",
            if health.healthy {
                "✓ Healthy"
            } else {
                "✗ Unhealthy"
            }
        );
        println!("  Total Orders: {}", health.total_orders);
        println!("  Completed: {}", health.completed_orders);
        println!("  Failed: {}", health.failed_orders);
        println!(
            "  Success Rate: {:.1}%",
            if health.total_orders > 0 {
                100.0 * health.completed_orders as f64 / health.total_orders as f64
            } else {
                0.0
            }
        );
        println!("  Avg Quality Score: {:.1}/100", health.avg_quality_score);

        println!("\n## Strategy Distribution");
        let stats = self.strategy_stats();
        let total =
            stats.market_orders + stats.limit_orders + stats.twap_orders + stats.vwap_orders;
        if total > 0 {
            println!(
                "  Market: {} ({:.1}%)",
                stats.market_orders,
                100.0 * stats.market_orders as f64 / total as f64
            );
            println!(
                "  Limit:  {} ({:.1}%)",
                stats.limit_orders,
                100.0 * stats.limit_orders as f64 / total as f64
            );
            println!(
                "  TWAP:   {} ({:.1}%)",
                stats.twap_orders,
                100.0 * stats.twap_orders as f64 / total as f64
            );
            println!(
                "  VWAP:   {} ({:.1}%)",
                stats.vwap_orders,
                100.0 * stats.vwap_orders as f64 / total as f64
            );
        } else {
            println!("  No orders submitted yet");
        }

        println!("\n## Execution Analytics");
        let report = self.execution_report();
        report.print();

        if !health.recent_errors.is_empty() {
            println!("\n## Recent Errors");
            for (i, error) in health.recent_errors.iter().enumerate() {
                println!("  {}. {}", i + 1, error);
            }
        }

        println!("\n## Active Orders");
        let active = self.active_orders();
        if active.is_empty() {
            println!("  No active orders");
        } else {
            println!("  {} active orders", active.len());
            for order_id in active.iter().take(5) {
                if let Some(status) = self.get_order_status(order_id) {
                    println!(
                        "    - {}: {:.1}% filled",
                        order_id,
                        100.0 * status.filled_quantity / status.total_quantity
                    );
                }
            }
            if active.len() > 5 {
                println!("    ... and {} more", active.len() - 5);
            }
        }

        println!();
    }
}

impl Default for InstrumentedExecutionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy usage statistics
#[derive(Debug, Clone)]
pub struct StrategyStats {
    pub market_orders: u64,
    pub limit_orders: u64,
    pub twap_orders: u64,
    pub vwap_orders: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_instrumented_manager_creation() {
        let manager = InstrumentedExecutionManager::new();
        assert!(manager.is_healthy());
    }

    #[test]
    fn test_submit_order_records_metrics() {
        let mut manager = InstrumentedExecutionManager::new();

        let order_id = manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        assert!(manager.get_order_status(&order_id).is_some());

        let health = manager.health_status();
        assert_eq!(health.total_orders, 1);
    }

    #[test]
    fn test_strategy_counters() {
        let mut manager = InstrumentedExecutionManager::new();

        manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::TWAP {
                duration: Duration::from_secs(10),
                num_slices: 5,
            },
            limit_price: None,
            venues: None,
        });

        let stats = manager.strategy_stats();
        assert_eq!(stats.market_orders, 1);
        assert_eq!(stats.twap_orders, 1);
    }

    #[test]
    fn test_process_updates_metrics() {
        let mut manager = InstrumentedExecutionManager::new();

        manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        manager.process();

        // Processing should not panic
        assert!(manager.is_healthy());
    }

    #[test]
    fn test_record_execution() {
        let mut manager = InstrumentedExecutionManager::new();

        let record = ExecutionRecord {
            order_id: "TEST-1".to_string(),
            quantity: 100.0,
            execution_price: 50.05,
            benchmark_price: 50.0,
            timestamp: Instant::now(),
            side: Side::Buy,
            venue: "TEST-EXCHANGE".to_string(),
        };

        manager.record_execution(record);

        let report = manager.execution_report();
        assert_eq!(report.total_executions, 1);
    }

    #[test]
    fn test_export_metrics() {
        let manager = InstrumentedExecutionManager::new();
        let result = manager.export_metrics();
        assert!(result.is_ok());

        let text = result.unwrap();
        assert!(text.contains("vision_execution"));
    }

    #[test]
    fn test_health_status() {
        let mut manager = InstrumentedExecutionManager::new();

        manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        let health = manager.health_status();
        assert!(health.healthy);
        assert_eq!(health.total_orders, 1);
        assert_eq!(health.completed_orders, 0);
    }

    #[test]
    fn test_reset() {
        let mut manager = InstrumentedExecutionManager::new();

        manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        manager.reset();

        let health = manager.health_status();
        assert_eq!(health.total_orders, 0);

        let stats = manager.strategy_stats();
        assert_eq!(stats.market_orders, 0);
    }

    #[test]
    fn test_twap_execution_with_metrics() {
        let mut manager = InstrumentedExecutionManager::new();

        let _order_id = manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
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
        for _ in 0..5 {
            thread::sleep(Duration::from_millis(700));
            manager.process();
        }

        let health = manager.health_status();
        assert!(health.total_orders > 0);
    }
}
