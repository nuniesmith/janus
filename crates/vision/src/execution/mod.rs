//! Advanced Order Execution
//!
//! This module provides sophisticated order execution algorithms and analytics
//! for minimizing market impact and optimizing execution quality.
//!
//! # Components
//!
//! - **TWAP**: Time-Weighted Average Price execution
//! - **VWAP**: Volume-Weighted Average Price execution
//! - **Analytics**: Execution quality metrics and slippage tracking
//! - **Manager**: Order lifecycle management and smart routing
//!
//! # Examples
//!
//! ## TWAP Execution
//!
//! ```rust
//! use vision::execution::twap::{TWAPExecutor, TWAPConfig};
//! use std::time::Duration;
//!
//! let config = TWAPConfig {
//!     total_quantity: 10000.0,
//!     duration: Duration::from_secs(300),
//!     num_slices: 10,
//!     ..Default::default()
//! };
//!
//! let mut executor = TWAPExecutor::new(config);
//! executor.start();
//!
//! // Execute slices
//! while !executor.is_complete() {
//!     if let Some(slice) = executor.next_slice() {
//!         println!("Execute {} units", slice.quantity);
//!         executor.mark_executed(slice.quantity, 100.5);
//!     }
//! }
//!
//! let avg_price = executor.average_price();
//! println!("Average execution price: {:?}", avg_price);
//! ```
//!
//! ## VWAP Execution
//!
//! ```rust
//! use vision::execution::vwap::{VWAPExecutor, VWAPConfig, VolumeProfile};
//! use std::time::Duration;
//!
//! // Create U-shaped intraday volume profile
//! let profile = VolumeProfile::intraday_u_shape(8);
//!
//! let config = VWAPConfig {
//!     total_quantity: 10000.0,
//!     duration: Duration::from_secs(3600),
//!     volume_profile: profile,
//!     participation_rate: 0.2,
//!     ..Default::default()
//! };
//!
//! let mut executor = VWAPExecutor::new(config);
//! executor.start();
//!
//! while !executor.is_complete() {
//!     if let Some(slice) = executor.next_slice() {
//!         let market_volume = 5000.0; // Observed market volume
//!         executor.mark_executed(slice.quantity, 100.5, Some(market_volume));
//!     }
//! }
//!
//! let vwap = executor.vwap_price();
//! println!("VWAP: {:?}", vwap);
//! ```
//!
//! ## Execution Analytics
//!
//! ```rust
//! use vision::execution::analytics::{ExecutionAnalytics, ExecutionRecord, Side};
//! use std::time::Instant;
//!
//! let mut analytics = ExecutionAnalytics::new();
//!
//! // Record executions
//! analytics.record_execution(ExecutionRecord {
//!     order_id: "ORD001".to_string(),
//!     quantity: 1000.0,
//!     execution_price: 100.5,
//!     benchmark_price: 100.0,
//!     timestamp: Instant::now(),
//!     side: Side::Buy,
//!     venue: "NYSE".to_string(),
//! });
//!
//! // Generate report
//! let report = analytics.generate_report();
//! println!("Average Slippage: {:.2} bps", report.average_slippage_bps);
//! println!("Total Cost: ${:.2}", report.total_cost);
//! println!("Quality Score: {:.1}/100", report.quality_score);
//! ```
//!
//! ## Complete Execution Manager
//!
//! ```rust
//! use vision::execution::manager::{ExecutionManager, OrderRequest, ExecutionStrategy};
//! use vision::execution::analytics::Side;
//! use std::time::Duration;
//!
//! let mut manager = ExecutionManager::new();
//!
//! // Submit TWAP order
//! let order_id = manager.submit_order(OrderRequest {
//!     symbol: "AAPL".to_string(),
//!     quantity: 10000.0,
//!     side: Side::Buy,
//!     strategy: ExecutionStrategy::TWAP {
//!         duration: Duration::from_secs(300),
//!         num_slices: 10,
//!     },
//!     limit_price: Some(150.0),
//!     venues: None,
//! });
//!
//! // Process execution
//! loop {
//!     manager.process();
//!
//!     if let Some(status) = manager.get_order_status(&order_id) {
//!         if status.is_complete() {
//!             println!("Order complete: filled {}/{}",
//!                      status.filled_quantity,
//!                      status.total_quantity);
//!             break;
//!         }
//!     }
//!
//!     std::thread::sleep(Duration::from_millis(100));
//! }
//!
//! // Get execution analytics
//! let report = manager.execution_report();
//! println!("Execution quality: {:.1}/100", report.quality_score);
//! ```
//!
//! ## Integration with Risk Management
//!
//! ```rust,ignore
//! use vision::execution::manager::{ExecutionManager, OrderRequest, ExecutionStrategy};
//! use vision::risk::RiskManager;
//! use vision::adaptive::AdaptiveSystem;
//!
//! let mut manager = ExecutionManager::new();
//! let mut risk_manager = RiskManager::new(risk_config);
//! let mut adaptive = AdaptiveSystem::new();
//!
//! // Get signal with adaptive threshold
//! let (calibrated_confidence, threshold, should_trade) =
//!     adaptive.process_prediction(model_confidence);
//!
//! if should_trade {
//!     // Calculate position size with regime adjustment
//!     let base_size = adaptive.get_adjusted_position_size(10000.0);
//!
//!     // Apply risk limits
//!     let final_size = risk_manager.calculate_position_size(
//!         base_size,
//!         entry_price,
//!         stop_loss,
//!     );
//!
//!     // Submit order with VWAP execution
//!     let order_id = manager.submit_order(OrderRequest {
//!         symbol: "AAPL".to_string(),
//!         quantity: final_size,
//!         side: Side::Buy,
//!         strategy: ExecutionStrategy::VWAP {
//!             duration: Duration::from_secs(300),
//!             volume_profile: VolumeProfile::intraday_u_shape(10),
//!         },
//!         limit_price: Some(entry_price),
//!         venues: None,
//!     });
//! }
//! ```

pub mod analytics;
pub mod instrumented;
pub mod manager;
pub mod metrics;
pub mod twap;
pub mod vwap;

// Re-exports for convenience
pub use analytics::{ExecutionAnalytics, ExecutionRecord, ExecutionReport, Side, VenueStats};
pub use instrumented::{HealthStatus, InstrumentedExecutionManager, StrategyStats};
pub use manager::{
    ExecutionManager, ExecutionStrategy, OrderId, OrderRequest, OrderState, OrderStatus, Venue,
};
pub use metrics::{EXECUTION_METRICS, ExecutionMetrics, ExecutionMetricsCollector};
pub use twap::{ExecutionSlice, TWAPConfig, TWAPExecutor, TWAPStatistics};
pub use vwap::{VWAPConfig, VWAPExecutor, VWAPSlice, VWAPStatistics, VolumeProfile};

/// Execution module version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_twap_basic_flow() {
        let config = TWAPConfig {
            total_quantity: 1000.0,
            duration: Duration::from_secs(10),
            num_slices: 5,
            ..Default::default()
        };

        let mut executor = TWAPExecutor::new(config);
        executor.start();

        assert!(!executor.is_complete());
        assert_eq!(executor.slices().len(), 5);
    }

    #[test]
    fn test_vwap_basic_flow() {
        let profile = VolumeProfile::uniform(5);
        let config = VWAPConfig {
            total_quantity: 1000.0,
            duration: Duration::from_secs(10),
            volume_profile: profile,
            ..Default::default()
        };

        let mut executor = VWAPExecutor::new(config);
        executor.start();

        assert!(!executor.is_complete());
        assert_eq!(executor.slices().len(), 5);
    }

    #[test]
    fn test_analytics_basic() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(ExecutionRecord {
            order_id: "TEST1".to_string(),
            quantity: 100.0,
            execution_price: 100.5,
            benchmark_price: 100.0,
            timestamp: std::time::Instant::now(),
            side: Side::Buy,
            venue: "NYSE".to_string(),
        });

        let report = analytics.generate_report();
        assert_eq!(report.total_executions, 1);
        assert!(report.quality_score > 0.0);
    }

    #[test]
    fn test_manager_basic() {
        let mut manager = ExecutionManager::new();

        let order_id = manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
            quantity: 100.0,
            side: Side::Buy,
            strategy: ExecutionStrategy::Market,
            limit_price: None,
            venues: None,
        });

        assert!(manager.get_order_status(&order_id).is_some());
    }

    #[test]
    fn test_integration_twap_manager() {
        let mut manager = ExecutionManager::new();

        let order_id = manager.submit_order(OrderRequest {
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

        // Process slices
        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(1100));
            manager.process();
        }

        if let Some(status) = manager.get_order_status(&order_id) {
            assert!(status.filled_quantity > 0.0);
        }
    }

    #[test]
    fn test_integration_vwap_manager() {
        let mut manager = ExecutionManager::new();

        let order_id = manager.submit_order(OrderRequest {
            symbol: "TEST".to_string(),
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

        if let Some(status) = manager.get_order_status(&order_id) {
            assert!(status.filled_quantity > 0.0);
        }
    }
}
