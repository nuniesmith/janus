//! Execution Strategies
//!
//! Advanced order execution algorithms designed to minimize market impact
//! and achieve optimal execution prices for large orders.
//!
//! ## Available Strategies
//!
//! - **TWAP** (Time-Weighted Average Price): Splits orders evenly over time
//! - **VWAP** (Volume-Weighted Average Price): Splits orders based on volume profile
//! - **Iceberg**: Hides large orders by showing small "tip" orders
//!
//! ## Usage Example
//!
//! ```no_run
//! use janus_execution::strategies::{TwapConfig, TwapExecutor};
//! use janus_execution::types::OrderSide;
//! use rust_decimal::Decimal;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure TWAP strategy
//! let config = TwapConfig {
//!     total_quantity: Decimal::from(100),
//!     symbol: "BTCUSD".to_string(),
//!     exchange: "bybit".to_string(),
//!     side: OrderSide::Buy,
//!     duration_secs: 300,  // 5 minutes
//!     num_slices: 10,
//!     min_interval_secs: 10,
//!     use_limit_orders: false,
//!     limit_price: None,
//!     max_price_deviation_pct: None,
//!     allow_partial: true,
//!     cancel_at_end: false,
//! };
//!
//! // Create executor
//! let executor = TwapExecutor::new(config)?;
//!
//! // Execute strategy
//! let order_submitter = |order| {
//!     // Submit order to exchange
//!     Ok("order_id".to_string())
//! };
//!
//! let result = executor.start(order_submitter).await?;
//! println!("TWAP completed: {} filled", result.filled_quantity);
//! # Ok(())
//! # }
//! ```

pub mod iceberg;
pub mod twap;
pub mod vwap;

// Re-export main types
pub use iceberg::{IcebergConfig, IcebergExecutor, IcebergState, IcebergStatus};
pub use twap::{TwapConfig, TwapExecutor, TwapState, TwapStatus};
pub use vwap::{VolumeBucket, VwapConfig, VwapExecutor, VwapState, VwapStatus};
