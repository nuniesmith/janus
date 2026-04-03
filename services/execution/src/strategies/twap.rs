//! TWAP (Time-Weighted Average Price) Execution Strategy
//!
//! Splits a large order into smaller child orders distributed evenly over time
//! to minimize market impact and achieve an average execution price close to TWAP.

use crate::error::{ExecutionError, Result};
use crate::types::{Order, OrderSide, OrderTypeEnum};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, info, warn};

/// TWAP strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapConfig {
    /// Total quantity to execute
    pub total_quantity: Decimal,

    /// Symbol to trade
    pub symbol: String,

    /// Exchange to use
    pub exchange: String,

    /// Order side (Buy/Sell)
    pub side: OrderSide,

    /// Duration over which to execute (seconds)
    pub duration_secs: u64,

    /// Number of child orders (slices)
    pub num_slices: usize,

    /// Minimum time between orders (seconds)
    pub min_interval_secs: u64,

    /// Whether to use limit orders (vs market)
    pub use_limit_orders: bool,

    /// Price limit for limit orders (optional)
    pub limit_price: Option<Decimal>,

    /// Maximum deviation from limit price (%)
    pub max_price_deviation_pct: Option<Decimal>,

    /// Allow partial fills
    pub allow_partial: bool,

    /// Cancel unfilled orders at end
    pub cancel_at_end: bool,
}

impl TwapConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.total_quantity <= Decimal::ZERO {
            return Err(ExecutionError::Validation(
                "Total quantity must be positive".to_string(),
            ));
        }

        if self.duration_secs == 0 {
            return Err(ExecutionError::Validation(
                "Duration must be positive".to_string(),
            ));
        }

        if self.num_slices == 0 {
            return Err(ExecutionError::Validation(
                "Number of slices must be positive".to_string(),
            ));
        }

        if self.min_interval_secs * (self.num_slices as u64) > self.duration_secs {
            return Err(ExecutionError::Validation(
                "Min interval * num_slices exceeds total duration".to_string(),
            ));
        }

        if self.use_limit_orders && self.limit_price.is_none() {
            return Err(ExecutionError::Validation(
                "Limit price required for limit orders".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate slice size
    pub fn slice_size(&self) -> Decimal {
        self.total_quantity / Decimal::from(self.num_slices)
    }

    /// Calculate interval between slices
    pub fn slice_interval(&self) -> Duration {
        Duration::from_secs(self.duration_secs / self.num_slices as u64)
    }
}

/// TWAP execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapState {
    /// Configuration
    pub config: TwapConfig,

    /// Current status
    pub status: TwapStatus,

    /// Total quantity filled
    pub filled_quantity: Decimal,

    /// Average fill price
    pub average_price: Decimal,

    /// Number of child orders created
    pub orders_created: usize,

    /// Number of child orders filled
    pub orders_filled: usize,

    /// Number of child orders cancelled
    pub orders_cancelled: usize,

    /// Start time
    pub start_time: Option<DateTime<Utc>>,

    /// End time
    pub end_time: Option<DateTime<Utc>>,

    /// Child order IDs
    pub child_order_ids: Vec<String>,

    /// Error message (if failed)
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TwapStatus {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}

impl TwapState {
    /// Create new TWAP state
    pub fn new(config: TwapConfig) -> Self {
        Self {
            config,
            status: TwapStatus::Pending,
            filled_quantity: Decimal::ZERO,
            average_price: Decimal::ZERO,
            orders_created: 0,
            orders_filled: 0,
            orders_cancelled: 0,
            start_time: None,
            end_time: None,
            child_order_ids: Vec::new(),
            error: None,
        }
    }

    /// Calculate fill percentage
    pub fn fill_percentage(&self) -> Decimal {
        if self.config.total_quantity > Decimal::ZERO {
            (self.filled_quantity / self.config.total_quantity) * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }

    /// Calculate elapsed time
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.and_then(|start| {
            self.end_time.or(Some(Utc::now())).map(|end| {
                let duration = end.signed_duration_since(start);
                Duration::from_secs(duration.num_seconds().max(0) as u64)
            })
        })
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            TwapStatus::Completed | TwapStatus::Cancelled | TwapStatus::Failed
        )
    }
}

/// TWAP strategy executor
pub struct TwapExecutor {
    state: Arc<RwLock<TwapState>>,
}

impl TwapExecutor {
    /// Create new TWAP executor
    pub fn new(config: TwapConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            state: Arc::new(RwLock::new(TwapState::new(config))),
        })
    }

    /// Get current state
    pub async fn state(&self) -> TwapState {
        self.state.read().await.clone()
    }

    /// Start execution
    pub async fn start<F>(&self, mut order_submitter: F) -> Result<TwapState>
    where
        F: FnMut(Order) -> Result<String> + Send,
    {
        let mut state = self.state.write().await;

        if state.status != TwapStatus::Pending {
            return Err(ExecutionError::InvalidOrderState(
                "TWAP already started".to_string(),
            ));
        }

        state.status = TwapStatus::Running;
        state.start_time = Some(Utc::now());
        let config = state.config.clone();
        drop(state);

        info!(
            "Starting TWAP: {} {} {} over {}s in {} slices",
            config.side,
            config.total_quantity,
            config.symbol,
            config.duration_secs,
            config.num_slices
        );

        // Calculate slice parameters
        let slice_size = config.slice_size();
        let slice_interval = config.slice_interval();

        // Execute slices
        let mut interval_timer = interval(slice_interval);
        let mut total_filled = Decimal::ZERO;
        let mut total_cost = Decimal::ZERO;

        for slice_num in 0..config.num_slices {
            // Wait for next interval (skip first tick)
            if slice_num > 0 {
                interval_timer.tick().await;
            }

            // Check if cancelled
            let current_status = self.state.read().await.status;
            if current_status == TwapStatus::Cancelled {
                info!(
                    "TWAP cancelled at slice {}/{}",
                    slice_num + 1,
                    config.num_slices
                );
                break;
            }

            // Calculate quantity for this slice (handle rounding on last slice)
            let slice_qty = if slice_num == config.num_slices - 1 {
                config.total_quantity - total_filled
            } else {
                slice_size
            };

            if slice_qty <= Decimal::ZERO {
                debug!("Slice {} has no remaining quantity", slice_num + 1);
                continue;
            }

            // Create child order
            let child_order = Order::new(
                format!("twap_slice_{}_{}", slice_num + 1, uuid::Uuid::new_v4()),
                config.symbol.clone(),
                config.exchange.clone(),
                config.side,
                if config.use_limit_orders {
                    OrderTypeEnum::Limit
                } else {
                    OrderTypeEnum::Market
                },
                slice_qty,
            );

            debug!(
                "Submitting TWAP slice {}/{}: {} {}",
                slice_num + 1,
                config.num_slices,
                slice_qty,
                config.symbol
            );

            // Submit order
            match order_submitter(child_order.clone()) {
                Ok(order_id) => {
                    let mut state = self.state.write().await;
                    state.orders_created += 1;
                    state.child_order_ids.push(order_id.clone());

                    // Simulate fill for now (in real implementation, wait for fill callback)
                    // This is a placeholder - actual implementation would track fills
                    let fill_price = config.limit_price.unwrap_or(Decimal::from(50000));
                    total_filled += slice_qty;
                    total_cost += slice_qty * fill_price;
                    state.filled_quantity = total_filled;
                    state.average_price = if total_filled > Decimal::ZERO {
                        total_cost / total_filled
                    } else {
                        Decimal::ZERO
                    };
                    state.orders_filled += 1;

                    info!(
                        "TWAP slice {}/{} submitted: {} filled at {}",
                        slice_num + 1,
                        config.num_slices,
                        slice_qty,
                        fill_price
                    );
                }
                Err(e) => {
                    warn!("Failed to submit TWAP slice {}: {}", slice_num + 1, e);
                    if !config.allow_partial {
                        let mut state = self.state.write().await;
                        state.status = TwapStatus::Failed;
                        state.error = Some(e.to_string());
                        state.end_time = Some(Utc::now());
                        return Err(e);
                    }
                }
            }
        }

        // Finalize
        let mut state = self.state.write().await;
        state.end_time = Some(Utc::now());

        if state.status == TwapStatus::Cancelled {
            info!(
                "TWAP cancelled: filled {}/{}",
                state.filled_quantity, config.total_quantity
            );
        } else {
            state.status = TwapStatus::Completed;
            info!(
                "TWAP completed: filled {}/{} at avg price {}",
                state.filled_quantity, config.total_quantity, state.average_price
            );
        }

        Ok(state.clone())
    }

    /// Cancel execution
    pub async fn cancel(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if state.status != TwapStatus::Running {
            return Err(ExecutionError::InvalidOrderState(
                "TWAP not running".to_string(),
            ));
        }

        state.status = TwapStatus::Cancelled;
        info!("TWAP cancellation requested");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_config_validation() {
        let config = TwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            num_slices: 10,
            min_interval_secs: 1,
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            cancel_at_end: false,
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_twap_config_invalid_quantity() {
        let config = TwapConfig {
            total_quantity: Decimal::ZERO,
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            num_slices: 10,
            min_interval_secs: 1,
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            cancel_at_end: false,
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_slice_calculation() {
        let config = TwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            num_slices: 10,
            min_interval_secs: 1,
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            cancel_at_end: false,
        };

        assert_eq!(config.slice_size(), Decimal::from(10));
        assert_eq!(config.slice_interval(), Duration::from_secs(6));
    }

    #[test]
    fn test_twap_state_creation() {
        let config = TwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            num_slices: 10,
            min_interval_secs: 1,
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            cancel_at_end: false,
        };

        let state = TwapState::new(config);
        assert_eq!(state.status, TwapStatus::Pending);
        assert_eq!(state.filled_quantity, Decimal::ZERO);
        assert_eq!(state.orders_created, 0);
    }

    #[test]
    fn test_fill_percentage() {
        let config = TwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            num_slices: 10,
            min_interval_secs: 1,
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            cancel_at_end: false,
        };

        let mut state = TwapState::new(config);
        state.filled_quantity = Decimal::from(50);

        assert_eq!(state.fill_percentage(), Decimal::from(50));
    }

    #[tokio::test]
    async fn test_twap_executor_creation() {
        let config = TwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            num_slices: 10,
            min_interval_secs: 1,
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            cancel_at_end: false,
        };

        let executor = TwapExecutor::new(config);
        assert!(executor.is_ok());

        let state = executor.unwrap().state().await;
        assert_eq!(state.status, TwapStatus::Pending);
    }

    #[tokio::test]
    async fn test_twap_execution() {
        let config = TwapConfig {
            total_quantity: Decimal::from(10),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 5,
            num_slices: 5,
            min_interval_secs: 0,
            use_limit_orders: true,
            limit_price: Some(Decimal::from(50000)),
            max_price_deviation_pct: None,
            allow_partial: true,
            cancel_at_end: false,
        };

        let executor = TwapExecutor::new(config).unwrap();

        // Mock order submitter
        let order_submitter =
            |order: Order| -> Result<String> { Ok(format!("order_{}", order.id)) };

        let result = executor.start(order_submitter).await;
        assert!(result.is_ok());

        let final_state = result.unwrap();
        assert_eq!(final_state.status, TwapStatus::Completed);
        assert_eq!(final_state.filled_quantity, Decimal::from(10));
        assert_eq!(final_state.orders_created, 5);
    }
}
