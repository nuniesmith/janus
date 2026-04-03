//! Iceberg Order Execution Strategy
//!
//! Splits a large order into smaller visible "tip" orders while keeping the rest hidden.
//! As each tip order fills, a new one is automatically placed until the total quantity is filled.

use crate::error::{ExecutionError, Result};
use crate::types::{Order, OrderSide, OrderTypeEnum};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Iceberg order configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergConfig {
    /// Total quantity to execute
    pub total_quantity: Decimal,

    /// Symbol to trade
    pub symbol: String,

    /// Exchange to use
    pub exchange: String,

    /// Order side (Buy/Sell)
    pub side: OrderSide,

    /// Visible quantity per tip order
    pub tip_size: Decimal,

    /// Minimum tip size (to handle remainders)
    pub min_tip_size: Decimal,

    /// Price for limit orders
    pub limit_price: Option<Decimal>,

    /// Use market orders instead of limit
    pub use_market_orders: bool,

    /// Maximum price slippage allowed (%)
    pub max_slippage_pct: Option<Decimal>,

    /// Time to wait between tip orders (milliseconds)
    pub tip_delay_ms: u64,

    /// Maximum number of active tip orders
    pub max_active_tips: usize,

    /// Cancel unfilled tips at end
    pub cancel_unfilled: bool,

    /// Allow partial fills
    pub allow_partial: bool,

    /// Variance in tip size (randomize to avoid detection)
    /// Value between 0.0 and 1.0 for percentage variance
    pub tip_variance_pct: Option<Decimal>,
}

impl IcebergConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.total_quantity <= Decimal::ZERO {
            return Err(ExecutionError::Validation(
                "Total quantity must be positive".to_string(),
            ));
        }

        if self.tip_size <= Decimal::ZERO {
            return Err(ExecutionError::Validation(
                "Tip size must be positive".to_string(),
            ));
        }

        if self.tip_size > self.total_quantity {
            return Err(ExecutionError::Validation(
                "Tip size cannot exceed total quantity".to_string(),
            ));
        }

        if self.min_tip_size <= Decimal::ZERO {
            return Err(ExecutionError::Validation(
                "Min tip size must be positive".to_string(),
            ));
        }

        if self.min_tip_size > self.tip_size {
            return Err(ExecutionError::Validation(
                "Min tip size cannot exceed tip size".to_string(),
            ));
        }

        if !self.use_market_orders && self.limit_price.is_none() {
            return Err(ExecutionError::Validation(
                "Limit price required for limit orders".to_string(),
            ));
        }

        if self.max_active_tips == 0 {
            return Err(ExecutionError::Validation(
                "Max active tips must be positive".to_string(),
            ));
        }

        if let Some(variance) = self.tip_variance_pct {
            if variance < Decimal::ZERO || variance > Decimal::ONE {
                return Err(ExecutionError::Validation(
                    "Tip variance must be between 0 and 1".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Calculate number of tip orders needed
    pub fn estimate_tips(&self) -> usize {
        let tips = (self.total_quantity / self.tip_size).ceil();
        tips.to_string().parse::<usize>().unwrap_or(1)
    }

    /// Calculate tip size for current iteration (with optional variance)
    pub fn calculate_tip_size(&self, remaining: Decimal, iteration: usize) -> Decimal {
        let base_size = if remaining < self.tip_size {
            if remaining >= self.min_tip_size {
                remaining
            } else {
                Decimal::ZERO
            }
        } else {
            self.tip_size
        };

        // Apply variance if configured
        if let Some(variance) = self.tip_variance_pct {
            // Simple pseudo-random variance based on iteration
            // In production, use proper random number generation
            let variance_factor = Decimal::ONE
                + (variance * Decimal::from((iteration % 10) as i64 - 5) / Decimal::from(5));
            let varied_size = base_size * variance_factor;

            // Clamp to reasonable bounds
            if varied_size < self.min_tip_size {
                self.min_tip_size
            } else if varied_size > remaining {
                remaining
            } else {
                varied_size
            }
        } else {
            base_size
        }
    }
}

/// Iceberg execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergState {
    /// Configuration
    pub config: IcebergConfig,

    /// Current status
    pub status: IcebergStatus,

    /// Total quantity filled
    pub filled_quantity: Decimal,

    /// Remaining quantity
    pub remaining_quantity: Decimal,

    /// Average fill price
    pub average_price: Decimal,

    /// Number of tip orders created
    pub tips_created: usize,

    /// Number of tip orders filled
    pub tips_filled: usize,

    /// Number of tip orders cancelled
    pub tips_cancelled: usize,

    /// Currently active tip order IDs
    pub active_tips: Vec<String>,

    /// Start time
    pub start_time: Option<DateTime<Utc>>,

    /// End time
    pub end_time: Option<DateTime<Utc>>,

    /// All child order IDs
    pub child_order_ids: Vec<String>,

    /// Error message (if failed)
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IcebergStatus {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}

impl IcebergState {
    /// Create new Iceberg state
    pub fn new(config: IcebergConfig) -> Self {
        Self {
            remaining_quantity: config.total_quantity,
            config,
            status: IcebergStatus::Pending,
            filled_quantity: Decimal::ZERO,
            average_price: Decimal::ZERO,
            tips_created: 0,
            tips_filled: 0,
            tips_cancelled: 0,
            active_tips: Vec::new(),
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
            IcebergStatus::Completed | IcebergStatus::Cancelled | IcebergStatus::Failed
        )
    }

    /// Check if more tips needed
    pub fn needs_more_tips(&self) -> bool {
        self.remaining_quantity >= self.config.min_tip_size
            && self.active_tips.len() < self.config.max_active_tips
    }
}

/// Iceberg strategy executor
pub struct IcebergExecutor {
    state: Arc<RwLock<IcebergState>>,
}

impl IcebergExecutor {
    /// Create new Iceberg executor
    pub fn new(config: IcebergConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            state: Arc::new(RwLock::new(IcebergState::new(config))),
        })
    }

    /// Get current state
    pub async fn state(&self) -> IcebergState {
        self.state.read().await.clone()
    }

    /// Start execution
    pub async fn start<F>(&self, mut order_submitter: F) -> Result<IcebergState>
    where
        F: FnMut(Order) -> Result<String> + Send,
    {
        let mut state = self.state.write().await;

        if state.status != IcebergStatus::Pending {
            return Err(ExecutionError::InvalidOrderState(
                "Iceberg already started".to_string(),
            ));
        }

        state.status = IcebergStatus::Running;
        state.start_time = Some(Utc::now());
        let config = state.config.clone();
        drop(state);

        info!(
            "Starting Iceberg: {} {} {} with tip size {}",
            config.side, config.total_quantity, config.symbol, config.tip_size
        );

        let mut total_filled = Decimal::ZERO;
        let mut total_cost = Decimal::ZERO;
        let mut iteration = 0;

        // Main execution loop
        loop {
            let state = self.state.read().await;
            let remaining = state.remaining_quantity;
            let status = state.status;
            drop(state);

            // Check if cancelled
            if status == IcebergStatus::Cancelled {
                info!("Iceberg cancelled");
                break;
            }

            // Check if complete
            if remaining < config.min_tip_size {
                info!("Iceberg complete - no remaining quantity");
                break;
            }

            // Calculate tip size for this iteration
            let tip_qty = config.calculate_tip_size(remaining, iteration);

            if tip_qty < config.min_tip_size {
                debug!("Tip quantity below minimum, stopping");
                break;
            }

            // Create tip order
            let tip_order = Order::new(
                format!("iceberg_tip_{}_{}", iteration + 1, uuid::Uuid::new_v4()),
                config.symbol.clone(),
                config.exchange.clone(),
                config.side,
                if config.use_market_orders {
                    OrderTypeEnum::Market
                } else {
                    OrderTypeEnum::Limit
                },
                tip_qty,
            );

            debug!(
                "Submitting Iceberg tip {}: {} {} ({}% of total)",
                iteration + 1,
                tip_qty,
                config.symbol,
                (tip_qty / config.total_quantity) * Decimal::from(100)
            );

            // Submit tip order
            match order_submitter(tip_order.clone()) {
                Ok(order_id) => {
                    let mut state = self.state.write().await;
                    state.tips_created += 1;
                    state.active_tips.push(order_id.clone());
                    state.child_order_ids.push(order_id.clone());

                    // Simulate fill (in real implementation, wait for fill callback)
                    let fill_price = config.limit_price.unwrap_or(Decimal::from(50000));
                    total_filled += tip_qty;
                    total_cost += tip_qty * fill_price;

                    state.filled_quantity = total_filled;
                    state.remaining_quantity = config.total_quantity - total_filled;
                    state.average_price = if total_filled > Decimal::ZERO {
                        total_cost / total_filled
                    } else {
                        Decimal::ZERO
                    };
                    state.tips_filled += 1;

                    // Remove from active (simulated immediate fill)
                    state.active_tips.retain(|id| id != &order_id);

                    info!(
                        "Iceberg tip {} filled: {} at {} (total: {}/{})",
                        iteration + 1,
                        tip_qty,
                        fill_price,
                        total_filled,
                        config.total_quantity
                    );
                }
                Err(e) => {
                    warn!("Failed to submit Iceberg tip {}: {}", iteration + 1, e);
                    if !config.allow_partial {
                        let mut state = self.state.write().await;
                        state.status = IcebergStatus::Failed;
                        state.error = Some(e.to_string());
                        state.end_time = Some(Utc::now());
                        return Err(e);
                    }
                }
            }

            iteration += 1;

            // Delay before next tip (if configured)
            if config.tip_delay_ms > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(config.tip_delay_ms)).await;
            }
        }

        // Finalize
        let mut state = self.state.write().await;
        state.end_time = Some(Utc::now());

        if state.status == IcebergStatus::Cancelled {
            info!(
                "Iceberg cancelled: filled {}/{}",
                state.filled_quantity, config.total_quantity
            );
        } else {
            state.status = IcebergStatus::Completed;
            info!(
                "Iceberg completed: filled {}/{} at avg price {} ({} tips)",
                state.filled_quantity,
                config.total_quantity,
                state.average_price,
                state.tips_created
            );
        }

        Ok(state.clone())
    }

    /// Cancel execution
    pub async fn cancel(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if state.status != IcebergStatus::Running {
            return Err(ExecutionError::InvalidOrderState(
                "Iceberg not running".to_string(),
            ));
        }

        state.status = IcebergStatus::Cancelled;
        info!("Iceberg cancellation requested");

        Ok(())
    }

    /// Handle fill callback (to be called when a tip order fills)
    pub async fn on_tip_filled(
        &self,
        order_id: &str,
        filled_qty: Decimal,
        fill_price: Decimal,
    ) -> Result<()> {
        let mut state = self.state.write().await;

        // Update fill statistics
        state.filled_quantity += filled_qty;
        state.remaining_quantity = state.config.total_quantity - state.filled_quantity;

        // Update average price
        let total_cost =
            state.average_price * (state.filled_quantity - filled_qty) + fill_price * filled_qty;
        state.average_price = if state.filled_quantity > Decimal::ZERO {
            total_cost / state.filled_quantity
        } else {
            Decimal::ZERO
        };

        // Remove from active tips
        state.active_tips.retain(|id| id != order_id);
        state.tips_filled += 1;

        debug!(
            "Tip {} filled: {} @ {} (total filled: {})",
            order_id, filled_qty, fill_price, state.filled_quantity
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_config_validation() {
        let config = IcebergConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            tip_size: Decimal::from(10),
            min_tip_size: Decimal::from(1),
            limit_price: Some(Decimal::from(50000)),
            use_market_orders: false,
            max_slippage_pct: None,
            tip_delay_ms: 100,
            max_active_tips: 1,
            cancel_unfilled: true,
            allow_partial: true,
            tip_variance_pct: None,
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_iceberg_config_invalid_tip_size() {
        let config = IcebergConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            tip_size: Decimal::from(150), // Exceeds total
            min_tip_size: Decimal::from(1),
            limit_price: Some(Decimal::from(50000)),
            use_market_orders: false,
            max_slippage_pct: None,
            tip_delay_ms: 100,
            max_active_tips: 1,
            cancel_unfilled: true,
            allow_partial: true,
            tip_variance_pct: None,
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_estimate_tips() {
        let config = IcebergConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            tip_size: Decimal::from(10),
            min_tip_size: Decimal::from(1),
            limit_price: Some(Decimal::from(50000)),
            use_market_orders: false,
            max_slippage_pct: None,
            tip_delay_ms: 100,
            max_active_tips: 1,
            cancel_unfilled: true,
            allow_partial: true,
            tip_variance_pct: None,
        };

        assert_eq!(config.estimate_tips(), 10);
    }

    #[test]
    fn test_calculate_tip_size() {
        let config = IcebergConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            tip_size: Decimal::from(10),
            min_tip_size: Decimal::from(1),
            limit_price: Some(Decimal::from(50000)),
            use_market_orders: false,
            max_slippage_pct: None,
            tip_delay_ms: 100,
            max_active_tips: 1,
            cancel_unfilled: true,
            allow_partial: true,
            tip_variance_pct: None,
        };

        // Normal tip
        assert_eq!(
            config.calculate_tip_size(Decimal::from(50), 0),
            Decimal::from(10)
        );

        // Last tip (smaller)
        assert_eq!(
            config.calculate_tip_size(Decimal::from(5), 0),
            Decimal::from(5)
        );

        // Below minimum
        assert_eq!(
            config.calculate_tip_size(Decimal::from_str_exact("0.5").unwrap(), 0),
            Decimal::ZERO
        );
    }

    #[test]
    fn test_iceberg_state_creation() {
        let config = IcebergConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            tip_size: Decimal::from(10),
            min_tip_size: Decimal::from(1),
            limit_price: Some(Decimal::from(50000)),
            use_market_orders: false,
            max_slippage_pct: None,
            tip_delay_ms: 100,
            max_active_tips: 1,
            cancel_unfilled: true,
            allow_partial: true,
            tip_variance_pct: None,
        };

        let state = IcebergState::new(config.clone());
        assert_eq!(state.status, IcebergStatus::Pending);
        assert_eq!(state.filled_quantity, Decimal::ZERO);
        assert_eq!(state.remaining_quantity, config.total_quantity);
        assert_eq!(state.tips_created, 0);
    }

    #[test]
    fn test_fill_percentage() {
        let config = IcebergConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            tip_size: Decimal::from(10),
            min_tip_size: Decimal::from(1),
            limit_price: Some(Decimal::from(50000)),
            use_market_orders: false,
            max_slippage_pct: None,
            tip_delay_ms: 100,
            max_active_tips: 1,
            cancel_unfilled: true,
            allow_partial: true,
            tip_variance_pct: None,
        };

        let mut state = IcebergState::new(config);
        state.filled_quantity = Decimal::from(50);

        assert_eq!(state.fill_percentage(), Decimal::from(50));
    }

    #[tokio::test]
    async fn test_iceberg_executor_creation() {
        let config = IcebergConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            tip_size: Decimal::from(10),
            min_tip_size: Decimal::from(1),
            limit_price: Some(Decimal::from(50000)),
            use_market_orders: false,
            max_slippage_pct: None,
            tip_delay_ms: 100,
            max_active_tips: 1,
            cancel_unfilled: true,
            allow_partial: true,
            tip_variance_pct: None,
        };

        let executor = IcebergExecutor::new(config);
        assert!(executor.is_ok());

        let state = executor.unwrap().state().await;
        assert_eq!(state.status, IcebergStatus::Pending);
    }

    #[tokio::test]
    async fn test_iceberg_execution() {
        let config = IcebergConfig {
            total_quantity: Decimal::from(10),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            tip_size: Decimal::from(2),
            min_tip_size: Decimal::from(1),
            limit_price: Some(Decimal::from(50000)),
            use_market_orders: false,
            max_slippage_pct: None,
            tip_delay_ms: 0,
            max_active_tips: 1,
            cancel_unfilled: true,
            allow_partial: true,
            tip_variance_pct: None,
        };

        let executor = IcebergExecutor::new(config).unwrap();

        // Mock order submitter
        let order_submitter =
            |order: Order| -> Result<String> { Ok(format!("order_{}", order.id)) };

        let result = executor.start(order_submitter).await;
        assert!(result.is_ok());

        let final_state = result.unwrap();
        assert_eq!(final_state.status, IcebergStatus::Completed);
        assert_eq!(final_state.filled_quantity, Decimal::from(10));
        assert_eq!(final_state.tips_created, 5); // 10 / 2 = 5 tips
    }
}
