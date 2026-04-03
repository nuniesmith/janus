//! VWAP (Volume-Weighted Average Price) Execution Strategy
//!
//! Splits a large order into smaller child orders distributed according to
//! historical volume patterns to minimize market impact and achieve execution
//! price close to VWAP.

use crate::error::{ExecutionError, Result};
use crate::types::{Order, OrderSide, OrderTypeEnum};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, info, warn};

/// Volume profile for a time bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeBucket {
    /// Start time (minutes from midnight)
    pub start_minute: u32,

    /// End time (minutes from midnight)
    pub end_minute: u32,

    /// Historical volume percentage for this bucket
    pub volume_pct: Decimal,
}

/// VWAP strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapConfig {
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

    /// Volume profile (historical volume distribution)
    /// If None, falls back to uniform distribution
    pub volume_profile: Option<Vec<VolumeBucket>>,

    /// Number of time buckets (if no volume profile provided)
    pub num_buckets: usize,

    /// Minimum order size (to avoid dust orders)
    pub min_order_size: Decimal,

    /// Whether to use limit orders (vs market)
    pub use_limit_orders: bool,

    /// Price limit for limit orders (optional)
    pub limit_price: Option<Decimal>,

    /// Maximum deviation from limit price (%)
    pub max_price_deviation_pct: Option<Decimal>,

    /// Allow partial fills
    pub allow_partial: bool,

    /// Participate rate (% of market volume to target)
    /// e.g., 0.1 = participate in 10% of market volume
    pub participation_rate: Option<Decimal>,
}

impl VwapConfig {
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

        if self.num_buckets == 0 {
            return Err(ExecutionError::Validation(
                "Number of buckets must be positive".to_string(),
            ));
        }

        if self.min_order_size <= Decimal::ZERO {
            return Err(ExecutionError::Validation(
                "Min order size must be positive".to_string(),
            ));
        }

        if let Some(profile) = &self.volume_profile {
            let total_pct: Decimal = profile.iter().map(|b| b.volume_pct).sum();
            if (total_pct - Decimal::from(100)).abs() > Decimal::from_str_exact("0.01").unwrap() {
                return Err(ExecutionError::Validation(format!(
                    "Volume profile percentages must sum to 100, got {}",
                    total_pct
                )));
            }
        }

        if let Some(rate) = self.participation_rate {
            if rate <= Decimal::ZERO || rate > Decimal::ONE {
                return Err(ExecutionError::Validation(
                    "Participation rate must be between 0 and 1".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Get volume distribution for buckets
    pub fn get_volume_distribution(&self) -> Vec<Decimal> {
        if let Some(profile) = &self.volume_profile {
            profile.iter().map(|b| b.volume_pct).collect()
        } else {
            // Uniform distribution
            let pct_per_bucket = Decimal::from(100) / Decimal::from(self.num_buckets);
            vec![pct_per_bucket; self.num_buckets]
        }
    }

    /// Calculate quantities for each bucket
    pub fn calculate_bucket_quantities(&self) -> Vec<Decimal> {
        let distribution = self.get_volume_distribution();
        let mut quantities = Vec::new();

        for pct in distribution {
            let qty = (self.total_quantity * pct) / Decimal::from(100);
            if qty >= self.min_order_size {
                quantities.push(qty);
            } else {
                quantities.push(Decimal::ZERO);
            }
        }

        quantities
    }

    /// Calculate bucket duration
    pub fn bucket_duration(&self) -> Duration {
        Duration::from_secs(self.duration_secs / self.num_buckets as u64)
    }
}

/// VWAP execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapState {
    /// Configuration
    pub config: VwapConfig,

    /// Current status
    pub status: VwapStatus,

    /// Total quantity filled
    pub filled_quantity: Decimal,

    /// Average fill price
    pub average_price: Decimal,

    /// VWAP benchmark price (if available)
    pub vwap_benchmark: Option<Decimal>,

    /// Number of child orders created
    pub orders_created: usize,

    /// Number of child orders filled
    pub orders_filled: usize,

    /// Number of child orders cancelled
    pub orders_cancelled: usize,

    /// Current bucket being executed
    pub current_bucket: usize,

    /// Filled quantity per bucket
    pub bucket_fills: HashMap<usize, Decimal>,

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
pub enum VwapStatus {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}

impl VwapState {
    /// Create new VWAP state
    pub fn new(config: VwapConfig) -> Self {
        Self {
            config,
            status: VwapStatus::Pending,
            filled_quantity: Decimal::ZERO,
            average_price: Decimal::ZERO,
            vwap_benchmark: None,
            orders_created: 0,
            orders_filled: 0,
            orders_cancelled: 0,
            current_bucket: 0,
            bucket_fills: HashMap::new(),
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
            VwapStatus::Completed | VwapStatus::Cancelled | VwapStatus::Failed
        )
    }

    /// Calculate slippage vs VWAP (if benchmark available)
    pub fn slippage(&self) -> Option<Decimal> {
        self.vwap_benchmark.map(|vwap| {
            if vwap > Decimal::ZERO {
                ((self.average_price - vwap) / vwap) * Decimal::from(100)
            } else {
                Decimal::ZERO
            }
        })
    }
}

/// VWAP strategy executor
pub struct VwapExecutor {
    state: Arc<RwLock<VwapState>>,
}

impl VwapExecutor {
    /// Create new VWAP executor
    pub fn new(config: VwapConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            state: Arc::new(RwLock::new(VwapState::new(config))),
        })
    }

    /// Get current state
    pub async fn state(&self) -> VwapState {
        self.state.read().await.clone()
    }

    /// Set VWAP benchmark price
    pub async fn set_benchmark(&self, vwap: Decimal) {
        let mut state = self.state.write().await;
        state.vwap_benchmark = Some(vwap);
    }

    /// Start execution
    pub async fn start<F>(&self, mut order_submitter: F) -> Result<VwapState>
    where
        F: FnMut(Order) -> Result<String> + Send,
    {
        let mut state = self.state.write().await;

        if state.status != VwapStatus::Pending {
            return Err(ExecutionError::InvalidOrderState(
                "VWAP already started".to_string(),
            ));
        }

        state.status = VwapStatus::Running;
        state.start_time = Some(Utc::now());
        let config = state.config.clone();
        drop(state);

        info!(
            "Starting VWAP: {} {} {} over {}s in {} buckets",
            config.side,
            config.total_quantity,
            config.symbol,
            config.duration_secs,
            config.num_buckets
        );

        // Calculate bucket quantities
        let bucket_quantities = config.calculate_bucket_quantities();
        let bucket_duration = config.bucket_duration();

        // Execute buckets
        let mut interval_timer = interval(bucket_duration);
        let mut total_filled = Decimal::ZERO;
        let mut total_cost = Decimal::ZERO;

        for (bucket_idx, &bucket_qty) in bucket_quantities.iter().enumerate() {
            // Wait for next bucket (skip first tick)
            if bucket_idx > 0 {
                interval_timer.tick().await;
            }

            // Update current bucket
            {
                let mut state = self.state.write().await;
                state.current_bucket = bucket_idx;
            }

            // Check if cancelled
            let current_status = self.state.read().await.status;
            if current_status == VwapStatus::Cancelled {
                info!(
                    "VWAP cancelled at bucket {}/{}",
                    bucket_idx + 1,
                    config.num_buckets
                );
                break;
            }

            if bucket_qty <= Decimal::ZERO {
                debug!(
                    "Bucket {} has zero quantity (below min order size)",
                    bucket_idx + 1
                );
                continue;
            }

            // Create child order for this bucket
            let child_order = Order::new(
                format!("vwap_bucket_{}_{}", bucket_idx + 1, uuid::Uuid::new_v4()),
                config.symbol.clone(),
                config.exchange.clone(),
                config.side,
                if config.use_limit_orders {
                    OrderTypeEnum::Limit
                } else {
                    OrderTypeEnum::Market
                },
                bucket_qty,
            );

            debug!(
                "Submitting VWAP bucket {}/{}: {} {} ({:.2}% of total volume)",
                bucket_idx + 1,
                config.num_buckets,
                bucket_qty,
                config.symbol,
                (bucket_qty / config.total_quantity) * Decimal::from(100)
            );

            // Submit order
            match order_submitter(child_order.clone()) {
                Ok(order_id) => {
                    let mut state = self.state.write().await;
                    state.orders_created += 1;
                    state.child_order_ids.push(order_id.clone());

                    // Simulate fill (in real implementation, wait for fill callback)
                    let fill_price = config.limit_price.unwrap_or(Decimal::from(50000));
                    total_filled += bucket_qty;
                    total_cost += bucket_qty * fill_price;

                    state.filled_quantity = total_filled;
                    state.average_price = if total_filled > Decimal::ZERO {
                        total_cost / total_filled
                    } else {
                        Decimal::ZERO
                    };
                    state.orders_filled += 1;
                    state.bucket_fills.insert(bucket_idx, bucket_qty);

                    info!(
                        "VWAP bucket {}/{} submitted: {} filled at {}",
                        bucket_idx + 1,
                        config.num_buckets,
                        bucket_qty,
                        fill_price
                    );
                }
                Err(e) => {
                    warn!("Failed to submit VWAP bucket {}: {}", bucket_idx + 1, e);
                    if !config.allow_partial {
                        let mut state = self.state.write().await;
                        state.status = VwapStatus::Failed;
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

        if state.status == VwapStatus::Cancelled {
            info!(
                "VWAP cancelled: filled {}/{}",
                state.filled_quantity, config.total_quantity
            );
        } else {
            state.status = VwapStatus::Completed;

            let slippage_msg = state
                .slippage()
                .map(|s| format!(" (slippage: {:.4}%)", s))
                .unwrap_or_default();

            info!(
                "VWAP completed: filled {}/{} at avg price {}{}",
                state.filled_quantity, config.total_quantity, state.average_price, slippage_msg
            );
        }

        Ok(state.clone())
    }

    /// Cancel execution
    pub async fn cancel(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if state.status != VwapStatus::Running {
            return Err(ExecutionError::InvalidOrderState(
                "VWAP not running".to_string(),
            ));
        }

        state.status = VwapStatus::Cancelled;
        info!("VWAP cancellation requested");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap_config_validation() {
        let config = VwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            volume_profile: None,
            num_buckets: 10,
            min_order_size: Decimal::from_str_exact("0.001").unwrap(),
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            participation_rate: Some(Decimal::from_str_exact("0.1").unwrap()),
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_volume_profile_validation() {
        let profile = vec![
            VolumeBucket {
                start_minute: 0,
                end_minute: 30,
                volume_pct: Decimal::from(40),
            },
            VolumeBucket {
                start_minute: 30,
                end_minute: 60,
                volume_pct: Decimal::from(60),
            },
        ];

        let config = VwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            volume_profile: Some(profile),
            num_buckets: 2,
            min_order_size: Decimal::from_str_exact("0.001").unwrap(),
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            participation_rate: None,
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_bucket_quantity_calculation() {
        let config = VwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            volume_profile: None,
            num_buckets: 10,
            min_order_size: Decimal::from_str_exact("0.001").unwrap(),
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            participation_rate: None,
        };

        let quantities = config.calculate_bucket_quantities();
        assert_eq!(quantities.len(), 10);

        // Each bucket should have 10 (100 / 10)
        for qty in &quantities {
            assert_eq!(*qty, Decimal::from(10));
        }
    }

    #[test]
    fn test_vwap_state_creation() {
        let config = VwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            volume_profile: None,
            num_buckets: 10,
            min_order_size: Decimal::from_str_exact("0.001").unwrap(),
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            participation_rate: None,
        };

        let state = VwapState::new(config);
        assert_eq!(state.status, VwapStatus::Pending);
        assert_eq!(state.filled_quantity, Decimal::ZERO);
        assert_eq!(state.current_bucket, 0);
    }

    #[test]
    fn test_slippage_calculation() {
        let config = VwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            volume_profile: None,
            num_buckets: 10,
            min_order_size: Decimal::from_str_exact("0.001").unwrap(),
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            participation_rate: None,
        };

        let mut state = VwapState::new(config);
        state.average_price = Decimal::from(50500);
        state.vwap_benchmark = Some(Decimal::from(50000));

        let slippage = state.slippage().unwrap();
        assert_eq!(slippage, Decimal::ONE); // 1% slippage
    }

    #[tokio::test]
    async fn test_vwap_executor_creation() {
        let config = VwapConfig {
            total_quantity: Decimal::from(100),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 60,
            volume_profile: None,
            num_buckets: 10,
            min_order_size: Decimal::from_str_exact("0.001").unwrap(),
            use_limit_orders: false,
            limit_price: None,
            max_price_deviation_pct: None,
            allow_partial: true,
            participation_rate: None,
        };

        let executor = VwapExecutor::new(config);
        assert!(executor.is_ok());

        let state = executor.unwrap().state().await;
        assert_eq!(state.status, VwapStatus::Pending);
    }

    #[tokio::test]
    async fn test_vwap_execution() {
        let config = VwapConfig {
            total_quantity: Decimal::from(10),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            duration_secs: 5,
            volume_profile: None,
            num_buckets: 5,
            min_order_size: Decimal::from_str_exact("0.001").unwrap(),
            use_limit_orders: true,
            limit_price: Some(Decimal::from(50000)),
            max_price_deviation_pct: None,
            allow_partial: true,
            participation_rate: None,
        };

        let executor = VwapExecutor::new(config).unwrap();

        // Mock order submitter
        let order_submitter =
            |order: Order| -> Result<String> { Ok(format!("order_{}", order.id)) };

        let result = executor.start(order_submitter).await;
        assert!(result.is_ok());

        let final_state = result.unwrap();
        assert_eq!(final_state.status, VwapStatus::Completed);
        assert_eq!(final_state.filled_quantity, Decimal::from(10));
        assert_eq!(final_state.orders_created, 5);
    }
}
