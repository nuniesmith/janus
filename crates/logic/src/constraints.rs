//! Typestate constraints for compile-time safety.
//!
//! Uses Rust's type system to enforce that orders are verified
//! before execution, preventing runtime errors.

use common::{JanusError, Order, Result, RiskEngine};

/// Typestate marker for unchecked orders
#[derive(Debug, Clone, Copy)]
pub struct Unchecked;

/// Typestate marker for verified orders
#[derive(Debug, Clone, Copy)]
pub struct Verified;

/// Order with typestate - ensures orders are verified before execution
pub struct TypedOrder<State> {
    pub order: Order,
    _state: std::marker::PhantomData<State>,
}

impl TypedOrder<Unchecked> {
    /// Create a new unchecked order
    pub fn new(order: Order) -> Self {
        Self {
            order,
            _state: std::marker::PhantomData,
        }
    }

    /// Verify the order against risk constraints
    /// Returns a Verified order if compliant, or an error if not
    pub fn verify(self, risk_engine: &dyn RiskEngine) -> Result<TypedOrder<Verified>> {
        if !risk_engine.is_valid(&self.order) {
            return Err(JanusError::RiskViolation(
                "Order failed risk validation".to_string(),
            ));
        }

        let compliance = risk_engine.verify_order(&self.order)?;
        if compliance < 0.5 {
            return Err(JanusError::RiskViolation(format!(
                "Order compliance score too low: {:.2}",
                compliance
            )));
        }

        Ok(TypedOrder {
            order: self.order,
            _state: std::marker::PhantomData,
        })
    }
}

impl TypedOrder<Verified> {
    /// Get the inner order (only available for verified orders)
    pub fn into_order(self) -> Order {
        self.order
    }

    /// Get a reference to the order
    pub fn order(&self) -> &Order {
        &self.order
    }
}

/// Wash sale constraint - prevents buying and selling the same asset
/// within a short time window (for tax compliance)
///
/// IRS wash sale rule: Cannot claim a loss on a security if you buy
/// a "substantially identical" security within 30 days before or after.
pub struct WashSaleConstraint {
    /// Time window in seconds to check for wash sales
    window_seconds: u64,
}

impl WashSaleConstraint {
    /// Create a new wash sale constraint
    ///
    /// # Arguments
    /// * `window_seconds` - Time window to check for wash sales (default IRS rule is 30 days = 2,592,000 seconds)
    pub fn new(window_seconds: u64) -> Self {
        Self { window_seconds }
    }

    /// Create with default 30-day window (IRS standard)
    pub fn default_30_day() -> Self {
        Self::new(30 * 24 * 60 * 60) // 30 days in seconds
    }

    /// Check if an order would violate wash sale rules
    ///
    /// Returns `true` if the order is compliant (no wash sale violation)
    /// Returns `false` if a wash sale violation would occur
    ///
    /// A wash sale occurs when:
    /// 1. You sell a security at a loss
    /// 2. Within the window period (before or after), you buy the same or substantially identical security
    pub fn check(&self, order: &Order, recent_trades: &[common::Trade]) -> bool {
        use chrono::Duration;

        // Only check if this is a buy order (wash sale occurs when buying after selling at loss)
        if order.side != common::OrderSide::Buy {
            return true; // Sell orders don't trigger wash sale by themselves
        }

        let order_time = order.timestamp;
        let window_duration = Duration::seconds(self.window_seconds as i64);
        let window_start = order_time - window_duration;
        let window_end = order_time + window_duration;

        // Look for recent sells of the same symbol within the window
        for trade in recent_trades {
            // Must be a sell of the same symbol
            if trade.symbol != order.symbol || trade.side != common::OrderSide::Sell {
                continue;
            }

            // Check if trade is within the wash sale window
            if trade.timestamp >= window_start && trade.timestamp <= window_end {
                // This is a potential wash sale situation
                // In a real implementation, we'd also check if the sell was at a loss
                // by comparing with the cost basis, but that requires position tracking

                // For now, flag any buy within window of a sell as potential wash sale
                return false; // Wash sale violation detected
            }
        }

        true // No wash sale violation
    }

    /// Check with loss verification
    ///
    /// More accurate wash sale detection that considers whether the sale was at a loss
    ///
    /// # Arguments
    /// * `order` - The buy order to check
    /// * `recent_trades` - Recent trades to check against
    /// * `cost_basis` - The average cost basis for the symbol (if known)
    pub fn check_with_loss_verification(
        &self,
        order: &Order,
        recent_trades: &[common::Trade],
        cost_basis: Option<f64>,
    ) -> WashSaleResult {
        use chrono::Duration;

        if order.side != common::OrderSide::Buy {
            return WashSaleResult::Compliant;
        }

        let order_time = order.timestamp;
        let window_duration = Duration::seconds(self.window_seconds as i64);
        let window_start = order_time - window_duration;
        let window_end = order_time + window_duration;

        let mut potential_violations = Vec::new();

        for trade in recent_trades {
            if trade.symbol != order.symbol || trade.side != common::OrderSide::Sell {
                continue;
            }

            if trade.timestamp >= window_start && trade.timestamp <= window_end {
                let sale_price = trade.price.value();

                // Check if this was a loss sale
                let is_loss = match cost_basis {
                    Some(basis) => sale_price < basis,
                    None => true, // Assume potential loss if cost basis unknown
                };

                if is_loss {
                    potential_violations.push(WashSaleViolation {
                        sell_trade_id: trade.id,
                        sell_price: sale_price,
                        sell_timestamp: trade.timestamp,
                        cost_basis,
                    });
                }
            }
        }

        if potential_violations.is_empty() {
            WashSaleResult::Compliant
        } else {
            WashSaleResult::Violation(potential_violations)
        }
    }
}

/// Result of wash sale check
#[derive(Debug, Clone)]
pub enum WashSaleResult {
    /// No wash sale violation
    Compliant,
    /// One or more wash sale violations detected
    Violation(Vec<WashSaleViolation>),
}

/// Details of a wash sale violation
#[derive(Debug, Clone)]
pub struct WashSaleViolation {
    /// ID of the sell trade that triggered the wash sale
    pub sell_trade_id: uuid::Uuid,
    /// Price at which the security was sold
    pub sell_price: f64,
    /// Timestamp of the sell trade
    pub sell_timestamp: chrono::DateTime<chrono::Utc>,
    /// Cost basis if known
    pub cost_basis: Option<f64>,
}

/// Risk limit constraint - enforces position size limits
pub struct RiskLimitConstraint {
    max_position_size: f64,
    #[allow(dead_code)]
    max_daily_loss: f64,
}

impl RiskLimitConstraint {
    pub fn new(max_position_size: f64, max_daily_loss: f64) -> Self {
        Self {
            max_position_size,
            max_daily_loss,
        }
    }

    /// Check if an order violates risk limits
    pub fn check(&self, order: &Order, current_position: f64) -> Result<f64> {
        let new_position = current_position
            + if order.side == common::OrderSide::Buy {
                order.quantity.value()
            } else {
                -order.quantity.value()
            };

        if new_position.abs() > self.max_position_size {
            return Err(JanusError::RiskViolation(format!(
                "Position size {} exceeds limit {}",
                new_position.abs(),
                self.max_position_size
            )));
        }

        // Return compliance score
        let compliance = 1.0 - (new_position.abs() / self.max_position_size).min(1.0);
        Ok(compliance)
    }
}
