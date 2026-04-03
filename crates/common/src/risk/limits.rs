//! Risk Limits Module
//!
//! Provides configurable risk limits and validation for trading operations
//! to prevent catastrophic losses in prop firm trading scenarios.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Risk limit violation errors
#[derive(Error, Debug, Clone)]
pub enum RiskError {
    #[error("Order size ${0:.2} exceeds maximum allowed ${1:.2}")]
    OrderTooLarge(f64, f64),

    #[error("Total exposure ${0:.2} would exceed limit ${1:.2}")]
    ExposureLimitExceeded(f64, f64),

    #[error("Maximum number of open positions ({0}) reached")]
    TooManyPositions(usize),

    #[error("Daily loss limit hit: -${0:.2} (limit: ${1:.2})")]
    DailyLossLimitHit(f64, f64),

    #[error("Drawdown {0:.2}% exceeds maximum allowed {1:.2}%")]
    DrawdownLimitExceeded(f64, f64),

    #[error("Risk limits are disabled - order rejected for safety")]
    RiskLimitsDisabled,

    #[error("Invalid risk configuration: {0}")]
    InvalidConfiguration(String),
}

/// Risk limits configuration
///
/// All monetary values are in USD. These limits provide hard stops
/// to prevent prop firm account blow-ups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Enable/disable risk limit enforcement
    /// WARNING: Should always be true in production!
    pub enabled: bool,

    /// Maximum USD value for a single order
    /// Prevents accidentally placing massive orders
    #[serde(default = "default_max_order_size")]
    pub max_order_size_usd: f64,

    /// Maximum total USD exposure across all positions
    /// Sum of absolute values of all position sizes
    #[serde(default = "default_max_position_size")]
    pub max_position_size_usd: f64,

    /// Maximum daily loss in USD (positive number)
    /// Trading stops when daily P&L drops below -this_value
    #[serde(default = "default_max_daily_loss")]
    pub max_daily_loss_usd: f64,

    /// Maximum drawdown from peak equity (percentage)
    /// E.g., 10.0 means stop if account drops 10% from high water mark
    #[serde(default = "default_max_drawdown")]
    pub max_drawdown_pct: f64,

    /// Maximum number of simultaneously open positions
    /// Prevents over-diversification and reduces risk
    #[serde(default = "default_max_open_positions")]
    pub max_open_positions: usize,

    /// Maximum leverage multiplier
    /// E.g., 3.0 means max 3x leverage on margin trading
    #[serde(default = "default_max_leverage")]
    pub max_leverage: f64,
}

// Default values for risk limits (conservative for prop trading)
fn default_max_order_size() -> f64 {
    10_000.0 // $10k max per order
}

fn default_max_position_size() -> f64 {
    100_000.0 // $100k total exposure
}

fn default_max_daily_loss() -> f64 {
    5_000.0 // $5k max loss per day
}

fn default_max_drawdown() -> f64 {
    10.0 // 10% max drawdown from peak
}

fn default_max_open_positions() -> usize {
    20 // Max 20 concurrent positions
}

fn default_max_leverage() -> f64 {
    2.0 // Max 2x leverage
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            enabled: true,
            max_order_size_usd: default_max_order_size(),
            max_position_size_usd: default_max_position_size(),
            max_daily_loss_usd: default_max_daily_loss(),
            max_drawdown_pct: default_max_drawdown(),
            max_open_positions: default_max_open_positions(),
            max_leverage: default_max_leverage(),
        }
    }
}

impl RiskLimits {
    /// Validate configuration values
    pub fn validate(&self) -> Result<(), RiskError> {
        if self.max_order_size_usd <= 0.0 {
            return Err(RiskError::InvalidConfiguration(
                "max_order_size_usd must be positive".to_string(),
            ));
        }
        if self.max_position_size_usd <= 0.0 {
            return Err(RiskError::InvalidConfiguration(
                "max_position_size_usd must be positive".to_string(),
            ));
        }
        if self.max_daily_loss_usd <= 0.0 {
            return Err(RiskError::InvalidConfiguration(
                "max_daily_loss_usd must be positive".to_string(),
            ));
        }
        if self.max_drawdown_pct <= 0.0 || self.max_drawdown_pct > 100.0 {
            return Err(RiskError::InvalidConfiguration(
                "max_drawdown_pct must be between 0 and 100".to_string(),
            ));
        }
        if self.max_open_positions == 0 {
            return Err(RiskError::InvalidConfiguration(
                "max_open_positions must be at least 1".to_string(),
            ));
        }
        if self.max_leverage <= 0.0 {
            return Err(RiskError::InvalidConfiguration(
                "max_leverage must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Order to be validated against risk limits
#[derive(Debug, Clone)]
pub struct Order {
    pub symbol: String,
    pub size_usd: f64,
    pub leverage: f64,
}

/// Current portfolio state for risk checking
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    /// Current open positions
    pub positions: Vec<Position>,
    /// Peak equity (high water mark) in USD
    pub peak_equity_usd: f64,
    /// Current equity in USD
    pub current_equity_usd: f64,
    /// Daily P&L in USD (can be negative)
    pub daily_pnl_usd: f64,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub size_usd: f64,
}

impl PortfolioState {
    /// Calculate total exposure across all positions
    pub fn total_exposure_usd(&self) -> f64 {
        self.positions.iter().map(|p| p.size_usd.abs()).sum()
    }

    /// Calculate current drawdown percentage from peak
    pub fn drawdown_pct(&self) -> f64 {
        if self.peak_equity_usd <= 0.0 {
            return 0.0;
        }
        ((self.peak_equity_usd - self.current_equity_usd) / self.peak_equity_usd) * 100.0
    }

    /// Number of open positions
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }
}

/// Risk checker validates orders against configured limits
pub struct RiskChecker {
    limits: RiskLimits,
}

impl RiskChecker {
    /// Create new risk checker with given limits
    pub fn new(limits: RiskLimits) -> Result<Self, RiskError> {
        limits.validate()?;
        Ok(Self { limits })
    }

    /// Check if order violates risk limits
    ///
    /// Returns Ok(()) if order is safe, Err(RiskError) if it violates limits
    pub fn check_order(&self, order: &Order, portfolio: &PortfolioState) -> Result<(), RiskError> {
        // If risk limits disabled, reject for safety (fail-closed)
        if !self.limits.enabled {
            return Err(RiskError::RiskLimitsDisabled);
        }

        // Check 1: Order size limit
        if order.size_usd > self.limits.max_order_size_usd {
            return Err(RiskError::OrderTooLarge(
                order.size_usd,
                self.limits.max_order_size_usd,
            ));
        }

        // Check 2: Total exposure limit (current + new order)
        let new_total_exposure = portfolio.total_exposure_usd() + order.size_usd;
        if new_total_exposure > self.limits.max_position_size_usd {
            return Err(RiskError::ExposureLimitExceeded(
                new_total_exposure,
                self.limits.max_position_size_usd,
            ));
        }

        // Check 3: Maximum open positions
        // (Only count if opening a new position, not adding to existing)
        let is_new_position = !portfolio.positions.iter().any(|p| p.symbol == order.symbol);
        if is_new_position && portfolio.position_count() >= self.limits.max_open_positions {
            return Err(RiskError::TooManyPositions(self.limits.max_open_positions));
        }

        // Check 4: Daily loss limit
        if portfolio.daily_pnl_usd < -self.limits.max_daily_loss_usd {
            return Err(RiskError::DailyLossLimitHit(
                portfolio.daily_pnl_usd.abs(),
                self.limits.max_daily_loss_usd,
            ));
        }

        // Check 5: Drawdown limit
        let drawdown = portfolio.drawdown_pct();
        if drawdown > self.limits.max_drawdown_pct {
            return Err(RiskError::DrawdownLimitExceeded(
                drawdown,
                self.limits.max_drawdown_pct,
            ));
        }

        // Check 6: Leverage limit
        if order.leverage > self.limits.max_leverage {
            return Err(RiskError::InvalidConfiguration(format!(
                "Order leverage {:.2}x exceeds maximum {:.2}x",
                order.leverage, self.limits.max_leverage
            )));
        }

        Ok(())
    }

    /// Get current limits
    pub fn limits(&self) -> &RiskLimits {
        &self.limits
    }

    /// Calculate risk utilization percentages
    pub fn utilization(&self, portfolio: &PortfolioState) -> RiskUtilization {
        RiskUtilization {
            exposure_pct: (portfolio.total_exposure_usd() / self.limits.max_position_size_usd)
                * 100.0,
            position_count_pct: (portfolio.position_count() as f64
                / self.limits.max_open_positions as f64)
                * 100.0,
            daily_loss_pct: (portfolio.daily_pnl_usd.abs() / self.limits.max_daily_loss_usd)
                * 100.0,
            drawdown_pct: portfolio.drawdown_pct(),
        }
    }
}

/// Risk utilization metrics (percentage of limits used)
#[derive(Debug, Clone, Serialize)]
pub struct RiskUtilization {
    /// Percentage of max exposure used
    pub exposure_pct: f64,
    /// Percentage of max positions used
    pub position_count_pct: f64,
    /// Percentage of daily loss limit used
    pub daily_loss_pct: f64,
    /// Current drawdown percentage
    pub drawdown_pct: f64,
}

impl fmt::Display for RiskUtilization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Exposure: {:.1}%, Positions: {:.1}%, Daily Loss: {:.1}%, Drawdown: {:.1}%",
            self.exposure_pct, self.position_count_pct, self.daily_loss_pct, self.drawdown_pct
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_limits() -> RiskLimits {
        RiskLimits {
            enabled: true,
            max_order_size_usd: 10_000.0,
            max_position_size_usd: 50_000.0,
            max_daily_loss_usd: 2_000.0,
            max_drawdown_pct: 10.0,
            max_open_positions: 5,
            max_leverage: 2.0,
        }
    }

    fn test_portfolio() -> PortfolioState {
        PortfolioState {
            positions: vec![
                Position {
                    symbol: "BTC".to_string(),
                    size_usd: 20_000.0,
                },
                Position {
                    symbol: "ETH".to_string(),
                    size_usd: 15_000.0,
                },
            ],
            peak_equity_usd: 100_000.0,
            current_equity_usd: 95_000.0,
            daily_pnl_usd: -500.0,
        }
    }

    #[test]
    fn test_order_within_limits() {
        let checker = RiskChecker::new(test_limits()).unwrap();
        let order = Order {
            symbol: "SOL".to_string(),
            size_usd: 5_000.0,
            leverage: 1.0,
        };
        let portfolio = test_portfolio();

        assert!(checker.check_order(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_order_too_large() {
        let checker = RiskChecker::new(test_limits()).unwrap();
        let order = Order {
            symbol: "BTC".to_string(),
            size_usd: 15_000.0,
            leverage: 1.0,
        };
        let portfolio = test_portfolio();

        match checker.check_order(&order, &portfolio) {
            Err(RiskError::OrderTooLarge(_, _)) => (),
            _ => panic!("Expected OrderTooLarge error"),
        }
    }

    #[test]
    fn test_exposure_limit_exceeded() {
        let checker = RiskChecker::new(test_limits()).unwrap();
        // Order size is within max_order_size (10k) but would exceed total exposure limit (50k)
        // Current portfolio has 35k (20k BTC + 15k ETH), adding 10k SOL = 45k, still ok
        // We need a portfolio closer to the limit
        let mut portfolio = test_portfolio();
        // Increase existing positions to get closer to limit: 25k + 20k = 45k
        portfolio.positions = vec![
            Position {
                symbol: "BTC".to_string(),
                size_usd: 25_000.0,
            },
            Position {
                symbol: "ETH".to_string(),
                size_usd: 20_000.0,
            },
        ];
        let order = Order {
            symbol: "SOL".to_string(),
            size_usd: 10_000.0, // Would bring total to 55k > 50k limit
            leverage: 1.0,
        };

        match checker.check_order(&order, &portfolio) {
            Err(RiskError::ExposureLimitExceeded(_, _)) => (),
            other => panic!("Expected ExposureLimitExceeded error, got {:?}", other),
        }
    }

    #[test]
    fn test_too_many_positions() {
        let checker = RiskChecker::new(test_limits()).unwrap();
        let mut portfolio = test_portfolio();
        portfolio.positions = vec![
            Position {
                symbol: "BTC".to_string(),
                size_usd: 5_000.0,
            },
            Position {
                symbol: "ETH".to_string(),
                size_usd: 5_000.0,
            },
            Position {
                symbol: "SOL".to_string(),
                size_usd: 5_000.0,
            },
            Position {
                symbol: "AVAX".to_string(),
                size_usd: 5_000.0,
            },
            Position {
                symbol: "MATIC".to_string(),
                size_usd: 5_000.0,
            },
        ];

        let order = Order {
            symbol: "DOGE".to_string(),
            size_usd: 1_000.0,
            leverage: 1.0,
        };

        match checker.check_order(&order, &portfolio) {
            Err(RiskError::TooManyPositions(_)) => (),
            _ => panic!("Expected TooManyPositions error"),
        }
    }

    #[test]
    fn test_daily_loss_limit_hit() {
        let checker = RiskChecker::new(test_limits()).unwrap();
        let mut portfolio = test_portfolio();
        portfolio.daily_pnl_usd = -2_500.0; // Exceeds -2000 limit

        let order = Order {
            symbol: "BTC".to_string(),
            size_usd: 1_000.0,
            leverage: 1.0,
        };

        match checker.check_order(&order, &portfolio) {
            Err(RiskError::DailyLossLimitHit(_, _)) => (),
            _ => panic!("Expected DailyLossLimitHit error"),
        }
    }

    #[test]
    fn test_drawdown_limit_exceeded() {
        let checker = RiskChecker::new(test_limits()).unwrap();
        let mut portfolio = test_portfolio();
        portfolio.current_equity_usd = 85_000.0; // 15% drawdown from 100k

        let order = Order {
            symbol: "BTC".to_string(),
            size_usd: 1_000.0,
            leverage: 1.0,
        };

        match checker.check_order(&order, &portfolio) {
            Err(RiskError::DrawdownLimitExceeded(_, _)) => (),
            _ => panic!("Expected DrawdownLimitExceeded error"),
        }
    }

    #[test]
    fn test_risk_utilization() {
        let checker = RiskChecker::new(test_limits()).unwrap();
        let portfolio = test_portfolio();
        let util = checker.utilization(&portfolio);

        assert!((util.exposure_pct - 70.0).abs() < 0.1); // 35k / 50k = 70%
        assert!((util.position_count_pct - 40.0).abs() < 0.1); // 2 / 5 = 40%
        assert!((util.daily_loss_pct - 25.0).abs() < 0.1); // 500 / 2000 = 25%
        assert!((util.drawdown_pct - 5.0).abs() < 0.1); // 5k / 100k = 5%
    }

    #[test]
    fn test_disabled_limits_reject() {
        let mut limits = test_limits();
        limits.enabled = false;
        let checker = RiskChecker::new(limits).unwrap();

        let order = Order {
            symbol: "BTC".to_string(),
            size_usd: 1_000.0,
            leverage: 1.0,
        };
        let portfolio = test_portfolio();

        match checker.check_order(&order, &portfolio) {
            Err(RiskError::RiskLimitsDisabled) => (),
            _ => panic!("Expected RiskLimitsDisabled error"),
        }
    }
}
