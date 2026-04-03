//! Risk management module for portfolio-level controls.
//!
//! This module provides comprehensive risk management tools including:
//! - Portfolio-level risk limits
//! - Drawdown-based position sizing
//! - Correlation-based signal filtering
//! - Kelly criterion optimization
//! - Risk budgeting across positions
//! - Volatility-based adjustments
//!
//! # Overview
//!
//! The risk management pipeline:
//! 1. **Signal Filtering** → Remove correlated/risky signals
//! 2. **Position Sizing** → Calculate optimal position sizes
//! 3. **Risk Limits** → Enforce portfolio-level constraints
//! 4. **Monitoring** → Track exposure and drawdown
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use vision::risk::*;
//!
//! // 1. Create risk manager
//! let config = RiskConfig {
//!     max_portfolio_risk: 0.20,      // Max 20% of capital at risk
//!     max_position_size: 0.10,       // Max 10% per position
//!     max_drawdown_limit: 0.15,      // Stop at 15% drawdown
//!     max_correlation: 0.7,          // Filter highly correlated signals
//!     ..Default::default()
//! };
//!
//! let mut risk_mgr = RiskManager::new(config);
//!
//! // 2. Filter signals
//! let filtered = risk_mgr.filter_signals(&signals, &prices)?;
//!
//! // 3. Calculate position sizes
//! for signal in filtered {
//!     let size = risk_mgr.calculate_position_size(&signal, current_capital, &prices)?;
//!     // Execute trade with size...
//! }
//!
//! // 4. Monitor risk
//! risk_mgr.update_positions(&positions);
//! if risk_mgr.is_risk_limit_exceeded() {
//!     // Close positions or reduce exposure
//! }
//! ```

pub mod correlation;
pub mod kelly;
pub mod limits;
pub mod monitor;
pub mod sizing;

// Re-exports
pub use correlation::{CorrelationFilter, CorrelationMatrix};
pub use kelly::{KellyCalculator, KellyOptimizer, OptimalFraction};
pub use limits::{DrawdownMonitor, RiskLimits, ViolationType};
pub use monitor::{PortfolioRisk, RiskMonitor, RiskReport};
pub use sizing::{
    PositionSizer, SizingMethod, VolatilityAdjuster, VolatilityEstimate, VolatilityWindow,
};

use crate::backtest::Position;
use crate::signals::TradingSignal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for risk management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum portfolio risk (fraction of capital)
    pub max_portfolio_risk: f64,

    /// Maximum position size (fraction of capital)
    pub max_position_size: f64,

    /// Maximum drawdown before stopping (fraction)
    pub max_drawdown_limit: f64,

    /// Maximum correlation between positions
    pub max_correlation: f64,

    /// Correlation lookback window (number of periods)
    pub correlation_window: usize,

    /// Enable Kelly criterion sizing
    pub use_kelly: bool,

    /// Kelly fraction multiplier (e.g., 0.5 for half-Kelly)
    pub kelly_fraction: f64,

    /// Enable volatility-based adjustments
    pub use_volatility_scaling: bool,

    /// Target volatility for scaling
    pub target_volatility: f64,

    /// Volatility lookback window
    pub volatility_window: usize,

    /// Risk-free rate for calculations
    pub risk_free_rate: f64,

    /// Maximum number of concurrent positions
    pub max_positions: usize,

    /// Maximum exposure per asset (fraction of capital)
    pub max_asset_exposure: f64,

    /// Minimum confidence for position entry
    pub min_confidence: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_portfolio_risk: 0.20,     // 20% max risk
            max_position_size: 0.10,      // 10% max per position
            max_drawdown_limit: 0.15,     // 15% max drawdown
            max_correlation: 0.7,         // Max 0.7 correlation
            correlation_window: 50,       // 50 periods lookback
            use_kelly: false,             // Conservative default
            kelly_fraction: 0.5,          // Half-Kelly
            use_volatility_scaling: true, // Enable vol scaling
            target_volatility: 0.15,      // 15% annual volatility
            volatility_window: 20,        // 20 periods
            risk_free_rate: 0.0,          // 0% risk-free rate
            max_positions: 5,             // Max 5 concurrent
            max_asset_exposure: 0.15,     // 15% per asset
            min_confidence: 0.6,          // 60% min confidence
        }
    }
}

/// Main risk management coordinator
pub struct RiskManager {
    config: RiskConfig,
    correlation_filter: CorrelationFilter,
    kelly_calculator: Option<KellyCalculator>,
    position_sizer: PositionSizer,
    risk_monitor: RiskMonitor,
    drawdown_monitor: DrawdownMonitor,
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(config: RiskConfig) -> Self {
        let correlation_filter =
            CorrelationFilter::new(config.max_correlation, config.correlation_window);

        let kelly_calculator = if config.use_kelly {
            Some(KellyCalculator::new(config.kelly_fraction))
        } else {
            None
        };

        let position_sizer = PositionSizer::new(
            config.max_position_size,
            config.use_volatility_scaling,
            config.target_volatility,
            config.volatility_window,
        );

        let risk_monitor = RiskMonitor::new(config.max_portfolio_risk, config.max_positions);

        let drawdown_monitor = DrawdownMonitor::new(config.max_drawdown_limit);

        Self {
            config,
            correlation_filter,
            kelly_calculator,
            position_sizer,
            risk_monitor,
            drawdown_monitor,
        }
    }

    /// Filter signals based on correlation and risk limits
    pub fn filter_signals(
        &mut self,
        signals: &[TradingSignal],
        _prices: &HashMap<String, f64>,
    ) -> Result<Vec<TradingSignal>, RiskError> {
        let mut filtered = Vec::new();

        // First, filter by confidence
        for signal in signals {
            if signal.confidence < self.config.min_confidence {
                continue;
            }

            // Check if we can add this position
            if !self.risk_monitor.can_add_position() {
                continue;
            }

            // Check correlation with existing positions
            if self
                .correlation_filter
                .should_filter(&signal.asset, &filtered)
            {
                continue;
            }

            filtered.push(signal.clone());
        }

        Ok(filtered)
    }

    /// Calculate optimal position size for a signal
    pub fn calculate_position_size(
        &mut self,
        signal: &TradingSignal,
        capital: f64,
        price: f64,
        volatility: Option<f64>,
    ) -> Result<f64, RiskError> {
        // Check drawdown
        if self.drawdown_monitor.is_stopped() {
            return Ok(0.0);
        }

        // Base size from position sizer
        let mut size = self
            .position_sizer
            .calculate_size(capital, price, volatility)?;

        // Apply Kelly sizing if enabled
        if let Some(ref kelly_calc) = self.kelly_calculator {
            let kelly_size = kelly_calc.calculate_size(capital, price, signal.confidence);
            size = size.min(kelly_size); // Take the smaller
        }

        // Apply drawdown scaling
        let drawdown_scale = self.drawdown_monitor.get_scaling_factor();
        size *= drawdown_scale;

        // Ensure within limits
        let max_size = (capital * self.config.max_position_size) / price;
        size = size.min(max_size);

        Ok(size)
    }

    /// Update risk monitor with current positions
    pub fn update_positions(&mut self, positions: &HashMap<String, Position>) {
        self.risk_monitor.update_positions(positions);
    }

    /// Update drawdown monitor with current equity
    pub fn update_equity(&mut self, current_equity: f64, peak_equity: f64) {
        self.drawdown_monitor.update(current_equity, peak_equity);
    }

    /// Check if risk limits are exceeded
    pub fn is_risk_limit_exceeded(&self) -> bool {
        self.risk_monitor.is_limit_exceeded() || self.drawdown_monitor.is_stopped()
    }

    /// Get current risk report
    pub fn get_risk_report(&self) -> RiskReport {
        self.risk_monitor.generate_report()
    }

    /// Get current drawdown percentage
    pub fn current_drawdown(&self) -> f64 {
        self.drawdown_monitor.current_drawdown_pct()
    }

    /// Check if trading should be stopped
    pub fn should_stop_trading(&self) -> bool {
        self.drawdown_monitor.is_stopped()
    }

    /// Reset risk manager state
    pub fn reset(&mut self) {
        self.drawdown_monitor.reset();
        self.correlation_filter.reset();
    }

    /// Get configuration
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }
}

/// Risk management errors
#[derive(Debug, Clone)]
pub enum RiskError {
    /// Risk limit would be exceeded
    RiskLimitExceeded(String),

    /// Invalid configuration
    InvalidConfig(String),

    /// Calculation error
    CalculationError(String),

    /// Insufficient data
    InsufficientData(String),
}

impl std::fmt::Display for RiskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskError::RiskLimitExceeded(msg) => write!(f, "Risk limit exceeded: {}", msg),
            RiskError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            RiskError::CalculationError(msg) => write!(f, "Calculation error: {}", msg),
            RiskError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
        }
    }
}

impl std::error::Error for RiskError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_config_default() {
        let config = RiskConfig::default();
        assert_eq!(config.max_portfolio_risk, 0.20);
        assert_eq!(config.max_position_size, 0.10);
        assert_eq!(config.max_drawdown_limit, 0.15);
    }

    #[test]
    fn test_risk_manager_creation() {
        let config = RiskConfig::default();
        let mgr = RiskManager::new(config);
        assert!(!mgr.is_risk_limit_exceeded());
    }
}
