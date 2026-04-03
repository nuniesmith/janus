//! # Risk Limits and Validation Module
//!
//! Validates trading signals against configured risk limits:
//! - Maximum position size limits
//! - Portfolio exposure limits
//! - Maximum concurrent positions
//! - Daily loss limits
//! - Per-symbol exposure limits
//! - Correlation-based risk checks

use crate::risk::{PortfolioState, RiskConfig, RiskError};
use crate::signal::types::TradingSignal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum position size as dollar amount
    pub max_position_value: f64,

    /// Maximum portfolio exposure as dollar amount
    pub max_portfolio_exposure: f64,

    /// Maximum concurrent positions
    pub max_concurrent_positions: usize,

    /// Maximum daily loss as dollar amount
    pub max_daily_loss: f64,

    /// Maximum exposure per symbol (dollar amount)
    pub max_symbol_exposure: f64,

    /// Minimum confidence threshold for signals
    pub min_confidence: f64,

    /// Minimum strength threshold for signals
    pub min_strength: f64,

    /// Enable correlation checks
    pub check_correlation: bool,

    /// Maximum correlation allowed between positions (0.0 to 1.0)
    pub max_correlation: f64,
}

impl RiskLimits {
    /// Create risk limits from risk config
    pub fn from_config(config: &RiskConfig) -> Self {
        Self {
            max_position_value: config.account_balance * config.max_position_size_pct,
            max_portfolio_exposure: config.account_balance * config.max_portfolio_exposure_pct,
            max_concurrent_positions: config.max_concurrent_positions,
            max_daily_loss: config.account_balance * config.max_daily_loss_pct,
            max_symbol_exposure: config.account_balance * config.per_symbol_exposure_pct,
            min_confidence: 0.5, // Default minimum confidence
            min_strength: 0.3,   // Default minimum strength
            check_correlation: config.check_correlation,
            max_correlation: 0.7, // Don't allow highly correlated positions
        }
    }

    /// Create custom risk limits
    pub fn new(
        max_position_value: f64,
        max_portfolio_exposure: f64,
        max_concurrent_positions: usize,
        max_daily_loss: f64,
    ) -> Self {
        Self {
            max_position_value,
            max_portfolio_exposure,
            max_concurrent_positions,
            max_daily_loss,
            max_symbol_exposure: max_position_value,
            min_confidence: 0.5,
            min_strength: 0.3,
            check_correlation: false,
            max_correlation: 0.7,
        }
    }

    /// Set minimum quality thresholds
    pub fn with_quality_thresholds(mut self, min_confidence: f64, min_strength: f64) -> Self {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
        self.min_strength = min_strength.clamp(0.0, 1.0);
        self
    }

    /// Set correlation checking
    pub fn with_correlation_check(mut self, enabled: bool, max_correlation: f64) -> Self {
        self.check_correlation = enabled;
        self.max_correlation = max_correlation.clamp(0.0, 1.0);
        self
    }
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_value: 1000.0,
            max_portfolio_exposure: 5000.0,
            max_concurrent_positions: 5,
            max_daily_loss: 500.0,
            max_symbol_exposure: 2000.0,
            min_confidence: 0.5,
            min_strength: 0.3,
            check_correlation: false,
            max_correlation: 0.7,
        }
    }
}

/// Risk validator that enforces risk limits
pub struct RiskValidator {
    limits: RiskLimits,
}

impl RiskValidator {
    pub fn new(limits: RiskLimits) -> Self {
        Self { limits }
    }

    /// Validate a signal against all risk limits
    pub fn validate_signal(
        &self,
        signal: &TradingSignal,
        portfolio: &PortfolioState,
    ) -> Result<(), RiskError> {
        // Check signal quality
        self.validate_signal_quality(signal)?;

        // Check position count limit
        self.validate_position_count(portfolio)?;

        // Check daily loss limit
        self.validate_daily_loss(portfolio)?;

        // Check portfolio exposure would not exceed limit
        self.validate_portfolio_exposure(signal, portfolio)?;

        // Check per-symbol exposure
        self.validate_symbol_exposure(signal, portfolio)?;

        Ok(())
    }

    /// Validate signal quality (confidence and strength)
    fn validate_signal_quality(&self, signal: &TradingSignal) -> Result<(), RiskError> {
        if signal.confidence < self.limits.min_confidence {
            return Err(RiskError::CalculationError {
                reason: format!(
                    "Signal confidence {} below minimum {}",
                    signal.confidence, self.limits.min_confidence
                ),
            });
        }

        if signal.strength < self.limits.min_strength {
            return Err(RiskError::CalculationError {
                reason: format!(
                    "Signal strength {} below minimum {}",
                    signal.strength, self.limits.min_strength
                ),
            });
        }

        Ok(())
    }

    /// Validate position count doesn't exceed maximum
    fn validate_position_count(&self, portfolio: &PortfolioState) -> Result<(), RiskError> {
        if portfolio.position_count() >= self.limits.max_concurrent_positions {
            return Err(RiskError::MaxPositionsReached {
                max: self.limits.max_concurrent_positions,
            });
        }
        Ok(())
    }

    /// Validate daily loss hasn't exceeded limit
    fn validate_daily_loss(&self, portfolio: &PortfolioState) -> Result<(), RiskError> {
        // Daily loss is negative P&L
        let daily_loss = -portfolio.daily_pnl.min(0.0);

        if daily_loss >= self.limits.max_daily_loss {
            return Err(RiskError::DailyLossLimitReached {
                actual: daily_loss,
                max: self.limits.max_daily_loss,
            });
        }

        Ok(())
    }

    /// Validate portfolio exposure with new position
    fn validate_portfolio_exposure(
        &self,
        signal: &TradingSignal,
        portfolio: &PortfolioState,
    ) -> Result<(), RiskError> {
        let current_exposure = portfolio.total_exposure();

        // Estimate new position value
        let new_position_value = if let Some(entry_price) = signal.entry_price {
            // We don't know exact quantity yet, so use a very conservative estimate
            // Assume position will be at most 1% of entry price as quantity
            entry_price * 0.01
        } else {
            self.limits.max_position_value * 0.1 // Very conservative
        };

        let total_exposure = current_exposure + new_position_value;

        if total_exposure > self.limits.max_portfolio_exposure {
            return Err(RiskError::PortfolioExposureLimitExceeded {
                actual: total_exposure,
                max: self.limits.max_portfolio_exposure,
            });
        }

        Ok(())
    }

    /// Validate per-symbol exposure
    fn validate_symbol_exposure(
        &self,
        signal: &TradingSignal,
        portfolio: &PortfolioState,
    ) -> Result<(), RiskError> {
        let current_symbol_exposure = portfolio.exposure_for_symbol(&signal.symbol);

        // Estimate new position contribution
        let new_position_value = if let Some(entry_price) = signal.entry_price {
            // Very conservative estimate - assume 1% of price as quantity
            entry_price * 0.01
        } else {
            self.limits.max_position_value * 0.1
        };

        let total_symbol_exposure = current_symbol_exposure + new_position_value;

        if total_symbol_exposure > self.limits.max_symbol_exposure {
            return Err(RiskError::SymbolExposureLimitExceeded {
                symbol: signal.symbol.clone(),
                actual: total_symbol_exposure,
                max: self.limits.max_symbol_exposure,
            });
        }

        Ok(())
    }

    /// Validate position size against limit
    pub fn validate_position_size(&self, position_value: f64) -> Result<(), RiskError> {
        if position_value > self.limits.max_position_value {
            return Err(RiskError::PositionSizeLimitExceeded {
                actual: position_value,
                max: self.limits.max_position_value,
            });
        }
        Ok(())
    }

    /// Update risk limits
    pub fn update_limits(&mut self, limits: RiskLimits) {
        self.limits = limits;
    }

    /// Get current limits
    pub fn limits(&self) -> &RiskLimits {
        &self.limits
    }
}

/// Correlation calculator for position correlation checks
pub struct CorrelationCalculator {
    /// Historical correlation matrix (symbol pairs -> correlation)
    correlations: HashMap<(String, String), f64>,
}

impl CorrelationCalculator {
    pub fn new() -> Self {
        Self {
            correlations: HashMap::new(),
        }
    }

    /// Add correlation between two symbols
    pub fn add_correlation(&mut self, symbol1: String, symbol2: String, correlation: f64) {
        let key = Self::create_key(&symbol1, &symbol2);
        self.correlations.insert(key, correlation.clamp(-1.0, 1.0));
    }

    /// Get correlation between two symbols
    pub fn get_correlation(&self, symbol1: &str, symbol2: &str) -> Option<f64> {
        let key = Self::create_key(symbol1, symbol2);
        self.correlations.get(&key).copied()
    }

    /// Check if adding a position would violate correlation limits
    pub fn check_correlation_risk(
        &self,
        new_symbol: &str,
        portfolio: &PortfolioState,
        max_correlation: f64,
    ) -> Result<(), RiskError> {
        for existing_symbol in portfolio.positions.keys() {
            if let Some(corr) = self.get_correlation(new_symbol, existing_symbol)
                && corr.abs() > max_correlation
            {
                return Err(RiskError::CalculationError {
                    reason: format!(
                        "High correlation ({:.2}) between {} and {}",
                        corr, new_symbol, existing_symbol
                    ),
                });
            }
        }
        Ok(())
    }

    /// Create consistent key for symbol pairs
    fn create_key(symbol1: &str, symbol2: &str) -> (String, String) {
        if symbol1 <= symbol2 {
            (symbol1.to_string(), symbol2.to_string())
        } else {
            (symbol2.to_string(), symbol1.to_string())
        }
    }

    /// Load correlations from data (e.g., historical price correlations)
    pub fn load_correlations(&mut self, correlations: HashMap<(String, String), f64>) {
        for (key, value) in correlations {
            self.correlations.insert(key, value.clamp(-1.0, 1.0));
        }
    }
}

impl Default for CorrelationCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::{Position, PositionSide};
    use crate::signal::types::{SignalSource, SignalType, Timeframe};

    fn create_test_limits() -> RiskLimits {
        RiskLimits {
            max_position_value: 10000.0,
            max_portfolio_exposure: 50000.0,
            max_concurrent_positions: 5,
            max_daily_loss: 500.0,
            max_symbol_exposure: 20000.0,
            min_confidence: 0.6,
            min_strength: 0.4,
            check_correlation: false,
            max_correlation: 0.7,
        }
    }

    fn create_test_signal(symbol: &str, confidence: f64, strength: f64) -> TradingSignal {
        TradingSignal::new(
            symbol.to_string(),
            SignalType::Buy,
            Timeframe::H1,
            confidence,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        )
        .with_strength(strength)
        .with_entry_price(50000.0)
    }

    #[test]
    fn test_risk_limits_from_config() {
        let config = RiskConfig {
            account_balance: 10000.0,
            max_position_size_pct: 0.1,
            max_portfolio_exposure_pct: 0.5,
            max_concurrent_positions: 5,
            max_daily_loss_pct: 0.05,
            per_symbol_exposure_pct: 0.2,
            ..Default::default()
        };

        let limits = RiskLimits::from_config(&config);

        assert_eq!(limits.max_position_value, 1000.0);
        assert_eq!(limits.max_portfolio_exposure, 5000.0);
        assert_eq!(limits.max_concurrent_positions, 5);
        assert_eq!(limits.max_daily_loss, 500.0);
        assert_eq!(limits.max_symbol_exposure, 2000.0);
    }

    #[test]
    fn test_validate_signal_quality_pass() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let signal = create_test_signal("BTC/USD", 0.8, 0.7);

        let result = validator.validate_signal_quality(&signal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_signal_quality_low_confidence() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let signal = create_test_signal("BTC/USD", 0.3, 0.7); // Low confidence
        let result = validator.validate_signal_quality(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_signal_quality_low_strength() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let signal = create_test_signal("BTC/USD", 0.8, 0.2); // Low strength
        let result = validator.validate_signal_quality(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_position_count_pass() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let mut portfolio = PortfolioState::new(10000.0);

        // Add 3 positions (under limit of 5)
        for i in 0..3 {
            portfolio.add_position(
                format!("SYMBOL{}", i),
                Position {
                    symbol: format!("SYMBOL{}", i),
                    entry_price: 100.0,
                    quantity: 1.0,
                    side: PositionSide::Long,
                    stop_loss: Some(95.0),
                    take_profit: Some(110.0),
                },
            );
        }

        let result = validator.validate_position_count(&portfolio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_position_count_limit_reached() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let mut portfolio = PortfolioState::new(10000.0);

        // Add 5 positions (at limit)
        for i in 0..5 {
            portfolio.add_position(
                format!("SYMBOL{}", i),
                Position {
                    symbol: format!("SYMBOL{}", i),
                    entry_price: 100.0,
                    quantity: 1.0,
                    side: PositionSide::Long,
                    stop_loss: Some(95.0),
                    take_profit: Some(110.0),
                },
            );
        }

        let result = validator.validate_position_count(&portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_daily_loss_pass() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let mut portfolio = PortfolioState::new(10000.0);
        portfolio.daily_pnl = -200.0; // Lost $200 (under $500 limit)

        let result = validator.validate_daily_loss(&portfolio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_daily_loss_limit_reached() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let mut portfolio = PortfolioState::new(10000.0);
        portfolio.daily_pnl = -600.0; // Lost $600 (over $500 limit)

        let result = validator.validate_daily_loss(&portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_portfolio_exposure_pass() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let mut portfolio = PortfolioState::new(10000.0);

        // Add position with $2000 exposure (under $5000 limit)
        portfolio.add_position(
            "BTC/USD".to_string(),
            Position {
                symbol: "BTC/USD".to_string(),
                entry_price: 50000.0,
                quantity: 0.04,
                side: PositionSide::Long,
                stop_loss: Some(49000.0),
                take_profit: Some(52000.0),
            },
        );

        let signal = create_test_signal("ETH/USD", 0.8, 0.7);
        let result = validator.validate_portfolio_exposure(&signal, &portfolio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_symbol_exposure_pass() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        let mut portfolio = PortfolioState::new(10000.0);

        // Add small BTC position
        portfolio.add_position(
            "BTC/USD".to_string(),
            Position {
                symbol: "BTC/USD".to_string(),
                entry_price: 50000.0,
                quantity: 0.01,
                side: PositionSide::Long,
                stop_loss: Some(49000.0),
                take_profit: Some(52000.0),
            },
        );

        // Try to add another BTC signal
        let signal = create_test_signal("BTC/USD", 0.8, 0.7);
        let result = validator.validate_symbol_exposure(&signal, &portfolio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_position_size() {
        let limits = create_test_limits();
        let validator = RiskValidator::new(limits);

        // Within limit
        assert!(validator.validate_position_size(8000.0).is_ok());

        // Over limit
        assert!(validator.validate_position_size(15000.0).is_err());
    }

    #[test]
    fn test_correlation_calculator() {
        let mut calc = CorrelationCalculator::new();

        calc.add_correlation("BTC/USD".to_string(), "ETH/USD".to_string(), 0.85);

        let corr = calc.get_correlation("BTC/USD", "ETH/USD");
        assert_eq!(corr, Some(0.85));

        // Should work in reverse order too
        let corr_rev = calc.get_correlation("ETH/USD", "BTC/USD");
        assert_eq!(corr_rev, Some(0.85));
    }

    #[test]
    fn test_correlation_risk_check() {
        let mut calc = CorrelationCalculator::new();
        calc.add_correlation("BTC/USD".to_string(), "ETH/USD".to_string(), 0.9);

        let mut portfolio = PortfolioState::new(10000.0);
        portfolio.add_position(
            "BTC/USD".to_string(),
            Position {
                symbol: "BTC/USD".to_string(),
                entry_price: 50000.0,
                quantity: 0.1,
                side: PositionSide::Long,
                stop_loss: Some(49000.0),
                take_profit: Some(52000.0),
            },
        );

        // Should fail due to high correlation (0.9 > 0.7 max)
        let result = calc.check_correlation_risk("ETH/USD", &portfolio, 0.7);
        assert!(result.is_err());

        // Should pass with higher threshold
        let result = calc.check_correlation_risk("ETH/USD", &portfolio, 0.95);
        assert!(result.is_ok());
    }
}
