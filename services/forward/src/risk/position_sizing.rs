//! # Position Sizing Module
//!
//! Position sizing algorithms for optimal trade sizing based on risk parameters.
//! Supports multiple sizing methods:
//! - Fixed Fractional: Risk a fixed percentage of account per trade
//! - Kelly Criterion: Optimal sizing based on win rate and avg win/loss
//! - Volatility-Based: Adjust size based on market volatility
//! - Fixed Dollar: Risk a fixed dollar amount per trade

use crate::risk::{MarketData, RiskConfig, RiskError};
use crate::signal::types::TradingSignal;
use serde::{Deserialize, Serialize};

/// Position sizing method
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum SizingMethod {
    /// Risk a fixed percentage of account equity per trade
    #[default]
    FixedFractional,

    /// Kelly Criterion optimal sizing
    Kelly {
        /// Win rate (0.0 to 1.0)
        win_rate: f64,
        /// Average win / average loss ratio
        avg_win_loss_ratio: f64,
    },

    /// Volatility-based sizing (smaller positions in high volatility)
    VolatilityBased {
        /// Target volatility (annualized)
        target_volatility: f64,
    },

    /// Fixed dollar amount risk per trade
    FixedDollar {
        /// Dollar amount to risk
        amount: f64,
    },

    /// ATR-based sizing (normalize by Average True Range)
    AtrBased {
        /// Target ATR multiple for sizing
        target_atr_multiple: f64,
    },
}

/// Calculated position size result
#[derive(Debug, Clone)]
pub struct PositionSize {
    /// Number of units/contracts to trade
    pub quantity: f64,

    /// Total position value (quantity * price)
    pub position_value: f64,

    /// Dollar amount at risk
    pub risk_amount: f64,

    /// Risk as percentage of account
    pub risk_percentage: f64,

    /// Sizing method used
    pub method: SizingMethod,

    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl PositionSize {
    /// Create a new position size
    pub fn new(
        quantity: f64,
        price: f64,
        risk_amount: f64,
        account_balance: f64,
        method: SizingMethod,
    ) -> Self {
        Self {
            quantity,
            position_value: quantity * price,
            risk_amount,
            risk_percentage: risk_amount / account_balance,
            method,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Check if position size is valid
    pub fn is_valid(&self) -> bool {
        self.quantity > 0.0 && self.risk_amount >= 0.0 && self.risk_percentage >= 0.0
    }
}

/// Position sizer that calculates optimal position sizes
pub struct PositionSizer {
    config: RiskConfig,
}

impl PositionSizer {
    pub fn new(config: RiskConfig) -> Self {
        Self { config }
    }

    /// Calculate position size for a signal
    pub fn calculate_position_size(
        &self,
        signal: &TradingSignal,
        market_data: &MarketData,
        method: &SizingMethod,
    ) -> Result<PositionSize, RiskError> {
        // Validate we have entry price
        let entry_price = signal.entry_price.ok_or_else(|| RiskError::MissingData {
            field: "entry_price".to_string(),
        })?;

        // Validate we have stop loss for risk calculation
        let stop_loss = signal.stop_loss.ok_or_else(|| RiskError::MissingData {
            field: "stop_loss".to_string(),
        })?;

        // Calculate risk per unit
        let risk_per_unit = (entry_price - stop_loss).abs();
        if risk_per_unit <= 0.0 {
            return Err(RiskError::InvalidStopLoss {
                reason: "Stop loss must be different from entry price".to_string(),
            });
        }

        match method {
            SizingMethod::FixedFractional => {
                self.calculate_fixed_fractional(entry_price, risk_per_unit)
            }
            SizingMethod::Kelly {
                win_rate,
                avg_win_loss_ratio,
            } => self.calculate_kelly(entry_price, risk_per_unit, *win_rate, *avg_win_loss_ratio),
            SizingMethod::VolatilityBased { target_volatility } => self.calculate_volatility_based(
                entry_price,
                risk_per_unit,
                market_data,
                *target_volatility,
            ),
            SizingMethod::FixedDollar { amount } => {
                self.calculate_fixed_dollar(entry_price, risk_per_unit, *amount)
            }
            SizingMethod::AtrBased {
                target_atr_multiple,
            } => self.calculate_atr_based(
                entry_price,
                risk_per_unit,
                market_data,
                *target_atr_multiple,
            ),
        }
    }

    /// Fixed fractional sizing: risk fixed % of account
    fn calculate_fixed_fractional(
        &self,
        entry_price: f64,
        risk_per_unit: f64,
    ) -> Result<PositionSize, RiskError> {
        let risk_amount = self.config.account_balance * self.config.risk_per_trade_pct;
        let quantity = risk_amount / risk_per_unit;

        // Apply max position size limit
        let max_position_value = self.config.account_balance * self.config.max_position_size_pct;
        let position_value = quantity * entry_price;

        let (final_quantity, final_risk) = if position_value > max_position_value {
            let limited_quantity = max_position_value / entry_price;
            let limited_risk = limited_quantity * risk_per_unit;
            (limited_quantity, limited_risk)
        } else {
            (quantity, risk_amount)
        };

        Ok(PositionSize::new(
            final_quantity,
            entry_price,
            final_risk,
            self.config.account_balance,
            SizingMethod::FixedFractional,
        ))
    }

    /// Kelly Criterion sizing: optimal growth rate
    fn calculate_kelly(
        &self,
        entry_price: f64,
        risk_per_unit: f64,
        win_rate: f64,
        avg_win_loss_ratio: f64,
    ) -> Result<PositionSize, RiskError> {
        // Validate inputs
        if win_rate <= 0.0 || win_rate >= 1.0 {
            return Err(RiskError::InvalidConfiguration {
                reason: "Win rate must be between 0 and 1".to_string(),
            });
        }

        if avg_win_loss_ratio <= 0.0 {
            return Err(RiskError::InvalidConfiguration {
                reason: "Average win/loss ratio must be positive".to_string(),
            });
        }

        // Kelly formula: f = (p * b - q) / b
        // where p = win rate, q = loss rate, b = avg win/loss ratio
        let loss_rate = 1.0 - win_rate;
        let kelly_fraction = (win_rate * avg_win_loss_ratio - loss_rate) / avg_win_loss_ratio;

        // Use half-Kelly for safety (less aggressive)
        let safe_kelly = (kelly_fraction * 0.5).max(0.0);

        // Cap at configured max risk
        let final_fraction = safe_kelly.min(self.config.risk_per_trade_pct * 2.0);

        let risk_amount = self.config.account_balance * final_fraction;
        let quantity = risk_amount / risk_per_unit;

        // Apply max position size limit
        let max_position_value = self.config.account_balance * self.config.max_position_size_pct;
        let position_value = quantity * entry_price;

        let (final_quantity, final_risk) = if position_value > max_position_value {
            let limited_quantity = max_position_value / entry_price;
            let limited_risk = limited_quantity * risk_per_unit;
            (limited_quantity, limited_risk)
        } else {
            (quantity, risk_amount)
        };

        let mut pos = PositionSize::new(
            final_quantity,
            entry_price,
            final_risk,
            self.config.account_balance,
            SizingMethod::Kelly {
                win_rate,
                avg_win_loss_ratio,
            },
        );

        pos = pos.with_metadata(
            "kelly_fraction".to_string(),
            format!("{:.4}", kelly_fraction),
        );
        pos = pos.with_metadata("safe_kelly".to_string(), format!("{:.4}", safe_kelly));

        Ok(pos)
    }

    /// Volatility-based sizing: reduce size in high volatility
    fn calculate_volatility_based(
        &self,
        entry_price: f64,
        risk_per_unit: f64,
        market_data: &MarketData,
        target_volatility: f64,
    ) -> Result<PositionSize, RiskError> {
        let volatility = market_data
            .volatility
            .ok_or_else(|| RiskError::MissingData {
                field: "volatility".to_string(),
            })?;

        if volatility <= 0.0 {
            return Err(RiskError::CalculationError {
                reason: "Volatility must be positive".to_string(),
            });
        }

        // Adjust risk based on volatility ratio
        // If current volatility > target, reduce position size
        let volatility_ratio = target_volatility / volatility;
        let adjusted_risk_pct = self.config.risk_per_trade_pct * volatility_ratio.min(2.0);

        let risk_amount = self.config.account_balance * adjusted_risk_pct;
        let quantity = risk_amount / risk_per_unit;

        // Apply max position size limit
        let max_position_value = self.config.account_balance * self.config.max_position_size_pct;
        let position_value = quantity * entry_price;

        let (final_quantity, final_risk) = if position_value > max_position_value {
            let limited_quantity = max_position_value / entry_price;
            let limited_risk = limited_quantity * risk_per_unit;
            (limited_quantity, limited_risk)
        } else {
            (quantity, risk_amount)
        };

        let mut pos = PositionSize::new(
            final_quantity,
            entry_price,
            final_risk,
            self.config.account_balance,
            SizingMethod::VolatilityBased { target_volatility },
        );

        pos = pos.with_metadata("volatility".to_string(), format!("{:.6}", volatility));
        pos = pos.with_metadata(
            "volatility_ratio".to_string(),
            format!("{:.4}", volatility_ratio),
        );

        Ok(pos)
    }

    /// Fixed dollar sizing: risk fixed dollar amount
    fn calculate_fixed_dollar(
        &self,
        entry_price: f64,
        risk_per_unit: f64,
        risk_amount: f64,
    ) -> Result<PositionSize, RiskError> {
        if risk_amount <= 0.0 {
            return Err(RiskError::InvalidConfiguration {
                reason: "Risk amount must be positive".to_string(),
            });
        }

        let quantity = risk_amount / risk_per_unit;

        // Apply max position size limit
        let max_position_value = self.config.account_balance * self.config.max_position_size_pct;
        let position_value = quantity * entry_price;

        let (final_quantity, final_risk) = if position_value > max_position_value {
            let limited_quantity = max_position_value / entry_price;
            let limited_risk = limited_quantity * risk_per_unit;
            (limited_quantity, limited_risk)
        } else {
            (quantity, risk_amount)
        };

        Ok(PositionSize::new(
            final_quantity,
            entry_price,
            final_risk,
            self.config.account_balance,
            SizingMethod::FixedDollar {
                amount: risk_amount,
            },
        ))
    }

    /// ATR-based sizing: normalize by Average True Range
    fn calculate_atr_based(
        &self,
        entry_price: f64,
        risk_per_unit: f64,
        market_data: &MarketData,
        target_atr_multiple: f64,
    ) -> Result<PositionSize, RiskError> {
        let atr = market_data.atr.ok_or_else(|| RiskError::MissingData {
            field: "atr".to_string(),
        })?;

        if atr <= 0.0 {
            return Err(RiskError::CalculationError {
                reason: "ATR must be positive".to_string(),
            });
        }

        // Adjust risk based on ATR
        // Idea: if stop is tight relative to ATR, we can size larger
        let atr_ratio = (target_atr_multiple * atr) / risk_per_unit;
        let adjusted_risk_pct = self.config.risk_per_trade_pct * atr_ratio.min(2.0);

        let risk_amount = self.config.account_balance * adjusted_risk_pct;
        let quantity = risk_amount / risk_per_unit;

        // Apply max position size limit
        let max_position_value = self.config.account_balance * self.config.max_position_size_pct;
        let position_value = quantity * entry_price;

        let (final_quantity, final_risk) = if position_value > max_position_value {
            let limited_quantity = max_position_value / entry_price;
            let limited_risk = limited_quantity * risk_per_unit;
            (limited_quantity, limited_risk)
        } else {
            (quantity, risk_amount)
        };

        let mut pos = PositionSize::new(
            final_quantity,
            entry_price,
            final_risk,
            self.config.account_balance,
            SizingMethod::AtrBased {
                target_atr_multiple,
            },
        );

        pos = pos.with_metadata("atr".to_string(), format!("{:.2}", atr));
        pos = pos.with_metadata("atr_ratio".to_string(), format!("{:.4}", atr_ratio));

        Ok(pos)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: RiskConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::types::{SignalSource, SignalType, Timeframe};

    fn create_test_config() -> RiskConfig {
        RiskConfig {
            account_balance: 10000.0,
            risk_per_trade_pct: 0.01,   // 1%
            max_position_size_pct: 1.0, // 100% for testing (no limit)
            ..Default::default()
        }
    }

    fn create_test_signal(entry: f64, stop: f64) -> TradingSignal {
        TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Buy,
            Timeframe::H1,
            0.8,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        )
        .with_entry_price(entry)
        .with_stop_loss(stop)
    }

    #[test]
    fn test_fixed_fractional_sizing() {
        let config = create_test_config();
        let sizer = PositionSizer::new(config);

        let signal = create_test_signal(50000.0, 49000.0);
        let market_data = MarketData::new(50000.0);

        let result =
            sizer.calculate_position_size(&signal, &market_data, &SizingMethod::FixedFractional);

        assert!(result.is_ok());
        let pos = result.unwrap();

        // Risk: $100 (1% of $10,000)
        // Risk per unit: $1,000 (50000 - 49000)
        // Quantity: 0.1 BTC
        assert!((pos.risk_amount - 100.0).abs() < 0.01);
        assert!((pos.quantity - 0.1).abs() < 0.001);
        assert!(pos.is_valid());
    }

    #[test]
    fn test_fixed_fractional_with_limit() {
        // Use config with actual 10% limit for this test
        let config = RiskConfig {
            account_balance: 10000.0,
            risk_per_trade_pct: 0.01,   // 1%
            max_position_size_pct: 0.1, // 10% limit
            ..Default::default()
        };
        let sizer = PositionSizer::new(config);

        // Very tight stop = large position, should be limited
        let signal = create_test_signal(50000.0, 49900.0);
        let market_data = MarketData::new(50000.0);

        let result =
            sizer.calculate_position_size(&signal, &market_data, &SizingMethod::FixedFractional);

        assert!(result.is_ok());
        let pos = result.unwrap();

        // Position should be limited to 10% of account = $1,000
        assert!(pos.position_value <= 1000.0 + 0.01);
    }

    #[test]
    fn test_kelly_sizing() {
        let config = create_test_config();
        let sizer = PositionSizer::new(config);

        let signal = create_test_signal(50000.0, 49000.0);
        let market_data = MarketData::new(50000.0);

        let method = SizingMethod::Kelly {
            win_rate: 0.6,
            avg_win_loss_ratio: 2.0,
        };

        let result = sizer.calculate_position_size(&signal, &market_data, &method);

        assert!(result.is_ok());
        let pos = result.unwrap();

        // Kelly fraction = (0.6 * 2.0 - 0.4) / 2.0 = 0.4
        // Half-Kelly = 0.2 (20%)
        // But capped at 2% (twice risk_per_trade_pct)
        assert!(pos.is_valid());
        assert!(pos.metadata.contains_key("kelly_fraction"));
    }

    #[test]
    fn test_volatility_based_sizing() {
        let config = create_test_config();
        let sizer = PositionSizer::new(config);

        let signal = create_test_signal(50000.0, 49000.0);
        let market_data = MarketData::new(50000.0).with_volatility(0.02); // 2% volatility

        let method = SizingMethod::VolatilityBased {
            target_volatility: 0.03, // 3% target
        };

        let result = sizer.calculate_position_size(&signal, &market_data, &method);

        assert!(result.is_ok());
        let pos = result.unwrap();

        // Lower current volatility vs target = larger position
        assert!(pos.is_valid());
        assert!(pos.risk_percentage > 0.01); // Should be > base 1%
    }

    #[test]
    fn test_fixed_dollar_sizing() {
        let config = create_test_config();
        let sizer = PositionSizer::new(config);

        let signal = create_test_signal(50000.0, 49000.0);
        let market_data = MarketData::new(50000.0);

        let method = SizingMethod::FixedDollar { amount: 200.0 };

        let result = sizer.calculate_position_size(&signal, &market_data, &method);

        assert!(result.is_ok());
        let pos = result.unwrap();

        // Risk $200, risk per unit $1000 -> 0.2 BTC
        assert!((pos.risk_amount - 200.0).abs() < 0.01);
        assert!((pos.quantity - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_atr_based_sizing() {
        let config = create_test_config();
        let sizer = PositionSizer::new(config);

        let signal = create_test_signal(50000.0, 49000.0);
        let market_data = MarketData::new(50000.0).with_atr(500.0);

        let method = SizingMethod::AtrBased {
            target_atr_multiple: 2.0,
        };

        let result = sizer.calculate_position_size(&signal, &market_data, &method);

        assert!(result.is_ok());
        let pos = result.unwrap();

        // Target: 2 * 500 = 1000 (same as our stop distance)
        // So ATR ratio = 1.0, no adjustment
        assert!(pos.is_valid());
        assert!(pos.metadata.contains_key("atr"));
    }

    #[test]
    fn test_missing_entry_price() {
        let config = create_test_config();
        let sizer = PositionSizer::new(config);

        let signal = TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Buy,
            Timeframe::H1,
            0.8,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        );

        let market_data = MarketData::new(50000.0);

        let result =
            sizer.calculate_position_size(&signal, &market_data, &SizingMethod::FixedFractional);

        assert!(result.is_err());
    }

    #[test]
    fn test_missing_stop_loss() {
        let config = create_test_config();
        let sizer = PositionSizer::new(config);

        let signal = TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Buy,
            Timeframe::H1,
            0.8,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        )
        .with_entry_price(50000.0);

        let market_data = MarketData::new(50000.0);

        let result =
            sizer.calculate_position_size(&signal, &market_data, &SizingMethod::FixedFractional);

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_kelly_parameters() {
        let config = create_test_config();
        let sizer = PositionSizer::new(config);

        let signal = create_test_signal(50000.0, 49000.0);
        let market_data = MarketData::new(50000.0);

        // Invalid win rate
        let method = SizingMethod::Kelly {
            win_rate: 1.5,
            avg_win_loss_ratio: 2.0,
        };

        let result = sizer.calculate_position_size(&signal, &market_data, &method);
        assert!(result.is_err());
    }
}
