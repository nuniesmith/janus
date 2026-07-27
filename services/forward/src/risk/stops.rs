//! # Stop Loss and Take Profit Calculation Module
//!
//! Calculates optimal stop loss and take profit levels for trading signals.
//! Supports multiple methods:
//! - ATR-based: Use Average True Range for dynamic stops
//! - Percentage: Fixed percentage from entry
//! - Support/Resistance: Use technical levels
//! - Volatility-based: Adjust based on market volatility
//! - Trailing: Dynamic stops that follow price

use crate::risk::{MarketData, RiskConfig, RiskError};
use crate::signal::types::TradingSignal;
use serde::{Deserialize, Serialize};

/// Minimum stop distance as a fraction of entry price (0.1%).
///
/// A stop nearer than this is indistinguishable from noise, so the floor is
/// sound — but it must CLAMP, not reject: see `clamp_stop_to_min_distance`.
/// Rejecting on it discarded ~30% of all directional decisions (59.8% of all
/// gate blocks) because ATR-derived stops off 1-minute candles routinely land
/// inside it. Named rather than inlined so the guard and the clamp can never
/// drift apart.
const MIN_STOP_DISTANCE_PCT: f64 = 0.001;

/// Stop loss calculation method
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopLossMethod {
    /// ATR-based stop (e.g., 2x ATR from entry)
    Atr {
        /// Multiplier for ATR (e.g., 2.0 = 2x ATR)
        multiplier: f64,
    },

    /// Fixed percentage from entry
    Percentage {
        /// Stop distance as percentage (e.g., 0.02 = 2%)
        percent: f64,
    },

    /// Use support/resistance levels
    SupportResistance,

    /// Volatility-based (use recent volatility)
    Volatility {
        /// Standard deviations from entry
        std_devs: f64,
    },

    /// Recent high/low based
    HighLow {
        /// Lookback periods
        lookback: usize,
        /// Buffer percentage below low/above high
        buffer_pct: f64,
    },
}

impl Default for StopLossMethod {
    fn default() -> Self {
        StopLossMethod::Atr { multiplier: 2.0 }
    }
}

/// Take profit calculation method
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TakeProfitMethod {
    /// Risk/Reward ratio based (e.g., 2:1)
    RiskReward {
        /// R/R ratio (2.0 = 2:1)
        ratio: f64,
    },

    /// ATR-based take profit
    Atr {
        /// Multiplier for ATR
        multiplier: f64,
    },

    /// Fixed percentage gain
    Percentage {
        /// Target gain percentage
        percent: f64,
    },

    /// Use resistance/support as target
    SupportResistance,

    /// Multiple targets (partial exits)
    MultiTarget {
        /// Array of (percentage of position, R/R ratio) pairs
        targets: Vec<(f64, f64)>,
    },
}

impl Default for TakeProfitMethod {
    fn default() -> Self {
        TakeProfitMethod::RiskReward { ratio: 2.0 }
    }
}

/// Stop loss calculator
pub struct StopLossCalculator {
    config: RiskConfig,
}

impl StopLossCalculator {
    pub fn new(config: RiskConfig) -> Self {
        Self { config }
    }

    /// Calculate stop loss for a signal
    pub fn calculate_stop_loss(
        &self,
        signal: &TradingSignal,
        market_data: &MarketData,
        method: &StopLossMethod,
    ) -> Result<f64, RiskError> {
        let entry_price = signal.entry_price.ok_or_else(|| RiskError::MissingData {
            field: "entry_price".to_string(),
        })?;

        let stop_loss = match method {
            StopLossMethod::Atr { multiplier } => {
                self.calculate_atr_stop(signal, entry_price, market_data, *multiplier)?
            }
            StopLossMethod::Percentage { percent } => {
                self.calculate_percentage_stop(signal, entry_price, *percent)
            }
            StopLossMethod::SupportResistance => {
                self.calculate_support_resistance_stop(signal, entry_price, market_data)?
            }
            StopLossMethod::Volatility { std_devs } => {
                self.calculate_volatility_stop(signal, entry_price, market_data, *std_devs)?
            }
            StopLossMethod::HighLow {
                lookback: _,
                buffer_pct,
            } => self.calculate_high_low_stop(signal, entry_price, market_data, *buffer_pct)?,
        };

        // CLAMP, don't reject (2026-07-27). A stop computed INSIDE the minimum
        // distance is a reason to widen the stop, not to discard the signal —
        // but `validate_stop_loss` rejected it, and that single rejection was
        // 59.8% of ALL gate blocks and ~30% of every directional decision
        // (~6,571 in the Gate-A window). The gate named "risk" was throwing
        // away one signal in three for a units artifact: stops are ATR-derived
        // and janus ingests 1m candles, so a calm 1-minute ATR is routinely
        // smaller than MIN_STOP_DISTANCE_PCT of price (for BTC at ~$100k the
        // floor is ~$100 — under a quiet 1m ATR). Observed rejecting BTC/USDT
        // and POL/USDT entries, and not merely advisory: the same rejection
        // surfaced as "risk ENFORCE: blocking live entry".
        //
        // Widening to the floor is the conservative direction: the resulting
        // stop is FARTHER from entry (more room, smaller position for a fixed
        // risk budget), never tighter. The too-far and wrong-side checks in
        // `validate_stop_loss` still apply and can still reject.
        let stop_loss = self.clamp_stop_to_min_distance(signal, entry_price, stop_loss);

        // Validate stop loss
        self.validate_stop_loss(signal, entry_price, stop_loss)?;

        Ok(stop_loss)
    }

    /// Push a too-close stop out to [`MIN_STOP_DISTANCE_PCT`], preserving side.
    ///
    /// Returns the stop unchanged when it is already at or beyond the floor,
    /// or when the inputs are degenerate (non-positive entry, or a stop on the
    /// wrong side — the latter is left for `validate_stop_loss` to reject
    /// rather than silently "fixed" into a valid-looking stop).
    fn clamp_stop_to_min_distance(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        stop_loss: f64,
    ) -> f64 {
        if entry_price <= 0.0 || stop_loss <= 0.0 {
            return stop_loss;
        }
        let bullish = signal.signal_type.is_bullish();
        let bearish = signal.signal_type.is_bearish();
        // Wrong-side stops are a different fault; don't mask them.
        if (bullish && stop_loss >= entry_price) || (bearish && stop_loss <= entry_price) {
            return stop_loss;
        }
        let distance_pct = ((entry_price - stop_loss).abs()) / entry_price;
        if distance_pct >= MIN_STOP_DISTANCE_PCT {
            return stop_loss;
        }
        let floor_distance = entry_price * MIN_STOP_DISTANCE_PCT;
        let widened = if bullish {
            entry_price - floor_distance
        } else if bearish {
            entry_price + floor_distance
        } else {
            return stop_loss;
        };
        tracing::debug!(
            entry_price,
            original_stop = stop_loss,
            widened_stop = widened,
            original_distance_pct = distance_pct,
            min_distance_pct = MIN_STOP_DISTANCE_PCT,
            "risk: stop inside the minimum distance — widened to the floor (was rejected before 2026-07-27)"
        );
        widened
    }

    /// ATR-based stop loss
    fn calculate_atr_stop(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        market_data: &MarketData,
        multiplier: f64,
    ) -> Result<f64, RiskError> {
        let atr = market_data.atr.ok_or_else(|| RiskError::MissingData {
            field: "atr".to_string(),
        })?;

        if atr <= 0.0 {
            return Err(RiskError::CalculationError {
                reason: "ATR must be positive".to_string(),
            });
        }

        let stop_distance = atr * multiplier;

        let stop_loss = if signal.signal_type.is_bullish() {
            entry_price - stop_distance
        } else {
            entry_price + stop_distance
        };

        Ok(stop_loss)
    }

    /// Percentage-based stop loss
    fn calculate_percentage_stop(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        percent: f64,
    ) -> f64 {
        let stop_distance = entry_price * percent;

        if signal.signal_type.is_bullish() {
            entry_price - stop_distance
        } else {
            entry_price + stop_distance
        }
    }

    /// Support/resistance-based stop loss
    fn calculate_support_resistance_stop(
        &self,
        signal: &TradingSignal,
        _entry_price: f64,
        market_data: &MarketData,
    ) -> Result<f64, RiskError> {
        if signal.signal_type.is_bullish() {
            // For long positions, stop below support
            let support = market_data.support.ok_or_else(|| RiskError::MissingData {
                field: "support".to_string(),
            })?;

            // Add small buffer below support (0.1%)
            let buffer = support * 0.001;
            Ok(support - buffer)
        } else {
            // For short positions, stop above resistance
            let resistance = market_data
                .resistance
                .ok_or_else(|| RiskError::MissingData {
                    field: "resistance".to_string(),
                })?;

            // Add small buffer above resistance
            let buffer = resistance * 0.001;
            Ok(resistance + buffer)
        }
    }

    /// Volatility-based stop loss
    fn calculate_volatility_stop(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        market_data: &MarketData,
        std_devs: f64,
    ) -> Result<f64, RiskError> {
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

        let stop_distance = entry_price * volatility * std_devs;

        let stop_loss = if signal.signal_type.is_bullish() {
            entry_price - stop_distance
        } else {
            entry_price + stop_distance
        };

        Ok(stop_loss)
    }

    /// High/Low-based stop loss
    fn calculate_high_low_stop(
        &self,
        signal: &TradingSignal,
        _entry_price: f64,
        market_data: &MarketData,
        buffer_pct: f64,
    ) -> Result<f64, RiskError> {
        if signal.signal_type.is_bullish() {
            let recent_low = market_data
                .recent_low
                .ok_or_else(|| RiskError::MissingData {
                    field: "recent_low".to_string(),
                })?;

            let buffer = recent_low * buffer_pct;
            Ok(recent_low - buffer)
        } else {
            let recent_high = market_data
                .recent_high
                .ok_or_else(|| RiskError::MissingData {
                    field: "recent_high".to_string(),
                })?;

            let buffer = recent_high * buffer_pct;
            Ok(recent_high + buffer)
        }
    }

    /// Validate stop loss
    fn validate_stop_loss(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        stop_loss: f64,
    ) -> Result<(), RiskError> {
        if stop_loss <= 0.0 {
            return Err(RiskError::InvalidStopLoss {
                reason: "Stop loss must be positive".to_string(),
            });
        }

        // For bullish signals, stop must be below entry
        if signal.signal_type.is_bullish() && stop_loss >= entry_price {
            return Err(RiskError::InvalidStopLoss {
                reason: "Stop loss must be below entry for long positions".to_string(),
            });
        }

        // For bearish signals, stop must be above entry
        if signal.signal_type.is_bearish() && stop_loss <= entry_price {
            return Err(RiskError::InvalidStopLoss {
                reason: "Stop loss must be above entry for short positions".to_string(),
            });
        }

        // Check if stop is too close. Reaching here now means the clamp above
        // could not fix it (degenerate/wrong-side input), so rejecting is
        // correct — the floor itself is sound, it was the INPUT that was wrong.
        let stop_distance_pct = ((entry_price - stop_loss).abs() / entry_price).abs();
        if stop_distance_pct < MIN_STOP_DISTANCE_PCT {
            return Err(RiskError::InvalidStopLoss {
                reason: "Stop loss too close to entry (< 0.1%)".to_string(),
            });
        }

        // Check if stop is too far (more than 50%)
        if stop_distance_pct > 0.5 {
            return Err(RiskError::InvalidStopLoss {
                reason: "Stop loss too far from entry (> 50%)".to_string(),
            });
        }

        Ok(())
    }

    pub fn update_config(&mut self, config: RiskConfig) {
        self.config = config;
    }
}

/// Take profit calculator
pub struct TakeProfitCalculator {
    config: RiskConfig,
}

impl TakeProfitCalculator {
    pub fn new(config: RiskConfig) -> Self {
        Self { config }
    }

    /// Calculate take profit for a signal
    pub fn calculate_take_profit(
        &self,
        signal: &TradingSignal,
        market_data: &MarketData,
    ) -> Result<f64, RiskError> {
        let entry_price = signal.entry_price.ok_or_else(|| RiskError::MissingData {
            field: "entry_price".to_string(),
        })?;

        let stop_loss = signal.stop_loss.ok_or_else(|| RiskError::MissingData {
            field: "stop_loss".to_string(),
        })?;

        // Default: use risk/reward ratio
        let method = TakeProfitMethod::RiskReward {
            ratio: self.config.default_risk_reward,
        };

        self.calculate_take_profit_with_method(signal, market_data, &method, entry_price, stop_loss)
    }

    /// Calculate take profit with specific method
    pub fn calculate_take_profit_with_method(
        &self,
        signal: &TradingSignal,
        market_data: &MarketData,
        method: &TakeProfitMethod,
        entry_price: f64,
        stop_loss: f64,
    ) -> Result<f64, RiskError> {
        let take_profit = match method {
            TakeProfitMethod::RiskReward { ratio } => {
                self.calculate_risk_reward_tp(signal, entry_price, stop_loss, *ratio)
            }
            TakeProfitMethod::Atr { multiplier } => {
                self.calculate_atr_tp(signal, entry_price, market_data, *multiplier)?
            }
            TakeProfitMethod::Percentage { percent } => {
                self.calculate_percentage_tp(signal, entry_price, *percent)
            }
            TakeProfitMethod::SupportResistance => {
                self.calculate_support_resistance_tp(signal, market_data)?
            }
            TakeProfitMethod::MultiTarget { targets } => {
                // For multi-target, return the first target
                if let Some((_, ratio)) = targets.first() {
                    self.calculate_risk_reward_tp(signal, entry_price, stop_loss, *ratio)
                } else {
                    self.calculate_risk_reward_tp(signal, entry_price, stop_loss, 2.0)
                }
            }
        };

        // Validate take profit
        self.validate_take_profit(signal, entry_price, take_profit)?;

        Ok(take_profit)
    }

    /// Risk/Reward ratio-based take profit
    fn calculate_risk_reward_tp(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        stop_loss: f64,
        ratio: f64,
    ) -> f64 {
        let risk = (entry_price - stop_loss).abs();
        let reward = risk * ratio;

        if signal.signal_type.is_bullish() {
            entry_price + reward
        } else {
            entry_price - reward
        }
    }

    /// ATR-based take profit
    fn calculate_atr_tp(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        market_data: &MarketData,
        multiplier: f64,
    ) -> Result<f64, RiskError> {
        let atr = market_data.atr.ok_or_else(|| RiskError::MissingData {
            field: "atr".to_string(),
        })?;

        if atr <= 0.0 {
            return Err(RiskError::CalculationError {
                reason: "ATR must be positive".to_string(),
            });
        }

        let tp_distance = atr * multiplier;

        let take_profit = if signal.signal_type.is_bullish() {
            entry_price + tp_distance
        } else {
            entry_price - tp_distance
        };

        Ok(take_profit)
    }

    /// Percentage-based take profit
    fn calculate_percentage_tp(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        percent: f64,
    ) -> f64 {
        let tp_distance = entry_price * percent;

        if signal.signal_type.is_bullish() {
            entry_price + tp_distance
        } else {
            entry_price - tp_distance
        }
    }

    /// Support/Resistance-based take profit
    fn calculate_support_resistance_tp(
        &self,
        signal: &TradingSignal,
        market_data: &MarketData,
    ) -> Result<f64, RiskError> {
        if signal.signal_type.is_bullish() {
            // For long positions, target is resistance
            let resistance = market_data
                .resistance
                .ok_or_else(|| RiskError::MissingData {
                    field: "resistance".to_string(),
                })?;

            // Small buffer before resistance
            let buffer = resistance * 0.001;
            Ok(resistance - buffer)
        } else {
            // For short positions, target is support
            let support = market_data.support.ok_or_else(|| RiskError::MissingData {
                field: "support".to_string(),
            })?;

            // Small buffer above support
            let buffer = support * 0.001;
            Ok(support + buffer)
        }
    }

    /// Validate take profit
    fn validate_take_profit(
        &self,
        signal: &TradingSignal,
        entry_price: f64,
        take_profit: f64,
    ) -> Result<(), RiskError> {
        if take_profit <= 0.0 {
            return Err(RiskError::InvalidTakeProfit {
                reason: "Take profit must be positive".to_string(),
            });
        }

        // For bullish signals, TP must be above entry
        if signal.signal_type.is_bullish() && take_profit <= entry_price {
            return Err(RiskError::InvalidTakeProfit {
                reason: "Take profit must be above entry for long positions".to_string(),
            });
        }

        // For bearish signals, TP must be below entry
        if signal.signal_type.is_bearish() && take_profit >= entry_price {
            return Err(RiskError::InvalidTakeProfit {
                reason: "Take profit must be below entry for short positions".to_string(),
            });
        }

        Ok(())
    }

    pub fn update_config(&mut self, config: RiskConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::types::{SignalSource, SignalType, Timeframe};

    fn create_test_config() -> RiskConfig {
        RiskConfig::default()
    }

    fn create_buy_signal(entry: f64) -> TradingSignal {
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
    }

    fn create_sell_signal(entry: f64) -> TradingSignal {
        TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Sell,
            Timeframe::H1,
            0.8,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        )
        .with_entry_price(entry)
    }

    #[test]
    fn test_atr_stop_loss_buy() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);
        let market_data = MarketData::new(50000.0).with_atr(500.0);

        let method = StopLossMethod::Atr { multiplier: 2.0 };
        let result = calculator.calculate_stop_loss(&signal, &market_data, &method);

        assert!(result.is_ok());
        let stop = result.unwrap();

        // Stop should be 2 * ATR below entry
        // 50000 - (2 * 500) = 49000
        assert!((stop - 49000.0).abs() < 0.01);
    }

    #[test]
    fn test_atr_stop_loss_sell() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_sell_signal(50000.0);
        let market_data = MarketData::new(50000.0).with_atr(500.0);

        let method = StopLossMethod::Atr { multiplier: 2.0 };
        let result = calculator.calculate_stop_loss(&signal, &market_data, &method);

        assert!(result.is_ok());
        let stop = result.unwrap();

        // Stop should be 2 * ATR above entry
        // 50000 + (2 * 500) = 51000
        assert!((stop - 51000.0).abs() < 0.01);
    }

    #[test]
    fn test_percentage_stop_loss() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);
        let market_data = MarketData::new(50000.0);

        let method = StopLossMethod::Percentage { percent: 0.02 }; // 2%
        let result = calculator.calculate_stop_loss(&signal, &market_data, &method);

        assert!(result.is_ok());
        let stop = result.unwrap();

        // 50000 - (50000 * 0.02) = 49000
        assert!((stop - 49000.0).abs() < 0.01);
    }

    #[test]
    fn test_support_resistance_stop() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);
        let market_data = MarketData::new(50000.0).with_support(49500.0);

        let method = StopLossMethod::SupportResistance;
        let result = calculator.calculate_stop_loss(&signal, &market_data, &method);

        assert!(result.is_ok());
        let stop = result.unwrap();

        // Should be slightly below support
        assert!(stop < 49500.0);
        assert!(stop > 49400.0);
    }

    #[test]
    fn test_volatility_stop_loss() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);
        let market_data = MarketData::new(50000.0).with_volatility(0.02); // 2% volatility

        let method = StopLossMethod::Volatility { std_devs: 2.0 };
        let result = calculator.calculate_stop_loss(&signal, &market_data, &method);

        assert!(result.is_ok());
        let stop = result.unwrap();

        // 50000 - (50000 * 0.02 * 2.0) = 48000
        assert!((stop - 48000.0).abs() < 0.01);
    }

    #[test]
    fn test_high_low_stop() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);
        let market_data = MarketData::new(50000.0).with_range(51000.0, 49000.0);

        let method = StopLossMethod::HighLow {
            lookback: 20,
            buffer_pct: 0.001,
        };
        let result = calculator.calculate_stop_loss(&signal, &market_data, &method);

        assert!(result.is_ok());
        let stop = result.unwrap();

        // Should be below recent low with buffer
        assert!(stop < 49000.0);
    }

    #[test]
    fn test_risk_reward_take_profit() {
        let config = create_test_config();
        let calculator = TakeProfitCalculator::new(config);

        let signal = create_buy_signal(50000.0).with_stop_loss(49000.0);
        let market_data = MarketData::new(50000.0);

        let result = calculator.calculate_take_profit(&signal, &market_data);

        assert!(result.is_ok());
        let tp = result.unwrap();

        // Risk: 1000, Default R/R: 2.0, Reward: 2000
        // TP: 50000 + 2000 = 52000
        assert!((tp - 52000.0).abs() < 0.01);
    }

    #[test]
    fn test_atr_take_profit() {
        let config = create_test_config();
        let calculator = TakeProfitCalculator::new(config);

        let signal = create_buy_signal(50000.0).with_stop_loss(49000.0);
        let market_data = MarketData::new(50000.0).with_atr(500.0);

        let method = TakeProfitMethod::Atr { multiplier: 3.0 };
        let result = calculator.calculate_take_profit_with_method(
            &signal,
            &market_data,
            &method,
            50000.0,
            49000.0,
        );

        assert!(result.is_ok());
        let tp = result.unwrap();

        // 50000 + (3 * 500) = 51500
        assert!((tp - 51500.0).abs() < 0.01);
    }

    #[test]
    fn test_percentage_take_profit() {
        let config = create_test_config();
        let calculator = TakeProfitCalculator::new(config);

        let signal = create_buy_signal(50000.0).with_stop_loss(49000.0);
        let market_data = MarketData::new(50000.0);

        let method = TakeProfitMethod::Percentage { percent: 0.05 }; // 5%
        let result = calculator.calculate_take_profit_with_method(
            &signal,
            &market_data,
            &method,
            50000.0,
            49000.0,
        );

        assert!(result.is_ok());
        let tp = result.unwrap();

        // 50000 + (50000 * 0.05) = 52500
        assert!((tp - 52500.0).abs() < 0.01);
    }

    #[test]
    fn test_stop_too_close_is_widened_not_rejected() {
        // CONTRACT CHANGE 2026-07-27: a stop inside the minimum distance used
        // to be REJECTED, which discarded ~30% of all directional decisions
        // (59.8% of every gate block) because ATR-derived stops off 1-minute
        // candles routinely land inside the floor. It is now widened to the
        // floor — the signal survives, with a *wider* (more conservative) stop.
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);
        let market_data = MarketData::new(50000.0);

        // 0.01% — an order of magnitude inside the 0.1% floor.
        let method = StopLossMethod::Percentage { percent: 0.0001 };
        let stop = calculator
            .calculate_stop_loss(&signal, &market_data, &method)
            .expect("a too-close stop must be widened, not rejected");

        // Long: widened DOWNWARD to exactly the floor.
        let expected = 50000.0 * (1.0 - MIN_STOP_DISTANCE_PCT);
        assert!(
            (stop - expected).abs() < 1e-6,
            "expected widen to {expected}, got {stop}"
        );
        // The stop moved AWAY from entry, never toward it.
        let naive = 50000.0 * (1.0 - 0.0001);
        assert!(stop < naive, "clamp must widen, not tighten");
    }

    #[test]
    fn test_stop_too_close_short_widens_upward() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_sell_signal(50000.0);
        let market_data = MarketData::new(50000.0);

        let method = StopLossMethod::Percentage { percent: 0.0001 };
        let stop = calculator
            .calculate_stop_loss(&signal, &market_data, &method)
            .expect("short: too-close stop must be widened");

        let expected = 50000.0 * (1.0 + MIN_STOP_DISTANCE_PCT);
        assert!((stop - expected).abs() < 1e-6, "got {stop}");
        assert!(stop > 50000.0, "short stop must stay ABOVE entry");
    }

    #[test]
    fn test_adequate_stop_is_left_untouched() {
        // The clamp must be a no-op for stops already beyond the floor —
        // it widens only what would otherwise have been thrown away.
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);
        let market_data = MarketData::new(50000.0);

        let method = StopLossMethod::Percentage { percent: 0.02 }; // 2%, well clear
        let stop = calculator
            .calculate_stop_loss(&signal, &market_data, &method)
            .expect("an adequate stop stays valid");
        assert!((stop - 49000.0).abs() < 1e-6, "got {stop}");
    }

    #[test]
    fn test_invalid_stop_wrong_direction() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);

        // Manually test validation with stop above entry for buy
        let result = calculator.validate_stop_loss(&signal, 50000.0, 51000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_atr_data() {
        let config = create_test_config();
        let calculator = StopLossCalculator::new(config);

        let signal = create_buy_signal(50000.0);
        let market_data = MarketData::new(50000.0); // No ATR

        let method = StopLossMethod::Atr { multiplier: 2.0 };
        let result = calculator.calculate_stop_loss(&signal, &market_data, &method);

        assert!(result.is_err());
    }
}
