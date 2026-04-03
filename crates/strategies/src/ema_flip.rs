//! EMA Flip Trading Strategy
//!
//! Implementation of the 8/21 EMA Long-Short Flip Strategy with ATR-based stops.
//!
//! Strategy Logic:
//! - Long Entry: Price > 21 EMA AND pullback to 8 EMA (low touches)
//! - Short Entry: Price < 21 EMA AND rally to 8 EMA (high touches)
//! - Stop & Reverse: Crossover triggers immediate flip to opposite position
//! - ATR-based trailing stops instead of fixed percentage

use crate::indicators::IndicatorCalculator;
use crate::models::{Candle, Direction, Signal, SignalType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// EMA Flip Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmaFlipConfig {
    pub fast_ema_period: usize,
    pub slow_ema_period: usize,
    pub atr_period: usize,
    pub atr_multiplier: f64,
    pub min_confidence: f64,
}

impl Default for EmaFlipConfig {
    fn default() -> Self {
        Self {
            fast_ema_period: 8,
            slow_ema_period: 21,
            atr_period: 14,
            atr_multiplier: 1.5,
            min_confidence: 0.7,
        }
    }
}

/// EMA Flip Strategy
pub struct EmaFlipStrategy {
    config: EmaFlipConfig,
    calculator: IndicatorCalculator,
}

impl EmaFlipStrategy {
    /// Create a new EMA flip strategy
    pub fn new(config: EmaFlipConfig) -> Self {
        let calculator = IndicatorCalculator::new(
            config.fast_ema_period,
            config.slow_ema_period,
            config.atr_period,
        );

        Self { config, calculator }
    }
}

impl Default for EmaFlipStrategy {
    fn default() -> Self {
        Self::new(EmaFlipConfig::default())
    }
}

impl EmaFlipStrategy {
    /// Analyze market data and generate signals
    pub fn analyze(
        &self,
        symbol: &str,
        timeframe: &str,
        candles: &[Candle],
    ) -> Result<Option<Signal>, Box<dyn std::error::Error>> {
        if candles.len() < self.config.slow_ema_period + 10 {
            return Ok(None);
        }

        // Extract price data
        let close: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let high: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let low: Vec<f64> = candles.iter().map(|c| c.low).collect();

        // Calculate indicators
        let indicators = self.calculator.calculate_all(&close, &high, &low)?;

        // Get current values (last candle)
        let idx = candles.len() - 1;
        let current_candle = &candles[idx];
        let current_close = current_candle.close;
        let current_high = current_candle.high;
        let current_low = current_candle.low;

        let fast_ema = indicators.ema_fast[idx];
        let slow_ema = indicators.ema_slow[idx];
        let atr = indicators.atr[idx];

        // Skip if indicators not ready
        if fast_ema.is_nan() || slow_ema.is_nan() || atr.is_nan() {
            return Ok(None);
        }

        // Determine trend direction
        let trend = if current_close > slow_ema {
            Direction::Long
        } else {
            Direction::Short
        };

        // Check for entry opportunity
        let opportunity = self.check_entry_opportunity(
            current_close,
            current_high,
            current_low,
            fast_ema,
            slow_ema,
            trend,
        );

        if let Some((direction, confidence, reason)) = opportunity {
            // Calculate entry levels
            let entry_price = current_close;
            let stop_loss = self.calculate_stop_loss(entry_price, atr, direction);
            let take_profits = self.calculate_take_profits(entry_price, stop_loss, direction);

            let risk_reward =
                ((take_profits[0] - entry_price).abs() / (entry_price - stop_loss).abs()).abs();

            let mut signal = Signal::new(
                SignalType::Opportunity,
                symbol.to_string(),
                timeframe.to_string(),
                direction,
                entry_price,
                confidence,
            );

            signal.ema_fast = Some(fast_ema);
            signal.ema_slow = Some(slow_ema);
            signal.atr = Some(atr);
            signal.suggested_entry = Some(entry_price);
            signal.suggested_stop_loss = Some(stop_loss);
            signal.suggested_tp1 = Some(take_profits[0]);
            signal.suggested_tp2 = Some(take_profits[1]);
            signal.suggested_tp3 = Some(take_profits[2]);
            signal.suggested_tp4 = Some(take_profits[3]);
            signal.risk_reward_ratio = Some(risk_reward);
            signal.reason = Some(reason);

            return Ok(Some(signal));
        }

        Ok(None)
    }

    /// Check for entry opportunity
    fn check_entry_opportunity(
        &self,
        close: f64,
        high: f64,
        low: f64,
        fast_ema: f64,
        slow_ema: f64,
        trend: Direction,
    ) -> Option<(Direction, f64, String)> {
        match trend {
            Direction::Long => {
                // Long setup: Price above slow EMA, pullback to fast EMA
                if close > slow_ema && low <= fast_ema && close > fast_ema {
                    let distance_from_slow = ((close - slow_ema) / slow_ema) * 100.0;
                    let pullback_depth = ((high - low) / close) * 100.0;

                    // Calculate confidence based on setup quality
                    let mut confidence: f64 = 0.7;

                    // Better confidence if price bounced strongly from fast EMA
                    if close > fast_ema * 1.002 {
                        confidence += 0.1;
                    }

                    // Better confidence if not too far from slow EMA
                    if distance_from_slow < 2.0 {
                        confidence += 0.1;
                    }

                    // Better confidence with good pullback depth
                    if pullback_depth > 0.5 && pullback_depth < 3.0 {
                        confidence += 0.1;
                    }

                    let reason = format!(
                        "Long opportunity: Pullback to 8 EMA in uptrend. Distance from 21 EMA: {:.2}%, Pullback depth: {:.2}%",
                        distance_from_slow, pullback_depth
                    );

                    return Some((Direction::Long, confidence.min(1.0), reason));
                }
            }
            Direction::Short => {
                // Short setup: Price below slow EMA, rally to fast EMA
                if close < slow_ema && high >= fast_ema && close < fast_ema {
                    let distance_from_slow = ((slow_ema - close) / slow_ema) * 100.0;
                    let rally_depth = ((high - low) / close) * 100.0;

                    let mut confidence: f64 = 0.7;

                    if close < fast_ema * 0.998 {
                        confidence += 0.1;
                    }

                    if distance_from_slow < 2.0 {
                        confidence += 0.1;
                    }

                    if rally_depth > 0.5 && rally_depth < 3.0 {
                        confidence += 0.1;
                    }

                    let reason = format!(
                        "Short opportunity: Rally to 8 EMA in downtrend. Distance from 21 EMA: {:.2}%, Rally depth: {:.2}%",
                        distance_from_slow, rally_depth
                    );

                    return Some((Direction::Short, confidence.min(1.0), reason));
                }
            }
        }

        None
    }

    /// Calculate stop loss using ATR
    fn calculate_stop_loss(&self, entry_price: f64, atr: f64, direction: Direction) -> f64 {
        let stop_distance = atr * self.config.atr_multiplier;

        match direction {
            Direction::Long => entry_price - stop_distance,
            Direction::Short => entry_price + stop_distance,
        }
    }

    /// Calculate take profit levels
    fn calculate_take_profits(
        &self,
        entry_price: f64,
        stop_loss: f64,
        direction: Direction,
    ) -> Vec<f64> {
        let risk = (entry_price - stop_loss).abs();
        let rr_ratios = [1.0, 2.0, 3.0, 5.0];

        rr_ratios
            .iter()
            .map(|rr| match direction {
                Direction::Long => entry_price + (risk * rr),
                Direction::Short => entry_price - (risk * rr),
            })
            .collect()
    }

    /// Check if EMA crossover occurred (for stop and reverse)
    pub fn check_ema_crossover(
        &self,
        candles: &[Candle],
    ) -> Result<Option<EmaCrossover>, Box<dyn std::error::Error>> {
        if candles.len() < self.config.slow_ema_period + 2 {
            return Ok(None);
        }

        let close: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let high: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let low: Vec<f64> = candles.iter().map(|c| c.low).collect();

        let indicators = self.calculator.calculate_all(&close, &high, &low)?;

        let idx = candles.len() - 1;
        let prev_idx = idx - 1;

        let fast_now = indicators.ema_fast[idx];
        let slow_now = indicators.ema_slow[idx];
        let fast_prev = indicators.ema_fast[prev_idx];
        let slow_prev = indicators.ema_slow[prev_idx];

        // Bullish crossover: fast crosses above slow
        if fast_prev <= slow_prev && fast_now > slow_now {
            return Ok(Some(EmaCrossover {
                direction: Direction::Long,
                price: candles[idx].close,
                timestamp: candles[idx].timestamp,
            }));
        }

        // Bearish crossover: fast crosses below slow
        if fast_prev >= slow_prev && fast_now < slow_now {
            return Ok(Some(EmaCrossover {
                direction: Direction::Short,
                price: candles[idx].close,
                timestamp: candles[idx].timestamp,
            }));
        }

        Ok(None)
    }

    /// Update trailing stop
    pub fn update_trailing_stop(
        &self,
        entry_price: f64,
        current_price: f64,
        current_stop: f64,
        atr: f64,
        direction: Direction,
    ) -> f64 {
        let trail_distance = atr * self.config.atr_multiplier;

        match direction {
            Direction::Long => {
                let new_stop = current_price - trail_distance;
                // Only move stop up, never down
                if new_stop > current_stop && new_stop > entry_price {
                    new_stop
                } else {
                    current_stop
                }
            }
            Direction::Short => {
                let new_stop = current_price + trail_distance;
                // Only move stop down, never up
                if new_stop < current_stop && new_stop < entry_price {
                    new_stop
                } else {
                    current_stop
                }
            }
        }
    }
}

/// EMA crossover event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmaCrossover {
    pub direction: Direction,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn create_test_candles(count: usize, start_price: f64) -> Vec<Candle> {
        (0..count)
            .map(|i| Candle {
                timestamp: Utc::now(),
                open: start_price + i as f64,
                high: start_price + i as f64 + 10.0,
                low: start_price + i as f64 - 10.0,
                close: start_price + i as f64,
                volume: 1000.0,
            })
            .collect()
    }

    #[test]
    fn test_strategy_creation() {
        let strategy = EmaFlipStrategy::default();
        assert_eq!(strategy.config.fast_ema_period, 8);
        assert_eq!(strategy.config.slow_ema_period, 21);
    }

    #[test]
    fn test_stop_loss_calculation() {
        let strategy = EmaFlipStrategy::default();

        let stop_long = strategy.calculate_stop_loss(50000.0, 1000.0, Direction::Long);
        assert_eq!(stop_long, 50000.0 - 1500.0); // ATR * 1.5

        let stop_short = strategy.calculate_stop_loss(50000.0, 1000.0, Direction::Short);
        assert_eq!(stop_short, 50000.0 + 1500.0);
    }

    #[test]
    fn test_take_profit_calculation() {
        let strategy = EmaFlipStrategy::default();

        let tps = strategy.calculate_take_profits(50000.0, 48500.0, Direction::Long);
        assert_eq!(tps.len(), 4);
        assert_eq!(tps[0], 51500.0); // 1R
        assert_eq!(tps[1], 53000.0); // 2R
        assert_eq!(tps[2], 54500.0); // 3R
        assert_eq!(tps[3], 57500.0); // 5R
    }

    #[test]
    fn test_trailing_stop_update() {
        let strategy = EmaFlipStrategy::default();

        // Long position: stop should move up
        // With ATR=1000 and multiplier=1.5, trail_distance=1500.
        // new_stop = 52100 - 1500 = 50600 > current_stop(50500) AND > entry(50000) → moves up.
        let new_stop =
            strategy.update_trailing_stop(50000.0, 52100.0, 50500.0, 1000.0, Direction::Long);
        assert!(new_stop > 50500.0); // Stop moved up

        // Long position: stop shouldn't move down
        let new_stop =
            strategy.update_trailing_stop(50000.0, 49000.0, 50500.0, 1000.0, Direction::Long);
        assert_eq!(new_stop, 50500.0); // Stop stayed same
    }
}
