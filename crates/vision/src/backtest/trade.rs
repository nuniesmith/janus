//! Trade tracking and position management for backtesting.
//!
//! This module provides types for tracking individual trades, positions,
//! and their results during backtesting simulations.

use crate::signals::{SignalType, TradingSignal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Status of a trade
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeStatus {
    /// Trade is currently open
    Open,
    /// Trade was closed normally
    Closed,
    /// Trade was stopped out (stop-loss hit)
    StoppedOut,
    /// Trade hit take-profit target
    TakeProfit,
    /// Trade was closed due to signal reversal
    Reversed,
}

impl fmt::Display for TradeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TradeStatus::Open => write!(f, "OPEN"),
            TradeStatus::Closed => write!(f, "CLOSED"),
            TradeStatus::StoppedOut => write!(f, "STOPPED_OUT"),
            TradeStatus::TakeProfit => write!(f, "TAKE_PROFIT"),
            TradeStatus::Reversed => write!(f, "REVERSED"),
        }
    }
}

/// Current position in an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Asset symbol
    pub asset: String,

    /// Position side (Buy = Long, Sell = Short)
    pub side: SignalType,

    /// Entry price
    pub entry_price: f64,

    /// Position size (quantity or contracts)
    pub size: f64,

    /// Entry timestamp
    pub entry_time: DateTime<Utc>,

    /// Entry signal that opened this position
    pub entry_signal: TradingSignal,

    /// Current unrealized P&L
    pub unrealized_pnl: f64,

    /// Stop-loss price (if set)
    pub stop_loss: Option<f64>,

    /// Take-profit price (if set)
    pub take_profit: Option<f64>,
}

impl Position {
    /// Create a new position from a signal
    pub fn new(signal: &TradingSignal, entry_price: f64, size: f64) -> Self {
        Self {
            asset: signal.asset.clone(),
            side: signal.signal_type,
            entry_price,
            size,
            entry_time: signal.timestamp,
            entry_signal: signal.clone(),
            unrealized_pnl: 0.0,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Update unrealized P&L based on current price
    pub fn update_pnl(&mut self, current_price: f64) {
        let price_change = match self.side {
            SignalType::Buy => current_price - self.entry_price,
            SignalType::Sell => self.entry_price - current_price,
            _ => 0.0,
        };
        self.unrealized_pnl = price_change * self.size;
    }

    /// Get the current return as a percentage
    pub fn return_pct(&self) -> f64 {
        let position_value = self.entry_price * self.size;
        if position_value == 0.0 {
            return 0.0;
        }
        (self.unrealized_pnl / position_value) * 100.0
    }

    /// Set stop-loss at a percentage below entry
    pub fn set_stop_loss_pct(&mut self, pct: f64) {
        let stop_price = match self.side {
            SignalType::Buy => self.entry_price * (1.0 - pct),
            SignalType::Sell => self.entry_price * (1.0 + pct),
            _ => self.entry_price,
        };
        self.stop_loss = Some(stop_price);
    }

    /// Set take-profit at a percentage above entry
    pub fn set_take_profit_pct(&mut self, pct: f64) {
        let tp_price = match self.side {
            SignalType::Buy => self.entry_price * (1.0 + pct),
            SignalType::Sell => self.entry_price * (1.0 - pct),
            _ => self.entry_price,
        };
        self.take_profit = Some(tp_price);
    }

    /// Check if stop-loss would be hit at given price
    pub fn hits_stop_loss(&self, price: f64) -> bool {
        if let Some(stop) = self.stop_loss {
            match self.side {
                SignalType::Buy => price <= stop,
                SignalType::Sell => price >= stop,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if take-profit would be hit at given price
    pub fn hits_take_profit(&self, price: f64) -> bool {
        if let Some(tp) = self.take_profit {
            match self.side {
                SignalType::Buy => price >= tp,
                SignalType::Sell => price <= tp,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Calculate the value of the position at entry
    pub fn entry_value(&self) -> f64 {
        self.entry_price * self.size
    }

    /// Duration of the position in seconds
    pub fn duration_seconds(&self, current_time: DateTime<Utc>) -> i64 {
        (current_time - self.entry_time).num_seconds()
    }
}

/// Completed trade with full details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade ID
    pub id: usize,

    /// Asset symbol
    pub asset: String,

    /// Trade side (Buy = Long, Sell = Short)
    pub side: SignalType,

    /// Entry price
    pub entry_price: f64,

    /// Exit price
    pub exit_price: f64,

    /// Position size
    pub size: f64,

    /// Entry timestamp
    pub entry_time: DateTime<Utc>,

    /// Exit timestamp
    pub exit_time: DateTime<Utc>,

    /// Trade status at close
    pub status: TradeStatus,

    /// Gross P&L (before fees)
    pub gross_pnl: f64,

    /// Commission paid on entry
    pub entry_commission: f64,

    /// Commission paid on exit
    pub exit_commission: f64,

    /// Slippage cost on entry
    pub entry_slippage: f64,

    /// Slippage cost on exit
    pub exit_slippage: f64,

    /// Net P&L (after all fees)
    pub net_pnl: f64,

    /// Return as percentage of entry value
    pub return_pct: f64,

    /// Entry signal confidence
    pub entry_confidence: f64,

    /// Exit signal confidence (if applicable)
    pub exit_confidence: Option<f64>,

    /// Number of candles/bars held
    pub duration_bars: usize,

    /// Maximum favorable excursion (best unrealized profit)
    pub mfe: f64,

    /// Maximum adverse excursion (worst unrealized loss)
    pub mae: f64,
}

impl Trade {
    /// Create a new trade from a position and exit details
    pub fn from_position(
        id: usize,
        position: &Position,
        exit_price: f64,
        exit_time: DateTime<Utc>,
        status: TradeStatus,
        commission_rate: f64,
        slippage_rate: f64,
    ) -> Self {
        // Calculate price change based on direction
        let price_change = match position.side {
            SignalType::Buy => exit_price - position.entry_price,
            SignalType::Sell => position.entry_price - exit_price,
            _ => 0.0,
        };

        // Calculate gross P&L
        let gross_pnl = price_change * position.size;

        // Calculate commissions
        let entry_value = position.entry_price * position.size;
        let exit_value = exit_price * position.size;
        let entry_commission = entry_value * commission_rate;
        let exit_commission = exit_value * commission_rate;

        // Calculate slippage
        let entry_slippage = entry_value * slippage_rate;
        let exit_slippage = exit_value * slippage_rate;

        // Net P&L
        let total_costs = entry_commission + exit_commission + entry_slippage + exit_slippage;
        let net_pnl = gross_pnl - total_costs;

        // Return percentage
        let return_pct = if entry_value != 0.0 {
            (net_pnl / entry_value) * 100.0
        } else {
            0.0
        };

        // Duration in bars (approximation)
        let duration_secs = (exit_time - position.entry_time).num_seconds();
        let duration_bars = (duration_secs / 60) as usize; // Assuming 1-minute bars

        Self {
            id,
            asset: position.asset.clone(),
            side: position.side,
            entry_price: position.entry_price,
            exit_price,
            size: position.size,
            entry_time: position.entry_time,
            exit_time,
            status,
            gross_pnl,
            entry_commission,
            exit_commission,
            entry_slippage,
            exit_slippage,
            net_pnl,
            return_pct,
            entry_confidence: position.entry_signal.confidence,
            exit_confidence: None,
            duration_bars,
            mfe: gross_pnl.max(0.0), // Simplified
            mae: gross_pnl.min(0.0), // Simplified
        }
    }

    /// Check if this was a winning trade
    pub fn is_winner(&self) -> bool {
        self.net_pnl > 0.0
    }

    /// Check if this was a losing trade
    pub fn is_loser(&self) -> bool {
        self.net_pnl < 0.0
    }

    /// Get total fees paid
    pub fn total_fees(&self) -> f64 {
        self.entry_commission + self.exit_commission + self.entry_slippage + self.exit_slippage
    }

    /// Get trade duration in seconds
    pub fn duration_seconds(&self) -> i64 {
        (self.exit_time - self.entry_time).num_seconds()
    }

    /// Get trade duration in hours
    pub fn duration_hours(&self) -> f64 {
        self.duration_seconds() as f64 / 3600.0
    }

    /// Get R-multiple (profit relative to initial risk)
    pub fn r_multiple(&self, initial_risk: f64) -> Option<f64> {
        if initial_risk > 0.0 {
            Some(self.net_pnl / initial_risk)
        } else {
            None
        }
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Trade #{}: {} {} @ {:.2} -> {:.2} | P&L: ${:.2} ({:.2}%) | Duration: {:.1}h | {}",
            self.id,
            self.side,
            self.asset,
            self.entry_price,
            self.exit_price,
            self.net_pnl,
            self.return_pct,
            self.duration_hours(),
            self.status
        )
    }
}

impl fmt::Display for Trade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trade(#{} {} {} P&L=${:.2} ret={:.2}%)",
            self.id, self.side, self.asset, self.net_pnl, self.return_pct
        )
    }
}

/// Simple trade result for quick metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TradeResult {
    /// Entry price
    pub entry_price: f64,

    /// Exit price
    pub exit_price: f64,

    /// Net P&L
    pub pnl: f64,

    /// Duration in number of candles/bars
    pub duration_candles: usize,

    /// Was this a winning trade?
    pub is_winner: bool,
}

impl TradeResult {
    /// Create a new trade result
    pub fn new(entry_price: f64, exit_price: f64, pnl: f64, duration_candles: usize) -> Self {
        Self {
            entry_price,
            exit_price,
            pnl,
            duration_candles,
            is_winner: pnl > 0.0,
        }
    }

    /// Get return percentage
    pub fn return_pct(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        ((self.exit_price - self.entry_price) / self.entry_price) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signal(signal_type: SignalType, asset: &str, confidence: f64) -> TradingSignal {
        TradingSignal::new(signal_type, confidence, asset.to_string())
    }

    #[test]
    fn test_position_creation() {
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.85);
        let position = Position::new(&signal, 50000.0, 0.1);

        assert_eq!(position.asset, "BTCUSD");
        assert_eq!(position.side, SignalType::Buy);
        assert_eq!(position.entry_price, 50000.0);
        assert_eq!(position.size, 0.1);
        assert_eq!(position.unrealized_pnl, 0.0);
    }

    #[test]
    fn test_position_pnl_long() {
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.85);
        let mut position = Position::new(&signal, 50000.0, 0.1);

        // Price goes up - profit
        position.update_pnl(51000.0);
        assert_eq!(position.unrealized_pnl, 100.0); // (51000 - 50000) * 0.1

        // Price goes down - loss
        position.update_pnl(49000.0);
        assert_eq!(position.unrealized_pnl, -100.0); // (49000 - 50000) * 0.1
    }

    #[test]
    fn test_position_pnl_short() {
        let signal = create_test_signal(SignalType::Sell, "BTCUSD", 0.85);
        let mut position = Position::new(&signal, 50000.0, 0.1);

        // Price goes down - profit
        position.update_pnl(49000.0);
        assert_eq!(position.unrealized_pnl, 100.0); // (50000 - 49000) * 0.1

        // Price goes up - loss
        position.update_pnl(51000.0);
        assert_eq!(position.unrealized_pnl, -100.0); // (50000 - 51000) * 0.1
    }

    #[test]
    fn test_position_return_pct() {
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.85);
        let mut position = Position::new(&signal, 50000.0, 0.1);

        position.update_pnl(51000.0);
        assert!((position.return_pct() - 2.0).abs() < 0.01); // 2% gain
    }

    #[test]
    fn test_position_stop_loss() {
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.85);
        let mut position = Position::new(&signal, 50000.0, 0.1);

        position.set_stop_loss_pct(0.02); // 2% stop
        assert_eq!(position.stop_loss, Some(49000.0));

        assert!(!position.hits_stop_loss(49500.0));
        assert!(position.hits_stop_loss(48900.0));
    }

    #[test]
    fn test_position_take_profit() {
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.85);
        let mut position = Position::new(&signal, 50000.0, 0.1);

        position.set_take_profit_pct(0.05); // 5% target
        assert_eq!(position.take_profit, Some(52500.0));

        assert!(!position.hits_take_profit(52000.0));
        assert!(position.hits_take_profit(52600.0));
    }

    #[test]
    fn test_trade_from_position_long_winner() {
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.85);
        let position = Position::new(&signal, 50000.0, 0.1);

        let exit_time = Utc::now();
        let trade = Trade::from_position(
            1,
            &position,
            51000.0,
            exit_time,
            TradeStatus::Closed,
            0.001,
            0.0005,
        );

        assert_eq!(trade.id, 1);
        assert_eq!(trade.asset, "BTCUSD");
        assert_eq!(trade.side, SignalType::Buy);
        assert_eq!(trade.entry_price, 50000.0);
        assert_eq!(trade.exit_price, 51000.0);
        assert!(trade.gross_pnl > 0.0);
        assert!(trade.is_winner());
        assert!(!trade.is_loser());
    }

    #[test]
    fn test_trade_from_position_long_loser() {
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.85);
        let position = Position::new(&signal, 50000.0, 0.1);

        let exit_time = Utc::now();
        let trade = Trade::from_position(
            1,
            &position,
            49000.0,
            exit_time,
            TradeStatus::StoppedOut,
            0.001,
            0.0005,
        );

        assert!(trade.gross_pnl < 0.0);
        assert!(!trade.is_winner());
        assert!(trade.is_loser());
    }

    #[test]
    fn test_trade_from_position_short_winner() {
        let signal = create_test_signal(SignalType::Sell, "BTCUSD", 0.85);
        let position = Position::new(&signal, 50000.0, 0.1);

        let exit_time = Utc::now();
        let trade = Trade::from_position(
            1,
            &position,
            49000.0,
            exit_time,
            TradeStatus::Closed,
            0.001,
            0.0005,
        );

        assert!(trade.gross_pnl > 0.0);
        assert!(trade.is_winner());
    }

    #[test]
    fn test_trade_fees() {
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.85);
        let position = Position::new(&signal, 50000.0, 0.1);

        let exit_time = Utc::now();
        let trade = Trade::from_position(
            1,
            &position,
            51000.0,
            exit_time,
            TradeStatus::Closed,
            0.001,
            0.0005,
        );

        // Entry value: 50000 * 0.1 = 5000
        // Exit value: 51000 * 0.1 = 5100
        let expected_commission = (5000.0 * 0.001) + (5100.0 * 0.001);
        let expected_slippage = (5000.0 * 0.0005) + (5100.0 * 0.0005);
        let expected_total_fees = expected_commission + expected_slippage;

        assert!((trade.total_fees() - expected_total_fees).abs() < 0.01);
        assert!(trade.net_pnl < trade.gross_pnl); // Net should be less due to fees
    }

    #[test]
    fn test_trade_result() {
        let result = TradeResult::new(100.0, 110.0, 50.0, 10);

        assert_eq!(result.entry_price, 100.0);
        assert_eq!(result.exit_price, 110.0);
        assert_eq!(result.pnl, 50.0);
        assert_eq!(result.duration_candles, 10);
        assert!(result.is_winner);
        assert_eq!(result.return_pct(), 10.0);
    }

    #[test]
    fn test_trade_status_display() {
        assert_eq!(TradeStatus::Open.to_string(), "OPEN");
        assert_eq!(TradeStatus::Closed.to_string(), "CLOSED");
        assert_eq!(TradeStatus::StoppedOut.to_string(), "STOPPED_OUT");
        assert_eq!(TradeStatus::TakeProfit.to_string(), "TAKE_PROFIT");
        assert_eq!(TradeStatus::Reversed.to_string(), "REVERSED");
    }
}
