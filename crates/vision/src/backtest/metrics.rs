//! Performance metrics calculation for backtesting.
//!
//! This module provides comprehensive performance metrics for evaluating
//! trading strategies, including risk-adjusted returns, drawdown analysis,
//! and signal quality assessment.

use super::trade::Trade;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete performance metrics for a trading strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Return Metrics
    /// Total return as percentage
    pub total_return_pct: f64,

    /// Annualized return percentage
    pub annualized_return_pct: f64,

    /// Average return per trade
    pub avg_return_per_trade: f64,

    /// Median return per trade
    pub median_return_per_trade: f64,

    // Risk Metrics
    /// Sharpe ratio (risk-adjusted return)
    pub sharpe_ratio: f64,

    /// Sortino ratio (downside deviation adjusted)
    pub sortino_ratio: f64,

    /// Maximum drawdown as percentage
    pub max_drawdown_pct: f64,

    /// Maximum drawdown in dollar amount
    pub max_drawdown_dollars: f64,

    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,

    // Trade Statistics
    /// Total number of trades
    pub total_trades: usize,

    /// Number of winning trades
    pub winning_trades: usize,

    /// Number of losing trades
    pub losing_trades: usize,

    /// Win rate as percentage (0-1)
    pub win_rate: f64,

    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,

    /// Average winning trade P&L
    pub avg_win: f64,

    /// Average losing trade P&L
    pub avg_loss: f64,

    /// Largest winning trade
    pub largest_win: f64,

    /// Largest losing trade
    pub largest_loss: f64,

    /// Average win/loss ratio
    pub avg_win_loss_ratio: f64,

    // Duration Metrics
    /// Average trade duration in hours
    pub avg_trade_duration_hours: f64,

    /// Average winning trade duration in hours
    pub avg_win_duration_hours: f64,

    /// Average losing trade duration in hours
    pub avg_loss_duration_hours: f64,

    // Consecutive Streaks
    /// Maximum consecutive wins
    pub max_consecutive_wins: usize,

    /// Maximum consecutive losses
    pub max_consecutive_losses: usize,

    // Expectancy
    /// Expected value per trade
    pub expectancy: f64,

    /// Standard deviation of returns
    pub returns_std_dev: f64,

    // Total P&L
    /// Total profit from all trades
    pub total_pnl: f64,

    /// Gross profit (sum of all wins)
    pub gross_profit: f64,

    /// Gross loss (sum of all losses)
    pub gross_loss: f64,

    /// Total fees paid
    pub total_fees: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return_pct: 0.0,
            annualized_return_pct: 0.0,
            avg_return_per_trade: 0.0,
            median_return_per_trade: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown_pct: 0.0,
            max_drawdown_dollars: 0.0,
            calmar_ratio: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            avg_win_loss_ratio: 0.0,
            avg_trade_duration_hours: 0.0,
            avg_win_duration_hours: 0.0,
            avg_loss_duration_hours: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            expectancy: 0.0,
            returns_std_dev: 0.0,
            total_pnl: 0.0,
            gross_profit: 0.0,
            gross_loss: 0.0,
            total_fees: 0.0,
        }
    }
}

impl PerformanceMetrics {
    /// Get a summary string of key metrics
    pub fn summary(&self) -> String {
        format!(
            "Performance Summary:\n\
             Total Return: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             Max Drawdown: {:.2}%\n\
             Win Rate: {:.2}%\n\
             Profit Factor: {:.2}\n\
             Total Trades: {}\n\
             Total P&L: ${:.2}",
            self.total_return_pct,
            self.sharpe_ratio,
            self.max_drawdown_pct,
            self.win_rate * 100.0,
            self.profit_factor,
            self.total_trades,
            self.total_pnl
        )
    }

    /// Check if the strategy meets basic profitability criteria
    pub fn is_profitable(&self) -> bool {
        self.total_return_pct > 0.0 && self.profit_factor > 1.0
    }

    /// Get a quality score (0-100) based on multiple factors
    pub fn quality_score(&self) -> f64 {
        let mut score = 0.0;

        // Return component (max 25 points) - scale returns to 0-25 range
        score += (self.total_return_pct / 4.0).min(25.0);

        // Sharpe ratio component (max 25 points)
        score += (self.sharpe_ratio * 5.0).min(25.0);

        // Win rate component (max 25 points)
        score += (self.win_rate * 25.0).min(25.0);

        // Profit factor component (max 25 points)
        score += ((self.profit_factor - 1.0) * 10.0).min(25.0);

        score.clamp(0.0, 100.0)
    }
}

/// Risk-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Value at Risk (95% confidence)
    pub var_95: f64,

    /// Conditional Value at Risk (expected shortfall)
    pub cvar_95: f64,

    /// Maximum drawdown duration in days
    pub max_drawdown_duration_days: f64,

    /// Current drawdown from peak
    pub current_drawdown_pct: f64,

    /// Ulcer Index (drawdown pain measure)
    pub ulcer_index: f64,

    /// Daily volatility
    pub daily_volatility: f64,

    /// Downside deviation
    pub downside_deviation: f64,
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            var_95: 0.0,
            cvar_95: 0.0,
            max_drawdown_duration_days: 0.0,
            current_drawdown_pct: 0.0,
            ulcer_index: 0.0,
            daily_volatility: 0.0,
            downside_deviation: 0.0,
        }
    }
}

/// Signal quality and accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQuality {
    /// Correlation between signal confidence and actual returns
    pub confidence_correlation: f64,

    /// Average confidence of winning trades
    pub avg_winning_confidence: f64,

    /// Average confidence of losing trades
    pub avg_losing_confidence: f64,

    /// Calibration error (how well confidence predicts outcomes)
    pub calibration_error: f64,

    /// Percentage of high-confidence signals that were winners
    pub high_confidence_accuracy: f64,

    /// Number of signals generated
    pub total_signals: usize,

    /// Number of signals that resulted in trades
    pub signals_traded: usize,

    /// Signal-to-trade conversion rate
    pub signal_trade_rate: f64,
}

impl Default for SignalQuality {
    fn default() -> Self {
        Self {
            confidence_correlation: 0.0,
            avg_winning_confidence: 0.0,
            avg_losing_confidence: 0.0,
            calibration_error: 0.0,
            high_confidence_accuracy: 0.0,
            total_signals: 0,
            signals_traded: 0,
            signal_trade_rate: 0.0,
        }
    }
}

/// Trade statistics breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeStats {
    /// Statistics by asset
    pub by_asset: HashMap<String, AssetStats>,

    /// Statistics by signal type (Buy/Sell)
    pub by_signal_type: HashMap<String, SignalTypeStats>,

    /// Monthly breakdown
    pub monthly_returns: Vec<MonthlyReturn>,
}

impl Default for TradeStats {
    fn default() -> Self {
        Self {
            by_asset: HashMap::new(),
            by_signal_type: HashMap::new(),
            monthly_returns: Vec::new(),
        }
    }
}

/// Per-asset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetStats {
    pub asset: String,
    pub total_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_return: f64,
}

/// Per-signal-type statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalTypeStats {
    pub signal_type: String,
    pub total_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_return: f64,
}

/// Monthly return breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyReturn {
    pub year: i32,
    pub month: u32,
    pub return_pct: f64,
    pub num_trades: usize,
}

/// Calculator for performance metrics
pub struct MetricsCalculator {
    initial_capital: f64,
    risk_free_rate: f64,
}

impl MetricsCalculator {
    /// Create a new metrics calculator
    pub fn new(initial_capital: f64, risk_free_rate: f64) -> Self {
        Self {
            initial_capital,
            risk_free_rate,
        }
    }

    /// Calculate comprehensive performance metrics from trades
    pub fn calculate(&self, trades: &[Trade]) -> PerformanceMetrics {
        if trades.is_empty() {
            return PerformanceMetrics::default();
        }

        let total_trades = trades.len();
        let winners: Vec<_> = trades.iter().filter(|t| t.is_winner()).collect();
        let losers: Vec<_> = trades.iter().filter(|t| t.is_loser()).collect();

        let winning_trades = winners.len();
        let losing_trades = losers.len();
        let win_rate = winning_trades as f64 / total_trades as f64;

        // P&L calculations
        let total_pnl: f64 = trades.iter().map(|t| t.net_pnl).sum();
        let gross_profit: f64 = winners.iter().map(|t| t.net_pnl).sum();
        let gross_loss: f64 = losers.iter().map(|t| t.net_pnl.abs()).sum();
        let total_fees: f64 = trades.iter().map(|t| t.total_fees()).sum();

        // Return metrics
        let total_return_pct = (total_pnl / self.initial_capital) * 100.0;

        // Average and median returns
        let returns: Vec<f64> = trades.iter().map(|t| t.return_pct).collect();
        let avg_return_per_trade = returns.iter().sum::<f64>() / returns.len() as f64;
        let median_return_per_trade = self.median(&returns);

        // Win/Loss statistics
        let avg_win = if winning_trades > 0 {
            gross_profit / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss = if losing_trades > 0 {
            gross_loss / losing_trades as f64
        } else {
            0.0
        };

        let avg_win_loss_ratio = if avg_loss != 0.0 {
            avg_win / avg_loss
        } else {
            0.0
        };

        let largest_win = winners.iter().map(|t| t.net_pnl).fold(0.0, f64::max);
        let largest_loss = losers.iter().map(|t| t.net_pnl).fold(0.0, f64::min);

        // Profit factor
        let profit_factor = if gross_loss != 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Risk metrics
        let returns_std_dev = self.standard_deviation(&returns);
        let sharpe_ratio = self.calculate_sharpe_ratio(&returns, returns_std_dev);
        let sortino_ratio = self.calculate_sortino_ratio(&returns);

        // Drawdown
        let (max_drawdown_pct, max_drawdown_dollars) = self.calculate_max_drawdown(trades);

        let calmar_ratio = if max_drawdown_pct != 0.0 {
            total_return_pct / max_drawdown_pct.abs()
        } else {
            0.0
        };

        // Duration metrics
        let avg_trade_duration_hours =
            trades.iter().map(|t| t.duration_hours()).sum::<f64>() / total_trades as f64;

        let avg_win_duration_hours = if winning_trades > 0 {
            winners.iter().map(|t| t.duration_hours()).sum::<f64>() / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss_duration_hours = if losing_trades > 0 {
            losers.iter().map(|t| t.duration_hours()).sum::<f64>() / losing_trades as f64
        } else {
            0.0
        };

        // Consecutive streaks
        let (max_consecutive_wins, max_consecutive_losses) = self.calculate_streaks(trades);

        // Expectancy
        let expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss);

        // Annualized return (assume 252 trading days)
        let total_days = if let (Some(first), Some(last)) = (trades.first(), trades.last()) {
            (last.exit_time - first.entry_time).num_days() as f64
        } else {
            365.0
        };
        let years = (total_days / 365.0).max(1.0);
        let annualized_return_pct = total_return_pct / years;

        PerformanceMetrics {
            total_return_pct,
            annualized_return_pct,
            avg_return_per_trade,
            median_return_per_trade,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown_pct,
            max_drawdown_dollars,
            calmar_ratio,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            profit_factor,
            avg_win,
            avg_loss,
            largest_win,
            largest_loss,
            avg_win_loss_ratio,
            avg_trade_duration_hours,
            avg_win_duration_hours,
            avg_loss_duration_hours,
            max_consecutive_wins,
            max_consecutive_losses,
            expectancy,
            returns_std_dev,
            total_pnl,
            gross_profit,
            gross_loss,
            total_fees,
        }
    }

    /// Calculate signal quality metrics
    pub fn calculate_signal_quality(
        &self,
        trades: &[Trade],
        total_signals: usize,
    ) -> SignalQuality {
        if trades.is_empty() {
            return SignalQuality::default();
        }

        let winners: Vec<_> = trades.iter().filter(|t| t.is_winner()).collect();
        let losers: Vec<_> = trades.iter().filter(|t| t.is_loser()).collect();

        let avg_winning_confidence = if !winners.is_empty() {
            winners.iter().map(|t| t.entry_confidence).sum::<f64>() / winners.len() as f64
        } else {
            0.0
        };

        let avg_losing_confidence = if !losers.is_empty() {
            losers.iter().map(|t| t.entry_confidence).sum::<f64>() / losers.len() as f64
        } else {
            0.0
        };

        // High confidence trades (>0.8)
        let high_conf_trades: Vec<_> = trades.iter().filter(|t| t.entry_confidence > 0.8).collect();
        let high_conf_winners = high_conf_trades.iter().filter(|t| t.is_winner()).count();
        let high_confidence_accuracy = if !high_conf_trades.is_empty() {
            high_conf_winners as f64 / high_conf_trades.len() as f64
        } else {
            0.0
        };

        // Confidence correlation with returns
        let confidence_correlation = self.calculate_correlation(
            &trades
                .iter()
                .map(|t| t.entry_confidence)
                .collect::<Vec<_>>(),
            &trades.iter().map(|t| t.return_pct).collect::<Vec<_>>(),
        );

        let signals_traded = trades.len();
        let signal_trade_rate = if total_signals > 0 {
            signals_traded as f64 / total_signals as f64
        } else {
            0.0
        };

        SignalQuality {
            confidence_correlation,
            avg_winning_confidence,
            avg_losing_confidence,
            calibration_error: (avg_winning_confidence - avg_losing_confidence).abs(),
            high_confidence_accuracy,
            total_signals,
            signals_traded,
            signal_trade_rate,
        }
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self, returns: &[f64], std_dev: f64) -> f64 {
        if std_dev == 0.0 || returns.is_empty() {
            return 0.0;
        }

        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let excess_return = avg_return - self.risk_free_rate;

        // Annualize (assume daily returns, 252 trading days)
        (excess_return / std_dev) * (252.0_f64).sqrt()
    }

    /// Calculate Sortino ratio (uses downside deviation)
    fn calculate_sortino_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_dev = self.standard_deviation(&downside_returns);
        if downside_dev == 0.0 {
            return 0.0;
        }

        let excess_return = avg_return - self.risk_free_rate;
        (excess_return / downside_dev) * (252.0_f64).sqrt()
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, trades: &[Trade]) -> (f64, f64) {
        if trades.is_empty() {
            return (0.0, 0.0);
        }

        let mut equity = self.initial_capital;
        let mut peak = equity;
        let mut max_dd_pct = 0.0;
        let mut max_dd_dollars = 0.0;

        for trade in trades {
            equity += trade.net_pnl;

            if equity > peak {
                peak = equity;
            }

            let dd_dollars = peak - equity;
            let dd_pct = if peak > 0.0 {
                (dd_dollars / peak) * 100.0
            } else {
                0.0
            };

            if dd_pct > max_dd_pct {
                max_dd_pct = dd_pct;
                max_dd_dollars = dd_dollars;
            }
        }

        (max_dd_pct, max_dd_dollars)
    }

    /// Calculate consecutive win/loss streaks
    fn calculate_streaks(&self, trades: &[Trade]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.is_winner() {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else if trade.is_loser() {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }

        (max_wins, max_losses)
    }

    /// Calculate standard deviation
    fn standard_deviation(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate median
    fn median(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Calculate correlation between two series
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::trade::Position;
    use crate::signals::{SignalType, TradingSignal};
    use chrono::Utc;

    fn create_test_trade(id: usize, pnl: f64, return_pct: f64, confidence: f64) -> Trade {
        let signal = TradingSignal::new(SignalType::Buy, confidence, "BTCUSD".to_string());
        let position = Position::new(&signal, 100.0, 1.0);
        let mut trade = Trade::from_position(
            id,
            &position,
            100.0 + pnl,
            Utc::now(),
            super::super::trade::TradeStatus::Closed,
            0.0,
            0.0,
        );
        trade.net_pnl = pnl;
        trade.return_pct = return_pct;
        trade
    }

    #[test]
    fn test_metrics_calculator_empty() {
        let calc = MetricsCalculator::new(10000.0, 0.0);
        let metrics = calc.calculate(&[]);

        assert_eq!(metrics.total_trades, 0);
        assert_eq!(metrics.total_pnl, 0.0);
    }

    #[test]
    fn test_metrics_calculator_basic() {
        let calc = MetricsCalculator::new(10000.0, 0.0);

        let trades = vec![
            create_test_trade(1, 100.0, 1.0, 0.8),
            create_test_trade(2, -50.0, -0.5, 0.7),
            create_test_trade(3, 150.0, 1.5, 0.9),
        ];

        let metrics = calc.calculate(&trades);

        assert_eq!(metrics.total_trades, 3);
        assert_eq!(metrics.winning_trades, 2);
        assert_eq!(metrics.losing_trades, 1);
        assert!((metrics.win_rate - 0.6667).abs() < 0.01);
        assert_eq!(metrics.total_pnl, 200.0);
    }

    #[test]
    fn test_win_rate_calculation() {
        let calc = MetricsCalculator::new(10000.0, 0.0);

        let trades = vec![
            create_test_trade(1, 100.0, 1.0, 0.8),
            create_test_trade(2, 50.0, 0.5, 0.7),
            create_test_trade(3, -30.0, -0.3, 0.6),
            create_test_trade(4, 80.0, 0.8, 0.85),
        ];

        let metrics = calc.calculate(&trades);

        assert_eq!(metrics.winning_trades, 3);
        assert_eq!(metrics.losing_trades, 1);
        assert_eq!(metrics.win_rate, 0.75);
    }

    #[test]
    fn test_profit_factor() {
        let calc = MetricsCalculator::new(10000.0, 0.0);

        let trades = vec![
            create_test_trade(1, 100.0, 1.0, 0.8),
            create_test_trade(2, 200.0, 2.0, 0.9),
            create_test_trade(3, -50.0, -0.5, 0.7),
            create_test_trade(4, -100.0, -1.0, 0.6),
        ];

        let metrics = calc.calculate(&trades);

        // Gross profit: 300, Gross loss: 150
        assert_eq!(metrics.gross_profit, 300.0);
        assert_eq!(metrics.gross_loss, 150.0);
        assert_eq!(metrics.profit_factor, 2.0);
    }

    #[test]
    fn test_signal_quality() {
        let calc = MetricsCalculator::new(10000.0, 0.0);

        let trades = vec![
            create_test_trade(1, 100.0, 1.0, 0.9),  // High conf winner
            create_test_trade(2, 50.0, 0.5, 0.85),  // High conf winner
            create_test_trade(3, -30.0, -0.3, 0.6), // Low conf loser
        ];

        let quality = calc.calculate_signal_quality(&trades, 10);

        assert!(quality.avg_winning_confidence > quality.avg_losing_confidence);
        assert_eq!(quality.signals_traded, 3);
        assert_eq!(quality.signal_trade_rate, 0.3);
        assert_eq!(quality.high_confidence_accuracy, 1.0); // Both high-conf trades won
    }

    #[test]
    fn test_performance_quality_score() {
        let mut metrics = PerformanceMetrics::default();
        metrics.total_return_pct = 50.0;
        metrics.sharpe_ratio = 2.0;
        metrics.win_rate = 0.6;
        metrics.profit_factor = 2.5;

        let score = metrics.quality_score();
        assert!(score > 50.0);
        assert!(score <= 100.0);
    }

    #[test]
    fn test_is_profitable() {
        let mut metrics = PerformanceMetrics::default();
        assert!(!metrics.is_profitable());

        metrics.total_return_pct = 10.0;
        metrics.profit_factor = 1.5;
        assert!(metrics.is_profitable());
    }
}
