//! Performance Metrics Module
//!
//! This module provides performance calculation and analysis for trading strategies,
//! including Sharpe ratio, Calmar ratio, drawdown analysis, and other key metrics.

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;

/// Performance metrics for a trading strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Returns
    pub total_return: f64,
    pub total_return_percent: f64,
    pub annualized_return: f64,
    pub cagr: f64,

    // Risk metrics
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_percent: f64,
    pub avg_drawdown: f64,

    // Trade statistics
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,

    // Win/Loss analysis
    pub avg_win: f64,
    pub avg_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub avg_win_loss_ratio: f64,

    // Consistency metrics
    pub expectancy: f64,
    pub consecutive_wins: usize,
    pub consecutive_losses: usize,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,

    // Time analysis
    pub avg_trade_duration_hours: f64,
    pub avg_bars_in_trade: f64,
    pub total_days_traded: usize,

    // Exposure
    pub time_in_market_percent: f64,
    pub buy_and_hold_return: f64,
}

impl PerformanceMetrics {
    /// Calculate comprehensive performance metrics from trade results
    pub fn calculate(trades: &[TradeResult], initial_capital: f64, total_days: f64) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        // Calculate returns
        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let total_return = total_pnl;
        let total_return_percent = (total_pnl / initial_capital) * 100.0;

        // Calculate CAGR (Compound Annual Growth Rate)
        let years = total_days / 365.25;
        let final_value = initial_capital + total_pnl;
        let cagr = if years > 0.0 {
            ((final_value / initial_capital).powf(1.0 / years) - 1.0) * 100.0
        } else {
            0.0
        };

        // Separate wins and losses
        let wins: Vec<f64> = trades
            .iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .collect();
        let losses: Vec<f64> = trades
            .iter()
            .filter(|t| t.pnl <= 0.0)
            .map(|t| t.pnl)
            .collect();

        let total_trades = trades.len();
        let winning_trades = wins.len();
        let losing_trades = losses.len();
        let win_rate = (winning_trades as f64 / total_trades as f64) * 100.0;

        // Win/Loss statistics
        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };

        let largest_win = wins.iter().cloned().fold(0.0, f64::max);
        let largest_loss = losses.iter().cloned().fold(0.0, f64::min);

        let avg_win_loss_ratio = if avg_loss != 0.0 {
            avg_win / avg_loss.abs()
        } else {
            0.0
        };

        // Profit factor
        let gross_profit: f64 = wins.iter().sum();
        let gross_loss: f64 = losses.iter().sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            0.0
        };

        // Expectancy
        let expectancy = (win_rate / 100.0 * avg_win) - ((1.0 - win_rate / 100.0) * avg_loss.abs());

        // Calculate drawdown
        let (max_dd, max_dd_pct, avg_dd) = Self::calculate_drawdown(trades, initial_capital);

        // Calculate Sharpe ratio
        let returns: Vec<f64> = trades
            .iter()
            .map(|t| (t.pnl / initial_capital) * 100.0)
            .collect();
        let sharpe_ratio = Self::calculate_sharpe_ratio(&returns);

        // Calculate Sortino ratio
        let sortino_ratio = Self::calculate_sortino_ratio(&returns);

        // Calculate Calmar ratio
        let calmar_ratio = if max_dd_pct != 0.0 {
            cagr / max_dd_pct.abs()
        } else {
            0.0
        };

        // Calculate consecutive wins/losses
        let (max_cons_wins, max_cons_losses, current_cons_wins, current_cons_losses) =
            Self::calculate_consecutive_streaks(trades);

        // Time analysis
        let trade_durations: Vec<f64> = trades
            .iter()
            .filter_map(|t| {
                t.exit_time
                    .map(|exit| exit.signed_duration_since(t.entry_time).num_hours() as f64)
            })
            .collect();

        let avg_trade_duration_hours = if !trade_durations.is_empty() {
            trade_durations.iter().sum::<f64>() / trade_durations.len() as f64
        } else {
            0.0
        };

        // Estimate annualized return
        let annualized_return = if years > 0.0 {
            total_return_percent / years
        } else {
            0.0
        };

        Self {
            total_return,
            total_return_percent,
            annualized_return,
            cagr,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown: max_dd,
            max_drawdown_percent: max_dd_pct,
            avg_drawdown: avg_dd,
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
            expectancy,
            consecutive_wins: current_cons_wins,
            consecutive_losses: current_cons_losses,
            max_consecutive_wins: max_cons_wins,
            max_consecutive_losses: max_cons_losses,
            avg_trade_duration_hours,
            avg_bars_in_trade: 0.0,
            total_days_traded: total_days as usize,
            time_in_market_percent: 0.0,
            buy_and_hold_return: 0.0,
        }
    }

    /// Calculate drawdown statistics
    fn calculate_drawdown(trades: &[TradeResult], initial_capital: f64) -> (f64, f64, f64) {
        let mut equity = initial_capital;
        let mut peak = initial_capital;
        let mut max_dd = 0.0;
        let mut max_dd_pct = 0.0;
        let mut drawdowns = Vec::new();

        for trade in trades {
            equity += trade.pnl;

            if equity > peak {
                peak = equity;
            }

            let dd = peak - equity;
            let dd_pct = (dd / peak) * 100.0;

            if dd > max_dd {
                max_dd = dd;
            }

            if dd_pct > max_dd_pct {
                max_dd_pct = dd_pct;
            }

            if dd > 0.0 {
                drawdowns.push(dd_pct);
            }
        }

        let avg_dd = if !drawdowns.is_empty() {
            drawdowns.iter().sum::<f64>() / drawdowns.len() as f64
        } else {
            0.0
        };

        (max_dd, max_dd_pct, avg_dd)
    }

    /// Calculate Sharpe ratio (risk-free rate assumed to be 0)
    fn calculate_sharpe_ratio(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean_return = returns.mean();
        let std_dev = returns.std_dev();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualized Sharpe (assuming daily returns)
        let sharpe = mean_return / std_dev;
        sharpe * (252.0_f64).sqrt() // Annualize assuming 252 trading days
    }

    /// Calculate Sortino ratio (downside deviation only)
    fn calculate_sortino_ratio(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean_return = returns.mean();

        // Calculate downside deviation (only negative returns)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside_returns.is_empty() {
            return 0.0;
        }

        let downside_deviation = downside_returns.std_dev();

        if downside_deviation == 0.0 {
            return 0.0;
        }

        let sortino = mean_return / downside_deviation;
        sortino * (252.0_f64).sqrt()
    }

    /// Calculate consecutive win/loss streaks
    fn calculate_consecutive_streaks(trades: &[TradeResult]) -> (usize, usize, usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.pnl > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }

        (max_wins, max_losses, current_wins, current_losses)
    }

    /// Print formatted metrics report
    pub fn print_report(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║              📊 PERFORMANCE METRICS REPORT                   ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        println!("Returns:");
        println!(
            "  Total Return:         ${:.2} ({:.2}%)",
            self.total_return, self.total_return_percent
        );
        println!("  Annualized Return:    {:.2}%", self.annualized_return);
        println!("  CAGR:                 {:.2}%", self.cagr);
        println!();

        println!("Risk Metrics:");
        println!("  Sharpe Ratio:         {:.2}", self.sharpe_ratio);
        println!("  Sortino Ratio:        {:.2}", self.sortino_ratio);
        println!("  Calmar Ratio:         {:.2}", self.calmar_ratio);
        println!(
            "  Max Drawdown:         ${:.2} ({:.2}%)",
            self.max_drawdown, self.max_drawdown_percent
        );
        println!("  Avg Drawdown:         {:.2}%", self.avg_drawdown);
        println!();

        println!("Trade Statistics:");
        println!("  Total Trades:         {}", self.total_trades);
        println!(
            "  Winning Trades:       {} ({:.1}%)",
            self.winning_trades, self.win_rate
        );
        println!("  Losing Trades:        {}", self.losing_trades);
        println!("  Profit Factor:        {:.2}", self.profit_factor);
        println!();

        println!("Win/Loss Analysis:");
        println!("  Average Win:          ${:.2}", self.avg_win);
        println!("  Average Loss:         ${:.2}", self.avg_loss);
        println!("  Largest Win:          ${:.2}", self.largest_win);
        println!("  Largest Loss:         ${:.2}", self.largest_loss);
        println!("  Win/Loss Ratio:       {:.2}", self.avg_win_loss_ratio);
        println!("  Expectancy:           ${:.2}", self.expectancy);
        println!();

        println!("Consistency:");
        println!("  Max Consecutive Wins:    {}", self.max_consecutive_wins);
        println!("  Max Consecutive Losses:  {}", self.max_consecutive_losses);
        println!(
            "  Avg Trade Duration:      {:.1} hours",
            self.avg_trade_duration_hours
        );
        println!();
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            total_return_percent: 0.0,
            annualized_return: 0.0,
            cagr: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            max_drawdown_percent: 0.0,
            avg_drawdown: 0.0,
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
            expectancy: 0.0,
            consecutive_wins: 0,
            consecutive_losses: 0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            avg_trade_duration_hours: 0.0,
            avg_bars_in_trade: 0.0,
            total_days_traded: 0,
            time_in_market_percent: 0.0,
            buy_and_hold_return: 0.0,
        }
    }
}

/// Individual trade result for performance calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    pub entry_time: DateTime<Utc>,
    pub exit_time: Option<DateTime<Utc>>,
    pub pnl: f64,
    pub pnl_percent: f64,
    pub direction: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_performance_calculation() {
        let trades = vec![
            TradeResult {
                entry_time: Utc::now(),
                exit_time: Some(Utc::now()),
                pnl: 100.0,
                pnl_percent: 2.0,
                direction: "long".to_string(),
            },
            TradeResult {
                entry_time: Utc::now(),
                exit_time: Some(Utc::now()),
                pnl: -50.0,
                pnl_percent: -1.0,
                direction: "short".to_string(),
            },
        ];

        let metrics = PerformanceMetrics::calculate(&trades, 5000.0, 30.0);

        assert_eq!(metrics.total_trades, 2);
        assert_eq!(metrics.winning_trades, 1);
        assert_eq!(metrics.losing_trades, 1);
        assert_eq!(metrics.win_rate, 50.0);
        assert_eq!(metrics.total_return, 50.0);
    }
}
