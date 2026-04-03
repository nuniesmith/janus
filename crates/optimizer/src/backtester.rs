//! Backtester Module
//!
//! Provides the backtesting engine for evaluating trading strategies during optimization.
//! This module wraps the existing `janus-backtest` vectorized indicators and provides
//! a simplified interface for the optimizer.

use crate::constraints::SampledParams;
use crate::error::{OptimizerError, Result};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum data points required for backtesting
pub const MIN_DATA_POINTS: usize = 100;

/// Default initial balance for backtesting
pub const DEFAULT_INITIAL_BALANCE: f64 = 10_000.0;

/// Default slippage in basis points
pub const DEFAULT_SLIPPAGE_BPS: f64 = 5.0;

/// Default commission in basis points
pub const DEFAULT_COMMISSION_BPS: f64 = 6.0;

/// Parameters for running a backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestParams {
    /// Fast EMA period
    pub ema_fast_period: u32,

    /// Slow EMA period
    pub ema_slow_period: u32,

    /// ATR calculation period
    pub atr_length: u32,

    /// ATR multiplier for trailing stop
    pub atr_multiplier: f64,

    /// Minimum trailing stop percentage
    pub min_trailing_stop_pct: f64,

    /// Minimum EMA spread percentage to confirm trend
    pub min_ema_spread_pct: f64,

    /// Minimum profit percentage to allow exit
    pub min_profit_pct: f64,

    /// Take profit target percentage
    pub take_profit_pct: f64,

    /// Whether to require higher timeframe alignment
    pub require_htf_alignment: bool,

    /// Higher timeframe period in minutes
    pub htf_timeframe_minutes: u32,

    /// Cooldown bars between trades
    pub cooldown_bars: u32,

    /// Minimum hold time in bars
    pub min_hold_bars: u32,

    /// Initial balance for simulation
    pub initial_balance: f64,

    /// Slippage in basis points
    pub slippage_bps: f64,

    /// Commission in basis points
    pub commission_bps: f64,
}

impl Default for BacktestParams {
    fn default() -> Self {
        Self {
            ema_fast_period: 9,
            ema_slow_period: 28,
            atr_length: 14,
            atr_multiplier: 2.0,
            min_trailing_stop_pct: 0.5,
            min_ema_spread_pct: 0.20,
            min_profit_pct: 0.40,
            take_profit_pct: 5.0,
            require_htf_alignment: true,
            htf_timeframe_minutes: 15,
            cooldown_bars: 5,
            min_hold_bars: 5,
            initial_balance: DEFAULT_INITIAL_BALANCE,
            slippage_bps: DEFAULT_SLIPPAGE_BPS,
            commission_bps: DEFAULT_COMMISSION_BPS,
        }
    }
}

impl BacktestParams {
    /// Create new params with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create params from sampled optimizer parameters
    pub fn from_sampled(sampled: &SampledParams) -> Self {
        Self {
            ema_fast_period: sampled.get_int("ema_fast_period").unwrap_or(9) as u32,
            ema_slow_period: sampled.get_int("ema_slow_period").unwrap_or(28) as u32,
            atr_length: sampled.get_int("atr_length").unwrap_or(14) as u32,
            atr_multiplier: sampled.get("atr_multiplier").unwrap_or(2.0),
            min_trailing_stop_pct: sampled.get("min_trailing_stop_pct").unwrap_or(0.5),
            min_ema_spread_pct: sampled.get("min_ema_spread_pct").unwrap_or(0.20),
            min_profit_pct: sampled.get("min_profit_pct").unwrap_or(0.40),
            take_profit_pct: sampled.get("take_profit_pct").unwrap_or(5.0),
            require_htf_alignment: sampled.get_bool("require_htf_alignment").unwrap_or(true),
            htf_timeframe_minutes: sampled.get_int("htf_timeframe_minutes").unwrap_or(15) as u32,
            cooldown_bars: sampled.get_int("cooldown_bars").unwrap_or(5) as u32,
            min_hold_bars: 5, // Fixed for now
            initial_balance: DEFAULT_INITIAL_BALANCE,
            slippage_bps: DEFAULT_SLIPPAGE_BPS,
            commission_bps: DEFAULT_COMMISSION_BPS,
        }
    }

    /// Set initial balance
    pub fn with_initial_balance(mut self, balance: f64) -> Self {
        self.initial_balance = balance;
        self
    }

    /// Set slippage
    pub fn with_slippage_bps(mut self, bps: f64) -> Self {
        self.slippage_bps = bps;
        self
    }

    /// Set commission
    pub fn with_commission_bps(mut self, bps: f64) -> Self {
        self.commission_bps = bps;
        self
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        if self.ema_fast_period >= self.ema_slow_period {
            return Err(OptimizerError::invalid_param(
                "ema_periods",
                "fast period must be less than slow period",
            ));
        }

        if self.ema_slow_period - self.ema_fast_period < 10 {
            return Err(OptimizerError::invalid_param(
                "ema_periods",
                "gap between fast and slow must be at least 10",
            ));
        }

        if self.min_ema_spread_pct < 0.0 {
            return Err(OptimizerError::invalid_param(
                "min_ema_spread_pct",
                "must be non-negative",
            ));
        }

        if self.take_profit_pct <= self.min_profit_pct {
            return Err(OptimizerError::invalid_param(
                "take_profit_pct",
                "must be greater than min_profit_pct",
            ));
        }

        Ok(())
    }
}

/// Results from running a backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total number of trades
    pub total_trades: usize,

    /// Number of winning trades
    pub winning_trades: usize,

    /// Number of losing trades
    pub losing_trades: usize,

    /// Total P&L percentage
    pub total_pnl_pct: f64,

    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,

    /// Win rate percentage (0-100)
    pub win_rate: f64,

    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,

    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,

    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,

    /// Average trades per day
    pub trades_per_day: f64,

    /// Average trade duration in minutes
    pub avg_trade_duration_minutes: f64,

    /// Maximum consecutive winning trades
    pub max_consecutive_wins: usize,

    /// Maximum consecutive losing trades
    pub max_consecutive_losses: usize,

    /// Average winner percentage
    pub avg_winner_pct: f64,

    /// Average loser percentage
    pub avg_loser_pct: f64,

    /// Largest winner percentage
    pub largest_winner_pct: f64,

    /// Largest loser percentage
    pub largest_loser_pct: f64,
}

impl Default for BacktestResult {
    fn default() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            total_pnl_pct: 0.0,
            max_drawdown_pct: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            trades_per_day: 0.0,
            avg_trade_duration_minutes: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            avg_winner_pct: 0.0,
            avg_loser_pct: 0.0,
            largest_winner_pct: 0.0,
            largest_loser_pct: 0.0,
        }
    }
}

impl BacktestResult {
    /// Check if this is a valid result (has enough trades)
    pub fn is_valid(&self, min_trades: usize) -> bool {
        self.total_trades >= min_trades
    }

    /// Check if this is a profitable result
    pub fn is_profitable(&self) -> bool {
        self.total_pnl_pct > 0.0 && self.profit_factor > 1.0
    }

    /// Get risk-adjusted return (return / max_drawdown)
    pub fn risk_adjusted_return(&self) -> f64 {
        if self.max_drawdown_pct > 0.0 {
            self.total_pnl_pct / self.max_drawdown_pct
        } else {
            self.total_pnl_pct
        }
    }

    /// Convert to janus-core OptimizedParams format
    pub fn to_summary(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("total_trades".to_string(), self.total_trades as f64);
        map.insert("winning_trades".to_string(), self.winning_trades as f64);
        map.insert("losing_trades".to_string(), self.losing_trades as f64);
        map.insert("total_pnl_pct".to_string(), self.total_pnl_pct);
        map.insert("max_drawdown_pct".to_string(), self.max_drawdown_pct);
        map.insert("win_rate".to_string(), self.win_rate);
        map.insert("profit_factor".to_string(), self.profit_factor);
        map.insert("sharpe_ratio".to_string(), self.sharpe_ratio);
        map.insert("trades_per_day".to_string(), self.trades_per_day);
        map
    }
}

/// Backtest engine for running strategy simulations
pub struct BacktestEngine {
    /// Backtest parameters
    params: BacktestParams,
}

impl BacktestEngine {
    /// Create a new backtest engine with the given parameters
    pub fn new(params: BacktestParams) -> Self {
        Self { params }
    }

    /// Create with default parameters
    pub fn default_engine() -> Self {
        Self::new(BacktestParams::default())
    }

    /// Get the parameters
    pub fn params(&self) -> &BacktestParams {
        &self.params
    }

    /// Run a backtest on the provided OHLC data
    ///
    /// The DataFrame must contain columns: timestamp, open, high, low, close, volume
    pub fn run(&self, data: &DataFrame) -> Result<BacktestResult> {
        // Validate data
        let n_rows = data.height();
        if n_rows < MIN_DATA_POINTS {
            return Err(OptimizerError::insufficient_data(MIN_DATA_POINTS, n_rows));
        }

        // Validate required columns
        self.validate_columns(data)?;

        // Calculate indicators
        let enriched = self.calculate_indicators(data)?;

        // Run simulation
        let result = self.simulate(&enriched)?;

        Ok(result)
    }

    /// Validate that required columns exist
    fn validate_columns(&self, data: &DataFrame) -> Result<()> {
        let required = ["open", "high", "low", "close"];
        for col in required {
            if data.column(col).is_err() {
                return Err(OptimizerError::InvalidParameter {
                    name: "data".to_string(),
                    reason: format!("Missing required column: {}", col),
                });
            }
        }
        Ok(())
    }

    /// Calculate technical indicators on the data
    fn calculate_indicators(&self, data: &DataFrame) -> Result<DataFrame> {
        let close = data.column("close")?.f64()?;
        let high = data.column("high")?.f64()?;
        let low = data.column("low")?.f64()?;

        // Calculate EMAs
        let ema_fast = self.calculate_ema(close, self.params.ema_fast_period as usize)?;
        let ema_slow = self.calculate_ema(close, self.params.ema_slow_period as usize)?;

        // Calculate ATR
        let atr = self.calculate_atr(high, low, close, self.params.atr_length as usize)?;

        // Calculate EMA spread percentage
        let ema_spread: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(fast, slow)| {
                if *slow != 0.0 {
                    ((fast - slow) / slow) * 100.0
                } else {
                    0.0
                }
            })
            .collect();

        // Add new columns to dataframe
        let mut df = data.clone();
        df.with_column(Series::new("ema_fast".into(), ema_fast))?;
        df.with_column(Series::new("ema_slow".into(), ema_slow))?;
        df.with_column(Series::new("atr".into(), atr))?;
        df.with_column(Series::new("ema_spread_pct".into(), ema_spread))?;

        Ok(df)
    }

    /// Calculate EMA using standard formula
    fn calculate_ema(&self, prices: &ChunkedArray<Float64Type>, period: usize) -> Result<Vec<f64>> {
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = Vec::with_capacity(prices.len());
        let mut prev_ema: Option<f64> = None;

        for price in prices.iter() {
            let current_ema = match (price, prev_ema) {
                (Some(p), Some(prev)) => {
                    let new_ema = alpha * p + (1.0 - alpha) * prev;
                    prev_ema = Some(new_ema);
                    new_ema
                }
                (Some(p), None) => {
                    prev_ema = Some(p);
                    p
                }
                (None, Some(prev)) => prev,
                (None, None) => 0.0,
            };
            ema.push(current_ema);
        }

        Ok(ema)
    }

    /// Calculate ATR (Average True Range)
    fn calculate_atr(
        &self,
        high: &ChunkedArray<Float64Type>,
        low: &ChunkedArray<Float64Type>,
        close: &ChunkedArray<Float64Type>,
        period: usize,
    ) -> Result<Vec<f64>> {
        let mut tr_values = Vec::with_capacity(high.len());
        let mut prev_close: Option<f64> = None;

        // Calculate True Range
        for i in 0..high.len() {
            let h = high.get(i).unwrap_or(0.0);
            let l = low.get(i).unwrap_or(0.0);
            let c = close.get(i).unwrap_or(0.0);

            let tr = if let Some(pc) = prev_close {
                (h - l).max((h - pc).abs()).max((l - pc).abs())
            } else {
                h - l
            };

            tr_values.push(tr);
            prev_close = Some(c);
        }

        // Calculate ATR as EMA of TR
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut atr = Vec::with_capacity(tr_values.len());
        let mut prev_atr: Option<f64> = None;

        for tr in tr_values {
            let current_atr = match prev_atr {
                Some(prev) => {
                    let new_atr = alpha * tr + (1.0 - alpha) * prev;
                    prev_atr = Some(new_atr);
                    new_atr
                }
                None => {
                    prev_atr = Some(tr);
                    tr
                }
            };
            atr.push(current_atr);
        }

        Ok(atr)
    }

    /// Run the trading simulation
    fn simulate(&self, data: &DataFrame) -> Result<BacktestResult> {
        let n_rows = data.height();
        let close = data.column("close")?.f64()?;
        let ema_fast = data.column("ema_fast")?.f64()?;
        let ema_slow = data.column("ema_slow")?.f64()?;
        let ema_spread = data.column("ema_spread_pct")?.f64()?;
        let atr = data.column("atr")?.f64()?;

        let mut balance = self.params.initial_balance;
        let mut position: Option<Position> = None;
        let mut trades: Vec<TradeRecord> = Vec::new();
        let mut equity_curve = Vec::with_capacity(n_rows);
        let mut peak_equity = balance;
        let mut max_drawdown_pct = 0.0;
        let mut last_trade_bar: Option<usize> = None;

        // Track previous EMA values for crossover detection
        let mut prev_ema_fast: Option<f64> = None;
        let mut prev_ema_slow: Option<f64> = None;

        // Simulation loop
        for i in 0..n_rows {
            let price = close.get(i).unwrap_or(0.0);
            let fast = ema_fast.get(i).unwrap_or(0.0);
            let slow = ema_slow.get(i).unwrap_or(0.0);
            let spread = ema_spread.get(i).unwrap_or(0.0);
            let current_atr = atr.get(i).unwrap_or(0.0);

            // Update equity curve
            let current_equity = if let Some(ref pos) = position {
                let unrealized_pnl = (price - pos.entry_price) / pos.entry_price * 100.0;
                balance * (1.0 + unrealized_pnl / 100.0)
            } else {
                balance
            };
            equity_curve.push(current_equity);

            // Update max drawdown
            if current_equity > peak_equity {
                peak_equity = current_equity;
            }
            let drawdown = (peak_equity - current_equity) / peak_equity * 100.0;
            if drawdown > max_drawdown_pct {
                max_drawdown_pct = drawdown;
            }

            // Check for exit signals if in position
            if let Some(ref pos) = position {
                let pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0;
                let bars_held = i - pos.entry_bar;
                let trailing_stop = pos.entry_price
                    * (1.0
                        - (current_atr * self.params.atr_multiplier / pos.entry_price)
                            .max(self.params.min_trailing_stop_pct / 100.0));

                let should_exit =
                    // Take profit hit
                    pnl_pct >= self.params.take_profit_pct ||
                    // Trailing stop hit
                    price < trailing_stop ||
                    // EMA reversal (bearish crossover)
                    (prev_ema_fast.is_some() && prev_ema_slow.is_some() &&
                     prev_ema_fast.unwrap() >= prev_ema_slow.unwrap() &&
                     fast < slow) ||
                    // Below minimum profit and held long enough
                    (bars_held >= self.params.min_hold_bars as usize &&
                     pnl_pct < -self.params.min_trailing_stop_pct);

                if should_exit {
                    // Close position
                    let exit_price = price * (1.0 - self.params.slippage_bps / 10000.0);
                    let final_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0;
                    let commission_pct = self.params.commission_bps / 100.0;
                    let net_pnl_pct = final_pnl_pct - commission_pct;

                    balance *= 1.0 + net_pnl_pct / 100.0;

                    trades.push(TradeRecord {
                        entry_bar: pos.entry_bar,
                        exit_bar: i,
                        entry_price: pos.entry_price,
                        exit_price,
                        pnl_pct: net_pnl_pct,
                        duration_bars: i - pos.entry_bar,
                    });

                    position = None;
                    last_trade_bar = Some(i);
                }
            }

            // Check for entry signals if not in position
            if position.is_none() {
                // Check cooldown
                let cooldown_ok = last_trade_bar
                    .map(|b| i - b >= self.params.cooldown_bars as usize)
                    .unwrap_or(true);

                // Check for bullish crossover
                let crossover = prev_ema_fast.is_some()
                    && prev_ema_slow.is_some()
                    && prev_ema_fast.unwrap() <= prev_ema_slow.unwrap()
                    && fast > slow;

                // Check EMA spread threshold
                let spread_ok = spread.abs() >= self.params.min_ema_spread_pct;

                if cooldown_ok && crossover && spread_ok {
                    // Enter long position
                    let entry_price = price * (1.0 + self.params.slippage_bps / 10000.0);
                    position = Some(Position {
                        entry_bar: i,
                        entry_price,
                    });
                }
            }

            // Update previous values
            prev_ema_fast = Some(fast);
            prev_ema_slow = Some(slow);
        }

        // Calculate final metrics
        let result = self.calculate_metrics(&trades, &equity_curve, max_drawdown_pct, n_rows);

        Ok(result)
    }

    /// Calculate performance metrics from trades
    fn calculate_metrics(
        &self,
        trades: &[TradeRecord],
        _equity_curve: &[f64],
        max_drawdown_pct: f64,
        n_rows: usize,
    ) -> BacktestResult {
        let total_trades = trades.len();

        if total_trades == 0 {
            return BacktestResult {
                max_drawdown_pct,
                ..Default::default()
            };
        }

        let winners: Vec<_> = trades.iter().filter(|t| t.pnl_pct > 0.0).collect();
        let losers: Vec<_> = trades.iter().filter(|t| t.pnl_pct <= 0.0).collect();

        let winning_trades = winners.len();
        let losing_trades = losers.len();
        let win_rate = (winning_trades as f64 / total_trades as f64) * 100.0;

        let total_pnl_pct: f64 = trades.iter().map(|t| t.pnl_pct).sum();
        let gross_profit: f64 = winners.iter().map(|t| t.pnl_pct).sum();
        let gross_loss: f64 = losers.iter().map(|t| t.pnl_pct.abs()).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            10.0 // Cap at 10 if no losses
        } else {
            0.0
        };

        let avg_winner_pct = if winning_trades > 0 {
            gross_profit / winning_trades as f64
        } else {
            0.0
        };

        let avg_loser_pct = if losing_trades > 0 {
            gross_loss / losing_trades as f64
        } else {
            0.0
        };

        let largest_winner_pct = winners
            .iter()
            .map(|t| t.pnl_pct)
            .fold(0.0f64, |a, b| a.max(b));

        let largest_loser_pct = losers
            .iter()
            .map(|t| t.pnl_pct.abs())
            .fold(0.0f64, |a, b| a.max(b));

        // Calculate consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) = self.calculate_streaks(trades);

        // Calculate average trade duration
        let avg_duration: f64 =
            trades.iter().map(|t| t.duration_bars as f64).sum::<f64>() / total_trades as f64;

        // Estimate trades per day (assuming 1-hour candles, 24 candles per day)
        let data_days = n_rows as f64 / 24.0;
        let trades_per_day = if data_days > 0.0 {
            total_trades as f64 / data_days
        } else {
            0.0
        };

        // Calculate Sharpe ratio (simplified)
        let returns: Vec<f64> = trades.iter().map(|t| t.pnl_pct).collect();
        let sharpe_ratio = self.calculate_sharpe(&returns);
        let sortino_ratio = self.calculate_sortino(&returns);

        BacktestResult {
            total_trades,
            winning_trades,
            losing_trades,
            total_pnl_pct,
            max_drawdown_pct,
            win_rate,
            profit_factor,
            sharpe_ratio,
            sortino_ratio,
            trades_per_day,
            avg_trade_duration_minutes: avg_duration * 60.0, // Assuming 1-hour bars
            max_consecutive_wins,
            max_consecutive_losses,
            avg_winner_pct,
            avg_loser_pct,
            largest_winner_pct,
            largest_loser_pct,
        }
    }

    /// Calculate consecutive win/loss streaks
    fn calculate_streaks(&self, trades: &[TradeRecord]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.pnl_pct > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }

        (max_wins, max_losses)
    }

    /// Calculate annualized Sharpe ratio
    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            // Annualize assuming daily trades
            (mean / std_dev) * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }

    /// Calculate annualized Sortino ratio (downside deviation only)
    fn calculate_sortino(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        if downside_returns.is_empty() {
            return if mean > 0.0 { 10.0 } else { 0.0 }; // Cap if no downside
        }

        let downside_variance: f64 =
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
        let downside_dev = downside_variance.sqrt();

        if downside_dev > 0.0 {
            (mean / downside_dev) * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }
}

/// Internal position tracking
#[derive(Debug, Clone)]
struct Position {
    entry_bar: usize,
    entry_price: f64,
}

/// Internal trade record
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TradeRecord {
    entry_bar: usize,
    exit_bar: usize,
    entry_price: f64,
    exit_price: f64,
    pnl_pct: f64,
    duration_bars: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> DataFrame {
        // Create simple uptrending data
        let n = 200;
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);

        let mut price = 100.0;
        for i in 0..n {
            // Add some trend and noise
            let trend = if i < 100 { 0.1 } else { -0.05 };
            let noise = (i as f64 * 0.1).sin() * 0.5;
            price += trend + noise;

            let o = price;
            let c = price + trend;
            let h = o.max(c) + 0.5;
            let l = o.min(c) - 0.5;

            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
        }

        DataFrame::new(vec![
            Column::new("open".into(), open),
            Column::new("high".into(), high),
            Column::new("low".into(), low),
            Column::new("close".into(), close),
        ])
        .unwrap()
    }

    #[test]
    fn test_backtest_params_default() {
        let params = BacktestParams::default();
        assert_eq!(params.ema_fast_period, 9);
        assert_eq!(params.ema_slow_period, 28);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_backtest_params_validation() {
        // Invalid: fast >= slow
        let params = BacktestParams {
            ema_fast_period: 30,
            ema_slow_period: 28,
            ..Default::default()
        };
        assert!(params.validate().is_err());

        // Invalid: gap too small
        let mut params = BacktestParams {
            ema_fast_period: 20,
            ..Default::default()
        };
        params.ema_slow_period = 25;
        assert!(params.validate().is_err());

        // Valid
        params.ema_fast_period = 9;
        params.ema_slow_period = 28;
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_backtest_engine_creation() {
        let engine = BacktestEngine::default_engine();
        assert_eq!(engine.params().ema_fast_period, 9);
    }

    #[test]
    fn test_backtest_run() {
        let engine = BacktestEngine::default_engine();
        let data = create_test_data();
        let result = engine.run(&data);

        assert!(result.is_ok());
        let result = result.unwrap();

        // Should have some trades given the trending data
        // (may be 0 depending on exact parameters, but shouldn't error)
        assert!(result.win_rate >= 0.0 && result.win_rate <= 100.0);
        assert!(result.max_drawdown_pct >= 0.0);
    }

    #[test]
    fn test_backtest_result_methods() {
        let result = BacktestResult {
            total_trades: 20,
            winning_trades: 12,
            losing_trades: 8,
            total_pnl_pct: 15.0,
            max_drawdown_pct: 5.0,
            win_rate: 60.0,
            profit_factor: 1.8,
            ..Default::default()
        };

        assert!(result.is_valid(10));
        assert!(!result.is_valid(25));
        assert!(result.is_profitable());
        assert_eq!(result.risk_adjusted_return(), 3.0); // 15 / 5
    }

    #[test]
    fn test_insufficient_data() {
        let engine = BacktestEngine::default_engine();
        let data = DataFrame::new(vec![
            Column::new("open".into(), vec![100.0; 50]),
            Column::new("high".into(), vec![101.0; 50]),
            Column::new("low".into(), vec![99.0; 50]),
            Column::new("close".into(), vec![100.5; 50]),
        ])
        .unwrap();

        let result = engine.run(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_columns() {
        let engine = BacktestEngine::default_engine();
        let data = DataFrame::new(vec![
            Column::new("open".into(), vec![100.0; 200]),
            Column::new("close".into(), vec![100.5; 200]),
            // Missing high and low
        ])
        .unwrap();

        let result = engine.run(&data);
        assert!(result.is_err());
    }
}
