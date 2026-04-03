//! # Replay Engine
//!
//! Event-driven backtesting engine that replays historical ticks through strategies.
//!
//! ## Architecture
//!
//! ```text
//! Historical Data → Temporal Fortress → Strategy → Execution Simulator → Metrics
//!                         ↓
//!                   Zero Lookahead
//! ```
//!
//! The replay engine:
//! 1. Loads historical tick data
//! 2. Feeds ticks to the temporal fortress one-by-one
//! 3. Invokes strategy logic at each tick
//! 4. Simulates order execution with realistic fills
//! 5. Tracks performance metrics and generates reports

use crate::data_loader::Tick;
use crate::fortress::TemporalFortress;
use chrono::{DateTime, Utc};
use janus_compliance::{ComplianceSheriff, HyroTraderRules};
use janus_strategies::Signal;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur during replay
#[derive(Error, Debug)]
pub enum ReplayError {
    #[error("Fortress error: {0}")]
    Fortress(#[from] crate::fortress::FortressError),

    #[error("Strategy error: {0}")]
    Strategy(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Invalid configuration: {0}")]
    Config(String),
}

/// Simple account for backtesting
#[derive(Debug, Clone)]
pub struct Account {
    pub balance_usdt: f64,
    pub used_margin_usdt: f64,
}

/// Replay configuration
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Initial account balance (USDT)
    pub initial_balance: f64,

    /// Trading symbol
    pub symbol: String,

    /// Prop firm rules to enforce
    pub prop_firm_rules: Option<HyroTraderRules>,

    /// Slippage model (basis points)
    pub slippage_bps: f64,

    /// Commission per trade (basis points)
    pub commission_bps: f64,

    /// Maximum lookback for indicators (events)
    pub max_lookback: usize,

    /// Whether to log every tick (verbose)
    pub verbose: bool,

    /// Tick-by-tick mode (true) or bar-by-bar (false)
    pub tick_by_tick: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            initial_balance: 10_000.0,
            symbol: "BTCUSD".to_string(),
            prop_firm_rules: None,
            slippage_bps: 5.0,   // 0.05%
            commission_bps: 6.0, // 0.06% (Bybit taker fee)
            max_lookback: 1000,
            verbose: false,
            tick_by_tick: true,
        }
    }
}

/// A simulated position
#[derive(Debug, Clone)]
struct Position {
    symbol: String,
    side: PositionSide,
    entry_price: f64,
    quantity: f64,
    entry_time: DateTime<Utc>,
    unrealized_pnl: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PositionSide {
    Long,
    Short,
}

/// A completed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub commission: f64,
    pub duration_seconds: i64,
}

/// Backtest performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestMetrics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub total_return_pct: f64,
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub total_commission: f64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_hours: f64,
}

/// Replay engine for backtesting
pub struct ReplayEngine {
    config: ReplayConfig,
    fortress: TemporalFortress<Tick>,
    account: Account,
    sheriff: Option<ComplianceSheriff>,
    position: Option<Position>,
    trades: Vec<Trade>,
    equity_curve: Vec<(DateTime<Utc>, f64)>,
    peak_equity: f64,
    max_drawdown: f64,
}

impl ReplayEngine {
    /// Create a new replay engine
    pub fn new(config: ReplayConfig) -> Self {
        let initial_balance = config.initial_balance;

        let account = Account {
            balance_usdt: initial_balance,
            used_margin_usdt: 0.0,
        };

        let sheriff = config
            .prop_firm_rules
            .as_ref()
            .map(|rules| ComplianceSheriff::new(rules.clone(), initial_balance));

        Self {
            config,
            fortress: TemporalFortress::new(),
            account,
            sheriff,
            position: None,
            trades: Vec::new(),
            equity_curve: Vec::new(),
            peak_equity: initial_balance,
            max_drawdown: 0.0,
        }
    }

    /// Run backtest with the given ticks and strategy function
    pub fn run<F>(
        &mut self,
        ticks: Vec<Tick>,
        mut strategy_fn: F,
    ) -> Result<BacktestMetrics, ReplayError>
    where
        F: FnMut(&[&Tick]) -> Result<Signal, String>,
    {
        info!("Starting backtest with {} ticks", ticks.len());

        if ticks.is_empty() {
            return Err(ReplayError::Config("No ticks provided".to_string()));
        }

        let start_time = ticks.first().unwrap().timestamp;
        let end_time = ticks.last().unwrap().timestamp;

        // Load ticks into fortress
        self.fortress.load_events(ticks);

        // Record initial equity
        self.equity_curve
            .push((start_time, self.account.balance_usdt));

        // Replay loop
        while self.fortress.has_more() {
            let next_time = self.fortress.peek_next_time().unwrap();

            // Advance fortress to next tick
            self.fortress.advance_to(next_time)?;

            // Get current tick
            let current_ticks = self.fortress.get_at(next_time)?;
            if current_ticks.is_empty() {
                continue;
            }

            let current_tick = &current_ticks[0];
            let current_price = current_tick.price;

            // Update position mark-to-market
            self.update_position_mtm(current_price);

            // Get recent ticks for strategy
            let recent = self.fortress.get_recent(self.config.max_lookback);

            // Invoke strategy
            let signal = strategy_fn(&recent).map_err(ReplayError::Strategy)?;

            if self.config.verbose {
                debug!(
                    "Tick: {} @ {} | Signal: {:?} | Position: {:?}",
                    next_time,
                    current_price,
                    signal,
                    self.position.as_ref().map(|p| p.side)
                );
            }

            // Execute signal
            self.execute_signal(signal, current_price, next_time)?;

            // Record equity
            let equity = self.calculate_equity(current_price);
            self.equity_curve.push((next_time, equity));

            // Track drawdown
            if equity > self.peak_equity {
                self.peak_equity = equity;
            }
            let drawdown = self.peak_equity - equity;
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }
        }

        // Close any remaining position at final price
        if self.position.is_some() {
            let final_tick = self.fortress.get_recent(1);
            if !final_tick.is_empty() {
                let final_price = final_tick[0].price;
                let final_time = final_tick[0].timestamp;
                info!("Closing final position at {}", final_price);
                self.close_position(final_price, final_time)?;
            }
        }

        info!("Backtest complete. Total trades: {}", self.trades.len());

        // Calculate metrics
        self.calculate_metrics(start_time, end_time)
    }

    /// Update position mark-to-market
    fn update_position_mtm(&mut self, current_price: f64) {
        if let Some(ref mut pos) = self.position {
            let price_diff = match pos.side {
                PositionSide::Long => current_price - pos.entry_price,
                PositionSide::Short => pos.entry_price - current_price,
            };
            pos.unrealized_pnl = price_diff * pos.quantity;
        }
    }

    /// Calculate current equity (balance + unrealized PnL)
    fn calculate_equity(&self, current_price: f64) -> f64 {
        let mut equity = self.account.balance_usdt;

        if let Some(ref pos) = self.position {
            let price_diff = match pos.side {
                PositionSide::Long => current_price - pos.entry_price,
                PositionSide::Short => pos.entry_price - current_price,
            };
            equity += price_diff * pos.quantity;
        }

        equity
    }

    /// Execute a trading signal
    fn execute_signal(
        &mut self,
        signal: Signal,
        price: f64,
        time: DateTime<Utc>,
    ) -> Result<(), ReplayError> {
        match signal {
            Signal::Buy => {
                if self.position.is_none() {
                    self.open_position(PositionSide::Long, price, time)?;
                } else if let Some(ref pos) = self.position
                    && pos.side == PositionSide::Short
                {
                    // Close short, open long
                    self.close_position(price, time)?;
                    self.open_position(PositionSide::Long, price, time)?;
                }
            }
            Signal::Sell => {
                if self.position.is_none() {
                    self.open_position(PositionSide::Short, price, time)?;
                } else if let Some(ref pos) = self.position
                    && pos.side == PositionSide::Long
                {
                    // Close long, open short
                    self.close_position(price, time)?;
                    self.open_position(PositionSide::Short, price, time)?;
                }
            }
            Signal::Close => {
                if self.position.is_some() {
                    self.close_position(price, time)?;
                }
            }
            Signal::None => {
                // Do nothing
            }
        }

        Ok(())
    }

    /// Open a new position
    fn open_position(
        &mut self,
        side: PositionSide,
        price: f64,
        time: DateTime<Utc>,
    ) -> Result<(), ReplayError> {
        // Apply slippage
        let slippage_multiplier = 1.0 + (self.config.slippage_bps / 10000.0);
        let execution_price = match side {
            PositionSide::Long => price * slippage_multiplier,
            PositionSide::Short => price / slippage_multiplier,
        };

        // Calculate position size (use 80% of available balance for safety)
        let available = self.account.balance_usdt * 0.8;
        let quantity = available / execution_price;

        // Check prop firm rules if applicable
        if let Some(ref sheriff) = self.sheriff {
            let stop_loss = match side {
                PositionSide::Long => Some(execution_price * 0.98), // 2% stop loss
                PositionSide::Short => Some(execution_price * 1.02), // 2% stop loss
            };

            let order_risk = (execution_price * quantity) * 0.02; // 2% risk
            let current_equity = self.calculate_equity(execution_price);

            if sheriff
                .validate_order(current_equity, order_risk, stop_loss)
                .is_err()
            {
                warn!("Position rejected by prop firm rules");
                return Ok(());
            }
        }

        // Calculate commission
        let notional = execution_price * quantity;
        let commission = notional * (self.config.commission_bps / 10000.0);
        self.account.balance_usdt -= commission;

        let position = Position {
            symbol: self.config.symbol.clone(),
            side,
            entry_price: execution_price,
            quantity,
            entry_time: time,
            unrealized_pnl: 0.0,
        };

        info!(
            "OPEN {:?} @ {} | Qty: {:.4} | Commission: {:.2}",
            side, execution_price, quantity, commission
        );

        self.position = Some(position);
        Ok(())
    }

    /// Close the current position
    fn close_position(&mut self, price: f64, time: DateTime<Utc>) -> Result<(), ReplayError> {
        let pos = self.position.take().ok_or_else(|| {
            ReplayError::Execution("Attempted to close non-existent position".to_string())
        })?;

        // Apply slippage
        let slippage_multiplier = 1.0 + (self.config.slippage_bps / 10000.0);
        let execution_price = match pos.side {
            PositionSide::Long => price / slippage_multiplier,
            PositionSide::Short => price * slippage_multiplier,
        };

        // Calculate PnL
        let price_diff = match pos.side {
            PositionSide::Long => execution_price - pos.entry_price,
            PositionSide::Short => pos.entry_price - execution_price,
        };
        let gross_pnl = price_diff * pos.quantity;

        // Calculate commission
        let notional = execution_price * pos.quantity;
        let commission = notional * (self.config.commission_bps / 10000.0);
        let net_pnl = gross_pnl - commission;

        // Update account
        self.account.balance_usdt += net_pnl;

        let pnl_pct = (net_pnl / (pos.entry_price * pos.quantity)) * 100.0;
        let duration = (time - pos.entry_time).num_seconds();

        info!(
            "CLOSE {:?} @ {} | PnL: {:.2} ({:.2}%) | Duration: {}s",
            pos.side, execution_price, net_pnl, pnl_pct, duration
        );

        // Record trade
        let trade = Trade {
            symbol: pos.symbol,
            entry_time: pos.entry_time,
            exit_time: time,
            side: format!("{:?}", pos.side),
            entry_price: pos.entry_price,
            exit_price: execution_price,
            quantity: pos.quantity,
            pnl: net_pnl,
            pnl_pct,
            commission,
            duration_seconds: duration,
        };

        self.trades.push(trade);
        Ok(())
    }

    /// Calculate performance metrics
    fn calculate_metrics(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<BacktestMetrics, ReplayError> {
        let total_trades = self.trades.len();

        if total_trades == 0 {
            return Ok(BacktestMetrics {
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                win_rate: 0.0,
                total_pnl: 0.0,
                total_return_pct: 0.0,
                max_drawdown: 0.0,
                max_drawdown_pct: 0.0,
                sharpe_ratio: 0.0,
                profit_factor: 0.0,
                avg_win: 0.0,
                avg_loss: 0.0,
                largest_win: 0.0,
                largest_loss: 0.0,
                total_commission: 0.0,
                start_time,
                end_time,
                duration_hours: 0.0,
            });
        }

        let winning_trades = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = self.trades.iter().filter(|t| t.pnl <= 0.0).count();
        let win_rate = (winning_trades as f64 / total_trades as f64) * 100.0;

        let total_pnl: f64 = self.trades.iter().map(|t| t.pnl).sum();
        let total_return_pct = (total_pnl / self.config.initial_balance) * 100.0;

        let total_commission: f64 = self.trades.iter().map(|t| t.commission).sum();

        let wins: Vec<f64> = self
            .trades
            .iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .collect();
        let losses: Vec<f64> = self
            .trades
            .iter()
            .filter(|t| t.pnl <= 0.0)
            .map(|t| t.pnl)
            .collect();

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

        let largest_win = wins.iter().copied().fold(0.0f64, f64::max);
        let largest_loss = losses.iter().copied().fold(0.0f64, f64::min);

        let gross_profit: f64 = wins.iter().sum();
        let gross_loss: f64 = losses.iter().sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            0.0
        };

        // Calculate Sharpe ratio (simplified - assumes daily returns)
        let returns: Vec<f64> = self.trades.iter().map(|t| t.pnl_pct).collect();
        let sharpe_ratio = if !returns.is_empty() {
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.0 {
                mean_return / std_dev
            } else {
                0.0
            }
        } else {
            0.0
        };

        let max_drawdown_pct = (self.max_drawdown / self.peak_equity) * 100.0;
        let duration_hours = (end_time - start_time).num_seconds() as f64 / 3600.0;

        Ok(BacktestMetrics {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            total_pnl,
            total_return_pct,
            max_drawdown: self.max_drawdown,
            max_drawdown_pct,
            sharpe_ratio,
            profit_factor,
            avg_win,
            avg_loss,
            largest_win,
            largest_loss,
            total_commission,
            start_time,
            end_time,
            duration_hours,
        })
    }

    /// Get the completed trades
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Get the equity curve
    pub fn equity_curve(&self) -> &[(DateTime<Utc>, f64)] {
        &self.equity_curve
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_indicators::IncrementalEma;

    fn create_test_ticks(count: usize) -> Vec<Tick> {
        let mut ticks = Vec::new();
        let start = Utc::now();

        for i in 0..count {
            let timestamp = start + chrono::Duration::seconds(i as i64);
            // Create a simple sine wave pattern
            let price = 50000.0 + ((i as f64 / 10.0).sin() * 100.0);

            ticks.push(Tick {
                timestamp,
                symbol: "BTCUSD".to_string(),
                price,
                volume: 1.0,
                side: crate::data_loader::Side::Buy,
            });
        }

        ticks
    }

    #[test]
    fn test_replay_engine_initialization() {
        let config = ReplayConfig::default();
        let engine = ReplayEngine::new(config.clone());

        assert_eq!(engine.account.balance_usdt, config.initial_balance);
        assert!(engine.position.is_none());
        assert_eq!(engine.trades.len(), 0);
    }

    #[test]
    fn test_replay_with_strategy() {
        let config = ReplayConfig {
            initial_balance: 10_000.0,
            symbol: "BTCUSD".to_string(),
            verbose: false,
            ..Default::default()
        };

        let mut engine = ReplayEngine::new(config);
        let ticks = create_test_ticks(500);

        // Simple strategy using incremental EMAs
        let mut ema_fast = IncrementalEma::new(8);
        let mut ema_slow = IncrementalEma::new(21);
        let mut prev_fast: Option<f64> = None;
        let mut prev_slow: Option<f64> = None;

        let strategy_fn = |ticks: &[&Tick]| -> Result<Signal, String> {
            if ticks.is_empty() {
                return Ok(Signal::None);
            }
            let price = ticks[0].price;

            let fast = ema_fast.update(price);
            let slow = ema_slow.update(price);

            let signal = if let (Some(pf), Some(ps)) = (prev_fast, prev_slow) {
                if pf <= ps && fast > slow {
                    Signal::Buy
                } else if pf >= ps && fast < slow {
                    Signal::Sell
                } else {
                    Signal::None
                }
            } else {
                Signal::None
            };

            prev_fast = Some(fast);
            prev_slow = Some(slow);

            Ok(signal)
        };

        let result = engine.run(ticks, strategy_fn);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        // Should have executed some trades with sine wave price pattern
        println!("Metrics: {:#?}", metrics);
    }
}
