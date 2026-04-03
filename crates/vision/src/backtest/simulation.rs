//! Backtesting simulation engine for strategy validation.
//!
//! This module provides a realistic simulation environment for testing
//! trading signals on historical data, including position management,
//! commission/slippage modeling, and various position sizing strategies.

use super::metrics::MetricsCalculator;
use super::trade::{Position, Trade, TradeStatus};
use crate::signals::{SignalType, TradingSignal};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Position sizing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSizing {
    /// Fixed percentage of current capital
    FixedPercent(f64),

    /// Fixed dollar amount per trade
    FixedDollar(f64),

    /// Kelly Criterion for optimal growth
    Kelly {
        win_rate: f64,
        avg_win_loss_ratio: f64,
    },

    /// Volatility-based sizing (risk parity)
    VolatilityBased { target_volatility: f64 },

    /// Risk-based sizing (fixed risk per trade)
    RiskBased {
        risk_percent: f64,
        stop_loss_percent: f64,
    },
}

impl Default for PositionSizing {
    fn default() -> Self {
        PositionSizing::FixedPercent(0.1) // 10% of capital
    }
}

impl PositionSizing {
    /// Calculate position size based on strategy
    pub fn calculate_size(&self, capital: f64, price: f64, volatility: Option<f64>) -> f64 {
        match self {
            PositionSizing::FixedPercent(pct) => {
                let position_value = capital * pct;
                position_value / price
            }
            PositionSizing::FixedDollar(amount) => amount / price,
            PositionSizing::Kelly {
                win_rate,
                avg_win_loss_ratio,
            } => {
                // Kelly formula: f = (p * b - q) / b
                // where p = win rate, q = 1 - p, b = avg_win/avg_loss
                let q = 1.0 - win_rate;
                let kelly_fraction = ((win_rate * avg_win_loss_ratio) - q) / avg_win_loss_ratio;
                let safe_fraction = kelly_fraction.clamp(0.0, 0.25); // Cap at 25%
                let position_value = capital * safe_fraction;
                position_value / price
            }
            PositionSizing::VolatilityBased { target_volatility } => {
                let vol = volatility.unwrap_or(0.02);
                let scale_factor = target_volatility / vol.max(0.001);
                let position_value = capital * scale_factor.clamp(0.05, 0.5);
                position_value / price
            }
            PositionSizing::RiskBased {
                risk_percent,
                stop_loss_percent,
            } => {
                // Size based on fixed risk amount
                let risk_amount = capital * risk_percent;
                let position_value = risk_amount / stop_loss_percent;
                position_value / price
            }
        }
    }
}

/// Configuration for backtesting simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Initial capital
    pub initial_capital: f64,

    /// Commission rate (as decimal, e.g., 0.001 = 0.1%)
    pub commission_rate: f64,

    /// Slippage rate (as decimal)
    pub slippage_rate: f64,

    /// Position sizing strategy
    pub position_size: PositionSizing,

    /// Minimum confidence threshold for signals
    pub min_confidence: f64,

    /// Maximum number of concurrent positions
    pub max_positions: usize,

    /// Enable stop-loss orders
    pub use_stop_loss: bool,

    /// Stop-loss percentage (if enabled)
    pub stop_loss_pct: f64,

    /// Enable take-profit orders
    pub use_take_profit: bool,

    /// Take-profit percentage (if enabled)
    pub take_profit_pct: f64,

    /// Allow short positions
    pub allow_shorts: bool,

    /// Maximum position hold time in hours (0 = unlimited)
    pub max_hold_hours: f64,

    /// Risk-free rate for Sharpe ratio calculation
    pub risk_free_rate: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission_rate: 0.001, // 0.1%
            slippage_rate: 0.0005,  // 0.05%
            position_size: PositionSizing::default(),
            min_confidence: 0.6,
            max_positions: 1,
            use_stop_loss: true,
            stop_loss_pct: 0.02, // 2%
            use_take_profit: false,
            take_profit_pct: 0.05, // 5%
            allow_shorts: true,
            max_hold_hours: 0.0, // Unlimited
            risk_free_rate: 0.0,
        }
    }
}

/// Current state of the simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    /// Current capital
    pub capital: f64,

    /// Peak capital (for drawdown calculation)
    pub peak_capital: f64,

    /// Current open positions
    pub open_positions: HashMap<String, Position>,

    /// Completed trades
    pub closed_trades: Vec<Trade>,

    /// Number of signals processed
    pub signals_processed: usize,

    /// Number of signals that resulted in trades
    pub signals_traded: usize,

    /// Current equity (capital + unrealized P&L)
    pub current_equity: f64,

    /// Total fees paid
    pub total_fees_paid: f64,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            capital: 0.0,
            peak_capital: 0.0,
            open_positions: HashMap::new(),
            closed_trades: Vec::new(),
            signals_processed: 0,
            signals_traded: 0,
            current_equity: 0.0,
            total_fees_paid: 0.0,
        }
    }
}

/// Main backtesting simulation engine
pub struct BacktestSimulation {
    config: SimulationConfig,
    state: SimulationState,
    next_trade_id: usize,
}

impl BacktestSimulation {
    /// Create a new backtesting simulation
    pub fn new(config: SimulationConfig) -> Self {
        let initial_capital = config.initial_capital;
        Self {
            config,
            state: SimulationState {
                capital: initial_capital,
                peak_capital: initial_capital,
                current_equity: initial_capital,
                ..Default::default()
            },
            next_trade_id: 1,
        }
    }

    /// Process a trading signal
    pub fn process_signal(&mut self, signal: &TradingSignal, current_price: f64) {
        self.state.signals_processed += 1;

        // Check minimum confidence
        if signal.confidence < self.config.min_confidence {
            return;
        }

        // Check if we already have a position in this asset
        let should_update = self.state.open_positions.contains_key(&signal.asset);

        if should_update {
            // Determine action needed for existing position
            let action = {
                let position = self.state.open_positions.get_mut(&signal.asset).unwrap();
                position.update_pnl(current_price);

                // Check stop-loss first (highest priority)
                if self.config.use_stop_loss && position.hits_stop_loss(current_price) {
                    Some((
                        position.asset.clone(),
                        position.stop_loss.unwrap(),
                        TradeStatus::StoppedOut,
                    ))
                }
                // Check take-profit
                else if self.config.use_take_profit && position.hits_take_profit(current_price) {
                    Some((
                        position.asset.clone(),
                        position.take_profit.unwrap(),
                        TradeStatus::TakeProfit,
                    ))
                }
                // Check for close signal
                else if signal.signal_type == SignalType::Close {
                    Some((position.asset.clone(), current_price, TradeStatus::Closed))
                }
                // Check for position reversal (opposite signal)
                else if let Some(opposite) = position.side.opposite() {
                    if signal.signal_type == opposite
                        && signal.confidence >= self.config.min_confidence
                    {
                        Some((position.asset.clone(), current_price, TradeStatus::Reversed))
                    } else {
                        None
                    }
                }
                // Check max hold time (lowest priority)
                else if self.config.max_hold_hours > 0.0 {
                    let hold_hours = position.duration_seconds(Utc::now()) as f64 / 3600.0;
                    if hold_hours >= self.config.max_hold_hours {
                        Some((position.asset.clone(), current_price, TradeStatus::Closed))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            // Execute action if needed
            if let Some((asset, price, status)) = action {
                self.close_position(&asset, price, status);
            }
        } else {
            // Try to open new position
            self.try_open_position(signal, current_price);
        }
    }

    /// Try to open a new position
    fn try_open_position(&mut self, signal: &TradingSignal, price: f64) {
        // Check if signal is actionable
        if !signal.signal_type.is_entry() {
            return;
        }

        // Check if we can open more positions
        if self.state.open_positions.len() >= self.config.max_positions {
            return;
        }

        // Check if shorts are allowed
        if signal.signal_type == SignalType::Sell && !self.config.allow_shorts {
            return;
        }

        // Calculate position size
        let size = self
            .config
            .position_size
            .calculate_size(self.state.capital, price, None);

        // Ensure we have enough capital
        let position_value = price * size;
        if position_value > self.state.capital {
            return;
        }

        // Apply entry slippage
        let slippage = price * self.config.slippage_rate;
        let entry_price = match signal.signal_type {
            SignalType::Buy => price + slippage,
            SignalType::Sell => price - slippage,
            _ => price,
        };

        // Create position
        let mut position = Position::new(signal, entry_price, size);

        // Set stop-loss if enabled
        if self.config.use_stop_loss {
            position.set_stop_loss_pct(self.config.stop_loss_pct);
        }

        // Set take-profit if enabled
        if self.config.use_take_profit {
            position.set_take_profit_pct(self.config.take_profit_pct);
        }

        // Calculate and deduct entry commission
        let entry_commission = position_value * self.config.commission_rate;
        self.state.capital -= position_value + entry_commission;
        self.state.total_fees_paid += entry_commission;

        // Track signal conversion
        self.state.signals_traded += 1;

        // Add to open positions
        self.state
            .open_positions
            .insert(signal.asset.clone(), position);
    }

    /// Close an existing position
    fn close_position(&mut self, asset: &str, exit_price: f64, status: TradeStatus) {
        if let Some(position) = self.state.open_positions.remove(asset) {
            // Apply exit slippage
            let slippage = exit_price * self.config.slippage_rate;
            let actual_exit_price = match position.side {
                SignalType::Buy => exit_price - slippage,
                SignalType::Sell => exit_price + slippage,
                _ => exit_price,
            };

            // Create trade record
            let trade = Trade::from_position(
                self.next_trade_id,
                &position,
                actual_exit_price,
                Utc::now(),
                status,
                self.config.commission_rate,
                self.config.slippage_rate,
            );

            self.next_trade_id += 1;

            // Update capital
            let exit_value = actual_exit_price * position.size;
            self.state.capital += exit_value + trade.net_pnl;
            self.state.total_fees_paid += trade.total_fees();

            // Update peak capital
            if self.state.capital > self.state.peak_capital {
                self.state.peak_capital = self.state.capital;
            }

            // Record trade
            self.state.closed_trades.push(trade);
        }
    }

    /// Update all open positions with current prices
    pub fn update_positions(&mut self, prices: &HashMap<String, f64>) {
        let mut to_close = Vec::new();

        for (asset, position) in self.state.open_positions.iter_mut() {
            if let Some(&price) = prices.get(asset) {
                position.update_pnl(price);

                // Check stop-loss
                if self.config.use_stop_loss && position.hits_stop_loss(price) {
                    to_close.push((
                        asset.clone(),
                        position.stop_loss.unwrap(),
                        TradeStatus::StoppedOut,
                    ));
                    continue;
                }

                // Check take-profit
                if self.config.use_take_profit && position.hits_take_profit(price) {
                    to_close.push((
                        asset.clone(),
                        position.take_profit.unwrap(),
                        TradeStatus::TakeProfit,
                    ));
                }
            }
        }

        // Close positions that hit targets
        for (asset, price, status) in to_close {
            self.close_position(&asset, price, status);
        }

        // Update equity
        self.update_equity(prices);
    }

    /// Update current equity (capital + unrealized P&L)
    fn update_equity(&mut self, prices: &HashMap<String, f64>) {
        let mut unrealized_pnl = 0.0;

        for (asset, position) in &self.state.open_positions {
            if let Some(&price) = prices.get(asset) {
                let pnl = match position.side {
                    SignalType::Buy => (price - position.entry_price) * position.size,
                    SignalType::Sell => (position.entry_price - price) * position.size,
                    _ => 0.0,
                };
                unrealized_pnl += pnl;
            }
        }

        self.state.current_equity = self.state.capital + unrealized_pnl;
    }

    /// Close all open positions at given prices
    pub fn close_all_positions(&mut self, prices: &HashMap<String, f64>) {
        let assets: Vec<String> = self.state.open_positions.keys().cloned().collect();

        for asset in assets {
            if let Some(&price) = prices.get(&asset) {
                self.close_position(&asset, price, TradeStatus::Closed);
            }
        }
    }

    /// Get the current simulation state
    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    /// Get configuration
    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    /// Calculate performance metrics
    pub fn calculate_metrics(&self) -> super::metrics::PerformanceMetrics {
        let calculator =
            MetricsCalculator::new(self.config.initial_capital, self.config.risk_free_rate);
        calculator.calculate(&self.state.closed_trades)
    }

    /// Calculate signal quality metrics
    pub fn calculate_signal_quality(&self) -> super::metrics::SignalQuality {
        let calculator =
            MetricsCalculator::new(self.config.initial_capital, self.config.risk_free_rate);
        calculator.calculate_signal_quality(&self.state.closed_trades, self.state.signals_processed)
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let metrics = self.calculate_metrics();
        format!(
            "Backtest Summary:\n\
             Initial Capital: ${:.2}\n\
             Final Capital: ${:.2}\n\
             Total Return: {:.2}%\n\
             Total Trades: {}\n\
             Win Rate: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             Max Drawdown: {:.2}%\n\
             Signals Processed: {}\n\
             Signals Traded: {} ({:.2}%)",
            self.config.initial_capital,
            self.state.capital,
            metrics.total_return_pct,
            metrics.total_trades,
            metrics.win_rate * 100.0,
            metrics.sharpe_ratio,
            metrics.max_drawdown_pct,
            self.state.signals_processed,
            self.state.signals_traded,
            if self.state.signals_processed > 0 {
                (self.state.signals_traded as f64 / self.state.signals_processed as f64) * 100.0
            } else {
                0.0
            }
        )
    }

    /// Reset the simulation to initial state
    pub fn reset(&mut self) {
        let initial_capital = self.config.initial_capital;
        self.state = SimulationState {
            capital: initial_capital,
            peak_capital: initial_capital,
            current_equity: initial_capital,
            ..Default::default()
        };
        self.next_trade_id = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signal(signal_type: SignalType, asset: &str, confidence: f64) -> TradingSignal {
        TradingSignal::new(signal_type, confidence, asset.to_string())
    }

    #[test]
    fn test_simulation_creation() {
        let config = SimulationConfig::default();
        let sim = BacktestSimulation::new(config.clone());

        assert_eq!(sim.state.capital, config.initial_capital);
        assert_eq!(sim.state.open_positions.len(), 0);
        assert_eq!(sim.state.closed_trades.len(), 0);
    }

    #[test]
    fn test_position_sizing_fixed_percent() {
        let sizing = PositionSizing::FixedPercent(0.1);
        let size = sizing.calculate_size(10000.0, 50000.0, None);

        // 10% of 10000 = 1000, divided by price 50000 = 0.02
        assert!((size - 0.02).abs() < 0.0001);
    }

    #[test]
    fn test_position_sizing_fixed_dollar() {
        let sizing = PositionSizing::FixedDollar(1000.0);
        let size = sizing.calculate_size(10000.0, 50000.0, None);

        // 1000 / 50000 = 0.02
        assert!((size - 0.02).abs() < 0.0001);
    }

    #[test]
    fn test_position_sizing_kelly() {
        let sizing = PositionSizing::Kelly {
            win_rate: 0.6,
            avg_win_loss_ratio: 1.5,
        };
        let size = sizing.calculate_size(10000.0, 50000.0, None);

        // Kelly fraction = (0.6 * 1.5 - 0.4) / 1.5 = 0.333...
        // But capped at 0.25 (25%)
        // Position value = 10000 * 0.25 = 2500
        // Size = 2500 / 50000 = 0.05
        assert!(size > 0.0);
        assert!(size <= 0.05);
    }

    #[test]
    fn test_open_long_position() {
        let config = SimulationConfig::default();
        let mut sim = BacktestSimulation::new(config);

        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.8);
        sim.process_signal(&signal, 50000.0);

        assert_eq!(sim.state.open_positions.len(), 1);
        assert!(sim.state.open_positions.contains_key("BTCUSD"));
        assert_eq!(sim.state.signals_traded, 1);
    }

    #[test]
    fn test_open_short_position() {
        let config = SimulationConfig::default();
        let mut sim = BacktestSimulation::new(config);

        let signal = create_test_signal(SignalType::Sell, "BTCUSD", 0.8);
        sim.process_signal(&signal, 50000.0);

        assert_eq!(sim.state.open_positions.len(), 1);
        let position = sim.state.open_positions.get("BTCUSD").unwrap();
        assert_eq!(position.side, SignalType::Sell);
    }

    #[test]
    fn test_close_position_on_signal() {
        let config = SimulationConfig::default();
        let mut sim = BacktestSimulation::new(config);

        // Open position
        let buy_signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.8);
        sim.process_signal(&buy_signal, 50000.0);
        assert_eq!(sim.state.open_positions.len(), 1);

        // Process close signal - this will close the position
        let close_signal = create_test_signal(SignalType::Close, "BTCUSD", 0.8);
        sim.process_signal(&close_signal, 51000.0);

        // Verify position was closed
        assert_eq!(sim.state.open_positions.len(), 0);
        assert_eq!(sim.state.closed_trades.len(), 1);
    }

    #[test]
    fn test_reverse_position() {
        let config = SimulationConfig::default();
        let mut sim = BacktestSimulation::new(config);

        // Open long
        let buy_signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.8);
        sim.process_signal(&buy_signal, 50000.0);

        // Reverse to short
        let sell_signal = create_test_signal(SignalType::Sell, "BTCUSD", 0.85);
        sim.process_signal(&sell_signal, 51000.0);

        // Should have closed the long
        assert_eq!(sim.state.closed_trades.len(), 1);
        assert_eq!(sim.state.closed_trades[0].status, TradeStatus::Reversed);
    }

    #[test]
    fn test_min_confidence_filter() {
        let config = SimulationConfig {
            min_confidence: 0.7,
            ..Default::default()
        };
        let mut sim = BacktestSimulation::new(config);

        // Low confidence signal - should be ignored
        let weak_signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.6);
        sim.process_signal(&weak_signal, 50000.0);
        assert_eq!(sim.state.open_positions.len(), 0);

        // High confidence signal - should be accepted
        let strong_signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.8);
        sim.process_signal(&strong_signal, 50000.0);
        assert_eq!(sim.state.open_positions.len(), 1);
    }

    #[test]
    fn test_max_positions_limit() {
        let config = SimulationConfig {
            max_positions: 1,
            ..Default::default()
        };
        let mut sim = BacktestSimulation::new(config);

        // Open first position
        let signal1 = create_test_signal(SignalType::Buy, "BTCUSD", 0.8);
        sim.process_signal(&signal1, 50000.0);
        assert_eq!(sim.state.open_positions.len(), 1);

        // Try to open second - should be rejected
        let signal2 = create_test_signal(SignalType::Buy, "ETHUSDT", 0.8);
        sim.process_signal(&signal2, 3000.0);
        assert_eq!(sim.state.open_positions.len(), 1);
    }

    #[test]
    fn test_stop_loss() {
        let config = SimulationConfig {
            use_stop_loss: true,
            stop_loss_pct: 0.02,
            ..Default::default()
        };
        let mut sim = BacktestSimulation::new(config);

        // Open long position at 50000
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.8);
        sim.process_signal(&signal, 50000.0);

        let position = sim.state.open_positions.get("BTCUSD").unwrap();
        assert!(position.stop_loss.is_some());

        // Stop should be 2% below entry price (which includes slippage)
        // Entry price = 50000 + (50000 * 0.0005) = 50025
        // Stop = 50025 * (1.0 - 0.02) = 49024.5
        let entry_with_slippage = 50000.0 + (50000.0 * sim.config.slippage_rate);
        let expected_stop = entry_with_slippage * (1.0 - 0.02);
        assert!((position.stop_loss.unwrap() - expected_stop).abs() < 1.0);
    }

    #[test]
    fn test_metrics_calculation() {
        let config = SimulationConfig::default();
        let mut sim = BacktestSimulation::new(config);

        // Execute some trades - open and close
        let buy = create_test_signal(SignalType::Buy, "BTCUSD", 0.8);
        sim.process_signal(&buy, 50000.0);

        // Verify position opened
        assert_eq!(sim.state.open_positions.len(), 1);

        let close = create_test_signal(SignalType::Close, "BTCUSD", 0.8);
        sim.process_signal(&close, 51000.0);

        // Verify position closed
        assert_eq!(sim.state.open_positions.len(), 0);

        let metrics = sim.calculate_metrics();
        assert_eq!(metrics.total_trades, 1);
        assert!(metrics.total_pnl != 0.0);
    }

    #[test]
    fn test_reset() {
        let config = SimulationConfig::default();
        let mut sim = BacktestSimulation::new(config.clone());

        // Make some trades
        let signal = create_test_signal(SignalType::Buy, "BTCUSD", 0.8);
        sim.process_signal(&signal, 50000.0);

        // Reset
        sim.reset();

        assert_eq!(sim.state.capital, config.initial_capital);
        assert_eq!(sim.state.open_positions.len(), 0);
        assert_eq!(sim.state.closed_trades.len(), 0);
        assert_eq!(sim.state.signals_processed, 0);
    }
}
