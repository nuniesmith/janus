//! # Risk Metrics Module
//!
//! Calculate risk and performance metrics for trading signals and portfolios:
//! - Portfolio risk (heat, exposure, concentration)
//! - Sharpe ratio estimation
//! - Maximum drawdown tracking
//! - Expected value calculations
//! - Win rate and profit factor analysis

use crate::risk::PortfolioState;
use serde::{Deserialize, Serialize};

/// Portfolio risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioRisk {
    /// Total portfolio value at risk (sum of all position risks)
    pub total_risk: f64,

    /// Portfolio heat (total risk as % of account)
    pub portfolio_heat: f64,

    /// Total portfolio exposure (sum of all position values)
    pub total_exposure: f64,

    /// Exposure as percentage of account
    pub exposure_percentage: f64,

    /// Number of open positions
    pub position_count: usize,

    /// Average risk per position
    pub avg_risk_per_position: f64,

    /// Maximum single position risk
    pub max_position_risk: f64,

    /// Concentration risk (largest position as % of portfolio)
    pub concentration_risk: f64,

    /// Diversification score (0.0 to 1.0, higher is more diversified)
    pub diversification_score: f64,
}

impl PortfolioRisk {
    /// Calculate portfolio risk from current portfolio state
    pub fn calculate(portfolio: &PortfolioState, account_balance: f64) -> Self {
        if portfolio.positions.is_empty() {
            return Self::empty();
        }

        let position_count = portfolio.positions.len();
        let mut total_risk = 0.0;
        let mut max_position_risk: f64 = 0.0;
        let mut max_position_value: f64 = 0.0;
        let total_exposure = portfolio.total_exposure();

        // Calculate risks
        for position in portfolio.positions.values() {
            if let Some(risk) = position.risk_amount() {
                total_risk += risk;
                max_position_risk = max_position_risk.max(risk);
            }

            let position_value = position.position_value();
            max_position_value = max_position_value.max(position_value);
        }

        let portfolio_heat = total_risk / account_balance;
        let exposure_percentage = total_exposure / account_balance;
        let avg_risk_per_position = total_risk / position_count as f64;
        let concentration_risk = max_position_value / total_exposure;

        // Diversification: inverse of concentration, adjusted for count
        // Perfect diversification would be equal-weighted positions
        let ideal_position_size = total_exposure / position_count as f64;
        let mut size_variance = 0.0;
        for position in portfolio.positions.values() {
            let diff = position.position_value() - ideal_position_size;
            size_variance += diff * diff;
        }
        let size_std_dev = (size_variance / position_count as f64).sqrt();
        let diversification_score = 1.0 - (size_std_dev / ideal_position_size).min(1.0);

        Self {
            total_risk,
            portfolio_heat,
            total_exposure,
            exposure_percentage,
            position_count,
            avg_risk_per_position,
            max_position_risk,
            concentration_risk,
            diversification_score,
        }
    }

    /// Create empty portfolio risk (no positions)
    pub fn empty() -> Self {
        Self {
            total_risk: 0.0,
            portfolio_heat: 0.0,
            total_exposure: 0.0,
            exposure_percentage: 0.0,
            position_count: 0,
            avg_risk_per_position: 0.0,
            max_position_risk: 0.0,
            concentration_risk: 0.0,
            diversification_score: 1.0,
        }
    }

    /// Check if portfolio is within healthy risk parameters
    pub fn is_healthy(&self, max_heat: f64, max_exposure: f64) -> bool {
        self.portfolio_heat <= max_heat && self.exposure_percentage <= max_exposure
    }
}

/// Trade performance metrics tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of trades
    pub total_trades: usize,

    /// Number of winning trades
    pub winning_trades: usize,

    /// Number of losing trades
    pub losing_trades: usize,

    /// Win rate (percentage)
    pub win_rate: f64,

    /// Total profit from all trades
    pub total_profit: f64,

    /// Total loss from all trades
    pub total_loss: f64,

    /// Average win
    pub avg_win: f64,

    /// Average loss
    pub avg_loss: f64,

    /// Profit factor (total profit / total loss)
    pub profit_factor: f64,

    /// Largest win
    pub largest_win: f64,

    /// Largest loss
    pub largest_loss: f64,

    /// Average R multiple (avg win / avg loss)
    pub avg_r_multiple: f64,

    /// Expected value per trade
    pub expected_value: f64,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            total_profit: 0.0,
            total_loss: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            avg_r_multiple: 0.0,
            expected_value: 0.0,
        }
    }

    /// Add a trade result
    pub fn add_trade(&mut self, pnl: f64) {
        self.total_trades += 1;

        if pnl > 0.0 {
            self.winning_trades += 1;
            self.total_profit += pnl;
            self.largest_win = self.largest_win.max(pnl);
        } else if pnl < 0.0 {
            self.losing_trades += 1;
            self.total_loss += pnl.abs();
            self.largest_loss = self.largest_loss.max(pnl.abs());
        }

        self.recalculate();
    }

    /// Recalculate derived metrics
    fn recalculate(&mut self) {
        self.win_rate = if self.total_trades > 0 {
            self.winning_trades as f64 / self.total_trades as f64
        } else {
            0.0
        };

        self.avg_win = if self.winning_trades > 0 {
            self.total_profit / self.winning_trades as f64
        } else {
            0.0
        };

        self.avg_loss = if self.losing_trades > 0 {
            self.total_loss / self.losing_trades as f64
        } else {
            0.0
        };

        self.profit_factor = if self.total_loss > 0.0 {
            self.total_profit / self.total_loss
        } else {
            0.0
        };

        self.avg_r_multiple = if self.avg_loss > 0.0 {
            self.avg_win / self.avg_loss
        } else {
            0.0
        };

        // Expected value = (win_rate * avg_win) - (loss_rate * avg_loss)
        let loss_rate = 1.0 - self.win_rate;
        self.expected_value = (self.win_rate * self.avg_win) - (loss_rate * self.avg_loss);
    }

    /// Calculate Kelly criterion optimal bet size
    pub fn kelly_fraction(&self) -> f64 {
        if self.avg_loss > 0.0 && self.win_rate > 0.0 && self.win_rate < 1.0 {
            let b = self.avg_win / self.avg_loss;
            let p = self.win_rate;
            let q = 1.0 - p;
            ((b * p) - q) / b
        } else {
            0.0
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Drawdown tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownTracker {
    /// Peak equity value
    pub peak_equity: f64,

    /// Current equity value
    pub current_equity: f64,

    /// Current drawdown (peak - current)
    pub current_drawdown: f64,

    /// Current drawdown percentage
    pub current_drawdown_pct: f64,

    /// Maximum drawdown ever experienced
    pub max_drawdown: f64,

    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,

    /// Number of periods in current drawdown
    pub drawdown_duration: usize,

    /// Longest drawdown duration
    pub max_drawdown_duration: usize,
}

impl DrawdownTracker {
    /// Create new drawdown tracker
    pub fn new(initial_equity: f64) -> Self {
        Self {
            peak_equity: initial_equity,
            current_equity: initial_equity,
            current_drawdown: 0.0,
            current_drawdown_pct: 0.0,
            max_drawdown: 0.0,
            max_drawdown_pct: 0.0,
            drawdown_duration: 0,
            max_drawdown_duration: 0,
        }
    }

    /// Update with new equity value
    pub fn update(&mut self, equity: f64) {
        self.current_equity = equity;

        // Update peak if we have a new high
        if equity > self.peak_equity {
            self.peak_equity = equity;
            self.drawdown_duration = 0;
        } else {
            self.drawdown_duration += 1;
        }

        // Calculate current drawdown
        self.current_drawdown = self.peak_equity - equity;
        self.current_drawdown_pct = if self.peak_equity > 0.0 {
            self.current_drawdown / self.peak_equity
        } else {
            0.0
        };

        // Update max drawdown
        if self.current_drawdown > self.max_drawdown {
            self.max_drawdown = self.current_drawdown;
            self.max_drawdown_pct = self.current_drawdown_pct;
        }

        // Update max duration
        if self.drawdown_duration > self.max_drawdown_duration {
            self.max_drawdown_duration = self.drawdown_duration;
        }
    }

    /// Check if in drawdown
    pub fn is_in_drawdown(&self) -> bool {
        self.current_drawdown > 0.0
    }

    /// Get recovery percentage needed to reach peak
    pub fn recovery_needed_pct(&self) -> f64 {
        if self.current_equity > 0.0 {
            (self.peak_equity - self.current_equity) / self.current_equity
        } else {
            0.0
        }
    }
}

/// Sharpe ratio calculator
pub struct SharpeCalculator {
    returns: Vec<f64>,
    risk_free_rate: f64,
}

impl SharpeCalculator {
    /// Create new Sharpe calculator
    pub fn new(risk_free_rate: f64) -> Self {
        Self {
            returns: Vec::new(),
            risk_free_rate,
        }
    }

    /// Add a return observation
    pub fn add_return(&mut self, return_value: f64) {
        self.returns.push(return_value);
    }

    /// Calculate Sharpe ratio
    pub fn calculate(&self) -> Option<f64> {
        if self.returns.len() < 2 {
            return None;
        }

        let mean = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (self.returns.len() - 1) as f64;

        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            Some((mean - self.risk_free_rate) / std_dev)
        } else {
            None
        }
    }

    /// Calculate annualized Sharpe ratio
    pub fn calculate_annualized(&self, periods_per_year: f64) -> Option<f64> {
        self.calculate()
            .map(|sharpe| sharpe * periods_per_year.sqrt())
    }
}

/// Comprehensive risk metrics aggregator
pub struct RiskMetrics;

impl RiskMetrics {
    /// Calculate portfolio risk metrics
    pub fn calculate_portfolio_risk(
        portfolio: &PortfolioState,
        account_balance: f64,
    ) -> PortfolioRisk {
        PortfolioRisk::calculate(portfolio, account_balance)
    }

    /// Calculate expected value for a signal
    pub fn calculate_expected_value(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
        let loss_rate = 1.0 - win_rate;
        (win_rate * avg_win) - (loss_rate * avg_loss)
    }

    /// Calculate risk-adjusted return
    pub fn calculate_risk_adjusted_return(return_value: f64, risk_amount: f64) -> f64 {
        if risk_amount > 0.0 {
            return_value / risk_amount
        } else {
            0.0
        }
    }

    /// Calculate position correlation (simplified)
    pub fn calculate_correlation(returns1: &[f64], returns2: &[f64]) -> Option<f64> {
        if returns1.len() != returns2.len() || returns1.len() < 2 {
            return None;
        }

        let n = returns1.len() as f64;
        let mean1 = returns1.iter().sum::<f64>() / n;
        let mean2 = returns2.iter().sum::<f64>() / n;

        let mut covariance = 0.0;
        let mut variance1 = 0.0;
        let mut variance2 = 0.0;

        for i in 0..returns1.len() {
            let diff1 = returns1[i] - mean1;
            let diff2 = returns2[i] - mean2;
            covariance += diff1 * diff2;
            variance1 += diff1 * diff1;
            variance2 += diff2 * diff2;
        }

        let std1 = (variance1 / n).sqrt();
        let std2 = (variance2 / n).sqrt();

        if std1 > 0.0 && std2 > 0.0 {
            Some((covariance / n) / (std1 * std2))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::{Position, PositionSide};

    #[test]
    fn test_portfolio_risk_empty() {
        let portfolio = PortfolioState::new(10000.0);
        let risk = PortfolioRisk::calculate(&portfolio, 10000.0);

        assert_eq!(risk.position_count, 0);
        assert_eq!(risk.total_risk, 0.0);
        assert_eq!(risk.portfolio_heat, 0.0);
    }

    #[test]
    fn test_portfolio_risk_single_position() {
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

        let risk = PortfolioRisk::calculate(&portfolio, 10000.0);

        assert_eq!(risk.position_count, 1);
        assert_eq!(risk.total_risk, 100.0); // (50000 - 49000) * 0.1
        assert_eq!(risk.portfolio_heat, 0.01); // 100 / 10000
        assert_eq!(risk.total_exposure, 5000.0); // 50000 * 0.1
    }

    #[test]
    fn test_portfolio_risk_multiple_positions() {
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

        portfolio.add_position(
            "ETH/USD".to_string(),
            Position {
                symbol: "ETH/USD".to_string(),
                entry_price: 3000.0,
                quantity: 1.0,
                side: PositionSide::Long,
                stop_loss: Some(2900.0),
                take_profit: Some(3200.0),
            },
        );

        let risk = PortfolioRisk::calculate(&portfolio, 10000.0);

        assert_eq!(risk.position_count, 2);
        assert_eq!(risk.total_risk, 200.0); // 100 + 100
        assert_eq!(risk.portfolio_heat, 0.02);
    }

    #[test]
    fn test_performance_metrics_winning_trades() {
        let mut metrics = PerformanceMetrics::new();

        metrics.add_trade(100.0);
        metrics.add_trade(150.0);
        metrics.add_trade(80.0);

        assert_eq!(metrics.total_trades, 3);
        assert_eq!(metrics.winning_trades, 3);
        assert_eq!(metrics.win_rate, 1.0);
        assert_eq!(metrics.total_profit, 330.0);
        assert_eq!(metrics.avg_win, 110.0);
    }

    #[test]
    fn test_performance_metrics_mixed_trades() {
        let mut metrics = PerformanceMetrics::new();

        metrics.add_trade(200.0); // Win
        metrics.add_trade(-100.0); // Loss
        metrics.add_trade(150.0); // Win
        metrics.add_trade(-50.0); // Loss

        assert_eq!(metrics.total_trades, 4);
        assert_eq!(metrics.winning_trades, 2);
        assert_eq!(metrics.losing_trades, 2);
        assert_eq!(metrics.win_rate, 0.5);
        assert_eq!(metrics.total_profit, 350.0);
        assert_eq!(metrics.total_loss, 150.0);
        assert!((metrics.profit_factor - 2.333).abs() < 0.01);
    }

    #[test]
    fn test_performance_metrics_expected_value() {
        let mut metrics = PerformanceMetrics::new();

        // Win rate 60%, avg win $200, avg loss $100
        metrics.add_trade(200.0);
        metrics.add_trade(200.0);
        metrics.add_trade(200.0);
        metrics.add_trade(-100.0);
        metrics.add_trade(-100.0);

        // EV = (0.6 * 200) - (0.4 * 100) = 120 - 40 = 80
        assert!((metrics.expected_value - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_drawdown_tracker() {
        let mut tracker = DrawdownTracker::new(10000.0);

        // New peak
        tracker.update(11000.0);
        assert_eq!(tracker.peak_equity, 11000.0);
        assert_eq!(tracker.current_drawdown, 0.0);

        // Drawdown begins
        tracker.update(10500.0);
        assert_eq!(tracker.current_drawdown, 500.0);
        assert!((tracker.current_drawdown_pct - 0.04545).abs() < 0.001);

        // Deeper drawdown
        tracker.update(9500.0);
        assert_eq!(tracker.current_drawdown, 1500.0);
        assert_eq!(tracker.max_drawdown, 1500.0);
        assert!(tracker.is_in_drawdown());

        // Recovery to new peak
        tracker.update(12000.0);
        assert_eq!(tracker.current_drawdown, 0.0);
        assert!(!tracker.is_in_drawdown());
        assert_eq!(tracker.max_drawdown, 1500.0); // Historical max preserved
    }

    #[test]
    fn test_sharpe_calculator() {
        let mut calc = SharpeCalculator::new(0.02); // 2% risk-free rate

        // Add some returns
        let returns = vec![0.05, 0.03, 0.08, 0.02, 0.06];
        for r in returns {
            calc.add_return(r);
        }

        let sharpe = calc.calculate();
        assert!(sharpe.is_some());
        assert!(sharpe.unwrap() > 0.0);
    }

    #[test]
    fn test_expected_value_calculation() {
        let ev = RiskMetrics::calculate_expected_value(0.6, 200.0, 100.0);
        assert!((ev - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_risk_adjusted_return() {
        let rar = RiskMetrics::calculate_risk_adjusted_return(200.0, 100.0);
        assert_eq!(rar, 2.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let returns1 = vec![0.01, 0.02, -0.01, 0.03, 0.02];
        let returns2 = vec![0.01, 0.015, -0.005, 0.025, 0.02];

        let corr = RiskMetrics::calculate_correlation(&returns1, &returns2);
        assert!(corr.is_some());
        let c = corr.unwrap();
        assert!(c > 0.8); // Should be highly correlated
        assert!(c <= 1.0);
    }

    #[test]
    fn test_kelly_fraction() {
        let mut metrics = PerformanceMetrics::new();

        // Win rate 60%, avg win $200, avg loss $100
        metrics.add_trade(200.0);
        metrics.add_trade(200.0);
        metrics.add_trade(200.0);
        metrics.add_trade(-100.0);
        metrics.add_trade(-100.0);

        let kelly = metrics.kelly_fraction();
        // Kelly = (0.6 * 2.0 - 0.4) / 2.0 = 0.4
        assert!((kelly - 0.4).abs() < 0.01);
    }
}
