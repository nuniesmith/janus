//! Almgren-Chriss Optimal Execution Model
//!
//! Implementation of the Almgren-Chriss framework for optimal trade execution.
//! This model minimizes a combination of execution cost and risk by finding
//! the optimal trading trajectory.
//!
//! # Key Concepts
//!
//! - **Temporary Impact**: Price impact that decays quickly (within a trade)
//! - **Permanent Impact**: Lasting price impact from information leakage
//! - **Execution Risk**: Variance of execution cost due to price volatility
//! - **Risk Aversion**: Trade-off parameter between cost and risk
//!
//! # Reference
//!
//! Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions."
//! Journal of Risk, 3, 5-40.
//!
//! # Example
//!
//! ```rust,ignore
//! use cerebellum::impact::AlmgrenChriss;
//!
//! let model = AlmgrenChriss::new(AlmgrenChrissConfig {
//!     total_shares: 100_000.0,
//!     time_horizon: 1.0,  // 1 day
//!     num_intervals: 20,
//!     volatility: 0.02,   // 2% daily vol
//!     permanent_impact: 0.1,
//!     temporary_impact: 0.01,
//!     risk_aversion: 1e-6,
//!     ..Default::default()
//! });
//!
//! let trajectory = model.optimal_trajectory();
//! ```

use crate::common::Result;
use serde::{Deserialize, Serialize};

/// Configuration for Almgren-Chriss model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlmgrenChrissConfig {
    /// Total number of shares to execute
    pub total_shares: f64,

    /// Time horizon for execution (in trading days)
    pub time_horizon: f64,

    /// Number of trading intervals
    pub num_intervals: usize,

    /// Daily volatility (sigma)
    pub volatility: f64,

    /// Permanent impact coefficient (gamma)
    /// Price impact per share that persists
    pub permanent_impact: f64,

    /// Temporary impact coefficient (eta)
    /// Price impact per share/time that decays
    pub temporary_impact: f64,

    /// Risk aversion parameter (lambda)
    /// Higher = more risk averse = faster execution
    pub risk_aversion: f64,

    /// Initial stock price
    pub initial_price: f64,

    /// Bid-ask spread (as fraction of price)
    pub spread: f64,

    /// Fixed cost per trade
    pub fixed_cost: f64,

    /// Daily trading volume (for volume participation limits)
    pub daily_volume: f64,

    /// Maximum participation rate (fraction of volume)
    pub max_participation: f64,
}

impl Default for AlmgrenChrissConfig {
    fn default() -> Self {
        Self {
            total_shares: 10_000.0,
            time_horizon: 1.0,
            num_intervals: 10,
            volatility: 0.02,
            permanent_impact: 0.1,
            temporary_impact: 0.01,
            risk_aversion: 1e-6,
            initial_price: 100.0,
            spread: 0.001,
            fixed_cost: 0.0,
            daily_volume: 1_000_000.0,
            max_participation: 0.1,
        }
    }
}

/// Trading trajectory - the optimal execution schedule
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionTrajectory {
    /// Time points (as fraction of horizon)
    pub times: Vec<f64>,

    /// Shares remaining at each time point
    pub holdings: Vec<f64>,

    /// Trade sizes at each interval
    pub trades: Vec<f64>,

    /// Trading rate at each interval (shares per unit time)
    pub trading_rates: Vec<f64>,

    /// Expected cost at each step
    pub expected_costs: Vec<f64>,

    /// Cumulative expected cost
    pub total_expected_cost: f64,

    /// Execution risk (variance of cost)
    pub execution_risk: f64,

    /// Expected shortfall (mean + lambda * variance)
    pub expected_shortfall: f64,
}

/// Cost breakdown for analysis
#[derive(Debug, Clone, Serialize)]
pub struct CostBreakdown {
    /// Permanent impact cost
    pub permanent_cost: f64,

    /// Temporary impact cost
    pub temporary_cost: f64,

    /// Spread cost
    pub spread_cost: f64,

    /// Fixed costs
    pub fixed_cost: f64,

    /// Total expected cost
    pub total_cost: f64,

    /// Risk (variance)
    pub risk: f64,

    /// Utility (cost + lambda * risk)
    pub utility: f64,
}

/// Almgren-Chriss optimal execution model
#[derive(Debug, Clone)]
pub struct AlmgrenChriss {
    /// Model configuration
    config: AlmgrenChrissConfig,

    /// Time step size
    tau: f64,

    /// Computed kappa (urgency parameter)
    kappa: f64,

    /// Computed optimal trajectory
    trajectory: Option<ExecutionTrajectory>,
}

impl Default for AlmgrenChriss {
    fn default() -> Self {
        Self::new()
    }
}

impl AlmgrenChriss {
    /// Create a new Almgren-Chriss model with default configuration
    pub fn new() -> Self {
        Self::with_config(AlmgrenChrissConfig::default())
    }

    /// Create a new model with custom configuration
    pub fn with_config(config: AlmgrenChrissConfig) -> Self {
        let tau = config.time_horizon / config.num_intervals as f64;
        let kappa = Self::compute_kappa(&config, tau);

        let mut model = Self {
            config,
            tau,
            kappa,
            trajectory: None,
        };

        // Pre-compute optimal trajectory
        model.trajectory = Some(model.compute_trajectory());
        model
    }

    /// Compute the kappa (urgency) parameter
    ///
    /// kappa = sqrt(lambda * sigma^2 / eta)
    fn compute_kappa(config: &AlmgrenChrissConfig, tau: f64) -> f64 {
        let eta_tilde = config.temporary_impact / tau;
        if eta_tilde <= 0.0 || config.volatility <= 0.0 {
            return 0.0;
        }

        (config.risk_aversion * config.volatility.powi(2) / eta_tilde).sqrt()
    }

    /// Compute the optimal trading trajectory
    fn compute_trajectory(&self) -> ExecutionTrajectory {
        let n = self.config.num_intervals;
        let x0 = self.config.total_shares;
        let t_total = self.config.time_horizon;

        let mut times = Vec::with_capacity(n + 1);
        let mut holdings = Vec::with_capacity(n + 1);
        let mut trades = Vec::with_capacity(n);
        let mut trading_rates = Vec::with_capacity(n);
        let mut expected_costs = Vec::with_capacity(n);

        // Time grid
        for i in 0..=n {
            times.push(i as f64 * self.tau);
        }

        // Compute holdings using the analytical solution
        // x_j = x_0 * sinh(kappa * (T - t_j)) / sinh(kappa * T)
        if self.kappa.abs() < 1e-10 {
            // Linear trajectory when kappa -> 0
            for i in 0..=n {
                let frac = 1.0 - (i as f64 / n as f64);
                holdings.push(x0 * frac);
            }
        } else {
            let sinh_kt = (self.kappa * t_total).sinh();
            for i in 0..=n {
                let t_j = times[i];
                let remaining_time = t_total - t_j;
                let x_j = x0 * (self.kappa * remaining_time).sinh() / sinh_kt;
                holdings.push(x_j);
            }
        }

        // Compute trades and costs
        let mut total_cost = 0.0;
        let mut total_risk = 0.0;

        for i in 0..n {
            // Trade size
            let n_i = holdings[i] - holdings[i + 1];
            trades.push(n_i);

            // Trading rate
            let rate = n_i / self.tau;
            trading_rates.push(rate);

            // Expected cost for this interval
            // Temporary impact: eta * n_i^2 / tau
            // Permanent impact: gamma * n_i * x_i
            let temp_cost = self.config.temporary_impact * n_i.powi(2) / self.tau;
            let perm_cost = self.config.permanent_impact * n_i * holdings[i];
            let spread_cost = 0.5 * self.config.spread * self.config.initial_price * n_i.abs();

            let interval_cost = temp_cost + perm_cost + spread_cost;
            expected_costs.push(interval_cost);
            total_cost += interval_cost;

            // Risk contribution: sigma^2 * tau * x_i^2
            total_risk += self.config.volatility.powi(2) * self.tau * holdings[i + 1].powi(2);
        }

        // Add fixed costs
        total_cost += self.config.fixed_cost * n as f64;

        let expected_shortfall = total_cost + self.config.risk_aversion * total_risk;

        ExecutionTrajectory {
            times,
            holdings,
            trades,
            trading_rates,
            expected_costs,
            total_expected_cost: total_cost,
            execution_risk: total_risk,
            expected_shortfall,
        }
    }

    /// Get the optimal trading trajectory
    pub fn optimal_trajectory(&self) -> ExecutionTrajectory {
        self.trajectory
            .clone()
            .unwrap_or_else(|| self.compute_trajectory())
    }

    /// Get the trade size for a specific interval
    pub fn trade_at_interval(&self, interval: usize) -> Option<f64> {
        self.trajectory
            .as_ref()
            .and_then(|t| t.trades.get(interval).copied())
    }

    /// Get holdings at a specific time
    pub fn holdings_at_time(&self, time: f64) -> f64 {
        let x0 = self.config.total_shares;
        let t_total = self.config.time_horizon;

        if time <= 0.0 {
            return x0;
        }
        if time >= t_total {
            return 0.0;
        }

        if self.kappa.abs() < 1e-10 {
            // Linear interpolation
            x0 * (1.0 - time / t_total)
        } else {
            let sinh_kt = (self.kappa * t_total).sinh();
            x0 * (self.kappa * (t_total - time)).sinh() / sinh_kt
        }
    }

    /// Get the trading rate at a specific time
    pub fn trading_rate_at_time(&self, time: f64) -> f64 {
        let x0 = self.config.total_shares;
        let t_total = self.config.time_horizon;

        if time < 0.0 || time > t_total {
            return 0.0;
        }

        if self.kappa.abs() < 1e-10 {
            // Constant rate (TWAP)
            x0 / t_total
        } else {
            let sinh_kt = (self.kappa * t_total).sinh();
            x0 * self.kappa * (self.kappa * (t_total - time)).cosh() / sinh_kt
        }
    }

    /// Compute detailed cost breakdown
    pub fn cost_breakdown(&self) -> CostBreakdown {
        let traj = self.optimal_trajectory();

        let mut permanent_cost = 0.0;
        let mut temporary_cost = 0.0;
        let mut spread_cost = 0.0;

        for i in 0..self.config.num_intervals {
            let n_i = traj.trades[i];
            let x_i = traj.holdings[i];

            permanent_cost += self.config.permanent_impact * n_i * x_i;
            temporary_cost += self.config.temporary_impact * n_i.powi(2) / self.tau;
            spread_cost += 0.5 * self.config.spread * self.config.initial_price * n_i.abs();
        }

        let fixed_cost = self.config.fixed_cost * self.config.num_intervals as f64;
        let total_cost = permanent_cost + temporary_cost + spread_cost + fixed_cost;
        let utility = total_cost + self.config.risk_aversion * traj.execution_risk;

        CostBreakdown {
            permanent_cost,
            temporary_cost,
            spread_cost,
            fixed_cost,
            total_cost,
            risk: traj.execution_risk,
            utility,
        }
    }

    /// Compute the efficient frontier (cost vs risk trade-off)
    pub fn efficient_frontier(&self, num_points: usize) -> Vec<(f64, f64)> {
        let mut frontier = Vec::with_capacity(num_points);

        // Vary risk aversion from very low to very high
        let lambda_min: f64 = 1e-10;
        let lambda_max: f64 = 1e-3;

        for i in 0..num_points {
            let frac = i as f64 / (num_points - 1) as f64;
            let lambda = lambda_min * (lambda_max / lambda_min).powf(frac);

            let mut config = self.config.clone();
            config.risk_aversion = lambda;

            let model = AlmgrenChriss::with_config(config);
            let traj = model.optimal_trajectory();

            frontier.push((traj.execution_risk, traj.total_expected_cost));
        }

        frontier
    }

    /// Compute optimal trajectory with volume constraints
    pub fn constrained_trajectory(&self) -> ExecutionTrajectory {
        let mut traj = self.compute_trajectory();

        // Apply participation rate constraints
        let volume_per_interval = self.config.daily_volume * self.tau;
        let max_trade = volume_per_interval * self.config.max_participation;

        let mut constrained = false;
        for i in 0..traj.trades.len() {
            if traj.trades[i].abs() > max_trade {
                constrained = true;
                let sign = traj.trades[i].signum();
                traj.trades[i] = sign * max_trade;
                traj.trading_rates[i] = traj.trades[i] / self.tau;
            }
        }

        // If constrained, recalculate holdings and costs
        if constrained {
            let mut remaining = self.config.total_shares;
            for i in 0..traj.trades.len() {
                traj.holdings[i] = remaining;
                remaining -= traj.trades[i];
            }
            traj.holdings[traj.trades.len()] = remaining.max(0.0);

            // Recalculate costs
            traj.total_expected_cost = 0.0;
            traj.execution_risk = 0.0;

            for i in 0..traj.trades.len() {
                let n_i = traj.trades[i];
                let x_i = traj.holdings[i];

                let temp_cost = self.config.temporary_impact * n_i.powi(2) / self.tau;
                let perm_cost = self.config.permanent_impact * n_i * x_i;
                let spread_cost = 0.5 * self.config.spread * self.config.initial_price * n_i.abs();

                traj.expected_costs[i] = temp_cost + perm_cost + spread_cost;
                traj.total_expected_cost += traj.expected_costs[i];

                traj.execution_risk +=
                    self.config.volatility.powi(2) * self.tau * traj.holdings[i + 1].powi(2);
            }

            traj.total_expected_cost += self.config.fixed_cost * traj.trades.len() as f64;
            traj.expected_shortfall =
                traj.total_expected_cost + self.config.risk_aversion * traj.execution_risk;
        }

        traj
    }

    /// Adaptive execution - update trajectory based on current state
    pub fn adaptive_update(
        &self,
        current_time: f64,
        current_holdings: f64,
        current_price: f64,
        realized_volatility: Option<f64>,
    ) -> ExecutionTrajectory {
        let remaining_time = (self.config.time_horizon - current_time).max(0.0);

        if remaining_time <= 0.0 || current_holdings <= 0.0 {
            return ExecutionTrajectory {
                times: vec![current_time],
                holdings: vec![0.0],
                trades: vec![current_holdings],
                trading_rates: vec![f64::INFINITY],
                expected_costs: vec![0.0],
                total_expected_cost: 0.0,
                execution_risk: 0.0,
                expected_shortfall: 0.0,
            };
        }

        // Update volatility if provided
        let vol = realized_volatility.unwrap_or(self.config.volatility);

        // Recalculate optimal trajectory from current state
        let remaining_intervals = ((remaining_time / self.tau).ceil() as usize).max(1);

        let new_config = AlmgrenChrissConfig {
            total_shares: current_holdings,
            time_horizon: remaining_time,
            num_intervals: remaining_intervals,
            volatility: vol,
            initial_price: current_price,
            ..self.config.clone()
        };

        let new_model = AlmgrenChriss::with_config(new_config);
        new_model.optimal_trajectory()
    }

    /// Get the urgency parameter (kappa)
    pub fn urgency(&self) -> f64 {
        self.kappa
    }

    /// Get configuration
    pub fn config(&self) -> &AlmgrenChrissConfig {
        &self.config
    }

    /// Estimate market impact for a given trade
    pub fn estimate_impact(&self, trade_size: f64, execution_time: f64) -> f64 {
        if execution_time <= 0.0 {
            return f64::INFINITY;
        }

        let rate = trade_size / execution_time;

        // Permanent impact
        let permanent = self.config.permanent_impact * trade_size;

        // Temporary impact (depends on rate)
        let temporary = self.config.temporary_impact * rate;

        // Total impact as fraction of price
        (permanent + temporary) / self.config.initial_price
    }

    /// Compare with TWAP strategy
    pub fn compare_with_twap(&self) -> TwapComparison {
        let ac_traj = self.optimal_trajectory();

        // TWAP: constant trading rate
        let n = self.config.num_intervals;
        let x0 = self.config.total_shares;
        let trade_per_interval = x0 / n as f64;

        let mut twap_cost = 0.0;
        let mut twap_risk = 0.0;

        for i in 0..n {
            let holdings_after = x0 - (i + 1) as f64 * trade_per_interval;
            let holdings_before = x0 - i as f64 * trade_per_interval;

            // Costs
            twap_cost += self.config.temporary_impact * trade_per_interval.powi(2) / self.tau;
            twap_cost += self.config.permanent_impact * trade_per_interval * holdings_before;
            twap_cost +=
                0.5 * self.config.spread * self.config.initial_price * trade_per_interval.abs();

            // Risk
            twap_risk += self.config.volatility.powi(2) * self.tau * holdings_after.powi(2);
        }

        TwapComparison {
            ac_cost: ac_traj.total_expected_cost,
            ac_risk: ac_traj.execution_risk,
            twap_cost,
            twap_risk,
            cost_savings: twap_cost - ac_traj.total_expected_cost,
            risk_reduction: twap_risk - ac_traj.execution_risk,
            cost_savings_pct: (twap_cost - ac_traj.total_expected_cost) / twap_cost * 100.0,
            risk_reduction_pct: (twap_risk - ac_traj.execution_risk) / twap_risk * 100.0,
        }
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Comparison with TWAP strategy
#[derive(Debug, Clone, Serialize)]
pub struct TwapComparison {
    /// Almgren-Chriss expected cost
    pub ac_cost: f64,
    /// Almgren-Chriss risk
    pub ac_risk: f64,
    /// TWAP expected cost
    pub twap_cost: f64,
    /// TWAP risk
    pub twap_risk: f64,
    /// Cost savings (TWAP - AC)
    pub cost_savings: f64,
    /// Risk reduction (TWAP - AC)
    pub risk_reduction: f64,
    /// Cost savings percentage
    pub cost_savings_pct: f64,
    /// Risk reduction percentage
    pub risk_reduction_pct: f64,
}

/// Extension for multi-asset execution
#[derive(Debug, Clone)]
pub struct MultiAssetExecution {
    /// Individual asset models
    models: Vec<AlmgrenChriss>,
    /// Correlation matrix
    correlations: Vec<Vec<f64>>,
}

impl MultiAssetExecution {
    /// Create a new multi-asset execution model
    pub fn new(configs: Vec<AlmgrenChrissConfig>, correlations: Vec<Vec<f64>>) -> Self {
        let models: Vec<AlmgrenChriss> = configs
            .into_iter()
            .map(AlmgrenChriss::with_config)
            .collect();

        Self {
            models,
            correlations,
        }
    }

    /// Get individual trajectories
    pub fn trajectories(&self) -> Vec<ExecutionTrajectory> {
        self.models.iter().map(|m| m.optimal_trajectory()).collect()
    }

    /// Compute portfolio-level risk
    pub fn portfolio_risk(&self) -> f64 {
        let trajs = self.trajectories();
        let n_assets = self.models.len();

        let mut total_risk = 0.0;

        // Sum of individual risks
        for i in 0..n_assets {
            total_risk += trajs[i].execution_risk;
        }

        // Cross-correlation terms (2 * sum of covariances)
        for i in 0..n_assets {
            for j in (i + 1)..n_assets {
                if i < self.correlations.len() && j < self.correlations[i].len() {
                    let cov = self.correlations[i][j]
                        * trajs[i].execution_risk.sqrt()
                        * trajs[j].execution_risk.sqrt();
                    total_risk += 2.0 * cov;
                }
            }
        }

        total_risk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = AlmgrenChriss::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_default_trajectory() {
        let model = AlmgrenChriss::new();
        let traj = model.optimal_trajectory();

        // Should have correct number of points
        assert_eq!(traj.times.len(), 11); // num_intervals + 1
        assert_eq!(traj.trades.len(), 10);
        assert_eq!(traj.holdings.len(), 11);

        // Holdings should start at total_shares and end near 0
        assert!((traj.holdings[0] - 10_000.0).abs() < 0.01);
        assert!(traj.holdings[10] < 1.0);

        // All trades should be positive (selling)
        for trade in &traj.trades {
            assert!(*trade > 0.0);
        }
    }

    #[test]
    fn test_holdings_sum_to_total() {
        let model = AlmgrenChriss::new();
        let traj = model.optimal_trajectory();

        let total_traded: f64 = traj.trades.iter().sum();
        assert!((total_traded - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn test_high_risk_aversion_front_loads() {
        let low_ra = AlmgrenChriss::with_config(AlmgrenChrissConfig {
            risk_aversion: 1e-10,
            ..Default::default()
        });

        let high_ra = AlmgrenChriss::with_config(AlmgrenChrissConfig {
            risk_aversion: 1e-3,
            ..Default::default()
        });

        let low_traj = low_ra.optimal_trajectory();
        let high_traj = high_ra.optimal_trajectory();

        // High risk aversion should trade more in early intervals
        assert!(high_traj.trades[0] > low_traj.trades[0]);
    }

    #[test]
    fn test_zero_risk_aversion_is_twap() {
        let model = AlmgrenChriss::with_config(AlmgrenChrissConfig {
            risk_aversion: 0.0,
            ..Default::default()
        });

        let traj = model.optimal_trajectory();

        // All trades should be approximately equal (TWAP)
        let avg_trade = traj.trades.iter().sum::<f64>() / traj.trades.len() as f64;
        for trade in &traj.trades {
            assert!((trade - avg_trade).abs() / avg_trade < 0.01);
        }
    }

    #[test]
    fn test_holdings_at_time() {
        let model = AlmgrenChriss::new();

        // At t=0, should have all shares
        let h0 = model.holdings_at_time(0.0);
        assert!((h0 - 10_000.0).abs() < 0.01);

        // At t=T, should have no shares
        let ht = model.holdings_at_time(1.0);
        assert!(ht.abs() < 0.01);

        // At t=0.5, should have some shares
        let h_mid = model.holdings_at_time(0.5);
        assert!(h_mid > 0.0 && h_mid < 10_000.0);
    }

    #[test]
    fn test_cost_breakdown() {
        let model = AlmgrenChriss::new();
        let costs = model.cost_breakdown();

        // All costs should be non-negative
        assert!(costs.permanent_cost >= 0.0);
        assert!(costs.temporary_cost >= 0.0);
        assert!(costs.spread_cost >= 0.0);
        assert!(costs.total_cost > 0.0);
    }

    #[test]
    fn test_efficient_frontier() {
        let model = AlmgrenChriss::new();
        let frontier = model.efficient_frontier(10);

        assert_eq!(frontier.len(), 10);

        // Frontier should show risk-return trade-off
        // Higher risk aversion = lower risk, potentially higher cost
        for i in 1..frontier.len() {
            // Risk should generally decrease with higher lambda
            assert!(frontier[i].0 <= frontier[i - 1].0 + 1e-6);
        }
    }

    #[test]
    fn test_twap_comparison() {
        let model = AlmgrenChriss::with_config(AlmgrenChrissConfig {
            risk_aversion: 1e-5,
            ..Default::default()
        });

        let comparison = model.compare_with_twap();

        // AC should generally have lower utility (cost + risk-adjusted)
        // but the comparison depends on parameters
        assert!(comparison.ac_cost > 0.0);
        assert!(comparison.twap_cost > 0.0);
    }

    #[test]
    fn test_constrained_trajectory() {
        let model = AlmgrenChriss::with_config(AlmgrenChrissConfig {
            total_shares: 100_000.0,
            daily_volume: 10_000.0,
            max_participation: 0.1,
            ..Default::default()
        });

        let constrained = model.constrained_trajectory();

        // Check participation constraint
        let max_allowed = 10_000.0 * 0.1 * 0.1; // volume * participation * tau
        for trade in &constrained.trades {
            assert!(trade.abs() <= max_allowed + 1.0); // Small tolerance
        }
    }

    #[test]
    fn test_adaptive_update() {
        let model = AlmgrenChriss::new();

        // Update at midpoint with half shares remaining
        let updated = model.adaptive_update(0.5, 5_000.0, 100.0, None);

        // Should have trajectory for remaining time
        assert!(updated.holdings[0] <= 5_000.0 + 1.0);
    }

    #[test]
    fn test_estimate_impact() {
        let model = AlmgrenChriss::new();

        // Faster execution = higher impact
        let fast_impact = model.estimate_impact(1000.0, 0.1);
        let slow_impact = model.estimate_impact(1000.0, 1.0);

        assert!(fast_impact > slow_impact);
    }

    #[test]
    fn test_multi_asset() {
        let configs = vec![
            AlmgrenChrissConfig::default(),
            AlmgrenChrissConfig {
                total_shares: 5_000.0,
                ..Default::default()
            },
        ];

        let correlations = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let multi = MultiAssetExecution::new(configs, correlations);
        let trajs = multi.trajectories();

        assert_eq!(trajs.len(), 2);

        let portfolio_risk = multi.portfolio_risk();
        assert!(portfolio_risk > 0.0);
    }
}
