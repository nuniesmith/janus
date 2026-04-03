//! Portfolio Optimization Module
//!
//! This module provides comprehensive portfolio optimization tools including:
//! - Mean-Variance Optimization (Markowitz)
//! - Risk Parity / Equal Risk Contribution
//! - Black-Litterman Model
//! - Portfolio analytics and rebalancing utilities
//!
//! # Examples
//!
//! ## Mean-Variance Optimization
//!
//! ```rust,ignore
//! use janus_vision::portfolio::{MeanVarianceOptimizer, OptimizationObjective};
//!
//! let returns = vec![0.10, 0.12, 0.08];
//! let covariance = vec![
//!     vec![0.04, 0.01, 0.02],
//!     vec![0.01, 0.09, 0.01],
//!     vec![0.02, 0.01, 0.16],
//! ];
//! let symbols = vec!["AAPL", "GOOGL", "MSFT"].iter().map(|s| s.to_string()).collect();
//!
//! let optimizer = MeanVarianceOptimizer::new(returns, covariance, symbols)
//!     .unwrap()
//!     .with_risk_free_rate(0.02);
//!
//! let result = optimizer.optimize(OptimizationObjective::MaxSharpe).unwrap();
//! println!("Sharpe Ratio: {:.4}", result.sharpe_ratio.unwrap());
//! ```
//!
//! ## Risk Parity
//!
//! ```rust,ignore
//! use janus_vision::portfolio::{RiskParityOptimizer, RiskParityMethod};
//!
//! let covariance = vec![
//!     vec![0.04, 0.01, 0.02],
//!     vec![0.01, 0.09, 0.01],
//!     vec![0.02, 0.01, 0.16],
//! ];
//! let symbols = vec!["AAPL", "GOOGL", "MSFT"].iter().map(|s| s.to_string()).collect();
//!
//! let optimizer = RiskParityOptimizer::new(covariance, symbols)
//!     .unwrap()
//!     .with_method(RiskParityMethod::EqualRiskContribution);
//!
//! let result = optimizer.optimize().unwrap();
//! for (symbol, risk_pct) in result.symbols.iter().zip(result.risk_contribution_pct.iter()) {
//!     println!("{}: {:.2}% risk contribution", symbol, risk_pct * 100.0);
//! }
//! ```
//!
//! ## Black-Litterman
//!
//! ```rust,ignore
//! use janus_vision::portfolio::{BlackLittermanOptimizer, View};
//!
//! let symbols = vec!["AAPL", "GOOGL", "MSFT"].iter().map(|s| s.to_string()).collect();
//! let covariance = vec![
//!     vec![0.04, 0.01, 0.02],
//!     vec![0.01, 0.09, 0.01],
//!     vec![0.02, 0.01, 0.16],
//! ];
//! let market_weights = vec![0.4, 0.3, 0.3];
//!
//! let optimizer = BlackLittermanOptimizer::new(symbols, covariance)
//!     .unwrap()
//!     .with_market_cap_weights(market_weights)
//!     .unwrap()
//!     .add_view(View::absolute("AAPL".to_string(), 0.15, 0.8))
//!     .add_view(View::relative("GOOGL".to_string(), "MSFT".to_string(), 0.05, 0.7));
//!
//! let result = optimizer.optimize().unwrap();
//! println!("Portfolio Return: {:.4}", result.portfolio_return);
//! ```

pub mod black_litterman;
pub mod mean_variance;
pub mod risk_parity;

pub use black_litterman::{
    BlackLittermanConfig, BlackLittermanOptimizer, BlackLittermanResult, View,
};
pub use mean_variance::{
    MeanVarianceOptimizer, OptimizationObjective, OptimizationResult, PortfolioConstraints,
};
pub use risk_parity::{RiskBudget, RiskParityMethod, RiskParityOptimizer, RiskParityResult};

use nalgebra::{DMatrix, DVector};

/// Portfolio analytics utilities
pub struct PortfolioAnalytics;

impl PortfolioAnalytics {
    /// Calculate portfolio return given weights and expected returns
    pub fn portfolio_return(weights: &[f64], returns: &[f64]) -> f64 {
        weights.iter().zip(returns.iter()).map(|(w, r)| w * r).sum()
    }

    /// Calculate portfolio variance given weights and covariance matrix
    pub fn portfolio_variance(weights: &[f64], covariance: &[Vec<f64>]) -> f64 {
        let n = weights.len();
        let w = DVector::from_vec(weights.to_vec());
        let cov_flat: Vec<f64> = covariance
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let cov = DMatrix::from_row_slice(n, n, &cov_flat);
        w.dot(&(&cov * &w))
    }

    /// Calculate portfolio volatility (standard deviation)
    pub fn portfolio_volatility(weights: &[f64], covariance: &[Vec<f64>]) -> f64 {
        Self::portfolio_variance(weights, covariance).sqrt()
    }

    /// Calculate Sharpe ratio
    pub fn sharpe_ratio(
        weights: &[f64],
        returns: &[f64],
        covariance: &[Vec<f64>],
        risk_free_rate: f64,
    ) -> f64 {
        let port_return = Self::portfolio_return(weights, returns);
        let port_vol = Self::portfolio_volatility(weights, covariance);
        if port_vol > 1e-10 {
            (port_return - risk_free_rate) / port_vol
        } else {
            0.0
        }
    }

    /// Calculate portfolio beta relative to market
    pub fn portfolio_beta(weights: &[f64], asset_betas: &[f64]) -> f64 {
        weights
            .iter()
            .zip(asset_betas.iter())
            .map(|(w, b)| w * b)
            .sum()
    }

    /// Calculate turnover between two portfolios
    pub fn turnover(weights_old: &[f64], weights_new: &[f64]) -> f64 {
        weights_old
            .iter()
            .zip(weights_new.iter())
            .map(|(w1, w2)| (w1 - w2).abs())
            .sum::<f64>()
            / 2.0
    }

    /// Calculate tracking error vs benchmark
    pub fn tracking_error(
        weights: &[f64],
        benchmark_weights: &[f64],
        covariance: &[Vec<f64>],
    ) -> f64 {
        let active_weights: Vec<f64> = weights
            .iter()
            .zip(benchmark_weights.iter())
            .map(|(w, b)| w - b)
            .collect();
        Self::portfolio_volatility(&active_weights, covariance)
    }

    /// Calculate information ratio
    pub fn information_ratio(
        weights: &[f64],
        benchmark_weights: &[f64],
        returns: &[f64],
        covariance: &[Vec<f64>],
    ) -> f64 {
        let active_return = Self::portfolio_return(weights, returns)
            - Self::portfolio_return(benchmark_weights, returns);
        let te = Self::tracking_error(weights, benchmark_weights, covariance);
        if te > 1e-10 { active_return / te } else { 0.0 }
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown(portfolio_values: &[f64]) -> f64 {
        if portfolio_values.is_empty() {
            return 0.0;
        }

        let mut max_value = portfolio_values[0];
        let mut max_dd = 0.0;

        for &value in portfolio_values.iter() {
            if value > max_value {
                max_value = value;
            }
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }

    /// Calculate Sortino ratio (downside risk-adjusted return)
    pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, target_return: f64) -> f64 {
        let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        // Downside deviation (only negative deviations from target)
        let downside_variance: f64 = returns
            .iter()
            .map(|&r| {
                let diff = r - target_return;
                if diff < 0.0 { diff * diff } else { 0.0 }
            })
            .sum::<f64>()
            / returns.len() as f64;

        let downside_dev = downside_variance.sqrt();
        if downside_dev > 1e-10 {
            (avg_return - risk_free_rate) / downside_dev
        } else {
            0.0
        }
    }

    /// Calculate Value at Risk (VaR) using historical simulation
    pub fn var_historical(returns: &[f64], confidence_level: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence_level) * returns.len() as f64) as usize;
        let index = index.min(returns.len() - 1);

        -sorted_returns[index] // Negative of the return at the threshold
    }

    /// Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
    pub fn cvar_historical(returns: &[f64], confidence_level: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let var = Self::var_historical(returns, confidence_level);
        let threshold = -var;

        let tail_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r <= threshold)
            .copied()
            .collect();

        if tail_returns.is_empty() {
            return var;
        }

        let cvar: f64 = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
        -cvar
    }
}

/// Portfolio rebalancing utilities
pub struct PortfolioRebalancer;

impl PortfolioRebalancer {
    /// Calculate trades needed to rebalance from current to target weights
    pub fn calculate_trades(
        current_weights: &[f64],
        target_weights: &[f64],
        portfolio_value: f64,
    ) -> Vec<f64> {
        current_weights
            .iter()
            .zip(target_weights.iter())
            .map(|(current, target)| (target - current) * portfolio_value)
            .collect()
    }

    /// Apply rebalancing threshold (only trade if drift exceeds threshold)
    pub fn threshold_rebalance(
        current_weights: &[f64],
        target_weights: &[f64],
        threshold: f64,
    ) -> Vec<f64> {
        current_weights
            .iter()
            .zip(target_weights.iter())
            .map(|(current, target)| {
                let diff = target - current;
                if diff.abs() > threshold {
                    *target
                } else {
                    *current
                }
            })
            .collect()
    }

    /// Calculate rebalancing frequency based on turnover cost
    pub fn optimal_rebalancing_frequency(
        tracking_error_cost: f64,
        transaction_cost_bps: f64,
        expected_turnover: f64,
    ) -> f64 {
        // Simple heuristic: trade-off between tracking error and transaction costs
        // Returns number of days between rebalances

        if expected_turnover < 1e-10 {
            return f64::INFINITY;
        }

        let cost_per_rebalance = expected_turnover * transaction_cost_bps / 10000.0;
        let te_cost_per_day = tracking_error_cost;

        // Optimal frequency minimizes total cost
        (cost_per_rebalance / te_cost_per_day).sqrt()
    }
}

/// Covariance estimation utilities
pub struct CovarianceEstimator;

impl CovarianceEstimator {
    /// Estimate covariance matrix from return series
    pub fn from_returns(returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_assets = returns.len();
        if n_assets == 0 {
            return vec![];
        }

        let n_periods = returns[0].len();
        if n_periods == 0 {
            return vec![vec![0.0; n_assets]; n_assets];
        }

        // Calculate means
        let means: Vec<f64> = returns
            .iter()
            .map(|r| r.iter().sum::<f64>() / n_periods as f64)
            .collect();

        // Calculate covariance
        let mut covariance = vec![vec![0.0; n_assets]; n_assets];

        for i in 0..n_assets {
            for j in 0..n_assets {
                let mut cov = 0.0;
                for t in 0..n_periods {
                    cov += (returns[i][t] - means[i]) * (returns[j][t] - means[j]);
                }
                covariance[i][j] = cov / (n_periods - 1) as f64;
            }
        }

        covariance
    }

    /// Apply exponential weighting to covariance (more recent data weighted more)
    pub fn exponential_weighted(
        returns: &[Vec<f64>],
        lambda: f64, // decay factor (e.g., 0.94 for RiskMetrics)
    ) -> Vec<Vec<f64>> {
        let n_assets = returns.len();
        if n_assets == 0 {
            return vec![];
        }

        let n_periods = returns[0].len();
        if n_periods == 0 {
            return vec![vec![0.0; n_assets]; n_assets];
        }

        // Calculate exponentially weighted means
        let mut weight_sum = 0.0;
        let mut weighted_means = vec![0.0; n_assets];

        for t in 0..n_periods {
            let weight = lambda.powi((n_periods - 1 - t) as i32);
            weight_sum += weight;
            for i in 0..n_assets {
                weighted_means[i] += weight * returns[i][t];
            }
        }

        for mean in weighted_means.iter_mut() {
            *mean /= weight_sum;
        }

        // Calculate exponentially weighted covariance
        let mut covariance = vec![vec![0.0; n_assets]; n_assets];

        for i in 0..n_assets {
            for j in 0..n_assets {
                let mut cov = 0.0;
                for t in 0..n_periods {
                    let weight = lambda.powi((n_periods - 1 - t) as i32);
                    cov += weight
                        * (returns[i][t] - weighted_means[i])
                        * (returns[j][t] - weighted_means[j]);
                }
                covariance[i][j] = cov / weight_sum;
            }
        }

        covariance
    }

    /// Shrinkage estimator (Ledoit-Wolf)
    pub fn ledoit_wolf_shrinkage(sample_cov: &[Vec<f64>], shrinkage: f64) -> Vec<Vec<f64>> {
        let n = sample_cov.len();
        if n == 0 {
            return vec![];
        }

        // Target: constant correlation matrix
        let mut avg_variance = 0.0;
        for i in 0..n {
            avg_variance += sample_cov[i][i];
        }
        avg_variance /= n as f64;

        let mut avg_covariance = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    avg_covariance += sample_cov[i][j];
                }
            }
        }
        avg_covariance /= (n * (n - 1)) as f64;

        // Shrink sample covariance toward target
        let mut shrunk_cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    shrunk_cov[i][j] =
                        (1.0 - shrinkage) * sample_cov[i][j] + shrinkage * avg_variance;
                } else {
                    shrunk_cov[i][j] =
                        (1.0 - shrinkage) * sample_cov[i][j] + shrinkage * avg_covariance;
                }
            }
        }

        shrunk_cov
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_analytics() {
        let weights = vec![0.4, 0.3, 0.3];
        let returns = vec![0.10, 0.12, 0.08];
        let covariance = vec![
            vec![0.04, 0.01, 0.02],
            vec![0.01, 0.09, 0.01],
            vec![0.02, 0.01, 0.16],
        ];

        let port_return = PortfolioAnalytics::portfolio_return(&weights, &returns);
        assert!((port_return - 0.1).abs() < 0.01);

        let port_vol = PortfolioAnalytics::portfolio_volatility(&weights, &covariance);
        assert!(port_vol > 0.0);

        let sharpe = PortfolioAnalytics::sharpe_ratio(&weights, &returns, &covariance, 0.02);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_turnover() {
        let old_weights = vec![0.4, 0.3, 0.3];
        let new_weights = vec![0.3, 0.4, 0.3];

        let turnover = PortfolioAnalytics::turnover(&old_weights, &new_weights);
        assert!((turnover - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_max_drawdown() {
        let values = vec![100.0, 110.0, 105.0, 95.0, 100.0, 120.0];
        let mdd = PortfolioAnalytics::max_drawdown(&values);

        // Max was 110, min after that was 95, so DD = (110-95)/110 = 13.64%
        assert!((mdd - 0.1364).abs() < 0.01);
    }

    #[test]
    fn test_var_cvar() {
        let returns = vec![-0.05, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05];
        let var = PortfolioAnalytics::var_historical(&returns, 0.95);
        assert!(var > 0.0);

        let cvar = PortfolioAnalytics::cvar_historical(&returns, 0.95);
        assert!(cvar >= var);
    }

    #[test]
    fn test_rebalancing() {
        let current = vec![0.5, 0.3, 0.2];
        let target = vec![0.4, 0.3, 0.3];
        let portfolio_value = 100000.0;

        let trades = PortfolioRebalancer::calculate_trades(&current, &target, portfolio_value);
        assert_eq!(trades.len(), 3);
        assert!((trades[0] + 10000.0).abs() < 1.0); // Sell $10k of asset 0
        assert!((trades[2] - 10000.0).abs() < 1.0); // Buy $10k of asset 2
    }

    #[test]
    fn test_threshold_rebalance() {
        let current = vec![0.51, 0.29, 0.20];
        let target = vec![0.50, 0.30, 0.20];
        let threshold = 0.02;

        let new_weights = PortfolioRebalancer::threshold_rebalance(&current, &target, threshold);
        // Differences are small, should keep current weights
        assert_eq!(new_weights, current);
    }

    #[test]
    fn test_covariance_estimation() {
        let returns = vec![
            vec![0.01, 0.02, -0.01, 0.03],
            vec![0.02, 0.01, 0.00, 0.02],
            vec![-0.01, 0.03, 0.02, 0.01],
        ];

        let cov = CovarianceEstimator::from_returns(&returns);
        assert_eq!(cov.len(), 3);
        assert_eq!(cov[0].len(), 3);

        // Diagonal elements (variances) should be positive
        for i in 0..3 {
            assert!(cov[i][i] > 0.0);
        }

        // Covariance should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((cov[i][j] - cov[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_exponential_weighted_covariance() {
        let returns = vec![vec![0.01, 0.02, -0.01, 0.03], vec![0.02, 0.01, 0.00, 0.02]];

        let cov = CovarianceEstimator::exponential_weighted(&returns, 0.94);
        assert_eq!(cov.len(), 2);

        // Should be symmetric
        assert!((cov[0][1] - cov[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_ledoit_wolf_shrinkage() {
        let sample_cov = vec![
            vec![0.04, 0.01, 0.02],
            vec![0.01, 0.09, 0.01],
            vec![0.02, 0.01, 0.16],
        ];

        let shrunk = CovarianceEstimator::ledoit_wolf_shrinkage(&sample_cov, 0.2);
        assert_eq!(shrunk.len(), 3);

        // Diagonal elements should be between sample and average
        for i in 0..3 {
            assert!(shrunk[i][i] > 0.0);
        }
    }
}
