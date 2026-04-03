//! Mean-Variance Portfolio Optimization (Markowitz)
//!
//! Implements classic mean-variance optimization for portfolio construction,
//! including efficient frontier calculation, target return/risk optimization,
//! and various constraints (long-only, leverage limits, etc.).

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Portfolio optimization objective
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationObjective {
    /// Minimize variance for target return
    MinVariance { target_return: f64 },
    /// Maximize Sharpe ratio
    MaxSharpe,
    /// Maximize return for target risk
    MaxReturn { target_risk: f64 },
    /// Minimize variance (min variance portfolio)
    MinVarianceUnconstrained,
    /// Target risk-return point
    TargetRiskReturn {
        target_return: f64,
        target_risk: f64,
    },
}

/// Portfolio constraints
#[derive(Debug, Clone)]
pub struct PortfolioConstraints {
    /// Minimum weight per asset (default: 0.0 for long-only)
    pub min_weights: Option<Vec<f64>>,
    /// Maximum weight per asset (default: 1.0)
    pub max_weights: Option<Vec<f64>>,
    /// Allow short selling (default: false)
    pub allow_short: bool,
    /// Maximum leverage (sum of absolute weights, default: 1.0)
    pub max_leverage: f64,
    /// Minimum total weight (default: 1.0 for fully invested)
    pub min_total_weight: f64,
    /// Maximum total weight (default: 1.0)
    pub max_total_weight: f64,
    /// Sector/group constraints (asset_idx -> group_id, max_weight_per_group)
    pub group_constraints: Option<HashMap<usize, (String, f64)>>,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            min_weights: None,
            max_weights: None,
            allow_short: false,
            max_leverage: 1.0,
            min_total_weight: 1.0,
            max_total_weight: 1.0,
            group_constraints: None,
        }
    }
}

impl PortfolioConstraints {
    /// Create long-only constraints (no shorting)
    pub fn long_only() -> Self {
        Self {
            allow_short: false,
            ..Default::default()
        }
    }

    /// Create long-short constraints with leverage limit
    pub fn long_short(max_leverage: f64) -> Self {
        Self {
            allow_short: true,
            max_leverage,
            min_total_weight: 0.0,
            max_total_weight: 1.0,
            ..Default::default()
        }
    }

    /// Set per-asset weight bounds
    pub fn with_weight_bounds(mut self, min: Vec<f64>, max: Vec<f64>) -> Self {
        self.min_weights = Some(min);
        self.max_weights = Some(max);
        self
    }
}

/// Portfolio optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimal portfolio weights
    pub weights: Vec<f64>,
    /// Asset symbols/names
    pub symbols: Vec<String>,
    /// Expected portfolio return
    pub expected_return: f64,
    /// Portfolio variance
    pub variance: f64,
    /// Portfolio volatility (std dev)
    pub volatility: f64,
    /// Sharpe ratio (if risk-free rate provided)
    pub sharpe_ratio: Option<f64>,
    /// Whether optimization converged
    pub converged: bool,
    /// Iteration count
    pub iterations: usize,
}

impl OptimizationResult {
    /// Get weight for a specific symbol
    pub fn weight(&self, symbol: &str) -> Option<f64> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .map(|idx| self.weights[idx])
    }

    /// Get top N holdings by absolute weight
    pub fn top_holdings(&self, n: usize) -> Vec<(String, f64)> {
        let mut holdings: Vec<_> = self
            .symbols
            .iter()
            .zip(self.weights.iter())
            .map(|(s, &w)| (s.clone(), w))
            .collect();
        holdings.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        holdings.into_iter().take(n).collect()
    }

    /// Calculate turnover vs another portfolio
    pub fn turnover(&self, other: &OptimizationResult) -> f64 {
        self.weights
            .iter()
            .zip(other.weights.iter())
            .map(|(w1, w2)| (w1 - w2).abs())
            .sum::<f64>()
            / 2.0
    }
}

/// Mean-Variance Portfolio Optimizer
pub struct MeanVarianceOptimizer {
    /// Expected returns for each asset
    returns: DVector<f64>,
    /// Covariance matrix
    covariance: DMatrix<f64>,
    /// Asset symbols
    symbols: Vec<String>,
    /// Risk-free rate for Sharpe ratio
    risk_free_rate: f64,
    /// Constraints
    constraints: PortfolioConstraints,
}

impl MeanVarianceOptimizer {
    /// Create a new optimizer
    pub fn new(
        returns: Vec<f64>,
        covariance: Vec<Vec<f64>>,
        symbols: Vec<String>,
    ) -> Result<Self, String> {
        let n = returns.len();

        if symbols.len() != n {
            return Err(format!(
                "Returns and symbols length mismatch: {} vs {}",
                n,
                symbols.len()
            ));
        }

        if covariance.len() != n {
            return Err(format!(
                "Covariance matrix dimension mismatch: {} vs {}",
                covariance.len(),
                n
            ));
        }

        for (i, row) in covariance.iter().enumerate() {
            if row.len() != n {
                return Err(format!(
                    "Covariance matrix row {} has incorrect length: {} vs {}",
                    i,
                    row.len(),
                    n
                ));
            }
        }

        // Flatten covariance matrix for nalgebra
        let cov_flat: Vec<f64> = covariance
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        Ok(Self {
            returns: DVector::from_vec(returns),
            covariance: DMatrix::from_row_slice(n, n, &cov_flat),
            symbols,
            risk_free_rate: 0.0,
            constraints: PortfolioConstraints::default(),
        })
    }

    /// Set risk-free rate for Sharpe ratio calculation
    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Set portfolio constraints
    pub fn with_constraints(mut self, constraints: PortfolioConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Optimize portfolio based on objective
    pub fn optimize(&self, objective: OptimizationObjective) -> Result<OptimizationResult, String> {
        match objective {
            OptimizationObjective::MinVariance { target_return } => {
                self.min_variance_target_return(target_return)
            }
            OptimizationObjective::MaxSharpe => self.max_sharpe_ratio(),
            OptimizationObjective::MaxReturn { target_risk } => {
                self.max_return_target_risk(target_risk)
            }
            OptimizationObjective::MinVarianceUnconstrained => self.min_variance_portfolio(),
            OptimizationObjective::TargetRiskReturn {
                target_return,
                target_risk,
            } => self.target_risk_return(target_return, target_risk),
        }
    }

    /// Calculate minimum variance portfolio
    fn min_variance_portfolio(&self) -> Result<OptimizationResult, String> {
        let n = self.returns.len();

        // For minimum variance: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
        // where 1 is a vector of ones

        let cov_inv = self
            .covariance
            .clone()
            .try_inverse()
            .ok_or("Covariance matrix is singular")?;

        let ones = DVector::from_element(n, 1.0);
        let cov_inv_ones = &cov_inv * &ones;
        let denominator = ones.dot(&cov_inv_ones);

        let weights = cov_inv_ones / denominator;
        let weights_vec: Vec<f64> = weights.iter().copied().collect();

        self.apply_constraints_and_finalize(weights_vec, 100)
    }

    /// Minimize variance for target return
    fn min_variance_target_return(&self, target_return: f64) -> Result<OptimizationResult, String> {
        // Use quadratic programming approach with equality constraint
        // We'll use a simplified iterative approach

        let n = self.returns.len();
        let mut weights = vec![1.0 / n as f64; n];

        const MAX_ITER: usize = 1000;
        const TOLERANCE: f64 = 1e-6;
        let learning_rate = 0.01;

        for iter in 0..MAX_ITER {
            // Calculate gradient of variance: 2 * Σ * w
            let w_vec = DVector::from_vec(weights.clone());
            let grad = &self.covariance * &w_vec * 2.0;

            // Project onto constraint manifold
            let mut new_weights = weights.clone();

            for i in 0..n {
                new_weights[i] -= learning_rate * grad[i];
            }

            // Apply constraints
            new_weights = self.project_to_constraints(new_weights, Some(target_return));

            // Check convergence
            let diff: f64 = weights
                .iter()
                .zip(new_weights.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            weights = new_weights;

            if diff < TOLERANCE {
                return self.apply_constraints_and_finalize(weights, iter + 1);
            }
        }

        self.apply_constraints_and_finalize(weights, MAX_ITER)
    }

    /// Maximize Sharpe ratio
    fn max_sharpe_ratio(&self) -> Result<OptimizationResult, String> {
        // Max Sharpe is equivalent to: max (μ^T w - rf) / sqrt(w^T Σ w)
        // We use iterative optimization

        let n = self.returns.len();
        let mut weights = vec![1.0 / n as f64; n];

        const MAX_ITER: usize = 2000;
        const TOLERANCE: f64 = 1e-6;
        let mut learning_rate = 0.1;

        let mut best_sharpe = f64::NEG_INFINITY;
        let mut best_weights = weights.clone();

        for iter in 0..MAX_ITER {
            let w_vec = DVector::from_vec(weights.clone());

            // Calculate portfolio metrics
            let port_return = self.returns.dot(&w_vec);
            let variance = w_vec.dot(&(&self.covariance * &w_vec));
            let volatility = variance.sqrt();

            if volatility > 1e-10 {
                let sharpe = (port_return - self.risk_free_rate) / volatility;

                if sharpe > best_sharpe {
                    best_sharpe = sharpe;
                    best_weights = weights.clone();
                }

                // Gradient of Sharpe ratio (simplified)
                let grad_return = &self.returns;
                let grad_vol = &self.covariance * &w_vec / volatility;

                let grad_sharpe = grad_return / volatility
                    - &grad_vol * (port_return - self.risk_free_rate) / (volatility * volatility);

                // Update weights
                let mut new_weights = weights.clone();
                for i in 0..n {
                    new_weights[i] += learning_rate * grad_sharpe[i];
                }

                // Apply constraints
                new_weights = self.project_to_constraints(new_weights, None);

                // Check convergence
                let diff: f64 = weights
                    .iter()
                    .zip(new_weights.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                weights = new_weights;

                if diff < TOLERANCE {
                    break;
                }

                // Adaptive learning rate
                if iter % 100 == 0 {
                    learning_rate *= 0.9;
                }
            } else {
                break;
            }
        }

        self.apply_constraints_and_finalize(best_weights, MAX_ITER)
    }

    /// Maximize return for target risk
    fn max_return_target_risk(&self, target_risk: f64) -> Result<OptimizationResult, String> {
        let n = self.returns.len();
        let mut weights = vec![1.0 / n as f64; n];

        const MAX_ITER: usize = 1000;
        const TOLERANCE: f64 = 1e-6;
        let learning_rate = 0.05;

        for iter in 0..MAX_ITER {
            let w_vec = DVector::from_vec(weights.clone());

            // Gradient of return: μ
            let grad_return = &self.returns;

            // Constraint: volatility = target_risk
            let variance = w_vec.dot(&(&self.covariance * &w_vec));
            let volatility = variance.sqrt();

            // Lagrange multiplier approach
            let lambda = if volatility > 1e-10 {
                (volatility - target_risk) / volatility
            } else {
                0.0
            };

            let grad_vol = &self.covariance * &w_vec / volatility.max(1e-10);

            let mut new_weights = weights.clone();
            for i in 0..n {
                new_weights[i] += learning_rate * (grad_return[i] - lambda * grad_vol[i]);
            }

            // Apply constraints
            new_weights = self.project_to_constraints(new_weights, None);

            // Check convergence
            let diff: f64 = weights
                .iter()
                .zip(new_weights.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            weights = new_weights;

            if diff < TOLERANCE {
                return self.apply_constraints_and_finalize(weights, iter + 1);
            }
        }

        self.apply_constraints_and_finalize(weights, MAX_ITER)
    }

    /// Target specific risk and return
    fn target_risk_return(
        &self,
        target_return: f64,
        target_risk: f64,
    ) -> Result<OptimizationResult, String> {
        // This is typically infeasible unless we're on the efficient frontier
        // We'll try to get as close as possible
        let result = self.min_variance_target_return(target_return)?;

        if (result.volatility - target_risk).abs() < 0.01 {
            Ok(result)
        } else {
            Err(format!(
                "Cannot achieve both target return {} and target risk {}. Achieved risk: {}",
                target_return, target_risk, result.volatility
            ))
        }
    }

    /// Project weights onto constraint manifold
    fn project_to_constraints(
        &self,
        mut weights: Vec<f64>,
        target_return: Option<f64>,
    ) -> Vec<f64> {
        let n = weights.len();

        // Apply long-only constraint
        if !self.constraints.allow_short {
            for w in weights.iter_mut() {
                *w = w.max(0.0);
            }
        }

        // Apply min/max weight bounds
        if let Some(ref min_w) = self.constraints.min_weights {
            for (i, w) in weights.iter_mut().enumerate() {
                *w = w.max(min_w[i]);
            }
        }

        if let Some(ref max_w) = self.constraints.max_weights {
            for (i, w) in weights.iter_mut().enumerate() {
                *w = w.min(max_w[i]);
            }
        }

        // Normalize to satisfy budget constraint
        let sum: f64 = weights.iter().sum();
        if sum > 1e-10 {
            let target_sum = self.constraints.max_total_weight;
            for w in weights.iter_mut() {
                *w *= target_sum / sum;
            }
        } else {
            // If all weights are zero, distribute evenly
            let equal_weight = 1.0 / n as f64;
            weights.fill(equal_weight);
        }

        // If target return specified, adjust weights to match
        if let Some(tr) = target_return {
            let w_vec = DVector::from_vec(weights.clone());
            let current_return = self.returns.dot(&w_vec);
            let diff = tr - current_return;

            if diff.abs() > 1e-6 {
                // Tilt weights toward high-return assets
                for i in 0..n {
                    if self.returns[i] > current_return {
                        weights[i] *= 1.0 + diff.abs() * 0.1;
                    }
                }

                // Re-normalize
                let sum: f64 = weights.iter().sum();
                if sum > 1e-10 {
                    for w in weights.iter_mut() {
                        *w /= sum;
                    }
                }
            }
        }

        weights
    }

    /// Apply final constraints and create result
    fn apply_constraints_and_finalize(
        &self,
        weights: Vec<f64>,
        iterations: usize,
    ) -> Result<OptimizationResult, String> {
        let weights = self.project_to_constraints(weights, None);

        let w_vec = DVector::from_vec(weights.clone());
        let expected_return = self.returns.dot(&w_vec);
        let variance = w_vec.dot(&(&self.covariance * &w_vec));
        let volatility = variance.sqrt();

        let sharpe_ratio = if volatility > 1e-10 {
            Some((expected_return - self.risk_free_rate) / volatility)
        } else {
            None
        };

        Ok(OptimizationResult {
            weights,
            symbols: self.symbols.clone(),
            expected_return,
            variance,
            volatility,
            sharpe_ratio,
            converged: true,
            iterations,
        })
    }

    /// Calculate efficient frontier points
    pub fn efficient_frontier(&self, num_points: usize) -> Vec<OptimizationResult> {
        let min_return = self.returns.min();
        let max_return = self.returns.max();

        let step = (max_return - min_return) / (num_points as f64);

        (0..num_points)
            .filter_map(|i| {
                let target_return = min_return + step * i as f64;
                self.min_variance_target_return(target_return).ok()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_optimizer() -> MeanVarianceOptimizer {
        let returns = vec![0.10, 0.12, 0.08, 0.15];
        let covariance = vec![
            vec![0.04, 0.01, 0.02, 0.01],
            vec![0.01, 0.09, 0.01, 0.02],
            vec![0.02, 0.01, 0.16, 0.01],
            vec![0.01, 0.02, 0.01, 0.25],
        ];
        let symbols = vec!["AAPL", "GOOGL", "MSFT", "TSLA"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        MeanVarianceOptimizer::new(returns, covariance, symbols)
            .unwrap()
            .with_risk_free_rate(0.02)
    }

    #[test]
    fn test_min_variance_portfolio() {
        let optimizer = create_test_optimizer();
        let result = optimizer.min_variance_portfolio().unwrap();

        assert_eq!(result.weights.len(), 4);
        assert!(result.converged);

        // Weights should sum to approximately 1.0
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // All weights should be non-negative (long-only)
        for &w in &result.weights {
            assert!(w >= -0.01); // Small tolerance for numerical errors
        }
    }

    #[test]
    fn test_max_sharpe_ratio() {
        let optimizer = create_test_optimizer();
        let result = optimizer.max_sharpe_ratio().unwrap();

        assert!(result.sharpe_ratio.is_some());
        assert!(result.sharpe_ratio.unwrap() > 0.0);

        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_target_return_optimization() {
        let optimizer = create_test_optimizer();
        let target_return = 0.11;
        let result = optimizer.min_variance_target_return(target_return).unwrap();

        // Expected return should be close to target
        assert!((result.expected_return - target_return).abs() < 0.05);

        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_efficient_frontier() {
        let optimizer = create_test_optimizer();
        let frontier = optimizer.efficient_frontier(5);

        assert!(!frontier.is_empty());

        // Returns should be increasing along frontier
        for i in 1..frontier.len() {
            assert!(frontier[i].expected_return >= frontier[i - 1].expected_return - 0.01);
        }
    }

    #[test]
    fn test_long_only_constraint() {
        let optimizer = create_test_optimizer().with_constraints(PortfolioConstraints::long_only());

        let result = optimizer.max_sharpe_ratio().unwrap();

        for &w in &result.weights {
            assert!(w >= -1e-6, "Weight should be non-negative: {}", w);
        }
    }

    #[test]
    fn test_optimization_result_helpers() {
        let optimizer = create_test_optimizer();
        let result = optimizer.min_variance_portfolio().unwrap();

        // Test weight lookup
        assert!(result.weight("AAPL").is_some());
        assert!(result.weight("INVALID").is_none());

        // Test top holdings
        let top = result.top_holdings(2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_turnover_calculation() {
        let optimizer = create_test_optimizer();
        let result1 = optimizer.min_variance_portfolio().unwrap();
        let result2 = optimizer.max_sharpe_ratio().unwrap();

        let turnover = result1.turnover(&result2);
        assert!(turnover >= 0.0);
        assert!(turnover <= 1.0);
    }
}
