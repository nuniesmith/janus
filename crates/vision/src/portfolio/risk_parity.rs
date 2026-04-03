//! Risk Parity Portfolio Optimization
//!
//! Implements risk parity (equal risk contribution) portfolio allocation,
//! where each asset contributes equally to the total portfolio risk.
//! This approach is also known as Equal Risk Contribution (ERC).

use nalgebra::{DMatrix, DVector};

/// Risk parity optimization method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskParityMethod {
    /// Equal risk contribution (standard risk parity)
    EqualRiskContribution,
    /// Inverse volatility weighting
    InverseVolatility,
    /// Risk budgeting with custom risk budgets
    RiskBudgeting,
}

/// Risk budgets for portfolio assets
#[derive(Debug, Clone)]
pub struct RiskBudget {
    /// Asset symbols
    pub symbols: Vec<String>,
    /// Risk budget for each asset (should sum to 1.0)
    pub budgets: Vec<f64>,
}

impl RiskBudget {
    /// Create equal risk budget (standard risk parity)
    pub fn equal(symbols: Vec<String>) -> Self {
        let n = symbols.len();
        let budget = 1.0 / n as f64;
        Self {
            symbols,
            budgets: vec![budget; n],
        }
    }

    /// Create custom risk budget
    pub fn custom(symbols: Vec<String>, budgets: Vec<f64>) -> Result<Self, String> {
        if symbols.len() != budgets.len() {
            return Err("Symbols and budgets length mismatch".to_string());
        }

        let sum: f64 = budgets.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(format!("Risk budgets must sum to 1.0, got {}", sum));
        }

        for &b in &budgets {
            if b < 0.0 || b > 1.0 {
                return Err(format!("Risk budget must be in [0, 1], got {}", b));
            }
        }

        Ok(Self { symbols, budgets })
    }
}

/// Risk parity portfolio result
#[derive(Debug, Clone)]
pub struct RiskParityResult {
    /// Optimal portfolio weights
    pub weights: Vec<f64>,
    /// Asset symbols
    pub symbols: Vec<String>,
    /// Risk contribution for each asset
    pub risk_contributions: Vec<f64>,
    /// Total portfolio volatility
    pub volatility: f64,
    /// Percentage risk contribution for each asset
    pub risk_contribution_pct: Vec<f64>,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final error (deviation from target risk contributions)
    pub error: f64,
}

impl RiskParityResult {
    /// Get weight for a specific symbol
    pub fn weight(&self, symbol: &str) -> Option<f64> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .map(|idx| self.weights[idx])
    }

    /// Get risk contribution for a specific symbol
    pub fn risk_contribution(&self, symbol: &str) -> Option<f64> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .map(|idx| self.risk_contributions[idx])
    }

    /// Check if risk contributions are approximately equal
    pub fn is_equal_risk(&self, tolerance: f64) -> bool {
        let target = 1.0 / self.symbols.len() as f64;
        self.risk_contribution_pct
            .iter()
            .all(|&rc| (rc - target).abs() < tolerance)
    }

    /// Get maximum deviation from equal risk
    pub fn max_risk_deviation(&self) -> f64 {
        let target = 1.0 / self.symbols.len() as f64;
        self.risk_contribution_pct
            .iter()
            .map(|&rc| (rc - target).abs())
            .fold(0.0f64, |a, b| a.max(b))
    }
}

/// Risk Parity Portfolio Optimizer
pub struct RiskParityOptimizer {
    /// Covariance matrix
    covariance: DMatrix<f64>,
    /// Asset symbols
    symbols: Vec<String>,
    /// Risk budgets (optional)
    risk_budget: Option<RiskBudget>,
    /// Optimization method
    method: RiskParityMethod,
}

impl RiskParityOptimizer {
    /// Create a new risk parity optimizer
    pub fn new(covariance: Vec<Vec<f64>>, symbols: Vec<String>) -> Result<Self, String> {
        let n = symbols.len();

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

        // Flatten covariance matrix
        let cov_flat: Vec<f64> = covariance
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        Ok(Self {
            covariance: DMatrix::from_row_slice(n, n, &cov_flat),
            symbols,
            risk_budget: None,
            method: RiskParityMethod::EqualRiskContribution,
        })
    }

    /// Set the optimization method
    pub fn with_method(mut self, method: RiskParityMethod) -> Self {
        self.method = method;
        self
    }

    /// Set custom risk budgets
    pub fn with_risk_budget(mut self, budget: RiskBudget) -> Self {
        self.risk_budget = Some(budget);
        self
    }

    /// Optimize the portfolio
    pub fn optimize(&self) -> Result<RiskParityResult, String> {
        match self.method {
            RiskParityMethod::EqualRiskContribution => self.equal_risk_contribution(),
            RiskParityMethod::InverseVolatility => self.inverse_volatility(),
            RiskParityMethod::RiskBudgeting => {
                if let Some(ref budget) = self.risk_budget {
                    self.risk_budgeting(budget)
                } else {
                    self.equal_risk_contribution()
                }
            }
        }
    }

    /// Calculate portfolio using inverse volatility weighting
    fn inverse_volatility(&self) -> Result<RiskParityResult, String> {
        let n = self.symbols.len();

        // Extract volatilities from diagonal
        let mut volatilities = vec![0.0; n];
        for i in 0..n {
            volatilities[i] = self.covariance[(i, i)].sqrt();
            if volatilities[i] < 1e-10 {
                return Err(format!("Asset {} has zero volatility", i));
            }
        }

        // Weights are inverse of volatility
        let mut weights: Vec<f64> = volatilities.iter().map(|v| 1.0 / v).collect();

        // Normalize
        let sum: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum;
        }

        self.finalize_result(weights, 1)
    }

    /// Calculate equal risk contribution portfolio
    fn equal_risk_contribution(&self) -> Result<RiskParityResult, String> {
        let n = self.symbols.len();
        let target_budgets = vec![1.0 / n as f64; n];
        self.optimize_risk_budgets(&target_budgets)
    }

    /// Calculate risk budgeting portfolio
    fn risk_budgeting(&self, budget: &RiskBudget) -> Result<RiskParityResult, String> {
        self.optimize_risk_budgets(&budget.budgets)
    }

    /// Core optimization algorithm for risk budgets
    fn optimize_risk_budgets(&self, target_budgets: &[f64]) -> Result<RiskParityResult, String> {
        let n = self.symbols.len();

        // Initialize with inverse volatility
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let vol = self.covariance[(i, i)].sqrt();
            weights[i] = if vol > 1e-10 { 1.0 / vol } else { 1.0 };
        }

        // Normalize
        let sum: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum;
        }

        const MAX_ITER: usize = 1000;
        const TOLERANCE: f64 = 1e-6;
        let mut learning_rate = 0.5;

        let mut best_error = f64::INFINITY;
        let mut best_weights = weights.clone();
        let mut no_improvement_count = 0;

        for iter in 0..MAX_ITER {
            // Calculate current risk contributions
            let risk_contribs = self.calculate_risk_contributions(&weights);

            // Calculate error (deviation from target budgets)
            let error: f64 = risk_contribs
                .iter()
                .zip(target_budgets.iter())
                .map(|(rc, tb)| (rc - tb).powi(2))
                .sum::<f64>()
                .sqrt();

            if error < best_error {
                best_error = error;
                best_weights = weights.clone();
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
            }

            // Check convergence
            if error < TOLERANCE {
                return self.finalize_result(weights, iter + 1);
            }

            // Reduce learning rate if no improvement
            if no_improvement_count > 20 {
                learning_rate *= 0.5;
                no_improvement_count = 0;
            }

            // Update weights using gradient descent
            let mut new_weights = weights.clone();

            for i in 0..n {
                // Gradient: how risk contribution changes with weight
                let delta = 1e-6;
                let mut weights_plus = weights.clone();
                weights_plus[i] += delta;

                // Normalize
                let sum: f64 = weights_plus.iter().sum();
                for w in weights_plus.iter_mut() {
                    *w /= sum;
                }

                let rc_plus = self.calculate_risk_contributions(&weights_plus);
                let gradient = (rc_plus[i] - risk_contribs[i]) / delta;

                // Update toward target
                let error_i = risk_contribs[i] - target_budgets[i];
                new_weights[i] -= learning_rate * error_i * gradient;

                // Keep weights positive
                new_weights[i] = new_weights[i].max(1e-8);
            }

            // Normalize new weights
            let sum: f64 = new_weights.iter().sum();
            for w in new_weights.iter_mut() {
                *w /= sum;
            }

            // Check for weight changes
            let weight_diff: f64 = weights
                .iter()
                .zip(new_weights.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            weights = new_weights;

            if weight_diff < TOLERANCE && iter > 50 {
                return self.finalize_result(weights, iter + 1);
            }

            // Adaptive learning rate
            if iter % 100 == 99 {
                learning_rate *= 0.9;
            }
        }

        // Return best result found
        self.finalize_result(best_weights, MAX_ITER)
    }

    /// Calculate risk contributions for given weights
    fn calculate_risk_contributions(&self, weights: &[f64]) -> Vec<f64> {
        let n = weights.len();
        let w_vec = DVector::from_vec(weights.to_vec());

        // Portfolio variance: w^T * Sigma * w
        let cov_w = &self.covariance * &w_vec;
        let variance = w_vec.dot(&cov_w);
        let volatility = variance.sqrt();

        if volatility < 1e-10 {
            return vec![1.0 / n as f64; n];
        }

        // Marginal contribution to risk: (Sigma * w) / sigma_p
        let mut risk_contributions = vec![0.0; n];
        for i in 0..n {
            // RUSTCODE_i = w_i * (Sigma * w)_i / sigma_p
            risk_contributions[i] = weights[i] * cov_w[i] / volatility;
        }

        // Normalize to sum to 1
        let sum: f64 = risk_contributions.iter().sum();
        if sum > 1e-10 {
            for rc in risk_contributions.iter_mut() {
                *rc /= sum;
            }
        }

        risk_contributions
    }

    /// Finalize the optimization result
    fn finalize_result(
        &self,
        weights: Vec<f64>,
        iterations: usize,
    ) -> Result<RiskParityResult, String> {
        let risk_contributions = self.calculate_risk_contributions(&weights);

        // Calculate portfolio volatility
        let w_vec = DVector::from_vec(weights.clone());
        let cov_w = &self.covariance * &w_vec;
        let variance = w_vec.dot(&cov_w);
        let volatility = variance.sqrt();

        // Calculate percentage risk contributions
        let risk_contribution_pct = risk_contributions.clone();

        // Calculate final error
        let n = weights.len();
        let target = 1.0 / n as f64;
        let error: f64 = risk_contribution_pct
            .iter()
            .map(|rc| (rc - target).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(RiskParityResult {
            weights,
            symbols: self.symbols.clone(),
            risk_contributions,
            volatility,
            risk_contribution_pct,
            converged: error < 1e-3,
            iterations,
            error,
        })
    }

    /// Calculate the diversification ratio
    pub fn diversification_ratio(&self, weights: &[f64]) -> f64 {
        let n = weights.len();

        // Weighted average volatility
        let mut weighted_vol = 0.0;
        for i in 0..n {
            let vol = self.covariance[(i, i)].sqrt();
            weighted_vol += weights[i] * vol;
        }

        // Portfolio volatility
        let w_vec = DVector::from_vec(weights.to_vec());
        let variance = w_vec.dot(&(&self.covariance * &w_vec));
        let portfolio_vol = variance.sqrt();

        if portfolio_vol > 1e-10 {
            weighted_vol / portfolio_vol
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_optimizer() -> RiskParityOptimizer {
        let covariance = vec![
            vec![0.04, 0.01, 0.02],
            vec![0.01, 0.09, 0.015],
            vec![0.02, 0.015, 0.16],
        ];
        let symbols = vec!["AAPL", "GOOGL", "MSFT"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        RiskParityOptimizer::new(covariance, symbols).unwrap()
    }

    #[test]
    fn test_inverse_volatility() {
        let optimizer = create_test_optimizer();
        let result = optimizer
            .with_method(RiskParityMethod::InverseVolatility)
            .optimize()
            .unwrap();

        assert_eq!(result.weights.len(), 3);

        // Weights should sum to 1.0
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Lower volatility assets should have higher weights
        // Volatilities: sqrt(0.04)=0.2, sqrt(0.09)=0.3, sqrt(0.16)=0.4
        assert!(result.weights[0] > result.weights[1]);
        assert!(result.weights[1] > result.weights[2]);
    }

    #[test]
    fn test_equal_risk_contribution() {
        let optimizer = create_test_optimizer();
        let result = optimizer
            .with_method(RiskParityMethod::EqualRiskContribution)
            .optimize()
            .unwrap();

        assert_eq!(result.weights.len(), 3);

        // Weights should sum to 1.0
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Risk contributions should be approximately equal
        let target = 1.0 / 3.0;
        for &rc in &result.risk_contribution_pct {
            assert!(
                (rc - target).abs() < 0.1,
                "Risk contribution {} far from target {}",
                rc,
                target
            );
        }

        // Risk contributions should sum to 1.0
        let rc_sum: f64 = result.risk_contribution_pct.iter().sum();
        assert!((rc_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_risk_budgeting() {
        let optimizer = create_test_optimizer();
        let budget = RiskBudget::custom(
            vec!["AAPL", "GOOGL", "MSFT"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            vec![0.5, 0.3, 0.2],
        )
        .unwrap();

        let result = optimizer
            .with_method(RiskParityMethod::RiskBudgeting)
            .with_risk_budget(budget)
            .optimize()
            .unwrap();

        // Weights should sum to 1.0
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Risk contributions should approximately match budgets
        assert!((result.risk_contribution_pct[0] - 0.5).abs() < 0.15);
        assert!((result.risk_contribution_pct[1] - 0.3).abs() < 0.15);
        assert!((result.risk_contribution_pct[2] - 0.2).abs() < 0.15);
    }

    #[test]
    fn test_risk_budget_validation() {
        let symbols: Vec<String> = vec!["A", "B"].iter().map(|s| s.to_string()).collect();

        // Should fail - budgets don't sum to 1
        let result = RiskBudget::custom(symbols.clone(), vec![0.5, 0.6]);
        assert!(result.is_err());

        // Should fail - negative budget
        let result = RiskBudget::custom(symbols.clone(), vec![-0.1, 1.1]);
        assert!(result.is_err());

        // Should succeed
        let result = RiskBudget::custom(symbols.clone(), vec![0.4, 0.6]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_result_helpers() {
        let optimizer = create_test_optimizer();
        let result = optimizer.optimize().unwrap();

        // Test weight lookup
        assert!(result.weight("AAPL").is_some());
        assert!(result.weight("INVALID").is_none());

        // Test risk contribution lookup
        assert!(result.risk_contribution("GOOGL").is_some());
        assert!(result.risk_contribution("INVALID").is_none());

        // Test equal risk check
        assert!(result.is_equal_risk(0.2));

        // Test max deviation
        let dev = result.max_risk_deviation();
        assert!(dev >= 0.0);
        assert!(dev < 0.5);
    }

    #[test]
    fn test_diversification_ratio() {
        let optimizer = create_test_optimizer();
        let result = optimizer.optimize().unwrap();

        let div_ratio = optimizer.diversification_ratio(&result.weights);
        assert!(div_ratio >= 1.0, "Diversification ratio should be >= 1.0");
        assert!(
            div_ratio < 10.0,
            "Diversification ratio seems unreasonably high"
        );
    }

    #[test]
    fn test_convergence() {
        let optimizer = create_test_optimizer();
        let result = optimizer.optimize().unwrap();

        assert!(result.converged || result.error < 0.01);
        assert!(result.iterations > 0);
    }
}
