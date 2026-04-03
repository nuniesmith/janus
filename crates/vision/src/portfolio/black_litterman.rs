//! Black-Litterman Portfolio Optimization
//!
//! Implements the Black-Litterman model which combines market equilibrium
//! returns with investor views to generate expected returns for portfolio optimization.
//! This provides a Bayesian framework for incorporating subjective views into
//! mean-variance optimization.

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Investor view on asset returns
#[derive(Debug, Clone)]
pub struct View {
    /// Description of the view
    pub description: String,
    /// Asset picks (symbol -> weight in view)
    /// e.g., {"AAPL": 1.0, "GOOGL": -1.0} means AAPL will outperform GOOGL
    pub picks: HashMap<String, f64>,
    /// Expected return for this view (absolute or relative)
    pub expected_return: f64,
    /// Confidence level (0.0 to 1.0, higher = more confident)
    pub confidence: f64,
}

impl View {
    /// Create a new view
    pub fn new(description: String, expected_return: f64, confidence: f64) -> Self {
        Self {
            description,
            picks: HashMap::new(),
            expected_return,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Add an asset to the view
    pub fn add_asset(mut self, symbol: String, weight: f64) -> Self {
        self.picks.insert(symbol, weight);
        self
    }

    /// Create an absolute view (single asset will have return R)
    pub fn absolute(symbol: String, expected_return: f64, confidence: f64) -> Self {
        Self::new(
            format!("{} absolute return view", symbol),
            expected_return,
            confidence,
        )
        .add_asset(symbol, 1.0)
    }

    /// Create a relative view (asset A will outperform asset B by R)
    pub fn relative(
        symbol_a: String,
        symbol_b: String,
        outperformance: f64,
        confidence: f64,
    ) -> Self {
        Self::new(
            format!("{} vs {} relative view", symbol_a, symbol_b),
            outperformance,
            confidence,
        )
        .add_asset(symbol_a, 1.0)
        .add_asset(symbol_b, -1.0)
    }
}

/// Black-Litterman model configuration
#[derive(Debug, Clone)]
pub struct BlackLittermanConfig {
    /// Risk aversion coefficient (typically 2.5 to 3.5)
    pub risk_aversion: f64,
    /// Tau parameter (uncertainty in prior, typically 0.01 to 0.05)
    pub tau: f64,
    /// Use market capitalization weights as prior (if false, use equal weights)
    pub use_market_cap_weights: bool,
}

impl Default for BlackLittermanConfig {
    fn default() -> Self {
        Self {
            risk_aversion: 2.5,
            tau: 0.025,
            use_market_cap_weights: true,
        }
    }
}

/// Black-Litterman optimization result
#[derive(Debug, Clone)]
pub struct BlackLittermanResult {
    /// Posterior expected returns (after incorporating views)
    pub expected_returns: Vec<f64>,
    /// Posterior covariance matrix
    pub posterior_covariance: Vec<Vec<f64>>,
    /// Asset symbols
    pub symbols: Vec<String>,
    /// Prior (equilibrium) expected returns
    pub prior_returns: Vec<f64>,
    /// Optimal portfolio weights
    pub weights: Vec<f64>,
    /// Portfolio expected return
    pub portfolio_return: f64,
    /// Portfolio volatility
    pub portfolio_volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
}

impl BlackLittermanResult {
    /// Get expected return for a specific symbol
    pub fn expected_return(&self, symbol: &str) -> Option<f64> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .map(|idx| self.expected_returns[idx])
    }

    /// Get weight for a specific symbol
    pub fn weight(&self, symbol: &str) -> Option<f64> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .map(|idx| self.weights[idx])
    }

    /// Compare posterior vs prior returns
    pub fn return_adjustments(&self) -> Vec<(String, f64, f64, f64)> {
        self.symbols
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (
                    s.clone(),
                    self.prior_returns[i],
                    self.expected_returns[i],
                    self.expected_returns[i] - self.prior_returns[i],
                )
            })
            .collect()
    }
}

/// Black-Litterman Portfolio Optimizer
pub struct BlackLittermanOptimizer {
    /// Asset symbols
    symbols: Vec<String>,
    /// Covariance matrix of returns
    covariance: DMatrix<f64>,
    /// Market capitalization weights (optional)
    market_cap_weights: Option<Vec<f64>>,
    /// Investor views
    views: Vec<View>,
    /// Configuration
    config: BlackLittermanConfig,
    /// Risk-free rate
    risk_free_rate: f64,
}

impl BlackLittermanOptimizer {
    /// Create a new Black-Litterman optimizer
    pub fn new(symbols: Vec<String>, covariance: Vec<Vec<f64>>) -> Result<Self, String> {
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
            symbols,
            covariance: DMatrix::from_row_slice(n, n, &cov_flat),
            market_cap_weights: None,
            views: Vec::new(),
            config: BlackLittermanConfig::default(),
            risk_free_rate: 0.0,
        })
    }

    /// Set market capitalization weights
    pub fn with_market_cap_weights(mut self, weights: Vec<f64>) -> Result<Self, String> {
        if weights.len() != self.symbols.len() {
            return Err("Market cap weights length mismatch".to_string());
        }

        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(format!("Market cap weights must sum to 1.0, got {}", sum));
        }

        self.market_cap_weights = Some(weights);
        Ok(self)
    }

    /// Add an investor view
    pub fn add_view(mut self, view: View) -> Self {
        self.views.push(view);
        self
    }

    /// Add multiple views
    pub fn with_views(mut self, views: Vec<View>) -> Self {
        self.views.extend(views);
        self
    }

    /// Set configuration
    pub fn with_config(mut self, config: BlackLittermanConfig) -> Self {
        self.config = config;
        self
    }

    /// Set risk-free rate
    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Optimize the portfolio using Black-Litterman
    pub fn optimize(&self) -> Result<BlackLittermanResult, String> {
        let n = self.symbols.len();

        // Step 1: Calculate prior (equilibrium) returns
        let prior_weights = self.get_prior_weights();
        let prior_returns = self.calculate_equilibrium_returns(&prior_weights);

        // Step 2: Build view matrices
        let (p_matrix, q_vector, omega_matrix) = self.build_view_matrices()?;

        // Step 3: Calculate posterior returns using Black-Litterman formula
        let posterior_returns = if !self.views.is_empty() {
            self.calculate_posterior_returns(&prior_returns, &p_matrix, &q_vector, &omega_matrix)?
        } else {
            // No views, use prior returns
            prior_returns.clone()
        };

        // Step 4: Calculate posterior covariance
        let posterior_cov = if !self.views.is_empty() {
            self.calculate_posterior_covariance(&p_matrix, &omega_matrix)?
        } else {
            self.covariance.clone()
        };

        // Step 5: Optimize portfolio weights using posterior distribution
        let weights = self.optimize_weights(&posterior_returns)?;

        // Step 6: Calculate portfolio metrics
        let w_vec = DVector::from_vec(weights.clone());
        let r_vec = DVector::from_vec(posterior_returns.iter().copied().collect());
        let portfolio_return = w_vec.dot(&r_vec);

        let variance = w_vec.dot(&(&posterior_cov * &w_vec));
        let portfolio_volatility = variance.sqrt();

        let sharpe_ratio = if portfolio_volatility > 1e-10 {
            (portfolio_return - self.risk_free_rate) / portfolio_volatility
        } else {
            0.0
        };

        // Convert posterior covariance back to Vec<Vec<f64>>
        let posterior_covariance: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| posterior_cov[(i, j)]).collect())
            .collect();

        Ok(BlackLittermanResult {
            expected_returns: posterior_returns.iter().copied().collect(),
            posterior_covariance,
            symbols: self.symbols.clone(),
            prior_returns: prior_returns.iter().copied().collect(),
            weights,
            portfolio_return,
            portfolio_volatility,
            sharpe_ratio,
        })
    }

    /// Get prior weights (market cap or equal weight)
    fn get_prior_weights(&self) -> DVector<f64> {
        if self.config.use_market_cap_weights {
            if let Some(ref weights) = self.market_cap_weights {
                return DVector::from_vec(weights.clone());
            }
        }

        // Equal weights as fallback
        let n = self.symbols.len();
        DVector::from_element(n, 1.0 / n as f64)
    }

    /// Calculate equilibrium (implied) returns using reverse optimization
    /// Π = δ * Σ * w_mkt
    fn calculate_equilibrium_returns(&self, market_weights: &DVector<f64>) -> DVector<f64> {
        &self.covariance * market_weights * self.config.risk_aversion
    }

    /// Build view matrices P, Q, and Ω
    fn build_view_matrices(&self) -> Result<(DMatrix<f64>, DVector<f64>, DMatrix<f64>), String> {
        let n = self.symbols.len();
        let k = self.views.len();

        if k == 0 {
            // No views - return empty matrices
            return Ok((
                DMatrix::zeros(0, n),
                DVector::zeros(0),
                DMatrix::zeros(0, 0),
            ));
        }

        // P matrix (k x n): pick matrix
        let mut p_data = vec![0.0; k * n];

        // Q vector (k x 1): view returns
        let mut q_data = vec![0.0; k];

        // Ω matrix (k x k): view uncertainty (diagonal)
        let mut omega_data = vec![0.0; k * k];

        for (view_idx, view) in self.views.iter().enumerate() {
            // Set Q
            q_data[view_idx] = view.expected_return;

            // Set P
            for (symbol, &weight) in &view.picks {
                if let Some(asset_idx) = self.symbols.iter().position(|s| s == symbol) {
                    p_data[view_idx * n + asset_idx] = weight;
                } else {
                    return Err(format!("Unknown symbol in view: {}", symbol));
                }
            }

            // Set Ω (diagonal element based on confidence)
            // Lower confidence = higher uncertainty
            // Ω_ii = (1/confidence - 1) * τ * P * Σ * P^T
            let p_row = DVector::from_vec((0..n).map(|j| p_data[view_idx * n + j]).collect());
            let variance = p_row.dot(&(&self.covariance * &p_row));

            let uncertainty = if view.confidence > 0.01 {
                ((1.0 / view.confidence) - 1.0) * self.config.tau * variance
            } else {
                variance * self.config.tau * 100.0 // Very uncertain
            };

            omega_data[view_idx * k + view_idx] = uncertainty;
        }

        Ok((
            DMatrix::from_row_slice(k, n, &p_data),
            DVector::from_vec(q_data),
            DMatrix::from_row_slice(k, k, &omega_data),
        ))
    }

    /// Calculate posterior expected returns using Black-Litterman formula
    /// E[R] = [(τΣ)^-1 + P^T Ω^-1 P]^-1 [(τΣ)^-1 Π + P^T Ω^-1 Q]
    fn calculate_posterior_returns(
        &self,
        prior_returns: &DVector<f64>,
        p_matrix: &DMatrix<f64>,
        q_vector: &DVector<f64>,
        omega_matrix: &DMatrix<f64>,
    ) -> Result<DVector<f64>, String> {
        let tau_cov = &self.covariance * self.config.tau;

        // Inverse of tau * Sigma
        let tau_cov_inv = tau_cov
            .clone()
            .try_inverse()
            .ok_or("Tau * Covariance matrix is singular")?;

        // Inverse of Omega
        let omega_inv = omega_matrix
            .clone()
            .try_inverse()
            .ok_or("Omega matrix is singular")?;

        // Left side: (τΣ)^-1 + P^T Ω^-1 P
        let pt_omega_inv_p = p_matrix.transpose() * &omega_inv * p_matrix;
        let left = &tau_cov_inv + &pt_omega_inv_p;

        let left_inv = left.try_inverse().ok_or("Left matrix is singular")?;

        // Right side: (τΣ)^-1 Π + P^T Ω^-1 Q
        let term1 = &tau_cov_inv * prior_returns;
        let term2 = p_matrix.transpose() * &omega_inv * q_vector;
        let right = term1 + term2;

        // Posterior returns
        Ok(&left_inv * right)
    }

    /// Calculate posterior covariance
    /// Σ_post = Σ + [(τΣ)^-1 + P^T Ω^-1 P]^-1
    fn calculate_posterior_covariance(
        &self,
        p_matrix: &DMatrix<f64>,
        omega_matrix: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, String> {
        let tau_cov = &self.covariance * self.config.tau;

        let tau_cov_inv = tau_cov
            .clone()
            .try_inverse()
            .ok_or("Tau * Covariance matrix is singular")?;

        let omega_inv = omega_matrix
            .clone()
            .try_inverse()
            .ok_or("Omega matrix is singular")?;

        let pt_omega_inv_p = p_matrix.transpose() * &omega_inv * p_matrix;
        let sum = &tau_cov_inv + &pt_omega_inv_p;

        let sum_inv = sum.try_inverse().ok_or("Sum matrix is singular")?;

        Ok(&self.covariance + sum_inv)
    }

    /// Optimize weights given expected returns
    fn optimize_weights(&self, expected_returns: &DVector<f64>) -> Result<Vec<f64>, String> {
        // Use mean-variance optimization: w = Σ^-1 * E[R] / λ
        // Then normalize to sum to 1

        let cov_inv = self
            .covariance
            .clone()
            .try_inverse()
            .ok_or("Covariance matrix is singular")?;

        let weights_unnorm = &cov_inv * expected_returns / self.config.risk_aversion;

        // Normalize to sum to 1
        let sum: f64 = weights_unnorm.iter().sum();
        let weights = if sum.abs() > 1e-10 {
            weights_unnorm / sum
        } else {
            let n = self.symbols.len();
            DVector::from_element(n, 1.0 / n as f64)
        };

        // Apply long-only constraint
        let mut weights_vec: Vec<f64> = weights.iter().copied().collect();
        for w in weights_vec.iter_mut() {
            *w = w.max(0.0);
        }

        // Re-normalize after applying constraints
        let sum: f64 = weights_vec.iter().sum();
        if sum > 1e-10 {
            for w in weights_vec.iter_mut() {
                *w /= sum;
            }
        }

        Ok(weights_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_optimizer() -> BlackLittermanOptimizer {
        let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let covariance = vec![
            vec![0.0400, 0.0100, 0.0150, 0.0120],
            vec![0.0100, 0.0900, 0.0200, 0.0150],
            vec![0.0150, 0.0200, 0.0625, 0.0180],
            vec![0.0120, 0.0150, 0.0180, 0.1600],
        ];

        let market_weights = vec![0.3, 0.25, 0.25, 0.2];

        BlackLittermanOptimizer::new(symbols, covariance)
            .unwrap()
            .with_market_cap_weights(market_weights)
            .unwrap()
            .with_risk_free_rate(0.02)
    }

    #[test]
    fn test_no_views_optimization() {
        let optimizer = create_test_optimizer();
        let result = optimizer.optimize().unwrap();

        assert_eq!(result.symbols.len(), 4);
        assert_eq!(result.expected_returns.len(), 4);
        assert_eq!(result.weights.len(), 4);

        // Weights should sum to 1.0
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_absolute_view() {
        let optimizer =
            create_test_optimizer().add_view(View::absolute("AAPL".to_string(), 0.15, 0.8));

        let result = optimizer.optimize().unwrap();

        // AAPL's posterior return should be influenced by the view
        let aapl_return = result.expected_return("AAPL").unwrap();
        assert!(aapl_return > 0.0);

        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_relative_view() {
        let optimizer = create_test_optimizer().add_view(View::relative(
            "AAPL".to_string(),
            "GOOGL".to_string(),
            0.05,
            0.7,
        ));

        let result = optimizer.optimize().unwrap();

        let aapl_return = result.expected_return("AAPL").unwrap();
        let googl_return = result.expected_return("GOOGL").unwrap();

        // AAPL should have higher expected return than GOOGL
        // (though not exactly 5% due to Bayesian blending)
        assert!(aapl_return > googl_return - 0.01);
    }

    #[test]
    fn test_multiple_views() {
        let optimizer = create_test_optimizer()
            .add_view(View::absolute("AAPL".to_string(), 0.12, 0.8))
            .add_view(View::relative(
                "MSFT".to_string(),
                "AMZN".to_string(),
                0.03,
                0.6,
            ));

        let result = optimizer.optimize().unwrap();

        assert_eq!(result.symbols.len(), 4);

        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_return_adjustments() {
        let optimizer =
            create_test_optimizer().add_view(View::absolute("AAPL".to_string(), 0.15, 0.9));

        let result = optimizer.optimize().unwrap();
        let adjustments = result.return_adjustments();

        assert_eq!(adjustments.len(), 4);

        // Check that adjustments contain symbol, prior, posterior, and delta
        for (symbol, prior, posterior, delta) in adjustments {
            assert!(!symbol.is_empty());
            assert!((delta - (posterior - prior)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_view_confidence_impact() {
        // High confidence view
        let optimizer_high =
            create_test_optimizer().add_view(View::absolute("AAPL".to_string(), 0.20, 0.95));
        let result_high = optimizer_high.optimize().unwrap();

        // Low confidence view
        let optimizer_low =
            create_test_optimizer().add_view(View::absolute("AAPL".to_string(), 0.20, 0.3));
        let result_low = optimizer_low.optimize().unwrap();

        let aapl_return_high = result_high.expected_return("AAPL").unwrap();
        let aapl_return_low = result_low.expected_return("AAPL").unwrap();

        // High confidence should pull posterior closer to view
        let prior_return = result_high.prior_returns[0];
        let high_diff = (aapl_return_high - prior_return).abs();
        let low_diff = (aapl_return_low - prior_return).abs();

        assert!(
            high_diff > low_diff,
            "High confidence should adjust more from prior"
        );
    }

    #[test]
    fn test_market_cap_weights() {
        let weights = vec![0.4, 0.3, 0.2, 0.1];
        let result = create_test_optimizer()
            .with_market_cap_weights(weights)
            .unwrap()
            .optimize();

        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_market_cap_weights() {
        let weights = vec![0.4, 0.3, 0.2]; // Wrong length
        let result = create_test_optimizer().with_market_cap_weights(weights);

        assert!(result.is_err());
    }

    #[test]
    fn test_config_customization() {
        let config = BlackLittermanConfig {
            risk_aversion: 3.0,
            tau: 0.05,
            use_market_cap_weights: false,
        };

        let optimizer = create_test_optimizer().with_config(config);
        let result = optimizer.optimize().unwrap();

        assert!(result.portfolio_return.is_finite());
        assert!(result.portfolio_volatility > 0.0);
    }
}
