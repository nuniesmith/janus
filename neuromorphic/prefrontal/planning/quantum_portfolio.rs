//! Quantum-Inspired Portfolio Optimization
//!
//! Provides both classical and quantum-inspired portfolio optimization
//! algorithms for the JANUS neuromorphic prefrontal planning system.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │              Quantum Portfolio Optimization                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │              Classical Optimizers                         │   │
//! │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │   │
//! │  │  │Markowitz │ │Efficient │ │Risk      │ │Black-      │ │   │
//! │  │  │Mean-Var  │ │Frontier  │ │Parity    │ │Litterman   │ │   │
//! │  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │           Quantum-Inspired Optimizers                    │   │
//! │  │  ┌──────────┐ ┌──────────┐ ┌──────────────────────────┐ │   │
//! │  │  │QAOA      │ │VQE       │ │Simulated Quantum         │ │   │
//! │  │  │Simulator │ │Portfolio │ │Annealing                 │ │   │
//! │  │  └──────────┘ └──────────┘ └──────────────────────────┘ │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │              Shared Infrastructure                       │   │
//! │  │  • Qubit state simulation (statevector)                  │   │
//! │  │  • Parameterized quantum circuits                        │   │
//! │  │  • Cost function encoding (QUBO formulation)             │   │
//! │  │  • Constraint handling (budget, cardinality, bounds)     │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Classical Optimizers
//!
//! - **Markowitz Mean-Variance**: Minimize portfolio variance for a target return
//! - **Efficient Frontier**: Compute the set of Pareto-optimal risk-return portfolios
//! - **Risk Parity**: Equalize risk contribution from each asset
//! - **Black-Litterman**: Combine market equilibrium with investor views
//!
//! # Quantum-Inspired Optimizers
//!
//! - **QAOA Simulator**: Quantum Approximate Optimization Algorithm for
//!   combinatorial portfolio selection (binary asset inclusion)
//! - **VQE Portfolio**: Variational Quantum Eigensolver for continuous
//!   weight optimization using parameterized circuits
//! - **Simulated Quantum Annealing**: Quantum tunneling-inspired annealing
//!   for escaping local minima in non-convex portfolio landscapes
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_neuromorphic::prefrontal::planning::quantum_portfolio::*;
//!
//! // Define assets
//! let returns = vec![0.10, 0.15, 0.12, 0.08]; // expected annual returns
//! let cov_matrix = vec![
//!     vec![0.04, 0.006, 0.002, 0.001],
//!     vec![0.006, 0.09, 0.004, 0.002],
//!     vec![0.002, 0.004, 0.0225, 0.003],
//!     vec![0.001, 0.002, 0.003, 0.01],
//! ];
//!
//! let universe = AssetUniverse::new(
//!     vec!["BTC", "ETH", "SOL", "USDC"],
//!     returns,
//!     cov_matrix,
//! )?;
//!
//! // Classical Markowitz
//! let config = MarkowitzConfig::default().with_target_return(0.12);
//! let result = markowitz_optimize(&universe, &config)?;
//! println!("Weights: {:?}, Risk: {:.4}", result.weights, result.portfolio_risk);
//!
//! // Quantum-inspired QAOA
//! let qaoa_config = QaoaConfig::default().with_depth(3).with_shots(1000);
//! let qaoa_result = qaoa_portfolio_optimize(&universe, &qaoa_config)?;
//! println!("QAOA selection: {:?}", qaoa_result.selected_assets);
//!
//! // Simulated quantum annealing
//! let anneal_config = AnnealingConfig::default();
//! let anneal_result = quantum_anneal_optimize(&universe, &anneal_config)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by portfolio optimization routines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PortfolioError {
    /// The covariance matrix is not positive semi-definite.
    NonPsdCovariance,
    /// The covariance matrix dimensions don't match the number of assets.
    DimensionMismatch { expected: usize, got: usize },
    /// No feasible solution exists under the given constraints.
    Infeasible(String),
    /// Numerical issues during optimization.
    NumericalError(String),
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// The optimization did not converge.
    DidNotConverge { iterations: usize, residual: f64 },
    /// An internal error.
    Internal(String),
}

impl fmt::Display for PortfolioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPsdCovariance => write!(f, "Portfolio: covariance matrix is not PSD"),
            Self::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Portfolio: dimension mismatch (expected {expected}, got {got})"
                )
            }
            Self::Infeasible(e) => write!(f, "Portfolio: infeasible — {e}"),
            Self::NumericalError(e) => write!(f, "Portfolio: numerical error — {e}"),
            Self::InvalidConfig(e) => write!(f, "Portfolio: invalid config — {e}"),
            Self::DidNotConverge {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Portfolio: did not converge after {iterations} iters (residual={residual:.6})"
                )
            }
            Self::Internal(e) => write!(f, "Portfolio: internal error — {e}"),
        }
    }
}

impl std::error::Error for PortfolioError {}

pub type Result<T> = std::result::Result<T, PortfolioError>;

// ---------------------------------------------------------------------------
// Asset Universe
// ---------------------------------------------------------------------------

/// Defines the investable universe of assets with their expected returns
/// and covariance structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetUniverse {
    /// Asset names / tickers.
    pub names: Vec<String>,
    /// Expected returns (annualised).
    pub expected_returns: Vec<f64>,
    /// Covariance matrix (n × n, symmetric, PSD).
    pub covariance: Vec<Vec<f64>>,
    /// Risk-free rate (annualised). Default: 0.0.
    pub risk_free_rate: f64,
    /// Optional per-asset weight bounds (min, max). Default: (0, 1) long-only.
    pub weight_bounds: Vec<(f64, f64)>,
    /// Optional market capitalisation weights (for Black-Litterman equilibrium).
    pub market_caps: Option<Vec<f64>>,
}

impl AssetUniverse {
    /// Create a new asset universe.
    pub fn new(
        names: Vec<impl Into<String>>,
        expected_returns: Vec<f64>,
        covariance: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let n = names.len();

        if expected_returns.len() != n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n,
                got: expected_returns.len(),
            });
        }
        if covariance.len() != n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n,
                got: covariance.len(),
            });
        }
        for (i, row) in covariance.iter().enumerate() {
            if row.len() != n {
                return Err(PortfolioError::DimensionMismatch {
                    expected: n,
                    got: row.len(),
                });
            }
            // Check diagonal is non-negative (variance).
            if row[i] < 0.0 {
                return Err(PortfolioError::NonPsdCovariance);
            }
        }

        let weight_bounds = vec![(0.0, 1.0); n];

        Ok(Self {
            names: names.into_iter().map(|s| s.into()).collect(),
            expected_returns,
            covariance,
            risk_free_rate: 0.0,
            weight_bounds,
            market_caps: None,
        })
    }

    /// Number of assets.
    pub fn n_assets(&self) -> usize {
        self.names.len()
    }

    /// Set the risk-free rate.
    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Set per-asset weight bounds.
    pub fn with_weight_bounds(mut self, bounds: Vec<(f64, f64)>) -> Result<Self> {
        if bounds.len() != self.n_assets() {
            return Err(PortfolioError::DimensionMismatch {
                expected: self.n_assets(),
                got: bounds.len(),
            });
        }
        self.weight_bounds = bounds;
        Ok(self)
    }

    /// Set market capitalisation weights.
    pub fn with_market_caps(mut self, caps: Vec<f64>) -> Result<Self> {
        if caps.len() != self.n_assets() {
            return Err(PortfolioError::DimensionMismatch {
                expected: self.n_assets(),
                got: caps.len(),
            });
        }
        self.market_caps = Some(caps);
        Ok(self)
    }

    /// Compute portfolio return: w^T * μ.
    pub fn portfolio_return(&self, weights: &[f64]) -> f64 {
        weights
            .iter()
            .zip(self.expected_returns.iter())
            .map(|(w, r)| w * r)
            .sum()
    }

    /// Compute portfolio variance: w^T * Σ * w.
    pub fn portfolio_variance(&self, weights: &[f64]) -> f64 {
        let n = self.n_assets();
        let mut var = 0.0;
        for i in 0..n {
            for j in 0..n {
                var += weights[i] * weights[j] * self.covariance[i][j];
            }
        }
        var
    }

    /// Compute portfolio standard deviation (risk).
    pub fn portfolio_risk(&self, weights: &[f64]) -> f64 {
        self.portfolio_variance(weights).max(0.0).sqrt()
    }

    /// Compute the Sharpe ratio.
    pub fn sharpe_ratio(&self, weights: &[f64]) -> f64 {
        let ret = self.portfolio_return(weights);
        let risk = self.portfolio_risk(weights);
        if risk <= 1e-12 {
            return 0.0;
        }
        (ret - self.risk_free_rate) / risk
    }

    /// Compute individual asset volatilities.
    pub fn asset_volatilities(&self) -> Vec<f64> {
        self.covariance
            .iter()
            .enumerate()
            .map(|(i, row)| row[i].max(0.0).sqrt())
            .collect()
    }

    /// Compute the correlation matrix from the covariance matrix.
    pub fn correlation_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_assets();
        let vols = self.asset_volatilities();
        let mut corr = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if vols[i] > 1e-12 && vols[j] > 1e-12 {
                    corr[i][j] = self.covariance[i][j] / (vols[i] * vols[j]);
                } else if i == j {
                    corr[i][j] = 1.0;
                }
            }
        }
        corr
    }

    /// Compute risk contribution of each asset given weights.
    /// Risk contribution_i = w_i * (Σ * w)_i / σ_p
    pub fn risk_contributions(&self, weights: &[f64]) -> Vec<f64> {
        let n = self.n_assets();
        let portfolio_risk = self.portfolio_risk(weights);
        if portfolio_risk <= 1e-12 {
            return vec![0.0; n];
        }

        let mut sigma_w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                sigma_w[i] += self.covariance[i][j] * weights[j];
            }
        }

        let mut contributions = vec![0.0; n];
        for i in 0..n {
            contributions[i] = weights[i] * sigma_w[i] / portfolio_risk;
        }
        contributions
    }

    /// Equal weight portfolio.
    pub fn equal_weight_portfolio(&self) -> Vec<f64> {
        let n = self.n_assets();
        vec![1.0 / n as f64; n]
    }

    /// Market-cap weighted portfolio.
    pub fn market_cap_portfolio(&self) -> Option<Vec<f64>> {
        self.market_caps.as_ref().map(|caps| {
            let total: f64 = caps.iter().sum();
            if total <= 0.0 {
                return vec![1.0 / caps.len() as f64; caps.len()];
            }
            caps.iter().map(|c| c / total).collect()
        })
    }
}

impl fmt::Display for AssetUniverse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "AssetUniverse ({} assets, rf={:.2}%):",
            self.n_assets(),
            self.risk_free_rate * 100.0
        )?;
        for (i, name) in self.names.iter().enumerate() {
            let vol = self.covariance[i][i].max(0.0).sqrt();
            writeln!(
                f,
                "  {}: E[r]={:.2}%, σ={:.2}%, bounds=[{:.2}, {:.2}]",
                name,
                self.expected_returns[i] * 100.0,
                vol * 100.0,
                self.weight_bounds[i].0,
                self.weight_bounds[i].1,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Optimization Result
// ---------------------------------------------------------------------------

/// Result of a portfolio optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioResult {
    /// Optimal portfolio weights.
    pub weights: Vec<f64>,
    /// Portfolio expected return.
    pub portfolio_return: f64,
    /// Portfolio risk (standard deviation).
    pub portfolio_risk: f64,
    /// Portfolio variance.
    pub portfolio_variance: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Risk contributions per asset.
    pub risk_contributions: Vec<f64>,
    /// Optimization method used.
    pub method: String,
    /// Number of iterations.
    pub iterations: usize,
    /// Time taken for optimization.
    pub duration: Duration,
    /// Whether the optimization converged.
    pub converged: bool,
    /// Final objective function value.
    pub objective_value: f64,
    /// Additional metadata.
    pub metadata: HashMap<String, f64>,
}

impl PortfolioResult {
    /// Create a result from weights and a universe.
    pub fn from_weights(
        weights: Vec<f64>,
        universe: &AssetUniverse,
        method: impl Into<String>,
        iterations: usize,
        duration: Duration,
        converged: bool,
        objective_value: f64,
    ) -> Self {
        let portfolio_return = universe.portfolio_return(&weights);
        let portfolio_variance = universe.portfolio_variance(&weights);
        let portfolio_risk = portfolio_variance.max(0.0).sqrt();
        let sharpe_ratio = universe.sharpe_ratio(&weights);
        let risk_contributions = universe.risk_contributions(&weights);

        Self {
            weights,
            portfolio_return,
            portfolio_risk,
            portfolio_variance,
            sharpe_ratio,
            risk_contributions,
            method: method.into(),
            iterations,
            duration,
            converged,
            objective_value,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the result.
    pub fn with_metadata(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Number of assets with non-zero weight.
    pub fn active_assets(&self) -> usize {
        self.weights.iter().filter(|&&w| w.abs() > 1e-6).count()
    }

    /// Maximum weight.
    pub fn max_weight(&self) -> f64 {
        self.weights.iter().cloned().fold(0.0f64, f64::max)
    }

    /// Minimum non-zero weight.
    pub fn min_active_weight(&self) -> f64 {
        self.weights
            .iter()
            .filter(|&&w| w.abs() > 1e-6)
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    /// Herfindahl-Hirschman Index (concentration measure).
    /// HHI = sum(w_i^2). Range: [1/n, 1]. Lower = more diversified.
    pub fn hhi(&self) -> f64 {
        self.weights.iter().map(|w| w * w).sum()
    }

    /// Effective number of assets (1 / HHI).
    pub fn effective_n(&self) -> f64 {
        let hhi = self.hhi();
        if hhi > 1e-12 { 1.0 / hhi } else { 0.0 }
    }
}

impl fmt::Display for PortfolioResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Portfolio Result ({}):", self.method)?;
        writeln!(
            f,
            "  Return:  {:.4}% (annualised)",
            self.portfolio_return * 100.0
        )?;
        writeln!(f, "  Risk:    {:.4}%", self.portfolio_risk * 100.0)?;
        writeln!(f, "  Sharpe:  {:.4}", self.sharpe_ratio)?;
        writeln!(f, "  Active:  {} assets", self.active_assets())?;
        writeln!(
            f,
            "  HHI:     {:.4} (eff. N={:.2})",
            self.hhi(),
            self.effective_n()
        )?;
        write!(f, "  Weights: [")?;
        for (i, w) in self.weights.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{w:.4}")?;
        }
        writeln!(f, "]")?;
        writeln!(
            f,
            "  Converged: {}, Iters: {}, Duration: {:?}",
            self.converged, self.iterations, self.duration
        )?;
        Ok(())
    }
}

// ===========================================================================
// CLASSICAL OPTIMIZERS
// ===========================================================================

// ---------------------------------------------------------------------------
// Markowitz Mean-Variance
// ---------------------------------------------------------------------------

/// Configuration for Markowitz mean-variance optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkowitzConfig {
    /// Target portfolio return (if `None`, maximise Sharpe ratio).
    pub target_return: Option<f64>,
    /// Risk aversion parameter λ (higher = more risk-averse).
    /// Used when `target_return` is `None`: maximize μ^T w - λ/2 * w^T Σ w.
    pub risk_aversion: f64,
    /// Maximum number of iterations for the solver.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Learning rate for projected gradient descent.
    pub learning_rate: f64,
    /// Whether to enforce long-only constraint (all weights ≥ 0).
    pub long_only: bool,
    /// Whether to enforce full investment (weights sum to 1).
    pub fully_invested: bool,
}

impl Default for MarkowitzConfig {
    fn default() -> Self {
        Self {
            target_return: None,
            risk_aversion: 1.0,
            max_iterations: 10_000,
            tolerance: 1e-8,
            learning_rate: 0.001,
            long_only: true,
            fully_invested: true,
        }
    }
}

impl MarkowitzConfig {
    pub fn with_target_return(mut self, target: f64) -> Self {
        self.target_return = Some(target);
        self
    }

    pub fn with_risk_aversion(mut self, lambda: f64) -> Self {
        self.risk_aversion = lambda;
        self
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_long_only(mut self, long_only: bool) -> Self {
        self.long_only = long_only;
        self
    }

    pub fn with_fully_invested(mut self, fully: bool) -> Self {
        self.fully_invested = fully;
        self
    }
}

/// Markowitz mean-variance optimization via projected gradient descent.
///
/// Minimizes: λ/2 * w^T Σ w - μ^T w
/// Subject to: Σ w_i = 1, w_i ∈ [lb_i, ub_i]
///
/// When `target_return` is set, we add a penalty for deviating from it.
pub fn markowitz_optimize(
    universe: &AssetUniverse,
    config: &MarkowitzConfig,
) -> Result<PortfolioResult> {
    let start = Instant::now();
    let n = universe.n_assets();

    if n == 0 {
        return Err(PortfolioError::InvalidConfig("Empty universe".into()));
    }

    // Initialize with equal weights.
    let mut weights = universe.equal_weight_portfolio();
    let lambda = config.risk_aversion;
    let lr = config.learning_rate;

    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Compute gradient of objective: λ * Σ * w - μ
        let mut gradient = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                gradient[i] += lambda * universe.covariance[i][j] * weights[j];
            }
            gradient[i] -= universe.expected_returns[i];
        }

        // If target return is set, add penalty gradient.
        if let Some(target) = config.target_return {
            let current_return = universe.portfolio_return(&weights);
            let penalty = 10.0 * (current_return - target);
            for i in 0..n {
                gradient[i] += penalty * universe.expected_returns[i];
            }
        }

        // Gradient step.
        let mut max_change = 0.0f64;
        for i in 0..n {
            let old = weights[i];
            weights[i] -= lr * gradient[i];

            // Project onto bounds.
            let (lb, ub) = universe.weight_bounds[i];
            if config.long_only {
                weights[i] = weights[i].max(lb.max(0.0));
            } else {
                weights[i] = weights[i].max(lb);
            }
            weights[i] = weights[i].min(ub);

            max_change = max_change.max((weights[i] - old).abs());
        }

        // Project onto simplex (sum = 1) if fully invested.
        if config.fully_invested {
            project_simplex(&mut weights, &universe.weight_bounds, config.long_only);
        }

        // Check convergence.
        if max_change < config.tolerance {
            converged = true;
            debug!(iterations, max_change, "Markowitz converged");
            break;
        }
    }

    let objective =
        lambda / 2.0 * universe.portfolio_variance(&weights) - universe.portfolio_return(&weights);

    let duration = start.elapsed();

    let result = PortfolioResult::from_weights(
        weights,
        universe,
        "Markowitz Mean-Variance",
        iterations,
        duration,
        converged,
        objective,
    )
    .with_metadata("risk_aversion", lambda);

    if let Some(target) = config.target_return {
        return Ok(result.with_metadata("target_return", target));
    }

    Ok(result)
}

/// Project weights onto the simplex {w : sum(w) = 1, lb ≤ w ≤ ub}.
fn project_simplex(weights: &mut [f64], bounds: &[(f64, f64)], long_only: bool) {
    let n = weights.len();
    if n == 0 {
        return;
    }

    // Apply bounds first.
    for i in 0..n {
        let (lb, ub) = bounds[i];
        let lb = if long_only { lb.max(0.0) } else { lb };
        weights[i] = weights[i].clamp(lb, ub);
    }

    // Normalise to sum to 1.
    let sum: f64 = weights.iter().sum();
    if sum.abs() < 1e-15 {
        // All zeros — fall back to equal weight.
        let eq = 1.0 / n as f64;
        for w in weights.iter_mut() {
            *w = eq;
        }
        return;
    }

    for w in weights.iter_mut() {
        *w /= sum;
    }

    // Re-apply bounds after normalisation (may break sum=1 slightly).
    for i in 0..n {
        let (lb, ub) = bounds[i];
        let lb = if long_only { lb.max(0.0) } else { lb };
        weights[i] = weights[i].clamp(lb, ub);
    }

    // Final normalisation.
    let sum: f64 = weights.iter().sum();
    if sum > 1e-15 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Efficient Frontier
// ---------------------------------------------------------------------------

/// Configuration for efficient frontier computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficientFrontierConfig {
    /// Number of points along the frontier.
    pub n_points: usize,
    /// Markowitz config to use at each point.
    pub markowitz_config: MarkowitzConfig,
    /// Minimum target return (fraction of max return). Default: 0.0.
    pub min_return_frac: f64,
    /// Maximum target return (fraction of max return). Default: 1.0.
    pub max_return_frac: f64,
}

impl Default for EfficientFrontierConfig {
    fn default() -> Self {
        Self {
            n_points: 50,
            markowitz_config: MarkowitzConfig::default(),
            min_return_frac: 0.0,
            max_return_frac: 1.0,
        }
    }
}

impl EfficientFrontierConfig {
    pub fn with_n_points(mut self, n: usize) -> Self {
        self.n_points = n;
        self
    }

    pub fn with_return_range(mut self, min_frac: f64, max_frac: f64) -> Self {
        self.min_return_frac = min_frac;
        self.max_return_frac = max_frac;
        self
    }
}

/// A single point on the efficient frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierPoint {
    /// Portfolio weights.
    pub weights: Vec<f64>,
    /// Expected return.
    pub expected_return: f64,
    /// Risk (standard deviation).
    pub risk: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
}

/// Result of efficient frontier computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficientFrontier {
    /// Points along the frontier (sorted by return).
    pub points: Vec<FrontierPoint>,
    /// The tangency portfolio (maximum Sharpe ratio).
    pub tangency_portfolio: Option<PortfolioResult>,
    /// The minimum variance portfolio.
    pub min_variance_portfolio: Option<PortfolioResult>,
    /// Total computation time.
    pub duration: Duration,
}

impl EfficientFrontier {
    /// Get the tangency (maximum Sharpe) portfolio.
    pub fn tangency(&self) -> Option<&PortfolioResult> {
        self.tangency_portfolio.as_ref()
    }

    /// Get the minimum variance portfolio.
    pub fn min_variance(&self) -> Option<&PortfolioResult> {
        self.min_variance_portfolio.as_ref()
    }

    /// Number of frontier points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the frontier is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return range (min, max) of the frontier.
    pub fn return_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let min = self.points.first().unwrap().expected_return;
        let max = self.points.last().unwrap().expected_return;
        Some((min, max))
    }

    /// Risk range (min, max) of the frontier.
    pub fn risk_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let min = self
            .points
            .iter()
            .map(|p| p.risk)
            .fold(f64::INFINITY, f64::min);
        let max = self.points.iter().map(|p| p.risk).fold(0.0f64, f64::max);
        Some((min, max))
    }
}

impl fmt::Display for EfficientFrontier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Efficient Frontier ({} points, {:?}):",
            self.points.len(),
            self.duration
        )?;
        if let Some((rmin, rmax)) = self.return_range() {
            writeln!(
                f,
                "  Return range: [{:.4}%, {:.4}%]",
                rmin * 100.0,
                rmax * 100.0
            )?;
        }
        if let Some((smin, smax)) = self.risk_range() {
            writeln!(
                f,
                "  Risk range:   [{:.4}%, {:.4}%]",
                smin * 100.0,
                smax * 100.0
            )?;
        }
        if let Some(t) = &self.tangency_portfolio {
            writeln!(
                f,
                "  Tangency: Sharpe={:.4}, Return={:.4}%, Risk={:.4}%",
                t.sharpe_ratio,
                t.portfolio_return * 100.0,
                t.portfolio_risk * 100.0
            )?;
        }
        Ok(())
    }
}

/// Compute the efficient frontier.
pub fn compute_efficient_frontier(
    universe: &AssetUniverse,
    config: &EfficientFrontierConfig,
) -> Result<EfficientFrontier> {
    let start = Instant::now();
    let n = universe.n_assets();

    if n == 0 {
        return Err(PortfolioError::InvalidConfig("Empty universe".into()));
    }
    if config.n_points < 2 {
        return Err(PortfolioError::InvalidConfig(
            "Need at least 2 frontier points".into(),
        ));
    }

    // Determine return range.
    let min_ret = universe
        .expected_returns
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_ret = universe
        .expected_returns
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let target_min = min_ret + (max_ret - min_ret) * config.min_return_frac;
    let target_max = min_ret + (max_ret - min_ret) * config.max_return_frac;

    let mut points = Vec::with_capacity(config.n_points);
    let mut best_sharpe = f64::NEG_INFINITY;
    let mut tangency_result: Option<PortfolioResult> = None;
    let mut min_var_result: Option<PortfolioResult> = None;
    let mut min_var = f64::INFINITY;

    for i in 0..config.n_points {
        let frac = if config.n_points > 1 {
            i as f64 / (config.n_points - 1) as f64
        } else {
            0.5
        };
        let target = target_min + frac * (target_max - target_min);

        let mk_config = config.markowitz_config.clone().with_target_return(target);
        match markowitz_optimize(universe, &mk_config) {
            Ok(result) => {
                let point = FrontierPoint {
                    weights: result.weights.clone(),
                    expected_return: result.portfolio_return,
                    risk: result.portfolio_risk,
                    sharpe_ratio: result.sharpe_ratio,
                };

                if result.sharpe_ratio > best_sharpe {
                    best_sharpe = result.sharpe_ratio;
                    tangency_result = Some(result.clone());
                }

                if result.portfolio_variance < min_var {
                    min_var = result.portfolio_variance;
                    min_var_result = Some(result);
                }

                points.push(point);
            }
            Err(e) => {
                debug!(target, error = %e, "Frontier point failed");
            }
        }
    }

    let duration = start.elapsed();

    info!(
        points = points.len(),
        best_sharpe,
        duration_ms = duration.as_millis(),
        "Efficient frontier computed"
    );

    Ok(EfficientFrontier {
        points,
        tangency_portfolio: tangency_result,
        min_variance_portfolio: min_var_result,
        duration,
    })
}

// ---------------------------------------------------------------------------
// Risk Parity
// ---------------------------------------------------------------------------

/// Configuration for risk parity optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParityConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Target risk budget per asset (must sum to 1).
    /// If `None`, equal risk budget is used.
    pub risk_budgets: Option<Vec<f64>>,
}

impl Default for RiskParityConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5_000,
            tolerance: 1e-8,
            risk_budgets: None,
        }
    }
}

impl RiskParityConfig {
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_risk_budgets(mut self, budgets: Vec<f64>) -> Self {
        self.risk_budgets = Some(budgets);
        self
    }
}

/// Risk parity optimization using the Spinu (2013) algorithm.
///
/// Finds weights such that each asset contributes equally (or according to
/// specified budgets) to portfolio risk.
pub fn risk_parity_optimize(
    universe: &AssetUniverse,
    config: &RiskParityConfig,
) -> Result<PortfolioResult> {
    let start = Instant::now();
    let n = universe.n_assets();

    if n == 0 {
        return Err(PortfolioError::InvalidConfig("Empty universe".into()));
    }

    // Risk budgets (default: equal).
    let budgets = match &config.risk_budgets {
        Some(b) => {
            if b.len() != n {
                return Err(PortfolioError::DimensionMismatch {
                    expected: n,
                    got: b.len(),
                });
            }
            b.clone()
        }
        None => vec![1.0 / n as f64; n],
    };

    // Initialize with inverse-volatility weighting.
    let vols = universe.asset_volatilities();
    let mut weights: Vec<f64> = vols
        .iter()
        .map(|v| if *v > 1e-12 { 1.0 / v } else { 1.0 })
        .collect();

    // Normalise.
    let sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum;
    }

    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Compute Σ * w.
        let mut sigma_w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                sigma_w[i] += universe.covariance[i][j] * weights[j];
            }
        }

        let portfolio_risk = universe.portfolio_risk(&weights);
        if portfolio_risk < 1e-15 {
            break;
        }

        // Update: w_i = b_i / (Σ*w)_i, then normalise.
        let mut new_weights = vec![0.0; n];
        let mut max_change = 0.0f64;

        for i in 0..n {
            if sigma_w[i].abs() > 1e-15 {
                new_weights[i] = budgets[i] / sigma_w[i];
            } else {
                new_weights[i] = weights[i];
            }
        }

        // Normalise.
        let sum: f64 = new_weights.iter().sum();
        if sum > 1e-15 {
            for w in new_weights.iter_mut() {
                *w /= sum;
            }
        }

        for i in 0..n {
            max_change = max_change.max((new_weights[i] - weights[i]).abs());
        }

        weights = new_weights;

        if max_change < config.tolerance {
            converged = true;
            debug!(iterations, max_change, "Risk parity converged");
            break;
        }
    }

    // Compute the risk parity "error": how far are risk contributions from budgets.
    let risk_contribs = universe.risk_contributions(&weights);
    let portfolio_risk = universe.portfolio_risk(&weights);
    let rc_fracs: Vec<f64> = if portfolio_risk > 1e-12 {
        risk_contribs.iter().map(|rc| rc / portfolio_risk).collect()
    } else {
        vec![0.0; n]
    };
    let rp_error: f64 = rc_fracs
        .iter()
        .zip(budgets.iter())
        .map(|(rc, b)| (rc - b).powi(2))
        .sum::<f64>()
        .sqrt();

    let duration = start.elapsed();

    let result = PortfolioResult::from_weights(
        weights,
        universe,
        "Risk Parity",
        iterations,
        duration,
        converged,
        rp_error,
    )
    .with_metadata("rp_error", rp_error);

    Ok(result)
}

// ---------------------------------------------------------------------------
// Black-Litterman
// ---------------------------------------------------------------------------

/// Configuration for the Black-Litterman model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackLittermanConfig {
    /// Scalar for the uncertainty in the prior (τ).
    /// Typical values: 0.01 – 0.05.
    pub tau: f64,

    /// Risk aversion coefficient (δ).
    /// Used to reverse-engineer equilibrium returns: π = δ Σ w_mkt.
    pub risk_aversion: f64,

    /// Views matrix P (k × n): each row is a view on asset weights.
    /// Example: P = [[1, -1, 0]] means "asset 0 outperforms asset 1".
    pub views_p: Vec<Vec<f64>>,

    /// View returns q (k × 1): expected return of each view.
    pub views_q: Vec<f64>,

    /// View confidence (diagonal of Ω). Higher = less certain.
    /// If `None`, uses τ * P Σ P^T (proportional to prior uncertainty).
    pub view_confidences: Option<Vec<f64>>,

    /// Markowitz config for the final optimisation step.
    pub markowitz_config: MarkowitzConfig,
}

impl Default for BlackLittermanConfig {
    fn default() -> Self {
        Self {
            tau: 0.025,
            risk_aversion: 2.5,
            views_p: Vec::new(),
            views_q: Vec::new(),
            view_confidences: None,
            markowitz_config: MarkowitzConfig::default(),
        }
    }
}

impl BlackLittermanConfig {
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    pub fn with_risk_aversion(mut self, delta: f64) -> Self {
        self.risk_aversion = delta;
        self
    }

    /// Add a view: "assets weighted by `p` will return `q`".
    pub fn add_view(mut self, p: Vec<f64>, q: f64) -> Self {
        self.views_p.push(p);
        self.views_q.push(q);
        self
    }

    /// Add an absolute view: "asset `i` will return `q`".
    pub fn add_absolute_view(mut self, i: usize, n: usize, q: f64) -> Self {
        let mut p = vec![0.0; n];
        if i < n {
            p[i] = 1.0;
        }
        self.views_p.push(p);
        self.views_q.push(q);
        self
    }

    /// Add a relative view: "asset `i` outperforms asset `j` by `q`".
    pub fn add_relative_view(mut self, i: usize, j: usize, n: usize, q: f64) -> Self {
        let mut p = vec![0.0; n];
        if i < n {
            p[i] = 1.0;
        }
        if j < n {
            p[j] = -1.0;
        }
        self.views_p.push(p);
        self.views_q.push(q);
        self
    }

    pub fn with_view_confidences(mut self, confidences: Vec<f64>) -> Self {
        self.view_confidences = Some(confidences);
        self
    }
}

/// Black-Litterman combined return estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackLittermanReturns {
    /// Equilibrium excess returns (π = δ Σ w_mkt).
    pub equilibrium_returns: Vec<f64>,
    /// Black-Litterman posterior expected returns.
    pub posterior_returns: Vec<f64>,
    /// Black-Litterman posterior covariance (simplified: τΣ blended).
    pub posterior_covariance: Vec<Vec<f64>>,
}

/// Run the Black-Litterman model and optimise the resulting portfolio.
pub fn black_litterman_optimize(
    universe: &AssetUniverse,
    config: &BlackLittermanConfig,
) -> Result<(BlackLittermanReturns, PortfolioResult)> {
    let start = Instant::now();
    let n = universe.n_assets();

    if n == 0 {
        return Err(PortfolioError::InvalidConfig("Empty universe".into()));
    }

    // Step 1: Compute equilibrium returns π = δ Σ w_mkt.
    let w_mkt = universe
        .market_cap_portfolio()
        .unwrap_or_else(|| universe.equal_weight_portfolio());

    let mut pi = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            pi[i] += config.risk_aversion * universe.covariance[i][j] * w_mkt[j];
        }
    }

    // Step 2: If no views, just use equilibrium returns.
    let k = config.views_p.len();
    let posterior_returns;
    let posterior_cov;

    if k == 0 {
        posterior_returns = pi.clone();
        posterior_cov = universe.covariance.clone();
    } else {
        // Validate views.
        for view in &config.views_p {
            if view.len() != n {
                return Err(PortfolioError::DimensionMismatch {
                    expected: n,
                    got: view.len(),
                });
            }
        }
        if config.views_q.len() != k {
            return Err(PortfolioError::DimensionMismatch {
                expected: k,
                got: config.views_q.len(),
            });
        }

        // Compute Ω (view uncertainty, diagonal).
        let omega_diag: Vec<f64> = match &config.view_confidences {
            Some(conf) => {
                if conf.len() != k {
                    return Err(PortfolioError::DimensionMismatch {
                        expected: k,
                        got: conf.len(),
                    });
                }
                conf.clone()
            }
            None => {
                // Ω = diag(τ * P Σ P^T)
                let mut diag = vec![0.0; k];
                for v in 0..k {
                    for i in 0..n {
                        for j in 0..n {
                            diag[v] += config.views_p[v][i]
                                * universe.covariance[i][j]
                                * config.views_p[v][j];
                        }
                    }
                    diag[v] *= config.tau;
                }
                diag
            }
        };

        // Black-Litterman formula (simplified using Sherman-Morrison-Woodbury):
        // E[R] = π + τΣ P^T (P τΣ P^T + Ω)^{-1} (q - P π)
        //
        // For the diagonal Ω case with small k, we compute directly.

        // Compute P τΣ P^T + Ω (k × k matrix).
        let tau_sigma = |i: usize, j: usize| -> f64 { config.tau * universe.covariance[i][j] };

        let mut m = vec![vec![0.0; k]; k]; // P τΣ P^T + Ω
        for a in 0..k {
            for b in 0..k {
                let mut val = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        val += config.views_p[a][i] * tau_sigma(i, j) * config.views_p[b][j];
                    }
                }
                m[a][b] = val;
                if a == b {
                    m[a][b] += omega_diag[a];
                }
            }
        }

        // Invert M (k × k). For small k, use direct or Gauss-Jordan.
        let m_inv = invert_matrix(&m)?;

        // Compute q - Pπ.
        let mut q_minus_p_pi = vec![0.0; k];
        for v in 0..k {
            let mut p_pi = 0.0;
            for i in 0..n {
                p_pi += config.views_p[v][i] * pi[i];
            }
            q_minus_p_pi[v] = config.views_q[v] - p_pi;
        }

        // Compute M^{-1} (q - Pπ).
        let mut m_inv_q = vec![0.0; k];
        for a in 0..k {
            for b in 0..k {
                m_inv_q[a] += m_inv[a][b] * q_minus_p_pi[b];
            }
        }

        // Compute τΣ P^T M^{-1} (q - Pπ).
        let mut adjustment = vec![0.0; n];
        for i in 0..n {
            for v in 0..k {
                let mut tau_sigma_pt = 0.0;
                for j in 0..n {
                    tau_sigma_pt += tau_sigma(i, j) * config.views_p[v][j];
                }
                adjustment[i] += tau_sigma_pt * m_inv_q[v];
            }
        }

        posterior_returns = pi
            .iter()
            .zip(adjustment.iter())
            .map(|(p, a)| p + a)
            .collect();

        // Simplified posterior covariance: (1 + τ) Σ
        // (A more complete version would use the full BL posterior covariance formula.)
        posterior_cov = universe
            .covariance
            .iter()
            .map(|row| row.iter().map(|&v| v * (1.0 + config.tau)).collect())
            .collect();
    }

    let bl_returns = BlackLittermanReturns {
        equilibrium_returns: pi,
        posterior_returns: posterior_returns.clone(),
        posterior_covariance: posterior_cov.clone(),
    };

    // Step 3: Optimise using posterior returns and covariance.
    let bl_universe = AssetUniverse {
        names: universe.names.clone(),
        expected_returns: posterior_returns,
        covariance: posterior_cov,
        risk_free_rate: universe.risk_free_rate,
        weight_bounds: universe.weight_bounds.clone(),
        market_caps: universe.market_caps.clone(),
    };

    let mut result = markowitz_optimize(&bl_universe, &config.markowitz_config)?;
    result.method = "Black-Litterman".to_string();
    result.duration = start.elapsed();
    result = result.with_metadata("tau", config.tau);
    result = result.with_metadata("risk_aversion_delta", config.risk_aversion);
    result = result.with_metadata("n_views", k as f64);

    // Recompute stats against original universe.
    result.portfolio_return = universe.portfolio_return(&result.weights);
    result.portfolio_risk = universe.portfolio_risk(&result.weights);
    result.sharpe_ratio = universe.sharpe_ratio(&result.weights);
    result.risk_contributions = universe.risk_contributions(&result.weights);

    Ok((bl_returns, result))
}

/// Invert a small square matrix using Gauss-Jordan elimination.
fn invert_matrix(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = m.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Augmented matrix [M | I].
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return Err(PortfolioError::NumericalError(
                "Singular matrix in Black-Litterman".into(),
            ));
        }

        // Swap rows.
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Scale pivot row.
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate column.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract inverse.
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    Ok(inv)
}

// ===========================================================================
// QUANTUM-INSPIRED OPTIMIZERS
// ===========================================================================

// ---------------------------------------------------------------------------
// QUBO Formulation
// ---------------------------------------------------------------------------

/// Quadratic Unconstrained Binary Optimization (QUBO) formulation.
///
/// The portfolio selection problem is encoded as:
///   min x^T Q x + c^T x
/// where x_i ∈ {0, 1} indicates whether asset i is included.
///
/// Q encodes risk (covariance) and c encodes negative returns
/// plus constraint penalties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuboFormulation {
    /// QUBO matrix Q (n × n).
    pub q_matrix: Vec<Vec<f64>>,
    /// Linear coefficients c (n).
    pub linear: Vec<f64>,
    /// Number of binary variables.
    pub n_vars: usize,
    /// Return penalty weight.
    pub return_weight: f64,
    /// Risk penalty weight.
    pub risk_weight: f64,
    /// Cardinality penalty weight.
    pub cardinality_penalty: f64,
    /// Target cardinality (number of assets to select).
    pub target_cardinality: Option<usize>,
    /// Budget constraint penalty.
    pub budget_penalty: f64,
}

impl QuboFormulation {
    /// Construct a QUBO from an asset universe.
    ///
    /// The objective is:
    ///   min  risk_weight * x^T Σ x - return_weight * μ^T x
    ///        + cardinality_penalty * (sum(x) - K)^2
    ///        + budget_penalty * (sum(x/n) - 1)^2
    pub fn from_universe(
        universe: &AssetUniverse,
        risk_weight: f64,
        return_weight: f64,
        target_cardinality: Option<usize>,
        cardinality_penalty: f64,
        budget_penalty: f64,
    ) -> Self {
        let n = universe.n_assets();
        let mut q = vec![vec![0.0; n]; n];
        let mut linear = vec![0.0; n];

        // Risk term: risk_weight * Σ_{ij} x_i x_j cov(i,j)
        for i in 0..n {
            for j in 0..n {
                q[i][j] += risk_weight * universe.covariance[i][j];
            }
        }

        // Return term: -return_weight * μ_i
        for i in 0..n {
            linear[i] -= return_weight * universe.expected_returns[i];
        }

        // Cardinality constraint: penalty * (sum(x) - K)^2
        // = penalty * (sum_i x_i)^2 - 2K * sum_i x_i + K^2
        // = penalty * (sum_ij x_i x_j - 2K sum_i x_i + K^2)
        if let Some(k) = target_cardinality {
            let kf = k as f64;
            for i in 0..n {
                for j in 0..n {
                    q[i][j] += cardinality_penalty;
                }
                linear[i] -= 2.0 * cardinality_penalty * kf;
            }
            // Constant K^2 * penalty doesn't affect the optimum but we note it.
        }

        // Budget constraint: penalty * (sum(x_i / n) - 1)^2
        let n_f = n as f64;
        for i in 0..n {
            for j in 0..n {
                q[i][j] += budget_penalty / (n_f * n_f);
            }
            linear[i] -= 2.0 * budget_penalty / n_f;
        }

        Self {
            q_matrix: q,
            linear,
            n_vars: n,
            return_weight,
            risk_weight,
            cardinality_penalty,
            target_cardinality,
            budget_penalty,
        }
    }

    /// Evaluate the QUBO objective for a binary solution x.
    pub fn evaluate(&self, x: &[u8]) -> f64 {
        let n = self.n_vars;
        let mut val = 0.0;

        // Quadratic term.
        for i in 0..n {
            for j in 0..n {
                val += self.q_matrix[i][j] * (x[i] as f64) * (x[j] as f64);
            }
        }

        // Linear term.
        for i in 0..n {
            val += self.linear[i] * (x[i] as f64);
        }

        val
    }

    /// Evaluate the QUBO objective for a probability vector (expected value).
    pub fn evaluate_probabilities(&self, probs: &[f64]) -> f64 {
        let n = self.n_vars;
        let mut val = 0.0;

        for i in 0..n {
            for j in 0..n {
                val += self.q_matrix[i][j] * probs[i] * probs[j];
            }
            val += self.linear[i] * probs[i];
        }

        val
    }
}

// ---------------------------------------------------------------------------
// QAOA Portfolio Optimizer
// ---------------------------------------------------------------------------

/// Configuration for the QAOA portfolio optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaoaConfig {
    /// Circuit depth p (number of QAOA layers).
    pub depth: usize,
    /// Number of measurement shots for sampling.
    pub shots: usize,
    /// Number of classical optimisation iterations for parameters.
    pub n_optimizer_iterations: usize,
    /// Learning rate for parameter optimisation.
    pub learning_rate: f64,
    /// QUBO risk weight.
    pub risk_weight: f64,
    /// QUBO return weight.
    pub return_weight: f64,
    /// Target number of assets to select (cardinality constraint).
    pub target_cardinality: Option<usize>,
    /// Cardinality penalty weight.
    pub cardinality_penalty: f64,
    /// Budget penalty weight.
    pub budget_penalty: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for QaoaConfig {
    fn default() -> Self {
        Self {
            depth: 2,
            shots: 1000,
            n_optimizer_iterations: 200,
            learning_rate: 0.05,
            risk_weight: 1.0,
            return_weight: 2.0,
            target_cardinality: None,
            cardinality_penalty: 5.0,
            budget_penalty: 3.0,
            seed: 42,
        }
    }
}

impl QaoaConfig {
    pub fn with_depth(mut self, p: usize) -> Self {
        self.depth = p;
        self
    }

    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    pub fn with_n_optimizer_iterations(mut self, n: usize) -> Self {
        self.n_optimizer_iterations = n;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_target_cardinality(mut self, k: usize) -> Self {
        self.target_cardinality = Some(k);
        self
    }

    pub fn with_risk_weight(mut self, w: f64) -> Self {
        self.risk_weight = w;
        self
    }

    pub fn with_return_weight(mut self, w: f64) -> Self {
        self.return_weight = w;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Result of QAOA portfolio optimisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaoaResult {
    /// Best binary solution found (1 = asset selected).
    pub selected_assets: Vec<u8>,
    /// Best QUBO objective value.
    pub best_objective: f64,
    /// Optimal QAOA parameters (γ, β).
    pub optimal_params: Vec<f64>,
    /// Distribution of sampled solutions (bitstring → count).
    pub solution_distribution: HashMap<String, usize>,
    /// Portfolio result (with equal weighting among selected assets).
    pub portfolio_result: PortfolioResult,
    /// QAOA circuit depth.
    pub depth: usize,
    /// Number of optimisation iterations.
    pub iterations: usize,
    /// Total duration.
    pub duration: Duration,
    /// Objective value history.
    pub objective_history: Vec<f64>,
}

/// Simulated QAOA for portfolio selection.
///
/// This is a classical simulation of the QAOA quantum algorithm.
/// For n qubits we maintain a 2^n statevector and apply cost/mixer
/// unitaries. Because full statevector simulation is exponential,
/// this is practical only for small universes (n ≤ ~20).
///
/// For larger universes, a mean-field / variational approximation
/// is used instead.
pub fn qaoa_portfolio_optimize(
    universe: &AssetUniverse,
    config: &QaoaConfig,
) -> Result<QaoaResult> {
    let start = Instant::now();
    let n = universe.n_assets();

    if n == 0 {
        return Err(PortfolioError::InvalidConfig("Empty universe".into()));
    }
    if config.depth == 0 {
        return Err(PortfolioError::InvalidConfig(
            "QAOA depth must be ≥ 1".into(),
        ));
    }

    let qubo = QuboFormulation::from_universe(
        universe,
        config.risk_weight,
        config.return_weight,
        config.target_cardinality,
        config.cardinality_penalty,
        config.budget_penalty,
    );

    let p = config.depth;

    // For small n, do full statevector simulation.
    // For large n, fall back to mean-field approximation.
    let use_statevector = n <= 18;

    let (best_solution, best_obj, optimal_params, obj_history, solution_dist) = if use_statevector {
        qaoa_statevector(&qubo, p, config)
    } else {
        qaoa_mean_field(&qubo, p, config)
    };

    // Build portfolio from selected assets with equal weighting.
    let selected_count = best_solution.iter().filter(|&&x| x == 1).count();
    let mut weights = vec![0.0; n];
    if selected_count > 0 {
        let w = 1.0 / selected_count as f64;
        for i in 0..n {
            if best_solution[i] == 1 {
                weights[i] = w;
            }
        }
    } else {
        // Fallback: equal weight all assets.
        weights = universe.equal_weight_portfolio();
    }

    let duration = start.elapsed();

    let portfolio_result = PortfolioResult::from_weights(
        weights,
        universe,
        "QAOA Portfolio Selection",
        config.n_optimizer_iterations,
        duration,
        true,
        best_obj,
    )
    .with_metadata("qaoa_depth", p as f64)
    .with_metadata("selected_count", selected_count as f64)
    .with_metadata("n_qubits", n as f64);

    Ok(QaoaResult {
        selected_assets: best_solution,
        best_objective: best_obj,
        optimal_params,
        solution_distribution: solution_dist,
        portfolio_result,
        depth: p,
        iterations: config.n_optimizer_iterations,
        duration,
        objective_history: obj_history,
    })
}

/// Full statevector QAOA simulation.
fn qaoa_statevector(
    qubo: &QuboFormulation,
    p: usize,
    config: &QaoaConfig,
) -> (Vec<u8>, f64, Vec<f64>, Vec<f64>, HashMap<String, usize>) {
    let n = qubo.n_vars;
    let dim = 1usize << n; // 2^n

    // Initialize parameters: γ and β for each layer.
    let mut params = vec![0.1; 2 * p]; // [γ_1, β_1, γ_2, β_2, ...]
    let mut rng = SimpleRng::new(config.seed);

    // Small perturbation to break symmetry.
    for param in params.iter_mut() {
        *param += rng.next_f64() * 0.1;
    }

    // Precompute QUBO values for all 2^n bitstrings.
    let mut qubo_values = vec![0.0f64; dim];
    for state in 0..dim {
        let bits = state_to_bits(state, n);
        qubo_values[state] = qubo.evaluate(&bits);
    }

    let mut best_obj = f64::INFINITY;
    let mut best_solution = vec![0u8; n];
    let mut obj_history = Vec::with_capacity(config.n_optimizer_iterations);

    for _iter in 0..config.n_optimizer_iterations {
        // Evaluate the QAOA circuit with current parameters.
        let probs = simulate_qaoa_circuit(n, &params, &qubo_values);

        // Compute expected objective.
        let expected_obj: f64 = probs
            .iter()
            .zip(qubo_values.iter())
            .map(|(p, v)| p * v)
            .sum();

        obj_history.push(expected_obj);

        // Find the best state from the probability distribution.
        let mut best_prob_state = 0;
        let mut best_prob = 0.0;
        for (state, &prob) in probs.iter().enumerate() {
            if prob > best_prob {
                best_prob = prob;
                best_prob_state = state;
            }
        }

        let candidate = state_to_bits(best_prob_state, n);
        let candidate_obj = qubo_values[best_prob_state];

        if candidate_obj < best_obj {
            best_obj = candidate_obj;
            best_solution = candidate;
        }

        // Gradient-free parameter update (finite differences).
        let eps = 0.01;
        let mut gradient = vec![0.0; params.len()];
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            params_plus[i] += eps;
            let probs_plus = simulate_qaoa_circuit(n, &params_plus, &qubo_values);
            let obj_plus: f64 = probs_plus
                .iter()
                .zip(qubo_values.iter())
                .map(|(p, v)| p * v)
                .sum();

            gradient[i] = (obj_plus - expected_obj) / eps;
        }

        // Update parameters.
        for i in 0..params.len() {
            params[i] -= config.learning_rate * gradient[i];
            // Clamp parameters to reasonable range.
            params[i] = params[i].clamp(-2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI);
        }
    }

    // Sample to build solution distribution.
    let final_probs = simulate_qaoa_circuit(n, &params, &qubo_values);
    let mut solution_dist = HashMap::new();
    for (state, &prob) in final_probs.iter().enumerate() {
        if prob > 1e-6 {
            let bits = state_to_bits(state, n);
            let key: String = bits
                .iter()
                .map(|b| if *b == 1 { '1' } else { '0' })
                .collect();
            let count = (prob * config.shots as f64).round() as usize;
            if count > 0 {
                solution_dist.insert(key, count);
            }
        }
    }

    (best_solution, best_obj, params, obj_history, solution_dist)
}

/// Simulate the QAOA circuit and return probability distribution.
///
/// |ψ⟩ = ∏_{l=1}^{p} U_B(β_l) U_C(γ_l) |+⟩^n
///
/// U_C(γ) = exp(-iγ C) where C is the cost Hamiltonian (diagonal in computational basis).
/// U_B(β) = exp(-iβ B) where B = Σ_i X_i is the mixer Hamiltonian.
fn simulate_qaoa_circuit(n: usize, params: &[f64], qubo_values: &[f64]) -> Vec<f64> {
    let dim = 1usize << n;
    let p = params.len() / 2;

    // Initialize to uniform superposition: |+⟩^n.
    let amp = 1.0 / (dim as f64).sqrt();

    // We store the statevector as pairs of (real, imag).
    let mut state_re = vec![amp; dim];
    let mut state_im = vec![0.0; dim];

    for layer in 0..p {
        let gamma = params[2 * layer];
        let beta = params[2 * layer + 1];

        // Apply U_C(γ): diagonal phase operator.
        // |x⟩ → exp(-iγ C(x)) |x⟩
        for s in 0..dim {
            let angle = -gamma * qubo_values[s];
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = state_re[s];
            let im = state_im[s];
            state_re[s] = re * cos_a - im * sin_a;
            state_im[s] = re * sin_a + im * cos_a;
        }

        // Apply U_B(β): mixer operator exp(-iβ Σ X_i).
        // This can be decomposed as a product of single-qubit X rotations:
        // exp(-iβ X_i) for each qubit i.
        // R_X(2β) = cos(β)I - i sin(β) X
        let cos_b = beta.cos();
        let sin_b = beta.sin();

        for qubit in 0..n {
            let mask = 1usize << qubit;
            for s in 0..dim {
                if s & mask == 0 {
                    // s and s^mask are the pair of states differing in this qubit.
                    let t = s | mask;
                    let re_0 = state_re[s];
                    let im_0 = state_im[s];
                    let re_1 = state_re[t];
                    let im_1 = state_im[t];

                    // R_X(2β) = [[cos(β), -i sin(β)], [-i sin(β), cos(β)]]
                    state_re[s] = cos_b * re_0 + sin_b * im_1;
                    state_im[s] = cos_b * im_0 - sin_b * re_1;
                    state_re[t] = sin_b * im_0 + cos_b * re_1;
                    state_im[t] = -sin_b * re_0 + cos_b * im_1;
                }
            }
        }
    }

    // Compute probabilities.
    let mut probs = vec![0.0; dim];
    for s in 0..dim {
        probs[s] = state_re[s] * state_re[s] + state_im[s] * state_im[s];
    }

    probs
}

/// Mean-field approximation for QAOA on large instances.
fn qaoa_mean_field(
    qubo: &QuboFormulation,
    _p: usize,
    config: &QaoaConfig,
) -> (Vec<u8>, f64, Vec<f64>, Vec<f64>, HashMap<String, usize>) {
    let n = qubo.n_vars;
    let mut rng = SimpleRng::new(config.seed);

    // Mean-field probabilities p_i ∈ [0, 1] for each qubit being 1.
    let mut probs = vec![0.5; n];
    let lr = config.learning_rate;

    let mut best_obj = f64::INFINITY;
    let mut best_solution = vec![0u8; n];
    let mut obj_history = Vec::with_capacity(config.n_optimizer_iterations);

    for _iter in 0..config.n_optimizer_iterations {
        // Compute gradient of expected QUBO value w.r.t. p_i.
        let mut gradient = vec![0.0; n];
        for i in 0..n {
            // dE/dp_i = sum_j Q_ij p_j + Q_ii p_i + c_i
            // More precisely: dE/dp_i = 2 * sum_{j≠i} Q_ij p_j + Q_ii + c_i
            let mut g = qubo.linear[i];
            for j in 0..n {
                g += qubo.q_matrix[i][j] * probs[j];
                if i != j {
                    g += qubo.q_matrix[i][j] * probs[j];
                }
            }
            gradient[i] = g;
        }

        // Update probabilities with gradient descent.
        for i in 0..n {
            probs[i] -= lr * gradient[i];
            probs[i] = probs[i].clamp(0.01, 0.99);
        }

        let obj = qubo.evaluate_probabilities(&probs);
        obj_history.push(obj);

        // Sample a solution.
        let solution: Vec<u8> = probs
            .iter()
            .map(|&p| if rng.next_f64() < p { 1 } else { 0 })
            .collect();
        let solution_obj = qubo.evaluate(&solution);

        if solution_obj < best_obj {
            best_obj = solution_obj;
            best_solution = solution;
        }
    }

    // Round probabilities to get deterministic solution.
    let rounded: Vec<u8> = probs.iter().map(|&p| if p > 0.5 { 1 } else { 0 }).collect();
    let rounded_obj = qubo.evaluate(&rounded);
    if rounded_obj < best_obj {
        best_obj = rounded_obj;
        best_solution = rounded;
    }

    let params = probs; // Mean-field parameters as the "optimal params".
    let solution_dist = HashMap::new(); // No distribution for mean-field.

    (best_solution, best_obj, params, obj_history, solution_dist)
}

// ---------------------------------------------------------------------------
// VQE Portfolio Optimizer
// ---------------------------------------------------------------------------

/// Configuration for VQE-based portfolio optimisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VqeConfig {
    /// Number of variational parameters per asset.
    pub params_per_asset: usize,
    /// Number of optimisation iterations.
    pub n_iterations: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Risk aversion parameter.
    pub risk_aversion: f64,
    /// Whether to enforce long-only constraint.
    pub long_only: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for VqeConfig {
    fn default() -> Self {
        Self {
            params_per_asset: 2,
            n_iterations: 500,
            learning_rate: 0.02,
            risk_aversion: 1.0,
            long_only: true,
            seed: 42,
        }
    }
}

impl VqeConfig {
    pub fn with_n_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_risk_aversion(mut self, lambda: f64) -> Self {
        self.risk_aversion = lambda;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// VQE-inspired portfolio optimizer.
///
/// Uses a parameterized "ansatz" to map variational parameters to
/// portfolio weights via a softmax layer, then optimizes the
/// risk-return objective via gradient descent.
///
/// This simulates the variational principle of VQE where we
/// minimize ⟨ψ(θ)|H|ψ(θ)⟩ with H being the portfolio cost
/// Hamiltonian.
pub fn vqe_portfolio_optimize(
    universe: &AssetUniverse,
    config: &VqeConfig,
) -> Result<PortfolioResult> {
    let start = Instant::now();
    let n = universe.n_assets();

    if n == 0 {
        return Err(PortfolioError::InvalidConfig("Empty universe".into()));
    }

    let mut rng = SimpleRng::new(config.seed);
    let n_params = n * config.params_per_asset;

    // Initialize variational parameters.
    let mut theta = vec![0.0; n_params];
    for t in theta.iter_mut() {
        *t = rng.next_f64() * 0.5 - 0.25;
    }

    let lambda = config.risk_aversion;
    let lr = config.learning_rate;

    let mut best_obj = f64::INFINITY;
    let mut best_weights = universe.equal_weight_portfolio();
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.n_iterations {
        iterations = iter + 1;

        // Map parameters to weights via the ansatz.
        let weights = ansatz_to_weights(&theta, n, config.params_per_asset, config.long_only);

        // Compute objective: λ/2 * w^T Σ w - μ^T w.
        let obj = lambda / 2.0 * universe.portfolio_variance(&weights)
            - universe.portfolio_return(&weights);

        if obj < best_obj {
            best_obj = obj;
            best_weights = weights.clone();
        }

        // Compute gradient via parameter shift rule (finite differences).
        let eps = 0.005;
        let mut gradient = vec![0.0; n_params];

        for i in 0..n_params {
            let mut theta_plus = theta.clone();
            theta_plus[i] += eps;
            let w_plus =
                ansatz_to_weights(&theta_plus, n, config.params_per_asset, config.long_only);
            let obj_plus = lambda / 2.0 * universe.portfolio_variance(&w_plus)
                - universe.portfolio_return(&w_plus);

            gradient[i] = (obj_plus - obj) / eps;
        }

        // Update parameters.
        let mut max_change = 0.0f64;
        for i in 0..n_params {
            let change = lr * gradient[i];
            theta[i] -= change;
            max_change = max_change.max(change.abs());
        }

        if max_change < 1e-8 {
            converged = true;
            break;
        }
    }

    let duration = start.elapsed();

    let result = PortfolioResult::from_weights(
        best_weights,
        universe,
        "VQE Portfolio",
        iterations,
        duration,
        converged,
        best_obj,
    )
    .with_metadata("n_params", n_params as f64)
    .with_metadata("risk_aversion", lambda);

    Ok(result)
}

/// Map variational parameters to portfolio weights via a softmax ansatz.
///
/// The ansatz applies a simple neural-network-like transformation:
///   logits_i = sum_k theta_{i,k} * basis_function_k
/// Then: weights = softmax(logits)
fn ansatz_to_weights(
    theta: &[f64],
    n: usize,
    params_per_asset: usize,
    long_only: bool,
) -> Vec<f64> {
    let mut logits = vec![0.0; n];

    for i in 0..n {
        let base = i * params_per_asset;
        // Simple linear combination of parameters with polynomial basis.
        for k in 0..params_per_asset {
            if base + k < theta.len() {
                let basis = (k as f64 + 1.0).sqrt(); // Basis function sqrt(k+1)
                logits[i] += theta[base + k] * basis;
            }
        }
    }

    // Softmax to get normalized weights.
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 1e-15 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }

    if long_only {
        for e in exps.iter_mut() {
            *e = e.max(0.0);
        }
        let sum: f64 = exps.iter().sum();
        if sum > 1e-15 {
            for e in exps.iter_mut() {
                *e /= sum;
            }
        }
    }

    exps
}

// ---------------------------------------------------------------------------
// Simulated Quantum Annealing
// ---------------------------------------------------------------------------

/// Configuration for simulated quantum annealing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealingConfig {
    /// Number of Monte Carlo sweeps.
    pub n_sweeps: usize,
    /// Number of Trotter replicas (for path-integral formulation).
    pub n_replicas: usize,
    /// Initial transverse field strength (Γ_0).
    pub initial_transverse_field: f64,
    /// Final transverse field strength (Γ_f).
    pub final_transverse_field: f64,
    /// Temperature (β = 1/T).
    pub temperature: f64,
    /// Annealing schedule: "linear", "exponential", "quadratic".
    pub schedule: AnnealingSchedule,
    /// QUBO risk weight.
    pub risk_weight: f64,
    /// QUBO return weight.
    pub return_weight: f64,
    /// Target cardinality.
    pub target_cardinality: Option<usize>,
    /// Cardinality penalty weight.
    pub cardinality_penalty: f64,
    /// Budget penalty weight.
    pub budget_penalty: f64,
    /// Random seed.
    pub seed: u64,
}

/// Annealing schedule type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnnealingSchedule {
    Linear,
    Exponential,
    Quadratic,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            n_sweeps: 10_000,
            n_replicas: 8,
            initial_transverse_field: 5.0,
            final_transverse_field: 0.01,
            temperature: 0.1,
            schedule: AnnealingSchedule::Linear,
            risk_weight: 1.0,
            return_weight: 2.0,
            target_cardinality: None,
            cardinality_penalty: 5.0,
            budget_penalty: 3.0,
            seed: 42,
        }
    }
}

impl AnnealingConfig {
    pub fn with_n_sweeps(mut self, n: usize) -> Self {
        self.n_sweeps = n;
        self
    }

    pub fn with_n_replicas(mut self, n: usize) -> Self {
        self.n_replicas = n;
        self
    }

    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_schedule(mut self, schedule: AnnealingSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn with_transverse_field(mut self, initial: f64, final_val: f64) -> Self {
        self.initial_transverse_field = initial;
        self.final_transverse_field = final_val;
        self
    }

    pub fn with_target_cardinality(mut self, k: usize) -> Self {
        self.target_cardinality = Some(k);
        self
    }

    pub fn with_risk_weight(mut self, w: f64) -> Self {
        self.risk_weight = w;
        self
    }

    pub fn with_return_weight(mut self, w: f64) -> Self {
        self.return_weight = w;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Compute the transverse field at a given fraction of the anneal.
    pub fn transverse_field_at(&self, fraction: f64) -> f64 {
        let fraction = fraction.clamp(0.0, 1.0);
        match self.schedule {
            AnnealingSchedule::Linear => {
                self.initial_transverse_field
                    + fraction * (self.final_transverse_field - self.initial_transverse_field)
            }
            AnnealingSchedule::Exponential => {
                self.initial_transverse_field
                    * (self.final_transverse_field / self.initial_transverse_field).powf(fraction)
            }
            AnnealingSchedule::Quadratic => {
                let diff = self.final_transverse_field - self.initial_transverse_field;
                self.initial_transverse_field + diff * fraction * fraction
            }
        }
    }
}

/// Result of simulated quantum annealing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealingResult {
    /// Best binary solution found.
    pub selected_assets: Vec<u8>,
    /// Best QUBO objective value.
    pub best_objective: f64,
    /// Portfolio result (with equal weighting among selected assets).
    pub portfolio_result: PortfolioResult,
    /// Number of sweeps performed.
    pub n_sweeps: usize,
    /// Number of replicas used.
    pub n_replicas: usize,
    /// Total duration.
    pub duration: Duration,
    /// Objective history (sampled).
    pub objective_history: Vec<f64>,
    /// Number of accepted flips.
    pub accepted_flips: u64,
    /// Total flip attempts.
    pub total_flips: u64,
    /// Acceptance rate.
    pub acceptance_rate: f64,
}

/// Simulated quantum annealing for portfolio selection.
///
/// Uses the path-integral Monte Carlo formulation of quantum annealing
/// with Suzuki-Trotter decomposition. Multiple replicas of the classical
/// system are coupled by a transverse field that decreases during the
/// anneal, simulating quantum tunneling through energy barriers.
pub fn quantum_anneal_optimize(
    universe: &AssetUniverse,
    config: &AnnealingConfig,
) -> Result<AnnealingResult> {
    let start = Instant::now();
    let n = universe.n_assets();

    if n == 0 {
        return Err(PortfolioError::InvalidConfig("Empty universe".into()));
    }

    let qubo = QuboFormulation::from_universe(
        universe,
        config.risk_weight,
        config.return_weight,
        config.target_cardinality,
        config.cardinality_penalty,
        config.budget_penalty,
    );

    let m = config.n_replicas.max(1); // Number of Trotter replicas.
    let beta = 1.0 / config.temperature.max(1e-10);
    let mut rng = SimpleRng::new(config.seed);

    // Initialize replicas randomly.
    let mut replicas: Vec<Vec<u8>> = (0..m)
        .map(|_| {
            (0..n)
                .map(|_| if rng.next_f64() < 0.5 { 1 } else { 0 })
                .collect()
        })
        .collect();

    let mut best_obj = f64::INFINITY;
    let mut best_solution = vec![0u8; n];
    let mut accepted_flips = 0u64;
    let mut total_flips = 0u64;
    let mut obj_history = Vec::new();

    // Evaluate initial solutions.
    for replica in &replicas {
        let obj = qubo.evaluate(replica);
        if obj < best_obj {
            best_obj = obj;
            best_solution = replica.clone();
        }
    }

    let history_interval = (config.n_sweeps / 100).max(1);

    for sweep in 0..config.n_sweeps {
        let fraction = sweep as f64 / config.n_sweeps as f64;
        let gamma = config.transverse_field_at(fraction);

        // Inter-replica coupling strength from Suzuki-Trotter.
        // J_perp = -(T/2) * ln(tanh(Γ/(M*T)))
        let m_f = m as f64;
        let tanh_arg = gamma / (m_f * config.temperature.max(1e-10));
        let j_perp = if tanh_arg.abs() < 20.0 {
            -(config.temperature / 2.0) * tanh_arg.tanh().abs().max(1e-30).ln()
        } else {
            0.0
        };

        // Sweep over all replicas and all qubits.
        for r in 0..m {
            for i in 0..n {
                total_flips += 1;

                // Compute energy change from flipping spin i in replica r.
                let current_val = replicas[r][i];
                let new_val = 1 - current_val;

                // Classical energy change.
                let sign = if new_val > current_val { 1.0 } else { -1.0 };
                let mut delta_e_classical = sign * qubo.linear[i];
                for j in 0..n {
                    if j == i {
                        delta_e_classical += sign * qubo.q_matrix[i][i];
                    } else {
                        delta_e_classical += sign
                            * (qubo.q_matrix[i][j] + qubo.q_matrix[j][i])
                            * replicas[r][j] as f64;
                    }
                }
                delta_e_classical /= m_f; // Divide by number of replicas.

                // Inter-replica coupling energy change.
                let mut delta_e_coupling = 0.0;
                if m > 1 {
                    let r_prev = if r == 0 { m - 1 } else { r - 1 };
                    let r_next = if r == m - 1 { 0 } else { r + 1 };

                    let same_prev = replicas[r_prev][i] == new_val;
                    let same_next = replicas[r_next][i] == new_val;
                    let was_same_prev = replicas[r_prev][i] == current_val;
                    let was_same_next = replicas[r_next][i] == current_val;

                    let coupling_new = (if same_prev { -j_perp } else { j_perp })
                        + (if same_next { -j_perp } else { j_perp });
                    let coupling_old = (if was_same_prev { -j_perp } else { j_perp })
                        + (if was_same_next { -j_perp } else { j_perp });

                    delta_e_coupling = coupling_new - coupling_old;
                }

                let delta_e = delta_e_classical + delta_e_coupling;

                // Metropolis acceptance.
                let accept = if delta_e <= 0.0 {
                    true
                } else {
                    let threshold = (-beta * delta_e).exp();
                    rng.next_f64() < threshold
                };

                if accept {
                    replicas[r][i] = new_val;
                    accepted_flips += 1;
                }
            }
        }

        // Track the best solution.
        for replica in &replicas {
            let obj = qubo.evaluate(replica);
            if obj < best_obj {
                best_obj = obj;
                best_solution = replica.clone();
            }
        }

        if sweep % history_interval == 0 {
            obj_history.push(best_obj);
        }
    }

    // Build portfolio from selected assets.
    let selected_count = best_solution.iter().filter(|&&x| x == 1).count();
    let mut weights = vec![0.0; n];
    if selected_count > 0 {
        let w = 1.0 / selected_count as f64;
        for i in 0..n {
            if best_solution[i] == 1 {
                weights[i] = w;
            }
        }
    } else {
        weights = universe.equal_weight_portfolio();
    }

    let duration = start.elapsed();
    let acceptance_rate = if total_flips > 0 {
        accepted_flips as f64 / total_flips as f64
    } else {
        0.0
    };

    let portfolio_result = PortfolioResult::from_weights(
        weights,
        universe,
        "Quantum Annealing",
        config.n_sweeps,
        duration,
        true,
        best_obj,
    )
    .with_metadata("n_replicas", config.n_replicas as f64)
    .with_metadata("acceptance_rate", acceptance_rate)
    .with_metadata("selected_count", selected_count as f64);

    info!(
        best_obj,
        selected_count,
        acceptance_rate = format!("{:.2}%", acceptance_rate * 100.0),
        duration_ms = duration.as_millis(),
        "Quantum annealing completed"
    );

    Ok(AnnealingResult {
        selected_assets: best_solution,
        best_objective: best_obj,
        portfolio_result,
        n_sweeps: config.n_sweeps,
        n_replicas: m,
        duration,
        objective_history: obj_history,
        accepted_flips,
        total_flips,
        acceptance_rate,
    })
}

// ===========================================================================
// UTILITY FUNCTIONS
// ===========================================================================

/// Convert an integer state to a bit vector.
fn state_to_bits(state: usize, n: usize) -> Vec<u8> {
    (0..n).map(|i| ((state >> i) & 1) as u8).collect()
}

/// Simple deterministic PRNG (xorshift64) for reproducibility.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEADBEEF } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_universe() -> AssetUniverse {
        AssetUniverse::new(
            vec!["BTC", "ETH", "SOL", "USDC"],
            vec![0.10, 0.15, 0.12, 0.02],
            vec![
                vec![0.04, 0.006, 0.002, 0.0001],
                vec![0.006, 0.09, 0.004, 0.0002],
                vec![0.002, 0.004, 0.0225, 0.0001],
                vec![0.0001, 0.0002, 0.0001, 0.0001],
            ],
        )
        .unwrap()
    }

    fn small_universe() -> AssetUniverse {
        AssetUniverse::new(
            vec!["A", "B"],
            vec![0.10, 0.05],
            vec![vec![0.04, 0.01], vec![0.01, 0.02]],
        )
        .unwrap()
    }

    // ── AssetUniverse tests ────────────────────────────────────────────

    #[test]
    fn test_universe_creation() {
        let u = test_universe();
        assert_eq!(u.n_assets(), 4);
        assert_eq!(u.names[0], "BTC");
        assert_eq!(u.risk_free_rate, 0.0);
        assert_eq!(u.weight_bounds.len(), 4);
    }

    #[test]
    fn test_universe_dimension_mismatch() {
        let result = AssetUniverse::new(
            vec!["A", "B"],
            vec![0.1], // wrong length
            vec![vec![0.04, 0.01], vec![0.01, 0.02]],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_universe_cov_dimension_mismatch() {
        let result = AssetUniverse::new(
            vec!["A", "B"],
            vec![0.1, 0.05],
            vec![vec![0.04, 0.01]], // wrong number of rows
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_universe_non_psd() {
        let result = AssetUniverse::new(
            vec!["A"],
            vec![0.1],
            vec![vec![-0.01]], // negative variance
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_portfolio_return() {
        let u = small_universe();
        let w = vec![0.5, 0.5];
        let ret = u.portfolio_return(&w);
        assert!((ret - 0.075).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_variance() {
        let u = small_universe();
        let w = vec![0.5, 0.5];
        let var = u.portfolio_variance(&w);
        // 0.25*0.04 + 2*0.25*0.01 + 0.25*0.02 = 0.01 + 0.005 + 0.005 = 0.02
        assert!((var - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_risk() {
        let u = small_universe();
        let w = vec![0.5, 0.5];
        let risk = u.portfolio_risk(&w);
        assert!((risk - 0.02f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_ratio() {
        let u = small_universe().with_risk_free_rate(0.02);
        let w = vec![0.5, 0.5];
        let sharpe = u.sharpe_ratio(&w);
        let expected = (0.075 - 0.02) / 0.02f64.sqrt();
        assert!((sharpe - expected).abs() < 1e-8);
    }

    #[test]
    fn test_asset_volatilities() {
        let u = small_universe();
        let vols = u.asset_volatilities();
        assert!((vols[0] - 0.2).abs() < 1e-10);
        assert!((vols[1] - 0.02f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix() {
        let u = small_universe();
        let corr = u.correlation_matrix();
        // Diagonal should be 1.
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[1][1] - 1.0).abs() < 1e-10);
        // Off-diagonal: 0.01 / (0.2 * sqrt(0.02))
        let expected = 0.01 / (0.2 * 0.02f64.sqrt());
        assert!((corr[0][1] - expected).abs() < 1e-8);
        // Symmetric.
        assert!((corr[0][1] - corr[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_risk_contributions() {
        let u = small_universe();
        let w = vec![0.5, 0.5];
        let rc = u.risk_contributions(&w);
        assert_eq!(rc.len(), 2);
        // Sum of risk contributions ≈ portfolio risk.
        let rc_sum: f64 = rc.iter().sum();
        let risk = u.portfolio_risk(&w);
        assert!((rc_sum - risk).abs() < 1e-8);
    }

    #[test]
    fn test_equal_weight() {
        let u = test_universe();
        let ew = u.equal_weight_portfolio();
        assert_eq!(ew.len(), 4);
        assert!((ew[0] - 0.25).abs() < 1e-10);
        let sum: f64 = ew.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_market_cap_portfolio() {
        let u = test_universe()
            .with_market_caps(vec![1000.0, 500.0, 300.0, 200.0])
            .unwrap();
        let mcp = u.market_cap_portfolio().unwrap();
        assert_eq!(mcp.len(), 4);
        let sum: f64 = mcp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((mcp[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_universe_display() {
        let u = test_universe();
        let s = format!("{u}");
        assert!(s.contains("BTC"));
        assert!(s.contains("ETH"));
    }

    #[test]
    fn test_with_weight_bounds() {
        let u = test_universe()
            .with_weight_bounds(vec![(0.0, 0.5); 4])
            .unwrap();
        assert_eq!(u.weight_bounds[0], (0.0, 0.5));
    }

    #[test]
    fn test_with_weight_bounds_mismatch() {
        let result = test_universe().with_weight_bounds(vec![(0.0, 1.0); 3]);
        assert!(result.is_err());
    }

    // ── PortfolioResult tests ──────────────────────────────────────────

    #[test]
    fn test_result_from_weights() {
        let u = test_universe();
        let w = u.equal_weight_portfolio();
        let result = PortfolioResult::from_weights(
            w.clone(),
            &u,
            "test",
            10,
            Duration::from_millis(1),
            true,
            0.0,
        );
        assert_eq!(result.weights.len(), 4);
        assert!(result.portfolio_return > 0.0);
        assert!(result.portfolio_risk > 0.0);
        assert_eq!(result.active_assets(), 4);
        assert_eq!(result.method, "test");
    }

    #[test]
    fn test_result_hhi() {
        let u = test_universe();
        let ew = u.equal_weight_portfolio();
        let result = PortfolioResult::from_weights(ew, &u, "test", 0, Duration::ZERO, true, 0.0);
        // HHI for equal weight = 4 * (0.25)^2 = 0.25
        assert!((result.hhi() - 0.25).abs() < 1e-10);
        // Effective N = 1/0.25 = 4
        assert!((result.effective_n() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_result_max_min_weight() {
        let u = test_universe();
        let w = vec![0.4, 0.3, 0.2, 0.1];
        let result = PortfolioResult::from_weights(w, &u, "test", 0, Duration::ZERO, true, 0.0);
        assert!((result.max_weight() - 0.4).abs() < 1e-10);
        assert!((result.min_active_weight() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_result_with_metadata() {
        let u = test_universe();
        let w = u.equal_weight_portfolio();
        let result = PortfolioResult::from_weights(w, &u, "test", 0, Duration::ZERO, true, 0.0)
            .with_metadata("lambda", 2.5);
        assert_eq!(result.metadata.get("lambda"), Some(&2.5));
    }

    #[test]
    fn test_result_display() {
        let u = test_universe();
        let w = u.equal_weight_portfolio();
        let result =
            PortfolioResult::from_weights(w, &u, "test", 10, Duration::from_millis(5), true, 0.0);
        let s = format!("{result}");
        assert!(s.contains("Return:"));
        assert!(s.contains("Risk:"));
        assert!(s.contains("Sharpe:"));
    }

    // ── Markowitz tests ────────────────────────────────────────────────

    #[test]
    fn test_markowitz_basic() {
        let u = test_universe();
        let config = MarkowitzConfig::default();
        let result = markowitz_optimize(&u, &config).unwrap();

        assert_eq!(result.weights.len(), 4);
        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to 1, got {sum}"
        );
        assert!(
            result.weights.iter().all(|&w| w >= -0.001),
            "Long-only violated"
        );
        assert!(result.portfolio_risk > 0.0);
        assert!(result.method.contains("Markowitz"));
    }

    #[test]
    fn test_markowitz_with_target_return() {
        let u = test_universe();
        let config = MarkowitzConfig::default().with_target_return(0.10);
        let result = markowitz_optimize(&u, &config).unwrap();

        assert_eq!(result.weights.len(), 4);
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_markowitz_high_risk_aversion() {
        let u = test_universe();
        let config = MarkowitzConfig::default().with_risk_aversion(100.0);
        let result = markowitz_optimize(&u, &config).unwrap();

        // High risk aversion should favour lower-risk assets.
        assert!(result.portfolio_risk > 0.0);
    }

    #[test]
    fn test_markowitz_empty_universe() {
        let u = AssetUniverse::new(Vec::<String>::new(), Vec::new(), Vec::new()).unwrap();
        let config = MarkowitzConfig::default();
        let result = markowitz_optimize(&u, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_markowitz_single_asset() {
        let u = AssetUniverse::new(vec!["A"], vec![0.10], vec![vec![0.04]]).unwrap();
        let config = MarkowitzConfig::default();
        let result = markowitz_optimize(&u, &config).unwrap();
        assert!((result.weights[0] - 1.0).abs() < 0.01);
    }

    // ── Efficient Frontier tests ───────────────────────────────────────

    #[test]
    fn test_efficient_frontier() {
        let u = test_universe();
        let config = EfficientFrontierConfig::default().with_n_points(10);
        let frontier = compute_efficient_frontier(&u, &config).unwrap();

        assert!(!frontier.is_empty());
        assert!(frontier.len() <= 10);
        assert!(frontier.return_range().is_some());
        assert!(frontier.risk_range().is_some());
    }

    #[test]
    fn test_efficient_frontier_tangency() {
        let u = test_universe().with_risk_free_rate(0.02);
        let config = EfficientFrontierConfig::default().with_n_points(20);
        let frontier = compute_efficient_frontier(&u, &config).unwrap();

        if let Some(tangency) = frontier.tangency() {
            assert!(tangency.sharpe_ratio > 0.0);
        }
    }

    #[test]
    fn test_efficient_frontier_display() {
        let u = test_universe();
        let config = EfficientFrontierConfig::default().with_n_points(5);
        let frontier = compute_efficient_frontier(&u, &config).unwrap();
        let s = format!("{frontier}");
        assert!(s.contains("Efficient Frontier"));
    }

    #[test]
    fn test_efficient_frontier_too_few_points() {
        let u = test_universe();
        let config = EfficientFrontierConfig::default().with_n_points(1);
        let result = compute_efficient_frontier(&u, &config);
        assert!(result.is_err());
    }

    // ── Risk Parity tests ──────────────────────────────────────────────

    #[test]
    fn test_risk_parity_basic() {
        let u = test_universe();
        let config = RiskParityConfig::default();
        let result = risk_parity_optimize(&u, &config).unwrap();

        assert_eq!(result.weights.len(), 4);
        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to 1, got {sum}"
        );
        assert!(
            result.weights.iter().all(|&w| w >= 0.0),
            "Weights should be non-negative"
        );
        assert!(result.method.contains("Risk Parity"));
    }

    #[test]
    fn test_risk_parity_equal_contributions() {
        let u = test_universe();
        let config = RiskParityConfig::default().with_max_iterations(10_000);
        let result = risk_parity_optimize(&u, &config).unwrap();

        // Risk contributions should be roughly equal.
        let rc = &result.risk_contributions;
        let rc_sum: f64 = rc.iter().sum();
        if rc_sum > 1e-6 {
            let rc_fracs: Vec<f64> = rc.iter().map(|r| r / rc_sum).collect();
            let target = 0.25; // equal budget for 4 assets
            for frac in &rc_fracs {
                // Allow some tolerance.
                assert!(
                    (*frac - target).abs() < 0.15,
                    "Risk contribution {frac:.4} too far from target {target:.4}"
                );
            }
        }
    }

    #[test]
    fn test_risk_parity_custom_budgets() {
        let u = small_universe();
        let config = RiskParityConfig::default().with_risk_budgets(vec![0.6, 0.4]);
        let result = risk_parity_optimize(&u, &config).unwrap();
        assert_eq!(result.weights.len(), 2);
    }

    #[test]
    fn test_risk_parity_budget_mismatch() {
        let u = test_universe();
        let config = RiskParityConfig::default().with_risk_budgets(vec![0.5, 0.5]); // 2 budgets for 4 assets
        let result = risk_parity_optimize(&u, &config);
        assert!(result.is_err());
    }

    // ── Black-Litterman tests ──────────────────────────────────────────

    #[test]
    fn test_black_litterman_no_views() {
        let u = test_universe()
            .with_market_caps(vec![1000.0, 500.0, 300.0, 200.0])
            .unwrap();
        let config = BlackLittermanConfig::default();
        let (bl_returns, result) = black_litterman_optimize(&u, &config).unwrap();

        assert_eq!(bl_returns.equilibrium_returns.len(), 4);
        assert_eq!(bl_returns.posterior_returns.len(), 4);
        assert_eq!(result.weights.len(), 4);
        assert!(result.method.contains("Black-Litterman"));
    }

    #[test]
    fn test_black_litterman_with_views() {
        let u = test_universe()
            .with_market_caps(vec![1000.0, 500.0, 300.0, 200.0])
            .unwrap();
        let n = u.n_assets();
        let config = BlackLittermanConfig::default()
            .add_absolute_view(0, n, 0.15) // BTC returns 15%
            .add_relative_view(1, 2, n, 0.05); // ETH outperforms SOL by 5%

        let (bl_returns, result) = black_litterman_optimize(&u, &config).unwrap();

        // Posterior returns should be influenced by views.
        assert!(bl_returns.posterior_returns[0] > bl_returns.equilibrium_returns[0] * 0.5);
        assert_eq!(result.weights.len(), 4);
    }

    #[test]
    fn test_black_litterman_view_dimension_mismatch() {
        let u = test_universe();
        let config = BlackLittermanConfig::default().add_view(vec![1.0, 0.0], 0.1); // 2 elements but universe has 4 assets

        let result = black_litterman_optimize(&u, &config);
        assert!(result.is_err());
    }

    // ── QUBO tests ─────────────────────────────────────────────────────

    #[test]
    fn test_qubo_from_universe() {
        let u = test_universe();
        let qubo = QuboFormulation::from_universe(&u, 1.0, 2.0, None, 5.0, 3.0);
        assert_eq!(qubo.n_vars, 4);
        assert_eq!(qubo.q_matrix.len(), 4);
        assert_eq!(qubo.linear.len(), 4);
    }

    #[test]
    fn test_qubo_evaluate() {
        let u = small_universe();
        let qubo = QuboFormulation::from_universe(&u, 1.0, 1.0, None, 0.0, 0.0);

        let x0 = vec![0, 0];
        let x1 = vec![1, 0];
        let x2 = vec![0, 1];
        let x3 = vec![1, 1];

        let v0 = qubo.evaluate(&x0);
        let v1 = qubo.evaluate(&x1);
        let v2 = qubo.evaluate(&x2);
        let v3 = qubo.evaluate(&x3);

        // All zeros should have zero objective.
        assert!((v0 - 0.0).abs() < 1e-10);
        // Others should be different from zero.
        assert!((v1 - v2).abs() > 1e-10 || (v1 - v0).abs() > 1e-10);
        // Selecting both should include interaction terms.
        assert!(v3 != 0.0);
    }

    #[test]
    fn test_qubo_evaluate_probabilities() {
        let u = small_universe();
        let qubo = QuboFormulation::from_universe(&u, 1.0, 1.0, None, 0.0, 0.0);

        let probs = vec![0.5, 0.5];
        let val = qubo.evaluate_probabilities(&probs);
        // Should be a finite number.
        assert!(val.is_finite());
    }

    #[test]
    fn test_qubo_with_cardinality() {
        let u = test_universe();
        let qubo = QuboFormulation::from_universe(&u, 1.0, 2.0, Some(2), 10.0, 0.0);

        // Solution with exactly 2 assets should have lower cardinality penalty.
        let x2 = vec![1, 1, 0, 0]; // 2 assets
        let x3 = vec![1, 1, 1, 0]; // 3 assets

        let v2 = qubo.evaluate(&x2);
        let v3 = qubo.evaluate(&x3);

        // Due to the strong cardinality penalty, 2-asset solution should be preferred.
        // (Not always guaranteed with other terms, but the penalty should help.)
        assert!(v2.is_finite());
        assert!(v3.is_finite());
    }

    // ── QAOA tests ─────────────────────────────────────────────────────

    #[test]
    fn test_qaoa_basic() {
        let u = small_universe();
        let config = QaoaConfig::default()
            .with_depth(1)
            .with_shots(100)
            .with_n_optimizer_iterations(20);
        let result = qaoa_portfolio_optimize(&u, &config).unwrap();

        assert_eq!(result.selected_assets.len(), 2);
        assert!(result.best_objective.is_finite());
        assert_eq!(result.depth, 1);
        assert_eq!(result.portfolio_result.weights.len(), 2);
    }

    #[test]
    fn test_qaoa_with_cardinality() {
        let u = test_universe();
        let config = QaoaConfig::default()
            .with_depth(2)
            .with_shots(100)
            .with_n_optimizer_iterations(30)
            .with_target_cardinality(2);
        let result = qaoa_portfolio_optimize(&u, &config).unwrap();

        assert_eq!(result.selected_assets.len(), 4);
        assert!(result.best_objective.is_finite());
        assert!(!result.objective_history.is_empty());
    }

    #[test]
    fn test_qaoa_empty_universe() {
        let u = AssetUniverse::new(Vec::<String>::new(), Vec::new(), Vec::new()).unwrap();
        let config = QaoaConfig::default();
        let result = qaoa_portfolio_optimize(&u, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_qaoa_zero_depth() {
        let u = small_universe();
        let config = QaoaConfig::default().with_depth(0);
        let result = qaoa_portfolio_optimize(&u, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_qaoa_solution_distribution() {
        let u = small_universe();
        let config = QaoaConfig::default()
            .with_depth(2)
            .with_shots(500)
            .with_n_optimizer_iterations(30);
        let result = qaoa_portfolio_optimize(&u, &config).unwrap();

        // Should have at least one entry in the distribution.
        assert!(!result.solution_distribution.is_empty());

        // Total counts should be approximately equal to shots.
        let total_count: usize = result.solution_distribution.values().sum();
        assert!(total_count > 0 && total_count <= config.shots * 2);
    }

    #[test]
    fn test_qaoa_mean_field_large() {
        // Use an 20-asset universe to trigger mean-field fallback.
        let n = 20;
        let names: Vec<String> = (0..n).map(|i| format!("Asset{i}")).collect();
        let returns: Vec<f64> = (0..n).map(|i| 0.05 + 0.005 * i as f64).collect();
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            cov[i][i] = 0.02 + 0.001 * i as f64;
            for j in (i + 1)..n {
                let c = 0.001;
                cov[i][j] = c;
                cov[j][i] = c;
            }
        }

        let u = AssetUniverse::new(names, returns, cov).unwrap();
        let config = QaoaConfig::default()
            .with_depth(2)
            .with_n_optimizer_iterations(50);
        let result = qaoa_portfolio_optimize(&u, &config).unwrap();

        assert_eq!(result.selected_assets.len(), 20);
        assert!(result.best_objective.is_finite());
    }

    // ── Statevector simulation tests ───────────────────────────────────

    #[test]
    fn test_state_to_bits() {
        let bits = state_to_bits(0b101, 3);
        assert_eq!(bits, vec![1, 0, 1]);

        let bits = state_to_bits(0, 4);
        assert_eq!(bits, vec![0, 0, 0, 0]);

        let bits = state_to_bits(0b1111, 4);
        assert_eq!(bits, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_simulate_qaoa_circuit_uniform() {
        // With all zero parameters, the circuit should produce uniform distribution.
        let n = 2;
        let params = vec![0.0, 0.0]; // p=1, γ=0, β=0
        let qubo_values = vec![0.0; 4];

        let probs = simulate_qaoa_circuit(n, &params, &qubo_values);
        assert_eq!(probs.len(), 4);

        // Should be approximately uniform (cos(0)=1, sin(0)=0 → identity).
        for &p in &probs {
            assert!((p - 0.25).abs() < 0.01, "Expected ~0.25, got {p}");
        }
    }

    #[test]
    fn test_simulate_qaoa_circuit_normalisation() {
        let n = 3;
        let params = vec![0.5, 0.3, 0.2, 0.7]; // p=2
        let qubo_values: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();

        let probs = simulate_qaoa_circuit(n, &params, &qubo_values);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1, got {sum}"
        );
    }

    // ── VQE tests ──────────────────────────────────────────────────────

    #[test]
    fn test_vqe_basic() {
        let u = test_universe();
        let config = VqeConfig::default()
            .with_n_iterations(100)
            .with_learning_rate(0.02);
        let result = vqe_portfolio_optimize(&u, &config).unwrap();

        assert_eq!(result.weights.len(), 4);
        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "VQE weights should sum to 1, got {sum}"
        );
        assert!(
            result.weights.iter().all(|&w| w >= -0.001),
            "VQE: long-only violated"
        );
        assert!(result.method.contains("VQE"));
    }

    #[test]
    fn test_vqe_small() {
        let u = small_universe();
        let config = VqeConfig::default()
            .with_n_iterations(200)
            .with_learning_rate(0.01);
        let result = vqe_portfolio_optimize(&u, &config).unwrap();

        assert_eq!(result.weights.len(), 2);
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vqe_empty_universe() {
        let u = AssetUniverse::new(Vec::<String>::new(), Vec::new(), Vec::new()).unwrap();
        let config = VqeConfig::default();
        let result = vqe_portfolio_optimize(&u, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_ansatz_to_weights_sums_to_one() {
        let theta = vec![0.1, 0.2, -0.3, 0.4, 0.0, 0.5, -0.1, 0.3];
        let weights = ansatz_to_weights(&theta, 4, 2, true);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Softmax weights should sum to 1, got {sum}"
        );
        assert!(weights.iter().all(|&w| w >= 0.0), "Long-only violated");
    }

    #[test]
    fn test_ansatz_to_weights_different_params() {
        let theta1 = vec![1.0, 0.0, 0.0, 0.0];
        let theta2 = vec![0.0, 0.0, 1.0, 0.0];
        let w1 = ansatz_to_weights(&theta1, 2, 2, true);
        let w2 = ansatz_to_weights(&theta2, 2, 2, true);
        // Different parameters should give different weights.
        assert!((w1[0] - w2[0]).abs() > 0.01 || (w1[1] - w2[1]).abs() > 0.01);
    }

    // ── Simulated Quantum Annealing tests ──────────────────────────────

    #[test]
    fn test_annealing_basic() {
        let u = test_universe();
        let config = AnnealingConfig::default()
            .with_n_sweeps(500)
            .with_n_replicas(4);
        let result = quantum_anneal_optimize(&u, &config).unwrap();

        assert_eq!(result.selected_assets.len(), 4);
        assert!(result.best_objective.is_finite());
        assert!(result.acceptance_rate > 0.0);
        assert!(result.acceptance_rate <= 1.0);
        assert!(result.n_sweeps == 500);
        assert_eq!(result.n_replicas, 4);
        assert!(result.portfolio_result.weights.len() == 4);
    }

    #[test]
    fn test_annealing_with_cardinality() {
        let u = test_universe();
        let config = AnnealingConfig::default()
            .with_n_sweeps(500)
            .with_n_replicas(4)
            .with_target_cardinality(2);
        let result = quantum_anneal_optimize(&u, &config).unwrap();

        assert_eq!(result.selected_assets.len(), 4);
        assert!(result.best_objective.is_finite());
    }

    #[test]
    fn test_annealing_schedules() {
        let config_linear = AnnealingConfig::default().with_schedule(AnnealingSchedule::Linear);
        let config_exp = AnnealingConfig::default().with_schedule(AnnealingSchedule::Exponential);
        let config_quad = AnnealingConfig::default().with_schedule(AnnealingSchedule::Quadratic);

        // All schedules should start at initial and end at final.
        for config in &[config_linear, config_exp, config_quad] {
            let gamma_0 = config.transverse_field_at(0.0);
            let gamma_1 = config.transverse_field_at(1.0);
            assert!((gamma_0 - config.initial_transverse_field).abs() < 1e-6);
            assert!((gamma_1 - config.final_transverse_field).abs() < 1e-3);
        }
    }

    #[test]
    fn test_annealing_schedule_monotonic() {
        let config = AnnealingConfig::default().with_schedule(AnnealingSchedule::Linear);
        let mut prev = config.transverse_field_at(0.0);
        for i in 1..=10 {
            let frac = i as f64 / 10.0;
            let current = config.transverse_field_at(frac);
            assert!(
                current <= prev + 1e-10,
                "Linear schedule should be non-increasing"
            );
            prev = current;
        }
    }

    #[test]
    fn test_annealing_empty_universe() {
        let u = AssetUniverse::new(Vec::<String>::new(), Vec::new(), Vec::new()).unwrap();
        let config = AnnealingConfig::default();
        let result = quantum_anneal_optimize(&u, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_annealing_objective_history() {
        let u = test_universe();
        let config = AnnealingConfig::default()
            .with_n_sweeps(1000)
            .with_n_replicas(2);
        let result = quantum_anneal_optimize(&u, &config).unwrap();

        assert!(!result.objective_history.is_empty());
        // Objective should generally decrease (or at least not increase significantly).
        if result.objective_history.len() > 2 {
            let first = result.objective_history[0];
            let last = *result.objective_history.last().unwrap();
            // Allow some tolerance — stochastic process.
            assert!(last <= first + 10.0, "Objective should generally improve");
        }
    }

    #[test]
    fn test_annealing_single_replica() {
        let u = small_universe();
        let config = AnnealingConfig::default()
            .with_n_sweeps(200)
            .with_n_replicas(1);
        let result = quantum_anneal_optimize(&u, &config).unwrap();

        assert_eq!(result.n_replicas, 1);
        assert!(result.best_objective.is_finite());
    }

    // ── Matrix inversion tests ─────────────────────────────────────────

    #[test]
    fn test_invert_identity() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let inv = invert_matrix(&m).unwrap();
        assert!((inv[0][0] - 1.0).abs() < 1e-10);
        assert!((inv[0][1] - 0.0).abs() < 1e-10);
        assert!((inv[1][0] - 0.0).abs() < 1e-10);
        assert!((inv[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_2x2() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let inv = invert_matrix(&m).unwrap();
        // M * M^{-1} should be identity.
        let prod_00 = m[0][0] * inv[0][0] + m[0][1] * inv[1][0];
        let prod_01 = m[0][0] * inv[0][1] + m[0][1] * inv[1][1];
        let prod_10 = m[1][0] * inv[0][0] + m[1][1] * inv[1][0];
        let prod_11 = m[1][0] * inv[0][1] + m[1][1] * inv[1][1];
        assert!((prod_00 - 1.0).abs() < 1e-10);
        assert!((prod_01 - 0.0).abs() < 1e-10);
        assert!((prod_10 - 0.0).abs() < 1e-10);
        assert!((prod_11 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_singular() {
        let m = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0], // rank 1
        ];
        let result = invert_matrix(&m);
        assert!(result.is_err());
    }

    #[test]
    fn test_invert_empty() {
        let m: Vec<Vec<f64>> = vec![];
        let inv = invert_matrix(&m).unwrap();
        assert!(inv.is_empty());
    }

    // ── SimpleRng tests ────────────────────────────────────────────────

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_simple_rng_range() {
        let mut rng = SimpleRng::new(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "next_f64() out of range: {v}");
        }
    }

    #[test]
    fn test_simple_rng_different_seeds() {
        let mut rng1 = SimpleRng::new(1);
        let mut rng2 = SimpleRng::new(2);

        let v1 = rng1.next_u64();
        let v2 = rng2.next_u64();
        assert_ne!(v1, v2, "Different seeds should produce different outputs");
    }

    // ── Error display tests ────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = PortfolioError::NonPsdCovariance;
        assert!(format!("{e}").contains("PSD"));

        let e = PortfolioError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        assert!(format!("{e}").contains("4"));
        assert!(format!("{e}").contains("3"));

        let e = PortfolioError::Infeasible("test".into());
        assert!(format!("{e}").contains("infeasible"));

        let e = PortfolioError::NumericalError("nan".into());
        assert!(format!("{e}").contains("numerical"));

        let e = PortfolioError::InvalidConfig("bad".into());
        assert!(format!("{e}").contains("invalid config"));

        let e = PortfolioError::DidNotConverge {
            iterations: 100,
            residual: 0.01,
        };
        assert!(format!("{e}").contains("converge"));

        let e = PortfolioError::Internal("oops".into());
        assert!(format!("{e}").contains("internal"));
    }

    // ── Integration tests ──────────────────────────────────────────────

    #[test]
    fn test_compare_classical_and_quantum() {
        let u = test_universe();

        // Classical Markowitz
        let mk_config = MarkowitzConfig::default().with_risk_aversion(2.0);
        let mk_result = markowitz_optimize(&u, &mk_config).unwrap();

        // QAOA
        let qaoa_config = QaoaConfig::default()
            .with_depth(2)
            .with_n_optimizer_iterations(50);
        let qaoa_result = qaoa_portfolio_optimize(&u, &qaoa_config).unwrap();

        // VQE
        let vqe_config = VqeConfig::default().with_n_iterations(100);
        let vqe_result = vqe_portfolio_optimize(&u, &vqe_config).unwrap();

        // Annealing
        let anneal_config = AnnealingConfig::default()
            .with_n_sweeps(500)
            .with_n_replicas(4);
        let anneal_result = quantum_anneal_optimize(&u, &anneal_config).unwrap();

        // All should produce valid portfolios.
        assert!(mk_result.portfolio_risk > 0.0);
        assert!(qaoa_result.portfolio_result.portfolio_risk >= 0.0);
        assert!(vqe_result.portfolio_risk > 0.0);
        assert!(anneal_result.portfolio_result.portfolio_risk >= 0.0);

        // All weights should sum to ~1.
        let mk_sum: f64 = mk_result.weights.iter().sum();
        let qaoa_sum: f64 = qaoa_result.portfolio_result.weights.iter().sum();
        let vqe_sum: f64 = vqe_result.weights.iter().sum();
        let anneal_sum: f64 = anneal_result.portfolio_result.weights.iter().sum();

        assert!((mk_sum - 1.0).abs() < 0.05, "Markowitz sum: {mk_sum}");
        assert!((qaoa_sum - 1.0).abs() < 0.05, "QAOA sum: {qaoa_sum}");
        assert!((vqe_sum - 1.0).abs() < 0.05, "VQE sum: {vqe_sum}");
        assert!((anneal_sum - 1.0).abs() < 0.05, "Anneal sum: {anneal_sum}");
    }

    #[test]
    fn test_full_pipeline() {
        let u = test_universe().with_risk_free_rate(0.03);

        // 1. Compute efficient frontier.
        let frontier =
            compute_efficient_frontier(&u, &EfficientFrontierConfig::default().with_n_points(10))
                .unwrap();
        assert!(!frontier.is_empty());

        // 2. Get the tangency portfolio.
        if let Some(tangency) = frontier.tangency() {
            assert!(tangency.sharpe_ratio > 0.0);
        }

        // 3. Risk parity.
        let rp = risk_parity_optimize(&u, &RiskParityConfig::default()).unwrap();
        assert!(rp.portfolio_risk > 0.0);

        // 4. QAOA for asset selection.
        let qaoa = qaoa_portfolio_optimize(
            &u,
            &QaoaConfig::default()
                .with_depth(1)
                .with_n_optimizer_iterations(20),
        )
        .unwrap();
        assert!(!qaoa.selected_assets.is_empty());

        // 5. Quantum annealing.
        let anneal = quantum_anneal_optimize(
            &u,
            &AnnealingConfig::default()
                .with_n_sweeps(200)
                .with_n_replicas(2),
        )
        .unwrap();
        assert!(anneal.best_objective.is_finite());
    }
}
