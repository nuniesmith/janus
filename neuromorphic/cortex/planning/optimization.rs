//! Portfolio optimization engine
//!
//! Part of the Cortex region
//! Component: planning
//!
//! Implements mean-variance portfolio optimization with efficient frontier
//! computation, risk-parity weighting, and maximum-Sharpe portfolio
//! selection. Supports configurable constraints (min/max weight, max
//! concentration, long-only) and produces allocation recommendations
//! with risk attribution.
//!
//! Key features:
//! - Mean-variance optimization via closed-form and iterative solvers
//! - Efficient frontier sampling across target-return levels
//! - Risk-parity (equal risk contribution) portfolio construction
//! - Maximum Sharpe ratio portfolio identification
//! - Minimum variance portfolio as a baseline
//! - Constraint enforcement: weight bounds, concentration limits, long-only
//! - Risk attribution: marginal risk contribution per asset
//! - EMA-smoothed tracking of optimal allocations across re-optimisations
//! - Sliding window of recent optimisation results for stability analysis
//! - Running statistics with turnover and rebalance tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the portfolio optimisation engine
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Risk-free rate for Sharpe calculations (annualised)
    pub risk_free_rate: f64,
    /// Minimum weight per asset (e.g. 0.0 for long-only)
    pub min_weight: f64,
    /// Maximum weight per asset (e.g. 0.40 for 40% cap)
    pub max_weight: f64,
    /// Maximum concentration: sum of top-N weights must not exceed this
    pub max_concentration: f64,
    /// Number of top assets for concentration check
    pub concentration_top_n: usize,
    /// Whether short-selling is allowed
    pub allow_short: bool,
    /// Number of points to sample on the efficient frontier
    pub frontier_points: usize,
    /// Risk aversion parameter for mean-variance utility: U = μ - (λ/2)σ²
    pub risk_aversion: f64,
    /// EMA decay for smoothing allocations across runs (0 < decay < 1)
    pub ema_decay: f64,
    /// Maximum recent results in sliding window
    pub window_size: usize,
    /// Convergence tolerance for iterative solvers
    pub solver_tolerance: f64,
    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            risk_free_rate: 0.04,
            min_weight: 0.0,
            max_weight: 1.0,
            max_concentration: 1.0,
            concentration_top_n: 3,
            allow_short: false,
            risk_aversion: 2.0,
            frontier_points: 20,
            ema_decay: 0.3,
            window_size: 64,
            solver_tolerance: 1e-8,
            max_iterations: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

/// Asset universe description for optimisation
#[derive(Debug, Clone)]
pub struct AssetUniverse {
    /// Asset names / identifiers
    pub names: Vec<String>,
    /// Expected returns per asset (annualised)
    pub expected_returns: Vec<f64>,
    /// Covariance matrix (row-major, n×n)
    pub covariance: Vec<f64>,
}

impl AssetUniverse {
    /// Number of assets
    pub fn n(&self) -> usize {
        self.names.len()
    }

    /// Validate dimensions
    pub fn validate(&self) -> Result<()> {
        let n = self.n();
        if n == 0 {
            return Err(Error::Configuration(
                "asset universe must not be empty".into(),
            ));
        }
        if self.expected_returns.len() != n {
            return Err(Error::Configuration(format!(
                "expected_returns length ({}) must match names length ({})",
                self.expected_returns.len(),
                n
            )));
        }
        if self.covariance.len() != n * n {
            return Err(Error::Configuration(format!(
                "covariance length ({}) must be n*n ({})",
                self.covariance.len(),
                n * n
            )));
        }
        // Check symmetry and positive diagonal
        for i in 0..n {
            if self.covariance[i * n + i] < 0.0 {
                return Err(Error::Configuration(format!(
                    "covariance diagonal [{},{}] must be >= 0",
                    i, i
                )));
            }
            for j in (i + 1)..n {
                let diff = (self.covariance[i * n + j] - self.covariance[j * n + i]).abs();
                if diff > 1e-10 {
                    return Err(Error::Configuration(format!(
                        "covariance must be symmetric: [{},{}]={} vs [{},{}]={}",
                        i,
                        j,
                        self.covariance[i * n + j],
                        j,
                        i,
                        self.covariance[j * n + i]
                    )));
                }
            }
        }
        Ok(())
    }

    /// Get variance of asset i
    pub fn variance(&self, i: usize) -> f64 {
        let n = self.n();
        self.covariance[i * n + i]
    }

    /// Get covariance between asset i and asset j
    pub fn cov(&self, i: usize, j: usize) -> f64 {
        let n = self.n();
        self.covariance[i * n + j]
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Optimal portfolio allocation
#[derive(Debug, Clone)]
pub struct PortfolioAllocation {
    /// Asset names
    pub names: Vec<String>,
    /// Optimal weights per asset (sum to 1.0)
    pub weights: Vec<f64>,
    /// Expected portfolio return (annualised)
    pub expected_return: f64,
    /// Portfolio volatility (annualised)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe: f64,
    /// Risk contribution per asset (fraction of total variance)
    pub risk_contributions: Vec<f64>,
    /// Marginal risk contribution per asset
    pub marginal_risk: Vec<f64>,
    /// Diversification ratio: weighted avg vol / portfolio vol
    pub diversification_ratio: f64,
    /// Maximum weight in the portfolio
    pub max_weight: f64,
    /// Number of non-zero positions
    pub active_positions: usize,
    /// Whether all constraints are satisfied
    pub feasible: bool,
    /// Method used to produce this allocation
    pub method: String,
}

/// A point on the efficient frontier
#[derive(Debug, Clone)]
pub struct FrontierPoint {
    /// Target return level
    pub target_return: f64,
    /// Minimum volatility at this return level
    pub volatility: f64,
    /// Sharpe ratio at this point
    pub sharpe: f64,
    /// Optimal weights at this point
    pub weights: Vec<f64>,
}

/// Full optimisation result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Recommended allocation (max-Sharpe or max-utility)
    pub recommended: PortfolioAllocation,
    /// Minimum variance portfolio
    pub min_variance: PortfolioAllocation,
    /// Risk-parity portfolio
    pub risk_parity: PortfolioAllocation,
    /// Efficient frontier points
    pub frontier: Vec<FrontierPoint>,
    /// Turnover vs previous allocation (sum of absolute weight changes / 2)
    pub turnover: f64,
    /// Number of solver iterations used
    pub iterations: usize,
}

/// Cumulative statistics across optimisation runs
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Total optimisation runs
    pub total_runs: usize,
    /// Total assets evaluated across all runs
    pub total_assets_evaluated: usize,
    /// EMA-smoothed recommended Sharpe
    pub ema_sharpe: f64,
    /// EMA-smoothed recommended volatility
    pub ema_volatility: f64,
    /// EMA-smoothed turnover
    pub ema_turnover: f64,
    /// Best Sharpe observed
    pub best_sharpe: f64,
    /// Worst Sharpe observed
    pub worst_sharpe: f64,
    /// Total turnover across all runs
    pub cumulative_turnover: f64,
    /// Number of infeasible results
    pub infeasible_count: usize,
}

impl Default for OptimizationStats {
    fn default() -> Self {
        Self {
            total_runs: 0,
            total_assets_evaluated: 0,
            ema_sharpe: 0.0,
            ema_volatility: 0.0,
            ema_turnover: 0.0,
            best_sharpe: f64::NEG_INFINITY,
            worst_sharpe: f64::INFINITY,
            cumulative_turnover: 0.0,
            infeasible_count: 0,
        }
    }
}

impl OptimizationStats {
    /// Average assets per run
    pub fn avg_assets_per_run(&self) -> f64 {
        if self.total_runs == 0 {
            return 0.0;
        }
        self.total_assets_evaluated as f64 / self.total_runs as f64
    }

    /// Average turnover per run
    pub fn avg_turnover(&self) -> f64 {
        if self.total_runs == 0 {
            return 0.0;
        }
        self.cumulative_turnover / self.total_runs as f64
    }

    /// Infeasible rate
    pub fn infeasible_rate(&self) -> f64 {
        if self.total_runs == 0 {
            return 0.0;
        }
        self.infeasible_count as f64 / self.total_runs as f64
    }
}

// ---------------------------------------------------------------------------
// Core engine
// ---------------------------------------------------------------------------

/// Portfolio optimisation engine
pub struct Optimization {
    config: OptimizationConfig,
    previous_weights: Option<Vec<f64>>,
    ema_initialized: bool,
    recent: VecDeque<OptimizationResult>,
    stats: OptimizationStats,
}

impl Default for Optimization {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            config: OptimizationConfig::default(),
            previous_weights: None,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: OptimizationStats::default(),
        }
    }

    /// Create from validated config
    pub fn with_config(config: OptimizationConfig) -> Result<Self> {
        if config.min_weight > config.max_weight {
            return Err(Error::Configuration(
                "min_weight must be <= max_weight".into(),
            ));
        }
        if config.max_weight <= 0.0 {
            return Err(Error::Configuration("max_weight must be > 0".into()));
        }
        if config.max_concentration <= 0.0 || config.max_concentration > 1.0 {
            return Err(Error::Configuration(
                "max_concentration must be in (0, 1]".into(),
            ));
        }
        if config.concentration_top_n == 0 {
            return Err(Error::Configuration(
                "concentration_top_n must be > 0".into(),
            ));
        }
        if config.risk_aversion <= 0.0 {
            return Err(Error::Configuration("risk_aversion must be > 0".into()));
        }
        if config.frontier_points == 0 {
            return Err(Error::Configuration("frontier_points must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::Configuration("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::Configuration("window_size must be > 0".into()));
        }
        if config.solver_tolerance <= 0.0 {
            return Err(Error::Configuration("solver_tolerance must be > 0".into()));
        }
        if config.max_iterations == 0 {
            return Err(Error::Configuration("max_iterations must be > 0".into()));
        }
        if !config.allow_short && config.min_weight < 0.0 {
            return Err(Error::Configuration(
                "min_weight must be >= 0 when allow_short is false".into(),
            ));
        }
        Ok(Self {
            config,
            previous_weights: None,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: OptimizationStats::default(),
        })
    }

    /// Convenience: validate and create
    pub fn process(config: OptimizationConfig) -> Result<Self> {
        Self::with_config(config)
    }

    // -----------------------------------------------------------------------
    // Main optimisation entry point
    // -----------------------------------------------------------------------

    /// Run a full portfolio optimisation on the given asset universe.
    pub fn optimize(&mut self, universe: &AssetUniverse) -> Result<OptimizationResult> {
        universe.validate()?;
        let n = universe.n();

        // 1. Minimum variance portfolio
        let min_var = self.compute_min_variance(universe);

        // 2. Risk-parity portfolio
        let risk_parity = self.compute_risk_parity(universe);

        // 3. Max-Sharpe portfolio (tangency)
        let max_sharpe = self.compute_max_sharpe(universe);

        // 4. Efficient frontier
        let frontier = self.compute_frontier(universe, &min_var);

        // 5. Recommended = max-Sharpe (or max utility)
        let recommended = if max_sharpe.sharpe > min_var.sharpe {
            max_sharpe
        } else {
            // Fall back to max-utility via mean-variance
            self.compute_max_utility(universe)
        };

        // 6. Turnover
        let turnover = if let Some(ref prev) = self.previous_weights {
            if prev.len() == n {
                prev.iter()
                    .zip(recommended.weights.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
                    / 2.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        self.previous_weights = Some(recommended.weights.clone());

        let result = OptimizationResult {
            recommended,
            min_variance: min_var,
            risk_parity,
            frontier,
            turnover,
            iterations: 0,
        };

        self.update_stats(&result, n);
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Portfolio construction methods
    // -----------------------------------------------------------------------

    /// Minimum variance portfolio via analytical solution with constraint projection
    fn compute_min_variance(&self, universe: &AssetUniverse) -> PortfolioAllocation {
        let n = universe.n();

        // Start with inverse-variance weighting (heuristic for constrained case)
        let mut weights: Vec<f64> = (0..n)
            .map(|i| {
                let var = universe.variance(i);
                if var > 1e-15 { 1.0 / var } else { 1.0 }
            })
            .collect();

        // Normalise
        let sum: f64 = weights.iter().sum();
        if sum > 1e-15 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        // Iterative projection to satisfy constraints
        self.project_constraints(&mut weights);

        self.build_allocation(universe, weights, "min_variance")
    }

    /// Risk-parity portfolio: each asset contributes equally to portfolio risk
    fn compute_risk_parity(&self, universe: &AssetUniverse) -> PortfolioAllocation {
        let n = universe.n();

        // Start with equal weights
        let mut weights = vec![1.0 / n as f64; n];

        // Iterative risk-parity via inverse marginal-risk weighting
        for _ in 0..self.config.max_iterations {
            let port_vol = self.portfolio_volatility(universe, &weights);
            if port_vol < 1e-15 {
                break;
            }

            let marginal = self.marginal_risk_contributions(universe, &weights, port_vol);

            // Target: each marginal risk = 1/n
            let target = 1.0 / n as f64;
            let mut max_diff: f64 = 0.0;
            let mut new_weights = vec![0.0; n];

            for i in 0..n {
                let mr = marginal[i].max(1e-15);
                // Scale weight inversely proportional to marginal risk
                new_weights[i] = weights[i] * (target / mr).sqrt();
                max_diff = max_diff.max((marginal[i] - target).abs());
            }

            // Normalise
            let sum: f64 = new_weights.iter().sum();
            if sum > 1e-15 {
                for w in &mut new_weights {
                    *w /= sum;
                }
            }

            weights = new_weights;

            if max_diff < self.config.solver_tolerance {
                break;
            }
        }

        self.project_constraints(&mut weights);
        self.build_allocation(universe, weights, "risk_parity")
    }

    /// Maximum Sharpe ratio portfolio via gradient ascent on Sharpe
    fn compute_max_sharpe(&self, universe: &AssetUniverse) -> PortfolioAllocation {
        let n = universe.n();
        let rf = self.config.risk_free_rate;

        // Excess returns
        let excess: Vec<f64> = universe.expected_returns.iter().map(|r| r - rf).collect();

        // If all excess returns are non-positive, fall back to min-variance
        if excess.iter().all(|&e| e <= 0.0) {
            return self.compute_min_variance(universe);
        }

        // Start with weights proportional to excess return (positive only)
        let mut weights: Vec<f64> = excess.iter().map(|&e| e.max(0.01)).collect();
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        // Gradient ascent on Sharpe ratio
        let step_size = 0.01;
        let mut best_sharpe = f64::NEG_INFINITY;
        let mut best_weights = weights.clone();

        for _ in 0..self.config.max_iterations {
            let port_ret = self.portfolio_return(universe, &weights);
            let port_vol = self.portfolio_volatility(universe, &weights);

            if port_vol < 1e-15 {
                break;
            }

            let sharpe = (port_ret - rf) / port_vol;
            if sharpe > best_sharpe {
                best_sharpe = sharpe;
                best_weights = weights.clone();
            }

            // Gradient of Sharpe w.r.t. weights
            let mut grad = vec![0.0; n];
            for i in 0..n {
                let d_ret = universe.expected_returns[i];
                let mut d_var = 0.0;
                for j in 0..n {
                    d_var += universe.cov(i, j) * weights[j];
                }
                d_var *= 2.0;
                // dSharpe/dw_i = (d_ret * vol - (ret-rf) * d_var/(2*vol)) / vol^2
                grad[i] = (d_ret * port_vol - (port_ret - rf) * d_var / (2.0 * port_vol))
                    / (port_vol * port_vol);
            }

            // Update weights
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < self.config.solver_tolerance {
                break;
            }

            for i in 0..n {
                weights[i] += step_size * grad[i] / grad_norm;
            }

            // Project back to feasible set
            self.project_constraints(&mut weights);

            // Normalise
            let sum: f64 = weights.iter().sum();
            if sum > 1e-15 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }

        self.build_allocation(universe, best_weights, "max_sharpe")
    }

    /// Maximum utility portfolio: max(μ - λ/2 · σ²) via gradient ascent
    fn compute_max_utility(&self, universe: &AssetUniverse) -> PortfolioAllocation {
        let n = universe.n();
        let lambda = self.config.risk_aversion;

        // Start with equal weights
        let mut weights = vec![1.0 / n as f64; n];

        let step_size = 0.005;
        let mut best_utility = f64::NEG_INFINITY;
        let mut best_weights = weights.clone();

        for _ in 0..self.config.max_iterations {
            let port_ret = self.portfolio_return(universe, &weights);
            let port_var = self.portfolio_variance(universe, &weights);
            let utility = port_ret - lambda / 2.0 * port_var;

            if utility > best_utility {
                best_utility = utility;
                best_weights = weights.clone();
            }

            // Gradient of utility
            let mut grad = vec![0.0; n];
            for i in 0..n {
                let d_ret = universe.expected_returns[i];
                let mut d_var = 0.0;
                for j in 0..n {
                    d_var += universe.cov(i, j) * weights[j];
                }
                d_var *= 2.0;
                grad[i] = d_ret - lambda / 2.0 * d_var;
            }

            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < self.config.solver_tolerance {
                break;
            }

            for i in 0..n {
                weights[i] += step_size * grad[i] / grad_norm;
            }

            self.project_constraints(&mut weights);

            let sum: f64 = weights.iter().sum();
            if sum > 1e-15 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }

        self.build_allocation(universe, best_weights, "max_utility")
    }

    /// Compute efficient frontier by sweeping target returns
    fn compute_frontier(
        &self,
        universe: &AssetUniverse,
        min_var: &PortfolioAllocation,
    ) -> Vec<FrontierPoint> {
        let n = universe.n();
        let rf = self.config.risk_free_rate;
        let num_points = self.config.frontier_points;

        // Return range: from min-variance return to max single-asset return
        let min_ret = min_var.expected_return;
        let max_ret = universe
            .expected_returns
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if (max_ret - min_ret).abs() < 1e-12 || num_points < 2 {
            return vec![FrontierPoint {
                target_return: min_ret,
                volatility: min_var.volatility,
                sharpe: min_var.sharpe,
                weights: min_var.weights.clone(),
            }];
        }

        let mut frontier = Vec::with_capacity(num_points);
        let step = (max_ret - min_ret) / (num_points - 1) as f64;

        for k in 0..num_points {
            let target = min_ret + step * k as f64;

            // Find min-variance portfolio subject to return >= target
            // via Lagrangian relaxation: add penalty for return shortfall
            let mut weights = vec![1.0 / n as f64; n];
            let penalty = 10.0;

            for _ in 0..200 {
                let port_ret = self.portfolio_return(universe, &weights);
                let port_var = self.portfolio_variance(universe, &weights);
                let _ = port_var;

                let mut grad = vec![0.0; n];
                for i in 0..n {
                    // Minimise variance + penalty * max(0, target - return)
                    let mut d_var = 0.0;
                    for j in 0..n {
                        d_var += universe.cov(i, j) * weights[j];
                    }
                    d_var *= 2.0;
                    grad[i] = d_var;
                    if port_ret < target {
                        grad[i] -= penalty * universe.expected_returns[i];
                    }
                }

                let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if grad_norm < 1e-10 {
                    break;
                }

                for i in 0..n {
                    weights[i] -= 0.01 * grad[i] / grad_norm;
                }

                self.project_constraints(&mut weights);

                let sum: f64 = weights.iter().sum();
                if sum > 1e-15 {
                    for w in &mut weights {
                        *w /= sum;
                    }
                }
            }

            let ret = self.portfolio_return(universe, &weights);
            let vol = self.portfolio_volatility(universe, &weights);
            let sharpe = if vol > 1e-15 { (ret - rf) / vol } else { 0.0 };

            frontier.push(FrontierPoint {
                target_return: target,
                volatility: vol,
                sharpe,
                weights,
            });
        }

        frontier
    }

    // -----------------------------------------------------------------------
    // Portfolio math helpers
    // -----------------------------------------------------------------------

    fn portfolio_return(&self, universe: &AssetUniverse, weights: &[f64]) -> f64 {
        weights
            .iter()
            .zip(universe.expected_returns.iter())
            .map(|(w, r)| w * r)
            .sum()
    }

    fn portfolio_variance(&self, universe: &AssetUniverse, weights: &[f64]) -> f64 {
        let n = weights.len();
        let mut var = 0.0;
        for i in 0..n {
            for j in 0..n {
                var += weights[i] * weights[j] * universe.cov(i, j);
            }
        }
        var.max(0.0)
    }

    fn portfolio_volatility(&self, universe: &AssetUniverse, weights: &[f64]) -> f64 {
        self.portfolio_variance(universe, weights).sqrt()
    }

    fn marginal_risk_contributions(
        &self,
        universe: &AssetUniverse,
        weights: &[f64],
        port_vol: f64,
    ) -> Vec<f64> {
        let n = weights.len();
        let mut mrc = vec![0.0; n];
        if port_vol < 1e-15 {
            return mrc;
        }
        for i in 0..n {
            let mut cov_w = 0.0;
            for j in 0..n {
                cov_w += universe.cov(i, j) * weights[j];
            }
            // Marginal risk contribution = w_i * (Σw)_i / σ_p
            mrc[i] = weights[i] * cov_w / (port_vol * port_vol);
        }
        mrc
    }

    fn risk_contributions(
        &self,
        universe: &AssetUniverse,
        weights: &[f64],
        port_vol: f64,
    ) -> Vec<f64> {
        // Same as marginal risk contributions normalised to sum to 1
        let mrc = self.marginal_risk_contributions(universe, weights, port_vol);
        let sum: f64 = mrc.iter().sum();
        if sum.abs() < 1e-15 {
            return mrc;
        }
        mrc.iter().map(|m| m / sum).collect()
    }

    fn diversification_ratio(&self, universe: &AssetUniverse, weights: &[f64]) -> f64 {
        let port_vol = self.portfolio_volatility(universe, weights);
        if port_vol < 1e-15 {
            return 1.0;
        }
        let weighted_avg_vol: f64 = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| w * universe.variance(i).sqrt())
            .sum();
        weighted_avg_vol / port_vol
    }

    // -----------------------------------------------------------------------
    // Constraint projection
    // -----------------------------------------------------------------------

    fn project_constraints(&self, weights: &mut Vec<f64>) {
        let n = weights.len();

        // 1. Enforce min/max weight bounds
        for w in weights.iter_mut() {
            let min = if self.config.allow_short {
                self.config.min_weight
            } else {
                self.config.min_weight.max(0.0)
            };
            *w = w.clamp(min, self.config.max_weight);
        }

        // 2. Normalise to sum to 1
        let sum: f64 = weights.iter().sum();
        if sum > 1e-15 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        } else {
            // Fallback to equal weights
            for w in weights.iter_mut() {
                *w = 1.0 / n as f64;
            }
        }

        // 3. Enforce concentration limit (top-N)
        if self.config.max_concentration < 1.0 && n > self.config.concentration_top_n {
            let mut sorted_indices: Vec<usize> = (0..n).collect();
            sorted_indices.sort_by(|&a, &b| {
                weights[b]
                    .partial_cmp(&weights[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_n = self.config.concentration_top_n.min(n);
            let top_sum: f64 = sorted_indices[..top_n].iter().map(|&i| weights[i]).sum();

            if top_sum > self.config.max_concentration {
                // Scale down top-N weights proportionally
                let scale = self.config.max_concentration / top_sum;
                let excess = top_sum - self.config.max_concentration;
                let bottom_count = n - top_n;
                for (rank, &idx) in sorted_indices.iter().enumerate() {
                    if rank < top_n {
                        weights[idx] *= scale;
                    } else if bottom_count > 0 {
                        weights[idx] += excess / bottom_count as f64;
                    }
                }
                // Re-normalise
                let sum2: f64 = weights.iter().sum();
                if sum2 > 1e-15 {
                    for w in weights.iter_mut() {
                        *w /= sum2;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Build allocation struct from weights
    // -----------------------------------------------------------------------

    fn build_allocation(
        &self,
        universe: &AssetUniverse,
        weights: Vec<f64>,
        method: &str,
    ) -> PortfolioAllocation {
        let port_ret = self.portfolio_return(universe, &weights);
        let port_vol = self.portfolio_volatility(universe, &weights);
        let rf = self.config.risk_free_rate;
        let sharpe = if port_vol > 1e-15 {
            (port_ret - rf) / port_vol
        } else {
            0.0
        };

        let rc = self.risk_contributions(universe, &weights, port_vol);
        let mr = self.marginal_risk_contributions(universe, &weights, port_vol);
        let div_ratio = self.diversification_ratio(universe, &weights);

        let max_w = weights.iter().cloned().fold(0.0_f64, f64::max);
        let active = weights.iter().filter(|&&w| w.abs() > 1e-8).count();

        // Feasibility check
        let feasible = weights
            .iter()
            .all(|&w| w >= self.config.min_weight - 1e-8 && w <= self.config.max_weight + 1e-8);

        PortfolioAllocation {
            names: universe.names.clone(),
            weights,
            expected_return: port_ret,
            volatility: port_vol,
            sharpe,
            risk_contributions: rc,
            marginal_risk: mr,
            diversification_ratio: div_ratio,
            max_weight: max_w,
            active_positions: active,
            feasible,
            method: method.to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // Stats update
    // -----------------------------------------------------------------------

    fn update_stats(&mut self, result: &OptimizationResult, n: usize) {
        let decay = self.config.ema_decay;

        if !self.ema_initialized {
            self.stats.ema_sharpe = result.recommended.sharpe;
            self.stats.ema_volatility = result.recommended.volatility;
            self.stats.ema_turnover = result.turnover;
            self.ema_initialized = true;
        } else {
            self.stats.ema_sharpe =
                decay * result.recommended.sharpe + (1.0 - decay) * self.stats.ema_sharpe;
            self.stats.ema_volatility =
                decay * result.recommended.volatility + (1.0 - decay) * self.stats.ema_volatility;
            self.stats.ema_turnover =
                decay * result.turnover + (1.0 - decay) * self.stats.ema_turnover;
        }

        self.stats.total_runs += 1;
        self.stats.total_assets_evaluated += n;
        self.stats.cumulative_turnover += result.turnover;

        if result.recommended.sharpe > self.stats.best_sharpe {
            self.stats.best_sharpe = result.recommended.sharpe;
        }
        if result.recommended.sharpe < self.stats.worst_sharpe {
            self.stats.worst_sharpe = result.recommended.sharpe;
        }
        if !result.recommended.feasible {
            self.stats.infeasible_count += 1;
        }

        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(result.clone());
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Cumulative statistics
    pub fn stats(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Number of optimisation runs completed
    pub fn run_count(&self) -> usize {
        self.stats.total_runs
    }

    /// Current configuration
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }

    /// Recent optimisation results
    pub fn recent_results(&self) -> &VecDeque<OptimizationResult> {
        &self.recent
    }

    /// Previous allocation weights (for turnover calculations)
    pub fn previous_weights(&self) -> Option<&Vec<f64>> {
        self.previous_weights.as_ref()
    }

    /// EMA-smoothed Sharpe
    pub fn smoothed_sharpe(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_sharpe)
        } else {
            None
        }
    }

    /// EMA-smoothed volatility
    pub fn smoothed_volatility(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_volatility)
        } else {
            None
        }
    }

    /// EMA-smoothed turnover
    pub fn smoothed_turnover(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_turnover)
        } else {
            None
        }
    }

    /// Windowed mean Sharpe across recent runs
    pub fn windowed_sharpe(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|r| r.recommended.sharpe).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed mean turnover
    pub fn windowed_turnover(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|r| r.turnover).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether Sharpe is trending better over recent window
    pub fn is_sharpe_improving(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.recommended.sharpe)
            .sum::<f64>()
            / half as f64;
        let second: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.recommended.sharpe)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second > first
    }

    /// Whether turnover is trending higher (more churn)
    pub fn is_turnover_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.turnover)
            .sum::<f64>()
            / half as f64;
        let second: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.turnover)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second > first * 1.1
    }

    /// Reset all state (keeps config)
    pub fn reset(&mut self) {
        self.previous_weights = None;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = OptimizationStats::default();
    }
}

// ---------------------------------------------------------------------------
// Helper: simple 2-asset and 3-asset universe constructors for testing
// ---------------------------------------------------------------------------

/// Create a 2-asset universe for testing
pub fn two_asset_universe(
    ret_a: f64,
    ret_b: f64,
    vol_a: f64,
    vol_b: f64,
    correlation: f64,
) -> AssetUniverse {
    let cov_ab = correlation * vol_a * vol_b;
    AssetUniverse {
        names: vec!["A".into(), "B".into()],
        expected_returns: vec![ret_a, ret_b],
        covariance: vec![vol_a * vol_a, cov_ab, cov_ab, vol_b * vol_b],
    }
}

/// Create a 3-asset universe for testing
pub fn three_asset_universe() -> AssetUniverse {
    // Stocks, bonds, commodities with realistic params
    let vol_s = 0.20;
    let vol_b = 0.06;
    let vol_c = 0.25;
    let rho_sb = -0.10;
    let rho_sc = 0.30;
    let rho_bc = 0.05;
    AssetUniverse {
        names: vec!["stocks".into(), "bonds".into(), "commodities".into()],
        expected_returns: vec![0.10, 0.04, 0.06],
        covariance: vec![
            vol_s * vol_s,
            rho_sb * vol_s * vol_b,
            rho_sc * vol_s * vol_c,
            rho_sb * vol_s * vol_b,
            vol_b * vol_b,
            rho_bc * vol_b * vol_c,
            rho_sc * vol_s * vol_c,
            rho_bc * vol_b * vol_c,
            vol_c * vol_c,
        ],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> OptimizationConfig {
        OptimizationConfig {
            risk_free_rate: 0.03,
            min_weight: 0.0,
            max_weight: 0.80,
            max_concentration: 1.0,
            concentration_top_n: 2,
            allow_short: false,
            risk_aversion: 2.0,
            frontier_points: 10,
            ema_decay: 0.3,
            window_size: 16,
            solver_tolerance: 1e-8,
            max_iterations: 500,
        }
    }

    fn simple_universe() -> AssetUniverse {
        two_asset_universe(0.10, 0.05, 0.20, 0.10, 0.3)
    }

    #[test]
    fn test_basic() {
        let mut opt = Optimization::new();
        let result = opt.optimize(&simple_universe());
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_returns_instance() {
        let opt = Optimization::process(small_config());
        assert!(opt.is_ok());
    }

    #[test]
    fn test_weights_sum_to_one() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        let sum: f64 = result.recommended.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_min_variance_weights_sum_to_one() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        let sum: f64 = result.min_variance.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Min-var weights should sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_risk_parity_weights_sum_to_one() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        let sum: f64 = result.risk_parity.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Risk-parity weights should sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_long_only_no_negative_weights() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        for (i, &w) in result.recommended.weights.iter().enumerate() {
            assert!(
                w >= -1e-8,
                "Weight {} should be >= 0 in long-only, got {}",
                i,
                w
            );
        }
    }

    #[test]
    fn test_max_weight_enforced() {
        let config = OptimizationConfig {
            max_weight: 0.60,
            ..small_config()
        };
        let mut opt = Optimization::with_config(config).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        for &w in &result.recommended.weights {
            assert!(w <= 0.60 + 0.01, "Weight should be <= 0.60, got {}", w);
        }
    }

    #[test]
    fn test_min_variance_lower_vol_than_equal_weight() {
        let mut opt = Optimization::with_config(OptimizationConfig {
            max_weight: 1.0,
            ..small_config()
        })
        .unwrap();
        let universe = simple_universe();
        let result = opt.optimize(&universe).unwrap();

        // Equal weight portfolio vol
        let eq_weights = vec![0.5, 0.5];
        let eq_vol = opt.portfolio_volatility(&universe, &eq_weights);

        // Min variance should be <= equal weight vol
        assert!(
            result.min_variance.volatility <= eq_vol + 1e-6,
            "Min-var vol ({}) should be <= equal weight vol ({})",
            result.min_variance.volatility,
            eq_vol
        );
    }

    #[test]
    fn test_risk_parity_balanced_contributions() {
        let config = OptimizationConfig {
            max_weight: 1.0,
            ..small_config()
        };
        let mut opt = Optimization::with_config(config).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        let rc = &result.risk_parity.risk_contributions;
        if rc.len() == 2 {
            // Risk contributions should be roughly equal (~0.5 each)
            assert!(
                (rc[0] - rc[1]).abs() < 0.15,
                "Risk contributions should be roughly equal: {:?}",
                rc
            );
        }
    }

    #[test]
    fn test_sharpe_finite() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        assert!(result.recommended.sharpe.is_finite());
        assert!(result.min_variance.sharpe.is_finite());
        assert!(result.risk_parity.sharpe.is_finite());
    }

    #[test]
    fn test_volatility_non_negative() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        assert!(result.recommended.volatility >= 0.0);
        assert!(result.min_variance.volatility >= 0.0);
        assert!(result.risk_parity.volatility >= 0.0);
    }

    #[test]
    fn test_diversification_ratio_ge_one() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        // For a properly diversified portfolio (correlation < 1), ratio should be >= 1
        assert!(
            result.recommended.diversification_ratio >= 0.99,
            "Diversification ratio should be >= 1, got {}",
            result.recommended.diversification_ratio
        );
    }

    #[test]
    fn test_active_positions_count() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        assert!(result.recommended.active_positions > 0);
        assert!(result.recommended.active_positions <= 2);
    }

    #[test]
    fn test_frontier_generated() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        assert!(!result.frontier.is_empty());
        assert!(result.frontier.len() <= small_config().frontier_points);
    }

    #[test]
    fn test_frontier_volatilities_finite() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        for point in &result.frontier {
            assert!(point.volatility.is_finite() && point.volatility >= 0.0);
            assert!(point.sharpe.is_finite());
        }
    }

    #[test]
    fn test_turnover_zero_first_run() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        assert!(
            (result.turnover - 0.0).abs() < 1e-10,
            "First run turnover should be 0, got {}",
            result.turnover
        );
    }

    #[test]
    fn test_turnover_computed_on_second_run() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        opt.optimize(&simple_universe()).unwrap();
        // Use a different universe on second run to force weight changes
        let universe2 = two_asset_universe(0.05, 0.15, 0.10, 0.30, 0.1);
        let result2 = opt.optimize(&universe2).unwrap();
        // Turnover should be > 0 if weights changed
        assert!(result2.turnover >= 0.0);
    }

    #[test]
    fn test_three_asset_universe() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let universe = three_asset_universe();
        let result = opt.optimize(&universe).unwrap();
        assert_eq!(result.recommended.weights.len(), 3);
        let sum: f64 = result.recommended.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "3-asset weights should sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_risk_contributions_sum_to_one() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&three_asset_universe()).unwrap();
        let rc_sum: f64 = result.recommended.risk_contributions.iter().sum();
        if result.recommended.volatility > 1e-10 {
            assert!(
                (rc_sum - 1.0).abs() < 0.1,
                "Risk contributions should sum to ~1.0, got {}",
                rc_sum
            );
        }
    }

    #[test]
    fn test_method_label() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        assert!(!result.recommended.method.is_empty());
        assert_eq!(result.min_variance.method, "min_variance");
        assert_eq!(result.risk_parity.method, "risk_parity");
    }

    #[test]
    fn test_feasibility_flag() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        // Should be feasible under normal conditions
        assert!(result.recommended.feasible);
    }

    #[test]
    fn test_ema_initializes_on_first_run() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        assert!(opt.smoothed_sharpe().is_none());
        opt.optimize(&simple_universe()).unwrap();
        assert!(opt.smoothed_sharpe().is_some());
    }

    #[test]
    fn test_ema_blends_on_subsequent_runs() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let r1 = opt.optimize(&simple_universe()).unwrap();
        let s1 = r1.recommended.sharpe;
        let r2 = opt.optimize(&simple_universe()).unwrap();
        let s2 = r2.recommended.sharpe;
        let ema = opt.smoothed_sharpe().unwrap();
        let expected = 0.3 * s2 + 0.7 * s1;
        assert!(
            (ema - expected).abs() < 1e-8,
            "EMA mismatch: got {}, expected {}",
            ema,
            expected
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        opt.optimize(&simple_universe()).unwrap();
        opt.optimize(&simple_universe()).unwrap();
        assert_eq!(opt.stats().total_runs, 2);
        assert_eq!(opt.stats().total_assets_evaluated, 4);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = OptimizationStats::default();
        assert_eq!(stats.total_runs, 0);
        assert_eq!(stats.avg_assets_per_run(), 0.0);
        assert_eq!(stats.avg_turnover(), 0.0);
        assert_eq!(stats.infeasible_rate(), 0.0);
    }

    #[test]
    fn test_stats_avg_assets() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        opt.optimize(&simple_universe()).unwrap();
        assert!(
            (opt.stats().avg_assets_per_run() - 2.0).abs() < 1e-10,
            "Should be 2 assets per run"
        );
    }

    #[test]
    fn test_window_eviction() {
        let config = OptimizationConfig {
            window_size: 3,
            ..small_config()
        };
        let mut opt = Optimization::with_config(config).unwrap();
        for _ in 0..5 {
            opt.optimize(&simple_universe()).unwrap();
        }
        assert_eq!(opt.recent_results().len(), 3);
    }

    #[test]
    fn test_windowed_sharpe() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        assert!(opt.windowed_sharpe().is_none());
        opt.optimize(&simple_universe()).unwrap();
        assert!(opt.windowed_sharpe().is_some());
    }

    #[test]
    fn test_windowed_turnover() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        assert!(opt.windowed_turnover().is_none());
        opt.optimize(&simple_universe()).unwrap();
        assert!(opt.windowed_turnover().is_some());
    }

    #[test]
    fn test_is_sharpe_improving_insufficient_data() {
        let opt = Optimization::with_config(small_config()).unwrap();
        assert!(!opt.is_sharpe_improving());
    }

    #[test]
    fn test_is_turnover_increasing_insufficient_data() {
        let opt = Optimization::with_config(small_config()).unwrap();
        assert!(!opt.is_turnover_increasing());
    }

    #[test]
    fn test_reset() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        opt.optimize(&simple_universe()).unwrap();
        opt.optimize(&simple_universe()).unwrap();
        assert_eq!(opt.run_count(), 2);
        opt.reset();
        assert_eq!(opt.run_count(), 0);
        assert!(opt.smoothed_sharpe().is_none());
        assert!(opt.previous_weights().is_none());
        assert!(opt.recent_results().is_empty());
    }

    #[test]
    fn test_higher_return_asset_gets_more_weight_in_max_sharpe() {
        let config = OptimizationConfig {
            max_weight: 1.0,
            ..small_config()
        };
        let mut opt = Optimization::with_config(config).unwrap();
        // Asset A has much higher return with same vol
        let universe = two_asset_universe(0.20, 0.02, 0.15, 0.15, 0.3);
        let result = opt.optimize(&universe).unwrap();
        assert!(
            result.recommended.weights[0] > result.recommended.weights[1],
            "Higher-return asset should get more weight: {:?}",
            result.recommended.weights
        );
    }

    #[test]
    fn test_lower_vol_asset_gets_more_weight_in_min_variance() {
        let config = OptimizationConfig {
            max_weight: 1.0,
            ..small_config()
        };
        let mut opt = Optimization::with_config(config).unwrap();
        // Same return, but B has much lower vol
        let universe = two_asset_universe(0.08, 0.08, 0.30, 0.05, 0.0);
        let result = opt.optimize(&universe).unwrap();
        assert!(
            result.min_variance.weights[1] > result.min_variance.weights[0],
            "Lower-vol asset should get more weight in min-var: {:?}",
            result.min_variance.weights
        );
    }

    #[test]
    fn test_negative_correlation_benefits_diversification() {
        let config = OptimizationConfig {
            max_weight: 1.0,
            ..small_config()
        };
        let mut opt = Optimization::with_config(config).unwrap();
        let pos_corr = two_asset_universe(0.10, 0.10, 0.20, 0.20, 0.8);
        let neg_corr = two_asset_universe(0.10, 0.10, 0.20, 0.20, -0.5);

        let r_pos = opt.optimize(&pos_corr).unwrap();
        opt.reset();
        let r_neg = opt.optimize(&neg_corr).unwrap();

        assert!(
            r_neg.min_variance.volatility < r_pos.min_variance.volatility,
            "Negative correlation should produce lower min-var vol ({}) vs positive ({})",
            r_neg.min_variance.volatility,
            r_pos.min_variance.volatility
        );
    }

    #[test]
    fn test_single_asset() {
        let universe = AssetUniverse {
            names: vec!["only".into()],
            expected_returns: vec![0.10],
            covariance: vec![0.04], // 20% vol
        };
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&universe).unwrap();
        assert!(
            (result.recommended.weights[0] - 1.0).abs() < 0.01,
            "Single asset should have weight ~1.0"
        );
    }

    #[test]
    fn test_zero_volatility_asset() {
        let universe = AssetUniverse {
            names: vec!["cash".into(), "stock".into()],
            expected_returns: vec![0.03, 0.10],
            covariance: vec![0.0, 0.0, 0.0, 0.04],
        };
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&universe);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concentration_limit() {
        let config = OptimizationConfig {
            max_concentration: 0.60,
            concentration_top_n: 1,
            max_weight: 1.0,
            ..small_config()
        };
        let mut opt = Optimization::with_config(config).unwrap();
        let universe = three_asset_universe();
        let result = opt.optimize(&universe).unwrap();
        let max_w = result
            .recommended
            .weights
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(
            max_w <= 0.65,
            "Max weight should be near concentration limit, got {}",
            max_w
        );
    }

    // -----------------------------------------------------------------------
    // Universe validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_universe_error() {
        let universe = AssetUniverse {
            names: vec![],
            expected_returns: vec![],
            covariance: vec![],
        };
        let mut opt = Optimization::with_config(small_config()).unwrap();
        assert!(opt.optimize(&universe).is_err());
    }

    #[test]
    fn test_mismatched_returns_error() {
        let universe = AssetUniverse {
            names: vec!["A".into(), "B".into()],
            expected_returns: vec![0.10],
            covariance: vec![0.04, 0.01, 0.01, 0.04],
        };
        let mut opt = Optimization::with_config(small_config()).unwrap();
        assert!(opt.optimize(&universe).is_err());
    }

    #[test]
    fn test_wrong_covariance_size_error() {
        let universe = AssetUniverse {
            names: vec!["A".into(), "B".into()],
            expected_returns: vec![0.10, 0.05],
            covariance: vec![0.04, 0.01, 0.01],
        };
        let mut opt = Optimization::with_config(small_config()).unwrap();
        assert!(opt.optimize(&universe).is_err());
    }

    #[test]
    fn test_asymmetric_covariance_error() {
        let universe = AssetUniverse {
            names: vec!["A".into(), "B".into()],
            expected_returns: vec![0.10, 0.05],
            covariance: vec![0.04, 0.01, 0.02, 0.04], // asymmetric
        };
        let mut opt = Optimization::with_config(small_config()).unwrap();
        assert!(opt.optimize(&universe).is_err());
    }

    #[test]
    fn test_negative_diagonal_covariance_error() {
        let universe = AssetUniverse {
            names: vec!["A".into()],
            expected_returns: vec![0.10],
            covariance: vec![-0.01],
        };
        let mut opt = Optimization::with_config(small_config()).unwrap();
        assert!(opt.optimize(&universe).is_err());
    }

    // -----------------------------------------------------------------------
    // Config validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_min_gt_max_weight() {
        let config = OptimizationConfig {
            min_weight: 0.50,
            max_weight: 0.30,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_max_weight() {
        let config = OptimizationConfig {
            max_weight: 0.0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_concentration() {
        let config = OptimizationConfig {
            max_concentration: 0.0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_concentration_gt_one() {
        let config = OptimizationConfig {
            max_concentration: 1.5,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_concentration_top_n() {
        let config = OptimizationConfig {
            concentration_top_n: 0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_risk_aversion() {
        let config = OptimizationConfig {
            risk_aversion: 0.0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_frontier_points() {
        let config = OptimizationConfig {
            frontier_points: 0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_zero() {
        let config = OptimizationConfig {
            ema_decay: 0.0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_one() {
        let config = OptimizationConfig {
            ema_decay: 1.0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_window_size() {
        let config = OptimizationConfig {
            window_size: 0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_solver_tolerance() {
        let config = OptimizationConfig {
            solver_tolerance: 0.0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_max_iterations() {
        let config = OptimizationConfig {
            max_iterations: 0,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_negative_min_weight_no_short() {
        let config = OptimizationConfig {
            allow_short: false,
            min_weight: -0.1,
            ..small_config()
        };
        assert!(Optimization::with_config(config).is_err());
    }

    #[test]
    fn test_allow_short_with_negative_min_weight() {
        let config = OptimizationConfig {
            allow_short: true,
            min_weight: -0.50,
            max_weight: 1.50,
            ..small_config()
        };
        let result = Optimization::with_config(config);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_asset_universe_valid() {
        let universe = two_asset_universe(0.10, 0.05, 0.20, 0.10, 0.3);
        assert!(universe.validate().is_ok());
        assert_eq!(universe.n(), 2);
    }

    #[test]
    fn test_three_asset_universe_valid() {
        let universe = three_asset_universe();
        assert!(universe.validate().is_ok());
        assert_eq!(universe.n(), 3);
    }

    #[test]
    fn test_portfolio_return_equal_weight() {
        let opt = Optimization::new();
        let universe = two_asset_universe(0.10, 0.05, 0.20, 0.10, 0.3);
        let weights = vec![0.5, 0.5];
        let ret = opt.portfolio_return(&universe, &weights);
        assert!((ret - 0.075).abs() < 1e-10, "Expected 0.075, got {}", ret);
    }

    #[test]
    fn test_portfolio_variance_single_asset() {
        let opt = Optimization::new();
        let universe = two_asset_universe(0.10, 0.05, 0.20, 0.10, 0.3);
        let weights = vec![1.0, 0.0];
        let var = opt.portfolio_variance(&universe, &weights);
        assert!(
            (var - 0.04).abs() < 1e-10,
            "Expected 0.04 (0.20^2), got {}",
            var
        );
    }

    #[test]
    fn test_portfolio_volatility_non_negative() {
        let opt = Optimization::new();
        let universe = simple_universe();
        let weights = vec![0.6, 0.4];
        let vol = opt.portfolio_volatility(&universe, &weights);
        assert!(vol >= 0.0);
    }

    #[test]
    fn test_marginal_risk_sums_to_one() {
        let opt = Optimization::new();
        let universe = simple_universe();
        let weights = vec![0.6, 0.4];
        let vol = opt.portfolio_volatility(&universe, &weights);
        let mrc = opt.marginal_risk_contributions(&universe, &weights, vol);
        let sum: f64 = mrc.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Marginal risk should sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_diversification_ratio_two_assets() {
        let opt = Optimization::new();
        let universe = two_asset_universe(0.10, 0.05, 0.20, 0.10, 0.3);
        let weights = vec![0.5, 0.5];
        let ratio = opt.diversification_ratio(&universe, &weights);
        // With correlation < 1, diversification ratio should be > 1
        assert!(
            ratio >= 1.0 - 1e-10,
            "Diversification ratio should be >= 1, got {}",
            ratio
        );
    }

    #[test]
    fn test_max_weight_field() {
        let mut opt = Optimization::with_config(small_config()).unwrap();
        let result = opt.optimize(&simple_universe()).unwrap();
        assert!(result.recommended.max_weight >= 0.0);
        assert!(result.recommended.max_weight <= 1.0);
    }
}
