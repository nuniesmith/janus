//! Monte Carlo simulation engine for strategy evaluation
//!
//! Part of the Cortex region
//! Component: planning
//!
//! Runs Monte Carlo simulations to estimate the distribution of future
//! portfolio returns under configurable stochastic models. Produces
//! risk metrics (VaR, CVaR, Sharpe, max drawdown) and convergence
//! diagnostics so upstream planners can size positions and evaluate
//! strategy alternatives with statistical confidence.
//!
//! Key features:
//! - Geometric Brownian Motion (GBM) path generation with drift & vol
//! - Configurable number of paths, time steps, and horizon
//! - Value-at-Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
//! - Maximum drawdown distribution across simulated paths
//! - Sharpe ratio estimation from simulated return distributions
//! - Convergence detection via running standard error of the mean
//! - EMA-smoothed result tracking across successive simulation runs
//! - Sliding window of recent simulation summaries for trend analysis
//! - Running statistics with per-run and cumulative tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Monte Carlo simulation engine
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of simulation paths per run
    pub num_paths: usize,
    /// Number of time steps per path
    pub num_steps: usize,
    /// Simulation horizon in days
    pub horizon_days: f64,
    /// Annualised expected return (drift) e.g. 0.10 = 10%
    pub drift: f64,
    /// Annualised volatility e.g. 0.20 = 20%
    pub volatility: f64,
    /// Initial portfolio value
    pub initial_value: f64,
    /// VaR confidence level (e.g. 0.95 for 95% VaR)
    pub var_confidence: f64,
    /// Risk-free rate for Sharpe calculation (annualised)
    pub risk_free_rate: f64,
    /// Seed for reproducible simulations (0 = use run counter)
    pub seed: u64,
    /// EMA decay for smoothing results across runs (0 < decay < 1)
    pub ema_decay: f64,
    /// Maximum recent summaries to keep in sliding window
    pub window_size: usize,
    /// Convergence threshold: stop early if SEM / mean < threshold
    pub convergence_threshold: f64,
    /// Minimum paths before convergence check
    pub min_paths_for_convergence: usize,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_paths: 10_000,
            num_steps: 252,
            horizon_days: 252.0,
            drift: 0.08,
            volatility: 0.20,
            initial_value: 100_000.0,
            var_confidence: 0.95,
            risk_free_rate: 0.04,
            seed: 0,
            ema_decay: 0.3,
            window_size: 64,
            convergence_threshold: 0.005,
            min_paths_for_convergence: 500,
        }
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Summary statistics for a single Monte Carlo simulation run
#[derive(Debug, Clone)]
pub struct SimulationSummary {
    /// Mean terminal return (fractional, e.g. 0.08 = +8%)
    pub mean_return: f64,
    /// Standard deviation of terminal returns
    pub std_return: f64,
    /// Median terminal return
    pub median_return: f64,
    /// Skewness of the return distribution
    pub skewness: f64,
    /// Excess kurtosis of the return distribution
    pub kurtosis: f64,
    /// Value-at-Risk at the configured confidence level (loss, positive = bad)
    pub var: f64,
    /// Conditional VaR / Expected Shortfall (loss, positive = bad)
    pub cvar: f64,
    /// Mean of maximum drawdown across all paths
    pub mean_max_drawdown: f64,
    /// Worst (largest) maximum drawdown observed
    pub worst_max_drawdown: f64,
    /// Simulated annualised Sharpe ratio
    pub sharpe: f64,
    /// Probability of loss (fraction of paths with negative return)
    pub loss_probability: f64,
    /// 5th percentile terminal return
    pub percentile_5: f64,
    /// 25th percentile terminal return
    pub percentile_25: f64,
    /// 75th percentile terminal return
    pub percentile_75: f64,
    /// 95th percentile terminal return
    pub percentile_95: f64,
    /// Number of paths simulated
    pub paths_simulated: usize,
    /// Whether the simulation converged early
    pub converged: bool,
    /// Standard error of the mean return estimate
    pub standard_error: f64,
    /// Confidence score [0, 1] based on sample size and convergence
    pub confidence: f64,
}

/// Cumulative statistics across multiple simulation runs
#[derive(Debug, Clone)]
pub struct MonteCarloStats {
    /// Total number of simulation runs
    pub total_runs: usize,
    /// Total paths simulated across all runs
    pub total_paths: usize,
    /// EMA-smoothed mean return
    pub ema_mean_return: f64,
    /// EMA-smoothed VaR
    pub ema_var: f64,
    /// EMA-smoothed CVaR
    pub ema_cvar: f64,
    /// EMA-smoothed Sharpe
    pub ema_sharpe: f64,
    /// EMA-smoothed mean max drawdown
    pub ema_mean_max_dd: f64,
    /// Best mean return observed in any single run
    pub best_mean_return: f64,
    /// Worst mean return observed in any single run
    pub worst_mean_return: f64,
    /// Number of runs that converged early
    pub converged_runs: usize,
    /// Sum of squared mean-return for variance tracking
    pub sum_sq_mean_return: f64,
}

impl Default for MonteCarloStats {
    fn default() -> Self {
        Self {
            total_runs: 0,
            total_paths: 0,
            ema_mean_return: 0.0,
            ema_var: 0.0,
            ema_cvar: 0.0,
            ema_sharpe: 0.0,
            ema_mean_max_dd: 0.0,
            best_mean_return: f64::NEG_INFINITY,
            worst_mean_return: f64::INFINITY,
            converged_runs: 0,
            sum_sq_mean_return: 0.0,
        }
    }
}

impl MonteCarloStats {
    /// Variance of mean returns across runs
    pub fn mean_return_variance(&self) -> f64 {
        if self.total_runs < 2 {
            return 0.0;
        }
        let avg = self.ema_mean_return;
        (self.sum_sq_mean_return / self.total_runs as f64 - avg * avg).max(0.0)
    }

    /// Standard deviation of mean returns across runs
    pub fn mean_return_std(&self) -> f64 {
        self.mean_return_variance().sqrt()
    }

    /// Fraction of runs that converged early
    pub fn convergence_rate(&self) -> f64 {
        if self.total_runs == 0 {
            return 0.0;
        }
        self.converged_runs as f64 / self.total_runs as f64
    }
}

// ---------------------------------------------------------------------------
// PRNG — simple xoshiro256** for reproducibility without external deps
// ---------------------------------------------------------------------------

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to seed state from a single u64
        let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        // Ensure non-zero state
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-300 {
                return (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core engine
// ---------------------------------------------------------------------------

/// Monte Carlo simulation engine for strategy evaluation
pub struct MonteCarlo {
    config: MonteCarloConfig,
    run_counter: u64,
    ema_initialized: bool,
    recent: VecDeque<SimulationSummary>,
    stats: MonteCarloStats,
}

impl Default for MonteCarlo {
    fn default() -> Self {
        Self::new()
    }
}

impl MonteCarlo {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            config: MonteCarloConfig::default(),
            run_counter: 0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: MonteCarloStats::default(),
        }
    }

    /// Create from validated config (returns `Err` on invalid params)
    pub fn with_config(config: MonteCarloConfig) -> Result<Self> {
        if config.num_paths == 0 {
            return Err(Error::Configuration("num_paths must be > 0".into()));
        }
        if config.num_steps == 0 {
            return Err(Error::Configuration("num_steps must be > 0".into()));
        }
        if config.horizon_days <= 0.0 {
            return Err(Error::Configuration("horizon_days must be > 0".into()));
        }
        if config.volatility < 0.0 {
            return Err(Error::Configuration("volatility must be >= 0".into()));
        }
        if config.initial_value <= 0.0 {
            return Err(Error::Configuration("initial_value must be > 0".into()));
        }
        if config.var_confidence <= 0.0 || config.var_confidence >= 1.0 {
            return Err(Error::Configuration(
                "var_confidence must be in (0, 1)".into(),
            ));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::Configuration("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::Configuration("window_size must be > 0".into()));
        }
        if config.convergence_threshold <= 0.0 {
            return Err(Error::Configuration(
                "convergence_threshold must be > 0".into(),
            ));
        }
        Ok(Self {
            config,
            run_counter: 0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: MonteCarloStats::default(),
        })
    }

    /// Convenience: validate and create
    pub fn process(config: MonteCarloConfig) -> Result<Self> {
        Self::with_config(config)
    }

    // -----------------------------------------------------------------------
    // Simulation
    // -----------------------------------------------------------------------

    /// Run a full Monte Carlo simulation and return a summary.
    pub fn simulate(&mut self) -> SimulationSummary {
        self.run_counter += 1;
        let seed = if self.config.seed != 0 {
            self.config.seed.wrapping_add(self.run_counter)
        } else {
            self.run_counter
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(1)
        };
        let mut rng = Rng::new(seed);

        let dt = self.config.horizon_days / 252.0 / self.config.num_steps as f64;
        let drift_dt =
            (self.config.drift - 0.5 * self.config.volatility * self.config.volatility) * dt;
        let vol_sqrt_dt = self.config.volatility * dt.sqrt();

        let mut terminal_returns = Vec::with_capacity(self.config.num_paths);
        let mut max_drawdowns = Vec::with_capacity(self.config.num_paths);

        let mut converged = false;
        let mut running_sum = 0.0;
        let mut running_sum_sq = 0.0;

        for i in 0..self.config.num_paths {
            let mut value = self.config.initial_value;
            let mut peak = value;
            let mut max_dd: f64 = 0.0;

            for _ in 0..self.config.num_steps {
                let z = rng.next_normal();
                value *= (drift_dt + vol_sqrt_dt * z).exp();
                if value > peak {
                    peak = value;
                }
                let dd = (peak - value) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }

            let ret = (value - self.config.initial_value) / self.config.initial_value;
            terminal_returns.push(ret);
            max_drawdowns.push(max_dd);

            // Running convergence check
            running_sum += ret;
            running_sum_sq += ret * ret;
            let n = (i + 1) as f64;
            if i + 1 >= self.config.min_paths_for_convergence && i + 1 < self.config.num_paths {
                let mean = running_sum / n;
                let var = (running_sum_sq / n - mean * mean).max(0.0);
                let sem = (var / n).sqrt();
                if mean.abs() > 1e-12 && (sem / mean.abs()) < self.config.convergence_threshold {
                    converged = true;
                    break;
                }
            }
        }

        let summary = Self::compute_summary(
            &mut terminal_returns,
            &max_drawdowns,
            &self.config,
            converged,
        );

        self.update_stats(&summary);
        summary
    }

    /// Run simulation with a custom drift and volatility override (for scenario sweeps)
    pub fn simulate_with_params(
        &mut self,
        drift: f64,
        volatility: f64,
    ) -> Result<SimulationSummary> {
        if volatility < 0.0 {
            return Err(Error::Configuration("volatility must be >= 0".into()));
        }
        let saved_drift = self.config.drift;
        let saved_vol = self.config.volatility;
        self.config.drift = drift;
        self.config.volatility = volatility;
        let result = self.simulate();
        self.config.drift = saved_drift;
        self.config.volatility = saved_vol;
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Summary computation (static, operates on collected terminal returns)
    // -----------------------------------------------------------------------

    fn compute_summary(
        returns: &mut Vec<f64>,
        max_drawdowns: &[f64],
        config: &MonteCarloConfig,
        converged: bool,
    ) -> SimulationSummary {
        let n = returns.len();
        assert!(n > 0);
        let nf = n as f64;

        // Sort for percentile / VaR computation
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Basic moments
        let mean: f64 = returns.iter().sum::<f64>() / nf;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / nf;
        let std_dev = variance.sqrt();

        let skewness = if std_dev > 1e-15 {
            returns
                .iter()
                .map(|r| ((r - mean) / std_dev).powi(3))
                .sum::<f64>()
                / nf
        } else {
            0.0
        };

        let kurtosis = if std_dev > 1e-15 {
            returns
                .iter()
                .map(|r| ((r - mean) / std_dev).powi(4))
                .sum::<f64>()
                / nf
                - 3.0
        } else {
            0.0
        };

        // Percentiles
        let percentile = |p: f64| -> f64 {
            let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
            returns[idx]
        };

        let median = percentile(0.5);
        let p5 = percentile(0.05);
        let p25 = percentile(0.25);
        let p75 = percentile(0.75);
        let p95 = percentile(0.95);

        // VaR: loss at confidence level (positive means loss)
        let var_idx =
            (((1.0 - config.var_confidence) * (n as f64 - 1.0)).round() as usize).min(n - 1);
        let var = -returns[var_idx]; // negate so positive = loss

        // CVaR: mean of returns below the VaR threshold
        let tail_count = var_idx + 1;
        let cvar = if tail_count > 0 {
            -returns[..tail_count].iter().sum::<f64>() / tail_count as f64
        } else {
            var
        };

        // Max drawdown stats
        let mean_max_dd: f64 = if max_drawdowns.is_empty() {
            0.0
        } else {
            max_drawdowns.iter().sum::<f64>() / max_drawdowns.len() as f64
        };
        let worst_max_dd: f64 = max_drawdowns.iter().cloned().fold(0.0_f64, f64::max);

        // Sharpe (annualised from per-horizon returns)
        let horizon_years = config.horizon_days / 252.0;
        let annualised_mean = mean / horizon_years;
        let annualised_std = std_dev / horizon_years.sqrt();
        let sharpe = if annualised_std > 1e-15 {
            (annualised_mean - config.risk_free_rate) / annualised_std
        } else {
            0.0
        };

        // Loss probability
        let loss_count = returns.iter().filter(|&&r| r < 0.0).count();
        let loss_probability = loss_count as f64 / nf;

        // Standard error
        let sem = if nf > 1.0 { std_dev / nf.sqrt() } else { 0.0 };

        // Confidence
        let sample_factor = (nf / 1000.0).min(1.0);
        let convergence_factor = if converged { 1.0 } else { 0.8 };
        let sem_factor = if mean.abs() > 1e-12 {
            (1.0 - (sem / mean.abs()).min(1.0)).max(0.0)
        } else {
            0.5
        };
        let confidence =
            (sample_factor * 0.4 + convergence_factor * 0.3 + sem_factor * 0.3).clamp(0.0, 1.0);

        SimulationSummary {
            mean_return: mean,
            std_return: std_dev,
            median_return: median,
            skewness,
            kurtosis,
            var,
            cvar,
            mean_max_drawdown: mean_max_dd,
            worst_max_drawdown: worst_max_dd,
            sharpe,
            loss_probability,
            percentile_5: p5,
            percentile_25: p25,
            percentile_75: p75,
            percentile_95: p95,
            paths_simulated: n,
            converged,
            standard_error: sem,
            confidence,
        }
    }

    // -----------------------------------------------------------------------
    // Stats update
    // -----------------------------------------------------------------------

    fn update_stats(&mut self, summary: &SimulationSummary) {
        let decay = self.config.ema_decay;

        if !self.ema_initialized {
            self.stats.ema_mean_return = summary.mean_return;
            self.stats.ema_var = summary.var;
            self.stats.ema_cvar = summary.cvar;
            self.stats.ema_sharpe = summary.sharpe;
            self.stats.ema_mean_max_dd = summary.mean_max_drawdown;
            self.ema_initialized = true;
        } else {
            self.stats.ema_mean_return =
                decay * summary.mean_return + (1.0 - decay) * self.stats.ema_mean_return;
            self.stats.ema_var = decay * summary.var + (1.0 - decay) * self.stats.ema_var;
            self.stats.ema_cvar = decay * summary.cvar + (1.0 - decay) * self.stats.ema_cvar;
            self.stats.ema_sharpe = decay * summary.sharpe + (1.0 - decay) * self.stats.ema_sharpe;
            self.stats.ema_mean_max_dd =
                decay * summary.mean_max_drawdown + (1.0 - decay) * self.stats.ema_mean_max_dd;
        }

        self.stats.total_runs += 1;
        self.stats.total_paths += summary.paths_simulated;
        self.stats.sum_sq_mean_return += summary.mean_return * summary.mean_return;

        if summary.mean_return > self.stats.best_mean_return {
            self.stats.best_mean_return = summary.mean_return;
        }
        if summary.mean_return < self.stats.worst_mean_return {
            self.stats.worst_mean_return = summary.mean_return;
        }
        if summary.converged {
            self.stats.converged_runs += 1;
        }

        // Window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(summary.clone());
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Cumulative statistics across all simulation runs
    pub fn stats(&self) -> &MonteCarloStats {
        &self.stats
    }

    /// Number of simulation runs completed
    pub fn run_count(&self) -> usize {
        self.stats.total_runs
    }

    /// Current configuration
    pub fn config(&self) -> &MonteCarloConfig {
        &self.config
    }

    /// Recent simulation summaries in the sliding window
    pub fn recent_summaries(&self) -> &VecDeque<SimulationSummary> {
        &self.recent
    }

    /// EMA-smoothed mean return
    pub fn smoothed_mean_return(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_mean_return)
        } else {
            None
        }
    }

    /// EMA-smoothed VaR
    pub fn smoothed_var(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_var)
        } else {
            None
        }
    }

    /// EMA-smoothed CVaR
    pub fn smoothed_cvar(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_cvar)
        } else {
            None
        }
    }

    /// EMA-smoothed Sharpe ratio
    pub fn smoothed_sharpe(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_sharpe)
        } else {
            None
        }
    }

    /// Windowed mean of mean-returns across recent runs
    pub fn windowed_mean_return(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.mean_return).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed mean VaR
    pub fn windowed_var(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|s| s.var).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed standard deviation of mean returns across recent runs
    pub fn windowed_return_std(&self) -> Option<f64> {
        if self.recent.len() < 2 {
            return None;
        }
        let n = self.recent.len() as f64;
        let mean = self.recent.iter().map(|s| s.mean_return).sum::<f64>() / n;
        let var = self
            .recent
            .iter()
            .map(|s| (s.mean_return - mean).powi(2))
            .sum::<f64>()
            / n;
        Some(var.sqrt())
    }

    /// Whether risk metrics are trending worse (higher VaR) over recent window
    pub fn is_risk_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half: f64 =
            self.recent.iter().take(half).map(|s| s.var).sum::<f64>() / half as f64;
        let second_half: f64 = self.recent.iter().skip(half).map(|s| s.var).sum::<f64>()
            / (self.recent.len() - half) as f64;
        second_half > first_half * 1.05
    }

    /// Whether returns are trending better over recent window
    pub fn is_return_improving(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| s.mean_return)
            .sum::<f64>()
            / half as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| s.mean_return)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second_half > first_half
    }

    /// Reset all state (keeps config)
    pub fn reset(&mut self) {
        self.run_counter = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = MonteCarloStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> MonteCarloConfig {
        MonteCarloConfig {
            num_paths: 500,
            num_steps: 50,
            horizon_days: 252.0,
            drift: 0.08,
            volatility: 0.20,
            initial_value: 100_000.0,
            var_confidence: 0.95,
            risk_free_rate: 0.04,
            seed: 42,
            ema_decay: 0.3,
            window_size: 16,
            convergence_threshold: 0.001,
            min_paths_for_convergence: 100,
        }
    }

    #[test]
    fn test_basic() {
        let mut mc = MonteCarlo::new();
        let summary = mc.simulate();
        assert!(summary.paths_simulated > 0);
    }

    #[test]
    fn test_process_returns_instance() {
        let mc = MonteCarlo::process(small_config());
        assert!(mc.is_ok());
    }

    #[test]
    fn test_default_runs() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert_eq!(mc.run_count(), 1);
        assert!(s.paths_simulated > 0);
        assert!(s.paths_simulated <= 500);
    }

    #[test]
    fn test_mean_return_reasonable() {
        let mut mc = MonteCarlo::with_config(MonteCarloConfig {
            num_paths: 5_000,
            num_steps: 252,
            seed: 123,
            ..small_config()
        })
        .unwrap();
        let s = mc.simulate();
        // With 8% drift and 20% vol, mean return should be roughly around 0.08
        // but stochastic so use wide bounds
        assert!(
            s.mean_return > -0.50,
            "mean_return too low: {}",
            s.mean_return
        );
        assert!(
            s.mean_return < 0.80,
            "mean_return too high: {}",
            s.mean_return
        );
    }

    #[test]
    fn test_std_return_positive() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert!(s.std_return >= 0.0);
    }

    #[test]
    fn test_var_positive() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        // VaR is a loss number; at 95% confidence with reasonable vol,
        // it could be positive (there IS a loss at the tail)
        // Just ensure it's finite
        assert!(s.var.is_finite());
    }

    #[test]
    fn test_cvar_ge_var() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        // CVaR (expected shortfall) should be >= VaR by definition
        assert!(
            s.cvar >= s.var - 1e-10,
            "CVaR ({}) should be >= VaR ({})",
            s.cvar,
            s.var
        );
    }

    #[test]
    fn test_loss_probability_bounded() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert!(s.loss_probability >= 0.0);
        assert!(s.loss_probability <= 1.0);
    }

    #[test]
    fn test_max_drawdown_bounded() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert!(s.mean_max_drawdown >= 0.0);
        assert!(s.mean_max_drawdown <= 1.0);
        assert!(s.worst_max_drawdown >= s.mean_max_drawdown);
    }

    #[test]
    fn test_percentiles_ordered() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert!(
            s.percentile_5 <= s.percentile_25,
            "p5 ({}) > p25 ({})",
            s.percentile_5,
            s.percentile_25
        );
        assert!(
            s.percentile_25 <= s.median_return,
            "p25 ({}) > median ({})",
            s.percentile_25,
            s.median_return
        );
        assert!(
            s.median_return <= s.percentile_75,
            "median ({}) > p75 ({})",
            s.median_return,
            s.percentile_75
        );
        assert!(
            s.percentile_75 <= s.percentile_95,
            "p75 ({}) > p95 ({})",
            s.percentile_75,
            s.percentile_95
        );
    }

    #[test]
    fn test_sharpe_finite() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert!(s.sharpe.is_finite());
    }

    #[test]
    fn test_zero_volatility_no_randomness() {
        let config = MonteCarloConfig {
            volatility: 0.0,
            drift: 0.10,
            num_paths: 100,
            num_steps: 100,
            seed: 99,
            ..small_config()
        };
        let mut mc = MonteCarlo::with_config(config).unwrap();
        let s = mc.simulate();
        // With zero volatility, all paths should produce exactly the same return
        assert!(
            s.std_return < 1e-10,
            "std should be ~0 with zero vol, got {}",
            s.std_return
        );
        assert_eq!(s.loss_probability, 0.0);
    }

    #[test]
    fn test_high_volatility_wider_distribution() {
        let low_vol = {
            let config = MonteCarloConfig {
                volatility: 0.05,
                seed: 42,
                num_paths: 2_000,
                ..small_config()
            };
            let mut mc = MonteCarlo::with_config(config).unwrap();
            mc.simulate()
        };
        let high_vol = {
            let config = MonteCarloConfig {
                volatility: 0.50,
                seed: 42,
                num_paths: 2_000,
                ..small_config()
            };
            let mut mc = MonteCarlo::with_config(config).unwrap();
            mc.simulate()
        };
        assert!(
            high_vol.std_return > low_vol.std_return,
            "Higher vol ({}) should produce wider dist than lower vol ({})",
            high_vol.std_return,
            low_vol.std_return
        );
    }

    #[test]
    fn test_convergence_detection() {
        let config = MonteCarloConfig {
            num_paths: 50_000,
            num_steps: 50,
            convergence_threshold: 0.05, // generous threshold
            min_paths_for_convergence: 100,
            seed: 77,
            ..small_config()
        };
        let mut mc = MonteCarlo::with_config(config).unwrap();
        let s = mc.simulate();
        // With a generous threshold and many paths, should converge early
        if s.converged {
            assert!(s.paths_simulated < 50_000);
        }
        // Either way, result should be valid
        assert!(s.mean_return.is_finite());
    }

    #[test]
    fn test_standard_error_decreases_with_more_paths() {
        let small = {
            let config = MonteCarloConfig {
                num_paths: 100,
                seed: 1,
                convergence_threshold: 1e-10, // prevent early stop
                ..small_config()
            };
            let mut mc = MonteCarlo::with_config(config).unwrap();
            mc.simulate()
        };
        let large = {
            let config = MonteCarloConfig {
                num_paths: 10_000,
                seed: 1,
                convergence_threshold: 1e-10,
                ..small_config()
            };
            let mut mc = MonteCarlo::with_config(config).unwrap();
            mc.simulate()
        };
        assert!(
            large.standard_error < small.standard_error,
            "More paths ({}) should have lower SEM ({}) than fewer ({}: {})",
            large.paths_simulated,
            large.standard_error,
            small.paths_simulated,
            small.standard_error
        );
    }

    #[test]
    fn test_confidence_bounded() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert!(s.confidence >= 0.0 && s.confidence <= 1.0);
    }

    #[test]
    fn test_reproducible_with_seed() {
        let run = |seed: u64| {
            let config = MonteCarloConfig {
                seed,
                num_paths: 200,
                convergence_threshold: 1e-10,
                ..small_config()
            };
            let mut mc = MonteCarlo::with_config(config).unwrap();
            mc.simulate()
        };
        let a = run(42);
        let b = run(42);
        assert!(
            (a.mean_return - b.mean_return).abs() < 1e-12,
            "Same seed should produce identical results"
        );
    }

    #[test]
    fn test_different_seeds_different_results() {
        let run = |seed: u64| {
            let config = MonteCarloConfig {
                seed,
                num_paths: 500,
                convergence_threshold: 1e-10,
                ..small_config()
            };
            let mut mc = MonteCarlo::with_config(config).unwrap();
            mc.simulate()
        };
        let a = run(1);
        let b = run(999);
        // Extremely unlikely to be identical
        assert!(
            (a.mean_return - b.mean_return).abs() > 1e-10,
            "Different seeds should produce different results"
        );
    }

    #[test]
    fn test_ema_initializes_on_first_run() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        assert!(mc.smoothed_mean_return().is_none());
        let s = mc.simulate();
        let ema = mc.smoothed_mean_return().unwrap();
        assert!(
            (ema - s.mean_return).abs() < 1e-12,
            "First EMA should equal first value"
        );
    }

    #[test]
    fn test_ema_blends_on_subsequent_runs() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s1 = mc.simulate();
        let s2 = mc.simulate();
        let ema = mc.smoothed_mean_return().unwrap();
        // EMA should be between the two values (or at least not equal to s2 alone)
        let expected = 0.3 * s2.mean_return + 0.7 * s1.mean_return;
        assert!(
            (ema - expected).abs() < 1e-10,
            "EMA mismatch: got {}, expected {}",
            ema,
            expected
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        mc.simulate();
        mc.simulate();
        mc.simulate();
        assert_eq!(mc.stats().total_runs, 3);
        assert!(mc.stats().total_paths > 0);
    }

    #[test]
    fn test_stats_best_worst_mean() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        mc.simulate();
        mc.simulate();
        mc.simulate();
        assert!(mc.stats().best_mean_return >= mc.stats().worst_mean_return);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = MonteCarloStats::default();
        assert_eq!(stats.total_runs, 0);
        assert_eq!(stats.total_paths, 0);
        assert_eq!(stats.mean_return_variance(), 0.0);
        assert_eq!(stats.convergence_rate(), 0.0);
    }

    #[test]
    fn test_windowed_mean_return() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        assert!(mc.windowed_mean_return().is_none());
        mc.simulate();
        assert!(mc.windowed_mean_return().is_some());
    }

    #[test]
    fn test_windowed_var() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        assert!(mc.windowed_var().is_none());
        mc.simulate();
        assert!(mc.windowed_var().is_some());
    }

    #[test]
    fn test_windowed_return_std() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        assert!(mc.windowed_return_std().is_none());
        mc.simulate();
        // Need at least 2 runs for std
        assert!(mc.windowed_return_std().is_none());
        mc.simulate();
        assert!(mc.windowed_return_std().is_some());
    }

    #[test]
    fn test_window_eviction() {
        let config = MonteCarloConfig {
            window_size: 3,
            num_paths: 100,
            convergence_threshold: 1e-10,
            ..small_config()
        };
        let mut mc = MonteCarlo::with_config(config).unwrap();
        for _ in 0..5 {
            mc.simulate();
        }
        assert_eq!(mc.recent_summaries().len(), 3);
    }

    #[test]
    fn test_simulate_with_params() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let result = mc.simulate_with_params(0.05, 0.30);
        assert!(result.is_ok());
        let s = result.unwrap();
        assert!(s.mean_return.is_finite());
        // Config should be restored
        assert!((mc.config().drift - 0.08).abs() < 1e-12);
        assert!((mc.config().volatility - 0.20).abs() < 1e-12);
    }

    #[test]
    fn test_simulate_with_params_negative_vol() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let result = mc.simulate_with_params(0.05, -0.10);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_risk_increasing_insufficient_data() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        assert!(!mc.is_risk_increasing());
        mc.simulate();
        assert!(!mc.is_risk_increasing());
    }

    #[test]
    fn test_is_return_improving_insufficient_data() {
        let mc = MonteCarlo::with_config(small_config()).unwrap();
        assert!(!mc.is_return_improving());
    }

    #[test]
    fn test_reset() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        mc.simulate();
        mc.simulate();
        assert_eq!(mc.run_count(), 2);
        mc.reset();
        assert_eq!(mc.run_count(), 0);
        assert!(mc.smoothed_mean_return().is_none());
        assert!(mc.recent_summaries().is_empty());
        assert_eq!(mc.stats().total_runs, 0);
    }

    #[test]
    fn test_negative_drift_increases_loss_probability() {
        let pos_drift = {
            let config = MonteCarloConfig {
                drift: 0.20,
                num_paths: 2_000,
                seed: 10,
                convergence_threshold: 1e-10,
                ..small_config()
            };
            let mut mc = MonteCarlo::with_config(config).unwrap();
            mc.simulate()
        };
        let neg_drift = {
            let config = MonteCarloConfig {
                drift: -0.20,
                num_paths: 2_000,
                seed: 10,
                convergence_threshold: 1e-10,
                ..small_config()
            };
            let mut mc = MonteCarlo::with_config(config).unwrap();
            mc.simulate()
        };
        assert!(
            neg_drift.loss_probability > pos_drift.loss_probability,
            "Negative drift loss prob ({}) should exceed positive ({})",
            neg_drift.loss_probability,
            pos_drift.loss_probability
        );
    }

    #[test]
    fn test_skewness_finite() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert!(s.skewness.is_finite());
    }

    #[test]
    fn test_kurtosis_finite() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        let s = mc.simulate();
        assert!(s.kurtosis.is_finite());
    }

    #[test]
    fn test_convergence_rate_tracking() {
        let config = MonteCarloConfig {
            num_paths: 50_000,
            convergence_threshold: 0.1, // very generous
            min_paths_for_convergence: 50,
            seed: 55,
            ..small_config()
        };
        let mut mc = MonteCarlo::with_config(config).unwrap();
        for _ in 0..5 {
            mc.simulate();
        }
        // convergence_rate should be in [0, 1]
        let rate = mc.stats().convergence_rate();
        assert!((0.0..=1.0).contains(&rate));
    }

    #[test]
    fn test_mean_return_variance_tracking() {
        let mut mc = MonteCarlo::with_config(small_config()).unwrap();
        mc.simulate();
        mc.simulate();
        mc.simulate();
        let var = mc.stats().mean_return_variance();
        assert!(var >= 0.0);
        let std = mc.stats().mean_return_std();
        assert!(std >= 0.0);
    }

    // -----------------------------------------------------------------------
    // Config validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_zero_paths() {
        let config = MonteCarloConfig {
            num_paths: 0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_steps() {
        let config = MonteCarloConfig {
            num_steps: 0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_horizon() {
        let config = MonteCarloConfig {
            horizon_days: 0.0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_negative_volatility() {
        let config = MonteCarloConfig {
            volatility: -0.1,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_initial_value() {
        let config = MonteCarloConfig {
            initial_value: 0.0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_var_confidence_zero() {
        let config = MonteCarloConfig {
            var_confidence: 0.0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_var_confidence_one() {
        let config = MonteCarloConfig {
            var_confidence: 1.0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_zero() {
        let config = MonteCarloConfig {
            ema_decay: 0.0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_one() {
        let config = MonteCarloConfig {
            ema_decay: 1.0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_window_size() {
        let config = MonteCarloConfig {
            window_size: 0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_convergence_threshold() {
        let config = MonteCarloConfig {
            convergence_threshold: 0.0,
            ..small_config()
        };
        assert!(MonteCarlo::with_config(config).is_err());
    }

    // -----------------------------------------------------------------------
    // RNG sanity checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_rng_uniform_in_range() {
        let mut rng = Rng::new(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "Uniform out of range: {}", v);
        }
    }

    #[test]
    fn test_rng_normal_finite() {
        let mut rng = Rng::new(456);
        for _ in 0..1000 {
            let v = rng.next_normal();
            assert!(v.is_finite(), "Normal produced non-finite: {}", v);
        }
    }

    #[test]
    fn test_rng_normal_mean_near_zero() {
        let mut rng = Rng::new(789);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| rng.next_normal()).sum();
        let mean = sum / n as f64;
        assert!(
            mean.abs() < 0.1,
            "Normal mean should be near 0, got {}",
            mean
        );
    }

    #[test]
    fn test_rng_reproducible() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
