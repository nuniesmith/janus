//! What-if scenario analysis engine for portfolio stress testing
//!
//! Part of the Cortex region
//! Component: planning
//!
//! Evaluates portfolio performance under user-defined scenarios by applying
//! hypothetical shocks to drift, volatility, correlation, and drawdown
//! parameters. Produces per-scenario risk metrics and comparative rankings
//! so upstream planners can identify vulnerabilities and hedge requirements.
//!
//! Key features:
//! - Configurable scenario definitions (shock magnitude, type, duration)
//! - Multi-metric evaluation per scenario (return, VaR, drawdown, Sharpe)
//! - Baseline comparison with delta reporting
//! - Scenario ranking by configurable objective (worst-case, best-case, etc.)
//! - Severity classification (mild, moderate, severe, extreme)
//! - EMA-smoothed tracking of scenario outcomes across successive evaluations
//! - Sliding window of recent evaluations for trend analysis
//! - Running statistics with per-scenario and aggregate tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Type of shock to apply in a scenario
#[derive(Debug, Clone, PartialEq)]
pub enum ShockType {
    /// Additive shock to drift (e.g. -0.10 means drift reduced by 10%)
    DriftShock,
    /// Multiplicative shock to volatility (e.g. 2.0 means vol doubles)
    VolatilityMultiplier,
    /// Direct drawdown shock (e.g. 0.20 means 20% drawdown applied)
    DrawdownShock,
    /// Combined: drift shock + vol multiplier
    Combined,
}

/// Severity classification for a scenario
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Minor market perturbation
    Mild,
    /// Noticeable stress
    Moderate,
    /// Significant market dislocation
    Severe,
    /// Tail event / black swan
    Extreme,
}

impl Severity {
    /// Numeric weight for ranking (higher = worse)
    pub fn weight(&self) -> f64 {
        match self {
            Severity::Mild => 1.0,
            Severity::Moderate => 2.0,
            Severity::Severe => 3.0,
            Severity::Extreme => 4.0,
        }
    }
}

/// Definition of a single scenario
#[derive(Debug, Clone)]
pub struct ScenarioDefinition {
    /// Human-readable name
    pub name: String,
    /// Type of shock
    pub shock_type: ShockType,
    /// Drift shock value (additive, used for DriftShock and Combined)
    pub drift_shock: f64,
    /// Volatility multiplier (used for VolatilityMultiplier and Combined)
    pub vol_multiplier: f64,
    /// Direct drawdown to apply (used for DrawdownShock)
    pub drawdown_shock: f64,
    /// Scenario duration in days (affects how long the shock persists)
    pub duration_days: f64,
    /// Assigned severity
    pub severity: Severity,
    /// Probability weight for this scenario (for expected-loss calculations)
    pub probability: f64,
}

impl ScenarioDefinition {
    /// Create a simple drift-shock scenario
    pub fn drift(name: &str, shock: f64, severity: Severity) -> Self {
        Self {
            name: name.to_string(),
            shock_type: ShockType::DriftShock,
            drift_shock: shock,
            vol_multiplier: 1.0,
            drawdown_shock: 0.0,
            duration_days: 252.0,
            severity,
            probability: 1.0,
        }
    }

    /// Create a volatility-multiplier scenario
    pub fn vol_spike(name: &str, multiplier: f64, severity: Severity) -> Self {
        Self {
            name: name.to_string(),
            shock_type: ShockType::VolatilityMultiplier,
            drift_shock: 0.0,
            vol_multiplier: multiplier,
            drawdown_shock: 0.0,
            duration_days: 252.0,
            severity,
            probability: 1.0,
        }
    }

    /// Create a drawdown-shock scenario
    pub fn drawdown(name: &str, dd: f64, severity: Severity) -> Self {
        Self {
            name: name.to_string(),
            shock_type: ShockType::DrawdownShock,
            drift_shock: 0.0,
            vol_multiplier: 1.0,
            drawdown_shock: dd,
            duration_days: 1.0,
            severity,
            probability: 1.0,
        }
    }

    /// Create a combined drift + vol scenario
    pub fn combined(name: &str, drift_shock: f64, vol_multiplier: f64, severity: Severity) -> Self {
        Self {
            name: name.to_string(),
            shock_type: ShockType::Combined,
            drift_shock,
            vol_multiplier,
            drawdown_shock: 0.0,
            duration_days: 252.0,
            severity,
            probability: 1.0,
        }
    }

    /// Set the probability weight
    pub fn with_probability(mut self, p: f64) -> Self {
        self.probability = p;
        self
    }

    /// Set the duration
    pub fn with_duration(mut self, days: f64) -> Self {
        self.duration_days = days;
        self
    }
}

/// Configuration for the scenario analysis engine
#[derive(Debug, Clone)]
pub struct ScenarioAnalysisConfig {
    /// Baseline annualised drift
    pub baseline_drift: f64,
    /// Baseline annualised volatility
    pub baseline_volatility: f64,
    /// Initial portfolio value
    pub initial_value: f64,
    /// Risk-free rate for Sharpe calculations
    pub risk_free_rate: f64,
    /// EMA decay for smoothing results across evaluations (0 < decay < 1)
    pub ema_decay: f64,
    /// Maximum recent evaluations in sliding window
    pub window_size: usize,
}

impl Default for ScenarioAnalysisConfig {
    fn default() -> Self {
        Self {
            baseline_drift: 0.08,
            baseline_volatility: 0.20,
            initial_value: 100_000.0,
            risk_free_rate: 0.04,
            ema_decay: 0.3,
            window_size: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Metrics computed for a single scenario
#[derive(Debug, Clone)]
pub struct ScenarioMetrics {
    /// Effective drift under the scenario
    pub effective_drift: f64,
    /// Effective volatility under the scenario
    pub effective_volatility: f64,
    /// Expected return over the scenario duration
    pub expected_return: f64,
    /// Expected portfolio value at end of scenario
    pub expected_value: f64,
    /// Approximate VaR (95%) using parametric normal assumption
    pub var_95: f64,
    /// Maximum expected drawdown (simplified estimate)
    pub expected_max_drawdown: f64,
    /// Sharpe ratio under the scenario
    pub sharpe: f64,
    /// Dollar P&L impact
    pub pnl_impact: f64,
    /// Return relative to baseline (delta)
    pub return_delta: f64,
    /// Volatility change relative to baseline (ratio)
    pub vol_ratio: f64,
    /// Composite risk score [0, 10] combining severity and metric impact
    pub risk_score: f64,
}

/// Result of evaluating a single scenario
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    /// The scenario definition
    pub definition: ScenarioDefinition,
    /// Computed metrics
    pub metrics: ScenarioMetrics,
    /// Rank among all evaluated scenarios (1 = worst)
    pub rank: usize,
}

/// Ranking criterion for scenario ordering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RankingCriterion {
    /// Rank by worst expected return (most negative first)
    WorstReturn,
    /// Rank by highest VaR (largest risk first)
    HighestVaR,
    /// Rank by worst drawdown
    WorstDrawdown,
    /// Rank by composite risk score (highest first)
    HighestRiskScore,
    /// Rank by P&L impact (most negative first)
    WorstPnL,
}

/// Full evaluation result across all scenarios
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Per-scenario results, ranked
    pub scenarios: Vec<ScenarioResult>,
    /// Baseline metrics (no shock)
    pub baseline: ScenarioMetrics,
    /// Probability-weighted expected loss across all scenarios
    pub expected_loss: f64,
    /// Worst-case scenario name
    pub worst_case: String,
    /// Best-case scenario name
    pub best_case: String,
    /// Number of scenarios evaluated
    pub scenario_count: usize,
    /// Number of scenarios that produce a loss
    pub loss_scenarios: usize,
    /// Average risk score across all scenarios
    pub average_risk_score: f64,
}

/// Cumulative statistics across multiple evaluations
#[derive(Debug, Clone)]
pub struct ScenarioAnalysisStats {
    /// Total evaluations run
    pub total_evaluations: usize,
    /// Total individual scenarios evaluated
    pub total_scenarios_evaluated: usize,
    /// EMA-smoothed expected loss
    pub ema_expected_loss: f64,
    /// EMA-smoothed average risk score
    pub ema_avg_risk_score: f64,
    /// Worst expected loss observed in any evaluation
    pub worst_expected_loss: f64,
    /// Most frequently worst-case scenario name
    pub most_common_worst_case: String,
    /// Count of how many times each scenario was worst
    pub worst_case_counts: Vec<(String, usize)>,
}

impl Default for ScenarioAnalysisStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_scenarios_evaluated: 0,
            ema_expected_loss: 0.0,
            ema_avg_risk_score: 0.0,
            worst_expected_loss: 0.0,
            most_common_worst_case: String::new(),
            worst_case_counts: Vec::new(),
        }
    }
}

impl ScenarioAnalysisStats {
    /// Average scenarios per evaluation
    pub fn avg_scenarios_per_eval(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.total_scenarios_evaluated as f64 / self.total_evaluations as f64
    }
}

// ---------------------------------------------------------------------------
// Preset scenario sets
// ---------------------------------------------------------------------------

/// Standard stress-test scenario set
pub fn standard_stress_scenarios() -> Vec<ScenarioDefinition> {
    vec![
        ScenarioDefinition::drift("mild_slowdown", -0.05, Severity::Mild).with_probability(0.30),
        ScenarioDefinition::drift("recession", -0.15, Severity::Moderate).with_probability(0.15),
        ScenarioDefinition::drift("severe_downturn", -0.30, Severity::Severe)
            .with_probability(0.05),
        ScenarioDefinition::vol_spike("vol_spike_1.5x", 1.5, Severity::Mild).with_probability(0.25),
        ScenarioDefinition::vol_spike("vol_spike_2x", 2.0, Severity::Moderate)
            .with_probability(0.10),
        ScenarioDefinition::vol_spike("vol_explosion_3x", 3.0, Severity::Severe)
            .with_probability(0.03),
        ScenarioDefinition::drawdown("flash_crash_10pct", 0.10, Severity::Moderate)
            .with_probability(0.08),
        ScenarioDefinition::drawdown("crash_25pct", 0.25, Severity::Severe).with_probability(0.03),
        ScenarioDefinition::drawdown("black_swan_50pct", 0.50, Severity::Extreme)
            .with_probability(0.01),
        ScenarioDefinition::combined("stagflation", -0.10, 1.8, Severity::Severe)
            .with_probability(0.05),
    ]
}

// ---------------------------------------------------------------------------
// Core engine
// ---------------------------------------------------------------------------

/// Scenario analysis engine for portfolio stress testing
pub struct ScenarioAnalysis {
    config: ScenarioAnalysisConfig,
    ema_initialized: bool,
    recent: VecDeque<EvaluationResult>,
    stats: ScenarioAnalysisStats,
}

impl Default for ScenarioAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl ScenarioAnalysis {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            config: ScenarioAnalysisConfig::default(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: ScenarioAnalysisStats::default(),
        }
    }

    /// Create from validated config
    pub fn with_config(config: ScenarioAnalysisConfig) -> Result<Self> {
        if config.baseline_volatility < 0.0 {
            return Err(Error::Configuration(
                "baseline_volatility must be >= 0".into(),
            ));
        }
        if config.initial_value <= 0.0 {
            return Err(Error::Configuration("initial_value must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::Configuration("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::Configuration("window_size must be > 0".into()));
        }
        Ok(Self {
            config,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: ScenarioAnalysisStats::default(),
        })
    }

    /// Convenience: validate and create
    pub fn process(config: ScenarioAnalysisConfig) -> Result<Self> {
        Self::with_config(config)
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate a set of scenarios against the current configuration.
    ///
    /// Scenarios are ranked by the given criterion (worst first).
    pub fn evaluate(
        &mut self,
        scenarios: &[ScenarioDefinition],
        criterion: RankingCriterion,
    ) -> Result<EvaluationResult> {
        if scenarios.is_empty() {
            return Err(Error::Configuration(
                "at least one scenario is required".into(),
            ));
        }

        // Compute baseline (no shock)
        let baseline = self.compute_metrics(
            self.config.baseline_drift,
            self.config.baseline_volatility,
            0.0,
            252.0,
            Severity::Mild,
        );

        // Evaluate each scenario
        let mut results: Vec<ScenarioResult> = scenarios
            .iter()
            .map(|def| {
                let (eff_drift, eff_vol) = self.apply_shock(def);
                let metrics = self.compute_metrics(
                    eff_drift,
                    eff_vol,
                    def.drawdown_shock,
                    def.duration_days,
                    def.severity,
                );
                ScenarioResult {
                    definition: def.clone(),
                    metrics,
                    rank: 0,
                }
            })
            .collect();

        // Sort by criterion (worst first)
        results.sort_by(|a, b| {
            let key_a = Self::ranking_key(&a.metrics, criterion);
            let key_b = Self::ranking_key(&b.metrics, criterion);
            key_a
                .partial_cmp(&key_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;

            // Compute deltas vs baseline
            r.metrics.return_delta = r.metrics.expected_return - baseline.expected_return;
            r.metrics.vol_ratio = if baseline.effective_volatility > 1e-15 {
                r.metrics.effective_volatility / baseline.effective_volatility
            } else {
                1.0
            };
        }

        // Expected loss (probability-weighted)
        let total_prob: f64 = results.iter().map(|r| r.definition.probability).sum();
        let expected_loss = if total_prob > 0.0 {
            results
                .iter()
                .map(|r| {
                    let loss = (-r.metrics.pnl_impact).max(0.0);
                    loss * r.definition.probability / total_prob
                })
                .sum()
        } else {
            0.0
        };

        // Worst / best case
        let worst_case = results
            .first()
            .map(|r| r.definition.name.clone())
            .unwrap_or_default();
        let best_case = results
            .last()
            .map(|r| r.definition.name.clone())
            .unwrap_or_default();

        let loss_scenarios = results
            .iter()
            .filter(|r| r.metrics.expected_return < 0.0)
            .count();

        let avg_risk_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.metrics.risk_score).sum::<f64>() / results.len() as f64
        };

        let eval = EvaluationResult {
            scenario_count: results.len(),
            scenarios: results,
            baseline,
            expected_loss,
            worst_case: worst_case.clone(),
            best_case,
            loss_scenarios,
            average_risk_score: avg_risk_score,
        };

        self.update_stats(&eval, &worst_case);
        Ok(eval)
    }

    /// Evaluate using the standard stress scenarios with WorstReturn ranking
    pub fn evaluate_standard(&mut self) -> Result<EvaluationResult> {
        let scenarios = standard_stress_scenarios();
        self.evaluate(&scenarios, RankingCriterion::WorstReturn)
    }

    /// Evaluate a single scenario and return its metrics
    pub fn evaluate_single(&self, scenario: &ScenarioDefinition) -> ScenarioMetrics {
        let (eff_drift, eff_vol) = self.apply_shock(scenario);
        let baseline = self.compute_metrics(
            self.config.baseline_drift,
            self.config.baseline_volatility,
            0.0,
            252.0,
            Severity::Mild,
        );
        let mut metrics = self.compute_metrics(
            eff_drift,
            eff_vol,
            scenario.drawdown_shock,
            scenario.duration_days,
            scenario.severity,
        );
        metrics.return_delta = metrics.expected_return - baseline.expected_return;
        metrics.vol_ratio = if baseline.effective_volatility > 1e-15 {
            metrics.effective_volatility / baseline.effective_volatility
        } else {
            1.0
        };
        metrics
    }

    // -----------------------------------------------------------------------
    // Internal: shock application
    // -----------------------------------------------------------------------

    fn apply_shock(&self, def: &ScenarioDefinition) -> (f64, f64) {
        match def.shock_type {
            ShockType::DriftShock => (
                self.config.baseline_drift + def.drift_shock,
                self.config.baseline_volatility,
            ),
            ShockType::VolatilityMultiplier => (
                self.config.baseline_drift,
                self.config.baseline_volatility * def.vol_multiplier,
            ),
            ShockType::DrawdownShock => {
                (self.config.baseline_drift, self.config.baseline_volatility)
            }
            ShockType::Combined => (
                self.config.baseline_drift + def.drift_shock,
                self.config.baseline_volatility * def.vol_multiplier,
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Internal: metric computation
    // -----------------------------------------------------------------------

    fn compute_metrics(
        &self,
        drift: f64,
        volatility: f64,
        drawdown_shock: f64,
        duration_days: f64,
        severity: Severity,
    ) -> ScenarioMetrics {
        let horizon_years = duration_days / 252.0;

        // GBM expected return: E[S/S0 - 1] = exp(μT) - 1
        let base_return = (drift * horizon_years).exp_m1();

        // Apply drawdown shock on top
        let expected_return = if drawdown_shock > 0.0 {
            (1.0 + base_return) * (1.0 - drawdown_shock) - 1.0
        } else {
            base_return
        };

        let expected_value = self.config.initial_value * (1.0 + expected_return);
        let pnl_impact = expected_value - self.config.initial_value;

        // Parametric VaR (95%): loss = -(μT - 1.645 σ√T)
        let period_std = volatility * horizon_years.sqrt();
        let var_95 = -(drift * horizon_years - 1.645 * period_std);

        // Simplified max drawdown estimate: E[MDD] ≈ σ √(2T ln(T)) for GBM
        // Use a practical approximation
        let expected_max_drawdown = if horizon_years > 0.0 && volatility > 0.0 {
            let t = horizon_years.max(0.01);
            let mdd_base = volatility * (2.0 * t * (t.max(1.0)).ln().max(0.1)).sqrt();
            // Add the drawdown shock if present
            (mdd_base + drawdown_shock).min(1.0)
        } else {
            drawdown_shock.min(1.0)
        };

        // Sharpe
        let annualised_std = if horizon_years > 0.0 {
            period_std / horizon_years.sqrt()
        } else {
            volatility
        };
        let annualised_return = if horizon_years > 0.0 {
            expected_return / horizon_years
        } else {
            0.0
        };
        let sharpe = if annualised_std > 1e-15 {
            (annualised_return - self.config.risk_free_rate) / annualised_std
        } else {
            0.0
        };

        // Risk score: combines severity weight, return impact, and vol increase
        let return_penalty = (-expected_return * 10.0).clamp(0.0, 5.0);
        let vol_penalty = ((volatility / self.config.baseline_volatility.max(1e-15)) - 1.0)
            .max(0.0)
            .min(3.0);
        let dd_penalty = (drawdown_shock * 5.0).min(3.0);
        let risk_score =
            (severity.weight() + return_penalty + vol_penalty + dd_penalty).clamp(0.0, 10.0);

        ScenarioMetrics {
            effective_drift: drift,
            effective_volatility: volatility,
            expected_return,
            expected_value,
            var_95,
            expected_max_drawdown,
            sharpe,
            pnl_impact,
            return_delta: 0.0, // filled in by evaluate()
            vol_ratio: 1.0,    // filled in by evaluate()
            risk_score,
        }
    }

    fn ranking_key(metrics: &ScenarioMetrics, criterion: RankingCriterion) -> f64 {
        match criterion {
            RankingCriterion::WorstReturn => metrics.expected_return, // ascending = worst first
            RankingCriterion::HighestVaR => -metrics.var_95, // descending = highest VaR first
            RankingCriterion::WorstDrawdown => -metrics.expected_max_drawdown,
            RankingCriterion::HighestRiskScore => -metrics.risk_score,
            RankingCriterion::WorstPnL => metrics.pnl_impact,
        }
    }

    // -----------------------------------------------------------------------
    // Stats update
    // -----------------------------------------------------------------------

    fn update_stats(&mut self, eval: &EvaluationResult, worst_case: &str) {
        let decay = self.config.ema_decay;

        if !self.ema_initialized {
            self.stats.ema_expected_loss = eval.expected_loss;
            self.stats.ema_avg_risk_score = eval.average_risk_score;
            self.ema_initialized = true;
        } else {
            self.stats.ema_expected_loss =
                decay * eval.expected_loss + (1.0 - decay) * self.stats.ema_expected_loss;
            self.stats.ema_avg_risk_score =
                decay * eval.average_risk_score + (1.0 - decay) * self.stats.ema_avg_risk_score;
        }

        self.stats.total_evaluations += 1;
        self.stats.total_scenarios_evaluated += eval.scenario_count;

        if eval.expected_loss > self.stats.worst_expected_loss {
            self.stats.worst_expected_loss = eval.expected_loss;
        }

        // Track worst-case counts
        if let Some(entry) = self
            .stats
            .worst_case_counts
            .iter_mut()
            .find(|(name, _)| name == worst_case)
        {
            entry.1 += 1;
        } else {
            self.stats
                .worst_case_counts
                .push((worst_case.to_string(), 1));
        }

        // Update most common worst case
        if let Some((name, _)) = self
            .stats
            .worst_case_counts
            .iter()
            .max_by_key(|(_, count)| *count)
        {
            self.stats.most_common_worst_case = name.clone();
        }

        // Window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(eval.clone());
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Cumulative statistics
    pub fn stats(&self) -> &ScenarioAnalysisStats {
        &self.stats
    }

    /// Number of evaluations completed
    pub fn evaluation_count(&self) -> usize {
        self.stats.total_evaluations
    }

    /// Current configuration
    pub fn config(&self) -> &ScenarioAnalysisConfig {
        &self.config
    }

    /// Recent evaluation results
    pub fn recent_evaluations(&self) -> &VecDeque<EvaluationResult> {
        &self.recent
    }

    /// EMA-smoothed expected loss
    pub fn smoothed_expected_loss(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_expected_loss)
        } else {
            None
        }
    }

    /// EMA-smoothed average risk score
    pub fn smoothed_risk_score(&self) -> Option<f64> {
        if self.ema_initialized {
            Some(self.stats.ema_avg_risk_score)
        } else {
            None
        }
    }

    /// Windowed mean of expected losses
    pub fn windowed_expected_loss(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|e| e.expected_loss).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Windowed mean of average risk scores
    pub fn windowed_risk_score(&self) -> Option<f64> {
        if self.recent.is_empty() {
            return None;
        }
        let sum: f64 = self.recent.iter().map(|e| e.average_risk_score).sum();
        Some(sum / self.recent.len() as f64)
    }

    /// Whether risk is trending worse over the recent window
    pub fn is_risk_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|e| e.expected_loss)
            .sum::<f64>()
            / half as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|e| e.expected_loss)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second_half > first_half * 1.05
    }

    /// Reset all state (keeps config)
    pub fn reset(&mut self) {
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = ScenarioAnalysisStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ScenarioAnalysisConfig {
        ScenarioAnalysisConfig {
            baseline_drift: 0.08,
            baseline_volatility: 0.20,
            initial_value: 100_000.0,
            risk_free_rate: 0.04,
            ema_decay: 0.3,
            window_size: 16,
        }
    }

    fn simple_scenarios() -> Vec<ScenarioDefinition> {
        vec![
            ScenarioDefinition::drift("mild_down", -0.05, Severity::Mild).with_probability(0.3),
            ScenarioDefinition::drift("recession", -0.20, Severity::Moderate).with_probability(0.1),
            ScenarioDefinition::vol_spike("vol_2x", 2.0, Severity::Moderate).with_probability(0.1),
            ScenarioDefinition::drawdown("crash", 0.25, Severity::Severe).with_probability(0.05),
        ]
    }

    #[test]
    fn test_basic() {
        let mut sa = ScenarioAnalysis::new();
        let result = sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_returns_instance() {
        let sa = ScenarioAnalysis::process(small_config());
        assert!(sa.is_ok());
    }

    #[test]
    fn test_evaluate_returns_all_scenarios() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let scenarios = simple_scenarios();
        let result = sa
            .evaluate(&scenarios, RankingCriterion::WorstReturn)
            .unwrap();
        assert_eq!(result.scenario_count, 4);
        assert_eq!(result.scenarios.len(), 4);
    }

    #[test]
    fn test_empty_scenarios_error() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa.evaluate(&[], RankingCriterion::WorstReturn);
        assert!(result.is_err());
    }

    #[test]
    fn test_ranking_worst_return_first() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        // First scenario should have the worst (lowest) expected return
        for i in 0..result.scenarios.len() - 1 {
            assert!(
                result.scenarios[i].metrics.expected_return
                    <= result.scenarios[i + 1].metrics.expected_return + 1e-10,
                "Scenario {} (return {}) should be <= scenario {} (return {})",
                i,
                result.scenarios[i].metrics.expected_return,
                i + 1,
                result.scenarios[i + 1].metrics.expected_return
            );
        }
    }

    #[test]
    fn test_ranking_highest_var() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::HighestVaR)
            .unwrap();
        for i in 0..result.scenarios.len() - 1 {
            assert!(
                result.scenarios[i].metrics.var_95
                    >= result.scenarios[i + 1].metrics.var_95 - 1e-10,
                "Scenario {} (VaR {}) should be >= scenario {} (VaR {})",
                i,
                result.scenarios[i].metrics.var_95,
                i + 1,
                result.scenarios[i + 1].metrics.var_95
            );
        }
    }

    #[test]
    fn test_ranking_highest_risk_score() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::HighestRiskScore)
            .unwrap();
        for i in 0..result.scenarios.len() - 1 {
            assert!(
                result.scenarios[i].metrics.risk_score
                    >= result.scenarios[i + 1].metrics.risk_score - 1e-10,
                "Scenario {} (score {}) should be >= scenario {} (score {})",
                i,
                result.scenarios[i].metrics.risk_score,
                i + 1,
                result.scenarios[i + 1].metrics.risk_score
            );
        }
    }

    #[test]
    fn test_ranks_assigned() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        for (i, r) in result.scenarios.iter().enumerate() {
            assert_eq!(r.rank, i + 1);
        }
    }

    #[test]
    fn test_baseline_no_shock() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        // Baseline should have positive expected return with positive drift
        assert!(
            result.baseline.expected_return > 0.0,
            "Baseline return should be positive, got {}",
            result.baseline.expected_return
        );
    }

    #[test]
    fn test_drift_shock_reduces_return() {
        let sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let baseline_scenario = ScenarioDefinition::drift("no_shock", 0.0, Severity::Mild);
        let shock_scenario = ScenarioDefinition::drift("shock", -0.20, Severity::Moderate);
        let baseline = sa.evaluate_single(&baseline_scenario);
        let shocked = sa.evaluate_single(&shock_scenario);
        assert!(
            shocked.expected_return < baseline.expected_return,
            "Shocked return ({}) should be less than baseline ({})",
            shocked.expected_return,
            baseline.expected_return
        );
    }

    #[test]
    fn test_vol_multiplier_increases_var() {
        let sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let low_vol = ScenarioDefinition::vol_spike("1x", 1.0, Severity::Mild);
        let high_vol = ScenarioDefinition::vol_spike("3x", 3.0, Severity::Severe);
        let low_metrics = sa.evaluate_single(&low_vol);
        let high_metrics = sa.evaluate_single(&high_vol);
        assert!(
            high_metrics.var_95 > low_metrics.var_95,
            "Higher vol VaR ({}) should exceed lower vol VaR ({})",
            high_metrics.var_95,
            low_metrics.var_95
        );
    }

    #[test]
    fn test_drawdown_shock_applied() {
        let sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let dd = ScenarioDefinition::drawdown("crash", 0.30, Severity::Severe);
        let metrics = sa.evaluate_single(&dd);
        // With a 30% drawdown shock, expected return should be significantly negative
        assert!(
            metrics.expected_return < 0.0,
            "30% drawdown should produce negative return, got {}",
            metrics.expected_return
        );
    }

    #[test]
    fn test_combined_shock() {
        let sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let combined = ScenarioDefinition::combined("stagflation", -0.15, 2.0, Severity::Severe);
        let metrics = sa.evaluate_single(&combined);
        assert!(
            metrics.effective_drift < sa.config().baseline_drift,
            "Combined shock should reduce drift"
        );
        assert!(
            metrics.effective_volatility > sa.config().baseline_volatility,
            "Combined shock should increase vol"
        );
    }

    #[test]
    fn test_return_delta_computed() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let scenarios = vec![ScenarioDefinition::drift("down", -0.10, Severity::Moderate)];
        let result = sa
            .evaluate(&scenarios, RankingCriterion::WorstReturn)
            .unwrap();
        let delta = result.scenarios[0].metrics.return_delta;
        assert!(
            delta < 0.0,
            "Return delta should be negative for drift shock, got {}",
            delta
        );
    }

    #[test]
    fn test_vol_ratio_computed() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let scenarios = vec![ScenarioDefinition::vol_spike("2x", 2.0, Severity::Moderate)];
        let result = sa
            .evaluate(&scenarios, RankingCriterion::WorstReturn)
            .unwrap();
        let ratio = result.scenarios[0].metrics.vol_ratio;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Vol ratio should be ~2.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_pnl_impact() {
        let sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let scenario = ScenarioDefinition::drawdown("crash", 0.20, Severity::Severe);
        let metrics = sa.evaluate_single(&scenario);
        // PnL impact should be negative for a crash
        assert!(
            metrics.pnl_impact < 0.0,
            "PnL impact should be negative, got {}",
            metrics.pnl_impact
        );
    }

    #[test]
    fn test_risk_score_bounded() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        for r in &result.scenarios {
            assert!(
                r.metrics.risk_score >= 0.0 && r.metrics.risk_score <= 10.0,
                "Risk score should be [0, 10], got {}",
                r.metrics.risk_score
            );
        }
    }

    #[test]
    fn test_risk_score_increases_with_severity() {
        let sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let mild = ScenarioDefinition::drift("mild", -0.02, Severity::Mild);
        let extreme = ScenarioDefinition::drawdown("extreme", 0.50, Severity::Extreme);
        let mild_score = sa.evaluate_single(&mild).risk_score;
        let extreme_score = sa.evaluate_single(&extreme).risk_score;
        assert!(
            extreme_score > mild_score,
            "Extreme score ({}) should exceed mild ({})",
            extreme_score,
            mild_score
        );
    }

    #[test]
    fn test_expected_loss_non_negative() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(
            result.expected_loss >= 0.0,
            "Expected loss should be >= 0, got {}",
            result.expected_loss
        );
    }

    #[test]
    fn test_loss_scenarios_count() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(result.loss_scenarios <= result.scenario_count);
    }

    #[test]
    fn test_worst_and_best_case_names() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(!result.worst_case.is_empty());
        assert!(!result.best_case.is_empty());
        assert_eq!(result.worst_case, result.scenarios[0].definition.name);
        assert_eq!(
            result.best_case,
            result.scenarios.last().unwrap().definition.name
        );
    }

    #[test]
    fn test_sharpe_finite() {
        let sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        for scenario in &simple_scenarios() {
            let metrics = sa.evaluate_single(scenario);
            assert!(
                metrics.sharpe.is_finite(),
                "Sharpe should be finite for {}",
                scenario.name
            );
        }
    }

    #[test]
    fn test_max_drawdown_bounded() {
        let sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        for scenario in &simple_scenarios() {
            let metrics = sa.evaluate_single(scenario);
            assert!(
                metrics.expected_max_drawdown >= 0.0 && metrics.expected_max_drawdown <= 1.0,
                "MDD should be in [0, 1], got {} for {}",
                metrics.expected_max_drawdown,
                scenario.name
            );
        }
    }

    #[test]
    fn test_standard_stress_scenarios_preset() {
        let scenarios = standard_stress_scenarios();
        assert_eq!(scenarios.len(), 10);
        // Should contain scenarios of all severity levels
        assert!(scenarios.iter().any(|s| s.severity == Severity::Mild));
        assert!(scenarios.iter().any(|s| s.severity == Severity::Moderate));
        assert!(scenarios.iter().any(|s| s.severity == Severity::Severe));
        assert!(scenarios.iter().any(|s| s.severity == Severity::Extreme));
    }

    #[test]
    fn test_evaluate_standard() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa.evaluate_standard();
        assert!(result.is_ok());
        let eval = result.unwrap();
        assert_eq!(eval.scenario_count, 10);
    }

    #[test]
    fn test_ema_initializes_on_first_eval() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        assert!(sa.smoothed_expected_loss().is_none());
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(sa.smoothed_expected_loss().is_some());
    }

    #[test]
    fn test_ema_blends_on_subsequent_evals() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let e1 = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        let loss1 = e1.expected_loss;
        let e2 = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        let loss2 = e2.expected_loss;
        let ema = sa.smoothed_expected_loss().unwrap();
        let expected = 0.3 * loss2 + 0.7 * loss1;
        assert!(
            (ema - expected).abs() < 1e-6,
            "EMA mismatch: got {}, expected {}",
            ema,
            expected
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert_eq!(sa.stats().total_evaluations, 2);
        assert_eq!(sa.stats().total_scenarios_evaluated, 8);
    }

    #[test]
    fn test_stats_worst_case_tracking() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(!sa.stats().most_common_worst_case.is_empty());
        assert!(!sa.stats().worst_case_counts.is_empty());
    }

    #[test]
    fn test_stats_defaults() {
        let stats = ScenarioAnalysisStats::default();
        assert_eq!(stats.total_evaluations, 0);
        assert_eq!(stats.avg_scenarios_per_eval(), 0.0);
    }

    #[test]
    fn test_avg_scenarios_per_eval() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(
            (sa.stats().avg_scenarios_per_eval() - 4.0).abs() < 1e-10,
            "Should be 4 scenarios per eval"
        );
    }

    #[test]
    fn test_windowed_expected_loss() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        assert!(sa.windowed_expected_loss().is_none());
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(sa.windowed_expected_loss().is_some());
    }

    #[test]
    fn test_windowed_risk_score() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        assert!(sa.windowed_risk_score().is_none());
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        let score = sa.windowed_risk_score().unwrap();
        assert!(score >= 0.0);
    }

    #[test]
    fn test_window_eviction() {
        let config = ScenarioAnalysisConfig {
            window_size: 3,
            ..small_config()
        };
        let mut sa = ScenarioAnalysis::with_config(config).unwrap();
        for _ in 0..5 {
            sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
                .unwrap();
        }
        assert_eq!(sa.recent_evaluations().len(), 3);
    }

    #[test]
    fn test_is_risk_increasing_insufficient_data() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        assert!(!sa.is_risk_increasing());
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(!sa.is_risk_increasing());
    }

    #[test]
    fn test_reset() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        sa.evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert_eq!(sa.evaluation_count(), 2);
        sa.reset();
        assert_eq!(sa.evaluation_count(), 0);
        assert!(sa.smoothed_expected_loss().is_none());
        assert!(sa.recent_evaluations().is_empty());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Mild < Severity::Moderate);
        assert!(Severity::Moderate < Severity::Severe);
        assert!(Severity::Severe < Severity::Extreme);
    }

    #[test]
    fn test_severity_weights() {
        assert!(Severity::Extreme.weight() > Severity::Severe.weight());
        assert!(Severity::Severe.weight() > Severity::Moderate.weight());
        assert!(Severity::Moderate.weight() > Severity::Mild.weight());
    }

    #[test]
    fn test_scenario_builder_with_probability() {
        let s = ScenarioDefinition::drift("test", -0.05, Severity::Mild).with_probability(0.25);
        assert!((s.probability - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_scenario_builder_with_duration() {
        let s = ScenarioDefinition::drift("test", -0.05, Severity::Mild).with_duration(60.0);
        assert!((s.duration_days - 60.0).abs() < 1e-12);
    }

    #[test]
    fn test_zero_vol_baseline() {
        let config = ScenarioAnalysisConfig {
            baseline_volatility: 0.0,
            ..small_config()
        };
        let mut sa = ScenarioAnalysis::with_config(config).unwrap();
        let scenarios = vec![ScenarioDefinition::drift(
            "shock",
            -0.10,
            Severity::Moderate,
        )];
        let result = sa.evaluate(&scenarios, RankingCriterion::WorstReturn);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ranking_criterion_worst_pnl() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstPnL)
            .unwrap();
        for i in 0..result.scenarios.len() - 1 {
            assert!(
                result.scenarios[i].metrics.pnl_impact
                    <= result.scenarios[i + 1].metrics.pnl_impact + 1e-6,
                "PnL ordering violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_ranking_criterion_worst_drawdown() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstDrawdown)
            .unwrap();
        for i in 0..result.scenarios.len() - 1 {
            assert!(
                result.scenarios[i].metrics.expected_max_drawdown
                    >= result.scenarios[i + 1].metrics.expected_max_drawdown - 1e-10,
                "Drawdown ordering violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_evaluate_single_consistent() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let scenario = ScenarioDefinition::drift("test", -0.10, Severity::Moderate);

        // evaluate_single should produce same core metrics as evaluate
        let single = sa.evaluate_single(&scenario);
        let result = sa
            .evaluate(
                std::slice::from_ref(&scenario),
                RankingCriterion::WorstReturn,
            )
            .unwrap();
        let batch = &result.scenarios[0].metrics;

        assert!(
            (single.expected_return - batch.expected_return).abs() < 1e-10,
            "Expected returns should match"
        );
        assert!(
            (single.var_95 - batch.var_95).abs() < 1e-10,
            "VaR should match"
        );
    }

    #[test]
    fn test_average_risk_score() {
        let mut sa = ScenarioAnalysis::with_config(small_config()).unwrap();
        let result = sa
            .evaluate(&simple_scenarios(), RankingCriterion::WorstReturn)
            .unwrap();
        assert!(result.average_risk_score >= 0.0);
        assert!(result.average_risk_score <= 10.0);
    }

    // -----------------------------------------------------------------------
    // Config validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_negative_vol() {
        let config = ScenarioAnalysisConfig {
            baseline_volatility: -0.1,
            ..small_config()
        };
        assert!(ScenarioAnalysis::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_initial_value() {
        let config = ScenarioAnalysisConfig {
            initial_value: 0.0,
            ..small_config()
        };
        assert!(ScenarioAnalysis::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_zero() {
        let config = ScenarioAnalysisConfig {
            ema_decay: 0.0,
            ..small_config()
        };
        assert!(ScenarioAnalysis::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_one() {
        let config = ScenarioAnalysisConfig {
            ema_decay: 1.0,
            ..small_config()
        };
        assert!(ScenarioAnalysis::with_config(config).is_err());
    }

    #[test]
    fn test_invalid_zero_window() {
        let config = ScenarioAnalysisConfig {
            window_size: 0,
            ..small_config()
        };
        assert!(ScenarioAnalysis::with_config(config).is_err());
    }
}
