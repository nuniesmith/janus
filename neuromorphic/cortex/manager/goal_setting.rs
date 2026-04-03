//! Portfolio objectives and constraints management
//!
//! Part of the Cortex region
//! Component: manager
//!
//! Defines and tracks portfolio-level goals including return targets, risk
//! budgets, drawdown limits, and allocation constraints. Provides real-time
//! evaluation of goal attainment and constraint satisfaction so upstream
//! planners can adjust strategies accordingly.
//!
//! Key features:
//! - Configurable goal definitions (return target, risk budget, drawdown limit)
//! - Multi-objective tracking with weighted priority
//! - Real-time constraint satisfaction checking
//! - Goal attainment scoring (0–1 normalised)
//! - Feasibility analysis across competing objectives
//! - EMA-smoothed tracking of goal progress across evaluation cycles
//! - Sliding window of recent evaluations for trend analysis
//! - Running statistics with historical attainment tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Type of portfolio goal
#[derive(Debug, Clone, PartialEq)]
pub enum GoalType {
    /// Achieve a minimum annualised return
    ReturnTarget,
    /// Keep portfolio volatility below a ceiling
    RiskBudget,
    /// Keep maximum drawdown below a ceiling
    DrawdownLimit,
    /// Keep Sharpe ratio above a floor
    SharpeFloor,
    /// Keep portfolio concentration (max single-asset weight) below a ceiling
    ConcentrationLimit,
    /// Maintain minimum cash / liquidity buffer
    LiquidityBuffer,
    /// Custom goal evaluated externally
    Custom,
}

/// Priority level for goal ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GoalPriority {
    /// Nice-to-have, optimise if possible
    Low,
    /// Standard importance
    Medium,
    /// Must-achieve, hard constraint
    High,
    /// Non-negotiable, system-critical
    Critical,
}

impl GoalPriority {
    /// Numeric weight for aggregation (higher = more important)
    pub fn weight(&self) -> f64 {
        match self {
            GoalPriority::Low => 1.0,
            GoalPriority::Medium => 2.0,
            GoalPriority::High => 4.0,
            GoalPriority::Critical => 8.0,
        }
    }
}

/// Direction of the constraint: value must be above or below the target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintDirection {
    /// Observed value should be >= target (e.g. return, Sharpe)
    AtLeast,
    /// Observed value should be <= target (e.g. volatility, drawdown)
    AtMost,
}

/// A single portfolio goal definition
#[derive(Debug, Clone)]
pub struct Goal {
    /// Unique goal identifier
    pub id: String,
    /// Human-readable label
    pub label: String,
    /// Goal type
    pub goal_type: GoalType,
    /// Target value
    pub target: f64,
    /// Constraint direction
    pub direction: ConstraintDirection,
    /// Priority
    pub priority: GoalPriority,
    /// Tolerance band: fraction of target within which we consider "nearly met"
    pub tolerance: f64,
    /// Whether this goal is currently active
    pub active: bool,
}

impl Goal {
    /// Evaluate attainment score for a given observed value.
    ///
    /// Returns a score in [0, 1] where 1.0 = fully met, 0.0 = completely unmet.
    /// Values within the tolerance band are linearly interpolated.
    pub fn attainment(&self, observed: f64) -> f64 {
        if self.target.abs() < 1e-15 {
            // Avoid division by zero; treat as met if observed is on the right side
            return match self.direction {
                ConstraintDirection::AtLeast => {
                    if observed >= 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                ConstraintDirection::AtMost => {
                    if observed <= 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
        }

        match self.direction {
            ConstraintDirection::AtLeast => {
                if observed >= self.target {
                    1.0
                } else {
                    let deficit = (self.target - observed) / self.target.abs();
                    if deficit <= self.tolerance {
                        1.0 - deficit / self.tolerance.max(1e-15)
                    } else {
                        0.0
                    }
                }
            }
            ConstraintDirection::AtMost => {
                if observed <= self.target {
                    1.0
                } else {
                    let excess = (observed - self.target) / self.target.abs();
                    if excess <= self.tolerance {
                        1.0 - excess / self.tolerance.max(1e-15)
                    } else {
                        0.0
                    }
                }
            }
        }
    }

    /// Whether the goal is satisfied (attainment == 1.0)
    pub fn is_satisfied(&self, observed: f64) -> bool {
        match self.direction {
            ConstraintDirection::AtLeast => observed >= self.target,
            ConstraintDirection::AtMost => observed <= self.target,
        }
    }

    /// Whether the goal is within tolerance band
    pub fn is_within_tolerance(&self, observed: f64) -> bool {
        self.attainment(observed) > 0.0
    }

    /// Breach magnitude: how far outside the target we are (0 if satisfied)
    pub fn breach(&self, observed: f64) -> f64 {
        if self.is_satisfied(observed) {
            return 0.0;
        }
        match self.direction {
            ConstraintDirection::AtLeast => self.target - observed,
            ConstraintDirection::AtMost => observed - self.target,
        }
    }
}

/// Current portfolio state snapshot for goal evaluation
#[derive(Debug, Clone)]
pub struct PortfolioState {
    /// Annualised return (realised or projected)
    pub annualised_return: f64,
    /// Annualised volatility
    pub volatility: f64,
    /// Maximum drawdown from peak (0–1)
    pub max_drawdown: f64,
    /// Sharpe ratio
    pub sharpe: f64,
    /// Maximum single-asset weight (0–1)
    pub max_concentration: f64,
    /// Cash / liquidity fraction (0–1)
    pub cash_fraction: f64,
    /// Total portfolio value
    pub portfolio_value: f64,
}

impl Default for PortfolioState {
    fn default() -> Self {
        Self {
            annualised_return: 0.0,
            volatility: 0.15,
            max_drawdown: 0.0,
            sharpe: 0.0,
            max_concentration: 0.20,
            cash_fraction: 0.05,
            portfolio_value: 1_000_000.0,
        }
    }
}

impl PortfolioState {
    /// Get the observed value for a given goal type
    pub fn value_for(&self, goal_type: &GoalType) -> f64 {
        match goal_type {
            GoalType::ReturnTarget => self.annualised_return,
            GoalType::RiskBudget => self.volatility,
            GoalType::DrawdownLimit => self.max_drawdown,
            GoalType::SharpeFloor => self.sharpe,
            GoalType::ConcentrationLimit => self.max_concentration,
            GoalType::LiquidityBuffer => self.cash_fraction,
            GoalType::Custom => 0.0,
        }
    }
}

/// Result of evaluating a single goal
#[derive(Debug, Clone)]
pub struct GoalEvaluation {
    /// Goal ID
    pub goal_id: String,
    /// Observed value
    pub observed: f64,
    /// Target value
    pub target: f64,
    /// Attainment score (0–1)
    pub attainment: f64,
    /// Whether the goal is fully satisfied
    pub satisfied: bool,
    /// Breach magnitude (0 if satisfied)
    pub breach: f64,
    /// Priority of the goal
    pub priority: GoalPriority,
}

/// Summary of a full evaluation cycle across all goals
#[derive(Debug, Clone)]
pub struct EvaluationSummary {
    /// Number of goals evaluated
    pub goals_evaluated: usize,
    /// Number of goals satisfied
    pub goals_satisfied: usize,
    /// Number of goals breached (beyond tolerance)
    pub goals_breached: usize,
    /// Weighted average attainment across all active goals
    pub weighted_attainment: f64,
    /// Minimum attainment across all goals (worst performer)
    pub min_attainment: f64,
    /// ID of the worst-performing goal
    pub worst_goal_id: String,
    /// Whether all critical goals are satisfied
    pub critical_goals_met: bool,
    /// Per-goal evaluations
    pub evaluations: Vec<GoalEvaluation>,
    /// Feasibility score: fraction of goals that are simultaneously achievable
    pub feasibility: f64,
}

/// Configuration for the GoalSetting engine
#[derive(Debug, Clone)]
pub struct GoalSettingConfig {
    /// Maximum number of goals
    pub max_goals: usize,
    /// EMA decay factor for smoothing attainment (0 < decay < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent evaluations
    pub window_size: usize,
    /// Default tolerance band for goals without explicit tolerance
    pub default_tolerance: f64,
}

impl Default for GoalSettingConfig {
    fn default() -> Self {
        Self {
            max_goals: 50,
            ema_decay: 0.1,
            window_size: 100,
            default_tolerance: 0.10,
        }
    }
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the goal-setting engine
#[derive(Debug, Clone)]
pub struct GoalSettingStats {
    /// Total evaluation cycles
    pub total_evaluations: u64,
    /// EMA of weighted attainment
    pub ema_attainment: f64,
    /// EMA of fraction of goals satisfied
    pub ema_satisfaction_rate: f64,
    /// EMA of worst-goal attainment
    pub ema_min_attainment: f64,
    /// Count of evaluations where all critical goals were met
    pub critical_met_count: u64,
    /// Count of evaluations where at least one critical goal was breached
    pub critical_breach_count: u64,
    /// Best weighted attainment observed
    pub best_attainment: f64,
    /// Worst weighted attainment observed
    pub worst_attainment: f64,
}

impl Default for GoalSettingStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            ema_attainment: 0.0,
            ema_satisfaction_rate: 0.0,
            ema_min_attainment: 0.0,
            critical_met_count: 0,
            critical_breach_count: 0,
            best_attainment: 0.0,
            worst_attainment: 1.0,
        }
    }
}

impl GoalSettingStats {
    /// Critical goal satisfaction rate over all evaluations
    pub fn critical_satisfaction_rate(&self) -> f64 {
        let total = self.critical_met_count + self.critical_breach_count;
        if total == 0 {
            return 1.0;
        }
        self.critical_met_count as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// GoalSetting Engine
// ---------------------------------------------------------------------------

/// Portfolio goal-setting and constraint-tracking engine.
///
/// Maintains a registry of goals, evaluates portfolio state against them,
/// and tracks attainment statistics over time.
pub struct GoalSetting {
    config: GoalSettingConfig,
    goals: Vec<Goal>,
    ema_initialized: bool,
    recent: VecDeque<EvaluationSummary>,
    stats: GoalSettingStats,
}

impl Default for GoalSetting {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalSetting {
    /// Create a new goal-setting engine with default configuration
    pub fn new() -> Self {
        Self {
            config: GoalSettingConfig::default(),
            goals: Vec::new(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: GoalSettingStats::default(),
        }
    }

    /// Create with explicit configuration
    pub fn with_config(config: GoalSettingConfig) -> Result<Self> {
        if config.max_goals == 0 {
            return Err(Error::InvalidInput("max_goals must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.default_tolerance < 0.0 || config.default_tolerance > 1.0 {
            return Err(Error::InvalidInput(
                "default_tolerance must be in [0, 1]".into(),
            ));
        }
        Ok(Self {
            config,
            goals: Vec::new(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: GoalSettingStats::default(),
        })
    }

    /// Main processing function (no-op entry point for trait conformance)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Goal management
    // -----------------------------------------------------------------------

    /// Add a goal to the registry
    pub fn add_goal(&mut self, goal: Goal) -> Result<()> {
        if self.goals.len() >= self.config.max_goals {
            return Err(Error::ResourceExhausted(format!(
                "maximum goal count ({}) reached",
                self.config.max_goals
            )));
        }
        if self.goals.iter().any(|g| g.id == goal.id) {
            return Err(Error::InvalidInput(format!(
                "goal with id '{}' already exists",
                goal.id
            )));
        }
        if goal.target.is_nan() || goal.target.is_infinite() {
            return Err(Error::InvalidInput("goal target must be finite".into()));
        }
        self.goals.push(goal);
        Ok(())
    }

    /// Remove a goal by ID
    pub fn remove_goal(&mut self, goal_id: &str) -> Result<()> {
        let idx = self
            .goals
            .iter()
            .position(|g| g.id == goal_id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", goal_id)))?;
        self.goals.remove(idx);
        Ok(())
    }

    /// Update a goal's target value
    pub fn update_target(&mut self, goal_id: &str, new_target: f64) -> Result<()> {
        if new_target.is_nan() || new_target.is_infinite() {
            return Err(Error::InvalidInput("target must be finite".into()));
        }
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", goal_id)))?;
        goal.target = new_target;
        Ok(())
    }

    /// Set a goal's active state
    pub fn set_active(&mut self, goal_id: &str, active: bool) -> Result<()> {
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", goal_id)))?;
        goal.active = active;
        Ok(())
    }

    /// Number of registered goals
    pub fn goal_count(&self) -> usize {
        self.goals.len()
    }

    /// Number of active goals
    pub fn active_goal_count(&self) -> usize {
        self.goals.iter().filter(|g| g.active).count()
    }

    /// Get a goal by ID
    pub fn get_goal(&self, goal_id: &str) -> Option<&Goal> {
        self.goals.iter().find(|g| g.id == goal_id)
    }

    /// Get all goals
    pub fn goals(&self) -> &[Goal] {
        &self.goals
    }

    /// Get goals filtered by priority
    pub fn goals_by_priority(&self, priority: GoalPriority) -> Vec<&Goal> {
        self.goals
            .iter()
            .filter(|g| g.priority == priority)
            .collect()
    }

    /// Get only critical goals
    pub fn critical_goals(&self) -> Vec<&Goal> {
        self.goals_by_priority(GoalPriority::Critical)
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate all active goals against the current portfolio state
    pub fn evaluate(&mut self, state: &PortfolioState) -> EvaluationSummary {
        let active_goals: Vec<&Goal> = self.goals.iter().filter(|g| g.active).collect();
        let goals_evaluated = active_goals.len();

        let mut evaluations = Vec::with_capacity(goals_evaluated);
        let mut goals_satisfied = 0usize;
        let mut goals_breached = 0usize;
        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;
        let mut min_attainment = 1.0f64;
        let mut worst_goal_id = String::new();
        let mut critical_goals_met = true;

        for goal in &active_goals {
            let observed = state.value_for(&goal.goal_type);
            let att = goal.attainment(observed);
            let satisfied = goal.is_satisfied(observed);
            let breach = goal.breach(observed);

            if satisfied {
                goals_satisfied += 1;
            } else if !goal.is_within_tolerance(observed) {
                goals_breached += 1;
            }

            if goal.priority == GoalPriority::Critical && !satisfied {
                critical_goals_met = false;
            }

            let w = goal.priority.weight();
            weighted_sum += att * w;
            weight_total += w;

            if att < min_attainment {
                min_attainment = att;
                worst_goal_id = goal.id.clone();
            }

            evaluations.push(GoalEvaluation {
                goal_id: goal.id.clone(),
                observed,
                target: goal.target,
                attainment: att,
                satisfied,
                breach,
                priority: goal.priority,
            });
        }

        let weighted_attainment = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            1.0
        };

        let feasibility = if goals_evaluated > 0 {
            goals_satisfied as f64 / goals_evaluated as f64
        } else {
            1.0
        };

        // Update stats
        self.stats.total_evaluations += 1;

        if critical_goals_met {
            self.stats.critical_met_count += 1;
        } else {
            self.stats.critical_breach_count += 1;
        }

        if weighted_attainment > self.stats.best_attainment {
            self.stats.best_attainment = weighted_attainment;
        }
        if weighted_attainment < self.stats.worst_attainment {
            self.stats.worst_attainment = weighted_attainment;
        }

        // EMA updates
        let satisfaction_rate = if goals_evaluated > 0 {
            goals_satisfied as f64 / goals_evaluated as f64
        } else {
            1.0
        };

        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_attainment = weighted_attainment;
            self.stats.ema_satisfaction_rate = satisfaction_rate;
            self.stats.ema_min_attainment = min_attainment;
            self.ema_initialized = true;
        } else {
            self.stats.ema_attainment =
                alpha * weighted_attainment + (1.0 - alpha) * self.stats.ema_attainment;
            self.stats.ema_satisfaction_rate =
                alpha * satisfaction_rate + (1.0 - alpha) * self.stats.ema_satisfaction_rate;
            self.stats.ema_min_attainment =
                alpha * min_attainment + (1.0 - alpha) * self.stats.ema_min_attainment;
        }

        let summary = EvaluationSummary {
            goals_evaluated,
            goals_satisfied,
            goals_breached,
            weighted_attainment,
            min_attainment,
            worst_goal_id,
            critical_goals_met,
            evaluations,
            feasibility,
        };

        // Store in sliding window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(summary.clone());

        summary
    }

    // -----------------------------------------------------------------------
    // Feasibility analysis
    // -----------------------------------------------------------------------

    /// Check whether a proposed set of targets is internally consistent.
    ///
    /// Returns a feasibility score (0–1). Simple heuristic: checks that
    /// return target doesn't exceed what the risk budget would allow under
    /// a reasonable Sharpe assumption.
    pub fn feasibility_check(&self) -> f64 {
        let return_goal = self
            .goals
            .iter()
            .find(|g| g.active && g.goal_type == GoalType::ReturnTarget);
        let risk_goal = self
            .goals
            .iter()
            .find(|g| g.active && g.goal_type == GoalType::RiskBudget);
        let sharpe_goal = self
            .goals
            .iter()
            .find(|g| g.active && g.goal_type == GoalType::SharpeFloor);

        match (return_goal, risk_goal) {
            (Some(ret), Some(risk)) => {
                // Implied Sharpe needed: return_target / risk_budget
                let implied_sharpe = if risk.target.abs() > 1e-15 {
                    ret.target / risk.target
                } else {
                    f64::INFINITY
                };

                // Check against Sharpe floor if set
                let sharpe_feasible = match sharpe_goal {
                    Some(sg) => implied_sharpe >= sg.target * 0.8, // 80% buffer
                    None => true,
                };

                // Heuristic: Sharpe > 3 is extremely unlikely
                let sharpe_reasonable = implied_sharpe <= 3.0;

                if sharpe_feasible && sharpe_reasonable {
                    1.0
                } else if sharpe_feasible || sharpe_reasonable {
                    0.5
                } else {
                    0.0
                }
            }
            _ => 1.0, // Can't assess feasibility without both return and risk goals
        }
    }

    /// Suggest adjusted targets to improve feasibility.
    ///
    /// Returns (suggested_return, suggested_vol) that would achieve
    /// the desired Sharpe within a reasonable risk budget.
    pub fn suggest_targets(&self, desired_sharpe: f64, max_vol: f64) -> (f64, f64) {
        let vol = max_vol.max(0.01);
        let ret = desired_sharpe * vol;
        (ret, vol)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Running statistics
    pub fn stats(&self) -> &GoalSettingStats {
        &self.stats
    }

    /// Configuration
    pub fn config(&self) -> &GoalSettingConfig {
        &self.config
    }

    /// Recent evaluation summaries (sliding window)
    pub fn recent_evaluations(&self) -> &VecDeque<EvaluationSummary> {
        &self.recent
    }

    /// EMA-smoothed weighted attainment
    pub fn smoothed_attainment(&self) -> f64 {
        self.stats.ema_attainment
    }

    /// EMA-smoothed satisfaction rate
    pub fn smoothed_satisfaction_rate(&self) -> f64 {
        self.stats.ema_satisfaction_rate
    }

    /// EMA-smoothed minimum attainment
    pub fn smoothed_min_attainment(&self) -> f64 {
        self.stats.ema_min_attainment
    }

    /// Windowed average attainment (over recent evaluations)
    pub fn windowed_attainment(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.weighted_attainment).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed satisfaction rate
    pub fn windowed_satisfaction_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.feasibility).sum();
        sum / self.recent.len() as f64
    }

    /// Whether attainment is trending upward based on sliding window
    pub fn is_attainment_improving(&self) -> bool {
        let n = self.recent.len();
        if n < 4 {
            return false;
        }
        let mid = n / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|s| s.weighted_attainment)
            .sum::<f64>()
            / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.weighted_attainment)
            .sum::<f64>()
            / (n - mid) as f64;
        second_half > first_half * 1.02 // 2% improvement threshold
    }

    /// Reset statistics and window (goals are preserved)
    pub fn reset(&mut self) {
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = GoalSettingStats::default();
    }
}

// ---------------------------------------------------------------------------
// Preset goals
// ---------------------------------------------------------------------------

/// Create a standard conservative portfolio goal set
pub fn preset_conservative_goals() -> Vec<Goal> {
    vec![
        Goal {
            id: "return_conservative".into(),
            label: "Conservative return target".into(),
            goal_type: GoalType::ReturnTarget,
            target: 0.06,
            direction: ConstraintDirection::AtLeast,
            priority: GoalPriority::Medium,
            tolerance: 0.20,
            active: true,
        },
        Goal {
            id: "risk_conservative".into(),
            label: "Conservative risk budget".into(),
            goal_type: GoalType::RiskBudget,
            target: 0.10,
            direction: ConstraintDirection::AtMost,
            priority: GoalPriority::High,
            tolerance: 0.10,
            active: true,
        },
        Goal {
            id: "dd_conservative".into(),
            label: "Conservative drawdown limit".into(),
            goal_type: GoalType::DrawdownLimit,
            target: 0.08,
            direction: ConstraintDirection::AtMost,
            priority: GoalPriority::Critical,
            tolerance: 0.05,
            active: true,
        },
        Goal {
            id: "liquidity_conservative".into(),
            label: "Conservative liquidity buffer".into(),
            goal_type: GoalType::LiquidityBuffer,
            target: 0.10,
            direction: ConstraintDirection::AtLeast,
            priority: GoalPriority::High,
            tolerance: 0.15,
            active: true,
        },
    ]
}

/// Create a standard aggressive portfolio goal set
pub fn preset_aggressive_goals() -> Vec<Goal> {
    vec![
        Goal {
            id: "return_aggressive".into(),
            label: "Aggressive return target".into(),
            goal_type: GoalType::ReturnTarget,
            target: 0.20,
            direction: ConstraintDirection::AtLeast,
            priority: GoalPriority::High,
            tolerance: 0.25,
            active: true,
        },
        Goal {
            id: "risk_aggressive".into(),
            label: "Aggressive risk budget".into(),
            goal_type: GoalType::RiskBudget,
            target: 0.30,
            direction: ConstraintDirection::AtMost,
            priority: GoalPriority::Medium,
            tolerance: 0.15,
            active: true,
        },
        Goal {
            id: "dd_aggressive".into(),
            label: "Aggressive drawdown limit".into(),
            goal_type: GoalType::DrawdownLimit,
            target: 0.20,
            direction: ConstraintDirection::AtMost,
            priority: GoalPriority::Critical,
            tolerance: 0.10,
            active: true,
        },
        Goal {
            id: "sharpe_aggressive".into(),
            label: "Aggressive Sharpe floor".into(),
            goal_type: GoalType::SharpeFloor,
            target: 0.80,
            direction: ConstraintDirection::AtLeast,
            priority: GoalPriority::Medium,
            tolerance: 0.20,
            active: true,
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn return_goal(id: &str, target: f64, priority: GoalPriority) -> Goal {
        Goal {
            id: id.into(),
            label: format!("Return {}", id),
            goal_type: GoalType::ReturnTarget,
            target,
            direction: ConstraintDirection::AtLeast,
            priority,
            tolerance: 0.10,
            active: true,
        }
    }

    fn risk_goal(id: &str, target: f64, priority: GoalPriority) -> Goal {
        Goal {
            id: id.into(),
            label: format!("Risk {}", id),
            goal_type: GoalType::RiskBudget,
            target,
            direction: ConstraintDirection::AtMost,
            priority,
            tolerance: 0.10,
            active: true,
        }
    }

    fn good_state() -> PortfolioState {
        PortfolioState {
            annualised_return: 0.12,
            volatility: 0.10,
            max_drawdown: 0.05,
            sharpe: 1.2,
            max_concentration: 0.15,
            cash_fraction: 0.10,
            portfolio_value: 1_100_000.0,
        }
    }

    fn poor_state() -> PortfolioState {
        PortfolioState {
            annualised_return: -0.05,
            volatility: 0.40,
            max_drawdown: 0.25,
            sharpe: -0.30,
            max_concentration: 0.50,
            cash_fraction: 0.02,
            portfolio_value: 800_000.0,
        }
    }

    // -- Construction --

    #[test]
    fn test_new_default() {
        let gs = GoalSetting::new();
        assert_eq!(gs.goal_count(), 0);
        assert!(gs.process().is_ok());
    }

    #[test]
    fn test_with_config() {
        let gs = GoalSetting::with_config(GoalSettingConfig::default());
        assert!(gs.is_ok());
    }

    #[test]
    fn test_invalid_config_zero_max_goals() {
        let mut cfg = GoalSettingConfig::default();
        cfg.max_goals = 0;
        assert!(GoalSetting::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_zero() {
        let mut cfg = GoalSettingConfig::default();
        cfg.ema_decay = 0.0;
        assert!(GoalSetting::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_one() {
        let mut cfg = GoalSettingConfig::default();
        cfg.ema_decay = 1.0;
        assert!(GoalSetting::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let mut cfg = GoalSettingConfig::default();
        cfg.window_size = 0;
        assert!(GoalSetting::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_tolerance_negative() {
        let mut cfg = GoalSettingConfig::default();
        cfg.default_tolerance = -0.1;
        assert!(GoalSetting::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_tolerance_too_large() {
        let mut cfg = GoalSettingConfig::default();
        cfg.default_tolerance = 1.5;
        assert!(GoalSetting::with_config(cfg).is_err());
    }

    // -- Goal management --

    #[test]
    fn test_add_goal() {
        let mut gs = GoalSetting::new();
        let g = return_goal("r1", 0.10, GoalPriority::Medium);
        assert!(gs.add_goal(g).is_ok());
        assert_eq!(gs.goal_count(), 1);
    }

    #[test]
    fn test_add_duplicate_goal() {
        let mut gs = GoalSetting::new();
        let g1 = return_goal("dup", 0.10, GoalPriority::Medium);
        let g2 = return_goal("dup", 0.20, GoalPriority::High);
        assert!(gs.add_goal(g1).is_ok());
        assert!(gs.add_goal(g2).is_err());
    }

    #[test]
    fn test_add_goal_nan_target() {
        let mut gs = GoalSetting::new();
        let g = return_goal("nan", f64::NAN, GoalPriority::Medium);
        assert!(gs.add_goal(g).is_err());
    }

    #[test]
    fn test_add_goal_inf_target() {
        let mut gs = GoalSetting::new();
        let g = return_goal("inf", f64::INFINITY, GoalPriority::Medium);
        assert!(gs.add_goal(g).is_err());
    }

    #[test]
    fn test_add_goal_exceeds_max() {
        let mut cfg = GoalSettingConfig::default();
        cfg.max_goals = 1;
        let mut gs = GoalSetting::with_config(cfg).unwrap();
        let g1 = return_goal("r1", 0.10, GoalPriority::Medium);
        let g2 = return_goal("r2", 0.20, GoalPriority::Medium);
        assert!(gs.add_goal(g1).is_ok());
        assert!(gs.add_goal(g2).is_err());
    }

    #[test]
    fn test_remove_goal() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        assert_eq!(gs.goal_count(), 1);
        assert!(gs.remove_goal("r1").is_ok());
        assert_eq!(gs.goal_count(), 0);
    }

    #[test]
    fn test_remove_goal_not_found() {
        let mut gs = GoalSetting::new();
        assert!(gs.remove_goal("nope").is_err());
    }

    #[test]
    fn test_update_target() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        assert!(gs.update_target("r1", 0.15).is_ok());
        assert!((gs.get_goal("r1").unwrap().target - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_update_target_nan() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        assert!(gs.update_target("r1", f64::NAN).is_err());
    }

    #[test]
    fn test_update_target_not_found() {
        let mut gs = GoalSetting::new();
        assert!(gs.update_target("nope", 0.10).is_err());
    }

    #[test]
    fn test_set_active() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        assert!(gs.set_active("r1", false).is_ok());
        assert!(!gs.get_goal("r1").unwrap().active);
        assert_eq!(gs.active_goal_count(), 0);
    }

    #[test]
    fn test_set_active_not_found() {
        let mut gs = GoalSetting::new();
        assert!(gs.set_active("nope", true).is_err());
    }

    #[test]
    fn test_goals_by_priority() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.add_goal(return_goal("r2", 0.20, GoalPriority::High))
            .unwrap();
        gs.add_goal(risk_goal("rk1", 0.15, GoalPriority::High))
            .unwrap();
        assert_eq!(gs.goals_by_priority(GoalPriority::High).len(), 2);
        assert_eq!(gs.goals_by_priority(GoalPriority::Medium).len(), 1);
        assert_eq!(gs.goals_by_priority(GoalPriority::Low).len(), 0);
    }

    #[test]
    fn test_critical_goals() {
        let mut gs = GoalSetting::new();
        gs.add_goal(Goal {
            id: "crit".into(),
            label: "Critical DD".into(),
            goal_type: GoalType::DrawdownLimit,
            target: 0.10,
            direction: ConstraintDirection::AtMost,
            priority: GoalPriority::Critical,
            tolerance: 0.05,
            active: true,
        })
        .unwrap();
        assert_eq!(gs.critical_goals().len(), 1);
    }

    // -- Goal attainment --

    #[test]
    fn test_attainment_at_least_satisfied() {
        let g = return_goal("r", 0.10, GoalPriority::Medium);
        assert_eq!(g.attainment(0.12), 1.0);
    }

    #[test]
    fn test_attainment_at_least_exactly_met() {
        let g = return_goal("r", 0.10, GoalPriority::Medium);
        assert_eq!(g.attainment(0.10), 1.0);
    }

    #[test]
    fn test_attainment_at_least_within_tolerance() {
        let g = return_goal("r", 0.10, GoalPriority::Medium);
        // tolerance = 0.10 (10% of target)
        // observed = 0.095 → deficit = 0.005/0.10 = 0.05 < 0.10
        let att = g.attainment(0.095);
        assert!(att > 0.0);
        assert!(att < 1.0);
    }

    #[test]
    fn test_attainment_at_least_breached() {
        let g = return_goal("r", 0.10, GoalPriority::Medium);
        // observed = 0.05 → deficit = 0.05/0.10 = 0.50 >> tolerance 0.10
        assert_eq!(g.attainment(0.05), 0.0);
    }

    #[test]
    fn test_attainment_at_most_satisfied() {
        let g = risk_goal("rk", 0.15, GoalPriority::High);
        assert_eq!(g.attainment(0.10), 1.0);
    }

    #[test]
    fn test_attainment_at_most_exactly_met() {
        let g = risk_goal("rk", 0.15, GoalPriority::High);
        assert_eq!(g.attainment(0.15), 1.0);
    }

    #[test]
    fn test_attainment_at_most_breached() {
        let g = risk_goal("rk", 0.15, GoalPriority::High);
        // observed = 0.30 → excess = 0.15/0.15 = 1.0 >> tolerance
        assert_eq!(g.attainment(0.30), 0.0);
    }

    #[test]
    fn test_attainment_zero_target_at_least() {
        let mut g = return_goal("r", 0.0, GoalPriority::Medium);
        g.target = 0.0;
        assert_eq!(g.attainment(0.05), 1.0);
        assert_eq!(g.attainment(-0.05), 0.0);
    }

    #[test]
    fn test_is_satisfied() {
        let g = return_goal("r", 0.10, GoalPriority::Medium);
        assert!(g.is_satisfied(0.10));
        assert!(g.is_satisfied(0.12));
        assert!(!g.is_satisfied(0.09));
    }

    #[test]
    fn test_breach_magnitude() {
        let g = return_goal("r", 0.10, GoalPriority::Medium);
        assert_eq!(g.breach(0.12), 0.0);
        assert!((g.breach(0.06) - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_breach_at_most() {
        let g = risk_goal("rk", 0.15, GoalPriority::High);
        assert_eq!(g.breach(0.10), 0.0);
        assert!((g.breach(0.20) - 0.05).abs() < 1e-10);
    }

    // -- Evaluation --

    #[test]
    fn test_evaluate_no_goals() {
        let mut gs = GoalSetting::new();
        let summary = gs.evaluate(&good_state());
        assert_eq!(summary.goals_evaluated, 0);
        assert_eq!(summary.weighted_attainment, 1.0);
        assert!(summary.critical_goals_met);
    }

    #[test]
    fn test_evaluate_good_state_all_met() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.add_goal(risk_goal("rk1", 0.15, GoalPriority::High))
            .unwrap();

        let summary = gs.evaluate(&good_state());
        assert_eq!(summary.goals_evaluated, 2);
        assert_eq!(summary.goals_satisfied, 2);
        assert_eq!(summary.goals_breached, 0);
        assert_eq!(summary.weighted_attainment, 1.0);
        assert!(summary.critical_goals_met);
    }

    #[test]
    fn test_evaluate_poor_state_some_breached() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.add_goal(risk_goal("rk1", 0.15, GoalPriority::High))
            .unwrap();

        let summary = gs.evaluate(&poor_state());
        // Return -0.05 < 0.10 → breached
        // Vol 0.40 > 0.15 → breached
        assert!(summary.goals_breached > 0);
        assert!(summary.weighted_attainment < 1.0);
    }

    #[test]
    fn test_evaluate_critical_goal_breach() {
        let mut gs = GoalSetting::new();
        gs.add_goal(Goal {
            id: "dd_crit".into(),
            label: "Critical DD".into(),
            goal_type: GoalType::DrawdownLimit,
            target: 0.10,
            direction: ConstraintDirection::AtMost,
            priority: GoalPriority::Critical,
            tolerance: 0.05,
            active: true,
        })
        .unwrap();

        // poor_state has max_drawdown = 0.25 > 0.10
        let summary = gs.evaluate(&poor_state());
        assert!(!summary.critical_goals_met);
    }

    #[test]
    fn test_evaluate_inactive_goals_skipped() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.set_active("r1", false).unwrap();

        let summary = gs.evaluate(&good_state());
        assert_eq!(summary.goals_evaluated, 0);
    }

    #[test]
    fn test_evaluate_worst_goal_tracked() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("good_ret", 0.05, GoalPriority::Medium))
            .unwrap();
        gs.add_goal(risk_goal("bad_risk", 0.05, GoalPriority::High))
            .unwrap();

        // good_state: return = 0.12 >= 0.05 (met), vol = 0.10 > 0.05 (breached)
        let summary = gs.evaluate(&good_state());
        assert_eq!(summary.worst_goal_id, "bad_risk");
        assert!(summary.min_attainment < 1.0);
    }

    // -- Statistics tracking --

    #[test]
    fn test_stats_initial() {
        let gs = GoalSetting::new();
        assert_eq!(gs.stats().total_evaluations, 0);
    }

    #[test]
    fn test_stats_after_evaluation() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.evaluate(&good_state());

        assert_eq!(gs.stats().total_evaluations, 1);
        assert!(gs.stats().ema_attainment > 0.0);
    }

    #[test]
    fn test_stats_best_worst_attainment() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.evaluate(&good_state());
        gs.evaluate(&poor_state());

        assert!(gs.stats().best_attainment >= gs.stats().worst_attainment);
    }

    #[test]
    fn test_stats_critical_satisfaction_rate() {
        let stats = GoalSettingStats {
            critical_met_count: 8,
            critical_breach_count: 2,
            ..Default::default()
        };
        assert!((stats.critical_satisfaction_rate() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_stats_critical_satisfaction_rate_zero() {
        let stats = GoalSettingStats::default();
        assert_eq!(stats.critical_satisfaction_rate(), 1.0);
    }

    // -- EMA tracking --

    #[test]
    fn test_ema_initializes_first_eval() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.evaluate(&good_state());
        assert_eq!(gs.smoothed_attainment(), 1.0);
    }

    #[test]
    fn test_ema_blends_on_subsequent() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.evaluate(&good_state()); // attainment = 1.0
        let att1 = gs.smoothed_attainment();

        gs.evaluate(&poor_state()); // attainment < 1.0
        let att2 = gs.smoothed_attainment();
        assert!(att2 < att1);
    }

    // -- Sliding window --

    #[test]
    fn test_recent_evaluations_stored() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.evaluate(&good_state());
        assert_eq!(gs.recent_evaluations().len(), 1);
    }

    #[test]
    fn test_recent_evaluations_windowed() {
        let mut cfg = GoalSettingConfig::default();
        cfg.window_size = 3;
        let mut gs = GoalSetting::with_config(cfg).unwrap();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();

        for _ in 0..10 {
            gs.evaluate(&good_state());
        }
        assert!(gs.recent_evaluations().len() <= 3);
    }

    #[test]
    fn test_windowed_attainment() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.evaluate(&good_state());
        assert_eq!(gs.windowed_attainment(), 1.0);
    }

    #[test]
    fn test_windowed_empty() {
        let gs = GoalSetting::new();
        assert_eq!(gs.windowed_attainment(), 0.0);
        assert_eq!(gs.windowed_satisfaction_rate(), 0.0);
    }

    // -- Trend detection --

    #[test]
    fn test_is_attainment_improving_insufficient_data() {
        let gs = GoalSetting::new();
        assert!(!gs.is_attainment_improving());
    }

    // -- Feasibility --

    #[test]
    fn test_feasibility_no_goals() {
        let gs = GoalSetting::new();
        assert_eq!(gs.feasibility_check(), 1.0);
    }

    #[test]
    fn test_feasibility_reasonable_targets() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.add_goal(risk_goal("rk", 0.15, GoalPriority::High))
            .unwrap();
        // implied Sharpe = 0.10 / 0.15 ≈ 0.67, reasonable
        assert_eq!(gs.feasibility_check(), 1.0);
    }

    #[test]
    fn test_feasibility_unreasonable_targets() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r", 0.50, GoalPriority::Medium))
            .unwrap();
        gs.add_goal(risk_goal("rk", 0.05, GoalPriority::High))
            .unwrap();
        // implied Sharpe = 0.50 / 0.05 = 10.0 >> 3.0
        assert!(gs.feasibility_check() < 1.0);
    }

    #[test]
    fn test_suggest_targets() {
        let gs = GoalSetting::new();
        let (ret, vol) = gs.suggest_targets(1.0, 0.15);
        assert!((ret - 0.15).abs() < 1e-10);
        assert!((vol - 0.15).abs() < 1e-10);
    }

    // -- Goal priority --

    #[test]
    fn test_priority_ordering() {
        assert!(GoalPriority::Low < GoalPriority::Medium);
        assert!(GoalPriority::Medium < GoalPriority::High);
        assert!(GoalPriority::High < GoalPriority::Critical);
    }

    #[test]
    fn test_priority_weight() {
        assert!(GoalPriority::Critical.weight() > GoalPriority::High.weight());
        assert!(GoalPriority::High.weight() > GoalPriority::Medium.weight());
        assert!(GoalPriority::Medium.weight() > GoalPriority::Low.weight());
    }

    // -- Constraint direction --

    #[test]
    fn test_constraint_direction_at_least() {
        let g = return_goal("r", 0.10, GoalPriority::Medium);
        assert!(g.is_satisfied(0.15));
        assert!(!g.is_satisfied(0.05));
    }

    #[test]
    fn test_constraint_direction_at_most() {
        let g = risk_goal("rk", 0.15, GoalPriority::High);
        assert!(g.is_satisfied(0.10));
        assert!(!g.is_satisfied(0.20));
    }

    // -- Portfolio state --

    #[test]
    fn test_portfolio_state_default() {
        let ps = PortfolioState::default();
        assert!(ps.portfolio_value > 0.0);
        assert!(ps.volatility > 0.0);
    }

    #[test]
    fn test_portfolio_state_value_for() {
        let ps = good_state();
        assert!((ps.value_for(&GoalType::ReturnTarget) - 0.12).abs() < 1e-10);
        assert!((ps.value_for(&GoalType::RiskBudget) - 0.10).abs() < 1e-10);
        assert!((ps.value_for(&GoalType::DrawdownLimit) - 0.05).abs() < 1e-10);
        assert!((ps.value_for(&GoalType::SharpeFloor) - 1.2).abs() < 1e-10);
    }

    // -- Presets --

    #[test]
    fn test_preset_conservative_goals() {
        let goals = preset_conservative_goals();
        assert!(!goals.is_empty());
        assert!(goals.iter().all(|g| g.active));
        // Should have at least one critical goal
        assert!(goals.iter().any(|g| g.priority == GoalPriority::Critical));
    }

    #[test]
    fn test_preset_aggressive_goals() {
        let goals = preset_aggressive_goals();
        assert!(!goals.is_empty());
        assert!(goals.iter().any(|g| g.goal_type == GoalType::SharpeFloor));
    }

    #[test]
    fn test_preset_goals_can_be_added() {
        let mut gs = GoalSetting::new();
        for g in preset_conservative_goals() {
            assert!(gs.add_goal(g).is_ok());
        }
        assert_eq!(gs.goal_count(), 4);
    }

    #[test]
    fn test_preset_conservative_met_in_good_state() {
        let mut gs = GoalSetting::new();
        for g in preset_conservative_goals() {
            gs.add_goal(g).unwrap();
        }
        let summary = gs.evaluate(&good_state());
        // good_state should satisfy conservative goals:
        // return 0.12 >= 0.06 ✓, vol 0.10 <= 0.10 ✓, dd 0.05 <= 0.08 ✓, cash 0.10 >= 0.10 ✓
        assert!(summary.critical_goals_met);
        assert_eq!(summary.goals_satisfied, summary.goals_evaluated);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.evaluate(&good_state());
        gs.evaluate(&poor_state());

        assert!(gs.stats().total_evaluations > 0);
        assert!(!gs.recent_evaluations().is_empty());

        gs.reset();

        assert_eq!(gs.stats().total_evaluations, 0);
        assert!(gs.recent_evaluations().is_empty());
        // Goals preserved
        assert_eq!(gs.goal_count(), 1);
    }

    // -- Integration-style --

    #[test]
    fn test_full_lifecycle() {
        let mut gs = GoalSetting::new();
        for g in preset_conservative_goals() {
            gs.add_goal(g).unwrap();
        }

        // Phase 1: good market
        for _ in 0..5 {
            let summary = gs.evaluate(&good_state());
            assert!(summary.critical_goals_met);
        }

        // Phase 2: deteriorating
        for _ in 0..5 {
            gs.evaluate(&poor_state());
        }

        // Stats should reflect both phases
        assert_eq!(gs.stats().total_evaluations, 10);
        assert!(gs.stats().ema_attainment < 1.0);
        assert!(gs.stats().critical_breach_count > 0);
        assert!(gs.stats().best_attainment > gs.stats().worst_attainment);
    }

    // -- Evaluation summary feasibility --

    #[test]
    fn test_evaluation_feasibility_score() {
        let mut gs = GoalSetting::new();
        gs.add_goal(return_goal("r1", 0.10, GoalPriority::Medium))
            .unwrap();
        gs.add_goal(risk_goal("rk1", 0.15, GoalPriority::High))
            .unwrap();

        let summary = gs.evaluate(&good_state());
        assert_eq!(summary.feasibility, 1.0);

        let summary2 = gs.evaluate(&poor_state());
        assert!(summary2.feasibility < 1.0);
    }
}
