//! Constraint Satisfaction Solver
//!
//! Part of the Prefrontal region
//! Component: ltn
//!
//! Provides a constraint satisfaction engine for trading rule enforcement.
//! Constraints are expressed as predicate-threshold pairs evaluated against
//! a named-variable context. The solver supports hard and soft constraints,
//! priority ordering, iterative propagation, and detailed violation reporting.
//!
//! ## Features
//!
//! - **Hard vs soft constraints**: Hard constraints must be satisfied or the
//!   entire solution is infeasible. Soft constraints contribute to a penalty
//!   score but do not block feasibility.
//! - **Priority ordering**: Constraints carry a priority (0 = highest) that
//!   determines evaluation order and violation severity weighting.
//! - **Predicate-threshold pairs**: Each constraint evaluates a named variable
//!   against a threshold using a comparison operator (≥, ≤, =, ≠, >, <).
//! - **Compound constraints**: AND / OR grouping of atomic constraints.
//! - **Propagation**: Iterative constraint tightening — when a variable is
//!   clamped by a hard constraint, downstream constraints are re-evaluated.
//! - **Violation reporting**: Detailed per-constraint violation records with
//!   severity, variable name, actual vs expected values.
//! - **EMA-smoothed feasibility signal**: Running feasibility fraction with
//!   configurable decay for downstream consumption.
//! - **Windowed diagnostics**: Recent violation rate, mean penalty, trend
//!   detection.
//! - **Running statistics**: Total checks, violations, feasibility rate,
//!   penalty distribution.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the constraint solver.
#[derive(Debug, Clone)]
pub struct ConstraintSolverConfig {
    /// EMA decay for feasibility smoothing (0 < decay < 1).
    pub ema_decay: f64,
    /// Minimum evaluations before EMA is considered initialised.
    pub min_samples: usize,
    /// Sliding window size for windowed diagnostics.
    pub window_size: usize,
    /// Maximum number of constraints that can be registered.
    pub max_constraints: usize,
    /// Maximum propagation iterations (prevents infinite loops).
    pub max_propagation_iters: usize,
    /// Default soft-constraint penalty weight.
    pub default_penalty_weight: f64,
}

impl Default for ConstraintSolverConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.1,
            min_samples: 5,
            window_size: 100,
            max_constraints: 256,
            max_propagation_iters: 10,
            default_penalty_weight: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison operator
// ---------------------------------------------------------------------------

/// Comparison operators for constraint evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    /// Greater than or equal (≥)
    Gte,
    /// Less than or equal (≤)
    Lte,
    /// Greater than (>)
    Gt,
    /// Less than (<)
    Lt,
    /// Equal (within epsilon)
    Eq,
    /// Not equal (outside epsilon)
    Ne,
}

impl CompareOp {
    /// Evaluate the comparison `actual op threshold`.
    pub fn evaluate(self, actual: f64, threshold: f64) -> bool {
        const EPS: f64 = 1e-9;
        match self {
            CompareOp::Gte => actual >= threshold - EPS,
            CompareOp::Lte => actual <= threshold + EPS,
            CompareOp::Gt => actual > threshold + EPS,
            CompareOp::Lt => actual < threshold - EPS,
            CompareOp::Eq => (actual - threshold).abs() < EPS,
            CompareOp::Ne => (actual - threshold).abs() >= EPS,
        }
    }

    /// Human-readable symbol.
    pub fn symbol(self) -> &'static str {
        match self {
            CompareOp::Gte => ">=",
            CompareOp::Lte => "<=",
            CompareOp::Gt => ">",
            CompareOp::Lt => "<",
            CompareOp::Eq => "==",
            CompareOp::Ne => "!=",
        }
    }
}

impl std::fmt::Display for CompareOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

// ---------------------------------------------------------------------------
// Constraint kind
// ---------------------------------------------------------------------------

/// Whether a constraint is hard (must-satisfy) or soft (penalty-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstraintKind {
    /// Hard constraint — violation means infeasible.
    Hard,
    /// Soft constraint — violation incurs a penalty but solution remains feasible.
    Soft,
}

// ---------------------------------------------------------------------------
// Atomic constraint
// ---------------------------------------------------------------------------

/// An atomic constraint: `variable op threshold`.
#[derive(Debug, Clone)]
pub struct AtomicConstraint {
    /// Unique identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Variable name to evaluate.
    pub variable: String,
    /// Comparison operator.
    pub op: CompareOp,
    /// Threshold value.
    pub threshold: f64,
    /// Hard or soft.
    pub kind: ConstraintKind,
    /// Priority (0 = highest).
    pub priority: u32,
    /// Penalty weight for soft constraints.
    pub penalty_weight: f64,
}

impl AtomicConstraint {
    /// Create a new hard constraint.
    pub fn hard(
        id: impl Into<String>,
        variable: impl Into<String>,
        op: CompareOp,
        threshold: f64,
    ) -> Self {
        Self {
            id: id.into(),
            description: String::new(),
            variable: variable.into(),
            op,
            threshold,
            kind: ConstraintKind::Hard,
            priority: 0,
            penalty_weight: 1.0,
        }
    }

    /// Create a new soft constraint.
    pub fn soft(
        id: impl Into<String>,
        variable: impl Into<String>,
        op: CompareOp,
        threshold: f64,
        penalty_weight: f64,
    ) -> Self {
        Self {
            id: id.into(),
            description: String::new(),
            variable: variable.into(),
            op,
            threshold,
            kind: ConstraintKind::Soft,
            priority: 0,
            penalty_weight,
        }
    }

    /// Builder: set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Evaluate against a variable context.
    pub fn evaluate(&self, variables: &HashMap<String, f64>) -> Option<bool> {
        variables
            .get(&self.variable)
            .map(|&actual| self.op.evaluate(actual, self.threshold))
    }

    /// Compute the penalty for a soft constraint violation.
    /// Returns 0 if satisfied, or `penalty_weight * |actual - threshold|` if not.
    pub fn penalty(&self, variables: &HashMap<String, f64>) -> f64 {
        match self.evaluate(variables) {
            Some(true) => 0.0,
            Some(false) => {
                let actual = variables[&self.variable];
                self.penalty_weight * (actual - self.threshold).abs()
            }
            None => 0.0, // missing variable — no penalty
        }
    }
}

// ---------------------------------------------------------------------------
// Compound constraint
// ---------------------------------------------------------------------------

/// Compound constraint: logical combination of constraint IDs.
#[derive(Debug, Clone)]
pub enum CompoundConstraint {
    /// All must be satisfied.
    And(Vec<String>),
    /// At least one must be satisfied.
    Or(Vec<String>),
}

// ---------------------------------------------------------------------------
// Violation record
// ---------------------------------------------------------------------------

/// A single constraint violation.
#[derive(Debug, Clone)]
pub struct Violation {
    /// Constraint ID that was violated.
    pub constraint_id: String,
    /// Variable that caused the violation.
    pub variable: String,
    /// Actual value observed.
    pub actual: f64,
    /// Threshold that was not met.
    pub threshold: f64,
    /// Comparison operator.
    pub op: CompareOp,
    /// Whether this was a hard or soft constraint.
    pub kind: ConstraintKind,
    /// Priority of the violated constraint.
    pub priority: u32,
    /// Penalty incurred (0 for hard constraints).
    pub penalty: f64,
}

impl Violation {
    /// How far the actual value is from satisfying the constraint.
    pub fn gap(&self) -> f64 {
        (self.actual - self.threshold).abs()
    }
}

// ---------------------------------------------------------------------------
// Solve result
// ---------------------------------------------------------------------------

/// Result of a constraint satisfaction check.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Whether all hard constraints are satisfied.
    pub feasible: bool,
    /// Total penalty from soft constraint violations.
    pub total_penalty: f64,
    /// Number of satisfied constraints.
    pub satisfied: usize,
    /// Number of violated constraints.
    pub violated: usize,
    /// Number of constraints with missing variables (skipped).
    pub skipped: usize,
    /// Detailed violation records.
    pub violations: Vec<Violation>,
    /// Satisfaction ratio (satisfied / (satisfied + violated)).
    pub satisfaction_ratio: f64,
    /// EMA-smoothed feasibility (from the engine, not this single check).
    pub ema_feasibility: f64,
}

// ---------------------------------------------------------------------------
// Window record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WindowRecord {
    feasible: bool,
    total_penalty: f64,
    violation_count: usize,
    satisfaction_ratio: f64,
    tick: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the constraint solver.
#[derive(Debug, Clone)]
pub struct ConstraintSolverStats {
    /// Total number of solve rounds.
    pub total_solves: u64,
    /// Total number of feasible outcomes.
    pub feasible_count: u64,
    /// Total number of infeasible outcomes.
    pub infeasible_count: u64,
    /// Total constraint violations across all rounds.
    pub total_violations: u64,
    /// Total penalty accumulated across all rounds.
    pub total_penalty: f64,
    /// Peak single-round penalty.
    pub peak_penalty: f64,
    /// Sum of satisfaction ratios (for mean).
    pub sum_satisfaction_ratio: f64,
    /// Registered constraint count.
    pub registered_constraints: usize,
    /// Hard constraint count.
    pub hard_count: usize,
    /// Soft constraint count.
    pub soft_count: usize,
}

impl Default for ConstraintSolverStats {
    fn default() -> Self {
        Self {
            total_solves: 0,
            feasible_count: 0,
            infeasible_count: 0,
            total_violations: 0,
            total_penalty: 0.0,
            peak_penalty: 0.0,
            sum_satisfaction_ratio: 0.0,
            registered_constraints: 0,
            hard_count: 0,
            soft_count: 0,
        }
    }
}

impl ConstraintSolverStats {
    /// Feasibility rate over all rounds.
    pub fn feasibility_rate(&self) -> f64 {
        if self.total_solves == 0 {
            return 0.0;
        }
        self.feasible_count as f64 / self.total_solves as f64
    }

    /// Mean satisfaction ratio over all rounds.
    pub fn mean_satisfaction_ratio(&self) -> f64 {
        if self.total_solves == 0 {
            return 0.0;
        }
        self.sum_satisfaction_ratio / self.total_solves as f64
    }

    /// Mean penalty per round.
    pub fn mean_penalty(&self) -> f64 {
        if self.total_solves == 0 {
            return 0.0;
        }
        self.total_penalty / self.total_solves as f64
    }

    /// Mean violations per round.
    pub fn mean_violations(&self) -> f64 {
        if self.total_solves == 0 {
            return 0.0;
        }
        self.total_violations as f64 / self.total_solves as f64
    }
}

// ---------------------------------------------------------------------------
// ConstraintSolver
// ---------------------------------------------------------------------------

/// Constraint satisfaction solver for trading rule enforcement.
///
/// Register constraints (hard or soft), set variable values, and call `solve()`
/// to check feasibility, compute penalties, and collect violation details.
pub struct ConstraintSolver {
    config: ConstraintSolverConfig,

    /// Registered atomic constraints, keyed by ID.
    constraints: HashMap<String, AtomicConstraint>,

    /// Evaluation order (sorted by priority then ID).
    eval_order: Vec<String>,

    /// Compound constraints.
    compounds: HashMap<String, CompoundConstraint>,

    /// Current variable values.
    variables: HashMap<String, f64>,

    /// EMA-smoothed feasibility signal.
    ema_feasibility: f64,
    ema_initialized: bool,

    /// Sliding window.
    recent: VecDeque<WindowRecord>,

    /// Current tick.
    current_tick: u64,

    /// Running statistics.
    stats: ConstraintSolverStats,
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSolver {
    /// Create a new solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(ConstraintSolverConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: ConstraintSolverConfig) -> Result<Self> {
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.max_constraints == 0 {
            return Err(Error::InvalidInput("max_constraints must be > 0".into()));
        }
        if config.max_propagation_iters == 0 {
            return Err(Error::InvalidInput(
                "max_propagation_iters must be > 0".into(),
            ));
        }
        Ok(Self {
            config,
            constraints: HashMap::new(),
            eval_order: Vec::new(),
            compounds: HashMap::new(),
            variables: HashMap::new(),
            ema_feasibility: 0.0,
            ema_initialized: false,
            recent: VecDeque::new(),
            current_tick: 0,
            stats: ConstraintSolverStats::default(),
        })
    }

    // -----------------------------------------------------------------------
    // Constraint management
    // -----------------------------------------------------------------------

    /// Register an atomic constraint.
    pub fn add_constraint(&mut self, constraint: AtomicConstraint) -> Result<()> {
        if self.constraints.contains_key(&constraint.id) {
            return Err(Error::InvalidInput(format!(
                "constraint '{}' already registered",
                constraint.id
            )));
        }
        if self.constraints.len() >= self.config.max_constraints {
            return Err(Error::ResourceExhausted(format!(
                "maximum constraints ({}) reached",
                self.config.max_constraints
            )));
        }

        match constraint.kind {
            ConstraintKind::Hard => self.stats.hard_count += 1,
            ConstraintKind::Soft => self.stats.soft_count += 1,
        }

        self.constraints.insert(constraint.id.clone(), constraint);
        self.rebuild_eval_order();
        self.stats.registered_constraints = self.constraints.len();
        Ok(())
    }

    /// Remove a constraint by ID.
    pub fn remove_constraint(&mut self, id: &str) -> bool {
        if let Some(c) = self.constraints.remove(id) {
            match c.kind {
                ConstraintKind::Hard => {
                    self.stats.hard_count = self.stats.hard_count.saturating_sub(1)
                }
                ConstraintKind::Soft => {
                    self.stats.soft_count = self.stats.soft_count.saturating_sub(1)
                }
            }
            self.rebuild_eval_order();
            self.stats.registered_constraints = self.constraints.len();
            true
        } else {
            false
        }
    }

    /// Add a compound constraint.
    pub fn add_compound(
        &mut self,
        id: impl Into<String>,
        compound: CompoundConstraint,
    ) -> Result<()> {
        let id = id.into();

        // Validate all referenced IDs exist
        let refs = match &compound {
            CompoundConstraint::And(ids) => ids,
            CompoundConstraint::Or(ids) => ids,
        };
        for ref_id in refs {
            if !self.constraints.contains_key(ref_id) {
                return Err(Error::NotFound(format!(
                    "referenced constraint '{}' not found",
                    ref_id
                )));
            }
        }
        if refs.is_empty() {
            return Err(Error::InvalidInput(
                "compound constraint must reference at least one constraint".into(),
            ));
        }

        self.compounds.insert(id, compound);
        Ok(())
    }

    /// Remove a compound constraint.
    pub fn remove_compound(&mut self, id: &str) -> bool {
        self.compounds.remove(id).is_some()
    }

    /// Get a constraint by ID.
    pub fn constraint(&self, id: &str) -> Option<&AtomicConstraint> {
        self.constraints.get(id)
    }

    /// Number of registered atomic constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Number of hard constraints.
    pub fn hard_count(&self) -> usize {
        self.stats.hard_count
    }

    /// Number of soft constraints.
    pub fn soft_count(&self) -> usize {
        self.stats.soft_count
    }

    /// All constraint IDs in evaluation order.
    pub fn constraint_ids(&self) -> &[String] {
        &self.eval_order
    }

    fn rebuild_eval_order(&mut self) {
        let mut ids: Vec<_> = self.constraints.keys().cloned().collect();
        ids.sort_by(|a, b| {
            let ca = &self.constraints[a];
            let cb = &self.constraints[b];
            ca.priority.cmp(&cb.priority).then_with(|| a.cmp(b))
        });
        self.eval_order = ids;
    }

    // -----------------------------------------------------------------------
    // Variable management
    // -----------------------------------------------------------------------

    /// Set a variable value.
    pub fn set_variable(&mut self, name: impl Into<String>, value: f64) {
        self.variables.insert(name.into(), value);
    }

    /// Set multiple variables at once.
    pub fn set_variables(&mut self, vars: &[(&str, f64)]) {
        for (name, value) in vars {
            self.variables.insert((*name).to_string(), *value);
        }
    }

    /// Get a variable value.
    pub fn variable(&self, name: &str) -> Option<f64> {
        self.variables.get(name).copied()
    }

    /// Clear all variable values.
    pub fn clear_variables(&mut self) {
        self.variables.clear();
    }

    /// Number of variables set.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    // -----------------------------------------------------------------------
    // Solving
    // -----------------------------------------------------------------------

    /// Check all constraints against current variable values.
    ///
    /// This is the main "tick" method. Returns a detailed `SolveResult`.
    pub fn solve(&mut self) -> SolveResult {
        self.current_tick += 1;

        let mut feasible = true;
        let mut total_penalty = 0.0;
        let mut satisfied = 0usize;
        let mut violated = 0usize;
        let mut skipped = 0usize;
        let mut violations = Vec::new();

        for id in &self.eval_order {
            let c = &self.constraints[id];
            match c.evaluate(&self.variables) {
                Some(true) => {
                    satisfied += 1;
                }
                Some(false) => {
                    violated += 1;
                    let actual = self.variables[&c.variable];
                    let penalty = c.penalty(&self.variables);
                    total_penalty += penalty;

                    if c.kind == ConstraintKind::Hard {
                        feasible = false;
                    }

                    violations.push(Violation {
                        constraint_id: c.id.clone(),
                        variable: c.variable.clone(),
                        actual,
                        threshold: c.threshold,
                        op: c.op,
                        kind: c.kind,
                        priority: c.priority,
                        penalty,
                    });
                }
                None => {
                    skipped += 1;
                }
            }
        }

        let total_evaluated = satisfied + violated;
        let satisfaction_ratio = if total_evaluated > 0 {
            satisfied as f64 / total_evaluated as f64
        } else {
            1.0 // no constraints evaluated = vacuously satisfied
        };

        // Update EMA
        let feas_val = if feasible { 1.0 } else { 0.0 };
        if !self.ema_initialized {
            self.ema_feasibility = feas_val;
            self.ema_initialized = true;
        } else {
            self.ema_feasibility = self.config.ema_decay * feas_val
                + (1.0 - self.config.ema_decay) * self.ema_feasibility;
        }

        // Update stats
        self.stats.total_solves += 1;
        if feasible {
            self.stats.feasible_count += 1;
        } else {
            self.stats.infeasible_count += 1;
        }
        self.stats.total_violations += violated as u64;
        self.stats.total_penalty += total_penalty;
        if total_penalty > self.stats.peak_penalty {
            self.stats.peak_penalty = total_penalty;
        }
        self.stats.sum_satisfaction_ratio += satisfaction_ratio;

        // Window record
        let record = WindowRecord {
            feasible,
            total_penalty,
            violation_count: violated,
            satisfaction_ratio,
            tick: self.current_tick,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        violations.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then_with(|| a.constraint_id.cmp(&b.constraint_id))
        });

        SolveResult {
            feasible,
            total_penalty,
            satisfied,
            violated,
            skipped,
            violations,
            satisfaction_ratio,
            ema_feasibility: self.ema_feasibility,
        }
    }

    /// Evaluate a compound constraint by ID.
    pub fn evaluate_compound(&self, id: &str) -> Result<bool> {
        let compound = self
            .compounds
            .get(id)
            .ok_or_else(|| Error::NotFound(format!("compound constraint '{}' not found", id)))?;

        match compound {
            CompoundConstraint::And(ids) => {
                for cid in ids {
                    let c = self.constraints.get(cid).ok_or_else(|| {
                        Error::NotFound(format!("constraint '{}' not found", cid))
                    })?;
                    match c.evaluate(&self.variables) {
                        Some(true) => {}
                        Some(false) => return Ok(false),
                        None => return Ok(false), // missing variable = not satisfied
                    }
                }
                Ok(true)
            }
            CompoundConstraint::Or(ids) => {
                for cid in ids {
                    let c = self.constraints.get(cid).ok_or_else(|| {
                        Error::NotFound(format!("constraint '{}' not found", cid))
                    })?;
                    match c.evaluate(&self.variables) {
                        Some(true) => return Ok(true),
                        _ => {}
                    }
                }
                Ok(false)
            }
        }
    }

    /// Propagate hard constraints by clamping variable values.
    ///
    /// For each hard constraint that is violated, adjust the variable to
    /// the boundary value. Repeats up to `max_propagation_iters` times
    /// or until no changes are made.
    ///
    /// Returns the number of variables that were adjusted.
    pub fn propagate(&mut self) -> usize {
        let mut total_adjustments = 0usize;

        for _iter in 0..self.config.max_propagation_iters {
            let mut adjusted = 0usize;

            // Collect adjustments first to avoid borrow issues
            let adjustments: Vec<(String, f64)> = self
                .eval_order
                .iter()
                .filter_map(|id| {
                    let c = &self.constraints[id];
                    if c.kind != ConstraintKind::Hard {
                        return None;
                    }
                    match c.evaluate(&self.variables) {
                        Some(false) => {
                            // Clamp variable to boundary
                            let boundary = match c.op {
                                CompareOp::Gte | CompareOp::Gt => c.threshold,
                                CompareOp::Lte | CompareOp::Lt => c.threshold,
                                CompareOp::Eq => c.threshold,
                                CompareOp::Ne => return None, // can't clamp for !=
                            };
                            Some((c.variable.clone(), boundary))
                        }
                        _ => None,
                    }
                })
                .collect();

            for (var, val) in adjustments {
                if let Some(current) = self.variables.get(&var) {
                    if (*current - val).abs() > 1e-12 {
                        self.variables.insert(var, val);
                        adjusted += 1;
                    }
                }
            }

            total_adjustments += adjusted;
            if adjusted == 0 {
                break;
            }
        }

        total_adjustments
    }

    /// Check if a specific constraint is satisfied.
    pub fn is_satisfied(&self, id: &str) -> Result<bool> {
        let c = self
            .constraints
            .get(id)
            .ok_or_else(|| Error::NotFound(format!("constraint '{}' not found", id)))?;
        Ok(c.evaluate(&self.variables).unwrap_or(false))
    }

    /// Get violations for a specific variable.
    pub fn violations_for_variable(&self, variable: &str) -> Vec<Violation> {
        let mut violations = Vec::new();
        for id in &self.eval_order {
            let c = &self.constraints[id];
            if c.variable != variable {
                continue;
            }
            if let Some(false) = c.evaluate(&self.variables) {
                let actual = self.variables[&c.variable];
                violations.push(Violation {
                    constraint_id: c.id.clone(),
                    variable: c.variable.clone(),
                    actual,
                    threshold: c.threshold,
                    op: c.op,
                    kind: c.kind,
                    priority: c.priority,
                    penalty: c.penalty(&self.variables),
                });
            }
        }
        violations
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current EMA-smoothed feasibility signal.
    pub fn ema_feasibility(&self) -> f64 {
        self.ema_feasibility
    }

    /// Whether EMA has enough samples.
    pub fn is_warmed_up(&self) -> bool {
        self.stats.total_solves >= self.config.min_samples as u64
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Running statistics.
    pub fn stats(&self) -> &ConstraintSolverStats {
        &self.stats
    }

    /// Confidence metric.
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_solves as f64;
        let min_s = self.config.min_samples as f64;
        (n / (n + min_s)).min(1.0)
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Feasibility rate over the recent window.
    pub fn windowed_feasibility_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let count = self.recent.iter().filter(|r| r.feasible).count();
        count as f64 / self.recent.len() as f64
    }

    /// Mean penalty over the recent window.
    pub fn windowed_mean_penalty(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.total_penalty).sum();
        sum / self.recent.len() as f64
    }

    /// Mean satisfaction ratio over the recent window.
    pub fn windowed_mean_satisfaction(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.satisfaction_ratio).sum();
        sum / self.recent.len() as f64
    }

    /// Mean violation count over the recent window.
    pub fn windowed_mean_violations(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: usize = self.recent.iter().map(|r| r.violation_count).sum();
        sum as f64 / self.recent.len() as f64
    }

    /// Detect whether feasibility is trending upward.
    pub fn is_feasibility_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half =
            self.recent.iter().take(mid).filter(|r| r.feasible).count() as f64 / mid as f64;
        let second_half = self.recent.iter().skip(mid).filter(|r| r.feasible).count() as f64
            / (self.recent.len() - mid) as f64;
        second_half > first_half + 0.05
    }

    /// Detect whether feasibility is trending downward.
    pub fn is_feasibility_decreasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half =
            self.recent.iter().take(mid).filter(|r| r.feasible).count() as f64 / mid as f64;
        let second_half = self.recent.iter().skip(mid).filter(|r| r.feasible).count() as f64
            / (self.recent.len() - mid) as f64;
        second_half < first_half - 0.05
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset state but keep constraints and variables.
    pub fn reset(&mut self) {
        self.ema_feasibility = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = ConstraintSolverStats {
            registered_constraints: self.constraints.len(),
            hard_count: self.stats.hard_count,
            soft_count: self.stats.soft_count,
            ..ConstraintSolverStats::default()
        };
    }

    /// Reset everything including constraints and variables.
    pub fn reset_all(&mut self) {
        self.constraints.clear();
        self.eval_order.clear();
        self.compounds.clear();
        self.variables.clear();
        self.ema_feasibility = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = ConstraintSolverStats::default();
    }

    /// Compatibility shim for the trait-based interface.
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // CompareOp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_gte() {
        assert!(CompareOp::Gte.evaluate(5.0, 5.0));
        assert!(CompareOp::Gte.evaluate(6.0, 5.0));
        assert!(!CompareOp::Gte.evaluate(4.0, 5.0));
    }

    #[test]
    fn test_compare_lte() {
        assert!(CompareOp::Lte.evaluate(5.0, 5.0));
        assert!(CompareOp::Lte.evaluate(4.0, 5.0));
        assert!(!CompareOp::Lte.evaluate(6.0, 5.0));
    }

    #[test]
    fn test_compare_gt() {
        assert!(CompareOp::Gt.evaluate(6.0, 5.0));
        assert!(!CompareOp::Gt.evaluate(5.0, 5.0));
        assert!(!CompareOp::Gt.evaluate(4.0, 5.0));
    }

    #[test]
    fn test_compare_lt() {
        assert!(CompareOp::Lt.evaluate(4.0, 5.0));
        assert!(!CompareOp::Lt.evaluate(5.0, 5.0));
        assert!(!CompareOp::Lt.evaluate(6.0, 5.0));
    }

    #[test]
    fn test_compare_eq() {
        assert!(CompareOp::Eq.evaluate(5.0, 5.0));
        assert!(!CompareOp::Eq.evaluate(5.1, 5.0));
    }

    #[test]
    fn test_compare_ne() {
        assert!(CompareOp::Ne.evaluate(5.1, 5.0));
        assert!(!CompareOp::Ne.evaluate(5.0, 5.0));
    }

    #[test]
    fn test_compare_op_symbol() {
        assert_eq!(CompareOp::Gte.symbol(), ">=");
        assert_eq!(CompareOp::Lte.symbol(), "<=");
        assert_eq!(CompareOp::Gt.symbol(), ">");
        assert_eq!(CompareOp::Lt.symbol(), "<");
        assert_eq!(CompareOp::Eq.symbol(), "==");
        assert_eq!(CompareOp::Ne.symbol(), "!=");
    }

    #[test]
    fn test_compare_op_display() {
        let s = format!("{}", CompareOp::Gte);
        assert_eq!(s, ">=");
    }

    // -----------------------------------------------------------------------
    // AtomicConstraint tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hard_constraint() {
        let c = AtomicConstraint::hard("c1", "drawdown", CompareOp::Lte, 0.05);
        assert_eq!(c.kind, ConstraintKind::Hard);
        assert_eq!(c.variable, "drawdown");
        assert_eq!(c.op, CompareOp::Lte);
    }

    #[test]
    fn test_soft_constraint() {
        let c = AtomicConstraint::soft("c1", "vol", CompareOp::Lte, 0.2, 2.0);
        assert_eq!(c.kind, ConstraintKind::Soft);
        assert!((c.penalty_weight - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_constraint_with_description() {
        let c = AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0)
            .with_description("x must be non-negative");
        assert_eq!(c.description, "x must be non-negative");
    }

    #[test]
    fn test_constraint_with_priority() {
        let c = AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0).with_priority(5);
        assert_eq!(c.priority, 5);
    }

    #[test]
    fn test_constraint_evaluate_satisfied() {
        let c = AtomicConstraint::hard("c1", "x", CompareOp::Gte, 10.0);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 15.0);
        assert_eq!(c.evaluate(&vars), Some(true));
    }

    #[test]
    fn test_constraint_evaluate_violated() {
        let c = AtomicConstraint::hard("c1", "x", CompareOp::Gte, 10.0);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        assert_eq!(c.evaluate(&vars), Some(false));
    }

    #[test]
    fn test_constraint_evaluate_missing_variable() {
        let c = AtomicConstraint::hard("c1", "x", CompareOp::Gte, 10.0);
        let vars = HashMap::new();
        assert_eq!(c.evaluate(&vars), None);
    }

    #[test]
    fn test_constraint_penalty_satisfied() {
        let c = AtomicConstraint::soft("c1", "x", CompareOp::Gte, 10.0, 2.0);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 15.0);
        assert!((c.penalty(&vars) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_constraint_penalty_violated() {
        let c = AtomicConstraint::soft("c1", "x", CompareOp::Gte, 10.0, 2.0);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 7.0);
        // penalty = 2.0 * |7 - 10| = 6.0
        assert!((c.penalty(&vars) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_constraint_penalty_missing_var() {
        let c = AtomicConstraint::soft("c1", "x", CompareOp::Gte, 10.0, 2.0);
        let vars = HashMap::new();
        assert!((c.penalty(&vars) - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Violation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_violation_gap() {
        let v = Violation {
            constraint_id: "c1".into(),
            variable: "x".into(),
            actual: 7.0,
            threshold: 10.0,
            op: CompareOp::Gte,
            kind: ConstraintKind::Hard,
            priority: 0,
            penalty: 0.0,
        };
        assert!((v.gap() - 3.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Solver construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_construction() {
        let solver = ConstraintSolver::new();
        assert_eq!(solver.constraint_count(), 0);
        assert_eq!(solver.current_tick(), 0);
    }

    #[test]
    fn test_invalid_config_ema_zero() {
        let mut cfg = ConstraintSolverConfig::default();
        cfg.ema_decay = 0.0;
        assert!(ConstraintSolver::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_one() {
        let mut cfg = ConstraintSolverConfig::default();
        cfg.ema_decay = 1.0;
        assert!(ConstraintSolver::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_window_zero() {
        let mut cfg = ConstraintSolverConfig::default();
        cfg.window_size = 0;
        assert!(ConstraintSolver::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_constraints_zero() {
        let mut cfg = ConstraintSolverConfig::default();
        cfg.max_constraints = 0;
        assert!(ConstraintSolver::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_propagation_zero() {
        let mut cfg = ConstraintSolverConfig::default();
        cfg.max_propagation_iters = 0;
        assert!(ConstraintSolver::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Constraint registration
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_constraint() {
        let mut solver = ConstraintSolver::new();
        let c = AtomicConstraint::hard("max_dd", "drawdown", CompareOp::Lte, 0.05);
        assert!(solver.add_constraint(c).is_ok());
        assert_eq!(solver.constraint_count(), 1);
        assert_eq!(solver.hard_count(), 1);
        assert_eq!(solver.soft_count(), 0);
    }

    #[test]
    fn test_add_duplicate_constraint() {
        let mut solver = ConstraintSolver::new();
        let c1 = AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0);
        let c2 = AtomicConstraint::hard("c1", "y", CompareOp::Gte, 0.0);
        solver.add_constraint(c1).unwrap();
        assert!(solver.add_constraint(c2).is_err());
    }

    #[test]
    fn test_add_constraint_max_capacity() {
        let mut cfg = ConstraintSolverConfig::default();
        cfg.max_constraints = 2;
        let mut solver = ConstraintSolver::with_config(cfg).unwrap();

        solver
            .add_constraint(AtomicConstraint::hard("a", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("b", "y", CompareOp::Gte, 0.0))
            .unwrap();
        assert!(
            solver
                .add_constraint(AtomicConstraint::hard("c", "z", CompareOp::Gte, 0.0))
                .is_err()
        );
    }

    #[test]
    fn test_remove_constraint() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        assert!(solver.remove_constraint("c1"));
        assert_eq!(solver.constraint_count(), 0);
        assert_eq!(solver.hard_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut solver = ConstraintSolver::new();
        assert!(!solver.remove_constraint("nope"));
    }

    #[test]
    fn test_remove_soft_constraint() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::soft("s1", "x", CompareOp::Lte, 1.0, 1.0))
            .unwrap();
        assert_eq!(solver.soft_count(), 1);
        solver.remove_constraint("s1");
        assert_eq!(solver.soft_count(), 0);
    }

    #[test]
    fn test_constraint_lookup() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 5.0))
            .unwrap();
        let c = solver.constraint("c1").unwrap();
        assert_eq!(c.variable, "x");
        assert!((c.threshold - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_constraint_ids_sorted_by_priority() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(
                AtomicConstraint::hard("low_pri", "x", CompareOp::Gte, 0.0).with_priority(10),
            )
            .unwrap();
        solver
            .add_constraint(
                AtomicConstraint::hard("high_pri", "y", CompareOp::Gte, 0.0).with_priority(1),
            )
            .unwrap();

        let ids = solver.constraint_ids();
        assert_eq!(ids[0], "high_pri");
        assert_eq!(ids[1], "low_pri");
    }

    // -----------------------------------------------------------------------
    // Variable management
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_variable() {
        let mut solver = ConstraintSolver::new();
        solver.set_variable("x", 42.0);
        assert!((solver.variable("x").unwrap() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_variables_batch() {
        let mut solver = ConstraintSolver::new();
        solver.set_variables(&[("x", 1.0), ("y", 2.0)]);
        assert_eq!(solver.variable_count(), 2);
        assert!((solver.variable("x").unwrap() - 1.0).abs() < 1e-10);
        assert!((solver.variable("y").unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_variable_missing() {
        let solver = ConstraintSolver::new();
        assert!(solver.variable("nope").is_none());
    }

    #[test]
    fn test_clear_variables() {
        let mut solver = ConstraintSolver::new();
        solver.set_variable("x", 1.0);
        solver.clear_variables();
        assert_eq!(solver.variable_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Solving
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_no_constraints() {
        let mut solver = ConstraintSolver::new();
        let result = solver.solve();
        assert!(result.feasible);
        assert_eq!(result.satisfied, 0);
        assert_eq!(result.violated, 0);
        assert!((result.satisfaction_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_all_satisfied() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 5.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("c2", "y", CompareOp::Lte, 10.0))
            .unwrap();
        solver.set_variables(&[("x", 7.0), ("y", 8.0)]);

        let result = solver.solve();
        assert!(result.feasible);
        assert_eq!(result.satisfied, 2);
        assert_eq!(result.violated, 0);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_solve_hard_violation() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard(
                "max_dd",
                "drawdown",
                CompareOp::Lte,
                0.05,
            ))
            .unwrap();
        solver.set_variable("drawdown", 0.08);

        let result = solver.solve();
        assert!(!result.feasible);
        assert_eq!(result.violated, 1);
        assert_eq!(result.violations[0].constraint_id, "max_dd");
    }

    #[test]
    fn test_solve_soft_violation_still_feasible() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::soft(
                "vol_target",
                "volatility",
                CompareOp::Lte,
                0.15,
                2.0,
            ))
            .unwrap();
        solver.set_variable("volatility", 0.20);

        let result = solver.solve();
        assert!(result.feasible); // soft constraint doesn't break feasibility
        assert_eq!(result.violated, 1);
        assert!(result.total_penalty > 0.0);
        // penalty = 2.0 * |0.20 - 0.15| = 0.1
        assert!((result.total_penalty - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_solve_mixed_hard_soft() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("h1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::soft("s1", "y", CompareOp::Lte, 1.0, 1.0))
            .unwrap();

        // Hard satisfied, soft violated
        solver.set_variables(&[("x", 5.0), ("y", 2.0)]);
        let result = solver.solve();
        assert!(result.feasible);
        assert_eq!(result.violated, 1);

        // Hard violated
        solver.set_variables(&[("x", -1.0), ("y", 0.5)]);
        let result = solver.solve();
        assert!(!result.feasible);
    }

    #[test]
    fn test_solve_missing_variable_skipped() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        // Don't set variable "x"

        let result = solver.solve();
        assert!(result.feasible); // skipped, not violated
        assert_eq!(result.skipped, 1);
        assert_eq!(result.violated, 0);
    }

    #[test]
    fn test_solve_increments_tick() {
        let mut solver = ConstraintSolver::new();
        solver.solve();
        solver.solve();
        assert_eq!(solver.current_tick(), 2);
    }

    #[test]
    fn test_solve_ema_initializes() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver.set_variable("x", 5.0);

        let result = solver.solve();
        assert!((result.ema_feasibility - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_ema_smoothing() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();

        // First: feasible
        solver.set_variable("x", 5.0);
        solver.solve();

        // Second: infeasible
        solver.set_variable("x", -1.0);
        let r = solver.solve();
        // ema = 0.1 * 0.0 + 0.9 * 1.0 = 0.9
        assert!((r.ema_feasibility - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_solve_satisfaction_ratio() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("c2", "y", CompareOp::Gte, 0.0))
            .unwrap();

        solver.set_variables(&[("x", 5.0), ("y", -1.0)]);
        let result = solver.solve();
        assert!((result.satisfaction_ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_solve_violations_sorted_by_priority() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(
                AtomicConstraint::hard("low", "x", CompareOp::Gte, 10.0).with_priority(5),
            )
            .unwrap();
        solver
            .add_constraint(
                AtomicConstraint::hard("high", "y", CompareOp::Gte, 10.0).with_priority(1),
            )
            .unwrap();
        solver.set_variables(&[("x", 0.0), ("y", 0.0)]);

        let result = solver.solve();
        assert_eq!(result.violations[0].constraint_id, "high");
        assert_eq!(result.violations[1].constraint_id, "low");
    }

    // -----------------------------------------------------------------------
    // Compound constraints
    // -----------------------------------------------------------------------

    #[test]
    fn test_compound_and_satisfied() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("a", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("b", "y", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_compound(
                "both",
                CompoundConstraint::And(vec!["a".into(), "b".into()]),
            )
            .unwrap();

        solver.set_variables(&[("x", 1.0), ("y", 1.0)]);
        assert!(solver.evaluate_compound("both").unwrap());
    }

    #[test]
    fn test_compound_and_violated() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("a", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("b", "y", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_compound(
                "both",
                CompoundConstraint::And(vec!["a".into(), "b".into()]),
            )
            .unwrap();

        solver.set_variables(&[("x", 1.0), ("y", -1.0)]);
        assert!(!solver.evaluate_compound("both").unwrap());
    }

    #[test]
    fn test_compound_or_one_satisfied() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("a", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("b", "y", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_compound(
                "either",
                CompoundConstraint::Or(vec!["a".into(), "b".into()]),
            )
            .unwrap();

        solver.set_variables(&[("x", 1.0), ("y", -1.0)]);
        assert!(solver.evaluate_compound("either").unwrap());
    }

    #[test]
    fn test_compound_or_none_satisfied() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("a", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("b", "y", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_compound(
                "either",
                CompoundConstraint::Or(vec!["a".into(), "b".into()]),
            )
            .unwrap();

        solver.set_variables(&[("x", -1.0), ("y", -1.0)]);
        assert!(!solver.evaluate_compound("either").unwrap());
    }

    #[test]
    fn test_compound_nonexistent() {
        let solver = ConstraintSolver::new();
        assert!(solver.evaluate_compound("nope").is_err());
    }

    #[test]
    fn test_add_compound_empty() {
        let mut solver = ConstraintSolver::new();
        assert!(
            solver
                .add_compound("empty", CompoundConstraint::And(vec![]))
                .is_err()
        );
    }

    #[test]
    fn test_add_compound_missing_ref() {
        let mut solver = ConstraintSolver::new();
        assert!(
            solver
                .add_compound("bad", CompoundConstraint::And(vec!["nonexistent".into()]))
                .is_err()
        );
    }

    #[test]
    fn test_remove_compound() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("a", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_compound("c", CompoundConstraint::And(vec!["a".into()]))
            .unwrap();
        assert!(solver.remove_compound("c"));
        assert!(!solver.remove_compound("c"));
    }

    // -----------------------------------------------------------------------
    // Propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagate_clamps_variable() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard(
                "min_balance",
                "balance",
                CompareOp::Gte,
                1000.0,
            ))
            .unwrap();
        solver.set_variable("balance", 500.0);

        let adjustments = solver.propagate();
        assert_eq!(adjustments, 1);
        assert!((solver.variable("balance").unwrap() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_no_change_when_satisfied() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard(
                "min_balance",
                "balance",
                CompareOp::Gte,
                1000.0,
            ))
            .unwrap();
        solver.set_variable("balance", 2000.0);

        let adjustments = solver.propagate();
        assert_eq!(adjustments, 0);
        assert!((solver.variable("balance").unwrap() - 2000.0).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_only_hard() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::soft(
                "soft_limit",
                "x",
                CompareOp::Lte,
                10.0,
                1.0,
            ))
            .unwrap();
        solver.set_variable("x", 20.0);

        let adjustments = solver.propagate();
        assert_eq!(adjustments, 0); // soft constraints are not propagated
        assert!((solver.variable("x").unwrap() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_lte() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard(
                "max_risk",
                "risk",
                CompareOp::Lte,
                0.05,
            ))
            .unwrap();
        solver.set_variable("risk", 0.10);

        solver.propagate();
        assert!((solver.variable("risk").unwrap() - 0.05).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // is_satisfied
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_satisfied() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver.set_variable("x", 5.0);
        assert!(solver.is_satisfied("c1").unwrap());

        solver.set_variable("x", -1.0);
        assert!(!solver.is_satisfied("c1").unwrap());
    }

    #[test]
    fn test_is_satisfied_nonexistent() {
        let solver = ConstraintSolver::new();
        assert!(solver.is_satisfied("nope").is_err());
    }

    #[test]
    fn test_is_satisfied_missing_var() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        assert!(!solver.is_satisfied("c1").unwrap());
    }

    // -----------------------------------------------------------------------
    // violations_for_variable
    // -----------------------------------------------------------------------

    #[test]
    fn test_violations_for_variable() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 10.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("c2", "x", CompareOp::Lte, 20.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("c3", "y", CompareOp::Gte, 0.0))
            .unwrap();

        solver.set_variables(&[("x", 5.0), ("y", -1.0)]);

        let violations = solver.violations_for_variable("x");
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].constraint_id, "c1");
    }

    #[test]
    fn test_violations_for_variable_empty() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver.set_variable("x", 5.0);

        let violations = solver.violations_for_variable("x");
        assert!(violations.is_empty());
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let solver = ConstraintSolver::new();
        let s = solver.stats();
        assert_eq!(s.total_solves, 0);
        assert!((s.feasibility_rate() - 0.0).abs() < 1e-10);
        assert!((s.mean_satisfaction_ratio() - 0.0).abs() < 1e-10);
        assert!((s.mean_penalty() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_after_solves() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();

        solver.set_variable("x", 5.0);
        solver.solve();

        solver.set_variable("x", -1.0);
        solver.solve();

        let s = solver.stats();
        assert_eq!(s.total_solves, 2);
        assert_eq!(s.feasible_count, 1);
        assert_eq!(s.infeasible_count, 1);
        assert!((s.feasibility_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_peak_penalty() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::soft("s1", "x", CompareOp::Lte, 1.0, 1.0))
            .unwrap();

        solver.set_variable("x", 3.0);
        solver.solve(); // penalty = 2.0

        solver.set_variable("x", 5.0);
        solver.solve(); // penalty = 4.0

        assert!((solver.stats().peak_penalty - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_mean_violations() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();

        solver.set_variable("x", -1.0);
        solver.solve(); // 1 violation

        solver.set_variable("x", 5.0);
        solver.solve(); // 0 violations

        assert!((solver.stats().mean_violations() - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    #[test]
    fn test_windowed_feasibility_rate() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();

        solver.set_variable("x", 5.0);
        solver.solve(); // feasible

        solver.set_variable("x", -1.0);
        solver.solve(); // infeasible

        assert!((solver.windowed_feasibility_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_feasibility_rate_empty() {
        let solver = ConstraintSolver::new();
        assert!((solver.windowed_feasibility_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_penalty() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::soft("s1", "x", CompareOp::Lte, 1.0, 1.0))
            .unwrap();

        solver.set_variable("x", 3.0);
        solver.solve(); // penalty = 2.0

        solver.set_variable("x", 1.0);
        solver.solve(); // penalty = 0.0

        assert!((solver.windowed_mean_penalty() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_penalty_empty() {
        let solver = ConstraintSolver::new();
        assert!((solver.windowed_mean_penalty() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_satisfaction() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("c2", "y", CompareOp::Gte, 0.0))
            .unwrap();

        // 2/2 satisfied
        solver.set_variables(&[("x", 1.0), ("y", 1.0)]);
        solver.solve();

        // 1/2 satisfied
        solver.set_variables(&[("x", 1.0), ("y", -1.0)]);
        solver.solve();

        // mean = (1.0 + 0.5) / 2 = 0.75
        assert!((solver.windowed_mean_satisfaction() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_satisfaction_empty() {
        let solver = ConstraintSolver::new();
        assert!((solver.windowed_mean_satisfaction() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_violations() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();

        solver.set_variable("x", -1.0);
        solver.solve();

        solver.set_variable("x", 5.0);
        solver.solve();

        assert!((solver.windowed_mean_violations() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_violations_empty() {
        let solver = ConstraintSolver::new();
        assert!((solver.windowed_mean_violations() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_eviction() {
        let mut cfg = ConstraintSolverConfig::default();
        cfg.window_size = 3;
        let mut solver = ConstraintSolver::with_config(cfg).unwrap();

        for _ in 0..5 {
            solver.solve();
        }

        assert!(solver.recent.len() <= 3);
    }

    #[test]
    fn test_is_feasibility_increasing() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();

        // First half: infeasible
        for _ in 0..5 {
            solver.set_variable("x", -1.0);
            solver.solve();
        }
        // Second half: feasible
        for _ in 0..5 {
            solver.set_variable("x", 5.0);
            solver.solve();
        }

        assert!(solver.is_feasibility_increasing());
    }

    #[test]
    fn test_is_feasibility_decreasing() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();

        // First half: feasible
        for _ in 0..5 {
            solver.set_variable("x", 5.0);
            solver.solve();
        }
        // Second half: infeasible
        for _ in 0..5 {
            solver.set_variable("x", -1.0);
            solver.solve();
        }

        assert!(solver.is_feasibility_decreasing());
    }

    #[test]
    fn test_trend_insufficient_data() {
        let mut solver = ConstraintSolver::new();
        solver.solve();
        assert!(!solver.is_feasibility_increasing());
        assert!(!solver.is_feasibility_decreasing());
    }

    // -----------------------------------------------------------------------
    // Warmup & confidence
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_warmed_up() {
        let mut solver = ConstraintSolver::new();
        for _ in 0..4 {
            solver.solve();
        }
        assert!(!solver.is_warmed_up());
        solver.solve();
        assert!(solver.is_warmed_up());
    }

    #[test]
    fn test_confidence_increases() {
        let mut solver = ConstraintSolver::new();
        let c0 = solver.confidence();
        solver.solve();
        let c1 = solver.confidence();
        for _ in 0..10 {
            solver.solve();
        }
        let c11 = solver.confidence();

        assert!(c1 > c0);
        assert!(c11 > c1);
        assert!(c11 <= 1.0);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver.set_variable("x", 5.0);
        solver.solve();
        solver.solve();

        solver.reset();

        assert_eq!(solver.current_tick(), 0);
        assert_eq!(solver.stats().total_solves, 0);
        assert_eq!(solver.constraint_count(), 1); // kept
        assert!(solver.variable("x").is_some()); // kept
        assert_eq!(solver.stats().hard_count, 1); // kept
    }

    #[test]
    fn test_reset_all() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver.set_variable("x", 5.0);
        solver.solve();

        solver.reset_all();

        assert_eq!(solver.current_tick(), 0);
        assert_eq!(solver.stats().total_solves, 0);
        assert_eq!(solver.constraint_count(), 0);
        assert_eq!(solver.variable_count(), 0);
        assert_eq!(solver.stats().hard_count, 0);
        assert_eq!(solver.stats().soft_count, 0);
    }

    // -----------------------------------------------------------------------
    // Process compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_process() {
        let solver = ConstraintSolver::new();
        assert!(solver.process().is_ok());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_with_eq_constraint() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("exact", "x", CompareOp::Eq, 42.0))
            .unwrap();

        solver.set_variable("x", 42.0);
        assert!(solver.solve().feasible);

        solver.set_variable("x", 42.1);
        assert!(!solver.solve().feasible);
    }

    #[test]
    fn test_solve_with_ne_constraint() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("not_zero", "x", CompareOp::Ne, 0.0))
            .unwrap();

        solver.set_variable("x", 1.0);
        assert!(solver.solve().feasible);

        solver.set_variable("x", 0.0);
        assert!(!solver.solve().feasible);
    }

    #[test]
    fn test_multiple_constraints_same_variable() {
        let mut solver = ConstraintSolver::new();
        // x must be in [10, 20]
        solver
            .add_constraint(AtomicConstraint::hard("min", "x", CompareOp::Gte, 10.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::hard("max", "x", CompareOp::Lte, 20.0))
            .unwrap();

        solver.set_variable("x", 15.0);
        assert!(solver.solve().feasible);

        solver.set_variable("x", 5.0);
        let r = solver.solve();
        assert!(!r.feasible);
        assert_eq!(r.violated, 1);

        solver.set_variable("x", 25.0);
        let r = solver.solve();
        assert!(!r.feasible);
        assert_eq!(r.violated, 1);
    }

    #[test]
    fn test_penalty_accumulates_across_constraints() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::soft("s1", "x", CompareOp::Lte, 1.0, 1.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::soft("s2", "y", CompareOp::Lte, 1.0, 2.0))
            .unwrap();

        solver.set_variables(&[("x", 3.0), ("y", 3.0)]);
        let r = solver.solve();
        // penalty = 1.0 * 2.0 + 2.0 * 2.0 = 6.0
        assert!((r.total_penalty - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_registered_constraints_updated() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("a", "x", CompareOp::Gte, 0.0))
            .unwrap();
        assert_eq!(solver.stats().registered_constraints, 1);

        solver
            .add_constraint(AtomicConstraint::soft("b", "y", CompareOp::Lte, 1.0, 1.0))
            .unwrap();
        assert_eq!(solver.stats().registered_constraints, 2);

        solver.remove_constraint("a");
        assert_eq!(solver.stats().registered_constraints, 1);
    }

    #[test]
    fn test_solve_result_fields_consistency() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("c1", "x", CompareOp::Gte, 0.0))
            .unwrap();
        solver
            .add_constraint(AtomicConstraint::soft("c2", "y", CompareOp::Lte, 1.0, 1.0))
            .unwrap();

        solver.set_variables(&[("x", 5.0), ("y", 2.0)]);
        let r = solver.solve();

        assert_eq!(r.satisfied + r.violated + r.skipped, 2);
        assert!(r.feasible); // hard is satisfied
        assert_eq!(r.violated, 1); // soft violated
        assert!((r.total_penalty - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_ne_does_not_clamp() {
        let mut solver = ConstraintSolver::new();
        solver
            .add_constraint(AtomicConstraint::hard("not_zero", "x", CompareOp::Ne, 0.0))
            .unwrap();
        solver.set_variable("x", 0.0);

        let adjustments = solver.propagate();
        assert_eq!(adjustments, 0); // Ne cannot be clamped
        assert!((solver.variable("x").unwrap() - 0.0).abs() < 1e-10);
    }
}
