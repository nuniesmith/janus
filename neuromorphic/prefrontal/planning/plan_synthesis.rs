//! Plan Synthesis — Synthesise Executable Action Plans
//!
//! Part of the Prefrontal region
//! Component: planning
//!
//! Combines goals, constraints, and available actions into coherent
//! executable plans. Each plan consists of ordered steps with dependency
//! relationships, resource requirements, and expected outcomes.
//!
//! ## Features
//!
//! - **Step-based plans**: Plans are sequences of named steps, each with
//!   prerequisites, resource costs, expected duration, and success
//!   probability.
//! - **Dependency DAG**: Steps declare dependencies on other steps.
//!   The synthesiser validates that the dependency graph is acyclic
//!   and produces a valid topological execution order.
//! - **Resource allocation**: Steps consume named resources. The engine
//!   checks that total resource usage does not exceed budgets and
//!   identifies resource conflicts.
//! - **Plan scoring**: Plans are scored by expected value (Σ step_value
//!   × success_prob), total cost, risk-adjusted return, and feasibility.
//! - **Multi-plan ranking**: Generate and compare alternative plans,
//!   ranking them by composite score.
//! - **Plan validation**: Cycle detection, missing dependencies,
//!   resource overflows, orphaned steps.
//! - **Plan lifecycle**: Plans move through Draft → Validated →
//!   Executing → Completed / Failed / Cancelled states.
//! - **Running statistics**: Total plans created, success rate, mean
//!   score, resource utilisation.

use crate::common::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the plan synthesis engine.
#[derive(Debug, Clone)]
pub struct PlanSynthesisConfig {
    /// Maximum number of plans that can be registered.
    pub max_plans: usize,
    /// Maximum number of steps per plan.
    pub max_steps_per_plan: usize,
    /// Maximum number of named resources tracked.
    pub max_resources: usize,
}

impl Default for PlanSynthesisConfig {
    fn default() -> Self {
        Self {
            max_plans: 128,
            max_steps_per_plan: 64,
            max_resources: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Plan state
// ---------------------------------------------------------------------------

/// Lifecycle state of a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanState {
    /// Plan is being constructed.
    Draft,
    /// Plan has been validated and is ready for execution.
    Validated,
    /// Plan is currently executing.
    Executing,
    /// Plan completed successfully.
    Completed,
    /// Plan failed during execution.
    Failed,
    /// Plan was cancelled.
    Cancelled,
}

impl PlanState {
    /// Whether the plan is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            PlanState::Completed | PlanState::Failed | PlanState::Cancelled
        )
    }

    /// Whether the plan is still alive (not terminal).
    pub fn is_alive(self) -> bool {
        !self.is_terminal()
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            PlanState::Draft => "Draft",
            PlanState::Validated => "Validated",
            PlanState::Executing => "Executing",
            PlanState::Completed => "Completed",
            PlanState::Failed => "Failed",
            PlanState::Cancelled => "Cancelled",
        }
    }
}

impl std::fmt::Display for PlanState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Step state
// ---------------------------------------------------------------------------

/// Lifecycle state of a single step within a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepState {
    /// Waiting for dependencies to complete.
    Pending,
    /// Ready to execute (all dependencies met).
    Ready,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Done,
    /// Failed.
    Failed,
    /// Skipped (e.g. due to upstream failure).
    Skipped,
}

impl StepState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            StepState::Done | StepState::Failed | StepState::Skipped
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            StepState::Pending => "Pending",
            StepState::Ready => "Ready",
            StepState::Running => "Running",
            StepState::Done => "Done",
            StepState::Failed => "Failed",
            StepState::Skipped => "Skipped",
        }
    }
}

impl std::fmt::Display for StepState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Plan step
// ---------------------------------------------------------------------------

/// A single step within a plan.
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Unique step identifier (within the plan).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    pub description: String,
    /// IDs of steps that must complete before this step can start.
    pub dependencies: Vec<String>,
    /// Resource costs: resource_name → amount consumed.
    pub resource_costs: HashMap<String, f64>,
    /// Expected duration in ticks.
    pub expected_duration: u64,
    /// Estimated success probability [0, 1].
    pub success_prob: f64,
    /// Expected value if this step succeeds.
    pub expected_value: f64,
    /// Current state.
    pub state: StepState,
    /// Priority (0 = highest) — used for ordering when multiple steps
    /// are ready simultaneously.
    pub priority: u32,
}

impl PlanStep {
    /// Create a new step with default values.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            dependencies: Vec::new(),
            resource_costs: HashMap::new(),
            expected_duration: 1,
            success_prob: 1.0,
            expected_value: 0.0,
            state: StepState::Pending,
            priority: 0,
        }
    }

    /// Builder: add a dependency.
    pub fn with_dependency(mut self, dep_id: impl Into<String>) -> Self {
        self.dependencies.push(dep_id.into());
        self
    }

    /// Builder: add a resource cost.
    pub fn with_resource(mut self, name: impl Into<String>, cost: f64) -> Self {
        self.resource_costs.insert(name.into(), cost);
        self
    }

    /// Builder: set expected duration.
    pub fn with_duration(mut self, ticks: u64) -> Self {
        self.expected_duration = ticks;
        self
    }

    /// Builder: set success probability.
    pub fn with_success_prob(mut self, prob: f64) -> Self {
        self.success_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Builder: set expected value.
    pub fn with_value(mut self, value: f64) -> Self {
        self.expected_value = value;
        self
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Whether all dependencies are in a terminal-success state.
    fn dependencies_met(&self, steps: &HashMap<String, PlanStep>) -> bool {
        self.dependencies.iter().all(|dep_id| {
            steps
                .get(dep_id)
                .map_or(false, |s| s.state == StepState::Done)
        })
    }

    /// Whether any dependency has failed.
    #[allow(dead_code)]
    fn any_dependency_failed(&self, steps: &HashMap<String, PlanStep>) -> bool {
        self.dependencies.iter().any(|dep_id| {
            steps.get(dep_id).map_or(false, |s| {
                s.state == StepState::Failed || s.state == StepState::Skipped
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// An executable plan consisting of ordered steps.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Unique plan identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    pub description: String,
    /// Goal ID this plan is targeting (if any).
    pub goal_id: Option<String>,
    /// Steps keyed by step ID.
    pub steps: HashMap<String, PlanStep>,
    /// Step insertion order.
    step_order: Vec<String>,
    /// Current state.
    pub state: PlanState,
    /// Composite score (computed during validation).
    pub score: f64,
    /// Tick when the plan was created.
    pub created_tick: u64,
    /// Tick when the plan entered a terminal state.
    pub completed_tick: Option<u64>,
}

impl Plan {
    fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            goal_id: None,
            steps: HashMap::new(),
            step_order: Vec::new(),
            state: PlanState::Draft,
            score: 0.0,
            created_tick: 0,
            completed_tick: None,
        }
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Step IDs in insertion order.
    pub fn step_ids(&self) -> &[String] {
        &self.step_order
    }

    /// Get a step by ID.
    pub fn step(&self, id: &str) -> Option<&PlanStep> {
        self.steps.get(id)
    }

    /// Total expected value: Σ step_value × success_prob.
    pub fn expected_value(&self) -> f64 {
        self.steps
            .values()
            .map(|s| s.expected_value * s.success_prob)
            .sum()
    }

    /// Total resource cost: sum of all step resource costs per resource.
    pub fn total_resource_costs(&self) -> HashMap<String, f64> {
        let mut totals: HashMap<String, f64> = HashMap::new();
        for step in self.steps.values() {
            for (res, cost) in &step.resource_costs {
                *totals.entry(res.clone()).or_insert(0.0) += cost;
            }
        }
        totals
    }

    /// Total expected duration (sum of all step durations — worst case
    /// if sequential; actual depends on parallelism).
    pub fn total_duration(&self) -> u64 {
        self.steps.values().map(|s| s.expected_duration).sum()
    }

    /// Critical path duration (longest path through the dependency DAG).
    pub fn critical_path_duration(&self) -> u64 {
        let mut memo: HashMap<String, u64> = HashMap::new();
        let mut max_dur = 0u64;
        for id in &self.step_order {
            let dur = self.critical_path_from(id, &mut memo);
            if dur > max_dur {
                max_dur = dur;
            }
        }
        max_dur
    }

    fn critical_path_from(&self, id: &str, memo: &mut HashMap<String, u64>) -> u64 {
        if let Some(&cached) = memo.get(id) {
            return cached;
        }
        let step = match self.steps.get(id) {
            Some(s) => s,
            None => return 0,
        };
        let max_dep = step
            .dependencies
            .iter()
            .map(|dep| self.critical_path_from(dep, memo))
            .max()
            .unwrap_or(0);
        let total = max_dep + step.expected_duration;
        memo.insert(id.to_string(), total);
        total
    }

    /// Mean success probability across all steps.
    pub fn mean_success_prob(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.steps.values().map(|s| s.success_prob).sum();
        sum / self.steps.len() as f64
    }

    /// Compound success probability (product of all step probs).
    pub fn compound_success_prob(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.steps.values().map(|s| s.success_prob).product::<f64>()
    }

    /// Progress: fraction of steps in terminal-success state.
    pub fn progress(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let done = self
            .steps
            .values()
            .filter(|s| s.state == StepState::Done)
            .count();
        done as f64 / self.steps.len() as f64
    }

    /// IDs of steps that are currently ready (dependencies met, not
    /// yet started).
    pub fn ready_steps(&self) -> Vec<String> {
        let mut ready: Vec<_> = self
            .step_order
            .iter()
            .filter(|id| {
                let step = &self.steps[id.as_str()];
                step.state == StepState::Ready
                    || (step.state == StepState::Pending && step.dependencies_met(&self.steps))
            })
            .cloned()
            .collect();
        // Sort by priority then ID for determinism.
        ready.sort_by(|a, b| {
            let sa = &self.steps[a.as_str()];
            let sb = &self.steps[b.as_str()];
            sa.priority.cmp(&sb.priority).then_with(|| a.cmp(b))
        });
        ready
    }
}

// ---------------------------------------------------------------------------
// Validation result
// ---------------------------------------------------------------------------

/// Result of validating a plan.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the plan is valid (no errors).
    pub valid: bool,
    /// Validation errors.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
    /// Topological execution order (if valid).
    pub execution_order: Vec<String>,
    /// Computed plan score.
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Plan score
// ---------------------------------------------------------------------------

/// Breakdown of a plan's composite score.
#[derive(Debug, Clone)]
pub struct PlanScoreBreakdown {
    /// Expected value: Σ step_value × success_prob.
    pub expected_value: f64,
    /// Compound success probability.
    pub compound_success_prob: f64,
    /// Risk-adjusted return: expected_value × compound_success_prob.
    pub risk_adjusted_return: f64,
    /// Resource efficiency: expected_value / total_resource_cost.
    pub resource_efficiency: f64,
    /// Time efficiency: expected_value / critical_path_duration.
    pub time_efficiency: f64,
    /// Composite score (weighted combination).
    pub composite: f64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the plan synthesis engine.
#[derive(Debug, Clone)]
pub struct PlanSynthesisStats {
    /// Total plans created.
    pub total_plans: u64,
    /// Plans in each state.
    pub state_counts: HashMap<PlanState, usize>,
    /// Total plans completed successfully.
    pub completed_count: u64,
    /// Total plans failed.
    pub failed_count: u64,
    /// Total plans cancelled.
    pub cancelled_count: u64,
    /// Sum of scores (for mean).
    pub sum_scores: f64,
    /// Number of scored plans.
    pub scored_count: u64,
    /// Peak plan score observed.
    pub peak_score: f64,
    /// Current active plans.
    pub active_plans: usize,
}

impl Default for PlanSynthesisStats {
    fn default() -> Self {
        Self {
            total_plans: 0,
            state_counts: HashMap::new(),
            completed_count: 0,
            failed_count: 0,
            cancelled_count: 0,
            sum_scores: 0.0,
            scored_count: 0,
            peak_score: 0.0,
            active_plans: 0,
        }
    }
}

impl PlanSynthesisStats {
    /// Success rate: completed / (completed + failed).
    pub fn success_rate(&self) -> f64 {
        let total = self.completed_count + self.failed_count;
        if total == 0 {
            return 0.0;
        }
        self.completed_count as f64 / total as f64
    }

    /// Mean plan score.
    pub fn mean_score(&self) -> f64 {
        if self.scored_count == 0 {
            return 0.0;
        }
        self.sum_scores / self.scored_count as f64
    }
}

// ---------------------------------------------------------------------------
// PlanSynthesis engine
// ---------------------------------------------------------------------------

/// Plan synthesis engine.
///
/// Create plans with steps and dependencies, validate them, compute
/// scores, and manage execution lifecycle.
pub struct PlanSynthesis {
    config: PlanSynthesisConfig,

    /// Registered plans keyed by ID.
    plans: HashMap<String, Plan>,

    /// Resource budgets: resource_name → maximum available.
    resource_budgets: HashMap<String, f64>,

    /// Current tick.
    current_tick: u64,

    /// Running statistics.
    stats: PlanSynthesisStats,
}

impl Default for PlanSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanSynthesis {
    /// Create a new engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(PlanSynthesisConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: PlanSynthesisConfig) -> Result<Self> {
        if config.max_plans == 0 {
            return Err(Error::InvalidInput("max_plans must be > 0".into()));
        }
        if config.max_steps_per_plan == 0 {
            return Err(Error::InvalidInput("max_steps_per_plan must be > 0".into()));
        }
        Ok(Self {
            config,
            plans: HashMap::new(),
            resource_budgets: HashMap::new(),
            current_tick: 0,
            stats: PlanSynthesisStats::default(),
        })
    }

    // -----------------------------------------------------------------------
    // Resource budgets
    // -----------------------------------------------------------------------

    /// Set a resource budget.
    pub fn set_resource_budget(&mut self, name: impl Into<String>, budget: f64) -> Result<()> {
        let name = name.into();
        if self.resource_budgets.len() >= self.config.max_resources
            && !self.resource_budgets.contains_key(&name)
        {
            return Err(Error::ResourceExhausted(format!(
                "maximum resources ({}) reached",
                self.config.max_resources
            )));
        }
        self.resource_budgets.insert(name, budget);
        Ok(())
    }

    /// Get a resource budget.
    pub fn resource_budget(&self, name: &str) -> Option<f64> {
        self.resource_budgets.get(name).copied()
    }

    /// Clear all resource budgets.
    pub fn clear_resource_budgets(&mut self) {
        self.resource_budgets.clear();
    }

    // -----------------------------------------------------------------------
    // Plan management
    // -----------------------------------------------------------------------

    /// Create a new plan in Draft state.
    pub fn create_plan(&mut self, id: impl Into<String>, name: impl Into<String>) -> Result<()> {
        let id = id.into();
        if self.plans.contains_key(&id) {
            return Err(Error::InvalidInput(format!("plan '{}' already exists", id)));
        }
        if self.plans.len() >= self.config.max_plans {
            return Err(Error::ResourceExhausted(format!(
                "maximum plans ({}) reached",
                self.config.max_plans
            )));
        }

        let mut plan = Plan::new(&id, name);
        plan.created_tick = self.current_tick;
        self.plans.insert(id, plan);
        self.stats.total_plans += 1;
        self.recount_states();
        Ok(())
    }

    /// Remove a plan by ID.
    pub fn remove_plan(&mut self, id: &str) -> bool {
        let removed = self.plans.remove(id).is_some();
        if removed {
            self.recount_states();
        }
        removed
    }

    /// Get a plan by ID.
    pub fn plan(&self, id: &str) -> Option<&Plan> {
        self.plans.get(id)
    }

    /// Get a mutable reference to a plan.
    pub fn plan_mut(&mut self, id: &str) -> Option<&mut Plan> {
        self.plans.get_mut(id)
    }

    /// Number of plans.
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// All plan IDs.
    pub fn plan_ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self.plans.keys().cloned().collect();
        ids.sort();
        ids
    }

    // -----------------------------------------------------------------------
    // Step management
    // -----------------------------------------------------------------------

    /// Add a step to a plan.
    pub fn add_step(&mut self, plan_id: &str, step: PlanStep) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;

        if plan.state != PlanState::Draft {
            return Err(Error::InvalidState(format!(
                "cannot add steps to plan '{}' in state {}",
                plan_id, plan.state
            )));
        }
        if plan.steps.contains_key(&step.id) {
            return Err(Error::InvalidInput(format!(
                "step '{}' already exists in plan '{}'",
                step.id, plan_id
            )));
        }
        if plan.steps.len() >= self.config.max_steps_per_plan {
            return Err(Error::ResourceExhausted(format!(
                "maximum steps ({}) reached for plan '{}'",
                self.config.max_steps_per_plan, plan_id
            )));
        }

        plan.step_order.push(step.id.clone());
        plan.steps.insert(step.id.clone(), step);
        Ok(())
    }

    /// Remove a step from a plan.
    pub fn remove_step(&mut self, plan_id: &str, step_id: &str) -> Result<bool> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;

        if plan.state != PlanState::Draft {
            return Err(Error::InvalidState(format!(
                "cannot remove steps from plan '{}' in state {}",
                plan_id, plan.state
            )));
        }

        let removed = plan.steps.remove(step_id).is_some();
        if removed {
            plan.step_order.retain(|s| s != step_id);
            // Also remove from other steps' dependencies
            for step in plan.steps.values_mut() {
                step.dependencies.retain(|d| d != step_id);
            }
        }
        Ok(removed)
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /// Validate a plan: check for cycles, missing dependencies,
    /// resource overflows, and compute score.
    pub fn validate(&mut self, plan_id: &str) -> Result<ValidationResult> {
        let plan = self
            .plans
            .get(plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check empty plan
        if plan.steps.is_empty() {
            warnings.push("plan has no steps".into());
        }

        // Check for missing dependencies
        for step in plan.steps.values() {
            for dep in &step.dependencies {
                if !plan.steps.contains_key(dep) {
                    errors.push(format!(
                        "step '{}' depends on '{}' which does not exist",
                        step.id, dep
                    ));
                }
            }
        }

        // Check for self-dependencies
        for step in plan.steps.values() {
            if step.dependencies.contains(&step.id) {
                errors.push(format!("step '{}' depends on itself", step.id));
            }
        }

        // Check for cycles via topological sort
        let topo_result = self.topological_sort(plan);
        let execution_order = match topo_result {
            Ok(order) => order,
            Err(cycle_msg) => {
                errors.push(cycle_msg);
                Vec::new()
            }
        };

        // Check resource budgets
        let total_costs = plan.total_resource_costs();
        for (res, cost) in &total_costs {
            if let Some(&budget) = self.resource_budgets.get(res) {
                if *cost > budget {
                    errors.push(format!(
                        "resource '{}' over budget: needs {:.2}, budget {:.2}",
                        res, cost, budget
                    ));
                }
            }
        }

        // Check success probabilities
        for step in plan.steps.values() {
            if step.success_prob <= 0.0 {
                warnings.push(format!(
                    "step '{}' has zero or negative success probability",
                    step.id
                ));
            }
        }

        // Check for orphan steps (no dependencies and not depended on)
        if plan.steps.len() > 1 {
            let depended_on: HashSet<String> = plan
                .steps
                .values()
                .flat_map(|s| s.dependencies.iter().cloned())
                .collect();
            let has_deps: HashSet<String> = plan
                .steps
                .values()
                .filter(|s| !s.dependencies.is_empty())
                .map(|s| s.id.clone())
                .collect();
            for step in plan.steps.values() {
                if !depended_on.contains(&step.id) && !has_deps.contains(&step.id) {
                    warnings.push(format!(
                        "step '{}' is an orphan (no dependencies and not depended on)",
                        step.id
                    ));
                }
            }
        }

        let valid = errors.is_empty();
        let score = if valid {
            self.compute_score(plan).composite
        } else {
            0.0
        };

        // Update plan state if valid
        if valid {
            if let Some(p) = self.plans.get_mut(plan_id) {
                p.state = PlanState::Validated;
                p.score = score;
            }
            self.stats.scored_count += 1;
            self.stats.sum_scores += score;
            if score > self.stats.peak_score {
                self.stats.peak_score = score;
            }
            self.recount_states();
        }

        Ok(ValidationResult {
            valid,
            errors,
            warnings,
            execution_order,
            score,
        })
    }

    /// Topological sort of steps (Kahn's algorithm).
    fn topological_sort(&self, plan: &Plan) -> std::result::Result<Vec<String>, String> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();

        for step in plan.steps.values() {
            in_degree.entry(step.id.clone()).or_insert(0);
            adjacency.entry(step.id.clone()).or_insert_with(Vec::new);
            for dep in &step.dependencies {
                if plan.steps.contains_key(dep) {
                    adjacency
                        .entry(dep.clone())
                        .or_insert_with(Vec::new)
                        .push(step.id.clone());
                    *in_degree.entry(step.id.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(id, _)| id.clone())
            .collect();

        // Sort queue for deterministic output.
        let mut q_vec: Vec<_> = queue.drain(..).collect();
        q_vec.sort();
        queue.extend(q_vec);

        let mut order = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node.clone());
            if let Some(neighbours) = adjacency.get(&node) {
                let mut next_ready = Vec::new();
                for nbr in neighbours {
                    if let Some(deg) = in_degree.get_mut(nbr) {
                        *deg -= 1;
                        if *deg == 0 {
                            next_ready.push(nbr.clone());
                        }
                    }
                }
                next_ready.sort();
                queue.extend(next_ready);
            }
        }

        if order.len() != plan.steps.len() {
            Err("dependency cycle detected".into())
        } else {
            Ok(order)
        }
    }

    // -----------------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------------

    /// Compute a detailed score breakdown for a plan.
    pub fn compute_score(&self, plan: &Plan) -> PlanScoreBreakdown {
        let expected_value = plan.expected_value();
        let compound_success_prob = plan.compound_success_prob();
        let risk_adjusted_return = expected_value * compound_success_prob;

        let total_cost: f64 = plan.total_resource_costs().values().sum();
        let resource_efficiency = if total_cost > 0.0 {
            expected_value / total_cost
        } else {
            expected_value // free plan has infinite efficiency, capped
        };

        let crit_path = plan.critical_path_duration();
        let time_efficiency = if crit_path > 0 {
            expected_value / crit_path as f64
        } else {
            expected_value
        };

        // Composite: weighted combination
        let composite =
            0.4 * risk_adjusted_return + 0.3 * resource_efficiency + 0.3 * time_efficiency;

        PlanScoreBreakdown {
            expected_value,
            compound_success_prob,
            risk_adjusted_return,
            resource_efficiency,
            time_efficiency,
            composite,
        }
    }

    /// Rank all validated plans by composite score (descending).
    pub fn rank_plans(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<(String, f64)> = self
            .plans
            .values()
            .filter(|p| p.state != PlanState::Draft)
            .map(|p| (p.id.clone(), p.score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    // -----------------------------------------------------------------------
    // Execution lifecycle
    // -----------------------------------------------------------------------

    /// Start executing a validated plan.
    pub fn start_execution(&mut self, plan_id: &str) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;
        if plan.state != PlanState::Validated {
            return Err(Error::InvalidState(format!(
                "plan '{}' is in state {}, expected Validated",
                plan_id, plan.state
            )));
        }
        plan.state = PlanState::Executing;

        // Mark steps with no dependencies as Ready
        let ready_ids: Vec<String> = plan
            .steps
            .values()
            .filter(|s| s.dependencies.is_empty() && s.state == StepState::Pending)
            .map(|s| s.id.clone())
            .collect();
        for id in ready_ids {
            if let Some(step) = plan.steps.get_mut(&id) {
                step.state = StepState::Ready;
            }
        }

        self.recount_states();
        Ok(())
    }

    /// Mark a step as started (Running).
    pub fn start_step(&mut self, plan_id: &str, step_id: &str) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;
        if plan.state != PlanState::Executing {
            return Err(Error::InvalidState(format!(
                "plan '{}' is not executing",
                plan_id
            )));
        }

        let step = plan
            .steps
            .get_mut(step_id)
            .ok_or_else(|| Error::NotFound(format!("step '{}' not found", step_id)))?;
        if step.state != StepState::Ready {
            return Err(Error::InvalidState(format!(
                "step '{}' is in state {}, expected Ready",
                step_id, step.state
            )));
        }
        step.state = StepState::Running;
        Ok(())
    }

    /// Mark a step as completed (Done), and advance dependents.
    pub fn complete_step(&mut self, plan_id: &str, step_id: &str) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;
        if plan.state != PlanState::Executing {
            return Err(Error::InvalidState(format!(
                "plan '{}' is not executing",
                plan_id
            )));
        }

        let step = plan
            .steps
            .get_mut(step_id)
            .ok_or_else(|| Error::NotFound(format!("step '{}' not found", step_id)))?;
        if step.state != StepState::Running {
            return Err(Error::InvalidState(format!(
                "step '{}' is in state {}, expected Running",
                step_id, step.state
            )));
        }
        step.state = StepState::Done;

        // Advance dependents
        let dependents: Vec<String> = plan
            .steps
            .values()
            .filter(|s| s.dependencies.contains(&step_id.to_string()))
            .map(|s| s.id.clone())
            .collect();
        for dep_id in dependents {
            let all_deps_done = {
                let dep = &plan.steps[&dep_id];
                dep.dependencies_met(&plan.steps)
            };
            if all_deps_done {
                if let Some(dep) = plan.steps.get_mut(&dep_id) {
                    if dep.state == StepState::Pending {
                        dep.state = StepState::Ready;
                    }
                }
            }
        }

        // Check if all steps are done → complete plan
        if plan.steps.values().all(|s| s.state == StepState::Done) {
            plan.state = PlanState::Completed;
            plan.completed_tick = Some(self.current_tick);
            self.stats.completed_count += 1;
            self.recount_states();
        }

        Ok(())
    }

    /// Mark a step as failed and skip dependents.
    pub fn fail_step(&mut self, plan_id: &str, step_id: &str) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;
        if plan.state != PlanState::Executing {
            return Err(Error::InvalidState(format!(
                "plan '{}' is not executing",
                plan_id
            )));
        }

        let step = plan
            .steps
            .get_mut(step_id)
            .ok_or_else(|| Error::NotFound(format!("step '{}' not found", step_id)))?;
        if step.state != StepState::Running {
            return Err(Error::InvalidState(format!(
                "step '{}' is in state {}, expected Running",
                step_id, step.state
            )));
        }
        step.state = StepState::Failed;

        // Skip dependents recursively
        let mut to_skip: Vec<String> = Vec::new();
        self.collect_downstream(plan_id, step_id, &mut to_skip);
        for skip_id in &to_skip {
            if let Some(p) = self.plans.get_mut(plan_id) {
                if let Some(s) = p.steps.get_mut(skip_id) {
                    if !s.state.is_terminal() {
                        s.state = StepState::Skipped;
                    }
                }
            }
        }

        // Mark plan as failed
        if let Some(p) = self.plans.get_mut(plan_id) {
            p.state = PlanState::Failed;
            p.completed_tick = Some(self.current_tick);
        }
        self.stats.failed_count += 1;
        self.recount_states();

        Ok(())
    }

    fn collect_downstream(&self, plan_id: &str, step_id: &str, result: &mut Vec<String>) {
        if let Some(plan) = self.plans.get(plan_id) {
            for s in plan.steps.values() {
                if s.dependencies.contains(&step_id.to_string()) && !result.contains(&s.id) {
                    result.push(s.id.clone());
                    self.collect_downstream(plan_id, &s.id, result);
                }
            }
        }
    }

    /// Cancel a plan.
    pub fn cancel_plan(&mut self, plan_id: &str) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;
        if plan.state.is_terminal() {
            return Err(Error::InvalidState(format!(
                "plan '{}' is already in terminal state {}",
                plan_id, plan.state
            )));
        }
        plan.state = PlanState::Cancelled;
        plan.completed_tick = Some(self.current_tick);
        self.stats.cancelled_count += 1;
        self.recount_states();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Tick
    // -----------------------------------------------------------------------

    /// Advance the tick counter.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Running statistics.
    pub fn stats(&self) -> &PlanSynthesisStats {
        &self.stats
    }

    fn recount_states(&mut self) {
        let mut counts: HashMap<PlanState, usize> = HashMap::new();
        let mut active = 0usize;
        for plan in self.plans.values() {
            *counts.entry(plan.state).or_insert(0) += 1;
            if plan.state.is_alive() {
                active += 1;
            }
        }
        self.stats.state_counts = counts;
        self.stats.active_plans = active;
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset everything.
    pub fn reset(&mut self) {
        self.plans.clear();
        self.resource_budgets.clear();
        self.current_tick = 0;
        self.stats = PlanSynthesisStats::default();
    }

    /// Main processing function (compatibility shim).
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
    // PlanState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_state_terminal() {
        assert!(!PlanState::Draft.is_terminal());
        assert!(!PlanState::Validated.is_terminal());
        assert!(!PlanState::Executing.is_terminal());
        assert!(PlanState::Completed.is_terminal());
        assert!(PlanState::Failed.is_terminal());
        assert!(PlanState::Cancelled.is_terminal());
    }

    #[test]
    fn test_plan_state_alive() {
        assert!(PlanState::Draft.is_alive());
        assert!(!PlanState::Completed.is_alive());
    }

    #[test]
    fn test_plan_state_label() {
        assert_eq!(PlanState::Draft.label(), "Draft");
        assert_eq!(PlanState::Validated.label(), "Validated");
    }

    #[test]
    fn test_plan_state_display() {
        assert_eq!(format!("{}", PlanState::Executing), "Executing");
    }

    // -----------------------------------------------------------------------
    // StepState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_state_terminal() {
        assert!(!StepState::Pending.is_terminal());
        assert!(!StepState::Ready.is_terminal());
        assert!(!StepState::Running.is_terminal());
        assert!(StepState::Done.is_terminal());
        assert!(StepState::Failed.is_terminal());
        assert!(StepState::Skipped.is_terminal());
    }

    #[test]
    fn test_step_state_label() {
        assert_eq!(StepState::Pending.label(), "Pending");
        assert_eq!(StepState::Done.label(), "Done");
    }

    #[test]
    fn test_step_state_display() {
        assert_eq!(format!("{}", StepState::Running), "Running");
    }

    // -----------------------------------------------------------------------
    // PlanStep tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_new() {
        let step = PlanStep::new("s1", "Step 1");
        assert_eq!(step.id, "s1");
        assert_eq!(step.name, "Step 1");
        assert_eq!(step.state, StepState::Pending);
        assert!((step.success_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_builders() {
        let step = PlanStep::new("s1", "Step 1")
            .with_dependency("s0")
            .with_resource("capital", 1000.0)
            .with_duration(5)
            .with_success_prob(0.9)
            .with_value(500.0)
            .with_priority(2)
            .with_description("A test step");

        assert_eq!(step.dependencies, vec!["s0"]);
        assert!((step.resource_costs["capital"] - 1000.0).abs() < 1e-10);
        assert_eq!(step.expected_duration, 5);
        assert!((step.success_prob - 0.9).abs() < 1e-10);
        assert!((step.expected_value - 500.0).abs() < 1e-10);
        assert_eq!(step.priority, 2);
        assert_eq!(step.description, "A test step");
    }

    #[test]
    fn test_step_success_prob_clamped() {
        let step = PlanStep::new("s1", "S1").with_success_prob(1.5);
        assert!((step.success_prob - 1.0).abs() < 1e-10);

        let step2 = PlanStep::new("s2", "S2").with_success_prob(-0.5);
        assert!((step2.success_prob - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Plan tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_expected_value() {
        let mut plan = Plan::new("p1", "Plan 1");
        plan.steps.insert(
            "a".into(),
            PlanStep::new("a", "A")
                .with_value(100.0)
                .with_success_prob(0.8),
        );
        plan.steps.insert(
            "b".into(),
            PlanStep::new("b", "B")
                .with_value(200.0)
                .with_success_prob(0.5),
        );
        // EV = 100*0.8 + 200*0.5 = 80 + 100 = 180
        assert!((plan.expected_value() - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_plan_total_resource_costs() {
        let mut plan = Plan::new("p1", "Plan 1");
        plan.steps.insert(
            "a".into(),
            PlanStep::new("a", "A").with_resource("capital", 100.0),
        );
        plan.steps.insert(
            "b".into(),
            PlanStep::new("b", "B")
                .with_resource("capital", 50.0)
                .with_resource("time", 10.0),
        );
        let costs = plan.total_resource_costs();
        assert!((costs["capital"] - 150.0).abs() < 1e-10);
        assert!((costs["time"] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_plan_total_duration() {
        let mut plan = Plan::new("p1", "Plan 1");
        plan.steps
            .insert("a".into(), PlanStep::new("a", "A").with_duration(3));
        plan.steps
            .insert("b".into(), PlanStep::new("b", "B").with_duration(5));
        assert_eq!(plan.total_duration(), 8);
    }

    #[test]
    fn test_plan_critical_path_duration() {
        let mut plan = Plan::new("p1", "Plan 1");
        plan.steps
            .insert("a".into(), PlanStep::new("a", "A").with_duration(3));
        plan.steps.insert(
            "b".into(),
            PlanStep::new("b", "B")
                .with_duration(5)
                .with_dependency("a"),
        );
        plan.steps
            .insert("c".into(), PlanStep::new("c", "C").with_duration(2));
        plan.step_order = vec!["a".into(), "b".into(), "c".into()];

        // Critical path: a(3) → b(5) = 8, or c(2). Max = 8.
        assert_eq!(plan.critical_path_duration(), 8);
    }

    #[test]
    fn test_plan_mean_success_prob() {
        let mut plan = Plan::new("p1", "Plan 1");
        plan.steps
            .insert("a".into(), PlanStep::new("a", "A").with_success_prob(0.8));
        plan.steps
            .insert("b".into(), PlanStep::new("b", "B").with_success_prob(0.6));
        assert!((plan.mean_success_prob() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_plan_mean_success_prob_empty() {
        let plan = Plan::new("p1", "Plan 1");
        assert!((plan.mean_success_prob() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_plan_compound_success_prob() {
        let mut plan = Plan::new("p1", "Plan 1");
        plan.steps
            .insert("a".into(), PlanStep::new("a", "A").with_success_prob(0.8));
        plan.steps
            .insert("b".into(), PlanStep::new("b", "B").with_success_prob(0.5));
        // 0.8 * 0.5 = 0.4
        assert!((plan.compound_success_prob() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_plan_progress() {
        let mut plan = Plan::new("p1", "Plan 1");
        let mut s1 = PlanStep::new("a", "A");
        s1.state = StepState::Done;
        plan.steps.insert("a".into(), s1);

        let s2 = PlanStep::new("b", "B");
        plan.steps.insert("b".into(), s2);

        assert!((plan.progress() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_plan_progress_empty() {
        let plan = Plan::new("p1", "Plan 1");
        assert!((plan.progress() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_plan_ready_steps() {
        let mut plan = Plan::new("p1", "Plan 1");
        plan.steps.insert("a".into(), PlanStep::new("a", "A"));
        plan.steps
            .insert("b".into(), PlanStep::new("b", "B").with_dependency("a"));
        plan.step_order = vec!["a".into(), "b".into()];

        let ready = plan.ready_steps();
        assert_eq!(ready, vec!["a"]); // b depends on a
    }

    // -----------------------------------------------------------------------
    // Configuration validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_config_max_plans() {
        let mut cfg = PlanSynthesisConfig::default();
        cfg.max_plans = 0;
        assert!(PlanSynthesis::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_steps() {
        let mut cfg = PlanSynthesisConfig::default();
        cfg.max_steps_per_plan = 0;
        assert!(PlanSynthesis::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Engine construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_default() {
        let ps = PlanSynthesis::new();
        assert_eq!(ps.plan_count(), 0);
        assert_eq!(ps.current_tick(), 0);
    }

    // -----------------------------------------------------------------------
    // Plan management
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_plan() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "Plan 1").unwrap();
        assert_eq!(ps.plan_count(), 1);
        let p = ps.plan("p1").unwrap();
        assert_eq!(p.name, "Plan 1");
        assert_eq!(p.state, PlanState::Draft);
    }

    #[test]
    fn test_create_plan_duplicate() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "Plan 1").unwrap();
        assert!(ps.create_plan("p1", "Dup").is_err());
    }

    #[test]
    fn test_create_plan_max_capacity() {
        let mut cfg = PlanSynthesisConfig::default();
        cfg.max_plans = 2;
        let mut ps = PlanSynthesis::with_config(cfg).unwrap();
        ps.create_plan("a", "A").unwrap();
        ps.create_plan("b", "B").unwrap();
        assert!(ps.create_plan("c", "C").is_err());
    }

    #[test]
    fn test_remove_plan() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "Plan 1").unwrap();
        assert!(ps.remove_plan("p1"));
        assert_eq!(ps.plan_count(), 0);
        assert!(!ps.remove_plan("p1"));
    }

    #[test]
    fn test_plan_ids_sorted() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("z", "Z").unwrap();
        ps.create_plan("a", "A").unwrap();
        ps.create_plan("m", "M").unwrap();
        assert_eq!(ps.plan_ids(), vec!["a", "m", "z"]);
    }

    // -----------------------------------------------------------------------
    // Step management
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_step() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "Plan 1").unwrap();
        ps.add_step("p1", PlanStep::new("s1", "Step 1")).unwrap();

        let p = ps.plan("p1").unwrap();
        assert_eq!(p.step_count(), 1);
        assert_eq!(p.step_ids(), &["s1"]);
        assert!(p.step("s1").is_some());
    }

    #[test]
    fn test_add_step_duplicate() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("s1", "S1")).unwrap();
        assert!(ps.add_step("p1", PlanStep::new("s1", "S1 dup")).is_err());
    }

    #[test]
    fn test_add_step_nonexistent_plan() {
        let mut ps = PlanSynthesis::new();
        assert!(ps.add_step("nope", PlanStep::new("s1", "S1")).is_err());
    }

    #[test]
    fn test_add_step_non_draft_plan() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("s1", "S1").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        // Plan is now Validated, can't add steps
        assert!(ps.add_step("p1", PlanStep::new("s2", "S2")).is_err());
    }

    #[test]
    fn test_add_step_max_capacity() {
        let mut cfg = PlanSynthesisConfig::default();
        cfg.max_steps_per_plan = 2;
        let mut ps = PlanSynthesis::with_config(cfg).unwrap();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A")).unwrap();
        ps.add_step("p1", PlanStep::new("b", "B")).unwrap();
        assert!(ps.add_step("p1", PlanStep::new("c", "C")).is_err());
    }

    #[test]
    fn test_remove_step() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("s1", "S1")).unwrap();
        assert!(ps.remove_step("p1", "s1").unwrap());
        assert_eq!(ps.plan("p1").unwrap().step_count(), 0);
    }

    #[test]
    fn test_remove_step_nonexistent() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        assert!(!ps.remove_step("p1", "nope").unwrap());
    }

    #[test]
    fn test_remove_step_removes_from_dependencies() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A")).unwrap();
        ps.add_step("p1", PlanStep::new("b", "B").with_dependency("a"))
            .unwrap();
        ps.remove_step("p1", "a").unwrap();

        let b = ps.plan("p1").unwrap().step("b").unwrap();
        assert!(b.dependencies.is_empty());
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_simple() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("s1", "S1").with_value(10.0))
            .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("s2", "S2")
                .with_dependency("s1")
                .with_value(20.0),
        )
        .unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(result.valid);
        assert!(result.errors.is_empty());
        assert_eq!(result.execution_order, vec!["s1", "s2"]);
        assert_eq!(ps.plan("p1").unwrap().state, PlanState::Validated);
    }

    #[test]
    fn test_validate_empty_plan() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(result.valid); // empty is valid but warned
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_validate_cycle() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_dependency("b"))
            .unwrap();
        ps.add_step("p1", PlanStep::new("b", "B").with_dependency("a"))
            .unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("cycle")));
    }

    #[test]
    fn test_validate_self_dependency() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_dependency("a"))
            .unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("itself")));
    }

    #[test]
    fn test_validate_missing_dependency() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_dependency("nonexistent"))
            .unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("does not exist")));
    }

    #[test]
    fn test_validate_resource_overflow() {
        let mut ps = PlanSynthesis::new();
        ps.set_resource_budget("capital", 100.0).unwrap();

        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_resource("capital", 60.0))
            .unwrap();
        ps.add_step("p1", PlanStep::new("b", "B").with_resource("capital", 60.0))
            .unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("over budget")));
    }

    #[test]
    fn test_validate_resource_within_budget() {
        let mut ps = PlanSynthesis::new();
        ps.set_resource_budget("capital", 200.0).unwrap();

        ps.create_plan("p1", "P1").unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("a", "A")
                .with_resource("capital", 60.0)
                .with_value(10.0),
        )
        .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("b", "B")
                .with_resource("capital", 60.0)
                .with_value(10.0),
        )
        .unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_validate_nonexistent_plan() {
        let mut ps = PlanSynthesis::new();
        assert!(ps.validate("nope").is_err());
    }

    #[test]
    fn test_validate_warns_zero_success_prob() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_success_prob(0.0))
            .unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(result.valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("zero or negative"))
        );
    }

    #[test]
    fn test_validate_warns_orphan() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.add_step("p1", PlanStep::new("b", "B").with_value(10.0))
            .unwrap();
        // Both independent of each other → both are orphans

        let result = ps.validate("p1").unwrap();
        assert!(result.valid);
        assert!(result.warnings.iter().any(|w| w.contains("orphan")));
    }

    // -----------------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_score() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("a", "A")
                .with_value(100.0)
                .with_success_prob(0.9)
                .with_duration(5)
                .with_resource("capital", 50.0),
        )
        .unwrap();

        let breakdown = ps.compute_score(ps.plan("p1").unwrap());
        assert!(breakdown.expected_value > 0.0);
        assert!(breakdown.compound_success_prob > 0.0);
        assert!(breakdown.risk_adjusted_return > 0.0);
        assert!(breakdown.resource_efficiency > 0.0);
        assert!(breakdown.time_efficiency > 0.0);
        assert!(breakdown.composite > 0.0);
    }

    #[test]
    fn test_rank_plans() {
        let mut ps = PlanSynthesis::new();

        ps.create_plan("low", "Low").unwrap();
        ps.add_step(
            "low",
            PlanStep::new("s1", "S1")
                .with_value(10.0)
                .with_success_prob(0.5),
        )
        .unwrap();
        ps.validate("low").unwrap();

        ps.create_plan("high", "High").unwrap();
        ps.add_step(
            "high",
            PlanStep::new("s1", "S1")
                .with_value(100.0)
                .with_success_prob(0.9),
        )
        .unwrap();
        ps.validate("high").unwrap();

        let ranked = ps.rank_plans();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, "high");
        assert!(ranked[0].1 > ranked[1].1);
    }

    // -----------------------------------------------------------------------
    // Execution lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_start_execution() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();

        assert_eq!(ps.plan("p1").unwrap().state, PlanState::Executing);
        // Step "a" has no deps, should be Ready
        assert_eq!(
            ps.plan("p1").unwrap().step("a").unwrap().state,
            StepState::Ready
        );
    }

    #[test]
    fn test_start_execution_non_validated() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        assert!(ps.start_execution("p1").is_err());
    }

    #[test]
    fn test_start_step() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        ps.start_step("p1", "a").unwrap();

        assert_eq!(
            ps.plan("p1").unwrap().step("a").unwrap().state,
            StepState::Running
        );
    }

    #[test]
    fn test_start_step_not_ready() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("b", "B")
                .with_dependency("a")
                .with_value(10.0),
        )
        .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        // b is Pending, not Ready
        assert!(ps.start_step("p1", "b").is_err());
    }

    #[test]
    fn test_complete_step() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        ps.start_step("p1", "a").unwrap();
        ps.complete_step("p1", "a").unwrap();

        assert_eq!(
            ps.plan("p1").unwrap().step("a").unwrap().state,
            StepState::Done
        );
        // Single step, plan should be completed
        assert_eq!(ps.plan("p1").unwrap().state, PlanState::Completed);
    }

    #[test]
    fn test_complete_step_advances_dependents() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("b", "B")
                .with_dependency("a")
                .with_value(10.0),
        )
        .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        ps.start_step("p1", "a").unwrap();
        ps.complete_step("p1", "a").unwrap();

        // b should now be Ready
        assert_eq!(
            ps.plan("p1").unwrap().step("b").unwrap().state,
            StepState::Ready
        );
    }

    #[test]
    fn test_complete_step_not_running() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        // a is Ready, not Running
        assert!(ps.complete_step("p1", "a").is_err());
    }

    #[test]
    fn test_fail_step() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("b", "B")
                .with_dependency("a")
                .with_value(10.0),
        )
        .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        ps.start_step("p1", "a").unwrap();
        ps.fail_step("p1", "a").unwrap();

        assert_eq!(
            ps.plan("p1").unwrap().step("a").unwrap().state,
            StepState::Failed
        );
        // b should be Skipped
        assert_eq!(
            ps.plan("p1").unwrap().step("b").unwrap().state,
            StepState::Skipped
        );
        // Plan should be Failed
        assert_eq!(ps.plan("p1").unwrap().state, PlanState::Failed);
    }

    #[test]
    fn test_fail_step_not_running() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        assert!(ps.fail_step("p1", "a").is_err());
    }

    #[test]
    fn test_cancel_plan() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.cancel_plan("p1").unwrap();
        assert_eq!(ps.plan("p1").unwrap().state, PlanState::Cancelled);
    }

    #[test]
    fn test_cancel_plan_already_terminal() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.cancel_plan("p1").unwrap();
        assert!(ps.cancel_plan("p1").is_err());
    }

    #[test]
    fn test_cancel_plan_nonexistent() {
        let mut ps = PlanSynthesis::new();
        assert!(ps.cancel_plan("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // Full execution flow
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_execution_flow() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "Trading Plan").unwrap();

        ps.add_step(
            "p1",
            PlanStep::new("analyze", "Market Analysis")
                .with_value(50.0)
                .with_duration(2)
                .with_success_prob(0.95),
        )
        .unwrap();

        ps.add_step(
            "p1",
            PlanStep::new("enter", "Enter Position")
                .with_dependency("analyze")
                .with_value(100.0)
                .with_duration(1)
                .with_success_prob(0.8)
                .with_resource("capital", 10000.0),
        )
        .unwrap();

        ps.add_step(
            "p1",
            PlanStep::new("exit", "Exit Position")
                .with_dependency("enter")
                .with_value(200.0)
                .with_duration(1)
                .with_success_prob(0.9),
        )
        .unwrap();

        // Validate
        let result = ps.validate("p1").unwrap();
        assert!(result.valid);
        assert_eq!(result.execution_order, vec!["analyze", "enter", "exit"]);
        assert!(result.score > 0.0);

        // Execute
        ps.start_execution("p1").unwrap();
        assert_eq!(ps.plan("p1").unwrap().ready_steps(), vec!["analyze"]);

        ps.start_step("p1", "analyze").unwrap();
        ps.complete_step("p1", "analyze").unwrap();

        assert_eq!(ps.plan("p1").unwrap().ready_steps(), vec!["enter"]);

        ps.start_step("p1", "enter").unwrap();
        ps.complete_step("p1", "enter").unwrap();

        ps.start_step("p1", "exit").unwrap();
        ps.complete_step("p1", "exit").unwrap();

        assert_eq!(ps.plan("p1").unwrap().state, PlanState::Completed);
        assert!((ps.plan("p1").unwrap().progress() - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Resource budgets
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_resource_budget() {
        let mut ps = PlanSynthesis::new();
        ps.set_resource_budget("capital", 10000.0).unwrap();
        assert!((ps.resource_budget("capital").unwrap() - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_resource_budget_missing() {
        let ps = PlanSynthesis::new();
        assert!(ps.resource_budget("nope").is_none());
    }

    #[test]
    fn test_clear_resource_budgets() {
        let mut ps = PlanSynthesis::new();
        ps.set_resource_budget("capital", 10000.0).unwrap();
        ps.clear_resource_budgets();
        assert!(ps.resource_budget("capital").is_none());
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let ps = PlanSynthesis::new();
        let s = ps.stats();
        assert_eq!(s.total_plans, 0);
        assert!((s.success_rate() - 0.0).abs() < 1e-10);
        assert!((s.mean_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_after_operations() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        ps.start_step("p1", "a").unwrap();
        ps.complete_step("p1", "a").unwrap();

        let s = ps.stats();
        assert_eq!(s.total_plans, 1);
        assert_eq!(s.completed_count, 1);
        assert!((s.success_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_success_rate_mixed() {
        let mut ps = PlanSynthesis::new();

        // Completed plan
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        ps.start_step("p1", "a").unwrap();
        ps.complete_step("p1", "a").unwrap();

        // Failed plan
        ps.create_plan("p2", "P2").unwrap();
        ps.add_step("p2", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p2").unwrap();
        ps.start_execution("p2").unwrap();
        ps.start_step("p2", "a").unwrap();
        ps.fail_step("p2", "a").unwrap();

        assert!((ps.stats().success_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_active_plans() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.create_plan("p2", "P2").unwrap();

        assert_eq!(ps.stats().active_plans, 2);

        ps.cancel_plan("p1").unwrap();
        assert_eq!(ps.stats().active_plans, 1);
    }

    #[test]
    fn test_stats_peak_score() {
        let mut ps = PlanSynthesis::new();

        ps.create_plan("low", "Low").unwrap();
        ps.add_step(
            "low",
            PlanStep::new("s", "S")
                .with_value(10.0)
                .with_success_prob(0.5),
        )
        .unwrap();
        ps.validate("low").unwrap();

        ps.create_plan("high", "High").unwrap();
        ps.add_step(
            "high",
            PlanStep::new("s", "S")
                .with_value(100.0)
                .with_success_prob(0.9),
        )
        .unwrap();
        ps.validate("high").unwrap();

        assert!(ps.stats().peak_score > 0.0);
        assert!(
            ps.stats().peak_score >= ps.plan("high").unwrap().score
                || ps.stats().peak_score >= ps.plan("low").unwrap().score
        );
    }

    // -----------------------------------------------------------------------
    // Tick
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick() {
        let mut ps = PlanSynthesis::new();
        ps.tick();
        ps.tick();
        assert_eq!(ps.current_tick(), 2);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.set_resource_budget("capital", 1000.0).unwrap();
        ps.tick();

        ps.reset();

        assert_eq!(ps.plan_count(), 0);
        assert_eq!(ps.current_tick(), 0);
        assert!(ps.resource_budget("capital").is_none());
    }

    // -----------------------------------------------------------------------
    // Process compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_process() {
        let ps = PlanSynthesis::new();
        assert!(ps.process().is_ok());
    }

    // -----------------------------------------------------------------------
    // Topological sort: diamond dependency
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_diamond_dependency() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        //   a
        //  / \
        // b   c
        //  \ /
        //   d
        ps.add_step("p1", PlanStep::new("a", "A").with_value(1.0))
            .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("b", "B").with_dependency("a").with_value(1.0),
        )
        .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("c", "C").with_dependency("a").with_value(1.0),
        )
        .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("d", "D")
                .with_dependency("b")
                .with_dependency("c")
                .with_value(1.0),
        )
        .unwrap();

        let result = ps.validate("p1").unwrap();
        assert!(result.valid);
        // a must come first, d must come last, b and c in between
        assert_eq!(result.execution_order[0], "a");
        assert_eq!(*result.execution_order.last().unwrap(), "d");
    }

    // -----------------------------------------------------------------------
    // Parallel steps: multiple ready at once
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_ready_steps() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("a", "A").with_value(10.0).with_priority(2),
        )
        .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("b", "B").with_value(10.0).with_priority(1),
        )
        .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();

        let ready = ps.plan("p1").unwrap().ready_steps();
        // Sorted by priority: b (priority 1) before a (priority 2)
        assert_eq!(ready, vec!["b", "a"]);
    }

    // -----------------------------------------------------------------------
    // Fail cascading skips deep dependents
    // -----------------------------------------------------------------------

    #[test]
    fn test_fail_cascading_skip() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("b", "B")
                .with_dependency("a")
                .with_value(10.0),
        )
        .unwrap();
        ps.add_step(
            "p1",
            PlanStep::new("c", "C")
                .with_dependency("b")
                .with_value(10.0),
        )
        .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        ps.start_step("p1", "a").unwrap();
        ps.fail_step("p1", "a").unwrap();

        // b and c should both be skipped
        assert_eq!(
            ps.plan("p1").unwrap().step("b").unwrap().state,
            StepState::Skipped
        );
        assert_eq!(
            ps.plan("p1").unwrap().step("c").unwrap().state,
            StepState::Skipped
        );
    }

    // -----------------------------------------------------------------------
    // plan_mut
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_mut() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();

        let p = ps.plan_mut("p1").unwrap();
        p.description = "Updated".into();

        assert_eq!(ps.plan("p1").unwrap().description, "Updated");
    }

    #[test]
    fn test_plan_mut_none() {
        let mut ps = PlanSynthesis::new();
        assert!(ps.plan_mut("nope").is_none());
    }

    // -----------------------------------------------------------------------
    // Plan goal_id
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_goal_id() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();

        let p = ps.plan_mut("p1").unwrap();
        p.goal_id = Some("goal_123".into());

        assert_eq!(ps.plan("p1").unwrap().goal_id.as_deref(), Some("goal_123"));
    }

    // -----------------------------------------------------------------------
    // Completed tick tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_completed_tick_recorded() {
        let mut ps = PlanSynthesis::new();
        ps.create_plan("p1", "P1").unwrap();
        ps.add_step("p1", PlanStep::new("a", "A").with_value(10.0))
            .unwrap();
        ps.validate("p1").unwrap();
        ps.start_execution("p1").unwrap();
        ps.tick();
        ps.tick();
        ps.start_step("p1", "a").unwrap();
        ps.complete_step("p1", "a").unwrap();

        assert_eq!(ps.plan("p1").unwrap().completed_tick, Some(2));
    }
}
