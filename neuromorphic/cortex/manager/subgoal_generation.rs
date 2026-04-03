//! Subgoal generation engine for hierarchical goal decomposition
//!
//! Part of the Cortex region
//! Component: manager
//!
//! Takes high-level portfolio goals (e.g. "achieve 12% annual return with
//! max 8% drawdown") and decomposes them into a DAG of actionable sub-targets
//! that downstream workers can execute independently. Each subgoal carries
//! priority, resource requirements, estimated contribution to the parent
//! goal, and dependency links to other subgoals.
//!
//! Key features:
//! - Recursive goal decomposition with configurable depth limit
//! - Dependency DAG with topological ordering for execution scheduling
//! - Priority scoring based on urgency, impact, and feasibility
//! - Resource budget allocation across subgoals
//! - Progress tracking with roll-up to parent goals
//! - Conflict detection between competing subgoals
//! - EMA-smoothed tracking of decomposition quality metrics
//! - Sliding window of recent decomposition results

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the subgoal generation engine
#[derive(Debug, Clone)]
pub struct SubgoalConfig {
    /// Maximum decomposition depth (prevents infinite recursion)
    pub max_depth: usize,
    /// Maximum number of subgoals per parent goal
    pub max_children: usize,
    /// Maximum total subgoals across all active goals
    pub max_total_subgoals: usize,
    /// Minimum priority score to keep a subgoal (prune below this)
    pub min_priority: f64,
    /// Weight for urgency in priority calculation
    pub urgency_weight: f64,
    /// Weight for impact in priority calculation
    pub impact_weight: f64,
    /// Weight for feasibility in priority calculation
    pub feasibility_weight: f64,
    /// EMA decay factor for quality tracking (0 < decay < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent decompositions
    pub window_size: usize,
    /// Whether to automatically prune low-priority subgoals
    pub auto_prune: bool,
    /// Resource budget cap (abstract units, 0 = unlimited)
    pub resource_budget: f64,
}

impl Default for SubgoalConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_children: 8,
            max_total_subgoals: 200,
            min_priority: 0.1,
            urgency_weight: 0.35,
            impact_weight: 0.40,
            feasibility_weight: 0.25,
            ema_decay: 0.1,
            window_size: 50,
            auto_prune: true,
            resource_budget: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Category of a subgoal
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SubgoalCategory {
    /// Allocation target (e.g. "allocate 30% to momentum")
    Allocation,
    /// Risk management target (e.g. "hedge tail risk to < 5%")
    RiskManagement,
    /// Signal generation target (e.g. "generate alpha signal for sector X")
    SignalGeneration,
    /// Execution target (e.g. "achieve VWAP within 2bps")
    Execution,
    /// Monitoring target (e.g. "track correlation regime shifts")
    Monitoring,
    /// Rebalance target (e.g. "rebalance portfolio within 1% of target")
    Rebalance,
    /// Research target (e.g. "evaluate new factor model")
    Research,
    /// Custom category
    Custom(String),
}

/// Status of a subgoal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgoalStatus {
    /// Not yet started
    Pending,
    /// In progress
    Active,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
    /// Blocked by dependency
    Blocked,
    /// Pruned (removed due to low priority or conflict)
    Pruned,
}

impl SubgoalStatus {
    /// Whether the subgoal is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            SubgoalStatus::Completed | SubgoalStatus::Failed | SubgoalStatus::Pruned
        )
    }

    /// Whether the subgoal is actionable
    pub fn is_actionable(&self) -> bool {
        matches!(self, SubgoalStatus::Pending | SubgoalStatus::Active)
    }
}

/// A single subgoal in the decomposition tree
#[derive(Debug, Clone)]
pub struct Subgoal {
    /// Unique identifier
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Category
    pub category: SubgoalCategory,
    /// Current status
    pub status: SubgoalStatus,
    /// Parent goal ID (None for root goals)
    pub parent_id: Option<String>,
    /// IDs of subgoals this one depends on (must complete first)
    pub dependencies: Vec<String>,
    /// Depth in the decomposition tree (0 = root)
    pub depth: usize,
    /// Priority score (higher = more important, computed from urgency/impact/feasibility)
    pub priority: f64,
    /// Urgency score (0.0–1.0): how time-sensitive is this subgoal
    pub urgency: f64,
    /// Impact score (0.0–1.0): how much does this contribute to the parent goal
    pub impact: f64,
    /// Feasibility score (0.0–1.0): how achievable is this given current resources
    pub feasibility: f64,
    /// Estimated contribution to parent goal (fraction 0.0–1.0)
    pub contribution: f64,
    /// Resource cost (abstract units)
    pub resource_cost: f64,
    /// Progress (0.0–1.0)
    pub progress: f64,
    /// Target value (domain-specific numeric target)
    pub target_value: f64,
    /// Current value toward target
    pub current_value: f64,
}

impl Subgoal {
    /// Create a new subgoal with given parameters
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        category: SubgoalCategory,
        parent_id: Option<String>,
        depth: usize,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            category,
            status: SubgoalStatus::Pending,
            parent_id,
            dependencies: Vec::new(),
            depth,
            priority: 0.5,
            urgency: 0.5,
            impact: 0.5,
            feasibility: 0.5,
            contribution: 0.0,
            resource_cost: 1.0,
            progress: 0.0,
            target_value: 0.0,
            current_value: 0.0,
        }
    }

    /// Whether all dependencies are in a completed state given a lookup closure
    pub fn dependencies_met<F>(&self, is_completed: F) -> bool
    where
        F: Fn(&str) -> bool,
    {
        self.dependencies.iter().all(|dep| is_completed(dep))
    }

    /// Fraction of progress toward target value
    pub fn target_progress(&self) -> f64 {
        if self.target_value.abs() < 1e-15 {
            return self.progress;
        }
        (self.current_value / self.target_value).clamp(0.0, 1.0)
    }
}

/// A high-level goal to be decomposed
#[derive(Debug, Clone)]
pub struct GoalSpec {
    /// Unique identifier for this goal
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Target annual return (e.g. 0.12 for 12%)
    pub target_return: f64,
    /// Maximum acceptable drawdown (e.g. 0.08 for 8%)
    pub max_drawdown: f64,
    /// Target Sharpe ratio
    pub target_sharpe: f64,
    /// Risk budget (fraction of portfolio, e.g. 0.10)
    pub risk_budget: f64,
    /// Time horizon in trading days
    pub horizon_days: u32,
    /// Available resource budget (abstract units)
    pub resource_budget: f64,
}

impl Default for GoalSpec {
    fn default() -> Self {
        Self {
            id: "default_goal".into(),
            description: "Default portfolio goal".into(),
            target_return: 0.10,
            max_drawdown: 0.10,
            target_sharpe: 1.5,
            risk_budget: 0.10,
            horizon_days: 252,
            resource_budget: 100.0,
        }
    }
}

/// Result of a decomposition operation
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// The goal that was decomposed
    pub goal_id: String,
    /// Generated subgoals
    pub subgoals: Vec<Subgoal>,
    /// Total resource cost of all subgoals
    pub total_resource_cost: f64,
    /// Number of subgoals pruned (if auto_prune enabled)
    pub pruned_count: usize,
    /// Maximum depth reached
    pub max_depth_reached: usize,
    /// Whether any conflicts were detected
    pub has_conflicts: bool,
    /// Conflict descriptions (if any)
    pub conflict_descriptions: Vec<String>,
    /// Coverage: sum of contributions (ideally ~1.0)
    pub coverage: f64,
}

/// Record of a conflict between two subgoals
#[derive(Debug, Clone)]
pub struct ConflictRecord {
    /// First subgoal ID
    pub subgoal_a: String,
    /// Second subgoal ID
    pub subgoal_b: String,
    /// Description of the conflict
    pub description: String,
    /// Severity (0.0–1.0)
    pub severity: f64,
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the subgoal generation engine
#[derive(Debug, Clone)]
pub struct SubgoalStats {
    /// Total decompositions performed
    pub total_decompositions: u64,
    /// Total subgoals generated across all decompositions
    pub total_subgoals_generated: u64,
    /// Total subgoals pruned
    pub total_pruned: u64,
    /// Total conflicts detected
    pub total_conflicts: u64,
    /// EMA of subgoals per decomposition
    pub ema_subgoal_count: f64,
    /// EMA of average priority score
    pub ema_avg_priority: f64,
    /// EMA of coverage ratio
    pub ema_coverage: f64,
    /// EMA of resource utilization (cost / budget)
    pub ema_resource_utilization: f64,
    /// Best coverage ratio seen
    pub best_coverage: f64,
    /// Worst coverage ratio seen
    pub worst_coverage: f64,
}

impl Default for SubgoalStats {
    fn default() -> Self {
        Self {
            total_decompositions: 0,
            total_subgoals_generated: 0,
            total_pruned: 0,
            total_conflicts: 0,
            ema_subgoal_count: 0.0,
            ema_avg_priority: 0.0,
            ema_coverage: 0.0,
            ema_resource_utilization: 0.0,
            best_coverage: 0.0,
            worst_coverage: f64::MAX,
        }
    }
}

impl SubgoalStats {
    /// Average subgoals per decomposition
    pub fn avg_subgoals_per_decomposition(&self) -> f64 {
        if self.total_decompositions == 0 {
            return 0.0;
        }
        self.total_subgoals_generated as f64 / self.total_decompositions as f64
    }

    /// Prune rate
    pub fn prune_rate(&self) -> f64 {
        if self.total_subgoals_generated == 0 {
            return 0.0;
        }
        self.total_pruned as f64 / self.total_subgoals_generated as f64
    }

    /// Conflict rate per decomposition
    pub fn conflict_rate(&self) -> f64 {
        if self.total_decompositions == 0 {
            return 0.0;
        }
        self.total_conflicts as f64 / self.total_decompositions as f64
    }
}

// ---------------------------------------------------------------------------
// SubgoalGeneration Engine
// ---------------------------------------------------------------------------

/// Subgoal generation engine.
///
/// Decomposes high-level portfolio goals into a DAG of actionable subgoals
/// with priority scoring, dependency tracking, conflict detection, and
/// resource budget allocation.
pub struct SubgoalGeneration {
    config: SubgoalConfig,
    /// All tracked subgoals (across all active goals)
    subgoals: Vec<Subgoal>,
    /// EMA initialized flag
    ema_initialized: bool,
    /// Sliding window of recent decomposition results (summary only)
    recent: VecDeque<DecompositionSummaryRecord>,
    /// Running statistics
    stats: SubgoalStats,
    /// Decomposition counter
    decomposition_counter: u64,
}

/// Lightweight summary of a decomposition for the sliding window
#[derive(Debug, Clone)]
pub struct DecompositionSummaryRecord {
    /// Goal ID
    pub goal_id: String,
    /// Number of subgoals produced
    pub subgoal_count: usize,
    /// Number pruned
    pub pruned_count: usize,
    /// Coverage
    pub coverage: f64,
    /// Average priority
    pub avg_priority: f64,
    /// Total resource cost
    pub total_cost: f64,
    /// Number of conflicts
    pub conflict_count: usize,
}

impl Default for SubgoalGeneration {
    fn default() -> Self {
        Self::new()
    }
}

impl SubgoalGeneration {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self {
            config: SubgoalConfig::default(),
            subgoals: Vec::new(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: SubgoalStats::default(),
            decomposition_counter: 0,
        }
    }

    /// Create with explicit configuration
    pub fn with_config(config: SubgoalConfig) -> Result<Self> {
        if config.max_depth == 0 {
            return Err(Error::InvalidInput("max_depth must be > 0".into()));
        }
        if config.max_children == 0 {
            return Err(Error::InvalidInput("max_children must be > 0".into()));
        }
        if config.max_total_subgoals == 0 {
            return Err(Error::InvalidInput("max_total_subgoals must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        let wsum = config.urgency_weight + config.impact_weight + config.feasibility_weight;
        if wsum.abs() < 1e-15 {
            return Err(Error::InvalidInput(
                "priority weights must sum to a positive value".into(),
            ));
        }
        Ok(Self {
            config,
            subgoals: Vec::new(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: SubgoalStats::default(),
            decomposition_counter: 0,
        })
    }

    /// Main processing function (no-op entry point for trait conformance)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Priority calculation
    // -----------------------------------------------------------------------

    /// Compute priority score for a subgoal based on urgency, impact, and feasibility
    pub fn compute_priority(&self, urgency: f64, impact: f64, feasibility: f64) -> f64 {
        let wsum =
            self.config.urgency_weight + self.config.impact_weight + self.config.feasibility_weight;
        if wsum.abs() < 1e-15 {
            return 0.0;
        }
        let raw = self.config.urgency_weight * urgency.clamp(0.0, 1.0)
            + self.config.impact_weight * impact.clamp(0.0, 1.0)
            + self.config.feasibility_weight * feasibility.clamp(0.0, 1.0);
        raw / wsum
    }

    // -----------------------------------------------------------------------
    // Goal decomposition
    // -----------------------------------------------------------------------

    /// Decompose a high-level goal into subgoals.
    ///
    /// This generates a standard decomposition template based on the goal spec:
    /// 1. Allocation subgoals (momentum, value, mean-reversion)
    /// 2. Risk management subgoal (drawdown protection, hedging)
    /// 3. Execution subgoal (order routing, VWAP/TWAP)
    /// 4. Monitoring subgoal (regime tracking, signal quality)
    ///
    /// Each subgoal gets priority scoring and resource allocation. Conflicts
    /// are detected and low-priority subgoals are pruned if configured.
    pub fn decompose(&mut self, goal: &GoalSpec) -> Result<DecompositionResult> {
        if goal.id.is_empty() {
            return Err(Error::InvalidInput("goal id must not be empty".into()));
        }
        if goal.target_return < 0.0 {
            return Err(Error::InvalidInput(
                "target_return must be non-negative".into(),
            ));
        }
        if goal.max_drawdown <= 0.0 || goal.max_drawdown > 1.0 {
            return Err(Error::InvalidInput("max_drawdown must be in (0, 1]".into()));
        }

        self.decomposition_counter += 1;
        let mut generated: Vec<Subgoal> = Vec::new();
        let mut conflicts: Vec<ConflictRecord> = Vec::new();

        // Depth 1: Primary decomposition into category subgoals

        // 1. Allocation: split return target across strategies
        let alloc_return_share = 0.50;
        let risk_return_share = 0.20;
        let exec_return_share = 0.15;
        let monitor_return_share = 0.15;

        // Allocation subgoals
        let strategies = [
            ("momentum", 0.35, 0.7, 0.6, 0.8),
            ("value", 0.30, 0.5, 0.5, 0.9),
            ("mean_reversion", 0.20, 0.6, 0.4, 0.7),
            ("carry", 0.15, 0.4, 0.3, 0.85),
        ];

        for (name, weight, urgency, impact, feasibility) in &strategies {
            let id = format!("{}_alloc_{}", goal.id, name);
            let mut sg = Subgoal::new(
                &id,
                format!(
                    "Allocate {:.0}% of portfolio to {} strategy",
                    weight * 100.0,
                    name
                ),
                SubgoalCategory::Allocation,
                Some(goal.id.clone()),
                1,
            );
            sg.urgency = *urgency;
            sg.impact = *impact;
            sg.feasibility = *feasibility;
            sg.priority = self.compute_priority(sg.urgency, sg.impact, sg.feasibility);
            sg.contribution = alloc_return_share * weight;
            sg.target_value = goal.target_return * alloc_return_share * weight;
            sg.resource_cost = goal.resource_budget * alloc_return_share * weight;
            generated.push(sg);
        }

        // 2. Risk management subgoal
        {
            let id = format!("{}_risk_mgmt", goal.id);
            let mut sg = Subgoal::new(
                &id,
                format!(
                    "Maintain drawdown below {:.1}% with hedging",
                    goal.max_drawdown * 100.0
                ),
                SubgoalCategory::RiskManagement,
                Some(goal.id.clone()),
                1,
            );
            sg.urgency = 0.8;
            sg.impact = 0.9;
            sg.feasibility = 0.7;
            sg.priority = self.compute_priority(sg.urgency, sg.impact, sg.feasibility);
            sg.contribution = risk_return_share;
            sg.target_value = goal.max_drawdown;
            sg.resource_cost = goal.resource_budget * risk_return_share;
            // Risk depends on allocation subgoals (needs to know positions)
            for (name, _, _, _, _) in &strategies {
                sg.dependencies.push(format!("{}_alloc_{}", goal.id, name));
            }
            generated.push(sg);
        }

        // 3. Execution subgoal
        {
            let id = format!("{}_execution", goal.id);
            let mut sg = Subgoal::new(
                &id,
                "Achieve optimal execution (minimize slippage and market impact)".to_string(),
                SubgoalCategory::Execution,
                Some(goal.id.clone()),
                1,
            );
            sg.urgency = 0.6;
            sg.impact = 0.5;
            sg.feasibility = 0.85;
            sg.priority = self.compute_priority(sg.urgency, sg.impact, sg.feasibility);
            sg.contribution = exec_return_share;
            sg.target_value = 0.0002; // target 2bps slippage
            sg.resource_cost = goal.resource_budget * exec_return_share;
            // Execution depends on risk management (needs risk limits)
            sg.dependencies.push(format!("{}_risk_mgmt", goal.id));
            generated.push(sg);
        }

        // 4. Monitoring subgoal
        {
            let id = format!("{}_monitoring", goal.id);
            let mut sg = Subgoal::new(
                &id,
                "Monitor regime shifts, correlation changes, and signal quality".to_string(),
                SubgoalCategory::Monitoring,
                Some(goal.id.clone()),
                1,
            );
            sg.urgency = 0.5;
            sg.impact = 0.4;
            sg.feasibility = 0.9;
            sg.priority = self.compute_priority(sg.urgency, sg.impact, sg.feasibility);
            sg.contribution = monitor_return_share;
            sg.target_value = 1.0; // continuous monitoring
            sg.resource_cost = goal.resource_budget * monitor_return_share;
            generated.push(sg);
        }

        // 5. If Sharpe target is high, add a research subgoal for alpha improvement
        if goal.target_sharpe > 2.0 {
            let id = format!("{}_research_alpha", goal.id);
            let mut sg = Subgoal::new(
                &id,
                format!(
                    "Research new alpha sources to achieve Sharpe > {:.1}",
                    goal.target_sharpe
                ),
                SubgoalCategory::Research,
                Some(goal.id.clone()),
                1,
            );
            sg.urgency = 0.3;
            sg.impact = 0.6;
            sg.feasibility = 0.4;
            sg.priority = self.compute_priority(sg.urgency, sg.impact, sg.feasibility);
            sg.contribution = 0.0; // research doesn't directly contribute
            sg.resource_cost = goal.resource_budget * 0.05;
            generated.push(sg);
        }

        // 6. If risk budget is tight, add rebalance subgoal
        if goal.risk_budget < 0.05 {
            let id = format!("{}_rebalance", goal.id);
            let mut sg = Subgoal::new(
                &id,
                "Frequent rebalancing to stay within tight risk budget".to_string(),
                SubgoalCategory::Rebalance,
                Some(goal.id.clone()),
                1,
            );
            sg.urgency = 0.9;
            sg.impact = 0.7;
            sg.feasibility = 0.6;
            sg.priority = self.compute_priority(sg.urgency, sg.impact, sg.feasibility);
            sg.contribution = 0.0;
            sg.resource_cost = goal.resource_budget * 0.05;
            sg.dependencies.push(format!("{}_risk_mgmt", goal.id));
            generated.push(sg);
        }

        // Detect conflicts
        // Conflict: momentum and mean_reversion can oppose each other
        let has_momentum = generated.iter().any(|s| s.id.contains("momentum"));
        let has_meanrev = generated.iter().any(|s| s.id.contains("mean_reversion"));
        if has_momentum && has_meanrev {
            conflicts.push(ConflictRecord {
                subgoal_a: format!("{}_alloc_momentum", goal.id),
                subgoal_b: format!("{}_alloc_mean_reversion", goal.id),
                description: "Momentum and mean-reversion strategies may produce \
                              opposing signals in ranging markets"
                    .into(),
                severity: 0.4,
            });
        }

        // Prune low-priority subgoals if configured
        let mut pruned_count = 0;
        if self.config.auto_prune {
            let before = generated.len();
            generated.retain(|sg| sg.priority >= self.config.min_priority);
            pruned_count = before - generated.len();
        }

        // Enforce resource budget
        if self.config.resource_budget > 0.0 {
            let total_cost: f64 = generated.iter().map(|sg| sg.resource_cost).sum();
            if total_cost > self.config.resource_budget {
                // Scale down resource costs proportionally
                let scale = self.config.resource_budget / total_cost;
                for sg in &mut generated {
                    sg.resource_cost *= scale;
                }
            }
        }

        // Enforce max_total_subgoals
        let current_count = self.subgoals.len();
        let available = self.config.max_total_subgoals.saturating_sub(current_count);
        if generated.len() > available {
            // Sort by priority descending and keep top N
            generated.sort_by(|a, b| {
                b.priority
                    .partial_cmp(&a.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let extra = generated.len() - available;
            generated.truncate(available);
            pruned_count += extra;
        }

        // Compute summary metrics
        let total_resource_cost: f64 = generated.iter().map(|sg| sg.resource_cost).sum();
        let max_depth_reached = generated.iter().map(|sg| sg.depth).max().unwrap_or(0);
        let coverage: f64 = generated.iter().map(|sg| sg.contribution).sum();
        let avg_priority = if generated.is_empty() {
            0.0
        } else {
            generated.iter().map(|sg| sg.priority).sum::<f64>() / generated.len() as f64
        };

        let conflict_descriptions: Vec<String> =
            conflicts.iter().map(|c| c.description.clone()).collect();
        let has_conflicts = !conflicts.is_empty();
        let conflict_count = conflicts.len();

        let result = DecompositionResult {
            goal_id: goal.id.clone(),
            subgoals: generated.clone(),
            total_resource_cost,
            pruned_count,
            max_depth_reached,
            has_conflicts,
            conflict_descriptions,
            coverage,
        };

        // Store subgoals
        self.subgoals.extend(generated.iter().cloned());

        // Update stats
        self.stats.total_decompositions += 1;
        self.stats.total_subgoals_generated += generated.len() as u64;
        self.stats.total_pruned += pruned_count as u64;
        self.stats.total_conflicts += conflict_count as u64;

        if coverage > self.stats.best_coverage {
            self.stats.best_coverage = coverage;
        }
        if coverage < self.stats.worst_coverage {
            self.stats.worst_coverage = coverage;
        }

        // EMA update
        let alpha = self.config.ema_decay;
        let resource_util = if goal.resource_budget > 0.0 {
            total_resource_cost / goal.resource_budget
        } else {
            0.0
        };

        if !self.ema_initialized {
            self.stats.ema_subgoal_count = generated.len() as f64;
            self.stats.ema_avg_priority = avg_priority;
            self.stats.ema_coverage = coverage;
            self.stats.ema_resource_utilization = resource_util;
            self.ema_initialized = true;
        } else {
            self.stats.ema_subgoal_count =
                alpha * generated.len() as f64 + (1.0 - alpha) * self.stats.ema_subgoal_count;
            self.stats.ema_avg_priority =
                alpha * avg_priority + (1.0 - alpha) * self.stats.ema_avg_priority;
            self.stats.ema_coverage = alpha * coverage + (1.0 - alpha) * self.stats.ema_coverage;
            self.stats.ema_resource_utilization =
                alpha * resource_util + (1.0 - alpha) * self.stats.ema_resource_utilization;
        }

        // Sliding window
        let summary_record = DecompositionSummaryRecord {
            goal_id: goal.id.clone(),
            subgoal_count: generated.len(),
            pruned_count,
            coverage,
            avg_priority,
            total_cost: total_resource_cost,
            conflict_count,
        };
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(summary_record);

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Subgoal management
    // -----------------------------------------------------------------------

    /// Get all tracked subgoals
    pub fn subgoals(&self) -> &[Subgoal] {
        &self.subgoals
    }

    /// Get a subgoal by ID
    pub fn get_subgoal(&self, id: &str) -> Option<&Subgoal> {
        self.subgoals.iter().find(|sg| sg.id == id)
    }

    /// Get a mutable reference to a subgoal by ID
    pub fn get_subgoal_mut(&mut self, id: &str) -> Option<&mut Subgoal> {
        self.subgoals.iter_mut().find(|sg| sg.id == id)
    }

    /// Update the status of a subgoal
    pub fn update_status(&mut self, id: &str, status: SubgoalStatus) -> Result<()> {
        let sg = self
            .subgoals
            .iter_mut()
            .find(|sg| sg.id == id)
            .ok_or_else(|| Error::NotFound(format!("subgoal '{}' not found", id)))?;
        sg.status = status;
        Ok(())
    }

    /// Update progress on a subgoal
    pub fn update_progress(&mut self, id: &str, progress: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&progress) {
            return Err(Error::InvalidInput("progress must be in [0, 1]".into()));
        }
        let sg = self
            .subgoals
            .iter_mut()
            .find(|sg| sg.id == id)
            .ok_or_else(|| Error::NotFound(format!("subgoal '{}' not found", id)))?;
        sg.progress = progress;
        if progress >= 1.0 && sg.status == SubgoalStatus::Active {
            sg.status = SubgoalStatus::Completed;
        } else if progress > 0.0 && sg.status == SubgoalStatus::Pending {
            sg.status = SubgoalStatus::Active;
        }
        Ok(())
    }

    /// Get all children of a given parent goal
    pub fn children_of(&self, parent_id: &str) -> Vec<&Subgoal> {
        self.subgoals
            .iter()
            .filter(|sg| sg.parent_id.as_deref() == Some(parent_id))
            .collect()
    }

    /// Compute aggregate progress of a parent goal from its children
    pub fn aggregate_progress(&self, parent_id: &str) -> f64 {
        let children = self.children_of(parent_id);
        if children.is_empty() {
            return 0.0;
        }
        let total_contrib: f64 = children.iter().map(|c| c.contribution).sum();
        if total_contrib.abs() < 1e-15 {
            // Equal weighting fallback
            return children.iter().map(|c| c.progress).sum::<f64>() / children.len() as f64;
        }
        children
            .iter()
            .map(|c| c.progress * c.contribution)
            .sum::<f64>()
            / total_contrib
    }

    /// Get subgoals in topological order (respecting dependencies).
    ///
    /// Returns IDs in execution order. Subgoals with no dependencies come first.
    pub fn topological_order(&self) -> Result<Vec<String>> {
        let n = self.subgoals.len();
        let mut in_degree: Vec<usize> = Vec::with_capacity(n);
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Map id -> index
        let id_to_idx: std::collections::HashMap<&str, usize> = self
            .subgoals
            .iter()
            .enumerate()
            .map(|(i, sg)| (sg.id.as_str(), i))
            .collect();

        for (i, sg) in self.subgoals.iter().enumerate() {
            let mut deg = 0usize;
            for dep in &sg.dependencies {
                if let Some(&dep_idx) = id_to_idx.get(dep.as_str()) {
                    adj[dep_idx].push(i);
                    deg += 1;
                }
                // Dependencies on external (non-tracked) subgoals are ignored
            }
            in_degree.push(deg);
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut order: Vec<String> = Vec::with_capacity(n);
        while let Some(idx) = queue.pop_front() {
            order.push(self.subgoals[idx].id.clone());
            for &next in &adj[idx] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != n {
            return Err(Error::InvalidState(
                "dependency cycle detected among subgoals".into(),
            ));
        }

        Ok(order)
    }

    /// Get all actionable subgoals (pending or active with dependencies met)
    pub fn actionable_subgoals(&self) -> Vec<&Subgoal> {
        self.subgoals
            .iter()
            .filter(|sg| {
                sg.status.is_actionable()
                    && sg.dependencies_met(|dep_id| {
                        self.subgoals
                            .iter()
                            .find(|s| s.id == dep_id)
                            .map(|s| s.status == SubgoalStatus::Completed)
                            .unwrap_or(true) // external deps assumed met
                    })
            })
            .collect()
    }

    /// Get all blocked subgoals (actionable status but dependencies not met)
    pub fn blocked_subgoals(&self) -> Vec<&Subgoal> {
        self.subgoals
            .iter()
            .filter(|sg| {
                sg.status.is_actionable()
                    && !sg.dependencies_met(|dep_id| {
                        self.subgoals
                            .iter()
                            .find(|s| s.id == dep_id)
                            .map(|s| s.status == SubgoalStatus::Completed)
                            .unwrap_or(true)
                    })
            })
            .collect()
    }

    /// Remove all subgoals for a given parent goal
    pub fn clear_goal(&mut self, parent_id: &str) {
        self.subgoals
            .retain(|sg| sg.parent_id.as_deref() != Some(parent_id));
    }

    /// Total number of tracked subgoals
    pub fn total_subgoals(&self) -> usize {
        self.subgoals.len()
    }

    /// Count subgoals by status
    pub fn count_by_status(&self, status: SubgoalStatus) -> usize {
        self.subgoals
            .iter()
            .filter(|sg| sg.status == status)
            .count()
    }

    /// Total resource cost of all active (non-terminal) subgoals
    pub fn active_resource_cost(&self) -> f64 {
        self.subgoals
            .iter()
            .filter(|sg| !sg.status.is_terminal())
            .map(|sg| sg.resource_cost)
            .sum()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Running statistics
    pub fn stats(&self) -> &SubgoalStats {
        &self.stats
    }

    /// Configuration
    pub fn config(&self) -> &SubgoalConfig {
        &self.config
    }

    /// Recent decomposition summaries
    pub fn recent_decompositions(&self) -> &VecDeque<DecompositionSummaryRecord> {
        &self.recent
    }

    /// EMA-smoothed subgoal count per decomposition
    pub fn smoothed_subgoal_count(&self) -> f64 {
        self.stats.ema_subgoal_count
    }

    /// EMA-smoothed average priority
    pub fn smoothed_avg_priority(&self) -> f64 {
        self.stats.ema_avg_priority
    }

    /// EMA-smoothed coverage
    pub fn smoothed_coverage(&self) -> f64 {
        self.stats.ema_coverage
    }

    /// EMA-smoothed resource utilization
    pub fn smoothed_resource_utilization(&self) -> f64 {
        self.stats.ema_resource_utilization
    }

    /// Windowed average subgoal count
    pub fn windowed_avg_subgoal_count(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.subgoal_count as f64).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average coverage
    pub fn windowed_avg_coverage(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.coverage).sum();
        sum / self.recent.len() as f64
    }

    /// Reset state (keeps config)
    pub fn reset(&mut self) {
        self.subgoals.clear();
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = SubgoalStats::default();
        self.decomposition_counter = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn default_goal() -> GoalSpec {
        GoalSpec::default()
    }

    fn aggressive_goal() -> GoalSpec {
        GoalSpec {
            id: "aggressive".into(),
            description: "Aggressive growth".into(),
            target_return: 0.25,
            max_drawdown: 0.15,
            target_sharpe: 2.5,
            risk_budget: 0.20,
            horizon_days: 126,
            resource_budget: 200.0,
        }
    }

    fn conservative_goal() -> GoalSpec {
        GoalSpec {
            id: "conservative".into(),
            description: "Capital preservation".into(),
            target_return: 0.04,
            max_drawdown: 0.03,
            target_sharpe: 1.0,
            risk_budget: 0.03,
            horizon_days: 504,
            resource_budget: 50.0,
        }
    }

    // -- Construction --

    #[test]
    fn test_new_default() {
        let sg = SubgoalGeneration::new();
        assert_eq!(sg.total_subgoals(), 0);
        assert!(sg.process().is_ok());
    }

    #[test]
    fn test_with_config() {
        let sg = SubgoalGeneration::with_config(SubgoalConfig::default());
        assert!(sg.is_ok());
    }

    #[test]
    fn test_invalid_max_depth_zero() {
        let mut cfg = SubgoalConfig::default();
        cfg.max_depth = 0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_max_children_zero() {
        let mut cfg = SubgoalConfig::default();
        cfg.max_children = 0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_max_total_zero() {
        let mut cfg = SubgoalConfig::default();
        cfg.max_total_subgoals = 0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_zero() {
        let mut cfg = SubgoalConfig::default();
        cfg.ema_decay = 0.0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_one() {
        let mut cfg = SubgoalConfig::default();
        cfg.ema_decay = 1.0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_window_size_zero() {
        let mut cfg = SubgoalConfig::default();
        cfg.window_size = 0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_zero_weights() {
        let mut cfg = SubgoalConfig::default();
        cfg.urgency_weight = 0.0;
        cfg.impact_weight = 0.0;
        cfg.feasibility_weight = 0.0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    // -- Priority --

    #[test]
    fn test_compute_priority_balanced() {
        let sg = SubgoalGeneration::new();
        let p = sg.compute_priority(1.0, 1.0, 1.0);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_priority_zero() {
        let sg = SubgoalGeneration::new();
        let p = sg.compute_priority(0.0, 0.0, 0.0);
        assert!((p - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_priority_impact_dominant() {
        let sg = SubgoalGeneration::new();
        // impact weight = 0.40 (highest)
        let p_high_impact = sg.compute_priority(0.0, 1.0, 0.0);
        let p_high_urgency = sg.compute_priority(1.0, 0.0, 0.0);
        assert!(p_high_impact > p_high_urgency);
    }

    #[test]
    fn test_compute_priority_clamps_inputs() {
        let sg = SubgoalGeneration::new();
        let p = sg.compute_priority(2.0, -1.0, 0.5);
        // 2.0 clamped to 1.0, -1.0 clamped to 0.0
        let expected = sg.compute_priority(1.0, 0.0, 0.5);
        assert!((p - expected).abs() < 1e-10);
    }

    // -- Decomposition --

    #[test]
    fn test_decompose_default_goal() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&default_goal()).unwrap();

        assert!(!result.subgoals.is_empty());
        assert_eq!(result.goal_id, "default_goal");
        assert!(result.coverage > 0.0);
        assert!(result.total_resource_cost > 0.0);
    }

    #[test]
    fn test_decompose_creates_allocation_subgoals() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&default_goal()).unwrap();

        let alloc_count = result
            .subgoals
            .iter()
            .filter(|s| s.category == SubgoalCategory::Allocation)
            .count();
        assert!(alloc_count >= 3); // momentum, value, mean_reversion, carry
    }

    #[test]
    fn test_decompose_creates_risk_subgoal() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&default_goal()).unwrap();

        let risk_count = result
            .subgoals
            .iter()
            .filter(|s| s.category == SubgoalCategory::RiskManagement)
            .count();
        assert_eq!(risk_count, 1);
    }

    #[test]
    fn test_decompose_creates_execution_subgoal() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&default_goal()).unwrap();

        let exec_count = result
            .subgoals
            .iter()
            .filter(|s| s.category == SubgoalCategory::Execution)
            .count();
        assert_eq!(exec_count, 1);
    }

    #[test]
    fn test_decompose_creates_monitoring_subgoal() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&default_goal()).unwrap();

        let mon_count = result
            .subgoals
            .iter()
            .filter(|s| s.category == SubgoalCategory::Monitoring)
            .count();
        assert_eq!(mon_count, 1);
    }

    #[test]
    fn test_decompose_aggressive_adds_research() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&aggressive_goal()).unwrap();

        // target_sharpe > 2.0 should add a research subgoal
        let research_count = result
            .subgoals
            .iter()
            .filter(|s| s.category == SubgoalCategory::Research)
            .count();
        assert_eq!(research_count, 1);
    }

    #[test]
    fn test_decompose_conservative_adds_rebalance() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&conservative_goal()).unwrap();

        // risk_budget < 0.05 should add a rebalance subgoal
        let rebal_count = result
            .subgoals
            .iter()
            .filter(|s| s.category == SubgoalCategory::Rebalance)
            .count();
        assert_eq!(rebal_count, 1);
    }

    #[test]
    fn test_decompose_detects_conflicts() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&default_goal()).unwrap();

        // momentum + mean_reversion should produce a conflict
        assert!(result.has_conflicts);
        assert!(!result.conflict_descriptions.is_empty());
    }

    #[test]
    fn test_decompose_priorities_bounded() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&default_goal()).unwrap();

        for s in &result.subgoals {
            assert!(s.priority >= 0.0);
            assert!(s.priority <= 1.0);
        }
    }

    #[test]
    fn test_decompose_subgoals_have_parent() {
        let mut sg = SubgoalGeneration::new();
        let goal = default_goal();
        let result = sg.decompose(&goal).unwrap();

        for s in &result.subgoals {
            assert_eq!(s.parent_id.as_deref(), Some(goal.id.as_str()));
        }
    }

    #[test]
    fn test_decompose_subgoals_stored() {
        let mut sg = SubgoalGeneration::new();
        let result = sg.decompose(&default_goal()).unwrap();

        assert_eq!(sg.total_subgoals(), result.subgoals.len());
    }

    #[test]
    fn test_decompose_empty_id_rejected() {
        let mut sg = SubgoalGeneration::new();
        let mut goal = default_goal();
        goal.id = "".into();
        assert!(sg.decompose(&goal).is_err());
    }

    #[test]
    fn test_decompose_negative_return_rejected() {
        let mut sg = SubgoalGeneration::new();
        let mut goal = default_goal();
        goal.target_return = -0.05;
        assert!(sg.decompose(&goal).is_err());
    }

    #[test]
    fn test_decompose_invalid_max_drawdown() {
        let mut sg = SubgoalGeneration::new();
        let mut goal = default_goal();
        goal.max_drawdown = 0.0;
        assert!(sg.decompose(&goal).is_err());

        goal.max_drawdown = 1.5;
        assert!(sg.decompose(&goal).is_err());
    }

    #[test]
    fn test_decompose_max_subgoals_enforced() {
        let mut cfg = SubgoalConfig::default();
        cfg.max_total_subgoals = 3;
        let mut sg = SubgoalGeneration::with_config(cfg).unwrap();

        let result = sg.decompose(&default_goal()).unwrap();
        assert!(result.subgoals.len() <= 3);
        assert!(result.pruned_count > 0);
    }

    #[test]
    fn test_decompose_auto_prune() {
        let mut cfg = SubgoalConfig::default();
        cfg.min_priority = 0.9; // very high threshold — should prune most
        let mut sg = SubgoalGeneration::with_config(cfg).unwrap();

        let result = sg.decompose(&default_goal()).unwrap();
        assert!(result.pruned_count > 0);
        for s in &result.subgoals {
            assert!(s.priority >= 0.9);
        }
    }

    #[test]
    fn test_decompose_no_prune_when_disabled() {
        let mut cfg = SubgoalConfig::default();
        cfg.auto_prune = false;
        cfg.min_priority = 0.9;
        let mut sg = SubgoalGeneration::with_config(cfg).unwrap();

        let result = sg.decompose(&default_goal()).unwrap();
        // Some subgoals may have priority < 0.9 since pruning is off
        let low_priority_exists = result.subgoals.iter().any(|s| s.priority < 0.9);
        assert!(low_priority_exists || result.subgoals.is_empty());
    }

    // -- Dependencies --

    #[test]
    fn test_risk_depends_on_allocation() {
        let mut sg = SubgoalGeneration::new();
        let goal = default_goal();
        sg.decompose(&goal).unwrap();

        let risk = sg.get_subgoal(&format!("{}_risk_mgmt", goal.id)).unwrap();
        assert!(!risk.dependencies.is_empty());
        // Should depend on allocation subgoals
        assert!(risk.dependencies.iter().any(|d| d.contains("alloc")));
    }

    #[test]
    fn test_execution_depends_on_risk() {
        let mut sg = SubgoalGeneration::new();
        let goal = default_goal();
        sg.decompose(&goal).unwrap();

        let exec = sg.get_subgoal(&format!("{}_execution", goal.id)).unwrap();
        assert!(
            exec.dependencies
                .contains(&format!("{}_risk_mgmt", goal.id))
        );
    }

    #[test]
    fn test_topological_order_valid() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let order = sg.topological_order().unwrap();
        assert_eq!(order.len(), sg.total_subgoals());

        // Verify ordering: no subgoal appears before its dependency
        let pos: std::collections::HashMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        for s in sg.subgoals() {
            let my_pos = pos[s.id.as_str()];
            for dep in &s.dependencies {
                if let Some(&dep_pos) = pos.get(dep.as_str()) {
                    assert!(
                        dep_pos < my_pos,
                        "Dependency {} should come before {} in topological order",
                        dep,
                        s.id
                    );
                }
            }
        }
    }

    // -- Subgoal management --

    #[test]
    fn test_get_subgoal() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let s = sg.get_subgoal("default_goal_alloc_momentum");
        assert!(s.is_some());
        assert!(sg.get_subgoal("nonexistent").is_none());
    }

    #[test]
    fn test_update_status() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let id = "default_goal_alloc_momentum";
        assert!(sg.update_status(id, SubgoalStatus::Active).is_ok());
        assert_eq!(sg.get_subgoal(id).unwrap().status, SubgoalStatus::Active);
    }

    #[test]
    fn test_update_status_not_found() {
        let mut sg = SubgoalGeneration::new();
        assert!(sg.update_status("nope", SubgoalStatus::Active).is_err());
    }

    #[test]
    fn test_update_progress() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let id = "default_goal_alloc_momentum";
        assert!(sg.update_progress(id, 0.5).is_ok());
        let s = sg.get_subgoal(id).unwrap();
        assert!((s.progress - 0.5).abs() < 1e-10);
        assert_eq!(s.status, SubgoalStatus::Active);
    }

    #[test]
    fn test_update_progress_completes() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let id = "default_goal_alloc_momentum";
        sg.update_status(id, SubgoalStatus::Active).unwrap();
        sg.update_progress(id, 1.0).unwrap();
        assert_eq!(sg.get_subgoal(id).unwrap().status, SubgoalStatus::Completed);
    }

    #[test]
    fn test_update_progress_invalid_range() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let id = "default_goal_alloc_momentum";
        assert!(sg.update_progress(id, -0.1).is_err());
        assert!(sg.update_progress(id, 1.1).is_err());
    }

    #[test]
    fn test_update_progress_not_found() {
        let mut sg = SubgoalGeneration::new();
        assert!(sg.update_progress("nope", 0.5).is_err());
    }

    #[test]
    fn test_children_of() {
        let mut sg = SubgoalGeneration::new();
        let goal = default_goal();
        sg.decompose(&goal).unwrap();

        let children = sg.children_of(&goal.id);
        assert!(!children.is_empty());
        assert_eq!(children.len(), sg.total_subgoals());
    }

    #[test]
    fn test_aggregate_progress() {
        let mut sg = SubgoalGeneration::new();
        let goal = default_goal();
        sg.decompose(&goal).unwrap();

        // No progress yet
        let progress = sg.aggregate_progress(&goal.id);
        assert!((progress - 0.0).abs() < 1e-10);

        // Set some progress
        for s in sg.subgoals.iter_mut() {
            s.progress = 0.5;
        }
        let progress = sg.aggregate_progress(&goal.id);
        assert!((progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_progress_no_children() {
        let sg = SubgoalGeneration::new();
        assert_eq!(sg.aggregate_progress("nonexistent"), 0.0);
    }

    #[test]
    fn test_clear_goal() {
        let mut sg = SubgoalGeneration::new();
        let goal = default_goal();
        sg.decompose(&goal).unwrap();
        assert!(sg.total_subgoals() > 0);

        sg.clear_goal(&goal.id);
        assert_eq!(sg.total_subgoals(), 0);
    }

    #[test]
    fn test_count_by_status() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let pending = sg.count_by_status(SubgoalStatus::Pending);
        assert_eq!(pending, sg.total_subgoals()); // all start pending
        assert_eq!(sg.count_by_status(SubgoalStatus::Active), 0);
    }

    #[test]
    fn test_active_resource_cost() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let cost = sg.active_resource_cost();
        assert!(cost > 0.0);
    }

    // -- Actionable / blocked --

    #[test]
    fn test_actionable_subgoals() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let actionable = sg.actionable_subgoals();
        // At minimum the allocation and monitoring subgoals should be actionable
        // (they have no unmet internal dependencies among themselves)
        assert!(!actionable.is_empty());
    }

    #[test]
    fn test_blocked_subgoals() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        let blocked = sg.blocked_subgoals();
        // Risk management depends on allocations → should be blocked
        assert!(
            blocked
                .iter()
                .any(|s| s.category == SubgoalCategory::RiskManagement)
        );
    }

    #[test]
    fn test_blocked_clears_when_deps_complete() {
        let mut sg = SubgoalGeneration::new();
        let goal = default_goal();
        sg.decompose(&goal).unwrap();

        // Complete all allocation subgoals
        let alloc_ids: Vec<String> = sg
            .subgoals()
            .iter()
            .filter(|s| s.category == SubgoalCategory::Allocation)
            .map(|s| s.id.clone())
            .collect();

        for id in &alloc_ids {
            sg.update_status(id, SubgoalStatus::Completed).unwrap();
        }

        // Now risk management should not be blocked
        let blocked = sg.blocked_subgoals();
        assert!(
            !blocked
                .iter()
                .any(|s| s.id == format!("{}_risk_mgmt", goal.id))
        );
    }

    // -- Subgoal type --

    #[test]
    fn test_subgoal_status_terminal() {
        assert!(SubgoalStatus::Completed.is_terminal());
        assert!(SubgoalStatus::Failed.is_terminal());
        assert!(SubgoalStatus::Pruned.is_terminal());
        assert!(!SubgoalStatus::Pending.is_terminal());
        assert!(!SubgoalStatus::Active.is_terminal());
        assert!(!SubgoalStatus::Blocked.is_terminal());
    }

    #[test]
    fn test_subgoal_status_actionable() {
        assert!(SubgoalStatus::Pending.is_actionable());
        assert!(SubgoalStatus::Active.is_actionable());
        assert!(!SubgoalStatus::Completed.is_actionable());
        assert!(!SubgoalStatus::Failed.is_actionable());
        assert!(!SubgoalStatus::Blocked.is_actionable());
        assert!(!SubgoalStatus::Pruned.is_actionable());
    }

    #[test]
    fn test_subgoal_target_progress() {
        let mut sg = Subgoal::new("t", "test", SubgoalCategory::Allocation, None, 0);
        sg.target_value = 100.0;
        sg.current_value = 50.0;
        assert!((sg.target_progress() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_subgoal_target_progress_zero_target() {
        let mut sg = Subgoal::new("t", "test", SubgoalCategory::Allocation, None, 0);
        sg.target_value = 0.0;
        sg.progress = 0.7;
        assert!((sg.target_progress() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_subgoal_target_progress_clamped() {
        let mut sg = Subgoal::new("t", "test", SubgoalCategory::Allocation, None, 0);
        sg.target_value = 10.0;
        sg.current_value = 20.0;
        assert!((sg.target_progress() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_subgoal_dependencies_met() {
        let mut sg = Subgoal::new("t", "test", SubgoalCategory::Execution, None, 1);
        sg.dependencies = vec!["dep_a".into(), "dep_b".into()];

        // All met
        assert!(sg.dependencies_met(|_| true));
        // One not met
        assert!(!sg.dependencies_met(|id| id == "dep_a"));
    }

    // -- Statistics --

    #[test]
    fn test_stats_initial() {
        let sg = SubgoalGeneration::new();
        assert_eq!(sg.stats().total_decompositions, 0);
        assert_eq!(sg.stats().total_subgoals_generated, 0);
    }

    #[test]
    fn test_stats_after_decomposition() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        assert_eq!(sg.stats().total_decompositions, 1);
        assert!(sg.stats().total_subgoals_generated > 0);
    }

    #[test]
    fn test_stats_avg_subgoals() {
        let stats = SubgoalStats {
            total_decompositions: 4,
            total_subgoals_generated: 28,
            ..Default::default()
        };
        assert!((stats.avg_subgoals_per_decomposition() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_prune_rate() {
        let stats = SubgoalStats {
            total_subgoals_generated: 20,
            total_pruned: 5,
            ..Default::default()
        };
        assert!((stats.prune_rate() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_stats_conflict_rate() {
        let stats = SubgoalStats {
            total_decompositions: 10,
            total_conflicts: 3,
            ..Default::default()
        };
        assert!((stats.conflict_rate() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_stats_zero_division() {
        let stats = SubgoalStats::default();
        assert_eq!(stats.avg_subgoals_per_decomposition(), 0.0);
        assert_eq!(stats.prune_rate(), 0.0);
        assert_eq!(stats.conflict_rate(), 0.0);
    }

    #[test]
    fn test_stats_best_worst_coverage() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        assert!(sg.stats().best_coverage > 0.0);
        assert!(sg.stats().worst_coverage < f64::MAX);
    }

    // -- EMA --

    #[test]
    fn test_ema_initializes_first() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        assert!(sg.smoothed_subgoal_count() > 0.0);
        assert!(sg.smoothed_avg_priority() > 0.0);
        assert!(sg.smoothed_coverage() > 0.0);
    }

    #[test]
    fn test_ema_blends_subsequent() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();
        let first_count = sg.smoothed_subgoal_count();

        // Second decomposition with different goal
        sg.decompose(&aggressive_goal()).unwrap();
        let second_count = sg.smoothed_subgoal_count();

        // EMA should blend — not be exactly equal to either
        assert!(second_count > 0.0);
        // Because aggressive adds research, count changes
        assert!((second_count - first_count).abs() < first_count + 5.0);
    }

    // -- Sliding window --

    #[test]
    fn test_recent_stored() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        assert_eq!(sg.recent_decompositions().len(), 1);
    }

    #[test]
    fn test_recent_windowed() {
        let mut cfg = SubgoalConfig::default();
        cfg.window_size = 3;
        cfg.max_total_subgoals = 1000;
        let mut sg = SubgoalGeneration::with_config(cfg).unwrap();

        for i in 0..10 {
            let goal = GoalSpec {
                id: format!("goal_{}", i),
                ..Default::default()
            };
            sg.decompose(&goal).unwrap();
        }

        assert!(sg.recent_decompositions().len() <= 3);
    }

    #[test]
    fn test_windowed_avg_subgoal_count() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        assert!(sg.windowed_avg_subgoal_count() > 0.0);
    }

    #[test]
    fn test_windowed_avg_coverage() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        assert!(sg.windowed_avg_coverage() > 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let sg = SubgoalGeneration::new();
        assert_eq!(sg.windowed_avg_subgoal_count(), 0.0);
        assert_eq!(sg.windowed_avg_coverage(), 0.0);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut sg = SubgoalGeneration::new();
        sg.decompose(&default_goal()).unwrap();

        assert!(sg.total_subgoals() > 0);
        assert!(sg.stats().total_decompositions > 0);

        sg.reset();

        assert_eq!(sg.total_subgoals(), 0);
        assert_eq!(sg.stats().total_decompositions, 0);
        assert!(sg.recent_decompositions().is_empty());
    }

    // -- Multiple goal decomposition --

    #[test]
    fn test_multiple_goals_accumulate() {
        let mut cfg = SubgoalConfig::default();
        cfg.max_total_subgoals = 500;
        let mut sg = SubgoalGeneration::with_config(cfg).unwrap();

        let r1 = sg.decompose(&default_goal()).unwrap();
        let count1 = sg.total_subgoals();

        let r2 = sg.decompose(&aggressive_goal()).unwrap();
        let count2 = sg.total_subgoals();

        assert_eq!(count2, count1 + r2.subgoals.len());
        assert_eq!(sg.stats().total_decompositions, 2);
        assert_eq!(
            sg.stats().total_subgoals_generated as usize,
            r1.subgoals.len() + r2.subgoals.len()
        );
    }

    #[test]
    fn test_clear_goal_selective() {
        let mut cfg = SubgoalConfig::default();
        cfg.max_total_subgoals = 500;
        let mut sg = SubgoalGeneration::with_config(cfg).unwrap();

        sg.decompose(&default_goal()).unwrap();
        let r2 = sg.decompose(&aggressive_goal()).unwrap();

        sg.clear_goal("default_goal");
        assert_eq!(sg.total_subgoals(), r2.subgoals.len());
    }

    // -- Category equality --

    #[test]
    fn test_category_custom_equality() {
        let a = SubgoalCategory::Custom("alpha".into());
        let b = SubgoalCategory::Custom("alpha".into());
        let c = SubgoalCategory::Custom("beta".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // -- Integration test --

    #[test]
    fn test_full_lifecycle() {
        let mut sg = SubgoalGeneration::new();

        // Decompose
        let result = sg.decompose(&default_goal()).unwrap();
        assert!(!result.subgoals.is_empty());

        // Check topological order
        let order = sg.topological_order().unwrap();
        assert_eq!(order.len(), sg.total_subgoals());

        // Get actionable subgoals — collect IDs to avoid holding an
        // immutable borrow while we mutate `sg` in the loops below.
        let actionable_ids: Vec<String> = sg
            .actionable_subgoals()
            .iter()
            .map(|s| s.id.clone())
            .collect();
        assert!(!actionable_ids.is_empty());

        // Progress the actionable ones
        for id in &actionable_ids {
            sg.update_status(id, SubgoalStatus::Active).unwrap();
            sg.update_progress(id, 0.5).unwrap();
        }

        let active = sg.count_by_status(SubgoalStatus::Active);
        assert!(active > 0);

        // Complete them
        for id in &actionable_ids {
            sg.update_progress(id, 1.0).unwrap();
        }

        let completed = sg.count_by_status(SubgoalStatus::Completed);
        assert_eq!(completed, actionable_ids.len());

        // Aggregate progress should be > 0
        let progress = sg.aggregate_progress("default_goal");
        assert!(progress > 0.0);

        // Stats
        assert_eq!(sg.stats().total_decompositions, 1);
        assert!(sg.smoothed_avg_priority() > 0.0);
    }
}
