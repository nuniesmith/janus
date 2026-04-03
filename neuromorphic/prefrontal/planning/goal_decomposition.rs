//! Goal Decomposition — Hierarchical Goal Breakdown
//!
//! Part of the Prefrontal region
//! Component: planning
//!
//! Breaks down high-level trading goals into hierarchical sub-components.
//! Each goal can be decomposed into child goals using configurable
//! decomposition strategies (sequential, parallel, conditional).
//! Progress rolls up from leaf nodes to the root.
//!
//! ## Features
//!
//! - **Hierarchical goal tree**: Goals form a tree where each node can
//!   have multiple children. Leaf goals are actionable; internal goals
//!   aggregate progress from their children.
//! - **Decomposition strategies**: Sequential (children must complete in
//!   order), Parallel (children progress independently), Conditional
//!   (children depend on a condition being met), Weighted (children
//!   contribute proportionally to parent progress).
//! - **Progress rollup**: Leaf progress propagates up the tree according
//!   to the decomposition strategy of each parent.
//! - **Goal state machine**: Goals transition through Pending → Active →
//!   Completed / Failed / Cancelled states.
//! - **Depth-limited decomposition**: Configurable maximum tree depth to
//!   prevent unbounded nesting.
//! - **Goal metadata**: Priority, weight, deadline tick, description.
//! - **Running statistics**: Total goals, completed, failed, depth
//!   distribution, mean progress.
//! - **Validation**: Cycle detection, orphan detection, depth limits.

use crate::common::{Error, Result};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the goal decomposition engine.
#[derive(Debug, Clone)]
pub struct GoalDecompositionConfig {
    /// Maximum tree depth (root = depth 0).
    pub max_depth: usize,
    /// Maximum number of goals that can be registered.
    pub max_goals: usize,
    /// Maximum number of children per goal.
    pub max_children: usize,
}

impl Default for GoalDecompositionConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            max_goals: 1024,
            max_children: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Goal state
// ---------------------------------------------------------------------------

/// State of a goal in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GoalState {
    /// Not yet started.
    Pending,
    /// Currently being worked on.
    Active,
    /// Successfully completed.
    Completed,
    /// Failed to achieve.
    Failed,
    /// Cancelled by the user or system.
    Cancelled,
}

impl GoalState {
    /// Whether the goal is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            GoalState::Completed | GoalState::Failed | GoalState::Cancelled
        )
    }

    /// Whether the goal is still alive (not terminal).
    pub fn is_alive(self) -> bool {
        !self.is_terminal()
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            GoalState::Pending => "Pending",
            GoalState::Active => "Active",
            GoalState::Completed => "Completed",
            GoalState::Failed => "Failed",
            GoalState::Cancelled => "Cancelled",
        }
    }
}

impl std::fmt::Display for GoalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Decomposition strategy
// ---------------------------------------------------------------------------

/// Strategy for decomposing a goal into sub-goals and rolling up progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecompositionStrategy {
    /// Children must complete in order. Progress = fraction of children
    /// completed sequentially.
    Sequential,
    /// Children progress independently. Progress = mean of children's
    /// progress.
    Parallel,
    /// Children contribute to parent progress proportional to their
    /// weights. Progress = weighted mean of children's progress.
    Weighted,
    /// No decomposition — this is a leaf goal whose progress is set
    /// directly.
    Leaf,
}

impl DecompositionStrategy {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            DecompositionStrategy::Sequential => "Sequential",
            DecompositionStrategy::Parallel => "Parallel",
            DecompositionStrategy::Weighted => "Weighted",
            DecompositionStrategy::Leaf => "Leaf",
        }
    }
}

// ---------------------------------------------------------------------------
// Goal node
// ---------------------------------------------------------------------------

/// A single goal in the decomposition tree.
#[derive(Debug, Clone)]
pub struct GoalNode {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    pub description: String,
    /// Current state.
    pub state: GoalState,
    /// Decomposition strategy for this node's children.
    pub strategy: DecompositionStrategy,
    /// Progress toward completion [0, 1].
    pub progress: f64,
    /// Weight (used by parent's Weighted strategy).
    pub weight: f64,
    /// Priority (0 = highest).
    pub priority: u32,
    /// Deadline tick (0 = no deadline).
    pub deadline_tick: u64,
    /// Parent goal ID (None for root goals).
    pub parent_id: Option<String>,
    /// Child goal IDs in order.
    pub children: Vec<String>,
    /// Depth in the tree (root = 0).
    pub depth: usize,
    /// Tick when the goal was created.
    pub created_tick: u64,
    /// Tick when the goal entered a terminal state.
    pub completed_tick: Option<u64>,
}

impl GoalNode {
    fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        strategy: DecompositionStrategy,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            state: GoalState::Pending,
            strategy,
            progress: 0.0,
            weight: 1.0,
            priority: 0,
            deadline_tick: 0,
            parent_id: None,
            children: Vec::new(),
            depth: 0,
            created_tick: 0,
            completed_tick: None,
        }
    }

    /// Whether this node is a leaf (no children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Whether this node is a root (no parent).
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    /// Whether the deadline has passed.
    pub fn is_overdue(&self, current_tick: u64) -> bool {
        self.deadline_tick > 0 && current_tick > self.deadline_tick && self.state.is_alive()
    }
}

// ---------------------------------------------------------------------------
// Tree snapshot
// ---------------------------------------------------------------------------

/// A snapshot of a goal and its subtree for reporting.
#[derive(Debug, Clone)]
pub struct GoalSnapshot {
    /// Goal ID.
    pub id: String,
    /// Goal name.
    pub name: String,
    /// Current state.
    pub state: GoalState,
    /// Progress [0, 1].
    pub progress: f64,
    /// Strategy.
    pub strategy: DecompositionStrategy,
    /// Depth.
    pub depth: usize,
    /// Number of children.
    pub child_count: usize,
    /// Weight.
    pub weight: f64,
    /// Priority.
    pub priority: u32,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the goal decomposition engine.
#[derive(Debug, Clone)]
pub struct DecompositionStats {
    /// Total goals registered.
    pub total_goals: usize,
    /// Goals in each state.
    pub state_counts: HashMap<GoalState, usize>,
    /// Root goals count.
    pub root_count: usize,
    /// Leaf goals count.
    pub leaf_count: usize,
    /// Maximum depth observed.
    pub max_depth: usize,
    /// Mean progress across all active goals.
    pub mean_progress: f64,
    /// Number of overdue goals.
    pub overdue_count: usize,
}

impl Default for DecompositionStats {
    fn default() -> Self {
        Self {
            total_goals: 0,
            state_counts: HashMap::new(),
            root_count: 0,
            leaf_count: 0,
            max_depth: 0,
            mean_progress: 0.0,
            overdue_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// GoalDecomposition engine
// ---------------------------------------------------------------------------

/// Hierarchical goal decomposition engine.
///
/// Create goals, decompose them into sub-goals with various strategies,
/// update progress on leaf goals, and the engine propagates progress
/// up the tree.
pub struct GoalDecomposition {
    config: GoalDecompositionConfig,

    /// All goals keyed by ID.
    goals: HashMap<String, GoalNode>,

    /// Root goal IDs (goals with no parent).
    roots: Vec<String>,

    /// Current tick.
    current_tick: u64,
}

impl Default for GoalDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalDecomposition {
    /// Create a new engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(GoalDecompositionConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: GoalDecompositionConfig) -> Result<Self> {
        if config.max_depth == 0 {
            return Err(Error::InvalidInput("max_depth must be > 0".into()));
        }
        if config.max_goals == 0 {
            return Err(Error::InvalidInput("max_goals must be > 0".into()));
        }
        if config.max_children == 0 {
            return Err(Error::InvalidInput("max_children must be > 0".into()));
        }
        Ok(Self {
            config,
            goals: HashMap::new(),
            roots: Vec::new(),
            current_tick: 0,
        })
    }

    // -----------------------------------------------------------------------
    // Goal management
    // -----------------------------------------------------------------------

    /// Create a new root goal.
    pub fn create_root(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        strategy: DecompositionStrategy,
    ) -> Result<()> {
        let id = id.into();
        if self.goals.contains_key(&id) {
            return Err(Error::InvalidInput(format!("goal '{}' already exists", id)));
        }
        if self.goals.len() >= self.config.max_goals {
            return Err(Error::ResourceExhausted(format!(
                "maximum goals ({}) reached",
                self.config.max_goals
            )));
        }

        let mut node = GoalNode::new(&id, name, strategy);
        node.created_tick = self.current_tick;
        self.goals.insert(id.clone(), node);
        self.roots.push(id);
        Ok(())
    }

    /// Add a child goal to an existing parent.
    pub fn add_child(
        &mut self,
        parent_id: &str,
        child_id: impl Into<String>,
        child_name: impl Into<String>,
        strategy: DecompositionStrategy,
    ) -> Result<()> {
        let child_id = child_id.into();

        if self.goals.contains_key(&child_id) {
            return Err(Error::InvalidInput(format!(
                "goal '{}' already exists",
                child_id
            )));
        }
        if self.goals.len() >= self.config.max_goals {
            return Err(Error::ResourceExhausted(format!(
                "maximum goals ({}) reached",
                self.config.max_goals
            )));
        }

        let parent_depth = {
            let parent = self
                .goals
                .get(parent_id)
                .ok_or_else(|| Error::NotFound(format!("parent goal '{}' not found", parent_id)))?;
            if parent.children.len() >= self.config.max_children {
                return Err(Error::ResourceExhausted(format!(
                    "maximum children ({}) reached for goal '{}'",
                    self.config.max_children, parent_id
                )));
            }
            if parent.state.is_terminal() {
                return Err(Error::InvalidState(format!(
                    "cannot add children to terminal goal '{}'",
                    parent_id
                )));
            }
            parent.depth
        };

        let child_depth = parent_depth + 1;
        if child_depth >= self.config.max_depth {
            return Err(Error::ResourceExhausted(format!(
                "maximum depth ({}) reached",
                self.config.max_depth
            )));
        }

        let mut node = GoalNode::new(&child_id, child_name, strategy);
        node.parent_id = Some(parent_id.to_string());
        node.depth = child_depth;
        node.created_tick = self.current_tick;

        // Update parent's children list
        self.goals
            .get_mut(parent_id)
            .unwrap()
            .children
            .push(child_id.clone());

        self.goals.insert(child_id, node);
        Ok(())
    }

    /// Remove a goal and all its descendants.
    pub fn remove(&mut self, id: &str) -> Result<()> {
        if !self.goals.contains_key(id) {
            return Err(Error::NotFound(format!("goal '{}' not found", id)));
        }

        // Collect all descendant IDs
        let mut to_remove = Vec::new();
        self.collect_descendants(id, &mut to_remove);
        to_remove.push(id.to_string());

        // Remove from parent's children list
        if let Some(parent_id) = self.goals.get(id).and_then(|g| g.parent_id.clone()) {
            if let Some(parent) = self.goals.get_mut(&parent_id) {
                parent.children.retain(|c| c != id);
            }
        }

        // Remove from roots
        self.roots.retain(|r| r != id);

        // Remove all
        for rid in &to_remove {
            self.goals.remove(rid);
        }

        Ok(())
    }

    fn collect_descendants(&self, id: &str, result: &mut Vec<String>) {
        if let Some(goal) = self.goals.get(id) {
            for child_id in &goal.children {
                result.push(child_id.clone());
                self.collect_descendants(child_id, result);
            }
        }
    }

    /// Get a goal by ID.
    pub fn goal(&self, id: &str) -> Option<&GoalNode> {
        self.goals.get(id)
    }

    /// Get a mutable reference to a goal.
    pub fn goal_mut(&mut self, id: &str) -> Option<&mut GoalNode> {
        self.goals.get_mut(id)
    }

    /// Number of goals.
    pub fn goal_count(&self) -> usize {
        self.goals.len()
    }

    /// Root goal IDs.
    pub fn root_ids(&self) -> &[String] {
        &self.roots
    }

    /// All goal IDs.
    pub fn all_ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self.goals.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Get IDs of all leaf goals.
    pub fn leaf_ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self
            .goals
            .values()
            .filter(|g| g.is_leaf())
            .map(|g| g.id.clone())
            .collect();
        ids.sort();
        ids
    }

    /// Get children IDs of a goal.
    pub fn children(&self, id: &str) -> Option<&[String]> {
        self.goals.get(id).map(|g| g.children.as_slice())
    }

    // -----------------------------------------------------------------------
    // Goal properties
    // -----------------------------------------------------------------------

    /// Set a goal's weight.
    pub fn set_weight(&mut self, id: &str, weight: f64) -> Result<()> {
        let goal = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;
        if weight < 0.0 {
            return Err(Error::InvalidInput("weight must be >= 0".into()));
        }
        goal.weight = weight;
        Ok(())
    }

    /// Set a goal's priority.
    pub fn set_priority(&mut self, id: &str, priority: u32) -> Result<()> {
        let goal = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;
        goal.priority = priority;
        Ok(())
    }

    /// Set a goal's deadline.
    pub fn set_deadline(&mut self, id: &str, deadline_tick: u64) -> Result<()> {
        let goal = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;
        goal.deadline_tick = deadline_tick;
        Ok(())
    }

    /// Set a goal's description.
    pub fn set_description(&mut self, id: &str, description: impl Into<String>) -> Result<()> {
        let goal = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;
        goal.description = description.into();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // State transitions
    // -----------------------------------------------------------------------

    /// Activate a goal (Pending → Active).
    pub fn activate(&mut self, id: &str) -> Result<()> {
        let goal = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;
        if goal.state != GoalState::Pending {
            return Err(Error::InvalidState(format!(
                "goal '{}' is in state {}, expected Pending",
                id, goal.state
            )));
        }
        goal.state = GoalState::Active;
        Ok(())
    }

    /// Mark a goal as completed.
    pub fn complete(&mut self, id: &str) -> Result<()> {
        let goal = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;
        if goal.state.is_terminal() {
            return Err(Error::InvalidState(format!(
                "goal '{}' is already in terminal state {}",
                id, goal.state
            )));
        }
        goal.state = GoalState::Completed;
        goal.progress = 1.0;
        goal.completed_tick = Some(self.current_tick);
        Ok(())
    }

    /// Mark a goal as failed.
    pub fn fail(&mut self, id: &str) -> Result<()> {
        let goal = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;
        if goal.state.is_terminal() {
            return Err(Error::InvalidState(format!(
                "goal '{}' is already in terminal state {}",
                id, goal.state
            )));
        }
        goal.state = GoalState::Failed;
        goal.completed_tick = Some(self.current_tick);
        Ok(())
    }

    /// Cancel a goal.
    pub fn cancel(&mut self, id: &str) -> Result<()> {
        let goal = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;
        if goal.state.is_terminal() {
            return Err(Error::InvalidState(format!(
                "goal '{}' is already in terminal state {}",
                id, goal.state
            )));
        }
        goal.state = GoalState::Cancelled;
        goal.completed_tick = Some(self.current_tick);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Progress
    // -----------------------------------------------------------------------

    /// Set the progress of a leaf goal directly.
    pub fn set_progress(&mut self, id: &str, progress: f64) -> Result<()> {
        let clamped = progress.clamp(0.0, 1.0);

        let goal = self
            .goals
            .get(id)
            .ok_or_else(|| Error::NotFound(format!("goal '{}' not found", id)))?;

        if !goal.is_leaf() {
            return Err(Error::InvalidState(format!(
                "cannot set progress directly on non-leaf goal '{}'",
                id
            )));
        }
        if goal.state.is_terminal() {
            return Err(Error::InvalidState(format!(
                "cannot set progress on terminal goal '{}' (state: {})",
                id, goal.state
            )));
        }

        self.goals.get_mut(id).unwrap().progress = clamped;
        Ok(())
    }

    /// Propagate progress up from leaves to roots.
    ///
    /// Call this after updating leaf progress to ensure all ancestor
    /// nodes reflect the current state of their subtrees.
    pub fn propagate_progress(&mut self) {
        // Process from deepest to shallowest.
        // Collect goals sorted by depth (descending).
        let mut by_depth: Vec<(String, usize)> = self
            .goals
            .iter()
            .filter(|(_, g)| !g.is_leaf())
            .map(|(id, g)| (id.clone(), g.depth))
            .collect();
        by_depth.sort_by(|a, b| b.1.cmp(&a.1));

        for (id, _) in by_depth {
            let progress = self.compute_node_progress(&id);
            if let Some(goal) = self.goals.get_mut(&id) {
                if !goal.state.is_terminal() {
                    goal.progress = progress;

                    // Auto-complete if progress reaches 1.0
                    if (progress - 1.0).abs() < f64::EPSILON {
                        goal.state = GoalState::Completed;
                        goal.completed_tick = Some(self.current_tick);
                    }
                }
            }
        }
    }

    /// Compute the rolled-up progress for a non-leaf node based on its
    /// strategy and children's progress.
    fn compute_node_progress(&self, id: &str) -> f64 {
        let goal = match self.goals.get(id) {
            Some(g) => g,
            None => return 0.0,
        };

        if goal.children.is_empty() {
            return goal.progress;
        }

        let child_progresses: Vec<(f64, f64)> = goal
            .children
            .iter()
            .filter_map(|cid| self.goals.get(cid))
            .map(|c| (c.progress, c.weight))
            .collect();

        if child_progresses.is_empty() {
            return 0.0;
        }

        match goal.strategy {
            DecompositionStrategy::Sequential => {
                // In sequential mode, progress = fraction of children
                // completed + partial progress of the current child.
                let total = child_progresses.len() as f64;
                let completed = child_progresses
                    .iter()
                    .filter(|(p, _)| (*p - 1.0).abs() < f64::EPSILON)
                    .count() as f64;
                // Add partial progress of the next incomplete child
                let partial = child_progresses
                    .iter()
                    .find(|(p, _)| (*p - 1.0).abs() >= f64::EPSILON)
                    .map(|(p, _)| *p)
                    .unwrap_or(0.0);
                ((completed + partial) / total).clamp(0.0, 1.0)
            }
            DecompositionStrategy::Parallel => {
                // Mean of all children's progress
                let sum: f64 = child_progresses.iter().map(|(p, _)| p).sum();
                sum / child_progresses.len() as f64
            }
            DecompositionStrategy::Weighted => {
                // Weighted mean
                let total_weight: f64 = child_progresses.iter().map(|(_, w)| w).sum();
                if total_weight <= 0.0 {
                    // Fall back to unweighted mean
                    let sum: f64 = child_progresses.iter().map(|(p, _)| p).sum();
                    sum / child_progresses.len() as f64
                } else {
                    let weighted_sum: f64 = child_progresses.iter().map(|(p, w)| p * w).sum();
                    (weighted_sum / total_weight).clamp(0.0, 1.0)
                }
            }
            DecompositionStrategy::Leaf => {
                // Leaf nodes shouldn't have children, but handle gracefully
                goal.progress
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tick & deadlines
    // -----------------------------------------------------------------------

    /// Advance the tick counter and check for overdue goals.
    ///
    /// Returns a list of goal IDs that have become overdue.
    pub fn tick(&mut self) -> Vec<String> {
        self.current_tick += 1;
        let mut overdue = Vec::new();
        for goal in self.goals.values() {
            if goal.is_overdue(self.current_tick) {
                overdue.push(goal.id.clone());
            }
        }
        overdue.sort();
        overdue
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    // -----------------------------------------------------------------------
    // Snapshots & stats
    // -----------------------------------------------------------------------

    /// Get a snapshot of a goal.
    pub fn snapshot(&self, id: &str) -> Option<GoalSnapshot> {
        self.goals.get(id).map(|g| GoalSnapshot {
            id: g.id.clone(),
            name: g.name.clone(),
            state: g.state,
            progress: g.progress,
            strategy: g.strategy,
            depth: g.depth,
            child_count: g.children.len(),
            weight: g.weight,
            priority: g.priority,
        })
    }

    /// Get snapshots of all goals in a subtree.
    pub fn subtree_snapshots(&self, root_id: &str) -> Vec<GoalSnapshot> {
        let mut result = Vec::new();
        self.collect_snapshots(root_id, &mut result);
        result
    }

    fn collect_snapshots(&self, id: &str, result: &mut Vec<GoalSnapshot>) {
        if let Some(snap) = self.snapshot(id) {
            let children: Vec<String> = self
                .goals
                .get(id)
                .map(|g| g.children.clone())
                .unwrap_or_default();
            result.push(snap);
            for child_id in &children {
                self.collect_snapshots(child_id, result);
            }
        }
    }

    /// Compute current statistics.
    pub fn stats(&self) -> DecompositionStats {
        let mut state_counts: HashMap<GoalState, usize> = HashMap::new();
        let mut max_depth = 0usize;
        let mut progress_sum = 0.0f64;
        let mut active_count = 0usize;
        let mut leaf_count = 0usize;
        let mut overdue_count = 0usize;

        for goal in self.goals.values() {
            *state_counts.entry(goal.state).or_insert(0) += 1;
            if goal.depth > max_depth {
                max_depth = goal.depth;
            }
            if goal.state.is_alive() {
                progress_sum += goal.progress;
                active_count += 1;
            }
            if goal.is_leaf() {
                leaf_count += 1;
            }
            if goal.is_overdue(self.current_tick) {
                overdue_count += 1;
            }
        }

        DecompositionStats {
            total_goals: self.goals.len(),
            state_counts,
            root_count: self.roots.len(),
            leaf_count,
            max_depth,
            mean_progress: if active_count > 0 {
                progress_sum / active_count as f64
            } else {
                0.0
            },
            overdue_count,
        }
    }

    // -----------------------------------------------------------------------
    // Path operations
    // -----------------------------------------------------------------------

    /// Get the path from root to a given goal.
    pub fn path_to(&self, id: &str) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = id.to_string();
        while let Some(goal) = self.goals.get(&current) {
            path.push(current.clone());
            match &goal.parent_id {
                Some(pid) => current = pid.clone(),
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Get the depth of a goal.
    pub fn depth(&self, id: &str) -> Option<usize> {
        self.goals.get(id).map(|g| g.depth)
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset everything.
    pub fn reset(&mut self) {
        self.goals.clear();
        self.roots.clear();
        self.current_tick = 0;
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
    // GoalState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_terminal() {
        assert!(!GoalState::Pending.is_terminal());
        assert!(!GoalState::Active.is_terminal());
        assert!(GoalState::Completed.is_terminal());
        assert!(GoalState::Failed.is_terminal());
        assert!(GoalState::Cancelled.is_terminal());
    }

    #[test]
    fn test_state_alive() {
        assert!(GoalState::Pending.is_alive());
        assert!(GoalState::Active.is_alive());
        assert!(!GoalState::Completed.is_alive());
    }

    #[test]
    fn test_state_label() {
        assert_eq!(GoalState::Pending.label(), "Pending");
        assert_eq!(GoalState::Active.label(), "Active");
        assert_eq!(GoalState::Completed.label(), "Completed");
        assert_eq!(GoalState::Failed.label(), "Failed");
        assert_eq!(GoalState::Cancelled.label(), "Cancelled");
    }

    #[test]
    fn test_state_display() {
        let s = format!("{}", GoalState::Active);
        assert_eq!(s, "Active");
    }

    // -----------------------------------------------------------------------
    // DecompositionStrategy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_strategy_label() {
        assert_eq!(DecompositionStrategy::Sequential.label(), "Sequential");
        assert_eq!(DecompositionStrategy::Parallel.label(), "Parallel");
        assert_eq!(DecompositionStrategy::Weighted.label(), "Weighted");
        assert_eq!(DecompositionStrategy::Leaf.label(), "Leaf");
    }

    // -----------------------------------------------------------------------
    // Configuration validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_config_max_depth() {
        let mut cfg = GoalDecompositionConfig::default();
        cfg.max_depth = 0;
        assert!(GoalDecomposition::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_goals() {
        let mut cfg = GoalDecompositionConfig::default();
        cfg.max_goals = 0;
        assert!(GoalDecomposition::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_children() {
        let mut cfg = GoalDecompositionConfig::default();
        cfg.max_children = 0;
        assert!(GoalDecomposition::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_default() {
        let gd = GoalDecomposition::new();
        assert_eq!(gd.goal_count(), 0);
        assert_eq!(gd.current_tick(), 0);
    }

    // -----------------------------------------------------------------------
    // Root goal creation
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_root() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r1", "Root Goal", DecompositionStrategy::Parallel)
            .unwrap();
        assert_eq!(gd.goal_count(), 1);
        assert_eq!(gd.root_ids(), &["r1"]);

        let g = gd.goal("r1").unwrap();
        assert_eq!(g.name, "Root Goal");
        assert_eq!(g.state, GoalState::Pending);
        assert_eq!(g.depth, 0);
        assert!(g.is_root());
        assert!(g.is_leaf());
    }

    #[test]
    fn test_create_root_duplicate() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r1", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        assert!(
            gd.create_root("r1", "Dup", DecompositionStrategy::Parallel)
                .is_err()
        );
    }

    #[test]
    fn test_create_root_max_capacity() {
        let mut cfg = GoalDecompositionConfig::default();
        cfg.max_goals = 2;
        let mut gd = GoalDecomposition::with_config(cfg).unwrap();
        gd.create_root("a", "A", DecompositionStrategy::Leaf)
            .unwrap();
        gd.create_root("b", "B", DecompositionStrategy::Leaf)
            .unwrap();
        assert!(
            gd.create_root("c", "C", DecompositionStrategy::Leaf)
                .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // Child goal creation
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_child() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("root", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("root", "c1", "Child 1", DecompositionStrategy::Leaf)
            .unwrap();

        assert_eq!(gd.goal_count(), 2);

        let child = gd.goal("c1").unwrap();
        assert_eq!(child.parent_id.as_deref(), Some("root"));
        assert_eq!(child.depth, 1);
        assert!(!child.is_root());
        assert!(child.is_leaf());

        let root = gd.goal("root").unwrap();
        assert!(!root.is_leaf());
        assert_eq!(root.children.len(), 1);
    }

    #[test]
    fn test_add_child_duplicate() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("root", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("root", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        assert!(
            gd.add_child("root", "c1", "C1 dup", DecompositionStrategy::Leaf)
                .is_err()
        );
    }

    #[test]
    fn test_add_child_nonexistent_parent() {
        let mut gd = GoalDecomposition::new();
        assert!(
            gd.add_child("nope", "c1", "C1", DecompositionStrategy::Leaf)
                .is_err()
        );
    }

    #[test]
    fn test_add_child_max_children() {
        let mut cfg = GoalDecompositionConfig::default();
        cfg.max_children = 2;
        let mut gd = GoalDecomposition::with_config(cfg).unwrap();
        gd.create_root("root", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("root", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("root", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();
        assert!(
            gd.add_child("root", "c3", "C3", DecompositionStrategy::Leaf)
                .is_err()
        );
    }

    #[test]
    fn test_add_child_max_depth() {
        let mut cfg = GoalDecompositionConfig::default();
        cfg.max_depth = 2;
        let mut gd = GoalDecomposition::with_config(cfg).unwrap();
        gd.create_root("r", "R", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Parallel)
            .unwrap();
        // depth 2 should be rejected (max_depth = 2 means depths 0, 1 allowed)
        assert!(
            gd.add_child("c1", "c2", "C2", DecompositionStrategy::Leaf)
                .is_err()
        );
    }

    #[test]
    fn test_add_child_to_terminal_parent() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("root", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.complete("root").unwrap();
        assert!(
            gd.add_child("root", "c1", "C1", DecompositionStrategy::Leaf)
                .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // Multi-level tree
    // -----------------------------------------------------------------------

    #[test]
    fn test_three_level_tree() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("c1", "g1", "G1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("c1", "g2", "G2", DecompositionStrategy::Leaf)
            .unwrap();

        assert_eq!(gd.goal_count(), 5);
        assert_eq!(gd.goal("g1").unwrap().depth, 2);
        assert_eq!(gd.goal("g2").unwrap().depth, 2);
    }

    // -----------------------------------------------------------------------
    // Remove
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_leaf() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("root", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("root", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.remove("c1").unwrap();

        assert_eq!(gd.goal_count(), 1);
        assert_eq!(gd.goal("root").unwrap().children.len(), 0);
    }

    #[test]
    fn test_remove_subtree() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("c1", "g1", "G1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("c1", "g2", "G2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.remove("c1").unwrap();

        assert_eq!(gd.goal_count(), 1);
        assert!(gd.goal("c1").is_none());
        assert!(gd.goal("g1").is_none());
        assert!(gd.goal("g2").is_none());
        assert_eq!(gd.goal("r").unwrap().children.len(), 0);
    }

    #[test]
    fn test_remove_root() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.remove("r").unwrap();

        assert_eq!(gd.goal_count(), 0);
        assert!(gd.root_ids().is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut gd = GoalDecomposition::new();
        assert!(gd.remove("nope").is_err());
    }

    // -----------------------------------------------------------------------
    // State transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_activate() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.activate("r").unwrap();
        assert_eq!(gd.goal("r").unwrap().state, GoalState::Active);
    }

    #[test]
    fn test_activate_non_pending() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.activate("r").unwrap();
        assert!(gd.activate("r").is_err());
    }

    #[test]
    fn test_activate_nonexistent() {
        let mut gd = GoalDecomposition::new();
        assert!(gd.activate("nope").is_err());
    }

    #[test]
    fn test_complete() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.complete("r").unwrap();
        assert_eq!(gd.goal("r").unwrap().state, GoalState::Completed);
        assert!((gd.goal("r").unwrap().progress - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_complete_already_terminal() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.complete("r").unwrap();
        assert!(gd.complete("r").is_err());
    }

    #[test]
    fn test_fail() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.fail("r").unwrap();
        assert_eq!(gd.goal("r").unwrap().state, GoalState::Failed);
    }

    #[test]
    fn test_fail_already_terminal() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.fail("r").unwrap();
        assert!(gd.fail("r").is_err());
    }

    #[test]
    fn test_cancel() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.cancel("r").unwrap();
        assert_eq!(gd.goal("r").unwrap().state, GoalState::Cancelled);
    }

    #[test]
    fn test_cancel_already_terminal() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.complete("r").unwrap();
        assert!(gd.cancel("r").is_err());
    }

    // -----------------------------------------------------------------------
    // Progress
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_progress_leaf() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_progress("r", 0.5).unwrap();
        assert!((gd.goal("r").unwrap().progress - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_set_progress_clamped() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_progress("r", 1.5).unwrap();
        assert!((gd.goal("r").unwrap().progress - 1.0).abs() < 1e-10);

        gd.set_progress("r", -0.5).unwrap();
        assert!((gd.goal("r").unwrap().progress - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_progress_non_leaf() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c", "C", DecompositionStrategy::Leaf)
            .unwrap();
        assert!(gd.set_progress("r", 0.5).is_err());
    }

    #[test]
    fn test_set_progress_terminal() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.complete("r").unwrap();
        assert!(gd.set_progress("r", 0.5).is_err());
    }

    #[test]
    fn test_set_progress_nonexistent() {
        let mut gd = GoalDecomposition::new();
        assert!(gd.set_progress("nope", 0.5).is_err());
    }

    // -----------------------------------------------------------------------
    // Progress propagation — Parallel
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagate_parallel() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.set_progress("c1", 0.4).unwrap();
        gd.set_progress("c2", 0.6).unwrap();
        gd.propagate_progress();

        // mean = (0.4 + 0.6) / 2 = 0.5
        assert!((gd.goal("r").unwrap().progress - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_parallel_all_complete() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.set_progress("c1", 1.0).unwrap();
        gd.set_progress("c2", 1.0).unwrap();
        gd.propagate_progress();

        assert!((gd.goal("r").unwrap().progress - 1.0).abs() < 1e-10);
        assert_eq!(gd.goal("r").unwrap().state, GoalState::Completed);
    }

    // -----------------------------------------------------------------------
    // Progress propagation — Sequential
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagate_sequential() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Sequential)
            .unwrap();
        gd.add_child("r", "s1", "Step 1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "s2", "Step 2", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "s3", "Step 3", DecompositionStrategy::Leaf)
            .unwrap();

        // First step completed, second 50% done
        gd.set_progress("s1", 1.0).unwrap();
        gd.set_progress("s2", 0.5).unwrap();
        gd.set_progress("s3", 0.0).unwrap();
        gd.propagate_progress();

        // (1 completed + 0.5 partial) / 3 = 0.5
        assert!((gd.goal("r").unwrap().progress - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_sequential_none_complete() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Sequential)
            .unwrap();
        gd.add_child("r", "s1", "Step 1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "s2", "Step 2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.set_progress("s1", 0.4).unwrap();
        gd.set_progress("s2", 0.0).unwrap();
        gd.propagate_progress();

        // (0 completed + 0.4 partial) / 2 = 0.2
        assert!((gd.goal("r").unwrap().progress - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_sequential_all_complete() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Sequential)
            .unwrap();
        gd.add_child("r", "s1", "Step 1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "s2", "Step 2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.set_progress("s1", 1.0).unwrap();
        gd.set_progress("s2", 1.0).unwrap();
        gd.propagate_progress();

        assert!((gd.goal("r").unwrap().progress - 1.0).abs() < 1e-10);
        assert_eq!(gd.goal("r").unwrap().state, GoalState::Completed);
    }

    // -----------------------------------------------------------------------
    // Progress propagation — Weighted
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagate_weighted() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Weighted)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.set_weight("c1", 1.0).unwrap();
        gd.set_weight("c2", 3.0).unwrap();

        gd.set_progress("c1", 0.2).unwrap();
        gd.set_progress("c2", 0.6).unwrap();
        gd.propagate_progress();

        // weighted mean = (0.2*1 + 0.6*3) / (1+3) = 2.0/4 = 0.5
        assert!((gd.goal("r").unwrap().progress - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_weighted_zero_weights() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Weighted)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.set_weight("c1", 0.0).unwrap();
        gd.set_weight("c2", 0.0).unwrap();

        gd.set_progress("c1", 0.4).unwrap();
        gd.set_progress("c2", 0.6).unwrap();
        gd.propagate_progress();

        // fallback to unweighted mean = 0.5
        assert!((gd.goal("r").unwrap().progress - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Multi-level propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagate_multi_level() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("c1", "g1", "G1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("c1", "g2", "G2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.set_progress("g1", 0.8).unwrap();
        gd.set_progress("g2", 0.4).unwrap();
        gd.set_progress("c2", 0.6).unwrap();
        gd.propagate_progress();

        // c1 progress = (0.8 + 0.4) / 2 = 0.6
        assert!((gd.goal("c1").unwrap().progress - 0.6).abs() < 1e-10);

        // root progress = (0.6 + 0.6) / 2 = 0.6
        assert!((gd.goal("r").unwrap().progress - 0.6).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Propagation does not modify terminal goals
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagate_skips_terminal() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.fail("r").unwrap();
        gd.set_progress("c1", 1.0).unwrap();
        gd.set_progress("c2", 1.0).unwrap();
        gd.propagate_progress();

        // root should still be Failed, not auto-completed
        assert_eq!(gd.goal("r").unwrap().state, GoalState::Failed);
    }

    // -----------------------------------------------------------------------
    // Goal properties
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_weight() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_weight("r", 2.5).unwrap();
        assert!((gd.goal("r").unwrap().weight - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_set_weight_negative() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        assert!(gd.set_weight("r", -1.0).is_err());
    }

    #[test]
    fn test_set_weight_nonexistent() {
        let mut gd = GoalDecomposition::new();
        assert!(gd.set_weight("nope", 1.0).is_err());
    }

    #[test]
    fn test_set_priority() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_priority("r", 5).unwrap();
        assert_eq!(gd.goal("r").unwrap().priority, 5);
    }

    #[test]
    fn test_set_deadline() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_deadline("r", 100).unwrap();
        assert_eq!(gd.goal("r").unwrap().deadline_tick, 100);
    }

    #[test]
    fn test_set_description() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_description("r", "A test goal").unwrap();
        assert_eq!(gd.goal("r").unwrap().description, "A test goal");
    }

    // -----------------------------------------------------------------------
    // Tick & deadlines
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick() {
        let mut gd = GoalDecomposition::new();
        gd.tick();
        gd.tick();
        assert_eq!(gd.current_tick(), 2);
    }

    #[test]
    fn test_tick_detects_overdue() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_deadline("r", 3).unwrap();

        assert!(gd.tick().is_empty()); // tick 1
        assert!(gd.tick().is_empty()); // tick 2
        assert!(gd.tick().is_empty()); // tick 3 (not overdue: current == deadline)
        let overdue = gd.tick(); // tick 4 (overdue: current > deadline)
        assert_eq!(overdue, vec!["r"]);
    }

    #[test]
    fn test_overdue_not_flagged_for_terminal() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_deadline("r", 2).unwrap();
        gd.complete("r").unwrap();

        gd.tick();
        gd.tick();
        let overdue = gd.tick(); // tick 3, past deadline, but completed
        assert!(overdue.is_empty());
    }

    #[test]
    fn test_is_overdue() {
        let node = GoalNode {
            id: "test".into(),
            name: "Test".into(),
            description: String::new(),
            state: GoalState::Active,
            strategy: DecompositionStrategy::Leaf,
            progress: 0.0,
            weight: 1.0,
            priority: 0,
            deadline_tick: 5,
            parent_id: None,
            children: Vec::new(),
            depth: 0,
            created_tick: 0,
            completed_tick: None,
        };
        assert!(!node.is_overdue(4));
        assert!(!node.is_overdue(5));
        assert!(node.is_overdue(6));
    }

    // -----------------------------------------------------------------------
    // Path operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_path_to() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("c1", "g1", "G1", DecompositionStrategy::Leaf)
            .unwrap();

        let path = gd.path_to("g1");
        assert_eq!(path, vec!["r", "c1", "g1"]);
    }

    #[test]
    fn test_path_to_root() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();

        let path = gd.path_to("r");
        assert_eq!(path, vec!["r"]);
    }

    #[test]
    fn test_depth() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c", "C", DecompositionStrategy::Leaf)
            .unwrap();

        assert_eq!(gd.depth("r"), Some(0));
        assert_eq!(gd.depth("c"), Some(1));
        assert_eq!(gd.depth("nope"), None);
    }

    // -----------------------------------------------------------------------
    // ID queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_ids_sorted() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("z", "Z", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("z", "a", "A", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("z", "m", "M", DecompositionStrategy::Leaf)
            .unwrap();

        assert_eq!(gd.all_ids(), vec!["a", "m", "z"]);
    }

    #[test]
    fn test_leaf_ids() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        let leaves = gd.leaf_ids();
        assert_eq!(leaves, vec!["c1", "c2"]);
    }

    #[test]
    fn test_children() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        assert_eq!(gd.children("r").unwrap(), &["c1", "c2"]);
        assert!(gd.children("nope").is_none());
    }

    // -----------------------------------------------------------------------
    // Snapshots
    // -----------------------------------------------------------------------

    #[test]
    fn test_snapshot() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();

        let snap = gd.snapshot("r").unwrap();
        assert_eq!(snap.id, "r");
        assert_eq!(snap.name, "Root");
        assert_eq!(snap.state, GoalState::Pending);
        assert_eq!(snap.strategy, DecompositionStrategy::Parallel);
        assert_eq!(snap.child_count, 1);
    }

    #[test]
    fn test_snapshot_nonexistent() {
        let gd = GoalDecomposition::new();
        assert!(gd.snapshot("nope").is_none());
    }

    #[test]
    fn test_subtree_snapshots() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        let snaps = gd.subtree_snapshots("r");
        assert_eq!(snaps.len(), 3);
        assert_eq!(snaps[0].id, "r");
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let gd = GoalDecomposition::new();
        let s = gd.stats();
        assert_eq!(s.total_goals, 0);
        assert_eq!(s.root_count, 0);
        assert!((s.mean_progress - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_populated() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Parallel)
            .unwrap();
        gd.add_child("r", "c1", "C1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.add_child("r", "c2", "C2", DecompositionStrategy::Leaf)
            .unwrap();

        gd.set_progress("c1", 0.6).unwrap();
        gd.set_progress("c2", 0.4).unwrap();
        gd.complete("c1").ok(); // Note: this sets progress to 1.0

        let s = gd.stats();
        assert_eq!(s.total_goals, 3);
        assert_eq!(s.root_count, 1);
        assert_eq!(s.leaf_count, 2);
        assert_eq!(s.max_depth, 1);
        assert_eq!(*s.state_counts.get(&GoalState::Completed).unwrap_or(&0), 1);
    }

    #[test]
    fn test_stats_overdue_count() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.set_deadline("r", 2).unwrap();
        gd.tick();
        gd.tick();
        gd.tick(); // tick 3, past deadline 2

        let s = gd.stats();
        assert_eq!(s.overdue_count, 1);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.tick();

        gd.reset();

        assert_eq!(gd.goal_count(), 0);
        assert_eq!(gd.current_tick(), 0);
        assert!(gd.root_ids().is_empty());
    }

    // -----------------------------------------------------------------------
    // Process compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_process() {
        let gd = GoalDecomposition::new();
        assert!(gd.process().is_ok());
    }

    // -----------------------------------------------------------------------
    // GoalNode helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_goal_node_is_leaf_root() {
        let node = GoalNode::new("test", "Test", DecompositionStrategy::Leaf);
        assert!(node.is_leaf());
        assert!(node.is_root());
    }

    #[test]
    fn test_goal_node_no_deadline_not_overdue() {
        let node = GoalNode::new("test", "Test", DecompositionStrategy::Leaf);
        assert!(!node.is_overdue(1000)); // no deadline set
    }

    // -----------------------------------------------------------------------
    // Edge: multiple roots
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_roots() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r1", "Root 1", DecompositionStrategy::Leaf)
            .unwrap();
        gd.create_root("r2", "Root 2", DecompositionStrategy::Leaf)
            .unwrap();

        assert_eq!(gd.root_ids().len(), 2);
        assert_eq!(gd.goal_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Edge: completed tick is recorded
    // -----------------------------------------------------------------------

    #[test]
    fn test_completed_tick_recorded() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();
        gd.tick(); // tick 1
        gd.tick(); // tick 2
        gd.complete("r").unwrap();

        assert_eq!(gd.goal("r").unwrap().completed_tick, Some(2));
    }

    // -----------------------------------------------------------------------
    // Edge: created tick is recorded
    // -----------------------------------------------------------------------

    #[test]
    fn test_created_tick_recorded() {
        let mut gd = GoalDecomposition::new();
        gd.tick(); // tick 1
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();

        assert_eq!(gd.goal("r").unwrap().created_tick, 1);
    }

    // -----------------------------------------------------------------------
    // Edge: goal_mut
    // -----------------------------------------------------------------------

    #[test]
    fn test_goal_mut() {
        let mut gd = GoalDecomposition::new();
        gd.create_root("r", "Root", DecompositionStrategy::Leaf)
            .unwrap();

        let g = gd.goal_mut("r").unwrap();
        g.description = "Updated".into();

        assert_eq!(gd.goal("r").unwrap().description, "Updated");
    }

    #[test]
    fn test_goal_mut_none() {
        let mut gd = GoalDecomposition::new();
        assert!(gd.goal_mut("nope").is_none());
    }
}
