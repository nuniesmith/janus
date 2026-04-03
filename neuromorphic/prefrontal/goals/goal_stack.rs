//! Goal hierarchy management
//!
//! Part of the Prefrontal region
//! Component: goals
//!
//! This module implements a hierarchical goal stack for managing trading objectives
//! with support for goal decomposition, prioritization, progress tracking, and
//! dynamic goal adjustment based on market conditions.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

/// Goal status indicating current state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    /// Goal is pending activation
    Pending,
    /// Goal is currently being pursued
    Active,
    /// Goal has been achieved
    Achieved,
    /// Goal has failed
    Failed,
    /// Goal was abandoned
    Abandoned,
    /// Goal is blocked by dependencies
    Blocked,
    /// Goal is paused
    Paused,
}

/// Goal type categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalType {
    /// Strategic long-term goals (e.g., annual return target)
    Strategic,
    /// Tactical medium-term goals (e.g., monthly profit target)
    Tactical,
    /// Operational short-term goals (e.g., daily P&L target)
    Operational,
    /// Risk management goals (e.g., max drawdown limit)
    RiskManagement,
    /// Learning/improvement goals (e.g., reduce false signals)
    Learning,
    /// System maintenance goals (e.g., uptime target)
    System,
}

/// Priority level for goals
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Critical - must be achieved
    Critical = 4,
    /// High priority
    High = 3,
    /// Medium priority
    Medium = 2,
    /// Low priority
    Low = 1,
    /// Optional - nice to have
    Optional = 0,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Medium
    }
}

/// A goal with associated metadata
#[derive(Debug, Clone)]
pub struct Goal {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Goal type
    pub goal_type: GoalType,
    /// Priority level
    pub priority: Priority,
    /// Current status
    pub status: GoalStatus,
    /// Target value to achieve
    pub target_value: f64,
    /// Current progress value
    pub current_value: f64,
    /// Progress percentage (0.0 - 1.0)
    pub progress: f64,
    /// Deadline timestamp (optional)
    pub deadline: Option<u64>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
    /// Parent goal ID (if this is a subgoal)
    pub parent_id: Option<String>,
    /// Child goal IDs
    pub child_ids: Vec<String>,
    /// Dependency goal IDs (must be achieved first)
    pub dependencies: Vec<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl Goal {
    /// Create a new goal
    pub fn new(id: &str, name: &str, target_value: f64) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            goal_type: GoalType::Tactical,
            priority: Priority::Medium,
            status: GoalStatus::Pending,
            target_value,
            current_value: 0.0,
            progress: 0.0,
            deadline: None,
            created_at: 0,
            updated_at: 0,
            parent_id: None,
            child_ids: Vec::new(),
            dependencies: Vec::new(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a goal builder
    pub fn builder(id: &str, name: &str) -> GoalBuilder {
        GoalBuilder::new(id, name)
    }

    /// Update progress based on current value
    pub fn update_progress(&mut self, current_value: f64) {
        self.current_value = current_value;
        if self.target_value != 0.0 {
            self.progress = (current_value / self.target_value).clamp(0.0, 1.0);
        } else {
            self.progress = if current_value >= 0.0 { 1.0 } else { 0.0 };
        }
    }

    /// Check if goal is complete
    pub fn is_complete(&self) -> bool {
        self.status == GoalStatus::Achieved || self.progress >= 1.0
    }

    /// Check if goal is active
    pub fn is_active(&self) -> bool {
        self.status == GoalStatus::Active
    }

    /// Check if goal has failed or been abandoned
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            GoalStatus::Achieved | GoalStatus::Failed | GoalStatus::Abandoned
        )
    }
}

/// Builder for creating goals
pub struct GoalBuilder {
    goal: Goal,
}

impl GoalBuilder {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            goal: Goal::new(id, name, 0.0),
        }
    }

    pub fn description(mut self, desc: &str) -> Self {
        self.goal.description = desc.to_string();
        self
    }

    pub fn goal_type(mut self, goal_type: GoalType) -> Self {
        self.goal.goal_type = goal_type;
        self
    }

    pub fn priority(mut self, priority: Priority) -> Self {
        self.goal.priority = priority;
        self
    }

    pub fn target(mut self, target: f64) -> Self {
        self.goal.target_value = target;
        self
    }

    pub fn deadline(mut self, deadline: u64) -> Self {
        self.goal.deadline = Some(deadline);
        self
    }

    pub fn parent(mut self, parent_id: &str) -> Self {
        self.goal.parent_id = Some(parent_id.to_string());
        self
    }

    pub fn dependency(mut self, dep_id: &str) -> Self {
        self.goal.dependencies.push(dep_id.to_string());
        self
    }

    pub fn tag(mut self, tag: &str) -> Self {
        self.goal.tags.push(tag.to_string());
        self
    }

    pub fn build(self) -> Goal {
        self.goal
    }
}

/// Configuration for goal stack
#[derive(Debug, Clone)]
pub struct GoalStackConfig {
    /// Maximum number of active goals
    pub max_active_goals: usize,
    /// Maximum goal depth (for hierarchies)
    pub max_depth: usize,
    /// Auto-activate goals when dependencies are met
    pub auto_activate: bool,
    /// Auto-mark goals as achieved when progress hits 100%
    pub auto_complete: bool,
    /// Enable goal decomposition
    pub enable_decomposition: bool,
    /// History retention count
    pub history_retention: usize,
}

impl Default for GoalStackConfig {
    fn default() -> Self {
        Self {
            max_active_goals: 10,
            max_depth: 5,
            auto_activate: true,
            auto_complete: true,
            enable_decomposition: true,
            history_retention: 100,
        }
    }
}

/// Event emitted by the goal stack
#[derive(Debug, Clone)]
pub struct GoalEvent {
    pub goal_id: String,
    pub event_type: GoalEventType,
    pub timestamp: u64,
    pub details: String,
}

/// Types of goal events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalEventType {
    Created,
    Activated,
    ProgressUpdated,
    Achieved,
    Failed,
    Abandoned,
    Blocked,
    Unblocked,
    Decomposed,
    PriorityChanged,
}

/// Goal hierarchy management
pub struct GoalStack {
    /// Configuration
    config: GoalStackConfig,
    /// All goals indexed by ID
    goals: HashMap<String, Goal>,
    /// Active goal IDs in priority order
    active_stack: VecDeque<String>,
    /// Completed goal history
    history: VecDeque<GoalEvent>,
    /// Event log
    events: Vec<GoalEvent>,
    /// Current timestamp
    current_time: u64,
    /// Goal counter for ID generation
    goal_counter: u64,
}

impl Default for GoalStack {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalStack {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_config(GoalStackConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GoalStackConfig) -> Self {
        Self {
            config,
            goals: HashMap::new(),
            active_stack: VecDeque::new(),
            history: VecDeque::new(),
            events: Vec::new(),
            current_time: 0,
            goal_counter: 0,
        }
    }

    /// Set current timestamp
    pub fn set_time(&mut self, timestamp: u64) {
        self.current_time = timestamp;
    }

    /// Generate a unique goal ID
    pub fn generate_id(&mut self) -> String {
        self.goal_counter += 1;
        format!("goal_{}", self.goal_counter)
    }

    /// Add a goal to the stack
    pub fn add_goal(&mut self, mut goal: Goal) -> Result<String> {
        // Set timestamps
        goal.created_at = self.current_time;
        goal.updated_at = self.current_time;

        let id = goal.id.clone();

        // Validate parent exists
        if let Some(ref parent_id) = goal.parent_id {
            if !self.goals.contains_key(parent_id) {
                return Err(Error::NotFound(format!(
                    "Parent goal {} not found",
                    parent_id
                )));
            }
            // Add this as child of parent
            if let Some(parent) = self.goals.get_mut(parent_id) {
                parent.child_ids.push(id.clone());
            }
        }

        // Validate dependencies exist
        for dep_id in &goal.dependencies {
            if !self.goals.contains_key(dep_id) {
                return Err(Error::NotFound(format!(
                    "Dependency goal {} not found",
                    dep_id
                )));
            }
        }

        // Check if dependencies are met
        let deps_met = self.check_dependencies(&goal.dependencies);
        if !deps_met {
            goal.status = GoalStatus::Blocked;
        }

        self.emit_event(&id, GoalEventType::Created, "Goal created");
        self.goals.insert(id.clone(), goal);

        Ok(id)
    }

    /// Activate a goal
    pub fn activate_goal(&mut self, goal_id: &str) -> Result<()> {
        // First, get the dependencies and check terminal status without holding mutable borrow
        let (is_terminal, dependencies) = {
            let goal = self
                .goals
                .get(goal_id)
                .ok_or_else(|| Error::NotFound(format!("Goal {} not found", goal_id)))?;
            (goal.is_terminal(), goal.dependencies.clone())
        };

        if is_terminal {
            return Err(Error::InvalidState(format!(
                "Cannot activate terminal goal {}",
                goal_id
            )));
        }

        // Check dependencies
        let deps_met = self.check_dependencies(&dependencies);
        if !deps_met {
            if let Some(goal) = self.goals.get_mut(goal_id) {
                goal.status = GoalStatus::Blocked;
            }
            return Err(Error::InvalidState(format!(
                "Goal {} has unmet dependencies",
                goal_id
            )));
        }

        // Now get mutable borrow to update the goal
        let goal = self
            .goals
            .get_mut(goal_id)
            .ok_or_else(|| Error::NotFound(format!("Goal {} not found", goal_id)))?;

        // Check max active goals
        if self.active_stack.len() >= self.config.max_active_goals
            && !self.active_stack.contains(&goal_id.to_string())
        {
            return Err(Error::InvalidState(format!(
                "Maximum active goals ({}) reached",
                self.config.max_active_goals
            )));
        }

        goal.status = GoalStatus::Active;
        goal.updated_at = self.current_time;

        if !self.active_stack.contains(&goal_id.to_string()) {
            // Insert based on priority
            let priority = goal.priority;
            let insert_pos = self
                .active_stack
                .iter()
                .position(|id| {
                    self.goals
                        .get(id)
                        .map(|g| g.priority < priority)
                        .unwrap_or(true)
                })
                .unwrap_or(self.active_stack.len());

            self.active_stack.insert(insert_pos, goal_id.to_string());
        }

        self.emit_event(goal_id, GoalEventType::Activated, "Goal activated");
        Ok(())
    }

    /// Update goal progress
    pub fn update_progress(&mut self, goal_id: &str, current_value: f64) -> Result<()> {
        let goal = self
            .goals
            .get_mut(goal_id)
            .ok_or_else(|| Error::NotFound(format!("Goal {} not found", goal_id)))?;

        let old_progress = goal.progress;
        goal.update_progress(current_value);
        goal.updated_at = self.current_time;

        let new_progress = goal.progress;
        let _ = goal; // Explicitly release the borrow

        self.emit_event(
            goal_id,
            GoalEventType::ProgressUpdated,
            &format!(
                "Progress: {:.1}% -> {:.1}%",
                old_progress * 100.0,
                new_progress * 100.0
            ),
        );

        // Auto-complete if enabled
        if self.config.auto_complete && new_progress >= 1.0 {
            self.mark_achieved(goal_id)?;
        }

        Ok(())
    }

    /// Mark a goal as achieved
    pub fn mark_achieved(&mut self, goal_id: &str) -> Result<()> {
        // First, update the goal and extract parent_id before releasing borrow
        let parent_id = {
            let goal = self
                .goals
                .get_mut(goal_id)
                .ok_or_else(|| Error::NotFound(format!("Goal {} not found", goal_id)))?;

            goal.status = GoalStatus::Achieved;
            goal.progress = 1.0;
            goal.updated_at = self.current_time;

            goal.parent_id.clone()
        };

        // Remove from active stack
        self.active_stack.retain(|id| id != goal_id);

        self.emit_event(goal_id, GoalEventType::Achieved, "Goal achieved");

        // Check if this unblocks other goals
        if self.config.auto_activate {
            self.check_and_activate_blocked_goals();
        }

        // Update parent progress if applicable
        if let Some(parent_id) = parent_id {
            self.update_parent_progress(&parent_id)?;
        }

        Ok(())
    }

    /// Mark a goal as failed
    pub fn mark_failed(&mut self, goal_id: &str, reason: &str) -> Result<()> {
        {
            let goal = self
                .goals
                .get_mut(goal_id)
                .ok_or_else(|| Error::NotFound(format!("Goal {} not found", goal_id)))?;

            goal.status = GoalStatus::Failed;
            goal.updated_at = self.current_time;
        }

        // Remove from active stack
        self.active_stack.retain(|id| id != goal_id);

        self.emit_event(goal_id, GoalEventType::Failed, reason);
        Ok(())
    }

    /// Abandon a goal
    pub fn abandon_goal(&mut self, goal_id: &str, reason: &str) -> Result<()> {
        {
            let goal = self
                .goals
                .get_mut(goal_id)
                .ok_or_else(|| Error::NotFound(format!("Goal {} not found", goal_id)))?;

            goal.status = GoalStatus::Abandoned;
            goal.updated_at = self.current_time;
        }

        // Remove from active stack
        self.active_stack.retain(|id| id != goal_id);

        self.emit_event(goal_id, GoalEventType::Abandoned, reason);
        Ok(())
    }

    /// Decompose a goal into subgoals
    pub fn decompose(&mut self, goal_id: &str, subgoals: Vec<Goal>) -> Result<Vec<String>> {
        if !self.config.enable_decomposition {
            return Err(Error::InvalidState("Decomposition is disabled".to_string()));
        }

        // Validate goal exists
        if !self.goals.contains_key(goal_id) {
            return Err(Error::NotFound(format!("Goal {} not found", goal_id)));
        }

        // Check depth
        let current_depth = self.get_goal_depth(goal_id);
        if current_depth >= self.config.max_depth {
            return Err(Error::InvalidState(format!(
                "Maximum depth {} reached",
                self.config.max_depth
            )));
        }

        let mut subgoal_ids = Vec::new();

        for mut subgoal in subgoals {
            subgoal.parent_id = Some(goal_id.to_string());
            let id = self.add_goal(subgoal)?;
            subgoal_ids.push(id);
        }

        self.emit_event(
            goal_id,
            GoalEventType::Decomposed,
            &format!("Decomposed into {} subgoals", subgoal_ids.len()),
        );

        Ok(subgoal_ids)
    }

    /// Get goal by ID
    pub fn get_goal(&self, goal_id: &str) -> Option<&Goal> {
        self.goals.get(goal_id)
    }

    /// Get mutable goal by ID
    pub fn get_goal_mut(&mut self, goal_id: &str) -> Option<&mut Goal> {
        self.goals.get_mut(goal_id)
    }

    /// Get all active goals
    pub fn active_goals(&self) -> Vec<&Goal> {
        self.active_stack
            .iter()
            .filter_map(|id| self.goals.get(id))
            .collect()
    }

    /// Get highest priority active goal
    pub fn top_goal(&self) -> Option<&Goal> {
        self.active_stack.front().and_then(|id| self.goals.get(id))
    }

    /// Get goals by type
    pub fn goals_by_type(&self, goal_type: GoalType) -> Vec<&Goal> {
        self.goals
            .values()
            .filter(|g| g.goal_type == goal_type)
            .collect()
    }

    /// Get goals by status
    pub fn goals_by_status(&self, status: GoalStatus) -> Vec<&Goal> {
        self.goals.values().filter(|g| g.status == status).collect()
    }

    /// Get child goals
    pub fn get_children(&self, goal_id: &str) -> Vec<&Goal> {
        if let Some(goal) = self.goals.get(goal_id) {
            goal.child_ids
                .iter()
                .filter_map(|id| self.goals.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get goal depth in hierarchy
    fn get_goal_depth(&self, goal_id: &str) -> usize {
        let mut depth = 0;
        let mut current_id = goal_id.to_string();

        while let Some(goal) = self.goals.get(&current_id) {
            if let Some(ref parent_id) = goal.parent_id {
                depth += 1;
                current_id = parent_id.clone();
            } else {
                break;
            }
        }

        depth
    }

    /// Check if all dependencies are met
    fn check_dependencies(&self, dependencies: &[String]) -> bool {
        dependencies.iter().all(|dep_id| {
            self.goals
                .get(dep_id)
                .map(|g| g.status == GoalStatus::Achieved)
                .unwrap_or(false)
        })
    }

    /// Check and activate blocked goals whose dependencies are now met
    fn check_and_activate_blocked_goals(&mut self) {
        let blocked_ids: Vec<String> = self
            .goals
            .iter()
            .filter(|(_, g)| g.status == GoalStatus::Blocked)
            .map(|(id, _)| id.clone())
            .collect();

        for goal_id in blocked_ids {
            if let Some(goal) = self.goals.get(&goal_id) {
                if self.check_dependencies(&goal.dependencies) {
                    // Dependencies met, try to activate
                    let _ = self.activate_goal(&goal_id);
                }
            }
        }
    }

    /// Update parent goal progress based on children
    fn update_parent_progress(&mut self, parent_id: &str) -> Result<()> {
        let children: Vec<(f64, GoalStatus)> = self
            .goals
            .get(parent_id)
            .map(|p| {
                p.child_ids
                    .iter()
                    .filter_map(|id| self.goals.get(id))
                    .map(|c| (c.progress, c.status))
                    .collect()
            })
            .unwrap_or_default();

        if children.is_empty() {
            return Ok(());
        }

        let total_progress: f64 = children.iter().map(|(p, _)| p).sum();
        let avg_progress = total_progress / children.len() as f64;

        if let Some(parent) = self.goals.get_mut(parent_id) {
            parent.progress = avg_progress;
            parent.updated_at = self.current_time;

            // If all children achieved, mark parent achieved
            if children.iter().all(|(_, s)| *s == GoalStatus::Achieved) {
                parent.status = GoalStatus::Achieved;
                self.active_stack.retain(|id| id != parent_id);
                self.emit_event(parent_id, GoalEventType::Achieved, "All subgoals achieved");
            }
        }

        Ok(())
    }

    /// Emit a goal event
    fn emit_event(&mut self, goal_id: &str, event_type: GoalEventType, details: &str) {
        let event = GoalEvent {
            goal_id: goal_id.to_string(),
            event_type,
            timestamp: self.current_time,
            details: details.to_string(),
        };

        self.events.push(event.clone());
        self.history.push_back(event);

        // Trim history
        while self.history.len() > self.config.history_retention {
            self.history.pop_front();
        }
    }

    /// Get recent events
    pub fn recent_events(&self, count: usize) -> Vec<&GoalEvent> {
        self.events.iter().rev().take(count).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> GoalStackStats {
        let total = self.goals.len();
        let active = self.goals_by_status(GoalStatus::Active).len();
        let achieved = self.goals_by_status(GoalStatus::Achieved).len();
        let failed = self.goals_by_status(GoalStatus::Failed).len();
        let blocked = self.goals_by_status(GoalStatus::Blocked).len();

        let avg_progress: f64 = if !self.goals.is_empty() {
            self.goals.values().map(|g| g.progress).sum::<f64>() / total as f64
        } else {
            0.0
        };

        GoalStackStats {
            total_goals: total,
            active_goals: active,
            achieved_goals: achieved,
            failed_goals: failed,
            blocked_goals: blocked,
            average_progress: avg_progress,
            event_count: self.events.len(),
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Statistics for goal stack
#[derive(Debug, Clone)]
pub struct GoalStackStats {
    pub total_goals: usize,
    pub active_goals: usize,
    pub achieved_goals: usize,
    pub failed_goals: usize,
    pub blocked_goals: usize,
    pub average_progress: f64,
    pub event_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = GoalStack::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_add_goal() {
        let mut stack = GoalStack::new();
        let goal = Goal::new("g1", "Test Goal", 100.0);
        let id = stack.add_goal(goal).unwrap();
        assert_eq!(id, "g1");
        assert!(stack.get_goal("g1").is_some());
    }

    #[test]
    fn test_goal_builder() {
        let goal = Goal::builder("g1", "Profit Target")
            .description("Achieve monthly profit target")
            .goal_type(GoalType::Tactical)
            .priority(Priority::High)
            .target(10000.0)
            .deadline(1000000)
            .tag("monthly")
            .tag("profit")
            .build();

        assert_eq!(goal.name, "Profit Target");
        assert_eq!(goal.priority, Priority::High);
        assert_eq!(goal.target_value, 10000.0);
        assert_eq!(goal.tags.len(), 2);
    }

    #[test]
    fn test_activate_goal() {
        let mut stack = GoalStack::new();
        let goal = Goal::new("g1", "Test Goal", 100.0);
        stack.add_goal(goal).unwrap();
        stack.activate_goal("g1").unwrap();

        let g = stack.get_goal("g1").unwrap();
        assert_eq!(g.status, GoalStatus::Active);
        assert_eq!(stack.active_goals().len(), 1);
    }

    #[test]
    fn test_update_progress() {
        let mut stack = GoalStack::new();
        let goal = Goal::new("g1", "Test Goal", 100.0);
        stack.add_goal(goal).unwrap();
        stack.activate_goal("g1").unwrap();
        stack.update_progress("g1", 50.0).unwrap();

        let g = stack.get_goal("g1").unwrap();
        assert!((g.progress - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_auto_complete() {
        let mut stack = GoalStack::new();
        let goal = Goal::new("g1", "Test Goal", 100.0);
        stack.add_goal(goal).unwrap();
        stack.activate_goal("g1").unwrap();
        stack.update_progress("g1", 100.0).unwrap();

        let g = stack.get_goal("g1").unwrap();
        assert_eq!(g.status, GoalStatus::Achieved);
    }

    #[test]
    fn test_priority_ordering() {
        let mut stack = GoalStack::new();

        let low = Goal::builder("g1", "Low")
            .priority(Priority::Low)
            .target(100.0)
            .build();
        let high = Goal::builder("g2", "High")
            .priority(Priority::High)
            .target(100.0)
            .build();
        let medium = Goal::builder("g3", "Medium")
            .priority(Priority::Medium)
            .target(100.0)
            .build();

        stack.add_goal(low).unwrap();
        stack.add_goal(high).unwrap();
        stack.add_goal(medium).unwrap();

        stack.activate_goal("g1").unwrap();
        stack.activate_goal("g2").unwrap();
        stack.activate_goal("g3").unwrap();

        let top = stack.top_goal().unwrap();
        assert_eq!(top.id, "g2"); // High priority
    }

    #[test]
    fn test_dependencies() {
        let mut stack = GoalStack::new();

        let goal1 = Goal::new("g1", "First Goal", 100.0);
        let goal2 = Goal::builder("g2", "Second Goal")
            .target(100.0)
            .dependency("g1")
            .build();

        stack.add_goal(goal1).unwrap();
        stack.add_goal(goal2).unwrap();

        // g2 should be blocked
        let g2 = stack.get_goal("g2").unwrap();
        assert_eq!(g2.status, GoalStatus::Blocked);

        // Complete g1
        stack.activate_goal("g1").unwrap();
        stack.mark_achieved("g1").unwrap();

        // g2 should now be active (auto_activate is on)
        let g2 = stack.get_goal("g2").unwrap();
        assert_eq!(g2.status, GoalStatus::Active);
    }

    #[test]
    fn test_decompose() {
        let mut stack = GoalStack::new();
        let parent = Goal::new("p1", "Parent Goal", 100.0);
        stack.add_goal(parent).unwrap();

        let subgoals = vec![
            Goal::new("s1", "Subgoal 1", 50.0),
            Goal::new("s2", "Subgoal 2", 50.0),
        ];

        let ids = stack.decompose("p1", subgoals).unwrap();
        assert_eq!(ids.len(), 2);

        let parent = stack.get_goal("p1").unwrap();
        assert_eq!(parent.child_ids.len(), 2);

        let child = stack.get_goal("s1").unwrap();
        assert_eq!(child.parent_id, Some("p1".to_string()));
    }

    #[test]
    fn test_parent_progress_update() {
        let mut stack = GoalStack::with_config(GoalStackConfig {
            auto_complete: false, // Disable for this test
            ..Default::default()
        });

        let parent = Goal::new("p1", "Parent Goal", 100.0);
        stack.add_goal(parent).unwrap();
        stack.activate_goal("p1").unwrap();

        let subgoals = vec![
            Goal::new("s1", "Subgoal 1", 50.0),
            Goal::new("s2", "Subgoal 2", 50.0),
        ];
        stack.decompose("p1", subgoals).unwrap();

        // Update child progress
        stack.activate_goal("s1").unwrap();
        stack.activate_goal("s2").unwrap();
        stack.mark_achieved("s1").unwrap();

        let parent = stack.get_goal("p1").unwrap();
        assert!((parent.progress - 0.5).abs() < 0.001);

        // Complete second child
        stack.mark_achieved("s2").unwrap();

        let parent = stack.get_goal("p1").unwrap();
        assert_eq!(parent.status, GoalStatus::Achieved);
    }

    #[test]
    fn test_mark_failed() {
        let mut stack = GoalStack::new();
        let goal = Goal::new("g1", "Test Goal", 100.0);
        stack.add_goal(goal).unwrap();
        stack.activate_goal("g1").unwrap();
        stack.mark_failed("g1", "Target not achievable").unwrap();

        let g = stack.get_goal("g1").unwrap();
        assert_eq!(g.status, GoalStatus::Failed);
        assert!(stack.active_goals().is_empty());
    }

    #[test]
    fn test_abandon_goal() {
        let mut stack = GoalStack::new();
        let goal = Goal::new("g1", "Test Goal", 100.0);
        stack.add_goal(goal).unwrap();
        stack.activate_goal("g1").unwrap();
        stack.abandon_goal("g1", "Changed priorities").unwrap();

        let g = stack.get_goal("g1").unwrap();
        assert_eq!(g.status, GoalStatus::Abandoned);
    }

    #[test]
    fn test_goals_by_type() {
        let mut stack = GoalStack::new();

        let tactical = Goal::builder("g1", "Tactical")
            .goal_type(GoalType::Tactical)
            .target(100.0)
            .build();
        let risk = Goal::builder("g2", "Risk")
            .goal_type(GoalType::RiskManagement)
            .target(100.0)
            .build();

        stack.add_goal(tactical).unwrap();
        stack.add_goal(risk).unwrap();

        let tactical_goals = stack.goals_by_type(GoalType::Tactical);
        assert_eq!(tactical_goals.len(), 1);
        assert_eq!(tactical_goals[0].id, "g1");
    }

    #[test]
    fn test_max_active_goals() {
        let config = GoalStackConfig {
            max_active_goals: 2,
            ..Default::default()
        };
        let mut stack = GoalStack::with_config(config);

        for i in 0..3 {
            let goal = Goal::new(&format!("g{}", i), &format!("Goal {}", i), 100.0);
            stack.add_goal(goal).unwrap();
        }

        stack.activate_goal("g0").unwrap();
        stack.activate_goal("g1").unwrap();
        let result = stack.activate_goal("g2");

        assert!(result.is_err());
    }

    #[test]
    fn test_stats() {
        let mut stack = GoalStack::new();

        for i in 0..5 {
            let goal = Goal::new(&format!("g{}", i), &format!("Goal {}", i), 100.0);
            stack.add_goal(goal).unwrap();
        }

        stack.activate_goal("g0").unwrap();
        stack.activate_goal("g1").unwrap();
        stack.mark_achieved("g0").unwrap();
        stack.mark_failed("g1", "test").unwrap();

        let stats = stack.stats();
        assert_eq!(stats.total_goals, 5);
        assert_eq!(stats.achieved_goals, 1);
        assert_eq!(stats.failed_goals, 1);
    }

    #[test]
    fn test_generate_id() {
        let mut stack = GoalStack::new();
        let id1 = stack.generate_id();
        let id2 = stack.generate_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_events() {
        let mut stack = GoalStack::new();
        let goal = Goal::new("g1", "Test Goal", 100.0);
        stack.add_goal(goal).unwrap();
        stack.activate_goal("g1").unwrap();

        let events = stack.recent_events(10);
        assert!(events.len() >= 2);
        assert!(
            events
                .iter()
                .any(|e| e.event_type == GoalEventType::Created)
        );
        assert!(
            events
                .iter()
                .any(|e| e.event_type == GoalEventType::Activated)
        );
    }

    #[test]
    fn test_get_children() {
        let mut stack = GoalStack::new();
        let parent = Goal::new("p1", "Parent Goal", 100.0);
        stack.add_goal(parent).unwrap();

        let subgoals = vec![
            Goal::new("s1", "Subgoal 1", 50.0),
            Goal::new("s2", "Subgoal 2", 50.0),
        ];
        stack.decompose("p1", subgoals).unwrap();

        let children = stack.get_children("p1");
        assert_eq!(children.len(), 2);
    }
}
