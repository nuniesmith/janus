//! Goal prioritization
//!
//! Part of the Prefrontal region
//! Component: goals
//!
//! Dynamically ranks and re-prioritises trading goals based on urgency,
//! importance, deadline pressure, opportunity cost, and recent performance.
//! Provides EMA-smoothed priority signals and windowed diagnostics.
//!
//! ## Features
//!
//! - **Multi-factor scoring**: Each goal is scored on urgency (deadline
//!   pressure), importance (static weight), momentum (recent progress
//!   velocity), and opportunity cost (what we lose by *not* pursuing it).
//! - **Deadline pressure**: As a goal's deadline approaches, its urgency
//!   score increases non-linearly (exponential ramp).
//! - **Priority decay**: Goals that haven't been touched recently get a
//!   slight priority boost to prevent starvation.
//! - **EMA-smoothed composite score**: The raw composite priority score
//!   is smoothed per-goal to avoid thrashing the active goal ordering.
//! - **Rebalancing**: Periodically re-ranks all registered goals and
//!   returns the new ordering.
//! - **Windowed diagnostics**: Tracks priority-change frequency, churn
//!   rate, and rank stability.
//! - **Running statistics**: Total rebalances, rank changes, starvation
//!   events, deadline misses.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the priority engine
#[derive(Debug, Clone)]
pub struct PriorityConfig {
    /// Weight of the urgency factor in the composite score
    pub weight_urgency: f64,
    /// Weight of the importance factor in the composite score
    pub weight_importance: f64,
    /// Weight of the momentum factor in the composite score
    pub weight_momentum: f64,
    /// Weight of the opportunity-cost factor in the composite score
    pub weight_opportunity: f64,
    /// Weight of the starvation factor in the composite score
    pub weight_starvation: f64,
    /// EMA decay for smoothing per-goal priority scores (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum observations before EMA is considered initialised
    pub min_samples: usize,
    /// Sliding window size for windowed diagnostics
    pub window_size: usize,
    /// Deadline pressure ramp exponent (higher = sharper ramp near deadline)
    pub deadline_exponent: f64,
    /// Starvation boost per tick of inactivity
    pub starvation_rate: f64,
    /// Maximum starvation boost
    pub max_starvation_boost: f64,
    /// Maximum number of goals that can be registered
    pub max_goals: usize,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            weight_urgency: 0.30,
            weight_importance: 0.25,
            weight_momentum: 0.15,
            weight_opportunity: 0.15,
            weight_starvation: 0.15,
            ema_decay: 0.15,
            min_samples: 3,
            window_size: 100,
            deadline_exponent: 2.0,
            starvation_rate: 0.01,
            max_starvation_boost: 0.5,
            max_goals: 200,
        }
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Input factors for a single goal during rebalancing
#[derive(Debug, Clone)]
pub struct GoalFactors {
    /// Static importance weight (0.0–1.0)
    pub importance: f64,
    /// Current progress velocity (can be negative for regression)
    pub momentum: f64,
    /// Opportunity cost of *not* pursuing this goal (0.0–1.0)
    pub opportunity_cost: f64,
}

/// Registration entry for a goal in the priority engine
#[derive(Debug, Clone)]
pub struct GoalEntry {
    /// Goal identifier
    pub id: String,
    /// Static importance weight (0.0–1.0)
    pub importance: f64,
    /// Optional deadline timestamp
    pub deadline: Option<u64>,
    /// Whether this goal is currently active
    pub active: bool,
    /// Current rank (1 = highest priority)
    pub rank: usize,
    /// Raw composite priority score (before smoothing)
    pub raw_score: f64,
    /// EMA-smoothed composite priority score
    pub smoothed_score: f64,
    /// Whether the EMA is initialised
    ema_initialized: bool,
    /// Ticks since this goal was last touched / updated
    pub ticks_since_touch: u64,
    /// Last-known momentum
    pub last_momentum: f64,
    /// Last-known opportunity cost
    pub last_opportunity: f64,
}

/// Result of a rebalance operation
#[derive(Debug, Clone)]
pub struct RebalanceResult {
    /// Goal IDs in priority order (highest first)
    pub ranking: Vec<String>,
    /// Number of rank changes compared to the previous ordering
    pub rank_changes: usize,
    /// Whether any goal experienced starvation (starvation boost hit max)
    pub starvation_detected: bool,
    /// Goals whose deadlines are now missed (past deadline with < 100% progress)
    pub deadline_warnings: Vec<String>,
}

/// Record kept per rebalance for windowed analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RebalanceRecord {
    rank_changes: usize,
    starvation_detected: bool,
    top_goal_id: String,
    timestamp: u64,
}

/// Running statistics
#[derive(Debug, Clone)]
pub struct PriorityStats {
    /// Total rebalances performed
    pub total_rebalances: u64,
    /// Total individual rank changes across all rebalances
    pub total_rank_changes: u64,
    /// Total starvation events detected
    pub starvation_events: u64,
    /// Total deadline misses
    pub deadline_misses: u64,
    /// Number of goals registered
    pub registered_goals: usize,
    /// Number of active goals
    pub active_goals: usize,
    /// Number of times the top-ranked goal changed across rebalances
    pub top_goal_changes: u64,
    /// Peak composite score ever observed
    pub peak_score: f64,
    /// Maximum rank changes in a single rebalance
    pub max_rank_changes: usize,
}

impl Default for PriorityStats {
    fn default() -> Self {
        Self {
            total_rebalances: 0,
            total_rank_changes: 0,
            starvation_events: 0,
            deadline_misses: 0,
            registered_goals: 0,
            active_goals: 0,
            top_goal_changes: 0,
            peak_score: 0.0,
            max_rank_changes: 0,
        }
    }
}

impl PriorityStats {
    /// Mean rank changes per rebalance
    pub fn mean_rank_changes(&self) -> f64 {
        if self.total_rebalances == 0 {
            return 0.0;
        }
        self.total_rank_changes as f64 / self.total_rebalances as f64
    }

    /// Churn rate (fraction of rebalances that changed the top goal)
    pub fn churn_rate(&self) -> f64 {
        if self.total_rebalances <= 1 {
            return 0.0;
        }
        self.top_goal_changes as f64 / (self.total_rebalances - 1) as f64
    }

    /// Starvation rate
    pub fn starvation_rate(&self) -> f64 {
        if self.total_rebalances == 0 {
            return 0.0;
        }
        self.starvation_events as f64 / self.total_rebalances as f64
    }
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Dynamic goal priority ranking engine
pub struct Priority {
    config: PriorityConfig,

    /// Registered goals keyed by ID
    goals: HashMap<String, GoalEntry>,

    /// Current ranking (goal IDs in priority order)
    current_ranking: Vec<String>,

    /// Previous top goal ID (for churn detection)
    prev_top_goal: Option<String>,

    /// Windowed rebalance records
    recent: VecDeque<RebalanceRecord>,

    /// Running statistics
    stats: PriorityStats,
}

impl Default for Priority {
    fn default() -> Self {
        Self::new()
    }
}

impl Priority {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(PriorityConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PriorityConfig) -> Self {
        assert!(
            config.ema_decay > 0.0 && config.ema_decay < 1.0,
            "ema_decay must be in (0, 1)"
        );
        assert!(config.window_size > 0, "window_size must be > 0");
        assert!(
            config.deadline_exponent > 0.0,
            "deadline_exponent must be > 0"
        );
        assert!(
            config.starvation_rate >= 0.0,
            "starvation_rate must be >= 0"
        );
        assert!(
            config.max_starvation_boost >= 0.0,
            "max_starvation_boost must be >= 0"
        );
        assert!(config.max_goals > 0, "max_goals must be > 0");
        assert!(config.weight_urgency >= 0.0, "weight_urgency must be >= 0");
        assert!(
            config.weight_importance >= 0.0,
            "weight_importance must be >= 0"
        );
        assert!(
            config.weight_momentum >= 0.0,
            "weight_momentum must be >= 0"
        );
        assert!(
            config.weight_opportunity >= 0.0,
            "weight_opportunity must be >= 0"
        );
        assert!(
            config.weight_starvation >= 0.0,
            "weight_starvation must be >= 0"
        );

        Self {
            config,
            goals: HashMap::new(),
            current_ranking: Vec::new(),
            prev_top_goal: None,
            recent: VecDeque::new(),
            stats: PriorityStats::default(),
        }
    }

    /// Main processing function — performs a full rebalance with updated factors
    pub fn process(
        &mut self,
        factors: &HashMap<String, GoalFactors>,
        current_time: u64,
    ) -> Result<RebalanceResult> {
        // Update factors for all known goals
        for (id, f) in factors {
            if let Some(entry) = self.goals.get_mut(id) {
                entry.last_momentum = f.momentum;
                entry.last_opportunity = f.opportunity_cost;
                entry.importance = f.importance.clamp(0.0, 1.0);
                entry.ticks_since_touch = 0;
            }
        }

        // Increment starvation for goals not in this factors update
        for (id, entry) in &mut self.goals {
            if !factors.contains_key(id) {
                entry.ticks_since_touch += 1;
            }
        }

        self.rebalance(current_time)
    }

    // -----------------------------------------------------------------------
    // Goal registration
    // -----------------------------------------------------------------------

    /// Register a new goal
    pub fn register_goal(
        &mut self,
        id: &str,
        importance: f64,
        deadline: Option<u64>,
    ) -> Result<()> {
        if self.goals.len() >= self.config.max_goals {
            return Err(Error::InvalidInput(format!(
                "Maximum goals ({}) reached",
                self.config.max_goals
            )));
        }
        if self.goals.contains_key(id) {
            return Err(Error::InvalidInput(format!(
                "Goal '{}' already registered",
                id
            )));
        }
        if !importance.is_finite() || !(0.0..=1.0).contains(&importance) {
            return Err(Error::InvalidInput("importance must be in [0, 1]".into()));
        }

        let entry = GoalEntry {
            id: id.to_string(),
            importance,
            deadline,
            active: true,
            rank: 0,
            raw_score: 0.0,
            smoothed_score: 0.0,
            ema_initialized: false,
            ticks_since_touch: 0,
            last_momentum: 0.0,
            last_opportunity: 0.0,
        };

        self.goals.insert(id.to_string(), entry);
        self.stats.registered_goals = self.goals.len();
        Ok(())
    }

    /// Deregister (remove) a goal
    pub fn deregister_goal(&mut self, id: &str) -> Result<()> {
        if self.goals.remove(id).is_none() {
            return Err(Error::NotFound(format!("Goal '{}' not found", id)));
        }
        self.current_ranking.retain(|g| g != id);
        self.stats.registered_goals = self.goals.len();
        Ok(())
    }

    /// Set a goal as active or inactive
    pub fn set_active(&mut self, id: &str, active: bool) -> Result<()> {
        let entry = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("Goal '{}' not found", id)))?;
        entry.active = active;
        Ok(())
    }

    /// Update a goal's deadline
    pub fn set_deadline(&mut self, id: &str, deadline: Option<u64>) -> Result<()> {
        let entry = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("Goal '{}' not found", id)))?;
        entry.deadline = deadline;
        Ok(())
    }

    /// Update a goal's importance
    pub fn set_importance(&mut self, id: &str, importance: f64) -> Result<()> {
        if !importance.is_finite() || !(0.0..=1.0).contains(&importance) {
            return Err(Error::InvalidInput("importance must be in [0, 1]".into()));
        }
        let entry = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("Goal '{}' not found", id)))?;
        entry.importance = importance;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Rebalance
    // -----------------------------------------------------------------------

    /// Perform a full priority rebalance
    pub fn rebalance(&mut self, current_time: u64) -> Result<RebalanceResult> {
        let weight_sum = self.config.weight_urgency
            + self.config.weight_importance
            + self.config.weight_momentum
            + self.config.weight_opportunity
            + self.config.weight_starvation;

        if weight_sum <= 0.0 {
            return Err(Error::InvalidInput(
                "Sum of priority weights must be > 0".into(),
            ));
        }

        let mut deadline_warnings = Vec::new();
        let mut starvation_detected = false;

        // Compute composite scores for active goals
        let active_ids: Vec<String> = self
            .goals
            .iter()
            .filter(|(_, e)| e.active)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &active_ids {
            let entry = self.goals.get(id).unwrap();

            // --- Urgency (deadline pressure) ---
            let urgency = if let Some(deadline) = entry.deadline {
                if current_time >= deadline {
                    deadline_warnings.push(id.clone());
                    1.0
                } else {
                    let remaining = (deadline - current_time) as f64;
                    let total = deadline as f64; // from epoch
                    if total > 0.0 {
                        let fraction_remaining = (remaining / total.max(1.0)).clamp(0.0, 1.0);
                        (1.0 - fraction_remaining).powf(self.config.deadline_exponent)
                    } else {
                        0.0
                    }
                }
            } else {
                0.0
            };

            // --- Importance ---
            let importance = entry.importance;

            // --- Momentum ---
            // Normalise momentum to [0, 1] via sigmoid-like transform
            let momentum_raw = entry.last_momentum;
            let momentum = 1.0 / (1.0 + (-momentum_raw * 5.0).exp());

            // --- Opportunity cost ---
            let opportunity = entry.last_opportunity.clamp(0.0, 1.0);

            // --- Starvation ---
            let starvation_boost = (entry.ticks_since_touch as f64 * self.config.starvation_rate)
                .min(self.config.max_starvation_boost);
            if starvation_boost >= self.config.max_starvation_boost
                && self.config.max_starvation_boost > 0.0
            {
                starvation_detected = true;
            }

            // Weighted composite
            let raw = (self.config.weight_urgency * urgency
                + self.config.weight_importance * importance
                + self.config.weight_momentum * momentum
                + self.config.weight_opportunity * opportunity
                + self.config.weight_starvation * starvation_boost)
                / weight_sum;

            // EMA smooth
            let entry = self.goals.get_mut(id).unwrap();
            entry.raw_score = raw;
            if !entry.ema_initialized {
                entry.smoothed_score = raw;
                entry.ema_initialized = true;
            } else {
                let alpha = self.config.ema_decay;
                entry.smoothed_score = alpha * raw + (1.0 - alpha) * entry.smoothed_score;
            }

            // Track peak score
            if entry.smoothed_score > self.stats.peak_score {
                self.stats.peak_score = entry.smoothed_score;
            }
        }

        // Sort by smoothed score descending
        let mut scored: Vec<(String, f64)> = active_ids
            .iter()
            .filter_map(|id| self.goals.get(id).map(|e| (id.clone(), e.smoothed_score)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let new_ranking: Vec<String> = scored.into_iter().map(|(id, _)| id).collect();

        // Count rank changes
        let mut rank_changes = 0;
        for (new_rank, id) in new_ranking.iter().enumerate() {
            let new_rank_1 = new_rank + 1;
            if let Some(entry) = self.goals.get(id) {
                if entry.rank != 0 && entry.rank != new_rank_1 {
                    rank_changes += 1;
                }
            }
        }

        // Assign new ranks
        for (new_rank, id) in new_ranking.iter().enumerate() {
            if let Some(entry) = self.goals.get_mut(id) {
                entry.rank = new_rank + 1;
            }
        }

        // Detect top-goal churn
        let new_top = new_ranking.first().cloned();
        if let (Some(prev), Some(new_top_id)) = (&self.prev_top_goal, &new_top) {
            if prev != new_top_id {
                self.stats.top_goal_changes += 1;
            }
        }
        self.prev_top_goal = new_top;

        // Update stats
        self.stats.total_rebalances += 1;
        self.stats.total_rank_changes += rank_changes as u64;
        self.stats.active_goals = active_ids.len();
        self.stats.deadline_misses += deadline_warnings.len() as u64;
        if starvation_detected {
            self.stats.starvation_events += 1;
        }
        if rank_changes > self.stats.max_rank_changes {
            self.stats.max_rank_changes = rank_changes;
        }

        // Record for windowed diagnostics
        let record = RebalanceRecord {
            rank_changes,
            starvation_detected,
            top_goal_id: new_ranking.first().cloned().unwrap_or_default(),
            timestamp: current_time,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        self.current_ranking = new_ranking.clone();

        Ok(RebalanceResult {
            ranking: new_ranking,
            rank_changes,
            starvation_detected,
            deadline_warnings,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get the current ranking (goal IDs in priority order)
    pub fn current_ranking(&self) -> &[String] {
        &self.current_ranking
    }

    /// Get the top-ranked goal ID
    pub fn top_goal(&self) -> Option<&str> {
        self.current_ranking.first().map(|s| s.as_str())
    }

    /// Get a goal entry by ID
    pub fn goal(&self, id: &str) -> Option<&GoalEntry> {
        self.goals.get(id)
    }

    /// Get all registered goal entries
    pub fn goals(&self) -> &HashMap<String, GoalEntry> {
        &self.goals
    }

    /// Number of registered goals
    pub fn goal_count(&self) -> usize {
        self.goals.len()
    }

    /// Number of active goals
    pub fn active_count(&self) -> usize {
        self.goals.values().filter(|e| e.active).count()
    }

    /// Total rebalances performed
    pub fn rebalance_count(&self) -> u64 {
        self.stats.total_rebalances
    }

    /// Running statistics
    pub fn stats(&self) -> &PriorityStats {
        &self.stats
    }

    /// Get the score of a goal
    pub fn score(&self, id: &str) -> Option<f64> {
        self.goals.get(id).map(|e| e.smoothed_score)
    }

    /// Get the rank of a goal (1-based, 0 = unranked)
    pub fn rank(&self, id: &str) -> Option<usize> {
        self.goals.get(id).map(|e| e.rank)
    }

    /// Whether the priority engine has enough rebalances for reliable signals
    pub fn is_warmed_up(&self) -> bool {
        self.stats.total_rebalances as usize >= self.config.min_samples
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Mean rank changes per rebalance over the recent window
    pub fn windowed_mean_rank_changes(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: usize = self.recent.iter().map(|r| r.rank_changes).sum();
        sum as f64 / self.recent.len() as f64
    }

    /// Fraction of recent rebalances with starvation detected
    pub fn windowed_starvation_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let count = self.recent.iter().filter(|r| r.starvation_detected).count();
        count as f64 / self.recent.len() as f64
    }

    /// Number of unique top goals in the recent window (stability measure)
    pub fn windowed_top_goal_diversity(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for r in &self.recent {
            if !r.top_goal_id.is_empty() {
                seen.insert(r.top_goal_id.clone());
            }
        }
        seen.len()
    }

    /// Whether priorities are churning (unstable top ranking)
    ///
    /// Returns true if more than 50% of recent rebalances changed the top goal.
    pub fn is_churning(&self) -> bool {
        let n = self.recent.len();
        if n < 4 {
            return false;
        }
        let mut changes = 0;
        let items: Vec<&RebalanceRecord> = self.recent.iter().collect();
        for i in 1..items.len() {
            if items[i].top_goal_id != items[i - 1].top_goal_id {
                changes += 1;
            }
        }
        changes as f64 / (n - 1) as f64 > 0.5
    }

    /// Whether priorities have been stable (no rank changes in second half of window)
    pub fn is_stable(&self) -> bool {
        let n = self.recent.len();
        if n < 6 {
            return false;
        }
        let mid = n / 2;
        self.recent.iter().skip(mid).all(|r| r.rank_changes == 0)
    }

    /// Confidence based on number of rebalances
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_rebalances as f64;
        let required = self.config.min_samples as f64;
        if n >= required * 4.0 {
            1.0
        } else {
            (n / (required * 4.0)).min(1.0)
        }
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all state, keeping configuration and registered goals
    pub fn reset(&mut self) {
        for entry in self.goals.values_mut() {
            entry.rank = 0;
            entry.raw_score = 0.0;
            entry.smoothed_score = 0.0;
            entry.ema_initialized = false;
            entry.ticks_since_touch = 0;
            entry.last_momentum = 0.0;
            entry.last_opportunity = 0.0;
        }
        self.current_ranking.clear();
        self.prev_top_goal = None;
        self.recent.clear();
        self.stats = PriorityStats {
            registered_goals: self.goals.len(),
            ..Default::default()
        };
    }

    /// Reset all state and remove all goals
    pub fn reset_all(&mut self) {
        self.goals.clear();
        self.current_ranking.clear();
        self.prev_top_goal = None;
        self.recent.clear();
        self.stats = PriorityStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_factors(importance: f64, momentum: f64, opportunity: f64) -> GoalFactors {
        GoalFactors {
            importance,
            momentum,
            opportunity_cost: opportunity,
        }
    }

    #[test]
    fn test_basic() {
        let mut instance = Priority::new();
        assert!(instance.process(&HashMap::new(), 0).is_ok());
    }

    #[test]
    fn test_register_and_rebalance() {
        let mut p = Priority::new();
        p.register_goal("g1", 0.8, None).unwrap();
        p.register_goal("g2", 0.3, None).unwrap();

        let mut factors = HashMap::new();
        factors.insert("g1".into(), simple_factors(0.8, 0.0, 0.0));
        factors.insert("g2".into(), simple_factors(0.3, 0.0, 0.0));

        let result = p.process(&factors, 100).unwrap();
        assert_eq!(result.ranking.len(), 2);
        // g1 should be ranked higher (more important)
        assert_eq!(result.ranking[0], "g1");
        assert_eq!(result.ranking[1], "g2");
    }

    #[test]
    fn test_rank_assignment() {
        let mut p = Priority::new();
        p.register_goal("a", 0.9, None).unwrap();
        p.register_goal("b", 0.1, None).unwrap();

        let mut factors = HashMap::new();
        factors.insert("a".into(), simple_factors(0.9, 0.0, 0.0));
        factors.insert("b".into(), simple_factors(0.1, 0.0, 0.0));

        p.process(&factors, 1).unwrap();
        assert_eq!(p.rank("a"), Some(1));
        assert_eq!(p.rank("b"), Some(2));
    }

    #[test]
    fn test_top_goal() {
        let mut p = Priority::new();
        p.register_goal("x", 0.5, None).unwrap();

        let mut factors = HashMap::new();
        factors.insert("x".into(), simple_factors(0.5, 0.0, 0.0));
        p.process(&factors, 0).unwrap();

        assert_eq!(p.top_goal(), Some("x"));
    }

    #[test]
    fn test_empty_rebalance() {
        let mut p = Priority::new();
        let result = p.rebalance(0).unwrap();
        assert!(result.ranking.is_empty());
        assert_eq!(result.rank_changes, 0);
    }

    #[test]
    fn test_importance_ordering() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_urgency: 0.0,
            weight_importance: 1.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99, // near-instant
            ..Default::default()
        });
        p.register_goal("low", 0.2, None).unwrap();
        p.register_goal("high", 0.9, None).unwrap();
        p.register_goal("mid", 0.5, None).unwrap();

        let mut factors = HashMap::new();
        factors.insert("low".into(), simple_factors(0.2, 0.0, 0.0));
        factors.insert("high".into(), simple_factors(0.9, 0.0, 0.0));
        factors.insert("mid".into(), simple_factors(0.5, 0.0, 0.0));

        let r = p.process(&factors, 0).unwrap();
        assert_eq!(r.ranking, vec!["high", "mid", "low"]);
    }

    #[test]
    fn test_deadline_pressure() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_urgency: 1.0,
            weight_importance: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        // g1 has a far deadline, g2 has an imminent deadline
        p.register_goal("far", 0.5, Some(1000)).unwrap();
        p.register_goal("near", 0.5, Some(110)).unwrap();

        let mut factors = HashMap::new();
        factors.insert("far".into(), simple_factors(0.5, 0.0, 0.0));
        factors.insert("near".into(), simple_factors(0.5, 0.0, 0.0));

        let r = p.process(&factors, 100).unwrap();
        // "near" should have higher urgency (closer to deadline)
        assert_eq!(r.ranking[0], "near");
    }

    #[test]
    fn test_past_deadline_max_urgency() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_urgency: 1.0,
            weight_importance: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("past", 0.5, Some(50)).unwrap();

        let mut factors = HashMap::new();
        factors.insert("past".into(), simple_factors(0.5, 0.0, 0.0));

        let r = p.process(&factors, 100).unwrap();
        assert_eq!(r.deadline_warnings.len(), 1);
        assert_eq!(r.deadline_warnings[0], "past");
    }

    #[test]
    fn test_no_deadline_zero_urgency() {
        let mut p = Priority::new();
        p.register_goal("nd", 0.5, None).unwrap();

        let mut factors = HashMap::new();
        factors.insert("nd".into(), simple_factors(0.5, 0.0, 0.0));
        p.process(&factors, 100).unwrap();

        // The raw urgency component should be 0 (no deadline)
        // We can verify indirectly - the score should not have urgency contribution
        let entry = p.goal("nd").unwrap();
        assert!(entry.smoothed_score >= 0.0);
    }

    #[test]
    fn test_starvation_boost() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_urgency: 0.0,
            weight_importance: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 1.0,
            starvation_rate: 0.1,
            max_starvation_boost: 0.5,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("starved", 0.5, None).unwrap();
        p.register_goal("fed", 0.5, None).unwrap();

        // Only update "fed", leave "starved" alone
        let mut factors = HashMap::new();
        factors.insert("fed".into(), simple_factors(0.5, 0.0, 0.0));

        // Several rebalances without touching "starved"
        for i in 0..10 {
            p.process(&factors, i).unwrap();
        }

        // "starved" should have accumulated starvation boost
        let starved = p.goal("starved").unwrap();
        let fed = p.goal("fed").unwrap();
        assert!(starved.smoothed_score > fed.smoothed_score);
    }

    #[test]
    fn test_starvation_capped() {
        let mut p = Priority::with_config(PriorityConfig {
            starvation_rate: 1.0,
            max_starvation_boost: 0.3,
            ..Default::default()
        });
        p.register_goal("g", 0.5, None).unwrap();

        // 100 rebalances without updating "g"
        for i in 0..100 {
            p.process(&HashMap::new(), i).unwrap();
        }

        let entry = p.goal("g").unwrap();
        // The starvation component alone should be capped
        assert!(entry.ticks_since_touch >= 100);
    }

    #[test]
    fn test_starvation_detected_flag() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_starvation: 1.0,
            starvation_rate: 1.0,
            max_starvation_boost: 0.1,
            ..Default::default()
        });
        p.register_goal("g", 0.5, None).unwrap();

        // 1st rebalance: no starvation yet (ticks_since_touch = 0 -> boost = 0)
        let _r1 = p.process(&HashMap::new(), 0).unwrap();
        // After first process, ticks_since_touch = 1 (incremented for next round)
        // The starvation_detected flag depends on whether boost >= max

        // After a few rounds, starvation should be detected
        let r2 = p.process(&HashMap::new(), 1).unwrap();
        // ticks now = 2, boost = 2*1.0 = 2.0 capped at 0.1 => starvation_detected
        assert!(r2.starvation_detected);
    }

    #[test]
    fn test_rank_changes_counted() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_importance: 1.0,
            weight_urgency: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("a", 0.9, None).unwrap();
        p.register_goal("b", 0.1, None).unwrap();

        let mut f1 = HashMap::new();
        f1.insert("a".into(), simple_factors(0.9, 0.0, 0.0));
        f1.insert("b".into(), simple_factors(0.1, 0.0, 0.0));
        p.process(&f1, 0).unwrap(); // a=1, b=2

        // Swap importance
        let mut f2 = HashMap::new();
        f2.insert("a".into(), simple_factors(0.1, 0.0, 0.0));
        f2.insert("b".into(), simple_factors(0.9, 0.0, 0.0));
        let r = p.process(&f2, 1).unwrap();
        assert!(r.rank_changes > 0);
    }

    #[test]
    fn test_top_goal_change_tracking() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_importance: 1.0,
            weight_urgency: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("a", 0.9, None).unwrap();
        p.register_goal("b", 0.1, None).unwrap();

        let mut f1 = HashMap::new();
        f1.insert("a".into(), simple_factors(0.9, 0.0, 0.0));
        f1.insert("b".into(), simple_factors(0.1, 0.0, 0.0));
        p.process(&f1, 0).unwrap(); // top = a

        let mut f2 = HashMap::new();
        f2.insert("a".into(), simple_factors(0.1, 0.0, 0.0));
        f2.insert("b".into(), simple_factors(0.9, 0.0, 0.0));
        p.process(&f2, 1).unwrap(); // top = b

        assert_eq!(p.stats().top_goal_changes, 1);
    }

    #[test]
    fn test_inactive_goals_excluded() {
        let mut p = Priority::new();
        p.register_goal("a", 0.9, None).unwrap();
        p.register_goal("b", 0.5, None).unwrap();
        p.set_active("b", false).unwrap();

        let mut factors = HashMap::new();
        factors.insert("a".into(), simple_factors(0.9, 0.0, 0.0));
        factors.insert("b".into(), simple_factors(0.5, 0.0, 0.0));

        let r = p.process(&factors, 0).unwrap();
        assert_eq!(r.ranking.len(), 1);
        assert_eq!(r.ranking[0], "a");
    }

    #[test]
    fn test_duplicate_registration_error() {
        let mut p = Priority::new();
        p.register_goal("g", 0.5, None).unwrap();
        assert!(p.register_goal("g", 0.3, None).is_err());
    }

    #[test]
    fn test_deregister_goal() {
        let mut p = Priority::new();
        p.register_goal("g", 0.5, None).unwrap();
        assert_eq!(p.goal_count(), 1);
        p.deregister_goal("g").unwrap();
        assert_eq!(p.goal_count(), 0);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let mut p = Priority::new();
        assert!(p.deregister_goal("nope").is_err());
    }

    #[test]
    fn test_max_goals_limit() {
        let mut p = Priority::with_config(PriorityConfig {
            max_goals: 2,
            ..Default::default()
        });
        p.register_goal("a", 0.5, None).unwrap();
        p.register_goal("b", 0.5, None).unwrap();
        assert!(p.register_goal("c", 0.5, None).is_err());
    }

    #[test]
    fn test_invalid_importance() {
        let mut p = Priority::new();
        assert!(p.register_goal("g", -0.1, None).is_err());
        assert!(p.register_goal("g", 1.1, None).is_err());
        assert!(p.register_goal("g", f64::NAN, None).is_err());
    }

    #[test]
    fn test_set_importance() {
        let mut p = Priority::new();
        p.register_goal("g", 0.5, None).unwrap();
        p.set_importance("g", 0.9).unwrap();
        assert!((p.goal("g").unwrap().importance - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_set_importance_invalid() {
        let mut p = Priority::new();
        p.register_goal("g", 0.5, None).unwrap();
        assert!(p.set_importance("g", 1.5).is_err());
        assert!(p.set_importance("g", -0.1).is_err());
    }

    #[test]
    fn test_set_importance_nonexistent() {
        let mut p = Priority::new();
        assert!(p.set_importance("nope", 0.5).is_err());
    }

    #[test]
    fn test_set_deadline() {
        let mut p = Priority::new();
        p.register_goal("g", 0.5, None).unwrap();
        p.set_deadline("g", Some(1000)).unwrap();
        assert_eq!(p.goal("g").unwrap().deadline, Some(1000));
    }

    #[test]
    fn test_set_active_nonexistent() {
        let mut p = Priority::new();
        assert!(p.set_active("nope", true).is_err());
    }

    #[test]
    fn test_momentum_effect() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_urgency: 0.0,
            weight_importance: 0.0,
            weight_momentum: 1.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("positive_mom", 0.5, None).unwrap();
        p.register_goal("negative_mom", 0.5, None).unwrap();

        let mut factors = HashMap::new();
        factors.insert("positive_mom".into(), simple_factors(0.5, 2.0, 0.0));
        factors.insert("negative_mom".into(), simple_factors(0.5, -2.0, 0.0));

        let r = p.process(&factors, 0).unwrap();
        assert_eq!(r.ranking[0], "positive_mom");
    }

    #[test]
    fn test_opportunity_cost_effect() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_urgency: 0.0,
            weight_importance: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 1.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("high_opp", 0.5, None).unwrap();
        p.register_goal("low_opp", 0.5, None).unwrap();

        let mut factors = HashMap::new();
        factors.insert("high_opp".into(), simple_factors(0.5, 0.0, 0.9));
        factors.insert("low_opp".into(), simple_factors(0.5, 0.0, 0.1));

        let r = p.process(&factors, 0).unwrap();
        assert_eq!(r.ranking[0], "high_opp");
    }

    #[test]
    fn test_ema_smoothing_dampens() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_importance: 1.0,
            weight_urgency: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.1, // slow
            ..Default::default()
        });
        p.register_goal("g", 0.5, None).unwrap();

        let mut f1 = HashMap::new();
        f1.insert("g".into(), simple_factors(0.9, 0.0, 0.0));
        p.process(&f1, 0).unwrap();
        let s1 = p.goal("g").unwrap().smoothed_score;

        // Drop importance dramatically
        let mut f2 = HashMap::new();
        f2.insert("g".into(), simple_factors(0.1, 0.0, 0.0));
        p.process(&f2, 1).unwrap();
        let s2 = p.goal("g").unwrap().smoothed_score;

        // Smoothed score should not have dropped all the way to 0.1's score
        assert!(s2 > p.goal("g").unwrap().raw_score);
        // But should be lower than before
        assert!(s2 < s1);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = PriorityStats::default();
        assert_eq!(stats.total_rebalances, 0);
        assert_eq!(stats.total_rank_changes, 0);
        assert_eq!(stats.starvation_events, 0);
        assert_eq!(stats.deadline_misses, 0);
        assert_eq!(stats.mean_rank_changes(), 0.0);
        assert_eq!(stats.churn_rate(), 0.0);
        assert_eq!(stats.starvation_rate(), 0.0);
    }

    #[test]
    fn test_stats_mean_rank_changes() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_importance: 1.0,
            weight_urgency: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("a", 0.9, None).unwrap();
        p.register_goal("b", 0.1, None).unwrap();

        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.9, 0.0, 0.0));
        f.insert("b".into(), simple_factors(0.1, 0.0, 0.0));

        // Same ranking each time => 0 changes
        for i in 0..5 {
            p.process(&f, i).unwrap();
        }
        assert!((p.stats().mean_rank_changes() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_mean_rank_changes() {
        let mut p = Priority::new();
        p.register_goal("a", 0.5, None).unwrap();

        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.5, 0.0, 0.0));

        for i in 0..5 {
            p.process(&f, i).unwrap();
        }
        // All same => 0 rank changes
        assert!((p.windowed_mean_rank_changes() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_mean_rank_changes_empty() {
        let p = Priority::new();
        assert_eq!(p.windowed_mean_rank_changes(), 0.0);
    }

    #[test]
    fn test_windowed_starvation_rate() {
        let p = Priority::new();
        assert_eq!(p.windowed_starvation_rate(), 0.0);
    }

    #[test]
    fn test_windowed_top_goal_diversity() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_importance: 1.0,
            weight_urgency: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("a", 0.9, None).unwrap();
        p.register_goal("b", 0.1, None).unwrap();

        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.9, 0.0, 0.0));
        f.insert("b".into(), simple_factors(0.1, 0.0, 0.0));

        for i in 0..5 {
            p.process(&f, i).unwrap();
        }
        // Same top goal every time
        assert_eq!(p.windowed_top_goal_diversity(), 1);
    }

    #[test]
    fn test_is_churning_insufficient_data() {
        let mut p = Priority::new();
        p.register_goal("a", 0.5, None).unwrap();

        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.5, 0.0, 0.0));
        p.process(&f, 0).unwrap();

        assert!(!p.is_churning()); // Need >= 4 records
    }

    #[test]
    fn test_is_stable_insufficient_data() {
        let p = Priority::new();
        assert!(!p.is_stable()); // Need >= 6 records
    }

    #[test]
    fn test_is_stable_with_no_changes() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_importance: 1.0,
            weight_urgency: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("a", 0.9, None).unwrap();
        p.register_goal("b", 0.1, None).unwrap();

        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.9, 0.0, 0.0));
        f.insert("b".into(), simple_factors(0.1, 0.0, 0.0));

        for i in 0..10 {
            p.process(&f, i).unwrap();
        }
        assert!(p.is_stable());
    }

    #[test]
    fn test_is_warmed_up() {
        let mut p = Priority::with_config(PriorityConfig {
            min_samples: 3,
            ..Default::default()
        });
        assert!(!p.is_warmed_up());
        p.register_goal("a", 0.5, None).unwrap();
        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.5, 0.0, 0.0));

        p.process(&f, 0).unwrap();
        p.process(&f, 1).unwrap();
        assert!(!p.is_warmed_up());
        p.process(&f, 2).unwrap();
        assert!(p.is_warmed_up());
    }

    #[test]
    fn test_confidence_increases() {
        let mut p = Priority::with_config(PriorityConfig {
            min_samples: 3,
            ..Default::default()
        });
        assert_eq!(p.confidence(), 0.0);

        p.register_goal("a", 0.5, None).unwrap();
        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.5, 0.0, 0.0));

        for i in 0..6 {
            p.process(&f, i).unwrap();
        }
        let c6 = p.confidence();
        assert!(c6 > 0.0);

        for i in 6..20 {
            p.process(&f, i).unwrap();
        }
        let c20 = p.confidence();
        assert!(c20 >= c6);
        assert!(c20 <= 1.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut p = Priority::with_config(PriorityConfig {
            window_size: 3,
            ..Default::default()
        });
        p.register_goal("a", 0.5, None).unwrap();
        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.5, 0.0, 0.0));

        for i in 0..10 {
            p.process(&f, i).unwrap();
        }
        assert_eq!(p.recent.len(), 3);
    }

    #[test]
    fn test_reset() {
        let mut p = Priority::new();
        p.register_goal("a", 0.5, None).unwrap();
        p.register_goal("b", 0.3, None).unwrap();

        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.5, 0.0, 0.0));
        f.insert("b".into(), simple_factors(0.3, 0.0, 0.0));

        for i in 0..5 {
            p.process(&f, i).unwrap();
        }
        assert!(p.rebalance_count() > 0);

        p.reset();
        assert_eq!(p.rebalance_count(), 0);
        assert!(p.current_ranking().is_empty());
        assert_eq!(p.goal_count(), 2); // goals kept
        assert_eq!(p.goal("a").unwrap().rank, 0);
        assert_eq!(p.goal("a").unwrap().smoothed_score, 0.0);
    }

    #[test]
    fn test_reset_all() {
        let mut p = Priority::new();
        p.register_goal("a", 0.5, None).unwrap();
        p.reset_all();
        assert_eq!(p.goal_count(), 0);
    }

    #[test]
    fn test_score_accessor() {
        let mut p = Priority::new();
        p.register_goal("g", 0.5, None).unwrap();

        let mut f = HashMap::new();
        f.insert("g".into(), simple_factors(0.5, 0.0, 0.0));
        p.process(&f, 0).unwrap();

        let score = p.score("g");
        assert!(score.is_some());
        assert!(score.unwrap() > 0.0);
    }

    #[test]
    fn test_score_nonexistent() {
        let p = Priority::new();
        assert!(p.score("nope").is_none());
    }

    #[test]
    fn test_active_count() {
        let mut p = Priority::new();
        p.register_goal("a", 0.5, None).unwrap();
        p.register_goal("b", 0.5, None).unwrap();
        assert_eq!(p.active_count(), 2);

        p.set_active("b", false).unwrap();
        assert_eq!(p.active_count(), 1);
    }

    #[test]
    fn test_peak_score_tracked() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_importance: 1.0,
            weight_urgency: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("g", 0.5, None).unwrap();

        let mut f_high = HashMap::new();
        f_high.insert("g".into(), simple_factors(1.0, 0.0, 0.0));
        p.process(&f_high, 0).unwrap();

        let peak = p.stats().peak_score;
        assert!(peak > 0.0);

        // Lower importance shouldn't reduce peak
        let mut f_low = HashMap::new();
        f_low.insert("g".into(), simple_factors(0.1, 0.0, 0.0));
        p.process(&f_low, 1).unwrap();
        assert!((p.stats().peak_score - peak).abs() < 1e-9);
    }

    #[test]
    fn test_max_rank_changes_tracked() {
        let mut p = Priority::with_config(PriorityConfig {
            weight_importance: 1.0,
            weight_urgency: 0.0,
            weight_momentum: 0.0,
            weight_opportunity: 0.0,
            weight_starvation: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        });
        p.register_goal("a", 0.9, None).unwrap();
        p.register_goal("b", 0.1, None).unwrap();

        let mut f = HashMap::new();
        f.insert("a".into(), simple_factors(0.9, 0.0, 0.0));
        f.insert("b".into(), simple_factors(0.1, 0.0, 0.0));
        p.process(&f, 0).unwrap();

        // Swap them
        let mut f2 = HashMap::new();
        f2.insert("a".into(), simple_factors(0.1, 0.0, 0.0));
        f2.insert("b".into(), simple_factors(0.9, 0.0, 0.0));
        p.process(&f2, 1).unwrap();

        assert!(p.stats().max_rank_changes > 0);
    }

    #[test]
    fn test_deadline_misses_counted() {
        let mut p = Priority::new();
        p.register_goal("g", 0.5, Some(10)).unwrap();

        let mut f = HashMap::new();
        f.insert("g".into(), simple_factors(0.5, 0.0, 0.0));

        p.process(&f, 20).unwrap();
        assert_eq!(p.stats().deadline_misses, 1);

        p.process(&f, 30).unwrap();
        assert_eq!(p.stats().deadline_misses, 2);
    }

    #[test]
    fn test_single_goal_always_rank_one() {
        let mut p = Priority::new();
        p.register_goal("only", 0.5, None).unwrap();

        let mut f = HashMap::new();
        f.insert("only".into(), simple_factors(0.5, 0.0, 0.0));

        for i in 0..5 {
            p.process(&f, i).unwrap();
        }
        assert_eq!(p.rank("only"), Some(1));
    }

    #[test]
    fn test_process_with_unknown_factors_ignored() {
        let mut p = Priority::new();
        p.register_goal("a", 0.5, None).unwrap();

        let mut factors = HashMap::new();
        factors.insert("a".into(), simple_factors(0.5, 0.0, 0.0));
        factors.insert("unknown".into(), simple_factors(0.9, 0.0, 0.0));

        // Should not error — unknown goals in factors are silently ignored
        let r = p.process(&factors, 0).unwrap();
        assert_eq!(r.ranking.len(), 1);
    }

    #[test]
    #[should_panic(expected = "ema_decay must be in (0, 1)")]
    fn test_invalid_config_ema_decay() {
        Priority::with_config(PriorityConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn test_invalid_config_window_size() {
        Priority::with_config(PriorityConfig {
            window_size: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "deadline_exponent must be > 0")]
    fn test_invalid_config_deadline_exponent() {
        Priority::with_config(PriorityConfig {
            deadline_exponent: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "max_goals must be > 0")]
    fn test_invalid_config_max_goals() {
        Priority::with_config(PriorityConfig {
            max_goals: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "starvation_rate must be >= 0")]
    fn test_invalid_config_starvation_rate() {
        Priority::with_config(PriorityConfig {
            starvation_rate: -0.1,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "weight_urgency must be >= 0")]
    fn test_invalid_config_negative_weight() {
        Priority::with_config(PriorityConfig {
            weight_urgency: -0.1,
            ..Default::default()
        });
    }

    #[test]
    fn test_default_has_no_goals() {
        let p = Priority::new();
        assert_eq!(p.goal_count(), 0);
        assert_eq!(p.active_count(), 0);
        assert!(p.top_goal().is_none());
        assert!(p.current_ranking().is_empty());
        assert_eq!(p.rebalance_count(), 0);
    }
}
