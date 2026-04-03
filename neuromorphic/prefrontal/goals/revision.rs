//! Goal revision based on feedback
//!
//! Part of the Prefrontal region
//! Component: goals
//!
//! Decides when and how trading goals should be revised based on ongoing
//! performance feedback. Monitors achievement rates, drawdown, and market
//! regime signals to propose automatic target adjustments.
//!
//! ## Features
//!
//! - **Performance-based revision**: When achievement rate drops below a
//!   configurable threshold, targets are loosened; when it consistently
//!   exceeds expectations, targets are tightened (raised).
//! - **Cooldown enforcement**: After a revision, further revisions are
//!   blocked for a configurable number of ticks to prevent oscillation.
//! - **Graduated adjustments**: Revision magnitude scales with how far
//!   performance deviates from the target band (proportional control).
//! - **Revision limits**: Maximum number of revisions per goal, and
//!   maximum cumulative adjustment (as fraction of original target).
//! - **EMA-smoothed revision pressure**: A smoothed signal indicating
//!   how strongly the system wants to revise — upstream can use this
//!   as a continuous input.
//! - **Revision history**: Each revision is recorded with a reason,
//!   direction, and magnitude for auditability.
//! - **Windowed diagnostics**: Recent revision frequency, mean magnitude,
//!   and direction bias.
//! - **Running statistics**: Total revisions, upward/downward counts,
//!   cumulative adjustment, peak pressure.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the revision engine
#[derive(Debug, Clone)]
pub struct RevisionConfig {
    /// Achievement rate below which targets are loosened (0.0–1.0)
    pub loosen_threshold: f64,
    /// Achievement rate above which targets are tightened (0.0–1.0)
    pub tighten_threshold: f64,
    /// Base adjustment fraction per revision (e.g. 0.05 = 5% of current target)
    pub base_adjustment: f64,
    /// Maximum adjustment fraction per single revision
    pub max_single_adjustment: f64,
    /// Maximum cumulative adjustment as fraction of original target
    pub max_cumulative_adjustment: f64,
    /// Cooldown ticks after a revision before another can occur
    pub cooldown_ticks: u64,
    /// Maximum total revisions allowed per goal (0 = unlimited)
    pub max_revisions: usize,
    /// EMA decay for revision pressure smoothing (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum observations before revision is considered
    pub min_samples: usize,
    /// Sliding window size for windowed diagnostics
    pub window_size: usize,
    /// Minimum absolute deviation from threshold to trigger revision
    pub dead_zone: f64,
    /// Proportional gain — scales adjustment with deviation magnitude
    pub proportional_gain: f64,
}

impl Default for RevisionConfig {
    fn default() -> Self {
        Self {
            loosen_threshold: 0.3,
            tighten_threshold: 0.8,
            base_adjustment: 0.05,
            max_single_adjustment: 0.20,
            max_cumulative_adjustment: 0.50,
            cooldown_ticks: 10,
            max_revisions: 0, // unlimited
            ema_decay: 0.1,
            min_samples: 5,
            window_size: 100,
            dead_zone: 0.02,
            proportional_gain: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Direction of a revision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionDirection {
    /// Target was loosened (made easier)
    Loosen,
    /// Target was tightened (made harder)
    Tighten,
}

/// A recorded revision event
#[derive(Debug, Clone)]
pub struct RevisionEvent {
    /// Goal identifier
    pub goal_id: String,
    /// Direction of revision
    pub direction: RevisionDirection,
    /// Adjustment fraction applied (signed: negative = loosen, positive = tighten)
    pub adjustment: f64,
    /// Old target value
    pub old_target: f64,
    /// New target value
    pub new_target: f64,
    /// Achievement rate at the time of revision
    pub achievement_rate: f64,
    /// Revision pressure at the time
    pub pressure: f64,
    /// Tick / timestamp
    pub tick: u64,
    /// Human-readable reason
    pub reason: String,
}

/// State tracked per registered goal
#[derive(Debug, Clone)]
pub struct GoalRevisionState {
    /// Goal identifier
    pub id: String,
    /// Original target value (never changes)
    pub original_target: f64,
    /// Current target value (adjusted by revisions)
    pub current_target: f64,
    /// Cumulative adjustment fraction from original
    pub cumulative_adjustment: f64,
    /// Number of revisions applied
    pub revision_count: usize,
    /// Ticks remaining in cooldown (0 = ready)
    pub cooldown_remaining: u64,
    /// Whether further revisions are locked (max reached)
    pub locked: bool,
}

/// Result of a revision check
#[derive(Debug, Clone)]
pub struct RevisionCheckResult {
    /// Whether a revision was applied
    pub revised: bool,
    /// Revision event (if any)
    pub event: Option<RevisionEvent>,
    /// Current revision pressure for this goal
    pub pressure: f64,
    /// Whether the goal is in cooldown
    pub in_cooldown: bool,
    /// Whether the goal is locked from further revisions
    pub locked: bool,
}

/// Performance input for a single goal
#[derive(Debug, Clone)]
pub struct PerformanceInput {
    /// Current achievement rate (0.0–1.0)
    pub achievement_rate: f64,
    /// Optional external pressure signal (-1.0 to 1.0, negative = loosen, positive = tighten)
    pub external_pressure: Option<f64>,
}

/// Record for windowed analysis
#[derive(Debug, Clone)]
struct RevisionRecord {
    goal_id: String,
    direction: RevisionDirection,
    adjustment_magnitude: f64,
    tick: u64,
}

/// Running statistics
#[derive(Debug, Clone)]
pub struct RevisionStats {
    /// Total revision events across all goals
    pub total_revisions: u64,
    /// Total loosening revisions
    pub loosen_count: u64,
    /// Total tightening revisions
    pub tighten_count: u64,
    /// Sum of absolute adjustment magnitudes
    pub sum_abs_adjustment: f64,
    /// Sum of squared adjustment magnitudes
    pub sum_sq_adjustment: f64,
    /// Maximum single adjustment magnitude observed
    pub max_adjustment: f64,
    /// Number of revision checks performed
    pub total_checks: u64,
    /// Number of times a revision was blocked by cooldown
    pub cooldown_blocks: u64,
    /// Number of times a revision was blocked by max-revision lock
    pub lock_blocks: u64,
    /// Peak revision pressure observed
    pub peak_pressure: f64,
    /// Number of goals currently registered
    pub registered_goals: usize,
}

impl Default for RevisionStats {
    fn default() -> Self {
        Self {
            total_revisions: 0,
            loosen_count: 0,
            tighten_count: 0,
            sum_abs_adjustment: 0.0,
            sum_sq_adjustment: 0.0,
            max_adjustment: 0.0,
            total_checks: 0,
            cooldown_blocks: 0,
            lock_blocks: 0,
            peak_pressure: 0.0,
            registered_goals: 0,
        }
    }
}

impl RevisionStats {
    /// Mean adjustment magnitude per revision
    pub fn mean_adjustment(&self) -> f64 {
        if self.total_revisions == 0 {
            return 0.0;
        }
        self.sum_abs_adjustment / self.total_revisions as f64
    }

    /// Adjustment magnitude variance
    pub fn adjustment_variance(&self) -> f64 {
        if self.total_revisions < 2 {
            return 0.0;
        }
        let mean = self.mean_adjustment();
        (self.sum_sq_adjustment / self.total_revisions as f64 - mean * mean).max(0.0)
    }

    /// Adjustment magnitude standard deviation
    pub fn adjustment_std(&self) -> f64 {
        self.adjustment_variance().sqrt()
    }

    /// Revision rate (revisions per check)
    pub fn revision_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.total_revisions as f64 / self.total_checks as f64
    }

    /// Directional bias: positive = more tightening, negative = more loosening
    pub fn direction_bias(&self) -> f64 {
        if self.total_revisions == 0 {
            return 0.0;
        }
        (self.tighten_count as f64 - self.loosen_count as f64) / self.total_revisions as f64
    }

    /// Cooldown block rate
    pub fn cooldown_block_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.cooldown_blocks as f64 / self.total_checks as f64
    }
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Goal revision engine — decides when and how to adjust trading targets
pub struct Revision {
    config: RevisionConfig,

    /// Registered goals keyed by ID
    goals: HashMap<String, GoalRevisionState>,

    /// Per-goal EMA revision pressure
    ema_pressure: HashMap<String, f64>,
    /// Per-goal EMA initialised flag
    ema_initialized: HashMap<String, bool>,

    /// Global tick counter
    current_tick: u64,

    /// Revision history (all events)
    history: Vec<RevisionEvent>,

    /// Windowed records
    recent: VecDeque<RevisionRecord>,

    /// Running statistics
    stats: RevisionStats,
}

impl Default for Revision {
    fn default() -> Self {
        Self::new()
    }
}

impl Revision {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(RevisionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: RevisionConfig) -> Self {
        assert!(
            config.ema_decay > 0.0 && config.ema_decay < 1.0,
            "ema_decay must be in (0, 1)"
        );
        assert!(config.window_size > 0, "window_size must be > 0");
        assert!(
            config.loosen_threshold >= 0.0 && config.loosen_threshold <= 1.0,
            "loosen_threshold must be in [0, 1]"
        );
        assert!(
            config.tighten_threshold >= 0.0 && config.tighten_threshold <= 1.0,
            "tighten_threshold must be in [0, 1]"
        );
        assert!(
            config.loosen_threshold < config.tighten_threshold,
            "loosen_threshold must be < tighten_threshold"
        );
        assert!(config.base_adjustment > 0.0, "base_adjustment must be > 0");
        assert!(
            config.max_single_adjustment >= config.base_adjustment,
            "max_single_adjustment must be >= base_adjustment"
        );
        assert!(
            config.max_cumulative_adjustment > 0.0,
            "max_cumulative_adjustment must be > 0"
        );
        assert!(config.dead_zone >= 0.0, "dead_zone must be >= 0");
        assert!(
            config.proportional_gain >= 0.0,
            "proportional_gain must be >= 0"
        );

        Self {
            config,
            goals: HashMap::new(),
            ema_pressure: HashMap::new(),
            ema_initialized: HashMap::new(),
            current_tick: 0,
            history: Vec::new(),
            recent: VecDeque::new(),
            stats: RevisionStats::default(),
        }
    }

    /// Main processing function — checks a single goal and potentially revises it
    pub fn process(
        &mut self,
        goal_id: &str,
        input: &PerformanceInput,
        tick: u64,
    ) -> Result<RevisionCheckResult> {
        self.current_tick = tick;
        self.check_and_revise(goal_id, input)
    }

    /// Batch process — checks all registered goals
    pub fn process_all(
        &mut self,
        inputs: &HashMap<String, PerformanceInput>,
        tick: u64,
    ) -> Result<Vec<RevisionCheckResult>> {
        self.current_tick = tick;
        let goal_ids: Vec<String> = self.goals.keys().cloned().collect();
        let mut results = Vec::new();

        for id in &goal_ids {
            if let Some(input) = inputs.get(id) {
                results.push(self.check_and_revise(id, input)?);
            } else {
                // Tick cooldown even if no input provided
                if let Some(state) = self.goals.get_mut(id) {
                    if state.cooldown_remaining > 0 {
                        state.cooldown_remaining = state.cooldown_remaining.saturating_sub(1);
                    }
                }
            }
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Goal registration
    // -----------------------------------------------------------------------

    /// Register a goal for revision tracking
    pub fn register_goal(&mut self, id: &str, target: f64) -> Result<()> {
        if self.goals.contains_key(id) {
            return Err(Error::InvalidInput(format!(
                "Goal '{}' already registered",
                id
            )));
        }
        if !target.is_finite() || target == 0.0 {
            return Err(Error::InvalidInput(
                "target must be finite and non-zero".into(),
            ));
        }

        self.goals.insert(
            id.to_string(),
            GoalRevisionState {
                id: id.to_string(),
                original_target: target,
                current_target: target,
                cumulative_adjustment: 0.0,
                revision_count: 0,
                cooldown_remaining: 0,
                locked: false,
            },
        );
        self.ema_pressure.insert(id.to_string(), 0.0);
        self.ema_initialized.insert(id.to_string(), false);
        self.stats.registered_goals = self.goals.len();
        Ok(())
    }

    /// Deregister a goal
    pub fn deregister_goal(&mut self, id: &str) -> Result<()> {
        if self.goals.remove(id).is_none() {
            return Err(Error::NotFound(format!("Goal '{}' not found", id)));
        }
        self.ema_pressure.remove(id);
        self.ema_initialized.remove(id);
        self.stats.registered_goals = self.goals.len();
        Ok(())
    }

    /// Manually set a goal's target (bypasses revision logic)
    pub fn set_target(&mut self, id: &str, new_target: f64) -> Result<()> {
        if !new_target.is_finite() || new_target == 0.0 {
            return Err(Error::InvalidInput(
                "target must be finite and non-zero".into(),
            ));
        }
        let state = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("Goal '{}' not found", id)))?;

        state.current_target = new_target;
        state.cumulative_adjustment =
            (new_target - state.original_target) / state.original_target.abs();
        Ok(())
    }

    /// Lock a goal from further automatic revisions
    pub fn lock_goal(&mut self, id: &str) -> Result<()> {
        let state = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("Goal '{}' not found", id)))?;
        state.locked = true;
        Ok(())
    }

    /// Unlock a goal for automatic revisions
    pub fn unlock_goal(&mut self, id: &str) -> Result<()> {
        let state = self
            .goals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("Goal '{}' not found", id)))?;
        state.locked = false;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Core revision logic
    // -----------------------------------------------------------------------

    fn check_and_revise(
        &mut self,
        goal_id: &str,
        input: &PerformanceInput,
    ) -> Result<RevisionCheckResult> {
        if !input.achievement_rate.is_finite() {
            return Err(Error::InvalidInput(
                "achievement_rate must be finite".into(),
            ));
        }

        let achievement = input.achievement_rate.clamp(0.0, 1.0);
        self.stats.total_checks += 1;

        // Compute revision pressure:
        // Negative = loosen pressure, Positive = tighten pressure
        let pressure = if achievement < self.config.loosen_threshold {
            let deviation = self.config.loosen_threshold - achievement;
            if deviation > self.config.dead_zone {
                -(deviation * self.config.proportional_gain).min(1.0)
            } else {
                0.0
            }
        } else if achievement > self.config.tighten_threshold {
            let deviation = achievement - self.config.tighten_threshold;
            if deviation > self.config.dead_zone {
                (deviation * self.config.proportional_gain).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Add external pressure if provided
        let total_pressure = if let Some(ext) = input.external_pressure {
            (pressure + ext.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)
        } else {
            pressure
        };

        // Update EMA
        let initialized = self.ema_initialized.get(goal_id).copied().unwrap_or(false);
        if !initialized {
            self.ema_pressure
                .insert(goal_id.to_string(), total_pressure);
            self.ema_initialized.insert(goal_id.to_string(), true);
        } else {
            let alpha = self.config.ema_decay;
            let prev = self.ema_pressure.get(goal_id).copied().unwrap_or(0.0);
            self.ema_pressure.insert(
                goal_id.to_string(),
                alpha * total_pressure + (1.0 - alpha) * prev,
            );
        }

        let smoothed_pressure = self.ema_pressure.get(goal_id).copied().unwrap_or(0.0);

        // Track peak pressure
        if smoothed_pressure.abs() > self.stats.peak_pressure {
            self.stats.peak_pressure = smoothed_pressure.abs();
        }

        // Get goal state
        let state = match self.goals.get(goal_id) {
            Some(s) => s.clone(),
            None => {
                return Err(Error::NotFound(format!("Goal '{}' not found", goal_id)));
            }
        };

        // Check locked
        if state.locked {
            self.stats.lock_blocks += 1;
            return Ok(RevisionCheckResult {
                revised: false,
                event: None,
                pressure: smoothed_pressure,
                in_cooldown: state.cooldown_remaining > 0,
                locked: true,
            });
        }

        // Check cooldown
        if state.cooldown_remaining > 0 {
            // Tick down cooldown
            if let Some(s) = self.goals.get_mut(goal_id) {
                s.cooldown_remaining = s.cooldown_remaining.saturating_sub(1);
            }
            self.stats.cooldown_blocks += 1;
            return Ok(RevisionCheckResult {
                revised: false,
                event: None,
                pressure: smoothed_pressure,
                in_cooldown: true,
                locked: false,
            });
        }

        // Check max revisions
        if self.config.max_revisions > 0 && state.revision_count >= self.config.max_revisions {
            if let Some(s) = self.goals.get_mut(goal_id) {
                s.locked = true;
            }
            self.stats.lock_blocks += 1;
            return Ok(RevisionCheckResult {
                revised: false,
                event: None,
                pressure: smoothed_pressure,
                in_cooldown: false,
                locked: true,
            });
        }

        // Check if pressure is strong enough (using raw pressure, not smoothed)
        if total_pressure.abs() < 1e-9 {
            return Ok(RevisionCheckResult {
                revised: false,
                event: None,
                pressure: smoothed_pressure,
                in_cooldown: false,
                locked: false,
            });
        }

        // Determine direction and compute adjustment
        let direction = if total_pressure < 0.0 {
            RevisionDirection::Loosen
        } else {
            RevisionDirection::Tighten
        };

        // Scale adjustment proportionally to pressure magnitude
        let adjustment_frac = (self.config.base_adjustment * total_pressure.abs())
            .min(self.config.max_single_adjustment);

        // Check cumulative limit
        let new_cumulative = match direction {
            RevisionDirection::Loosen => state.cumulative_adjustment - adjustment_frac,
            RevisionDirection::Tighten => state.cumulative_adjustment + adjustment_frac,
        };

        if new_cumulative.abs() > self.config.max_cumulative_adjustment {
            // Would exceed cumulative limit — clip or skip
            let headroom =
                self.config.max_cumulative_adjustment - state.cumulative_adjustment.abs();
            if headroom <= 1e-12 {
                return Ok(RevisionCheckResult {
                    revised: false,
                    event: None,
                    pressure: smoothed_pressure,
                    in_cooldown: false,
                    locked: false,
                });
            }
            // Apply only the available headroom
            let clipped = adjustment_frac.min(headroom);
            return self.apply_revision(
                goal_id,
                direction,
                clipped,
                achievement,
                smoothed_pressure,
            );
        }

        self.apply_revision(
            goal_id,
            direction,
            adjustment_frac,
            achievement,
            smoothed_pressure,
        )
    }

    fn apply_revision(
        &mut self,
        goal_id: &str,
        direction: RevisionDirection,
        adjustment_frac: f64,
        achievement_rate: f64,
        pressure: f64,
    ) -> Result<RevisionCheckResult> {
        let state = self
            .goals
            .get_mut(goal_id)
            .ok_or_else(|| Error::NotFound(format!("Goal '{}' not found", goal_id)))?;

        let old_target = state.current_target;

        // Apply adjustment: loosen reduces target, tighten increases target
        // For negative targets (short bias), the direction is inverted
        let signed_adjustment = match direction {
            RevisionDirection::Loosen => {
                if old_target > 0.0 {
                    -adjustment_frac * old_target.abs()
                } else {
                    adjustment_frac * old_target.abs()
                }
            }
            RevisionDirection::Tighten => {
                if old_target > 0.0 {
                    adjustment_frac * old_target.abs()
                } else {
                    -adjustment_frac * old_target.abs()
                }
            }
        };

        let new_target = old_target + signed_adjustment;
        state.current_target = new_target;
        state.cumulative_adjustment = match direction {
            RevisionDirection::Loosen => state.cumulative_adjustment - adjustment_frac,
            RevisionDirection::Tighten => state.cumulative_adjustment + adjustment_frac,
        };
        state.revision_count += 1;
        state.cooldown_remaining = self.config.cooldown_ticks;

        let reason = match direction {
            RevisionDirection::Loosen => format!(
                "Achievement rate {:.1}% below threshold {:.1}% — loosened by {:.2}%",
                achievement_rate * 100.0,
                self.config.loosen_threshold * 100.0,
                adjustment_frac * 100.0,
            ),
            RevisionDirection::Tighten => format!(
                "Achievement rate {:.1}% above threshold {:.1}% — tightened by {:.2}%",
                achievement_rate * 100.0,
                self.config.tighten_threshold * 100.0,
                adjustment_frac * 100.0,
            ),
        };

        let event = RevisionEvent {
            goal_id: goal_id.to_string(),
            direction,
            adjustment: adjustment_frac,
            old_target,
            new_target,
            achievement_rate,
            pressure,
            tick: self.current_tick,
            reason,
        };

        self.history.push(event.clone());

        // Update stats
        self.stats.total_revisions += 1;
        match direction {
            RevisionDirection::Loosen => self.stats.loosen_count += 1,
            RevisionDirection::Tighten => self.stats.tighten_count += 1,
        }
        self.stats.sum_abs_adjustment += adjustment_frac;
        self.stats.sum_sq_adjustment += adjustment_frac * adjustment_frac;
        if adjustment_frac > self.stats.max_adjustment {
            self.stats.max_adjustment = adjustment_frac;
        }

        // Record for windowed diagnostics
        self.recent.push_back(RevisionRecord {
            goal_id: goal_id.to_string(),
            direction,
            adjustment_magnitude: adjustment_frac,
            tick: self.current_tick,
        });
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        Ok(RevisionCheckResult {
            revised: true,
            event: Some(event),
            pressure,
            in_cooldown: false,
            locked: false,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get a goal's revision state
    pub fn goal_state(&self, id: &str) -> Option<&GoalRevisionState> {
        self.goals.get(id)
    }

    /// Get all goal states
    pub fn all_goal_states(&self) -> &HashMap<String, GoalRevisionState> {
        &self.goals
    }

    /// Get the current target for a goal
    pub fn current_target(&self, id: &str) -> Option<f64> {
        self.goals.get(id).map(|s| s.current_target)
    }

    /// Get the original target for a goal
    pub fn original_target(&self, id: &str) -> Option<f64> {
        self.goals.get(id).map(|s| s.original_target)
    }

    /// Get the cumulative adjustment fraction for a goal
    pub fn cumulative_adjustment(&self, id: &str) -> Option<f64> {
        self.goals.get(id).map(|s| s.cumulative_adjustment)
    }

    /// Get the current smoothed revision pressure for a goal
    pub fn revision_pressure(&self, id: &str) -> f64 {
        self.ema_pressure.get(id).copied().unwrap_or(0.0)
    }

    /// Whether a goal is in cooldown
    pub fn in_cooldown(&self, id: &str) -> bool {
        self.goals
            .get(id)
            .map(|s| s.cooldown_remaining > 0)
            .unwrap_or(false)
    }

    /// Whether a goal is locked
    pub fn is_locked(&self, id: &str) -> bool {
        self.goals.get(id).map(|s| s.locked).unwrap_or(false)
    }

    /// How many revisions have been applied to a goal
    pub fn revision_count(&self, id: &str) -> usize {
        self.goals.get(id).map(|s| s.revision_count).unwrap_or(0)
    }

    /// Number of registered goals
    pub fn goal_count(&self) -> usize {
        self.goals.len()
    }

    /// Full revision history
    pub fn history(&self) -> &[RevisionEvent] {
        &self.history
    }

    /// Recent events for a specific goal
    pub fn goal_history(&self, id: &str) -> Vec<&RevisionEvent> {
        self.history.iter().filter(|e| e.goal_id == id).collect()
    }

    /// Running statistics
    pub fn stats(&self) -> &RevisionStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Revision frequency in the recent window (revisions per tick span)
    pub fn windowed_revision_frequency(&self) -> f64 {
        if self.recent.len() < 2 {
            return 0.0;
        }
        let first_tick = self.recent.front().map(|r| r.tick).unwrap_or(0);
        let last_tick = self.recent.back().map(|r| r.tick).unwrap_or(0);
        let span = last_tick.saturating_sub(first_tick);
        if span == 0 {
            return self.recent.len() as f64;
        }
        self.recent.len() as f64 / span as f64
    }

    /// Mean adjustment magnitude over the recent window
    pub fn windowed_mean_adjustment(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.adjustment_magnitude).sum();
        sum / self.recent.len() as f64
    }

    /// Direction bias over the recent window
    /// Positive = more tightening, negative = more loosening
    pub fn windowed_direction_bias(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let tighten = self
            .recent
            .iter()
            .filter(|r| r.direction == RevisionDirection::Tighten)
            .count();
        let loosen = self
            .recent
            .iter()
            .filter(|r| r.direction == RevisionDirection::Loosen)
            .count();
        (tighten as f64 - loosen as f64) / self.recent.len() as f64
    }

    /// How many distinct goals were revised in the recent window
    pub fn windowed_goal_diversity(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for r in &self.recent {
            seen.insert(r.goal_id.clone());
        }
        seen.len()
    }

    /// Whether revision activity is increasing (comparing first/second half of window)
    pub fn is_revision_rate_increasing(&self) -> bool {
        let n = self.recent.len();
        if n < 6 {
            return false;
        }
        let mid = n / 2;
        let first_half = mid;
        let second_half = n - mid;

        // Compare counts (both halves may span different tick ranges, but
        // for a simple heuristic we just compare counts)
        second_half as f64 > first_half as f64 * 1.3
    }

    /// Confidence based on check count
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_checks as f64;
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

    /// Reset all state, keeping configuration and goal registrations.
    /// Goal targets are restored to their original values.
    pub fn reset(&mut self) {
        for (id, state) in &mut self.goals {
            state.current_target = state.original_target;
            state.cumulative_adjustment = 0.0;
            state.revision_count = 0;
            state.cooldown_remaining = 0;
            state.locked = false;
            self.ema_pressure.insert(id.clone(), 0.0);
            self.ema_initialized.insert(id.clone(), false);
        }
        // Need to clone ids to avoid double-borrow
        let ids: Vec<String> = self.goals.keys().cloned().collect();
        for id in ids {
            self.ema_pressure.insert(id.clone(), 0.0);
            self.ema_initialized.insert(id, false);
        }
        self.current_tick = 0;
        self.history.clear();
        self.recent.clear();
        self.stats = RevisionStats {
            registered_goals: self.goals.len(),
            ..Default::default()
        };
    }

    /// Reset everything including goal registrations
    pub fn reset_all(&mut self) {
        self.goals.clear();
        self.ema_pressure.clear();
        self.ema_initialized.clear();
        self.current_tick = 0;
        self.history.clear();
        self.recent.clear();
        self.stats = RevisionStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn low_performance() -> PerformanceInput {
        PerformanceInput {
            achievement_rate: 0.1,
            external_pressure: None,
        }
    }

    fn high_performance() -> PerformanceInput {
        PerformanceInput {
            achievement_rate: 0.95,
            external_pressure: None,
        }
    }

    fn mid_performance() -> PerformanceInput {
        PerformanceInput {
            achievement_rate: 0.5,
            external_pressure: None,
        }
    }

    #[test]
    fn test_basic() {
        let mut instance = Revision::new();
        assert!(
            instance
                .process("nonexistent", &mid_performance(), 0)
                .is_err()
        );
    }

    #[test]
    fn test_new_default() {
        let r = Revision::new();
        assert_eq!(r.goal_count(), 0);
        assert_eq!(r.stats().total_revisions, 0);
        assert_eq!(r.stats().total_checks, 0);
    }

    #[test]
    fn test_register_goal() {
        let mut r = Revision::new();
        r.register_goal("g1", 100.0).unwrap();
        assert_eq!(r.goal_count(), 1);
        assert_eq!(r.current_target("g1"), Some(100.0));
        assert_eq!(r.original_target("g1"), Some(100.0));
    }

    #[test]
    fn test_register_duplicate() {
        let mut r = Revision::new();
        r.register_goal("g1", 100.0).unwrap();
        assert!(r.register_goal("g1", 200.0).is_err());
    }

    #[test]
    fn test_register_invalid_target() {
        let mut r = Revision::new();
        assert!(r.register_goal("g", 0.0).is_err());
        assert!(r.register_goal("g", f64::NAN).is_err());
        assert!(r.register_goal("g", f64::INFINITY).is_err());
    }

    #[test]
    fn test_deregister_goal() {
        let mut r = Revision::new();
        r.register_goal("g1", 100.0).unwrap();
        r.deregister_goal("g1").unwrap();
        assert_eq!(r.goal_count(), 0);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let mut r = Revision::new();
        assert!(r.deregister_goal("nope").is_err());
    }

    #[test]
    fn test_no_revision_in_dead_zone() {
        let mut r = Revision::with_config(RevisionConfig {
            loosen_threshold: 0.3,
            tighten_threshold: 0.8,
            dead_zone: 0.05,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        // Achievement = 0.28, deviation from loosen(0.3) = 0.02 < dead_zone(0.05)
        let input = PerformanceInput {
            achievement_rate: 0.28,
            external_pressure: None,
        };
        let result = r.process("g", &input, 1).unwrap();
        assert!(!result.revised);
    }

    #[test]
    fn test_loosen_on_low_performance() {
        let mut r = Revision::with_config(RevisionConfig {
            loosen_threshold: 0.3,
            tighten_threshold: 0.8,
            base_adjustment: 0.10,
            max_single_adjustment: 0.20,
            dead_zone: 0.0,
            proportional_gain: 1.0,
            cooldown_ticks: 0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        let result = r.process("g", &low_performance(), 1).unwrap();
        assert!(result.revised);
        let event = result.event.unwrap();
        assert_eq!(event.direction, RevisionDirection::Loosen);
        assert!(event.new_target < event.old_target);
        assert!(r.current_target("g").unwrap() < 100.0);
    }

    #[test]
    fn test_tighten_on_high_performance() {
        let mut r = Revision::with_config(RevisionConfig {
            loosen_threshold: 0.3,
            tighten_threshold: 0.8,
            base_adjustment: 0.10,
            max_single_adjustment: 0.20,
            dead_zone: 0.0,
            proportional_gain: 1.0,
            cooldown_ticks: 0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        let result = r.process("g", &high_performance(), 1).unwrap();
        assert!(result.revised);
        let event = result.event.unwrap();
        assert_eq!(event.direction, RevisionDirection::Tighten);
        assert!(event.new_target > event.old_target);
        assert!(r.current_target("g").unwrap() > 100.0);
    }

    #[test]
    fn test_no_revision_in_band() {
        let mut r = Revision::new();
        r.register_goal("g", 100.0).unwrap();

        let result = r.process("g", &mid_performance(), 1).unwrap();
        assert!(!result.revised);
        assert_eq!(r.current_target("g").unwrap(), 100.0);
    }

    #[test]
    fn test_cooldown_blocks_revision() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 5,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        // First revision should go through
        let r1 = r.process("g", &low_performance(), 1).unwrap();
        assert!(r1.revised);

        // Next one should be blocked by cooldown
        let r2 = r.process("g", &low_performance(), 2).unwrap();
        assert!(!r2.revised);
        assert!(r2.in_cooldown);
    }

    #[test]
    fn test_cooldown_expires() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 2,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap(); // revised, cooldown = 2
        r.process("g", &low_performance(), 2).unwrap(); // cooldown = 1
        r.process("g", &low_performance(), 3).unwrap(); // cooldown = 0 (ticked to 0)
        let r4 = r.process("g", &low_performance(), 4).unwrap(); // should go through
        assert!(r4.revised);
    }

    #[test]
    fn test_max_revisions_locks_goal() {
        let mut r = Revision::with_config(RevisionConfig {
            max_revisions: 2,
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap(); // revision 1
        r.process("g", &low_performance(), 2).unwrap(); // revision 2

        let r3 = r.process("g", &low_performance(), 3).unwrap();
        assert!(!r3.revised);
        assert!(r3.locked);
        assert!(r.is_locked("g"));
    }

    #[test]
    fn test_max_cumulative_adjustment() {
        let mut r = Revision::with_config(RevisionConfig {
            base_adjustment: 0.30,
            max_single_adjustment: 0.30,
            max_cumulative_adjustment: 0.50,
            cooldown_ticks: 0,
            dead_zone: 0.0,
            proportional_gain: 5.0, // high gain to ensure full base_adjustment
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap(); // -30%
        let cum1 = r.cumulative_adjustment("g").unwrap();
        assert!((cum1.abs() - 0.30).abs() < 0.01);

        r.process("g", &low_performance(), 2).unwrap(); // would be -30% more but cumulative caps at 50%
        let cum2 = r.cumulative_adjustment("g").unwrap();
        assert!(cum2.abs() <= 0.50 + 1e-9);
    }

    #[test]
    fn test_lock_and_unlock() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.lock_goal("g").unwrap();
        assert!(r.is_locked("g"));

        let result = r.process("g", &low_performance(), 1).unwrap();
        assert!(!result.revised);
        assert!(result.locked);

        r.unlock_goal("g").unwrap();
        assert!(!r.is_locked("g"));

        let result2 = r.process("g", &low_performance(), 2).unwrap();
        assert!(result2.revised);
    }

    #[test]
    fn test_lock_nonexistent() {
        let mut r = Revision::new();
        assert!(r.lock_goal("nope").is_err());
    }

    #[test]
    fn test_unlock_nonexistent() {
        let mut r = Revision::new();
        assert!(r.unlock_goal("nope").is_err());
    }

    #[test]
    fn test_set_target() {
        let mut r = Revision::new();
        r.register_goal("g", 100.0).unwrap();
        r.set_target("g", 120.0).unwrap();
        assert_eq!(r.current_target("g").unwrap(), 120.0);
        assert_eq!(r.original_target("g").unwrap(), 100.0);
        // cumulative should reflect the change
        assert!((r.cumulative_adjustment("g").unwrap() - 0.20).abs() < 1e-9);
    }

    #[test]
    fn test_set_target_invalid() {
        let mut r = Revision::new();
        r.register_goal("g", 100.0).unwrap();
        assert!(r.set_target("g", 0.0).is_err());
        assert!(r.set_target("g", f64::NAN).is_err());
    }

    #[test]
    fn test_set_target_nonexistent() {
        let mut r = Revision::new();
        assert!(r.set_target("nope", 100.0).is_err());
    }

    #[test]
    fn test_external_pressure_loosen() {
        let mut r = Revision::with_config(RevisionConfig {
            loosen_threshold: 0.3,
            tighten_threshold: 0.8,
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        // Mid performance (no natural pressure) + negative external pressure => loosen
        let input = PerformanceInput {
            achievement_rate: 0.5,
            external_pressure: Some(-0.5),
        };
        let result = r.process("g", &input, 1).unwrap();
        assert!(result.revised);
        assert_eq!(
            result.event.as_ref().unwrap().direction,
            RevisionDirection::Loosen
        );
    }

    #[test]
    fn test_external_pressure_tighten() {
        let mut r = Revision::with_config(RevisionConfig {
            loosen_threshold: 0.3,
            tighten_threshold: 0.8,
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        let input = PerformanceInput {
            achievement_rate: 0.5,
            external_pressure: Some(0.5),
        };
        let result = r.process("g", &input, 1).unwrap();
        assert!(result.revised);
        assert_eq!(
            result.event.as_ref().unwrap().direction,
            RevisionDirection::Tighten
        );
    }

    #[test]
    fn test_revision_pressure_signal() {
        let mut r = Revision::new();
        r.register_goal("g", 100.0).unwrap();

        // Low performance → negative pressure
        r.process("g", &low_performance(), 1).unwrap();
        assert!(r.revision_pressure("g") < 0.0);

        // Nonexistent goal → 0
        assert_eq!(r.revision_pressure("nope"), 0.0);
    }

    #[test]
    fn test_history_recorded() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap();
        r.process("g", &low_performance(), 2).unwrap();

        assert_eq!(r.history().len(), 2);
        assert_eq!(r.goal_history("g").len(), 2);
        assert_eq!(r.goal_history("nonexistent").len(), 0);
    }

    #[test]
    fn test_process_all() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("a", 100.0).unwrap();
        r.register_goal("b", 200.0).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), low_performance());
        inputs.insert("b".to_string(), high_performance());

        let results = r.process_all(&inputs, 1).unwrap();
        assert_eq!(results.len(), 2);

        // Both should have been revised
        let revised_count = results.iter().filter(|r| r.revised).count();
        assert_eq!(revised_count, 2);
    }

    #[test]
    fn test_process_all_missing_input() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 5,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("a", 100.0).unwrap();
        r.register_goal("b", 200.0).unwrap();

        // Only provide input for "a"
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), low_performance());

        let results = r.process_all(&inputs, 1).unwrap();
        // Only "a" should have a result
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_stats_tracking() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap();
        r.process("g", &high_performance(), 2).unwrap();
        r.process("g", &mid_performance(), 3).unwrap();

        assert_eq!(r.stats().total_checks, 3);
        assert_eq!(r.stats().total_revisions, 2);
        assert_eq!(r.stats().loosen_count, 1);
        assert_eq!(r.stats().tighten_count, 1);
    }

    #[test]
    fn test_stats_mean_adjustment() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            base_adjustment: 0.10,
            max_cumulative_adjustment: 1.0,
            proportional_gain: 5.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap();
        r.process("g", &low_performance(), 2).unwrap();

        assert!(r.stats().mean_adjustment() > 0.0);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = RevisionStats::default();
        assert_eq!(stats.total_revisions, 0);
        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.mean_adjustment(), 0.0);
        assert_eq!(stats.revision_rate(), 0.0);
        assert_eq!(stats.direction_bias(), 0.0);
        assert_eq!(stats.cooldown_block_rate(), 0.0);
        assert_eq!(stats.adjustment_variance(), 0.0);
        assert_eq!(stats.adjustment_std(), 0.0);
    }

    #[test]
    fn test_stats_revision_rate() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap(); // revised
        r.process("g", &mid_performance(), 2).unwrap(); // not revised

        assert!((r.stats().revision_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_direction_bias() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap(); // loosen
        r.process("g", &low_performance(), 2).unwrap(); // loosen

        // All loosening → bias = -1.0
        assert!((r.stats().direction_bias() - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_stats_cooldown_block_rate() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 3,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap(); // revised
        r.process("g", &low_performance(), 2).unwrap(); // blocked
        r.process("g", &low_performance(), 3).unwrap(); // blocked

        // 2 out of 3 checks were cooldown-blocked
        assert!((r.stats().cooldown_block_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_lock_blocks() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();
        r.lock_goal("g").unwrap();

        r.process("g", &low_performance(), 1).unwrap();
        r.process("g", &low_performance(), 2).unwrap();

        assert_eq!(r.stats().lock_blocks, 2);
    }

    #[test]
    fn test_windowed_mean_adjustment() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap();
        r.process("g", &low_performance(), 2).unwrap();

        assert!(r.windowed_mean_adjustment() > 0.0);
    }

    #[test]
    fn test_windowed_mean_adjustment_empty() {
        let r = Revision::new();
        assert_eq!(r.windowed_mean_adjustment(), 0.0);
    }

    #[test]
    fn test_windowed_direction_bias() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        // All loosening
        r.process("g", &low_performance(), 1).unwrap();
        r.process("g", &low_performance(), 2).unwrap();
        assert!(r.windowed_direction_bias() < 0.0);
    }

    #[test]
    fn test_windowed_direction_bias_empty() {
        let r = Revision::new();
        assert_eq!(r.windowed_direction_bias(), 0.0);
    }

    #[test]
    fn test_windowed_goal_diversity() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("a", 100.0).unwrap();
        r.register_goal("b", 200.0).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), low_performance());
        inputs.insert("b".to_string(), low_performance());
        r.process_all(&inputs, 1).unwrap();

        assert_eq!(r.windowed_goal_diversity(), 2);
    }

    #[test]
    fn test_windowed_revision_frequency() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        // 3 revisions over ticks 1..=3 (span = 2)
        r.process("g", &low_performance(), 1).unwrap();
        r.process("g", &low_performance(), 2).unwrap();
        r.process("g", &low_performance(), 3).unwrap();

        // frequency = 3 / 2 = 1.5
        assert!((r.windowed_revision_frequency() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_revision_frequency_empty() {
        let r = Revision::new();
        assert_eq!(r.windowed_revision_frequency(), 0.0);
    }

    #[test]
    fn test_windowed_revision_frequency_single() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();
        r.process("g", &low_performance(), 1).unwrap();
        // Only 1 record => 0.0 (need >= 2 for a span)
        assert_eq!(r.windowed_revision_frequency(), 0.0);
    }

    #[test]
    fn test_is_revision_rate_increasing_insufficient() {
        let r = Revision::new();
        assert!(!r.is_revision_rate_increasing());
    }

    #[test]
    fn test_confidence_increases() {
        let mut r = Revision::with_config(RevisionConfig {
            min_samples: 3,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        assert_eq!(r.confidence(), 0.0);

        for i in 0..6 {
            r.process("g", &mid_performance(), i).unwrap();
        }
        let c6 = r.confidence();
        assert!(c6 > 0.0);

        for i in 6..20 {
            r.process("g", &mid_performance(), i).unwrap();
        }
        let c20 = r.confidence();
        assert!(c20 >= c6);
        assert!(c20 <= 1.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut r = Revision::with_config(RevisionConfig {
            window_size: 3,
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 10.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        for i in 0..10 {
            r.process("g", &low_performance(), i).unwrap();
        }
        assert!(r.recent.len() <= 3);
    }

    #[test]
    fn test_reset() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();
        r.process("g", &low_performance(), 1).unwrap();
        assert!(r.current_target("g").unwrap() != 100.0);
        assert!(r.stats().total_revisions > 0);

        r.reset();
        assert_eq!(r.current_target("g").unwrap(), 100.0);
        assert_eq!(r.original_target("g").unwrap(), 100.0);
        assert_eq!(r.cumulative_adjustment("g").unwrap(), 0.0);
        assert_eq!(r.stats().total_revisions, 0);
        assert_eq!(r.stats().total_checks, 0);
        assert!(r.history().is_empty());
        assert!(r.recent.is_empty());
        assert!(!r.is_locked("g"));
        assert_eq!(r.goal_count(), 1); // goal kept
    }

    #[test]
    fn test_reset_all() {
        let mut r = Revision::new();
        r.register_goal("g", 100.0).unwrap();
        r.reset_all();
        assert_eq!(r.goal_count(), 0);
    }

    #[test]
    fn test_in_cooldown_accessor() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 5,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();
        assert!(!r.in_cooldown("g"));

        r.process("g", &low_performance(), 1).unwrap();
        assert!(r.in_cooldown("g"));
    }

    #[test]
    fn test_revision_count_accessor() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();
        assert_eq!(r.revision_count("g"), 0);
        assert_eq!(r.revision_count("nonexistent"), 0);

        r.process("g", &low_performance(), 1).unwrap();
        assert_eq!(r.revision_count("g"), 1);
    }

    #[test]
    fn test_invalid_achievement_rate() {
        let mut r = Revision::new();
        r.register_goal("g", 100.0).unwrap();

        let input = PerformanceInput {
            achievement_rate: f64::NAN,
            external_pressure: None,
        };
        assert!(r.process("g", &input, 1).is_err());
    }

    #[test]
    fn test_nonexistent_goal_process() {
        let mut r = Revision::new();
        assert!(r.process("nope", &mid_performance(), 1).is_err());
    }

    #[test]
    fn test_peak_pressure_tracked() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap();
        assert!(r.stats().peak_pressure > 0.0);
    }

    #[test]
    fn test_negative_target_loosen_direction() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("short_target", -100.0).unwrap();

        // Low performance → loosen → for a negative target, loosening means
        // making the target less negative (easier to achieve)
        let result = r.process("short_target", &low_performance(), 1).unwrap();
        assert!(result.revised);
        let event = result.event.unwrap();
        assert_eq!(event.direction, RevisionDirection::Loosen);
        // For negative target, loosening means new_target > old_target (less negative)
        assert!(event.new_target > event.old_target);
    }

    #[test]
    fn test_proportional_scaling() {
        let mut r = Revision::with_config(RevisionConfig {
            loosen_threshold: 0.3,
            tighten_threshold: 0.8,
            base_adjustment: 0.10,
            max_single_adjustment: 0.50,
            proportional_gain: 2.0,
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        // Achievement = 0.0 → deviation = 0.3, pressure = min(0.3*2.0, 1.0) = 0.6
        // adjustment = min(0.10 * 0.6, 0.50) = 0.06
        let result = r
            .process(
                "g",
                &PerformanceInput {
                    achievement_rate: 0.0,
                    external_pressure: None,
                },
                1,
            )
            .unwrap();

        assert!(result.revised);
        let adj = result.event.unwrap().adjustment;
        assert!((adj - 0.06).abs() < 1e-9);
    }

    #[test]
    fn test_event_reason_contains_percentage() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        let result = r.process("g", &low_performance(), 1).unwrap();
        let event = result.event.unwrap();
        assert!(event.reason.contains('%'));
    }

    #[test]
    #[should_panic(expected = "ema_decay must be in (0, 1)")]
    fn test_invalid_config_ema_decay() {
        Revision::with_config(RevisionConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn test_invalid_config_window_size() {
        Revision::with_config(RevisionConfig {
            window_size: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "loosen_threshold must be < tighten_threshold")]
    fn test_invalid_config_threshold_order() {
        Revision::with_config(RevisionConfig {
            loosen_threshold: 0.8,
            tighten_threshold: 0.3,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "base_adjustment must be > 0")]
    fn test_invalid_config_base_adjustment() {
        Revision::with_config(RevisionConfig {
            base_adjustment: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "max_single_adjustment must be >= base_adjustment")]
    fn test_invalid_config_max_single_less_than_base() {
        Revision::with_config(RevisionConfig {
            base_adjustment: 0.10,
            max_single_adjustment: 0.05,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "max_cumulative_adjustment must be > 0")]
    fn test_invalid_config_max_cumulative() {
        Revision::with_config(RevisionConfig {
            max_cumulative_adjustment: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "dead_zone must be >= 0")]
    fn test_invalid_config_dead_zone() {
        Revision::with_config(RevisionConfig {
            dead_zone: -0.1,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "proportional_gain must be >= 0")]
    fn test_invalid_config_proportional_gain() {
        Revision::with_config(RevisionConfig {
            proportional_gain: -1.0,
            ..Default::default()
        });
    }

    #[test]
    fn test_goal_state_accessor() {
        let mut r = Revision::new();
        r.register_goal("g", 100.0).unwrap();

        let state = r.goal_state("g").unwrap();
        assert_eq!(state.id, "g");
        assert_eq!(state.original_target, 100.0);
        assert_eq!(state.current_target, 100.0);
        assert_eq!(state.cumulative_adjustment, 0.0);
        assert_eq!(state.revision_count, 0);
        assert!(!state.locked);
    }

    #[test]
    fn test_goal_state_nonexistent() {
        let r = Revision::new();
        assert!(r.goal_state("nope").is_none());
    }

    #[test]
    fn test_all_goal_states() {
        let mut r = Revision::new();
        r.register_goal("a", 100.0).unwrap();
        r.register_goal("b", 200.0).unwrap();
        assert_eq!(r.all_goal_states().len(), 2);
    }

    #[test]
    fn test_current_target_nonexistent() {
        let r = Revision::new();
        assert!(r.current_target("nope").is_none());
    }

    #[test]
    fn test_original_target_nonexistent() {
        let r = Revision::new();
        assert!(r.original_target("nope").is_none());
    }

    #[test]
    fn test_is_locked_nonexistent() {
        let r = Revision::new();
        assert!(!r.is_locked("nope"));
    }

    #[test]
    fn test_in_cooldown_nonexistent() {
        let r = Revision::new();
        assert!(!r.in_cooldown("nope"));
    }

    #[test]
    fn test_registered_goals_stat_updated() {
        let mut r = Revision::new();
        assert_eq!(r.stats().registered_goals, 0);

        r.register_goal("a", 100.0).unwrap();
        assert_eq!(r.stats().registered_goals, 1);

        r.register_goal("b", 200.0).unwrap();
        assert_eq!(r.stats().registered_goals, 2);

        r.deregister_goal("a").unwrap();
        assert_eq!(r.stats().registered_goals, 1);
    }

    #[test]
    fn test_multiple_goals_independent_cooldowns() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 2,
            dead_zone: 0.0,
            max_cumulative_adjustment: 1.0,
            ..Default::default()
        });
        r.register_goal("a", 100.0).unwrap();
        r.register_goal("b", 200.0).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), low_performance());
        inputs.insert("b".to_string(), low_performance());

        // Both revised on tick 1
        let results = r.process_all(&inputs, 1).unwrap();
        assert!(results.iter().all(|r| r.revised));

        // Both blocked on tick 2
        let results2 = r.process_all(&inputs, 2).unwrap();
        assert!(results2.iter().all(|r| !r.revised));
    }

    #[test]
    fn test_max_adjustment_tracked() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            base_adjustment: 0.10,
            max_cumulative_adjustment: 1.0,
            proportional_gain: 5.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        r.process("g", &low_performance(), 1).unwrap();
        assert!(r.stats().max_adjustment > 0.0);
    }

    #[test]
    fn test_achievement_clamped_to_01() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        // Achievement > 1.0 should be clamped
        let input = PerformanceInput {
            achievement_rate: 1.5,
            external_pressure: None,
        };
        let result = r.process("g", &input, 1).unwrap();
        // Clamped to 1.0 which is above tighten_threshold (0.8), so should tighten
        assert!(result.revised);
    }

    #[test]
    fn test_achievement_negative_clamped() {
        let mut r = Revision::with_config(RevisionConfig {
            cooldown_ticks: 0,
            dead_zone: 0.0,
            ..Default::default()
        });
        r.register_goal("g", 100.0).unwrap();

        let input = PerformanceInput {
            achievement_rate: -0.5,
            external_pressure: None,
        };
        let result = r.process("g", &input, 1).unwrap();
        // Clamped to 0.0, which is below loosen_threshold => loosen
        assert!(result.revised);
        assert_eq!(result.event.unwrap().direction, RevisionDirection::Loosen);
    }
}
