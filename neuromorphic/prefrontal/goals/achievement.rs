//! Goal achievement tracking
//!
//! Part of the Prefrontal region
//! Component: goals
//!
//! Tracks progress towards trading goals, detecting achievement streaks,
//! milestone completions, and regression. Provides EMA-smoothed achievement
//! rate signals and windowed diagnostics for upstream decision-making.
//!
//! ## Features
//!
//! - **Achievement rate tracking**: Rolling ratio of achieved vs attempted goals
//! - **Streak detection**: Consecutive achievement / failure streaks with
//!   configurable streak thresholds for triggering regime changes
//! - **Milestone tracking**: Named milestones with target values and deadlines;
//!   emits milestone events when crossed
//! - **EMA-smoothed signals**: Achievement rate and progress velocity are
//!   smoothed to reduce noise
//! - **Regression detection**: Compares first/second half of the observation
//!   window to flag declining achievement rates
//! - **Running statistics**: Total achievements, failures, best/worst streaks,
//!   time-to-achievement distribution
//! - **Windowed diagnostics**: Recent achievement rate, streak length, and
//!   progress velocity for downstream consumers

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the achievement tracker
#[derive(Debug, Clone)]
pub struct AchievementConfig {
    /// EMA decay factor for achievement rate smoothing (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum observations before EMA is considered initialised
    pub min_samples: usize,
    /// Sliding window size for windowed diagnostics
    pub window_size: usize,
    /// Streak length that triggers a "hot streak" flag
    pub hot_streak_threshold: usize,
    /// Streak length that triggers a "cold streak" flag
    pub cold_streak_threshold: usize,
    /// Progress velocity EMA decay (0 < decay < 1)
    pub velocity_ema_decay: f64,
    /// Maximum number of milestones that can be registered
    pub max_milestones: usize,
}

impl Default for AchievementConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.1,
            min_samples: 5,
            window_size: 100,
            hot_streak_threshold: 5,
            cold_streak_threshold: 3,
            velocity_ema_decay: 0.15,
            max_milestones: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Outcome of a single goal attempt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalOutcome {
    /// Goal was achieved
    Achieved,
    /// Goal was not achieved (failed / expired)
    Failed,
}

/// A named milestone with a target value
#[derive(Debug, Clone)]
pub struct Milestone {
    /// Unique name
    pub name: String,
    /// Target value to cross
    pub target: f64,
    /// Optional deadline (timestamp)
    pub deadline: Option<u64>,
    /// Whether the milestone has been reached
    pub reached: bool,
    /// Timestamp when milestone was reached (if ever)
    pub reached_at: Option<u64>,
}

/// Result of recording an outcome
#[derive(Debug, Clone)]
pub struct AchievementResult {
    /// Current smoothed achievement rate (0.0–1.0)
    pub achievement_rate: f64,
    /// Current streak length (positive = wins, negative = losses)
    pub streak: i64,
    /// Whether a hot streak is active
    pub hot_streak: bool,
    /// Whether a cold streak is active
    pub cold_streak: bool,
    /// Current smoothed progress velocity
    pub velocity: f64,
    /// Milestones crossed by this observation (names)
    pub milestones_crossed: Vec<String>,
}

/// Record kept per observation for windowed analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ObservationRecord {
    achieved: bool,
    progress: f64,
    timestamp: u64,
}

/// Running statistics
#[derive(Debug, Clone)]
pub struct AchievementStats {
    /// Total outcomes recorded
    pub total_outcomes: u64,
    /// Total achievements
    pub total_achieved: u64,
    /// Total failures
    pub total_failed: u64,
    /// Longest winning streak ever observed
    pub best_streak: u64,
    /// Longest losing streak ever observed
    pub worst_streak: u64,
    /// Current streak (positive = wins, negative = losses)
    pub current_streak: i64,
    /// Sum of time-to-achievement for achieved goals (for mean calc)
    pub sum_time_to_achieve: f64,
    /// Count of goals with known time-to-achievement
    pub time_to_achieve_count: u64,
    /// Total milestones reached
    pub milestones_reached: u64,
    /// Peak smoothed achievement rate observed
    pub peak_achievement_rate: f64,
    /// Trough smoothed achievement rate observed
    pub trough_achievement_rate: f64,
}

impl Default for AchievementStats {
    fn default() -> Self {
        Self {
            total_outcomes: 0,
            total_achieved: 0,
            total_failed: 0,
            best_streak: 0,
            worst_streak: 0,
            current_streak: 0,
            sum_time_to_achieve: 0.0,
            time_to_achieve_count: 0,
            milestones_reached: 0,
            peak_achievement_rate: 0.0,
            trough_achievement_rate: 1.0,
        }
    }
}

impl AchievementStats {
    /// Overall achievement rate
    pub fn achievement_rate(&self) -> f64 {
        if self.total_outcomes == 0 {
            return 0.0;
        }
        self.total_achieved as f64 / self.total_outcomes as f64
    }

    /// Mean time to achievement (returns 0 if no data)
    pub fn mean_time_to_achieve(&self) -> f64 {
        if self.time_to_achieve_count == 0 {
            return 0.0;
        }
        self.sum_time_to_achieve / self.time_to_achieve_count as f64
    }

    /// Failure rate
    pub fn failure_rate(&self) -> f64 {
        if self.total_outcomes == 0 {
            return 0.0;
        }
        self.total_failed as f64 / self.total_outcomes as f64
    }
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Tracks goal achievement with EMA-smoothed signals and streak detection
pub struct Achievement {
    config: AchievementConfig,

    // EMA state
    ema_achievement_rate: f64,
    ema_initialized: bool,

    // Velocity EMA
    ema_velocity: f64,
    velocity_initialized: bool,
    last_progress: Option<f64>,

    // Streak tracking
    current_streak: i64,

    // Milestones
    milestones: Vec<Milestone>,
    cumulative_progress: f64,

    // Windowed observations
    recent: VecDeque<ObservationRecord>,

    // Running stats
    stats: AchievementStats,
}

impl Default for Achievement {
    fn default() -> Self {
        Self::new()
    }
}

impl Achievement {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(AchievementConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AchievementConfig) -> Self {
        assert!(
            config.ema_decay > 0.0 && config.ema_decay < 1.0,
            "ema_decay must be in (0, 1)"
        );
        assert!(
            config.velocity_ema_decay > 0.0 && config.velocity_ema_decay < 1.0,
            "velocity_ema_decay must be in (0, 1)"
        );
        assert!(config.window_size > 0, "window_size must be > 0");
        assert!(
            config.hot_streak_threshold > 0,
            "hot_streak_threshold must be > 0"
        );
        assert!(
            config.cold_streak_threshold > 0,
            "cold_streak_threshold must be > 0"
        );

        Self {
            config,
            ema_achievement_rate: 0.0,
            ema_initialized: false,
            ema_velocity: 0.0,
            velocity_initialized: false,
            last_progress: None,
            current_streak: 0,
            milestones: Vec::new(),
            cumulative_progress: 0.0,
            recent: VecDeque::new(),
            stats: AchievementStats::default(),
        }
    }

    /// Main processing function — records an outcome and returns the updated signals
    pub fn process(
        &mut self,
        outcome: GoalOutcome,
        progress: f64,
        timestamp: u64,
    ) -> Result<AchievementResult> {
        self.record(outcome, progress, timestamp, None)
    }

    /// Record a goal outcome with an optional time-to-achievement duration
    pub fn record(
        &mut self,
        outcome: GoalOutcome,
        progress: f64,
        timestamp: u64,
        time_to_achieve: Option<f64>,
    ) -> Result<AchievementResult> {
        if !progress.is_finite() {
            return Err(Error::InvalidInput("progress must be finite".into()));
        }

        let achieved = outcome == GoalOutcome::Achieved;
        let value = if achieved { 1.0 } else { 0.0 };

        // Update EMA achievement rate
        if !self.ema_initialized {
            self.ema_achievement_rate = value;
            self.ema_initialized = true;
        } else {
            let alpha = self.config.ema_decay;
            self.ema_achievement_rate = alpha * value + (1.0 - alpha) * self.ema_achievement_rate;
        }

        // Update velocity EMA
        let velocity = if let Some(prev) = self.last_progress {
            progress - prev
        } else {
            0.0
        };
        if !self.velocity_initialized {
            self.ema_velocity = velocity;
            self.velocity_initialized = true;
        } else {
            let alpha = self.config.velocity_ema_decay;
            self.ema_velocity = alpha * velocity + (1.0 - alpha) * self.ema_velocity;
        }
        self.last_progress = Some(progress);

        // Update cumulative progress
        self.cumulative_progress += progress.max(0.0);

        // Update streak
        if achieved {
            if self.current_streak >= 0 {
                self.current_streak += 1;
            } else {
                self.current_streak = 1;
            }
        } else if self.current_streak <= 0 {
            self.current_streak -= 1;
        } else {
            self.current_streak = -1;
        }

        // Update stats
        self.stats.total_outcomes += 1;
        if achieved {
            self.stats.total_achieved += 1;
            if let Some(tta) = time_to_achieve {
                if tta >= 0.0 && tta.is_finite() {
                    self.stats.sum_time_to_achieve += tta;
                    self.stats.time_to_achieve_count += 1;
                }
            }
        } else {
            self.stats.total_failed += 1;
        }
        self.stats.current_streak = self.current_streak;

        if self.current_streak > 0 {
            let streak_u64 = self.current_streak as u64;
            if streak_u64 > self.stats.best_streak {
                self.stats.best_streak = streak_u64;
            }
        } else {
            let streak_u64 = self.current_streak.unsigned_abs();
            if streak_u64 > self.stats.worst_streak {
                self.stats.worst_streak = streak_u64;
            }
        }

        // Track peak / trough achievement rates (after min_samples)
        if self.stats.total_outcomes as usize >= self.config.min_samples {
            if self.ema_achievement_rate > self.stats.peak_achievement_rate {
                self.stats.peak_achievement_rate = self.ema_achievement_rate;
            }
            if self.ema_achievement_rate < self.stats.trough_achievement_rate {
                self.stats.trough_achievement_rate = self.ema_achievement_rate;
            }
        }

        // Check milestones
        let milestones_crossed = self.check_milestones(timestamp);

        // Record in window
        self.recent.push_back(ObservationRecord {
            achieved,
            progress,
            timestamp,
        });
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        let hot = self.current_streak >= self.config.hot_streak_threshold as i64;
        let cold = self.current_streak <= -(self.config.cold_streak_threshold as i64);

        Ok(AchievementResult {
            achievement_rate: self.ema_achievement_rate,
            streak: self.current_streak,
            hot_streak: hot,
            cold_streak: cold,
            velocity: self.ema_velocity,
            milestones_crossed,
        })
    }

    // -----------------------------------------------------------------------
    // Milestones
    // -----------------------------------------------------------------------

    /// Register a named milestone
    pub fn add_milestone(&mut self, name: &str, target: f64, deadline: Option<u64>) -> Result<()> {
        if self.milestones.len() >= self.config.max_milestones {
            return Err(Error::InvalidInput(format!(
                "Maximum milestones ({}) reached",
                self.config.max_milestones
            )));
        }
        if self.milestones.iter().any(|m| m.name == name) {
            return Err(Error::InvalidInput(format!(
                "Milestone '{}' already exists",
                name
            )));
        }
        if !target.is_finite() || target < 0.0 {
            return Err(Error::InvalidInput(
                "Milestone target must be finite and non-negative".into(),
            ));
        }
        self.milestones.push(Milestone {
            name: name.to_string(),
            target,
            deadline,
            reached: false,
            reached_at: None,
        });
        Ok(())
    }

    /// Check milestones against cumulative progress, returning newly crossed names
    fn check_milestones(&mut self, timestamp: u64) -> Vec<String> {
        let mut crossed = Vec::new();
        for m in &mut self.milestones {
            if !m.reached && self.cumulative_progress >= m.target {
                m.reached = true;
                m.reached_at = Some(timestamp);
                crossed.push(m.name.clone());
                self.stats.milestones_reached += 1;
            }
        }
        crossed
    }

    /// Get all milestones
    pub fn milestones(&self) -> &[Milestone] {
        &self.milestones
    }

    /// Get a specific milestone by name
    pub fn milestone(&self, name: &str) -> Option<&Milestone> {
        self.milestones.iter().find(|m| m.name == name)
    }

    /// Check if all milestones have been reached
    pub fn all_milestones_reached(&self) -> bool {
        !self.milestones.is_empty() && self.milestones.iter().all(|m| m.reached)
    }

    /// Count of unreached milestones whose deadline has passed
    pub fn expired_milestones(&self, current_time: u64) -> usize {
        self.milestones
            .iter()
            .filter(|m| !m.reached && m.deadline.map_or(false, |d| current_time > d))
            .count()
    }

    // -----------------------------------------------------------------------
    // Signal accessors
    // -----------------------------------------------------------------------

    /// Current EMA-smoothed achievement rate
    pub fn smoothed_achievement_rate(&self) -> f64 {
        if !self.ema_initialized {
            return 0.0;
        }
        self.ema_achievement_rate
    }

    /// Current EMA-smoothed progress velocity
    pub fn smoothed_velocity(&self) -> f64 {
        if !self.velocity_initialized {
            return 0.0;
        }
        self.ema_velocity
    }

    /// Whether the EMA has enough samples to be meaningful
    pub fn is_warmed_up(&self) -> bool {
        self.stats.total_outcomes as usize >= self.config.min_samples
    }

    /// Current streak (positive = consecutive achievements, negative = consecutive failures)
    pub fn current_streak(&self) -> i64 {
        self.current_streak
    }

    /// Whether a hot streak is active
    pub fn is_hot_streak(&self) -> bool {
        self.current_streak >= self.config.hot_streak_threshold as i64
    }

    /// Whether a cold streak is active
    pub fn is_cold_streak(&self) -> bool {
        self.current_streak <= -(self.config.cold_streak_threshold as i64)
    }

    /// Cumulative progress across all recorded observations
    pub fn cumulative_progress(&self) -> f64 {
        self.cumulative_progress
    }

    /// Total observations recorded
    pub fn observation_count(&self) -> u64 {
        self.stats.total_outcomes
    }

    /// Running statistics
    pub fn stats(&self) -> &AchievementStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Achievement rate over the recent window
    pub fn windowed_achievement_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let achieved = self.recent.iter().filter(|r| r.achieved).count();
        achieved as f64 / self.recent.len() as f64
    }

    /// Mean progress over the recent window
    pub fn windowed_mean_progress(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.progress).sum();
        sum / self.recent.len() as f64
    }

    /// Progress standard deviation over the recent window
    pub fn windowed_progress_std(&self) -> f64 {
        if self.recent.len() < 2 {
            return 0.0;
        }
        let mean = self.windowed_mean_progress();
        let var: f64 = self
            .recent
            .iter()
            .map(|r| (r.progress - mean).powi(2))
            .sum::<f64>()
            / (self.recent.len() - 1) as f64;
        var.sqrt()
    }

    /// Detect whether achievement rate is declining (regression)
    ///
    /// Compares the achievement rate in the first half of the window to the
    /// second half.  Returns `true` if the second half is meaningfully worse.
    pub fn is_regressing(&self) -> bool {
        let n = self.recent.len();
        if n < 6 {
            return false;
        }
        let mid = n / 2;
        let first_half_rate =
            self.recent.iter().take(mid).filter(|r| r.achieved).count() as f64 / mid as f64;
        let second_half_rate =
            self.recent.iter().skip(mid).filter(|r| r.achieved).count() as f64 / (n - mid) as f64;

        // Regression if second half is at least 20% lower
        second_half_rate < first_half_rate * 0.8
    }

    /// Detect whether achievement rate is improving
    pub fn is_improving(&self) -> bool {
        let n = self.recent.len();
        if n < 6 {
            return false;
        }
        let mid = n / 2;
        let first_half_rate =
            self.recent.iter().take(mid).filter(|r| r.achieved).count() as f64 / mid as f64;
        let second_half_rate =
            self.recent.iter().skip(mid).filter(|r| r.achieved).count() as f64 / (n - mid) as f64;

        second_half_rate > first_half_rate * 1.2
    }

    /// Confidence score based on sample count
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_outcomes as f64;
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

    /// Reset all state, keeping configuration
    pub fn reset(&mut self) {
        self.ema_achievement_rate = 0.0;
        self.ema_initialized = false;
        self.ema_velocity = 0.0;
        self.velocity_initialized = false;
        self.last_progress = None;
        self.current_streak = 0;
        self.milestones.iter_mut().for_each(|m| {
            m.reached = false;
            m.reached_at = None;
        });
        self.cumulative_progress = 0.0;
        self.recent.clear();
        self.stats = AchievementStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn achieved(progress: f64) -> (GoalOutcome, f64) {
        (GoalOutcome::Achieved, progress)
    }

    #[allow(dead_code)]
    fn failed(progress: f64) -> (GoalOutcome, f64) {
        (GoalOutcome::Failed, progress)
    }

    #[test]
    fn test_basic() {
        let mut instance = Achievement::new();
        assert!(instance.process(GoalOutcome::Achieved, 1.0, 0).is_ok());
    }

    #[test]
    fn test_single_achievement() {
        let mut a = Achievement::new();
        let r = a.process(GoalOutcome::Achieved, 1.0, 100).unwrap();
        assert!((r.achievement_rate - 1.0).abs() < 1e-9);
        assert_eq!(r.streak, 1);
        assert!(!r.hot_streak);
        assert!(!r.cold_streak);
        assert_eq!(a.stats().total_achieved, 1);
        assert_eq!(a.stats().total_failed, 0);
    }

    #[test]
    fn test_single_failure() {
        let mut a = Achievement::new();
        let r = a.process(GoalOutcome::Failed, 0.3, 100).unwrap();
        assert!((r.achievement_rate - 0.0).abs() < 1e-9);
        assert_eq!(r.streak, -1);
        assert!(!r.hot_streak);
        assert!(!r.cold_streak);
        assert_eq!(a.stats().total_failed, 1);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut a = Achievement::with_config(AchievementConfig {
            ema_decay: 0.5,
            ..Default::default()
        });
        // First: Achieved => ema = 1.0
        a.process(GoalOutcome::Achieved, 1.0, 1).unwrap();
        assert!((a.smoothed_achievement_rate() - 1.0).abs() < 1e-9);

        // Second: Failed => ema = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        a.process(GoalOutcome::Failed, 0.0, 2).unwrap();
        assert!((a.smoothed_achievement_rate() - 0.5).abs() < 1e-9);

        // Third: Failed => ema = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        a.process(GoalOutcome::Failed, 0.0, 3).unwrap();
        assert!((a.smoothed_achievement_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_streak_tracking_wins() {
        let mut a = Achievement::with_config(AchievementConfig {
            hot_streak_threshold: 3,
            ..Default::default()
        });
        for i in 0..3 {
            let r = a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
            if i < 2 {
                assert!(!r.hot_streak);
            } else {
                assert!(r.hot_streak);
                assert_eq!(r.streak, 3);
            }
        }
        assert_eq!(a.stats().best_streak, 3);
    }

    #[test]
    fn test_streak_tracking_losses() {
        let mut a = Achievement::with_config(AchievementConfig {
            cold_streak_threshold: 2,
            ..Default::default()
        });
        let r1 = a.process(GoalOutcome::Failed, 0.0, 1).unwrap();
        assert!(!r1.cold_streak);
        assert_eq!(r1.streak, -1);

        let r2 = a.process(GoalOutcome::Failed, 0.0, 2).unwrap();
        assert!(r2.cold_streak);
        assert_eq!(r2.streak, -2);
        assert_eq!(a.stats().worst_streak, 2);
    }

    #[test]
    fn test_streak_resets_on_switch() {
        let mut a = Achievement::new();
        a.process(GoalOutcome::Achieved, 1.0, 1).unwrap();
        a.process(GoalOutcome::Achieved, 1.0, 2).unwrap();
        assert_eq!(a.current_streak(), 2);

        a.process(GoalOutcome::Failed, 0.0, 3).unwrap();
        assert_eq!(a.current_streak(), -1);

        a.process(GoalOutcome::Achieved, 1.0, 4).unwrap();
        assert_eq!(a.current_streak(), 1);
    }

    #[test]
    fn test_milestone_basic() {
        let mut a = Achievement::new();
        a.add_milestone("first_win", 1.0, None).unwrap();
        a.add_milestone("ten_progress", 10.0, None).unwrap();

        let r = a.process(GoalOutcome::Achieved, 1.0, 100).unwrap();
        assert_eq!(r.milestones_crossed.len(), 1);
        assert_eq!(r.milestones_crossed[0], "first_win");
        assert!(a.milestone("first_win").unwrap().reached);
        assert!(!a.milestone("ten_progress").unwrap().reached);
    }

    #[test]
    fn test_milestone_cumulative() {
        let mut a = Achievement::new();
        a.add_milestone("five_total", 5.0, None).unwrap();

        for i in 0..4 {
            let r = a.process(GoalOutcome::Achieved, 1.5, i).unwrap();
            // cumulative: 1.5, 3.0, 4.5, 6.0
            if i < 3 {
                assert!(r.milestones_crossed.is_empty());
            } else {
                assert_eq!(r.milestones_crossed, vec!["five_total"]);
            }
        }
    }

    #[test]
    fn test_milestone_duplicate_name() {
        let mut a = Achievement::new();
        a.add_milestone("m1", 1.0, None).unwrap();
        let err = a.add_milestone("m1", 2.0, None);
        assert!(err.is_err());
    }

    #[test]
    fn test_milestone_max_limit() {
        let mut a = Achievement::with_config(AchievementConfig {
            max_milestones: 2,
            ..Default::default()
        });
        a.add_milestone("m1", 1.0, None).unwrap();
        a.add_milestone("m2", 2.0, None).unwrap();
        assert!(a.add_milestone("m3", 3.0, None).is_err());
    }

    #[test]
    fn test_milestone_invalid_target() {
        let mut a = Achievement::new();
        assert!(a.add_milestone("bad", -1.0, None).is_err());
        assert!(a.add_milestone("bad2", f64::NAN, None).is_err());
        assert!(a.add_milestone("bad3", f64::INFINITY, None).is_err());
    }

    #[test]
    fn test_all_milestones_reached() {
        let mut a = Achievement::new();
        assert!(!a.all_milestones_reached()); // no milestones => false
        a.add_milestone("m1", 1.0, None).unwrap();
        a.add_milestone("m2", 2.0, None).unwrap();
        assert!(!a.all_milestones_reached());

        a.process(GoalOutcome::Achieved, 1.0, 1).unwrap();
        assert!(!a.all_milestones_reached());

        a.process(GoalOutcome::Achieved, 1.0, 2).unwrap();
        assert!(a.all_milestones_reached());
    }

    #[test]
    fn test_expired_milestones() {
        let mut a = Achievement::new();
        a.add_milestone("m1", 100.0, Some(50)).unwrap();
        a.add_milestone("m2", 100.0, Some(200)).unwrap();
        a.add_milestone("m3", 100.0, None).unwrap();

        // At time 100, m1's deadline (50) has passed, m2's (200) has not
        assert_eq!(a.expired_milestones(100), 1);
    }

    #[test]
    fn test_velocity_tracking() {
        let mut a = Achievement::with_config(AchievementConfig {
            velocity_ema_decay: 1.0 - 1e-15, // near-instant to simplify
            ..AchievementConfig {
                velocity_ema_decay: 0.99,
                ..Default::default()
            }
        });
        // progress: 0.5, 1.0, 1.5 => velocities: 0, 0.5, 0.5
        a.process(GoalOutcome::Achieved, 0.5, 1).unwrap();
        // First velocity = 0 (no previous)
        assert!(a.smoothed_velocity().abs() < 1e-9);

        a.process(GoalOutcome::Achieved, 1.0, 2).unwrap();
        // velocity = 0.5
        assert!((a.smoothed_velocity() - 0.5).abs() < 0.05);

        a.process(GoalOutcome::Achieved, 1.5, 3).unwrap();
        assert!((a.smoothed_velocity() - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_windowed_achievement_rate() {
        let mut a = Achievement::with_config(AchievementConfig {
            window_size: 10,
            ..Default::default()
        });
        // 7 achieved, 3 failed
        for i in 0..10 {
            let outcome = if i < 7 {
                GoalOutcome::Achieved
            } else {
                GoalOutcome::Failed
            };
            a.process(outcome, 1.0, i).unwrap();
        }
        assert!((a.windowed_achievement_rate() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_achievement_rate_empty() {
        let a = Achievement::new();
        assert_eq!(a.windowed_achievement_rate(), 0.0);
    }

    #[test]
    fn test_windowed_mean_progress() {
        let mut a = Achievement::with_config(AchievementConfig {
            window_size: 4,
            ..Default::default()
        });
        // progress: 1.0, 2.0, 3.0, 4.0 => mean = 2.5
        for i in 1..=4 {
            a.process(GoalOutcome::Achieved, i as f64, i as u64)
                .unwrap();
        }
        assert!((a.windowed_mean_progress() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_progress_std() {
        let mut a = Achievement::with_config(AchievementConfig {
            window_size: 4,
            ..Default::default()
        });
        // Constant progress => std = 0
        for i in 0..4 {
            a.process(GoalOutcome::Achieved, 5.0, i).unwrap();
        }
        assert!(a.windowed_progress_std().abs() < 1e-9);
    }

    #[test]
    fn test_windowed_progress_std_variable() {
        let mut a = Achievement::with_config(AchievementConfig {
            window_size: 100,
            ..Default::default()
        });
        a.process(GoalOutcome::Achieved, 1.0, 0).unwrap();
        a.process(GoalOutcome::Achieved, 3.0, 1).unwrap();
        // std of [1.0, 3.0] with sample std = sqrt(2)
        let std = a.windowed_progress_std();
        assert!((std - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_progress_std_insufficient() {
        let mut a = Achievement::new();
        assert_eq!(a.windowed_progress_std(), 0.0);
        a.process(GoalOutcome::Achieved, 1.0, 0).unwrap();
        assert_eq!(a.windowed_progress_std(), 0.0); // need >= 2
    }

    #[test]
    fn test_is_regressing() {
        let mut a = Achievement::with_config(AchievementConfig {
            window_size: 20,
            ..Default::default()
        });
        // First 5: all achieved, last 5: all failed  (need >= 6)
        for i in 0..5 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        for i in 5..10 {
            a.process(GoalOutcome::Failed, 0.0, i).unwrap();
        }
        assert!(a.is_regressing());
    }

    #[test]
    fn test_not_regressing_consistent() {
        let mut a = Achievement::with_config(AchievementConfig {
            window_size: 20,
            ..Default::default()
        });
        for i in 0..10 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        assert!(!a.is_regressing());
    }

    #[test]
    fn test_not_regressing_insufficient_data() {
        let mut a = Achievement::new();
        for i in 0..3 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        assert!(!a.is_regressing()); // need >= 6
    }

    #[test]
    fn test_is_improving() {
        let mut a = Achievement::with_config(AchievementConfig {
            window_size: 20,
            ..Default::default()
        });
        // First half: mostly failures, second half: all achieved
        for i in 0..5 {
            a.process(GoalOutcome::Failed, 0.0, i).unwrap();
        }
        for i in 5..10 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        assert!(a.is_improving());
    }

    #[test]
    fn test_stats_achievement_rate() {
        let mut a = Achievement::new();
        a.process(GoalOutcome::Achieved, 1.0, 1).unwrap();
        a.process(GoalOutcome::Achieved, 1.0, 2).unwrap();
        a.process(GoalOutcome::Failed, 0.0, 3).unwrap();
        a.process(GoalOutcome::Achieved, 1.0, 4).unwrap();
        // 3 out of 4
        assert!((a.stats().achievement_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_stats_failure_rate() {
        let mut a = Achievement::new();
        a.process(GoalOutcome::Achieved, 1.0, 1).unwrap();
        a.process(GoalOutcome::Failed, 0.0, 2).unwrap();
        assert!((a.stats().failure_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_mean_time_to_achieve() {
        let mut a = Achievement::new();
        a.record(GoalOutcome::Achieved, 1.0, 100, Some(10.0))
            .unwrap();
        a.record(GoalOutcome::Achieved, 1.0, 200, Some(20.0))
            .unwrap();
        a.record(GoalOutcome::Failed, 0.0, 300, None).unwrap();
        assert!((a.stats().mean_time_to_achieve() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_mean_time_to_achieve_empty() {
        let a = Achievement::new();
        assert_eq!(a.stats().mean_time_to_achieve(), 0.0);
    }

    #[test]
    fn test_is_warmed_up() {
        let mut a = Achievement::with_config(AchievementConfig {
            min_samples: 3,
            ..Default::default()
        });
        assert!(!a.is_warmed_up());
        a.process(GoalOutcome::Achieved, 1.0, 1).unwrap();
        a.process(GoalOutcome::Achieved, 1.0, 2).unwrap();
        assert!(!a.is_warmed_up());
        a.process(GoalOutcome::Achieved, 1.0, 3).unwrap();
        assert!(a.is_warmed_up());
    }

    #[test]
    fn test_confidence_increases_with_samples() {
        let mut a = Achievement::with_config(AchievementConfig {
            min_samples: 5,
            ..Default::default()
        });
        let c0 = a.confidence();
        assert_eq!(c0, 0.0);

        for i in 0..10 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        let c10 = a.confidence();
        assert!(c10 > c0);

        for i in 10..20 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        let c20 = a.confidence();
        assert!(c20 >= c10);
        assert!(c20 <= 1.0);
    }

    #[test]
    fn test_confidence_caps_at_one() {
        let mut a = Achievement::with_config(AchievementConfig {
            min_samples: 2,
            ..Default::default()
        });
        for i in 0..100 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        assert!((a.confidence() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_eviction() {
        let mut a = Achievement::with_config(AchievementConfig {
            window_size: 3,
            ..Default::default()
        });
        for i in 0..5 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        assert_eq!(a.recent.len(), 3);
    }

    #[test]
    fn test_reset() {
        let mut a = Achievement::new();
        a.add_milestone("m1", 1.0, None).unwrap();
        for i in 0..5 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        assert!(a.milestone("m1").unwrap().reached);
        assert!(a.observation_count() > 0);

        a.reset();
        assert_eq!(a.observation_count(), 0);
        assert_eq!(a.current_streak(), 0);
        assert_eq!(a.smoothed_achievement_rate(), 0.0);
        assert_eq!(a.smoothed_velocity(), 0.0);
        assert_eq!(a.cumulative_progress(), 0.0);
        assert!(a.recent.is_empty());
        // Milestones are kept but reset
        assert!(!a.milestone("m1").unwrap().reached);
    }

    #[test]
    fn test_invalid_progress() {
        let mut a = Achievement::new();
        assert!(a.process(GoalOutcome::Achieved, f64::NAN, 0).is_err());
        assert!(a.process(GoalOutcome::Achieved, f64::INFINITY, 0).is_err());
    }

    #[test]
    fn test_cumulative_progress_only_positive() {
        let mut a = Achievement::new();
        a.process(GoalOutcome::Failed, -5.0, 1).unwrap();
        // Negative progress should not contribute to cumulative
        assert!((a.cumulative_progress() - 0.0).abs() < 1e-9);

        a.process(GoalOutcome::Achieved, 3.0, 2).unwrap();
        assert!((a.cumulative_progress() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_peak_trough_tracking() {
        let mut a = Achievement::with_config(AchievementConfig {
            ema_decay: 0.5,
            min_samples: 2,
            ..Default::default()
        });
        // Build up to peak
        a.process(GoalOutcome::Achieved, 1.0, 1).unwrap();
        a.process(GoalOutcome::Achieved, 1.0, 2).unwrap(); // ema = 1.0, min_samples met
        assert!((a.stats().peak_achievement_rate - 1.0).abs() < 1e-9);

        // Bring down
        a.process(GoalOutcome::Failed, 0.0, 3).unwrap(); // ema = 0.5
        a.process(GoalOutcome::Failed, 0.0, 4).unwrap(); // ema = 0.25
        assert!(a.stats().trough_achievement_rate < 0.5);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = AchievementStats::default();
        assert_eq!(stats.total_outcomes, 0);
        assert_eq!(stats.total_achieved, 0);
        assert_eq!(stats.total_failed, 0);
        assert_eq!(stats.best_streak, 0);
        assert_eq!(stats.worst_streak, 0);
        assert_eq!(stats.current_streak, 0);
        assert_eq!(stats.achievement_rate(), 0.0);
        assert_eq!(stats.failure_rate(), 0.0);
        assert_eq!(stats.mean_time_to_achieve(), 0.0);
    }

    #[test]
    fn test_best_and_worst_streaks_accumulate() {
        let mut a = Achievement::new();
        // 3 wins
        for i in 0..3 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        assert_eq!(a.stats().best_streak, 3);

        // 2 losses
        for i in 3..5 {
            a.process(GoalOutcome::Failed, 0.0, i).unwrap();
        }
        assert_eq!(a.stats().worst_streak, 2);
        assert_eq!(a.stats().best_streak, 3); // unchanged

        // 5 wins — new best
        for i in 5..10 {
            a.process(GoalOutcome::Achieved, 1.0, i).unwrap();
        }
        assert_eq!(a.stats().best_streak, 5);
    }

    #[test]
    fn test_milestone_reached_at_timestamp() {
        let mut a = Achievement::new();
        a.add_milestone("first", 2.0, None).unwrap();
        a.process(GoalOutcome::Achieved, 1.0, 100).unwrap();
        assert!(a.milestone("first").unwrap().reached_at.is_none());

        a.process(GoalOutcome::Achieved, 1.5, 200).unwrap();
        assert_eq!(a.milestone("first").unwrap().reached_at, Some(200));
    }

    #[test]
    fn test_milestones_reached_stat() {
        let mut a = Achievement::new();
        a.add_milestone("m1", 1.0, None).unwrap();
        a.add_milestone("m2", 2.0, None).unwrap();
        a.process(GoalOutcome::Achieved, 1.0, 1).unwrap();
        assert_eq!(a.stats().milestones_reached, 1);
        a.process(GoalOutcome::Achieved, 1.0, 2).unwrap();
        assert_eq!(a.stats().milestones_reached, 2);
    }

    #[test]
    #[should_panic(expected = "ema_decay must be in (0, 1)")]
    fn test_invalid_config_ema_decay_zero() {
        Achievement::with_config(AchievementConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "ema_decay must be in (0, 1)")]
    fn test_invalid_config_ema_decay_one() {
        Achievement::with_config(AchievementConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "velocity_ema_decay must be in (0, 1)")]
    fn test_invalid_config_velocity_ema_decay() {
        Achievement::with_config(AchievementConfig {
            velocity_ema_decay: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn test_invalid_config_window_size() {
        Achievement::with_config(AchievementConfig {
            window_size: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "hot_streak_threshold must be > 0")]
    fn test_invalid_config_hot_streak_threshold() {
        Achievement::with_config(AchievementConfig {
            hot_streak_threshold: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "cold_streak_threshold must be > 0")]
    fn test_invalid_config_cold_streak_threshold() {
        Achievement::with_config(AchievementConfig {
            cold_streak_threshold: 0,
            ..Default::default()
        });
    }

    #[test]
    fn test_default_has_no_observations() {
        let a = Achievement::new();
        assert_eq!(a.observation_count(), 0);
        assert_eq!(a.current_streak(), 0);
        assert!(!a.is_hot_streak());
        assert!(!a.is_cold_streak());
        assert!(!a.is_warmed_up());
    }

    #[test]
    fn test_negative_time_to_achieve_ignored() {
        let mut a = Achievement::new();
        a.record(GoalOutcome::Achieved, 1.0, 1, Some(-10.0))
            .unwrap();
        assert_eq!(a.stats().time_to_achieve_count, 0);
    }

    #[test]
    fn test_process_returns_milestones_crossed() {
        let mut a = Achievement::new();
        a.add_milestone("quick", 0.5, None).unwrap();
        let r = a.process(GoalOutcome::Achieved, 0.5, 1).unwrap();
        assert_eq!(r.milestones_crossed, vec!["quick"]);

        // Should not cross again
        let r2 = a.process(GoalOutcome::Achieved, 0.5, 2).unwrap();
        assert!(r2.milestones_crossed.is_empty());
    }
}
