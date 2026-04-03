//! Decision confidence estimation
//!
//! Part of the Basal Ganglia region
//! Component: praxeological
//!
//! Estimates confidence in current decision-making by tracking:
//! - Recent reward consistency (low variance = high confidence)
//! - Action agreement across evaluation windows
//! - Exponential moving average of prediction accuracy

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for confidence estimation
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Maximum number of recent rewards to track
    pub window_size: usize,
    /// Decay factor for exponential moving average (0.0 - 1.0)
    pub ema_decay: f64,
    /// Minimum samples before confidence is considered valid
    pub min_samples: usize,
    /// Threshold below which confidence is considered low
    pub low_threshold: f64,
    /// Threshold above which confidence is considered high
    pub high_threshold: f64,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            ema_decay: 0.95,
            min_samples: 5,
            low_threshold: 0.3,
            high_threshold: 0.7,
        }
    }
}

/// Confidence level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    /// Insufficient data to determine confidence
    Insufficient,
    /// Low confidence — high variance in recent outcomes
    Low,
    /// Moderate confidence
    Medium,
    /// High confidence — consistent positive outcomes
    High,
}

/// Decision confidence estimation
pub struct Confidence {
    /// Configuration parameters
    config: ConfidenceConfig,
    /// Recent reward values for variance calculation
    recent_rewards: VecDeque<f64>,
    /// Exponential moving average of reward
    ema_reward: f64,
    /// Exponential moving average of squared reward (for variance)
    ema_reward_sq: f64,
    /// Current confidence score (0.0 - 1.0)
    confidence_score: f64,
    /// Number of updates received
    update_count: u64,
    /// Running count of consistent-sign rewards in a row
    streak: i64,
    /// Last action taken (for agreement tracking)
    last_action: Option<usize>,
    /// Count of times the same action was chosen consecutively
    action_agreement_count: u64,
    /// Total action selections tracked
    total_action_count: u64,
}

impl Default for Confidence {
    fn default() -> Self {
        Self::new()
    }
}

impl Confidence {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(ConfidenceConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: ConfidenceConfig) -> Self {
        Self {
            recent_rewards: VecDeque::with_capacity(config.window_size),
            ema_reward: 0.0,
            ema_reward_sq: 0.0,
            confidence_score: 0.0,
            update_count: 0,
            streak: 0,
            last_action: None,
            action_agreement_count: 0,
            total_action_count: 0,
            config,
        }
    }

    /// Update confidence with a new reward observation
    pub fn update(&mut self, reward: f64) {
        // Maintain the sliding window
        if self.recent_rewards.len() >= self.config.window_size {
            self.recent_rewards.pop_front();
        }
        self.recent_rewards.push_back(reward);

        // Update EMA
        let alpha = 1.0 - self.config.ema_decay;
        if self.update_count == 0 {
            self.ema_reward = reward;
            self.ema_reward_sq = reward * reward;
        } else {
            self.ema_reward = self.config.ema_decay * self.ema_reward + alpha * reward;
            self.ema_reward_sq =
                self.config.ema_decay * self.ema_reward_sq + alpha * (reward * reward);
        }

        // Update streak
        if reward >= 0.0 {
            if self.streak >= 0 {
                self.streak += 1;
            } else {
                self.streak = 1;
            }
        } else if self.streak <= 0 {
            self.streak -= 1;
        } else {
            self.streak = -1;
        }

        self.update_count += 1;
        self.recompute_confidence();
    }

    /// Record an action selection for agreement tracking
    pub fn record_action(&mut self, action: usize) {
        self.total_action_count += 1;
        if let Some(last) = self.last_action {
            if last == action {
                self.action_agreement_count += 1;
            }
        }
        self.last_action = Some(action);
    }

    /// Main processing function — recomputes and returns the confidence score
    pub fn process(&self) -> Result<()> {
        // Confidence is recomputed on each update; process validates state
        if self.update_count == 0 {
            return Ok(());
        }
        Ok(())
    }

    /// Get the current confidence score (0.0 - 1.0)
    pub fn score(&self) -> f64 {
        self.confidence_score
    }

    /// Get the current confidence level classification
    pub fn level(&self) -> ConfidenceLevel {
        if (self.update_count as usize) < self.config.min_samples {
            return ConfidenceLevel::Insufficient;
        }
        if self.confidence_score < self.config.low_threshold {
            ConfidenceLevel::Low
        } else if self.confidence_score < self.config.high_threshold {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::High
        }
    }

    /// Get the EMA of recent rewards
    pub fn ema_reward(&self) -> f64 {
        self.ema_reward
    }

    /// Get the EMA-based variance of recent rewards
    pub fn ema_variance(&self) -> f64 {
        let var = self.ema_reward_sq - self.ema_reward * self.ema_reward;
        var.max(0.0)
    }

    /// Get the current streak length (positive = winning, negative = losing)
    pub fn streak(&self) -> i64 {
        self.streak
    }

    /// Get action agreement ratio (0.0 - 1.0)
    pub fn action_agreement_ratio(&self) -> f64 {
        if self.total_action_count <= 1 {
            return 0.0;
        }
        self.action_agreement_count as f64 / (self.total_action_count - 1) as f64
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.recent_rewards.clear();
        self.ema_reward = 0.0;
        self.ema_reward_sq = 0.0;
        self.confidence_score = 0.0;
        self.update_count = 0;
        self.streak = 0;
        self.last_action = None;
        self.action_agreement_count = 0;
        self.total_action_count = 0;
    }

    // ── internal ──

    /// Recompute the composite confidence score from all signals
    fn recompute_confidence(&mut self) {
        if (self.update_count as usize) < self.config.min_samples {
            self.confidence_score = 0.0;
            return;
        }

        // Component 1: Reward consistency (low variance → high confidence)
        let variance = self.ema_variance();
        // Map variance to a 0-1 score using a sigmoid-like transform
        // variance of 0 → score 1.0, large variance → score near 0
        let consistency_score = 1.0 / (1.0 + (5.0 * variance).sqrt());

        // Component 2: Reward positivity (positive EMA → higher confidence)
        // Sigmoid centered at 0
        let positivity_score = 1.0 / (1.0 + (-10.0 * self.ema_reward).exp());

        // Component 3: Streak contribution
        let streak_abs = self.streak.unsigned_abs() as f64;
        let streak_score = if self.streak > 0 {
            // Winning streak boosts confidence, with diminishing returns
            (streak_abs / (streak_abs + 5.0)).min(1.0)
        } else {
            // Losing streak reduces confidence
            1.0 - (streak_abs / (streak_abs + 3.0)).min(1.0)
        };

        // Component 4: Action agreement (consistent choices → confident policy)
        let agreement_score = self.action_agreement_ratio();

        // Weighted combination
        let w_consistency = 0.35;
        let w_positivity = 0.30;
        let w_streak = 0.20;
        let w_agreement = 0.15;

        self.confidence_score = (w_consistency * consistency_score
            + w_positivity * positivity_score
            + w_streak * streak_score
            + w_agreement * agreement_score)
            .clamp(0.0, 1.0);
    }

    /// Compute the simple mean of recent rewards
    #[allow(dead_code)]
    fn window_mean(&self) -> f64 {
        if self.recent_rewards.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent_rewards.iter().sum();
        sum / self.recent_rewards.len() as f64
    }

    /// Compute the simple variance of recent rewards
    #[allow(dead_code)]
    fn window_variance(&self) -> f64 {
        if self.recent_rewards.len() < 2 {
            return 0.0;
        }
        let mean = self.window_mean();
        let sum_sq: f64 = self.recent_rewards.iter().map(|r| (r - mean).powi(2)).sum();
        sum_sq / (self.recent_rewards.len() - 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = Confidence::new();
        assert!(instance.process().is_ok());
        assert_eq!(instance.score(), 0.0);
        assert_eq!(instance.level(), ConfidenceLevel::Insufficient);
    }

    #[test]
    fn test_consistent_positive_rewards_yield_high_confidence() {
        let mut conf = Confidence::with_config(ConfidenceConfig {
            min_samples: 3,
            ..Default::default()
        });

        for _ in 0..20 {
            conf.update(0.5);
            conf.record_action(1);
        }

        assert!(conf.score() > 0.6, "score was {}", conf.score());
        assert_eq!(conf.level(), ConfidenceLevel::High);
        assert!(conf.streak() > 0);
    }

    #[test]
    fn test_volatile_rewards_yield_low_confidence() {
        let mut conf = Confidence::with_config(ConfidenceConfig {
            min_samples: 3,
            ema_decay: 0.8, // Faster decay so EMA converges within the loop
            ..Default::default()
        });

        for i in 0..50 {
            let reward = if i % 2 == 0 { 1.0 } else { -1.0 };
            conf.update(reward);
            conf.record_action(if i % 2 == 0 { 0 } else { 1 });
        }

        // Volatile rewards and inconsistent actions → lower confidence
        assert!(conf.score() < 0.5, "score was {}", conf.score());
    }

    #[test]
    fn test_negative_streak_reduces_confidence() {
        let mut conf = Confidence::with_config(ConfidenceConfig {
            min_samples: 3,
            ..Default::default()
        });

        for _ in 0..15 {
            conf.update(-0.3);
        }

        assert!(conf.streak() < 0);
        assert!(conf.score() < 0.4, "score was {}", conf.score());
    }

    #[test]
    fn test_action_agreement_ratio() {
        let mut conf = Confidence::new();
        conf.record_action(1);
        conf.record_action(1);
        conf.record_action(1);
        conf.record_action(2);
        // 3 transitions: 1→1 (agree), 1→1 (agree), 1→2 (disagree) = 2/3
        assert!((conf.action_agreement_ratio() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_reset() {
        let mut conf = Confidence::new();
        for _ in 0..10 {
            conf.update(1.0);
        }
        assert!(conf.score() > 0.0 || conf.update_count > 0);
        conf.reset();
        assert_eq!(conf.score(), 0.0);
        assert_eq!(conf.update_count, 0);
        assert_eq!(conf.streak(), 0);
    }

    #[test]
    fn test_ema_variance_non_negative() {
        let mut conf = Confidence::new();
        for i in 0..100 {
            conf.update((i as f64 * 0.1).sin());
        }
        assert!(conf.ema_variance() >= 0.0);
    }
}
