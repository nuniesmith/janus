//! Gating threshold adaptation
//!
//! Part of the Thalamus region - manages adaptive thresholds for gating
//! signals based on market conditions, volatility, and system load.

use crate::common::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Threshold adaptation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationStrategy {
    /// Fixed threshold - no adaptation
    Fixed,
    /// Moving average based adaptation
    MovingAverage,
    /// Percentile-based adaptation
    Percentile,
    /// Volatility-scaled adaptation
    VolatilityScaled,
    /// Feedback-driven adaptation
    Feedback,
    /// Hybrid combining multiple strategies
    Hybrid,
}

impl Default for AdaptationStrategy {
    fn default() -> Self {
        Self::VolatilityScaled
    }
}

/// Threshold direction for asymmetric gating
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdDirection {
    /// Upper threshold only
    Upper,
    /// Lower threshold only
    Lower,
    /// Both upper and lower
    Both,
    /// Symmetric around mean
    Symmetric,
}

impl Default for ThresholdDirection {
    fn default() -> Self {
        Self::Both
    }
}

/// Threshold violation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    /// Value exceeded upper threshold
    AboveUpper,
    /// Value below lower threshold
    BelowLower,
    /// No violation
    None,
}

/// Threshold state
#[derive(Debug, Clone)]
pub struct ThresholdState {
    /// Current upper threshold
    pub upper: f64,
    /// Current lower threshold
    pub lower: f64,
    /// Base threshold value
    pub base: f64,
    /// Current adaptation factor
    pub adaptation_factor: f64,
    /// Whether threshold is currently relaxed
    pub relaxed: bool,
    /// Relaxation factor (1.0 = no relaxation)
    pub relaxation_factor: f64,
}

impl Default for ThresholdState {
    fn default() -> Self {
        Self {
            upper: 0.8,
            lower: 0.2,
            base: 0.5,
            adaptation_factor: 1.0,
            relaxed: false,
            relaxation_factor: 1.0,
        }
    }
}

impl ThresholdState {
    /// Create with specific bounds
    pub fn with_bounds(lower: f64, upper: f64) -> Self {
        Self {
            upper,
            lower,
            base: (upper + lower) / 2.0,
            ..Default::default()
        }
    }

    /// Get effective upper threshold
    pub fn effective_upper(&self) -> f64 {
        self.upper * self.adaptation_factor * self.relaxation_factor
    }

    /// Get effective lower threshold
    pub fn effective_lower(&self) -> f64 {
        self.lower / (self.adaptation_factor * self.relaxation_factor)
    }

    /// Get threshold band width
    pub fn band_width(&self) -> f64 {
        self.effective_upper() - self.effective_lower()
    }

    /// Check if value violates threshold
    pub fn check_violation(&self, value: f64) -> ViolationType {
        if value > self.effective_upper() {
            ViolationType::AboveUpper
        } else if value < self.effective_lower() {
            ViolationType::BelowLower
        } else {
            ViolationType::None
        }
    }

    /// Get distance from threshold (positive if within band)
    pub fn distance_from_threshold(&self, value: f64) -> f64 {
        let upper = self.effective_upper();
        let lower = self.effective_lower();

        if value > upper {
            -(value - upper)
        } else if value < lower {
            -(lower - value)
        } else {
            // Return minimum distance to either threshold
            (upper - value).min(value - lower)
        }
    }

    /// Normalize value to threshold band (0 = lower, 1 = upper)
    pub fn normalize(&self, value: f64) -> f64 {
        let width = self.band_width();
        if width == 0.0 {
            return 0.5;
        }
        ((value - self.effective_lower()) / width).clamp(0.0, 1.0)
    }
}

/// Threshold check result
#[derive(Debug, Clone)]
pub struct ThresholdCheck {
    /// Original value
    pub value: f64,
    /// Whether value passes threshold
    pub passes: bool,
    /// Violation type
    pub violation: ViolationType,
    /// Distance from nearest threshold
    pub distance: f64,
    /// Normalized position in band
    pub normalized: f64,
    /// Current threshold state
    pub threshold_state: ThresholdState,
    /// Confidence in threshold validity
    pub confidence: f64,
}

impl ThresholdCheck {
    /// Get margin to nearest threshold (positive if within band)
    pub fn margin(&self) -> f64 {
        self.distance
    }

    /// Check if value is near threshold (within margin)
    pub fn is_near_threshold(&self, margin: f64) -> bool {
        self.distance.abs() < margin
    }
}

/// Threshold adaptation configuration
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    /// Base threshold values
    pub base_lower: f64,
    pub base_upper: f64,
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
    /// Direction for thresholding
    pub direction: ThresholdDirection,
    /// Adaptation rate (0-1)
    pub adaptation_rate: f64,
    /// History window size
    pub window_size: usize,
    /// Minimum adaptation factor
    pub min_factor: f64,
    /// Maximum adaptation factor
    pub max_factor: f64,
    /// Volatility scale factor
    pub volatility_scale: f64,
    /// Percentile for percentile-based adaptation
    pub percentile: f64,
    /// Enable hysteresis to prevent oscillation
    pub hysteresis: bool,
    /// Hysteresis band width
    pub hysteresis_band: f64,
    /// Relaxation decay rate
    pub relaxation_decay: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            base_lower: 0.2,
            base_upper: 0.8,
            strategy: AdaptationStrategy::VolatilityScaled,
            direction: ThresholdDirection::Both,
            adaptation_rate: 0.1,
            window_size: 100,
            min_factor: 0.5,
            max_factor: 2.0,
            volatility_scale: 1.0,
            percentile: 0.95,
            hysteresis: true,
            hysteresis_band: 0.05,
            relaxation_decay: 0.99,
        }
    }
}

/// Threshold statistics
#[derive(Debug, Clone, Default)]
pub struct ThresholdStats {
    /// Total checks performed
    pub total_checks: u64,
    /// Checks that passed
    pub passed_checks: u64,
    /// Checks that failed (violated threshold)
    pub failed_checks: u64,
    /// Upper violations
    pub upper_violations: u64,
    /// Lower violations
    pub lower_violations: u64,
    /// Current pass rate
    pub pass_rate: f64,
    /// Average value seen
    pub avg_value: f64,
    /// Value standard deviation
    pub std_dev: f64,
    /// Threshold adaptations performed
    pub adaptations: u64,
    /// Current threshold upper
    pub current_upper: f64,
    /// Current threshold lower
    pub current_lower: f64,
}

/// Feedback signal for threshold adjustment
#[derive(Debug, Clone)]
pub struct ThresholdFeedback {
    /// Signal identifier
    pub signal_id: String,
    /// Whether the gating decision was correct
    pub correct: bool,
    /// Actual outcome value (if known)
    pub outcome: Option<f64>,
    /// Suggested adjustment (-1 to +1)
    pub adjustment: f64,
    /// Confidence in feedback
    pub confidence: f64,
}

impl ThresholdFeedback {
    /// Create positive feedback (decision was correct)
    pub fn positive(signal_id: impl Into<String>) -> Self {
        Self {
            signal_id: signal_id.into(),
            correct: true,
            outcome: None,
            adjustment: 0.0,
            confidence: 1.0,
        }
    }

    /// Create negative feedback (decision was wrong)
    pub fn negative(signal_id: impl Into<String>, adjustment: f64) -> Self {
        Self {
            signal_id: signal_id.into(),
            correct: false,
            outcome: None,
            adjustment: adjustment.clamp(-1.0, 1.0),
            confidence: 1.0,
        }
    }

    /// Set outcome value
    pub fn with_outcome(mut self, outcome: f64) -> Self {
        self.outcome = Some(outcome);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Gating threshold adaptation system
pub struct Threshold {
    /// Configuration
    config: ThresholdConfig,
    /// Current state
    state: Arc<RwLock<ThresholdState>>,
    /// Value history for adaptation
    history: Arc<RwLock<VecDeque<f64>>>,
    /// Statistics
    stats: Arc<RwLock<ThresholdStats>>,
    /// Running mean for Welford's algorithm
    running_mean: Arc<RwLock<f64>>,
    /// Running variance
    running_variance: Arc<RwLock<f64>>,
    /// Sample count
    sample_count: Arc<RwLock<u64>>,
    /// Feedback history
    feedback_history: Arc<RwLock<VecDeque<ThresholdFeedback>>>,
    /// Last violation state for hysteresis
    last_violation: Arc<RwLock<ViolationType>>,
}

impl Default for Threshold {
    fn default() -> Self {
        Self::new()
    }
}

impl Threshold {
    /// Create a new threshold manager
    pub fn new() -> Self {
        Self::with_config(ThresholdConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ThresholdConfig) -> Self {
        let initial_state = ThresholdState::with_bounds(config.base_lower, config.base_upper);

        Self {
            state: Arc::new(RwLock::new(initial_state)),
            history: Arc::new(RwLock::new(VecDeque::with_capacity(config.window_size))),
            stats: Arc::new(RwLock::new(ThresholdStats::default())),
            running_mean: Arc::new(RwLock::new(0.0)),
            running_variance: Arc::new(RwLock::new(0.0)),
            sample_count: Arc::new(RwLock::new(0)),
            feedback_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            last_violation: Arc::new(RwLock::new(ViolationType::None)),
            config,
        }
    }

    /// Create with bounds
    pub fn with_bounds(lower: f64, upper: f64) -> Self {
        let mut config = ThresholdConfig::default();
        config.base_lower = lower;
        config.base_upper = upper;
        Self::with_config(config)
    }

    /// Check a value against the threshold
    pub async fn check(&self, value: f64) -> ThresholdCheck {
        // Update statistics
        self.update_statistics(value).await;

        // Get current state
        let state = self.state.read().await.clone();

        // Check violation with hysteresis
        let violation = self.check_with_hysteresis(value, &state).await;

        // Update last violation
        *self.last_violation.write().await = violation;

        let passes = matches!(violation, ViolationType::None);
        let distance = state.distance_from_threshold(value);
        let normalized = state.normalize(value);

        // Calculate confidence based on history
        let confidence = self.calculate_confidence().await;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_checks += 1;
            if passes {
                stats.passed_checks += 1;
            } else {
                stats.failed_checks += 1;
                match violation {
                    ViolationType::AboveUpper => stats.upper_violations += 1,
                    ViolationType::BelowLower => stats.lower_violations += 1,
                    _ => {}
                }
            }
            stats.pass_rate = stats.passed_checks as f64 / stats.total_checks.max(1) as f64;
            stats.current_upper = state.effective_upper();
            stats.current_lower = state.effective_lower();
        }

        // Adapt threshold based on strategy
        self.adapt().await;

        ThresholdCheck {
            value,
            passes,
            violation,
            distance,
            normalized,
            threshold_state: state,
            confidence,
        }
    }

    /// Check with hysteresis to prevent oscillation
    async fn check_with_hysteresis(&self, value: f64, state: &ThresholdState) -> ViolationType {
        if !self.config.hysteresis {
            return state.check_violation(value);
        }

        let last = *self.last_violation.read().await;
        let current = state.check_violation(value);

        // Apply hysteresis band
        match (last, current) {
            // If was above upper, only return to normal if clearly below
            (ViolationType::AboveUpper, ViolationType::None) => {
                if value < state.effective_upper() - self.config.hysteresis_band {
                    ViolationType::None
                } else {
                    ViolationType::AboveUpper
                }
            }
            // If was below lower, only return to normal if clearly above
            (ViolationType::BelowLower, ViolationType::None) => {
                if value > state.effective_lower() + self.config.hysteresis_band {
                    ViolationType::None
                } else {
                    ViolationType::BelowLower
                }
            }
            _ => current,
        }
    }

    /// Update running statistics
    async fn update_statistics(&self, value: f64) {
        // Add to history
        {
            let mut history = self.history.write().await;
            history.push_back(value);
            if history.len() > self.config.window_size {
                history.pop_front();
            }
        }

        // Update Welford's online algorithm
        {
            let mut count = self.sample_count.write().await;
            let mut mean = self.running_mean.write().await;
            let mut variance = self.running_variance.write().await;

            *count += 1;
            let delta = value - *mean;
            *mean += delta / *count as f64;
            let delta2 = value - *mean;
            *variance += delta * delta2;
        }

        // Update stats
        {
            let count = *self.sample_count.read().await;
            let mean = *self.running_mean.read().await;
            let variance = *self.running_variance.read().await;

            let mut stats = self.stats.write().await;
            stats.avg_value = mean;
            if count > 1 {
                stats.std_dev = (variance / (count - 1) as f64).sqrt();
            }
        }
    }

    /// Adapt threshold based on configured strategy
    async fn adapt(&self) {
        let new_factor = match self.config.strategy {
            AdaptationStrategy::Fixed => return,
            AdaptationStrategy::MovingAverage => self.adapt_moving_average().await,
            AdaptationStrategy::Percentile => self.adapt_percentile().await,
            AdaptationStrategy::VolatilityScaled => self.adapt_volatility().await,
            AdaptationStrategy::Feedback => self.adapt_feedback().await,
            AdaptationStrategy::Hybrid => self.adapt_hybrid().await,
        };

        // Apply adaptation with rate limiting
        let clamped_factor = new_factor.clamp(self.config.min_factor, self.config.max_factor);

        {
            let mut state = self.state.write().await;
            let current = state.adaptation_factor;
            state.adaptation_factor = current * (1.0 - self.config.adaptation_rate)
                + clamped_factor * self.config.adaptation_rate;

            // Decay relaxation
            if state.relaxed {
                state.relaxation_factor *= self.config.relaxation_decay;
                if state.relaxation_factor < 1.01 {
                    state.relaxed = false;
                    state.relaxation_factor = 1.0;
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.adaptations += 1;
        }
    }

    /// Adapt using moving average
    async fn adapt_moving_average(&self) -> f64 {
        let history = self.history.read().await;
        if history.is_empty() {
            return 1.0;
        }

        let avg: f64 = history.iter().sum::<f64>() / history.len() as f64;
        let base = (self.config.base_upper + self.config.base_lower) / 2.0;

        if base == 0.0 {
            return 1.0;
        }

        (avg / base).clamp(self.config.min_factor, self.config.max_factor)
    }

    /// Adapt using percentile
    async fn adapt_percentile(&self) -> f64 {
        let history = self.history.read().await;
        if history.is_empty() {
            return 1.0;
        }

        let mut sorted: Vec<f64> = history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() as f64) * self.config.percentile) as usize;
        let percentile_value = sorted
            .get(idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(0.5);

        let base_range = self.config.base_upper - self.config.base_lower;
        if base_range == 0.0 {
            return 1.0;
        }

        (percentile_value / self.config.base_upper)
            .clamp(self.config.min_factor, self.config.max_factor)
    }

    /// Adapt based on volatility
    async fn adapt_volatility(&self) -> f64 {
        let std_dev = {
            let stats = self.stats.read().await;
            stats.std_dev
        };

        // Scale threshold based on volatility
        // Higher volatility -> wider thresholds
        let volatility_factor = 1.0 + (std_dev * self.config.volatility_scale);
        volatility_factor.clamp(self.config.min_factor, self.config.max_factor)
    }

    /// Adapt based on feedback
    async fn adapt_feedback(&self) -> f64 {
        let feedback_history = self.feedback_history.read().await;
        if feedback_history.is_empty() {
            return 1.0;
        }

        // Calculate weighted adjustment from recent feedback
        let mut total_adjustment = 0.0;
        let mut total_weight = 0.0;

        for (i, feedback) in feedback_history.iter().enumerate() {
            // More recent feedback has higher weight
            let recency_weight = 1.0 / (1.0 + i as f64);
            let weight = feedback.confidence * recency_weight;

            total_adjustment += feedback.adjustment * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return 1.0;
        }

        let avg_adjustment = total_adjustment / total_weight;

        // Convert adjustment to factor
        1.0 + avg_adjustment * 0.5
    }

    /// Hybrid adaptation combining multiple strategies
    async fn adapt_hybrid(&self) -> f64 {
        let volatility = self.adapt_volatility().await;
        let percentile = self.adapt_percentile().await;
        let feedback = self.adapt_feedback().await;

        // Weighted combination
        let factor = volatility * 0.4 + percentile * 0.3 + feedback * 0.3;
        factor.clamp(self.config.min_factor, self.config.max_factor)
    }

    /// Calculate confidence in current threshold
    async fn calculate_confidence(&self) -> f64 {
        let count = *self.sample_count.read().await;

        // More samples = higher confidence
        let sample_confidence = (count as f64 / self.config.window_size as f64).min(1.0);

        // Stable statistics = higher confidence
        let stats = self.stats.read().await;
        let stability_confidence = if stats.std_dev > 0.0 {
            (1.0 / (1.0 + stats.std_dev)).min(1.0)
        } else {
            0.5
        };

        (sample_confidence + stability_confidence) / 2.0
    }

    /// Provide feedback for threshold adaptation
    pub async fn provide_feedback(&self, feedback: ThresholdFeedback) {
        let mut history = self.feedback_history.write().await;
        history.push_front(feedback);
        if history.len() > 100 {
            history.pop_back();
        }
    }

    /// Temporarily relax threshold
    pub async fn relax(&self, factor: f64) {
        let mut state = self.state.write().await;
        state.relaxed = true;
        state.relaxation_factor = factor.max(1.0);
    }

    /// Tighten threshold
    pub async fn tighten(&self, factor: f64) {
        let mut state = self.state.write().await;
        state.relaxed = false;
        state.relaxation_factor = 1.0;
        state.adaptation_factor *= factor.clamp(0.5, 1.0);
    }

    /// Set absolute threshold bounds
    pub async fn set_bounds(&self, lower: f64, upper: f64) {
        let mut state = self.state.write().await;
        state.lower = lower;
        state.upper = upper;
        state.base = (upper + lower) / 2.0;
    }

    /// Get current threshold state
    pub async fn state(&self) -> ThresholdState {
        self.state.read().await.clone()
    }

    /// Get statistics
    pub async fn stats(&self) -> ThresholdStats {
        self.stats.read().await.clone()
    }

    /// Reset to initial state
    pub async fn reset(&self) {
        {
            let mut state = self.state.write().await;
            *state = ThresholdState::with_bounds(self.config.base_lower, self.config.base_upper);
        }
        {
            let mut history = self.history.write().await;
            history.clear();
        }
        {
            let mut stats = self.stats.write().await;
            *stats = ThresholdStats::default();
        }
        {
            *self.running_mean.write().await = 0.0;
            *self.running_variance.write().await = 0.0;
            *self.sample_count.write().await = 0;
        }
        {
            let mut feedback = self.feedback_history.write().await;
            feedback.clear();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ThresholdConfig {
        &self.config
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_state_creation() {
        let state = ThresholdState::with_bounds(0.2, 0.8);
        assert!((state.lower - 0.2).abs() < 0.001);
        assert!((state.upper - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_threshold_violation() {
        let state = ThresholdState::with_bounds(0.2, 0.8);

        assert_eq!(state.check_violation(0.5), ViolationType::None);
        assert_eq!(state.check_violation(0.9), ViolationType::AboveUpper);
        assert_eq!(state.check_violation(0.1), ViolationType::BelowLower);
    }

    #[test]
    fn test_threshold_normalize() {
        let state = ThresholdState::with_bounds(0.2, 0.8);

        let normalized = state.normalize(0.5);
        assert!((normalized - 0.5).abs() < 0.001);

        let normalized_low = state.normalize(0.2);
        assert!((normalized_low - 0.0).abs() < 0.001);

        let normalized_high = state.normalize(0.8);
        assert!((normalized_high - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_threshold_distance() {
        let state = ThresholdState::with_bounds(0.2, 0.8);

        // Value in middle should have positive distance to both thresholds
        let dist_middle = state.distance_from_threshold(0.5);
        assert!(dist_middle > 0.0);

        // Value above upper should have negative distance
        let dist_above = state.distance_from_threshold(0.9);
        assert!(dist_above < 0.0);
    }

    #[test]
    fn test_band_width() {
        let state = ThresholdState::with_bounds(0.2, 0.8);
        let width = state.band_width();
        assert!((width - 0.6).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_check_value() {
        let threshold = Threshold::with_bounds(0.2, 0.8);

        let result = threshold.check(0.5).await;
        assert!(result.passes);
        assert_eq!(result.violation, ViolationType::None);

        let result = threshold.check(0.9).await;
        assert!(!result.passes);
        assert_eq!(result.violation, ViolationType::AboveUpper);

        let result = threshold.check(0.1).await;
        assert!(!result.passes);
        assert_eq!(result.violation, ViolationType::BelowLower);
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let threshold = Threshold::new();

        for i in 0..100 {
            threshold.check(0.5 + (i as f64 % 10.0) / 100.0).await;
        }

        let stats = threshold.stats().await;
        assert_eq!(stats.total_checks, 100);
        assert!(stats.avg_value > 0.0);
    }

    #[tokio::test]
    async fn test_volatility_adaptation() {
        let mut config = ThresholdConfig::default();
        config.strategy = AdaptationStrategy::VolatilityScaled;
        config.adaptation_rate = 0.5; // Fast adaptation for testing
        let threshold = Threshold::with_config(config);

        // Add volatile values
        for i in 0..50 {
            let value = if i % 2 == 0 { 0.3 } else { 0.7 };
            threshold.check(value).await;
        }

        let state = threshold.state().await;
        // Adaptation factor should increase due to volatility
        assert!(state.adaptation_factor > 1.0);
    }

    #[tokio::test]
    async fn test_relax_threshold() {
        let threshold = Threshold::with_bounds(0.2, 0.8);

        threshold.relax(1.5).await;

        let state = threshold.state().await;
        assert!(state.relaxed);
        assert!(state.effective_upper() > 0.8);
    }

    #[tokio::test]
    async fn test_tighten_threshold() {
        let threshold = Threshold::with_bounds(0.2, 0.8);

        // First check some values to have an adaptation factor
        for _ in 0..10 {
            threshold.check(0.5).await;
        }

        threshold.tighten(0.8).await;

        let state = threshold.state().await;
        assert!(!state.relaxed);
    }

    #[tokio::test]
    async fn test_feedback_adaptation() {
        let mut config = ThresholdConfig::default();
        config.strategy = AdaptationStrategy::Feedback;
        config.adaptation_rate = 0.5;
        let threshold = Threshold::with_config(config);

        // Provide feedback suggesting threshold is too tight
        for _ in 0..10 {
            threshold
                .provide_feedback(ThresholdFeedback::negative("sig", 0.5))
                .await;
        }

        // Check to trigger adaptation
        threshold.check(0.5).await;

        let state = threshold.state().await;
        // Should adapt based on feedback
        assert!(state.adaptation_factor != 1.0);
    }

    #[tokio::test]
    async fn test_reset() {
        let threshold = Threshold::new();

        for _ in 0..50 {
            threshold.check(0.5).await;
        }

        let stats_before = threshold.stats().await;
        assert!(stats_before.total_checks > 0);

        threshold.reset().await;

        let stats_after = threshold.stats().await;
        assert_eq!(stats_after.total_checks, 0);
    }

    #[tokio::test]
    async fn test_set_bounds() {
        let threshold = Threshold::new();

        threshold.set_bounds(0.3, 0.7).await;

        let state = threshold.state().await;
        assert!((state.lower - 0.3).abs() < 0.001);
        assert!((state.upper - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_threshold_check_helpers() {
        let check = ThresholdCheck {
            value: 0.5,
            passes: true,
            violation: ViolationType::None,
            distance: 0.1,
            normalized: 0.5,
            threshold_state: ThresholdState::default(),
            confidence: 0.9,
        };

        assert!((check.margin() - 0.1).abs() < 0.001);
        assert!(!check.is_near_threshold(0.05));
        assert!(check.is_near_threshold(0.15));
    }

    #[test]
    fn test_feedback_creation() {
        let positive = ThresholdFeedback::positive("sig1");
        assert!(positive.correct);

        let negative = ThresholdFeedback::negative("sig2", -0.3);
        assert!(!negative.correct);
        assert!((negative.adjustment - (-0.3)).abs() < 0.001);
    }

    #[test]
    fn test_process_compatibility() {
        let threshold = Threshold::new();
        assert!(threshold.process().is_ok());
    }
}
