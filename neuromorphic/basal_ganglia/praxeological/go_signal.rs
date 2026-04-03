//! Action initiation (direct pathway)
//!
//! Part of the Basal Ganglia region
//! Component: praxeological
//!
//! The Go Signal implements the direct pathway of the basal ganglia,
//! which facilitates action initiation. Key features:
//! - Action facilitation based on expected value
//! - Dopamine-modulated signal strength
//! - Competition with NoGo signals
//! - Threshold-based action release
//! - Learning from positive outcomes

use crate::common::Result;
use std::collections::HashMap;

/// Configuration for Go signal generation
#[derive(Debug, Clone)]
pub struct GoSignalConfig {
    /// Base threshold for action initiation
    pub base_threshold: f64,
    /// Dopamine sensitivity (how much DA affects signal)
    pub dopamine_sensitivity: f64,
    /// Learning rate for value updates
    pub learning_rate: f64,
    /// Decay rate for signal strength
    pub decay_rate: f64,
    /// Minimum signal strength
    pub min_signal: f64,
    /// Maximum signal strength
    pub max_signal: f64,
    /// Temperature for softmax competition
    pub temperature: f64,
}

impl Default for GoSignalConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.5,
            dopamine_sensitivity: 1.0,
            learning_rate: 0.1,
            decay_rate: 0.95,
            min_signal: 0.0,
            max_signal: 1.0,
            temperature: 1.0,
        }
    }
}

/// State of a Go signal for a particular action
#[derive(Debug, Clone)]
pub struct GoState {
    /// Current signal strength (0.0 - 1.0)
    pub strength: f64,
    /// Learned value association
    pub value: f64,
    /// Number of times this action was initiated
    pub initiation_count: u64,
    /// Running average of outcomes
    pub avg_outcome: f64,
    /// Last update timestamp
    pub last_update: i64,
}

impl Default for GoState {
    fn default() -> Self {
        Self {
            strength: 0.5,
            value: 0.0,
            initiation_count: 0,
            avg_outcome: 0.0,
            last_update: 0,
        }
    }
}

/// Result of Go signal evaluation
#[derive(Debug, Clone)]
pub struct GoEvaluation {
    /// Action identifier
    pub action_id: String,
    /// Go signal strength (higher = more facilitation)
    pub go_strength: f64,
    /// Whether threshold was exceeded
    pub above_threshold: bool,
    /// Effective threshold used
    pub threshold: f64,
    /// Dopamine modulation applied
    pub dopamine_effect: f64,
    /// Confidence in the signal
    pub confidence: f64,
}

/// Action context for Go signal computation
#[derive(Debug, Clone)]
pub struct ActionContext {
    /// Action identifier
    pub action_id: String,
    /// Expected reward/value
    pub expected_value: f64,
    /// Current dopamine level (0.0 - 2.0, 1.0 = baseline)
    pub dopamine_level: f64,
    /// Urgency factor (0.0 - 1.0)
    pub urgency: f64,
    /// External facilitation signal
    pub facilitation: f64,
    /// Timestamp
    pub timestamp: i64,
}

/// Statistics about Go signal processing
#[derive(Debug, Clone, Default)]
pub struct GoStats {
    /// Total evaluations performed
    pub total_evaluations: u64,
    /// Actions initiated (above threshold)
    pub actions_initiated: u64,
    /// Actions suppressed (below threshold)
    pub actions_suppressed: u64,
    /// Average Go signal strength
    pub avg_go_strength: f64,
    /// Average dopamine effect
    pub avg_dopamine_effect: f64,
}

/// Action initiation (direct pathway) - Go Signal
///
/// Implements the direct pathway of the basal ganglia, which
/// facilitates action initiation by disinhibiting the thalamus.
/// Works in competition with NoGo signals (indirect pathway).
pub struct GoSignal {
    /// Configuration
    config: GoSignalConfig,
    /// Per-action Go states
    action_states: HashMap<String, GoState>,
    /// Current dopamine level
    dopamine_level: f64,
    /// Global facilitation bias
    facilitation_bias: f64,
    /// Statistics
    stats: GoStats,
    /// Running sum for averaging
    strength_sum: f64,
    /// Dopamine effect sum
    dopamine_sum: f64,
}

impl Default for GoSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl GoSignal {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(GoSignalConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GoSignalConfig) -> Self {
        Self {
            config,
            action_states: HashMap::new(),
            dopamine_level: 1.0, // Baseline
            facilitation_bias: 0.0,
            stats: GoStats::default(),
            strength_sum: 0.0,
            dopamine_sum: 0.0,
        }
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Evaluate Go signal for a single action
    pub fn evaluate(&mut self, context: &ActionContext) -> GoEvaluation {
        self.stats.total_evaluations += 1;

        // Calculate base Go signal from expected value
        // (computed before mutable borrow of action_states)
        let value_signal = self.sigmoid(context.expected_value);

        // Apply dopamine modulation
        // Higher dopamine = stronger Go signals (reward prediction)
        let dopamine_effect = (context.dopamine_level - 1.0) * self.config.dopamine_sensitivity;
        let da_modulated = value_signal * (1.0 + dopamine_effect);

        // Apply urgency boost
        let urgency_boost = context.urgency * 0.2;

        // Apply external facilitation
        let external_boost = context.facilitation * 0.3 + self.facilitation_bias;

        // Combine all factors
        let raw_strength = da_modulated + urgency_boost + external_boost;

        // Get or create action state (mutable borrow of action_states)
        let state = self
            .action_states
            .entry(context.action_id.clone())
            .or_default();

        // Apply learned value bias
        let learned_bias = state.value * 0.2;
        let combined = raw_strength + learned_bias;

        // Clamp to valid range
        let go_strength = combined
            .max(self.config.min_signal)
            .min(self.config.max_signal);

        // Calculate effective threshold
        let threshold = self.config.base_threshold;
        let above_threshold = go_strength > threshold;

        // Update state
        state.strength = go_strength;
        state.last_update = context.timestamp;

        if above_threshold {
            state.initiation_count += 1;
            self.stats.actions_initiated += 1;
        } else {
            self.stats.actions_suppressed += 1;
        }

        // Calculate confidence based on signal clarity
        let confidence = (go_strength - threshold).abs() / (1.0 - threshold).max(0.01);

        // Update running statistics
        self.strength_sum += go_strength;
        self.dopamine_sum += dopamine_effect.abs();
        let n = self.stats.total_evaluations as f64;
        self.stats.avg_go_strength = self.strength_sum / n;
        self.stats.avg_dopamine_effect = self.dopamine_sum / n;

        GoEvaluation {
            action_id: context.action_id.clone(),
            go_strength,
            above_threshold,
            threshold,
            dopamine_effect,
            confidence: confidence.min(1.0),
        }
    }

    /// Evaluate multiple actions and return sorted by Go strength
    pub fn evaluate_batch(&mut self, contexts: &[ActionContext]) -> Vec<GoEvaluation> {
        let mut evaluations: Vec<_> = contexts.iter().map(|ctx| self.evaluate(ctx)).collect();

        // Sort by Go strength (descending)
        evaluations.sort_by(|a, b| {
            b.go_strength
                .partial_cmp(&a.go_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        evaluations
    }

    /// Select best action above threshold
    pub fn select_action(&mut self, contexts: &[ActionContext]) -> Option<GoEvaluation> {
        let evaluations = self.evaluate_batch(contexts);
        evaluations.into_iter().find(|e| e.above_threshold)
    }

    /// Update Go signal based on outcome (learning)
    pub fn update_from_outcome(&mut self, action_id: &str, outcome: f64) {
        if let Some(state) = self.action_states.get_mut(action_id) {
            // TD-like update for value
            let prediction_error = outcome - state.value;
            state.value += self.config.learning_rate * prediction_error;

            // Update running average outcome
            let n = state.initiation_count.max(1) as f64;
            state.avg_outcome = state.avg_outcome * (n - 1.0) / n + outcome / n;
        }
    }

    /// Set global dopamine level
    pub fn set_dopamine(&mut self, level: f64) {
        self.dopamine_level = level.clamp(0.0, 2.0);
    }

    /// Get current dopamine level
    pub fn dopamine_level(&self) -> f64 {
        self.dopamine_level
    }

    /// Set global facilitation bias
    pub fn set_facilitation_bias(&mut self, bias: f64) {
        self.facilitation_bias = bias.clamp(-0.5, 0.5);
    }

    /// Get state for a specific action
    pub fn action_state(&self, action_id: &str) -> Option<&GoState> {
        self.action_states.get(action_id)
    }

    /// Get all action states
    pub fn all_states(&self) -> &HashMap<String, GoState> {
        &self.action_states
    }

    /// Get statistics
    pub fn stats(&self) -> &GoStats {
        &self.stats
    }

    /// Get initiation rate
    pub fn initiation_rate(&self) -> f64 {
        if self.stats.total_evaluations == 0 {
            0.0
        } else {
            self.stats.actions_initiated as f64 / self.stats.total_evaluations as f64
        }
    }

    /// Apply decay to all action strengths
    pub fn decay(&mut self) {
        for state in self.action_states.values_mut() {
            state.strength *= self.config.decay_rate;
        }
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.action_states.clear();
        self.dopamine_level = 1.0;
        self.facilitation_bias = 0.0;
        self.stats = GoStats::default();
        self.strength_sum = 0.0;
        self.dopamine_sum = 0.0;
    }

    /// Reset statistics only
    pub fn reset_stats(&mut self) {
        self.stats = GoStats::default();
        self.strength_sum = 0.0;
        self.dopamine_sum = 0.0;
    }

    // --- Private methods ---

    /// Sigmoid activation function
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x / self.config.temperature).exp())
    }
}

/// Compute competitive Go signals using softmax
pub fn competitive_go_signals(evaluations: &[GoEvaluation], temperature: f64) -> Vec<f64> {
    if evaluations.is_empty() {
        return vec![];
    }

    let strengths: Vec<f64> = evaluations.iter().map(|e| e.go_strength).collect();

    // Softmax normalization
    let max_strength = strengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = strengths
        .iter()
        .map(|&s| ((s - max_strength) / temperature).exp())
        .sum();

    strengths
        .iter()
        .map(|&s| ((s - max_strength) / temperature).exp() / exp_sum)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_context(action_id: &str, expected_value: f64) -> ActionContext {
        ActionContext {
            action_id: action_id.to_string(),
            expected_value,
            dopamine_level: 1.0,
            urgency: 0.0,
            facilitation: 0.0,
            timestamp: 0,
        }
    }

    #[test]
    fn test_basic() {
        let instance = GoSignal::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_evaluate_single_action() {
        let mut go_signal = GoSignal::new();
        let context = create_context("action_1", 1.0);

        let eval = go_signal.evaluate(&context);

        assert_eq!(eval.action_id, "action_1");
        assert!(eval.go_strength > 0.0);
        assert!(eval.go_strength <= 1.0);
    }

    #[test]
    fn test_higher_value_higher_signal() {
        let mut go_signal = GoSignal::new();

        let low_value = create_context("low", 0.1);
        let high_value = create_context("high", 2.0);

        let eval_low = go_signal.evaluate(&low_value);
        let eval_high = go_signal.evaluate(&high_value);

        assert!(
            eval_high.go_strength > eval_low.go_strength,
            "Higher expected value should produce stronger Go signal"
        );
    }

    #[test]
    fn test_dopamine_modulation() {
        let mut go_signal = GoSignal::new();

        // Low dopamine context
        let mut low_da = create_context("action", 1.0);
        low_da.dopamine_level = 0.5;
        let eval_low = go_signal.evaluate(&low_da);

        // High dopamine context
        let mut high_da = create_context("action2", 1.0);
        high_da.dopamine_level = 1.5;
        let eval_high = go_signal.evaluate(&high_da);

        assert!(
            eval_high.go_strength > eval_low.go_strength,
            "Higher dopamine should increase Go signal"
        );
        assert!(eval_high.dopamine_effect > 0.0);
        assert!(eval_low.dopamine_effect < 0.0);
    }

    #[test]
    fn test_threshold_detection() {
        let config = GoSignalConfig {
            base_threshold: 0.5,
            ..Default::default()
        };
        let mut go_signal = GoSignal::with_config(config);

        // High value should be above threshold
        let high_context = create_context("high", 2.0);
        let eval_high = go_signal.evaluate(&high_context);
        assert!(eval_high.above_threshold);

        // Very low value might be below threshold
        let low_context = create_context("low", -2.0);
        let eval_low = go_signal.evaluate(&low_context);
        // Signal will still be positive due to sigmoid, but might be below threshold
        assert!(eval_low.go_strength < eval_high.go_strength);
    }

    #[test]
    fn test_learning_from_outcome() {
        let mut go_signal = GoSignal::new();

        let context = create_context("learnable", 0.5);
        go_signal.evaluate(&context);

        // Positive outcome should increase value
        go_signal.update_from_outcome("learnable", 1.0);
        let state = go_signal.action_state("learnable").unwrap();
        assert!(state.value > 0.0);

        // Negative outcome should decrease value
        go_signal.update_from_outcome("learnable", -1.0);
        let _state = go_signal.action_state("learnable").unwrap();
        // Value should have decreased (may still be positive)
    }

    #[test]
    fn test_urgency_boost() {
        let mut go_signal = GoSignal::new();

        let mut normal = create_context("normal", 0.5);
        normal.urgency = 0.0;

        let mut urgent = create_context("urgent", 0.5);
        urgent.urgency = 1.0;

        let eval_normal = go_signal.evaluate(&normal);
        let eval_urgent = go_signal.evaluate(&urgent);

        assert!(
            eval_urgent.go_strength > eval_normal.go_strength,
            "Urgency should boost Go signal"
        );
    }

    #[test]
    fn test_batch_evaluation() {
        let mut go_signal = GoSignal::new();

        let contexts = vec![
            create_context("low", 0.1),
            create_context("high", 2.0),
            create_context("medium", 1.0),
        ];

        let evals = go_signal.evaluate_batch(&contexts);

        // Should be sorted by strength descending
        assert_eq!(evals[0].action_id, "high");
        assert!(evals[0].go_strength >= evals[1].go_strength);
        assert!(evals[1].go_strength >= evals[2].go_strength);
    }

    #[test]
    fn test_select_action() {
        let config = GoSignalConfig {
            base_threshold: 0.5,
            ..Default::default()
        };
        let mut go_signal = GoSignal::with_config(config);

        let contexts = vec![create_context("high", 2.0), create_context("low", 0.1)];

        let selected = go_signal.select_action(&contexts);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().action_id, "high");
    }

    #[test]
    fn test_statistics() {
        let mut go_signal = GoSignal::new();

        for i in 0..10 {
            let context = create_context(&format!("action_{}", i), i as f64 * 0.2);
            go_signal.evaluate(&context);
        }

        let stats = go_signal.stats();
        assert_eq!(stats.total_evaluations, 10);
        assert!(stats.actions_initiated + stats.actions_suppressed == 10);
    }

    #[test]
    fn test_decay() {
        let mut go_signal = GoSignal::new();

        let context = create_context("action", 1.0);
        go_signal.evaluate(&context);

        let initial_strength = go_signal.action_state("action").unwrap().strength;

        go_signal.decay();

        let decayed_strength = go_signal.action_state("action").unwrap().strength;
        assert!(decayed_strength < initial_strength);
    }

    #[test]
    fn test_reset() {
        let mut go_signal = GoSignal::new();

        let context = create_context("action", 1.0);
        go_signal.evaluate(&context);
        go_signal.set_dopamine(1.5);

        go_signal.reset();

        assert!(go_signal.action_state("action").is_none());
        assert_eq!(go_signal.dopamine_level(), 1.0);
        assert_eq!(go_signal.stats().total_evaluations, 0);
    }

    #[test]
    fn test_competitive_go_signals() {
        let evaluations = vec![
            GoEvaluation {
                action_id: "a".to_string(),
                go_strength: 0.8,
                above_threshold: true,
                threshold: 0.5,
                dopamine_effect: 0.0,
                confidence: 0.5,
            },
            GoEvaluation {
                action_id: "b".to_string(),
                go_strength: 0.6,
                above_threshold: true,
                threshold: 0.5,
                dopamine_effect: 0.0,
                confidence: 0.5,
            },
        ];

        let probs = competitive_go_signals(&evaluations, 1.0);

        assert_eq!(probs.len(), 2);
        assert!(
            probs[0] > probs[1],
            "Higher strength should have higher probability"
        );

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Probabilities should sum to 1");
    }

    #[test]
    fn test_initiation_rate() {
        let config = GoSignalConfig {
            base_threshold: 0.6,
            ..Default::default()
        };
        let mut go_signal = GoSignal::with_config(config);

        // Some above threshold, some below
        for i in 0..10 {
            let value = if i < 5 { 2.0 } else { -1.0 };
            let context = create_context(&format!("action_{}", i), value);
            go_signal.evaluate(&context);
        }

        let rate = go_signal.initiation_rate();
        assert!(rate > 0.0 && rate < 1.0);
    }

    #[test]
    fn test_facilitation_bias() {
        let mut go_signal = GoSignal::new();

        let context = create_context("action", 0.5);
        let eval_no_bias = go_signal.evaluate(&context);

        go_signal.set_facilitation_bias(0.3);
        let context2 = create_context("action2", 0.5);
        let eval_with_bias = go_signal.evaluate(&context2);

        assert!(
            eval_with_bias.go_strength > eval_no_bias.go_strength,
            "Facilitation bias should increase Go signal"
        );
    }
}
