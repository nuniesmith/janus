//! Activation threshold for execution
//!
//! Part of the Basal Ganglia region
//! Component: praxeological
//!
//! Implements an evidence-accumulation threshold model for gating
//! trade execution decisions. Inspired by drift-diffusion models (DDM),
//! evidence accumulates toward a decision boundary. When the accumulated
//! evidence exceeds the activation threshold, execution is triggered.
//!
//! Features:
//! - Adaptive threshold that adjusts based on recent decision quality
//! - Urgency signal that lowers the threshold over time (prevents indecision)
//! - Separate thresholds for entry vs exit decisions
//! - Hysteresis to prevent rapid on/off oscillation

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Type of decision being gated
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionType {
    /// Entry into a new position
    Entry,
    /// Exit from an existing position
    Exit,
    /// Increase an existing position
    ScaleIn,
    /// Decrease an existing position
    ScaleOut,
}

/// Result of a threshold evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdResult {
    /// Evidence has not yet reached the threshold — do not act
    BelowThreshold,
    /// Evidence has crossed the threshold — execute the decision
    Triggered,
    /// Evidence is in the hysteresis band — maintain previous state
    Hysteresis,
}

/// Configuration for the activation threshold
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    /// Base activation threshold for entry decisions (0.0 - 1.0)
    pub entry_threshold: f64,
    /// Base activation threshold for exit decisions (0.0 - 1.0)
    pub exit_threshold: f64,
    /// Base activation threshold for scale-in decisions
    pub scale_in_threshold: f64,
    /// Base activation threshold for scale-out decisions
    pub scale_out_threshold: f64,
    /// Hysteresis band width (prevents rapid toggling)
    pub hysteresis_band: f64,
    /// Urgency growth rate per time step (lowers threshold over time)
    pub urgency_rate: f64,
    /// Maximum urgency reduction (floor for threshold lowering)
    pub max_urgency_reduction: f64,
    /// Adaptation rate for threshold adjustment based on outcomes
    pub adaptation_rate: f64,
    /// Window size for tracking decision outcomes
    pub outcome_window: usize,
    /// Evidence decay factor per time step (0.0 = no decay, 1.0 = instant decay)
    pub evidence_decay: f64,
    /// Minimum allowed threshold after adaptation
    pub min_threshold: f64,
    /// Maximum allowed threshold after adaptation
    pub max_threshold: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            entry_threshold: 0.65,
            exit_threshold: 0.55,
            scale_in_threshold: 0.70,
            scale_out_threshold: 0.50,
            hysteresis_band: 0.05,
            urgency_rate: 0.005,
            max_urgency_reduction: 0.20,
            adaptation_rate: 0.02,
            outcome_window: 30,
            evidence_decay: 0.02,
            min_threshold: 0.30,
            max_threshold: 0.95,
        }
    }
}

/// A single evidence observation
#[derive(Debug, Clone)]
pub struct Evidence {
    /// The evidence value (0.0 - 1.0), where 1.0 = strongest support
    pub value: f64,
    /// Weight of this evidence source (default 1.0)
    pub weight: f64,
    /// Optional label for the evidence source
    pub source: Option<String>,
}

impl Evidence {
    /// Create a new evidence observation with default weight
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            weight: 1.0,
            source: None,
        }
    }

    /// Create a new weighted evidence observation
    pub fn weighted(value: f64, weight: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            weight: weight.max(0.0),
            source: None,
        }
    }

    /// Create a new labeled evidence observation
    pub fn labeled(value: f64, weight: f64, source: impl Into<String>) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            weight: weight.max(0.0),
            source: Some(source.into()),
        }
    }
}

/// Decision outcome for threshold adaptation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionOutcome {
    /// The decision type that was made
    pub decision_type: DecisionType,
    /// The accumulated evidence at the time of the decision
    pub evidence_at_decision: f64,
    /// The reward/result of the decision (positive = good, negative = bad)
    pub reward: f64,
}

/// Activation threshold for execution gating
pub struct Threshold {
    /// Configuration parameters
    config: ThresholdConfig,
    /// Current accumulated evidence (weighted sum)
    accumulated_evidence: f64,
    /// Total weight of accumulated evidence
    total_weight: f64,
    /// Number of time steps since last reset/trigger
    steps_since_reset: u64,
    /// Current urgency signal (reduces effective threshold over time)
    urgency: f64,
    /// Adaptive threshold offsets per decision type (learned from outcomes)
    entry_offset: f64,
    exit_offset: f64,
    scale_in_offset: f64,
    scale_out_offset: f64,
    /// Previous threshold result (for hysteresis)
    previous_result: ThresholdResult,
    /// Recent decision outcomes for adaptation
    recent_outcomes: VecDeque<DecisionOutcome>,
    /// Total number of evaluations performed
    evaluation_count: u64,
    /// Total number of triggers
    trigger_count: u64,
    /// Individual evidence contributions (for inspection)
    evidence_sources: Vec<(String, f64)>,
}

impl Default for Threshold {
    fn default() -> Self {
        Self::new()
    }
}

impl Threshold {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(ThresholdConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: ThresholdConfig) -> Self {
        Self {
            recent_outcomes: VecDeque::with_capacity(config.outcome_window),
            config,
            accumulated_evidence: 0.0,
            total_weight: 0.0,
            steps_since_reset: 0,
            urgency: 0.0,
            entry_offset: 0.0,
            exit_offset: 0.0,
            scale_in_offset: 0.0,
            scale_out_offset: 0.0,
            previous_result: ThresholdResult::BelowThreshold,
            evaluation_count: 0,
            trigger_count: 0,
            evidence_sources: Vec::new(),
        }
    }

    /// Main processing function — validates internal state
    pub fn process(&self) -> Result<()> {
        if self.config.entry_threshold < self.config.min_threshold
            || self.config.entry_threshold > self.config.max_threshold
        {
            return Err(Error::InvalidInput(
                "Entry threshold out of configured bounds".into(),
            ));
        }
        Ok(())
    }

    /// Add a piece of evidence to the accumulator
    pub fn add_evidence(&mut self, evidence: Evidence) {
        let weighted_value = evidence.value * evidence.weight;
        self.accumulated_evidence += weighted_value;
        self.total_weight += evidence.weight;

        if let Some(ref source) = evidence.source {
            self.evidence_sources.push((source.clone(), weighted_value));
        }
    }

    /// Add multiple evidence observations at once
    pub fn add_evidence_batch(&mut self, evidences: &[Evidence]) {
        for e in evidences {
            self.add_evidence(e.clone());
        }
    }

    /// Advance one time step (applies evidence decay and urgency growth)
    pub fn tick(&mut self) {
        // Decay accumulated evidence
        if self.config.evidence_decay > 0.0 {
            self.accumulated_evidence *= 1.0 - self.config.evidence_decay;
            self.total_weight *= 1.0 - self.config.evidence_decay;
        }

        // Grow urgency (up to maximum)
        self.urgency =
            (self.urgency + self.config.urgency_rate).min(self.config.max_urgency_reduction);

        self.steps_since_reset += 1;
    }

    /// Evaluate whether the accumulated evidence crosses the threshold
    /// for the given decision type.
    pub fn evaluate(&mut self, decision_type: DecisionType) -> ThresholdResult {
        self.evaluation_count += 1;

        let normalized_evidence = self.normalized_evidence();
        let effective_threshold = self.effective_threshold(decision_type);

        let upper_bound = effective_threshold;
        let lower_bound = effective_threshold - self.config.hysteresis_band;

        let result = if normalized_evidence >= upper_bound {
            ThresholdResult::Triggered
        } else if normalized_evidence >= lower_bound
            && self.previous_result == ThresholdResult::Triggered
        {
            // In hysteresis band and was previously triggered — hold state
            ThresholdResult::Hysteresis
        } else {
            ThresholdResult::BelowThreshold
        };

        if result == ThresholdResult::Triggered
            && self.previous_result != ThresholdResult::Triggered
        {
            self.trigger_count += 1;
        }

        self.previous_result = result;
        result
    }

    /// Record the outcome of a triggered decision for threshold adaptation
    pub fn record_outcome(&mut self, outcome: DecisionOutcome) {
        if self.recent_outcomes.len() >= self.config.outcome_window {
            self.recent_outcomes.pop_front();
        }
        self.recent_outcomes.push_back(outcome);

        // Adapt: if recent decisions have been bad, raise threshold (be more cautious)
        // If recent decisions have been good, lower threshold (be more responsive)
        self.adapt_thresholds();
    }

    /// Reset the evidence accumulator (typically after a decision is made)
    pub fn reset_evidence(&mut self) {
        self.accumulated_evidence = 0.0;
        self.total_weight = 0.0;
        self.urgency = 0.0;
        self.steps_since_reset = 0;
        self.previous_result = ThresholdResult::BelowThreshold;
        self.evidence_sources.clear();
    }

    /// Full reset including adaptation state
    pub fn reset_all(&mut self) {
        self.reset_evidence();
        self.entry_offset = 0.0;
        self.exit_offset = 0.0;
        self.scale_in_offset = 0.0;
        self.scale_out_offset = 0.0;
        self.recent_outcomes.clear();
        self.evaluation_count = 0;
        self.trigger_count = 0;
    }

    /// Get the current normalized evidence level (0.0 - 1.0)
    pub fn normalized_evidence(&self) -> f64 {
        if self.total_weight <= 0.0 {
            return 0.0;
        }
        (self.accumulated_evidence / self.total_weight).clamp(0.0, 1.0)
    }

    /// Get the raw accumulated evidence (not normalized)
    pub fn raw_evidence(&self) -> f64 {
        self.accumulated_evidence
    }

    /// Get the effective threshold for a decision type (base + adaptation - urgency)
    pub fn effective_threshold(&self, decision_type: DecisionType) -> f64 {
        let base = match decision_type {
            DecisionType::Entry => self.config.entry_threshold,
            DecisionType::Exit => self.config.exit_threshold,
            DecisionType::ScaleIn => self.config.scale_in_threshold,
            DecisionType::ScaleOut => self.config.scale_out_threshold,
        };

        let offset = match decision_type {
            DecisionType::Entry => self.entry_offset,
            DecisionType::Exit => self.exit_offset,
            DecisionType::ScaleIn => self.scale_in_offset,
            DecisionType::ScaleOut => self.scale_out_offset,
        };

        // Apply offset and subtract urgency, then clamp
        (base + offset - self.urgency).clamp(self.config.min_threshold, self.config.max_threshold)
    }

    /// Get the current urgency level
    pub fn urgency(&self) -> f64 {
        self.urgency
    }

    /// Get the number of time steps since last reset
    pub fn steps_since_reset(&self) -> u64 {
        self.steps_since_reset
    }

    /// Get total evaluations performed
    pub fn evaluation_count(&self) -> u64 {
        self.evaluation_count
    }

    /// Get total triggers
    pub fn trigger_count(&self) -> u64 {
        self.trigger_count
    }

    /// Get evidence source contributions (label, weighted_value)
    pub fn evidence_contributions(&self) -> &[(String, f64)] {
        &self.evidence_sources
    }

    /// Get the trigger rate (triggers / evaluations)
    pub fn trigger_rate(&self) -> f64 {
        if self.evaluation_count == 0 {
            return 0.0;
        }
        self.trigger_count as f64 / self.evaluation_count as f64
    }

    // ── internal ──

    /// Adapt thresholds based on recent decision outcomes
    fn adapt_thresholds(&mut self) {
        if self.recent_outcomes.is_empty() {
            return;
        }

        // Compute average reward per decision type
        let mut entry_sum = 0.0_f64;
        let mut entry_count = 0u32;
        let mut exit_sum = 0.0_f64;
        let mut exit_count = 0u32;
        let mut scale_in_sum = 0.0_f64;
        let mut scale_in_count = 0u32;
        let mut scale_out_sum = 0.0_f64;
        let mut scale_out_count = 0u32;

        for outcome in &self.recent_outcomes {
            match outcome.decision_type {
                DecisionType::Entry => {
                    entry_sum += outcome.reward;
                    entry_count += 1;
                }
                DecisionType::Exit => {
                    exit_sum += outcome.reward;
                    exit_count += 1;
                }
                DecisionType::ScaleIn => {
                    scale_in_sum += outcome.reward;
                    scale_in_count += 1;
                }
                DecisionType::ScaleOut => {
                    scale_out_sum += outcome.reward;
                    scale_out_count += 1;
                }
            }
        }

        let rate = self.config.adaptation_rate;
        let half_range = (self.config.max_threshold - self.config.min_threshold) / 2.0;

        // Negative average reward → raise threshold (be more cautious)
        // Positive average reward → lower threshold (be more responsive)
        if entry_count > 0 {
            let avg = entry_sum / entry_count as f64;
            self.entry_offset = (self.entry_offset - rate * avg).clamp(-half_range, half_range);
        }
        if exit_count > 0 {
            let avg = exit_sum / exit_count as f64;
            self.exit_offset = (self.exit_offset - rate * avg).clamp(-half_range, half_range);
        }
        if scale_in_count > 0 {
            let avg = scale_in_sum / scale_in_count as f64;
            self.scale_in_offset =
                (self.scale_in_offset - rate * avg).clamp(-half_range, half_range);
        }
        if scale_out_count > 0 {
            let avg = scale_out_sum / scale_out_count as f64;
            self.scale_out_offset =
                (self.scale_out_offset - rate * avg).clamp(-half_range, half_range);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = Threshold::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_evidence_accumulation_and_trigger() {
        let mut thresh = Threshold::with_config(ThresholdConfig {
            entry_threshold: 0.60,
            hysteresis_band: 0.0,
            urgency_rate: 0.0,
            evidence_decay: 0.0,
            ..Default::default()
        });

        // Add evidence below threshold
        thresh.add_evidence(Evidence::new(0.5));
        let result = thresh.evaluate(DecisionType::Entry);
        assert_eq!(result, ThresholdResult::BelowThreshold);

        // Reset and add strong evidence
        thresh.reset_evidence();
        thresh.add_evidence(Evidence::new(0.8));
        thresh.add_evidence(Evidence::new(0.9));
        let result = thresh.evaluate(DecisionType::Entry);
        assert_eq!(result, ThresholdResult::Triggered);
    }

    #[test]
    fn test_urgency_lowers_effective_threshold() {
        let mut thresh = Threshold::with_config(ThresholdConfig {
            entry_threshold: 0.70,
            urgency_rate: 0.10,
            max_urgency_reduction: 0.30,
            evidence_decay: 0.0,
            hysteresis_band: 0.0,
            ..Default::default()
        });

        let initial = thresh.effective_threshold(DecisionType::Entry);
        assert!((initial - 0.70).abs() < 1e-9);

        // Several ticks of urgency
        for _ in 0..3 {
            thresh.tick();
        }

        let after_urgency = thresh.effective_threshold(DecisionType::Entry);
        assert!(
            after_urgency < initial,
            "expected {} < {}",
            after_urgency,
            initial
        );
        assert!(
            (after_urgency - 0.40).abs() < 1e-9,
            "expected ~0.40, got {}",
            after_urgency
        );
    }

    #[test]
    fn test_urgency_capped_at_max() {
        let mut thresh = Threshold::with_config(ThresholdConfig {
            urgency_rate: 0.10,
            max_urgency_reduction: 0.15,
            ..Default::default()
        });

        for _ in 0..100 {
            thresh.tick();
        }

        assert!(
            (thresh.urgency() - 0.15).abs() < 1e-9,
            "urgency should be capped at 0.15, got {}",
            thresh.urgency()
        );
    }

    #[test]
    fn test_evidence_decay() {
        let mut thresh = Threshold::with_config(ThresholdConfig {
            evidence_decay: 0.50,
            urgency_rate: 0.0,
            ..Default::default()
        });

        thresh.add_evidence(Evidence::new(1.0));
        let before = thresh.raw_evidence();

        thresh.tick();
        let after = thresh.raw_evidence();

        assert!(
            after < before,
            "evidence should decay: {} >= {}",
            after,
            before
        );
        assert!((after - before * 0.5).abs() < 1e-9, "expected 50% decay");
    }

    #[test]
    fn test_hysteresis_band() {
        let mut thresh = Threshold::with_config(ThresholdConfig {
            entry_threshold: 0.60,
            hysteresis_band: 0.10,
            urgency_rate: 0.0,
            evidence_decay: 0.0,
            ..Default::default()
        });

        // First, push above threshold to trigger
        thresh.add_evidence(Evidence::new(0.80));
        thresh.add_evidence(Evidence::new(0.70));
        let result = thresh.evaluate(DecisionType::Entry);
        assert_eq!(result, ThresholdResult::Triggered);

        // Now reset and add evidence in hysteresis band (0.50 - 0.60)
        thresh.accumulated_evidence = 0.0;
        thresh.total_weight = 0.0;
        thresh.add_evidence(Evidence::new(0.55));
        let result = thresh.evaluate(DecisionType::Entry);
        assert_eq!(result, ThresholdResult::Hysteresis);

        // Drop below hysteresis band
        thresh.accumulated_evidence = 0.0;
        thresh.total_weight = 0.0;
        thresh.add_evidence(Evidence::new(0.40));
        let result = thresh.evaluate(DecisionType::Entry);
        assert_eq!(result, ThresholdResult::BelowThreshold);
    }

    #[test]
    fn test_different_thresholds_per_decision_type() {
        let thresh = Threshold::with_config(ThresholdConfig {
            entry_threshold: 0.70,
            exit_threshold: 0.50,
            scale_in_threshold: 0.80,
            scale_out_threshold: 0.40,
            ..Default::default()
        });

        assert!(
            thresh.effective_threshold(DecisionType::Entry)
                > thresh.effective_threshold(DecisionType::Exit)
        );
        assert!(
            thresh.effective_threshold(DecisionType::ScaleIn)
                > thresh.effective_threshold(DecisionType::Entry)
        );
        assert!(
            thresh.effective_threshold(DecisionType::ScaleOut)
                < thresh.effective_threshold(DecisionType::Exit)
        );
    }

    #[test]
    fn test_adaptation_raises_threshold_on_bad_outcomes() {
        let mut thresh = Threshold::with_config(ThresholdConfig {
            entry_threshold: 0.60,
            adaptation_rate: 0.10,
            urgency_rate: 0.0,
            ..Default::default()
        });

        let initial = thresh.effective_threshold(DecisionType::Entry);

        // Record several bad entry outcomes
        for _ in 0..10 {
            thresh.record_outcome(DecisionOutcome {
                decision_type: DecisionType::Entry,
                evidence_at_decision: 0.65,
                reward: -1.0,
            });
        }

        let adapted = thresh.effective_threshold(DecisionType::Entry);
        assert!(
            adapted > initial,
            "bad outcomes should raise threshold: {} <= {}",
            adapted,
            initial
        );
    }

    #[test]
    fn test_weighted_evidence() {
        let mut thresh = Threshold::with_config(ThresholdConfig {
            entry_threshold: 0.60,
            hysteresis_band: 0.0,
            urgency_rate: 0.0,
            evidence_decay: 0.0,
            ..Default::default()
        });

        // High-weight low evidence vs low-weight high evidence
        thresh.add_evidence(Evidence::weighted(0.3, 3.0));
        thresh.add_evidence(Evidence::weighted(0.9, 1.0));

        // Weighted average: (0.3*3 + 0.9*1) / (3+1) = 1.8/4 = 0.45
        let normalized = thresh.normalized_evidence();
        assert!(
            (normalized - 0.45).abs() < 1e-9,
            "expected 0.45, got {}",
            normalized
        );
    }

    #[test]
    fn test_labeled_evidence_tracking() {
        let mut thresh = Threshold::new();
        thresh.add_evidence(Evidence::labeled(0.8, 1.0, "momentum"));
        thresh.add_evidence(Evidence::labeled(0.6, 1.0, "mean_reversion"));

        let contributions = thresh.evidence_contributions();
        assert_eq!(contributions.len(), 2);
        assert_eq!(contributions[0].0, "momentum");
        assert_eq!(contributions[1].0, "mean_reversion");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut thresh = Threshold::new();
        thresh.add_evidence(Evidence::new(0.9));
        thresh.tick();
        thresh.tick();
        thresh.evaluate(DecisionType::Entry);

        thresh.reset_evidence();
        assert_eq!(thresh.normalized_evidence(), 0.0);
        assert_eq!(thresh.urgency(), 0.0);
        assert_eq!(thresh.steps_since_reset(), 0);
    }

    #[test]
    fn test_trigger_rate() {
        let mut thresh = Threshold::with_config(ThresholdConfig {
            entry_threshold: 0.50,
            hysteresis_band: 0.0,
            urgency_rate: 0.0,
            evidence_decay: 0.0,
            ..Default::default()
        });

        // Evaluation 1: below threshold
        thresh.add_evidence(Evidence::new(0.3));
        thresh.evaluate(DecisionType::Entry);
        thresh.reset_evidence();

        // Evaluation 2: above threshold
        thresh.add_evidence(Evidence::new(0.9));
        thresh.evaluate(DecisionType::Entry);
        thresh.reset_evidence();

        assert_eq!(thresh.evaluation_count(), 2);
        assert_eq!(thresh.trigger_count(), 1);
        assert!((thresh.trigger_rate() - 0.5).abs() < 1e-9);
    }
}
