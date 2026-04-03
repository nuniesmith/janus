//! Rule Compliance Scoring
//!
//! Part of the Prefrontal region
//! Component: ltn
//!
//! Aggregates compliance signals from multiple trading rules into a unified
//! compliance score. Each rule is evaluated independently and the results
//! are combined using configurable aggregation (weighted mean, min, product).
//!
//! ## Features
//!
//! - **Multi-rule aggregation**: Register named rules with weights, evaluate
//!   each independently, and aggregate into a single compliance score
//! - **EMA-smoothed compliance signal**: Running compliance score with
//!   configurable decay for downstream consumption
//! - **Grade thresholds**: Map numeric score to discrete compliance grades
//!   (Critical / Poor / Fair / Good / Excellent)
//! - **Per-rule breakdown**: Detailed per-rule scores, violation counts,
//!   and running statistics
//! - **Violation tracking**: Count and log violations (score below threshold)
//!   per rule and globally
//! - **Windowed diagnostics**: Recent compliance rate, mean score, grade
//!   distribution, trend detection
//! - **Running statistics**: Total evaluations, mean/min/max compliance,
//!   violation frequency, grade histogram
//! - **Configurable aggregation**: Weighted mean, minimum, product, or
//!   custom aggregation strategies

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the compliance scoring engine.
#[derive(Debug, Clone)]
pub struct ComplianceScoreConfig {
    /// EMA decay for compliance score smoothing (0 < decay < 1).
    pub ema_decay: f64,
    /// Minimum evaluations before EMA is considered initialised.
    pub min_samples: usize,
    /// Sliding window size for windowed diagnostics.
    pub window_size: usize,
    /// Score threshold below which a rule is considered violated.
    pub violation_threshold: f64,
    /// Maximum number of rules that can be registered.
    pub max_rules: usize,
    /// Grade thresholds (ascending): [critical, poor, fair, good].
    /// Scores below critical → Critical, below poor → Poor, etc.
    /// Scores at or above good → Excellent.
    pub grade_thresholds: [f64; 4],
}

impl Default for ComplianceScoreConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.1,
            min_samples: 5,
            window_size: 100,
            violation_threshold: 0.5,
            max_rules: 256,
            grade_thresholds: [0.2, 0.4, 0.6, 0.8],
        }
    }
}

// ---------------------------------------------------------------------------
// Compliance grade
// ---------------------------------------------------------------------------

/// Discrete compliance grade derived from a numeric score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplianceGrade {
    /// Score < critical threshold — immediate action required.
    Critical,
    /// Score < poor threshold — significant compliance issues.
    Poor,
    /// Score < fair threshold — marginal compliance.
    Fair,
    /// Score < good threshold — acceptable compliance.
    Good,
    /// Score >= good threshold — full compliance.
    Excellent,
}

impl ComplianceGrade {
    /// Numeric weight for grade comparison (higher = better).
    pub fn weight(self) -> u8 {
        match self {
            ComplianceGrade::Critical => 0,
            ComplianceGrade::Poor => 1,
            ComplianceGrade::Fair => 2,
            ComplianceGrade::Good => 3,
            ComplianceGrade::Excellent => 4,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            ComplianceGrade::Critical => "Critical",
            ComplianceGrade::Poor => "Poor",
            ComplianceGrade::Fair => "Fair",
            ComplianceGrade::Good => "Good",
            ComplianceGrade::Excellent => "Excellent",
        }
    }
}

impl std::fmt::Display for ComplianceGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Derive a grade from a numeric score using the given thresholds.
pub fn score_to_grade(score: f64, thresholds: &[f64; 4]) -> ComplianceGrade {
    if score < thresholds[0] {
        ComplianceGrade::Critical
    } else if score < thresholds[1] {
        ComplianceGrade::Poor
    } else if score < thresholds[2] {
        ComplianceGrade::Fair
    } else if score < thresholds[3] {
        ComplianceGrade::Good
    } else {
        ComplianceGrade::Excellent
    }
}

// ---------------------------------------------------------------------------
// Aggregation strategy
// ---------------------------------------------------------------------------

/// Strategy for combining per-rule scores into an aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregationStrategy {
    /// Weighted arithmetic mean (default).
    WeightedMean,
    /// Minimum score across all rules.
    Min,
    /// Product of all scores.
    Product,
    /// Arithmetic mean (equal weights).
    Mean,
}

// ---------------------------------------------------------------------------
// Rule record
// ---------------------------------------------------------------------------

/// A registered compliance rule with running statistics.
#[derive(Debug, Clone)]
pub struct RuleRecord {
    /// Rule identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Weight for weighted aggregation.
    pub weight: f64,
    /// Current compliance score for this rule [0, 1].
    pub score: f64,
    /// Number of times this rule has been evaluated.
    pub eval_count: u64,
    /// Number of times this rule has been violated.
    pub violation_count: u64,
    /// Sum of scores (for running mean).
    sum_score: f64,
    /// Minimum score observed.
    pub min_score: f64,
    /// Maximum score observed.
    pub max_score: f64,
}

impl RuleRecord {
    fn new(id: impl Into<String>, weight: f64) -> Self {
        Self {
            id: id.into(),
            description: String::new(),
            weight,
            score: 1.0,
            eval_count: 0,
            violation_count: 0,
            sum_score: 0.0,
            min_score: 1.0,
            max_score: 0.0,
        }
    }

    #[allow(dead_code)]
    fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Update the rule with a new score observation.
    fn update(&mut self, score: f64, violation_threshold: f64) {
        let s = score.clamp(0.0, 1.0);
        self.score = s;
        self.eval_count += 1;
        self.sum_score += s;
        if s < self.min_score {
            self.min_score = s;
        }
        if s > self.max_score {
            self.max_score = s;
        }
        if s < violation_threshold {
            self.violation_count += 1;
        }
    }

    /// Running mean score.
    pub fn mean_score(&self) -> f64 {
        if self.eval_count == 0 {
            return 1.0;
        }
        self.sum_score / self.eval_count as f64
    }

    /// Violation rate for this rule.
    pub fn violation_rate(&self) -> f64 {
        if self.eval_count == 0 {
            return 0.0;
        }
        self.violation_count as f64 / self.eval_count as f64
    }

    /// Reset running statistics but keep identity and weight.
    fn reset_stats(&mut self) {
        self.score = 1.0;
        self.eval_count = 0;
        self.violation_count = 0;
        self.sum_score = 0.0;
        self.min_score = 1.0;
        self.max_score = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Rule evaluation event
// ---------------------------------------------------------------------------

/// Per-rule evaluation result within a single round.
#[derive(Debug, Clone)]
pub struct RuleEvaluation {
    /// Rule ID.
    pub rule_id: String,
    /// Score for this round.
    pub score: f64,
    /// Whether the rule was violated this round.
    pub violated: bool,
    /// Weight of this rule.
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// Evaluation result
// ---------------------------------------------------------------------------

/// Result of a single compliance evaluation round.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Aggregate compliance score [0, 1].
    pub aggregate_score: f64,
    /// EMA-smoothed aggregate score.
    pub ema_score: f64,
    /// Compliance grade for this round.
    pub grade: ComplianceGrade,
    /// Per-rule evaluations.
    pub rule_evaluations: Vec<RuleEvaluation>,
    /// Number of rules violated this round.
    pub violations: usize,
    /// Total rules evaluated.
    pub total_rules: usize,
}

// ---------------------------------------------------------------------------
// Window record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct WindowRecord {
    aggregate_score: f64,
    grade: ComplianceGrade,
    violation_count: usize,
    #[allow(dead_code)]
    rule_count: usize,
    #[allow(dead_code)]
    tick: u64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the compliance scoring engine.
#[derive(Debug, Clone)]
pub struct ComplianceStats {
    /// Total number of evaluation rounds.
    pub total_evaluations: u64,
    /// Total individual rule violations across all rounds.
    pub total_violations: u64,
    /// Rounds where at least one violation occurred.
    pub rounds_with_violations: u64,
    /// Peak aggregate score observed.
    pub peak_score: f64,
    /// Lowest aggregate score observed.
    pub trough_score: f64,
    /// Sum of aggregate scores (for mean).
    pub sum_score: f64,
    /// Grade histogram.
    pub grade_counts: HashMap<ComplianceGrade, u64>,
    /// Number of registered rules.
    pub registered_rules: usize,
}

impl Default for ComplianceStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_violations: 0,
            rounds_with_violations: 0,
            peak_score: 0.0,
            trough_score: 1.0,
            sum_score: 0.0,
            grade_counts: HashMap::new(),
            registered_rules: 0,
        }
    }
}

impl ComplianceStats {
    /// Mean aggregate score over all rounds.
    pub fn mean_score(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.sum_score / self.total_evaluations as f64
    }

    /// Fraction of rounds with at least one violation.
    pub fn violation_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.rounds_with_violations as f64 / self.total_evaluations as f64
    }

    /// Mean violations per round.
    pub fn mean_violations_per_round(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.total_violations as f64 / self.total_evaluations as f64
    }

    /// Count for a specific grade.
    pub fn grade_count(&self, grade: ComplianceGrade) -> u64 {
        self.grade_counts.get(&grade).copied().unwrap_or(0)
    }

    /// Most frequent grade.
    pub fn dominant_grade(&self) -> Option<ComplianceGrade> {
        self.grade_counts
            .iter()
            .max_by_key(|&(_, &count)| count)
            .map(|(&grade, _)| grade)
    }
}

// ---------------------------------------------------------------------------
// ComplianceScore engine
// ---------------------------------------------------------------------------

/// Rule compliance scoring engine.
///
/// Register named rules with weights, feed per-rule compliance scores,
/// and the engine aggregates them into a unified compliance signal with
/// EMA smoothing, grade assignment, and windowed diagnostics.
pub struct ComplianceScore {
    config: ComplianceScoreConfig,

    /// Registered rules, keyed by ID.
    rules: HashMap<String, RuleRecord>,

    /// Insertion order for deterministic iteration.
    rule_order: Vec<String>,

    /// Aggregation strategy.
    aggregation: AggregationStrategy,

    /// EMA-smoothed aggregate score.
    ema_score: f64,
    ema_initialized: bool,

    /// Sliding window for diagnostics.
    recent: VecDeque<WindowRecord>,

    /// Current tick.
    current_tick: u64,

    /// Running statistics.
    stats: ComplianceStats,
}

impl Default for ComplianceScore {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplianceScore {
    /// Create a new compliance scoring engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(ComplianceScoreConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: ComplianceScoreConfig) -> Result<Self> {
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.max_rules == 0 {
            return Err(Error::InvalidInput("max_rules must be > 0".into()));
        }
        if config.violation_threshold < 0.0 || config.violation_threshold > 1.0 {
            return Err(Error::InvalidInput(
                "violation_threshold must be in [0, 1]".into(),
            ));
        }
        Ok(Self {
            config,
            rules: HashMap::new(),
            rule_order: Vec::new(),
            aggregation: AggregationStrategy::WeightedMean,
            ema_score: 0.0,
            ema_initialized: false,
            recent: VecDeque::new(),
            current_tick: 0,
            stats: ComplianceStats::default(),
        })
    }

    /// Set the aggregation strategy.
    pub fn set_aggregation(&mut self, strategy: AggregationStrategy) {
        self.aggregation = strategy;
    }

    /// Get the current aggregation strategy.
    pub fn aggregation(&self) -> AggregationStrategy {
        self.aggregation
    }

    // -----------------------------------------------------------------------
    // Rule management
    // -----------------------------------------------------------------------

    /// Register a new compliance rule with a given weight.
    pub fn register_rule(&mut self, id: impl Into<String>, weight: f64) -> Result<()> {
        let id = id.into();
        if self.rules.contains_key(&id) {
            return Err(Error::InvalidInput(format!(
                "rule '{}' already registered",
                id
            )));
        }
        if self.rules.len() >= self.config.max_rules {
            return Err(Error::ResourceExhausted(format!(
                "maximum rules ({}) reached",
                self.config.max_rules
            )));
        }
        if weight < 0.0 {
            return Err(Error::InvalidInput("weight must be >= 0".into()));
        }
        self.rules.insert(id.clone(), RuleRecord::new(&id, weight));
        self.rule_order.push(id);
        self.stats.registered_rules = self.rules.len();
        Ok(())
    }

    /// Register a rule with a description.
    pub fn register_rule_with_desc(
        &mut self,
        id: impl Into<String>,
        weight: f64,
        description: impl Into<String>,
    ) -> Result<()> {
        let id = id.into();
        self.register_rule(&id, weight)?;
        if let Some(rule) = self.rules.get_mut(&id) {
            rule.description = description.into();
        }
        Ok(())
    }

    /// Remove a rule by ID.
    pub fn deregister_rule(&mut self, id: &str) -> bool {
        let removed = self.rules.remove(id).is_some();
        if removed {
            self.rule_order.retain(|r| r != id);
            self.stats.registered_rules = self.rules.len();
        }
        removed
    }

    /// Update a single rule's compliance score.
    pub fn update_rule(&mut self, id: &str, score: f64) -> Result<()> {
        let threshold = self.config.violation_threshold;
        let rule = self
            .rules
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("rule '{}' not found", id)))?;
        rule.update(score, threshold);
        Ok(())
    }

    /// Batch-update multiple rules at once.
    pub fn update_rules(&mut self, updates: &[(&str, f64)]) -> Result<()> {
        for (id, score) in updates {
            self.update_rule(id, *score)?;
        }
        Ok(())
    }

    /// Get a rule record by ID.
    pub fn rule(&self, id: &str) -> Option<&RuleRecord> {
        self.rules.get(id)
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// All rule IDs in registration order.
    pub fn rule_ids(&self) -> &[String] {
        &self.rule_order
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate all rules and produce an aggregate compliance score.
    ///
    /// This is the main "tick" method. Call after updating rule scores
    /// to compute the aggregate, update EMA, assign grade, and record
    /// windowed diagnostics.
    pub fn evaluate(&mut self) -> EvaluationResult {
        self.current_tick += 1;

        let mut rule_evaluations = Vec::with_capacity(self.rules.len());
        let mut scores_and_weights = Vec::with_capacity(self.rules.len());
        let mut violations = 0usize;

        for id in &self.rule_order {
            let rule = &self.rules[id];
            let violated = rule.score < self.config.violation_threshold;
            if violated {
                violations += 1;
            }
            rule_evaluations.push(RuleEvaluation {
                rule_id: id.clone(),
                score: rule.score,
                violated,
                weight: rule.weight,
            });
            scores_and_weights.push((rule.score, rule.weight));
        }

        // Compute aggregate
        let aggregate_score = if scores_and_weights.is_empty() {
            1.0 // vacuous compliance
        } else {
            match self.aggregation {
                AggregationStrategy::WeightedMean => {
                    let total_weight: f64 = scores_and_weights.iter().map(|(_, w)| w).sum();
                    if total_weight <= 0.0 {
                        scores_and_weights.iter().map(|(s, _)| s).sum::<f64>()
                            / scores_and_weights.len() as f64
                    } else {
                        scores_and_weights.iter().map(|(s, w)| s * w).sum::<f64>() / total_weight
                    }
                }
                AggregationStrategy::Min => scores_and_weights
                    .iter()
                    .map(|(s, _)| *s)
                    .fold(f64::INFINITY, f64::min)
                    .max(0.0),
                AggregationStrategy::Product => {
                    scores_and_weights.iter().map(|(s, _)| *s).product::<f64>()
                }
                AggregationStrategy::Mean => {
                    scores_and_weights.iter().map(|(s, _)| s).sum::<f64>()
                        / scores_and_weights.len() as f64
                }
            }
        };

        let aggregate_score = aggregate_score.clamp(0.0, 1.0);

        // Update EMA
        if !self.ema_initialized {
            self.ema_score = aggregate_score;
            self.ema_initialized = true;
        } else {
            self.ema_score = self.config.ema_decay * aggregate_score
                + (1.0 - self.config.ema_decay) * self.ema_score;
        }

        // Compute grade
        let grade = score_to_grade(aggregate_score, &self.config.grade_thresholds);

        // Update stats
        self.stats.total_evaluations += 1;
        self.stats.total_violations += violations as u64;
        self.stats.sum_score += aggregate_score;
        if aggregate_score > self.stats.peak_score {
            self.stats.peak_score = aggregate_score;
        }
        if aggregate_score < self.stats.trough_score {
            self.stats.trough_score = aggregate_score;
        }
        if violations > 0 {
            self.stats.rounds_with_violations += 1;
        }
        *self.stats.grade_counts.entry(grade).or_insert(0) += 1;

        // Window record
        let record = WindowRecord {
            aggregate_score,
            grade,
            violation_count: violations,
            rule_count: self.rules.len(),
            tick: self.current_tick,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        EvaluationResult {
            aggregate_score,
            ema_score: self.ema_score,
            grade,
            rule_evaluations,
            violations,
            total_rules: self.rules.len(),
        }
    }

    /// Compute the current aggregate score without advancing the tick
    /// or updating any state. Useful for preview / dry-run checks.
    pub fn preview_score(&self) -> f64 {
        let scores_and_weights: Vec<(f64, f64)> = self
            .rule_order
            .iter()
            .map(|id| {
                let r = &self.rules[id];
                (r.score, r.weight)
            })
            .collect();

        if scores_and_weights.is_empty() {
            return 1.0;
        }

        let agg = match self.aggregation {
            AggregationStrategy::WeightedMean => {
                let total_weight: f64 = scores_and_weights.iter().map(|(_, w)| w).sum();
                if total_weight <= 0.0 {
                    scores_and_weights.iter().map(|(s, _)| s).sum::<f64>()
                        / scores_and_weights.len() as f64
                } else {
                    scores_and_weights.iter().map(|(s, w)| s * w).sum::<f64>() / total_weight
                }
            }
            AggregationStrategy::Min => scores_and_weights
                .iter()
                .map(|(s, _)| *s)
                .fold(f64::INFINITY, f64::min)
                .max(0.0),
            AggregationStrategy::Product => {
                scores_and_weights.iter().map(|(s, _)| *s).product::<f64>()
            }
            AggregationStrategy::Mean => {
                scores_and_weights.iter().map(|(s, _)| s).sum::<f64>()
                    / scores_and_weights.len() as f64
            }
        };
        agg.clamp(0.0, 1.0)
    }

    /// Get the current compliance grade based on the EMA score.
    pub fn current_grade(&self) -> ComplianceGrade {
        score_to_grade(self.ema_score, &self.config.grade_thresholds)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current EMA-smoothed aggregate compliance score.
    pub fn ema_score(&self) -> f64 {
        self.ema_score
    }

    /// Whether the EMA has had enough samples to be reliable.
    pub fn is_warmed_up(&self) -> bool {
        self.stats.total_evaluations >= self.config.min_samples as u64
    }

    /// Current tick count.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Running statistics.
    pub fn stats(&self) -> &ComplianceStats {
        &self.stats
    }

    /// Confidence metric: increases as more evaluations are performed.
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_evaluations as f64;
        let min_s = self.config.min_samples as f64;
        (n / (n + min_s)).min(1.0)
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Mean aggregate score over the recent window.
    pub fn windowed_mean_score(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.aggregate_score).sum();
        sum / self.recent.len() as f64
    }

    /// Fraction of recent rounds that contained at least one violation.
    pub fn windowed_violation_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let count = self.recent.iter().filter(|r| r.violation_count > 0).count();
        count as f64 / self.recent.len() as f64
    }

    /// Most common grade in the recent window.
    pub fn windowed_dominant_grade(&self) -> Option<ComplianceGrade> {
        if self.recent.is_empty() {
            return None;
        }
        let mut counts: HashMap<ComplianceGrade, usize> = HashMap::new();
        for r in &self.recent {
            *counts.entry(r.grade).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(g, _)| g)
    }

    /// Detect whether compliance is trending upward over the window.
    pub fn is_compliance_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.aggregate_score)
            .sum::<f64>()
            / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.aggregate_score)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half > first_half + 0.01
    }

    /// Detect whether compliance is trending downward over the window.
    pub fn is_compliance_decreasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let mid = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.aggregate_score)
            .sum::<f64>()
            / mid as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.aggregate_score)
            .sum::<f64>()
            / (self.recent.len() - mid) as f64;
        second_half < first_half - 0.01
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all internal state but keep rules registered.
    pub fn reset(&mut self) {
        self.ema_score = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = ComplianceStats {
            registered_rules: self.rules.len(),
            ..ComplianceStats::default()
        };
        for rule in self.rules.values_mut() {
            rule.reset_stats();
        }
    }

    /// Reset everything including rules.
    pub fn reset_all(&mut self) {
        self.rules.clear();
        self.rule_order.clear();
        self.ema_score = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.current_tick = 0;
        self.stats = ComplianceStats::default();
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
    // Grade tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_to_grade_critical() {
        let thresholds = [0.2, 0.4, 0.6, 0.8];
        assert_eq!(score_to_grade(0.1, &thresholds), ComplianceGrade::Critical);
    }

    #[test]
    fn test_score_to_grade_poor() {
        let thresholds = [0.2, 0.4, 0.6, 0.8];
        assert_eq!(score_to_grade(0.3, &thresholds), ComplianceGrade::Poor);
    }

    #[test]
    fn test_score_to_grade_fair() {
        let thresholds = [0.2, 0.4, 0.6, 0.8];
        assert_eq!(score_to_grade(0.5, &thresholds), ComplianceGrade::Fair);
    }

    #[test]
    fn test_score_to_grade_good() {
        let thresholds = [0.2, 0.4, 0.6, 0.8];
        assert_eq!(score_to_grade(0.7, &thresholds), ComplianceGrade::Good);
    }

    #[test]
    fn test_score_to_grade_excellent() {
        let thresholds = [0.2, 0.4, 0.6, 0.8];
        assert_eq!(score_to_grade(0.9, &thresholds), ComplianceGrade::Excellent);
    }

    #[test]
    fn test_grade_weight_ordering() {
        assert!(ComplianceGrade::Excellent.weight() > ComplianceGrade::Good.weight());
        assert!(ComplianceGrade::Good.weight() > ComplianceGrade::Fair.weight());
        assert!(ComplianceGrade::Fair.weight() > ComplianceGrade::Poor.weight());
        assert!(ComplianceGrade::Poor.weight() > ComplianceGrade::Critical.weight());
    }

    #[test]
    fn test_grade_label() {
        assert_eq!(ComplianceGrade::Critical.label(), "Critical");
        assert_eq!(ComplianceGrade::Excellent.label(), "Excellent");
    }

    #[test]
    fn test_grade_display() {
        let s = format!("{}", ComplianceGrade::Good);
        assert_eq!(s, "Good");
    }

    // -----------------------------------------------------------------------
    // Construction & configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_default() {
        let cs = ComplianceScore::new();
        assert_eq!(cs.rule_count(), 0);
        assert_eq!(cs.current_tick(), 0);
    }

    #[test]
    fn test_invalid_config_ema_zero() {
        let mut cfg = ComplianceScoreConfig::default();
        cfg.ema_decay = 0.0;
        assert!(ComplianceScore::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_one() {
        let mut cfg = ComplianceScoreConfig::default();
        cfg.ema_decay = 1.0;
        assert!(ComplianceScore::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_window_zero() {
        let mut cfg = ComplianceScoreConfig::default();
        cfg.window_size = 0;
        assert!(ComplianceScore::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_rules_zero() {
        let mut cfg = ComplianceScoreConfig::default();
        cfg.max_rules = 0;
        assert!(ComplianceScore::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_violation_threshold() {
        let mut cfg = ComplianceScoreConfig::default();
        cfg.violation_threshold = 1.5;
        assert!(ComplianceScore::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Rule management
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_rule() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("position_limit", 1.0).unwrap();
        assert_eq!(cs.rule_count(), 1);
        assert!(cs.rule("position_limit").is_some());
    }

    #[test]
    fn test_register_rule_with_desc() {
        let mut cs = ComplianceScore::new();
        cs.register_rule_with_desc("r1", 1.0, "Max position size check")
            .unwrap();
        let r = cs.rule("r1").unwrap();
        assert_eq!(r.description, "Max position size check");
    }

    #[test]
    fn test_register_duplicate() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        assert!(cs.register_rule("r1", 2.0).is_err());
    }

    #[test]
    fn test_register_max_capacity() {
        let mut cfg = ComplianceScoreConfig::default();
        cfg.max_rules = 2;
        let mut cs = ComplianceScore::with_config(cfg).unwrap();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();
        assert!(cs.register_rule("c", 1.0).is_err());
    }

    #[test]
    fn test_register_negative_weight() {
        let mut cs = ComplianceScore::new();
        assert!(cs.register_rule("bad", -1.0).is_err());
    }

    #[test]
    fn test_deregister_rule() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        assert!(cs.deregister_rule("r1"));
        assert_eq!(cs.rule_count(), 0);
        assert!(!cs.deregister_rule("r1"));
    }

    #[test]
    fn test_rule_ids_in_order() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("c", 1.0).unwrap();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();
        assert_eq!(cs.rule_ids(), &["c", "a", "b"]);
    }

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_rule() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.update_rule("r1", 0.8).unwrap();
        assert!((cs.rule("r1").unwrap().score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_update_nonexistent() {
        let mut cs = ComplianceScore::new();
        assert!(cs.update_rule("nope", 0.5).is_err());
    }

    #[test]
    fn test_update_rules_batch() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();
        cs.update_rules(&[("a", 0.9), ("b", 0.7)]).unwrap();
        assert!((cs.rule("a").unwrap().score - 0.9).abs() < 1e-10);
        assert!((cs.rule("b").unwrap().score - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_update_clamps() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.update_rule("r1", 1.5).unwrap();
        assert!((cs.rule("r1").unwrap().score - 1.0).abs() < 1e-10);
        cs.update_rule("r1", -0.5).unwrap();
        assert!((cs.rule("r1").unwrap().score - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Rule record stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_rule_mean_score() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.update_rule("r1", 0.4).unwrap();
        cs.update_rule("r1", 0.6).unwrap();
        assert!((cs.rule("r1").unwrap().mean_score() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rule_violation_rate() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.update_rule("r1", 0.3).unwrap(); // violation (< 0.5)
        cs.update_rule("r1", 0.8).unwrap(); // ok
        assert!((cs.rule("r1").unwrap().violation_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rule_min_max() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.update_rule("r1", 0.3).unwrap();
        cs.update_rule("r1", 0.9).unwrap();
        cs.update_rule("r1", 0.6).unwrap();
        let r = cs.rule("r1").unwrap();
        assert!((r.min_score - 0.3).abs() < 1e-10);
        assert!((r.max_score - 0.9).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_empty() {
        let mut cs = ComplianceScore::new();
        let result = cs.evaluate();
        assert!((result.aggregate_score - 1.0).abs() < 1e-10);
        assert_eq!(result.grade, ComplianceGrade::Excellent);
        assert_eq!(cs.current_tick(), 1);
    }

    #[test]
    fn test_evaluate_weighted_mean() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 3.0).unwrap();
        cs.update_rule("a", 0.2).unwrap();
        cs.update_rule("b", 0.6).unwrap();

        let result = cs.evaluate();
        // weighted mean = (0.2*1 + 0.6*3) / (1+3) = 2.0/4 = 0.5
        assert!((result.aggregate_score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_min_aggregation() {
        let mut cs = ComplianceScore::new();
        cs.set_aggregation(AggregationStrategy::Min);
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();
        cs.update_rule("a", 0.3).unwrap();
        cs.update_rule("b", 0.8).unwrap();

        let result = cs.evaluate();
        assert!((result.aggregate_score - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_product_aggregation() {
        let mut cs = ComplianceScore::new();
        cs.set_aggregation(AggregationStrategy::Product);
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();
        cs.update_rule("a", 0.5).unwrap();
        cs.update_rule("b", 0.8).unwrap();

        let result = cs.evaluate();
        assert!((result.aggregate_score - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_mean_aggregation() {
        let mut cs = ComplianceScore::new();
        cs.set_aggregation(AggregationStrategy::Mean);
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();
        cs.update_rule("a", 0.4).unwrap();
        cs.update_rule("b", 0.6).unwrap();

        let result = cs.evaluate();
        assert!((result.aggregate_score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_detects_violations() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();
        cs.update_rule("a", 0.3).unwrap(); // violation
        cs.update_rule("b", 0.8).unwrap(); // ok

        let result = cs.evaluate();
        assert_eq!(result.violations, 1);
    }

    #[test]
    fn test_evaluate_ema_initializes() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.update_rule("r1", 0.7).unwrap();

        let result = cs.evaluate();
        assert!((result.ema_score - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_ema_smoothing() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.update_rule("r1", 0.5).unwrap();
        cs.evaluate(); // ema = 0.5

        cs.update_rule("r1", 1.0).unwrap();
        let r2 = cs.evaluate();
        // ema = 0.1 * 1.0 + 0.9 * 0.5 = 0.55
        assert!((r2.ema_score - 0.55).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_grade_assignment() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        cs.update_rule("r1", 0.1).unwrap();
        assert_eq!(cs.evaluate().grade, ComplianceGrade::Critical);

        cs.update_rule("r1", 0.3).unwrap();
        assert_eq!(cs.evaluate().grade, ComplianceGrade::Poor);

        cs.update_rule("r1", 0.5).unwrap();
        assert_eq!(cs.evaluate().grade, ComplianceGrade::Fair);

        cs.update_rule("r1", 0.7).unwrap();
        assert_eq!(cs.evaluate().grade, ComplianceGrade::Good);

        cs.update_rule("r1", 0.9).unwrap();
        assert_eq!(cs.evaluate().grade, ComplianceGrade::Excellent);
    }

    #[test]
    fn test_evaluate_rule_evaluations() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 2.0).unwrap();
        cs.update_rule("a", 0.9).unwrap();
        cs.update_rule("b", 0.3).unwrap();

        let result = cs.evaluate();
        assert_eq!(result.rule_evaluations.len(), 2);
        assert_eq!(result.rule_evaluations[0].rule_id, "a");
        assert!(!result.rule_evaluations[0].violated);
        assert_eq!(result.rule_evaluations[1].rule_id, "b");
        assert!(result.rule_evaluations[1].violated);
    }

    // -----------------------------------------------------------------------
    // Preview
    // -----------------------------------------------------------------------

    #[test]
    fn test_preview_score() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();
        cs.update_rule("a", 0.6).unwrap();
        cs.update_rule("b", 0.4).unwrap();

        let preview = cs.preview_score();
        // weighted mean with equal weights = mean = 0.5
        assert!((preview - 0.5).abs() < 1e-10);
        // Should not have advanced the tick
        assert_eq!(cs.current_tick(), 0);
    }

    // -----------------------------------------------------------------------
    // Aggregation strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_aggregation() {
        let mut cs = ComplianceScore::new();
        cs.set_aggregation(AggregationStrategy::Min);
        assert_eq!(cs.aggregation(), AggregationStrategy::Min);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let cs = ComplianceScore::new();
        let s = cs.stats();
        assert_eq!(s.total_evaluations, 0);
        assert!((s.mean_score() - 0.0).abs() < 1e-10);
        assert!((s.violation_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_after_evaluations() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        cs.update_rule("r1", 0.3).unwrap();
        cs.evaluate(); // violation

        cs.update_rule("r1", 0.8).unwrap();
        cs.evaluate(); // ok

        let s = cs.stats();
        assert_eq!(s.total_evaluations, 2);
        assert_eq!(s.total_violations, 1);
        assert_eq!(s.rounds_with_violations, 1);
        assert!((s.violation_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_mean_score() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        cs.update_rule("r1", 0.4).unwrap();
        cs.evaluate();

        cs.update_rule("r1", 0.6).unwrap();
        cs.evaluate();

        assert!((cs.stats().mean_score() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_peak_and_trough() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        cs.update_rule("r1", 0.3).unwrap();
        cs.evaluate();

        cs.update_rule("r1", 0.9).unwrap();
        cs.evaluate();

        assert!((cs.stats().peak_score - 0.9).abs() < 1e-10);
        assert!((cs.stats().trough_score - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_stats_grade_counts() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        cs.update_rule("r1", 0.9).unwrap();
        cs.evaluate(); // Excellent

        cs.update_rule("r1", 0.9).unwrap();
        cs.evaluate(); // Excellent

        cs.update_rule("r1", 0.1).unwrap();
        cs.evaluate(); // Critical

        assert_eq!(cs.stats().grade_count(ComplianceGrade::Excellent), 2);
        assert_eq!(cs.stats().grade_count(ComplianceGrade::Critical), 1);
    }

    #[test]
    fn test_stats_dominant_grade() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        for _ in 0..5 {
            cs.update_rule("r1", 0.9).unwrap();
            cs.evaluate();
        }
        for _ in 0..2 {
            cs.update_rule("r1", 0.1).unwrap();
            cs.evaluate();
        }

        assert_eq!(
            cs.stats().dominant_grade(),
            Some(ComplianceGrade::Excellent)
        );
    }

    #[test]
    fn test_stats_mean_violations_per_round() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 1.0).unwrap();

        cs.update_rules(&[("a", 0.3), ("b", 0.3)]).unwrap();
        cs.evaluate(); // 2 violations

        cs.update_rules(&[("a", 0.9), ("b", 0.9)]).unwrap();
        cs.evaluate(); // 0 violations

        // mean = 2/2 = 1.0
        assert!((cs.stats().mean_violations_per_round() - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    #[test]
    fn test_windowed_mean_score() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        cs.update_rule("r1", 0.4).unwrap();
        cs.evaluate();

        cs.update_rule("r1", 0.6).unwrap();
        cs.evaluate();

        assert!((cs.windowed_mean_score() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_score_empty() {
        let cs = ComplianceScore::new();
        assert!((cs.windowed_mean_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_violation_rate() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        cs.update_rule("r1", 0.3).unwrap();
        cs.evaluate();

        cs.update_rule("r1", 0.8).unwrap();
        cs.evaluate();

        assert!((cs.windowed_violation_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_violation_rate_empty() {
        let cs = ComplianceScore::new();
        assert!((cs.windowed_violation_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_dominant_grade() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        for _ in 0..3 {
            cs.update_rule("r1", 0.9).unwrap();
            cs.evaluate();
        }
        cs.update_rule("r1", 0.1).unwrap();
        cs.evaluate();

        assert_eq!(
            cs.windowed_dominant_grade(),
            Some(ComplianceGrade::Excellent)
        );
    }

    #[test]
    fn test_windowed_dominant_grade_empty() {
        let cs = ComplianceScore::new();
        assert_eq!(cs.windowed_dominant_grade(), None);
    }

    #[test]
    fn test_is_compliance_increasing() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        for i in 0..10 {
            cs.update_rule("r1", i as f64 / 10.0).unwrap();
            cs.evaluate();
        }
        assert!(cs.is_compliance_increasing());
    }

    #[test]
    fn test_is_compliance_decreasing() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        for i in (0..10).rev() {
            cs.update_rule("r1", i as f64 / 10.0).unwrap();
            cs.evaluate();
        }
        assert!(cs.is_compliance_decreasing());
    }

    #[test]
    fn test_trend_insufficient_data() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.evaluate();
        assert!(!cs.is_compliance_increasing());
        assert!(!cs.is_compliance_decreasing());
    }

    // -----------------------------------------------------------------------
    // Warmup & confidence
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_warmed_up() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();

        for _ in 0..4 {
            cs.evaluate();
        }
        assert!(!cs.is_warmed_up());

        cs.evaluate();
        assert!(cs.is_warmed_up());
    }

    #[test]
    fn test_confidence_increases() {
        let mut cs = ComplianceScore::new();
        let c0 = cs.confidence();
        cs.register_rule("r1", 1.0).unwrap();
        cs.evaluate();
        let c1 = cs.confidence();
        for _ in 0..10 {
            cs.evaluate();
        }
        let c11 = cs.confidence();

        assert!(c1 > c0);
        assert!(c11 > c1);
        assert!(c11 <= 1.0);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 1.0).unwrap();
        cs.register_rule("b", 2.0).unwrap();
        cs.update_rules(&[("a", 0.5), ("b", 0.8)]).unwrap();
        cs.evaluate();
        cs.evaluate();

        cs.reset();

        assert_eq!(cs.current_tick(), 0);
        assert_eq!(cs.stats().total_evaluations, 0);
        assert_eq!(cs.rule_count(), 2); // rules kept
        assert!((cs.rule("a").unwrap().score - 1.0).abs() < 1e-10); // reset to default
    }

    #[test]
    fn test_reset_all() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.evaluate();

        cs.reset_all();

        assert_eq!(cs.current_tick(), 0);
        assert_eq!(cs.stats().total_evaluations, 0);
        assert_eq!(cs.rule_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Process compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_process() {
        let cs = ComplianceScore::new();
        assert!(cs.process().is_ok());
    }

    // -----------------------------------------------------------------------
    // current_grade
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_grade() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("r1", 1.0).unwrap();
        cs.update_rule("r1", 0.9).unwrap();
        cs.evaluate();
        assert_eq!(cs.current_grade(), ComplianceGrade::Excellent);
    }

    // -----------------------------------------------------------------------
    // Window eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_window_eviction() {
        let mut cfg = ComplianceScoreConfig::default();
        cfg.window_size = 3;
        let mut cs = ComplianceScore::with_config(cfg).unwrap();
        cs.register_rule("r1", 1.0).unwrap();

        for _ in 0..5 {
            cs.evaluate();
        }

        assert!(cs.recent.len() <= 3);
    }

    // -----------------------------------------------------------------------
    // Registered rules stat
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_registered_rules() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 1.0).unwrap();
        assert_eq!(cs.stats().registered_rules, 1);

        cs.register_rule("b", 1.0).unwrap();
        assert_eq!(cs.stats().registered_rules, 2);

        cs.deregister_rule("a");
        assert_eq!(cs.stats().registered_rules, 1);
    }

    // -----------------------------------------------------------------------
    // Zero-weight rules
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_weight_rules_fallback_to_mean() {
        let mut cs = ComplianceScore::new();
        cs.register_rule("a", 0.0).unwrap();
        cs.register_rule("b", 0.0).unwrap();
        cs.update_rule("a", 0.4).unwrap();
        cs.update_rule("b", 0.6).unwrap();

        let result = cs.evaluate();
        // All zero weights → fallback to plain mean = 0.5
        assert!((result.aggregate_score - 0.5).abs() < 1e-10);
    }
}
