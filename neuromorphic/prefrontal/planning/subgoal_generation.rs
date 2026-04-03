//! Subgoal Generation — Generate Intermediate Subgoals
//!
//! Part of the Prefrontal region
//! Component: planning
//!
//! Generates intermediate subgoals by analysing the gap between current
//! state and target state, then producing a series of stepping-stone
//! goals that bridge that gap. Supports gap analysis, intermediate
//! target generation, priority-based ordering, and dependency chain
//! construction.
//!
//! ## Features
//!
//! - **Gap analysis**: Compare current metric values against target
//!   values and compute per-metric gaps (absolute and relative).
//! - **Intermediate target generation**: Given a gap, produce N evenly
//!   spaced intermediate targets (linear interpolation) or
//!   exponentially spaced targets (for compounding metrics).
//! - **Priority-based ordering**: Subgoals are assigned priorities
//!   based on gap severity, urgency, and configurable weighting.
//! - **Dependency chain construction**: Generated subgoals form a
//!   sequential chain where each subgoal depends on the previous one.
//! - **Milestone markers**: Optionally mark specific subgoals as
//!   milestones (e.g. 25%, 50%, 75%) for progress tracking.
//! - **Feasibility filtering**: Filter out subgoals whose targets
//!   are unreachable given constraints (max step size, min duration).
//! - **Running statistics**: Total subgoals generated, completion
//!   rates, mean gap sizes, generation history.
//! - **Batch generation**: Generate subgoals for multiple metrics
//!   simultaneously, interleaving them by priority.

use crate::common::{Error, Result};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the subgoal generation engine.
#[derive(Debug, Clone)]
pub struct SubgoalGenerationConfig {
    /// Default number of intermediate subgoals per gap.
    pub default_num_subgoals: usize,
    /// Maximum number of subgoals that can be generated in one batch.
    pub max_subgoals_per_batch: usize,
    /// Maximum total subgoals tracked by the engine.
    pub max_total_subgoals: usize,
    /// Maximum step size as a fraction of the total gap per subgoal.
    pub max_step_fraction: f64,
    /// Minimum absolute gap required to generate subgoals.
    pub min_gap_threshold: f64,
    /// Milestone fractions (e.g. [0.25, 0.5, 0.75]).
    pub milestone_fractions: Vec<f64>,
    /// Priority weight for gap magnitude (larger gap = higher priority).
    pub gap_weight: f64,
    /// Priority weight for urgency (closer deadline = higher priority).
    pub urgency_weight: f64,
}

impl Default for SubgoalGenerationConfig {
    fn default() -> Self {
        Self {
            default_num_subgoals: 5,
            max_subgoals_per_batch: 64,
            max_total_subgoals: 1024,
            max_step_fraction: 0.5,
            min_gap_threshold: 1e-6,
            milestone_fractions: vec![0.25, 0.5, 0.75],
            gap_weight: 0.6,
            urgency_weight: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolation strategy
// ---------------------------------------------------------------------------

/// Strategy for spacing intermediate targets between current and goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolationStrategy {
    /// Equally spaced targets (linear interpolation).
    Linear,
    /// Exponentially spaced targets (for compounding metrics like
    /// portfolio value). Steps grow larger as you approach the target.
    Exponential,
    /// Front-loaded: larger steps early, smaller steps later.
    /// Useful for drawdown recovery where initial gains are easier.
    FrontLoaded,
    /// Back-loaded: smaller steps early, larger steps later.
    /// Useful for momentum-based targets.
    BackLoaded,
}

impl InterpolationStrategy {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            InterpolationStrategy::Linear => "Linear",
            InterpolationStrategy::Exponential => "Exponential",
            InterpolationStrategy::FrontLoaded => "FrontLoaded",
            InterpolationStrategy::BackLoaded => "BackLoaded",
        }
    }
}

// ---------------------------------------------------------------------------
// Gap analysis
// ---------------------------------------------------------------------------

/// Result of analysing the gap between current and target values.
#[derive(Debug, Clone)]
pub struct GapAnalysis {
    /// Metric name.
    pub metric: String,
    /// Current value.
    pub current: f64,
    /// Target value.
    pub target: f64,
    /// Absolute gap (target - current). Positive means we need to
    /// increase; negative means we need to decrease.
    pub absolute_gap: f64,
    /// Relative gap as a fraction of target. Positive = shortfall,
    /// negative = overshoot.
    pub relative_gap: f64,
    /// Gap severity [0, 1]: how far we are from the target normalised
    /// by the target magnitude. Higher = more severe.
    pub severity: f64,
    /// Whether the gap is significant (above min threshold).
    pub significant: bool,
    /// Direction: +1 if we need to increase, -1 if we need to decrease.
    pub direction: f64,
}

impl GapAnalysis {
    /// Perform gap analysis for a single metric.
    pub fn analyse(
        metric: impl Into<String>,
        current: f64,
        target: f64,
        min_threshold: f64,
    ) -> Self {
        let metric = metric.into();
        let absolute_gap = target - current;
        let relative_gap = if target.abs() > f64::EPSILON {
            absolute_gap / target.abs()
        } else if current.abs() > f64::EPSILON {
            absolute_gap / current.abs()
        } else {
            0.0
        };
        let severity = relative_gap.abs().min(1.0);
        let significant = absolute_gap.abs() > min_threshold;
        let direction = if absolute_gap >= 0.0 { 1.0 } else { -1.0 };

        Self {
            metric,
            current,
            target,
            absolute_gap,
            relative_gap,
            severity,
            significant,
            direction,
        }
    }

    /// Whether we are already at or past the target.
    pub fn is_achieved(&self) -> bool {
        if self.direction >= 0.0 {
            self.current >= self.target - f64::EPSILON
        } else {
            self.current <= self.target + f64::EPSILON
        }
    }
}

// ---------------------------------------------------------------------------
// Subgoal
// ---------------------------------------------------------------------------

/// A single generated subgoal with a target value and metadata.
#[derive(Debug, Clone)]
pub struct Subgoal {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Metric this subgoal targets.
    pub metric: String,
    /// Target value for this subgoal.
    pub target_value: f64,
    /// Starting value (previous subgoal's target or current value).
    pub start_value: f64,
    /// Step size (target_value - start_value).
    pub step_size: f64,
    /// Fraction of total gap this subgoal covers [0, 1].
    pub gap_fraction: f64,
    /// Cumulative fraction of gap covered when this subgoal is
    /// achieved [0, 1].
    pub cumulative_fraction: f64,
    /// Priority (lower = higher priority).
    pub priority: u32,
    /// Whether this subgoal is a milestone.
    pub is_milestone: bool,
    /// Sequence index within the generated chain (0-based).
    pub sequence_index: usize,
    /// Total number of subgoals in the chain.
    pub chain_length: usize,
    /// ID of the previous subgoal in the chain (None for first).
    pub predecessor: Option<String>,
    /// Whether this subgoal has been achieved.
    pub achieved: bool,
    /// Actual value when achieved (if any).
    pub achieved_value: Option<f64>,
}

impl Subgoal {
    /// Mark this subgoal as achieved with the given actual value.
    pub fn mark_achieved(&mut self, actual_value: f64) {
        self.achieved = true;
        self.achieved_value = Some(actual_value);
    }

    /// Whether this subgoal is the final one in its chain.
    pub fn is_final(&self) -> bool {
        self.sequence_index + 1 >= self.chain_length
    }

    /// Whether this subgoal is the first one in its chain.
    pub fn is_first(&self) -> bool {
        self.sequence_index == 0
    }

    /// How much of this step has been completed given a current value.
    /// Returns a value in [0, 1].
    pub fn step_progress(&self, current: f64) -> f64 {
        if self.step_size.abs() < f64::EPSILON {
            return 1.0; // no gap → already there
        }
        let progress = (current - self.start_value) / self.step_size;
        progress.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Generation request
// ---------------------------------------------------------------------------

/// A request to generate subgoals for a single metric.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Metric name.
    pub metric: String,
    /// Current value.
    pub current: f64,
    /// Target value.
    pub target: f64,
    /// Number of intermediate subgoals to generate.
    pub num_subgoals: Option<usize>,
    /// Interpolation strategy.
    pub strategy: InterpolationStrategy,
    /// Base priority for this metric's subgoals.
    pub base_priority: u32,
    /// Urgency factor [0, 1]: higher = more urgent.
    pub urgency: f64,
    /// Optional deadline tick (0 = no deadline).
    pub deadline_tick: u64,
}

impl GenerationRequest {
    /// Create a basic request with linear interpolation and defaults.
    pub fn new(metric: impl Into<String>, current: f64, target: f64) -> Self {
        Self {
            metric: metric.into(),
            current,
            target,
            num_subgoals: None,
            strategy: InterpolationStrategy::Linear,
            base_priority: 0,
            urgency: 0.5,
            deadline_tick: 0,
        }
    }

    /// Builder: set number of subgoals.
    pub fn with_num_subgoals(mut self, n: usize) -> Self {
        self.num_subgoals = Some(n);
        self
    }

    /// Builder: set interpolation strategy.
    pub fn with_strategy(mut self, strategy: InterpolationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Builder: set base priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.base_priority = priority;
        self
    }

    /// Builder: set urgency.
    pub fn with_urgency(mut self, urgency: f64) -> Self {
        self.urgency = urgency.clamp(0.0, 1.0);
        self
    }

    /// Builder: set deadline tick.
    pub fn with_deadline(mut self, deadline: u64) -> Self {
        self.deadline_tick = deadline;
        self
    }
}

// ---------------------------------------------------------------------------
// Generation result
// ---------------------------------------------------------------------------

/// Result of generating subgoals for one or more metrics.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated subgoals in priority order.
    pub subgoals: Vec<Subgoal>,
    /// Gap analyses for each metric.
    pub gap_analyses: Vec<GapAnalysis>,
    /// Metrics that were skipped (gap too small or already achieved).
    pub skipped_metrics: Vec<String>,
    /// Total number of subgoals generated.
    pub total_generated: usize,
    /// Number of milestone subgoals.
    pub milestone_count: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the subgoal generation engine.
#[derive(Debug, Clone)]
pub struct SubgoalStats {
    /// Total subgoals generated across all invocations.
    pub total_generated: u64,
    /// Total subgoals achieved.
    pub total_achieved: u64,
    /// Total generation invocations.
    pub generation_count: u64,
    /// Mean gap severity across all analyses.
    pub mean_gap_severity: f64,
    /// Sum of gap severities (for running mean).
    sum_severity: f64,
    /// Number of gap analyses performed.
    analysis_count: u64,
    /// Total metrics skipped (gap too small).
    pub total_skipped: u64,
    /// Total milestones generated.
    pub total_milestones: u64,
    /// Total milestones achieved.
    pub milestones_achieved: u64,
    /// Current tracked subgoals.
    pub current_tracked: usize,
}

impl Default for SubgoalStats {
    fn default() -> Self {
        Self {
            total_generated: 0,
            total_achieved: 0,
            generation_count: 0,
            mean_gap_severity: 0.0,
            sum_severity: 0.0,
            analysis_count: 0,
            total_skipped: 0,
            total_milestones: 0,
            milestones_achieved: 0,
            current_tracked: 0,
        }
    }
}

impl SubgoalStats {
    /// Achievement rate: achieved / generated.
    pub fn achievement_rate(&self) -> f64 {
        if self.total_generated == 0 {
            return 0.0;
        }
        self.total_achieved as f64 / self.total_generated as f64
    }

    /// Milestone achievement rate.
    pub fn milestone_achievement_rate(&self) -> f64 {
        if self.total_milestones == 0 {
            return 0.0;
        }
        self.milestones_achieved as f64 / self.total_milestones as f64
    }

    /// Mean subgoals per generation.
    pub fn mean_subgoals_per_generation(&self) -> f64 {
        if self.generation_count == 0 {
            return 0.0;
        }
        self.total_generated as f64 / self.generation_count as f64
    }

    fn update_severity(&mut self, severity: f64) {
        self.sum_severity += severity;
        self.analysis_count += 1;
        self.mean_gap_severity = self.sum_severity / self.analysis_count as f64;
    }
}

// ---------------------------------------------------------------------------
// SubgoalGeneration engine
// ---------------------------------------------------------------------------

/// Subgoal generation engine.
///
/// Analyse gaps between current and target states, generate intermediate
/// subgoals with various interpolation strategies, and track their
/// completion.
pub struct SubgoalGeneration {
    config: SubgoalGenerationConfig,

    /// All tracked subgoals keyed by ID.
    subgoals: HashMap<String, Subgoal>,

    /// Subgoal chains keyed by metric name → Vec of subgoal IDs.
    chains: HashMap<String, Vec<String>>,

    /// Running ID counter for generating unique IDs.
    next_id: u64,

    /// Current tick.
    current_tick: u64,

    /// Running statistics.
    stats: SubgoalStats,
}

impl Default for SubgoalGeneration {
    fn default() -> Self {
        Self::new()
    }
}

impl SubgoalGeneration {
    /// Create a new engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(SubgoalGenerationConfig::default()).unwrap()
    }

    /// Create with explicit configuration.
    pub fn with_config(config: SubgoalGenerationConfig) -> Result<Self> {
        if config.default_num_subgoals == 0 {
            return Err(Error::InvalidInput(
                "default_num_subgoals must be > 0".into(),
            ));
        }
        if config.max_subgoals_per_batch == 0 {
            return Err(Error::InvalidInput(
                "max_subgoals_per_batch must be > 0".into(),
            ));
        }
        if config.max_total_subgoals == 0 {
            return Err(Error::InvalidInput("max_total_subgoals must be > 0".into()));
        }
        if config.max_step_fraction <= 0.0 || config.max_step_fraction > 1.0 {
            return Err(Error::InvalidInput(
                "max_step_fraction must be in (0, 1]".into(),
            ));
        }
        Ok(Self {
            config,
            subgoals: HashMap::new(),
            chains: HashMap::new(),
            next_id: 0,
            current_tick: 0,
            stats: SubgoalStats::default(),
        })
    }

    fn generate_id(&mut self) -> String {
        let id = format!("sg_{}", self.next_id);
        self.next_id += 1;
        id
    }

    // -----------------------------------------------------------------------
    // Gap analysis
    // -----------------------------------------------------------------------

    /// Perform gap analysis for a single metric.
    pub fn analyse_gap(&mut self, metric: &str, current: f64, target: f64) -> GapAnalysis {
        let analysis = GapAnalysis::analyse(metric, current, target, self.config.min_gap_threshold);
        self.stats.update_severity(analysis.severity);
        analysis
    }

    /// Perform gap analysis for multiple metrics.
    pub fn analyse_gaps(&mut self, metrics: &[(&str, f64, f64)]) -> Vec<GapAnalysis> {
        metrics
            .iter()
            .map(|(metric, current, target)| self.analyse_gap(metric, *current, *target))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Interpolation
    // -----------------------------------------------------------------------

    /// Generate intermediate target values between start and end.
    pub fn interpolate(
        &self,
        start: f64,
        end: f64,
        n: usize,
        strategy: InterpolationStrategy,
    ) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }

        let mut targets = Vec::with_capacity(n);

        match strategy {
            InterpolationStrategy::Linear => {
                for i in 1..=n {
                    let t = i as f64 / (n + 1) as f64;
                    targets.push(start + t * (end - start));
                }
            }
            InterpolationStrategy::Exponential => {
                // Use exponential spacing: targets grow larger toward the end
                let base = if start.abs() > f64::EPSILON {
                    (end / start).abs()
                } else {
                    (end - start).abs() + 1.0
                };

                if start.abs() > f64::EPSILON && start.signum() == end.signum() && base > 0.0 {
                    let log_base = base.ln();
                    for i in 1..=n {
                        let t = i as f64 / (n + 1) as f64;
                        let factor = (t * log_base).exp();
                        targets.push(start * factor);
                    }
                } else {
                    // Fall back to linear if exponential doesn't make sense
                    for i in 1..=n {
                        let t = i as f64 / (n + 1) as f64;
                        targets.push(start + t * (end - start));
                    }
                }
            }
            InterpolationStrategy::FrontLoaded => {
                // Larger steps early: use square root spacing
                for i in 1..=n {
                    let t = (i as f64 / (n + 1) as f64).sqrt();
                    targets.push(start + t * (end - start));
                }
            }
            InterpolationStrategy::BackLoaded => {
                // Smaller steps early: use square spacing
                for i in 1..=n {
                    let t_raw = i as f64 / (n + 1) as f64;
                    let t = t_raw * t_raw;
                    targets.push(start + t * (end - start));
                }
            }
        }

        targets
    }

    // -----------------------------------------------------------------------
    // Subgoal generation
    // -----------------------------------------------------------------------

    /// Generate subgoals for a single metric.
    pub fn generate(&mut self, request: &GenerationRequest) -> GenerationResult {
        self.generate_batch(std::slice::from_ref(request))
    }

    /// Generate subgoals for multiple metrics simultaneously.
    pub fn generate_batch(&mut self, requests: &[GenerationRequest]) -> GenerationResult {
        let mut all_subgoals = Vec::new();
        let mut gap_analyses = Vec::new();
        let mut skipped_metrics = Vec::new();
        let mut milestone_count = 0usize;

        for request in requests {
            let gap = self.analyse_gap(&request.metric, request.current, request.target);
            gap_analyses.push(gap.clone());

            if !gap.significant || gap.is_achieved() {
                skipped_metrics.push(request.metric.clone());
                self.stats.total_skipped += 1;
                continue;
            }

            let n = request
                .num_subgoals
                .unwrap_or(self.config.default_num_subgoals)
                .min(self.config.max_subgoals_per_batch);

            if n == 0 {
                skipped_metrics.push(request.metric.clone());
                continue;
            }

            // Check capacity
            let remaining_capacity = self
                .config
                .max_total_subgoals
                .saturating_sub(self.subgoals.len());
            let actual_n = n.min(remaining_capacity);

            if actual_n == 0 {
                skipped_metrics.push(request.metric.clone());
                continue;
            }

            // Generate intermediate targets
            let targets =
                self.interpolate(request.current, request.target, actual_n, request.strategy);

            // Apply max step size constraint
            let total_gap = gap.absolute_gap.abs();
            let max_step = total_gap * self.config.max_step_fraction;

            let mut chain_ids = Vec::new();
            let mut prev_value = request.current;
            let mut prev_id: Option<String> = None;

            for (i, &target_value) in targets.iter().enumerate() {
                let mut step_size = target_value - prev_value;

                // Enforce max step size
                if step_size.abs() > max_step && max_step > 0.0 {
                    // Clamp to max step
                    let clamped_target = prev_value + step_size.signum() * max_step;
                    step_size = clamped_target - prev_value;
                    // We still use the original target for the subgoal
                    // but note the constraint
                }

                let cumulative_fraction = if total_gap > 0.0 {
                    ((target_value - request.current).abs() / total_gap).min(1.0)
                } else {
                    1.0
                };

                let gap_fraction = if total_gap > 0.0 {
                    (step_size.abs() / total_gap).min(1.0)
                } else {
                    0.0
                };

                // Check if this is a milestone
                let is_milestone = self.config.milestone_fractions.iter().any(|&mf| {
                    (cumulative_fraction - mf).abs() < 1.0 / (2.0 * actual_n as f64 + 1.0)
                });

                // Compute priority
                let gap_priority = (gap.severity * self.config.gap_weight * 100.0) as u32;
                let urgency_priority =
                    (request.urgency * self.config.urgency_weight * 100.0) as u32;
                let sequence_priority = i as u32;
                let priority =
                    request.base_priority + gap_priority + urgency_priority + sequence_priority;

                let id = self.generate_id();

                let subgoal = Subgoal {
                    id: id.clone(),
                    name: format!(
                        "{} → {:.4} ({:.0}%)",
                        request.metric,
                        target_value,
                        cumulative_fraction * 100.0
                    ),
                    metric: request.metric.clone(),
                    target_value,
                    start_value: prev_value,
                    step_size,
                    gap_fraction,
                    cumulative_fraction,
                    priority,
                    is_milestone,
                    sequence_index: i,
                    chain_length: actual_n,
                    predecessor: prev_id.clone(),
                    achieved: false,
                    achieved_value: None,
                };

                if is_milestone {
                    milestone_count += 1;
                }

                chain_ids.push(id.clone());
                self.subgoals.insert(id.clone(), subgoal.clone());
                all_subgoals.push(subgoal);

                prev_value = target_value;
                prev_id = Some(id);
            }

            self.chains.insert(request.metric.clone(), chain_ids);
        }

        // Sort by priority
        all_subgoals.sort_by(|a, b| a.priority.cmp(&b.priority));

        let total_generated = all_subgoals.len();

        // Update stats
        self.stats.total_generated += total_generated as u64;
        self.stats.generation_count += 1;
        self.stats.total_milestones += milestone_count as u64;
        self.stats.current_tracked = self.subgoals.len();

        GenerationResult {
            subgoals: all_subgoals,
            gap_analyses,
            skipped_metrics,
            total_generated,
            milestone_count,
        }
    }

    // -----------------------------------------------------------------------
    // Subgoal management
    // -----------------------------------------------------------------------

    /// Get a subgoal by ID.
    pub fn subgoal(&self, id: &str) -> Option<&Subgoal> {
        self.subgoals.get(id)
    }

    /// Get a mutable reference to a subgoal.
    pub fn subgoal_mut(&mut self, id: &str) -> Option<&mut Subgoal> {
        self.subgoals.get_mut(id)
    }

    /// Number of tracked subgoals.
    pub fn subgoal_count(&self) -> usize {
        self.subgoals.len()
    }

    /// Get the chain of subgoal IDs for a metric.
    pub fn chain(&self, metric: &str) -> Option<&[String]> {
        self.chains.get(metric).map(|v| v.as_slice())
    }

    /// Get all tracked metrics.
    pub fn tracked_metrics(&self) -> Vec<String> {
        let mut metrics: Vec<_> = self.chains.keys().cloned().collect();
        metrics.sort();
        metrics
    }

    /// Mark a subgoal as achieved.
    pub fn achieve(&mut self, id: &str, actual_value: f64) -> Result<()> {
        let subgoal = self
            .subgoals
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("subgoal '{}' not found", id)))?;
        if subgoal.achieved {
            return Err(Error::InvalidState(format!(
                "subgoal '{}' is already achieved",
                id
            )));
        }
        subgoal.mark_achieved(actual_value);
        self.stats.total_achieved += 1;
        if subgoal.is_milestone {
            self.stats.milestones_achieved += 1;
        }
        Ok(())
    }

    /// Get the next unachieved subgoal in a metric's chain.
    pub fn next_subgoal(&self, metric: &str) -> Option<&Subgoal> {
        self.chains.get(metric).and_then(|chain| {
            chain
                .iter()
                .find_map(|id| self.subgoals.get(id).filter(|sg| !sg.achieved))
        })
    }

    /// Get all unachieved subgoals across all metrics, sorted by priority.
    pub fn pending_subgoals(&self) -> Vec<&Subgoal> {
        let mut pending: Vec<_> = self.subgoals.values().filter(|sg| !sg.achieved).collect();
        pending.sort_by(|a, b| a.priority.cmp(&b.priority));
        pending
    }

    /// Get all achieved subgoals for a metric.
    pub fn achieved_subgoals(&self, metric: &str) -> Vec<&Subgoal> {
        self.chains
            .get(metric)
            .map(|chain| {
                chain
                    .iter()
                    .filter_map(|id| self.subgoals.get(id).filter(|sg| sg.achieved))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the overall progress for a metric's chain [0, 1].
    pub fn chain_progress(&self, metric: &str) -> f64 {
        match self.chains.get(metric) {
            Some(chain) if !chain.is_empty() => {
                let achieved = chain
                    .iter()
                    .filter(|id| {
                        self.subgoals
                            .get(id.as_str())
                            .map_or(false, |sg| sg.achieved)
                    })
                    .count();
                achieved as f64 / chain.len() as f64
            }
            _ => 0.0,
        }
    }

    /// Remove all subgoals for a metric.
    pub fn clear_metric(&mut self, metric: &str) {
        if let Some(chain) = self.chains.remove(metric) {
            for id in &chain {
                self.subgoals.remove(id);
            }
        }
        self.stats.current_tracked = self.subgoals.len();
    }

    /// Remove all achieved subgoals (cleanup).
    pub fn prune_achieved(&mut self) {
        let achieved_ids: Vec<String> = self
            .subgoals
            .iter()
            .filter(|(_, sg)| sg.achieved)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &achieved_ids {
            self.subgoals.remove(id);
        }

        // Also clean up chains
        for chain in self.chains.values_mut() {
            chain.retain(|id| !achieved_ids.contains(id));
        }

        // Remove empty chains
        self.chains.retain(|_, chain| !chain.is_empty());
        self.stats.current_tracked = self.subgoals.len();
    }

    // -----------------------------------------------------------------------
    // Tick
    // -----------------------------------------------------------------------

    /// Advance the tick counter.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Running statistics.
    pub fn stats(&self) -> &SubgoalStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all state.
    pub fn reset(&mut self) {
        self.subgoals.clear();
        self.chains.clear();
        self.next_id = 0;
        self.current_tick = 0;
        self.stats = SubgoalStats::default();
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
    // InterpolationStrategy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_strategy_label() {
        assert_eq!(InterpolationStrategy::Linear.label(), "Linear");
        assert_eq!(InterpolationStrategy::Exponential.label(), "Exponential");
        assert_eq!(InterpolationStrategy::FrontLoaded.label(), "FrontLoaded");
        assert_eq!(InterpolationStrategy::BackLoaded.label(), "BackLoaded");
    }

    // -----------------------------------------------------------------------
    // GapAnalysis tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gap_analysis_positive() {
        let gap = GapAnalysis::analyse("pnl", 100.0, 200.0, 1e-6);
        assert_eq!(gap.metric, "pnl");
        assert!((gap.absolute_gap - 100.0).abs() < 1e-10);
        assert!((gap.relative_gap - 0.5).abs() < 1e-10);
        assert!(gap.direction > 0.0);
        assert!(gap.significant);
        assert!(!gap.is_achieved());
    }

    #[test]
    fn test_gap_analysis_negative() {
        let gap = GapAnalysis::analyse("drawdown", 0.10, 0.05, 1e-6);
        assert!((gap.absolute_gap - (-0.05)).abs() < 1e-10);
        assert!(gap.direction < 0.0);
        assert!(gap.significant);
        assert!(!gap.is_achieved());
    }

    #[test]
    fn test_gap_analysis_achieved() {
        let gap = GapAnalysis::analyse("target", 200.0, 100.0, 1e-6);
        // current >= target (for a target we need to increase to)
        // Actually, current=200 > target=100, gap is negative, direction=-1
        // is_achieved: direction < 0, current <= target? No, 200 > 100, not achieved
        // Wait, let me re-read: absolute_gap = target - current = 100 - 200 = -100
        // direction = -1 (we need to decrease)
        // is_achieved: direction < 0 → current <= target + eps → 200 <= 100? No
        assert!(!gap.is_achieved());

        // But if current is below or at target for decrease:
        let gap2 = GapAnalysis::analyse("drawdown", 0.03, 0.05, 1e-6);
        // absolute_gap = 0.05 - 0.03 = 0.02, direction = +1
        // is_achieved: direction >= 0 → current >= target - eps → 0.03 >= 0.05? No
        assert!(!gap2.is_achieved());

        // Actually achieved:
        let gap3 = GapAnalysis::analyse("pnl", 200.0, 200.0, 1e-6);
        assert!(gap3.is_achieved());
    }

    #[test]
    fn test_gap_analysis_zero_target() {
        let gap = GapAnalysis::analyse("metric", 0.5, 0.0, 1e-6);
        assert!((gap.absolute_gap - (-0.5)).abs() < 1e-10);
        // relative_gap uses current as denominator when target is 0
        assert!((gap.relative_gap - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gap_analysis_both_zero() {
        let gap = GapAnalysis::analyse("metric", 0.0, 0.0, 1e-6);
        assert!((gap.absolute_gap - 0.0).abs() < 1e-10);
        assert!((gap.relative_gap - 0.0).abs() < 1e-10);
        assert!(!gap.significant);
        assert!(gap.is_achieved());
    }

    #[test]
    fn test_gap_analysis_insignificant() {
        let gap = GapAnalysis::analyse("metric", 1.0, 1.0 + 1e-8, 1e-6);
        assert!(!gap.significant);
    }

    #[test]
    fn test_gap_severity_capped() {
        let gap = GapAnalysis::analyse("metric", 0.0, 100.0, 1e-6);
        assert!(gap.severity <= 1.0);
    }

    // -----------------------------------------------------------------------
    // Subgoal tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_subgoal_mark_achieved() {
        let mut sg = Subgoal {
            id: "sg_0".into(),
            name: "test".into(),
            metric: "pnl".into(),
            target_value: 150.0,
            start_value: 100.0,
            step_size: 50.0,
            gap_fraction: 0.5,
            cumulative_fraction: 0.5,
            priority: 0,
            is_milestone: false,
            sequence_index: 0,
            chain_length: 2,
            predecessor: None,
            achieved: false,
            achieved_value: None,
        };

        sg.mark_achieved(148.0);
        assert!(sg.achieved);
        assert!((sg.achieved_value.unwrap() - 148.0).abs() < 1e-10);
    }

    #[test]
    fn test_subgoal_is_first_final() {
        let sg = Subgoal {
            id: "sg_0".into(),
            name: "test".into(),
            metric: "pnl".into(),
            target_value: 150.0,
            start_value: 100.0,
            step_size: 50.0,
            gap_fraction: 0.5,
            cumulative_fraction: 0.5,
            priority: 0,
            is_milestone: false,
            sequence_index: 0,
            chain_length: 2,
            predecessor: None,
            achieved: false,
            achieved_value: None,
        };

        assert!(sg.is_first());
        assert!(!sg.is_final());

        let sg_last = Subgoal {
            sequence_index: 1,
            ..sg
        };
        assert!(!sg_last.is_first());
        assert!(sg_last.is_final());
    }

    #[test]
    fn test_subgoal_step_progress() {
        let sg = Subgoal {
            id: "sg_0".into(),
            name: "test".into(),
            metric: "pnl".into(),
            target_value: 200.0,
            start_value: 100.0,
            step_size: 100.0,
            gap_fraction: 1.0,
            cumulative_fraction: 1.0,
            priority: 0,
            is_milestone: false,
            sequence_index: 0,
            chain_length: 1,
            predecessor: None,
            achieved: false,
            achieved_value: None,
        };

        assert!((sg.step_progress(100.0) - 0.0).abs() < 1e-10);
        assert!((sg.step_progress(150.0) - 0.5).abs() < 1e-10);
        assert!((sg.step_progress(200.0) - 1.0).abs() < 1e-10);
        assert!((sg.step_progress(250.0) - 1.0).abs() < 1e-10); // clamped
        assert!((sg.step_progress(50.0) - 0.0).abs() < 1e-10); // clamped
    }

    #[test]
    fn test_subgoal_step_progress_zero_step() {
        let sg = Subgoal {
            id: "sg_0".into(),
            name: "test".into(),
            metric: "pnl".into(),
            target_value: 100.0,
            start_value: 100.0,
            step_size: 0.0,
            gap_fraction: 0.0,
            cumulative_fraction: 1.0,
            priority: 0,
            is_milestone: false,
            sequence_index: 0,
            chain_length: 1,
            predecessor: None,
            achieved: false,
            achieved_value: None,
        };

        assert!((sg.step_progress(100.0) - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // GenerationRequest tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generation_request_new() {
        let req = GenerationRequest::new("pnl", 100.0, 200.0);
        assert_eq!(req.metric, "pnl");
        assert!((req.current - 100.0).abs() < 1e-10);
        assert!((req.target - 200.0).abs() < 1e-10);
        assert_eq!(req.strategy, InterpolationStrategy::Linear);
        assert!(req.num_subgoals.is_none());
    }

    #[test]
    fn test_generation_request_builders() {
        let req = GenerationRequest::new("pnl", 100.0, 200.0)
            .with_num_subgoals(3)
            .with_strategy(InterpolationStrategy::Exponential)
            .with_priority(5)
            .with_urgency(0.8)
            .with_deadline(100);

        assert_eq!(req.num_subgoals, Some(3));
        assert_eq!(req.strategy, InterpolationStrategy::Exponential);
        assert_eq!(req.base_priority, 5);
        assert!((req.urgency - 0.8).abs() < 1e-10);
        assert_eq!(req.deadline_tick, 100);
    }

    #[test]
    fn test_generation_request_urgency_clamped() {
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_urgency(1.5);
        assert!((req.urgency - 1.0).abs() < 1e-10);

        let req2 = GenerationRequest::new("pnl", 100.0, 200.0).with_urgency(-0.5);
        assert!((req2.urgency - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Configuration validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_config_num_subgoals() {
        let mut cfg = SubgoalGenerationConfig::default();
        cfg.default_num_subgoals = 0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_per_batch() {
        let mut cfg = SubgoalGenerationConfig::default();
        cfg.max_subgoals_per_batch = 0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_max_total() {
        let mut cfg = SubgoalGenerationConfig::default();
        cfg.max_total_subgoals = 0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_step_fraction_zero() {
        let mut cfg = SubgoalGenerationConfig::default();
        cfg.max_step_fraction = 0.0;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_step_fraction_over_one() {
        let mut cfg = SubgoalGenerationConfig::default();
        cfg.max_step_fraction = 1.5;
        assert!(SubgoalGeneration::with_config(cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_default() {
        let sg = SubgoalGeneration::new();
        assert_eq!(sg.subgoal_count(), 0);
        assert_eq!(sg.current_tick(), 0);
    }

    // -----------------------------------------------------------------------
    // Interpolation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_linear() {
        let sg = SubgoalGeneration::new();
        let targets = sg.interpolate(0.0, 100.0, 3, InterpolationStrategy::Linear);
        assert_eq!(targets.len(), 3);
        assert!((targets[0] - 25.0).abs() < 1e-10);
        assert!((targets[1] - 50.0).abs() < 1e-10);
        assert!((targets[2] - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_linear_single() {
        let sg = SubgoalGeneration::new();
        let targets = sg.interpolate(0.0, 100.0, 1, InterpolationStrategy::Linear);
        assert_eq!(targets.len(), 1);
        assert!((targets[0] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_linear_negative_direction() {
        let sg = SubgoalGeneration::new();
        let targets = sg.interpolate(100.0, 0.0, 3, InterpolationStrategy::Linear);
        assert_eq!(targets.len(), 3);
        assert!((targets[0] - 75.0).abs() < 1e-10);
        assert!((targets[1] - 50.0).abs() < 1e-10);
        assert!((targets[2] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_zero_count() {
        let sg = SubgoalGeneration::new();
        let targets = sg.interpolate(0.0, 100.0, 0, InterpolationStrategy::Linear);
        assert!(targets.is_empty());
    }

    #[test]
    fn test_interpolate_front_loaded() {
        let sg = SubgoalGeneration::new();
        let targets = sg.interpolate(0.0, 100.0, 3, InterpolationStrategy::FrontLoaded);
        assert_eq!(targets.len(), 3);
        // Front-loaded: first step should be larger than linear (25)
        assert!(targets[0] > 25.0);
        // Each subsequent step should be smaller
        let _steps: Vec<f64> = targets.windows(2).map(|w| w[1] - w[0]).collect();
        // Actually the step from 0 to targets[0] should be bigger
        let first_step = targets[0];
        let last_step = 100.0 - targets[2]; // remaining gap
        assert!(first_step > last_step);
    }

    #[test]
    fn test_interpolate_back_loaded() {
        let sg = SubgoalGeneration::new();
        let targets = sg.interpolate(0.0, 100.0, 3, InterpolationStrategy::BackLoaded);
        assert_eq!(targets.len(), 3);
        // Back-loaded: first step should be smaller than linear (25)
        assert!(targets[0] < 25.0);
    }

    #[test]
    fn test_interpolate_exponential() {
        let sg = SubgoalGeneration::new();
        let targets = sg.interpolate(100.0, 200.0, 3, InterpolationStrategy::Exponential);
        assert_eq!(targets.len(), 3);
        // All targets should be between 100 and 200
        for t in &targets {
            assert!(*t > 100.0);
            assert!(*t < 200.0);
        }
        // Should be monotonically increasing
        for w in targets.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn test_interpolate_exponential_zero_start() {
        let sg = SubgoalGeneration::new();
        // Should fall back to linear when start is 0
        let targets = sg.interpolate(0.0, 100.0, 3, InterpolationStrategy::Exponential);
        assert_eq!(targets.len(), 3);
        // Should still be monotonically increasing
        for w in targets.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    // -----------------------------------------------------------------------
    // Gap analysis via engine
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyse_gap() {
        let mut sg = SubgoalGeneration::new();
        let gap = sg.analyse_gap("pnl", 100.0, 200.0);
        assert_eq!(gap.metric, "pnl");
        assert!((gap.absolute_gap - 100.0).abs() < 1e-10);
        assert!(gap.significant);
    }

    #[test]
    fn test_analyse_gaps_multiple() {
        let mut sg = SubgoalGeneration::new();
        let gaps = sg.analyse_gaps(&[("pnl", 100.0, 200.0), ("dd", 0.1, 0.05)]);
        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0].metric, "pnl");
        assert_eq!(gaps[1].metric, "dd");
    }

    // -----------------------------------------------------------------------
    // Subgoal generation — single metric
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_single() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        let result = sg.generate(&req);

        assert_eq!(result.total_generated, 3);
        assert_eq!(result.subgoals.len(), 3);
        assert!(result.skipped_metrics.is_empty());
        assert_eq!(result.gap_analyses.len(), 1);

        // All subgoals should target "pnl"
        for sub in &result.subgoals {
            assert_eq!(sub.metric, "pnl");
            assert!(!sub.achieved);
        }

        // First subgoal should be first in chain
        let first = result
            .subgoals
            .iter()
            .min_by_key(|s| s.sequence_index)
            .unwrap();
        assert!(first.is_first());
        assert!(first.predecessor.is_none());

        // Subgoals should be registered
        assert_eq!(sg.subgoal_count(), 3);
    }

    #[test]
    fn test_generate_creates_chain() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        sg.generate(&req);

        let chain = sg.chain("pnl").unwrap();
        assert_eq!(chain.len(), 3);

        // Each subgoal (except first) should reference predecessor
        for (i, id) in chain.iter().enumerate() {
            let sub = sg.subgoal(id).unwrap();
            assert_eq!(sub.sequence_index, i);
            assert_eq!(sub.chain_length, 3);
            if i == 0 {
                assert!(sub.predecessor.is_none());
            } else {
                assert_eq!(sub.predecessor.as_deref(), Some(chain[i - 1].as_str()));
            }
        }
    }

    #[test]
    fn test_generate_targets_increase() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(4);
        let result = sg.generate(&req);

        let mut targets: Vec<f64> = result.subgoals.iter().map(|s| s.target_value).collect();
        targets.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // All targets should be between 100 and 200
        for t in &targets {
            assert!(*t > 100.0);
            assert!(*t < 200.0);
        }
    }

    #[test]
    fn test_generate_skips_achieved() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 200.0, 200.0).with_num_subgoals(3);
        let result = sg.generate(&req);

        assert_eq!(result.total_generated, 0);
        assert_eq!(result.skipped_metrics.len(), 1);
        assert_eq!(result.skipped_metrics[0], "pnl");
    }

    #[test]
    fn test_generate_skips_tiny_gap() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 100.0 + 1e-8).with_num_subgoals(3);
        let result = sg.generate(&req);

        assert_eq!(result.total_generated, 0);
        assert_eq!(result.skipped_metrics.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Batch generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_batch() {
        let mut sg = SubgoalGeneration::new();
        let requests = vec![
            GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(2),
            GenerationRequest::new("sharpe", 0.5, 1.5).with_num_subgoals(2),
        ];
        let result = sg.generate_batch(&requests);

        assert_eq!(result.total_generated, 4);
        assert_eq!(result.gap_analyses.len(), 2);
        assert_eq!(sg.subgoal_count(), 4);

        // Should have chains for both metrics
        assert!(sg.chain("pnl").is_some());
        assert!(sg.chain("sharpe").is_some());
    }

    #[test]
    fn test_generate_batch_mixed_skip() {
        let mut sg = SubgoalGeneration::new();
        let requests = vec![
            GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(2),
            GenerationRequest::new("achieved", 100.0, 100.0).with_num_subgoals(2),
        ];
        let result = sg.generate_batch(&requests);

        assert_eq!(result.total_generated, 2);
        assert_eq!(result.skipped_metrics.len(), 1);
        assert_eq!(result.skipped_metrics[0], "achieved");
    }

    // -----------------------------------------------------------------------
    // Milestones
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_milestones() {
        let mut sg = SubgoalGeneration::new();
        // With default milestone fractions [0.25, 0.5, 0.75] and 7 subgoals,
        // some should be flagged as milestones
        let req = GenerationRequest::new("pnl", 0.0, 100.0).with_num_subgoals(7);
        let result = sg.generate(&req);

        assert!(result.milestone_count > 0);
        let milestone_subs: Vec<_> = result.subgoals.iter().filter(|s| s.is_milestone).collect();
        assert!(!milestone_subs.is_empty());
    }

    // -----------------------------------------------------------------------
    // Achievement
    // -----------------------------------------------------------------------

    #[test]
    fn test_achieve_subgoal() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(2);
        let result = sg.generate(&req);

        let first_id = &result.subgoals[0].id;
        sg.achieve(first_id, 125.0).unwrap();

        let sub = sg.subgoal(first_id).unwrap();
        assert!(sub.achieved);
        assert!((sub.achieved_value.unwrap() - 125.0).abs() < 1e-10);
    }

    #[test]
    fn test_achieve_already_achieved() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(1);
        let result = sg.generate(&req);

        let id = &result.subgoals[0].id;
        sg.achieve(id, 150.0).unwrap();
        assert!(sg.achieve(id, 160.0).is_err());
    }

    #[test]
    fn test_achieve_nonexistent() {
        let mut sg = SubgoalGeneration::new();
        assert!(sg.achieve("nope", 100.0).is_err());
    }

    // -----------------------------------------------------------------------
    // next_subgoal
    // -----------------------------------------------------------------------

    #[test]
    fn test_next_subgoal() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        let result = sg.generate(&req);

        // First unachieved should be sequence 0
        let next = sg.next_subgoal("pnl").unwrap();
        assert_eq!(next.sequence_index, 0);

        // Achieve first, next should be sequence 1
        sg.achieve(&result.subgoals[0].id, 125.0).unwrap();
        let next = sg.next_subgoal("pnl").unwrap();
        assert_eq!(next.sequence_index, 1);
    }

    #[test]
    fn test_next_subgoal_all_achieved() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(1);
        let result = sg.generate(&req);

        sg.achieve(&result.subgoals[0].id, 150.0).unwrap();
        assert!(sg.next_subgoal("pnl").is_none());
    }

    #[test]
    fn test_next_subgoal_no_metric() {
        let sg = SubgoalGeneration::new();
        assert!(sg.next_subgoal("nope").is_none());
    }

    // -----------------------------------------------------------------------
    // pending_subgoals
    // -----------------------------------------------------------------------

    #[test]
    fn test_pending_subgoals() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        let result = sg.generate(&req);

        assert_eq!(sg.pending_subgoals().len(), 3);

        sg.achieve(&result.subgoals[0].id, 125.0).unwrap();
        assert_eq!(sg.pending_subgoals().len(), 2);
    }

    #[test]
    fn test_pending_subgoals_sorted_by_priority() {
        let mut sg = SubgoalGeneration::new();
        let requests = vec![
            GenerationRequest::new("pnl", 100.0, 200.0)
                .with_num_subgoals(1)
                .with_priority(10),
            GenerationRequest::new("sharpe", 0.5, 1.5)
                .with_num_subgoals(1)
                .with_priority(1),
        ];
        sg.generate_batch(&requests);

        let pending = sg.pending_subgoals();
        assert_eq!(pending.len(), 2);
        // Lower priority number = higher priority, should come first
        assert!(pending[0].priority <= pending[1].priority);
    }

    // -----------------------------------------------------------------------
    // achieved_subgoals
    // -----------------------------------------------------------------------

    #[test]
    fn test_achieved_subgoals() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        let result = sg.generate(&req);

        assert!(sg.achieved_subgoals("pnl").is_empty());

        sg.achieve(&result.subgoals[0].id, 125.0).unwrap();
        assert_eq!(sg.achieved_subgoals("pnl").len(), 1);
    }

    #[test]
    fn test_achieved_subgoals_no_metric() {
        let sg = SubgoalGeneration::new();
        assert!(sg.achieved_subgoals("nope").is_empty());
    }

    // -----------------------------------------------------------------------
    // chain_progress
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_progress() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(4);
        let result = sg.generate(&req);

        assert!((sg.chain_progress("pnl") - 0.0).abs() < 1e-10);

        sg.achieve(&result.subgoals[0].id, 125.0).unwrap();
        assert!((sg.chain_progress("pnl") - 0.25).abs() < 1e-10);

        sg.achieve(&result.subgoals[1].id, 150.0).unwrap();
        assert!((sg.chain_progress("pnl") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_chain_progress_no_metric() {
        let sg = SubgoalGeneration::new();
        assert!((sg.chain_progress("nope") - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // clear_metric
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_metric() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        sg.generate(&req);

        sg.clear_metric("pnl");
        assert_eq!(sg.subgoal_count(), 0);
        assert!(sg.chain("pnl").is_none());
    }

    #[test]
    fn test_clear_metric_nonexistent() {
        let mut sg = SubgoalGeneration::new();
        sg.clear_metric("nope"); // should not panic
    }

    // -----------------------------------------------------------------------
    // prune_achieved
    // -----------------------------------------------------------------------

    #[test]
    fn test_prune_achieved() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        let result = sg.generate(&req);

        sg.achieve(&result.subgoals[0].id, 125.0).unwrap();
        sg.prune_achieved();

        assert_eq!(sg.subgoal_count(), 2);
        // Chain should only have 2 remaining IDs
        assert_eq!(sg.chain("pnl").unwrap().len(), 2);
    }

    #[test]
    fn test_prune_achieved_removes_empty_chains() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(1);
        let result = sg.generate(&req);

        sg.achieve(&result.subgoals[0].id, 150.0).unwrap();
        sg.prune_achieved();

        assert_eq!(sg.subgoal_count(), 0);
        assert!(sg.chain("pnl").is_none());
    }

    // -----------------------------------------------------------------------
    // tracked_metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_tracked_metrics() {
        let mut sg = SubgoalGeneration::new();
        let requests = vec![
            GenerationRequest::new("z_metric", 0.0, 1.0).with_num_subgoals(1),
            GenerationRequest::new("a_metric", 0.0, 1.0).with_num_subgoals(1),
        ];
        sg.generate_batch(&requests);

        let metrics = sg.tracked_metrics();
        assert_eq!(metrics, vec!["a_metric", "z_metric"]);
    }

    // -----------------------------------------------------------------------
    // Tick
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick() {
        let mut sg = SubgoalGeneration::new();
        sg.tick();
        sg.tick();
        assert_eq!(sg.current_tick(), 2);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let sg = SubgoalGeneration::new();
        let s = sg.stats();
        assert_eq!(s.total_generated, 0);
        assert_eq!(s.total_achieved, 0);
        assert!((s.achievement_rate() - 0.0).abs() < 1e-10);
        assert!((s.milestone_achievement_rate() - 0.0).abs() < 1e-10);
        assert!((s.mean_subgoals_per_generation() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_after_generation() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        sg.generate(&req);

        let s = sg.stats();
        assert_eq!(s.total_generated, 3);
        assert_eq!(s.generation_count, 1);
        assert!((s.mean_subgoals_per_generation() - 3.0).abs() < 1e-10);
        assert_eq!(s.current_tracked, 3);
    }

    #[test]
    fn test_stats_after_achievement() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(2);
        let result = sg.generate(&req);

        sg.achieve(&result.subgoals[0].id, 125.0).unwrap();

        let s = sg.stats();
        assert_eq!(s.total_achieved, 1);
        assert!((s.achievement_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_gap_severity() {
        let mut sg = SubgoalGeneration::new();
        sg.analyse_gap("pnl", 100.0, 200.0);

        let s = sg.stats();
        assert!(s.mean_gap_severity > 0.0);
    }

    #[test]
    fn test_stats_skipped() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 100.0).with_num_subgoals(3);
        sg.generate(&req);

        assert_eq!(sg.stats().total_skipped, 1);
    }

    #[test]
    fn test_stats_milestone_achievement() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 0.0, 100.0).with_num_subgoals(7);
        let result = sg.generate(&req);

        let milestones: Vec<_> = result.subgoals.iter().filter(|s| s.is_milestone).collect();

        if !milestones.is_empty() {
            sg.achieve(&milestones[0].id, milestones[0].target_value)
                .unwrap();
            assert_eq!(sg.stats().milestones_achieved, 1);
            assert!(sg.stats().milestone_achievement_rate() > 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Capacity limits
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_total_subgoals() {
        let mut cfg = SubgoalGenerationConfig::default();
        cfg.max_total_subgoals = 5;
        let mut sg = SubgoalGeneration::with_config(cfg).unwrap();

        let req = GenerationRequest::new("pnl", 0.0, 100.0).with_num_subgoals(10);
        let result = sg.generate(&req);

        assert!(result.total_generated <= 5);
        assert!(sg.subgoal_count() <= 5);
    }

    #[test]
    fn test_max_subgoals_per_batch() {
        let mut cfg = SubgoalGenerationConfig::default();
        cfg.max_subgoals_per_batch = 3;
        let mut sg = SubgoalGeneration::with_config(cfg).unwrap();

        let req = GenerationRequest::new("pnl", 0.0, 100.0).with_num_subgoals(10);
        let result = sg.generate(&req);

        assert!(result.total_generated <= 3);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(3);
        sg.generate(&req);
        sg.tick();

        sg.reset();

        assert_eq!(sg.subgoal_count(), 0);
        assert_eq!(sg.current_tick(), 0);
        assert_eq!(sg.stats().total_generated, 0);
        assert!(sg.tracked_metrics().is_empty());
    }

    // -----------------------------------------------------------------------
    // Process compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_process() {
        let sg = SubgoalGeneration::new();
        assert!(sg.process().is_ok());
    }

    // -----------------------------------------------------------------------
    // subgoal_mut
    // -----------------------------------------------------------------------

    #[test]
    fn test_subgoal_mut() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(1);
        let result = sg.generate(&req);

        let id = &result.subgoals[0].id;
        let sub = sg.subgoal_mut(id).unwrap();
        sub.priority = 999;

        assert_eq!(sg.subgoal(id).unwrap().priority, 999);
    }

    #[test]
    fn test_subgoal_mut_nonexistent() {
        let mut sg = SubgoalGeneration::new();
        assert!(sg.subgoal_mut("nope").is_none());
    }

    // -----------------------------------------------------------------------
    // Edge: decreasing target
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_decreasing_target() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("drawdown", 0.10, 0.02).with_num_subgoals(3);
        let result = sg.generate(&req);

        assert_eq!(result.total_generated, 3);

        // Targets should be decreasing
        let mut prev_target = 0.10;
        for sub in &result.subgoals {
            assert!(sub.target_value < prev_target + 1e-10);
            prev_target = sub.target_value;
        }
    }

    // -----------------------------------------------------------------------
    // Edge: generate replaces chain for same metric
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_replaces_chain() {
        let mut sg = SubgoalGeneration::new();

        // First generation
        let req1 = GenerationRequest::new("pnl", 100.0, 200.0).with_num_subgoals(2);
        sg.generate(&req1);
        assert_eq!(sg.chain("pnl").unwrap().len(), 2);

        let initial_count = sg.subgoal_count();

        // Second generation for same metric — replaces chain
        let req2 = GenerationRequest::new("pnl", 150.0, 300.0).with_num_subgoals(3);
        sg.generate(&req2);

        // Chain now points to new subgoals
        assert_eq!(sg.chain("pnl").unwrap().len(), 3);
        // Old subgoals are still in memory (not cleaned up automatically)
        assert_eq!(sg.subgoal_count(), initial_count + 3);
    }

    // -----------------------------------------------------------------------
    // Cumulative fraction ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_cumulative_fractions_increase() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", 0.0, 100.0).with_num_subgoals(5);
        let _result = sg.generate(&req);

        let chain = sg.chain("pnl").unwrap();
        let mut prev_frac = 0.0;
        for id in chain {
            let sub = sg.subgoal(id).unwrap();
            assert!(sub.cumulative_fraction >= prev_frac - 1e-10);
            prev_frac = sub.cumulative_fraction;
        }
    }

    // -----------------------------------------------------------------------
    // Negative values
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_with_negative_values() {
        let mut sg = SubgoalGeneration::new();
        let req = GenerationRequest::new("pnl", -50.0, 50.0).with_num_subgoals(3);
        let result = sg.generate(&req);

        assert_eq!(result.total_generated, 3);
        // All targets should be between -50 and 50
        for sub in &result.subgoals {
            assert!(sub.target_value > -50.0);
            assert!(sub.target_value < 50.0);
        }
    }
}
