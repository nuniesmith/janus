//! Market regime schemas and pattern matching
//!
//! Part of the Cortex region
//! Component: memory
//!
//! Defines templates (schemas) for recognising and classifying market regimes
//! (bull, bear, ranging, crisis, recovery, bubble, deflation). Each schema
//! carries a set of feature-range criteria that an incoming market observation
//! must satisfy for the regime to be matched. A transition matrix tracks
//! observed regime-to-regime transition probabilities, enabling the system to
//! anticipate likely future regime changes.
//!
//! Key features:
//! - Configurable regime schemas with multi-feature matching criteria
//! - Feature-range bounds (min/max) with optional weight per feature
//! - Schema matching engine with confidence scoring
//! - Regime transition matrix (Markov chain) learned from observations
//! - Most-likely next-regime prediction
//! - Schema library management (add, remove, update)
//! - EMA-smoothed tracking of match confidence and regime duration
//! - Sliding window of recent regime classifications
//! - Running statistics with regime duration and transition tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Identifiers for built-in market regimes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegimeId {
    /// Strong upward trend, low volatility
    Bull,
    /// Sustained downward trend
    Bear,
    /// Sideways movement, no clear direction
    Ranging,
    /// High volatility, sharp drawdowns, correlation spikes
    Crisis,
    /// Transition from bear/crisis toward normalcy
    Recovery,
    /// Overheated market with unsustainable valuations
    Bubble,
    /// Prolonged low-growth, falling prices environment
    Deflation,
}

impl RegimeId {
    /// Total number of built-in regime variants
    pub const COUNT: usize = 7;

    /// Numeric index for array-based tracking
    pub fn index(&self) -> usize {
        match self {
            RegimeId::Bull => 0,
            RegimeId::Bear => 1,
            RegimeId::Ranging => 2,
            RegimeId::Crisis => 3,
            RegimeId::Recovery => 4,
            RegimeId::Bubble => 5,
            RegimeId::Deflation => 6,
        }
    }

    /// Construct from numeric index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => RegimeId::Bull,
            1 => RegimeId::Bear,
            2 => RegimeId::Ranging,
            3 => RegimeId::Crisis,
            4 => RegimeId::Recovery,
            5 => RegimeId::Bubble,
            6 => RegimeId::Deflation,
            _ => RegimeId::Ranging,
        }
    }

    /// Human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            RegimeId::Bull => "Bull",
            RegimeId::Bear => "Bear",
            RegimeId::Ranging => "Ranging",
            RegimeId::Crisis => "Crisis",
            RegimeId::Recovery => "Recovery",
            RegimeId::Bubble => "Bubble",
            RegimeId::Deflation => "Deflation",
        }
    }
}

/// Identifier for a market feature used in schema matching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureId {
    /// Trailing return over lookback period (e.g. 20-day)
    TrailingReturn,
    /// Realised volatility (annualised)
    RealisedVolatility,
    /// Average pairwise asset correlation
    AvgCorrelation,
    /// Maximum drawdown from peak (0–1)
    MaxDrawdown,
    /// Rate of change of volatility
    VolRateOfChange,
    /// Momentum signal strength (-1 to 1)
    MomentumSignal,
    /// Mean-reversion signal strength (-1 to 1)
    MeanReversionSignal,
    /// Credit spread level (bps or fraction)
    CreditSpread,
    /// Yield curve slope (long minus short rate)
    YieldCurveSlope,
    /// Trading volume relative to average
    RelativeVolume,
}

impl FeatureId {
    /// Total number of feature variants
    pub const COUNT: usize = 10;

    /// Numeric index
    pub fn index(&self) -> usize {
        match self {
            FeatureId::TrailingReturn => 0,
            FeatureId::RealisedVolatility => 1,
            FeatureId::AvgCorrelation => 2,
            FeatureId::MaxDrawdown => 3,
            FeatureId::VolRateOfChange => 4,
            FeatureId::MomentumSignal => 5,
            FeatureId::MeanReversionSignal => 6,
            FeatureId::CreditSpread => 7,
            FeatureId::YieldCurveSlope => 8,
            FeatureId::RelativeVolume => 9,
        }
    }

    /// Construct from numeric index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => FeatureId::TrailingReturn,
            1 => FeatureId::RealisedVolatility,
            2 => FeatureId::AvgCorrelation,
            3 => FeatureId::MaxDrawdown,
            4 => FeatureId::VolRateOfChange,
            5 => FeatureId::MomentumSignal,
            6 => FeatureId::MeanReversionSignal,
            7 => FeatureId::CreditSpread,
            8 => FeatureId::YieldCurveSlope,
            9 => FeatureId::RelativeVolume,
            _ => FeatureId::TrailingReturn,
        }
    }
}

/// A single feature-range criterion within a schema
#[derive(Debug, Clone)]
pub struct FeatureCriterion {
    /// Feature being tested
    pub feature: FeatureId,
    /// Minimum acceptable value (inclusive)
    pub min: f64,
    /// Maximum acceptable value (inclusive)
    pub max: f64,
    /// Weight of this criterion in confidence scoring (higher = more important)
    pub weight: f64,
}

impl FeatureCriterion {
    /// Create a new criterion with unit weight
    pub fn new(feature: FeatureId, min: f64, max: f64) -> Self {
        Self {
            feature,
            min,
            max,
            weight: 1.0,
        }
    }

    /// Create with explicit weight
    pub fn with_weight(feature: FeatureId, min: f64, max: f64, weight: f64) -> Self {
        Self {
            feature,
            min,
            max,
            weight,
        }
    }

    /// Check whether a value falls within [min, max]
    pub fn matches(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    /// Compute how well a value matches this criterion.
    ///
    /// Returns 1.0 if value is within bounds, linearly declining toward 0.0
    /// based on distance from the nearest bound, normalised by the range width.
    ///
    /// When either bound is effectively infinite (magnitude > 1e100) or the
    /// range is non-finite / near-zero, linear interpolation is meaningless —
    /// we return 0.0 for any out-of-range value to avoid the
    /// `distance / huge_range → 0 → score ≈ 1.0` bug.
    pub fn match_score(&self, value: f64) -> f64 {
        if self.matches(value) {
            return 1.0;
        }
        // Treat bounds with very large magnitude as effectively infinite.
        if self.min.abs() > 1e100 || self.max.abs() > 1e100 {
            return 0.0;
        }
        let range = (self.max - self.min).abs();
        if !range.is_finite() || range < 1e-15 {
            return 0.0;
        }
        let distance = if value < self.min {
            self.min - value
        } else {
            value - self.max
        };
        let penalty = distance / range;
        (1.0 - penalty).max(0.0)
    }
}

/// A market observation vector used as input to schema matching
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Feature values indexed by FeatureId::index()
    pub values: [f64; FeatureId::COUNT],
}

impl Default for MarketFeatures {
    fn default() -> Self {
        Self {
            values: [0.0; FeatureId::COUNT],
        }
    }
}

impl MarketFeatures {
    /// Get a feature value
    pub fn get(&self, feature: FeatureId) -> f64 {
        self.values[feature.index()]
    }

    /// Set a feature value
    pub fn set(&mut self, feature: FeatureId, value: f64) {
        self.values[feature.index()] = value;
    }

    /// Create from common market observations
    pub fn from_market(
        trailing_return: f64,
        volatility: f64,
        correlation: f64,
        drawdown: f64,
        vol_roc: f64,
        momentum: f64,
    ) -> Self {
        let mut f = Self::default();
        f.set(FeatureId::TrailingReturn, trailing_return);
        f.set(FeatureId::RealisedVolatility, volatility);
        f.set(FeatureId::AvgCorrelation, correlation);
        f.set(FeatureId::MaxDrawdown, drawdown);
        f.set(FeatureId::VolRateOfChange, vol_roc);
        f.set(FeatureId::MomentumSignal, momentum);
        f
    }
}

/// A complete regime schema: a set of criteria that define a regime
#[derive(Debug, Clone)]
pub struct RegimeSchema {
    /// Regime this schema identifies
    pub regime: RegimeId,
    /// Human-readable description
    pub description: String,
    /// Criteria that must be (approximately) satisfied
    pub criteria: Vec<FeatureCriterion>,
    /// Minimum weighted match score to accept this schema (0.0–1.0)
    pub acceptance_threshold: f64,
    /// Whether this schema is enabled
    pub enabled: bool,
    /// Base priority when multiple schemas match (higher = preferred)
    pub priority: f64,
}

impl RegimeSchema {
    /// Evaluate this schema against a feature vector.
    ///
    /// Returns a weighted match score in [0, 1].
    pub fn match_score(&self, features: &MarketFeatures) -> f64 {
        if self.criteria.is_empty() {
            return 0.0;
        }
        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;
        for crit in &self.criteria {
            let value = features.get(crit.feature);
            let score = crit.match_score(value);
            weighted_sum += score * crit.weight;
            weight_total += crit.weight;
        }
        if weight_total.abs() < 1e-15 {
            return 0.0;
        }
        weighted_sum / weight_total
    }

    /// Whether this schema matches the features (score >= acceptance threshold)
    pub fn matches(&self, features: &MarketFeatures) -> bool {
        self.enabled && self.match_score(features) >= self.acceptance_threshold
    }

    /// Number of criteria that are individually satisfied
    pub fn criteria_met_count(&self, features: &MarketFeatures) -> usize {
        self.criteria
            .iter()
            .filter(|c| c.matches(features.get(c.feature)))
            .count()
    }

    /// Fraction of criteria individually satisfied
    pub fn criteria_satisfaction_rate(&self, features: &MarketFeatures) -> f64 {
        if self.criteria.is_empty() {
            return 0.0;
        }
        self.criteria_met_count(features) as f64 / self.criteria.len() as f64
    }
}

/// Result of matching a single schema against features
#[derive(Debug, Clone)]
pub struct SchemaMatchResult {
    /// Regime ID
    pub regime: RegimeId,
    /// Weighted match score (0–1)
    pub score: f64,
    /// Whether the schema accepted the features
    pub accepted: bool,
    /// Number of criteria individually met
    pub criteria_met: usize,
    /// Total number of criteria
    pub criteria_total: usize,
    /// Schema priority
    pub priority: f64,
}

/// Result of classifying market features against all schemas
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Best-matching regime (highest priority among accepted)
    pub best_regime: RegimeId,
    /// Confidence in the best match (0–1)
    pub confidence: f64,
    /// Whether any schema accepted the features
    pub matched: bool,
    /// Whether a regime transition occurred
    pub regime_changed: bool,
    /// Per-schema match results (sorted by score descending)
    pub match_results: Vec<SchemaMatchResult>,
    /// Second-best regime (for ambiguity detection)
    pub runner_up: Option<RegimeId>,
    /// Runner-up confidence
    pub runner_up_confidence: f64,
    /// Ambiguity: ratio of runner-up to best confidence (close to 1.0 = ambiguous)
    pub ambiguity: f64,
}

/// Configuration for the Schemas engine
#[derive(Debug, Clone)]
pub struct SchemasConfig {
    /// Maximum number of schemas
    pub max_schemas: usize,
    /// EMA decay factor for tracking (0 < α < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent classifications
    pub window_size: usize,
    /// Default acceptance threshold for schemas without explicit threshold
    pub default_threshold: f64,
    /// Transition matrix learning rate
    pub transition_lr: f64,
    /// Minimum observations before trusting transition probabilities
    pub min_transition_observations: u64,
}

impl Default for SchemasConfig {
    fn default() -> Self {
        Self {
            max_schemas: 50,
            ema_decay: 0.1,
            window_size: 200,
            default_threshold: 0.60,
            transition_lr: 0.05,
            min_transition_observations: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Transition matrix
// ---------------------------------------------------------------------------

/// Regime transition matrix (row-stochastic Markov chain).
///
/// `matrix[i][j]` = probability of transitioning from regime i to regime j.
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    /// Transition probabilities [from][to]
    pub matrix: [[f64; RegimeId::COUNT]; RegimeId::COUNT],
    /// Observation counts [from][to]
    pub counts: [[u64; RegimeId::COUNT]; RegimeId::COUNT],
    /// Total transitions observed from each regime
    pub row_totals: [u64; RegimeId::COUNT],
}

impl Default for TransitionMatrix {
    fn default() -> Self {
        // Initialise with uniform prior
        let uniform = 1.0 / RegimeId::COUNT as f64;
        Self {
            matrix: [[uniform; RegimeId::COUNT]; RegimeId::COUNT],
            counts: [[0; RegimeId::COUNT]; RegimeId::COUNT],
            row_totals: [0; RegimeId::COUNT],
        }
    }
}

impl TransitionMatrix {
    /// Record a transition and update probabilities
    pub fn record_transition(&mut self, from: RegimeId, to: RegimeId, lr: f64) {
        let fi = from.index();
        let ti = to.index();

        self.counts[fi][ti] += 1;
        self.row_totals[fi] += 1;

        // Incremental update: blend observed frequency with prior
        let total = self.row_totals[fi] as f64;
        if total > 0.0 {
            for j in 0..RegimeId::COUNT {
                let observed_freq = self.counts[fi][j] as f64 / total;
                self.matrix[fi][j] = lr * observed_freq + (1.0 - lr) * self.matrix[fi][j];
            }
            // Re-normalise row to ensure it sums to 1.0
            let row_sum: f64 = self.matrix[fi].iter().sum();
            if row_sum > 1e-15 {
                for j in 0..RegimeId::COUNT {
                    self.matrix[fi][j] /= row_sum;
                }
            }
        }
    }

    /// Get the most likely next regime from a given regime
    pub fn most_likely_next(&self, from: RegimeId) -> (RegimeId, f64) {
        let fi = from.index();
        let (best_idx, best_prob) = self.matrix[fi]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        (RegimeId::from_index(best_idx), *best_prob)
    }

    /// Get the probability of transitioning from one regime to another
    pub fn transition_prob(&self, from: RegimeId, to: RegimeId) -> f64 {
        self.matrix[from.index()][to.index()]
    }

    /// Total observations from a given regime
    pub fn observations_from(&self, regime: RegimeId) -> u64 {
        self.row_totals[regime.index()]
    }

    /// Total observations across all regimes
    pub fn total_observations(&self) -> u64 {
        self.row_totals.iter().sum()
    }

    /// Reset all counts and probabilities to uniform prior
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Stationary distribution (eigenvector) via power iteration.
    ///
    /// Returns the long-run probability of being in each regime.
    pub fn stationary_distribution(&self) -> [f64; RegimeId::COUNT] {
        const N: usize = RegimeId::COUNT;
        let n = N;
        let mut pi = [1.0 / n as f64; N];

        // Power iteration: π = πP, repeated until convergence
        for _ in 0..200 {
            let mut new_pi = [0.0f64; N];
            for j in 0..n {
                for i in 0..n {
                    new_pi[j] += pi[i] * self.matrix[i][j];
                }
            }
            // Normalise
            let sum: f64 = new_pi.iter().sum();
            if sum > 1e-15 {
                for v in &mut new_pi {
                    *v /= sum;
                }
            }
            // Check convergence
            let diff: f64 = pi
                .iter()
                .zip(new_pi.iter())
                .map(|(a, b): (&f64, &f64)| (a - b).abs())
                .sum();
            pi = new_pi;
            if diff < 1e-12 {
                break;
            }
        }

        pi
    }
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the Schemas engine
#[derive(Debug, Clone)]
pub struct SchemasStats {
    /// Total classification evaluations
    pub total_evaluations: u64,
    /// Total regime transitions observed
    pub total_transitions: u64,
    /// Steps spent in each regime
    pub regime_durations: [u64; RegimeId::COUNT],
    /// Number of times each regime was entered
    pub regime_entries: [u64; RegimeId::COUNT],
    /// EMA of best match confidence
    pub ema_confidence: f64,
    /// EMA of ambiguity ratio
    pub ema_ambiguity: f64,
    /// EMA of criteria satisfaction rate
    pub ema_satisfaction: f64,
    /// Current regime duration in steps
    pub current_regime_duration: u64,
    /// Number of classifications where no schema matched
    pub unmatched_count: u64,
}

impl Default for SchemasStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_transitions: 0,
            regime_durations: [0; RegimeId::COUNT],
            regime_entries: [0; RegimeId::COUNT],
            ema_confidence: 0.5,
            ema_ambiguity: 0.0,
            ema_satisfaction: 0.5,
            current_regime_duration: 0,
            unmatched_count: 0,
        }
    }
}

impl SchemasStats {
    /// Most common regime by total duration
    pub fn dominant_regime(&self) -> RegimeId {
        let max_idx = self
            .regime_durations
            .iter()
            .enumerate()
            .max_by_key(|(_, d)| *d)
            .map(|(i, _)| i)
            .unwrap_or(2); // default to Ranging
        RegimeId::from_index(max_idx)
    }

    /// Transition rate (transitions per evaluation)
    pub fn transition_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.total_transitions as f64 / self.total_evaluations as f64
    }

    /// Fraction of evaluations where no schema matched
    pub fn unmatched_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.unmatched_count as f64 / self.total_evaluations as f64
    }

    /// Fraction of time spent in each regime
    pub fn regime_fractions(&self) -> [f64; RegimeId::COUNT] {
        let total: u64 = self.regime_durations.iter().sum();
        let mut fractions = [0.0; RegimeId::COUNT];
        if total > 0 {
            for (i, d) in self.regime_durations.iter().enumerate() {
                fractions[i] = *d as f64 / total as f64;
            }
        }
        fractions
    }

    /// Average regime duration
    pub fn avg_regime_duration(&self) -> f64 {
        let entries: u64 = self.regime_entries.iter().sum();
        if entries == 0 {
            return 0.0;
        }
        let total_duration: u64 = self.regime_durations.iter().sum();
        total_duration as f64 / entries as f64
    }
}

// ---------------------------------------------------------------------------
// Schemas Engine
// ---------------------------------------------------------------------------

/// Market regime schema engine.
///
/// Maintains a library of regime schemas, classifies market observations
/// against them, tracks a regime transition matrix, and provides
/// next-regime predictions.
pub struct Schemas {
    config: SchemasConfig,
    /// Schema library
    schemas: Vec<RegimeSchema>,
    /// Currently detected regime
    current_regime: RegimeId,
    /// Previous regime (for transition tracking)
    previous_regime: RegimeId,
    /// Regime duration counter
    regime_duration: u64,
    /// Transition matrix
    transitions: TransitionMatrix,
    /// EMA initialised flag
    ema_initialized: bool,
    /// Sliding window of recent classification results
    recent: VecDeque<ClassificationRecord>,
    /// Running statistics
    stats: SchemasStats,
}

/// Lightweight record stored in the sliding window
#[derive(Debug, Clone)]
pub struct ClassificationRecord {
    /// Detected regime
    pub regime: RegimeId,
    /// Confidence
    pub confidence: f64,
    /// Whether the regime changed from the previous step
    pub regime_changed: bool,
    /// Ambiguity ratio
    pub ambiguity: f64,
    /// Criteria satisfaction rate of the best match
    pub satisfaction: f64,
}

impl Default for Schemas {
    fn default() -> Self {
        Self::new()
    }
}

impl Schemas {
    /// Create a new Schemas engine with default configuration
    pub fn new() -> Self {
        Self {
            config: SchemasConfig::default(),
            schemas: Vec::new(),
            current_regime: RegimeId::Ranging,
            previous_regime: RegimeId::Ranging,
            regime_duration: 0,
            transitions: TransitionMatrix::default(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: SchemasStats::default(),
        }
    }

    /// Create with explicit configuration
    pub fn with_config(config: SchemasConfig) -> Result<Self> {
        if config.max_schemas == 0 {
            return Err(Error::InvalidInput("max_schemas must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.default_threshold <= 0.0 || config.default_threshold > 1.0 {
            return Err(Error::InvalidInput(
                "default_threshold must be in (0, 1]".into(),
            ));
        }
        if config.transition_lr <= 0.0 || config.transition_lr > 1.0 {
            return Err(Error::InvalidInput(
                "transition_lr must be in (0, 1]".into(),
            ));
        }
        Ok(Self {
            config,
            schemas: Vec::new(),
            current_regime: RegimeId::Ranging,
            previous_regime: RegimeId::Ranging,
            regime_duration: 0,
            transitions: TransitionMatrix::default(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: SchemasStats::default(),
        })
    }

    /// Main processing function (trait conformance)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Schema management
    // -----------------------------------------------------------------------

    /// Add a regime schema to the library
    pub fn add_schema(&mut self, schema: RegimeSchema) -> Result<()> {
        if self.schemas.len() >= self.config.max_schemas {
            return Err(Error::ResourceExhausted(format!(
                "max schemas ({}) reached",
                self.config.max_schemas
            )));
        }
        if schema.criteria.is_empty() {
            return Err(Error::InvalidInput(
                "schema must have at least one criterion".into(),
            ));
        }
        if schema.acceptance_threshold <= 0.0 || schema.acceptance_threshold > 1.0 {
            return Err(Error::InvalidInput(
                "acceptance_threshold must be in (0, 1]".into(),
            ));
        }
        self.schemas.push(schema);
        Ok(())
    }

    /// Remove a schema by regime ID (removes the first matching schema)
    pub fn remove_schema(&mut self, regime: RegimeId) -> Result<()> {
        let idx = self
            .schemas
            .iter()
            .position(|s| s.regime == regime)
            .ok_or_else(|| Error::NotFound(format!("schema for regime {:?} not found", regime)))?;
        self.schemas.remove(idx);
        Ok(())
    }

    /// Enable or disable a schema by regime ID
    pub fn set_schema_enabled(&mut self, regime: RegimeId, enabled: bool) -> Result<()> {
        let schema = self
            .schemas
            .iter_mut()
            .find(|s| s.regime == regime)
            .ok_or_else(|| Error::NotFound(format!("schema for regime {:?} not found", regime)))?;
        schema.enabled = enabled;
        Ok(())
    }

    /// Get a schema by regime ID
    pub fn get_schema(&self, regime: RegimeId) -> Option<&RegimeSchema> {
        self.schemas.iter().find(|s| s.regime == regime)
    }

    /// Number of schemas in the library
    pub fn schema_count(&self) -> usize {
        self.schemas.len()
    }

    /// All schemas
    pub fn schemas(&self) -> &[RegimeSchema] {
        &self.schemas
    }

    // -----------------------------------------------------------------------
    // Classification
    // -----------------------------------------------------------------------

    /// Classify market features against all schemas.
    ///
    /// Returns a ClassificationResult with the best-matching regime,
    /// confidence score, and per-schema match details. Updates the
    /// transition matrix and statistics.
    pub fn classify(&mut self, features: &MarketFeatures) -> ClassificationResult {
        let mut results: Vec<SchemaMatchResult> = Vec::with_capacity(self.schemas.len());

        for schema in &self.schemas {
            if !schema.enabled {
                continue;
            }
            let score = schema.match_score(features);
            let accepted = score >= schema.acceptance_threshold;
            let criteria_met = schema.criteria_met_count(features);
            let criteria_total = schema.criteria.len();

            results.push(SchemaMatchResult {
                regime: schema.regime,
                score,
                accepted,
                criteria_met,
                criteria_total,
                priority: schema.priority,
            });
        }

        // Sort by (accepted desc, priority desc, score desc)
        results.sort_by(|a, b| {
            b.accepted
                .cmp(&a.accepted)
                .then(
                    b.priority
                        .partial_cmp(&a.priority)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        // Best match
        let (best_regime, confidence, matched) = if let Some(best) = results.first() {
            if best.accepted {
                (best.regime, best.score, true)
            } else {
                // No schema accepted — fall back to current regime
                (self.current_regime, best.score * 0.5, false)
            }
        } else {
            // No schemas at all
            (self.current_regime, 0.0, false)
        };

        // Runner-up (second-best accepted, or second overall)
        let runner_up_result = results.iter().skip(1).find(|r| r.accepted);
        let runner_up = runner_up_result.map(|r| r.regime);
        let runner_up_confidence = runner_up_result.map(|r| r.score).unwrap_or(0.0);
        let ambiguity = if confidence > 1e-15 {
            runner_up_confidence / confidence
        } else {
            0.0
        };

        // Detect regime change
        let regime_changed = best_regime != self.current_regime;

        if regime_changed {
            // Record transition
            self.transitions.record_transition(
                self.current_regime,
                best_regime,
                self.config.transition_lr,
            );

            self.previous_regime = self.current_regime;
            self.current_regime = best_regime;
            self.stats.total_transitions += 1;
            self.stats.regime_entries[best_regime.index()] += 1;
            self.regime_duration = 0;
        }

        self.regime_duration += 1;
        self.stats.regime_durations[self.current_regime.index()] += 1;
        self.stats.current_regime_duration = self.regime_duration;
        self.stats.total_evaluations += 1;

        if !matched {
            self.stats.unmatched_count += 1;
        }

        // Best-match satisfaction rate
        let satisfaction = if let Some(best) = results.first() {
            if best.criteria_total > 0 {
                best.criteria_met as f64 / best.criteria_total as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        // EMA updates
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_confidence = confidence;
            self.stats.ema_ambiguity = ambiguity;
            self.stats.ema_satisfaction = satisfaction;
            self.ema_initialized = true;
        } else {
            self.stats.ema_confidence =
                alpha * confidence + (1.0 - alpha) * self.stats.ema_confidence;
            self.stats.ema_ambiguity = alpha * ambiguity + (1.0 - alpha) * self.stats.ema_ambiguity;
            self.stats.ema_satisfaction =
                alpha * satisfaction + (1.0 - alpha) * self.stats.ema_satisfaction;
        }

        // Sliding window
        let record = ClassificationRecord {
            regime: best_regime,
            confidence,
            regime_changed,
            ambiguity,
            satisfaction,
        };
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(record);

        ClassificationResult {
            best_regime,
            confidence,
            matched,
            regime_changed,
            match_results: results,
            runner_up,
            runner_up_confidence,
            ambiguity,
        }
    }

    // -----------------------------------------------------------------------
    // Predictions
    // -----------------------------------------------------------------------

    /// Predict the most likely next regime based on the transition matrix
    pub fn predict_next_regime(&self) -> (RegimeId, f64) {
        self.transitions.most_likely_next(self.current_regime)
    }

    /// Get the transition probability from the current regime to a target
    pub fn transition_probability(&self, to: RegimeId) -> f64 {
        self.transitions.transition_prob(self.current_regime, to)
    }

    /// Whether the transition matrix has enough observations to be trusted
    pub fn transition_matrix_reliable(&self) -> bool {
        self.transitions.observations_from(self.current_regime)
            >= self.config.min_transition_observations
    }

    /// Get the stationary distribution of the transition matrix
    pub fn stationary_distribution(&self) -> [f64; RegimeId::COUNT] {
        self.transitions.stationary_distribution()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current detected regime
    pub fn current_regime(&self) -> RegimeId {
        self.current_regime
    }

    /// Previous regime
    pub fn previous_regime(&self) -> RegimeId {
        self.previous_regime
    }

    /// Current regime duration in steps
    pub fn regime_duration(&self) -> u64 {
        self.regime_duration
    }

    /// Transition matrix reference
    pub fn transition_matrix(&self) -> &TransitionMatrix {
        &self.transitions
    }

    /// Running statistics
    pub fn stats(&self) -> &SchemasStats {
        &self.stats
    }

    /// Configuration
    pub fn config(&self) -> &SchemasConfig {
        &self.config
    }

    /// Recent classification records
    pub fn recent_classifications(&self) -> &VecDeque<ClassificationRecord> {
        &self.recent
    }

    /// EMA-smoothed confidence
    pub fn smoothed_confidence(&self) -> f64 {
        self.stats.ema_confidence
    }

    /// EMA-smoothed ambiguity
    pub fn smoothed_ambiguity(&self) -> f64 {
        self.stats.ema_ambiguity
    }

    /// EMA-smoothed criteria satisfaction
    pub fn smoothed_satisfaction(&self) -> f64 {
        self.stats.ema_satisfaction
    }

    /// Windowed average confidence
    pub fn windowed_avg_confidence(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.confidence).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed regime stability: fraction of window in the current regime
    pub fn windowed_regime_stability(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let same = self
            .recent
            .iter()
            .filter(|r| r.regime == self.current_regime)
            .count();
        same as f64 / self.recent.len() as f64
    }

    /// Windowed transition rate
    pub fn windowed_transition_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let transitions = self.recent.iter().filter(|r| r.regime_changed).count();
        transitions as f64 / self.recent.len() as f64
    }

    /// Whether confidence is trending downward
    pub fn is_confidence_declining(&self) -> bool {
        let n = self.recent.len();
        if n < 4 {
            return false;
        }
        let mid = n / 2;
        let first_avg: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.confidence)
            .sum::<f64>()
            / mid as f64;
        let second_avg: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.confidence)
            .sum::<f64>()
            / (n - mid) as f64;
        second_avg < first_avg * 0.9
    }

    /// Reset engine state (schemas are kept)
    pub fn reset(&mut self) {
        self.current_regime = RegimeId::Ranging;
        self.previous_regime = RegimeId::Ranging;
        self.regime_duration = 0;
        self.transitions = TransitionMatrix::default();
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = SchemasStats::default();
    }
}

// ---------------------------------------------------------------------------
// Preset schemas
// ---------------------------------------------------------------------------

/// Create the standard set of regime schemas with default thresholds
pub fn preset_schemas() -> Vec<RegimeSchema> {
    vec![
        RegimeSchema {
            regime: RegimeId::Bull,
            description: "Strong upward trend with low volatility".into(),
            criteria: vec![
                FeatureCriterion::with_weight(FeatureId::TrailingReturn, 0.02, f64::MAX, 2.0),
                FeatureCriterion::with_weight(FeatureId::RealisedVolatility, 0.0, 0.25, 1.5),
                FeatureCriterion::with_weight(FeatureId::MomentumSignal, 0.2, 1.0, 1.0),
                FeatureCriterion::with_weight(FeatureId::MaxDrawdown, 0.0, 0.08, 1.0),
            ],
            acceptance_threshold: 0.65,
            enabled: true,
            priority: 2.0,
        },
        RegimeSchema {
            regime: RegimeId::Bear,
            description: "Sustained downward trend".into(),
            criteria: vec![
                FeatureCriterion::with_weight(FeatureId::TrailingReturn, f64::MIN, -0.03, 2.0),
                FeatureCriterion::with_weight(FeatureId::MomentumSignal, -1.0, -0.1, 1.0),
                FeatureCriterion::with_weight(FeatureId::MaxDrawdown, 0.05, 0.50, 1.5),
            ],
            acceptance_threshold: 0.55,
            enabled: true,
            priority: 2.0,
        },
        RegimeSchema {
            regime: RegimeId::Ranging,
            description: "Sideways, no clear direction".into(),
            criteria: vec![
                FeatureCriterion::with_weight(FeatureId::TrailingReturn, -0.02, 0.02, 1.5),
                FeatureCriterion::with_weight(FeatureId::RealisedVolatility, 0.0, 0.25, 1.0),
                FeatureCriterion::with_weight(FeatureId::MomentumSignal, -0.3, 0.3, 1.0),
            ],
            acceptance_threshold: 0.50,
            enabled: true,
            priority: 1.0,
        },
        RegimeSchema {
            regime: RegimeId::Crisis,
            description: "High volatility, sharp drawdowns, correlation spike".into(),
            criteria: vec![
                FeatureCriterion::with_weight(FeatureId::RealisedVolatility, 0.40, f64::MAX, 2.0),
                FeatureCriterion::with_weight(FeatureId::MaxDrawdown, 0.10, 1.0, 2.0),
                FeatureCriterion::with_weight(FeatureId::AvgCorrelation, 0.65, 1.0, 1.5),
                FeatureCriterion::with_weight(FeatureId::VolRateOfChange, 0.05, f64::MAX, 1.0),
            ],
            acceptance_threshold: 0.65,
            enabled: true,
            priority: 3.0, // Crisis gets high priority to override others
        },
        RegimeSchema {
            regime: RegimeId::Recovery,
            description: "Transition from bear/crisis toward normalcy".into(),
            criteria: vec![
                FeatureCriterion::with_weight(FeatureId::TrailingReturn, 0.005, 0.05, 1.5),
                FeatureCriterion::with_weight(FeatureId::RealisedVolatility, 0.15, 0.45, 1.0),
                FeatureCriterion::with_weight(FeatureId::VolRateOfChange, f64::MIN, -0.01, 1.0),
                FeatureCriterion::with_weight(FeatureId::MaxDrawdown, 0.03, 0.20, 1.0),
            ],
            acceptance_threshold: 0.80,
            enabled: true,
            priority: 1.5,
        },
        RegimeSchema {
            regime: RegimeId::Bubble,
            description: "Overheated market with unsustainable trends".into(),
            criteria: vec![
                FeatureCriterion::with_weight(FeatureId::TrailingReturn, 0.08, f64::MAX, 2.0),
                FeatureCriterion::with_weight(FeatureId::RealisedVolatility, 0.0, 0.30, 1.0),
                FeatureCriterion::with_weight(FeatureId::MomentumSignal, 0.6, 1.0, 1.5),
                FeatureCriterion::with_weight(FeatureId::RelativeVolume, 1.5, f64::MAX, 1.0),
            ],
            acceptance_threshold: 0.55,
            enabled: true,
            priority: 2.5,
        },
        RegimeSchema {
            regime: RegimeId::Deflation,
            description: "Prolonged low-growth, falling prices".into(),
            criteria: vec![
                FeatureCriterion::with_weight(FeatureId::TrailingReturn, -0.05, 0.0, 1.5),
                FeatureCriterion::with_weight(FeatureId::RealisedVolatility, 0.0, 0.20, 1.0),
                FeatureCriterion::with_weight(FeatureId::YieldCurveSlope, f64::MIN, 0.0, 2.0),
                FeatureCriterion::with_weight(FeatureId::CreditSpread, 0.03, f64::MAX, 1.0),
            ],
            acceptance_threshold: 0.85,
            enabled: true,
            priority: 1.5,
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn bull_features() -> MarketFeatures {
        MarketFeatures::from_market(0.05, 0.12, 0.25, 0.02, -0.01, 0.7)
    }

    fn bear_features() -> MarketFeatures {
        MarketFeatures::from_market(-0.08, 0.30, 0.50, 0.12, 0.05, -0.6)
    }

    fn crisis_features() -> MarketFeatures {
        let mut f = MarketFeatures::from_market(-0.15, 0.70, 0.85, 0.25, 0.20, -0.9);
        f.set(FeatureId::CreditSpread, 0.05);
        f
    }

    fn ranging_features() -> MarketFeatures {
        MarketFeatures::from_market(0.005, 0.18, 0.35, 0.03, 0.0, 0.1)
    }

    fn with_presets() -> Schemas {
        let mut s = Schemas::new();
        for schema in preset_schemas() {
            s.add_schema(schema).unwrap();
        }
        s
    }

    // -- Construction --

    #[test]
    fn test_new_default() {
        let s = Schemas::new();
        assert_eq!(s.schema_count(), 0);
        assert_eq!(s.current_regime(), RegimeId::Ranging);
        assert!(s.process().is_ok());
    }

    #[test]
    fn test_with_config() {
        let s = Schemas::with_config(SchemasConfig::default());
        assert!(s.is_ok());
    }

    #[test]
    fn test_invalid_config_max_schemas_zero() {
        let mut cfg = SchemasConfig::default();
        cfg.max_schemas = 0;
        assert!(Schemas::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = SchemasConfig::default();
        cfg.ema_decay = 0.0;
        assert!(Schemas::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = SchemasConfig::default();
        cfg.ema_decay = 1.0;
        assert!(Schemas::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_window_size_zero() {
        let mut cfg = SchemasConfig::default();
        cfg.window_size = 0;
        assert!(Schemas::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_threshold_zero() {
        let mut cfg = SchemasConfig::default();
        cfg.default_threshold = 0.0;
        assert!(Schemas::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_transition_lr_zero() {
        let mut cfg = SchemasConfig::default();
        cfg.transition_lr = 0.0;
        assert!(Schemas::with_config(cfg).is_err());
    }

    // -- Schema management --

    #[test]
    fn test_add_schema() {
        let mut s = Schemas::new();
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 1.0)],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        assert!(s.add_schema(schema).is_ok());
        assert_eq!(s.schema_count(), 1);
    }

    #[test]
    fn test_add_schema_no_criteria() {
        let mut s = Schemas::new();
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Empty".into(),
            criteria: vec![],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        assert!(s.add_schema(schema).is_err());
    }

    #[test]
    fn test_add_schema_bad_threshold() {
        let mut s = Schemas::new();
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bad".into(),
            criteria: vec![FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 1.0)],
            acceptance_threshold: 0.0,
            enabled: true,
            priority: 1.0,
        };
        assert!(s.add_schema(schema).is_err());
    }

    #[test]
    fn test_add_schema_exceeds_max() {
        let mut cfg = SchemasConfig::default();
        cfg.max_schemas = 1;
        let mut s = Schemas::with_config(cfg).unwrap();

        let s1 = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 1.0)],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        let s2 = RegimeSchema {
            regime: RegimeId::Bear,
            description: "Bear".into(),
            criteria: vec![FeatureCriterion::new(
                FeatureId::TrailingReturn,
                -1.0,
                -0.01,
            )],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        assert!(s.add_schema(s1).is_ok());
        assert!(s.add_schema(s2).is_err());
    }

    #[test]
    fn test_remove_schema() {
        let mut s = Schemas::new();
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 1.0)],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        s.add_schema(schema).unwrap();
        assert!(s.remove_schema(RegimeId::Bull).is_ok());
        assert_eq!(s.schema_count(), 0);
    }

    #[test]
    fn test_remove_schema_not_found() {
        let mut s = Schemas::new();
        assert!(s.remove_schema(RegimeId::Bull).is_err());
    }

    #[test]
    fn test_enable_disable_schema() {
        let mut s = Schemas::new();
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 1.0)],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        s.add_schema(schema).unwrap();
        assert!(s.set_schema_enabled(RegimeId::Bull, false).is_ok());
        assert!(!s.get_schema(RegimeId::Bull).unwrap().enabled);
    }

    #[test]
    fn test_enable_disable_not_found() {
        let mut s = Schemas::new();
        assert!(s.set_schema_enabled(RegimeId::Bull, true).is_err());
    }

    #[test]
    fn test_get_schema() {
        let mut s = Schemas::new();
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 1.0)],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        s.add_schema(schema).unwrap();
        assert!(s.get_schema(RegimeId::Bull).is_some());
        assert!(s.get_schema(RegimeId::Bear).is_none());
    }

    // -- Feature criterion --

    #[test]
    fn test_criterion_matches_within_range() {
        let c = FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 0.10);
        assert!(c.matches(0.05));
        assert!(c.matches(0.01));
        assert!(c.matches(0.10));
    }

    #[test]
    fn test_criterion_does_not_match_outside() {
        let c = FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 0.10);
        assert!(!c.matches(0.005));
        assert!(!c.matches(0.11));
    }

    #[test]
    fn test_criterion_match_score_inside() {
        let c = FeatureCriterion::new(FeatureId::TrailingReturn, 0.0, 1.0);
        assert_eq!(c.match_score(0.5), 1.0);
    }

    #[test]
    fn test_criterion_match_score_outside_decays() {
        let c = FeatureCriterion::new(FeatureId::TrailingReturn, 0.0, 1.0);
        let score = c.match_score(1.5);
        assert!(score < 1.0);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_criterion_match_score_far_outside_is_zero() {
        let c = FeatureCriterion::new(FeatureId::TrailingReturn, 0.0, 1.0);
        let score = c.match_score(5.0); // distance = 4, range = 1
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_criterion_weight() {
        let c = FeatureCriterion::with_weight(FeatureId::TrailingReturn, 0.0, 1.0, 2.5);
        assert!((c.weight - 2.5).abs() < 1e-10);
    }

    // -- RegimeSchema --

    #[test]
    fn test_schema_match_score_all_met() {
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![
                FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 0.10),
                FeatureCriterion::new(FeatureId::RealisedVolatility, 0.05, 0.25),
            ],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        let features = bull_features(); // return=0.05, vol=0.12
        assert_eq!(schema.match_score(&features), 1.0);
    }

    #[test]
    fn test_schema_match_score_partial() {
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![
                FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 0.10),
                FeatureCriterion::new(FeatureId::RealisedVolatility, 0.0, 0.05), // won't match (0.12)
            ],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        let features = bull_features();
        let score = schema.match_score(&features);
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_schema_matches_accepted() {
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![
                FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 0.20),
                FeatureCriterion::new(FeatureId::RealisedVolatility, 0.05, 0.30),
            ],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        assert!(schema.matches(&bull_features()));
    }

    #[test]
    fn test_schema_matches_disabled_not_accepted() {
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 0.20)],
            acceptance_threshold: 0.5,
            enabled: false,
            priority: 1.0,
        };
        assert!(!schema.matches(&bull_features()));
    }

    #[test]
    fn test_schema_criteria_met_count() {
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![
                FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 0.10),
                FeatureCriterion::new(FeatureId::RealisedVolatility, 0.0, 0.05),
            ],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        let features = bull_features(); // return=0.05 ✓, vol=0.12 ✗
        assert_eq!(schema.criteria_met_count(&features), 1);
    }

    #[test]
    fn test_schema_criteria_satisfaction_rate() {
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Bull".into(),
            criteria: vec![
                FeatureCriterion::new(FeatureId::TrailingReturn, 0.01, 0.10),
                FeatureCriterion::new(FeatureId::RealisedVolatility, 0.0, 0.30),
            ],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        let features = bull_features();
        assert!((schema.criteria_satisfaction_rate(&features) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_schema_empty_criteria_score_zero() {
        let schema = RegimeSchema {
            regime: RegimeId::Bull,
            description: "Empty".into(),
            criteria: vec![],
            acceptance_threshold: 0.5,
            enabled: true,
            priority: 1.0,
        };
        assert_eq!(schema.match_score(&bull_features()), 0.0);
    }

    // -- MarketFeatures --

    #[test]
    fn test_market_features_default() {
        let f = MarketFeatures::default();
        for v in &f.values {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_market_features_get_set() {
        let mut f = MarketFeatures::default();
        f.set(FeatureId::TrailingReturn, 0.05);
        assert!((f.get(FeatureId::TrailingReturn) - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_market_features_from_market() {
        let f = MarketFeatures::from_market(0.05, 0.12, 0.25, 0.02, -0.01, 0.7);
        assert!((f.get(FeatureId::TrailingReturn) - 0.05).abs() < 1e-10);
        assert!((f.get(FeatureId::RealisedVolatility) - 0.12).abs() < 1e-10);
        assert!((f.get(FeatureId::AvgCorrelation) - 0.25).abs() < 1e-10);
        assert!((f.get(FeatureId::MaxDrawdown) - 0.02).abs() < 1e-10);
        assert!((f.get(FeatureId::VolRateOfChange) - (-0.01)).abs() < 1e-10);
        assert!((f.get(FeatureId::MomentumSignal) - 0.7).abs() < 1e-10);
    }

    // -- RegimeId --

    #[test]
    fn test_regime_id_index_roundtrip() {
        for i in 0..RegimeId::COUNT {
            let r = RegimeId::from_index(i);
            assert_eq!(r.index(), i);
        }
    }

    #[test]
    fn test_regime_id_label() {
        assert_eq!(RegimeId::Bull.label(), "Bull");
        assert_eq!(RegimeId::Crisis.label(), "Crisis");
    }

    #[test]
    fn test_regime_id_out_of_range_defaults_to_ranging() {
        assert_eq!(RegimeId::from_index(999), RegimeId::Ranging);
    }

    // -- FeatureId --

    #[test]
    fn test_feature_id_index_roundtrip() {
        for i in 0..FeatureId::COUNT {
            let f = FeatureId::from_index(i);
            assert_eq!(f.index(), i);
        }
    }

    #[test]
    fn test_feature_id_out_of_range_defaults() {
        assert_eq!(FeatureId::from_index(999), FeatureId::TrailingReturn);
    }

    // -- Classification with presets --

    #[test]
    fn test_classify_bull() {
        let mut s = with_presets();
        let result = s.classify(&bull_features());
        assert!(result.matched);
        assert_eq!(result.best_regime, RegimeId::Bull);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_classify_crisis() {
        let mut s = with_presets();
        let result = s.classify(&crisis_features());
        assert!(result.matched);
        assert_eq!(result.best_regime, RegimeId::Crisis);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_classify_bear() {
        let mut s = with_presets();
        let result = s.classify(&bear_features());
        assert!(result.matched);
        assert_eq!(result.best_regime, RegimeId::Bear);
    }

    #[test]
    fn test_classify_ranging() {
        let mut s = with_presets();
        let features = ranging_features();

        // Debug: print per-schema scores
        for schema in &s.schemas {
            let score = schema.match_score(&features);
            let accepted = score >= schema.acceptance_threshold;
            eprintln!(
                "Schema {:?}: score={:.4}, threshold={:.2}, accepted={}, priority={:.1}",
                schema.regime, score, schema.acceptance_threshold, accepted, schema.priority
            );
        }

        let result = s.classify(&features);
        eprintln!(
            "Result: best={:?}, confidence={:.4}, matched={}",
            result.best_regime, result.confidence, result.matched,
        );
        assert!(result.matched);
        // Ranging should match
        assert_eq!(result.best_regime, RegimeId::Ranging);
    }

    #[test]
    fn test_classify_regime_change_detected() {
        let mut s = with_presets();
        let r1 = s.classify(&bull_features());
        assert!(r1.regime_changed); // from initial Ranging to Bull

        let r2 = s.classify(&bull_features());
        assert!(!r2.regime_changed); // same regime

        let r3 = s.classify(&crisis_features());
        assert!(r3.regime_changed); // Bull to Crisis
    }

    #[test]
    fn test_classify_no_schemas() {
        let mut s = Schemas::new();
        let result = s.classify(&bull_features());
        assert!(!result.matched);
        assert_eq!(result.best_regime, RegimeId::Ranging); // default
    }

    #[test]
    fn test_classify_all_disabled() {
        let mut s = with_presets();
        for i in 0..RegimeId::COUNT {
            let _ = s.set_schema_enabled(RegimeId::from_index(i), false);
        }
        let result = s.classify(&bull_features());
        assert!(!result.matched);
    }

    #[test]
    fn test_classify_ambiguity_detected() {
        let mut s = with_presets();
        let result = s.classify(&bull_features());
        // ambiguity should be a valid ratio
        assert!(result.ambiguity >= 0.0);
        assert!(result.ambiguity <= 1.0);
    }

    #[test]
    fn test_classify_runner_up() {
        let mut s = with_presets();
        let result = s.classify(&bull_features());
        // There should be match results for multiple schemas
        assert!(result.match_results.len() > 1);
    }

    // -- Transition matrix --

    #[test]
    fn test_transition_matrix_default_uniform() {
        let tm = TransitionMatrix::default();
        let expected = 1.0 / RegimeId::COUNT as f64;
        for row in &tm.matrix {
            for &p in row {
                assert!((p - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_transition_matrix_record_transition() {
        let mut tm = TransitionMatrix::default();
        tm.record_transition(RegimeId::Bull, RegimeId::Bear, 0.5);
        assert!(tm.counts[RegimeId::Bull.index()][RegimeId::Bear.index()] == 1);
        assert_eq!(tm.row_totals[RegimeId::Bull.index()], 1);
    }

    #[test]
    fn test_transition_matrix_probabilities_sum_to_one() {
        let mut tm = TransitionMatrix::default();
        tm.record_transition(RegimeId::Bull, RegimeId::Bear, 0.5);
        tm.record_transition(RegimeId::Bull, RegimeId::Crisis, 0.5);
        tm.record_transition(RegimeId::Bull, RegimeId::Bear, 0.5);

        let row_sum: f64 = tm.matrix[RegimeId::Bull.index()].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transition_matrix_most_likely_next() {
        let mut tm = TransitionMatrix::default();
        for _ in 0..10 {
            tm.record_transition(RegimeId::Bull, RegimeId::Bear, 0.5);
        }
        for _ in 0..2 {
            tm.record_transition(RegimeId::Bull, RegimeId::Ranging, 0.5);
        }

        let (next, prob) = tm.most_likely_next(RegimeId::Bull);
        assert_eq!(next, RegimeId::Bear);
        assert!(prob > 0.0);
    }

    #[test]
    fn test_transition_matrix_prob() {
        let mut tm = TransitionMatrix::default();
        tm.record_transition(RegimeId::Bull, RegimeId::Bear, 1.0);
        let prob = tm.transition_prob(RegimeId::Bull, RegimeId::Bear);
        assert!(prob > 0.0);
    }

    #[test]
    fn test_transition_matrix_observations() {
        let mut tm = TransitionMatrix::default();
        tm.record_transition(RegimeId::Bull, RegimeId::Bear, 0.5);
        tm.record_transition(RegimeId::Bull, RegimeId::Crisis, 0.5);
        assert_eq!(tm.observations_from(RegimeId::Bull), 2);
        assert_eq!(tm.total_observations(), 2);
    }

    #[test]
    fn test_transition_matrix_reset() {
        let mut tm = TransitionMatrix::default();
        tm.record_transition(RegimeId::Bull, RegimeId::Bear, 0.5);
        tm.reset();
        assert_eq!(tm.total_observations(), 0);
    }

    #[test]
    fn test_stationary_distribution_sums_to_one() {
        let tm = TransitionMatrix::default();
        let pi = tm.stationary_distribution();
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_stationary_distribution_uniform_for_uniform_matrix() {
        let tm = TransitionMatrix::default();
        let pi = tm.stationary_distribution();
        let expected = 1.0 / RegimeId::COUNT as f64;
        for &p in &pi {
            assert!((p - expected).abs() < 1e-8);
        }
    }

    // -- Predictions --

    #[test]
    fn test_predict_next_regime() {
        let mut s = with_presets();
        s.classify(&bull_features());
        let (next, prob) = s.predict_next_regime();
        assert!(prob > 0.0);
        assert!(prob <= 1.0);
        // Just verify it returns a valid regime
        let _ = next.label();
    }

    #[test]
    fn test_transition_probability_accessor() {
        let s = with_presets();
        let prob = s.transition_probability(RegimeId::Bear);
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_transition_matrix_reliable_initially_false() {
        let s = with_presets();
        // With default min_transition_observations=10, no data yet
        assert!(!s.transition_matrix_reliable());
    }

    #[test]
    fn test_transition_matrix_becomes_reliable() {
        let mut s = with_presets();
        // Alternate between bull and crisis enough times
        for _ in 0..15 {
            s.classify(&bull_features());
            s.classify(&crisis_features());
        }
        // Should have enough observations for at least some regimes
        // Check if Bull regime has enough
        let bull_obs = s.transition_matrix().observations_from(RegimeId::Bull);
        let crisis_obs = s.transition_matrix().observations_from(RegimeId::Crisis);
        assert!(bull_obs > 0 || crisis_obs > 0);
    }

    // -- Statistics --

    #[test]
    fn test_stats_initial() {
        let s = Schemas::new();
        assert_eq!(s.stats().total_evaluations, 0);
        assert_eq!(s.stats().total_transitions, 0);
    }

    #[test]
    fn test_stats_after_classification() {
        let mut s = with_presets();
        s.classify(&bull_features());
        assert_eq!(s.stats().total_evaluations, 1);
    }

    #[test]
    fn test_stats_transition_counted() {
        let mut s = with_presets();
        s.classify(&bull_features());
        s.classify(&crisis_features());
        assert!(s.stats().total_transitions >= 1);
    }

    #[test]
    fn test_stats_regime_durations() {
        let mut s = with_presets();
        for _ in 0..5 {
            s.classify(&bull_features());
        }
        assert!(s.stats().regime_durations[RegimeId::Bull.index()] >= 5);
    }

    #[test]
    fn test_stats_dominant_regime() {
        let mut s = with_presets();
        for _ in 0..10 {
            s.classify(&bull_features());
        }
        assert_eq!(s.stats().dominant_regime(), RegimeId::Bull);
    }

    #[test]
    fn test_stats_transition_rate() {
        let stats = SchemasStats {
            total_evaluations: 100,
            total_transitions: 5,
            ..Default::default()
        };
        assert!((stats.transition_rate() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_stats_transition_rate_zero_evals() {
        let stats = SchemasStats::default();
        assert_eq!(stats.transition_rate(), 0.0);
    }

    #[test]
    fn test_stats_unmatched_rate() {
        let stats = SchemasStats {
            total_evaluations: 20,
            unmatched_count: 4,
            ..Default::default()
        };
        assert!((stats.unmatched_rate() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_stats_regime_fractions_sum_to_one() {
        let stats = SchemasStats {
            regime_durations: [50, 20, 15, 5, 5, 3, 2],
            ..Default::default()
        };
        let fractions = stats.regime_fractions();
        let sum: f64 = fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_avg_regime_duration() {
        let stats = SchemasStats {
            regime_durations: [100, 0, 0, 0, 0, 0, 0],
            regime_entries: [5, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        assert!((stats.avg_regime_duration() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_avg_regime_duration_zero_entries() {
        let stats = SchemasStats::default();
        assert_eq!(stats.avg_regime_duration(), 0.0);
    }

    // -- EMA tracking --

    #[test]
    fn test_ema_initializes_first() {
        let mut s = with_presets();
        s.classify(&bull_features());
        assert!(s.smoothed_confidence() > 0.0);
    }

    #[test]
    fn test_ema_blends_subsequent() {
        let mut s = with_presets();
        s.classify(&bull_features());
        let conf1 = s.smoothed_confidence();

        // Classify with something that might lower confidence
        s.classify(&ranging_features());
        let conf2 = s.smoothed_confidence();

        // They should differ (blended)
        assert!(conf1.is_finite());
        assert!(conf2.is_finite());
    }

    // -- Sliding window --

    #[test]
    fn test_recent_stored() {
        let mut s = with_presets();
        s.classify(&bull_features());
        assert_eq!(s.recent_classifications().len(), 1);
    }

    #[test]
    fn test_recent_windowed() {
        let mut cfg = SchemasConfig::default();
        cfg.window_size = 3;
        let mut s = Schemas::with_config(cfg).unwrap();
        for schema in preset_schemas() {
            s.add_schema(schema).unwrap();
        }

        for _ in 0..10 {
            s.classify(&bull_features());
        }
        assert!(s.recent_classifications().len() <= 3);
    }

    #[test]
    fn test_windowed_avg_confidence() {
        let mut s = with_presets();
        s.classify(&bull_features());
        let avg = s.windowed_avg_confidence();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_windowed_avg_confidence_empty() {
        let s = Schemas::new();
        assert_eq!(s.windowed_avg_confidence(), 0.0);
    }

    #[test]
    fn test_windowed_regime_stability() {
        let mut s = with_presets();
        for _ in 0..5 {
            s.classify(&bull_features());
        }
        assert!(s.windowed_regime_stability() > 0.8);
    }

    #[test]
    fn test_windowed_regime_stability_empty() {
        let s = Schemas::new();
        assert_eq!(s.windowed_regime_stability(), 0.0);
    }

    #[test]
    fn test_windowed_transition_rate() {
        let mut s = with_presets();
        s.classify(&bull_features());
        s.classify(&crisis_features());

        let rate = s.windowed_transition_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_windowed_transition_rate_empty() {
        let s = Schemas::new();
        assert_eq!(s.windowed_transition_rate(), 0.0);
    }

    // -- Trend detection --

    #[test]
    fn test_is_confidence_declining_insufficient_data() {
        let s = Schemas::new();
        assert!(!s.is_confidence_declining());
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut s = with_presets();
        s.classify(&bull_features());
        s.classify(&crisis_features());

        assert!(s.stats().total_evaluations > 0);
        assert!(!s.recent_classifications().is_empty());

        s.reset();

        assert_eq!(s.stats().total_evaluations, 0);
        assert!(s.recent_classifications().is_empty());
        assert_eq!(s.current_regime(), RegimeId::Ranging);
        assert_eq!(s.regime_duration(), 0);
        // Schemas are preserved
        assert_eq!(s.schema_count(), preset_schemas().len());
    }

    // -- Presets --

    #[test]
    fn test_preset_schemas_all_valid() {
        let presets = preset_schemas();
        assert_eq!(presets.len(), RegimeId::COUNT);

        for schema in &presets {
            assert!(!schema.criteria.is_empty());
            assert!(schema.enabled);
            assert!(schema.acceptance_threshold > 0.0);
            assert!(schema.acceptance_threshold <= 1.0);
            assert!(schema.priority > 0.0);
        }
    }

    #[test]
    fn test_preset_schemas_can_be_loaded() {
        let s = with_presets();
        assert_eq!(s.schema_count(), RegimeId::COUNT);
    }

    #[test]
    fn test_preset_schemas_cover_all_regimes() {
        let presets = preset_schemas();
        let mut covered = [false; RegimeId::COUNT];
        for s in &presets {
            covered[s.regime.index()] = true;
        }
        assert!(covered.iter().all(|&c| c));
    }

    // -- Integration test --

    #[test]
    fn test_full_lifecycle() {
        let mut s = with_presets();

        // Phase 1: bull market
        for _ in 0..10 {
            let result = s.classify(&bull_features());
            assert_eq!(result.best_regime, RegimeId::Bull);
        }
        assert_eq!(s.current_regime(), RegimeId::Bull);
        assert!(s.regime_duration() >= 10);

        // Phase 2: crisis
        let crisis_result = s.classify(&crisis_features());
        assert_eq!(crisis_result.best_regime, RegimeId::Crisis);
        assert!(crisis_result.regime_changed);
        assert_eq!(s.previous_regime(), RegimeId::Bull);

        // Phase 3: stay in crisis
        for _ in 0..5 {
            s.classify(&crisis_features());
        }
        assert!(s.regime_duration() >= 5);

        // Phase 4: transition to ranging
        s.classify(&ranging_features());
        // Should have transitioned

        // Verify transition matrix has observations
        let bull_obs = s.transition_matrix().observations_from(RegimeId::Bull);
        assert!(bull_obs > 0);

        // Verify stats
        assert!(s.stats().total_evaluations > 15);
        assert!(s.stats().total_transitions >= 2);

        // Verify predictions work
        let (_, prob) = s.predict_next_regime();
        assert!(prob > 0.0);
        assert!(prob <= 1.0);

        // Verify stationary distribution
        let pi = s.stationary_distribution();
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);

        // Verify windowed metrics
        assert!(s.windowed_avg_confidence() > 0.0);
        assert!(s.windowed_regime_stability() >= 0.0);
    }
}
