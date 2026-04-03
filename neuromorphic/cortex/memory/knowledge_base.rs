//! Extracted long-term knowledge store
//!
//! Part of the Cortex region
//! Component: memory
//!
//! Stores and manages long-term accumulated knowledge about markets, including
//! learned patterns, strategy performance summaries, regime transition
//! probabilities, and asset characteristic profiles. Provides retrieval,
//! relevance scoring, and knowledge decay mechanisms so downstream components
//! can query the most pertinent knowledge for current conditions.
//!
//! Key features:
//! - Typed knowledge entries (pattern, strategy summary, regime transition, asset profile)
//! - Relevance scoring based on recency, frequency of access, and context similarity
//! - Configurable knowledge decay with half-life mechanism
//! - Regime transition probability matrix with Bayesian updates
//! - Strategy performance summaries with confidence intervals
//! - Pattern library with match-strength scoring
//! - EMA-smoothed tracking of knowledge utilization and quality
//! - Sliding window of recent queries for access pattern analysis
//! - Running statistics for audit and diagnostics

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Type of knowledge entry
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KnowledgeType {
    /// A learned market pattern (e.g. "mean-reversion after VIX spike")
    Pattern,
    /// Summary of a strategy's historical performance
    StrategySummary,
    /// Regime transition probability observation
    RegimeTransition,
    /// Characteristics of a specific asset or asset class
    AssetProfile,
    /// Correlation structure observation
    CorrelationStructure,
    /// Market microstructure observation (spread, depth, etc.)
    Microstructure,
    /// Custom knowledge type
    Custom(String),
}

/// Confidence level in a knowledge entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConfidenceLevel {
    /// Hypothesis — observed once or twice, untested
    Low,
    /// Moderate — observed multiple times, partially validated
    Medium,
    /// High — observed consistently, validated across regimes
    High,
    /// Established — long track record, extensively validated
    Established,
}

impl ConfidenceLevel {
    /// Numeric weight for aggregation (higher = more confident)
    pub fn weight(&self) -> f64 {
        match self {
            ConfidenceLevel::Low => 0.25,
            ConfidenceLevel::Medium => 0.50,
            ConfidenceLevel::High => 0.75,
            ConfidenceLevel::Established => 1.0,
        }
    }

    /// Promote confidence if observation count meets threshold
    pub fn promote(self, observations: u64) -> Self {
        match self {
            ConfidenceLevel::Low if observations >= 5 => ConfidenceLevel::Medium,
            ConfidenceLevel::Medium if observations >= 20 => ConfidenceLevel::High,
            ConfidenceLevel::High if observations >= 100 => ConfidenceLevel::Established,
            other => other,
        }
    }
}

/// A single knowledge entry in the knowledge base
#[derive(Debug, Clone)]
pub struct KnowledgeEntry {
    /// Unique identifier
    pub id: String,
    /// Human-readable label
    pub label: String,
    /// Type of knowledge
    pub knowledge_type: KnowledgeType,
    /// Confidence level
    pub confidence: ConfidenceLevel,
    /// Primary numeric value (interpretation depends on type)
    pub value: f64,
    /// Secondary value (e.g. standard deviation, p-value)
    pub uncertainty: f64,
    /// Number of observations that contributed to this entry
    pub observation_count: u64,
    /// Relevance score (computed, 0.0–1.0)
    pub relevance: f64,
    /// Number of times this entry has been accessed / queried
    pub access_count: u64,
    /// Step when this entry was last updated
    pub last_updated: u64,
    /// Step when this entry was last accessed
    pub last_accessed: u64,
    /// Tags for contextual matching (e.g. regime, asset class)
    pub tags: Vec<String>,
    /// Whether this entry is currently active (not decayed away)
    pub active: bool,
}

impl KnowledgeEntry {
    /// Create a new knowledge entry
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        knowledge_type: KnowledgeType,
        value: f64,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            knowledge_type,
            confidence: ConfidenceLevel::Low,
            value,
            uncertainty: 1.0,
            observation_count: 1,
            relevance: 0.5,
            access_count: 0,
            last_updated: 0,
            last_accessed: 0,
            tags: Vec::new(),
            active: true,
        }
    }

    /// Update this entry with a new observation using incremental mean/variance
    pub fn update(&mut self, new_value: f64, step: u64) {
        self.observation_count += 1;
        let n = self.observation_count as f64;

        // Incremental mean
        let old_mean = self.value;
        self.value = old_mean + (new_value - old_mean) / n;

        // Incremental variance (Welford's)
        if n > 1.0 {
            let delta = new_value - old_mean;
            let delta2 = new_value - self.value;
            // We track uncertainty as standard deviation
            let old_var = self.uncertainty * self.uncertainty * (n - 1.0);
            let new_var = old_var + delta * delta2;
            self.uncertainty = (new_var / n).sqrt();
        }

        self.last_updated = step;
        self.confidence = self.confidence.promote(self.observation_count);
    }

    /// Record an access to this entry
    pub fn record_access(&mut self, step: u64) {
        self.access_count += 1;
        self.last_accessed = step;
    }

    /// Age of this entry in steps since last update
    pub fn age(&self, current_step: u64) -> u64 {
        current_step.saturating_sub(self.last_updated)
    }

    /// Compute a recency score (1.0 = just updated, decays toward 0)
    pub fn recency_score(&self, current_step: u64, half_life: f64) -> f64 {
        let age = self.age(current_step) as f64;
        if half_life <= 0.0 {
            return if age == 0.0 { 1.0 } else { 0.0 };
        }
        (-age * (2.0_f64.ln()) / half_life).exp()
    }

    /// Compute a composite relevance score
    pub fn compute_relevance(
        &self,
        current_step: u64,
        half_life: f64,
        context_tags: &[String],
    ) -> f64 {
        let recency = self.recency_score(current_step, half_life);
        let confidence_w = self.confidence.weight();

        // Tag overlap score
        let tag_score = if context_tags.is_empty() || self.tags.is_empty() {
            0.5 // Neutral when no tags to compare
        } else {
            let overlap = self
                .tags
                .iter()
                .filter(|t| context_tags.contains(t))
                .count();
            overlap as f64 / self.tags.len().max(context_tags.len()) as f64
        };

        // Access frequency bonus (diminishing returns)
        let access_bonus = (self.access_count as f64).ln().max(0.0) / 10.0;

        // Weighted combination
        let raw =
            0.35 * recency + 0.30 * confidence_w + 0.25 * tag_score + 0.10 * access_bonus.min(0.3);
        raw.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Strategy performance summary
// ---------------------------------------------------------------------------

/// Summary of a strategy's historical performance
#[derive(Debug, Clone)]
pub struct StrategyPerformanceSummary {
    /// Strategy name
    pub strategy_name: String,
    /// Mean annualised return
    pub mean_return: f64,
    /// Standard deviation of return
    pub return_std: f64,
    /// Mean Sharpe ratio
    pub mean_sharpe: f64,
    /// Maximum drawdown observed
    pub max_drawdown: f64,
    /// Win rate (fraction of profitable periods)
    pub win_rate: f64,
    /// Number of evaluation periods
    pub periods: u64,
    /// Confidence interval half-width on mean return (95%)
    pub ci_half_width: f64,
    /// Best regime(s) for this strategy
    pub best_regimes: Vec<String>,
    /// Worst regime(s) for this strategy
    pub worst_regimes: Vec<String>,
}

impl StrategyPerformanceSummary {
    /// Create a new summary from initial observation
    pub fn new(strategy_name: impl Into<String>, return_obs: f64, sharpe: f64) -> Self {
        Self {
            strategy_name: strategy_name.into(),
            mean_return: return_obs,
            return_std: 0.0,
            mean_sharpe: sharpe,
            max_drawdown: 0.0,
            win_rate: if return_obs > 0.0 { 1.0 } else { 0.0 },
            periods: 1,
            ci_half_width: f64::INFINITY,
            best_regimes: Vec::new(),
            worst_regimes: Vec::new(),
        }
    }

    /// Update with a new observation period
    pub fn update(&mut self, return_obs: f64, sharpe: f64, drawdown: f64) {
        self.periods += 1;
        let n = self.periods as f64;

        // Incremental mean return
        let old_mean = self.mean_return;
        self.mean_return = old_mean + (return_obs - old_mean) / n;

        // Incremental std (Welford's)
        if n > 1.0 {
            let delta = return_obs - old_mean;
            let delta2 = return_obs - self.mean_return;
            let old_var = self.return_std * self.return_std * (n - 1.0);
            let new_var = old_var + delta * delta2;
            self.return_std = (new_var / n).sqrt();
        }

        // Incremental mean Sharpe
        self.mean_sharpe += (sharpe - self.mean_sharpe) / n;

        // Max drawdown
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }

        // Win rate
        let wins = self.win_rate * (n - 1.0) + if return_obs > 0.0 { 1.0 } else { 0.0 };
        self.win_rate = wins / n;

        // 95% CI half-width
        if n > 1.0 {
            self.ci_half_width = 1.96 * self.return_std / n.sqrt();
        }
    }

    /// Whether the strategy has a statistically significant positive return
    pub fn is_significant_positive(&self) -> bool {
        self.periods >= 10 && self.mean_return > self.ci_half_width
    }
}

// ---------------------------------------------------------------------------
// Regime transition matrix
// ---------------------------------------------------------------------------

/// Number of discrete regimes tracked
pub const NUM_REGIMES: usize = 5;

/// Regime transition probability matrix with Bayesian updating.
///
/// Rows = source regime, columns = destination regime.
/// Initialized with uniform priors.
#[derive(Debug, Clone)]
pub struct RegimeTransitionMatrix {
    /// Transition counts (row = from, col = to)
    counts: [[u64; NUM_REGIMES]; NUM_REGIMES],
    /// Prior pseudo-counts (Dirichlet prior)
    prior: f64,
    /// Total transitions observed
    total_transitions: u64,
}

impl Default for RegimeTransitionMatrix {
    fn default() -> Self {
        Self::new(1.0) // Uniform Dirichlet prior
    }
}

impl RegimeTransitionMatrix {
    /// Create with specified Dirichlet prior strength
    pub fn new(prior: f64) -> Self {
        Self {
            counts: [[0; NUM_REGIMES]; NUM_REGIMES],
            prior: prior.max(0.0),
            total_transitions: 0,
        }
    }

    /// Record an observed transition from one regime to another
    pub fn record_transition(&mut self, from: usize, to: usize) {
        if from < NUM_REGIMES && to < NUM_REGIMES {
            self.counts[from][to] += 1;
            self.total_transitions += 1;
        }
    }

    /// Get the posterior transition probability P(to | from)
    pub fn probability(&self, from: usize, to: usize) -> f64 {
        if from >= NUM_REGIMES || to >= NUM_REGIMES {
            return 0.0;
        }
        let row_total: f64 = self.counts[from]
            .iter()
            .map(|c| *c as f64 + self.prior)
            .sum();
        if row_total <= 0.0 {
            return 1.0 / NUM_REGIMES as f64;
        }
        (self.counts[from][to] as f64 + self.prior) / row_total
    }

    /// Get the full transition probability row for a given source regime
    pub fn transition_row(&self, from: usize) -> [f64; NUM_REGIMES] {
        let mut row = [0.0; NUM_REGIMES];
        for to in 0..NUM_REGIMES {
            row[to] = self.probability(from, to);
        }
        row
    }

    /// Most likely destination regime from a given source
    pub fn most_likely_next(&self, from: usize) -> usize {
        let row = self.transition_row(from);
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Total transitions observed
    pub fn total_transitions(&self) -> u64 {
        self.total_transitions
    }

    /// Transition counts for a specific (from, to) pair
    pub fn count(&self, from: usize, to: usize) -> u64 {
        if from < NUM_REGIMES && to < NUM_REGIMES {
            self.counts[from][to]
        } else {
            0
        }
    }

    /// Stationary distribution (eigenvector of transition matrix, via power iteration)
    pub fn stationary_distribution(&self) -> [f64; NUM_REGIMES] {
        let mut dist = [1.0 / NUM_REGIMES as f64; NUM_REGIMES];

        // Power iteration (50 steps is plenty for 5x5 matrix)
        for _ in 0..50 {
            let mut next = [0.0; NUM_REGIMES];
            for to in 0..NUM_REGIMES {
                for from in 0..NUM_REGIMES {
                    next[to] += dist[from] * self.probability(from, to);
                }
            }
            // Normalise
            let sum: f64 = next.iter().sum();
            if sum > 1e-15 {
                for v in &mut next {
                    *v /= sum;
                }
            }
            dist = next;
        }

        dist
    }

    /// Reset all counts (keep prior)
    pub fn reset(&mut self) {
        self.counts = [[0; NUM_REGIMES]; NUM_REGIMES];
        self.total_transitions = 0;
    }
}

// ---------------------------------------------------------------------------
// Pattern entry
// ---------------------------------------------------------------------------

/// A learned market pattern
#[derive(Debug, Clone)]
pub struct PatternEntry {
    /// Pattern name
    pub name: String,
    /// Description of the pattern
    pub description: String,
    /// Feature vector that characterizes the pattern
    pub feature_vector: Vec<f64>,
    /// Historical accuracy (fraction of times the pattern prediction was correct)
    pub accuracy: f64,
    /// Number of times the pattern was observed
    pub observation_count: u64,
    /// Average return following pattern detection
    pub avg_subsequent_return: f64,
    /// Standard deviation of return following pattern
    pub return_std: f64,
    /// Applicable regime tags
    pub regime_tags: Vec<String>,
}

impl PatternEntry {
    /// Create a new pattern entry
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        feature_vector: Vec<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            feature_vector,
            accuracy: 0.0,
            observation_count: 0,
            avg_subsequent_return: 0.0,
            return_std: 0.0,
            regime_tags: Vec::new(),
        }
    }

    /// Compute match strength against an observation feature vector (cosine similarity)
    pub fn match_strength(&self, observation: &[f64]) -> f64 {
        if self.feature_vector.len() != observation.len() {
            return -1.0;
        }
        let dot: f64 = self
            .feature_vector
            .iter()
            .zip(observation.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f64 = self
            .feature_vector
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let norm_b: f64 = observation.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-15 || norm_b < 1e-15 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Update pattern statistics with a new observation
    pub fn update_stats(&mut self, prediction_correct: bool, subsequent_return: f64) {
        self.observation_count += 1;
        let n = self.observation_count as f64;

        // Incremental accuracy
        let hit = if prediction_correct { 1.0 } else { 0.0 };
        self.accuracy += (hit - self.accuracy) / n;

        // Incremental mean return
        let old_mean = self.avg_subsequent_return;
        self.avg_subsequent_return = old_mean + (subsequent_return - old_mean) / n;

        // Incremental std
        if n > 1.0 {
            let delta = subsequent_return - old_mean;
            let delta2 = subsequent_return - self.avg_subsequent_return;
            let old_var = self.return_std * self.return_std * (n - 1.0);
            let new_var = old_var + delta * delta2;
            self.return_std = (new_var / n).sqrt();
        }
    }

    /// Information ratio of the pattern (return / risk)
    pub fn information_ratio(&self) -> f64 {
        if self.return_std.abs() < 1e-15 {
            return 0.0;
        }
        self.avg_subsequent_return / self.return_std
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the KnowledgeBase engine
#[derive(Debug, Clone)]
pub struct KnowledgeBaseConfig {
    /// Maximum number of knowledge entries
    pub max_entries: usize,
    /// Half-life for recency decay (in steps)
    pub recency_half_life: f64,
    /// Minimum relevance score below which entries are pruned
    pub min_relevance: f64,
    /// Whether to auto-prune low-relevance entries
    pub auto_prune: bool,
    /// EMA decay factor for quality metrics (0 < decay < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent queries
    pub window_size: usize,
    /// Dirichlet prior strength for regime transition matrix
    pub transition_prior: f64,
    /// Minimum observations for a pattern to be considered reliable
    pub min_pattern_observations: u64,
}

impl Default for KnowledgeBaseConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            recency_half_life: 500.0,
            min_relevance: 0.05,
            auto_prune: true,
            ema_decay: 0.1,
            window_size: 100,
            transition_prior: 1.0,
            min_pattern_observations: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Query record
// ---------------------------------------------------------------------------

/// Record of a knowledge query
#[derive(Debug, Clone)]
pub struct QueryRecord {
    /// Tags used in the query
    pub context_tags: Vec<String>,
    /// Knowledge type filter (if any)
    pub type_filter: Option<KnowledgeType>,
    /// Number of results returned
    pub results_count: usize,
    /// Average relevance of results
    pub avg_relevance: f64,
    /// Step when query was made
    pub step: u64,
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the knowledge base
#[derive(Debug, Clone)]
pub struct KnowledgeBaseStats {
    /// Total entries added
    pub total_entries_added: u64,
    /// Total entries pruned
    pub total_entries_pruned: u64,
    /// Total queries executed
    pub total_queries: u64,
    /// Total updates to existing entries
    pub total_updates: u64,
    /// EMA of average query relevance
    pub ema_avg_relevance: f64,
    /// EMA of query result count
    pub ema_result_count: f64,
    /// EMA of knowledge base utilization (entries / max_entries)
    pub ema_utilization: f64,
    /// Best single-entry relevance seen in a query
    pub peak_relevance: f64,
    /// Count of entries by confidence level
    pub confidence_counts: [u64; 4], // Low, Medium, High, Established
}

impl Default for KnowledgeBaseStats {
    fn default() -> Self {
        Self {
            total_entries_added: 0,
            total_entries_pruned: 0,
            total_queries: 0,
            total_updates: 0,
            ema_avg_relevance: 0.0,
            ema_result_count: 0.0,
            ema_utilization: 0.0,
            peak_relevance: 0.0,
            confidence_counts: [0; 4],
        }
    }
}

impl KnowledgeBaseStats {
    /// Prune rate (pruned / added)
    pub fn prune_rate(&self) -> f64 {
        if self.total_entries_added == 0 {
            return 0.0;
        }
        self.total_entries_pruned as f64 / self.total_entries_added as f64
    }

    /// Average results per query
    pub fn avg_results_per_query(&self) -> f64 {
        if self.total_queries == 0 {
            return 0.0;
        }
        self.ema_result_count
    }

    fn confidence_index(level: ConfidenceLevel) -> usize {
        match level {
            ConfidenceLevel::Low => 0,
            ConfidenceLevel::Medium => 1,
            ConfidenceLevel::High => 2,
            ConfidenceLevel::Established => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// KnowledgeBase Engine
// ---------------------------------------------------------------------------

/// Long-term knowledge store.
///
/// Maintains a collection of typed knowledge entries with relevance scoring,
/// decay, and pruning. Also holds a regime transition matrix and a pattern
/// library for specialized knowledge.
pub struct KnowledgeBase {
    config: KnowledgeBaseConfig,
    /// General knowledge entries
    entries: Vec<KnowledgeEntry>,
    /// Strategy performance summaries (keyed by strategy name, stored in a Vec)
    strategy_summaries: Vec<StrategyPerformanceSummary>,
    /// Regime transition matrix
    transitions: RegimeTransitionMatrix,
    /// Pattern library
    patterns: Vec<PatternEntry>,
    /// Current step counter
    step: u64,
    /// EMA initialized flag
    ema_initialized: bool,
    /// Sliding window of recent queries
    recent_queries: VecDeque<QueryRecord>,
    /// Running statistics
    stats: KnowledgeBaseStats,
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeBase {
    /// Create a new knowledge base with default configuration
    pub fn new() -> Self {
        Self {
            config: KnowledgeBaseConfig::default(),
            entries: Vec::new(),
            strategy_summaries: Vec::new(),
            transitions: RegimeTransitionMatrix::default(),
            patterns: Vec::new(),
            step: 0,
            ema_initialized: false,
            recent_queries: VecDeque::new(),
            stats: KnowledgeBaseStats::default(),
        }
    }

    /// Create with explicit configuration
    pub fn with_config(config: KnowledgeBaseConfig) -> Result<Self> {
        if config.max_entries == 0 {
            return Err(Error::InvalidInput("max_entries must be > 0".into()));
        }
        if config.recency_half_life <= 0.0 {
            return Err(Error::InvalidInput("recency_half_life must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.transition_prior < 0.0 {
            return Err(Error::InvalidInput("transition_prior must be >= 0".into()));
        }
        Ok(Self {
            transitions: RegimeTransitionMatrix::new(config.transition_prior),
            config,
            entries: Vec::new(),
            strategy_summaries: Vec::new(),
            patterns: Vec::new(),
            step: 0,
            ema_initialized: false,
            recent_queries: VecDeque::new(),
            stats: KnowledgeBaseStats::default(),
        })
    }

    /// Main processing function (no-op entry point for trait conformance)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Step management
    // -----------------------------------------------------------------------

    /// Advance the internal step counter
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Current step
    pub fn current_step(&self) -> u64 {
        self.step
    }

    // -----------------------------------------------------------------------
    // Entry management
    // -----------------------------------------------------------------------

    /// Add a knowledge entry
    pub fn add_entry(&mut self, mut entry: KnowledgeEntry) -> Result<()> {
        if self.entries.iter().any(|e| e.id == entry.id) {
            return Err(Error::InvalidInput(format!(
                "entry with id '{}' already exists",
                entry.id
            )));
        }
        if entry.id.is_empty() {
            return Err(Error::InvalidInput("entry id must not be empty".into()));
        }

        entry.last_updated = self.step;
        entry.active = true;

        // Auto-prune if at capacity
        if self.entries.len() >= self.config.max_entries {
            if self.config.auto_prune {
                self.prune_least_relevant(1);
            } else {
                return Err(Error::ResourceExhausted(format!(
                    "maximum entries ({}) reached",
                    self.config.max_entries
                )));
            }
        }

        // Track confidence
        self.stats.confidence_counts[KnowledgeBaseStats::confidence_index(entry.confidence)] += 1;
        self.stats.total_entries_added += 1;
        self.entries.push(entry);
        Ok(())
    }

    /// Update an existing entry with a new observation
    pub fn update_entry(&mut self, id: &str, new_value: f64) -> Result<()> {
        let step = self.step;
        let entry = self
            .entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or_else(|| Error::NotFound(format!("entry '{}' not found", id)))?;

        let old_conf = entry.confidence;
        entry.update(new_value, step);
        let new_conf = entry.confidence;

        // Update confidence counts if promoted
        if old_conf != new_conf {
            let old_idx = KnowledgeBaseStats::confidence_index(old_conf);
            let new_idx = KnowledgeBaseStats::confidence_index(new_conf);
            if self.stats.confidence_counts[old_idx] > 0 {
                self.stats.confidence_counts[old_idx] -= 1;
            }
            self.stats.confidence_counts[new_idx] += 1;
        }

        self.stats.total_updates += 1;
        Ok(())
    }

    /// Remove an entry by ID
    pub fn remove_entry(&mut self, id: &str) -> Result<()> {
        let idx = self
            .entries
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| Error::NotFound(format!("entry '{}' not found", id)))?;
        let conf = self.entries[idx].confidence;
        let cidx = KnowledgeBaseStats::confidence_index(conf);
        if self.stats.confidence_counts[cidx] > 0 {
            self.stats.confidence_counts[cidx] -= 1;
        }
        self.entries.remove(idx);
        Ok(())
    }

    /// Get an entry by ID
    pub fn get_entry(&self, id: &str) -> Option<&KnowledgeEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Get a mutable entry by ID (records access)
    pub fn get_entry_mut(&mut self, id: &str) -> Option<&mut KnowledgeEntry> {
        let step = self.step;
        let entry = self.entries.iter_mut().find(|e| e.id == id)?;
        entry.record_access(step);
        Some(entry)
    }

    /// Number of active entries
    pub fn entry_count(&self) -> usize {
        self.entries.iter().filter(|e| e.active).count()
    }

    /// Total entries (including inactive)
    pub fn total_entries(&self) -> usize {
        self.entries.len()
    }

    /// Prune the N least relevant entries
    fn prune_least_relevant(&mut self, count: usize) {
        if count == 0 || self.entries.is_empty() {
            return;
        }

        // Compute relevance for all entries
        let step = self.step;
        let half_life = self.config.recency_half_life;
        let empty_tags: Vec<String> = Vec::new();
        for entry in &mut self.entries {
            entry.relevance = entry.compute_relevance(step, half_life, &empty_tags);
        }

        // Sort by relevance ascending
        self.entries.sort_by(|a, b| {
            a.relevance
                .partial_cmp(&b.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove the bottom `count` entries
        let to_remove = count.min(self.entries.len());
        for _ in 0..to_remove {
            if let Some(entry) = self.entries.first() {
                let cidx = KnowledgeBaseStats::confidence_index(entry.confidence);
                if self.stats.confidence_counts[cidx] > 0 {
                    self.stats.confidence_counts[cidx] -= 1;
                }
            }
            self.entries.remove(0);
            self.stats.total_entries_pruned += 1;
        }
    }

    /// Prune all entries below the minimum relevance threshold
    pub fn prune_below_threshold(&mut self) {
        let step = self.step;
        let half_life = self.config.recency_half_life;
        let min_rel = self.config.min_relevance;
        let empty_tags: Vec<String> = Vec::new();

        let mut to_remove = Vec::new();
        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.relevance = entry.compute_relevance(step, half_life, &empty_tags);
            if entry.relevance < min_rel {
                to_remove.push(i);
            }
        }

        // Remove in reverse order to preserve indices
        for i in to_remove.into_iter().rev() {
            let conf = self.entries[i].confidence;
            let cidx = KnowledgeBaseStats::confidence_index(conf);
            if self.stats.confidence_counts[cidx] > 0 {
                self.stats.confidence_counts[cidx] -= 1;
            }
            self.entries.remove(i);
            self.stats.total_entries_pruned += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Querying
    // -----------------------------------------------------------------------

    /// Query the knowledge base for entries matching the given context.
    ///
    /// Returns entries sorted by relevance (highest first), limited to `max_results`.
    pub fn query(
        &mut self,
        context_tags: &[String],
        type_filter: Option<&KnowledgeType>,
        max_results: usize,
    ) -> Vec<&KnowledgeEntry> {
        let step = self.step;
        let half_life = self.config.recency_half_life;

        // Compute relevance and record access
        for entry in &mut self.entries {
            if !entry.active {
                continue;
            }
            entry.relevance = entry.compute_relevance(step, half_life, context_tags);
        }

        // Filter and sort
        let mut candidates: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.active && type_filter.map(|t| e.knowledge_type == *t).unwrap_or(true)
            })
            .map(|(i, _)| i)
            .collect();

        candidates.sort_by(|&a, &b| {
            self.entries[b]
                .relevance
                .partial_cmp(&self.entries[a].relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(max_results);

        // Record access for returned entries
        for &idx in &candidates {
            self.entries[idx].record_access(step);
        }

        // Compute query stats
        let avg_relevance = if candidates.is_empty() {
            0.0
        } else {
            candidates
                .iter()
                .map(|&i| self.entries[i].relevance)
                .sum::<f64>()
                / candidates.len() as f64
        };

        let results_count = candidates.len();

        // Track peak relevance
        if let Some(&first_idx) = candidates.first() {
            let top_rel = self.entries[first_idx].relevance;
            if top_rel > self.stats.peak_relevance {
                self.stats.peak_relevance = top_rel;
            }
        }

        // Update stats
        self.stats.total_queries += 1;
        let alpha = self.config.ema_decay;
        let utilization = self.entries.len() as f64 / self.config.max_entries as f64;

        if !self.ema_initialized {
            self.stats.ema_avg_relevance = avg_relevance;
            self.stats.ema_result_count = results_count as f64;
            self.stats.ema_utilization = utilization;
            self.ema_initialized = true;
        } else {
            self.stats.ema_avg_relevance =
                alpha * avg_relevance + (1.0 - alpha) * self.stats.ema_avg_relevance;
            self.stats.ema_result_count =
                alpha * results_count as f64 + (1.0 - alpha) * self.stats.ema_result_count;
            self.stats.ema_utilization =
                alpha * utilization + (1.0 - alpha) * self.stats.ema_utilization;
        }

        // Record query in sliding window
        let qr = QueryRecord {
            context_tags: context_tags.to_vec(),
            type_filter: type_filter.cloned(),
            results_count,
            avg_relevance,
            step,
        };
        if self.recent_queries.len() >= self.config.window_size {
            self.recent_queries.pop_front();
        }
        self.recent_queries.push_back(qr);

        // Return references
        candidates.iter().map(|&i| &self.entries[i]).collect()
    }

    /// Find the most relevant single entry for a given context
    pub fn most_relevant(
        &mut self,
        context_tags: &[String],
        type_filter: Option<&KnowledgeType>,
    ) -> Option<&KnowledgeEntry> {
        let results = self.query(context_tags, type_filter, 1);
        // We need to return a reference that outlives the borrow — query already
        // computed relevance and updated access, so we can just find the best
        if results.is_empty() {
            return None;
        }
        let id = results[0].id.clone();
        self.entries.iter().find(|e| e.id == id)
    }

    // -----------------------------------------------------------------------
    // Strategy summaries
    // -----------------------------------------------------------------------

    /// Add or update a strategy performance summary
    pub fn update_strategy_summary(
        &mut self,
        strategy_name: &str,
        return_obs: f64,
        sharpe: f64,
        drawdown: f64,
    ) {
        if let Some(summary) = self
            .strategy_summaries
            .iter_mut()
            .find(|s| s.strategy_name == strategy_name)
        {
            summary.update(return_obs, sharpe, drawdown);
        } else {
            let mut summary = StrategyPerformanceSummary::new(strategy_name, return_obs, sharpe);
            if drawdown > summary.max_drawdown {
                summary.max_drawdown = drawdown;
            }
            self.strategy_summaries.push(summary);
        }
    }

    /// Get a strategy summary by name
    pub fn get_strategy_summary(&self, name: &str) -> Option<&StrategyPerformanceSummary> {
        self.strategy_summaries
            .iter()
            .find(|s| s.strategy_name == name)
    }

    /// Get all strategy summaries
    pub fn strategy_summaries(&self) -> &[StrategyPerformanceSummary] {
        &self.strategy_summaries
    }

    /// Get strategies ranked by mean Sharpe ratio (descending)
    pub fn ranked_strategies(&self) -> Vec<&StrategyPerformanceSummary> {
        let mut sorted: Vec<&StrategyPerformanceSummary> = self.strategy_summaries.iter().collect();
        sorted.sort_by(|a, b| {
            b.mean_sharpe
                .partial_cmp(&a.mean_sharpe)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    // -----------------------------------------------------------------------
    // Regime transitions
    // -----------------------------------------------------------------------

    /// Record a regime transition observation
    pub fn record_regime_transition(&mut self, from: usize, to: usize) {
        self.transitions.record_transition(from, to);
    }

    /// Get regime transition probability
    pub fn regime_transition_probability(&self, from: usize, to: usize) -> f64 {
        self.transitions.probability(from, to)
    }

    /// Get the full transition row for a given source regime
    pub fn regime_transition_row(&self, from: usize) -> [f64; NUM_REGIMES] {
        self.transitions.transition_row(from)
    }

    /// Most likely next regime given current
    pub fn most_likely_next_regime(&self, from: usize) -> usize {
        self.transitions.most_likely_next(from)
    }

    /// Get the regime transition matrix
    pub fn transition_matrix(&self) -> &RegimeTransitionMatrix {
        &self.transitions
    }

    /// Get the stationary distribution of regimes
    pub fn stationary_distribution(&self) -> [f64; NUM_REGIMES] {
        self.transitions.stationary_distribution()
    }

    // -----------------------------------------------------------------------
    // Patterns
    // -----------------------------------------------------------------------

    /// Add a pattern to the library
    pub fn add_pattern(&mut self, pattern: PatternEntry) -> Result<()> {
        if pattern.name.is_empty() {
            return Err(Error::InvalidInput("pattern name must not be empty".into()));
        }
        if self.patterns.iter().any(|p| p.name == pattern.name) {
            return Err(Error::InvalidInput(format!(
                "pattern '{}' already exists",
                pattern.name
            )));
        }
        self.patterns.push(pattern);
        Ok(())
    }

    /// Get a pattern by name
    pub fn get_pattern(&self, name: &str) -> Option<&PatternEntry> {
        self.patterns.iter().find(|p| p.name == name)
    }

    /// Get a mutable pattern by name
    pub fn get_pattern_mut(&mut self, name: &str) -> Option<&mut PatternEntry> {
        self.patterns.iter_mut().find(|p| p.name == name)
    }

    /// Find patterns matching an observation, sorted by match strength descending
    pub fn match_patterns(
        &self,
        observation: &[f64],
        min_strength: f64,
    ) -> Vec<(&PatternEntry, f64)> {
        let mut matches: Vec<(&PatternEntry, f64)> = self
            .patterns
            .iter()
            .filter_map(|p| {
                let strength = p.match_strength(observation);
                if strength >= min_strength {
                    Some((p, strength))
                } else {
                    None
                }
            })
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Number of patterns in the library
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Get reliable patterns (above min observation threshold)
    pub fn reliable_patterns(&self) -> Vec<&PatternEntry> {
        self.patterns
            .iter()
            .filter(|p| p.observation_count >= self.config.min_pattern_observations)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Running statistics
    pub fn stats(&self) -> &KnowledgeBaseStats {
        &self.stats
    }

    /// Configuration
    pub fn config(&self) -> &KnowledgeBaseConfig {
        &self.config
    }

    /// Recent queries (sliding window)
    pub fn recent_queries(&self) -> &VecDeque<QueryRecord> {
        &self.recent_queries
    }

    /// EMA-smoothed average query relevance
    pub fn smoothed_avg_relevance(&self) -> f64 {
        self.stats.ema_avg_relevance
    }

    /// EMA-smoothed result count
    pub fn smoothed_result_count(&self) -> f64 {
        self.stats.ema_result_count
    }

    /// EMA-smoothed utilization
    pub fn smoothed_utilization(&self) -> f64 {
        self.stats.ema_utilization
    }

    /// Windowed average relevance
    pub fn windowed_avg_relevance(&self) -> f64 {
        if self.recent_queries.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent_queries.iter().map(|q| q.avg_relevance).sum();
        sum / self.recent_queries.len() as f64
    }

    /// Windowed average result count
    pub fn windowed_avg_result_count(&self) -> f64 {
        if self.recent_queries.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .recent_queries
            .iter()
            .map(|q| q.results_count as f64)
            .sum();
        sum / self.recent_queries.len() as f64
    }

    /// Whether knowledge quality is improving (avg relevance trending up)
    pub fn is_quality_improving(&self) -> bool {
        let n = self.recent_queries.len();
        if n < 4 {
            return false;
        }
        let mid = n / 2;
        let first_avg: f64 = self
            .recent_queries
            .iter()
            .take(mid)
            .map(|q| q.avg_relevance)
            .sum::<f64>()
            / mid as f64;
        let second_avg: f64 = self
            .recent_queries
            .iter()
            .skip(mid)
            .map(|q| q.avg_relevance)
            .sum::<f64>()
            / (n - mid) as f64;
        second_avg > first_avg * 1.05
    }

    /// Reset all state (entries, patterns, transitions, stats)
    pub fn reset(&mut self) {
        self.entries.clear();
        self.strategy_summaries.clear();
        self.transitions.reset();
        self.patterns.clear();
        self.step = 0;
        self.ema_initialized = false;
        self.recent_queries.clear();
        self.stats = KnowledgeBaseStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn make_entry(id: &str, kt: KnowledgeType) -> KnowledgeEntry {
        let mut e = KnowledgeEntry::new(id, format!("Label {}", id), kt, 0.5);
        e.tags = vec!["bull".into(), "crypto".into()];
        e
    }

    fn make_pattern(name: &str, features: Vec<f64>) -> PatternEntry {
        PatternEntry::new(name, format!("Pattern {}", name), features)
    }

    // -- Construction --

    #[test]
    fn test_new_default() {
        let kb = KnowledgeBase::new();
        assert_eq!(kb.entry_count(), 0);
        assert_eq!(kb.total_entries(), 0);
        assert_eq!(kb.current_step(), 0);
        assert!(kb.process().is_ok());
    }

    #[test]
    fn test_with_config() {
        let kb = KnowledgeBase::with_config(KnowledgeBaseConfig::default());
        assert!(kb.is_ok());
    }

    #[test]
    fn test_invalid_max_entries_zero() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.max_entries = 0;
        assert!(KnowledgeBase::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_half_life_zero() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.recency_half_life = 0.0;
        assert!(KnowledgeBase::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_zero() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.ema_decay = 0.0;
        assert!(KnowledgeBase::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_one() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.ema_decay = 1.0;
        assert!(KnowledgeBase::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_window_zero() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.window_size = 0;
        assert!(KnowledgeBase::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_prior_negative() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.transition_prior = -1.0;
        assert!(KnowledgeBase::with_config(cfg).is_err());
    }

    // -- Entry management --

    #[test]
    fn test_add_entry() {
        let mut kb = KnowledgeBase::new();
        let e = make_entry("e1", KnowledgeType::Pattern);
        assert!(kb.add_entry(e).is_ok());
        assert_eq!(kb.entry_count(), 1);
    }

    #[test]
    fn test_add_duplicate_entry() {
        let mut kb = KnowledgeBase::new();
        let e1 = make_entry("dup", KnowledgeType::Pattern);
        let e2 = make_entry("dup", KnowledgeType::StrategySummary);
        assert!(kb.add_entry(e1).is_ok());
        assert!(kb.add_entry(e2).is_err());
    }

    #[test]
    fn test_add_entry_empty_id() {
        let mut kb = KnowledgeBase::new();
        let e = KnowledgeEntry::new("", "label", KnowledgeType::Pattern, 0.5);
        assert!(kb.add_entry(e).is_err());
    }

    #[test]
    fn test_add_entry_auto_prune_at_capacity() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.max_entries = 2;
        cfg.auto_prune = true;
        let mut kb = KnowledgeBase::with_config(cfg).unwrap();

        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        kb.add_entry(make_entry("e2", KnowledgeType::Pattern))
            .unwrap();
        // Third entry should trigger auto-prune
        assert!(
            kb.add_entry(make_entry("e3", KnowledgeType::Pattern))
                .is_ok()
        );
        assert_eq!(kb.total_entries(), 2);
        assert!(kb.stats().total_entries_pruned > 0);
    }

    #[test]
    fn test_add_entry_no_prune_at_capacity() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.max_entries = 1;
        cfg.auto_prune = false;
        let mut kb = KnowledgeBase::with_config(cfg).unwrap();

        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        assert!(
            kb.add_entry(make_entry("e2", KnowledgeType::Pattern))
                .is_err()
        );
    }

    #[test]
    fn test_update_entry() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        assert!(kb.update_entry("e1", 0.8).is_ok());
        let e = kb.get_entry("e1").unwrap();
        assert_eq!(e.observation_count, 2);
        assert!(kb.stats().total_updates > 0);
    }

    #[test]
    fn test_update_entry_not_found() {
        let mut kb = KnowledgeBase::new();
        assert!(kb.update_entry("nope", 0.5).is_err());
    }

    #[test]
    fn test_remove_entry() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        assert_eq!(kb.total_entries(), 1);
        assert!(kb.remove_entry("e1").is_ok());
        assert_eq!(kb.total_entries(), 0);
    }

    #[test]
    fn test_remove_entry_not_found() {
        let mut kb = KnowledgeBase::new();
        assert!(kb.remove_entry("nope").is_err());
    }

    #[test]
    fn test_get_entry() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        assert!(kb.get_entry("e1").is_some());
        assert!(kb.get_entry("nope").is_none());
    }

    #[test]
    fn test_get_entry_mut_records_access() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        {
            let e = kb.get_entry_mut("e1").unwrap();
            assert_eq!(e.access_count, 1);
        }
        {
            let e = kb.get_entry_mut("e1").unwrap();
            assert_eq!(e.access_count, 2);
        }
    }

    // -- Knowledge entry methods --

    #[test]
    fn test_entry_update_incremental_mean() {
        let mut e = KnowledgeEntry::new("t", "test", KnowledgeType::Pattern, 10.0);
        e.update(20.0, 1);
        assert!((e.value - 15.0).abs() < 1e-10);
        assert_eq!(e.observation_count, 2);
    }

    #[test]
    fn test_entry_update_confidence_promotes() {
        let mut e = KnowledgeEntry::new("t", "test", KnowledgeType::Pattern, 0.5);
        assert_eq!(e.confidence, ConfidenceLevel::Low);

        for i in 1..6 {
            e.update(0.5, i);
        }
        assert_eq!(e.confidence, ConfidenceLevel::Medium);
    }

    #[test]
    fn test_entry_recency_score_zero_age() {
        let e = KnowledgeEntry::new("t", "test", KnowledgeType::Pattern, 0.5);
        assert!((e.recency_score(0, 100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entry_recency_score_decays() {
        let mut e = KnowledgeEntry::new("t", "test", KnowledgeType::Pattern, 0.5);
        e.last_updated = 0;
        let r100 = e.recency_score(100, 100.0);
        let r200 = e.recency_score(200, 100.0);
        assert!(r100 > r200);
        // At half-life, recency should be ~0.5
        assert!((r100 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_entry_compute_relevance_bounded() {
        let e = make_entry("t", KnowledgeType::Pattern);
        let tags = vec!["bull".into(), "crypto".into()];
        let rel = e.compute_relevance(0, 100.0, &tags);
        assert!((0.0..=1.0).contains(&rel));
    }

    #[test]
    fn test_entry_compute_relevance_matching_tags_higher() {
        let e = make_entry("t", KnowledgeType::Pattern);
        let matching = vec!["bull".into(), "crypto".into()];
        let non_matching = vec!["bear".into(), "equity".into()];

        let rel_match = e.compute_relevance(0, 100.0, &matching);
        let rel_no_match = e.compute_relevance(0, 100.0, &non_matching);
        assert!(rel_match > rel_no_match);
    }

    #[test]
    fn test_entry_age() {
        let mut e = KnowledgeEntry::new("t", "test", KnowledgeType::Pattern, 0.5);
        e.last_updated = 10;
        assert_eq!(e.age(50), 40);
    }

    // -- Confidence level --

    #[test]
    fn test_confidence_ordering() {
        assert!(ConfidenceLevel::Low < ConfidenceLevel::Medium);
        assert!(ConfidenceLevel::Medium < ConfidenceLevel::High);
        assert!(ConfidenceLevel::High < ConfidenceLevel::Established);
    }

    #[test]
    fn test_confidence_weight() {
        assert!(ConfidenceLevel::Established.weight() > ConfidenceLevel::High.weight());
        assert!(ConfidenceLevel::High.weight() > ConfidenceLevel::Medium.weight());
        assert!(ConfidenceLevel::Medium.weight() > ConfidenceLevel::Low.weight());
    }

    #[test]
    fn test_confidence_promote() {
        assert_eq!(ConfidenceLevel::Low.promote(3), ConfidenceLevel::Low);
        assert_eq!(ConfidenceLevel::Low.promote(5), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::Medium.promote(20), ConfidenceLevel::High);
        assert_eq!(
            ConfidenceLevel::High.promote(100),
            ConfidenceLevel::Established
        );
        assert_eq!(
            ConfidenceLevel::Established.promote(1000),
            ConfidenceLevel::Established
        );
    }

    // -- Querying --

    #[test]
    fn test_query_empty_kb() {
        let mut kb = KnowledgeBase::new();
        let tags = vec!["bull".into()];
        let results = kb.query(&tags, None, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_returns_results() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        kb.add_entry(make_entry("e2", KnowledgeType::StrategySummary))
            .unwrap();

        let tags = vec!["bull".into()];
        let results = kb.query(&tags, None, 10);
        assert_eq!(results.len(), 2);
        assert!(kb.stats().total_queries > 0);
    }

    #[test]
    fn test_query_type_filter() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        kb.add_entry(make_entry("e2", KnowledgeType::StrategySummary))
            .unwrap();

        let tags = vec!["bull".into()];
        let results = kb.query(&tags, Some(&KnowledgeType::Pattern), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].knowledge_type, KnowledgeType::Pattern);
    }

    #[test]
    fn test_query_max_results() {
        let mut kb = KnowledgeBase::new();
        for i in 0..10 {
            kb.add_entry(make_entry(&format!("e{}", i), KnowledgeType::Pattern))
                .unwrap();
        }

        let tags = vec!["bull".into()];
        let results = kb.query(&tags, None, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_sorted_by_relevance() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        kb.add_entry(make_entry("e2", KnowledgeType::Pattern))
            .unwrap();

        let tags = vec!["bull".into()];
        let results = kb.query(&tags, None, 10);

        for i in 1..results.len() {
            assert!(results[i - 1].relevance >= results[i].relevance);
        }
    }

    #[test]
    fn test_query_records_access() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        let tags: Vec<String> = Vec::new();
        kb.query(&tags, None, 10);

        let e = kb.get_entry("e1").unwrap();
        assert!(e.access_count > 0);
    }

    #[test]
    fn test_query_stats_updated() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        let tags: Vec<String> = Vec::new();
        kb.query(&tags, None, 10);

        assert_eq!(kb.stats().total_queries, 1);
        assert!(kb.smoothed_avg_relevance() >= 0.0);
    }

    #[test]
    fn test_query_window() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        let tags: Vec<String> = Vec::new();
        kb.query(&tags, None, 10);
        assert_eq!(kb.recent_queries().len(), 1);
    }

    #[test]
    fn test_most_relevant() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        kb.add_entry(make_entry("e2", KnowledgeType::StrategySummary))
            .unwrap();

        let tags = vec!["bull".into()];
        let result = kb.most_relevant(&tags, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_most_relevant_empty() {
        let mut kb = KnowledgeBase::new();
        let tags = vec!["bull".into()];
        assert!(kb.most_relevant(&tags, None).is_none());
    }

    // -- Pruning --

    #[test]
    fn test_prune_below_threshold() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.min_relevance = 0.99; // Very high threshold
        cfg.recency_half_life = 10.0;
        let mut kb = KnowledgeBase::with_config(cfg).unwrap();

        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        // Advance step far to make entry old
        kb.step = 10000;
        kb.prune_below_threshold();

        // Entry should have been pruned (very old, below high threshold)
        assert_eq!(kb.total_entries(), 0);
        assert!(kb.stats().total_entries_pruned > 0);
    }

    // -- Strategy summaries --

    #[test]
    fn test_update_strategy_summary_new() {
        let mut kb = KnowledgeBase::new();
        kb.update_strategy_summary("momentum", 0.12, 1.5, 0.05);

        let s = kb.get_strategy_summary("momentum").unwrap();
        assert_eq!(s.strategy_name, "momentum");
        assert!((s.mean_return - 0.12).abs() < 1e-10);
        assert!((s.mean_sharpe - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_update_strategy_summary_existing() {
        let mut kb = KnowledgeBase::new();
        kb.update_strategy_summary("momentum", 0.10, 1.5, 0.05);
        kb.update_strategy_summary("momentum", 0.20, 2.0, 0.08);

        let s = kb.get_strategy_summary("momentum").unwrap();
        assert_eq!(s.periods, 2);
        assert!((s.mean_return - 0.15).abs() < 1e-10);
        assert!(s.max_drawdown >= 0.08 - 1e-10);
    }

    #[test]
    fn test_strategy_summary_win_rate() {
        let mut kb = KnowledgeBase::new();
        kb.update_strategy_summary("test", 0.10, 1.0, 0.05);
        kb.update_strategy_summary("test", -0.05, -0.5, 0.10);

        let s = kb.get_strategy_summary("test").unwrap();
        assert!((s.win_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_strategy_summary_ci() {
        let mut kb = KnowledgeBase::new();
        for i in 0..20 {
            kb.update_strategy_summary("test", 0.01 * i as f64, 1.0, 0.05);
        }

        let s = kb.get_strategy_summary("test").unwrap();
        assert!(s.ci_half_width.is_finite());
        assert!(s.ci_half_width > 0.0);
    }

    #[test]
    fn test_strategy_summary_is_significant_positive() {
        let s = StrategyPerformanceSummary {
            strategy_name: "test".into(),
            mean_return: 0.10,
            return_std: 0.05,
            mean_sharpe: 2.0,
            max_drawdown: 0.05,
            win_rate: 0.8,
            periods: 50,
            ci_half_width: 0.02,
            best_regimes: vec![],
            worst_regimes: vec![],
        };
        assert!(s.is_significant_positive()); // 0.10 > 0.02
    }

    #[test]
    fn test_strategy_summary_not_significant() {
        let s = StrategyPerformanceSummary {
            strategy_name: "test".into(),
            mean_return: 0.01,
            return_std: 0.10,
            mean_sharpe: 0.1,
            max_drawdown: 0.20,
            win_rate: 0.45,
            periods: 5, // Not enough periods
            ci_half_width: 0.10,
            best_regimes: vec![],
            worst_regimes: vec![],
        };
        assert!(!s.is_significant_positive());
    }

    #[test]
    fn test_ranked_strategies() {
        let mut kb = KnowledgeBase::new();
        kb.update_strategy_summary("low", 0.05, 0.5, 0.10);
        kb.update_strategy_summary("high", 0.20, 2.5, 0.05);
        kb.update_strategy_summary("mid", 0.10, 1.5, 0.08);

        let ranked = kb.ranked_strategies();
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].strategy_name, "high");
        assert_eq!(ranked[1].strategy_name, "mid");
        assert_eq!(ranked[2].strategy_name, "low");
    }

    // -- Regime transitions --

    #[test]
    fn test_record_regime_transition() {
        let mut kb = KnowledgeBase::new();
        kb.record_regime_transition(0, 1);
        kb.record_regime_transition(0, 1);
        kb.record_regime_transition(0, 2);

        // With prior = 1.0, P(1|0) = (2+1) / (2+1+1+1+1) = 3/6 = 0.5 (approx)
        let p = kb.regime_transition_probability(0, 1);
        assert!(p > 0.0);
        assert!(p < 1.0);
    }

    #[test]
    fn test_regime_transition_uniform_prior() {
        let kb = KnowledgeBase::new();
        // No transitions, so probabilities should be uniform
        let p = kb.regime_transition_probability(0, 1);
        assert!((p - 1.0 / NUM_REGIMES as f64).abs() < 1e-10);
    }

    #[test]
    fn test_regime_transition_row_sums_to_one() {
        let mut kb = KnowledgeBase::new();
        kb.record_regime_transition(0, 1);
        kb.record_regime_transition(0, 2);
        kb.record_regime_transition(0, 0);

        let row = kb.regime_transition_row(0);
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_most_likely_next_regime() {
        let mut kb = KnowledgeBase::new();
        // Heavily bias 0 → 3
        for _ in 0..50 {
            kb.record_regime_transition(0, 3);
        }
        assert_eq!(kb.most_likely_next_regime(0), 3);
    }

    #[test]
    fn test_stationary_distribution_sums_to_one() {
        let mut kb = KnowledgeBase::new();
        for _ in 0..20 {
            kb.record_regime_transition(0, 1);
            kb.record_regime_transition(1, 2);
            kb.record_regime_transition(2, 0);
        }

        let dist = kb.stationary_distribution();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_transition_matrix_total() {
        let mut kb = KnowledgeBase::new();
        kb.record_regime_transition(0, 1);
        kb.record_regime_transition(1, 2);
        assert_eq!(kb.transition_matrix().total_transitions(), 2);
    }

    #[test]
    fn test_transition_matrix_count() {
        let mut kb = KnowledgeBase::new();
        kb.record_regime_transition(2, 3);
        kb.record_regime_transition(2, 3);
        assert_eq!(kb.transition_matrix().count(2, 3), 2);
    }

    #[test]
    fn test_transition_matrix_out_of_bounds() {
        let kb = KnowledgeBase::new();
        assert_eq!(kb.regime_transition_probability(99, 0), 0.0);
        assert_eq!(kb.transition_matrix().count(99, 0), 0);
    }

    #[test]
    fn test_transition_matrix_reset() {
        let mut kb = KnowledgeBase::new();
        kb.record_regime_transition(0, 1);
        assert_eq!(kb.transition_matrix().total_transitions(), 1);

        // Reset via full kb reset
        kb.reset();
        assert_eq!(kb.transition_matrix().total_transitions(), 0);
    }

    // -- Patterns --

    #[test]
    fn test_add_pattern() {
        let mut kb = KnowledgeBase::new();
        let p = make_pattern("reversal", vec![1.0, 0.0, -1.0]);
        assert!(kb.add_pattern(p).is_ok());
        assert_eq!(kb.pattern_count(), 1);
    }

    #[test]
    fn test_add_pattern_duplicate() {
        let mut kb = KnowledgeBase::new();
        kb.add_pattern(make_pattern("dup", vec![1.0])).unwrap();
        assert!(kb.add_pattern(make_pattern("dup", vec![0.0])).is_err());
    }

    #[test]
    fn test_add_pattern_empty_name() {
        let mut kb = KnowledgeBase::new();
        assert!(kb.add_pattern(make_pattern("", vec![1.0])).is_err());
    }

    #[test]
    fn test_get_pattern() {
        let mut kb = KnowledgeBase::new();
        kb.add_pattern(make_pattern("p1", vec![1.0, 0.5])).unwrap();
        assert!(kb.get_pattern("p1").is_some());
        assert!(kb.get_pattern("nope").is_none());
    }

    #[test]
    fn test_get_pattern_mut() {
        let mut kb = KnowledgeBase::new();
        kb.add_pattern(make_pattern("p1", vec![1.0])).unwrap();
        let p = kb.get_pattern_mut("p1").unwrap();
        p.accuracy = 0.8;
        assert!((kb.get_pattern("p1").unwrap().accuracy - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_match_patterns() {
        let mut kb = KnowledgeBase::new();
        kb.add_pattern(make_pattern("up", vec![1.0, 0.0])).unwrap();
        kb.add_pattern(make_pattern("down", vec![-1.0, 0.0]))
            .unwrap();

        let observation = vec![0.9, 0.1];
        let matches = kb.match_patterns(&observation, 0.5);
        assert!(!matches.is_empty());
        // "up" should match better than "down"
        assert_eq!(matches[0].0.name, "up");
        assert!(matches[0].1 > 0.5);
    }

    #[test]
    fn test_match_patterns_threshold() {
        let mut kb = KnowledgeBase::new();
        kb.add_pattern(make_pattern("p1", vec![1.0, 0.0])).unwrap();

        let observation = vec![-1.0, 0.0]; // Opposite direction
        let matches = kb.match_patterns(&observation, 0.5);
        assert!(matches.is_empty()); // cosine sim ≈ -1.0, below threshold
    }

    #[test]
    fn test_match_patterns_dim_mismatch() {
        let mut kb = KnowledgeBase::new();
        kb.add_pattern(make_pattern("p1", vec![1.0, 0.0])).unwrap();

        let observation = vec![1.0]; // Different dimension
        let matches = kb.match_patterns(&observation, 0.0);
        assert!(matches.is_empty()); // match_strength returns 0 for dim mismatch
    }

    #[test]
    fn test_pattern_match_strength_identical() {
        let p = make_pattern("p", vec![1.0, 0.0, 0.0]);
        let obs = vec![1.0, 0.0, 0.0];
        assert!((p.match_strength(&obs) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_match_strength_orthogonal() {
        let p = make_pattern("p", vec![1.0, 0.0]);
        let obs = vec![0.0, 1.0];
        assert!(p.match_strength(&obs).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_match_strength_zero_vector() {
        let p = make_pattern("p", vec![0.0, 0.0]);
        let obs = vec![1.0, 0.0];
        assert_eq!(p.match_strength(&obs), 0.0);
    }

    #[test]
    fn test_pattern_update_stats() {
        let mut p = make_pattern("p", vec![1.0]);
        p.update_stats(true, 0.05);
        p.update_stats(false, -0.02);

        assert_eq!(p.observation_count, 2);
        assert!((p.accuracy - 0.5).abs() < 1e-10);
        assert!((p.avg_subsequent_return - 0.015).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_information_ratio() {
        let mut p = make_pattern("p", vec![1.0]);
        p.avg_subsequent_return = 0.10;
        p.return_std = 0.05;
        assert!((p.information_ratio() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_information_ratio_zero_std() {
        let p = make_pattern("p", vec![1.0]);
        assert_eq!(p.information_ratio(), 0.0);
    }

    #[test]
    fn test_reliable_patterns() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.min_pattern_observations = 3;
        let mut kb = KnowledgeBase::with_config(cfg).unwrap();

        let mut p1 = make_pattern("reliable", vec![1.0]);
        p1.observation_count = 5;
        let p2 = make_pattern("new", vec![0.0]);

        kb.add_pattern(p1).unwrap();
        kb.add_pattern(p2).unwrap();

        let reliable = kb.reliable_patterns();
        assert_eq!(reliable.len(), 1);
        assert_eq!(reliable[0].name, "reliable");
    }

    // -- Statistics --

    #[test]
    fn test_stats_initial() {
        let kb = KnowledgeBase::new();
        assert_eq!(kb.stats().total_entries_added, 0);
        assert_eq!(kb.stats().total_queries, 0);
    }

    #[test]
    fn test_stats_prune_rate() {
        let stats = KnowledgeBaseStats {
            total_entries_added: 20,
            total_entries_pruned: 5,
            ..Default::default()
        };
        assert!((stats.prune_rate() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_stats_prune_rate_zero() {
        let stats = KnowledgeBaseStats::default();
        assert_eq!(stats.prune_rate(), 0.0);
    }

    #[test]
    fn test_stats_confidence_counts() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        assert_eq!(kb.stats().confidence_counts[0], 1); // Low confidence initially
    }

    // -- EMA tracking --

    #[test]
    fn test_ema_initializes_on_first_query() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        let tags: Vec<String> = Vec::new();
        kb.query(&tags, None, 10);

        assert!(kb.smoothed_avg_relevance() >= 0.0);
        assert!(kb.smoothed_result_count() > 0.0);
        assert!(kb.smoothed_utilization() > 0.0);
    }

    #[test]
    fn test_ema_blends() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        let tags: Vec<String> = Vec::new();
        kb.query(&tags, None, 10);
        let r1 = kb.smoothed_result_count();

        // Add more entries and query again
        kb.add_entry(make_entry("e2", KnowledgeType::Pattern))
            .unwrap();
        kb.add_entry(make_entry("e3", KnowledgeType::Pattern))
            .unwrap();
        kb.query(&tags, None, 10);
        let r2 = kb.smoothed_result_count();

        assert!(r2 > r1);
    }

    // -- Sliding window --

    #[test]
    fn test_recent_queries_windowed() {
        let mut cfg = KnowledgeBaseConfig::default();
        cfg.window_size = 3;
        let mut kb = KnowledgeBase::with_config(cfg).unwrap();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();

        let tags: Vec<String> = Vec::new();
        for _ in 0..10 {
            kb.query(&tags, None, 10);
        }

        assert!(kb.recent_queries().len() <= 3);
    }

    #[test]
    fn test_windowed_avg_relevance() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        let tags: Vec<String> = Vec::new();
        kb.query(&tags, None, 10);

        assert!(kb.windowed_avg_relevance() >= 0.0);
    }

    #[test]
    fn test_windowed_avg_result_count() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        let tags: Vec<String> = Vec::new();
        kb.query(&tags, None, 10);

        assert!(kb.windowed_avg_result_count() > 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let kb = KnowledgeBase::new();
        assert_eq!(kb.windowed_avg_relevance(), 0.0);
        assert_eq!(kb.windowed_avg_result_count(), 0.0);
    }

    // -- Quality trend --

    #[test]
    fn test_is_quality_improving_insufficient_data() {
        let kb = KnowledgeBase::new();
        assert!(!kb.is_quality_improving());
    }

    // -- Step management --

    #[test]
    fn test_advance_step() {
        let mut kb = KnowledgeBase::new();
        assert_eq!(kb.current_step(), 0);
        kb.advance_step();
        assert_eq!(kb.current_step(), 1);
        kb.advance_step();
        assert_eq!(kb.current_step(), 2);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut kb = KnowledgeBase::new();
        kb.add_entry(make_entry("e1", KnowledgeType::Pattern))
            .unwrap();
        kb.add_pattern(make_pattern("p1", vec![1.0])).unwrap();
        kb.record_regime_transition(0, 1);
        kb.update_strategy_summary("test", 0.10, 1.5, 0.05);
        kb.advance_step();

        let tags: Vec<String> = Vec::new();
        kb.query(&tags, None, 10);

        kb.reset();

        assert_eq!(kb.total_entries(), 0);
        assert_eq!(kb.pattern_count(), 0);
        assert_eq!(kb.transition_matrix().total_transitions(), 0);
        assert!(kb.strategy_summaries().is_empty());
        assert_eq!(kb.current_step(), 0);
        assert!(kb.recent_queries().is_empty());
        assert_eq!(kb.stats().total_entries_added, 0);
    }

    // -- Knowledge type --

    #[test]
    fn test_knowledge_type_equality() {
        assert_eq!(KnowledgeType::Pattern, KnowledgeType::Pattern);
        assert_ne!(KnowledgeType::Pattern, KnowledgeType::StrategySummary);
    }

    #[test]
    fn test_knowledge_type_custom() {
        let a = KnowledgeType::Custom("alpha".into());
        let b = KnowledgeType::Custom("alpha".into());
        let c = KnowledgeType::Custom("beta".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // -- Integration test --

    #[test]
    fn test_full_lifecycle() {
        let mut kb = KnowledgeBase::new();

        // Add entries
        kb.add_entry(make_entry("pattern_1", KnowledgeType::Pattern))
            .unwrap();
        kb.add_entry(make_entry("strat_1", KnowledgeType::StrategySummary))
            .unwrap();
        kb.add_entry(make_entry("regime_1", KnowledgeType::RegimeTransition))
            .unwrap();

        // Update entries over time
        for i in 1..=10 {
            kb.advance_step();
            kb.update_entry("pattern_1", 0.05 * i as f64).unwrap();
        }

        // Add strategy summaries
        for i in 0..15 {
            kb.update_strategy_summary("momentum", 0.01 * i as f64, 1.0, 0.05);
        }

        // Record regime transitions
        for _ in 0..20 {
            kb.record_regime_transition(0, 1);
            kb.record_regime_transition(1, 2);
            kb.record_regime_transition(2, 0);
        }

        // Add patterns
        kb.add_pattern(make_pattern("breakout", vec![1.0, 0.5, 0.2]))
            .unwrap();
        let p = kb.get_pattern_mut("breakout").unwrap();
        for _ in 0..10 {
            p.update_stats(true, 0.03);
        }

        // Query
        let tags = vec!["bull".into(), "crypto".into()];
        let results = kb.query(&tags, None, 5);
        assert!(!results.is_empty());

        // Check pattern matching
        let matches = kb.match_patterns(&[0.9, 0.5, 0.1], 0.5);
        assert!(!matches.is_empty());

        // Check strategy ranking
        let ranked = kb.ranked_strategies();
        assert!(!ranked.is_empty());

        // Check regime transitions
        let dist = kb.stationary_distribution();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);

        // Check confidence promotion
        let e = kb.get_entry("pattern_1").unwrap();
        assert!(e.confidence >= ConfidenceLevel::Medium);
        assert_eq!(e.observation_count, 11); // 1 initial + 10 updates

        // Stats
        assert!(kb.stats().total_entries_added >= 3);
        assert!(kb.stats().total_queries >= 1);
        assert!(kb.stats().total_updates >= 10);
    }
}
