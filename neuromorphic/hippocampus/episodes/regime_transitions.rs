//! Regime change episodes
//!
//! Part of the Hippocampus region
//! Component: episodes
//!
//! This module tracks and stores regime transition episodes - significant shifts
//! in market behavior that affect trading strategy selection. It provides:
//! - Detection and classification of regime changes
//! - Historical tracking of transitions
//! - Pattern analysis across regime changes
//! - Transition probability estimation
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Regime Transitions                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                               │
//! │  Regime States:                                              │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
//! │  │ Trending │ │  Ranging │ │ Volatile │ │  Crisis  │       │
//! │  │   ↗↘    │ │   ↔↔    │ │   ⚡⚡   │ │   🔥🔥  │       │
//! │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
//! │       ↑↓           ↑↓           ↑↓           ↑↓             │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │              Transition Matrix                       │    │
//! │  │                                                       │    │
//! │  │    From\To  Trend  Range  Vol   Crisis              │    │
//! │  │    Trend    0.6    0.25   0.1   0.05                │    │
//! │  │    Range    0.3    0.5    0.15  0.05                │    │
//! │  │    Vol      0.2    0.3    0.4   0.1                 │    │
//! │  │    Crisis   0.1    0.2    0.3   0.4                 │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │                                                               │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    /// Strong upward trend
    BullTrending,
    /// Strong downward trend
    BearTrending,
    /// Sideways/consolidation
    Ranging,
    /// High volatility (no clear direction)
    HighVolatility,
    /// Low volatility (quiet market)
    LowVolatility,
    /// Crisis/stress conditions
    Crisis,
    /// Recovery from crisis
    Recovery,
    /// Unknown/insufficient data
    Unknown,
}

impl MarketRegime {
    /// Get risk level for this regime (0.0 - 1.0)
    pub fn risk_level(&self) -> f64 {
        match self {
            MarketRegime::BullTrending => 0.3,
            MarketRegime::BearTrending => 0.5,
            MarketRegime::Ranging => 0.2,
            MarketRegime::HighVolatility => 0.7,
            MarketRegime::LowVolatility => 0.1,
            MarketRegime::Crisis => 0.95,
            MarketRegime::Recovery => 0.5,
            MarketRegime::Unknown => 0.5,
        }
    }

    /// Get recommended position sizing multiplier
    pub fn position_multiplier(&self) -> f64 {
        match self {
            MarketRegime::BullTrending => 1.2,
            MarketRegime::BearTrending => 0.8,
            MarketRegime::Ranging => 1.0,
            MarketRegime::HighVolatility => 0.5,
            MarketRegime::LowVolatility => 1.1,
            MarketRegime::Crisis => 0.2,
            MarketRegime::Recovery => 0.7,
            MarketRegime::Unknown => 0.5,
        }
    }

    /// Check if this is a high-risk regime
    pub fn is_high_risk(&self) -> bool {
        matches!(
            self,
            MarketRegime::Crisis | MarketRegime::HighVolatility | MarketRegime::BearTrending
        )
    }

    /// Get all regime variants
    pub fn all_regimes() -> Vec<MarketRegime> {
        vec![
            MarketRegime::BullTrending,
            MarketRegime::BearTrending,
            MarketRegime::Ranging,
            MarketRegime::HighVolatility,
            MarketRegime::LowVolatility,
            MarketRegime::Crisis,
            MarketRegime::Recovery,
        ]
    }
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::BullTrending => write!(f, "Bull Trending"),
            MarketRegime::BearTrending => write!(f, "Bear Trending"),
            MarketRegime::Ranging => write!(f, "Ranging"),
            MarketRegime::HighVolatility => write!(f, "High Volatility"),
            MarketRegime::LowVolatility => write!(f, "Low Volatility"),
            MarketRegime::Crisis => write!(f, "Crisis"),
            MarketRegime::Recovery => write!(f, "Recovery"),
            MarketRegime::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Cause/trigger for regime transition
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionCause {
    /// Gradual shift over time
    Gradual,
    /// News/event driven
    NewsEvent { headline: String },
    /// Central bank action
    CentralBankAction,
    /// Macro data release
    MacroDataRelease { indicator: String },
    /// Technical breakout
    TechnicalBreakout { level: String },
    /// Liquidity event
    LiquidityEvent,
    /// Correlation break
    CorrelationBreak,
    /// Unknown/automatic detection
    Unknown,
}

/// A single regime transition episode
#[derive(Debug, Clone)]
pub struct RegimeTransition {
    /// Unique identifier
    pub id: u64,
    /// Symbol this transition applies to
    pub symbol: String,
    /// Previous regime
    pub from_regime: MarketRegime,
    /// New regime
    pub to_regime: MarketRegime,
    /// Timestamp when transition was detected
    pub timestamp: i64,
    /// Duration of transition (milliseconds)
    pub transition_duration_ms: u64,
    /// Confidence in the transition detection (0.0 - 1.0)
    pub confidence: f64,
    /// What caused/triggered the transition
    pub cause: TransitionCause,
    /// Price at transition start
    pub start_price: f64,
    /// Price at transition end
    pub end_price: f64,
    /// Volatility at transition start
    pub start_volatility: f64,
    /// Volatility at transition end
    pub end_volatility: f64,
    /// Volume during transition (relative to average)
    pub transition_volume_ratio: f64,
    /// Associated metadata
    pub metadata: HashMap<String, String>,
    /// Performance after transition (e.g., 1-day return)
    pub subsequent_return: Option<f64>,
    /// Was this transition predicted?
    pub was_predicted: bool,
    /// Actions taken in response
    pub actions_taken: Vec<String>,
}

impl RegimeTransition {
    /// Create a new regime transition
    pub fn new(
        id: u64,
        symbol: &str,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
        timestamp: i64,
    ) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            from_regime,
            to_regime,
            timestamp,
            transition_duration_ms: 0,
            confidence: 0.0,
            cause: TransitionCause::Unknown,
            start_price: 0.0,
            end_price: 0.0,
            start_volatility: 0.0,
            end_volatility: 0.0,
            transition_volume_ratio: 1.0,
            metadata: HashMap::new(),
            subsequent_return: None,
            was_predicted: false,
            actions_taken: Vec::new(),
        }
    }

    /// Check if this was a transition to a higher risk regime
    pub fn is_risk_increase(&self) -> bool {
        self.to_regime.risk_level() > self.from_regime.risk_level()
    }

    /// Calculate the change in risk level
    pub fn risk_change(&self) -> f64 {
        self.to_regime.risk_level() - self.from_regime.risk_level()
    }

    /// Get the price change percentage
    pub fn price_change_pct(&self) -> f64 {
        if self.start_price == 0.0 {
            return 0.0;
        }
        (self.end_price - self.start_price) / self.start_price * 100.0
    }

    /// Get the volatility change ratio
    pub fn volatility_change_ratio(&self) -> f64 {
        if self.start_volatility == 0.0 {
            return 1.0;
        }
        self.end_volatility / self.start_volatility
    }
}

/// Configuration for regime transitions tracking
#[derive(Debug, Clone)]
pub struct RegimeTransitionsConfig {
    /// Maximum number of transitions to keep in history
    pub max_transitions: usize,
    /// Minimum confidence to record a transition
    pub min_confidence: f64,
    /// Minimum time between transitions (to avoid noise)
    pub min_transition_interval_ms: u64,
    /// Enable transition matrix learning
    pub learn_transitions: bool,
    /// Prior count for transition matrix (Bayesian smoothing)
    pub prior_count: f64,
}

impl Default for RegimeTransitionsConfig {
    fn default() -> Self {
        Self {
            max_transitions: 1000,
            min_confidence: 0.6,
            min_transition_interval_ms: 60_000, // 1 minute
            learn_transitions: true,
            prior_count: 1.0,
        }
    }
}

/// Transition matrix for regime change probabilities
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    /// Count of transitions from regime i to regime j
    counts: HashMap<(MarketRegime, MarketRegime), f64>,
    /// Total transitions from each regime
    totals: HashMap<MarketRegime, f64>,
    /// Prior count for smoothing
    prior: f64,
}

impl TransitionMatrix {
    /// Create a new transition matrix with uniform prior
    pub fn new(prior: f64) -> Self {
        let regimes = MarketRegime::all_regimes();
        let mut counts = HashMap::new();
        let mut totals = HashMap::new();

        // Initialize with prior counts
        for &from in &regimes {
            totals.insert(from, prior * regimes.len() as f64);
            for &to in &regimes {
                counts.insert((from, to), prior);
            }
        }

        Self {
            counts,
            totals,
            prior,
        }
    }

    /// Record a transition
    pub fn record(&mut self, from: MarketRegime, to: MarketRegime) {
        *self.counts.entry((from, to)).or_insert(self.prior) += 1.0;
        *self.totals.entry(from).or_insert(self.prior) += 1.0;
    }

    /// Get the probability of transitioning from one regime to another
    pub fn probability(&self, from: MarketRegime, to: MarketRegime) -> f64 {
        let count = self.counts.get(&(from, to)).copied().unwrap_or(self.prior);
        let total = self
            .totals
            .get(&from)
            .copied()
            .unwrap_or(self.prior * MarketRegime::all_regimes().len() as f64);

        if total == 0.0 {
            return 0.0;
        }
        count / total
    }

    /// Get the most likely next regime given current regime
    pub fn most_likely_next(&self, from: MarketRegime) -> MarketRegime {
        let regimes = MarketRegime::all_regimes();

        regimes
            .into_iter()
            .max_by(|a, b| {
                self.probability(from, *a)
                    .partial_cmp(&self.probability(from, *b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(MarketRegime::Unknown)
    }

    /// Get probability distribution for next regime
    pub fn next_distribution(&self, from: MarketRegime) -> HashMap<MarketRegime, f64> {
        MarketRegime::all_regimes()
            .into_iter()
            .map(|to| (to, self.probability(from, to)))
            .collect()
    }

    /// Get transition counts for analysis
    pub fn get_counts(&self) -> &HashMap<(MarketRegime, MarketRegime), f64> {
        &self.counts
    }
}

/// Statistics about regime transitions
#[derive(Debug, Clone)]
pub struct TransitionStats {
    /// Total transitions recorded
    pub total_transitions: usize,
    /// Transitions per symbol
    pub transitions_by_symbol: HashMap<String, usize>,
    /// Current regime per symbol
    pub current_regimes: HashMap<String, MarketRegime>,
    /// Average time in each regime (milliseconds)
    pub avg_regime_duration: HashMap<MarketRegime, f64>,
    /// Count of transitions to high-risk regimes
    pub high_risk_transitions: usize,
    /// Count of transitions from crisis
    pub crisis_recoveries: usize,
}

/// Regime change episodes tracker
pub struct RegimeTransitions {
    /// Configuration
    config: RegimeTransitionsConfig,
    /// All recorded transitions (oldest first)
    transitions: VecDeque<RegimeTransition>,
    /// Transitions indexed by symbol
    by_symbol: HashMap<String, Vec<u64>>,
    /// Current regime per symbol
    current_regimes: HashMap<String, MarketRegime>,
    /// Last transition timestamp per symbol
    last_transition_time: HashMap<String, i64>,
    /// Transition probability matrix
    transition_matrix: TransitionMatrix,
    /// Next transition ID
    next_id: u64,
    /// Regime duration tracking (start timestamp per symbol)
    regime_start_times: HashMap<String, i64>,
    /// Accumulated duration per regime
    regime_durations: HashMap<MarketRegime, Vec<u64>>,
}

impl Default for RegimeTransitions {
    fn default() -> Self {
        Self::new()
    }
}

impl RegimeTransitions {
    /// Create a new regime transitions tracker
    pub fn new() -> Self {
        Self::with_config(RegimeTransitionsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: RegimeTransitionsConfig) -> Self {
        Self {
            transition_matrix: TransitionMatrix::new(config.prior_count),
            config,
            transitions: VecDeque::new(),
            by_symbol: HashMap::new(),
            current_regimes: HashMap::new(),
            last_transition_time: HashMap::new(),
            next_id: 1,
            regime_start_times: HashMap::new(),
            regime_durations: HashMap::new(),
        }
    }

    /// Record a regime transition
    pub fn record_transition(
        &mut self,
        symbol: &str,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
        timestamp: i64,
        confidence: f64,
    ) -> Option<u64> {
        // Check minimum confidence
        if confidence < self.config.min_confidence {
            return None;
        }

        // Check minimum interval
        if let Some(&last_time) = self.last_transition_time.get(symbol) {
            let interval = (timestamp - last_time) as u64;
            if interval < self.config.min_transition_interval_ms {
                return None;
            }
        }

        // Calculate regime duration
        let duration = if let Some(&start) = self.regime_start_times.get(symbol) {
            (timestamp - start) as u64
        } else {
            0
        };

        // Record duration for the previous regime
        self.regime_durations
            .entry(from_regime)
            .or_insert_with(Vec::new)
            .push(duration);

        // Create transition record
        let id = self.next_id;
        self.next_id += 1;

        let mut transition = RegimeTransition::new(id, symbol, from_regime, to_regime, timestamp);
        transition.confidence = confidence;
        transition.transition_duration_ms = duration;

        // Add to collections
        self.transitions.push_back(transition);
        self.by_symbol
            .entry(symbol.to_string())
            .or_insert_with(Vec::new)
            .push(id);
        self.current_regimes.insert(symbol.to_string(), to_regime);
        self.last_transition_time
            .insert(symbol.to_string(), timestamp);
        self.regime_start_times
            .insert(symbol.to_string(), timestamp);

        // Update transition matrix
        if self.config.learn_transitions {
            self.transition_matrix.record(from_regime, to_regime);
        }

        // Prune if needed
        self.prune_if_needed();

        Some(id)
    }

    /// Record a detailed transition
    pub fn record_detailed_transition(&mut self, mut transition: RegimeTransition) -> Option<u64> {
        // Check minimum confidence
        if transition.confidence < self.config.min_confidence {
            return None;
        }

        // Check minimum interval
        if let Some(&last_time) = self.last_transition_time.get(&transition.symbol) {
            let interval = (transition.timestamp - last_time) as u64;
            if interval < self.config.min_transition_interval_ms {
                return None;
            }
        }

        // Assign ID
        transition.id = self.next_id;
        self.next_id += 1;
        let id = transition.id;

        // Update state
        self.current_regimes
            .insert(transition.symbol.clone(), transition.to_regime);
        self.last_transition_time
            .insert(transition.symbol.clone(), transition.timestamp);
        self.regime_start_times
            .insert(transition.symbol.clone(), transition.timestamp);

        // Update transition matrix
        if self.config.learn_transitions {
            self.transition_matrix
                .record(transition.from_regime, transition.to_regime);
        }

        // Store
        self.by_symbol
            .entry(transition.symbol.clone())
            .or_insert_with(Vec::new)
            .push(id);
        self.transitions.push_back(transition);

        // Prune if needed
        self.prune_if_needed();

        Some(id)
    }

    /// Set the current regime for a symbol (without recording a transition)
    pub fn set_regime(&mut self, symbol: &str, regime: MarketRegime, timestamp: i64) {
        self.current_regimes.insert(symbol.to_string(), regime);
        self.regime_start_times
            .insert(symbol.to_string(), timestamp);
    }

    /// Get the current regime for a symbol
    pub fn current_regime(&self, symbol: &str) -> MarketRegime {
        self.current_regimes
            .get(symbol)
            .copied()
            .unwrap_or(MarketRegime::Unknown)
    }

    /// Get transition by ID
    pub fn get(&self, id: u64) -> Option<&RegimeTransition> {
        self.transitions.iter().find(|t| t.id == id)
    }

    /// Get mutable transition by ID
    pub fn get_mut(&mut self, id: u64) -> Option<&mut RegimeTransition> {
        self.transitions.iter_mut().find(|t| t.id == id)
    }

    /// Get recent transitions
    pub fn recent(&self, count: usize) -> Vec<&RegimeTransition> {
        self.transitions.iter().rev().take(count).collect()
    }

    /// Get transitions for a symbol
    pub fn for_symbol(&self, symbol: &str) -> Vec<&RegimeTransition> {
        self.by_symbol
            .get(symbol)
            .map(|ids| ids.iter().filter_map(|id| self.get(*id)).collect())
            .unwrap_or_default()
    }

    /// Get transitions between specific regimes
    pub fn between_regimes(&self, from: MarketRegime, to: MarketRegime) -> Vec<&RegimeTransition> {
        self.transitions
            .iter()
            .filter(|t| t.from_regime == from && t.to_regime == to)
            .collect()
    }

    /// Get transitions to high-risk regimes
    pub fn high_risk_transitions(&self) -> Vec<&RegimeTransition> {
        self.transitions
            .iter()
            .filter(|t| t.is_risk_increase() && t.to_regime.is_high_risk())
            .collect()
    }

    /// Get probability of transitioning to another regime
    pub fn transition_probability(&self, from: MarketRegime, to: MarketRegime) -> f64 {
        self.transition_matrix.probability(from, to)
    }

    /// Get most likely next regime
    pub fn predict_next_regime(&self, current: MarketRegime) -> MarketRegime {
        self.transition_matrix.most_likely_next(current)
    }

    /// Get regime probabilities for a symbol
    pub fn regime_probabilities(&self, symbol: &str) -> HashMap<MarketRegime, f64> {
        let current = self.current_regime(symbol);
        self.transition_matrix.next_distribution(current)
    }

    /// Update subsequent return for a transition
    pub fn update_subsequent_return(&mut self, id: u64, return_pct: f64) {
        if let Some(t) = self.get_mut(id) {
            t.subsequent_return = Some(return_pct);
        }
    }

    /// Record an action taken for a transition
    pub fn record_action(&mut self, id: u64, action: &str) {
        if let Some(t) = self.get_mut(id) {
            t.actions_taken.push(action.to_string());
        }
    }

    /// Get statistics
    pub fn stats(&self) -> TransitionStats {
        let mut transitions_by_symbol: HashMap<String, usize> = HashMap::new();
        for t in &self.transitions {
            *transitions_by_symbol.entry(t.symbol.clone()).or_insert(0) += 1;
        }

        let high_risk_transitions = self
            .transitions
            .iter()
            .filter(|t| t.is_risk_increase() && t.to_regime.is_high_risk())
            .count();

        let crisis_recoveries = self
            .transitions
            .iter()
            .filter(|t| t.from_regime == MarketRegime::Crisis)
            .count();

        // Calculate average regime durations
        let avg_regime_duration: HashMap<MarketRegime, f64> = self
            .regime_durations
            .iter()
            .map(|(regime, durations)| {
                let avg = if durations.is_empty() {
                    0.0
                } else {
                    durations.iter().map(|&d| d as f64).sum::<f64>() / durations.len() as f64
                };
                (*regime, avg)
            })
            .collect();

        TransitionStats {
            total_transitions: self.transitions.len(),
            transitions_by_symbol,
            current_regimes: self.current_regimes.clone(),
            avg_regime_duration,
            high_risk_transitions,
            crisis_recoveries,
        }
    }

    /// Get the transition matrix
    pub fn transition_matrix(&self) -> &TransitionMatrix {
        &self.transition_matrix
    }

    /// Prune old transitions if over limit
    fn prune_if_needed(&mut self) {
        while self.transitions.len() > self.config.max_transitions {
            if let Some(old) = self.transitions.pop_front() {
                // Remove from symbol index
                if let Some(ids) = self.by_symbol.get_mut(&old.symbol) {
                    ids.retain(|&id| id != old.id);
                }
            }
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.by_symbol.clear();
        self.current_regimes.clear();
        self.last_transition_time.clear();
        self.regime_start_times.clear();
        self.regime_durations.clear();
        self.transition_matrix = TransitionMatrix::new(self.config.prior_count);
        self.next_id = 1;
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // No-op for basic processing
        Ok(())
    }

    /// Get total number of recorded transitions
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = RegimeTransitions::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_regime_risk_levels() {
        assert!(MarketRegime::Crisis.risk_level() > MarketRegime::Ranging.risk_level());
        assert!(
            MarketRegime::HighVolatility.risk_level() > MarketRegime::LowVolatility.risk_level()
        );
    }

    #[test]
    fn test_regime_is_high_risk() {
        assert!(MarketRegime::Crisis.is_high_risk());
        assert!(MarketRegime::HighVolatility.is_high_risk());
        assert!(!MarketRegime::Ranging.is_high_risk());
        assert!(!MarketRegime::LowVolatility.is_high_risk());
    }

    #[test]
    fn test_record_transition() {
        let mut tracker = RegimeTransitions::new();

        let id = tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            1000,
            0.8,
        );

        assert!(id.is_some());
        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.current_regime("BTCUSD"), MarketRegime::BullTrending);
    }

    #[test]
    fn test_transition_minimum_confidence() {
        let mut tracker = RegimeTransitions::with_config(RegimeTransitionsConfig {
            min_confidence: 0.7,
            ..Default::default()
        });

        // Low confidence - should be rejected
        let id = tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            1000,
            0.5, // Below threshold
        );

        assert!(id.is_none());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_transition_minimum_interval() {
        let mut tracker = RegimeTransitions::with_config(RegimeTransitionsConfig {
            min_transition_interval_ms: 60_000,
            ..Default::default()
        });

        // First transition
        tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            1000,
            0.8,
        );

        // Too soon - should be rejected
        let id = tracker.record_transition(
            "BTCUSD",
            MarketRegime::BullTrending,
            MarketRegime::HighVolatility,
            30_000, // Only 29 seconds later
            0.8,
        );

        assert!(id.is_none());
        assert_eq!(tracker.len(), 1);

        // After interval - should succeed
        let id = tracker.record_transition(
            "BTCUSD",
            MarketRegime::BullTrending,
            MarketRegime::HighVolatility,
            100_000, // More than 60 seconds later
            0.8,
        );

        assert!(id.is_some());
        assert_eq!(tracker.len(), 2);
    }

    #[test]
    fn test_transition_matrix() {
        let mut matrix = TransitionMatrix::new(1.0);

        // Record some transitions
        matrix.record(MarketRegime::Ranging, MarketRegime::BullTrending);
        matrix.record(MarketRegime::Ranging, MarketRegime::BullTrending);
        matrix.record(MarketRegime::Ranging, MarketRegime::BearTrending);

        // Bull trending should be more likely than bear from ranging
        let prob_bull = matrix.probability(MarketRegime::Ranging, MarketRegime::BullTrending);
        let prob_bear = matrix.probability(MarketRegime::Ranging, MarketRegime::BearTrending);

        assert!(prob_bull > prob_bear);
    }

    #[test]
    fn test_most_likely_next() {
        let mut matrix = TransitionMatrix::new(0.1);

        // Make bull trending very likely from ranging
        for _ in 0..10 {
            matrix.record(MarketRegime::Ranging, MarketRegime::BullTrending);
        }

        let next = matrix.most_likely_next(MarketRegime::Ranging);
        assert_eq!(next, MarketRegime::BullTrending);
    }

    #[test]
    fn test_for_symbol() {
        let mut tracker = RegimeTransitions::new();

        tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            1000,
            0.8,
        );
        tracker.record_transition(
            "ETHUSD",
            MarketRegime::Ranging,
            MarketRegime::BearTrending,
            2000,
            0.8,
        );
        tracker.record_transition(
            "BTCUSD",
            MarketRegime::BullTrending,
            MarketRegime::HighVolatility,
            100_000,
            0.8,
        );

        let btc_transitions = tracker.for_symbol("BTCUSD");
        assert_eq!(btc_transitions.len(), 2);

        let eth_transitions = tracker.for_symbol("ETHUSD");
        assert_eq!(eth_transitions.len(), 1);
    }

    #[test]
    fn test_between_regimes() {
        let mut tracker = RegimeTransitions::new();

        tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            1000,
            0.8,
        );
        tracker.record_transition(
            "ETHUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            2000,
            0.8,
        );
        tracker.record_transition(
            "BTCUSD",
            MarketRegime::BullTrending,
            MarketRegime::Crisis,
            100_000,
            0.9,
        );

        let ranging_to_bull =
            tracker.between_regimes(MarketRegime::Ranging, MarketRegime::BullTrending);
        assert_eq!(ranging_to_bull.len(), 2);

        let bull_to_crisis =
            tracker.between_regimes(MarketRegime::BullTrending, MarketRegime::Crisis);
        assert_eq!(bull_to_crisis.len(), 1);
    }

    #[test]
    fn test_high_risk_transitions() {
        let mut tracker = RegimeTransitions::new();

        tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            1000,
            0.8,
        );
        tracker.record_transition(
            "BTCUSD",
            MarketRegime::BullTrending,
            MarketRegime::Crisis,
            100_000,
            0.9,
        );
        tracker.record_transition(
            "ETHUSD",
            MarketRegime::LowVolatility,
            MarketRegime::HighVolatility,
            3000,
            0.85,
        );

        let high_risk = tracker.high_risk_transitions();
        assert_eq!(high_risk.len(), 2); // Crisis and HighVolatility
    }

    #[test]
    fn test_transition_risk_change() {
        let transition = RegimeTransition::new(
            1,
            "BTCUSD",
            MarketRegime::LowVolatility,
            MarketRegime::Crisis,
            1000,
        );

        assert!(transition.is_risk_increase());
        assert!(transition.risk_change() > 0.0);
    }

    #[test]
    fn test_transition_price_change() {
        let mut transition = RegimeTransition::new(
            1,
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BearTrending,
            1000,
        );
        transition.start_price = 100.0;
        transition.end_price = 90.0;

        let pct = transition.price_change_pct();
        assert!((pct - (-10.0)).abs() < 0.01);
    }

    #[test]
    fn test_stats() {
        let mut tracker = RegimeTransitions::new();

        tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::Crisis,
            1000,
            0.8,
        );
        tracker.record_transition(
            "BTCUSD",
            MarketRegime::Crisis,
            MarketRegime::Recovery,
            100_000,
            0.8,
        );

        let stats = tracker.stats();
        assert_eq!(stats.total_transitions, 2);
        assert_eq!(stats.high_risk_transitions, 1);
        assert_eq!(stats.crisis_recoveries, 1);
    }

    #[test]
    fn test_set_regime() {
        let mut tracker = RegimeTransitions::new();

        // Set initial regime without transition
        tracker.set_regime("BTCUSD", MarketRegime::Ranging, 1000);
        assert_eq!(tracker.current_regime("BTCUSD"), MarketRegime::Ranging);
        assert_eq!(tracker.len(), 0); // No transition recorded
    }

    #[test]
    fn test_update_subsequent_return() {
        let mut tracker = RegimeTransitions::new();

        let id = tracker
            .record_transition(
                "BTCUSD",
                MarketRegime::Ranging,
                MarketRegime::BullTrending,
                1000,
                0.8,
            )
            .unwrap();

        tracker.update_subsequent_return(id, 5.0);

        let transition = tracker.get(id).unwrap();
        assert_eq!(transition.subsequent_return, Some(5.0));
    }

    #[test]
    fn test_record_action() {
        let mut tracker = RegimeTransitions::new();

        let id = tracker
            .record_transition(
                "BTCUSD",
                MarketRegime::Ranging,
                MarketRegime::Crisis,
                1000,
                0.9,
            )
            .unwrap();

        tracker.record_action(id, "Reduced position by 50%");
        tracker.record_action(id, "Set tighter stop loss");

        let transition = tracker.get(id).unwrap();
        assert_eq!(transition.actions_taken.len(), 2);
    }

    #[test]
    fn test_prune() {
        let config = RegimeTransitionsConfig {
            max_transitions: 3,
            min_transition_interval_ms: 0, // Disable interval check for test
            ..Default::default()
        };
        let mut tracker = RegimeTransitions::with_config(config);

        for i in 0..5 {
            tracker.record_transition(
                "BTCUSD",
                MarketRegime::Ranging,
                MarketRegime::BullTrending,
                i * 1000,
                0.8,
            );
        }

        // Should have pruned to max_transitions
        assert_eq!(tracker.len(), 3);
    }

    #[test]
    fn test_regime_display() {
        assert_eq!(format!("{}", MarketRegime::BullTrending), "Bull Trending");
        assert_eq!(format!("{}", MarketRegime::Crisis), "Crisis");
    }

    #[test]
    fn test_regime_probabilities() {
        let mut tracker = RegimeTransitions::new();

        // Record some transitions to build matrix
        tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            1000,
            0.8,
        );

        let probs = tracker.regime_probabilities("BTCUSD");
        assert!(!probs.is_empty());

        // All probabilities should sum to approximately 1.0
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut tracker = RegimeTransitions::new();

        tracker.record_transition(
            "BTCUSD",
            MarketRegime::Ranging,
            MarketRegime::BullTrending,
            1000,
            0.8,
        );

        tracker.clear();

        assert!(tracker.is_empty());
        assert_eq!(tracker.current_regime("BTCUSD"), MarketRegime::Unknown);
    }
}
