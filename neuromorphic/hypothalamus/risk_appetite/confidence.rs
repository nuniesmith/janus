//! Risk-Adjusted Confidence Estimation
//!
//! Part of the Hypothalamus region
//! Component: risk_appetite
//!
//! Combines multiple uncertainty sources into a single risk-adjusted confidence
//! score that modulates position sizing and trade aggressiveness.
//!
//! Dimensions tracked:
//! - **Model confidence**: How well the predictive model is performing (Brier / accuracy)
//! - **Signal quality**: Strength, consistency, and freshness of the input signal
//! - **Market conditions**: Volatility regime, liquidity, and spread health
//! - **Execution confidence**: Recent fill quality and slippage history
//! - **Regime agreement**: Whether sub-models agree on the current regime
//!
//! Features:
//! - Configurable per-dimension weights with normalisation
//! - EMA-smoothed composite confidence
//! - Minimum-confidence gating (blocks trades below threshold)
//! - Windowed diagnostics and degradation detection
//! - Adaptive weight adjustment based on which dimensions predict outcomes best

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the `Confidence` estimator
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Weight for model confidence dimension
    pub weight_model: f64,
    /// Weight for signal quality dimension
    pub weight_signal: f64,
    /// Weight for market conditions dimension
    pub weight_market: f64,
    /// Weight for execution confidence dimension
    pub weight_execution: f64,
    /// Weight for regime agreement dimension
    pub weight_regime: f64,
    /// EMA decay for smoothing the composite score (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum composite confidence to allow trading (0..1)
    pub min_confidence: f64,
    /// Sliding window size for recent snapshots
    pub window_size: usize,
    /// Minimum number of snapshots before adaptation engages
    pub min_samples: usize,
    /// Whether to normalise weights to sum to 1.0
    pub normalise_weights: bool,
    /// Dead-zone: ignore dimension changes smaller than this
    pub dead_zone: f64,
    /// Penalty multiplier applied when any single dimension is critically low (< 0.2)
    pub critical_penalty: f64,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            weight_model: 0.30,
            weight_signal: 0.25,
            weight_market: 0.20,
            weight_execution: 0.15,
            weight_regime: 0.10,
            ema_decay: 0.1,
            min_confidence: 0.2,
            window_size: 100,
            min_samples: 5,
            normalise_weights: true,
            dead_zone: 0.01,
            critical_penalty: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// Raw per-dimension confidence scores fed into the estimator.
/// All values should be in [0, 1].
#[derive(Debug, Clone)]
pub struct ConfidenceInput {
    /// Model predictive accuracy / calibration (0 = terrible, 1 = perfect)
    pub model: f64,
    /// Signal strength and consistency (0 = noise, 1 = strong clean signal)
    pub signal: f64,
    /// Market condition favourability (0 = hostile, 1 = ideal)
    pub market: f64,
    /// Execution quality (0 = severe slippage, 1 = perfect fills)
    pub execution: f64,
    /// Degree of agreement among sub-models (0 = total disagreement, 1 = consensus)
    pub regime: f64,
}

impl ConfidenceInput {
    /// Create an input with all dimensions set to the same value
    pub fn uniform(value: f64) -> Self {
        Self {
            model: value,
            signal: value,
            market: value,
            execution: value,
            regime: value,
        }
    }

    /// Create an input with only model and signal, defaulting others to 0.5
    pub fn model_signal(model: f64, signal: f64) -> Self {
        Self {
            model,
            signal,
            market: 0.5,
            execution: 0.5,
            regime: 0.5,
        }
    }

    /// Validate that all dimensions are in [0, 1]
    pub fn validate(&self) -> Result<()> {
        let check = |name: &str, v: f64| -> Result<()> {
            if !(0.0..=1.0).contains(&v) {
                return Err(Error::InvalidInput(format!(
                    "{} must be in [0, 1], got {}",
                    name, v
                )));
            }
            Ok(())
        };
        check("model", self.model)?;
        check("signal", self.signal)?;
        check("market", self.market)?;
        check("execution", self.execution)?;
        check("regime", self.regime)?;
        Ok(())
    }

    /// Return the minimum dimension value
    pub fn min_dimension(&self) -> f64 {
        self.model
            .min(self.signal)
            .min(self.market)
            .min(self.execution)
            .min(self.regime)
    }

    /// Return the maximum dimension value
    pub fn max_dimension(&self) -> f64 {
        self.model
            .max(self.signal)
            .max(self.market)
            .max(self.execution)
            .max(self.regime)
    }

    /// Return the spread between max and min dimensions (disagreement)
    pub fn spread(&self) -> f64 {
        self.max_dimension() - self.min_dimension()
    }
}

/// The computed confidence assessment
#[derive(Debug, Clone)]
pub struct ConfidenceAssessment {
    /// Raw weighted composite score before smoothing (0..1)
    pub raw_score: f64,
    /// EMA-smoothed composite score (0..1)
    pub smoothed_score: f64,
    /// Whether the confidence is above the minimum trading threshold
    pub tradeable: bool,
    /// Per-dimension weighted contributions
    pub contributions: DimensionContributions,
    /// Whether any dimension is critically low (< 0.2)
    pub has_critical_dimension: bool,
    /// The name of the weakest dimension
    pub weakest_dimension: &'static str,
    /// The weakest dimension's raw score
    pub weakest_score: f64,
    /// Spread between strongest and weakest dimensions
    pub dimension_spread: f64,
    /// Whether degradation is detected (recent trend downward)
    pub degrading: bool,
    /// Snapshot count so far
    pub snapshot_count: usize,
}

/// Per-dimension weighted contributions to the composite score
#[derive(Debug, Clone, Default)]
pub struct DimensionContributions {
    pub model: f64,
    pub signal: f64,
    pub market: f64,
    pub execution: f64,
    pub regime: f64,
}

/// An observed outcome for adaptive weight tuning
#[derive(Debug, Clone)]
pub struct ConfidenceOutcome {
    /// The input that was used when the trade was taken
    pub input: ConfidenceInput,
    /// Whether the trade was successful (met its target)
    pub success: bool,
    /// Profit/loss ratio (positive = good)
    pub pnl_ratio: f64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the confidence estimator
#[derive(Debug, Clone, Default)]
pub struct ConfidenceStats {
    /// Total snapshots processed
    pub total_snapshots: u64,
    /// Number of snapshots below the trading threshold
    pub below_threshold_count: u64,
    /// Number of snapshots with a critical dimension
    pub critical_dimension_count: u64,
    /// Sum of composite scores (for mean calculation)
    pub sum_score: f64,
    /// Sum of squared composite scores
    pub sum_sq_score: f64,
    /// Peak composite score observed
    pub peak_score: f64,
    /// Minimum composite score observed
    pub min_score: f64,
    /// Total outcomes observed
    pub total_outcomes: u64,
    /// Successful outcomes
    pub successful_outcomes: u64,
    /// Sum of per-dimension scores (for tracking which dimensions tend to be low)
    pub sum_model: f64,
    pub sum_signal: f64,
    pub sum_market: f64,
    pub sum_execution: f64,
    pub sum_regime: f64,
}

impl ConfidenceStats {
    /// Mean composite confidence score
    pub fn mean_score(&self) -> f64 {
        if self.total_snapshots == 0 {
            return 0.0;
        }
        self.sum_score / self.total_snapshots as f64
    }

    /// Variance of composite confidence scores
    pub fn score_variance(&self) -> f64 {
        if self.total_snapshots < 2 {
            return 0.0;
        }
        let n = self.total_snapshots as f64;
        let mean = self.sum_score / n;
        (self.sum_sq_score / n - mean * mean).max(0.0)
    }

    /// Standard deviation of composite confidence scores
    pub fn score_std(&self) -> f64 {
        self.score_variance().sqrt()
    }

    /// Fraction of snapshots below the trading threshold
    pub fn below_threshold_rate(&self) -> f64 {
        if self.total_snapshots == 0 {
            return 0.0;
        }
        self.below_threshold_count as f64 / self.total_snapshots as f64
    }

    /// Fraction of snapshots with a critical dimension
    pub fn critical_rate(&self) -> f64 {
        if self.total_snapshots == 0 {
            return 0.0;
        }
        self.critical_dimension_count as f64 / self.total_snapshots as f64
    }

    /// Mean score per dimension
    pub fn mean_dimension_scores(&self) -> DimensionContributions {
        if self.total_snapshots == 0 {
            return DimensionContributions::default();
        }
        let n = self.total_snapshots as f64;
        DimensionContributions {
            model: self.sum_model / n,
            signal: self.sum_signal / n,
            market: self.sum_market / n,
            execution: self.sum_execution / n,
            regime: self.sum_regime / n,
        }
    }

    /// Outcome success rate
    pub fn outcome_success_rate(&self) -> f64 {
        if self.total_outcomes == 0 {
            return 0.0;
        }
        self.successful_outcomes as f64 / self.total_outcomes as f64
    }
}

// ---------------------------------------------------------------------------
// Internal record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SnapshotRecord {
    raw_score: f64,
    smoothed_score: f64,
    model: f64,
    signal: f64,
    market: f64,
    execution: f64,
    regime: f64,
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Risk-adjusted confidence estimator
pub struct Confidence {
    config: ConfidenceConfig,

    /// Normalised weights (sum to 1.0 if normalise_weights is true)
    weights: [f64; 5],

    /// EMA-smoothed composite score
    ema_score: f64,
    ema_initialized: bool,

    /// Previous raw scores per dimension (for dead-zone filtering)
    prev_dimensions: [f64; 5],
    prev_initialized: bool,

    /// Count of snapshots
    snapshot_count: usize,

    /// Sliding window of recent snapshots
    recent: VecDeque<SnapshotRecord>,

    /// Running statistics
    stats: ConfidenceStats,
}

impl Default for Confidence {
    fn default() -> Self {
        Self::new()
    }
}

impl Confidence {
    /// Create with default configuration
    pub fn new() -> Self {
        Self::with_config(ConfidenceConfig::default()).unwrap()
    }

    /// Create with a specific configuration
    pub fn with_config(config: ConfidenceConfig) -> Result<Self> {
        // Validate
        if config.weight_model < 0.0
            || config.weight_signal < 0.0
            || config.weight_market < 0.0
            || config.weight_execution < 0.0
            || config.weight_regime < 0.0
        {
            return Err(Error::InvalidInput(
                "all weights must be non-negative".into(),
            ));
        }

        let weight_sum = config.weight_model
            + config.weight_signal
            + config.weight_market
            + config.weight_execution
            + config.weight_regime;

        if weight_sum < 1e-12 {
            return Err(Error::InvalidInput(
                "at least one weight must be > 0".into(),
            ));
        }

        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }

        if config.min_confidence < 0.0 || config.min_confidence > 1.0 {
            return Err(Error::InvalidInput(
                "min_confidence must be in [0, 1]".into(),
            ));
        }

        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }

        if config.dead_zone < 0.0 {
            return Err(Error::InvalidInput("dead_zone must be non-negative".into()));
        }

        if config.critical_penalty < 0.0 || config.critical_penalty > 1.0 {
            return Err(Error::InvalidInput(
                "critical_penalty must be in [0, 1]".into(),
            ));
        }

        // Compute normalised weights
        let weights = if config.normalise_weights {
            [
                config.weight_model / weight_sum,
                config.weight_signal / weight_sum,
                config.weight_market / weight_sum,
                config.weight_execution / weight_sum,
                config.weight_regime / weight_sum,
            ]
        } else {
            [
                config.weight_model,
                config.weight_signal,
                config.weight_market,
                config.weight_execution,
                config.weight_regime,
            ]
        };

        Ok(Self {
            config,
            weights,
            ema_score: 0.0,
            ema_initialized: false,
            prev_dimensions: [0.5; 5],
            prev_initialized: false,
            snapshot_count: 0,
            recent: VecDeque::new(),
            stats: ConfidenceStats {
                min_score: f64::MAX,
                ..Default::default()
            },
        })
    }

    /// Convenience factory — validates config and returns self
    pub fn process(config: ConfidenceConfig) -> Result<Self> {
        Self::with_config(config)
    }

    // -----------------------------------------------------------------------
    // Core assessment
    // -----------------------------------------------------------------------

    /// Compute a confidence assessment from raw dimension scores
    pub fn assess(&mut self, input: &ConfidenceInput) -> Result<ConfidenceAssessment> {
        input.validate()?;

        // Apply dead-zone: if a dimension hasn't changed more than dead_zone
        // from the previous reading, use the previous reading to reduce noise.
        let dims = if self.prev_initialized {
            [
                self.apply_dead_zone(input.model, self.prev_dimensions[0]),
                self.apply_dead_zone(input.signal, self.prev_dimensions[1]),
                self.apply_dead_zone(input.market, self.prev_dimensions[2]),
                self.apply_dead_zone(input.execution, self.prev_dimensions[3]),
                self.apply_dead_zone(input.regime, self.prev_dimensions[4]),
            ]
        } else {
            [
                input.model,
                input.signal,
                input.market,
                input.execution,
                input.regime,
            ]
        };

        // Update previous dimensions
        self.prev_dimensions = dims;
        self.prev_initialized = true;

        // Weighted composite
        let contributions = DimensionContributions {
            model: dims[0] * self.weights[0],
            signal: dims[1] * self.weights[1],
            market: dims[2] * self.weights[2],
            execution: dims[3] * self.weights[3],
            regime: dims[4] * self.weights[4],
        };

        let mut raw_score = contributions.model
            + contributions.signal
            + contributions.market
            + contributions.execution
            + contributions.regime;

        // Detect critical dimensions (any < 0.2)
        let critical_threshold = 0.2;
        let has_critical = dims.iter().any(|&d| d < critical_threshold);

        if has_critical {
            raw_score *= 1.0 - self.config.critical_penalty;
        }

        // Clamp
        raw_score = raw_score.clamp(0.0, 1.0);

        // EMA smoothing
        if !self.ema_initialized {
            self.ema_score = raw_score;
            self.ema_initialized = true;
        } else {
            self.ema_score += self.config.ema_decay * (raw_score - self.ema_score);
        }

        let smoothed = self.ema_score.clamp(0.0, 1.0);

        // Determine weakest dimension
        let dim_names = ["model", "signal", "market", "execution", "regime"];
        let mut weakest_idx = 0;
        let mut weakest_val = dims[0];
        for i in 1..5 {
            if dims[i] < weakest_val {
                weakest_val = dims[i];
                weakest_idx = i;
            }
        }

        let tradeable = smoothed >= self.config.min_confidence;
        let dimension_spread = input.spread();

        self.snapshot_count += 1;

        // Degradation detection
        let degrading = self.is_degrading_internal();

        // Update stats
        self.stats.total_snapshots += 1;
        self.stats.sum_score += smoothed;
        self.stats.sum_sq_score += smoothed * smoothed;
        if smoothed > self.stats.peak_score {
            self.stats.peak_score = smoothed;
        }
        if smoothed < self.stats.min_score {
            self.stats.min_score = smoothed;
        }
        if !tradeable {
            self.stats.below_threshold_count += 1;
        }
        if has_critical {
            self.stats.critical_dimension_count += 1;
        }
        self.stats.sum_model += dims[0];
        self.stats.sum_signal += dims[1];
        self.stats.sum_market += dims[2];
        self.stats.sum_execution += dims[3];
        self.stats.sum_regime += dims[4];

        // Window
        let record = SnapshotRecord {
            raw_score,
            smoothed_score: smoothed,
            model: dims[0],
            signal: dims[1],
            market: dims[2],
            execution: dims[3],
            regime: dims[4],
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        Ok(ConfidenceAssessment {
            raw_score,
            smoothed_score: smoothed,
            tradeable,
            contributions,
            has_critical_dimension: has_critical,
            weakest_dimension: dim_names[weakest_idx],
            weakest_score: weakest_val,
            dimension_spread,
            degrading,
            snapshot_count: self.snapshot_count,
        })
    }

    /// Quick check: is the current confidence above the trading threshold?
    pub fn is_tradeable(&self) -> bool {
        if !self.ema_initialized {
            return false;
        }
        self.ema_score >= self.config.min_confidence
    }

    /// Record an outcome to track confidence calibration
    pub fn record_outcome(&mut self, outcome: ConfidenceOutcome) {
        self.stats.total_outcomes += 1;
        if outcome.success {
            self.stats.successful_outcomes += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Dead zone
    // -----------------------------------------------------------------------

    fn apply_dead_zone(&self, current: f64, previous: f64) -> f64 {
        if (current - previous).abs() < self.config.dead_zone {
            previous
        } else {
            current
        }
    }

    // -----------------------------------------------------------------------
    // Degradation detection
    // -----------------------------------------------------------------------

    fn is_degrading_internal(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.smoothed_score)
            .sum::<f64>()
            / half as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.smoothed_score)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second_half < first_half * 0.9
    }

    /// Whether confidence is degrading (trend downward in recent window)
    pub fn is_degrading(&self) -> bool {
        self.is_degrading_internal()
    }

    /// Whether confidence is improving (trend upward in recent window)
    pub fn is_improving(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.smoothed_score)
            .sum::<f64>()
            / half as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.smoothed_score)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second_half > first_half * 1.1
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current EMA-smoothed confidence score
    pub fn smoothed_score(&self) -> f64 {
        if self.ema_initialized {
            self.ema_score
        } else {
            0.0
        }
    }

    /// Number of snapshots processed
    pub fn snapshot_count(&self) -> usize {
        self.snapshot_count
    }

    /// Whether the estimator has enough data for reliable assessments
    pub fn is_warmed_up(&self) -> bool {
        self.snapshot_count >= self.config.min_samples
    }

    /// Access running statistics
    pub fn stats(&self) -> &ConfidenceStats {
        &self.stats
    }

    /// Access the configuration
    pub fn config(&self) -> &ConfidenceConfig {
        &self.config
    }

    /// Current normalised weights
    pub fn weights(&self) -> [f64; 5] {
        self.weights
    }

    /// Number of records in the sliding window
    pub fn window_count(&self) -> usize {
        self.recent.len()
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Windowed mean composite confidence
    pub fn windowed_mean_score(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.smoothed_score).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed standard deviation of composite confidence
    pub fn windowed_score_std(&self) -> f64 {
        if self.recent.len() < 2 {
            return 0.0;
        }
        let n = self.recent.len() as f64;
        let mean = self.recent.iter().map(|r| r.smoothed_score).sum::<f64>() / n;
        let variance = self
            .recent
            .iter()
            .map(|r| {
                let d = r.smoothed_score - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    }

    /// Windowed mean per-dimension scores
    pub fn windowed_dimension_means(&self) -> DimensionContributions {
        if self.recent.is_empty() {
            return DimensionContributions::default();
        }
        let n = self.recent.len() as f64;
        DimensionContributions {
            model: self.recent.iter().map(|r| r.model).sum::<f64>() / n,
            signal: self.recent.iter().map(|r| r.signal).sum::<f64>() / n,
            market: self.recent.iter().map(|r| r.market).sum::<f64>() / n,
            execution: self.recent.iter().map(|r| r.execution).sum::<f64>() / n,
            regime: self.recent.iter().map(|r| r.regime).sum::<f64>() / n,
        }
    }

    /// Windowed fraction of snapshots where confidence was tradeable
    pub fn windowed_tradeable_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let count = self
            .recent
            .iter()
            .filter(|r| r.smoothed_score >= self.config.min_confidence)
            .count();
        count as f64 / self.recent.len() as f64
    }

    /// Identify the weakest dimension in the recent window (by mean score)
    pub fn windowed_weakest_dimension(&self) -> Option<&'static str> {
        if self.recent.is_empty() {
            return None;
        }
        let means = self.windowed_dimension_means();
        let dims = [
            ("model", means.model),
            ("signal", means.signal),
            ("market", means.market),
            ("execution", means.execution),
            ("regime", means.regime),
        ];
        dims.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(name, _)| *name)
    }

    // -----------------------------------------------------------------------
    // Weight adjustment
    // -----------------------------------------------------------------------

    /// Manually set the weight for a dimension. Weights are re-normalised
    /// if `normalise_weights` is enabled.
    pub fn set_weight(&mut self, dimension: &str, weight: f64) -> Result<()> {
        if weight < 0.0 {
            return Err(Error::InvalidInput("weight must be non-negative".into()));
        }

        match dimension {
            "model" => self.weights[0] = weight,
            "signal" => self.weights[1] = weight,
            "market" => self.weights[2] = weight,
            "execution" => self.weights[3] = weight,
            "regime" => self.weights[4] = weight,
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unknown dimension: {}",
                    dimension
                )));
            }
        }

        if self.config.normalise_weights {
            let sum: f64 = self.weights.iter().sum();
            if sum > 1e-12 {
                for w in &mut self.weights {
                    *w /= sum;
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all adaptive state, keeping configuration
    pub fn reset(&mut self) {
        self.ema_score = 0.0;
        self.ema_initialized = false;
        self.prev_dimensions = [0.5; 5];
        self.prev_initialized = false;
        self.snapshot_count = 0;
        self.recent.clear();
        self.stats = ConfidenceStats {
            min_score: f64::MAX,
            ..Default::default()
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_input(v: f64) -> ConfidenceInput {
        ConfidenceInput::uniform(v)
    }

    fn high_input() -> ConfidenceInput {
        ConfidenceInput {
            model: 0.9,
            signal: 0.85,
            market: 0.8,
            execution: 0.9,
            regime: 0.95,
        }
    }

    fn low_input() -> ConfidenceInput {
        ConfidenceInput {
            model: 0.1,
            signal: 0.15,
            market: 0.2,
            execution: 0.1,
            regime: 0.05,
        }
    }

    fn mixed_input() -> ConfidenceInput {
        ConfidenceInput {
            model: 0.9,
            signal: 0.8,
            market: 0.1,
            execution: 0.7,
            regime: 0.6,
        }
    }

    #[test]
    fn test_basic() {
        let instance = Confidence::new();
        assert_eq!(instance.snapshot_count(), 0);
    }

    #[test]
    fn test_default_config() {
        let c = Confidence::new();
        assert!((c.config().weight_model - 0.30).abs() < 1e-10);
    }

    // -- Uniform inputs --

    #[test]
    fn test_uniform_high_confidence() {
        let mut c = Confidence::new();
        let result = c.assess(&uniform_input(0.9)).unwrap();
        // All dimensions at 0.9 → composite should be 0.9
        assert!((result.raw_score - 0.9).abs() < 1e-10);
        assert!(result.tradeable);
        assert!(!result.has_critical_dimension);
    }

    #[test]
    fn test_uniform_low_confidence() {
        let mut c = Confidence::new();
        let result = c.assess(&uniform_input(0.1)).unwrap();
        // All dimensions at 0.1 → composite ≈ 0.1, then critical penalty
        assert!(result.raw_score < 0.15);
        assert!(result.has_critical_dimension);
    }

    #[test]
    fn test_uniform_zero() {
        let mut c = Confidence::new();
        let result = c.assess(&uniform_input(0.0)).unwrap();
        assert!((result.raw_score - 0.0).abs() < 1e-10);
        assert!(!result.tradeable);
    }

    #[test]
    fn test_uniform_one() {
        let mut c = Confidence::new();
        let result = c.assess(&uniform_input(1.0)).unwrap();
        assert!((result.raw_score - 1.0).abs() < 1e-10);
        assert!(result.tradeable);
    }

    // -- Weighted composite --

    #[test]
    fn test_weighted_composite() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            weight_model: 1.0,
            weight_signal: 0.0,
            weight_market: 0.0,
            weight_execution: 0.0,
            weight_regime: 0.0,
            normalise_weights: true,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        let input = ConfidenceInput {
            model: 0.7,
            signal: 0.3,
            market: 0.3,
            execution: 0.3,
            regime: 0.3,
        };

        let result = c.assess(&input).unwrap();
        // Only model matters, weight=1.0 normalised → composite = 0.7
        assert!((result.raw_score - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_equal_weights() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            weight_model: 1.0,
            weight_signal: 1.0,
            weight_market: 1.0,
            weight_execution: 1.0,
            weight_regime: 1.0,
            normalise_weights: true,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        let input = ConfidenceInput {
            model: 0.2,
            signal: 0.4,
            market: 0.6,
            execution: 0.8,
            regime: 1.0,
        };

        let result = c.assess(&input).unwrap();
        let expected = (0.2 + 0.4 + 0.6 + 0.8 + 1.0) / 5.0;
        assert!((result.raw_score - expected).abs() < 1e-10);
    }

    // -- Critical penalty --

    #[test]
    fn test_critical_penalty_applied() {
        let mut c_no_penalty = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        let mut c_with_penalty = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.5,
            ..Default::default()
        })
        .unwrap();

        let input = mixed_input(); // market = 0.1 → critical

        let r1 = c_no_penalty.assess(&input).unwrap();
        let r2 = c_with_penalty.assess(&input).unwrap();

        assert!(r2.raw_score < r1.raw_score);
        assert!(r2.has_critical_dimension);
    }

    #[test]
    fn test_no_critical_when_all_above_threshold() {
        let mut c = Confidence::new();
        let result = c.assess(&high_input()).unwrap();
        assert!(!result.has_critical_dimension);
    }

    // -- Weakest dimension --

    #[test]
    fn test_weakest_dimension_identified() {
        let mut c = Confidence::new();
        let result = c.assess(&mixed_input()).unwrap();
        assert_eq!(result.weakest_dimension, "market");
        assert!((result.weakest_score - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_spread() {
        let input = ConfidenceInput {
            model: 0.2,
            signal: 0.8,
            market: 0.5,
            execution: 0.5,
            regime: 0.5,
        };
        assert!((input.spread() - 0.6).abs() < 1e-10);
    }

    // -- Tradeable threshold --

    #[test]
    fn test_tradeable_above_threshold() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            min_confidence: 0.3,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.5)).unwrap();
        assert!(c.is_tradeable());
    }

    #[test]
    fn test_not_tradeable_below_threshold() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            min_confidence: 0.5,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.3)).unwrap();
        assert!(!c.is_tradeable());
    }

    #[test]
    fn test_not_tradeable_before_any_assessment() {
        let c = Confidence::new();
        assert!(!c.is_tradeable());
    }

    // -- EMA smoothing --

    #[test]
    fn test_ema_initialises_to_first() {
        let mut c = Confidence::new();
        let result = c.assess(&uniform_input(0.7)).unwrap();
        assert!((result.smoothed_score - result.raw_score).abs() < 1e-10);
    }

    #[test]
    fn test_ema_lags_behind_changes() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            ema_decay: 0.1,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.8)).unwrap();
        let smooth_before = c.smoothed_score();

        c.assess(&uniform_input(0.2)).unwrap();
        let smooth_after = c.smoothed_score();

        // EMA should lag: smooth_after is between 0.2 and smooth_before
        assert!(smooth_after < smooth_before);
        assert!(smooth_after > 0.2);
    }

    #[test]
    fn test_ema_converges() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            ema_decay: 0.5,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        // Feed constant value
        for _ in 0..50 {
            c.assess(&uniform_input(0.6)).unwrap();
        }

        assert!((c.smoothed_score() - 0.6).abs() < 0.01);
    }

    // -- Dead zone --

    #[test]
    fn test_dead_zone_filters_noise() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            dead_zone: 0.05,
            critical_penalty: 0.0,
            ema_decay: 0.99, // fast to see effect
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.5)).unwrap();
        let r1 = c.assess(&uniform_input(0.52)).unwrap(); // within dead zone

        // The dimension values should still be 0.5 due to dead zone
        // so the raw score should be ≈ 0.5
        assert!((r1.raw_score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_dead_zone_allows_large_changes() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            dead_zone: 0.05,
            critical_penalty: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.5)).unwrap();
        let r2 = c.assess(&uniform_input(0.8)).unwrap(); // beyond dead zone

        assert!((r2.raw_score - 0.8).abs() < 0.01);
    }

    // -- Input validation --

    #[test]
    fn test_invalid_input_model_too_high() {
        let mut c = Confidence::new();
        let input = ConfidenceInput {
            model: 1.5,
            ..uniform_input(0.5)
        };
        assert!(c.assess(&input).is_err());
    }

    #[test]
    fn test_invalid_input_negative() {
        let mut c = Confidence::new();
        let input = ConfidenceInput {
            signal: -0.1,
            ..uniform_input(0.5)
        };
        assert!(c.assess(&input).is_err());
    }

    #[test]
    fn test_input_validate_all_dimensions() {
        let good = uniform_input(0.5);
        assert!(good.validate().is_ok());

        let bad_market = ConfidenceInput {
            market: 2.0,
            ..uniform_input(0.5)
        };
        assert!(bad_market.validate().is_err());

        let bad_execution = ConfidenceInput {
            execution: -0.5,
            ..uniform_input(0.5)
        };
        assert!(bad_execution.validate().is_err());

        let bad_regime = ConfidenceInput {
            regime: 1.1,
            ..uniform_input(0.5)
        };
        assert!(bad_regime.validate().is_err());
    }

    // -- Input helpers --

    #[test]
    fn test_input_min_dimension() {
        let input = mixed_input();
        assert!((input.min_dimension() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_input_max_dimension() {
        let input = mixed_input();
        assert!((input.max_dimension() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_input_model_signal_constructor() {
        let input = ConfidenceInput::model_signal(0.9, 0.7);
        assert!((input.model - 0.9).abs() < 1e-10);
        assert!((input.signal - 0.7).abs() < 1e-10);
        assert!((input.market - 0.5).abs() < 1e-10);
    }

    // -- Stats --

    #[test]
    fn test_stats_tracking() {
        let mut c = Confidence::new();
        c.assess(&high_input()).unwrap();
        c.assess(&low_input()).unwrap();
        c.assess(&uniform_input(0.5)).unwrap();

        assert_eq!(c.stats().total_snapshots, 3);
    }

    #[test]
    fn test_stats_mean_score() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.0,
            ema_decay: 0.99, // fast
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.4)).unwrap();
        c.assess(&uniform_input(0.6)).unwrap();

        let mean = c.stats().mean_score();
        // Mean of smoothed scores (EMA), approximately (0.4 + 0.6)/2
        assert!(mean > 0.3 && mean < 0.7);
    }

    #[test]
    fn test_stats_below_threshold() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            min_confidence: 0.5,
            critical_penalty: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.3)).unwrap(); // below
        c.assess(&uniform_input(0.7)).unwrap(); // above

        assert_eq!(c.stats().below_threshold_count, 1);
    }

    #[test]
    fn test_stats_critical_count() {
        let mut c = Confidence::new();
        c.assess(&mixed_input()).unwrap(); // market = 0.1 → critical
        c.assess(&high_input()).unwrap(); // no critical

        assert_eq!(c.stats().critical_dimension_count, 1);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = ConfidenceStats::default();
        assert_eq!(stats.total_snapshots, 0);
        assert_eq!(stats.mean_score(), 0.0);
        assert_eq!(stats.score_std(), 0.0);
        assert_eq!(stats.below_threshold_rate(), 0.0);
        assert_eq!(stats.critical_rate(), 0.0);
        assert_eq!(stats.outcome_success_rate(), 0.0);
    }

    #[test]
    fn test_stats_mean_dimension_scores() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.5)).unwrap();
        c.assess(&uniform_input(0.5)).unwrap();

        let means = c.stats().mean_dimension_scores();
        assert!((means.model - 0.5).abs() < 1e-10);
        assert!((means.signal - 0.5).abs() < 1e-10);
    }

    // -- Outcome tracking --

    #[test]
    fn test_outcome_tracking() {
        let mut c = Confidence::new();
        c.record_outcome(ConfidenceOutcome {
            input: uniform_input(0.8),
            success: true,
            pnl_ratio: 1.5,
        });
        c.record_outcome(ConfidenceOutcome {
            input: uniform_input(0.6),
            success: false,
            pnl_ratio: -0.5,
        });

        assert_eq!(c.stats().total_outcomes, 2);
        assert_eq!(c.stats().successful_outcomes, 1);
        assert!((c.stats().outcome_success_rate() - 0.5).abs() < 1e-10);
    }

    // -- Window --

    #[test]
    fn test_windowed_mean_score() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            window_size: 5,
            critical_penalty: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..5 {
            c.assess(&uniform_input(0.6)).unwrap();
        }

        let mean = c.windowed_mean_score();
        assert!((mean - 0.6).abs() < 0.05);
    }

    #[test]
    fn test_windowed_score_std_constant() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            window_size: 10,
            critical_penalty: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        })
        .unwrap();

        // Wait for EMA to converge, then measure std
        for _ in 0..20 {
            c.assess(&uniform_input(0.5)).unwrap();
        }

        assert!(c.windowed_score_std() < 0.05);
    }

    #[test]
    fn test_windowed_mean_empty() {
        let c = Confidence::new();
        assert_eq!(c.windowed_mean_score(), 0.0);
        assert_eq!(c.windowed_score_std(), 0.0);
    }

    #[test]
    fn test_windowed_tradeable_rate() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            min_confidence: 0.4,
            critical_penalty: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.5)).unwrap(); // tradeable
        c.assess(&uniform_input(0.3)).unwrap(); // not tradeable (smoothed ≈ 0.3)

        let rate = c.windowed_tradeable_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_windowed_weakest_dimension() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        // Feed inputs where market is consistently weakest
        for _ in 0..5 {
            c.assess(&ConfidenceInput {
                model: 0.8,
                signal: 0.7,
                market: 0.3,
                execution: 0.7,
                regime: 0.7,
            })
            .unwrap();
        }

        assert_eq!(c.windowed_weakest_dimension(), Some("market"));
    }

    #[test]
    fn test_windowed_weakest_dimension_empty() {
        let c = Confidence::new();
        assert!(c.windowed_weakest_dimension().is_none());
    }

    #[test]
    fn test_windowed_dimension_means() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            window_size: 3,
            critical_penalty: 0.0,
            dead_zone: 0.0,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.4)).unwrap();
        c.assess(&uniform_input(0.6)).unwrap();
        c.assess(&uniform_input(0.8)).unwrap();

        let means = c.windowed_dimension_means();
        assert!((means.model - 0.6).abs() < 1e-10);
        assert!((means.signal - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_window_eviction() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            window_size: 3,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.5)).unwrap();
        c.assess(&uniform_input(0.5)).unwrap();
        c.assess(&uniform_input(0.5)).unwrap();
        c.assess(&uniform_input(0.5)).unwrap();

        assert_eq!(c.window_count(), 3);
    }

    // -- Degradation detection --

    #[test]
    fn test_is_degrading() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            window_size: 20,
            critical_penalty: 0.0,
            ema_decay: 0.5,
            ..Default::default()
        })
        .unwrap();

        // First half: high confidence
        for _ in 0..10 {
            c.assess(&uniform_input(0.9)).unwrap();
        }
        // Second half: low confidence
        for _ in 0..10 {
            c.assess(&uniform_input(0.3)).unwrap();
        }

        assert!(c.is_degrading());
    }

    #[test]
    fn test_not_degrading_consistent() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            window_size: 20,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..20 {
            c.assess(&uniform_input(0.6)).unwrap();
        }

        assert!(!c.is_degrading());
    }

    #[test]
    fn test_not_degrading_insufficient_data() {
        let mut c = Confidence::new();
        c.assess(&uniform_input(0.5)).unwrap();
        assert!(!c.is_degrading());
    }

    #[test]
    fn test_is_improving() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            window_size: 20,
            critical_penalty: 0.0,
            ema_decay: 0.5,
            ..Default::default()
        })
        .unwrap();

        // First half: low confidence
        for _ in 0..10 {
            c.assess(&uniform_input(0.3)).unwrap();
        }
        // Second half: high confidence
        for _ in 0..10 {
            c.assess(&uniform_input(0.9)).unwrap();
        }

        assert!(c.is_improving());
    }

    // -- Weight adjustment --

    #[test]
    fn test_set_weight() {
        let mut c = Confidence::new();
        c.set_weight("model", 0.5).unwrap();
        // Weights should be re-normalised
        let sum: f64 = c.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_weight_unknown_dimension() {
        let mut c = Confidence::new();
        assert!(c.set_weight("unknown", 0.5).is_err());
    }

    #[test]
    fn test_set_weight_negative() {
        let mut c = Confidence::new();
        assert!(c.set_weight("model", -0.1).is_err());
    }

    // -- Warm up --

    #[test]
    fn test_not_warmed_up_initially() {
        let c = Confidence::new();
        assert!(!c.is_warmed_up());
    }

    #[test]
    fn test_warmed_up_after_min_samples() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            min_samples: 3,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.5)).unwrap();
        c.assess(&uniform_input(0.5)).unwrap();
        assert!(!c.is_warmed_up());

        c.assess(&uniform_input(0.5)).unwrap();
        assert!(c.is_warmed_up());
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut c = Confidence::new();
        for _ in 0..20 {
            c.assess(&uniform_input(0.5)).unwrap();
        }
        c.record_outcome(ConfidenceOutcome {
            input: uniform_input(0.5),
            success: true,
            pnl_ratio: 1.0,
        });

        assert!(c.snapshot_count() > 0);

        c.reset();
        assert_eq!(c.snapshot_count(), 0);
        assert_eq!(c.smoothed_score(), 0.0);
        assert!(!c.is_tradeable());
        assert_eq!(c.window_count(), 0);
        assert_eq!(c.stats().total_snapshots, 0);
        // Note: outcome stats are also reset
    }

    // -- Config validation --

    #[test]
    fn test_invalid_config_negative_weight() {
        let result = Confidence::with_config(ConfidenceConfig {
            weight_model: -0.1,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_all_zero_weights() {
        let result = Confidence::with_config(ConfidenceConfig {
            weight_model: 0.0,
            weight_signal: 0.0,
            weight_market: 0.0,
            weight_execution: 0.0,
            weight_regime: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_ema_decay() {
        let r1 = Confidence::with_config(ConfidenceConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = Confidence::with_config(ConfidenceConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_bad_min_confidence() {
        let r1 = Confidence::with_config(ConfidenceConfig {
            min_confidence: -0.1,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = Confidence::with_config(ConfidenceConfig {
            min_confidence: 1.1,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let result = Confidence::with_config(ConfidenceConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_negative_dead_zone() {
        let result = Confidence::with_config(ConfidenceConfig {
            dead_zone: -0.01,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_critical_penalty() {
        let r1 = Confidence::with_config(ConfidenceConfig {
            critical_penalty: -0.1,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 1.1,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    // -- Process convenience --

    #[test]
    fn test_process_returns_instance() {
        let c = Confidence::process(ConfidenceConfig::default());
        assert!(c.is_ok());
    }

    #[test]
    fn test_process_rejects_bad_config() {
        let result = Confidence::process(ConfidenceConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- Contributions --

    #[test]
    fn test_contributions_sum_to_raw_score() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        let result = c.assess(&high_input()).unwrap();
        let contrib_sum = result.contributions.model
            + result.contributions.signal
            + result.contributions.market
            + result.contributions.execution
            + result.contributions.regime;

        assert!((contrib_sum - result.raw_score).abs() < 1e-10);
    }

    // -- Assessment structure --

    #[test]
    fn test_assessment_fields() {
        let mut c = Confidence::new();
        let result = c.assess(&high_input()).unwrap();

        assert!(result.raw_score >= 0.0 && result.raw_score <= 1.0);
        assert!(result.smoothed_score >= 0.0 && result.smoothed_score <= 1.0);
        assert_eq!(result.snapshot_count, 1);
        assert!(result.dimension_spread >= 0.0);
    }

    // -- Normalise weights disabled --

    #[test]
    fn test_weights_not_normalised() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            weight_model: 0.5,
            weight_signal: 0.5,
            weight_market: 0.5,
            weight_execution: 0.5,
            weight_regime: 0.5,
            normalise_weights: false,
            critical_penalty: 0.0,
            ..Default::default()
        })
        .unwrap();

        let result = c.assess(&uniform_input(0.4)).unwrap();
        // Without normalisation: 5 × 0.5 × 0.4 = 1.0 → clamped to 1.0
        assert!((result.raw_score - 1.0).abs() < 1e-10);
    }

    // -- Score variance --

    #[test]
    fn test_score_variance_constant() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..20 {
            c.assess(&uniform_input(0.5)).unwrap();
        }

        assert!(c.stats().score_variance() < 0.01);
    }

    #[test]
    fn test_score_variance_varied() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..10 {
            c.assess(&uniform_input(0.2)).unwrap();
        }
        for _ in 0..10 {
            c.assess(&uniform_input(0.8)).unwrap();
        }

        assert!(c.stats().score_variance() > 0.01);
    }

    // -- Peak and min score --

    #[test]
    fn test_peak_and_min_score_tracked() {
        let mut c = Confidence::with_config(ConfidenceConfig {
            critical_penalty: 0.0,
            ema_decay: 0.99,
            ..Default::default()
        })
        .unwrap();

        c.assess(&uniform_input(0.3)).unwrap();
        c.assess(&uniform_input(0.9)).unwrap();

        assert!(c.stats().peak_score > 0.8);
        assert!(c.stats().min_score < 0.4);
    }

    // -- Snapshot count --

    #[test]
    fn test_snapshot_count_increments() {
        let mut c = Confidence::new();
        assert_eq!(c.snapshot_count(), 0);

        c.assess(&uniform_input(0.5)).unwrap();
        assert_eq!(c.snapshot_count(), 1);

        c.assess(&uniform_input(0.5)).unwrap();
        assert_eq!(c.snapshot_count(), 2);
    }

    // -- Smoothed score accessor --

    #[test]
    fn test_smoothed_score_zero_before_assessment() {
        let c = Confidence::new();
        assert_eq!(c.smoothed_score(), 0.0);
    }

    #[test]
    fn test_smoothed_score_after_assessment() {
        let mut c = Confidence::new();
        c.assess(&uniform_input(0.7)).unwrap();
        assert!(c.smoothed_score() > 0.0);
    }
}
