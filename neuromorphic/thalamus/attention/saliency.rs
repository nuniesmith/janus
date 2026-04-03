//! Saliency map generation
//!
//! Part of the Thalamus region
//! Component: attention
//!
//! Generates a saliency map over a set of named market signals, scoring each
//! by its surprise (deviation from expectation), novelty (how recently it
//! changed), and raw magnitude.  The composite saliency score determines
//! which signals deserve the system's attention.
//!
//! ## Features
//!
//! - **Surprise scoring**: Measures how far a signal deviates from its
//!   EMA-smoothed baseline — large deviations are more salient.
//! - **Novelty scoring**: Tracks how recently each signal changed
//!   significantly; fresh changes are more salient than stale ones.
//! - **Magnitude scoring**: Raw absolute signal level contributes to
//!   saliency so that large signals are not ignored even if they are
//!   expected.
//! - **Configurable weights**: The three components (surprise, novelty,
//!   magnitude) can be weighted independently.
//! - **Temporal decay**: Saliency scores decay exponentially over time
//!   if a signal is not refreshed, preventing stale highlights.
//! - **Top-K selection**: Returns only the K most salient signals to
//!   bound downstream processing.
//! - **Per-signal statistics**: Tracks each signal's baseline, variance,
//!   peak saliency, and refresh count.
//! - **EMA-smoothed aggregate saliency**: The mean saliency across all
//!   signals is smoothed for trend monitoring.

use std::collections::{HashMap, VecDeque};

use crate::common::{Error, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the saliency map generator.
#[derive(Debug, Clone)]
pub struct SaliencyConfig {
    /// EMA decay factor for the per-signal baseline estimator (0, 1).
    /// Closer to 1 → slower adaptation to new signal levels.
    pub baseline_ema_decay: f64,
    /// EMA decay factor for the per-signal variance estimator (0, 1).
    pub variance_ema_decay: f64,
    /// Weight of the surprise component in the composite score.
    pub surprise_weight: f64,
    /// Weight of the novelty component in the composite score.
    pub novelty_weight: f64,
    /// Weight of the magnitude component in the composite score.
    pub magnitude_weight: f64,
    /// Decay rate per tick applied to the novelty score.  Must be in (0, 1).
    /// Each tick, novelty is multiplied by `(1 - novelty_decay)`.
    pub novelty_decay: f64,
    /// Threshold on the relative change (|new - old| / baseline) above
    /// which a signal update is considered "novel".  Below this, the
    /// novelty counter is not refreshed.
    pub novelty_change_threshold: f64,
    /// Number of top-K signals to return from `compute()`.
    pub top_k: usize,
    /// EMA decay for the smoothed aggregate saliency (0, 1).
    /// Set to 0 to disable smoothing.
    pub ema_decay: f64,
    /// Maximum number of recent aggregate saliency values to retain for
    /// windowed statistics.
    pub window_size: usize,
    /// Minimum variance floor to prevent division by zero in surprise
    /// computation.
    pub min_variance: f64,
    /// Maximum magnitude reference: signal magnitudes are divided by this
    /// value before scoring.  This normalises the magnitude component to
    /// roughly [0, 1].  If 0, magnitude scoring is disabled.
    pub magnitude_reference: f64,
}

impl Default for SaliencyConfig {
    fn default() -> Self {
        Self {
            baseline_ema_decay: 0.9,
            variance_ema_decay: 0.9,
            surprise_weight: 1.0,
            novelty_weight: 0.5,
            magnitude_weight: 0.3,
            novelty_decay: 0.2,
            novelty_change_threshold: 0.01,
            top_k: 5,
            ema_decay: 0.8,
            window_size: 200,
            min_variance: 1e-12,
            magnitude_reference: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// A single signal observation to update the saliency map.
#[derive(Debug, Clone)]
pub struct SignalObservation {
    /// Identifier for the signal (e.g. "btc_return", "eth_spread",
    /// "volume_spike").
    pub signal_id: String,
    /// Current value of the signal.
    pub value: f64,
}

/// A scored entry in the saliency map.
#[derive(Debug, Clone)]
pub struct SaliencyEntry {
    /// Signal identifier.
    pub signal_id: String,
    /// Composite saliency score (higher = more salient).
    pub score: f64,
    /// Surprise component (z-score magnitude).
    pub surprise: f64,
    /// Novelty component (recency of significant change).
    pub novelty: f64,
    /// Magnitude component (normalised absolute level).
    pub magnitude: f64,
    /// Current baseline estimate for this signal.
    pub baseline: f64,
    /// Current variance estimate for this signal.
    pub variance: f64,
}

/// Output of a saliency computation.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// Top-K most salient signals, sorted by score descending.
    pub entries: Vec<SaliencyEntry>,
    /// Mean saliency across all tracked signals (not just top-K).
    pub mean_saliency: f64,
    /// EMA-smoothed mean saliency.
    pub smoothed_mean_saliency: f64,
    /// Maximum saliency score in this snapshot.
    pub max_saliency: f64,
    /// Total number of tracked signals.
    pub total_signals: usize,
}

/// Per-signal statistics.
#[derive(Debug, Clone)]
pub struct SignalStats {
    /// Total observations ingested for this signal.
    pub observations: usize,
    /// Current baseline (EMA of values).
    pub baseline: f64,
    /// Current variance estimate.
    pub variance: f64,
    /// Peak saliency score ever observed.
    pub peak_saliency: f64,
    /// Current novelty level.
    pub novelty: f64,
    /// Most recent value.
    pub last_value: f64,
}

/// Aggregate statistics for the saliency engine.
#[derive(Debug, Clone, Default)]
pub struct SaliencyStats {
    /// Total observations ingested across all signals.
    pub total_observations: usize,
    /// Total saliency computations performed.
    pub total_computations: usize,
    /// Number of distinct signals tracked.
    pub distinct_signals: usize,
    /// Cumulative sum of mean saliency values (for averaging).
    pub sum_mean_saliency: f64,
}

impl SaliencyStats {
    /// Average mean saliency across all computations.
    pub fn avg_mean_saliency(&self) -> f64 {
        if self.total_computations == 0 {
            0.0
        } else {
            self.sum_mean_saliency / self.total_computations as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SignalRecord {
    value: f64,
    baseline: f64,
    variance: f64,
    baseline_initialized: bool,
    novelty: f64,
    observations: usize,
    peak_saliency: f64,
}

// ---------------------------------------------------------------------------
// Saliency
// ---------------------------------------------------------------------------

/// Saliency map generator.
///
/// Call [`observe`] to feed signal values, then call [`compute`] to obtain
/// the top-K most salient signals with their scores.
pub struct Saliency {
    config: SaliencyConfig,
    /// Per-signal records.
    signals: HashMap<String, SignalRecord>,
    /// EMA state for smoothed aggregate saliency.
    ema_mean_saliency: f64,
    ema_initialized: bool,
    /// Windowed history of mean saliency values.
    history: VecDeque<f64>,
    /// Running statistics.
    stats: SaliencyStats,
}

impl Default for Saliency {
    fn default() -> Self {
        Self::new()
    }
}

impl Saliency {
    /// Create a new instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(SaliencyConfig::default())
    }

    /// Create a new instance with the given configuration.
    pub fn with_config(config: SaliencyConfig) -> Self {
        Self {
            signals: HashMap::new(),
            ema_mean_saliency: 0.0,
            ema_initialized: false,
            history: VecDeque::with_capacity(config.window_size),
            stats: SaliencyStats::default(),
            config,
        }
    }

    /// Validate configuration parameters.
    pub fn process(&self) -> Result<()> {
        if self.config.baseline_ema_decay <= 0.0 || self.config.baseline_ema_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "baseline_ema_decay must be in (0, 1)".into(),
            ));
        }
        if self.config.variance_ema_decay <= 0.0 || self.config.variance_ema_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "variance_ema_decay must be in (0, 1)".into(),
            ));
        }
        if self.config.surprise_weight < 0.0 {
            return Err(Error::InvalidInput("surprise_weight must be >= 0".into()));
        }
        if self.config.novelty_weight < 0.0 {
            return Err(Error::InvalidInput("novelty_weight must be >= 0".into()));
        }
        if self.config.magnitude_weight < 0.0 {
            return Err(Error::InvalidInput("magnitude_weight must be >= 0".into()));
        }
        if self.config.novelty_decay <= 0.0 || self.config.novelty_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "novelty_decay must be in (0, 1)".into(),
            ));
        }
        if self.config.novelty_change_threshold < 0.0 {
            return Err(Error::InvalidInput(
                "novelty_change_threshold must be >= 0".into(),
            ));
        }
        if self.config.top_k == 0 {
            return Err(Error::InvalidInput("top_k must be > 0".into()));
        }
        if self.config.ema_decay < 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in [0, 1)".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.min_variance < 0.0 {
            return Err(Error::InvalidInput("min_variance must be >= 0".into()));
        }
        Ok(())
    }

    // -- Observation -------------------------------------------------------

    /// Observe a single signal value.
    pub fn observe(&mut self, obs: &SignalObservation) {
        self.stats.total_observations += 1;

        if !self.signals.contains_key(&obs.signal_id) {
            self.stats.distinct_signals += 1;
        }

        let alpha_b = self.config.baseline_ema_decay;
        let alpha_v = self.config.variance_ema_decay;

        if let Some(record) = self.signals.get_mut(&obs.signal_id) {
            let old_value = record.value;
            record.value = obs.value;
            record.observations += 1;

            if record.baseline_initialized {
                // Update baseline EMA.
                let old_baseline = record.baseline;
                record.baseline = alpha_b * record.baseline + (1.0 - alpha_b) * obs.value;

                // Update variance EMA.
                let deviation = obs.value - old_baseline;
                record.variance =
                    alpha_v * record.variance + (1.0 - alpha_v) * deviation * deviation;
            } else {
                record.baseline = obs.value;
                record.variance = 0.0;
                record.baseline_initialized = true;
            }

            // Update novelty: refresh if change is significant.
            let rel_change = if record.baseline.abs() > 1e-30 {
                (obs.value - old_value).abs() / record.baseline.abs()
            } else {
                (obs.value - old_value).abs()
            };
            if rel_change > self.config.novelty_change_threshold {
                record.novelty = 1.0;
            }
        } else {
            self.signals.insert(
                obs.signal_id.clone(),
                SignalRecord {
                    value: obs.value,
                    baseline: obs.value,
                    variance: 0.0,
                    baseline_initialized: true,
                    novelty: 1.0, // first observation is novel
                    observations: 1,
                    peak_saliency: 0.0,
                },
            );
        }
    }

    /// Observe a batch of signals.
    pub fn observe_batch(&mut self, observations: &[SignalObservation]) {
        for obs in observations {
            self.observe(obs);
        }
    }

    // -- Computation -------------------------------------------------------

    /// Advance one tick and compute the saliency map.
    ///
    /// This decays novelty scores and produces the top-K saliency ranking.
    pub fn compute(&mut self) -> SaliencyMap {
        self.stats.total_computations += 1;

        let mut all_entries: Vec<SaliencyEntry> = Vec::with_capacity(self.signals.len());

        for (id, record) in &mut self.signals {
            // Surprise: z-score magnitude.
            let std_dev = (record.variance + self.config.min_variance).sqrt();
            let surprise = ((record.value - record.baseline) / std_dev).abs();

            // Magnitude: normalised absolute level.
            let magnitude = if self.config.magnitude_reference > 0.0 {
                record.value.abs() / self.config.magnitude_reference
            } else {
                0.0
            };

            // Composite score.
            let score = self.config.surprise_weight * surprise
                + self.config.novelty_weight * record.novelty
                + self.config.magnitude_weight * magnitude;

            // Track peak saliency.
            if score > record.peak_saliency {
                record.peak_saliency = score;
            }

            all_entries.push(SaliencyEntry {
                signal_id: id.clone(),
                score,
                surprise,
                novelty: record.novelty,
                magnitude,
                baseline: record.baseline,
                variance: record.variance,
            });

            // Decay novelty for next tick.
            record.novelty *= 1.0 - self.config.novelty_decay;
        }

        // Sort by score descending.
        all_entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Aggregate metrics.
        let total_signals = all_entries.len();
        let mean_saliency = if total_signals > 0 {
            all_entries.iter().map(|e| e.score).sum::<f64>() / total_signals as f64
        } else {
            0.0
        };
        let max_saliency = all_entries.first().map(|e| e.score).unwrap_or(0.0);

        // EMA smoothing of mean saliency.
        let smoothed = if self.config.ema_decay > 0.0 {
            if self.ema_initialized {
                let s = self.config.ema_decay * self.ema_mean_saliency
                    + (1.0 - self.config.ema_decay) * mean_saliency;
                self.ema_mean_saliency = s;
                s
            } else {
                self.ema_mean_saliency = mean_saliency;
                self.ema_initialized = true;
                mean_saliency
            }
        } else {
            mean_saliency
        };

        // History.
        self.history.push_back(mean_saliency);
        while self.history.len() > self.config.window_size {
            self.history.pop_front();
        }

        self.stats.sum_mean_saliency += mean_saliency;

        // Top-K selection.
        all_entries.truncate(self.config.top_k);

        SaliencyMap {
            entries: all_entries,
            mean_saliency,
            smoothed_mean_saliency: smoothed,
            max_saliency,
            total_signals,
        }
    }

    // -- Accessors ---------------------------------------------------------

    /// Get per-signal statistics.
    pub fn signal_stats(&self) -> HashMap<String, SignalStats> {
        self.signals
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    SignalStats {
                        observations: r.observations,
                        baseline: r.baseline,
                        variance: r.variance,
                        peak_saliency: r.peak_saliency,
                        novelty: r.novelty,
                        last_value: r.value,
                    },
                )
            })
            .collect()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> &SaliencyStats {
        &self.stats
    }

    /// Number of distinct signals currently tracked.
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }

    /// Windowed mean of recent mean-saliency values.
    pub fn windowed_mean(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().sum();
        Some(sum / self.history.len() as f64)
    }

    /// Windowed standard deviation of recent mean-saliency values.
    pub fn windowed_std(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let mean = self.windowed_mean().unwrap();
        let var: f64 = self.history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Get the saliency score for a specific signal.
    ///
    /// This recomputes the score on-the-fly from the signal's current state
    /// without decaying novelty.  Returns 0 if the signal is not tracked.
    pub fn score_for(&self, signal_id: &str) -> f64 {
        let Some(record) = self.signals.get(signal_id) else {
            return 0.0;
        };
        let std_dev = (record.variance + self.config.min_variance).sqrt();
        let surprise = ((record.value - record.baseline) / std_dev).abs();
        let magnitude = if self.config.magnitude_reference > 0.0 {
            record.value.abs() / self.config.magnitude_reference
        } else {
            0.0
        };
        self.config.surprise_weight * surprise
            + self.config.novelty_weight * record.novelty
            + self.config.magnitude_weight * magnitude
    }

    /// Remove a signal from the saliency map.
    pub fn remove_signal(&mut self, signal_id: &str) {
        self.signals.remove(signal_id);
    }

    /// Reset all state (signals, EMA, history, stats).
    pub fn reset(&mut self) {
        self.signals.clear();
        self.ema_mean_saliency = 0.0;
        self.ema_initialized = false;
        self.history.clear();
        self.stats = SaliencyStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(id: &str, value: f64) -> SignalObservation {
        SignalObservation {
            signal_id: id.to_string(),
            value,
        }
    }

    fn default_config() -> SaliencyConfig {
        SaliencyConfig {
            baseline_ema_decay: 0.5,
            variance_ema_decay: 0.5,
            surprise_weight: 1.0,
            novelty_weight: 0.5,
            magnitude_weight: 0.3,
            novelty_decay: 0.3,
            novelty_change_threshold: 0.01,
            top_k: 5,
            ema_decay: 0.5,
            window_size: 100,
            min_variance: 1e-12,
            magnitude_reference: 100.0,
        }
    }

    fn default_saliency() -> Saliency {
        Saliency::with_config(default_config())
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = Saliency::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_process_invalid_baseline_ema_decay_zero() {
        let s = Saliency::with_config(SaliencyConfig {
            baseline_ema_decay: 0.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_baseline_ema_decay_one() {
        let s = Saliency::with_config(SaliencyConfig {
            baseline_ema_decay: 1.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_variance_ema_decay() {
        let s = Saliency::with_config(SaliencyConfig {
            variance_ema_decay: 0.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_surprise_weight() {
        let s = Saliency::with_config(SaliencyConfig {
            surprise_weight: -1.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_novelty_weight() {
        let s = Saliency::with_config(SaliencyConfig {
            novelty_weight: -1.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_magnitude_weight() {
        let s = Saliency::with_config(SaliencyConfig {
            magnitude_weight: -1.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_novelty_decay_zero() {
        let s = Saliency::with_config(SaliencyConfig {
            novelty_decay: 0.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_novelty_decay_one() {
        let s = Saliency::with_config(SaliencyConfig {
            novelty_decay: 1.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_novelty_change_threshold() {
        let s = Saliency::with_config(SaliencyConfig {
            novelty_change_threshold: -1.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_top_k() {
        let s = Saliency::with_config(SaliencyConfig {
            top_k: 0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_ema_decay() {
        let s = Saliency::with_config(SaliencyConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_window_size() {
        let s = Saliency::with_config(SaliencyConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_invalid_min_variance() {
        let s = Saliency::with_config(SaliencyConfig {
            min_variance: -1.0,
            ..Default::default()
        });
        assert!(s.process().is_err());
    }

    #[test]
    fn test_process_valid_ema_decay_zero() {
        let s = Saliency::with_config(SaliencyConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(s.process().is_ok());
    }

    #[test]
    fn test_process_valid_min_variance_zero() {
        let s = Saliency::with_config(SaliencyConfig {
            min_variance: 0.0,
            ..Default::default()
        });
        assert!(s.process().is_ok());
    }

    // -- Observation -------------------------------------------------------

    #[test]
    fn test_observe_creates_signal() {
        let mut s = default_saliency();
        s.observe(&sig("btc_return", 0.05));
        assert_eq!(s.signal_count(), 1);
        assert_eq!(s.stats().total_observations, 1);
    }

    #[test]
    fn test_observe_multiple_signals() {
        let mut s = default_saliency();
        s.observe(&sig("btc_return", 0.05));
        s.observe(&sig("eth_return", -0.02));
        s.observe(&sig("vol_spike", 1.5));
        assert_eq!(s.signal_count(), 3);
        assert_eq!(s.stats().distinct_signals, 3);
    }

    #[test]
    fn test_observe_updates_existing() {
        let mut s = default_saliency();
        s.observe(&sig("btc", 100.0));
        s.observe(&sig("btc", 105.0));
        assert_eq!(s.signal_count(), 1);
        assert_eq!(s.stats().total_observations, 2);

        let ss = s.signal_stats();
        assert_eq!(ss["btc"].observations, 2);
        assert!((ss["btc"].last_value - 105.0).abs() < 1e-12);
    }

    #[test]
    fn test_observe_batch() {
        let mut s = default_saliency();
        s.observe_batch(&[sig("a", 1.0), sig("b", 2.0), sig("c", 3.0)]);
        assert_eq!(s.signal_count(), 3);
        assert_eq!(s.stats().total_observations, 3);
    }

    // -- Surprise scoring --------------------------------------------------

    #[test]
    fn test_surprise_increases_with_deviation() {
        let mut s = default_saliency();

        // Build a baseline around 100.
        for _ in 0..20 {
            s.observe(&sig("price", 100.0));
            s.compute();
        }

        // Small deviation.
        s.observe(&sig("price", 101.0));
        let map_small = s.compute();
        let small_score = map_small
            .entries
            .iter()
            .find(|e| e.signal_id == "price")
            .unwrap()
            .surprise;

        // Reset and build same baseline.
        s.reset();
        for _ in 0..20 {
            s.observe(&sig("price", 100.0));
            s.compute();
        }

        // Large deviation.
        s.observe(&sig("price", 150.0));
        let map_large = s.compute();
        let large_score = map_large
            .entries
            .iter()
            .find(|e| e.signal_id == "price")
            .unwrap()
            .surprise;

        assert!(
            large_score > small_score,
            "larger deviation should be more surprising: large={}, small={}",
            large_score,
            small_score
        );
    }

    #[test]
    fn test_surprise_zero_for_constant_signal() {
        let mut s = default_saliency();
        // After many constant observations, surprise should be near zero.
        for _ in 0..50 {
            s.observe(&sig("const", 42.0));
            s.compute();
        }
        let map = s.compute();
        if let Some(entry) = map.entries.iter().find(|e| e.signal_id == "const") {
            // Surprise should be very small (value == baseline).
            // With EMA adaptation, baseline converges to value, so deviation → 0.
            assert!(
                entry.surprise < 0.5,
                "constant signal should have low surprise, got {}",
                entry.surprise
            );
        }
    }

    // -- Novelty scoring ---------------------------------------------------

    #[test]
    fn test_novelty_high_on_first_observation() {
        let mut s = default_saliency();
        s.observe(&sig("new_signal", 1.0));
        let map = s.compute();
        let entry = map
            .entries
            .iter()
            .find(|e| e.signal_id == "new_signal")
            .unwrap();
        // First observation should have full novelty (1.0) before decay.
        // After one compute() decay: novelty = 1.0 * (1 - 0.3) = 0.7.
        // But the score was computed BEFORE the decay in compute(), so
        // the novelty component should reflect the pre-decay value of 1.0.
        assert!(
            entry.novelty > 0.5,
            "new signal should have high novelty, got {}",
            entry.novelty
        );
    }

    #[test]
    fn test_novelty_decays_over_ticks() {
        let mut s = default_saliency();
        s.observe(&sig("sig", 1.0));

        let map1 = s.compute();
        let n1 = map1
            .entries
            .iter()
            .find(|e| e.signal_id == "sig")
            .unwrap()
            .novelty;

        // No new observations, just compute again.
        let map2 = s.compute();
        let n2 = map2
            .entries
            .iter()
            .find(|e| e.signal_id == "sig")
            .unwrap()
            .novelty;

        assert!(n2 < n1, "novelty should decay: tick1={}, tick2={}", n1, n2);
    }

    #[test]
    fn test_novelty_refreshed_on_significant_change() {
        let mut s = default_saliency();
        s.observe(&sig("sig", 100.0));

        // Decay novelty.
        for _ in 0..10 {
            s.compute();
        }

        let ss_before = s.signal_stats();
        let novelty_before = ss_before["sig"].novelty;
        assert!(novelty_before < 0.5, "novelty should have decayed");

        // Significant change.
        s.observe(&sig("sig", 200.0));
        let ss_after = s.signal_stats();
        assert!(
            ss_after["sig"].novelty > novelty_before,
            "novelty should be refreshed after significant change"
        );
    }

    #[test]
    fn test_novelty_not_refreshed_on_small_change() {
        let mut s = Saliency::with_config(SaliencyConfig {
            novelty_change_threshold: 0.1, // 10% threshold
            ..default_config()
        });
        s.observe(&sig("sig", 100.0));
        s.compute(); // decay novelty once

        let ss_before = s.signal_stats();
        let novelty_before = ss_before["sig"].novelty;

        // Small change (< 10%).
        s.observe(&sig("sig", 100.5)); // 0.5% change
        let ss_after = s.signal_stats();
        assert!(
            ss_after["sig"].novelty <= novelty_before + 1e-10,
            "novelty should not refresh for small change"
        );
    }

    // -- Magnitude scoring -------------------------------------------------

    #[test]
    fn test_magnitude_scales_with_value() {
        let mut s = Saliency::with_config(SaliencyConfig {
            magnitude_reference: 100.0,
            ..default_config()
        });
        s.observe(&sig("small", 10.0));
        s.observe(&sig("large", 90.0));
        let map = s.compute();

        let small = map.entries.iter().find(|e| e.signal_id == "small").unwrap();
        let large = map.entries.iter().find(|e| e.signal_id == "large").unwrap();

        assert!(
            large.magnitude > small.magnitude,
            "larger signal should have higher magnitude: large={}, small={}",
            large.magnitude,
            small.magnitude
        );
    }

    #[test]
    fn test_magnitude_disabled_when_reference_zero() {
        let mut s = Saliency::with_config(SaliencyConfig {
            magnitude_reference: 0.0,
            ..default_config()
        });
        s.observe(&sig("sig", 1000.0));
        let map = s.compute();
        let entry = map.entries.iter().find(|e| e.signal_id == "sig").unwrap();
        assert!(
            entry.magnitude.abs() < 1e-12,
            "magnitude should be 0 when reference is 0"
        );
    }

    // -- Top-K selection ---------------------------------------------------

    #[test]
    fn test_top_k_selection() {
        let mut s = Saliency::with_config(SaliencyConfig {
            top_k: 2,
            ..default_config()
        });
        s.observe(&sig("a", 10.0));
        s.observe(&sig("b", 50.0));
        s.observe(&sig("c", 30.0));
        let map = s.compute();

        assert!(
            map.entries.len() <= 2,
            "should return at most top_k entries, got {}",
            map.entries.len()
        );
        assert_eq!(map.total_signals, 3);
    }

    #[test]
    fn test_top_k_sorted_descending() {
        let mut s = Saliency::with_config(SaliencyConfig {
            top_k: 10,
            ..default_config()
        });
        s.observe(&sig("a", 10.0));
        s.observe(&sig("b", 50.0));
        s.observe(&sig("c", 30.0));
        let map = s.compute();

        for i in 1..map.entries.len() {
            assert!(
                map.entries[i].score <= map.entries[i - 1].score,
                "entries should be sorted descending by score"
            );
        }
    }

    #[test]
    fn test_top_k_fewer_signals_than_k() {
        let mut s = Saliency::with_config(SaliencyConfig {
            top_k: 10,
            ..default_config()
        });
        s.observe(&sig("a", 1.0));
        s.observe(&sig("b", 2.0));
        let map = s.compute();
        assert_eq!(map.entries.len(), 2, "should return all available signals");
    }

    // -- Composite scoring -------------------------------------------------

    #[test]
    fn test_composite_score_positive() {
        let mut s = default_saliency();
        s.observe(&sig("sig", 50.0));
        let map = s.compute();
        let entry = map.entries.iter().find(|e| e.signal_id == "sig").unwrap();
        assert!(
            entry.score >= 0.0,
            "composite score should be non-negative, got {}",
            entry.score
        );
    }

    #[test]
    fn test_composite_weights_affect_ranking() {
        // With only magnitude weight, the largest signal should rank first.
        let mut s = Saliency::with_config(SaliencyConfig {
            surprise_weight: 0.0,
            novelty_weight: 0.0,
            magnitude_weight: 1.0,
            magnitude_reference: 100.0,
            top_k: 10,
            ..default_config()
        });
        s.observe(&sig("small", 10.0));
        s.observe(&sig("large", 90.0));
        let map = s.compute();
        assert_eq!(map.entries[0].signal_id, "large");
    }

    // -- Aggregate metrics -------------------------------------------------

    #[test]
    fn test_mean_saliency() {
        let mut s = default_saliency();
        s.observe(&sig("a", 10.0));
        s.observe(&sig("b", 20.0));
        let map = s.compute();
        assert!(map.mean_saliency > 0.0, "mean saliency should be positive");
        assert_eq!(map.total_signals, 2);
    }

    #[test]
    fn test_max_saliency() {
        let mut s = default_saliency();
        s.observe(&sig("a", 10.0));
        s.observe(&sig("b", 200.0));
        let map = s.compute();
        assert!(
            map.max_saliency >= map.mean_saliency,
            "max should be >= mean"
        );
    }

    #[test]
    fn test_empty_compute() {
        let mut s = default_saliency();
        let map = s.compute();
        assert!(map.entries.is_empty());
        assert_eq!(map.total_signals, 0);
        assert!(map.mean_saliency.abs() < 1e-12);
        assert!(map.max_saliency.abs() < 1e-12);
    }

    // -- EMA smoothing -----------------------------------------------------

    #[test]
    fn test_ema_initialization() {
        let mut s = default_saliency();
        s.observe(&sig("a", 100.0));
        let map = s.compute();
        assert!(
            (map.smoothed_mean_saliency - map.mean_saliency).abs() < 1e-10,
            "first computation should initialise EMA to raw value"
        );
    }

    #[test]
    fn test_ema_lags() {
        let mut s = Saliency::with_config(SaliencyConfig {
            ema_decay: 0.9,
            ..default_config()
        });

        // First tick: establish baseline.
        s.observe(&sig("a", 100.0));
        let map1 = s.compute();
        let smooth1 = map1.smoothed_mean_saliency;

        // Second tick: big change.
        s.observe(&sig("a", 500.0));
        let map2 = s.compute();

        // Smoothed should lag behind raw.
        if (map2.mean_saliency - smooth1).abs() > 0.01 {
            let gap_smooth = (map2.smoothed_mean_saliency - smooth1).abs();
            let gap_raw = (map2.mean_saliency - smooth1).abs();
            assert!(
                gap_smooth <= gap_raw + 1e-10,
                "EMA should lag: smooth_gap={}, raw_gap={}",
                gap_smooth,
                gap_raw
            );
        }
    }

    #[test]
    fn test_ema_disabled_when_zero() {
        let mut s = Saliency::with_config(SaliencyConfig {
            ema_decay: 0.0,
            ..default_config()
        });
        s.observe(&sig("a", 100.0));
        let map = s.compute();
        assert!(
            (map.smoothed_mean_saliency - map.mean_saliency).abs() < 1e-10,
            "with ema_decay=0, smoothed should equal raw"
        );
    }

    // -- Per-signal statistics ---------------------------------------------

    #[test]
    fn test_signal_stats() {
        let mut s = default_saliency();
        s.observe(&sig("btc", 100.0));
        s.observe(&sig("btc", 105.0));
        s.observe(&sig("btc", 110.0));

        let ss = s.signal_stats();
        assert_eq!(ss["btc"].observations, 3);
        assert!((ss["btc"].last_value - 110.0).abs() < 1e-12);
        assert!(ss["btc"].baseline > 0.0);
    }

    #[test]
    fn test_peak_saliency_tracked() {
        let mut s = default_saliency();

        // Build baseline.
        for _ in 0..10 {
            s.observe(&sig("sig", 100.0));
            s.compute();
        }

        // Spike.
        s.observe(&sig("sig", 500.0));
        s.compute();

        // Return to normal.
        for _ in 0..10 {
            s.observe(&sig("sig", 100.0));
            s.compute();
        }

        let ss = s.signal_stats();
        assert!(
            ss["sig"].peak_saliency > 0.0,
            "peak saliency should have been recorded"
        );
    }

    // -- score_for accessor ------------------------------------------------

    #[test]
    fn test_score_for_existing_signal() {
        let mut s = default_saliency();
        s.observe(&sig("btc", 100.0));
        s.compute();
        let score = s.score_for("btc");
        assert!(score >= 0.0);
    }

    #[test]
    fn test_score_for_missing_signal() {
        let s = default_saliency();
        assert!(s.score_for("nonexistent").abs() < 1e-12);
    }

    // -- Remove signal -----------------------------------------------------

    #[test]
    fn test_remove_signal() {
        let mut s = default_saliency();
        s.observe(&sig("btc", 100.0));
        assert_eq!(s.signal_count(), 1);
        s.remove_signal("btc");
        assert_eq!(s.signal_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_signal() {
        let mut s = default_saliency();
        s.remove_signal("nope"); // should not panic
        assert_eq!(s.signal_count(), 0);
    }

    // -- Windowed statistics -----------------------------------------------

    #[test]
    fn test_windowed_mean() {
        let mut s = default_saliency();
        s.observe(&sig("a", 100.0));
        s.compute();
        s.compute();
        s.compute();
        let mean = s.windowed_mean().unwrap();
        assert!(mean >= 0.0);
    }

    #[test]
    fn test_windowed_mean_empty() {
        let s = default_saliency();
        assert!(s.windowed_mean().is_none());
    }

    #[test]
    fn test_windowed_std_insufficient() {
        let mut s = default_saliency();
        s.observe(&sig("a", 100.0));
        s.compute();
        assert!(s.windowed_std().is_none());
    }

    #[test]
    fn test_windowed_std_constant() {
        let mut s = Saliency::with_config(SaliencyConfig {
            surprise_weight: 0.0,
            novelty_weight: 0.0,
            magnitude_weight: 1.0,
            magnitude_reference: 100.0,
            novelty_decay: 0.5,
            ..default_config()
        });
        // Constant signal → constant magnitude → constant mean saliency.
        for _ in 0..10 {
            s.observe(&sig("a", 50.0));
            s.compute();
        }
        let std = s.windowed_std().unwrap();
        assert!(
            std < 1e-6,
            "constant saliency should have near-zero std, got {}",
            std
        );
    }

    // -- Reset -------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut s = default_saliency();
        s.observe(&sig("a", 100.0));
        s.observe(&sig("b", 200.0));
        s.compute();
        s.compute();

        assert!(s.signal_count() > 0);
        assert!(s.stats().total_computations > 0);

        s.reset();

        assert_eq!(s.signal_count(), 0);
        assert_eq!(s.stats().total_computations, 0);
        assert_eq!(s.stats().total_observations, 0);
        assert!(s.windowed_mean().is_none());
    }

    // -- Stats tracking ----------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut s = default_saliency();
        s.observe(&sig("a", 100.0));
        s.observe(&sig("b", 200.0));
        s.compute();
        s.compute();

        let st = s.stats();
        assert_eq!(st.total_observations, 2);
        assert_eq!(st.total_computations, 2);
        assert_eq!(st.distinct_signals, 2);
    }

    #[test]
    fn test_avg_mean_saliency() {
        let mut s = default_saliency();
        s.observe(&sig("a", 100.0));
        s.compute();
        s.compute();
        assert!(
            s.stats().avg_mean_saliency() >= 0.0,
            "avg mean saliency should be non-negative"
        );
    }

    #[test]
    fn test_stats_defaults() {
        let s = default_saliency();
        assert_eq!(s.stats().total_observations, 0);
        assert_eq!(s.stats().total_computations, 0);
        assert_eq!(s.stats().distinct_signals, 0);
        assert_eq!(s.stats().avg_mean_saliency(), 0.0);
    }

    // -- Window eviction ---------------------------------------------------

    #[test]
    fn test_window_eviction() {
        let mut s = Saliency::with_config(SaliencyConfig {
            window_size: 3,
            ..default_config()
        });
        s.observe(&sig("a", 100.0));
        for _ in 0..10 {
            s.compute();
        }
        assert_eq!(s.history.len(), 3);
    }

    // -- Baseline and variance convergence ---------------------------------

    #[test]
    fn test_baseline_converges() {
        let mut s = Saliency::with_config(SaliencyConfig {
            baseline_ema_decay: 0.5,
            ..default_config()
        });
        for _ in 0..100 {
            s.observe(&sig("sig", 42.0));
        }
        let ss = s.signal_stats();
        assert!(
            (ss["sig"].baseline - 42.0).abs() < 0.01,
            "baseline should converge to constant value, got {}",
            ss["sig"].baseline
        );
    }

    #[test]
    fn test_variance_grows_with_variation() {
        let mut s_const = Saliency::with_config(SaliencyConfig {
            variance_ema_decay: 0.5,
            ..default_config()
        });
        let mut s_vary = Saliency::with_config(SaliencyConfig {
            variance_ema_decay: 0.5,
            ..default_config()
        });

        for i in 0..50 {
            s_const.observe(&sig("sig", 100.0));
            s_const.compute();

            let val = 100.0 + 10.0 * ((i as f64) * 0.5).sin();
            s_vary.observe(&sig("sig", val));
            s_vary.compute();
        }

        let var_const = s_const.signal_stats()["sig"].variance;
        let var_vary = s_vary.signal_stats()["sig"].variance;

        assert!(
            var_vary > var_const,
            "varying signal should have higher variance: const={}, vary={}",
            var_const,
            var_vary
        );
    }

    // -- Negative values ---------------------------------------------------

    #[test]
    fn test_negative_signal_values() {
        let mut s = default_saliency();
        s.observe(&sig("neg", -50.0));
        let map = s.compute();
        let entry = map.entries.iter().find(|e| e.signal_id == "neg").unwrap();
        assert!(
            entry.score >= 0.0,
            "saliency score should be non-negative even for negative signals"
        );
        assert!(entry.magnitude > 0.0, "magnitude should use absolute value");
    }

    // -- Multiple signals interaction --------------------------------------

    #[test]
    fn test_multiple_signals_independent() {
        let mut s = default_saliency();
        s.observe(&sig("a", 100.0));
        s.observe(&sig("b", 200.0));
        s.compute();

        // Updating only "a" should not affect "b"'s baseline.
        let baseline_b_before = s.signal_stats()["b"].baseline;
        s.observe(&sig("a", 999.0));
        let baseline_b_after = s.signal_stats()["b"].baseline;

        assert!(
            (baseline_b_before - baseline_b_after).abs() < 1e-12,
            "signals should be independent"
        );
    }
}
