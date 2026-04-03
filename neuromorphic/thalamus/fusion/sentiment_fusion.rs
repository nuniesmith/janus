//! Sentiment data fusion
//!
//! Part of the Thalamus region
//! Component: fusion
//!
//! Combines sentiment signals from multiple sources (news feeds, social media,
//! on-chain sentiment, analyst ratings, etc.) into a unified sentiment score.
//!
//! ## Features
//!
//! - **Confidence-weighted averaging**: Each source contributes proportionally
//!   to its declared confidence and historical accuracy.
//! - **Staleness decay**: Old readings are exponentially decayed so that stale
//!   sources don't dominate the fused signal.
//! - **Conflict detection**: When sources strongly disagree the fused output
//!   includes a conflict metric so downstream consumers can widen spreads or
//!   reduce position sizes.
//! - **EMA smoothing**: The raw fused score is smoothed with an exponential
//!   moving average to reduce jitter.
//! - **Source tracking**: Per-source accuracy statistics are maintained so that
//!   sources that consistently agree with realised outcomes earn higher weight.

use std::collections::{HashMap, VecDeque};

use crate::common::{Error, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the sentiment fusion engine.
#[derive(Debug, Clone)]
pub struct SentimentFusionConfig {
    /// EMA decay factor for the fused score (0, 1). Closer to 1 → more
    /// smoothing / slower reaction.
    pub ema_decay: f64,
    /// Half-life (in seconds) for the staleness decay applied to source
    /// readings.  A reading that is `staleness_half_life_secs` old will
    /// contribute half its original weight.
    pub staleness_half_life_secs: f64,
    /// Minimum number of active (non-stale) sources required before the
    /// engine produces a fused score.  Below this threshold `fuse()` returns
    /// `None`.
    pub min_sources: usize,
    /// Maximum number of recent fused values to keep for windowed statistics.
    pub window_size: usize,
    /// Threshold (0, 2] on the conflict metric above which
    /// `FusedSentiment::is_conflicted` returns `true`.
    /// Conflict is measured as the weighted standard deviation of source
    /// scores — a value of 0 means perfect agreement, 2 means maximally
    /// opposed.
    pub conflict_threshold: f64,
    /// Maximum age (seconds) before a source reading is considered fully stale
    /// and excluded from fusion entirely.
    pub max_staleness_secs: f64,
    /// EMA decay for tracking per-source accuracy.
    pub accuracy_ema_decay: f64,
}

impl Default for SentimentFusionConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.8,
            staleness_half_life_secs: 300.0, // 5 minutes
            min_sources: 1,
            window_size: 200,
            conflict_threshold: 0.5,
            max_staleness_secs: 3600.0, // 1 hour
            accuracy_ema_decay: 0.95,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// A sentiment reading from a single source.
#[derive(Debug, Clone)]
pub struct SentimentReading {
    /// Identifier for the source (e.g. "cryptopanic", "twitter", "onchain").
    pub source_id: String,
    /// Sentiment score in [-1, 1].  -1 = maximally bearish, +1 = maximally
    /// bullish, 0 = neutral.
    pub score: f64,
    /// Source-declared confidence in [0, 1].
    pub confidence: f64,
    /// Timestamp of the reading (seconds since epoch or an arbitrary
    /// monotonic reference — must be consistent across calls).
    pub timestamp: f64,
}

/// The fused sentiment output.
#[derive(Debug, Clone)]
pub struct FusedSentiment {
    /// Fused sentiment score in [-1, 1].
    pub score: f64,
    /// EMA-smoothed fused score.
    pub smoothed_score: f64,
    /// Aggregate confidence in [0, 1].
    pub confidence: f64,
    /// Conflict metric — weighted standard deviation of source scores.
    pub conflict: f64,
    /// Whether sources are in significant disagreement.
    pub is_conflicted: bool,
    /// Number of non-stale sources that contributed to this fusion.
    pub active_sources: usize,
    /// The timestamp of the most recent reading used.
    pub latest_timestamp: f64,
}

/// Per-source accuracy statistics.
#[derive(Debug, Clone)]
pub struct SourceStats {
    /// Total readings ingested from this source.
    pub readings: usize,
    /// EMA-smoothed accuracy score (higher = more accurate historically).
    pub accuracy: f64,
    /// Most recent reading score.
    pub last_score: f64,
    /// Timestamp of the most recent reading.
    pub last_timestamp: f64,
}

/// Aggregate statistics for the fusion engine.
#[derive(Debug, Clone, Default)]
pub struct SentimentFusionStats {
    /// Total number of readings ingested across all sources.
    pub total_readings: usize,
    /// Total number of fused outputs produced.
    pub total_fusions: usize,
    /// Number of fusions that were flagged as conflicted.
    pub conflict_count: usize,
    /// Number of fusions skipped because `min_sources` was not met.
    pub insufficient_source_count: usize,
    /// Number of distinct sources seen.
    pub distinct_sources: usize,
    /// Number of outcome observations recorded.
    pub outcome_observations: usize,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Internal record for a source's most recent reading plus derived weight.
#[derive(Debug, Clone)]
struct SourceRecord {
    score: f64,
    confidence: f64,
    timestamp: f64,
    accuracy: f64,
    total_readings: usize,
}

// ---------------------------------------------------------------------------
// SentimentFusion
// ---------------------------------------------------------------------------

/// Multi-source sentiment fusion engine.
///
/// Call [`ingest`] to feed readings from individual sources, then call
/// [`fuse`] to obtain the aggregated sentiment at a given point in time.
pub struct SentimentFusion {
    config: SentimentFusionConfig,
    /// Most recent record per source.
    sources: HashMap<String, SourceRecord>,
    /// EMA state for the smoothed fused score.
    ema_score: f64,
    ema_initialized: bool,
    /// Windowed history of fused scores for stats.
    history: VecDeque<f64>,
    /// Running statistics.
    stats: SentimentFusionStats,
}

impl Default for SentimentFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentFusion {
    /// Create a new instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(SentimentFusionConfig::default())
    }

    /// Create a new instance with the given configuration.
    pub fn with_config(config: SentimentFusionConfig) -> Self {
        Self {
            sources: HashMap::new(),
            ema_score: 0.0,
            ema_initialized: false,
            history: VecDeque::with_capacity(config.window_size),
            stats: SentimentFusionStats::default(),
            config,
        }
    }

    /// Validate configuration parameters.
    pub fn process(&self) -> Result<()> {
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.staleness_half_life_secs <= 0.0 {
            return Err(Error::InvalidInput(
                "staleness_half_life_secs must be > 0".into(),
            ));
        }
        if self.config.max_staleness_secs <= 0.0 {
            return Err(Error::InvalidInput("max_staleness_secs must be > 0".into()));
        }
        if self.config.conflict_threshold <= 0.0 {
            return Err(Error::InvalidInput("conflict_threshold must be > 0".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.accuracy_ema_decay <= 0.0 || self.config.accuracy_ema_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "accuracy_ema_decay must be in (0, 1)".into(),
            ));
        }
        Ok(())
    }

    // -- Ingestion ---------------------------------------------------------

    /// Ingest a single sentiment reading.
    ///
    /// The reading is stored per-source; only the most recent reading per
    /// source is kept.
    pub fn ingest(&mut self, reading: &SentimentReading) -> Result<()> {
        if reading.score < -1.0 || reading.score > 1.0 {
            return Err(Error::InvalidInput(
                "sentiment score must be in [-1, 1]".into(),
            ));
        }
        if reading.confidence < 0.0 || reading.confidence > 1.0 {
            return Err(Error::InvalidInput("confidence must be in [0, 1]".into()));
        }

        self.stats.total_readings += 1;

        let accuracy = self
            .sources
            .get(&reading.source_id)
            .map(|r| r.accuracy)
            .unwrap_or(0.5); // default prior

        if !self.sources.contains_key(&reading.source_id) {
            self.stats.distinct_sources += 1;
        }

        self.sources.insert(
            reading.source_id.clone(),
            SourceRecord {
                score: reading.score,
                confidence: reading.confidence,
                timestamp: reading.timestamp,
                accuracy,
                total_readings: self
                    .sources
                    .get(&reading.source_id)
                    .map(|r| r.total_readings + 1)
                    .unwrap_or(1),
            },
        );

        Ok(())
    }

    // -- Fusion ------------------------------------------------------------

    /// Produce the fused sentiment score at time `now`.
    ///
    /// Returns `None` if fewer than `min_sources` non-stale sources are
    /// available.
    pub fn fuse(&mut self, now: f64) -> Option<FusedSentiment> {
        let decay_rate = (0.5_f64).ln() / self.config.staleness_half_life_secs;

        // Collect active (non-stale) sources with their weights.
        let mut entries: Vec<(f64, f64)> = Vec::new(); // (weight, score)
        let mut latest_ts: f64 = f64::NEG_INFINITY;

        for record in self.sources.values() {
            let age = (now - record.timestamp).max(0.0);
            if age > self.config.max_staleness_secs {
                continue;
            }
            let staleness_weight = (decay_rate * age).exp(); // in (0, 1]
            let weight = record.confidence * staleness_weight * record.accuracy;
            if weight <= 0.0 {
                continue;
            }
            entries.push((weight, record.score));
            if record.timestamp > latest_ts {
                latest_ts = record.timestamp;
            }
        }

        if entries.len() < self.config.min_sources {
            self.stats.insufficient_source_count += 1;
            return None;
        }

        // Weighted mean
        let total_weight: f64 = entries.iter().map(|(w, _)| w).sum();
        if total_weight <= 0.0 {
            self.stats.insufficient_source_count += 1;
            return None;
        }

        let raw_score: f64 = entries.iter().map(|(w, s)| w * s).sum::<f64>() / total_weight;

        // Weighted standard deviation (conflict metric)
        let variance: f64 = entries
            .iter()
            .map(|(w, s)| w * (s - raw_score).powi(2))
            .sum::<f64>()
            / total_weight;
        let conflict = variance.sqrt();

        // Confidence: normalised weight capped at 1
        let max_possible_weight = entries.len() as f64; // if every source had weight 1
        let confidence = (total_weight / max_possible_weight).min(1.0);

        // EMA smoothing
        let smoothed = if self.ema_initialized {
            let s =
                self.config.ema_decay * self.ema_score + (1.0 - self.config.ema_decay) * raw_score;
            self.ema_score = s;
            s
        } else {
            self.ema_score = raw_score;
            self.ema_initialized = true;
            raw_score
        };

        // Window history
        self.history.push_back(raw_score);
        while self.history.len() > self.config.window_size {
            self.history.pop_front();
        }

        let is_conflicted = conflict > self.config.conflict_threshold;
        self.stats.total_fusions += 1;
        if is_conflicted {
            self.stats.conflict_count += 1;
        }

        Some(FusedSentiment {
            score: raw_score,
            smoothed_score: smoothed,
            confidence,
            conflict,
            is_conflicted,
            active_sources: entries.len(),
            latest_timestamp: latest_ts,
        })
    }

    // -- Outcome feedback --------------------------------------------------

    /// Record a realised outcome so that source accuracy can be updated.
    ///
    /// `outcome` should be in [-1, 1] representing the realised directional
    /// move.  `at_timestamp` is the time at which the outcome was observed.
    /// Each source's accuracy is updated based on how close its most recent
    /// reading (before `at_timestamp`) was to the outcome.
    pub fn record_outcome(&mut self, outcome: f64, at_timestamp: f64) {
        let alpha = self.config.accuracy_ema_decay;
        self.stats.outcome_observations += 1;

        for record in self.sources.values_mut() {
            if record.timestamp > at_timestamp {
                continue; // reading was after the outcome — skip
            }
            // Accuracy: 1 - |predicted - actual| / 2  → maps [0, 2] error to [0, 1] accuracy
            let error = (record.score - outcome).abs();
            let instant_accuracy = 1.0 - (error / 2.0);
            record.accuracy = alpha * record.accuracy + (1.0 - alpha) * instant_accuracy;
        }
    }

    // -- Accessors ---------------------------------------------------------

    /// Get per-source statistics.
    pub fn source_stats(&self) -> HashMap<String, SourceStats> {
        self.sources
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    SourceStats {
                        readings: r.total_readings,
                        accuracy: r.accuracy,
                        last_score: r.score,
                        last_timestamp: r.timestamp,
                    },
                )
            })
            .collect()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> &SentimentFusionStats {
        &self.stats
    }

    /// Current number of distinct sources that have been ingested.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Windowed mean of recent fused scores.
    pub fn windowed_mean(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().sum();
        Some(sum / self.history.len() as f64)
    }

    /// Windowed standard deviation of recent fused scores.
    pub fn windowed_std(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let mean = self.windowed_mean().unwrap();
        let var: f64 = self.history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Reset all state (sources, EMA, history, stats).
    pub fn reset(&mut self) {
        self.sources.clear();
        self.ema_score = 0.0;
        self.ema_initialized = false;
        self.history.clear();
        self.stats = SentimentFusionStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(source: &str, score: f64, confidence: f64, ts: f64) -> SentimentReading {
        SentimentReading {
            source_id: source.to_string(),
            score,
            confidence,
            timestamp: ts,
        }
    }

    fn default_fusion() -> SentimentFusion {
        SentimentFusion::with_config(SentimentFusionConfig {
            ema_decay: 0.5,
            staleness_half_life_secs: 60.0,
            min_sources: 1,
            window_size: 100,
            conflict_threshold: 0.4,
            max_staleness_secs: 300.0,
            accuracy_ema_decay: 0.9,
        })
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = SentimentFusion::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_process_invalid_ema_decay() {
        let sf = SentimentFusion::with_config(SentimentFusionConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(sf.process().is_err());
    }

    #[test]
    fn test_process_invalid_staleness_half_life() {
        let sf = SentimentFusion::with_config(SentimentFusionConfig {
            staleness_half_life_secs: -1.0,
            ..Default::default()
        });
        assert!(sf.process().is_err());
    }

    #[test]
    fn test_process_invalid_window_size() {
        let sf = SentimentFusion::with_config(SentimentFusionConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(sf.process().is_err());
    }

    #[test]
    fn test_process_invalid_conflict_threshold() {
        let sf = SentimentFusion::with_config(SentimentFusionConfig {
            conflict_threshold: 0.0,
            ..Default::default()
        });
        assert!(sf.process().is_err());
    }

    #[test]
    fn test_process_invalid_accuracy_ema() {
        let sf = SentimentFusion::with_config(SentimentFusionConfig {
            accuracy_ema_decay: 1.0,
            ..Default::default()
        });
        assert!(sf.process().is_err());
    }

    // -- Ingestion ---------------------------------------------------------

    #[test]
    fn test_ingest_valid() {
        let mut sf = default_fusion();
        assert!(sf.ingest(&reading("a", 0.5, 0.9, 100.0)).is_ok());
        assert_eq!(sf.source_count(), 1);
        assert_eq!(sf.stats().total_readings, 1);
    }

    #[test]
    fn test_ingest_invalid_score_too_high() {
        let mut sf = default_fusion();
        assert!(sf.ingest(&reading("a", 1.5, 0.9, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_invalid_score_too_low() {
        let mut sf = default_fusion();
        assert!(sf.ingest(&reading("a", -1.1, 0.9, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_invalid_confidence() {
        let mut sf = default_fusion();
        assert!(sf.ingest(&reading("a", 0.5, 1.1, 100.0)).is_err());
        assert!(sf.ingest(&reading("a", 0.5, -0.1, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_replaces_old_reading() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.2, 0.9, 100.0)).unwrap();
        sf.ingest(&reading("a", 0.8, 0.9, 110.0)).unwrap();
        assert_eq!(sf.source_count(), 1);
        assert_eq!(sf.stats().total_readings, 2);

        let ss = sf.source_stats();
        assert!((ss["a"].last_score - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_ingest_multiple_sources() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.2, 0.9, 100.0)).unwrap();
        sf.ingest(&reading("b", 0.4, 0.8, 100.0)).unwrap();
        sf.ingest(&reading("c", -0.1, 0.7, 100.0)).unwrap();
        assert_eq!(sf.source_count(), 3);
        assert_eq!(sf.stats().distinct_sources, 3);
    }

    // -- Fusion basics -----------------------------------------------------

    #[test]
    fn test_fuse_single_source() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.6, 1.0, 100.0)).unwrap();
        let fused = sf.fuse(100.0).unwrap();
        assert!(
            (fused.score - 0.6).abs() < 0.01,
            "single source should yield its own score, got {}",
            fused.score
        );
        assert_eq!(fused.active_sources, 1);
        assert!(
            (fused.conflict).abs() < 1e-12,
            "single source → zero conflict"
        );
    }

    #[test]
    fn test_fuse_two_agreeing_sources() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.5, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("b", 0.5, 1.0, 100.0)).unwrap();
        let fused = sf.fuse(100.0).unwrap();
        assert!((fused.score - 0.5).abs() < 1e-12);
        assert!(fused.conflict < 1e-12);
        assert!(!fused.is_conflicted);
    }

    #[test]
    fn test_fuse_weighted_by_confidence() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            ema_decay: 0.5,
            staleness_half_life_secs: 1e6, // effectively no decay
            min_sources: 1,
            window_size: 100,
            conflict_threshold: 2.0,
            max_staleness_secs: 1e6,
            accuracy_ema_decay: 0.9,
        });
        // Source a: score=1.0, confidence=1.0  → weight ~ 0.5 (accuracy default)
        // Source b: score=0.0, confidence=0.5  → weight ~ 0.25
        sf.ingest(&reading("a", 1.0, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("b", 0.0, 0.5, 100.0)).unwrap();
        let fused = sf.fuse(100.0).unwrap();
        // weighted mean: (0.5*1.0 + 0.25*0.0) / (0.5+0.25) = 0.5/0.75 ≈ 0.667
        assert!(
            (fused.score - 2.0 / 3.0).abs() < 0.01,
            "expected ~0.667, got {}",
            fused.score
        );
    }

    #[test]
    fn test_fuse_insufficient_sources() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            min_sources: 3,
            ..SentimentFusionConfig::default()
        });
        sf.ingest(&reading("a", 0.5, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("b", 0.5, 1.0, 100.0)).unwrap();
        assert!(sf.fuse(100.0).is_none());
        assert_eq!(sf.stats().insufficient_source_count, 1);
    }

    // -- Staleness ---------------------------------------------------------

    #[test]
    fn test_staleness_decay_reduces_weight() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            staleness_half_life_secs: 60.0,
            max_staleness_secs: 600.0,
            min_sources: 1,
            ema_decay: 0.5,
            window_size: 100,
            conflict_threshold: 2.0,
            accuracy_ema_decay: 0.9,
        });
        // Source a: fresh, score = 1.0
        // Source b: 60s old (one half-life), score = 0.0
        sf.ingest(&reading("a", 1.0, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("b", 0.0, 1.0, 40.0)).unwrap(); // 60s before now=100

        let fused = sf.fuse(100.0).unwrap();
        // b's weight should be ~half of a's → score should be closer to 1.0 than 0.5
        assert!(
            fused.score > 0.55,
            "stale source should contribute less, got {}",
            fused.score
        );
    }

    #[test]
    fn test_fully_stale_source_excluded() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            max_staleness_secs: 60.0,
            min_sources: 1,
            staleness_half_life_secs: 10.0,
            ema_decay: 0.5,
            window_size: 100,
            conflict_threshold: 2.0,
            accuracy_ema_decay: 0.9,
        });
        sf.ingest(&reading("a", 0.5, 1.0, 0.0)).unwrap();
        // At t=100, the reading is 100s old which exceeds max_staleness_secs=60
        let fused = sf.fuse(100.0);
        assert!(fused.is_none(), "fully stale reading should be excluded");
    }

    // -- Conflict detection ------------------------------------------------

    #[test]
    fn test_conflict_detection() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            conflict_threshold: 0.3,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            min_sources: 1,
            ema_decay: 0.5,
            window_size: 100,
            accuracy_ema_decay: 0.9,
        });
        // Strongly opposing sources
        sf.ingest(&reading("bull", 0.9, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("bear", -0.9, 1.0, 100.0)).unwrap();
        let fused = sf.fuse(100.0).unwrap();
        assert!(
            fused.is_conflicted,
            "opposing sources should trigger conflict, conflict={}",
            fused.conflict
        );
        assert!(fused.conflict > 0.3);
    }

    #[test]
    fn test_no_conflict_when_agreeing() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            conflict_threshold: 0.3,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            min_sources: 1,
            ema_decay: 0.5,
            window_size: 100,
            accuracy_ema_decay: 0.9,
        });
        sf.ingest(&reading("a", 0.5, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("b", 0.52, 1.0, 100.0)).unwrap();
        let fused = sf.fuse(100.0).unwrap();
        assert!(!fused.is_conflicted);
    }

    // -- EMA smoothing -----------------------------------------------------

    #[test]
    fn test_ema_smoothing() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            ema_decay: 0.8,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            min_sources: 1,
            window_size: 100,
            conflict_threshold: 2.0,
            accuracy_ema_decay: 0.9,
        });

        // First fusion: EMA initialises to raw score
        sf.ingest(&reading("a", 1.0, 1.0, 100.0)).unwrap();
        let f1 = sf.fuse(100.0).unwrap();
        assert!((f1.smoothed_score - 1.0).abs() < 1e-12);

        // Second fusion with very different score: smoothed should lag
        sf.ingest(&reading("a", -1.0, 1.0, 110.0)).unwrap();
        let f2 = sf.fuse(110.0).unwrap();
        // EMA: 0.8 * 1.0 + 0.2 * (-1.0) = 0.6
        assert!(
            (f2.smoothed_score - 0.6).abs() < 0.01,
            "smoothed should be ~0.6, got {}",
            f2.smoothed_score
        );
        assert!(
            (f2.score - (-1.0)).abs() < 1e-12,
            "raw score should be -1.0"
        );
    }

    // -- Outcome feedback --------------------------------------------------

    #[test]
    fn test_record_outcome_updates_accuracy() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            accuracy_ema_decay: 0.5,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            min_sources: 1,
            ema_decay: 0.5,
            window_size: 100,
            conflict_threshold: 2.0,
        });
        sf.ingest(&reading("good", 0.8, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("bad", -0.5, 1.0, 100.0)).unwrap();

        // Outcome was bullish (0.9)
        sf.record_outcome(0.9, 110.0);

        let ss = sf.source_stats();
        // "good" predicted 0.8, outcome 0.9 → error=0.1, instant_acc = 1 - 0.05 = 0.95
        // "bad" predicted -0.5, outcome 0.9 → error=1.4, instant_acc = 1 - 0.7 = 0.3
        // With alpha=0.5: good_acc = 0.5*0.5 + 0.5*0.95 = 0.725
        //                 bad_acc  = 0.5*0.5 + 0.5*0.3  = 0.4
        assert!(
            ss["good"].accuracy > ss["bad"].accuracy,
            "accurate source should have higher accuracy: good={}, bad={}",
            ss["good"].accuracy,
            ss["bad"].accuracy
        );
        assert_eq!(sf.stats().outcome_observations, 1);
    }

    #[test]
    fn test_record_outcome_ignores_future_readings() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.5, 1.0, 200.0)).unwrap();
        let before = sf.source_stats()["a"].accuracy;
        // Outcome at t=100, but reading was at t=200 (after outcome) → should skip
        sf.record_outcome(0.9, 100.0);
        let after = sf.source_stats()["a"].accuracy;
        assert!(
            (before - after).abs() < 1e-12,
            "accuracy should not change for future readings"
        );
    }

    // -- Windowed statistics -----------------------------------------------

    #[test]
    fn test_windowed_mean() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.2, 1.0, 100.0)).unwrap();
        sf.fuse(100.0).unwrap();
        sf.ingest(&reading("a", 0.8, 1.0, 110.0)).unwrap();
        sf.fuse(110.0).unwrap();
        let mean = sf.windowed_mean().unwrap();
        assert!(
            (mean - 0.5).abs() < 0.01,
            "mean of 0.2 and 0.8 should be ~0.5, got {}",
            mean
        );
    }

    #[test]
    fn test_windowed_std() {
        let mut sf = default_fusion();
        // Ingest identical scores → std should be 0
        for i in 0..5 {
            sf.ingest(&reading("a", 0.5, 1.0, 100.0 + i as f64))
                .unwrap();
            sf.fuse(100.0 + i as f64);
        }
        let std = sf.windowed_std().unwrap();
        assert!(std < 1e-12, "constant scores should have zero std");
    }

    #[test]
    fn test_windowed_mean_empty() {
        let sf = default_fusion();
        assert!(sf.windowed_mean().is_none());
    }

    #[test]
    fn test_windowed_std_insufficient() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.5, 1.0, 100.0)).unwrap();
        sf.fuse(100.0);
        assert!(sf.windowed_std().is_none()); // need >= 2
    }

    // -- Reset -------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut sf = default_fusion();
        for i in 0..10 {
            sf.ingest(&reading("a", 0.1 * i as f64, 1.0, 100.0 + i as f64))
                .unwrap();
            sf.fuse(100.0 + i as f64);
        }
        assert!(sf.source_count() > 0);
        assert!(sf.stats().total_fusions > 0);

        sf.reset();

        assert_eq!(sf.source_count(), 0);
        assert_eq!(sf.stats().total_fusions, 0);
        assert_eq!(sf.stats().total_readings, 0);
        assert!(sf.windowed_mean().is_none());
    }

    // -- Stats tracking ----------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.5, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("b", 0.6, 1.0, 100.0)).unwrap();
        sf.fuse(100.0);
        sf.fuse(100.0);

        let s = sf.stats();
        assert_eq!(s.total_readings, 2);
        assert_eq!(s.total_fusions, 2);
        assert_eq!(s.distinct_sources, 2);
    }

    #[test]
    fn test_conflict_count_in_stats() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            conflict_threshold: 0.1,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            min_sources: 1,
            ema_decay: 0.5,
            window_size: 100,
            accuracy_ema_decay: 0.9,
        });
        sf.ingest(&reading("a", 1.0, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("b", -1.0, 1.0, 100.0)).unwrap();
        sf.fuse(100.0);
        assert_eq!(sf.stats().conflict_count, 1);
    }

    // -- Edge cases --------------------------------------------------------

    #[test]
    fn test_boundary_scores() {
        let mut sf = default_fusion();
        assert!(sf.ingest(&reading("a", -1.0, 0.0, 100.0)).is_ok());
        assert!(sf.ingest(&reading("b", 1.0, 1.0, 100.0)).is_ok());
    }

    #[test]
    fn test_zero_confidence_source_low_weight() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            min_sources: 1,
            ema_decay: 0.5,
            window_size: 100,
            conflict_threshold: 2.0,
            accuracy_ema_decay: 0.9,
        });
        sf.ingest(&reading("high", 0.8, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("zero", -0.8, 0.0, 100.0)).unwrap();
        let fused = sf.fuse(100.0).unwrap();
        // zero-confidence source should contribute nothing
        assert!(
            (fused.score - 0.8).abs() < 0.01,
            "zero-confidence source should not affect fused score, got {}",
            fused.score
        );
    }

    #[test]
    fn test_fuse_latest_timestamp() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.5, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("b", 0.5, 1.0, 200.0)).unwrap();
        let fused = sf.fuse(200.0).unwrap();
        assert!((fused.latest_timestamp - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_source_stats_readings_count() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.1, 1.0, 100.0)).unwrap();
        sf.ingest(&reading("a", 0.2, 1.0, 110.0)).unwrap();
        sf.ingest(&reading("a", 0.3, 1.0, 120.0)).unwrap();
        let ss = sf.source_stats();
        assert_eq!(ss["a"].readings, 3);
    }

    #[test]
    fn test_accuracy_improves_with_correct_predictions() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            accuracy_ema_decay: 0.5,
            ..SentimentFusionConfig::default()
        });
        sf.ingest(&reading("a", 0.9, 1.0, 100.0)).unwrap();

        // Multiple outcomes that confirm the source's prediction
        for i in 1..=5 {
            sf.record_outcome(0.9, 100.0 + i as f64);
        }

        let ss = sf.source_stats();
        // After several perfect predictions, accuracy should be very high
        assert!(
            ss["a"].accuracy > 0.9,
            "accuracy should be very high after correct predictions, got {}",
            ss["a"].accuracy
        );
    }

    #[test]
    fn test_window_eviction() {
        let mut sf = SentimentFusion::with_config(SentimentFusionConfig {
            window_size: 3,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            min_sources: 1,
            ema_decay: 0.5,
            conflict_threshold: 2.0,
            accuracy_ema_decay: 0.9,
        });
        for i in 0..10 {
            sf.ingest(&reading("a", 0.1 * i as f64, 1.0, 100.0 + i as f64))
                .unwrap();
            sf.fuse(100.0 + i as f64);
        }
        // Only the last 3 fused scores should be in the window
        assert_eq!(sf.history.len(), 3);
    }

    #[test]
    fn test_neutral_sentiment() {
        let mut sf = default_fusion();
        sf.ingest(&reading("a", 0.0, 1.0, 100.0)).unwrap();
        let fused = sf.fuse(100.0).unwrap();
        assert!(fused.score.abs() < 1e-12);
    }
}
