//! Price data fusion
//!
//! Part of the Thalamus region
//! Component: fusion
//!
//! Aggregates price quotes from multiple sources (exchanges, OTC desks,
//! index providers, internal models) into a unified fair-value estimate.
//!
//! ## Features
//!
//! - **Multi-source aggregation**: Combines price observations from an
//!   arbitrary number of named sources using confidence-weighted averaging.
//! - **Outlier detection**: Identifies and down-weights price quotes that
//!   deviate significantly from the consensus, using a configurable
//!   z-score threshold relative to the running median.
//! - **Fair value estimation**: Produces a robust fair-value price by
//!   combining outlier-filtered, confidence-weighted inputs.
//! - **Volatility tracking**: Maintains a realised-volatility estimate
//!   from successive fair-value changes using an EMA of squared returns.
//! - **Confidence scoring**: Outputs an aggregate confidence score that
//!   reflects source agreement, staleness, and number of active sources.
//! - **EMA smoothing**: The raw fused price is smoothed with an
//!   exponential moving average to reduce tick-level noise.
//! - **Per-source statistics**: Tracks each source's deviation from
//!   consensus, observation count, and outlier rate.

use std::collections::{HashMap, VecDeque};

use crate::common::{Error, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the price fusion engine.
#[derive(Debug, Clone)]
pub struct PriceFusionConfig {
    /// EMA decay factor for the smoothed fair-value price (0, 1).
    /// Closer to 1 → more smoothing / slower reaction.
    pub ema_decay: f64,
    /// EMA decay factor for the realised-volatility estimator (0, 1).
    pub volatility_ema_decay: f64,
    /// Z-score threshold above which a source's price is considered an
    /// outlier and down-weighted.  For example, 2.5 means prices more than
    /// 2.5 standard deviations from the consensus median are penalised.
    pub outlier_z_threshold: f64,
    /// Weight multiplier applied to outlier sources (0, 1).
    /// 0 → fully exclude outliers, 1 → treat them normally.
    pub outlier_weight: f64,
    /// Maximum age (seconds) before a source's quote is considered stale
    /// and excluded from fusion.
    pub max_staleness_secs: f64,
    /// Half-life (in seconds) for the staleness decay applied to source
    /// weights.  A quote `staleness_half_life_secs` old contributes half
    /// its original weight.
    pub staleness_half_life_secs: f64,
    /// Minimum number of active (non-stale) sources required before the
    /// engine produces a fused price.  Below this threshold `fuse()` returns
    /// `None`.
    pub min_sources: usize,
    /// Maximum number of recent fused prices to retain for windowed stats.
    pub window_size: usize,
    /// Minimum number of observations before volatility is considered
    /// initialised.
    pub volatility_min_samples: usize,
}

impl Default for PriceFusionConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.85,
            volatility_ema_decay: 0.94,
            outlier_z_threshold: 2.5,
            outlier_weight: 0.1,
            max_staleness_secs: 30.0,
            staleness_half_life_secs: 10.0,
            min_sources: 1,
            window_size: 500,
            volatility_min_samples: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// A price observation from a single source.
#[derive(Debug, Clone)]
pub struct PriceObservation {
    /// Source identifier (e.g. "binance_spot", "coinbase", "index_xyz").
    pub source_id: String,
    /// Observed price.
    pub price: f64,
    /// Source-declared confidence in [0, 1].
    pub confidence: f64,
    /// Timestamp of the observation (seconds since epoch or monotonic ref).
    pub timestamp: f64,
}

/// The fused price output.
#[derive(Debug, Clone)]
pub struct FusedPrice {
    /// Fair-value price estimate (confidence-weighted, outlier-filtered).
    pub fair_value: f64,
    /// EMA-smoothed fair-value price.
    pub smoothed_price: f64,
    /// Aggregate confidence in [0, 1].
    pub confidence: f64,
    /// Realised volatility estimate (annualised, if inputs are per-second).
    /// `None` if not yet initialised (fewer than `volatility_min_samples`).
    pub volatility: Option<f64>,
    /// Number of active (non-stale) sources that contributed.
    pub active_sources: usize,
    /// Number of sources flagged as outliers in this fusion.
    pub outlier_count: usize,
    /// Maximum absolute deviation of any source from the fair value (price units).
    pub max_deviation: f64,
    /// Mean absolute deviation of active sources from the fair value.
    pub mean_deviation: f64,
    /// Timestamp of the most recent observation used.
    pub latest_timestamp: f64,
}

/// Per-source statistics.
#[derive(Debug, Clone)]
pub struct SourcePriceStats {
    /// Total observations ingested from this source.
    pub observations: usize,
    /// Number of times this source was flagged as an outlier.
    pub outlier_count: usize,
    /// EMA-smoothed absolute deviation from consensus.
    pub ema_deviation: f64,
    /// Most recent price from this source.
    pub last_price: f64,
    /// Most recent timestamp.
    pub last_timestamp: f64,
    /// Most recent confidence.
    pub last_confidence: f64,
}

/// Aggregate statistics for the fusion engine.
#[derive(Debug, Clone, Default)]
pub struct PriceFusionStats {
    /// Total observations ingested across all sources.
    pub total_observations: usize,
    /// Total fused outputs produced.
    pub total_fusions: usize,
    /// Number of fusions that included at least one outlier.
    pub fusions_with_outliers: usize,
    /// Total individual source-observations flagged as outliers.
    pub total_outlier_flags: usize,
    /// Number of fusions skipped because `min_sources` was not met.
    pub insufficient_source_count: usize,
    /// Number of distinct sources seen.
    pub distinct_sources: usize,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SourceRecord {
    price: f64,
    confidence: f64,
    timestamp: f64,
    observations: usize,
    outlier_count: usize,
    ema_deviation: f64,
    deviation_initialized: bool,
}

// ---------------------------------------------------------------------------
// PriceFusion
// ---------------------------------------------------------------------------

/// Multi-source price fusion engine.
///
/// Call [`ingest`] to feed price observations from individual sources, then
/// call [`fuse`] to obtain the aggregated fair-value estimate at a given
/// point in time.
pub struct PriceFusion {
    config: PriceFusionConfig,
    /// Most recent record per source.
    sources: HashMap<String, SourceRecord>,
    /// EMA state for smoothed fair-value price.
    ema_price: f64,
    ema_price_initialized: bool,
    /// Volatility tracking state.
    prev_fair_value: Option<f64>,
    ema_variance: f64,
    ema_variance_initialized: bool,
    volatility_samples: usize,
    /// Windowed history of fair-value prices.
    history: VecDeque<f64>,
    /// Running statistics.
    stats: PriceFusionStats,
}

impl Default for PriceFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl PriceFusion {
    /// Create a new instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(PriceFusionConfig::default())
    }

    /// Create a new instance with the given configuration.
    pub fn with_config(config: PriceFusionConfig) -> Self {
        Self {
            sources: HashMap::new(),
            ema_price: 0.0,
            ema_price_initialized: false,
            prev_fair_value: None,
            ema_variance: 0.0,
            ema_variance_initialized: false,
            volatility_samples: 0,
            history: VecDeque::with_capacity(config.window_size),
            stats: PriceFusionStats::default(),
            config,
        }
    }

    /// Validate configuration parameters.
    pub fn process(&self) -> Result<()> {
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.volatility_ema_decay <= 0.0 || self.config.volatility_ema_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "volatility_ema_decay must be in (0, 1)".into(),
            ));
        }
        if self.config.outlier_z_threshold <= 0.0 {
            return Err(Error::InvalidInput(
                "outlier_z_threshold must be > 0".into(),
            ));
        }
        if self.config.outlier_weight < 0.0 || self.config.outlier_weight > 1.0 {
            return Err(Error::InvalidInput(
                "outlier_weight must be in [0, 1]".into(),
            ));
        }
        if self.config.max_staleness_secs <= 0.0 {
            return Err(Error::InvalidInput("max_staleness_secs must be > 0".into()));
        }
        if self.config.staleness_half_life_secs <= 0.0 {
            return Err(Error::InvalidInput(
                "staleness_half_life_secs must be > 0".into(),
            ));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        Ok(())
    }

    // -- Ingestion ---------------------------------------------------------

    /// Ingest a single price observation from a source.
    pub fn ingest(&mut self, obs: &PriceObservation) -> Result<()> {
        if obs.price <= 0.0 {
            return Err(Error::InvalidInput("price must be > 0".into()));
        }
        if obs.confidence < 0.0 || obs.confidence > 1.0 {
            return Err(Error::InvalidInput("confidence must be in [0, 1]".into()));
        }

        self.stats.total_observations += 1;

        if !self.sources.contains_key(&obs.source_id) {
            self.stats.distinct_sources += 1;
        }

        let prev = self.sources.get(&obs.source_id);
        let prev_obs = prev.map(|r| r.observations).unwrap_or(0);
        let prev_outlier = prev.map(|r| r.outlier_count).unwrap_or(0);
        let prev_dev = prev.map(|r| r.ema_deviation).unwrap_or(0.0);
        let dev_init = prev.map(|r| r.deviation_initialized).unwrap_or(false);

        self.sources.insert(
            obs.source_id.clone(),
            SourceRecord {
                price: obs.price,
                confidence: obs.confidence,
                timestamp: obs.timestamp,
                observations: prev_obs + 1,
                outlier_count: prev_outlier,
                ema_deviation: prev_dev,
                deviation_initialized: dev_init,
            },
        );

        Ok(())
    }

    // -- Fusion ------------------------------------------------------------

    /// Produce the fused fair-value price at time `now`.
    ///
    /// Returns `None` if fewer than `min_sources` non-stale sources are
    /// available.
    pub fn fuse(&mut self, now: f64) -> Option<FusedPrice> {
        let decay_rate = (0.5_f64).ln() / self.config.staleness_half_life_secs;

        // Step 1: Collect active (non-stale) sources with raw weights.
        struct ActiveSource {
            id: String,
            price: f64,
            weight: f64,
        }

        let mut active: Vec<ActiveSource> = Vec::new();
        let mut latest_ts = f64::NEG_INFINITY;

        for (id, record) in &self.sources {
            let age = (now - record.timestamp).max(0.0);
            if age > self.config.max_staleness_secs {
                continue;
            }
            let staleness_weight = (decay_rate * age).exp();
            let weight = record.confidence * staleness_weight;
            if weight <= 0.0 {
                continue;
            }
            active.push(ActiveSource {
                id: id.clone(),
                price: record.price,
                weight,
            });
            if record.timestamp > latest_ts {
                latest_ts = record.timestamp;
            }
        }

        if active.len() < self.config.min_sources {
            self.stats.insufficient_source_count += 1;
            return None;
        }

        // Step 2: Compute median price for outlier detection.
        let mut prices: Vec<f64> = active.iter().map(|s| s.price).collect();
        prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if prices.len() % 2 == 0 && prices.len() >= 2 {
            (prices[prices.len() / 2 - 1] + prices[prices.len() / 2]) / 2.0
        } else {
            prices[prices.len() / 2]
        };

        // Compute MAD (median absolute deviation) for robust std estimate.
        let mut abs_devs: Vec<f64> = prices.iter().map(|p| (p - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if abs_devs.len() % 2 == 0 && abs_devs.len() >= 2 {
            (abs_devs[abs_devs.len() / 2 - 1] + abs_devs[abs_devs.len() / 2]) / 2.0
        } else {
            abs_devs[abs_devs.len() / 2]
        };
        // Convert MAD to std-equivalent (for normal distribution, σ ≈ 1.4826 * MAD)
        let mut robust_std = mad * 1.4826;

        // Fallback: when MAD is zero (majority of sources agree exactly) but
        // some sources deviate, use the mean absolute deviation instead so
        // that genuine outliers are still detected.
        if robust_std <= 0.0 && abs_devs.len() >= 2 {
            let mean_abs_dev = abs_devs.iter().sum::<f64>() / abs_devs.len() as f64;
            if mean_abs_dev > 0.0 {
                // Use mean_abs_dev directly as the scale; a source deviating
                // more than outlier_z_threshold × mean_abs_dev is flagged.
                robust_std = mean_abs_dev;
            }
        }

        // Step 3: Flag outliers and adjust weights.
        let mut outlier_count = 0usize;
        for src in &mut active {
            if robust_std > 0.0 {
                let z = (src.price - median).abs() / robust_std;
                if z > self.config.outlier_z_threshold {
                    src.weight *= self.config.outlier_weight;
                    outlier_count += 1;
                    if let Some(rec) = self.sources.get_mut(&src.id) {
                        rec.outlier_count += 1;
                    }
                    self.stats.total_outlier_flags += 1;
                }
            }
        }

        // Step 4: Weighted mean → fair value.
        let total_weight: f64 = active.iter().map(|s| s.weight).sum();
        if total_weight <= 0.0 {
            self.stats.insufficient_source_count += 1;
            return None;
        }
        let fair_value: f64 = active.iter().map(|s| s.weight * s.price).sum::<f64>() / total_weight;

        // Deviations from fair value.
        let deviations: Vec<f64> = active
            .iter()
            .map(|s| (s.price - fair_value).abs())
            .collect();
        let max_deviation = deviations.iter().cloned().fold(0.0_f64, f64::max);
        let mean_deviation = if deviations.is_empty() {
            0.0
        } else {
            deviations.iter().sum::<f64>() / deviations.len() as f64
        };

        // Update per-source deviation EMA.
        let dev_alpha = self.config.ema_decay;
        for src in &active {
            let dev = (src.price - fair_value).abs();
            if let Some(rec) = self.sources.get_mut(&src.id) {
                if rec.deviation_initialized {
                    rec.ema_deviation = dev_alpha * rec.ema_deviation + (1.0 - dev_alpha) * dev;
                } else {
                    rec.ema_deviation = dev;
                    rec.deviation_initialized = true;
                }
            }
        }

        // Confidence: combines agreement (inverse of relative deviation) and
        // source count saturation.
        let relative_dev = if fair_value > 0.0 {
            mean_deviation / fair_value
        } else {
            0.0
        };
        let agreement_score = (1.0 - relative_dev * 100.0).clamp(0.0, 1.0);
        let count_score = (active.len() as f64 / 3.0).min(1.0); // saturates at 3 sources
        let weight_score = (total_weight / active.len() as f64).min(1.0);
        let confidence =
            (agreement_score * 0.5 + count_score * 0.25 + weight_score * 0.25).clamp(0.0, 1.0);

        // Step 5: EMA smoothing.
        let smoothed = if self.ema_price_initialized {
            let s =
                self.config.ema_decay * self.ema_price + (1.0 - self.config.ema_decay) * fair_value;
            self.ema_price = s;
            s
        } else {
            self.ema_price = fair_value;
            self.ema_price_initialized = true;
            fair_value
        };

        // Step 6: Volatility tracking (EMA of squared log-returns).
        let volatility = if let Some(prev) = self.prev_fair_value {
            if prev > 0.0 && fair_value > 0.0 {
                let log_return = (fair_value / prev).ln();
                let sq_return = log_return * log_return;
                self.volatility_samples += 1;

                if self.ema_variance_initialized {
                    self.ema_variance = self.config.volatility_ema_decay * self.ema_variance
                        + (1.0 - self.config.volatility_ema_decay) * sq_return;
                } else {
                    self.ema_variance = sq_return;
                    self.ema_variance_initialized = true;
                }

                if self.volatility_samples >= self.config.volatility_min_samples {
                    Some(self.ema_variance.sqrt())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        self.prev_fair_value = Some(fair_value);

        // History.
        self.history.push_back(fair_value);
        while self.history.len() > self.config.window_size {
            self.history.pop_front();
        }

        // Stats.
        self.stats.total_fusions += 1;
        if outlier_count > 0 {
            self.stats.fusions_with_outliers += 1;
        }

        Some(FusedPrice {
            fair_value,
            smoothed_price: smoothed,
            confidence,
            volatility,
            active_sources: active.len(),
            outlier_count,
            max_deviation,
            mean_deviation,
            latest_timestamp: latest_ts,
        })
    }

    // -- Accessors ---------------------------------------------------------

    /// Get per-source statistics.
    pub fn source_stats(&self) -> HashMap<String, SourcePriceStats> {
        self.sources
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    SourcePriceStats {
                        observations: r.observations,
                        outlier_count: r.outlier_count,
                        ema_deviation: r.ema_deviation,
                        last_price: r.price,
                        last_timestamp: r.timestamp,
                        last_confidence: r.confidence,
                    },
                )
            })
            .collect()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> &PriceFusionStats {
        &self.stats
    }

    /// Current number of distinct sources seen.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Windowed mean of recent fair-value prices.
    pub fn windowed_mean(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().sum();
        Some(sum / self.history.len() as f64)
    }

    /// Windowed standard deviation of recent fair-value prices.
    pub fn windowed_std(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let mean = self.windowed_mean().unwrap();
        let var: f64 = self.history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Get the most recent smoothed price (if any fusions have occurred).
    pub fn last_smoothed_price(&self) -> Option<f64> {
        if self.ema_price_initialized {
            Some(self.ema_price)
        } else {
            None
        }
    }

    /// Get the current volatility estimate (if initialised).
    pub fn current_volatility(&self) -> Option<f64> {
        if self.volatility_samples >= self.config.volatility_min_samples
            && self.ema_variance_initialized
        {
            Some(self.ema_variance.sqrt())
        } else {
            None
        }
    }

    /// Reset all state (sources, EMA, volatility, history, stats).
    pub fn reset(&mut self) {
        self.sources.clear();
        self.ema_price = 0.0;
        self.ema_price_initialized = false;
        self.prev_fair_value = None;
        self.ema_variance = 0.0;
        self.ema_variance_initialized = false;
        self.volatility_samples = 0;
        self.history.clear();
        self.stats = PriceFusionStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(source: &str, price: f64, confidence: f64, ts: f64) -> PriceObservation {
        PriceObservation {
            source_id: source.to_string(),
            price,
            confidence,
            timestamp: ts,
        }
    }

    fn default_config() -> PriceFusionConfig {
        PriceFusionConfig {
            ema_decay: 0.5,
            volatility_ema_decay: 0.5,
            outlier_z_threshold: 2.0,
            outlier_weight: 0.1,
            max_staleness_secs: 60.0,
            staleness_half_life_secs: 30.0,
            min_sources: 1,
            window_size: 100,
            volatility_min_samples: 3,
        }
    }

    fn default_fusion() -> PriceFusion {
        PriceFusion::with_config(default_config())
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = PriceFusion::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_process_invalid_ema_decay_zero() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_invalid_ema_decay_one() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_invalid_volatility_ema() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            volatility_ema_decay: 0.0,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_invalid_outlier_z_threshold() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 0.0,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_invalid_outlier_weight_negative() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_weight: -0.1,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_invalid_outlier_weight_above_one() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_weight: 1.1,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_invalid_max_staleness() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            max_staleness_secs: 0.0,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_invalid_staleness_half_life() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            staleness_half_life_secs: -1.0,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_invalid_window_size() {
        let pf = PriceFusion::with_config(PriceFusionConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(pf.process().is_err());
    }

    #[test]
    fn test_process_valid_outlier_weight_boundaries() {
        // 0.0 and 1.0 should both be valid
        let pf0 = PriceFusion::with_config(PriceFusionConfig {
            outlier_weight: 0.0,
            ..Default::default()
        });
        assert!(pf0.process().is_ok());

        let pf1 = PriceFusion::with_config(PriceFusionConfig {
            outlier_weight: 1.0,
            ..Default::default()
        });
        assert!(pf1.process().is_ok());
    }

    // -- Ingestion ---------------------------------------------------------

    #[test]
    fn test_ingest_valid() {
        let mut pf = default_fusion();
        assert!(pf.ingest(&obs("binance", 50000.0, 0.9, 100.0)).is_ok());
        assert_eq!(pf.source_count(), 1);
        assert_eq!(pf.stats().total_observations, 1);
    }

    #[test]
    fn test_ingest_zero_price() {
        let mut pf = default_fusion();
        assert!(pf.ingest(&obs("a", 0.0, 0.9, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_negative_price() {
        let mut pf = default_fusion();
        assert!(pf.ingest(&obs("a", -1.0, 0.9, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_invalid_confidence_high() {
        let mut pf = default_fusion();
        assert!(pf.ingest(&obs("a", 100.0, 1.1, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_invalid_confidence_negative() {
        let mut pf = default_fusion();
        assert!(pf.ingest(&obs("a", 100.0, -0.1, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_replaces_old_observation() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 0.9, 100.0)).unwrap();
        pf.ingest(&obs("a", 200.0, 0.8, 110.0)).unwrap();
        assert_eq!(pf.source_count(), 1);
        assert_eq!(pf.stats().total_observations, 2);

        let ss = pf.source_stats();
        assert!((ss["a"].last_price - 200.0).abs() < 1e-12);
        assert_eq!(ss["a"].observations, 2);
    }

    #[test]
    fn test_ingest_multiple_sources() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 0.9, 100.0)).unwrap();
        pf.ingest(&obs("b", 101.0, 0.8, 100.0)).unwrap();
        pf.ingest(&obs("c", 99.0, 0.7, 100.0)).unwrap();
        assert_eq!(pf.source_count(), 3);
        assert_eq!(pf.stats().distinct_sources, 3);
    }

    // -- Fusion basics -----------------------------------------------------

    #[test]
    fn test_fuse_single_source() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 50000.0, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert!(
            (fused.fair_value - 50000.0).abs() < 0.01,
            "single source should yield its own price, got {}",
            fused.fair_value
        );
        assert_eq!(fused.active_sources, 1);
        assert_eq!(fused.outlier_count, 0);
    }

    #[test]
    fn test_fuse_two_agreeing_sources() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 100.0, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert!((fused.fair_value - 100.0).abs() < 1e-12);
        assert!(fused.max_deviation < 1e-12);
        assert!(fused.mean_deviation < 1e-12);
    }

    #[test]
    fn test_fuse_weighted_by_confidence() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            staleness_half_life_secs: 1e6, // effectively no decay
            max_staleness_secs: 1e6,
            outlier_z_threshold: 100.0, // disable outlier detection
            ..default_config()
        });
        // Source a: price=100, confidence=1.0
        // Source b: price=200, confidence=0.0 (zero weight)
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 200.0, 0.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        // Zero-confidence source should not contribute
        assert!(
            (fused.fair_value - 100.0).abs() < 0.01,
            "zero-confidence source should not contribute, got {}",
            fused.fair_value
        );
    }

    #[test]
    fn test_fuse_insufficient_sources() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            min_sources: 3,
            ..default_config()
        });
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 100.0, 1.0, 100.0)).unwrap();
        assert!(pf.fuse(100.0).is_none());
        assert_eq!(pf.stats().insufficient_source_count, 1);
    }

    #[test]
    fn test_fuse_latest_timestamp() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 90.0)).unwrap();
        pf.ingest(&obs("b", 100.0, 1.0, 95.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert!((fused.latest_timestamp - 95.0).abs() < 1e-12);
    }

    // -- Outlier detection -------------------------------------------------

    #[test]
    fn test_outlier_detection_extreme_price() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 2.0,
            outlier_weight: 0.0, // fully exclude outliers
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        // Three agreeing sources and one extreme outlier
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 100.5, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("c", 99.5, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("outlier", 500.0, 1.0, 100.0)).unwrap();

        let fused = pf.fuse(100.0).unwrap();
        assert!(
            fused.outlier_count >= 1,
            "extreme price should be flagged as outlier"
        );
        // Fair value should be close to 100, not pulled towards 500
        assert!(
            (fused.fair_value - 100.0).abs() < 5.0,
            "outlier should be filtered, fair value should be ~100, got {}",
            fused.fair_value
        );
    }

    #[test]
    fn test_no_outlier_when_agreeing() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 100.1, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("c", 99.9, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert_eq!(fused.outlier_count, 0);
    }

    #[test]
    fn test_outlier_weight_reduces_influence() {
        let mut pf_exclude = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 1.5,
            outlier_weight: 0.0, // fully exclude
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        let mut pf_include = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 1.5,
            outlier_weight: 0.5, // partially include
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });

        let readings = [
            obs("a", 100.0, 1.0, 100.0),
            obs("b", 100.0, 1.0, 100.0),
            obs("c", 100.0, 1.0, 100.0),
            obs("outlier", 200.0, 1.0, 100.0),
        ];
        for r in &readings {
            pf_exclude.ingest(r).unwrap();
            pf_include.ingest(r).unwrap();
        }

        let fused_ex = pf_exclude.fuse(100.0).unwrap();
        let fused_in = pf_include.fuse(100.0).unwrap();

        // With partial inclusion, the outlier pulls fair value higher
        assert!(
            fused_in.fair_value > fused_ex.fair_value,
            "partial outlier inclusion should pull price higher: exclude={}, include={}",
            fused_ex.fair_value,
            fused_in.fair_value
        );
    }

    #[test]
    fn test_outlier_count_in_stats() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 1.5,
            outlier_weight: 0.0,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("c", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("out", 500.0, 1.0, 100.0)).unwrap();
        pf.fuse(100.0);

        assert!(pf.stats().total_outlier_flags >= 1);
        assert!(pf.stats().fusions_with_outliers >= 1);
    }

    #[test]
    fn test_source_outlier_count_tracked() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 1.5,
            outlier_weight: 0.1,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("c", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("outlier", 500.0, 1.0, 100.0)).unwrap();
        pf.fuse(100.0);

        let ss = pf.source_stats();
        assert!(
            ss["outlier"].outlier_count >= 1,
            "outlier source should have outlier_count >= 1"
        );
        assert_eq!(ss["a"].outlier_count, 0);
    }

    // -- Staleness ---------------------------------------------------------

    #[test]
    fn test_staleness_decay_reduces_weight() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            staleness_half_life_secs: 10.0,
            max_staleness_secs: 120.0,
            outlier_z_threshold: 100.0, // disable outlier detection
            ..default_config()
        });
        // Source a: fresh (t=100), price=100
        // Source b: stale (t=50, so 50s old at now=100), price=200
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 200.0, 1.0, 50.0)).unwrap();

        let fused = pf.fuse(100.0).unwrap();
        // b is 5 half-lives old → weight ≈ 1/32, so fair value should be
        // much closer to 100 than 200.
        assert!(
            (fused.fair_value - 100.0).abs() < 20.0,
            "stale source should have low weight, fair value should be near 100, got {}",
            fused.fair_value
        );
    }

    #[test]
    fn test_fully_stale_source_excluded() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            max_staleness_secs: 30.0,
            min_sources: 1,
            ..default_config()
        });
        pf.ingest(&obs("a", 100.0, 1.0, 0.0)).unwrap();
        // At t=100, the reading is 100s old → exceeds max_staleness_secs=30
        let fused = pf.fuse(100.0);
        assert!(fused.is_none(), "fully stale source should be excluded");
    }

    // -- EMA smoothing -----------------------------------------------------

    #[test]
    fn test_ema_initialization() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        let f1 = pf.fuse(100.0).unwrap();
        // First fusion: EMA initialises to raw value
        assert!((f1.smoothed_price - f1.fair_value).abs() < 1e-12);
    }

    #[test]
    fn test_ema_lags() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            ema_decay: 0.8,
            outlier_z_threshold: 100.0,
            ..default_config()
        });

        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        let f1 = pf.fuse(100.0).unwrap();
        let price1 = f1.fair_value;

        pf.ingest(&obs("a", 200.0, 1.0, 101.0)).unwrap();
        let f2 = pf.fuse(101.0).unwrap();
        let price2 = f2.fair_value;

        // Smoothed should be between price1 and price2
        assert!(
            f2.smoothed_price > price1.min(price2) && f2.smoothed_price < price1.max(price2),
            "EMA should lag: smoothed={}, price1={}, price2={}",
            f2.smoothed_price,
            price1,
            price2
        );
        // EMA: 0.8 * 100 + 0.2 * 200 = 120
        assert!(
            (f2.smoothed_price - 120.0).abs() < 1.0,
            "expected ~120, got {}",
            f2.smoothed_price
        );
    }

    #[test]
    fn test_last_smoothed_price_none_before_fusion() {
        let pf = default_fusion();
        assert!(pf.last_smoothed_price().is_none());
    }

    #[test]
    fn test_last_smoothed_price_after_fusion() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.fuse(100.0);
        assert!(pf.last_smoothed_price().is_some());
    }

    // -- Volatility tracking -----------------------------------------------

    #[test]
    fn test_volatility_not_available_initially() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            volatility_min_samples: 5,
            ..default_config()
        });
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        let f1 = pf.fuse(100.0).unwrap();
        assert!(f1.volatility.is_none());
        assert!(pf.current_volatility().is_none());
    }

    #[test]
    fn test_volatility_available_after_min_samples() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            volatility_min_samples: 3,
            volatility_ema_decay: 0.5,
            outlier_z_threshold: 100.0,
            ..default_config()
        });

        // Generate enough samples
        for i in 0..10 {
            let price = 100.0 + (i as f64) * 0.5;
            pf.ingest(&obs("a", price, 1.0, 100.0 + i as f64)).unwrap();
            pf.fuse(100.0 + i as f64);
        }

        assert!(
            pf.current_volatility().is_some(),
            "volatility should be available after enough samples"
        );
    }

    #[test]
    fn test_volatility_zero_for_constant_price() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            volatility_min_samples: 3,
            volatility_ema_decay: 0.5,
            ..default_config()
        });

        for i in 0..10 {
            pf.ingest(&obs("a", 100.0, 1.0, 100.0 + i as f64)).unwrap();
            pf.fuse(100.0 + i as f64);
        }

        let vol = pf.current_volatility().unwrap();
        assert!(
            vol < 1e-12,
            "constant price should have zero volatility, got {}",
            vol
        );
    }

    #[test]
    fn test_volatility_increases_with_price_movement() {
        let mut pf_calm = PriceFusion::with_config(PriceFusionConfig {
            volatility_min_samples: 3,
            volatility_ema_decay: 0.5,
            outlier_z_threshold: 100.0,
            ..default_config()
        });
        let mut pf_wild = PriceFusion::with_config(PriceFusionConfig {
            volatility_min_samples: 3,
            volatility_ema_decay: 0.5,
            outlier_z_threshold: 100.0,
            ..default_config()
        });

        for i in 0..10 {
            let calm_price = 100.0 + 0.01 * (i as f64);
            let wild_price = 100.0 + 5.0 * ((i as f64).sin());
            pf_calm
                .ingest(&obs("a", calm_price, 1.0, i as f64))
                .unwrap();
            pf_calm.fuse(i as f64);
            pf_wild
                .ingest(&obs("a", wild_price.max(1.0), 1.0, i as f64))
                .unwrap();
            pf_wild.fuse(i as f64);
        }

        let vol_calm = pf_calm.current_volatility().unwrap_or(0.0);
        let vol_wild = pf_wild.current_volatility().unwrap_or(0.0);
        assert!(
            vol_wild > vol_calm,
            "volatile prices should have higher vol: calm={}, wild={}",
            vol_calm,
            vol_wild
        );
    }

    // -- Deviation tracking ------------------------------------------------

    #[test]
    fn test_deviation_tracking() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 100.0, // disable outlier detection
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        // Use 3 sources so the fair value is pulled towards the majority
        // (close, close2 ≈ 100) and "far" at 150 deviates more from it.
        pf.ingest(&obs("close", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("close2", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("far", 150.0, 1.0, 100.0)).unwrap();
        pf.fuse(100.0);

        let ss = pf.source_stats();
        assert!(
            ss["far"].ema_deviation > ss["close"].ema_deviation,
            "farther source should have higher deviation: far={}, close={}",
            ss["far"].ema_deviation,
            ss["close"].ema_deviation
        );
    }

    #[test]
    fn test_max_deviation() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 100.0,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 120.0, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert!(fused.max_deviation > 0.0);
        assert!(fused.max_deviation >= fused.mean_deviation);
    }

    #[test]
    fn test_mean_deviation_symmetric_sources() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 100.0,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        pf.ingest(&obs("a", 95.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 105.0, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert!(
            (fused.fair_value - 100.0).abs() < 0.01,
            "symmetric sources should average to midpoint"
        );
        assert!(
            (fused.mean_deviation - 5.0).abs() < 0.01,
            "mean deviation should be 5.0, got {}",
            fused.mean_deviation
        );
    }

    // -- Confidence scoring ------------------------------------------------

    #[test]
    fn test_confidence_higher_with_more_sources() {
        let mut pf_few = default_fusion();
        let mut pf_many = default_fusion();

        pf_few.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        let fused_few = pf_few.fuse(100.0).unwrap();

        pf_many.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf_many.ingest(&obs("b", 100.0, 1.0, 100.0)).unwrap();
        pf_many.ingest(&obs("c", 100.0, 1.0, 100.0)).unwrap();
        let fused_many = pf_many.fuse(100.0).unwrap();

        assert!(
            fused_many.confidence >= fused_few.confidence,
            "more agreeing sources should give higher confidence: few={}, many={}",
            fused_few.confidence,
            fused_many.confidence
        );
    }

    #[test]
    fn test_confidence_lower_with_disagreement() {
        let mut pf_agree = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 100.0,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        let mut pf_disagree = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 100.0,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });

        // Agreeing sources
        pf_agree.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf_agree.ingest(&obs("b", 100.0, 1.0, 100.0)).unwrap();
        let fused_agree = pf_agree.fuse(100.0).unwrap();

        // Disagreeing sources
        pf_disagree.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf_disagree.ingest(&obs("b", 105.0, 1.0, 100.0)).unwrap();
        let fused_disagree = pf_disagree.fuse(100.0).unwrap();

        assert!(
            fused_agree.confidence >= fused_disagree.confidence,
            "disagreement should lower confidence: agree={}, disagree={}",
            fused_agree.confidence,
            fused_disagree.confidence
        );
    }

    #[test]
    fn test_confidence_in_valid_range() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert!(
            fused.confidence >= 0.0 && fused.confidence <= 1.0,
            "confidence should be in [0, 1], got {}",
            fused.confidence
        );
    }

    // -- Windowed statistics -----------------------------------------------

    #[test]
    fn test_windowed_mean() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.fuse(100.0);
        pf.ingest(&obs("a", 200.0, 1.0, 101.0)).unwrap();
        pf.fuse(101.0);
        let mean = pf.windowed_mean().unwrap();
        assert!(
            (mean - 150.0).abs() < 0.01,
            "mean of 100 and 200 should be 150, got {}",
            mean
        );
    }

    #[test]
    fn test_windowed_std_constant() {
        let mut pf = default_fusion();
        for i in 0..5 {
            pf.ingest(&obs("a", 100.0, 1.0, 100.0 + i as f64)).unwrap();
            pf.fuse(100.0 + i as f64);
        }
        let std = pf.windowed_std().unwrap();
        assert!(
            std < 1e-12,
            "constant prices should have zero std, got {}",
            std
        );
    }

    #[test]
    fn test_windowed_mean_empty() {
        let pf = default_fusion();
        assert!(pf.windowed_mean().is_none());
    }

    #[test]
    fn test_windowed_std_insufficient() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.fuse(100.0);
        assert!(pf.windowed_std().is_none());
    }

    // -- Reset -------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut pf = default_fusion();
        for i in 0..10 {
            let price = 100.0 + i as f64;
            pf.ingest(&obs("a", price, 1.0, 100.0 + i as f64)).unwrap();
            pf.fuse(100.0 + i as f64);
        }
        assert!(pf.source_count() > 0);
        assert!(pf.stats().total_fusions > 0);
        assert!(pf.last_smoothed_price().is_some());

        pf.reset();

        assert_eq!(pf.source_count(), 0);
        assert_eq!(pf.stats().total_fusions, 0);
        assert_eq!(pf.stats().total_observations, 0);
        assert!(pf.windowed_mean().is_none());
        assert!(pf.last_smoothed_price().is_none());
        assert!(pf.current_volatility().is_none());
    }

    // -- Stats tracking ----------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 100.0, 1.0, 100.0)).unwrap();
        pf.fuse(100.0);
        pf.fuse(100.0);

        let s = pf.stats();
        assert_eq!(s.total_observations, 2);
        assert_eq!(s.total_fusions, 2);
        assert_eq!(s.distinct_sources, 2);
    }

    // -- Window eviction ---------------------------------------------------

    #[test]
    fn test_window_eviction() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            window_size: 3,
            ..default_config()
        });
        for i in 0..10 {
            pf.ingest(&obs("a", 100.0, 1.0, i as f64)).unwrap();
            pf.fuse(i as f64);
        }
        assert_eq!(pf.history.len(), 3);
    }

    // -- Edge cases --------------------------------------------------------

    #[test]
    fn test_all_zero_confidence_returns_none() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 0.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 200.0, 0.0, 100.0)).unwrap();
        // All weights are zero → should return None (insufficient weight)
        let fused = pf.fuse(100.0);
        assert!(
            fused.is_none(),
            "all-zero-confidence should not produce a fused price"
        );
    }

    #[test]
    fn test_single_source_confidence_score() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        // With only 1 source, confidence should still be positive (but not maximal)
        assert!(fused.confidence > 0.0);
    }

    #[test]
    fn test_very_close_prices_no_outlier() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 2.0,
            ..default_config()
        });
        pf.ingest(&obs("a", 100.000, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 100.001, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("c", 99.999, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert_eq!(fused.outlier_count, 0);
    }

    #[test]
    fn test_source_stats_after_multiple_fusions() {
        let mut pf = default_fusion();
        pf.ingest(&obs("a", 100.0, 0.9, 100.0)).unwrap();
        pf.fuse(100.0);
        pf.ingest(&obs("a", 101.0, 0.95, 101.0)).unwrap();
        pf.fuse(101.0);

        let ss = pf.source_stats();
        assert_eq!(ss["a"].observations, 2);
        assert!((ss["a"].last_price - 101.0).abs() < 1e-12);
        assert!((ss["a"].last_confidence - 0.95).abs() < 1e-12);
    }

    #[test]
    fn test_active_sources_count() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            max_staleness_secs: 10.0,
            ..default_config()
        });
        pf.ingest(&obs("a", 100.0, 1.0, 95.0)).unwrap();
        pf.ingest(&obs("b", 100.0, 1.0, 98.0)).unwrap();
        pf.ingest(&obs("c", 100.0, 1.0, 80.0)).unwrap(); // stale at t=100
        let fused = pf.fuse(100.0).unwrap();
        assert_eq!(
            fused.active_sources, 2,
            "only 2 non-stale sources should be active"
        );
    }

    #[test]
    fn test_fair_value_between_source_prices() {
        let mut pf = PriceFusion::with_config(PriceFusionConfig {
            outlier_z_threshold: 100.0,
            staleness_half_life_secs: 1e6,
            max_staleness_secs: 1e6,
            ..default_config()
        });
        pf.ingest(&obs("a", 90.0, 1.0, 100.0)).unwrap();
        pf.ingest(&obs("b", 110.0, 1.0, 100.0)).unwrap();
        let fused = pf.fuse(100.0).unwrap();
        assert!(
            fused.fair_value >= 90.0 && fused.fair_value <= 110.0,
            "fair value should be between source prices, got {}",
            fused.fair_value
        );
    }
}
