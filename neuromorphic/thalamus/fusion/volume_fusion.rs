//! Volume data fusion
//!
//! Part of the Thalamus region
//! Component: fusion
//!
//! Aggregates volume data from multiple exchanges / venues into a unified
//! view of market-wide trading activity.
//!
//! ## Features
//!
//! - **Multi-venue aggregation**: Combines volume observations from an
//!   arbitrary number of named venues into a single consolidated picture.
//! - **VWAP tracking**: Maintains a running volume-weighted average price
//!   across all venues.
//! - **Volume profile**: Builds a coarse price-bucketed volume profile so
//!   downstream components can reason about where liquidity concentrates.
//! - **Anomaly detection**: Flags volume spikes that exceed a configurable
//!   multiple of the EMA-smoothed baseline.
//! - **EMA smoothing**: Total volume rate is smoothed with an exponential
//!   moving average to reduce noise.
//! - **Per-venue statistics**: Tracks each venue's contribution share,
//!   observation count, and last-seen timestamp.

use std::collections::{HashMap, VecDeque};

use crate::common::{Error, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the volume fusion engine.
#[derive(Debug, Clone)]
pub struct VolumeFusionConfig {
    /// EMA decay factor for the smoothed total volume rate (0, 1).
    /// Closer to 1 → more smoothing / slower reaction.
    pub ema_decay: f64,
    /// Multiplier above the EMA-smoothed volume that triggers an anomaly flag.
    /// For example 3.0 means a reading > 3× the smoothed rate is anomalous.
    pub anomaly_multiplier: f64,
    /// Number of price buckets in the volume profile.
    pub profile_buckets: usize,
    /// Maximum number of recent fused snapshots to retain for windowed stats.
    pub window_size: usize,
    /// Minimum number of venues required before producing a fused snapshot.
    pub min_venues: usize,
    /// Maximum age (seconds) before a venue's reading is considered stale and
    /// excluded from the current fusion snapshot.
    pub max_staleness_secs: f64,
    /// Width of each volume-profile price bucket (in price units).
    /// Set to 0.0 to disable profiling.
    pub profile_bucket_width: f64,
    /// Reference price around which the profile is centred. If `None`, the
    /// engine auto-centres on the first observed mid price.
    pub profile_center_price: Option<f64>,
}

impl Default for VolumeFusionConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.85,
            anomaly_multiplier: 3.0,
            profile_buckets: 50,
            window_size: 500,
            min_venues: 1,
            max_staleness_secs: 30.0,
            profile_bucket_width: 1.0,
            profile_center_price: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// A volume observation from a single venue.
#[derive(Debug, Clone)]
pub struct VolumeObservation {
    /// Venue identifier (e.g. "binance", "coinbase").
    pub venue_id: String,
    /// Total volume traded in this observation period (base-currency units).
    pub volume: f64,
    /// Volume-weighted average price for this observation period.
    pub vwap: f64,
    /// Timestamp of the observation (seconds since epoch or monotonic ref).
    pub timestamp: f64,
    /// Optional: number of trades in this observation period.
    pub trade_count: Option<u64>,
}

/// The fused volume snapshot.
#[derive(Debug, Clone)]
pub struct FusedVolume {
    /// Total volume across all active venues for this snapshot.
    pub total_volume: f64,
    /// EMA-smoothed total volume rate.
    pub smoothed_volume: f64,
    /// Consolidated VWAP across all active venues.
    pub vwap: f64,
    /// Whether the current total volume is anomalously high.
    pub is_anomaly: bool,
    /// Ratio of current volume to smoothed volume (spike magnitude).
    pub spike_ratio: f64,
    /// Number of active (non-stale) venues contributing.
    pub active_venues: usize,
    /// Venue-level volume shares (venue_id → fraction of total).
    pub venue_shares: HashMap<String, f64>,
    /// Total trade count (if all venues reported it).
    pub total_trade_count: Option<u64>,
    /// Timestamp of the most recent observation used.
    pub latest_timestamp: f64,
}

/// Per-venue statistics.
#[derive(Debug, Clone)]
pub struct VenueStats {
    /// Total observations ingested from this venue.
    pub observations: usize,
    /// Cumulative volume from this venue.
    pub cumulative_volume: f64,
    /// Most recent volume observation.
    pub last_volume: f64,
    /// Most recent VWAP.
    pub last_vwap: f64,
    /// Most recent timestamp.
    pub last_timestamp: f64,
    /// Historical share of total volume (EMA-smoothed).
    pub ema_share: f64,
}

/// Aggregate statistics for the fusion engine.
#[derive(Debug, Clone, Default)]
pub struct VolumeFusionStats {
    /// Total observations ingested.
    pub total_observations: usize,
    /// Total fused snapshots produced.
    pub total_fusions: usize,
    /// Number of snapshots flagged as anomalous.
    pub anomaly_count: usize,
    /// Number of snapshots skipped due to insufficient venues.
    pub insufficient_venue_count: usize,
    /// Number of distinct venues seen.
    pub distinct_venues: usize,
    /// Cumulative volume across all observations.
    pub cumulative_volume: f64,
}

/// A single bucket in the volume profile.
#[derive(Debug, Clone)]
pub struct ProfileBucket {
    /// Lower price bound (inclusive).
    pub price_low: f64,
    /// Upper price bound (exclusive).
    pub price_high: f64,
    /// Total volume accumulated in this bucket.
    pub volume: f64,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct VenueRecord {
    volume: f64,
    vwap: f64,
    timestamp: f64,
    trade_count: Option<u64>,
    cumulative_volume: f64,
    observation_count: usize,
    ema_share: f64,
    share_initialized: bool,
}

// ---------------------------------------------------------------------------
// VolumeFusion
// ---------------------------------------------------------------------------

/// Multi-venue volume fusion engine.
///
/// Call [`ingest`] to feed observations from individual venues, then call
/// [`fuse`] to obtain the aggregated volume snapshot at a given time.
pub struct VolumeFusion {
    config: VolumeFusionConfig,
    /// Most recent record per venue.
    venues: HashMap<String, VenueRecord>,
    /// EMA state for smoothed total volume.
    ema_volume: f64,
    ema_initialized: bool,
    /// Volume profile buckets.
    profile: Vec<f64>,
    profile_center: Option<f64>,
    /// Windowed history of total volumes.
    history: VecDeque<f64>,
    /// Running statistics.
    stats: VolumeFusionStats,
}

impl Default for VolumeFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl VolumeFusion {
    /// Create a new instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(VolumeFusionConfig::default())
    }

    /// Create a new instance with the given configuration.
    pub fn with_config(config: VolumeFusionConfig) -> Self {
        let n_buckets = config.profile_buckets;
        Self {
            profile: vec![0.0; n_buckets],
            profile_center: config.profile_center_price,
            venues: HashMap::new(),
            ema_volume: 0.0,
            ema_initialized: false,
            history: VecDeque::with_capacity(config.window_size),
            stats: VolumeFusionStats::default(),
            config,
        }
    }

    /// Validate configuration parameters.
    pub fn process(&self) -> Result<()> {
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.anomaly_multiplier <= 0.0 {
            return Err(Error::InvalidInput("anomaly_multiplier must be > 0".into()));
        }
        if self.config.profile_buckets == 0 {
            return Err(Error::InvalidInput("profile_buckets must be > 0".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.max_staleness_secs <= 0.0 {
            return Err(Error::InvalidInput("max_staleness_secs must be > 0".into()));
        }
        if self.config.profile_bucket_width < 0.0 {
            return Err(Error::InvalidInput(
                "profile_bucket_width must be >= 0".into(),
            ));
        }
        Ok(())
    }

    // -- Ingestion ---------------------------------------------------------

    /// Ingest a single volume observation from a venue.
    pub fn ingest(&mut self, obs: &VolumeObservation) -> Result<()> {
        if obs.volume < 0.0 {
            return Err(Error::InvalidInput("volume must be >= 0".into()));
        }
        if obs.vwap < 0.0 {
            return Err(Error::InvalidInput("vwap must be >= 0".into()));
        }

        self.stats.total_observations += 1;
        self.stats.cumulative_volume += obs.volume;

        if !self.venues.contains_key(&obs.venue_id) {
            self.stats.distinct_venues += 1;
        }

        let prev = self.venues.get(&obs.venue_id);
        let cum = prev.map(|r| r.cumulative_volume).unwrap_or(0.0) + obs.volume;
        let obs_count = prev.map(|r| r.observation_count).unwrap_or(0) + 1;
        let prev_share = prev.map(|r| r.ema_share).unwrap_or(0.0);
        let share_init = prev.map(|r| r.share_initialized).unwrap_or(false);

        self.venues.insert(
            obs.venue_id.clone(),
            VenueRecord {
                volume: obs.volume,
                vwap: obs.vwap,
                timestamp: obs.timestamp,
                trade_count: obs.trade_count,
                cumulative_volume: cum,
                observation_count: obs_count,
                ema_share: prev_share,
                share_initialized: share_init,
            },
        );

        // Update volume profile
        self.update_profile(obs.vwap, obs.volume);

        Ok(())
    }

    // -- Fusion ------------------------------------------------------------

    /// Produce a fused volume snapshot at time `now`.
    ///
    /// Returns `None` if fewer than `min_venues` non-stale venues are
    /// available.
    pub fn fuse(&mut self, now: f64) -> Option<FusedVolume> {
        // Collect active (non-stale) venues into owned data to avoid
        // holding an immutable borrow on self.venues while we later need
        // a mutable borrow to update share EMAs.
        struct ActiveVenue {
            id: String,
            volume: f64,
            vwap: f64,
            timestamp: f64,
            trade_count: Option<u64>,
        }

        let mut active: Vec<ActiveVenue> = Vec::new();
        for (id, rec) in &self.venues {
            let age = (now - rec.timestamp).max(0.0);
            if age <= self.config.max_staleness_secs {
                active.push(ActiveVenue {
                    id: id.clone(),
                    volume: rec.volume,
                    vwap: rec.vwap,
                    timestamp: rec.timestamp,
                    trade_count: rec.trade_count,
                });
            }
        }

        if active.len() < self.config.min_venues {
            self.stats.insufficient_venue_count += 1;
            return None;
        }

        // Aggregate
        let total_volume: f64 = active.iter().map(|a| a.volume).sum();
        let mut latest_ts = f64::NEG_INFINITY;

        // VWAP: volume-weighted average of per-venue VWAPs
        let total_vol_for_vwap: f64 = active
            .iter()
            .filter(|a| a.volume > 0.0)
            .map(|a| a.volume)
            .sum();
        let vwap = if total_vol_for_vwap > 0.0 {
            active
                .iter()
                .filter(|a| a.volume > 0.0)
                .map(|a| a.vwap * a.volume)
                .sum::<f64>()
                / total_vol_for_vwap
        } else {
            // Fallback: simple average
            let count = active.len() as f64;
            if count > 0.0 {
                active.iter().map(|a| a.vwap).sum::<f64>() / count
            } else {
                0.0
            }
        };

        // Per-venue shares
        let mut venue_shares = HashMap::new();
        for a in &active {
            if total_volume > 0.0 {
                venue_shares.insert(a.id.clone(), a.volume / total_volume);
            } else {
                venue_shares.insert(a.id.clone(), 0.0);
            }
            if a.timestamp > latest_ts {
                latest_ts = a.timestamp;
            }
        }

        // Total trade count (only if all venues reported it)
        let trade_counts: Vec<u64> = active.iter().filter_map(|a| a.trade_count).collect();
        let active_count = active.len();
        let total_trade_count = if trade_counts.len() == active_count {
            Some(trade_counts.iter().sum())
        } else {
            None
        };

        // EMA smoothing
        let smoothed = if self.ema_initialized {
            let s = self.config.ema_decay * self.ema_volume
                + (1.0 - self.config.ema_decay) * total_volume;
            self.ema_volume = s;
            s
        } else {
            self.ema_volume = total_volume;
            self.ema_initialized = true;
            total_volume
        };

        // Anomaly detection
        let spike_ratio = if smoothed > 0.0 {
            total_volume / smoothed
        } else {
            1.0
        };
        let is_anomaly = self.ema_initialized && spike_ratio > self.config.anomaly_multiplier;

        // Update per-venue share EMA (no borrow conflict since `active` is owned)
        let share_alpha = self.config.ema_decay;
        for (id, share) in &venue_shares {
            if let Some(rec) = self.venues.get_mut(id) {
                if rec.share_initialized {
                    rec.ema_share = share_alpha * rec.ema_share + (1.0 - share_alpha) * share;
                } else {
                    rec.ema_share = *share;
                    rec.share_initialized = true;
                }
            }
        }

        // History
        self.history.push_back(total_volume);
        while self.history.len() > self.config.window_size {
            self.history.pop_front();
        }

        self.stats.total_fusions += 1;
        if is_anomaly {
            self.stats.anomaly_count += 1;
        }

        Some(FusedVolume {
            total_volume,
            smoothed_volume: smoothed,
            vwap,
            is_anomaly,
            spike_ratio,
            active_venues: active_count,
            venue_shares,
            total_trade_count,
            latest_timestamp: latest_ts,
        })
    }

    // -- Volume profile ----------------------------------------------------

    fn update_profile(&mut self, price: f64, volume: f64) {
        if self.config.profile_bucket_width <= 0.0 || volume <= 0.0 {
            return;
        }
        let center = match self.profile_center {
            Some(c) => c,
            None => {
                self.profile_center = Some(price);
                price
            }
        };

        let n = self.config.profile_buckets;
        let half = n as f64 / 2.0;
        let bucket_idx_f = ((price - center) / self.config.profile_bucket_width + half).floor();
        let bucket_idx = bucket_idx_f as isize;
        if bucket_idx >= 0 && (bucket_idx as usize) < n {
            self.profile[bucket_idx as usize] += volume;
        }
    }

    /// Get the current volume profile as a list of price buckets.
    pub fn volume_profile(&self) -> Vec<ProfileBucket> {
        let center = self.profile_center.unwrap_or(0.0);
        let n = self.config.profile_buckets;
        let half = n as f64 / 2.0;
        let w = self.config.profile_bucket_width;

        (0..n)
            .map(|i| {
                let low = center + (i as f64 - half) * w;
                ProfileBucket {
                    price_low: low,
                    price_high: low + w,
                    volume: self.profile[i],
                }
            })
            .collect()
    }

    /// Get the price bucket with the highest accumulated volume (Point of
    /// Control). Returns `None` if no volume has been recorded.
    pub fn point_of_control(&self) -> Option<ProfileBucket> {
        if self.profile.iter().all(|v| *v <= 0.0) {
            return None;
        }
        let (max_idx, _) = self
            .profile
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        let center = self.profile_center.unwrap_or(0.0);
        let n = self.config.profile_buckets;
        let half = n as f64 / 2.0;
        let w = self.config.profile_bucket_width;
        let low = center + (max_idx as f64 - half) * w;

        Some(ProfileBucket {
            price_low: low,
            price_high: low + w,
            volume: self.profile[max_idx],
        })
    }

    // -- Accessors ---------------------------------------------------------

    /// Get per-venue statistics.
    pub fn venue_stats(&self) -> HashMap<String, VenueStats> {
        self.venues
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    VenueStats {
                        observations: r.observation_count,
                        cumulative_volume: r.cumulative_volume,
                        last_volume: r.volume,
                        last_vwap: r.vwap,
                        last_timestamp: r.timestamp,
                        ema_share: r.ema_share,
                    },
                )
            })
            .collect()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> &VolumeFusionStats {
        &self.stats
    }

    /// Current number of distinct venues seen.
    pub fn venue_count(&self) -> usize {
        self.venues.len()
    }

    /// Windowed mean of recent total volume readings.
    pub fn windowed_mean(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().sum();
        Some(sum / self.history.len() as f64)
    }

    /// Windowed standard deviation of recent total volume readings.
    pub fn windowed_std(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let mean = self.windowed_mean().unwrap();
        let var: f64 = self.history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Reset all state (venues, EMA, profile, history, stats).
    pub fn reset(&mut self) {
        self.venues.clear();
        self.ema_volume = 0.0;
        self.ema_initialized = false;
        self.profile = vec![0.0; self.config.profile_buckets];
        self.profile_center = self.config.profile_center_price;
        self.history.clear();
        self.stats = VolumeFusionStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(venue: &str, volume: f64, vwap: f64, ts: f64) -> VolumeObservation {
        VolumeObservation {
            venue_id: venue.to_string(),
            volume,
            vwap,
            timestamp: ts,
            trade_count: None,
        }
    }

    fn obs_with_trades(
        venue: &str,
        volume: f64,
        vwap: f64,
        ts: f64,
        trades: u64,
    ) -> VolumeObservation {
        VolumeObservation {
            venue_id: venue.to_string(),
            volume,
            vwap,
            timestamp: ts,
            trade_count: Some(trades),
        }
    }

    fn default_config() -> VolumeFusionConfig {
        VolumeFusionConfig {
            ema_decay: 0.5,
            anomaly_multiplier: 3.0,
            profile_buckets: 20,
            window_size: 100,
            min_venues: 1,
            max_staleness_secs: 60.0,
            profile_bucket_width: 10.0,
            profile_center_price: Some(100.0),
        }
    }

    fn default_fusion() -> VolumeFusion {
        VolumeFusion::with_config(default_config())
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn test_basic() {
        let instance = VolumeFusion::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_process_invalid_ema_decay() {
        let vf = VolumeFusion::with_config(VolumeFusionConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(vf.process().is_err());
    }

    #[test]
    fn test_process_invalid_anomaly_multiplier() {
        let vf = VolumeFusion::with_config(VolumeFusionConfig {
            anomaly_multiplier: -1.0,
            ..Default::default()
        });
        assert!(vf.process().is_err());
    }

    #[test]
    fn test_process_invalid_profile_buckets() {
        let vf = VolumeFusion::with_config(VolumeFusionConfig {
            profile_buckets: 0,
            ..Default::default()
        });
        assert!(vf.process().is_err());
    }

    #[test]
    fn test_process_invalid_window_size() {
        let vf = VolumeFusion::with_config(VolumeFusionConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(vf.process().is_err());
    }

    #[test]
    fn test_process_invalid_max_staleness() {
        let vf = VolumeFusion::with_config(VolumeFusionConfig {
            max_staleness_secs: 0.0,
            ..Default::default()
        });
        assert!(vf.process().is_err());
    }

    #[test]
    fn test_process_invalid_bucket_width() {
        let vf = VolumeFusion::with_config(VolumeFusionConfig {
            profile_bucket_width: -1.0,
            ..Default::default()
        });
        assert!(vf.process().is_err());
    }

    // -- Ingestion ---------------------------------------------------------

    #[test]
    fn test_ingest_valid() {
        let mut vf = default_fusion();
        assert!(vf.ingest(&obs("binance", 100.0, 50000.0, 100.0)).is_ok());
        assert_eq!(vf.venue_count(), 1);
        assert_eq!(vf.stats().total_observations, 1);
    }

    #[test]
    fn test_ingest_negative_volume() {
        let mut vf = default_fusion();
        assert!(vf.ingest(&obs("binance", -1.0, 50000.0, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_negative_vwap() {
        let mut vf = default_fusion();
        assert!(vf.ingest(&obs("binance", 100.0, -1.0, 100.0)).is_err());
    }

    #[test]
    fn test_ingest_multiple_venues() {
        let mut vf = default_fusion();
        vf.ingest(&obs("binance", 100.0, 50000.0, 100.0)).unwrap();
        vf.ingest(&obs("coinbase", 80.0, 50010.0, 100.0)).unwrap();
        vf.ingest(&obs("kraken", 60.0, 49990.0, 100.0)).unwrap();
        assert_eq!(vf.venue_count(), 3);
        assert_eq!(vf.stats().distinct_venues, 3);
    }

    #[test]
    fn test_ingest_replaces_old_reading() {
        let mut vf = default_fusion();
        vf.ingest(&obs("binance", 100.0, 50000.0, 100.0)).unwrap();
        vf.ingest(&obs("binance", 200.0, 50100.0, 110.0)).unwrap();
        assert_eq!(vf.venue_count(), 1);
        assert_eq!(vf.stats().total_observations, 2);

        let vs = vf.venue_stats();
        assert!((vs["binance"].last_volume - 200.0).abs() < 1e-12);
        assert!((vs["binance"].cumulative_volume - 300.0).abs() < 1e-12);
    }

    // -- Fusion basics -----------------------------------------------------

    #[test]
    fn test_fuse_single_venue() {
        let mut vf = default_fusion();
        vf.ingest(&obs("binance", 100.0, 50000.0, 100.0)).unwrap();
        let fused = vf.fuse(100.0).unwrap();
        assert!((fused.total_volume - 100.0).abs() < 1e-12);
        assert!((fused.vwap - 50000.0).abs() < 1e-12);
        assert_eq!(fused.active_venues, 1);
    }

    #[test]
    fn test_fuse_two_venues_vwap() {
        let mut vf = default_fusion();
        // binance: 100 vol @ 50000
        // coinbase: 200 vol @ 50300
        vf.ingest(&obs("binance", 100.0, 50000.0, 100.0)).unwrap();
        vf.ingest(&obs("coinbase", 200.0, 50300.0, 100.0)).unwrap();
        let fused = vf.fuse(100.0).unwrap();
        // VWAP = (100*50000 + 200*50300) / 300 = (5000000 + 10060000) / 300 = 50200
        assert!(
            (fused.vwap - 50200.0).abs() < 0.01,
            "expected VWAP ~50200, got {}",
            fused.vwap
        );
        assert!((fused.total_volume - 300.0).abs() < 1e-12);
        assert_eq!(fused.active_venues, 2);
    }

    #[test]
    fn test_fuse_venue_shares() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        vf.ingest(&obs("b", 300.0, 100.0, 100.0)).unwrap();
        let fused = vf.fuse(100.0).unwrap();
        assert!((fused.venue_shares["a"] - 0.25).abs() < 1e-12);
        assert!((fused.venue_shares["b"] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_fuse_insufficient_venues() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            min_venues: 3,
            ..default_config()
        });
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        vf.ingest(&obs("b", 100.0, 100.0, 100.0)).unwrap();
        assert!(vf.fuse(100.0).is_none());
        assert_eq!(vf.stats().insufficient_venue_count, 1);
    }

    #[test]
    fn test_fuse_latest_timestamp() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        vf.ingest(&obs("b", 100.0, 100.0, 200.0)).unwrap();
        let fused = vf.fuse(200.0).unwrap();
        assert!((fused.latest_timestamp - 200.0).abs() < 1e-12);
    }

    // -- Staleness ---------------------------------------------------------

    #[test]
    fn test_stale_venue_excluded() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            max_staleness_secs: 30.0,
            min_venues: 1,
            ..default_config()
        });
        vf.ingest(&obs("old", 100.0, 100.0, 10.0)).unwrap();
        vf.ingest(&obs("fresh", 200.0, 100.0, 90.0)).unwrap();
        // At t=100: old is 90s stale (> 30s), fresh is 10s
        let fused = vf.fuse(100.0).unwrap();
        assert_eq!(fused.active_venues, 1);
        assert!((fused.total_volume - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_all_stale_returns_none() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            max_staleness_secs: 10.0,
            min_venues: 1,
            ..default_config()
        });
        vf.ingest(&obs("a", 100.0, 100.0, 10.0)).unwrap();
        assert!(vf.fuse(100.0).is_none());
    }

    // -- Anomaly detection -------------------------------------------------

    #[test]
    fn test_anomaly_detection_spike() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            anomaly_multiplier: 2.0,
            ema_decay: 0.9,
            ..default_config()
        });

        // Build up a baseline with moderate volume
        for i in 0..20 {
            vf.ingest(&obs("a", 100.0, 100.0, i as f64)).unwrap();
            vf.fuse(i as f64);
        }

        // Now inject a massive spike
        vf.ingest(&obs("a", 10000.0, 100.0, 20.0)).unwrap();
        let fused = vf.fuse(20.0).unwrap();
        assert!(
            fused.is_anomaly,
            "spike should trigger anomaly, ratio={}",
            fused.spike_ratio
        );
        assert!(fused.spike_ratio > 2.0);
    }

    #[test]
    fn test_no_anomaly_normal_volume() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        let fused = vf.fuse(100.0).unwrap();
        assert!(!fused.is_anomaly);
    }

    // -- EMA smoothing -----------------------------------------------------

    #[test]
    fn test_ema_smoothing_initialisation() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        let f1 = vf.fuse(100.0).unwrap();
        // First fusion: EMA initialises to raw value
        assert!((f1.smoothed_volume - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_ema_smoothing_lags() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            ema_decay: 0.8,
            ..default_config()
        });
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        vf.fuse(100.0); // EMA = 100

        vf.ingest(&obs("a", 500.0, 100.0, 101.0)).unwrap();
        let f2 = vf.fuse(101.0).unwrap();
        // EMA: 0.8 * 100 + 0.2 * 500 = 80 + 100 = 180
        assert!(
            (f2.smoothed_volume - 180.0).abs() < 0.01,
            "expected ~180, got {}",
            f2.smoothed_volume
        );
    }

    // -- Trade count -------------------------------------------------------

    #[test]
    fn test_total_trade_count_all_reported() {
        let mut vf = default_fusion();
        vf.ingest(&obs_with_trades("a", 100.0, 100.0, 100.0, 50))
            .unwrap();
        vf.ingest(&obs_with_trades("b", 200.0, 100.0, 100.0, 80))
            .unwrap();
        let fused = vf.fuse(100.0).unwrap();
        assert_eq!(fused.total_trade_count, Some(130));
    }

    #[test]
    fn test_total_trade_count_partial() {
        let mut vf = default_fusion();
        vf.ingest(&obs_with_trades("a", 100.0, 100.0, 100.0, 50))
            .unwrap();
        vf.ingest(&obs("b", 200.0, 100.0, 100.0)).unwrap();
        let fused = vf.fuse(100.0).unwrap();
        assert!(
            fused.total_trade_count.is_none(),
            "trade count should be None when not all venues report it"
        );
    }

    // -- Volume profile ----------------------------------------------------

    #[test]
    fn test_volume_profile_basic() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            profile_buckets: 10,
            profile_bucket_width: 10.0,
            profile_center_price: Some(100.0),
            ..default_config()
        });
        vf.ingest(&obs("a", 500.0, 100.0, 100.0)).unwrap();
        let profile = vf.volume_profile();
        assert_eq!(profile.len(), 10);

        // Volume should be concentrated near the center
        let total_vol: f64 = profile.iter().map(|b| b.volume).sum();
        assert!((total_vol - 500.0).abs() < 1e-12);
    }

    #[test]
    fn test_point_of_control() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            profile_buckets: 20,
            profile_bucket_width: 10.0,
            profile_center_price: Some(100.0),
            ..default_config()
        });
        // Most volume at price 100
        vf.ingest(&obs("a", 1000.0, 100.0, 100.0)).unwrap();
        vf.ingest(&obs("b", 100.0, 150.0, 100.0)).unwrap();

        let poc = vf.point_of_control().unwrap();
        assert!(
            poc.volume >= 1000.0,
            "PoC should be at the high-volume bucket"
        );
        assert!(poc.price_low <= 100.0 && poc.price_high > 100.0);
    }

    #[test]
    fn test_point_of_control_empty() {
        let vf = default_fusion();
        assert!(vf.point_of_control().is_none());
    }

    #[test]
    fn test_profile_auto_centers() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            profile_center_price: None,
            profile_buckets: 20,
            profile_bucket_width: 10.0,
            ..default_config()
        });
        vf.ingest(&obs("a", 100.0, 50000.0, 100.0)).unwrap();
        let profile = vf.volume_profile();
        // Should be centred around 50000
        let mid = &profile[10]; // middle bucket
        assert!(
            mid.price_low <= 50000.0 && mid.price_high > 50000.0,
            "profile should auto-centre on first price"
        );
    }

    // -- Windowed statistics -----------------------------------------------

    #[test]
    fn test_windowed_mean() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        vf.fuse(100.0);
        vf.ingest(&obs("a", 300.0, 100.0, 101.0)).unwrap();
        vf.fuse(101.0);
        let mean = vf.windowed_mean().unwrap();
        assert!(
            (mean - 200.0).abs() < 0.01,
            "mean of 100 and 300 should be 200, got {}",
            mean
        );
    }

    #[test]
    fn test_windowed_std() {
        let mut vf = default_fusion();
        for i in 0..10 {
            vf.ingest(&obs("a", 100.0, 100.0, 100.0 + i as f64))
                .unwrap();
            vf.fuse(100.0 + i as f64);
        }
        let std = vf.windowed_std().unwrap();
        assert!(std < 1e-12, "constant volumes should have zero std");
    }

    #[test]
    fn test_windowed_mean_empty() {
        let vf = default_fusion();
        assert!(vf.windowed_mean().is_none());
    }

    #[test]
    fn test_windowed_std_insufficient() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        vf.fuse(100.0);
        assert!(vf.windowed_std().is_none());
    }

    // -- Reset -------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut vf = default_fusion();
        for i in 0..10 {
            vf.ingest(&obs("a", 100.0, 100.0, i as f64)).unwrap();
            vf.fuse(i as f64);
        }
        assert!(vf.venue_count() > 0);
        assert!(vf.stats().total_fusions > 0);

        vf.reset();

        assert_eq!(vf.venue_count(), 0);
        assert_eq!(vf.stats().total_fusions, 0);
        assert_eq!(vf.stats().total_observations, 0);
        assert!(vf.windowed_mean().is_none());
        assert!(vf.point_of_control().is_none());
    }

    // -- Stats tracking ----------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        vf.ingest(&obs("b", 200.0, 100.0, 100.0)).unwrap();
        vf.fuse(100.0);
        vf.fuse(100.0);

        let s = vf.stats();
        assert_eq!(s.total_observations, 2);
        assert_eq!(s.total_fusions, 2);
        assert_eq!(s.distinct_venues, 2);
        assert!((s.cumulative_volume - 300.0).abs() < 1e-12);
    }

    #[test]
    fn test_anomaly_count_in_stats() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            anomaly_multiplier: 2.0,
            ema_decay: 0.99,
            ..default_config()
        });
        // Build baseline
        for i in 0..20 {
            vf.ingest(&obs("a", 100.0, 100.0, i as f64)).unwrap();
            vf.fuse(i as f64);
        }
        // Spike
        vf.ingest(&obs("a", 50000.0, 100.0, 20.0)).unwrap();
        vf.fuse(20.0);
        assert!(vf.stats().anomaly_count >= 1);
    }

    // -- Venue stats -------------------------------------------------------

    #[test]
    fn test_venue_stats() {
        let mut vf = default_fusion();
        vf.ingest(&obs("binance", 100.0, 50000.0, 100.0)).unwrap();
        vf.ingest(&obs("binance", 200.0, 50100.0, 110.0)).unwrap();

        let vs = vf.venue_stats();
        let bs = &vs["binance"];
        assert_eq!(bs.observations, 2);
        assert!((bs.cumulative_volume - 300.0).abs() < 1e-12);
        assert!((bs.last_volume - 200.0).abs() < 1e-12);
        assert!((bs.last_vwap - 50100.0).abs() < 1e-12);
        assert!((bs.last_timestamp - 110.0).abs() < 1e-12);
    }

    #[test]
    fn test_venue_ema_share_updates() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        vf.ingest(&obs("b", 100.0, 100.0, 100.0)).unwrap();
        vf.fuse(100.0);

        let vs = vf.venue_stats();
        // Equal volume → each should have ~50% share
        assert!(
            (vs["a"].ema_share - 0.5).abs() < 0.01,
            "expected ~0.5 share for a, got {}",
            vs["a"].ema_share
        );
    }

    // -- Window eviction ---------------------------------------------------

    #[test]
    fn test_window_eviction() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            window_size: 3,
            ..default_config()
        });
        for i in 0..10 {
            vf.ingest(&obs("a", 100.0, 100.0, i as f64)).unwrap();
            vf.fuse(i as f64);
        }
        assert_eq!(vf.history.len(), 3);
    }

    // -- Zero volume -------------------------------------------------------

    #[test]
    fn test_zero_volume_ok() {
        let mut vf = default_fusion();
        vf.ingest(&obs("a", 0.0, 100.0, 100.0)).unwrap();
        let fused = vf.fuse(100.0).unwrap();
        assert!(fused.total_volume.abs() < 1e-12);
    }

    #[test]
    fn test_zero_volume_vwap_fallback() {
        let mut vf = default_fusion();
        // All venues have zero volume → VWAP should use simple average fallback
        vf.ingest(&obs("a", 0.0, 100.0, 100.0)).unwrap();
        vf.ingest(&obs("b", 0.0, 200.0, 100.0)).unwrap();
        let fused = vf.fuse(100.0).unwrap();
        assert!(
            (fused.vwap - 150.0).abs() < 0.01,
            "zero-volume VWAP should fallback to simple average, got {}",
            fused.vwap
        );
    }

    // -- Profile disabled --------------------------------------------------

    #[test]
    fn test_profile_disabled() {
        let mut vf = VolumeFusion::with_config(VolumeFusionConfig {
            profile_bucket_width: 0.0,
            ..default_config()
        });
        vf.ingest(&obs("a", 100.0, 100.0, 100.0)).unwrap();
        // Profile should have no volume
        assert!(vf.point_of_control().is_none());
    }
}
