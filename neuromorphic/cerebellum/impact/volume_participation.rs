//! Volume participation rate controller
//!
//! Part of the Cerebellum region
//! Component: impact
//!
//! Manages execution speed relative to observed market volume to control
//! market impact and regulatory risk. Enforces maximum participation rate
//! limits and provides optimal child-order slice sizing with fill-probability
//! weighting and adaptive rate adjustment from observed fill patterns.
//!
//! Key features:
//! - Real-time participation rate tracking against configurable limits
//! - Optimal slice sizing based on target participation and volume forecast
//! - Adaptive rate adjustment from observed market volume patterns
//! - Fill-probability-weighted scheduling across time buckets
//! - Volume forecast with EMA-smoothed rate estimation
//! - Burst detection and throttling
//! - Running statistics with compliance tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for the volume participation controller
#[derive(Debug, Clone)]
pub struct VolumeParticipationConfig {
    /// Target participation rate (fraction of market volume, e.g. 0.05 = 5%)
    pub target_rate: f64,
    /// Maximum allowed participation rate (hard limit, e.g. 0.10 = 10%)
    pub max_rate: f64,
    /// Warning threshold as fraction of max_rate (e.g. 0.80 = warn at 80% of max)
    pub warning_threshold: f64,
    /// EMA decay factor for smoothed volume rate estimation (0 < decay < 1)
    pub ema_decay: f64,
    /// Volume bucket duration in seconds (how often to snapshot market volume)
    pub bucket_duration_secs: f64,
    /// Number of volume buckets to keep for rate calculation
    pub num_buckets: usize,
    /// Minimum market volume per bucket to consider valid (avoids div-by-zero in thin markets)
    pub min_bucket_volume: f64,
    /// Minimum number of completed buckets before rate estimates are reliable
    pub min_buckets: usize,
    /// Maximum burst ratio: how much the instantaneous rate can exceed target before throttling
    pub max_burst_ratio: f64,
    /// Maximum number of observations in the sliding window
    pub window_size: usize,
    /// Minimum slice size (base currency units); orders below this are not worthwhile
    pub min_slice_size: f64,
    /// Maximum slice size (base currency units); safety cap on single child order
    pub max_slice_size: f64,
}

impl Default for VolumeParticipationConfig {
    fn default() -> Self {
        Self {
            target_rate: 0.05,
            max_rate: 0.10,
            warning_threshold: 0.80,
            ema_decay: 0.92,
            bucket_duration_secs: 60.0,
            num_buckets: 30,
            min_bucket_volume: 1.0,
            min_buckets: 3,
            max_burst_ratio: 2.0,
            window_size: 500,
            min_slice_size: 0.01,
            max_slice_size: 1_000_000.0,
        }
    }
}

/// A volume snapshot for a completed time bucket
#[derive(Debug, Clone)]
pub struct VolumeBucket {
    /// Total market volume observed in this bucket (base currency)
    pub market_volume: f64,
    /// Our executed volume in this bucket (base currency)
    pub our_volume: f64,
    /// Bucket start timestamp (seconds since epoch or arbitrary reference)
    pub start_time: f64,
    /// Bucket end timestamp
    pub end_time: f64,
}

impl VolumeBucket {
    /// Participation rate for this bucket
    pub fn participation_rate(&self) -> f64 {
        if self.market_volume <= 0.0 {
            return 0.0;
        }
        self.our_volume / self.market_volume
    }
}

/// Compliance status for the participation rate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceStatus {
    /// Rate is within target — normal operation
    Normal,
    /// Rate is above warning threshold but below hard limit
    Warning,
    /// Rate is at or above the hard limit — must throttle
    Breach,
    /// Insufficient data to determine compliance
    Unknown,
}

/// Result of a slice-sizing calculation
#[derive(Debug, Clone)]
pub struct SliceRecommendation {
    /// Recommended slice size (base currency units)
    pub slice_size: f64,
    /// Expected participation rate if this slice is fully executed
    pub expected_rate: f64,
    /// Whether this recommendation was throttled down from the ideal size
    pub throttled: bool,
    /// Current compliance status
    pub compliance: ComplianceStatus,
    /// Headroom: how much more volume we can execute before hitting the limit
    pub headroom: f64,
    /// Confidence in the recommendation (based on volume forecast quality)
    pub confidence: f64,
    /// Forecasted market volume for the next bucket (base currency)
    pub volume_forecast: f64,
}

/// Record of a volume update for windowed analysis
#[derive(Debug, Clone)]
struct VolumeRecord {
    market_volume: f64,
    _our_volume: f64,
    rate: f64,
    compliance: ComplianceStatus,
}

/// Running statistics for the volume participation controller
#[derive(Debug, Clone, Default)]
pub struct VolumeParticipationStats {
    /// Total volume buckets processed
    pub total_buckets: usize,
    /// Total market volume observed (base currency)
    pub total_market_volume: f64,
    /// Total our executed volume (base currency)
    pub total_our_volume: f64,
    /// Number of buckets where participation exceeded the warning threshold
    pub warning_count: usize,
    /// Number of buckets where participation breached the hard limit
    pub breach_count: usize,
    /// Number of throttle events (slice was reduced)
    pub throttle_count: usize,
    /// Sum of participation rates across all valid buckets (for mean calculation)
    pub sum_rates: f64,
    /// Sum of squared rates (for variance calculation)
    pub sum_sq_rates: f64,
    /// Count of valid buckets (with sufficient market volume)
    pub valid_bucket_count: usize,
    /// Maximum instantaneous participation rate observed
    pub peak_rate: f64,
    /// Maximum market volume in a single bucket
    pub peak_market_volume: f64,
    /// Minimum market volume in a valid bucket
    pub min_market_volume: f64,
    /// Number of slice recommendations made
    pub recommendations_made: usize,
    /// Number of times a burst was detected
    pub burst_detections: usize,
}

impl VolumeParticipationStats {
    /// Overall participation rate across all time
    pub fn overall_rate(&self) -> f64 {
        if self.total_market_volume <= 0.0 {
            return 0.0;
        }
        self.total_our_volume / self.total_market_volume
    }

    /// Mean participation rate across valid buckets
    pub fn mean_rate(&self) -> f64 {
        if self.valid_bucket_count == 0 {
            return 0.0;
        }
        self.sum_rates / self.valid_bucket_count as f64
    }

    /// Variance of participation rate
    pub fn rate_variance(&self) -> f64 {
        if self.valid_bucket_count < 2 {
            return 0.0;
        }
        let mean = self.mean_rate();
        let var = self.sum_sq_rates / self.valid_bucket_count as f64 - mean * mean;
        var.max(0.0)
    }

    /// Standard deviation of participation rate
    pub fn rate_std(&self) -> f64 {
        self.rate_variance().sqrt()
    }

    /// Compliance rate: fraction of valid buckets that were within limits
    pub fn compliance_rate(&self) -> f64 {
        if self.valid_bucket_count == 0 {
            return 1.0;
        }
        let compliant = self.valid_bucket_count - self.breach_count;
        compliant as f64 / self.valid_bucket_count as f64
    }

    /// Warning rate: fraction of valid buckets that triggered warnings
    pub fn warning_rate(&self) -> f64 {
        if self.valid_bucket_count == 0 {
            return 0.0;
        }
        self.warning_count as f64 / self.valid_bucket_count as f64
    }

    /// Breach rate: fraction of valid buckets that breached the limit
    pub fn breach_rate(&self) -> f64 {
        if self.valid_bucket_count == 0 {
            return 0.0;
        }
        self.breach_count as f64 / self.valid_bucket_count as f64
    }

    /// Mean market volume per bucket
    pub fn mean_market_volume(&self) -> f64 {
        if self.valid_bucket_count == 0 {
            return 0.0;
        }
        self.total_market_volume / self.valid_bucket_count as f64
    }
}

/// Volume participation rate controller
///
/// Tracks execution speed relative to market volume, enforces participation
/// rate limits, and provides optimal child-order slice sizing to achieve
/// a target participation rate without breaching regulatory constraints.
pub struct VolumeParticipation {
    config: VolumeParticipationConfig,
    /// Recent completed volume buckets (ring buffer)
    buckets: VecDeque<VolumeBucket>,
    /// EMA-smoothed market volume rate (volume per second)
    ema_volume_rate: f64,
    /// Whether EMA has been initialized
    ema_initialized: bool,
    /// EMA-smoothed participation rate
    ema_participation_rate: f64,
    /// Whether participation rate EMA has been initialized
    participation_ema_initialized: bool,
    /// Cumulative our volume in the current (incomplete) bucket
    current_our_volume: f64,
    /// Cumulative market volume in the current (incomplete) bucket
    current_market_volume: f64,
    /// Start time of the current bucket
    current_bucket_start: f64,
    /// Whether the current bucket has been started
    bucket_started: bool,
    /// Recent volume records for windowed analysis
    recent: VecDeque<VolumeRecord>,
    /// Running statistics
    stats: VolumeParticipationStats,
}

impl Default for VolumeParticipation {
    fn default() -> Self {
        Self::new()
    }
}

impl VolumeParticipation {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(VolumeParticipationConfig::default())
    }

    /// Create a new instance with the given configuration
    pub fn with_config(config: VolumeParticipationConfig) -> Self {
        let mut stats = VolumeParticipationStats::default();
        stats.min_market_volume = f64::MAX;

        Self {
            buckets: VecDeque::with_capacity(config.num_buckets),
            ema_volume_rate: 0.0,
            ema_initialized: false,
            ema_participation_rate: 0.0,
            participation_ema_initialized: false,
            current_our_volume: 0.0,
            current_market_volume: 0.0,
            current_bucket_start: 0.0,
            bucket_started: false,
            recent: VecDeque::new(),
            stats,
            config,
        }
    }

    /// Main processing function — validates configuration
    pub fn process(&self) -> Result<()> {
        if self.config.target_rate <= 0.0 || self.config.target_rate > 1.0 {
            return Err(Error::InvalidInput("target_rate must be in (0, 1]".into()));
        }
        if self.config.max_rate <= 0.0 || self.config.max_rate > 1.0 {
            return Err(Error::InvalidInput("max_rate must be in (0, 1]".into()));
        }
        if self.config.max_rate < self.config.target_rate {
            return Err(Error::InvalidInput(
                "max_rate must be >= target_rate".into(),
            ));
        }
        if self.config.warning_threshold <= 0.0 || self.config.warning_threshold >= 1.0 {
            return Err(Error::InvalidInput(
                "warning_threshold must be in (0, 1)".into(),
            ));
        }
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.bucket_duration_secs <= 0.0 {
            return Err(Error::InvalidInput(
                "bucket_duration_secs must be > 0".into(),
            ));
        }
        if self.config.num_buckets == 0 {
            return Err(Error::InvalidInput("num_buckets must be > 0".into()));
        }
        if self.config.min_bucket_volume < 0.0 {
            return Err(Error::InvalidInput("min_bucket_volume must be >= 0".into()));
        }
        if self.config.max_burst_ratio < 1.0 {
            return Err(Error::InvalidInput("max_burst_ratio must be >= 1.0".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.min_slice_size < 0.0 {
            return Err(Error::InvalidInput("min_slice_size must be >= 0".into()));
        }
        if self.config.max_slice_size <= 0.0 {
            return Err(Error::InvalidInput("max_slice_size must be > 0".into()));
        }
        if self.config.max_slice_size < self.config.min_slice_size {
            return Err(Error::InvalidInput(
                "max_slice_size must be >= min_slice_size".into(),
            ));
        }
        Ok(())
    }

    /// Record a completed volume bucket
    ///
    /// Call this at the end of each time bucket with the observed market
    /// volume and our executed volume for that period.
    pub fn record_bucket(&mut self, bucket: VolumeBucket) -> Result<()> {
        if bucket.market_volume < 0.0 {
            return Err(Error::InvalidInput("market_volume must be >= 0".into()));
        }
        if bucket.our_volume < 0.0 {
            return Err(Error::InvalidInput("our_volume must be >= 0".into()));
        }
        if bucket.end_time < bucket.start_time {
            return Err(Error::InvalidInput("end_time must be >= start_time".into()));
        }

        let duration = bucket.end_time - bucket.start_time;
        let rate = bucket.participation_rate();
        let is_valid = bucket.market_volume >= self.config.min_bucket_volume;

        // Update EMA of volume rate
        if duration > 0.0 {
            let volume_rate = bucket.market_volume / duration;
            if self.ema_initialized {
                self.ema_volume_rate = self.config.ema_decay * self.ema_volume_rate
                    + (1.0 - self.config.ema_decay) * volume_rate;
            } else {
                self.ema_volume_rate = volume_rate;
                self.ema_initialized = true;
            }
        }

        // Update EMA of participation rate
        if is_valid {
            if self.participation_ema_initialized {
                self.ema_participation_rate = self.config.ema_decay * self.ema_participation_rate
                    + (1.0 - self.config.ema_decay) * rate;
            } else {
                self.ema_participation_rate = rate;
                self.participation_ema_initialized = true;
            }
        }

        // Determine compliance
        let compliance = if !is_valid {
            ComplianceStatus::Unknown
        } else if rate >= self.config.max_rate {
            ComplianceStatus::Breach
        } else if rate >= self.config.max_rate * self.config.warning_threshold {
            ComplianceStatus::Warning
        } else {
            ComplianceStatus::Normal
        };

        // Update stats
        self.stats.total_buckets += 1;
        self.stats.total_market_volume += bucket.market_volume;
        self.stats.total_our_volume += bucket.our_volume;

        if is_valid {
            self.stats.valid_bucket_count += 1;
            self.stats.sum_rates += rate;
            self.stats.sum_sq_rates += rate * rate;

            if rate > self.stats.peak_rate {
                self.stats.peak_rate = rate;
            }
            if bucket.market_volume > self.stats.peak_market_volume {
                self.stats.peak_market_volume = bucket.market_volume;
            }
            if bucket.market_volume < self.stats.min_market_volume {
                self.stats.min_market_volume = bucket.market_volume;
            }

            match compliance {
                ComplianceStatus::Warning => self.stats.warning_count += 1,
                ComplianceStatus::Breach => self.stats.breach_count += 1,
                _ => {}
            }
        }

        // Add to ring buffer
        self.buckets.push_back(bucket);
        while self.buckets.len() > self.config.num_buckets {
            self.buckets.pop_front();
        }

        // Add to windowed records
        let record = VolumeRecord {
            market_volume: self.buckets.back().unwrap().market_volume,
            _our_volume: self.buckets.back().unwrap().our_volume,
            rate,
            compliance,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        Ok(())
    }

    /// Update the current in-progress bucket with incremental volume
    ///
    /// Call this as trades occur within the current time bucket. When the
    /// bucket is complete, call `record_bucket` with the final totals.
    pub fn update_current(
        &mut self,
        market_volume_delta: f64,
        our_volume_delta: f64,
        timestamp: f64,
    ) {
        if !self.bucket_started {
            self.current_bucket_start = timestamp;
            self.bucket_started = true;
        }
        self.current_market_volume += market_volume_delta.max(0.0);
        self.current_our_volume += our_volume_delta.max(0.0);
    }

    /// Flush the current in-progress bucket as a completed bucket
    pub fn flush_current_bucket(&mut self, end_time: f64) -> Result<()> {
        if !self.bucket_started {
            return Ok(()); // Nothing to flush
        }

        let bucket = VolumeBucket {
            market_volume: self.current_market_volume,
            our_volume: self.current_our_volume,
            start_time: self.current_bucket_start,
            end_time,
        };

        self.current_market_volume = 0.0;
        self.current_our_volume = 0.0;
        self.bucket_started = false;

        self.record_bucket(bucket)
    }

    /// Get the recommended slice size for the next child order
    ///
    /// `remaining_quantity` is the total remaining quantity to execute.
    /// The recommendation respects the target participation rate and
    /// hard limits.
    pub fn recommend_slice(&mut self, remaining_quantity: f64) -> SliceRecommendation {
        self.stats.recommendations_made += 1;

        // Forecast market volume for the next bucket
        let volume_forecast = self.forecast_bucket_volume();
        let confidence = self.compute_confidence();

        // If we don't have enough data, return a conservative default
        if !self.ema_initialized || self.stats.valid_bucket_count < self.config.min_buckets {
            let default_slice = remaining_quantity
                .min(self.config.max_slice_size)
                .max(self.config.min_slice_size)
                .min(remaining_quantity);

            return SliceRecommendation {
                slice_size: default_slice,
                expected_rate: 0.0,
                throttled: false,
                compliance: ComplianceStatus::Unknown,
                headroom: f64::MAX,
                confidence: 0.0,
                volume_forecast,
            };
        }

        // Ideal slice: target_rate * forecasted market volume
        let ideal_slice = self.config.target_rate * volume_forecast;

        // Check burst: if current participation rate is already high, throttle
        let current_rate = self.current_participation_rate();
        let burst_throttle = if current_rate > self.config.target_rate * self.config.max_burst_ratio
        {
            self.stats.burst_detections += 1;
            0.5 // reduce to 50% of ideal
        } else if current_rate > self.config.target_rate {
            // Linearly reduce between target and burst limit
            let excess = current_rate - self.config.target_rate;
            let burst_range = self.config.target_rate * (self.config.max_burst_ratio - 1.0);
            if burst_range > 0.0 {
                1.0 - 0.5 * (excess / burst_range).min(1.0)
            } else {
                0.5
            }
        } else {
            1.0
        };

        let adjusted_slice = ideal_slice * burst_throttle;

        // Headroom: how much more we can execute before hitting max_rate
        let headroom = if volume_forecast > 0.0 {
            let max_our_volume = self.config.max_rate * volume_forecast;
            let current_in_bucket = self.current_our_volume;
            (max_our_volume - current_in_bucket).max(0.0)
        } else {
            0.0
        };

        // Don't exceed headroom
        let constrained_slice = adjusted_slice.min(headroom);
        let throttled = (constrained_slice - ideal_slice).abs() > 1e-10 || burst_throttle < 1.0;

        if throttled {
            self.stats.throttle_count += 1;
        }

        // Apply absolute bounds
        let final_slice = constrained_slice
            .min(remaining_quantity)
            .min(self.config.max_slice_size)
            .max(0.0);

        // Don't recommend below minimum unless it's all that's left
        let final_slice = if final_slice < self.config.min_slice_size
            && remaining_quantity >= self.config.min_slice_size
        {
            self.config.min_slice_size
        } else {
            final_slice
        };

        // Expected rate if this slice executes
        let expected_rate = if volume_forecast > 0.0 {
            (self.current_our_volume + final_slice) / volume_forecast
        } else {
            0.0
        };

        // Compliance check
        let compliance = if expected_rate >= self.config.max_rate {
            ComplianceStatus::Breach
        } else if expected_rate >= self.config.max_rate * self.config.warning_threshold {
            ComplianceStatus::Warning
        } else {
            ComplianceStatus::Normal
        };

        SliceRecommendation {
            slice_size: final_slice,
            expected_rate,
            throttled,
            compliance,
            headroom,
            confidence,
            volume_forecast,
        }
    }

    /// Forecast market volume for the next bucket
    ///
    /// Uses EMA-smoothed volume rate × bucket duration.
    pub fn forecast_bucket_volume(&self) -> f64 {
        if !self.ema_initialized {
            // Fallback: use mean of observed buckets
            if self.buckets.is_empty() {
                return 0.0;
            }
            let sum: f64 = self.buckets.iter().map(|b| b.market_volume).sum();
            return sum / self.buckets.len() as f64;
        }
        self.ema_volume_rate * self.config.bucket_duration_secs
    }

    /// Current instantaneous participation rate (from in-progress bucket + recent)
    pub fn current_participation_rate(&self) -> f64 {
        // Use EMA if available
        if self.participation_ema_initialized {
            return self.ema_participation_rate;
        }

        // Otherwise use current bucket
        if self.current_market_volume > self.config.min_bucket_volume {
            return self.current_our_volume / self.current_market_volume;
        }

        // Fall back to recent buckets
        self.recent_participation_rate()
    }

    /// Participation rate from recent completed buckets
    pub fn recent_participation_rate(&self) -> f64 {
        let valid_buckets: Vec<&VolumeBucket> = self
            .buckets
            .iter()
            .filter(|b| b.market_volume >= self.config.min_bucket_volume)
            .collect();

        if valid_buckets.is_empty() {
            return 0.0;
        }

        let total_market: f64 = valid_buckets.iter().map(|b| b.market_volume).sum();
        let total_ours: f64 = valid_buckets.iter().map(|b| b.our_volume).sum();

        if total_market <= 0.0 {
            return 0.0;
        }
        total_ours / total_market
    }

    /// Current compliance status
    pub fn compliance_status(&self) -> ComplianceStatus {
        let rate = self.current_participation_rate();
        if self.stats.valid_bucket_count < self.config.min_buckets {
            ComplianceStatus::Unknown
        } else if rate >= self.config.max_rate {
            ComplianceStatus::Breach
        } else if rate >= self.config.max_rate * self.config.warning_threshold {
            ComplianceStatus::Warning
        } else {
            ComplianceStatus::Normal
        }
    }

    /// Whether a burst is currently detected
    pub fn is_burst_detected(&self) -> bool {
        let rate = self.current_participation_rate();
        rate > self.config.target_rate * self.config.max_burst_ratio
    }

    /// EMA-smoothed volume rate (volume per second)
    pub fn smoothed_volume_rate(&self) -> f64 {
        if self.ema_initialized {
            self.ema_volume_rate
        } else {
            0.0
        }
    }

    /// EMA-smoothed participation rate
    pub fn smoothed_participation_rate(&self) -> f64 {
        if self.participation_ema_initialized {
            self.ema_participation_rate
        } else {
            0.0
        }
    }

    /// Whether there is enough data for reliable estimates
    pub fn has_sufficient_data(&self) -> bool {
        self.stats.valid_bucket_count >= self.config.min_buckets
    }

    /// Number of completed buckets in the ring buffer
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Total buckets ever processed
    pub fn total_buckets_processed(&self) -> usize {
        self.stats.total_buckets
    }

    /// Reference to running statistics
    pub fn stats(&self) -> &VolumeParticipationStats {
        &self.stats
    }

    /// Recent completed buckets
    pub fn recent_buckets(&self) -> &VecDeque<VolumeBucket> {
        &self.buckets
    }

    /// Windowed mean participation rate
    pub fn windowed_mean_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let valid: Vec<f64> = self
            .recent
            .iter()
            .filter(|r| r.market_volume > self.config.min_bucket_volume)
            .map(|r| r.rate)
            .collect();
        if valid.is_empty() {
            return 0.0;
        }
        valid.iter().sum::<f64>() / valid.len() as f64
    }

    /// Windowed compliance rate (fraction within limits)
    pub fn windowed_compliance_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 1.0;
        }
        let valid: Vec<&VolumeRecord> = self
            .recent
            .iter()
            .filter(|r| r.compliance != ComplianceStatus::Unknown)
            .collect();
        if valid.is_empty() {
            return 1.0;
        }
        let compliant = valid
            .iter()
            .filter(|r| {
                r.compliance == ComplianceStatus::Normal
                    || r.compliance == ComplianceStatus::Warning
            })
            .count();
        compliant as f64 / valid.len() as f64
    }

    /// Windowed breach count
    pub fn windowed_breach_count(&self) -> usize {
        self.recent
            .iter()
            .filter(|r| r.compliance == ComplianceStatus::Breach)
            .count()
    }

    /// Check if participation rate is trending upward (second half of window > first)
    pub fn is_rate_trending_up(&self) -> bool {
        let valid: Vec<f64> = self
            .recent
            .iter()
            .filter(|r| r.market_volume > self.config.min_bucket_volume)
            .map(|r| r.rate)
            .collect();

        let n = valid.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;

        let first_half_mean: f64 = valid.iter().take(mid).sum::<f64>() / mid as f64;
        let second_half_mean: f64 = valid.iter().skip(mid).sum::<f64>() / (n - mid) as f64;

        second_half_mean > first_half_mean * 1.15
    }

    /// Compute confidence in the volume forecast and rate estimates
    fn compute_confidence(&self) -> f64 {
        if self.stats.valid_bucket_count == 0 {
            return 0.0;
        }

        // Bucket confidence: ramps over min_buckets * 3
        let bucket_conf = (self.stats.valid_bucket_count as f64
            / (self.config.min_buckets as f64 * 3.0))
            .min(1.0);

        // Stability confidence: lower rate variance = higher confidence
        let stability_conf = if self.stats.valid_bucket_count > 1 {
            let cv = if self.stats.mean_rate() > 0.0 {
                self.stats.rate_std() / self.stats.mean_rate()
            } else {
                0.0
            };
            (1.0 - cv).clamp(0.0, 1.0)
        } else {
            0.5
        };

        (bucket_conf * stability_conf).sqrt()
    }

    /// Reset all state and statistics
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.ema_volume_rate = 0.0;
        self.ema_initialized = false;
        self.ema_participation_rate = 0.0;
        self.participation_ema_initialized = false;
        self.current_our_volume = 0.0;
        self.current_market_volume = 0.0;
        self.current_bucket_start = 0.0;
        self.bucket_started = false;
        self.recent.clear();
        self.stats = VolumeParticipationStats::default();
        self.stats.min_market_volume = f64::MAX;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_config() -> VolumeParticipationConfig {
        VolumeParticipationConfig {
            target_rate: 0.05,
            max_rate: 0.10,
            warning_threshold: 0.80,
            ema_decay: 0.5,
            bucket_duration_secs: 60.0,
            num_buckets: 10,
            min_bucket_volume: 1.0,
            min_buckets: 2,
            max_burst_ratio: 2.0,
            window_size: 50,
            min_slice_size: 0.01,
            max_slice_size: 10000.0,
        }
    }

    fn make_bucket(market: f64, ours: f64, start: f64, end: f64) -> VolumeBucket {
        VolumeBucket {
            market_volume: market,
            our_volume: ours,
            start_time: start,
            end_time: end,
        }
    }

    fn balanced_bucket(i: usize) -> VolumeBucket {
        // 5% participation rate (target)
        make_bucket(1000.0, 50.0, (i as f64) * 60.0, ((i + 1) as f64) * 60.0)
    }

    #[test]
    fn test_basic() {
        let instance = VolumeParticipation::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_empty_state() {
        let model = VolumeParticipation::with_config(simple_config());
        assert_eq!(model.bucket_count(), 0);
        assert_eq!(model.total_buckets_processed(), 0);
        assert_eq!(model.smoothed_volume_rate(), 0.0);
        assert_eq!(model.smoothed_participation_rate(), 0.0);
        assert!(!model.has_sufficient_data());
    }

    #[test]
    fn test_record_bucket() {
        let mut model = VolumeParticipation::with_config(simple_config());
        model.record_bucket(balanced_bucket(0)).unwrap();

        assert_eq!(model.bucket_count(), 1);
        assert_eq!(model.total_buckets_processed(), 1);
        assert!(model.smoothed_volume_rate() > 0.0);
    }

    #[test]
    fn test_participation_rate_calculated() {
        let bucket = make_bucket(1000.0, 50.0, 0.0, 60.0);
        assert!((bucket.participation_rate() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_participation_rate_zero_market() {
        let bucket = make_bucket(0.0, 50.0, 0.0, 60.0);
        assert_eq!(bucket.participation_rate(), 0.0);
    }

    #[test]
    fn test_compliance_normal() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Low participation rate
        for i in 0..5 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    10.0,
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }

        assert_eq!(model.compliance_status(), ComplianceStatus::Normal);
    }

    #[test]
    fn test_compliance_warning() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Rate at 85% of max (0.085), above warning threshold (0.08)
        for i in 0..5 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    85.0,
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }

        assert_eq!(model.compliance_status(), ComplianceStatus::Warning);
    }

    #[test]
    fn test_compliance_breach() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Rate at 15% (above max 10%)
        for i in 0..5 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    150.0,
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }

        assert_eq!(model.compliance_status(), ComplianceStatus::Breach);
    }

    #[test]
    fn test_compliance_unknown_insufficient_data() {
        let model = VolumeParticipation::with_config(simple_config());
        assert_eq!(model.compliance_status(), ComplianceStatus::Unknown);
    }

    #[test]
    fn test_volume_forecast() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Record buckets with market volume = 1000 over 60 seconds
        // Volume rate = 1000/60 ≈ 16.67 per second
        // Forecast for 60s bucket ≈ 1000
        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let forecast = model.forecast_bucket_volume();
        assert!(
            (forecast - 1000.0).abs() < 200.0,
            "forecast should be near 1000, got {}",
            forecast
        );
    }

    #[test]
    fn test_volume_forecast_empty() {
        let model = VolumeParticipation::with_config(simple_config());
        assert_eq!(model.forecast_bucket_volume(), 0.0);
    }

    #[test]
    fn test_recommend_slice_basic() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Build up data
        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rec = model.recommend_slice(10000.0);
        assert!(rec.slice_size > 0.0, "slice should be > 0");
        assert!(
            rec.slice_size <= 10000.0,
            "slice should not exceed remaining: {}",
            rec.slice_size
        );
        assert!(
            rec.slice_size <= model.config.max_slice_size,
            "slice should not exceed max_slice_size"
        );
    }

    #[test]
    fn test_recommend_slice_respects_target_rate() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rec = model.recommend_slice(100000.0);

        // Slice should be approximately target_rate * forecast_volume
        let expected_slice = model.config.target_rate * model.forecast_bucket_volume();
        assert!(
            (rec.slice_size - expected_slice).abs() < expected_slice * 0.5,
            "slice {} should be near {} (target * forecast)",
            rec.slice_size,
            expected_slice
        );
    }

    #[test]
    fn test_recommend_slice_insufficient_data() {
        let mut model = VolumeParticipation::with_config(simple_config());

        let rec = model.recommend_slice(1000.0);
        assert_eq!(rec.confidence, 0.0);
        assert_eq!(rec.compliance, ComplianceStatus::Unknown);
    }

    #[test]
    fn test_recommend_slice_respects_remaining() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let small_remaining = 1.0;
        let rec = model.recommend_slice(small_remaining);
        assert!(
            rec.slice_size <= small_remaining + 1e-10,
            "slice {} should not exceed remaining {}",
            rec.slice_size,
            small_remaining
        );
    }

    #[test]
    fn test_recommend_slice_min_slice() {
        let mut model = VolumeParticipation::with_config(VolumeParticipationConfig {
            min_slice_size: 10.0,
            ..simple_config()
        });

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rec = model.recommend_slice(100.0);
        assert!(
            rec.slice_size >= 10.0 || rec.slice_size <= 0.0,
            "slice should be >= min or 0: {}",
            rec.slice_size
        );
    }

    #[test]
    fn test_recommend_slice_throttled_when_rate_high() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Record buckets with high participation
        for i in 0..5 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    150.0, // 15% rate > target 5%
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }

        let rec = model.recommend_slice(100000.0);
        assert!(
            rec.throttled,
            "should be throttled when current rate is high"
        );
    }

    #[test]
    fn test_burst_detection() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Record buckets with rate = 15% (3× target of 5%, above burst ratio of 2×)
        for i in 0..5 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    150.0,
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }

        assert!(
            model.is_burst_detected(),
            "should detect burst at 15% vs target 5% with burst ratio 2"
        );
    }

    #[test]
    fn test_no_burst_at_target() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        assert!(
            !model.is_burst_detected(),
            "should not detect burst at target rate"
        );
    }

    #[test]
    fn test_update_current() {
        let mut model = VolumeParticipation::with_config(simple_config());

        model.update_current(100.0, 5.0, 0.0);
        model.update_current(100.0, 5.0, 30.0);

        assert!((model.current_market_volume - 200.0).abs() < 1e-10);
        assert!((model.current_our_volume - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_flush_current_bucket() {
        let mut model = VolumeParticipation::with_config(simple_config());

        model.update_current(1000.0, 50.0, 0.0);
        model.flush_current_bucket(60.0).unwrap();

        assert_eq!(model.bucket_count(), 1);
        assert!((model.current_market_volume - 0.0).abs() < 1e-10);
        assert!((model.current_our_volume - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_flush_empty_is_noop() {
        let mut model = VolumeParticipation::with_config(simple_config());
        model.flush_current_bucket(60.0).unwrap();
        assert_eq!(model.bucket_count(), 0);
    }

    #[test]
    fn test_bucket_ring_buffer_capped() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..20 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        assert_eq!(model.bucket_count(), 10); // num_buckets = 10
        assert_eq!(model.total_buckets_processed(), 20);
    }

    #[test]
    fn test_stats_overall_rate() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rate = model.stats().overall_rate();
        assert!(
            (rate - 0.05).abs() < 1e-10,
            "overall rate should be 0.05, got {}",
            rate
        );
    }

    #[test]
    fn test_stats_mean_rate() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        assert!(
            (model.stats().mean_rate() - 0.05).abs() < 1e-10,
            "mean rate should be 0.05, got {}",
            model.stats().mean_rate()
        );
    }

    #[test]
    fn test_stats_rate_variance_constant() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..10 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        assert!(
            model.stats().rate_variance() < 1e-10,
            "constant rates should have ~0 variance, got {}",
            model.stats().rate_variance()
        );
    }

    #[test]
    fn test_stats_compliance_rate() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // All within limits
        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        assert!(
            (model.stats().compliance_rate() - 1.0).abs() < 1e-10,
            "all compliant should give rate 1.0, got {}",
            model.stats().compliance_rate()
        );
    }

    #[test]
    fn test_stats_breach_tracked() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // One breach
        model
            .record_bucket(make_bucket(1000.0, 150.0, 0.0, 60.0))
            .unwrap();

        assert_eq!(model.stats().breach_count, 1);
    }

    #[test]
    fn test_stats_warning_tracked() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Rate at 85% of max = 0.085 (above warning threshold 0.08)
        model
            .record_bucket(make_bucket(1000.0, 85.0, 0.0, 60.0))
            .unwrap();

        assert_eq!(model.stats().warning_count, 1);
    }

    #[test]
    fn test_stats_peak_rate() {
        let mut model = VolumeParticipation::with_config(simple_config());

        model
            .record_bucket(make_bucket(1000.0, 50.0, 0.0, 60.0))
            .unwrap();
        model
            .record_bucket(make_bucket(1000.0, 100.0, 60.0, 120.0))
            .unwrap();
        model
            .record_bucket(make_bucket(1000.0, 30.0, 120.0, 180.0))
            .unwrap();

        assert!(
            (model.stats().peak_rate - 0.10).abs() < 1e-10,
            "peak rate should be 0.10, got {}",
            model.stats().peak_rate
        );
    }

    #[test]
    fn test_stats_defaults() {
        let stats = VolumeParticipationStats::default();
        assert_eq!(stats.overall_rate(), 0.0);
        assert_eq!(stats.mean_rate(), 0.0);
        assert_eq!(stats.rate_variance(), 0.0);
        assert_eq!(stats.rate_std(), 0.0);
        assert_eq!(stats.compliance_rate(), 1.0);
        assert_eq!(stats.warning_rate(), 0.0);
        assert_eq!(stats.breach_rate(), 0.0);
        assert_eq!(stats.mean_market_volume(), 0.0);
    }

    #[test]
    fn test_windowed_mean_rate() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let wm = model.windowed_mean_rate();
        assert!(
            (wm - 0.05).abs() < 1e-10,
            "windowed mean rate should be 0.05, got {}",
            wm
        );
    }

    #[test]
    fn test_windowed_compliance_rate() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        assert!(
            (model.windowed_compliance_rate() - 1.0).abs() < 1e-10,
            "all compliant should give windowed rate 1.0"
        );
    }

    #[test]
    fn test_windowed_breach_count() {
        let mut model = VolumeParticipation::with_config(simple_config());

        model.record_bucket(balanced_bucket(0)).unwrap();
        model
            .record_bucket(make_bucket(1000.0, 150.0, 60.0, 120.0))
            .unwrap();
        model.record_bucket(balanced_bucket(2)).unwrap();

        assert_eq!(model.windowed_breach_count(), 1);
    }

    #[test]
    fn test_is_rate_trending_up() {
        let mut model = VolumeParticipation::with_config(VolumeParticipationConfig {
            window_size: 20,
            ..simple_config()
        });

        // First half: low rate
        for i in 0..10 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    10.0,
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }
        // Second half: much higher rate
        for i in 10..20 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    80.0,
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }

        assert!(model.is_rate_trending_up());
    }

    #[test]
    fn test_not_trending_up_stable() {
        let mut model = VolumeParticipation::with_config(VolumeParticipationConfig {
            window_size: 20,
            ..simple_config()
        });

        for i in 0..20 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        assert!(!model.is_rate_trending_up());
    }

    #[test]
    fn test_not_trending_up_insufficient_data() {
        let mut model = VolumeParticipation::with_config(simple_config());
        for i in 0..4 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    100.0,
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }
        assert!(!model.is_rate_trending_up());
    }

    #[test]
    fn test_confidence_zero_no_data() {
        let mut model = VolumeParticipation::with_config(simple_config());
        let rec = model.recommend_slice(1000.0);
        assert_eq!(rec.confidence, 0.0);
    }

    #[test]
    fn test_confidence_increases_with_data() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..10 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rec = model.recommend_slice(1000.0);
        assert!(
            rec.confidence > 0.0,
            "confidence should be > 0 with data, got {}",
            rec.confidence
        );
    }

    #[test]
    fn test_headroom_decreases_with_current_volume() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rec1 = model.recommend_slice(100000.0);
        let headroom1 = rec1.headroom;

        // Simulate current bucket volume usage
        model.update_current(1000.0, 80.0, 300.0);

        let rec2 = model.recommend_slice(100000.0);
        let headroom2 = rec2.headroom;

        assert!(
            headroom2 < headroom1,
            "headroom should decrease as we execute: {} vs {}",
            headroom2,
            headroom1
        );
    }

    #[test]
    fn test_reset() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..10 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }
        model.update_current(100.0, 5.0, 600.0);

        assert!(model.bucket_count() > 0);
        assert!(model.total_buckets_processed() > 0);

        model.reset();

        assert_eq!(model.bucket_count(), 0);
        assert_eq!(model.total_buckets_processed(), 0);
        assert_eq!(model.smoothed_volume_rate(), 0.0);
        assert_eq!(model.smoothed_participation_rate(), 0.0);
        assert!(!model.has_sufficient_data());
        assert!((model.current_market_volume - 0.0).abs() < 1e-10);
        assert!((model.current_our_volume - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_negative_market_volume() {
        let mut model = VolumeParticipation::with_config(simple_config());
        assert!(
            model
                .record_bucket(make_bucket(-1.0, 0.0, 0.0, 60.0))
                .is_err()
        );
    }

    #[test]
    fn test_invalid_negative_our_volume() {
        let mut model = VolumeParticipation::with_config(simple_config());
        assert!(
            model
                .record_bucket(make_bucket(1000.0, -1.0, 0.0, 60.0))
                .is_err()
        );
    }

    #[test]
    fn test_invalid_end_before_start() {
        let mut model = VolumeParticipation::with_config(simple_config());
        assert!(
            model
                .record_bucket(make_bucket(1000.0, 50.0, 60.0, 0.0))
                .is_err()
        );
    }

    #[test]
    fn test_invalid_config_target_rate() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            target_rate: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_max_rate() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            max_rate: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_max_less_than_target() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            target_rate: 0.10,
            max_rate: 0.05,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_warning_threshold() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            warning_threshold: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_bucket_duration() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            bucket_duration_secs: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_num_buckets() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            num_buckets: 0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_burst_ratio() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            max_burst_ratio: 0.5,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_window_size() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_negative_min_slice() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            min_slice_size: -1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_max_slice_zero() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            max_slice_size: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_max_less_than_min_slice() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            min_slice_size: 100.0,
            max_slice_size: 10.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_negative_min_bucket_volume() {
        let model = VolumeParticipation::with_config(VolumeParticipationConfig {
            min_bucket_volume: -1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_recent_participation_rate() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rate = model.recent_participation_rate();
        assert!(
            (rate - 0.05).abs() < 1e-10,
            "recent rate should be 0.05, got {}",
            rate
        );
    }

    #[test]
    fn test_recent_participation_rate_empty() {
        let model = VolumeParticipation::with_config(simple_config());
        assert_eq!(model.recent_participation_rate(), 0.0);
    }

    #[test]
    fn test_has_sufficient_data() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Need min_buckets = 2
        model.record_bucket(balanced_bucket(0)).unwrap();
        assert!(!model.has_sufficient_data());

        model.record_bucket(balanced_bucket(1)).unwrap();
        assert!(model.has_sufficient_data());
    }

    #[test]
    fn test_recommendation_count_tracked() {
        let mut model = VolumeParticipation::with_config(simple_config());

        model.recommend_slice(1000.0);
        model.recommend_slice(1000.0);
        model.recommend_slice(1000.0);

        assert_eq!(model.stats().recommendations_made, 3);
    }

    #[test]
    fn test_ema_volume_rate_converges() {
        let mut model = VolumeParticipation::with_config(VolumeParticipationConfig {
            ema_decay: 0.5,
            ..simple_config()
        });

        // All buckets have rate = 1000/60 ≈ 16.67 per second
        for i in 0..20 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rate = model.smoothed_volume_rate();
        let expected = 1000.0 / 60.0;
        assert!(
            (rate - expected).abs() < 1.0,
            "EMA volume rate should converge to {}, got {}",
            expected,
            rate
        );
    }

    #[test]
    fn test_window_eviction() {
        let mut model = VolumeParticipation::with_config(VolumeParticipationConfig {
            window_size: 5,
            ..simple_config()
        });

        for i in 0..20 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        assert_eq!(model.recent.len(), 5);
    }

    #[test]
    fn test_windowed_mean_rate_empty() {
        let model = VolumeParticipation::with_config(simple_config());
        assert_eq!(model.windowed_mean_rate(), 0.0);
    }

    #[test]
    fn test_min_market_volume_filter() {
        let mut model = VolumeParticipation::with_config(VolumeParticipationConfig {
            min_bucket_volume: 100.0,
            ..simple_config()
        });

        // Sub-threshold volume bucket
        model
            .record_bucket(make_bucket(10.0, 5.0, 0.0, 60.0))
            .unwrap();

        assert_eq!(model.stats().valid_bucket_count, 0);
    }

    #[test]
    fn test_update_current_ignores_negatives() {
        let mut model = VolumeParticipation::with_config(simple_config());

        model.update_current(-100.0, -50.0, 0.0);

        assert!((model.current_market_volume - 0.0).abs() < 1e-10);
        assert!((model.current_our_volume - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_peak_market_volume_tracked() {
        let mut model = VolumeParticipation::with_config(simple_config());

        model
            .record_bucket(make_bucket(500.0, 25.0, 0.0, 60.0))
            .unwrap();
        model
            .record_bucket(make_bucket(2000.0, 100.0, 60.0, 120.0))
            .unwrap();
        model
            .record_bucket(make_bucket(800.0, 40.0, 120.0, 180.0))
            .unwrap();

        assert!(
            (model.stats().peak_market_volume - 2000.0).abs() < 1e-10,
            "peak market volume should be 2000, got {}",
            model.stats().peak_market_volume
        );
    }

    #[test]
    fn test_min_market_volume_tracked() {
        let mut model = VolumeParticipation::with_config(simple_config());

        model
            .record_bucket(make_bucket(500.0, 25.0, 0.0, 60.0))
            .unwrap();
        model
            .record_bucket(make_bucket(200.0, 10.0, 60.0, 120.0))
            .unwrap();
        model
            .record_bucket(make_bucket(800.0, 40.0, 120.0, 180.0))
            .unwrap();

        assert!(
            (model.stats().min_market_volume - 200.0).abs() < 1e-10,
            "min market volume should be 200, got {}",
            model.stats().min_market_volume
        );
    }

    #[test]
    fn test_slice_recommendation_volume_forecast() {
        let mut model = VolumeParticipation::with_config(simple_config());

        for i in 0..5 {
            model.record_bucket(balanced_bucket(i)).unwrap();
        }

        let rec = model.recommend_slice(100000.0);
        assert!(
            rec.volume_forecast > 0.0,
            "forecast should be > 0, got {}",
            rec.volume_forecast
        );
    }

    #[test]
    fn test_throttle_count_tracked() {
        let mut model = VolumeParticipation::with_config(simple_config());

        // Build high participation history so recommendations get throttled
        for i in 0..5 {
            model
                .record_bucket(make_bucket(
                    1000.0,
                    120.0,
                    i as f64 * 60.0,
                    (i + 1) as f64 * 60.0,
                ))
                .unwrap();
        }

        model.recommend_slice(100000.0);

        // At 12% rate (above target 5%), throttling should occur
        assert!(
            model.stats().throttle_count > 0,
            "should have throttled at least once"
        );
    }
}
