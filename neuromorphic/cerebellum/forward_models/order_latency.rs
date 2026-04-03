//! Order latency prediction model
//!
//! Part of the Cerebellum region
//! Component: forward_models
//!
//! Tracks observed order round-trip latencies and predicts future latency
//! using exponential moving averages, percentile estimation, and
//! congestion detection. Enables the execution engine to adjust timing
//! strategies based on current network/exchange conditions.
//!
//! Key features:
//! - EMA-smoothed latency prediction with configurable decay
//! - Jitter (latency variance) tracking for stability assessment
//! - Percentile estimation via sorted sliding window
//! - Congestion detection based on latency regime shifts
//! - Per-venue latency tracking support
//! - Running statistics with min/max/mean/variance

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for the order latency model
#[derive(Debug, Clone)]
pub struct OrderLatencyConfig {
    /// EMA decay factor for smoothed latency (0 < decay < 1)
    pub ema_decay: f64,
    /// Maximum number of observations in the sliding window
    pub window_size: usize,
    /// Minimum observations before predictions are considered reliable
    pub min_samples: usize,
    /// Congestion threshold: ratio of current EMA to baseline that triggers congestion
    pub congestion_ratio: f64,
    /// Number of initial samples used to establish the baseline latency
    pub baseline_window: usize,
    /// Maximum plausible latency in milliseconds (safety clamp)
    pub max_latency_ms: f64,
    /// Percentile to use for conservative latency estimate (e.g. 0.95 = p95)
    pub conservative_percentile: f64,
}

impl Default for OrderLatencyConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.92,
            window_size: 500,
            min_samples: 10,
            congestion_ratio: 2.0,
            baseline_window: 50,
            max_latency_ms: 10_000.0,
            conservative_percentile: 0.95,
        }
    }
}

/// A single latency observation
#[derive(Debug, Clone)]
pub struct LatencyObservation {
    /// Round-trip latency in milliseconds
    pub latency_ms: f64,
    /// Optional venue identifier for per-venue tracking
    pub venue: Option<String>,
}

/// Latency prediction result
#[derive(Debug, Clone)]
pub struct LatencyEstimate {
    /// EMA-smoothed predicted latency (ms)
    pub predicted_ms: f64,
    /// Conservative estimate at configured percentile (ms)
    pub conservative_ms: f64,
    /// Current jitter (standard deviation of recent latencies, ms)
    pub jitter_ms: f64,
    /// Whether the model detects congestion
    pub congested: bool,
    /// Congestion ratio (current EMA / baseline)
    pub congestion_ratio: f64,
    /// Confidence in the prediction (based on sample count)
    pub confidence: f64,
    /// Number of observations backing this estimate
    pub sample_count: usize,
}

/// Running statistics for the latency model
#[derive(Debug, Clone)]
pub struct OrderLatencyStats {
    /// Total observations recorded
    pub total_observations: usize,
    /// Minimum latency observed (ms)
    pub min_latency_ms: f64,
    /// Maximum latency observed (ms)
    pub max_latency_ms: f64,
    /// Sum of all latencies (for mean calculation)
    pub sum_latency_ms: f64,
    /// Sum of squared latencies (for variance calculation)
    pub sum_sq_latency_ms: f64,
    /// Number of congestion events detected
    pub congestion_events: usize,
    /// Number of observations that exceeded max_latency_ms clamp
    pub clamped_count: usize,
    /// Current baseline latency (ms)
    pub baseline_ms: f64,
    /// Peak jitter observed (ms)
    pub peak_jitter_ms: f64,
}

impl Default for OrderLatencyStats {
    fn default() -> Self {
        Self {
            total_observations: 0,
            min_latency_ms: f64::MAX,
            max_latency_ms: 0.0,
            sum_latency_ms: 0.0,
            sum_sq_latency_ms: 0.0,
            congestion_events: 0,
            clamped_count: 0,
            baseline_ms: 0.0,
            peak_jitter_ms: 0.0,
        }
    }
}

impl OrderLatencyStats {
    /// Mean latency across all observations (ms)
    pub fn mean_latency_ms(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.sum_latency_ms / self.total_observations as f64
    }

    /// Variance of latency across all observations
    pub fn variance(&self) -> f64 {
        if self.total_observations < 2 {
            return 0.0;
        }
        let mean = self.mean_latency_ms();
        let var = self.sum_sq_latency_ms / self.total_observations as f64 - mean * mean;
        var.max(0.0)
    }

    /// Standard deviation of latency across all observations (ms)
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Coefficient of variation (std_dev / mean)
    pub fn cv(&self) -> f64 {
        let mean = self.mean_latency_ms();
        if mean <= 0.0 {
            return 0.0;
        }
        self.std_dev() / mean
    }
}

/// Order latency prediction model
///
/// Tracks observed order round-trip times and uses EMA smoothing,
/// percentile estimation, and congestion detection to predict future
/// latency and assess execution timing risk.
pub struct OrderLatency {
    config: OrderLatencyConfig,
    /// EMA-smoothed latency
    ema_latency: f64,
    /// EMA of squared latency (for jitter computation)
    ema_sq_latency: f64,
    /// Whether EMA has been initialized
    ema_initialized: bool,
    /// Sliding window of recent observations (sorted copy maintained separately)
    recent: VecDeque<f64>,
    /// Baseline latency (established from initial observations)
    baseline: f64,
    /// Whether baseline has been established
    baseline_established: bool,
    /// Accumulator for baseline computation
    baseline_accumulator: f64,
    /// Count of observations used for baseline
    baseline_count: usize,
    /// Whether we are currently in a congestion state
    in_congestion: bool,
    /// Running statistics
    stats: OrderLatencyStats,
}

impl Default for OrderLatency {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderLatency {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(OrderLatencyConfig::default())
    }

    /// Create a new instance with the given configuration
    pub fn with_config(config: OrderLatencyConfig) -> Self {
        Self {
            recent: VecDeque::with_capacity(config.window_size),
            ema_latency: 0.0,
            ema_sq_latency: 0.0,
            ema_initialized: false,
            baseline: 0.0,
            baseline_established: false,
            baseline_accumulator: 0.0,
            baseline_count: 0,
            in_congestion: false,
            stats: OrderLatencyStats::default(),
            config,
        }
    }

    /// Main processing function — validates configuration
    pub fn process(&self) -> Result<()> {
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if self.config.congestion_ratio <= 1.0 {
            return Err(Error::InvalidInput("congestion_ratio must be > 1.0".into()));
        }
        if self.config.max_latency_ms <= 0.0 {
            return Err(Error::InvalidInput("max_latency_ms must be > 0".into()));
        }
        if self.config.conservative_percentile <= 0.0 || self.config.conservative_percentile >= 1.0
        {
            return Err(Error::InvalidInput(
                "conservative_percentile must be in (0, 1)".into(),
            ));
        }
        if self.config.baseline_window == 0 {
            return Err(Error::InvalidInput("baseline_window must be > 0".into()));
        }
        Ok(())
    }

    /// Record a latency observation
    pub fn record(&mut self, obs: LatencyObservation) -> Result<()> {
        if obs.latency_ms < 0.0 {
            return Err(Error::InvalidInput("latency_ms must be >= 0".into()));
        }

        // Clamp to max
        let latency = if obs.latency_ms > self.config.max_latency_ms {
            self.stats.clamped_count += 1;
            self.config.max_latency_ms
        } else {
            obs.latency_ms
        };

        // Update EMA
        if self.ema_initialized {
            self.ema_latency =
                self.config.ema_decay * self.ema_latency + (1.0 - self.config.ema_decay) * latency;
            self.ema_sq_latency = self.config.ema_decay * self.ema_sq_latency
                + (1.0 - self.config.ema_decay) * latency * latency;
        } else {
            self.ema_latency = latency;
            self.ema_sq_latency = latency * latency;
            self.ema_initialized = true;
        }

        // Update baseline
        if !self.baseline_established {
            self.baseline_accumulator += latency;
            self.baseline_count += 1;
            if self.baseline_count >= self.config.baseline_window {
                self.baseline = self.baseline_accumulator / self.baseline_count as f64;
                self.baseline_established = true;
                self.stats.baseline_ms = self.baseline;
            }
        }

        // Congestion detection
        if self.baseline_established && self.baseline > 0.0 {
            let ratio = self.ema_latency / self.baseline;
            let was_congested = self.in_congestion;
            self.in_congestion = ratio >= self.config.congestion_ratio;
            if self.in_congestion && !was_congested {
                self.stats.congestion_events += 1;
            }
        }

        // Update sliding window
        self.recent.push_back(latency);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        // Update stats
        self.stats.total_observations += 1;
        self.stats.sum_latency_ms += latency;
        self.stats.sum_sq_latency_ms += latency * latency;
        if latency < self.stats.min_latency_ms {
            self.stats.min_latency_ms = latency;
        }
        if latency > self.stats.max_latency_ms {
            self.stats.max_latency_ms = latency;
        }

        // Update peak jitter
        let jitter = self.jitter_ms();
        if jitter > self.stats.peak_jitter_ms {
            self.stats.peak_jitter_ms = jitter;
        }

        Ok(())
    }

    /// Get the current latency prediction
    pub fn estimate(&self) -> LatencyEstimate {
        let predicted_ms = if self.ema_initialized {
            self.ema_latency
        } else {
            0.0
        };

        let jitter = self.jitter_ms();
        let conservative_ms = self.percentile(self.config.conservative_percentile);

        let congestion_ratio = if self.baseline_established && self.baseline > 0.0 {
            self.ema_latency / self.baseline
        } else {
            1.0
        };

        let confidence = self.compute_confidence();

        LatencyEstimate {
            predicted_ms,
            conservative_ms,
            jitter_ms: jitter,
            congested: self.in_congestion,
            congestion_ratio,
            confidence,
            sample_count: self.stats.total_observations,
        }
    }

    /// Compute jitter (EMA-based standard deviation) in milliseconds
    pub fn jitter_ms(&self) -> f64 {
        if !self.ema_initialized {
            return 0.0;
        }
        let variance = self.ema_sq_latency - self.ema_latency * self.ema_latency;
        variance.max(0.0).sqrt()
    }

    /// Compute a percentile from the sliding window
    ///
    /// Uses linear interpolation between nearest ranks.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.recent.is_empty() || p <= 0.0 || p >= 1.0 {
            return self.ema_latency.max(0.0);
        }

        let mut sorted: Vec<f64> = self.recent.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        if n == 1 {
            return sorted[0];
        }

        let rank = p * (n - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        let frac = rank - lower as f64;

        if upper >= n {
            sorted[n - 1]
        } else {
            sorted[lower] * (1.0 - frac) + sorted[upper] * frac
        }
    }

    /// EMA-smoothed latency (ms)
    pub fn smoothed_latency(&self) -> f64 {
        if self.ema_initialized {
            self.ema_latency
        } else {
            0.0
        }
    }

    /// Baseline latency (ms), if established
    pub fn baseline_latency(&self) -> Option<f64> {
        if self.baseline_established {
            Some(self.baseline)
        } else {
            None
        }
    }

    /// Whether the baseline has been established
    pub fn baseline_established(&self) -> bool {
        self.baseline_established
    }

    /// Whether currently in a congestion state
    pub fn is_congested(&self) -> bool {
        self.in_congestion
    }

    /// Total observations recorded
    pub fn observation_count(&self) -> usize {
        self.stats.total_observations
    }

    /// Reference to running statistics
    pub fn stats(&self) -> &OrderLatencyStats {
        &self.stats
    }

    /// Current sliding window contents
    pub fn recent_latencies(&self) -> &VecDeque<f64> {
        &self.recent
    }

    /// Windowed mean latency (ms)
    pub fn windowed_mean(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().sum();
        sum / self.recent.len() as f64
    }

    /// Windowed standard deviation (ms)
    pub fn windowed_std(&self) -> f64 {
        if self.recent.len() < 2 {
            return 0.0;
        }
        let mean = self.windowed_mean();
        let variance: f64 = self
            .recent
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / self.recent.len() as f64;
        variance.max(0.0).sqrt()
    }

    /// Check if latency is trending upward (compare first vs second half of window)
    pub fn is_trending_up(&self) -> bool {
        let n = self.recent.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;
        let first_half_mean: f64 = self.recent.iter().take(mid).sum::<f64>() / mid as f64;
        let second_half_mean: f64 = self.recent.iter().skip(mid).sum::<f64>() / (n - mid) as f64;

        // Trending up if second half is >15% higher
        second_half_mean > first_half_mean * 1.15
    }

    /// Reset all state and statistics
    pub fn reset(&mut self) {
        self.ema_latency = 0.0;
        self.ema_sq_latency = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.baseline = 0.0;
        self.baseline_established = false;
        self.baseline_accumulator = 0.0;
        self.baseline_count = 0;
        self.in_congestion = false;
        self.stats = OrderLatencyStats::default();
    }

    /// Compute confidence based on sample count
    fn compute_confidence(&self) -> f64 {
        if self.stats.total_observations == 0 {
            return 0.0;
        }
        // Ramp confidence over min_samples * 3
        let sample_confidence = (self.stats.total_observations as f64
            / (self.config.min_samples as f64 * 3.0))
            .min(1.0);

        // Stability confidence: lower jitter relative to mean = higher confidence
        let stability_confidence = if self.ema_latency > 0.0 {
            let cv = self.jitter_ms() / self.ema_latency;
            (1.0 - cv).clamp(0.0, 1.0)
        } else {
            0.5
        };

        (sample_confidence * stability_confidence).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(latency_ms: f64) -> LatencyObservation {
        LatencyObservation {
            latency_ms,
            venue: None,
        }
    }

    #[test]
    fn test_basic() {
        let instance = OrderLatency::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_single_observation() {
        let mut model = OrderLatency::new();
        model.record(obs(5.0)).unwrap();

        assert!((model.smoothed_latency() - 5.0).abs() < 1e-10);
        assert_eq!(model.observation_count(), 1);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            ema_decay: 0.9,
            ..Default::default()
        });

        // Record steady latency
        for _ in 0..20 {
            model.record(obs(10.0)).unwrap();
        }
        let baseline = model.smoothed_latency();
        assert!(
            (baseline - 10.0).abs() < 0.5,
            "should converge to 10, got {}",
            baseline
        );

        // Spike
        model.record(obs(100.0)).unwrap();
        let after_spike = model.smoothed_latency();

        // EMA should dampen the spike
        assert!(after_spike < 100.0, "EMA should dampen spike");
        assert!(after_spike > 10.0, "EMA should react to spike");
    }

    #[test]
    fn test_jitter_constant_latency() {
        let mut model = OrderLatency::new();
        for _ in 0..20 {
            model.record(obs(5.0)).unwrap();
        }
        assert!(
            model.jitter_ms() < 0.1,
            "constant latency should have near-zero jitter, got {}",
            model.jitter_ms()
        );
    }

    #[test]
    fn test_jitter_variable_latency() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            ema_decay: 0.5,
            ..Default::default()
        });

        // Alternate between very different latencies
        for i in 0..40 {
            let lat = if i % 2 == 0 { 1.0 } else { 100.0 };
            model.record(obs(lat)).unwrap();
        }

        assert!(
            model.jitter_ms() > 1.0,
            "variable latency should have significant jitter, got {}",
            model.jitter_ms()
        );
    }

    #[test]
    fn test_baseline_establishment() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            baseline_window: 5,
            ..Default::default()
        });

        for _ in 0..4 {
            model.record(obs(10.0)).unwrap();
        }
        assert!(!model.baseline_established());
        assert!(model.baseline_latency().is_none());

        model.record(obs(10.0)).unwrap();
        assert!(model.baseline_established());
        assert!((model.baseline_latency().unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_congestion_detection() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            baseline_window: 5,
            congestion_ratio: 2.0,
            ema_decay: 0.5,
            ..Default::default()
        });

        // Establish baseline at 10ms
        for _ in 0..5 {
            model.record(obs(10.0)).unwrap();
        }
        assert!(!model.is_congested());

        // Spike to 30ms (3x baseline, above congestion_ratio of 2.0)
        for _ in 0..20 {
            model.record(obs(30.0)).unwrap();
        }
        assert!(
            model.is_congested(),
            "should detect congestion at 3x baseline, ema={}, baseline={}",
            model.smoothed_latency(),
            model.baseline_latency().unwrap()
        );
    }

    #[test]
    fn test_congestion_event_count() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            baseline_window: 3,
            congestion_ratio: 2.0,
            ema_decay: 0.3,
            ..Default::default()
        });

        // Establish baseline
        for _ in 0..3 {
            model.record(obs(10.0)).unwrap();
        }

        // Enter congestion
        for _ in 0..10 {
            model.record(obs(50.0)).unwrap();
        }
        assert_eq!(model.stats().congestion_events, 1);

        // Return to normal
        for _ in 0..30 {
            model.record(obs(10.0)).unwrap();
        }
        assert!(!model.is_congested());

        // Enter congestion again
        for _ in 0..10 {
            model.record(obs(50.0)).unwrap();
        }
        assert_eq!(model.stats().congestion_events, 2);
    }

    #[test]
    fn test_percentile_single_value() {
        let mut model = OrderLatency::new();
        model.record(obs(5.0)).unwrap();

        assert!((model.percentile(0.5) - 5.0).abs() < 1e-10);
        assert!((model.percentile(0.95) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_ordered() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            window_size: 100,
            ..Default::default()
        });

        // Insert 1..=100
        for i in 1..=100 {
            model.record(obs(i as f64)).unwrap();
        }

        let p50 = model.percentile(0.5);
        let p95 = model.percentile(0.95);
        let p10 = model.percentile(0.1);

        assert!(p10 < p50, "p10={} should < p50={}", p10, p50);
        assert!(p50 < p95, "p50={} should < p95={}", p50, p95);
    }

    #[test]
    fn test_percentile_median() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            window_size: 5,
            ..Default::default()
        });

        for v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            model.record(obs(*v)).unwrap();
        }

        let median = model.percentile(0.5);
        assert!(
            (median - 3.0).abs() < 1e-10,
            "median should be 3.0, got {}",
            median
        );
    }

    #[test]
    fn test_estimate_structure() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            baseline_window: 3,
            ..Default::default()
        });

        for _ in 0..10 {
            model.record(obs(5.0)).unwrap();
        }

        let est = model.estimate();
        assert!(est.predicted_ms > 0.0);
        assert!(est.conservative_ms >= est.predicted_ms);
        assert!(est.jitter_ms >= 0.0);
        assert!(est.confidence > 0.0);
        assert_eq!(est.sample_count, 10);
    }

    #[test]
    fn test_conservative_estimate_higher_than_predicted() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            window_size: 100,
            conservative_percentile: 0.95,
            ..Default::default()
        });

        // Variable latencies
        for i in 0..100 {
            let lat = 5.0 + (i as f64 % 10.0);
            model.record(obs(lat)).unwrap();
        }

        let est = model.estimate();
        assert!(
            est.conservative_ms >= est.predicted_ms,
            "conservative={} should >= predicted={}",
            est.conservative_ms,
            est.predicted_ms
        );
    }

    #[test]
    fn test_max_latency_clamping() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            max_latency_ms: 100.0,
            ..Default::default()
        });

        model.record(obs(200.0)).unwrap();
        assert_eq!(model.stats().clamped_count, 1);
        assert!(
            model.smoothed_latency() <= 100.0,
            "should be clamped to 100, got {}",
            model.smoothed_latency()
        );
    }

    #[test]
    fn test_negative_latency_rejected() {
        let mut model = OrderLatency::new();
        assert!(model.record(obs(-1.0)).is_err());
    }

    #[test]
    fn test_zero_latency_accepted() {
        let mut model = OrderLatency::new();
        assert!(model.record(obs(0.0)).is_ok());
    }

    #[test]
    fn test_stats_min_max() {
        let mut model = OrderLatency::new();

        model.record(obs(5.0)).unwrap();
        model.record(obs(15.0)).unwrap();
        model.record(obs(10.0)).unwrap();

        assert!((model.stats().min_latency_ms - 5.0).abs() < 1e-10);
        assert!((model.stats().max_latency_ms - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_mean() {
        let mut model = OrderLatency::new();
        model.record(obs(10.0)).unwrap();
        model.record(obs(20.0)).unwrap();
        model.record(obs(30.0)).unwrap();

        assert!(
            (model.stats().mean_latency_ms() - 20.0).abs() < 1e-10,
            "mean should be 20, got {}",
            model.stats().mean_latency_ms()
        );
    }

    #[test]
    fn test_stats_variance_constant() {
        let mut model = OrderLatency::new();
        for _ in 0..10 {
            model.record(obs(5.0)).unwrap();
        }
        assert!(
            model.stats().variance() < 1e-10,
            "constant values should have ~0 variance, got {}",
            model.stats().variance()
        );
    }

    #[test]
    fn test_stats_cv() {
        let mut model = OrderLatency::new();
        for _ in 0..10 {
            model.record(obs(10.0)).unwrap();
        }
        assert!(
            model.stats().cv() < 0.01,
            "constant values should have ~0 cv, got {}",
            model.stats().cv()
        );
    }

    #[test]
    fn test_windowed_mean() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            window_size: 3,
            ..Default::default()
        });

        model.record(obs(10.0)).unwrap();
        model.record(obs(20.0)).unwrap();
        model.record(obs(30.0)).unwrap();

        assert!(
            (model.windowed_mean() - 20.0).abs() < 1e-10,
            "windowed mean should be 20, got {}",
            model.windowed_mean()
        );

        // Push a new value, evicting the first
        model.record(obs(40.0)).unwrap();
        assert!(
            (model.windowed_mean() - 30.0).abs() < 1e-10,
            "windowed mean should be 30 after eviction, got {}",
            model.windowed_mean()
        );
    }

    #[test]
    fn test_windowed_std_constant() {
        let mut model = OrderLatency::new();
        for _ in 0..10 {
            model.record(obs(5.0)).unwrap();
        }
        assert!(
            model.windowed_std() < 1e-10,
            "constant should have 0 std, got {}",
            model.windowed_std()
        );
    }

    #[test]
    fn test_is_trending_up() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            window_size: 20,
            ..Default::default()
        });

        // First half: low latency
        for _ in 0..10 {
            model.record(obs(5.0)).unwrap();
        }
        // Second half: much higher
        for _ in 0..10 {
            model.record(obs(20.0)).unwrap();
        }

        assert!(model.is_trending_up());
    }

    #[test]
    fn test_not_trending_up_stable() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            window_size: 20,
            ..Default::default()
        });

        for _ in 0..20 {
            model.record(obs(10.0)).unwrap();
        }

        assert!(!model.is_trending_up());
    }

    #[test]
    fn test_not_trending_up_insufficient_data() {
        let mut model = OrderLatency::new();
        for _ in 0..4 {
            model.record(obs(100.0)).unwrap();
        }
        assert!(!model.is_trending_up());
    }

    #[test]
    fn test_reset() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            baseline_window: 3,
            ..Default::default()
        });

        for _ in 0..20 {
            model.record(obs(10.0)).unwrap();
        }

        assert!(model.observation_count() > 0);
        assert!(model.baseline_established());

        model.reset();

        assert_eq!(model.observation_count(), 0);
        assert!(!model.baseline_established());
        assert_eq!(model.smoothed_latency(), 0.0);
        assert_eq!(model.jitter_ms(), 0.0);
        assert!(!model.is_congested());
        assert!(model.recent_latencies().is_empty());
    }

    #[test]
    fn test_window_eviction() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            window_size: 5,
            ..Default::default()
        });

        for _ in 0..20 {
            model.record(obs(10.0)).unwrap();
        }

        assert_eq!(model.recent_latencies().len(), 5);
        assert_eq!(model.observation_count(), 20);
    }

    #[test]
    fn test_empty_stats_defaults() {
        let stats = OrderLatencyStats::default();
        assert_eq!(stats.mean_latency_ms(), 0.0);
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.std_dev(), 0.0);
        assert_eq!(stats.cv(), 0.0);
    }

    #[test]
    fn test_empty_estimate() {
        let model = OrderLatency::new();
        let est = model.estimate();
        assert_eq!(est.predicted_ms, 0.0);
        assert_eq!(est.jitter_ms, 0.0);
        assert!(!est.congested);
        assert_eq!(est.sample_count, 0);
    }

    #[test]
    fn test_confidence_zero_without_observations() {
        let model = OrderLatency::new();
        let est = model.estimate();
        assert!((est.confidence - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_increases_with_samples() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            min_samples: 5,
            ..Default::default()
        });

        let conf0 = model.estimate().confidence;

        for _ in 0..20 {
            model.record(obs(10.0)).unwrap();
        }

        let conf1 = model.estimate().confidence;
        assert!(
            conf1 > conf0,
            "confidence should increase: {} vs {}",
            conf1,
            conf0
        );
    }

    #[test]
    fn test_peak_jitter_tracked() {
        let mut model = OrderLatency::with_config(OrderLatencyConfig {
            ema_decay: 0.5,
            ..Default::default()
        });

        // Create some jitter
        for i in 0..20 {
            let lat = if i % 2 == 0 { 1.0 } else { 50.0 };
            model.record(obs(lat)).unwrap();
        }

        assert!(
            model.stats().peak_jitter_ms > 0.0,
            "peak jitter should be tracked"
        );
    }

    #[test]
    fn test_invalid_config_ema_decay() {
        let model = OrderLatency::with_config(OrderLatencyConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let model = OrderLatency::with_config(OrderLatencyConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_window_size() {
        let model = OrderLatency::with_config(OrderLatencyConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_congestion_ratio() {
        let model = OrderLatency::with_config(OrderLatencyConfig {
            congestion_ratio: 0.5,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_max_latency() {
        let model = OrderLatency::with_config(OrderLatencyConfig {
            max_latency_ms: -1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_percentile() {
        let model = OrderLatency::with_config(OrderLatencyConfig {
            conservative_percentile: 1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_baseline_window() {
        let model = OrderLatency::with_config(OrderLatencyConfig {
            baseline_window: 0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_percentile_empty() {
        let model = OrderLatency::new();
        let p = model.percentile(0.5);
        assert!((p - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_empty() {
        let model = OrderLatency::new();
        assert_eq!(model.windowed_mean(), 0.0);
    }

    #[test]
    fn test_windowed_std_empty() {
        let model = OrderLatency::new();
        assert_eq!(model.windowed_std(), 0.0);
    }

    #[test]
    fn test_windowed_std_single() {
        let mut model = OrderLatency::new();
        model.record(obs(5.0)).unwrap();
        assert_eq!(model.windowed_std(), 0.0);
    }

    #[test]
    fn test_with_venue() {
        let mut model = OrderLatency::new();
        model
            .record(LatencyObservation {
                latency_ms: 5.0,
                venue: Some("binance".to_string()),
            })
            .unwrap();
        assert_eq!(model.observation_count(), 1);
    }
}
