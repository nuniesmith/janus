//! # Latency Simulation Model
//!
//! Simulates network and exchange processing latency for realistic order
//! fill timing in backtesting and paper trading. Supports multiple
//! distribution models for latency sampling.
//!
//! # Models
//!
//! - **Fixed**: Constant latency (useful for testing).
//! - **Normal**: Gaussian-distributed latency with mean and std deviation.
//! - **Uniform**: Uniformly distributed between min and max.
//! - **LogNormal**: Log-normally distributed (heavy right tail, realistic).
//! - **Empirical**: Sampled from a histogram of observed latencies.
//! - **Composite**: Gateway latency + exchange processing + network jitter.
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_lob::latency::*;
//! use std::time::Duration;
//!
//! // Normal distribution: mean 5ms, stddev 2ms
//! let model = LatencyModel::normal(
//!     Duration::from_millis(5),
//!     Duration::from_millis(2),
//! );
//!
//! let latency = model.sample_latency();
//! println!("Simulated latency: {:?}", latency);
//!
//! // Composite model for realistic simulation
//! let model = LatencyModel::composite(
//!     Duration::from_millis(1),   // gateway
//!     Duration::from_micros(100), // exchange processing
//!     Duration::from_millis(3),   // network mean
//!     Duration::from_millis(1),   // network jitter stddev
//! );
//! ```

use rand::Rng;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tracing::trace;

// ---------------------------------------------------------------------------
// Global RNG state for deterministic sequencing when no thread_rng is desired.
// ---------------------------------------------------------------------------
static LATENCY_RNG_COUNTER: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Latency Estimate
// ---------------------------------------------------------------------------

/// A sampled latency estimate with component breakdown.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LatencyEstimate {
    /// Total end-to-end latency.
    pub total: Duration,
    /// Gateway / API serialisation latency component.
    pub gateway: Duration,
    /// Exchange matching engine processing latency.
    pub exchange: Duration,
    /// Network round-trip component.
    pub network: Duration,
}

impl LatencyEstimate {
    /// Create a simple estimate with only a total value.
    pub fn simple(total: Duration) -> Self {
        Self {
            total,
            gateway: Duration::ZERO,
            exchange: Duration::ZERO,
            network: total,
        }
    }

    /// Create a composite estimate from individual components.
    pub fn composite(gateway: Duration, exchange: Duration, network: Duration) -> Self {
        Self {
            total: gateway + exchange + network,
            gateway,
            exchange,
            network,
        }
    }

    /// Total latency in microseconds.
    pub fn total_micros(&self) -> u64 {
        self.total.as_micros() as u64
    }

    /// Total latency in milliseconds (f64).
    pub fn total_millis_f64(&self) -> f64 {
        self.total.as_secs_f64() * 1000.0
    }
}

impl fmt::Display for LatencyEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.gateway.is_zero() && self.exchange.is_zero() {
            write!(f, "{:.1}µs", self.total.as_micros())
        } else {
            write!(
                f,
                "{:.1}µs (gw: {}µs, ex: {}µs, net: {}µs)",
                self.total.as_micros(),
                self.gateway.as_micros(),
                self.exchange.as_micros(),
                self.network.as_micros(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Latency Statistics
// ---------------------------------------------------------------------------

/// Running statistics for observed latencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Number of samples collected.
    pub count: u64,
    /// Sum of all latencies (for computing mean).
    pub sum_micros: u64,
    /// Sum of squared latencies (for computing variance).
    pub sum_sq_micros: u128,
    /// Minimum observed latency.
    pub min: Duration,
    /// Maximum observed latency.
    pub max: Duration,
    /// Last sampled latency.
    pub last: Duration,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            count: 0,
            sum_micros: 0,
            sum_sq_micros: 0,
            min: Duration::from_secs(u64::MAX),
            max: Duration::ZERO,
            last: Duration::ZERO,
        }
    }
}

impl LatencyStats {
    /// Record a new latency sample.
    pub fn record(&mut self, latency: Duration) {
        let micros = latency.as_micros() as u64;
        self.count += 1;
        self.sum_micros += micros;
        self.sum_sq_micros += (micros as u128) * (micros as u128);
        if latency < self.min {
            self.min = latency;
        }
        if latency > self.max {
            self.max = latency;
        }
        self.last = latency;
    }

    /// Mean latency.
    pub fn mean(&self) -> Duration {
        if self.count == 0 {
            return Duration::ZERO;
        }
        Duration::from_micros(self.sum_micros / self.count)
    }

    /// Variance in microseconds squared.
    pub fn variance_micros(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.sum_micros as f64 / n;
        (self.sum_sq_micros as f64 / n) - (mean * mean)
    }

    /// Standard deviation.
    pub fn stddev(&self) -> Duration {
        let var = self.variance_micros();
        if var <= 0.0 {
            return Duration::ZERO;
        }
        Duration::from_micros(var.sqrt() as u64)
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.count == 0 {
            write!(f, "LatencyStats(no samples)")
        } else {
            write!(
                f,
                "LatencyStats(n={}, mean={}µs, stddev={}µs, min={}µs, max={}µs)",
                self.count,
                self.mean().as_micros(),
                self.stddev().as_micros(),
                self.min.as_micros(),
                self.max.as_micros(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Latency Model
// ---------------------------------------------------------------------------

/// Latency simulation model for order processing.
///
/// Each variant represents a different statistical model for generating
/// realistic latency samples. The model is applied to each fill to simulate
/// the delay between order submission and fill confirmation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyModel {
    /// Zero latency (instant fills). Useful for unit tests.
    Zero,

    /// Constant fixed latency.
    Fixed {
        /// The fixed latency duration.
        latency: Duration,
    },

    /// Normally distributed latency.
    ///
    /// Samples are clamped to `[0, mean + 5*stddev]` to avoid
    /// negative or absurdly large values.
    Normal {
        /// Mean latency.
        mean: Duration,
        /// Standard deviation.
        stddev: Duration,
    },

    /// Uniformly distributed latency between min and max.
    Uniform {
        /// Minimum latency.
        min: Duration,
        /// Maximum latency.
        max: Duration,
    },

    /// Log-normally distributed latency.
    ///
    /// The log of the latency (in microseconds) is normally distributed
    /// with the given `mu` (mean of log) and `sigma` (stddev of log).
    /// This produces a heavy right tail which is realistic for network latency.
    LogNormal {
        /// Mean of the log of latency in microseconds.
        mu: f64,
        /// Standard deviation of the log of latency in microseconds.
        sigma: f64,
    },

    /// Empirical distribution: sample from a histogram of observed latencies.
    ///
    /// Buckets are `(latency, cumulative_probability)` pairs, sorted by
    /// latency. Sampling picks a random CDF value and interpolates.
    Empirical {
        /// Sorted `(latency, cumulative_probability)` pairs.
        /// The last entry must have probability 1.0.
        buckets: Vec<(Duration, f64)>,
    },

    /// Composite model: sum of independent gateway, exchange, and network components.
    ///
    /// - Gateway: fixed
    /// - Exchange: fixed
    /// - Network: normally distributed
    Composite {
        /// Fixed gateway serialisation latency.
        gateway: Duration,
        /// Fixed exchange matching engine latency.
        exchange: Duration,
        /// Mean network latency.
        network_mean: Duration,
        /// Standard deviation of network jitter.
        network_stddev: Duration,
    },
}

impl LatencyModel {
    // ── Constructors ───────────────────────────────────────────────────

    /// Zero-latency model (instant fills).
    pub fn zero() -> Self {
        Self::Zero
    }

    /// Fixed constant latency.
    pub fn fixed(latency: Duration) -> Self {
        Self::Fixed { latency }
    }

    /// Normally distributed latency.
    pub fn normal(mean: Duration, stddev: Duration) -> Self {
        Self::Normal { mean, stddev }
    }

    /// Uniformly distributed latency between `min` and `max`.
    pub fn uniform(min: Duration, max: Duration) -> Self {
        let (min, max) = if min > max { (max, min) } else { (min, max) };
        Self::Uniform { min, max }
    }

    /// Log-normally distributed latency.
    ///
    /// # Parameters
    /// - `median`: The median latency (50th percentile).
    /// - `p95_ratio`: The ratio of 95th percentile to median.
    ///   e.g., if median is 5ms and p95 is 15ms, pass `p95_ratio = 3.0`.
    pub fn log_normal(median: Duration, p95_ratio: f64) -> Self {
        // ln(latency) ~ Normal(mu, sigma)
        // median = exp(mu) → mu = ln(median_micros)
        // p95 / median = exp(1.645 * sigma) → sigma = ln(p95_ratio) / 1.645
        let mu = (median.as_micros() as f64).ln();
        let sigma = if p95_ratio > 1.0 {
            p95_ratio.ln() / 1.645
        } else {
            0.1 // minimal jitter
        };
        Self::LogNormal { mu, sigma }
    }

    /// Empirical distribution from observed latency samples.
    ///
    /// Accepts a sorted slice of `(latency, cdf_probability)` pairs.
    /// The last entry's probability should be 1.0.
    pub fn empirical(buckets: Vec<(Duration, f64)>) -> Self {
        debug_assert!(
            buckets
                .last()
                .map(|(_, p)| (*p - 1.0).abs() < 1e-6)
                .unwrap_or(true),
            "Last bucket CDF should be ~1.0"
        );
        Self::Empirical { buckets }
    }

    /// Composite model combining gateway + exchange + network jitter.
    pub fn composite(
        gateway: Duration,
        exchange: Duration,
        network_mean: Duration,
        network_stddev: Duration,
    ) -> Self {
        Self::Composite {
            gateway,
            exchange,
            network_mean,
            network_stddev,
        }
    }

    // ── Sampling ───────────────────────────────────────────────────────

    /// Sample a latency duration from this model.
    ///
    /// Uses `rand::thread_rng()` for randomness.
    pub fn sample_latency(&self) -> Duration {
        let mut rng = rand::rng();
        self.sample_with_rng(&mut rng)
    }

    /// Sample a latency duration using a specific RNG (for determinism).
    pub fn sample_with_rng<R: Rng>(&self, rng: &mut R) -> Duration {
        let latency = match self {
            Self::Zero => Duration::ZERO,

            Self::Fixed { latency } => *latency,

            Self::Normal { mean, stddev } => {
                let mean_us = mean.as_micros() as f64;
                let std_us = stddev.as_micros() as f64;
                let sample = sample_normal(rng, mean_us, std_us);
                // Clamp to [0, mean + 5*stddev] to avoid negative / extreme values.
                let clamped = sample.max(0.0).min(mean_us + 5.0 * std_us);
                Duration::from_micros(clamped as u64)
            }

            Self::Uniform { min, max } => {
                let min_us = min.as_micros() as u64;
                let max_us = max.as_micros() as u64;
                if min_us >= max_us {
                    *min
                } else {
                    let sample = rng.random_range(min_us..=max_us);
                    Duration::from_micros(sample)
                }
            }

            Self::LogNormal { mu, sigma } => {
                let z = sample_normal(rng, 0.0, 1.0);
                let log_latency = mu + sigma * z;
                let micros = log_latency.exp().max(0.0);
                // Cap at 10 seconds to prevent absurd outliers.
                let capped = micros.min(10_000_000.0);
                Duration::from_micros(capped as u64)
            }

            Self::Empirical { buckets } => {
                if buckets.is_empty() {
                    Duration::ZERO
                } else if buckets.len() == 1 {
                    buckets[0].0
                } else {
                    let u: f64 = rng.random();
                    sample_empirical(buckets, u)
                }
            }

            Self::Composite {
                gateway,
                exchange,
                network_mean,
                network_stddev,
            } => {
                let net_mean_us = network_mean.as_micros() as f64;
                let net_std_us = network_stddev.as_micros() as f64;
                let net_sample = sample_normal(rng, net_mean_us, net_std_us).max(0.0);
                let network = Duration::from_micros(net_sample as u64);
                *gateway + *exchange + network
            }
        };

        trace!(
            model = %self,
            latency_us = latency.as_micros(),
            "Latency sampled"
        );

        latency
    }

    /// Sample a full latency estimate with component breakdown.
    pub fn sample_estimate(&self) -> LatencyEstimate {
        match self {
            Self::Composite {
                gateway,
                exchange,
                network_mean,
                network_stddev,
            } => {
                let mut rng = rand::rng();
                let net_mean_us = network_mean.as_micros() as f64;
                let net_std_us = network_stddev.as_micros() as f64;
                let net_sample = sample_normal(&mut rng, net_mean_us, net_std_us).max(0.0);
                let network = Duration::from_micros(net_sample as u64);
                LatencyEstimate::composite(*gateway, *exchange, network)
            }
            _ => {
                let total = self.sample_latency();
                LatencyEstimate::simple(total)
            }
        }
    }

    /// Get the expected (mean) latency for this model.
    pub fn expected_latency(&self) -> Duration {
        match self {
            Self::Zero => Duration::ZERO,
            Self::Fixed { latency } => *latency,
            Self::Normal { mean, .. } => *mean,
            Self::Uniform { min, max } => {
                let avg = (min.as_micros() + max.as_micros()) / 2;
                Duration::from_micros(avg as u64)
            }
            Self::LogNormal { mu, sigma } => {
                // E[X] = exp(mu + sigma²/2)
                let mean_us = (mu + sigma * sigma / 2.0).exp();
                Duration::from_micros(mean_us as u64)
            }
            Self::Empirical { buckets } => {
                if buckets.is_empty() {
                    return Duration::ZERO;
                }
                // Weighted average using CDF differences.
                let mut prev_cdf = 0.0;
                let mut weighted_sum = 0.0;
                for (dur, cdf) in buckets {
                    let weight = cdf - prev_cdf;
                    weighted_sum += dur.as_micros() as f64 * weight;
                    prev_cdf = *cdf;
                }
                Duration::from_micros(weighted_sum as u64)
            }
            Self::Composite {
                gateway,
                exchange,
                network_mean,
                ..
            } => *gateway + *exchange + *network_mean,
        }
    }

    /// Model name for display / logging.
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Zero => "Zero",
            Self::Fixed { .. } => "Fixed",
            Self::Normal { .. } => "Normal",
            Self::Uniform { .. } => "Uniform",
            Self::LogNormal { .. } => "LogNormal",
            Self::Empirical { .. } => "Empirical",
            Self::Composite { .. } => "Composite",
        }
    }
}

impl Default for LatencyModel {
    fn default() -> Self {
        // Default: 5ms mean, 2ms stddev — representative of co-located exchange access.
        Self::Normal {
            mean: Duration::from_millis(5),
            stddev: Duration::from_millis(2),
        }
    }
}

impl fmt::Display for LatencyModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "Zero"),
            Self::Fixed { latency } => write!(f, "Fixed({}µs)", latency.as_micros()),
            Self::Normal { mean, stddev } => {
                write!(
                    f,
                    "Normal(mean={}µs, std={}µs)",
                    mean.as_micros(),
                    stddev.as_micros()
                )
            }
            Self::Uniform { min, max } => {
                write!(f, "Uniform({}µs..{}µs)", min.as_micros(), max.as_micros())
            }
            Self::LogNormal { mu, sigma } => {
                write!(f, "LogNormal(µ={:.2}, σ={:.2})", mu, sigma)
            }
            Self::Empirical { buckets } => {
                write!(f, "Empirical({} buckets)", buckets.len())
            }
            Self::Composite {
                gateway,
                exchange,
                network_mean,
                network_stddev,
            } => {
                write!(
                    f,
                    "Composite(gw={}µs, ex={}µs, net=Normal({}µs, {}µs))",
                    gateway.as_micros(),
                    exchange.as_micros(),
                    network_mean.as_micros(),
                    network_stddev.as_micros(),
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sample from a normal distribution using the Box-Muller transform.
fn sample_normal<R: Rng>(rng: &mut R, mean: f64, stddev: f64) -> f64 {
    if stddev <= 0.0 {
        return mean;
    }
    // Box-Muller transform
    let u1: f64 = rng.random::<f64>().max(1e-15); // avoid ln(0)
    let u2: f64 = rng.random::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + stddev * z
}

/// Sample from an empirical CDF distribution via linear interpolation.
fn sample_empirical(buckets: &[(Duration, f64)], u: f64) -> Duration {
    // Find the first bucket where CDF >= u.
    let idx = buckets
        .iter()
        .position(|(_, cdf)| *cdf >= u)
        .unwrap_or(buckets.len() - 1);

    if idx == 0 {
        return buckets[0].0;
    }

    // Linear interpolation between bucket[idx-1] and bucket[idx].
    let (dur_lo, cdf_lo) = buckets[idx - 1];
    let (dur_hi, cdf_hi) = buckets[idx];

    let cdf_range = cdf_hi - cdf_lo;
    if cdf_range <= 0.0 {
        return dur_hi;
    }

    let frac = (u - cdf_lo) / cdf_range;
    let lo_us = dur_lo.as_micros() as f64;
    let hi_us = dur_hi.as_micros() as f64;
    let interpolated = lo_us + frac * (hi_us - lo_us);

    Duration::from_micros(interpolated.max(0.0) as u64)
}

/// Generate a deterministic latency sample from a counter (for testing).
/// Not used in production — prefer `sample_latency()`.
#[allow(dead_code)]
fn deterministic_sample(base_micros: u64, jitter_micros: u64) -> Duration {
    let counter = LATENCY_RNG_COUNTER.fetch_add(1, Ordering::Relaxed);
    // Simple hash of counter for pseudo-random jitter.
    let hash = counter
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let jitter = if jitter_micros > 0 {
        hash % jitter_micros
    } else {
        0
    };
    Duration::from_micros(base_micros + jitter)
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── LatencyEstimate ────────────────────────────────────────────────

    #[test]
    fn test_estimate_simple() {
        let est = LatencyEstimate::simple(Duration::from_millis(5));
        assert_eq!(est.total, Duration::from_millis(5));
        assert_eq!(est.gateway, Duration::ZERO);
        assert_eq!(est.exchange, Duration::ZERO);
        assert_eq!(est.network, Duration::from_millis(5));
        assert_eq!(est.total_micros(), 5000);
    }

    #[test]
    fn test_estimate_composite() {
        let est = LatencyEstimate::composite(
            Duration::from_millis(1),
            Duration::from_micros(500),
            Duration::from_millis(3),
        );
        assert_eq!(est.total, Duration::from_micros(4500));
        assert_eq!(est.gateway, Duration::from_millis(1));
        assert_eq!(est.exchange, Duration::from_micros(500));
        assert_eq!(est.network, Duration::from_millis(3));
    }

    #[test]
    fn test_estimate_total_millis_f64() {
        let est = LatencyEstimate::simple(Duration::from_micros(5500));
        let millis = est.total_millis_f64();
        assert!((millis - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_estimate_display_simple() {
        let est = LatencyEstimate::simple(Duration::from_millis(5));
        let s = format!("{}", est);
        assert!(s.contains("5000"));
    }

    #[test]
    fn test_estimate_display_composite() {
        let est = LatencyEstimate::composite(
            Duration::from_millis(1),
            Duration::from_micros(100),
            Duration::from_millis(3),
        );
        let s = format!("{}", est);
        assert!(s.contains("gw:"));
        assert!(s.contains("ex:"));
        assert!(s.contains("net:"));
    }

    // ── LatencyStats ───────────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let stats = LatencyStats::default();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean(), Duration::ZERO);
        assert_eq!(stats.variance_micros(), 0.0);
    }

    #[test]
    fn test_stats_record() {
        let mut stats = LatencyStats::default();
        stats.record(Duration::from_millis(5));
        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(15));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean(), Duration::from_millis(10));
        assert_eq!(stats.min, Duration::from_millis(5));
        assert_eq!(stats.max, Duration::from_millis(15));
        assert_eq!(stats.last, Duration::from_millis(15));
    }

    #[test]
    fn test_stats_variance() {
        let mut stats = LatencyStats::default();
        // All same → variance 0
        stats.record(Duration::from_millis(5));
        stats.record(Duration::from_millis(5));
        stats.record(Duration::from_millis(5));
        assert!(stats.variance_micros().abs() < 1.0);
    }

    #[test]
    fn test_stats_stddev() {
        let mut stats = LatencyStats::default();
        stats.record(Duration::from_millis(2));
        stats.record(Duration::from_millis(4));
        stats.record(Duration::from_millis(6));
        stats.record(Duration::from_millis(8));
        // mean = 5ms, variance = mean of squares - square of mean
        // = (4+16+36+64)/4 - 25 = 30 - 25 = 5 → stddev ≈ 2236 µs
        let stddev = stats.stddev();
        assert!(stddev.as_micros() > 2000);
        assert!(stddev.as_micros() < 2500);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = LatencyStats::default();
        stats.record(Duration::from_millis(10));
        stats.reset();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.sum_micros, 0);
    }

    #[test]
    fn test_stats_display_empty() {
        let stats = LatencyStats::default();
        let s = format!("{}", stats);
        assert!(s.contains("no samples"));
    }

    #[test]
    fn test_stats_display_with_data() {
        let mut stats = LatencyStats::default();
        stats.record(Duration::from_millis(5));
        let s = format!("{}", stats);
        assert!(s.contains("n=1"));
    }

    // ── LatencyModel ───────────────────────────────────────────────────

    #[test]
    fn test_zero_model() {
        let model = LatencyModel::zero();
        assert_eq!(model.sample_latency(), Duration::ZERO);
        assert_eq!(model.expected_latency(), Duration::ZERO);
        assert_eq!(model.model_name(), "Zero");
    }

    #[test]
    fn test_fixed_model() {
        let model = LatencyModel::fixed(Duration::from_millis(10));
        assert_eq!(model.sample_latency(), Duration::from_millis(10));
        assert_eq!(model.expected_latency(), Duration::from_millis(10));
        assert_eq!(model.model_name(), "Fixed");
    }

    #[test]
    fn test_normal_model_non_negative() {
        let model = LatencyModel::normal(Duration::from_millis(5), Duration::from_millis(2));
        // Sample many times and ensure non-negative.
        for _ in 0..100 {
            let lat = model.sample_latency();
            assert!(lat >= Duration::ZERO);
        }
        assert_eq!(model.expected_latency(), Duration::from_millis(5));
    }

    #[test]
    fn test_normal_model_zero_stddev() {
        let model = LatencyModel::normal(Duration::from_millis(5), Duration::ZERO);
        // With zero stddev, all samples should be exactly the mean.
        for _ in 0..10 {
            assert_eq!(model.sample_latency(), Duration::from_millis(5));
        }
    }

    #[test]
    fn test_uniform_model() {
        let model = LatencyModel::uniform(Duration::from_millis(3), Duration::from_millis(7));
        for _ in 0..100 {
            let lat = model.sample_latency();
            assert!(lat >= Duration::from_millis(3));
            assert!(lat <= Duration::from_millis(7));
        }
        assert_eq!(model.expected_latency(), Duration::from_millis(5));
    }

    #[test]
    fn test_uniform_model_swapped_bounds() {
        // Constructor should swap min/max if inverted.
        let model = LatencyModel::uniform(Duration::from_millis(10), Duration::from_millis(2));
        for _ in 0..50 {
            let lat = model.sample_latency();
            assert!(lat >= Duration::from_millis(2));
            assert!(lat <= Duration::from_millis(10));
        }
    }

    #[test]
    fn test_log_normal_model_positive() {
        let model = LatencyModel::log_normal(
            Duration::from_millis(5),
            3.0, // p95 = 15ms
        );
        for _ in 0..100 {
            let lat = model.sample_latency();
            assert!(lat > Duration::ZERO);
        }
        assert_eq!(model.model_name(), "LogNormal");
    }

    #[test]
    fn test_log_normal_expected() {
        let model = LatencyModel::log_normal(Duration::from_millis(5), 2.0);
        let expected = model.expected_latency();
        // Expected should be >= median for log-normal.
        assert!(expected >= Duration::from_millis(5));
    }

    #[test]
    fn test_empirical_model() {
        let buckets = vec![
            (Duration::from_millis(1), 0.1),
            (Duration::from_millis(3), 0.5),
            (Duration::from_millis(5), 0.8),
            (Duration::from_millis(10), 0.95),
            (Duration::from_millis(50), 1.0),
        ];
        let model = LatencyModel::empirical(buckets);
        for _ in 0..100 {
            let lat = model.sample_latency();
            assert!(lat >= Duration::from_millis(1));
            assert!(lat <= Duration::from_millis(50));
        }
        assert_eq!(model.model_name(), "Empirical");
    }

    #[test]
    fn test_empirical_model_single_bucket() {
        let buckets = vec![(Duration::from_millis(5), 1.0)];
        let model = LatencyModel::empirical(buckets);
        assert_eq!(model.sample_latency(), Duration::from_millis(5));
    }

    #[test]
    fn test_empirical_model_empty() {
        let model = LatencyModel::empirical(Vec::new());
        assert_eq!(model.sample_latency(), Duration::ZERO);
    }

    #[test]
    fn test_empirical_expected() {
        let buckets = vec![
            (Duration::from_millis(2), 0.5),
            (Duration::from_millis(8), 1.0),
        ];
        let model = LatencyModel::empirical(buckets);
        let expected = model.expected_latency();
        // Weighted: 2*0.5 + 8*0.5 = 5ms
        assert_eq!(expected, Duration::from_millis(5));
    }

    #[test]
    fn test_composite_model() {
        let model = LatencyModel::composite(
            Duration::from_millis(1),
            Duration::from_micros(100),
            Duration::from_millis(3),
            Duration::from_millis(1),
        );
        for _ in 0..100 {
            let lat = model.sample_latency();
            // At minimum, gateway + exchange = 1.1ms.
            assert!(lat >= Duration::from_micros(1100));
        }
        assert_eq!(model.model_name(), "Composite");
    }

    #[test]
    fn test_composite_expected() {
        let model = LatencyModel::composite(
            Duration::from_millis(1),
            Duration::from_micros(500),
            Duration::from_millis(3),
            Duration::from_millis(1),
        );
        let expected = model.expected_latency();
        assert_eq!(expected, Duration::from_micros(4500));
    }

    #[test]
    fn test_composite_estimate() {
        let model = LatencyModel::composite(
            Duration::from_millis(1),
            Duration::from_micros(100),
            Duration::from_millis(3),
            Duration::from_millis(1),
        );
        let est = model.sample_estimate();
        assert_eq!(est.gateway, Duration::from_millis(1));
        assert_eq!(est.exchange, Duration::from_micros(100));
        assert!(est.network > Duration::ZERO);
        assert_eq!(est.total, est.gateway + est.exchange + est.network);
    }

    #[test]
    fn test_non_composite_estimate() {
        let model = LatencyModel::fixed(Duration::from_millis(5));
        let est = model.sample_estimate();
        assert_eq!(est.total, Duration::from_millis(5));
        assert_eq!(est.gateway, Duration::ZERO);
    }

    #[test]
    fn test_default_model() {
        let model = LatencyModel::default();
        assert_eq!(model.model_name(), "Normal");
        assert_eq!(model.expected_latency(), Duration::from_millis(5));
    }

    // ── Display ────────────────────────────────────────────────────────

    #[test]
    fn test_display_zero() {
        let s = format!("{}", LatencyModel::zero());
        assert_eq!(s, "Zero");
    }

    #[test]
    fn test_display_fixed() {
        let s = format!("{}", LatencyModel::fixed(Duration::from_millis(5)));
        assert!(s.contains("5000"));
    }

    #[test]
    fn test_display_normal() {
        let s = format!(
            "{}",
            LatencyModel::normal(Duration::from_millis(5), Duration::from_millis(2),)
        );
        assert!(s.contains("Normal"));
        assert!(s.contains("5000"));
        assert!(s.contains("2000"));
    }

    #[test]
    fn test_display_uniform() {
        let s = format!(
            "{}",
            LatencyModel::uniform(Duration::from_millis(2), Duration::from_millis(8),)
        );
        assert!(s.contains("Uniform"));
    }

    #[test]
    fn test_display_lognormal() {
        let s = format!(
            "{}",
            LatencyModel::log_normal(Duration::from_millis(5), 2.0,)
        );
        assert!(s.contains("LogNormal"));
    }

    #[test]
    fn test_display_empirical() {
        let model = LatencyModel::empirical(vec![
            (Duration::from_millis(1), 0.5),
            (Duration::from_millis(5), 1.0),
        ]);
        let s = format!("{}", model);
        assert!(s.contains("2 buckets"));
    }

    #[test]
    fn test_display_composite() {
        let model = LatencyModel::composite(
            Duration::from_millis(1),
            Duration::from_micros(100),
            Duration::from_millis(3),
            Duration::from_millis(1),
        );
        let s = format!("{}", model);
        assert!(s.contains("Composite"));
    }

    // ── Helpers ────────────────────────────────────────────────────────

    #[test]
    fn test_sample_normal_zero_stddev() {
        let mut rng = rand::rng();
        let sample = sample_normal(&mut rng, 100.0, 0.0);
        assert_eq!(sample, 100.0);
    }

    #[test]
    fn test_sample_empirical_edge_cases() {
        let buckets = vec![
            (Duration::from_millis(1), 0.3),
            (Duration::from_millis(5), 0.7),
            (Duration::from_millis(10), 1.0),
        ];

        // u = 0.0 → first bucket
        let lat = sample_empirical(&buckets, 0.0);
        assert_eq!(lat, Duration::from_millis(1));

        // u = 1.0 → last bucket
        let lat = sample_empirical(&buckets, 1.0);
        assert!(lat <= Duration::from_millis(10));
    }

    #[test]
    fn test_sample_empirical_interpolation() {
        let buckets = vec![
            (Duration::from_millis(0), 0.0),
            (Duration::from_millis(10), 1.0),
        ];

        // u = 0.5 → should interpolate to ~5ms
        let lat = sample_empirical(&buckets, 0.5);
        assert!(lat >= Duration::from_millis(4));
        assert!(lat <= Duration::from_millis(6));
    }

    #[test]
    fn test_deterministic_sample() {
        let d1 = deterministic_sample(1000, 500);
        let d2 = deterministic_sample(1000, 500);
        // Both should be in range [1000, 1500).
        assert!(d1 >= Duration::from_micros(1000));
        assert!(d1 < Duration::from_micros(1500));
        assert!(d2 >= Duration::from_micros(1000));
        assert!(d2 < Duration::from_micros(1500));
        // They should be different (different counter values).
        // Note: could theoretically be equal, but extremely unlikely.
    }

    #[test]
    fn test_deterministic_sample_zero_jitter() {
        let d = deterministic_sample(5000, 0);
        assert_eq!(d, Duration::from_micros(5000));
    }

    #[test]
    fn test_sample_with_rng_deterministic() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let model = LatencyModel::normal(Duration::from_millis(5), Duration::from_millis(2));

        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let s1 = model.sample_with_rng(&mut rng1);
        let s2 = model.sample_with_rng(&mut rng2);

        assert_eq!(s1, s2, "Same seed should produce same latency");
    }
}
