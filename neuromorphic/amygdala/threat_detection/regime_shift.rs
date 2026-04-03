//! Sudden regime shifts
//!
//! Part of the Amygdala region
//! Component: threat_detection
//!
//! Detects market regime changes using multiple statistical methods:
//! - Hidden Markov Model-inspired state detection
//! - Variance ratio analysis
//! - Structural break detection (CUSUM)
//! - Mean reversion parameter shifts
//! - Volatility clustering changes

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for regime shift detection
#[derive(Debug, Clone)]
pub struct RegimeShiftConfig {
    /// Window size for baseline calculations
    pub baseline_window: usize,
    /// Window size for current regime analysis
    pub analysis_window: usize,
    /// Threshold for detecting significant mean shift (in std devs)
    pub mean_shift_threshold: f64,
    /// Threshold for detecting variance regime change
    pub variance_ratio_threshold: f64,
    /// CUSUM threshold for structural break
    pub cusum_threshold: f64,
    /// Minimum observations before detection is active
    pub min_observations: usize,
    /// Decay factor for exponential statistics
    pub ema_decay: f64,
    /// Confidence level for regime classification
    pub confidence_threshold: f64,
}

impl Default for RegimeShiftConfig {
    fn default() -> Self {
        Self {
            baseline_window: 200,
            analysis_window: 50,
            mean_shift_threshold: 2.5,
            variance_ratio_threshold: 2.0,
            cusum_threshold: 4.0,
            min_observations: 100,
            ema_decay: 0.94,
            confidence_threshold: 0.7,
        }
    }
}

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Low volatility, trending market
    LowVolTrending,
    /// Low volatility, mean reverting
    LowVolMeanReverting,
    /// High volatility, trending
    HighVolTrending,
    /// High volatility, mean reverting
    HighVolMeanReverting,
    /// Crisis/stress regime
    Crisis,
    /// Transition between regimes
    Transitional,
    /// Unknown/insufficient data
    Unknown,
}

impl MarketRegime {
    /// Returns true if this is a high-risk regime
    pub fn is_high_risk(&self) -> bool {
        matches!(
            self,
            MarketRegime::Crisis | MarketRegime::HighVolTrending | MarketRegime::Transitional
        )
    }
}

/// Regime shift detection result
#[derive(Debug, Clone)]
pub struct RegimeShiftDetection {
    /// Whether a regime shift was detected
    pub shift_detected: bool,
    /// Current detected regime
    pub current_regime: MarketRegime,
    /// Previous regime (if shift detected)
    pub previous_regime: Option<MarketRegime>,
    /// Confidence in the detection (0.0 - 1.0)
    pub confidence: f64,
    /// Individual detection signals
    pub signals: RegimeSignals,
    /// Timestamp of detection
    pub timestamp: i64,
}

/// Individual regime detection signals
#[derive(Debug, Clone, Default)]
pub struct RegimeSignals {
    /// Mean shift z-score
    pub mean_shift_zscore: f64,
    /// Variance ratio (current / baseline)
    pub variance_ratio: f64,
    /// CUSUM statistic
    pub cusum_value: f64,
    /// Hurst exponent estimate (mean reversion indicator)
    pub hurst_estimate: f64,
    /// Volatility clustering measure
    pub volatility_persistence: f64,
}

/// Data point for regime analysis
#[derive(Debug, Clone)]
pub struct RegimeDataPoint {
    /// Value (price, return, etc.)
    pub value: f64,
    /// Timestamp
    pub timestamp: i64,
    /// Optional volume
    pub volume: Option<f64>,
}

/// Statistics for a regime window
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct WindowStats {
    mean: f64,
    variance: f64,
    std_dev: f64,
    count: usize,
}

impl WindowStats {
    fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = if count > 1 {
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };

        Self {
            mean,
            variance,
            std_dev: variance.sqrt(),
            count,
        }
    }
}

/// Sudden regime shifts detector
pub struct RegimeShift {
    config: RegimeShiftConfig,
    /// Rolling window of data points
    data: VecDeque<RegimeDataPoint>,
    /// Returns calculated from prices
    returns: VecDeque<f64>,
    /// Current detected regime
    current_regime: MarketRegime,
    /// CUSUM positive accumulator
    cusum_pos: f64,
    /// CUSUM negative accumulator
    cusum_neg: f64,
    /// EMA of volatility
    ema_volatility: f64,
    /// EMA of squared volatility (for clustering)
    ema_vol_squared: f64,
    /// Baseline statistics (from longer window)
    baseline_stats: WindowStats,
    /// Number of regime shifts detected
    total_shifts: usize,
    /// Last shift timestamp
    last_shift_timestamp: Option<i64>,
}

impl Default for RegimeShift {
    fn default() -> Self {
        Self::new()
    }
}

impl RegimeShift {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(RegimeShiftConfig::default())
    }

    /// Create a new instance with custom config
    pub fn with_config(config: RegimeShiftConfig) -> Self {
        Self {
            config,
            data: VecDeque::new(),
            returns: VecDeque::new(),
            current_regime: MarketRegime::Unknown,
            cusum_pos: 0.0,
            cusum_neg: 0.0,
            ema_volatility: 0.0,
            ema_vol_squared: 0.0,
            baseline_stats: WindowStats::default(),
            total_shifts: 0,
            last_shift_timestamp: None,
        }
    }

    /// Add a new data point and check for regime shift
    pub fn update(&mut self, point: RegimeDataPoint) -> RegimeShiftDetection {
        let timestamp = point.timestamp;

        // Calculate return if we have previous data
        if let Some(prev) = self.data.back() {
            if prev.value > 0.0 {
                let ret = (point.value / prev.value).ln();
                self.returns.push_back(ret);

                // Maintain returns window
                while self.returns.len() > self.config.baseline_window {
                    self.returns.pop_front();
                }
            }
        }

        // Add to data window
        self.data.push_back(point);
        while self.data.len() > self.config.baseline_window {
            self.data.pop_front();
        }

        // Perform detection
        self.detect_regime_shift(timestamp)
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // This is a stateless check - actual processing happens in update()
        Ok(())
    }

    /// Detect regime shift based on current data
    fn detect_regime_shift(&mut self, timestamp: i64) -> RegimeShiftDetection {
        // Check if we have enough data
        if self.returns.len() < self.config.min_observations {
            return RegimeShiftDetection {
                shift_detected: false,
                current_regime: MarketRegime::Unknown,
                previous_regime: None,
                confidence: 0.0,
                signals: RegimeSignals::default(),
                timestamp,
            };
        }

        // Calculate signals
        let signals = self.calculate_signals();

        // Update CUSUM
        self.update_cusum(&signals);

        // Classify current regime
        let new_regime = self.classify_regime(&signals);

        // Check for regime shift
        let shift_detected = self.is_regime_shift(&signals, new_regime);

        // Calculate confidence
        let confidence = self.calculate_confidence(&signals, shift_detected);

        let previous_regime = if shift_detected {
            let prev = Some(self.current_regime);
            self.current_regime = new_regime;
            self.total_shifts += 1;
            self.last_shift_timestamp = Some(timestamp);
            prev
        } else {
            None
        };

        RegimeShiftDetection {
            shift_detected,
            current_regime: self.current_regime,
            previous_regime,
            confidence,
            signals,
            timestamp,
        }
    }

    /// Calculate all regime detection signals
    fn calculate_signals(&mut self) -> RegimeSignals {
        let returns: Vec<f64> = self.returns.iter().copied().collect();

        // Split into baseline and analysis windows
        let baseline_end = returns.len().saturating_sub(self.config.analysis_window);
        let baseline_returns = &returns[..baseline_end];
        let analysis_returns = &returns[baseline_end..];

        // Calculate window statistics
        self.baseline_stats = WindowStats::from_values(baseline_returns);
        let analysis_stats = WindowStats::from_values(analysis_returns);

        // Mean shift z-score
        let mean_shift_zscore = if self.baseline_stats.std_dev > 1e-10 {
            (analysis_stats.mean - self.baseline_stats.mean) / self.baseline_stats.std_dev
        } else {
            0.0
        };

        // Variance ratio
        let variance_ratio = if self.baseline_stats.variance > 1e-10 {
            analysis_stats.variance / self.baseline_stats.variance
        } else {
            1.0
        };

        // Update EMA volatility
        let current_vol = analysis_stats.std_dev;
        if self.ema_volatility == 0.0 {
            self.ema_volatility = current_vol;
            self.ema_vol_squared = current_vol * current_vol;
        } else {
            let alpha = 1.0 - self.config.ema_decay;
            self.ema_volatility = self.config.ema_decay * self.ema_volatility + alpha * current_vol;
            self.ema_vol_squared =
                self.config.ema_decay * self.ema_vol_squared + alpha * current_vol * current_vol;
        }

        // Volatility persistence (GARCH-like measure)
        let vol_variance = self.ema_vol_squared - self.ema_volatility.powi(2);
        let volatility_persistence = if self.ema_volatility > 1e-10 {
            (vol_variance.max(0.0).sqrt() / self.ema_volatility).min(1.0)
        } else {
            0.0
        };

        // Hurst exponent estimate using R/S analysis (simplified)
        let hurst_estimate = self.estimate_hurst(&returns);

        RegimeSignals {
            mean_shift_zscore,
            variance_ratio,
            cusum_value: self.cusum_pos.max(self.cusum_neg.abs()),
            hurst_estimate,
            volatility_persistence,
        }
    }

    /// Update CUSUM statistics
    fn update_cusum(&mut self, signals: &RegimeSignals) {
        // CUSUM for mean shift detection
        let deviation = signals.mean_shift_zscore;

        // Positive CUSUM (detecting upward shift)
        self.cusum_pos = (self.cusum_pos + deviation - 0.5).max(0.0);

        // Negative CUSUM (detecting downward shift)
        self.cusum_neg = (self.cusum_neg - deviation - 0.5).min(0.0);

        // Apply decay to prevent drift
        self.cusum_pos *= 0.99;
        self.cusum_neg *= 0.99;
    }

    /// Estimate Hurst exponent using simplified R/S analysis
    fn estimate_hurst(&self, returns: &[f64]) -> f64 {
        if returns.len() < 20 {
            return 0.5; // Default to random walk
        }

        // Calculate for different window sizes
        let window_sizes = [10, 20, 40];
        let mut log_rs_sum = 0.0;
        let mut log_n_sum = 0.0;
        let mut count = 0;

        for &n in &window_sizes {
            if n > returns.len() {
                continue;
            }

            let mut rs_sum = 0.0;
            let mut rs_count = 0;

            for chunk in returns.windows(n) {
                let mean = chunk.iter().sum::<f64>() / n as f64;

                // Calculate cumulative deviations
                let mut cumsum = 0.0;
                let mut max_cumsum = f64::NEG_INFINITY;
                let mut min_cumsum = f64::INFINITY;

                for &x in chunk {
                    cumsum += x - mean;
                    max_cumsum = max_cumsum.max(cumsum);
                    min_cumsum = min_cumsum.min(cumsum);
                }

                let range = max_cumsum - min_cumsum;
                let std_dev =
                    (chunk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();

                if std_dev > 1e-10 {
                    rs_sum += range / std_dev;
                    rs_count += 1;
                }
            }

            if rs_count > 0 {
                let avg_rs = rs_sum / rs_count as f64;
                if avg_rs > 0.0 {
                    log_rs_sum += avg_rs.ln();
                    log_n_sum += (n as f64).ln();
                    count += 1;
                }
            }
        }

        // Estimate Hurst from slope (simplified)
        if count >= 2 {
            // Rough estimate - proper estimation would use linear regression
            (log_rs_sum / count as f64) / (log_n_sum / count as f64).max(1.0)
        } else {
            0.5
        }
        .clamp(0.0, 1.0)
    }

    /// Classify current regime based on signals
    fn classify_regime(&self, signals: &RegimeSignals) -> MarketRegime {
        // Check for crisis regime
        if signals.cusum_value > self.config.cusum_threshold
            || signals.variance_ratio > self.config.variance_ratio_threshold * 2.0
        {
            return MarketRegime::Crisis;
        }

        // Check for transitional regime
        if signals.cusum_value > self.config.cusum_threshold * 0.5
            || signals.mean_shift_zscore.abs() > self.config.mean_shift_threshold * 0.8
        {
            return MarketRegime::Transitional;
        }

        // Determine volatility regime
        let is_high_vol = signals.variance_ratio > self.config.variance_ratio_threshold * 0.5;

        // Determine trending vs mean-reverting (Hurst > 0.5 = trending)
        let is_trending = signals.hurst_estimate > 0.55;

        match (is_high_vol, is_trending) {
            (false, false) => MarketRegime::LowVolMeanReverting,
            (false, true) => MarketRegime::LowVolTrending,
            (true, false) => MarketRegime::HighVolMeanReverting,
            (true, true) => MarketRegime::HighVolTrending,
        }
    }

    /// Check if a regime shift has occurred
    fn is_regime_shift(&self, signals: &RegimeSignals, new_regime: MarketRegime) -> bool {
        // Regime change detected
        if new_regime != self.current_regime && self.current_regime != MarketRegime::Unknown {
            return true;
        }

        // CUSUM threshold breach
        if signals.cusum_value > self.config.cusum_threshold {
            return true;
        }

        // Strong mean shift
        if signals.mean_shift_zscore.abs() > self.config.mean_shift_threshold {
            return true;
        }

        // Extreme variance ratio change
        if signals.variance_ratio > self.config.variance_ratio_threshold * 1.5
            || signals.variance_ratio < 1.0 / (self.config.variance_ratio_threshold * 1.5)
        {
            return true;
        }

        false
    }

    /// Calculate confidence in regime detection
    fn calculate_confidence(&self, signals: &RegimeSignals, shift_detected: bool) -> f64 {
        let mut confidence = 0.0;

        // Base confidence from sample size
        let sample_ratio = self.returns.len() as f64 / self.config.baseline_window as f64;
        confidence += 0.2 * sample_ratio.min(1.0);

        // Confidence from mean shift clarity
        let mean_shift_clarity =
            (signals.mean_shift_zscore.abs() / self.config.mean_shift_threshold).min(1.0);
        confidence += 0.25 * mean_shift_clarity;

        // Confidence from variance ratio
        let var_ratio_clarity = if signals.variance_ratio > 1.0 {
            ((signals.variance_ratio - 1.0) / (self.config.variance_ratio_threshold - 1.0)).min(1.0)
        } else {
            ((1.0 / signals.variance_ratio - 1.0) / (self.config.variance_ratio_threshold - 1.0))
                .min(1.0)
        };
        confidence += 0.25 * var_ratio_clarity;

        // Confidence from CUSUM
        let cusum_clarity = (signals.cusum_value / self.config.cusum_threshold).min(1.0);
        confidence += 0.2 * cusum_clarity;

        // Boost if shift detected
        if shift_detected {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    /// Get current regime
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get total number of regime shifts detected
    pub fn total_shifts(&self) -> usize {
        self.total_shifts
    }

    /// Get last shift timestamp
    pub fn last_shift_timestamp(&self) -> Option<i64> {
        self.last_shift_timestamp
    }

    /// Get current CUSUM value
    pub fn cusum_value(&self) -> f64 {
        self.cusum_pos.max(self.cusum_neg.abs())
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.data.clear();
        self.returns.clear();
        self.current_regime = MarketRegime::Unknown;
        self.cusum_pos = 0.0;
        self.cusum_neg = 0.0;
        self.ema_volatility = 0.0;
        self.ema_vol_squared = 0.0;
        self.baseline_stats = WindowStats::default();
        self.total_shifts = 0;
        self.last_shift_timestamp = None;
    }

    /// Check if detector has enough data
    pub fn is_ready(&self) -> bool {
        self.returns.len() >= self.config.min_observations
    }

    /// Get number of observations
    pub fn observation_count(&self) -> usize {
        self.returns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = RegimeShift::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_default_config() {
        let config = RegimeShiftConfig::default();
        assert_eq!(config.baseline_window, 200);
        assert_eq!(config.analysis_window, 50);
        assert_eq!(config.min_observations, 100);
    }

    #[test]
    fn test_regime_classification() {
        let mut detector = RegimeShift::new();

        // Feed enough data to make the detector ready
        let base_price = 100.0;
        for i in 0..150 {
            // Small oscillation around base price
            let variation = (i as f64 * 0.1).sin() * 0.5;
            let price = base_price + variation;
            let point = RegimeDataPoint {
                value: price,
                timestamp: i as i64 * 1000,
                volume: Some(1000.0),
            };
            detector.update(point);
        }

        // Verify detector is ready and produces a regime classification
        assert!(detector.is_ready());
        let regime = detector.current_regime();
        // The detector should produce some regime (not necessarily a specific one)
        // as the exact classification depends on signal calculations
        assert!(
            regime != MarketRegime::Unknown,
            "Detector should classify a regime after sufficient data"
        );
    }

    #[test]
    fn test_regime_shift_detection() {
        let config = RegimeShiftConfig {
            baseline_window: 100,
            analysis_window: 25,
            min_observations: 50,
            ..Default::default()
        };
        let mut detector = RegimeShift::with_config(config);

        // Generate stable data first
        let mut price = 100.0;
        for i in 0..80 {
            price *= 1.0 + 0.001 * (i as f64 * 0.5).sin(); // Small oscillation
            let point = RegimeDataPoint {
                value: price,
                timestamp: i as i64 * 1000,
                volume: Some(1000.0),
            };
            detector.update(point);
        }

        // Introduce a sudden volatility spike
        for i in 80..120 {
            let shock = if i % 2 == 0 { 0.05 } else { -0.04 }; // High volatility
            price *= 1.0 + shock;
            let point = RegimeDataPoint {
                value: price,
                timestamp: i as i64 * 1000,
                volume: Some(5000.0), // Higher volume too
            };
            let detection = detector.update(point);

            // At some point we should detect a shift
            if detection.shift_detected {
                assert!(
                    detection.current_regime.is_high_risk()
                        || detection.current_regime == MarketRegime::Transitional
                );
                break;
            }
        }

        // Should have detected at least one shift
        assert!(
            detector.total_shifts() >= 1,
            "Expected regime shift, detected: {}",
            detector.total_shifts()
        );
    }

    #[test]
    fn test_cusum_tracking() {
        let mut detector = RegimeShift::new();

        // Feed data with gradual mean shift
        let mut price = 100.0;
        for i in 0..200 {
            // Add gradual upward drift after i=100
            let drift = if i > 100 { 0.005 } else { 0.0 };
            price *= 1.0 + drift + 0.001 * ((i as f64).sin());

            let point = RegimeDataPoint {
                value: price,
                timestamp: i as i64 * 1000,
                volume: Some(1000.0),
            };
            detector.update(point);
        }

        // CUSUM should have increased due to the drift
        let cusum = detector.cusum_value();
        assert!(
            cusum > 0.0,
            "CUSUM should be positive with upward drift: {}",
            cusum
        );
    }

    #[test]
    fn test_reset() {
        let mut detector = RegimeShift::new();

        // Add some data
        for i in 0..50 {
            let point = RegimeDataPoint {
                value: 100.0 + i as f64 * 0.1,
                timestamp: i as i64 * 1000,
                volume: Some(1000.0),
            };
            detector.update(point);
        }

        assert!(detector.observation_count() > 0);

        detector.reset();

        assert_eq!(detector.observation_count(), 0);
        assert_eq!(detector.current_regime(), MarketRegime::Unknown);
        assert_eq!(detector.total_shifts(), 0);
    }

    #[test]
    fn test_market_regime_is_high_risk() {
        assert!(MarketRegime::Crisis.is_high_risk());
        assert!(MarketRegime::HighVolTrending.is_high_risk());
        assert!(MarketRegime::Transitional.is_high_risk());
        assert!(!MarketRegime::LowVolMeanReverting.is_high_risk());
        assert!(!MarketRegime::LowVolTrending.is_high_risk());
    }

    #[test]
    fn test_insufficient_data() {
        let mut detector = RegimeShift::new();

        // Add just a few points
        for i in 0..10 {
            let point = RegimeDataPoint {
                value: 100.0 + i as f64,
                timestamp: i as i64 * 1000,
                volume: Some(1000.0),
            };
            let detection = detector.update(point);

            // Should not be ready yet
            assert!(!detection.shift_detected);
            assert_eq!(detection.current_regime, MarketRegime::Unknown);
            assert_eq!(detection.confidence, 0.0);
        }

        assert!(!detector.is_ready());
    }
}
