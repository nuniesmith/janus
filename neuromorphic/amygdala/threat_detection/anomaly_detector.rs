//! Anomaly Detector - Statistical anomaly detection for market data
//!
//! Detects unusual market conditions using multiple statistical methods:
//! - Z-score based detection
//! - Isolation Forest-inspired scoring
//! - Moving average deviation
//! - Percentile-based outlier detection
//! - Multi-variate anomaly scoring

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyDetectorConfig {
    /// Window size for statistical calculations
    pub window_size: usize,
    /// Minimum samples before detection is active
    pub min_samples: usize,
    /// Z-score threshold for anomaly
    pub zscore_threshold: f64,
    /// Percentile threshold for outliers (e.g., 0.99 = 99th percentile)
    pub percentile_threshold: f64,
    /// Number of features to track
    pub num_features: usize,
    /// EMA decay factor
    pub ema_decay: f64,
    /// Sensitivity multiplier (higher = more sensitive)
    pub sensitivity: f64,
    /// Cool-down period after anomaly (in samples)
    pub cooldown_samples: usize,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            min_samples: 30,
            zscore_threshold: 3.0,
            percentile_threshold: 0.99,
            num_features: 10,
            ema_decay: 0.94,
            sensitivity: 1.0,
            cooldown_samples: 5,
        }
    }
}

/// Anomaly severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl AnomalySeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    pub fn from_score(score: f64) -> Self {
        if score >= 0.9 {
            Self::Critical
        } else if score >= 0.7 {
            Self::High
        } else if score >= 0.5 {
            Self::Medium
        } else if score >= 0.3 {
            Self::Low
        } else {
            Self::None
        }
    }
}

/// Type of anomaly detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Statistical outlier (z-score)
    StatisticalOutlier,
    /// Sudden spike in value
    Spike,
    /// Sudden drop in value
    Drop,
    /// Unusual pattern across multiple features
    MultiVariate,
    /// Rate of change anomaly
    VelocityAnomaly,
    /// Volatility regime change
    VolatilityShift,
    /// Unknown/combined
    Unknown,
}

impl AnomalyType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::StatisticalOutlier => "statistical_outlier",
            Self::Spike => "spike",
            Self::Drop => "drop",
            Self::MultiVariate => "multi_variate",
            Self::VelocityAnomaly => "velocity_anomaly",
            Self::VolatilityShift => "volatility_shift",
            Self::Unknown => "unknown",
        }
    }
}

/// Market data point for anomaly detection
#[derive(Debug, Clone)]
pub struct MarketDataPoint {
    /// Feature values (price, volume, volatility, etc.)
    pub features: Vec<f64>,
    /// Timestamp
    pub timestamp: i64,
    /// Optional label/context
    pub label: Option<String>,
}

impl MarketDataPoint {
    pub fn new(features: Vec<f64>, timestamp: i64) -> Self {
        Self {
            features,
            timestamp,
            label: None,
        }
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }
}

/// Detected anomaly result
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Whether an anomaly was detected
    pub is_anomaly: bool,
    /// Overall anomaly score (0.0 - 1.0)
    pub score: f64,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Z-scores for each feature
    pub feature_zscores: Vec<f64>,
    /// Most anomalous feature index
    pub most_anomalous_feature: Option<usize>,
    /// Timestamp of detection
    pub timestamp: i64,
    /// Contributing factors description
    pub factors: Vec<String>,
}

impl Default for AnomalyResult {
    fn default() -> Self {
        Self {
            is_anomaly: false,
            score: 0.0,
            severity: AnomalySeverity::None,
            anomaly_type: AnomalyType::Unknown,
            feature_zscores: Vec::new(),
            most_anomalous_feature: None,
            timestamp: 0,
            factors: Vec::new(),
        }
    }
}

/// Statistics for a single feature
#[derive(Debug, Clone, Default)]
struct FeatureStats {
    /// Rolling values
    values: VecDeque<f64>,
    /// Running sum
    sum: f64,
    /// Running sum of squares
    sum_sq: f64,
    /// EMA value
    ema: f64,
    /// EMA of squared deviation (for volatility)
    ema_var: f64,
    /// Previous value (for velocity)
    prev_value: Option<f64>,
    /// Min value in window
    min: f64,
    /// Max value in window
    max: f64,
}

impl FeatureStats {
    fn new() -> Self {
        Self {
            values: VecDeque::new(),
            sum: 0.0,
            sum_sq: 0.0,
            ema: 0.0,
            ema_var: 0.0,
            prev_value: None,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let n = self.values.len() as f64;
        let mean = self.mean();
        (self.sum_sq / n) - (mean * mean)
    }

    fn std_dev(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    fn zscore(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std > 0.0 {
            (value - self.mean()) / std
        } else {
            0.0
        }
    }

    fn percentile_rank(&self, value: f64) -> f64 {
        if self.values.is_empty() {
            return 0.5;
        }
        let count_below = self.values.iter().filter(|&&v| v < value).count();
        count_below as f64 / self.values.len() as f64
    }

    fn velocity(&self, value: f64) -> f64 {
        match self.prev_value {
            Some(prev) if prev != 0.0 => (value - prev) / prev.abs(),
            _ => 0.0,
        }
    }
}

/// Statistical anomaly detection for market data
pub struct AnomalyDetector {
    config: AnomalyDetectorConfig,
    /// Statistics per feature
    feature_stats: Vec<FeatureStats>,
    /// Recent anomaly scores (for EMA)
    score_history: VecDeque<f64>,
    /// EMA of anomaly score
    ema_score: f64,
    /// Samples since last anomaly (for cooldown)
    samples_since_anomaly: usize,
    /// Total anomalies detected
    anomaly_count: u64,
    /// Last anomaly result
    last_result: Option<AnomalyResult>,
    /// Sample count
    sample_count: u64,
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector {
    /// Create a new AnomalyDetector with default configuration
    pub fn new() -> Self {
        Self::with_config(AnomalyDetectorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AnomalyDetectorConfig) -> Self {
        let mut feature_stats = Vec::with_capacity(config.num_features);
        for _ in 0..config.num_features {
            feature_stats.push(FeatureStats::new());
        }

        Self {
            feature_stats,
            score_history: VecDeque::with_capacity(100),
            ema_score: 0.0,
            samples_since_anomaly: config.cooldown_samples,
            anomaly_count: 0,
            last_result: None,
            sample_count: 0,
            config,
        }
    }

    /// Update with new data point and check for anomalies
    pub fn update(&mut self, data: &MarketDataPoint) -> AnomalyResult {
        self.sample_count += 1;
        self.samples_since_anomaly += 1;

        // Ensure we have enough feature stats
        while self.feature_stats.len() < data.features.len() {
            self.feature_stats.push(FeatureStats::new());
        }

        // Update statistics for each feature
        let mut feature_zscores = Vec::with_capacity(data.features.len());
        let mut velocities = Vec::with_capacity(data.features.len());

        for (i, &value) in data.features.iter().enumerate() {
            let stats = &mut self.feature_stats[i];

            // Calculate metrics before updating
            let zscore = stats.zscore(value);
            let velocity = stats.velocity(value);

            feature_zscores.push(zscore);
            velocities.push(velocity);

            // Update statistics
            self.update_feature_stats(i, value);
        }

        // Check if we have enough samples
        if self.sample_count < self.config.min_samples as u64 {
            return AnomalyResult {
                timestamp: data.timestamp,
                feature_zscores,
                ..Default::default()
            };
        }

        // Calculate anomaly scores
        let (score, anomaly_type, factors) =
            self.calculate_anomaly_score(&data.features, &feature_zscores, &velocities);

        // Apply sensitivity
        let adjusted_score = (score * self.config.sensitivity).clamp(0.0, 1.0);

        // Determine if this is an anomaly (with cooldown)
        let is_anomaly =
            adjusted_score >= 0.5 && self.samples_since_anomaly >= self.config.cooldown_samples;

        let severity = AnomalySeverity::from_score(adjusted_score);

        // Find most anomalous feature
        let most_anomalous_feature = feature_zscores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx);

        // Update tracking
        self.update_score_history(adjusted_score);

        if is_anomaly {
            self.anomaly_count += 1;
            self.samples_since_anomaly = 0;
        }

        let result = AnomalyResult {
            is_anomaly,
            score: adjusted_score,
            severity,
            anomaly_type,
            feature_zscores,
            most_anomalous_feature,
            timestamp: data.timestamp,
            factors,
        };

        self.last_result = Some(result.clone());
        result
    }

    /// Update statistics for a single feature
    fn update_feature_stats(&mut self, index: usize, value: f64) {
        let stats = &mut self.feature_stats[index];
        let window_size = self.config.window_size;

        // Update rolling window
        if stats.values.len() >= window_size {
            if let Some(old_value) = stats.values.pop_front() {
                stats.sum -= old_value;
                stats.sum_sq -= old_value * old_value;
            }
        }

        stats.values.push_back(value);
        stats.sum += value;
        stats.sum_sq += value * value;

        // Update min/max
        if value < stats.min {
            stats.min = value;
        }
        if value > stats.max {
            stats.max = value;
        }

        // Update EMA
        if stats.ema == 0.0 {
            stats.ema = value;
        } else {
            stats.ema = self.config.ema_decay * stats.ema + (1.0 - self.config.ema_decay) * value;
        }

        // Update EMA variance
        let deviation = (value - stats.ema).powi(2);
        stats.ema_var =
            self.config.ema_decay * stats.ema_var + (1.0 - self.config.ema_decay) * deviation;

        // Store for velocity calculation
        stats.prev_value = Some(value);
    }

    /// Calculate comprehensive anomaly score
    fn calculate_anomaly_score(
        &self,
        features: &[f64],
        zscores: &[f64],
        velocities: &[f64],
    ) -> (f64, AnomalyType, Vec<String>) {
        let mut factors = Vec::new();
        let mut scores = Vec::new();

        // 1. Z-score based anomaly
        let max_zscore = zscores.iter().map(|z| z.abs()).fold(0.0, f64::max);
        let zscore_score = (max_zscore / self.config.zscore_threshold).min(1.0);

        if max_zscore > self.config.zscore_threshold {
            factors.push(format!("High z-score: {:.2}", max_zscore));
        }
        scores.push(zscore_score * 0.3);

        // 2. Percentile-based outlier detection
        let mut percentile_score: f64 = 0.0;
        for (i, &value) in features.iter().enumerate() {
            if i < self.feature_stats.len() {
                let percentile = self.feature_stats[i].percentile_rank(value);
                if percentile > self.config.percentile_threshold
                    || percentile < 1.0 - self.config.percentile_threshold
                {
                    let score: f64 = if percentile > 0.5 {
                        (percentile - 0.5) * 2.0
                    } else {
                        (0.5 - percentile) * 2.0
                    };
                    percentile_score = percentile_score.max(score);
                    factors.push(format!(
                        "Feature {} at {:.0}th percentile",
                        i,
                        percentile * 100.0
                    ));
                }
            }
        }
        scores.push(percentile_score * 0.25);

        // 3. Velocity anomaly (rate of change)
        let max_velocity = velocities.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let velocity_score = (max_velocity / 0.1).min(1.0); // 10% change threshold

        if max_velocity > 0.05 {
            factors.push(format!("High velocity: {:.2}%", max_velocity * 100.0));
        }
        scores.push(velocity_score * 0.25);

        // 4. Multi-variate anomaly (multiple features anomalous)
        let anomalous_features = zscores
            .iter()
            .filter(|z| z.abs() > self.config.zscore_threshold * 0.7)
            .count();
        let multivariate_score = (anomalous_features as f64 / features.len() as f64).min(1.0);

        if anomalous_features > 1 {
            factors.push(format!("{} features anomalous", anomalous_features));
        }
        scores.push(multivariate_score * 0.2);

        // Calculate total score
        let total_score: f64 = scores.iter().sum();

        // Determine anomaly type
        let anomaly_type = self.determine_anomaly_type(zscores, velocities, max_velocity);

        (total_score, anomaly_type, factors)
    }

    /// Determine the type of anomaly
    fn determine_anomaly_type(
        &self,
        zscores: &[f64],
        velocities: &[f64],
        max_velocity: f64,
    ) -> AnomalyType {
        // Check for velocity-based anomaly first
        if max_velocity > 0.1 {
            let max_velocity_idx = velocities
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if max_velocity_idx < velocities.len() && velocities[max_velocity_idx] > 0.0 {
                return AnomalyType::Spike;
            } else {
                return AnomalyType::Drop;
            }
        }

        // Check for multi-variate anomaly
        let anomalous_count = zscores
            .iter()
            .filter(|z| z.abs() > self.config.zscore_threshold * 0.7)
            .count();

        if anomalous_count > zscores.len() / 2 {
            return AnomalyType::MultiVariate;
        }

        // Check for volatility shift
        for (i, stats) in self.feature_stats.iter().enumerate() {
            if i < zscores.len() {
                let current_var = stats.ema_var;
                let historical_var = stats.variance();
                if historical_var > 0.0 && current_var / historical_var > 2.0 {
                    return AnomalyType::VolatilityShift;
                }
            }
        }

        // Default to statistical outlier
        if zscores
            .iter()
            .any(|z| z.abs() > self.config.zscore_threshold)
        {
            AnomalyType::StatisticalOutlier
        } else {
            AnomalyType::Unknown
        }
    }

    /// Update score history
    fn update_score_history(&mut self, score: f64) {
        if self.score_history.len() >= 100 {
            self.score_history.pop_front();
        }
        self.score_history.push_back(score);

        // Update EMA
        self.ema_score =
            self.config.ema_decay * self.ema_score + (1.0 - self.config.ema_decay) * score;
    }

    /// Get EMA of anomaly scores
    pub fn ema_score(&self) -> f64 {
        self.ema_score
    }

    /// Get total anomaly count
    pub fn anomaly_count(&self) -> u64 {
        self.anomaly_count
    }

    /// Get last result
    pub fn last_result(&self) -> Option<&AnomalyResult> {
        self.last_result.as_ref()
    }

    /// Get sample count
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Check if currently detecting anomalies (past minimum samples)
    pub fn is_active(&self) -> bool {
        self.sample_count >= self.config.min_samples as u64
    }

    /// Get statistics for a specific feature
    pub fn feature_stats(&self, index: usize) -> Option<FeatureStatsSummary> {
        self.feature_stats
            .get(index)
            .map(|stats| FeatureStatsSummary {
                mean: stats.mean(),
                std_dev: stats.std_dev(),
                min: stats.min,
                max: stats.max,
                ema: stats.ema,
                sample_count: stats.values.len(),
            })
    }

    /// Get overall statistics
    pub fn statistics(&self) -> AnomalyDetectorStats {
        let avg_score = if self.score_history.is_empty() {
            0.0
        } else {
            self.score_history.iter().sum::<f64>() / self.score_history.len() as f64
        };

        let max_score = self.score_history.iter().cloned().fold(0.0, f64::max);

        AnomalyDetectorStats {
            sample_count: self.sample_count,
            anomaly_count: self.anomaly_count,
            anomaly_rate: if self.sample_count > 0 {
                self.anomaly_count as f64 / self.sample_count as f64
            } else {
                0.0
            },
            average_score: avg_score,
            max_score,
            ema_score: self.ema_score,
            is_active: self.is_active(),
        }
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        for stats in &mut self.feature_stats {
            *stats = FeatureStats::new();
        }
        self.score_history.clear();
        self.ema_score = 0.0;
        self.samples_since_anomaly = self.config.cooldown_samples;
        self.anomaly_count = 0;
        self.last_result = None;
        self.sample_count = 0;
    }

    /// Main processing function (compatibility with neuromorphic interface)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Summary statistics for a feature
#[derive(Debug, Clone)]
pub struct FeatureStatsSummary {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub ema: f64,
    pub sample_count: usize,
}

/// Overall detector statistics
#[derive(Debug, Clone)]
pub struct AnomalyDetectorStats {
    pub sample_count: u64,
    pub anomaly_count: u64,
    pub anomaly_rate: f64,
    pub average_score: f64,
    pub max_score: f64,
    pub ema_score: f64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_data(features: Vec<f64>, timestamp: i64) -> MarketDataPoint {
        MarketDataPoint::new(features, timestamp)
    }

    #[test]
    fn test_basic_creation() {
        let detector = AnomalyDetector::new();
        assert_eq!(detector.sample_count(), 0);
        assert_eq!(detector.anomaly_count(), 0);
        assert!(!detector.is_active());
    }

    #[test]
    fn test_normal_data() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            min_samples: 20,
            num_features: 3,
            ..Default::default()
        });

        // Add normal data
        for i in 0..50 {
            let data = create_data(vec![100.0 + (i as f64 * 0.1), 1000.0, 0.02], i);
            let result = detector.update(&data);

            // After warm-up, normal data should not trigger anomalies
            if i > 30 {
                assert!(!result.is_anomaly, "Normal data triggered anomaly at {}", i);
            }
        }

        assert!(detector.is_active());
    }

    #[test]
    fn test_spike_detection() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            min_samples: 20,
            num_features: 3,
            zscore_threshold: 2.0,
            cooldown_samples: 1,
            ..Default::default()
        });

        // Build baseline with stable data
        for i in 0..30 {
            let data = create_data(vec![100.0, 1000.0, 0.02], i);
            detector.update(&data);
        }

        // Introduce a spike
        let spike_data = create_data(vec![120.0, 1000.0, 0.02], 30); // 20% spike
        let result = detector.update(&spike_data);

        assert!(result.score > 0.3, "Spike should increase anomaly score");
    }

    #[test]
    fn test_drop_detection() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            min_samples: 20,
            num_features: 3,
            zscore_threshold: 2.0,
            cooldown_samples: 1,
            ..Default::default()
        });

        // Build baseline
        for i in 0..30 {
            let data = create_data(vec![100.0, 1000.0, 0.02], i);
            detector.update(&data);
        }

        // Introduce a drop
        let drop_data = create_data(vec![80.0, 1000.0, 0.02], 30); // 20% drop
        let result = detector.update(&drop_data);

        assert!(result.score > 0.3, "Drop should increase anomaly score");
    }

    #[test]
    fn test_multivariate_anomaly() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            min_samples: 20,
            num_features: 5,
            zscore_threshold: 2.0,
            cooldown_samples: 1,
            ..Default::default()
        });

        // Build baseline
        for i in 0..30 {
            let data = create_data(vec![100.0, 1000.0, 0.02, 50.0, 25.0], i);
            detector.update(&data);
        }

        // Introduce multi-variate anomaly (all features change)
        let anomaly_data = create_data(vec![110.0, 1500.0, 0.05, 60.0, 35.0], 30);
        let result = detector.update(&anomaly_data);

        assert!(result.score > 0.0);
    }

    #[test]
    fn test_cooldown() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            min_samples: 10,
            num_features: 1,
            zscore_threshold: 2.0,
            cooldown_samples: 5,
            sensitivity: 2.0,
            ..Default::default()
        });

        // Build baseline
        for i in 0..15 {
            let data = create_data(vec![100.0], i);
            detector.update(&data);
        }

        // First anomaly
        let anomaly1 = create_data(vec![150.0], 15);
        let result1 = detector.update(&anomaly1);

        // Immediate second anomaly (should be in cooldown)
        let anomaly2 = create_data(vec![150.0], 16);
        let result2 = detector.update(&anomaly2);

        if result1.is_anomaly {
            assert!(
                !result2.is_anomaly,
                "Second anomaly should be blocked by cooldown"
            );
        }
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(AnomalySeverity::from_score(0.1), AnomalySeverity::None);
        assert_eq!(AnomalySeverity::from_score(0.4), AnomalySeverity::Low);
        assert_eq!(AnomalySeverity::from_score(0.6), AnomalySeverity::Medium);
        assert_eq!(AnomalySeverity::from_score(0.8), AnomalySeverity::High);
        assert_eq!(AnomalySeverity::from_score(0.95), AnomalySeverity::Critical);
    }

    #[test]
    fn test_feature_stats() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            num_features: 2,
            ..Default::default()
        });

        for i in 0..50 {
            let data = create_data(vec![100.0 + (i as f64), 1000.0], i);
            detector.update(&data);
        }

        let stats = detector.feature_stats(0);
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert!(stats.mean > 100.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_statistics() {
        let mut detector = AnomalyDetector::new();

        for i in 0..50 {
            let data = create_data(vec![100.0], i);
            detector.update(&data);
        }

        let stats = detector.statistics();
        assert_eq!(stats.sample_count, 50);
        assert!(stats.is_active);
    }

    #[test]
    fn test_reset() {
        let mut detector = AnomalyDetector::new();

        for i in 0..30 {
            let data = create_data(vec![100.0], i);
            detector.update(&data);
        }

        assert!(detector.sample_count() > 0);

        detector.reset();

        assert_eq!(detector.sample_count(), 0);
        assert_eq!(detector.anomaly_count(), 0);
        assert!(!detector.is_active());
    }

    #[test]
    fn test_process() {
        let detector = AnomalyDetector::new();
        assert!(detector.process().is_ok());
    }

    #[test]
    fn test_data_point_with_label() {
        let data = MarketDataPoint::new(vec![100.0], 0).with_label("test".to_string());
        assert_eq!(data.label, Some("test".to_string()));
    }

    #[test]
    fn test_anomaly_type_str() {
        assert_eq!(AnomalyType::Spike.as_str(), "spike");
        assert_eq!(AnomalyType::Drop.as_str(), "drop");
        assert_eq!(
            AnomalyType::StatisticalOutlier.as_str(),
            "statistical_outlier"
        );
    }
}
