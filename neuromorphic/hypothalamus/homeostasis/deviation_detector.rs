//! Detect deviations from target
//!
//! Part of the Hypothalamus region
//! Component: homeostasis
//!
//! This module implements multi-metric deviation detection using statistical
//! analysis to identify when portfolio metrics deviate significantly from
//! their target values or historical norms.

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

/// Type of deviation detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviationType {
    /// Above upper threshold
    UpperBreach,
    /// Below lower threshold
    LowerBreach,
    /// Rapid change (rate of change too high)
    RapidChange,
    /// Sustained deviation over time
    Sustained,
    /// Statistical outlier (beyond n standard deviations)
    Outlier,
    /// Trend deviation (moving away from target)
    TrendDeviation,
}

/// Severity of deviation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviationSeverity {
    /// Within normal range
    Normal,
    /// Minor deviation - monitoring needed
    Minor,
    /// Moderate deviation - attention needed
    Moderate,
    /// Major deviation - action needed
    Major,
    /// Critical deviation - immediate action required
    Critical,
}

impl DeviationSeverity {
    /// Get numeric value for calculations
    pub fn value(&self) -> f64 {
        match self {
            DeviationSeverity::Normal => 0.0,
            DeviationSeverity::Minor => 0.25,
            DeviationSeverity::Moderate => 0.50,
            DeviationSeverity::Major => 0.75,
            DeviationSeverity::Critical => 1.0,
        }
    }
}

/// Configuration for a tracked metric
#[derive(Debug, Clone)]
pub struct MetricConfig {
    /// Metric name
    pub name: String,
    /// Target value
    pub target: f64,
    /// Upper threshold (absolute)
    pub upper_threshold: f64,
    /// Lower threshold (absolute)
    pub lower_threshold: f64,
    /// Warning threshold as percentage of limit
    pub warning_pct: f64,
    /// Number of standard deviations for outlier detection
    pub outlier_stddev: f64,
    /// Maximum rate of change per period
    pub max_rate_of_change: f64,
    /// Sustained deviation periods before alert
    pub sustained_periods: usize,
    /// Enable trend analysis
    pub track_trend: bool,
    /// History window size
    pub history_size: usize,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            target: 0.0,
            upper_threshold: f64::INFINITY,
            lower_threshold: f64::NEG_INFINITY,
            warning_pct: 0.80,
            outlier_stddev: 3.0,
            max_rate_of_change: f64::INFINITY,
            sustained_periods: 5,
            track_trend: true,
            history_size: 100,
        }
    }
}

impl MetricConfig {
    pub fn new(name: &str, target: f64) -> Self {
        Self {
            name: name.to_string(),
            target,
            ..Default::default()
        }
    }

    pub fn with_thresholds(mut self, lower: f64, upper: f64) -> Self {
        self.lower_threshold = lower;
        self.upper_threshold = upper;
        self
    }

    pub fn with_symmetric_threshold(mut self, threshold: f64) -> Self {
        self.upper_threshold = self.target + threshold;
        self.lower_threshold = self.target - threshold;
        self
    }

    pub fn with_percentage_threshold(mut self, pct: f64) -> Self {
        let deviation = self.target.abs() * pct;
        self.upper_threshold = self.target + deviation;
        self.lower_threshold = self.target - deviation;
        self
    }
}

/// State for a tracked metric
#[derive(Debug, Clone)]
pub struct MetricState {
    /// Current value
    pub current_value: f64,
    /// Historical values
    pub history: VecDeque<f64>,
    /// Historical timestamps
    pub timestamps: VecDeque<u64>,
    /// Running mean
    pub mean: f64,
    /// Running variance
    pub variance: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Consecutive periods above/below threshold
    pub consecutive_deviations: usize,
    /// Last deviation direction (positive = above target)
    pub last_deviation_sign: i8,
    /// Trend slope (rate of change)
    pub trend_slope: f64,
    /// Last update timestamp
    pub last_update: u64,
}

impl Default for MetricState {
    fn default() -> Self {
        Self {
            current_value: 0.0,
            history: VecDeque::new(),
            timestamps: VecDeque::new(),
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            consecutive_deviations: 0,
            last_deviation_sign: 0,
            trend_slope: 0.0,
            last_update: 0,
        }
    }
}

impl MetricState {
    /// Update statistics with new value
    /// Returns (previous_mean, previous_std_dev) for outlier detection
    pub fn update(&mut self, value: f64, timestamp: u64, max_history: usize) -> (f64, f64) {
        // Save previous statistics for outlier detection
        let prev_mean = self.mean;
        let prev_std_dev = self.std_dev;

        self.current_value = value;
        self.last_update = timestamp;

        // Add to history
        self.history.push_back(value);
        self.timestamps.push_back(timestamp);

        // Trim history
        while self.history.len() > max_history {
            self.history.pop_front();
            self.timestamps.pop_front();
        }

        // Update statistics
        if !self.history.is_empty() {
            let n = self.history.len() as f64;
            self.mean = self.history.iter().sum::<f64>() / n;

            if self.history.len() > 1 {
                self.variance = self
                    .history
                    .iter()
                    .map(|v| (v - self.mean).powi(2))
                    .sum::<f64>()
                    / (n - 1.0);
                self.std_dev = self.variance.sqrt();
            }

            // Calculate trend if enough data
            if self.history.len() >= 3 {
                self.trend_slope = self.calculate_trend();
            }
        }

        (prev_mean, prev_std_dev)
    }

    /// Calculate linear trend slope using least squares
    fn calculate_trend(&self) -> f64 {
        let n = self.history.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let x_mean = (n - 1.0) / 2.0;
        let y_mean = self.mean;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in self.history.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Get z-score for current value
    pub fn z_score(&self) -> f64 {
        if self.std_dev < 1e-10 {
            0.0
        } else {
            (self.current_value - self.mean) / self.std_dev
        }
    }

    /// Get z-score for current value using previous statistics (for outlier detection)
    pub fn z_score_vs_previous(&self, prev_mean: f64, prev_std_dev: f64) -> f64 {
        if prev_std_dev < 1e-10 {
            0.0
        } else {
            (self.current_value - prev_mean) / prev_std_dev
        }
    }

    /// Get rate of change from previous value
    pub fn rate_of_change(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let prev = self
            .history
            .get(self.history.len() - 2)
            .copied()
            .unwrap_or(0.0);
        if prev.abs() < 1e-10 {
            0.0
        } else {
            (self.current_value - prev) / prev.abs()
        }
    }
}

/// A detected deviation event
#[derive(Debug, Clone)]
pub struct DeviationEvent {
    /// Metric name
    pub metric: String,
    /// Deviation type
    pub deviation_type: DeviationType,
    /// Severity level
    pub severity: DeviationSeverity,
    /// Current value
    pub current_value: f64,
    /// Target value
    pub target_value: f64,
    /// Threshold that was breached (if applicable)
    pub threshold: Option<f64>,
    /// Deviation amount (absolute)
    pub deviation_amount: f64,
    /// Deviation percentage from target
    pub deviation_pct: f64,
    /// Z-score if applicable
    pub z_score: Option<f64>,
    /// Timestamp
    pub timestamp: u64,
    /// Description
    pub description: String,
}

/// Deviation detector configuration
#[derive(Debug, Clone)]
pub struct DeviationDetectorConfig {
    /// Default outlier threshold in standard deviations
    pub default_outlier_stddev: f64,
    /// Enable automatic severity escalation
    pub auto_escalate: bool,
    /// Periods before escalation
    pub escalation_periods: usize,
    /// Maximum alerts per metric per period
    pub max_alerts_per_metric: usize,
    /// Alert cooldown in milliseconds
    pub alert_cooldown_ms: u64,
}

impl Default for DeviationDetectorConfig {
    fn default() -> Self {
        Self {
            default_outlier_stddev: 3.0,
            auto_escalate: true,
            escalation_periods: 3,
            max_alerts_per_metric: 10,
            alert_cooldown_ms: 60_000,
        }
    }
}

/// Detect deviations from target
pub struct DeviationDetector {
    /// Configuration
    config: DeviationDetectorConfig,
    /// Metric configurations
    metric_configs: HashMap<String, MetricConfig>,
    /// Metric states
    metric_states: HashMap<String, MetricState>,
    /// Recent deviation events
    events: VecDeque<DeviationEvent>,
    /// Alert counts per metric
    alert_counts: HashMap<String, usize>,
    /// Last alert time per metric
    last_alert_times: HashMap<String, u64>,
    /// Maximum events to store
    max_events: usize,
}

impl Default for DeviationDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviationDetector {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_config(DeviationDetectorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DeviationDetectorConfig) -> Self {
        Self {
            config,
            metric_configs: HashMap::new(),
            metric_states: HashMap::new(),
            events: VecDeque::new(),
            alert_counts: HashMap::new(),
            last_alert_times: HashMap::new(),
            max_events: 1000,
        }
    }

    /// Register a metric for tracking
    pub fn register_metric(&mut self, config: MetricConfig) {
        let name = config.name.clone();
        self.metric_configs.insert(name.clone(), config);
        self.metric_states
            .insert(name.clone(), MetricState::default());
        self.alert_counts.insert(name.clone(), 0);
    }

    /// Update a metric value and check for deviations
    pub fn update(&mut self, metric_name: &str, value: f64, timestamp: u64) -> Vec<DeviationEvent> {
        let config = match self.metric_configs.get(metric_name) {
            Some(c) => c.clone(),
            None => return Vec::new(),
        };

        // Update state and get previous statistics for outlier detection
        // Only use prev_stats for outlier detection if we have enough history
        let prev_stats = if let Some(state) = self.metric_states.get_mut(metric_name) {
            let history_len = state.history.len();
            let (prev_mean, prev_std_dev) = state.update(value, timestamp, config.history_size);
            // Only use previous stats if we had sufficient history before this update
            if history_len >= 10 && prev_std_dev > 1e-10 {
                Some((prev_mean, prev_std_dev))
            } else {
                None
            }
        } else {
            None
        };

        // Check for deviations
        self.detect_deviations(metric_name, timestamp, prev_stats)
    }

    /// Detect deviations for a metric
    fn detect_deviations(
        &mut self,
        metric_name: &str,
        timestamp: u64,
        prev_stats: Option<(f64, f64)>,
    ) -> Vec<DeviationEvent> {
        let mut events = Vec::new();

        let config = match self.metric_configs.get(metric_name) {
            Some(c) => c.clone(),
            None => return events,
        };

        // Extract all needed state data first to avoid borrow conflicts
        let (
            value,
            _z_score,
            z_score_prev,
            roc,
            history_len,
            trend_slope,
            last_deviation_sign,
            consecutive_deviations,
        ): (f64, f64, f64, f64, usize, f64, i8, usize) = {
            let state = match self.metric_states.get(metric_name) {
                Some(s) => s,
                None => return events,
            };
            let z_prev = if let Some((prev_mean, prev_std_dev)) = prev_stats {
                state.z_score_vs_previous(prev_mean, prev_std_dev)
            } else {
                state.z_score()
            };
            (
                state.current_value,
                state.z_score(),
                z_prev,
                state.rate_of_change(),
                state.history.len(),
                state.trend_slope,
                state.last_deviation_sign,
                state.consecutive_deviations,
            )
        };

        let target = config.target;

        // Check cooldown
        if let Some(&last_time) = self.last_alert_times.get(metric_name) {
            if timestamp.saturating_sub(last_time) < self.config.alert_cooldown_ms {
                return events;
            }
        }

        // Check upper threshold breach
        if value > config.upper_threshold {
            let event = self.create_deviation_event_simple(
                metric_name,
                DeviationType::UpperBreach,
                value,
                target,
                Some(config.upper_threshold),
                timestamp,
                consecutive_deviations,
            );
            events.push(event);
        }

        // Check lower threshold breach
        if value < config.lower_threshold {
            let event = self.create_deviation_event_simple(
                metric_name,
                DeviationType::LowerBreach,
                value,
                target,
                Some(config.lower_threshold),
                timestamp,
                consecutive_deviations,
            );
            events.push(event);
        }

        // Check for statistical outlier (use z-score vs previous statistics)
        if history_len >= 10 {
            if z_score_prev.abs() > config.outlier_stddev {
                let mut event = self.create_deviation_event_simple(
                    metric_name,
                    DeviationType::Outlier,
                    value,
                    target,
                    None,
                    timestamp,
                    consecutive_deviations,
                );
                event.z_score = Some(z_score_prev);
                event.description = format!(
                    "{} is a statistical outlier: z-score {:.2} exceeds {:.1} std devs",
                    metric_name, z_score_prev, config.outlier_stddev
                );
                events.push(event);
            }
        }

        // Check rate of change
        if roc.abs() > config.max_rate_of_change {
            let event = DeviationEvent {
                metric: metric_name.to_string(),
                deviation_type: DeviationType::RapidChange,
                severity: if roc.abs() > config.max_rate_of_change * 2.0 {
                    DeviationSeverity::Major
                } else {
                    DeviationSeverity::Moderate
                },
                current_value: value,
                target_value: target,
                threshold: Some(config.max_rate_of_change),
                deviation_amount: roc.abs(),
                deviation_pct: roc * 100.0,
                z_score: None,
                timestamp,
                description: format!(
                    "{} changing too rapidly: {:.2}% change exceeds {:.2}% limit",
                    metric_name,
                    roc * 100.0,
                    config.max_rate_of_change * 100.0
                ),
            };
            events.push(event);
        }

        // Check for sustained deviation and update state
        let deviation_sign = if value > target {
            1i8
        } else if value < target {
            -1i8
        } else {
            0i8
        };

        let new_consecutive: usize = if deviation_sign == last_deviation_sign && deviation_sign != 0
        {
            consecutive_deviations + 1
        } else {
            1
        };

        // Update state with new deviation tracking
        if let Some(state) = self.metric_states.get_mut(metric_name) {
            state.consecutive_deviations = new_consecutive;
            state.last_deviation_sign = deviation_sign;
        }

        if new_consecutive >= config.sustained_periods {
            let event = self.create_deviation_event_simple(
                metric_name,
                DeviationType::Sustained,
                value,
                target,
                None,
                timestamp,
                new_consecutive,
            );
            events.push(event);
        }

        // Check trend deviation
        if config.track_trend && history_len >= 10 {
            let trend_direction = if trend_slope > 0.0 { 1 } else { -1 };
            let target_direction = if value < target { 1 } else { -1 };

            // Alert if trending away from target
            if trend_direction != target_direction && trend_slope.abs() > 0.01 {
                let event = DeviationEvent {
                    metric: metric_name.to_string(),
                    deviation_type: DeviationType::TrendDeviation,
                    severity: DeviationSeverity::Minor,
                    current_value: value,
                    target_value: target,
                    threshold: None,
                    deviation_amount: trend_slope.abs(),
                    deviation_pct: (value - target) / target.abs().max(1.0) * 100.0,
                    z_score: None,
                    timestamp,
                    description: format!(
                        "{} trending away from target: slope {:.4}",
                        metric_name, trend_slope
                    ),
                };
                events.push(event);
            }
        }

        // Record events and update tracking
        if !events.is_empty() {
            self.last_alert_times
                .insert(metric_name.to_string(), timestamp);
            *self
                .alert_counts
                .entry(metric_name.to_string())
                .or_insert(0) += events.len();

            for event in &events {
                self.events.push_back(event.clone());
            }

            while self.events.len() > self.max_events {
                self.events.pop_front();
            }
        }

        events
    }

    /// Create a deviation event (simplified version that takes consecutive_deviations directly)
    fn create_deviation_event_simple(
        &self,
        metric_name: &str,
        deviation_type: DeviationType,
        value: f64,
        target: f64,
        threshold: Option<f64>,
        timestamp: u64,
        consecutive_deviations: usize,
    ) -> DeviationEvent {
        let deviation_amount = (value - target).abs();
        let deviation_pct = if target.abs() > 1e-10 {
            (value - target) / target.abs() * 100.0
        } else {
            0.0
        };

        let severity = self.calculate_severity(deviation_pct.abs(), consecutive_deviations);

        let description = match deviation_type {
            DeviationType::UpperBreach => format!(
                "{} ({:.4}) exceeds upper threshold ({:.4})",
                metric_name,
                value,
                threshold.unwrap_or(0.0)
            ),
            DeviationType::LowerBreach => format!(
                "{} ({:.4}) below lower threshold ({:.4})",
                metric_name,
                value,
                threshold.unwrap_or(0.0)
            ),
            DeviationType::Sustained => format!(
                "{} sustained deviation for {} periods",
                metric_name, consecutive_deviations
            ),
            DeviationType::Outlier => format!("{} is a statistical outlier", metric_name),
            DeviationType::RapidChange => format!("{} changing rapidly", metric_name),
            DeviationType::TrendDeviation => format!("{} trending away from target", metric_name),
        };

        DeviationEvent {
            metric: metric_name.to_string(),
            deviation_type,
            severity,
            current_value: value,
            target_value: target,
            threshold,
            deviation_amount,
            deviation_pct,
            z_score: None,
            timestamp,
            description,
        }
    }

    /// Calculate severity based on deviation magnitude
    fn calculate_severity(&self, deviation_pct: f64, consecutive: usize) -> DeviationSeverity {
        let base_severity = if deviation_pct < 5.0 {
            DeviationSeverity::Minor
        } else if deviation_pct < 10.0 {
            DeviationSeverity::Moderate
        } else if deviation_pct < 25.0 {
            DeviationSeverity::Major
        } else {
            DeviationSeverity::Critical
        };

        // Escalate if sustained
        if self.config.auto_escalate && consecutive >= self.config.escalation_periods {
            match base_severity {
                DeviationSeverity::Normal => DeviationSeverity::Minor,
                DeviationSeverity::Minor => DeviationSeverity::Moderate,
                DeviationSeverity::Moderate => DeviationSeverity::Major,
                DeviationSeverity::Major | DeviationSeverity::Critical => {
                    DeviationSeverity::Critical
                }
            }
        } else {
            base_severity
        }
    }

    /// Get current value for a metric
    pub fn get_current_value(&self, metric_name: &str) -> Option<f64> {
        self.metric_states.get(metric_name).map(|s| s.current_value)
    }

    /// Get deviation from target for a metric
    pub fn get_deviation(&self, metric_name: &str) -> Option<(f64, f64)> {
        let config = self.metric_configs.get(metric_name)?;
        let state = self.metric_states.get(metric_name)?;

        let deviation = state.current_value - config.target;
        let deviation_pct = if config.target.abs() > 1e-10 {
            deviation / config.target.abs() * 100.0
        } else {
            0.0
        };

        Some((deviation, deviation_pct))
    }

    /// Get recent events
    pub fn get_recent_events(&self, count: usize) -> Vec<&DeviationEvent> {
        self.events.iter().rev().take(count).collect()
    }

    /// Get events for a specific metric
    pub fn get_events_for_metric(&self, metric_name: &str) -> Vec<&DeviationEvent> {
        self.events
            .iter()
            .filter(|e| e.metric == metric_name)
            .collect()
    }

    /// Get events above a severity threshold
    pub fn get_events_by_severity(&self, min_severity: DeviationSeverity) -> Vec<&DeviationEvent> {
        self.events
            .iter()
            .filter(|e| e.severity >= min_severity)
            .collect()
    }

    /// Get summary of all metrics
    pub fn get_summary(&self) -> DeviationSummary {
        let mut metrics = Vec::new();

        for (name, config) in &self.metric_configs {
            if let Some(state) = self.metric_states.get(name) {
                let deviation = state.current_value - config.target;
                let deviation_pct = if config.target.abs() > 1e-10 {
                    deviation / config.target.abs() * 100.0
                } else {
                    0.0
                };

                let within_bounds = state.current_value >= config.lower_threshold
                    && state.current_value <= config.upper_threshold;

                metrics.push(MetricSummary {
                    name: name.clone(),
                    current_value: state.current_value,
                    target_value: config.target,
                    deviation,
                    deviation_pct,
                    within_bounds,
                    consecutive_deviations: state.consecutive_deviations,
                    trend_slope: state.trend_slope,
                    z_score: state.z_score(),
                });
            }
        }

        let total_events = self.events.len();
        let critical_events = self
            .events
            .iter()
            .filter(|e| e.severity == DeviationSeverity::Critical)
            .count();

        DeviationSummary {
            metrics,
            total_events,
            critical_events,
            metrics_tracked: self.metric_configs.len(),
        }
    }

    /// Clear alert counts (call periodically)
    pub fn reset_alert_counts(&mut self) {
        for count in self.alert_counts.values_mut() {
            *count = 0;
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Summary of a single metric
#[derive(Debug, Clone)]
pub struct MetricSummary {
    pub name: String,
    pub current_value: f64,
    pub target_value: f64,
    pub deviation: f64,
    pub deviation_pct: f64,
    pub within_bounds: bool,
    pub consecutive_deviations: usize,
    pub trend_slope: f64,
    pub z_score: f64,
}

/// Summary of all tracked metrics
#[derive(Debug, Clone)]
pub struct DeviationSummary {
    pub metrics: Vec<MetricSummary>,
    pub total_events: usize,
    pub critical_events: usize,
    pub metrics_tracked: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = DeviationDetector::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_register_metric() {
        let mut detector = DeviationDetector::new();
        let config = MetricConfig::new("test_metric", 100.0).with_symmetric_threshold(10.0);
        detector.register_metric(config);

        assert!(detector.metric_configs.contains_key("test_metric"));
        assert!(detector.metric_states.contains_key("test_metric"));
    }

    #[test]
    fn test_upper_breach_detection() {
        let mut detector = DeviationDetector::new();
        let config = MetricConfig::new("value", 100.0).with_thresholds(90.0, 110.0);
        detector.register_metric(config);

        let events = detector.update("value", 115.0, 1000);

        assert!(!events.is_empty());
        assert!(
            events
                .iter()
                .any(|e| e.deviation_type == DeviationType::UpperBreach)
        );
    }

    #[test]
    fn test_lower_breach_detection() {
        let mut detector = DeviationDetector::new();
        let config = MetricConfig::new("value", 100.0).with_thresholds(90.0, 110.0);
        detector.register_metric(config);

        let events = detector.update("value", 85.0, 1000);

        assert!(!events.is_empty());
        assert!(
            events
                .iter()
                .any(|e| e.deviation_type == DeviationType::LowerBreach)
        );
    }

    #[test]
    fn test_within_bounds_no_alert() {
        let mut detector = DeviationDetector::new();
        let config = MetricConfig::new("value", 100.0).with_thresholds(90.0, 110.0);
        detector.register_metric(config);

        let events = detector.update("value", 105.0, 1000);

        // Should not have threshold breach events
        assert!(!events.iter().any(|e| matches!(
            e.deviation_type,
            DeviationType::UpperBreach | DeviationType::LowerBreach
        )));
    }

    #[test]
    fn test_outlier_detection() {
        // Use a detector with no cooldown to avoid test timing issues
        let mut detector_config = DeviationDetectorConfig::default();
        detector_config.alert_cooldown_ms = 0; // Disable cooldown for testing
        let mut detector = DeviationDetector::with_config(detector_config);

        let mut config = MetricConfig::new("value", 100.0);
        config.outlier_stddev = 2.0;
        config.upper_threshold = f64::INFINITY;
        config.lower_threshold = f64::NEG_INFINITY;
        config.track_trend = false; // Disable trend detection to avoid triggering other alerts
        detector.register_metric(config);

        // Build up history with stable values (all exactly 100.0 to minimize variance)
        for i in 0..20 {
            detector.update("value", 100.0, i as u64 * 100000); // Large gaps to exceed cooldown
        }

        // Now add an outlier - 200 is way beyond 2 std devs from mean of 100
        let events = detector.update("value", 200.0, 30 * 100000);

        assert!(
            events
                .iter()
                .any(|e| e.deviation_type == DeviationType::Outlier),
            "Expected Outlier event but got: {:?}",
            events
        );
    }

    #[test]
    fn test_rapid_change_detection() {
        let mut detector = DeviationDetector::new();
        let mut config = MetricConfig::new("value", 100.0);
        config.max_rate_of_change = 0.10; // 10% max change
        config.upper_threshold = f64::INFINITY;
        config.lower_threshold = f64::NEG_INFINITY;
        detector.register_metric(config);

        // Initial value
        detector.update("value", 100.0, 1000);

        // 25% change should trigger
        let events = detector.update("value", 125.0, 2000);

        assert!(
            events
                .iter()
                .any(|e| e.deviation_type == DeviationType::RapidChange)
        );
    }

    #[test]
    fn test_sustained_deviation() {
        let mut detector = DeviationDetector::new();
        let mut config = MetricConfig::new("value", 100.0);
        config.sustained_periods = 3;
        config.upper_threshold = f64::INFINITY;
        config.lower_threshold = f64::NEG_INFINITY;
        detector.register_metric(config);

        // Create sustained deviation
        for i in 0..5 {
            let _ = detector.update("value", 110.0, i * 100000);
        }

        let state = detector.metric_states.get("value").unwrap();
        assert!(state.consecutive_deviations >= 3);
    }

    #[test]
    fn test_get_deviation() {
        let mut detector = DeviationDetector::new();
        let config = MetricConfig::new("test", 100.0);
        detector.register_metric(config);

        detector.update("test", 120.0, 1000);

        let (deviation, deviation_pct) = detector.get_deviation("test").unwrap();
        assert!((deviation - 20.0).abs() < 0.01);
        assert!((deviation_pct - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_severity_calculation() {
        let detector = DeviationDetector::new();

        // Minor: < 5%
        assert_eq!(
            detector.calculate_severity(3.0, 0),
            DeviationSeverity::Minor
        );

        // Moderate: 5-10%
        assert_eq!(
            detector.calculate_severity(7.0, 0),
            DeviationSeverity::Moderate
        );

        // Major: 10-25%
        assert_eq!(
            detector.calculate_severity(15.0, 0),
            DeviationSeverity::Major
        );

        // Critical: > 25%
        assert_eq!(
            detector.calculate_severity(30.0, 0),
            DeviationSeverity::Critical
        );
    }

    #[test]
    fn test_severity_escalation() {
        let config = DeviationDetectorConfig {
            auto_escalate: true,
            escalation_periods: 3,
            ..Default::default()
        };
        let detector = DeviationDetector::with_config(config);

        // Minor with escalation should become Moderate
        assert_eq!(
            detector.calculate_severity(3.0, 5),
            DeviationSeverity::Moderate
        );
    }

    #[test]
    fn test_metric_state_statistics() {
        let mut state = MetricState::default();

        let values = [100.0, 102.0, 98.0, 101.0, 99.0];
        for (i, &v) in values.iter().enumerate() {
            state.update(v, i as u64 * 1000, 100);
        }

        assert!((state.mean - 100.0).abs() < 1.0);
        assert!(state.std_dev > 0.0);
        assert_eq!(state.history.len(), 5);
    }

    #[test]
    fn test_z_score_calculation() {
        let mut state = MetricState::default();

        // Add values with known mean and std dev
        for i in 0..20 {
            state.update(100.0, i * 1000, 100);
        }

        // Mean should be 100, std dev should be ~0
        state.update(100.0, 20000, 100);
        assert!(state.z_score().abs() < 0.1);
    }

    #[test]
    fn test_trend_calculation() {
        let mut state = MetricState::default();

        // Create upward trend
        for i in 0..10 {
            state.update(100.0 + i as f64 * 5.0, i as u64 * 1000, 100);
        }

        assert!(state.trend_slope > 0.0);
    }

    #[test]
    fn test_percentage_threshold() {
        let config = MetricConfig::new("test", 100.0).with_percentage_threshold(0.10); // 10%

        assert!((config.upper_threshold - 110.0).abs() < 0.01);
        assert!((config.lower_threshold - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let mut detector = DeviationDetector::new();

        let config1 = MetricConfig::new("metric1", 100.0).with_symmetric_threshold(10.0);
        let config2 = MetricConfig::new("metric2", 50.0).with_symmetric_threshold(5.0);

        detector.register_metric(config1);
        detector.register_metric(config2);

        detector.update("metric1", 105.0, 1000);
        detector.update("metric2", 48.0, 1000);

        let summary = detector.get_summary();

        assert_eq!(summary.metrics_tracked, 2);
        assert_eq!(summary.metrics.len(), 2);
    }

    #[test]
    fn test_cooldown() {
        let config = DeviationDetectorConfig {
            alert_cooldown_ms: 10000,
            ..Default::default()
        };
        let mut detector = DeviationDetector::with_config(config);

        let metric_config = MetricConfig::new("test", 100.0).with_thresholds(90.0, 110.0);
        detector.register_metric(metric_config);

        // First breach should alert
        let events1 = detector.update("test", 120.0, 1000);
        assert!(!events1.is_empty());

        // Second breach within cooldown should not alert
        let events2 = detector.update("test", 125.0, 5000);
        assert!(events2.is_empty());

        // After cooldown should alert again
        let events3 = detector.update("test", 130.0, 15000);
        assert!(!events3.is_empty());
    }

    #[test]
    fn test_get_events_for_metric() {
        let mut detector = DeviationDetector::new();

        let config1 = MetricConfig::new("m1", 100.0).with_thresholds(90.0, 110.0);
        let config2 = MetricConfig::new("m2", 100.0).with_thresholds(90.0, 110.0);

        detector.register_metric(config1);
        detector.register_metric(config2);

        detector.update("m1", 120.0, 1000);
        detector.update("m2", 80.0, 2000);

        let m1_events = detector.get_events_for_metric("m1");
        let m2_events = detector.get_events_for_metric("m2");

        assert!(m1_events.iter().all(|e| e.metric == "m1"));
        assert!(m2_events.iter().all(|e| e.metric == "m2"));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(DeviationSeverity::Critical > DeviationSeverity::Major);
        assert!(DeviationSeverity::Major > DeviationSeverity::Moderate);
        assert!(DeviationSeverity::Moderate > DeviationSeverity::Minor);
        assert!(DeviationSeverity::Minor > DeviationSeverity::Normal);
    }
}
