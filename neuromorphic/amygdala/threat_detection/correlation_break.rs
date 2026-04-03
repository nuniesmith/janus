//! Correlation breakdown detection
//!
//! Part of the Amygdala region
//! Component: threat_detection
//!
//! Detects when historical correlations between assets break down,
//! which can indicate:
//! - Market stress and regime changes
//! - Liquidity crises
//! - Contagion events
//! - Black swan scenarios
//!
//! Methods used:
//! - Rolling correlation monitoring
//! - Correlation change velocity
//! - Multi-asset correlation matrix analysis
//! - DCC-GARCH inspired dynamic correlation

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

/// Configuration for correlation break detection
#[derive(Debug, Clone)]
pub struct CorrelationBreakConfig {
    /// Window size for correlation calculation
    pub correlation_window: usize,
    /// Window size for baseline correlation
    pub baseline_window: usize,
    /// Minimum samples before calculation is valid
    pub min_samples: usize,
    /// Threshold for significant correlation change
    pub change_threshold: f64,
    /// Threshold for correlation break (larger change)
    pub break_threshold: f64,
    /// Velocity threshold (correlation change per period)
    pub velocity_threshold: f64,
    /// EMA decay for smoothing
    pub ema_decay: f64,
}

impl Default for CorrelationBreakConfig {
    fn default() -> Self {
        Self {
            correlation_window: 50,
            baseline_window: 200,
            min_samples: 30,
            change_threshold: 0.3,
            break_threshold: 0.5,
            velocity_threshold: 0.1,
            ema_decay: 0.94,
        }
    }
}

/// Severity of correlation break
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakSeverity {
    /// No significant break
    None,
    /// Minor deviation from normal
    Minor,
    /// Significant correlation change
    Significant,
    /// Major correlation breakdown
    Major,
    /// Critical - correlations completely broken
    Critical,
}

impl BreakSeverity {
    /// Create from correlation change magnitude
    pub fn from_change(change: f64, change_threshold: f64, break_threshold: f64) -> Self {
        let abs_change = change.abs();
        if abs_change >= break_threshold * 1.5 {
            Self::Critical
        } else if abs_change >= break_threshold {
            Self::Major
        } else if abs_change >= change_threshold {
            Self::Significant
        } else if abs_change >= change_threshold * 0.5 {
            Self::Minor
        } else {
            Self::None
        }
    }

    /// Whether this severity requires action
    pub fn requires_action(&self) -> bool {
        matches!(self, Self::Major | Self::Critical)
    }

    /// Risk multiplier for this severity
    pub fn risk_multiplier(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Minor => 1.2,
            Self::Significant => 1.5,
            Self::Major => 2.0,
            Self::Critical => 3.0,
        }
    }
}

/// A pair of assets being monitored
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct AssetPair {
    pub asset_a: String,
    pub asset_b: String,
}

impl AssetPair {
    pub fn new(asset_a: impl Into<String>, asset_b: impl Into<String>) -> Self {
        let a = asset_a.into();
        let b = asset_b.into();
        // Normalize order for consistent hashing
        if a <= b {
            Self {
                asset_a: a,
                asset_b: b,
            }
        } else {
            Self {
                asset_a: b,
                asset_b: a,
            }
        }
    }
}

/// Correlation statistics for a pair
#[derive(Debug, Clone)]
pub struct PairCorrelation {
    /// Asset pair
    pub pair: AssetPair,
    /// Current correlation coefficient (-1 to 1)
    pub current_correlation: f64,
    /// Baseline (historical) correlation
    pub baseline_correlation: f64,
    /// Change from baseline
    pub correlation_change: f64,
    /// Velocity of change (per period)
    pub change_velocity: f64,
    /// Severity of break
    pub severity: BreakSeverity,
    /// Number of samples used
    pub sample_count: usize,
    /// Whether calculation is reliable
    pub is_reliable: bool,
}

/// Result of correlation break detection
#[derive(Debug, Clone)]
pub struct CorrelationBreakDetection {
    /// Whether a break was detected
    pub break_detected: bool,
    /// Overall severity across all pairs
    pub overall_severity: BreakSeverity,
    /// Number of pairs with significant changes
    pub pairs_with_changes: usize,
    /// Number of pairs with breaks
    pub pairs_with_breaks: usize,
    /// Individual pair correlations
    pub pair_correlations: Vec<PairCorrelation>,
    /// Average correlation change magnitude
    pub avg_change_magnitude: f64,
    /// Maximum correlation change
    pub max_change: f64,
    /// Timestamp of detection
    pub timestamp: i64,
}

/// Returns data for an asset
#[derive(Debug, Clone)]
pub struct AssetReturn {
    /// Asset identifier
    pub asset: String,
    /// Return value
    pub return_value: f64,
    /// Timestamp
    pub timestamp: i64,
}

/// Internal state for tracking an asset's returns
#[derive(Debug, Clone)]
struct AssetState {
    returns: VecDeque<f64>,
    ema_return: f64,
}

impl AssetState {
    fn new(capacity: usize) -> Self {
        Self {
            returns: VecDeque::with_capacity(capacity),
            ema_return: 0.0,
        }
    }

    fn add_return(&mut self, ret: f64, window: usize, ema_decay: f64) {
        self.returns.push_back(ret);
        while self.returns.len() > window {
            self.returns.pop_front();
        }

        // Update EMA
        if self.ema_return == 0.0 {
            self.ema_return = ret;
        } else {
            self.ema_return = ema_decay * self.ema_return + (1.0 - ema_decay) * ret;
        }
    }
}

/// Correlation breakdown detector
///
/// Monitors correlations between assets and detects when they
/// break down, which can signal market stress or regime changes.
pub struct CorrelationBreak {
    config: CorrelationBreakConfig,
    /// Per-asset return data
    asset_states: HashMap<String, AssetState>,
    /// Historical correlations for each pair
    pair_correlations: HashMap<AssetPair, VecDeque<f64>>,
    /// Baseline correlations
    baseline_correlations: HashMap<AssetPair, f64>,
    /// Previous correlation for velocity calculation
    previous_correlations: HashMap<AssetPair, f64>,
    /// Last detection result
    last_detection: Option<CorrelationBreakDetection>,
    /// Total breaks detected
    total_breaks: usize,
}

impl Default for CorrelationBreak {
    fn default() -> Self {
        Self::new()
    }
}

impl CorrelationBreak {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(CorrelationBreakConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CorrelationBreakConfig) -> Self {
        Self {
            config,
            asset_states: HashMap::new(),
            pair_correlations: HashMap::new(),
            baseline_correlations: HashMap::new(),
            previous_correlations: HashMap::new(),
            last_detection: None,
            total_breaks: 0,
        }
    }

    /// Update with new returns for multiple assets
    pub fn update(&mut self, returns: &[AssetReturn]) -> CorrelationBreakDetection {
        let timestamp = returns.first().map(|r| r.timestamp).unwrap_or(0);

        // Update individual asset states
        for ret in returns {
            let state = self
                .asset_states
                .entry(ret.asset.clone())
                .or_insert_with(|| AssetState::new(self.config.baseline_window));

            state.add_return(
                ret.return_value,
                self.config.baseline_window,
                self.config.ema_decay,
            );
        }

        // Calculate correlations for all pairs
        self.calculate_pair_correlations(timestamp)
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Actual processing done in update()
        Ok(())
    }

    /// Calculate correlations between all asset pairs
    fn calculate_pair_correlations(&mut self, timestamp: i64) -> CorrelationBreakDetection {
        let assets: Vec<String> = self.asset_states.keys().cloned().collect();
        let mut pair_results = Vec::new();
        let mut total_change = 0.0;
        let mut max_change = 0.0;
        let mut pairs_with_changes = 0;
        let mut pairs_with_breaks = 0;

        // Calculate correlation for each pair
        for i in 0..assets.len() {
            for j in (i + 1)..assets.len() {
                let pair = AssetPair::new(&assets[i], &assets[j]);

                if let Some(correlation) = self.calculate_correlation(&pair) {
                    // Store correlation history
                    let history = self
                        .pair_correlations
                        .entry(pair.clone())
                        .or_insert_with(|| VecDeque::with_capacity(self.config.baseline_window));

                    history.push_back(correlation);
                    while history.len() > self.config.baseline_window {
                        history.pop_front();
                    }

                    // Calculate baseline if we have enough data
                    let baseline = self.calculate_baseline(&pair);

                    // Get previous correlation for velocity
                    let prev_corr = self.previous_correlations.get(&pair).copied();

                    // Calculate change and velocity
                    let change = correlation - baseline;
                    let velocity = prev_corr.map(|p| correlation - p).unwrap_or(0.0);

                    // Determine severity
                    let severity = BreakSeverity::from_change(
                        change,
                        self.config.change_threshold,
                        self.config.break_threshold,
                    );

                    let sample_count = self.get_pair_sample_count(&pair);
                    let is_reliable = sample_count >= self.config.min_samples;

                    // Track statistics
                    let abs_change = change.abs();
                    total_change += abs_change;
                    if abs_change > max_change {
                        max_change = abs_change;
                    }
                    if severity != BreakSeverity::None {
                        pairs_with_changes += 1;
                    }
                    if severity.requires_action() {
                        pairs_with_breaks += 1;
                    }

                    pair_results.push(PairCorrelation {
                        pair: pair.clone(),
                        current_correlation: correlation,
                        baseline_correlation: baseline,
                        correlation_change: change,
                        change_velocity: velocity,
                        severity,
                        sample_count,
                        is_reliable,
                    });

                    // Store for next iteration
                    self.previous_correlations.insert(pair.clone(), correlation);
                    self.baseline_correlations.entry(pair).or_insert(baseline);
                }
            }
        }

        // Calculate overall severity
        let overall_severity = if pairs_with_breaks > assets.len() / 2 {
            BreakSeverity::Critical
        } else if pairs_with_breaks > 0 {
            BreakSeverity::Major
        } else if pairs_with_changes > assets.len() / 2 {
            BreakSeverity::Significant
        } else if pairs_with_changes > 0 {
            BreakSeverity::Minor
        } else {
            BreakSeverity::None
        };

        let avg_change = if !pair_results.is_empty() {
            total_change / pair_results.len() as f64
        } else {
            0.0
        };

        let break_detected = overall_severity.requires_action();
        if break_detected {
            self.total_breaks += 1;
        }

        let detection = CorrelationBreakDetection {
            break_detected,
            overall_severity,
            pairs_with_changes,
            pairs_with_breaks,
            pair_correlations: pair_results,
            avg_change_magnitude: avg_change,
            max_change,
            timestamp,
        };

        self.last_detection = Some(detection.clone());
        detection
    }

    /// Calculate Pearson correlation between two assets
    fn calculate_correlation(&self, pair: &AssetPair) -> Option<f64> {
        let state_a = self.asset_states.get(&pair.asset_a)?;
        let state_b = self.asset_states.get(&pair.asset_b)?;

        let window = self
            .config
            .correlation_window
            .min(state_a.returns.len())
            .min(state_b.returns.len());

        if window < self.config.min_samples {
            return None;
        }

        // Get the most recent returns from each
        let returns_a: Vec<f64> = state_a.returns.iter().rev().take(window).copied().collect();
        let returns_b: Vec<f64> = state_b.returns.iter().rev().take(window).copied().collect();

        self.pearson_correlation(&returns_a, &returns_b)
    }

    /// Calculate Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }

        let n = x.len() as f64;

        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let std_x = var_x.sqrt();
        let std_y = var_y.sqrt();

        if std_x < 1e-10 || std_y < 1e-10 {
            return None;
        }

        let correlation = cov / (std_x * std_y);
        Some(correlation.clamp(-1.0, 1.0))
    }

    /// Calculate baseline correlation from history
    fn calculate_baseline(&self, pair: &AssetPair) -> f64 {
        if let Some(history) = self.pair_correlations.get(pair) {
            if history.len() >= self.config.min_samples {
                let sum: f64 = history.iter().sum();
                return sum / history.len() as f64;
            }
        }

        // Return existing baseline if available
        self.baseline_correlations.get(pair).copied().unwrap_or(0.0)
    }

    /// Get sample count for a pair
    fn get_pair_sample_count(&self, pair: &AssetPair) -> usize {
        let count_a = self
            .asset_states
            .get(&pair.asset_a)
            .map(|s| s.returns.len())
            .unwrap_or(0);
        let count_b = self
            .asset_states
            .get(&pair.asset_b)
            .map(|s| s.returns.len())
            .unwrap_or(0);
        count_a.min(count_b)
    }

    /// Get correlation matrix for all monitored assets
    pub fn correlation_matrix(&self) -> HashMap<AssetPair, f64> {
        let mut matrix = HashMap::new();
        let assets: Vec<String> = self.asset_states.keys().cloned().collect();

        for i in 0..assets.len() {
            for j in (i + 1)..assets.len() {
                let pair = AssetPair::new(&assets[i], &assets[j]);
                if let Some(corr) = self.calculate_correlation(&pair) {
                    matrix.insert(pair, corr);
                }
            }
        }

        matrix
    }

    /// Get last detection result
    pub fn last_detection(&self) -> Option<&CorrelationBreakDetection> {
        self.last_detection.as_ref()
    }

    /// Get total number of breaks detected
    pub fn total_breaks(&self) -> usize {
        self.total_breaks
    }

    /// Get list of monitored assets
    pub fn monitored_assets(&self) -> Vec<&String> {
        self.asset_states.keys().collect()
    }

    /// Get number of monitored pairs
    pub fn pair_count(&self) -> usize {
        let n = self.asset_states.len();
        if n < 2 { 0 } else { n * (n - 1) / 2 }
    }

    /// Check if detector has enough data
    pub fn is_ready(&self) -> bool {
        self.asset_states.len() >= 2
            && self
                .asset_states
                .values()
                .all(|s| s.returns.len() >= self.config.min_samples)
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.asset_states.clear();
        self.pair_correlations.clear();
        self.baseline_correlations.clear();
        self.previous_correlations.clear();
        self.last_detection = None;
        self.total_breaks = 0;
    }

    /// Remove an asset from monitoring
    pub fn remove_asset(&mut self, asset: &str) {
        self.asset_states.remove(asset);
        // Remove related pairs
        self.pair_correlations
            .retain(|k, _| k.asset_a != asset && k.asset_b != asset);
        self.baseline_correlations
            .retain(|k, _| k.asset_a != asset && k.asset_b != asset);
        self.previous_correlations
            .retain(|k, _| k.asset_a != asset && k.asset_b != asset);
    }

    /// Get statistics summary
    pub fn stats(&self) -> CorrelationStats {
        let avg_correlation = if !self.baseline_correlations.is_empty() {
            self.baseline_correlations.values().sum::<f64>()
                / self.baseline_correlations.len() as f64
        } else {
            0.0
        };

        CorrelationStats {
            asset_count: self.asset_states.len(),
            pair_count: self.pair_count(),
            total_breaks: self.total_breaks,
            avg_baseline_correlation: avg_correlation,
            is_ready: self.is_ready(),
        }
    }
}

/// Summary statistics
#[derive(Debug, Clone)]
pub struct CorrelationStats {
    pub asset_count: usize,
    pub pair_count: usize,
    pub total_breaks: usize,
    pub avg_baseline_correlation: f64,
    pub is_ready: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn create_returns(asset: &str, values: &[f64], base_ts: i64) -> Vec<AssetReturn> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| AssetReturn {
                asset: asset.to_string(),
                return_value: v,
                timestamp: base_ts + i as i64 * 1000,
            })
            .collect()
    }

    #[test]
    fn test_basic() {
        let instance = CorrelationBreak::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_asset_pair_normalization() {
        let pair1 = AssetPair::new("BTC", "ETH");
        let pair2 = AssetPair::new("ETH", "BTC");
        assert_eq!(pair1, pair2);
    }

    #[test]
    fn test_perfect_positive_correlation() {
        let mut detector = CorrelationBreak::with_config(CorrelationBreakConfig {
            correlation_window: 10,
            min_samples: 5,
            ..Default::default()
        });

        // Generate perfectly correlated returns
        for i in 0..20 {
            let ret = 0.01 * (i as f64).sin();
            let returns = vec![
                AssetReturn {
                    asset: "A".to_string(),
                    return_value: ret,
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "B".to_string(),
                    return_value: ret, // Same return = perfect correlation
                    timestamp: i * 1000,
                },
            ];
            detector.update(&returns);
        }

        let matrix = detector.correlation_matrix();
        let pair = AssetPair::new("A", "B");

        if let Some(&corr) = matrix.get(&pair) {
            assert!(
                corr > 0.95,
                "Perfect positive correlation expected, got {}",
                corr
            );
        }
    }

    #[test]
    fn test_perfect_negative_correlation() {
        let mut detector = CorrelationBreak::with_config(CorrelationBreakConfig {
            correlation_window: 10,
            min_samples: 5,
            ..Default::default()
        });

        // Generate perfectly negatively correlated returns
        for i in 0..20 {
            let ret = 0.01 * (i as f64).sin();
            let returns = vec![
                AssetReturn {
                    asset: "A".to_string(),
                    return_value: ret,
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "B".to_string(),
                    return_value: -ret, // Opposite return
                    timestamp: i * 1000,
                },
            ];
            detector.update(&returns);
        }

        let matrix = detector.correlation_matrix();
        let pair = AssetPair::new("A", "B");

        if let Some(&corr) = matrix.get(&pair) {
            assert!(
                corr < -0.95,
                "Perfect negative correlation expected, got {}",
                corr
            );
        }
    }

    #[test]
    fn test_correlation_break_detection() {
        let config = CorrelationBreakConfig {
            correlation_window: 20,
            baseline_window: 50,
            min_samples: 10,
            change_threshold: 0.3,
            break_threshold: 0.5,
            ..Default::default()
        };
        let mut detector = CorrelationBreak::with_config(config);

        // Phase 1: Establish high positive correlation baseline
        for i in 0..40 {
            let ret = 0.01 * (i as f64 * 0.5).sin();
            let returns = vec![
                AssetReturn {
                    asset: "A".to_string(),
                    return_value: ret,
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "B".to_string(),
                    return_value: ret * 0.95 + 0.001, // Highly correlated
                    timestamp: i * 1000,
                },
            ];
            detector.update(&returns);
        }

        // Phase 2: Break correlation - make B move opposite
        let mut break_detected = false;
        for i in 40..60 {
            let ret = 0.01 * (i as f64 * 0.5).sin();
            let returns = vec![
                AssetReturn {
                    asset: "A".to_string(),
                    return_value: ret,
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "B".to_string(),
                    return_value: -ret * 0.8, // Now negatively correlated
                    timestamp: i * 1000,
                },
            ];
            let detection = detector.update(&returns);
            if detection.break_detected {
                break_detected = true;
            }
        }

        // Should have detected the correlation break
        assert!(
            break_detected || detector.total_breaks() > 0,
            "Should detect correlation break"
        );
    }

    #[test]
    fn test_multiple_assets() {
        let mut detector = CorrelationBreak::with_config(CorrelationBreakConfig {
            correlation_window: 10,
            min_samples: 5,
            ..Default::default()
        });

        // Add 4 assets
        for i in 0..20 {
            let returns = vec![
                AssetReturn {
                    asset: "A".to_string(),
                    return_value: 0.01 * (i as f64).sin(),
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "B".to_string(),
                    return_value: 0.01 * (i as f64 + 0.5).sin(),
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "C".to_string(),
                    return_value: 0.01 * (i as f64 + 1.0).sin(),
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "D".to_string(),
                    return_value: 0.01 * (i as f64 + 1.5).sin(),
                    timestamp: i * 1000,
                },
            ];
            detector.update(&returns);
        }

        // Should have 6 pairs: AB, AC, AD, BC, BD, CD
        assert_eq!(detector.pair_count(), 6);
        assert_eq!(detector.monitored_assets().len(), 4);
    }

    #[test]
    fn test_severity_levels() {
        let threshold = 0.3;
        let break_threshold = 0.5;

        assert_eq!(
            BreakSeverity::from_change(0.1, threshold, break_threshold),
            BreakSeverity::None
        );
        assert_eq!(
            BreakSeverity::from_change(0.2, threshold, break_threshold),
            BreakSeverity::Minor
        );
        assert_eq!(
            BreakSeverity::from_change(0.35, threshold, break_threshold),
            BreakSeverity::Significant
        );
        assert_eq!(
            BreakSeverity::from_change(0.55, threshold, break_threshold),
            BreakSeverity::Major
        );
        assert_eq!(
            BreakSeverity::from_change(0.8, threshold, break_threshold),
            BreakSeverity::Critical
        );
    }

    #[test]
    fn test_reset() {
        let mut detector = CorrelationBreak::new();

        // Add some data
        for i in 0..30 {
            let returns = vec![
                AssetReturn {
                    asset: "A".to_string(),
                    return_value: 0.01,
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "B".to_string(),
                    return_value: 0.01,
                    timestamp: i * 1000,
                },
            ];
            detector.update(&returns);
        }

        assert!(!detector.monitored_assets().is_empty());

        detector.reset();

        assert_eq!(detector.monitored_assets().len(), 0);
        assert_eq!(detector.pair_count(), 0);
        assert_eq!(detector.total_breaks(), 0);
    }

    #[test]
    fn test_remove_asset() {
        let mut detector = CorrelationBreak::new();

        // Add 3 assets
        for i in 0..30 {
            let returns = vec![
                AssetReturn {
                    asset: "A".to_string(),
                    return_value: 0.01,
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "B".to_string(),
                    return_value: 0.01,
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "C".to_string(),
                    return_value: 0.01,
                    timestamp: i * 1000,
                },
            ];
            detector.update(&returns);
        }

        assert_eq!(detector.pair_count(), 3); // AB, AC, BC

        detector.remove_asset("B");

        assert_eq!(detector.pair_count(), 1); // Only AC
        assert_eq!(detector.monitored_assets().len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut detector = CorrelationBreak::with_config(CorrelationBreakConfig {
            min_samples: 5,
            ..Default::default()
        });

        for i in 0..30 {
            let returns = vec![
                AssetReturn {
                    asset: "A".to_string(),
                    return_value: 0.01,
                    timestamp: i * 1000,
                },
                AssetReturn {
                    asset: "B".to_string(),
                    return_value: 0.01,
                    timestamp: i * 1000,
                },
            ];
            detector.update(&returns);
        }

        let stats = detector.stats();
        assert_eq!(stats.asset_count, 2);
        assert_eq!(stats.pair_count, 1);
        assert!(stats.is_ready);
    }

    #[test]
    fn test_insufficient_data() {
        let detector = CorrelationBreak::new();
        assert!(!detector.is_ready());

        let matrix = detector.correlation_matrix();
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_risk_multiplier() {
        assert_eq!(BreakSeverity::None.risk_multiplier(), 1.0);
        assert!(BreakSeverity::Minor.risk_multiplier() > 1.0);
        assert!(BreakSeverity::Critical.risk_multiplier() > BreakSeverity::Major.risk_multiplier());
    }
}
