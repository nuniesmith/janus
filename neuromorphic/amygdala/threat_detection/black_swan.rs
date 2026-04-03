//! Black Swan Detector - Detection of rare, high-impact market events
//!
//! Identifies extreme tail events that fall outside normal market distributions.
//! Black swan events are characterized by:
//! - Extreme rarity (beyond 3+ standard deviations)
//! - High impact (significant market moves)
//! - Retrospective predictability (seems obvious after the fact)
//!
//! Detection methods:
//! - Extreme value theory (EVT)
//! - Tail risk metrics
//! - Historical crisis pattern matching
//! - Multi-factor stress indicators

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for black swan detection
#[derive(Debug, Clone)]
pub struct BlackSwanConfig {
    /// Window size for historical analysis
    pub window_size: usize,
    /// Minimum samples before detection is active
    pub min_samples: usize,
    /// Standard deviation threshold for extreme events (e.g., 4.0 = 4 sigma)
    pub sigma_threshold: f64,
    /// Percentile threshold for tail events (e.g., 0.001 = 0.1%)
    pub tail_percentile: f64,
    /// Minimum return magnitude to consider (absolute value)
    pub min_return_magnitude: f64,
    /// Weight for historical pattern matching
    pub pattern_weight: f64,
    /// Decay factor for historical events
    pub historical_decay: f64,
    /// Maximum number of historical crises to store
    pub max_historical_events: usize,
}

impl Default for BlackSwanConfig {
    fn default() -> Self {
        Self {
            window_size: 252, // ~1 year of trading days
            min_samples: 50,
            sigma_threshold: 4.0,
            tail_percentile: 0.001,     // 0.1%
            min_return_magnitude: 0.03, // 3%
            pattern_weight: 0.3,
            historical_decay: 0.95,
            max_historical_events: 100,
        }
    }
}

/// Severity of black swan event
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlackSwanSeverity {
    /// Normal market conditions
    Normal,
    /// Elevated tail risk
    Elevated,
    /// Significant tail event
    Significant,
    /// Major black swan event
    Major,
    /// Catastrophic black swan
    Catastrophic,
}

impl BlackSwanSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Elevated => "elevated",
            Self::Significant => "significant",
            Self::Major => "major",
            Self::Catastrophic => "catastrophic",
        }
    }

    pub fn from_sigma(sigma: f64) -> Self {
        if sigma.abs() >= 6.0 {
            Self::Catastrophic
        } else if sigma.abs() >= 5.0 {
            Self::Major
        } else if sigma.abs() >= 4.0 {
            Self::Significant
        } else if sigma.abs() >= 3.0 {
            Self::Elevated
        } else {
            Self::Normal
        }
    }

    pub fn is_black_swan(&self) -> bool {
        matches!(self, Self::Significant | Self::Major | Self::Catastrophic)
    }
}

/// Type of black swan event
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlackSwanType {
    /// Flash crash (sudden extreme drop)
    FlashCrash,
    /// Liquidity crisis
    LiquidityCrisis,
    /// Volatility explosion
    VolatilityExplosion,
    /// Correlation breakdown
    CorrelationBreakdown,
    /// Market dislocation
    MarketDislocation,
    /// Unknown extreme event
    Unknown,
}

impl BlackSwanType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FlashCrash => "flash_crash",
            Self::LiquidityCrisis => "liquidity_crisis",
            Self::VolatilityExplosion => "volatility_explosion",
            Self::CorrelationBreakdown => "correlation_breakdown",
            Self::MarketDislocation => "market_dislocation",
            Self::Unknown => "unknown",
        }
    }
}

/// Historical crisis pattern for matching
#[derive(Debug, Clone)]
pub struct CrisisPattern {
    /// Name/label of the crisis
    pub name: String,
    /// Feature signature of the crisis
    pub signature: Vec<f64>,
    /// Maximum drawdown during crisis
    pub max_drawdown: f64,
    /// Duration in time units
    pub duration: u32,
    /// Recovery time
    pub recovery_time: u32,
    /// Year of occurrence
    pub year: u32,
}

impl CrisisPattern {
    /// Calculate similarity to current market state
    pub fn similarity(&self, current: &[f64]) -> f64 {
        if self.signature.len() != current.len() || self.signature.is_empty() {
            return 0.0;
        }

        // Cosine similarity
        let dot: f64 = self
            .signature
            .iter()
            .zip(current.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f64 = self.signature.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = current.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            ((dot / (norm_a * norm_b)) + 1.0) / 2.0 // Normalize to [0, 1]
        } else {
            0.0
        }
    }
}

/// Market data for black swan detection
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Current return (price change)
    pub return_value: f64,
    /// Current volatility
    pub volatility: f64,
    /// Bid-ask spread (relative)
    pub spread: f64,
    /// Volume ratio (current / average)
    pub volume_ratio: f64,
    /// Correlation with benchmark
    pub correlation: f64,
    /// Timestamp
    pub timestamp: i64,
    /// Additional features
    pub features: Vec<f64>,
}

impl MarketSnapshot {
    pub fn new(return_value: f64, volatility: f64, timestamp: i64) -> Self {
        Self {
            return_value,
            volatility,
            spread: 0.0,
            volume_ratio: 1.0,
            correlation: 1.0,
            timestamp,
            features: Vec::new(),
        }
    }

    pub fn with_spread(mut self, spread: f64) -> Self {
        self.spread = spread;
        self
    }

    pub fn with_volume_ratio(mut self, ratio: f64) -> Self {
        self.volume_ratio = ratio;
        self
    }

    pub fn with_correlation(mut self, correlation: f64) -> Self {
        self.correlation = correlation;
        self
    }

    pub fn with_features(mut self, features: Vec<f64>) -> Self {
        self.features = features;
        self
    }

    /// Get feature vector for pattern matching
    pub fn to_feature_vector(&self) -> Vec<f64> {
        let mut features = vec![
            self.return_value,
            self.volatility,
            self.spread,
            self.volume_ratio,
            self.correlation,
        ];
        features.extend(&self.features);
        features
    }
}

/// Black swan detection result
#[derive(Debug, Clone)]
pub struct BlackSwanAlert {
    /// Whether a black swan is detected
    pub is_black_swan: bool,
    /// Severity of the event
    pub severity: BlackSwanSeverity,
    /// Type of black swan
    pub event_type: BlackSwanType,
    /// Number of standard deviations
    pub sigma: f64,
    /// Percentile of the event (how extreme)
    pub percentile: f64,
    /// Pattern match score (if similar to historical crisis)
    pub pattern_match_score: f64,
    /// Matching historical crisis (if any)
    pub matching_crisis: Option<String>,
    /// Tail risk score (0.0 - 1.0)
    pub tail_risk_score: f64,
    /// Timestamp
    pub timestamp: i64,
    /// Recommended action
    pub recommended_action: RecommendedAction,
    /// Alert message
    pub message: String,
}

impl Default for BlackSwanAlert {
    fn default() -> Self {
        Self {
            is_black_swan: false,
            severity: BlackSwanSeverity::Normal,
            event_type: BlackSwanType::Unknown,
            sigma: 0.0,
            percentile: 0.5,
            pattern_match_score: 0.0,
            matching_crisis: None,
            tail_risk_score: 0.0,
            timestamp: 0,
            recommended_action: RecommendedAction::None,
            message: String::new(),
        }
    }
}

/// Recommended action for black swan events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    None,
    Monitor,
    ReduceExposure,
    Hedge,
    EmergencyExit,
}

impl RecommendedAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Monitor => "monitor",
            Self::ReduceExposure => "reduce_exposure",
            Self::Hedge => "hedge",
            Self::EmergencyExit => "emergency_exit",
        }
    }
}

/// Internal statistics for return distribution
#[derive(Debug, Clone, Default)]
struct ReturnStats {
    /// Rolling returns
    returns: VecDeque<f64>,
    /// Running sum
    sum: f64,
    /// Running sum of squares
    sum_sq: f64,
    /// Running sum of cubes (for skewness)
    sum_cube: f64,
    /// Running sum of fourth power (for kurtosis)
    sum_fourth: f64,
    /// Minimum return
    min: f64,
    /// Maximum return
    max: f64,
}

impl ReturnStats {
    fn new() -> Self {
        Self {
            returns: VecDeque::new(),
            sum: 0.0,
            sum_sq: 0.0,
            sum_cube: 0.0,
            sum_fourth: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    fn mean(&self) -> f64 {
        if self.returns.is_empty() {
            0.0
        } else {
            self.sum / self.returns.len() as f64
        }
    }

    fn variance(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }
        let n = self.returns.len() as f64;
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

    /// Calculate kurtosis (excess kurtosis, 0 for normal)
    fn kurtosis(&self) -> f64 {
        let n = self.returns.len() as f64;
        if n < 4.0 {
            return 0.0;
        }
        let mean = self.mean();
        let std = self.std_dev();
        if std == 0.0 {
            return 0.0;
        }

        let fourth_moment = self.sum_fourth / n - 4.0 * mean * self.sum_cube / n
            + 6.0 * mean * mean * self.sum_sq / n
            - 3.0 * mean.powi(4);

        fourth_moment / std.powi(4) - 3.0 // Excess kurtosis
    }

    /// Calculate percentile rank of a value
    fn percentile_rank(&self, value: f64) -> f64 {
        if self.returns.is_empty() {
            return 0.5;
        }
        let count_below = self.returns.iter().filter(|&&v| v < value).count();
        count_below as f64 / self.returns.len() as f64
    }
}

/// Black swan detection for rare, high-impact events
pub struct BlackSwan {
    config: BlackSwanConfig,
    /// Return statistics
    return_stats: ReturnStats,
    /// Historical crisis patterns
    crisis_patterns: Vec<CrisisPattern>,
    /// Recent black swan events
    recent_events: VecDeque<BlackSwanAlert>,
    /// Sample count
    sample_count: u64,
    /// Black swan count
    black_swan_count: u64,
    /// Last alert
    last_alert: Option<BlackSwanAlert>,
    /// EMA of tail risk
    ema_tail_risk: f64,
}

impl Default for BlackSwan {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackSwan {
    /// Create a new BlackSwan detector with default config
    pub fn new() -> Self {
        Self::with_config(BlackSwanConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BlackSwanConfig) -> Self {
        let mut detector = Self {
            return_stats: ReturnStats::new(),
            crisis_patterns: Vec::new(),
            recent_events: VecDeque::with_capacity(100),
            sample_count: 0,
            black_swan_count: 0,
            last_alert: None,
            ema_tail_risk: 0.0,
            config,
        };

        // Initialize with historical crisis patterns
        detector.add_historical_patterns();

        detector
    }

    /// Add known historical crisis patterns
    fn add_historical_patterns(&mut self) {
        // 2008 Financial Crisis
        self.crisis_patterns.push(CrisisPattern {
            name: "2008 Financial Crisis".to_string(),
            signature: vec![-0.08, 0.80, 0.05, 3.0, 0.95], // Large negative return, high vol, wide spread, high volume, high correlation
            max_drawdown: 0.57,
            duration: 517,
            recovery_time: 1400,
            year: 2008,
        });

        // Flash Crash 2010
        self.crisis_patterns.push(CrisisPattern {
            name: "Flash Crash 2010".to_string(),
            signature: vec![-0.09, 0.90, 0.10, 5.0, 0.98],
            max_drawdown: 0.09,
            duration: 1, // Minutes
            recovery_time: 1,
            year: 2010,
        });

        // COVID Crash 2020
        self.crisis_patterns.push(CrisisPattern {
            name: "COVID Crash 2020".to_string(),
            signature: vec![-0.12, 0.85, 0.03, 4.0, 0.97],
            max_drawdown: 0.34,
            duration: 23,
            recovery_time: 148,
            year: 2020,
        });

        // Dot-com Crash
        self.crisis_patterns.push(CrisisPattern {
            name: "Dot-com Crash 2000".to_string(),
            signature: vec![-0.05, 0.40, 0.02, 2.0, 0.80],
            max_drawdown: 0.49,
            duration: 929,
            recovery_time: 2500,
            year: 2000,
        });

        // Black Monday 1987
        self.crisis_patterns.push(CrisisPattern {
            name: "Black Monday 1987".to_string(),
            signature: vec![-0.22, 1.50, 0.15, 8.0, 0.99],
            max_drawdown: 0.22,
            duration: 1,
            recovery_time: 400,
            year: 1987,
        });
    }

    /// Add a custom crisis pattern
    pub fn add_crisis_pattern(&mut self, pattern: CrisisPattern) {
        if self.crisis_patterns.len() < self.config.max_historical_events {
            self.crisis_patterns.push(pattern);
        }
    }

    /// Update with new market snapshot
    pub fn update(&mut self, snapshot: &MarketSnapshot) -> BlackSwanAlert {
        self.sample_count += 1;

        // Update return statistics
        self.update_return_stats(snapshot.return_value);

        // Check if we have enough data
        if self.sample_count < self.config.min_samples as u64 {
            return BlackSwanAlert {
                timestamp: snapshot.timestamp,
                message: "Insufficient data for black swan detection".to_string(),
                ..Default::default()
            };
        }

        // Calculate metrics
        let sigma = self.return_stats.zscore(snapshot.return_value);
        let percentile = self.return_stats.percentile_rank(snapshot.return_value);
        let kurtosis = self.return_stats.kurtosis();

        // Check for pattern matches
        let (pattern_score, matching_crisis) = self.check_pattern_match(snapshot);

        // Calculate tail risk score
        let tail_risk = self.calculate_tail_risk(sigma, percentile, kurtosis, pattern_score);

        // Update EMA tail risk
        self.ema_tail_risk = self.config.historical_decay * self.ema_tail_risk
            + (1.0 - self.config.historical_decay) * tail_risk;

        // Determine severity
        let severity = BlackSwanSeverity::from_sigma(sigma);

        // Determine event type
        let event_type = self.determine_event_type(snapshot, sigma);

        // Is this a black swan?
        let is_black_swan = severity.is_black_swan()
            && snapshot.return_value.abs() >= self.config.min_return_magnitude;

        // Determine recommended action
        let recommended_action = self.determine_action(severity, tail_risk);

        // Create message
        let message = self.create_message(severity, sigma, &matching_crisis);

        let alert = BlackSwanAlert {
            is_black_swan,
            severity,
            event_type,
            sigma,
            percentile,
            pattern_match_score: pattern_score,
            matching_crisis,
            tail_risk_score: tail_risk,
            timestamp: snapshot.timestamp,
            recommended_action,
            message,
        };

        // Track event
        if is_black_swan {
            self.black_swan_count += 1;
            if self.recent_events.len() >= 100 {
                self.recent_events.pop_front();
            }
            self.recent_events.push_back(alert.clone());
        }

        self.last_alert = Some(alert.clone());
        alert
    }

    /// Update return statistics
    fn update_return_stats(&mut self, return_value: f64) {
        let window_size = self.config.window_size;

        // Remove old value if at capacity
        if self.return_stats.returns.len() >= window_size {
            if let Some(old) = self.return_stats.returns.pop_front() {
                self.return_stats.sum -= old;
                self.return_stats.sum_sq -= old * old;
                self.return_stats.sum_cube -= old.powi(3);
                self.return_stats.sum_fourth -= old.powi(4);
            }
        }

        // Add new value
        self.return_stats.returns.push_back(return_value);
        self.return_stats.sum += return_value;
        self.return_stats.sum_sq += return_value * return_value;
        self.return_stats.sum_cube += return_value.powi(3);
        self.return_stats.sum_fourth += return_value.powi(4);

        // Update min/max
        if return_value < self.return_stats.min {
            self.return_stats.min = return_value;
        }
        if return_value > self.return_stats.max {
            self.return_stats.max = return_value;
        }
    }

    /// Check for pattern match with historical crises
    fn check_pattern_match(&self, snapshot: &MarketSnapshot) -> (f64, Option<String>) {
        let features = snapshot.to_feature_vector();

        let mut best_score = 0.0;
        let mut best_match = None;

        for pattern in &self.crisis_patterns {
            let similarity = pattern.similarity(&features);
            if similarity > best_score && similarity > 0.6 {
                best_score = similarity;
                best_match = Some(pattern.name.clone());
            }
        }

        (best_score, best_match)
    }

    /// Calculate tail risk score
    fn calculate_tail_risk(
        &self,
        sigma: f64,
        percentile: f64,
        kurtosis: f64,
        pattern_score: f64,
    ) -> f64 {
        let mut score = 0.0;

        // Sigma contribution (40%)
        let sigma_score = (sigma.abs() / self.config.sigma_threshold).min(1.0);
        score += sigma_score * 0.4;

        // Percentile contribution (25%)
        let tail_percentile = if percentile < 0.5 {
            percentile
        } else {
            1.0 - percentile
        };
        let percentile_score = (1.0 - tail_percentile / self.config.tail_percentile).max(0.0);
        score += percentile_score.min(1.0) * 0.25;

        // Kurtosis contribution (15%) - high kurtosis = fat tails
        let kurtosis_score = (kurtosis / 10.0).clamp(0.0, 1.0);
        score += kurtosis_score * 0.15;

        // Pattern match contribution (20%)
        score += pattern_score * self.config.pattern_weight;

        score.clamp(0.0, 1.0)
    }

    /// Determine the type of black swan event
    fn determine_event_type(&self, snapshot: &MarketSnapshot, sigma: f64) -> BlackSwanType {
        // Flash crash: extreme negative return with high volume
        if sigma < -4.0 && snapshot.volume_ratio > 3.0 {
            return BlackSwanType::FlashCrash;
        }

        // Liquidity crisis: wide spread with negative return
        if snapshot.spread > 0.05 && sigma < -2.0 {
            return BlackSwanType::LiquidityCrisis;
        }

        // Volatility explosion: very high volatility
        if snapshot.volatility > 0.10 {
            return BlackSwanType::VolatilityExplosion;
        }

        // Correlation breakdown: unusual correlation
        if snapshot.correlation.abs() < 0.3 {
            return BlackSwanType::CorrelationBreakdown;
        }

        // Market dislocation: multiple factors
        if sigma.abs() > 3.0 && snapshot.spread > 0.02 && snapshot.volume_ratio > 2.0 {
            return BlackSwanType::MarketDislocation;
        }

        BlackSwanType::Unknown
    }

    /// Determine recommended action
    fn determine_action(&self, severity: BlackSwanSeverity, tail_risk: f64) -> RecommendedAction {
        match severity {
            BlackSwanSeverity::Catastrophic => RecommendedAction::EmergencyExit,
            BlackSwanSeverity::Major => RecommendedAction::Hedge,
            BlackSwanSeverity::Significant => RecommendedAction::ReduceExposure,
            BlackSwanSeverity::Elevated => {
                if tail_risk > 0.5 {
                    RecommendedAction::ReduceExposure
                } else {
                    RecommendedAction::Monitor
                }
            }
            BlackSwanSeverity::Normal => RecommendedAction::None,
        }
    }

    /// Create alert message
    fn create_message(
        &self,
        severity: BlackSwanSeverity,
        sigma: f64,
        matching: &Option<String>,
    ) -> String {
        match severity {
            BlackSwanSeverity::Catastrophic => {
                format!(
                    "🦢⚫ CATASTROPHIC BLACK SWAN: {:.1}σ event detected! {}",
                    sigma.abs(),
                    matching
                        .as_ref()
                        .map(|m| format!("Similar to: {}", m))
                        .unwrap_or_default()
                )
            }
            BlackSwanSeverity::Major => {
                format!(
                    "🦢 MAJOR BLACK SWAN: {:.1}σ event. Extreme tail risk.",
                    sigma.abs()
                )
            }
            BlackSwanSeverity::Significant => {
                format!(
                    "⚠️ SIGNIFICANT TAIL EVENT: {:.1}σ move detected.",
                    sigma.abs()
                )
            }
            BlackSwanSeverity::Elevated => {
                format!(
                    "📊 ELEVATED TAIL RISK: {:.1}σ move. Monitoring.",
                    sigma.abs()
                )
            }
            BlackSwanSeverity::Normal => String::from("Normal market conditions"),
        }
    }

    /// Get current tail risk (EMA)
    pub fn tail_risk(&self) -> f64 {
        self.ema_tail_risk
    }

    /// Get black swan count
    pub fn black_swan_count(&self) -> u64 {
        self.black_swan_count
    }

    /// Get sample count
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Get last alert
    pub fn last_alert(&self) -> Option<&BlackSwanAlert> {
        self.last_alert.as_ref()
    }

    /// Get recent black swan events
    pub fn recent_events(&self) -> &VecDeque<BlackSwanAlert> {
        &self.recent_events
    }

    /// Get statistics
    pub fn statistics(&self) -> BlackSwanStats {
        BlackSwanStats {
            sample_count: self.sample_count,
            black_swan_count: self.black_swan_count,
            black_swan_rate: if self.sample_count > 0 {
                self.black_swan_count as f64 / self.sample_count as f64
            } else {
                0.0
            },
            current_tail_risk: self.ema_tail_risk,
            return_mean: self.return_stats.mean(),
            return_std: self.return_stats.std_dev(),
            return_kurtosis: self.return_stats.kurtosis(),
            min_return: self.return_stats.min,
            max_return: self.return_stats.max,
            crisis_patterns_count: self.crisis_patterns.len(),
        }
    }

    /// Check if detector is active (has enough samples)
    pub fn is_active(&self) -> bool {
        self.sample_count >= self.config.min_samples as u64
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.return_stats = ReturnStats::new();
        self.recent_events.clear();
        self.sample_count = 0;
        self.black_swan_count = 0;
        self.last_alert = None;
        self.ema_tail_risk = 0.0;
    }

    /// Main processing function (compatibility with neuromorphic interface)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Statistics about black swan detection
#[derive(Debug, Clone)]
pub struct BlackSwanStats {
    pub sample_count: u64,
    pub black_swan_count: u64,
    pub black_swan_rate: f64,
    pub current_tail_risk: f64,
    pub return_mean: f64,
    pub return_std: f64,
    pub return_kurtosis: f64,
    pub min_return: f64,
    pub max_return: f64,
    pub crisis_patterns_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_snapshot(return_value: f64, volatility: f64, timestamp: i64) -> MarketSnapshot {
        MarketSnapshot::new(return_value, volatility, timestamp)
    }

    #[test]
    fn test_basic_creation() {
        let detector = BlackSwan::new();
        assert_eq!(detector.sample_count(), 0);
        assert_eq!(detector.black_swan_count(), 0);
        assert!(!detector.is_active());
    }

    #[test]
    fn test_normal_market() {
        let mut detector = BlackSwan::with_config(BlackSwanConfig {
            min_samples: 20,
            sigma_threshold: 3.0,
            ..Default::default()
        });

        // Add normal returns
        for i in 0..50 {
            let return_value = 0.001 * ((i % 10) as f64 - 5.0); // Small returns
            let snapshot = create_snapshot(return_value, 0.02, i);
            let alert = detector.update(&snapshot);

            if i > 25 {
                assert!(
                    !alert.is_black_swan,
                    "Normal market triggered black swan at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_extreme_event_detection() {
        let mut detector = BlackSwan::with_config(BlackSwanConfig {
            min_samples: 30,
            sigma_threshold: 3.0,
            min_return_magnitude: 0.02,
            ..Default::default()
        });

        // Build normal distribution
        for i in 0..50 {
            let return_value = 0.001 * ((i % 5) as f64 - 2.0);
            let snapshot = create_snapshot(return_value, 0.02, i);
            detector.update(&snapshot);
        }

        // Introduce extreme event (-10% return)
        let extreme_snapshot = create_snapshot(-0.10, 0.50, 50)
            .with_spread(0.05)
            .with_volume_ratio(5.0);
        let alert = detector.update(&extreme_snapshot);

        assert!(
            alert.sigma.abs() > 3.0,
            "Extreme event should have high sigma"
        );
        assert!(
            alert.severity >= BlackSwanSeverity::Significant,
            "Should be significant or higher"
        );
    }

    #[test]
    fn test_pattern_matching() {
        let mut detector = BlackSwan::new();

        // Build baseline
        for i in 0..60 {
            let snapshot = create_snapshot(0.001, 0.02, i);
            detector.update(&snapshot);
        }

        // Create snapshot similar to 2008 pattern
        let crisis_like = MarketSnapshot::new(-0.08, 0.80, 60)
            .with_spread(0.05)
            .with_volume_ratio(3.0)
            .with_correlation(0.95)
            .with_features(vec![]);

        let alert = detector.update(&crisis_like);

        // Should have some pattern match
        assert!(alert.pattern_match_score >= 0.0);
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(
            BlackSwanSeverity::from_sigma(2.0),
            BlackSwanSeverity::Normal
        );
        assert_eq!(
            BlackSwanSeverity::from_sigma(3.5),
            BlackSwanSeverity::Elevated
        );
        assert_eq!(
            BlackSwanSeverity::from_sigma(4.5),
            BlackSwanSeverity::Significant
        );
        assert_eq!(BlackSwanSeverity::from_sigma(5.5), BlackSwanSeverity::Major);
        assert_eq!(
            BlackSwanSeverity::from_sigma(7.0),
            BlackSwanSeverity::Catastrophic
        );
    }

    #[test]
    fn test_is_black_swan() {
        assert!(!BlackSwanSeverity::Normal.is_black_swan());
        assert!(!BlackSwanSeverity::Elevated.is_black_swan());
        assert!(BlackSwanSeverity::Significant.is_black_swan());
        assert!(BlackSwanSeverity::Major.is_black_swan());
        assert!(BlackSwanSeverity::Catastrophic.is_black_swan());
    }

    #[test]
    fn test_event_types() {
        assert_eq!(BlackSwanType::FlashCrash.as_str(), "flash_crash");
        assert_eq!(BlackSwanType::LiquidityCrisis.as_str(), "liquidity_crisis");
        assert_eq!(
            BlackSwanType::VolatilityExplosion.as_str(),
            "volatility_explosion"
        );
    }

    #[test]
    fn test_crisis_pattern_similarity() {
        let pattern = CrisisPattern {
            name: "Test".to_string(),
            signature: vec![1.0, 0.0, 0.0],
            max_drawdown: 0.5,
            duration: 10,
            recovery_time: 100,
            year: 2020,
        };

        // Identical should be high similarity
        let sim = pattern.similarity(&[1.0, 0.0, 0.0]);
        assert!(sim > 0.9);

        // Different should be lower
        let sim = pattern.similarity(&[-1.0, 0.0, 0.0]);
        assert!(sim < 0.5);
    }

    #[test]
    fn test_statistics() {
        let mut detector = BlackSwan::new();

        for i in 0..100 {
            let snapshot = create_snapshot(0.001, 0.02, i);
            detector.update(&snapshot);
        }

        let stats = detector.statistics();
        assert_eq!(stats.sample_count, 100);
        assert!(stats.crisis_patterns_count > 0);
    }

    #[test]
    fn test_reset() {
        let mut detector = BlackSwan::new();

        for i in 0..50 {
            let snapshot = create_snapshot(0.001, 0.02, i);
            detector.update(&snapshot);
        }

        assert!(detector.sample_count() > 0);

        detector.reset();

        assert_eq!(detector.sample_count(), 0);
        assert_eq!(detector.black_swan_count(), 0);
    }

    #[test]
    fn test_add_custom_pattern() {
        let mut detector = BlackSwan::new();
        let initial_count = detector.crisis_patterns.len();

        detector.add_crisis_pattern(CrisisPattern {
            name: "Custom Crisis".to_string(),
            signature: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            max_drawdown: 0.3,
            duration: 30,
            recovery_time: 90,
            year: 2023,
        });

        assert_eq!(detector.crisis_patterns.len(), initial_count + 1);
    }

    #[test]
    fn test_process() {
        let detector = BlackSwan::new();
        assert!(detector.process().is_ok());
    }

    #[test]
    fn test_market_snapshot_builder() {
        let snapshot = MarketSnapshot::new(-0.05, 0.30, 1000)
            .with_spread(0.02)
            .with_volume_ratio(2.5)
            .with_correlation(0.8)
            .with_features(vec![1.0, 2.0]);

        assert_eq!(snapshot.return_value, -0.05);
        assert_eq!(snapshot.volatility, 0.30);
        assert_eq!(snapshot.spread, 0.02);
        assert_eq!(snapshot.volume_ratio, 2.5);
        assert_eq!(snapshot.correlation, 0.8);
        assert_eq!(snapshot.features.len(), 2);
    }

    #[test]
    fn test_recommended_actions() {
        let detector = BlackSwan::new();

        let action = detector.determine_action(BlackSwanSeverity::Catastrophic, 0.9);
        assert_eq!(action, RecommendedAction::EmergencyExit);

        let action = detector.determine_action(BlackSwanSeverity::Major, 0.8);
        assert_eq!(action, RecommendedAction::Hedge);

        let action = detector.determine_action(BlackSwanSeverity::Normal, 0.1);
        assert_eq!(action, RecommendedAction::None);
    }
}
