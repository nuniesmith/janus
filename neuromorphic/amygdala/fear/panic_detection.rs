//! Market panic detection
//!
//! Detects market panic conditions using multiple indicators:
//! - Rapid price decline velocity
//! - Volume surge detection
//! - Volatility regime shifts
//! - Bid-ask spread explosion
//! - Order book depth collapse
//! - Correlated selling across assets

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for panic detection
#[derive(Debug, Clone)]
pub struct PanicConfig {
    /// Window size for rolling calculations
    pub window_size: usize,
    /// Minimum samples before detection is reliable
    pub min_samples: usize,
    /// Threshold for price decline rate (% per minute)
    pub price_decline_threshold: f64,
    /// Threshold for volume surge (multiple of average)
    pub volume_surge_threshold: f64,
    /// Threshold for volatility spike (multiple of average)
    pub volatility_spike_threshold: f64,
    /// Threshold for spread widening (multiple of average)
    pub spread_widening_threshold: f64,
    /// Overall panic score threshold
    pub panic_threshold: f64,
    /// Critical panic threshold requiring immediate action
    pub critical_threshold: f64,
    /// EMA decay factor for smoothing
    pub ema_decay: f64,
}

impl Default for PanicConfig {
    fn default() -> Self {
        Self {
            window_size: 60,
            min_samples: 10,
            price_decline_threshold: 2.0,    // 2% per minute
            volume_surge_threshold: 5.0,     // 5x average volume
            volatility_spike_threshold: 3.0, // 3x average volatility
            spread_widening_threshold: 4.0,  // 4x average spread
            panic_threshold: 0.6,
            critical_threshold: 0.85,
            ema_decay: 0.92,
        }
    }
}

/// Panic severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanicLevel {
    /// Normal market conditions
    None,
    /// Elevated stress, increased monitoring
    Elevated,
    /// High stress, consider reducing exposure
    High,
    /// Panic conditions, protective actions needed
    Panic,
    /// Critical panic, emergency protocols
    Critical,
}

impl PanicLevel {
    pub fn from_score(score: f64, panic_threshold: f64, critical_threshold: f64) -> Self {
        if score >= critical_threshold {
            Self::Critical
        } else if score >= panic_threshold {
            Self::Panic
        } else if score >= panic_threshold * 0.7 {
            Self::High
        } else if score >= panic_threshold * 0.4 {
            Self::Elevated
        } else {
            Self::None
        }
    }

    /// Whether this level requires protective action
    pub fn requires_action(&self) -> bool {
        matches!(self, Self::High | Self::Panic | Self::Critical)
    }

    /// Whether this is a critical emergency
    pub fn is_critical(&self) -> bool {
        matches!(self, Self::Critical)
    }
}

/// Individual panic indicator scores
#[derive(Debug, Clone, Default)]
pub struct PanicIndicators {
    /// Price decline velocity score (0.0 - 1.0)
    pub price_decline: f64,
    /// Volume surge score
    pub volume_surge: f64,
    /// Volatility spike score
    pub volatility_spike: f64,
    /// Spread widening score
    pub spread_widening: f64,
    /// Depth collapse score
    pub depth_collapse: f64,
    /// Correlated selling score
    pub correlated_selling: f64,
}

impl PanicIndicators {
    /// Calculate weighted average of all indicators
    pub fn weighted_average(&self, weights: &PanicWeights) -> f64 {
        let sum = self.price_decline * weights.price_decline
            + self.volume_surge * weights.volume_surge
            + self.volatility_spike * weights.volatility_spike
            + self.spread_widening * weights.spread_widening
            + self.depth_collapse * weights.depth_collapse
            + self.correlated_selling * weights.correlated_selling;

        let weight_sum = weights.price_decline
            + weights.volume_surge
            + weights.volatility_spike
            + weights.spread_widening
            + weights.depth_collapse
            + weights.correlated_selling;

        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            0.0
        }
    }

    /// Get the maximum indicator value
    pub fn max_indicator(&self) -> f64 {
        self.price_decline
            .max(self.volume_surge)
            .max(self.volatility_spike)
            .max(self.spread_widening)
            .max(self.depth_collapse)
            .max(self.correlated_selling)
    }
}

/// Weights for panic indicators
#[derive(Debug, Clone)]
pub struct PanicWeights {
    pub price_decline: f64,
    pub volume_surge: f64,
    pub volatility_spike: f64,
    pub spread_widening: f64,
    pub depth_collapse: f64,
    pub correlated_selling: f64,
}

impl Default for PanicWeights {
    fn default() -> Self {
        Self {
            price_decline: 0.30,
            volume_surge: 0.15,
            volatility_spike: 0.20,
            spread_widening: 0.15,
            depth_collapse: 0.10,
            correlated_selling: 0.10,
        }
    }
}

/// Market data point for panic detection
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Current price
    pub price: f64,
    /// Trade volume in this period
    pub volume: f64,
    /// Best bid price
    pub bid: f64,
    /// Best ask price
    pub ask: f64,
    /// Total bid depth
    pub bid_depth: f64,
    /// Total ask depth
    pub ask_depth: f64,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
}

/// Result of panic assessment
#[derive(Debug, Clone)]
pub struct PanicAssessment {
    /// Overall panic score (0.0 - 1.0)
    pub score: f64,
    /// Panic severity level
    pub level: PanicLevel,
    /// Individual indicator scores
    pub indicators: PanicIndicators,
    /// Number of samples used
    pub sample_count: usize,
    /// Whether assessment is reliable
    pub is_reliable: bool,
    /// Timestamp of assessment
    pub timestamp: i64,
    /// Recommended action
    pub recommended_action: PanicAction,
}

/// Recommended actions based on panic level
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PanicAction {
    /// Continue normal operations
    None,
    /// Increase monitoring frequency
    IncreaseMonitoring,
    /// Reduce position sizes
    ReduceExposure,
    /// Halt new positions
    HaltNewPositions,
    /// Emergency exit
    EmergencyExit,
}

impl From<PanicLevel> for PanicAction {
    fn from(level: PanicLevel) -> Self {
        match level {
            PanicLevel::None => Self::None,
            PanicLevel::Elevated => Self::IncreaseMonitoring,
            PanicLevel::High => Self::ReduceExposure,
            PanicLevel::Panic => Self::HaltNewPositions,
            PanicLevel::Critical => Self::EmergencyExit,
        }
    }
}

/// Internal statistics for calculations
#[derive(Debug, Clone, Default)]
struct RollingStats {
    price_returns: VecDeque<f64>,
    volumes: VecDeque<f64>,
    spreads: VecDeque<f64>,
    depths: VecDeque<f64>,
    volatilities: VecDeque<f64>,
    last_price: Option<f64>,
    last_timestamp: Option<i64>,
}

impl RollingStats {
    fn new(capacity: usize) -> Self {
        Self {
            price_returns: VecDeque::with_capacity(capacity),
            volumes: VecDeque::with_capacity(capacity),
            spreads: VecDeque::with_capacity(capacity),
            depths: VecDeque::with_capacity(capacity),
            volatilities: VecDeque::with_capacity(capacity),
            last_price: None,
            last_timestamp: None,
        }
    }

    fn mean(data: &VecDeque<f64>) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    #[allow(dead_code)]
    fn std_dev(data: &VecDeque<f64>) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = Self::mean(data);
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Market panic detection system
///
/// Monitors multiple market stress indicators to detect panic conditions
/// that may require protective action.
pub struct PanicDetection {
    config: PanicConfig,
    weights: PanicWeights,
    stats: RollingStats,
    /// Exponential moving average of panic score
    ema_panic_score: f64,
    /// Current panic level
    current_level: PanicLevel,
    /// Count of consecutive high-panic readings
    consecutive_panic_count: usize,
    /// Historical panic events for analysis
    panic_history: VecDeque<PanicAssessment>,
}

impl Default for PanicDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl PanicDetection {
    /// Create a new panic detection instance with default config
    pub fn new() -> Self {
        Self::with_config(PanicConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PanicConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            weights: PanicWeights::default(),
            stats: RollingStats::new(window_size),
            ema_panic_score: 0.0,
            current_level: PanicLevel::None,
            consecutive_panic_count: 0,
            panic_history: VecDeque::with_capacity(100),
        }
    }

    /// Set custom weights for indicators
    pub fn with_weights(mut self, weights: PanicWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Main processing function - update with new market snapshot
    pub fn process(&mut self, snapshot: &MarketSnapshot) -> Result<PanicAssessment> {
        // Update rolling statistics
        self.update_stats(snapshot);

        // Calculate individual indicators
        let indicators = self.calculate_indicators(snapshot);

        // Calculate overall panic score
        let raw_score = indicators.weighted_average(&self.weights);

        // Apply EMA smoothing
        self.ema_panic_score = self.config.ema_decay * self.ema_panic_score
            + (1.0 - self.config.ema_decay) * raw_score;

        // Determine panic level
        let level = PanicLevel::from_score(
            self.ema_panic_score,
            self.config.panic_threshold,
            self.config.critical_threshold,
        );

        // Track consecutive panic readings
        if level.requires_action() {
            self.consecutive_panic_count += 1;
        } else {
            self.consecutive_panic_count = 0;
        }

        self.current_level = level;

        let sample_count = self.stats.price_returns.len();
        let is_reliable = sample_count >= self.config.min_samples;

        let assessment = PanicAssessment {
            score: self.ema_panic_score,
            level,
            indicators,
            sample_count,
            is_reliable,
            timestamp: snapshot.timestamp,
            recommended_action: level.into(),
        };

        // Store in history if significant
        if level.requires_action() {
            self.panic_history.push_back(assessment.clone());
            if self.panic_history.len() > 100 {
                self.panic_history.pop_front();
            }
        }

        Ok(assessment)
    }

    /// Update rolling statistics with new snapshot
    fn update_stats(&mut self, snapshot: &MarketSnapshot) {
        let window = self.config.window_size;

        // Calculate price return if we have previous price
        if let Some(last_price) = self.stats.last_price {
            if last_price > 0.0 {
                let ret = (snapshot.price - last_price) / last_price;
                self.stats.price_returns.push_back(ret);
                if self.stats.price_returns.len() > window {
                    self.stats.price_returns.pop_front();
                }

                // Calculate realized volatility (absolute return)
                self.stats.volatilities.push_back(ret.abs());
                if self.stats.volatilities.len() > window {
                    self.stats.volatilities.pop_front();
                }
            }
        }

        // Update volume
        self.stats.volumes.push_back(snapshot.volume);
        if self.stats.volumes.len() > window {
            self.stats.volumes.pop_front();
        }

        // Update spread
        let spread = if snapshot.bid > 0.0 {
            (snapshot.ask - snapshot.bid) / snapshot.bid
        } else {
            0.0
        };
        self.stats.spreads.push_back(spread);
        if self.stats.spreads.len() > window {
            self.stats.spreads.pop_front();
        }

        // Update depth
        let total_depth = snapshot.bid_depth + snapshot.ask_depth;
        self.stats.depths.push_back(total_depth);
        if self.stats.depths.len() > window {
            self.stats.depths.pop_front();
        }

        self.stats.last_price = Some(snapshot.price);
        self.stats.last_timestamp = Some(snapshot.timestamp);
    }

    /// Calculate all panic indicators
    fn calculate_indicators(&self, snapshot: &MarketSnapshot) -> PanicIndicators {
        PanicIndicators {
            price_decline: self.calculate_price_decline_score(),
            volume_surge: self.calculate_volume_surge_score(snapshot.volume),
            volatility_spike: self.calculate_volatility_spike_score(),
            spread_widening: self.calculate_spread_widening_score(snapshot),
            depth_collapse: self.calculate_depth_collapse_score(snapshot),
            correlated_selling: self.calculate_correlated_selling_score(),
        }
    }

    /// Calculate price decline velocity score
    fn calculate_price_decline_score(&self) -> f64 {
        if self.stats.price_returns.len() < 2 {
            return 0.0;
        }

        // Sum recent returns (negative = decline)
        let recent_return: f64 = self.stats.price_returns.iter().rev().take(5).sum();

        // Convert to percentage and compare to threshold
        let decline_pct = -recent_return * 100.0;

        if decline_pct <= 0.0 {
            return 0.0;
        }

        // Score based on decline magnitude
        let score = (decline_pct / self.config.price_decline_threshold).min(1.0);

        // Apply non-linear scaling for extreme moves
        if score > 0.7 {
            0.7 + (score - 0.7) * 1.5
        } else {
            score
        }
        .min(1.0)
    }

    /// Calculate volume surge score
    fn calculate_volume_surge_score(&self, current_volume: f64) -> f64 {
        let avg_volume = RollingStats::mean(&self.stats.volumes);

        if avg_volume <= 0.0 {
            return 0.0;
        }

        let volume_ratio = current_volume / avg_volume;

        if volume_ratio <= 1.0 {
            return 0.0;
        }

        // Score based on how much volume exceeds average
        ((volume_ratio - 1.0) / (self.config.volume_surge_threshold - 1.0)).min(1.0)
    }

    /// Calculate volatility spike score
    fn calculate_volatility_spike_score(&self) -> f64 {
        if self.stats.volatilities.len() < self.config.min_samples {
            return 0.0;
        }

        let avg_vol = RollingStats::mean(&self.stats.volatilities);

        if avg_vol <= 0.0 {
            return 0.0;
        }

        // Get recent volatility (last 5 samples)
        let recent_vol: f64 = self.stats.volatilities.iter().rev().take(5).sum::<f64>() / 5.0;

        let vol_ratio = recent_vol / avg_vol;

        if vol_ratio <= 1.0 {
            return 0.0;
        }

        ((vol_ratio - 1.0) / (self.config.volatility_spike_threshold - 1.0)).min(1.0)
    }

    /// Calculate spread widening score
    fn calculate_spread_widening_score(&self, snapshot: &MarketSnapshot) -> f64 {
        let avg_spread = RollingStats::mean(&self.stats.spreads);

        if avg_spread <= 0.0 || snapshot.bid <= 0.0 {
            return 0.0;
        }

        let current_spread = (snapshot.ask - snapshot.bid) / snapshot.bid;
        let spread_ratio = current_spread / avg_spread;

        if spread_ratio <= 1.0 {
            return 0.0;
        }

        ((spread_ratio - 1.0) / (self.config.spread_widening_threshold - 1.0)).min(1.0)
    }

    /// Calculate depth collapse score
    fn calculate_depth_collapse_score(&self, snapshot: &MarketSnapshot) -> f64 {
        let avg_depth = RollingStats::mean(&self.stats.depths);

        if avg_depth <= 0.0 {
            return 0.0;
        }

        let current_depth = snapshot.bid_depth + snapshot.ask_depth;
        let depth_ratio = current_depth / avg_depth;

        // Score increases as depth decreases
        if depth_ratio >= 1.0 {
            return 0.0;
        }

        // Inverse ratio - lower depth = higher score
        (1.0 - depth_ratio).min(1.0)
    }

    /// Calculate correlated selling score
    /// Note: This is a simplified version - full implementation would track multiple assets
    fn calculate_correlated_selling_score(&self) -> f64 {
        if self.stats.price_returns.len() < self.config.min_samples {
            return 0.0;
        }

        // Count consecutive negative returns
        let consecutive_negative = self
            .stats
            .price_returns
            .iter()
            .rev()
            .take_while(|&&r| r < 0.0)
            .count();

        // Score based on consecutive negative returns
        let score = (consecutive_negative as f64 / 10.0).min(1.0);

        // Boost if returns are significantly negative
        let recent_returns: Vec<_> = self.stats.price_returns.iter().rev().take(5).collect();
        let avg_recent = recent_returns.iter().copied().sum::<f64>() / recent_returns.len() as f64;

        if avg_recent < -0.01 {
            (score * 1.5).min(1.0)
        } else {
            score
        }
    }

    /// Get current panic level
    pub fn current_level(&self) -> PanicLevel {
        self.current_level
    }

    /// Get current smoothed panic score
    pub fn current_score(&self) -> f64 {
        self.ema_panic_score
    }

    /// Get consecutive panic count
    pub fn consecutive_panic_count(&self) -> usize {
        self.consecutive_panic_count
    }

    /// Check if currently in panic state
    pub fn is_panicking(&self) -> bool {
        self.current_level.requires_action()
    }

    /// Check if in critical state
    pub fn is_critical(&self) -> bool {
        self.current_level.is_critical()
    }

    /// Get panic history
    pub fn panic_history(&self) -> &VecDeque<PanicAssessment> {
        &self.panic_history
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.stats = RollingStats::new(self.config.window_size);
        self.ema_panic_score = 0.0;
        self.current_level = PanicLevel::None;
        self.consecutive_panic_count = 0;
    }

    /// Get statistics summary
    pub fn stats_summary(&self) -> PanicStats {
        PanicStats {
            sample_count: self.stats.price_returns.len(),
            avg_volume: RollingStats::mean(&self.stats.volumes),
            avg_spread: RollingStats::mean(&self.stats.spreads),
            avg_volatility: RollingStats::mean(&self.stats.volatilities),
            current_panic_score: self.ema_panic_score,
            panic_events: self.panic_history.len(),
        }
    }
}

/// Summary statistics
#[derive(Debug, Clone)]
pub struct PanicStats {
    pub sample_count: usize,
    pub avg_volume: f64,
    pub avg_spread: f64,
    pub avg_volatility: f64,
    pub current_panic_score: f64,
    pub panic_events: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_normal_snapshot(price: f64, timestamp: i64) -> MarketSnapshot {
        MarketSnapshot {
            price,
            volume: 1000.0,
            bid: price * 0.999,
            ask: price * 1.001,
            bid_depth: 50000.0,
            ask_depth: 50000.0,
            timestamp,
        }
    }

    fn create_panic_snapshot(price: f64, timestamp: i64) -> MarketSnapshot {
        MarketSnapshot {
            price,
            volume: 10000.0,   // 10x volume
            bid: price * 0.99, // Wide spread
            ask: price * 1.02,
            bid_depth: 10000.0, // Collapsed depth
            ask_depth: 5000.0,
            timestamp,
        }
    }

    #[test]
    fn test_basic() {
        let instance = PanicDetection::new();
        assert_eq!(instance.current_level(), PanicLevel::None);
    }

    #[test]
    fn test_normal_market_no_panic() {
        let mut detector = PanicDetection::new();

        // Feed normal market data
        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.01); // Slight uptrend
            let snapshot = create_normal_snapshot(price, i * 1000);
            let assessment = detector.process(&snapshot).unwrap();

            // Should never trigger panic in normal conditions
            assert_eq!(assessment.level, PanicLevel::None);
        }
    }

    #[test]
    fn test_price_decline_detection() {
        let mut detector = PanicDetection::new();

        // Establish baseline
        for i in 0..20 {
            let snapshot = create_normal_snapshot(100.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        // Simulate rapid decline
        let mut price = 100.0;
        for i in 20..30 {
            price *= 0.97; // 3% decline per tick
            let snapshot = create_normal_snapshot(price, i * 1000);
            let assessment = detector.process(&snapshot).unwrap();

            // After several declines, should detect elevated/panic
            if i > 25 {
                assert!(
                    assessment.indicators.price_decline > 0.3,
                    "Price decline indicator should be elevated"
                );
            }
        }
    }

    #[test]
    fn test_volume_surge_detection() {
        let mut detector = PanicDetection::new();

        // Establish baseline with normal volume
        for i in 0..30 {
            let snapshot = create_normal_snapshot(100.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        // Now send high volume
        let mut high_vol_snapshot = create_normal_snapshot(100.0, 30000);
        high_vol_snapshot.volume = 8000.0; // 8x normal

        let assessment = detector.process(&high_vol_snapshot).unwrap();
        assert!(
            assessment.indicators.volume_surge > 0.5,
            "Volume surge should be detected"
        );
    }

    #[test]
    fn test_full_panic_scenario() {
        let mut detector = PanicDetection::with_config(PanicConfig {
            min_samples: 5,
            ..Default::default()
        });

        // Establish baseline
        for i in 0..20 {
            let snapshot = create_normal_snapshot(100.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        // Trigger panic: rapid decline with volume surge and spread widening
        let mut price = 100.0;
        let mut final_level = PanicLevel::None;

        for i in 20..40 {
            price *= 0.96; // 4% decline per tick
            let snapshot = create_panic_snapshot(price, i * 1000);
            let assessment = detector.process(&snapshot).unwrap();
            final_level = assessment.level;
        }

        // Should reach at least High panic level
        assert!(
            matches!(
                final_level,
                PanicLevel::High | PanicLevel::Panic | PanicLevel::Critical
            ),
            "Should detect panic in extreme conditions, got: {:?}",
            final_level
        );
    }

    #[test]
    fn test_panic_recovery() {
        let mut detector = PanicDetection::new();

        // Establish baseline
        for i in 0..20 {
            let snapshot = create_normal_snapshot(100.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        // Brief panic
        for i in 20..25 {
            let snapshot = create_panic_snapshot(95.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        // Return to normal
        for i in 25..50 {
            let snapshot = create_normal_snapshot(95.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        // Should recover (EMA decays)
        assert!(
            detector.current_score() < detector.config.panic_threshold,
            "Should recover from panic"
        );
    }

    #[test]
    fn test_depth_collapse_detection() {
        let mut detector = PanicDetection::new();

        // Establish baseline with normal depth
        for i in 0..30 {
            let snapshot = create_normal_snapshot(100.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        // Now collapse depth
        let mut low_depth_snapshot = create_normal_snapshot(100.0, 30000);
        low_depth_snapshot.bid_depth = 5000.0; // 10% of normal
        low_depth_snapshot.ask_depth = 5000.0;

        let assessment = detector.process(&low_depth_snapshot).unwrap();
        assert!(
            assessment.indicators.depth_collapse > 0.7,
            "Depth collapse should be detected"
        );
    }

    #[test]
    fn test_spread_widening_detection() {
        let mut detector = PanicDetection::new();

        // Establish baseline
        for i in 0..30 {
            let snapshot = create_normal_snapshot(100.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        // Wide spread snapshot
        let wide_spread = MarketSnapshot {
            price: 100.0,
            volume: 1000.0,
            bid: 99.0, // 1% spread instead of 0.2%
            ask: 101.0,
            bid_depth: 50000.0,
            ask_depth: 50000.0,
            timestamp: 30000,
        };

        let assessment = detector.process(&wide_spread).unwrap();
        assert!(
            assessment.indicators.spread_widening > 0.3,
            "Spread widening should be detected"
        );
    }

    #[test]
    fn test_reset() {
        let mut detector = PanicDetection::new();

        // Build up some state
        for i in 0..30 {
            let snapshot = create_panic_snapshot(100.0 * 0.99_f64.powi(i), i as i64 * 1000);
            detector.process(&snapshot).unwrap();
        }

        detector.reset();

        assert_eq!(detector.current_level(), PanicLevel::None);
        assert_eq!(detector.current_score(), 0.0);
        assert_eq!(detector.consecutive_panic_count(), 0);
    }

    #[test]
    fn test_panic_level_from_score() {
        let panic_threshold = 0.6;
        let critical_threshold = 0.85;

        assert_eq!(
            PanicLevel::from_score(0.1, panic_threshold, critical_threshold),
            PanicLevel::None
        );
        assert_eq!(
            PanicLevel::from_score(0.3, panic_threshold, critical_threshold),
            PanicLevel::Elevated
        );
        assert_eq!(
            PanicLevel::from_score(0.5, panic_threshold, critical_threshold),
            PanicLevel::High
        );
        assert_eq!(
            PanicLevel::from_score(0.7, panic_threshold, critical_threshold),
            PanicLevel::Panic
        );
        assert_eq!(
            PanicLevel::from_score(0.9, panic_threshold, critical_threshold),
            PanicLevel::Critical
        );
    }

    #[test]
    fn test_panic_action_mapping() {
        assert_eq!(PanicAction::from(PanicLevel::None), PanicAction::None);
        assert_eq!(
            PanicAction::from(PanicLevel::Elevated),
            PanicAction::IncreaseMonitoring
        );
        assert_eq!(
            PanicAction::from(PanicLevel::High),
            PanicAction::ReduceExposure
        );
        assert_eq!(
            PanicAction::from(PanicLevel::Panic),
            PanicAction::HaltNewPositions
        );
        assert_eq!(
            PanicAction::from(PanicLevel::Critical),
            PanicAction::EmergencyExit
        );
    }

    #[test]
    fn test_stats_summary() {
        let mut detector = PanicDetection::new();

        for i in 0..20 {
            let snapshot = create_normal_snapshot(100.0, i * 1000);
            detector.process(&snapshot).unwrap();
        }

        let stats = detector.stats_summary();
        assert!(stats.sample_count > 0);
        assert!(stats.avg_volume > 0.0);
    }
}
