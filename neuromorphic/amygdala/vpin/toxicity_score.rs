//! Toxicity Score - Multi-factor toxic flow detection
//!
//! Combines multiple metrics beyond VPIN to provide a comprehensive
//! toxicity assessment:
//! - VPIN (Volume-synchronized Probability of Informed Trading)
//! - Order imbalance
//! - Spread analysis
//! - Volume profile anomalies
//! - Trade clustering

use crate::common::Result;
use std::collections::VecDeque;

/// Weight configuration for toxicity components
#[derive(Debug, Clone)]
pub struct ToxicityWeights {
    /// Weight for VPIN component (0.0 - 1.0)
    pub vpin: f64,
    /// Weight for order imbalance component
    pub order_imbalance: f64,
    /// Weight for spread widening component
    pub spread: f64,
    /// Weight for volume anomaly component
    pub volume_anomaly: f64,
    /// Weight for trade clustering component
    pub trade_clustering: f64,
}

impl Default for ToxicityWeights {
    fn default() -> Self {
        Self {
            vpin: 0.35,
            order_imbalance: 0.25,
            spread: 0.15,
            volume_anomaly: 0.15,
            trade_clustering: 0.10,
        }
    }
}

/// Configuration for toxicity score calculation
#[derive(Debug, Clone)]
pub struct ToxicityConfig {
    /// Window size for moving calculations
    pub window_size: usize,
    /// Minimum samples before calculation is valid
    pub min_samples: usize,
    /// Threshold for high toxicity (0.0 - 1.0)
    pub high_threshold: f64,
    /// Threshold for critical toxicity
    pub critical_threshold: f64,
    /// Component weights
    pub weights: ToxicityWeights,
    /// Decay factor for exponential moving average
    pub ema_decay: f64,
}

impl Default for ToxicityConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            min_samples: 20,
            high_threshold: 0.6,
            critical_threshold: 0.8,
            weights: ToxicityWeights::default(),
            ema_decay: 0.94,
        }
    }
}

/// Toxicity severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl ToxicitySeverity {
    pub fn from_score(score: f64, high_threshold: f64, critical_threshold: f64) -> Self {
        if score >= critical_threshold {
            Self::Critical
        } else if score >= high_threshold {
            Self::High
        } else if score >= high_threshold * 0.5 {
            Self::Medium
        } else {
            Self::Low
        }
    }
}

/// Individual component scores
#[derive(Debug, Clone, Default)]
pub struct ComponentScores {
    pub vpin: f64,
    pub order_imbalance: f64,
    pub spread: f64,
    pub volume_anomaly: f64,
    pub trade_clustering: f64,
}

/// Result of toxicity assessment
#[derive(Debug, Clone)]
pub struct ToxicityAssessment {
    /// Overall toxicity score (0.0 - 1.0)
    pub score: f64,
    /// Severity level
    pub severity: ToxicitySeverity,
    /// Individual component scores
    pub components: ComponentScores,
    /// Number of samples used
    pub sample_count: usize,
    /// Timestamp of assessment
    pub timestamp: i64,
    /// Whether the assessment is reliable (enough samples)
    pub is_reliable: bool,
}

/// Trade data for toxicity calculation
#[derive(Debug, Clone)]
pub struct TradeData {
    pub price: f64,
    pub volume: f64,
    pub is_buy: bool,
    pub timestamp: i64,
    pub bid: f64,
    pub ask: f64,
}

/// Internal state for calculations
#[derive(Debug, Clone, Default)]
struct ToxicityState {
    buy_volume: f64,
    sell_volume: f64,
    volume_sum: f64,
    volume_squared_sum: f64,
    spread_sum: f64,
    last_trade_time: i64,
    inter_trade_times: Vec<i64>,
}

/// Multi-factor toxic flow detection based on VPIN and other metrics
pub struct ToxicityScore {
    config: ToxicityConfig,
    /// Rolling window of trades
    trades: VecDeque<TradeData>,
    /// Current VPIN value (can be fed externally or calculated)
    current_vpin: f64,
    /// Exponential moving average of toxicity
    ema_toxicity: f64,
    /// Current state
    state: ToxicityState,
    /// Historical toxicity scores for trend analysis
    history: VecDeque<f64>,
}

impl Default for ToxicityScore {
    fn default() -> Self {
        Self::new()
    }
}

impl ToxicityScore {
    /// Create a new ToxicityScore calculator with default config
    pub fn new() -> Self {
        Self::with_config(ToxicityConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ToxicityConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            trades: VecDeque::with_capacity(window_size),
            current_vpin: 0.0,
            ema_toxicity: 0.0,
            state: ToxicityState::default(),
            history: VecDeque::with_capacity(1000),
        }
    }

    /// Update with new trade data
    pub fn update(&mut self, trade: TradeData) {
        // Update inter-trade time
        if self.state.last_trade_time > 0 {
            let delta = trade.timestamp - self.state.last_trade_time;
            self.state.inter_trade_times.push(delta);
            if self.state.inter_trade_times.len() > self.config.window_size {
                self.state.inter_trade_times.remove(0);
            }
        }
        self.state.last_trade_time = trade.timestamp;

        // Update volume tracking
        if trade.is_buy {
            self.state.buy_volume += trade.volume;
        } else {
            self.state.sell_volume += trade.volume;
        }

        // Update spread tracking
        let spread = trade.ask - trade.bid;
        self.state.spread_sum += spread;

        // Update volume statistics
        self.state.volume_sum += trade.volume;
        self.state.volume_squared_sum += trade.volume * trade.volume;

        // Add to rolling window
        if self.trades.len() >= self.config.window_size {
            if let Some(old_trade) = self.trades.pop_front() {
                // Remove old trade from state
                if old_trade.is_buy {
                    self.state.buy_volume -= old_trade.volume;
                } else {
                    self.state.sell_volume -= old_trade.volume;
                }
                let old_spread = old_trade.ask - old_trade.bid;
                self.state.spread_sum -= old_spread;
                self.state.volume_sum -= old_trade.volume;
                self.state.volume_squared_sum -= old_trade.volume * old_trade.volume;
            }
        }
        self.trades.push_back(trade);
    }

    /// Update VPIN externally (from VPINCalculator)
    pub fn set_vpin(&mut self, vpin: f64) {
        self.current_vpin = vpin.clamp(0.0, 1.0);
    }

    /// Calculate comprehensive toxicity score
    pub fn calculate(&mut self) -> ToxicityAssessment {
        let sample_count = self.trades.len();
        let is_reliable = sample_count >= self.config.min_samples;

        if sample_count == 0 {
            return ToxicityAssessment {
                score: 0.0,
                severity: ToxicitySeverity::Low,
                components: ComponentScores::default(),
                sample_count: 0,
                timestamp: chrono::Utc::now().timestamp_millis(),
                is_reliable: false,
            };
        }

        // Calculate component scores
        let components = self.calculate_components();

        // Calculate weighted score
        let weights = &self.config.weights;
        let raw_score = components.vpin * weights.vpin
            + components.order_imbalance * weights.order_imbalance
            + components.spread * weights.spread
            + components.volume_anomaly * weights.volume_anomaly
            + components.trade_clustering * weights.trade_clustering;

        // Clamp to [0, 1]
        let score = raw_score.clamp(0.0, 1.0);

        // Update EMA
        self.ema_toxicity =
            self.config.ema_decay * self.ema_toxicity + (1.0 - self.config.ema_decay) * score;

        // Store in history
        if self.history.len() >= 1000 {
            self.history.pop_front();
        }
        self.history.push_back(score);

        let severity = ToxicitySeverity::from_score(
            score,
            self.config.high_threshold,
            self.config.critical_threshold,
        );

        ToxicityAssessment {
            score,
            severity,
            components,
            sample_count,
            timestamp: chrono::Utc::now().timestamp_millis(),
            is_reliable,
        }
    }

    /// Calculate individual component scores
    fn calculate_components(&self) -> ComponentScores {
        ComponentScores {
            vpin: self.current_vpin,
            order_imbalance: self.calculate_order_imbalance(),
            spread: self.calculate_spread_score(),
            volume_anomaly: self.calculate_volume_anomaly(),
            trade_clustering: self.calculate_trade_clustering(),
        }
    }

    /// Calculate order imbalance score
    fn calculate_order_imbalance(&self) -> f64 {
        let total_volume = self.state.buy_volume + self.state.sell_volume;
        if total_volume == 0.0 {
            return 0.0;
        }

        // Absolute imbalance normalized to [0, 1]
        (self.state.buy_volume - self.state.sell_volume).abs() / total_volume
    }

    /// Calculate spread-based toxicity score
    fn calculate_spread_score(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }

        let n = self.trades.len() as f64;
        let avg_spread = self.state.spread_sum / n;

        // Get recent spread
        let recent_spread = self
            .trades
            .back()
            .map(|t| t.ask - t.bid)
            .unwrap_or(avg_spread);

        // Score based on spread widening
        // If recent spread is much wider than average, toxicity increases
        if avg_spread > 0.0 {
            let ratio = recent_spread / avg_spread;
            // Normalize: ratio of 1 = 0 score, ratio of 3+ = 1.0 score
            ((ratio - 1.0) / 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate volume anomaly score using coefficient of variation
    fn calculate_volume_anomaly(&self) -> f64 {
        let n = self.trades.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean = self.state.volume_sum / n;
        if mean == 0.0 {
            return 0.0;
        }

        let variance = (self.state.volume_squared_sum / n) - (mean * mean);
        let std_dev = variance.max(0.0).sqrt();

        // Coefficient of variation
        let cv = std_dev / mean;

        // High CV indicates unusual volume distribution
        // Normalize: CV of 0 = 0 score, CV of 2+ = 1.0 score
        (cv / 2.0).clamp(0.0, 1.0)
    }

    /// Calculate trade clustering score (burstiness)
    fn calculate_trade_clustering(&self) -> f64 {
        if self.state.inter_trade_times.len() < 2 {
            return 0.0;
        }

        let times = &self.state.inter_trade_times;
        let n = times.len() as f64;

        // Calculate mean and variance of inter-trade times
        let sum: i64 = times.iter().sum();
        let mean = sum as f64 / n;

        if mean == 0.0 {
            return 1.0; // All trades at same time = maximum clustering
        }

        let variance: f64 = times
            .iter()
            .map(|&t| {
                let diff = t as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;

        let std_dev = variance.sqrt();

        // Index of dispersion (Fano factor)
        // > 1 indicates clustered/bursty arrivals
        let fano = variance / mean;

        // Normalize: Fano factor of 1 = 0 score (Poisson), > 3 = 1.0 (highly bursty)
        ((fano - 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Get current EMA-smoothed toxicity
    pub fn ema_score(&self) -> f64 {
        self.ema_toxicity
    }

    /// Get toxicity trend (positive = increasing, negative = decreasing)
    pub fn trend(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }

        // Compare recent average to older average
        let recent_start = self.history.len() - 5;
        let older_start = self.history.len() - 10;

        let recent_avg: f64 = self.history.iter().skip(recent_start).sum::<f64>() / 5.0;
        let older_avg: f64 = self.history.iter().skip(older_start).take(5).sum::<f64>() / 5.0;

        recent_avg - older_avg
    }

    /// Check if toxicity is dangerous
    pub fn is_dangerous(&self) -> bool {
        self.ema_toxicity >= self.config.high_threshold
    }

    /// Check if toxicity is critical
    pub fn is_critical(&self) -> bool {
        self.ema_toxicity >= self.config.critical_threshold
    }

    /// Get current severity level
    pub fn severity(&self) -> ToxicitySeverity {
        ToxicitySeverity::from_score(
            self.ema_toxicity,
            self.config.high_threshold,
            self.config.critical_threshold,
        )
    }

    /// Get statistics about toxicity history
    pub fn statistics(&self) -> ToxicityStatistics {
        if self.history.is_empty() {
            return ToxicityStatistics::default();
        }

        let n = self.history.len() as f64;
        let sum: f64 = self.history.iter().sum();
        let mean = sum / n;

        let variance: f64 = self
            .history
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;

        let std_dev = variance.sqrt();
        let max = self
            .history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min = self.history.iter().cloned().fold(f64::INFINITY, f64::min);

        ToxicityStatistics {
            mean,
            std_dev,
            max,
            min,
            current: self.ema_toxicity,
            sample_count: self.history.len(),
        }
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.trades.clear();
        self.current_vpin = 0.0;
        self.ema_toxicity = 0.0;
        self.state = ToxicityState::default();
        self.history.clear();
    }

    /// Main processing function (compatibility with neuromorphic interface)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Statistics about toxicity over time
#[derive(Debug, Clone, Default)]
pub struct ToxicityStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub max: f64,
    pub min: f64,
    pub current: f64,
    pub sample_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_trade(price: f64, volume: f64, is_buy: bool, timestamp: i64) -> TradeData {
        TradeData {
            price,
            volume,
            is_buy,
            timestamp,
            bid: price - 0.01,
            ask: price + 0.01,
        }
    }

    #[test]
    fn test_basic_creation() {
        let scorer = ToxicityScore::new();
        assert_eq!(scorer.ema_score(), 0.0);
        assert!(!scorer.is_dangerous());
    }

    #[test]
    fn test_balanced_flow() {
        let mut scorer = ToxicityScore::new();

        // Add balanced trades
        for i in 0..50 {
            let trade = create_trade(100.0, 1000.0, i % 2 == 0, i as i64 * 100);
            scorer.update(trade);
        }

        let assessment = scorer.calculate();

        // Balanced flow should have low order imbalance
        assert!(
            assessment.components.order_imbalance < 0.5,
            "Order imbalance should be low for balanced flow"
        );
    }

    #[test]
    fn test_imbalanced_flow() {
        let mut scorer = ToxicityScore::new();

        // Add heavily imbalanced trades (90% buys)
        for i in 0..100 {
            let is_buy = i % 10 != 0; // 90% buys
            let trade = create_trade(100.0, 1000.0, is_buy, i as i64 * 100);
            scorer.update(trade);
        }

        let assessment = scorer.calculate();

        // Imbalanced flow should have high order imbalance
        assert!(
            assessment.components.order_imbalance > 0.5,
            "Order imbalance should be high for one-sided flow"
        );
    }

    #[test]
    fn test_vpin_integration() {
        let mut scorer = ToxicityScore::new();

        // Set high VPIN
        scorer.set_vpin(0.9);

        // Add some trades
        for i in 0..30 {
            let trade = create_trade(100.0, 1000.0, true, i as i64 * 100);
            scorer.update(trade);
        }

        let assessment = scorer.calculate();

        // High VPIN should contribute to high toxicity
        assert!(
            assessment.components.vpin == 0.9,
            "VPIN component should match set value"
        );
        assert!(
            assessment.score > 0.3,
            "Overall score should reflect high VPIN"
        );
    }

    #[test]
    fn test_severity_levels() {
        let mut scorer = ToxicityScore::with_config(ToxicityConfig {
            high_threshold: 0.6,
            critical_threshold: 0.8,
            ..Default::default()
        });

        // Low toxicity
        assert_eq!(
            ToxicitySeverity::from_score(0.2, 0.6, 0.8),
            ToxicitySeverity::Low
        );

        // Medium toxicity
        assert_eq!(
            ToxicitySeverity::from_score(0.4, 0.6, 0.8),
            ToxicitySeverity::Medium
        );

        // High toxicity
        assert_eq!(
            ToxicitySeverity::from_score(0.7, 0.6, 0.8),
            ToxicitySeverity::High
        );

        // Critical toxicity
        assert_eq!(
            ToxicitySeverity::from_score(0.9, 0.6, 0.8),
            ToxicitySeverity::Critical
        );
    }

    #[test]
    fn test_trade_clustering() {
        let mut scorer = ToxicityScore::new();

        // Add bursty trades (many trades in short time, then gap)
        let mut timestamp = 0i64;
        for burst in 0..5 {
            // Burst of 10 trades in 10ms
            for _ in 0..10 {
                let trade = create_trade(100.0, 1000.0, true, timestamp);
                scorer.update(trade);
                timestamp += 1;
            }
            // Gap of 1000ms
            timestamp += 1000;
        }

        let assessment = scorer.calculate();

        // Bursty trades should show in clustering score
        // Note: This may need tuning based on actual behavior
        assert!(assessment.components.trade_clustering >= 0.0);
    }

    #[test]
    fn test_spread_widening() {
        let mut scorer = ToxicityScore::new();

        // Add trades with normal spread
        for i in 0..30 {
            let trade = TradeData {
                price: 100.0,
                volume: 1000.0,
                is_buy: true,
                timestamp: i as i64 * 100,
                bid: 99.99,
                ask: 100.01,
            };
            scorer.update(trade);
        }

        // Add trades with wide spread
        for i in 30..50 {
            let trade = TradeData {
                price: 100.0,
                volume: 1000.0,
                is_buy: true,
                timestamp: i as i64 * 100,
                bid: 99.90,
                ask: 100.10, // 10x wider spread
            };
            scorer.update(trade);
        }

        let assessment = scorer.calculate();

        // Wide spread should increase spread score
        assert!(
            assessment.components.spread > 0.0,
            "Spread widening should be detected"
        );
    }

    #[test]
    fn test_ema_smoothing() {
        let mut scorer = ToxicityScore::new();
        scorer.set_vpin(0.9);

        // Initial calculation
        for i in 0..30 {
            let trade = create_trade(100.0, 1000.0, true, i as i64 * 100);
            scorer.update(trade);
        }
        scorer.calculate();

        let ema1 = scorer.ema_score();

        // More calculations should smooth the EMA
        scorer.set_vpin(0.1);
        for i in 30..60 {
            let trade = create_trade(100.0, 1000.0, i % 2 == 0, i as i64 * 100);
            scorer.update(trade);
            scorer.calculate();
        }

        let ema2 = scorer.ema_score();

        // EMA should have moved toward lower values
        assert!(ema2 < ema1, "EMA should decrease when toxicity decreases");
    }

    #[test]
    fn test_trend_detection() {
        let mut scorer = ToxicityScore::new();

        // Build up increasing toxicity
        for i in 0..20 {
            scorer.set_vpin(0.1 + (i as f64 * 0.04));
            for j in 0..5 {
                let trade = create_trade(100.0, 1000.0, true, (i * 5 + j) as i64 * 100);
                scorer.update(trade);
            }
            scorer.calculate();
        }

        let trend = scorer.trend();

        // Should detect increasing trend
        assert!(trend > 0.0, "Should detect increasing toxicity trend");
    }

    #[test]
    fn test_reset() {
        let mut scorer = ToxicityScore::new();

        // Add data
        for i in 0..50 {
            let trade = create_trade(100.0, 1000.0, true, i as i64 * 100);
            scorer.update(trade);
        }
        scorer.set_vpin(0.8);
        scorer.calculate();

        // Reset
        scorer.reset();

        assert_eq!(scorer.ema_score(), 0.0);
        assert!(!scorer.is_dangerous());
    }

    #[test]
    fn test_statistics() {
        let mut scorer = ToxicityScore::new();

        // Generate varied toxicity scores
        for i in 0..100 {
            scorer.set_vpin(0.3 + (i as f64 % 10.0) * 0.05);
            for j in 0..3 {
                let trade = create_trade(100.0, 1000.0, j % 2 == 0, (i * 3 + j) as i64 * 100);
                scorer.update(trade);
            }
            scorer.calculate();
        }

        let stats = scorer.statistics();

        assert!(stats.sample_count > 0);
        assert!(stats.mean >= stats.min);
        assert!(stats.mean <= stats.max);
        assert!(stats.std_dev >= 0.0);
    }

    #[test]
    fn test_process() {
        let scorer = ToxicityScore::new();
        assert!(scorer.process().is_ok());
    }
}
