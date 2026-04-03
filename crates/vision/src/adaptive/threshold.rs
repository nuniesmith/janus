//! Dynamic Threshold Calibration
//!
//! This module provides adaptive thresholding mechanisms that adjust decision
//! thresholds based on market conditions, volatility, and historical performance.

use std::collections::VecDeque;

/// Configuration for dynamic threshold calibration
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    /// Base confidence threshold for signals (0-1)
    pub base_confidence: f64,
    /// Lookback window for performance tracking
    pub lookback_window: usize,
    /// Minimum samples before adjusting thresholds
    pub min_samples: usize,
    /// Maximum threshold adjustment factor
    pub max_adjustment: f64,
    /// Target win rate for calibration
    pub target_win_rate: f64,
    /// Learning rate for threshold updates
    pub learning_rate: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            base_confidence: 0.7,
            lookback_window: 100,
            min_samples: 20,
            max_adjustment: 0.3,
            target_win_rate: 0.55,
            learning_rate: 0.01,
        }
    }
}

/// Dynamic threshold calibrator
pub struct ThresholdCalibrator {
    config: ThresholdConfig,
    current_threshold: f64,
    performance_history: VecDeque<TradeOutcome>,
    volatility_buffer: VecDeque<f64>,
}

#[derive(Debug, Clone)]
struct TradeOutcome {
    #[allow(dead_code)]
    confidence: f64,
    was_profitable: bool,
    profit: f64,
}

impl ThresholdCalibrator {
    /// Create a new threshold calibrator
    pub fn new(config: ThresholdConfig) -> Self {
        let current_threshold = config.base_confidence;
        let lookback_window = config.lookback_window;
        Self {
            config,
            current_threshold,
            performance_history: VecDeque::with_capacity(lookback_window),
            volatility_buffer: VecDeque::with_capacity(lookback_window),
        }
    }

    /// Get the current adjusted threshold
    pub fn current_threshold(&self) -> f64 {
        self.current_threshold
    }

    /// Update threshold based on a trade outcome
    pub fn update(&mut self, confidence: f64, profit: f64) {
        let outcome = TradeOutcome {
            confidence,
            was_profitable: profit > 0.0,
            profit,
        };

        self.performance_history.push_back(outcome);

        // Maintain window size
        while self.performance_history.len() > self.config.lookback_window {
            self.performance_history.pop_front();
        }

        // Recalibrate if we have enough samples
        if self.performance_history.len() >= self.config.min_samples {
            self.calibrate();
        }
    }

    /// Update volatility for volatility-adjusted thresholds
    pub fn update_volatility(&mut self, volatility: f64) {
        self.volatility_buffer.push_back(volatility);
        while self.volatility_buffer.len() > self.config.lookback_window {
            self.volatility_buffer.pop_front();
        }
    }

    /// Calibrate the threshold based on performance
    fn calibrate(&mut self) {
        let current_win_rate = self.calculate_win_rate();
        let error = self.config.target_win_rate - current_win_rate;

        // Adjust threshold: if win rate too low, lower threshold; if too high, raise it
        let adjustment = -error * self.config.learning_rate;

        self.current_threshold += adjustment;

        // Clamp to reasonable bounds
        let min_threshold = self.config.base_confidence - self.config.max_adjustment;
        let max_threshold = self.config.base_confidence + self.config.max_adjustment;
        self.current_threshold = self.current_threshold.clamp(min_threshold, max_threshold);
    }

    /// Calculate current win rate
    fn calculate_win_rate(&self) -> f64 {
        if self.performance_history.is_empty() {
            return 0.0;
        }

        let wins = self
            .performance_history
            .iter()
            .filter(|o| o.was_profitable)
            .count();
        wins as f64 / self.performance_history.len() as f64
    }

    /// Get volatility-adjusted threshold
    pub fn volatility_adjusted_threshold(&self) -> f64 {
        if self.volatility_buffer.is_empty() {
            return self.current_threshold;
        }

        let current_vol = self.volatility_buffer.back().copied().unwrap_or(0.0);
        let avg_vol =
            self.volatility_buffer.iter().sum::<f64>() / self.volatility_buffer.len() as f64;

        if avg_vol == 0.0 {
            return self.current_threshold;
        }

        // In high volatility, require higher confidence
        let vol_ratio = current_vol / avg_vol;
        let vol_adjustment = (vol_ratio - 1.0) * 0.1; // 10% adjustment per unit volatility increase

        let adjusted = self.current_threshold + vol_adjustment;
        adjusted.clamp(0.0, 1.0)
    }

    /// Get confidence-based threshold (for different confidence levels)
    pub fn get_threshold_for_confidence(&self, confidence: f64) -> bool {
        confidence >= self.volatility_adjusted_threshold()
    }

    /// Reset the calibrator
    pub fn reset(&mut self) {
        self.current_threshold = self.config.base_confidence;
        self.performance_history.clear();
        self.volatility_buffer.clear();
    }

    /// Get performance statistics
    pub fn statistics(&self) -> ThresholdStats {
        let win_rate = self.calculate_win_rate();
        let avg_profit = if self.performance_history.is_empty() {
            0.0
        } else {
            self.performance_history
                .iter()
                .map(|o| o.profit)
                .sum::<f64>()
                / self.performance_history.len() as f64
        };

        let avg_winning_profit = {
            let wins: Vec<f64> = self
                .performance_history
                .iter()
                .filter(|o| o.was_profitable)
                .map(|o| o.profit)
                .collect();
            if wins.is_empty() {
                0.0
            } else {
                wins.iter().sum::<f64>() / wins.len() as f64
            }
        };

        let avg_losing_profit = {
            let losses: Vec<f64> = self
                .performance_history
                .iter()
                .filter(|o| !o.was_profitable)
                .map(|o| o.profit)
                .collect();
            if losses.is_empty() {
                0.0
            } else {
                losses.iter().sum::<f64>() / losses.len() as f64
            }
        };

        ThresholdStats {
            current_threshold: self.current_threshold,
            base_threshold: self.config.base_confidence,
            win_rate,
            avg_profit,
            avg_winning_profit,
            avg_losing_profit,
            sample_count: self.performance_history.len(),
        }
    }
}

/// Statistics about threshold performance
#[derive(Debug, Clone)]
pub struct ThresholdStats {
    pub current_threshold: f64,
    pub base_threshold: f64,
    pub win_rate: f64,
    pub avg_profit: f64,
    pub avg_winning_profit: f64,
    pub avg_losing_profit: f64,
    pub sample_count: usize,
}

/// Multi-level threshold system
pub struct MultiLevelThreshold {
    /// Conservative threshold (high confidence required)
    pub conservative: f64,
    /// Moderate threshold (medium confidence required)
    pub moderate: f64,
    /// Aggressive threshold (low confidence required)
    pub aggressive: f64,
}

impl MultiLevelThreshold {
    /// Create new multi-level thresholds
    pub fn new(base_threshold: f64) -> Self {
        Self {
            conservative: base_threshold + 0.15,
            moderate: base_threshold,
            aggressive: base_threshold - 0.15,
        }
    }

    /// Get threshold for a risk level
    pub fn get_threshold(&self, risk_level: RiskLevel) -> f64 {
        match risk_level {
            RiskLevel::Conservative => self.conservative,
            RiskLevel::Moderate => self.moderate,
            RiskLevel::Aggressive => self.aggressive,
        }
    }

    /// Adjust all thresholds by a factor
    pub fn adjust(&mut self, factor: f64) {
        self.conservative = (self.conservative * factor).clamp(0.0, 1.0);
        self.moderate = (self.moderate * factor).clamp(0.0, 1.0);
        self.aggressive = (self.aggressive * factor).clamp(0.0, 1.0);
    }

    /// Adjust based on volatility
    pub fn adjust_for_volatility(&mut self, volatility_ratio: f64) {
        // Higher volatility -> higher thresholds
        let adjustment = (volatility_ratio - 1.0) * 0.1;
        self.conservative += adjustment;
        self.moderate += adjustment;
        self.aggressive += adjustment;

        // Clamp to valid range
        self.conservative = self.conservative.clamp(0.0, 1.0);
        self.moderate = self.moderate.clamp(0.0, 1.0);
        self.aggressive = self.aggressive.clamp(0.0, 1.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Conservative,
    Moderate,
    Aggressive,
}

/// Percentile-based threshold calculator
pub struct PercentileThreshold {
    confidence_history: VecDeque<f64>,
    window_size: usize,
}

impl PercentileThreshold {
    /// Create a new percentile threshold calculator
    pub fn new(window_size: usize) -> Self {
        Self {
            confidence_history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Update with a new confidence value
    pub fn update(&mut self, confidence: f64) {
        self.confidence_history.push_back(confidence);
        while self.confidence_history.len() > self.window_size {
            self.confidence_history.pop_front();
        }
    }

    /// Get threshold at a specific percentile
    pub fn get_threshold_at_percentile(&self, percentile: f64) -> f64 {
        if self.confidence_history.is_empty() {
            return 0.5;
        }

        let mut sorted: Vec<f64> = self.confidence_history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((sorted.len() - 1) as f64 * percentile) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Get top N percent threshold
    pub fn get_top_n_percent_threshold(&self, top_percent: f64) -> f64 {
        self.get_threshold_at_percentile(1.0 - top_percent)
    }
}

/// Adaptive threshold that combines multiple signals
pub struct AdaptiveThreshold {
    calibrator: ThresholdCalibrator,
    #[allow(dead_code)]
    multi_level: MultiLevelThreshold,
    percentile: PercentileThreshold,
    current_risk_level: RiskLevel,
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold system
    pub fn new(config: ThresholdConfig) -> Self {
        let base_threshold = config.base_confidence;
        Self {
            calibrator: ThresholdCalibrator::new(config.clone()),
            multi_level: MultiLevelThreshold::new(base_threshold),
            percentile: PercentileThreshold::new(config.lookback_window),
            current_risk_level: RiskLevel::Moderate,
        }
    }

    /// Update with trade outcome and confidence
    pub fn update(&mut self, confidence: f64, profit: f64, volatility: f64) {
        self.calibrator.update(confidence, profit);
        self.calibrator.update_volatility(volatility);
        self.percentile.update(confidence);
    }

    /// Get the current threshold for the current risk level
    pub fn get_threshold(&self) -> f64 {
        let base = self.calibrator.volatility_adjusted_threshold();
        let level_adjustment = match self.current_risk_level {
            RiskLevel::Conservative => 0.15,
            RiskLevel::Moderate => 0.0,
            RiskLevel::Aggressive => -0.15,
        };
        (base + level_adjustment).clamp(0.0, 1.0)
    }

    /// Set risk level
    pub fn set_risk_level(&mut self, level: RiskLevel) {
        self.current_risk_level = level;
    }

    /// Check if confidence meets threshold
    pub fn should_trade(&self, confidence: f64) -> bool {
        confidence >= self.get_threshold()
    }

    /// Get comprehensive statistics
    pub fn get_stats(&self) -> ThresholdStats {
        self.calibrator.statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_calibrator_initialization() {
        let config = ThresholdConfig::default();
        let calibrator = ThresholdCalibrator::new(config.clone());
        assert_eq!(calibrator.current_threshold(), config.base_confidence);
    }

    #[test]
    fn test_threshold_calibrator_update() {
        let config = ThresholdConfig {
            min_samples: 5,
            lookback_window: 10,
            ..Default::default()
        };
        let mut calibrator = ThresholdCalibrator::new(config);

        // Add some profitable trades
        for _ in 0..10 {
            calibrator.update(0.8, 100.0);
        }

        let stats = calibrator.statistics();
        assert_eq!(stats.win_rate, 1.0);
        assert!(stats.avg_profit > 0.0);
    }

    #[test]
    fn test_threshold_adjustment_low_win_rate() {
        let config = ThresholdConfig {
            min_samples: 5,
            target_win_rate: 0.6,
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut calibrator = ThresholdCalibrator::new(config.clone());
        let initial_threshold = calibrator.current_threshold();

        // Add many losing trades (low win rate)
        for _ in 0..10 {
            calibrator.update(0.7, -100.0);
        }

        // Threshold should decrease to allow more trades
        assert!(
            calibrator.current_threshold() < initial_threshold,
            "Threshold should decrease with low win rate"
        );
    }

    #[test]
    fn test_threshold_adjustment_high_win_rate() {
        let config = ThresholdConfig {
            min_samples: 5,
            target_win_rate: 0.5,
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut calibrator = ThresholdCalibrator::new(config.clone());
        let initial_threshold = calibrator.current_threshold();

        // Add many winning trades (high win rate)
        for _ in 0..10 {
            calibrator.update(0.7, 100.0);
        }

        // Threshold should increase to be more selective
        assert!(
            calibrator.current_threshold() > initial_threshold,
            "Threshold should increase with high win rate"
        );
    }

    #[test]
    fn test_volatility_adjusted_threshold() {
        let config = ThresholdConfig::default();
        let mut calibrator = ThresholdCalibrator::new(config);

        // Add normal volatility
        for _ in 0..10 {
            calibrator.update_volatility(0.02);
        }

        let normal_threshold = calibrator.volatility_adjusted_threshold();

        // Add high volatility
        calibrator.update_volatility(0.06);

        let high_vol_threshold = calibrator.volatility_adjusted_threshold();

        // High volatility should result in higher threshold
        assert!(high_vol_threshold > normal_threshold);
    }

    #[test]
    fn test_multi_level_threshold() {
        let threshold = MultiLevelThreshold::new(0.7);

        assert!(threshold.conservative > threshold.moderate);
        assert!(threshold.moderate > threshold.aggressive);

        assert_eq!(
            threshold.get_threshold(RiskLevel::Conservative),
            threshold.conservative
        );
        assert_eq!(
            threshold.get_threshold(RiskLevel::Moderate),
            threshold.moderate
        );
        assert_eq!(
            threshold.get_threshold(RiskLevel::Aggressive),
            threshold.aggressive
        );
    }

    #[test]
    fn test_multi_level_adjust() {
        let mut threshold = MultiLevelThreshold::new(0.7);
        let original_moderate = threshold.moderate;

        threshold.adjust(1.1);

        assert!(threshold.moderate > original_moderate);
    }

    #[test]
    fn test_multi_level_volatility_adjustment() {
        let mut threshold = MultiLevelThreshold::new(0.7);
        let original = threshold.moderate;

        // High volatility (2x normal)
        threshold.adjust_for_volatility(2.0);

        assert!(threshold.moderate > original);
    }

    #[test]
    fn test_percentile_threshold() {
        let mut percentile = PercentileThreshold::new(100);

        // Add confidence values
        for i in 0..100 {
            percentile.update(i as f64 / 100.0);
        }

        let median = percentile.get_threshold_at_percentile(0.5);
        assert!((median - 0.5).abs() < 0.1);

        let top_10 = percentile.get_top_n_percent_threshold(0.1);
        assert!(top_10 > 0.85);
    }

    #[test]
    fn test_adaptive_threshold() {
        let config = ThresholdConfig {
            min_samples: 5,
            ..Default::default()
        };
        let mut adaptive = AdaptiveThreshold::new(config);

        adaptive.set_risk_level(RiskLevel::Conservative);
        let conservative_threshold = adaptive.get_threshold();

        adaptive.set_risk_level(RiskLevel::Aggressive);
        let aggressive_threshold = adaptive.get_threshold();

        assert!(conservative_threshold > aggressive_threshold);
    }

    #[test]
    fn test_adaptive_should_trade() {
        let config = ThresholdConfig::default();
        let mut adaptive = AdaptiveThreshold::new(config);

        adaptive.set_risk_level(RiskLevel::Moderate);

        // High confidence should pass
        assert!(adaptive.should_trade(0.95));

        // Low confidence should fail
        assert!(!adaptive.should_trade(0.3));
    }

    #[test]
    fn test_calibrator_reset() {
        let config = ThresholdConfig::default();
        let mut calibrator = ThresholdCalibrator::new(config.clone());

        for _ in 0..10 {
            calibrator.update(0.8, 100.0);
        }

        calibrator.reset();

        assert_eq!(calibrator.current_threshold(), config.base_confidence);
        assert_eq!(calibrator.performance_history.len(), 0);
    }

    #[test]
    fn test_threshold_bounds() {
        let config = ThresholdConfig {
            min_samples: 5,
            max_adjustment: 0.2,
            learning_rate: 1.0, // High learning rate to test bounds
            ..Default::default()
        };
        let mut calibrator = ThresholdCalibrator::new(config.clone());

        // Try to push threshold very high with many wins
        for _ in 0..100 {
            calibrator.update(0.9, 1000.0);
        }

        let max_expected = config.base_confidence + config.max_adjustment;
        assert!(calibrator.current_threshold() <= max_expected);

        calibrator.reset();

        // Try to push threshold very low with many losses
        for _ in 0..100 {
            calibrator.update(0.5, -1000.0);
        }

        let min_expected = config.base_confidence - config.max_adjustment;
        assert!(calibrator.current_threshold() >= min_expected);
    }

    #[test]
    fn test_statistics() {
        let config = ThresholdConfig {
            min_samples: 5,
            ..Default::default()
        };
        let mut calibrator = ThresholdCalibrator::new(config);

        calibrator.update(0.8, 100.0);
        calibrator.update(0.7, -50.0);
        calibrator.update(0.9, 150.0);

        let stats = calibrator.statistics();

        assert_eq!(stats.sample_count, 3);
        assert!(stats.win_rate > 0.0 && stats.win_rate < 1.0);
        assert!(stats.avg_profit != 0.0);
        assert!(stats.avg_winning_profit > 0.0);
        assert!(stats.avg_losing_profit < 0.0);
    }
}
