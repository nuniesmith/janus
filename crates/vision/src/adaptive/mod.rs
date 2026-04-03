//! Adaptive Thresholding & Dynamic Calibration
//!
//! This module provides tools for adapting trading decisions to market conditions:
//!
//! - **Regime Detection**: Identify market regimes (trending, ranging, volatile, calm)
//! - **Dynamic Thresholds**: Adjust confidence thresholds based on performance and volatility
//! - **Confidence Calibration**: Calibrate model outputs to actual probabilities
//!
//! # Examples
//!
//! ## Basic Regime Detection
//!
//! ```
//! use vision::adaptive::regime::{RegimeDetector, RegimeConfig, MarketRegime};
//!
//! let config = RegimeConfig::default();
//! let mut detector = RegimeDetector::new(config);
//!
//! // Update with new prices
//! for price in vec![100.0, 102.0, 105.0, 108.0, 112.0] {
//!     let regime = detector.update(price);
//!     println!("Current regime: {:?}", regime);
//! }
//! ```
//!
//! ## Dynamic Threshold Calibration
//!
//! ```
//! use vision::adaptive::threshold::{ThresholdCalibrator, ThresholdConfig};
//!
//! let config = ThresholdConfig::default();
//! let mut calibrator = ThresholdCalibrator::new(config);
//!
//! // Update with trade outcomes
//! calibrator.update(0.8, 100.0);  // confidence=0.8, profit=100
//! calibrator.update(0.6, -50.0);  // confidence=0.6, profit=-50
//!
//! let threshold = calibrator.current_threshold();
//! println!("Adjusted threshold: {}", threshold);
//! ```
//!
//! ## Confidence Calibration
//!
//! ```
//! use vision::adaptive::calibration::{CombinedCalibrator, CalibrationConfig};
//!
//! let config = CalibrationConfig::default();
//! let mut calibrator = CombinedCalibrator::new(config);
//!
//! // Add calibration samples
//! calibrator.add_sample(0.9, true);   // predicted=0.9, actual=true
//! calibrator.add_sample(0.3, false);  // predicted=0.3, actual=false
//!
//! // Calibrate new predictions
//! let calibrated = calibrator.calibrate(0.7);
//! println!("Calibrated confidence: {}", calibrated);
//! ```
//!
//! ## Complete Adaptive System
//!
//! ```
//! use vision::adaptive::{
//!     regime::{RegimeDetector, RegimeConfig, RegimeAdjuster},
//!     threshold::{AdaptiveThreshold, ThresholdConfig},
//!     calibration::{CombinedCalibrator, CalibrationConfig},
//! };
//!
//! // Initialize components
//! let regime_detector = RegimeDetector::new(RegimeConfig::default());
//! let regime_adjuster = RegimeAdjuster::default();
//! let adaptive_threshold = AdaptiveThreshold::new(ThresholdConfig::default());
//! let calibrator = CombinedCalibrator::new(CalibrationConfig::default());
//!
//! // In your trading loop:
//! // 1. Detect current regime
//! // 2. Get regime-adjusted threshold
//! // 3. Calibrate model confidence
//! // 4. Make trading decision
//! ```

pub mod calibration;
pub mod regime;
pub mod threshold;

pub use calibration::{
    CalibrationConfig, CalibrationMetrics, CombinedCalibrator, IsotonicCalibration, PlattScaling,
};
pub use regime::{MarketRegime, RegimeAdjuster, RegimeConfig, RegimeDetector, RegimeMultipliers};
pub use threshold::{
    AdaptiveThreshold, MultiLevelThreshold, PercentileThreshold, RiskLevel, ThresholdCalibrator,
    ThresholdConfig, ThresholdStats,
};

/// Complete adaptive system combining all components
pub struct AdaptiveSystem {
    pub regime_detector: RegimeDetector,
    pub regime_adjuster: RegimeAdjuster,
    pub threshold: AdaptiveThreshold,
    pub calibrator: CombinedCalibrator,
}

impl AdaptiveSystem {
    /// Create a new adaptive system with default configurations
    pub fn new() -> Self {
        Self {
            regime_detector: RegimeDetector::new(RegimeConfig::default()),
            regime_adjuster: RegimeAdjuster::default(),
            threshold: AdaptiveThreshold::new(ThresholdConfig::default()),
            calibrator: CombinedCalibrator::new(CalibrationConfig::default()),
        }
    }

    /// Create a new adaptive system with custom configurations
    pub fn with_configs(
        regime_config: RegimeConfig,
        threshold_config: ThresholdConfig,
        calibration_config: CalibrationConfig,
    ) -> Self {
        Self {
            regime_detector: RegimeDetector::new(regime_config),
            regime_adjuster: RegimeAdjuster::default(),
            threshold: AdaptiveThreshold::new(threshold_config),
            calibrator: CombinedCalibrator::new(calibration_config),
        }
    }

    /// Update the system with new market data
    ///
    /// Returns the current market regime
    pub fn update_market(&mut self, price: f64, volatility: f64) -> MarketRegime {
        let regime = self.regime_detector.update(price);
        self.threshold.update(0.0, 0.0, volatility); // Dummy update for volatility
        regime
    }

    /// Update with a trade outcome for threshold and calibration learning
    pub fn update_trade(&mut self, predicted_confidence: f64, profit: f64, volatility: f64) {
        // Update threshold calibrator
        self.threshold
            .update(predicted_confidence, profit, volatility);

        // Update confidence calibrator
        let actual_outcome = profit > 0.0;
        self.calibrator
            .add_sample(predicted_confidence, actual_outcome);
    }

    /// Process a model prediction through the adaptive system
    ///
    /// Returns (calibrated_confidence, adjusted_threshold, should_trade)
    pub fn process_prediction(&self, raw_confidence: f64) -> (f64, f64, bool) {
        // Step 1: Calibrate the raw confidence
        let calibrated_confidence = self.calibrator.calibrate(raw_confidence);

        // Step 2: Get regime-adjusted threshold
        let regime = self.regime_detector.current_regime();
        let base_threshold = self.threshold.get_threshold();
        let adjusted_threshold = self
            .regime_adjuster
            .adjust_confidence_threshold(base_threshold, regime)
            .clamp(0.0, 1.0); // Ensure threshold stays in valid range

        // Step 3: Make trading decision
        let should_trade = calibrated_confidence >= adjusted_threshold;

        (calibrated_confidence, adjusted_threshold, should_trade)
    }

    /// Get regime-adjusted position size
    pub fn get_adjusted_position_size(&self, base_size: f64) -> f64 {
        let regime = self.regime_detector.current_regime();
        self.regime_adjuster.adjust_position_size(base_size, regime)
    }

    /// Get current market regime
    pub fn current_regime(&self) -> MarketRegime {
        self.regime_detector.current_regime()
    }

    /// Get comprehensive system statistics
    pub fn get_stats(&self) -> AdaptiveSystemStats {
        AdaptiveSystemStats {
            regime: self.regime_detector.current_regime(),
            threshold_stats: self.threshold.get_stats(),
            calibration_metrics: self.calibrator.get_metrics(),
        }
    }

    /// Reset all components of the adaptive system
    pub fn reset(&mut self) {
        self.regime_detector.reset();
        self.threshold = AdaptiveThreshold::new(ThresholdConfig::default());
        self.calibrator.reset();
    }
}

impl Default for AdaptiveSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the complete adaptive system
#[derive(Debug, Clone)]
pub struct AdaptiveSystemStats {
    pub regime: MarketRegime,
    pub threshold_stats: ThresholdStats,
    pub calibration_metrics: CalibrationMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_system_initialization() {
        let system = AdaptiveSystem::new();
        assert_eq!(system.current_regime(), MarketRegime::Unknown);
    }

    #[test]
    fn test_adaptive_system_market_update() {
        let mut system = AdaptiveSystem::new();

        // Update with uptrending prices
        for i in 0..30 {
            let price = 100.0 + i as f64 * 2.0;
            let volatility = 0.02;
            system.update_market(price, volatility);
        }

        // Regime should be detected after enough data
        let regime = system.current_regime();
        assert_ne!(regime, MarketRegime::Unknown);
    }

    #[test]
    fn test_adaptive_system_trade_update() {
        let mut system = AdaptiveSystem::new();

        // Simulate trades
        for i in 0..30 {
            let confidence = 0.7 + (i as f64 * 0.01);
            let profit = if i % 2 == 0 { 100.0 } else { -50.0 };
            let volatility = 0.02;
            system.update_trade(confidence, profit, volatility);
        }

        let stats = system.get_stats();
        assert!(stats.threshold_stats.sample_count > 0);
    }

    #[test]
    fn test_adaptive_system_process_prediction() {
        let mut system = AdaptiveSystem::new();

        // Add some calibration data
        for _ in 0..50 {
            system.update_trade(0.8, 100.0, 0.02);
        }

        let (calibrated, threshold, should_trade) = system.process_prediction(0.75);

        assert!(calibrated >= 0.0 && calibrated <= 1.0);
        assert!(threshold >= 0.0 && threshold <= 1.0);
        assert_eq!(should_trade, calibrated >= threshold);
    }

    #[test]
    fn test_adaptive_system_position_sizing() {
        let mut system = AdaptiveSystem::new();

        // Create a trending market
        for i in 0..30 {
            system.update_market(100.0 + i as f64 * 2.0, 0.02);
        }

        let base_size = 1000.0;
        let adjusted_size = system.get_adjusted_position_size(base_size);

        assert!(adjusted_size > 0.0);
    }

    #[test]
    fn test_adaptive_system_reset() {
        let mut system = AdaptiveSystem::new();

        // Add data
        for i in 0..30 {
            system.update_market(100.0 + i as f64, 0.02);
            system.update_trade(0.8, 100.0, 0.02);
        }

        system.reset();

        assert_eq!(system.current_regime(), MarketRegime::Unknown);
    }

    #[test]
    fn test_adaptive_system_custom_configs() {
        let regime_config = RegimeConfig {
            trend_window: 15,
            volatility_window: 15,
            ..Default::default()
        };
        let threshold_config = ThresholdConfig {
            base_confidence: 0.65,
            ..Default::default()
        };
        let calibration_config = CalibrationConfig {
            num_bins: 5,
            ..Default::default()
        };

        let system =
            AdaptiveSystem::with_configs(regime_config, threshold_config, calibration_config);

        let stats = system.get_stats();
        assert_eq!(stats.regime, MarketRegime::Unknown);
    }

    #[test]
    fn test_adaptive_system_stats() {
        let mut system = AdaptiveSystem::new();

        for i in 0..30 {
            system.update_market(100.0 + i as f64, 0.02);
            system.update_trade(0.75, 50.0, 0.02);
        }

        let stats = system.get_stats();
        assert!(stats.calibration_metrics.sample_count > 0);
    }
}
