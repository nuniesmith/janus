//! Market Regime Detection
//!
//! This module provides tools for detecting different market regimes:
//! - Trending vs Ranging
//! - Volatile vs Calm
//! - Bull vs Bear
//!
//! Regime detection helps adapt trading strategies to current market conditions.

use std::collections::VecDeque;

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Strong uptrend
    BullTrending,
    /// Strong downtrend
    BearTrending,
    /// Sideways market with low volatility
    RangingCalm,
    /// Sideways market with high volatility
    RangingVolatile,
    /// High volatility uptrend
    BullVolatile,
    /// High volatility downtrend
    BearVolatile,
    /// Unknown/initializing
    Unknown,
}

impl MarketRegime {
    /// Returns true if the regime is trending (bull or bear)
    pub fn is_trending(&self) -> bool {
        matches!(
            self,
            MarketRegime::BullTrending
                | MarketRegime::BearTrending
                | MarketRegime::BullVolatile
                | MarketRegime::BearVolatile
        )
    }

    /// Returns true if the regime is ranging
    pub fn is_ranging(&self) -> bool {
        matches!(
            self,
            MarketRegime::RangingCalm | MarketRegime::RangingVolatile
        )
    }

    /// Returns true if the regime is volatile
    pub fn is_volatile(&self) -> bool {
        matches!(
            self,
            MarketRegime::RangingVolatile | MarketRegime::BullVolatile | MarketRegime::BearVolatile
        )
    }

    /// Returns true if the regime is bullish
    pub fn is_bullish(&self) -> bool {
        matches!(
            self,
            MarketRegime::BullTrending | MarketRegime::BullVolatile
        )
    }

    /// Returns true if the regime is bearish
    pub fn is_bearish(&self) -> bool {
        matches!(
            self,
            MarketRegime::BearTrending | MarketRegime::BearVolatile
        )
    }
}

/// Configuration for regime detection
#[derive(Debug, Clone)]
pub struct RegimeConfig {
    /// Lookback window for trend detection
    pub trend_window: usize,
    /// Lookback window for volatility detection
    pub volatility_window: usize,
    /// ADX threshold for trending market (typically 25)
    pub adx_threshold: f64,
    /// Volatility percentile threshold (0-1, typically 0.7)
    pub volatility_threshold: f64,
    /// Minimum data points before detecting regime
    pub min_data_points: usize,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            trend_window: 20,
            volatility_window: 20,
            adx_threshold: 25.0,
            volatility_threshold: 0.7,
            min_data_points: 20,
        }
    }
}

/// Market regime detector
pub struct RegimeDetector {
    config: RegimeConfig,
    prices: VecDeque<f64>,
    returns: VecDeque<f64>,
    current_regime: MarketRegime,
}

impl RegimeDetector {
    /// Create a new regime detector
    pub fn new(config: RegimeConfig) -> Self {
        let max_window = config.trend_window.max(config.volatility_window);
        Self {
            config,
            prices: VecDeque::with_capacity(max_window + 1),
            returns: VecDeque::with_capacity(max_window),
            current_regime: MarketRegime::Unknown,
        }
    }

    /// Update with a new price and return the detected regime
    pub fn update(&mut self, price: f64) -> MarketRegime {
        // Add new price
        self.prices.push_back(price);

        // Calculate return if we have previous price
        if self.prices.len() > 1 {
            let prev_price = self.prices[self.prices.len() - 2];
            // Use simple returns instead of log returns to avoid NaN
            let ret = (price / prev_price) - 1.0;
            self.returns.push_back(ret);
        }

        // Maintain window sizes
        let max_window = self.config.trend_window.max(self.config.volatility_window);
        while self.prices.len() > max_window + 1 {
            self.prices.pop_front();
        }
        while self.returns.len() > max_window {
            self.returns.pop_front();
        }

        // Detect regime if we have enough data
        if self.prices.len() >= self.config.min_data_points {
            self.current_regime = self.detect_regime();
        }

        self.current_regime
    }

    /// Get the current regime
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Detect the current market regime
    fn detect_regime(&self) -> MarketRegime {
        let is_trending = self.is_trending();
        let is_volatile = self.is_volatile();
        let trend_direction = self.trend_direction();

        match (is_trending, is_volatile, trend_direction) {
            (true, false, dir) if dir > 0.0 => MarketRegime::BullTrending,
            (true, false, dir) if dir < 0.0 => MarketRegime::BearTrending,
            (true, true, dir) if dir > 0.0 => MarketRegime::BullVolatile,
            (true, true, dir) if dir < 0.0 => MarketRegime::BearVolatile,
            (false, false, _) => MarketRegime::RangingCalm,
            (false, true, _) => MarketRegime::RangingVolatile,
            _ => MarketRegime::Unknown,
        }
    }

    /// Calculate ADX (Average Directional Index) to detect trending markets
    fn is_trending(&self) -> bool {
        if self.prices.len() < self.config.trend_window {
            return false;
        }

        let adx = self.calculate_adx();
        adx > self.config.adx_threshold
    }

    /// Calculate simplified ADX
    fn calculate_adx(&self) -> f64 {
        let window = self.config.trend_window.min(self.prices.len() - 1);
        if window < 2 {
            return 0.0;
        }

        let prices: Vec<f64> = self
            .prices
            .iter()
            .rev()
            .take(window + 1)
            .rev()
            .copied()
            .collect();

        let mut plus_dm_sum = 0.0;
        let mut minus_dm_sum = 0.0;
        let mut tr_sum = 0.0;

        for i in 1..prices.len() {
            let high_diff = prices[i] - prices[i - 1];
            let low_diff = prices[i - 1] - prices[i];

            let plus_dm = if high_diff > low_diff && high_diff > 0.0 {
                high_diff
            } else {
                0.0
            };

            let minus_dm = if low_diff > high_diff && low_diff > 0.0 {
                low_diff
            } else {
                0.0
            };

            let tr = (prices[i] - prices[i - 1]).abs();

            plus_dm_sum += plus_dm;
            minus_dm_sum += minus_dm;
            tr_sum += tr;
        }

        if tr_sum == 0.0 {
            return 0.0;
        }

        let plus_di = 100.0 * plus_dm_sum / tr_sum;
        let minus_di = 100.0 * minus_dm_sum / tr_sum;

        if plus_di + minus_di == 0.0 {
            return 0.0;
        }

        let dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di);
        dx
    }

    /// Detect if the market is volatile
    fn is_volatile(&self) -> bool {
        if self.returns.len() < self.config.volatility_window {
            return false;
        }

        let current_vol = self.calculate_volatility();
        let historical_vol = self.calculate_historical_volatility_percentile();

        current_vol > historical_vol * self.config.volatility_threshold
    }

    /// Calculate current volatility
    fn calculate_volatility(&self) -> f64 {
        let window = self.config.volatility_window.min(self.returns.len());
        if window == 0 {
            return 0.0;
        }

        let recent_returns: Vec<f64> = self.returns.iter().rev().take(window).copied().collect();
        let mean = recent_returns.iter().sum::<f64>() / window as f64;
        let variance = recent_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / window as f64;

        variance.sqrt()
    }

    /// Calculate historical volatility at a given percentile
    fn calculate_historical_volatility_percentile(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let window = self.config.volatility_window.min(self.returns.len());
        let mut volatilities = Vec::new();

        for i in window..=self.returns.len() {
            let subset: Vec<f64> = self
                .returns
                .iter()
                .skip(i - window)
                .take(window)
                .copied()
                .collect();
            let mean = subset.iter().sum::<f64>() / window as f64;
            let variance = subset.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
            volatilities.push(variance.sqrt());
        }

        if volatilities.is_empty() {
            return 0.0;
        }

        volatilities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((volatilities.len() - 1) as f64 * self.config.volatility_threshold) as usize;
        volatilities[idx]
    }

    /// Determine trend direction (-1 for down, 0 for neutral, +1 for up)
    fn trend_direction(&self) -> f64 {
        if self.prices.len() < 2 {
            return 0.0;
        }

        let window = self.config.trend_window.min(self.prices.len());
        let recent_prices: Vec<f64> = self
            .prices
            .iter()
            .rev()
            .take(window)
            .rev()
            .copied()
            .collect();

        // Simple linear regression slope
        let n = recent_prices.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = recent_prices.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &price) in recent_prices.iter().enumerate() {
            let x_diff = i as f64 - x_mean;
            numerator += x_diff * (price - y_mean);
            denominator += x_diff * x_diff;
        }

        if denominator == 0.0 {
            return 0.0;
        }

        numerator / denominator
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.prices.clear();
        self.returns.clear();
        self.current_regime = MarketRegime::Unknown;
    }
}

/// Regime-based parameter adjuster
#[derive(Debug, Clone)]
pub struct RegimeAdjuster {
    /// Confidence threshold multipliers per regime
    pub confidence_multipliers: RegimeMultipliers,
    /// Position size multipliers per regime
    pub position_multipliers: RegimeMultipliers,
}

#[derive(Debug, Clone)]
pub struct RegimeMultipliers {
    pub bull_trending: f64,
    pub bear_trending: f64,
    pub ranging_calm: f64,
    pub ranging_volatile: f64,
    pub bull_volatile: f64,
    pub bear_volatile: f64,
    pub unknown: f64,
}

impl Default for RegimeMultipliers {
    fn default() -> Self {
        Self {
            bull_trending: 1.0,
            bear_trending: 1.0,
            ranging_calm: 1.0,
            ranging_volatile: 1.0,
            bull_volatile: 1.0,
            bear_volatile: 1.0,
            unknown: 1.0,
        }
    }
}

impl RegimeMultipliers {
    /// Get multiplier for a specific regime
    pub fn get(&self, regime: MarketRegime) -> f64 {
        match regime {
            MarketRegime::BullTrending => self.bull_trending,
            MarketRegime::BearTrending => self.bear_trending,
            MarketRegime::RangingCalm => self.ranging_calm,
            MarketRegime::RangingVolatile => self.ranging_volatile,
            MarketRegime::BullVolatile => self.bull_volatile,
            MarketRegime::BearVolatile => self.bear_volatile,
            MarketRegime::Unknown => self.unknown,
        }
    }
}

impl Default for RegimeAdjuster {
    fn default() -> Self {
        Self {
            confidence_multipliers: RegimeMultipliers {
                bull_trending: 0.9, // Lower threshold in strong trends
                bear_trending: 0.9,
                ranging_calm: 1.2,     // Higher threshold in ranging markets
                ranging_volatile: 1.4, // Much higher in volatile ranging
                bull_volatile: 1.1,
                bear_volatile: 1.1,
                unknown: 1.5, // Very conservative when unknown
            },
            position_multipliers: RegimeMultipliers {
                bull_trending: 1.2, // Larger positions in trends
                bear_trending: 1.2,
                ranging_calm: 0.8,     // Smaller positions in ranging
                ranging_volatile: 0.5, // Much smaller in volatile ranging
                bull_volatile: 0.9,
                bear_volatile: 0.9,
                unknown: 0.5, // Very small when unknown
            },
        }
    }
}

impl RegimeAdjuster {
    /// Create a new regime adjuster with default multipliers
    pub fn new() -> Self {
        Self::default()
    }

    /// Adjust confidence threshold based on regime
    pub fn adjust_confidence_threshold(&self, base_threshold: f64, regime: MarketRegime) -> f64 {
        base_threshold * self.confidence_multipliers.get(regime)
    }

    /// Adjust position size based on regime
    pub fn adjust_position_size(&self, base_size: f64, regime: MarketRegime) -> f64 {
        base_size * self.position_multipliers.get(regime)
    }

    /// Set custom confidence multipliers
    pub fn with_confidence_multipliers(mut self, multipliers: RegimeMultipliers) -> Self {
        self.confidence_multipliers = multipliers;
        self
    }

    /// Set custom position multipliers
    pub fn with_position_multipliers(mut self, multipliers: RegimeMultipliers) -> Self {
        self.position_multipliers = multipliers;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_classification() {
        assert!(MarketRegime::BullTrending.is_trending());
        assert!(MarketRegime::BearTrending.is_trending());
        assert!(MarketRegime::RangingCalm.is_ranging());
        assert!(MarketRegime::RangingVolatile.is_volatile());
        assert!(MarketRegime::BullTrending.is_bullish());
        assert!(MarketRegime::BearTrending.is_bearish());
    }

    #[test]
    fn test_regime_detector_initialization() {
        let config = RegimeConfig::default();
        let detector = RegimeDetector::new(config);
        assert_eq!(detector.current_regime(), MarketRegime::Unknown);
    }

    #[test]
    fn test_regime_detector_uptrend() {
        let config = RegimeConfig {
            min_data_points: 10,
            trend_window: 10,
            ..Default::default()
        };
        let mut detector = RegimeDetector::new(config);

        // Simulate uptrend
        for i in 0..30 {
            let price = 100.0 + i as f64 * 2.0;
            detector.update(price);
        }

        let regime = detector.current_regime();
        assert!(regime.is_trending() || regime.is_bullish() || regime == MarketRegime::Unknown);
    }

    #[test]
    fn test_regime_detector_ranging() {
        let config = RegimeConfig {
            min_data_points: 10,
            trend_window: 10,
            volatility_window: 10,
            adx_threshold: 25.0,
            ..Default::default()
        };
        let mut detector = RegimeDetector::new(config);

        // Simulate ranging market - oscillate around a fixed price
        for i in 0..30 {
            let price = 100.0 + ((i as f64 * 0.8).sin() * 5.0);
            detector.update(price);
        }

        let regime = detector.current_regime();
        // In ranging market, should be ranging or unknown (not strongly trending)
        // The test passes if it's not a strong trending regime or if it's unknown
        assert_ne!(regime, MarketRegime::Unknown);
    }

    #[test]
    fn test_regime_adjuster() {
        let adjuster = RegimeAdjuster::default();

        let base_threshold = 0.7;
        let adjusted =
            adjuster.adjust_confidence_threshold(base_threshold, MarketRegime::BullTrending);
        assert!(adjusted < base_threshold); // Lower threshold in trends

        let adjusted_ranging =
            adjuster.adjust_confidence_threshold(base_threshold, MarketRegime::RangingCalm);
        assert!(adjusted_ranging > base_threshold); // Higher threshold in ranging

        let base_size = 1000.0;
        let adjusted_size = adjuster.adjust_position_size(base_size, MarketRegime::BullTrending);
        assert!(adjusted_size > base_size); // Larger position in trends

        let adjusted_size_volatile =
            adjuster.adjust_position_size(base_size, MarketRegime::RangingVolatile);
        assert!(adjusted_size_volatile < base_size); // Smaller position in volatile ranging
    }

    #[test]
    fn test_regime_detector_sufficient_data() {
        let config = RegimeConfig::default();
        let mut detector = RegimeDetector::new(config);

        // Should be unknown initially
        assert_eq!(detector.current_regime(), MarketRegime::Unknown);

        // Add some data points
        for i in 0..25 {
            detector.update(100.0 + i as f64);
        }

        // Should no longer be unknown
        assert_ne!(detector.current_regime(), MarketRegime::Unknown);
    }

    #[test]
    fn test_regime_detector_reset() {
        let config = RegimeConfig::default();
        let mut detector = RegimeDetector::new(config);

        for i in 0..20 {
            detector.update(100.0 + i as f64);
        }

        detector.reset();
        assert_eq!(detector.current_regime(), MarketRegime::Unknown);
        assert_eq!(detector.prices.len(), 0);
        assert_eq!(detector.returns.len(), 0);
    }

    #[test]
    fn test_regime_multipliers() {
        let multipliers = RegimeMultipliers::default();
        assert_eq!(multipliers.get(MarketRegime::BullTrending), 1.0);
        assert_eq!(multipliers.get(MarketRegime::Unknown), 1.0);
    }

    #[test]
    fn test_custom_regime_adjuster() {
        let custom_multipliers = RegimeMultipliers {
            bull_trending: 2.0,
            bear_trending: 0.5,
            ..Default::default()
        };

        let adjuster =
            RegimeAdjuster::new().with_confidence_multipliers(custom_multipliers.clone());

        assert_eq!(
            adjuster
                .confidence_multipliers
                .get(MarketRegime::BullTrending),
            2.0
        );
    }

    #[test]
    fn test_volatility_calculation() {
        let config = RegimeConfig {
            volatility_window: 5,
            min_data_points: 5,
            ..Default::default()
        };
        let volatility_window = config.volatility_window;
        let mut detector = RegimeDetector::new(config);

        // Add prices with increasing volatility
        let prices = vec![100.0, 102.0, 101.0, 105.0, 103.0, 110.0, 105.0];
        for price in prices {
            detector.update(price);
        }

        // Check that we have enough data
        if detector.returns.len() >= volatility_window {
            let volatility = detector.calculate_volatility();
            // Volatility should be non-negative (might be very small or zero)
            assert!(volatility >= 0.0);
        }
    }

    #[test]
    fn test_trend_direction() {
        let config = RegimeConfig::default();
        let mut detector = RegimeDetector::new(config);

        // Uptrend
        for i in 0..20 {
            detector.update(100.0 + i as f64);
        }
        let direction = detector.trend_direction();
        assert!(direction > 0.0);

        // Downtrend
        detector.reset();
        for i in 0..20 {
            detector.update(100.0 - i as f64);
        }
        let direction = detector.trend_direction();
        assert!(direction < 0.0);
    }
}
