//! Volume-Synchronized Probability of Informed Trading (VPIN)
//!
//! VPIN is a metric that estimates the probability of informed trading
//! in a market. It was developed by Easley, López de Prado, and O'Hara
//! and is used to detect toxic order flow that may precede flash crashes.
//!
//! ## Algorithm
//!
//! 1. Aggregate trades into volume buckets of fixed size
//! 2. Classify trades as buy or sell using bulk volume classification (BVC)
//! 3. Calculate order imbalance for each bucket
//! 4. Compute VPIN as average absolute imbalance over rolling window
//!
//! ## Key Features
//!
//! - Volume-time synchronization (buckets by volume, not time)
//! - Bulk volume classification (no tick rule needed)
//! - Rolling window calculation for smooth updates
//! - Configurable bucket size and window length

use crate::common::{Error, Result};
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Configuration for VPIN calculation
#[derive(Debug, Clone)]
pub struct VpinConfig {
    /// Size of each volume bucket (in base units)
    pub bucket_volume: f64,
    /// Number of buckets in rolling window
    pub window_buckets: usize,
    /// Standard deviation for BVC probability calculation
    pub sigma: f64,
    /// High VPIN threshold (warning level)
    pub high_threshold: f64,
    /// Critical VPIN threshold (danger level)
    pub critical_threshold: f64,
    /// Minimum price change for trade direction inference
    pub min_price_change: f64,
}

impl Default for VpinConfig {
    fn default() -> Self {
        Self {
            bucket_volume: 10000.0,  // 10k units per bucket
            window_buckets: 50,      // 50 bucket rolling window
            sigma: 0.01,             // 1% price standard deviation
            high_threshold: 0.5,     // 50% informed trading = warning
            critical_threshold: 0.7, // 70% informed trading = critical
            min_price_change: 0.0001, // Minimum tick for direction
        }
    }
}

/// A single trade for VPIN calculation
#[derive(Debug, Clone)]
pub struct Trade {
    /// Trade price
    pub price: f64,
    /// Trade volume
    pub volume: f64,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
}

/// A completed volume bucket
#[derive(Debug, Clone)]
pub struct VolumeBucket {
    /// Start timestamp of bucket
    pub start_time: i64,
    /// End timestamp of bucket
    pub end_time: i64,
    /// Total volume in bucket
    pub volume: f64,
    /// Buy volume (classified)
    pub buy_volume: f64,
    /// Sell volume (classified)
    pub sell_volume: f64,
    /// Volume-weighted average price
    pub vwap: f64,
    /// Number of trades in bucket
    pub trade_count: usize,
    /// Order imbalance (|buy - sell| / total)
    pub imbalance: f64,
}

impl VolumeBucket {
    fn new(start_time: i64) -> Self {
        Self {
            start_time,
            end_time: start_time,
            volume: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
            vwap: 0.0,
            trade_count: 0,
            imbalance: 0.0,
        }
    }

    fn calculate_imbalance(&mut self) {
        if self.volume > 0.0 {
            self.imbalance = (self.buy_volume - self.sell_volume).abs() / self.volume;
        }
    }
}

/// VPIN severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VpinSeverity {
    /// Normal market conditions
    Normal,
    /// Elevated informed trading
    Elevated,
    /// High informed trading (warning)
    High,
    /// Critical - potential flash crash conditions
    Critical,
}

impl VpinSeverity {
    pub fn from_vpin(vpin: f64, high: f64, critical: f64) -> Self {
        if vpin >= critical {
            Self::Critical
        } else if vpin >= high {
            Self::High
        } else if vpin >= high * 0.6 {
            Self::Elevated
        } else {
            Self::Normal
        }
    }
}

/// Result of VPIN calculation
#[derive(Debug, Clone)]
pub struct VpinResult {
    /// Current VPIN value (0.0 - 1.0)
    pub vpin: f64,
    /// Severity assessment
    pub severity: VpinSeverity,
    /// Number of complete buckets
    pub bucket_count: usize,
    /// Whether calculation is reliable (enough buckets)
    pub is_reliable: bool,
    /// Average order imbalance
    pub avg_imbalance: f64,
    /// Standard deviation of imbalance
    pub imbalance_std: f64,
    /// Buy volume ratio over window
    pub buy_ratio: f64,
    /// Calculation timestamp
    pub timestamp: i64,
}

/// Volume-Synchronized Probability of Informed Trading Calculator
///
/// This implements the VPIN metric for detecting toxic order flow
/// and potential flash crash conditions.
pub struct VpinCalculator {
    /// Configuration parameters
    config: VpinConfig,
    /// Completed volume buckets
    buckets: VecDeque<VolumeBucket>,
    /// Current bucket being filled
    current_bucket: VolumeBucket,
    /// Accumulated volume in current bucket
    accumulated_volume: f64,
    /// Last trade price for direction inference
    last_price: Option<f64>,
    /// Price accumulator for VWAP
    price_volume_sum: f64,
    /// Total trades processed
    total_trades: u64,
    /// Last VPIN calculation result
    last_vpin: f64,
    /// Exponential moving average of VPIN
    ema_vpin: f64,
    /// EMA decay factor
    ema_alpha: f64,
}

impl Default for VpinCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl VpinCalculator {
    /// Create a new VPIN calculator with default configuration
    pub fn new() -> Self {
        Self::with_config(VpinConfig::default())
    }

    /// Create a new VPIN calculator with custom configuration
    pub fn with_config(config: VpinConfig) -> Self {
        let ema_alpha = 2.0 / (config.window_buckets as f64 + 1.0);
        Self {
            buckets: VecDeque::with_capacity(config.window_buckets + 1),
            current_bucket: VolumeBucket::new(0),
            accumulated_volume: 0.0,
            last_price: None,
            price_volume_sum: 0.0,
            total_trades: 0,
            last_vpin: 0.0,
            ema_vpin: 0.0,
            ema_alpha,
            config,
        }
    }

    /// Process a single trade
    pub fn process_trade(&mut self, trade: &Trade) -> Result<Option<VpinResult>> {
        if trade.volume <= 0.0 {
            return Err(Error::InvalidInput("Trade volume must be positive".into()));
        }
        if trade.price <= 0.0 {
            return Err(Error::InvalidInput("Trade price must be positive".into()));
        }

        self.total_trades += 1;

        // Initialize current bucket if needed
        if self.current_bucket.trade_count == 0 {
            self.current_bucket = VolumeBucket::new(trade.timestamp);
        }

        // Classify trade using bulk volume classification
        let (buy_volume, sell_volume) = self.classify_trade(trade);

        // Add to current bucket
        self.add_to_bucket(trade, buy_volume, sell_volume);

        // Update last price
        self.last_price = Some(trade.price);

        // Check if bucket is complete
        let mut result = None;
        if self.accumulated_volume >= self.config.bucket_volume {
            self.complete_bucket(trade.timestamp);
            result = Some(self.calculate_vpin(trade.timestamp));
        }

        Ok(result)
    }

    /// Process multiple trades in batch
    pub fn process_trades(&mut self, trades: &[Trade]) -> Result<Vec<VpinResult>> {
        let mut results = Vec::new();
        for trade in trades {
            if let Some(result) = self.process_trade(trade)? {
                results.push(result);
            }
        }
        Ok(results)
    }

    /// Get current VPIN without processing new trades
    pub fn current_vpin(&self) -> VpinResult {
        self.calculate_vpin(chrono::Utc::now().timestamp_millis())
    }

    /// Get the last calculated VPIN value
    pub fn last_vpin_value(&self) -> f64 {
        self.last_vpin
    }

    /// Get EMA smoothed VPIN
    pub fn ema_vpin(&self) -> f64 {
        self.ema_vpin
    }

    /// Get number of completed buckets
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Check if we have enough data for reliable calculation
    pub fn is_reliable(&self) -> bool {
        self.buckets.len() >= self.config.window_buckets / 2
    }

    /// Get current severity level
    pub fn severity(&self) -> VpinSeverity {
        VpinSeverity::from_vpin(
            self.last_vpin,
            self.config.high_threshold,
            self.config.critical_threshold,
        )
    }

    /// Reset the calculator state
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.current_bucket = VolumeBucket::new(0);
        self.accumulated_volume = 0.0;
        self.last_price = None;
        self.price_volume_sum = 0.0;
        self.total_trades = 0;
        self.last_vpin = 0.0;
        self.ema_vpin = 0.0;
    }

    /// Get statistics about the calculator state
    pub fn stats(&self) -> VpinStats {
        VpinStats {
            total_trades: self.total_trades,
            completed_buckets: self.buckets.len(),
            current_bucket_fill: self.accumulated_volume / self.config.bucket_volume,
            last_vpin: self.last_vpin,
            ema_vpin: self.ema_vpin,
            is_reliable: self.is_reliable(),
            severity: self.severity(),
        }
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        // This is a stateful processor, actual work done in process_trade
        Ok(())
    }

    // --- Private methods ---

    /// Classify trade volume as buy or sell using Bulk Volume Classification
    fn classify_trade(&self, trade: &Trade) -> (f64, f64) {
        // Use price change for direction probability
        let price_change = match self.last_price {
            Some(last) => trade.price - last,
            None => 0.0,
        };

        // Calculate probability of buy using cumulative normal distribution
        // This is the BVC approach from the original VPIN paper
        let z = price_change / self.config.sigma;
        let buy_prob = self.standard_normal_cdf(z);

        // Distribute volume based on probability
        let buy_volume = trade.volume * buy_prob;
        let sell_volume = trade.volume * (1.0 - buy_prob);

        (buy_volume, sell_volume)
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(&self, x: f64) -> f64 {
        // Using error function approximation
        0.5 * (1.0 + self.erf(x / (2.0_f64).sqrt()))
    }

    /// Error function approximation (Abramowitz and Stegun)
    fn erf(&self, x: f64) -> f64 {
        // Constants for approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Add trade to current bucket
    fn add_to_bucket(&mut self, trade: &Trade, buy_volume: f64, sell_volume: f64) {
        self.current_bucket.volume += trade.volume;
        self.current_bucket.buy_volume += buy_volume;
        self.current_bucket.sell_volume += sell_volume;
        self.current_bucket.trade_count += 1;
        self.current_bucket.end_time = trade.timestamp;

        self.price_volume_sum += trade.price * trade.volume;
        self.accumulated_volume += trade.volume;
    }

    /// Complete current bucket and start a new one
    fn complete_bucket(&mut self, timestamp: i64) {
        // Calculate VWAP for the bucket
        if self.current_bucket.volume > 0.0 {
            self.current_bucket.vwap = self.price_volume_sum / self.current_bucket.volume;
        }

        // Handle volume overflow (trade that pushes over bucket boundary)
        let overflow = self.accumulated_volume - self.config.bucket_volume;

        // Adjust bucket volume if there's overflow
        if overflow > 0.0 && self.current_bucket.volume > 0.0 {
            let ratio = self.config.bucket_volume / self.current_bucket.volume;
            self.current_bucket.volume = self.config.bucket_volume;
            self.current_bucket.buy_volume *= ratio;
            self.current_bucket.sell_volume *= ratio;
        }

        // Calculate order imbalance
        self.current_bucket.calculate_imbalance();

        // Add to completed buckets
        self.buckets.push_back(self.current_bucket.clone());

        // Maintain window size
        while self.buckets.len() > self.config.window_buckets {
            self.buckets.pop_front();
        }

        // Start new bucket
        self.current_bucket = VolumeBucket::new(timestamp);
        self.accumulated_volume = overflow.max(0.0);
        self.price_volume_sum = 0.0;
    }

    /// Calculate VPIN from completed buckets
    fn calculate_vpin(&mut self, timestamp: i64) -> VpinResult {
        if self.buckets.is_empty() {
            return VpinResult {
                vpin: 0.0,
                severity: VpinSeverity::Normal,
                bucket_count: 0,
                is_reliable: false,
                avg_imbalance: 0.0,
                imbalance_std: 0.0,
                buy_ratio: 0.5,
                timestamp,
            };
        }

        // Calculate VPIN as average absolute order imbalance
        let n = self.buckets.len() as f64;
        let mut sum_imbalance = 0.0;
        let mut sum_imbalance_sq = 0.0;
        let mut total_buy = 0.0;
        let mut total_volume = 0.0;

        for bucket in &self.buckets {
            sum_imbalance += bucket.imbalance;
            sum_imbalance_sq += bucket.imbalance * bucket.imbalance;
            total_buy += bucket.buy_volume;
            total_volume += bucket.volume;
        }

        let avg_imbalance = sum_imbalance / n;
        let variance = (sum_imbalance_sq / n) - (avg_imbalance * avg_imbalance);
        let std_dev = variance.max(0.0).sqrt();

        // VPIN is the average order imbalance
        let vpin = avg_imbalance;

        // Calculate buy ratio
        let buy_ratio = if total_volume > 0.0 {
            total_buy / total_volume
        } else {
            0.5
        };

        // Update EMA
        self.ema_vpin = self.ema_alpha * vpin + (1.0 - self.ema_alpha) * self.ema_vpin;
        self.last_vpin = vpin;

        let severity = VpinSeverity::from_vpin(
            vpin,
            self.config.high_threshold,
            self.config.critical_threshold,
        );

        VpinResult {
            vpin,
            severity,
            bucket_count: self.buckets.len(),
            is_reliable: self.is_reliable(),
            avg_imbalance,
            imbalance_std: std_dev,
            buy_ratio,
            timestamp,
        }
    }
}

/// Statistics about the VPIN calculator state
#[derive(Debug, Clone)]
pub struct VpinStats {
    /// Total trades processed
    pub total_trades: u64,
    /// Number of completed volume buckets
    pub completed_buckets: usize,
    /// Fill ratio of current bucket (0.0 - 1.0)
    pub current_bucket_fill: f64,
    /// Last calculated VPIN
    pub last_vpin: f64,
    /// Exponential moving average of VPIN
    pub ema_vpin: f64,
    /// Whether enough data for reliable calculation
    pub is_reliable: bool,
    /// Current severity level
    pub severity: VpinSeverity,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_trade(price: f64, volume: f64, timestamp: i64) -> Trade {
        Trade {
            price,
            volume,
            timestamp,
        }
    }

    #[test]
    fn test_new_calculator() {
        let calc = VpinCalculator::new();
        assert_eq!(calc.bucket_count(), 0);
        assert!(!calc.is_reliable());
        assert_eq!(calc.last_vpin_value(), 0.0);
    }

    #[test]
    fn test_with_config() {
        let config = VpinConfig {
            bucket_volume: 5000.0,
            window_buckets: 20,
            ..Default::default()
        };
        let calc = VpinCalculator::with_config(config);
        assert_eq!(calc.config.bucket_volume, 5000.0);
        assert_eq!(calc.config.window_buckets, 20);
    }

    #[test]
    fn test_single_trade() {
        let mut calc = VpinCalculator::new();
        let trade = create_trade(100.0, 100.0, 1000);
        let result = calc.process_trade(&trade).unwrap();

        // Single trade shouldn't complete a bucket with default config
        assert!(result.is_none());
        assert_eq!(calc.total_trades, 1);
    }

    #[test]
    fn test_bucket_completion() {
        let mut calc = VpinCalculator::with_config(VpinConfig {
            bucket_volume: 100.0,
            window_buckets: 5,
            ..Default::default()
        });

        // Fill first bucket
        let trade1 = create_trade(100.0, 50.0, 1000);
        assert!(calc.process_trade(&trade1).unwrap().is_none());

        let trade2 = create_trade(100.5, 60.0, 1001);
        let result = calc.process_trade(&trade2).unwrap();

        // Should complete bucket
        assert!(result.is_some());
        assert_eq!(calc.bucket_count(), 1);
    }

    #[test]
    fn test_vpin_calculation() {
        let mut calc = VpinCalculator::with_config(VpinConfig {
            bucket_volume: 100.0,
            window_buckets: 3,
            sigma: 0.01,
            ..Default::default()
        });

        // Create trades with consistent buy pressure (prices going up)
        for i in 0..10 {
            let price = 100.0 + (i as f64 * 0.1);
            let trade = create_trade(price, 50.0, 1000 + i);
            let _ = calc.process_trade(&trade);
        }

        let result = calc.current_vpin();
        assert!(result.bucket_count > 0);
        // With consistent buy pressure, VPIN should be elevated
        assert!(result.vpin >= 0.0 && result.vpin <= 1.0);
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(
            VpinSeverity::from_vpin(0.1, 0.5, 0.7),
            VpinSeverity::Normal
        );
        assert_eq!(
            VpinSeverity::from_vpin(0.35, 0.5, 0.7),
            VpinSeverity::Elevated
        );
        assert_eq!(
            VpinSeverity::from_vpin(0.6, 0.5, 0.7),
            VpinSeverity::High
        );
        assert_eq!(
            VpinSeverity::from_vpin(0.8, 0.5, 0.7),
            VpinSeverity::Critical
        );
    }

    #[test]
    fn test_invalid_trade() {
        let mut calc = VpinCalculator::new();

        // Zero volume should fail
        let trade = create_trade(100.0, 0.0, 1000);
        assert!(calc.process_trade(&trade).is_err());

        // Negative price should fail
        let trade = create_trade(-100.0, 100.0, 1000);
        assert!(calc.process_trade(&trade).is_err());
    }

    #[test]
    fn test_reset() {
        let mut calc = VpinCalculator::with_config(VpinConfig {
            bucket_volume: 100.0,
            window_buckets: 5,
            ..Default::default()
        });

        // Add some trades
        for i in 0..5 {
            let trade = create_trade(100.0, 50.0, 1000 + i);
            let _ = calc.process_trade(&trade);
        }

        assert!(calc.bucket_count() > 0);

        calc.reset();

        assert_eq!(calc.bucket_count(), 0);
        assert_eq!(calc.total_trades, 0);
        assert_eq!(calc.last_vpin_value(), 0.0);
    }

    #[test]
    fn test_stats() {
        let mut calc = VpinCalculator::with_config(VpinConfig {
            bucket_volume: 100.0,
            window_buckets: 5,
            ..Default::default()
        });

        // Add trades to partially fill bucket
        let trade = create_trade(100.0, 50.0, 1000);
        let _ = calc.process_trade(&trade);

        let stats = calc.stats();
        assert_eq!(stats.total_trades, 1);
        assert_eq!(stats.completed_buckets, 0);
        assert!((stats.current_bucket_fill - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_batch_processing() {
        let mut calc = VpinCalculator::with_config(VpinConfig {
            bucket_volume: 100.0,
            window_buckets: 5,
            ..Default::default()
        });

        let trades: Vec<Trade> = (0..20)
            .map(|i| create_trade(100.0 + (i as f64 * 0.01), 50.0, 1000 + i))
            .collect();

        let results = calc.process_trades(&trades).unwrap();

        // Should have completed several buckets
        assert!(!results.is_empty());
        assert!(calc.bucket_count() > 0);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut calc = VpinCalculator::with_config(VpinConfig {
            bucket_volume: 100.0,
            window_buckets: 5,
            ..Default::default()
        });

        // Process enough trades to get EMA working
        for i in 0..30 {
            let price = 100.0 + (i as f64 * 0.1);
            let trade = create_trade(price, 50.0, 1000 + i);
            let _ = calc.process_trade(&trade);
        }

        let ema = calc.ema_vpin();
        let instant = calc.last_vpin_value();

        // EMA should be smoothed (different from instant)
        // After enough updates, EMA should be close but not necessarily equal
        assert!(ema >= 0.0 && ema <= 1.0);
        assert!(instant >= 0.0 && instant <= 1.0);
    }

    #[test]
    fn test_buy_sell_classification() {
        let mut calc = VpinCalculator::with_config(VpinConfig {
            bucket_volume: 1000.0,
            window_buckets: 5,
            sigma: 0.01,
            ..Default::default()
        });

        // Strong upward price movement should classify as more buys
        let trade1 = create_trade(100.0, 100.0, 1000);
        let _ = calc.process_trade(&trade1);

        let trade2 = create_trade(102.0, 100.0, 1001); // 2% up
        let _ = calc.process_trade(&trade2);

        // The current bucket should have more buy volume than sell volume
        assert!(calc.current_bucket.buy_volume > calc.current_bucket.sell_volume);
    }

    #[test]
    fn test_process_compatibility() {
        let calc = VpinCalculator::new();
        assert!(calc.process().is_ok());
    }
}
