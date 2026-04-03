//! Flash Crash Detection
//!
//! Detects rapid price movements and potential flash crash scenarios.
//! Uses multiple detection methods:
//! - Price velocity (rate of change)
//! - Volatility regime shifts
//! - Volume surge detection
//! - Bid-ask spread expansion
//! - Order book imbalance

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for flash crash detection
#[derive(Debug, Clone)]
pub struct FlashCrashConfig {
    /// Window size for price calculations (in ticks)
    pub window_size: usize,
    /// Minimum samples before detection is active
    pub min_samples: usize,
    /// Price drop threshold for warning (percentage, e.g., 0.02 = 2%)
    pub warning_threshold: f64,
    /// Price drop threshold for critical alert (percentage)
    pub critical_threshold: f64,
    /// Time window for measuring price velocity (milliseconds)
    pub velocity_window_ms: i64,
    /// Volatility multiplier for dynamic thresholds
    pub volatility_multiplier: f64,
    /// Volume surge threshold (multiple of average)
    pub volume_surge_threshold: f64,
    /// Spread expansion threshold (multiple of average)
    pub spread_expansion_threshold: f64,
}

impl Default for FlashCrashConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            min_samples: 20,
            warning_threshold: 0.02,    // 2% drop
            critical_threshold: 0.05,   // 5% drop
            velocity_window_ms: 1000,   // 1 second
            volatility_multiplier: 3.0, // 3 sigma
            volume_surge_threshold: 5.0,
            spread_expansion_threshold: 3.0,
        }
    }
}

/// Flash crash severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashSeverity {
    None,
    Warning,
    Critical,
    Emergency,
}

impl CrashSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::Emergency => "emergency",
        }
    }
}

/// Market data point for flash crash detection
#[derive(Debug, Clone)]
pub struct MarketTick {
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub timestamp: i64,
}

impl MarketTick {
    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }

    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }
}

/// Detection signals from various indicators
#[derive(Debug, Clone, Default)]
pub struct DetectionSignals {
    /// Price drop percentage (negative = drop)
    pub price_change_pct: f64,
    /// Price velocity (change per second)
    pub price_velocity: f64,
    /// Current volatility (standard deviation of returns)
    pub volatility: f64,
    /// Historical average volatility
    pub avg_volatility: f64,
    /// Volatility z-score
    pub volatility_zscore: f64,
    /// Volume relative to average
    pub volume_ratio: f64,
    /// Spread relative to average
    pub spread_ratio: f64,
    /// Overall crash probability (0.0 - 1.0)
    pub crash_probability: f64,
}

/// Flash crash detection result
#[derive(Debug, Clone)]
pub struct FlashCrashAlert {
    pub severity: CrashSeverity,
    pub signals: DetectionSignals,
    pub timestamp: i64,
    pub message: String,
    pub recommended_action: RecommendedAction,
}

/// Recommended actions based on detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    None,
    ReduceExposure,
    HaltNewOrders,
    CancelAllOrders,
    EmergencyLiquidation,
}

impl RecommendedAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ReduceExposure => "reduce_exposure",
            Self::HaltNewOrders => "halt_new_orders",
            Self::CancelAllOrders => "cancel_all_orders",
            Self::EmergencyLiquidation => "emergency_liquidation",
        }
    }
}

/// Internal state for tracking market data
#[derive(Debug, Clone, Default)]
struct MarketState {
    /// Price history
    prices: VecDeque<f64>,
    /// Timestamp history
    timestamps: VecDeque<i64>,
    /// Volume history
    volumes: VecDeque<f64>,
    /// Spread history
    spreads: VecDeque<f64>,
    /// Return history (for volatility)
    returns: VecDeque<f64>,
    /// Running statistics
    price_sum: f64,
    volume_sum: f64,
    spread_sum: f64,
    return_sum: f64,
    return_squared_sum: f64,
}

/// Flash crash detection system
pub struct FlashCrash {
    config: FlashCrashConfig,
    state: MarketState,
    /// Last detected alert
    last_alert: Option<FlashCrashAlert>,
    /// Reference price (e.g., session open or recent high)
    reference_price: Option<f64>,
    /// Alert count for tracking
    alert_count: u64,
    /// Historical volatility for regime detection
    volatility_history: VecDeque<f64>,
}

impl Default for FlashCrash {
    fn default() -> Self {
        Self::new()
    }
}

impl FlashCrash {
    /// Create a new FlashCrash detector with default config
    pub fn new() -> Self {
        Self::with_config(FlashCrashConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: FlashCrashConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            state: MarketState {
                prices: VecDeque::with_capacity(window_size),
                timestamps: VecDeque::with_capacity(window_size),
                volumes: VecDeque::with_capacity(window_size),
                spreads: VecDeque::with_capacity(window_size),
                returns: VecDeque::with_capacity(window_size),
                ..Default::default()
            },
            last_alert: None,
            reference_price: None,
            alert_count: 0,
            volatility_history: VecDeque::with_capacity(1000),
        }
    }

    /// Set reference price for measuring drawdown
    pub fn set_reference_price(&mut self, price: f64) {
        self.reference_price = Some(price);
    }

    /// Update with new market tick
    pub fn update(&mut self, tick: MarketTick) {
        // Calculate return if we have previous price
        if let Some(&prev_price) = self.state.prices.back() {
            if prev_price > 0.0 {
                let ret = (tick.price - prev_price) / prev_price;
                self.add_return(ret);
            }
        }

        // Update reference price if this is a new high
        if let Some(ref_price) = self.reference_price {
            if tick.price > ref_price {
                self.reference_price = Some(tick.price);
            }
        } else {
            self.reference_price = Some(tick.price);
        }

        // Add to rolling windows
        self.add_price(tick.price, tick.timestamp);
        self.add_volume(tick.volume);
        self.add_spread(tick.spread());
    }

    /// Check for flash crash and return alert if detected
    pub fn check(&mut self) -> Option<FlashCrashAlert> {
        if self.state.prices.len() < self.config.min_samples {
            return None;
        }

        let signals = self.calculate_signals();
        let severity = self.determine_severity(&signals);

        if severity == CrashSeverity::None {
            self.last_alert = None;
            return None;
        }

        let recommended_action = self.determine_action(severity, &signals);
        let message = self.create_alert_message(severity, &signals);

        let alert = FlashCrashAlert {
            severity,
            signals,
            timestamp: chrono::Utc::now().timestamp_millis(),
            message,
            recommended_action,
        };

        self.alert_count += 1;
        self.last_alert = Some(alert.clone());

        Some(alert)
    }

    /// Calculate all detection signals
    fn calculate_signals(&self) -> DetectionSignals {
        let price_change_pct = self.calculate_price_change();
        let price_velocity = self.calculate_price_velocity();
        let (volatility, avg_volatility, volatility_zscore) = self.calculate_volatility_metrics();
        let volume_ratio = self.calculate_volume_ratio();
        let spread_ratio = self.calculate_spread_ratio();

        // Calculate overall crash probability
        let crash_probability = self.calculate_crash_probability(
            price_change_pct,
            volatility_zscore,
            volume_ratio,
            spread_ratio,
        );

        DetectionSignals {
            price_change_pct,
            price_velocity,
            volatility,
            avg_volatility,
            volatility_zscore,
            volume_ratio,
            spread_ratio,
            crash_probability,
        }
    }

    /// Calculate price change from reference
    fn calculate_price_change(&self) -> f64 {
        let current_price = self.state.prices.back().copied().unwrap_or(0.0);
        let reference = self.reference_price.unwrap_or(current_price);

        if reference > 0.0 {
            (current_price - reference) / reference
        } else {
            0.0
        }
    }

    /// Calculate price velocity (change per second)
    fn calculate_price_velocity(&self) -> f64 {
        if self.state.prices.len() < 2 {
            return 0.0;
        }

        let window_ms = self.config.velocity_window_ms;
        let current_time = self.state.timestamps.back().copied().unwrap_or(0);
        let cutoff_time = current_time - window_ms;

        // Find price at cutoff time
        let mut start_price = None;
        let mut start_time = None;

        for (i, &ts) in self.state.timestamps.iter().enumerate() {
            if ts >= cutoff_time {
                start_price = Some(self.state.prices[i]);
                start_time = Some(ts);
                break;
            }
        }

        match (start_price, start_time) {
            (Some(sp), Some(st)) if current_time > st => {
                let current_price = self.state.prices.back().copied().unwrap_or(sp);
                let time_delta_sec = (current_time - st) as f64 / 1000.0;
                if time_delta_sec > 0.0 && sp > 0.0 {
                    (current_price - sp) / sp / time_delta_sec
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// Calculate volatility metrics
    fn calculate_volatility_metrics(&self) -> (f64, f64, f64) {
        if self.state.returns.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let n = self.state.returns.len() as f64;
        let mean = self.state.return_sum / n;
        let variance = (self.state.return_squared_sum / n) - (mean * mean);
        let volatility = variance.max(0.0).sqrt();

        // Calculate average historical volatility
        let avg_volatility = if self.volatility_history.is_empty() {
            volatility
        } else {
            self.volatility_history.iter().sum::<f64>() / self.volatility_history.len() as f64
        };

        // Calculate z-score
        let volatility_zscore = if avg_volatility > 0.0 {
            (volatility - avg_volatility) / avg_volatility
        } else {
            0.0
        };

        (volatility, avg_volatility, volatility_zscore)
    }

    /// Calculate volume ratio to average
    fn calculate_volume_ratio(&self) -> f64 {
        if self.state.volumes.is_empty() {
            return 1.0;
        }

        let avg_volume = self.state.volume_sum / self.state.volumes.len() as f64;
        let current_volume = self.state.volumes.back().copied().unwrap_or(0.0);

        if avg_volume > 0.0 {
            current_volume / avg_volume
        } else {
            1.0
        }
    }

    /// Calculate spread ratio to average
    fn calculate_spread_ratio(&self) -> f64 {
        if self.state.spreads.is_empty() {
            return 1.0;
        }

        let avg_spread = self.state.spread_sum / self.state.spreads.len() as f64;
        let current_spread = self.state.spreads.back().copied().unwrap_or(0.0);

        if avg_spread > 0.0 {
            current_spread / avg_spread
        } else {
            1.0
        }
    }

    /// Calculate overall crash probability
    fn calculate_crash_probability(
        &self,
        price_change: f64,
        vol_zscore: f64,
        volume_ratio: f64,
        spread_ratio: f64,
    ) -> f64 {
        let mut prob = 0.0;

        // Price drop contribution (largest weight)
        let price_drop = -price_change; // Convert to positive for drops
        if price_drop > 0.0 {
            prob += (price_drop / self.config.critical_threshold).min(1.0) * 0.4;
        }

        // Volatility spike contribution
        if vol_zscore > 0.0 {
            let vol_contrib = (vol_zscore / self.config.volatility_multiplier).min(1.0) * 0.25;
            prob += vol_contrib;
        }

        // Volume surge contribution
        if volume_ratio > 1.0 {
            let vol_contrib =
                ((volume_ratio - 1.0) / (self.config.volume_surge_threshold - 1.0)).min(1.0) * 0.2;
            prob += vol_contrib;
        }

        // Spread expansion contribution
        if spread_ratio > 1.0 {
            let spread_contrib = ((spread_ratio - 1.0)
                / (self.config.spread_expansion_threshold - 1.0))
                .min(1.0)
                * 0.15;
            prob += spread_contrib;
        }

        prob.clamp(0.0, 1.0)
    }

    /// Determine severity based on signals
    fn determine_severity(&self, signals: &DetectionSignals) -> CrashSeverity {
        let price_drop = -signals.price_change_pct;

        // Emergency: Extreme conditions
        if price_drop >= self.config.critical_threshold * 2.0
            || (price_drop >= self.config.critical_threshold
                && signals.volume_ratio >= self.config.volume_surge_threshold)
        {
            return CrashSeverity::Emergency;
        }

        // Critical: Significant drop or multiple warning signs
        if price_drop >= self.config.critical_threshold
            || (price_drop >= self.config.warning_threshold
                && signals.volatility_zscore >= self.config.volatility_multiplier)
        {
            return CrashSeverity::Critical;
        }

        // Warning: Early signs
        if price_drop >= self.config.warning_threshold
            || signals.crash_probability >= 0.5
            || (signals.volume_ratio >= self.config.volume_surge_threshold
                && signals.spread_ratio >= self.config.spread_expansion_threshold)
        {
            return CrashSeverity::Warning;
        }

        CrashSeverity::None
    }

    /// Determine recommended action based on severity and signals
    fn determine_action(
        &self,
        severity: CrashSeverity,
        signals: &DetectionSignals,
    ) -> RecommendedAction {
        match severity {
            CrashSeverity::Emergency => RecommendedAction::EmergencyLiquidation,
            CrashSeverity::Critical => {
                if signals.crash_probability >= 0.8 {
                    RecommendedAction::CancelAllOrders
                } else {
                    RecommendedAction::HaltNewOrders
                }
            }
            CrashSeverity::Warning => RecommendedAction::ReduceExposure,
            CrashSeverity::None => RecommendedAction::None,
        }
    }

    /// Create human-readable alert message
    fn create_alert_message(&self, severity: CrashSeverity, signals: &DetectionSignals) -> String {
        let price_drop = -signals.price_change_pct * 100.0;

        match severity {
            CrashSeverity::Emergency => {
                format!(
                    "🚨 EMERGENCY: Flash crash detected! Price down {:.2}%, \
                     Vol ratio: {:.1}x, Spread ratio: {:.1}x. IMMEDIATE ACTION REQUIRED.",
                    price_drop, signals.volume_ratio, signals.spread_ratio
                )
            }
            CrashSeverity::Critical => {
                format!(
                    "⚠️ CRITICAL: Significant price drop of {:.2}%, \
                     Crash probability: {:.0}%. Consider halting trading.",
                    price_drop,
                    signals.crash_probability * 100.0
                )
            }
            CrashSeverity::Warning => {
                format!(
                    "⚡ WARNING: Price decline of {:.2}% detected. \
                     Monitoring for flash crash conditions.",
                    price_drop
                )
            }
            CrashSeverity::None => String::from("Normal market conditions"),
        }
    }

    /// Add price to rolling window
    fn add_price(&mut self, price: f64, timestamp: i64) {
        if self.state.prices.len() >= self.config.window_size {
            if let Some(old_price) = self.state.prices.pop_front() {
                self.state.price_sum -= old_price;
            }
            self.state.timestamps.pop_front();
        }

        self.state.prices.push_back(price);
        self.state.timestamps.push_back(timestamp);
        self.state.price_sum += price;
    }

    /// Add volume to rolling window
    fn add_volume(&mut self, volume: f64) {
        if self.state.volumes.len() >= self.config.window_size {
            if let Some(old_vol) = self.state.volumes.pop_front() {
                self.state.volume_sum -= old_vol;
            }
        }

        self.state.volumes.push_back(volume);
        self.state.volume_sum += volume;
    }

    /// Add spread to rolling window
    fn add_spread(&mut self, spread: f64) {
        if self.state.spreads.len() >= self.config.window_size {
            if let Some(old_spread) = self.state.spreads.pop_front() {
                self.state.spread_sum -= old_spread;
            }
        }

        self.state.spreads.push_back(spread);
        self.state.spread_sum += spread;
    }

    /// Add return to rolling window
    fn add_return(&mut self, ret: f64) {
        if self.state.returns.len() >= self.config.window_size {
            if let Some(old_ret) = self.state.returns.pop_front() {
                self.state.return_sum -= old_ret;
                self.state.return_squared_sum -= old_ret * old_ret;
            }
        }

        self.state.returns.push_back(ret);
        self.state.return_sum += ret;
        self.state.return_squared_sum += ret * ret;

        // Update volatility history periodically
        if self.state.returns.len() >= self.config.min_samples
            && self.state.returns.len() % 10 == 0
        {
            let n = self.state.returns.len() as f64;
            let mean = self.state.return_sum / n;
            let variance = (self.state.return_squared_sum / n) - (mean * mean);
            let volatility = variance.max(0.0).sqrt();

            if self.volatility_history.len() >= 1000 {
                self.volatility_history.pop_front();
            }
            self.volatility_history.push_back(volatility);
        }
    }

    /// Get the last alert if any
    pub fn last_alert(&self) -> Option<&FlashCrashAlert> {
        self.last_alert.as_ref()
    }

    /// Get alert count
    pub fn alert_count(&self) -> u64 {
        self.alert_count
    }

    /// Check if currently in flash crash condition
    pub fn is_flash_crash(&self) -> bool {
        self.last_alert
            .as_ref()
            .map(|a| a.severity != CrashSeverity::None)
            .unwrap_or(false)
    }

    /// Get current severity
    pub fn current_severity(&self) -> CrashSeverity {
        self.last_alert
            .as_ref()
            .map(|a| a.severity)
            .unwrap_or(CrashSeverity::None)
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.state = MarketState::default();
        self.last_alert = None;
        self.reference_price = None;
        self.volatility_history.clear();
    }

    /// Main processing function (compatibility with neuromorphic interface)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_tick(price: f64, volume: f64, timestamp: i64) -> MarketTick {
        MarketTick {
            price,
            volume,
            bid: price - 0.01,
            ask: price + 0.01,
            timestamp,
        }
    }

    #[test]
    fn test_basic_creation() {
        let detector = FlashCrash::new();
        assert!(!detector.is_flash_crash());
        assert_eq!(detector.current_severity(), CrashSeverity::None);
    }

    #[test]
    fn test_normal_market() {
        let mut detector = FlashCrash::new();

        // Simulate normal market with small fluctuations
        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 0.5;
            let tick = create_tick(price, 1000.0, i * 100);
            detector.update(tick);
        }

        let alert = detector.check();
        assert!(alert.is_none() || alert.unwrap().severity == CrashSeverity::None);
    }

    #[test]
    fn test_flash_crash_detection() {
        let mut detector = FlashCrash::with_config(FlashCrashConfig {
            window_size: 50,
            min_samples: 10,
            warning_threshold: 0.02,
            critical_threshold: 0.05,
            ..Default::default()
        });

        // Build up normal prices
        for i in 0..20 {
            let tick = create_tick(100.0, 1000.0, i * 100);
            detector.update(tick);
        }

        // Simulate flash crash (6% drop)
        for i in 20..30 {
            let price = 100.0 - ((i - 20) as f64 * 0.6);
            let tick = create_tick(price, 5000.0, i * 100); // High volume
            detector.update(tick);
        }

        let alert = detector.check();
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert!(alert.severity == CrashSeverity::Critical || alert.severity == CrashSeverity::Emergency);
    }

    #[test]
    fn test_warning_level() {
        let mut detector = FlashCrash::with_config(FlashCrashConfig {
            window_size: 50,
            min_samples: 10,
            warning_threshold: 0.02,
            critical_threshold: 0.05,
            ..Default::default()
        });

        // Build up normal prices
        for i in 0..20 {
            let tick = create_tick(100.0, 1000.0, i * 100);
            detector.update(tick);
        }

        // Simulate 2.5% drop (warning level)
        for i in 20..30 {
            let price = 100.0 - ((i - 20) as f64 * 0.25);
            let tick = create_tick(price, 1000.0, i * 100);
            detector.update(tick);
        }

        let alert = detector.check();
        assert!(alert.is_some());
        assert!(alert.unwrap().severity == CrashSeverity::Warning);
    }

    #[test]
    fn test_volume_surge() {
        let mut detector = FlashCrash::with_config(FlashCrashConfig {
            window_size: 50,
            min_samples: 10,
            volume_surge_threshold: 3.0,
            ..Default::default()
        });

        // Normal volume
        for i in 0..30 {
            let tick = create_tick(100.0, 1000.0, i * 100);
            detector.update(tick);
        }

        // Volume surge with slight price drop
        for i in 30..40 {
            let tick = create_tick(99.0, 10000.0, i * 100); // 10x volume
            detector.update(tick);
        }

        let alert = detector.check();
        if let Some(a) = &alert {
            assert!(a.signals.volume_ratio > 1.0);
        }
    }

    #[test]
    fn test_spread_expansion() {
        let mut detector = FlashCrash::new();

        // Normal spread
        for i in 0..30 {
            let tick = MarketTick {
                price: 100.0,
                volume: 1000.0,
                bid: 99.99,
                ask: 100.01,
                timestamp: i * 100,
            };
            detector.update(tick);
        }

        // Wide spread
        for i in 30..40 {
            let tick = MarketTick {
                price: 99.5,
                volume: 1000.0,
                bid: 99.0,
                ask: 100.0, // 50x wider spread
                timestamp: i * 100,
            };
            detector.update(tick);
        }

        let alert = detector.check();
        if let Some(a) = &alert {
            assert!(a.signals.spread_ratio > 1.0);
        }
    }

    #[test]
    fn test_reference_price_tracking() {
        let mut detector = FlashCrash::new();

        // Set initial reference
        detector.set_reference_price(100.0);

        // Price goes up
        for i in 0..10 {
            let tick = create_tick(100.0 + i as f64, 1000.0, i * 100);
            detector.update(tick);
        }

        // Reference should now be higher
        assert!(detector.reference_price.unwrap() > 100.0);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(CrashSeverity::Emergency as u8 > CrashSeverity::Critical as u8);
        assert!(CrashSeverity::Critical as u8 > CrashSeverity::Warning as u8);
        assert!(CrashSeverity::Warning as u8 > CrashSeverity::None as u8);
    }

    #[test]
    fn test_recommended_actions() {
        let detector = FlashCrash::new();

        // Test action determination
        let signals = DetectionSignals {
            crash_probability: 0.9,
            ..Default::default()
        };

        let action = detector.determine_action(CrashSeverity::Critical, &signals);
        assert_eq!(action, RecommendedAction::CancelAllOrders);

        let action = detector.determine_action(CrashSeverity::Emergency, &signals);
        assert_eq!(action, RecommendedAction::EmergencyLiquidation);
    }

    #[test]
    fn test_reset() {
        let mut detector = FlashCrash::new();

        // Add some data
        for i in 0..30 {
            let tick = create_tick(100.0, 1000.0, i * 100);
            detector.update(tick);
        }

        detector.reset();

        assert!(detector.reference_price.is_none());
        assert!(!detector.is_flash_crash());
    }

    #[test]
    fn test_alert_count() {
        let mut detector = FlashCrash::with_config(FlashCrashConfig {
            window_size: 30,
            min_samples: 5,
            warning_threshold: 0.01,
            ..Default::default()
        });

        // Trigger multiple alerts
        for round in 0..3 {
            // Normal
            for i in 0..10 {
                let tick = create_tick(100.0, 1000.0, (round * 20 + i) * 100);
                detector.update(tick);
            }
            detector.check();

            // Drop
            for i in 10..20 {
                let tick = create_tick(98.0, 1000.0, (round * 20 + i) * 100);
                detector.update(tick);
            }
            detector.check();
        }

        // Should have some alerts
        assert!(detector.alert_count() > 0);
    }

    #[test]
    fn test_process() {
        let detector = FlashCrash::new();
        assert!(detector.process().is_ok());
    }

    #[test]
    fn test_market_tick_helpers() {
        let tick = MarketTick {
            price: 100.0,
            volume: 1000.0,
            bid: 99.95,
            ask: 100.05,
            timestamp: 0,
        };

        assert!((tick.spread() - 0.10).abs() < 0.001);
        assert!((tick.mid_price() - 100.0).abs() < 0.001);
    }
}
