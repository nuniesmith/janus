//! Signal filtering and validation.
//!
//! This module provides filters to validate and filter trading signals
//! based on various criteria such as confidence, risk limits, and timing.

use super::types::{SignalBatch, SignalType, TradingSignal};
use std::collections::HashMap;

/// Configuration for signal filtering
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Minimum confidence threshold
    pub min_confidence: f64,

    /// Maximum number of concurrent positions
    pub max_positions: usize,

    /// Maximum signal age in seconds before considered stale
    pub max_signal_age_seconds: i64,

    /// Minimum time between signals for the same asset (seconds)
    pub min_signal_interval_seconds: i64,

    /// Maximum position size as fraction of capital
    pub max_position_size: f64,

    /// Enable signal type filters
    pub allowed_signal_types: Vec<SignalType>,

    /// Assets to allow (empty = all allowed)
    pub allowed_assets: Vec<String>,

    /// Assets to block
    pub blocked_assets: Vec<String>,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            max_positions: 10,
            max_signal_age_seconds: 60,
            min_signal_interval_seconds: 300, // 5 minutes
            max_position_size: 0.1,           // 10% of capital
            allowed_signal_types: vec![SignalType::Buy, SignalType::Sell, SignalType::Close],
            allowed_assets: vec![],
            blocked_assets: vec![],
        }
    }
}

impl FilterConfig {
    /// Create a conservative filter config (stricter rules)
    pub fn conservative() -> Self {
        Self {
            min_confidence: 0.8,
            max_positions: 5,
            max_signal_age_seconds: 30,
            min_signal_interval_seconds: 600, // 10 minutes
            max_position_size: 0.05,          // 5% of capital
            ..Default::default()
        }
    }

    /// Create an aggressive filter config (looser rules)
    pub fn aggressive() -> Self {
        Self {
            min_confidence: 0.6,
            max_positions: 20,
            max_signal_age_seconds: 120,
            min_signal_interval_seconds: 60, // 1 minute
            max_position_size: 0.2,          // 20% of capital
            ..Default::default()
        }
    }
}

/// Reason why a signal was filtered out
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FilterReason {
    /// Confidence below minimum threshold
    LowConfidence,
    /// Signal is too old
    Stale,
    /// Too soon since last signal for this asset
    TooFrequent,
    /// Maximum positions already reached
    MaxPositionsReached,
    /// Position size too large
    PositionSizeTooLarge,
    /// Signal type not allowed
    SignalTypeNotAllowed,
    /// Asset not in allowed list
    AssetNotAllowed,
    /// Asset is blocked
    AssetBlocked,
    /// Hold signals are typically not actionable
    HoldSignal,
}

impl FilterReason {
    pub fn description(&self) -> &str {
        match self {
            FilterReason::LowConfidence => "Confidence below minimum threshold",
            FilterReason::Stale => "Signal is too old",
            FilterReason::TooFrequent => "Too soon since last signal",
            FilterReason::MaxPositionsReached => "Maximum positions reached",
            FilterReason::PositionSizeTooLarge => "Position size exceeds limit",
            FilterReason::SignalTypeNotAllowed => "Signal type not allowed",
            FilterReason::AssetNotAllowed => "Asset not in allowed list",
            FilterReason::AssetBlocked => "Asset is blocked",
            FilterReason::HoldSignal => "Hold signal (no action needed)",
        }
    }
}

/// Result of filtering a signal
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Whether the signal passed all filters
    pub passed: bool,
    /// Reasons why the signal was filtered (if any)
    pub reasons: Vec<FilterReason>,
}

impl FilterResult {
    pub fn pass() -> Self {
        Self {
            passed: true,
            reasons: vec![],
        }
    }

    pub fn fail(reason: FilterReason) -> Self {
        Self {
            passed: false,
            reasons: vec![reason],
        }
    }

    pub fn fail_multiple(reasons: Vec<FilterReason>) -> Self {
        Self {
            passed: false,
            reasons,
        }
    }
}

/// Signal filter that applies various validation rules
pub struct SignalFilter {
    config: FilterConfig,
    last_signal_time: HashMap<String, i64>,
    current_positions: usize,
}

impl SignalFilter {
    /// Create a new signal filter
    pub fn new(config: FilterConfig) -> Self {
        Self {
            config,
            last_signal_time: HashMap::new(),
            current_positions: 0,
        }
    }

    /// Update the current number of open positions
    pub fn set_positions(&mut self, count: usize) {
        self.current_positions = count;
    }

    /// Record that a signal was executed for an asset
    pub fn record_signal(&mut self, asset: &str, timestamp: i64) {
        self.last_signal_time.insert(asset.to_string(), timestamp);
    }

    /// Clear the signal history
    pub fn clear_history(&mut self) {
        self.last_signal_time.clear();
    }

    /// Filter a single signal
    pub fn filter(&self, signal: &TradingSignal) -> FilterResult {
        let mut reasons = Vec::new();

        // Check confidence
        if signal.confidence < self.config.min_confidence {
            reasons.push(FilterReason::LowConfidence);
        }

        // Check if signal is stale
        if signal.is_stale(self.config.max_signal_age_seconds) {
            reasons.push(FilterReason::Stale);
        }

        // Check signal type
        if !self
            .config
            .allowed_signal_types
            .contains(&signal.signal_type)
        {
            reasons.push(FilterReason::SignalTypeNotAllowed);
        }

        // Check if hold signal
        if signal.signal_type.is_hold() {
            reasons.push(FilterReason::HoldSignal);
        }

        // Check asset whitelist
        if !self.config.allowed_assets.is_empty()
            && !self.config.allowed_assets.contains(&signal.asset)
        {
            reasons.push(FilterReason::AssetNotAllowed);
        }

        // Check asset blacklist
        if self.config.blocked_assets.contains(&signal.asset) {
            reasons.push(FilterReason::AssetBlocked);
        }

        // Check position size
        if let Some(size) = signal.suggested_size {
            if size > self.config.max_position_size {
                reasons.push(FilterReason::PositionSizeTooLarge);
            }
        }

        // Check position limits (only for entry signals)
        if signal.signal_type.is_entry() && self.current_positions >= self.config.max_positions {
            reasons.push(FilterReason::MaxPositionsReached);
        }

        // Check signal frequency
        if let Some(&last_time) = self.last_signal_time.get(&signal.asset) {
            let time_since_last = signal.timestamp.timestamp() - last_time;
            if time_since_last < self.config.min_signal_interval_seconds {
                reasons.push(FilterReason::TooFrequent);
            }
        }

        if reasons.is_empty() {
            FilterResult::pass()
        } else {
            FilterResult::fail_multiple(reasons)
        }
    }

    /// Filter a batch of signals
    pub fn filter_batch<'a>(
        &self,
        batch: &'a SignalBatch,
    ) -> Vec<(&'a TradingSignal, FilterResult)> {
        batch
            .signals
            .iter()
            .map(|signal| (signal, self.filter(signal)))
            .collect()
    }

    /// Get only the signals that passed filtering
    pub fn get_passed<'a>(&self, batch: &'a SignalBatch) -> Vec<&'a TradingSignal> {
        batch
            .signals
            .iter()
            .filter(|signal| self.filter(signal).passed)
            .collect()
    }

    /// Get signals that failed filtering with their reasons
    pub fn get_failed<'a>(
        &self,
        batch: &'a SignalBatch,
    ) -> Vec<(&'a TradingSignal, Vec<FilterReason>)> {
        batch
            .signals
            .iter()
            .filter_map(|signal| {
                let result = self.filter(signal);
                if !result.passed {
                    Some((signal, result.reasons))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get filtering statistics
    pub fn get_stats(&self, batch: &SignalBatch) -> FilterStats {
        let results = self.filter_batch(batch);
        let total = results.len();
        let passed = results.iter().filter(|(_, r)| r.passed).count();
        let failed = total - passed;

        let mut reason_counts: HashMap<FilterReason, usize> = HashMap::new();
        for (_, result) in &results {
            for reason in &result.reasons {
                *reason_counts.entry(reason.clone()).or_insert(0) += 1;
            }
        }

        FilterStats {
            total_signals: total,
            passed: passed,
            failed: failed,
            pass_rate: if total > 0 {
                passed as f64 / total as f64
            } else {
                0.0
            },
            reason_counts,
        }
    }
}

/// Statistics about filtering results
#[derive(Debug, Clone)]
pub struct FilterStats {
    pub total_signals: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f64,
    pub reason_counts: HashMap<FilterReason, usize>,
}

impl FilterStats {
    pub fn summary(&self) -> String {
        format!(
            "Filter Stats: {}/{} passed ({:.1}%), {} failed",
            self.passed,
            self.total_signals,
            self.pass_rate * 100.0,
            self.failed
        )
    }

    pub fn top_failure_reasons(&self, n: usize) -> Vec<(FilterReason, usize)> {
        let mut reasons: Vec<_> = self.reason_counts.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1));
        reasons
            .into_iter()
            .take(n)
            .map(|(r, c)| (r.clone(), *c))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_signal(signal_type: SignalType, confidence: f64, asset: &str) -> TradingSignal {
        TradingSignal::new(signal_type, confidence, asset.to_string())
    }

    #[test]
    fn test_filter_config_default() {
        let config = FilterConfig::default();
        assert_eq!(config.min_confidence, 0.7);
        assert_eq!(config.max_positions, 10);
    }

    #[test]
    fn test_filter_config_presets() {
        let conservative = FilterConfig::conservative();
        assert_eq!(conservative.min_confidence, 0.8);
        assert_eq!(conservative.max_positions, 5);

        let aggressive = FilterConfig::aggressive();
        assert_eq!(aggressive.min_confidence, 0.6);
        assert_eq!(aggressive.max_positions, 20);
    }

    #[test]
    fn test_filter_low_confidence() {
        let config = FilterConfig::default();
        let filter = SignalFilter::new(config);

        let signal = create_test_signal(SignalType::Buy, 0.5, "BTCUSD");
        let result = filter.filter(&signal);

        assert!(!result.passed);
        assert!(result.reasons.contains(&FilterReason::LowConfidence));
    }

    #[test]
    fn test_filter_high_confidence() {
        let config = FilterConfig::default();
        let filter = SignalFilter::new(config);

        let signal = create_test_signal(SignalType::Buy, 0.85, "BTCUSD");
        let result = filter.filter(&signal);

        assert!(result.passed);
    }

    #[test]
    fn test_filter_hold_signal() {
        let config = FilterConfig::default();
        let filter = SignalFilter::new(config);

        let signal = create_test_signal(SignalType::Hold, 0.85, "BTCUSD");
        let result = filter.filter(&signal);

        assert!(!result.passed);
        assert!(result.reasons.contains(&FilterReason::HoldSignal));
    }

    #[test]
    fn test_filter_max_positions() {
        let config = FilterConfig::default();
        let mut filter = SignalFilter::new(config);
        filter.set_positions(10); // At max

        let signal = create_test_signal(SignalType::Buy, 0.85, "BTCUSD");
        let result = filter.filter(&signal);

        assert!(!result.passed);
        assert!(result.reasons.contains(&FilterReason::MaxPositionsReached));
    }

    #[test]
    fn test_filter_max_positions_close_signal() {
        let config = FilterConfig::default();
        let mut filter = SignalFilter::new(config);
        filter.set_positions(10); // At max

        // Close signals should not be affected by position limits
        let signal = create_test_signal(SignalType::Close, 0.85, "BTCUSD");
        let result = filter.filter(&signal);

        assert!(result.passed || !result.reasons.contains(&FilterReason::MaxPositionsReached));
    }

    #[test]
    fn test_filter_blocked_asset() {
        let mut config = FilterConfig::default();
        config.blocked_assets = vec!["BTCUSD".to_string()];
        let filter = SignalFilter::new(config);

        let signal = create_test_signal(SignalType::Buy, 0.85, "BTCUSD");
        let result = filter.filter(&signal);

        assert!(!result.passed);
        assert!(result.reasons.contains(&FilterReason::AssetBlocked));
    }

    #[test]
    fn test_filter_allowed_assets() {
        let mut config = FilterConfig::default();
        config.allowed_assets = vec!["ETHUSDT".to_string()];
        let filter = SignalFilter::new(config);

        let signal = create_test_signal(SignalType::Buy, 0.85, "BTCUSD");
        let result = filter.filter(&signal);

        assert!(!result.passed);
        assert!(result.reasons.contains(&FilterReason::AssetNotAllowed));
    }

    #[test]
    fn test_filter_position_size() {
        let config = FilterConfig::default();
        let filter = SignalFilter::new(config);

        let signal = create_test_signal(SignalType::Buy, 0.85, "BTCUSD").with_size(0.5);
        let result = filter.filter(&signal);

        assert!(!result.passed);
        assert!(result.reasons.contains(&FilterReason::PositionSizeTooLarge));
    }

    #[test]
    fn test_filter_signal_frequency() {
        let config = FilterConfig::default();
        let mut filter = SignalFilter::new(config);

        let now = Utc::now().timestamp();
        filter.record_signal("BTCUSD", now);

        // Signal too soon
        let signal = create_test_signal(SignalType::Buy, 0.85, "BTCUSD");
        let result = filter.filter(&signal);

        assert!(!result.passed);
        assert!(result.reasons.contains(&FilterReason::TooFrequent));
    }

    #[test]
    fn test_filter_batch() {
        let config = FilterConfig::default();
        let filter = SignalFilter::new(config);

        let mut batch = SignalBatch::empty();
        batch.add(create_test_signal(SignalType::Buy, 0.85, "BTCUSD"));
        batch.add(create_test_signal(SignalType::Sell, 0.50, "ETHUSDT")); // Low confidence
        batch.add(create_test_signal(SignalType::Hold, 0.90, "SOLUSDT")); // Hold

        let results = filter.filter_batch(&batch);
        assert_eq!(results.len(), 3);

        let passed = filter.get_passed(&batch);
        assert_eq!(passed.len(), 1); // Only first signal passes
    }

    #[test]
    fn test_filter_stats() {
        let config = FilterConfig::default();
        let filter = SignalFilter::new(config);

        let mut batch = SignalBatch::empty();
        batch.add(create_test_signal(SignalType::Buy, 0.85, "BTCUSD"));
        batch.add(create_test_signal(SignalType::Sell, 0.50, "ETHUSDT"));
        batch.add(create_test_signal(SignalType::Hold, 0.90, "SOLUSDT"));

        let stats = filter.get_stats(&batch);
        assert_eq!(stats.total_signals, 3);
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.failed, 2);
        assert!((stats.pass_rate - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_clear_history() {
        let config = FilterConfig::default();
        let mut filter = SignalFilter::new(config);

        filter.record_signal("BTCUSD", Utc::now().timestamp());
        assert!(!filter.last_signal_time.is_empty());

        filter.clear_history();
        assert!(filter.last_signal_time.is_empty());
    }
}
