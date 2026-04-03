//! Trading signal types and structures.
//!
//! This module defines the core signal types used to represent trading decisions
//! generated from model predictions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Direction of the trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalType {
    /// Open a long position (buy)
    Buy,
    /// Open a short position (sell)
    Sell,
    /// Hold current position (no action)
    Hold,
    /// Close current position
    Close,
}

impl fmt::Display for SignalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::Hold => write!(f, "HOLD"),
            SignalType::Close => write!(f, "CLOSE"),
        }
    }
}

impl SignalType {
    /// Returns true if this is an entry signal (Buy or Sell)
    pub fn is_entry(&self) -> bool {
        matches!(self, SignalType::Buy | SignalType::Sell)
    }

    /// Returns true if this is an exit signal (Close)
    pub fn is_exit(&self) -> bool {
        matches!(self, SignalType::Close)
    }

    /// Returns true if this is a hold signal
    pub fn is_hold(&self) -> bool {
        matches!(self, SignalType::Hold)
    }

    /// Returns the opposite signal type (Buy <-> Sell)
    pub fn opposite(&self) -> Option<Self> {
        match self {
            SignalType::Buy => Some(SignalType::Sell),
            SignalType::Sell => Some(SignalType::Buy),
            _ => None,
        }
    }
}

/// Strength/confidence level of a signal
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum SignalStrength {
    /// Very weak signal (confidence < 0.6)
    VeryWeak,
    /// Weak signal (confidence 0.6-0.7)
    Weak,
    /// Moderate signal (confidence 0.7-0.8)
    Moderate,
    /// Strong signal (confidence 0.8-0.9)
    Strong,
    /// Very strong signal (confidence >= 0.9)
    VeryStrong,
}

impl SignalStrength {
    /// Create a strength level from a confidence value
    pub fn from_confidence(confidence: f64) -> Self {
        if confidence >= 0.9 {
            SignalStrength::VeryStrong
        } else if confidence >= 0.8 {
            SignalStrength::Strong
        } else if confidence >= 0.7 {
            SignalStrength::Moderate
        } else if confidence >= 0.6 {
            SignalStrength::Weak
        } else {
            SignalStrength::VeryWeak
        }
    }

    /// Get the minimum confidence for this strength level
    pub fn min_confidence(&self) -> f64 {
        match self {
            SignalStrength::VeryWeak => 0.0,
            SignalStrength::Weak => 0.6,
            SignalStrength::Moderate => 0.7,
            SignalStrength::Strong => 0.8,
            SignalStrength::VeryStrong => 0.9,
        }
    }
}

impl fmt::Display for SignalStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalStrength::VeryWeak => write!(f, "VERY_WEAK"),
            SignalStrength::Weak => write!(f, "WEAK"),
            SignalStrength::Moderate => write!(f, "MODERATE"),
            SignalStrength::Strong => write!(f, "STRONG"),
            SignalStrength::VeryStrong => write!(f, "VERY_STRONG"),
        }
    }
}

/// Trading signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Type of signal (Buy, Sell, Hold, Close)
    pub signal_type: SignalType,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Signal strength derived from confidence
    pub strength: SignalStrength,

    /// Timestamp when the signal was generated
    pub timestamp: DateTime<Utc>,

    /// Asset identifier (e.g., "BTCUSD", "ETHUSDT")
    pub asset: String,

    /// Suggested position size as a fraction of capital (0.0 to 1.0)
    pub suggested_size: Option<f64>,

    /// Model prediction probabilities for each class [buy, hold, sell]
    pub class_probabilities: Option<Vec<f64>>,

    /// Additional metadata (e.g., features, model version)
    pub metadata: Option<serde_json::Value>,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(signal_type: SignalType, confidence: f64, asset: String) -> Self {
        Self {
            signal_type,
            confidence: confidence.clamp(0.0, 1.0),
            strength: SignalStrength::from_confidence(confidence),
            timestamp: Utc::now(),
            asset,
            suggested_size: None,
            class_probabilities: None,
            metadata: None,
        }
    }

    /// Create a signal with custom timestamp
    pub fn with_timestamp(
        signal_type: SignalType,
        confidence: f64,
        asset: String,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            signal_type,
            confidence: confidence.clamp(0.0, 1.0),
            strength: SignalStrength::from_confidence(confidence),
            timestamp,
            asset,
            suggested_size: None,
            class_probabilities: None,
            metadata: None,
        }
    }

    /// Set the suggested position size
    pub fn with_size(mut self, size: f64) -> Self {
        self.suggested_size = Some(size.clamp(0.0, 1.0));
        self
    }

    /// Set the class probabilities
    pub fn with_probabilities(mut self, probs: Vec<f64>) -> Self {
        self.class_probabilities = Some(probs);
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Check if the signal meets a minimum confidence threshold
    pub fn meets_threshold(&self, min_confidence: f64) -> bool {
        self.confidence >= min_confidence
    }

    /// Check if the signal is actionable (not Hold and meets minimum threshold)
    pub fn is_actionable(&self, min_confidence: f64) -> bool {
        !self.signal_type.is_hold() && self.meets_threshold(min_confidence)
    }

    /// Get age of the signal in seconds
    pub fn age_seconds(&self) -> i64 {
        let now = Utc::now();
        (now - self.timestamp).num_seconds()
    }

    /// Check if signal is stale (older than max_age_seconds)
    pub fn is_stale(&self, max_age_seconds: i64) -> bool {
        self.age_seconds() > max_age_seconds
    }

    /// Get a human-readable summary of the signal
    pub fn summary(&self) -> String {
        format!(
            "{} {} @ {:.2}% confidence ({})",
            self.signal_type,
            self.asset,
            self.confidence * 100.0,
            self.strength
        )
    }
}

impl fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Signal({} {} conf={:.2}% strength={} time={})",
            self.signal_type,
            self.asset,
            self.confidence * 100.0,
            self.strength,
            self.timestamp.format("%Y-%m-%d %H:%M:%S")
        )
    }
}

/// Collection of signals with filtering and analysis capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalBatch {
    /// All signals in this batch
    pub signals: Vec<TradingSignal>,

    /// Timestamp when this batch was created
    pub batch_timestamp: DateTime<Utc>,
}

impl SignalBatch {
    /// Create a new signal batch
    pub fn new(signals: Vec<TradingSignal>) -> Self {
        Self {
            signals,
            batch_timestamp: Utc::now(),
        }
    }

    /// Create an empty signal batch
    pub fn empty() -> Self {
        Self::default()
    }

    /// Add a signal to the batch
    pub fn add(&mut self, signal: TradingSignal) {
        self.signals.push(signal);
    }

    /// Get signals by type
    pub fn by_type(&self, signal_type: SignalType) -> Vec<&TradingSignal> {
        self.signals
            .iter()
            .filter(|s| s.signal_type == signal_type)
            .collect()
    }

    /// Get signals by asset
    pub fn by_asset(&self, asset: &str) -> Vec<&TradingSignal> {
        self.signals.iter().filter(|s| s.asset == asset).collect()
    }

    /// Get signals meeting minimum confidence
    pub fn above_confidence(&self, min_confidence: f64) -> Vec<&TradingSignal> {
        self.signals
            .iter()
            .filter(|s| s.confidence >= min_confidence)
            .collect()
    }

    /// Get only actionable signals (non-Hold, above threshold)
    pub fn actionable(&self, min_confidence: f64) -> Vec<&TradingSignal> {
        self.signals
            .iter()
            .filter(|s| s.is_actionable(min_confidence))
            .collect()
    }

    /// Get count of signals by type
    pub fn count_by_type(&self, signal_type: SignalType) -> usize {
        self.signals
            .iter()
            .filter(|s| s.signal_type == signal_type)
            .count()
    }

    /// Get average confidence across all signals
    pub fn average_confidence(&self) -> f64 {
        if self.signals.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.signals.iter().map(|s| s.confidence).sum();
        sum / self.signals.len() as f64
    }

    /// Get the strongest signal in the batch
    pub fn strongest(&self) -> Option<&TradingSignal> {
        self.signals
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }

    /// Remove stale signals
    pub fn remove_stale(&mut self, max_age_seconds: i64) {
        self.signals.retain(|s| !s.is_stale(max_age_seconds));
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        format!(
            "SignalBatch: {} signals (Buy: {}, Sell: {}, Hold: {}, Close: {}), Avg Confidence: {:.2}%",
            self.signals.len(),
            self.count_by_type(SignalType::Buy),
            self.count_by_type(SignalType::Sell),
            self.count_by_type(SignalType::Hold),
            self.count_by_type(SignalType::Close),
            self.average_confidence() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type_display() {
        assert_eq!(SignalType::Buy.to_string(), "BUY");
        assert_eq!(SignalType::Sell.to_string(), "SELL");
        assert_eq!(SignalType::Hold.to_string(), "HOLD");
        assert_eq!(SignalType::Close.to_string(), "CLOSE");
    }

    #[test]
    fn test_signal_type_predicates() {
        assert!(SignalType::Buy.is_entry());
        assert!(SignalType::Sell.is_entry());
        assert!(!SignalType::Hold.is_entry());
        assert!(!SignalType::Close.is_entry());

        assert!(SignalType::Close.is_exit());
        assert!(!SignalType::Buy.is_exit());

        assert!(SignalType::Hold.is_hold());
        assert!(!SignalType::Buy.is_hold());
    }

    #[test]
    fn test_signal_type_opposite() {
        assert_eq!(SignalType::Buy.opposite(), Some(SignalType::Sell));
        assert_eq!(SignalType::Sell.opposite(), Some(SignalType::Buy));
        assert_eq!(SignalType::Hold.opposite(), None);
        assert_eq!(SignalType::Close.opposite(), None);
    }

    #[test]
    fn test_signal_strength_from_confidence() {
        assert_eq!(
            SignalStrength::from_confidence(0.95),
            SignalStrength::VeryStrong
        );
        assert_eq!(
            SignalStrength::from_confidence(0.85),
            SignalStrength::Strong
        );
        assert_eq!(
            SignalStrength::from_confidence(0.75),
            SignalStrength::Moderate
        );
        assert_eq!(SignalStrength::from_confidence(0.65), SignalStrength::Weak);
        assert_eq!(
            SignalStrength::from_confidence(0.50),
            SignalStrength::VeryWeak
        );
    }

    #[test]
    fn test_trading_signal_creation() {
        let signal = TradingSignal::new(SignalType::Buy, 0.85, "BTCUSD".to_string());

        assert_eq!(signal.signal_type, SignalType::Buy);
        assert_eq!(signal.confidence, 0.85);
        assert_eq!(signal.strength, SignalStrength::Strong);
        assert_eq!(signal.asset, "BTCUSD");
        assert!(signal.suggested_size.is_none());
    }

    #[test]
    fn test_trading_signal_confidence_clamping() {
        let signal1 = TradingSignal::new(SignalType::Buy, 1.5, "BTCUSD".to_string());
        assert_eq!(signal1.confidence, 1.0);

        let signal2 = TradingSignal::new(SignalType::Sell, -0.5, "ETHUSDT".to_string());
        assert_eq!(signal2.confidence, 0.0);
    }

    #[test]
    fn test_trading_signal_with_builder() {
        let signal = TradingSignal::new(SignalType::Buy, 0.85, "BTCUSD".to_string())
            .with_size(0.25)
            .with_probabilities(vec![0.85, 0.10, 0.05]);

        assert_eq!(signal.suggested_size, Some(0.25));
        assert_eq!(signal.class_probabilities, Some(vec![0.85, 0.10, 0.05]));
    }

    #[test]
    fn test_signal_meets_threshold() {
        let signal = TradingSignal::new(SignalType::Buy, 0.75, "BTCUSD".to_string());

        assert!(signal.meets_threshold(0.7));
        assert!(signal.meets_threshold(0.75));
        assert!(!signal.meets_threshold(0.8));
    }

    #[test]
    fn test_signal_is_actionable() {
        let buy_signal = TradingSignal::new(SignalType::Buy, 0.75, "BTCUSD".to_string());
        assert!(buy_signal.is_actionable(0.7));

        let hold_signal = TradingSignal::new(SignalType::Hold, 0.75, "BTCUSD".to_string());
        assert!(!hold_signal.is_actionable(0.7));

        let weak_signal = TradingSignal::new(SignalType::Buy, 0.6, "BTCUSD".to_string());
        assert!(!weak_signal.is_actionable(0.7));
    }

    #[test]
    fn test_signal_batch() {
        let mut batch = SignalBatch::empty();

        batch.add(TradingSignal::new(
            SignalType::Buy,
            0.85,
            "BTCUSD".to_string(),
        ));
        batch.add(TradingSignal::new(
            SignalType::Sell,
            0.75,
            "ETHUSDT".to_string(),
        ));
        batch.add(TradingSignal::new(
            SignalType::Hold,
            0.60,
            "SOLUSDT".to_string(),
        ));

        assert_eq!(batch.signals.len(), 3);
        assert_eq!(batch.count_by_type(SignalType::Buy), 1);
        assert_eq!(batch.count_by_type(SignalType::Sell), 1);
        assert_eq!(batch.count_by_type(SignalType::Hold), 1);
    }

    #[test]
    fn test_signal_batch_filtering() {
        let mut batch = SignalBatch::empty();

        batch.add(TradingSignal::new(
            SignalType::Buy,
            0.85,
            "BTCUSD".to_string(),
        ));
        batch.add(TradingSignal::new(
            SignalType::Buy,
            0.65,
            "ETHUSDT".to_string(),
        ));
        batch.add(TradingSignal::new(
            SignalType::Sell,
            0.75,
            "BTCUSD".to_string(),
        ));

        let btc_signals = batch.by_asset("BTCUSD");
        assert_eq!(btc_signals.len(), 2);

        let high_conf = batch.above_confidence(0.75);
        assert_eq!(high_conf.len(), 2);

        let actionable = batch.actionable(0.7);
        assert_eq!(actionable.len(), 2);
    }

    #[test]
    fn test_signal_batch_strongest() {
        let mut batch = SignalBatch::empty();

        batch.add(TradingSignal::new(
            SignalType::Buy,
            0.75,
            "BTCUSD".to_string(),
        ));
        batch.add(TradingSignal::new(
            SignalType::Sell,
            0.90,
            "ETHUSDT".to_string(),
        ));
        batch.add(TradingSignal::new(
            SignalType::Buy,
            0.65,
            "SOLUSDT".to_string(),
        ));

        let strongest = batch.strongest().unwrap();
        assert_eq!(strongest.confidence, 0.90);
        assert_eq!(strongest.asset, "ETHUSDT");
    }

    #[test]
    fn test_signal_batch_average_confidence() {
        let mut batch = SignalBatch::empty();

        batch.add(TradingSignal::new(
            SignalType::Buy,
            0.8,
            "BTCUSD".to_string(),
        ));
        batch.add(TradingSignal::new(
            SignalType::Sell,
            0.6,
            "ETHUSDT".to_string(),
        ));

        assert_eq!(batch.average_confidence(), 0.7);
    }
}
