//! # JANUS Signal Types
//!
//! Canonical signal types for the JANUS signal generation service.
//! These types represent the output of technical analysis and ML models.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Signal type indicating the trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SignalType {
    /// Strong buy signal (high confidence)
    StrongBuy,
    /// Regular buy signal
    Buy,
    /// Hold current position
    Hold,
    /// Regular sell signal
    Sell,
    /// Strong sell signal (high confidence)
    StrongSell,
}

impl SignalType {
    /// Convert signal type to numeric value for ML models (-1.0 to 1.0)
    pub fn to_numeric(&self) -> f64 {
        match self {
            SignalType::StrongBuy => 1.0,
            SignalType::Buy => 0.5,
            SignalType::Hold => 0.0,
            SignalType::Sell => -0.5,
            SignalType::StrongSell => -1.0,
        }
    }

    /// Create signal type from numeric value
    pub fn from_numeric(value: f64) -> Self {
        if value >= 0.75 {
            SignalType::StrongBuy
        } else if value >= 0.25 {
            SignalType::Buy
        } else if value > -0.25 {
            SignalType::Hold
        } else if value > -0.75 {
            SignalType::Sell
        } else {
            SignalType::StrongSell
        }
    }

    /// Check if signal is actionable (not Hold)
    pub fn is_actionable(&self) -> bool {
        !matches!(self, SignalType::Hold)
    }

    /// Check if signal is bullish
    pub fn is_bullish(&self) -> bool {
        matches!(self, SignalType::Buy | SignalType::StrongBuy)
    }

    /// Check if signal is bearish
    pub fn is_bearish(&self) -> bool {
        matches!(self, SignalType::Sell | SignalType::StrongSell)
    }
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::StrongBuy => write!(f, "STRONG_BUY"),
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Hold => write!(f, "HOLD"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::StrongSell => write!(f, "STRONG_SELL"),
        }
    }
}

/// Trading timeframe
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Timeframe {
    /// 1 minute
    #[serde(rename = "1m")]
    M1,
    /// 5 minutes
    #[serde(rename = "5m")]
    M5,
    /// 15 minutes
    #[serde(rename = "15m")]
    M15,
    /// 1 hour
    #[serde(rename = "1h")]
    H1,
    /// 4 hours
    #[serde(rename = "4h")]
    H4,
    /// 1 day
    #[serde(rename = "1d")]
    D1,
}

impl Timeframe {
    /// Get timeframe duration in seconds
    pub fn to_seconds(&self) -> u64 {
        match self {
            Timeframe::M1 => 60,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::H1 => 3600,
            Timeframe::H4 => 14400,
            Timeframe::D1 => 86400,
        }
    }

    /// Get timeframe as string (e.g., "1m", "5m")
    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::M1 => "1m",
            Timeframe::M5 => "5m",
            Timeframe::M15 => "15m",
            Timeframe::H1 => "1h",
            Timeframe::H4 => "4h",
            Timeframe::D1 => "1d",
        }
    }
}

impl std::fmt::Display for Timeframe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Signal source (which component generated the signal)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalSource {
    /// Technical indicator (EMA, RSI, etc.)
    TechnicalIndicator { name: String },
    /// Machine learning model
    MlModel { model_id: String, version: String },
    /// Model inference (ONNX/ML inference)
    ModelInference { model_name: String, version: String },
    /// Combined strategy
    Strategy { name: String },
    /// Manual signal
    Manual { user_id: String },
}

/// Core trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Unique signal ID
    pub signal_id: String,

    /// Signal type (Buy, Sell, etc.)
    pub signal_type: SignalType,

    /// Trading symbol (e.g., "BTC/USD")
    pub symbol: String,

    /// Timeframe for this signal
    pub timeframe: Timeframe,

    /// Signal confidence (0.0 to 1.0)
    pub confidence: f64,

    /// Signal strength/magnitude (0.0 to 1.0)
    pub strength: f64,

    /// Timestamp when signal was generated
    pub timestamp: DateTime<Utc>,

    /// Signal source
    pub source: SignalSource,

    /// Optional entry price
    pub entry_price: Option<f64>,

    /// Optional stop loss price
    pub stop_loss: Option<f64>,

    /// Optional take profit price
    pub take_profit: Option<f64>,

    /// Predicted signal duration in seconds
    pub predicted_duration_seconds: Option<u64>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(
        symbol: String,
        signal_type: SignalType,
        timeframe: Timeframe,
        confidence: f64,
        source: SignalSource,
    ) -> Self {
        Self {
            signal_id: uuid::Uuid::new_v4().to_string(),
            signal_type,
            symbol,
            timeframe,
            confidence: confidence.clamp(0.0, 1.0),
            strength: 0.5, // Default strength
            timestamp: Utc::now(),
            source,
            entry_price: None,
            stop_loss: None,
            take_profit: None,
            predicted_duration_seconds: None,
            metadata: HashMap::new(),
        }
    }

    /// Set signal strength
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set entry price
    pub fn with_entry_price(mut self, price: f64) -> Self {
        self.entry_price = Some(price);
        self
    }

    /// Set stop loss price
    pub fn with_stop_loss(mut self, price: f64) -> Self {
        self.stop_loss = Some(price);
        self
    }

    /// Set take profit price
    pub fn with_take_profit(mut self, price: f64) -> Self {
        self.take_profit = Some(price);
        self
    }

    /// Set predicted duration
    pub fn with_duration(mut self, seconds: u64) -> Self {
        self.predicted_duration_seconds = Some(seconds);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Calculate risk/reward ratio if stop loss and take profit are set
    pub fn risk_reward_ratio(&self) -> Option<f64> {
        match (self.entry_price, self.stop_loss, self.take_profit) {
            (Some(entry), Some(sl), Some(tp)) => {
                let risk = (entry - sl).abs();
                let reward = (tp - entry).abs();
                if risk > 0.0 {
                    Some(reward / risk)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if signal meets minimum quality threshold
    pub fn meets_threshold(&self, min_confidence: f64, min_strength: f64) -> bool {
        self.confidence >= min_confidence && self.strength >= min_strength
    }

    /// Get signal age in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.timestamp).num_seconds()
    }

    /// Check if signal is stale (older than specified seconds)
    pub fn is_stale(&self, max_age_seconds: i64) -> bool {
        self.age_seconds() > max_age_seconds
    }
}

/// Batch of signals for efficient processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalBatch {
    /// Batch ID
    pub batch_id: String,

    /// Signals in this batch
    pub signals: Vec<TradingSignal>,

    /// Batch timestamp
    pub timestamp: DateTime<Utc>,

    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

impl SignalBatch {
    /// Create a new signal batch
    pub fn new(signals: Vec<TradingSignal>) -> Self {
        Self {
            batch_id: uuid::Uuid::new_v4().to_string(),
            signals,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Filter signals by minimum thresholds
    pub fn filter_by_threshold(&mut self, min_confidence: f64, min_strength: f64) {
        self.signals
            .retain(|s| s.meets_threshold(min_confidence, min_strength));
    }

    /// Remove stale signals
    pub fn remove_stale(&mut self, max_age_seconds: i64) {
        self.signals.retain(|s| !s.is_stale(max_age_seconds));
    }

    /// Get signals by type
    pub fn filter_by_type(&self, signal_type: SignalType) -> Vec<&TradingSignal> {
        self.signals
            .iter()
            .filter(|s| s.signal_type == signal_type)
            .collect()
    }

    /// Get actionable signals only
    pub fn actionable_signals(&self) -> Vec<&TradingSignal> {
        self.signals
            .iter()
            .filter(|s| s.signal_type.is_actionable())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type_numeric_conversion() {
        assert_eq!(SignalType::StrongBuy.to_numeric(), 1.0);
        assert_eq!(SignalType::Buy.to_numeric(), 0.5);
        assert_eq!(SignalType::Hold.to_numeric(), 0.0);
        assert_eq!(SignalType::Sell.to_numeric(), -0.5);
        assert_eq!(SignalType::StrongSell.to_numeric(), -1.0);

        assert_eq!(SignalType::from_numeric(0.9), SignalType::StrongBuy);
        assert_eq!(SignalType::from_numeric(0.3), SignalType::Buy);
        assert_eq!(SignalType::from_numeric(0.0), SignalType::Hold);
        assert_eq!(SignalType::from_numeric(-0.3), SignalType::Sell);
        assert_eq!(SignalType::from_numeric(-0.9), SignalType::StrongSell);
    }

    #[test]
    fn test_signal_type_checks() {
        assert!(SignalType::Buy.is_bullish());
        assert!(SignalType::StrongBuy.is_bullish());
        assert!(SignalType::Sell.is_bearish());
        assert!(SignalType::StrongSell.is_bearish());
        assert!(!SignalType::Hold.is_actionable());
        assert!(SignalType::Buy.is_actionable());
    }

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(Timeframe::M1.to_seconds(), 60);
        assert_eq!(Timeframe::M5.to_seconds(), 300);
        assert_eq!(Timeframe::H1.to_seconds(), 3600);
        assert_eq!(Timeframe::D1.to_seconds(), 86400);
        assert_eq!(Timeframe::M1.as_str(), "1m");
    }

    #[test]
    fn test_signal_creation() {
        let signal = TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Buy,
            Timeframe::H1,
            0.85,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        );

        assert_eq!(signal.symbol, "BTC/USD");
        assert_eq!(signal.signal_type, SignalType::Buy);
        assert_eq!(signal.confidence, 0.85);
        assert_eq!(signal.timeframe, Timeframe::H1);
        assert!(!signal.signal_id.is_empty());
    }

    #[test]
    fn test_signal_builder_pattern() {
        let signal = TradingSignal::new(
            "ETH/USD".to_string(),
            SignalType::Buy,
            Timeframe::M15,
            0.9,
            SignalSource::Strategy {
                name: "EMA_FLIP".to_string(),
            },
        )
        .with_strength(0.8)
        .with_entry_price(2000.0)
        .with_stop_loss(1950.0)
        .with_take_profit(2100.0)
        .with_duration(1800);

        assert_eq!(signal.strength, 0.8);
        assert_eq!(signal.entry_price, Some(2000.0));
        assert_eq!(signal.stop_loss, Some(1950.0));
        assert_eq!(signal.take_profit, Some(2100.0));
        assert_eq!(signal.predicted_duration_seconds, Some(1800));
    }

    #[test]
    fn test_risk_reward_ratio() {
        let signal = TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Buy,
            Timeframe::H1,
            0.8,
            SignalSource::TechnicalIndicator {
                name: "RSI".to_string(),
            },
        )
        .with_entry_price(50000.0)
        .with_stop_loss(49000.0)
        .with_take_profit(52000.0);

        let rr = signal.risk_reward_ratio().unwrap();
        assert!((rr - 2.0).abs() < 0.01); // Risk: 1000, Reward: 2000, R/R = 2.0
    }

    #[test]
    fn test_signal_threshold() {
        let signal = TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Buy,
            Timeframe::H1,
            0.7,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        )
        .with_strength(0.6);

        assert!(signal.meets_threshold(0.5, 0.5));
        assert!(signal.meets_threshold(0.7, 0.6));
        assert!(!signal.meets_threshold(0.8, 0.6));
        assert!(!signal.meets_threshold(0.7, 0.7));
    }

    #[test]
    fn test_signal_batch() {
        let signals = vec![
            TradingSignal::new(
                "BTC/USD".to_string(),
                SignalType::Buy,
                Timeframe::H1,
                0.8,
                SignalSource::TechnicalIndicator {
                    name: "EMA".to_string(),
                },
            ),
            TradingSignal::new(
                "ETH/USD".to_string(),
                SignalType::Sell,
                Timeframe::M15,
                0.6,
                SignalSource::TechnicalIndicator {
                    name: "RSI".to_string(),
                },
            ),
            TradingSignal::new(
                "SOL/USD".to_string(),
                SignalType::Hold,
                Timeframe::H4,
                0.5,
                SignalSource::Strategy {
                    name: "MACD".to_string(),
                },
            ),
        ];

        let mut batch = SignalBatch::new(signals);
        assert_eq!(batch.signals.len(), 3);

        // Filter by threshold
        batch.filter_by_threshold(0.7, 0.5);
        assert_eq!(batch.signals.len(), 1);
        assert_eq!(batch.signals[0].symbol, "BTC/USD");
    }

    #[test]
    fn test_batch_actionable_signals() {
        let signals = vec![
            TradingSignal::new(
                "BTC/USD".to_string(),
                SignalType::Buy,
                Timeframe::H1,
                0.8,
                SignalSource::TechnicalIndicator {
                    name: "EMA".to_string(),
                },
            ),
            TradingSignal::new(
                "ETH/USD".to_string(),
                SignalType::Hold,
                Timeframe::M15,
                0.6,
                SignalSource::TechnicalIndicator {
                    name: "RSI".to_string(),
                },
            ),
            TradingSignal::new(
                "SOL/USD".to_string(),
                SignalType::Sell,
                Timeframe::H4,
                0.7,
                SignalSource::Strategy {
                    name: "MACD".to_string(),
                },
            ),
        ];

        let batch = SignalBatch::new(signals);
        let actionable = batch.actionable_signals();
        assert_eq!(actionable.len(), 2);
    }
}
