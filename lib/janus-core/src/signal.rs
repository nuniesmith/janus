//! Signal types and broadcast bus for inter-module communication

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalType {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold (no action)
    Hold,
    /// Close existing position
    Close,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::Hold => write!(f, "HOLD"),
            SignalType::Close => write!(f, "CLOSE"),
        }
    }
}

/// Signal priority for routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignalPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Trading signal shared across all modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Unique signal ID
    pub id: String,
    /// Trading symbol (e.g., BTCUSD)
    pub symbol: String,
    /// Signal type (buy, sell, hold, close)
    pub signal_type: SignalType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Signal timestamp
    pub timestamp: DateTime<Utc>,
    /// Signal priority
    pub priority: SignalPriority,
    /// Source module that generated the signal
    pub source: String,
    /// Strategy ID that generated this signal
    pub strategy_id: Option<String>,
    /// Target price (optional)
    pub target_price: Option<f64>,
    /// Stop loss price (optional)
    pub stop_loss: Option<f64>,
    /// Take profit price (optional)
    pub take_profit: Option<f64>,
    /// Position size (optional)
    pub quantity: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Signal {
    /// Create a new signal
    pub fn new(symbol: impl Into<String>, signal_type: SignalType, confidence: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.into(),
            signal_type,
            confidence: confidence.clamp(0.0, 1.0),
            timestamp: Utc::now(),
            priority: SignalPriority::Normal,
            source: "unknown".to_string(),
            strategy_id: None,
            target_price: None,
            stop_loss: None,
            take_profit: None,
            quantity: None,
            metadata: HashMap::new(),
        }
    }

    /// Builder pattern methods
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    pub fn with_priority(mut self, priority: SignalPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_strategy(mut self, strategy_id: impl Into<String>) -> Self {
        self.strategy_id = Some(strategy_id.into());
        self
    }

    pub fn with_target_price(mut self, price: f64) -> Self {
        self.target_price = Some(price);
        self
    }

    pub fn with_stop_loss(mut self, price: f64) -> Self {
        self.stop_loss = Some(price);
        self
    }

    pub fn with_take_profit(mut self, price: f64) -> Self {
        self.take_profit = Some(price);
        self
    }

    pub fn with_quantity(mut self, quantity: f64) -> Self {
        self.quantity = Some(quantity);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if signal is actionable (not a hold)
    pub fn is_actionable(&self) -> bool {
        !matches!(self.signal_type, SignalType::Hold)
    }

    /// Check if signal meets confidence threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Signal broadcast bus for inter-module communication
///
/// This allows modules to publish signals and other modules to subscribe
/// without direct coupling.
pub struct SignalBus {
    /// Broadcast sender for signals
    tx: broadcast::Sender<Signal>,
    /// Channel capacity
    capacity: usize,
}

impl SignalBus {
    /// Create a new signal bus
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx, capacity }
    }

    /// Publish a signal to all subscribers
    pub fn publish(&self, signal: Signal) -> crate::Result<usize> {
        let receivers = self.tx.send(signal)?;
        Ok(receivers)
    }

    /// Subscribe to signals
    pub fn subscribe(&self) -> broadcast::Receiver<Signal> {
        self.tx.subscribe()
    }

    /// Get current subscriber count
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Get channel capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for SignalBus {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl Clone for SignalBus {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            capacity: self.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new("BTCUSD", SignalType::Buy, 0.85)
            .with_source("forward")
            .with_strategy("momentum_v1");

        assert_eq!(signal.symbol, "BTCUSD");
        assert_eq!(signal.signal_type, SignalType::Buy);
        assert_eq!(signal.confidence, 0.85);
        assert_eq!(signal.source, "forward");
        assert!(signal.is_actionable());
    }

    #[test]
    fn test_signal_bus() {
        let bus = SignalBus::new(100);
        let _rx = bus.subscribe();

        let signal = Signal::new("ETHUSDT", SignalType::Sell, 0.9);
        let _ = bus.publish(signal.clone());

        // Note: In a real async test, you'd await the receive
        assert_eq!(bus.subscriber_count(), 1);
    }

    #[test]
    fn test_confidence_clamping() {
        let signal = Signal::new("BTCUSD", SignalType::Buy, 1.5);
        assert_eq!(signal.confidence, 1.0);

        let signal2 = Signal::new("BTCUSD", SignalType::Buy, -0.5);
        assert_eq!(signal2.confidence, 0.0);
    }
}
