//! # WebSocket Message Types
//!
//! Comprehensive message type definitions for WebSocket communication.
//! This includes all message types for client-server communication,
//! signal updates, risk alerts, portfolio updates, and market data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Main WebSocket message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    // Client -> Server
    #[serde(rename = "subscribe")]
    Subscribe(SubscribeRequest),

    #[serde(rename = "unsubscribe")]
    Unsubscribe(UnsubscribeRequest),

    #[serde(rename = "ping")]
    Ping,

    // Server -> Client
    #[serde(rename = "signal_update")]
    SignalUpdate(SignalUpdate),

    #[serde(rename = "portfolio_update")]
    PortfolioUpdate(PortfolioUpdate),

    #[serde(rename = "risk_alert")]
    RiskAlert(RiskAlert),

    #[serde(rename = "performance_update")]
    PerformanceUpdate(PerformanceUpdate),

    #[serde(rename = "market_data")]
    MarketData(MarketDataUpdate),

    #[serde(rename = "pong")]
    Pong,

    #[serde(rename = "error")]
    Error(ErrorMessage),

    // System
    #[serde(rename = "welcome")]
    Welcome(WelcomeMessage),

    #[serde(rename = "goodbye")]
    Goodbye(GoodbyeMessage),
}

/// Subscribe request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeRequest {
    pub symbols: Option<Vec<String>>,
    pub min_confidence: Option<f64>,
    pub signal_types: Option<Vec<String>>,
    pub portfolio_updates: bool,
    pub risk_alerts: bool,
}

/// Unsubscribe request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscribeRequest {
    pub symbols: Option<Vec<String>>,
}

/// Welcome message sent on connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelcomeMessage {
    pub session_id: Uuid,
    pub server_version: String,
    pub timestamp: DateTime<Utc>,
    pub capabilities: Vec<String>,
}

impl Default for WelcomeMessage {
    fn default() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            server_version: "1.0.0".to_string(),
            timestamp: Utc::now(),
            capabilities: vec![
                "signals".to_string(),
                "portfolio".to_string(),
                "risk_alerts".to_string(),
                "market_data".to_string(),
            ],
        }
    }
}

/// Goodbye message sent on disconnect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoodbyeMessage {
    pub reason: String,
    pub reconnect: bool,
}

/// Error message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    pub code: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Signal type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    Entry,
    Exit,
    StopLoss,
    TakeProfit,
    Rebalance,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Entry => write!(f, "Entry"),
            SignalType::Exit => write!(f, "Exit"),
            SignalType::StopLoss => write!(f, "StopLoss"),
            SignalType::TakeProfit => write!(f, "TakeProfit"),
            SignalType::Rebalance => write!(f, "Rebalance"),
        }
    }
}

/// Action type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Buy,
    Sell,
    Hold,
}

/// Signal update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalUpdate {
    pub signal_id: Uuid,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub signal_type: SignalType,
    pub action: Action,
    pub confidence: f64,
    pub entry_price: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Vec<f64>,
    pub position_size: Option<f64>,
    pub risk_reward_ratio: Option<f64>,
    pub strategy: String,
    pub timeframe: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Portfolio update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioUpdate {
    pub timestamp: DateTime<Utc>,
    pub portfolio_id: Uuid,
    pub total_value: f64,
    pub cash: f64,
    pub positions: Vec<PositionSnapshot>,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub daily_pnl: f64,
    pub daily_return: f64,
    pub total_return: f64,
    pub exposures: HashMap<String, f64>,
}

/// Position snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSnapshot {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub position_value: f64,
    pub weight: f64,
    pub opened_at: DateTime<Utc>,
}

/// Risk alert severity
/// Alert severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Critical => write!(f, "Critical"),
            AlertSeverity::High => write!(f, "High"),
            AlertSeverity::Medium => write!(f, "Medium"),
            AlertSeverity::Low => write!(f, "Low"),
        }
    }
}

/// Risk alert type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskAlertType {
    PositionSizeExceeded,
    DailyLossLimitApproaching,
    DailyLossLimitExceeded,
    CorrelationRiskHigh,
    VolatilitySpike,
    DrawdownLimit,
    LeverageExceeded,
    MarginCall,
    ConcentrationRisk,
}

/// Risk alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlert {
    pub alert_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub alert_type: RiskAlertType,
    pub message: String,
    pub affected_symbols: Vec<String>,
    pub current_value: f64,
    pub threshold: f64,
    pub recommended_action: Option<String>,
}

/// Performance update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceUpdate {
    pub timestamp: DateTime<Utc>,
    pub portfolio_id: Uuid,
    pub period: String, // "daily", "weekly", "monthly", "all_time"
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub average_win: f64,
    pub average_loss: f64,
}

/// Market data type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataType {
    Tick,
    Candle,
    OrderBook,
    Trade,
}

/// Market data update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataUpdate {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub data_type: MarketDataType,
    pub data: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_update_serialization() {
        let signal = SignalUpdate {
            signal_id: Uuid::new_v4(),
            symbol: "BTCUSD".to_string(),
            timestamp: Utc::now(),
            signal_type: SignalType::Entry,
            action: Action::Buy,
            confidence: 0.85,
            entry_price: 43250.0,
            stop_loss: Some(42000.0),
            take_profit: vec![44000.0, 45000.0],
            position_size: Some(0.5),
            risk_reward_ratio: Some(2.5),
            strategy: "momentum".to_string(),
            timeframe: "1h".to_string(),
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("BTCUSD"));

        let deserialized: SignalUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.symbol, "BTCUSD");
    }

    #[test]
    fn test_websocket_message_serialization() {
        let signal = SignalUpdate {
            signal_id: Uuid::new_v4(),
            symbol: "ETHUSDT".to_string(),
            timestamp: Utc::now(),
            signal_type: SignalType::Exit,
            action: Action::Sell,
            confidence: 0.75,
            entry_price: 2450.0,
            stop_loss: None,
            take_profit: vec![],
            position_size: None,
            risk_reward_ratio: None,
            strategy: "mean_reversion".to_string(),
            timeframe: "15m".to_string(),
            metadata: HashMap::new(),
        };

        let message = WebSocketMessage::SignalUpdate(signal);
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("signal_update"));

        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        match deserialized {
            WebSocketMessage::SignalUpdate(s) => assert_eq!(s.symbol, "ETHUSDT"),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_risk_alert_serialization() {
        let alert = RiskAlert {
            alert_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            severity: AlertSeverity::High,
            alert_type: RiskAlertType::DailyLossLimitApproaching,
            message: "Daily loss approaching limit".to_string(),
            affected_symbols: vec!["BTCUSD".to_string()],
            current_value: -450.0,
            threshold: -500.0,
            recommended_action: Some("Consider reducing position sizes".to_string()),
        };

        let json = serde_json::to_string(&alert).unwrap();
        let deserialized: RiskAlert = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.current_value, -450.0);
    }

    #[test]
    fn test_portfolio_update_serialization() {
        let update = PortfolioUpdate {
            timestamp: Utc::now(),
            portfolio_id: Uuid::new_v4(),
            total_value: 10500.0,
            cash: 5000.0,
            positions: vec![],
            unrealized_pnl: 500.0,
            realized_pnl: 1000.0,
            daily_pnl: 250.0,
            daily_return: 0.025,
            total_return: 0.05,
            exposures: HashMap::new(),
        };

        let json = serde_json::to_string(&update).unwrap();
        let deserialized: PortfolioUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_value, 10500.0);
    }

    #[test]
    fn test_subscribe_request_serialization() {
        let request = SubscribeRequest {
            symbols: Some(vec!["BTCUSD".to_string(), "ETHUSDT".to_string()]),
            min_confidence: Some(0.7),
            signal_types: Some(vec!["Entry".to_string()]),
            portfolio_updates: true,
            risk_alerts: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: SubscribeRequest = serde_json::from_str(&json).unwrap();
        assert!(deserialized.symbols.is_some());
        assert_eq!(deserialized.min_confidence, Some(0.7));
    }

    #[test]
    fn test_welcome_message_default() {
        let welcome = WelcomeMessage::default();
        assert_eq!(welcome.server_version, "1.0.0");
        assert!(welcome.capabilities.contains(&"signals".to_string()));
    }
}
