//! Data models for trades, signals, and market data
//!
//! This module contains all the core data structures used throughout
//! the trading system, including trades, signals, account state, and
//! performance metrics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod account;
pub mod performance;
pub mod prop_firm;

// Re-export common types
pub use account::{AccountState, AppConfig, ConfigLoader, HyroTraderConfig};
pub use performance::{PerformanceMetrics, TradeResult};
pub use prop_firm::{ChallengeType, PropFirmValidator};

/// Trading direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
}

impl Direction {
    pub fn opposite(&self) -> Self {
        match self {
            Direction::Long => Direction::Short,
            Direction::Short => Direction::Long,
        }
    }
}

/// Trade status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeStatus {
    Open,
    Closed,
    Cancelled,
}

/// Exit reason for closed trades
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExitReason {
    TakeProfit1,
    TakeProfit2,
    TakeProfit3,
    TakeProfit4,
    StopLoss,
    TrailingStop,
    Manual,
    EmaFlip,
    RuleViolation,
}

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Opportunity,
    Entry,
    Exit,
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub signal_id: String,
    pub signal_type: SignalType,

    // Market info
    pub symbol: String,
    pub timeframe: String,
    pub direction: Direction,

    // Signal details
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub confidence: f64, // 0.0 to 1.0

    // Technical indicators
    pub ema_fast: Option<f64>,
    pub ema_slow: Option<f64>,
    pub atr: Option<f64>,

    // Suggested levels
    pub suggested_entry: Option<f64>,
    pub suggested_stop_loss: Option<f64>,
    pub suggested_tp1: Option<f64>,
    pub suggested_tp2: Option<f64>,
    pub suggested_tp3: Option<f64>,
    pub suggested_tp4: Option<f64>,

    // Risk metrics
    pub risk_reward_ratio: Option<f64>,
    pub position_size_usd: Option<f64>,

    // Metadata
    pub reason: Option<String>,
}

impl Signal {
    /// Create a new signal
    pub fn new(
        signal_type: SignalType,
        symbol: String,
        timeframe: String,
        direction: Direction,
        price: f64,
        confidence: f64,
    ) -> Self {
        Self {
            signal_id: Uuid::new_v4().to_string(),
            signal_type,
            symbol,
            timeframe,
            direction,
            timestamp: Utc::now(),
            price,
            confidence,
            ema_fast: None,
            ema_slow: None,
            atr: None,
            suggested_entry: None,
            suggested_stop_loss: None,
            suggested_tp1: None,
            suggested_tp2: None,
            suggested_tp3: None,
            suggested_tp4: None,
            risk_reward_ratio: None,
            position_size_usd: None,
            reason: None,
        }
    }
}

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Position data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub direction: Direction,
    pub entry_price: f64,
    pub current_price: f64,
    pub position_size: f64,
    pub unrealized_pnl: f64,
    pub stop_loss: f64,
    pub take_profit: Option<f64>,
}
