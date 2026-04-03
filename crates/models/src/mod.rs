//! Data models for trades, signals, and market data
//!
//! This module contains all the core data structures used throughout
//! the trading system, including trades, signals, account state, and
//! performance metrics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

pub mod account;
pub mod performance;
pub mod prop_firm;

/// Trading direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
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

/// Trade execution and tracking
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Trade {
    pub id: i64,
    pub trade_id: String,
    pub signal_id: Option<String>,

    // Basic info
    pub symbol: String,
    pub direction: Direction,
    pub status: TradeStatus,

    // Entry details
    pub entry_time: DateTime<Utc>,
    pub entry_price: f64,
    pub position_size: f64, // USD value
    pub leverage: i32,

    // Exit details
    pub exit_time: Option<DateTime<Utc>>,
    pub exit_price: Option<f64>,
    pub exit_reason: Option<String>,

    // Risk management
    pub initial_stop_loss: f64,
    pub current_stop_loss: Option<f64>,
    pub take_profit_1: Option<f64>,
    pub take_profit_2: Option<f64>,
    pub take_profit_3: Option<f64>,
    pub take_profit_4: Option<f64>,

    // Performance
    pub pnl_usd: Option<f64>,
    pub pnl_percent: Option<f64>,
    pub risk_reward_ratio: Option<f64>,
    pub commission_paid: Option<f64>,

    // TP/SL tracking
    pub tp1_hit: bool,
    pub tp1_time: Option<DateTime<Utc>>,
    pub tp2_hit: bool,
    pub tp2_time: Option<DateTime<Utc>>,
    pub tp3_hit: bool,
    pub tp3_time: Option<DateTime<Utc>>,
    pub tp4_hit: bool,
    pub tp4_time: Option<DateTime<Utc>>,

    // Partial exits
    pub remaining_position_percent: f64,
    pub total_closed_percent: f64,

    // Metadata
    pub execution_type: String, // manual, auto
    pub platform: Option<String>,
    pub notes: Option<String>,

    // Timestamps
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Trade {
    /// Create a new trade
    pub fn new(
        symbol: String,
        direction: Direction,
        entry_price: f64,
        position_size: f64,
        stop_loss: f64,
        leverage: i32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: 0,
            trade_id: Uuid::new_v4().to_string(),
            signal_id: None,
            symbol,
            direction,
            status: TradeStatus::Open,
            entry_time: now,
            entry_price,
            position_size,
            leverage,
            exit_time: None,
            exit_price: None,
            exit_reason: None,
            initial_stop_loss: stop_loss,
            current_stop_loss: Some(stop_loss),
            take_profit_1: None,
            take_profit_2: None,
            take_profit_3: None,
            take_profit_4: None,
            pnl_usd: None,
            pnl_percent: None,
            risk_reward_ratio: None,
            commission_paid: None,
            tp1_hit: false,
            tp1_time: None,
            tp2_hit: false,
            tp2_time: None,
            tp3_hit: false,
            tp3_time: None,
            tp4_hit: false,
            tp4_time: None,
            remaining_position_percent: 100.0,
            total_closed_percent: 0.0,
            execution_type: "manual".to_string(),
            platform: None,
            notes: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Calculate PnL for this trade
    pub fn calculate_pnl(&self, exit_price: f64) -> (f64, f64) {
        let pnl_percent = match self.direction {
            Direction::Long => ((exit_price - self.entry_price) / self.entry_price) * 100.0,
            Direction::Short => ((self.entry_price - exit_price) / self.entry_price) * 100.0,
        };

        let pnl_usd = (self.position_size * pnl_percent / 100.0) * self.leverage as f64;
        (pnl_usd, pnl_percent)
    }

    /// Close the trade
    pub fn close(&mut self, exit_price: f64, reason: ExitReason) {
        self.exit_time = Some(Utc::now());
        self.exit_price = Some(exit_price);
        self.exit_reason = Some(format!("{:?}", reason));
        self.status = TradeStatus::Closed;

        let (pnl_usd, pnl_percent) = self.calculate_pnl(exit_price);
        self.pnl_usd = Some(pnl_usd);
        self.pnl_percent = Some(pnl_percent);
        self.updated_at = Utc::now();
    }
}

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Opportunity,
    Entry,
    Exit,
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Signal {
    pub id: i64,
    pub signal_id: String,
    pub signal_type: String,

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

    // Status
    pub status: String,
    pub executed: bool,
    pub discord_sent: bool,

    // Compliance
    pub prop_firm_compliant: Option<bool>,
    pub rule_violations: Option<String>,

    // Metadata
    pub reason: Option<String>,
    pub metadata: Option<String>,

    pub created_at: DateTime<Utc>,
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
            id: 0,
            signal_id: Uuid::new_v4().to_string(),
            signal_type: format!("{:?}", signal_type),
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
            status: "pending".to_string(),
            executed: false,
            discord_sent: false,
            prop_firm_compliant: None,
            rule_violations: None,
            reason: None,
            metadata: None,
            created_at: Utc::now(),
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

/// Market data with indicators
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MarketData {
    pub candles: Vec<Candle>,
    pub ema_fast: Vec<f64>,
    pub ema_slow: Vec<f64>,
    pub atr: Vec<f64>,
}

/// Position sizing result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PositionSize {
    pub position_size_usd: f64,
    pub position_size_contracts: f64,
    pub risk_amount: f64,
    pub risk_percent: f64,
    pub stop_loss_distance: f64,
    pub leverage: i32,
}

/// Daily statistics
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
#[allow(dead_code)]
pub struct DailyStats {
    pub id: i64,
    pub date: DateTime<Utc>,

    // Trade counts
    pub total_trades: i32,
    pub winning_trades: i32,
    pub losing_trades: i32,

    // Performance
    pub total_pnl_usd: f64,
    pub total_pnl_percent: f64,
    pub win_rate: Option<f64>,
    pub profit_factor: Option<f64>,

    // Risk metrics
    pub max_drawdown_percent: Option<f64>,
    pub sharpe_ratio: Option<f64>,

    // Account
    pub starting_balance: Option<f64>,
    pub ending_balance: Option<f64>,

    // TP/SL stats
    pub tp1_hits: i32,
    pub tp2_hits: i32,
    pub tp3_hits: i32,
    pub tp4_hits: i32,
    pub sl_hits: i32,

    // Compliance
    pub prop_firm_compliant: bool,
    pub rule_violations: i32,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
