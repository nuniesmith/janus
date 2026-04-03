//! # Database Models
//!
//! Domain models mapped to database tables using SQLx.
//!
//! These models represent the database schema and include serialization
//! to/from domain types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::FromRow;
use uuid::Uuid;

// ===== Signal Models =====

/// Database model for trading signals
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct SignalRecord {
    pub signal_id: Uuid,
    pub symbol: String,
    pub signal_type: String,
    pub timeframe: String,
    pub confidence: f64,
    pub strength: f64,
    pub timestamp: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub position_size: Option<f64>,
    pub risk_amount: Option<f64>,
    pub risk_reward_ratio: Option<f64>,
    pub source_type: String,
    pub source_name: Option<String>,
    pub strategy_name: Option<String>,
    pub strategy_score: Option<f64>,
    pub model_name: Option<String>,
    pub model_version: Option<String>,
    pub model_confidence: Option<f64>,
    pub indicators: Option<JsonValue>,
    pub metadata: Option<JsonValue>,
    pub status: String,
    pub executed_at: Option<DateTime<Utc>>,
    pub execution_price: Option<f64>,
    pub closed_at: Option<DateTime<Utc>>,
    pub close_price: Option<f64>,
    pub pnl: Option<f64>,
    pub pnl_percentage: Option<f64>,
    pub filtered: bool,
    pub is_backtest: bool,
}

/// Signal insert model (subset of fields for creation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewSignal {
    pub signal_id: Uuid,
    pub symbol: String,
    pub signal_type: String,
    pub timeframe: String,
    pub confidence: f64,
    pub strength: f64,
    pub timestamp: DateTime<Utc>,
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub position_size: Option<f64>,
    pub risk_amount: Option<f64>,
    pub risk_reward_ratio: Option<f64>,
    pub source_type: String,
    pub source_name: Option<String>,
    pub strategy_name: Option<String>,
    pub strategy_score: Option<f64>,
    pub model_name: Option<String>,
    pub model_version: Option<String>,
    pub model_confidence: Option<f64>,
    pub indicators: Option<JsonValue>,
    pub metadata: Option<JsonValue>,
    pub filtered: bool,
    pub is_backtest: bool,
}

// ===== Portfolio Models =====

/// Database model for portfolios
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct PortfolioRecord {
    pub portfolio_id: Uuid,
    pub name: String,
    pub account_id: String,
    pub initial_balance: f64,
    pub current_balance: f64,
    pub total_pnl: f64,
    pub total_pnl_percentage: f64,
    pub daily_pnl: f64,
    pub daily_pnl_percentage: f64,
    pub max_drawdown: f64,
    pub current_drawdown: f64,
    pub sharpe_ratio: Option<f64>,
    pub total_exposure: f64,
    pub exposure_percentage: f64,
    pub active_positions: i32,
    pub total_positions_opened: i32,
    pub winning_positions: i32,
    pub losing_positions: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_reset_at: Option<DateTime<Utc>>,
    pub status: String,
    pub risk_config: Option<JsonValue>,
}

/// New portfolio model for creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewPortfolio {
    pub name: String,
    pub account_id: String,
    pub initial_balance: f64,
    pub risk_config: Option<JsonValue>,
}

// ===== Position Models =====

/// Database model for positions
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct PositionRecord {
    pub position_id: Uuid,
    pub portfolio_id: Uuid,
    pub signal_id: Option<Uuid>,
    pub symbol: String,
    pub side: String,
    pub entry_price: f64,
    pub quantity: f64,
    pub position_value: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub risk_amount: Option<f64>,
    pub risk_percentage: Option<f64>,
    pub risk_reward_ratio: Option<f64>,
    pub current_price: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    pub unrealized_pnl_percentage: Option<f64>,
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
    pub exit_price: Option<f64>,
    pub realized_pnl: Option<f64>,
    pub realized_pnl_percentage: Option<f64>,
    pub exit_reason: Option<String>,
    pub status: String,
    pub metadata: Option<JsonValue>,
    pub notes: Option<String>,
}

/// New position model for creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewPosition {
    pub portfolio_id: Uuid,
    pub signal_id: Option<Uuid>,
    pub symbol: String,
    pub side: String,
    pub entry_price: f64,
    pub quantity: f64,
    pub position_value: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub risk_amount: Option<f64>,
    pub risk_percentage: Option<f64>,
    pub risk_reward_ratio: Option<f64>,
    pub metadata: Option<JsonValue>,
    pub notes: Option<String>,
}

/// Position update model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    pub current_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    pub unrealized_pnl_percentage: Option<f64>,
}

/// Position close model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosePosition {
    pub exit_price: f64,
    pub realized_pnl: f64,
    pub realized_pnl_percentage: f64,
    pub exit_reason: String,
}

// ===== Position Update History =====

/// Database model for position updates (audit trail)
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct PositionUpdateRecord {
    pub update_id: i64,
    pub position_id: Uuid,
    pub update_type: String,
    pub price: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    pub unrealized_pnl_percentage: Option<f64>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Option<JsonValue>,
}

// ===== Portfolio Snapshot Models =====

/// Database model for portfolio snapshots
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct PortfolioSnapshotRecord {
    pub snapshot_id: i64,
    pub portfolio_id: Uuid,
    pub snapshot_date: chrono::NaiveDate,
    pub snapshot_time: DateTime<Utc>,
    pub balance: f64,
    pub equity: f64,
    pub daily_pnl: Option<f64>,
    pub daily_pnl_percentage: Option<f64>,
    pub cumulative_pnl: Option<f64>,
    pub cumulative_pnl_percentage: Option<f64>,
    pub exposure: Option<f64>,
    pub exposure_percentage: Option<f64>,
    pub drawdown: Option<f64>,
    pub active_positions: Option<i32>,
    pub metadata: Option<JsonValue>,
}

// ===== Trade Metrics Models =====

/// Database model for trade metrics
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct TradeMetricRecord {
    pub metric_id: i64,
    pub position_id: Uuid,
    pub signal_id: Option<Uuid>,
    pub portfolio_id: Uuid,
    pub symbol: String,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub initial_risk: Option<f64>,
    pub actual_risk: Option<f64>,
    pub risk_reward_ratio: Option<f64>,
    pub gross_pnl: f64,
    pub gross_pnl_percentage: f64,
    pub net_pnl: Option<f64>,
    pub net_pnl_percentage: Option<f64>,
    pub commission: f64,
    pub slippage: f64,
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub holding_duration_seconds: Option<i64>,
    pub holding_duration_bars: Option<i32>,
    pub exit_reason: Option<String>,
    pub hit_stop_loss: bool,
    pub hit_take_profit: bool,
    pub entry_volatility: Option<f64>,
    pub entry_atr: Option<f64>,
    pub entry_rsi: Option<f64>,
    pub exit_volatility: Option<f64>,
    pub exit_atr: Option<f64>,
    pub exit_rsi: Option<f64>,
    pub max_favorable_excursion: Option<f64>,
    pub max_favorable_excursion_pct: Option<f64>,
    pub max_adverse_excursion: Option<f64>,
    pub max_adverse_excursion_pct: Option<f64>,
    pub metadata: Option<JsonValue>,
    pub created_at: DateTime<Utc>,
}

// ===== Performance Statistics Models =====

/// Database model for performance statistics
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct PerformanceStatsRecord {
    pub stat_id: i64,
    pub portfolio_id: Uuid,
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_trades: i32,
    pub winning_trades: i32,
    pub losing_trades: i32,
    pub breakeven_trades: i32,
    pub win_rate: Option<f64>,
    pub loss_rate: Option<f64>,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub net_profit: f64,
    pub avg_win: Option<f64>,
    pub avg_loss: Option<f64>,
    pub avg_win_pct: Option<f64>,
    pub avg_loss_pct: Option<f64>,
    pub largest_win: Option<f64>,
    pub largest_loss: Option<f64>,
    pub largest_win_pct: Option<f64>,
    pub largest_loss_pct: Option<f64>,
    pub max_consecutive_wins: Option<i32>,
    pub max_consecutive_losses: Option<i32>,
    pub current_streak: Option<i32>,
    pub current_streak_type: Option<String>,
    pub profit_factor: Option<f64>,
    pub expectancy: Option<f64>,
    pub expectancy_pct: Option<f64>,
    pub sharpe_ratio: Option<f64>,
    pub sortino_ratio: Option<f64>,
    pub calmar_ratio: Option<f64>,
    pub max_drawdown: Option<f64>,
    pub max_drawdown_duration_days: Option<i32>,
    pub avg_drawdown: Option<f64>,
    pub recovery_factor: Option<f64>,
    pub avg_r_multiple: Option<f64>,
    pub avg_holding_duration_seconds: Option<i64>,
    pub median_holding_duration_seconds: Option<i64>,
    pub total_volume: Option<f64>,
    pub total_commission: Option<f64>,
    pub metadata: Option<JsonValue>,
    pub calculated_at: DateTime<Utc>,
}

// ===== Risk Metrics Models =====

/// Database model for risk metrics
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct RiskMetricRecord {
    pub risk_id: i64,
    pub portfolio_id: Uuid,
    pub measured_at: DateTime<Utc>,
    pub portfolio_value: f64,
    pub cash_balance: f64,
    pub total_exposure: Option<f64>,
    pub exposure_percentage: Option<f64>,
    pub gross_exposure: Option<f64>,
    pub net_exposure: Option<f64>,
    pub active_positions: i32,
    pub max_position_size: Option<f64>,
    pub avg_position_size: Option<f64>,
    pub largest_position_pct: Option<f64>,
    pub top_5_concentration_pct: Option<f64>,
    pub avg_correlation: Option<f64>,
    pub max_correlation: Option<f64>,
    pub portfolio_heat: Option<f64>,
    pub total_risk_amount: Option<f64>,
    pub avg_risk_per_position: Option<f64>,
    pub max_risk_per_position: Option<f64>,
    pub var_95: Option<f64>,
    pub var_99: Option<f64>,
    pub cvar_95: Option<f64>,
    pub portfolio_volatility: Option<f64>,
    pub realized_volatility: Option<f64>,
    pub current_drawdown: Option<f64>,
    pub drawdown_from_peak: Option<f64>,
    pub peak_portfolio_value: Option<f64>,
    pub limits_exceeded: Option<JsonValue>,
    pub metadata: Option<JsonValue>,
}

// ===== Signal Performance Models =====

/// Database model for signal performance tracking
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct SignalPerformanceRecord {
    pub performance_id: i64,
    pub signal_id: Uuid,
    pub position_id: Option<Uuid>,
    pub symbol: String,
    pub signal_type: String,
    pub timeframe: String,
    pub signal_confidence: Option<f64>,
    pub signal_strength: Option<f64>,
    pub source_type: Option<String>,
    pub strategy_name: Option<String>,
    pub model_name: Option<String>,
    pub was_executed: bool,
    pub execution_delay_seconds: Option<i32>,
    pub actual_entry_price: Option<f64>,
    pub slippage: Option<f64>,
    pub slippage_pct: Option<f64>,
    pub outcome: Option<String>,
    pub pnl: Option<f64>,
    pub pnl_percentage: Option<f64>,
    pub entry_accuracy: Option<f64>,
    pub stop_hit: Option<bool>,
    pub target_hit: Option<bool>,
    pub signal_timestamp: DateTime<Utc>,
    pub execution_timestamp: Option<DateTime<Utc>>,
    pub exit_timestamp: Option<DateTime<Utc>>,
    pub metadata: Option<JsonValue>,
    pub created_at: DateTime<Utc>,
}

// ===== Strategy Performance Models =====

/// Database model for strategy performance
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct StrategyPerformanceRecord {
    pub strategy_perf_id: i64,
    pub strategy_name: String,
    pub portfolio_id: Option<Uuid>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub timeframe: Option<String>,
    pub total_signals: i32,
    pub signals_executed: i32,
    pub signals_filtered: i32,
    pub execution_rate: Option<f64>,
    pub total_trades: i32,
    pub winning_trades: i32,
    pub losing_trades: i32,
    pub win_rate: Option<f64>,
    pub total_pnl: f64,
    pub avg_pnl: Option<f64>,
    pub profit_factor: Option<f64>,
    pub avg_confidence: Option<f64>,
    pub avg_strength: Option<f64>,
    pub sharpe_ratio: Option<f64>,
    pub metadata: Option<JsonValue>,
    pub calculated_at: DateTime<Utc>,
}

// ===== Query Result Types =====

/// Aggregated signal statistics
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct SignalStats {
    pub total_count: i64,
    pub buy_count: i64,
    pub sell_count: i64,
    pub hold_count: i64,
    pub avg_confidence: Option<f64>,
    pub avg_strength: Option<f64>,
    pub filtered_count: i64,
    pub filter_rate: Option<f64>,
}

/// Portfolio summary
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub portfolio_id: Uuid,
    pub name: String,
    pub current_balance: f64,
    pub total_pnl: f64,
    pub total_pnl_percentage: f64,
    pub active_positions: i32,
    pub total_exposure: f64,
    pub win_rate: Option<f64>,
    pub sharpe_ratio: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_signal_serialization() {
        let signal = NewSignal {
            signal_id: Uuid::new_v4(),
            symbol: "BTC/USD".to_string(),
            signal_type: "Buy".to_string(),
            timeframe: "1h".to_string(),
            confidence: 0.85,
            strength: 0.75,
            timestamp: Utc::now(),
            entry_price: Some(50000.0),
            stop_loss: Some(49000.0),
            take_profit: Some(52000.0),
            position_size: Some(0.1),
            risk_amount: Some(100.0),
            risk_reward_ratio: Some(2.0),
            source_type: "technical".to_string(),
            source_name: Some("EMA_Cross".to_string()),
            strategy_name: None,
            strategy_score: None,
            model_name: None,
            model_version: None,
            model_confidence: None,
            indicators: None,
            metadata: None,
            filtered: false,
            is_backtest: false,
        };

        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("BTC/USD"));
        assert!(json.contains("Buy"));
    }

    #[test]
    fn test_new_portfolio_creation() {
        let portfolio = NewPortfolio {
            name: "Test Portfolio".to_string(),
            account_id: "ACC123".to_string(),
            initial_balance: 10000.0,
            risk_config: None,
        };

        assert_eq!(portfolio.name, "Test Portfolio");
        assert_eq!(portfolio.initial_balance, 10000.0);
    }

    #[test]
    fn test_position_update() {
        let update = PositionUpdate {
            current_price: Some(51000.0),
            stop_loss: Some(49500.0),
            take_profit: Some(53000.0),
            unrealized_pnl: Some(100.0),
            unrealized_pnl_percentage: Some(2.0),
        };

        assert_eq!(update.current_price, Some(51000.0));
        assert_eq!(update.unrealized_pnl, Some(100.0));
    }
}
