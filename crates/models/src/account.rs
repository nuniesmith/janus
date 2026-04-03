//! Account and Configuration Management
//!
//! This module handles account state, configuration loading, and application settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub app: AppSettings,
    pub server: ServerSettings,
    pub database: DatabaseSettings,
    pub exchange: ExchangeSettings,
    pub strategy: StrategySettings,
    pub risk: RiskSettings,
    pub discord: DiscordSettings,
    pub monitoring: MonitoringSettings,
    pub backtesting: BacktestSettings,
    pub optimization: OptimizationSettings,
    pub logging: LoggingSettings,
    pub prop_firm: PropFirmSettings,
    pub automation: AutomationSettings,
    pub profit_allocation: ProfitAllocationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub name: String,
    pub version: String,
    pub environment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSettings {
    pub url: String,
    pub max_connections: u32,
    pub pool_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeSettings {
    pub default: String,
    pub testnet: bool,
    pub api_key_env: String,
    pub api_secret_env: String,
    pub bybit: BybitSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitSettings {
    pub api_url: String,
    pub ws_url: String,
    pub recv_window: u64,
    pub rate_limit_per_second: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySettings {
    pub name: String,
    pub fast_ema: usize,
    pub slow_ema: usize,
    pub atr_length: usize,
    pub atr_multiplier: f64,
    pub timeframes: TimeframeSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeframeSettings {
    pub primary: String,
    pub secondary: String,
    pub monitoring: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSettings {
    pub max_risk_per_trade_percent: f64,
    pub daily_drawdown_percent: f64,
    pub max_drawdown_percent: f64,
    pub max_exposure_percent: f64,
    pub default_account_size: f64,
    pub use_kelly_criterion: bool,
    pub kelly_fraction: f64,
    pub stop_loss_required: bool,
    pub stop_loss_atr_multiplier: f64,
    pub trailing_stop_enabled: bool,
    pub take_profit: TakeProfitSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeProfitSettings {
    pub enabled: bool,
    pub tp1_percent: f64,
    pub tp2_percent: f64,
    pub tp3_percent: f64,
    pub tp4_percent: f64,
    pub tp1_rr: f64,
    pub tp2_rr: f64,
    pub tp3_rr: f64,
    pub tp4_rr: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordSettings {
    pub enabled: bool,
    pub webhook_url_env: String,
    pub send_opportunities: bool,
    pub send_entries: bool,
    pub send_exits: bool,
    pub send_daily_summary: bool,
    pub rate_limit_seconds: u64,
    pub colors: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSettings {
    pub symbols: Vec<String>,
    pub scan_interval_seconds: u64,
    pub min_volume_24h: f64,
    pub min_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestSettings {
    pub initial_capital: f64,
    pub commission_percent: f64,
    pub slippage_percent: f64,
    pub start_date: String,
    pub end_date: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    pub n_trials: usize,
    pub timeout_hours: u64,
    pub metric: String,
    pub ranges: OptimizationRanges,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRanges {
    pub fast_ema: Vec<usize>,
    pub slow_ema: Vec<usize>,
    pub atr_multiplier: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    pub level: String,
    pub format: String,
    pub file: String,
    pub console: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropFirmSettings {
    pub provider: String,
    pub config_file: String,
    pub challenge_type: String,
    pub account_size: f64,
    pub validate_all_trades: bool,
    pub auto_reject_non_compliant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationSettings {
    pub enabled: bool,
    pub manual_trading_only: bool,
    pub require_confirmation: bool,
    pub max_concurrent_positions: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitAllocationSettings {
    pub wife_tax_percent: f64,
    pub personal_pay_percent: f64,
    pub crypto_exchange_percent: f64,
    pub hardware_wallet_percent: f64,
    pub expand_accounts_percent: f64,
}

/// HyroTrader prop firm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyroTraderConfig {
    pub platform: String,
    pub challenge_types: HashMap<String, ChallengeTypeConfig>,
    pub account_sizes: HashMap<String, AccountSizeConfig>,
    pub risk_rules: RiskRulesConfig,
    pub trading_requirements: TradingRequirementsConfig,
    pub prohibited_practices: Vec<String>,
    pub funded_account: FundedAccountConfig,
    pub platform_details: PlatformDetailsConfig,
    pub compliance_checks: ComplianceChecksConfig,
    pub wife_tax: WifeTaxConfig,
    pub scaling_strategy: ScalingStrategyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeTypeConfig {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profit_target_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_trading_days: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phases: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification_required: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_1: Option<PhaseConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_2: Option<PhaseConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    pub profit_target_percent: f64,
    pub minimum_trading_days: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSizeConfig {
    pub initial_balance: f64,
    pub one_step_fee: f64,
    pub two_step_fee: f64,
    pub max_daily_loss: f64,
    pub max_drawdown: f64,
    pub profit_target_one_step: f64,
    pub profit_target_two_step_phase1: f64,
    pub profit_target_two_step_phase2: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskRulesConfig {
    pub daily_drawdown_percent: f64,
    pub max_drawdown_percent: f64,
    pub max_risk_per_trade_percent: f64,
    pub max_exposure_percent: f64,
    pub cumulative_position_limit_multiplier: f64,
    pub stop_loss_required: bool,
    pub stop_loss_setup_time_minutes: i64,
    pub stop_loss_grace_period_minutes: i64,
    pub soft_breach_allowed: bool,
    pub soft_breach_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingRequirementsConfig {
    pub minimum_trade_value_percent: f64,
    pub minimum_pnl_percent: f64,
    pub stop_loss_order_types: Vec<String>,
    pub invalid_order_types: Vec<String>,
    pub required_margin_modes: Vec<String>,
    pub forbidden_margin_modes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundedAccountConfig {
    pub profit_split_initial_percent: f64,
    pub profit_split_max_percent: f64,
    pub minimum_payout_usd: f64,
    pub payout_processing_hours: u64,
    pub payout_currency: String,
    pub scaling_months_required: u64,
    pub fee_refund_on_first_payout: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformDetailsConfig {
    pub exchange: String,
    pub alternative_exchange: String,
    pub contract_type: String,
    pub max_leverage: u32,
    pub data_source: String,
    pub time_zone: String,
    pub daily_reset_hour: u32,
    pub support_24_7: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceChecksConfig {
    pub check_stop_loss_timing: bool,
    pub check_risk_per_trade: bool,
    pub check_daily_drawdown: bool,
    pub check_max_drawdown: bool,
    pub check_minimum_trading_days: bool,
    pub check_minimum_trade_size: bool,
    pub check_prohibited_pairs: bool,
    pub check_margin_mode: bool,
    pub validate_order_types: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WifeTaxConfig {
    pub enabled: bool,
    pub percentage: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingStrategyConfig {
    pub start_account_size: f64,
    pub target_account_size: f64,
    pub number_of_accounts_target: i32,
    pub personal_crypto_accounts: PersonalCryptoAccountsConfig,
    pub profit_allocation: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalCryptoAccountsConfig {
    pub exchanges: Vec<String>,
    pub initial_funding_per_exchange: f64,
    pub target_per_exchange: f64,
}

/// Configuration loader
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load main application configuration
    pub fn load_app_config<P: AsRef<Path>>(path: P) -> anyhow::Result<AppConfig> {
        let content = fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load HyroTrader configuration
    pub fn load_hyrotrader_config<P: AsRef<Path>>(path: P) -> anyhow::Result<HyroTraderConfig> {
        let content = fs::read_to_string(path)?;
        let config: HyroTraderConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from default paths
    pub fn load_default() -> anyhow::Result<(AppConfig, HyroTraderConfig)> {
        let app_config = Self::load_app_config("config/config.toml")?;
        let hyro_config = Self::load_hyrotrader_config("config/hyrotrader.json")?;
        Ok((app_config, hyro_config))
    }
}

/// Account state tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountState {
    pub account_id: String,
    pub account_size: f64,
    pub current_balance: f64,
    pub available_balance: f64,
    pub margin_used: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub open_positions: Vec<String>,
    pub daily_pnl: f64,
    pub weekly_pnl: f64,
    pub monthly_pnl: f64,
}

impl AccountState {
    pub fn new(account_size: f64) -> Self {
        Self {
            account_id: uuid::Uuid::new_v4().to_string(),
            account_size,
            current_balance: account_size,
            available_balance: account_size,
            margin_used: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            open_positions: Vec::new(),
            daily_pnl: 0.0,
            weekly_pnl: 0.0,
            monthly_pnl: 0.0,
        }
    }

    /// Calculate equity (balance + unrealized PnL)
    pub fn equity(&self) -> f64 {
        self.current_balance + self.unrealized_pnl
    }

    /// Calculate used margin percentage
    pub fn margin_usage_percent(&self) -> f64 {
        (self.margin_used / self.current_balance) * 100.0
    }

    /// Calculate daily PnL percentage
    pub fn daily_pnl_percent(&self) -> f64 {
        (self.daily_pnl / self.account_size) * 100.0
    }

    /// Update balance after trade
    pub fn update_balance(&mut self, pnl: f64) {
        self.current_balance += pnl;
        self.realized_pnl += pnl;
        self.daily_pnl += pnl;
        self.available_balance = self.current_balance - self.margin_used;
    }

    /// Reset daily statistics
    pub fn reset_daily_stats(&mut self) {
        self.daily_pnl = 0.0;
    }
}
