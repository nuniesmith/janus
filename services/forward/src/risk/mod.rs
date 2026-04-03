//! # Risk Management Module
//!
//! Comprehensive risk management for trading signals including:
//! - Position sizing (fixed fractional, Kelly criterion, volatility-based)
//! - Stop loss and take profit calculation (ATR-based, percentage, support/resistance)
//! - Risk limits and validation (position limits, exposure limits, portfolio heat)
//! - Risk metrics (R/R ratio, expected value, Sharpe estimation)

pub mod limits;
pub mod metrics;
pub mod position_sizing;
pub mod stops;

use crate::signal::types::TradingSignal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use limits::{RiskLimits, RiskValidator};
pub use metrics::{PortfolioRisk, RiskMetrics};
pub use position_sizing::{PositionSize, PositionSizer, SizingMethod};
pub use stops::{StopLossCalculator, StopLossMethod, TakeProfitCalculator};

/// Comprehensive risk configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Account balance/equity
    pub account_balance: f64,

    /// Risk percentage per trade (0.01 = 1%)
    pub risk_per_trade_pct: f64,

    /// Maximum position size as percentage of account (0.1 = 10%)
    pub max_position_size_pct: f64,

    /// Maximum portfolio exposure (0.5 = 50%)
    pub max_portfolio_exposure_pct: f64,

    /// Minimum risk/reward ratio
    pub min_risk_reward_ratio: f64,

    /// Default stop loss method
    pub default_stop_method: StopLossMethod,

    /// Default position sizing method
    pub default_sizing_method: SizingMethod,

    /// Maximum concurrent positions
    pub max_concurrent_positions: usize,

    /// Maximum daily loss (as percentage of account)
    pub max_daily_loss_pct: f64,

    /// Per-symbol exposure limit (percentage of account)
    pub per_symbol_exposure_pct: f64,

    /// Enable correlation-based risk checks
    pub check_correlation: bool,

    /// ATR multiplier for stop loss (e.g., 2.0 = 2x ATR)
    pub atr_stop_multiplier: f64,

    /// ATR multiplier for take profit (e.g., 3.0 = 3x ATR)
    pub atr_tp_multiplier: f64,

    /// Default risk/reward ratio for TP calculation
    pub default_risk_reward: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            account_balance: 10000.0,
            risk_per_trade_pct: 0.01,        // 1% risk per trade
            max_position_size_pct: 0.1,      // 10% max position size
            max_portfolio_exposure_pct: 0.5, // 50% max exposure
            min_risk_reward_ratio: 1.5,      // Minimum 1.5:1 R/R
            default_stop_method: StopLossMethod::Atr { multiplier: 2.0 },
            default_sizing_method: SizingMethod::FixedFractional,
            max_concurrent_positions: 5,
            max_daily_loss_pct: 0.05,     // 5% max daily loss
            per_symbol_exposure_pct: 0.2, // 20% per symbol
            check_correlation: true,
            atr_stop_multiplier: 2.0,
            atr_tp_multiplier: 3.0,
            default_risk_reward: 2.0, // Default 2:1 R/R
        }
    }
}

/// Market data needed for risk calculations
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Current price
    pub current_price: f64,

    /// Average True Range (ATR)
    pub atr: Option<f64>,

    /// Support level
    pub support: Option<f64>,

    /// Resistance level
    pub resistance: Option<f64>,

    /// Recent volatility (standard deviation)
    pub volatility: Option<f64>,

    /// Recent high
    pub recent_high: Option<f64>,

    /// Recent low
    pub recent_low: Option<f64>,
}

impl MarketData {
    pub fn new(current_price: f64) -> Self {
        Self {
            current_price,
            atr: None,
            support: None,
            resistance: None,
            volatility: None,
            recent_high: None,
            recent_low: None,
        }
    }

    pub fn with_atr(mut self, atr: f64) -> Self {
        self.atr = Some(atr);
        self
    }

    pub fn with_support(mut self, support: f64) -> Self {
        self.support = Some(support);
        self
    }

    pub fn with_resistance(mut self, resistance: f64) -> Self {
        self.resistance = Some(resistance);
        self
    }

    pub fn with_volatility(mut self, volatility: f64) -> Self {
        self.volatility = Some(volatility);
        self
    }

    pub fn with_range(mut self, high: f64, low: f64) -> Self {
        self.recent_high = Some(high);
        self.recent_low = Some(low);
        self
    }
}

/// Current portfolio state for risk checks
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    /// Current open positions by symbol
    pub positions: HashMap<String, Position>,

    /// Daily profit/loss
    pub daily_pnl: f64,

    /// Total portfolio value
    pub total_value: f64,
}

impl PortfolioState {
    pub fn new(total_value: f64) -> Self {
        Self {
            positions: HashMap::new(),
            daily_pnl: 0.0,
            total_value,
        }
    }

    pub fn add_position(&mut self, symbol: String, position: Position) {
        self.positions.insert(symbol, position);
    }

    pub fn total_exposure(&self) -> f64 {
        self.positions.values().map(|p| p.position_value()).sum()
    }

    pub fn position_count(&self) -> usize {
        self.positions.len()
    }

    pub fn exposure_for_symbol(&self, symbol: &str) -> f64 {
        self.positions
            .get(symbol)
            .map(|p| p.position_value())
            .unwrap_or(0.0)
    }
}

/// Represents an open position
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub entry_price: f64,
    pub quantity: f64,
    pub side: PositionSide,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
}

impl Position {
    pub fn position_value(&self) -> f64 {
        self.entry_price * self.quantity
    }

    pub fn risk_amount(&self) -> Option<f64> {
        self.stop_loss.map(|sl| {
            let risk_per_unit = (self.entry_price - sl).abs();
            risk_per_unit * self.quantity
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

/// Main risk manager that coordinates all risk management components
pub struct RiskManager {
    config: RiskConfig,
    position_sizer: PositionSizer,
    stop_calculator: StopLossCalculator,
    tp_calculator: TakeProfitCalculator,
    validator: RiskValidator,
    /// Optional param query handle for per-asset risk config overrides
    param_query_handle: Option<crate::ParamQueryHandle>,
}

impl RiskManager {
    pub fn new(config: RiskConfig) -> Self {
        let limits = RiskLimits::from_config(&config);

        Self {
            position_sizer: PositionSizer::new(config.clone()),
            stop_calculator: StopLossCalculator::new(config.clone()),
            tp_calculator: TakeProfitCalculator::new(config.clone()),
            validator: RiskValidator::new(limits),
            config,
            param_query_handle: None,
        }
    }

    /// Set the parameter query handle for per-asset config overrides
    ///
    /// When set, the risk manager can query optimized risk parameters
    /// per-asset (e.g., ATR multiplier, take profit %, max position size).
    pub fn set_param_query_handle(&mut self, handle: crate::ParamQueryHandle) {
        tracing::info!("Risk manager: param query handle attached");
        self.param_query_handle = Some(handle);
    }

    /// Check if trading is enabled for an asset via optimized params
    pub async fn is_trading_enabled(&self, asset: &str) -> bool {
        if let Some(ref handle) = self.param_query_handle {
            handle.is_trading_enabled(asset).await
        } else {
            true // Default: allow if no param handle
        }
    }

    /// Get the max position size for an asset (uses optimized params if available)
    pub async fn get_max_position_size_for_asset(&self, asset: &str) -> f64 {
        if let Some(ref handle) = self.param_query_handle
            && let Some(max_size) = handle.get_max_position_size(asset).await
        {
            return max_size;
        }
        // Fall back to default config
        self.config.account_balance * self.config.max_position_size_pct
    }

    /// Get the per-asset risk config if available
    pub async fn get_asset_risk_config(
        &self,
        asset: &str,
    ) -> Option<crate::param_reload::AssetRiskConfig> {
        if let Some(ref handle) = self.param_query_handle {
            handle.get_asset_risk_config(asset).await
        } else {
            None
        }
    }

    /// Apply risk management to a trading signal
    pub fn apply_risk_management(
        &self,
        mut signal: TradingSignal,
        market_data: &MarketData,
        portfolio: &PortfolioState,
    ) -> Result<TradingSignal, RiskError> {
        // Validate signal against risk limits
        self.validator.validate_signal(&signal, portfolio)?;

        // Calculate stop loss if not already set
        if signal.stop_loss.is_none() {
            let stop = self.stop_calculator.calculate_stop_loss(
                &signal,
                market_data,
                &self.config.default_stop_method,
            )?;
            signal = signal.with_stop_loss(stop);
        }

        // Calculate take profit if not already set
        if signal.take_profit.is_none() {
            let tp = self
                .tp_calculator
                .calculate_take_profit(&signal, market_data)?;
            signal = signal.with_take_profit(tp);
        }

        // Verify minimum R/R ratio
        if let Some(rr) = signal.risk_reward_ratio()
            && rr < self.config.min_risk_reward_ratio
        {
            return Err(RiskError::InsufficientRiskReward {
                actual: rr,
                required: self.config.min_risk_reward_ratio,
            });
        }

        // Calculate position size
        let position_size = self.position_sizer.calculate_position_size(
            &signal,
            market_data,
            &self.config.default_sizing_method,
        )?;

        // Add risk metrics to signal metadata
        signal = self.add_risk_metadata(signal, &position_size, market_data);

        Ok(signal)
    }

    /// Calculate position size for a signal
    pub fn calculate_position_size(
        &self,
        signal: &TradingSignal,
        market_data: &MarketData,
        method: &SizingMethod,
    ) -> Result<PositionSize, RiskError> {
        self.position_sizer
            .calculate_position_size(signal, market_data, method)
    }

    /// Validate if a signal meets risk requirements
    pub fn validate_signal(
        &self,
        signal: &TradingSignal,
        portfolio: &PortfolioState,
    ) -> Result<(), RiskError> {
        self.validator.validate_signal(signal, portfolio)
    }

    /// Calculate portfolio risk metrics
    pub fn calculate_portfolio_risk(&self, portfolio: &PortfolioState) -> PortfolioRisk {
        RiskMetrics::calculate_portfolio_risk(portfolio, self.config.account_balance)
    }

    /// Add risk-related metadata to signal
    fn add_risk_metadata(
        &self,
        mut signal: TradingSignal,
        position_size: &PositionSize,
        market_data: &MarketData,
    ) -> TradingSignal {
        signal = signal.with_metadata(
            "position_size_units".to_string(),
            position_size.quantity.to_string(),
        );
        signal = signal.with_metadata(
            "position_value".to_string(),
            position_size.position_value.to_string(),
        );
        signal = signal.with_metadata(
            "risk_amount".to_string(),
            position_size.risk_amount.to_string(),
        );
        signal = signal.with_metadata(
            "risk_percentage".to_string(),
            format!("{:.2}%", position_size.risk_percentage * 100.0),
        );

        if let Some(rr) = signal.risk_reward_ratio() {
            signal = signal.with_metadata("risk_reward_ratio".to_string(), format!("{:.2}", rr));
        }

        if let Some(atr) = market_data.atr {
            signal = signal.with_metadata("atr".to_string(), atr.to_string());
        }

        signal
    }

    /// Apply risk management with per-asset config overrides
    ///
    /// This method checks for optimized risk parameters for the given asset
    /// and applies them if available. Falls back to default config if not.
    pub async fn apply_risk_management_with_asset_config(
        &self,
        signal: TradingSignal,
        market_data: &MarketData,
        portfolio: &PortfolioState,
    ) -> Result<TradingSignal, RiskError> {
        // Extract asset from symbol (e.g., "BTC/USD" -> "BTC")
        let asset = signal.symbol.split('/').next().unwrap_or(&signal.symbol);

        // Check if trading is enabled for this asset
        if !self.is_trading_enabled(asset).await {
            return Err(RiskError::InvalidConfiguration {
                reason: format!("Trading disabled for asset {} via optimized params", asset),
            });
        }

        // Check position size against per-asset limit
        let max_position_size = self.get_max_position_size_for_asset(asset).await;
        let current_exposure = portfolio.exposure_for_symbol(&signal.symbol);
        if current_exposure >= max_position_size {
            return Err(RiskError::PositionSizeLimitExceeded {
                actual: current_exposure,
                max: max_position_size,
            });
        }

        // If we have per-asset risk config, we could potentially override
        // stop loss / take profit calculations here in the future
        // For now, delegate to the standard risk management
        self.apply_risk_management(signal, market_data, portfolio)
    }

    /// Update risk configuration
    pub fn update_config(&mut self, config: RiskConfig) {
        let limits = RiskLimits::from_config(&config);
        self.validator = RiskValidator::new(limits);
        self.position_sizer = PositionSizer::new(config.clone());
        self.stop_calculator = StopLossCalculator::new(config.clone());
        self.tp_calculator = TakeProfitCalculator::new(config.clone());
        self.config = config;
    }

    /// Check if param reload is configured
    pub fn has_param_reload(&self) -> bool {
        self.param_query_handle.is_some()
    }

    /// Get current risk configuration
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }
}

/// Risk management errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum RiskError {
    #[error("Position size exceeds maximum: {actual} > {max}")]
    PositionSizeLimitExceeded { actual: f64, max: f64 },

    #[error("Portfolio exposure exceeds limit: {actual} > {max}")]
    PortfolioExposureLimitExceeded { actual: f64, max: f64 },

    #[error("Maximum concurrent positions reached: {max}")]
    MaxPositionsReached { max: usize },

    #[error("Daily loss limit reached: {actual} >= {max}")]
    DailyLossLimitReached { actual: f64, max: f64 },

    #[error("Symbol exposure limit exceeded for {symbol}: {actual} > {max}")]
    SymbolExposureLimitExceeded {
        symbol: String,
        actual: f64,
        max: f64,
    },

    #[error("Insufficient risk/reward ratio: {actual:.2} < {required:.2}")]
    InsufficientRiskReward { actual: f64, required: f64 },

    #[error("Invalid stop loss: {reason}")]
    InvalidStopLoss { reason: String },

    #[error("Invalid take profit: {reason}")]
    InvalidTakeProfit { reason: String },

    #[error("Missing required data: {field}")]
    MissingData { field: String },

    #[error("Invalid configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Calculation error: {reason}")]
    CalculationError { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::types::{SignalSource, SignalType, Timeframe};

    #[test]
    fn test_risk_config_default() {
        let config = RiskConfig::default();
        assert_eq!(config.account_balance, 10000.0);
        assert_eq!(config.risk_per_trade_pct, 0.01);
        assert_eq!(config.min_risk_reward_ratio, 1.5);
    }

    #[test]
    fn test_market_data_builder() {
        let data = MarketData::new(50000.0)
            .with_atr(500.0)
            .with_support(49000.0)
            .with_resistance(51000.0);

        assert_eq!(data.current_price, 50000.0);
        assert_eq!(data.atr, Some(500.0));
        assert_eq!(data.support, Some(49000.0));
        assert_eq!(data.resistance, Some(51000.0));
    }

    #[test]
    fn test_portfolio_state() {
        let mut portfolio = PortfolioState::new(10000.0);

        portfolio.add_position(
            "BTC/USD".to_string(),
            Position {
                symbol: "BTC/USD".to_string(),
                entry_price: 50000.0,
                quantity: 0.1,
                side: PositionSide::Long,
                stop_loss: Some(49000.0),
                take_profit: Some(52000.0),
            },
        );

        assert_eq!(portfolio.position_count(), 1);
        assert_eq!(portfolio.total_exposure(), 5000.0);
        assert_eq!(portfolio.exposure_for_symbol("BTC/USD"), 5000.0);
    }

    #[test]
    fn test_position_risk_amount() {
        let position = Position {
            symbol: "BTC/USD".to_string(),
            entry_price: 50000.0,
            quantity: 0.1,
            side: PositionSide::Long,
            stop_loss: Some(49000.0),
            take_profit: Some(52000.0),
        };

        let risk = position.risk_amount().unwrap();
        assert_eq!(risk, 100.0); // (50000 - 49000) * 0.1
    }

    #[test]
    fn test_risk_manager_creation() {
        let config = RiskConfig::default();
        let manager = RiskManager::new(config.clone());
        assert_eq!(manager.config().account_balance, 10000.0);
    }

    #[test]
    fn test_risk_manager_apply() {
        let config = RiskConfig::default();
        let manager = RiskManager::new(config);

        let signal = TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Buy,
            Timeframe::H1,
            0.8,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        )
        .with_entry_price(50000.0);

        let market_data = MarketData::new(50000.0).with_atr(500.0);
        let portfolio = PortfolioState::new(10000.0);

        let result = manager.apply_risk_management(signal, &market_data, &portfolio);
        if let Err(e) = &result {
            eprintln!("Risk management error: {:?}", e);
        }
        assert!(
            result.is_ok(),
            "Failed to apply risk management: {:?}",
            result.err()
        );

        let enhanced_signal = result.unwrap();
        assert!(enhanced_signal.stop_loss.is_some());
        assert!(enhanced_signal.take_profit.is_some());
        assert!(enhanced_signal.metadata.contains_key("position_size_units"));
    }
}
