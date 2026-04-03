//! # Parameter Appliers
//!
//! This module provides implementations of the `ParamApplier` trait for various
//! components of the Forward service that need to receive optimized parameters.
//!
//! ## Appliers
//!
//! - `NoOpApplier` - Does nothing (for testing)
//! - `LoggingApplier` - Logs received params (for debugging)
//! - `IndicatorParamApplier` - Updates indicator analyzer configurations
//! - `RiskParamApplier` - Updates risk management parameters
//! - `StrategyParamApplier` - Updates strategy-specific parameters

use super::manager::ParamApplier;
use crate::indicators::{IndicatorAnalyzer, IndicatorConfig};
use crate::risk::RiskManager;
use crate::strategies::StrategyConfig;
use janus_core::optimized_params::OptimizedParams;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Type alias for the complex nested analyzer map
pub type AnalyzerMap = Arc<RwLock<HashMap<String, Arc<RwLock<IndicatorAnalyzer>>>>>;

// ============================================================================
// NoOpApplier - Testing/placeholder
// ============================================================================

/// A no-op param applier for testing
pub struct NoOpApplier {
    name: String,
}

impl NoOpApplier {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait::async_trait]
impl ParamApplier for NoOpApplier {
    async fn apply_params(&self, params: &OptimizedParams) -> anyhow::Result<()> {
        debug!(
            applier = self.name,
            asset = params.asset,
            "NoOp applier received params"
        );
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// LoggingApplier - Debugging/monitoring
// ============================================================================

/// A logging param applier that just logs received params
pub struct LoggingApplier {
    name: String,
}

impl LoggingApplier {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait::async_trait]
impl ParamApplier for LoggingApplier {
    async fn apply_params(&self, params: &OptimizedParams) -> anyhow::Result<()> {
        info!(
            applier = self.name,
            asset = params.asset,
            ema_fast = params.ema_fast_period,
            ema_slow = params.ema_slow_period,
            atr_length = params.atr_length,
            atr_multiplier = params.atr_multiplier,
            min_ema_spread = params.min_ema_spread_pct,
            min_profit = params.min_profit_pct,
            take_profit = params.take_profit_pct,
            cooldown_seconds = params.trade_cooldown_seconds,
            score = params.optimization_score,
            optimized_at = params.optimized_at,
            "Logging applier received optimized params"
        );
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// IndicatorParamApplier - Updates indicator configurations per asset
// ============================================================================

/// Applier for indicator analyzer configurations
///
/// Maintains per-asset indicator configs and updates them when new
/// optimized parameters arrive.
pub struct IndicatorParamApplier {
    /// Per-asset indicator configurations
    configs: Arc<RwLock<HashMap<String, IndicatorConfig>>>,

    /// Per-asset analyzers (optional - if analyzers are shared)
    analyzers: Option<AnalyzerMap>,
}

impl IndicatorParamApplier {
    /// Create a new indicator param applier
    pub fn new() -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
            analyzers: None,
        }
    }

    /// Create with shared analyzers map
    pub fn with_analyzers(analyzers: AnalyzerMap) -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
            analyzers: Some(analyzers),
        }
    }

    /// Get the indicator config for an asset
    pub async fn get_config(&self, asset: &str) -> Option<IndicatorConfig> {
        let configs = self.configs.read().await;
        configs.get(asset).cloned()
    }

    /// Get all current configs
    pub async fn get_all_configs(&self) -> HashMap<String, IndicatorConfig> {
        let configs = self.configs.read().await;
        configs.clone()
    }

    /// Convert OptimizedParams to IndicatorConfig
    fn params_to_indicator_config(params: &OptimizedParams) -> IndicatorConfig {
        IndicatorConfig {
            ema_fast_period: params.ema_fast_period as usize,
            ema_slow_period: params.ema_slow_period as usize,
            atr_period: params.atr_length as usize,
            // Keep defaults for other values not in OptimizedParams
            ..IndicatorConfig::default()
        }
    }
}

impl Default for IndicatorParamApplier {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ParamApplier for IndicatorParamApplier {
    async fn apply_params(&self, params: &OptimizedParams) -> anyhow::Result<()> {
        let asset = &params.asset;
        let new_config = Self::params_to_indicator_config(params);

        // Store the config
        {
            let mut configs = self.configs.write().await;
            let old_config = configs.insert(asset.clone(), new_config.clone());

            if let Some(old) = old_config {
                info!(
                    asset = asset,
                    old_ema_fast = old.ema_fast_period,
                    new_ema_fast = new_config.ema_fast_period,
                    old_ema_slow = old.ema_slow_period,
                    new_ema_slow = new_config.ema_slow_period,
                    "Updated indicator config for asset"
                );
            } else {
                info!(
                    asset = asset,
                    ema_fast = new_config.ema_fast_period,
                    ema_slow = new_config.ema_slow_period,
                    "Applied new indicator config for asset"
                );
            }
        }

        // If we have shared analyzers, recreate the analyzer for this asset
        if let Some(analyzers) = &self.analyzers {
            let mut analyzers_map = analyzers.write().await;

            // Create new analyzer with updated config
            let new_analyzer = IndicatorAnalyzer::new(new_config.clone());
            analyzers_map.insert(asset.clone(), Arc::new(RwLock::new(new_analyzer)));

            debug!(
                asset = asset,
                "Recreated indicator analyzer with new config"
            );
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "IndicatorParamApplier"
    }
}

// ============================================================================
// RiskParamApplier - Updates risk management parameters per asset
// ============================================================================

/// Applier for risk management configurations
///
/// Updates risk parameters like position size, stop loss, take profit
/// based on optimized parameters.
pub struct RiskParamApplier {
    /// Shared risk manager (for future use when update hooks are added)
    #[allow(dead_code)]
    risk_manager: Arc<RwLock<RiskManager>>,

    /// Per-asset risk configurations
    asset_configs: Arc<RwLock<HashMap<String, AssetRiskConfig>>>,
}

/// Asset-specific risk configuration derived from OptimizedParams
#[derive(Debug, Clone)]
pub struct AssetRiskConfig {
    /// Asset symbol
    pub asset: String,

    /// ATR multiplier for stop loss
    pub atr_multiplier: f64,

    /// Minimum trailing stop percentage
    pub min_trailing_stop_pct: f64,

    /// Take profit percentage
    pub take_profit_pct: f64,

    /// Minimum profit to allow exit
    pub min_profit_pct: f64,

    /// Maximum position size in USD
    pub max_position_size_usd: f64,

    /// Trade cooldown in seconds
    pub trade_cooldown_seconds: u64,

    /// Whether trading is enabled for this asset
    pub enabled: bool,
}

impl From<&OptimizedParams> for AssetRiskConfig {
    fn from(params: &OptimizedParams) -> Self {
        Self {
            asset: params.asset.clone(),
            atr_multiplier: params.atr_multiplier,
            min_trailing_stop_pct: params.min_trailing_stop_pct,
            take_profit_pct: params.take_profit_pct,
            min_profit_pct: params.min_profit_pct,
            max_position_size_usd: params.max_position_size_usd,
            trade_cooldown_seconds: params.trade_cooldown_seconds,
            enabled: params.enabled,
        }
    }
}

impl RiskParamApplier {
    /// Create a new risk param applier
    pub fn new(risk_manager: Arc<RwLock<RiskManager>>) -> Self {
        Self {
            risk_manager,
            asset_configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get risk config for an asset
    pub async fn get_asset_config(&self, asset: &str) -> Option<AssetRiskConfig> {
        let configs = self.asset_configs.read().await;
        configs.get(asset).cloned()
    }

    /// Check if trading is enabled for an asset
    pub async fn is_trading_enabled(&self, asset: &str) -> bool {
        let configs = self.asset_configs.read().await;
        configs.get(asset).map(|c| c.enabled).unwrap_or(true)
    }

    /// Get max position size for an asset
    pub async fn get_max_position_size(&self, asset: &str) -> Option<f64> {
        let configs = self.asset_configs.read().await;
        configs.get(asset).map(|c| c.max_position_size_usd)
    }
}

#[async_trait::async_trait]
impl ParamApplier for RiskParamApplier {
    async fn apply_params(&self, params: &OptimizedParams) -> anyhow::Result<()> {
        let asset = &params.asset;
        let risk_config = AssetRiskConfig::from(params);

        // Store asset-specific config
        {
            let mut configs = self.asset_configs.write().await;
            let old = configs.insert(asset.clone(), risk_config.clone());

            if old.is_some() {
                info!(
                    asset = asset,
                    atr_mult = risk_config.atr_multiplier,
                    take_profit = risk_config.take_profit_pct,
                    max_pos = risk_config.max_position_size_usd,
                    enabled = risk_config.enabled,
                    "Updated risk config for asset"
                );
            } else {
                info!(
                    asset = asset,
                    atr_mult = risk_config.atr_multiplier,
                    take_profit = risk_config.take_profit_pct,
                    max_pos = risk_config.max_position_size_usd,
                    enabled = risk_config.enabled,
                    "Applied new risk config for asset"
                );
            }
        }

        // Note: The RiskManager itself may need additional methods to handle
        // per-asset configuration updates. For now, we store the configs
        // and they can be queried when making risk decisions.

        if !params.enabled {
            warn!(
                asset = asset,
                "Trading disabled for asset via optimized params"
            );
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "RiskParamApplier"
    }
}

// ============================================================================
// StrategyParamApplier - Updates strategy parameters per asset
// ============================================================================

/// Applier for strategy configurations
///
/// Updates strategy-specific parameters like EMA spread thresholds,
/// hold times, and other strategy settings.
pub struct StrategyParamApplier {
    /// Per-asset strategy configurations
    configs: Arc<RwLock<HashMap<String, OptimizedStrategyConfig>>>,
}

/// Strategy configuration derived from OptimizedParams
#[derive(Debug, Clone)]
pub struct OptimizedStrategyConfig {
    /// Asset symbol
    pub asset: String,

    /// Minimum EMA spread percentage to confirm trend
    pub min_ema_spread_pct: f64,

    /// Minimum profit percentage to allow exit
    pub min_profit_pct: f64,

    /// Minimum hold time in minutes
    pub min_hold_minutes: u32,

    /// Whether to require higher timeframe alignment
    pub require_htf_alignment: bool,

    /// Higher timeframe for trend filter (minutes)
    pub htf_timeframe_minutes: u32,

    /// Prefer trailing stop exit over EMA reversal
    pub prefer_trailing_stop_exit: bool,

    /// Trade cooldown in seconds
    pub trade_cooldown_seconds: u64,

    /// Base strategy config (shared settings)
    pub base_config: StrategyConfig,
}

impl From<&OptimizedParams> for OptimizedStrategyConfig {
    fn from(params: &OptimizedParams) -> Self {
        Self {
            asset: params.asset.clone(),
            min_ema_spread_pct: params.min_ema_spread_pct,
            min_profit_pct: params.min_profit_pct,
            min_hold_minutes: params.min_hold_minutes,
            require_htf_alignment: params.require_htf_alignment,
            htf_timeframe_minutes: params.htf_timeframe_minutes,
            prefer_trailing_stop_exit: params.prefer_trailing_stop_exit,
            trade_cooldown_seconds: params.trade_cooldown_seconds,
            base_config: StrategyConfig::default(),
        }
    }
}

impl StrategyParamApplier {
    /// Create a new strategy param applier
    pub fn new() -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get strategy config for an asset
    pub async fn get_config(&self, asset: &str) -> Option<OptimizedStrategyConfig> {
        let configs = self.configs.read().await;
        configs.get(asset).cloned()
    }

    /// Get all configs
    pub async fn get_all_configs(&self) -> HashMap<String, OptimizedStrategyConfig> {
        let configs = self.configs.read().await;
        configs.clone()
    }

    /// Check minimum EMA spread for an asset
    pub async fn check_min_ema_spread(&self, asset: &str, spread: f64) -> bool {
        let configs = self.configs.read().await;
        if let Some(config) = configs.get(asset) {
            spread >= config.min_ema_spread_pct
        } else {
            // Default: require at least 0.2% spread
            spread >= 0.2
        }
    }

    /// Check if hold time requirement is met
    pub async fn check_min_hold_time(&self, asset: &str, held_minutes: u32) -> bool {
        let configs = self.configs.read().await;
        if let Some(config) = configs.get(asset) {
            held_minutes >= config.min_hold_minutes
        } else {
            // Default: require at least 15 minutes
            held_minutes >= 15
        }
    }
}

impl Default for StrategyParamApplier {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ParamApplier for StrategyParamApplier {
    async fn apply_params(&self, params: &OptimizedParams) -> anyhow::Result<()> {
        let asset = &params.asset;
        let strategy_config = OptimizedStrategyConfig::from(params);

        {
            let mut configs = self.configs.write().await;
            let old = configs.insert(asset.clone(), strategy_config.clone());

            if old.is_some() {
                info!(
                    asset = asset,
                    min_ema_spread = strategy_config.min_ema_spread_pct,
                    min_profit = strategy_config.min_profit_pct,
                    min_hold = strategy_config.min_hold_minutes,
                    require_htf = strategy_config.require_htf_alignment,
                    "Updated strategy config for asset"
                );
            } else {
                info!(
                    asset = asset,
                    min_ema_spread = strategy_config.min_ema_spread_pct,
                    min_profit = strategy_config.min_profit_pct,
                    min_hold = strategy_config.min_hold_minutes,
                    require_htf = strategy_config.require_htf_alignment,
                    "Applied new strategy config for asset"
                );
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "StrategyParamApplier"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params(asset: &str) -> OptimizedParams {
        let mut params = OptimizedParams::new(asset);
        params.ema_fast_period = 12;
        params.ema_slow_period = 26;
        params.atr_length = 14;
        params.atr_multiplier = 2.5;
        params.min_ema_spread_pct = 0.25;
        params.min_profit_pct = 0.5;
        params.take_profit_pct = 3.0;
        params.min_hold_minutes = 20;
        params.max_position_size_usd = 100.0;
        params.enabled = true;
        params.optimization_score = 0.85;
        params
    }

    #[tokio::test]
    async fn test_noop_applier() {
        let applier = NoOpApplier::new("test");
        let params = create_test_params("BTC");

        assert_eq!(applier.name(), "test");
        assert!(applier.apply_params(&params).await.is_ok());
    }

    #[tokio::test]
    async fn test_logging_applier() {
        let applier = LoggingApplier::new("test_logger");
        let params = create_test_params("ETH");

        assert_eq!(applier.name(), "test_logger");
        assert!(applier.apply_params(&params).await.is_ok());
    }

    #[tokio::test]
    async fn test_indicator_applier_new_config() {
        let applier = IndicatorParamApplier::new();
        let params = create_test_params("BTC");

        // Initially no config
        assert!(applier.get_config("BTC").await.is_none());

        // Apply params
        applier.apply_params(&params).await.unwrap();

        // Config should exist now
        let config = applier.get_config("BTC").await.unwrap();
        assert_eq!(config.ema_fast_period, 12);
        assert_eq!(config.ema_slow_period, 26);
        assert_eq!(config.atr_period, 14);
    }

    #[tokio::test]
    async fn test_indicator_applier_update_config() {
        let applier = IndicatorParamApplier::new();

        // Apply initial params
        let mut params = create_test_params("BTC");
        applier.apply_params(&params).await.unwrap();

        let config = applier.get_config("BTC").await.unwrap();
        assert_eq!(config.ema_fast_period, 12);

        // Update params
        params.ema_fast_period = 8;
        applier.apply_params(&params).await.unwrap();

        let config = applier.get_config("BTC").await.unwrap();
        assert_eq!(config.ema_fast_period, 8);
    }

    #[tokio::test]
    async fn test_indicator_applier_multiple_assets() {
        let applier = IndicatorParamApplier::new();

        let btc_params = create_test_params("BTC");
        let eth_params = create_test_params("ETH");

        applier.apply_params(&btc_params).await.unwrap();
        applier.apply_params(&eth_params).await.unwrap();

        let all_configs = applier.get_all_configs().await;
        assert_eq!(all_configs.len(), 2);
        assert!(all_configs.contains_key("BTC"));
        assert!(all_configs.contains_key("ETH"));
    }

    #[tokio::test]
    async fn test_risk_applier() {
        use crate::risk::RiskConfig;
        let risk_manager = Arc::new(RwLock::new(RiskManager::new(RiskConfig::default())));
        let applier = RiskParamApplier::new(risk_manager);

        let params = create_test_params("BTC");
        applier.apply_params(&params).await.unwrap();

        let config = applier.get_asset_config("BTC").await.unwrap();
        assert_eq!(config.atr_multiplier, 2.5);
        assert_eq!(config.take_profit_pct, 3.0);
        assert_eq!(config.max_position_size_usd, 100.0);
        assert!(config.enabled);
    }

    #[tokio::test]
    async fn test_risk_applier_trading_enabled() {
        use crate::risk::RiskConfig;
        let risk_manager = Arc::new(RwLock::new(RiskManager::new(RiskConfig::default())));
        let applier = RiskParamApplier::new(risk_manager);

        let mut params = create_test_params("BTC");
        params.enabled = true;
        applier.apply_params(&params).await.unwrap();
        assert!(applier.is_trading_enabled("BTC").await);

        params.enabled = false;
        applier.apply_params(&params).await.unwrap();
        assert!(!applier.is_trading_enabled("BTC").await);
    }

    #[tokio::test]
    async fn test_strategy_applier() {
        let applier = StrategyParamApplier::new();

        let params = create_test_params("BTC");
        applier.apply_params(&params).await.unwrap();

        let config = applier.get_config("BTC").await.unwrap();
        assert_eq!(config.min_ema_spread_pct, 0.25);
        assert_eq!(config.min_profit_pct, 0.5);
        assert_eq!(config.min_hold_minutes, 20);
    }

    #[tokio::test]
    async fn test_strategy_applier_spread_check() {
        let applier = StrategyParamApplier::new();

        let mut params = create_test_params("BTC");
        params.min_ema_spread_pct = 0.3;
        applier.apply_params(&params).await.unwrap();

        // Below threshold
        assert!(!applier.check_min_ema_spread("BTC", 0.2).await);

        // At threshold
        assert!(applier.check_min_ema_spread("BTC", 0.3).await);

        // Above threshold
        assert!(applier.check_min_ema_spread("BTC", 0.5).await);
    }

    #[tokio::test]
    async fn test_strategy_applier_hold_time_check() {
        let applier = StrategyParamApplier::new();

        let mut params = create_test_params("BTC");
        params.min_hold_minutes = 15;
        applier.apply_params(&params).await.unwrap();

        // Below threshold
        assert!(!applier.check_min_hold_time("BTC", 10).await);

        // At threshold
        assert!(applier.check_min_hold_time("BTC", 15).await);

        // Above threshold
        assert!(applier.check_min_hold_time("BTC", 30).await);
    }

    #[tokio::test]
    async fn test_strategy_applier_unknown_asset_defaults() {
        let applier = StrategyParamApplier::new();

        // Unknown asset should use defaults
        assert!(applier.check_min_ema_spread("UNKNOWN", 0.2).await);
        assert!(!applier.check_min_ema_spread("UNKNOWN", 0.1).await);

        assert!(applier.check_min_hold_time("UNKNOWN", 15).await);
        assert!(!applier.check_min_hold_time("UNKNOWN", 10).await);
    }

    #[tokio::test]
    async fn test_asset_risk_config_from_params() {
        let params = create_test_params("BTC");
        let config = AssetRiskConfig::from(&params);

        assert_eq!(config.asset, "BTC");
        assert_eq!(config.atr_multiplier, 2.5);
        assert_eq!(config.max_position_size_usd, 100.0);
        assert!(config.enabled);
    }

    #[tokio::test]
    async fn test_optimized_strategy_config_from_params() {
        let params = create_test_params("ETH");
        let config = OptimizedStrategyConfig::from(&params);

        assert_eq!(config.asset, "ETH");
        assert_eq!(config.min_ema_spread_pct, 0.25);
        assert_eq!(config.min_hold_minutes, 20);
    }
}
