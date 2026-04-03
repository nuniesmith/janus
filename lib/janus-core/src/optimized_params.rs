//! Optimized Parameters Module
//!
//! Receives and applies optimized trading parameters from the FKS Optimizer service
//! via Redis pub/sub. Enables hot-reloading of strategy parameters without restart.
//!
//! # Redis Integration
//!
//! The optimizer publishes parameters to:
//! - Hash: `fks:{instance}:optimized_params` - Stores current params per asset
//! - Channel: `fks:{instance}:param_updates` - Notifies of param changes
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_core::optimized_params::{OptimizedParams, ParamUpdateListener};
//!
//! // Load params from Redis
//! let params = OptimizedParams::load_from_redis(&redis_client, "BTC").await?;
//!
//! // Subscribe to updates
//! let listener = ParamUpdateListener::new("redis://localhost:6379", "default").await?;
//! while let Some(update) = listener.next_update().await {
//!     println!("New params for {}: {:?}", update.asset, update.params);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, error, info, warn};

/// Optimized parameters for a single asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedParams {
    /// Asset symbol (e.g., "BTC", "ETH", "SOL")
    pub asset: String,

    /// Fast EMA period (typically 8-14)
    #[serde(default = "default_ema_fast")]
    pub ema_fast_period: u32,

    /// Slow EMA period (typically 21-40)
    #[serde(default = "default_ema_slow")]
    pub ema_slow_period: u32,

    /// ATR calculation period
    #[serde(default = "default_atr_length")]
    pub atr_length: u32,

    /// ATR multiplier for trailing stop
    #[serde(default = "default_atr_multiplier")]
    pub atr_multiplier: f64,

    /// Minimum trailing stop percentage
    #[serde(default = "default_min_trailing_stop")]
    pub min_trailing_stop_pct: f64,

    /// Minimum EMA spread to confirm trend
    #[serde(default = "default_min_ema_spread")]
    pub min_ema_spread_pct: f64,

    /// Minimum profit percentage to allow exit
    #[serde(default = "default_min_profit")]
    pub min_profit_pct: f64,

    /// Take profit target percentage
    #[serde(default = "default_take_profit")]
    pub take_profit_pct: f64,

    /// Cooldown between trades in seconds
    #[serde(default = "default_cooldown")]
    pub trade_cooldown_seconds: u64,

    /// Whether to require higher timeframe alignment
    #[serde(default = "default_require_htf")]
    pub require_htf_alignment: bool,

    /// Higher timeframe for trend filter (minutes)
    #[serde(default = "default_htf_timeframe")]
    pub htf_timeframe_minutes: u32,

    /// Maximum position size in USD
    #[serde(default = "default_max_position")]
    pub max_position_size_usd: f64,

    /// Whether this asset is enabled for trading
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Minimum hold time in minutes
    #[serde(default = "default_min_hold")]
    pub min_hold_minutes: u32,

    /// Prefer trailing stop exit over EMA reversal
    #[serde(default = "default_prefer_trailing")]
    pub prefer_trailing_stop_exit: bool,

    /// When these params were optimized (ISO 8601)
    #[serde(default)]
    pub optimized_at: String,

    /// Optimization score (higher is better)
    #[serde(default)]
    pub optimization_score: f64,

    /// Backtest results summary
    #[serde(default)]
    pub backtest_result: BacktestResultSummary,
}

/// Summary of backtest results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BacktestResultSummary {
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub total_pnl_pct: f64,
    pub max_drawdown_pct: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
    pub trades_per_day: f64,
}

// Default value functions for serde
fn default_ema_fast() -> u32 {
    9
}
fn default_ema_slow() -> u32 {
    28
}
fn default_atr_length() -> u32 {
    14
}
fn default_atr_multiplier() -> f64 {
    2.0
}
fn default_min_trailing_stop() -> f64 {
    0.5
}
fn default_min_ema_spread() -> f64 {
    0.20
}
fn default_min_profit() -> f64 {
    0.40
}
fn default_take_profit() -> f64 {
    5.0
}
fn default_cooldown() -> u64 {
    1800
}
fn default_require_htf() -> bool {
    true
}
fn default_htf_timeframe() -> u32 {
    15
}
fn default_max_position() -> f64 {
    20.0
}
fn default_enabled() -> bool {
    true
}
fn default_min_hold() -> u32 {
    15
}
fn default_prefer_trailing() -> bool {
    true
}

impl Default for OptimizedParams {
    fn default() -> Self {
        Self {
            asset: String::new(),
            ema_fast_period: default_ema_fast(),
            ema_slow_period: default_ema_slow(),
            atr_length: default_atr_length(),
            atr_multiplier: default_atr_multiplier(),
            min_trailing_stop_pct: default_min_trailing_stop(),
            min_ema_spread_pct: default_min_ema_spread(),
            min_profit_pct: default_min_profit(),
            take_profit_pct: default_take_profit(),
            trade_cooldown_seconds: default_cooldown(),
            require_htf_alignment: default_require_htf(),
            htf_timeframe_minutes: default_htf_timeframe(),
            max_position_size_usd: default_max_position(),
            enabled: default_enabled(),
            min_hold_minutes: default_min_hold(),
            prefer_trailing_stop_exit: default_prefer_trailing(),
            optimized_at: String::new(),
            optimization_score: 0.0,
            backtest_result: BacktestResultSummary::default(),
        }
    }
}

impl OptimizedParams {
    /// Create new params for an asset with defaults
    pub fn new(asset: impl Into<String>) -> Self {
        Self {
            asset: asset.into(),
            ..Default::default()
        }
    }

    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ParamError> {
        if self.ema_fast_period >= self.ema_slow_period {
            return Err(ParamError::InvalidConfig(
                "EMA fast period must be less than slow period".into(),
            ));
        }

        if self.ema_fast_period < 3 || self.ema_slow_period < 5 {
            return Err(ParamError::InvalidConfig("EMA periods too small".into()));
        }

        if self.atr_multiplier <= 0.0 {
            return Err(ParamError::InvalidConfig(
                "ATR multiplier must be positive".into(),
            ));
        }

        if self.min_ema_spread_pct < 0.0 || self.min_ema_spread_pct > 10.0 {
            return Err(ParamError::InvalidConfig(
                "Min EMA spread must be between 0 and 10%".into(),
            ));
        }

        Ok(())
    }

    /// Check if trading is allowed with these params
    pub fn is_trading_enabled(&self) -> bool {
        self.enabled && self.max_position_size_usd > 0.0
    }
}

/// Notification types from the optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParamNotification {
    /// Single asset params updated
    ParamUpdate {
        asset: String,
        timestamp: String,
        params: OptimizedParams,
    },
    /// Optimization cycle started
    OptimizationStarted {
        timestamp: String,
        assets: Vec<String>,
    },
    /// Optimization cycle completed
    OptimizationComplete {
        timestamp: String,
        successful: u32,
        failed: u32,
        assets: Vec<String>,
    },
    /// Single asset optimization failed
    OptimizationFailed {
        timestamp: String,
        asset: String,
        error: String,
    },
}

/// Error types for parameter operations
#[derive(Debug, thiserror::Error)]
pub enum ParamError {
    #[error("Redis error: {0}")]
    Redis(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    #[error("Connection error: {0}")]
    Connection(String),
}

/// Manager for optimized parameters with caching and updates
pub struct ParamManager {
    /// Current params cache (asset -> params)
    params: Arc<RwLock<HashMap<String, OptimizedParams>>>,

    /// Redis instance ID for key prefix
    instance_id: String,

    /// Channel for broadcasting param updates
    update_tx: broadcast::Sender<ParamNotification>,
}

impl ParamManager {
    /// Create a new param manager
    pub fn new(instance_id: impl Into<String>) -> Self {
        let (update_tx, _) = broadcast::channel(64);
        Self {
            params: Arc::new(RwLock::new(HashMap::new())),
            instance_id: instance_id.into(),
            update_tx,
        }
    }

    /// Get the Redis key prefix
    pub fn redis_key(&self, suffix: &str) -> String {
        format!("fks:{}:{}", self.instance_id, suffix)
    }

    /// Get the param updates channel name
    pub fn updates_channel(&self) -> String {
        self.redis_key("param_updates")
    }

    /// Get the params hash key
    pub fn params_hash_key(&self) -> String {
        self.redis_key("optimized_params")
    }

    /// Subscribe to param update notifications
    pub fn subscribe(&self) -> broadcast::Receiver<ParamNotification> {
        self.update_tx.subscribe()
    }

    /// Get params for an asset (from cache)
    pub async fn get(&self, asset: &str) -> Option<OptimizedParams> {
        let params = self.params.read().await;
        params.get(asset).cloned()
    }

    /// Get all cached params
    pub async fn get_all(&self) -> HashMap<String, OptimizedParams> {
        let params = self.params.read().await;
        params.clone()
    }

    /// Update params for an asset
    pub async fn update(&self, params: OptimizedParams) {
        let asset = params.asset.clone();
        let notification = ParamNotification::ParamUpdate {
            asset: asset.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            params: params.clone(),
        };

        // Update cache
        {
            let mut cache = self.params.write().await;
            cache.insert(asset.clone(), params);
        }

        // Broadcast notification
        if let Err(e) = self.update_tx.send(notification) {
            debug!("No subscribers for param update: {}", e);
        }

        info!("Updated optimized params for {}", asset);
    }

    /// Load all params from Redis
    #[cfg(feature = "redis")]
    pub async fn load_from_redis(&self, client: &redis::Client) -> Result<usize, ParamError> {
        use redis::AsyncCommands;

        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| ParamError::Redis(e.to_string()))?;

        let key = self.params_hash_key();
        let all_params: HashMap<String, String> = conn
            .hgetall(&key)
            .await
            .map_err(|e| ParamError::Redis(e.to_string()))?;

        let mut count = 0;
        let mut cache = self.params.write().await;

        for (asset, json) in all_params {
            // Skip metadata keys
            if asset.starts_with('_') {
                continue;
            }

            match serde_json::from_str::<OptimizedParams>(&json) {
                Ok(params) => {
                    cache.insert(asset, params);
                    count += 1;
                }
                Err(e) => {
                    warn!("Failed to parse params for {}: {}", asset, e);
                }
            }
        }

        info!("Loaded {} optimized params from Redis", count);
        Ok(count)
    }

    /// Process a notification from Redis pub/sub
    pub async fn process_notification(&self, json: &str) -> Result<(), ParamError> {
        let notification: ParamNotification =
            serde_json::from_str(json).map_err(|e| ParamError::Serialization(e.to_string()))?;

        match &notification {
            ParamNotification::ParamUpdate { asset, params, .. } => {
                let mut cache = self.params.write().await;
                cache.insert(asset.clone(), params.clone());
                info!(
                    "Applied optimized params for {} (score: {:.2})",
                    asset, params.optimization_score
                );
            }
            ParamNotification::OptimizationStarted { assets, .. } => {
                info!("Optimization started for: {:?}", assets);
            }
            ParamNotification::OptimizationComplete {
                successful, failed, ..
            } => {
                info!(
                    "Optimization complete: {} successful, {} failed",
                    successful, failed
                );
            }
            ParamNotification::OptimizationFailed { asset, error, .. } => {
                warn!("Optimization failed for {}: {}", asset, error);
            }
        }

        // Broadcast to subscribers
        if let Err(e) = self.update_tx.send(notification) {
            debug!("No subscribers for notification: {}", e);
        }

        Ok(())
    }
}

/// Listener for Redis pub/sub param updates
#[cfg(feature = "redis")]
pub struct ParamUpdateListener {
    manager: Arc<ParamManager>,
    pubsub: Option<redis::aio::PubSub>,
}

#[cfg(feature = "redis")]
impl ParamUpdateListener {
    /// Create a new listener
    pub async fn new(redis_url: &str, instance_id: impl Into<String>) -> Result<Self, ParamError> {
        let manager = Arc::new(ParamManager::new(instance_id));

        let client =
            redis::Client::open(redis_url).map_err(|e| ParamError::Connection(e.to_string()))?;

        let mut pubsub = client
            .get_async_pubsub()
            .await
            .map_err(|e| ParamError::Connection(e.to_string()))?;

        let channel = manager.updates_channel();
        pubsub
            .subscribe(&channel)
            .await
            .map_err(|e| ParamError::Redis(e.to_string()))?;

        info!("Subscribed to param updates on channel: {}", channel);

        Ok(Self {
            manager,
            pubsub: Some(pubsub),
        })
    }

    /// Get the param manager
    pub fn manager(&self) -> Arc<ParamManager> {
        Arc::clone(&self.manager)
    }

    /// Wait for and process the next update
    pub async fn next_update(&mut self) -> Option<ParamNotification> {
        use futures_util::StreamExt;

        let pubsub = self.pubsub.as_mut()?;

        while let Some(msg) = pubsub.on_message().next().await {
            let payload: String = match msg.get_payload() {
                Ok(p) => p,
                Err(e) => {
                    error!("Failed to get message payload: {}", e);
                    continue;
                }
            };

            match self.manager.process_notification(&payload).await {
                Ok(()) => {
                    // Return the notification to the caller
                    if let Ok(notification) = serde_json::from_str(&payload) {
                        return Some(notification);
                    }
                }
                Err(e) => {
                    error!("Failed to process notification: {}", e);
                }
            }
        }

        None
    }

    /// Run the listener loop (blocking)
    pub async fn run(&mut self) {
        info!("Starting param update listener loop");
        while let Some(notification) = self.next_update().await {
            debug!("Processed notification: {:?}", notification);
        }
        warn!("Param update listener loop ended");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = OptimizedParams::default();
        assert_eq!(params.ema_fast_period, 9);
        assert_eq!(params.ema_slow_period, 28);
        assert!(params.enabled);
    }

    #[test]
    fn test_params_validation() {
        let mut params = OptimizedParams::new("BTC");
        assert!(params.validate().is_ok());

        // Invalid: fast >= slow
        params.ema_fast_period = 30;
        params.ema_slow_period = 20;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_notification_serialization() {
        let notification = ParamNotification::ParamUpdate {
            asset: "BTC".to_string(),
            timestamp: "2025-01-15T12:00:00Z".to_string(),
            params: OptimizedParams::new("BTC"),
        };

        let json = serde_json::to_string(&notification).unwrap();
        assert!(json.contains("param_update"));
        assert!(json.contains("BTC"));

        let parsed: ParamNotification = serde_json::from_str(&json).unwrap();
        match parsed {
            ParamNotification::ParamUpdate { asset, .. } => {
                assert_eq!(asset, "BTC");
            }
            _ => panic!("Wrong notification type"),
        }
    }

    #[tokio::test]
    async fn test_param_manager() {
        let manager = ParamManager::new("test");

        // Initially empty
        assert!(manager.get("BTC").await.is_none());

        // Update
        let params = OptimizedParams::new("BTC");
        manager.update(params.clone()).await;

        // Should now exist
        let retrieved = manager.get("BTC").await.unwrap();
        assert_eq!(retrieved.asset, "BTC");
        assert_eq!(retrieved.ema_fast_period, params.ema_fast_period);
    }

    #[test]
    fn test_redis_keys() {
        let manager = ParamManager::new("personal");
        assert_eq!(manager.params_hash_key(), "fks:personal:optimized_params");
        assert_eq!(manager.updates_channel(), "fks:personal:param_updates");
    }

    #[test]
    fn test_trading_enabled() {
        let mut params = OptimizedParams::new("BTC");
        assert!(params.is_trading_enabled());

        params.enabled = false;
        assert!(!params.is_trading_enabled());

        params.enabled = true;
        params.max_position_size_usd = 0.0;
        assert!(!params.is_trading_enabled());
    }
}
