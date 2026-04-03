//! # Parameter Hot-Reload Module
//!
//! This module handles hot-reloading of optimized trading parameters from Redis.
//! It subscribes to the `fks:{instance}:param_updates` channel and applies
//! new parameters to strategies at runtime without service restart.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                         Parameter Hot-Reload                              │
//! ├──────────────────────────────────────────────────────────────────────────┤
//! │                                                                           │
//! │  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐  │
//! │  │   Redis     │───▶│ ParamReloadTask  │───▶│  ParamApplier (trait)   │  │
//! │  │  Pub/Sub    │    │  (background)    │    │  - SignalGenerator      │  │
//! │  └─────────────┘    └──────────────────┘    │  - RiskManager          │  │
//! │                              │              │  - Strategies           │  │
//! │                              ▼              └─────────────────────────┘  │
//! │                     ┌──────────────────┐                                  │
//! │                     │  ParamCache      │                                  │
//! │                     │  (in-memory)     │                                  │
//! │                     └──────────────────┘                                  │
//! │                                                                           │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::param_reload::{ParamReloadManager, ParamReloadConfig};
//!
//! // Create manager
//! let config = ParamReloadConfig::from_env();
//! let manager = ParamReloadManager::new(config).await?;
//!
//! // Start background reload task
//! let handle = manager.start_background_task().await?;
//!
//! // Get current params for an asset
//! if let Some(params) = manager.get_params("BTC").await {
//!     println!("BTC EMA fast: {}", params.ema_fast_period);
//! }
//! ```

use janus_core::optimized_params::{OptimizedParams, ParamManager, ParamNotification};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, error, info, warn};

/// Configuration for parameter hot-reload
#[derive(Debug, Clone)]
pub struct ParamReloadConfig {
    /// Redis URL for pub/sub connection
    pub redis_url: String,

    /// Instance ID for key namespacing
    pub instance_id: String,

    /// Whether hot-reload is enabled
    pub enabled: bool,

    /// Reconnection delay in milliseconds
    pub reconnect_delay_ms: u64,

    /// Maximum reconnection attempts (0 = unlimited)
    pub max_reconnect_attempts: u32,
}

impl Default for ParamReloadConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            instance_id: "default".to_string(),
            enabled: true,
            reconnect_delay_ms: 5000,
            max_reconnect_attempts: 0, // Unlimited
        }
    }
}

impl ParamReloadConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            instance_id: std::env::var("FKS_INSTANCE_ID").unwrap_or_else(|_| "default".to_string()),
            enabled: std::env::var("ENABLE_PARAM_RELOAD")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            reconnect_delay_ms: std::env::var("PARAM_RELOAD_RECONNECT_MS")
                .unwrap_or_else(|_| "5000".to_string())
                .parse()
                .unwrap_or(5000),
            max_reconnect_attempts: std::env::var("PARAM_RELOAD_MAX_RETRIES")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
        }
    }

    /// Create with custom Redis URL
    pub fn with_redis_url(mut self, url: impl Into<String>) -> Self {
        self.redis_url = url.into();
        self
    }

    /// Create with custom instance ID
    pub fn with_instance_id(mut self, id: impl Into<String>) -> Self {
        self.instance_id = id.into();
        self
    }

    /// Disable hot-reload
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// Trait for components that can apply optimized parameters
#[async_trait::async_trait]
pub trait ParamApplier: Send + Sync {
    /// Apply new optimized parameters for an asset
    async fn apply_params(&self, params: &OptimizedParams) -> anyhow::Result<()>;

    /// Get the name of this applier for logging
    fn name(&self) -> &str;
}

/// Statistics about parameter reloads
#[derive(Debug, Clone, Default)]
pub struct ReloadStats {
    /// Total params received
    pub total_received: u64,

    /// Successfully applied params
    pub successful_applies: u64,

    /// Failed applies
    pub failed_applies: u64,

    /// Last successful reload timestamp
    pub last_success: Option<String>,

    /// Last error message
    pub last_error: Option<String>,

    /// Per-asset reload counts
    pub per_asset: HashMap<String, u64>,
}

/// Manager for parameter hot-reload
pub struct ParamReloadManager {
    /// Configuration
    config: ParamReloadConfig,

    /// Parameter manager from janus-core
    param_manager: Arc<ParamManager>,

    /// Registered param appliers
    appliers: Arc<RwLock<Vec<Arc<dyn ParamApplier>>>>,

    /// Reload statistics
    stats: Arc<RwLock<ReloadStats>>,

    /// Channel for broadcasting reload events internally
    reload_tx: broadcast::Sender<ParamNotification>,

    /// Whether the background task is running
    running: Arc<RwLock<bool>>,
}

impl ParamReloadManager {
    /// Create a new parameter reload manager
    pub fn new(config: ParamReloadConfig) -> Self {
        let (reload_tx, _) = broadcast::channel(64);

        Self {
            param_manager: Arc::new(ParamManager::new(&config.instance_id)),
            config,
            appliers: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(ReloadStats::default())),
            reload_tx,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Register a parameter applier
    pub async fn register_applier(&self, applier: Arc<dyn ParamApplier>) {
        let mut appliers = self.appliers.write().await;
        info!(applier = applier.name(), "Registered param applier");
        appliers.push(applier);
    }

    /// Get current parameters for an asset
    pub async fn get_params(&self, asset: &str) -> Option<OptimizedParams> {
        self.param_manager.get(asset).await
    }

    /// Get all cached parameters
    pub async fn get_all_params(&self) -> HashMap<String, OptimizedParams> {
        self.param_manager.get_all().await
    }

    /// Subscribe to reload notifications
    pub fn subscribe(&self) -> broadcast::Receiver<ParamNotification> {
        self.reload_tx.subscribe()
    }

    /// Get reload statistics
    pub async fn stats(&self) -> ReloadStats {
        self.stats.read().await.clone()
    }

    /// Check if the reload task is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Load initial parameters from Redis
    #[cfg(feature = "redis")]
    pub async fn load_initial(&self) -> anyhow::Result<usize> {
        let client = redis::Client::open(self.config.redis_url.as_str())?;
        let count = self
            .param_manager
            .load_from_redis(&client)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to load params: {:?}", e))?;

        // Apply loaded params to all registered appliers
        let all_params = self.param_manager.get_all().await;
        for params in all_params.values() {
            self.apply_to_all(params).await?;
        }

        info!(count = count, "Loaded initial optimized params from Redis");
        Ok(count)
    }

    /// Load initial parameters from Redis (stub when redis feature disabled)
    #[cfg(not(feature = "redis"))]
    pub async fn load_initial(&self) -> anyhow::Result<usize> {
        warn!("Redis feature not enabled, skipping initial param load");
        Ok(0)
    }

    /// Start the background reload task
    pub async fn start_background_task(
        self: Arc<Self>,
    ) -> anyhow::Result<tokio::task::JoinHandle<()>> {
        if !self.config.enabled {
            info!("Parameter hot-reload is disabled");
            return Ok(tokio::spawn(async {}));
        }

        // Mark as running
        {
            let mut running = self.running.write().await;
            if *running {
                return Err(anyhow::anyhow!("Reload task already running"));
            }
            *running = true;
        }

        let manager = Arc::clone(&self);
        let handle = tokio::spawn(async move {
            manager.run_reload_loop().await;
        });

        info!(
            instance_id = self.config.instance_id,
            "Started parameter hot-reload background task"
        );

        Ok(handle)
    }

    /// Main reload loop
    async fn run_reload_loop(&self) {
        let mut reconnect_attempts = 0u32;

        loop {
            match self.connect_and_listen().await {
                Ok(()) => {
                    // Clean exit
                    break;
                }
                Err(e) => {
                    error!(error = %e, "Parameter reload connection error");

                    reconnect_attempts += 1;
                    if self.config.max_reconnect_attempts > 0
                        && reconnect_attempts >= self.config.max_reconnect_attempts
                    {
                        error!(
                            attempts = reconnect_attempts,
                            "Max reconnection attempts reached, stopping reload task"
                        );
                        break;
                    }

                    // Wait before reconnecting
                    tokio::time::sleep(tokio::time::Duration::from_millis(
                        self.config.reconnect_delay_ms,
                    ))
                    .await;

                    info!(
                        attempt = reconnect_attempts,
                        "Attempting to reconnect to Redis for param updates"
                    );
                }
            }
        }

        // Mark as stopped
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Connect to Redis and listen for param updates
    #[cfg(feature = "redis")]
    async fn connect_and_listen(&self) -> anyhow::Result<()> {
        use futures_util::StreamExt;

        let client = redis::Client::open(self.config.redis_url.as_str())?;
        let mut pubsub = client.get_async_pubsub().await?;

        let channel = self.param_manager.updates_channel();
        pubsub.subscribe(&channel).await?;

        info!(channel = channel, "Subscribed to param update channel");

        let mut stream = pubsub.on_message();

        while let Some(msg) = stream.next().await {
            let payload: String = msg.get_payload()?;

            match self.process_notification(&payload).await {
                Ok(()) => {
                    debug!("Processed param notification");
                }
                Err(e) => {
                    error!(error = %e, "Failed to process param notification");

                    // Update error stats
                    let mut stats = self.stats.write().await;
                    stats.failed_applies += 1;
                    stats.last_error = Some(e.to_string());
                }
            }
        }

        Ok(())
    }

    /// Connect stub when redis feature disabled
    #[cfg(not(feature = "redis"))]
    async fn connect_and_listen(&self) -> anyhow::Result<()> {
        warn!("Redis feature not enabled, param reload disabled");
        // Just sleep forever to keep the task alive
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
        }
    }

    /// Process a notification from Redis
    async fn process_notification(&self, json: &str) -> anyhow::Result<()> {
        let notification: ParamNotification = serde_json::from_str(json)?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_received += 1;
        }

        match &notification {
            ParamNotification::ParamUpdate { asset, params, .. } => {
                info!(
                    asset = asset,
                    score = params.optimization_score,
                    "Received optimized params update"
                );

                // Update the param manager's cache
                self.param_manager.update(params.clone()).await;

                // Apply to all registered appliers
                self.apply_to_all(params).await?;

                // Update per-asset stats
                {
                    let mut stats = self.stats.write().await;
                    *stats.per_asset.entry(asset.clone()).or_insert(0) += 1;
                    stats.successful_applies += 1;
                    stats.last_success = Some(chrono::Utc::now().to_rfc3339());
                }
            }
            ParamNotification::OptimizationStarted { assets, .. } => {
                info!(assets = ?assets, "Optimization started for assets");
            }
            ParamNotification::OptimizationComplete {
                successful, failed, ..
            } => {
                info!(
                    successful = successful,
                    failed = failed,
                    "Optimization cycle completed"
                );
            }
            ParamNotification::OptimizationFailed { asset, error, .. } => {
                warn!(
                    asset = asset,
                    error = error,
                    "Optimization failed for asset"
                );
            }
        }

        // Broadcast to internal subscribers
        let _ = self.reload_tx.send(notification);

        Ok(())
    }

    /// Apply parameters to all registered appliers
    async fn apply_to_all(&self, params: &OptimizedParams) -> anyhow::Result<()> {
        let appliers = self.appliers.read().await;

        for applier in appliers.iter() {
            match applier.apply_params(params).await {
                Ok(()) => {
                    debug!(
                        applier = applier.name(),
                        asset = params.asset,
                        "Applied params to applier"
                    );
                }
                Err(e) => {
                    error!(
                        applier = applier.name(),
                        asset = params.asset,
                        error = %e,
                        "Failed to apply params to applier"
                    );
                    // Continue to other appliers, don't fail completely
                }
            }
        }

        Ok(())
    }

    /// Manually update parameters (useful for testing)
    pub async fn manual_update(&self, params: OptimizedParams) -> anyhow::Result<()> {
        info!(asset = params.asset, "Manually updating optimized params");

        self.param_manager.update(params.clone()).await;
        self.apply_to_all(&params).await?;

        Ok(())
    }

    /// Stop the background reload task
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("Stopping parameter hot-reload task");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_reload::appliers::{LoggingApplier, NoOpApplier};

    #[test]
    fn test_config_default() {
        let config = ParamReloadConfig::default();
        assert_eq!(config.redis_url, "redis://localhost:6379");
        assert_eq!(config.instance_id, "default");
        assert!(config.enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = ParamReloadConfig::default()
            .with_redis_url("redis://custom:6380")
            .with_instance_id("test")
            .disabled();

        assert_eq!(config.redis_url, "redis://custom:6380");
        assert_eq!(config.instance_id, "test");
        assert!(!config.enabled);
    }

    #[tokio::test]
    async fn test_manager_creation() {
        let config = ParamReloadConfig::default().disabled();
        let manager = ParamReloadManager::new(config);

        assert!(!manager.is_running().await);
    }

    #[tokio::test]
    async fn test_register_applier() {
        let config = ParamReloadConfig::default().disabled();
        let manager = ParamReloadManager::new(config);

        let applier = Arc::new(NoOpApplier::new("test"));
        manager.register_applier(applier).await;

        let appliers = manager.appliers.read().await;
        assert_eq!(appliers.len(), 1);
    }

    #[tokio::test]
    async fn test_manual_update() {
        let config = ParamReloadConfig::default().disabled();
        let manager = ParamReloadManager::new(config);

        let applier = Arc::new(LoggingApplier::new("test"));
        manager.register_applier(applier).await;

        let params = OptimizedParams::new("BTC");
        manager.manual_update(params).await.unwrap();

        let cached = manager.get_params("BTC").await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().asset, "BTC");
    }

    #[tokio::test]
    async fn test_stats() {
        let config = ParamReloadConfig::default().disabled();
        let manager = ParamReloadManager::new(config);

        let stats = manager.stats().await;
        assert_eq!(stats.total_received, 0);
        assert_eq!(stats.successful_applies, 0);
    }

    #[tokio::test]
    async fn test_subscribe() {
        let config = ParamReloadConfig::default().disabled();
        let manager = ParamReloadManager::new(config);

        let mut receiver = manager.subscribe();

        // Receiver should be created without blocking
        assert!(receiver.try_recv().is_err());
    }
}
