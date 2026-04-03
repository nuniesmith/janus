//! # Model Checkpoint Reload Listener
//!
//! Subscribes to Redis pub/sub checkpoint notifications published by the
//! backward service and triggers model reload in the forward service's
//! [`ModelCache`](crate::inference::ModelCache).
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                     Model Hot-Reload Pipeline                            │
//! ├──────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Backward Service                      Forward Service                   │
//! │  ┌─────────────────┐                  ┌──────────────────────────────┐   │
//! │  │ save_checkpoint  │──Redis Pub/Sub──▶│  ModelReloadListener         │   │
//! │  │ + publish notify │                  │  - subscribes to channel     │   │
//! │  └─────────────────┘                  │  - triggers model reload     │   │
//! │                                        │  - tracks reload stats       │   │
//! │                                        └──────────┬───────────────────┘   │
//! │                                                   │                      │
//! │                                                   ▼                      │
//! │                                        ┌──────────────────────────────┐   │
//! │                                        │  ModelCache / ModelInference │   │
//! │                                        │  (re-loads ONNX / weights)   │   │
//! │                                        └──────────────────────────────┘   │
//! │                                                                          │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::param_reload::model_reload::{ModelReloadConfig, ModelReloadListener};
//!
//! let config = ModelReloadConfig::from_env();
//! let listener = Arc::new(ModelReloadListener::new(config));
//!
//! // Register a reload handler
//! listener.set_handler(|notification| {
//!     // reload model from notification.model_path
//!     Ok(())
//! }).await;
//!
//! // Start background task
//! let handle = listener.clone().start().await?;
//! ```

use janus_core::checkpoint_notify::{CheckpointNotification, checkpoint_channel};
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
#[allow(unused_imports)]
use tracing::{debug, error, info, warn};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the model reload listener.
#[derive(Debug, Clone)]
pub struct ModelReloadConfig {
    /// Redis URL for pub/sub subscription.
    pub redis_url: String,

    /// Instance ID for channel namespacing.
    pub instance_id: String,

    /// Whether model hot-reload is enabled.
    pub enabled: bool,

    /// Reconnection delay in milliseconds after a connection loss.
    pub reconnect_delay_ms: u64,

    /// Maximum reconnection attempts (0 = unlimited).
    pub max_reconnect_attempts: u32,
}

impl Default for ModelReloadConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            instance_id: "default".to_string(),
            enabled: true,
            reconnect_delay_ms: 5000,
            max_reconnect_attempts: 0,
        }
    }
}

impl ModelReloadConfig {
    /// Load configuration from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        Self {
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            instance_id: std::env::var("FKS_INSTANCE_ID").unwrap_or_else(|_| "default".to_string()),
            enabled: std::env::var("ENABLE_MODEL_RELOAD")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            reconnect_delay_ms: std::env::var("MODEL_RELOAD_RECONNECT_MS")
                .unwrap_or_else(|_| "5000".to_string())
                .parse()
                .unwrap_or(5000),
            max_reconnect_attempts: std::env::var("MODEL_RELOAD_MAX_RETRIES")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
        }
    }

    /// Builder: set Redis URL.
    pub fn with_redis_url(mut self, url: impl Into<String>) -> Self {
        self.redis_url = url.into();
        self
    }

    /// Builder: set instance ID.
    pub fn with_instance_id(mut self, id: impl Into<String>) -> Self {
        self.instance_id = id.into();
        self
    }

    /// Builder: disable model reload.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Statistics about model reload operations.
#[derive(Debug, Clone, Default)]
pub struct ModelReloadStats {
    /// Total checkpoint notifications received.
    pub total_received: u64,

    /// Successful model reloads.
    pub successful_reloads: u64,

    /// Failed model reloads.
    pub failed_reloads: u64,

    /// Last successfully loaded checkpoint version.
    pub last_version: u64,

    /// ISO-8601 timestamp of the last successful reload.
    pub last_success: Option<String>,

    /// Last error message (if any).
    pub last_error: Option<String>,

    /// Path to the last successfully loaded model.
    pub last_model_path: Option<String>,
}

// ─── Reload handler trait ─────────────────────────────────────────────────────

/// Trait for components that can reload a model from a checkpoint notification.
///
/// Implement this for your inference engine or model cache so the listener
/// can trigger reloads automatically.
#[async_trait::async_trait]
pub trait ModelReloadHandler: Send + Sync {
    /// Called when a new checkpoint notification is received.
    ///
    /// The implementor should load the model from `notification.model_path`
    /// and update internal state.  Returns `Ok(())` on success.
    async fn reload_model(&self, notification: &CheckpointNotification) -> anyhow::Result<()>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

// ─── Listener ─────────────────────────────────────────────────────────────────

/// Listens for model checkpoint notifications on Redis pub/sub and dispatches
/// them to registered [`ModelReloadHandler`]s.
pub struct ModelReloadListener {
    config: ModelReloadConfig,

    /// Registered reload handlers.
    handlers: Arc<RwLock<Vec<Arc<dyn ModelReloadHandler>>>>,

    /// Reload statistics.
    stats: Arc<RwLock<ModelReloadStats>>,

    /// Internal broadcast channel so other parts of the forward service can
    /// subscribe to checkpoint events without touching Redis directly.
    notify_tx: broadcast::Sender<CheckpointNotification>,

    /// Whether the background task is running.
    running: Arc<RwLock<bool>>,
}

impl ModelReloadListener {
    /// Create a new model reload listener.
    pub fn new(config: ModelReloadConfig) -> Self {
        let (notify_tx, _) = broadcast::channel(32);

        Self {
            config,
            handlers: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(ModelReloadStats::default())),
            notify_tx,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Register a handler that will be called on each checkpoint notification.
    pub async fn register_handler(&self, handler: Arc<dyn ModelReloadHandler>) {
        let mut handlers = self.handlers.write().await;
        info!(handler = handler.name(), "Registered model reload handler");
        handlers.push(handler);
    }

    /// Subscribe to an internal broadcast of checkpoint notifications.
    pub fn subscribe(&self) -> broadcast::Receiver<CheckpointNotification> {
        self.notify_tx.subscribe()
    }

    /// Get reload statistics.
    pub async fn stats(&self) -> ModelReloadStats {
        self.stats.read().await.clone()
    }

    /// Whether the background listener task is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Start the background listener task.
    ///
    /// Returns a `JoinHandle` that can be used to await or abort the task.
    pub async fn start(self: Arc<Self>) -> anyhow::Result<tokio::task::JoinHandle<()>> {
        if !self.config.enabled {
            info!("Model hot-reload is disabled");
            return Ok(tokio::spawn(async {}));
        }

        {
            let mut running = self.running.write().await;
            if *running {
                return Err(anyhow::anyhow!("Model reload listener already running"));
            }
            *running = true;
        }

        let listener = Arc::clone(&self);
        let handle = tokio::spawn(async move {
            listener.run_loop().await;
        });

        info!(
            instance_id = %self.config.instance_id,
            "Started model checkpoint reload listener"
        );

        Ok(handle)
    }

    /// Stop the background listener.
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("Stopping model reload listener");
    }

    /// Manually trigger a reload from a notification (useful for testing).
    pub async fn manual_reload(&self, notification: CheckpointNotification) -> anyhow::Result<()> {
        info!(
            model_path = %notification.model_path,
            version = notification.version,
            "Manually triggering model reload"
        );
        self.dispatch_notification(&notification).await
    }

    // ── Internal ──────────────────────────────────────────────────────────

    /// Main reconnect loop.
    async fn run_loop(&self) {
        let mut reconnect_attempts = 0u32;

        loop {
            if !*self.running.read().await {
                break;
            }

            match self.connect_and_listen().await {
                Ok(()) => break,
                Err(e) => {
                    error!(error = %e, "Model reload listener connection error");

                    reconnect_attempts += 1;
                    if self.config.max_reconnect_attempts > 0
                        && reconnect_attempts >= self.config.max_reconnect_attempts
                    {
                        error!(
                            attempts = reconnect_attempts,
                            "Max reconnect attempts reached — stopping model reload listener"
                        );
                        break;
                    }

                    tokio::time::sleep(tokio::time::Duration::from_millis(
                        self.config.reconnect_delay_ms,
                    ))
                    .await;

                    info!(
                        attempt = reconnect_attempts,
                        "Attempting to reconnect for model checkpoint updates"
                    );
                }
            }
        }

        let mut running = self.running.write().await;
        *running = false;
    }

    /// Connect to Redis and listen for checkpoint notifications.
    #[cfg(feature = "redis")]
    async fn connect_and_listen(&self) -> anyhow::Result<()> {
        use futures_util::StreamExt;

        let channel = checkpoint_channel(&self.config.instance_id);
        let client = redis::Client::open(self.config.redis_url.as_str())?;
        let mut pubsub = client.get_async_pubsub().await?;

        pubsub.subscribe(&channel).await?;
        info!(channel = %channel, "Subscribed to model checkpoint channel");

        let mut stream = pubsub.on_message();

        while let Some(msg) = stream.next().await {
            if !*self.running.read().await {
                break;
            }

            let payload: String = msg.get_payload()?;

            match CheckpointNotification::from_json(&payload) {
                Ok(notification) => {
                    {
                        let mut stats = self.stats.write().await;
                        stats.total_received += 1;
                    }

                    if let Err(e) = self.dispatch_notification(&notification).await {
                        error!(
                            error = %e,
                            model_path = %notification.model_path,
                            "Failed to dispatch model reload"
                        );
                        let mut stats = self.stats.write().await;
                        stats.failed_reloads += 1;
                        stats.last_error = Some(e.to_string());
                    }
                }
                Err(e) => {
                    error!(
                        error = %e,
                        "Failed to deserialise checkpoint notification"
                    );
                    let mut stats = self.stats.write().await;
                    stats.failed_reloads += 1;
                    stats.last_error = Some(format!("Deserialisation error: {e}"));
                }
            }
        }

        Ok(())
    }

    /// Stub when `redis` feature is disabled.
    #[cfg(not(feature = "redis"))]
    async fn connect_and_listen(&self) -> anyhow::Result<()> {
        warn!("Redis feature not enabled — model reload listener disabled");
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
            if !*self.running.read().await {
                break;
            }
        }
        Ok(())
    }

    /// Dispatch a notification to all registered handlers and broadcast it
    /// to internal subscribers.
    async fn dispatch_notification(
        &self,
        notification: &CheckpointNotification,
    ) -> anyhow::Result<()> {
        info!(
            model_path = %notification.model_path,
            model_name = %notification.model_name,
            version = notification.version,
            training_step = notification.training_step,
            "Processing model checkpoint notification"
        );

        let handlers = self.handlers.read().await;
        let mut any_failed = false;

        for handler in handlers.iter() {
            match handler.reload_model(notification).await {
                Ok(()) => {
                    debug!(
                        handler = handler.name(),
                        model_path = %notification.model_path,
                        "Handler successfully reloaded model"
                    );
                }
                Err(e) => {
                    error!(
                        handler = handler.name(),
                        error = %e,
                        "Handler failed to reload model"
                    );
                    any_failed = true;
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            if any_failed {
                stats.failed_reloads += 1;
                stats.last_error = Some("One or more handlers failed".to_string());
            } else {
                stats.successful_reloads += 1;
                stats.last_version = notification.version;
                stats.last_success = Some(chrono::Utc::now().to_rfc3339());
                stats.last_model_path = Some(notification.model_path.clone());
            }
        }

        // Broadcast to internal subscribers (ignore if no receivers)
        let _ = self.notify_tx.send(notification.clone());

        if any_failed {
            Err(anyhow::anyhow!("One or more model reload handlers failed"))
        } else {
            Ok(())
        }
    }
}

// ─── Convenience handler implementations ──────────────────────────────────────

/// A no-op handler that logs the notification but does not reload anything.
/// Useful for testing and development.
pub struct LoggingReloadHandler {
    name: String,
}

impl LoggingReloadHandler {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait::async_trait]
impl ModelReloadHandler for LoggingReloadHandler {
    async fn reload_model(&self, notification: &CheckpointNotification) -> anyhow::Result<()> {
        info!(
            handler = %self.name,
            model_path = %notification.model_path,
            model_name = %notification.model_name,
            version = notification.version,
            training_step = notification.training_step,
            "LoggingReloadHandler: received checkpoint notification (no-op)"
        );
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A handler that calls a user-supplied async closure on each notification.
pub struct CallbackReloadHandler<F>
where
    F: Fn(
            CheckpointNotification,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>>
        + Send
        + Sync,
{
    name: String,
    callback: F,
}

impl<F> CallbackReloadHandler<F>
where
    F: Fn(
            CheckpointNotification,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>>
        + Send
        + Sync,
{
    pub fn new(name: impl Into<String>, callback: F) -> Self {
        Self {
            name: name.into(),
            callback,
        }
    }
}

#[async_trait::async_trait]
impl<F> ModelReloadHandler for CallbackReloadHandler<F>
where
    F: Fn(
            CheckpointNotification,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>>
        + Send
        + Sync,
{
    async fn reload_model(&self, notification: &CheckpointNotification) -> anyhow::Result<()> {
        (self.callback)(notification.clone()).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ModelReloadConfig::default();
        assert_eq!(config.redis_url, "redis://localhost:6379");
        assert_eq!(config.instance_id, "default");
        assert!(config.enabled);
        assert_eq!(config.reconnect_delay_ms, 5000);
        assert_eq!(config.max_reconnect_attempts, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = ModelReloadConfig::default()
            .with_redis_url("redis://custom:6380")
            .with_instance_id("staging")
            .disabled();

        assert_eq!(config.redis_url, "redis://custom:6380");
        assert_eq!(config.instance_id, "staging");
        assert!(!config.enabled);
    }

    #[tokio::test]
    async fn test_listener_creation() {
        let listener = ModelReloadListener::new(ModelReloadConfig::default());
        assert!(!listener.is_running().await);

        let stats = listener.stats().await;
        assert_eq!(stats.total_received, 0);
        assert_eq!(stats.successful_reloads, 0);
    }

    #[tokio::test]
    async fn test_register_handler() {
        let listener = ModelReloadListener::new(ModelReloadConfig::default());
        let handler = Arc::new(LoggingReloadHandler::new("test-handler"));

        listener.register_handler(handler).await;

        let handlers = listener.handlers.read().await;
        assert_eq!(handlers.len(), 1);
    }

    #[tokio::test]
    async fn test_manual_reload_with_logging_handler() {
        let listener = ModelReloadListener::new(ModelReloadConfig::default());
        let handler = Arc::new(LoggingReloadHandler::new("logging"));
        listener.register_handler(handler).await;

        let notification = CheckpointNotification::new("model.bin", "test_model")
            .with_version(1)
            .with_training_step(500);

        let result = listener.manual_reload(notification).await;
        assert!(result.is_ok());

        let stats = listener.stats().await;
        assert_eq!(stats.successful_reloads, 1);
        assert_eq!(stats.last_version, 1);
        assert!(stats.last_model_path.is_some());
        assert_eq!(stats.last_model_path.unwrap(), "model.bin");
    }

    #[tokio::test]
    async fn test_manual_reload_no_handlers() {
        let listener = ModelReloadListener::new(ModelReloadConfig::default());

        let notification = CheckpointNotification::new("model.bin", "test");
        let result = listener.manual_reload(notification).await;
        // Should succeed even with no handlers
        assert!(result.is_ok());

        let stats = listener.stats().await;
        assert_eq!(stats.successful_reloads, 1);
    }

    #[tokio::test]
    async fn test_subscribe_receives_notifications() {
        let listener = ModelReloadListener::new(ModelReloadConfig::default());
        let mut rx = listener.subscribe();

        let notification = CheckpointNotification::new("model.bin", "test").with_version(42);

        // Manual reload should broadcast
        listener.manual_reload(notification).await.unwrap();

        let received = rx.try_recv();
        assert!(received.is_ok());
        assert_eq!(received.unwrap().version, 42);
    }

    #[tokio::test]
    async fn test_disabled_listener_start() {
        let config = ModelReloadConfig::default().disabled();
        let listener = Arc::new(ModelReloadListener::new(config));

        let handle = listener.start().await;
        assert!(handle.is_ok());
        // Should return immediately since it's disabled
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let listener = ModelReloadListener::new(ModelReloadConfig::default());
        let handler = Arc::new(LoggingReloadHandler::new("stats-test"));
        listener.register_handler(handler).await;

        // Trigger multiple reloads
        for i in 0..5 {
            let n = CheckpointNotification::new("model.bin", "test").with_version(i);
            listener.manual_reload(n).await.unwrap();
        }

        let stats = listener.stats().await;
        assert_eq!(stats.successful_reloads, 5);
        assert_eq!(stats.last_version, 4);
        assert_eq!(stats.failed_reloads, 0);
    }

    #[tokio::test]
    async fn test_callback_handler() {
        let counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let counter_clone = counter.clone();

        let handler = Arc::new(CallbackReloadHandler::new(
            "counter",
            move |_notification: CheckpointNotification| {
                let c = counter_clone.clone();
                Box::pin(async move {
                    c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(())
                })
            },
        ));

        let listener = ModelReloadListener::new(ModelReloadConfig::default());
        listener.register_handler(handler).await;

        let n = CheckpointNotification::new("model.bin", "test");
        listener.manual_reload(n).await.unwrap();

        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn test_logging_handler_name() {
        let handler = LoggingReloadHandler::new("my-handler");
        assert_eq!(handler.name(), "my-handler");
    }
}
