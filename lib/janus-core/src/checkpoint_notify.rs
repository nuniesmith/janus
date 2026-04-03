//! # Model Checkpoint Notification
//!
//! Shared types for notifying downstream services (e.g. Forward) when the
//! Backward service saves a new model checkpoint.
//!
//! ## Redis Channel
//!
//! Notifications are published on `fks:{instance_id}:model_checkpoints` as
//! JSON-serialised [`CheckpointNotification`] messages.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_core::checkpoint_notify::{CheckpointNotification, checkpoint_channel};
//!
//! // Publisher (backward service)
//! let channel = checkpoint_channel("default");
//! let notification = CheckpointNotification::new(
//!     "checkpoints/backward/latest_model.bin",
//!     "lstm_dqn_v1",
//! );
//! let json = serde_json::to_string(&notification).unwrap();
//! // redis.publish(channel, json).await?;
//!
//! // Subscriber (forward service)
//! let channel = checkpoint_channel("default");
//! // redis.subscribe(channel).await?;
//! // ... on message:
//! let notification: CheckpointNotification = serde_json::from_str(&payload)?;
//! println!("New checkpoint: {}", notification.model_path);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Returns the Redis pub/sub channel name used for model checkpoint
/// notifications for the given instance.
///
/// Format: `fks:{instance_id}:model_checkpoints`
pub fn checkpoint_channel(instance_id: &str) -> String {
    format!("fks:{instance_id}:model_checkpoints")
}

/// A notification emitted by the backward service when a new model
/// checkpoint has been saved to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointNotification {
    /// Absolute or relative path to the saved checkpoint file.
    pub model_path: String,

    /// Human-readable model name / architecture identifier
    /// (e.g. `"lstm_dqn_v1"`).
    pub model_name: String,

    /// Monotonically increasing version number for this checkpoint lineage.
    /// Can be used by consumers to skip stale notifications that arrive
    /// out of order.
    pub version: u64,

    /// ISO-8601 / RFC-3339 timestamp of when the checkpoint was saved.
    pub saved_at: String,

    /// Training step (gradient step count) at which this checkpoint was
    /// produced.  `0` if unknown.
    pub training_step: u64,

    /// Optional key-value metadata attached by the producer (e.g. loss,
    /// mean Q, learning rate at checkpoint time).
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl CheckpointNotification {
    /// Create a new checkpoint notification with sensible defaults.
    ///
    /// `version` and `training_step` default to `0`; callers should set
    /// them explicitly when the information is available.
    pub fn new(model_path: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_name: model_name.into(),
            version: 0,
            saved_at: chrono::Utc::now().to_rfc3339(),
            training_step: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set the version number.
    pub fn with_version(mut self, version: u64) -> Self {
        self.version = version;
        self
    }

    /// Set the training step.
    pub fn with_training_step(mut self, step: u64) -> Self {
        self.training_step = step;
        self
    }

    /// Insert a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Serialise to JSON for publishing over Redis.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialise from a JSON payload received over Redis.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ─── Notifier (publisher side) ────────────────────────────────────────────────

/// Configuration for the checkpoint notifier.
#[derive(Debug, Clone)]
pub struct CheckpointNotifierConfig {
    /// Redis URL for pub/sub.
    pub redis_url: String,
    /// Instance ID used for channel namespacing.
    pub instance_id: String,
    /// Whether notification publishing is enabled.
    pub enabled: bool,
}

impl Default for CheckpointNotifierConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            instance_id: "default".to_string(),
            enabled: true,
        }
    }
}

impl CheckpointNotifierConfig {
    /// Load from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        Self {
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            instance_id: std::env::var("FKS_INSTANCE_ID").unwrap_or_else(|_| "default".to_string()),
            enabled: std::env::var("ENABLE_CHECKPOINT_NOTIFY")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
        }
    }
}

/// Publishes model checkpoint notifications to Redis pub/sub.
///
/// Used by the backward service after saving a new checkpoint file.
pub struct CheckpointNotifier {
    config: CheckpointNotifierConfig,
    channel: String,
}

impl CheckpointNotifier {
    /// Create a new notifier.  Does **not** open a Redis connection yet —
    /// connections are created on each [`publish`](Self::publish) call to
    /// keep the notifier lightweight and resilient to transient failures.
    pub fn new(config: CheckpointNotifierConfig) -> Self {
        let channel = checkpoint_channel(&config.instance_id);
        Self { config, channel }
    }

    /// The Redis channel this notifier publishes to.
    pub fn channel(&self) -> &str {
        &self.channel
    }

    /// Publish a checkpoint notification.
    ///
    /// If the notifier is disabled or the Redis connection fails, the error
    /// is returned but should generally be treated as non-fatal by callers
    /// (the checkpoint was already saved to disk).
    #[cfg(feature = "redis")]
    pub async fn publish(&self, notification: &CheckpointNotification) -> anyhow::Result<()> {
        if !self.config.enabled {
            tracing::debug!("Checkpoint notification disabled — skipping publish");
            return Ok(());
        }

        let json = notification.to_json()?;

        let client = redis::Client::open(self.config.redis_url.as_str())
            .map_err(|e| anyhow::anyhow!("Redis client error: {e}"))?;

        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| anyhow::anyhow!("Redis connection error: {e}"))?;

        redis::cmd("PUBLISH")
            .arg(&self.channel)
            .arg(&json)
            .query_async::<i64>(&mut conn)
            .await
            .map_err(|e| anyhow::anyhow!("Redis PUBLISH error: {e}"))?;

        tracing::info!(
            channel = %self.channel,
            model_path = %notification.model_path,
            version = notification.version,
            "Published model checkpoint notification"
        );

        Ok(())
    }

    /// Stub when the `redis` feature is disabled.
    #[cfg(not(feature = "redis"))]
    pub async fn publish(&self, _notification: &CheckpointNotification) -> anyhow::Result<()> {
        tracing::warn!("Redis feature not enabled — checkpoint notification not published");
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_channel_format() {
        assert_eq!(checkpoint_channel("prod"), "fks:prod:model_checkpoints");
        assert_eq!(
            checkpoint_channel("default"),
            "fks:default:model_checkpoints"
        );
    }

    #[test]
    fn test_notification_new() {
        let n = CheckpointNotification::new("checkpoints/backward/latest_model.bin", "lstm_dqn_v1");
        assert_eq!(n.model_path, "checkpoints/backward/latest_model.bin");
        assert_eq!(n.model_name, "lstm_dqn_v1");
        assert_eq!(n.version, 0);
        assert_eq!(n.training_step, 0);
        assert!(n.metadata.is_empty());
        assert!(!n.saved_at.is_empty());
    }

    #[test]
    fn test_notification_builder() {
        let n = CheckpointNotification::new("model.bin", "test")
            .with_version(42)
            .with_training_step(1000)
            .with_metadata("loss", "0.0023")
            .with_metadata("mean_q", "1.45");

        assert_eq!(n.version, 42);
        assert_eq!(n.training_step, 1000);
        assert_eq!(n.metadata.get("loss").unwrap(), "0.0023");
        assert_eq!(n.metadata.get("mean_q").unwrap(), "1.45");
    }

    #[test]
    fn test_notification_serde_round_trip() {
        let original = CheckpointNotification::new("path/to/model.bin", "lstm_v2")
            .with_version(7)
            .with_training_step(5000)
            .with_metadata("lr", "3e-4");

        let json = original.to_json().unwrap();
        let parsed = CheckpointNotification::from_json(&json).unwrap();

        assert_eq!(parsed.model_path, original.model_path);
        assert_eq!(parsed.model_name, original.model_name);
        assert_eq!(parsed.version, 7);
        assert_eq!(parsed.training_step, 5000);
        assert_eq!(parsed.saved_at, original.saved_at);
        assert_eq!(parsed.metadata.get("lr").unwrap(), "3e-4");
    }

    #[test]
    fn test_notification_json_contains_expected_fields() {
        let n = CheckpointNotification::new("model.bin", "test_model").with_version(1);

        let json = n.to_json().unwrap();
        assert!(json.contains("model_path"));
        assert!(json.contains("model_name"));
        assert!(json.contains("version"));
        assert!(json.contains("saved_at"));
        assert!(json.contains("training_step"));
        assert!(json.contains("model.bin"));
        assert!(json.contains("test_model"));
    }

    #[test]
    fn test_notifier_config_default() {
        let config = CheckpointNotifierConfig::default();
        assert_eq!(config.redis_url, "redis://localhost:6379");
        assert_eq!(config.instance_id, "default");
        assert!(config.enabled);
    }

    #[test]
    fn test_notifier_channel() {
        let notifier = CheckpointNotifier::new(CheckpointNotifierConfig::default());
        assert_eq!(notifier.channel(), "fks:default:model_checkpoints");
    }

    #[test]
    fn test_notifier_custom_instance() {
        let config = CheckpointNotifierConfig {
            instance_id: "staging".to_string(),
            ..Default::default()
        };
        let notifier = CheckpointNotifier::new(config);
        assert_eq!(notifier.channel(), "fks:staging:model_checkpoints");
    }

    #[test]
    fn test_notification_deserialize_with_missing_metadata() {
        // Simulate a JSON payload without the optional `metadata` field
        let json = r#"{
            "model_path": "model.bin",
            "model_name": "test",
            "version": 1,
            "saved_at": "2025-01-01T00:00:00Z",
            "training_step": 100
        }"#;

        let parsed = CheckpointNotification::from_json(json).unwrap();
        assert_eq!(parsed.model_path, "model.bin");
        assert!(parsed.metadata.is_empty());
    }
}
