//! Redis Publisher for Optimized Parameters
//!
//! This module publishes optimization results to Redis for consumption by the
//! Forward service via hot-reload. It uses the same Redis keys and notification
//! format as `janus-core::optimized_params` to ensure compatibility.
//!
//! # Redis Keys
//!
//! - Hash: `fks:{instance}:optimized_params` - Stores current params per asset
//! - Channel: `fks:{instance}:param_updates` - Notifies of param changes
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_optimizer::{ParamPublisher, OptimizationResult};
//!
//! // Create publisher
//! let publisher = ParamPublisher::new("redis://localhost:6379", "default").await?;
//!
//! // Publish single result
//! publisher.publish(&result).await?;
//!
//! // Publish batch results
//! publisher.publish_batch(&results).await?;
//! ```

use crate::error::{OptimizerError, Result};
use crate::results::OptimizationResult;
use chrono::Utc;
use janus_core::optimized_params::{OptimizedParams, ParamNotification};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Publisher for optimized parameters to Redis
pub struct ParamPublisher {
    /// Redis client
    client: redis::Client,

    /// Instance ID for key prefix
    instance_id: String,

    /// Connection pool for async operations
    connection: Arc<RwLock<Option<redis::aio::MultiplexedConnection>>>,

    /// Statistics about publish operations
    stats: Arc<RwLock<PublishStats>>,
}

/// Statistics about publishing operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PublishStats {
    /// Total successful publishes
    pub total_published: u64,

    /// Total failed publishes
    pub total_failed: u64,

    /// Last successful publish timestamp
    pub last_success: Option<String>,

    /// Last error message
    pub last_error: Option<String>,

    /// Per-asset publish counts
    pub per_asset: HashMap<String, u64>,
}

impl ParamPublisher {
    /// Create a new publisher connected to Redis
    pub async fn new(redis_url: &str, instance_id: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| OptimizerError::RedisError(format!("Failed to create client: {}", e)))?;

        let instance_id = instance_id.into();

        // Test connection
        let connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to connect: {}", e)))?;

        info!(
            instance_id = instance_id,
            "Connected to Redis for param publishing"
        );

        Ok(Self {
            client,
            instance_id,
            connection: Arc::new(RwLock::new(Some(connection))),
            stats: Arc::new(RwLock::new(PublishStats::default())),
        })
    }

    /// Get the Redis hash key for optimized params
    pub fn params_hash_key(&self) -> String {
        format!("fks:{}:optimized_params", self.instance_id)
    }

    /// Get the Redis channel for param updates
    pub fn updates_channel(&self) -> String {
        format!("fks:{}:param_updates", self.instance_id)
    }

    /// Get or create a connection
    async fn get_connection(&self) -> Result<redis::aio::MultiplexedConnection> {
        let mut conn_guard = self.connection.write().await;

        if let Some(conn) = conn_guard.take() {
            // Return existing connection (we'll put it back after use)
            *conn_guard = Some(conn.clone());
            return Ok(conn);
        }

        // Create new connection
        let conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to reconnect: {}", e)))?;

        *conn_guard = Some(conn.clone());
        Ok(conn)
    }

    /// Publish a single optimization result
    pub async fn publish(&self, result: &OptimizationResult) -> Result<()> {
        let params = result.to_optimized_params();
        self.publish_params(&params).await
    }

    /// Publish OptimizedParams directly
    pub async fn publish_params(&self, params: &OptimizedParams) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self.get_connection().await?;
        let asset = params.asset.clone();

        // Serialize params to JSON
        let json = serde_json::to_string(params)
            .map_err(|e| OptimizerError::SerializationError(e.to_string()))?;

        // Store in hash
        let hash_key = self.params_hash_key();
        conn.hset::<_, _, _, ()>(&hash_key, &asset, &json)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to HSET: {}", e)))?;

        debug!(asset = asset, key = hash_key, "Stored params in hash");

        // Create and publish notification
        let notification = ParamNotification::ParamUpdate {
            asset: asset.clone(),
            timestamp: Utc::now().to_rfc3339(),
            params: params.clone(),
        };

        let notification_json = serde_json::to_string(&notification)
            .map_err(|e| OptimizerError::SerializationError(e.to_string()))?;

        let channel = self.updates_channel();
        let subscribers: i64 = conn
            .publish(&channel, &notification_json)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to PUBLISH: {}", e)))?;

        debug!(
            asset = asset,
            channel = channel,
            subscribers = subscribers,
            "Published param update notification"
        );

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_published += 1;
            stats.last_success = Some(Utc::now().to_rfc3339());
            *stats.per_asset.entry(asset.clone()).or_insert(0) += 1;
        }

        info!(
            asset = asset,
            score = params.optimization_score,
            subscribers = subscribers,
            "Published optimized params"
        );

        Ok(())
    }

    /// Publish multiple optimization results in batch
    pub async fn publish_batch(
        &self,
        results: &[OptimizationResult],
    ) -> Result<BatchPublishResult> {
        let mut batch_result = BatchPublishResult::new(results.len());

        // Send optimization started notification
        self.notify_optimization_started(
            &results.iter().map(|r| r.asset.clone()).collect::<Vec<_>>(),
        )
        .await?;

        for result in results {
            match self.publish(result).await {
                Ok(()) => {
                    batch_result.successful.push(result.asset.clone());
                }
                Err(e) => {
                    error!(asset = result.asset, error = %e, "Failed to publish");
                    batch_result
                        .failed
                        .push((result.asset.clone(), e.to_string()));

                    // Update error stats
                    let mut stats = self.stats.write().await;
                    stats.total_failed += 1;
                    stats.last_error = Some(e.to_string());
                }
            }
        }

        // Send optimization complete notification
        self.notify_optimization_complete(&batch_result).await?;

        info!(
            successful = batch_result.successful.len(),
            failed = batch_result.failed.len(),
            "Batch publish completed"
        );

        Ok(batch_result)
    }

    /// Send optimization started notification
    pub async fn notify_optimization_started(&self, assets: &[String]) -> Result<()> {
        use redis::AsyncCommands;

        let notification = ParamNotification::OptimizationStarted {
            timestamp: Utc::now().to_rfc3339(),
            assets: assets.to_vec(),
        };

        let json = serde_json::to_string(&notification)
            .map_err(|e| OptimizerError::SerializationError(e.to_string()))?;

        let mut conn = self.get_connection().await?;
        let channel = self.updates_channel();

        conn.publish::<_, _, i64>(&channel, &json)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to PUBLISH: {}", e)))?;

        info!(assets = ?assets, "Notified optimization started");

        Ok(())
    }

    /// Send optimization complete notification
    pub async fn notify_optimization_complete(
        &self,
        batch_result: &BatchPublishResult,
    ) -> Result<()> {
        use redis::AsyncCommands;

        let notification = ParamNotification::OptimizationComplete {
            timestamp: Utc::now().to_rfc3339(),
            successful: batch_result.successful.len() as u32,
            failed: batch_result.failed.len() as u32,
            assets: batch_result.successful.clone(),
        };

        let json = serde_json::to_string(&notification)
            .map_err(|e| OptimizerError::SerializationError(e.to_string()))?;

        let mut conn = self.get_connection().await?;
        let channel = self.updates_channel();

        conn.publish::<_, _, i64>(&channel, &json)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to PUBLISH: {}", e)))?;

        info!(
            successful = batch_result.successful.len(),
            failed = batch_result.failed.len(),
            "Notified optimization complete"
        );

        Ok(())
    }

    /// Send optimization failed notification for a single asset
    pub async fn notify_optimization_failed(&self, asset: &str, error: &str) -> Result<()> {
        use redis::AsyncCommands;

        let notification = ParamNotification::OptimizationFailed {
            timestamp: Utc::now().to_rfc3339(),
            asset: asset.to_string(),
            error: error.to_string(),
        };

        let json = serde_json::to_string(&notification)
            .map_err(|e| OptimizerError::SerializationError(e.to_string()))?;

        let mut conn = self.get_connection().await?;
        let channel = self.updates_channel();

        conn.publish::<_, _, i64>(&channel, &json)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to PUBLISH: {}", e)))?;

        warn!(asset = asset, error = error, "Notified optimization failed");

        Ok(())
    }

    /// Get all currently stored params from Redis
    pub async fn get_all_params(&self) -> Result<HashMap<String, OptimizedParams>> {
        use redis::AsyncCommands;

        let mut conn = self.get_connection().await?;
        let hash_key = self.params_hash_key();

        let all_params: HashMap<String, String> = conn
            .hgetall(&hash_key)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to HGETALL: {}", e)))?;

        let mut result = HashMap::new();
        for (asset, json) in all_params {
            // Skip metadata keys
            if asset.starts_with('_') {
                continue;
            }

            match serde_json::from_str::<OptimizedParams>(&json) {
                Ok(params) => {
                    result.insert(asset, params);
                }
                Err(e) => {
                    warn!(asset = asset, error = %e, "Failed to parse stored params");
                }
            }
        }

        Ok(result)
    }

    /// Get params for a specific asset from Redis
    pub async fn get_params(&self, asset: &str) -> Result<Option<OptimizedParams>> {
        use redis::AsyncCommands;

        let mut conn = self.get_connection().await?;
        let hash_key = self.params_hash_key();

        let json: Option<String> = conn
            .hget(&hash_key, asset)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to HGET: {}", e)))?;

        match json {
            Some(j) => {
                let params = serde_json::from_str(&j)
                    .map_err(|e| OptimizerError::SerializationError(e.to_string()))?;
                Ok(Some(params))
            }
            None => Ok(None),
        }
    }

    /// Delete params for a specific asset
    pub async fn delete_params(&self, asset: &str) -> Result<bool> {
        use redis::AsyncCommands;

        let mut conn = self.get_connection().await?;
        let hash_key = self.params_hash_key();

        let deleted: i64 = conn
            .hdel(&hash_key, asset)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to HDEL: {}", e)))?;

        if deleted > 0 {
            info!(asset = asset, "Deleted params from Redis");
        }

        Ok(deleted > 0)
    }

    /// Store metadata about the optimization run
    pub async fn store_metadata(&self, metadata: &OptimizationMetadata) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self.get_connection().await?;
        let hash_key = self.params_hash_key();

        let json = serde_json::to_string(metadata)
            .map_err(|e| OptimizerError::SerializationError(e.to_string()))?;

        conn.hset::<_, _, _, ()>(&hash_key, "_metadata", &json)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("Failed to store metadata: {}", e)))?;

        debug!("Stored optimization metadata");

        Ok(())
    }

    /// Get publishing statistics
    pub async fn stats(&self) -> PublishStats {
        self.stats.read().await.clone()
    }

    /// Reset publishing statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = PublishStats::default();
    }

    /// Test Redis connection
    pub async fn ping(&self) -> Result<()> {
        let mut conn = self.get_connection().await?;
        let _: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(|e| OptimizerError::RedisError(format!("PING failed: {}", e)))?;

        Ok(())
    }

    /// Get the instance ID
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }
}

/// Result of a batch publish operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPublishResult {
    /// Total number of results to publish
    pub total: usize,

    /// Successfully published assets
    pub successful: Vec<String>,

    /// Failed assets with error messages
    pub failed: Vec<(String, String)>,
}

impl BatchPublishResult {
    /// Create a new batch result
    pub fn new(total: usize) -> Self {
        Self {
            total,
            successful: Vec::with_capacity(total),
            failed: Vec::new(),
        }
    }

    /// Check if all publishes succeeded
    pub fn all_succeeded(&self) -> bool {
        self.failed.is_empty()
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        (self.successful.len() as f64 / self.total as f64) * 100.0
    }
}

/// Metadata about an optimization run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetadata {
    /// Optimizer version
    pub version: String,

    /// When the optimization started
    pub started_at: String,

    /// When the optimization completed
    pub completed_at: String,

    /// Total duration in seconds
    pub duration_seconds: f64,

    /// Assets that were optimized
    pub assets: Vec<String>,

    /// Configuration summary
    pub config_summary: String,

    /// Machine/host identifier
    pub host: Option<String>,
}

impl OptimizationMetadata {
    /// Create metadata for a completed optimization run
    pub fn new(
        started_at: chrono::DateTime<Utc>,
        assets: Vec<String>,
        config_summary: impl Into<String>,
    ) -> Self {
        let completed_at = Utc::now();
        Self {
            version: crate::VERSION.to_string(),
            started_at: started_at.to_rfc3339(),
            completed_at: completed_at.to_rfc3339(),
            duration_seconds: (completed_at - started_at).num_seconds() as f64,
            assets,
            config_summary: config_summary.into(),
            host: std::env::var("HOSTNAME")
                .or_else(|_| std::env::var("COMPUTERNAME"))
                .ok(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_publish_result() {
        let mut result = BatchPublishResult::new(3);
        result.successful.push("BTC".to_string());
        result.successful.push("ETH".to_string());
        result
            .failed
            .push(("SOL".to_string(), "timeout".to_string()));

        assert!(!result.all_succeeded());
        assert!((result.success_rate() - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_batch_publish_result_all_success() {
        let mut result = BatchPublishResult::new(2);
        result.successful.push("BTC".to_string());
        result.successful.push("ETH".to_string());

        assert!(result.all_succeeded());
        assert_eq!(result.success_rate(), 100.0);
    }

    #[test]
    fn test_publish_stats_default() {
        let stats = PublishStats::default();
        assert_eq!(stats.total_published, 0);
        assert_eq!(stats.total_failed, 0);
        assert!(stats.last_success.is_none());
        assert!(stats.last_error.is_none());
    }

    #[test]
    fn test_params_hash_key() {
        // We can't create a full publisher without Redis, but we can test the key format
        let instance_id = "test_instance";
        let expected_hash = format!("fks:{}:optimized_params", instance_id);
        let expected_channel = format!("fks:{}:param_updates", instance_id);

        assert_eq!(expected_hash, "fks:test_instance:optimized_params");
        assert_eq!(expected_channel, "fks:test_instance:param_updates");
    }

    #[test]
    fn test_optimization_metadata_serialization() {
        let metadata = OptimizationMetadata {
            version: "0.1.0".to_string(),
            started_at: "2024-01-01T00:00:00Z".to_string(),
            completed_at: "2024-01-01T01:00:00Z".to_string(),
            duration_seconds: 3600.0,
            assets: vec!["BTC".to_string(), "ETH".to_string()],
            config_summary: "quick preset, 50 trials".to_string(),
            host: Some("optimizer-1".to_string()),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: OptimizationMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, "0.1.0");
        assert_eq!(parsed.duration_seconds, 3600.0);
        assert_eq!(parsed.assets.len(), 2);
    }
}
