//! Redis Cache Module for Registry
//!
//! This module provides Redis-based persistence and caching for the asset
//! and service registries. It supports:
//!
//! - Persistent storage of registry data
//! - Cache-through pattern for reads
//! - Write-through pattern for mutations
//! - TTL-based cache expiration for heartbeats
//! - Graceful fallback when Redis is unavailable
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_registry_lib::cache::{RegistryCache, CacheConfig};
//!
//! let config = CacheConfig::default();
//! let cache = RegistryCache::new(config).await?;
//!
//! // Cache an asset
//! cache.set_asset(&asset).await?;
//!
//! // Retrieve a cached asset
//! if let Some(asset) = cache.get_asset("btc-usdt").await? {
//!     println!("Found asset: {}", asset.name);
//! }
//! ```

use crate::{Asset, AssetStatus, ServiceHealth, ServiceInstance};
use chrono::Utc;
use redis::{AsyncCommands, Client, aio::ConnectionManager};
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Cache error types
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("Redis connection error: {0}")]
    ConnectionError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Cache operation failed: {0}")]
    OperationFailed(String),

    #[error("Cache disabled")]
    Disabled,

    #[error("Key not found: {0}")]
    KeyNotFound(String),
}

pub type Result<T> = std::result::Result<T, CacheError>;

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Redis connection URL
    pub redis_url: String,

    /// Key prefix for all cache keys
    pub key_prefix: String,

    /// Default TTL for cached items (in seconds)
    pub default_ttl: u64,

    /// TTL for service heartbeats (in seconds)
    pub heartbeat_ttl: u64,

    /// TTL for asset data (in seconds)
    pub asset_ttl: u64,

    /// Enable caching
    pub enabled: bool,

    /// Connection timeout (in milliseconds)
    pub connection_timeout_ms: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            key_prefix: "janus:registry:".to_string(),
            default_ttl: 3600, // 1 hour
            heartbeat_ttl: 60, // 1 minute
            asset_ttl: 86400,  // 24 hours
            enabled: true,
            connection_timeout_ms: 5000,
        }
    }
}

impl CacheConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(url) = std::env::var("REDIS_URL") {
            config.redis_url = url;
        }

        if let Ok(prefix) = std::env::var("REDIS_KEY_PREFIX") {
            config.key_prefix = prefix;
        }

        if let Ok(ttl) = std::env::var("REDIS_DEFAULT_TTL")
            && let Ok(ttl) = ttl.parse()
        {
            config.default_ttl = ttl;
        }

        if let Ok(enabled) = std::env::var("REDIS_ENABLED") {
            config.enabled = enabled.to_lowercase() != "false" && enabled != "0";
        }

        config
    }

    /// Format a cache key with prefix
    pub fn format_key(&self, key: &str) -> String {
        format!("{}{}", self.key_prefix, key)
    }
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: AtomicU64,
    /// Total cache misses
    pub misses: AtomicU64,
    /// Total cache sets
    pub sets: AtomicU64,
    /// Total cache deletes
    pub deletes: AtomicU64,
    /// Total errors
    pub errors: AtomicU64,
}

impl CacheStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_set(&self) {
        self.sets.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_delete(&self) {
        self.deletes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get a snapshot of stats
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            sets: self.sets.load(Ordering::Relaxed),
            deletes: self.deletes.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            hit_rate: self.hit_rate(),
        }
    }
}

/// Snapshot of cache statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub sets: u64,
    pub deletes: u64,
    pub errors: u64,
    pub hit_rate: f64,
}

/// Registry cache with Redis persistence
pub struct RegistryCache {
    config: CacheConfig,
    client: Option<Client>,
    connection: RwLock<Option<ConnectionManager>>,
    stats: Arc<CacheStats>,
    connected: AtomicBool,
}

impl RegistryCache {
    /// Create a new registry cache
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let client = if config.enabled {
            Some(
                Client::open(config.redis_url.as_str())
                    .map_err(|e| CacheError::ConnectionError(e.to_string()))?,
            )
        } else {
            None
        };

        let cache = Self {
            config: config.clone(),
            client,
            connection: RwLock::new(None),
            stats: Arc::new(CacheStats::new()),
            connected: AtomicBool::new(false),
        };

        if config.enabled
            && let Err(e) = cache.connect().await
        {
            warn!("Redis connection failed, cache will be disabled: {}", e);
        }

        Ok(cache)
    }

    /// Connect to Redis
    pub async fn connect(&self) -> Result<()> {
        if !self.config.enabled {
            return Err(CacheError::Disabled);
        }

        let Some(ref client) = self.client else {
            return Err(CacheError::ConnectionError("No Redis client".to_string()));
        };

        let connection = ConnectionManager::new(client.clone())
            .await
            .map_err(|e| CacheError::ConnectionError(e.to_string()))?;

        *self.connection.write().await = Some(connection);
        self.connected.store(true, Ordering::SeqCst);

        info!("Connected to Redis at {}", self.config.redis_url);
        Ok(())
    }

    /// Check if connected to Redis
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Get cache statistics
    pub fn stats(&self) -> Arc<CacheStats> {
        Arc::clone(&self.stats)
    }

    /// Get stats snapshot
    pub fn stats_snapshot(&self) -> CacheStatsSnapshot {
        self.stats.snapshot()
    }

    // ========================================================================
    // Generic cache operations
    // ========================================================================

    /// Get a value from cache
    async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let redis_key = self.config.format_key(key);

        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            match conn.get::<_, Option<String>>(&redis_key).await {
                Ok(Some(data)) => {
                    self.stats.record_hit();
                    let value: T = serde_json::from_str(&data)
                        .map_err(|e| CacheError::DeserializationError(e.to_string()))?;
                    debug!("Cache hit for key: {}", redis_key);
                    return Ok(Some(value));
                }
                Ok(None) => {
                    self.stats.record_miss();
                    debug!("Cache miss for key: {}", redis_key);
                    return Ok(None);
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis GET error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        self.stats.record_miss();
        Ok(None)
    }

    /// Set a value in cache with TTL
    async fn set<T: Serialize>(&self, key: &str, value: &T, ttl_secs: u64) -> Result<()> {
        let redis_key = self.config.format_key(key);
        let data = serde_json::to_string(value)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            match conn.set_ex::<_, _, ()>(&redis_key, &data, ttl_secs).await {
                Ok(()) => {
                    self.stats.record_set();
                    debug!("Cached value for key: {} (TTL: {}s)", redis_key, ttl_secs);
                    return Ok(());
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis SET error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        Ok(())
    }

    /// Set a value in cache without TTL (persistent)
    #[allow(dead_code)]
    async fn set_persistent<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let redis_key = self.config.format_key(key);
        let data = serde_json::to_string(value)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            match conn.set::<_, _, ()>(&redis_key, &data).await {
                Ok(()) => {
                    self.stats.record_set();
                    debug!("Cached persistent value for key: {}", redis_key);
                    return Ok(());
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis SET error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        Ok(())
    }

    /// Delete a value from cache
    async fn delete(&self, key: &str) -> Result<bool> {
        let redis_key = self.config.format_key(key);

        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            match conn.del::<_, i64>(&redis_key).await {
                Ok(count) => {
                    if count > 0 {
                        self.stats.record_delete();
                        debug!("Deleted key: {}", redis_key);
                    }
                    return Ok(count > 0);
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis DEL error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        Ok(false)
    }

    /// Check if a key exists
    async fn exists(&self, key: &str) -> Result<bool> {
        let redis_key = self.config.format_key(key);

        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            match conn.exists::<_, bool>(&redis_key).await {
                Ok(exists) => {
                    return Ok(exists);
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis EXISTS error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        Ok(false)
    }

    /// Get all keys matching a pattern
    async fn keys(&self, pattern: &str) -> Result<Vec<String>> {
        let redis_pattern = self.config.format_key(pattern);

        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            match conn.keys::<_, Vec<String>>(&redis_pattern).await {
                Ok(keys) => {
                    // Strip prefix from keys
                    let prefix_len = self.config.key_prefix.len();
                    let stripped: Vec<String> = keys
                        .into_iter()
                        .map(|k| k[prefix_len..].to_string())
                        .collect();
                    return Ok(stripped);
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis KEYS error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        Ok(Vec::new())
    }

    // ========================================================================
    // Asset operations
    // ========================================================================

    /// Cache an asset
    pub async fn set_asset(&self, asset: &Asset) -> Result<()> {
        let key = format!("asset:{}", asset.id);
        self.set(&key, asset, self.config.asset_ttl).await?;

        // Also index by symbol
        let symbol_key = format!("asset:symbol:{}", asset.symbol.to_uppercase());
        self.set(&symbol_key, &asset.id, self.config.asset_ttl)
            .await?;

        Ok(())
    }

    /// Get a cached asset by ID
    pub async fn get_asset(&self, id: &str) -> Result<Option<Asset>> {
        let key = format!("asset:{}", id);
        self.get(&key).await
    }

    /// Get a cached asset by symbol
    pub async fn get_asset_by_symbol(&self, symbol: &str) -> Result<Option<Asset>> {
        let symbol_key = format!("asset:symbol:{}", symbol.to_uppercase());

        // First get the ID from symbol index
        if let Some(id) = self.get::<String>(&symbol_key).await? {
            return self.get_asset(&id).await;
        }

        Ok(None)
    }

    /// Delete a cached asset
    pub async fn delete_asset(&self, id: &str, symbol: &str) -> Result<()> {
        let key = format!("asset:{}", id);
        self.delete(&key).await?;

        let symbol_key = format!("asset:symbol:{}", symbol.to_uppercase());
        self.delete(&symbol_key).await?;

        Ok(())
    }

    /// Get all cached asset IDs
    pub async fn list_asset_ids(&self) -> Result<Vec<String>> {
        let keys = self.keys("asset:*").await?;

        // Filter to only asset IDs (not symbol mappings)
        let ids: Vec<String> = keys
            .into_iter()
            .filter(|k| k.starts_with("asset:") && !k.contains(":symbol:"))
            .map(|k| k.strip_prefix("asset:").unwrap_or(&k).to_string())
            .collect();

        Ok(ids)
    }

    /// Update asset status in cache
    pub async fn update_asset_status(&self, id: &str, status: AssetStatus) -> Result<()> {
        if let Some(mut asset) = self.get_asset(id).await? {
            asset.status = status;
            asset.updated_at = Utc::now();
            self.set_asset(&asset).await?;
        }
        Ok(())
    }

    // ========================================================================
    // Service operations
    // ========================================================================

    /// Cache a service instance
    pub async fn set_service(&self, service: &ServiceInstance) -> Result<()> {
        let key = format!("service:{}", service.id);
        self.set(&key, service, self.config.default_ttl).await?;

        // Add to type index (as a set member)
        let type_key = format!("service:type:{:?}", service.service_type);
        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            let redis_key = self.config.format_key(&type_key);
            let _ = conn.sadd::<_, _, ()>(&redis_key, &service.id).await;
        }

        Ok(())
    }

    /// Get a cached service by ID
    pub async fn get_service(&self, id: &str) -> Result<Option<ServiceInstance>> {
        let key = format!("service:{}", id);
        self.get(&key).await
    }

    /// Delete a cached service
    pub async fn delete_service(&self, id: &str, service_type: &str) -> Result<()> {
        let key = format!("service:{}", id);
        self.delete(&key).await?;

        // Remove from type index
        let type_key = format!("service:type:{}", service_type);
        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            let redis_key = self.config.format_key(&type_key);
            let _ = conn.srem::<_, _, ()>(&redis_key, id).await;
        }

        Ok(())
    }

    /// Get all service IDs of a given type
    pub async fn list_services_by_type(&self, service_type: &str) -> Result<Vec<String>> {
        let type_key = format!("service:type:{}", service_type);

        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            let redis_key = self.config.format_key(&type_key);
            match conn.smembers::<_, Vec<String>>(&redis_key).await {
                Ok(ids) => return Ok(ids),
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis SMEMBERS error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        Ok(Vec::new())
    }

    /// Update service health in cache
    pub async fn update_service_health(&self, id: &str, health: ServiceHealth) -> Result<()> {
        if let Some(mut service) = self.get_service(id).await? {
            service.health = health;
            service.last_health_check = Some(Utc::now());
            service.last_heartbeat = Utc::now();
            self.set_service(&service).await?;
        }
        Ok(())
    }

    /// Update service heartbeat
    pub async fn service_heartbeat(&self, id: &str) -> Result<()> {
        let heartbeat_key = format!("service:heartbeat:{}", id);
        let now = Utc::now();
        self.set(&heartbeat_key, &now, self.config.heartbeat_ttl)
            .await?;

        // Also update the service's last_heartbeat field
        if let Some(mut service) = self.get_service(id).await? {
            service.last_heartbeat = now;
            self.set_service(&service).await?;
        }

        Ok(())
    }

    /// Check if a service has a recent heartbeat
    pub async fn has_recent_heartbeat(&self, id: &str) -> Result<bool> {
        let heartbeat_key = format!("service:heartbeat:{}", id);
        self.exists(&heartbeat_key).await
    }

    /// Get all stale service IDs (no recent heartbeat)
    pub async fn get_stale_services(&self) -> Result<Vec<String>> {
        let keys = self.keys("service:*").await?;
        let mut stale = Vec::new();

        for key in keys {
            if key.starts_with("service:")
                && !key.contains(":type:")
                && !key.contains(":heartbeat:")
            {
                let id = key.strip_prefix("service:").unwrap_or(&key);
                if !self.has_recent_heartbeat(id).await? {
                    stale.push(id.to_string());
                }
            }
        }

        Ok(stale)
    }

    // ========================================================================
    // Utility operations
    // ========================================================================

    /// Clear all cached data
    pub async fn clear(&self) -> Result<()> {
        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            let pattern = format!("{}*", self.config.key_prefix);
            let keys: Vec<String> = conn
                .keys(&pattern)
                .await
                .map_err(|e| CacheError::OperationFailed(e.to_string()))?;

            if !keys.is_empty() {
                conn.del::<_, ()>(&keys)
                    .await
                    .map_err(|e| CacheError::OperationFailed(e.to_string()))?;

                info!("Cleared {} keys from cache", keys.len());
            }
        }

        Ok(())
    }

    /// Health check
    pub async fn health_check(&self) -> bool {
        if !self.config.enabled {
            return true; // Cache disabled is still healthy
        }

        if let Some(ref conn) = *self.connection.read().await {
            let mut conn = conn.clone();
            match redis::cmd("PING").query_async::<String>(&mut conn).await {
                Ok(response) if response == "PONG" => {
                    self.connected.store(true, Ordering::SeqCst);
                    return true;
                }
                _ => {
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        false
    }

    /// Reconnect if disconnected
    pub async fn reconnect_if_needed(&self) -> Result<()> {
        if !self.is_connected() && self.config.enabled {
            self.connect().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.redis_url, "redis://localhost:6379");
        assert!(config.enabled);
        assert_eq!(config.default_ttl, 3600);
    }

    #[test]
    fn test_cache_config_format_key() {
        let config = CacheConfig::default();
        assert_eq!(config.format_key("test"), "janus:registry:test");
    }

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats::new();

        stats.record_hit();
        stats.record_hit();
        stats.record_miss();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.hits, 2);
        assert_eq!(snapshot.misses, 1);
        assert!((snapshot.hit_rate - 0.666).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_cache_disabled() {
        let config = CacheConfig {
            enabled: false,
            ..Default::default()
        };

        let cache = RegistryCache::new(config).await.unwrap();
        assert!(!cache.is_connected());

        // Health check should still pass when disabled
        assert!(cache.health_check().await);
    }
}
