//! Redis Cache Layer Module
//!
//! This module provides Redis-based caching for API responses and data aggregation.
//! It supports:
//! - TTL-based cache expiration
//! - Serialization/deserialization of complex types
//! - Connection pooling
//! - Graceful fallback when Redis is unavailable
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::thalamus::sources::cache::{RedisCache, CacheKey};
//!
//! let cache = RedisCache::new(redis_config).await?;
//!
//! // Cache a value
//! cache.set(CacheKey::news("bitcoin"), &news_data, None).await?;
//!
//! // Retrieve a cached value
//! if let Some(data) = cache.get::<NewsData>(CacheKey::news("bitcoin")).await? {
//!     println!("Cache hit!");
//! }
//! ```

use super::config::RedisConfig;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Cache errors
#[derive(Debug, Error)]
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

/// Cache key types for different data sources
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheKey {
    /// News articles by topic/keyword
    News(String),

    /// News sentiment aggregate
    NewsSentiment(String),

    /// Weather data by location
    Weather(String),

    /// Celestial data (moon phase, space weather)
    Celestial(String),

    /// API response by source and endpoint
    ApiResponse { source: String, endpoint: String },

    /// Aggregated external data
    AggregatedData(String),

    /// Service registry entry
    ServiceRegistry(String),

    /// Asset registry entry
    AssetRegistry(String),

    /// Custom key
    Custom(String),
}

impl CacheKey {
    /// Create a news cache key
    pub fn news(topic: impl Into<String>) -> Self {
        Self::News(topic.into())
    }

    /// Create a news sentiment cache key
    pub fn news_sentiment(symbol: impl Into<String>) -> Self {
        Self::NewsSentiment(symbol.into())
    }

    /// Create a weather cache key
    pub fn weather(location: impl Into<String>) -> Self {
        Self::Weather(location.into())
    }

    /// Create a celestial cache key
    pub fn celestial(data_type: impl Into<String>) -> Self {
        Self::Celestial(data_type.into())
    }

    /// Create an API response cache key
    pub fn api_response(source: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self::ApiResponse {
            source: source.into(),
            endpoint: endpoint.into(),
        }
    }

    /// Create an aggregated data cache key
    pub fn aggregated(id: impl Into<String>) -> Self {
        Self::AggregatedData(id.into())
    }

    /// Create a service registry cache key
    pub fn service(service_id: impl Into<String>) -> Self {
        Self::ServiceRegistry(service_id.into())
    }

    /// Create an asset registry cache key
    pub fn asset(asset_id: impl Into<String>) -> Self {
        Self::AssetRegistry(asset_id.into())
    }

    /// Create a custom cache key
    pub fn custom(key: impl Into<String>) -> Self {
        Self::Custom(key.into())
    }

    /// Convert to string representation for Redis
    pub fn to_redis_key(&self, prefix: &str) -> String {
        match self {
            Self::News(topic) => format!("{}news:{}", prefix, topic),
            Self::NewsSentiment(symbol) => format!("{}sentiment:{}", prefix, symbol),
            Self::Weather(location) => format!("{}weather:{}", prefix, location),
            Self::Celestial(data_type) => format!("{}celestial:{}", prefix, data_type),
            Self::ApiResponse { source, endpoint } => {
                format!("{}api:{}:{}", prefix, source, endpoint)
            }
            Self::AggregatedData(id) => format!("{}aggregated:{}", prefix, id),
            Self::ServiceRegistry(service_id) => format!("{}service:{}", prefix, service_id),
            Self::AssetRegistry(asset_id) => format!("{}asset:{}", prefix, asset_id),
            Self::Custom(key) => format!("{}{}", prefix, key),
        }
    }
}

/// Cached entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    /// The cached value
    pub value: T,

    /// When the entry was created
    pub created_at: DateTime<Utc>,

    /// When the entry expires (if set)
    pub expires_at: Option<DateTime<Utc>>,

    /// Source of the data
    pub source: Option<String>,

    /// Cache hit count
    pub hits: u64,
}

impl<T> CacheEntry<T> {
    /// Create a new cache entry
    pub fn new(value: T) -> Self {
        Self {
            value,
            created_at: Utc::now(),
            expires_at: None,
            source: None,
            hits: 0,
        }
    }

    /// Create with TTL
    pub fn with_ttl(value: T, ttl: Duration) -> Self {
        let created_at = Utc::now();
        let expires_at = created_at + chrono::Duration::from_std(ttl).unwrap_or_default();
        Self {
            value,
            created_at,
            expires_at: Some(expires_at),
            source: None,
            hits: 0,
        }
    }

    /// Set the source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Check if the entry is expired
    pub fn is_expired(&self) -> bool {
        self.expires_at.map(|exp| Utc::now() > exp).unwrap_or(false)
    }

    /// Get remaining TTL
    pub fn remaining_ttl(&self) -> Option<Duration> {
        self.expires_at.and_then(|exp| {
            let now = Utc::now();
            if exp > now {
                (exp - now).to_std().ok()
            } else {
                None
            }
        })
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
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cache hit
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache set
    pub fn record_set(&self) {
        self.sets.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache delete
    pub fn record_delete(&self) {
        self.deletes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an error
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get hit rate
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

    /// Get snapshot of stats
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub sets: u64,
    pub deletes: u64,
    pub errors: u64,
    pub hit_rate: f64,
}

/// In-memory cache for fallback when Redis is unavailable
#[derive(Debug)]
struct InMemoryCache {
    data: RwLock<std::collections::HashMap<String, (Vec<u8>, Option<std::time::Instant>)>>,
    max_entries: usize,
}

impl InMemoryCache {
    fn new(max_entries: usize) -> Self {
        Self {
            data: RwLock::new(std::collections::HashMap::new()),
            max_entries,
        }
    }

    async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let data = self.data.read().await;
        data.get(key).and_then(|(value, expires)| {
            if expires
                .map(|e| e > std::time::Instant::now())
                .unwrap_or(true)
            {
                Some(value.clone())
            } else {
                None
            }
        })
    }

    async fn set(&self, key: String, value: Vec<u8>, ttl: Option<Duration>) {
        let mut data = self.data.write().await;

        // Evict oldest entries if at capacity
        if data.len() >= self.max_entries {
            // Simple eviction: remove first entry (not ideal, but simple)
            if let Some(first_key) = data.keys().next().cloned() {
                data.remove(&first_key);
            }
        }

        let expires = ttl.map(|t| std::time::Instant::now() + t);
        data.insert(key, (value, expires));
    }

    async fn delete(&self, key: &str) -> bool {
        let mut data = self.data.write().await;
        data.remove(key).is_some()
    }

    async fn clear(&self) {
        let mut data = self.data.write().await;
        data.clear();
    }

    async fn len(&self) -> usize {
        self.data.read().await.len()
    }
}

/// Redis cache client with fallback to in-memory cache
pub struct RedisCache {
    config: RedisConfig,
    /// Redis client (reserved for connection management)
    #[allow(dead_code)]
    client: Option<redis::Client>,
    connection: RwLock<Option<redis::aio::MultiplexedConnection>>,
    fallback: InMemoryCache,
    stats: Arc<CacheStats>,
    connected: AtomicBool,
}

impl RedisCache {
    /// Create a new Redis cache
    pub async fn new(config: RedisConfig) -> Result<Self> {
        let cache = Self {
            config: config.clone(),
            client: None,
            connection: RwLock::new(None),
            fallback: InMemoryCache::new(10000), // 10k entries max
            stats: Arc::new(CacheStats::new()),
            connected: AtomicBool::new(false),
        };

        if config.enabled {
            // Try to connect, but don't fail if Redis is unavailable
            if let Err(e) = cache.connect().await {
                warn!("Redis connection failed, using in-memory fallback: {}", e);
            }
        }

        Ok(cache)
    }

    /// Connect to Redis
    pub async fn connect(&self) -> Result<()> {
        if !self.config.enabled {
            return Err(CacheError::Disabled);
        }

        let client = redis::Client::open(self.config.url.as_str())
            .map_err(|e| CacheError::ConnectionError(e.to_string()))?;

        let connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| CacheError::ConnectionError(e.to_string()))?;

        *self.connection.write().await = Some(connection);
        self.connected.store(true, Ordering::SeqCst);

        info!("Connected to Redis at {}", self.config.url);
        Ok(())
    }

    /// Check if connected to Redis
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Get a value from cache
    pub async fn get<T: DeserializeOwned>(&self, key: CacheKey) -> Result<Option<T>> {
        let redis_key = key.to_redis_key(&self.config.key_prefix);

        // Try Redis first
        if let Some(ref mut conn) = *self.connection.write().await {
            match redis::cmd("GET")
                .arg(&redis_key)
                .query_async::<Option<Vec<u8>>>(conn)
                .await
            {
                Ok(Some(data)) => {
                    self.stats.record_hit();
                    let value: CacheEntry<T> = serde_json::from_slice(&data)
                        .map_err(|e| CacheError::DeserializationError(e.to_string()))?;

                    if !value.is_expired() {
                        debug!("Cache hit for key: {}", redis_key);
                        return Ok(Some(value.value));
                    }
                }
                Ok(None) => {
                    self.stats.record_miss();
                    debug!("Cache miss for key: {}", redis_key);
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis GET error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        // Fallback to in-memory cache
        if let Some(data) = self.fallback.get(&redis_key).await {
            self.stats.record_hit();
            let value: CacheEntry<T> = serde_json::from_slice(&data)
                .map_err(|e| CacheError::DeserializationError(e.to_string()))?;

            if !value.is_expired() {
                debug!("In-memory cache hit for key: {}", redis_key);
                return Ok(Some(value.value));
            }
        }

        self.stats.record_miss();
        Ok(None)
    }

    /// Set a value in cache
    pub async fn set<T: Serialize>(
        &self,
        key: CacheKey,
        value: &T,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let redis_key = key.to_redis_key(&self.config.key_prefix);
        let ttl = ttl.unwrap_or(self.config.ttl_duration());

        let entry = CacheEntry::with_ttl(value, ttl);
        let data = serde_json::to_vec(&entry)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        // Try Redis first
        if let Some(ref mut conn) = *self.connection.write().await {
            match redis::cmd("SETEX")
                .arg(&redis_key)
                .arg(ttl.as_secs() as i64)
                .arg(&data)
                .query_async::<()>(conn)
                .await
            {
                Ok(()) => {
                    self.stats.record_set();
                    debug!("Cached value for key: {} (TTL: {:?})", redis_key, ttl);
                    return Ok(());
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis SET error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        // Fallback to in-memory cache
        self.fallback.set(redis_key.clone(), data, Some(ttl)).await;
        self.stats.record_set();
        debug!(
            "Cached value in memory for key: {} (TTL: {:?})",
            redis_key, ttl
        );

        Ok(())
    }

    /// Delete a value from cache
    pub async fn delete(&self, key: CacheKey) -> Result<bool> {
        let redis_key = key.to_redis_key(&self.config.key_prefix);
        let mut deleted = false;

        // Try Redis first
        if let Some(ref mut conn) = *self.connection.write().await {
            match redis::cmd("DEL")
                .arg(&redis_key)
                .query_async::<i64>(conn)
                .await
            {
                Ok(count) => {
                    deleted = count > 0;
                    if deleted {
                        self.stats.record_delete();
                    }
                }
                Err(e) => {
                    self.stats.record_error();
                    warn!("Redis DEL error: {}", e);
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        // Also delete from in-memory cache
        if self.fallback.delete(&redis_key).await {
            deleted = true;
            self.stats.record_delete();
        }

        Ok(deleted)
    }

    /// Get or set a value with a factory function
    pub async fn get_or_set<T, F, Fut>(
        &self,
        key: CacheKey,
        ttl: Option<Duration>,
        factory: F,
    ) -> Result<T>
    where
        T: Serialize + DeserializeOwned + Clone,
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        // Try to get from cache first
        if let Some(value) = self.get::<T>(key.clone()).await? {
            return Ok(value);
        }

        // Cache miss - call factory
        let value = factory().await?;

        // Cache the result
        self.set(key, &value, ttl).await?;

        Ok(value)
    }

    /// Clear all cached data
    pub async fn clear(&self) -> Result<()> {
        // Clear in-memory cache
        self.fallback.clear().await;

        // Clear Redis (flush keys with our prefix)
        if let Some(ref mut conn) = *self.connection.write().await {
            let pattern = format!("{}*", self.config.key_prefix);
            let keys: Vec<String> = redis::cmd("KEYS")
                .arg(&pattern)
                .query_async(conn)
                .await
                .map_err(|e| CacheError::OperationFailed(e.to_string()))?;

            if !keys.is_empty() {
                redis::cmd("DEL")
                    .arg(&keys)
                    .query_async::<()>(conn)
                    .await
                    .map_err(|e| CacheError::OperationFailed(e.to_string()))?;

                info!("Cleared {} keys from Redis cache", keys.len());
            }
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> Arc<CacheStats> {
        Arc::clone(&self.stats)
    }

    /// Get stats snapshot
    pub fn stats_snapshot(&self) -> CacheStatsSnapshot {
        self.stats.snapshot()
    }

    /// Health check
    pub async fn health_check(&self) -> bool {
        if !self.config.enabled {
            return true; // Cache is disabled, consider healthy
        }

        if let Some(ref mut conn) = *self.connection.write().await {
            match redis::cmd("PING").query_async::<String>(conn).await {
                Ok(response) if response == "PONG" => {
                    self.connected.store(true, Ordering::SeqCst);
                    return true;
                }
                _ => {
                    self.connected.store(false, Ordering::SeqCst);
                }
            }
        }

        // If Redis is down, we're still functional with in-memory cache
        true
    }

    /// Reconnect to Redis if disconnected
    pub async fn reconnect_if_needed(&self) -> Result<()> {
        if !self.is_connected() && self.config.enabled {
            self.connect().await?;
        }
        Ok(())
    }

    /// Get the number of entries in the fallback cache
    pub async fn fallback_size(&self) -> usize {
        self.fallback.len().await
    }
}

/// Cache-aware wrapper for API clients
pub struct CachedClient<C> {
    client: C,
    cache: Arc<RedisCache>,
    default_ttl: Duration,
}

impl<C> CachedClient<C> {
    /// Create a new cached client wrapper
    pub fn new(client: C, cache: Arc<RedisCache>, default_ttl: Duration) -> Self {
        Self {
            client,
            cache,
            default_ttl,
        }
    }

    /// Get the underlying client
    pub fn client(&self) -> &C {
        &self.client
    }

    /// Get the cache
    pub fn cache(&self) -> &Arc<RedisCache> {
        &self.cache
    }

    /// Execute a cached query
    pub async fn cached_query<T, F, Fut>(
        &self,
        key: CacheKey,
        ttl: Option<Duration>,
        query: F,
    ) -> Result<T>
    where
        T: Serialize + DeserializeOwned + Clone,
        F: FnOnce(&C) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let ttl = ttl.unwrap_or(self.default_ttl);

        // Try cache first
        if let Some(value) = self.cache.get::<T>(key.clone()).await? {
            return Ok(value);
        }

        // Cache miss - execute query
        let value = query(&self.client).await?;

        // Cache the result
        self.cache.set(key, &value, Some(ttl)).await?;

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_to_redis_key() {
        let prefix = "janus:";

        assert_eq!(
            CacheKey::news("bitcoin").to_redis_key(prefix),
            "janus:news:bitcoin"
        );

        assert_eq!(
            CacheKey::weather("new-york").to_redis_key(prefix),
            "janus:weather:new-york"
        );

        assert_eq!(
            CacheKey::api_response("newsapi", "headlines").to_redis_key(prefix),
            "janus:api:newsapi:headlines"
        );
    }

    #[test]
    fn test_cache_entry() {
        let entry = CacheEntry::new("test-value");
        assert!(!entry.is_expired());
        assert!(entry.remaining_ttl().is_none());
    }

    #[test]
    fn test_cache_entry_with_ttl() {
        let entry = CacheEntry::with_ttl("test-value", Duration::from_secs(300));
        assert!(!entry.is_expired());
        assert!(entry.remaining_ttl().is_some());
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
    async fn test_in_memory_cache() {
        let cache = InMemoryCache::new(100);

        // Set a value
        cache
            .set(
                "test-key".to_string(),
                b"test-value".to_vec(),
                Some(Duration::from_secs(300)),
            )
            .await;

        // Get the value
        let value = cache.get("test-key").await;
        assert!(value.is_some());
        assert_eq!(value.unwrap(), b"test-value");

        // Delete the value
        assert!(cache.delete("test-key").await);
        assert!(cache.get("test-key").await.is_none());
    }

    #[tokio::test]
    async fn test_redis_cache_disabled() {
        let config = RedisConfig {
            enabled: false,
            ..Default::default()
        };

        let cache = RedisCache::new(config).await.unwrap();
        assert!(!cache.is_connected());

        // Should still work with in-memory fallback
        cache
            .set(CacheKey::news("test"), &"value", None)
            .await
            .unwrap();
        let value: Option<String> = cache.get(CacheKey::news("test")).await.unwrap();
        assert_eq!(value, Some("value".to_string()));
    }

    #[tokio::test]
    async fn test_cache_stats_tracking() {
        let config = RedisConfig {
            enabled: false,
            ..Default::default()
        };

        let cache = RedisCache::new(config).await.unwrap();

        // Miss
        let _: Option<String> = cache.get(CacheKey::news("missing")).await.unwrap();

        // Set
        cache
            .set(CacheKey::news("test"), &"value", None)
            .await
            .unwrap();

        // Hit
        let _: Option<String> = cache.get(CacheKey::news("test")).await.unwrap();

        let stats = cache.stats_snapshot();
        assert_eq!(stats.misses, 1); // Initial miss for "missing" key
        assert_eq!(stats.sets, 1);
        assert_eq!(stats.hits, 1); // Hit for "test" key after set
    }
}
