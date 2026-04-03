//! Distributed locking for backfill operations using Redis
//!
//! This module implements a distributed lock mechanism to prevent concurrent
//! backfill operations on the same gap across multiple data-factory instances.
//!
//! ## Features
//! - RAII-style lock guard (automatic release on drop)
//! - Configurable TTL for automatic expiration (crash recovery)
//! - Graceful lock contention handling
//! - Prometheus metrics for monitoring
//! - Lock extension for long-running operations
//!
//! ## Usage
//! ```rust,ignore
//! use janus_data::backfill::lock::{BackfillLock, LockConfig, LockMetrics};
//! use prometheus::Registry;
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! # let redis_client = redis::Client::open("redis://127.0.0.1:6379")?;
//! # let config = LockConfig::default();
//! # let registry = Registry::new();
//! # let metrics = Arc::new(LockMetrics::new(&registry)?);
//! # let lock = BackfillLock::new(redis_client, config, metrics);
//! # async fn do_backfill_work() -> anyhow::Result<()> { Ok(()) }
//! # let gap_id = "test_gap";
//! // Try to acquire lock
//! let guard = match lock.acquire(gap_id).await? {
//!     Some(guard) => guard,
//!     None => {
//!         // Another instance is processing this gap
//!         return Ok(());
//!     }
//! };
//!
//! // Perform backfill work (lock automatically released when guard drops)
//! do_backfill_work().await?;
//!
//! Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use prometheus::{IntCounter, IntGauge, Registry};
use redis::AsyncCommands;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Metrics for backfill locking operations
#[derive(Clone)]
pub struct LockMetrics {
    /// Total number of successful lock acquisitions
    pub locks_acquired: IntCounter,
    /// Total number of lock acquisition failures (already locked)
    pub locks_contended: IntCounter,
    /// Total number of lock releases
    pub locks_released: IntCounter,
    /// Total number of lock extensions
    pub locks_extended: IntCounter,
    /// Number of currently held locks
    pub locks_held: IntGauge,
    /// Number of lock release errors
    pub lock_release_errors: IntCounter,
}

impl LockMetrics {
    /// Create new metrics instance and register with Prometheus
    pub fn new(registry: &Registry) -> Result<Self> {
        let locks_acquired = IntCounter::new(
            "backfill_locks_acquired_total",
            "Total number of backfill locks successfully acquired",
        )?;
        let locks_contended = IntCounter::new(
            "backfill_locks_contended_total",
            "Total number of backfill lock acquisition failures due to contention",
        )?;
        let locks_released = IntCounter::new(
            "backfill_locks_released_total",
            "Total number of backfill locks released",
        )?;
        let locks_extended = IntCounter::new(
            "backfill_locks_extended_total",
            "Total number of backfill lock extensions",
        )?;
        let locks_held = IntGauge::new(
            "backfill_locks_held",
            "Number of backfill locks currently held by this instance",
        )?;
        let lock_release_errors = IntCounter::new(
            "backfill_lock_release_errors_total",
            "Total number of errors during lock release",
        )?;

        registry.register(Box::new(locks_acquired.clone()))?;
        registry.register(Box::new(locks_contended.clone()))?;
        registry.register(Box::new(locks_released.clone()))?;
        registry.register(Box::new(locks_extended.clone()))?;
        registry.register(Box::new(locks_held.clone()))?;
        registry.register(Box::new(lock_release_errors.clone()))?;

        Ok(Self {
            locks_acquired,
            locks_contended,
            locks_released,
            locks_extended,
            locks_held,
            lock_release_errors,
        })
    }
}

/// Configuration for backfill locking
#[derive(Debug, Clone)]
pub struct LockConfig {
    /// Time-to-live for locks (default: 5 minutes)
    /// This ensures locks are released if a worker crashes
    pub ttl: Duration,

    /// Key prefix for Redis keys (default: "backfill:lock:")
    pub key_prefix: String,

    /// Extension interval - when to extend the lock (default: TTL / 2)
    /// The lock guard will automatically extend the lock at this interval
    pub extension_interval: Duration,
}

impl Default for LockConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(300), // 5 minutes
            key_prefix: "backfill:lock:".to_string(),
            extension_interval: Duration::from_secs(150), // 2.5 minutes
        }
    }
}

/// Distributed lock manager for backfill operations
pub struct BackfillLock {
    redis: redis::Client,
    config: LockConfig,
    metrics: Arc<LockMetrics>,
}

impl BackfillLock {
    /// Create a new BackfillLock instance
    pub fn new(redis: redis::Client, config: LockConfig, metrics: Arc<LockMetrics>) -> Self {
        Self {
            redis,
            config,
            metrics,
        }
    }

    /// Attempt to acquire a lock for the given gap ID
    ///
    /// Returns:
    /// - `Ok(Some(guard))` if lock was successfully acquired
    /// - `Ok(None)` if lock is already held by another instance
    /// - `Err(_)` if a Redis error occurred
    pub async fn acquire(&self, gap_id: &str) -> Result<Option<LockGuard>> {
        let key = format!("{}{}", self.config.key_prefix, gap_id);
        let lock_id = Uuid::new_v4().to_string();
        let ttl_secs = self.config.ttl.as_secs();

        debug!(
            gap_id = %gap_id,
            lock_id = %lock_id,
            ttl_secs = ttl_secs,
            "Attempting to acquire backfill lock"
        );

        let mut conn = self
            .redis
            .get_multiplexed_async_connection()
            .await
            .context("Failed to get Redis connection for lock acquisition")?;

        // Use SET with NX (only set if not exists) and EX (expiration)
        // This is atomic and prevents race conditions
        let result: Option<String> = redis::cmd("SET")
            .arg(&key)
            .arg(&lock_id)
            .arg("NX") // Only set if key doesn't exist
            .arg("EX") // Set expiration in seconds
            .arg(ttl_secs)
            .query_async(&mut conn)
            .await
            .context("Failed to execute Redis SET NX EX command")?;

        if result.is_some() {
            // Lock acquired successfully
            info!(
                gap_id = %gap_id,
                lock_id = %lock_id,
                ttl_secs = ttl_secs,
                "Successfully acquired backfill lock"
            );

            self.metrics.locks_acquired.inc();
            self.metrics.locks_held.inc();

            Ok(Some(LockGuard {
                redis: self.redis.clone(),
                key,
                lock_id,
                acquired_at: Instant::now(),
                extension_interval: self.config.extension_interval,
                ttl: self.config.ttl,
                metrics: self.metrics.clone(),
                released: false,
            }))
        } else {
            // Lock is already held
            debug!(
                gap_id = %gap_id,
                "Lock acquisition failed: already held by another instance"
            );

            self.metrics.locks_contended.inc();
            Ok(None)
        }
    }

    /// Check if a lock exists for the given gap ID
    ///
    /// This can be used to check lock status without attempting acquisition
    pub async fn is_locked(&self, gap_id: &str) -> Result<bool> {
        let key = format!("{}{}", self.config.key_prefix, gap_id);
        let mut conn = self
            .redis
            .get_multiplexed_async_connection()
            .await
            .context("Failed to get Redis connection for lock check")?;

        let exists: bool = conn
            .exists(&key)
            .await
            .context("Failed to check lock existence")?;

        Ok(exists)
    }

    /// Get the remaining TTL for a lock
    ///
    /// Returns:
    /// - `Ok(Some(duration))` if lock exists
    /// - `Ok(None)` if lock doesn't exist
    /// - `Err(_)` if a Redis error occurred
    pub async fn get_ttl(&self, gap_id: &str) -> Result<Option<Duration>> {
        let key = format!("{}{}", self.config.key_prefix, gap_id);
        let mut conn = self
            .redis
            .get_multiplexed_async_connection()
            .await
            .context("Failed to get Redis connection for TTL check")?;

        let ttl_secs: i64 = conn.ttl(&key).await.context("Failed to get lock TTL")?;

        match ttl_secs {
            -2 => Ok(None),                // Key doesn't exist
            -1 => Ok(Some(Duration::MAX)), // Key exists but has no expiration
            secs if secs > 0 => Ok(Some(Duration::from_secs(secs as u64))),
            _ => Ok(None),
        }
    }
}

/// RAII guard for a distributed lock
///
/// The lock is automatically released when this guard is dropped.
/// The guard also supports manual extension for long-running operations.
pub struct LockGuard {
    redis: redis::Client,
    key: String,
    lock_id: String,
    acquired_at: Instant,
    extension_interval: Duration,
    ttl: Duration,
    metrics: Arc<LockMetrics>,
    released: bool,
}

impl LockGuard {
    /// Get the lock ID (UUID) for this guard
    pub fn lock_id(&self) -> &str {
        &self.lock_id
    }

    /// Get the key for this lock
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Get the time elapsed since lock acquisition
    pub fn elapsed(&self) -> Duration {
        self.acquired_at.elapsed()
    }

    /// Check if the lock should be extended based on elapsed time
    pub fn should_extend(&self) -> bool {
        !self.released && self.elapsed() >= self.extension_interval
    }

    /// Extend the lock TTL
    ///
    /// This should be called periodically for long-running backfill operations
    /// to prevent the lock from expiring while work is still in progress.
    ///
    /// Returns `Ok(true)` if the lock was successfully extended, `Ok(false)`
    /// if the lock no longer exists or was taken by another instance.
    pub async fn extend(&mut self) -> Result<bool> {
        if self.released {
            return Ok(false);
        }

        debug!(
            key = %self.key,
            lock_id = %self.lock_id,
            elapsed_secs = self.elapsed().as_secs(),
            "Extending lock TTL"
        );

        let mut conn = self
            .redis
            .get_multiplexed_async_connection()
            .await
            .context("Failed to get Redis connection for lock extension")?;

        // Use Lua script to atomically check lock ownership and extend TTL
        // This prevents extending a lock that was released and re-acquired by another instance
        let script = r#"
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("EXPIRE", KEYS[1], ARGV[2])
            else
                return 0
            end
        "#;

        let result: i32 = redis::Script::new(script)
            .key(&self.key)
            .arg(&self.lock_id)
            .arg(self.ttl.as_secs())
            .invoke_async(&mut conn)
            .await
            .context("Failed to execute lock extension script")?;

        if result == 1 {
            info!(
                key = %self.key,
                lock_id = %self.lock_id,
                new_ttl_secs = self.ttl.as_secs(),
                "Successfully extended lock TTL"
            );
            self.metrics.locks_extended.inc();
            Ok(true)
        } else {
            warn!(
                key = %self.key,
                lock_id = %self.lock_id,
                "Failed to extend lock: lock no longer held or was taken by another instance"
            );
            self.released = true;
            self.metrics.locks_held.dec();
            Ok(false)
        }
    }

    /// Manually release the lock
    ///
    /// The lock is automatically released on drop, but this method allows
    /// for explicit early release if needed.
    pub async fn release(&mut self) -> Result<()> {
        if self.released {
            return Ok(());
        }

        debug!(
            key = %self.key,
            lock_id = %self.lock_id,
            "Manually releasing lock"
        );

        self.release_internal().await
    }

    /// Internal release implementation
    async fn release_internal(&mut self) -> Result<()> {
        let mut conn = self
            .redis
            .get_multiplexed_async_connection()
            .await
            .context("Failed to get Redis connection for lock release")?;

        // Use Lua script to atomically check lock ownership before deleting
        // This prevents deleting a lock that was released and re-acquired by another instance
        let script = r#"
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("DEL", KEYS[1])
            else
                return 0
            end
        "#;

        let result: i32 = redis::Script::new(script)
            .key(&self.key)
            .arg(&self.lock_id)
            .invoke_async(&mut conn)
            .await
            .context("Failed to execute lock release script")?;

        if result == 1 {
            info!(
                key = %self.key,
                lock_id = %self.lock_id,
                held_duration_secs = self.elapsed().as_secs(),
                "Successfully released lock"
            );
            self.metrics.locks_released.inc();
            self.metrics.locks_held.dec();
            self.released = true;
            Ok(())
        } else {
            warn!(
                key = %self.key,
                lock_id = %self.lock_id,
                "Lock was already released or taken by another instance"
            );
            self.released = true;
            self.metrics.locks_held.dec();
            Ok(())
        }
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if !self.released {
            // We can't await in Drop, so we spawn a task to release the lock
            // This is fire-and-forget, but that's acceptable since the lock
            // will expire anyway due to TTL
            let redis = self.redis.clone();
            let key = self.key.clone();
            let lock_id = self.lock_id.clone();
            let metrics = self.metrics.clone();

            tokio::spawn(async move {
                match redis.get_multiplexed_async_connection().await {
                    Ok(mut conn) => {
                        let script = r#"
                            if redis.call("GET", KEYS[1]) == ARGV[1] then
                                return redis.call("DEL", KEYS[1])
                            else
                                return 0
                            end
                        "#;

                        match redis::Script::new(script)
                            .key(&key)
                            .arg(&lock_id)
                            .invoke_async(&mut conn)
                            .await
                        {
                            Ok(1) => {
                                debug!(key = %key, lock_id = %lock_id, "Lock released in Drop");
                                metrics.locks_released.inc();
                                metrics.locks_held.dec();
                            }
                            Ok(_) => {
                                debug!(
                                    key = %key,
                                    lock_id = %lock_id,
                                    "Lock was already released in Drop"
                                );
                                metrics.locks_held.dec();
                            }
                            Err(e) => {
                                warn!(
                                    key = %key,
                                    lock_id = %lock_id,
                                    error = %e,
                                    "Failed to release lock in Drop"
                                );
                                metrics.lock_release_errors.inc();
                                metrics.locks_held.dec();
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            key = %key,
                            lock_id = %lock_id,
                            error = %e,
                            "Failed to get Redis connection in Drop"
                        );
                        metrics.lock_release_errors.inc();
                        metrics.locks_held.dec();
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test Redis client.
    ///
    /// Resolution order for the URL:
    ///   1. `REDIS_URL` environment variable (may include password, e.g.
    ///      `redis://:secret@127.0.0.1:6379/`)
    ///   2. Unauthenticated localhost fallback (`redis://127.0.0.1:6379/`)
    ///
    /// Returns `None` — skipping the caller's test — when Redis is unreachable
    /// **or** when authentication fails (NOAUTH).  This prevents panics when
    /// the local Docker Redis has a password but no `REDIS_URL` is exported.
    async fn test_redis_client() -> Option<redis::Client> {
        let url =
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379/".to_string());

        let client = redis::Client::open(url.as_str()).ok()?;

        // Obtain a connection and send a real PING to verify that auth passes.
        // `get_multiplexed_async_connection` only checks TCP reachability; a
        // password-protected Redis will accept the connection but reject every
        // subsequent command with NOAUTH.
        match client.get_multiplexed_async_connection().await {
            Ok(mut conn) => {
                let pong: redis::RedisResult<String> =
                    redis::cmd("PING").query_async(&mut conn).await;
                match pong {
                    Ok(_) => Some(client),
                    Err(_) => {
                        eprintln!(
                            "Skipping test: Redis at {url} requires authentication. \
                             Set REDIS_URL=redis://:<password>@127.0.0.1:6379/ to enable."
                        );
                        None
                    }
                }
            }
            Err(_) => None,
        }
    }

    #[tokio::test]
    async fn test_acquire_and_release() {
        let client = match test_redis_client().await {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: Redis not available");
                return;
            }
        };

        let registry = Registry::new();
        let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
        let config = LockConfig::default();
        let lock = BackfillLock::new(client, config, metrics.clone());

        let gap_id = format!("test_gap_{}", Uuid::new_v4());

        // Acquire lock
        let mut guard = lock.acquire(&gap_id).await.unwrap().unwrap();
        assert_eq!(metrics.locks_acquired.get(), 1);
        assert_eq!(metrics.locks_held.get(), 1);

        // Release lock
        guard.release().await.unwrap();
        assert_eq!(metrics.locks_released.get(), 1);
        assert_eq!(metrics.locks_held.get(), 0);
    }

    #[tokio::test]
    async fn test_lock_contention() {
        let client = match test_redis_client().await {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: Redis not available");
                return;
            }
        };

        let registry = Registry::new();
        let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
        let config = LockConfig::default();
        let lock = BackfillLock::new(client, config, metrics.clone());

        let gap_id = format!("test_gap_{}", Uuid::new_v4());

        // First acquisition succeeds
        let _guard1 = lock.acquire(&gap_id).await.unwrap().unwrap();
        assert_eq!(metrics.locks_acquired.get(), 1);

        // Second acquisition fails (contention)
        let guard2 = lock.acquire(&gap_id).await.unwrap();
        assert!(guard2.is_none());
        assert_eq!(metrics.locks_contended.get(), 1);
    }

    #[tokio::test]
    async fn test_lock_expiration() {
        let client = match test_redis_client().await {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: Redis not available");
                return;
            }
        };

        let registry = Registry::new();
        let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
        let config = LockConfig {
            ttl: Duration::from_secs(1), // Short TTL for testing
            ..Default::default()
        };
        let lock = BackfillLock::new(client, config, metrics);

        let gap_id = format!("test_gap_{}", Uuid::new_v4());

        // Acquire lock
        let _guard = lock.acquire(&gap_id).await.unwrap().unwrap();

        // Wait for expiration
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Should be able to acquire again
        let guard2 = lock.acquire(&gap_id).await.unwrap();
        assert!(guard2.is_some());
    }

    #[tokio::test]
    async fn test_lock_extension() {
        let client = match test_redis_client().await {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: Redis not available");
                return;
            }
        };

        let registry = Registry::new();
        let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
        let config = LockConfig::default();
        let lock = BackfillLock::new(client, config, metrics.clone());

        let gap_id = format!("test_gap_{}", Uuid::new_v4());

        // Acquire lock
        let mut guard = lock.acquire(&gap_id).await.unwrap().unwrap();

        // Extend lock
        let extended = guard.extend().await.unwrap();
        assert!(extended);
        assert_eq!(metrics.locks_extended.get(), 1);
    }

    #[tokio::test]
    async fn test_is_locked() {
        let client = match test_redis_client().await {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: Redis not available");
                return;
            }
        };

        let registry = Registry::new();
        let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
        let config = LockConfig::default();
        let lock = BackfillLock::new(client, config, metrics);

        let gap_id = format!("test_gap_{}", Uuid::new_v4());

        // Initially not locked
        assert!(!lock.is_locked(&gap_id).await.unwrap());

        // Acquire lock
        let _guard = lock.acquire(&gap_id).await.unwrap().unwrap();

        // Now locked
        assert!(lock.is_locked(&gap_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_get_ttl() {
        let client = match test_redis_client().await {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: Redis not available");
                return;
            }
        };

        let registry = Registry::new();
        let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
        let config = LockConfig::default();
        let lock = BackfillLock::new(client, config, metrics);

        let gap_id = format!("test_gap_{}", Uuid::new_v4());

        // No TTL initially
        assert!(lock.get_ttl(&gap_id).await.unwrap().is_none());

        // Acquire lock
        let _guard = lock.acquire(&gap_id).await.unwrap().unwrap();

        // Has TTL now
        let ttl = lock.get_ttl(&gap_id).await.unwrap();
        assert!(ttl.is_some());
        assert!(ttl.unwrap() <= Duration::from_secs(300));
    }
}
