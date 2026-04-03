//! # Redis Persistence for Strategy Affinity State
//!
//! Provides durable persistence of `StrategyAffinityTracker` state via Redis,
//! ensuring affinity data (per-strategy, per-asset performance records) survives
//! service restarts.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────┐       ┌───────────┐
//! │  TradingPipeline    │       │   Redis   │
//! │  ┌───────────────┐  │  save │           │
//! │  │ StrategyGate  │──┼──────▶│  KEY:     │
//! │  │  └─ Affinity  │  │       │  janus:   │
//! │  │     Tracker    │◀─┼──────│  affinity │
//! │  └───────────────┘  │ load  │  :state   │
//! └─────────────────────┘       └───────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::persistence::AffinityRedisStore;
//!
//! let store = AffinityRedisStore::new("redis://127.0.0.1:6379").await?;
//!
//! // Save current state
//! let tracker = pipeline.strategy_gate().await.tracker();
//! store.save(tracker).await?;
//!
//! // Load state on startup
//! if let Some(restored) = store.load().await? {
//!     // Replace tracker in pipeline
//! }
//! ```

use anyhow::{Context, Result};
use redis::AsyncCommands;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use janus_strategies::affinity::StrategyAffinityTracker;

// ════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════

/// Default Redis key for affinity state.
const DEFAULT_KEY: &str = "janus:affinity:state";

/// Default TTL for the affinity state key (7 days).
const DEFAULT_TTL_SECS: u64 = 7 * 24 * 3600;

/// Maximum acceptable payload size (10 MB). If the serialized state
/// exceeds this, something is likely wrong.
const MAX_PAYLOAD_BYTES: usize = 10 * 1024 * 1024;

// ════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════

/// Configuration for Redis affinity persistence.
#[derive(Debug, Clone)]
pub struct AffinityRedisConfig {
    /// Redis connection URL (e.g. `redis://127.0.0.1:6379`).
    pub redis_url: String,

    /// Redis key under which affinity state is stored.
    pub key: String,

    /// TTL for the state key in seconds. Set to 0 to disable expiry.
    pub ttl_secs: u64,

    /// Connection timeout.
    pub connect_timeout: Duration,

    /// Whether to log a warning (instead of erroring) when load finds
    /// no saved state. Useful for first-time boot.
    pub warn_on_missing: bool,
}

impl Default for AffinityRedisConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://127.0.0.1:6379".to_string(),
            key: DEFAULT_KEY.to_string(),
            ttl_secs: DEFAULT_TTL_SECS,
            connect_timeout: Duration::from_secs(5),
            warn_on_missing: true,
        }
    }
}

impl AffinityRedisConfig {
    /// Create a config from environment variables with sensible defaults.
    ///
    /// Environment variables:
    /// - `REDIS_URL` or `BRAIN_AFFINITY_REDIS_URL` — Redis connection string
    /// - `BRAIN_AFFINITY_REDIS_KEY` — Redis key (default: `janus:affinity:state`)
    /// - `BRAIN_AFFINITY_REDIS_TTL_SECS` — TTL in seconds (default: 604800 = 7 days)
    /// - `BRAIN_AFFINITY_REDIS_CONNECT_TIMEOUT_SECS` — Connection timeout (default: 5)
    pub fn from_env() -> Self {
        let redis_url = std::env::var("BRAIN_AFFINITY_REDIS_URL")
            .or_else(|_| std::env::var("REDIS_URL"))
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

        let key =
            std::env::var("BRAIN_AFFINITY_REDIS_KEY").unwrap_or_else(|_| DEFAULT_KEY.to_string());

        let ttl_secs: u64 = std::env::var("BRAIN_AFFINITY_REDIS_TTL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_TTL_SECS);

        let connect_timeout_secs: u64 = std::env::var("BRAIN_AFFINITY_REDIS_CONNECT_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        Self {
            redis_url,
            key,
            ttl_secs,
            connect_timeout: Duration::from_secs(connect_timeout_secs),
            warn_on_missing: true,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Store
// ════════════════════════════════════════════════════════════════════

/// Persistent store for `StrategyAffinityTracker` state backed by Redis.
///
/// Uses the tracker's built-in `save_state()` / `load_state()` JSON
/// serialization, storing the bytes under a configurable Redis key with
/// an optional TTL.
pub struct AffinityRedisStore {
    client: redis::aio::ConnectionManager,
    config: AffinityRedisConfig,
}

impl AffinityRedisStore {
    /// Connect to Redis and return a ready-to-use store.
    pub async fn new(config: AffinityRedisConfig) -> Result<Self> {
        info!(
            "Connecting to Redis for affinity persistence (url={}, key={})",
            config.redis_url, config.key
        );

        let redis_client = redis::Client::open(config.redis_url.as_str())
            .context("Failed to create Redis client for affinity persistence")?;

        let client = tokio::time::timeout(
            config.connect_timeout,
            redis::aio::ConnectionManager::new(redis_client),
        )
        .await
        .context("Redis connection timed out")?
        .context("Failed to connect to Redis for affinity persistence")?;

        info!("✅ Connected to Redis for affinity persistence");

        Ok(Self { client, config })
    }

    /// Connect using default environment-based configuration.
    pub async fn from_env() -> Result<Self> {
        Self::new(AffinityRedisConfig::from_env()).await
    }

    /// Save the affinity tracker's state to Redis.
    ///
    /// Serializes via `StrategyAffinityTracker::save_state()` (JSON) and
    /// stores the result under the configured key with TTL.
    pub async fn save(&self, tracker: &StrategyAffinityTracker) -> Result<()> {
        let data = tracker
            .save_state()
            .context("Failed to serialize affinity tracker state")?;

        let size = data.len();
        if size > MAX_PAYLOAD_BYTES {
            return Err(anyhow::anyhow!(
                "Affinity state payload too large: {} bytes (max {})",
                size,
                MAX_PAYLOAD_BYTES,
            ));
        }

        let mut conn = self.client.clone();

        if self.config.ttl_secs > 0 {
            conn.set_ex::<_, _, ()>(&self.config.key, data.as_slice(), self.config.ttl_secs)
                .await
                .context("Failed to write affinity state to Redis")?;
        } else {
            conn.set::<_, _, ()>(&self.config.key, data.as_slice())
                .await
                .context("Failed to write affinity state to Redis")?;
        }

        info!(
            "✅ Saved affinity state to Redis (key={}, size={} bytes)",
            self.config.key, size
        );

        Ok(())
    }

    /// Load the affinity tracker's state from Redis.
    ///
    /// Returns `Ok(Some(tracker))` if state was found and deserialized,
    /// `Ok(None)` if no state exists (first boot), or `Err` on failure.
    pub async fn load(&self) -> Result<Option<StrategyAffinityTracker>> {
        let mut conn = self.client.clone();

        let data: Option<Vec<u8>> = conn
            .get(&self.config.key)
            .await
            .context("Failed to read affinity state from Redis")?;

        match data {
            Some(bytes) if bytes.is_empty() => {
                if self.config.warn_on_missing {
                    warn!(
                        "Affinity state key exists but is empty (key={})",
                        self.config.key
                    );
                }
                Ok(None)
            }
            Some(bytes) => {
                let tracker = StrategyAffinityTracker::load_state(&bytes)
                    .context("Failed to deserialize affinity tracker state from Redis")?;

                info!(
                    "✅ Loaded affinity state from Redis (key={}, size={} bytes, strategies={}, assets={})",
                    self.config.key,
                    bytes.len(),
                    tracker.known_strategies().len(),
                    tracker.known_assets().len(),
                );

                Ok(Some(tracker))
            }
            None => {
                if self.config.warn_on_missing {
                    warn!(
                        "No affinity state found in Redis (key={}). Starting with fresh tracker.",
                        self.config.key,
                    );
                }
                Ok(None)
            }
        }
    }

    /// Delete the persisted affinity state from Redis.
    pub async fn delete(&self) -> Result<()> {
        let mut conn = self.client.clone();
        conn.del::<_, ()>(&self.config.key)
            .await
            .context("Failed to delete affinity state from Redis")?;

        info!(
            "Deleted affinity state from Redis (key={})",
            self.config.key
        );
        Ok(())
    }

    /// Check whether a persisted state exists in Redis.
    pub async fn exists(&self) -> Result<bool> {
        let mut conn = self.client.clone();
        let exists: bool = conn
            .exists(&self.config.key)
            .await
            .context("Failed to check affinity state existence in Redis")?;
        Ok(exists)
    }

    /// Get the TTL remaining on the persisted state key (in seconds).
    /// Returns `None` if the key does not exist or has no expiry.
    pub async fn ttl(&self) -> Result<Option<i64>> {
        let mut conn = self.client.clone();
        let ttl: i64 = redis::cmd("TTL")
            .arg(&self.config.key)
            .query_async(&mut conn)
            .await
            .context("Failed to query TTL for affinity state key")?;

        if ttl < 0 { Ok(None) } else { Ok(Some(ttl)) }
    }

    /// Get the underlying config.
    pub fn config(&self) -> &AffinityRedisConfig {
        &self.config
    }
}

// ════════════════════════════════════════════════════════════════════
// Pipeline integration helpers
// ════════════════════════════════════════════════════════════════════

use crate::brain_wiring::TradingPipeline;
use std::sync::Arc;

/// Save the affinity tracker state from a `TradingPipeline` into Redis.
///
/// This acquires a read lock on the strategy gate, serializes the tracker,
/// and writes to Redis. Safe to call from any async context.
pub async fn save_pipeline_affinity(
    pipeline: &Arc<TradingPipeline>,
    store: &AffinityRedisStore,
) -> Result<()> {
    let gate = pipeline.strategy_gate().await;
    let tracker = gate.tracker();
    store.save(tracker).await
}

/// Load affinity tracker state from Redis and inject it into a
/// `TradingPipeline`'s strategy gate.
///
/// If no state is found in Redis, the pipeline's existing (fresh) tracker
/// is left unchanged and `Ok(false)` is returned. On successful restore,
/// `Ok(true)` is returned.
pub async fn load_pipeline_affinity(
    pipeline: &Arc<TradingPipeline>,
    store: &AffinityRedisStore,
    min_trades: usize,
) -> Result<bool> {
    match store.load().await? {
        Some(mut restored) => {
            // Ensure the min_trades setting matches the current config
            restored.min_trades_for_confidence = min_trades;

            // Replace the tracker inside the strategy gate
            let mut gate = pipeline.strategy_gate_mut().await;
            let current_tracker = gate.tracker_mut();
            *current_tracker = restored;

            info!("✅ Restored affinity tracker state from Redis into pipeline");
            Ok(true)
        }
        None => {
            debug!("No affinity state to restore — pipeline using fresh tracker");
            Ok(false)
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Periodic auto-save helper
// ════════════════════════════════════════════════════════════════════

/// Spawn a background task that periodically saves the pipeline's affinity
/// state to Redis.
///
/// Returns a `JoinHandle` that can be aborted on shutdown.
///
/// # Parameters
///
/// * `pipeline` — The trading pipeline whose affinity state to persist.
/// * `store` — The Redis store to write to.
/// * `interval` — How often to auto-save.
pub fn spawn_affinity_autosave(
    pipeline: Arc<TradingPipeline>,
    store: Arc<AffinityRedisStore>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    info!(
        "Starting affinity auto-save task (interval={}s)",
        interval.as_secs()
    );

    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // Skip the first immediate tick
        ticker.tick().await;

        loop {
            ticker.tick().await;

            match save_pipeline_affinity(&pipeline, &store).await {
                Ok(()) => {
                    debug!("Auto-saved affinity state to Redis");
                }
                Err(e) => {
                    error!("Failed to auto-save affinity state to Redis: {}", e);
                }
            }
        }
    })
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AffinityRedisConfig::default();
        assert_eq!(config.redis_url, "redis://127.0.0.1:6379");
        assert_eq!(config.key, "janus:affinity:state");
        assert_eq!(config.ttl_secs, 7 * 24 * 3600);
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert!(config.warn_on_missing);
    }

    #[test]
    fn test_config_from_env_defaults() {
        // Without any env vars set, should use defaults
        let config = AffinityRedisConfig::from_env();
        // redis_url will be whatever REDIS_URL or default
        assert!(!config.redis_url.is_empty());
        assert!(!config.key.is_empty());
        assert!(config.ttl_secs > 0);
    }

    #[test]
    fn test_tracker_serialization_roundtrip() {
        let mut tracker = StrategyAffinityTracker::new(5);
        tracker.record_trade_result("EMAFlip", "BTCUSD", 100.0, true);
        tracker.record_trade_result("EMAFlip", "BTCUSD", -30.0, false);
        tracker.record_trade_result("MeanRev", "ETHUSD", 50.0, true);

        let bytes = tracker.save_state().expect("serialize should succeed");
        assert!(!bytes.is_empty());
        assert!(bytes.len() < MAX_PAYLOAD_BYTES);

        let restored =
            StrategyAffinityTracker::load_state(&bytes).expect("deserialize should succeed");

        assert_eq!(
            restored.known_strategies().len(),
            tracker.known_strategies().len()
        );
        assert_eq!(restored.known_assets().len(), tracker.known_assets().len());

        // Verify data integrity
        let perf = restored
            .get_performance("EMAFlip", "BTCUSD")
            .expect("should have EMAFlip/BTCUSD");
        assert_eq!(perf.total_trades, 2);
    }

    #[test]
    fn test_empty_tracker_serialization() {
        let tracker = StrategyAffinityTracker::new(10);
        let bytes = tracker.save_state().expect("serialize should succeed");
        let restored =
            StrategyAffinityTracker::load_state(&bytes).expect("deserialize should succeed");
        assert_eq!(restored.min_trades_for_confidence, 10);
        assert!(restored.known_strategies().is_empty());
    }

    #[test]
    fn test_payload_size_guard() {
        // Ensure a large but realistic tracker doesn't exceed the limit
        let mut tracker = StrategyAffinityTracker::new(5);
        for i in 0..100 {
            for j in 0..50 {
                tracker.record_trade_result(
                    &format!("Strategy_{}", i),
                    &format!("ASSET_{}", j),
                    100.0 * (i as f64 - 50.0),
                    i % 2 == 0,
                );
            }
        }

        let bytes = tracker.save_state().expect("serialize should succeed");
        assert!(
            bytes.len() < MAX_PAYLOAD_BYTES,
            "Payload size {} exceeds limit {}",
            bytes.len(),
            MAX_PAYLOAD_BYTES
        );
    }
}
