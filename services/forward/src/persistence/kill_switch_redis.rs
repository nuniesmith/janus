//! # Redis-Backed Kill Switch for Multi-Instance Coordination
//!
//! Provides a shared kill switch backed by Redis so that multiple
//! `janus-forward` instances can coordinate emergency trading halts.
//!
//! When any instance activates the kill switch, all instances observe
//! the state change within their next poll interval.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────┐     ┌──────────────────┐
//! │  Forward Inst 1  │     │  Forward Inst 2  │
//! │  ┌────────────┐  │     │  ┌────────────┐  │
//! │  │ KillSwitch │──┼──┐  │  │ KillSwitch │  │
//! │  │  (local)   │  │  │  │  │  (local)   │  │
//! │  └────────────┘  │  │  │  └────────────┘  │
//! │        ↕         │  │  │        ↕         │
//! │  ┌────────────┐  │  │  │  ┌────────────┐  │
//! │  │ RedisSync  │──┼──┼──┼──│ RedisSync  │  │
//! │  └────────────┘  │  │  │  └────────────┘  │
//! └──────────────────┘  │  └──────────────────┘
//!                       ▼
//!                ┌────────────┐
//!                │   Redis    │
//!                │            │
//!                │ KEY:       │
//!                │ janus:     │
//!                │ kill_switch│
//!                └────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::persistence::kill_switch_redis::{
//!     RedisKillSwitch, RedisKillSwitchConfig,
//! };
//!
//! // Create and connect
//! let config = RedisKillSwitchConfig::from_env();
//! let ks = RedisKillSwitch::new(config).await?;
//!
//! // Activate — all instances will see this
//! ks.activate("operator", "manual safety halt").await?;
//!
//! // Check
//! assert!(ks.is_killed().await?);
//!
//! // Deactivate
//! ks.deactivate("operator", "market stabilized").await?;
//!
//! // Spawn a background sync task that polls Redis and updates the
//! // local TradingPipeline kill switch
//! let handle = ks.spawn_sync_task(pipeline.clone());
//! ```

use anyhow::{Context, Result};
use chrono::Utc;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::brain_wiring::TradingPipeline;

// ════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════

/// Default Redis key for the shared kill switch.
const DEFAULT_KEY: &str = "janus:kill_switch";

/// Default Redis key for the kill switch metadata / audit log.
const DEFAULT_META_KEY: &str = "janus:kill_switch:meta";

/// Default poll interval for syncing local state from Redis.
const DEFAULT_POLL_INTERVAL_MS: u64 = 1000;

/// Default TTL for the kill switch key. 0 = no expiry.
/// A non-zero TTL acts as a dead-man's switch: if no instance
/// refreshes the key, the kill switch auto-deactivates.
const DEFAULT_TTL_SECS: u64 = 0;

// ════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════

/// Configuration for the Redis-backed kill switch.
#[derive(Debug, Clone)]
pub struct RedisKillSwitchConfig {
    /// Redis connection URL (e.g. `redis://127.0.0.1:6379`).
    pub redis_url: String,

    /// Redis key for the kill switch state.
    pub key: String,

    /// Redis key for kill switch metadata (who activated, when, reason).
    pub meta_key: String,

    /// How often (in milliseconds) to poll Redis for state changes.
    /// Lower values = faster propagation but more Redis load.
    pub poll_interval_ms: u64,

    /// Optional TTL for the kill switch key (seconds).
    /// When set to a non-zero value, the kill switch will auto-deactivate
    /// if not refreshed within this period (dead-man's switch behavior).
    /// Set to 0 to disable auto-expiry.
    pub ttl_secs: u64,

    /// Redis connection timeout.
    pub connect_timeout: Duration,

    /// Instance identifier (hostname, pod name, etc.) for audit trail.
    pub instance_id: String,
}

impl Default for RedisKillSwitchConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://127.0.0.1:6379".to_string(),
            key: DEFAULT_KEY.to_string(),
            meta_key: DEFAULT_META_KEY.to_string(),
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            ttl_secs: DEFAULT_TTL_SECS,
            connect_timeout: Duration::from_secs(5),
            instance_id: hostname_or_default(),
        }
    }
}

impl RedisKillSwitchConfig {
    /// Build config from environment variables with sensible defaults.
    ///
    /// Environment variables:
    /// - `BRAIN_KILL_SWITCH_REDIS_URL` or `REDIS_URL` — Redis connection string
    /// - `BRAIN_KILL_SWITCH_REDIS_KEY` — key for kill switch state
    /// - `BRAIN_KILL_SWITCH_POLL_MS` — poll interval in milliseconds
    /// - `BRAIN_KILL_SWITCH_TTL_SECS` — TTL for auto-deactivation (0 = disabled)
    /// - `BRAIN_KILL_SWITCH_INSTANCE_ID` — instance identifier for audit
    pub fn from_env() -> Self {
        let redis_url = std::env::var("BRAIN_KILL_SWITCH_REDIS_URL")
            .or_else(|_| std::env::var("REDIS_URL"))
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

        let key = std::env::var("BRAIN_KILL_SWITCH_REDIS_KEY")
            .unwrap_or_else(|_| DEFAULT_KEY.to_string());

        let meta_key = format!("{}:meta", key);

        let poll_interval_ms: u64 = std::env::var("BRAIN_KILL_SWITCH_POLL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_POLL_INTERVAL_MS);

        let ttl_secs: u64 = std::env::var("BRAIN_KILL_SWITCH_TTL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_TTL_SECS);

        let connect_timeout_secs: u64 = std::env::var("BRAIN_KILL_SWITCH_CONNECT_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        let instance_id = std::env::var("BRAIN_KILL_SWITCH_INSTANCE_ID")
            .unwrap_or_else(|_| hostname_or_default());

        Self {
            redis_url,
            key,
            meta_key,
            poll_interval_ms,
            ttl_secs,
            connect_timeout: Duration::from_secs(connect_timeout_secs),
            instance_id,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Kill Switch Metadata
// ════════════════════════════════════════════════════════════════════

/// Metadata recorded when the kill switch state changes.
/// Stored as JSON in the Redis meta key for audit purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchEvent {
    /// Whether the kill switch is now active.
    pub active: bool,

    /// Who triggered the change (operator name, system, watchdog, etc.).
    pub actor: String,

    /// Human-readable reason for the change.
    pub reason: String,

    /// Which instance made the change.
    pub instance_id: String,

    /// ISO 8601 timestamp of the event.
    pub timestamp: String,
}

/// Current state of the shared kill switch as read from Redis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchState {
    /// Whether the kill switch is currently active.
    pub active: bool,

    /// The most recent event (activation or deactivation).
    pub last_event: Option<KillSwitchEvent>,

    /// Remaining TTL on the key, if any. `None` means no expiry.
    pub ttl_remaining_secs: Option<i64>,
}

// ════════════════════════════════════════════════════════════════════
// Redis Kill Switch
// ════════════════════════════════════════════════════════════════════

/// A Redis-backed kill switch that coordinates trading halts across
/// multiple `janus-forward` instances.
///
/// The kill switch state is stored as a simple `"1"` / `"0"` value
/// in Redis under the configured key, making it inspectable via
/// `redis-cli GET janus:kill_switch`.
///
/// Metadata (who activated, when, reason) is stored separately under
/// the meta key as a JSON-serialized `KillSwitchEvent`.
pub struct RedisKillSwitch {
    client: redis::aio::ConnectionManager,
    config: RedisKillSwitchConfig,
    /// Cached local state to reduce Redis reads during fast-path checks.
    local_cache: RwLock<bool>,
}

impl RedisKillSwitch {
    /// Connect to Redis and create a new shared kill switch.
    pub async fn new(config: RedisKillSwitchConfig) -> Result<Self> {
        info!(
            "Connecting to Redis for shared kill switch (url={}, key={}, poll={}ms)",
            config.redis_url, config.key, config.poll_interval_ms,
        );

        let redis_client = redis::Client::open(config.redis_url.as_str())
            .context("Failed to create Redis client for kill switch")?;

        let client = tokio::time::timeout(
            config.connect_timeout,
            redis::aio::ConnectionManager::new(redis_client),
        )
        .await
        .context("Redis connection timed out for kill switch")?
        .context("Failed to connect to Redis for kill switch")?;

        info!(
            "✅ Connected to Redis for shared kill switch (key={})",
            config.key
        );

        // Read initial state from Redis
        let mut conn = client.clone();
        let current: Option<String> = conn.get(&config.key).await.unwrap_or(None);

        let initial_state = current.as_deref() == Some("1");

        if initial_state {
            warn!(
                "🛑 Shared kill switch is ACTIVE on startup (key={})",
                config.key,
            );
        } else {
            info!(
                "Shared kill switch is inactive on startup (key={})",
                config.key,
            );
        }

        Ok(Self {
            client,
            config,
            local_cache: RwLock::new(initial_state),
        })
    }

    /// Connect using environment-based configuration.
    pub async fn from_env() -> Result<Self> {
        Self::new(RedisKillSwitchConfig::from_env()).await
    }

    /// Activate the kill switch.
    ///
    /// This writes `"1"` to Redis. All instances polling this key will
    /// observe the change within their poll interval and block new trades.
    pub async fn activate(&self, actor: &str, reason: &str) -> Result<()> {
        let mut conn = self.client.clone();

        // Set the kill switch key
        if self.config.ttl_secs > 0 {
            conn.set_ex::<_, _, ()>(&self.config.key, "1", self.config.ttl_secs)
                .await
                .context("Failed to activate kill switch in Redis")?;
        } else {
            conn.set::<_, _, ()>(&self.config.key, "1")
                .await
                .context("Failed to activate kill switch in Redis")?;
        }

        // Record metadata
        let event = KillSwitchEvent {
            active: true,
            actor: actor.to_string(),
            reason: reason.to_string(),
            instance_id: self.config.instance_id.clone(),
            timestamp: Utc::now().to_rfc3339(),
        };

        let meta_json =
            serde_json::to_string(&event).context("Failed to serialize kill switch event")?;

        conn.set::<_, _, ()>(&self.config.meta_key, meta_json.as_str())
            .await
            .context("Failed to write kill switch metadata to Redis")?;

        // Push to the audit list (keep last 100 events)
        let audit_key = format!("{}:audit", self.config.key);
        let _: () = conn
            .lpush(&audit_key, meta_json.as_str())
            .await
            .unwrap_or(());
        let _: () = conn.ltrim(&audit_key, 0, 99).await.unwrap_or(());

        // Update local cache
        *self.local_cache.write().await = true;

        warn!(
            "🛑 Shared kill switch ACTIVATED by '{}' on instance '{}': {}",
            actor, self.config.instance_id, reason,
        );

        Ok(())
    }

    /// Deactivate the kill switch.
    ///
    /// This writes `"0"` to Redis. All instances will resume normal
    /// trading within their poll interval.
    pub async fn deactivate(&self, actor: &str, reason: &str) -> Result<()> {
        let mut conn = self.client.clone();

        // Set the kill switch key to "0"
        conn.set::<_, _, ()>(&self.config.key, "0")
            .await
            .context("Failed to deactivate kill switch in Redis")?;

        // Remove TTL since we're deactivating
        let _: () = redis::cmd("PERSIST")
            .arg(&self.config.key)
            .query_async(&mut conn)
            .await
            .unwrap_or(());

        // Record metadata
        let event = KillSwitchEvent {
            active: false,
            actor: actor.to_string(),
            reason: reason.to_string(),
            instance_id: self.config.instance_id.clone(),
            timestamp: Utc::now().to_rfc3339(),
        };

        let meta_json =
            serde_json::to_string(&event).context("Failed to serialize kill switch event")?;

        conn.set::<_, _, ()>(&self.config.meta_key, meta_json.as_str())
            .await
            .context("Failed to write kill switch metadata to Redis")?;

        // Push to audit list
        let audit_key = format!("{}:audit", self.config.key);
        let _: () = conn
            .lpush(&audit_key, meta_json.as_str())
            .await
            .unwrap_or(());
        let _: () = conn.ltrim(&audit_key, 0, 99).await.unwrap_or(());

        // Update local cache
        *self.local_cache.write().await = false;

        info!(
            "✅ Shared kill switch DEACTIVATED by '{}' on instance '{}': {}",
            actor, self.config.instance_id, reason,
        );

        Ok(())
    }

    /// Check whether the kill switch is currently active.
    ///
    /// This reads from Redis (not the local cache) for the most
    /// up-to-date state. For fast-path checks in hot loops, use
    /// `is_killed_cached()` instead.
    pub async fn is_killed(&self) -> Result<bool> {
        let mut conn = self.client.clone();
        let val: Option<String> = conn
            .get(&self.config.key)
            .await
            .context("Failed to read kill switch state from Redis")?;

        let active = val.as_deref() == Some("1");

        // Update local cache
        *self.local_cache.write().await = active;

        Ok(active)
    }

    /// Check the locally cached kill switch state.
    ///
    /// This does NOT read from Redis — it returns the last-known state
    /// from the most recent poll or explicit read. Use this in hot paths
    /// where Redis latency is unacceptable.
    pub async fn is_killed_cached(&self) -> bool {
        *self.local_cache.read().await
    }

    /// Get the full kill switch state including metadata and TTL.
    pub async fn state(&self) -> Result<KillSwitchState> {
        let mut conn = self.client.clone();

        let active_val: Option<String> = conn
            .get(&self.config.key)
            .await
            .context("Failed to read kill switch state")?;

        let active = active_val.as_deref() == Some("1");

        let meta_json: Option<String> = conn.get(&self.config.meta_key).await.unwrap_or(None);

        let last_event =
            meta_json.and_then(|json| serde_json::from_str::<KillSwitchEvent>(&json).ok());

        let ttl: i64 = redis::cmd("TTL")
            .arg(&self.config.key)
            .query_async(&mut conn)
            .await
            .unwrap_or(-1);

        let ttl_remaining_secs = if ttl >= 0 { Some(ttl) } else { None };

        // Update local cache
        *self.local_cache.write().await = active;

        Ok(KillSwitchState {
            active,
            last_event,
            ttl_remaining_secs,
        })
    }

    /// Get the audit trail (last N events).
    pub async fn audit_trail(&self, limit: usize) -> Result<Vec<KillSwitchEvent>> {
        let mut conn = self.client.clone();
        let audit_key = format!("{}:audit", self.config.key);

        let entries: Vec<String> = conn
            .lrange(&audit_key, 0, (limit as isize) - 1)
            .await
            .unwrap_or_default();

        let events: Vec<KillSwitchEvent> = entries
            .iter()
            .filter_map(|json| serde_json::from_str(json).ok())
            .collect();

        Ok(events)
    }

    /// Refresh the TTL on the kill switch key (for dead-man's switch mode).
    ///
    /// Call this periodically if you're using TTL-based auto-deactivation
    /// to prevent the kill switch from expiring while intentionally active.
    pub async fn refresh_ttl(&self) -> Result<()> {
        if self.config.ttl_secs == 0 {
            return Ok(());
        }

        let mut conn = self.client.clone();
        let active: Option<String> = conn.get(&self.config.key).await.unwrap_or(None);

        if active.as_deref() == Some("1") {
            conn.expire::<_, ()>(&self.config.key, self.config.ttl_secs as i64)
                .await
                .context("Failed to refresh kill switch TTL")?;

            debug!(
                "Refreshed kill switch TTL (key={}, ttl={}s)",
                self.config.key, self.config.ttl_secs,
            );
        }

        Ok(())
    }

    /// Delete the kill switch key and metadata from Redis.
    /// Use with caution — this removes all state and audit history.
    pub async fn purge(&self) -> Result<()> {
        let mut conn = self.client.clone();
        let audit_key = format!("{}:audit", self.config.key);

        conn.del::<_, ()>(&self.config.key)
            .await
            .context("Failed to delete kill switch key")?;
        conn.del::<_, ()>(&self.config.meta_key)
            .await
            .context("Failed to delete kill switch meta key")?;
        conn.del::<_, ()>(&audit_key)
            .await
            .context("Failed to delete kill switch audit key")?;

        *self.local_cache.write().await = false;

        warn!("Purged all kill switch state from Redis");

        Ok(())
    }

    /// Get the underlying configuration.
    pub fn config(&self) -> &RedisKillSwitchConfig {
        &self.config
    }

    // ────────────────────────────────────────────────────────────────
    // Sync task
    // ────────────────────────────────────────────────────────────────

    /// Perform a single sync poll from Redis and update the local pipeline.
    ///
    /// Returns `true` if the kill switch state changed on this poll.
    pub async fn sync_once(&self, pipeline: &TradingPipeline) -> Result<bool> {
        let remote_active = self.is_killed().await?;
        let local_active = pipeline.is_killed().await;

        if remote_active != local_active {
            if remote_active {
                warn!("🛑 Shared kill switch activated remotely — syncing to local pipeline");
                pipeline.activate_kill_switch().await;
            } else {
                info!("✅ Shared kill switch deactivated remotely — syncing to local pipeline");
                pipeline.deactivate_kill_switch().await;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Spawn a background task that polls Redis at the configured interval
    /// and synchronizes the local `TradingPipeline` kill switch.
    ///
    /// Returns a `JoinHandle` that can be aborted on shutdown.
    pub fn spawn_sync_task(
        self: &Arc<Self>,
        pipeline: Arc<TradingPipeline>,
    ) -> tokio::task::JoinHandle<()> {
        let ks = Arc::clone(self);
        let interval_ms = ks.config.poll_interval_ms;

        info!(
            "Starting kill switch sync task (key={}, poll={}ms)",
            ks.config.key, interval_ms,
        );

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(interval_ms));
            // Skip the first immediate tick
            ticker.tick().await;

            let mut consecutive_errors: u32 = 0;

            loop {
                ticker.tick().await;

                match ks.sync_once(&pipeline).await {
                    Ok(changed) => {
                        if changed {
                            info!("Kill switch sync: state changed");
                        }
                        consecutive_errors = 0;
                    }
                    Err(e) => {
                        consecutive_errors += 1;
                        if consecutive_errors <= 3 {
                            warn!("Kill switch sync error ({}/3): {}", consecutive_errors, e,);
                        } else if consecutive_errors == 4 {
                            error!(
                                "Kill switch sync: {} consecutive errors. \
                                 Suppressing further warnings. Last error: {}",
                                consecutive_errors, e,
                            );
                        }
                        // On persistent Redis failure, activate local kill switch
                        // as a safety measure after 10 consecutive errors
                        if consecutive_errors == 10 {
                            error!(
                                "🛑 Kill switch sync: 10 consecutive Redis failures. \
                                 Activating local kill switch as safety measure."
                            );
                            pipeline.activate_kill_switch().await;
                        }
                    }
                }
            }
        })
    }

    /// Spawn a sync task that also periodically refreshes the TTL
    /// (for dead-man's switch mode).
    pub fn spawn_sync_task_with_ttl_refresh(
        self: &Arc<Self>,
        pipeline: Arc<TradingPipeline>,
    ) -> tokio::task::JoinHandle<()> {
        let ks = Arc::clone(self);
        let interval_ms = ks.config.poll_interval_ms;
        let ttl_secs = ks.config.ttl_secs;

        if ttl_secs == 0 {
            // No TTL configured — use the regular sync task
            return ks.spawn_sync_task(pipeline);
        }

        info!(
            "Starting kill switch sync+TTL task (key={}, poll={}ms, ttl={}s)",
            ks.config.key, interval_ms, ttl_secs,
        );

        // Refresh TTL every ttl/3 seconds to avoid accidental expiry
        let refresh_every = std::cmp::max(
            ttl_secs / 3,
            1, // at least every second
        );

        tokio::spawn(async move {
            let mut sync_ticker = tokio::time::interval(Duration::from_millis(interval_ms));
            let mut ttl_ticker = tokio::time::interval(Duration::from_secs(refresh_every));

            // Skip immediate ticks
            sync_ticker.tick().await;
            ttl_ticker.tick().await;

            loop {
                tokio::select! {
                    _ = sync_ticker.tick() => {
                        if let Err(e) = ks.sync_once(&pipeline).await {
                            warn!("Kill switch sync error: {}", e);
                        }
                    }
                    _ = ttl_ticker.tick() => {
                        if let Err(e) = ks.refresh_ttl().await {
                            warn!("Kill switch TTL refresh error: {}", e);
                        }
                    }
                }
            }
        })
    }
}

// ════════════════════════════════════════════════════════════════════
// Pipeline Integration Helpers
// ════════════════════════════════════════════════════════════════════

/// Wire a `RedisKillSwitch` to a `TradingPipeline`, synchronizing
/// the initial state from Redis on startup.
///
/// Returns the `RedisKillSwitch` wrapped in an `Arc` for sharing.
pub async fn wire_redis_kill_switch(
    pipeline: &Arc<TradingPipeline>,
    config: RedisKillSwitchConfig,
) -> Result<Arc<RedisKillSwitch>> {
    let ks = RedisKillSwitch::new(config).await?;

    // Sync initial state
    let remote_active = ks.is_killed().await?;
    if remote_active {
        warn!("🛑 Redis kill switch is active on startup — activating local pipeline");
        pipeline.activate_kill_switch().await;
    }

    Ok(Arc::new(ks))
}

/// Convenience function to wire the kill switch from environment
/// variables, sync initial state, and start the background sync task.
///
/// Returns the `Arc<RedisKillSwitch>` and the sync task `JoinHandle`.
pub async fn wire_and_spawn_redis_kill_switch(
    pipeline: Arc<TradingPipeline>,
) -> Result<(Arc<RedisKillSwitch>, tokio::task::JoinHandle<()>)> {
    let config = RedisKillSwitchConfig::from_env();
    let ks = wire_redis_kill_switch(&pipeline, config).await?;
    let handle = ks.spawn_sync_task(pipeline);
    Ok((ks, handle))
}

// ════════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════════

/// Get the system hostname, falling back to a default if unavailable.
fn hostname_or_default() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown-instance".to_string())
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RedisKillSwitchConfig::default();

        assert_eq!(config.redis_url, "redis://127.0.0.1:6379");
        assert_eq!(config.key, "janus:kill_switch");
        assert_eq!(config.meta_key, "janus:kill_switch:meta");
        assert_eq!(config.poll_interval_ms, 1000);
        assert_eq!(config.ttl_secs, 0);
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert!(!config.instance_id.is_empty());
    }

    #[test]
    fn test_config_from_env_defaults() {
        // With no env vars set, should use defaults
        let config = RedisKillSwitchConfig::from_env();

        assert!(!config.redis_url.is_empty());
        assert!(!config.key.is_empty());
        assert!(config.poll_interval_ms > 0);
        assert!(!config.instance_id.is_empty());
    }

    #[test]
    fn test_kill_switch_event_serialization() {
        let event = KillSwitchEvent {
            active: true,
            actor: "operator".to_string(),
            reason: "market crash".to_string(),
            instance_id: "pod-1".to_string(),
            timestamp: "2024-01-15T10:30:00Z".to_string(),
        };

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: KillSwitchEvent = serde_json::from_str(&json).unwrap();

        assert!(deserialized.active);
        assert_eq!(deserialized.actor, "operator");
        assert_eq!(deserialized.reason, "market crash");
        assert_eq!(deserialized.instance_id, "pod-1");
    }

    #[test]
    fn test_kill_switch_state_serialization() {
        let state = KillSwitchState {
            active: true,
            last_event: Some(KillSwitchEvent {
                active: true,
                actor: "watchdog".to_string(),
                reason: "dead component detected".to_string(),
                instance_id: "pod-2".to_string(),
                timestamp: "2024-01-15T10:31:00Z".to_string(),
            }),
            ttl_remaining_secs: Some(3600),
        };

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: KillSwitchState = serde_json::from_str(&json).unwrap();

        assert!(deserialized.active);
        assert!(deserialized.last_event.is_some());
        assert_eq!(deserialized.ttl_remaining_secs, Some(3600));
    }

    #[test]
    fn test_kill_switch_state_no_event() {
        let state = KillSwitchState {
            active: false,
            last_event: None,
            ttl_remaining_secs: None,
        };

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: KillSwitchState = serde_json::from_str(&json).unwrap();

        assert!(!deserialized.active);
        assert!(deserialized.last_event.is_none());
        assert!(deserialized.ttl_remaining_secs.is_none());
    }

    #[test]
    fn test_hostname_or_default_returns_non_empty() {
        let hostname = hostname_or_default();
        assert!(!hostname.is_empty());
    }

    #[test]
    fn test_config_meta_key_derived_from_key() {
        let mut config = RedisKillSwitchConfig::default();
        config.key = "custom:kill".to_string();
        config.meta_key = format!("{}:meta", config.key);

        assert_eq!(config.meta_key, "custom:kill:meta");
    }
}
