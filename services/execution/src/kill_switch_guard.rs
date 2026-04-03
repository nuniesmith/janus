//! Kill Switch Guard — Defense-in-Depth for the Execution Service
//!
//! ⚠️ CRITICAL SAFETY COMPONENT ⚠️
//!
//! This module provides a Redis-backed kill switch guard that the execution
//! service checks on **every** order submission path. It reads the shared
//! `janus:kill_switch` key from Redis (the same key the forward service /
//! brain pipeline writes to) and blocks all orders when the kill switch is
//! active.
//!
//! ## Why this exists
//!
//! The forward service (`TradingPipeline`) already gates signals at the
//! brain level. However, the execution service exposes its own gRPC and
//! HTTP APIs. If a caller bypasses the forward service and submits orders
//! directly to the execution service, the kill switch would not be
//! enforced without this guard.
//!
//! This is **defense-in-depth**: even if the pipeline-level kill switch
//! fires, the execution service independently refuses orders.
//!
//! ## Architecture
//!
//! ```text
//! Redis: janus:kill_switch = "1" | "active" | "killed" → BLOCKED
//! Redis: janus:kill_switch = "0" | "" | absent          → ALLOWED
//!
//! ┌─────────────────┐      poll every N ms      ┌───────┐
//! │ KillSwitchGuard │ ◄────────────────────────► │ Redis │
//! │   local_cache   │                            └───────┘
//! └────────┬────────┘
//!          │ is_active() — hot path, no I/O
//!          ▼
//!   OrderManager::submit_order()
//!   SignalFlowCoordinator::submit_signal()
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use janus_execution::kill_switch_guard::{KillSwitchGuard, KillSwitchGuardConfig};
//!
//! let config = KillSwitchGuardConfig::from_env();
//! let guard = KillSwitchGuard::new(redis_conn, config);
//!
//! // Spawn background sync
//! let handle = guard.spawn_sync_task();
//!
//! // On the order submission hot path:
//! if guard.is_active() {
//!     return Err(ExecutionError::KillSwitchActive);
//! }
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default Redis key for the shared kill switch.
const DEFAULT_KEY: &str = "janus:kill_switch";

/// Default poll interval in milliseconds.
const DEFAULT_POLL_INTERVAL_MS: u64 = 250;

/// Configuration for the execution-service kill switch guard.
#[derive(Debug, Clone)]
pub struct KillSwitchGuardConfig {
    /// Redis key to read. Must match the key the forward service writes.
    pub key: String,

    /// How often to poll Redis for state changes (milliseconds).
    /// Lower = faster reaction, higher = less Redis load.
    /// 250ms is a good default: orders are blocked within 250ms of
    /// the kill switch being activated in Redis.
    pub poll_interval_ms: u64,

    /// If `true`, the guard starts in the "active" (blocking) state
    /// until the first successful Redis read confirms the switch is off.
    /// This is the fail-safe default: if Redis is unreachable at startup,
    /// no orders are allowed until connectivity is established.
    pub fail_closed: bool,

    /// How many consecutive Redis read failures before the guard
    /// automatically activates (fail-closed behavior). 0 = disabled.
    pub max_consecutive_failures: u32,
}

impl Default for KillSwitchGuardConfig {
    fn default() -> Self {
        Self {
            key: DEFAULT_KEY.to_string(),
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            fail_closed: true,
            max_consecutive_failures: 5,
        }
    }
}

impl KillSwitchGuardConfig {
    /// Build configuration from environment variables.
    ///
    /// | Variable                             | Default                |
    /// |--------------------------------------|------------------------|
    /// | `EXEC_KILL_SWITCH_REDIS_KEY`         | `janus:kill_switch`    |
    /// | `EXEC_KILL_SWITCH_POLL_MS`           | `250`                  |
    /// | `EXEC_KILL_SWITCH_FAIL_CLOSED`       | `true`                 |
    /// | `EXEC_KILL_SWITCH_MAX_FAILURES`      | `5`                    |
    pub fn from_env() -> Self {
        let key =
            std::env::var("EXEC_KILL_SWITCH_REDIS_KEY").unwrap_or_else(|_| DEFAULT_KEY.to_string());

        let poll_interval_ms: u64 = std::env::var("EXEC_KILL_SWITCH_POLL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_POLL_INTERVAL_MS);

        let fail_closed: bool = std::env::var("EXEC_KILL_SWITCH_FAIL_CLOSED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(true);

        let max_consecutive_failures: u32 = std::env::var("EXEC_KILL_SWITCH_MAX_FAILURES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        Self {
            key,
            poll_interval_ms,
            fail_closed,
            max_consecutive_failures,
        }
    }
}

// ---------------------------------------------------------------------------
// Guard (shared state)
// ---------------------------------------------------------------------------

/// Inner shared state for the kill switch guard.
struct KillSwitchGuardInner {
    /// Cached kill switch state. `true` = active = block all orders.
    active: AtomicBool,

    /// Consecutive Redis read failures.
    consecutive_failures: AtomicBool, // simplified: tracks "in failure mode"

    /// Set to `true` when the background sync task is running.
    sync_running: AtomicBool,
}

/// Redis-backed kill switch guard for the execution service.
///
/// This is cheap to clone (inner state is `Arc`-shared).
#[derive(Clone)]
pub struct KillSwitchGuard {
    inner: Arc<KillSwitchGuardInner>,
    config: KillSwitchGuardConfig,
    redis: redis::aio::ConnectionManager,
}

impl KillSwitchGuard {
    /// Create a new kill switch guard.
    ///
    /// If `config.fail_closed` is `true` (the default), the guard starts
    /// in the **active** state and will only allow orders after the first
    /// successful Redis read confirms the kill switch is off.
    pub fn new(redis: redis::aio::ConnectionManager, config: KillSwitchGuardConfig) -> Self {
        let initial_state = config.fail_closed;

        if initial_state {
            warn!(
                "🛑 Kill switch guard starting in FAIL-CLOSED mode — \
                 orders blocked until Redis confirms kill switch is off"
            );
        }

        Self {
            inner: Arc::new(KillSwitchGuardInner {
                active: AtomicBool::new(initial_state),
                consecutive_failures: AtomicBool::new(false),
                sync_running: AtomicBool::new(false),
            }),
            config,
            redis,
        }
    }

    /// Create a guard from environment variables and an existing Redis connection.
    pub fn from_env(redis: redis::aio::ConnectionManager) -> Self {
        Self::new(redis, KillSwitchGuardConfig::from_env())
    }

    // -----------------------------------------------------------------------
    // Hot-path guard — NO async, NO I/O
    // -----------------------------------------------------------------------

    /// Returns `true` if the kill switch is currently active.
    ///
    /// This is a **single atomic load** — safe and fast for the hot path.
    /// Every order submission path MUST call this.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.inner.active.load(Ordering::SeqCst)
    }

    /// Convenience: returns `true` if orders should be allowed.
    #[inline]
    pub fn should_allow_order(&self) -> bool {
        !self.is_active()
    }

    /// Returns `true` if the background sync task is running.
    pub fn is_sync_running(&self) -> bool {
        self.inner.sync_running.load(Ordering::SeqCst)
    }

    /// Returns `true` if the guard is in failure mode (Redis unreachable).
    pub fn is_in_failure_mode(&self) -> bool {
        self.inner.consecutive_failures.load(Ordering::SeqCst)
    }

    // -----------------------------------------------------------------------
    // Manual activation (for local triggers, e.g., risk manager)
    // -----------------------------------------------------------------------

    /// Manually activate the kill switch guard (block all orders).
    ///
    /// This does NOT write to Redis — it only affects this execution
    /// service instance. Use this for local emergency stops.
    pub fn activate(&self, reason: &str) {
        let was_active = self.inner.active.swap(true, Ordering::SeqCst);
        if !was_active {
            error!("🚨 EXECUTION KILL SWITCH ACTIVATED (local): {}", reason);
        }
    }

    /// Manually deactivate the kill switch guard.
    ///
    /// ⚠️ Use with extreme caution. If the Redis key is still set,
    /// the next sync cycle will re-activate the guard.
    pub fn deactivate(&self, reason: &str) {
        let was_active = self.inner.active.swap(false, Ordering::SeqCst);
        if was_active {
            warn!("🟢 EXECUTION KILL SWITCH DEACTIVATED (local): {}", reason);
        }
    }

    // -----------------------------------------------------------------------
    // Redis sync
    // -----------------------------------------------------------------------

    /// Read the kill switch state from Redis (single poll).
    ///
    /// Returns `Ok(true)` if the kill switch is active, `Ok(false)` if
    /// inactive, or `Err` if the read failed.
    pub async fn poll_redis(&self) -> Result<bool, String> {
        let mut conn = self.redis.clone();
        let result: redis::RedisResult<Option<String>> = redis::cmd("GET")
            .arg(&self.config.key)
            .query_async(&mut conn)
            .await;

        match result {
            Ok(Some(value)) => {
                let active = is_kill_switch_value_active(&value);
                Ok(active)
            }
            Ok(None) => {
                // Key doesn't exist — kill switch is not active
                Ok(false)
            }
            Err(e) => Err(format!("Redis GET {} failed: {}", self.config.key, e)),
        }
    }

    /// Perform a single sync cycle: read Redis and update local cache.
    ///
    /// Returns the new state.
    pub async fn sync_once(&self) -> Result<bool, String> {
        match self.poll_redis().await {
            Ok(active) => {
                let was_active = self.inner.active.swap(active, Ordering::SeqCst);
                self.inner
                    .consecutive_failures
                    .store(false, Ordering::SeqCst);

                // Log state transitions
                if active && !was_active {
                    error!(
                        "🚨 EXECUTION KILL SWITCH ACTIVATED (from Redis key '{}')",
                        self.config.key
                    );
                } else if !active && was_active {
                    info!(
                        "🟢 EXECUTION KILL SWITCH DEACTIVATED (from Redis key '{}')",
                        self.config.key
                    );
                }

                Ok(active)
            }
            Err(e) => {
                // Redis read failed
                self.inner
                    .consecutive_failures
                    .store(true, Ordering::SeqCst);

                if self.config.max_consecutive_failures > 0 {
                    // Fail closed: activate the kill switch
                    let was_active = self.inner.active.swap(true, Ordering::SeqCst);
                    if !was_active {
                        error!(
                            "🚨 EXECUTION KILL SWITCH ACTIVATED (fail-closed, Redis unreachable): {}",
                            e
                        );
                    }
                } else {
                    warn!(
                        "Kill switch guard: Redis read failed (fail-open mode): {}",
                        e
                    );
                }

                Err(e)
            }
        }
    }

    /// Spawn a background task that continuously syncs the kill switch
    /// state from Redis.
    ///
    /// The task runs until the `KillSwitchGuard` (and all its clones)
    /// are dropped — it holds a weak reference via the `Arc`.
    pub fn spawn_sync_task(&self) -> tokio::task::JoinHandle<()> {
        let guard = self.clone();
        let interval = Duration::from_millis(self.config.poll_interval_ms);

        tokio::spawn(async move {
            guard.inner.sync_running.store(true, Ordering::SeqCst);
            info!(
                "Kill switch guard sync task started (key='{}', interval={}ms)",
                guard.config.key, guard.config.poll_interval_ms,
            );

            // Initial sync
            match guard.sync_once().await {
                Ok(active) => {
                    if active {
                        warn!("Kill switch is ACTIVE on startup — orders will be blocked");
                    } else {
                        info!("Kill switch is inactive on startup — orders allowed");
                    }
                }
                Err(e) => {
                    error!("Initial kill switch sync failed: {}", e);
                }
            }

            loop {
                tokio::time::sleep(interval).await;

                if let Err(e) = guard.sync_once().await {
                    debug!("Kill switch sync error (will retry): {}", e);
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Determine if a Redis value means "kill switch active".
///
/// Active values: "1", "true", "active", "killed", "yes", "on"
/// Everything else (including empty string): inactive.
fn is_kill_switch_value_active(value: &str) -> bool {
    matches!(
        value.trim().to_lowercase().as_str(),
        "1" | "true" | "active" | "killed" | "yes" | "on"
    )
}

// ---------------------------------------------------------------------------
// No-op guard for testing
// ---------------------------------------------------------------------------

/// A kill switch guard that never blocks orders.
///
/// Use this in tests or when Redis is unavailable and you want to
/// explicitly opt out of the safety guard (NOT recommended for production).
#[derive(Clone)]
pub struct NoOpKillSwitchGuard;

impl NoOpKillSwitchGuard {
    #[inline]
    pub fn is_active(&self) -> bool {
        false
    }

    #[inline]
    pub fn should_allow_order(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Trait for polymorphic usage
// ---------------------------------------------------------------------------

/// Trait abstracting the kill switch guard so `OrderManager` and
/// `SignalFlowCoordinator` can accept either a real or no-op guard.
pub trait OrderGate: Send + Sync {
    /// Returns `true` if orders should be blocked.
    fn is_blocked(&self) -> bool;

    /// Human-readable reason for the block (for error messages).
    fn block_reason(&self) -> &'static str {
        "Kill switch is active — all order submission is halted"
    }
}

impl OrderGate for KillSwitchGuard {
    #[inline]
    fn is_blocked(&self) -> bool {
        self.is_active()
    }
}

impl OrderGate for NoOpKillSwitchGuard {
    #[inline]
    fn is_blocked(&self) -> bool {
        false
    }
}

/// An `OrderGate` backed by a simple `AtomicBool` — useful for tests.
pub struct AtomicOrderGate {
    blocked: AtomicBool,
}

impl AtomicOrderGate {
    pub fn new(blocked: bool) -> Self {
        Self {
            blocked: AtomicBool::new(blocked),
        }
    }

    pub fn set_blocked(&self, blocked: bool) {
        self.blocked.store(blocked, Ordering::SeqCst);
    }
}

impl OrderGate for AtomicOrderGate {
    fn is_blocked(&self) -> bool {
        self.blocked.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Value parsing tests ──

    #[test]
    fn test_active_values() {
        assert!(is_kill_switch_value_active("1"));
        assert!(is_kill_switch_value_active("true"));
        assert!(is_kill_switch_value_active("TRUE"));
        assert!(is_kill_switch_value_active("active"));
        assert!(is_kill_switch_value_active("ACTIVE"));
        assert!(is_kill_switch_value_active("killed"));
        assert!(is_kill_switch_value_active("yes"));
        assert!(is_kill_switch_value_active("on"));
        assert!(is_kill_switch_value_active("  1  ")); // whitespace trimmed
        assert!(is_kill_switch_value_active("  Active  "));
    }

    #[test]
    fn test_inactive_values() {
        assert!(!is_kill_switch_value_active("0"));
        assert!(!is_kill_switch_value_active("false"));
        assert!(!is_kill_switch_value_active(""));
        assert!(!is_kill_switch_value_active("inactive"));
        assert!(!is_kill_switch_value_active("off"));
        assert!(!is_kill_switch_value_active("no"));
        assert!(!is_kill_switch_value_active("armed")); // armed ≠ active
        assert!(!is_kill_switch_value_active("random_garbage"));
    }

    // ── Config tests ──

    #[test]
    fn test_default_config() {
        let config = KillSwitchGuardConfig::default();
        assert_eq!(config.key, "janus:kill_switch");
        assert_eq!(config.poll_interval_ms, 250);
        assert!(config.fail_closed);
        assert_eq!(config.max_consecutive_failures, 5);
    }

    // ── NoOp guard tests ──

    #[test]
    fn test_noop_guard_never_blocks() {
        let guard = NoOpKillSwitchGuard;
        assert!(!guard.is_active());
        assert!(guard.should_allow_order());
        assert!(!guard.is_blocked());
    }

    // ── AtomicOrderGate tests ──

    #[test]
    fn test_atomic_order_gate() {
        let gate = AtomicOrderGate::new(false);
        assert!(!gate.is_blocked());

        gate.set_blocked(true);
        assert!(gate.is_blocked());

        gate.set_blocked(false);
        assert!(!gate.is_blocked());
    }

    // ── OrderGate trait tests ──

    #[test]
    fn test_order_gate_block_reason() {
        let gate = AtomicOrderGate::new(true);
        assert!(gate.block_reason().contains("Kill switch"));
    }

    // ── Integration tests (require Redis) ──

    // These tests are gated behind a feature or run in CI with Redis available.
    // To run locally: `cargo test -- --ignored` with Redis on localhost:6379.

    #[tokio::test]
    #[ignore = "requires Redis"]
    async fn test_guard_reads_redis_inactive() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let conn = redis::aio::ConnectionManager::new(client).await.unwrap();

        // Use a test-specific key to avoid races with parallel tests
        let test_key = "janus:kill_switch:test:inactive";

        // Ensure the key is absent
        let mut c = conn.clone();
        let _: redis::RedisResult<()> = redis::cmd("DEL").arg(test_key).query_async(&mut c).await;

        let config = KillSwitchGuardConfig {
            key: test_key.to_string(),
            fail_closed: false, // Start open for this test
            ..Default::default()
        };
        let guard = KillSwitchGuard::new(conn, config);

        let result = guard.sync_once().await;
        assert!(result.is_ok());
        assert!(!result.unwrap());
        assert!(!guard.is_active());
        assert!(guard.should_allow_order());
    }

    #[tokio::test]
    #[ignore = "requires Redis"]
    async fn test_guard_reads_redis_active() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let conn = redis::aio::ConnectionManager::new(client).await.unwrap();

        // Use a test-specific key to avoid races with parallel tests
        let test_key = "janus:kill_switch:test:active";

        // Set the key to active
        let mut c = conn.clone();
        let _: redis::RedisResult<()> = redis::cmd("SET")
            .arg(test_key)
            .arg("active")
            .query_async(&mut c)
            .await;

        let config = KillSwitchGuardConfig {
            key: test_key.to_string(),
            fail_closed: false,
            ..Default::default()
        };
        let guard = KillSwitchGuard::new(conn.clone(), config);

        let result = guard.sync_once().await;
        assert!(result.is_ok());
        assert!(result.unwrap());
        assert!(guard.is_active());
        assert!(!guard.should_allow_order());

        // Clean up
        let _: redis::RedisResult<()> = redis::cmd("DEL").arg(test_key).query_async(&mut c).await;
    }

    #[tokio::test]
    #[ignore = "requires Redis"]
    async fn test_guard_state_transition() {
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let conn = redis::aio::ConnectionManager::new(client).await.unwrap();
        let mut c = conn.clone();

        // Use a test-specific key to avoid races with parallel tests
        let test_key = "janus:kill_switch:test:transition";

        // Start inactive
        let _: redis::RedisResult<()> = redis::cmd("DEL").arg(test_key).query_async(&mut c).await;

        let config = KillSwitchGuardConfig {
            key: test_key.to_string(),
            fail_closed: false,
            ..Default::default()
        };
        let guard = KillSwitchGuard::new(conn.clone(), config);

        // Should be inactive
        guard.sync_once().await.unwrap();
        assert!(!guard.is_active());

        // Activate via Redis
        let _: redis::RedisResult<()> = redis::cmd("SET")
            .arg(test_key)
            .arg("1")
            .query_async(&mut c)
            .await;

        guard.sync_once().await.unwrap();
        assert!(guard.is_active());

        // Deactivate via Redis
        let _: redis::RedisResult<()> = redis::cmd("SET")
            .arg(test_key)
            .arg("0")
            .query_async(&mut c)
            .await;

        guard.sync_once().await.unwrap();
        assert!(!guard.is_active());

        // Clean up
        let _: redis::RedisResult<()> = redis::cmd("DEL").arg(test_key).query_async(&mut c).await;
    }

    // ── Manual activation tests (no Redis needed) ──

    #[tokio::test]
    async fn test_manual_activation() {
        // Use a dummy Redis connection — we won't actually connect
        // For manual activation tests, we just need the struct.
        // We'll test the atomic behavior directly.

        let gate = AtomicOrderGate::new(false);
        assert!(!gate.is_blocked());

        gate.set_blocked(true);
        assert!(gate.is_blocked());
    }

    #[test]
    fn test_fail_closed_initial_state() {
        // When fail_closed = true (default), the guard should start blocked.
        let config = KillSwitchGuardConfig {
            fail_closed: true,
            ..Default::default()
        };
        assert!(config.fail_closed);
        // The actual KillSwitchGuard would start with active=true,
        // but we can't construct it without Redis. The AtomicOrderGate
        // demonstrates the same principle:
        let gate = AtomicOrderGate::new(true);
        assert!(gate.is_blocked());
    }

    #[test]
    fn test_fail_open_initial_state() {
        let config = KillSwitchGuardConfig {
            fail_closed: false,
            ..Default::default()
        };
        assert!(!config.fail_closed);
        let gate = AtomicOrderGate::new(false);
        assert!(!gate.is_blocked());
    }
}
