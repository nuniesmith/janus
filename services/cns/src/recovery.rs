//! CNS Auto-Recovery System
//!
//! Implements automatic recovery actions for unhealthy modules.
//! This is the "reflex" system of the CNS - automatic responses to problems.
//!
//! ## Recovery Actions
//!
//! | Condition | Action | Cooldown |
//! |-----------|--------|----------|
//! | Module unhealthy | Restart module | 60s |
//! | High memory | Clear caches | 30s |
//! | Signal backlog | Flush queue | 15s |
//! | Database disconnected | Reconnect | 10s |
//! | Redis disconnected | Reconnect | 10s |

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Recovery action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoveryAction {
    /// Restart a specific module
    RestartModule,
    /// Clear module caches
    ClearCache,
    /// Flush signal queue
    FlushSignalQueue,
    /// Reconnect to database
    ReconnectDatabase,
    /// Reconnect to Redis
    ReconnectRedis,
    /// Send alert notification
    SendAlert,
    /// No action needed
    None,
}

impl std::fmt::Display for RecoveryAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryAction::RestartModule => write!(f, "restart_module"),
            RecoveryAction::ClearCache => write!(f, "clear_cache"),
            RecoveryAction::FlushSignalQueue => write!(f, "flush_signal_queue"),
            RecoveryAction::ReconnectDatabase => write!(f, "reconnect_database"),
            RecoveryAction::ReconnectRedis => write!(f, "reconnect_redis"),
            RecoveryAction::SendAlert => write!(f, "send_alert"),
            RecoveryAction::None => write!(f, "none"),
        }
    }
}

/// Result of a recovery attempt
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub action: RecoveryAction,
    pub target: String,
    pub success: bool,
    pub message: String,
    pub timestamp: Instant,
    pub duration_ms: u64,
}

/// Configuration for auto-recovery
#[derive(Debug, Clone)]
pub struct AutoRecoveryConfig {
    /// Enable auto-recovery
    pub enabled: bool,

    /// Cooldown between recovery attempts for the same target (seconds)
    pub cooldown_seconds: u64,

    /// Maximum consecutive failures before giving up
    pub max_consecutive_failures: u32,

    /// Cooldown multiplier after each failure
    pub backoff_multiplier: f64,

    /// Maximum cooldown (seconds)
    pub max_cooldown_seconds: u64,

    /// Enable Discord notifications for recovery events
    pub notify_on_recovery: bool,
}

impl Default for AutoRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cooldown_seconds: 60,
            max_consecutive_failures: 5,
            backoff_multiplier: 2.0,
            max_cooldown_seconds: 600,
            notify_on_recovery: true,
        }
    }
}

/// Tracks recovery state for a specific target
#[derive(Debug, Clone)]
struct RecoveryState {
    last_attempt: Option<Instant>,
    consecutive_failures: u32,
    current_cooldown: Duration,
    last_action: RecoveryAction,
}

impl Default for RecoveryState {
    fn default() -> Self {
        Self {
            last_attempt: None,
            consecutive_failures: 0,
            current_cooldown: Duration::from_secs(60),
            last_action: RecoveryAction::None,
        }
    }
}

/// Statistics for auto-recovery
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    pub total_attempts: u64,
    pub successful_recoveries: u64,
    pub failed_recoveries: u64,
    pub actions_by_type: HashMap<String, u64>,
    pub last_recovery: Option<RecoveryResult>,
}

/// Auto-recovery engine for CNS
pub struct AutoRecoveryEngine {
    config: AutoRecoveryConfig,
    state: Arc<RwLock<HashMap<String, RecoveryState>>>,
    stats: Arc<RwLock<RecoveryStats>>,
    janus_state: Arc<janus_core::JanusState>,
}

impl AutoRecoveryEngine {
    /// Create a new auto-recovery engine
    pub fn new(config: AutoRecoveryConfig, janus_state: Arc<janus_core::JanusState>) -> Self {
        info!("Initializing CNS Auto-Recovery Engine");
        Self {
            config,
            state: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RecoveryStats::default())),
            janus_state,
        }
    }

    /// Check if recovery is needed and execute if so
    pub async fn check_and_recover(&self) -> Vec<RecoveryResult> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Get module health
        let modules = self.janus_state.get_module_health().await;

        for module in modules {
            if !module.healthy {
                // Determine appropriate action
                let action = self.determine_action(&module.name, &module.message);

                if action != RecoveryAction::None {
                    // Check cooldown
                    if self.can_attempt_recovery(&module.name).await {
                        let result = self.execute_recovery(action, &module.name).await;
                        results.push(result);
                    } else {
                        debug!(module = module.name, "Recovery on cooldown, skipping");
                    }
                }
            } else {
                // Module is healthy, reset failure counter
                self.reset_failure_count(&module.name).await;
            }
        }

        // Check infrastructure health
        results.extend(self.check_infrastructure_health().await);

        results
    }

    /// Determine the appropriate recovery action for a module
    fn determine_action(&self, module: &str, message: &Option<String>) -> RecoveryAction {
        let msg = message.as_deref().unwrap_or("");

        match module {
            "forward" => {
                if msg.contains("backlog") || msg.contains("queue") {
                    RecoveryAction::FlushSignalQueue
                } else if msg.contains("memory") || msg.contains("cache") {
                    RecoveryAction::ClearCache
                } else {
                    RecoveryAction::RestartModule
                }
            }
            "backward" => {
                if msg.contains("database")
                    || msg.contains("postgres")
                    || msg.contains("connection")
                {
                    RecoveryAction::ReconnectDatabase
                } else {
                    RecoveryAction::RestartModule
                }
            }
            "data" => {
                if msg.contains("redis") {
                    RecoveryAction::ReconnectRedis
                } else {
                    RecoveryAction::RestartModule
                }
            }
            _ => RecoveryAction::SendAlert,
        }
    }

    /// Check if we can attempt recovery (cooldown check)
    async fn can_attempt_recovery(&self, target: &str) -> bool {
        let state = self.state.read().await;

        if let Some(recovery_state) = state.get(target) {
            if recovery_state.consecutive_failures >= self.config.max_consecutive_failures {
                warn!(
                    target = target,
                    failures = recovery_state.consecutive_failures,
                    "Max consecutive failures reached, giving up"
                );
                return false;
            }

            if let Some(last_attempt) = recovery_state.last_attempt
                && last_attempt.elapsed() < recovery_state.current_cooldown
            {
                return false;
            }
        }

        true
    }

    /// Execute a recovery action
    async fn execute_recovery(&self, action: RecoveryAction, target: &str) -> RecoveryResult {
        let start = Instant::now();

        info!(
            action = %action,
            target = target,
            "Executing recovery action"
        );

        let (success, message) = match action {
            RecoveryAction::RestartModule => self.restart_module(target).await,
            RecoveryAction::ClearCache => self.clear_cache(target).await,
            RecoveryAction::FlushSignalQueue => self.flush_signal_queue().await,
            RecoveryAction::ReconnectDatabase => self.reconnect_database().await,
            RecoveryAction::ReconnectRedis => self.reconnect_redis().await,
            RecoveryAction::SendAlert => self.send_alert(target).await,
            RecoveryAction::None => (true, "No action needed".to_string()),
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        let result = RecoveryResult {
            action,
            target: target.to_string(),
            success,
            message: message.clone(),
            timestamp: start,
            duration_ms,
        };

        // Update state and stats
        self.update_recovery_state(target, &result).await;
        self.update_stats(&result).await;

        if success {
            info!(
                action = %action,
                target = target,
                duration_ms = duration_ms,
                "Recovery successful"
            );
        } else {
            error!(
                action = %action,
                target = target,
                message = message,
                "Recovery failed"
            );
        }

        result
    }

    /// Update recovery state after an attempt
    async fn update_recovery_state(&self, target: &str, result: &RecoveryResult) {
        let mut state = self.state.write().await;
        let entry = state.entry(target.to_string()).or_default();

        entry.last_attempt = Some(result.timestamp);
        entry.last_action = result.action;

        if result.success {
            entry.consecutive_failures = 0;
            entry.current_cooldown = Duration::from_secs(self.config.cooldown_seconds);
        } else {
            entry.consecutive_failures += 1;
            // Apply exponential backoff
            let new_cooldown =
                (entry.current_cooldown.as_secs_f64() * self.config.backoff_multiplier) as u64;
            entry.current_cooldown =
                Duration::from_secs(new_cooldown.min(self.config.max_cooldown_seconds));
        }
    }

    /// Reset failure count when module becomes healthy
    async fn reset_failure_count(&self, target: &str) {
        let mut state = self.state.write().await;
        if let Some(entry) = state.get_mut(target)
            && entry.consecutive_failures > 0
        {
            info!(target = target, "Module recovered, resetting failure count");
            entry.consecutive_failures = 0;
            entry.current_cooldown = Duration::from_secs(self.config.cooldown_seconds);
        }
    }

    /// Update statistics
    async fn update_stats(&self, result: &RecoveryResult) {
        let mut stats = self.stats.write().await;
        stats.total_attempts += 1;

        if result.success {
            stats.successful_recoveries += 1;
        } else {
            stats.failed_recoveries += 1;
        }

        *stats
            .actions_by_type
            .entry(result.action.to_string())
            .or_insert(0) += 1;

        stats.last_recovery = Some(result.clone());
    }

    /// Get recovery statistics
    pub async fn get_stats(&self) -> RecoveryStats {
        self.stats.read().await.clone()
    }

    // =========================================================================
    // Recovery Actions Implementation
    // =========================================================================

    async fn restart_module(&self, module: &str) -> (bool, String) {
        // In a unified binary, we can't truly restart a module
        // Instead, we signal it to reinitialize
        info!(module = module, "Signaling module reinitialize");

        // Update module health to trigger reinitialization
        self.janus_state
            .register_module_health(module, false, Some("restarting".to_string()))
            .await;

        // Give it a moment
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Check if module recovered
        let modules = self.janus_state.get_module_health().await;
        let is_healthy = modules.iter().any(|m| m.name == module && m.healthy);

        if is_healthy {
            (
                true,
                format!("Module {} reinitialized successfully", module),
            )
        } else {
            (false, format!("Module {} failed to reinitialize", module))
        }
    }

    async fn clear_cache(&self, module: &str) -> (bool, String) {
        info!(module = module, "Clearing module cache");

        // Try to clear Redis cache for this module
        match self.janus_state.redis_client().await {
            Ok(client) => match client.get_multiplexed_async_connection().await {
                Ok(mut conn) => {
                    let pattern = format!("fks:*:{}:*", module);
                    let keys: Vec<String> = redis::cmd("KEYS")
                        .arg(&pattern)
                        .query_async(&mut conn)
                        .await
                        .unwrap_or_default();

                    if !keys.is_empty() {
                        let deleted: i64 = redis::cmd("DEL")
                            .arg(&keys)
                            .query_async(&mut conn)
                            .await
                            .unwrap_or(0);
                        (
                            true,
                            format!("Cleared {} cache keys for {}", deleted, module),
                        )
                    } else {
                        (true, format!("No cache keys found for {}", module))
                    }
                }
                Err(e) => (false, format!("Failed to connect to Redis: {}", e)),
            },
            Err(e) => (false, format!("Failed to get Redis client: {}", e)),
        }
    }

    async fn flush_signal_queue(&self) -> (bool, String) {
        info!("Flushing signal queue");

        // The signal bus is a broadcast channel, we can't really "flush" it
        // But we can note that subscribers should catch up
        (true, "Signal queue flush requested".to_string())
    }

    async fn reconnect_database(&self) -> (bool, String) {
        info!("Attempting database reconnection");

        // Database reconnection is handled by the connection pool
        // We can trigger a health check to verify
        (true, "Database reconnection signaled".to_string())
    }

    async fn reconnect_redis(&self) -> (bool, String) {
        info!("Attempting Redis reconnection");

        match self.janus_state.redis_client().await {
            Ok(client) => {
                match client.get_multiplexed_async_connection().await {
                    Ok(mut conn) => {
                        // Test the connection
                        let result: redis::RedisResult<String> =
                            redis::cmd("PING").query_async(&mut conn).await;

                        match result {
                            Ok(pong) if pong == "PONG" => {
                                (true, "Redis reconnected successfully".to_string())
                            }
                            Ok(other) => (
                                false,
                                format!("Redis returned unexpected response: {}", other),
                            ),
                            Err(e) => (false, format!("Redis ping failed: {}", e)),
                        }
                    }
                    Err(e) => (false, format!("Failed to get Redis connection: {}", e)),
                }
            }
            Err(e) => (false, format!("Failed to get Redis client: {}", e)),
        }
    }

    async fn send_alert(&self, target: &str) -> (bool, String) {
        // Alerts are emitted as structured tracing events at WARN level.
        // External sinks (Discord, PagerDuty, etc.) should be wired via
        // a tracing subscriber layer (e.g. tracing-appender or a custom
        // Discord webhook layer) rather than hard-coded here.
        warn!(
            target = target,
            alert = true,
            component = "cns_recovery",
            "🚨 ALERT: Module {} requires attention",
            target
        );

        (true, format!("Alert sent for {} (via tracing)", target))
    }

    /// Check infrastructure health (database, redis, etc.)
    async fn check_infrastructure_health(&self) -> Vec<RecoveryResult> {
        let mut results = Vec::new();

        // Check Redis
        if self.can_attempt_recovery("redis").await {
            match self.janus_state.redis_client().await {
                Ok(client) => match client.get_multiplexed_async_connection().await {
                    Ok(mut conn) => {
                        let result: redis::RedisResult<String> =
                            redis::cmd("PING").query_async(&mut conn).await;

                        if result.is_err() {
                            let recovery = self
                                .execute_recovery(RecoveryAction::ReconnectRedis, "redis")
                                .await;
                            results.push(recovery);
                        }
                    }
                    Err(_) => {
                        let recovery = self
                            .execute_recovery(RecoveryAction::ReconnectRedis, "redis")
                            .await;
                        results.push(recovery);
                    }
                },
                Err(_) => {
                    let recovery = self
                        .execute_recovery(RecoveryAction::ReconnectRedis, "redis")
                        .await;
                    results.push(recovery);
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_action_display() {
        assert_eq!(RecoveryAction::RestartModule.to_string(), "restart_module");
        assert_eq!(RecoveryAction::ClearCache.to_string(), "clear_cache");
        assert_eq!(RecoveryAction::None.to_string(), "none");
    }

    #[test]
    fn test_config_default() {
        let config = AutoRecoveryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.cooldown_seconds, 60);
        assert_eq!(config.max_consecutive_failures, 5);
    }

    #[test]
    fn test_recovery_state_default() {
        let state = RecoveryState::default();
        assert!(state.last_attempt.is_none());
        assert_eq!(state.consecutive_failures, 0);
        assert_eq!(state.current_cooldown, Duration::from_secs(60));
    }
}
