//! # JANUS CNS (Central Nervous System) Service Library
//!
//! The Central Nervous System service for JANUS trading system.
//! Provides health monitoring, metrics collection, and auto-recovery.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    JANUS CNS Service                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │   Health     │  │   Metrics    │  │   Brain      │     │
//! │  │  Monitoring  │  │  Collection  │  │ Coordinator  │     │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
//! │         │                  │                  │             │
//! │         └──────────────────┼──────────────────┘             │
//! │                            │                                │
//! │                   ┌────────▼────────┐                       │
//! │                   │  Auto-Recovery  │                       │
//! │                   └────────┬────────┘                       │
//! │                            │                                │
//! │                      ┌─────▼─────┐                          │
//! │                      │   REST    │                          │
//! │                      │    API    │                          │
//! │                      └───────────┘                          │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod recovery;

pub use recovery::{
    AutoRecoveryConfig, AutoRecoveryEngine, RecoveryAction, RecoveryResult, RecoveryStats,
};

use anyhow::Result;
use std::sync::Arc;
use tracing::{error, info, warn};

/// CNS service version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// CNS service name
pub const SERVICE_NAME: &str = "janus-cns";

/// CNS service configuration
#[derive(Debug, Clone)]
pub struct CnsServiceConfig {
    /// Service host
    pub host: String,

    /// HTTP API port
    pub http_port: u16,

    /// Health check interval in seconds
    pub health_interval: u64,

    /// Auto-recovery enabled
    pub auto_recovery: bool,

    /// Enable metrics
    pub enable_metrics: bool,

    /// Metrics port
    pub metrics_port: u16,
}

impl Default for CnsServiceConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            http_port: 9090,
            health_interval: 10,
            auto_recovery: true,
            enable_metrics: true,
            metrics_port: 9091,
        }
    }
}

/// CNS service - Central Nervous System for health monitoring
pub struct CnsService {
    config: CnsServiceConfig,
}

impl CnsService {
    /// Create a new CNS service instance
    pub fn new(config: CnsServiceConfig) -> Self {
        info!("Initializing JANUS CNS Service v{}", VERSION);
        Self { config }
    }

    /// Get service configuration
    pub fn config(&self) -> &CnsServiceConfig {
        &self.config
    }

    /// Start the service
    ///
    /// Note: Runtime health monitoring, metrics collection, and auto-recovery
    /// are handled by the `BrainRuntime` watchdog (`crates/cns/src/watchdog.rs`)
    /// and preflight system (`crates/cns/src/preflight/`). This standalone
    /// service entry point delegates to those subsystems when started via the
    /// unified JANUS binary (see `start_module` below).
    #[tracing::instrument(skip(self), fields(host = %self.config.host, port = self.config.http_port))]
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting JANUS CNS Service on {}:{}",
            self.config.host, self.config.http_port
        );

        if self.config.auto_recovery {
            info!("Auto-recovery engine enabled (managed by BrainRuntime watchdog)");
        }
        if self.config.enable_metrics {
            info!(
                "Metrics collection enabled on port {}",
                self.config.metrics_port
            );
        }
        info!(
            "Health check interval: {}s (managed by BrainRuntime preflight & watchdog)",
            self.config.health_interval
        );

        Ok(())
    }

    /// Shutdown the service gracefully
    #[tracing::instrument(skip(self))]
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down JANUS CNS Service");
        Ok(())
    }

    /// Health check
    #[tracing::instrument(skip(self))]
    pub async fn health_check(&self) -> bool {
        true
    }
}

/// Start the CNS module as part of the unified JANUS system
///
/// This function is called by the unified JANUS binary to start the CNS health monitoring module.
#[tracing::instrument(name = "cns::start_module", skip(state))]
pub async fn start_module(state: Arc<janus_core::JanusState>) -> janus_core::Result<()> {
    info!("CNS module registered — waiting for start command...");

    state
        .register_module_health("cns", true, Some("standby".to_string()))
        .await;

    // ── Wait for services to be started via API / web interface ──────
    if !state.wait_for_services_start().await {
        info!("CNS module: shutdown requested before services started");
        state
            .register_module_health("cns", false, Some("shutdown_before_start".to_string()))
            .await;
        return Ok(());
    }

    info!("Starting CNS module with unified JANUS integration...");

    state
        .register_module_health("cns", true, Some("starting".to_string()))
        .await;

    // Create CNS service configuration from JANUS config
    let cns_config = CnsServiceConfig {
        host: "0.0.0.0".to_string(),
        http_port: state.config.ports.http + 300, // Offset to avoid conflict
        health_interval: state.config.cns.health_check_interval_secs,
        auto_recovery: state.config.cns.enable_reflexes,
        enable_metrics: false, // Metrics handled by janus-api
        metrics_port: 0,
    };

    // Create CNS service
    let service = CnsService::new(cns_config);

    // Start service
    service
        .start()
        .await
        .map_err(|e| janus_core::Error::module("cns", e.to_string()))?;

    state
        .register_module_health("cns", true, Some("running".to_string()))
        .await;

    // Create auto-recovery engine if enabled
    let recovery_engine = if state.config.cns.enable_reflexes {
        info!("Auto-recovery (reflexes) enabled");
        let recovery_config = AutoRecoveryConfig {
            enabled: true,
            cooldown_seconds: 60,
            max_consecutive_failures: 5,
            backoff_multiplier: 2.0,
            max_cooldown_seconds: 600,
            notify_on_recovery: true,
        };
        Some(Arc::new(AutoRecoveryEngine::new(
            recovery_config,
            state.clone(),
        )))
    } else {
        info!("Auto-recovery (reflexes) disabled");
        None
    };

    // Spawn health monitoring loop
    let state_clone = state.clone();
    let recovery_engine_clone = recovery_engine.clone();
    let health_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
            state_clone.config.cns.health_check_interval_secs,
        ));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Check health of all modules
                    let modules = state_clone.get_module_health().await;
                    let all_healthy = modules.iter().all(|m| m.healthy);

                    if all_healthy {
                        info!("CNS health check: All {} modules healthy", modules.len());
                    } else {
                        let unhealthy: Vec<_> = modules.iter()
                            .filter(|m| !m.healthy)
                            .map(|m| &m.name)
                            .collect();
                        warn!("CNS health check: Unhealthy modules: {:?}", unhealthy);

                        // Trigger auto-recovery if enabled
                        if let Some(ref engine) = recovery_engine_clone {
                            info!("Triggering auto-recovery for unhealthy modules...");
                            let results = engine.check_and_recover().await;
                            for result in results {
                                if result.success {
                                    info!(
                                        "Recovery action {} on {} succeeded: {}",
                                        result.action, result.target, result.message
                                    );
                                } else {
                                    error!(
                                        "Recovery action {} on {} failed: {}",
                                        result.action, result.target, result.message
                                    );
                                }
                            }

                            // Log recovery stats
                            let stats = engine.get_stats().await;
                            info!(
                                "CNS recovery stats: total={}, success={}, failed={}",
                                stats.total_attempts,
                                stats.successful_recoveries,
                                stats.failed_recoveries
                            );
                        }
                    }

                    // Update CNS module health
                    state_clone
                        .register_module_health("cns", true, Some(format!("monitoring {} modules", modules.len())))
                        .await;
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                    if state_clone.is_shutdown_requested() {
                        break;
                    }
                }
            }
        }
        info!("CNS health monitoring loop exited");
    });

    // Spawn metrics collection loop
    let state_clone = state.clone();
    let metrics_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Collect metrics
                    let health = state_clone.health_status().await;
                    info!(
                        "CNS metrics: uptime={}s, signals_generated={}, signals_persisted={}",
                        health.uptime_seconds,
                        health.signals_generated,
                        health.signals_persisted
                    );

                    // Record metrics
                    let metrics = janus_core::metrics::metrics();
                    for module in &health.modules {
                        metrics.record_module_health(
                            &module.name,
                            module.healthy,
                            state_clone.uptime_seconds() as f64,
                        );
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                    if state_clone.is_shutdown_requested() {
                        break;
                    }
                }
            }
        }
        info!("CNS metrics collection loop exited");
    });

    // Wait for shutdown
    while !state.is_shutdown_requested() {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        // Update health periodically
        state
            .register_module_health("cns", true, Some("running".to_string()))
            .await;
    }

    info!("CNS module shutting down...");

    // Cancel tasks
    health_task.abort();
    metrics_task.abort();

    // Shutdown service
    service
        .shutdown()
        .await
        .map_err(|e| janus_core::Error::module("cns", e.to_string()))?;

    state
        .register_module_health("cns", false, Some("stopped".to_string()))
        .await;

    info!("CNS module exited");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CnsServiceConfig::default();
        assert_eq!(config.http_port, 9090);
        assert_eq!(config.health_interval, 10);
        assert!(config.auto_recovery);
    }

    #[test]
    fn test_service_constants() {
        assert_eq!(SERVICE_NAME, "janus-cns");
        assert!(!VERSION.is_empty());
    }

    #[tokio::test]
    async fn test_service_creation() {
        let config = CnsServiceConfig::default();
        let service = CnsService::new(config);
        assert!(service.health_check().await);
    }
}
