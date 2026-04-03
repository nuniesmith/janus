//! Shared application state for all JANUS modules

use crate::{Config, MarketDataBus, SignalBus, metrics::metrics};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::{RwLock, watch};

// ---------------------------------------------------------------------------
// LogLevelController — trait-object interface for runtime log level changes
// ---------------------------------------------------------------------------

/// A type-erased interface for changing the tracing log filter at runtime.
///
/// This is stored in [`JanusState`] so the API module can expose a
/// `POST /api/log-level` endpoint without depending on `tracing_subscriber`
/// internals.  The concrete implementation lives in [`crate::logging`] and
/// is wired up in `main()` after `init_logging()` returns.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` because `JanusState` is shared
/// across Tokio tasks.
pub trait LogLevelController: Send + Sync {
    /// Change the operational (stdout) log filter at runtime.
    ///
    /// `filter_str` uses the same syntax as `RUST_LOG`, e.g.:
    /// - `"debug"`
    /// - `"info,janus=trace"`
    /// - `"warn,janus::supervisor=debug,hyper=info"`
    ///
    /// Returns `Ok(())` on success or a human-readable error message.
    fn set_log_level(&self, filter_str: &str) -> Result<(), String>;

    /// Returns the current filter string, if available.
    fn current_filter(&self) -> Option<String>;
}

/// Service lifecycle state — controls whether processing modules are active.
///
/// On startup JANUS enters `Standby`: the API is live but Forward, Backward,
/// CNS and Data modules wait for an explicit start command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceState {
    /// Pre-flight passed, API up, processing modules waiting for start command
    Standby,
    /// All enabled processing modules are running
    Running,
    /// Services were explicitly stopped via API
    Stopped,
}

impl std::fmt::Display for ServiceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceState::Standby => write!(f, "standby"),
            ServiceState::Running => write!(f, "running"),
            ServiceState::Stopped => write!(f, "stopped"),
        }
    }
}

/// Module health status
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModuleHealth {
    pub name: String,
    pub healthy: bool,
    #[serde(skip)]
    pub last_check: std::time::Instant,
    pub message: Option<String>,
}

/// Shared application state accessible by all modules
pub struct JanusState {
    /// Configuration
    pub config: Config,

    /// Signal broadcast bus for inter-module communication
    pub signal_bus: SignalBus,

    /// Market data broadcast bus for live data streaming (Data → Forward)
    ///
    /// The Data module publishes normalised [`MarketDataEvent`](crate::MarketDataEvent)s
    /// here when live market data ingestion is active.  The Forward module
    /// subscribes to consume them for indicator calculation and
    /// strategy-driven signal generation.
    pub market_data_bus: MarketDataBus,

    /// Service start time
    start_time: Instant,

    /// Shutdown flag
    shutdown_requested: AtomicBool,

    /// Module health statuses
    module_health: RwLock<Vec<ModuleHealth>>,

    /// Total signals generated counter
    signals_generated: AtomicU64,

    /// Total signals persisted counter
    signals_persisted: AtomicU64,

    /// Redis connection (lazy initialized)
    redis_client: RwLock<Option<redis::Client>>,

    /// Service lifecycle watch channel — sender side.
    /// Modules subscribe via `wait_for_services_start()`.
    service_state_tx: watch::Sender<ServiceState>,

    /// Service lifecycle watch channel — receiver template for cloning.
    service_state_rx: watch::Receiver<ServiceState>,

    /// Optional runtime log-level controller.
    ///
    /// Set via [`set_log_level_controller`] after `init_logging()` in `main()`.
    /// The API module reads this to expose `POST /api/log-level`.
    log_level_controller: RwLock<Option<Box<dyn LogLevelController>>>,
}

impl JanusState {
    /// Create new application state.
    ///
    /// Services start in [`ServiceState::Standby`] — the API module comes up
    /// immediately but processing modules (Forward, Backward, CNS, Data) will
    /// block on [`wait_for_services_start`] until an explicit start command is
    /// issued through the API or web interface.
    pub async fn new(config: Config) -> crate::Result<Self> {
        let signal_bus = SignalBus::new(1000);
        let market_data_bus = MarketDataBus::new(5000);
        let (service_state_tx, service_state_rx) = watch::channel(ServiceState::Standby);

        Ok(Self {
            config,
            signal_bus,
            market_data_bus,
            start_time: Instant::now(),
            shutdown_requested: AtomicBool::new(false),
            module_health: RwLock::new(Vec::new()),
            signals_generated: AtomicU64::new(0),
            signals_persisted: AtomicU64::new(0),
            redis_client: RwLock::new(None),
            service_state_tx,
            service_state_rx,
            log_level_controller: RwLock::new(None),
        })
    }

    // ── Log level control ─────────────────────────────────────────────

    /// Install a runtime log-level controller.
    ///
    /// Called once from `main()` after `init_logging()` returns.
    /// The controller is then available to the API via
    /// [`set_log_level`](Self::set_log_level) and
    /// [`current_log_filter`](Self::current_log_filter).
    pub async fn set_log_level_controller(&self, controller: Box<dyn LogLevelController>) {
        let mut guard = self.log_level_controller.write().await;
        *guard = Some(controller);
        tracing::debug!("log-level controller installed in JanusState");
    }

    /// Change the runtime log filter.
    ///
    /// Delegates to the installed [`LogLevelController`].  Returns an error
    /// if no controller has been installed yet.
    pub async fn set_log_level(&self, filter_str: &str) -> Result<(), String> {
        let guard = self.log_level_controller.read().await;
        match guard.as_ref() {
            Some(ctrl) => ctrl.set_log_level(filter_str),
            None => Err("no log-level controller installed".to_string()),
        }
    }

    /// Returns the current log filter string, if a controller is installed.
    pub async fn current_log_filter(&self) -> Option<String> {
        let guard = self.log_level_controller.read().await;
        guard.as_ref().and_then(|ctrl| ctrl.current_filter())
    }

    // ── Uptime & shutdown ─────────────────────────────────────────────

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Check if shutdown has been requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }

    /// Request shutdown
    pub fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
    }

    /// Perform graceful shutdown
    pub async fn shutdown(&self) -> crate::Result<()> {
        tracing::info!("Initiating graceful shutdown...");
        self.request_shutdown();

        // Also move services to Stopped so any waiting modules unblock
        let _ = self.service_state_tx.send(ServiceState::Stopped);

        // Close Redis connection if open and update metric
        let mut redis = self.redis_client.write().await;
        if redis.is_some() {
            metrics().redis_connected.set(0.0);
        }
        *redis = None;

        tracing::info!("Shutdown complete");
        Ok(())
    }

    // ── Service lifecycle management ──────────────────────────────────

    /// Transition processing services to [`ServiceState::Running`].
    ///
    /// All modules blocked on [`wait_for_services_start`] will proceed.
    /// Returns `true` if the state actually changed.
    pub fn start_services(&self) -> bool {
        self.service_state_tx.send_if_modified(|current| {
            if *current == ServiceState::Running {
                false
            } else {
                tracing::info!("Service state: {} → running", current);
                *current = ServiceState::Running;
                true
            }
        })
    }

    /// Transition processing services to [`ServiceState::Stopped`].
    ///
    /// Modules should check [`are_services_active`] in their hot loops and
    /// wind down gracefully when it returns `false`.
    /// Returns `true` if the state actually changed.
    pub fn stop_services(&self) -> bool {
        self.service_state_tx.send_if_modified(|current| {
            if *current == ServiceState::Stopped {
                false
            } else {
                tracing::info!("Service state: {} → stopped", current);
                *current = ServiceState::Stopped;
                true
            }
        })
    }

    /// Returns the current [`ServiceState`].
    pub fn service_state(&self) -> ServiceState {
        *self.service_state_tx.borrow()
    }

    /// Returns `true` when processing modules should be actively running.
    pub fn are_services_active(&self) -> bool {
        *self.service_state_tx.borrow() == ServiceState::Running
    }

    /// Block until services are started (state becomes [`ServiceState::Running`])
    /// **or** a shutdown is requested.
    ///
    /// Returns `true` if services were started, `false` if a shutdown was
    /// requested while still waiting.
    pub async fn wait_for_services_start(&self) -> bool {
        let mut rx = self.service_state_rx.clone();

        // Fast path — already running
        if *rx.borrow_and_update() == ServiceState::Running {
            return true;
        }

        loop {
            tokio::select! {
                result = rx.changed() => {
                    match result {
                        Ok(()) => {
                            if *rx.borrow() == ServiceState::Running {
                                return true;
                            }
                            // State changed to something other than Running
                            // (e.g. Stopped) — keep waiting unless shutdown
                            if self.is_shutdown_requested() {
                                return false;
                            }
                        }
                        Err(_) => {
                            // Sender dropped — treat as shutdown
                            return false;
                        }
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(250)) => {
                    if self.is_shutdown_requested() {
                        return false;
                    }
                }
            }
        }
    }

    /// Subscribe to service-state changes.
    ///
    /// Useful for modules that need to react to stop commands mid-loop.
    pub fn subscribe_service_state(&self) -> watch::Receiver<ServiceState> {
        self.service_state_rx.clone()
    }

    /// Register a module's health status
    pub async fn register_module_health(
        &self,
        name: impl Into<String>,
        healthy: bool,
        message: Option<String>,
    ) {
        let mut health = self.module_health.write().await;
        let name = name.into();

        // Update existing or add new
        if let Some(existing) = health.iter_mut().find(|h| h.name == name) {
            existing.healthy = healthy;
            existing.last_check = Instant::now();
            existing.message = message;
        } else {
            health.push(ModuleHealth {
                name,
                healthy,
                last_check: Instant::now(),
                message,
            });
        }
    }

    /// Get all module health statuses
    pub async fn get_module_health(&self) -> Vec<ModuleHealth> {
        self.module_health.read().await.clone()
    }

    /// Check if all modules are healthy
    pub async fn all_modules_healthy(&self) -> bool {
        let health = self.module_health.read().await;
        health.iter().all(|h| h.healthy)
    }

    /// Increment signals generated counter
    pub fn increment_signals_generated(&self) {
        self.signals_generated.fetch_add(1, Ordering::SeqCst);
    }

    /// Get signals generated count
    pub fn signals_generated(&self) -> u64 {
        self.signals_generated.load(Ordering::SeqCst)
    }

    /// Increment signals persisted counter
    pub fn increment_signals_persisted(&self) {
        self.signals_persisted.fetch_add(1, Ordering::SeqCst);
    }

    /// Get signals persisted count
    pub fn signals_persisted(&self) -> u64 {
        self.signals_persisted.load(Ordering::SeqCst)
    }

    /// Get or create the Redis client handle.
    ///
    /// This is intentionally cheap — `redis::Client::open` only parses
    /// the URL; it does **not** open a TCP connection.  Callers obtain
    /// an actual connection via `client.get_multiplexed_async_connection()`.
    pub async fn redis_client(&self) -> crate::Result<redis::Client> {
        let mut client = self.redis_client.write().await;

        if client.is_none() {
            match redis::Client::open(self.config.redis.url.as_str()) {
                Ok(new_client) => {
                    *client = Some(new_client);
                }
                Err(e) => {
                    metrics().redis_connected.set(0.0);
                    return Err(e.into());
                }
            }
        }

        Ok(client.as_ref().unwrap().clone())
    }

    /// Probe Redis connectivity and update the `janus_redis_connected`
    /// Prometheus gauge.
    ///
    /// Call once at startup (from `main()`) and optionally from periodic
    /// health-check loops.  The method never returns an error — it logs
    /// warnings and sets the gauge to `0` on failure so the process can
    /// continue booting even if Redis is temporarily unavailable.
    pub async fn probe_redis(&self) {
        let client = match self.redis_client().await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Redis probe: failed to create client — {e}");
                metrics().redis_connected.set(0.0);
                return;
            }
        };

        match client.get_multiplexed_async_connection().await {
            Ok(mut conn) => {
                // Actual PING to confirm the server responds.
                let pong: Result<String, _> = redis::cmd("PING").query_async(&mut conn).await;
                match pong {
                    Ok(_) => {
                        metrics().redis_connected.set(1.0);
                        tracing::info!("Redis probe: connected ✓");
                    }
                    Err(e) => {
                        metrics().redis_connected.set(0.0);
                        tracing::warn!("Redis probe: PING failed — {e}");
                    }
                }
            }
            Err(e) => {
                metrics().redis_connected.set(0.0);
                tracing::warn!("Redis probe: connection failed — {e}");
            }
        }
    }

    /// Get comprehensive health status
    pub async fn health_status(&self) -> HealthStatus {
        let module_health = self.get_module_health().await;
        let all_healthy = module_health.iter().all(|h| h.healthy);

        HealthStatus {
            status: if all_healthy { "healthy" } else { "degraded" }.to_string(),
            uptime_seconds: self.uptime_seconds(),
            signals_generated: self.signals_generated(),
            signals_persisted: self.signals_persisted(),
            modules: module_health
                .iter()
                .map(|h| ModuleHealthSummary {
                    name: h.name.clone(),
                    healthy: h.healthy,
                    message: h.message.clone(),
                })
                .collect(),
            shutdown_requested: self.is_shutdown_requested(),
            service_state: self.service_state(),
        }
    }
}

/// Health status response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub uptime_seconds: u64,
    pub signals_generated: u64,
    pub signals_persisted: u64,
    pub modules: Vec<ModuleHealthSummary>,
    pub shutdown_requested: bool,
    pub service_state: ServiceState,
}

/// Module health summary for API responses
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModuleHealthSummary {
    pub name: String,
    pub healthy: bool,
    pub message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_state_creation() {
        let config = Config::default();
        let state = JanusState::new(config).await.unwrap();

        assert!(!state.is_shutdown_requested());
        assert_eq!(state.signals_generated(), 0);
        assert_eq!(state.service_state(), ServiceState::Standby);
        assert!(!state.are_services_active());
    }

    #[tokio::test]
    async fn test_service_lifecycle() {
        let config = Config::default();
        let state = JanusState::new(config).await.unwrap();

        // Starts in standby
        assert_eq!(state.service_state(), ServiceState::Standby);
        assert!(!state.are_services_active());

        // Start services
        assert!(state.start_services());
        assert_eq!(state.service_state(), ServiceState::Running);
        assert!(state.are_services_active());

        // Idempotent
        assert!(!state.start_services());

        // Stop services
        assert!(state.stop_services());
        assert_eq!(state.service_state(), ServiceState::Stopped);
        assert!(!state.are_services_active());

        // Idempotent
        assert!(!state.stop_services());
    }

    #[tokio::test]
    async fn test_wait_for_services_start() {
        let config = Config::default();
        let state = std::sync::Arc::new(JanusState::new(config).await.unwrap());

        let state2 = state.clone();
        let handle = tokio::spawn(async move { state2.wait_for_services_start().await });

        // Give the waiter a moment to park
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Start services — waiter should unblock
        state.start_services();
        let result = tokio::time::timeout(tokio::time::Duration::from_secs(2), handle)
            .await
            .expect("timed out")
            .expect("task panicked");

        assert!(result);
    }

    #[tokio::test]
    async fn test_module_health_registration() {
        let config = Config::default();
        let state = JanusState::new(config).await.unwrap();

        state.register_module_health("forward", true, None).await;
        state
            .register_module_health("backward", true, Some("running".to_string()))
            .await;

        let health = state.get_module_health().await;
        assert_eq!(health.len(), 2);
        assert!(state.all_modules_healthy().await);
    }

    #[tokio::test]
    async fn test_signal_counters() {
        let config = Config::default();
        let state = JanusState::new(config).await.unwrap();

        state.increment_signals_generated();
        state.increment_signals_generated();
        state.increment_signals_persisted();

        assert_eq!(state.signals_generated(), 2);
        assert_eq!(state.signals_persisted(), 1);
    }
}
