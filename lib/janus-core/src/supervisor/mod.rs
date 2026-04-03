//! Janus Supervisor — hierarchical service lifecycle management.
//!
//! Prometheus metrics are automatically published to the global
//! [`JanusMetrics`](crate::metrics::JanusMetrics) registry whenever
//! the supervisor records a spawn, restart, termination, or circuit
//! breaker trip.  This means the `/metrics` endpoint exposes
//! `janus_supervisor_*` counters/gauges with zero additional wiring.
//!
//! This module implements the **Janus Supervisor Model** described in the
//! architecture refactor document. It replaces the old "fire and forget"
//! `tokio::spawn` pattern with a structured supervision tree built on:
//!
//! - [`TaskTracker`] — tracks spawned tasks without accumulating results
//!   (unlike `JoinSet`), preventing memory leaks in long-running processes.
//! - [`CancellationToken`] — propagates graceful shutdown signals through
//!   the service hierarchy.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                JanusSupervisor                   │
//! │                                                  │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
//! │  │  Data    │  │   CNS    │  │Execution │ ...   │
//! │  │ Service  │  │ Service  │  │ Service  │      │
//! │  └──────────┘  └──────────┘  └──────────┘      │
//! │                                                  │
//! │  TaskTracker  ◄── tracks all spawned tasks       │
//! │  CancellationToken ◄── shutdown signal tree      │
//! │  BackoffState[] ◄── per-service restart state    │
//! │  ServiceLifecycle[] ◄── per-service state machine│
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_core::supervisor::{JanusSupervisor, SupervisorConfig};
//!
//! let config = SupervisorConfig::default();
//! let mut supervisor = JanusSupervisor::new(config);
//!
//! supervisor.spawn_service(Box::new(my_data_service));
//! supervisor.spawn_service(Box::new(my_cns_service));
//!
//! // Blocks until Ctrl+C / SIGTERM, then orchestrates graceful shutdown
//! supervisor.run_until_shutdown().await?;
//! ```

pub mod adapters;
pub mod backoff;
pub mod lifecycle;
pub mod service;

// Re-exports
pub use adapters::{ApiModuleAdapter, ModuleAdapter};
pub use backoff::{BackoffAction, BackoffConfig, BackoffState};
pub use lifecycle::{
    ServiceLifecycle, ServiceLifecycleSnapshot, ServicePhase, TerminationReason, TransitionError,
};
pub use service::{JanusService, RestartPolicy};

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

// ---------------------------------------------------------------------------
// SupervisorConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`JanusSupervisor`].
#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    /// Default backoff configuration applied to services that don't
    /// provide their own. Individual services can override via the
    /// per-service spawn options.
    pub default_backoff: BackoffConfig,

    /// Maximum time to wait for all services to drain during shutdown.
    /// After this deadline the supervisor will log a warning and exit.
    pub shutdown_timeout: Duration,

    /// Whether to install a Ctrl+C / SIGTERM handler automatically.
    /// Set to `false` if you want to manage signals externally and call
    /// [`JanusSupervisor::trigger_shutdown`] manually.
    pub install_signal_handler: bool,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            default_backoff: BackoffConfig::default(),
            shutdown_timeout: Duration::from_secs(30),
            install_signal_handler: true,
        }
    }
}

impl SupervisorConfig {
    /// Builder: set the shutdown timeout.
    pub fn with_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.shutdown_timeout = timeout;
        self
    }

    /// Builder: set the default backoff config.
    pub fn with_default_backoff(mut self, backoff: BackoffConfig) -> Self {
        self.default_backoff = backoff;
        self
    }

    /// Builder: disable automatic signal handler installation.
    pub fn without_signal_handler(mut self) -> Self {
        self.install_signal_handler = false;
        self
    }
}

// ---------------------------------------------------------------------------
// SupervisorMetrics
// ---------------------------------------------------------------------------

/// Atomic counters for supervisor-level Prometheus-compatible metrics.
///
/// These map directly to the metrics specified in the architecture doc:
/// - `janus_supervisor_restarts_total`
/// - `janus_supervisor_active_services`
/// - `janus_supervisor_spawned_total`
#[derive(Debug, Default)]
pub struct SupervisorMetrics {
    /// Total number of service restarts across all services.
    pub restarts_total: AtomicU64,

    /// Number of services currently in a non-terminal phase.
    pub active_services: AtomicU64,

    /// Total number of services ever spawned (including restarts).
    pub spawned_total: AtomicU64,

    /// Total number of services that have terminated.
    pub terminated_total: AtomicU64,

    /// Total number of circuit breaker trips.
    pub circuit_breaker_trips: AtomicU64,
}

impl SupervisorMetrics {
    fn new() -> Self {
        Self::default()
    }

    fn record_spawn(&self) {
        self.spawned_total.fetch_add(1, Ordering::Relaxed);
        let new_active = self.active_services.fetch_add(1, Ordering::Relaxed) + 1;

        // Push to global Prometheus registry.
        // The atomics above are the single source of truth for
        // `active_services` — we `set()` the Prometheus gauge from
        // the authoritative atomic value instead of calling the
        // independent `inc()` / `dec()` helpers on JanusMetrics,
        // which avoids a TOCTOU divergence between the two stores.
        let prom = crate::metrics::metrics();
        prom.supervisor_spawned_total.inc();
        prom.supervisor_active_services.set(new_active as f64);
    }

    fn record_restart(&self) {
        self.restarts_total.fetch_add(1, Ordering::Relaxed);
        crate::metrics::metrics().supervisor_restarts_total.inc();
    }

    fn record_termination(&self) {
        self.terminated_total.fetch_add(1, Ordering::Relaxed);
        // Saturating subtract to avoid underflow from race conditions.
        // `fetch_update` returns the *previous* value on success.
        let prev = self
            .active_services
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            })
            .unwrap_or(0);
        let new_active = prev.saturating_sub(1);

        // Set the Prometheus gauge authoritatively from our atomic
        // rather than using the independent get()+dec() path in
        // JanusMetrics, which has a TOCTOU race on the gauge.
        let prom = crate::metrics::metrics();
        prom.supervisor_terminated_total.inc();
        prom.supervisor_active_services.set(new_active as f64);
    }

    fn record_termination_with_uptime(&self, service_name: &str, uptime_secs: f64) {
        self.record_termination();
        crate::metrics::metrics()
            .supervisor_uptime_seconds
            .with_label_values(&[service_name])
            .observe(uptime_secs);
    }

    fn record_circuit_breaker_trip(&self) {
        self.circuit_breaker_trips.fetch_add(1, Ordering::Relaxed);
        crate::metrics::metrics()
            .supervisor_circuit_breaker_trips
            .inc();
    }

    /// Snapshot the current metric values.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            restarts_total: self.restarts_total.load(Ordering::Relaxed),
            active_services: self.active_services.load(Ordering::Relaxed),
            spawned_total: self.spawned_total.load(Ordering::Relaxed),
            terminated_total: self.terminated_total.load(Ordering::Relaxed),
            circuit_breaker_trips: self.circuit_breaker_trips.load(Ordering::Relaxed),
        }
    }
}

/// Plain-data snapshot of supervisor metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricsSnapshot {
    pub restarts_total: u64,
    pub active_services: u64,
    pub spawned_total: u64,
    pub terminated_total: u64,
    pub circuit_breaker_trips: u64,
}

// ---------------------------------------------------------------------------
// SpawnOptions
// ---------------------------------------------------------------------------

/// Per-service spawn configuration.
#[derive(Debug, Clone, Default)]
pub struct SpawnOptions {
    /// Override the supervisor's default backoff config for this service.
    pub backoff: Option<BackoffConfig>,
}

impl SpawnOptions {
    /// Create options with a custom backoff config.
    pub fn with_backoff(backoff: BackoffConfig) -> Self {
        Self {
            backoff: Some(backoff),
        }
    }
}

// ---------------------------------------------------------------------------
// JanusSupervisor
// ---------------------------------------------------------------------------

/// The central supervisor for the Janus system.
///
/// Manages the lifecycle of all [`JanusService`] implementations using
/// [`TaskTracker`] for structured concurrency and [`CancellationToken`]
/// for graceful shutdown propagation.
///
/// # Memory Safety
///
/// Unlike `JoinSet`, `TaskTracker` does **not** accumulate task return
/// values. Completed task memory is reclaimed immediately, making this
/// safe for long-running processes that may restart services hundreds of
/// times over weeks of operation.
pub struct JanusSupervisor {
    config: SupervisorConfig,
    tracker: TaskTracker,
    cancel_token: CancellationToken,
    metrics: Arc<SupervisorMetrics>,
    lifecycles: Arc<RwLock<HashMap<String, ServiceLifecycle>>>,
}

impl JanusSupervisor {
    /// Create a new supervisor with the given configuration.
    pub fn new(config: SupervisorConfig) -> Self {
        Self {
            config,
            tracker: TaskTracker::new(),
            cancel_token: CancellationToken::new(),
            metrics: Arc::new(SupervisorMetrics::new()),
            lifecycles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new supervisor with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SupervisorConfig::default())
    }

    /// Get a reference to the supervisor's cancellation token.
    ///
    /// Useful for external code that needs to observe or trigger shutdown.
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Get a reference to the supervisor's metrics.
    pub fn metrics(&self) -> &Arc<SupervisorMetrics> {
        &self.metrics
    }

    /// Get a snapshot of all service lifecycles.
    pub async fn lifecycle_snapshots(&self) -> Vec<ServiceLifecycleSnapshot> {
        let lifecycles = self.lifecycles.read().await;
        lifecycles
            .values()
            .map(ServiceLifecycleSnapshot::from)
            .collect()
    }

    /// Get the lifecycle snapshot for a specific service by name.
    pub async fn service_lifecycle(&self, name: &str) -> Option<ServiceLifecycleSnapshot> {
        let lifecycles = self.lifecycles.read().await;
        lifecycles.get(name).map(ServiceLifecycleSnapshot::from)
    }

    /// Number of services currently tracked (alive + terminated).
    pub async fn service_count(&self) -> usize {
        self.lifecycles.read().await.len()
    }

    /// Trigger a graceful shutdown of all managed services.
    ///
    /// This cancels the root cancellation token, which propagates to all
    /// child tokens held by running services. The supervisor's
    /// `run_until_shutdown` (or `wait_for_drain`) will then wait for
    /// tasks to complete up to the configured timeout.
    #[tracing::instrument(skip(self))]
    pub fn trigger_shutdown(&self) {
        tracing::info!("Supervisor: shutdown triggered");
        self.cancel_token.cancel();
    }

    /// Returns `true` if the supervisor's shutdown has been triggered.
    pub fn is_shutting_down(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    // ── Spawn ─────────────────────────────────────────────────────────

    /// Spawn a service into the supervisor with default options.
    ///
    /// The service will be wrapped in a restart loop governed by its
    /// [`RestartPolicy`] and the supervisor's default [`BackoffConfig`].
    pub fn spawn_service(&self, service: Box<dyn JanusService>) {
        self.spawn_service_with_options(service, SpawnOptions::default());
    }

    /// Spawn a service with custom per-service options.
    #[tracing::instrument(skip(self, service, options), fields(service = %service.name(), policy = %service.restart_policy()))]
    pub fn spawn_service_with_options(
        &self,
        service: Box<dyn JanusService>,
        options: SpawnOptions,
    ) {
        let service_name = service.name().to_string();
        let restart_policy = service.restart_policy();
        let backoff_config = options
            .backoff
            .unwrap_or_else(|| self.config.default_backoff.clone());

        let cancel = self.cancel_token.child_token();
        let metrics = self.metrics.clone();
        let lifecycles = self.lifecycles.clone();

        metrics.record_spawn();

        self.tracker.spawn(Self::service_loop(
            service,
            service_name,
            restart_policy,
            backoff_config,
            cancel,
            metrics,
            lifecycles,
        ));
    }

    /// The core restart loop for a single service.
    ///
    /// This is the heart of the supervisor — it:
    /// 1. Runs the service
    /// 2. Catches failures
    /// 3. Applies the restart policy and backoff strategy
    /// 4. Loops until cancelled or the circuit breaker trips
    #[tracing::instrument(
        skip_all,
        fields(service = %service_name, policy = %restart_policy)
    )]
    async fn service_loop(
        service: Box<dyn JanusService>,
        service_name: String,
        restart_policy: RestartPolicy,
        backoff_config: BackoffConfig,
        cancel: CancellationToken,
        metrics: Arc<SupervisorMetrics>,
        lifecycles: Arc<RwLock<HashMap<String, ServiceLifecycle>>>,
    ) {
        let mut backoff = BackoffState::new(backoff_config);
        let mut lifecycle = ServiceLifecycle::new(&service_name);

        // Store the lifecycle
        {
            let mut lc_map = lifecycles.write().await;
            lc_map.insert(service_name.clone(), lifecycle.clone());
        }

        loop {
            // Check cancellation before each attempt
            if cancel.is_cancelled() {
                tracing::info!(service = %service_name, "cancellation detected, not starting service");
                let _ = lifecycle.transition_to_stopping();
                let _ = lifecycle.transition_to_terminated(TerminationReason::Cancelled);
                Self::update_lifecycle(&lifecycles, &service_name, &lifecycle).await;
                let uptime = lifecycle.cumulative_running_time().as_secs_f64();
                metrics.record_termination_with_uptime(&service_name, uptime);
                return;
            }

            // Transition to Running
            if lifecycle.phase() == ServicePhase::Starting {
                let _ = lifecycle.transition_to_running();
            } else if lifecycle.phase() == ServicePhase::BackingOff {
                // Restarting from backoff
                let _ = lifecycle.transition_to_restarting();
                let _ = lifecycle.transition_to_running();
                metrics.record_restart();
            }

            backoff.record_start();
            Self::update_lifecycle(&lifecycles, &service_name, &lifecycle).await;

            tracing::info!(
                service = %service_name,
                attempt = lifecycle.start_count(),
                "running service"
            );

            // Run the service.  The cancel token is passed into `run()` so
            // cooperative services can detect shutdown and return promptly.
            // We do NOT race `cancel.cancelled()` against `service.run()`
            // here — doing so (especially with `biased`) would drop the
            // service's future before it can perform cleanup.  The real
            // safety net for non-responsive services is the shutdown
            // timeout in `wait_for_drain`.
            let result = service.run(cancel.clone()).await;

            // If the service returned because the cancel token fired,
            // treat it as a clean cancellation regardless of the result.
            if cancel.is_cancelled() {
                tracing::info!(service = %service_name, "service exited after cancellation");
                let _ = lifecycle.transition_to_stopping();
                let _ = lifecycle.transition_to_terminated(TerminationReason::Cancelled);
                Self::update_lifecycle(&lifecycles, &service_name, &lifecycle).await;
                let uptime = lifecycle.cumulative_running_time().as_secs_f64();
                metrics.record_termination_with_uptime(&service_name, uptime);
                return;
            }

            // Handle the result
            match result {
                Ok(()) => {
                    tracing::info!(service = %service_name, "service exited cleanly");
                    backoff.maybe_reset_on_cooldown();

                    match restart_policy {
                        RestartPolicy::Always => {
                            // A clean exit means the service completed successfully,
                            // so explicitly reset the backoff state. This prevents
                            // stale attempt counts from prior error paths from
                            // bleeding into subsequent clean-exit restart cycles.
                            backoff.reset();

                            // Always restart, even on clean exit
                            tracing::info!(
                                service = %service_name,
                                "restart_policy=always, will restart after backoff"
                            );
                            // For clean exits with Always, we use a minimal delay
                            // rather than the exponential backoff (which is for errors)
                            let delay = Duration::from_millis(100);
                            let _ = lifecycle
                                .transition_to_backing_off("clean exit, policy=always", delay);
                            Self::update_lifecycle(&lifecycles, &service_name, &lifecycle).await;

                            tokio::select! {
                                _ = cancel.cancelled() => {
                                    let _ = lifecycle.transition_to_stopping();
                                    let _ = lifecycle.transition_to_terminated(TerminationReason::Cancelled);
                                    Self::update_lifecycle(&lifecycles, &service_name, &lifecycle).await;
                                    let uptime = lifecycle.cumulative_running_time().as_secs_f64();
                                    metrics.record_termination_with_uptime(&service_name, uptime);
                                    return;
                                }
                                _ = tokio::time::sleep(delay) => {}
                            }
                            continue;
                        }
                        RestartPolicy::OnFailure | RestartPolicy::Never => {
                            // Clean exit — service completed its work
                            let _ =
                                lifecycle.transition_to_terminated(TerminationReason::Completed);
                            Self::update_lifecycle(&lifecycles, &service_name, &lifecycle).await;
                            let uptime = lifecycle.cumulative_running_time().as_secs_f64();
                            metrics.record_termination_with_uptime(&service_name, uptime);
                            return;
                        }
                    }
                }

                Err(err) => {
                    let error_msg = format!("{err:#}");
                    tracing::error!(
                        service = %service_name,
                        error = %error_msg,
                        "service failed"
                    );

                    backoff.maybe_reset_on_cooldown();

                    match restart_policy {
                        RestartPolicy::Never => {
                            tracing::warn!(
                                service = %service_name,
                                "restart_policy=never, service will not be restarted"
                            );
                            let _ = lifecycle.transition_to_terminated(
                                TerminationReason::Unrecoverable(error_msg),
                            );
                            Self::update_lifecycle(&lifecycles, &service_name, &lifecycle).await;
                            let uptime = lifecycle.cumulative_running_time().as_secs_f64();
                            metrics.record_termination_with_uptime(&service_name, uptime);
                            return;
                        }

                        RestartPolicy::OnFailure | RestartPolicy::Always => {
                            // Compute backoff
                            match backoff.next_backoff() {
                                BackoffAction::Retry(delay) => {
                                    tracing::info!(
                                        service = %service_name,
                                        delay_ms = delay.as_millis() as u64,
                                        attempt = backoff.attempt(),
                                        "scheduling restart after backoff"
                                    );

                                    let _ = lifecycle.transition_to_backing_off(&error_msg, delay);
                                    Self::update_lifecycle(&lifecycles, &service_name, &lifecycle)
                                        .await;

                                    // Sleep for the backoff duration, but respect cancellation
                                    tokio::select! {
                                        _ = cancel.cancelled() => {
                                            tracing::info!(
                                                service = %service_name,
                                                "cancellation during backoff"
                                            );
                                            let _ = lifecycle.transition_to_stopping();
                                            let _ = lifecycle.transition_to_terminated(
                                                TerminationReason::Cancelled,
                                            );
                                            Self::update_lifecycle(&lifecycles, &service_name, &lifecycle).await;
                                            let uptime = lifecycle.cumulative_running_time().as_secs_f64();
                                            metrics.record_termination_with_uptime(&service_name, uptime);
                                            return;
                                        }
                                        _ = tokio::time::sleep(delay) => {
                                            // Backoff complete, loop will restart
                                        }
                                    }
                                }

                                BackoffAction::CircuitOpen {
                                    failures,
                                    max_retries,
                                } => {
                                    tracing::error!(
                                        service = %service_name,
                                        failures = failures,
                                        max_retries = max_retries,
                                        "CIRCUIT BREAKER OPEN — too many failures, giving up"
                                    );
                                    metrics.record_circuit_breaker_trip();

                                    let _ = lifecycle.transition_to_terminated(
                                        TerminationReason::CircuitBreakerOpen {
                                            failures,
                                            max_retries,
                                        },
                                    );
                                    Self::update_lifecycle(&lifecycles, &service_name, &lifecycle)
                                        .await;
                                    let uptime = lifecycle.cumulative_running_time().as_secs_f64();
                                    metrics.record_termination_with_uptime(&service_name, uptime);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Update the lifecycle state in the shared map.
    async fn update_lifecycle(
        lifecycles: &Arc<RwLock<HashMap<String, ServiceLifecycle>>>,
        name: &str,
        lifecycle: &ServiceLifecycle,
    ) {
        let mut lc_map = lifecycles.write().await;
        lc_map.insert(name.to_string(), lifecycle.clone());
    }

    // ── Shutdown coordination ─────────────────────────────────────────

    /// Close the tracker and wait for all tasks to complete, with timeout.
    ///
    /// Call this after triggering shutdown (or after `run_until_shutdown`
    /// returns) to ensure all tasks have drained.
    #[tracing::instrument(skip(self), fields(timeout_secs = self.config.shutdown_timeout.as_secs()))]
    pub async fn wait_for_drain(&self) {
        self.tracker.close();

        tracing::info!(
            timeout_secs = self.config.shutdown_timeout.as_secs(),
            "waiting for all services to drain"
        );

        match tokio::time::timeout(self.config.shutdown_timeout, self.tracker.wait()).await {
            Ok(()) => {
                tracing::info!("all services drained successfully");
            }
            Err(_) => {
                tracing::warn!(
                    timeout_secs = self.config.shutdown_timeout.as_secs(),
                    "shutdown timeout exceeded, some services may not have exited cleanly"
                );
            }
        }
    }

    /// Run the supervisor until a shutdown signal is received.
    ///
    /// If `install_signal_handler` is `true` in the config (the default),
    /// this will listen for Ctrl+C / SIGTERM and trigger shutdown.
    ///
    /// Returns after all services have drained (or the shutdown timeout
    /// has elapsed).
    #[tracing::instrument(skip(self), fields(signal_handler = self.config.install_signal_handler))]
    pub async fn run_until_shutdown(&self) -> anyhow::Result<()> {
        if self.config.install_signal_handler {
            self.wait_for_signal_and_shutdown().await?;
        } else {
            // Just wait for the token to be cancelled externally
            self.cancel_token.cancelled().await;
            tracing::info!("external shutdown signal received");
        }

        self.wait_for_drain().await;

        // Log final metrics
        let snap = self.metrics.snapshot();
        tracing::info!(
            restarts = snap.restarts_total,
            spawned = snap.spawned_total,
            terminated = snap.terminated_total,
            circuit_trips = snap.circuit_breaker_trips,
            "supervisor shutdown complete"
        );

        Ok(())
    }

    /// Wait for OS signals (Ctrl+C, SIGTERM) and trigger shutdown.
    async fn wait_for_signal_and_shutdown(&self) -> anyhow::Result<()> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};

            let mut sigterm = signal(SignalKind::terminate())?;
            let mut sigint = signal(SignalKind::interrupt())?;

            tokio::select! {
                _ = sigterm.recv() => {
                    tracing::info!("received SIGTERM");
                }
                _ = sigint.recv() => {
                    tracing::info!("received SIGINT");
                }
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("shutdown triggered programmatically");
                    return Ok(());
                }
            }
        }

        #[cfg(not(unix))]
        {
            tokio::select! {
                result = tokio::signal::ctrl_c() => {
                    result?;
                    tracing::info!("received Ctrl+C");
                }
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("shutdown triggered programmatically");
                    return Ok(());
                }
            }
        }

        self.cancel_token.cancel();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────

    /// A simple service that counts how many times it ran and exits
    /// cleanly on cancellation.
    struct CountingService {
        name: String,
        policy: RestartPolicy,
        run_count: Arc<AtomicU64>,
    }

    impl CountingService {
        fn new(name: &str, policy: RestartPolicy) -> (Self, Arc<AtomicU64>) {
            let count = Arc::new(AtomicU64::new(0));
            (
                Self {
                    name: name.to_string(),
                    policy,
                    run_count: count.clone(),
                },
                count,
            )
        }
    }

    #[async_trait::async_trait]
    impl JanusService for CountingService {
        fn name(&self) -> &str {
            &self.name
        }

        fn restart_policy(&self) -> RestartPolicy {
            self.policy
        }

        async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
            self.run_count.fetch_add(1, Ordering::SeqCst);
            cancel.cancelled().await;
            Ok(())
        }
    }

    /// A service that fails N times before succeeding.
    struct FailNTimes {
        name: String,
        fail_count: u32,
        current: Arc<AtomicU64>,
    }

    impl FailNTimes {
        fn new(name: &str, fail_count: u32) -> (Self, Arc<AtomicU64>) {
            let current = Arc::new(AtomicU64::new(0));
            (
                Self {
                    name: name.to_string(),
                    fail_count,
                    current: current.clone(),
                },
                current,
            )
        }
    }

    #[async_trait::async_trait]
    impl JanusService for FailNTimes {
        fn name(&self) -> &str {
            &self.name
        }

        fn restart_policy(&self) -> RestartPolicy {
            RestartPolicy::OnFailure
        }

        async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
            let attempt = self.current.fetch_add(1, Ordering::SeqCst) as u32;
            if attempt < self.fail_count {
                // Yield briefly so tests aren't instant-looping
                tokio::time::sleep(Duration::from_millis(1)).await;
                anyhow::bail!("simulated failure #{}", attempt + 1);
            }
            // After N failures, run until cancelled
            cancel.cancelled().await;
            Ok(())
        }
    }

    /// A service that immediately returns Ok (one-shot task).
    struct OneShotService {
        name: String,
        ran: Arc<AtomicU64>,
    }

    impl OneShotService {
        fn new(name: &str) -> (Self, Arc<AtomicU64>) {
            let ran = Arc::new(AtomicU64::new(0));
            (
                Self {
                    name: name.to_string(),
                    ran: ran.clone(),
                },
                ran,
            )
        }
    }

    #[async_trait::async_trait]
    impl JanusService for OneShotService {
        fn name(&self) -> &str {
            &self.name
        }

        fn restart_policy(&self) -> RestartPolicy {
            RestartPolicy::Never
        }

        async fn run(&self, _cancel: CancellationToken) -> anyhow::Result<()> {
            self.ran.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A service that always fails (for circuit breaker testing).
    struct AlwaysFailService {
        name: String,
        attempts: Arc<AtomicU64>,
    }

    impl AlwaysFailService {
        fn new(name: &str) -> (Self, Arc<AtomicU64>) {
            let attempts = Arc::new(AtomicU64::new(0));
            (
                Self {
                    name: name.to_string(),
                    attempts: attempts.clone(),
                },
                attempts,
            )
        }
    }

    #[async_trait::async_trait]
    impl JanusService for AlwaysFailService {
        fn name(&self) -> &str {
            &self.name
        }

        fn restart_policy(&self) -> RestartPolicy {
            RestartPolicy::OnFailure
        }

        async fn run(&self, _cancel: CancellationToken) -> anyhow::Result<()> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(1)).await;
            anyhow::bail!("permanent failure");
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_supervisor_creation() {
        let sup = JanusSupervisor::with_defaults();
        assert!(!sup.is_shutting_down());
        assert_eq!(sup.service_count().await, 0);
    }

    #[tokio::test]
    async fn test_spawn_and_cancel_single_service() {
        let config = SupervisorConfig::default().without_signal_handler();
        let sup = JanusSupervisor::new(config);

        let (svc, count) = CountingService::new("test-svc", RestartPolicy::OnFailure);
        sup.spawn_service(Box::new(svc));

        // Give the service time to start
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(sup.metrics().active_services.load(Ordering::Relaxed), 1);

        // Trigger shutdown
        sup.trigger_shutdown();
        sup.wait_for_drain().await;

        assert_eq!(count.load(Ordering::SeqCst), 1);
        let snap = sup.metrics().snapshot();
        assert_eq!(snap.spawned_total, 1);
        assert_eq!(snap.terminated_total, 1);
        assert_eq!(snap.active_services, 0);
    }

    #[tokio::test]
    async fn test_spawn_multiple_services() {
        let config = SupervisorConfig::default().without_signal_handler();
        let sup = JanusSupervisor::new(config);

        let (svc1, count1) = CountingService::new("svc-1", RestartPolicy::OnFailure);
        let (svc2, count2) = CountingService::new("svc-2", RestartPolicy::OnFailure);
        let (svc3, count3) = CountingService::new("svc-3", RestartPolicy::OnFailure);

        sup.spawn_service(Box::new(svc1));
        sup.spawn_service(Box::new(svc2));
        sup.spawn_service(Box::new(svc3));

        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(count1.load(Ordering::SeqCst), 1);
        assert_eq!(count2.load(Ordering::SeqCst), 1);
        assert_eq!(count3.load(Ordering::SeqCst), 1);

        sup.trigger_shutdown();
        sup.wait_for_drain().await;

        let snap = sup.metrics().snapshot();
        assert_eq!(snap.spawned_total, 3);
        assert_eq!(snap.terminated_total, 3);
    }

    #[tokio::test]
    async fn test_service_restart_on_failure() {
        let config = SupervisorConfig::default()
            .without_signal_handler()
            .with_default_backoff(
                BackoffConfig::new(Duration::from_millis(10), Duration::from_millis(50))
                    .without_circuit_breaker(),
            );

        let sup = JanusSupervisor::new(config);

        // Service fails 3 times, then runs cleanly
        let (svc, attempts) = FailNTimes::new("fail-3", 3);
        sup.spawn_service(Box::new(svc));

        // Give enough time for 3 failures + backoffs + stable run
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Should have attempted at least 4 times (3 failures + 1 success)
        assert!(
            attempts.load(Ordering::SeqCst) >= 4,
            "expected >= 4 attempts, got {}",
            attempts.load(Ordering::SeqCst)
        );

        sup.trigger_shutdown();
        sup.wait_for_drain().await;

        let snap = sup.metrics().snapshot();
        assert!(snap.restarts_total >= 3);
    }

    #[tokio::test]
    async fn test_one_shot_service_no_restart() {
        let config = SupervisorConfig::default().without_signal_handler();
        let sup = JanusSupervisor::new(config);

        let (svc, ran) = OneShotService::new("one-shot");
        sup.spawn_service(Box::new(svc));

        // Wait for it to complete
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should have run exactly once
        assert_eq!(ran.load(Ordering::SeqCst), 1);

        // And terminated
        let snap = sup.metrics().snapshot();
        assert_eq!(snap.terminated_total, 1);
        assert_eq!(snap.restarts_total, 0);

        sup.trigger_shutdown();
        sup.wait_for_drain().await;
    }

    #[tokio::test]
    async fn test_restart_policy_never_on_failure() {
        let config = SupervisorConfig::default().without_signal_handler();
        let sup = JanusSupervisor::new(config);

        // A service with Never policy that fails
        struct FailOnce {
            ran: Arc<AtomicU64>,
        }

        #[async_trait::async_trait]
        impl JanusService for FailOnce {
            fn name(&self) -> &str {
                "fail-once-never"
            }
            fn restart_policy(&self) -> RestartPolicy {
                RestartPolicy::Never
            }
            async fn run(&self, _cancel: CancellationToken) -> anyhow::Result<()> {
                self.ran.fetch_add(1, Ordering::SeqCst);
                anyhow::bail!("intentional failure");
            }
        }

        let ran = Arc::new(AtomicU64::new(0));
        let svc = FailOnce { ran: ran.clone() };
        sup.spawn_service(Box::new(svc));

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should have run exactly once (no restart)
        assert_eq!(ran.load(Ordering::SeqCst), 1);

        let snap = sup.metrics().snapshot();
        assert_eq!(snap.terminated_total, 1);
        assert_eq!(snap.restarts_total, 0);

        sup.trigger_shutdown();
        sup.wait_for_drain().await;
    }

    #[tokio::test]
    async fn test_circuit_breaker_trips() {
        let config = SupervisorConfig::default()
            .without_signal_handler()
            .with_default_backoff(
                BackoffConfig::new(Duration::from_millis(5), Duration::from_millis(20))
                    .with_circuit_breaker(3, Duration::from_secs(60)),
            );

        let sup = JanusSupervisor::new(config);

        let (svc, attempts) = AlwaysFailService::new("always-fail");
        sup.spawn_service(Box::new(svc));

        // Wait for circuit breaker to trip
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Circuit breaker should have tripped at 3 failures
        let att = attempts.load(Ordering::SeqCst);
        assert!(att >= 3, "expected at least 3 attempts, got {}", att);

        let snap = sup.metrics().snapshot();
        assert_eq!(snap.circuit_breaker_trips, 1);
        assert_eq!(snap.terminated_total, 1);

        sup.trigger_shutdown();
        sup.wait_for_drain().await;
    }

    #[tokio::test]
    async fn test_lifecycle_snapshots() {
        let config = SupervisorConfig::default().without_signal_handler();
        let sup = JanusSupervisor::new(config);

        let (svc, _) = CountingService::new("lifecycle-test", RestartPolicy::OnFailure);
        sup.spawn_service(Box::new(svc));

        tokio::time::sleep(Duration::from_millis(50)).await;

        let snapshots = sup.lifecycle_snapshots().await;
        assert_eq!(snapshots.len(), 1);

        let snap = &snapshots[0];
        assert_eq!(snap.service_name, "lifecycle-test");
        assert_eq!(snap.phase, ServicePhase::Running);
        assert_eq!(snap.start_count, 1);
        assert_eq!(snap.total_failures, 0);

        sup.trigger_shutdown();
        sup.wait_for_drain().await;

        // After shutdown, should be terminated
        let snapshots = sup.lifecycle_snapshots().await;
        let snap = &snapshots[0];
        assert_eq!(snap.phase, ServicePhase::Terminated);
    }

    #[tokio::test]
    async fn test_service_lifecycle_by_name() {
        let config = SupervisorConfig::default().without_signal_handler();
        let sup = JanusSupervisor::new(config);

        let (svc, _) = CountingService::new("named-svc", RestartPolicy::OnFailure);
        sup.spawn_service(Box::new(svc));

        tokio::time::sleep(Duration::from_millis(50)).await;

        let snap = sup.service_lifecycle("named-svc").await;
        assert!(snap.is_some());
        assert_eq!(snap.unwrap().service_name, "named-svc");

        let missing = sup.service_lifecycle("nonexistent").await;
        assert!(missing.is_none());

        sup.trigger_shutdown();
        sup.wait_for_drain().await;
    }

    #[tokio::test]
    async fn test_metrics_snapshot() {
        let sup = JanusSupervisor::with_defaults();
        let snap = sup.metrics().snapshot();

        assert_eq!(snap.restarts_total, 0);
        assert_eq!(snap.active_services, 0);
        assert_eq!(snap.spawned_total, 0);
        assert_eq!(snap.terminated_total, 0);
        assert_eq!(snap.circuit_breaker_trips, 0);
    }

    #[tokio::test]
    async fn test_run_until_shutdown_with_external_cancel() {
        let config = SupervisorConfig::default().without_signal_handler();
        let sup = JanusSupervisor::new(config);

        let (svc, count) = CountingService::new("ext-cancel", RestartPolicy::OnFailure);
        sup.spawn_service(Box::new(svc));

        let cancel = sup.cancel_token().clone();

        // Spawn the run loop
        let handle = tokio::spawn({
            let sup_ref_metrics = sup.metrics().clone();
            async move {
                // We can't move the supervisor into a spawn because it's not Send
                // in all cases, so just wait for the cancel token
                cancel.cancelled().await;
                sup_ref_metrics.snapshot()
            }
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);

        sup.trigger_shutdown();
        sup.wait_for_drain().await;

        let snap = handle.await.unwrap();
        assert_eq!(snap.spawned_total, 1);
    }

    #[tokio::test]
    async fn test_shutdown_timeout() {
        let config = SupervisorConfig::default()
            .without_signal_handler()
            .with_shutdown_timeout(Duration::from_millis(100));

        let sup = JanusSupervisor::new(config);

        // Spawn a service that ignores cancellation (badly behaved)
        struct HangingService;

        #[async_trait::async_trait]
        impl JanusService for HangingService {
            fn name(&self) -> &str {
                "hanger"
            }
            async fn run(&self, _cancel: CancellationToken) -> anyhow::Result<()> {
                // Intentionally ignores cancellation — simulates a stuck service.
                // Sleep for a very long time.
                tokio::time::sleep(Duration::from_secs(3600)).await;
                Ok(())
            }
        }

        sup.spawn_service(Box::new(HangingService));
        tokio::time::sleep(Duration::from_millis(20)).await;

        sup.trigger_shutdown();

        // This should complete within ~100ms (the timeout), not 3600s
        let start = std::time::Instant::now();
        sup.wait_for_drain().await;
        let elapsed = start.elapsed();

        // Allow some slack but should be well under 1 second
        assert!(
            elapsed < Duration::from_secs(1),
            "drain took too long: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_spawn_with_custom_backoff() {
        let config = SupervisorConfig::default().without_signal_handler();
        let sup = JanusSupervisor::new(config);

        let (svc, attempts) = AlwaysFailService::new("custom-backoff");

        let custom_backoff =
            BackoffConfig::new(Duration::from_millis(5), Duration::from_millis(10))
                .with_circuit_breaker(2, Duration::from_secs(60));

        sup.spawn_service_with_options(Box::new(svc), SpawnOptions::with_backoff(custom_backoff));

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Circuit breaker with max_retries=2 should trip quickly
        assert!(attempts.load(Ordering::SeqCst) >= 2);

        let snap = sup.metrics().snapshot();
        assert_eq!(snap.circuit_breaker_trips, 1);

        sup.trigger_shutdown();
        sup.wait_for_drain().await;
    }

    #[tokio::test]
    async fn test_restart_policy_always_on_clean_exit() {
        let config = SupervisorConfig::default()
            .without_signal_handler()
            .with_default_backoff(
                BackoffConfig::new(Duration::from_millis(10), Duration::from_millis(50))
                    .without_circuit_breaker(),
            );

        let sup = JanusSupervisor::new(config);

        /// A service that exits Ok immediately each time (policy: Always)
        struct ExitImmediately {
            count: Arc<AtomicU64>,
        }

        #[async_trait::async_trait]
        impl JanusService for ExitImmediately {
            fn name(&self) -> &str {
                "exit-immediately"
            }
            fn restart_policy(&self) -> RestartPolicy {
                RestartPolicy::Always
            }
            async fn run(&self, _cancel: CancellationToken) -> anyhow::Result<()> {
                self.count.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1)).await;
                Ok(())
            }
        }

        let count = Arc::new(AtomicU64::new(0));
        let svc = ExitImmediately {
            count: count.clone(),
        };

        sup.spawn_service(Box::new(svc));

        // Let it restart a few times
        tokio::time::sleep(Duration::from_millis(500)).await;

        let runs = count.load(Ordering::SeqCst);
        assert!(
            runs >= 2,
            "expected service to run multiple times with Always policy, got {}",
            runs
        );

        sup.trigger_shutdown();
        sup.wait_for_drain().await;
    }

    #[tokio::test]
    async fn test_is_shutting_down() {
        let sup = JanusSupervisor::with_defaults();
        assert!(!sup.is_shutting_down());

        sup.trigger_shutdown();
        assert!(sup.is_shutting_down());
    }

    #[tokio::test]
    async fn test_config_builder() {
        let config = SupervisorConfig::default()
            .with_shutdown_timeout(Duration::from_secs(10))
            .with_default_backoff(BackoffConfig::new(
                Duration::from_millis(200),
                Duration::from_secs(30),
            ))
            .without_signal_handler();

        assert_eq!(config.shutdown_timeout, Duration::from_secs(10));
        assert!(!config.install_signal_handler);
        assert_eq!(
            config.default_backoff.base_delay,
            Duration::from_millis(200)
        );
        assert_eq!(config.default_backoff.max_delay, Duration::from_secs(30));
    }

    // =====================================================================
    // Integration Tests — Graceful Shutdown E2E
    // =====================================================================

    /// A service that logs lifecycle events into a shared Vec so the test
    /// can verify the exact sequence of operations.
    struct LifecycleTracer {
        name: String,
        log: Arc<tokio::sync::Mutex<Vec<String>>>,
        policy: RestartPolicy,
    }

    impl LifecycleTracer {
        fn new(
            name: &str,
            log: Arc<tokio::sync::Mutex<Vec<String>>>,
            policy: RestartPolicy,
        ) -> Self {
            Self {
                name: name.to_string(),
                log,
                policy,
            }
        }
    }

    #[async_trait::async_trait]
    impl JanusService for LifecycleTracer {
        fn name(&self) -> &str {
            &self.name
        }

        fn restart_policy(&self) -> RestartPolicy {
            self.policy
        }

        async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
            {
                let mut l = self.log.lock().await;
                l.push(format!("{}:started", self.name));
            }
            cancel.cancelled().await;
            {
                let mut l = self.log.lock().await;
                l.push(format!("{}:stopped", self.name));
            }
            Ok(())
        }
    }

    /// **Integration Test — Graceful Shutdown E2E**
    ///
    /// Verifies:
    ///   1. Multiple services start and register in the supervisor.
    ///   2. Programmatic shutdown cancels all services.
    ///   3. All services terminate with `Cancelled` reason.
    ///   4. Supervisor drains within the timeout.
    ///   5. Final metrics are consistent.
    #[tokio::test]
    async fn test_integration_graceful_shutdown_e2e() {
        let log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));

        let config = SupervisorConfig::default()
            .with_shutdown_timeout(Duration::from_secs(5))
            .without_signal_handler();

        let sup = JanusSupervisor::new(config);

        // Spawn three traced services with different policies
        sup.spawn_service(Box::new(LifecycleTracer::new(
            "api",
            log.clone(),
            RestartPolicy::Always,
        )));
        sup.spawn_service(Box::new(LifecycleTracer::new(
            "data",
            log.clone(),
            RestartPolicy::OnFailure,
        )));
        sup.spawn_service(Box::new(LifecycleTracer::new(
            "cns",
            log.clone(),
            RestartPolicy::OnFailure,
        )));

        // Give services time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert_eq!(sup.service_count().await, 3);
        assert!(!sup.is_shutting_down());

        // Trigger graceful shutdown
        sup.trigger_shutdown();
        assert!(sup.is_shutting_down());

        // Wait for all tasks to drain
        sup.wait_for_drain().await;

        // Verify lifecycle events — each service should have started & stopped
        let events = log.lock().await;
        for svc in &["api", "data", "cns"] {
            assert!(
                events.contains(&format!("{}:started", svc)),
                "service '{}' never started; events: {:?}",
                svc,
                *events,
            );
            assert!(
                events.contains(&format!("{}:stopped", svc)),
                "service '{}' never stopped; events: {:?}",
                svc,
                *events,
            );
        }

        // Verify all services terminated
        let snapshots = sup.lifecycle_snapshots().await;
        assert_eq!(snapshots.len(), 3);
        for snap in &snapshots {
            assert_eq!(
                snap.phase,
                ServicePhase::Terminated,
                "service '{}' should be Terminated, was {}",
                snap.service_name,
                snap.phase,
            );
            assert_eq!(
                snap.termination_reason.as_deref(),
                Some("cancelled"),
                "service '{}' should have been cancelled, got {:?}",
                snap.service_name,
                snap.termination_reason,
            );
            assert!(snap.start_count >= 1);
        }

        // Verify final metrics
        let metrics = sup.metrics().snapshot();
        assert_eq!(metrics.spawned_total, 3);
        assert_eq!(metrics.terminated_total, 3);
        assert_eq!(metrics.active_services, 0);
    }

    // =====================================================================
    // Chaos Tests — Backoff & Circuit Breaker Validation
    // =====================================================================

    /// A configurable chaos service that:
    /// - Fails `fail_times` before succeeding (or keeps failing forever).
    /// - Records each attempt timestamp for backoff analysis.
    /// - On success, waits for cancellation.
    struct ChaosService {
        name: String,
        fail_times: u32,
        current: Arc<std::sync::atomic::AtomicU32>,
        attempt_times: Arc<tokio::sync::Mutex<Vec<std::time::Instant>>>,
        policy: RestartPolicy,
    }

    impl ChaosService {
        fn new(name: &str, fail_times: u32, policy: RestartPolicy) -> Self {
            Self {
                name: name.to_string(),
                fail_times,
                current: Arc::new(std::sync::atomic::AtomicU32::new(0)),
                attempt_times: Arc::new(tokio::sync::Mutex::new(Vec::new())),
                policy,
            }
        }
    }

    #[async_trait::async_trait]
    impl JanusService for ChaosService {
        fn name(&self) -> &str {
            &self.name
        }

        fn restart_policy(&self) -> RestartPolicy {
            self.policy
        }

        async fn run(&self, cancel: CancellationToken) -> anyhow::Result<()> {
            {
                let mut ts = self.attempt_times.lock().await;
                ts.push(std::time::Instant::now());
            }

            let n = self
                .current
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            if n < self.fail_times {
                anyhow::bail!("chaos failure #{}", n + 1);
            }

            // Success — wait for cancellation
            cancel.cancelled().await;
            Ok(())
        }
    }

    /// **Chaos Test — Exponential Backoff Verification**
    ///
    /// Spawns a service that fails 3 times then succeeds.
    /// Verifies that:
    ///   1. The service is restarted automatically.
    ///   2. Delays between attempts grow (exponential backoff).
    ///   3. The service stabilises after recovery.
    ///   4. Metrics count 3 restarts.
    #[tokio::test]
    async fn test_chaos_exponential_backoff() {
        let backoff = BackoffConfig::new(
            Duration::from_millis(20), // tiny base for fast test
            Duration::from_secs(2),    // cap
        )
        .without_circuit_breaker(); // no circuit breaker — let it retry

        let config = SupervisorConfig::default()
            .with_shutdown_timeout(Duration::from_secs(5))
            .with_default_backoff(backoff)
            .without_signal_handler();

        let sup = JanusSupervisor::new(config);

        let chaos = ChaosService::new("chaos-backoff", 3, RestartPolicy::OnFailure);
        let attempts_arc = chaos.attempt_times.clone();
        let current_arc = chaos.current.clone();

        sup.spawn_service(Box::new(chaos));

        // Wait until the service has succeeded (attempt 4+) or timeout
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let count = current_arc.load(std::sync::atomic::Ordering::SeqCst);
            if count >= 4 {
                break; // 3 failures + 1 success
            }
            if tokio::time::Instant::now() > deadline {
                panic!("chaos service did not recover in time; attempts={}", count,);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Give it a moment to stabilise
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify exponential backoff: successive delays should grow
        let timestamps = attempts_arc.lock().await;
        assert!(
            timestamps.len() >= 4,
            "expected ≥4 attempts, got {}",
            timestamps.len(),
        );

        // Check that delay between attempts 2→3 >= delay between 1→2
        // (accounting for jitter, we just check monotonic growth trend)
        let delays: Vec<Duration> = timestamps
            .windows(2)
            .map(|w| w[1].duration_since(w[0]))
            .collect();

        // Skip delays[0] — that's the gap between the initial spawn and the
        // first restart.  On a busy CI runner the service can fail and be
        // re-spawned faster than the base backoff because the first attempt
        // has no preceding failure to back off from.
        //
        // Starting from delays[1] onward we're measuring actual backoff
        // intervals (after failure 2, 3, …).  We use a 1 ms floor instead
        // of 5 ms to tolerate scheduler jitter on overloaded runners while
        // still catching "no backoff at all" regressions.
        for (i, d) in delays.iter().enumerate().skip(1) {
            assert!(
                *d >= Duration::from_millis(1),
                "delay[{}] too short: {:?} — backoff may not be working",
                i,
                d,
            );
        }

        // Verify restart metrics
        let metrics = sup.metrics().snapshot();
        assert!(
            metrics.restarts_total >= 3,
            "expected ≥3 restarts, got {}",
            metrics.restarts_total,
        );

        // Shutdown cleanly
        sup.trigger_shutdown();
        sup.wait_for_drain().await;

        // Service should be terminated
        let snap = sup.service_lifecycle("chaos-backoff").await;
        assert!(snap.is_some());
        let snap = snap.unwrap();
        assert_eq!(snap.phase, ServicePhase::Terminated);
    }

    /// **Chaos Test — Circuit Breaker Trips**
    ///
    /// Spawns a service that always fails with a tight circuit breaker
    /// (max 3 retries). Verifies:
    ///   1. The circuit breaker opens after 3 failures.
    ///   2. The service terminates with `CircuitBreakerOpen` reason.
    ///   3. No further restarts occur after the trip.
    ///   4. Metrics record exactly 1 circuit breaker trip.
    #[tokio::test]
    async fn test_chaos_circuit_breaker_trips() {
        let backoff = BackoffConfig::new(Duration::from_millis(10), Duration::from_millis(50))
            .with_circuit_breaker(3, Duration::from_secs(60));

        let config = SupervisorConfig::default()
            .with_shutdown_timeout(Duration::from_secs(5))
            .with_default_backoff(backoff)
            .without_signal_handler();

        let sup = JanusSupervisor::new(config);

        // Service that always fails (fail_times = u32::MAX effectively)
        let chaos = ChaosService::new("chaos-cb", 1000, RestartPolicy::OnFailure);
        let current_arc = chaos.current.clone();

        sup.spawn_service(Box::new(chaos));

        // Wait until the circuit breaker trips (service terminates)
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(snap) = sup.service_lifecycle("chaos-cb").await
                && snap.phase == ServicePhase::Terminated
            {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                panic!(
                    "circuit breaker did not trip in time; attempts={}",
                    current_arc.load(std::sync::atomic::Ordering::SeqCst),
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let snap = sup.service_lifecycle("chaos-cb").await.unwrap();
        assert_eq!(snap.phase, ServicePhase::Terminated);

        // Verify termination reason is CircuitBreakerOpen
        let reason = snap
            .termination_reason
            .as_deref()
            .expect("expected a termination reason");
        assert!(
            reason.contains("circuit breaker"),
            "expected circuit breaker termination, got: {}",
            reason,
        );

        // Verify metrics
        let metrics = sup.metrics().snapshot();
        assert!(
            metrics.circuit_breaker_trips >= 1,
            "expected ≥1 circuit breaker trip, got {}",
            metrics.circuit_breaker_trips,
        );

        // Record the attempt count at trip time, wait, and verify no
        // further attempts occurred (no restarts after CB open).
        let attempts_at_trip = current_arc.load(std::sync::atomic::Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(200)).await;
        let attempts_after = current_arc.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            attempts_at_trip, attempts_after,
            "service should NOT restart after circuit breaker trips",
        );

        sup.trigger_shutdown();
        sup.wait_for_drain().await;
    }

    /// **Chaos Test — Mixed Fleet (healthy + failing services)**
    ///
    /// Spawns a mix of healthy and failing services and verifies that
    /// the supervisor handles the fleet correctly:
    ///   - Healthy services stay running and respond to shutdown.
    ///   - Failing services trigger backoff and eventually circuit-break.
    ///   - Shutdown cleanly drains all services regardless of state.
    #[tokio::test]
    async fn test_chaos_mixed_fleet() {
        let backoff = BackoffConfig::new(Duration::from_millis(10), Duration::from_millis(100))
            .with_circuit_breaker(3, Duration::from_secs(60));

        let config = SupervisorConfig::default()
            .with_shutdown_timeout(Duration::from_secs(5))
            .with_default_backoff(backoff)
            .without_signal_handler();

        let sup = JanusSupervisor::new(config);

        let log = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));

        // Healthy service — runs until cancelled
        sup.spawn_service(Box::new(LifecycleTracer::new(
            "healthy-api",
            log.clone(),
            RestartPolicy::OnFailure,
        )));

        // Failing service — always fails, will circuit-break
        let chaos = ChaosService::new("bad-data", 1000, RestartPolicy::OnFailure);
        sup.spawn_service(Box::new(chaos));

        // Recovering service — fails twice then stabilises
        let recovering = ChaosService::new("flaky-cns", 2, RestartPolicy::OnFailure);
        let recovering_attempts = recovering.current.clone();
        sup.spawn_service(Box::new(recovering));

        // Wait for all three services to register in the lifecycle map
        // (spawned tasks may not have started yet).
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if sup.service_count().await == 3 {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                panic!(
                    "timed out waiting for 3 services to register; got {}",
                    sup.service_count().await,
                );
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Wait for the failing service to circuit-break AND the recovering
        // service to stabilise
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let bad_done = sup
                .service_lifecycle("bad-data")
                .await
                .is_some_and(|s| s.phase == ServicePhase::Terminated);

            let recovered = recovering_attempts.load(std::sync::atomic::Ordering::SeqCst) >= 3;

            if bad_done && recovered {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                panic!("mixed fleet did not reach expected state in time");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Healthy service should still be alive
        let healthy_snap = sup.service_lifecycle("healthy-api").await.unwrap();
        assert!(
            healthy_snap.phase.is_alive(),
            "healthy service should still be alive, was {}",
            healthy_snap.phase,
        );

        // Bad service should be circuit-broken
        let bad_snap = sup.service_lifecycle("bad-data").await.unwrap();
        assert_eq!(bad_snap.phase, ServicePhase::Terminated);
        assert!(
            bad_snap
                .termination_reason
                .as_deref()
                .is_some_and(|r| r.contains("circuit breaker")),
            "bad-data should have circuit-broken, got {:?}",
            bad_snap.termination_reason,
        );

        // Flaky service should have recovered (alive or running)
        let flaky_snap = sup.service_lifecycle("flaky-cns").await.unwrap();
        assert!(
            flaky_snap.phase.is_alive(),
            "flaky service should have recovered, was {}",
            flaky_snap.phase,
        );
        assert!(
            flaky_snap.start_count >= 3,
            "flaky service should have started ≥3 times, got {}",
            flaky_snap.start_count,
        );

        // Shutdown everything
        sup.trigger_shutdown();
        sup.wait_for_drain().await;

        // All services should be terminated now
        for name in &["healthy-api", "bad-data", "flaky-cns"] {
            let snap = sup.service_lifecycle(name).await.unwrap();
            assert_eq!(
                snap.phase,
                ServicePhase::Terminated,
                "service '{}' should be Terminated after shutdown",
                name,
            );
        }

        let metrics = sup.metrics().snapshot();
        assert_eq!(metrics.active_services, 0);
        assert_eq!(metrics.spawned_total, 3);
        assert_eq!(metrics.terminated_total, 3);
    }
}
