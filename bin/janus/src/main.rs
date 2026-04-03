//! JANUS - Unified FKS Service (Supervisor Architecture)
//!
//! This is the main entry point for the consolidated JANUS service.
//! It replaces the old "fire and forget" `tokio::spawn` pattern with a
//! structured **Janus Supervisor** that manages service lifecycles through:
//!
//! - [`TaskTracker`] for structured concurrency without memory leaks
//! - [`CancellationToken`] for graceful shutdown signal propagation
//! - [`ModuleAdapter`] to bridge existing `start_module()` services
//! - Multi-layer tracing: operational stdout + HFT non-blocking file appender
//! - Exponential backoff with jitter for automatic restart on failure
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    JanusSupervisor                       │
//! │                                                         │
//! │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐ │
//! │  │   API   │ │ Forward │ │Backward │ │ CNS  │ Data  │ │
//! │  │ Always  │ │OnFailure│ │OnFailure│ │OnFail│OnFail │ │
//! │  └─────────┘ └─────────┘ └─────────┘ └──────────────┘ │
//! │                                                         │
//! │  TaskTracker  ◄── tracks all, reclaims on completion    │
//! │  CancellationToken ◄── Ctrl+C / SIGTERM propagation     │
//! │  BackoffState[] ◄── per-service restart with jitter     │
//! │  ServiceLifecycle[] ◄── Starting→Running→BackingOff→... │
//! └─────────────────────────────────────────────────────────┘
//! ```

use std::sync::Arc;

use janus_core::{
    Config, JanusState, init_logging,
    logging::LoggingConfig,
    supervisor::{
        ApiModuleAdapter, BackoffConfig, JanusSupervisor, ModuleAdapter, SpawnOptions,
        SupervisorConfig,
    },
};
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── 1. Environment & Logging ─────────────────────────────────────
    let _ = dotenvy::dotenv();

    // Initialize multi-layer logging:
    //   Layer 1 (Ops):  stdout, filtered by RUST_LOG, runtime-reloadable
    //   Layer 2 (HFT):  non-blocking rolling file → ./logs/hft/hft.log
    //
    // The guard MUST live until the end of main() to flush HFT buffers.
    let logging_config = LoggingConfig::default();
    let logging_guard = init_logging(logging_config)?;

    info!("Starting JANUS v1.0.0");
    info!("═══════════════════════════════════════════════════════════");

    // ── 2. Configuration ─────────────────────────────────────────────
    let config = Config::load()?;
    config.validate()?;

    info!("Configuration loaded:");
    info!("  Environment: {}", config.service.environment);
    info!("  HTTP Port: {}", config.ports.http);
    info!("  Metrics Port: {}", config.ports.metrics);
    info!("  Modules Enabled:");
    info!("    - Forward:  {}", config.modules.forward);
    info!("    - Backward: {}", config.modules.backward);
    info!("    - CNS:      {}", config.modules.cns);
    info!("    - API:      {}", config.modules.api);
    info!("    - Data:     {}", config.modules.data);

    // Auto-start flag controls whether processing modules begin work
    // immediately or wait for an explicit API command.
    //
    // This is INDEPENDENT of the supervisor's service spawning:
    //   - The supervisor always spawns and manages all enabled modules
    //     (lifecycle, restart, backoff, circuit breaker).
    //   - `start_services()` flips the shared `ServiceState` from
    //     `Standby` → `Running`, which unblocks modules that call
    //     `state.wait_for_services_start()` inside their `start_module()`.
    //
    // In other words, the supervisor keeps modules alive; this flag
    // controls whether they actively process data or idle in standby.
    let auto_start = std::env::var("JANUS_AUTO_START")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);

    // ── 3. Shared State ──────────────────────────────────────────────
    let state = Arc::new(JanusState::new(config.clone()).await?);
    info!("Shared state initialized (service_state: standby)");

    // ── 3a. Probe Redis ──────────────────────────────────────────────
    // Set the janus_redis_connected metric immediately so Prometheus
    // alert rules (RedisDisconnected) see the real state before the
    // `for: 3m` window expires.  This is non-fatal — if Redis is
    // temporarily down, Janus continues and modules retry later.
    state.probe_redis().await;

    // Install runtime log-level controller so `POST /api/log-level` works.
    if let Some(ctrl) = logging_guard.create_controller() {
        state.set_log_level_controller(ctrl).await;
        info!("Runtime log-level controller installed into JanusState");
    } else {
        warn!("No reload handle available — runtime log-level changes will be disabled");
    }

    // ── 4. Build Supervisor ──────────────────────────────────────────
    //
    // The supervisor replaces the old Vec<JoinHandle> + manual shutdown
    // loop with structured concurrency:
    //
    //  - TaskTracker: tracks all spawned tasks, reclaims memory on
    //    completion (no result accumulation like JoinSet)
    //  - CancellationToken: propagates shutdown to all child tokens
    //  - BackoffConfig: exponential backoff with jitter per service
    //  - Circuit breaker: stops restarting after N failures in window T

    let supervisor_config = SupervisorConfig::default()
        .with_shutdown_timeout(std::time::Duration::from_secs(30))
        .with_default_backoff(
            BackoffConfig::new(
                std::time::Duration::from_millis(100), // base delay
                std::time::Duration::from_secs(60),    // max delay
            )
            .with_cooldown(std::time::Duration::from_secs(300)) // reset after 5 min stable
            .with_circuit_breaker(10, std::time::Duration::from_secs(600)), // trip after 10 failures in 10 min
        );

    let supervisor = JanusSupervisor::new(supervisor_config);

    // ── 5. Spawn Services via ModuleAdapter ──────────────────────────
    //
    // Each existing service's `start_module(Arc<JanusState>)` function
    // is wrapped in a ModuleAdapter that bridges:
    //   CancellationToken → state.request_shutdown()
    //
    // The supervisor handles:
    //   - Lifecycle tracking (Starting → Running → BackingOff → ...)
    //   - Automatic restart on failure with exponential backoff
    //   - Circuit breaker for persistent failures
    //   - Graceful shutdown propagation

    // API module: always-on, always restarts (even on clean exit)
    if config.modules.api {
        info!("Spawning API module (always-on, policy: always)...");
        let api_adapter =
            ApiModuleAdapter::new(state.clone(), |s| Box::pin(janus_api::start_module(s)));
        supervisor.spawn_service(Box::new(api_adapter));
    }

    // Forward module: restarts on failure only
    if config.modules.forward {
        info!("Spawning Forward module (policy: on_failure)...");
        let adapter = ModuleAdapter::on_failure("forward", state.clone(), |s| {
            Box::pin(janus_forward::start_module(s))
        });
        supervisor.spawn_service(Box::new(adapter));
    }

    // Backward module: restarts on failure only
    if config.modules.backward {
        info!("Spawning Backward module (policy: on_failure)...");
        let adapter = ModuleAdapter::on_failure("backward", state.clone(), |s| {
            Box::pin(janus_backward::start_module(s))
        });
        supervisor.spawn_service(Box::new(adapter));
    }

    // CNS module: restarts on failure only
    if config.modules.cns {
        info!("Spawning CNS module (policy: on_failure)...");
        let adapter = ModuleAdapter::on_failure("cns", state.clone(), |s| {
            Box::pin(janus_cns_service::start_module(s))
        });
        supervisor.spawn_service(Box::new(adapter));
    }

    // Data module: restarts on failure, with tighter circuit breaker
    // (data integrity is critical — stop faster on persistent failures)
    if config.modules.data {
        info!("Spawning Data module (policy: on_failure, tight circuit breaker)...");
        let adapter = ModuleAdapter::on_failure("data", state.clone(), |s| {
            Box::pin(janus_data::start_module(s))
        });
        let data_backoff = BackoffConfig::new(
            std::time::Duration::from_millis(200),
            std::time::Duration::from_secs(30),
        )
        .with_circuit_breaker(5, std::time::Duration::from_secs(300));

        supervisor.spawn_service_with_options(
            Box::new(adapter),
            SpawnOptions::with_backoff(data_backoff),
        );
    }

    // ── 6. Service State Management ──────────────────────────────────

    info!("═══════════════════════════════════════════════════════════");
    info!(
        "Supervisor active: {} services spawned",
        supervisor.service_count().await
    );
    info!("API is live on port {}", config.ports.http);

    if auto_start {
        info!("JANUS_AUTO_START=true — starting processing services immediately");
        state.start_services();
    } else {
        info!("JANUS is in STANDBY mode");
        info!("  Processing modules are waiting for a start command.");
        info!("  Use one of:");
        info!(
            "    POST http://localhost:{}/api/services/start",
            config.ports.http
        );
        info!("    or the web interface to begin processing.");
    }

    info!("Press Ctrl+C to stop.");
    info!("═══════════════════════════════════════════════════════════");

    // ── 7. Run Until Shutdown ────────────────────────────────────────
    //
    // This blocks until Ctrl+C / SIGTERM is received, then:
    //   1. Cancels the root CancellationToken (propagates to all services)
    //   2. Closes the TaskTracker (prevents new task spawning)
    //   3. Waits for all tasks to drain (with configurable timeout)
    //   4. Logs final supervisor metrics

    supervisor.run_until_shutdown().await?;

    // ── 8. Post-Shutdown Cleanup ─────────────────────────────────────

    // Log final supervisor metrics
    let metrics = supervisor.metrics().snapshot();
    info!("═══════════════════════════════════════════════════════════");
    info!("Supervisor Metrics (final):");
    info!("  Services spawned:      {}", metrics.spawned_total);
    info!("  Services terminated:   {}", metrics.terminated_total);
    info!("  Total restarts:        {}", metrics.restarts_total);
    info!("  Circuit breaker trips: {}", metrics.circuit_breaker_trips);

    // Log final lifecycle snapshots for each service
    let snapshots = supervisor.lifecycle_snapshots().await;
    if !snapshots.is_empty() {
        info!("Service Lifecycle Summary:");
        for snap in &snapshots {
            info!(
                "  {} [{}] starts={} failures={} running={:.1}s",
                snap.service_name,
                snap.phase,
                snap.start_count,
                snap.total_failures,
                snap.cumulative_running_secs,
            );
            if let Some(ref reason) = snap.termination_reason {
                info!("    termination: {}", reason);
            }
            if let Some(ref err) = snap.last_error {
                warn!("    last error: {}", err);
            }
        }
    }

    // Perform final state cleanup (close Redis, flush WAL, etc.)
    state.shutdown().await?;

    info!("═══════════════════════════════════════════════════════════");
    info!("JANUS shutdown complete — all services drained cleanly");
    info!("═══════════════════════════════════════════════════════════");

    // logging_guard drops here — HFT non-blocking buffer is flushed
    drop(logging_guard);

    Ok(())
}
