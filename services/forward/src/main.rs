//! # JANUS Forward Service - Main Entry Point
//!
//! Real-time signal generation and streaming service.
//! Handles forward-looking operations including:
//! - Real-time signal generation
//! - WebSocket streaming
//! - Risk management (real-time)
//! - ML inference
//! - Market data processing
//! - **Parameter hot-reload from optimizer**
//! - **Brain-inspired trading pipeline (regime → hypothalamus → amygdala → gating → correlation)**
//!
//! ## Boot Sequence
//!
//! 1. Load configuration (service config, param-reload config, brain runtime config)
//! 2. Boot `BrainRuntime` (pre-flight checks, pipeline init, optional watchdog)
//! 3. Start `ForwardService` (REST, gRPC, WebSocket, metrics)
//! 4. Optionally start parameter hot-reload
//! 5. Run heartbeat loop for watchdog
//! 6. Wait for shutdown signal

use anyhow::Result;
use janus_forward::brain_runtime::{PreflightRuntimeConfig, RuntimeState, WatchdogRuntimeConfig};
use janus_forward::{
    AffinityRedisConfig, AffinityRedisStore, BrainGatedConfig, BrainGatedExecutionClient,
    BrainHealthReport, BrainHealthState, BrainRuntime, BrainRuntimeConfig, ExecutionClient,
    ExecutionClientConfig, FeedbackGrpcServer, ForwardService, ForwardServiceConfig,
    ParamReloadConfig, TradingPipelineConfig, load_pipeline_affinity, save_pipeline_affinity,
    spawn_affinity_autosave,
};
use std::sync::Arc;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,janus_forward=debug".to_string()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║        JANUS FORWARD SERVICE - Real-time Signals         ║");
    info!("║   Signal Generation • WebSocket Streaming • Risk Mgmt    ║");
    info!("║       Parameter Hot-Reload • Brain-Inspired Pipeline     ║");
    info!("╚═══════════════════════════════════════════════════════════╝");

    // ── Step 1: Load configuration ─────────────────────────────────
    let config = load_config()?;
    let param_reload_config = load_param_reload_config();
    let brain_config = load_brain_runtime_config();

    // ── Step 2: Check for preflight-only mode ──────────────────────
    let preflight_only = std::env::var("BRAIN_PREFLIGHT_ONLY")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);

    if preflight_only {
        info!("Running in preflight-only mode (dry run)...");
        let mut runtime = BrainRuntime::new(brain_config);
        let report = runtime.preflight_only().await;
        info!("\n{}", report.full_report());
        if report.is_boot_safe() {
            info!("✅ All pre-flight checks passed");
            return Ok(());
        } else {
            error!("❌ Pre-flight checks failed: {}", report.summary());
            std::process::exit(1);
        }
    }

    // ── Step 3: Boot BrainRuntime ──────────────────────────────────
    let enable_brain = std::env::var("ENABLE_BRAIN_RUNTIME")
        .unwrap_or_else(|_| "true".to_string())
        .parse::<bool>()
        .unwrap_or(true);

    let mut brain_runtime = if enable_brain {
        info!("Booting Brain Runtime (brain-inspired trading pipeline)...");
        let mut runtime = BrainRuntime::new(brain_config);

        match runtime.boot().await {
            Ok(report) => {
                info!("✅ Brain Runtime booted successfully");
                info!("   Boot report: {}", report.summary());
            }
            Err(e) => {
                error!("❌ Brain Runtime boot failed: {}", e);
                if std::env::var("BRAIN_ABORT_ON_BOOT_FAILURE")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse::<bool>()
                    .unwrap_or(true)
                {
                    return Err(e);
                }
                warn!(
                    "⚠️  Continuing despite brain runtime boot failure (BRAIN_ABORT_ON_BOOT_FAILURE=false)"
                );
            }
        }

        Some(runtime)
    } else {
        info!("Brain Runtime disabled (ENABLE_BRAIN_RUNTIME=false)");
        None
    };

    // Log brain health report
    if let Some(ref runtime) = brain_runtime {
        let health = runtime.health_report().await;
        log_brain_health(&health);
    }

    // ── Step 3b: Create BrainHealthState for REST endpoints ────────
    let brain_health_state: Option<Arc<BrainHealthState>> = if let Some(ref runtime) = brain_runtime
    {
        let pipeline = runtime.pipeline().map(Arc::clone);
        let watchdog_handle = runtime.watchdog_handle().cloned();
        let boot_passed = runtime
            .boot_report()
            .map(|r| r.is_boot_safe())
            .unwrap_or(false);
        let boot_summary = runtime.boot_report().map(|r| r.summary());
        let state = BrainHealthState::new(
            pipeline,
            watchdog_handle,
            boot_passed,
            boot_summary,
            RuntimeState::Running,
        );
        Some(Arc::new(state))
    } else {
        None
    };

    // ── Step 3c: Wire affinity persistence (load from Redis) ───────
    let affinity_store: Option<Arc<AffinityRedisStore>> = if let Some(ref runtime) = brain_runtime {
        if runtime.pipeline().is_some() {
            let redis_config = AffinityRedisConfig::from_env();
            match AffinityRedisStore::new(redis_config).await {
                Ok(store) => {
                    let store = Arc::new(store);
                    // Attempt to restore affinity state from the previous run
                    let min_trades = parse_env_u64("BRAIN_AFFINITY_MIN_TRADES", 10) as usize;
                    if let Some(pipeline) = runtime.pipeline() {
                        match load_pipeline_affinity(pipeline, &store, min_trades).await {
                            Ok(true) => {
                                info!("🧠✅ Restored strategy affinity state from Redis");
                            }
                            Ok(false) => {
                                info!(
                                    "🧠 No previous affinity state in Redis — attempting memory bootstrap"
                                );
                                // ── JFLOW-D: Bootstrap affinity from fks:memories:new ring buffer ──
                                // When Redis has no saved affinity state (fresh deployment or wiped
                                // Redis), replay the most recent janus_memories from the ring buffer
                                // so the brain starts warm rather than cold.
                                let bootstrap_days = std::env::var("JANUS_BOOTSTRAP_DAYS")
                                    .unwrap_or_else(|_| "30".to_string())
                                    .parse::<u64>()
                                    .unwrap_or(30);
                                let bootstrap_limit = std::env::var("JANUS_BOOTSTRAP_LIMIT")
                                    .unwrap_or_else(|_| "500".to_string())
                                    .parse::<isize>()
                                    .unwrap_or(500)
                                    .min(2000);

                                if bootstrap_days > 0 {
                                    bootstrap_affinity_from_redis_ring(
                                        pipeline,
                                        store.config().redis_url.clone(),
                                        bootstrap_limit,
                                    )
                                    .await;
                                }
                            }
                            Err(e) => {
                                warn!(
                                    "⚠️  Failed to load affinity state from Redis: {}. \
                                         Starting with fresh tracker.",
                                    e
                                );
                            }
                        }
                    }
                    Some(store)
                }
                Err(e) => {
                    warn!(
                        "⚠️  Failed to connect to Redis for affinity persistence: {}. \
                             Affinity state will NOT be persisted.",
                        e
                    );
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // ── Step 4: Create ForwardService ──────────────────────────────
    info!("Initializing Forward Service...");
    let mut service = ForwardService::new(config.clone()).await?;

    // Wire brain health state so REST endpoints are mounted
    if let Some(ref bhs) = brain_health_state {
        service.set_brain_health_state(Arc::clone(bhs));
        info!("🧠 Brain health state wired into ForwardService REST server");
    }

    // ── Step 4b: Wire brain-gated execution into SignalGenerator ───
    // When the brain runtime is enabled and has a pipeline, wrap the
    // execution client with the brain gate so that every signal
    // submission is evaluated through the full brain pipeline before
    // reaching the execution service.
    if let Some(ref runtime) = brain_runtime
        && let Some(pipeline) = runtime.pipeline()
    {
        // Only wire if execution is configured
        if let Some(ref exec_cfg) = config.execution_config {
            match ExecutionClient::new(exec_cfg.clone()).await {
                Ok(exec_client) => {
                    let gated_config = BrainGatedConfig {
                        allow_reduce_only: parse_env_bool("BRAIN_ALLOW_REDUCE_ONLY", true),
                        apply_scale_to_signal: parse_env_bool("BRAIN_APPLY_SCALE", true),
                        max_signal_age_secs: parse_env_u64("BRAIN_MAX_SIGNAL_AGE_SECS", 120),
                        record_trade_results: parse_env_bool("BRAIN_RECORD_TRADE_RESULTS", true),
                    };

                    let gated_client = BrainGatedExecutionClient::with_config(
                        exec_client,
                        Arc::clone(pipeline),
                        gated_config,
                    );

                    if service.set_brain_gated_client(gated_client) {
                        info!("🧠✅ Brain-gated execution wired into SignalGenerator");
                    } else {
                        warn!(
                            "⚠️  Could not wire brain-gated client into SignalGenerator \
                                 (Arc has multiple references). Brain gating will NOT be \
                                 active on the SignalGenerator path. Ensure brain wiring \
                                 happens before param reload starts."
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "⚠️  Failed to create execution client for brain gating: {}. \
                             Signals will not be brain-gated.",
                        e
                    );
                }
            }
        } else {
            info!(
                "Brain runtime active but no execution config — \
                 brain-gated execution not wired (signals won't be auto-executed)"
            );
        }
    }

    // ── Step 4c: Spawn PositionFeedback gRPC server (optional) ───────────
    // Reads JANUS_FEEDBACK_GRPC_PORT (default 50052).  The server runs
    // concurrently with everything else; a failure is logged but does NOT
    // crash the main service.
    let _feedback_grpc_handle = {
        let port: u16 = std::env::var("JANUS_FEEDBACK_GRPC_PORT")
            .unwrap_or_else(|_| "50052".to_string())
            .parse::<u16>()
            .unwrap_or(50052);
        let addr: std::net::SocketAddr = format!("0.0.0.0:{port}")
            .parse()
            .expect("valid feedback gRPC socket address");
        let svc = FeedbackGrpcServer::new(port).into_service();
        info!("🔌 PositionFeedbackService gRPC server on port {port}");
        tokio::spawn(async move {
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(svc)
                .serve(addr)
                .await
            {
                warn!("⚠️  PositionFeedbackService gRPC server exited: {e}");
            }
        })
    };

    // ── Step 5: Start param reload (optional) ──────────────────────
    let enable_param_reload = std::env::var("ENABLE_PARAM_RELOAD")
        .unwrap_or_else(|_| "true".to_string())
        .parse::<bool>()
        .unwrap_or(true);

    if enable_param_reload {
        info!("Starting Forward Service with parameter hot-reload...");

        // ── Step 5b: Spawn affinity autosave task ──────────────────
        let affinity_autosave_handle =
            if let (Some(runtime), Some(store)) = (&brain_runtime, &affinity_store) {
                if let Some(pipeline) = runtime.pipeline() {
                    let interval_secs = parse_env_u64("BRAIN_AFFINITY_AUTOSAVE_INTERVAL_SECS", 300);
                    let handle = spawn_affinity_autosave(
                        Arc::clone(pipeline),
                        Arc::clone(store),
                        std::time::Duration::from_secs(interval_secs),
                    );
                    info!("🧠 Affinity autosave started (interval={}s)", interval_secs);
                    Some(handle)
                } else {
                    None
                }
            } else {
                None
            };

        let reload_handle = match service.start_with_param_reload(param_reload_config).await {
            Ok(handle) => {
                info!("✅ Forward Service started with parameter hot-reload");
                Some(handle)
            }
            Err(e) => {
                warn!(
                    "Failed to start with param reload: {}. Starting without hot-reload.",
                    e
                );
                let service_clone = service;
                tokio::spawn(async move {
                    let mut svc = service_clone;
                    if let Err(e) = svc.start().await {
                        error!("Forward service error: {}", e);
                    }
                });
                None
            }
        };

        // ── Step 6: Start watchdog heartbeat loop ──────────────────
        let heartbeat_handle = if let Some(ref runtime) = brain_runtime {
            let heartbeat_ms = runtime.config().forward_heartbeat_ms;
            // We need to send heartbeats periodically. Since BrainRuntime
            // isn't Clone/Send-safe by itself, we use the watchdog handle
            // directly if available.
            if let Some(wh) = runtime.watchdog_handle() {
                let handle = wh.clone();
                let task = tokio::spawn(async move {
                    let mut interval =
                        tokio::time::interval(std::time::Duration::from_millis(heartbeat_ms));
                    loop {
                        interval.tick().await;
                        handle
                            .heartbeat(janus_forward::brain_runtime::components::FORWARD_SERVICE)
                            .await;
                        handle
                            .heartbeat(janus_forward::brain_runtime::components::TRADING_PIPELINE)
                            .await;
                    }
                });
                Some(task)
            } else {
                None
            }
        } else {
            None
        };

        // ── Step 7: Wait for shutdown signal ───────────────────────
        tokio::signal::ctrl_c().await?;
        info!("Shutdown signal received, stopping service...");

        // Stop affinity autosave
        if let Some(handle) = affinity_autosave_handle {
            handle.abort();
            let _ = handle.await;
        }

        // Stop heartbeat loop
        if let Some(hb) = heartbeat_handle {
            hb.abort();
            let _ = hb.await;
        }

        // Stop param reload
        if let Some(handle) = reload_handle {
            info!("Stopping parameter hot-reload...");
            let stats = handle.stats().await;
            info!(
                "Param reload stats - received: {}, applied: {}, failed: {}",
                stats.total_received, stats.successful_applies, stats.failed_applies
            );
            handle.stop().await;
        }

        // Log final brain health
        if let Some(ref runtime) = brain_runtime {
            let health = runtime.health_report().await;
            log_brain_health(&health);

            // Log final pipeline metrics
            if let Some(pipeline) = runtime.pipeline() {
                let metrics = pipeline.metrics_snapshot().await;
                info!(
                    "Pipeline final stats — evaluations: {}, proceeds: {}, blocks: {}, reduce_only: {}, avg_latency: {:.0}µs, block_rate: {:.1}%",
                    metrics.total_evaluations,
                    metrics.proceed_count,
                    metrics.block_count,
                    metrics.reduce_only_count,
                    metrics.avg_evaluation_us(),
                    metrics.block_rate_pct(),
                );
            }
        }

        // Save affinity state one final time before shutting down
        if let (Some(runtime), Some(store)) = (&brain_runtime, &affinity_store)
            && let Some(pipeline) = runtime.pipeline()
        {
            match save_pipeline_affinity(pipeline, store).await {
                Ok(()) => info!("🧠✅ Saved affinity state to Redis on shutdown"),
                Err(e) => warn!("⚠️  Failed to save affinity state on shutdown: {}", e),
            }
        }

        // Shutdown brain runtime
        if let Some(ref mut runtime) = brain_runtime {
            runtime.shutdown().await;
        }
    } else {
        info!("Starting Forward Service without parameter hot-reload...");

        let service_handle = tokio::spawn(async move {
            if let Err(e) = service.start().await {
                error!("Forward service error: {}", e);
            }
        });

        // Spawn affinity autosave task (non-param-reload path)
        let affinity_autosave_handle =
            if let (Some(runtime), Some(store)) = (&brain_runtime, &affinity_store) {
                if let Some(pipeline) = runtime.pipeline() {
                    let interval_secs = parse_env_u64("BRAIN_AFFINITY_AUTOSAVE_INTERVAL_SECS", 300);
                    let handle = spawn_affinity_autosave(
                        Arc::clone(pipeline),
                        Arc::clone(store),
                        std::time::Duration::from_secs(interval_secs),
                    );
                    info!("🧠 Affinity autosave started (interval={}s)", interval_secs);
                    Some(handle)
                } else {
                    None
                }
            } else {
                None
            };

        // Heartbeat loop (non-param-reload path)
        let heartbeat_handle = if let Some(ref runtime) = brain_runtime {
            let heartbeat_ms = runtime.config().forward_heartbeat_ms;
            if let Some(wh) = runtime.watchdog_handle() {
                let handle = wh.clone();
                let task = tokio::spawn(async move {
                    let mut interval =
                        tokio::time::interval(std::time::Duration::from_millis(heartbeat_ms));
                    loop {
                        interval.tick().await;
                        handle
                            .heartbeat(janus_forward::brain_runtime::components::FORWARD_SERVICE)
                            .await;
                        handle
                            .heartbeat(janus_forward::brain_runtime::components::TRADING_PIPELINE)
                            .await;
                    }
                });
                Some(task)
            } else {
                None
            }
        } else {
            None
        };

        // Wait for shutdown signal
        tokio::signal::ctrl_c().await?;
        info!("Shutdown signal received, stopping service...");

        // Stop affinity autosave
        if let Some(handle) = affinity_autosave_handle {
            handle.abort();
            let _ = handle.await;
        }

        if let Some(hb) = heartbeat_handle {
            hb.abort();
            let _ = hb.await;
        }

        // Log final brain health
        if let Some(ref runtime) = brain_runtime {
            let health = runtime.health_report().await;
            log_brain_health(&health);
        }

        // Save affinity state one final time before shutting down
        if let (Some(runtime), Some(store)) = (&brain_runtime, &affinity_store)
            && let Some(pipeline) = runtime.pipeline()
        {
            match save_pipeline_affinity(pipeline, store).await {
                Ok(()) => info!("🧠✅ Saved affinity state to Redis on shutdown"),
                Err(e) => warn!("⚠️  Failed to save affinity state on shutdown: {}", e),
            }
        }

        if let Some(ref mut runtime) = brain_runtime {
            runtime.shutdown().await;
        }

        service_handle.await?;
    }

    info!("✅ JANUS Forward Service stopped successfully");
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Configuration loaders
// ════════════════════════════════════════════════════════════════════

/// Load main service configuration from environment variables
fn load_config() -> Result<ForwardServiceConfig> {
    let execution_config = if std::env::var("ENABLE_EXECUTION")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false)
    {
        info!("Execution integration enabled");
        Some(ExecutionClientConfig {
            endpoint: std::env::var("EXECUTION_ENDPOINT")
                .unwrap_or_else(|_| "http://execution:50052".to_string()),
            connect_timeout_secs: std::env::var("EXECUTION_CONNECT_TIMEOUT")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .unwrap_or(10),
            request_timeout_secs: std::env::var("EXECUTION_REQUEST_TIMEOUT")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            enable_tls: std::env::var("EXECUTION_ENABLE_TLS")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
            max_retries: std::env::var("EXECUTION_MAX_RETRIES")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .unwrap_or(3),
            retry_backoff_ms: std::env::var("EXECUTION_RETRY_BACKOFF_MS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .unwrap_or(100),
            ..Default::default()
        })
    } else {
        info!("Execution integration disabled - signals will be generated but not executed");
        None
    };

    let config = ForwardServiceConfig {
        host: std::env::var("FORWARD_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
        grpc_port: std::env::var("FORWARD_GRPC_PORT")
            .unwrap_or_else(|_| "50051".to_string())
            .parse()?,
        rest_port: std::env::var("FORWARD_REST_PORT")
            .unwrap_or_else(|_| "8080".to_string())
            .parse()?,
        websocket_port: std::env::var("FORWARD_WS_PORT")
            .unwrap_or_else(|_| "8081".to_string())
            .parse()?,
        metrics_port: std::env::var("FORWARD_METRICS_PORT")
            .unwrap_or_else(|_| "9090".to_string())
            .parse()?,
        enable_metrics: std::env::var("FORWARD_ENABLE_METRICS")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true),
        execution_config,
        ..Default::default()
    };

    Ok(config)
}

/// Load parameter hot-reload configuration from environment variables
///
/// # Environment Variables
///
/// - `REDIS_URL` - Redis connection URL (default: redis://localhost:6379)
/// - `FKS_INSTANCE_ID` - Instance ID for Redis key namespacing (default: default)
/// - `ENABLE_PARAM_RELOAD` - Enable/disable param hot-reload (default: true)
/// - `PARAM_RELOAD_RECONNECT_MS` - Reconnection delay in milliseconds (default: 5000)
/// - `PARAM_RELOAD_MAX_RETRIES` - Max reconnection attempts, 0=unlimited (default: 0)
fn load_param_reload_config() -> ParamReloadConfig {
    let config = ParamReloadConfig::from_env();

    info!(
        redis_url = %config.redis_url,
        instance_id = %config.instance_id,
        enabled = config.enabled,
        "Loaded param reload configuration"
    );

    config
}

/// Load brain runtime configuration from environment variables
///
/// # Environment Variables
///
/// ## Runtime
/// - `BRAIN_ENFORCE_PREFLIGHT` - Abort boot if preflight checks fail (default: true)
/// - `BRAIN_AUTO_START_WATCHDOG` - Start watchdog automatically on boot (default: true)
/// - `BRAIN_WIRE_KILL_SWITCH` - Wire pipeline kill switch to watchdog (default: true)
/// - `BRAIN_HEARTBEAT_MS` - Heartbeat interval in milliseconds (default: 5000)
///
/// ## Pipeline
/// - `BRAIN_ENABLE_HYPOTHALAMUS` - Enable hypothalamus position scaling (default: true)
/// - `BRAIN_ENABLE_AMYGDALA` - Enable amygdala threat detection (default: true)
/// - `BRAIN_ENABLE_GATING` - Enable strategy gating (default: true)
/// - `BRAIN_ENABLE_CORRELATION` - Enable correlation filtering (default: true)
/// - `BRAIN_MAX_POSITION_SCALE` - Maximum position scale factor (default: 2.0)
/// - `BRAIN_MIN_POSITION_SCALE` - Minimum position scale factor (default: 0.1)
/// - `BRAIN_HIGH_RISK_SCALE` - Scale-down factor for high-risk regimes (default: 0.5)
/// - `BRAIN_MIN_REGIME_CONFIDENCE` - Minimum regime confidence to trade (default: 0.3)
/// - `BRAIN_ALLOW_CRISIS_POSITIONS` - Allow new positions during crisis (default: false)
///
/// ## Preflight
/// - `BRAIN_PREFLIGHT_TIMEOUT_SECS` - Global preflight timeout (default: 30)
/// - `BRAIN_SKIP_INFRUSTCODE_CHECKS` - Skip infrastructure checks (default: false)
///
/// ## Watchdog
/// - `BRAIN_WATCHDOG_CHECK_INTERVAL_MS` - Watchdog check interval (default: 5000)
/// - `BRAIN_WATCHDOG_DEGRADED_THRESHOLD` - Missed heartbeats for Degraded state (default: 3)
/// - `BRAIN_WATCHDOG_DEAD_THRESHOLD` - Missed heartbeats for Dead state (default: 5)
/// - `BRAIN_WATCHDOG_KILL_ON_CRITICAL_DEATH` - Kill trading on critical component death (default: true)
fn load_brain_runtime_config() -> BrainRuntimeConfig {
    let pipeline = TradingPipelineConfig {
        enable_hypothalamus_scaling: parse_env_bool("BRAIN_ENABLE_HYPOTHALAMUS", true),
        enable_amygdala_filter: parse_env_bool("BRAIN_ENABLE_AMYGDALA", true),
        enable_gating: parse_env_bool("BRAIN_ENABLE_GATING", true),
        enable_correlation_filter: parse_env_bool("BRAIN_ENABLE_CORRELATION", true),
        max_position_scale: parse_env_f64("BRAIN_MAX_POSITION_SCALE", 2.0),
        min_position_scale: parse_env_f64("BRAIN_MIN_POSITION_SCALE", 0.1),
        high_risk_scale_factor: parse_env_f64("BRAIN_HIGH_RISK_SCALE", 0.5),
        allow_new_positions_in_crisis: parse_env_bool("BRAIN_ALLOW_CRISIS_POSITIONS", false),
        min_regime_confidence: parse_env_f64("BRAIN_MIN_REGIME_CONFIDENCE", 0.3),
        ..Default::default()
    };

    let watchdog = WatchdogRuntimeConfig {
        check_interval_ms: parse_env_u64("BRAIN_WATCHDOG_CHECK_INTERVAL_MS", 5000),
        degraded_threshold: parse_env_u32("BRAIN_WATCHDOG_DEGRADED_THRESHOLD", 3),
        dead_threshold: parse_env_u32("BRAIN_WATCHDOG_DEAD_THRESHOLD", 5),
        kill_on_critical_death: parse_env_bool("BRAIN_WATCHDOG_KILL_ON_CRITICAL_DEATH", true),
    };

    let preflight = PreflightRuntimeConfig {
        global_timeout_secs: parse_env_u64("BRAIN_PREFLIGHT_TIMEOUT_SECS", 30),
        skip_infra_checks: parse_env_bool("BRAIN_SKIP_INFRUSTCODE_CHECKS", false),
        ..Default::default()
    };

    let config = BrainRuntimeConfig {
        pipeline,
        watchdog,
        preflight,
        enforce_preflight: parse_env_bool("BRAIN_ENFORCE_PREFLIGHT", true),
        auto_start_watchdog: parse_env_bool("BRAIN_AUTO_START_WATCHDOG", true),
        wire_kill_switch: parse_env_bool("BRAIN_WIRE_KILL_SWITCH", true),
        forward_heartbeat_ms: parse_env_u64("BRAIN_HEARTBEAT_MS", 5000),
    };

    info!(
        enforce_preflight = config.enforce_preflight,
        auto_start_watchdog = config.auto_start_watchdog,
        wire_kill_switch = config.wire_kill_switch,
        heartbeat_ms = config.forward_heartbeat_ms,
        hypothalamus = config.pipeline.enable_hypothalamus_scaling,
        amygdala = config.pipeline.enable_amygdala_filter,
        gating = config.pipeline.enable_gating,
        correlation = config.pipeline.enable_correlation_filter,
        min_confidence = config.pipeline.min_regime_confidence,
        "Loaded brain runtime configuration"
    );

    config
}

// ════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════

fn parse_env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}

fn parse_env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}

fn parse_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}

fn parse_env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}

/// Log a brain health report in a human-readable format.
fn log_brain_health(health: &BrainHealthReport) {
    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║                  Brain Health Report                     ║");
    info!("╠═══════════════════════════════════════════════════════════╣");
    info!("║  State:        {:>40}  ║", format!("{}", health.state));
    info!(
        "║  Boot passed:  {:>40}  ║",
        if health.boot_passed {
            "✅ YES"
        } else {
            "❌ NO"
        }
    );

    if let Some(ref summary) = health.boot_summary {
        info!("║  Boot summary: {:>40}  ║", summary);
    }

    if let Some(ref watchdog) = health.watchdog {
        info!(
            "║  Watchdog:     {:>40}  ║",
            format!(
                "{} total, {} alive, {} degraded, {} dead",
                watchdog.total_components,
                watchdog.alive_count,
                watchdog.degraded_count,
                watchdog.dead_count
            )
        );
    }

    if let Some(ref pipeline) = health.pipeline {
        info!(
            "║  Pipeline:     {:>40}  ║",
            format!(
                "{} evals, {:.1}% blocked, {:.0}µs avg",
                pipeline.total_evaluations, pipeline.block_rate_pct, pipeline.avg_evaluation_us
            )
        );
        info!(
            "║  Kill switch:  {:>40}  ║",
            if pipeline.is_killed {
                "🔴 ACTIVE"
            } else {
                "🟢 inactive"
            }
        );
    }

    info!(
        "║  Overall:      {:>40}  ║",
        if health.is_healthy() {
            "🟢 HEALTHY"
        } else {
            "🔴 UNHEALTHY"
        }
    );
    info!("╚═══════════════════════════════════════════════════════════╝");
}

// ════════════════════════════════════════════════════════════════════
// JFLOW-D: Memory Bootstrap
// ════════════════════════════════════════════════════════════════════

/// Bootstrap the affinity tracker from the `fks:memories:new` Redis ring buffer.
///
/// When Janus starts with no saved affinity state (fresh deployment or wiped Redis),
/// this function reads the most recent trade outcomes from the `fks:memories:new`
/// ring buffer written by the Python `MemoryRecorder`, parses each record, and
/// replays it into the `StrategyAffinityTracker` so the brain starts warm rather
/// than cold.
///
/// The ring buffer is bounded to 500 entries by the Python side (LPUSH + LTRIM).
/// Each entry is a JSON object with at minimum: `strategy`, `symbol`, `pnl`, `result`.
///
/// Controlled by env vars:
/// - `JANUS_BOOTSTRAP_DAYS`  — days of history; set to 0 to skip (default 30)
/// - `JANUS_BOOTSTRAP_LIMIT` — max ring-buffer entries to replay (default 500)
async fn bootstrap_affinity_from_redis_ring(
    pipeline: &Arc<janus_forward::TradingPipeline>,
    redis_url: String,
    limit: isize,
) {
    use redis::AsyncCommands;

    let client = match redis::Client::open(redis_url.as_str()) {
        Ok(c) => c,
        Err(e) => {
            warn!("🧠 Bootstrap: could not open Redis client: {}", e);
            return;
        }
    };

    let mut conn = match client.get_multiplexed_async_connection().await {
        Ok(c) => c,
        Err(e) => {
            warn!("🧠 Bootstrap: Redis connection failed: {}", e);
            return;
        }
    };

    let raw: Vec<Vec<u8>> = match conn.lrange("fks:memories:new", 0, limit - 1).await {
        Ok(v) => v,
        Err(e) => {
            warn!("🧠 Bootstrap: LRANGE fks:memories:new failed: {}", e);
            return;
        }
    };

    if raw.is_empty() {
        info!("🧠 Bootstrap: fks:memories:new is empty — affinity tracker starts cold");
        return;
    }

    let mut replayed: u64 = 0;
    let mut skipped: u64 = 0;

    {
        let mut gate = pipeline.strategy_gate_mut().await;
        let tracker = gate.tracker_mut();

        for bytes in &raw {
            let text = match std::str::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let mem: serde_json::Value = match serde_json::from_str(text) {
                Ok(v) => v,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };

            let strategy = mem
                .get("strategy")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            // Normalise symbol: "BTC/USD" → "BTCUSD" (Janus uses slash-free format)
            let symbol_raw = mem
                .get("symbol")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN");
            let asset = symbol_raw.replace('/', "").to_uppercase();

            let pnl = mem.get("pnl").and_then(|v| v.as_f64()).unwrap_or(0.0);

            // Determine win/loss: prefer the `result` field, fall back to sign of pnl
            let is_winner = mem
                .get("result")
                .and_then(|v| v.as_str())
                .map(|r| r == "win")
                .unwrap_or_else(|| pnl > 0.0);

            let rr_ratio = mem.get("rr_ratio").and_then(|v| v.as_f64());

            tracker.record_trade_result_with_rr(strategy, &asset, pnl, is_winner, rr_ratio);
            replayed += 1;
        }
    }

    info!(
        "🧠✅ Bootstrap: replayed {}/{} memories into affinity tracker ({} skipped)",
        replayed,
        raw.len(),
        skipped,
    );
}
