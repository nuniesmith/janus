//! # JANUS Forward Service
//!
//! Real-time signal generation and streaming service.
//! Handles forward-looking operations including:
//! - Real-time signal generation
//! - WebSocket streaming
//! - Risk management (real-time)
//! - ML inference
//! - Market data processing
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  JANUS Forward Service                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │   Signal     │  │  ML Model    │  │  Strategy    │     │
//! │  │  Generator   │  │  Inference   │  │  Engine      │     │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
//! │         │                  │                  │             │
//! │         └──────────────────┼──────────────────┘             │
//! │                            │                                │
//! │                   ┌────────▼────────┐                       │
//! │                   │ Risk Manager    │                       │
//! │                   └────────┬────────┘                       │
//! │                            │                                │
//! │         ┌──────────────────┴──────────────────┐            │
//! │         │                                      │            │
//! │    ┌────▼────┐                          ┌─────▼─────┐      │
//! │    │WebSocket│                          │   REST    │      │
//! │    │ Stream  │                          │    API    │      │
//! │    └─────────┘                          └───────────┘      │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod api;
pub mod brain_runtime;
pub mod brain_wiring;
pub mod execution;
pub mod features;
pub mod indicators;
pub mod inference;
pub mod metrics;
pub mod param_reload;
pub mod persistence;

// Re-export for convenience — the global janus-core Prometheus singleton.
use janus_core::metrics::metrics as janus_metrics;
pub mod regime;
pub mod regime_bridge;
pub mod regime_bridge_auth;
pub mod regime_bridge_proto;
pub mod regime_bridge_server;
pub mod risk;
pub mod signal;
pub mod strategies;
pub mod websocket;

// Re-exports for convenience
pub use api::brain_rest::BrainHealthState;
pub use api::feedback_grpc::{FeedbackGrpcServer, FeedbackGrpcService};
pub use api::server::RestServer;
pub use brain_runtime::{BrainHealthReport, BrainRuntime, BrainRuntimeConfig};
pub use brain_wiring::{
    PipelineStage, TradeAction, TradingDecision, TradingPipeline, TradingPipelineBuilder,
    TradingPipelineConfig,
};
pub use execution::{
    BrainGatedConfig, BrainGatedExecutionClient, ExecutionClient, ExecutionClientConfig,
    GatedExecutionStats, GatedSubmissionResult, SubmitSignalResponse,
};
pub use features::{FeatureConfig, FeatureEngineering, FeatureVector};
pub use indicators::{IndicatorAnalysis, IndicatorAnalyzer, IndicatorConfig};
pub use inference::{ModelCache, ModelInference, ModelMetrics, PredictionResult};
pub use metrics::{BrainPipelineMetricsCollector, JanusMetrics};
pub use param_reload::{
    IndicatorParamApplier, LoggingApplier, NoOpApplier, ParamApplier, ParamReloadConfig,
    ParamReloadManager, ReloadStats, RiskParamApplier, StrategyParamApplier,
};
pub use persistence::{
    AffinityRedisConfig, AffinityRedisStore, load_pipeline_affinity, save_pipeline_affinity,
    spawn_affinity_autosave,
};
// ParamReloadHandle is defined in this file and exported directly
pub use risk::{
    MarketData, PortfolioRisk, PortfolioState, Position, PositionSide, PositionSize, PositionSizer,
    RiskConfig, RiskError, RiskLimits, RiskManager, RiskMetrics, RiskValidator, SizingMethod,
    StopLossCalculator, StopLossMethod, TakeProfitCalculator,
};
pub use signal::{
    IndicatorValues, SignalBatch, SignalGenerator, SignalGeneratorConfig, SignalMetrics,
    SignalSource, SignalType, Timeframe, TradingSignal,
};
pub use strategies::{Strategy, StrategyConfig, StrategySignal};
pub use websocket::{
    ClientManager, DataServiceClient, DataServiceConfig, HeartbeatManager, SignalBroadcaster,
    WebSocketMessage, WebSocketServer,
};

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Forward service version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Forward service name
pub const SERVICE_NAME: &str = "janus-forward";

/// Service configuration
#[derive(Debug, Clone)]
pub struct ForwardServiceConfig {
    /// Service host
    pub host: String,

    /// gRPC port
    pub grpc_port: u16,

    /// REST API port
    pub rest_port: u16,

    /// WebSocket port
    pub websocket_port: u16,

    /// Signal generator configuration
    pub signal_config: SignalGeneratorConfig,

    /// Risk management configuration
    pub risk_config: RiskConfig,

    /// WebSocket configuration
    pub websocket_config: websocket::WebSocketConfig,

    /// Data service configuration
    pub data_service_config: Option<DataServiceConfig>,

    /// Execution client configuration (None = signals generated but not executed)
    pub execution_config: Option<ExecutionClientConfig>,

    /// Enable metrics endpoint
    pub enable_metrics: bool,

    /// Metrics port
    pub metrics_port: u16,
}

impl Default for ForwardServiceConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            grpc_port: 50051,
            rest_port: 8080,
            websocket_port: 8081,
            signal_config: SignalGeneratorConfig::default(),
            risk_config: RiskConfig::default(),
            websocket_config: websocket::WebSocketConfig::default(),
            data_service_config: None,
            execution_config: None,
            enable_metrics: true,
            metrics_port: 9090,
        }
    }
}

/// Forward service - Real-time signal generation and streaming
pub struct ForwardService {
    config: ForwardServiceConfig,
    signal_generator: Arc<SignalGenerator>,
    risk_manager: Arc<RwLock<RiskManager>>,
    portfolio: Arc<RwLock<PortfolioState>>,
    #[allow(dead_code)]
    websocket_server: Option<Arc<WebSocketServer>>,
    /// Optional brain health state for wiring brain REST endpoints into the
    /// combined REST server. Set via [`Self::set_brain_health_state`] before
    /// calling [`Self::start`].
    brain_health_state: Option<Arc<BrainHealthState>>,
}

impl ForwardService {
    /// Create a new forward service instance
    pub async fn new(config: ForwardServiceConfig) -> Result<Self> {
        info!("Initializing JANUS Forward Service v{}", VERSION);

        // Create signal generator with optional execution client
        let signal_generator = if config.execution_config.is_some() {
            Arc::new(
                SignalGenerator::new_with_execution(
                    config.signal_config.clone(),
                    config.execution_config.clone(),
                )
                .await?,
            )
        } else {
            Arc::new(SignalGenerator::new(config.signal_config.clone()))
        };

        let account_balance = config.risk_config.account_balance;
        let risk_manager = Arc::new(RwLock::new(RiskManager::new(config.risk_config.clone())));
        let portfolio = Arc::new(RwLock::new(PortfolioState::new(account_balance)));

        Ok(Self {
            config,
            signal_generator,
            risk_manager,
            portfolio,
            websocket_server: None,
            brain_health_state: None,
        })
    }

    /// Get signal generator
    pub fn signal_generator(&self) -> Arc<SignalGenerator> {
        Arc::clone(&self.signal_generator)
    }

    /// Set the brain health state so that brain REST endpoints (health,
    /// kill-switch, affinity) are mounted when [`Self::start`] is called.
    ///
    /// This must be called **before** `start()` or `start_with_param_reload()`.
    pub fn set_brain_health_state(&mut self, state: Arc<BrainHealthState>) {
        self.brain_health_state = Some(state);
    }

    /// Wire a brain-gated execution client into the signal generator.
    ///
    /// This must be called **before** any `Arc` clones of the signal generator
    /// are created (i.e. before `start()` or `start_with_param_reload()`),
    /// because it requires exclusive ownership of the `Arc`.
    ///
    /// Returns `true` if wiring succeeded, `false` if the `Arc` already has
    /// multiple strong references and cannot be mutated.
    pub fn set_brain_gated_client(&mut self, client: execution::BrainGatedExecutionClient) -> bool {
        if let Some(sg) = Arc::get_mut(&mut self.signal_generator) {
            sg.set_brain_gated_client(client);
            true
        } else {
            false
        }
    }

    /// Get risk manager
    pub fn risk_manager(&self) -> Arc<RwLock<RiskManager>> {
        Arc::clone(&self.risk_manager)
    }

    /// Get portfolio state
    pub fn portfolio(&self) -> Arc<RwLock<PortfolioState>> {
        Arc::clone(&self.portfolio)
    }

    /// Get service configuration
    pub fn config(&self) -> &ForwardServiceConfig {
        &self.config
    }

    /// Start the service
    #[tracing::instrument(skip(self), fields(grpc_port = self.config.grpc_port, rest_port = self.config.rest_port, ws_port = self.config.websocket_port))]
    pub async fn start(&mut self) -> Result<()> {
        info!(
            "Starting JANUS Forward Service on {}:{} (gRPC), {}:{} (REST), {}:{} (WebSocket)",
            self.config.host,
            self.config.grpc_port,
            self.config.host,
            self.config.rest_port,
            self.config.host,
            self.config.websocket_port
        );

        // Wrap signal generator in RwLock for API
        let signal_gen_locked = Arc::new(RwLock::new(SignalGenerator::new(
            self.config.signal_config.clone(),
        )));

        // Start REST API server — include brain health endpoints when available
        let rest_server = if let Some(ref brain_state) = self.brain_health_state {
            info!("🧠 Mounting brain REST endpoints on REST server");
            api::server::RestServer::with_brain_health(
                self.config.host.clone(),
                self.config.rest_port,
                signal_gen_locked,
                Arc::clone(&self.risk_manager),
                Arc::clone(&self.portfolio),
                Arc::clone(brain_state),
            )
        } else {
            api::server::RestServer::new(
                self.config.host.clone(),
                self.config.rest_port,
                signal_gen_locked,
                Arc::clone(&self.risk_manager),
                Arc::clone(&self.portfolio),
            )
        };

        // WebSocket server is initialized separately via WebSocketServer::start()
        // when configured. The REST server exposes /api/v1/metrics for Prometheus scraping.
        if self.websocket_server.is_some() {
            info!(
                "WebSocket server configured (started independently on port {})",
                self.config.websocket_port
            );
        }

        // Metrics are exposed via the REST /api/v1/metrics endpoint.
        // A dedicated metrics server is not needed — Prometheus scrapes the REST port.
        if self.config.enable_metrics {
            info!(
                "Metrics enabled — Prometheus scrape target: http://{}:{}/api/v1/metrics",
                self.config.host, self.config.rest_port
            );
        }

        // Start REST server (this will block)
        info!("✅ Starting REST API server...");
        rest_server.start().await?;

        Ok(())
    }

    /// Shutdown the service gracefully.
    ///
    /// Logs final signal metrics and brain health state, then releases
    /// internal references so background resources can be dropped.
    #[tracing::instrument(skip(self))]
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down JANUS Forward Service...");

        // Log final signal generation metrics
        let metrics = self.signal_generator.metrics();
        info!(
            "📊 Final signal metrics — generated: {}, filtered: {}, filter_rate: {:.1}%",
            metrics.total_generated(),
            metrics.total_filtered(),
            metrics.filter_rate() * 100.0,
        );

        // Log brain health state if available
        if let Some(ref bhs) = self.brain_health_state {
            let report = bhs.health_report().await;
            let status = if report.is_healthy() {
                "HEALTHY"
            } else {
                "UNHEALTHY"
            };
            info!("🧠 Brain state at shutdown: {} ({})", report.state, status);
            if let Some(ref pipeline_metrics) = report.pipeline {
                info!(
                    "🧠 Pipeline final — evals: {}, proceeds: {}, blocks: {}, reduce_only: {}, block_rate: {:.1}%",
                    pipeline_metrics.total_evaluations,
                    pipeline_metrics.proceed_count,
                    pipeline_metrics.block_count,
                    pipeline_metrics.reduce_only_count,
                    pipeline_metrics.block_rate_pct,
                );
            }
        }

        info!("✅ JANUS Forward Service shutdown complete");
        Ok(())
    }

    /// Start the service with parameter hot-reload enabled
    ///
    /// This method initializes the param reload manager, registers appliers,
    /// and starts a background task to listen for optimized parameter updates.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use janus_forward::{ForwardService, ForwardServiceConfig, ParamReloadConfig};
    ///
    /// let config = ForwardServiceConfig::default();
    /// let param_config = ParamReloadConfig::from_env();
    ///
    /// let mut service = ForwardService::new(config).await?;
    /// let reload_handle = service.start_with_param_reload(param_config).await?;
    /// ```
    pub async fn start_with_param_reload(
        &mut self,
        param_config: ParamReloadConfig,
    ) -> Result<ParamReloadHandle> {
        info!("Initializing parameter hot-reload...");

        // Create the reload manager
        let manager = Arc::new(ParamReloadManager::new(param_config.clone()));

        // Create and register appliers
        let indicator_applier = Arc::new(IndicatorParamApplier::new());
        let risk_applier = Arc::new(RiskParamApplier::new(Arc::clone(&self.risk_manager)));
        let strategy_applier = Arc::new(StrategyParamApplier::new());
        let logging_applier = Arc::new(LoggingApplier::new("forward_service"));

        manager.register_applier(indicator_applier.clone()).await;
        manager.register_applier(risk_applier.clone()).await;
        manager.register_applier(strategy_applier.clone()).await;
        manager.register_applier(logging_applier).await;

        // Load initial params from Redis
        if param_config.enabled {
            match manager.load_initial().await {
                Ok(count) => {
                    info!(count = count, "Loaded initial optimized params from Redis");
                }
                Err(e) => {
                    warn!(error = %e, "Failed to load initial params (continuing without them)");
                }
            }
        }

        // Start background reload task
        let reload_handle = manager.clone().start_background_task().await?;

        // Create the query handle (lightweight, cloneable)
        let query_handle = ParamQueryHandle {
            indicator_applier,
            risk_applier,
            strategy_applier,
        };

        // Wire the query handle to the signal generator
        // This allows the signal generator to query optimized configs per-asset
        if let Some(sg) = Arc::get_mut(&mut self.signal_generator) {
            sg.set_param_query_handle(query_handle.clone());
            info!("✅ Wired param query handle to signal generator");
        } else {
            warn!("Could not wire param handle to signal generator (Arc has multiple references)");
        }

        // Wire the query handle to the risk manager
        // This allows risk manager to query per-asset risk configs
        {
            let mut rm = self.risk_manager.write().await;
            rm.set_param_query_handle(query_handle.clone());
            info!("✅ Wired param query handle to risk manager");
        }

        // Create the full ParamReloadHandle with the task handle
        let param_reload_handle = ParamReloadHandle {
            manager,
            task_handle: reload_handle,
            query: query_handle,
        };

        info!("✅ Parameter hot-reload initialized");

        // Start the main service
        self.start().await?;

        Ok(param_reload_handle)
    }

    /// Get health status
    pub fn health_check(&self) -> HealthStatus {
        HealthStatus {
            service: SERVICE_NAME.to_string(),
            version: VERSION.to_string(),
            status: "healthy".to_string(),
            signal_metrics: SignalMetricsSnapshot {
                total_generated: self.signal_generator.metrics().total_generated(),
                total_filtered: self.signal_generator.metrics().total_filtered(),
                filter_rate: self.signal_generator.metrics().filter_rate(),
            },
        }
    }
}

/// Start the forward module as part of the unified JANUS system
///
/// This function is called by the unified JANUS binary to start the forward signal generation module.
#[tracing::instrument(name = "forward::start_module", skip(state))]
pub async fn start_module(state: Arc<janus_core::JanusState>) -> janus_core::Result<()> {
    info!("Forward module registered — waiting for start command...");

    state
        .register_module_health("forward", true, Some("standby".to_string()))
        .await;

    // ── Wait for services to be started via API / web interface ──────
    if !state.wait_for_services_start().await {
        info!("Forward module: shutdown requested before services started");
        state
            .register_module_health("forward", false, Some("shutdown_before_start".to_string()))
            .await;
        return Ok(());
    }

    info!("Starting Forward module with unified JANUS integration...");

    state
        .register_module_health("forward", true, Some("starting".to_string()))
        .await;

    // Check if execution is enabled via environment
    let execution_enabled = std::env::var("ENABLE_EXECUTION")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);

    let execution_config = if execution_enabled {
        info!("Execution integration enabled for Forward module");
        Some(ExecutionClientConfig {
            endpoint: std::env::var("EXECUTION_ENDPOINT")
                .unwrap_or_else(|_| "http://execution:50052".to_string()),
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            enable_tls: false,
            max_retries: 3,
            retry_backoff_ms: 100,
            ..Default::default()
        })
    } else {
        info!("Execution integration disabled - signals will be generated but not executed");
        None
    };

    // Create forward service configuration from JANUS config
    let forward_config = ForwardServiceConfig {
        host: "0.0.0.0".to_string(),
        grpc_port: state.config.ports.grpc,
        rest_port: state.config.ports.http + 100, // Offset to avoid conflict
        websocket_port: state.config.ports.websocket,
        signal_config: SignalGeneratorConfig::default(),
        risk_config: RiskConfig {
            account_balance: state.config.risk.account_balance,
            max_position_size_pct: state.config.risk.max_position_size_pct,
            ..Default::default()
        },
        websocket_config: websocket::WebSocketConfig::default(),
        data_service_config: None,
        execution_config,
        enable_metrics: false, // Metrics handled by janus-api
        metrics_port: 0,
    };

    // Create forward service
    let mut service = ForwardService::new(forward_config.clone())
        .await
        .map_err(|e| janus_core::Error::module("forward", e.to_string()))?;

    // ── Brain runtime boot & wiring (unified binary path) ──────────
    let enable_brain = std::env::var("ENABLE_BRAIN_RUNTIME")
        .unwrap_or_else(|_| "true".to_string())
        .parse::<bool>()
        .unwrap_or(true);

    let mut brain_runtime: Option<BrainRuntime> = None;
    let mut affinity_store: Option<Arc<persistence::AffinityRedisStore>> = None;
    let mut affinity_autosave_handle: Option<tokio::task::JoinHandle<()>> = None;
    // Saved clone of the brain health state for spawning the brain REST server
    // after the signal generator is extracted. The forward module's full
    // RestServer is never started in unified binary mode (port conflict with
    // janus_api on 8080), so we spawn a minimal brain-only listener instead.
    let mut brain_rest_state: Option<Arc<BrainHealthState>> = None;

    if enable_brain {
        info!("Booting Brain Runtime for unified binary...");

        // Load BrainRuntimeConfig from environment variables (same pattern as
        // the standalone forward binary) so that paper-trading deployments can
        // tune watchdog / kill-switch behaviour without a rebuild.
        let brain_config = {
            use brain_runtime::PreflightRuntimeConfig;
            use brain_runtime::WatchdogRuntimeConfig;

            fn parse_env_bool(key: &str, default: bool) -> bool {
                std::env::var(key)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(default)
            }
            fn parse_env_u64(key: &str, default: u64) -> u64 {
                std::env::var(key)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(default)
            }
            fn parse_env_u32(key: &str, default: u32) -> u32 {
                std::env::var(key)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(default)
            }

            let watchdog = WatchdogRuntimeConfig {
                check_interval_ms: parse_env_u64("BRAIN_WATCHDOG_CHECK_INTERVAL_MS", 5000),
                degraded_threshold: parse_env_u32("BRAIN_WATCHDOG_DEGRADED_THRESHOLD", 3),
                dead_threshold: parse_env_u32("BRAIN_WATCHDOG_DEAD_THRESHOLD", 5),
                kill_on_critical_death: parse_env_bool(
                    "BRAIN_WATCHDOG_KILL_ON_CRITICAL_DEATH",
                    true,
                ),
            };

            let preflight = PreflightRuntimeConfig::default();

            BrainRuntimeConfig {
                pipeline: brain_wiring::TradingPipelineConfig::default(),
                watchdog,
                preflight,
                enforce_preflight: parse_env_bool("BRAIN_ENFORCE_PREFLIGHT", true),
                auto_start_watchdog: parse_env_bool("BRAIN_AUTO_START_WATCHDOG", true),
                wire_kill_switch: parse_env_bool("BRAIN_WIRE_KILL_SWITCH", true),
                forward_heartbeat_ms: parse_env_u64("BRAIN_HEARTBEAT_MS", 5000),
            }
        };

        info!(
            "Brain config: auto_watchdog={}, wire_kill_switch={}, heartbeat_ms={}",
            brain_config.auto_start_watchdog,
            brain_config.wire_kill_switch,
            brain_config.forward_heartbeat_ms,
        );

        let mut runtime = BrainRuntime::new(brain_config);

        match runtime.boot().await {
            Ok(report) => {
                info!("✅ Brain Runtime booted: {}", report.summary());

                if let Some(pipeline) = runtime.pipeline() {
                    // --- Wire brain-gated execution into SignalGenerator ---
                    if let Some(ref exec_cfg) = forward_config.execution_config {
                        match ExecutionClient::new(exec_cfg.clone()).await {
                            Ok(exec_client) => {
                                let gated = execution::BrainGatedExecutionClient::new(
                                    exec_client,
                                    Arc::clone(pipeline),
                                );
                                if service.set_brain_gated_client(gated) {
                                    info!(
                                        "🧠✅ Brain-gated execution wired into SignalGenerator (unified)"
                                    );
                                } else {
                                    warn!(
                                        "⚠️  Could not wire brain-gated client (Arc has multiple refs)"
                                    );
                                }
                            }
                            Err(e) => {
                                warn!(
                                    "⚠️  Failed to create execution client for brain gating: {}",
                                    e
                                );
                            }
                        }
                    }

                    // --- Wire BrainHealthState into REST server ---
                    let watchdog_handle = runtime.watchdog_handle().cloned();
                    let boot_passed = runtime
                        .boot_report()
                        .map(|r| r.is_boot_safe())
                        .unwrap_or(false);
                    let boot_summary = runtime.boot_report().map(|r| r.summary());
                    let brain_health = Arc::new(BrainHealthState::new(
                        Some(Arc::clone(pipeline)),
                        watchdog_handle,
                        boot_passed,
                        boot_summary,
                        brain_runtime::RuntimeState::Running,
                    ));
                    brain_rest_state = Some(Arc::clone(&brain_health));
                    service.set_brain_health_state(brain_health);
                    info!("🧠 Brain health REST endpoints wired (unified)");

                    // --- Affinity persistence: load from Redis ---
                    let redis_cfg = persistence::AffinityRedisConfig::from_env();
                    if let Ok(store) = persistence::AffinityRedisStore::new(redis_cfg).await {
                        let store = Arc::new(store);
                        let min_trades: usize = std::env::var("BRAIN_AFFINITY_MIN_TRADES")
                            .unwrap_or_else(|_| "10".to_string())
                            .parse()
                            .unwrap_or(10);
                        match persistence::load_pipeline_affinity(pipeline, &store, min_trades)
                            .await
                        {
                            Ok(true) => info!("🧠✅ Restored affinity state from Redis (unified)"),
                            Ok(false) => info!("🧠 No previous affinity state — starting fresh"),
                            Err(e) => warn!("⚠️  Failed to load affinity from Redis: {}", e),
                        }

                        // Spawn autosave task
                        let interval_secs: u64 =
                            std::env::var("BRAIN_AFFINITY_AUTOSAVE_INTERVAL_SECS")
                                .unwrap_or_else(|_| "300".to_string())
                                .parse()
                                .unwrap_or(300);
                        let handle = persistence::spawn_affinity_autosave(
                            Arc::clone(pipeline),
                            Arc::clone(&store),
                            std::time::Duration::from_secs(interval_secs),
                        );
                        info!(
                            "🧠 Affinity autosave started (interval={}s, unified)",
                            interval_secs
                        );
                        affinity_autosave_handle = Some(handle);
                        affinity_store = Some(store);
                    } else {
                        warn!("⚠️  Redis unavailable — affinity persistence disabled (unified)");
                    }
                }
            }
            Err(e) => {
                warn!(
                    "⚠️  Brain Runtime boot failed in unified mode: {}. Continuing without brain gating.",
                    e
                );
            }
        }

        brain_runtime = Some(runtime);
    } else {
        info!("Brain Runtime disabled in unified mode (ENABLE_BRAIN_RUNTIME=false)");
    }

    // ── Extract watchdog handle for heartbeat wiring ─────────────────
    // If the brain runtime booted successfully and has a watchdog, extract
    // the handle so we can send heartbeats from the live data / signal gen
    // loops. This prevents the watchdog from killing the pipeline because
    // components appear dead.
    let watchdog_handle: Option<janus_cns::watchdog::WatchdogHandle> = brain_runtime
        .as_ref()
        .and_then(|rt| rt.watchdog_handle().cloned());

    if watchdog_handle.is_some() {
        info!("🐕 Watchdog handle extracted — will wire heartbeats into live loops");
    }

    state
        .register_module_health("forward", true, Some("running".to_string()))
        .await;

    // Subscribe to signals from signal bus
    let mut signal_rx = state.signal_bus.subscribe();
    let signal_generator = service.signal_generator();

    // ── Spawn brain REST server (forward REST port, default 8180) ─────────
    // In unified binary mode the forward module's full RestServer is never
    // started (it would conflict with janus_api on port 8080).  Instead we
    // spin up a minimal Axum listener that only mounts the brain health +
    // affinity routes.  Port = janus http port + 100 (matches the offset used
    // when constructing ForwardServiceConfig above).
    // Externally accessible once docker-compose maps 127.0.0.1:7001:8180.
    if let Some(brain_state) = brain_rest_state {
        let brain_port = forward_config.rest_port; // http_port + 100 = 8180
        tokio::spawn(async move {
            let router = crate::api::brain_rest::router(brain_state);
            let addr = format!("0.0.0.0:{}", brain_port);
            match tokio::net::TcpListener::bind(&addr).await {
                Ok(listener) => {
                    info!(
                        "🧠 Brain REST server listening on port {} (forward REST, unified mode)",
                        brain_port
                    );
                    if let Err(e) = axum::serve(listener, router).await {
                        tracing::warn!("⚠️  Brain REST server exited unexpectedly: {}", e);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "⚠️  Failed to bind brain REST server on port {}: {}",
                        brain_port,
                        e
                    );
                }
            }
        });
    }

    // ── Determine data source mode ───────────────────────────────────
    let data_source = std::env::var("DATA_SOURCE")
        .unwrap_or_else(|_| "synthetic".to_string())
        .to_lowercase();

    let live_data_mode = data_source == "live";

    if live_data_mode {
        info!(
            "📡 Forward module: DATA_SOURCE=live — consuming real market data from MarketDataBus"
        );
        info!(
            "   Synthetic signal generator DISABLED — signals will be driven by live indicators + strategies"
        );
    } else {
        info!(
            "🎲 Forward module: DATA_SOURCE={} — using synthetic (demo) signal generator",
            data_source
        );
    }

    // ── Spawn signal generation loop (live or synthetic) ─────────────
    let state_clone = state.clone();
    let signal_gen = Arc::clone(&signal_generator);

    let signal_gen_task = if live_data_mode {
        // ═══════════════════════════════════════════════════════════════
        // LIVE MODE: consume MarketDataEvents, run indicators + strategies
        // ═══════════════════════════════════════════════════════════════
        let mut market_rx = state_clone.market_data_bus.subscribe();
        let wd_handle = watchdog_handle.clone();

        // Read strategy config from JanusState
        let min_confidence = state_clone.config.forward.signals.min_confidence;
        let min_strength = state_clone.config.forward.signals.min_strength;

        // Read indicator config from JanusState
        let ind_config = IndicatorConfig {
            ema_fast_period: state_clone
                .config
                .forward
                .indicators
                .ema_periods
                .first()
                .copied()
                .unwrap_or(8) as usize,
            ema_slow_period: state_clone
                .config
                .forward
                .indicators
                .ema_periods
                .last()
                .copied()
                .unwrap_or(21) as usize,
            rsi_period: state_clone.config.forward.indicators.rsi_period as usize,
            macd_fast_period: state_clone.config.forward.indicators.macd_fast_period as usize,
            macd_slow_period: state_clone.config.forward.indicators.macd_slow_period as usize,
            macd_signal_period: state_clone.config.forward.indicators.macd_signal_period as usize,
            atr_period: state_clone.config.forward.indicators.atr_period as usize,
            bb_period: state_clone.config.forward.indicators.bollinger_period as usize,
            bb_std_dev: state_clone.config.forward.indicators.bollinger_std_dev,
        };

        // Strategy config from JanusState
        let rsi_overbought = state_clone.config.forward.indicators.rsi_overbought;
        let rsi_oversold = state_clone.config.forward.indicators.rsi_oversold;

        tokio::spawn(async move {
            use std::collections::HashMap;

            // Per-symbol indicator analyzers, keyed by "SYMBOL/QUOTE:interval"
            let mut analyzers: HashMap<String, IndicatorAnalyzer> = HashMap::new();

            let mut klines_processed: u64 = 0;
            let mut signals_generated: u64 = 0;
            let mut trades_skipped: u64 = 0;

            info!("🔄 Live signal generation loop started");
            info!(
                "   Indicator config: EMA {}/{}, RSI {}, MACD {}/{}/{}",
                ind_config.ema_fast_period,
                ind_config.ema_slow_period,
                ind_config.rsi_period,
                ind_config.macd_fast_period,
                ind_config.macd_slow_period,
                ind_config.macd_signal_period,
            );
            info!(
                "   Strategy thresholds: min_confidence={:.2}, min_strength={:.2}, RSI ob/os={:.0}/{:.0}",
                min_confidence, min_strength, rsi_overbought, rsi_oversold,
            );

            // ── Watchdog heartbeat sender ────────────────────────────
            // Send periodic heartbeats for all registered watchdog
            // components so the watchdog doesn't trip the kill switch.
            let mut heartbeat_interval = tokio::time::interval(tokio::time::Duration::from_secs(2));
            heartbeat_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            // Send initial heartbeats immediately to cover the startup window
            if let Some(ref wdh) = wd_handle {
                use brain_runtime::components;
                for comp in [
                    components::FORWARD_SERVICE,
                    components::DATA_FEED,
                    components::TRADING_PIPELINE,
                    components::RISK_MANAGER,
                    components::REGIME_DETECTOR,
                ] {
                    wdh.heartbeat(comp).await;
                }
                info!("🐕 Initial watchdog heartbeats sent for all registered components");
            }

            loop {
                tokio::select! {
                    // ── Watchdog heartbeat tick ───────────────────────
                    _ = heartbeat_interval.tick() => {
                        if let Some(ref wdh) = wd_handle {
                            use brain_runtime::components;
                            for comp in [
                                components::FORWARD_SERVICE,
                                components::DATA_FEED,
                                components::TRADING_PIPELINE,
                                components::RISK_MANAGER,
                                components::REGIME_DETECTOR,
                            ] {
                                wdh.heartbeat(comp).await;
                            }
                        }
                    }
                    result = market_rx.recv() => {
                        match result {
                            Ok(janus_core::MarketDataEvent::Kline(kline)) => {
                                // Only process closed klines for indicator calculation
                                if !kline.is_closed {
                                    continue;
                                }

                                let symbol_str = format!("{}", kline.symbol);
                                let analyzer_key = format!("{}:{}", symbol_str, kline.interval);

                                // Get or create analyzer for this symbol+interval
                                let analyzer = analyzers
                                    .entry(analyzer_key.clone())
                                    .or_insert_with(|| IndicatorAnalyzer::new(ind_config.clone()));

                                // Convert Decimal to f64 for indicator calculation
                                use rust_decimal::prelude::ToPrimitive;
                                let high = kline.high.to_f64().unwrap_or(0.0);
                                let low = kline.low.to_f64().unwrap_or(0.0);
                                let close = kline.close.to_f64().unwrap_or(0.0);

                                // Feed the candle into the indicator analyzer
                                if let Err(e) = analyzer.update_hlc(high, low, close) {
                                    warn!("[{}] Failed to update indicators: {}", analyzer_key, e);
                                    continue;
                                }

                                klines_processed += 1;

                                // Run full indicator analysis
                                let analysis = match analyzer.analyze().await {
                                    Ok(a) => a,
                                    Err(e) => {
                                        warn!("[{}] Indicator analysis failed: {}", analyzer_key, e);
                                        continue;
                                    }
                                };

                                // ── Strategy evaluation ──────────────────────
                                // Evaluate multiple strategy signals from the
                                // indicator analysis and produce a consensus.

                                let mut strategy_votes: Vec<(janus_core::SignalType, f64, String)> = Vec::new();

                                // 1. EMA Crossover strategy
                                if analysis.ema_cross > 0.5 {
                                    // Bullish EMA crossover
                                    strategy_votes.push((
                                        janus_core::SignalType::Buy,
                                        analysis.ema_cross.min(1.0),
                                        "ema_crossover".to_string(),
                                    ));
                                } else if analysis.ema_cross < -0.5 {
                                    // Bearish EMA crossover
                                    strategy_votes.push((
                                        janus_core::SignalType::Sell,
                                        analysis.ema_cross.abs().min(1.0),
                                        "ema_crossover".to_string(),
                                    ));
                                }

                                // 2. RSI Reversal strategy
                                if let Some(rsi) = analysis.rsi {
                                    if rsi <= rsi_oversold {
                                        // RSI oversold → potential buy
                                        let strength = ((rsi_oversold - rsi) / rsi_oversold).min(1.0);
                                        strategy_votes.push((
                                            janus_core::SignalType::Buy,
                                            0.5 + strength * 0.5,
                                            "rsi_reversal".to_string(),
                                        ));
                                    } else if rsi >= rsi_overbought {
                                        // RSI overbought → potential sell
                                        let strength = ((rsi - rsi_overbought) / (100.0 - rsi_overbought)).min(1.0);
                                        strategy_votes.push((
                                            janus_core::SignalType::Sell,
                                            0.5 + strength * 0.5,
                                            "rsi_reversal".to_string(),
                                        ));
                                    }
                                }

                                // 3. MACD Momentum strategy
                                if analysis.macd_cross > 0.5 {
                                    strategy_votes.push((
                                        janus_core::SignalType::Buy,
                                        analysis.macd_cross.min(1.0),
                                        "macd_momentum".to_string(),
                                    ));
                                } else if analysis.macd_cross < -0.5 {
                                    strategy_votes.push((
                                        janus_core::SignalType::Sell,
                                        analysis.macd_cross.abs().min(1.0),
                                        "macd_momentum".to_string(),
                                    ));
                                }

                                // 4. Bollinger Band breakout strategy
                                {
                                    let bb_pos: f64 = analysis.bb_position;
                                    if bb_pos <= 0.0 {
                                        // Price at or below lower band → buy
                                        strategy_votes.push((
                                            janus_core::SignalType::Buy,
                                            (0.5_f64 + bb_pos.abs() * 0.5).min(1.0),
                                            "bollinger_breakout".to_string(),
                                        ));
                                    } else if bb_pos >= 1.0 {
                                        // Price at or above upper band → sell
                                        strategy_votes.push((
                                            janus_core::SignalType::Sell,
                                            (0.5_f64 + (bb_pos - 1.0) * 0.5).min(1.0),
                                            "bollinger_breakout".to_string(),
                                        ));
                                    }
                                }

                                // 5. Trend strength confirmation
                                {
                                    let trend: f64 = analysis.trend_strength;
                                    if trend.abs() > 0.6 {
                                        let sig_type = if trend > 0.0 {
                                            janus_core::SignalType::Buy
                                        } else {
                                            janus_core::SignalType::Sell
                                        };
                                        strategy_votes.push((
                                            sig_type,
                                            trend.abs().min(1.0),
                                            "trend_strength".to_string(),
                                        ));
                                    }
                                }

                                // ── Consensus: require at least 2 strategies agreeing ──
                                if strategy_votes.is_empty() {
                                    continue;
                                }

                                let buy_votes: Vec<&(janus_core::SignalType, f64, String)> = strategy_votes
                                    .iter()
                                    .filter(|(st, _, _)| *st == janus_core::SignalType::Buy)
                                    .collect();
                                let sell_votes: Vec<&(janus_core::SignalType, f64, String)> = strategy_votes
                                    .iter()
                                    .filter(|(st, _, _)| *st == janus_core::SignalType::Sell)
                                    .collect();

                                let min_strategies = state_clone.config.forward.strategies.consensus.min_strategies as usize;
                                let min_agreement = state_clone.config.forward.strategies.consensus.min_agreement;

                                let (final_type, avg_confidence, strategies_used) =
                                    if buy_votes.len() >= min_strategies
                                        && buy_votes.len() as f64 / strategy_votes.len() as f64 >= min_agreement
                                    {
                                        let avg = buy_votes.iter().map(|(_, c, _)| c).sum::<f64>() / buy_votes.len() as f64;
                                        let strats: Vec<String> = buy_votes.iter().map(|(_, _, s)| s.clone()).collect();
                                        (janus_core::SignalType::Buy, avg, strats)
                                    } else if sell_votes.len() >= min_strategies
                                        && sell_votes.len() as f64 / strategy_votes.len() as f64 >= min_agreement
                                    {
                                        let avg = sell_votes.iter().map(|(_, c, _)| c).sum::<f64>() / sell_votes.len() as f64;
                                        let strats: Vec<String> = sell_votes.iter().map(|(_, _, s)| s.clone()).collect();
                                        (janus_core::SignalType::Sell, avg, strats)
                                    } else {
                                        // No consensus — Hold
                                        continue;
                                    };

                                // Apply minimum confidence/strength filters
                                if avg_confidence < min_confidence {
                                    tracing::debug!(
                                        "[{}] Signal confidence {:.2} below threshold {:.2} — skipping",
                                        analyzer_key, avg_confidence, min_confidence,
                                    );
                                    continue;
                                }

                                // Create the signal with full metadata
                                let signal = janus_core::Signal::new(&symbol_str, final_type, avg_confidence)
                                    .with_source("forward")
                                    .with_strategy(strategies_used.join("+"))
                                    .with_metadata("data_source", "live")
                                    .with_metadata("interval", &kline.interval)
                                    .with_metadata("exchange", format!("{}", kline.exchange))
                                    .with_metadata("close_price", format!("{}", kline.close))
                                    .with_metadata("strategies_count", format!("{}", strategies_used.len()))
                                    .with_metadata("total_votes", format!("{}", strategy_votes.len()));

                                // Add indicator values as metadata
                                let signal = if let Some(rsi) = analysis.rsi {
                                    signal.with_metadata("rsi", format!("{:.2}", rsi))
                                } else { signal };
                                let signal = if let Some(ema_f) = analysis.ema_fast {
                                    signal.with_metadata("ema_fast", format!("{:.4}", ema_f))
                                } else { signal };
                                let signal = if let Some(ema_s) = analysis.ema_slow {
                                    signal.with_metadata("ema_slow", format!("{:.4}", ema_s))
                                } else { signal };

                                // Publish to signal bus
                                match state_clone.signal_bus.publish(signal.clone()) {
                                    Ok(receivers) => {
                                        signals_generated += 1;

                                        // Record in Prometheus global metrics
                                        let sig_type_str = format!("{:?}", signal.signal_type).to_lowercase();
                                        janus_metrics().record_signal(
                                            "forward",
                                            &sig_type_str,
                                            &signal.symbol.to_string(),
                                            signal.confidence,
                                        );

                                        info!(
                                            "📊 [LIVE] Signal #{} {} {} {:?} confidence={:.2} strategies=[{}] close={} interval={} → {} receivers",
                                            signals_generated,
                                            signal.id,
                                            signal.symbol,
                                            signal.signal_type,
                                            signal.confidence,
                                            strategies_used.join(", "),
                                            kline.close,
                                            kline.interval,
                                            receivers,
                                        );
                                        state_clone.increment_signals_generated();
                                    }
                                    Err(e) => {
                                        tracing::error!("Failed to publish live signal: {}", e);
                                    }
                                }

                                // Submit to execution service if actionable
                                if final_type != janus_core::SignalType::Hold && avg_confidence >= 0.7
                                    && let Err(e) = signal_gen.submit_signal_to_execution(&signal).await {
                                    warn!("Failed to submit live signal to execution: {}", e);
                                }

                                // Update health with live stats
                                state_clone
                                    .register_module_health("forward", true, Some(format!(
                                        "live: {} klines processed, {} signals generated",
                                        klines_processed, signals_generated,
                                    )))
                                    .await;
                            }
                            Ok(janus_core::MarketDataEvent::Trade(_)) => {
                                // Trade events are high-frequency; we primarily use
                                // klines for indicator-driven signal generation.
                                // Trade data can be used later for order flow analysis.
                                trades_skipped += 1;
                            }
                            Ok(_other) => {
                                // OrderBook, Ticker, etc. — not yet used for signal generation
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                                warn!(
                                    "Forward market data consumer lagged by {} events — some klines may have been skipped",
                                    n
                                );
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                                info!("MarketDataBus closed — stopping live signal generation");
                                break;
                            }
                        }
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                        if state_clone.is_shutdown_requested() {
                            break;
                        }
                    }
                }
            }

            info!(
                "Live signal generation loop exited — {} klines processed, {} signals generated, {} trades skipped",
                klines_processed, signals_generated, trades_skipped,
            );
        })
    } else {
        // ═══════════════════════════════════════════════════════════════
        // SYNTHETIC MODE: original demo/random signal generator
        // ═══════════════════════════════════════════════════════════════
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
                state_clone.config.forward.signal_interval_secs,
            ));

            // Get enabled symbols from config
            let symbols: Vec<String> = state_clone.config.assets.enabled_symbols();
            info!(
                "Synthetic signal generation enabled for {} symbols: {:?}",
                symbols.len(),
                symbols
            );

            let mut rng_state: u64 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Generate signals periodically
                        info!("Forward module generating synthetic signals...");

                        // Simple pseudo-random number generator (xorshift)
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;

                        // Pick a random symbol
                        let symbol_idx = (rng_state as usize) % symbols.len();
                        let symbol = &symbols[symbol_idx];

                        // Generate random confidence (0.5 to 1.0)
                        let confidence = 0.5 + (((rng_state >> 8) % 500) as f64 / 1000.0);

                        // Determine signal type based on pseudo-random
                        let signal_type = if (rng_state >> 16).is_multiple_of(3) {
                            janus_core::SignalType::Buy
                        } else if (rng_state >> 16) % 3 == 1 {
                            janus_core::SignalType::Sell
                        } else {
                            janus_core::SignalType::Hold
                        };

                        // Create the signal
                        let signal = janus_core::Signal::new(symbol, signal_type, confidence)
                            .with_source("forward")
                            .with_strategy("demo_strategy")
                            .with_metadata("data_source", "synthetic");

                        // Publish to signal bus
                        match state_clone.signal_bus.publish(signal.clone()) {
                            Ok(receivers) => {
                                // Record in Prometheus global metrics
                                let sig_type_str = format!("{:?}", signal.signal_type).to_lowercase();
                                janus_metrics().record_signal(
                                    "forward",
                                    &sig_type_str,
                                    &signal.symbol.to_string(),
                                    signal.confidence,
                                );

                                info!(
                                    "Published synthetic signal {} for {} ({:?}, confidence: {:.2}) to {} receivers",
                                    signal.id, signal.symbol, signal.signal_type, signal.confidence, receivers
                                );
                                state_clone.increment_signals_generated();
                            }
                            Err(e) => {
                                tracing::error!("Failed to publish signal: {}", e);
                            }
                        }

                        // Submit to execution service if enabled and signal is actionable
                        if signal_type != janus_core::SignalType::Hold && confidence >= 0.7
                            && let Err(e) = signal_gen.submit_signal_to_execution(&signal).await {
                            warn!("Failed to submit signal to execution: {}", e);
                        }

                        // Update health
                        state_clone
                            .register_module_health("forward", true, Some("generating synthetic signals".to_string()))
                            .await;
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                        if state_clone.is_shutdown_requested() {
                            break;
                        }
                    }
                }
            }
            info!("Synthetic signal generation loop exited");
        })
    };

    // Spawn signal receiver loop (processes signals from other modules)
    let state_clone = state.clone();
    let signal_rx_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                Ok(signal) = signal_rx.recv() => {
                    let source_label = signal.metadata.get("data_source")
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    info!(
                        "Forward module received signal: {} {} {} (confidence: {:.2}, source: {}, data: {})",
                        signal.symbol,
                        signal.signal_type,
                        signal.source,
                        signal.confidence,
                        signal.source,
                        source_label,
                    );

                    // Route signal through risk validation and log outcome.
                    // Full brain-pipeline gating is handled by BrainGatedExecutionClient
                    // when wired via main.rs; here we perform a lightweight risk check.
                    if signal.confidence < 0.5 {
                        info!(
                            "Signal confidence {:.2} below threshold — skipping {}",
                            signal.confidence, signal.symbol
                        );
                    } else {
                        info!(
                            "✅ Signal accepted for processing: {} {} (confidence: {:.2})",
                            signal.symbol, signal.signal_type, signal.confidence
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
        info!("Forward signal receiver loop exited");
    });

    // Wait for shutdown
    while !state.is_shutdown_requested() {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        // Update health periodically
        state
            .register_module_health("forward", true, Some("running".to_string()))
            .await;
    }

    info!("Forward module shutting down...");

    // Stop affinity autosave
    if let Some(handle) = affinity_autosave_handle {
        handle.abort();
        let _ = handle.await;
    }

    // Save affinity state one final time
    if let (Some(runtime), Some(store)) = (&brain_runtime, &affinity_store)
        && let Some(pipeline) = runtime.pipeline()
    {
        match persistence::save_pipeline_affinity(pipeline, store).await {
            Ok(()) => info!("🧠✅ Saved affinity state to Redis on shutdown (unified)"),
            Err(e) => warn!("⚠️  Failed to save affinity on shutdown: {}", e),
        }
    }

    // Shutdown brain runtime
    if let Some(ref mut runtime) = brain_runtime {
        runtime.shutdown().await;
    }

    // Cancel tasks
    signal_gen_task.abort();
    signal_rx_task.abort();

    // Shutdown service
    service
        .shutdown()
        .await
        .map_err(|e| janus_core::Error::module("forward", e.to_string()))?;

    state
        .register_module_health("forward", false, Some("stopped".to_string()))
        .await;

    info!("Forward module exited");
    Ok(())
}

/// Health check status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    pub service: String,
    pub version: String,
    pub status: String,
    pub signal_metrics: SignalMetricsSnapshot,
}

/// Snapshot of signal metrics for health checks
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SignalMetricsSnapshot {
    pub total_generated: u64,
    pub total_filtered: u64,
    pub filter_rate: f64,
}

/// Handle returned when starting the service with param reload
///
/// Contains references to the reload manager and appliers for accessing
/// optimized parameters at runtime.
/// Lightweight handle for querying optimized parameters without owning the task
///
/// This handle can be cloned and shared across components (e.g., SignalGenerator)
/// to query the current optimized configs without needing to manage the background task.
#[derive(Clone)]
pub struct ParamQueryHandle {
    /// Indicator parameter applier - query for indicator configs
    pub indicator_applier: Arc<IndicatorParamApplier>,

    /// Risk parameter applier - query for risk configs
    pub risk_applier: Arc<RiskParamApplier>,

    /// Strategy parameter applier - query for strategy configs
    pub strategy_applier: Arc<StrategyParamApplier>,
}

impl ParamQueryHandle {
    /// Get current indicator config for an asset
    pub async fn get_indicator_config(&self, asset: &str) -> Option<IndicatorConfig> {
        self.indicator_applier.get_config(asset).await
    }

    /// Check if trading is enabled for an asset
    pub async fn is_trading_enabled(&self, asset: &str) -> bool {
        self.risk_applier.is_trading_enabled(asset).await
    }

    /// Check minimum EMA spread for an asset
    pub async fn check_min_ema_spread(&self, asset: &str, spread: f64) -> bool {
        self.strategy_applier
            .check_min_ema_spread(asset, spread)
            .await
    }

    /// Check minimum hold time for an asset
    pub async fn check_min_hold_time(&self, asset: &str, held_minutes: u32) -> bool {
        self.strategy_applier
            .check_min_hold_time(asset, held_minutes)
            .await
    }

    /// Get the asset risk config for risk management decisions
    pub async fn get_asset_risk_config(
        &self,
        asset: &str,
    ) -> Option<crate::param_reload::appliers::AssetRiskConfig> {
        self.risk_applier.get_asset_config(asset).await
    }

    /// Get max position size for an asset
    pub async fn get_max_position_size(&self, asset: &str) -> Option<f64> {
        self.risk_applier.get_max_position_size(asset).await
    }
}

pub struct ParamReloadHandle {
    /// The param reload manager
    pub manager: Arc<ParamReloadManager>,

    /// Background task handle for the reload listener
    pub task_handle: tokio::task::JoinHandle<()>,

    /// Query handle for accessing optimized params (can be cloned and shared)
    pub query: ParamQueryHandle,
}

impl ParamReloadHandle {
    /// Get current indicator config for an asset
    pub async fn get_indicator_config(&self, asset: &str) -> Option<IndicatorConfig> {
        self.query.get_indicator_config(asset).await
    }

    /// Check if trading is enabled for an asset
    pub async fn is_trading_enabled(&self, asset: &str) -> bool {
        self.query.is_trading_enabled(asset).await
    }

    /// Check minimum EMA spread for an asset
    pub async fn check_min_ema_spread(&self, asset: &str, spread: f64) -> bool {
        self.query.check_min_ema_spread(asset, spread).await
    }

    /// Check minimum hold time for an asset
    pub async fn check_min_hold_time(&self, asset: &str, held_minutes: u32) -> bool {
        self.query.check_min_hold_time(asset, held_minutes).await
    }

    /// Get the query handle (can be cloned and shared with other components)
    pub fn query_handle(&self) -> ParamQueryHandle {
        self.query.clone()
    }

    /// Get reload statistics
    pub async fn stats(&self) -> ReloadStats {
        self.manager.stats().await
    }

    /// Stop the param reload background task
    pub async fn stop(&self) {
        self.manager.stop().await;
        self.task_handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_creation() {
        let config = ForwardServiceConfig::default();
        let service = ForwardService::new(config).await.unwrap();

        assert_eq!(service.config().grpc_port, 50051);
        assert_eq!(service.config().rest_port, 8080);
        assert_eq!(service.config().websocket_port, 8081);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = ForwardServiceConfig::default();
        let service = ForwardService::new(config).await.unwrap();

        let health = service.health_check();
        assert_eq!(health.service, "janus-forward");
        assert_eq!(health.version, VERSION);
        assert_eq!(health.status, "healthy");
    }

    #[test]
    fn test_service_constants() {
        assert_eq!(SERVICE_NAME, "janus-forward");
        assert!(!VERSION.is_empty());
    }

    #[tokio::test]
    async fn test_param_query_handle_creation() {
        // Create appliers
        let indicator_applier = Arc::new(IndicatorParamApplier::new());
        let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
        let risk_applier = Arc::new(RiskParamApplier::new(risk_manager));
        let strategy_applier = Arc::new(StrategyParamApplier::new());

        // Create query handle
        let query_handle = ParamQueryHandle {
            indicator_applier,
            risk_applier,
            strategy_applier,
        };

        // Verify default behaviors (no params loaded)
        assert!(query_handle.is_trading_enabled("BTC").await);
        assert!(query_handle.get_indicator_config("BTC").await.is_none());
        // Default min EMA spread is 0.2%, so 0.5% should pass
        assert!(query_handle.check_min_ema_spread("BTC", 0.5).await);
        // Default min hold time is 15 minutes, so 20 should pass
        assert!(query_handle.check_min_hold_time("BTC", 20).await);
        // Below default thresholds should fail
        assert!(!query_handle.check_min_ema_spread("BTC", 0.1).await);
        assert!(!query_handle.check_min_hold_time("BTC", 10).await);
    }

    #[tokio::test]
    async fn test_signal_generator_param_reload_wiring() {
        use crate::signal::SignalGeneratorConfig;

        // Create signal generator
        let config = SignalGeneratorConfig::default();
        let mut signal_gen = SignalGenerator::new(config);

        // Initially no param reload
        assert!(!signal_gen.has_param_reload());

        // Create and wire param query handle
        let indicator_applier = Arc::new(IndicatorParamApplier::new());
        let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
        let risk_applier = Arc::new(RiskParamApplier::new(risk_manager));
        let strategy_applier = Arc::new(StrategyParamApplier::new());

        let query_handle = ParamQueryHandle {
            indicator_applier,
            risk_applier,
            strategy_applier,
        };

        signal_gen.set_param_query_handle(query_handle);

        // Now param reload is configured
        assert!(signal_gen.has_param_reload());

        // Verify query methods work
        assert!(signal_gen.is_trading_enabled("BTC").await);
        assert!(
            signal_gen
                .get_optimized_indicator_config("BTC")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_risk_manager_param_reload_wiring() {
        // Create risk manager
        let mut risk_manager = RiskManager::new(Default::default());

        // Initially no param reload
        assert!(!risk_manager.has_param_reload());

        // Create and wire param query handle
        let indicator_applier = Arc::new(IndicatorParamApplier::new());
        let rm_arc = Arc::new(RwLock::new(RiskManager::new(Default::default())));
        let risk_applier = Arc::new(RiskParamApplier::new(rm_arc));
        let strategy_applier = Arc::new(StrategyParamApplier::new());

        let query_handle = ParamQueryHandle {
            indicator_applier,
            risk_applier,
            strategy_applier,
        };

        risk_manager.set_param_query_handle(query_handle);

        // Now param reload is configured
        assert!(risk_manager.has_param_reload());

        // Verify query methods work (defaults since no params loaded)
        assert!(risk_manager.is_trading_enabled("BTC").await);
        assert!(risk_manager.get_asset_risk_config("BTC").await.is_none());
    }
}
