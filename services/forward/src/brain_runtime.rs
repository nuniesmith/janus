//! # Brain Runtime — Boot Orchestration Layer
//!
//! Orchestrates the full brain-inspired trading system startup:
//!
//! 1. **Pre-flight checks** — Run infrastructure, sensory, regulatory,
//!    strategy, and executive checks before enabling trading.
//! 2. **Watchdog registration** — Register all components with the CNS
//!    watchdog for continuous health monitoring.
//! 3. **Pipeline initialization** — Wire the `TradingPipeline` with
//!    correlation tracker, affinity state, and strategy gating.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        BrainRuntime                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
//! │  │  PreFlight   │───▶│   Watchdog   │───▶│   Trading    │     │
//! │  │   Runner     │    │  (heartbeat) │    │   Pipeline   │     │
//! │  └──────────────┘    └──────────────┘    └──────────────┘     │
//! │         │                    │                    │             │
//! │         ▼                    ▼                    ▼             │
//! │    Boot Report         Component           Trade Decisions     │
//! │                        Health                                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::brain_runtime::{BrainRuntime, BrainRuntimeConfig};
//!
//! let config = BrainRuntimeConfig::default();
//! let mut runtime = BrainRuntime::new(config);
//!
//! // Boot sequence
//! let report = runtime.boot().await?;
//! if !report.is_boot_safe() {
//!     eprintln!("Boot failed: {}", report.summary());
//!     return Err(anyhow!("Pre-flight checks failed"));
//! }
//!
//! // Start watchdog monitoring
//! runtime.start_watchdog().await;
//!
//! // Evaluate trades through the pipeline
//! let decision = runtime.pipeline().evaluate(...).await;
//! ```

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use janus_cns::preflight::{
    BootPhase, BootReport, Criticality, PreFlightCheck, PreFlightConfig, PreFlightRunner,
};
use janus_cns::watchdog::{
    CnsWatchdog, ComponentCriticality, ComponentRegistration, ComponentState, KillSwitch,
    NoOpKillSwitch, WatchdogConfig, WatchdogHandle, WatchdogSnapshot,
};
use janus_risk::CorrelationConfig;
use janus_strategies::gating::StrategyGatingConfig;

use crate::brain_wiring::{TradingPipeline, TradingPipelineBuilder, TradingPipelineConfig};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the brain runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainRuntimeConfig {
    /// Trading pipeline configuration.
    #[serde(default)]
    pub pipeline: TradingPipelineConfig,

    /// Watchdog configuration.
    #[serde(default)]
    pub watchdog: WatchdogRuntimeConfig,

    /// Pre-flight configuration.
    #[serde(default)]
    pub preflight: PreflightRuntimeConfig,

    /// Whether to enforce pre-flight checks (if false, warnings are logged
    /// but boot proceeds regardless).
    #[serde(default = "default_true")]
    pub enforce_preflight: bool,

    /// Whether to start the watchdog automatically on boot.
    #[serde(default = "default_true")]
    pub auto_start_watchdog: bool,

    /// Whether to wire the watchdog kill-switch to the pipeline kill-switch.
    #[serde(default = "default_true")]
    pub wire_kill_switch: bool,

    /// Heartbeat interval for the forward service component (ms).
    #[serde(default = "default_heartbeat_ms")]
    pub forward_heartbeat_ms: u64,
}

fn default_true() -> bool {
    true
}

fn default_heartbeat_ms() -> u64 {
    1000
}

impl Default for BrainRuntimeConfig {
    fn default() -> Self {
        Self {
            pipeline: TradingPipelineConfig::default(),
            watchdog: WatchdogRuntimeConfig::default(),
            preflight: PreflightRuntimeConfig::default(),
            enforce_preflight: true,
            auto_start_watchdog: true,
            wire_kill_switch: true,
            forward_heartbeat_ms: default_heartbeat_ms(),
        }
    }
}

/// Watchdog-specific runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogRuntimeConfig {
    /// Check interval in milliseconds.
    #[serde(default = "default_check_interval")]
    pub check_interval_ms: u64,

    /// Number of missed heartbeats before degraded.
    #[serde(default = "default_degraded_threshold")]
    pub degraded_threshold: u32,

    /// Number of missed heartbeats before dead.
    #[serde(default = "default_dead_threshold")]
    pub dead_threshold: u32,

    /// Whether to kill on critical component death.
    #[serde(default = "default_true")]
    pub kill_on_critical_death: bool,
}

fn default_check_interval() -> u64 {
    500
}
fn default_degraded_threshold() -> u32 {
    3
}
fn default_dead_threshold() -> u32 {
    5
}

impl Default for WatchdogRuntimeConfig {
    fn default() -> Self {
        Self {
            check_interval_ms: default_check_interval(),
            degraded_threshold: default_degraded_threshold(),
            dead_threshold: default_dead_threshold(),
            kill_on_critical_death: true,
        }
    }
}

impl WatchdogRuntimeConfig {
    /// Convert to the CNS watchdog config format.
    pub fn to_watchdog_config(&self) -> WatchdogConfig {
        WatchdogConfig {
            check_interval_ms: self.check_interval_ms,
            degraded_threshold: self.degraded_threshold,
            dead_threshold: self.dead_threshold,
            kill_on_critical_death: self.kill_on_critical_death,
            ..Default::default()
        }
    }
}

/// Pre-flight specific runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightRuntimeConfig {
    /// Whether to abort on critical check failure.
    #[serde(default = "default_true")]
    pub abort_on_critical: bool,

    /// Whether to run checks within a phase in parallel.
    #[serde(default = "default_true")]
    pub parallel_within_phase: bool,

    /// Global timeout for all pre-flight checks (seconds).
    #[serde(default = "default_preflight_timeout")]
    pub global_timeout_secs: u64,

    /// Which phases to run (if empty, all phases run).
    #[serde(default)]
    pub enabled_phases: Vec<String>,

    /// Whether to skip infrastructure checks (useful in dev/test).
    #[serde(default)]
    pub skip_infra_checks: bool,
}

fn default_preflight_timeout() -> u64 {
    60
}

impl Default for PreflightRuntimeConfig {
    fn default() -> Self {
        Self {
            abort_on_critical: true,
            parallel_within_phase: true,
            global_timeout_secs: default_preflight_timeout(),
            enabled_phases: Vec::new(),
            skip_infra_checks: false,
        }
    }
}

impl PreflightRuntimeConfig {
    /// Convert to the CNS preflight config format.
    pub fn to_preflight_config(&self) -> PreFlightConfig {
        PreFlightConfig {
            abort_on_critical: self.abort_on_critical,
            parallel_within_phase: self.parallel_within_phase,
            global_timeout: Duration::from_secs(self.global_timeout_secs),
        }
    }
}

// ============================================================================
// Boot State
// ============================================================================

/// The current state of the brain runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    /// Not yet booted.
    Uninitialized,
    /// Pre-flight checks running.
    Booting,
    /// Pre-flight passed, watchdog starting.
    Starting,
    /// Fully operational.
    Running,
    /// Shutting down.
    ShuttingDown,
    /// Stopped.
    Stopped,
    /// Boot failed.
    Failed,
}

impl std::fmt::Display for RuntimeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeState::Uninitialized => write!(f, "Uninitialized"),
            RuntimeState::Booting => write!(f, "Booting"),
            RuntimeState::Starting => write!(f, "Starting"),
            RuntimeState::Running => write!(f, "Running"),
            RuntimeState::ShuttingDown => write!(f, "ShuttingDown"),
            RuntimeState::Stopped => write!(f, "Stopped"),
            RuntimeState::Failed => write!(f, "Failed"),
        }
    }
}

// ============================================================================
// Component Descriptors
// ============================================================================

/// Known component names for watchdog registration.
pub mod components {
    /// The forward service main loop.
    pub const FORWARD_SERVICE: &str = "forward-service";
    /// Regime detection subsystem.
    pub const REGIME_DETECTOR: &str = "regime-detector";
    /// The trading pipeline.
    pub const TRADING_PIPELINE: &str = "trading-pipeline";
    /// WebSocket data feed.
    pub const DATA_FEED: &str = "data-feed";
    /// Execution client connection.
    pub const EXECUTION_CLIENT: &str = "execution-client";
    /// Risk manager.
    pub const RISK_MANAGER: &str = "risk-manager";
    /// Strategy engine.
    pub const STRATEGY_ENGINE: &str = "strategy-engine";
    /// Correlation tracker.
    pub const CORRELATION_TRACKER: &str = "correlation-tracker";
}

// ============================================================================
// Built-in Pre-flight Checks
// ============================================================================

/// A simple pre-flight check that always passes.
/// Used as a marker that the pipeline is constructable.
struct PipelineConstructCheck;

#[async_trait::async_trait]
impl PreFlightCheck for PipelineConstructCheck {
    fn name(&self) -> &str {
        "pipeline-construct"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Strategy
    }

    fn criticality(&self) -> Criticality {
        Criticality::Required
    }

    async fn execute(&self) -> Result<(), String> {
        // The fact that we got here means the pipeline config is valid
        // and we can construct a TradingPipeline.
        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some("TradingPipeline is constructable with current config".to_string())
    }
}

/// Check that the strategy gating config is valid.
struct GatingConfigCheck {
    config: StrategyGatingConfig,
}

impl GatingConfigCheck {
    fn new(config: StrategyGatingConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl PreFlightCheck for GatingConfigCheck {
    fn name(&self) -> &str {
        "strategy-gating-config"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Strategy
    }

    fn criticality(&self) -> Criticality {
        Criticality::Required
    }

    async fn execute(&self) -> Result<(), String> {
        // Validate the min_weight is sane
        if self.config.min_weight < 0.0 || self.config.min_weight > 1.0 {
            return Err(format!(
                "min_weight out of range [0, 1]: {}",
                self.config.min_weight,
            ));
        }
        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!(
            "min_weight={:.2}, allow_untested={}, assets_configured={}",
            self.config.min_weight,
            self.config.allow_untested,
            self.config.assets.len(),
        ))
    }
}

/// Check that the correlation config is valid.
struct CorrelationConfigCheck {
    config: CorrelationConfig,
}

impl CorrelationConfigCheck {
    fn new(config: CorrelationConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl PreFlightCheck for CorrelationConfigCheck {
    fn name(&self) -> &str {
        "correlation-config"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Strategy
    }

    fn criticality(&self) -> Criticality {
        Criticality::Required
    }

    async fn execute(&self) -> Result<(), String> {
        if self.config.window == 0 {
            return Err("correlation window cannot be 0".to_string());
        }
        if self.config.correlation_threshold <= 0.0 || self.config.correlation_threshold >= 1.0 {
            return Err(format!(
                "correlation_threshold out of range (0, 1): {}",
                self.config.correlation_threshold,
            ));
        }
        if self.config.max_correlated_positions == 0 {
            return Err("max_correlated_positions cannot be 0".to_string());
        }
        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!(
            "window={}, threshold={:.2}, max_correlated={}",
            self.config.window,
            self.config.correlation_threshold,
            self.config.max_correlated_positions,
        ))
    }
}

/// Check that the pipeline config values are within reasonable bounds.
struct PipelineConfigCheck {
    config: TradingPipelineConfig,
}

impl PipelineConfigCheck {
    fn new(config: TradingPipelineConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl PreFlightCheck for PipelineConfigCheck {
    fn name(&self) -> &str {
        "pipeline-config-validation"
    }

    fn phase(&self) -> BootPhase {
        BootPhase::Strategy
    }

    fn criticality(&self) -> Criticality {
        Criticality::Required
    }

    async fn execute(&self) -> Result<(), String> {
        if self.config.max_position_scale <= 0.0 {
            return Err("max_position_scale must be positive".to_string());
        }
        if self.config.min_position_scale < 0.0 {
            return Err("min_position_scale cannot be negative".to_string());
        }
        if self.config.min_position_scale >= self.config.max_position_scale {
            return Err(format!(
                "min_position_scale ({}) >= max_position_scale ({})",
                self.config.min_position_scale, self.config.max_position_scale,
            ));
        }
        if self.config.high_risk_scale_factor <= 0.0 || self.config.high_risk_scale_factor > 1.0 {
            return Err(format!(
                "high_risk_scale_factor out of range (0, 1]: {}",
                self.config.high_risk_scale_factor,
            ));
        }
        if self.config.min_regime_confidence < 0.0 || self.config.min_regime_confidence > 1.0 {
            return Err(format!(
                "min_regime_confidence out of range [0, 1]: {}",
                self.config.min_regime_confidence,
            ));
        }
        Ok(())
    }

    fn detail_on_success(&self) -> Option<String> {
        Some(format!(
            "scale=[{:.2}, {:.2}], risk_factor={:.2}, min_conf={:.2}",
            self.config.min_position_scale,
            self.config.max_position_scale,
            self.config.high_risk_scale_factor,
            self.config.min_regime_confidence,
        ))
    }
}

// ============================================================================
// Brain Runtime
// ============================================================================

/// The brain runtime — top-level orchestrator for the brain-inspired trading system.
///
/// Manages boot sequence (preflight), runtime monitoring (watchdog), and
/// the trading pipeline (regime→hypothalamus→amygdala→gating→correlation→execution).
pub struct BrainRuntime {
    config: BrainRuntimeConfig,
    state: RuntimeState,
    pipeline: Option<Arc<TradingPipeline>>,
    watchdog_handle: Option<WatchdogHandle>,
    boot_report: Option<BootReport>,
    watchdog_task: Option<tokio::task::JoinHandle<()>>,
    /// Dedicated OS thread for the watchdog runtime — ensures the watchdog
    /// keeps ticking even if the main Tokio runtime is starved or blocked.
    watchdog_thread: Option<std::thread::JoinHandle<()>>,
    /// Sender to signal the watchdog's dedicated runtime to shut down.
    watchdog_shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl BrainRuntime {
    /// Create a new brain runtime with the given configuration.
    pub fn new(config: BrainRuntimeConfig) -> Self {
        Self {
            config,
            state: RuntimeState::Uninitialized,
            pipeline: None,
            watchdog_handle: None,
            boot_report: None,
            watchdog_task: None,
            watchdog_thread: None,
            watchdog_shutdown: None,
        }
    }

    /// Create a runtime with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BrainRuntimeConfig::default())
    }

    /// Get the current runtime state.
    pub fn state(&self) -> RuntimeState {
        self.state
    }

    /// Get a reference to the trading pipeline (if initialized).
    pub fn pipeline(&self) -> Option<&Arc<TradingPipeline>> {
        self.pipeline.as_ref()
    }

    /// Get the boot report (if boot has been run).
    pub fn boot_report(&self) -> Option<&BootReport> {
        self.boot_report.as_ref()
    }

    /// Get the watchdog handle (if watchdog has been started).
    pub fn watchdog_handle(&self) -> Option<&WatchdogHandle> {
        self.watchdog_handle.as_ref()
    }

    /// Get the runtime configuration.
    pub fn config(&self) -> &BrainRuntimeConfig {
        &self.config
    }

    // ────────────────────────────────────────────────────────────────────
    // Boot Sequence
    // ────────────────────────────────────────────────────────────────────

    /// Execute the full boot sequence:
    /// 1. Run pre-flight checks
    /// 2. Initialize the trading pipeline
    /// 3. Optionally start the watchdog
    ///
    /// Returns the boot report. If `enforce_preflight` is true and checks
    /// fail, returns an error.
    pub async fn boot(&mut self) -> Result<&BootReport> {
        self.state = RuntimeState::Booting;

        info!("╔═══════════════════════════════════════════════════════════╗");
        info!("║          BRAIN RUNTIME — Boot Sequence Starting          ║");
        info!("╚═══════════════════════════════════════════════════════════╝");

        // ── Step 1: Pre-flight checks ──────────────────────────────────
        let report = self.run_preflight_checks().await;

        info!("\n{}", report.full_report());

        if !report.is_boot_safe() {
            if self.config.enforce_preflight {
                self.state = RuntimeState::Failed;
                self.boot_report = Some(report);
                return Err(anyhow::anyhow!(
                    "Pre-flight checks failed: {}",
                    self.boot_report.as_ref().unwrap().summary(),
                ));
            } else {
                warn!(
                    "⚠️  Pre-flight checks failed but enforcement is disabled: {}",
                    report.summary(),
                );
            }
        }

        self.boot_report = Some(report);

        // ── Step 2: Initialize the trading pipeline ────────────────────
        self.state = RuntimeState::Starting;
        info!("Initializing trading pipeline...");

        let pipeline = TradingPipelineBuilder::new()
            .config(self.config.pipeline.clone())
            .build();

        let pipeline = Arc::new(pipeline);
        self.pipeline = Some(pipeline.clone());

        info!("✅ Trading pipeline initialized");

        // ── Step 3: Optionally start the watchdog ──────────────────────
        if self.config.auto_start_watchdog {
            self.start_watchdog_internal(pipeline).await?;
        }

        self.state = RuntimeState::Running;
        info!("✅ Brain runtime is RUNNING");

        Ok(self.boot_report.as_ref().unwrap())
    }

    /// Run the full boot sequence, but with a pre-built pipeline
    /// (useful for testing or when you want to inject custom components).
    pub async fn boot_with_pipeline(
        &mut self,
        pipeline: Arc<TradingPipeline>,
    ) -> Result<&BootReport> {
        self.state = RuntimeState::Booting;

        let report = self.run_preflight_checks().await;
        info!("\n{}", report.summary());

        if !report.is_boot_safe() && self.config.enforce_preflight {
            self.state = RuntimeState::Failed;
            self.boot_report = Some(report);
            return Err(anyhow::anyhow!(
                "Pre-flight checks failed: {}",
                self.boot_report.as_ref().unwrap().summary(),
            ));
        }

        self.boot_report = Some(report);
        self.state = RuntimeState::Starting;

        self.pipeline = Some(pipeline.clone());

        if self.config.auto_start_watchdog {
            self.start_watchdog_internal(pipeline).await?;
        }

        self.state = RuntimeState::Running;
        Ok(self.boot_report.as_ref().unwrap())
    }

    /// Run only pre-flight checks without initializing the pipeline.
    /// Useful for dry-run / health diagnostics.
    pub async fn preflight_only(&mut self) -> BootReport {
        self.state = RuntimeState::Booting;
        let report = self.run_preflight_checks().await;
        self.state = if report.is_boot_safe() {
            RuntimeState::Uninitialized
        } else {
            RuntimeState::Failed
        };
        self.boot_report = Some(report);
        self.boot_report.as_ref().unwrap().clone()
    }

    // ────────────────────────────────────────────────────────────────────
    // Internal: Pre-flight
    // ────────────────────────────────────────────────────────────────────

    async fn run_preflight_checks(&self) -> BootReport {
        let preflight_config = self.config.preflight.to_preflight_config();
        let mut runner = PreFlightRunner::with_config(preflight_config);

        // Add built-in checks
        runner.add_check(Box::new(PipelineConfigCheck::new(
            self.config.pipeline.clone(),
        )));
        runner.add_check(Box::new(GatingConfigCheck::new(
            self.config.pipeline.gating.clone(),
        )));
        runner.add_check(Box::new(CorrelationConfigCheck::new(
            self.config.pipeline.correlation.clone(),
        )));
        runner.add_check(Box::new(PipelineConstructCheck));

        info!("Running {} pre-flight checks...", runner.check_count(),);

        runner.run().await
    }

    // ────────────────────────────────────────────────────────────────────
    // Internal: Watchdog
    // ────────────────────────────────────────────────────────────────────

    async fn start_watchdog_internal(&mut self, pipeline: Arc<TradingPipeline>) -> Result<()> {
        info!("Starting CNS watchdog...");

        let watchdog_config = self.config.watchdog.to_watchdog_config();

        // Create the watchdog with a pipeline-linked kill switch if configured
        let kill_switch: Arc<dyn KillSwitch> = if self.config.wire_kill_switch {
            Arc::new(PipelineKillSwitch {
                pipeline: pipeline.clone(),
            })
        } else {
            Arc::new(NoOpKillSwitch)
        };

        let watchdog = CnsWatchdog::new(watchdog_config, kill_switch);

        // Get handle before moving watchdog
        let handle = watchdog.handle();

        // Register core components
        watchdog
            .register(ComponentRegistration {
                name: components::FORWARD_SERVICE.to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_millis(self.config.forward_heartbeat_ms),
                description: Some("Forward service main loop".to_string()),
            })
            .await;

        watchdog
            .register(ComponentRegistration {
                name: components::TRADING_PIPELINE.to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_millis(self.config.forward_heartbeat_ms),
                description: Some("Brain-inspired trading pipeline".to_string()),
            })
            .await;

        watchdog
            .register(ComponentRegistration {
                name: components::REGIME_DETECTOR.to_string(),
                criticality: ComponentCriticality::Important,
                heartbeat_interval: Duration::from_millis(self.config.forward_heartbeat_ms * 2),
                description: Some("Regime detection subsystem".to_string()),
            })
            .await;

        watchdog
            .register(ComponentRegistration {
                name: components::DATA_FEED.to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_millis(self.config.forward_heartbeat_ms),
                description: Some("WebSocket data feed".to_string()),
            })
            .await;

        watchdog
            .register(ComponentRegistration {
                name: components::RISK_MANAGER.to_string(),
                criticality: ComponentCriticality::Critical,
                heartbeat_interval: Duration::from_millis(self.config.forward_heartbeat_ms * 2),
                description: Some("Risk management subsystem".to_string()),
            })
            .await;

        info!(
            "Registered {} components with watchdog",
            watchdog.component_count().await,
        );

        // ── Spawn watchdog on a DEDICATED OS thread with its own Tokio runtime ──
        // This ensures the watchdog keeps ticking even if the main async runtime
        // is starved by CPU-intensive work (ViViT inference, indicator computation,
        // etc.). If the main runtime panics (with panic=abort), the whole process
        // terminates and the supervisor restarts it — but runtime *starvation*
        // (where tasks are alive but starved of poll cycles) is the real risk,
        // and this dedicated thread eliminates it.
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let thread_handle = std::thread::Builder::new()
            .name("cns-watchdog".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to build watchdog Tokio runtime");

                rt.block_on(async move {
                    tokio::select! {
                        _ = watchdog.run() => {
                            info!("🐕 Watchdog run() completed on dedicated thread");
                        }
                        _ = shutdown_rx => {
                            info!("🐕 Watchdog received shutdown signal on dedicated thread");
                        }
                    }
                });
            })
            .map_err(|e| anyhow::anyhow!("Failed to spawn watchdog thread: {}", e))?;

        self.watchdog_handle = Some(handle);
        self.watchdog_thread = Some(thread_handle);
        self.watchdog_shutdown = Some(shutdown_tx);

        info!("✅ CNS watchdog started on dedicated OS thread (independent of main runtime)");
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────
    // Runtime operations
    // ────────────────────────────────────────────────────────────────────

    /// Send a heartbeat for a named component.
    pub async fn heartbeat(&self, component: &str) {
        if let Some(handle) = &self.watchdog_handle {
            handle.heartbeat(component).await;
        }
    }

    /// Send heartbeats for the core forward service components.
    pub async fn heartbeat_core(&self) {
        if let Some(handle) = &self.watchdog_handle {
            handle.heartbeat(components::FORWARD_SERVICE).await;
            handle.heartbeat(components::TRADING_PIPELINE).await;
        }
    }

    /// Get a watchdog health summary.
    ///
    /// Returns a `WatchdogSnapshot`-like summary built from the handle's
    /// component state map. The full `WatchdogSnapshot` is only available
    /// on `CnsWatchdog::snapshot()`, but the handle gives us enough data
    /// to build a useful overview.
    pub async fn watchdog_snapshot(&self) -> Option<WatchdogSnapshot> {
        if let Some(handle) = &self.watchdog_handle {
            let states = handle.summary().await;
            let total = states.len();
            let alive = states
                .values()
                .filter(|s| **s == ComponentState::Alive)
                .count();
            let degraded = states
                .values()
                .filter(|s| **s == ComponentState::Degraded)
                .count();
            let dead = states
                .values()
                .filter(|s| **s == ComponentState::Dead)
                .count();

            Some(WatchdogSnapshot {
                uptime_secs: 0.0, // Not available from handle alone
                total_components: total,
                alive_count: alive,
                degraded_count: degraded,
                dead_count: dead,
                components: Vec::new(), // Detailed component info not available from handle
                timestamp: Utc::now(),
            })
        } else {
            None
        }
    }

    /// Check if the runtime is operational.
    pub fn is_running(&self) -> bool {
        self.state == RuntimeState::Running
    }

    /// Shutdown the brain runtime gracefully.
    pub async fn shutdown(&mut self) {
        if self.state == RuntimeState::Stopped || self.state == RuntimeState::ShuttingDown {
            return;
        }

        info!("Brain runtime shutting down...");
        self.state = RuntimeState::ShuttingDown;

        // Activate kill switch on the pipeline to stop all trading
        if let Some(pipeline) = &self.pipeline {
            pipeline.activate_kill_switch().await;
        }

        // Note: We don't directly stop the watchdog here because the
        // JoinHandle approach means we'd need the original CnsWatchdog
        // to call .stop(). In production, the watchdog is stopped via
        // its shutdown channel when the CnsWatchdog is dropped.

        // Abort the watchdog task if it's still running (legacy path)
        if let Some(task) = self.watchdog_task.take() {
            task.abort();
            let _ = task.await;
        }

        // Signal the dedicated watchdog thread to shut down
        if let Some(tx) = self.watchdog_shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(thread) = self.watchdog_thread.take() {
            let _ = thread.join();
        }

        self.state = RuntimeState::Stopped;
        info!("✅ Brain runtime stopped");
    }

    /// Generate a health report combining boot report, watchdog state,
    /// and pipeline metrics.
    pub async fn health_report(&self) -> BrainHealthReport {
        let watchdog_snapshot = self.watchdog_snapshot().await;
        let pipeline_metrics = if let Some(pipeline) = &self.pipeline {
            let m = pipeline.metrics_snapshot().await;
            Some(PipelineHealthMetrics {
                total_evaluations: m.total_evaluations,
                proceed_count: m.proceed_count,
                block_count: m.block_count,
                reduce_only_count: m.reduce_only_count,
                avg_evaluation_us: m.avg_evaluation_us(),
                block_rate_pct: m.block_rate_pct(),
                is_killed: pipeline.is_killed().await,
            })
        } else {
            None
        };

        BrainHealthReport {
            state: self.state,
            boot_passed: self.boot_report.as_ref().is_some_and(|r| r.is_boot_safe()),
            boot_summary: self.boot_report.as_ref().map(|r| r.summary()),
            watchdog: watchdog_snapshot,
            pipeline: pipeline_metrics,
            timestamp: Utc::now(),
        }
    }
}

// ============================================================================
// Pipeline Kill Switch (bridges watchdog → pipeline)
// ============================================================================

/// A `KillSwitch` implementation that activates the trading pipeline's
/// kill switch when triggered by the watchdog.
struct PipelineKillSwitch {
    pipeline: Arc<TradingPipeline>,
}

#[async_trait::async_trait]
impl KillSwitch for PipelineKillSwitch {
    async fn trigger_kill(&self, reason: &str) -> Result<(), String> {
        warn!("🛑 Watchdog triggered kill switch on pipeline: {}", reason,);
        self.pipeline.activate_kill_switch().await;
        Ok(())
    }

    async fn is_killed(&self) -> bool {
        self.pipeline.is_killed().await
    }
}

// ============================================================================
// Health Report
// ============================================================================

/// Combined health report from all brain subsystems.
#[derive(Debug, Clone)]
pub struct BrainHealthReport {
    /// Current runtime state.
    pub state: RuntimeState,
    /// Whether the boot pre-flight checks passed.
    pub boot_passed: bool,
    /// Summary of the boot report.
    pub boot_summary: Option<String>,
    /// Watchdog snapshot (if running).
    pub watchdog: Option<WatchdogSnapshot>,
    /// Pipeline metrics (if initialized).
    pub pipeline: Option<PipelineHealthMetrics>,
    /// When this report was generated.
    pub timestamp: chrono::DateTime<Utc>,
}

impl BrainHealthReport {
    /// Overall health: runtime is running, boot passed, watchdog operational,
    /// pipeline not killed.
    pub fn is_healthy(&self) -> bool {
        self.state == RuntimeState::Running
            && self.boot_passed
            && self.watchdog.as_ref().is_none_or(|w| w.is_operational())
            && self.pipeline.as_ref().is_none_or(|p| !p.is_killed)
    }

    /// Format as a human-readable string.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Brain Runtime Health Report".to_string());
        lines.push("──────────────────────────────────".to_string());
        lines.push(format!("State:        {}", self.state));
        lines.push(format!("Boot Passed:  {}", self.boot_passed));
        lines.push(format!("Healthy:      {}", self.is_healthy()));
        lines.push(format!("Timestamp:    {}", self.timestamp));

        if let Some(ref summary) = self.boot_summary {
            lines.push(format!("Boot Summary: {}", summary));
        }

        if let Some(ref w) = self.watchdog {
            lines.push(format!(
                "Watchdog:     {} components ({} alive, {} degraded, {} dead)",
                w.total_components, w.alive_count, w.degraded_count, w.dead_count
            ));
            lines.push(format!("Health Score: {:.1}%", w.health_score()));
        }

        if let Some(ref p) = self.pipeline {
            lines.push(format!(
                "Pipeline:     {} evals ({} proceed, {} blocked, {} reduce-only)",
                p.total_evaluations, p.proceed_count, p.block_count, p.reduce_only_count
            ));
            lines.push(format!("Block Rate:   {:.1}%", p.block_rate_pct));
            lines.push(format!("Avg Eval:     {:.0}µs", p.avg_evaluation_us));
            lines.push(format!("Killed:       {}", p.is_killed));
        }

        lines.join("\n")
    }
}

/// Pipeline-specific health metrics.
#[derive(Debug, Clone)]
pub struct PipelineHealthMetrics {
    pub total_evaluations: u64,
    pub proceed_count: u64,
    pub block_count: u64,
    pub reduce_only_count: u64,
    pub avg_evaluation_us: f64,
    pub block_rate_pct: f64,
    pub is_killed: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain_wiring::make_test_signal;
    use janus_regime::MarketRegime;

    // ── Config tests ───────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let config = BrainRuntimeConfig::default();
        assert!(config.enforce_preflight);
        assert!(config.auto_start_watchdog);
        assert!(config.wire_kill_switch);
        assert_eq!(config.forward_heartbeat_ms, 1000);
    }

    #[test]
    fn test_watchdog_config_conversion() {
        let wrc = WatchdogRuntimeConfig {
            check_interval_ms: 250,
            degraded_threshold: 2,
            dead_threshold: 4,
            kill_on_critical_death: false,
        };
        let wc = wrc.to_watchdog_config();
        assert_eq!(wc.check_interval_ms, 250);
        assert_eq!(wc.degraded_threshold, 2);
        assert_eq!(wc.dead_threshold, 4);
        assert!(!wc.kill_on_critical_death);
    }

    #[test]
    fn test_preflight_config_conversion() {
        let prc = PreflightRuntimeConfig {
            abort_on_critical: false,
            parallel_within_phase: false,
            global_timeout_secs: 30,
            enabled_phases: Vec::new(),
            skip_infra_checks: true,
        };
        let pc = prc.to_preflight_config();
        assert!(!pc.abort_on_critical);
        assert!(!pc.parallel_within_phase);
        assert_eq!(pc.global_timeout, Duration::from_secs(30));
    }

    // ── Runtime state tests ────────────────────────────────────────────

    #[test]
    fn test_runtime_state_display() {
        assert_eq!(format!("{}", RuntimeState::Uninitialized), "Uninitialized");
        assert_eq!(format!("{}", RuntimeState::Booting), "Booting");
        assert_eq!(format!("{}", RuntimeState::Starting), "Starting");
        assert_eq!(format!("{}", RuntimeState::Running), "Running");
        assert_eq!(format!("{}", RuntimeState::ShuttingDown), "ShuttingDown");
        assert_eq!(format!("{}", RuntimeState::Stopped), "Stopped");
        assert_eq!(format!("{}", RuntimeState::Failed), "Failed");
    }

    // ── Pre-flight check tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_pipeline_construct_check_passes() {
        let check = PipelineConstructCheck;
        let result = check.run().await;
        assert!(result.outcome.is_pass());
        assert_eq!(result.phase, BootPhase::Strategy);
        assert_eq!(result.criticality, Criticality::Required);
    }

    #[tokio::test]
    async fn test_gating_config_check_passes_with_valid_config() {
        let config = StrategyGatingConfig::default();
        let check = GatingConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_pass());
    }

    #[tokio::test]
    async fn test_gating_config_check_fails_with_invalid_weight() {
        let config = StrategyGatingConfig {
            min_weight: -0.5,
            ..Default::default()
        };
        let check = GatingConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    #[tokio::test]
    async fn test_gating_config_check_fails_with_weight_above_1() {
        let config = StrategyGatingConfig {
            min_weight: 1.5,
            ..Default::default()
        };
        let check = GatingConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    #[tokio::test]
    async fn test_correlation_config_check_passes_with_defaults() {
        let config = CorrelationConfig::default();
        let check = CorrelationConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_pass());
    }

    #[tokio::test]
    async fn test_correlation_config_check_fails_zero_window() {
        let config = CorrelationConfig {
            window: 0,
            ..Default::default()
        };
        let check = CorrelationConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    #[tokio::test]
    async fn test_correlation_config_check_fails_bad_threshold() {
        let config = CorrelationConfig {
            correlation_threshold: 0.0,
            ..Default::default()
        };
        let check = CorrelationConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    #[tokio::test]
    async fn test_correlation_config_check_fails_zero_max_positions() {
        let config = CorrelationConfig {
            max_correlated_positions: 0,
            ..Default::default()
        };
        let check = CorrelationConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    #[tokio::test]
    async fn test_pipeline_config_check_passes_with_defaults() {
        let config = TradingPipelineConfig::default();
        let check = PipelineConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_pass());
    }

    #[tokio::test]
    async fn test_pipeline_config_check_fails_bad_max_scale() {
        let config = TradingPipelineConfig {
            max_position_scale: 0.0,
            ..Default::default()
        };
        let check = PipelineConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    #[tokio::test]
    async fn test_pipeline_config_check_fails_inverted_scales() {
        let config = TradingPipelineConfig {
            min_position_scale: 2.0,
            max_position_scale: 1.0,
            ..Default::default()
        };
        let check = PipelineConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    #[tokio::test]
    async fn test_pipeline_config_check_fails_bad_risk_factor() {
        let config = TradingPipelineConfig {
            high_risk_scale_factor: 1.5,
            ..Default::default()
        };
        let check = PipelineConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    #[tokio::test]
    async fn test_pipeline_config_check_fails_bad_confidence() {
        let config = TradingPipelineConfig {
            min_regime_confidence: -0.1,
            ..Default::default()
        };
        let check = PipelineConfigCheck::new(config);
        let result = check.run().await;
        assert!(result.outcome.is_fail());
    }

    // ── Runtime boot tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_runtime_new_is_uninitialized() {
        let runtime = BrainRuntime::new(BrainRuntimeConfig::default());
        assert_eq!(runtime.state(), RuntimeState::Uninitialized);
        assert!(runtime.pipeline().is_none());
        assert!(runtime.boot_report().is_none());
        assert!(runtime.watchdog_handle().is_none());
    }

    #[tokio::test]
    async fn test_runtime_boot_succeeds_with_defaults() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: false, // Skip watchdog for unit test
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);

        let report = runtime.boot().await.expect("Boot should succeed");
        assert!(report.is_boot_safe());
        assert_eq!(runtime.state(), RuntimeState::Running);
        assert!(runtime.pipeline().is_some());
        assert!(runtime.is_running());
    }

    #[tokio::test]
    async fn test_runtime_boot_fails_with_bad_config() {
        let config = BrainRuntimeConfig {
            pipeline: TradingPipelineConfig {
                max_position_scale: 0.0, // Invalid
                ..Default::default()
            },
            enforce_preflight: true,
            auto_start_watchdog: false,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);

        let result = runtime.boot().await;
        assert!(result.is_err());
        assert_eq!(runtime.state(), RuntimeState::Failed);
    }

    #[tokio::test]
    async fn test_runtime_boot_proceeds_with_bad_config_when_not_enforced() {
        let config = BrainRuntimeConfig {
            pipeline: TradingPipelineConfig {
                max_position_scale: 0.0, // Invalid
                ..Default::default()
            },
            enforce_preflight: false, // Don't enforce
            auto_start_watchdog: false,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);

        let report = runtime
            .boot()
            .await
            .expect("Boot should proceed despite failures");
        assert!(!report.is_boot_safe());
        assert_eq!(runtime.state(), RuntimeState::Running);
    }

    #[tokio::test]
    async fn test_runtime_preflight_only() {
        let config = BrainRuntimeConfig::default();
        let mut runtime = BrainRuntime::new(config);

        let report = runtime.preflight_only().await;
        assert!(report.is_boot_safe());
        assert!(runtime.pipeline().is_none()); // Pipeline should NOT be initialized
    }

    #[tokio::test]
    async fn test_runtime_boot_with_pipeline() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: false,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);

        let pipeline = Arc::new(TradingPipeline::new(TradingPipelineConfig::default()));
        let report = runtime
            .boot_with_pipeline(pipeline.clone())
            .await
            .expect("Boot should succeed");

        assert!(report.is_boot_safe());
        assert!(runtime.pipeline().is_some());
    }

    #[tokio::test]
    async fn test_runtime_shutdown() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: false,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);
        runtime.boot().await.expect("Boot should succeed");

        assert!(runtime.is_running());

        runtime.shutdown().await;
        assert_eq!(runtime.state(), RuntimeState::Stopped);

        // Pipeline kill switch should be active after shutdown
        if let Some(pipeline) = runtime.pipeline() {
            assert!(pipeline.is_killed().await);
        }
    }

    #[tokio::test]
    async fn test_runtime_double_shutdown_is_safe() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: false,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);
        runtime.boot().await.expect("Boot should succeed");

        runtime.shutdown().await;
        runtime.shutdown().await; // Should not panic
        assert_eq!(runtime.state(), RuntimeState::Stopped);
    }

    // ── Health report tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_report_before_boot() {
        let runtime = BrainRuntime::new(BrainRuntimeConfig::default());
        let report = runtime.health_report().await;

        assert!(!report.is_healthy());
        assert_eq!(report.state, RuntimeState::Uninitialized);
        assert!(!report.boot_passed);
        assert!(report.watchdog.is_none());
        assert!(report.pipeline.is_none());
    }

    #[tokio::test]
    async fn test_health_report_after_boot() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: false,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);
        runtime.boot().await.expect("Boot should succeed");

        let report = runtime.health_report().await;
        assert!(report.is_healthy());
        assert_eq!(report.state, RuntimeState::Running);
        assert!(report.boot_passed);

        let summary = report.summary();
        assert!(summary.contains("Running"));
        assert!(summary.contains("Healthy"));
    }

    #[tokio::test]
    async fn test_health_report_after_pipeline_evaluation() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: false,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);
        runtime.boot().await.expect("Boot should succeed");

        let pipeline = runtime.pipeline().unwrap().clone();
        let signal = make_test_signal(
            MarketRegime::Trending(janus_regime::TrendDirection::Bullish),
            0.85,
        );
        pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;

        let report = runtime.health_report().await;
        let metrics = report.pipeline.expect("Pipeline metrics should be present");
        assert_eq!(metrics.total_evaluations, 1);
        assert_eq!(metrics.proceed_count, 1);
    }

    #[tokio::test]
    async fn test_health_report_after_kill() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: false,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);
        runtime.boot().await.expect("Boot should succeed");

        let pipeline = runtime.pipeline().unwrap().clone();
        pipeline.activate_kill_switch().await;

        let report = runtime.health_report().await;
        assert!(!report.is_healthy()); // Kill switch active → not healthy
        let metrics = report.pipeline.expect("Pipeline metrics should be present");
        assert!(metrics.is_killed);
    }

    // ── Boot with watchdog tests ───────────────────────────────────────

    #[tokio::test]
    async fn test_runtime_boot_with_watchdog() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: true,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);
        runtime.boot().await.expect("Boot should succeed");

        assert!(runtime.watchdog_handle().is_some());

        // Give the watchdog a moment to start
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Send some heartbeats
        runtime.heartbeat(components::FORWARD_SERVICE).await;
        runtime.heartbeat(components::TRADING_PIPELINE).await;

        // Check watchdog snapshot
        let snapshot = runtime.watchdog_snapshot().await;
        assert!(snapshot.is_some());
        let snapshot = snapshot.unwrap();
        assert!(snapshot.total_components >= 4);

        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn test_runtime_heartbeat_core() {
        let config = BrainRuntimeConfig {
            auto_start_watchdog: true,
            ..Default::default()
        };
        let mut runtime = BrainRuntime::new(config);
        runtime.boot().await.expect("Boot should succeed");

        // heartbeat_core should not panic
        runtime.heartbeat_core().await;

        runtime.shutdown().await;
    }

    // ── Component names test ───────────────────────────────────────────

    #[test]
    fn test_component_names_are_unique() {
        let names = vec![
            components::FORWARD_SERVICE,
            components::REGIME_DETECTOR,
            components::TRADING_PIPELINE,
            components::DATA_FEED,
            components::EXECUTION_CLIENT,
            components::RISK_MANAGER,
            components::STRATEGY_ENGINE,
            components::CORRELATION_TRACKER,
        ];
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(names.len(), unique.len(), "Component names must be unique",);
    }

    // ── Pipeline kill switch bridge test ────────────────────────────────

    #[tokio::test]
    async fn test_pipeline_kill_switch_bridge() {
        let pipeline = Arc::new(TradingPipeline::new(TradingPipelineConfig::default()));
        let kill_switch = PipelineKillSwitch {
            pipeline: pipeline.clone(),
        };

        assert!(!kill_switch.is_killed().await);

        let _ = kill_switch.trigger_kill("test reason").await;

        assert!(kill_switch.is_killed().await);
        assert!(pipeline.is_killed().await);
    }
}
