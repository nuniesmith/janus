//! # Brain Health REST API
//!
//! Exposes the brain-inspired trading pipeline's health status via REST.
//!
//! ## Endpoints
//!
//! - `GET /api/v1/brain/health` — Full brain health report (runtime state,
//!   boot status, watchdog snapshot, pipeline metrics, kill switch status).
//! - `GET /api/v1/brain/pipeline` — Pipeline-only metrics snapshot.
//! - `POST /api/v1/brain/kill-switch/activate` — Activate the pipeline kill switch. **🔒 Protected**
//! - `POST /api/v1/brain/kill-switch/deactivate` — Deactivate the pipeline kill switch. **🔒 Protected**
//! - `GET /api/v1/brain/affinity` — Export current strategy affinity state as JSON.
//! - `POST /api/v1/brain/affinity/record` — Record a closed-trade outcome into the affinity tracker.
//! - `POST /api/v1/brain/affinity/reset` — Reset strategy affinity tracker to empty state. **🔒 Protected**
//!
//! ## Authentication
//!
//! Destructive endpoints (kill-switch activate/deactivate, affinity reset) are
//! protected by bearer-token authentication when `BRAIN_API_TOKEN` is set.
//! Read-only endpoints (health, pipeline, affinity export) remain public.
//!
//! If `BRAIN_API_TOKEN` is **not** set, all endpoints are open (suitable for
//! development). A warning is logged at startup when the token is absent.
//!
//! ## Architecture
//!
//! The endpoints read from a shared `BrainHealthState` that holds `Arc`
//! references to the pipeline and watchdog handle. The state is populated
//! during boot and injected into the Axum router.
//!
//! ```text
//! ┌───────────────┐       ┌──────────────────┐
//! │  GET /brain/  │──────▶│ BrainHealthState  │
//! │    health     │       │  ├─ pipeline      │
//! └───────────────┘       │  ├─ watchdog      │
//!                         │  ├─ boot_report   │
//!                         │  └─ runtime_state │
//!                         └──────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

use janus_strategies::affinity::StrategyAffinityTracker;

use crate::brain_runtime::{BrainHealthReport, PipelineHealthMetrics, RuntimeState};
use crate::brain_wiring::TradingPipeline;
use janus_cns::watchdog::{ComponentState, WatchdogHandle, WatchdogSnapshot};

// ════════════════════════════════════════════════════════════════════
// Shared State
// ════════════════════════════════════════════════════════════════════

/// Shared state for the brain health REST API.
///
/// Populated during boot from `BrainRuntime` components and injected
/// into the Axum router via `.with_state()`.
#[derive(Clone)]
pub struct BrainHealthState {
    /// The trading pipeline (if brain runtime is enabled).
    pipeline: Option<Arc<TradingPipeline>>,
    /// The watchdog handle (if watchdog was started).
    watchdog_handle: Option<WatchdogHandle>,
    /// Whether the boot pre-flight checks passed.
    boot_passed: Arc<RwLock<bool>>,
    /// Summary of the boot report.
    boot_summary: Arc<RwLock<Option<String>>>,
    /// Current runtime state (updated by the runtime owner).
    runtime_state: Arc<RwLock<RuntimeState>>,
}

impl BrainHealthState {
    /// Create a new state with the given components.
    pub fn new(
        pipeline: Option<Arc<TradingPipeline>>,
        watchdog_handle: Option<WatchdogHandle>,
        boot_passed: bool,
        boot_summary: Option<String>,
        runtime_state: RuntimeState,
    ) -> Self {
        Self {
            pipeline,
            watchdog_handle,
            boot_passed: Arc::new(RwLock::new(boot_passed)),
            boot_summary: Arc::new(RwLock::new(boot_summary)),
            runtime_state: Arc::new(RwLock::new(runtime_state)),
        }
    }

    /// Create a disabled/empty state (when brain runtime is not enabled).
    pub fn disabled() -> Self {
        Self {
            pipeline: None,
            watchdog_handle: None,
            boot_passed: Arc::new(RwLock::new(false)),
            boot_summary: Arc::new(RwLock::new(Some("Brain runtime disabled".to_string()))),
            runtime_state: Arc::new(RwLock::new(RuntimeState::Uninitialized)),
        }
    }

    /// Update the runtime state (called by the runtime owner).
    pub async fn set_runtime_state(&self, state: RuntimeState) {
        *self.runtime_state.write().await = state;
    }

    /// Update the boot status after boot completes.
    pub async fn set_boot_result(&self, passed: bool, summary: Option<String>) {
        *self.boot_passed.write().await = passed;
        *self.boot_summary.write().await = summary;
    }

    /// Generate a full health report from the current state.
    pub async fn health_report(&self) -> BrainHealthReport {
        let state = *self.runtime_state.read().await;
        let boot_passed = *self.boot_passed.read().await;
        let boot_summary = self.boot_summary.read().await.clone();

        let watchdog_snapshot = if let Some(ref wh) = self.watchdog_handle {
            // WatchdogHandle doesn't have snapshot() — build one from summary()
            let states: HashMap<String, ComponentState> = wh.summary().await;
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
                uptime_secs: 0.0,
                total_components: total,
                alive_count: alive,
                degraded_count: degraded,
                dead_count: dead,
                components: Vec::new(),
                timestamp: Utc::now(),
            })
        } else {
            None
        };

        let pipeline_metrics = if let Some(ref pipeline) = self.pipeline {
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
            state,
            boot_passed,
            boot_summary,
            watchdog: watchdog_snapshot,
            pipeline: pipeline_metrics,
            timestamp: Utc::now(),
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Response DTOs (Serializable)
// ════════════════════════════════════════════════════════════════════

/// JSON-serializable brain health response.
#[derive(Debug, Serialize, Deserialize)]
pub struct BrainHealthResponse {
    /// Whether the brain runtime is healthy overall.
    pub healthy: bool,
    /// Current runtime state.
    pub state: String,
    /// Whether boot pre-flight checks passed.
    pub boot_passed: bool,
    /// Boot report summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boot_summary: Option<String>,
    /// Watchdog health snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub watchdog: Option<WatchdogHealthDto>,
    /// Pipeline metrics snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<PipelineHealthDto>,
    /// ISO 8601 timestamp.
    pub timestamp: String,
}

/// Watchdog health DTO.
#[derive(Debug, Serialize, Deserialize)]
pub struct WatchdogHealthDto {
    pub uptime_secs: f64,
    pub total_components: usize,
    pub alive_count: usize,
    pub degraded_count: usize,
    pub dead_count: usize,
    pub health_score: f64,
    pub is_operational: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<ComponentHealthDto>,
}

/// Individual component health DTO.
#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentHealthDto {
    pub name: String,
    pub state: String,
    pub criticality: String,
    pub missed_heartbeats: u32,
    pub last_heartbeat_ago_secs: f64,
}

/// Pipeline health DTO.
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineHealthDto {
    pub total_evaluations: u64,
    pub proceed_count: u64,
    pub block_count: u64,
    pub reduce_only_count: u64,
    pub avg_evaluation_us: f64,
    pub block_rate_pct: f64,
    pub is_killed: bool,
}

/// Kill switch action response.
#[derive(Debug, Serialize, Deserialize)]
pub struct KillSwitchResponse {
    pub success: bool,
    pub is_killed: bool,
    pub message: String,
    pub timestamp: String,
}

/// Response DTO for the affinity export endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct AffinityExportResponse {
    /// Whether a pipeline with an affinity tracker is available.
    pub available: bool,
    /// Number of strategy-asset pairs tracked.
    pub pair_count: usize,
    /// The full affinity state serialised as a JSON value.
    /// `null` when the pipeline is not available.
    pub state: Option<serde_json::Value>,
    pub timestamp: String,
}

/// Response DTO for the affinity reset endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct AffinityResetResponse {
    pub success: bool,
    /// Number of strategy-asset pairs that were cleared.
    pub cleared_pairs: usize,
    pub message: String,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BrainApiError {
    pub error: String,
    pub message: String,
}

// ════════════════════════════════════════════════════════════════════
// Conversions
// ════════════════════════════════════════════════════════════════════

impl From<BrainHealthReport> for BrainHealthResponse {
    fn from(report: BrainHealthReport) -> Self {
        Self {
            healthy: report.is_healthy(),
            state: report.state.to_string(),
            boot_passed: report.boot_passed,
            boot_summary: report.boot_summary,
            watchdog: report.watchdog.map(WatchdogHealthDto::from),
            pipeline: report.pipeline.map(PipelineHealthDto::from),
            timestamp: report.timestamp.to_rfc3339(),
        }
    }
}

impl From<WatchdogSnapshot> for WatchdogHealthDto {
    fn from(snap: WatchdogSnapshot) -> Self {
        Self {
            uptime_secs: snap.uptime_secs,
            total_components: snap.total_components,
            alive_count: snap.alive_count,
            degraded_count: snap.degraded_count,
            dead_count: snap.dead_count,
            health_score: snap.health_score(),
            is_operational: snap.is_operational(),
            components: snap
                .components
                .into_iter()
                .map(|c| ComponentHealthDto {
                    name: c.name.clone(),
                    state: format!("{:?}", c.state),
                    criticality: format!("{:?}", c.criticality),
                    missed_heartbeats: c.missed_heartbeats,
                    last_heartbeat_ago_secs: c
                        .last_heartbeat_utc
                        .map(|t| (Utc::now() - t).num_milliseconds() as f64 / 1000.0)
                        .unwrap_or(-1.0),
                })
                .collect(),
        }
    }
}

impl From<PipelineHealthMetrics> for PipelineHealthDto {
    fn from(m: PipelineHealthMetrics) -> Self {
        Self {
            total_evaluations: m.total_evaluations,
            proceed_count: m.proceed_count,
            block_count: m.block_count,
            reduce_only_count: m.reduce_only_count,
            avg_evaluation_us: m.avg_evaluation_us,
            block_rate_pct: m.block_rate_pct,
            is_killed: m.is_killed,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Router
// ════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════
// Auth Middleware
// ════════════════════════════════════════════════════════════════════

/// Load the bearer token from `BRAIN_API_TOKEN` environment variable.
///
/// Returns `None` when the variable is unset or empty — in that case
/// all endpoints are unprotected and a warning is logged.
fn load_api_token() -> Option<String> {
    match std::env::var("BRAIN_API_TOKEN") {
        Ok(token) if !token.is_empty() => {
            info!(
                "🔒 Brain REST API: destructive endpoints protected by BRAIN_API_TOKEN ({} chars)",
                token.len()
            );
            Some(token)
        }
        _ => {
            warn!(
                "⚠️  BRAIN_API_TOKEN not set — kill-switch and affinity-reset endpoints are \
                 UNPROTECTED. Set BRAIN_API_TOKEN for production use."
            );
            None
        }
    }
}

/// Axum middleware that checks the `Authorization: Bearer <token>` header
/// against the configured `BRAIN_API_TOKEN`.
///
/// When no token is configured (dev mode), all requests pass through.
async fn require_brain_token(
    State(expected): State<Option<String>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    // No token configured → pass through (dev mode)
    let expected_token = match expected {
        Some(ref t) => t,
        None => return next.run(request).await,
    };

    // Extract the Authorization header
    let auth_header = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let provided = &value[7..];
            if provided == expected_token {
                next.run(request).await
            } else {
                warn!("🔒 Brain REST API: rejected request — invalid bearer token");
                (
                    StatusCode::FORBIDDEN,
                    Json(BrainApiError {
                        error: "forbidden".to_string(),
                        message: "Invalid bearer token".to_string(),
                    }),
                )
                    .into_response()
            }
        }
        Some(_) => {
            // Has Authorization header but not Bearer scheme
            (
                StatusCode::UNAUTHORIZED,
                Json(BrainApiError {
                    error: "unauthorized".to_string(),
                    message: "Expected Authorization: Bearer <token>".to_string(),
                }),
            )
                .into_response()
        }
        None => {
            // No Authorization header at all
            (
                StatusCode::UNAUTHORIZED,
                Json(BrainApiError {
                    error: "unauthorized".to_string(),
                    message:
                        "Missing Authorization header. Use: Authorization: Bearer <BRAIN_API_TOKEN>"
                            .to_string(),
                }),
            )
                .into_response()
        }
    }
}

/// Build the brain health Axum router.
///
/// Read-only endpoints (health, pipeline, affinity export) are public.
/// Destructive endpoints (kill-switch, affinity reset) are protected by
/// bearer-token auth when `BRAIN_API_TOKEN` is set.
///
/// Mount this into the main server router:
///
/// ```rust,ignore
/// let brain_router = brain_rest::router(brain_state);
/// let app = Router::new().merge(brain_router);
/// ```
pub fn router(state: Arc<BrainHealthState>) -> Router {
    let api_token = load_api_token();
    router_with_token(state, api_token)
}

/// Build the brain health Axum router with an explicitly provided API token.
///
/// This avoids reading `BRAIN_API_TOKEN` from the environment, which makes
/// it safe to call from concurrent tests without env-var races.
pub fn router_with_token(state: Arc<BrainHealthState>, api_token: Option<String>) -> Router {
    // Public read-only endpoints
    let public_routes = Router::new()
        .route("/api/v1/brain/health", get(brain_health_handler))
        .route("/api/v1/brain/pipeline", get(brain_pipeline_handler))
        .route("/api/v1/brain/affinity", get(affinity_export_handler))
        .route(
            "/api/v1/brain/affinity/record",
            post(affinity_record_handler),
        )
        .with_state(state.clone());

    // Protected destructive endpoints
    let protected_routes = Router::new()
        .route(
            "/api/v1/brain/kill-switch/activate",
            post(kill_switch_activate_handler),
        )
        .route(
            "/api/v1/brain/kill-switch/deactivate",
            post(kill_switch_deactivate_handler),
        )
        .route("/api/v1/brain/affinity/reset", post(affinity_reset_handler))
        .route_layer(middleware::from_fn_with_state(
            api_token,
            require_brain_token,
        ))
        .with_state(state);

    public_routes.merge(protected_routes)
}

// ════════════════════════════════════════════════════════════════════
// Handlers
// ════════════════════════════════════════════════════════════════════

/// `GET /api/v1/brain/health` — Full brain health report.
async fn brain_health_handler(
    State(state): State<Arc<BrainHealthState>>,
) -> Json<BrainHealthResponse> {
    let report = state.health_report().await;
    Json(BrainHealthResponse::from(report))
}

/// `GET /api/v1/brain/pipeline` — Pipeline-only metrics.
async fn brain_pipeline_handler(
    State(state): State<Arc<BrainHealthState>>,
) -> Result<Json<PipelineHealthDto>, Response> {
    let pipeline = state.pipeline.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(BrainApiError {
                error: "pipeline_unavailable".to_string(),
                message: "Brain pipeline is not initialized".to_string(),
            }),
        )
            .into_response()
    })?;

    let m = pipeline.metrics_snapshot().await;
    let dto = PipelineHealthDto {
        total_evaluations: m.total_evaluations,
        proceed_count: m.proceed_count,
        block_count: m.block_count,
        reduce_only_count: m.reduce_only_count,
        avg_evaluation_us: m.avg_evaluation_us(),
        block_rate_pct: m.block_rate_pct(),
        is_killed: pipeline.is_killed().await,
    };

    Ok(Json(dto))
}

/// `POST /api/v1/brain/kill-switch/activate` — Activate the kill switch.
async fn kill_switch_activate_handler(
    State(state): State<Arc<BrainHealthState>>,
) -> Result<Json<KillSwitchResponse>, Response> {
    let pipeline = state.pipeline.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(BrainApiError {
                error: "pipeline_unavailable".to_string(),
                message: "Brain pipeline is not initialized".to_string(),
            }),
        )
            .into_response()
    })?;

    pipeline.activate_kill_switch().await;

    warn!("🛑 Kill switch ACTIVATED via REST API");

    Ok(Json(KillSwitchResponse {
        success: true,
        is_killed: true,
        message: "Kill switch activated — all trading is halted".to_string(),
        timestamp: Utc::now().to_rfc3339(),
    }))
}

/// `POST /api/v1/brain/kill-switch/deactivate` — Deactivate the kill switch.
async fn kill_switch_deactivate_handler(
    State(state): State<Arc<BrainHealthState>>,
) -> Result<Json<KillSwitchResponse>, Response> {
    let pipeline = state.pipeline.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(BrainApiError {
                error: "pipeline_unavailable".to_string(),
                message: "Brain pipeline is not initialized".to_string(),
            }),
        )
            .into_response()
    })?;

    pipeline.deactivate_kill_switch().await;

    info!("🟢 Kill switch DEACTIVATED via REST API — trading resumed");

    Ok(Json(KillSwitchResponse {
        success: true,
        is_killed: false,
        message: "Kill switch deactivated — trading resumed".to_string(),
        timestamp: Utc::now().to_rfc3339(),
    }))
}

/// `GET /api/v1/brain/affinity` — Export the current strategy affinity state.
async fn affinity_export_handler(
    State(state): State<Arc<BrainHealthState>>,
) -> Json<AffinityExportResponse> {
    let pipeline = match state.pipeline.as_ref() {
        Some(p) => p,
        None => {
            return Json(AffinityExportResponse {
                available: false,
                pair_count: 0,
                state: None,
                timestamp: Utc::now().to_rfc3339(),
            });
        }
    };

    let gate = pipeline.strategy_gate().await;
    let tracker = gate.tracker();

    // Serialize the tracker state to a serde_json::Value for the response
    let state_json = match tracker.save_state() {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        Err(_) => serde_json::Value::Null,
    };

    let pair_count = tracker.pair_count();

    Json(AffinityExportResponse {
        available: true,
        pair_count,
        state: Some(state_json),
        timestamp: Utc::now().to_rfc3339(),
    })
}

// ── AffinityRecord ────────────────────────────────────────────────────────

/// Request body for `POST /api/v1/brain/affinity/record`.
#[derive(Debug, Deserialize)]
pub struct AffinityRecordRequest {
    /// Strategy name (e.g. "EMAFlip", "MomentumSurge").
    pub strategy: String,
    /// Asset symbol (e.g. "BTCUSD", "MES").
    pub asset: String,
    /// Realised P&L for this trade (positive = profit).
    pub pnl: f64,
    /// Whether this trade was a winner.
    pub is_winner: bool,
    /// Optional risk-reward ratio achieved (actual, not planned).
    pub rr_ratio: Option<f64>,
}

/// Response for `POST /api/v1/brain/affinity/record`.
#[derive(Debug, Serialize)]
pub struct AffinityRecordResponse {
    pub ok: bool,
    pub strategy: String,
    pub asset: String,
    pub timestamp: String,
}

/// `POST /api/v1/brain/affinity/record` — Record a closed-trade outcome.
///
/// Called by the Python `MemoryRecorder` after each trade closes. Feeds
/// real P&L data into the `StrategyAffinityTracker` so Janus learns which
/// strategy × asset combinations are performing well or poorly over time.
///
/// This endpoint is on the **public** router (no auth required) because it
/// is write-only, additive, and called only on the internal Docker network.
/// Use `POST /api/v1/brain/affinity/reset` (protected) to wipe all data.
async fn affinity_record_handler(
    State(state): State<Arc<BrainHealthState>>,
    Json(req): Json<AffinityRecordRequest>,
) -> Result<Json<AffinityRecordResponse>, Response> {
    let pipeline = state.pipeline.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(BrainApiError {
                error: "pipeline_unavailable".to_string(),
                message: "Brain pipeline is not initialized".to_string(),
            }),
        )
            .into_response()
    })?;

    {
        let mut gate = pipeline.strategy_gate_mut().await;
        gate.tracker_mut().record_trade_result_with_rr(
            &req.strategy,
            &req.asset,
            req.pnl,
            req.is_winner,
            req.rr_ratio,
        );
    }

    info!(
        strategy = %req.strategy,
        asset = %req.asset,
        pnl = req.pnl,
        is_winner = req.is_winner,
        "Affinity record updated via REST API"
    );

    Ok(Json(AffinityRecordResponse {
        ok: true,
        strategy: req.strategy,
        asset: req.asset,
        timestamp: Utc::now().to_rfc3339(),
    }))
}

/// `POST /api/v1/brain/affinity/reset` — Reset the strategy affinity tracker.
///
/// This clears all recorded strategy-asset performance data, resetting
/// affinity weights to their defaults. Use with caution in production —
/// the tracker will need to re-learn strategy affinities from scratch.
async fn affinity_reset_handler(
    State(state): State<Arc<BrainHealthState>>,
) -> Result<Json<AffinityResetResponse>, Response> {
    let pipeline = state.pipeline.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(BrainApiError {
                error: "pipeline_unavailable".to_string(),
                message: "Brain pipeline is not initialized".to_string(),
            }),
        )
            .into_response()
    })?;

    let mut gate = pipeline.strategy_gate_mut().await;
    let old_tracker = gate.tracker();
    let cleared_pairs = old_tracker.pair_count();

    // Get the min_trades config from the existing tracker before replacing
    // We need to preserve the min_trades setting
    let min_trades = old_tracker.min_trades_for_confidence;

    // Replace with a fresh tracker
    let fresh_tracker = StrategyAffinityTracker::new(min_trades);
    gate.set_tracker(fresh_tracker);

    warn!(
        "🔄 Affinity tracker RESET via REST API — cleared {} strategy-asset pairs",
        cleared_pairs
    );

    Ok(Json(AffinityResetResponse {
        success: true,
        cleared_pairs,
        message: format!(
            "Affinity tracker reset — {} strategy-asset pairs cleared",
            cleared_pairs
        ),
        timestamp: Utc::now().to_rfc3339(),
    }))
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain_wiring::{TradingPipeline, TradingPipelineConfig};
    use std::time::Duration;

    fn make_pipeline() -> Arc<TradingPipeline> {
        Arc::new(TradingPipeline::new(TradingPipelineConfig::default()))
    }

    fn make_state(pipeline: Option<Arc<TradingPipeline>>) -> Arc<BrainHealthState> {
        Arc::new(BrainHealthState::new(
            pipeline,
            None,
            true,
            Some("All checks passed".to_string()),
            RuntimeState::Running,
        ))
    }

    /// Spin up the brain REST router on a random port.
    /// Returns the base URL (e.g. `http://127.0.0.1:12345`).
    async fn spawn_server(state: Arc<BrainHealthState>) -> String {
        // Use router_with_token(None) to avoid env-var races between parallel tests
        let app = router_with_token(state, None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        tokio::time::sleep(Duration::from_millis(30)).await;

        format!("http://{}", addr)
    }

    /// Spin up the brain REST router **with** auth enabled.
    /// Returns `(base_url, token)`.
    ///
    /// Uses `router_with_token` to inject the token directly, avoiding
    /// env-var races when tests run in parallel.
    async fn spawn_server_with_auth(state: Arc<BrainHealthState>, token: &str) -> (String, String) {
        let app = router_with_token(state, Some(token.to_string()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        tokio::time::sleep(Duration::from_millis(30)).await;

        (format!("http://{}", addr), token.to_string())
    }

    #[tokio::test]
    async fn test_health_endpoint_returns_200() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let base = spawn_server(state).await;

        let resp = reqwest::get(format!("{}/api/v1/brain/health", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: BrainHealthResponse = resp.json().await.unwrap();
        assert!(json.healthy);
        assert_eq!(json.state, "Running");
        assert!(json.boot_passed);
        assert!(json.pipeline.is_some());
    }

    #[tokio::test]
    async fn test_health_endpoint_disabled_runtime() {
        let state = Arc::new(BrainHealthState::disabled());
        let base = spawn_server(state).await;

        let resp = reqwest::get(format!("{}/api/v1/brain/health", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: BrainHealthResponse = resp.json().await.unwrap();
        assert!(!json.healthy);
        assert_eq!(json.state, "Uninitialized");
        assert!(!json.boot_passed);
        assert!(json.pipeline.is_none());
    }

    #[tokio::test]
    async fn test_pipeline_endpoint_returns_metrics() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let base = spawn_server(state).await;

        let resp = reqwest::get(format!("{}/api/v1/brain/pipeline", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: PipelineHealthDto = resp.json().await.unwrap();
        assert_eq!(json.total_evaluations, 0);
        assert!(!json.is_killed);
    }

    #[tokio::test]
    async fn test_pipeline_endpoint_no_pipeline_returns_503() {
        let state = make_state(None);
        let base = spawn_server(state).await;

        let resp = reqwest::get(format!("{}/api/v1/brain/pipeline", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 503);
    }

    #[tokio::test]
    async fn test_kill_switch_activate() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline.clone()));
        let base = spawn_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/activate", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: KillSwitchResponse = resp.json().await.unwrap();
        assert!(json.success);
        assert!(json.is_killed);
        assert!(pipeline.is_killed().await);
    }

    #[tokio::test]
    async fn test_kill_switch_deactivate() {
        let pipeline = make_pipeline();
        pipeline.activate_kill_switch().await;
        assert!(pipeline.is_killed().await);

        let state = make_state(Some(pipeline.clone()));
        let base = spawn_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/deactivate", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: KillSwitchResponse = resp.json().await.unwrap();
        assert!(json.success);
        assert!(!json.is_killed);
        assert!(!pipeline.is_killed().await);
    }

    #[tokio::test]
    async fn test_kill_switch_no_pipeline_returns_503() {
        let state = make_state(None);
        let base = spawn_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/activate", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 503);
    }

    #[tokio::test]
    async fn test_health_state_update_runtime_state() {
        let state = BrainHealthState::disabled();
        assert_eq!(
            *state.runtime_state.read().await,
            RuntimeState::Uninitialized
        );

        state.set_runtime_state(RuntimeState::Running).await;
        assert_eq!(*state.runtime_state.read().await, RuntimeState::Running);
    }

    #[tokio::test]
    async fn test_health_state_update_boot_result() {
        let state = BrainHealthState::disabled();
        assert!(!*state.boot_passed.read().await);

        state
            .set_boot_result(true, Some("All passed".to_string()))
            .await;
        assert!(*state.boot_passed.read().await);
        assert_eq!(
            state.boot_summary.read().await.as_deref(),
            Some("All passed")
        );
    }

    #[tokio::test]
    async fn test_health_response_serialization() {
        let response = BrainHealthResponse {
            healthy: true,
            state: "Running".to_string(),
            boot_passed: true,
            boot_summary: Some("All checks passed".to_string()),
            watchdog: None,
            pipeline: Some(PipelineHealthDto {
                total_evaluations: 42,
                proceed_count: 30,
                block_count: 10,
                reduce_only_count: 2,
                avg_evaluation_us: 150.5,
                block_rate_pct: 23.8,
                is_killed: false,
            }),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"healthy\":true"));
        assert!(json.contains("\"total_evaluations\":42"));
        // watchdog should be absent (skip_serializing_if)
        assert!(!json.contains("watchdog"));
    }

    // ────────────────────────────────────────────────────────────────
    // Affinity endpoint tests
    // ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_affinity_export_no_pipeline_returns_unavailable() {
        let state = make_state(None);
        let base = spawn_server(state).await;

        let resp = reqwest::get(format!("{}/api/v1/brain/affinity", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: AffinityExportResponse = resp.json().await.unwrap();
        assert!(!json.available);
        assert_eq!(json.pair_count, 0);
        assert!(json.state.is_none());
    }

    #[tokio::test]
    async fn test_affinity_export_empty_tracker() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let base = spawn_server(state).await;

        let resp = reqwest::get(format!("{}/api/v1/brain/affinity", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: AffinityExportResponse = resp.json().await.unwrap();
        assert!(json.available);
        assert_eq!(json.pair_count, 0);
        assert!(json.state.is_some());
    }

    #[tokio::test]
    async fn test_affinity_export_with_recorded_trades() {
        let pipeline = make_pipeline();

        // Record some trades into the tracker
        {
            let mut gate = pipeline.strategy_gate_mut().await;
            let tracker = gate.tracker_mut();
            tracker.record_trade_result("ema_flip", "BTCUSDT", 150.0, true);
            tracker.record_trade_result("mean_reversion", "ETHUSDT", -30.0, false);
        }

        let state = make_state(Some(pipeline));
        let base = spawn_server(state).await;

        let resp = reqwest::get(format!("{}/api/v1/brain/affinity", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: AffinityExportResponse = resp.json().await.unwrap();
        assert!(json.available);
        assert_eq!(json.pair_count, 2);
        assert!(json.state.is_some());

        // The state JSON should contain our strategy names
        let state_str = serde_json::to_string(&json.state).unwrap();
        assert!(state_str.contains("ema_flip"));
        assert!(state_str.contains("mean_reversion"));
    }

    #[tokio::test]
    async fn test_affinity_reset_clears_tracker() {
        let pipeline = make_pipeline();

        // Record some trades
        {
            let mut gate = pipeline.strategy_gate_mut().await;
            let tracker = gate.tracker_mut();
            tracker.record_trade_result("ema_flip", "BTCUSDT", 100.0, true);
            tracker.record_trade_result("ema_flip", "ETHUSDT", 50.0, true);
            tracker.record_trade_result("squeeze", "BTCUSDT", -20.0, false);
        }

        // Verify 3 pairs exist
        {
            let gate = pipeline.strategy_gate().await;
            assert_eq!(gate.tracker().pair_count(), 3);
        }

        let state = make_state(Some(pipeline.clone()));
        let base = spawn_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/v1/brain/affinity/reset", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: AffinityResetResponse = resp.json().await.unwrap();
        assert!(json.success);
        assert_eq!(json.cleared_pairs, 3);
        assert!(json.message.contains("3"));

        // Verify tracker is now empty
        {
            let gate = pipeline.strategy_gate().await;
            assert_eq!(gate.tracker().pair_count(), 0);
        }
    }

    #[tokio::test]
    async fn test_affinity_reset_no_pipeline_returns_503() {
        let state = make_state(None);
        let base = spawn_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/v1/brain/affinity/reset", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 503);
    }

    #[tokio::test]
    async fn test_affinity_reset_preserves_min_trades_config() {
        use janus_strategies::gating::{StrategyGate, StrategyGatingConfig};

        // Create a pipeline with a custom min_trades setting
        let config = TradingPipelineConfig::default();
        let tracker = StrategyAffinityTracker::new(42);
        let gate = StrategyGate::new(StrategyGatingConfig::default(), tracker);
        let pipeline = Arc::new(
            crate::brain_wiring::TradingPipelineBuilder::new()
                .config(config)
                .strategy_gate(gate)
                .build(),
        );

        // Record a trade
        {
            let mut g = pipeline.strategy_gate_mut().await;
            g.tracker_mut()
                .record_trade_result("test", "BTCUSDT", 10.0, true);
        }

        let state = make_state(Some(pipeline.clone()));
        let base = spawn_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/v1/brain/affinity/reset", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // Verify min_trades was preserved after reset
        {
            let g = pipeline.strategy_gate().await;
            assert_eq!(g.tracker().min_trades_for_confidence, 42);
            assert_eq!(g.tracker().pair_count(), 0);
        }
    }

    #[tokio::test]
    async fn test_affinity_reset_empty_tracker_succeeds() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let base = spawn_server(state).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/v1/brain/affinity/reset", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: AffinityResetResponse = resp.json().await.unwrap();
        assert!(json.success);
        assert_eq!(json.cleared_pairs, 0);
    }

    #[tokio::test]
    async fn test_affinity_export_response_serialization() {
        let response = AffinityExportResponse {
            available: true,
            pair_count: 5,
            state: Some(serde_json::json!({"test": "data"})),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"available\":true"));
        assert!(json.contains("\"pair_count\":5"));
        assert!(json.contains("\"test\":\"data\""));
    }

    // ────────────────────────────────────────────────────────────────
    // Auth middleware tests
    // ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_auth_kill_switch_blocked_without_token() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let (base, _token) = spawn_server_with_auth(state, "test-secret-42").await;

        let client = reqwest::Client::new();

        // POST without auth header → 401
        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/activate", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        let json: BrainApiError = resp.json().await.unwrap();
        assert_eq!(json.error, "unauthorized");
    }

    #[tokio::test]
    async fn test_auth_kill_switch_blocked_with_wrong_token() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let (base, _token) = spawn_server_with_auth(state, "correct-token").await;

        let client = reqwest::Client::new();

        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/activate", base))
            .bearer_auth("wrong-token")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 403);

        let json: BrainApiError = resp.json().await.unwrap();
        assert_eq!(json.error, "forbidden");
    }

    #[tokio::test]
    async fn test_auth_kill_switch_allowed_with_correct_token() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline.clone()));
        let (base, token) = spawn_server_with_auth(state, "my-secret-token").await;

        let client = reqwest::Client::new();

        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/activate", base))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: KillSwitchResponse = resp.json().await.unwrap();
        assert!(json.success);
        assert!(json.is_killed);
        assert!(pipeline.is_killed().await);
    }

    #[tokio::test]
    async fn test_auth_affinity_reset_blocked_without_token() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let (base, _token) = spawn_server_with_auth(state, "secret-reset").await;

        let client = reqwest::Client::new();

        let resp = client
            .post(format!("{}/api/v1/brain/affinity/reset", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    async fn test_auth_affinity_reset_allowed_with_correct_token() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let (base, token) = spawn_server_with_auth(state, "secret-reset").await;

        let client = reqwest::Client::new();

        let resp = client
            .post(format!("{}/api/v1/brain/affinity/reset", base))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: AffinityResetResponse = resp.json().await.unwrap();
        assert!(json.success);
    }

    #[tokio::test]
    async fn test_auth_public_endpoints_always_accessible() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let (base, _token) = spawn_server_with_auth(state, "locked-down").await;

        // Health endpoint — no auth needed
        let resp = reqwest::get(format!("{}/api/v1/brain/health", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // Pipeline endpoint — no auth needed
        let resp = reqwest::get(format!("{}/api/v1/brain/pipeline", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // Affinity export — no auth needed
        let resp = reqwest::get(format!("{}/api/v1/brain/affinity", base))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_auth_non_bearer_scheme_rejected() {
        let pipeline = make_pipeline();
        let state = make_state(Some(pipeline));
        let (base, _token) = spawn_server_with_auth(state, "secret").await;

        let client = reqwest::Client::new();

        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/activate", base))
            .header("Authorization", "Basic dXNlcjpwYXNz")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        let json: BrainApiError = resp.json().await.unwrap();
        assert_eq!(json.error, "unauthorized");
        assert!(json.message.contains("Bearer"));
    }

    #[tokio::test]
    async fn test_auth_deactivate_also_protected() {
        let pipeline = make_pipeline();
        pipeline.activate_kill_switch().await;
        let state = make_state(Some(pipeline.clone()));
        let (base, token) = spawn_server_with_auth(state, "deact-secret").await;

        let client = reqwest::Client::new();

        // Without token → 401
        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/deactivate", base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);

        // With token → 200
        let resp = client
            .post(format!("{}/api/v1/brain/kill-switch/deactivate", base))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let json: KillSwitchResponse = resp.json().await.unwrap();
        assert!(!json.is_killed);
        assert!(!pipeline.is_killed().await);
    }

    #[tokio::test]
    async fn test_affinity_reset_response_serialization() {
        let response = AffinityResetResponse {
            success: true,
            cleared_pairs: 7,
            message: "Affinity tracker reset — 7 strategy-asset pairs cleared".to_string(),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"cleared_pairs\":7"));
        assert!(json.contains("7 strategy-asset pairs cleared"));
    }
}
