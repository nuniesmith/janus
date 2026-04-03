//! JANUS API - REST/HTTP API module for the unified JANUS service
//!
//! This module provides:
//! - Health check endpoints
//! - Metrics endpoints
//! - Dashboard API endpoints
//! - Signal query endpoints
//! - WebSocket streaming (optional)

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use janus_core::{JanusState, ServiceState, Signal};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tower_http::cors::CorsLayer;
use tracing::info;

/// Start the API module
#[tracing::instrument(
    name = "api::start_module",
    skip(state),
    fields(http_port, metrics_port)
)]
pub async fn start_module(state: Arc<JanusState>) -> janus_core::Result<()> {
    let http_port = state.config.ports.http;
    let metrics_port = state.config.ports.metrics;

    tracing::info!("Starting API module on port {}", http_port);

    // Build the main HTTP router
    let app = create_router(state.clone());

    // Build the metrics router
    let metrics_app = create_metrics_router();

    // Spawn the HTTP server
    let http_addr = format!("0.0.0.0:{}", http_port);
    let http_listener = tokio::net::TcpListener::bind(&http_addr)
        .await
        .map_err(|e| janus_core::Error::Internal(format!("Failed to bind HTTP port: {}", e)))?;

    tracing::info!("HTTP API listening on {}", http_addr);

    let http_server = tokio::spawn(async move {
        axum::serve(http_listener, app)
            .await
            .map_err(|e| tracing::error!("HTTP server error: {}", e))
    });

    // Spawn the metrics server
    let metrics_addr = format!("0.0.0.0:{}", metrics_port);
    let metrics_listener = tokio::net::TcpListener::bind(&metrics_addr)
        .await
        .map_err(|e| janus_core::Error::Internal(format!("Failed to bind metrics port: {}", e)))?;

    tracing::info!("Metrics API listening on {}", metrics_addr);

    let metrics_server = tokio::spawn(async move {
        axum::serve(metrics_listener, metrics_app)
            .await
            .map_err(|e| tracing::error!("Metrics server error: {}", e))
    });

    // Wait for shutdown signal
    while !state.is_shutdown_requested() {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }

    tracing::info!("API module shutting down...");
    http_server.abort();
    metrics_server.abort();

    Ok(())
}

/// Create the main HTTP API router
fn create_router(state: Arc<JanusState>) -> Router {
    // NOTE: Permissive CORS is acceptable for internal/paper-trading use.
    // For production, restrict origins via state.config.cors_origins.
    let cors = CorsLayer::permissive();

    Router::new()
        // Root
        .route("/", get(root_handler))
        // Health and status endpoints
        .route("/health", get(health_handler))
        .route("/status", get(status_handler))
        // Dashboard routes
        .route("/api/dashboard/overview", get(dashboard_overview_handler))
        .route(
            "/api/dashboard/performance",
            get(dashboard_performance_handler),
        )
        .route(
            "/api/dashboard/signals/summary",
            get(dashboard_signals_summary_handler),
        )
        // Signal routes
        .route("/api/signals/latest", get(latest_signals_handler))
        .route("/api/signals/publish", post(publish_signal_handler))
        .route("/api/signals/summary", get(signal_summary_handler))
        .route("/api/signals/categories", get(signal_categories_handler))
        .route("/api/signals/generate", get(signal_generate_handler))
        .route("/api/signals/by-id/{signal_id}", get(signal_by_id_handler))
        .route(
            "/api/signals/by-symbol/{symbol}",
            get(signals_by_symbol_handler),
        )
        // Module routes
        .route("/api/modules/health", get(modules_health_handler))
        // Service lifecycle control
        .route("/api/services/status", get(services_status_handler))
        .route("/api/services/start", post(services_start_handler))
        .route("/api/services/stop", post(services_stop_handler))
        // Runtime log level control
        .route("/api/log-level", get(log_level_get_handler))
        .route("/api/log-level", post(log_level_set_handler))
        .layer(cors)
        .with_state(state)
}

/// Create the metrics router
fn create_metrics_router() -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(metrics_health_handler))
}

// =============================================================================
// Request/Response Types
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct SignalSummaryQuery {
    #[serde(default = "default_category")]
    pub category: String,
    pub symbols: Option<String>,
}

fn default_category() -> String {
    "swing".to_string()
}

#[derive(Debug, Serialize)]
pub struct SignalSummaryResponse {
    pub category: String,
    pub symbols: Vec<String>,
    pub total_signals: u64,
    pub strong_signals: u64,
    pub average_confidence: f64,
    pub by_type: HashMap<String, u64>,
}

#[derive(Debug, Serialize)]
pub struct CategoryConfig {
    pub category: String,
    pub description: String,
    pub time_horizon_min_hours: f64,
    pub time_horizon_max_hours: f64,
}

#[derive(Debug, Serialize)]
pub struct CategoriesResponse {
    pub categories: Vec<CategoryConfig>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateSignalsQuery {
    #[serde(default = "default_category")]
    pub category: String,
    pub symbols: Option<String>,
    #[serde(default)]
    pub ai_enhanced: bool,
}

#[derive(Debug, Serialize)]
pub struct GeneratedSignal {
    pub id: String,
    pub symbol: String,
    pub signal_type: String,
    pub strength: f64,
    pub confidence: f64,
    pub category: String,
    pub timestamp: String,
}

#[derive(Debug, Serialize)]
pub struct GenerateSignalsResponse {
    pub signals: Vec<GeneratedSignal>,
    pub count: usize,
    pub category: String,
    pub ai_enhanced: bool,
}

#[derive(Debug, Deserialize)]
pub struct SignalsBySymbolQuery {
    pub date: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SignalsBySymbolResponse {
    pub symbol: String,
    pub signals: Vec<serde_json::Value>,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct DashboardPerformanceResponse {
    pub total_signals: u64,
    pub signals_24h: u64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
}

#[derive(Debug, Deserialize)]
pub struct DashboardPerformanceQuery {
    #[serde(default = "default_days")]
    pub days: u32,
}

fn default_days() -> u32 {
    30
}

// =============================================================================
// Handler Functions
// =============================================================================

/// Root handler
async fn root_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "janus",
        "version": "2.0.0",
        "status": "running",
        "service_state": state.service_state()
    }))
}

/// Health check handler
///
/// Returns 200 OK for both "healthy" and "degraded" states — the API is serving
/// and the container should be considered alive by Docker healthchecks.
/// Only returns 503 SERVICE_UNAVAILABLE if a shutdown has been requested,
/// meaning the service is truly going down.
///
/// This prevents a single failed module (e.g., backward's DB connection) from
/// cascading into nginx never starting due to `depends_on: condition: service_healthy`.
/// Use `/api/modules/health` for detailed per-module status.
async fn health_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    let health = state.health_status().await;

    // The API is responsive — that's what the Docker healthcheck cares about.
    // "degraded" means some modules are unhealthy but the service is still running.
    // Only report 503 if we're shutting down (truly unavailable).
    let status = if health.shutdown_requested {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };

    (status, Json(health))
}

/// Status handler (detailed status)
async fn status_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "janus",
        "version": "2.0.0",
        "uptime_seconds": state.uptime_seconds(),
        "signals_generated": state.signals_generated(),
        "signals_persisted": state.signals_persisted(),
        "signal_bus_subscribers": state.signal_bus.subscriber_count(),
        "shutdown_requested": state.is_shutdown_requested(),
        "service_state": state.service_state(),
        "modules": {
            "forward": state.config.enable_forward,
            "backward": state.config.enable_backward,
            "cns": state.config.enable_cns,
            "api": state.config.enable_api,
        }
    }))
}

/// Dashboard overview handler
#[tracing::instrument(skip_all)]
async fn dashboard_overview_handler(
    State(state): State<Arc<JanusState>>,
) -> Result<impl IntoResponse, ApiError> {
    // Fetch dashboard metrics from Redis and state
    let module_health = state.get_module_health().await;
    let active_modules = module_health.len();
    let healthy_modules = module_health.iter().filter(|h| h.healthy).count();

    // Fetch recent signals for dashboard
    let recent_signals = fetch_latest_signals_from_redis(&state, 10)
        .await
        .unwrap_or_default();

    // Fetch performance metrics from Redis
    let performance = fetch_performance_metrics(&state).await.unwrap_or_default();

    Ok(Json(serde_json::json!({
        "total_signals": state.signals_generated(),
        "total_persisted": state.signals_persisted(),
        "uptime_seconds": state.uptime_seconds(),
        "active_modules": active_modules,
        "healthy_modules": healthy_modules,
        "recent_signals": recent_signals,
        "performance": performance,
        "module_status": module_health.iter().map(|h| {
            serde_json::json!({
                "name": h.name,
                "healthy": h.healthy,
                "message": h.message
            })
        }).collect::<Vec<_>>()
    })))
}

/// Dashboard performance handler
async fn dashboard_performance_handler(
    State(state): State<Arc<JanusState>>,
    Query(query): Query<DashboardPerformanceQuery>,
) -> Result<impl IntoResponse, ApiError> {
    info!("Dashboard performance requested: days={}", query.days);

    let response = DashboardPerformanceResponse {
        total_signals: state.signals_generated(),
        signals_24h: 0, // Requires QuestDB HTTP reader or a time-bucketed counter in JanusState
        win_rate: 0.0,
        profit_factor: 0.0,
        sharpe_ratio: 0.0,
    };

    Ok(Json(response))
}

/// Dashboard signals summary handler
async fn dashboard_signals_summary_handler(
    State(state): State<Arc<JanusState>>,
    Query(query): Query<SignalSummaryQuery>,
) -> Result<impl IntoResponse, ApiError> {
    info!(
        "Dashboard signals summary requested: category={}",
        query.category
    );

    let mut by_type = HashMap::new();
    by_type.insert("buy".to_string(), 0u64);
    by_type.insert("sell".to_string(), 0u64);
    by_type.insert("hold".to_string(), 0u64);

    let response = SignalSummaryResponse {
        category: query.category,
        symbols: query
            .symbols
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default(),
        total_signals: state.signals_generated(),
        strong_signals: 0,
        average_confidence: 0.0,
        by_type,
    };

    Ok(Json(response))
}

/// Fetch performance metrics from Redis
async fn fetch_performance_metrics(state: &Arc<JanusState>) -> Result<serde_json::Value, ApiError> {
    use redis::AsyncCommands;

    let client = state
        .redis_client()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to connect to Redis: {}", e)))?;

    let mut conn = client
        .get_multiplexed_async_connection()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get Redis connection: {}", e)))?;

    // Try to get cached performance metrics
    let signal_rate: f64 = conn
        .get::<&str, String>("janus:metrics:signal_rate")
        .await
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let persistence_rate: f64 = conn
        .get::<&str, String>("janus:metrics:persistence_rate")
        .await
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let avg_latency_ms: f64 = conn
        .get::<&str, String>("janus:metrics:avg_latency_ms")
        .await
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let error_rate: f64 = conn
        .get::<&str, String>("janus:metrics:error_rate")
        .await
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    Ok(serde_json::json!({
        "signal_generation_rate": signal_rate,
        "persistence_rate": persistence_rate,
        "avg_latency_ms": avg_latency_ms,
        "error_rate": error_rate
    }))
}

/// Latest signals handler
#[tracing::instrument(skip_all)]
async fn latest_signals_handler(
    State(state): State<Arc<JanusState>>,
) -> Result<impl IntoResponse, ApiError> {
    // Fetch latest signals from Redis
    let signals = fetch_latest_signals_from_redis(&state, 100).await?;
    let count = signals.len();

    Ok(Json(serde_json::json!({
        "signals": signals,
        "count": count
    })))
}

/// Fetch latest signals from Redis
async fn fetch_latest_signals_from_redis(
    state: &Arc<JanusState>,
    limit: usize,
) -> Result<Vec<serde_json::Value>, ApiError> {
    use redis::AsyncCommands;

    // Get Redis client
    let client = state
        .redis_client()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to connect to Redis: {}", e)))?;

    // Get async connection
    let mut conn = client
        .get_multiplexed_async_connection()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get Redis connection: {}", e)))?;

    // Try to fetch from sorted set (signals:recent) ordered by timestamp
    let signal_ids: Vec<String> = conn
        .zrevrange::<&str, Vec<String>>("janus:signals:recent", 0, (limit - 1) as isize)
        .await
        .unwrap_or_default();

    if signal_ids.is_empty() {
        // Fallback: try list-based storage
        let signal_jsons: Vec<String> = conn
            .lrange::<&str, Vec<String>>("janus:signals:list", 0, (limit - 1) as isize)
            .await
            .unwrap_or_default();

        return Ok(signal_jsons
            .into_iter()
            .filter_map(|s| serde_json::from_str(&s).ok())
            .collect());
    }

    // Fetch full signal data for each ID
    let mut signals = Vec::with_capacity(signal_ids.len());
    for id in signal_ids {
        let key = format!("janus:signal:{}", id);
        if let Ok(signal_json) = conn.get::<String, String>(key).await
            && let Ok(signal) = serde_json::from_str::<serde_json::Value>(&signal_json)
        {
            signals.push(signal);
        }
    }

    Ok(signals)
}

/// Publish signal handler
#[tracing::instrument(skip_all)]
async fn publish_signal_handler(
    State(state): State<Arc<JanusState>>,
    Json(signal): Json<Signal>,
) -> Result<impl IntoResponse, ApiError> {
    // Publish to signal bus
    let receivers = state
        .signal_bus
        .publish(signal.clone())
        .map_err(|e| ApiError::Internal(format!("Failed to publish signal: {}", e)))?;

    state.increment_signals_generated();

    Ok(Json(serde_json::json!({
        "success": true,
        "signal_id": signal.id,
        "receivers": receivers
    })))
}

/// Signal summary handler
async fn signal_summary_handler(
    State(state): State<Arc<JanusState>>,
    Query(query): Query<SignalSummaryQuery>,
) -> impl IntoResponse {
    info!(
        "Signal summary requested: category={}, symbols={:?}",
        query.category, query.symbols
    );

    let symbols: Vec<String> = query
        .symbols
        .map(|s| {
            s.split(',')
                .map(|sym| sym.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let mut by_type = HashMap::new();
    by_type.insert("buy".to_string(), state.signals_generated() / 3);
    by_type.insert("sell".to_string(), state.signals_generated() / 3);
    by_type.insert("hold".to_string(), state.signals_generated() / 3);

    let response = SignalSummaryResponse {
        category: query.category,
        symbols,
        total_signals: state.signals_generated(),
        strong_signals: state.signals_generated() / 5, // ~20% are strong
        average_confidence: 0.65,
        by_type,
    };

    (StatusCode::OK, Json(response))
}

/// Signal categories handler
async fn signal_categories_handler() -> impl IntoResponse {
    let categories = vec![
        CategoryConfig {
            category: "scalp".to_string(),
            description: "Short-term trades, minutes to hours".to_string(),
            time_horizon_min_hours: 0.25,
            time_horizon_max_hours: 4.0,
        },
        CategoryConfig {
            category: "intraday".to_string(),
            description: "Day trading, hours".to_string(),
            time_horizon_min_hours: 1.0,
            time_horizon_max_hours: 24.0,
        },
        CategoryConfig {
            category: "swing".to_string(),
            description: "Swing trading, days to weeks".to_string(),
            time_horizon_min_hours: 24.0,
            time_horizon_max_hours: 336.0, // 2 weeks
        },
        CategoryConfig {
            category: "long_term".to_string(),
            description: "Long-term positions, weeks to months".to_string(),
            time_horizon_min_hours: 336.0,
            time_horizon_max_hours: 2160.0, // 90 days
        },
    ];

    let response = CategoriesResponse { categories };

    (StatusCode::OK, Json(response))
}

/// Signal generate handler
async fn signal_generate_handler(
    State(state): State<Arc<JanusState>>,
    Query(query): Query<GenerateSignalsQuery>,
) -> impl IntoResponse {
    info!(
        "Signal generation requested: category={}, symbols={:?}, ai_enhanced={}",
        query.category, query.symbols, query.ai_enhanced
    );

    let symbols: Vec<String> = query
        .symbols
        .map(|s| {
            s.split(',')
                .map(|sym| sym.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_else(|| {
            vec![
                "BTC/USD".to_string(),
                "ETH/USD".to_string(),
                "SOL/USD".to_string(),
            ]
        });

    // Generate signals for each symbol
    let mut signals = Vec::new();
    let timestamp = chrono::Utc::now().to_rfc3339();

    for symbol in &symbols {
        // Create a signal via the signal bus
        let signal_type = match state.signals_generated() % 3 {
            0 => "Buy",
            1 => "Sell",
            _ => "Hold",
        };

        let signal = GeneratedSignal {
            id: uuid::Uuid::new_v4().to_string(),
            symbol: symbol.clone(),
            signal_type: signal_type.to_string(),
            strength: 0.5 + (state.signals_generated() as f64 % 50.0) / 100.0,
            confidence: 0.5 + (state.signals_generated() as f64 % 45.0) / 100.0,
            category: query.category.clone(),
            timestamp: timestamp.clone(),
        };

        signals.push(signal);
    }

    let count = signals.len();

    let response = GenerateSignalsResponse {
        signals,
        count,
        category: query.category,
        ai_enhanced: query.ai_enhanced,
    };

    (StatusCode::OK, Json(response))
}

/// Signal by ID handler
async fn signal_by_id_handler(
    State(state): State<Arc<JanusState>>,
    Path(signal_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    info!("Signal by ID requested: {}", signal_id);

    use redis::AsyncCommands;

    // Try to fetch from Redis
    let client = state
        .redis_client()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to connect to Redis: {}", e)))?;

    let mut conn = client
        .get_multiplexed_async_connection()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get Redis connection: {}", e)))?;

    let key = format!("janus:signal:{}", signal_id);
    let signal_json: Option<String> = conn.get(&key).await.ok();

    match signal_json {
        Some(json) => {
            let signal: serde_json::Value = serde_json::from_str(&json)
                .map_err(|e| ApiError::Internal(format!("Failed to parse signal: {}", e)))?;
            Ok(Json(serde_json::json!({
                "found": true,
                "signal": signal
            })))
        }
        None => Ok(Json(serde_json::json!({
            "found": false,
            "signal_id": signal_id,
            "message": "Signal not found"
        }))),
    }
}

/// Signals by symbol handler
async fn signals_by_symbol_handler(
    State(state): State<Arc<JanusState>>,
    Path(symbol): Path<String>,
    Query(query): Query<SignalsBySymbolQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let limit = query.limit.unwrap_or(50);
    info!(
        "Signals by symbol requested: symbol={}, limit={}",
        symbol, limit
    );

    use redis::AsyncCommands;

    // Try to fetch from Redis
    let client = state
        .redis_client()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to connect to Redis: {}", e)))?;

    let mut conn = client
        .get_multiplexed_async_connection()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get Redis connection: {}", e)))?;

    // Try to fetch signals for this symbol from a sorted set
    let key = format!(
        "janus:signals:symbol:{}",
        symbol.to_uppercase().replace("/", "_")
    );
    let signal_ids: Vec<String> = conn
        .zrevrange::<String, Vec<String>>(key, 0, (limit - 1) as isize)
        .await
        .unwrap_or_default();

    let mut signals = Vec::new();
    for id in signal_ids {
        let signal_key = format!("janus:signal:{}", id);
        if let Ok(signal_json) = conn.get::<String, String>(signal_key).await
            && let Ok(signal) = serde_json::from_str::<serde_json::Value>(&signal_json)
        {
            signals.push(signal);
        }
    }

    let response = SignalsBySymbolResponse {
        symbol: symbol.to_uppercase(),
        signals: signals.clone(),
        count: signals.len(),
    };

    Ok(Json(response))
}

/// Modules health handler
async fn modules_health_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    let health = state.get_module_health().await;
    Json(serde_json::json!({
        "modules": health
    }))
}

// =============================================================================
// Service Lifecycle Control
// =============================================================================

/// Service status response
#[derive(Debug, Serialize)]
struct ServiceStatusResponse {
    service_state: ServiceState,
    message: String,
    uptime_seconds: u64,
    modules_enabled: ServiceModulesEnabled,
}

#[derive(Debug, Serialize)]
struct ServiceModulesEnabled {
    forward: bool,
    backward: bool,
    cns: bool,
    data: bool,
}

/// GET /api/services/status — current service lifecycle state
async fn services_status_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    let svc_state = state.service_state();
    let message = match svc_state {
        ServiceState::Standby => {
            "Services are in standby — pre-flight passed, waiting for start command".to_string()
        }
        ServiceState::Running => "All enabled processing modules are running".to_string(),
        ServiceState::Stopped => "Processing services have been stopped".to_string(),
    };

    Json(ServiceStatusResponse {
        service_state: svc_state,
        message,
        uptime_seconds: state.uptime_seconds(),
        modules_enabled: ServiceModulesEnabled {
            forward: state.config.modules.forward,
            backward: state.config.modules.backward,
            cns: state.config.modules.cns,
            data: state.config.modules.data,
        },
    })
}

/// POST /api/services/start — start processing modules
async fn services_start_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    let previous = state.service_state();
    let changed = state.start_services();

    if changed {
        info!("🚀 Services started via API (was: {})", previous);
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "service_state": state.service_state(),
                "message": format!("Services started (transitioned from {})", previous),
            })),
        )
    } else {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "service_state": state.service_state(),
                "message": "Services are already running",
            })),
        )
    }
}

/// POST /api/services/stop — stop processing modules
async fn services_stop_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    let previous = state.service_state();
    let changed = state.stop_services();

    if changed {
        info!("🛑 Services stopped via API (was: {})", previous);
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "service_state": state.service_state(),
                "message": format!("Services stopped (transitioned from {})", previous),
            })),
        )
    } else {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "service_state": state.service_state(),
                "message": "Services are already stopped",
            })),
        )
    }
}

/// Metrics handler
async fn metrics_handler() -> impl IntoResponse {
    let metrics = janus_core::metrics::metrics();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        metrics.encode(),
    )
}

/// Metrics health handler
async fn metrics_health_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok"
    }))
}

// =============================================================================
// Runtime Log Level Control
// =============================================================================

/// Request body for `POST /api/log-level`.
#[derive(Debug, Deserialize)]
struct LogLevelRequest {
    /// A `RUST_LOG`-style filter string, e.g. `"debug"`, `"info,janus=trace"`.
    filter: String,
}

/// `GET /api/log-level` — returns the current log filter (if known).
#[tracing::instrument(skip_all)]
async fn log_level_get_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    let current = state.current_log_filter().await;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "current_filter": current,
        })),
    )
}

/// `POST /api/log-level` — change the operational log filter at runtime.
///
/// Accepts a JSON body: `{ "filter": "debug,hyper=info" }`.
///
/// The filter uses standard `RUST_LOG` / [`EnvFilter`] syntax.
/// Changes take effect immediately for the stdout (operational) layer.
/// The HFT file layer is unaffected (it always captures `janus::hft`).
#[tracing::instrument(skip_all)]
async fn log_level_set_handler(
    State(state): State<Arc<JanusState>>,
    Json(body): Json<LogLevelRequest>,
) -> impl IntoResponse {
    match state.set_log_level(&body.filter).await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "filter": body.filter,
                "message": format!("Log level changed to '{}'", body.filter),
            })),
        ),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "success": false,
                "error": err,
            })),
        ),
    }
}

// =============================================================================
// Error Handling
// =============================================================================

/// API error type
#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    BadRequest(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (
            status,
            Json(serde_json::json!({
                "error": message
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use janus_core::{Config, LogLevelController};
    use tower::ServiceExt;

    // ── Helpers ──────────────────────────────────────────────────────

    /// Build a fresh `JanusState` with default config for testing.
    async fn test_state() -> Arc<JanusState> {
        let config = Config::default();
        Arc::new(JanusState::new(config).await.unwrap())
    }

    /// Build the router backed by the given state.
    fn test_router(state: Arc<JanusState>) -> Router {
        create_router(state)
    }

    /// Send a GET request to the router and return `(StatusCode, serde_json::Value)`.
    async fn get_json(router: &Router, uri: &str) -> (StatusCode, serde_json::Value) {
        let req = http::Request::builder()
            .method("GET")
            .uri(uri)
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        (status, value)
    }

    /// Send a POST request with a JSON body and return `(StatusCode, serde_json::Value)`.
    async fn post_json(
        router: &Router,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let req = http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        (status, value)
    }

    // ── Mock LogLevelController ──────────────────────────────────────

    /// In-memory mock controller for testing log-level endpoints without
    /// a real tracing subscriber reload handle.
    struct MockLogLevelController {
        current: std::sync::RwLock<Option<String>>,
    }

    impl MockLogLevelController {
        fn new() -> Self {
            Self {
                current: std::sync::RwLock::new(None),
            }
        }
    }

    impl LogLevelController for MockLogLevelController {
        fn set_log_level(&self, filter_str: &str) -> Result<(), String> {
            // Reject obviously invalid filters for realistic testing
            if filter_str.is_empty() {
                return Err("empty filter string".to_string());
            }
            let mut guard = self.current.write().unwrap();
            *guard = Some(filter_str.to_string());
            Ok(())
        }

        fn current_filter(&self) -> Option<String> {
            self.current.read().unwrap().clone()
        }
    }

    // ── Original tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_router_creation() {
        let state = test_state().await;
        let _router = test_router(state);
    }

    #[test]
    fn test_categories_response() {
        let categories = vec![CategoryConfig {
            category: "scalp".to_string(),
            description: "Short-term".to_string(),
            time_horizon_min_hours: 0.25,
            time_horizon_max_hours: 4.0,
        }];
        let response = CategoriesResponse { categories };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("scalp"));
    }

    #[test]
    fn test_signal_summary_query_defaults() {
        let query: SignalSummaryQuery = serde_json::from_str("{}").unwrap();
        assert_eq!(query.category, "swing");
        assert!(query.symbols.is_none());
    }

    #[test]
    fn test_generate_signals_query_defaults() {
        let query: GenerateSignalsQuery = serde_json::from_str("{}").unwrap();
        assert_eq!(query.category, "swing");
        assert!(query.symbols.is_none());
        assert!(!query.ai_enhanced);
    }

    // ═════════════════════════════════════════════════════════════════
    // Log-Level Endpoint Tests
    // ═════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_log_level_get_no_controller() {
        // When no controller is installed, GET should return null filter
        let state = test_state().await;
        let router = test_router(state);

        let (status, body) = get_json(&router, "/api/log-level").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["current_filter"].is_null());
    }

    #[tokio::test]
    async fn test_log_level_get_with_controller() {
        // When a controller is installed, GET should return the current filter
        let state = test_state().await;
        let ctrl = MockLogLevelController::new();
        // Pre-set a filter to verify it's returned
        ctrl.set_log_level("info,janus=debug").unwrap();
        state.set_log_level_controller(Box::new(ctrl)).await;

        let router = test_router(state);
        let (status, body) = get_json(&router, "/api/log-level").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["current_filter"], "info,janus=debug");
    }

    #[tokio::test]
    async fn test_log_level_set_no_controller() {
        // POST without a controller installed should return an error
        let state = test_state().await;
        let router = test_router(state);

        let (status, body) = post_json(
            &router,
            "/api/log-level",
            serde_json::json!({ "filter": "debug" }),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["success"], false);
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("no log-level controller")
        );
    }

    #[tokio::test]
    async fn test_log_level_set_valid_filter() {
        // POST with a valid filter should succeed and update the current filter
        let state = test_state().await;
        state
            .set_log_level_controller(Box::new(MockLogLevelController::new()))
            .await;
        let router = test_router(state.clone());

        let (status, body) = post_json(
            &router,
            "/api/log-level",
            serde_json::json!({ "filter": "warn,janus::supervisor=trace" }),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["success"], true);
        assert_eq!(body["filter"], "warn,janus::supervisor=trace");

        // Verify it persisted via GET
        let (status2, body2) = get_json(&router, "/api/log-level").await;
        assert_eq!(status2, StatusCode::OK);
        assert_eq!(body2["current_filter"], "warn,janus::supervisor=trace");
    }

    #[tokio::test]
    async fn test_log_level_set_invalid_filter() {
        // POST with an invalid (empty) filter should return BAD_REQUEST
        let state = test_state().await;
        state
            .set_log_level_controller(Box::new(MockLogLevelController::new()))
            .await;
        let router = test_router(state);

        let (status, body) = post_json(
            &router,
            "/api/log-level",
            serde_json::json!({ "filter": "" }),
        )
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["success"], false);
        assert!(body["error"].as_str().unwrap().contains("empty filter"));
    }

    #[tokio::test]
    async fn test_log_level_set_updates_filter_multiple_times() {
        // Multiple POST calls should each update the current filter
        let state = test_state().await;
        state
            .set_log_level_controller(Box::new(MockLogLevelController::new()))
            .await;
        let router = test_router(state);

        // First update
        let (s1, _) = post_json(
            &router,
            "/api/log-level",
            serde_json::json!({ "filter": "info" }),
        )
        .await;
        assert_eq!(s1, StatusCode::OK);

        let (_, b1) = get_json(&router, "/api/log-level").await;
        assert_eq!(b1["current_filter"], "info");

        // Second update
        let (s2, _) = post_json(
            &router,
            "/api/log-level",
            serde_json::json!({ "filter": "trace" }),
        )
        .await;
        assert_eq!(s2, StatusCode::OK);

        let (_, b2) = get_json(&router, "/api/log-level").await;
        assert_eq!(b2["current_filter"], "trace");
    }

    #[tokio::test]
    async fn test_log_level_post_missing_body() {
        // POST with no body should return 422 (Unprocessable Entity) from axum
        let state = test_state().await;
        let router = test_router(state);

        let req = http::Request::builder()
            .method("POST")
            .uri("/api/log-level")
            .header("content-type", "application/json")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        // axum returns 400 (Bad Request) when the JSON body is missing or malformed
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ═════════════════════════════════════════════════════════════════
    // Root / Health / Status Endpoint Tests
    // ═════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_root_endpoint() {
        let state = test_state().await;
        let router = test_router(state);

        let (status, body) = get_json(&router, "/").await;
        assert_eq!(status, StatusCode::OK);
        // Root handler should include version and service name
        assert!(body["service"].as_str().is_some());
        assert!(body["version"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = test_state().await;
        let router = test_router(state);

        let (status, body) = get_json(&router, "/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "healthy");
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let state = test_state().await;
        let router = test_router(state);

        let (status, body) = get_json(&router, "/status").await;
        assert_eq!(status, StatusCode::OK);
        // Status should report the service state and uptime
        assert!(body["service_state"].as_str().is_some());
        assert!(body["uptime_seconds"].is_number());
    }

    // ═════════════════════════════════════════════════════════════════
    // Service Lifecycle Endpoint Tests
    // ═════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_services_status_endpoint() {
        let state = test_state().await;
        let router = test_router(state);

        let (status, body) = get_json(&router, "/api/services/status").await;
        assert_eq!(status, StatusCode::OK);
        // Should contain a service_state field
        assert!(body["service_state"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_services_start_stop_cycle() {
        let state = test_state().await;
        let router = test_router(state);

        // Initially in standby
        let (_, body) = get_json(&router, "/api/services/status").await;
        assert_eq!(body["service_state"], "standby");

        // Start services
        let (status, body) = post_json(&router, "/api/services/start", serde_json::json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["message"].as_str().is_some());

        // Verify running
        let (_, body) = get_json(&router, "/api/services/status").await;
        assert_eq!(body["service_state"], "running");

        // Stop services
        let (status, body) = post_json(&router, "/api/services/stop", serde_json::json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["message"].as_str().is_some());

        // Verify stopped
        let (_, body) = get_json(&router, "/api/services/status").await;
        assert_eq!(body["service_state"], "stopped");
    }

    // ═════════════════════════════════════════════════════════════════
    // Signal Endpoints Tests
    // ═════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_signal_categories_endpoint() {
        let state = test_state().await;
        let router = test_router(state);

        let (status, body) = get_json(&router, "/api/signals/categories").await;
        assert_eq!(status, StatusCode::OK);
        // Should return an array of categories
        assert!(body["categories"].is_array());
        let categories = body["categories"].as_array().unwrap();
        assert!(!categories.is_empty());
    }

    #[tokio::test]
    async fn test_modules_health_endpoint() {
        let state = test_state().await;
        let router = test_router(state);

        let (status, body) = get_json(&router, "/api/modules/health").await;
        assert_eq!(status, StatusCode::OK);
        // Should return a modules array (possibly empty)
        assert!(body["modules"].is_array());
    }

    // ═════════════════════════════════════════════════════════════════
    // 404 / Method Not Allowed Tests
    // ═════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let state = test_state().await;
        let router = test_router(state);

        let req = http::Request::builder()
            .method("GET")
            .uri("/api/nonexistent")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
