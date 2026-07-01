//! JANUS API - REST/HTTP API module for the unified JANUS service
//!
//! This module provides:
//! - Health check endpoints
//! - Metrics endpoints
//! - Dashboard API endpoints
//! - Signal query endpoints
//! - WebSocket streaming (optional)

pub mod bars;
mod param_updates;
pub mod position_store;
pub mod sse_bars;
pub mod webui_contract;

use axum::{
    Extension, Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use janus_core::{
    DEFAULT_ATR_MULTIPLIER, GuidanceThresholds, JanusState, ParamManager, PositionClose,
    PositionEvent, PositionOutcome, PositionTracker, ServiceState, Signal, base_asset,
    compute_guidance,
};
use position_store::PositionEventStore;
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

    // Position event persistence (JFLOW-C): connect best-effort. The store
    // is always present in the request extensions — it's a no-op when the
    // DB is unreachable or the table is missing.
    let position_store = Arc::new(PositionEventStore::connect(&state.config.database.url).await);

    // Per-asset optimizer-tuned thresholds for guidance. Best-effort
    // initial load from Redis, then a background subscription so live
    // optimizer pushes refresh the cache without a restart. Missing
    // connection or empty hash → manager stays empty and guidance falls
    // back to default thresholds.
    let param_manager = Arc::new(ParamManager::new(&state.config.service.name));
    match state.redis_client().await {
        Ok(client) => match param_manager.load_from_redis(&client).await {
            Ok(count) => {
                tracing::info!(count, "Loaded optimized params from Redis for guidance");
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to load optimized params from Redis; guidance will use defaults"
                );
            }
        },
        Err(e) => {
            tracing::warn!(
                error = %e,
                "Redis unavailable at startup; guidance will use default thresholds"
            );
        }
    }
    let param_updates_task = param_updates::spawn(state.clone(), param_manager.clone());

    // Per-position guidance state: trailing give-back + sticky exits across
    // the repeated snapshots a producer pushes for one position_id.
    let position_tracker = Arc::new(PositionTracker::new());

    // Build the main HTTP router
    let app = create_router(
        state.clone(),
        position_store,
        param_manager,
        position_tracker,
    );

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
    param_updates_task.abort();

    Ok(())
}

/// Create the main HTTP API router
fn create_router(
    state: Arc<JanusState>,
    position_store: Arc<PositionEventStore>,
    param_manager: Arc<ParamManager>,
    position_tracker: Arc<PositionTracker>,
) -> Router {
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
        // Position event ingress (JFLOW-C foundation: receive + log, no guidance yet)
        .route("/api/v1/positions/event", post(position_event_handler))
        .route("/api/v1/positions/close", post(position_close_handler))
        // Live closed-candle stream for the FKS WebUI chart (D1 bridge)
        .route("/sse/bars/{symbol}", get(sse_bars::sse_bars_handler))
        // REST candle history for the chart's initial load (Track D contract):
        // columnar for the trading page, flat ms-timestamps for MiniChart/charts.
        .route("/bars/{symbol}", get(bars::bars_history_handler))
        .route("/bars/{symbol}/candles", get(bars::bars_candles_handler))
        // FKS WebUI front-page contract (Track D): per-asset scores, open
        // trades, and data/runtime health — served truthfully from janus state.
        .route(
            "/api/pipeline/scores/json",
            get(webui_contract::scores_handler),
        )
        .route("/api/trades/open", get(webui_contract::open_trades_handler))
        .route(
            "/factory/status",
            get(webui_contract::factory_status_handler),
        )
        .layer(Extension(position_store))
        .layer(Extension(param_manager))
        .layer(Extension(position_tracker))
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
    /// Marks `win_rate` / `profit_factor` / `sharpe_ratio` as not-yet-computed
    /// placeholders (no realised-PnL tracking is wired) so callers don't render
    /// the zeros as real performance. Drop this once backed by trade history.
    pub pnl_metrics_status: &'static str,
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
            "forward": state.config.modules.forward,
            "backward": state.config.modules.backward,
            "cns": state.config.modules.cns,
            "api": state.config.modules.api,
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
        pnl_metrics_status: "unimplemented_placeholder",
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
pub(crate) async fn fetch_latest_signals_from_redis(
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

    // Aggregate over the recently-persisted signals (written by the forward
    // signal→Redis persistence subscriber) instead of fabricating from a
    // counter. An empty window yields honest zeros rather than invented
    // buy/sell/hold thirds and a constant 0.65 confidence.
    let recent = fetch_latest_signals_from_redis(&state, 500)
        .await
        .unwrap_or_default();

    let mut by_type: HashMap<String, u64> = HashMap::new();
    for t in ["buy", "sell", "hold", "close"] {
        by_type.insert(t.to_string(), 0);
    }
    let mut confidence_sum = 0.0_f64;
    let mut confidence_n = 0_u64;
    let mut strong_signals = 0_u64;
    for sig in &recent {
        if let Some(t) = sig.get("signal_type").and_then(|v| v.as_str()) {
            *by_type.entry(t.to_lowercase()).or_insert(0) += 1;
        }
        if let Some(c) = sig.get("confidence").and_then(|v| v.as_f64()) {
            confidence_sum += c;
            confidence_n += 1;
            if c >= 0.7 {
                strong_signals += 1;
            }
        }
    }
    let average_confidence = if confidence_n > 0 {
        confidence_sum / confidence_n as f64
    } else {
        0.0
    };

    let response = SignalSummaryResponse {
        category: query.category,
        symbols,
        // Lifetime count of signals generated on the bus (a real counter);
        // the per-type / strength / confidence breakdown below reflects the
        // recent persisted window.
        total_signals: state.signals_generated(),
        strong_signals,
        average_confidence,
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

/// Receive a position snapshot from the execution side (JFLOW-C).
///
/// Pipeline: validate at the boundary → look up per-asset optimizer-tuned
/// thresholds (falling back to defaults) → log → best-effort persist via
/// the `PositionEventStore` → compute advisory guidance from the current
/// regime, P&L, and thresholds → return it in the 202 response. Guidance
/// is advisory; the producer decides whether to act on it.
#[tracing::instrument(
    skip(event, store, state, params, tracker),
    fields(symbol = %event.symbol, side = ?event.side, qty = event.qty)
)]
async fn position_event_handler(
    State(state): State<Arc<JanusState>>,
    Extension(store): Extension<Arc<PositionEventStore>>,
    Extension(params): Extension<Arc<ParamManager>>,
    Extension(tracker): Extension<Arc<PositionTracker>>,
    Json(event): Json<PositionEvent>,
) -> impl IntoResponse {
    if let Err(reason) = event.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "accepted": false,
                "error": reason,
            })),
        );
    }

    let regime = state.current_regime().await;
    let fear = state.current_threat().await;
    let asset = base_asset(&event.symbol);
    // Per-asset tuned thresholds + the optimizer's ATR multiplier (used to
    // size the volatility band below). Fall back to defaults when the asset
    // has no optimized params yet.
    let (mut thresholds, atr_multiplier) = match params.get(asset).await {
        Some(p) => (
            GuidanceThresholds::from_optimized_params(&p),
            p.atr_multiplier,
        ),
        None => (GuidanceThresholds::default(), DEFAULT_ATR_MULTIPLIER),
    };
    // If the producer attached a volatility hint, widen the stop so normal
    // ATR-sized noise doesn't trip an exit. Absent hint → thresholds unchanged.
    if let Some(atr_pct) = event.atr_pct {
        thresholds = thresholds.widen_for_volatility(atr_pct, atr_multiplier);
    }
    // Stateless score, then refine with this position's history (trailing
    // give-back, sticky exit). Untracked positions (no position_id) pass
    // through unchanged.
    let base = compute_guidance(&event, regime.as_deref(), thresholds, fear);
    let guidance = tracker.observe(&event, base).await;

    info!(
        symbol = %event.symbol,
        side = ?event.side,
        qty = event.qty,
        entry_price = event.entry_price,
        current_price = event.current_price,
        pnl_unrealized = event.pnl_unrealized,
        position_id = event.position_id.as_deref().unwrap_or(""),
        session_id = event.session_id.as_deref().unwrap_or(""),
        regime = regime.as_deref().unwrap_or(""),
        fear = fear.unwrap_or(f64::NAN),
        guidance_action = ?guidance.action,
        take_profit_ratio = thresholds.take_profit_ratio,
        stop_loss_ratio = thresholds.stop_loss_ratio,
        atr_pct = event.atr_pct.unwrap_or(f64::NAN),
        persisted = store.is_enabled(),
        "position event received"
    );
    store.record(&event).await;
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "accepted": true,
            "guidance": guidance,
        })),
    )
}

/// Receive a position-close event (JFLOW-C outcome capture).
///
/// Finalizes this position's accumulated guidance history in the tracker,
/// joins it with the realized outcome, logs and best-effort persists a
/// [`PositionOutcome`], and returns it in the 202 response. The outcome is
/// the closed-trade record the JanusAI service later compacts into
/// `janus_memories` so guidance quality can be evaluated and tuned.
#[tracing::instrument(
    skip(close, store, tracker, state),
    fields(symbol = %close.symbol, side = ?close.side, qty = close.qty)
)]
async fn position_close_handler(
    State(state): State<Arc<JanusState>>,
    Extension(store): Extension<Arc<PositionEventStore>>,
    Extension(tracker): Extension<Arc<PositionTracker>>,
    Json(close): Json<PositionClose>,
) -> impl IntoResponse {
    if let Err(reason) = close.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "accepted": false,
                "error": reason,
            })),
        );
    }

    // Pull (and drop) this position's accumulated guidance history so the
    // outcome can be joined with it. Untracked positions → None.
    let position_state = match &close.position_id {
        Some(id) => tracker.finalize(id).await,
        None => None,
    };
    let outcome = PositionOutcome::from_close(&close, position_state.as_ref());

    // Feed the realized outcome back into affinity learning, if a recorder is
    // installed (forward service) and the producer named the strategy. The
    // affinity tracker is keyed by (strategy, asset), so a close without a
    // strategy is persisted but can't be recorded.
    let recorded = match outcome.strategy.as_deref() {
        Some(strategy) => {
            state
                .record_affinity_outcome(
                    strategy,
                    base_asset(&outcome.symbol),
                    outcome.pnl_realized,
                    outcome.is_winner(),
                    outcome.rr_ratio,
                )
                .await
        }
        None => false,
    };

    // Feed every realized close into the execution gate's consecutive-loss
    // breaker (base-asset keyed, to match the gate's evaluation key). Unlike
    // affinity this is unconditional — the breaker counts losses per asset, not
    // per strategy. No-op if no recorder is installed (e.g. API-only deploys).
    let gate_recorded = state
        .record_gate_outcome(base_asset(&outcome.symbol), outcome.is_winner())
        .await;

    info!(
        symbol = %outcome.symbol,
        side = ?outcome.side,
        pnl_realized = outcome.pnl_realized,
        realized_ratio = outcome.realized_ratio,
        result = outcome.result.as_str(),
        strategy = outcome.strategy.as_deref().unwrap_or(""),
        peak_pnl_ratio = outcome.peak_pnl_ratio.unwrap_or(f64::NAN),
        samples = outcome.samples,
        last_guidance = outcome.last_guidance.map(|g| g.as_str()).unwrap_or(""),
        time_in_position_secs = outcome.time_in_position_secs.unwrap_or(f64::NAN),
        position_id = outcome.position_id.as_deref().unwrap_or(""),
        persisted = store.is_outcomes_enabled(),
        affinity_recorded = recorded,
        gate_recorded,
        "position close received"
    );
    store.record_outcome(&outcome).await;
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "accepted": true,
            "outcome": outcome,
        })),
    )
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

    /// Build the router backed by the given state, with a disabled
    /// position event store and an empty `ParamManager` (so guidance
    /// uses default thresholds). Per-asset threshold paths are
    /// exercised in `position_events::tests` and a dedicated handler
    /// test below.
    fn test_router(state: Arc<JanusState>) -> Router {
        create_router(
            state,
            Arc::new(PositionEventStore::disabled()),
            Arc::new(ParamManager::new("test")),
            Arc::new(PositionTracker::new()),
        )
    }

    /// Build the router with a pre-populated `ParamManager` so we can
    /// assert the optimizer-tuned threshold path. Each `OptimizedParams`
    /// carries its own `asset` field — that's the lookup key.
    async fn test_router_with_params(
        state: Arc<JanusState>,
        params: Vec<janus_core::OptimizedParams>,
    ) -> Router {
        let manager = ParamManager::new("test");
        for p in params {
            manager.update(p).await;
        }
        create_router(
            state,
            Arc::new(PositionEventStore::disabled()),
            Arc::new(manager),
            Arc::new(PositionTracker::new()),
        )
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

    // ── Position event endpoint (JFLOW-C) ────────────────────────────

    #[tokio::test]
    async fn position_event_accepts_valid_payload() {
        let router = test_router(test_state().await);
        let body = serde_json::json!({
            "symbol": "BTC-USD",
            "side": "Buy",
            "qty": 0.5,
            "entry_price": 60000.0,
            "current_price": 61000.0,
            "pnl_unrealized": 500.0,
            "position_id": "pos-1",
            "session_id": "sess-1"
        });
        let (status, value) = post_json(&router, "/api/v1/positions/event", body).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(value["accepted"], serde_json::Value::Bool(true));
        // Notional 30_000, pnl +500 = 1.67% — within bounds, no regime ⇒ hold.
        assert_eq!(value["guidance"]["action"], "hold");
    }

    #[tokio::test]
    async fn position_event_rejects_invalid_payload() {
        let router = test_router(test_state().await);
        let body = serde_json::json!({
            "symbol": "BTC-USD",
            "side": "Buy",
            "qty": -1.0,
            "entry_price": 60000.0,
            "current_price": 61000.0,
            "pnl_unrealized": 0.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/event", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["accepted"], serde_json::Value::Bool(false));
        assert!(value["error"].as_str().unwrap().contains("qty"));
    }

    #[tokio::test]
    async fn position_event_returns_exit_guidance_when_regime_is_crisis() {
        let state = test_state().await;
        state.set_current_regime("crisis_volatility_spike").await;
        let router = test_router(state);
        let body = serde_json::json!({
            "symbol": "BTC-USD",
            "side": "Buy",
            "qty": 0.5,
            "entry_price": 60000.0,
            "current_price": 61000.0,
            "pnl_unrealized": 500.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/event", body).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(value["guidance"]["action"], "exit");
        assert!(
            value["guidance"]["reason"]
                .as_str()
                .unwrap()
                .contains("regime"),
            "reason should mention regime, got: {}",
            value["guidance"]["reason"]
        );
    }

    #[tokio::test]
    async fn position_event_high_fear_forces_exit() {
        // Healthy +500 pnl, but a high amygdala threat forces an exit.
        let state = test_state().await;
        state.set_current_threat(0.9).await;
        let router = test_router(state);
        let body = serde_json::json!({
            "symbol": "BTC-USD",
            "side": "Buy",
            "qty": 0.5,
            "entry_price": 60000.0,
            "current_price": 61000.0,
            "pnl_unrealized": 500.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/event", body).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(value["guidance"]["action"], "exit");
        assert!(
            value["guidance"]["reason"]
                .as_str()
                .unwrap()
                .contains("fear"),
            "reason should mention fear, got: {}",
            value["guidance"]["reason"]
        );
    }

    #[tokio::test]
    async fn position_event_elevated_fear_banks_a_winner() {
        // +500 pnl is under the 5% take-profit, but elevated fear (0.6)
        // banks it early via Reduce.
        let state = test_state().await;
        state.set_current_threat(0.6).await;
        let router = test_router(state);
        let body = serde_json::json!({
            "symbol": "BTC-USD",
            "side": "Buy",
            "qty": 0.5,
            "entry_price": 60000.0,
            "current_price": 61000.0,
            "pnl_unrealized": 500.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/event", body).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(value["guidance"]["action"], "reduce");
    }

    #[tokio::test]
    async fn position_event_uses_per_asset_optimized_take_profit_threshold() {
        // BTC params tuned to take-profit 10% — the producer's snapshot at
        // +5% (1500 / 30_000) should now hold instead of reduce.
        let mut params = janus_core::OptimizedParams::new("BTC");
        params.take_profit_pct = 10.0;
        let router = test_router_with_params(test_state().await, vec![params]).await;
        let body = serde_json::json!({
            "symbol": "BTC-USD",
            "side": "Buy",
            "qty": 0.5,
            "entry_price": 60000.0,
            "current_price": 63000.0,
            "pnl_unrealized": 1500.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/event", body).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(
            value["guidance"]["action"], "hold",
            "tuned 10% take-profit should suppress the default 5% reduce trigger"
        );
    }

    #[tokio::test]
    async fn position_event_volatility_hint_widens_stop() {
        // -800 pnl on 30_000 notional ≈ -2.67%, which trips the default -2%
        // stop. With atr_pct=2.0 and the default 2.0 multiplier, the band is
        // 4%, so the stop widens past the move → hold instead of exit.
        let router = test_router(test_state().await);
        let body = serde_json::json!({
            "symbol": "BTC-USD",
            "side": "Buy",
            "qty": 0.5,
            "entry_price": 60000.0,
            "current_price": 58400.0,
            "pnl_unrealized": -800.0,
            "atr_pct": 2.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/event", body).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(
            value["guidance"]["action"], "hold",
            "volatility band should widen the stop past a -2.67% move"
        );
    }

    #[tokio::test]
    async fn position_event_rejects_invalid_atr_pct() {
        let router = test_router(test_state().await);
        let body = serde_json::json!({
            "symbol": "BTC-USD",
            "side": "Buy",
            "qty": 0.5,
            "entry_price": 60000.0,
            "current_price": 61000.0,
            "pnl_unrealized": 500.0,
            "atr_pct": -1.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/event", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["accepted"], false);
    }

    #[tokio::test]
    async fn position_event_trailing_reduces_a_fading_winner() {
        // Two snapshots for the same position_id hit the same router, so they
        // share one PositionTracker (Extension is Arc-shared across oneshot
        // calls). The position peaks at +10% (a take-profit Reduce), then
        // fades to +2% — a level the stateless rule would Hold, but the
        // trailing give-back rule banks the fading winner.
        let router = test_router(test_state().await);
        let peak = serde_json::json!({
            "symbol": "BTC-USD", "side": "Buy", "qty": 0.5,
            "entry_price": 60000.0, "current_price": 66000.0,
            "pnl_unrealized": 3000.0, "position_id": "pos-trail"
        });
        let (s1, v1) = post_json(&router, "/api/v1/positions/event", peak).await;
        assert_eq!(s1, StatusCode::ACCEPTED);
        assert_eq!(v1["guidance"]["action"], "reduce"); // +10% take-profit

        let giveback = serde_json::json!({
            "symbol": "BTC-USD", "side": "Buy", "qty": 0.5,
            "entry_price": 60000.0, "current_price": 61200.0,
            "pnl_unrealized": 600.0, "position_id": "pos-trail"
        });
        let (s2, v2) = post_json(&router, "/api/v1/positions/event", giveback).await;
        assert_eq!(s2, StatusCode::ACCEPTED);
        assert_eq!(
            v2["guidance"]["action"], "reduce",
            "trailing give-back should bank a fading winner the stateless rule would hold"
        );
        assert!(
            v2["guidance"]["reason"]
                .as_str()
                .unwrap()
                .contains("trailing"),
            "reason was: {}",
            v2["guidance"]["reason"]
        );
    }

    // ── Position close endpoint (outcome capture) ────────────────────

    #[tokio::test]
    async fn position_close_rejects_invalid_payload() {
        let router = test_router(test_state().await);
        let body = serde_json::json!({
            "symbol": "BTC-USD", "side": "Buy", "qty": -1.0,
            "entry_price": 60000.0, "exit_price": 61000.0, "pnl_realized": 100.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/close", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["accepted"], false);
        assert!(value["error"].as_str().unwrap().contains("qty"));
    }

    #[tokio::test]
    async fn position_close_untracked_returns_outcome() {
        // A close for a position we never saw open → outcome with no history.
        let router = test_router(test_state().await);
        let body = serde_json::json!({
            "symbol": "ETH-USD", "side": "Sell", "qty": 2.0,
            "entry_price": 3000.0, "exit_price": 2940.0, "pnl_realized": 120.0
        });
        let (status, value) = post_json(&router, "/api/v1/positions/close", body).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(value["accepted"], true);
        // 120 / 6000 = +2% → win, but nothing was tracked.
        assert_eq!(value["outcome"]["result"], "win");
        assert_eq!(value["outcome"]["samples"], 0);
        assert!(
            value["outcome"].get("last_guidance").is_none(),
            "untracked close should omit guidance history"
        );
    }

    #[tokio::test]
    async fn position_close_joins_tracked_guidance_history() {
        // Two snapshots build history under one position_id, then the close
        // joins the realized outcome with that history (shared tracker).
        let router = test_router(test_state().await);
        for (price, pnl) in [(66000.0, 3000.0), (61200.0, 600.0)] {
            let snap = serde_json::json!({
                "symbol": "BTC-USD", "side": "Buy", "qty": 0.5,
                "entry_price": 60000.0, "current_price": price,
                "pnl_unrealized": pnl, "position_id": "pos-x"
            });
            let (s, _) = post_json(&router, "/api/v1/positions/event", snap).await;
            assert_eq!(s, StatusCode::ACCEPTED);
        }

        let close = serde_json::json!({
            "symbol": "BTC-USD", "side": "Buy", "qty": 0.5,
            "entry_price": 60000.0, "exit_price": 61000.0,
            "pnl_realized": 500.0, "position_id": "pos-x"
        });
        let (status, value) = post_json(&router, "/api/v1/positions/close", close).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let o = &value["outcome"];
        assert_eq!(o["samples"], 2, "both snapshots should be counted");
        assert_eq!(
            o["last_guidance"], "reduce",
            "last advice was a trailing reduce"
        );
        assert_eq!(o["result"], "win"); // +500 realized
        let peak = o["peak_pnl_ratio"].as_f64().expect("peak recorded");
        assert!((peak - 0.10).abs() < 1e-6, "peak was {peak}");
    }

    /// One captured `record_trade` call: (strategy, asset, pnl, is_winner, rr_ratio).
    type RecordedCall = (String, String, f64, bool, Option<f64>);

    /// Captures affinity calls so we can assert the close handler feeds them.
    struct CapturingRecorder {
        calls: Arc<std::sync::Mutex<Vec<RecordedCall>>>,
    }

    #[async_trait::async_trait]
    impl janus_core::AffinityRecorder for CapturingRecorder {
        async fn record_trade(
            &self,
            strategy: &str,
            asset: &str,
            pnl: f64,
            is_winner: bool,
            rr_ratio: Option<f64>,
        ) {
            self.calls.lock().unwrap().push((
                strategy.to_string(),
                asset.to_string(),
                pnl,
                is_winner,
                rr_ratio,
            ));
        }
    }

    #[tokio::test]
    async fn position_close_with_strategy_feeds_affinity() {
        let state = test_state().await;
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        state
            .set_affinity_recorder(Box::new(CapturingRecorder {
                calls: calls.clone(),
            }))
            .await;
        let router = test_router(state);

        let close = serde_json::json!({
            "symbol": "BTC-USD", "side": "Buy", "qty": 0.5,
            "entry_price": 60000.0, "exit_price": 61000.0,
            "pnl_realized": 500.0, "strategy": "ema_cross", "rr_ratio": 2.5
        });
        let (status, value) = post_json(&router, "/api/v1/positions/close", close).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(value["outcome"]["strategy"], "ema_cross");

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "affinity recorder should be called once");
        // base_asset("BTC-USD") == "BTC"; +500 realized = win; rr passed through.
        assert_eq!(
            calls[0],
            (
                "ema_cross".to_string(),
                "BTC".to_string(),
                500.0,
                true,
                Some(2.5)
            )
        );
    }

    #[tokio::test]
    async fn position_close_without_strategy_skips_affinity() {
        let state = test_state().await;
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        state
            .set_affinity_recorder(Box::new(CapturingRecorder {
                calls: calls.clone(),
            }))
            .await;
        let router = test_router(state);

        // No "strategy" field → affinity recording is skipped, close still ok.
        let close = serde_json::json!({
            "symbol": "BTC-USD", "side": "Buy", "qty": 0.5,
            "entry_price": 60000.0, "exit_price": 61000.0, "pnl_realized": 500.0
        });
        let (status, _) = post_json(&router, "/api/v1/positions/close", close).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert!(
            calls.lock().unwrap().is_empty(),
            "no strategy ⇒ no affinity call"
        );
    }

    // ── /sse/bars/{symbol} ───────────────────────────────────────────

    #[tokio::test]
    async fn sse_bars_streams_matching_closed_klines() {
        use futures_util::StreamExt;
        use janus_core::{Exchange, KlineEvent, MarketDataEvent, Symbol};
        use rust_decimal::Decimal;

        let state = test_state().await;
        let router = test_router(state.clone());

        let req = http::Request::builder()
            .method("GET")
            .uri("/sse/bars/BTC-USDT") // separator form must match BTCUSDT
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let kline = |symbol: Symbol, interval: &str, close: i64| {
            MarketDataEvent::Kline(KlineEvent {
                exchange: Exchange::Binance,
                symbol,
                interval: interval.to_string(),
                open_time: 1_700_000_000_000_000,
                close_time: 1_700_000_059_999_000,
                open: Decimal::from(50_000),
                high: Decimal::from(50_100),
                low: Decimal::from(49_900),
                close: Decimal::from(close),
                volume: Decimal::from(10),
                quote_volume: None,
                trades: None,
                is_closed: true,
            })
        };

        // Non-matching events first: wrong symbol, then wrong interval.
        state
            .market_data_bus
            .publish(kline(Symbol::new("ETH", "USDT"), "1m", 1))
            .unwrap();
        state
            .market_data_bus
            .publish(kline(Symbol::new("BTC", "USDT"), "5m", 2))
            .unwrap();
        // The one frame the stream should emit.
        state
            .market_data_bus
            .publish(kline(Symbol::new("BTC", "USDT"), "1m", 50_050))
            .unwrap();

        let mut body = resp.into_body().into_data_stream();
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(2), body.next())
            .await
            .expect("timed out waiting for SSE frame")
            .expect("stream ended unexpectedly")
            .expect("body error");
        let frame = String::from_utf8(chunk.to_vec()).unwrap();

        assert!(frame.contains("event: bar"), "got frame: {frame}");
        assert!(frame.contains("\"close\":50050"), "got frame: {frame}");
        assert!(frame.contains("\"time\":1700000000"), "got frame: {frame}");
        assert!(
            !frame.contains("\"close\":1") && !frame.contains("\"close\":2"),
            "non-matching klines must be filtered out: {frame}"
        );
    }
}
