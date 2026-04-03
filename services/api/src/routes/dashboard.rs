//! Dashboard API routes for JANUS Rust Gateway.
//!
//! Provides endpoints for dashboard data aggregation and real-time updates.

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    routing::get,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

use crate::state::AppState;

/// Dashboard overview response.
#[derive(Debug, Serialize)]
struct DashboardOverview {
    portfolio_value: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    assets: Vec<AssetInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    signals: Vec<SignalInfo>,
    timestamp: String,
    system_status: SystemStatus,
    message: Option<String>,
}

/// Asset information.
#[derive(Debug, Serialize)]
struct AssetInfo {
    symbol: String,
    value: f64,
    quantity: f64,
    price: f64,
    change_24h: f64,
}

/// Signal information.
#[derive(Debug, Serialize)]
struct SignalInfo {
    symbol: String,
    signal_type: String,
    confidence: f64,
    timestamp: String,
}

/// System status information.
#[derive(Debug, Serialize)]
struct SystemStatus {
    status: String,
    active_services: u32,
    uptime_seconds: u64,
    latency_ms: f64,
    redis_connected: bool,
    grpc_connected: bool,
}

/// Performance metrics response.
#[derive(Debug, Serialize)]
struct PerformanceMetrics {
    period_days: u32,
    total_return: f64,
    total_return_pct: f64,
    sharpe_ratio: Option<f64>,
    max_drawdown: Option<f64>,
    win_rate: Option<f64>,
    assets: Vec<AssetPerformance>,
    message: Option<String>,
}

/// Asset performance data.
#[derive(Debug, Serialize)]
struct AssetPerformance {
    symbol: String,
    returns: f64,
    returns_pct: f64,
    volatility: f64,
}

/// Signal summary response.
#[derive(Debug, Serialize)]
struct SignalSummary {
    category: Option<String>,
    total_signals: u32,
    by_category: std::collections::HashMap<String, u32>,
    recent_signals: Vec<SignalInfo>,
    message: Option<String>,
}

/// Correlation matrix response.
#[derive(Debug, Serialize)]
struct CorrelationMatrix {
    symbols: Vec<String>,
    period_days: u32,
    matrix: Vec<Vec<f64>>,
    message: Option<String>,
}

/// Price chart response.
#[derive(Debug, Serialize)]
struct PriceChart {
    symbol: String,
    period_days: u32,
    data: Vec<PricePoint>,
    message: Option<String>,
}

/// Price point in time series.
#[derive(Debug, Serialize)]
struct PricePoint {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

/// Query parameters for performance endpoint.
#[derive(Debug, Deserialize)]
struct PerformanceQuery {
    #[serde(default = "default_days")]
    days: u32,
}

fn default_days() -> u32 {
    30
}

/// Query parameters for signal summary.
#[derive(Debug, Deserialize)]
struct SignalSummaryQuery {
    category: Option<String>,
}

/// Query parameters for correlation matrix.
#[derive(Debug, Deserialize)]
struct CorrelationQuery {
    symbols: Option<String>,
    #[serde(default = "default_correlation_days")]
    days: u32,
}

fn default_correlation_days() -> u32 {
    90
}

/// Query parameters for price chart.
#[derive(Debug, Deserialize)]
struct PriceChartQuery {
    #[serde(default = "default_days")]
    days: u32,
}

/// Get dashboard overview data.
///
/// Returns portfolio overview with assets, prices, and system status.
async fn get_dashboard_overview(
    State(state): State<Arc<AppState>>,
) -> Result<Json<DashboardOverview>, StatusCode> {
    info!("Dashboard overview requested");

    // Get system metrics
    let uptime = state.metrics.start_time.elapsed().as_secs();
    let redis_connected = state.signal_dispatcher.is_connected().await;
    let grpc_connected = state.janus_client.read().await.is_some();

    // Calculate average latency from recorded HTTP request durations
    let avg_latency = state.metrics.avg_request_latency_ms().unwrap_or(0.0);

    // Build system status
    let system_status = SystemStatus {
        status: if redis_connected && grpc_connected {
            "healthy".to_string()
        } else if redis_connected || grpc_connected {
            "degraded".to_string()
        } else {
            "unavailable".to_string()
        },
        active_services: if redis_connected { 1 } else { 0 } + if grpc_connected { 1 } else { 0 },
        uptime_seconds: uptime,
        latency_ms: avg_latency,
        redis_connected,
        grpc_connected,
    };

    // Portfolio data requires wiring to the forward service's position state
    // and QuestDB for historical asset values. Until then, return system
    // status (which is already live) with empty portfolio data.
    let response = DashboardOverview {
        portfolio_value: 0.0,
        assets: vec![],
        signals: vec![],
        timestamp: Utc::now().to_rfc3339(),
        system_status,
        message: Some(
            "Portfolio data populates once forward service position state is wired".to_string(),
        ),
    };

    Ok(Json(response))
}

/// Get portfolio performance metrics.
///
/// Returns performance metrics for tracked assets over the specified period.
async fn get_portfolio_performance(
    State(_state): State<Arc<AppState>>,
    Query(query): Query<PerformanceQuery>,
) -> Result<Json<PerformanceMetrics>, StatusCode> {
    info!("Portfolio performance requested: days={}", query.days);

    // Performance metrics require a QuestDB HTTP query client to read
    // from the `executions` table written by the forward service event
    // loop. Wire a reqwest-based QuestDB reader into AppState to enable
    // this (query: SELECT * FROM executions WHERE timestamp > dateadd(...)).
    let response = PerformanceMetrics {
        period_days: query.days,
        total_return: 0.0,
        total_return_pct: 0.0,
        sharpe_ratio: None,
        max_drawdown: None,
        win_rate: None,
        assets: vec![],
        message: Some(
            "Requires QuestDB HTTP reader for execution history — wire into AppState".to_string(),
        ),
    };

    Ok(Json(response))
}

/// Get signal summary for dashboard.
///
/// Returns signal summary by category.
async fn get_signal_summary(
    State(_state): State<Arc<AppState>>,
    Query(query): Query<SignalSummaryQuery>,
) -> Result<Json<SignalSummary>, StatusCode> {
    info!("Signal summary requested: category={:?}", query.category);

    // Signal summary can be sourced from Redis (recent signals via pub/sub
    // replay or a sorted set) or QuestDB (SELECT ... FROM signals WHERE ...).
    // Requires adding a QuestDB HTTP reader or Redis XRANGE query to AppState.
    let response = SignalSummary {
        category: query.category,
        total_signals: 0,
        by_category: std::collections::HashMap::new(),
        recent_signals: vec![],
        message: Some("Requires Redis XRANGE or QuestDB reader for signal history".to_string()),
    };

    Ok(Json(response))
}

/// Get asset correlation matrix.
///
/// Returns correlation matrix data for the specified symbols and period.
async fn get_correlation_matrix(
    State(_state): State<Arc<AppState>>,
    Query(query): Query<CorrelationQuery>,
) -> Result<Json<CorrelationMatrix>, StatusCode> {
    let symbol_list = query
        .symbols
        .as_ref()
        .map(|s| s.split(',').map(|sym| sym.trim().to_uppercase()).collect())
        .unwrap_or_else(Vec::new);

    info!(
        "Correlation matrix requested: symbols={:?}, days={}",
        symbol_list, query.days
    );

    // Correlation matrix can be computed from the `ticks` or `trades` tables
    // in QuestDB. Use the janus-risk CorrelationTracker or a direct QuestDB
    // query over the requested period. Requires a QuestDB HTTP reader in AppState.
    let response = CorrelationMatrix {
        symbols: symbol_list,
        period_days: query.days,
        matrix: vec![],
        message: Some(
            "Requires QuestDB reader to compute cross-asset correlation from tick data".to_string(),
        ),
    };

    Ok(Json(response))
}

/// Get price chart data for a symbol.
///
/// Returns price chart data for the specified symbol and period.
async fn get_price_chart(
    State(_state): State<Arc<AppState>>,
    axum::extract::Path(symbol): axum::extract::Path<String>,
    Query(query): Query<PriceChartQuery>,
) -> Result<Json<PriceChart>, StatusCode> {
    let symbol = symbol.to_uppercase();

    info!(
        "Price chart requested: symbol={}, days={}",
        symbol, query.days
    );

    // OHLCV chart data lives in the QuestDB `ticks` / `trades` tables.
    // Query with SAMPLE BY to aggregate into candles at the desired
    // resolution (e.g., 1h, 4h, 1d). Requires a QuestDB HTTP reader.
    let response = PriceChart {
        symbol: symbol.clone(),
        period_days: query.days,
        data: vec![],
        message: Some(
            "Requires QuestDB reader — query trades table with SAMPLE BY for OHLCV".to_string(),
        ),
    };

    Ok(Json(response))
}

/// Build and return the dashboard routes.
pub fn dashboard_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/dashboard/overview", get(get_dashboard_overview))
        .route("/api/dashboard/performance", get(get_portfolio_performance))
        .route("/api/dashboard/signals/summary", get(get_signal_summary))
        .route("/api/dashboard/correlation", get(get_correlation_matrix))
        .route("/api/dashboard/charts/price/{symbol}", get(get_price_chart))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;
    use crate::redis_dispatcher::SignalDispatcher;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_dashboard_overview() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = dashboard_routes().with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/dashboard/overview")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_performance_metrics() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = dashboard_routes().with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/dashboard/performance?days=30")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_signal_summary() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = dashboard_routes().with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/dashboard/signals/summary")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
