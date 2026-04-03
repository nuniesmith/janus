//! HTTP Admin Endpoints for FKS Execution Service
//!
//! Provides REST API for health checks, metrics, and administrative operations.

use crate::execution::histogram::global_latency_histograms;
use crate::notifications::NotificationManager;
use crate::orders::OrderManager;
use crate::sim::{DataRecorder, LiveFeedBridge, sim_prometheus_metrics};
use crate::types::OrderSide;
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// HTTP server state
#[derive(Clone)]
pub struct HttpState {
    /// Order manager
    pub order_manager: Arc<OrderManager>,
    /// Notification manager (optional)
    pub notification_manager: Option<Arc<NotificationManager>>,
    /// Data recorder for sim metrics (optional)
    pub data_recorder: Option<Arc<DataRecorder>>,
    /// Live feed bridge for sim metrics (optional)
    pub live_feed_bridge: Option<Arc<LiveFeedBridge>>,
}

impl HttpState {
    /// Create a new HttpState with required components
    pub fn new(order_manager: Arc<OrderManager>) -> Self {
        Self {
            order_manager,
            notification_manager: None,
            data_recorder: None,
            live_feed_bridge: None,
        }
    }

    /// Set the notification manager
    pub fn with_notification_manager(mut self, manager: Arc<NotificationManager>) -> Self {
        self.notification_manager = Some(manager);
        self
    }

    /// Set the data recorder for sim metrics
    pub fn with_data_recorder(mut self, recorder: Arc<DataRecorder>) -> Self {
        self.data_recorder = Some(recorder);
        self
    }

    /// Set the live feed bridge for sim metrics
    pub fn with_live_feed_bridge(mut self, bridge: Arc<LiveFeedBridge>) -> Self {
        self.live_feed_bridge = Some(bridge);
        self
    }
}

/// Create the HTTP router with all endpoints
pub fn create_router(state: HttpState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/health/ready", get(readiness_handler))
        .route("/health/live", get(liveness_handler))
        .route("/metrics", get(metrics_handler))
        .route("/sim/metrics", get(sim_metrics_handler))
        .route("/api/v1/orders", get(list_orders_handler))
        .route("/api/v1/orders/{order_id}", get(get_order_handler))
        .route(
            "/api/v1/orders/{order_id}/cancel",
            post(cancel_order_handler),
        )
        .route("/api/v1/orders/cancel-all", post(cancel_all_orders_handler))
        .route("/api/v1/stats", get(stats_handler))
        .route("/api/v1/admin/config", get(get_config_handler))
        // Discord notification endpoints
        .route(
            "/api/v1/notifications/test",
            post(test_notification_handler),
        )
        .route(
            "/api/v1/notifications/status",
            get(notification_status_handler),
        )
        .with_state(state)
}

// ============================================================================
// Health Check Endpoints
// ============================================================================

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    timestamp: i64,
    version: String,
    components: HashMap<String, ComponentHealth>,
}

#[derive(Debug, Serialize)]
struct ComponentHealth {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<serde_json::Value>,
}

/// GET /health - Overall health check
async fn health_handler(State(state): State<HttpState>) -> Result<Json<HealthResponse>, AppError> {
    let mut components = HashMap::new();

    // Check order manager
    let stats = state.order_manager.get_statistics();
    components.insert(
        "order_manager".to_string(),
        ComponentHealth {
            status: "UP".to_string(),
            message: Some(format!(
                "Active: {}, Completed: {}",
                stats.active_orders, stats.completed_orders
            )),
            details: Some(serde_json::json!({
                "active_orders": stats.active_orders,
                "completed_orders": stats.completed_orders,
                "total_tracked": stats.total_orders_tracked,
            })),
        },
    );

    // Overall status
    let all_healthy = components.values().all(|c| c.status == "UP");
    let status = if all_healthy { "UP" } else { "DOWN" };

    Ok(Json(HealthResponse {
        status: status.to_string(),
        timestamp: chrono::Utc::now().timestamp_millis(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        components,
    }))
}

/// GET /health/ready - Readiness probe
async fn readiness_handler(State(state): State<HttpState>) -> Response {
    // Check if service is ready to accept requests
    let _stats = state.order_manager.get_statistics();

    // Service is ready if order manager is accessible
    (StatusCode::OK, "READY").into_response()
}

/// GET /health/live - Liveness probe
async fn liveness_handler() -> Response {
    // Service is alive if it can respond
    (StatusCode::OK, "ALIVE").into_response()
}

// ============================================================================
// Metrics Endpoint
// ============================================================================

/// Metrics response (Prometheus format)
#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct MetricsResponse {
    metrics: Vec<Metric>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct Metric {
    name: String,
    help: String,
    r#type: String,
    value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<HashMap<String, String>>,
}

/// GET /metrics - Prometheus-compatible metrics
async fn metrics_handler(State(state): State<HttpState>) -> String {
    let stats = state.order_manager.get_statistics();
    let now = chrono::Utc::now().timestamp_millis();

    let mut output = String::new();

    // Order metrics
    output.push_str("# HELP execution_orders_active Number of active orders\n");
    output.push_str("# TYPE execution_orders_active gauge\n");
    output.push_str(&format!(
        "execution_orders_active {}\n",
        stats.active_orders
    ));
    output.push('\n');

    output.push_str("# HELP execution_orders_completed Number of completed orders\n");
    output.push_str("# TYPE execution_orders_completed gauge\n");
    output.push_str(&format!(
        "execution_orders_completed {}\n",
        stats.completed_orders
    ));
    output.push('\n');

    output.push_str("# HELP execution_orders_total Total number of orders tracked\n");
    output.push_str("# TYPE execution_orders_total counter\n");
    output.push_str(&format!(
        "execution_orders_total {}\n",
        stats.total_orders_tracked
    ));
    output.push('\n');

    // Service metrics
    output.push_str("# HELP execution_service_info Service information\n");
    output.push_str("# TYPE execution_service_info gauge\n");
    output.push_str(&format!(
        "execution_service_info{{version=\"{}\"}} 1\n",
        env!("CARGO_PKG_VERSION")
    ));
    output.push('\n');

    output.push_str("# HELP execution_service_timestamp_ms Current timestamp in milliseconds\n");
    output.push_str("# TYPE execution_service_timestamp_ms gauge\n");
    output.push_str(&format!("execution_service_timestamp_ms {}\n", now));
    output.push('\n');

    // Exchange WebSocket metrics
    output.push_str(&crate::exchanges::prometheus_metrics());
    output.push('\n');

    // Execution subsystem metrics (retry, arbitrage, best-execution, signal-flow)
    output.push_str(&crate::execution::execution_prometheus_metrics());
    output.push('\n');

    // Update sim metrics from live components before export
    if let Some(ref recorder) = state.data_recorder {
        crate::sim::update_recorder_stats(&recorder.stats());
    }
    if let Some(ref bridge) = state.live_feed_bridge {
        crate::sim::update_bridge_stats(&bridge.stats());
    }

    // Simulation metrics (DataRecorder, LiveFeedBridge, Replay)
    let sim_metrics = sim_prometheus_metrics();
    if !sim_metrics.is_empty() {
        output.push_str(
            "# ============================================================================\n",
        );
        output.push_str("# Simulation Metrics (Recorder, Bridge, Replay)\n");
        output.push_str(
            "# ============================================================================\n",
        );
        output.push_str(&sim_metrics);
    }

    output
}

/// GET /sim/metrics - Dedicated simulation metrics endpoint
///
/// Returns only simulation-related metrics (DataRecorder, LiveFeedBridge, Replay)
/// in Prometheus format. Useful for separate scraping configuration.
pub async fn sim_metrics_handler(State(state): State<HttpState>) -> String {
    // Update metrics from live components before export
    if let Some(ref recorder) = state.data_recorder {
        crate::sim::update_recorder_stats(&recorder.stats());
    }
    if let Some(ref bridge) = state.live_feed_bridge {
        crate::sim::update_bridge_stats(&bridge.stats());
    }

    sim_prometheus_metrics()
}

// ============================================================================
// Discord Notification Endpoints
// ============================================================================

/// Notification status response
#[derive(Debug, Serialize)]
struct NotificationStatusResponse {
    enabled: bool,
    discord_configured: bool,
    message: String,
}

/// GET /api/v1/notifications/status - Get notification configuration status
async fn notification_status_handler(
    State(state): State<HttpState>,
) -> Result<Json<NotificationStatusResponse>, AppError> {
    let (enabled, discord_configured) = match &state.notification_manager {
        Some(manager) => (manager.is_enabled(), manager.is_enabled()),
        None => (false, false),
    };

    let message = if enabled {
        "Discord notifications are enabled and configured".to_string()
    } else if discord_configured {
        "Discord is configured but notifications are disabled".to_string()
    } else {
        "Discord notifications are not configured. Set DISCORD_WEBHOOK_GENERAL and DISCORD_ENABLE_NOTIFICATIONS=true".to_string()
    };

    Ok(Json(NotificationStatusResponse {
        enabled,
        discord_configured,
        message,
    }))
}

/// Test notification request
#[derive(Debug, Deserialize)]
struct TestNotificationRequest {
    #[serde(default = "default_message")]
    message: String,
    #[serde(default = "default_notification_type")]
    notification_type: String,
}

fn default_message() -> String {
    "This is a test notification from FKS Execution Service".to_string()
}

fn default_notification_type() -> String {
    "signal".to_string()
}

/// POST /api/v1/notifications/test - Send a test notification
async fn test_notification_handler(
    State(state): State<HttpState>,
    Json(req): Json<TestNotificationRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    info!("Test notification request: type={}", req.notification_type);

    let manager = match &state.notification_manager {
        Some(m) => m,
        None => {
            return Ok(Json(serde_json::json!({
                "success": false,
                "error": "Notification manager not configured",
                "hint": "Set DISCORD_WEBHOOK_GENERAL and DISCORD_ENABLE_NOTIFICATIONS=true"
            })));
        }
    };

    if !manager.is_enabled() {
        return Ok(Json(serde_json::json!({
            "success": false,
            "error": "Notifications are disabled",
            "hint": "Set DISCORD_ENABLE_NOTIFICATIONS=true and provide DISCORD_WEBHOOK_GENERAL"
        })));
    }

    let result = match req.notification_type.as_str() {
        "signal" => {
            manager
                .notify_signal_received(
                    "TEST-SIGNAL-001",
                    "BTC/USDT",
                    OrderSide::Buy,
                    0.001,
                    Some(50000.0),
                    0.85,
                    "test-strategy",
                )
                .await
        }
        "error" => {
            manager
                .notify_error("Test Error", &req.message, "Test notification endpoint")
                .await
        }
        "summary" => manager.notify_daily_summary(10, 7, 3, 150.50, 0.70).await,
        _ => {
            manager
                .notify_error("Test Notification", &req.message, "Manual test from API")
                .await
        }
    };

    match result {
        Ok(()) => {
            info!("Test notification sent successfully");
            Ok(Json(serde_json::json!({
                "success": true,
                "message": "Test notification sent successfully",
                "notification_type": req.notification_type,
                "timestamp": chrono::Utc::now().to_rfc3339()
            })))
        }
        Err(e) => {
            warn!("Failed to send test notification: {}", e);
            Ok(Json(serde_json::json!({
                "success": false,
                "error": format!("Failed to send notification: {}", e),
                "notification_type": req.notification_type
            })))
        }
    }
}

// ============================================================================
// Order Management Endpoints
// ============================================================================

#[derive(Debug, Deserialize)]
struct ListOrdersQuery {
    #[serde(default)]
    symbol: Option<String>,
    #[serde(default)]
    exchange: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    status: Option<String>,
}

#[derive(Debug, Serialize)]
struct OrderSummary {
    id: String,
    exchange_order_id: Option<String>,
    symbol: String,
    exchange: String,
    side: String,
    order_type: String,
    status: String,
    quantity: String,
    filled_quantity: String,
    price: Option<String>,
    created_at: i64,
}

/// GET /api/v1/orders - List orders
async fn list_orders_handler(
    State(state): State<HttpState>,
    Query(query): Query<ListOrdersQuery>,
) -> Result<Json<Vec<OrderSummary>>, AppError> {
    let start = Instant::now();
    let histograms = global_latency_histograms();
    let mut orders = state.order_manager.get_active_orders();

    // Apply filters
    if let Some(ref symbol) = query.symbol {
        orders.retain(|o| o.symbol == *symbol);
    }
    if let Some(ref exchange) = query.exchange {
        orders.retain(|o| o.exchange == *exchange);
    }

    let summaries: Vec<OrderSummary> = orders
        .iter()
        .map(|o| OrderSummary {
            id: o.id.clone(),
            exchange_order_id: o.exchange_order_id.clone(),
            symbol: o.symbol.clone(),
            exchange: o.exchange.clone(),
            side: format!("{:?}", o.side),
            order_type: format!("{:?}", o.order_type),
            status: format!("{:?}", o.status),
            quantity: o.quantity.to_string(),
            filled_quantity: o.filled_quantity.to_string(),
            price: o.price.map(|p| p.to_string()),
            created_at: o.created_at.timestamp_millis(),
        })
        .collect();

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    histograms.record_api_call("http_list_orders", duration_ms);
    Ok(Json(summaries))
}

/// GET /api/v1/orders/:order_id - Get order details
async fn get_order_handler(
    State(state): State<HttpState>,
    Path(order_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let start = Instant::now();
    let histograms = global_latency_histograms();
    let order = state
        .order_manager
        .get_order(&order_id)
        .map_err(|e| AppError::NotFound(e.to_string()))?;

    // Convert to JSON
    let json = serde_json::json!({
        "id": order.id,
        "exchange_order_id": order.exchange_order_id,
        "signal_id": order.signal_id,
        "symbol": order.symbol,
        "exchange": order.exchange,
        "side": format!("{:?}", order.side),
        "order_type": format!("{:?}", order.order_type),
        "status": format!("{:?}", order.status),
        "quantity": order.quantity.to_string(),
        "filled_quantity": order.filled_quantity.to_string(),
        "remaining_quantity": order.remaining_quantity.to_string(),
        "price": order.price.map(|p| p.to_string()),
        "average_fill_price": order.average_fill_price.map(|p| p.to_string()),
        "time_in_force": format!("{:?}", order.time_in_force),
        "strategy": format!("{:?}", order.strategy),
        "created_at": order.created_at.to_rfc3339(),
        "updated_at": order.updated_at.to_rfc3339(),
        "fills": order.fills.len(),
    });

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    histograms.record_api_call("http_get_order", duration_ms);
    Ok(Json(json))
}

/// POST /api/v1/orders/:order_id/cancel - Cancel order
async fn cancel_order_handler(
    State(state): State<HttpState>,
    Path(order_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let start = Instant::now();
    let histograms = global_latency_histograms();
    info!(order_id = %order_id, "Cancel order request");

    state
        .order_manager
        .cancel_order(&order_id)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    histograms.record_order_cancellation("http", duration_ms);
    histograms.record_api_call("http_cancel_order", duration_ms);
    Ok(Json(serde_json::json!({
        "success": true,
        "order_id": order_id,
        "message": "Order cancelled successfully",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })))
}

#[derive(Debug, Deserialize)]
struct CancelAllRequest {
    #[serde(default)]
    symbol: Option<String>,
    #[serde(default)]
    exchange: Option<String>,
}

/// POST /api/v1/orders/cancel-all - Cancel all orders
async fn cancel_all_orders_handler(
    State(state): State<HttpState>,
    Json(req): Json<CancelAllRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let start = Instant::now();
    let histograms = global_latency_histograms();
    info!(
        symbol = ?req.symbol,
        exchange = ?req.exchange,
        "Cancel all orders request"
    );

    let mut orders = state.order_manager.get_active_orders();

    // Apply filters
    if let Some(ref symbol) = req.symbol {
        orders.retain(|o| o.symbol == *symbol);
    }
    if let Some(ref exchange) = req.exchange {
        orders.retain(|o| o.exchange == *exchange);
    }

    let mut cancelled_count = 0;
    let mut errors = Vec::new();

    for order in orders {
        match state.order_manager.cancel_order(&order.id).await {
            Ok(()) => {
                cancelled_count += 1;
            }
            Err(e) => {
                errors.push(format!("{}: {}", order.id, e));
            }
        }
    }

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    histograms.record_api_call("http_cancel_all_orders", duration_ms);
    Ok(Json(serde_json::json!({
        "success": true,
        "cancelled_count": cancelled_count,
        "errors": errors,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })))
}

// ============================================================================
// Statistics Endpoint
// ============================================================================

/// GET /api/v1/stats - Get service statistics
async fn stats_handler(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let stats = state.order_manager.get_statistics();

    Ok(Json(serde_json::json!({
        "orders": {
            "active": stats.active_orders,
            "completed": stats.completed_orders,
            "total_tracked": stats.total_orders_tracked,
        },
        "service": {
            "version": env!("CARGO_PKG_VERSION"),
            "name": env!("CARGO_PKG_NAME"),
            "uptime_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        },
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })))
}

// ============================================================================
// Configuration Endpoint
// ============================================================================

/// GET /api/v1/admin/config - Get current configuration
async fn get_config_handler(
    State(_state): State<HttpState>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Return safe config info (no secrets)
    Ok(Json(serde_json::json!({
        "service": {
            "version": env!("CARGO_PKG_VERSION"),
            "name": env!("CARGO_PKG_NAME"),
        },
        "features": {
            "order_management": true,
            "exchange_integration": true,
            "validation": true,
            "tracking": true,
            "persistence": true,
        },
        "limits": {
            "max_completed_cache": 1000,
        },
    })))
}

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug)]
enum AppError {
    NotFound(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = serde_json::json!({
            "error": message,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });

        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_error_response() {
        let err = AppError::NotFound("Order not found".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
