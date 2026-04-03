//! # Health Check and Metrics Module
//!
//! Production-grade health checks and Prometheus metrics for Project JANUS.
//!
//! ## Features
//!
//! - HTTP health check endpoints (`/health`, `/ready`)
//! - Prometheus metrics endpoint (`/metrics`)
//! - System state tracking
//! - Automatic metric collection
//! - Thread-safe counters and gauges
//! - Graceful shutdown coordination
//! - Dead man's switch for emergency position closing
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_health::{HealthServer, SystemHealth, Metrics};
//! use janus_health::shutdown::ShutdownCoordinator;
//!
//! // Start health check server
//! let health = SystemHealth::new();
//! let metrics = Arc::new(Metrics::new()?);
//! let server = HealthServer::new(8080, health.clone(), metrics.clone());
//! tokio::spawn(server.serve());
//!
//! // Update health status
//! health.mark_ready();
//! health.update_websocket_status(true);
//!
//! // Record metrics
//! metrics.ticks_total.inc();
//! metrics.signals_total.inc();
//!
//! // Setup graceful shutdown
//! let coordinator = ShutdownCoordinator::new();
//! coordinator.register_hook("cleanup", Box::new(|| {
//!     Box::pin(async { Ok(()) })
//! })).await;
//! ```

pub mod shutdown;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use chrono::{DateTime, Utc};
use prometheus::{Encoder, Gauge, IntCounter, IntGauge, Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: DateTime<Utc>,
    pub uptime_seconds: u64,
    pub components: ComponentStatus,
}

/// Individual component status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub websocket: bool,
    pub database: bool,
    pub strategy: bool,
    pub compliance: bool,
}

/// Shared system health state
#[derive(Clone)]
pub struct SystemHealth {
    inner: Arc<RwLock<SystemHealthInner>>,
}

struct SystemHealthInner {
    start_time: DateTime<Utc>,
    ready: bool,
    components: ComponentStatus,
}

impl SystemHealth {
    /// Create a new system health tracker
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(SystemHealthInner {
                start_time: Utc::now(),
                ready: false,
                components: ComponentStatus {
                    websocket: false,
                    database: false,
                    strategy: false,
                    compliance: false,
                },
            })),
        }
    }

    /// Mark system as ready
    pub async fn mark_ready(&self) {
        let mut inner = self.inner.write().await;
        inner.ready = true;
    }

    /// Mark system as not ready
    pub async fn mark_not_ready(&self) {
        let mut inner = self.inner.write().await;
        inner.ready = false;
    }

    /// Check if system is ready
    pub async fn is_ready(&self) -> bool {
        self.inner.read().await.ready
    }

    /// Update WebSocket connection status
    pub async fn update_websocket_status(&self, connected: bool) {
        let mut inner = self.inner.write().await;
        inner.components.websocket = connected;
    }

    /// Update database connection status
    pub async fn update_database_status(&self, connected: bool) {
        let mut inner = self.inner.write().await;
        inner.components.database = connected;
    }

    /// Update strategy status
    pub async fn update_strategy_status(&self, active: bool) {
        let mut inner = self.inner.write().await;
        inner.components.strategy = active;
    }

    /// Update compliance status
    pub async fn update_compliance_status(&self, active: bool) {
        let mut inner = self.inner.write().await;
        inner.components.compliance = active;
    }

    /// Get current health status
    pub async fn get_status(&self) -> HealthStatus {
        let inner = self.inner.read().await;
        let uptime = (Utc::now() - inner.start_time).num_seconds() as u64;

        HealthStatus {
            status: if inner.ready { "healthy" } else { "starting" }.to_string(),
            timestamp: Utc::now(),
            uptime_seconds: uptime,
            components: inner.components.clone(),
        }
    }

    /// Get readiness status (for k8s readiness probe)
    pub async fn get_readiness(&self) -> bool {
        let inner = self.inner.read().await;
        inner.ready
            && inner.components.websocket
            && inner.components.database
            && inner.components.strategy
    }
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self::new()
    }
}

/// Prometheus metrics
pub struct Metrics {
    registry: Registry,

    // Counters
    pub ticks_total: IntCounter,
    pub signals_total: IntCounter,
    pub orders_placed: IntCounter,
    pub orders_filled: IntCounter,
    pub orders_rejected: IntCounter,
    pub websocket_reconnects: IntCounter,
    pub database_writes: IntCounter,
    pub errors_total: IntCounter,

    // Gauges
    pub websocket_connected: IntGauge,
    pub database_connected: IntGauge,
    pub open_positions: IntGauge,
    pub account_balance: Gauge,
    pub unrealized_pnl: Gauge,
    pub tick_processing_duration_ms: Gauge,
}

impl Metrics {
    /// Create new metrics registry
    pub fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();

        // Counters
        let ticks_total = IntCounter::new("janus_ticks_total", "Total ticks processed")?;
        registry.register(Box::new(ticks_total.clone()))?;

        let signals_total = IntCounter::new("janus_signals_total", "Total signals generated")?;
        registry.register(Box::new(signals_total.clone()))?;

        let orders_placed = IntCounter::new("janus_orders_placed_total", "Total orders placed")?;
        registry.register(Box::new(orders_placed.clone()))?;

        let orders_filled = IntCounter::new("janus_orders_filled_total", "Total orders filled")?;
        registry.register(Box::new(orders_filled.clone()))?;

        let orders_rejected =
            IntCounter::new("janus_orders_rejected_total", "Total orders rejected")?;
        registry.register(Box::new(orders_rejected.clone()))?;

        let websocket_reconnects = IntCounter::new(
            "janus_websocket_reconnects_total",
            "Total WebSocket reconnections",
        )?;
        registry.register(Box::new(websocket_reconnects.clone()))?;

        let database_writes =
            IntCounter::new("janus_database_writes_total", "Total database writes")?;
        registry.register(Box::new(database_writes.clone()))?;

        let errors_total = IntCounter::new("janus_errors_total", "Total errors encountered")?;
        registry.register(Box::new(errors_total.clone()))?;

        // Gauges
        let websocket_connected =
            IntGauge::new("janus_websocket_connected", "WebSocket connection status")?;
        registry.register(Box::new(websocket_connected.clone()))?;

        let database_connected =
            IntGauge::new("janus_database_connected", "Database connection status")?;
        registry.register(Box::new(database_connected.clone()))?;

        let open_positions = IntGauge::new("janus_open_positions", "Number of open positions")?;
        registry.register(Box::new(open_positions.clone()))?;

        let account_balance = Gauge::new("janus_account_balance_usdt", "Account balance in USDT")?;
        registry.register(Box::new(account_balance.clone()))?;

        let unrealized_pnl = Gauge::new("janus_unrealized_pnl_usdt", "Unrealized P&L in USDT")?;
        registry.register(Box::new(unrealized_pnl.clone()))?;

        let tick_processing_duration_ms = Gauge::new(
            "janus_tick_processing_duration_ms",
            "Tick processing duration in milliseconds",
        )?;
        registry.register(Box::new(tick_processing_duration_ms.clone()))?;

        Ok(Self {
            registry,
            ticks_total,
            signals_total,
            orders_placed,
            orders_filled,
            orders_rejected,
            websocket_reconnects,
            database_writes,
            errors_total,
            websocket_connected,
            database_connected,
            open_positions,
            account_balance,
            unrealized_pnl,
            tick_processing_duration_ms,
        })
    }

    /// Encode metrics as Prometheus text format
    pub fn encode(&self) -> anyhow::Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new().expect("Failed to create metrics")
    }
}

/// Health check HTTP server
pub struct HealthServer {
    port: u16,
    health: SystemHealth,
    metrics: Arc<Metrics>,
}

impl HealthServer {
    /// Create a new health check server
    pub fn new(port: u16, health: SystemHealth, metrics: Arc<Metrics>) -> Self {
        Self {
            port,
            health,
            metrics,
        }
    }

    /// Start the HTTP server
    pub async fn serve(self) -> anyhow::Result<()> {
        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/ready", get(ready_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(AppState {
                health: self.health,
                metrics: self.metrics,
            });

        let addr = format!("0.0.0.0:{}", self.port);
        info!("🏥 Health check server listening on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

#[derive(Clone)]
struct AppState {
    health: SystemHealth,
    metrics: Arc<Metrics>,
}

/// Health check endpoint handler
async fn health_handler(State(state): State<AppState>) -> Response {
    let status = state.health.get_status().await;

    if status.status == "healthy" {
        (StatusCode::OK, Json(status)).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(status)).into_response()
    }
}

/// Readiness check endpoint handler
async fn ready_handler(State(state): State<AppState>) -> Response {
    let ready = state.health.get_readiness().await;

    if ready {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "ready": true,
                "timestamp": Utc::now()
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false,
                "timestamp": Utc::now()
            })),
        )
            .into_response()
    }
}

/// Prometheus metrics endpoint handler
async fn metrics_handler(State(state): State<AppState>) -> Response {
    match state.metrics.encode() {
        Ok(metrics) => (StatusCode::OK, metrics).into_response(),
        Err(e) => {
            error!("Failed to encode metrics: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics",
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_health_creation() {
        let health = SystemHealth::new();
        assert!(!health.is_ready().await);
    }

    #[tokio::test]
    async fn test_mark_ready() {
        let health = SystemHealth::new();
        health.mark_ready().await;
        assert!(health.is_ready().await);
    }

    #[tokio::test]
    async fn test_component_status() {
        let health = SystemHealth::new();

        health.update_websocket_status(true).await;
        health.update_database_status(true).await;

        let status = health.get_status().await;
        assert!(status.components.websocket);
        assert!(status.components.database);
    }

    #[tokio::test]
    async fn test_readiness() {
        let health = SystemHealth::new();

        // Not ready initially
        assert!(!health.get_readiness().await);

        // Mark components as ready
        health.mark_ready().await;
        health.update_websocket_status(true).await;
        health.update_database_status(true).await;
        health.update_strategy_status(true).await;

        // Should be ready now
        assert!(health.get_readiness().await);
    }

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new().unwrap();

        // Increment some counters
        metrics.ticks_total.inc();
        metrics.signals_total.inc();

        // Encode metrics
        let encoded = metrics.encode().unwrap();
        assert!(encoded.contains("janus_ticks_total"));
        assert!(encoded.contains("janus_signals_total"));
    }

    #[test]
    fn test_metrics_encode() {
        let metrics = Metrics::new().unwrap();

        metrics.ticks_total.inc_by(100);
        metrics.orders_placed.inc_by(5);
        metrics.account_balance.set(10000.0);

        let encoded = metrics.encode().unwrap();
        assert!(encoded.contains("janus_ticks_total 100"));
        assert!(encoded.contains("janus_orders_placed_total 5"));
        assert!(encoded.contains("janus_account_balance_usdt 10000"));
    }
}
