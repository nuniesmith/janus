//! Health Check and Metrics HTTP Server
//!
//! Provides HTTP endpoints for health checks and Prometheus metrics scraping.
//!
//! # Endpoints
//!
//! - `GET /health` - Basic health check (returns 200 if healthy)
//! - `GET /health/detailed` - Detailed health status with component info
//! - `GET /metrics` - Prometheus metrics endpoint
//! - `GET /status` - Service status including last optimization info
//!
//! # Usage
//!
//! ```rust,ignore
//! let server = HealthServer::new(state);
//! server.run().await?;
//! ```

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::AppState;

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub healthy: bool,
    pub uptime_seconds: f64,
    pub version: String,
}

/// Detailed health response with component status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedHealthResponse {
    pub status: String,
    pub healthy: bool,
    pub uptime_seconds: f64,
    pub version: String,
    pub components: ComponentStatus,
    pub config: ConfigSummary,
    pub last_optimization: Option<OptimizationStatusResponse>,
}

/// Component health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub database: ComponentHealth,
    pub redis: ComponentHealth,
    pub scheduler: ComponentHealth,
}

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: String,
    pub healthy: bool,
    pub message: Option<String>,
}

/// Configuration summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    pub assets: Vec<String>,
    pub optimization_interval: String,
    pub n_trials: usize,
    pub data_collection_enabled: bool,
    pub metrics_port: u16,
}

/// Last optimization status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatusResponse {
    pub started_at: String,
    pub completed_at: Option<String>,
    pub assets_optimized: Vec<String>,
    pub assets_failed: Vec<String>,
    pub duration_seconds: Option<f64>,
}

/// Service status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatusResponse {
    pub service: String,
    pub version: String,
    pub healthy: bool,
    pub uptime_seconds: f64,
    pub assets: Vec<String>,
    pub optimization_interval: String,
    pub next_optimization: Option<String>,
    pub last_optimization: Option<OptimizationStatusResponse>,
    pub metrics_port: u16,
}

/// Health server
pub struct HealthServer {
    state: Arc<AppState>,
}

impl HealthServer {
    /// Create a new health server
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    /// Run the health server
    pub async fn run(self) -> anyhow::Result<()> {
        let port = self.state.config.metrics_port;
        let addr = format!("0.0.0.0:{}", port);

        info!("Starting health/metrics server on {}", addr);

        // Create router
        let app = Router::new()
            .route("/health", get(health_check))
            .route("/health/detailed", get(detailed_health))
            .route("/metrics", get(metrics))
            .route("/status", get(service_status))
            .route("/", get(root))
            .layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
            .with_state(self.state.clone());

        // Create listener
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        // Create shutdown signal
        let mut shutdown_rx = self.state.shutdown_tx.subscribe();

        // Run server with graceful shutdown
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.recv().await;
                info!("Health server shutting down");
            })
            .await?;

        Ok(())
    }
}

/// Root endpoint - basic info
async fn root() -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "janus-optimizer",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": [
            "/health",
            "/health/detailed",
            "/metrics",
            "/status"
        ]
    }))
}

/// Basic health check endpoint
async fn health_check(State(state): State<Arc<AppState>>) -> Response {
    let healthy = *state.healthy.read().await;
    let uptime = state.metrics.uptime();

    let response = HealthResponse {
        status: if healthy {
            "ok".to_string()
        } else {
            "unhealthy".to_string()
        },
        healthy,
        uptime_seconds: uptime,
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let status = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status, Json(response)).into_response()
}

/// Detailed health check endpoint
async fn detailed_health(State(state): State<Arc<AppState>>) -> Response {
    let healthy = *state.healthy.read().await;
    let uptime = state.metrics.uptime();

    // Check components
    let db_healthy = true; // SQLite is always available if we got this far
    let redis_healthy = check_redis(&state.config.redis_url).await;
    let scheduler_healthy = healthy; // Assume scheduler is healthy if service is healthy

    let overall_healthy = healthy && db_healthy;

    let components = ComponentStatus {
        database: ComponentHealth {
            status: if db_healthy {
                "ok".to_string()
            } else {
                "error".to_string()
            },
            healthy: db_healthy,
            message: None,
        },
        redis: ComponentHealth {
            status: if redis_healthy {
                "ok".to_string()
            } else {
                "degraded".to_string()
            },
            healthy: redis_healthy,
            message: if !redis_healthy {
                Some("Redis not connected - operating in file-only mode".to_string())
            } else {
                None
            },
        },
        scheduler: ComponentHealth {
            status: if scheduler_healthy {
                "ok".to_string()
            } else {
                "error".to_string()
            },
            healthy: scheduler_healthy,
            message: None,
        },
    };

    let config = ConfigSummary {
        assets: state.config.assets.clone(),
        optimization_interval: state.config.optimization_interval.clone(),
        n_trials: state.config.n_trials,
        data_collection_enabled: state.config.data_collection_enabled,
        metrics_port: state.config.metrics_port,
    };

    let last_optimization =
        state
            .last_optimization
            .read()
            .await
            .as_ref()
            .map(|opt| OptimizationStatusResponse {
                started_at: opt.started_at.to_rfc3339(),
                completed_at: opt.completed_at.map(|t| t.to_rfc3339()),
                assets_optimized: opt.assets_optimized.clone(),
                assets_failed: opt.assets_failed.clone(),
                duration_seconds: opt.total_duration_secs,
            });

    let response = DetailedHealthResponse {
        status: if overall_healthy {
            "ok".to_string()
        } else {
            "degraded".to_string()
        },
        healthy: overall_healthy,
        uptime_seconds: uptime,
        version: env!("CARGO_PKG_VERSION").to_string(),
        components,
        config,
        last_optimization,
    };

    let status = if overall_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status, Json(response)).into_response()
}

/// Prometheus metrics endpoint
async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let metrics_text = state.metrics.get_metrics_text();

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        metrics_text,
    )
        .into_response()
}

/// Service status endpoint
async fn service_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let healthy = *state.healthy.read().await;
    let uptime = state.metrics.uptime();

    let last_optimization =
        state
            .last_optimization
            .read()
            .await
            .as_ref()
            .map(|opt| OptimizationStatusResponse {
                started_at: opt.started_at.to_rfc3339(),
                completed_at: opt.completed_at.map(|t| t.to_rfc3339()),
                assets_optimized: opt.assets_optimized.clone(),
                assets_failed: opt.assets_failed.clone(),
                duration_seconds: opt.total_duration_secs,
            });

    // Calculate next optimization time (approximate)
    let next_optimization = state.config.optimization_duration().ok().map(|dur| {
        let now = chrono::Utc::now();
        let next = now + chrono::Duration::from_std(dur).unwrap_or_default();
        next.to_rfc3339()
    });

    let response = ServiceStatusResponse {
        service: "janus-optimizer".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        healthy,
        uptime_seconds: uptime,
        assets: state.config.assets.clone(),
        optimization_interval: state.config.optimization_interval.clone(),
        next_optimization,
        last_optimization,
        metrics_port: state.config.metrics_port,
    };

    Json(response)
}

/// Check Redis connectivity
async fn check_redis(redis_url: &str) -> bool {
    match redis::Client::open(redis_url) {
        Ok(client) => {
            matches!(
                tokio::time::timeout(
                    Duration::from_secs(2),
                    client.get_multiplexed_async_connection(),
                )
                .await,
                Ok(Ok(_))
            )
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            healthy: true,
            uptime_seconds: 123.45,
            version: "0.1.0".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"healthy\":true"));
    }

    #[test]
    fn test_component_health() {
        let health = ComponentHealth {
            status: "ok".to_string(),
            healthy: true,
            message: None,
        };

        assert!(health.healthy);
        assert!(health.message.is_none());
    }

    #[test]
    fn test_config_summary() {
        let config = ConfigSummary {
            assets: vec!["BTC".to_string(), "ETH".to_string()],
            optimization_interval: "6h".to_string(),
            n_trials: 100,
            data_collection_enabled: true,
            metrics_port: 9092,
        };

        assert_eq!(config.assets.len(), 2);
        assert_eq!(config.n_trials, 100);
    }
}
