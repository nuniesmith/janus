//! HTTP server for health checks and metrics endpoints.
//!
//! This module provides a simple HTTP server that exposes:
//! - `/health` - Basic health check endpoint
//! - `/metrics` - Prometheus metrics endpoint

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;
use std::sync::Arc;
use tracing::{error, info};

/// HTTP server state
#[derive(Clone)]
pub struct HttpState {
    /// Service version
    pub version: String,
    /// Service start time
    pub start_time: std::time::Instant,
}

impl HttpState {
    /// Create new HTTP state
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            start_time: std::time::Instant::now(),
        }
    }
}

/// Start HTTP server for health and metrics
pub async fn start_http_server(
    port: u16,
    state: HttpState,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(Arc::new(state));

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting HTTP server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check handler
async fn health_handler(State(state): State<Arc<HttpState>>) -> Response {
    let uptime_secs = state.start_time.elapsed().as_secs();

    let response = json!({
        "status": "healthy",
        "service": "forward",
        "version": state.version,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": uptime_secs,
        "state": "wake",
        "description": "Janus Forward Service - Real-time trading execution"
    });

    (StatusCode::OK, Json(response)).into_response()
}

/// Metrics handler - Prometheus format
async fn metrics_handler() -> Response {
    match cns::metrics::CNSMetrics::encode_text() {
        Ok(metrics) => (
            StatusCode::OK,
            [("Content-Type", "text/plain; version=0.0.4")],
            metrics,
        )
            .into_response(),
        Err(e) => {
            error!("Failed to encode metrics: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [("Content-Type", "text/plain")],
                format!("Error encoding metrics: {}", e),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_http_state_creation() {
        let state = HttpState::new();
        assert!(!state.version.is_empty());
        assert!(state.start_time.elapsed().as_secs() < 1);
    }
}
