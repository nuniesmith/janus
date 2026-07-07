//! HTTP server for health checks and metrics endpoints.
//!
//! This module provides a simple HTTP server that exposes:
//! - `/health` - Basic health check endpoint
//! - `/metrics` - Prometheus metrics endpoint
//! - `/api/v1/experiences/sample` - recent experience vectors + payload for
//!   the UMAP "what is janus thinking" view (EXPERIENCE_PIPELINE.md §8)

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
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
    /// Experience store for the sample endpoint. `None` (or mock mode)
    /// degrades /api/v1/experiences/sample to an honest empty response.
    pub store: Option<Arc<crate::persistence::ExperienceStore>>,
}

impl HttpState {
    /// Create new HTTP state
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            start_time: std::time::Instant::now(),
            store: None,
        }
    }

    /// HTTP state with an experience store for the sample endpoint.
    pub fn with_store(store: Option<Arc<crate::persistence::ExperienceStore>>) -> Self {
        Self {
            store,
            ..Self::new()
        }
    }
}

impl Default for HttpState {
    fn default() -> Self {
        Self::new()
    }
}

/// Start HTTP server for health and metrics
pub async fn start_http_server(port: u16, state: HttpState) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route(
            "/api/v1/experiences/sample",
            get(experiences_sample_handler),
        )
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
        "service": "backward",
        "version": state.version,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": uptime_secs,
        "state": "sleep",
        "description": "Janus Backward Service - Memory consolidation and training"
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

/// Query parameters for `/api/v1/experiences/sample`.
#[derive(Debug, serde::Deserialize)]
pub struct SampleQuery {
    pub limit: Option<usize>,
    pub symbol: Option<String>,
    pub action: Option<u8>,
    pub since_ms: Option<i64>,
}

/// GET /api/v1/experiences/sample — recent experience vectors + payload,
/// newest-first, for the UMAP view. Degrades to `{points: [], total: 0}`
/// when no store is wired, mock mode is active, or Qdrant errors — the view
/// renders an honest empty state instead of breaking.
async fn experiences_sample_handler(
    State(state): State<Arc<HttpState>>,
    Query(q): Query<SampleQuery>,
) -> Response {
    let limit = q.limit.unwrap_or(2000).clamp(1, 5000);
    let Some(store) = state.store.as_ref() else {
        return Json(json!({ "points": [], "total": 0 })).into_response();
    };
    match store
        .sample(limit, q.symbol.as_deref(), q.action, q.since_ms)
        .await
    {
        Ok((points, total)) => Json(json!({ "points": points, "total": total })).into_response(),
        Err(e) => {
            error!("experiences/sample failed: {e:#}");
            Json(json!({ "points": [], "total": 0 })).into_response()
        }
    }
}
