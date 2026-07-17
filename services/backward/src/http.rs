//! HTTP server for health checks and metrics endpoints.
//!
//! This module provides a simple HTTP server that exposes:
//! - `/health` - Basic health check endpoint
//! - `/metrics` - Prometheus metrics endpoint
//! - `/api/v1/experiences/sample` - recent experience vectors + payload for
//!   the UMAP "what is janus thinking" view (EXPERIENCE_PIPELINE.md §8)
//! - `/api/v1/training/evals` - the parsed challenger `eval_history.jsonl`
//!   (last N sessions), so Gate-A day quotes numbers from an endpoint instead
//!   of a docker-exec archaeology session

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
        .route("/api/v1/training/evals", get(training_evals_handler))
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

    #[test]
    fn test_tail_evals_bounds_to_last_n_in_order() {
        use crate::tasks::promote::EvalRecord;
        let rec = |unix: u64| EvalRecord {
            unix,
            eval_winrate: 0.5,
            eval_samples: 10,
            eval_mean_reward: 0.0,
            eval_winrate_directional: 0.0,
            eval_samples_directional: 0,
        };
        let records: Vec<EvalRecord> = (1..=5).map(rec).collect();

        // Fewer records than the limit → everything, unchanged order.
        assert_eq!(tail_evals(records.clone(), 100).len(), 5);

        // More records than the limit → the LAST n, chronological order.
        let tail = tail_evals(records, 2);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].unix, 4);
        assert_eq!(tail[1].unix, 5);

        // Empty history → empty response.
        assert!(tail_evals(Vec::new(), 10).is_empty());
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

/// Query parameters for `/api/v1/training/evals`.
#[derive(Debug, serde::Deserialize)]
pub struct EvalsQuery {
    /// Maximum number of most-recent sessions to return (default 100,
    /// clamped to 1..=1000).
    pub limit: Option<usize>,
}

/// Bounded tail of the eval history: the last `limit` records, in
/// chronological (file) order. Pure — unit-tested.
fn tail_evals(
    records: Vec<crate::tasks::promote::EvalRecord>,
    limit: usize,
) -> Vec<crate::tasks::promote::EvalRecord> {
    let skip = records.len().saturating_sub(limit);
    records.into_iter().skip(skip).collect()
}

/// GET /api/v1/training/evals — the parsed challenger `eval_history.jsonl`
/// (see `tasks::promote`), bounded to the last N sessions (default 100).
///
/// Read-only observability: reads the same file the training session appends
/// and the promotion gate reads (`<challenger_dir>/eval_history.jsonl` inside
/// the checkpoints volume). The challenger dir resolves exactly like the
/// trainer's (`JANUS_TRAIN_CHALLENGER_DIR`, defaulting to the canonical
/// challenger subdir). A missing file yields an honest empty list.
async fn training_evals_handler(Query(q): Query<EvalsQuery>) -> Response {
    let limit = q.limit.unwrap_or(100).clamp(1, 1000);
    let challenger_dir = std::env::var(crate::tasks::train::TRAIN_CHALLENGER_DIR_ENV)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| crate::tasks::promote::DEFAULT_CHALLENGER_DIR.to_string());
    let records = crate::tasks::promote::read_eval_history(std::path::Path::new(&challenger_dir));
    let total_recorded = records.len();
    let evals = tail_evals(records, limit);
    Json(json!({
        "challenger_dir": challenger_dir,
        "total_recorded": total_recorded,
        "returned": evals.len(),
        "evals": evals,
    }))
    .into_response()
}
