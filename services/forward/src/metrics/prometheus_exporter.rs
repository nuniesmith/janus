//! # Prometheus Exporter
//!
//! HTTP endpoint for exposing Prometheus metrics from JANUS service.

use crate::metrics::JanusMetrics;
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use prometheus::{Encoder, TextEncoder};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{error, info};

/// Prometheus exporter for JANUS metrics
pub struct PrometheusExporter {
    metrics: Arc<JanusMetrics>,
    port: u16,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(metrics: Arc<JanusMetrics>, port: u16) -> Self {
        Self { metrics, port }
    }

    /// Start the Prometheus metrics server
    pub async fn start(self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("0.0.0.0:{}", self.port);
        info!("Starting Prometheus metrics server on {}", addr);

        let app = self.create_router();

        let listener = TcpListener::bind(&addr).await?;
        info!("Prometheus metrics server listening on {}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Create the metrics router
    fn create_router(self) -> Router {
        let metrics = self.metrics;

        Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler))
            .with_state(metrics)
    }
}

/// Handler for /metrics endpoint
async fn metrics_handler(State(metrics): State<Arc<JanusMetrics>>) -> Response {
    // Gather all metrics
    let metric_families = metrics.gather();

    // Encode to Prometheus text format
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();

    match encoder.encode(&metric_families, &mut buffer) {
        Ok(_) => {
            // Return metrics as text/plain
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
                .body(Body::from(buffer))
                .unwrap_or_else(|e| {
                    error!("Failed to build metrics response: {}", e);
                    Response::new(Body::from("Internal Server Error"))
                })
        }
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

/// Handler for /health endpoint
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// Start Prometheus exporter in background task
pub async fn start_metrics_server(
    metrics: Arc<JanusMetrics>,
    port: u16,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let exporter = PrometheusExporter::new(metrics, port);
        if let Err(e) = exporter.start().await {
            error!("Metrics server error: {}", e);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exporter_creation() {
        let metrics = Arc::new(JanusMetrics::default());
        let exporter = PrometheusExporter::new(metrics, 9090);
        assert_eq!(exporter.port, 9090);
    }

    #[tokio::test]
    async fn test_metrics_handler() {
        let metrics = Arc::new(JanusMetrics::default());

        // Record some test metrics
        metrics.system_metrics().record_http_request(0.1);
        metrics
            .signal_metrics()
            .record_signal_generated("BTC/USD", "1h", "BUY", 0.85, 0.75);

        let response = metrics_handler(State(metrics)).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
