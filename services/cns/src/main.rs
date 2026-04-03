//! # JANUS CNS Service
//!
//! The Central Nervous System service for JANUS trading system.
//! Provides health monitoring, metrics collection, and auto-recovery.

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use janus_cns::{Brain, BrainConfig, CNSMetrics, HealthCheckResponse, NeuromorphicBrain};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::signal;
use tracing::{error, info, warn};

/// Application state
struct AppState {
    brain: Arc<Brain>,
    neuromorphic_brain: Arc<tokio::sync::RwLock<NeuromorphicBrain>>,
    start_time: Instant,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("🧠 JANUS Central Nervous System (CNS) starting...");

    // Load configuration
    let config = load_config()?;
    info!("Configuration loaded successfully");

    // Create brain
    let brain = Arc::new(Brain::new(config.clone()));
    info!("Brain initialized with {} probes", brain.probe_count());

    // Create neuromorphic brain coordinator
    let neuromorphic_brain = Arc::new(tokio::sync::RwLock::new(NeuromorphicBrain::new()));
    info!("🧠 Neuromorphic brain coordinator initialized");

    // Create application state
    let state = Arc::new(AppState {
        brain: brain.clone(),
        neuromorphic_brain: neuromorphic_brain.clone(),
        start_time: Instant::now(),
    });

    // Build HTTP server
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/health/detailed", get(detailed_health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/status", get(status_handler))
        .route("/brain", get(brain_status_handler))
        .route("/brain/regions", get(brain_regions_handler))
        .route("/", get(root_handler))
        .with_state(state);

    // Determine bind address from environment or default to 9090
    let port = std::env::var("SERVICE_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(9090);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("🌐 Starting HTTP server on {}", addr);

    // Start brain in background
    let brain_handle = brain.clone();
    tokio::spawn(async move {
        if let Err(e) = brain_handle.start().await {
            error!("Brain error: {}", e);
        }
    });

    // Start HTTP server
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("✅ CNS service ready");
    info!("📊 Metrics: http://{}/metrics", addr);
    info!("🏥 Health: http://{}/health", addr);

    // Run server
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("🛑 CNS service shutdown complete");
    Ok(())
}

/// Load configuration from file or use defaults
fn load_config() -> anyhow::Result<BrainConfig> {
    // Try to load from config file
    match std::fs::read_to_string("config/cns.toml") {
        Ok(contents) => {
            info!("Loading configuration from config/cns.toml");
            let config: BrainConfig = toml::from_str(&contents)?;
            Ok(config)
        }
        Err(e) => {
            warn!("Could not load config file ({}), using defaults", e);
            Ok(BrainConfig::default())
        }
    }
}

/// Root handler - service information
async fn root_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let info = ServiceInfo {
        name: "JANUS Central Nervous System".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        endpoints: vec![
            "/health - Basic health check".to_string(),
            "/health/detailed - Detailed health information".to_string(),
            "/metrics - Prometheus metrics".to_string(),
            "/status - Service status".to_string(),
        ],
    };

    Json(info)
}

/// Health check handler
async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.brain.get_health().await {
        Some(signal) => {
            if signal.system_status.is_operational() {
                (StatusCode::OK, Json(signal)).into_response()
            } else {
                (StatusCode::SERVICE_UNAVAILABLE, Json(signal)).into_response()
            }
        }
        None => {
            // No health data yet
            let response = serde_json::json!({
                "status": "starting",
                "message": "Health checks initializing"
            });
            (StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response()
        }
    }
}

/// Detailed health check handler
async fn detailed_health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let start = Instant::now();

    // Perform immediate health check
    match state.brain.check_now().await {
        Ok(signal) => {
            let response = HealthCheckResponse {
                signal,
                processing_time_ms: start.elapsed().as_millis() as u64,
            };

            if response.signal.system_status.is_operational() {
                (StatusCode::OK, Json(response)).into_response()
            } else {
                (StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response()
            }
        }
        Err(e) => {
            error!("Health check failed: {}", e);
            let error_response = serde_json::json!({
                "status": "error",
                "message": format!("Health check failed: {}", e)
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
        }
    }
}

/// Metrics handler - Prometheus format
async fn metrics_handler() -> impl IntoResponse {
    match CNSMetrics::encode_text() {
        Ok(metrics) => (
            StatusCode::OK,
            [("Content-Type", "text/plain; version=0.0.4")],
            metrics,
        ),
        Err(e) => {
            error!("Failed to encode metrics: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [("Content-Type", "text/plain")],
                format!("Error encoding metrics: {}", e),
            )
        }
    }
}

/// Status handler - simple service status
async fn status_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let is_running = state.brain.is_running().await;
    let uptime = state.brain.uptime_seconds();

    let status = ServiceStatus {
        service: "janus-cns".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        running: is_running,
        uptime_seconds: uptime,
    };

    if is_running {
        (StatusCode::OK, Json(status))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(status))
    }
}

/// Brain status handler - neuromorphic brain coordinator status
async fn brain_status_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let brain = state.neuromorphic_brain.read().await;
    let summary = brain.health_summary();

    Ok(Json(serde_json::json!({
        "neuromorphic_brain": {
            "active": summary.active,
            "total_regions": summary.total_regions,
            "initialized_count": summary.initialized_count,
            "healthy_count": summary.healthy_count,
            "degraded_count": summary.degraded_count,
            "down_count": summary.down_count,
            "initialization_progress": summary.initialization_progress,
            "health_score": summary.health_score,
            "fully_initialized": brain.is_fully_initialized(),
        }
    })))
}

/// Brain regions handler - detailed status of each brain region
async fn brain_regions_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let brain = state.neuromorphic_brain.read().await;
    let statuses = brain.get_all_statuses();
    let init_order = brain.initialization_order();

    Ok(Json(serde_json::json!({
        "regions": statuses,
        "initialization_order": init_order.iter().map(|r| r.to_string()).collect::<Vec<_>>(),
    })))
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C signal");
        },
        _ = terminate => {
            info!("Received terminate signal");
        },
    }

    info!("🛑 Initiating graceful shutdown...");
}

/// Service information response
#[derive(Debug, Serialize, Deserialize)]
struct ServiceInfo {
    name: String,
    version: String,
    uptime_seconds: u64,
    endpoints: Vec<String>,
}

/// Service status response
#[derive(Debug, Serialize, Deserialize)]
struct ServiceStatus {
    service: String,
    version: String,
    running: bool,
    uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_loading() {
        // Should fall back to defaults if file doesn't exist
        let config = load_config().unwrap();
        assert!(config.health_check_interval_secs > 0);
    }
}
