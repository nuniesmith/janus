//! JANUS Rust Gateway Service
//!
//! High-performance API gateway for Project JANUS, replacing the Python FastAPI gateway.
//!
//! This service provides:
//! - REST API endpoints for signal management
//! - gRPC-Web proxy for Kotlin/JS clients (via tonic-web)
//! - Redis Pub/Sub for signal dispatch to Forward service
//! - Health checks for Docker/Kubernetes
//! - Dead Man's Switch heartbeat
//! - Prometheus metrics
//! - Rate limiting

use axum::{
    Json, Router,
    extract::Request,
    http::{Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use chrono::Utc;
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

mod config;
mod grpc;
mod metrics;
mod rate_limit;
mod redis_dispatcher;
mod routes;
mod state;

use config::Settings;

use rate_limit::{RateLimitConfig, RateLimitState, rate_limit_middleware};
use redis_dispatcher::SignalDispatcher;
use routes::{dashboard_routes, health_routes, metrics_routes, setup_routes, signal_routes};
use state::AppState;

/// Root endpoint response.
#[derive(Debug, Serialize)]
struct RootResponse {
    service: String,
    version: String,
    description: String,
    status: String,
    timestamp: String,
    features: Vec<String>,
}

/// Root endpoint handler.
async fn root() -> impl IntoResponse {
    let response = RootResponse {
        service: "janus-gateway".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "Rust Gateway for Project JANUS".to_string(),
        status: "running".to_string(),
        timestamp: Utc::now().to_rfc3339(),
        features: vec![
            "rest-api".to_string(),
            "redis-pubsub".to_string(),
            "prometheus-metrics".to_string(),
            "rate-limiting".to_string(),
            "grpc-web".to_string(),
        ],
    };

    (StatusCode::OK, Json(response))
}

/// Proxy headers middleware for handling X-Forwarded-* headers from reverse proxy.
///
/// This ensures that redirects use the correct protocol (https) when behind nginx.
async fn proxy_headers_middleware(request: Request, next: Next) -> Response {
    // Extract forwarded headers
    let _forwarded_proto = request
        .headers()
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let _forwarded_host = request
        .headers()
        .get("x-forwarded-host")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // For now, just pass through. Full implementation would modify
    // response URLs for redirects based on these headers.
    next.run(request).await
}

/// Metrics recording middleware.
///
/// Records request duration and status code for Prometheus metrics.
async fn metrics_middleware(state: Arc<AppState>, request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let start = Instant::now();

    // Increment in-flight counter
    state.metrics.http_requests_in_flight.inc();

    // Process request
    let response = next.run(request).await;

    // Record metrics
    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16();

    state
        .metrics
        .record_request(&method, &path, status, duration);

    // Decrement in-flight counter
    state.metrics.http_requests_in_flight.dec();

    response
}

/// Build CORS layer based on settings.
fn build_cors_layer(settings: &Settings) -> CorsLayer {
    let origins = settings.cors_origins_list();

    if origins.len() == 1 && origins[0] == "*" {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::DELETE,
                Method::OPTIONS,
            ])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
            .allow_credentials(false) // Can't use credentials with Any origin
    } else {
        let allowed_origins: Vec<_> = origins.iter().filter_map(|o| o.parse().ok()).collect();

        CorsLayer::new()
            .allow_origin(allowed_origins)
            .allow_methods([
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::DELETE,
                Method::OPTIONS,
            ])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
            .allow_credentials(true)
    }
}

/// Build the application router with all routes and middleware.
fn build_router(state: Arc<AppState>) -> Router {
    let settings = &state.settings;
    let cors = build_cors_layer(settings);

    // Create rate limit state
    let rate_limit_config = if settings.is_production() {
        RateLimitConfig::new(100, 50) // 100 req/s with burst of 50 in production
    } else {
        RateLimitConfig::new(1000, 500) // Higher limits for development
    };
    let rate_limit_state = RateLimitState::new(rate_limit_config);

    // Clone state for metrics middleware
    let metrics_state = state.clone();

    Router::new()
        // Root endpoint
        .route("/", get(root))
        // Health alias for Docker health checks
        .route("/health", get(routes::health::health_check))
        // Merge route modules
        .merge(health_routes())
        .merge(signal_routes())
        .merge(dashboard_routes())
        .merge(setup_routes())
        .merge(metrics_routes())
        // Add middleware (order matters - first added is outermost)
        .layer(middleware::from_fn(proxy_headers_middleware))
        .layer(middleware::from_fn_with_state(
            rate_limit_state,
            rate_limit_middleware,
        ))
        .layer(middleware::from_fn(move |req, next| {
            let state = metrics_state.clone();
            metrics_middleware(state, req, next)
        }))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        // Add shared state
        .with_state(state)
}

/// Initialize tracing/logging.
fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tower_http=debug"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_level(true))
        .with(filter)
        .init();
}

/// Graceful shutdown signal handler.
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, initiating graceful shutdown");
        }
        _ = terminate => {
            info!("Received SIGTERM, initiating graceful shutdown");
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    init_tracing();

    info!(
        "Starting Janus Gateway Service (Rust) v{}",
        env!("CARGO_PKG_VERSION")
    );

    // Load configuration
    let settings = Settings::from_env();
    info!(
        "Configuration loaded: service={}, port={}, env={}",
        settings.service_name, settings.port, settings.environment
    );

    // Initialize Redis signal dispatcher
    let signal_dispatcher = Arc::new(SignalDispatcher::new(&settings.redis_signal_url));

    match signal_dispatcher.connect().await {
        Ok(_) => info!("Signal dispatcher connected to Redis"),
        Err(e) => {
            warn!("Could not connect signal dispatcher to Redis: {}", e);
            // Continue anyway - dispatcher will retry or operate in degraded mode
        }
    }

    // Create application state (includes metrics)
    let state = Arc::new(AppState::new(settings.clone(), signal_dispatcher.clone()));

    // Update initial connection status in metrics
    let redis_connected = state.signal_dispatcher.is_connected().await;
    let grpc_connected = state.janus_client.read().await.is_some();
    state
        .metrics
        .update_connection_status(redis_connected, grpc_connected);

    // Connect to Janus gRPC services (optional - may not be running)
    match state.connect_janus_client().await {
        Ok(_) => {
            info!("Connected to Janus Rust services");
            state
                .metrics
                .update_connection_status(redis_connected, true);
        }
        Err(e) => warn!("Could not connect to Janus services: {}", e),
    }

    // Start heartbeat task (Dead Man's Switch)
    state.start_heartbeat(signal_dispatcher.clone()).await;

    // Build router
    let app = build_router(state.clone());

    // Bind address
    let addr = SocketAddr::from(([0, 0, 0, 0], settings.port));
    info!("Listening on {}", addr);
    info!("Metrics available at http://{}/metrics", addr);

    // Create listener
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Serve with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;

            // Perform cleanup
            info!("Shutting down...");
            state.shutdown().await;
            info!("Janus Gateway shutdown complete");
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_root_endpoint() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = build_router(state);

        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_cors_layer_wildcard() {
        let settings = Settings {
            cors_origins: "*".to_string(),
            ..Settings::from_env()
        };

        // Should not panic
        let _cors = build_cors_layer(&settings);
    }

    #[test]
    fn test_cors_layer_specific_origins() {
        let settings = Settings {
            cors_origins: "http://localhost:3000,http://localhost:8080".to_string(),
            ..Settings::from_env()
        };

        // Should not panic
        let _cors = build_cors_layer(&settings);
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
