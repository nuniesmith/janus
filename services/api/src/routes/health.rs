//! Health check routes for the JANUS Rust Gateway.
//!
//! This module provides health, readiness, and liveness endpoints
//! for Docker/Kubernetes health checks and monitoring.
//!
//! Port of `gateway/src/routers/health.py`.

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::state::AppState;

/// Component status for health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub status: String,
}

impl ComponentStatus {
    pub fn connected() -> Self {
        Self {
            status: "connected".to_string(),
        }
    }

    pub fn disconnected() -> Self {
        Self {
            status: "disconnected".to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn unknown() -> Self {
        Self {
            status: "unknown".to_string(),
        }
    }
}

/// Health check response structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub timestamp: String,
    pub forward_service: String,
    pub backward_service: String,
    pub components: HashMap<String, ComponentStatus>,
}

impl Default for HealthResponse {
    fn default() -> Self {
        Self {
            status: "healthy".to_string(),
            service: "janus-gateway".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: Utc::now().to_rfc3339(),
            forward_service: "disconnected".to_string(),
            backward_service: "disconnected".to_string(),
            components: HashMap::new(),
        }
    }
}

/// Readiness check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub timestamp: String,
}

/// Liveness check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessResponse {
    pub alive: bool,
    pub timestamp: String,
}

/// Test hello response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloResponse {
    pub message: String,
    pub status: String,
    pub timestamp: String,
}

/// Health check endpoint handler.
///
/// Returns overall health status including component connectivity.
pub async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut response = HealthResponse::default();

    // Check Redis connection
    let redis_status = if state.signal_dispatcher.is_connected().await {
        ComponentStatus::connected()
    } else {
        ComponentStatus::disconnected()
    };

    // Check gRPC client connection (if available)
    let grpc_status = if state.janus_client.read().await.is_some() {
        response.forward_service = "connected".to_string();
        response.backward_service = "connected".to_string();
        ComponentStatus::connected()
    } else {
        ComponentStatus::disconnected()
    };

    response
        .components
        .insert("redis".to_string(), redis_status);
    response
        .components
        .insert("janus_client".to_string(), grpc_status);

    (StatusCode::OK, Json(response))
}

/// Readiness probe for Kubernetes/Docker health checks.
///
/// Checks if the service is ready to accept traffic.
pub async fn readiness_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Service is ready if Redis is connected
    let ready = state.signal_dispatcher.is_connected().await;

    let response = ReadinessResponse {
        ready,
        timestamp: Utc::now().to_rfc3339(),
    };

    if ready {
        (StatusCode::OK, Json(response))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(response))
    }
}

/// Liveness probe for Kubernetes/Docker health checks.
///
/// Checks if the service is alive.
pub async fn liveness_check() -> impl IntoResponse {
    let response = LivenessResponse {
        alive: true,
        timestamp: Utc::now().to_rfc3339(),
    };

    (StatusCode::OK, Json(response))
}

/// Simple test endpoint for frontend connectivity testing.
pub async fn test_hello() -> impl IntoResponse {
    let response = HelloResponse {
        message: "Hello from FKS Trading API (Rust)".to_string(),
        status: "ok".to_string(),
        timestamp: Utc::now().to_rfc3339(),
    };

    (StatusCode::OK, Json(response))
}

/// Create the health routes router.
pub fn health_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Main health check
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/health/", get(health_check))
        // Readiness probe
        .route("/api/v1/health/ready", get(readiness_check))
        .route("/api/v1/health/ready/", get(readiness_check))
        // Liveness probe
        .route("/api/v1/health/live", get(liveness_check))
        .route("/api/v1/health/live/", get(liveness_check))
        // Test endpoint
        .route("/api/v1/test/hello", get(test_hello))
        .route("/api/v1/test/hello/", get(test_hello))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_default() {
        let response = HealthResponse::default();
        assert_eq!(response.status, "healthy");
        assert_eq!(response.service, "janus-gateway");
        assert!(!response.timestamp.is_empty());
    }

    #[test]
    fn test_component_status() {
        let connected = ComponentStatus::connected();
        assert_eq!(connected.status, "connected");

        let disconnected = ComponentStatus::disconnected();
        assert_eq!(disconnected.status, "disconnected");
    }

    #[test]
    fn test_readiness_response_serialization() {
        let response = ReadinessResponse {
            ready: true,
            timestamp: Utc::now().to_rfc3339(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"ready\":true"));
    }

    #[test]
    fn test_liveness_response_serialization() {
        let response = LivenessResponse {
            alive: true,
            timestamp: Utc::now().to_rfc3339(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"alive\":true"));
    }
}
