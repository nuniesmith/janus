//! JANUS Asset Registry Service
//!
//! REST API service for asset registry and service discovery.
//! Provides endpoints for:
//! - Asset management (CRUD operations)
//! - Service registration and discovery
//! - Health monitoring
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                JANUS Registry Service                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
//! │  │    Asset     │  │   Service    │  │   Health     │      │
//! │  │  Endpoints   │  │  Endpoints   │  │  Endpoints   │      │
//! │  └──────────────┘  └──────────────┘  └──────────────┘      │
//! │                                                              │
//! │  ┌────────────────────────────────────────────────────┐     │
//! │  │              Registry Manager                       │     │
//! │  │         (janus-registry-lib)                       │     │
//! │  └────────────────────────────────────────────────────┘     │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post, put},
};
use janus_registry_lib::{
    Asset, AssetStatus, AssetType, RegistryError, RegistryManager, ServiceEndpoint, ServiceHealth,
    ServiceInstance, ServiceType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    registry: Arc<RegistryManager>,
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct CreateAssetRequest {
    id: String,
    name: String,
    symbol: String,
    asset_type: AssetType,
    #[serde(default)]
    price_precision: Option<u8>,
    #[serde(default)]
    quantity_precision: Option<u8>,
    #[serde(default)]
    min_order_size: Option<f64>,
    #[serde(default)]
    exchanges: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UpdateAssetStatusRequest {
    status: AssetStatus,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegisterServiceRequest {
    name: String,
    service_type: ServiceType,
    version: String,
    protocol: String,
    host: String,
    port: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct UpdateServiceHealthRequest {
    health: ServiceHealth,
}

#[derive(Debug, Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn error(message: String) -> ApiResponse<()> {
        ApiResponse {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    service: String,
    version: String,
    status: String,
    asset_count: usize,
    service_count: usize,
}

// ============================================================================
// Asset Handlers
// ============================================================================

/// List all assets
async fn list_assets(State(state): State<AppState>) -> impl IntoResponse {
    let assets = state.registry.assets.list().await;
    Json(ApiResponse::success(assets))
}

/// Get asset by ID
async fn get_asset(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.registry.assets.get(&id).await {
        Ok(asset) => (StatusCode::OK, Json(ApiResponse::success(asset))).into_response(),
        Err(RegistryError::AssetNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!("Asset not found: {}", id))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Get asset by symbol
async fn get_asset_by_symbol(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
) -> impl IntoResponse {
    match state.registry.assets.get_by_symbol(&symbol).await {
        Ok(asset) => (StatusCode::OK, Json(ApiResponse::success(asset))).into_response(),
        Err(RegistryError::AssetNotFound(sym)) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!(
                "Asset not found: {}",
                sym
            ))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Create a new asset
async fn create_asset(
    State(state): State<AppState>,
    Json(req): Json<CreateAssetRequest>,
) -> impl IntoResponse {
    let mut asset = Asset::new(&req.id, &req.name, &req.symbol, req.asset_type);

    if let Some(precision) = req.price_precision {
        asset.price_precision = precision;
    }
    if let Some(precision) = req.quantity_precision {
        asset.quantity_precision = precision;
    }
    if let Some(size) = req.min_order_size {
        asset.min_order_size = size;
    }
    for exchange in req.exchanges {
        asset.add_exchange(&exchange);
    }

    match state.registry.assets.register(asset.clone()).await {
        Ok(()) => (StatusCode::CREATED, Json(ApiResponse::success(asset))).into_response(),
        Err(RegistryError::DuplicateAsset(id)) => (
            StatusCode::CONFLICT,
            Json(ApiResponse::<()>::error(format!("Duplicate asset: {}", id))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Update asset status
async fn update_asset_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateAssetStatusRequest>,
) -> impl IntoResponse {
    match state.registry.assets.update_status(&id, req.status).await {
        Ok(()) => match state.registry.assets.get(&id).await {
            Ok(asset) => (StatusCode::OK, Json(ApiResponse::success(asset))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<()>::error(e.to_string())),
            )
                .into_response(),
        },
        Err(RegistryError::AssetNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!("Asset not found: {}", id))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Delete an asset
async fn delete_asset(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.registry.assets.remove(&id).await {
        Ok(asset) => (StatusCode::OK, Json(ApiResponse::success(asset))).into_response(),
        Err(RegistryError::AssetNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!("Asset not found: {}", id))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// List assets by type
async fn list_assets_by_type(
    State(state): State<AppState>,
    Path(asset_type): Path<String>,
) -> impl IntoResponse {
    let asset_type = match asset_type.to_lowercase().as_str() {
        "crypto" => AssetType::Crypto,
        "forex" => AssetType::Forex,
        "equity" => AssetType::Equity,
        "futures" => AssetType::Futures,
        "options" => AssetType::Options,
        "commodity" => AssetType::Commodity,
        "index" => AssetType::Index,
        _ => AssetType::Other,
    };

    let assets = state.registry.assets.list_by_type(asset_type).await;
    Json(ApiResponse::success(assets))
}

// ============================================================================
// Service Discovery Handlers
// ============================================================================

/// List all services
async fn list_services(State(state): State<AppState>) -> impl IntoResponse {
    let services = state.registry.services.list().await;
    Json(ApiResponse::success(services))
}

/// Get service by ID
async fn get_service(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.registry.services.get(&id).await {
        Ok(service) => (StatusCode::OK, Json(ApiResponse::success(service))).into_response(),
        Err(RegistryError::ServiceNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!(
                "Service not found: {}",
                id
            ))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Register a new service
async fn register_service(
    State(state): State<AppState>,
    Json(req): Json<RegisterServiceRequest>,
) -> impl IntoResponse {
    let endpoint = ServiceEndpoint::new(&req.protocol, &req.host, req.port);
    let service = ServiceInstance::new(&req.name, req.service_type, &req.version, endpoint);

    match state.registry.services.register(service.clone()).await {
        Ok(id) => {
            let response = serde_json::json!({
                "id": id,
                "service": service
            });
            (StatusCode::CREATED, Json(ApiResponse::success(response))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Deregister a service
async fn deregister_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.registry.services.deregister(&id).await {
        Ok(service) => (StatusCode::OK, Json(ApiResponse::success(service))).into_response(),
        Err(RegistryError::ServiceNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!(
                "Service not found: {}",
                id
            ))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Service heartbeat
async fn service_heartbeat(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.registry.services.heartbeat(&id).await {
        Ok(()) => (StatusCode::OK, Json(ApiResponse::success("OK"))).into_response(),
        Err(RegistryError::ServiceNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!(
                "Service not found: {}",
                id
            ))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Update service health
async fn update_service_health(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateServiceHealthRequest>,
) -> impl IntoResponse {
    match state.registry.services.update_health(&id, req.health).await {
        Ok(()) => match state.registry.services.get(&id).await {
            Ok(service) => (StatusCode::OK, Json(ApiResponse::success(service))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<()>::error(e.to_string())),
            )
                .into_response(),
        },
        Err(RegistryError::ServiceNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!(
                "Service not found: {}",
                id
            ))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

/// Get services by type
async fn get_services_by_type(
    State(state): State<AppState>,
    Path(service_type): Path<String>,
) -> impl IntoResponse {
    let service_type = match service_type.to_lowercase().as_str() {
        "forward" => ServiceType::Forward,
        "backward" => ServiceType::Backward,
        "cns" => ServiceType::Cns,
        "data" => ServiceType::Data,
        "api" => ServiceType::Api,
        "execution" => ServiceType::Execution,
        _ => ServiceType::Custom,
    };

    let services = state.registry.services.get_by_type(service_type).await;
    Json(ApiResponse::success(services))
}

/// Get healthy services by type
async fn get_healthy_services_by_type(
    State(state): State<AppState>,
    Path(service_type): Path<String>,
) -> impl IntoResponse {
    let service_type = match service_type.to_lowercase().as_str() {
        "forward" => ServiceType::Forward,
        "backward" => ServiceType::Backward,
        "cns" => ServiceType::Cns,
        "data" => ServiceType::Data,
        "api" => ServiceType::Api,
        "execution" => ServiceType::Execution,
        _ => ServiceType::Custom,
    };

    let services = state
        .registry
        .services
        .get_healthy_by_type(service_type)
        .await;
    Json(ApiResponse::success(services))
}

// ============================================================================
// Health Handlers
// ============================================================================

/// Health check endpoint
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let asset_count = state.registry.assets.count().await;
    let service_count = state.registry.services.count().await;

    let response = HealthResponse {
        service: "janus-registry".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        status: "healthy".to_string(),
        asset_count,
        service_count,
    };

    Json(response)
}

/// Cleanup stale services
async fn cleanup_stale_services(State(state): State<AppState>) -> impl IntoResponse {
    let removed = state.registry.services.cleanup_stale().await;
    let response = serde_json::json!({
        "removed_count": removed.len(),
        "removed_ids": removed
    });
    Json(ApiResponse::success(response))
}

// ============================================================================
// Routing Map Handler
// ============================================================================

/// Routing map response returned by GET /api/v1/routing
#[derive(Debug, Serialize, Deserialize)]
struct RoutingMapResponse {
    /// symbol → preferred exchange name
    routes: HashMap<String, String>,
    /// "SYMBOL:exchange" → exchange-specific symbol string
    exchange_symbols: HashMap<String, String>,
}

/// Build a symbol→exchange routing map from the current asset registry.
///
/// For each asset the "best" exchange is chosen as follows:
///   1. If `"kraken"` appears anywhere in `asset.exchanges`, pick it.
///   2. Otherwise take the first entry in the list.
///
/// The `exchange_symbols` map is keyed by `"SYMBOL:exchange"` and its value
/// is the `exchange_symbol` field of the first matching trading pair (if any).
async fn get_routing_map(State(state): State<AppState>) -> impl IntoResponse {
    let assets = state.registry.assets.list().await;

    let mut routes: HashMap<String, String> = HashMap::new();
    let mut exchange_symbols: HashMap<String, String> = HashMap::new();

    for asset in &assets {
        if asset.exchanges.is_empty() {
            continue;
        }

        // Pick best exchange: prefer "kraken", else first in list
        let best_exchange = if asset.exchanges.iter().any(|e| e == "kraken") {
            "kraken".to_string()
        } else {
            asset.exchanges[0].clone()
        };

        routes.insert(asset.symbol.clone(), best_exchange.clone());

        // Populate exchange_symbols from trading pairs
        for pair in &asset.trading_pairs {
            if let Some(ref exch_sym) = pair.exchange_symbol {
                let key = format!("{}:{}", asset.symbol, best_exchange);
                exchange_symbols
                    .entry(key)
                    .or_insert_with(|| exch_sym.clone());
            }
        }
    }

    info!(
        route_count = routes.len(),
        symbol_count = exchange_symbols.len(),
        "Serving routing map"
    );

    Json(RoutingMapResponse {
        routes,
        exchange_symbols,
    })
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,janus_registry=debug".to_string()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║       JANUS REGISTRY SERVICE - Asset & Service Discovery  ║");
    info!("║   Assets • Services • Discovery • Health                  ║");
    info!("╚═══════════════════════════════════════════════════════════╝");

    // Create registry manager and initialize default assets
    let registry = Arc::new(RegistryManager::new());

    // Initialize default crypto assets
    if let Err(e) = registry.init_default_crypto_assets().await {
        tracing::warn!("Failed to initialize default assets: {}", e);
    }

    let state = AppState {
        registry: registry.clone(),
    };

    // Build router
    let app = Router::new()
        // Health endpoints
        .route("/health", get(health_check))
        .route("/admin/cleanup", post(cleanup_stale_services))
        // Asset endpoints
        .route("/api/v1/assets", get(list_assets))
        .route("/api/v1/assets", post(create_asset))
        .route("/api/v1/assets/:id", get(get_asset))
        .route("/api/v1/assets/:id", delete(delete_asset))
        .route("/api/v1/assets/:id/status", put(update_asset_status))
        .route("/api/v1/assets/symbol/:symbol", get(get_asset_by_symbol))
        .route("/api/v1/assets/type/:type", get(list_assets_by_type))
        // Routing map endpoint (consumed by execution service)
        .route("/api/v1/routing", get(get_routing_map))
        // Service discovery endpoints
        .route("/api/v1/services", get(list_services))
        .route("/api/v1/services", post(register_service))
        .route("/api/v1/services/:id", get(get_service))
        .route("/api/v1/services/:id", delete(deregister_service))
        .route("/api/v1/services/:id/heartbeat", post(service_heartbeat))
        .route("/api/v1/services/:id/health", put(update_service_health))
        .route("/api/v1/services/type/:type", get(get_services_by_type))
        .route(
            "/api/v1/services/type/:type/healthy",
            get(get_healthy_services_by_type),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Get port from environment or use default
    let port: u16 = std::env::var("REGISTRY_PORT")
        .unwrap_or_else(|_| "8085".to_string())
        .parse()
        .unwrap_or(8085);

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting JANUS Registry Service on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
