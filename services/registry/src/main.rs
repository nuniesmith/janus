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
use chrono::{DateTime, Utc};
use janus_registry_lib::{
    Asset, AssetStatus, AssetType, RegistryError, RegistryManager, ServiceEndpoint, ServiceHealth,
    ServiceInstance, ServiceType, TradingPair,
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
                exchange_symbols.entry(key).or_insert_with(|| exch_sym.clone());
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

    // Spawn background sync task to pull assets from the Python data service
    let sync_registry = registry.clone();
    tokio::spawn(async move {
        sync_from_python_loop(sync_registry).await;
    });

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

// ============================================================================
// Background Python Registry Sync
// ============================================================================

/// Continuously sync assets from the Python data service registry endpoint.
///
/// The loop runs every `REGISTRY_SYNC_INTERVAL_SECS` seconds (env var, default 300).
/// On each tick it fetches `GET {PYTHON_DATA_SERVICE_URL}/api/registry/sync` and
/// upserts every returned asset into the in-memory `RegistryManager`.
async fn sync_from_python_loop(registry: Arc<RegistryManager>) {
    let interval_secs: u64 = std::env::var("REGISTRY_SYNC_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    let base_url = std::env::var("PYTHON_DATA_SERVICE_URL")
        .unwrap_or_else(|_| "http://fks_ruby:8000".to_string());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();

    // Initial delay to let the Python service start up
    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
    tracing::info!("Python registry sync loop started (interval={}s)", interval_secs);

    loop {
        match sync_once(&client, &base_url, &registry).await {
            Ok(count) => tracing::info!(count, "Registry sync from Python complete"),
            Err(e) => tracing::warn!(error = %e, "Registry sync from Python failed"),
        }

        // Push Janus annotations back to Python
        match push_annotations_to_python(&client, &base_url, &registry).await {
            Ok(count) => {
                if count > 0 {
                    tracing::info!(count, "Pushed annotations to Python");
                }
            }
            Err(e) => tracing::warn!(error = %e, "Failed to push annotations to Python"),
        }

        // Sync Python service discovery into Janus registry
        match sync_python_services(&client, &base_url, &registry).await {
            Ok(count) => tracing::info!(count, "Synced Python services into registry"),
            Err(e) => tracing::warn!(error = %e, "Failed to sync Python services"),
        }

        tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
    }
}

/// Execute a single sync cycle: fetch the JSON catalog and upsert every asset.
async fn sync_once(
    client: &reqwest::Client,
    base_url: &str,
    registry: &RegistryManager,
) -> std::result::Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("{}/api/registry/sync", base_url.trim_end_matches('/'));
    tracing::debug!(url = %url, "Fetching Python asset catalog");

    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        return Err(format!("Python sync endpoint returned HTTP {}", resp.status()).into());
    }

    let assets: Vec<serde_json::Value> = resp.json().await?;
    let mut count = 0usize;

    for asset_json in &assets {
        match parse_python_asset(asset_json) {
            Ok(asset) => {
                let id = asset.id.clone();
                // Try to register; if already present, remove then re-register (upsert).
                match registry.assets.register(asset).await {
                    Ok(()) => {
                        count += 1;
                    }
                    Err(RegistryError::DuplicateAsset(_)) => {
                        // Upsert: remove old entry and re-register
                        if let Ok(new_asset) = parse_python_asset(asset_json) {
                            let _ = registry.assets.remove(&id).await;
                            if registry.assets.register(new_asset).await.is_ok() {
                                count += 1;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, id = %id, "Failed to register synced asset");
                    }
                }
            }
            Err(e) => {
                tracing::debug!(error = %e, "Failed to parse asset from Python catalog");
            }
        }
    }

    Ok(count)
}

// ============================================================================
// Bidirectional Sync: Janus → Python
// ============================================================================

/// Push Janus asset annotations (execution metadata) to the Python service.
///
/// For each registered asset that carries operational metadata (`last_trade_at`,
/// `preferred_exchange`, or `health`), an annotation payload is built and POSTed
/// to the Python `/api/registry/annotations` endpoint.
async fn push_annotations_to_python(
    client: &reqwest::Client,
    base_url: &str,
    registry: &RegistryManager,
) -> std::result::Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let assets = registry.assets.list().await;

    let mut annotations = Vec::new();
    for asset in &assets {
        // Only build an annotation when the asset carries at least one
        // operational metadata field that Janus enriches.
        let has_annotation_data = asset.metadata.contains_key("last_trade_at")
            || asset.metadata.contains_key("preferred_exchange")
            || asset.metadata.contains_key("health");

        if !has_annotation_data {
            continue;
        }

        let exchange = asset
            .metadata
            .get("preferred_exchange")
            .cloned()
            .or_else(|| asset.exchanges.first().cloned())
            .unwrap_or_default();

        let exchange_symbol = asset
            .trading_pairs
            .first()
            .and_then(|tp| tp.exchange_symbol.clone())
            .unwrap_or_default();

        let health = asset
            .metadata
            .get("health")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let mut extra_metadata = serde_json::Map::new();
        for (k, v) in &asset.metadata {
            if k != "last_trade_at" && k != "preferred_exchange" && k != "health" {
                extra_metadata.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
        }

        let ann = serde_json::json!({
            "symbol": asset.symbol,
            "exchange": exchange,
            "last_trade_at": asset.metadata.get("last_trade_at").cloned().unwrap_or_default(),
            "health": health,
            "exchange_symbol": exchange_symbol,
            "metadata": extra_metadata,
        });
        annotations.push(ann);
    }

    if annotations.is_empty() {
        return Ok(0);
    }

    let url = format!(
        "{}/api/registry/annotations",
        base_url.trim_end_matches('/')
    );
    let body = serde_json::json!({ "annotations": annotations });

    tracing::debug!(
        count = annotations.len(),
        url = %url,
        "Pushing annotations to Python"
    );

    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        return Err(format!(
            "Python annotations endpoint returned HTTP {}",
            resp.status()
        )
        .into());
    }

    let result: serde_json::Value = resp.json().await?;
    let accepted = result["accepted"].as_u64().unwrap_or(0) as usize;
    Ok(accepted)
}

/// Pull Python-side service discovery metadata and register each service
/// in the Janus registry.
///
/// GETs `{base_url}/api/registry/services` and registers each entry as a
/// `ServiceInstance`.  Deterministic IDs (`python-{name}`) are used so
/// repeated syncs update existing entries instead of creating duplicates.
async fn sync_python_services(
    client: &reqwest::Client,
    base_url: &str,
    registry: &RegistryManager,
) -> std::result::Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let url = format!(
        "{}/api/registry/services",
        base_url.trim_end_matches('/')
    );
    tracing::debug!(url = %url, "Fetching Python service catalog");

    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        return Err(format!(
            "Python services endpoint returned HTTP {}",
            resp.status()
        )
        .into());
    }

    let services: Vec<serde_json::Value> = resp.json().await?;
    let mut count = 0usize;

    for svc_json in &services {
        let name = match svc_json["name"].as_str() {
            Some(n) => n,
            None => continue,
        };
        let host = svc_json["host"].as_str().unwrap_or("localhost");
        let port = svc_json["port"].as_u64().unwrap_or(8000) as u16;
        let svc_type_str = svc_json["type"].as_str().unwrap_or("Custom");
        let health_str = svc_json["health"].as_str().unwrap_or("unknown");

        let service_type = match svc_type_str {
            "Data" => ServiceType::Data,
            "Api" | "Web" => ServiceType::Api,
            "Execution" | "Engine" => ServiceType::Execution,
            "Forward" => ServiceType::Forward,
            "Backward" => ServiceType::Backward,
            "Cns" => ServiceType::Cns,
            _ => ServiceType::Custom,
        };

        let health = match health_str.to_lowercase().as_str() {
            "healthy" => ServiceHealth::Healthy,
            "degraded" => ServiceHealth::Degraded,
            "unhealthy" => ServiceHealth::Unhealthy,
            _ => ServiceHealth::Unknown,
        };

        let deterministic_id = format!("python-{}", name);
        let endpoint = ServiceEndpoint::new("http", host, port);
        let now = Utc::now();

        let service = ServiceInstance {
            id: deterministic_id.clone(),
            name: format!("python-{}", name),
            service_type,
            version: "1.0.0".to_string(),
            endpoint,
            additional_endpoints: HashMap::new(),
            health,
            last_health_check: Some(now),
            metadata: HashMap::new(),
            registered_at: now,
            last_heartbeat: now,
        };

        // Deregister first to avoid duplicates in the by_type index,
        // then re-register with the updated state.
        let _ = registry.services.deregister(&deterministic_id).await;
        match registry.services.register(service).await {
            Ok(_) => count += 1,
            Err(e) => {
                tracing::debug!(error = %e, name = %name, "Failed to register Python service");
            }
        }
    }

    Ok(count)
}

// ============================================================================
// Python Asset Parsing
// ============================================================================

/// Parse a single JSON object (from the Python `/api/registry/sync` response)
/// into a Rust `Asset` struct.
fn parse_python_asset(
    json: &serde_json::Value,
) -> std::result::Result<Asset, Box<dyn std::error::Error + Send + Sync>> {
    let id = json["id"]
        .as_str()
        .ok_or("missing id")?
        .to_string();
    let name = json["name"]
        .as_str()
        .ok_or("missing name")?
        .to_string();
    let symbol = json["symbol"]
        .as_str()
        .ok_or("missing symbol")?
        .to_string();

    let asset_type = match json["asset_type"].as_str().unwrap_or("Other") {
        "Crypto" => AssetType::Crypto,
        "Forex" => AssetType::Forex,
        "Equity" => AssetType::Equity,
        "Futures" => AssetType::Futures,
        "Options" => AssetType::Options,
        "Commodity" => AssetType::Commodity,
        "Index" => AssetType::Index,
        _ => AssetType::Other,
    };

    let status = match json["status"].as_str().unwrap_or("Inactive") {
        "Active" => AssetStatus::Active,
        "Inactive" => AssetStatus::Inactive,
        "Suspended" => AssetStatus::Suspended,
        "Delisted" => AssetStatus::Delisted,
        _ => AssetStatus::Inactive,
    };

    // Parse trading pairs
    let mut trading_pairs = Vec::new();
    if let Some(pairs) = json["trading_pairs"].as_array() {
        for p in pairs {
            let base = p["base"].as_str().unwrap_or("").to_string();
            let quote = p["quote"].as_str().unwrap_or("USD").to_string();
            let pair_symbol = p["symbol"].as_str().unwrap_or("").to_string();
            let exchange_symbol = p["exchange_symbol"].as_str().map(|s| s.to_string());

            let mut tp = TradingPair::new(&base, &quote);
            tp.symbol = pair_symbol;
            tp.exchange_symbol = exchange_symbol;
            trading_pairs.push(tp);
        }
    }

    // Parse exchanges
    let exchanges: Vec<String> = json["exchanges"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // Parse metadata
    let metadata: HashMap<String, String> = json["metadata"]
        .as_object()
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                .collect()
        })
        .unwrap_or_default();

    // Parse timestamps (fall back to now)
    let now = Utc::now();
    let created_at = json["created_at"]
        .as_str()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or(now);
    let updated_at = json["updated_at"]
        .as_str()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or(now);

    Ok(Asset {
        id,
        name,
        symbol,
        asset_type,
        status,
        trading_pairs,
        price_precision: json["price_precision"].as_u64().unwrap_or(8) as u8,
        quantity_precision: json["quantity_precision"].as_u64().unwrap_or(8) as u8,
        min_order_size: json["min_order_size"].as_f64().unwrap_or(0.0001),
        max_order_size: json["max_order_size"].as_f64(),
        min_notional: json["min_notional"].as_f64(),
        tick_size: json["tick_size"].as_f64().unwrap_or(0.01),
        lot_size: json["lot_size"].as_f64().unwrap_or(0.001),
        maker_fee: json["maker_fee"].as_f64().unwrap_or(0.001),
        taker_fee: json["taker_fee"].as_f64().unwrap_or(0.001),
        exchanges,
        metadata,
        created_at,
        updated_at,
    })
}
