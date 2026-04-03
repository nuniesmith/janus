// API module for HTTP/WebSocket server
//
// Exposes:
// - Health check endpoint
// - Prometheus metrics endpoint
// - Gap analysis API
// - Historical metrics API
// - Technical indicators API
// - WebSocket streaming endpoint
// - JWT authentication (optional)

pub mod auth;
pub mod gap_analysis;
pub mod health;
pub mod indicators;
pub mod metrics;
pub mod signals;
pub mod websocket;
pub mod websocket_signals;

use axum::{
    Router, middleware,
    routing::{get, post},
};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::actors::indicator::IndicatorActor;
use crate::actors::router::NormalizedMessage;
use crate::actors::signal::SignalActor;
use crate::storage::StorageManager;
use auth::AuthConfig;

/// Shared application state for API handlers
#[derive(Clone)]
pub struct AppState {
    /// Storage manager for QuestDB and Redis
    pub storage: Arc<StorageManager>,

    /// Broadcast receiver for real-time messages
    pub broadcast_tx: broadcast::Sender<NormalizedMessage>,

    /// Indicator actor for real-time technical indicators
    pub indicator_actor: Option<Arc<IndicatorActor>>,

    /// Signal actor for trading signals
    pub signal_actor: Option<Arc<SignalActor>>,

    /// JWT authentication config (optional)
    pub auth_config: Option<Arc<AuthConfig>>,

    /// Service start time for uptime calculation
    pub start_time: std::time::Instant,

    /// Version string
    pub version: String,
}

impl AppState {
    #[allow(dead_code)]
    pub fn new(
        storage: Arc<StorageManager>,
        broadcast_tx: broadcast::Sender<NormalizedMessage>,
    ) -> Self {
        Self {
            storage,
            broadcast_tx,
            indicator_actor: None,
            signal_actor: None,
            auth_config: None,
            start_time: std::time::Instant::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Create AppState with indicator actor
    #[allow(dead_code)]
    pub fn with_indicator_actor(
        storage: Arc<StorageManager>,
        broadcast_tx: broadcast::Sender<NormalizedMessage>,
        indicator_actor: Arc<IndicatorActor>,
    ) -> Self {
        Self {
            storage,
            broadcast_tx,
            indicator_actor: Some(indicator_actor),
            signal_actor: None,
            auth_config: None,
            start_time: std::time::Instant::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Create AppState with both indicator and signal actors
    pub fn with_actors(
        storage: Arc<StorageManager>,
        broadcast_tx: broadcast::Sender<NormalizedMessage>,
        indicator_actor: Arc<IndicatorActor>,
        signal_actor: Arc<SignalActor>,
    ) -> Self {
        Self {
            storage,
            broadcast_tx,
            indicator_actor: Some(indicator_actor),
            signal_actor: Some(signal_actor),
            auth_config: None,
            start_time: std::time::Instant::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Set auth config
    pub fn with_auth(mut self, auth_config: Arc<AuthConfig>) -> Self {
        self.auth_config = Some(auth_config);
        self
    }
}

/// Build the API router with all endpoints
pub fn build_router(state: AppState) -> Router {
    // Initialize auth config from environment if available
    let auth_config = match AuthConfig::from_env() {
        Ok(config) => {
            if config.enabled {
                info!("JWT authentication is ENABLED for API endpoints");
                Some(Arc::new(config))
            } else {
                warn!(
                    "JWT authentication is DISABLED (set DATA_SERVICE_JWT_ENABLED=true to enable)"
                );
                None
            }
        }
        Err(e) => {
            warn!(
                "JWT authentication not configured: {} - Running without auth",
                e
            );
            None
        }
    };

    // Update state with auth config if available
    let state = if let Some(ref config) = auth_config {
        state.with_auth(config.clone())
    } else {
        state
    };

    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/health", get(health::health_handler))
        .route("/metrics", get(metrics::metrics_handler));

    // Protected API routes
    let mut protected_routes = Router::new()
        // Historical data APIs
        .route("/api/v1/gaps", get(gap_analysis::get_gaps_handler))
        .route("/api/v1/metrics", get(metrics::get_metrics_handler))
        // Technical indicators APIs
        .route(
            "/api/v1/indicators",
            get(indicators::list_indicators_handler),
        )
        .route(
            "/api/v1/indicators/info",
            get(indicators::get_indicators_info_handler),
        )
        .route(
            "/api/v1/indicators/{symbol}/{timeframe}",
            get(indicators::get_indicators_handler),
        )
        .route(
            "/api/v1/indicators/{symbol}/{timeframe}/status",
            get(indicators::get_warmup_status_handler),
        )
        .route(
            "/api/v1/indicators/warmup",
            post(indicators::warmup_indicators_handler),
        )
        .route(
            "/api/v1/indicators/warmup/deep",
            post(indicators::deep_warmup_indicators_handler),
        )
        // Signal APIs
        .route("/api/v1/signals", get(signals::list_all_signals_handler))
        .route(
            "/api/v1/signals/stats",
            get(signals::get_signal_stats_handler),
        )
        .route(
            "/api/v1/signals/{symbol}",
            get(signals::list_signals_by_symbol_handler),
        )
        .route(
            "/api/v1/signals/{symbol}/{timeframe}",
            get(signals::list_signals_handler),
        )
        .route(
            "/api/v1/signals/backtest",
            post(signals::run_backtest_handler),
        );

    // Apply auth middleware if configured
    if let Some(config) = auth_config {
        info!("Applying JWT authentication middleware to protected routes");
        protected_routes =
            protected_routes.layer(middleware::from_fn_with_state(config, auth::require_auth));
    }

    // WebSocket routes (no auth for now - can be added later)
    let ws_routes = Router::new()
        .route("/ws/stream", get(websocket::ws_handler))
        .route("/ws/signals", get(websocket_signals::ws_signals_handler));

    // Combine all routes
    Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .merge(ws_routes)
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        )
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state)
}
