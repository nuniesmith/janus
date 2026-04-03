//! # Combined REST API Server
//!
//! Combines signal generation, risk management, and brain health REST endpoints
//! into a single server.
//!
//! ## Endpoints
//!
//! ### Signal Endpoints
//! - `POST /api/v1/signals/generate` - Generate a single signal
//! - `POST /api/v1/signals/batch` - Generate multiple signals
//! - `GET /api/v1/health` - Health check
//! - `GET /api/v1/version` - Service version info
//!
//! ### Risk Management Endpoints
//! - `GET /api/v1/risk/config` - Get risk configuration
//! - `PUT /api/v1/risk/config` - Update risk configuration
//! - `GET /api/v1/risk/portfolio` - Get portfolio state
//! - `POST /api/v1/risk/portfolio/positions` - Add position to portfolio
//! - `DELETE /api/v1/risk/portfolio/positions/:symbol` - Remove position
//! - `GET /api/v1/risk/metrics` - Get risk metrics snapshot
//! - `GET /api/v1/risk/performance` - Get performance metrics
//! - `POST /api/v1/risk/validate` - Validate a signal
//! - `POST /api/v1/risk/calculate/position-size` - Calculate position size
//! - `POST /api/v1/risk/calculate/stop-loss` - Calculate stop loss
//! - `POST /api/v1/risk/calculate/take-profit` - Calculate take profit
//!
//! ### Brain Health Endpoints
//! - `GET /api/v1/brain/health` - Full brain health report
//! - `GET /api/v1/brain/pipeline` - Pipeline-only metrics
//! - `POST /api/v1/brain/kill-switch/activate` - Activate kill switch
//! - `POST /api/v1/brain/kill-switch/deactivate` - Deactivate kill switch

use crate::api::brain_rest::{self, BrainHealthState};
use crate::api::risk_rest::RiskApiState;
use crate::risk::{PortfolioState, RiskManager};
use crate::signal::SignalGenerator;
use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Combined REST server state
#[derive(Clone)]
pub struct RestServerState {
    signal_generator: Arc<RwLock<SignalGenerator>>,
    risk_api_state: Arc<RiskApiState>,
    brain_health_state: Option<Arc<BrainHealthState>>,
    start_time: std::time::Instant,
}

impl RestServerState {
    pub fn new(
        signal_generator: Arc<RwLock<SignalGenerator>>,
        risk_manager: Arc<RwLock<RiskManager>>,
        portfolio: Arc<RwLock<PortfolioState>>,
    ) -> Self {
        let risk_api_state = Arc::new(RiskApiState::new(risk_manager, portfolio));

        Self {
            signal_generator,
            risk_api_state,
            brain_health_state: None,
            start_time: std::time::Instant::now(),
        }
    }

    /// Create server state with brain health endpoints enabled.
    pub fn with_brain_health(
        signal_generator: Arc<RwLock<SignalGenerator>>,
        risk_manager: Arc<RwLock<RiskManager>>,
        portfolio: Arc<RwLock<PortfolioState>>,
        brain_state: Arc<BrainHealthState>,
    ) -> Self {
        let risk_api_state = Arc::new(RiskApiState::new(risk_manager, portfolio));

        Self {
            signal_generator,
            risk_api_state,
            brain_health_state: Some(brain_state),
            start_time: std::time::Instant::now(),
        }
    }

    /// Set the brain health state after construction.
    pub fn set_brain_health_state(&mut self, state: Arc<BrainHealthState>) {
        self.brain_health_state = Some(state);
    }
}

/// REST server
pub struct RestServer {
    port: u16,
    host: String,
    state: RestServerState,
}

impl RestServer {
    /// Create a new REST API server
    pub fn new(
        host: String,
        port: u16,
        signal_generator: Arc<RwLock<SignalGenerator>>,
        risk_manager: Arc<RwLock<RiskManager>>,
        portfolio: Arc<RwLock<PortfolioState>>,
    ) -> Self {
        info!("Initializing combined REST API server on {}:{}", host, port);

        let state = RestServerState::new(signal_generator, risk_manager, portfolio);

        Self { port, host, state }
    }

    /// Create a new REST API server with brain health endpoints.
    pub fn with_brain_health(
        host: String,
        port: u16,
        signal_generator: Arc<RwLock<SignalGenerator>>,
        risk_manager: Arc<RwLock<RiskManager>>,
        portfolio: Arc<RwLock<PortfolioState>>,
        brain_state: Arc<BrainHealthState>,
    ) -> Self {
        info!(
            "Initializing combined REST API server on {}:{} (brain health enabled)",
            host, port
        );

        let state = RestServerState::with_brain_health(
            signal_generator,
            risk_manager,
            portfolio,
            brain_state,
        );

        Self { port, host, state }
    }

    /// Build the combined router with all endpoints
    pub fn router(state: RestServerState) -> Router {
        // Create signal endpoints router
        let signal_router = Router::new()
            .route("/api/v1/signals/generate", post(generate_signal_handler))
            .route("/api/v1/signals/batch", post(generate_batch_handler))
            .with_state(state.clone());

        // Create health/info endpoints
        let health_router = Router::new()
            .route("/api/v1/health", get(health_handler))
            .route("/api/v1/version", get(version_handler))
            .with_state(state.clone());

        // Create risk management router with its own state
        let risk_router = RiskApiState::router(state.risk_api_state.clone());

        // Create brain health router (if brain runtime is enabled)
        let brain_router = state
            .brain_health_state
            .as_ref()
            .map(|brain_state| brain_rest::router(brain_state.clone()));

        // Combine all routers
        let mut app = Router::new()
            .merge(signal_router)
            .merge(health_router)
            .merge(risk_router);

        if let Some(brain) = brain_router {
            app = app.merge(brain);
        }

        app.layer(TraceLayer::new_for_http())
    }

    /// Start the REST API server
    pub async fn start(self) -> Result<()> {
        let addr = format!("{}:{}", self.host, self.port);
        info!("Starting combined REST API server on {}", addr);

        let app = Self::router(self.state);
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        info!("✅ REST API server listening on {}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }
}

// ===== DTOs for Signal Endpoints =====

use crate::indicators::IndicatorAnalysis;
use crate::signal::{Timeframe, TradingSignal};

/// Signal generation request
#[derive(Debug, Deserialize, Serialize)]
pub struct GenerateSignalRequest {
    pub symbol: String,
    pub timeframe: String,
    pub analysis: IndicatorAnalysisDto,
    pub current_price: f64,
    #[serde(default)]
    pub enable_ml: bool,
}

/// Indicator analysis DTO
#[derive(Debug, Deserialize, Serialize)]
pub struct IndicatorAnalysisDto {
    pub ema_fast: Option<f64>,
    pub ema_slow: Option<f64>,
    pub ema_cross: f64,
    pub rsi: Option<f64>,
    pub rsi_signal: f64,
    pub macd_line: Option<f64>,
    pub macd_signal: Option<f64>,
    pub macd_histogram: Option<f64>,
    pub macd_cross: f64,
    pub bb_upper: Option<f64>,
    pub bb_middle: Option<f64>,
    pub bb_lower: Option<f64>,
    pub bb_position: f64,
    pub atr: Option<f64>,
    pub trend_strength: f64,
    pub volatility: f64,
}

impl From<IndicatorAnalysisDto> for IndicatorAnalysis {
    fn from(dto: IndicatorAnalysisDto) -> Self {
        IndicatorAnalysis {
            ema_fast: dto.ema_fast,
            ema_slow: dto.ema_slow,
            ema_cross: dto.ema_cross,
            rsi: dto.rsi,
            rsi_signal: dto.rsi_signal,
            macd_line: dto.macd_line,
            macd_signal: dto.macd_signal,
            macd_histogram: dto.macd_histogram,
            macd_cross: dto.macd_cross,
            bb_upper: dto.bb_upper,
            bb_middle: dto.bb_middle,
            bb_lower: dto.bb_lower,
            bb_position: dto.bb_position,
            atr: dto.atr,
            trend_strength: dto.trend_strength,
            volatility: dto.volatility,
        }
    }
}

/// Signal response
#[derive(Debug, Serialize)]
pub struct SignalResponse {
    pub signal: Option<TradingSignalDto>,
    pub filtered: bool,
    pub processing_time_ms: f64,
}

/// Trading signal DTO
#[derive(Debug, Serialize)]
pub struct TradingSignalDto {
    pub signal_id: String,
    pub signal_type: String,
    pub symbol: String,
    pub timeframe: String,
    pub confidence: f64,
    pub strength: f64,
    pub timestamp: i64,
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl From<&TradingSignal> for TradingSignalDto {
    fn from(signal: &TradingSignal) -> Self {
        TradingSignalDto {
            signal_id: signal.signal_id.clone(),
            signal_type: format!("{:?}", signal.signal_type),
            symbol: signal.symbol.clone(),
            timeframe: signal.timeframe.as_str().to_string(),
            confidence: signal.confidence,
            strength: signal.strength,
            timestamp: signal.timestamp.timestamp(),
            entry_price: signal.entry_price,
            stop_loss: signal.stop_loss,
            take_profit: signal.take_profit,
            metadata: signal.metadata.clone(),
        }
    }
}

/// Batch request
#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub requests: Vec<GenerateSignalRequest>,
}

/// Batch response
#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub signals: Vec<SignalResponse>,
    pub total_processing_time_ms: f64,
    pub statistics: BatchStatistics,
}

/// Batch statistics
#[derive(Debug, Serialize)]
pub struct BatchStatistics {
    pub total: usize,
    pub successful: usize,
    pub filtered: usize,
    pub failed: usize,
}

/// Health response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub uptime_seconds: u64,
}

/// Version response
#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub version: String,
    pub service: String,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

// API Error type
enum ApiError {
    BadRequest(String),
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg),
            ApiError::InternalError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg)
            }
        };

        let body = Json(ErrorResponse {
            error: error.to_string(),
            message,
        });

        (status, body).into_response()
    }
}

// ===== Handlers =====

/// Health check endpoint
async fn health_handler(State(state): State<RestServerState>) -> Json<HealthResponse> {
    let signal_gen = state.signal_generator.read().await;
    let _ = signal_gen.metrics(); // Touch metrics to ensure service is responsive

    Json(HealthResponse {
        status: "healthy".to_string(),
        service: crate::SERVICE_NAME.to_string(),
        version: crate::VERSION.to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
    })
}

/// Version endpoint
async fn version_handler() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: crate::VERSION.to_string(),
        service: crate::SERVICE_NAME.to_string(),
    })
}

/// Generate signal endpoint
async fn generate_signal_handler(
    State(state): State<RestServerState>,
    Json(req): Json<GenerateSignalRequest>,
) -> Result<Json<SignalResponse>, ApiError> {
    let start = std::time::Instant::now();

    // Parse timeframe
    let timeframe = parse_timeframe(&req.timeframe)
        .map_err(|e| ApiError::BadRequest(format!("Invalid timeframe: {}", e)))?;

    // Convert analysis
    let analysis: IndicatorAnalysis = req.analysis.into();

    // Generate signal
    let signal_gen = state.signal_generator.read().await;
    let result = signal_gen
        .generate_from_analysis(req.symbol, timeframe, &analysis, req.current_price)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    let processing_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let (signal, filtered) = match result {
        Some(sig) => (Some(TradingSignalDto::from(&sig)), false),
        None => (None, true),
    };

    Ok(Json(SignalResponse {
        signal,
        filtered,
        processing_time_ms,
    }))
}

/// Generate batch endpoint
async fn generate_batch_handler(
    State(state): State<RestServerState>,
    Json(batch_req): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, ApiError> {
    let start = std::time::Instant::now();

    let mut signals = Vec::new();
    let mut stats = BatchStatistics {
        total: batch_req.requests.len(),
        successful: 0,
        filtered: 0,
        failed: 0,
    };

    for req in batch_req.requests {
        // Parse timeframe
        let timeframe = match parse_timeframe(&req.timeframe) {
            Ok(tf) => tf,
            Err(_) => {
                stats.failed += 1;
                signals.push(SignalResponse {
                    signal: None,
                    filtered: false,
                    processing_time_ms: 0.0,
                });
                continue;
            }
        };

        // Convert analysis
        let analysis: IndicatorAnalysis = req.analysis.into();

        // Generate signal
        let signal_gen = state.signal_generator.read().await;
        match signal_gen
            .generate_from_analysis(req.symbol, timeframe, &analysis, req.current_price)
            .await
        {
            Ok(Some(sig)) => {
                stats.successful += 1;
                signals.push(SignalResponse {
                    signal: Some(TradingSignalDto::from(&sig)),
                    filtered: false,
                    processing_time_ms: 0.0,
                });
            }
            Ok(None) => {
                stats.filtered += 1;
                signals.push(SignalResponse {
                    signal: None,
                    filtered: true,
                    processing_time_ms: 0.0,
                });
            }
            Err(_) => {
                stats.failed += 1;
                signals.push(SignalResponse {
                    signal: None,
                    filtered: false,
                    processing_time_ms: 0.0,
                });
            }
        }
    }

    let total_processing_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(BatchResponse {
        signals,
        total_processing_time_ms,
        statistics: stats,
    }))
}

// Helper functions

fn parse_timeframe(s: &str) -> Result<Timeframe, String> {
    match s {
        "1m" | "M1" => Ok(Timeframe::M1),
        "5m" | "M5" => Ok(Timeframe::M5),
        "15m" | "M15" => Ok(Timeframe::M15),
        "1h" | "H1" => Ok(Timeframe::H1),
        "4h" | "H4" => Ok(Timeframe::H4),
        "1d" | "D1" => Ok(Timeframe::D1),
        _ => Err(format!("Unknown timeframe: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_timeframe() {
        assert!(parse_timeframe("1m").is_ok());
        assert!(parse_timeframe("M1").is_ok());
        assert!(parse_timeframe("1h").is_ok());
        assert!(parse_timeframe("H1").is_ok());
        assert!(parse_timeframe("invalid").is_err());
    }

    #[tokio::test]
    async fn test_version_endpoint() {
        let response = version_handler().await;
        assert_eq!(response.service, crate::SERVICE_NAME);
        assert!(!response.version.is_empty());
    }
}
