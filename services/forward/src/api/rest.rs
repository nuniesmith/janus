//! # REST API for JANUS Service
//!
//! Implements REST API endpoints for trading signal generation using Axum.
//!
//! ## Endpoints
//!
//! - `POST /api/v1/signals/generate` - Generate a single signal
//! - `POST /api/v1/signals/batch` - Generate multiple signals
//! - `GET /api/v1/health` - Health check
//! - `GET /api/v1/metrics` - Prometheus metrics
//! - `GET /api/v1/version` - Service version info

use crate::indicators::IndicatorAnalysis;
use crate::signal::{SignalGenerator, Timeframe, TradingSignal};
use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

/// Process-wide start time for uptime tracking.
static SERVICE_START: OnceLock<std::time::Instant> = OnceLock::new();

fn service_uptime_secs() -> u64 {
    SERVICE_START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs()
}

/// REST API server
pub struct RestServer {
    port: u16,
    host: String,
    generator: Arc<RwLock<SignalGenerator>>,
}

impl RestServer {
    /// Create a new REST API server
    pub fn new(host: String, port: u16, generator: SignalGenerator) -> Self {
        info!("Initializing REST API server on {}:{}", host, port);
        Self {
            port,
            host,
            generator: Arc::new(RwLock::new(generator)),
        }
    }

    /// Build the router with all endpoints
    pub fn router(generator: Arc<RwLock<SignalGenerator>>) -> Router {
        Router::new()
            .route("/api/v1/health", get(health_handler))
            .route("/api/v1/version", get(version_handler))
            .route("/api/v1/signals/generate", post(generate_signal_handler))
            .route("/api/v1/signals/batch", post(generate_batch_handler))
            .route("/api/v1/metrics", get(metrics_handler))
            .with_state(generator)
    }

    /// Start the REST API server
    pub async fn start(self) -> Result<()> {
        let addr = format!("{}:{}", self.host, self.port);
        info!("Starting REST API server on {}", addr);

        let app = Self::router(self.generator);
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        axum::serve(listener, app).await?;

        Ok(())
    }
}

/// Application state
type AppState = Arc<RwLock<SignalGenerator>>;

// Request/Response types

/// Signal generation request
#[derive(Debug, Deserialize, Serialize)]
pub struct GenerateSignalRequest {
    pub symbol: String,
    pub timeframe: String, // "1m", "5m", "15m", "1h", "4h", "1d"
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

/// Metrics response
#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub signal_metrics: SignalMetricsDto,
    pub ml_metrics: Option<MlMetricsDto>,
}

#[derive(Debug, Serialize)]
pub struct SignalMetricsDto {
    pub total_generated: u64,
    pub total_filtered: u64,
    pub filter_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct MlMetricsDto {
    pub total_inferences: u64,
    pub avg_latency_us: u64,
    pub p99_latency_us: u64,
    pub avg_confidence: f64,
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

// Handlers

/// Health check endpoint
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        service: crate::SERVICE_NAME.to_string(),
        version: crate::VERSION.to_string(),
        uptime_seconds: service_uptime_secs(),
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
    State(generator): State<AppState>,
    Json(req): Json<GenerateSignalRequest>,
) -> Result<Json<SignalResponse>, ApiError> {
    let start = std::time::Instant::now();

    debug!("Generating signal for {} on {}", req.symbol, req.timeframe);

    // Parse timeframe
    let timeframe = parse_timeframe(&req.timeframe)
        .map_err(|e| ApiError::BadRequest(format!("Invalid timeframe: {}", e)))?;

    // Convert analysis
    let analysis: IndicatorAnalysis = req.analysis.into();

    // Generate signal
    let signal_gen = generator.read().await;
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
    State(generator): State<AppState>,
    Json(batch_req): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, ApiError> {
    let start = std::time::Instant::now();

    debug!("Processing batch of {} requests", batch_req.requests.len());

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
            Err(e) => {
                error!("Invalid timeframe: {}", e);
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
        let signal_gen = generator.read().await;
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
            Err(e) => {
                error!("Error generating signal: {}", e);
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

/// Metrics endpoint
async fn metrics_handler(State(generator): State<AppState>) -> Json<MetricsResponse> {
    let signal_gen = generator.read().await;
    let metrics = signal_gen.metrics();

    let signal_metrics = SignalMetricsDto {
        total_generated: metrics.total_generated(),
        total_filtered: metrics.total_filtered(),
        filter_rate: metrics.filter_rate(),
    };

    let ml_metrics = signal_gen.ml_metrics().await.map(|ml_met| MlMetricsDto {
        total_inferences: ml_met.total_inferences,
        avg_latency_us: ml_met.avg_latency_us(),
        p99_latency_us: ml_met.p99_latency_us(),
        avg_confidence: ml_met.avg_confidence,
    });

    Json(MetricsResponse {
        signal_metrics,
        ml_metrics,
    })
}

// Helper functions

fn parse_timeframe(s: &str) -> Result<Timeframe> {
    match s {
        "1m" => Ok(Timeframe::M1),
        "5m" => Ok(Timeframe::M5),
        "15m" => Ok(Timeframe::M15),
        "1h" => Ok(Timeframe::H1),
        "4h" => Ok(Timeframe::H4),
        "1d" => Ok(Timeframe::D1),
        _ => Err(anyhow::anyhow!("Unknown timeframe: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::SignalGeneratorConfig;

    #[test]
    fn test_parse_timeframe() {
        assert!(parse_timeframe("1m").is_ok());
        assert!(parse_timeframe("5m").is_ok());
        assert!(parse_timeframe("1h").is_ok());
        assert!(parse_timeframe("invalid").is_err());
    }

    #[test]
    fn test_rest_server_creation() {
        let config = SignalGeneratorConfig::default();
        let generator = SignalGenerator::new(config);
        let server = RestServer::new("0.0.0.0".to_string(), 8080, generator);
        assert_eq!(server.port, 8080);
        assert_eq!(server.host, "0.0.0.0");
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let response = health_handler().await;
        assert_eq!(response.status, "healthy");
        assert_eq!(response.service, crate::SERVICE_NAME);
    }

    #[tokio::test]
    async fn test_version_endpoint() {
        let response = version_handler().await;
        assert_eq!(response.service, crate::SERVICE_NAME);
        assert!(!response.version.is_empty());
    }
}
