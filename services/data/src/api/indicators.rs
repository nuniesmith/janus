//! Indicators API - Real-time technical indicator endpoints
//!
//! Provides HTTP endpoints for querying calculated indicator values.
//!
//! ## Endpoints:
//! - GET /api/v1/indicators - List all tracked symbol/timeframe pairs
//! - GET /api/v1/indicators/:symbol/:timeframe - Get indicators for specific pair
//! - GET /api/v1/indicators/:symbol/:timeframe/status - Get warmup status
//! - GET /api/v1/indicators/info - Get available indicator metadata
//! - POST /api/v1/indicators/warmup - Trigger indicator warmup from historical data

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use crate::api::AppState;

/// Response for indicator values
#[derive(Debug, Serialize)]
pub struct IndicatorResponse {
    pub symbol: String,
    pub exchange: String,
    pub timeframe: String,
    pub timestamp: i64,
    pub close: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ema_8: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ema_21: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ema_50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ema_200: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rsi_14: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub macd_line: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub macd_signal: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub macd_histogram: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atr_14: Option<f64>,
    pub calculated_at: i64,
}

/// Response for warmup status
#[derive(Debug, Serialize)]
pub struct WarmupStatusResponse {
    pub symbol: String,
    pub timeframe: String,
    pub all_ready: bool,
    pub candles_processed: u64,
    pub candles_needed: u64,
    pub indicators: HashMap<String, bool>,
}

/// Response for list of tracked pairs
#[derive(Debug, Serialize)]
pub struct TrackedPairsResponse {
    pub pairs: Vec<TrackedPair>,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct TrackedPair {
    pub symbol: String,
    pub timeframe: String,
}

/// Query parameters for indicators list
#[derive(Debug, Deserialize)]
pub struct IndicatorListQuery {
    /// Filter by symbol (optional)
    pub symbol: Option<String>,
    /// Filter by timeframe (optional)
    pub timeframe: Option<String>,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

/// Request body for warmup endpoint
#[derive(Debug, Deserialize)]
pub struct WarmupRequest {
    /// Symbols to warm up (e.g., ["BTCUSD", "ETHUSDT"])
    /// If empty, warms up all tracked pairs
    #[serde(default)]
    pub symbols: Vec<String>,
    /// Timeframe to warm up (default: "1m")
    #[serde(default = "default_timeframe")]
    pub timeframe: String,
    /// Maximum candles to fetch per symbol (default: 250)
    #[serde(default = "default_max_candles")]
    pub max_candles: Option<usize>,
}

fn default_timeframe() -> String {
    "1m".to_string()
}

fn default_max_candles() -> Option<usize> {
    Some(250)
}

/// Response for warmup operation
#[derive(Debug, Serialize)]
pub struct WarmupResponse {
    pub success: bool,
    pub message: String,
    pub results: Vec<WarmupPairResult>,
    pub total_candles_processed: usize,
    pub pairs_fully_warmed: usize,
    pub duration_ms: u64,
}

/// Result for a single pair warmup
#[derive(Debug, Serialize)]
pub struct WarmupPairResult {
    pub symbol: String,
    pub timeframe: String,
    pub candles_processed: usize,
    pub duration_ms: u64,
    pub all_indicators_ready: bool,
}

/// Get list of all tracked symbol/timeframe pairs
///
/// GET /api/v1/indicators
pub async fn list_indicators_handler(
    State(state): State<AppState>,
    Query(query): Query<IndicatorListQuery>,
) -> impl IntoResponse {
    // Get tracked pairs from indicator actor if available
    let pairs = if let Some(ref indicator_actor) = state.indicator_actor {
        indicator_actor
            .get_tracked_pairs()
            .await
            .into_iter()
            .map(|(symbol, timeframe)| TrackedPair { symbol, timeframe })
            .collect()
    } else {
        // Fallback: return empty list if indicator actor not available
        Vec::new()
    };

    // Apply filters if provided
    let filtered_pairs: Vec<_> = pairs
        .into_iter()
        .filter(|p| query.symbol.as_ref().is_none_or(|s| p.symbol.contains(s)))
        .filter(|p| query.timeframe.as_ref().is_none_or(|t| &p.timeframe == t))
        .collect();

    let count = filtered_pairs.len();

    (
        StatusCode::OK,
        Json(TrackedPairsResponse {
            pairs: filtered_pairs,
            count,
        }),
    )
}

/// Get indicator values for a specific symbol/timeframe
///
/// GET /api/v1/indicators/:symbol/:timeframe
pub async fn get_indicators_handler(
    State(state): State<AppState>,
    Path((symbol, timeframe)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // Query the IndicatorActor for live indicator data
    if let Some(ref indicator_actor) = state.indicator_actor {
        match indicator_actor.get_indicators(&symbol, &timeframe).await {
            Some(data) => {
                let response = IndicatorResponse {
                    symbol: data.symbol,
                    exchange: data.exchange,
                    timeframe: data.timeframe,
                    timestamp: data.timestamp,
                    close: data.close,
                    ema_8: data.ema_8,
                    ema_21: data.ema_21,
                    ema_50: data.ema_50,
                    ema_200: data.ema_200,
                    rsi_14: data.rsi_14,
                    macd_line: data.macd_line,
                    macd_signal: data.macd_signal,
                    macd_histogram: data.macd_histogram,
                    atr_14: data.atr_14,
                    calculated_at: data.calculated_at,
                };
                Ok((StatusCode::OK, Json(response)))
            }
            None => Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "not_found".to_string(),
                    message: format!(
                        "No indicator data found for symbol '{}' with timeframe '{}'. \
                         The pair may not be tracked or indicators are still warming up.",
                        symbol, timeframe
                    ),
                }),
            )),
        }
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "service_unavailable".to_string(),
                message: "Indicator actor is not available".to_string(),
            }),
        ))
    }
}

/// Get warmup status for a specific symbol/timeframe
///
/// GET /api/v1/indicators/:symbol/:timeframe/status
pub async fn get_warmup_status_handler(
    State(state): State<AppState>,
    Path((symbol, timeframe)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // Query the IndicatorActor for warmup status
    if let Some(ref indicator_actor) = state.indicator_actor {
        match indicator_actor.get_warmup_status(&symbol, &timeframe).await {
            Some(status) => {
                let mut indicators = HashMap::new();
                indicators.insert("ema_8".to_string(), status.ema_8_ready);
                indicators.insert("ema_21".to_string(), status.ema_21_ready);
                indicators.insert("ema_50".to_string(), status.ema_50_ready);
                indicators.insert("ema_200".to_string(), status.ema_200_ready);
                indicators.insert("rsi_14".to_string(), status.rsi_ready);
                indicators.insert("macd".to_string(), status.macd_ready);
                indicators.insert("atr_14".to_string(), status.atr_ready);

                let response = WarmupStatusResponse {
                    symbol,
                    timeframe,
                    all_ready: status.all_ready(),
                    candles_processed: status.candles_processed,
                    candles_needed: status.min_candles_needed(),
                    indicators,
                };
                Ok((StatusCode::OK, Json(response)))
            }
            None => Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "not_found".to_string(),
                    message: format!(
                        "No status data found for symbol '{}' with timeframe '{}'. \
                         The pair may not be tracked yet.",
                        symbol, timeframe
                    ),
                }),
            )),
        }
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "service_unavailable".to_string(),
                message: "Indicator actor is not available".to_string(),
            }),
        ))
    }
}

/// Trigger indicator warmup from historical candle data
///
/// POST /api/v1/indicators/warmup
pub async fn warmup_indicators_handler(
    State(state): State<AppState>,
    Json(request): Json<WarmupRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    use crate::backfill::IndicatorWarmup;

    let start = std::time::Instant::now();

    // Check if indicator actor is available
    let indicator_actor = match &state.indicator_actor {
        Some(actor) => actor.clone(),
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "service_unavailable".to_string(),
                    message: "Indicator actor is not available".to_string(),
                }),
            ));
        }
    };

    // Build list of pairs to warm up
    let pairs: Vec<(String, String)> = if request.symbols.is_empty() {
        // Warm up all currently tracked pairs
        indicator_actor.get_tracked_pairs().await
    } else {
        // Warm up specified symbols
        request
            .symbols
            .iter()
            .map(|s| (s.clone(), request.timeframe.clone()))
            .collect()
    };

    if pairs.is_empty() {
        return Ok((
            StatusCode::OK,
            Json(WarmupResponse {
                success: true,
                message: "No pairs to warm up".to_string(),
                results: vec![],
                total_candles_processed: 0,
                pairs_fully_warmed: 0,
                duration_ms: start.elapsed().as_millis() as u64,
            }),
        ));
    }

    info!("API Warmup: Starting warmup for {} pairs", pairs.len());

    // Create warmup instance and run
    let mut warmup = IndicatorWarmup::new(state.storage.clone(), indicator_actor);
    let warmup_results = warmup.warmup_all(&pairs).await;

    // Convert results
    let results: Vec<WarmupPairResult> = warmup_results
        .iter()
        .map(|r| WarmupPairResult {
            symbol: r.symbol.clone(),
            timeframe: r.timeframe.clone(),
            candles_processed: r.candles_processed,
            duration_ms: r.duration_ms,
            all_indicators_ready: r.all_indicators_ready,
        })
        .collect();

    let total_candles: usize = results.iter().map(|r| r.candles_processed).sum();
    let pairs_ready = results.iter().filter(|r| r.all_indicators_ready).count();
    let duration_ms = start.elapsed().as_millis() as u64;

    info!(
        "API Warmup: Complete - {} candles, {}/{} pairs ready, {}ms",
        total_candles,
        pairs_ready,
        results.len(),
        duration_ms
    );

    Ok((
        StatusCode::OK,
        Json(WarmupResponse {
            success: true,
            message: format!(
                "Warmup complete: {} candles processed, {}/{} pairs fully warmed",
                total_candles,
                pairs_ready,
                results.len()
            ),
            results,
            total_candles_processed: total_candles,
            pairs_fully_warmed: pairs_ready,
            duration_ms,
        }),
    ))
}

/// Trigger deep warmup from Binance REST API (historical klines)
///
/// POST /api/v1/indicators/warmup/deep
pub async fn deep_warmup_indicators_handler(
    State(state): State<AppState>,
    Json(request): Json<WarmupRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    use crate::backfill::deep_warmup;

    let start = std::time::Instant::now();

    // Check if indicator actor is available
    let indicator_actor = match &state.indicator_actor {
        Some(actor) => actor.clone(),
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "service_unavailable".to_string(),
                    message: "Indicator actor is not available".to_string(),
                }),
            ));
        }
    };

    // Build list of symbols and timeframes
    let symbols: Vec<String> = if request.symbols.is_empty() {
        // Get all tracked symbols
        let pairs = indicator_actor.get_tracked_pairs().await;
        pairs.into_iter().map(|(s, _)| s).collect()
    } else {
        request.symbols.clone()
    };

    let timeframes = vec![request.timeframe.clone()];
    let candles_per_pair = request.max_candles.unwrap_or(250);

    if symbols.is_empty() {
        return Ok((
            StatusCode::OK,
            Json(WarmupResponse {
                success: true,
                message: "No symbols to warm up".to_string(),
                results: vec![],
                total_candles_processed: 0,
                pairs_fully_warmed: 0,
                duration_ms: start.elapsed().as_millis() as u64,
            }),
        ));
    }

    info!(
        "API Deep Warmup: Fetching {} candles from Binance for {} symbols",
        candles_per_pair,
        symbols.len()
    );

    // Fetch from Binance and feed to indicator actor
    let fetch_results = deep_warmup(&indicator_actor, &symbols, &timeframes, candles_per_pair)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "deep_warmup_failed".to_string(),
                    message: format!("Failed to fetch historical candles: {}", e),
                }),
            )
        })?;

    // Convert results
    let results: Vec<WarmupPairResult> = fetch_results
        .iter()
        .map(|r| WarmupPairResult {
            symbol: r.symbol.clone(),
            timeframe: r.timeframe.clone(),
            candles_processed: r.candles_fetched,
            duration_ms: r.duration_ms,
            all_indicators_ready: r.candles_fetched >= 200, // Assume ready if 200+ candles
        })
        .collect();

    let total_candles: usize = results.iter().map(|r| r.candles_processed).sum();
    let pairs_ready = results.iter().filter(|r| r.all_indicators_ready).count();
    let duration_ms = start.elapsed().as_millis() as u64;

    info!(
        "API Deep Warmup: Complete - {} candles from Binance, {}/{} pairs ready, {}ms",
        total_candles,
        pairs_ready,
        results.len(),
        duration_ms
    );

    Ok((
        StatusCode::OK,
        Json(WarmupResponse {
            success: true,
            message: format!(
                "Deep warmup complete: {} candles fetched from Binance, {}/{} pairs fully warmed",
                total_candles,
                pairs_ready,
                results.len()
            ),
            results,
            total_candles_processed: total_candles,
            pairs_fully_warmed: pairs_ready,
            duration_ms,
        }),
    ))
}

/// Get all available indicators with their descriptions
///
/// GET /api/v1/indicators/info
pub async fn get_indicators_info_handler() -> impl IntoResponse {
    let info = serde_json::json!({
        "indicators": [
            {
                "name": "ema_8",
                "description": "Exponential Moving Average (8 periods)",
                "warmup_periods": 8
            },
            {
                "name": "ema_21",
                "description": "Exponential Moving Average (21 periods)",
                "warmup_periods": 21
            },
            {
                "name": "ema_50",
                "description": "Exponential Moving Average (50 periods)",
                "warmup_periods": 50
            },
            {
                "name": "ema_200",
                "description": "Exponential Moving Average (200 periods)",
                "warmup_periods": 200
            },
            {
                "name": "rsi_14",
                "description": "Relative Strength Index (14 periods)",
                "warmup_periods": 15,
                "range": [0, 100],
                "oversold": 30,
                "overbought": 70
            },
            {
                "name": "macd_line",
                "description": "MACD Line (12/26 EMA difference)",
                "warmup_periods": 26
            },
            {
                "name": "macd_signal",
                "description": "MACD Signal Line (9-period EMA of MACD)",
                "warmup_periods": 35
            },
            {
                "name": "macd_histogram",
                "description": "MACD Histogram (MACD - Signal)",
                "warmup_periods": 35
            },
            {
                "name": "atr_14",
                "description": "Average True Range (14 periods)",
                "warmup_periods": 14
            }
        ],
        "timeframes_supported": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "calculation_mode": "incremental",
        "update_trigger": "on_candle_close"
    });

    (StatusCode::OK, Json(info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicator_response_serialization() {
        let response = IndicatorResponse {
            symbol: "BTC-USDT".to_string(),
            exchange: "binance".to_string(),
            timeframe: "1m".to_string(),
            timestamp: 1672531200000,
            close: 50000.0,
            ema_8: Some(49900.0),
            ema_21: Some(49800.0),
            ema_50: None,
            ema_200: None,
            rsi_14: Some(55.5),
            macd_line: Some(100.0),
            macd_signal: Some(90.0),
            macd_histogram: Some(10.0),
            atr_14: Some(500.0),
            calculated_at: 1672531200100,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("BTC-USDT"));
        assert!(json.contains("49900"));
        // None values should be skipped
        assert!(!json.contains("ema_50"));
        assert!(!json.contains("ema_200"));
    }

    #[test]
    fn test_warmup_status_response() {
        let mut indicators = HashMap::new();
        indicators.insert("ema_8".to_string(), true);
        indicators.insert("ema_200".to_string(), false);

        let response = WarmupStatusResponse {
            symbol: "BTC-USDT".to_string(),
            timeframe: "1m".to_string(),
            all_ready: false,
            candles_processed: 50,
            candles_needed: 200,
            indicators,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("candles_processed"));
        assert!(json.contains("50"));
    }

    #[test]
    fn test_error_response() {
        let response = ErrorResponse {
            error: "not_found".to_string(),
            message: "Symbol not found".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("not_found"));
        assert!(json.contains("Symbol not found"));
    }
}
