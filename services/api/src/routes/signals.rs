//! Signal API routes for the JANUS Rust Gateway.
//!
//! Provides endpoints for signal retrieval, generation, and dispatch.
//!
//! Port of `gateway/src/routers/signals.py`.

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::redis_dispatcher::{DispatchResult, Signal};
use crate::state::AppState;

/// Query parameters for signal generation.
#[derive(Debug, Deserialize)]
pub struct GenerateSignalsQuery {
    #[serde(default = "default_category")]
    pub category: String,
    pub symbols: Option<String>,
    #[serde(default)]
    pub ai_enhanced: bool,
}

fn default_category() -> String {
    "swing".to_string()
}

/// Response for signal generation request.
#[derive(Debug, Serialize)]
pub struct GenerateSignalsResponse {
    pub task_id: String,
    pub status: String,
    pub message: String,
    pub category: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbols: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
}

/// Request body for manual signal dispatch.
#[derive(Debug, Deserialize)]
pub struct DispatchSignalRequest {
    pub symbol: String,
    pub side: String,
    #[serde(default = "default_strength")]
    pub strength: f64,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    pub predicted_duration_seconds: Option<i64>,
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
}

fn default_strength() -> f64 {
    0.5
}

fn default_confidence() -> f64 {
    0.5
}

/// Query parameters for loading signals from files.
#[derive(Debug, Deserialize)]
pub struct GetSignalsFromFilesQuery {
    pub date: Option<String>,
    pub category: Option<String>,
    pub symbol: Option<String>,
    #[serde(default = "default_true")]
    pub include_lot_size: bool,
}

fn default_true() -> bool {
    true
}

/// Query parameters for signal summary.
#[derive(Debug, Deserialize)]
pub struct SignalSummaryQuery {
    #[serde(default = "default_category")]
    pub category: String,
    pub symbols: Option<String>,
}

/// Signal response model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResponse {
    pub symbol: String,
    pub signal_type: String,
    pub category: String,
    pub entry_price: f64,
    pub take_profit: f64,
    pub stop_loss: f64,
    pub take_profit_pct: f64,
    pub stop_loss_pct: f64,
    pub risk_reward_ratio: f64,
    pub position_size_pct: f64,
    pub strength: String,
    pub confidence: f64,
    pub timestamp: String,
    pub is_valid: bool,
}

/// Signals by symbol response.
#[derive(Debug, Serialize)]
pub struct SignalsBySymbolResponse {
    pub symbol: String,
    pub date: String,
    pub signals: Vec<SignalResponse>,
}

/// Signal summary response.
#[derive(Debug, Serialize)]
pub struct SignalSummaryResponse {
    pub category: String,
    pub symbols: Vec<String>,
    pub total_signals: i32,
    pub strong_signals: i32,
    pub average_confidence: f64,
    pub message: String,
}

/// Category configuration.
#[derive(Debug, Clone, Serialize)]
pub struct CategoryConfig {
    pub category: String,
    pub description: String,
    pub time_horizon_min_hours: f64,
    pub time_horizon_max_hours: f64,
}

/// Categories response.
#[derive(Debug, Serialize)]
pub struct CategoriesResponse {
    pub categories: Vec<CategoryConfig>,
}

/// Generate trading signals endpoint.
///
/// This endpoint triggers signal generation (placeholder for now, will integrate with task queue).
pub async fn generate_signals(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<GenerateSignalsQuery>,
) -> impl IntoResponse {
    info!(
        "Signal generation requested: category={}, symbols={:?}, ai_enhanced={}",
        params.category, params.symbols, params.ai_enhanced
    );

    // For now, return a placeholder response
    // In full implementation, this would queue a task via Redis/background worker
    let response = if let Some(symbols) = params.symbols {
        let symbol_list: Vec<String> = symbols
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .filter(|s| !s.is_empty())
            .collect();

        GenerateSignalsResponse {
            task_id: uuid::Uuid::new_v4().to_string(),
            status: "queued".to_string(),
            message: format!(
                "Batch signal generation queued for {} symbols",
                symbol_list.len()
            ),
            category: params.category,
            symbols: Some(symbol_list),
            symbol: None,
        }
    } else {
        let symbol = "BTCUSD".to_string();
        GenerateSignalsResponse {
            task_id: uuid::Uuid::new_v4().to_string(),
            status: "queued".to_string(),
            message: format!("Signal generation queued for {}", symbol),
            category: params.category,
            symbols: None,
            symbol: Some(symbol),
        }
    };

    (StatusCode::OK, Json(response))
}

/// Manually dispatch a signal to the Rust forward service via Redis.
///
/// This endpoint allows direct signal dispatch for testing or manual triggers.
pub async fn dispatch_signal(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DispatchSignalRequest>,
) -> impl IntoResponse {
    if !state.signal_dispatcher.is_connected().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(DispatchResult::error("Signal dispatcher not available")),
        );
    }

    let signal = Signal::new(&request.symbol, &request.side)
        .with_strength(request.strength)
        .with_confidence(request.confidence)
        .with_prices(request.entry_price, request.stop_loss, request.take_profit);

    // Add predicted duration if provided
    let signal = if let Some(duration) = request.predicted_duration_seconds {
        Signal {
            predicted_duration_seconds: Some(duration),
            ..signal
        }
    } else {
        signal
    };

    match state.signal_dispatcher.dispatch_signal(&signal).await {
        Ok(subscribers) => {
            let result = DispatchResult::success(&request.symbol, subscribers);
            (StatusCode::OK, Json(result))
        }
        Err(e) => {
            error!("Error dispatching signal: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(DispatchResult::error(format!(
                    "Failed to dispatch signal: {}",
                    e
                ))),
            )
        }
    }
}

/// Load signals from JSON files in signals directory.
///
/// Uses the SignalService to load and enrich signals (placeholder implementation).
pub async fn get_signals_from_files(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<GetSignalsFromFilesQuery>,
) -> impl IntoResponse {
    let date = params
        .date
        .unwrap_or_else(|| Utc::now().format("%Y%m%d").to_string());

    // Placeholder response - in full implementation, this would read from signals directory
    let response = serde_json::json!({
        "date": date,
        "signals": {
            "scalp": [],
            "intraday": [],
            "swing": [],
            "long_term": []
        },
        "lot_size_enabled": params.include_lot_size,
        "category_filter": params.category,
        "symbol_filter": params.symbol,
        "message": "Signal file loading will be implemented"
    });

    (StatusCode::OK, Json(response))
}

/// Get a specific signal by ID.
pub async fn get_signal_by_id(
    State(_state): State<Arc<AppState>>,
    Path(signal_id): Path<String>,
    Query(params): Query<GetSignalsFromFilesQuery>,
) -> impl IntoResponse {
    let _date = params
        .date
        .unwrap_or_else(|| Utc::now().format("%Y%m%d").to_string());

    // Placeholder - in full implementation, this would look up the signal
    warn!("Signal lookup not yet implemented: {}", signal_id);

    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": "not_found",
            "message": format!("Signal {} not found", signal_id)
        })),
    )
}

/// Get all signals for a specific symbol.
pub async fn get_signals_by_symbol(
    State(_state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
    Query(params): Query<GetSignalsFromFilesQuery>,
) -> impl IntoResponse {
    let date = params
        .date
        .unwrap_or_else(|| Utc::now().format("%Y%m%d").to_string());

    let response = SignalsBySymbolResponse {
        symbol: symbol.to_uppercase(),
        date,
        signals: vec![], // Placeholder
    };

    (StatusCode::OK, Json(response))
}

/// Get signal summary statistics.
pub async fn get_signal_summary(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<SignalSummaryQuery>,
) -> impl IntoResponse {
    info!(
        "Signal summary requested: category={}, symbols={:?}",
        params.category, params.symbols
    );

    let symbols: Vec<String> = params
        .symbols
        .map(|s| {
            s.split(',')
                .map(|sym| sym.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let response = SignalSummaryResponse {
        category: params.category,
        symbols,
        total_signals: 0,
        strong_signals: 0,
        average_confidence: 0.0,
        message: "Signal summary calculation will be implemented".to_string(),
    };

    (StatusCode::OK, Json(response))
}

/// Get all trade categories and their configurations.
pub async fn get_categories() -> impl IntoResponse {
    let categories = vec![
        CategoryConfig {
            category: "scalp".to_string(),
            description: "Short-term trades, minutes to hours".to_string(),
            time_horizon_min_hours: 0.25,
            time_horizon_max_hours: 4.0,
        },
        CategoryConfig {
            category: "intraday".to_string(),
            description: "Day trading, hours".to_string(),
            time_horizon_min_hours: 1.0,
            time_horizon_max_hours: 24.0,
        },
        CategoryConfig {
            category: "swing".to_string(),
            description: "Swing trading, days to weeks".to_string(),
            time_horizon_min_hours: 24.0,
            time_horizon_max_hours: 336.0, // 2 weeks
        },
        CategoryConfig {
            category: "long_term".to_string(),
            description: "Long-term positions, weeks to months".to_string(),
            time_horizon_min_hours: 336.0,
            time_horizon_max_hours: 2160.0, // 90 days
        },
    ];

    let response = CategoriesResponse { categories };

    (StatusCode::OK, Json(response))
}

/// Create the signals routes router.
pub fn signal_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Signal generation
        .route("/api/signals/generate", get(generate_signals))
        // Manual dispatch
        .route("/api/signals/dispatch", post(dispatch_signal))
        // Load from files
        .route("/api/signals/from-files", get(get_signals_from_files))
        // Get by ID
        .route("/api/signals/by-id/{signal_id}", get(get_signal_by_id))
        // Get by symbol
        .route(
            "/api/signals/by-symbol/{symbol}",
            get(get_signals_by_symbol),
        )
        // Summary
        .route("/api/signals/summary", get(get_signal_summary))
        // Categories
        .route("/api/signals/categories", get(get_categories))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_signals_query_defaults() {
        let query: GenerateSignalsQuery = serde_json::from_str("{}").unwrap();
        assert_eq!(query.category, "swing");
        assert!(query.symbols.is_none());
        assert!(!query.ai_enhanced);
    }

    #[test]
    fn test_dispatch_signal_request_defaults() {
        let request: DispatchSignalRequest =
            serde_json::from_str(r#"{"symbol": "BTCUSD", "side": "Buy"}"#).unwrap();
        assert_eq!(request.symbol, "BTCUSD");
        assert_eq!(request.side, "Buy");
        assert_eq!(request.strength, 0.5);
        assert_eq!(request.confidence, 0.5);
    }

    #[test]
    fn test_category_config_serialization() {
        let config = CategoryConfig {
            category: "swing".to_string(),
            description: "Swing trading".to_string(),
            time_horizon_min_hours: 24.0,
            time_horizon_max_hours: 336.0,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("swing"));
        assert!(json.contains("24"));
    }

    #[test]
    fn test_categories_response() {
        let response = CategoriesResponse {
            categories: vec![CategoryConfig {
                category: "scalp".to_string(),
                description: "Short-term".to_string(),
                time_horizon_min_hours: 0.25,
                time_horizon_max_hours: 4.0,
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("scalp"));
        assert!(json.contains("categories"));
    }
}
