// Gap Analysis API endpoint
//
// Detects gaps (missing data periods) in historical trade data
// Uses optimized QuestDB window functions for precise gap detection

use axum::{
    Json,
    extract::{Query, State},
    http::StatusCode,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

use crate::api::AppState;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Invalid symbol: {0} - must be alphanumeric with optional dash/dot")]
    InvalidSymbol(String),

    #[error("Invalid exchange: {0} - not in whitelist")]
    InvalidExchange(String),

    #[error("Symbol too long: {0} characters (max 20)")]
    SymbolTooLong(usize),

    #[error("Exchange too long: {0} characters (max 20)")]
    ExchangeTooLong(usize),
}

#[derive(Debug, Deserialize)]
pub struct GapAnalysisQuery {
    /// Symbol to analyze (e.g., "BTC-USDT")
    pub symbol: String,

    /// Start time (RFC3339 format)
    pub start_time: String,

    /// End time (RFC3339 format)
    pub end_time: String,

    /// Minimum gap duration in milliseconds
    #[serde(default = "default_threshold")]
    pub threshold_ms: i64,

    /// Optional exchange filter
    pub exchange: Option<String>,
}

fn default_threshold() -> i64 {
    5000 // 5 seconds default
}

#[derive(Debug, Serialize)]
pub struct Gap {
    /// Timestamp when gap started (last received message before gap)
    pub gap_start: String,

    /// Timestamp when gap ended (first message after gap)
    pub gap_end: String,

    /// Duration of gap in milliseconds
    pub duration_ms: i64,

    /// Symbol
    pub symbol: String,

    /// Exchange
    pub exchange: String,
}

#[derive(Debug, Serialize)]
pub struct GapAnalysisResponse {
    /// List of detected gaps
    pub gaps: Vec<Gap>,

    /// Total number of gaps found
    pub total_gaps: i32,

    /// Total missing time in milliseconds
    pub total_missing_ms: i64,

    /// Query execution time in milliseconds
    pub query_time_ms: i64,
}

/// Gap analysis handler
///
/// Uses QuestDB window functions to detect gaps in trade data.
/// Corrected SQL from research document:
/// - Proper timestamp filtering syntax for QuestDB
/// - Correct datediff usage ('u' for microseconds)
/// - NULL handling for first row (no previous timestamp)
pub async fn get_gaps_handler(
    State(state): State<AppState>,
    Query(params): Query<GapAnalysisQuery>,
) -> Result<Json<GapAnalysisResponse>, (StatusCode, String)> {
    let query_start = Instant::now();

    // Validate and parse timestamps
    let start_time = params.start_time.parse::<DateTime<Utc>>().map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid start_time: {}", e),
        )
    })?;

    let end_time = params
        .end_time
        .parse::<DateTime<Utc>>()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid end_time: {}", e)))?;

    // Build the gap analysis query with validation
    let query = match build_gap_query(
        &params.symbol,
        &start_time,
        &end_time,
        params.threshold_ms,
        params.exchange.as_deref(),
    ) {
        Ok(q) => q,
        Err(e) => {
            return Err((StatusCode::BAD_REQUEST, format!("Invalid input: {}", e)));
        }
    };

    tracing::debug!("Executing gap analysis query: {}", query);

    // Execute query via QuestDB HTTP API
    match execute_gap_query(&state, &query).await {
        Ok(gaps) => {
            let total_gaps = gaps.len() as i32;
            let total_missing_ms: i64 = gaps.iter().map(|g| g.duration_ms).sum();
            let query_time_ms = query_start.elapsed().as_millis() as i64;

            Ok(Json(GapAnalysisResponse {
                gaps,
                total_gaps,
                total_missing_ms,
                query_time_ms,
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Query execution failed: {}", e),
        )),
    }
}

/// Validate symbol input
///
/// Symbols must be alphanumeric with optional dash or dot, max 20 chars
fn validate_symbol(s: &str) -> Result<(), ValidationError> {
    if s.len() > 20 {
        return Err(ValidationError::SymbolTooLong(s.len()));
    }

    if !s
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '.')
    {
        return Err(ValidationError::InvalidSymbol(s.to_string()));
    }

    Ok(())
}

/// Validate exchange input
///
/// Exchanges must be in the known whitelist
fn validate_exchange(s: &str) -> Result<(), ValidationError> {
    const KNOWN_EXCHANGES: &[&str] = &[
        "binance", "bybit", "okx", "kucoin", "coinbase", "kraken", "bitfinex", "huobi", "gateio",
        "mexc",
    ];

    if s.len() > 20 {
        return Err(ValidationError::ExchangeTooLong(s.len()));
    }

    let s_lower = s.to_lowercase();
    if KNOWN_EXCHANGES.contains(&s_lower.as_str()) {
        Ok(())
    } else {
        Err(ValidationError::InvalidExchange(s.to_string()))
    }
}

/// Build the gap analysis SQL query
///
/// Corrected query from research:
/// - Uses proper QuestDB timestamp syntax (>= and <)
/// - Uses datediff('u', ...) for microseconds, then divides by 1000 for milliseconds
/// - Filters out NULL prev_time (first row has no previous)
/// - Threshold comparison in microseconds (threshold_ms * 1000)
/// - Validates inputs before interpolation to prevent SQL injection
fn build_gap_query(
    symbol: &str,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
    threshold_ms: i64,
    exchange: Option<&str>,
) -> Result<String, ValidationError> {
    // Validate symbol (required)
    validate_symbol(symbol)?;

    let threshold_us = threshold_ms * 1000; // Convert to microseconds

    let exchange_filter = if let Some(ex) = exchange {
        // Validate exchange if provided
        validate_exchange(ex)?;
        format!(" AND exchange = '{}'", ex.to_lowercase())
    } else {
        String::new()
    };

    // Now safe to interpolate validated inputs
    Ok(format!(
        r#"
WITH time_diffs AS (
    SELECT
        symbol,
        exchange,
        ts AS current_time,
        lag(ts) OVER (PARTITION BY symbol, exchange ORDER BY ts) AS prev_time
    FROM trades_crypto
    WHERE symbol = '{symbol}'
      AND ts >= '{start}'
      AND ts < '{end}'
      {exchange_filter}
)
SELECT
    prev_time AS gap_start,
    current_time AS gap_end,
    datediff('u', prev_time, current_time) / 1000 AS gap_duration_ms,
    symbol,
    exchange
FROM time_diffs
WHERE datediff('u', prev_time, current_time) > {threshold_us}
  AND prev_time IS NOT NULL
ORDER BY gap_duration_ms DESC
LIMIT 1000
"#,
        symbol = symbol,
        start = start.to_rfc3339(),
        end = end.to_rfc3339(),
        threshold_us = threshold_us,
        exchange_filter = exchange_filter
    ))
}

/// Execute gap query against QuestDB HTTP API
///
/// QuestDB HTTP API endpoint: http://questdb:9000/exec
async fn execute_gap_query(
    _state: &AppState,
    query: &str,
) -> Result<Vec<Gap>, Box<dyn std::error::Error>> {
    // Get QuestDB HTTP endpoint from config
    let questdb_host = std::env::var("QUESTDB_HOST").unwrap_or_else(|_| "questdb".to_string());
    let questdb_http_port =
        std::env::var("QUESTDB_HTTP_PORT").unwrap_or_else(|_| "9000".to_string());

    let url = format!("http://{}:{}/exec", questdb_host, questdb_http_port);

    // Execute query
    let client = reqwest::Client::new();
    let response = client.get(&url).query(&[("query", query)]).send().await?;

    if !response.status().is_success() {
        let error_text = response.text().await?;
        return Err(format!("QuestDB query failed: {}", error_text).into());
    }

    let json: serde_json::Value = response.json().await?;

    // Parse QuestDB response format
    // Response structure:
    // {
    //   "query": "...",
    //   "columns": [...],
    //   "dataset": [
    //     ["2025-01-01T10:00:00.000000Z", "2025-01-01T10:05:30.000000Z", 330000, "BTC-USDT", "binance"],
    //     ...
    //   ],
    //   "count": 5
    // }

    let dataset = json["dataset"]
        .as_array()
        .ok_or("Missing dataset in response")?;

    let mut gaps = Vec::new();

    for row in dataset {
        let row_array = row.as_array().ok_or("Invalid row format")?;

        if row_array.len() >= 5 {
            let gap = Gap {
                gap_start: row_array[0].as_str().unwrap_or_default().to_string(),
                gap_end: row_array[1].as_str().unwrap_or_default().to_string(),
                duration_ms: row_array[2].as_i64().unwrap_or(0),
                symbol: row_array[3].as_str().unwrap_or_default().to_string(),
                exchange: row_array[4].as_str().unwrap_or_default().to_string(),
            };
            gaps.push(gap);
        }
    }

    Ok(gaps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_validation() {
        // Valid symbols
        assert!(validate_symbol("BTC-USDT").is_ok());
        assert!(validate_symbol("ETH.USD").is_ok());
        assert!(validate_symbol("SOL").is_ok());
        assert!(validate_symbol("BTC123").is_ok());

        // Invalid symbols
        assert!(validate_symbol("'; DROP TABLE trades; --").is_err());
        assert!(validate_symbol("BTC USDT").is_err()); // spaces not allowed
        assert!(validate_symbol("BTC/USDT").is_err()); // slash not allowed
        assert!(validate_symbol("A".repeat(21).as_str()).is_err()); // too long
    }

    #[test]
    fn test_exchange_validation() {
        // Valid exchanges
        assert!(validate_exchange("binance").is_ok());
        assert!(validate_exchange("Binance").is_ok()); // case insensitive
        assert!(validate_exchange("BYBIT").is_ok());
        assert!(validate_exchange("okx").is_ok());

        // Invalid exchanges
        assert!(validate_exchange("unknown_exchange").is_err());
        assert!(validate_exchange("'; DROP TABLE").is_err());
        assert!(validate_exchange("A".repeat(21).as_str()).is_err());
    }

    #[test]
    fn test_gap_query_generation() {
        let start = "2025-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let end = "2025-01-02T00:00:00Z".parse::<DateTime<Utc>>().unwrap();

        let query = build_gap_query("BTC-USDT", &start, &end, 5000, None).unwrap();

        assert!(query.contains("BTC-USDT"));
        assert!(query.contains("5000000")); // 5000ms * 1000 = 5000000 microseconds
        assert!(query.contains("datediff('u'"));
        assert!(query.contains("prev_time IS NOT NULL"));
    }

    #[test]
    fn test_gap_query_with_exchange_filter() {
        let start = "2025-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let end = "2025-01-02T00:00:00Z".parse::<DateTime<Utc>>().unwrap();

        let query = build_gap_query("BTC-USDT", &start, &end, 5000, Some("binance")).unwrap();

        assert!(query.contains("AND exchange = 'binance'"));
    }

    #[test]
    fn test_gap_query_injection_prevention() {
        let start = "2025-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let end = "2025-01-02T00:00:00Z".parse::<DateTime<Utc>>().unwrap();

        // Injection attempts should be rejected
        let result = build_gap_query("'; DROP TABLE trades; --", &start, &end, 5000, None);
        assert!(result.is_err());

        let result = build_gap_query("BTC-USDT", &start, &end, 5000, Some("'; DROP TABLE"));
        assert!(result.is_err());
    }

    #[test]
    fn test_default_threshold() {
        assert_eq!(default_threshold(), 5000);
    }

    #[test]
    fn test_gap_serialization() {
        let gap = Gap {
            gap_start: "2025-01-01T10:00:00.000000Z".to_string(),
            gap_end: "2025-01-01T10:05:30.000000Z".to_string(),
            duration_ms: 330000,
            symbol: "BTC-USDT".to_string(),
            exchange: "binance".to_string(),
        };

        let json = serde_json::to_string(&gap).unwrap();
        assert!(json.contains("330000"));
        assert!(json.contains("BTC-USDT"));
    }
}
