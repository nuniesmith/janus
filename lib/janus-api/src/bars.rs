//! REST candle history — `GET /bars/{symbol}` and `GET /bars/{symbol}/candles`.
//!
//! The Track D WebUI data contract (see `docs/PUBLIC_API.md` and `TODO.md`):
//! the FKS dashboard's charts load their **initial history** from these routes
//! and then tail live closed bars over [`/sse/bars/{symbol}`](crate::sse_bars).
//! Both read the `candles_crypto` QuestDB table that the Data module's candle
//! sink persists (#106), over QuestDB's HTTP `/exec` API (`QUESTDB_HTTP_URL`,
//! default `http://questdb:9000`).
//!
//! Two response shapes, mirroring what the WebUI already parses (it was built
//! against the retired Python data service — wire-compat means the UI works
//! unmodified):
//!
//! - `GET /bars/{symbol}?interval=1m&days_back=5&limit=1000` — **columnar**:
//!   `{"columns": ["timestamp", "open", ...], "data": [[ts, o, h, l, c, v], …]}`
//!   with `timestamp` as an ISO-8601 string (`new Date(row[0])` on the client).
//! - `GET /bars/{symbol}/candles?interval=1m&days_back=3&limit=500` — **flat**:
//!   `{"candles": [{"timestamp": <unix ms>, "open": …, "volume": …}, …]}`.
//!
//! Symbol matching is separator-insensitive (`BTCUSDT` ≡ `BTC-USDT` ≡
//! `btc/usdt`), mirroring the SSE route's normalisation, by comparing against
//! a separator-stripped uppercase form in SQL. All inputs are validated before
//! interpolation (QuestDB's HTTP API has no bind parameters). Everything but
//! the HTTP fetch is pure and unit-tested.

use std::sync::OnceLock;

use axum::{
    Json,
    extract::{Path, Query},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use serde_json::{Value, json};

/// Allowed bar intervals (matches the data service's whitelist).
const VALID_INTERVALS: &[&str] = &[
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M",
];

/// Hard caps so a bad query can't ask QuestDB for unbounded work.
const MAX_DAYS_BACK: u32 = 90;
const MAX_LIMIT: usize = 10_000;

fn default_interval() -> String {
    "1m".to_string()
}
fn default_days_back() -> u32 {
    5
}
fn default_limit() -> usize {
    1_000
}

/// Query parameters shared by both history routes.
#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    /// Bar interval (whitelisted; default `1m`).
    #[serde(default = "default_interval")]
    pub interval: String,
    /// Trailing window in days (clamped to 1..=90; default 5).
    #[serde(default = "default_days_back")]
    pub days_back: u32,
    /// Maximum bars returned (clamped to 1..=10_000; default 1000).
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// One parsed candle row.
#[derive(Debug, Clone, PartialEq)]
pub struct BarRow {
    /// ISO-8601 open time (as QuestDB returned it) — used by the columnar shape.
    pub ts_iso: String,
    /// Open time in unix milliseconds — used by the `/candles` shape.
    pub ts_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Validate a client-sent symbol: 1–20 chars of `[A-Za-z0-9_-/]`, starting
/// alphanumeric. Returns the **normalised** form (uppercase, separators
/// stripped) used for the SQL comparison.
fn validate_symbol(symbol: &str) -> Result<String, &'static str> {
    if symbol.is_empty() || symbol.len() > 20 {
        return Err("symbol must be 1-20 characters");
    }
    let mut chars = symbol.chars();
    let first = chars.next().expect("non-empty checked above");
    if !first.is_ascii_alphanumeric() {
        return Err("symbol must start with a letter or digit");
    }
    if !symbol
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '/' | '_'))
    {
        return Err("symbol contains invalid characters");
    }
    Ok(symbol
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_uppercase())
}

/// Validate the interval against the whitelist.
fn validate_interval(interval: &str) -> Result<&str, &'static str> {
    VALID_INTERVALS
        .iter()
        .find(|v| **v == interval)
        .copied()
        .ok_or("unsupported interval")
}

/// Build the QuestDB query. Inputs must already be validated/normalised —
/// `norm_symbol` is alphanumeric-uppercase, `interval` whitelisted, numbers
/// clamped — so interpolation is safe (the HTTP API has no bind parameters).
fn build_query(norm_symbol: &str, interval: &str, days_back: u32, limit: usize) -> String {
    let days = days_back.clamp(1, MAX_DAYS_BACK);
    let limit = limit.clamp(1, MAX_LIMIT);
    format!(
        "SELECT timestamp, open, high, low, close, volume FROM candles_crypto \
         WHERE upper(replace(replace(replace(symbol, '-', ''), '/', ''), '_', '')) = '{norm_symbol}' \
         AND interval = '{interval}' \
         AND timestamp > dateadd('d', -{days}, now()) \
         ORDER BY timestamp ASC LIMIT {limit}"
    )
}

/// Parse one timestamp cell from a QuestDB `/exec` response into
/// `(iso_string, unix_ms)`. QuestDB returns designated timestamps as ISO-8601
/// strings; a numeric cell is handled defensively by magnitude (µs / ms / s).
fn parse_ts(cell: &Value) -> Option<(String, i64)> {
    match cell {
        Value::String(s) => {
            let dt = chrono::DateTime::parse_from_rfc3339(s).ok()?;
            Some((s.clone(), dt.timestamp_millis()))
        }
        Value::Number(n) => {
            let v = n.as_i64()?;
            let ms = if v >= 100_000_000_000_000 {
                v / 1_000 // microseconds
            } else if v >= 100_000_000_000 {
                v // already milliseconds
            } else {
                v.checked_mul(1_000)? // seconds
            };
            let iso = chrono::DateTime::from_timestamp_millis(ms)?
                .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
            Some((iso, ms))
        }
        _ => None,
    }
}

/// Parse a QuestDB `/exec` JSON response (`{"columns": …, "dataset": [[…]]}`)
/// into rows. Malformed rows are skipped rather than failing the whole load.
fn parse_exec_response(body: &Value) -> Vec<BarRow> {
    let Some(dataset) = body.get("dataset").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut rows = Vec::with_capacity(dataset.len());
    for row in dataset {
        let Some(cells) = row.as_array() else {
            continue;
        };
        if cells.len() < 6 {
            continue;
        }
        let Some((ts_iso, ts_ms)) = parse_ts(&cells[0]) else {
            continue;
        };
        // Every OHLCV cell must be numeric; a null/string cell means the row is
        // malformed — skip it (like a bad timestamp) rather than render a
        // misleading 0.0 candle that would draw a wick crashing to zero.
        let (Some(open), Some(high), Some(low), Some(close), Some(volume)) = (
            cells[1].as_f64(),
            cells[2].as_f64(),
            cells[3].as_f64(),
            cells[4].as_f64(),
            cells[5].as_f64(),
        ) else {
            continue;
        };
        rows.push(BarRow {
            ts_iso,
            ts_ms,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    rows
}

/// Render the columnar `/bars/{symbol}` shape the trading page parses
/// (`data.columns` + `data.data`, `row[0]` fed to `new Date()`).
fn columnar_response(symbol: &str, interval: &str, rows: &[BarRow]) -> Value {
    json!({
        "symbol": symbol,
        "interval": interval,
        "columns": ["timestamp", "open", "high", "low", "close", "volume"],
        "data": rows
            .iter()
            .map(|r| json!([r.ts_iso, r.open, r.high, r.low, r.close, r.volume]))
            .collect::<Vec<_>>(),
        "count": rows.len(),
    })
}

/// Render the flat `/bars/{symbol}/candles` shape the chart components parse
/// (`data.candles[].timestamp` in unix **milliseconds**).
fn candles_response(symbol: &str, interval: &str, rows: &[BarRow]) -> Value {
    json!({
        "symbol": symbol,
        "interval": interval,
        "candles": rows
            .iter()
            .map(|r| json!({
                "timestamp": r.ts_ms,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }))
            .collect::<Vec<_>>(),
        "count": rows.len(),
    })
}

/// QuestDB HTTP endpoint (`QUESTDB_HTTP_URL`), resolved once.
fn questdb_http_url() -> &'static str {
    static URL: OnceLock<String> = OnceLock::new();
    URL.get_or_init(|| {
        std::env::var("QUESTDB_HTTP_URL").unwrap_or_else(|_| "http://questdb:9000".to_string())
    })
}

fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("static reqwest client config is valid")
    })
}

/// Validate inputs, run the QuestDB query, and parse the rows.
async fn fetch_bars(
    symbol: &str,
    query: &HistoryQuery,
) -> Result<Vec<BarRow>, (StatusCode, Json<Value>)> {
    let bad_request = |msg: &str| (StatusCode::BAD_REQUEST, Json(json!({ "error": msg })));
    let norm_symbol = validate_symbol(symbol).map_err(bad_request)?;
    let interval = validate_interval(&query.interval).map_err(bad_request)?;

    let sql = build_query(&norm_symbol, interval, query.days_back, query.limit);
    let resp = http_client()
        .get(format!("{}/exec", questdb_http_url()))
        .query(&[("query", sql.as_str())])
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("QuestDB unreachable: {e}") })),
            )
        })?;
    if !resp.status().is_success() {
        let status = resp.status();
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("QuestDB query failed ({status})") })),
        ));
    }
    let body: Value = resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("bad QuestDB response: {e}") })),
        )
    })?;
    Ok(parse_exec_response(&body))
}

/// `GET /bars/{symbol}` — columnar history (the trading page's chart load).
pub async fn bars_history_handler(
    Path(symbol): Path<String>,
    Query(query): Query<HistoryQuery>,
) -> impl IntoResponse {
    match fetch_bars(&symbol, &query).await {
        Ok(rows) => (
            StatusCode::OK,
            Json(columnar_response(&symbol, &query.interval, &rows)),
        ),
        Err(e) => e,
    }
}

/// `GET /bars/{symbol}/candles` — flat history (MiniChart + charts page).
pub async fn bars_candles_handler(
    Path(symbol): Path<String>,
    Query(query): Query<HistoryQuery>,
) -> impl IntoResponse {
    match fetch_bars(&symbol, &query).await {
        Ok(rows) => (
            StatusCode::OK,
            Json(candles_response(&symbol, &query.interval, &rows)),
        ),
        Err(e) => e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_validation_accepts_common_forms() {
        assert_eq!(validate_symbol("BTCUSDT").unwrap(), "BTCUSDT");
        assert_eq!(validate_symbol("BTC-USDT").unwrap(), "BTCUSDT");
        assert_eq!(validate_symbol("btc/usdt").unwrap(), "BTCUSDT");
        assert_eq!(validate_symbol("btc_usdt").unwrap(), "BTCUSDT");
        assert_eq!(validate_symbol("MGC").unwrap(), "MGC");
    }

    #[test]
    fn symbol_validation_rejects_injection_and_garbage() {
        assert!(validate_symbol("").is_err());
        assert!(validate_symbol("'; DROP TABLE candles_crypto; --").is_err());
        assert!(validate_symbol("BTC USDT").is_err());
        assert!(validate_symbol("-BTC").is_err());
        assert!(validate_symbol(&"A".repeat(21)).is_err());
    }

    #[test]
    fn interval_whitelist() {
        assert!(validate_interval("1m").is_ok());
        assert!(validate_interval("4h").is_ok());
        assert!(validate_interval("1M").is_ok());
        assert!(validate_interval("7m").is_err());
        assert!(validate_interval("1m; DROP").is_err());
    }

    #[test]
    fn build_query_normalises_and_clamps() {
        let q = build_query("BTCUSDT", "1m", 500, 1_000_000);
        assert!(q.contains("= 'BTCUSDT'"));
        assert!(q.contains("interval = '1m'"));
        assert!(q.contains(&format!("dateadd('d', -{MAX_DAYS_BACK}, now())")));
        assert!(q.contains(&format!("LIMIT {MAX_LIMIT}")));
        let q = build_query("ETHUSDT", "5m", 0, 0);
        assert!(q.contains("dateadd('d', -1, now())"));
        assert!(q.contains("LIMIT 1"));
    }

    #[test]
    fn parse_ts_iso_and_numeric() {
        // ISO string (QuestDB's default rendering).
        let (iso, ms) = parse_ts(&json!("2026-06-11T00:00:00.000000Z")).unwrap();
        assert_eq!(iso, "2026-06-11T00:00:00.000000Z");
        assert_eq!(ms, 1_781_136_000_000);
        // Microseconds / milliseconds / seconds by magnitude.
        assert_eq!(
            parse_ts(&json!(1_781_136_000_000_000i64)).unwrap().1,
            1_781_136_000_000
        );
        assert_eq!(
            parse_ts(&json!(1_781_136_000_000i64)).unwrap().1,
            1_781_136_000_000
        );
        assert_eq!(
            parse_ts(&json!(1_781_136_000i64)).unwrap().1,
            1_781_136_000_000
        );
        // Garbage.
        assert!(parse_ts(&json!("not-a-date")).is_none());
        assert!(parse_ts(&json!(null)).is_none());
    }

    fn sample_body() -> Value {
        json!({
            "columns": [
                {"name": "timestamp"}, {"name": "open"}, {"name": "high"},
                {"name": "low"}, {"name": "close"}, {"name": "volume"}
            ],
            "dataset": [
                ["2026-06-11T00:00:00.000000Z", 100.0, 110.0, 95.0, 105.0, 12.5],
                ["2026-06-11T00:01:00.000000Z", 105.0, 106.0, 99.0, 100.0, 8.0],
                ["bad-ts", 1.0, 1.0, 1.0, 1.0, 1.0],
                ["2026-06-11T00:02:00.000000Z", 1.0, null, 1.0, 1.0, 1.0],
                [12345]
            ],
            "count": 5
        })
    }

    #[test]
    fn parse_exec_response_skips_malformed_rows() {
        // Fixture has 5 rows: 2 valid, plus a bad timestamp, a null OHLCV cell,
        // and a too-short row — all three malformed rows must be dropped.
        let rows = parse_exec_response(&sample_body());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].open, 100.0);
        assert_eq!(rows[1].close, 100.0);
        assert!(parse_exec_response(&json!({})).is_empty());
    }

    #[test]
    fn columnar_shape_matches_trading_page_contract() {
        let rows = parse_exec_response(&sample_body());
        let out = columnar_response("BTCUSDT", "1m", &rows);
        // The client does: data.columns.indexOf('open') etc., new Date(row[0]).
        assert_eq!(out["columns"][0], "timestamp");
        assert_eq!(out["columns"][1], "open");
        let row0 = out["data"][0].as_array().unwrap();
        assert_eq!(row0[0], "2026-06-11T00:00:00.000000Z");
        assert_eq!(row0[1], 100.0);
        assert_eq!(row0[5], 12.5);
        assert_eq!(out["count"], 2);
    }

    #[test]
    fn candles_shape_matches_chart_contract() {
        let rows = parse_exec_response(&sample_body());
        let out = candles_response("BTCUSDT", "1m", &rows);
        // The client does: Math.floor(c.timestamp / 1000), c.volume ?? 0.
        let c0 = &out["candles"][0];
        assert_eq!(c0["timestamp"], 1_781_136_000_000i64);
        assert_eq!(c0["open"], 100.0);
        assert_eq!(c0["volume"], 12.5);
        assert_eq!(out["count"], 2);
    }
}
