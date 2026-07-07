//! Indicator catalog + compute — chart-page indicator auto-discovery (server side).
//!
//! Two routes power the fks-web chart's "add a Rust-computed indicator" UI:
//!
//! - `GET /api/indicators/catalog` — the **metadata** the chart's dropdown is
//!   built from. Serializes [`janus_indicators::descriptor::catalog()`] (the
//!   descriptor layer added in `indicators-ta` 0.3): one entry per indicator
//!   with its id, display name, chart pane category, and tunable param specs.
//!   Cheap and cacheable — pure metadata, no I/O.
//!
//! - `GET /api/indicators/compute?symbol=&interval=&indicator=<id>&<param>=<val>…`
//!   — **computes** an indicator over live candles and returns its line(s) in
//!   the `{ time /*seconds*/, value }` point shape the chart already renders
//!   (see fks-web `src/lib/server/indicators.ts` — its `/api/chart/:sym/indicators`
//!   returns `{ indicators: { <key>: [{time, value}] } }`). Candles come from
//!   the **same QuestDB path** the chart's history load uses ([`crate::bars`]),
//!   so a computed overlay lines up bar-for-bar with the price series.
//!
//! # Compute response shape (what the chart consumes)
//!
//! ```json
//! {
//!   "symbol": "BTCUSDT",
//!   "interval": "1m",
//!   "indicator": "rsi",
//!   "series": {
//!     "rsi_14": [ { "time": 1781136000, "value": 55.34 }, … ]
//!   },
//!   "count": 42
//! }
//! ```
//!
//! - `series` is an **object** mapping each output line's key → an array of
//!   `{ time, value }` points. Single-line indicators (RSI, EMA, ATR…) have one
//!   key; multi-line indicators emit several — MACD →
//!   `macd_line` / `macd_signal` / `macd_histogram`, Bollinger →
//!   `bb_middle` / `bb_upper` / `bb_lower` / `bb_bandwidth` / `bb_pct_b`,
//!   Stochastic → `stoch_k` / `stoch_d`, Keltner →
//!   `kc_upper` / `kc_middle` / `kc_lower`. Keys are the `indicators-ta` output
//!   column names, **lowercased** (`RSI_14` → `rsi_14`, `MACD_line` →
//!   `macd_line`), so `Object.assign(dest, resp.series)` drops straight into the
//!   chart's `Record<string, Point[]>`.
//! - `time` is epoch **seconds** (candle open-time ms ÷ 1000), matching
//!   lightweight-charts and the fks-web TS output. `value` is an `f64` rounded
//!   to 6 decimals. Warm-up rows (NaN) are omitted, so lines start at the first
//!   valid bar exactly like the TS `toPoints` helper.

use std::collections::HashMap;

use axum::{
    Json,
    extract::Query,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use janus_indicators::registry::registry;
use janus_indicators::types::Candle;
use serde_json::{Map, Value, json};

use crate::bars::{HistoryQuery, fetch_bars};

/// Query keys the compute endpoint consumes itself. Every *other* query
/// parameter is forwarded verbatim to the indicator factory as a tunable
/// param (e.g. `period`, `fast_period`, `std_dev`, `multiplier`).
const RESERVED_KEYS: &[&str] = &["symbol", "interval", "indicator", "days_back", "limit"];

fn default_days_back() -> u32 {
    5
}
fn default_limit() -> usize {
    1_000
}

/// Round to 6 decimals — mirrors the fks-web `round` helper so server-computed
/// and TS-computed overlays agree to the last rendered digit.
fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

/// Map one index-aligned `Vec<f64>` column to `[{time, value}]`, dropping NaN
/// warm-up entries and converting candle open-time from ms → epoch **seconds**.
fn column_points(candles: &[Candle], values: &[f64]) -> Value {
    let mut points = Vec::new();
    for (candle, &v) in candles.iter().zip(values) {
        if v.is_finite() {
            points.push(json!({ "time": candle.time / 1_000, "value": round6(v) }));
        }
    }
    Value::Array(points)
}

/// Compute `indicator` (by registry id) over `candles` with `params`, returning
/// its `key -> [{time, value}]` series object. Pure (no I/O) so it is
/// unit-testable without QuestDB.
///
/// # Errors
/// Returns the indicator error string on an unknown id, a bad param, or
/// insufficient data.
pub(crate) fn compute_series(
    indicator: &str,
    params: &HashMap<String, String>,
    candles: &[Candle],
) -> Result<Map<String, Value>, String> {
    let ind = registry()
        .create(indicator, params)
        .map_err(|e| e.to_string())?;
    let output = ind.calculate(candles).map_err(|e| e.to_string())?;

    // Sort column names so the serialized object key order is deterministic
    // (the underlying store is a HashMap).
    let mut names: Vec<&str> = output.columns().collect();
    names.sort_unstable();

    let mut series = Map::new();
    for name in names {
        if let Some(values) = output.get(name) {
            series.insert(name.to_ascii_lowercase(), column_points(candles, values));
        }
    }
    Ok(series)
}

/// `GET /api/indicators/catalog` — the indicator metadata the chart's dropdown
/// is built from. Pure/cacheable: `{ "count": N, "indicators": [ …descriptors… ] }`.
pub async fn catalog_handler() -> impl IntoResponse {
    let catalog = janus_indicators::descriptor::catalog();
    Json(json!({ "count": catalog.len(), "indicators": catalog }))
}

/// `GET /api/indicators/compute` — compute an indicator over live QuestDB
/// candles and return its line(s) in the chart's point shape.
pub async fn compute_handler(Query(params): Query<HashMap<String, String>>) -> Response {
    let bad_request =
        |msg: String| (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))).into_response();

    let Some(symbol) = params.get("symbol").cloned() else {
        return bad_request("missing required `symbol` query parameter".to_string());
    };
    let Some(indicator) = params.get("indicator").cloned() else {
        return bad_request("missing required `indicator` query parameter".to_string());
    };
    let interval = params
        .get("interval")
        .cloned()
        .unwrap_or_else(|| "1m".to_string());
    let days_back = params
        .get("days_back")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(default_days_back);
    let limit = params
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(default_limit);

    // Reuse the chart's exact QuestDB candle path (validation, normalisation,
    // clamping all live in `bars`).
    let history = HistoryQuery {
        interval: interval.clone(),
        days_back,
        limit,
    };
    let rows = match fetch_bars(&symbol, &history).await {
        Ok(rows) => rows,
        Err(e) => return e.into_response(),
    };
    let candles: Vec<Candle> = rows
        .iter()
        .map(|r| Candle {
            time: r.ts_ms,
            open: r.open,
            high: r.high,
            low: r.low,
            close: r.close,
            volume: r.volume,
        })
        .collect();

    // Everything that isn't a reserved key is an indicator param.
    let ind_params: HashMap<String, String> = params
        .iter()
        .filter(|(k, _)| !RESERVED_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    match compute_series(&indicator, &ind_params, &candles) {
        Ok(series) => {
            let count = series
                .values()
                .next()
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            (
                StatusCode::OK,
                Json(json!({
                    "symbol": symbol,
                    "interval": interval,
                    "indicator": indicator,
                    "series": series,
                    "count": count,
                })),
            )
                .into_response()
        }
        Err(msg) => bad_request(format!("indicator compute failed: {msg}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A monotonic ramp of candles — enough bars to clear any warm-up.
    fn ramp_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64) * 0.5 + ((i % 5) as f64);
                Candle {
                    time: 1_781_136_000_000 + (i as i64) * 60_000,
                    open: p,
                    high: p + 1.0,
                    low: p - 1.0,
                    close: p,
                    volume: 10.0 + i as f64,
                }
            })
            .collect()
    }

    #[test]
    fn catalog_has_21_descriptors() {
        let catalog = janus_indicators::descriptor::catalog();
        assert_eq!(catalog.len(), 21, "catalog should expose 21 indicators");
        assert!(catalog.iter().any(|d| d.id == "rsi"));
    }

    #[test]
    fn compute_rsi_has_expected_shape() {
        let candles = ramp_candles(60);
        let params: HashMap<String, String> = [("period".to_string(), "14".to_string())].into();
        let series = compute_series("rsi", &params, &candles).unwrap();

        // Single line, keyed by the lowercased output column name.
        let arr = series
            .get("rsi_14")
            .expect("rsi_14 line present")
            .as_array()
            .expect("line is an array");
        assert!(!arr.is_empty());
        // Warm-up NaN entries dropped → fewer points than candles.
        assert!(arr.len() < candles.len());

        let p0 = &arr[0];
        // `time` is epoch SECONDS (candle ms ÷ 1000), aligned to a candle and
        // past the 14-bar warm-up.
        let t0 = p0["time"].as_i64().unwrap();
        assert!(t0 >= 1_781_136_000 + 14 * 60, "first point clears warm-up");
        assert_eq!(t0 % 60, 0, "seconds unit, minute-aligned");
        assert!(p0["value"].as_f64().unwrap().is_finite());
        // Point shape is exactly {time, value}.
        assert_eq!(p0.as_object().unwrap().len(), 2);
    }

    #[test]
    fn compute_macd_is_multiline() {
        let series = compute_series("macd", &HashMap::new(), &ramp_candles(60)).unwrap();
        assert!(series.contains_key("macd_line"));
        assert!(series.contains_key("macd_signal"));
        assert!(series.contains_key("macd_histogram"));
    }

    #[test]
    fn compute_unknown_indicator_errors() {
        assert!(compute_series("no_such_indicator", &HashMap::new(), &ramp_candles(60)).is_err());
    }
}
