//! `GET /sse/bars/{symbol}` — live closed-candle SSE stream.
//!
//! Bridges the in-process [`MarketDataBus`](janus_core::MarketDataBus) to
//! HTTP consumers as a `text/event-stream` of `event: bar` frames. The FKS
//! WebUI's server adapter pipes this straight through to the dashboard chart
//! (`JANUS_BARS_SSE_URL` → `/sse/bars/:sym`), which expects the `BarUpdate`
//! shape: `{time, open, high, low, close, volume}` with `time` in Unix
//! seconds (the chart keys candles by bar open time).
//!
//! Filtering:
//! - `{symbol}` matches the exchange-form concatenation of the kline's
//!   base+quote, separator-insensitively — `BTCUSDT`, `btc-usdt` and
//!   `BTC/USDT` all select the same stream.
//! - `?interval=` selects the kline interval; defaults to `1m`, the base
//!   interval the WebUI builds its history from.
//!
//! Only closed klines are emitted (the Data module publishes nothing else).
//! Lagged broadcast slots are skipped — the chart re-syncs from QuestDB
//! history, so a dropped live bar self-heals on the next page load.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Path, Query, State},
    response::sse::{Event, KeepAlive, Sse},
};
use futures_util::Stream;
use janus_core::{JanusState, KlineEvent, MarketDataEvent};
use rust_decimal::prelude::ToPrimitive;
use serde::Deserialize;
use tokio::sync::broadcast;

#[derive(Debug, Deserialize)]
pub struct BarsQuery {
    /// Kline interval to stream (e.g. "1m", "5m").
    #[serde(default = "default_interval")]
    pub interval: String,
}

fn default_interval() -> String {
    "1m".to_string()
}

/// Normalise a symbol for comparison: uppercase with `-`/`/`/`_` separators
/// stripped, so every client spelling matches the bus `Symbol{base, quote}`.
fn normalize_symbol(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '-' | '/' | '_'))
        .collect::<String>()
        .to_uppercase()
}

/// Whether a kline belongs on the stream for `want_symbol` (already
/// normalised) at `want_interval`.
fn kline_matches(k: &KlineEvent, want_symbol: &str, want_interval: &str) -> bool {
    k.is_closed
        && k.interval == want_interval
        && normalize_symbol(&format!("{}{}", k.symbol.base, k.symbol.quote)) == want_symbol
}

/// Build the `event: bar` JSON payload (the WebUI `BarUpdate` shape, plus
/// `symbol`/`interval` context fields the client ignores).
fn bar_json(k: &KlineEvent) -> serde_json::Value {
    serde_json::json!({
        // Bus timestamps are Unix microseconds; the chart wants seconds.
        "time": k.open_time / 1_000_000,
        "open": k.open.to_f64().unwrap_or(0.0),
        "high": k.high.to_f64().unwrap_or(0.0),
        "low": k.low.to_f64().unwrap_or(0.0),
        "close": k.close.to_f64().unwrap_or(0.0),
        "volume": k.volume.to_f64().unwrap_or(0.0),
        "symbol": format!("{}{}", k.symbol.base, k.symbol.quote).to_uppercase(),
        "interval": k.interval,
    })
}

/// SSE handler: subscribe to the market data bus and stream matching closed
/// klines as `event: bar` frames until the client disconnects or the bus
/// closes. Keep-alive comments every 15 s hold idle proxy connections open.
pub async fn sse_bars_handler(
    Path(symbol): Path<String>,
    Query(query): Query<BarsQuery>,
    State(state): State<Arc<JanusState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.market_data_bus.subscribe();
    let want_symbol = normalize_symbol(&symbol);
    let want_interval = query.interval;

    tracing::info!(
        "SSE bars client connected: symbol={} interval={}",
        want_symbol,
        want_interval
    );

    let stream = futures_util::stream::unfold(
        (rx, want_symbol, want_interval),
        |(mut rx, want_symbol, want_interval)| async move {
            loop {
                match rx.recv().await {
                    Ok(MarketDataEvent::Kline(k))
                        if kline_matches(&k, &want_symbol, &want_interval) =>
                    {
                        let payload = serde_json::to_string(&bar_json(&k))
                            .expect("serde_json::Value serialization is infallible");
                        let event = Event::default().event("bar").data(payload);
                        return Some((Ok(event), (rx, want_symbol, want_interval)));
                    }
                    // Other symbols/intervals/event kinds: keep waiting.
                    Ok(_) => continue,
                    // Skipped slots: the chart re-syncs from history.
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    // Bus gone (shutdown): end the stream cleanly.
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        },
    );

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Symbol};
    use rust_decimal::Decimal;

    fn sample_kline() -> KlineEvent {
        KlineEvent {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTC", "USDT"),
            interval: "1m".to_string(),
            open_time: 1_700_000_000_000_000, // µs
            close_time: 1_700_000_059_999_000,
            open: Decimal::from(50_000),
            high: Decimal::from(50_100),
            low: Decimal::from(49_900),
            close: Decimal::from(50_050),
            volume: Decimal::new(105, 1), // 10.5
            quote_volume: None,
            trades: Some(42),
            is_closed: true,
        }
    }

    #[test]
    fn test_normalize_symbol_strips_separators() {
        assert_eq!(normalize_symbol("BTCUSDT"), "BTCUSDT");
        assert_eq!(normalize_symbol("btc-usdt"), "BTCUSDT");
        assert_eq!(normalize_symbol("BTC/USDT"), "BTCUSDT");
        assert_eq!(normalize_symbol("btc_usdt"), "BTCUSDT");
    }

    #[test]
    fn test_kline_matches_symbol_and_interval() {
        let k = sample_kline();
        assert!(kline_matches(&k, "BTCUSDT", "1m"));
        assert!(!kline_matches(&k, "ETHUSDT", "1m"));
        assert!(!kline_matches(&k, "BTCUSDT", "5m"));
    }

    #[test]
    fn test_kline_matches_rejects_open_candles() {
        let mut k = sample_kline();
        k.is_closed = false;
        assert!(!kline_matches(&k, "BTCUSDT", "1m"));
    }

    #[test]
    fn test_bar_json_shape() {
        let v = bar_json(&sample_kline());
        // µs → s, the chart's expected unit.
        assert_eq!(v["time"], 1_700_000_000_i64);
        assert_eq!(v["open"], 50_000.0);
        assert_eq!(v["high"], 50_100.0);
        assert_eq!(v["low"], 49_900.0);
        assert_eq!(v["close"], 50_050.0);
        assert_eq!(v["volume"], 10.5);
        assert_eq!(v["symbol"], "BTCUSDT");
        assert_eq!(v["interval"], "1m");
    }
}
