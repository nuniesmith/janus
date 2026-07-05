//! FKS WebUI front-page contract — `/api/pipeline/scores/json`,
//! `/api/trades/open`, `/factory/status`.
//!
//! The Track D data contract (see `docs/PUBLIC_API.md` and `TODO.md`): the
//! SvelteKit dashboard's overview page pre-fetches these three endpoints. They
//! were served by the retired Python "Ruby" service; this serves them
//! **truthfully from janus state** so fks's nginx/WebUI can repoint:
//!
//! - **`/api/pipeline/scores/json`** → `{assets: [...]}` from the latest
//!   per-symbol signals (Redis), mapping confidence → `score` and the signal
//!   type → `cnn_signal`. (`ruby_signal` is dropped — Ruby is gone.)
//! - **`/api/trades/open`** → `{trades: [...]}` from the [`PositionTracker`]'s
//!   currently-tracked positions (janus is a manual co-pilot — this reflects
//!   positions reported via `/api/v1/positions/event`, not a broker order book).
//! - **`/factory/status`** → the data/runtime module health (per-module worker
//!   map + uptime). janus-native shape; the Python factory's provider/gap fields
//!   are omitted (the WebUI treats them as optional).
//!
//! The pure mappers ([`signals_to_scores`], [`factory_status`],
//! [`open_positions_to_trades`]) are unit-tested; the handlers are thin glue
//! over them plus shared state. Every handler degrades to an empty-but-valid
//! payload rather than erroring — a broken backend must not crash the SSR page.

use std::sync::Arc;

use axum::{Extension, Json, extract::State, response::IntoResponse};
use janus_core::{JanusState, OpenPosition, PositionTracker, Side};
use serde_json::{Value, json};

use crate::fetch_latest_signals_from_redis;

/// Map the latest signal JSON blobs to the WebUI's `MarketAsset` score rows,
/// keeping the most recent signal per symbol. Input is newest-first (the Redis
/// `signals:recent` ordering), so the first blob seen for a symbol wins.
pub(crate) fn signals_to_scores(signals: &[Value]) -> Vec<Value> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for sig in signals {
        let Some(symbol) = sig.get("symbol").and_then(Value::as_str) else {
            continue;
        };
        if !seen.insert(symbol.to_string()) {
            continue; // older signal for a symbol already emitted
        }
        // `confidence` is 0..1; surface it as a 0..100 score (pass through if a
        // producer already used 0..100).
        let confidence = sig.get("confidence").and_then(Value::as_f64).unwrap_or(0.0);
        let score = if confidence <= 1.0 {
            confidence * 100.0
        } else {
            confidence
        };
        let signal = sig
            .get("signal_type")
            .and_then(Value::as_str)
            .unwrap_or("hold")
            .to_uppercase();
        // `target_price` is the only price the Signal carries; absent ⇒ 0.0.
        let price = sig
            .get("target_price")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        out.push(json!({
            "symbol": symbol,
            "name": symbol,
            "score": (score * 100.0).round() / 100.0,
            "signal": signal,
            "cnn_signal": signal,
            "price": price,
            "confidence": confidence,
            "source": sig.get("source").and_then(Value::as_str).unwrap_or(""),
            "strategy": sig.get("strategy_id").and_then(Value::as_str),
            "timestamp": sig.get("timestamp").cloned().unwrap_or(Value::Null),
        }));
    }
    out
}

/// Map per-module health to the WebUI's `FactoryStatus`. `workers` is
/// `(module_name, healthy)`.
pub(crate) fn factory_status(workers: &[(String, bool)], uptime_seconds: u64) -> Value {
    let all_healthy = !workers.is_empty() && workers.iter().all(|(_, h)| *h);
    let worker_map: serde_json::Map<String, Value> = workers
        .iter()
        .map(|(n, h)| {
            (
                n.clone(),
                Value::from(if *h { "healthy" } else { "unhealthy" }),
            )
        })
        .collect();
    let status = if workers.is_empty() {
        "unknown"
    } else if all_healthy {
        "running"
    } else {
        "degraded"
    };
    json!({
        "status": status,
        "healthy": all_healthy,
        "workers": worker_map,
        "uptime_seconds": uptime_seconds,
    })
}

/// FNV-1a 32-bit hash — a stable numeric id for a `position_id` string, so the
/// WebUI (whose `OpenTrade.id` is numeric) keys positions consistently across
/// polls. Collision-tolerant: it's a display key, not an identity.
fn stable_id(s: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for b in s.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Map tracked open positions to the WebUI's `OpenTrade` rows.
pub(crate) fn open_positions_to_trades(positions: &[OpenPosition]) -> Value {
    let trades: Vec<Value> = positions
        .iter()
        .map(|p| {
            json!({
                "id": stable_id(&p.position_id),
                "position_id": p.position_id,
                "asset": p.symbol,
                "direction": match p.side {
                    Side::Buy => "long",
                    Side::Sell => "short",
                },
                "entry": p.entry_price,
                "price": p.current_price,
                "pnl": p.pnl_unrealized,
                "contracts": p.qty,
                "seconds_in_position": p.seconds_in_position,
            })
        })
        .collect();
    let count = trades.len();
    json!({ "trades": trades, "count": count })
}

/// `GET /api/pipeline/scores/json` — per-asset signal scores for the overview.
pub async fn scores_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    // Degrade to empty on a Redis hiccup — the SSR page must not 500.
    let signals = fetch_latest_signals_from_redis(&state, 100)
        .await
        .unwrap_or_default();
    Json(json!({ "assets": signals_to_scores(&signals) }))
}

/// `GET /api/trades/open` — currently-tracked open positions.
pub async fn open_trades_handler(
    Extension(tracker): Extension<Arc<PositionTracker>>,
) -> impl IntoResponse {
    Json(open_positions_to_trades(&tracker.open_positions().await))
}

/// `GET /factory/status` — data/runtime module health for the overview header.
pub async fn factory_status_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    let workers: Vec<(String, bool)> = state
        .get_module_health()
        .await
        .into_iter()
        .map(|h| (h.name, h.healthy))
        .collect();
    Json(factory_status(&workers, state.uptime_seconds()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_keep_latest_per_symbol_and_map_fields() {
        let signals = vec![
            json!({"symbol": "BTCUSDT", "signal_type": "buy", "confidence": 0.82, "target_price": 65000.0, "source": "forward", "strategy_id": "ema_flip"}),
            json!({"symbol": "ETHUSDT", "signal_type": "sell", "confidence": 0.40}),
            // Older BTC signal — must be ignored (first occurrence wins).
            json!({"symbol": "BTCUSDT", "signal_type": "hold", "confidence": 0.10}),
        ];
        let scores = signals_to_scores(&signals);
        assert_eq!(scores.len(), 2);
        let btc = &scores[0];
        assert_eq!(btc["symbol"], "BTCUSDT");
        assert_eq!(btc["score"], 82.0);
        assert_eq!(btc["cnn_signal"], "BUY");
        assert_eq!(btc["price"], 65000.0);
        assert_eq!(btc["strategy"], "ema_flip");
        let eth = &scores[1];
        assert_eq!(eth["score"], 40.0);
        assert_eq!(eth["cnn_signal"], "SELL");
        assert_eq!(eth["price"], 0.0); // no target_price ⇒ 0.0
    }

    #[test]
    fn scores_skip_symbolless_and_default_missing() {
        let signals = vec![
            json!({"signal_type": "buy", "confidence": 0.9}), // no symbol ⇒ skipped
            json!({"symbol": "SOLUSDT"}),                     // bare ⇒ defaults
        ];
        let scores = signals_to_scores(&signals);
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0]["symbol"], "SOLUSDT");
        assert_eq!(scores[0]["score"], 0.0);
        assert_eq!(scores[0]["cnn_signal"], "HOLD");
        assert!(signals_to_scores(&[]).is_empty());
    }

    #[test]
    fn factory_status_reflects_health() {
        let healthy = factory_status(&[("forward".into(), true), ("data".into(), true)], 3600);
        assert_eq!(healthy["status"], "running");
        assert_eq!(healthy["healthy"], true);
        assert_eq!(healthy["workers"]["forward"], "healthy");
        assert_eq!(healthy["uptime_seconds"], 3600);

        let degraded = factory_status(&[("forward".into(), true), ("data".into(), false)], 10);
        assert_eq!(degraded["status"], "degraded");
        assert_eq!(degraded["healthy"], false);
        assert_eq!(degraded["workers"]["data"], "unhealthy");

        let empty = factory_status(&[], 0);
        assert_eq!(empty["status"], "unknown");
        assert_eq!(empty["healthy"], false);
    }

    #[test]
    fn open_trades_map_to_webui_shape() {
        let positions = vec![OpenPosition {
            position_id: "pos-1".to_string(),
            symbol: "BTC-USD".to_string(),
            side: Side::Buy,
            qty: 0.5,
            entry_price: 60_000.0,
            current_price: 61_000.0,
            pnl_unrealized: 500.0,
            peak_pnl_ratio: 0.02,
            samples: 4,
            last_action: janus_core::GuidanceAction::Hold,
            seconds_in_position: 120,
        }];
        let out = open_positions_to_trades(&positions);
        assert_eq!(out["count"], 1);
        let t = &out["trades"][0];
        assert_eq!(t["asset"], "BTC-USD");
        assert_eq!(t["direction"], "long");
        assert_eq!(t["entry"], 60_000.0);
        assert_eq!(t["pnl"], 500.0);
        assert_eq!(t["contracts"], 0.5);
        assert_eq!(t["id"], stable_id("pos-1"));
        // Empty ⇒ empty trades array, never null.
        assert_eq!(open_positions_to_trades(&[])["count"], 0);
    }

    #[test]
    fn stable_id_is_deterministic_and_distinct() {
        assert_eq!(stable_id("pos-1"), stable_id("pos-1"));
        assert_ne!(stable_id("pos-1"), stable_id("pos-2"));
    }
}
