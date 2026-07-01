//! # Signal → Redis persistence
//!
//! Bridges the in-process [`SignalBus`](janus_core::SignalBus) broadcast to
//! the Redis keys the janus-api read path serves, so the signal READ APIs and
//! the WebUI dashboard are actually populated.
//!
//! `SignalBus::publish` is a pure in-process `tokio::broadcast` — it never
//! touches Redis. Meanwhile the janus-api read handlers
//! (`/api/signals/latest`, `/api/signals/by-symbol`,
//! `/api/pipeline/scores/json`, dashboard `recent_signals`) read the keys
//! written here. Nothing else writes them, so without this subscriber every
//! one of those endpoints returns an empty set even while signals flow on the
//! bus. This task closes that gap.
//!
//! ## Keys written (must match `janus-api` `fetch_latest_signals_from_redis`)
//!
//! - `janus:signal:{id}`               — `SET` (TTL'd): the full serialized [`Signal`]
//! - `janus:signals:recent`            — `ZADD` id, score = event-time millis (global, newest-first)
//! - `janus:signals:symbol:{SYMBOL}`   — `ZADD` id, score = event-time millis (per-symbol; `UPPER`, `/`→`_`)
//!
//! The serialized `Signal` field names already satisfy the read contract
//! (`symbol`, `signal_type` lowercased, `confidence` 0..1, `target_price`,
//! `source`, `strategy_id`, `timestamp`) — see janus-api `signals_to_scores`.
//!
//! Best-effort and additive: Redis errors are logged, never propagated. A
//! signal that fails to persist is dropped, never retried, and never affects
//! the trading path.

use std::sync::Arc;
use std::time::Duration;

use janus_core::{JanusState, Signal};
use redis::AsyncCommands;
use tokio::sync::broadcast::error::RecvError;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

/// Keep at most this many ids in the global `janus:signals:recent` index.
/// Read handlers request up to 100; this leaves generous headroom while
/// bounding the ZSET so it can never grow without limit.
const RECENT_CAP: isize = 1000;

/// Keep at most this many ids per `janus:signals:symbol:{SYM}` index.
const PER_SYMBOL_CAP: isize = 200;

/// TTL (seconds) on the per-id body keys and the per-symbol index so evicted
/// / idle entries self-expire. 3 days comfortably outlives the recency window.
const SIGNAL_TTL_SECS: u64 = 3 * 24 * 60 * 60;

/// Spawn the background task that persists every signal published on the
/// in-process `SignalBus` into Redis. Returns the task handle (fire-and-forget;
/// the task self-exits on shutdown).
pub fn spawn_signal_persistence(state: Arc<JanusState>) -> JoinHandle<()> {
    // Subscribe up-front so signals published after this point are captured.
    let mut rx = state.signal_bus.subscribe();
    tokio::spawn(async move {
        info!("signal→Redis persistence subscriber started");
        loop {
            tokio::select! {
                recv = rx.recv() => match recv {
                    Ok(signal) => match persist_signal(&state, &signal).await {
                        Ok(()) => {
                            state.increment_signals_persisted();
                            debug!(
                                "persisted signal {} {} (id {}, conf {:.2})",
                                signal.symbol, signal.signal_type, signal.id, signal.confidence
                            );
                        }
                        Err(e) => {
                            warn!("failed to persist signal {} to Redis: {e:#}", signal.id);
                        }
                    },
                    Err(RecvError::Lagged(n)) => {
                        warn!("signal persistence lagged; {n} signal(s) dropped before persistence");
                    }
                    Err(RecvError::Closed) => {
                        info!("signal bus closed; signal persistence subscriber exiting");
                        break;
                    }
                },
                _ = tokio::time::sleep(Duration::from_millis(500)) => {
                    if state.is_shutdown_requested() {
                        info!("shutdown requested; signal persistence subscriber exiting");
                        break;
                    }
                }
            }
        }
    })
}

/// Write a single signal into the Redis keys the janus-api read path consumes.
async fn persist_signal(state: &Arc<JanusState>, signal: &Signal) -> anyhow::Result<()> {
    let client = state
        .redis_client()
        .await
        .map_err(|e| anyhow::anyhow!("redis client: {e}"))?;
    let mut conn = client.get_multiplexed_async_connection().await?;

    let body = serde_json::to_string(signal)?;
    // Score by event time so ZREVRANGE yields newest-first (matches the reader).
    let score = signal.timestamp.timestamp_millis();
    let id_key = format!("janus:signal:{}", signal.id);
    // Must match the by-symbol read key (janus-api lib.rs: UPPER, '/' → '_').
    let sym_key = format!(
        "janus:signals:symbol:{}",
        signal.symbol.to_uppercase().replace('/', "_")
    );

    // Full signal body, TTL'd so per-id keys self-expire once out of the window.
    let _: () = conn.set_ex(&id_key, body, SIGNAL_TTL_SECS).await?;

    // Global + per-symbol recency indexes (id scored by event time).
    let _: () = conn
        .zadd("janus:signals:recent", signal.id.as_str(), score)
        .await?;
    let _: () = conn.zadd(&sym_key, signal.id.as_str(), score).await?;

    // Cap the indexes to the newest N; give the per-symbol index a TTL so idle
    // symbols drop out. Rank 0 = lowest score = oldest, so remove [0, -(cap+1)].
    let _: () = conn
        .zremrangebyrank("janus:signals:recent", 0, -(RECENT_CAP + 1))
        .await?;
    let _: () = conn
        .zremrangebyrank(&sym_key, 0, -(PER_SYMBOL_CAP + 1))
        .await?;
    let _: () = conn.expire(&sym_key, SIGNAL_TTL_SECS as i64).await?;

    Ok(())
}
