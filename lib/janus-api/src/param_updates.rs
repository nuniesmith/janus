//! Live optimizer-param updates for janus-api guidance (JFLOW-C).
//!
//! Subscribes to the Redis pub/sub channel `fks:{instance}:param_updates`
//! and routes each notification into the per-asset cache used by
//! [`compute_guidance`](janus_core::compute_guidance). When Redis is
//! unreachable the loop sleeps with exponential backoff and retries
//! until the parent module signals shutdown — the in-memory cache
//! continues to serve whatever it last loaded.
//!
//! `services/forward` runs an analogous loop for its own appliers; the
//! plan to dedupe by sharing a single `ParamManager` via `JanusState`
//! lives in `TODO.md`.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use janus_core::{JanusState, ParamManager};
use tokio::task::JoinHandle;

const INITIAL_BACKOFF: Duration = Duration::from_millis(500);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Spawn the live-update subscription task. The returned handle can be
/// `.abort()`-ed at shutdown alongside the HTTP / metrics servers.
pub fn spawn(state: Arc<JanusState>, param_manager: Arc<ParamManager>) -> JoinHandle<()> {
    let redis_url = state.config.redis.url.clone();
    let channel = param_manager.updates_channel();
    tokio::spawn(async move {
        run_loop(state, param_manager, redis_url, channel).await;
    })
}

async fn run_loop(
    state: Arc<JanusState>,
    param_manager: Arc<ParamManager>,
    redis_url: String,
    channel: String,
) {
    let mut backoff = INITIAL_BACKOFF;
    while !state.is_shutdown_requested() {
        match connect_and_listen(&redis_url, &channel, &param_manager, &state).await {
            Ok(()) => {
                // Stream ended cleanly (shutdown or peer close). Reset
                // backoff before deciding whether to reconnect.
                backoff = INITIAL_BACKOFF;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    backoff_ms = backoff.as_millis() as u64,
                    "param_updates: subscription failed, retrying after backoff"
                );
            }
        }

        if state.is_shutdown_requested() {
            break;
        }

        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
    tracing::debug!("param_updates: shutdown signalled, subscription task exiting");
}

async fn connect_and_listen(
    redis_url: &str,
    channel: &str,
    param_manager: &Arc<ParamManager>,
    state: &Arc<JanusState>,
) -> Result<(), redis::RedisError> {
    let client = redis::Client::open(redis_url)?;
    let mut pubsub = client.get_async_pubsub().await?;
    pubsub.subscribe(channel).await?;
    tracing::info!(channel = channel, "param_updates: subscribed to channel");

    let mut stream = pubsub.on_message();
    while let Some(msg) = stream.next().await {
        if state.is_shutdown_requested() {
            return Ok(());
        }
        let payload: String = msg.get_payload()?;
        if let Err(e) = param_manager.process_notification(&payload).await {
            tracing::warn!(
                error = %e,
                "param_updates: failed to process notification"
            );
        }
    }
    Ok(())
}
