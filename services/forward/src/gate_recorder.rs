//! Bridges the API's position-close path into the execution gate's
//! consecutive-loss circuit breaker (Track C — close-driven gate feedback).
//!
//! The execution gate ([`ForwardGate`]) lives in the forward signal loop; the
//! position-close HTTP handler lives in `janus-api`. They share only
//! [`JanusState`](janus_core::JanusState). This adapter implements the
//! [`GateOutcomeRecorder`](janus_core::GateOutcomeRecorder) trait janus-core
//! defines and is installed onto `JanusState` at forward startup, so a realized
//! outcome posted to `/api/v1/positions/close` feeds the breaker in real time:
//! consecutive losses on an asset trip it, a win resets it.
//!
//! The gate is shared as `Arc<RwLock<ForwardGate>>` — the signal loop takes a
//! write guard per evaluation, and this recorder takes one per close. Both are
//! brief, synchronous critical sections (no `.await` held across the guard).

use std::sync::Arc;

use janus_core::GateOutcomeRecorder;
use tokio::sync::RwLock;

use crate::gate_integration::ForwardGate;

/// [`GateOutcomeRecorder`] backed by the shared [`ForwardGate`]. Feeds each
/// realized close into the gate's consecutive-loss breaker.
pub struct GateBreakerRecorder {
    gate: Arc<RwLock<ForwardGate>>,
}

impl GateBreakerRecorder {
    /// Wrap the shared gate handle as a breaker outcome recorder.
    pub fn new(gate: Arc<RwLock<ForwardGate>>) -> Self {
        Self { gate }
    }
}

#[async_trait::async_trait]
impl GateOutcomeRecorder for GateBreakerRecorder {
    async fn record_outcome(&self, asset: &str, is_win: bool, now_secs: u64) {
        // Brief write guard: `record_trade_outcome` is a couple of in-memory map
        // updates, no `.await` inside.
        self.gate
            .write()
            .await
            .record_trade_outcome(asset, is_win, now_secs);
        tracing::debug!(
            asset,
            is_win,
            "recorded closed-trade outcome into execution-gate breaker"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_integration::{GateVerdict, Side};

    #[tokio::test]
    async fn close_feed_trips_then_recovers_the_breaker() {
        let gate = Arc::new(RwLock::new(ForwardGate::new(false)));
        let rec = GateBreakerRecorder::new(Arc::clone(&gate));
        let now = 1_700_000_000;

        // Three losses on BTC (the default limit) → the breaker opens, and the
        // gate now blocks a BTC entry on the circuit-breaker gate.
        for _ in 0..3 {
            rec.record_outcome("BTC", false, now).await;
        }
        let ctx = ForwardGate::build_context(Side::Buy, "ok", 0.6, None, Vec::new());
        let blocked = gate
            .write()
            .await
            .evaluate_entry("BTC", Side::Buy, false, &ctx, now);
        assert_eq!(blocked.verdict, GateVerdict::BlockCircuitBreaker);

        // After the cooldown (default 900s) the breaker closes and entries clear.
        let cleared = gate
            .write()
            .await
            .evaluate_entry("BTC", Side::Buy, false, &ctx, now + 900);
        assert_eq!(cleared.verdict, GateVerdict::Pass);
    }
}
