//! Bridges the API's position-close path into the forward service's
//! strategy-affinity tracker (JFLOW-C outcome feedback).
//!
//! The affinity tracker lives inside the forward [`TradingPipeline`]; the
//! position-close HTTP handler lives in `janus-api`. They share only
//! [`JanusState`](janus_core::JanusState). This adapter implements the
//! [`AffinityRecorder`](janus_core::AffinityRecorder) trait janus-core
//! defines and is installed onto `JanusState` at forward startup, so a
//! realized outcome posted to `/api/v1/positions/close` updates affinity
//! weights in real time — complementing the startup `bootstrap_affinity_from_postgres`
//! replay rather than waiting for the next restart.

use std::sync::Arc;

use janus_core::AffinityRecorder;

use crate::brain_wiring::TradingPipeline;

/// [`AffinityRecorder`] backed by the forward [`TradingPipeline`]'s strategy
/// gate. Holds an `Arc` to the pipeline and delegates each recorded trade to
/// the gate's affinity tracker.
pub struct PipelineAffinityRecorder {
    pipeline: Arc<TradingPipeline>,
}

impl PipelineAffinityRecorder {
    /// Wrap a pipeline handle as an affinity recorder.
    pub fn new(pipeline: Arc<TradingPipeline>) -> Self {
        Self { pipeline }
    }
}

#[async_trait::async_trait]
impl AffinityRecorder for PipelineAffinityRecorder {
    async fn record_trade(
        &self,
        strategy: &str,
        asset: &str,
        pnl: f64,
        is_winner: bool,
        rr_ratio: Option<f64>,
    ) {
        // The gate sits behind the pipeline's async RwLock; recording needs a
        // write guard. The critical section is a single in-memory map update.
        self.pipeline
            .strategy_gate_mut()
            .await
            .tracker_mut()
            .record_trade_result_with_rr(strategy, asset, pnl, is_winner, rr_ratio);
        tracing::debug!(
            strategy,
            asset,
            pnl,
            is_winner,
            "recorded closed-trade outcome into affinity tracker"
        );
    }
}
