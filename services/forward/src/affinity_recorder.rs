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

/// Split a (possibly fused) consensus strategy label into its contributing
/// strategies.
///
/// The live consensus loop names a signal with the `+`-joined list of the
/// strategies that agreed (e.g. `"ema_flip+mean_reversion"`; see
/// `with_strategy(strategies_used.join("+"))` in `lib.rs`). The realized close
/// echoes that fused label back as the trade's single `strategy` field — the
/// individual contributor list (`strategies_used` metadata) lives on the
/// in-flight `Signal`, not on the `PositionClose`, so the fused label is the
/// only carrier of the contributor set at close time.
///
/// This reverses the join: trim whitespace and drop empty segments. A
/// non-fused name (no `+`) yields a single-element list, so single-strategy
/// closes are unchanged. Returns an empty vec only if the input is blank.
fn split_contributors(strategy: &str) -> Vec<&str> {
    strategy
        .split('+')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
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
        // Per-strategy attribution: the consensus loop fuses several agreeing
        // strategies into one signal (`ema_flip+mean_reversion+…`) and the
        // close carries only that fused label. To learn per-individual-strategy
        // affinity rather than on the fused label, attribute the realized
        // outcome to *each* contributing strategy for this asset (equal
        // credit/blame — every contributor agreed on the entry, so each records
        // the full trade outcome). The fused name is untouched on the signal
        // itself; this changes only what affinity records.
        //
        // Contributor source: split the fused label on `+` (the
        // `strategies_used` list isn't propagated to the close path). If the
        // label has no usable segments (blank), fall back to a single record
        // under the original name so behaviour never regresses.
        let contributors = split_contributors(strategy);

        // The gate sits behind the pipeline's async RwLock; recording needs a
        // write guard. The critical section is a handful of in-memory map
        // updates (one per contributor).
        let mut gate = self.pipeline.strategy_gate_mut().await;
        let tracker = gate.tracker_mut();
        if contributors.is_empty() {
            tracker.record_trade_result_with_rr(strategy, asset, pnl, is_winner, rr_ratio);
        } else {
            tracker.record_trade_result_for_strategies(
                &contributors,
                asset,
                pnl,
                is_winner,
                rr_ratio,
            );
        }
        drop(gate);

        tracing::debug!(
            strategy,
            contributors = ?contributors,
            asset,
            pnl,
            is_winner,
            "recorded closed-trade outcome into affinity tracker (per contributing strategy)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_contributors_splits_fused_label() {
        assert_eq!(
            split_contributors("ema_flip+mean_reversion+squeeze"),
            vec!["ema_flip", "mean_reversion", "squeeze"]
        );
    }

    #[test]
    fn split_contributors_single_strategy_is_one_element() {
        assert_eq!(split_contributors("ema_flip"), vec!["ema_flip"]);
    }

    #[test]
    fn split_contributors_trims_and_drops_empties() {
        // Tolerate the ", "-style join used elsewhere as well as stray "+".
        assert_eq!(
            split_contributors("ema_flip + mean_reversion"),
            vec!["ema_flip", "mean_reversion"]
        );
        assert_eq!(
            split_contributors("ema_flip++"),
            vec!["ema_flip"],
            "trailing separators must not create empty contributors"
        );
    }

    #[test]
    fn split_contributors_blank_is_empty() {
        assert!(split_contributors("").is_empty());
        assert!(split_contributors("  ").is_empty());
        assert!(split_contributors("+").is_empty());
    }
}
