//! The execution gate chain and its in-memory metrics.

use std::collections::{BTreeMap, HashMap};

use crate::adaptive_threshold::AdaptiveThreshold;
use crate::circuit_breaker::ConsecutiveLossBreaker;
use crate::context::{GateContext, Side};
use crate::correlation::CorrelationGuard;
use crate::verdict::GateVerdict;
use serde::{Deserialize, Serialize};

/// Fee/TP ratio above which an entry is rejected as fee-uneconomic (30%).
const MAX_FEE_TP_RATIO: f64 = 0.30;

/// In-memory block/eval counters, mirroring the Python gate's Redis counters
/// (`gate_blocks:{asset}:{reason}`, `gate_evals:{asset}`).
///
/// A Redis-backed adapter for cross-process / dashboard durability is a
/// follow-up (see the janus TODO); the per-reason label
/// ([`GateVerdict::as_counter_label`]) is already wire-compatible with the
/// Python key scheme.
#[derive(Debug, Default, Clone)]
pub struct GateMetrics {
    evals: HashMap<String, u64>,
    blocks: HashMap<(String, String), u64>,
}

impl GateMetrics {
    /// Increment the per-asset evaluation counter.
    pub fn record_eval(&mut self, asset: &str) {
        *self.evals.entry(asset.to_string()).or_insert(0) += 1;
    }

    /// Increment the per-asset, per-reason block counter.
    pub fn record_block(&mut self, asset: &str, verdict: GateVerdict) {
        *self
            .blocks
            .entry((asset.to_string(), verdict.as_counter_label().to_string()))
            .or_insert(0) += 1;
    }

    /// Total evaluations for an asset.
    pub fn evals(&self, asset: &str) -> u64 {
        self.evals.get(asset).copied().unwrap_or(0)
    }

    /// Block count for an asset + verdict.
    pub fn block_count(&self, asset: &str, verdict: GateVerdict) -> u64 {
        self.blocks
            .get(&(asset.to_string(), verdict.as_counter_label().to_string()))
            .copied()
            .unwrap_or(0)
    }

    /// Per-reason block counts plus a `total_evals` entry — mirrors Ruby's
    /// `ExecutionGate.get_block_stats`. Only non-zero reasons are included.
    pub fn block_stats(&self, asset: &str) -> BTreeMap<String, u64> {
        let mut stats = BTreeMap::new();
        for ((a, reason), count) in &self.blocks {
            if a == asset && *count > 0 {
                stats.insert(reason.clone(), *count);
            }
        }
        stats.insert("total_evals".to_string(), self.evals(asset));
        stats
    }

    /// Snapshot all counters for export (e.g. mirroring to Redis for Grafana).
    /// Entries are sorted for deterministic output.
    pub fn snapshot(&self) -> GateMetricsSnapshot {
        let mut evals: Vec<(String, u64)> =
            self.evals.iter().map(|(a, &c)| (a.clone(), c)).collect();
        evals.sort();
        let mut blocks: Vec<(String, String, u64)> = self
            .blocks
            .iter()
            .map(|((a, r), &c)| (a.clone(), r.clone(), c))
            .collect();
        blocks.sort();
        GateMetricsSnapshot { evals, blocks }
    }
}

/// A point-in-time export of all gate counters, for mirroring to Redis /
/// Grafana. Produced by [`GateMetrics::snapshot`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateMetricsSnapshot {
    /// `(asset, total_evals)`, sorted by asset.
    pub evals: Vec<(String, u64)>,
    /// `(asset, reason_label, count)`, sorted.
    pub blocks: Vec<(String, String, u64)>,
}

/// Chain-of-responsibility gate for trade entries.
///
/// Evaluates a series of gates in priority order and returns the first BLOCK
/// verdict encountered, or [`GateVerdict::Pass`] if all clear. Faithful port of
/// Ruby's `ExecutionGate` (`ruby/src/services/execution_gate.py`).
///
/// Evaluation is pure and synchronous. The breaker / correlation guard /
/// adaptive threshold are owned here and updated out-of-band by the trading loop
/// ([`record_trade_outcome`](Self::record_trade_outcome),
/// [`update_correlation`](Self::update_correlation)) — exactly as in the Python
/// design, where these were separate stateful collaborators.
#[derive(Debug, Default)]
pub struct ExecutionGate {
    /// Consecutive-loss circuit breaker (always present; cheap and in-memory).
    pub breaker: ConsecutiveLossBreaker,
    /// Optional correlation guard. When `None`, the correlation gate always passes.
    pub correlation_guard: Option<CorrelationGuard>,
    /// Optional adaptive threshold. When `None`, the CNN-confidence gate uses the
    /// context's fixed `min_confidence`.
    pub adaptive_threshold: Option<AdaptiveThreshold>,
    /// Block / eval counters.
    pub metrics: GateMetrics,
}

impl ExecutionGate {
    /// Construct a gate from its collaborators.
    pub fn new(
        breaker: ConsecutiveLossBreaker,
        correlation_guard: Option<CorrelationGuard>,
        adaptive_threshold: Option<AdaptiveThreshold>,
    ) -> Self {
        Self {
            breaker,
            correlation_guard,
            adaptive_threshold,
            metrics: GateMetrics::default(),
        }
    }

    /// Feed a closed-trade outcome to the stateful collaborators (circuit breaker
    /// and, if present, adaptive threshold). Call this when a position closes.
    pub fn record_trade_outcome(&mut self, asset: &str, is_win: bool, now_secs: u64) {
        self.breaker.record_outcome(asset, is_win, now_secs);
        if let Some(at) = self.adaptive_threshold.as_mut() {
            at.record_outcome(asset, is_win);
        }
    }

    /// Feed a new log-return to the correlation guard (no-op when absent).
    pub fn update_correlation(&mut self, asset: &str, log_return: f64) {
        if let Some(g) = self.correlation_guard.as_mut() {
            g.update(asset, log_return);
        }
    }

    /// Run all gates in priority order. Returns the first BLOCK `(verdict, reason)`
    /// encountered, or `(Pass, "")` if all gates clear.
    ///
    /// `is_add` marks a stack-add (not a first entry): the entry-only `quality`
    /// and `ao_divergence` gates are skipped for adds, matching the Python chain.
    /// `now_secs` is the current unix time (for the circuit-breaker cooldown).
    pub fn evaluate(
        &mut self,
        asset: &str,
        signal: Side,
        is_add: bool,
        ctx: &GateContext,
        now_secs: u64,
    ) -> (GateVerdict, String) {
        self.metrics.record_eval(asset);

        // 1. circuit breaker
        let (v, r) = self.check_circuit_breaker(asset, now_secs);
        if v.is_block() {
            return self.blocked(asset, v, r);
        }
        // 2. risk
        let (v, r) = check_risk(ctx);
        if v.is_block() {
            return self.blocked(asset, v, r);
        }
        // 3. volatility filter
        let (v, r) = check_vol_filter(ctx);
        if v.is_block() {
            return self.blocked(asset, v, r);
        }
        // 4 & 5. entry-only gates (skipped for stack-adds)
        if !is_add {
            let (v, r) = check_quality(ctx);
            if v.is_block() {
                return self.blocked(asset, v, r);
            }
            let (v, r) = check_ao_divergence(signal, ctx);
            if v.is_block() {
                return self.blocked(asset, v, r);
            }
        }
        // 6. fee viability
        let (v, r) = check_fee_viability(ctx);
        if v.is_block() {
            return self.blocked(asset, v, r);
        }
        // 7. CNN agreement
        let (v, r) = check_cnn_agreement(signal, ctx);
        if v.is_block() {
            return self.blocked(asset, v, r);
        }
        // 8. CNN confidence
        let (v, r) = self.check_cnn_confidence(asset, ctx);
        if v.is_block() {
            return self.blocked(asset, v, r);
        }
        // 9. correlation
        let (v, r) = self.check_correlation(asset, ctx);
        if v.is_block() {
            return self.blocked(asset, v, r);
        }

        (GateVerdict::Pass, String::new())
    }

    fn blocked(
        &mut self,
        asset: &str,
        verdict: GateVerdict,
        reason: String,
    ) -> (GateVerdict, String) {
        self.metrics.record_block(asset, verdict);
        (verdict, reason)
    }

    // ── Stateful gate checks (need owned collaborators) ──────────────

    fn check_circuit_breaker(&self, asset: &str, now_secs: u64) -> (GateVerdict, String) {
        if self.breaker.is_open(asset, now_secs) {
            let remaining = self.breaker.cooldown_remaining(asset, now_secs);
            let count = self.breaker.loss_count(asset);
            return (
                GateVerdict::BlockCircuitBreaker,
                format!("{count} consecutive losses, cooldown {remaining}s remaining"),
            );
        }
        (GateVerdict::Pass, String::new())
    }

    fn check_cnn_confidence(&self, asset: &str, ctx: &GateContext) -> (GateVerdict, String) {
        if !ctx.cnn_enabled {
            return (GateVerdict::Pass, String::new());
        }
        let Some(vote) = ctx.cnn_result else {
            return (GateVerdict::Pass, String::new());
        };
        // Prefer the per-asset adaptive threshold when configured.
        let threshold = match &self.adaptive_threshold {
            Some(at) => at.get_threshold(asset),
            None => ctx.min_confidence,
        };
        if vote.confidence < threshold {
            return (
                GateVerdict::BlockCnnConfidence,
                format!(
                    "confidence {:.2} < {:.2} threshold",
                    vote.confidence, threshold
                ),
            );
        }
        (GateVerdict::Pass, String::new())
    }

    fn check_correlation(&self, asset: &str, ctx: &GateContext) -> (GateVerdict, String) {
        let Some(guard) = &self.correlation_guard else {
            return (GateVerdict::Pass, String::new());
        };
        if guard.would_exceed_limit(asset, &ctx.open_assets) {
            let count = guard.get_correlated_count(asset, &ctx.open_assets);
            return (
                GateVerdict::BlockCorrelation,
                format!(
                    "{count} open positions correlated >= {:.0}% with {asset}",
                    guard.corr_threshold() * 100.0
                ),
            );
        }
        (GateVerdict::Pass, String::new())
    }
}

// ── Stateless gate checks (free functions) ───────────────────────────

fn check_risk(ctx: &GateContext) -> (GateVerdict, String) {
    if !ctx.can_trade {
        return (GateVerdict::BlockRisk, ctx.risk_reason.clone());
    }
    (GateVerdict::Pass, String::new())
}

fn check_vol_filter(ctx: &GateContext) -> (GateVerdict, String) {
    if ctx.vol_pct < ctx.vol_entry_min {
        return (
            GateVerdict::BlockVolFilter,
            format!(
                "volatility too low ({:.0}% < {:.0}%)",
                ctx.vol_pct * 100.0,
                ctx.vol_entry_min * 100.0
            ),
        );
    }
    if ctx.vol_pct > ctx.vol_entry_max {
        return (
            GateVerdict::BlockVolFilter,
            format!(
                "volatility too high ({:.0}% > {:.0}%)",
                ctx.vol_pct * 100.0,
                ctx.vol_entry_max * 100.0
            ),
        );
    }
    (GateVerdict::Pass, String::new())
}

fn check_quality(ctx: &GateContext) -> (GateVerdict, String) {
    if ctx.quality < ctx.qual_min {
        return (
            GateVerdict::BlockQuality,
            format!("quality {:.0} < {:.0} minimum", ctx.quality, ctx.qual_min),
        );
    }
    (GateVerdict::Pass, String::new())
}

fn check_ao_divergence(signal: Side, ctx: &GateContext) -> (GateVerdict, String) {
    match signal {
        Side::Buy if ctx.ao <= 0.0 => (
            GateVerdict::BlockAoDivergence,
            format!("AO={:.4} <= 0 for buy signal", ctx.ao),
        ),
        Side::Sell if ctx.ao >= 0.0 => (
            GateVerdict::BlockAoDivergence,
            format!("AO={:.4} >= 0 for sell signal", ctx.ao),
        ),
        _ => (GateVerdict::Pass, String::new()),
    }
}

fn check_fee_viability(ctx: &GateContext) -> (GateVerdict, String) {
    let round_trip = 2.0 * (ctx.fee_taker + ctx.fee_slip);
    let ratio = if ctx.tp_pct > 0.0 {
        round_trip / ctx.tp_pct
    } else {
        1.0
    };
    if ratio > MAX_FEE_TP_RATIO {
        return (
            GateVerdict::BlockFeeViability,
            format!(
                "fee/TP ratio {:.0}% > 30% (RT={:.4}%, TP={:.4}%)",
                ratio * 100.0,
                round_trip * 100.0,
                ctx.tp_pct * 100.0
            ),
        );
    }
    (GateVerdict::Pass, String::new())
}

fn check_cnn_agreement(signal: Side, ctx: &GateContext) -> (GateVerdict, String) {
    if !ctx.cnn_enabled {
        return (GateVerdict::Pass, String::new());
    }
    let Some(vote) = ctx.cnn_result else {
        return (GateVerdict::Pass, String::new());
    };
    if !vote.agrees_with(signal) {
        return (
            GateVerdict::BlockCnnDisagree,
            format!(
                "CNN says {} (conf={:.2}), signal={}",
                vote.side.as_str(),
                vote.confidence,
                signal.as_str()
            ),
        );
    }
    (GateVerdict::Pass, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CnnVote;

    const NOW: u64 = 1_000_000;

    /// A context that clears every gate for a `Buy` (quality ok, AO positive,
    /// fee/TP comfortably under 30%).
    fn passing_ctx() -> GateContext {
        GateContext {
            quality: 60.0,
            ao: 1.0,
            tp_pct: 0.01,
            ..Default::default()
        }
    }

    fn gate() -> ExecutionGate {
        ExecutionGate::default()
    }

    #[test]
    fn all_clear_passes() {
        let mut g = gate();
        let (v, r) = g.evaluate("btc", Side::Buy, false, &passing_ctx(), NOW);
        assert_eq!(v, GateVerdict::Pass, "reason: {r}");
        assert_eq!(g.metrics.evals("btc"), 1);
    }

    #[test]
    fn risk_blocks_when_cannot_trade() {
        let mut g = gate();
        let ctx = GateContext {
            can_trade: false,
            risk_reason: "daily loss hit".to_string(),
            ..passing_ctx()
        };
        let (v, r) = g.evaluate("btc", Side::Buy, false, &ctx, NOW);
        assert_eq!(v, GateVerdict::BlockRisk);
        assert_eq!(r, "daily loss hit");
    }

    #[test]
    fn vol_filter_bounds() {
        let mut g = gate();
        let low = GateContext {
            vol_pct: 0.10,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &low, NOW).0,
            GateVerdict::BlockVolFilter
        );
        let high = GateContext {
            vol_pct: 0.95,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &high, NOW).0,
            GateVerdict::BlockVolFilter
        );
        let ok = GateContext {
            vol_pct: 0.50,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &ok, NOW).0,
            GateVerdict::Pass
        );
    }

    #[test]
    fn quality_gate_and_add_skip() {
        let mut g = gate();
        let low_q = GateContext {
            quality: 30.0,
            ..passing_ctx()
        };
        // First entry: quality gate active → blocks.
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &low_q, NOW).0,
            GateVerdict::BlockQuality
        );
        // Stack-add: quality gate skipped → passes (everything else clears).
        assert_eq!(
            g.evaluate("btc", Side::Buy, true, &low_q, NOW).0,
            GateVerdict::Pass
        );
    }

    #[test]
    fn ao_divergence_direction() {
        let mut g = gate();
        let buy_bad = GateContext {
            ao: -0.5,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &buy_bad, NOW).0,
            GateVerdict::BlockAoDivergence
        );
        let sell_bad = GateContext {
            ao: 0.5,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Sell, false, &sell_bad, NOW).0,
            GateVerdict::BlockAoDivergence
        );
        // Correct directions pass.
        let sell_ok = GateContext {
            ao: -0.5,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Sell, false, &sell_ok, NOW).0,
            GateVerdict::Pass
        );
        // AO gate skipped on adds even when divergent.
        assert_eq!(
            g.evaluate("btc", Side::Buy, true, &buy_bad, NOW).0,
            GateVerdict::Pass
        );
    }

    #[test]
    fn fee_viability_ratio() {
        let mut g = gate();
        // RT = 2*(0.0006+0.0001) = 0.0014; tp 0.004 → 35% > 30% → block.
        let bad = GateContext {
            tp_pct: 0.004,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &bad, NOW).0,
            GateVerdict::BlockFeeViability
        );
        // tp 0.005 → 28% < 30% → pass.
        let ok = GateContext {
            tp_pct: 0.005,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &ok, NOW).0,
            GateVerdict::Pass
        );
    }

    #[test]
    fn cnn_gates_pass_through_when_disabled_or_absent() {
        let mut g = gate();
        // disabled
        let disabled = GateContext {
            cnn_enabled: false,
            cnn_result: Some(CnnVote::new(Side::Sell, 0.1)),
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &disabled, NOW).0,
            GateVerdict::Pass
        );
        // enabled but no vote
        let no_vote = GateContext {
            cnn_enabled: true,
            cnn_result: None,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &no_vote, NOW).0,
            GateVerdict::Pass
        );
    }

    #[test]
    fn cnn_agreement_and_confidence() {
        let mut g = gate();
        // disagree
        let disagree = GateContext {
            cnn_enabled: true,
            cnn_result: Some(CnnVote::new(Side::Sell, 0.9)),
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &disagree, NOW).0,
            GateVerdict::BlockCnnDisagree
        );
        // agree but low confidence (min_confidence 0.60)
        let low_conf = GateContext {
            cnn_enabled: true,
            cnn_result: Some(CnnVote::new(Side::Buy, 0.40)),
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &low_conf, NOW).0,
            GateVerdict::BlockCnnConfidence
        );
        // agree, confident
        let ok = GateContext {
            cnn_enabled: true,
            cnn_result: Some(CnnVote::new(Side::Buy, 0.80)),
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &ok, NOW).0,
            GateVerdict::Pass
        );
    }

    #[test]
    fn adaptive_threshold_overrides_min_confidence() {
        // base 0.60; after one loss → 0.62. A 0.61 vote should now be blocked
        // even though it clears the fixed 0.60 min_confidence.
        let mut g = ExecutionGate::new(
            ConsecutiveLossBreaker::default(),
            None,
            Some(AdaptiveThreshold::default()),
        );
        g.adaptive_threshold
            .as_mut()
            .unwrap()
            .record_outcome("btc", false); // → 0.62
        let ctx = GateContext {
            cnn_enabled: true,
            cnn_result: Some(CnnVote::new(Side::Buy, 0.61)),
            min_confidence: 0.60,
            ..passing_ctx()
        };
        assert_eq!(
            g.evaluate("btc", Side::Buy, false, &ctx, NOW).0,
            GateVerdict::BlockCnnConfidence
        );
    }

    #[test]
    fn circuit_breaker_gate_trips_and_recovers() {
        let mut g = gate();
        // record 3 losses → breaker opens
        g.record_trade_outcome("btc", false, NOW);
        g.record_trade_outcome("btc", false, NOW);
        g.record_trade_outcome("btc", false, NOW);
        let (v, _) = g.evaluate("btc", Side::Buy, false, &passing_ctx(), NOW);
        assert_eq!(v, GateVerdict::BlockCircuitBreaker);
        // after cooldown the gate clears again
        let (v, _) = g.evaluate("btc", Side::Buy, false, &passing_ctx(), NOW + 900);
        assert_eq!(v, GateVerdict::Pass);
    }

    #[test]
    fn correlation_gate_blocks_cluster() {
        let mut g = ExecutionGate::new(
            ConsecutiveLossBreaker::default(),
            Some(CorrelationGuard::new(50, 2, 0.7)),
            None,
        );
        let up: Vec<f64> = (0..20).map(|i| (i as f64) * 0.001).collect();
        for &v in &up {
            g.update_correlation("btc", v);
            g.update_correlation("eth", v);
            g.update_correlation("sol", v);
        }
        let ctx = GateContext {
            open_assets: vec!["eth".to_string(), "sol".to_string()],
            ..passing_ctx()
        };
        let (v, _) = g.evaluate("btc", Side::Buy, false, &ctx, NOW);
        assert_eq!(v, GateVerdict::BlockCorrelation);
    }

    #[test]
    fn priority_order_circuit_breaker_wins_over_risk() {
        // Both the circuit breaker AND risk would block; the breaker is gate #1
        // so its verdict must be the one returned.
        let mut g = gate();
        g.record_trade_outcome("btc", false, NOW);
        g.record_trade_outcome("btc", false, NOW);
        g.record_trade_outcome("btc", false, NOW);
        let ctx = GateContext {
            can_trade: false,
            ..passing_ctx()
        };
        let (v, _) = g.evaluate("btc", Side::Buy, false, &ctx, NOW);
        assert_eq!(v, GateVerdict::BlockCircuitBreaker);
    }

    #[test]
    fn metrics_track_evals_and_blocks() {
        let mut g = gate();
        let bad = GateContext {
            can_trade: false,
            ..passing_ctx()
        };
        g.evaluate("btc", Side::Buy, false, &bad, NOW);
        g.evaluate("btc", Side::Buy, false, &bad, NOW);
        g.evaluate("btc", Side::Buy, false, &passing_ctx(), NOW);
        assert_eq!(g.metrics.evals("btc"), 3);
        assert_eq!(g.metrics.block_count("btc", GateVerdict::BlockRisk), 2);
        let stats = g.metrics.block_stats("btc");
        assert_eq!(stats.get("block_risk"), Some(&2));
        assert_eq!(stats.get("total_evals"), Some(&3));

        // snapshot() mirrors the same counters as sorted vecs for export.
        let snap = g.metrics.snapshot();
        assert_eq!(snap.evals, vec![("btc".to_string(), 3)]);
        assert_eq!(
            snap.blocks,
            vec![("btc".to_string(), "block_risk".to_string(), 2)]
        );
    }
}
