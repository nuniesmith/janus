//! Forward-loop integration for the execution gate.
//!
//! Wraps [`janus_execution_gate::ExecutionGate`] for use in the live signal
//! loop: owns the gate state, reads its config from the environment, builds a
//! [`GateContext`] from the loop's per-signal state, and renders an advisory
//! verdict (with optional enforcement behind `JANUS_GATE_ENFORCE`).
//!
//! # Staged wiring
//!
//! This mirrors how prop-firm / risk enforcement landed (advisory first, then
//! enforce behind a flag). The gate is on the live path in *advisory* mode (set
//! `JANUS_GATE_ENFORCE=1` to let a block suppress the execution submit). Inputs
//! are fed incrementally:
//!
//! - **Live now:** the `RiskManager` verdict (`risk`); the CNN vote
//!   (`cnn_agreement` + `cnn_confidence`, active only when `ENABLE_CNN_INFERENCE`);
//!   and open positions + per-tick log-returns (`correlation`).
//! - **Follow-ups (janus `TODO.md`, Track C):** realised-vol / quality / AO / fee
//!   producer values, and closed-trade outcomes for the consecutive-loss breaker.
//!
//! Gates without a real input yet are **inert pass-throughs** — `build_context`
//! picks values that clear them, so the gate never emits a spurious block.

use std::env;

pub use janus_execution_gate::{
    CnnVote, ConsecutiveLossBreaker, CorrelationGuard, ExecutionGate, GateContext, GateVerdict,
    Side,
};

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Outcome of evaluating one prospective entry through the gate.
#[derive(Debug, Clone)]
pub struct GateOutcome {
    /// The first blocking verdict, or [`GateVerdict::Pass`].
    pub verdict: GateVerdict,
    /// Human-readable reason for a block (empty on pass).
    pub reason: String,
    /// Compact value for the signal's `gate` metadata field: `pass`, or
    /// `block_<reason>:<detail>` (e.g. `block_risk:rejected:daily loss`).
    pub metadata: String,
    /// `true` when enforcement is on **and** the verdict blocks — the caller
    /// should then suppress the execution submit (the signal still publishes to
    /// the bus for observability, preserving "no autonomous execution").
    pub enforce_block: bool,
}

/// Live-loop wrapper around the execution gate.
#[derive(Debug)]
pub struct ForwardGate {
    gate: ExecutionGate,
    enforce: bool,
}

impl Default for ForwardGate {
    fn default() -> Self {
        Self::new(false)
    }
}

impl ForwardGate {
    /// Construct with an explicit enforcement flag (used by tests / callers that
    /// already resolved config).
    pub fn new(enforce: bool) -> Self {
        // Enable the correlation guard so the correlation gate goes live once the
        // loop feeds it returns + open positions. The consecutive-loss breaker is
        // present by default; it stays closed until `record_trade_outcome` is fed
        // from the position-close path (a follow-up).
        let gate = ExecutionGate::new(
            ConsecutiveLossBreaker::default(),
            Some(CorrelationGuard::default()),
            None,
        );
        Self { gate, enforce }
    }

    /// Build from the environment. `JANUS_GATE_ENFORCE=1` makes a blocking
    /// verdict suppress the execution submit; otherwise the gate is advisory.
    pub fn from_env() -> Self {
        Self::new(env_flag("JANUS_GATE_ENFORCE"))
    }

    /// Whether a blocking verdict will suppress the execution submit.
    pub fn enforcing(&self) -> bool {
        self.enforce
    }

    /// Build a [`GateContext`] from the inputs available at signal emission.
    ///
    /// `risk_check` is the `RiskManager` verdict string (`"ok"` / `"rejected:…"`).
    /// `cnn_vote` is the CNN's `(side, confidence)` when CNN gating is active.
    /// `open_assets` are the symbols with open positions (for the correlation
    /// gate). Producer inputs not yet plumbed (vol / quality / AO / fees) are
    /// set to pass-through values so their gates stay inert — notably the AO
    /// pass-through is direction-dependent, hence `side` is required here.
    pub fn build_context(
        side: Side,
        risk_check: &str,
        min_confidence: f64,
        cnn_vote: Option<(Side, f64)>,
        open_assets: Vec<String>,
    ) -> GateContext {
        let rejected = risk_check.starts_with("rejected");
        GateContext {
            // Real inputs.
            can_trade: !rejected,
            risk_reason: if rejected {
                risk_check.to_string()
            } else {
                "risk limit".to_string()
            },
            min_confidence,
            cnn_enabled: cnn_vote.is_some(),
            cnn_result: cnn_vote.map(|(s, c)| CnnVote::new(s, c)),
            open_assets,
            // Pass-through (inert) inputs — see module docs. quality >= qual_min,
            // tp/fees viable, vol mid-band (Default), and AO matching the side.
            quality: 100.0,
            tp_pct: 0.01,
            ao: match side {
                Side::Buy => 1.0,
                Side::Sell => -1.0,
            },
            ..Default::default()
        }
    }

    /// Evaluate a prospective entry and render a [`GateOutcome`].
    pub fn evaluate_entry(
        &mut self,
        asset: &str,
        side: Side,
        is_add: bool,
        ctx: &GateContext,
        now_secs: u64,
    ) -> GateOutcome {
        let (verdict, reason) = self.gate.evaluate(asset, side, is_add, ctx, now_secs);
        let metadata = if verdict.is_pass() {
            "pass".to_string()
        } else if reason.is_empty() {
            verdict.as_counter_label().to_string()
        } else {
            format!("{}:{}", verdict.as_counter_label(), reason)
        };
        let enforce_block = self.enforce && verdict.is_block();
        GateOutcome {
            verdict,
            reason,
            metadata,
            enforce_block,
        }
    }

    /// Feed a closed-trade outcome to the breaker / adaptive threshold
    /// (follow-up: call from the position-close path).
    pub fn record_trade_outcome(&mut self, asset: &str, is_win: bool, now_secs: u64) {
        self.gate.record_trade_outcome(asset, is_win, now_secs);
    }

    /// Feed a log-return to the correlation guard (follow-up: call per tick).
    pub fn update_correlation(&mut self, asset: &str, log_return: f64) {
        self.gate.update_correlation(asset, log_return);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000;

    #[test]
    fn ok_risk_passes_for_both_sides() {
        let mut g = ForwardGate::new(false);
        for side in [Side::Buy, Side::Sell] {
            let ctx = ForwardGate::build_context(side, "ok", 0.6, None, Vec::new());
            let out = g.evaluate_entry("btc", side, false, &ctx, NOW);
            assert_eq!(
                out.verdict,
                GateVerdict::Pass,
                "side {side:?}: {}",
                out.reason
            );
            assert_eq!(out.metadata, "pass");
            assert!(!out.enforce_block);
        }
    }

    #[test]
    fn rejected_risk_blocks_and_surfaces_reason() {
        let mut g = ForwardGate::new(false);
        let ctx =
            ForwardGate::build_context(Side::Buy, "rejected:daily loss", 0.6, None, Vec::new());
        let out = g.evaluate_entry("btc", Side::Buy, false, &ctx, NOW);
        assert_eq!(out.verdict, GateVerdict::BlockRisk);
        assert!(out.metadata.starts_with("block_risk:rejected:daily loss"));
        // advisory: does not request a block
        assert!(!out.enforce_block);
    }

    #[test]
    fn enforce_block_only_when_enforcing() {
        let ctx = ForwardGate::build_context(Side::Buy, "rejected:x", 0.6, None, Vec::new());
        let mut advisory = ForwardGate::new(false);
        assert!(
            !advisory
                .evaluate_entry("btc", Side::Buy, false, &ctx, NOW)
                .enforce_block
        );
        let mut enforcing = ForwardGate::new(true);
        let out = enforcing.evaluate_entry("btc", Side::Buy, false, &ctx, NOW);
        assert!(out.enforce_block);
        assert_eq!(out.verdict, GateVerdict::BlockRisk);
    }

    #[test]
    fn cnn_vote_when_present_is_gated() {
        let mut g = ForwardGate::new(false);
        // CNN disagrees with a Buy → block.
        let ctx =
            ForwardGate::build_context(Side::Buy, "ok", 0.6, Some((Side::Sell, 0.9)), Vec::new());
        assert_eq!(
            g.evaluate_entry("btc", Side::Buy, false, &ctx, NOW).verdict,
            GateVerdict::BlockCnnDisagree
        );
        // CNN agrees + confident → pass.
        let ctx =
            ForwardGate::build_context(Side::Buy, "ok", 0.6, Some((Side::Buy, 0.8)), Vec::new());
        assert_eq!(
            g.evaluate_entry("btc", Side::Buy, false, &ctx, NOW).verdict,
            GateVerdict::Pass
        );
    }

    #[test]
    fn ao_passthrough_does_not_spuriously_block() {
        // The AO gate is direction-dependent; the pass-through must clear it for
        // both sides (Stage-1 inert behaviour, no real AO fed).
        let mut g = ForwardGate::new(false);
        let buy = ForwardGate::build_context(Side::Buy, "ok", 0.6, None, Vec::new());
        let sell = ForwardGate::build_context(Side::Sell, "ok", 0.6, None, Vec::new());
        assert_eq!(
            g.evaluate_entry("btc", Side::Buy, false, &buy, NOW).verdict,
            GateVerdict::Pass
        );
        assert_eq!(
            g.evaluate_entry("eth", Side::Sell, false, &sell, NOW)
                .verdict,
            GateVerdict::Pass
        );
    }

    #[test]
    fn correlation_gate_blocks_a_correlated_cluster() {
        // The guard is enabled by `new`; feed an identical up-trend to four
        // assets so they are perfectly correlated, then enter `btc` with three
        // correlated positions already open (default max_correlated = 3).
        let mut g = ForwardGate::new(false);
        let series: Vec<f64> = (0..20).map(|i| (i as f64) * 0.001).collect();
        for v in &series {
            for a in ["btc", "eth", "sol", "ada"] {
                g.update_correlation(a, *v);
            }
        }
        let open = vec!["eth".to_string(), "sol".to_string(), "ada".to_string()];
        let ctx = ForwardGate::build_context(Side::Buy, "ok", 0.6, None, open);
        assert_eq!(
            g.evaluate_entry("btc", Side::Buy, false, &ctx, NOW).verdict,
            GateVerdict::BlockCorrelation
        );
    }
}
