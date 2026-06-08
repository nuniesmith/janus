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
//! - **Live now (8 of 9 gates):** the `RiskManager` verdict (`risk`); the CNN
//!   vote (`cnn_agreement` + `cnn_confidence`, active only when
//!   `ENABLE_CNN_INFERENCE`); open positions + per-tick log-returns
//!   (`correlation`); and — via the `indicators-ta` signal engine
//!   ([`GateProducers`]) — the Awesome Oscillator (`ao`), volatility percentile
//!   (`vol_pct`), a momentum/wave quality proxy, and an ATR-derived `tp_pct` for
//!   fee viability.
//! - **Follow-up (janus `TODO.md`, Track C):** closed-trade outcomes for the
//!   consecutive-loss breaker — the 9th gate (needs the position-close path).
//!
//! Gates without a real input yet are **inert pass-throughs** — `build_context`
//! picks values that clear them, so the gate never emits a spurious block.

use std::collections::{HashMap, HashSet};
use std::env;

pub use janus_execution_gate::{
    BreakerSnapshotEntry, CnnVote, ConsecutiveLossBreaker, CorrelationGuard, ExecutionGate,
    GateContext, GateMetricsSnapshot, GateVerdict, Side,
};
use janus_indicators::{Candle, IndicatorConfig, Indicators, VolatilityPercentile};

/// Rolling window (bars) for the volatility-percentile tracker.
const VOL_PCT_WINDOW: usize = 100;

/// Default TP-target ATR multiple for the fee-viability gate's `tp_pct`
/// (`tp_pct = tp_atr_mult × ATR / close`) — ≈ a 2R target on a 2×ATR stop.
/// Override with `JANUS_GATE_TP_ATR_MULT`.
const DEFAULT_TP_ATR_MULT: f64 = 4.0;

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

/// Per-signal gate inputs derived from the `indicators-ta` engine state. A
/// `None` field leaves the corresponding gate an inert pass-through.
#[derive(Debug, Clone, Copy, Default)]
pub struct Producers {
    /// Awesome Oscillator (drives the AO-divergence gate).
    pub ao: Option<f64>,
    /// Volatility percentile in `[0, 1]` (drives the vol-filter gate).
    pub vol_pct: Option<f64>,
    /// Signal-quality score in `[0, 100]` (drives the quality gate).
    pub quality: Option<f64>,
    /// Take-profit target as a fraction of price (drives the fee-viability
    /// gate). Derived as `tp_atr_mult × ATR / close` — see [`GateProducers`].
    pub tp_pct: Option<f64>,
}

/// One closed OHLCV bar fed to [`GateProducers::on_candle`]. `time_ms` is the
/// candle open time in milliseconds.
#[derive(Debug, Clone, Copy)]
pub struct Bar {
    pub time_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Per-symbol producer state driven by the `indicators-ta` streaming signal
/// engine. Feed every closed candle via [`on_candle`](Self::on_candle); read the
/// current [`Producers`] for an asset via [`snapshot`](Self::snapshot).
///
/// `ao` comes straight from the engine; `vol_pct` from a `VolatilityPercentile`
/// tracker fed the engine's ATR; `quality` is a proxy blending the engine's
/// momentum + wave-ratio percentiles (a tunable stand-in for Ruby's quality
/// score — the fuller `compute_signal` confluence score is a later refinement);
/// `tp_pct` is `tp_atr_mult × ATR / close`, the take-profit target the
/// fee-viability gate measures round-trip fees against. A symbol reports
/// `Producers::default()` (all `None`) until the engine warms up, so the gates
/// stay inert during warmup rather than acting on noise.
pub struct GateProducers {
    engines: HashMap<String, Indicators>,
    vol: HashMap<String, VolatilityPercentile>,
    warmed: HashSet<String>,
    /// TP target as an ATR multiple, for the fee-viability gate's `tp_pct`.
    tp_atr_mult: f64,
}

impl Default for GateProducers {
    fn default() -> Self {
        Self::new()
    }
}

impl GateProducers {
    pub fn new() -> Self {
        let tp_atr_mult = std::env::var("JANUS_GATE_TP_ATR_MULT")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|m| *m > 0.0)
            .unwrap_or(DEFAULT_TP_ATR_MULT);
        Self {
            engines: HashMap::new(),
            vol: HashMap::new(),
            warmed: HashSet::new(),
            tp_atr_mult,
        }
    }

    /// Feed a closed [`Bar`] for `symbol`.
    pub fn on_candle(&mut self, symbol: &str, bar: Bar) {
        let ind = self
            .engines
            .entry(symbol.to_string())
            .or_insert_with(|| Indicators::new(IndicatorConfig::default()));
        let warm = ind.update(&Candle {
            time: bar.time_ms,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        });
        let vp = self
            .vol
            .entry(symbol.to_string())
            .or_insert_with(|| VolatilityPercentile::new(VOL_PCT_WINDOW));
        vp.update(ind.atr);
        if warm {
            self.warmed.insert(symbol.to_string());
        }
    }

    /// Current producer snapshot for `symbol` (all `None` until the engine warms).
    pub fn snapshot(&self, symbol: &str) -> Producers {
        if !self.warmed.contains(symbol) {
            return Producers::default();
        }
        let Some(ind) = self.engines.get(symbol) else {
            return Producers::default();
        };
        // Quality proxy: blend the momentum + wave-ratio percentiles (each 0–1)
        // into a 0–100 score. Tunable; see the type docs.
        let quality = ((ind.mom_pct + ind.wr_pct) * 0.5 * 100.0).clamp(0.0, 100.0);
        // Fee-viability TP target: `tp_atr_mult × ATR` as a fraction of the last
        // close. `None` until both ATR and a close are available.
        let tp_pct = match (ind.closes.back().copied(), ind.atr) {
            (Some(c), Some(a)) if c > 0.0 && a > 0.0 => Some(self.tp_atr_mult * a / c),
            _ => None,
        };
        Producers {
            ao: Some(ind.ao),
            vol_pct: self.vol.get(symbol).map(|v| v.vol_pct),
            quality: Some(quality),
            tp_pct,
        }
    }
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

    /// Snapshot the gate's block/eval counters for export (Redis/Grafana).
    pub fn metrics_snapshot(&self) -> GateMetricsSnapshot {
        self.gate.metrics.snapshot()
    }

    /// Export the consecutive-loss breaker state for cross-restart persistence.
    pub fn export_breaker(&self) -> Vec<BreakerSnapshotEntry> {
        self.gate.breaker.export()
    }

    /// Restore the consecutive-loss breaker state (e.g. on startup from Redis).
    pub fn import_breaker(&mut self, entries: Vec<BreakerSnapshotEntry>) {
        self.gate.breaker.import(entries);
    }

    /// Build a [`GateContext`] with no engine producers (all inert pass-throughs).
    /// Convenience over
    /// [`build_context_with_producers`](Self::build_context_with_producers).
    pub fn build_context(
        side: Side,
        risk_check: &str,
        min_confidence: f64,
        cnn_vote: Option<(Side, f64)>,
        open_assets: Vec<String>,
    ) -> GateContext {
        Self::build_context_with_producers(
            side,
            risk_check,
            min_confidence,
            cnn_vote,
            open_assets,
            Producers::default(),
        )
    }

    /// Build a [`GateContext`] from the per-signal state plus the `indicators-ta`
    /// engine [`Producers`].
    ///
    /// `risk_check` is the `RiskManager` verdict string (`"ok"` / `"rejected:…"`).
    /// `cnn_vote` is the CNN's `(side, confidence)` when CNN gating is active.
    /// `open_assets` are the symbols with open positions (correlation gate).
    /// `producers` carries the engine's `ao` / `vol_pct` / `quality`; any absent
    /// (`None`) field falls back to an inert pass-through so its gate can't fire —
    /// notably the AO pass-through is direction-dependent, hence `side` is needed.
    pub fn build_context_with_producers(
        side: Side,
        risk_check: &str,
        min_confidence: f64,
        cnn_vote: Option<(Side, f64)>,
        open_assets: Vec<String>,
        producers: Producers,
    ) -> GateContext {
        let rejected = risk_check.starts_with("rejected");
        let ao_passthrough = match side {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        };
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
            // Engine producers, with inert pass-throughs when not yet warmed:
            // vol mid-band, quality clears qual_min, AO matches the side, TP
            // viable. Fees keep the GateContext defaults (taker + slippage).
            ao: producers.ao.unwrap_or(ao_passthrough),
            vol_pct: producers.vol_pct.unwrap_or(0.5),
            quality: producers.quality.unwrap_or(100.0),
            tp_pct: producers.tp_pct.unwrap_or(0.01),
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

    #[test]
    fn producers_drive_vol_quality_ao_and_fee_gates() {
        let mut g = ForwardGate::new(false);
        // Real vol_pct below the entry band (< 0.15) → vol filter blocks.
        let ctx = ForwardGate::build_context_with_producers(
            Side::Buy,
            "ok",
            0.6,
            None,
            Vec::new(),
            Producers {
                vol_pct: Some(0.05),
                ..Default::default()
            },
        );
        assert_eq!(
            g.evaluate_entry("btc", Side::Buy, false, &ctx, NOW).verdict,
            GateVerdict::BlockVolFilter
        );
        // Real quality below the floor (< 50) → quality blocks.
        let ctx = ForwardGate::build_context_with_producers(
            Side::Buy,
            "ok",
            0.6,
            None,
            Vec::new(),
            Producers {
                quality: Some(20.0),
                ..Default::default()
            },
        );
        assert_eq!(
            g.evaluate_entry("btc", Side::Buy, false, &ctx, NOW).verdict,
            GateVerdict::BlockQuality
        );
        // Real AO opposing a Buy (ao <= 0) → AO-divergence blocks.
        let ctx = ForwardGate::build_context_with_producers(
            Side::Buy,
            "ok",
            0.6,
            None,
            Vec::new(),
            Producers {
                ao: Some(-0.5),
                ..Default::default()
            },
        );
        assert_eq!(
            g.evaluate_entry("btc", Side::Buy, false, &ctx, NOW).verdict,
            GateVerdict::BlockAoDivergence
        );
        // Tiny TP target → round-trip fees dominate (> 30%) → fee-viability blocks.
        let ctx = ForwardGate::build_context_with_producers(
            Side::Buy,
            "ok",
            0.6,
            None,
            Vec::new(),
            Producers {
                tp_pct: Some(0.002),
                ..Default::default()
            },
        );
        assert_eq!(
            g.evaluate_entry("btc", Side::Buy, false, &ctx, NOW).verdict,
            GateVerdict::BlockFeeViability
        );
    }

    #[test]
    fn gate_producers_report_once_warm() {
        let mut p = GateProducers::new();
        // Too few candles → engine not warm → all-None (gates stay inert).
        for i in 0..10 {
            let px = 100.0 + i as f64;
            p.on_candle(
                "btc",
                Bar {
                    time_ms: i as i64 * 60_000,
                    open: px,
                    high: px + 1.0,
                    low: px - 1.0,
                    close: px,
                    volume: 10.0,
                },
            );
        }
        let cold = p.snapshot("btc");
        assert!(cold.ao.is_none() && cold.vol_pct.is_none() && cold.quality.is_none());
        // Unknown symbol is always inert.
        assert!(p.snapshot("eth").ao.is_none());
        // Enough candles → engine warms and reports producers.
        for i in 10..150 {
            let px = 100.0 + ((i as f64) * 0.2).sin() * 5.0;
            p.on_candle(
                "btc",
                Bar {
                    time_ms: i as i64 * 60_000,
                    open: px,
                    high: px + 1.0,
                    low: px - 1.0,
                    close: px,
                    volume: 10.0,
                },
            );
        }
        let warm = p.snapshot("btc");
        assert!(warm.ao.is_some(), "ao should be reported once warm");
        assert!(
            warm.vol_pct.is_some(),
            "vol_pct should be reported once warm"
        );
        let q = warm.quality.expect("quality reported once warm");
        assert!((0.0..=100.0).contains(&q), "quality in 0..=100, got {q}");
        assert!(warm.tp_pct.is_some(), "tp_pct should be reported once warm");
    }
}
