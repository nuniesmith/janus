//! Gate-A janus-side verdict computation (GATE_A_PREREGISTRATION.md §4a).
//!
//! This module is the committed, re-runnable computation behind the
//! `gate_a_verdict` binary — the instrument for the pre-registered J1
//! criterion (fee-adjusted signal edge), the J4 informational report
//! (confidence-bucket monotonicity), and the M2 enforce-flip prerequisite
//! (blocked-vs-actionable separation). Before this existed, the janus
//! secondary verdict had no committed computation (the 2026-07-09 analysis
//! lived only in a memory note) and Aug 1 would have been adjectives.
//!
//! ## Frozen semantics implemented here (prereg §4a, amendment 2026-07-17)
//!
//! * **Window:** defaults to `2026-07-18T00:00:00Z → 2026-08-01T23:59:59Z`
//!   (CLI-overridable; the frozen window is the default so the committed
//!   binary reproduces the verdict with no arguments).
//! * **Fee-aware cohort only:** rewards written before
//!   `JANUS_EXPERIENCE_REWARD_FEE_BPS=8` deployed (2026-07-09) are fee-free
//!   and must be excluded. The boundary is conservative — a full UTC day
//!   after the deploy by default — CLI-overridable via `--fee-boundary-ms`,
//!   and ALWAYS recorded in the output header.
//! * **Population:** directional (Buy/Sell) decisions only, split into
//!   *actionable* (gate `blocked == false`) and *blocked* cohorts. Points
//!   with no recorded gate verdict are excluded from BOTH cohorts and
//!   counted separately (`unknown_gate`) — unknown gate status is never
//!   silently assumed either way.
//! * **J1 pass rule:** mean σ-normalized fee-adjusted reward > 0 with
//!   de-correlated t ≥ 2.0 AND long/short each ≥ 20% of directional
//!   decisions; a side under 20% yields "INSUFFICIENT-REGIME-COVERAGE",
//!   which is NOT a pass.
//!
//! ## The de-correlated t-statistic (the honest one)
//!
//! Rewards are 5-bar holding returns recorded every bar, so consecutive
//! rewards on the same symbol share up to 4 of their 5 bars — an MA(4)-style
//! overlap that makes the naive iid t overstate significance ~5x in variance
//! terms (the 2026-07-09 measurement: naive t = 7.3 → de-correlated t = 1.33).
//! The correction implemented here is a **Newey–West (Bartlett kernel) HAC
//! standard error with lag = the reward horizon (5 bars), computed on each
//! symbol's time-ordered reward series and pooled across symbols under a
//! cross-symbol independence assumption**. Bartlett weights keep the
//! long-run-variance estimate non-negative. Residual cross-*symbol*
//! correlation (regime beta) is NOT corrected by this estimator — that risk
//! is addressed by the J1 long/short coverage requirement, and the method
//! string in the output says so.
//!
//! ## Units caveat (printed in every report)
//!
//! Rewards are in volatility-σ units, not money, and every directional
//! decision is charged a full round-trip fee even when consecutive
//! same-direction decisions overlap one real position (conservative). A J1
//! pass licenses advisory bridging only — sizing requires USDT accounting.

use serde::Serialize;

use crate::persistence::experience_store::SamplePoint;

// ─── Frozen defaults (prereg §4a) ─────────────────────────────────────────────

/// Frozen measurement-window start (prereg §4a J1).
pub const DEFAULT_WINDOW_FROM: &str = "2026-07-18T00:00:00Z";
/// Frozen measurement-window end (inclusive; prereg §4a J1).
pub const DEFAULT_WINDOW_TO: &str = "2026-08-01T23:59:59Z";
/// Conservative fee-aware-cohort boundary: `JANUS_EXPERIENCE_REWARD_FEE_BPS=8`
/// deployed on 2026-07-09; the default boundary is the NEXT UTC midnight so no
/// fee-free reward can straddle in. Overridable via `--fee-boundary-ms`
/// (recorded in the output header either way). The frozen window starts
/// 2026-07-18, comfortably inside the fee-aware cohort.
pub const DEFAULT_FEE_BOUNDARY: &str = "2026-07-10T00:00:00Z";
/// The experience reward horizon in bars (Phase 2.5, forward/experience.rs):
/// consecutive per-bar rewards overlap up to `HORIZON_BARS - 1` bars, hence
/// the HAC lag below.
pub const HORIZON_BARS: usize = 5;
/// Long/short regime-coverage floor (prereg §4a J1): each side must be ≥ 20%
/// of directional decisions or the verdict is "insufficient regime coverage".
pub const MIN_SIDE_FRACTION: f64 = 0.20;
/// De-correlated t threshold for a J1 pass (prereg §4a J1) and for the
/// blocked-vs-actionable separation prerequisite.
pub const T_THRESHOLD: f64 = 2.0;

/// Action indices as persisted in the Qdrant payload
/// (`0=Buy, 1=Sell, 2=Hold, 3=Close` — experience_store.rs).
pub const ACTION_BUY: u8 = 0;
/// See [`ACTION_BUY`].
pub const ACTION_SELL: u8 = 1;

// ─── Decision extraction ──────────────────────────────────────────────────────

/// One experience point reduced to the fields the verdict statistics need.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    /// Unix epoch milliseconds of the decision bar.
    pub timestamp_ms: i64,
    /// Trading symbol (per-symbol series feed the HAC estimator).
    pub symbol: String,
    /// Action index (`0=Buy, 1=Sell, 2=Hold, 3=Close`).
    pub action_type: u8,
    /// σ-normalized, fee-adjusted 5-bar reward (fee-aware cohort only after
    /// the boundary filter).
    pub reward: f64,
    /// Signal confidence, when the producer recorded it (side-channel meta).
    pub confidence: Option<f64>,
    /// Gate verdict: `Some(true)` blocked, `Some(false)` actionable, `None`
    /// when the point predates the side-channel meta (excluded from both
    /// cohorts — never assumed).
    pub blocked: Option<bool>,
}

impl DecisionRecord {
    /// True for Buy/Sell recorded actions.
    pub fn is_directional(&self) -> bool {
        self.action_type == ACTION_BUY || self.action_type == ACTION_SELL
    }
}

/// Map a scrolled Qdrant point to a [`DecisionRecord`].
///
/// `timestamp_ms`, `action_symbol`, `action_type` and `reward` are required
/// (absent/malformed ⇒ `None`, counted as unparseable by the caller);
/// `confidence` and `blocked` are optional side-channel fields and map to
/// `None` when absent rather than being defaulted.
pub fn decision_from_point(point: &SamplePoint) -> Option<DecisionRecord> {
    let p = &point.payload;
    let timestamp_ms = p.get("timestamp_ms")?.as_i64()?;
    let symbol = p.get("action_symbol")?.as_str()?.to_string();
    let action = p.get("action_type")?.as_i64()?;
    if !(0..=3).contains(&action) {
        return None;
    }
    let reward = p.get("reward")?.as_f64()?;
    if !reward.is_finite() {
        return None;
    }
    Some(DecisionRecord {
        timestamp_ms,
        symbol,
        action_type: action as u8,
        reward,
        confidence: p.get("confidence").and_then(|v| v.as_f64()),
        blocked: p.get("blocked").and_then(|v| v.as_bool()),
    })
}

// ─── Population filter ────────────────────────────────────────────────────────

/// Window + fee-cohort bounds applied to every decision.
#[derive(Debug, Clone, Copy)]
pub struct PopulationFilter {
    /// Window start (inclusive, epoch ms).
    pub from_ms: i64,
    /// Window end (inclusive, epoch ms).
    pub to_ms: i64,
    /// Fee-aware-cohort boundary (inclusive, epoch ms): decisions strictly
    /// before this are fee-free and excluded.
    pub fee_boundary_ms: i64,
}

/// Where every scrolled point went — printed in the output header so the
/// population is auditable (prereg: "PRINT the boundary + total/filtered
/// counts in an output header").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PopulationCounts {
    /// Points returned by the Qdrant window scroll.
    pub scrolled: usize,
    /// Points that parsed into a [`DecisionRecord`].
    pub parsed: usize,
    /// Parsed decisions outside `[from_ms, to_ms]` (scroll already windows on
    /// timestamp; nonzero only if the CLI window is narrower than the scroll).
    pub out_of_window: usize,
    /// In-window decisions before the fee boundary (fee-free cohort — excluded).
    pub pre_fee_boundary: usize,
    /// Fee-aware in-window decisions whose recorded action is Hold/Close.
    pub non_directional: usize,
    /// Directional fee-aware decisions with NO recorded gate verdict —
    /// excluded from both cohorts, never assumed.
    pub unknown_gate: usize,
    /// Directional fee-aware decisions the gate blocked.
    pub blocked: usize,
    /// The J1 population: directional, fee-aware, gate-passed decisions.
    pub actionable: usize,
}

/// The filtered populations the statistics run on.
#[derive(Debug, Clone, Default)]
pub struct Cohorts {
    /// Directional, fee-aware, non-blocked — the J1 population.
    pub actionable: Vec<DecisionRecord>,
    /// Directional, fee-aware, gate-blocked — the enforce-flip comparison arm.
    pub blocked: Vec<DecisionRecord>,
    /// Audit trail of every exclusion.
    pub counts: PopulationCounts,
}

/// Apply the frozen population filter (prereg §4a J1): directional (Buy/Sell),
/// fee-aware cohort, split by gate verdict. Precedence of exclusion counting:
/// window → fee boundary → directionality → gate.
pub fn build_cohorts(
    decisions: impl IntoIterator<Item = DecisionRecord>,
    filter: &PopulationFilter,
) -> Cohorts {
    let mut cohorts = Cohorts::default();
    for d in decisions {
        cohorts.counts.parsed += 1;
        if d.timestamp_ms < filter.from_ms || d.timestamp_ms > filter.to_ms {
            cohorts.counts.out_of_window += 1;
            continue;
        }
        if d.timestamp_ms < filter.fee_boundary_ms {
            cohorts.counts.pre_fee_boundary += 1;
            continue;
        }
        if !d.is_directional() {
            cohorts.counts.non_directional += 1;
            continue;
        }
        match d.blocked {
            None => cohorts.counts.unknown_gate += 1,
            Some(true) => {
                cohorts.counts.blocked += 1;
                cohorts.blocked.push(d);
            }
            Some(false) => {
                cohorts.counts.actionable += 1;
                cohorts.actionable.push(d);
            }
        }
    }
    cohorts
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Arithmetic mean (0.0 for an empty slice).
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample variance (n−1 denominator; 0.0 when n < 2).
fn sample_variance(xs: &[f64], m: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
}

/// Naive iid t-statistic for `mean(xs) = 0`. This is the WRONG statistic for
/// the overlapping-horizon rewards (it overstates significance ~5x in
/// variance terms) and is reported only alongside the de-correlated t so the
/// gap is visible. Returns 0.0 when n < 2 or the variance is 0.
pub fn naive_t(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = sample_variance(xs, m);
    if var <= 0.0 {
        return 0.0;
    }
    m / (var / n as f64).sqrt()
}

/// Newey–West (Bartlett kernel) long-run variance of a single time-ordered
/// series around the given center:
///
/// `lrv = γ̂₀ + 2 · Σ_{l=1..lag} (1 − l/(lag+1)) · γ̂_l`
///
/// with `γ̂_l = (1/n) Σ_{t=l..n−1} (x_t − c)(x_{t−l} − c)`. Bartlett weights
/// guarantee `lrv ≥ 0`. `c` is the pooled (grand) mean — the quantity whose
/// standard error the caller is building.
fn newey_west_lrv(xs: &[f64], center: f64, lag: usize) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let gamma0 = xs.iter().map(|x| (x - center).powi(2)).sum::<f64>() / nf;
    let mut lrv = gamma0;
    for l in 1..=lag.min(n.saturating_sub(1)) {
        let weight = 1.0 - l as f64 / (lag as f64 + 1.0);
        let cov = xs[l..]
            .iter()
            .zip(xs[..n - l].iter())
            .map(|(a, b)| (a - center) * (b - center))
            .sum::<f64>()
            / nf;
        lrv += 2.0 * weight * cov;
    }
    lrv.max(0.0)
}

/// De-correlated (overlap-aware) t-statistic for the pooled mean of several
/// independent time-ordered series (one per symbol).
///
/// **Method (documented per prereg §4a J1):** Newey–West HAC standard error
/// with Bartlett kernel and `lag` = the reward horizon (5 bars), estimated on
/// each symbol's time-ordered reward series and pooled under cross-symbol
/// independence:
///
/// `Var(x̄) ≈ (1/N²) · Σ_symbols n_s · lrv_s`,  `t = x̄ / √Var(x̄)`
///
/// This corrects the per-symbol serial correlation induced by the 5-bar
/// overlapping reward horizon — the naive iid t overstates significance ~5x
/// in variance terms on such data (2026-07-09: naive 7.3 vs de-correlated
/// 1.33). Cross-symbol (regime-beta) correlation is deliberately NOT modeled
/// here; the J1 long/short coverage floor handles regime exposure.
///
/// Each inner slice MUST be time-ordered. Returns 0.0 when there is no data
/// or no variance.
pub fn decorrelated_t(series_by_symbol: &[Vec<f64>], lag: usize) -> f64 {
    let n_total: usize = series_by_symbol.iter().map(|s| s.len()).sum();
    if n_total < 2 {
        return 0.0;
    }
    let grand_mean = series_by_symbol.iter().flatten().sum::<f64>() / n_total as f64;
    let pooled: f64 = series_by_symbol
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.len() as f64 * newey_west_lrv(s, grand_mean, lag))
        .sum();
    let var_of_mean = pooled / (n_total as f64).powi(2);
    if var_of_mean <= 0.0 {
        return 0.0;
    }
    grand_mean / var_of_mean.sqrt()
}

/// Group the population's rewards into per-symbol, time-ordered series — the
/// shape [`decorrelated_t`] requires.
pub fn rewards_by_symbol(population: &[DecisionRecord]) -> Vec<Vec<f64>> {
    use std::collections::BTreeMap;
    let mut sorted: Vec<&DecisionRecord> = population.iter().collect();
    sorted.sort_by_key(|d| d.timestamp_ms);
    let mut groups: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for d in sorted {
        groups.entry(d.symbol.as_str()).or_default().push(d.reward);
    }
    groups.into_values().collect()
}

/// One-sided Welch t-statistic for `mean(a) > mean(b)` (positive ⇒ a's mean
/// exceeds b's). Used for the enforce-flip prerequisite with
/// a = actionable, b = blocked: `t ≥ 2` ⇒ blocked decisions are measurably
/// WORSE than actionable ones. Returns 0.0 when either side has n < 2 or the
/// pooled SE is 0.
pub fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (sample_variance(a, ma), sample_variance(b, mb));
    let se2 = va / a.len() as f64 + vb / b.len() as f64;
    if se2 <= 0.0 {
        return 0.0;
    }
    (ma - mb) / se2.sqrt()
}

// ─── Long/short mix (regime coverage) ─────────────────────────────────────────

/// Long/short composition of the directional population (prereg §4a J1:
/// each side ≥ 20% or the verdict is "insufficient regime coverage").
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct MixReport {
    pub n_long: usize,
    pub n_short: usize,
    pub frac_long: f64,
    pub frac_short: f64,
    /// True iff both sides are ≥ [`MIN_SIDE_FRACTION`] of the population.
    pub coverage_ok: bool,
}

/// Compute the long/short mix of a directional population.
pub fn long_short_mix(population: &[DecisionRecord]) -> MixReport {
    let n_long = population
        .iter()
        .filter(|d| d.action_type == ACTION_BUY)
        .count();
    let n_short = population
        .iter()
        .filter(|d| d.action_type == ACTION_SELL)
        .count();
    let n = n_long + n_short;
    if n == 0 {
        return MixReport::default();
    }
    let frac_long = n_long as f64 / n as f64;
    let frac_short = n_short as f64 / n as f64;
    MixReport {
        n_long,
        n_short,
        frac_long,
        frac_short,
        coverage_ok: frac_long >= MIN_SIDE_FRACTION && frac_short >= MIN_SIDE_FRACTION,
    }
}

// ─── Confidence buckets (J4, informational) ──────────────────────────────────

/// Mean reward within one confidence bucket.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ConfidenceBucket {
    /// Inclusive lower bound.
    pub lo: f64,
    /// Exclusive upper bound (inclusive for the final 0.9–1.0 bucket).
    pub hi: f64,
    pub n: usize,
    pub mean_reward: f64,
}

/// J4 (informational): mean reward per confidence bucket
/// `[0.6–0.7, 0.7–0.8, 0.8–0.9, 0.9–1.0]` over the population. Decisions
/// without a recorded confidence, or with confidence outside `[0.6, 1.0]`,
/// are omitted (the live consensus floor is 0.6, so sub-0.6 confidences do
/// not occur on the signal path).
pub fn confidence_buckets(population: &[DecisionRecord]) -> Vec<ConfidenceBucket> {
    const EDGES: [(f64, f64); 4] = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)];
    EDGES
        .iter()
        .map(|&(lo, hi)| {
            let rewards: Vec<f64> = population
                .iter()
                .filter_map(|d| {
                    let c = d.confidence?;
                    let in_bucket = if hi >= 1.0 {
                        c >= lo && c <= hi // final bucket includes 1.0
                    } else {
                        c >= lo && c < hi
                    };
                    in_bucket.then_some(d.reward)
                })
                .collect();
            ConfidenceBucket {
                lo,
                hi,
                n: rewards.len(),
                mean_reward: mean(&rewards),
            }
        })
        .collect()
}

// ─── Report assembly ──────────────────────────────────────────────────────────

/// J1 statistics + verdict.
#[derive(Debug, Clone, Serialize)]
pub struct J1Report {
    /// Actionable-population size.
    pub n: usize,
    /// Mean σ-normalized fee-adjusted reward (σ units, NOT money).
    pub mean_reward_sigma: f64,
    /// The naive iid t — reported ONLY to show the overlap inflation.
    pub naive_t: f64,
    /// The honest statistic: overlap-aware t (see [`decorrelated_t`]).
    pub decorrelated_t: f64,
    /// HAC lag used (= reward horizon in bars).
    pub hac_lag_bars: usize,
    /// Human description of the SE method (committed, not negotiable ex post).
    pub t_method: String,
    /// `PASS` / `FAIL` / `INSUFFICIENT-REGIME-COVERAGE` per prereg §4a J1.
    pub verdict: String,
}

/// Blocked-vs-actionable comparison — the M2 enforce-flip prerequisite
/// (prereg §4a: blocked must be WORSE, one-sided t ≥ 2, else enforcing the
/// gate only halves the sample).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BlockedComparison {
    pub n_actionable: usize,
    pub n_blocked: usize,
    pub mean_actionable: f64,
    pub mean_blocked: f64,
    /// One-sided Welch t for actionable > blocked (positive ⇒ gate separates
    /// in the right direction).
    pub one_sided_t: f64,
    /// True iff `one_sided_t ≥ 2.0` — the enforce-flip evidence bar.
    pub separation_ok: bool,
}

/// The full machine-readable verdict report (serialized as the binary's
/// stdout JSON; the human summary is rendered from the same struct).
#[derive(Debug, Clone, Serialize)]
pub struct VerdictReport {
    /// RFC3339 generation instant.
    pub generated_at: String,
    /// Qdrant collection scrolled.
    pub collection: String,
    /// Approximate total points in the collection at run time (integrity
    /// check: should grow ~10 rows/min while ingestion is healthy).
    pub collection_points_total: u64,
    pub window_from: String,
    pub window_to: String,
    pub window_from_ms: i64,
    pub window_to_ms: i64,
    /// The fee-aware-cohort boundary actually applied (header requirement).
    pub fee_boundary_ms: i64,
    pub fee_boundary: String,
    /// Standing caveat printed with every run.
    pub units_note: String,
    pub counts: PopulationCounts,
    pub j1: J1Report,
    pub long_short_mix: MixReport,
    pub j4_confidence_buckets: Vec<ConfidenceBucket>,
    pub blocked_vs_actionable: BlockedComparison,
}

/// Millisecond-timestamp → RFC3339 (UTC), falling back to the raw number.
fn ms_to_rfc3339(ms: i64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms)
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_else(|| format!("{ms}ms"))
}

/// Compute the full report from parsed decisions (pure aside from the clock).
pub fn compute_report(
    decisions: Vec<DecisionRecord>,
    filter: &PopulationFilter,
    scrolled: usize,
    collection: &str,
    collection_points_total: u64,
) -> VerdictReport {
    let mut cohorts = build_cohorts(decisions, filter);
    cohorts.counts.scrolled = scrolled;

    let actionable_rewards: Vec<f64> = cohorts.actionable.iter().map(|d| d.reward).collect();
    let blocked_rewards: Vec<f64> = cohorts.blocked.iter().map(|d| d.reward).collect();

    let mean_reward = mean(&actionable_rewards);
    let naive = naive_t(&actionable_rewards);
    let grouped = rewards_by_symbol(&cohorts.actionable);
    let decorr = decorrelated_t(&grouped, HORIZON_BARS);

    let mix = long_short_mix(&cohorts.actionable);
    let verdict = if !mix.coverage_ok {
        "INSUFFICIENT-REGIME-COVERAGE".to_string()
    } else if mean_reward > 0.0 && decorr >= T_THRESHOLD {
        "PASS".to_string()
    } else {
        "FAIL".to_string()
    };

    let one_sided_t = welch_t(&actionable_rewards, &blocked_rewards);

    VerdictReport {
        generated_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        collection: collection.to_string(),
        collection_points_total,
        window_from: ms_to_rfc3339(filter.from_ms),
        window_to: ms_to_rfc3339(filter.to_ms),
        window_from_ms: filter.from_ms,
        window_to_ms: filter.to_ms,
        fee_boundary_ms: filter.fee_boundary_ms,
        fee_boundary: ms_to_rfc3339(filter.fee_boundary_ms),
        units_note: "Rewards are volatility-σ units, NOT money; every directional decision is \
                     charged a full round-trip fee (conservative). A J1 pass licenses advisory \
                     bridging only — sizing requires funding-ledger-style USDT accounting."
            .to_string(),
        counts: cohorts.counts,
        j1: J1Report {
            n: actionable_rewards.len(),
            mean_reward_sigma: mean_reward,
            naive_t: naive,
            decorrelated_t: decorr,
            hac_lag_bars: HORIZON_BARS,
            t_method: format!(
                "Newey-West (Bartlett) HAC SE, lag={HORIZON_BARS} bars (= reward horizon), \
                 per-symbol time-ordered series pooled under cross-symbol independence; \
                 corrects the 5-bar overlapping-horizon autocorrelation that inflates the \
                 naive iid t ~5x in variance terms. Cross-symbol (regime-beta) correlation \
                 is not modeled; coverage is enforced by the long/short mix requirement."
            ),
            verdict,
        },
        long_short_mix: mix,
        j4_confidence_buckets: confidence_buckets(&cohorts.actionable),
        blocked_vs_actionable: BlockedComparison {
            n_actionable: actionable_rewards.len(),
            n_blocked: blocked_rewards.len(),
            mean_actionable: mean_reward,
            mean_blocked: mean(&blocked_rewards),
            one_sided_t,
            separation_ok: one_sided_t >= T_THRESHOLD,
        },
    }
}

/// Render the human-readable summary (stderr side of the binary; the JSON on
/// stdout is the machine-readable record).
pub fn render_human(r: &VerdictReport) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "══ Gate-A janus verdict (prereg §4a) ══");
    let _ = writeln!(s, "generated:       {}", r.generated_at);
    let _ = writeln!(
        s,
        "collection:      {} (~{} points total)",
        r.collection, r.collection_points_total
    );
    let _ = writeln!(s, "window:          {} → {}", r.window_from, r.window_to);
    let _ = writeln!(
        s,
        "fee boundary:    {} ({} ms) — fee-aware cohort only",
        r.fee_boundary, r.fee_boundary_ms
    );
    let c = &r.counts;
    let _ = writeln!(
        s,
        "population:      scrolled {} → parsed {} | excluded: out-of-window {}, \
         pre-fee-boundary {}, non-directional {}, unknown-gate {} | blocked {} | ACTIONABLE {}",
        c.scrolled,
        c.parsed,
        c.out_of_window,
        c.pre_fee_boundary,
        c.non_directional,
        c.unknown_gate,
        c.blocked,
        c.actionable
    );
    let _ = writeln!(s, "── J1: fee-adjusted signal edge ──");
    let _ = writeln!(
        s,
        "mean reward:     {:+.5} σ over n={} (naive t={:.2}, DE-CORRELATED t={:.2}, \
         bar: mean>0 ∧ t≥{:.1})",
        r.j1.mean_reward_sigma, r.j1.n, r.j1.naive_t, r.j1.decorrelated_t, T_THRESHOLD
    );
    let _ = writeln!(s, "t method:        {}", r.j1.t_method);
    let m = &r.long_short_mix;
    let _ = writeln!(
        s,
        "long/short mix:  {} long ({:.1}%) / {} short ({:.1}%) — coverage {}",
        m.n_long,
        m.frac_long * 100.0,
        m.n_short,
        m.frac_short * 100.0,
        if m.coverage_ok {
            "OK (both ≥ 20%)"
        } else {
            "INSUFFICIENT (a side < 20%)"
        }
    );
    let _ = writeln!(s, "J1 VERDICT:      {}", r.j1.verdict);
    let _ = writeln!(s, "── J4 (informational): confidence buckets ──");
    for b in &r.j4_confidence_buckets {
        let _ = writeln!(
            s,
            "  [{:.1}–{:.1}):  n={:<6} mean reward {:+.5} σ",
            b.lo, b.hi, b.n, b.mean_reward
        );
    }
    let bc = &r.blocked_vs_actionable;
    let _ = writeln!(s, "── enforce-flip prerequisite: blocked vs actionable ──");
    let _ = writeln!(
        s,
        "actionable:      n={} mean {:+.5} σ | blocked: n={} mean {:+.5} σ | \
         one-sided t (actionable > blocked) = {:.2} → separation {}",
        bc.n_actionable,
        bc.mean_actionable,
        bc.n_blocked,
        bc.mean_blocked,
        bc.one_sided_t,
        if bc.separation_ok {
            "OK (t ≥ 2)"
        } else {
            "NOT SHOWN (do not enforce)"
        }
    );
    let _ = writeln!(s, "note:            {}", r.units_note);
    s
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(
        ts: i64,
        symbol: &str,
        action: u8,
        reward: f64,
        confidence: Option<f64>,
        blocked: Option<bool>,
    ) -> DecisionRecord {
        DecisionRecord {
            timestamp_ms: ts,
            symbol: symbol.to_string(),
            action_type: action,
            reward,
            confidence,
            blocked,
        }
    }

    fn wide_filter() -> PopulationFilter {
        PopulationFilter {
            from_ms: 0,
            to_ms: i64::MAX,
            fee_boundary_ms: 0,
        }
    }

    /// Tiny deterministic LCG so the statistical tests are reproducible
    /// without pulling in a rand dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            // Numerical Recipes LCG constants; top 53 bits → [0, 1).
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Approximate standard normal (Irwin–Hall, 12 uniforms).
        fn next_gauss(&mut self) -> f64 {
            (0..12).map(|_| self.next_f64()).sum::<f64>() - 6.0
        }
    }

    // ── decision extraction ───────────────────────────────────────────────

    fn point(payload: serde_json::Map<String, serde_json::Value>) -> SamplePoint {
        SamplePoint {
            id: "test".into(),
            vector: Vec::new(),
            payload,
        }
    }

    fn full_payload() -> serde_json::Map<String, serde_json::Value> {
        let mut p = serde_json::Map::new();
        p.insert(
            "timestamp_ms".into(),
            serde_json::json!(1_784_332_800_000_i64),
        );
        p.insert("action_symbol".into(), serde_json::json!("BTCUSDT"));
        p.insert("action_type".into(), serde_json::json!(0));
        p.insert("reward".into(), serde_json::json!(0.12));
        p.insert("confidence".into(), serde_json::json!(0.72));
        p.insert("blocked".into(), serde_json::json!(false));
        p
    }

    #[test]
    fn test_decision_from_point_maps_all_fields() {
        let d = decision_from_point(&point(full_payload())).expect("should parse");
        assert_eq!(d.timestamp_ms, 1_784_332_800_000);
        assert_eq!(d.symbol, "BTCUSDT");
        assert_eq!(d.action_type, ACTION_BUY);
        assert!((d.reward - 0.12).abs() < 1e-12);
        assert_eq!(d.confidence, Some(0.72));
        assert_eq!(d.blocked, Some(false));
        assert!(d.is_directional());
    }

    #[test]
    fn test_decision_from_point_optional_fields_stay_none() {
        // Points that predate the side-channel meta have neither confidence
        // nor blocked — they must map to None, NOT be defaulted.
        let mut p = full_payload();
        p.remove("confidence");
        p.remove("blocked");
        let d = decision_from_point(&point(p)).expect("should parse");
        assert_eq!(d.confidence, None);
        assert_eq!(d.blocked, None);
    }

    #[test]
    fn test_decision_from_point_rejects_missing_or_bad_required() {
        for key in ["timestamp_ms", "action_symbol", "action_type", "reward"] {
            let mut p = full_payload();
            p.remove(key);
            assert!(
                decision_from_point(&point(p)).is_none(),
                "missing {key} must not parse"
            );
        }
        let mut p = full_payload();
        p.insert("action_type".into(), serde_json::json!(9));
        assert!(decision_from_point(&point(p)).is_none(), "bad action index");
        let mut p = full_payload();
        p.insert("reward".into(), serde_json::json!(f64::NAN));
        assert!(
            decision_from_point(&point(p)).is_none(),
            "non-finite reward must not parse"
        );
    }

    // ── population filter ─────────────────────────────────────────────────

    #[test]
    fn test_build_cohorts_applies_every_exclusion() {
        let filter = PopulationFilter {
            from_ms: 1_000,
            to_ms: 2_000,
            fee_boundary_ms: 1_500,
        };
        let decisions = vec![
            dec(500, "A", 0, 0.1, None, Some(false)),   // before window
            dec(2_500, "A", 0, 0.1, None, Some(false)), // after window
            dec(1_200, "A", 0, 0.1, None, Some(false)), // in window, PRE fee boundary
            dec(1_600, "A", 2, 0.0, None, Some(false)), // Hold — non-directional
            dec(1_600, "A", 3, 0.1, None, Some(false)), // Close — non-directional
            dec(1_700, "A", 0, 0.1, None, None),        // directional, unknown gate
            dec(1_800, "A", 1, -0.2, None, Some(true)), // directional, blocked
            dec(1_900, "A", 0, 0.3, None, Some(false)), // ACTIONABLE
            dec(1_500, "B", 1, 0.2, None, Some(false)), // boundary ts == fee boundary → in
        ];
        let cohorts = build_cohorts(decisions, &filter);
        let c = cohorts.counts;
        assert_eq!(c.parsed, 9);
        assert_eq!(c.out_of_window, 2);
        assert_eq!(c.pre_fee_boundary, 1);
        assert_eq!(c.non_directional, 2);
        assert_eq!(c.unknown_gate, 1);
        assert_eq!(c.blocked, 1);
        assert_eq!(c.actionable, 2);
        assert_eq!(cohorts.actionable.len(), 2);
        assert_eq!(cohorts.blocked.len(), 1);
        // The fee-boundary timestamp itself is INSIDE the fee-aware cohort.
        assert!(cohorts.actionable.iter().any(|d| d.timestamp_ms == 1_500));
    }

    // ── the overlap-aware t (the core of J1) ──────────────────────────────

    /// The load-bearing statistical test: rewards built as 5-bar overlapping
    /// sums of iid noise (exactly the experience pipeline's overlap
    /// structure) must yield a de-correlated t SUBSTANTIALLY below the naive
    /// iid t — the naive statistic's ~5x variance understatement is the bug
    /// the prereg method exists to fix.
    #[test]
    fn test_decorrelated_t_deflates_overlapping_horizon_significance() {
        let mut rng = Lcg(42);
        let n_bars = 4_000usize;
        let horizon = HORIZON_BARS;
        let drift = 0.02;

        // Per-bar iid shocks; reward_t = Σ_{k=0..4} shock_{t+k} + drift —
        // an MA(4) series exactly like the 5-bar holding-return overlap.
        let shocks: Vec<f64> = (0..n_bars + horizon).map(|_| rng.next_gauss()).collect();
        let rewards: Vec<f64> = (0..n_bars)
            .map(|t| shocks[t..t + horizon].iter().sum::<f64>() + drift)
            .collect();

        let naive = naive_t(&rewards);
        let decorr = decorrelated_t(&[rewards], HORIZON_BARS);

        assert!(naive.is_finite() && decorr.is_finite());
        // Theoretical SE inflation for Bartlett lag-5 on MA(4)-overlap data is
        // ~1.9x; require at least a 25% haircut so estimation noise can't
        // flip the assertion while a broken (no-op) correction still fails it.
        assert!(
            decorr.abs() < 0.75 * naive.abs(),
            "overlap correction must reduce |t| substantially: naive={naive:.3}, decorr={decorr:.3}"
        );
    }

    /// On genuinely iid data the correction must be roughly a no-op — the
    /// estimator only removes covariance that is actually there.
    #[test]
    fn test_decorrelated_t_close_to_naive_on_iid_data() {
        let mut rng = Lcg(7);
        let rewards: Vec<f64> = (0..4_000).map(|_| rng.next_gauss() + 0.05).collect();
        let naive = naive_t(&rewards);
        let decorr = decorrelated_t(&[rewards], HORIZON_BARS);
        assert!(
            (decorr - naive).abs() / naive.abs() < 0.25,
            "iid data should be ~unchanged: naive={naive:.3}, decorr={decorr:.3}"
        );
    }

    #[test]
    fn test_decorrelated_t_pools_across_symbols() {
        // Two symbols with the same positive drift: pooling must produce a
        // finite positive t larger than either symbol alone (more data).
        let mut rng = Lcg(99);
        let make = |rng: &mut Lcg| -> Vec<f64> {
            (0..1_000).map(|_| rng.next_gauss() * 0.5 + 0.1).collect()
        };
        let (a, b) = (make(&mut rng), make(&mut rng));
        let t_a = decorrelated_t(std::slice::from_ref(&a), HORIZON_BARS);
        let t_pooled = decorrelated_t(&[a, b], HORIZON_BARS);
        assert!(t_pooled > 0.0 && t_a > 0.0);
        assert!(
            t_pooled > t_a,
            "pooling two same-drift symbols should raise t: {t_a:.2} → {t_pooled:.2}"
        );
    }

    #[test]
    fn test_t_statistics_degenerate_inputs_are_zero() {
        assert_eq!(naive_t(&[]), 0.0);
        assert_eq!(naive_t(&[1.0]), 0.0);
        assert_eq!(naive_t(&[2.0, 2.0, 2.0]), 0.0, "zero variance");
        assert_eq!(decorrelated_t(&[], HORIZON_BARS), 0.0);
        assert_eq!(decorrelated_t(&[vec![1.0]], HORIZON_BARS), 0.0);
    }

    #[test]
    fn test_rewards_by_symbol_orders_by_time_within_symbol() {
        let pop = vec![
            dec(3, "B", 0, 0.3, None, Some(false)),
            dec(1, "A", 0, 0.1, None, Some(false)),
            dec(2, "A", 1, 0.2, None, Some(false)),
            dec(1, "B", 1, 0.4, None, Some(false)),
        ];
        let groups = rewards_by_symbol(&pop);
        // BTreeMap ⇒ deterministic symbol order A, B; each time-ordered.
        assert_eq!(groups, vec![vec![0.1, 0.2], vec![0.4, 0.3]]);
    }

    // ── welch (blocked vs actionable) ─────────────────────────────────────

    #[test]
    fn test_welch_t_separates_shifted_populations() {
        let mut rng = Lcg(1234);
        let actionable: Vec<f64> = (0..800).map(|_| rng.next_gauss() * 0.5 + 0.3).collect();
        let blocked: Vec<f64> = (0..800).map(|_| rng.next_gauss() * 0.5 - 0.1).collect();
        let t = welch_t(&actionable, &blocked);
        assert!(
            t > T_THRESHOLD,
            "clear separation must clear the bar, t={t:.2}"
        );
        // Symmetry: swapping the arms flips the sign.
        assert!(welch_t(&blocked, &actionable) < -T_THRESHOLD);
    }

    #[test]
    fn test_welch_t_no_separation_on_identical_distributions() {
        let mut rng = Lcg(555);
        let a: Vec<f64> = (0..2_000).map(|_| rng.next_gauss()).collect();
        let b: Vec<f64> = (0..2_000).map(|_| rng.next_gauss()).collect();
        let t = welch_t(&a, &b);
        assert!(
            t.abs() < T_THRESHOLD,
            "same-distribution arms must not fake separation, t={t:.2}"
        );
        assert_eq!(welch_t(&[], &a), 0.0, "empty arm yields 0");
    }

    // ── mix + buckets ─────────────────────────────────────────────────────

    #[test]
    fn test_long_short_mix_coverage_thresholds() {
        let make = |n_long: usize, n_short: usize| -> Vec<DecisionRecord> {
            (0..n_long)
                .map(|i| dec(i as i64, "A", ACTION_BUY, 0.0, None, Some(false)))
                .chain(
                    (0..n_short).map(|i| dec(i as i64, "A", ACTION_SELL, 0.0, None, Some(false))),
                )
                .collect()
        };
        // 19% short → insufficient.
        let m = long_short_mix(&make(81, 19));
        assert!(!m.coverage_ok);
        assert_eq!(m.n_short, 19);
        // Exactly 20% short → OK (bar is ≥).
        assert!(long_short_mix(&make(80, 20)).coverage_ok);
        // Balanced → OK.
        assert!(long_short_mix(&make(50, 50)).coverage_ok);
        // One-sided (the 07-09 98.5%-long failure shape) → insufficient.
        assert!(!long_short_mix(&make(197, 3)).coverage_ok);
        // Empty population → not OK.
        assert!(!long_short_mix(&[]).coverage_ok);
    }

    #[test]
    fn test_confidence_buckets_boundaries_and_means() {
        let pop = vec![
            dec(1, "A", 0, 0.10, Some(0.60), Some(false)), // [0.6,0.7)
            dec(2, "A", 0, 0.30, Some(0.699), Some(false)), // [0.6,0.7)
            dec(3, "A", 0, 0.50, Some(0.70), Some(false)), // [0.7,0.8)
            dec(4, "A", 0, 0.70, Some(0.90), Some(false)), // [0.9,1.0]
            dec(5, "A", 0, 0.90, Some(1.0), Some(false)),  // 1.0 inclusive
            dec(6, "A", 0, 9.99, None, Some(false)),       // no confidence — omitted
            dec(7, "A", 0, 9.99, Some(0.5), Some(false)),  // below 0.6 — omitted
        ];
        let buckets = confidence_buckets(&pop);
        assert_eq!(buckets.len(), 4);
        assert_eq!(buckets[0].n, 2);
        assert!((buckets[0].mean_reward - 0.20).abs() < 1e-12);
        assert_eq!(buckets[1].n, 1);
        assert!((buckets[1].mean_reward - 0.50).abs() < 1e-12);
        assert_eq!(buckets[2].n, 0);
        assert_eq!(buckets[3].n, 2);
        assert!((buckets[3].mean_reward - 0.80).abs() < 1e-12);
    }

    // ── report assembly ───────────────────────────────────────────────────

    #[test]
    fn test_compute_report_insufficient_coverage_is_not_a_pass() {
        // Strongly positive rewards but 100% long — the regime-beta failure
        // mode. The verdict must be INSUFFICIENT-REGIME-COVERAGE, not PASS.
        let decisions: Vec<DecisionRecord> = (0..200)
            .map(|i| dec(i, "A", ACTION_BUY, 0.5, Some(0.75), Some(false)))
            .collect();
        let report = compute_report(decisions, &wide_filter(), 200, "experiences", 1234);
        assert_eq!(report.j1.verdict, "INSUFFICIENT-REGIME-COVERAGE");
        assert_eq!(report.counts.actionable, 200);
        assert_eq!(report.collection_points_total, 1234);
    }

    #[test]
    fn test_compute_report_pass_and_fail_paths() {
        let mut rng = Lcg(2026);
        // Balanced longs/shorts, clearly positive mean → PASS.
        let decisions: Vec<DecisionRecord> = (0..2_000)
            .map(|i| {
                let action = if i % 2 == 0 { ACTION_BUY } else { ACTION_SELL };
                dec(
                    i,
                    if i % 3 == 0 { "A" } else { "B" },
                    action,
                    rng.next_gauss() * 0.3 + 0.2,
                    Some(0.8),
                    Some(false),
                )
            })
            .collect();
        let report = compute_report(decisions, &wide_filter(), 2_000, "experiences", 0);
        assert!(report.long_short_mix.coverage_ok);
        assert!(report.j1.decorrelated_t >= T_THRESHOLD);
        assert_eq!(report.j1.verdict, "PASS");

        // Mean ~0 → FAIL (coverage fine, edge absent).
        let mut rng = Lcg(2027);
        let decisions: Vec<DecisionRecord> = (0..2_000)
            .map(|i| {
                let action = if i % 2 == 0 { ACTION_BUY } else { ACTION_SELL };
                dec(
                    i,
                    "A",
                    action,
                    rng.next_gauss() * 0.3,
                    Some(0.8),
                    Some(false),
                )
            })
            .collect();
        let report = compute_report(decisions, &wide_filter(), 2_000, "experiences", 0);
        assert_eq!(report.j1.verdict, "FAIL");
    }

    #[test]
    fn test_report_serializes_and_renders() {
        let decisions = vec![
            dec(1, "A", ACTION_BUY, 0.1, Some(0.75), Some(false)),
            dec(2, "A", ACTION_SELL, -0.05, Some(0.65), Some(false)),
            dec(3, "A", ACTION_BUY, 0.2, None, Some(true)),
        ];
        let report = compute_report(decisions, &wide_filter(), 3, "experiences", 42);
        // Machine-readable side must serialize…
        let json = serde_json::to_string_pretty(&report).expect("report serializes");
        for key in [
            "fee_boundary_ms",
            "decorrelated_t",
            "long_short_mix",
            "j4_confidence_buckets",
            "blocked_vs_actionable",
            "counts",
        ] {
            assert!(json.contains(key), "JSON must contain {key}");
        }
        // …and the human side must carry the header requirements.
        let human = render_human(&report);
        assert!(human.contains("fee boundary"));
        assert!(human.contains("ACTIONABLE 2"));
        assert!(human.contains("J1 VERDICT"));
    }

    #[test]
    fn test_default_window_constants_parse() {
        // The frozen defaults must parse as RFC3339 and be ordered:
        // fee boundary < window start < window end.
        let from = chrono::DateTime::parse_from_rfc3339(DEFAULT_WINDOW_FROM)
            .expect("window-from parses")
            .timestamp_millis();
        let to = chrono::DateTime::parse_from_rfc3339(DEFAULT_WINDOW_TO)
            .expect("window-to parses")
            .timestamp_millis();
        let fee = chrono::DateTime::parse_from_rfc3339(DEFAULT_FEE_BOUNDARY)
            .expect("fee-boundary parses")
            .timestamp_millis();
        assert!(fee < from && from < to);
    }
}
