//! Experience pipeline producer (Phase 1) — the forward half of
//! `docs/architecture/EXPERIENCE_PIPELINE.md` (§3 the record, §4 transport).
//!
//! Two pieces:
//!
//! - [`ExperienceRecorder`] — owned by the live signal-loop task (no locks).
//!   Keeps a rolling close window and **one pending decision** per
//!   `SYMBOL:interval` key; when the next candle for that key closes, the
//!   pending decision completes with the honest Phase-1 reward — the signed
//!   next-bar log return (§3.3) — and the finished row is handed to the
//!   writer over a bounded channel (`try_send`: the hot path never awaits
//!   and never blocks on the pipeline).
//! - [`spawn_writer`] — a detached task that buffers rows and flushes a
//!   batch every `JANUS_EXPERIENCE_BATCH_ROWS` rows or
//!   `JANUS_EXPERIENCE_FLUSH_SECS` seconds into a complete Arrow IPC file in
//!   the tmpfs spool (`.arrow.tmp` → fsync → atomic rename → `.arrow`,
//!   §4.2), then LPUSHes an `IngestJob` doorbell onto Redis (§4.3). Backward
//!   (`services/backward/src/tasks/intake.rs`) BRPOPs the doorbell, ingests,
//!   and deletes the file; its sweep recovers anything the doorbell misses,
//!   so Redis errors here are logged + counted, never retried.
//!
//! The whole pipeline is **off by default** (`JANUS_EXPERIENCE_ENABLED`):
//! when disabled, nothing is constructed and the live loop is untouched.
//!
//! Phase-1 simplifications (all documented in the design):
//! - `state_raw` is emitted as null — threading the live CNN feature column
//!   through the loop is deferred (the Arrow field is nullable by design).
//! - `action_qty` is `0.0` — position sizing is not decided at signal time
//!   in this system (the risk check derives stops, not quantity; execution
//!   sizes orders downstream).
//! - The optional round-trip-fee subtraction on non-Hold rewards is skipped.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use arrow::array::{BinaryArray, BooleanArray, Float32Array, Int64Array, StringArray, UInt8Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use prometheus::{IntCounter, IntGauge, register_int_counter, register_int_gauge};
use serde::Serialize;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// GAF pooling size: `k=3` ⇒ 9 pooled GASF values, matching backward's
/// `QDRANT_EXPERIENCE_DIM` default and the existing collection contract
/// (design §3.1). Changing this requires bumping that env var *before* the
/// Qdrant collection is first created.
pub const GAF_K: usize = 3;

/// Closes per GAF window (§3.5). 64 bars ≈ one hour of 1m candles — enough
/// texture for a 3×3 pooled GASF while staying far below the analyzers'
/// warmup, so the recorder warms strictly earlier than the signal path.
pub const GAF_WINDOW: usize = 64;

/// Bounded hot-path → writer channel. When the writer stalls and this fills,
/// rows are dropped and counted — the live loop must never block (§6).
const CHANNEL_CAPACITY: usize = 1024;

/// Redis list the backward intake worker BRPOPs (§4.3). Must match
/// `services/backward/src/tasks/intake.rs`.
pub const REDIS_QUEUE_KEY: &str = "janus:experience:ingest";

/// A pending decision completes at the next candle unless the gap exceeds
/// `max(2 × interval, PENDING_DONE_FLOOR_SECS)` — then it completes with
/// `done = true` (§3.3/§3.4).
const PENDING_DONE_FLOOR_SECS: i64 = 300;

/// A gap beyond `interval × PENDING_DROP_FACTOR` drops the pending decision
/// instead of emitting a reward across it (§3.3).
const PENDING_DROP_FACTOR: i64 = 10;

// ─────────────────────────────────────────────────────────────────────────────
// Metrics (default prometheus registry — same as janus_core::metrics)
// ─────────────────────────────────────────────────────────────────────────────

struct ExperienceMetrics {
    rows_recorded: IntCounter,
    rows_dropped_channel: IntCounter,
    pending_gap_dropped: IntCounter,
    batches_spooled: IntCounter,
    batches_failed: IntCounter,
    spool_drop_oldest: IntCounter,
    redis_push_failures: IntCounter,
    spool_bytes: IntGauge,
}

fn metrics() -> &'static ExperienceMetrics {
    static METRICS: OnceLock<ExperienceMetrics> = OnceLock::new();
    METRICS.get_or_init(|| ExperienceMetrics {
        rows_recorded: register_int_counter!(
            "janus_experience_rows_recorded_total",
            "Completed experience rows handed to the writer"
        )
        .expect("register janus_experience_rows_recorded_total"),
        rows_dropped_channel: register_int_counter!(
            "janus_experience_rows_dropped_total",
            "Experience rows dropped because the writer channel was full"
        )
        .expect("register janus_experience_rows_dropped_total"),
        pending_gap_dropped: register_int_counter!(
            "janus_experience_pending_gap_dropped_total",
            "Pending decisions dropped because the candle gap exceeded 10x the interval"
        )
        .expect("register janus_experience_pending_gap_dropped_total"),
        batches_spooled: register_int_counter!(
            "janus_experience_batches_spooled_total",
            "Arrow IPC batches written to the experience spool"
        )
        .expect("register janus_experience_batches_spooled_total"),
        batches_failed: register_int_counter!(
            "janus_experience_batches_failed_total",
            "Experience batches that failed to spool (rows lost)"
        )
        .expect("register janus_experience_batches_failed_total"),
        spool_drop_oldest: register_int_counter!(
            "janus_experience_spool_drop_oldest_total",
            "Spooled batches deleted by the drop-oldest size bound"
        )
        .expect("register janus_experience_spool_drop_oldest_total"),
        redis_push_failures: register_int_counter!(
            "janus_experience_redis_push_failures_total",
            "IngestJob doorbell LPUSH failures (the backward sweep recovers these)"
        )
        .expect("register janus_experience_redis_push_failures_total"),
        spool_bytes: register_int_gauge!(
            "janus_experience_spool_bytes",
            "Total bytes of finished .arrow batches currently in the spool"
        )
        .expect("register janus_experience_spool_bytes"),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Env-driven configuration (§4.1). All defaults match the design and the
/// fks compose wiring.
#[derive(Debug, Clone)]
pub struct ExperienceConfig {
    pub enabled: bool,
    pub spool_dir: PathBuf,
    pub batch_rows: usize,
    pub flush_secs: u64,
    pub spool_max_mb: u64,
}

impl ExperienceConfig {
    pub fn from_env() -> Self {
        fn parse<T: std::str::FromStr>(key: &str, default: T) -> T {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        let enabled = std::env::var("JANUS_EXPERIENCE_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self {
            enabled,
            spool_dir: PathBuf::from(
                std::env::var("JANUS_EXPERIENCE_SPOOL_DIR")
                    .unwrap_or_else(|_| "/dev/shm/janus/experience".to_string()),
            ),
            batch_rows: parse("JANUS_EXPERIENCE_BATCH_ROWS", 256),
            flush_secs: parse("JANUS_EXPERIENCE_FLUSH_SECS", 60),
            spool_max_mb: parse("JANUS_EXPERIENCE_SPOOL_MAX_MB", 256),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reward scheme (Phase 2.5)
// ─────────────────────────────────────────────────────────────────────────────

/// How the per-decision reward is computed (design §7 Phase-2 point 4, the
/// writer-side "honest upgrade"). Phase-1 used a single-bar log return; this
/// makes it a **multi-bar holding return, fee-aware and volatility-normalized**
/// — the reward a trade held for `horizon` bars would realize, risk-adjusted so
/// magnitudes are comparable across regimes/symbols (the DQN reads `reward`
/// directly into its TD target, so scale matters).
///
/// [`RewardConfig::legacy`] reproduces the exact Phase-1 single-bar return
/// (horizon 1, no fee, no normalization) and is what [`ExperienceRecorder::new`]
/// uses — keeping the frozen behavior for callers/tests that want it. Production
/// wires [`RewardConfig::from_env`], whose defaults turn the upgrade ON.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardConfig {
    /// Bars a decision is "held" before its reward is realized (≥ 1). The
    /// reward is the signed log return from the decision close to the close
    /// `horizon` bars later. `1` = the Phase-1 next-bar return.
    pub horizon: usize,
    /// Round-trip fee as a fraction of notional (e.g. `0.0004` = 4 bps),
    /// subtracted from every non-Hold reward as a fixed cost — so Buy/Sell are
    /// not friction-free. Hold pays nothing.
    pub fee: f64,
    /// Divide the (fee-adjusted) return by a rolling volatility estimate so
    /// rewards are ~unit-scale and comparable across regimes. Off = raw return.
    pub vol_norm: bool,
    /// Volatility floor (per-bar σ) so a flat window can't blow the reward up.
    pub vol_floor: f64,
}

impl RewardConfig {
    /// The frozen Phase-1 scheme: single-bar signed log return, no fee, no
    /// normalization. Bit-for-bit compatible with the original recorder.
    pub const fn legacy() -> Self {
        Self {
            horizon: 1,
            fee: 0.0,
            vol_norm: false,
            vol_floor: 1e-4,
        }
    }

    /// Production scheme from the environment. Defaults turn Phase 2.5 ON:
    /// a 5-bar horizon, volatility-normalized, no fee (operator opts into a
    /// fee that matches the venue). All values clamp to safe ranges.
    pub fn from_env() -> Self {
        fn parse<T: std::str::FromStr>(key: &str, default: T) -> T {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        let horizon = parse::<usize>("JANUS_EXPERIENCE_REWARD_HORIZON", 5).max(1);
        let fee = parse::<f64>("JANUS_EXPERIENCE_REWARD_FEE_BPS", 0.0).max(0.0) / 10_000.0;
        let vol_norm = std::env::var("JANUS_EXPERIENCE_REWARD_VOL_NORM")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        // Clamp to a *real* floor (1e-8), not f64::MIN_POSITIVE (~2e-308): the
        // latter is effectively zero, so a `VOL_FLOOR=0` config + a flat window
        // (σ=0) would divide the reward by ~0 and yield ±inf, corrupting the
        // training signal. 1e-8 keeps the normaliser finite for any input.
        let vol_floor = parse::<f64>("JANUS_EXPERIENCE_REWARD_VOL_FLOOR", 1e-4).max(1e-8);
        Self {
            horizon,
            fee,
            vol_norm,
            vol_floor,
        }
    }
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self::legacy()
    }
}

/// Sample standard deviation of the one-bar log returns of `closes` (the σ used
/// to risk-normalize rewards). Returns `None` with < 2 usable returns.
fn rolling_sigma(closes: &VecDeque<f32>) -> Option<f64> {
    if closes.len() < 3 {
        return None;
    }
    let rets: Vec<f64> = closes
        .iter()
        .zip(closes.iter().skip(1))
        .filter_map(|(a, b)| {
            let (a, b) = (*a as f64, *b as f64);
            (a > 0.0 && b > 0.0).then(|| (b / a).ln())
        })
        .collect();
    if rets.len() < 2 {
        return None;
    }
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
    Some(var.sqrt())
}

/// A completed decision's reward, decomposed so a verdict script can separate
/// the signal from its fixed cost (GATE_A_PREREGISTRATION.md §4g: the J1
/// headline is fee-dominated, but gross vs net was never stored per row so the
/// two could not be told apart).
///
/// - `net` — the exact scalar the replay buffer and TD target consume. Frozen:
///   bit-identical to the pre-decomposition formula, same rounding, same
///   finite safety net. This is the ONLY field trading/training reads.
/// - `gross` — the same reward under the same normalization but WITHOUT the fee
///   subtracted (the pre-fee signal).
/// - `fee_sigma` — the fee constant actually subtracted, expressed in the same
///   (σ-normalized) units as `net`/`gross`, so `net ≈ gross - fee_sigma`.
///
/// All three are `0.0` for Hold (no fee, no reward). `gross`/`fee_sigma` carry
/// the same finite guard as `net`: a degenerate config never stores ±inf/NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RewardParts {
    net: f32,
    gross: f32,
    fee_sigma: f32,
}

/// Compute a completed decision's reward under `cfg`, decomposed into
/// [`RewardParts`]. `bars_held` is how many bars actually elapsed (≤ horizon;
/// less only when an episode boundary closes the position early), used to scale
/// the volatility normalizer (σ over N bars ≈ σ₁ × √N). `sigma1` is the per-bar
/// σ from [`rolling_sigma`].
///
/// The `net` field is computed through the identical operation sequence the
/// pre-decomposition code used — `gross`/`fee_sigma` are tracked alongside on
/// their own variables and never feed back into `net`, so `net` stays
/// bit-identical (the zero-change property the next measurement window rests on).
fn compute_reward(
    action: janus_core::SignalType,
    entry_close: f64,
    exit_close: f64,
    bars_held: usize,
    sigma1: Option<f64>,
    cfg: &RewardConfig,
) -> RewardParts {
    let dir = match action {
        janus_core::SignalType::Buy => 1.0_f64,
        janus_core::SignalType::Sell => -1.0_f64,
        // Hold (and anything non-directional) earns nothing and pays no fee.
        _ => {
            return RewardParts {
                net: 0.0,
                gross: 0.0,
                fee_sigma: 0.0,
            };
        }
    };
    let gross_raw = dir * (exit_close / entry_close).ln();
    // NET — the frozen sequence. `r` takes the identical bits of `gross_raw`,
    // then the same `-= fee` / `/= sigma_h` as before: do not reorder.
    let mut r = gross_raw;
    // Round-trip fee is a fixed cost on any directional action.
    r -= cfg.fee;
    // Decomposition tracked on their own variables, normalized identically to
    // `r` so the three stay comparable (`net ≈ gross - fee_sigma`).
    let mut gross = gross_raw;
    let mut fee = cfg.fee;
    if cfg.vol_norm {
        let n = bars_held.max(1) as f64;
        let sigma_h = sigma1.unwrap_or(0.0).max(cfg.vol_floor) * n.sqrt();
        r /= sigma_h;
        gross /= sigma_h;
        fee /= sigma_h;
    }
    // Final safety net: a degenerate config/input must never inject a non-finite
    // value into the replay buffer (the DQN reads `net` straight into its TD
    // target) or the Qdrant payload. Fall back to a neutral 0.0 — identical to
    // the frozen guard for `net`, applied to all three.
    fn finite(x: f64) -> f32 {
        let v = x as f32;
        if v.is_finite() { v } else { 0.0 }
    }
    RewardParts {
        net: finite(r),
        gross: finite(gross),
        fee_sigma: finite(fee),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Records
// ─────────────────────────────────────────────────────────────────────────────

/// The decision context observed at a closed candle (§3.5): the post-gate
/// outcome. Consensus Hold / no consensus / filtered ⇒ `Hold`; a Buy/Sell
/// that was suppressed from the execution submit is still that action, with
/// the suppression reason in `blocked` (the gate is part of the environment).
#[derive(Debug, Clone)]
pub struct DecisionCtx {
    pub action: janus_core::SignalType,
    pub confidence: f64,
    pub blocked: Option<String>,
    /// The **advisory** 9-gate verdict for this decision, independent of
    /// enforcement: `Some("pass")` or `Some("block_<gate>:<detail>")` for an
    /// actionable Buy/Sell that was run through the execution gate, `None` for a
    /// Hold (no entry to gate). It reuses the exact `GateOutcome::metadata`
    /// string already emitted to the `gate` signal metadata / Redis — the gates
    /// are never re-run to fill it.
    ///
    /// DISTINCT FROM `blocked`; the two populations must never be conflated:
    /// - `blocked` = the decision was *actually excluded* from the actionable
    ///   cohort (risk / prop-firm ENFORCE, a gate ENFORCE block, or below the
    ///   execution floor). With `JANUS_GATE_ENFORCE=0` it is `None` even when
    ///   the advisory gate would have blocked.
    /// - `advisory_gate` = what the 9-gate *would have said*, recorded for every
    ///   gated entry whether or not enforcement is on. This is what makes the
    ///   pre-registration's enforce-flip prerequisite ("a 9-gate-blocked entry
    ///   is worse than an actionable one") measurable WITHOUT flipping
    ///   enforcement in production — the population the t=24.98 print never
    ///   actually covered (GATE_A_PREREGISTRATION §4g).
    pub advisory_gate: Option<String>,
    /// Live market regime at the decision bar (side-channel for the UMAP
    /// view's coloring — travels in file metadata, not the Arrow schema).
    pub regime: Option<String>,
}

impl DecisionCtx {
    /// The overwhelmingly common case: janus decided not to act.
    pub fn hold(regime: Option<String>) -> Self {
        Self {
            action: janus_core::SignalType::Hold,
            confidence: 0.0,
            blocked: None,
            advisory_gate: None,
            regime,
        }
    }
}

/// One completed experience row — the frozen 10-field Arrow record (§3)
/// plus the side-channel context that travels in file metadata.
#[derive(Debug)]
pub struct ExperienceRow {
    pub state_gaf: Vec<f32>,
    pub state_raw: Option<Vec<f32>>,
    pub action_type: u8,
    pub action_symbol: String,
    pub action_qty: f32,
    /// The NET reward (fee-adjusted, σ-normalized) — the frozen scalar the
    /// replay buffer and TD target consume, bit-identical to Phase 2.5.
    pub reward: f32,
    /// The PRE-FEE reward under the same normalization as `reward`
    /// (`reward ≈ reward_gross - fee_sigma`). Travels the `rows_meta`
    /// side-channel into the Qdrant payload so a verdict script can separate
    /// signal from the fixed fee cost. Not an Arrow schema column — the frozen
    /// 10-field schema is untouched.
    pub reward_gross: f32,
    /// The fee constant actually subtracted, in the same σ-normalized units as
    /// `reward`/`reward_gross`. `0.0` for Hold and for a zero-fee config.
    pub fee_sigma: f32,
    pub next_state_gaf: Vec<f32>,
    pub next_state_raw: Option<Vec<f32>>,
    pub done: bool,
    pub timestamp_ms: i64,
    // Side-channel (file metadata, NOT schema fields — §3):
    pub interval: String,
    pub confidence: f64,
    pub blocked: Option<String>,
    /// The advisory 9-gate verdict (`pass` / `block_...`), independent of the
    /// `blocked` exclusion reason — see [`DecisionCtx::advisory_gate`].
    pub advisory_gate: Option<String>,
    pub regime: Option<String>,
}

struct Pending {
    close_time_ms: i64,
    close: f64,
    state_gaf: Vec<f32>,
    ctx: DecisionCtx,
    interval: String,
    symbol: String,
    /// Bars elapsed since this decision was armed. Realized at `horizon`.
    bars_elapsed: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Recorder
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an interval string like "1m" / "5m" / "1h" / "1d" to seconds.
/// Unknown formats fall back to 60s (the deployment's base interval) — only
/// the gap heuristics depend on this, not record correctness.
pub fn interval_secs(interval: &str) -> i64 {
    let s = interval.trim();
    if s.len() < 2 {
        return 60;
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let Ok(n) = num.parse::<i64>() else {
        return 60;
    };
    if n <= 0 {
        return 60;
    }
    match unit {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        "d" => n * 86_400,
        "w" => n * 604_800,
        _ => 60,
    }
}

/// Per-`SYMBOL:interval` close windows + pending-decision queues. Owned by
/// the live-loop task; all methods are synchronous and allocation-light.
///
/// With `reward.horizon > 1`, up to `horizon` decisions per key are in flight
/// at once (each realizes `horizon` bars after it was armed), so the queue is
/// a `VecDeque` — one completed row is still emitted per bar per key once the
/// pipeline is past its initial horizon fill.
pub struct ExperienceRecorder {
    windows: HashMap<String, VecDeque<f32>>,
    pending: HashMap<String, VecDeque<Pending>>,
    /// Last observed close time per key, for detecting stream gaps.
    last_close_ms: HashMap<String, i64>,
    tx: mpsc::Sender<ExperienceRow>,
    scratch: Vec<f32>,
    reward: RewardConfig,
}

impl ExperienceRecorder {
    /// Recorder with the frozen Phase-1 reward (single-bar log return). Kept
    /// for tests and any caller wanting the original behavior.
    pub fn new(tx: mpsc::Sender<ExperienceRow>) -> Self {
        Self::with_reward(tx, RewardConfig::legacy())
    }

    /// Recorder with an explicit reward scheme (Phase 2.5). Production passes
    /// [`RewardConfig::from_env`].
    pub fn with_reward(tx: mpsc::Sender<ExperienceRow>, reward: RewardConfig) -> Self {
        Self {
            windows: HashMap::new(),
            pending: HashMap::new(),
            last_close_ms: HashMap::new(),
            tx,
            scratch: Vec::with_capacity(GAF_WINDOW),
            reward,
        }
    }

    /// Observe one closed candle for `key` (= "SYMBOL:interval"). Advances the
    /// pending decisions for the key: any that reached `reward.horizon` bars
    /// realizes its reward (the fee-aware, vol-normalized holding return, §7),
    /// and a fresh decision is armed with `ctx`, once the GAF window is warm.
    /// A stream gap past the done floor closes all open decisions early
    /// (`done = true`); a gap past the drop factor discards them. Never blocks:
    /// rows are `try_send`-ed to the writer.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        key: &str,
        symbol: &str,
        interval: &str,
        close_time_ms: i64,
        close: f64,
        ctx: DecisionCtx,
    ) {
        if !close.is_finite() || close <= 0.0 {
            return;
        }
        let window = self.windows.entry(key.to_string()).or_default();
        window.push_back(close as f32);
        if window.len() > GAF_WINDOW {
            window.pop_front();
        }
        // gaf_flat needs a contiguous slice; reuse one scratch buffer.
        self.scratch.clear();
        self.scratch.extend(window.iter().copied());
        // Warm only once the window is FULL — a full window keeps every
        // stored vector the product of the same window length (stable
        // distribution for training/UMAP), and 64 bars pass quickly.
        let gaf_now = if self.scratch.len() >= GAF_WINDOW {
            common::gaf_flat(&self.scratch, GAF_K)
        } else {
            Vec::new()
        };
        // Per-bar σ for risk normalization (computed once, shared by every
        // decision realized on this bar).
        let sigma1 = rolling_sigma(window);

        let ivl_secs = interval_secs(interval);
        let prev_close_ms = self.last_close_ms.insert(key.to_string(), close_time_ms);
        let step_gap_ms = prev_close_ms.map(|t| close_time_ms.saturating_sub(t));
        let drop_ms = ivl_secs.saturating_mul(PENDING_DROP_FACTOR) * 1000;
        let done_ms = (2 * ivl_secs).max(PENDING_DONE_FLOOR_SECS) * 1000;

        let queue = self.pending.entry(key.to_string()).or_default();

        // A large gap between consecutive closes breaks the holding window:
        // drop everything in flight (the return across the gap is meaningless).
        if step_gap_ms.is_some_and(|g| g > drop_ms) {
            if !queue.is_empty() {
                metrics().pending_gap_dropped.inc_by(queue.len() as u64);
                queue.clear();
            }
        } else {
            // A moderate gap is an episode boundary: realize every open
            // decision now with done = true. Otherwise, realize only those
            // that have reached the horizon (done = false, continuous).
            let episode_end = step_gap_ms.is_some_and(|g| g > done_ms);
            for p in queue.iter_mut() {
                p.bars_elapsed += 1;
            }
            // Front of the queue is the oldest decision (largest bars_elapsed).
            while let Some(front) = queue.front() {
                let mature = front.bars_elapsed >= self.reward.horizon;
                if !(episode_end || mature) {
                    break;
                }
                // Once warm the window never goes cold; guard anyway — without a
                // next-state we can't emit a valid transition, so drop it.
                let Some(p) = queue.pop_front() else { break };
                if gaf_now.is_empty() {
                    continue;
                }
                let reward = compute_reward(
                    p.ctx.action,
                    p.close,
                    close,
                    p.bars_elapsed,
                    sigma1,
                    &self.reward,
                );
                let row = ExperienceRow {
                    state_gaf: p.state_gaf,
                    state_raw: None,
                    action_type: p.ctx.action.to_u8(),
                    action_symbol: p.symbol,
                    action_qty: 0.0,
                    // `reward` stays the NET scalar (frozen). The decomposition
                    // rides the side-channel; the replay buffer is untouched.
                    reward: reward.net,
                    reward_gross: reward.gross,
                    fee_sigma: reward.fee_sigma,
                    next_state_gaf: gaf_now.clone(),
                    next_state_raw: None,
                    done: episode_end,
                    timestamp_ms: p.close_time_ms,
                    interval: p.interval,
                    confidence: p.ctx.confidence,
                    regime: p.ctx.regime,
                    blocked: p.ctx.blocked,
                    advisory_gate: p.ctx.advisory_gate,
                };
                match self.tx.try_send(row) {
                    Ok(()) => metrics().rows_recorded.inc(),
                    Err(_) => metrics().rows_dropped_channel.inc(),
                }
            }
        }

        // Arm the new decision once warm.
        if !gaf_now.is_empty() {
            self.pending
                .entry(key.to_string())
                .or_default()
                .push_back(Pending {
                    close_time_ms,
                    close,
                    state_gaf: gaf_now,
                    interval: interval.to_string(),
                    ctx,
                    symbol: symbol.to_string(),
                    bars_elapsed: 0,
                });
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Writer
// ─────────────────────────────────────────────────────────────────────────────

/// The doorbell payload backward's intake parses
/// (`services/backward/src/worker.rs::IngestJob` — field names must match;
/// the integration test exercises the real consumer against this producer).
#[derive(Debug, Serialize)]
struct IngestJobMsg {
    batch_id: String,
    shm_path: String,
}

/// Spawn the spool-writer task. `redis` is the doorbell transport — `None`
/// (tests / Redis unavailable at boot) means files rely on backward's sweep,
/// which is the designed degraded mode (§4.3).
pub fn spawn_writer(
    cfg: ExperienceConfig,
    redis: Option<redis::Client>,
) -> (mpsc::Sender<ExperienceRow>, tokio::task::JoinHandle<()>) {
    let (tx, mut rx) = mpsc::channel::<ExperienceRow>(CHANNEL_CAPACITY);
    let task = tokio::spawn(async move {
        if let Err(e) = std::fs::create_dir_all(&cfg.spool_dir) {
            error!(
                dir = %cfg.spool_dir.display(),
                "experience writer: cannot create spool dir ({e}) — writer exiting; \
                 rows will be dropped via the full channel"
            );
            return;
        }
        info!(
            dir = %cfg.spool_dir.display(),
            batch_rows = cfg.batch_rows,
            flush_secs = cfg.flush_secs,
            spool_max_mb = cfg.spool_max_mb,
            "experience writer started"
        );
        let mut buf: Vec<ExperienceRow> = Vec::with_capacity(cfg.batch_rows.max(1));
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(cfg.flush_secs.max(1)));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                row = rx.recv() => match row {
                    Some(r) => {
                        buf.push(r);
                        if buf.len() >= cfg.batch_rows {
                            flush(&cfg, &redis, &mut buf).await;
                        }
                    }
                    None => {
                        // All senders dropped (loop exited / shutdown).
                        flush(&cfg, &redis, &mut buf).await;
                        info!("experience writer: channel closed — exiting");
                        break;
                    }
                },
                _ = tick.tick() => {
                    flush(&cfg, &redis, &mut buf).await;
                }
            }
        }
    });
    (tx, task)
}

/// Flush the buffered rows as one complete Arrow IPC batch (§4.2): bound the
/// spool, write `.tmp`, fsync, rename, ring the doorbell. Failures drop this
/// batch (counted) — experiences are droppable; trading is not.
async fn flush(
    cfg: &ExperienceConfig,
    redis: &Option<redis::Client>,
    buf: &mut Vec<ExperienceRow>,
) {
    if buf.is_empty() {
        return;
    }
    let rows = std::mem::take(buf);
    enforce_spool_bound(cfg);

    let batch_id = format!(
        "{}-{}",
        chrono::Utc::now().timestamp_millis(),
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    );
    let tmp_path = cfg.spool_dir.join(format!("{batch_id}.arrow.tmp"));
    let final_path = cfg.spool_dir.join(format!("{batch_id}.arrow"));

    // Files are ≤ a few hundred KB on tmpfs — a synchronous write here (in
    // the detached writer task, not the live loop) is faster than a
    // spawn_blocking round-trip.
    let row_count = rows.len();
    if let Err(e) = write_ipc(&tmp_path, &rows) {
        metrics().batches_failed.inc();
        warn!(
            batch_id,
            rows = row_count,
            "experience batch spool failed: {e:#}"
        );
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
        metrics().batches_failed.inc();
        warn!(batch_id, "experience batch rename failed: {e}");
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }
    metrics().batches_spooled.inc();
    tracing::debug!(batch_id, rows = row_count, "experience batch spooled");

    // Doorbell (best-effort; the sweep is the retry).
    if let Some(client) = redis {
        let msg = IngestJobMsg {
            batch_id: batch_id.clone(),
            shm_path: final_path.display().to_string(),
        };
        let payload = match serde_json::to_string(&msg) {
            Ok(p) => p,
            Err(e) => {
                warn!(batch_id, "experience doorbell serialize failed: {e}");
                return;
            }
        };
        let push = async {
            use redis::AsyncCommands;
            let mut conn = client.get_multiplexed_async_connection().await?;
            conn.lpush::<_, _, ()>(REDIS_QUEUE_KEY, payload).await?;
            Ok::<(), redis::RedisError>(())
        };
        match tokio::time::timeout(std::time::Duration::from_secs(5), push).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                metrics().redis_push_failures.inc();
                warn!(
                    batch_id,
                    "experience doorbell LPUSH failed ({e}) — sweep will recover"
                );
            }
            Err(_) => {
                metrics().redis_push_failures.inc();
                warn!(
                    batch_id,
                    "experience doorbell LPUSH timed out — sweep will recover"
                );
            }
        }
    }
}

/// Local copy of backward's frozen `expected_schema()`
/// (`services/backward/src/tasks/ingest.rs`). The integration test runs this
/// producer against the real consumer, so drift fails CI rather than
/// corrupting ingestion.
fn experience_schema(meta: HashMap<String, String>) -> Schema {
    Schema::new_with_metadata(
        vec![
            Field::new("state_gaf", DataType::Binary, false),
            Field::new("state_raw", DataType::Binary, true),
            Field::new("action_type", DataType::UInt8, false),
            Field::new("action_symbol", DataType::Utf8, false),
            Field::new("action_qty", DataType::Float32, false),
            Field::new("reward", DataType::Float32, false),
            Field::new("next_state_gaf", DataType::Binary, false),
            Field::new("next_state_raw", DataType::Binary, true),
            Field::new("done", DataType::Boolean, false),
            Field::new("timestamp_ms", DataType::Int64, false),
        ],
        meta,
    )
}

fn f32s_to_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Row-level side-channel entry for the file-metadata JSON (§3): the fields
/// the UMAP view wants that don't fit the frozen schema.
#[derive(Serialize)]
struct RowMeta<'a> {
    i: usize,
    interval: &'a str,
    confidence: f64,
    /// Reward decomposition (§4g). Always emitted for new rows; backward parses
    /// them as `Option`, so their absence on older files reads as "unknown"
    /// (never a fabricated 0.0 gross).
    reward_gross: f32,
    fee_sigma: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocked: Option<&'a str>,
    /// Advisory 9-gate verdict — kept out of the JSON when absent, so files
    /// from older producers and the backward consumer's `RowSideMeta` (which
    /// ignores unknown keys) are unaffected.
    #[serde(skip_serializing_if = "Option::is_none")]
    advisory_gate: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    regime: Option<&'a str>,
}

fn write_ipc(path: &std::path::Path, rows: &[ExperienceRow]) -> anyhow::Result<()> {
    let row_meta: Vec<RowMeta<'_>> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| RowMeta {
            i,
            interval: &r.interval,
            confidence: r.confidence,
            reward_gross: r.reward_gross,
            fee_sigma: r.fee_sigma,
            blocked: r.blocked.as_deref(),
            advisory_gate: r.advisory_gate.as_deref(),
            regime: r.regime.as_deref(),
        })
        .collect();
    let mut meta = HashMap::new();
    meta.insert("schema_version".to_string(), "1".to_string());
    meta.insert("producer".to_string(), "janus-forward".to_string());
    meta.insert("rows_meta".to_string(), serde_json::to_string(&row_meta)?);
    let schema = Arc::new(experience_schema(meta));

    let state_gaf: Vec<Vec<u8>> = rows
        .iter()
        .map(|r| f32s_to_le_bytes(&r.state_gaf))
        .collect();
    let next_gaf: Vec<Vec<u8>> = rows
        .iter()
        .map(|r| f32s_to_le_bytes(&r.next_state_gaf))
        .collect();
    let state_raw: Vec<Option<Vec<u8>>> = rows
        .iter()
        .map(|r| r.state_raw.as_deref().map(f32s_to_le_bytes))
        .collect();
    let next_raw: Vec<Option<Vec<u8>>> = rows
        .iter()
        .map(|r| r.next_state_raw.as_deref().map(f32s_to_le_bytes))
        .collect();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(BinaryArray::from_iter_values(state_gaf.iter())),
            Arc::new(BinaryArray::from_iter(state_raw)),
            Arc::new(UInt8Array::from_iter_values(
                rows.iter().map(|r| r.action_type),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.action_symbol.as_str()),
            )),
            Arc::new(Float32Array::from_iter_values(
                rows.iter().map(|r| r.action_qty),
            )),
            Arc::new(Float32Array::from_iter_values(
                rows.iter().map(|r| r.reward),
            )),
            Arc::new(BinaryArray::from_iter_values(next_gaf.iter())),
            Arc::new(BinaryArray::from_iter(next_raw)),
            Arc::new(BooleanArray::from_iter(rows.iter().map(|r| Some(r.done)))),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|r| r.timestamp_ms),
            )),
        ],
    )?;

    let file = File::create(path)?;
    let mut writer = FileWriter::try_new(file, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    // Atomic-rename protocol (§4.2): make the finished file durable before
    // it becomes visible under its final name.
    writer.into_inner()?.sync_all()?;
    Ok(())
}

/// Drop-oldest spool bound (§4.2): experiences must never fill /dev/shm.
/// Updates the spool-bytes gauge as a side effect.
fn enforce_spool_bound(cfg: &ExperienceConfig) {
    let max_bytes = cfg.spool_max_mb.saturating_mul(1024 * 1024);
    let mut entries: Vec<(std::path::PathBuf, u64)> = Vec::new();
    let mut total: u64 = 0;
    let Ok(dir) = std::fs::read_dir(&cfg.spool_dir) else {
        return;
    };
    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("arrow")
            && let Ok(md) = entry.metadata()
        {
            total += md.len();
            entries.push((path, md.len()));
        }
    }
    if total > max_bytes {
        // batch_id is <unix_ms>-<uuid>, so lexicographic order = age order.
        entries.sort();
        for (path, len) in entries {
            if total <= max_bytes {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(len);
                metrics().spool_drop_oldest.inc();
                warn!(path = %path.display(), "spool over bound — dropped oldest batch");
            }
        }
    }
    metrics().spool_bytes.set(total as i64);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::SignalType;

    const MIN: i64 = 60_000; // 1m in ms

    fn recorder() -> (ExperienceRecorder, mpsc::Receiver<ExperienceRow>) {
        let (tx, rx) = mpsc::channel(64);
        (ExperienceRecorder::new(tx), rx)
    }

    /// Drive `n` warmup candles through the recorder (all Holds).
    fn warm(rec: &mut ExperienceRecorder, key: &str, n: usize, start_ms: i64) -> i64 {
        let mut t = start_ms;
        for i in 0..n {
            rec.observe(
                key,
                "BTCUSDT",
                "1m",
                t,
                100.0 + (i as f64 % 7.0),
                DecisionCtx::hold(None),
            );
            t += MIN;
        }
        t
    }

    #[test]
    fn interval_parse() {
        assert_eq!(interval_secs("1m"), 60);
        assert_eq!(interval_secs("5m"), 300);
        assert_eq!(interval_secs("1h"), 3600);
        assert_eq!(interval_secs("1d"), 86_400);
        assert_eq!(interval_secs("bogus"), 60);
    }

    #[test]
    fn no_rows_before_window_warm() {
        let (mut rec, mut rx) = recorder();
        warm(&mut rec, "BTCUSDT:1m", GAF_WINDOW - 1, 0);
        assert!(rx.try_recv().is_err(), "cold window must not emit rows");
    }

    #[test]
    fn first_row_after_warmup_plus_one() {
        let (mut rec, mut rx) = recorder();
        // Bar GAF_WINDOW arms the first pending; bar GAF_WINDOW+1 completes it.
        warm(&mut rec, "BTCUSDT:1m", GAF_WINDOW + 1, 0);
        let row = rx.try_recv().expect("one completed row");
        assert_eq!(row.action_type, SignalType::Hold.to_u8());
        assert_eq!(row.state_gaf.len(), GAF_K * GAF_K);
        assert_eq!(row.next_state_gaf.len(), GAF_K * GAF_K);
        assert!(!row.done);
        assert_eq!(row.reward, 0.0, "hold reward is zero");
        assert!(row.state_raw.is_none());
    }

    #[test]
    fn buy_reward_sign_follows_price() {
        let (mut rec, mut rx) = recorder();
        let t = warm(&mut rec, "K:1m", GAF_WINDOW, 0);
        // Arm a Buy at close=100, complete at close=110 → positive reward.
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t,
            100.0,
            DecisionCtx {
                action: SignalType::Buy,
                confidence: 0.9,
                blocked: None,
                advisory_gate: None,
                regime: None,
            },
        );
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t + MIN,
            110.0,
            DecisionCtx::hold(None),
        );
        // Drain: the warmup itself completed pendings too — take the last row.
        let mut last = None;
        while let Ok(r) = rx.try_recv() {
            last = Some(r);
        }
        let row = last.expect("rows emitted");
        assert_eq!(row.action_type, SignalType::Buy.to_u8());
        assert!(
            (row.reward - (110.0f64 / 100.0).ln() as f32).abs() < 1e-6,
            "buy reward is the log return: {}",
            row.reward
        );
        assert_eq!(row.timestamp_ms, t, "timestamp is the DECISION bar");
        assert_eq!(row.confidence, 0.9);
    }

    #[test]
    fn sell_reward_sign_inverts() {
        let (mut rec, mut rx) = recorder();
        let t = warm(&mut rec, "K:1m", GAF_WINDOW, 0);
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t,
            100.0,
            DecisionCtx {
                action: SignalType::Sell,
                confidence: 0.8,
                blocked: Some("gate: test".to_string()),
                advisory_gate: None,
                regime: None,
            },
        );
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t + MIN,
            110.0,
            DecisionCtx::hold(None),
        );
        let mut last = None;
        while let Ok(r) = rx.try_recv() {
            last = Some(r);
        }
        let row = last.expect("rows emitted");
        assert_eq!(row.action_type, SignalType::Sell.to_u8());
        assert!(row.reward < 0.0, "sell into a rally is negative reward");
        assert_eq!(row.blocked.as_deref(), Some("gate: test"));
    }

    #[test]
    fn moderate_gap_completes_with_done() {
        let (mut rec, mut rx) = recorder();
        let t = warm(&mut rec, "K:1m", GAF_WINDOW, 0);
        // Next candle 8 minutes later: > max(2×1m, 5min) → done, ≤ 10×1m... no:
        // 8min > 10×1m? 10×1m = 10min, so 8min completes with done=true.
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t + 8 * MIN,
            101.0,
            DecisionCtx::hold(None),
        );
        let mut last = None;
        while let Ok(r) = rx.try_recv() {
            last = Some(r);
        }
        assert!(
            last.expect("row emitted").done,
            "gap past the done floor sets done"
        );
    }

    #[test]
    fn oversized_gap_drops_pending() {
        let (mut rec, mut rx) = recorder();
        let t = warm(&mut rec, "K:1m", GAF_WINDOW, 0);
        while rx.try_recv().is_ok() {}
        // 11 minutes > 10 × 1m → dropped, no row.
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t + 11 * MIN,
            101.0,
            DecisionCtx::hold(None),
        );
        assert!(
            rx.try_recv().is_err(),
            "oversized gap must drop the pending"
        );
    }

    #[test]
    fn keys_are_independent() {
        let (mut rec, mut rx) = recorder();
        // Interleave 1m and 5m for the same symbol — separate windows/pendings.
        let mut t = 0;
        for i in 0..(GAF_WINDOW + 1) {
            rec.observe(
                "BTCUSDT:1m",
                "BTCUSDT",
                "1m",
                t,
                100.0 + i as f64,
                DecisionCtx::hold(None),
            );
            rec.observe(
                "BTCUSDT:5m",
                "BTCUSDT",
                "5m",
                t * 5,
                200.0 + i as f64,
                DecisionCtx::hold(None),
            );
            t += MIN;
        }
        let mut count = 0;
        while rx.try_recv().is_ok() {
            count += 1;
        }
        assert_eq!(count, 2, "one completed row per key");
    }

    // ── Phase 2.5 reward scheme ──────────────────────────────────────────────

    fn recorder_with(cfg: RewardConfig) -> (ExperienceRecorder, mpsc::Receiver<ExperienceRow>) {
        let (tx, rx) = mpsc::channel(256);
        (ExperienceRecorder::with_reward(tx, cfg), rx)
    }

    #[test]
    fn legacy_config_matches_frozen_phase1() {
        // `new` must be exactly the legacy single-bar scheme.
        let cfg = RewardConfig::legacy();
        assert_eq!(cfg.horizon, 1);
        assert_eq!(cfg.fee, 0.0);
        assert!(!cfg.vol_norm);
        // compute_reward under legacy == signed single-bar log return.
        let r = compute_reward(SignalType::Buy, 100.0, 110.0, 1, Some(0.01), &cfg).net;
        assert!((r - (110.0f64 / 100.0).ln() as f32).abs() < 1e-6);
    }

    #[test]
    fn rolling_sigma_of_known_series() {
        // Flat series → zero σ.
        let flat: VecDeque<f32> = std::iter::repeat_n(100.0, 8).collect();
        assert_eq!(rolling_sigma(&flat), Some(0.0));
        // Too few points → None.
        let short: VecDeque<f32> = [100.0, 101.0].into_iter().collect();
        assert_eq!(rolling_sigma(&short), None);
        // A varying series → strictly positive σ.
        let vary: VecDeque<f32> = [100.0, 102.0, 101.0, 104.0, 103.0].into_iter().collect();
        assert!(rolling_sigma(&vary).unwrap() > 0.0);
    }

    #[test]
    fn fee_is_subtracted_from_directional_reward_only() {
        let cfg = RewardConfig {
            horizon: 1,
            fee: 0.001,
            vol_norm: false,
            vol_floor: 1e-4,
        };
        let buy = compute_reward(SignalType::Buy, 100.0, 110.0, 1, None, &cfg).net;
        assert!((buy - ((110.0f64 / 100.0).ln() - 0.001) as f32).abs() < 1e-6);
        // Hold pays no fee and earns nothing.
        assert_eq!(
            compute_reward(SignalType::Hold, 100.0, 110.0, 1, None, &cfg).net,
            0.0
        );
        // A losing Sell still pays the fee (fee is a cost, not a sign flip).
        let sell = compute_reward(SignalType::Sell, 100.0, 110.0, 1, None, &cfg).net;
        assert!(sell < 0.0);
        assert!((sell - (-(110.0f64 / 100.0).ln() - 0.001) as f32).abs() < 1e-6);
    }

    #[test]
    fn vol_norm_divides_by_sigma_scaled_by_horizon() {
        let cfg = RewardConfig {
            horizon: 4,
            fee: 0.0,
            vol_norm: true,
            vol_floor: 1e-6,
        };
        // sigma_h = 0.01 * sqrt(4) = 0.02 ⇒ reward = ln(1.1)/0.02.
        let r = compute_reward(SignalType::Buy, 100.0, 110.0, 4, Some(0.01), &cfg).net;
        let expected = ((110.0f64 / 100.0).ln() / (0.01 * 4.0_f64.sqrt())) as f32;
        assert!((r - expected).abs() < 1e-4, "got {r}, want {expected}");
    }

    #[test]
    fn vol_floor_bounds_a_flat_window() {
        let cfg = RewardConfig {
            horizon: 1,
            fee: 0.0,
            vol_norm: true,
            vol_floor: 0.01,
        };
        // σ=0 (flat) must fall back to the floor, keeping the reward finite.
        let r = compute_reward(SignalType::Buy, 100.0, 110.0, 1, Some(0.0), &cfg).net;
        let expected = ((110.0f64 / 100.0).ln() / 0.01) as f32;
        assert!(r.is_finite());
        assert!((r - expected).abs() < 1e-4);
    }

    #[test]
    fn degenerate_vol_floor_never_yields_non_finite_reward() {
        // A pathological config (near-zero floor) + a flat window must NOT inject
        // ±inf/NaN into the replay buffer — the finite guard falls back to 0.0.
        let cfg = RewardConfig {
            horizon: 1,
            fee: 0.0,
            vol_norm: true,
            vol_floor: 0.0, // as if an operator set VOL_FLOOR=0
        };
        let parts = compute_reward(SignalType::Buy, 100.0, 110.0, 1, Some(0.0), &cfg);
        assert!(
            parts.net.is_finite(),
            "net must stay finite, got {}",
            parts.net
        );
        // The decomposition scalars share the same finite guard — a degenerate
        // config must not push ±inf/NaN into the Qdrant payload either.
        assert!(
            parts.gross.is_finite() && parts.fee_sigma.is_finite(),
            "gross/fee_sigma must stay finite, got {} / {}",
            parts.gross,
            parts.fee_sigma
        );
        // Also verify from_env clamps to a real floor (>= 1e-8), never ~0.
        let parsed = RewardConfig::from_env();
        assert!(
            parsed.vol_floor >= 1e-8,
            "vol_floor must clamp to a real floor"
        );
    }

    /// ZERO-CHANGE PROOF — the pivot the whole measurement window rests on.
    /// The NET reward out of the decomposing `compute_reward` must be *bit-for-
    /// bit* the scalar the frozen pre-decomposition formula produced. `net`
    /// never derives from `gross - fee_sigma`; it walks the identical operation
    /// sequence. If this ever fails, instrumentation became a regime change —
    /// STOP. Grid: both directional signs × σ straddling the vol floor (incl.
    /// the degenerate σ=0 / None) × bars_held 1/5/20 × entry==exit and extreme
    /// prices × legacy/production/degenerate configs.
    #[test]
    fn net_reward_is_bit_identical_to_frozen_formula() {
        // The frozen formula, transcribed verbatim from the pre-decomposition
        // code, kept local so a future edit to `compute_reward` is checked
        // against this fixed reference instead of against itself.
        fn net_frozen(
            action: SignalType,
            entry: f64,
            exit: f64,
            bars: usize,
            sigma1: Option<f64>,
            cfg: &RewardConfig,
        ) -> f32 {
            let dir = match action {
                SignalType::Buy => 1.0_f64,
                SignalType::Sell => -1.0_f64,
                _ => return 0.0,
            };
            let mut r = dir * (exit / entry).ln();
            r -= cfg.fee;
            if cfg.vol_norm {
                let n = bars.max(1) as f64;
                let sigma_h = sigma1.unwrap_or(0.0).max(cfg.vol_floor) * n.sqrt();
                r /= sigma_h;
            }
            let out = r as f32;
            if out.is_finite() { out } else { 0.0 }
        }

        let actions = [SignalType::Buy, SignalType::Sell, SignalType::Hold];
        let prices: [(f64, f64); 5] = [
            (100.0, 110.0),         // up
            (100.0, 90.0),          // down
            (100.0, 100.0),         // entry == exit → ln(1) = 0
            (1.0, 1_000_000_000.0), // extreme up
            (50_000.0, 49_999.0),   // tiny move
        ];
        let bars_grid = [1usize, 5, 20];
        // σ straddling the vol floor, plus the flat-window None and Some(0.0)
        // degenerate cases that exercise the finite guard.
        let sigmas = [
            None,
            Some(0.0),
            Some(1e-6),
            Some(1e-4),
            Some(0.02),
            Some(1.0),
        ];
        let cfgs = [
            RewardConfig::legacy(),
            RewardConfig {
                horizon: 5,
                fee: 0.0008,
                vol_norm: true,
                vol_floor: 1e-4,
            },
            RewardConfig {
                horizon: 5,
                fee: 0.0008,
                vol_norm: false,
                vol_floor: 1e-4,
            },
            RewardConfig {
                horizon: 1,
                fee: 0.0,
                vol_norm: true,
                vol_floor: 0.0, // degenerate floor → exercises the guard
            },
            RewardConfig {
                horizon: 20,
                fee: 0.002,
                vol_norm: true,
                vol_floor: 1e-8,
            },
        ];

        for &action in &actions {
            for &(entry, exit) in &prices {
                for &bars in &bars_grid {
                    for &sigma in &sigmas {
                        for cfg in &cfgs {
                            let got = compute_reward(action, entry, exit, bars, sigma, cfg);
                            let want = net_frozen(action, entry, exit, bars, sigma, cfg);
                            // Bit-for-bit: compare raw f32 bits so a reordered op
                            // or a NaN-guard slip is caught exactly, not within a
                            // tolerance that could hide a one-ULP regime shift.
                            assert_eq!(
                                got.net.to_bits(),
                                want.to_bits(),
                                "NET drifted: action={action:?} entry={entry} exit={exit} \
                                 bars={bars} sigma={sigma:?} cfg={cfg:?} — got {} want {}",
                                got.net,
                                want
                            );
                        }
                    }
                }
            }
        }
    }

    /// The decomposition a verdict script will read: `gross` is the pre-fee
    /// reward under the same normalization, `fee_sigma` is that fee in σ units,
    /// and together they reconstruct `net` (§4g: this is exactly what lets "no
    /// signal" be told apart from "8 bps ate a small edge").
    #[test]
    fn decomposition_separates_gross_signal_from_fee_cost() {
        // 8 bps fee, σ-normalized over a 5-bar horizon — the production shape
        // that made the J1 headline fee-dominated.
        let cfg = RewardConfig {
            horizon: 5,
            fee: 0.0008,
            vol_norm: true,
            vol_floor: 1e-4,
        };
        let sigma1 = 0.01;
        let sigma_h = sigma1 * 5.0_f64.sqrt();
        let parts = compute_reward(SignalType::Buy, 100.0, 110.0, 5, Some(sigma1), &cfg);

        // gross = the pre-fee σ-normalized return (no fee term subtracted).
        let want_gross = ((110.0f64 / 100.0).ln() / sigma_h) as f32;
        assert!(
            (parts.gross - want_gross).abs() < 1e-4,
            "gross is the pre-fee σ-normalized return: got {} want {want_gross}",
            parts.gross
        );
        // fee_sigma = the fee constant expressed in the same σ units.
        let want_fee = (0.0008f64 / sigma_h) as f32;
        assert!(
            (parts.fee_sigma - want_fee).abs() < 1e-4,
            "fee_sigma is fee/sigma_h: got {} want {want_fee}",
            parts.fee_sigma
        );
        // net reconstructs from the decomposition (up to f32 rounding).
        assert!(
            (parts.net - (parts.gross - parts.fee_sigma)).abs() < 1e-4,
            "net ≈ gross - fee_sigma: net={} gross={} fee_sigma={}",
            parts.net,
            parts.gross,
            parts.fee_sigma
        );
        // The fee is a real drag: gross strictly exceeds net.
        assert!(parts.gross > parts.net);

        // Sell mirrors the sign on gross, but the fee cost stays positive.
        let sell = compute_reward(SignalType::Sell, 100.0, 110.0, 5, Some(sigma1), &cfg);
        assert!(sell.gross < 0.0, "sell into a rally has negative gross");
        assert!(sell.fee_sigma > 0.0, "fee is a cost regardless of side");
        assert!((sell.fee_sigma - want_fee).abs() < 1e-4);

        // Hold is all zeros — no signal, no fee, nothing to decompose.
        let hold = compute_reward(SignalType::Hold, 100.0, 110.0, 5, Some(sigma1), &cfg);
        assert_eq!((hold.net, hold.gross, hold.fee_sigma), (0.0, 0.0, 0.0));
    }

    #[test]
    fn horizon_reward_spans_multiple_bars() {
        // horizon 3, no fee/norm ⇒ reward is the 3-bar holding return.
        let (mut rec, mut rx) = recorder_with(RewardConfig {
            horizon: 3,
            fee: 0.0,
            vol_norm: false,
            vol_floor: 1e-4,
        });
        let t = warm(&mut rec, "K:1m", GAF_WINDOW, 0);
        while rx.try_recv().is_ok() {} // drain warmup completions
        // Buy at close=100, then three more bars up to 130.
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t,
            100.0,
            DecisionCtx {
                action: SignalType::Buy,
                confidence: 0.9,
                blocked: None,
                advisory_gate: None,
                regime: None,
            },
        );
        for (i, px) in [110.0, 120.0, 130.0].into_iter().enumerate() {
            rec.observe(
                "K:1m",
                "BTCUSDT",
                "1m",
                t + (i as i64 + 1) * MIN,
                px,
                DecisionCtx::hold(None),
            );
        }
        // Find the Buy row (the one non-Hold decision).
        let mut buy = None;
        while let Ok(r) = rx.try_recv() {
            if r.action_type == SignalType::Buy.to_u8() {
                buy = Some(r);
            }
        }
        let row = buy.expect("buy row realized after horizon");
        assert!(
            (row.reward - (130.0f64 / 100.0).ln() as f32).abs() < 1e-6,
            "3-bar holding return ln(130/100), got {}",
            row.reward
        );
        assert_eq!(row.timestamp_ms, t, "reward attributed to the decision bar");
        assert!(
            !row.done,
            "natural horizon completion is not an episode end"
        );
    }

    #[test]
    fn horizon_pipeline_emits_one_row_per_bar_once_filled() {
        // With horizon 3, after the horizon fills, every bar realizes exactly
        // one decision (row volume is preserved, not divided by the horizon).
        let (mut rec, mut rx) = recorder_with(RewardConfig {
            horizon: 3,
            fee: 0.0,
            vol_norm: false,
            vol_floor: 1e-4,
        });
        let mut t = warm(&mut rec, "K:1m", GAF_WINDOW, 0);
        while rx.try_recv().is_ok() {}
        // Drive 6 more warm bars; after the horizon fills, expect ~one row each.
        for i in 0..6 {
            rec.observe(
                "K:1m",
                "BTCUSDT",
                "1m",
                t,
                100.0 + i as f64,
                DecisionCtx::hold(None),
            );
            t += MIN;
        }
        let count = std::iter::from_fn(|| rx.try_recv().ok()).count();
        assert!(
            count >= 4,
            "steady-state emits ~one row per bar, got {count}"
        );
    }

    #[test]
    fn episode_boundary_closes_all_open_horizon_decisions() {
        let (mut rec, mut rx) = recorder_with(RewardConfig {
            horizon: 5,
            fee: 0.0,
            vol_norm: false,
            vol_floor: 1e-4,
        });
        let t = warm(&mut rec, "K:1m", GAF_WINDOW, 0);
        while rx.try_recv().is_ok() {}
        // Arm two decisions one bar apart, both still inside the 5-bar horizon.
        rec.observe("K:1m", "BTCUSDT", "1m", t, 100.0, DecisionCtx::hold(None));
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t + MIN,
            101.0,
            DecisionCtx::hold(None),
        );
        while rx.try_recv().is_ok() {}
        // A 6-minute gap (> done floor, < drop factor) ends the episode: both
        // open decisions realize with done = true even though horizon < 5.
        rec.observe(
            "K:1m",
            "BTCUSDT",
            "1m",
            t + 7 * MIN,
            102.0,
            DecisionCtx::hold(None),
        );
        let rows: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(!rows.is_empty(), "episode boundary flushes open decisions");
        assert!(
            rows.iter().all(|r| r.done),
            "all flushed rows are episode ends"
        );
    }

    #[tokio::test]
    async fn writer_flushes_at_batch_rows_and_spools_valid_ipc() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = ExperienceConfig {
            enabled: true,
            spool_dir: dir.path().to_path_buf(),
            batch_rows: 4,
            flush_secs: 3600, // rely on the row trigger
            spool_max_mb: 16,
        };
        let (tx, task) = spawn_writer(cfg, None);
        for i in 0..4usize {
            tx.send(ExperienceRow {
                state_gaf: vec![0.1; 9],
                state_raw: None,
                action_type: (i % 4) as u8,
                action_symbol: "BTCUSDT".to_string(),
                action_qty: 0.0,
                reward: 0.01,
                reward_gross: 0.02,
                fee_sigma: 0.01,
                next_state_gaf: vec![0.2; 9],
                next_state_raw: None,
                done: false,
                timestamp_ms: 1_700_000_000_000 + i as i64,
                interval: "1m".to_string(),
                confidence: 0.5,
                blocked: None,
                advisory_gate: None,
                regime: None,
            })
            .await
            .expect("send row");
        }
        // Poll for the finished .arrow file (rename is the last step).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let path = loop {
            let found = std::fs::read_dir(dir.path())
                .unwrap()
                .flatten()
                .map(|e| e.path())
                .find(|p| p.extension().and_then(|e| e.to_str()) == Some("arrow"));
            if let Some(p) = found {
                break p;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "flush did not produce a file"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        };
        // Read it back with the arrow FileReader — footer must be complete.
        let file = File::open(&path).unwrap();
        let reader = arrow::ipc::reader::FileReader::try_new(file, None).unwrap();
        let schema = reader.schema();
        assert_eq!(schema.fields().len(), 10);
        assert_eq!(
            schema.metadata().get("schema_version").map(String::as_str),
            Some("1")
        );
        let rows: usize = reader.map(|b| b.unwrap().num_rows()).sum();
        assert_eq!(rows, 4);
        drop(tx);
        let _ = task.await;
    }

    #[test]
    fn spool_bound_drops_oldest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = ExperienceConfig {
            enabled: true,
            spool_dir: dir.path().to_path_buf(),
            batch_rows: 1,
            flush_secs: 1,
            spool_max_mb: 0, // bound of zero: everything over
        };
        std::fs::write(dir.path().join("100-aaaa.arrow"), vec![0u8; 1024]).unwrap();
        std::fs::write(dir.path().join("200-bbbb.arrow"), vec![0u8; 1024]).unwrap();
        enforce_spool_bound(&cfg);
        assert!(
            !dir.path().join("100-aaaa.arrow").exists(),
            "oldest dropped"
        );
        // With max 0 both go; the point is oldest-first deletion ordering.
        assert!(!dir.path().join("200-bbbb.arrow").exists());
    }
}
