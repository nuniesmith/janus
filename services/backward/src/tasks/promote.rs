//! Challenger → champion promotion (SAFE, explicitly-invoked, guarded).
//!
//! # DESIGN
//!
//! The scheduler trains a *challenger* into
//! `checkpoints/backward/challenger/latest_model.bin` and records an
//! `eval_winrate` per session (see [`crate::tasks::train::run_training_session`]).
//! The LIVE *champion* at `checkpoints/backward/latest_model.bin` is what forward
//! inference hot-loads and trades on. Promotion **overwrites the live champion**,
//! so this module is deliberately isolated from every automatic code path and
//! every step is guarded.
//!
//! ## What promotion is (and is NOT)
//!
//! * It is a SEPARATE, explicitly-invoked path: the dedicated function
//!   [`promote_challenger`] plus the env-gated one-shot [`run_promotion_if_enabled`],
//!   reachable only via the `janus-backward promote` subcommand.
//! * It is NEVER wired into the periodic training scheduler
//!   ([`crate::tasks::train::run_training_scheduler`]) and can NEVER auto-fire:
//!   nothing calls it on a timer, on startup, or from any worker loop.
//!
//! ## The gate — ALL of these must hold to promote
//!
//! 1. **Explicit enable flag.** `JANUS_PROMOTE_ENABLED` must be truthy
//!    (`1/true/yes/on`). Absent/anything-else ⇒ no promotion. Deploying this
//!    module changes nothing until an operator opts in AND runs the subcommand.
//! 2. **Sustained out-performance over N consecutive sessions.** The challenger's
//!    most recent `min_sessions` recorded `eval_winrate`s must EACH clear a bar:
//!    * If a champion eval history is available, the bar is
//!      `champion_baseline + margin`.
//!    * **If champion eval history is NOT available** — which is the case today,
//!      because the champion is never trained or evaluated by the scheduler and
//!      so no `checkpoints/backward/eval_history.jsonl` is ever written — the bar
//!      is the absolute threshold `min_winrate`. This absolute-threshold fallback
//!      is the ACTIVE path in production and is documented as such.
//!    * Every one of the N recent sessions must also have `eval_samples > 0`; a
//!      zero-sample eval carries no signal and can never justify promotion.
//! 3. **Challenger checkpoint is present and loadable.** The challenger
//!    `latest_model.bin` must exist and successfully `LstmPredictor::load` — a
//!    missing/corrupt challenger is never promoted onto the live model.
//!
//! ## Safety of the overwrite itself
//!
//! * **Backup-before-overwrite (reversible).** If a champion `latest_model.bin`
//!   already exists it is first copied to a timestamped `latest_model.<unix>.bak`
//!   (never clobbering an existing backup) and the copy is size-verified. The
//!   champion is NEVER overwritten without a verified backup. First-ever
//!   promotion (no champion yet) has nothing to back up and is not destructive.
//! * **Atomic swap.** The challenger bytes are copied to a temp file in the
//!   champion dir and `rename`d over `latest_model.bin`, so a crash mid-copy can
//!   never leave a torn champion file.
//! * **Collision guard.** If the resolved challenger and champion dirs are the
//!   same directory, promotion refuses (a self-overwrite is meaningless and
//!   would risk destroying the only copy).
//! * **Loud logging.** On an ACTUAL promotion the module logs `warn!` with the
//!   promoted paths, the justifying winrates, the bar used and the backup path.
//! * **Notify only on promotion.** A [`CheckpointNotification`] pointing at the
//!   champion path is published ONLY after a successful overwrite, so forward
//!   inference hot-reloads the new champion — and only then. Rejections and the
//!   disabled path publish nothing.
//!
//! ## Metric history
//!
//! Rather than invent a new store, per-session challenger evals are appended as
//! JSON lines to `<challenger_dir>/eval_history.jsonl` by the training session
//! (see [`append_session_eval`], called from `run_training_session`). That file
//! lives INSIDE the challenger dir, so the scheduler still only ever writes the
//! challenger tree — the champion isolation invariant is preserved. This module
//! only READS that history to decide; it writes exclusively under the champion
//! dir and exclusively from the explicitly-invoked promotion path.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use janus_core::checkpoint_notify::{
    CheckpointNotification, CheckpointNotifier, CheckpointNotifierConfig,
};
use janus_ml::backend::{BackendDevice, CpuBackend};
use janus_ml::models::LstmPredictor;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

// ─── Env gates & defaults ──────────────────────────────────────────────────────

/// Master enable gate. Promotion refuses unless this is truthy (`1/true/yes/on`).
pub const PROMOTE_ENABLED_ENV: &str = "JANUS_PROMOTE_ENABLED";
/// Absolute `eval_winrate` bar used when no champion eval history exists.
pub const PROMOTE_MIN_WINRATE_ENV: &str = "JANUS_PROMOTE_MIN_WINRATE";
/// Margin the challenger must beat the champion baseline by (when a champion
/// eval history IS available).
pub const PROMOTE_MARGIN_ENV: &str = "JANUS_PROMOTE_MARGIN";
/// Number of most-recent consecutive sessions that must ALL clear the bar.
pub const PROMOTE_MIN_SESSIONS_ENV: &str = "JANUS_PROMOTE_MIN_SESSIONS";
/// Challenger checkpoint dir override (defaults to [`DEFAULT_CHALLENGER_DIR`]).
pub const PROMOTE_CHALLENGER_DIR_ENV: &str = "JANUS_PROMOTE_CHALLENGER_DIR";

/// The LIVE champion checkpoint dir — forward inference hot-loads
/// `latest_model.bin` from here. [`PromotionConfig::from_env`] ALWAYS targets
/// this exact directory; it is intentionally NOT env-overridable so the
/// production entrypoint can only ever write the real champion. Tests construct
/// [`PromotionConfig`] directly to redirect it at a tempdir.
pub const CHAMPION_CHECKPOINT_DIR: &str = "checkpoints/backward";
/// Default challenger checkpoint dir — a sub-directory of the champion dir.
pub const DEFAULT_CHALLENGER_DIR: &str = "checkpoints/backward/challenger";

/// Model checkpoint filename shared by challenger and champion.
const MODEL_FILE: &str = "latest_model.bin";
/// Per-session challenger eval history filename (JSON lines).
pub const EVAL_HISTORY_FILE: &str = "eval_history.jsonl";
/// Model-name tag on the promotion checkpoint notification (matches the tag the
/// trainer uses so the forward reloader treats it as the same lineage).
const MODEL_NAME: &str = "lstm_dqn_v1";

// Defaults (all conservative). Every one is env-overridable.
const DEFAULT_MIN_WINRATE: f64 = 0.55;
const DEFAULT_MARGIN: f64 = 0.05;
const DEFAULT_MIN_SESSIONS: usize = 3;

// ─── Recorded per-session eval ─────────────────────────────────────────────────

/// One recorded challenger evaluation, appended once per training session to
/// `<challenger_dir>/eval_history.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalRecord {
    /// Unix seconds when the record was written.
    pub unix: u64,
    /// Greedy-eval winrate for the session (see `ChallengerEval` in train.rs).
    pub eval_winrate: f64,
    /// Number of experiences evaluated (0 ⇒ no signal).
    pub eval_samples: usize,
    /// Mean realized reward over the eval slice.
    pub eval_mean_reward: f64,
    /// Directional-only winrate (Gate-A prereg §4a J2): greedy-match ∧
    /// reward ≥ 0 over recorded Buy/Sell actions only (Hold/Close excluded
    /// from numerator AND denominator). `#[serde(default)]` keeps pre-J2
    /// history lines parseable (they deserialize as 0.0 with 0 samples), and
    /// the field is additive so existing readers are unaffected.
    #[serde(default)]
    pub eval_winrate_directional: f64,
    /// Number of directional (Buy/Sell) recorded actions in the eval slice —
    /// the denominator of `eval_winrate_directional` (0 ⇒ no directional
    /// signal this session; prereg J2 requires ≥ 30 to count a session).
    #[serde(default)]
    pub eval_samples_directional: usize,
}

/// Current unix time in whole seconds (0 if the clock predates the epoch).
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Append a single [`EvalRecord`] as one JSON line to
/// `<dir>/eval_history.jsonl`, creating the dir/file if needed.
///
/// Used by the training session to record the just-computed challenger eval.
/// The caller treats a failure as non-fatal (log + continue) — recording a
/// metric must never abort training.
pub fn append_eval_record(dir: &Path, record: &EvalRecord) -> Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir).with_context(|| format!("create dir {}", dir.display()))?;
    let path = dir.join(EVAL_HISTORY_FILE);
    let mut line = serde_json::to_string(record).context("serialize eval record")?;
    line.push('\n');
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("open {} for append", path.display()))?;
    f.write_all(line.as_bytes())
        .with_context(|| format!("append eval record to {}", path.display()))?;
    Ok(())
}

/// Convenience used by the training session: build an [`EvalRecord`] stamped
/// with the current time and append it to the challenger dir.
pub fn append_session_eval(
    challenger_dir: &Path,
    eval_winrate: f64,
    eval_samples: usize,
    eval_mean_reward: f64,
    eval_winrate_directional: f64,
    eval_samples_directional: usize,
) -> Result<()> {
    append_eval_record(
        challenger_dir,
        &EvalRecord {
            unix: now_unix(),
            eval_winrate,
            eval_samples,
            eval_mean_reward,
            eval_winrate_directional,
            eval_samples_directional,
        },
    )
}

/// Read all [`EvalRecord`]s from `<dir>/eval_history.jsonl` in file order
/// (chronological). A missing file yields an empty vec; individual malformed
/// lines are skipped (with a `warn!`) rather than failing the whole read — a
/// single corrupt line must not wedge promotion decisions.
pub fn read_eval_history(dir: &Path) -> Vec<EvalRecord> {
    let path = dir.join(EVAL_HISTORY_FILE);
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<EvalRecord>(trimmed) {
            Ok(r) => out.push(r),
            Err(e) => {
                warn!(error = %e, path = %path.display(), "skipping malformed eval_history line")
            }
        }
    }
    out
}

/// Champion baseline winrate, if a champion eval history exists.
///
/// The scheduler NEVER trains or evaluates the champion, so
/// `<champion_dir>/eval_history.jsonl` is not produced in production and this
/// returns `None` — the [`evaluate_gate`] absolute-threshold path then applies
/// (documented in the module DESIGN). If such a file were ever present, the
/// most-recent recorded winrate is used as the baseline to beat by `margin`.
fn champion_baseline_winrate(champion_dir: &Path) -> Option<f64> {
    read_eval_history(champion_dir)
        .last()
        .map(|r| r.eval_winrate)
}

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Resolved promotion configuration. Every field has a safe default and
/// [`from_env`](Self::from_env) leaves promotion disabled unless explicitly
/// enabled and always targets the canonical champion dir.
#[derive(Debug, Clone)]
pub struct PromotionConfig {
    /// Master gate — false unless [`PROMOTE_ENABLED_ENV`] is truthy.
    pub enabled: bool,
    /// Absolute winrate bar (used when no champion baseline is available).
    pub min_winrate: f64,
    /// Margin over the champion baseline (used only when a baseline exists).
    pub margin: f64,
    /// Number of recent consecutive sessions that must all clear the bar (>=1).
    pub min_sessions: usize,
    /// Challenger checkpoint dir to read the candidate + history from.
    pub challenger_dir: String,
    /// Champion checkpoint dir to (safely) overwrite.
    pub champion_dir: String,
}

impl PromotionConfig {
    /// Resolve from the environment. Safe defaults: disabled, absolute bar 0.55,
    /// margin 0.05, 3 sessions, canonical challenger + champion dirs. The
    /// champion dir is fixed to [`CHAMPION_CHECKPOINT_DIR`] and is NOT
    /// env-overridable.
    pub fn from_env() -> Self {
        let enabled = parse_enabled(std::env::var(PROMOTE_ENABLED_ENV).ok().as_deref());
        let min_winrate = std::env::var(PROMOTE_MIN_WINRATE_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(DEFAULT_MIN_WINRATE);
        let margin = std::env::var(PROMOTE_MARGIN_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(DEFAULT_MARGIN);
        let min_sessions = std::env::var(PROMOTE_MIN_SESSIONS_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MIN_SESSIONS)
            .max(1);
        let challenger_dir = std::env::var(PROMOTE_CHALLENGER_DIR_ENV)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_CHALLENGER_DIR.to_string());
        Self {
            enabled,
            min_winrate,
            margin,
            min_sessions,
            challenger_dir,
            // NEVER env-overridable: the production entrypoint can only ever
            // write the real champion dir.
            champion_dir: CHAMPION_CHECKPOINT_DIR.to_string(),
        }
    }
}

/// Truthy-string parser shared by the enable gate. Only `1/true/yes/on`
/// (case/whitespace-insensitive) enable; everything else — including unset
/// (`None`) — is OFF.
fn parse_enabled(val: Option<&str>) -> bool {
    match val {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

// ─── Pure gate ──────────────────────────────────────────────────────────────────

/// Which basis the gate used to compute the bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateBasis {
    /// No champion baseline available ⇒ absolute `min_winrate`.
    AbsoluteThreshold,
    /// Champion baseline available ⇒ `baseline + margin`.
    ChampionMargin,
}

/// A passing gate result carrying the evidence for loud logging.
#[derive(Debug, Clone, PartialEq)]
struct GatePass {
    recent_winrates: Vec<f64>,
    bar: f64,
    basis: GateBasis,
}

/// Pure promotion gate (no filesystem, no models) so it is fully unit-testable.
///
/// Returns `Ok(GatePass)` when the most recent `cfg.min_sessions` records EACH
/// clear the bar and have `eval_samples > 0`; otherwise `Err(reason)`.
fn evaluate_gate(
    cfg: &PromotionConfig,
    records: &[EvalRecord],
    champion_baseline: Option<f64>,
) -> std::result::Result<GatePass, String> {
    let n = cfg.min_sessions.max(1);
    if records.len() < n {
        return Err(format!(
            "only {} recorded session(s); need {} consecutive",
            records.len(),
            n
        ));
    }

    let (bar, basis) = match champion_baseline {
        Some(b) => (b + cfg.margin, GateBasis::ChampionMargin),
        None => (cfg.min_winrate, GateBasis::AbsoluteThreshold),
    };

    let recent = &records[records.len() - n..];

    // Every recent session must carry signal AND clear the bar.
    for r in recent {
        if r.eval_samples == 0 {
            return Err(format!(
                "recent session at unix={} has 0 eval samples (no signal)",
                r.unix
            ));
        }
        // NaN/non-finite winrates must be REJECTED, not silently passed: a
        // bare `r.eval_winrate < bar` would let a NaN through (NaN compares
        // false either way). Require finiteness AND clearing the bar.
        if !r.eval_winrate.is_finite() || r.eval_winrate < bar {
            return Err(format!(
                "recent session winrate {:.4} < bar {:.4} ({:?})",
                r.eval_winrate, bar, basis
            ));
        }
    }

    Ok(GatePass {
        recent_winrates: recent.iter().map(|r| r.eval_winrate).collect(),
        bar,
        basis,
    })
}

// ─── Path helpers ────────────────────────────────────────────────────────────────

/// Lexically normalize (no fs access): drop `.`/empty components, resolve `..`.
fn lexical_normalize(p: &str) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in Path::new(p).components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// True if two dir spellings resolve to the same directory (lexically, and
/// canonically when both exist on disk).
fn same_dir(a: &str, b: &str) -> bool {
    if lexical_normalize(a) == lexical_normalize(b) {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Pick a non-existing timestamped backup path so an existing backup is never
/// clobbered: `latest_model.<unix>.bak`, falling back to
/// `latest_model.<unix>.<nanos>.bak` on the rare same-second collision.
fn backup_path(champion_dir: &Path, unix: u64) -> PathBuf {
    let candidate = champion_dir.join(format!("latest_model.{unix}.bak"));
    if !candidate.exists() {
        return candidate;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    champion_dir.join(format!("latest_model.{unix}.{nanos}.bak"))
}

// ─── Outcome ────────────────────────────────────────────────────────────────────

/// Result of a promotion attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum PromotionOutcome {
    /// Master gate off — nothing was read, written, or published.
    Disabled,
    /// Gate/precondition failed — champion untouched, nothing published.
    Rejected { reason: String },
    /// Champion overwritten (with a verified backup when one existed).
    Promoted {
        champion_path: String,
        backup_path: Option<String>,
        recent_winrates: Vec<f64>,
        bar: f64,
    },
}

// ─── The guarded promotion ────────────────────────────────────────────────────

/// Promote the challenger onto the LIVE champion, subject to EVERY safety guard
/// documented in the module DESIGN. This is the dedicated function; it performs
/// no work unless `cfg.enabled` is set and the gate passes.
///
/// * `cfg` — resolved [`PromotionConfig`] (use [`PromotionConfig::from_env`] in
///   production; construct directly in tests to redirect the champion dir).
/// * `notifier` — optional [`CheckpointNotifier`]; a notification is published
///   ONLY on an actual promotion.
pub async fn promote_challenger(
    cfg: &PromotionConfig,
    notifier: Option<&CheckpointNotifier>,
) -> Result<PromotionOutcome> {
    // ── Guard 1: master enable gate ───────────────────────────────────────
    if !cfg.enabled {
        info!(
            env = PROMOTE_ENABLED_ENV,
            "promotion disabled — set JANUS_PROMOTE_ENABLED=1 to allow; champion untouched"
        );
        return Ok(PromotionOutcome::Disabled);
    }

    let challenger_dir = Path::new(&cfg.challenger_dir);
    let champion_dir = Path::new(&cfg.champion_dir);

    // ── Guard 2: refuse a self-overwrite ──────────────────────────────────
    if same_dir(&cfg.challenger_dir, &cfg.champion_dir) {
        let reason =
            "challenger dir resolves onto the champion dir — refusing self-overwrite".to_string();
        warn!(
            challenger_dir = %cfg.challenger_dir,
            champion_dir = %cfg.champion_dir,
            "{reason}"
        );
        return Ok(PromotionOutcome::Rejected { reason });
    }

    // ── Guard 3: sustained out-performance over N sessions (pure gate) ─────
    let records = read_eval_history(challenger_dir);
    let baseline = champion_baseline_winrate(champion_dir);
    let pass = match evaluate_gate(cfg, &records, baseline) {
        Ok(p) => p,
        Err(reason) => {
            info!(reason = %reason, "promotion criteria not met — champion untouched");
            return Ok(PromotionOutcome::Rejected { reason });
        }
    };

    // ── Guard 4: challenger checkpoint present and loadable ────────────────
    let challenger_model = challenger_dir.join(MODEL_FILE);
    if !challenger_model.exists() {
        let reason = format!(
            "challenger checkpoint {} does not exist",
            challenger_model.display()
        );
        info!(reason = %reason, "promotion aborted — champion untouched");
        return Ok(PromotionOutcome::Rejected { reason });
    }
    // Load it through the exact loader forward inference uses; a corrupt or
    // architecturally-wrong file must never reach the live champion.
    if let Err(e) = LstmPredictor::<CpuBackend>::load(&challenger_model, BackendDevice::cpu()) {
        let reason = format!(
            "challenger checkpoint {} failed to load: {e}",
            challenger_model.display()
        );
        warn!(reason = %reason, "promotion aborted — champion untouched");
        return Ok(PromotionOutcome::Rejected { reason });
    }

    std::fs::create_dir_all(champion_dir)
        .with_context(|| format!("create champion dir {}", champion_dir.display()))?;
    let champion_model = champion_dir.join(MODEL_FILE);
    let unix = now_unix();

    // ── Guard 5: BACK UP the current champion before overwriting it ────────
    let backup = if champion_model.exists() {
        let backup = backup_path(champion_dir, unix);
        std::fs::copy(&champion_model, &backup).with_context(|| {
            format!(
                "backing up champion {} to {}",
                champion_model.display(),
                backup.display()
            )
        })?;
        // Verify the backup is a faithful copy BEFORE we touch the champion.
        let src_len = std::fs::metadata(&champion_model)?.len();
        let bak_len = std::fs::metadata(&backup)?.len();
        if src_len != bak_len {
            bail!(
                "champion backup size mismatch (src={src_len}, backup={bak_len}) — refusing to overwrite"
            );
        }
        Some(backup)
    } else {
        // First-ever promotion: nothing to back up, not a destructive overwrite.
        info!(
            champion = %champion_model.display(),
            "no existing champion to back up (first promotion)"
        );
        None
    };

    // ── Atomic swap: copy to temp in the champion dir, then rename over ────
    let tmp = champion_dir.join(format!("{MODEL_FILE}.promote.tmp.{unix}"));
    std::fs::copy(&challenger_model, &tmp).with_context(|| {
        format!(
            "staging challenger {} to {}",
            challenger_model.display(),
            tmp.display()
        )
    })?;
    std::fs::rename(&tmp, &champion_model).with_context(|| {
        format!(
            "atomically installing champion {}",
            champion_model.display()
        )
    })?;

    // ── Loud, unmistakable log of the promotion ───────────────────────────
    let backup_str = backup.as_ref().map(|p| p.display().to_string());
    warn!(
        champion = %champion_model.display(),
        challenger = %challenger_model.display(),
        backup = backup_str.as_deref().unwrap_or("<none: first promotion>"),
        recent_winrates = ?pass.recent_winrates,
        bar = pass.bar,
        basis = ?pass.basis,
        sessions = cfg.min_sessions,
        "PROMOTED challenger → LIVE champion (champion overwritten; backup written)"
    );

    // ── Notify downstream — ONLY on an actual promotion ────────────────────
    if let Some(n) = notifier {
        let champion_path = champion_model.to_string_lossy().to_string();
        let mut notification = CheckpointNotification::new(&champion_path, MODEL_NAME)
            .with_version(unix)
            .with_metadata("promotion", "true")
            .with_metadata("bar", format!("{:.6}", pass.bar));
        if let Some(&last) = pass.recent_winrates.last() {
            notification = notification.with_metadata("eval_winrate", format!("{last:.6}"));
        }
        if let Err(e) = n.publish(&notification).await {
            warn!(error = %e, "failed to publish promotion checkpoint notification (non-fatal; champion already updated)");
        }
    }

    Ok(PromotionOutcome::Promoted {
        champion_path: champion_model.to_string_lossy().to_string(),
        backup_path: backup_str,
        recent_winrates: pass.recent_winrates,
        bar: pass.bar,
    })
}

/// Env-gated one-shot entrypoint. Resolves [`PromotionConfig::from_env`], builds
/// a [`CheckpointNotifier`] from env when enabled, and calls
/// [`promote_challenger`]. Invoked ONLY by the `janus-backward promote`
/// subcommand — never on a timer, never from a worker, never from the scheduler.
pub async fn run_promotion_if_enabled() -> Result<PromotionOutcome> {
    let cfg = PromotionConfig::from_env();
    let notifier = if cfg.enabled {
        Some(CheckpointNotifier::new(CheckpointNotifierConfig::from_env()))
    } else {
        None
    };
    promote_challenger(&cfg, notifier.as_ref()).await
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(unix: u64, winrate: f64, samples: usize) -> EvalRecord {
        EvalRecord {
            unix,
            eval_winrate: winrate,
            eval_samples: samples,
            eval_mean_reward: 0.0,
            eval_winrate_directional: 0.0,
            eval_samples_directional: 0,
        }
    }

    /// Config with an explicit champion dir (tests redirect it at a tempdir).
    fn test_cfg(challenger: &Path, champion: &Path) -> PromotionConfig {
        PromotionConfig {
            enabled: true,
            min_winrate: 0.55,
            margin: 0.05,
            min_sessions: 3,
            challenger_dir: challenger.to_string_lossy().to_string(),
            champion_dir: champion.to_string_lossy().to_string(),
        }
    }

    /// Save a real (loadable) LstmPredictor checkpoint at `<dir>/latest_model.bin`
    /// and return its path.
    fn save_model(dir: &Path) -> PathBuf {
        use janus_ml::models::LstmConfig;
        std::fs::create_dir_all(dir).unwrap();
        let m = LstmPredictor::<CpuBackend>::new(LstmConfig::new(9, 64, 4), BackendDevice::cpu());
        let path = dir.join(MODEL_FILE);
        m.save(&path).expect("save model");
        path
    }

    // ── env / config ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_enabled_gate() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("false")));
        assert!(!parse_enabled(Some("nope")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("true")));
        assert!(parse_enabled(Some(" TRUE ")));
        assert!(parse_enabled(Some("Yes")));
        assert!(parse_enabled(Some("on")));
    }

    #[test]
    fn test_config_from_env_default_safe() {
        // With the promote env vars unset in the test process, the resolved
        // config must be safe: disabled and targeting the canonical champion.
        let cfg = PromotionConfig::from_env();
        assert!(!cfg.enabled, "promotion must default OFF");
        assert_eq!(cfg.champion_dir, CHAMPION_CHECKPOINT_DIR);
        assert_eq!(cfg.challenger_dir, DEFAULT_CHALLENGER_DIR);
        assert_ne!(
            cfg.challenger_dir, cfg.champion_dir,
            "challenger and champion dirs must differ"
        );
        assert!(cfg.min_sessions >= 1);
    }

    // ── history persistence ─────────────────────────────────────────────────

    #[test]
    fn test_eval_history_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(read_eval_history(tmp.path()).is_empty());
        append_eval_record(tmp.path(), &rec(1, 0.6, 10)).unwrap();
        append_session_eval(tmp.path(), 0.7, 20, 0.01, 0.55, 42).unwrap();
        let hist = read_eval_history(tmp.path());
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0], rec(1, 0.6, 10));
        assert_eq!(hist[1].eval_winrate, 0.7);
        assert_eq!(hist[1].eval_samples, 20);
        assert_eq!(hist[1].eval_winrate_directional, 0.55);
        assert_eq!(hist[1].eval_samples_directional, 42);
    }

    #[test]
    fn test_eval_history_reads_pre_directional_lines() {
        // History lines written BEFORE the J2 directional fields existed must
        // still parse (additive JSON fields, serde defaults) — the promotion
        // gate and any other reader keep working across the format bump.
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(EVAL_HISTORY_FILE);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"unix":100,"eval_winrate":0.62,"eval_samples":256,"eval_mean_reward":0.01}}"#
        )
        .unwrap();
        let hist = read_eval_history(tmp.path());
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].eval_winrate, 0.62);
        assert_eq!(hist[0].eval_samples, 256);
        assert_eq!(hist[0].eval_winrate_directional, 0.0, "defaults to 0.0");
        assert_eq!(hist[0].eval_samples_directional, 0, "defaults to 0");
    }

    #[test]
    fn test_read_eval_history_skips_malformed() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(EVAL_HISTORY_FILE);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{}", serde_json::to_string(&rec(1, 0.6, 5)).unwrap()).unwrap();
        writeln!(f, "not json at all").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "{}", serde_json::to_string(&rec(2, 0.8, 5)).unwrap()).unwrap();
        let hist = read_eval_history(tmp.path());
        assert_eq!(hist.len(), 2, "malformed/blank lines are skipped");
        assert_eq!(hist[0].unix, 1);
        assert_eq!(hist[1].unix, 2);
    }

    // ── pure gate ───────────────────────────────────────────────────────────

    #[test]
    fn test_gate_rejects_too_few_sessions() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_cfg(&tmp.path().join("c"), &tmp.path().join("m"));
        let records = vec![rec(1, 0.9, 10), rec(2, 0.9, 10)]; // only 2, need 3
        assert!(evaluate_gate(&cfg, &records, None).is_err());
    }

    #[test]
    fn test_gate_passes_absolute_threshold_when_no_champion_history() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_cfg(&tmp.path().join("c"), &tmp.path().join("m"));
        // 3 consecutive >= min_winrate(0.55), champion baseline None.
        let records = vec![rec(1, 0.60, 10), rec(2, 0.58, 10), rec(3, 0.57, 10)];
        let pass = evaluate_gate(&cfg, &records, None).expect("should pass");
        assert_eq!(pass.basis, GateBasis::AbsoluteThreshold);
        assert!((pass.bar - 0.55).abs() < 1e-12);
        assert_eq!(pass.recent_winrates.len(), 3);
    }

    #[test]
    fn test_gate_rejects_below_absolute_threshold() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_cfg(&tmp.path().join("c"), &tmp.path().join("m"));
        // The most recent (3rd) session dips below 0.55 → reject.
        let records = vec![rec(1, 0.60, 10), rec(2, 0.58, 10), rec(3, 0.50, 10)];
        assert!(evaluate_gate(&cfg, &records, None).is_err());
    }

    #[test]
    fn test_gate_only_considers_most_recent_n() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_cfg(&tmp.path().join("c"), &tmp.path().join("m"));
        // An old bad session is ignored; the last 3 all clear the bar.
        let records = vec![
            rec(1, 0.10, 10), // old & bad — ignored
            rec(2, 0.60, 10),
            rec(3, 0.61, 10),
            rec(4, 0.62, 10),
        ];
        assert!(evaluate_gate(&cfg, &records, None).is_ok());
    }

    #[test]
    fn test_gate_uses_champion_margin_when_baseline_available() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_cfg(&tmp.path().join("c"), &tmp.path().join("m"));
        let records = vec![rec(1, 0.62, 10), rec(2, 0.62, 10), rec(3, 0.62, 10)];
        // baseline 0.60 + margin 0.05 = 0.65 bar → 0.62 fails.
        assert!(evaluate_gate(&cfg, &records, Some(0.60)).is_err());
        // baseline 0.50 + 0.05 = 0.55 bar → 0.62 passes.
        let pass = evaluate_gate(&cfg, &records, Some(0.50)).expect("should pass");
        assert_eq!(pass.basis, GateBasis::ChampionMargin);
        assert!((pass.bar - 0.55).abs() < 1e-12);
    }

    #[test]
    fn test_gate_rejects_zero_sample_session() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_cfg(&tmp.path().join("c"), &tmp.path().join("m"));
        // High winrate but no eval samples in the latest session → no signal.
        let records = vec![rec(1, 0.9, 10), rec(2, 0.9, 10), rec(3, 0.9, 0)];
        assert!(evaluate_gate(&cfg, &records, None).is_err());
    }

    // ── full guarded promotion ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_promote_when_criteria_met_backs_up_and_overwrites() {
        let tmp = tempfile::tempdir().unwrap();
        let challenger = tmp.path().join("challenger");
        let champion = tmp.path().join("champion");

        // A distinct challenger checkpoint and an EXISTING champion so we can
        // prove the backup captured the OLD champion bytes.
        let challenger_model = save_model(&challenger);
        std::fs::create_dir_all(&champion).unwrap();
        let champion_model = champion.join(MODEL_FILE);
        std::fs::write(&champion_model, b"OLD-CHAMPION-BYTES").unwrap();

        let challenger_bytes = std::fs::read(&challenger_model).unwrap();

        // Three passing sessions.
        for (i, w) in [(1u64, 0.60), (2, 0.58), (3, 0.57)] {
            append_eval_record(&challenger, &rec(i, w, 10)).unwrap();
        }

        let cfg = test_cfg(&challenger, &champion);
        let outcome = promote_challenger(&cfg, None).await.expect("promote ok");

        let (champion_path, backup_path) = match outcome {
            PromotionOutcome::Promoted {
                champion_path,
                backup_path,
                ..
            } => (champion_path, backup_path),
            other => panic!("expected Promoted, got {other:?}"),
        };
        assert_eq!(champion_path, champion_model.to_string_lossy());

        // Champion now holds the CHALLENGER bytes.
        assert_eq!(std::fs::read(&champion_model).unwrap(), challenger_bytes);

        // A backup was created BEFORE the overwrite and holds the OLD bytes.
        let backup_path = backup_path.expect("backup must exist when a champion existed");
        assert!(Path::new(&backup_path).exists());
        assert_eq!(std::fs::read(&backup_path).unwrap(), b"OLD-CHAMPION-BYTES");

        // No temp staging file left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&champion)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("promote.tmp"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp staging file must be renamed away"
        );
    }

    #[tokio::test]
    async fn test_promote_first_time_needs_no_backup() {
        let tmp = tempfile::tempdir().unwrap();
        let challenger = tmp.path().join("challenger");
        let champion = tmp.path().join("champion"); // does not exist yet
        save_model(&challenger);
        for (i, w) in [(1u64, 0.60), (2, 0.60), (3, 0.60)] {
            append_eval_record(&challenger, &rec(i, w, 10)).unwrap();
        }
        let cfg = test_cfg(&challenger, &champion);
        let outcome = promote_challenger(&cfg, None).await.expect("promote ok");
        match outcome {
            PromotionOutcome::Promoted { backup_path, .. } => {
                assert!(backup_path.is_none(), "no backup on first promotion");
            }
            other => panic!("expected Promoted, got {other:?}"),
        }
        assert!(champion.join(MODEL_FILE).exists());
    }

    #[tokio::test]
    async fn test_no_promote_when_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let challenger = tmp.path().join("challenger");
        let champion = tmp.path().join("champion");
        save_model(&challenger);
        for (i, w) in [(1u64, 0.99), (2, 0.99), (3, 0.99)] {
            append_eval_record(&challenger, &rec(i, w, 10)).unwrap();
        }
        // Same criteria that WOULD pass, but disabled.
        let mut cfg = test_cfg(&challenger, &champion);
        cfg.enabled = false;

        let outcome = promote_challenger(&cfg, None).await.expect("ok");
        assert_eq!(outcome, PromotionOutcome::Disabled);
        // Champion was never created.
        assert!(
            !champion.join(MODEL_FILE).exists(),
            "champion must be untouched when disabled"
        );
    }

    #[tokio::test]
    async fn test_no_promote_when_criteria_unmet() {
        let tmp = tempfile::tempdir().unwrap();
        let challenger = tmp.path().join("challenger");
        let champion = tmp.path().join("champion");
        save_model(&challenger);
        std::fs::create_dir_all(&champion).unwrap();
        let champion_model = champion.join(MODEL_FILE);
        std::fs::write(&champion_model, b"OLD-CHAMPION-BYTES").unwrap();

        // Recent session dips below the absolute threshold → reject.
        for (i, w) in [(1u64, 0.60), (2, 0.60), (3, 0.40)] {
            append_eval_record(&challenger, &rec(i, w, 10)).unwrap();
        }
        let cfg = test_cfg(&challenger, &champion);
        let outcome = promote_challenger(&cfg, None).await.expect("ok");
        assert!(matches!(outcome, PromotionOutcome::Rejected { .. }));

        // Champion is byte-for-byte unchanged and NO backup was written.
        assert_eq!(
            std::fs::read(&champion_model).unwrap(),
            b"OLD-CHAMPION-BYTES"
        );
        let bak_count = std::fs::read_dir(&champion)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".bak"))
            .count();
        assert_eq!(bak_count, 0, "no backup on rejection");
    }

    #[tokio::test]
    async fn test_no_promote_when_challenger_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let challenger = tmp.path().join("challenger");
        let champion = tmp.path().join("champion");
        std::fs::create_dir_all(&challenger).unwrap();
        // History passes, but there is NO challenger model file.
        for (i, w) in [(1u64, 0.9), (2, 0.9), (3, 0.9)] {
            append_eval_record(&challenger, &rec(i, w, 10)).unwrap();
        }
        let cfg = test_cfg(&challenger, &champion);
        let outcome = promote_challenger(&cfg, None).await.expect("ok");
        assert!(matches!(outcome, PromotionOutcome::Rejected { .. }));
        assert!(!champion.join(MODEL_FILE).exists());
    }

    #[tokio::test]
    async fn test_refuses_self_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("same");
        save_model(&dir);
        for (i, w) in [(1u64, 0.9), (2, 0.9), (3, 0.9)] {
            append_eval_record(&dir, &rec(i, w, 10)).unwrap();
        }
        // challenger_dir == champion_dir → refuse.
        let cfg = test_cfg(&dir, &dir);
        let outcome = promote_challenger(&cfg, None).await.expect("ok");
        assert!(matches!(outcome, PromotionOutcome::Rejected { .. }));
    }
}
