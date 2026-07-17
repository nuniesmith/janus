//! Daily block-mix soak log (Gate-A prereg §4a J3).
//!
//! The J3 criterion — "the AO-fixed 9-gate block-mix stable through the
//! soak" — needs a **time series**, and until this module the only
//! observability was the rolling 1000-signal Redis window (3-day TTL) plus
//! ad-hoc samples: no series, no soak. This task appends one JSON line per
//! day (plus one at startup) with the gate's cumulative per-reason block
//! counters, pass count, and pass rate to an append-only file inside the
//! checkpoints volume — the same dir convention as the challenger's
//! `eval_history.jsonl`, so it survives redeploys.
//!
//! ## Semantics
//!
//! * **Opt-in, coupled to the exporter flag:** the log runs iff the
//!   gate-counter Redis exporter is enabled (`JANUS_GATE_METRICS_REDIS=1` —
//!   [`crate::gate_metrics_redis`]), because prereg J3 requires that flag for
//!   the soak (it also fixes breaker persistence). Default OFF: deploying
//!   this code changes nothing until the operator opts in.
//! * **Counters are cumulative since process start** (they are the gate's
//!   in-memory [`GateMetricsSnapshot`] counters). Readers compute daily
//!   deltas by differencing consecutive lines; the `startup: true` line
//!   marks a counter reset (restart) so a reader never diffs across one.
//! * **Crash-proof:** the writer is a supervised critical task (the
//!   #156/#157 pattern — `janus_core::supervisor::supervise`): a panic
//!   restarts it with capped backoff instead of silently ending the soak
//!   log. Each append is best-effort (log + continue on I/O error) and a
//!   single JSON line, so a torn write can corrupt at most one line and
//!   line-oriented readers skip it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::gate_integration::{ForwardGate, GateMetricsSnapshot};

/// Default append-only log path — inside the checkpoints volume (the same
/// convention as `checkpoints/backward/challenger/eval_history.jsonl`) so the
/// soak series survives container redeploys.
pub const DEFAULT_LOG_PATH: &str = "checkpoints/forward/gate_block_mix.jsonl";
/// Default snapshot interval: daily, per prereg §4a J3.
pub const DEFAULT_INTERVAL_SECS: u64 = 24 * 60 * 60;

/// Configuration for the block-mix soak log.
#[derive(Debug, Clone)]
pub struct BlockMixConfig {
    /// Runs iff the gate-metrics exporter is enabled
    /// (`JANUS_GATE_METRICS_REDIS`). Off by default.
    pub enabled: bool,
    /// Append-only JSONL path (`JANUS_GATE_BLOCK_MIX_PATH`).
    pub path: PathBuf,
    /// Snapshot interval (`JANUS_GATE_BLOCK_MIX_INTERVAL_SECS`, floored to
    /// 60s; default daily).
    pub interval: Duration,
}

impl Default for BlockMixConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            path: PathBuf::from(DEFAULT_LOG_PATH),
            interval: Duration::from_secs(DEFAULT_INTERVAL_SECS),
        }
    }
}

impl BlockMixConfig {
    /// Build from the environment. `enabled` deliberately reuses the
    /// exporter's own flag ([`crate::gate_metrics_redis::GateMetricsRedisConfig`])
    /// — J3 requires the exporter, and one opt-in switch arms both.
    pub fn from_env() -> Self {
        let enabled = crate::gate_metrics_redis::GateMetricsRedisConfig::from_env().enabled;
        let path = std::env::var("JANUS_GATE_BLOCK_MIX_PATH")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_LOG_PATH));
        let interval_secs: u64 = std::env::var("JANUS_GATE_BLOCK_MIX_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DEFAULT_INTERVAL_SECS);
        Self {
            enabled,
            path,
            interval: Duration::from_secs(interval_secs.max(60)),
        }
    }
}

/// One appended snapshot line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockMixRecord {
    /// Unix seconds when the snapshot was taken.
    pub unix: u64,
    /// RFC3339 of the same instant (human-side convenience).
    pub iso: String,
    /// True for the boot-time snapshot — marks a counter reset, so readers
    /// must not difference across it.
    #[serde(default)]
    pub startup: bool,
    /// Total gate evaluations since process start (all assets).
    pub evals_total: u64,
    /// Total blocks since process start (all assets, all reasons).
    pub blocks_total: u64,
    /// `evals_total − blocks_total` (an eval either blocks with exactly one
    /// first-block reason or passes — the gate is a first-block chain).
    pub pass_total: u64,
    /// `pass_total / evals_total` (0.0 when there are no evals yet).
    pub pass_rate: f64,
    /// Cumulative block count per reason, summed across assets. BTreeMap ⇒
    /// deterministic key order in the JSON.
    pub blocks_by_reason: BTreeMap<String, u64>,
}

/// Reduce a [`GateMetricsSnapshot`] (per-asset counters) to one
/// [`BlockMixRecord`] (per-reason totals + pass rate). Pure — unit-tested.
pub fn record_from_snapshot(
    snapshot: &GateMetricsSnapshot,
    unix: u64,
    iso: String,
    startup: bool,
) -> BlockMixRecord {
    let evals_total: u64 = snapshot.evals.iter().map(|(_, c)| *c).sum();
    let mut blocks_by_reason: BTreeMap<String, u64> = BTreeMap::new();
    for (_asset, reason, count) in &snapshot.blocks {
        *blocks_by_reason.entry(reason.clone()).or_default() += *count;
    }
    let blocks_total: u64 = blocks_by_reason.values().sum();
    let pass_total = evals_total.saturating_sub(blocks_total);
    let pass_rate = if evals_total > 0 {
        pass_total as f64 / evals_total as f64
    } else {
        0.0
    };
    BlockMixRecord {
        unix,
        iso,
        startup,
        evals_total,
        blocks_total,
        pass_total,
        pass_rate,
        blocks_by_reason,
    }
}

/// Append one record as a single JSON line to `path`, creating parent
/// directories and the file as needed. Never truncates — append-only.
pub fn append_record(path: &Path, record: &BlockMixRecord) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create dir {}", parent.display()))?;
    }
    let mut line = serde_json::to_string(record).context("serialize block-mix record")?;
    line.push('\n');
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {} for append", path.display()))?;
    f.write_all(line.as_bytes())
        .with_context(|| format!("append block-mix record to {}", path.display()))?;
    Ok(())
}

/// Snapshot the shared gate and append one line. Best-effort: an I/O failure
/// is logged and never propagates — losing one snapshot must not kill the
/// task (and the supervisor would only respawn it into the same disk state).
async fn write_snapshot(gate: &Arc<RwLock<ForwardGate>>, path: &Path, startup: bool) {
    let snapshot = gate.read().await.metrics_snapshot();
    let now = chrono::Utc::now();
    let record = record_from_snapshot(
        &snapshot,
        now.timestamp().max(0) as u64,
        now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        startup,
    );
    match append_record(path, &record) {
        Ok(()) => info!(
            path = %path.display(),
            evals = record.evals_total,
            blocks = record.blocks_total,
            pass_rate = format!("{:.4}", record.pass_rate),
            startup,
            "block-mix snapshot appended"
        ),
        Err(e) => warn!(
            error = %e,
            path = %path.display(),
            "block-mix snapshot append failed (non-fatal; next interval retries)"
        ),
    }
}

/// Spawn the supervised daily block-mix logger.
///
/// Returns `None` (and starts nothing) when disabled — symmetric with
/// [`crate::gate_metrics_redis::spawn`]. When enabled it writes one startup
/// snapshot immediately, then one line per interval. The interval loop runs
/// under [`janus_core::supervisor::supervise`] (the #156/#157 critical-task
/// pattern): a panic restarts the loop with capped backoff; the detached
/// task's abort at shutdown stops supervision cleanly.
pub fn spawn(gate: Arc<RwLock<ForwardGate>>, config: BlockMixConfig) -> Option<JoinHandle<()>> {
    if !config.enabled {
        info!(
            "block-mix soak log: disabled (set JANUS_GATE_METRICS_REDIS=1 to arm the gate \
             exporter + daily block-mix log)"
        );
        return None;
    }

    info!(
        path = %config.path.display(),
        interval_secs = config.interval.as_secs(),
        "✅ block-mix soak log started (daily gate-counter snapshots)"
    );

    Some(tokio::spawn(async move {
        // Startup snapshot: marks the counter-reset point for readers.
        write_snapshot(&gate, &config.path, true).await;

        // Supervised interval loop. The factory re-clones the handles per
        // (re)start; `|| false` never requests shutdown — the loop ends only
        // when this detached task is aborted (supervise treats the abort as
        // the shutdown path and stops).
        let path = config.path.clone();
        let interval = config.interval;
        janus_core::supervisor::supervise(
            "gate-block-mix-log",
            || false,
            move || {
                let gate = Arc::clone(&gate);
                let path = path.clone();
                async move {
                    loop {
                        tokio::time::sleep(interval).await;
                        write_snapshot(&gate, &path, false).await;
                    }
                }
            },
        )
        .await;
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> GateMetricsSnapshot {
        GateMetricsSnapshot {
            evals: vec![("btc".to_string(), 100), ("eth".to_string(), 50)],
            blocks: vec![
                ("btc".to_string(), "block_risk".to_string(), 30),
                ("btc".to_string(), "block_vol_filter".to_string(), 10),
                ("eth".to_string(), "block_risk".to_string(), 20),
                ("eth".to_string(), "block_quality".to_string(), 5),
            ],
        }
    }

    #[test]
    fn record_aggregates_reasons_across_assets_and_derives_pass_rate() {
        let rec =
            record_from_snapshot(&snap(), 1_800_000_000, "2027-01-15T00:00:00Z".into(), false);
        assert_eq!(rec.evals_total, 150);
        assert_eq!(rec.blocks_total, 65);
        assert_eq!(rec.pass_total, 85);
        assert!((rec.pass_rate - 85.0 / 150.0).abs() < 1e-12);
        // Per-reason sums cross assets: block_risk = 30 + 20.
        assert_eq!(rec.blocks_by_reason.get("block_risk"), Some(&50));
        assert_eq!(rec.blocks_by_reason.get("block_vol_filter"), Some(&10));
        assert_eq!(rec.blocks_by_reason.get("block_quality"), Some(&5));
        assert_eq!(rec.blocks_by_reason.len(), 3);
        assert!(!rec.startup);
    }

    #[test]
    fn record_empty_snapshot_is_zeroed_with_zero_pass_rate() {
        let rec = record_from_snapshot(
            &GateMetricsSnapshot::default(),
            1,
            "1970-01-01T00:00:01Z".into(),
            true,
        );
        assert_eq!(rec.evals_total, 0);
        assert_eq!(rec.blocks_total, 0);
        assert_eq!(rec.pass_total, 0);
        assert_eq!(rec.pass_rate, 0.0, "no evals must not divide by zero");
        assert!(rec.blocks_by_reason.is_empty());
        assert!(rec.startup);
    }

    #[test]
    fn record_serializes_to_one_json_line_and_round_trips() {
        let rec = record_from_snapshot(&snap(), 42, "iso".into(), true);
        let line = serde_json::to_string(&rec).unwrap();
        assert!(!line.contains('\n'), "one record = one line");
        for key in [
            "unix",
            "iso",
            "startup",
            "evals_total",
            "blocks_total",
            "pass_total",
            "pass_rate",
            "blocks_by_reason",
        ] {
            assert!(line.contains(key), "line must contain {key}: {line}");
        }
        let back: BlockMixRecord = serde_json::from_str(&line).unwrap();
        assert_eq!(back, rec);
        // A line written before the `startup` field existed still parses.
        let legacy: BlockMixRecord = serde_json::from_str(
            r#"{"unix":1,"iso":"x","evals_total":2,"blocks_total":1,"pass_total":1,"pass_rate":0.5,"blocks_by_reason":{}}"#,
        )
        .unwrap();
        assert!(!legacy.startup);
    }

    #[test]
    fn append_creates_parents_and_appends_without_truncating() {
        let tmp = tempfile::tempdir().unwrap();
        // Nested path exercises parent-dir creation (the checkpoints volume
        // starts without a forward/ subdir).
        let path = tmp.path().join("forward").join("gate_block_mix.jsonl");

        let a = record_from_snapshot(&snap(), 1, "t1".into(), true);
        let b = record_from_snapshot(&GateMetricsSnapshot::default(), 2, "t2".into(), false);
        append_record(&path, &a).unwrap();
        append_record(&path, &b).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "append-only: both lines present");
        let ra: BlockMixRecord = serde_json::from_str(lines[0]).unwrap();
        let rb: BlockMixRecord = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(ra, a);
        assert_eq!(rb, b);
        assert_eq!(ra.unix, 1);
        assert_eq!(rb.unix, 2, "chronological file order");
    }

    #[test]
    fn config_default_and_env_default_are_off() {
        let c = BlockMixConfig::default();
        assert!(!c.enabled, "must default OFF");
        assert_eq!(c.path, PathBuf::from(DEFAULT_LOG_PATH));
        assert_eq!(c.interval, Duration::from_secs(DEFAULT_INTERVAL_SECS));
    }

    #[tokio::test]
    async fn spawn_disabled_starts_nothing() {
        let gate = Arc::new(RwLock::new(ForwardGate::from_env()));
        let handle = spawn(gate, BlockMixConfig::default());
        assert!(handle.is_none(), "disabled config must not spawn a task");
    }

    #[tokio::test]
    async fn spawn_enabled_writes_startup_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gate_block_mix.jsonl");
        let gate = Arc::new(RwLock::new(ForwardGate::from_env()));
        let config = BlockMixConfig {
            enabled: true,
            path: path.clone(),
            // Long interval: only the startup snapshot should appear.
            interval: Duration::from_secs(3600),
        };
        let handle = spawn(gate, config).expect("enabled config spawns");

        // Wait (bounded) for the startup line to land, then stop the task.
        let mut found = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            if path.exists() && !std::fs::read_to_string(&path).unwrap().is_empty() {
                found = true;
                break;
            }
        }
        handle.abort();
        assert!(found, "startup snapshot must be appended promptly");
        let contents = std::fs::read_to_string(&path).unwrap();
        let rec: BlockMixRecord = serde_json::from_str(contents.lines().next().unwrap()).unwrap();
        assert!(rec.startup, "boot line is marked as a counter reset");
        assert_eq!(rec.evals_total, 0, "fresh gate has no evals yet");
    }
}
