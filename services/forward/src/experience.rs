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
}

impl DecisionCtx {
    /// The overwhelmingly common case: janus decided not to act.
    pub fn hold() -> Self {
        Self {
            action: janus_core::SignalType::Hold,
            confidence: 0.0,
            blocked: None,
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
    pub reward: f32,
    pub next_state_gaf: Vec<f32>,
    pub next_state_raw: Option<Vec<f32>>,
    pub done: bool,
    pub timestamp_ms: i64,
    // Side-channel (file metadata, NOT schema fields — §3):
    pub interval: String,
    pub confidence: f64,
    pub blocked: Option<String>,
}

struct Pending {
    close_time_ms: i64,
    close: f64,
    state_gaf: Vec<f32>,
    ctx: DecisionCtx,
    interval: String,
    interval_secs: i64,
    symbol: String,
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

/// Per-`SYMBOL:interval` close windows + pending-decision slots. Owned by
/// the live-loop task; all methods are synchronous and allocation-light.
pub struct ExperienceRecorder {
    windows: HashMap<String, VecDeque<f32>>,
    pending: HashMap<String, Pending>,
    tx: mpsc::Sender<ExperienceRow>,
    scratch: Vec<f32>,
}

impl ExperienceRecorder {
    pub fn new(tx: mpsc::Sender<ExperienceRow>) -> Self {
        Self {
            windows: HashMap::new(),
            pending: HashMap::new(),
            tx,
            scratch: Vec::with_capacity(GAF_WINDOW),
        }
    }

    /// Observe one closed candle for `key` (= "SYMBOL:interval"). Completes
    /// the previous pending decision for the key (reward = signed next-bar
    /// log return, §3.3) and arms a new one with `ctx`, once the GAF window
    /// is warm. Never blocks: the row is `try_send`-ed to the writer.
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

        // Complete the previous pending decision, if any.
        if let Some(p) = self.pending.remove(key) {
            let dt_ms = close_time_ms.saturating_sub(p.close_time_ms);
            let drop_ms = p.interval_secs.saturating_mul(PENDING_DROP_FACTOR) * 1000;
            if dt_ms > drop_ms {
                metrics().pending_gap_dropped.inc();
            } else if !gaf_now.is_empty() {
                let done_ms = (2 * p.interval_secs).max(PENDING_DONE_FLOOR_SECS) * 1000;
                let done = dt_ms > done_ms;
                let dir: f32 = match p.ctx.action {
                    janus_core::SignalType::Buy => 1.0,
                    janus_core::SignalType::Sell => -1.0,
                    _ => 0.0,
                };
                let reward = dir * ((close / p.close).ln() as f32);
                let row = ExperienceRow {
                    state_gaf: p.state_gaf,
                    state_raw: None,
                    action_type: p.ctx.action.to_u8(),
                    action_symbol: p.symbol,
                    action_qty: 0.0,
                    reward,
                    next_state_gaf: gaf_now.clone(),
                    next_state_raw: None,
                    done,
                    timestamp_ms: p.close_time_ms,
                    interval: p.interval,
                    confidence: p.ctx.confidence,
                    blocked: p.ctx.blocked,
                };
                match self.tx.try_send(row) {
                    Ok(()) => metrics().rows_recorded.inc(),
                    Err(_) => metrics().rows_dropped_channel.inc(),
                }
            }
            // If the window went cold (shouldn't happen once warm), the
            // pending decision is silently superseded below.
        }

        // Arm the new pending decision once warm.
        if !gaf_now.is_empty() {
            self.pending.insert(
                key.to_string(),
                Pending {
                    close_time_ms,
                    close,
                    state_gaf: gaf_now,
                    interval_secs: interval_secs(interval),
                    interval: interval.to_string(),
                    ctx,
                    symbol: symbol.to_string(),
                },
            );
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
    #[serde(skip_serializing_if = "Option::is_none")]
    blocked: Option<&'a str>,
}

fn write_ipc(path: &std::path::Path, rows: &[ExperienceRow]) -> anyhow::Result<()> {
    let row_meta: Vec<RowMeta<'_>> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| RowMeta {
            i,
            interval: &r.interval,
            confidence: r.confidence,
            blocked: r.blocked.as_deref(),
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
                DecisionCtx::hold(),
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
            },
        );
        rec.observe("K:1m", "BTCUSDT", "1m", t + MIN, 110.0, DecisionCtx::hold());
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
            },
        );
        rec.observe("K:1m", "BTCUSDT", "1m", t + MIN, 110.0, DecisionCtx::hold());
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
            DecisionCtx::hold(),
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
            DecisionCtx::hold(),
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
                DecisionCtx::hold(),
            );
            rec.observe(
                "BTCUSDT:5m",
                "BTCUSDT",
                "5m",
                t * 5,
                200.0 + i as f64,
                DecisionCtx::hold(),
            );
            t += MIN;
        }
        let mut count = 0;
        while rx.try_recv().is_ok() {
            count += 1;
        }
        assert_eq!(count, 2, "one completed row per key");
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
                next_state_gaf: vec![0.2; 9],
                next_state_raw: None,
                done: false,
                timestamp_ms: 1_700_000_000_000 + i as i64,
                interval: "1m".to_string(),
                confidence: 0.5,
                blocked: None,
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
