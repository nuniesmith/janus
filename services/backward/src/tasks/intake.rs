//! Experience intake worker — Redis BRPOP + spool sweep (design §4.3).
//!
//! The forward service writes finished Arrow IPC batches to a tmpfs spool
//! directory and `LPUSH`es an [`IngestJob`] (JSON) onto the
//! `janus:experience:ingest` Redis list as a doorbell. This worker:
//!
//! 1. `BRPOP`s the list (5 s timeout), parses the entry (bare `IngestJob` or
//!    a retry envelope `{"job": …, "attempts": N}`), and runs
//!    [`handle_ingest`] against the [`ExperienceStore`].
//! 2. On success it deletes the spool file; on failure it re-`LPUSH`es the
//!    job with an incremented attempt counter, parking the file into
//!    `<spool>/failed/` after [`MAX_INGEST_ATTEMPTS`] attempts.
//! 3. Sweeps the spool on startup and every 5 minutes for `*.arrow` files
//!    older than 2 minutes (crash / Redis-loss recovery), retries parked
//!    files in `failed/` hourly, and deletes stale `*.tmp` files (> 1 h).
//!
//! The queue is only an optimization: files are the source of truth and the
//! sweep converges on them. Duplicate delivery (sweep + late queue entry) is
//! harmless because point IDs are deterministic UUIDv5s (design §5).

use anyhow::{Context, Result};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::metrics::ExperienceMetricsCollector;
use crate::persistence::{ExperienceStore, ExperienceStoreConfig};
use crate::tasks::ingest::handle_ingest;
use crate::worker::IngestJob;

// ─── Contract constants (shared with the forward writer) ─────────────────────

/// Redis list carrying `IngestJob` JSON entries (design §4.3).
pub const INGEST_QUEUE_KEY: &str = "janus:experience:ingest";

/// Environment variable naming the spool directory (same one the forward
/// experience writer uses).
pub const SPOOL_DIR_ENV: &str = "JANUS_EXPERIENCE_SPOOL_DIR";

/// Default spool directory (tmpfs).
pub const DEFAULT_SPOOL_DIR: &str = "/dev/shm/janus/experience";

/// Maximum ingest attempts before a spool file is parked in `failed/`.
pub const MAX_INGEST_ATTEMPTS: u32 = 3;

/// BRPOP blocking timeout in seconds.
const BRPOP_TIMEOUT_SECS: f64 = 5.0;

/// Periodic spool sweep interval.
const SWEEP_INTERVAL: Duration = Duration::from_secs(300);

/// Retry interval for files parked in `failed/`.
const FAILED_RETRY_INTERVAL: Duration = Duration::from_secs(3600);

/// A spool file must be at least this old before the sweep picks it up
/// (avoids racing a just-written file whose queue entry is in flight).
const SWEEP_MIN_AGE: Duration = Duration::from_secs(120);

/// Stale `.tmp` files (crash mid-write, no IPC footer) older than this are
/// deleted by the sweep.
const TMP_MAX_AGE: Duration = Duration::from_secs(3600);

// ─── Queue entry envelope ─────────────────────────────────────────────────────

/// Retry envelope wrapped around an [`IngestJob`] when the worker re-queues a
/// failed job: `{"job": {"batch_id": …, "shm_path": …}, "attempts": N}`.
///
/// `attempts` counts ingest attempts already made. Producers push bare
/// `IngestJob` JSON (equivalent to `attempts = 0`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestEnvelope {
    /// The wrapped ingest job.
    pub job: IngestJob,
    /// Number of ingest attempts already made for this job.
    #[serde(default)]
    pub attempts: u32,
}

/// Parse a queue entry: first as a retry [`IngestEnvelope`], then as a bare
/// [`IngestJob`] (the shape the forward writer pushes).
pub fn parse_queue_entry(payload: &str) -> Result<IngestEnvelope> {
    if let Ok(envelope) = serde_json::from_str::<IngestEnvelope>(payload) {
        return Ok(envelope);
    }
    let job: IngestJob = serde_json::from_str(payload).with_context(|| {
        format!("Queue entry is neither IngestEnvelope nor IngestJob JSON: {payload}")
    })?;
    Ok(IngestEnvelope { job, attempts: 0 })
}

/// What to do with a job after a failed ingest attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureDisposition {
    /// Re-queue with the given (incremented) attempt count.
    Requeue(u32),
    /// Attempt budget exhausted — park the spool file in `failed/`.
    Park,
}

/// Decide whether a job that just failed (having previously made
/// `prev_attempts` attempts) should be re-queued or parked.
pub fn disposition_after_failure(prev_attempts: u32) -> FailureDisposition {
    let attempts = prev_attempts.saturating_add(1);
    if attempts >= MAX_INGEST_ATTEMPTS {
        FailureDisposition::Park
    } else {
        FailureDisposition::Requeue(attempts)
    }
}

// ─── Spool helpers ────────────────────────────────────────────────────────────

/// Resolve the spool directory from `JANUS_EXPERIENCE_SPOOL_DIR`, defaulting
/// to `/dev/shm/janus/experience`.
pub fn spool_dir_from_env() -> PathBuf {
    std::env::var(SPOOL_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_SPOOL_DIR))
}

/// Move a spool file into `<spool>/failed/` for later (hourly) retry.
///
/// Returns the parked path, or `Ok(None)` if the file no longer exists.
pub fn park_file(spool_dir: &Path, shm_path: &str) -> Result<Option<PathBuf>> {
    let src = Path::new(shm_path);
    if !src.exists() {
        return Ok(None);
    }
    let file_name = src
        .file_name()
        .with_context(|| format!("Spool path has no file name: {shm_path}"))?;

    let failed_dir = spool_dir.join("failed");
    std::fs::create_dir_all(&failed_dir)
        .with_context(|| format!("Failed to create {}", failed_dir.display()))?;

    let dest = failed_dir.join(file_name);
    std::fs::rename(src, &dest)
        .with_context(|| format!("Failed to park {} → {}", src.display(), dest.display()))?;
    Ok(Some(dest))
}

/// True if the file at `path` was last modified more than `min_age` ago.
/// Unreadable metadata counts as "not old enough" (skipped this round).
fn older_than(path: &Path, min_age: Duration) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|mtime| mtime.elapsed().ok())
        .is_some_and(|age| age >= min_age)
}

/// List `*.arrow` files directly under `dir` older than `min_age` and
/// synthesize [`IngestJob`]s for them (batch_id = file stem). Sorted by file
/// name, which is chronological for `<unix_ms>-<uuid>` batch IDs.
pub fn sweepable_jobs(dir: &Path, min_age: Duration) -> Vec<IngestJob> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut jobs: Vec<IngestJob> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension().and_then(|e| e.to_str()) == Some("arrow")
                && older_than(p, min_age)
        })
        .filter_map(|p| {
            let batch_id = p.file_stem()?.to_str()?.to_string();
            Some(IngestJob {
                batch_id,
                shm_path: p.to_string_lossy().into_owned(),
            })
        })
        .collect();

    jobs.sort_by(|a, b| a.shm_path.cmp(&b.shm_path));
    jobs
}

/// Delete `*.tmp` files directly under `dir` older than `max_age` (crash
/// leftovers with no IPC footer). Returns the number of files removed.
pub fn cleanup_tmp_files(dir: &Path, max_age: Duration) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };

    let mut removed = 0;
    for path in entries.flatten().map(|e| e.path()) {
        let is_tmp = path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("tmp");
        if is_tmp && older_than(&path, max_age) {
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    info!(path = %path.display(), "Deleted stale spool .tmp file");
                    removed += 1;
                }
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Failed to delete stale .tmp file")
                }
            }
        }
    }
    removed
}

// ─── Intake worker ────────────────────────────────────────────────────────────

/// Configuration for the intake worker.
#[derive(Debug, Clone)]
pub struct IntakeWorkerConfig {
    /// Redis URL for the ingest doorbell list.
    pub redis_url: String,
    /// Spool directory (see [`spool_dir_from_env`]).
    pub spool_dir: PathBuf,
    /// Experience store configuration.
    pub store_config: ExperienceStoreConfig,
}

impl IntakeWorkerConfig {
    /// Build from a Redis URL plus environment (spool dir + store config).
    pub fn from_env(redis_url: String) -> Self {
        Self {
            redis_url,
            spool_dir: spool_dir_from_env(),
            store_config: ExperienceStoreConfig::from_env(),
        }
    }
}

/// Run the experience intake worker until `cancel` fires.
///
/// Connects the [`ExperienceStore`] with retry/backoff (a real Qdrant that is
/// down is an error, never a silent mock fallback), then loops over
/// BRPOP-driven ingest and the periodic sweeps.
pub async fn run_intake_worker(
    config: IntakeWorkerConfig,
    metrics: ExperienceMetricsCollector,
    cancel: CancellationToken,
) {
    info!(
        spool_dir = %config.spool_dir.display(),
        queue = INGEST_QUEUE_KEY,
        "Experience intake worker starting"
    );

    let Some(store) =
        ExperienceStore::connect_with_retry(config.store_config.clone(), &cancel).await
    else {
        info!("Experience intake worker cancelled before store connected");
        return;
    };

    if let Err(e) = std::fs::create_dir_all(&config.spool_dir) {
        warn!(
            spool_dir = %config.spool_dir.display(),
            error = %e,
            "Failed to create spool directory — sweep will be a no-op until it exists"
        );
    }

    let mut redis_conn = connect_redis(&config.redis_url).await;

    let mut sweep_tick = tokio::time::interval(SWEEP_INTERVAL);
    let mut failed_retry_tick = tokio::time::interval(FAILED_RETRY_INTERVAL);
    // Both intervals fire immediately on the first tick → startup sweep +
    // startup retry of previously parked files.

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = sweep_tick.tick() => {
                run_sweep(&config, &store, &mut redis_conn, &metrics).await;
            }
            _ = failed_retry_tick.tick() => {
                retry_failed_files(&config, &store, &metrics).await;
            }
            payload = poll_queue(&mut redis_conn, &config.redis_url) => {
                if let Some(payload) = payload {
                    process_queue_entry(&payload, &config, &store, &mut redis_conn, &metrics).await;
                }
            }
        }
    }

    info!("Experience intake worker exited");
}

/// Try to establish a Redis connection (bounded by a 5 s timeout).
async fn connect_redis(redis_url: &str) -> Option<redis::aio::ConnectionManager> {
    let client = match redis::Client::open(redis_url) {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "Invalid Redis URL for experience intake — running sweep-only");
            return None;
        }
    };

    match tokio::time::timeout(
        Duration::from_secs(5),
        redis::aio::ConnectionManager::new(client),
    )
    .await
    {
        Ok(Ok(conn)) => {
            info!("Experience intake connected to Redis");
            Some(conn)
        }
        Ok(Err(e)) => {
            warn!(error = %e, "Experience intake failed to connect to Redis — sweep will recover batches");
            None
        }
        Err(_) => {
            warn!("Experience intake Redis connection timed out — sweep will recover batches");
            None
        }
    }
}

/// BRPOP one entry from the ingest queue. Returns `None` on timeout, error,
/// or missing connection (with pacing sleeps so the caller's loop never spins).
async fn poll_queue(
    conn: &mut Option<redis::aio::ConnectionManager>,
    redis_url: &str,
) -> Option<String> {
    match conn {
        Some(c) => {
            let res: redis::RedisResult<Option<(String, String)>> =
                c.brpop(INGEST_QUEUE_KEY, BRPOP_TIMEOUT_SECS).await;
            match res {
                Ok(Some((_key, payload))) => Some(payload),
                Ok(None) => None, // timeout, queue empty
                Err(e) => {
                    warn!(error = %e, "BRPOP failed — dropping Redis connection");
                    *conn = None;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    None
                }
            }
        }
        None => {
            // No connection: pace ourselves, then try to reconnect.
            tokio::time::sleep(Duration::from_secs(5)).await;
            *conn = connect_redis(redis_url).await;
            None
        }
    }
}

/// Parse and ingest one queue entry.
async fn process_queue_entry(
    payload: &str,
    config: &IntakeWorkerConfig,
    store: &ExperienceStore,
    redis_conn: &mut Option<redis::aio::ConnectionManager>,
    metrics: &ExperienceMetricsCollector,
) {
    let envelope = match parse_queue_entry(payload) {
        Ok(env) => env,
        Err(e) => {
            error!(error = %e, "Dropping malformed experience queue entry");
            metrics.batches_failed.inc();
            return;
        }
    };

    ingest_envelope(envelope, config, store, redis_conn, metrics, false).await;
}

/// Run [`handle_ingest`] for one envelope: delete the spool file on success,
/// re-queue or park on failure.
async fn ingest_envelope(
    envelope: IngestEnvelope,
    config: &IntakeWorkerConfig,
    store: &ExperienceStore,
    redis_conn: &mut Option<redis::aio::ConnectionManager>,
    metrics: &ExperienceMetricsCollector,
    from_sweep: bool,
) {
    let job = envelope.job.clone();

    match handle_ingest(job.clone(), Some(store)).await {
        Ok(m) => {
            match std::fs::remove_file(&job.shm_path) {
                Ok(()) => debug!(batch_id = %job.batch_id, "Deleted ingested spool file"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => warn!(
                    batch_id = %job.batch_id,
                    path = %job.shm_path,
                    error = %e,
                    "Failed to delete ingested spool file"
                ),
            }
            metrics.batches_ingested.inc();
            metrics.rows_ingested.inc_by(m.rows_ingested as u64);
            metrics.rows_skipped.inc_by(m.rows_skipped as u64);
            if from_sweep {
                metrics.sweep_recovered.inc();
            }
        }
        Err(e) => {
            warn!(
                batch_id = %job.batch_id,
                attempts = envelope.attempts,
                error = %format!("{e:#}"),
                "Experience ingest attempt failed"
            );

            match disposition_after_failure(envelope.attempts) {
                FailureDisposition::Park => {
                    match park_file(&config.spool_dir, &job.shm_path) {
                        Ok(Some(dest)) => warn!(
                            batch_id = %job.batch_id,
                            parked = %dest.display(),
                            "Parked spool file after {MAX_INGEST_ATTEMPTS} failed attempts"
                        ),
                        Ok(None) => warn!(
                            batch_id = %job.batch_id,
                            "Spool file already gone — nothing to park"
                        ),
                        Err(pe) => error!(
                            batch_id = %job.batch_id,
                            error = %pe,
                            "Failed to park spool file"
                        ),
                    }
                    metrics.batches_failed.inc();
                }
                FailureDisposition::Requeue(attempts) => {
                    let requeue = IngestEnvelope { job, attempts };
                    match serde_json::to_string(&requeue) {
                        Ok(json) => {
                            let pushed: Option<redis::RedisResult<usize>> = match redis_conn {
                                Some(c) => Some(c.lpush(INGEST_QUEUE_KEY, json).await),
                                None => None,
                            };
                            match pushed {
                                Some(Ok(_)) => metrics.batches_requeued.inc(),
                                Some(Err(e)) => {
                                    warn!(
                                        batch_id = %requeue.job.batch_id,
                                        error = %e,
                                        "Failed to re-queue ingest job — sweep will recover the file"
                                    );
                                    *redis_conn = None;
                                }
                                None => warn!(
                                    batch_id = %requeue.job.batch_id,
                                    "No Redis connection to re-queue ingest job — sweep will recover the file"
                                ),
                            }
                        }
                        Err(e) => error!(error = %e, "Failed to serialize retry envelope"),
                    }
                }
            }
        }
    }
}

/// Startup / periodic sweep: ingest `*.arrow` spool files older than
/// [`SWEEP_MIN_AGE`] (whose queue doorbells were lost) and clean stale `.tmp`
/// files.
async fn run_sweep(
    config: &IntakeWorkerConfig,
    store: &ExperienceStore,
    redis_conn: &mut Option<redis::aio::ConnectionManager>,
    metrics: &ExperienceMetricsCollector,
) {
    let jobs = sweepable_jobs(&config.spool_dir, SWEEP_MIN_AGE);
    if !jobs.is_empty() {
        info!(
            count = jobs.len(),
            "Sweep found orphaned experience spool files"
        );
    }

    for job in jobs {
        let envelope = IngestEnvelope { job, attempts: 0 };
        ingest_envelope(envelope, config, store, redis_conn, metrics, true).await;
    }

    cleanup_tmp_files(&config.spool_dir, TMP_MAX_AGE);
}

/// Hourly retry of files previously parked in `<spool>/failed/`. Successful
/// files are deleted; failures stay parked for the next round.
async fn retry_failed_files(
    config: &IntakeWorkerConfig,
    store: &ExperienceStore,
    metrics: &ExperienceMetricsCollector,
) {
    let failed_dir = config.spool_dir.join("failed");
    let jobs = sweepable_jobs(&failed_dir, Duration::ZERO);
    if jobs.is_empty() {
        return;
    }

    info!(count = jobs.len(), "Retrying parked experience spool files");

    for job in jobs {
        match handle_ingest(job.clone(), Some(store)).await {
            Ok(m) => {
                if let Err(e) = std::fs::remove_file(&job.shm_path)
                    && e.kind() != std::io::ErrorKind::NotFound
                {
                    warn!(path = %job.shm_path, error = %e, "Failed to delete recovered parked file");
                }
                metrics.batches_ingested.inc();
                metrics.rows_ingested.inc_by(m.rows_ingested as u64);
                metrics.rows_skipped.inc_by(m.rows_skipped as u64);
                metrics.sweep_recovered.inc();
                info!(batch_id = %job.batch_id, "Recovered parked experience batch");
            }
            Err(e) => {
                debug!(
                    batch_id = %job.batch_id,
                    error = %format!("{e:#}"),
                    "Parked batch still failing — leaving in failed/"
                );
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "janus_intake_test_{}_{}_{}",
            tag,
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_parse_bare_ingest_job() {
        let payload = r#"{"batch_id":"b-1","shm_path":"/dev/shm/janus/experience/b-1.arrow"}"#;
        let env = parse_queue_entry(payload).unwrap();
        assert_eq!(env.job.batch_id, "b-1");
        assert_eq!(env.attempts, 0);
    }

    #[test]
    fn test_parse_retry_envelope() {
        let payload = r#"{"job":{"batch_id":"b-2","shm_path":"/tmp/b-2.arrow"},"attempts":2}"#;
        let env = parse_queue_entry(payload).unwrap();
        assert_eq!(env.job.batch_id, "b-2");
        assert_eq!(env.attempts, 2);
    }

    #[test]
    fn test_parse_envelope_without_attempts_defaults_to_zero() {
        let payload = r#"{"job":{"batch_id":"b-3","shm_path":"/tmp/b-3.arrow"}}"#;
        let env = parse_queue_entry(payload).unwrap();
        assert_eq!(env.attempts, 0);
    }

    #[test]
    fn test_parse_garbage_fails() {
        assert!(parse_queue_entry("not json").is_err());
        assert!(parse_queue_entry(r#"{"unrelated":true}"#).is_err());
    }

    #[test]
    fn test_envelope_json_round_trip() {
        let env = IngestEnvelope {
            job: IngestJob {
                batch_id: "b-4".into(),
                shm_path: "/dev/shm/janus/experience/b-4.arrow".into(),
            },
            attempts: 1,
        };
        let json = serde_json::to_string(&env).unwrap();
        let parsed = parse_queue_entry(&json).unwrap();
        assert_eq!(parsed.job.batch_id, "b-4");
        assert_eq!(parsed.attempts, 1);
    }

    #[test]
    fn test_disposition_after_failure_thresholds() {
        assert_eq!(disposition_after_failure(0), FailureDisposition::Requeue(1));
        assert_eq!(disposition_after_failure(1), FailureDisposition::Requeue(2));
        assert_eq!(disposition_after_failure(2), FailureDisposition::Park);
        assert_eq!(disposition_after_failure(99), FailureDisposition::Park);
        assert_eq!(
            disposition_after_failure(u32::MAX),
            FailureDisposition::Park
        );
    }

    #[test]
    fn test_park_file_moves_into_failed_dir() {
        let dir = temp_dir("park");
        let src = dir.join("batch-1.arrow");
        std::fs::write(&src, b"data").unwrap();

        let parked = park_file(&dir, src.to_str().unwrap()).unwrap().unwrap();
        assert!(!src.exists());
        assert_eq!(parked, dir.join("failed").join("batch-1.arrow"));
        assert!(parked.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_park_file_missing_source_is_ok_none() {
        let dir = temp_dir("park_missing");
        let missing = dir.join("nope.arrow");
        let parked = park_file(&dir, missing.to_str().unwrap()).unwrap();
        assert!(parked.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sweepable_jobs_lists_arrow_files_and_skips_others() {
        let dir = temp_dir("sweep");
        std::fs::write(dir.join("100-aaaa.arrow"), b"a").unwrap();
        std::fs::write(dir.join("200-bbbb.arrow"), b"b").unwrap();
        std::fs::write(dir.join("300-cccc.arrow.tmp"), b"c").unwrap();
        std::fs::write(dir.join("notes.txt"), b"d").unwrap();
        std::fs::create_dir_all(dir.join("failed")).unwrap();

        // min_age = 0 so the just-written files qualify
        let jobs = sweepable_jobs(&dir, Duration::ZERO);
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].batch_id, "100-aaaa");
        assert_eq!(jobs[1].batch_id, "200-bbbb");
        assert!(jobs[0].shm_path.ends_with("100-aaaa.arrow"));

        // With a large min_age nothing qualifies
        assert!(sweepable_jobs(&dir, Duration::from_secs(3600)).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sweepable_jobs_missing_dir_is_empty() {
        let dir = std::env::temp_dir().join("janus_intake_test_definitely_missing");
        assert!(sweepable_jobs(&dir, Duration::ZERO).is_empty());
    }

    #[test]
    fn test_cleanup_tmp_files_only_removes_old_tmp() {
        let dir = temp_dir("tmp_cleanup");
        std::fs::write(dir.join("a.arrow.tmp"), b"x").unwrap();
        std::fs::write(dir.join("b.arrow"), b"y").unwrap();

        // Fresh .tmp is untouched with a 1h threshold
        assert_eq!(cleanup_tmp_files(&dir, Duration::from_secs(3600)), 0);
        assert!(dir.join("a.arrow.tmp").exists());

        // With a zero threshold the .tmp goes, the .arrow stays
        assert_eq!(cleanup_tmp_files(&dir, Duration::ZERO), 1);
        assert!(!dir.join("a.arrow.tmp").exists());
        assert!(dir.join("b.arrow").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_spool_dir_default() {
        // Do not mutate the environment (racy); only assert the default when
        // the variable is genuinely absent in the test environment.
        if std::env::var(SPOOL_DIR_ENV).is_err() {
            assert_eq!(spool_dir_from_env(), PathBuf::from(DEFAULT_SPOOL_DIR));
        }
    }

    /// End-to-end queue mechanics against a real Redis: LPUSH a bare job and
    /// a retry envelope, BRPOP both back, parse, and ingest via a mock store.
    #[tokio::test]
    #[ignore = "Requires REDIS_URL environment variable"]
    async fn test_brpop_intake_round_trip() {
        use arrow::array::{
            BinaryArray, BooleanArray, Float32Array, Int64Array, StringArray, UInt8Array,
        };
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::FileWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let redis_url =
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let test_key = format!("janus:test:experience:ingest:{}", uuid::Uuid::new_v4());

        // Write a real 1-row Arrow IPC spool file (9-dim state_gaf)
        let dir = temp_dir("brpop");
        let spool_path = dir.join("test-batch.arrow");
        let schema = Arc::new(Schema::new(vec![
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
        ]));
        let gaf = [0u8; 36];
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(BinaryArray::from_vec(vec![&gaf[..]])),
                Arc::new(BinaryArray::from(vec![None::<&[u8]>])),
                Arc::new(UInt8Array::from(vec![2u8])),
                Arc::new(StringArray::from(vec!["BTCUSDT"])),
                Arc::new(Float32Array::from(vec![0.0f32])),
                Arc::new(Float32Array::from(vec![0.001f32])),
                Arc::new(BinaryArray::from_vec(vec![&gaf[..]])),
                Arc::new(BinaryArray::from(vec![None::<&[u8]>])),
                Arc::new(BooleanArray::from(vec![false])),
                Arc::new(Int64Array::from(vec![1_700_000_000_000i64])),
            ],
        )
        .unwrap();
        {
            let file = std::fs::File::create(&spool_path).unwrap();
            let mut writer = FileWriter::try_new(file, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let client = redis::Client::open(redis_url).unwrap();
        let mut conn = redis::aio::ConnectionManager::new(client).await.unwrap();

        let job = IngestJob {
            batch_id: "test-batch".into(),
            shm_path: spool_path.to_string_lossy().into_owned(),
        };
        let bare = serde_json::to_string(&job).unwrap();
        let wrapped = serde_json::to_string(&IngestEnvelope {
            job: job.clone(),
            attempts: 1,
        })
        .unwrap();

        let _: usize = conn.lpush(&test_key, &bare).await.unwrap();
        let _: usize = conn.lpush(&test_key, &wrapped).await.unwrap();

        // BRPOP returns FIFO for LPUSH+BRPOP: bare entry first
        let (_, p1): (String, String) = conn.brpop(&test_key, 5.0).await.unwrap();
        let e1 = parse_queue_entry(&p1).unwrap();
        assert_eq!(e1.attempts, 0);
        assert_eq!(e1.job.batch_id, "test-batch");

        let (_, p2): (String, String) = conn.brpop(&test_key, 5.0).await.unwrap();
        let e2 = parse_queue_entry(&p2).unwrap();
        assert_eq!(e2.attempts, 1);

        // Timeout path: queue now empty
        let empty: Option<(String, String)> = conn.brpop(&test_key, 0.1).await.unwrap();
        assert!(empty.is_none());

        // Ingest the parsed job against a mock store
        let store = ExperienceStore::mock();
        let m = handle_ingest(e1.job, Some(&store)).await.unwrap();
        assert_eq!(m.rows_ingested, 1);
        assert_eq!(store.mock_point_count().await, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
