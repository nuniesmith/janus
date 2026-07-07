//! Unified-binary backfill wiring (roadmap P1: "Backfill goes live").
//!
//! The unified JANUS binary's live path (`DATA_SOURCE=live` in
//! [`crate::start_module`]) ingests Binance WS klines forward-only: QuestDB
//! `candles_crypto` starts empty and only accumulates from process start.
//! This module gives the unified path the two missing halves of the dormant
//! backfill subsystem:
//!
//! 1. **One-shot deep backfill** ([`run_deep_backfill`]): on startup, for
//!    every configured symbol × kline interval, fetch historical klines from
//!    Binance REST back to `JANUS_BACKFILL_DAYS` days (default 30, `0`
//!    disables) and persist them to the same `candles_crypto` table the live
//!    sink writes, page by page (≤1000 candles in memory per pair at a time).
//!
//! 2. **Candle gap filling** ([`CandleRangeBackfiller`] +
//!    [`make_candle_gap_filler`]): the repair half of the candle-scan →
//!    scheduler loop. The scan enqueues missing candle windows as
//!    [`GapInfo`]s carrying their interval; the scheduler calls back into
//!    [`CandleRangeBackfiller::backfill_range`] to fetch and persist exactly
//!    the missing window — under the scheduler's existing Redis lock,
//!    throttle, retry, and metrics machinery.
//!
//! ## Idempotency / dedup (verified 2026-07)
//!
//! `candles_crypto` has **no `DEDUP UPSERT KEYS`** on either of its possible
//! origins: the fks `infrastructure/config/questdb/init.sql` DDL declares
//! plain `TIMESTAMP(ts) PARTITION BY DAY WAL`, and an ILP-auto-created table
//! (the deployed reality — the live sink creates it on first write, and both
//! this crate's scan queries and the fks-web adapter address the designated
//! column as `timestamp`) has no dedup keys by default either. QuestDB will
//! happily store duplicate rows for the same
//! `(timestamp, symbol, exchange, interval)` and charts would double-count
//! them. Re-run safety therefore lives here: before writing,
//! [`CandleRangeBackfiller::backfill_range`] reads the candle open-times
//! already stored for the pair over the target window and skips any fetched
//! candle whose open-time is already present ([`partition_new_candles`],
//! counted on `backfill_dedup_hits_total` / `backfill_dedup_misses_total`).
//! This also makes backfill coexist with the live sink: whatever the sink
//! already wrote is coverage, not conflict. Enabling table-level dedup
//! (`ALTER TABLE candles_crypto DEDUP ENABLE UPSERT KEYS(timestamp, symbol,
//! exchange, interval)`) is a recommended belt-and-braces follow-up in the
//! fks repo, but is deliberately not issued from here — schema ownership
//! stays with the infra repo, and code-level idempotency must hold anyway
//! for the fleet of already-created tables.
//!
//! ## Isolation
//!
//! Everything here runs in dedicated tokio tasks spawned from
//! [`crate::start_module`]'s live path via [`spawn_backfill_and_scan`].
//! Errors are logged and counted
//! (`backfills_completed_total{status="failed"}`), never propagated into the
//! WebSocket ingestion path. Memory is bounded: one REST page (≤1000
//! candles) plus one coverage timestamp set per in-flight range, and the
//! shared [`BackfillThrottle`] gates concurrency and disk headroom.
//!
//! ## The completeness SLI is unaffected
//!
//! `data_completeness_percent` is computed purely from in-process WebSocket
//! counters (closed klines received vs wall-clock boundary expectations in
//! [`crate::start_module`]'s health loop) — it never reads QuestDB, so
//! historical rows appearing in `candles_crypto` cannot skew its freshness
//! expectation in either direction.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::actors::CandleData;
use crate::actors::indicator::CandleInput;
use crate::backfill::candle_scan::{
    self, CandleScanConfig, CandleTimestampSource, QuestDbCandleSource, timeframe_to_ms,
};
use crate::backfill::gap_integration::{GapIntegrationConfig, GapIntegrationManager};
use crate::backfill::historical_candles::HistoricalCandleFetcher;
use crate::backfill::lock::{BackfillLock, LockMetrics};
use crate::backfill::scheduler::{BackfillScheduler, CandleGapFiller, GapInfo};
use crate::backfill::throttle::{BackfillThrottle, ThrottleError};
use crate::metrics::prometheus_exporter::{
    BACKFILL_DEDUP_HITS, BACKFILL_DEDUP_MISSES, BACKFILL_DURATION, BACKFILL_THROTTLE_REJECTIONS,
    BACKFILLS_COMPLETED, QUESTDB_WRITES,
};
use crate::storage::IlpWriter;

/// Table both the live sink and the backfill write to.
const TABLE: &str = "candles_crypto";

/// Candles per Binance klines request (their hard limit is 1000).
const PAGE_LIMIT: usize = 1000;

/// Delay between paginated REST requests, so a full 30-day × N-pair deep
/// backfill trickles well under Binance's request weight limits.
const PAGE_DELAY_MS: u64 = 250;

/// Delay between symbol×interval pairs during the deep backfill.
const PAIR_DELAY_MS: u64 = 500;

// ═══════════════════════════════════════════════════════════════════════════
// Pure helpers (unit-tested)
// ═══════════════════════════════════════════════════════════════════════════

/// Parse `JANUS_BACKFILL_DAYS`: default 30 when unset or unparseable,
/// `0` = deep backfill disabled.
pub fn parse_backfill_days(raw: Option<&str>) -> u32 {
    match raw {
        None => 30,
        Some(v) => match v.trim().parse::<u32>() {
            Ok(n) => n,
            Err(_) => {
                warn!("JANUS_BACKFILL_DAYS='{v}' is not a non-negative integer — using default 30");
                30
            }
        },
    }
}

/// Number of candles a `days`-deep backfill spans on an `interval_ms` grid.
pub fn candles_in_days(days: u32, interval_ms: i64) -> u64 {
    if interval_ms <= 0 {
        return 0;
    }
    (days as u64) * 86_400_000 / (interval_ms as u64)
}

/// End (exclusive, unix ms) of a deep-backfill range on an `interval_ms`
/// grid: the open-time of the candle currently forming. Binance klines are
/// grid-aligned, so ending here excludes the in-progress candle — a deep
/// backfill must never persist a partially-formed candle (the live sink
/// writes it when it closes; writing both would duplicate the row with
/// bogus interim OHLCV).
pub fn deep_range_end(now_ms: i64, interval_ms: i64) -> i64 {
    if interval_ms <= 0 {
        return now_ms;
    }
    (now_ms / interval_ms) * interval_ms
}

/// Dedupe/idempotency guard: split a fetched page into the candles that are
/// genuinely missing from storage and a count of skipped ones.
///
/// A candle is kept only if its open-time is inside `[start_ms, end_ms)`
/// **and** not already present in `existing` (the open-times QuestDB already
/// holds — the table has no dedup keys, so writing it again would duplicate
/// the row). Returns `(missing, duplicates_skipped)`; candles outside the
/// window are dropped without counting as duplicates.
pub fn partition_new_candles(
    page: Vec<CandleInput>,
    existing: &HashSet<i64>,
    start_ms: i64,
    end_ms: i64,
) -> (Vec<CandleInput>, usize) {
    let mut missing = Vec::with_capacity(page.len());
    let mut duplicates = 0usize;
    for c in page {
        if c.timestamp < start_ms || c.timestamp >= end_ms {
            continue;
        }
        if existing.contains(&c.timestamp) {
            duplicates += 1;
            continue;
        }
        missing.push(c);
    }
    (missing, duplicates)
}

/// Convert a fetched historical kline into the exact row shape the live
/// candle sink writes (same table, same tags), so backfilled rows and live
/// rows form one series.
fn candle_input_to_data(c: &CandleInput, exchange: &str, interval_ms: i64) -> CandleData {
    CandleData {
        symbol: c.symbol.to_uppercase(),
        exchange: exchange.to_string(),
        open_time: c.timestamp,
        close_time: c.timestamp + interval_ms - 1,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume,
        interval: c.timeframe.clone(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CandleRangeBackfiller — fetch + dedupe + persist one candle range
// ═══════════════════════════════════════════════════════════════════════════

/// Fetches historical klines for an arbitrary `[start_ms, end_ms)` window and
/// persists the ones QuestDB does not already have. Shared by the one-shot
/// deep backfill and the scheduler's candle-gap filler.
pub struct CandleRangeBackfiller {
    fetcher: HistoricalCandleFetcher,
    source: Arc<QuestDbCandleSource>,
    questdb_host: String,
    ilp_port: u16,
    exchange: String,
}

impl CandleRangeBackfiller {
    pub fn new(
        source: Arc<QuestDbCandleSource>,
        questdb_host: String,
        ilp_port: u16,
        exchange: String,
    ) -> Self {
        Self {
            fetcher: HistoricalCandleFetcher::new(),
            source,
            questdb_host,
            ilp_port,
            exchange,
        }
    }

    /// Open-times already stored for `symbol`/`interval` covering
    /// `[start_ms, now]`, as a set for the dedupe guard.
    ///
    /// This read is what makes re-runs safe (the table has no dedup keys —
    /// see module docs), so a failure here aborts the range rather than
    /// risking duplicate rows.
    async fn existing_timestamps(
        &self,
        symbol: &str,
        interval: &str,
        start_ms: i64,
    ) -> Result<HashSet<i64>> {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let lookback_secs = ((now_ms - start_ms) / 1000).max(1) as u64;
        let timestamps = match self
            .source
            .candle_timestamps(
                &self.exchange,
                symbol,
                interval,
                Duration::from_secs(lookback_secs),
            )
            .await
        {
            Ok(t) => t,
            // Fresh deployment: the live sink ILP-creates the table on its
            // first closed kline, which may land after we start. No table ⇒
            // no rows ⇒ empty coverage; our own ILP writes create it too.
            Err(e) if e.to_string().contains("table does not exist") => {
                info!(
                    symbol,
                    interval, "dedupe guard: candles table absent — treating coverage as empty"
                );
                Vec::new()
            }
            Err(e) => {
                return Err(e).context("dedupe guard: failed to read existing candle timestamps");
            }
        };
        Ok(timestamps.into_iter().collect())
    }

    /// Backfill `[start_ms, end_ms)` for one symbol × interval, newest page
    /// first, skipping candles already stored. Returns the number of candles
    /// written. Memory stays bounded at one REST page (≤1000 candles) plus
    /// the timestamp set.
    /// (`backfills_running` is tracked by [`BackfillThrottle::execute_backfill`],
    /// which both the deep backfill and the scheduler wrap this call in.)
    pub async fn backfill_range(
        &self,
        symbol: &str,
        interval: &str,
        start_ms: i64,
        end_ms: i64,
    ) -> Result<u64> {
        let interval_ms = timeframe_to_ms(interval)
            .ok_or_else(|| anyhow!("unsupported backfill interval '{interval}'"))?;
        if end_ms <= start_ms {
            return Ok(0);
        }

        let existing = self.existing_timestamps(symbol, interval, start_ms).await?;
        debug!(
            symbol,
            interval,
            existing = existing.len(),
            "backfill_range: dedupe guard loaded"
        );

        let mut writer = IlpWriter::new(&self.questdb_host, self.ilp_port, PAGE_LIMIT, 100)
            .await
            .context("backfill: QuestDB ILP connect failed")?;

        // Hard page cap: the expected number of pages plus slack. Guards
        // against an API regression (e.g. endTime ignored) looping forever.
        let expected_candles = ((end_ms - start_ms) / interval_ms) as u64 + 1;
        let max_pages = expected_candles / PAGE_LIMIT as u64 + 4;

        let mut written: u64 = 0;
        let mut duplicates: u64 = 0;
        let mut end_cursor = end_ms - 1; // Binance endTime is inclusive of open-time
        let mut pages: u64 = 0;

        loop {
            pages += 1;
            if pages > max_pages {
                warn!(
                    symbol,
                    interval, pages, "backfill_range: page cap reached — stopping this range"
                );
                break;
            }

            let page = self
                .fetcher
                .fetch_candles_page(symbol, interval, PAGE_LIMIT, Some(end_cursor))
                .await
                .context("backfill: Binance klines fetch failed")?;
            let Some(oldest) = page.first().map(|c| c.timestamp) else {
                break; // no more history available
            };

            let (missing, dups) = partition_new_candles(page, &existing, start_ms, end_ms);
            duplicates += dups as u64;
            BACKFILL_DEDUP_HITS.inc_by(dups as f64);
            BACKFILL_DEDUP_MISSES.inc_by(missing.len() as f64);

            for candle in &missing {
                let data = candle_input_to_data(candle, &self.exchange, interval_ms);
                writer.write_candle(&data).await?;
            }
            if !missing.is_empty() {
                writer.flush().await.context("backfill: ILP flush failed")?;
                QUESTDB_WRITES
                    .with_label_values(&[TABLE])
                    .inc_by(missing.len() as f64);
                written += missing.len() as u64;
            }

            if oldest <= start_ms {
                break; // walked past the start of the window
            }
            end_cursor = oldest - 1;
            tokio::time::sleep(Duration::from_millis(PAGE_DELAY_MS)).await;
        }

        info!(
            symbol,
            interval, written, duplicates, "backfill_range complete"
        );
        Ok(written)
    }
}

/// Wrap a [`CandleRangeBackfiller`] as the scheduler's candle-gap filler
/// callback: the scheduler hands over a [`GapInfo`] carrying its interval
/// (set by the candle scan) and gets back the number of candles written.
pub fn make_candle_gap_filler(backfiller: Arc<CandleRangeBackfiller>) -> CandleGapFiller {
    Arc::new(move |gap: GapInfo| {
        let backfiller = backfiller.clone();
        Box::pin(async move {
            let interval = gap
                .interval
                .clone()
                .ok_or_else(|| anyhow!("candle gap without interval"))?;
            backfiller
                .backfill_range(
                    &gap.symbol,
                    &interval,
                    gap.start_time.timestamp_millis(),
                    gap.end_time.timestamp_millis(),
                )
                .await
        })
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// One-shot deep backfill
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the one-shot deep backfill.
#[derive(Debug, Clone)]
pub struct DeepBackfillConfig {
    /// How far back to fetch (`JANUS_BACKFILL_DAYS`, default 30, 0 = off).
    pub days: u32,
    /// Exchange label written to QuestDB rows (matches the live sink).
    pub exchange: String,
    /// Exchange-form symbols (e.g. `BTCUSDT`).
    pub symbols: Vec<String>,
    /// Kline intervals (from `DATA_KLINE_INTERVALS`, e.g. `["1m", "5m"]`).
    pub intervals: Vec<String>,
}

/// Run the one-shot deep backfill: every symbol × interval, sequentially
/// (bounded memory, gentle on Binance), each pair wrapped in the shared
/// throttle (disk gate + concurrency slot) and a best-effort Redis lock so
/// two instances never double-fetch the same pair. Errors are logged and
/// counted per pair; one bad pair never stops the rest.
pub async fn run_deep_backfill(
    backfiller: Arc<CandleRangeBackfiller>,
    throttle: Arc<BackfillThrottle>,
    lock: Arc<BackfillLock>,
    config: DeepBackfillConfig,
) {
    if config.days == 0 {
        info!("deep backfill: disabled (JANUS_BACKFILL_DAYS=0)");
        return;
    }

    let now_ms = chrono::Utc::now().timestamp_millis();
    let start_ms = now_ms - (config.days as i64) * 86_400_000;
    let pair_count = config.symbols.len() * config.intervals.len();

    info!(
        days = config.days,
        symbols = config.symbols.len(),
        intervals = ?config.intervals,
        pairs = pair_count,
        "🕰️  deep backfill starting (one-shot, sequential)"
    );

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut total_written: u64 = 0;

    for symbol in &config.symbols {
        for interval in &config.intervals {
            let Some(interval_ms) = timeframe_to_ms(interval) else {
                warn!(
                    symbol,
                    interval, "deep backfill: unsupported interval — skipping pair"
                );
                failed += 1;
                continue;
            };
            let estimated = candles_in_days(config.days, interval_ms) as usize;
            // Exclusive end at the current grid boundary: never persist the
            // in-progress candle (the live sink writes it when it closes).
            let end_ms = deep_range_end(now_ms, interval_ms);

            // Best-effort distributed lock: skip if another instance holds it;
            // proceed with a warning if Redis is unreachable (the dedupe guard
            // still keeps the write idempotent).
            let lock_key = format!("deep:{}:{}:{}", config.exchange, symbol, interval);
            let guard = match lock.acquire(&lock_key).await {
                Ok(Some(g)) => Some(g),
                Ok(None) => {
                    info!(
                        symbol,
                        interval, "deep backfill: pair locked by another instance — skipping"
                    );
                    continue;
                }
                Err(e) => {
                    warn!(symbol, interval, error = %e, "deep backfill: lock unavailable — proceeding unlocked");
                    None
                }
            };

            let started = std::time::Instant::now();
            let result = throttle
                .execute_backfill(estimated, || {
                    backfiller.backfill_range(symbol, interval, start_ms, end_ms)
                })
                .await;
            drop(guard);

            match result {
                Ok(written) => {
                    succeeded += 1;
                    total_written += written;
                    BACKFILLS_COMPLETED
                        .with_label_values(&[config.exchange.as_str(), symbol.as_str(), "success"])
                        .inc();
                    BACKFILL_DURATION
                        .with_label_values(&[config.exchange.as_str(), symbol.as_str()])
                        .observe(started.elapsed().as_secs_f64());
                }
                Err(e) => {
                    failed += 1;
                    if let ThrottleError::DiskFull { .. } = e {
                        BACKFILL_THROTTLE_REJECTIONS
                            .with_label_values(&["disk_full"])
                            .inc();
                    }
                    BACKFILLS_COMPLETED
                        .with_label_values(&[config.exchange.as_str(), symbol.as_str(), "failed"])
                        .inc();
                    warn!(symbol, interval, error = %e, "deep backfill: pair failed (gap scan will retry the holes)");
                }
            }

            tokio::time::sleep(Duration::from_millis(PAIR_DELAY_MS)).await;
        }
    }

    info!(
        succeeded,
        failed, total_written, "🕰️  deep backfill finished"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Orchestration — entry point called from the unified live path
// ═══════════════════════════════════════════════════════════════════════════

/// Everything the unified live path passes down from `JanusState` + env.
#[derive(Debug, Clone)]
pub struct UnifiedBackfillParams {
    /// Exchange label (matches the live sink's rows).
    pub exchange: String,
    /// Exchange-form symbols the live path ingests (e.g. `BTCUSDT`).
    pub symbols: Vec<String>,
    /// Live kline intervals (`DATA_KLINE_INTERVALS`).
    pub intervals: Vec<String>,
    /// QuestDB ILP endpoint (writes).
    pub questdb_host: String,
    pub ilp_port: u16,
    /// QuestDB HTTP endpoint (coverage/scan reads).
    pub questdb_http_url: String,
    /// Redis URL for the distributed backfill lock.
    pub redis_url: String,
}

/// Parse `JANUS_CANDLE_SCAN` for the unified binary: **default ON** (the
/// roadmap flips the default here; the standalone data binary keeps its
/// opt-in default via [`CandleScanConfig::from_env`]). Explicit
/// `0`/`false`/`off`/`no` disables.
pub fn parse_scan_enabled(raw: Option<&str>) -> bool {
    match raw {
        None => true,
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
    }
}

/// Which intervals the gap scanner covers: an explicitly set
/// `JANUS_CANDLE_SCAN_INTERVAL` wins (env-continuity with the standalone
/// binary), otherwise every live kline interval gets its own scan loop.
pub fn scan_intervals(env_interval: Option<&str>, live_intervals: &[String]) -> Vec<String> {
    match env_interval {
        Some(i) if !i.trim().is_empty() => vec![i.trim().to_string()],
        _ => live_intervals.to_vec(),
    }
}

/// Spawn the unified backfill pipeline: the one-shot deep backfill first,
/// then (unless `JANUS_CANDLE_SCAN` is off) the periodic candle-scan →
/// scheduler → candle-gap-filler loop.
///
/// Everything runs in detached tokio tasks — construction failures and
/// runtime errors are logged and counted, never surfaced to the live
/// WebSocket ingestion path. Returns `None` (spawning nothing) when both
/// halves are disabled or there is nothing to backfill.
pub fn spawn_backfill_and_scan(params: UnifiedBackfillParams) -> Option<JoinHandle<()>> {
    let days = parse_backfill_days(std::env::var("JANUS_BACKFILL_DAYS").ok().as_deref());
    let scan_enabled = parse_scan_enabled(std::env::var("JANUS_CANDLE_SCAN").ok().as_deref());

    if days == 0 && !scan_enabled {
        info!("backfill: deep backfill and gap scan both disabled — nothing to do");
        return None;
    }
    if params.symbols.is_empty() || params.intervals.is_empty() {
        warn!("backfill: no symbols/intervals configured — not starting");
        return None;
    }

    // Shared machinery. The Redis client is lazy (no connect here); a
    // malformed URL is a config error worth failing the whole subsystem for.
    let redis_client = match redis::Client::open(params.redis_url.as_str()) {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "backfill: invalid Redis URL — backfill subsystem not started");
            return None;
        }
    };
    // Lock metrics live on a dedicated registry (not the main /metrics
    // scrape) — same trade-off as the standalone binary's scan wiring.
    let lock_metrics = match LockMetrics::new(&prometheus::Registry::new()) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            warn!(error = %e, "backfill: lock metrics init failed — backfill subsystem not started");
            return None;
        }
    };
    let throttle = Arc::new(BackfillThrottle::new(Default::default()));
    let lock = Arc::new(BackfillLock::new(
        redis_client,
        Default::default(),
        lock_metrics,
    ));
    let source = Arc::new(QuestDbCandleSource::with_url(
        params.questdb_http_url.clone(),
    ));
    let backfiller = Arc::new(CandleRangeBackfiller::new(
        source.clone(),
        params.questdb_host.clone(),
        params.ilp_port,
        params.exchange.clone(),
    ));

    Some(tokio::spawn(async move {
        // ── Phase 1: one-shot deep backfill ──────────────────────────────
        run_deep_backfill(
            backfiller.clone(),
            throttle.clone(),
            lock.clone(),
            DeepBackfillConfig {
                days,
                exchange: params.exchange.clone(),
                symbols: params.symbols.clone(),
                intervals: params.intervals.clone(),
            },
        )
        .await;

        // ── Phase 2: periodic gap scan → scheduler → candle filler ───────
        if !scan_enabled {
            info!("backfill: gap scanner disabled (JANUS_CANDLE_SCAN=0)");
            return;
        }

        let scheduler = Arc::new(
            BackfillScheduler::new(Default::default(), throttle, lock)
                .with_candle_filler(make_candle_gap_filler(backfiller)),
        );
        {
            let scheduler = scheduler.clone();
            tokio::spawn(async move {
                if let Err(e) = scheduler.run().await {
                    warn!(error = %e, "backfill scheduler exited with error");
                }
            });
        }
        let manager = Arc::new(GapIntegrationManager::new(
            GapIntegrationConfig::default(),
            scheduler,
        ));

        // Env-continuity: all JANUS_CANDLE_SCAN_* tuning knobs are honoured;
        // unified defaults fill what the env leaves unset (symbols/exchange
        // from the live config, one scan loop per live kline interval).
        let mut base = CandleScanConfig::from_env();
        base.enabled = true;
        if base.symbols.is_empty() {
            base.symbols = params.symbols.clone();
        }
        if std::env::var("JANUS_CANDLE_SCAN_EXCHANGE").is_err() {
            base.exchange = params.exchange.clone();
        }

        let intervals = scan_intervals(
            std::env::var("JANUS_CANDLE_SCAN_INTERVAL").ok().as_deref(),
            &params.intervals,
        );
        for interval in intervals {
            let mut config = base.clone();
            config.interval = interval;
            // Fire-and-forget: the loop runs for the process lifetime.
            let _handle = candle_scan::spawn(source.clone(), manager.clone(), config);
        }
    }))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn candle(ts: i64) -> CandleInput {
        CandleInput {
            symbol: "BTCUSDT".to_string(),
            exchange: "binance".to_string(),
            timeframe: "1m".to_string(),
            timestamp: ts,
            open: 1.0,
            high: 2.0,
            low: 0.5,
            close: 1.5,
            volume: 10.0,
        }
    }

    #[test]
    fn backfill_days_defaults_and_parses() {
        assert_eq!(parse_backfill_days(None), 30);
        assert_eq!(parse_backfill_days(Some("30")), 30);
        assert_eq!(parse_backfill_days(Some("7")), 7);
        assert_eq!(parse_backfill_days(Some(" 90 ")), 90);
        // 0 is the documented off-switch.
        assert_eq!(parse_backfill_days(Some("0")), 0);
        // Garbage / negative → default, never a panic.
        assert_eq!(parse_backfill_days(Some("abc")), 30);
        assert_eq!(parse_backfill_days(Some("-5")), 30);
        assert_eq!(parse_backfill_days(Some("")), 30);
    }

    #[test]
    fn candles_in_days_math() {
        // 30 days of 1m candles: 30 × 1440 = 43_200.
        assert_eq!(candles_in_days(30, 60_000), 43_200);
        // 30 days of 5m candles: 8_640.
        assert_eq!(candles_in_days(30, 300_000), 8_640);
        // 30 days of 1h candles: 720.
        assert_eq!(candles_in_days(30, 3_600_000), 720);
        // Off-switch and degenerate interval.
        assert_eq!(candles_in_days(0, 60_000), 0);
        assert_eq!(candles_in_days(30, 0), 0);
        assert_eq!(candles_in_days(30, -1), 0);
    }

    #[test]
    fn dedupe_guard_skips_existing_and_out_of_window() {
        let min = 60_000i64;
        let existing: HashSet<i64> = [min, 3 * min].into_iter().collect();
        let page = vec![
            candle(0),       // before window → dropped silently
            candle(min),     // in window but already stored → duplicate
            candle(2 * min), // missing → kept
            candle(3 * min), // already stored → duplicate
            candle(4 * min), // missing → kept
            candle(5 * min), // == end (exclusive) → dropped silently
        ];

        let (missing, dups) = partition_new_candles(page, &existing, min, 5 * min);

        assert_eq!(
            missing.iter().map(|c| c.timestamp).collect::<Vec<_>>(),
            vec![2 * min, 4 * min]
        );
        assert_eq!(dups, 2);
    }

    #[test]
    fn dedupe_guard_rerun_is_noop() {
        // Second run of the same window: everything already stored → nothing
        // written, everything counted as duplicate. This is the re-run safety
        // the missing QuestDB DEDUP keys can't give us.
        let min = 60_000i64;
        let existing: HashSet<i64> = (1..=4).map(|i| i * min).collect();
        let page: Vec<CandleInput> = (1..=4).map(|i| candle(i * min)).collect();

        let (missing, dups) = partition_new_candles(page, &existing, min, 5 * min);

        assert!(missing.is_empty());
        assert_eq!(dups, 4);
    }

    #[test]
    fn deep_range_end_excludes_in_progress_candle() {
        let min = 60_000i64;
        // Mid-candle: clamp back to the boundary (the forming candle's open).
        assert_eq!(deep_range_end(10 * min + 42_000, min), 10 * min);
        // Exactly on a boundary: that candle has only just opened — excluded.
        assert_eq!(deep_range_end(10 * min, min), 10 * min);
        // Larger grid.
        assert_eq!(deep_range_end(7 * 300_000 + 1, 300_000), 7 * 300_000);
        // Degenerate interval: pass through rather than divide by zero.
        assert_eq!(deep_range_end(123, 0), 123);
    }

    #[test]
    fn scan_enabled_defaults_on_in_unified() {
        // Default ON — the roadmap flips the standalone binary's opt-in.
        assert!(parse_scan_enabled(None));
        assert!(parse_scan_enabled(Some("1")));
        assert!(parse_scan_enabled(Some("true")));
        assert!(parse_scan_enabled(Some("anything-else")));
        // Explicit off-switches.
        assert!(!parse_scan_enabled(Some("0")));
        assert!(!parse_scan_enabled(Some("false")));
        assert!(!parse_scan_enabled(Some("FALSE")));
        assert!(!parse_scan_enabled(Some(" off ")));
        assert!(!parse_scan_enabled(Some("no")));
    }

    #[test]
    fn scan_intervals_env_override_wins() {
        let live = vec!["1m".to_string(), "5m".to_string()];
        // No env override: one scan per live kline interval.
        assert_eq!(scan_intervals(None, &live), live);
        assert_eq!(scan_intervals(Some(""), &live), live);
        assert_eq!(scan_intervals(Some("  "), &live), live);
        // Explicit JANUS_CANDLE_SCAN_INTERVAL: exactly that one.
        assert_eq!(scan_intervals(Some("15m"), &live), vec!["15m".to_string()]);
        assert_eq!(scan_intervals(Some(" 1h "), &live), vec!["1h".to_string()]);
    }

    #[test]
    fn candle_conversion_matches_live_sink_shape() {
        let c = candle(1_700_000_000_000);
        let data = candle_input_to_data(&c, "binance", 60_000);
        assert_eq!(data.symbol, "BTCUSDT");
        assert_eq!(data.exchange, "binance");
        assert_eq!(data.interval, "1m");
        assert_eq!(data.open_time, 1_700_000_000_000);
        assert_eq!(data.close_time, 1_700_000_000_000 + 59_999);
        assert_eq!(data.open, 1.0);
        assert_eq!(data.volume, 10.0);
    }
}
