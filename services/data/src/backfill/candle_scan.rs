//! Periodic candle-scan reconciler (Track A).
//!
//! Closes the *gap-scan → reconcile → backfill* loop: on a timer, for each
//! configured symbol, it reads the candle open-times already in QuestDB and
//! feeds them to [`GapIntegrationManager::handle_candle_scan`], which detects
//! holes ([`janus_gap_detection::detect_candle_gaps`]), plans bounded backfill
//! windows ([`janus_gap_detection::plan_candle_backfill`]), and enqueues them on
//! the existing scheduler — reusing its filtering, dedup, metrics, and queue.
//!
//! # Design / verification boundary
//!
//! Opt-in (`JANUS_CANDLE_SCAN=1`) and **off by default**. The QuestDB read sits
//! behind the [`CandleTimestampSource`] trait so the orchestration
//! ([`run_scan_once`]) is unit-tested with an in-memory fake — **no live store
//! needed**. The production [`QuestDbCandleSource`] is a thin HTTP wrapper that
//! mirrors the indicator-warmup query path; it is only exercised against a live
//! QuestDB, so going live is a config flip rather than new code.
//!
//! # Not yet wired into startup
//!
//! The backfill scheduler / [`GapIntegrationManager`] are not constructed at
//! service startup today (the whole `backfill` subsystem is dormant library
//! infra). Wiring this loop in means, at startup:
//!
//! ```rust,ignore
//! let manager = Arc::new(GapIntegrationManager::new(cfg, scheduler));
//! let source = Arc::new(QuestDbCandleSource::from_env());
//! candle_scan::spawn(source, manager, CandleScanConfig::from_env());
//! ```
//!
//! That step needs a live QuestDB/Redis stack to verify, so it is intentionally
//! left to the operator rather than blind-wired here.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::backfill::gap_integration::GapIntegrationManager;
use crate::backfill::indicator_warmup::{parse_timestamp, validate_symbol, validate_timeframe};
use janus_gap_detection::ReconcilePlan;

/// Default scan cadence.
const DEFAULT_EVERY_SECS: u64 = 300;
/// Default lookback window per scan.
const DEFAULT_LOOKBACK_SECS: u64 = 86_400;
/// Default bar interval (timeframe string).
const DEFAULT_INTERVAL: &str = "1m";
/// Default exchange label.
const DEFAULT_EXCHANGE: &str = "binance";

/// Source of the candle open-times already present in storage for a symbol.
///
/// Abstracted so [`run_scan_once`] is testable without a live QuestDB. The
/// returned future is `Send` so the loop can run on a multi-threaded runtime.
pub trait CandleTimestampSource: Send + Sync {
    /// Return candle open-times (unix ms, **ascending**) for `symbol` on the
    /// `interval`-width grid (a timeframe string such as `"1m"`) over the
    /// trailing `lookback` window.
    fn candle_timestamps(
        &self,
        exchange: &str,
        symbol: &str,
        interval: &str,
        lookback: Duration,
    ) -> impl std::future::Future<Output = Result<Vec<i64>>> + Send;
}

/// Convert a timeframe string (`"1m"`, `"15m"`, `"1h"`, `"4h"`, `"1d"`, `"1w"`)
/// to its width in milliseconds. Returns `None` for an unparseable or
/// unsupported unit (e.g. `"1M"` months, which is not a fixed grid). Pure.
pub fn timeframe_to_ms(tf: &str) -> Option<i64> {
    let tf = tf.trim();
    let split = tf.find(|c: char| c.is_ascii_alphabetic())?;
    if split == 0 {
        return None; // no numeric part, e.g. "m"
    }
    let (num, unit) = tf.split_at(split);
    let n: i64 = num.parse().ok()?;
    if n <= 0 {
        return None;
    }
    let unit_ms: i64 = match unit {
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        "w" => 604_800_000,
        _ => return None,
    };
    n.checked_mul(unit_ms)
}

/// Configuration for the periodic candle-scan reconciler.
#[derive(Debug, Clone)]
pub struct CandleScanConfig {
    /// Whether the loop runs (`JANUS_CANDLE_SCAN`). Off by default.
    pub enabled: bool,
    /// Exchange label applied to enqueued gaps (`JANUS_CANDLE_SCAN_EXCHANGE`).
    pub exchange: String,
    /// Symbols to scan (`JANUS_CANDLE_SCAN_SYMBOLS`, comma-separated).
    pub symbols: Vec<String>,
    /// Bar interval as a timeframe string (`JANUS_CANDLE_SCAN_INTERVAL`).
    pub interval: String,
    /// Trailing window scanned each pass (`JANUS_CANDLE_SCAN_LOOKBACK_SECS`).
    pub lookback: Duration,
    /// Time between passes (`JANUS_CANDLE_SCAN_EVERY_SECS`).
    pub scan_every: Duration,
    /// Reconciliation tuning (window chunking / filtering / cap).
    pub plan: ReconcilePlan,
}

impl Default for CandleScanConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            exchange: DEFAULT_EXCHANGE.to_string(),
            symbols: Vec::new(),
            interval: DEFAULT_INTERVAL.to_string(),
            lookback: Duration::from_secs(DEFAULT_LOOKBACK_SECS),
            scan_every: Duration::from_secs(DEFAULT_EVERY_SECS),
            plan: ReconcilePlan::default(),
        }
    }
}

impl CandleScanConfig {
    /// Build from the environment.
    pub fn from_env() -> Self {
        let enabled = std::env::var("JANUS_CANDLE_SCAN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let exchange =
            std::env::var("JANUS_CANDLE_SCAN_EXCHANGE").unwrap_or_else(|_| DEFAULT_EXCHANGE.into());
        let symbols = std::env::var("JANUS_CANDLE_SCAN_SYMBOLS")
            .map(|raw| {
                raw.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let interval =
            std::env::var("JANUS_CANDLE_SCAN_INTERVAL").unwrap_or_else(|_| DEFAULT_INTERVAL.into());
        let lookback = Duration::from_secs(
            parse_env_u64("JANUS_CANDLE_SCAN_LOOKBACK_SECS", DEFAULT_LOOKBACK_SECS).max(1),
        );
        let scan_every = Duration::from_secs(
            parse_env_u64("JANUS_CANDLE_SCAN_EVERY_SECS", DEFAULT_EVERY_SECS).max(5),
        );

        let mut plan = ReconcilePlan::default();
        if let Some(v) = parse_env_opt::<u64>("JANUS_CANDLE_SCAN_MIN_MISSING") {
            plan.min_missing = v;
        }
        if let Some(v) = parse_env_opt::<u64>("JANUS_CANDLE_SCAN_MAX_WINDOW") {
            plan.max_candles_per_window = v;
        }
        if let Some(v) = parse_env_opt::<usize>("JANUS_CANDLE_SCAN_MAX_WINDOWS") {
            plan.max_windows = v;
        }

        Self {
            enabled,
            exchange,
            symbols,
            interval,
            lookback,
            scan_every,
            plan,
        }
    }
}

fn parse_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_env_opt<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Outcome of a single scan pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ScanReport {
    /// Symbols whose timestamps were fetched and reconciled.
    pub symbols_scanned: usize,
    /// Symbols whose timestamp fetch errored (logged, non-fatal).
    pub symbols_failed: usize,
    /// Total backfill windows planned (and submitted) across all symbols.
    pub windows_planned: usize,
}

/// Run a single reconcile pass over every configured symbol.
///
/// Pure orchestration over `source` — no timer, no spawning — so it is
/// unit-testable with an in-memory [`CandleTimestampSource`]. A per-symbol fetch
/// error is logged and counted, never fatal: one bad symbol must not stall the
/// rest of the scan.
pub async fn run_scan_once<S: CandleTimestampSource>(
    source: &S,
    manager: &GapIntegrationManager,
    config: &CandleScanConfig,
) -> ScanReport {
    let Some(interval_ms) = timeframe_to_ms(&config.interval) else {
        warn!(
            interval = %config.interval,
            "candle-scan: unsupported interval; skipping pass"
        );
        return ScanReport::default();
    };

    let mut report = ScanReport::default();
    for symbol in &config.symbols {
        match source
            .candle_timestamps(&config.exchange, symbol, &config.interval, config.lookback)
            .await
        {
            Ok(timestamps) => {
                let planned = manager
                    .handle_candle_scan(
                        &config.exchange,
                        symbol,
                        &config.interval,
                        &timestamps,
                        interval_ms,
                        &config.plan,
                    )
                    .await;
                report.symbols_scanned += 1;
                report.windows_planned += planned;
            }
            Err(e) => {
                warn!(symbol = %symbol, error = %e, "candle-scan: timestamp fetch failed");
                report.symbols_failed += 1;
            }
        }
    }
    report
}

/// Spawn the periodic scan loop.
///
/// Returns `None` (and starts nothing) when disabled or when no symbols are
/// configured — the loop is opt-in and never blocks startup.
pub fn spawn<S: CandleTimestampSource + 'static>(
    source: Arc<S>,
    manager: Arc<GapIntegrationManager>,
    config: CandleScanConfig,
) -> Option<JoinHandle<()>> {
    if !config.enabled {
        info!("candle-scan reconciler: disabled (set JANUS_CANDLE_SCAN=1 to enable)");
        return None;
    }
    if config.symbols.is_empty() {
        warn!(
            "candle-scan reconciler: enabled but JANUS_CANDLE_SCAN_SYMBOLS is empty; not starting"
        );
        return None;
    }

    info!(
        symbols = config.symbols.len(),
        interval = %config.interval,
        every_secs = config.scan_every.as_secs(),
        "✅ candle-scan reconciler started"
    );

    Some(tokio::spawn(async move {
        loop {
            let report = run_scan_once(source.as_ref(), manager.as_ref(), &config).await;
            if report.windows_planned > 0 || report.symbols_failed > 0 {
                info!(
                    scanned = report.symbols_scanned,
                    failed = report.symbols_failed,
                    windows = report.windows_planned,
                    "candle-scan pass complete"
                );
            }
            tokio::time::sleep(config.scan_every).await;
        }
    }))
}

/// Production [`CandleTimestampSource`]: reads candle open-times from QuestDB's
/// HTTP `/exec` endpoint, mirroring the indicator-warmup query path. Only
/// constructed when the scan is enabled, and verified against a live store.
pub struct QuestDbCandleSource {
    http_url: String,
    client: reqwest::Client,
}

impl QuestDbCandleSource {
    /// Build from `QUESTDB_HTTP_URL` (default `http://questdb:9000`).
    pub fn from_env() -> Self {
        let http_url =
            std::env::var("QUESTDB_HTTP_URL").unwrap_or_else(|_| "http://questdb:9000".to_string());
        Self::with_url(http_url)
    }

    /// Build against an explicit QuestDB HTTP endpoint (e.g. derived from the
    /// unified binary's `JanusState` config rather than the env default).
    pub fn with_url(http_url: String) -> Self {
        Self {
            http_url,
            client: reqwest::Client::new(),
        }
    }
}

impl CandleTimestampSource for QuestDbCandleSource {
    fn candle_timestamps(
        &self,
        _exchange: &str,
        symbol: &str,
        interval: &str,
        lookback: Duration,
    ) -> impl std::future::Future<Output = Result<Vec<i64>>> + Send {
        // Own everything so the returned future is 'static + Send.
        let http_url = self.http_url.clone();
        let client = self.client.clone();
        let symbol = symbol.to_string();
        let interval = interval.to_string();
        async move {
            // QuestDB's HTTP API has no bind parameters; inputs are validated
            // (alphanumeric symbol, known timeframe) before interpolation.
            let symbol = validate_symbol(&symbol).context("invalid scan symbol")?;
            let interval = validate_timeframe(&interval).context("invalid scan interval")?;
            let lookback_secs = lookback.as_secs().max(1);
            let query = format!(
                "SELECT timestamp FROM candles_crypto \
                 WHERE symbol = '{symbol}' AND interval = '{interval}' \
                 AND timestamp > dateadd('s', -{lookback_secs}, now()) \
                 ORDER BY timestamp ASC"
            );

            let resp = client
                .get(format!("{http_url}/exec"))
                .query(&[("query", query.as_str())])
                .send()
                .await
                .context("QuestDB request failed")?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("QuestDB scan query failed ({status}): {body}");
            }

            let json: serde_json::Value = resp.json().await.context("bad QuestDB JSON")?;
            let dataset = json["dataset"]
                .as_array()
                .context("missing dataset in QuestDB response")?;
            let mut out = Vec::with_capacity(dataset.len());
            for row in dataset {
                let cell = row.get(0).context("empty QuestDB row")?;
                out.push(parse_timestamp(cell).context("bad timestamp cell")?);
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backfill::gap_integration::GapIntegrationConfig;
    use crate::backfill::scheduler::BackfillScheduler;
    use crate::backfill::throttle::{BackfillThrottle, ThrottleConfig};
    use crate::backfill::{BackfillLock, lock::LockMetrics};

    const MIN_MS: i64 = 60_000;

    fn test_manager() -> Arc<GapIntegrationManager> {
        let throttle = Arc::new(BackfillThrottle::new(ThrottleConfig::default()));
        let redis_client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let lock = Arc::new(BackfillLock::new(
            redis_client,
            Default::default(),
            Arc::new(LockMetrics::new(&prometheus::Registry::new()).unwrap()),
        ));
        let scheduler = Arc::new(BackfillScheduler::new(Default::default(), throttle, lock));
        Arc::new(GapIntegrationManager::new(
            GapIntegrationConfig::default(),
            scheduler,
        ))
    }

    /// Fake source that returns the same series for every symbol.
    struct FakeSource(Vec<i64>);
    impl CandleTimestampSource for FakeSource {
        fn candle_timestamps(
            &self,
            _e: &str,
            _s: &str,
            _i: &str,
            _l: Duration,
        ) -> impl std::future::Future<Output = Result<Vec<i64>>> + Send {
            let ts = self.0.clone();
            async move { Ok(ts) }
        }
    }

    /// Fake source that always errors.
    struct FailSource;
    impl CandleTimestampSource for FailSource {
        fn candle_timestamps(
            &self,
            _e: &str,
            _s: &str,
            _i: &str,
            _l: Duration,
        ) -> impl std::future::Future<Output = Result<Vec<i64>>> + Send {
            std::future::ready(Err(anyhow::anyhow!("source unavailable")))
        }
    }

    fn scan_config(symbols: &[&str]) -> CandleScanConfig {
        CandleScanConfig {
            enabled: true,
            symbols: symbols.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn timeframe_to_ms_known_units() {
        assert_eq!(timeframe_to_ms("1m"), Some(60_000));
        assert_eq!(timeframe_to_ms("5m"), Some(300_000));
        assert_eq!(timeframe_to_ms("15m"), Some(900_000));
        assert_eq!(timeframe_to_ms("1h"), Some(3_600_000));
        assert_eq!(timeframe_to_ms("4h"), Some(14_400_000));
        assert_eq!(timeframe_to_ms("1d"), Some(86_400_000));
        assert_eq!(timeframe_to_ms("1w"), Some(604_800_000));
    }

    #[test]
    fn timeframe_to_ms_rejects_bad() {
        assert_eq!(timeframe_to_ms("1M"), None); // months: not a fixed grid
        assert_eq!(timeframe_to_ms("m"), None); // no number
        assert_eq!(timeframe_to_ms("10"), None); // no unit
        assert_eq!(timeframe_to_ms("0m"), None); // zero
        assert_eq!(timeframe_to_ms("zzz"), None);
        assert_eq!(timeframe_to_ms(""), None);
    }

    #[tokio::test]
    async fn run_scan_once_plans_and_submits() {
        let manager = test_manager();
        // 1-minute candles with a 3-candle hole between 1·MIN and 5·MIN.
        let source = FakeSource(vec![0, MIN_MS, 5 * MIN_MS, 6 * MIN_MS]);
        let config = scan_config(&["BTCUSDT"]);

        let report = run_scan_once(&source, &manager, &config).await;

        assert_eq!(report.symbols_scanned, 1);
        assert_eq!(report.symbols_failed, 0);
        assert_eq!(report.windows_planned, 1);
    }

    #[tokio::test]
    async fn run_scan_once_handles_multiple_symbols() {
        let manager = test_manager();
        let source = FakeSource(vec![0, MIN_MS, 5 * MIN_MS, 6 * MIN_MS]);
        let config = scan_config(&["BTCUSDT", "ETHUSDT"]);

        let report = run_scan_once(&source, &manager, &config).await;

        assert_eq!(report.symbols_scanned, 2);
        assert_eq!(report.windows_planned, 2);
    }

    #[tokio::test]
    async fn run_scan_once_counts_failures() {
        let manager = test_manager();
        let config = scan_config(&["BTCUSDT", "ETHUSDT"]);

        let report = run_scan_once(&FailSource, &manager, &config).await;

        assert_eq!(report.symbols_scanned, 0);
        assert_eq!(report.symbols_failed, 2);
        assert_eq!(report.windows_planned, 0);
    }

    #[tokio::test]
    async fn run_scan_once_unsupported_interval_is_noop() {
        let manager = test_manager();
        let source = FakeSource(vec![0, MIN_MS, 5 * MIN_MS]);
        let config = CandleScanConfig {
            interval: "bogus".to_string(),
            ..scan_config(&["BTCUSDT"])
        };

        let report = run_scan_once(&source, &manager, &config).await;

        assert_eq!(report, ScanReport::default());
    }

    #[test]
    fn config_from_env_defaults_disabled() {
        // No env set in this test process ⇒ disabled with sane defaults.
        let c = CandleScanConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.interval, "1m");
        assert!(c.symbols.is_empty());
        assert_eq!(c.scan_every, Duration::from_secs(300));
    }

    #[tokio::test]
    async fn spawn_returns_none_when_disabled_or_empty() {
        let manager = test_manager();
        let source = Arc::new(FakeSource(vec![]));

        // Disabled.
        let disabled = CandleScanConfig::default();
        assert!(spawn(source.clone(), manager.clone(), disabled).is_none());

        // Enabled but no symbols.
        let no_symbols = CandleScanConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(spawn(source, manager, no_symbols).is_none());
    }
}
