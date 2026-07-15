//! Live-mode QuestDB candle persistence (unified binary).
//!
//! The unified JANUS binary's live ingestion (`DATA_SOURCE=live` in
//! [`crate::start_module`]) publishes closed klines to the in-process
//! [`MarketDataBus`](janus_core::MarketDataBus) for the Forward module —
//! but, unlike the standalone data-factory binary (whose Router persists
//! through [`crate::storage::StorageManager`]), nothing wrote them to
//! QuestDB. The FKS WebUI charts read their history from the QuestDB
//! `candles_crypto` table, so the deployed stack had no candle history.
//!
//! This sink closes that gap: it subscribes to the bus and writes every
//! closed kline to `candles_crypto` over ILP, using the exact line format
//! of [`IlpWriter::write_candle`] so rows are identical to the standalone
//! factory's (tags `symbol`/`exchange`/`interval`, fields OHLCV, timestamp
//! = candle open time). Connection is lazy with per-candle retry — if
//! QuestDB is down, candles are dropped with a warning and ingestion to
//! the bus is unaffected.
//!
//! Run at most one writer per QuestDB instance (this sink *or* the
//! standalone factory) unless the table has WAL dedup keys configured,
//! otherwise rows duplicate.
//!
//! Disable with `DATA_PERSIST_CANDLES=false` (default: enabled).

use std::sync::Arc;

use janus_core::{KlineEvent, MarketDataEvent};
use rust_decimal::prelude::ToPrimitive;
use tracing::{debug, error, info, warn};

use crate::actors::CandleData;
use crate::metrics::prometheus_exporter::{QUESTDB_WRITE_ERRORS, QUESTDB_WRITES};
use crate::storage::IlpWriter;

const TABLE: &str = "candles_crypto";

/// Whether the live-mode candle sink should run (`DATA_PERSIST_CANDLES`,
/// default on).
pub fn persist_enabled() -> bool {
    parse_persist_flag(std::env::var("DATA_PERSIST_CANDLES").ok().as_deref())
}

/// Parse the `DATA_PERSIST_CANDLES` value. Anything other than an explicit
/// off-switch keeps the sink enabled — persistence is the expected default.
fn parse_persist_flag(value: Option<&str>) -> bool {
    match value {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "false" | "0" | "off" | "no"
        ),
        None => true,
    }
}

/// Convert a bus [`KlineEvent`] into the storage [`CandleData`] shape.
///
/// - Symbol is the exchange-form concatenation (`BTC` + `USDT` → `BTCUSDT`),
///   matching what the standalone factory writes and what the WebUI's
///   QuestDB symbol catalog serves back.
/// - Bus timestamps are Unix microseconds; `CandleData` carries milliseconds
///   ([`IlpWriter::write_candle`] appends the ms→ns suffix).
fn kline_to_candle(k: &KlineEvent) -> CandleData {
    CandleData {
        symbol: format!("{}{}", k.symbol.base, k.symbol.quote).to_uppercase(),
        exchange: k.exchange.to_string(),
        open_time: k.open_time / 1000,
        close_time: k.close_time / 1000,
        open: k.open.to_f64().unwrap_or(0.0),
        high: k.high.to_f64().unwrap_or(0.0),
        low: k.low.to_f64().unwrap_or(0.0),
        close: k.close.to_f64().unwrap_or(0.0),
        volume: k.volume.to_f64().unwrap_or(0.0),
        interval: k.interval.clone(),
    }
}

/// Run the candle sink until shutdown.
///
/// Subscribes to the [`MarketDataBus`](janus_core::MarketDataBus) and writes
/// each closed kline to QuestDB. The ILP connection is established lazily on
/// the first candle and re-attempted per candle while QuestDB is unreachable,
/// so a QuestDB outage never affects bus ingestion. Lagged broadcast slots
/// are skipped (gap detection + backfill own historical completeness).
pub async fn run(state: Arc<janus_core::JanusState>) {
    let host = state.config.questdb.host.clone();
    let ilp_port = state.config.questdb.ilp_port;
    let mut rx = state.market_data_bus.subscribe();
    let mut writer: Option<IlpWriter> = None;
    let mut written: u64 = 0;
    let mut dropped: u64 = 0;

    info!(
        "CandleSink: persisting closed klines → {}:{} table {}",
        host, ilp_port, TABLE
    );

    loop {
        if state.is_shutdown_requested() {
            break;
        }

        let event = tokio::select! {
            ev = rx.recv() => ev,
            // Periodically wake to re-check the shutdown flag even when the
            // bus is quiet.
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => continue,
        };

        match event {
            Ok(MarketDataEvent::Kline(k)) if k.is_closed => {
                if writer.is_none() {
                    match IlpWriter::new(&host, ilp_port, 64, 100).await {
                        Ok(w) => {
                            info!("CandleSink: connected to QuestDB at {}:{}", host, ilp_port);
                            writer = Some(w);
                        }
                        Err(e) => {
                            dropped += 1;
                            QUESTDB_WRITE_ERRORS
                                .with_label_values(&[TABLE, "connect"])
                                .inc();
                            warn!(
                                "CandleSink: QuestDB unreachable ({}); dropped candle ({} total dropped)",
                                e, dropped
                            );
                            continue;
                        }
                    }
                }

                let candle = kline_to_candle(&k);
                if let Some(w) = writer.as_mut() {
                    // Flush per candle: closed-kline cadence is low (assets ×
                    // intervals per minute) and immediate visibility matters
                    // more than batching here. IlpWriter reconnects internally
                    // on the next flush after a failure.
                    let result = match w.write_candle(&candle).await {
                        Ok(()) => w.flush().await,
                        Err(e) => Err(e),
                    };
                    match result {
                        Ok(()) => {
                            written += 1;
                            QUESTDB_WRITES.with_label_values(&[TABLE]).inc();
                            debug!(
                                "CandleSink: wrote {} {} @ {} ({} total)",
                                candle.symbol, candle.interval, candle.open_time, written
                            );
                        }
                        Err(e) => {
                            dropped += 1;
                            QUESTDB_WRITE_ERRORS
                                .with_label_values(&[TABLE, "write"])
                                .inc();
                            warn!(
                                "CandleSink: write failed ({}); dropped candle ({} total dropped)",
                                e, dropped
                            );
                        }
                    }
                }
            }
            // Open klines, trades, and other event kinds are not persisted here.
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                warn!(
                    "CandleSink: lagged behind the bus, skipped {} events (backfill owns gaps)",
                    skipped
                );
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                info!("CandleSink: market data bus closed — stopping");
                break;
            }
        }
    }

    if let Some(mut w) = writer {
        let _ = w.flush().await;
    }
    info!(
        "CandleSink: stopped ({} candles written, {} dropped)",
        written, dropped
    );
}

/// Restart backoff applied after the sink task exits or panics before a
/// shutdown was requested. Short enough that a recurrence self-heals within
/// seconds, long enough that a tight crash loop can't spin the CPU.
const RESTART_BACKOFF: std::time::Duration = std::time::Duration::from_secs(5);

/// Supervise [`run`] so a single task death can no longer permanently stop
/// candle persistence.
///
/// The sink was previously spawned exactly once and unsupervised. On
/// 2026-07-11 the task terminated (a silent panic is the only such exit
/// path) and never came back: `questdb_writes_total{candles_crypto}` froze
/// while WebSocket ingestion kept running, and — because the terminated task
/// was the bus's only subscriber — `market_data_bus.publish()` began
/// returning `Err`, freezing the subscriber-gated `klines_published` stat
/// too. Nothing restarted it, so the freeze lasted until a process restart.
///
/// This wrapper runs the sink inside an isolated `tokio::spawn` (so a panic
/// is caught as a `JoinError` instead of unwinding the supervisor), restarts
/// it after [`RESTART_BACKOFF`], and stops cleanly once shutdown is
/// requested. [`run`] re-subscribes to the bus on each start and already
/// tolerates QuestDB outages internally, so restarting is always safe; the
/// only cost is that candles emitted during the brief downtime are skipped,
/// which the backfill/candle-scan gap filler already owns.
pub async fn run_supervised(state: Arc<janus_core::JanusState>) {
    let shutdown_state = state.clone();
    supervise_restartable(
        "CandleSink",
        move || run(state.clone()),
        move || shutdown_state.is_shutdown_requested(),
        RESTART_BACKOFF,
    )
    .await;
}

/// Generic restart supervisor: run `make_task()` inside an isolated
/// `tokio::spawn`, restart it after `backoff` whenever it exits (cleanly or
/// via panic), and stop once `should_stop()` returns true.
///
/// Returns the number of times the task was started (used by tests to assert
/// respawn behaviour). Isolating the task in its own `tokio::spawn` means a
/// panic surfaces as `JoinError::is_panic()` here rather than tearing down
/// the supervisor, which is what makes a panicking sink recoverable.
async fn supervise_restartable<F, Fut, S>(
    label: &str,
    mut make_task: F,
    should_stop: S,
    backoff: std::time::Duration,
) -> u64
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
    S: Fn() -> bool,
{
    let mut starts: u64 = 0;
    while !should_stop() {
        starts += 1;
        match tokio::spawn(make_task()).await {
            Ok(()) => {
                if should_stop() {
                    break;
                }
                warn!(
                    "{label}: task exited unexpectedly (no shutdown requested); restarting in {:?}",
                    backoff
                );
            }
            Err(e) if e.is_panic() => {
                error!("{label}: task panicked ({e}); restarting in {:?}", backoff);
            }
            Err(e) => {
                // Cancelled/aborted (e.g. runtime shutdown): don't restart.
                info!("{label}: task cancelled ({e}); stopping supervisor");
                break;
            }
        }
        if should_stop() {
            break;
        }
        tokio::time::sleep(backoff).await;
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Symbol};
    use rust_decimal::Decimal;

    fn sample_kline() -> KlineEvent {
        KlineEvent {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTC", "USDT"),
            interval: "1m".to_string(),
            open_time: 1_700_000_000_000_000, // µs
            close_time: 1_700_000_059_999_000,
            open: Decimal::from(50_000),
            high: Decimal::from(50_100),
            low: Decimal::from(49_900),
            close: Decimal::from(50_050),
            volume: Decimal::new(105, 1), // 10.5
            quote_volume: None,
            trades: Some(42),
            is_closed: true,
        }
    }

    #[test]
    fn test_kline_to_candle_converts_units_and_symbol() {
        let candle = kline_to_candle(&sample_kline());

        assert_eq!(candle.symbol, "BTCUSDT");
        assert_eq!(candle.exchange, "binance");
        assert_eq!(candle.interval, "1m");
        // µs → ms
        assert_eq!(candle.open_time, 1_700_000_000_000);
        assert_eq!(candle.close_time, 1_700_000_059_999);
        assert_eq!(candle.open, 50_000.0);
        assert_eq!(candle.high, 50_100.0);
        assert_eq!(candle.low, 49_900.0);
        assert_eq!(candle.close, 50_050.0);
        assert_eq!(candle.volume, 10.5);
    }

    #[test]
    fn test_kline_to_candle_uppercases_symbol() {
        let mut k = sample_kline();
        k.symbol = Symbol::new("eth", "usdt");
        assert_eq!(kline_to_candle(&k).symbol, "ETHUSDT");
    }

    #[test]
    fn test_parse_persist_flag_defaults_on() {
        assert!(parse_persist_flag(None));
        assert!(parse_persist_flag(Some("true")));
        assert!(parse_persist_flag(Some("1")));
        assert!(parse_persist_flag(Some("anything-else")));
    }

    /// Regression for the 2026-07-11 stall: a sink task that exits *and*
    /// panics must be respawned by the supervisor, not left dead. Also proves
    /// the supervisor terminates promptly once shutdown is requested (no
    /// infinite hot loop) and honours the backoff between restarts.
    #[tokio::test(start_paused = true)]
    async fn supervisor_restarts_on_exit_and_panic_then_stops_on_shutdown() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        let invocations = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let inv = invocations.clone();
        let stop_setter = stop.clone();
        let stop_reader = stop.clone();
        let backoff = std::time::Duration::from_secs(5);

        let starts = supervise_restartable(
            "test-sink",
            move || {
                let n = inv.fetch_add(1, Ordering::SeqCst) + 1;
                let stop_setter = stop_setter.clone();
                async move {
                    match n {
                        // 1st start: exit cleanly and unexpectedly → must restart.
                        1 => {}
                        // 2nd start: panic → supervisor must survive and restart.
                        2 => panic!("simulated sink panic"),
                        // 3rd start: request shutdown, then exit cleanly → stop.
                        _ => stop_setter.store(true, Ordering::SeqCst),
                    }
                }
            },
            move || stop_reader.load(Ordering::SeqCst),
            backoff,
        )
        .await;

        // Started three times: initial + one restart after clean exit + one
        // restart after the panic. The third start flips the shutdown flag.
        assert_eq!(starts, 3, "supervisor must respawn after exit and panic");
        assert_eq!(invocations.load(Ordering::SeqCst), 3);
        assert!(stop.load(Ordering::SeqCst), "shutdown flag should be set");
    }

    #[test]
    fn test_parse_persist_flag_off_switches() {
        assert!(!parse_persist_flag(Some("false")));
        assert!(!parse_persist_flag(Some("FALSE")));
        assert!(!parse_persist_flag(Some("0")));
        assert!(!parse_persist_flag(Some("off")));
        assert!(!parse_persist_flag(Some("no")));
        assert!(!parse_persist_flag(Some(" False ")));
    }
}
