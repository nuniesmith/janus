//! Indicator Warmup Module
//!
//! This module provides functionality to warm up technical indicators from
//! historical candle data.  Indicators are ready immediately on startup
//! without waiting for real-time candle closes.
//!
//! ## Data source priority
//!
//! Candles are fetched using a three-tier fallback chain:
//!
//! 1. **Python data service** (`fks_ruby` — primary)
//!    Fetches from `GET /bars/{symbol}/candles`.  This is the authoritative
//!    store for all traded instruments (CME futures *and* crypto) — data is
//!    sourced from Massive/yfinance and cached in Postgres + Redis.
//!
//! 2. **QuestDB** (fallback)
//!    Queries `candles_crypto` directly.  Only works for crypto pairs that
//!    have already been ingested via WebSocket streams.
//!
//! 3. **Binance REST API** (last resort)
//!    Direct API call — only viable for crypto pairs, never for CME futures.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let warmup = IndicatorWarmup::new(storage.clone(), indicator_actor.clone());
//! warmup.warmup_from_history("MGC", "1m", 500).await?;
//! ```

use anyhow::{Context, Result, anyhow};
use regex::Regex;
use std::sync::Arc;
use std::sync::OnceLock;
use tracing::{debug, error, info, warn};

use crate::actors::indicator::{CandleInput, IndicatorActor, IndicatorMessage};
use crate::backfill::python_data_client::{
    CandleFetchRequest, PythonDataClient, PythonDataClientConfig, to_data_service_symbol,
};
use crate::storage::StorageManager;

/// Configuration for indicator warmup
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Maximum number of candles to fetch for warmup.
    /// 250 is enough for EMA-200 with a small buffer; increase if you need
    /// RSI/MACD to be fully settled over a longer history.
    pub max_candles: usize,
    /// Batch size for the candle-feed loop (yields to Tokio between batches).
    pub batch_size: usize,
    /// Skip warmup entirely when indicators are already warm.
    pub skip_if_warm: bool,
    /// Config for the Python data service HTTP client.
    /// When `None` the client is built from environment variables.
    pub python_client_config: Option<PythonDataClientConfig>,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            max_candles: 500, // Enough for EMA-200 + generous buffer
            batch_size: 50,
            skip_if_warm: true,
            python_client_config: None, // built from env vars on first use
        }
    }
}

/// Result of a warmup operation
#[derive(Debug)]
pub struct WarmupResult {
    pub symbol: String,
    pub timeframe: String,
    pub candles_processed: usize,
    pub duration_ms: u64,
    pub all_indicators_ready: bool,
}

/// Indicator warmup service.
///
/// Fetches historical OHLCV candles and feeds them into the `IndicatorActor`
/// so all incremental indicators (EMA, RSI, MACD, ATR) are ready before live
/// data arrives.
pub struct IndicatorWarmup {
    #[allow(dead_code)]
    storage: Arc<StorageManager>,
    indicator_actor: Arc<IndicatorActor>,
    config: WarmupConfig,
    /// Lazily-built Python data client (built once on first warmup call).
    python_client: Option<PythonDataClient>,
}

impl IndicatorWarmup {
    /// Create a new `IndicatorWarmup` with default configuration.
    ///
    /// The Python data service client will be built lazily from environment
    /// variables (`PYTHON_DATA_SERVICE_URL`, `DATA_SERVICE_API_KEY`, etc.)
    /// on the first call to `warmup_from_history`.
    pub fn new(storage: Arc<StorageManager>, indicator_actor: Arc<IndicatorActor>) -> Self {
        Self {
            storage,
            indicator_actor,
            config: WarmupConfig::default(),
            python_client: None,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        storage: Arc<StorageManager>,
        indicator_actor: Arc<IndicatorActor>,
        config: WarmupConfig,
    ) -> Self {
        Self {
            storage,
            indicator_actor,
            config,
            python_client: None,
        }
    }

    /// Return a reference to (or lazily build) the Python data client.
    ///
    /// Building is cheap — it just creates an `reqwest::Client` with the
    /// configured headers.  Errors here are non-fatal; the warmup will fall
    /// back to QuestDB / Binance if the client cannot be constructed.
    fn get_python_client(&mut self) -> Option<&PythonDataClient> {
        if self.python_client.is_none() {
            let cfg = self.config.python_client_config.clone().unwrap_or_default();
            match PythonDataClient::new(cfg) {
                Ok(client) => {
                    self.python_client = Some(client);
                }
                Err(e) => {
                    warn!(
                        "IndicatorWarmup: could not build PythonDataClient — {e}. \
                         Will fall back to QuestDB / Binance for warmup."
                    );
                }
            }
        }
        self.python_client.as_ref()
    }

    /// Warm up indicators for a single symbol/timeframe pair.
    ///
    /// Candle data is fetched using the three-tier priority chain described in
    /// the module docs.  The method is `&mut self` because it lazily builds
    /// the `PythonDataClient` on first call.
    pub async fn warmup_from_history(
        &mut self,
        symbol: &str,
        timeframe: &str,
        num_candles: Option<usize>,
    ) -> Result<WarmupResult> {
        let start = std::time::Instant::now();
        let candles_to_fetch = num_candles.unwrap_or(self.config.max_candles);

        info!(
            "IndicatorWarmup: Starting warmup for {}:{} (up to {} candles)",
            symbol, timeframe, candles_to_fetch
        );

        // Skip early if indicators are already warm.
        if self.config.skip_if_warm
            && let Some(status) = self
                .indicator_actor
                .get_warmup_status(symbol, timeframe)
                .await
            && status.all_ready()
        {
            info!(
                "IndicatorWarmup: {}:{} already warm, skipping",
                symbol, timeframe
            );
            return Ok(WarmupResult {
                symbol: symbol.to_string(),
                timeframe: timeframe.to_string(),
                candles_processed: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                all_indicators_ready: true,
            });
        }

        // ── Tier 1: Python data service ────────────────────────────────────
        let candles = self
            .fetch_from_python_service(symbol, timeframe, candles_to_fetch)
            .await;

        // ── Tier 2: QuestDB fallback ───────────────────────────────────────
        let candles = match candles {
            Some(c) if !c.is_empty() => {
                info!(
                    "IndicatorWarmup: [python-data] {} candles for {}:{}",
                    c.len(),
                    symbol,
                    timeframe
                );
                c
            }
            _ => {
                info!(
                    "IndicatorWarmup: python-data service empty/unavailable for {}:{}, \
                     falling back to QuestDB",
                    symbol, timeframe
                );
                match self
                    .fetch_historical_candles(symbol, timeframe, candles_to_fetch)
                    .await
                {
                    Ok(c) if !c.is_empty() => {
                        info!(
                            "IndicatorWarmup: [questdb] {} candles for {}:{}",
                            c.len(),
                            symbol,
                            timeframe
                        );
                        c
                    }
                    Ok(_) | Err(_) => {
                        // ── Tier 3: Binance REST API (crypto only) ─────────
                        info!(
                            "IndicatorWarmup: QuestDB empty for {}:{}, \
                             falling back to Binance REST",
                            symbol, timeframe
                        );
                        self.fetch_from_binance(symbol, timeframe, candles_to_fetch)
                            .await
                            .unwrap_or_default()
                    }
                }
            }
        };

        if candles.is_empty() {
            warn!(
                "IndicatorWarmup: No historical candles found for {}:{} \
                 (tried python-data, QuestDB, Binance)",
                symbol, timeframe
            );
            return Ok(WarmupResult {
                symbol: symbol.to_string(),
                timeframe: timeframe.to_string(),
                candles_processed: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                all_indicators_ready: false,
            });
        }

        // ── Feed candles to the indicator actor (chronological order) ──────
        let sender = self.indicator_actor.get_sender();
        let mut processed = 0usize;

        for candle in candles {
            if sender.send(IndicatorMessage::Candle(candle)).is_err() {
                error!(
                    "IndicatorWarmup: Failed to send candle to indicator actor — channel closed"
                );
                break;
            }
            processed += 1;

            // Yield periodically so the actor can drain its queue.
            if processed.is_multiple_of(self.config.batch_size) {
                tokio::task::yield_now().await;
            }
        }

        // ── Check final warmup status ──────────────────────────────────────
        let all_ready = self
            .indicator_actor
            .get_warmup_status(symbol, timeframe)
            .await
            .map(|s| s.all_ready())
            .unwrap_or(false);

        let duration_ms = start.elapsed().as_millis() as u64;

        info!(
            "IndicatorWarmup: Completed {}:{} — {} candles in {}ms, all_ready={}",
            symbol, timeframe, processed, duration_ms, all_ready
        );

        Ok(WarmupResult {
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            candles_processed: processed,
            duration_ms,
            all_indicators_ready: all_ready,
        })
    }

    /// Warm up all configured symbol/timeframe pairs sequentially.
    pub async fn warmup_all(
        &mut self,
        symbols: &[(String, String)], // (symbol, timeframe) pairs
    ) -> Vec<WarmupResult> {
        let mut results = Vec::with_capacity(symbols.len());

        for (symbol, timeframe) in symbols {
            match self.warmup_from_history(symbol, timeframe, None).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!(
                        "IndicatorWarmup: Failed to warm up {}:{}: {}",
                        symbol, timeframe, e
                    );
                    results.push(WarmupResult {
                        symbol: symbol.clone(),
                        timeframe: timeframe.clone(),
                        candles_processed: 0,
                        duration_ms: 0,
                        all_indicators_ready: false,
                    });
                }
            }
        }

        results
    }

    // =========================================================================
    // Data source tier 1 — Python data service
    // =========================================================================

    /// Attempt to fetch candles from the Python data service.
    ///
    /// Returns `Some(candles)` on success (which may be an empty vec if the
    /// symbol has no data yet), or `None` if the service is unreachable or
    /// returns an error.
    async fn fetch_from_python_service(
        &mut self,
        symbol: &str,
        timeframe: &str,
        limit: usize,
    ) -> Option<Vec<CandleInput>> {
        // Build the client lazily.
        let client = self.get_python_client()?;

        let ds_symbol = to_data_service_symbol(symbol);

        let request = CandleFetchRequest {
            symbol: ds_symbol.clone(),
            interval: timeframe.to_string(),
            limit: limit.min(10_000) as u32,
            days_back: 30,
            auto_fill: true,
        };

        match client.fetch_candles(&request).await {
            Ok(result) => {
                debug!(
                    "IndicatorWarmup: python-data returned {} candles for {} (requested {})",
                    result.candles.len(),
                    ds_symbol,
                    limit
                );
                Some(result.candles)
            }
            Err(e) => {
                warn!(
                    "IndicatorWarmup: python-data fetch failed for {}:{} — {}",
                    symbol, timeframe, e
                );
                None
            }
        }
    }

    // =========================================================================
    // Data source tier 3 — Binance REST API
    // =========================================================================

    /// Fetch candles from the Binance klines REST endpoint.
    ///
    /// This is the last-resort fallback and only works for crypto pairs
    /// (e.g. `BTCUSDT`).  CME futures symbols will return an empty vec.
    async fn fetch_from_binance(
        &self,
        symbol: &str,
        timeframe: &str,
        limit: usize,
    ) -> Result<Vec<CandleInput>> {
        use crate::backfill::historical_candles::HistoricalCandleFetcher;

        info!(
            "IndicatorWarmup: [binance] fetching {} candles for {}:{}",
            limit, symbol, timeframe
        );

        let fetcher = HistoricalCandleFetcher::new();
        fetcher.fetch_candles(symbol, timeframe, limit).await
    }

    // =========================================================================
    // Data source tier 2 — QuestDB direct query
    // =========================================================================

    /// Fetch historical candles from QuestDB (the `candles_crypto` table).
    async fn fetch_historical_candles(
        &self,
        symbol: &str,
        timeframe: &str,
        limit: usize,
    ) -> Result<Vec<CandleInput>> {
        // Validate inputs to prevent SQL injection
        // Symbol must be alphanumeric with optional underscores (e.g., BTCUSD, BTC_USDT)
        let validated_symbol = validate_symbol(symbol).context("Invalid symbol format")?;

        // Timeframe must match known patterns (e.g., 1m, 5m, 15m, 1h, 4h, 1d, 1w)
        let validated_timeframe =
            validate_timeframe(timeframe).context("Invalid timeframe format")?;

        // Limit must be within reasonable bounds
        let validated_limit = limit.min(10000); // Cap at 10k candles max

        // Query candles from QuestDB ordered by timestamp ascending (oldest first)
        // This ensures indicators process candles in chronological order
        // Note: QuestDB schema uses 'interval' column, but we map to timeframe in API
        //
        // SECURITY: Using validated/sanitized inputs. QuestDB doesn't support
        // parameterized queries via HTTP API, so we validate inputs strictly.
        let query = format!(
            r#"
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                exchange
            FROM candles_crypto
            WHERE symbol = '{}'
              AND interval = '{}'
            ORDER BY timestamp ASC
            LIMIT {}
            "#,
            validated_symbol, validated_timeframe, validated_limit
        );

        debug!("IndicatorWarmup: Executing query: {}", query);

        // Execute query via QuestDB HTTP API
        let candles = self
            .query_questdb(&query, &validated_symbol, &validated_timeframe)
            .await
            .context("QuestDB query failed")?;

        Ok(candles)
    }

    /// Execute a query against QuestDB and parse results into CandleInput
    async fn query_questdb(
        &self,
        query: &str,
        symbol: &str,
        timeframe: &str,
    ) -> Result<Vec<CandleInput>> {
        // Get QuestDB HTTP endpoint from storage manager
        // The storage manager should have the QuestDB connection info
        let questdb_url =
            std::env::var("QUESTDB_HTTP_URL").unwrap_or_else(|_| "http://questdb:9000".to_string());

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/exec", questdb_url))
            .query(&[("query", query)])
            .send()
            .await
            .context("Failed to send request to QuestDB")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("QuestDB query failed with status {}: {}", status, body);
        }

        let json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse QuestDB response")?;

        // Parse QuestDB response format
        // Response format: { "columns": [...], "dataset": [[...], ...], "count": N }
        let dataset = json["dataset"]
            .as_array()
            .context("Missing dataset in response")?;

        let mut candles = Vec::with_capacity(dataset.len());

        for row in dataset {
            let row = row.as_array().context("Invalid row format")?;
            if row.len() < 7 {
                warn!("IndicatorWarmup: Skipping incomplete row");
                continue;
            }

            // Parse row: [timestamp, open, high, low, close, volume, exchange]
            let timestamp = parse_timestamp(&row[0])?;
            let open = parse_float(&row[1])?;
            let high = parse_float(&row[2])?;
            let low = parse_float(&row[3])?;
            let close = parse_float(&row[4])?;
            let volume = parse_float(&row[5])?;
            let exchange = row[6].as_str().unwrap_or("binance").to_string();

            candles.push(CandleInput {
                symbol: symbol.to_string(),
                exchange,
                timeframe: timeframe.to_string(),
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        Ok(candles)
    }
}

// Static regex patterns for input validation (compiled once)
static SYMBOL_PATTERN: OnceLock<Regex> = OnceLock::new();
static TIMEFRAME_PATTERN: OnceLock<Regex> = OnceLock::new();

/// Allowed timeframes for candle queries
const VALID_TIMEFRAMES: &[&str] = &[
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M",
];

/// Validate and sanitize symbol input to prevent SQL injection
///
/// Valid symbols:
/// - Alphanumeric characters only (A-Z, 0-9)
/// - Optional underscores, hyphens, or forward slashes for pairs
/// - Max length of 20 characters
/// - Examples: BTCUSD, BTC_USDT, BTC/USDT, ETH-PERP
fn validate_symbol(symbol: &str) -> Result<String> {
    // Check length
    if symbol.is_empty() || symbol.len() > 20 {
        return Err(anyhow!("Symbol must be 1-20 characters"));
    }

    // Get or initialize the regex pattern
    let pattern = SYMBOL_PATTERN.get_or_init(|| {
        Regex::new(r"^[A-Za-z0-9][A-Za-z0-9_\-/]*[A-Za-z0-9]$|^[A-Za-z0-9]$").unwrap()
    });

    if !pattern.is_match(symbol) {
        return Err(anyhow!(
            "Invalid symbol format: must be alphanumeric with optional _-/ separators"
        ));
    }

    // Convert to uppercase for consistency
    Ok(symbol.to_uppercase())
}

/// Validate timeframe input against known valid values
///
/// Only allows specific timeframe strings to prevent injection
fn validate_timeframe(timeframe: &str) -> Result<String> {
    // Check against whitelist of valid timeframes
    if VALID_TIMEFRAMES.contains(&timeframe) {
        return Ok(timeframe.to_string());
    }

    // Also allow numeric patterns like "1min", "5min", "1hour" via regex
    let pattern = TIMEFRAME_PATTERN
        .get_or_init(|| Regex::new(r"^[1-9][0-9]?(m|min|h|hour|d|day|w|week|M|month)$").unwrap());

    if pattern.is_match(timeframe) {
        Ok(timeframe.to_string())
    } else {
        Err(anyhow!(
            "Invalid timeframe: must be one of {:?} or match pattern like '1m', '4h', '1d'",
            VALID_TIMEFRAMES
        ))
    }
}

/// Parse timestamp from QuestDB response (can be string or number)
fn parse_timestamp(value: &serde_json::Value) -> Result<i64> {
    match value {
        serde_json::Value::Number(n) => n.as_i64().context("Timestamp not an i64"),
        serde_json::Value::String(s) => {
            // Try parsing as ISO8601 datetime
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
                return Ok(dt.timestamp_millis());
            }
            // Try parsing as microseconds string
            s.parse::<i64>().context("Failed to parse timestamp string")
        }
        _ => anyhow::bail!("Invalid timestamp type"),
    }
}

/// Parse float from QuestDB response (can be string or number)
fn parse_float(value: &serde_json::Value) -> Result<f64> {
    match value {
        serde_json::Value::Number(n) => n.as_f64().context("Float conversion failed"),
        serde_json::Value::String(s) => s.parse::<f64>().context("Failed to parse float string"),
        serde_json::Value::Null => Ok(0.0),
        _ => anyhow::bail!("Invalid float type"),
    }
}

/// Command-line utility to run indicator warmup
pub async fn run_warmup_cli(
    storage: Arc<StorageManager>,
    indicator_actor: Arc<IndicatorActor>,
    symbols: Vec<(String, String)>,
) -> Result<()> {
    let mut warmup = IndicatorWarmup::new(storage, indicator_actor);

    println!(
        "Starting indicator warmup for {} symbol(s)...",
        symbols.len()
    );

    let results = warmup.warmup_all(&symbols).await;

    println!("\n=== Warmup Results ===");
    for result in &results {
        println!(
            "  {}:{} - {} candles, {}ms, ready={}",
            result.symbol,
            result.timeframe,
            result.candles_processed,
            result.duration_ms,
            result.all_indicators_ready
        );
    }

    let total_candles: usize = results.iter().map(|r| r.candles_processed).sum();
    let total_ready = results.iter().filter(|r| r.all_indicators_ready).count();

    println!(
        "\nTotal: {} candles processed, {}/{} pairs ready",
        total_candles,
        total_ready,
        results.len()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_config_default() {
        let config = WarmupConfig::default();
        assert_eq!(config.max_candles, 500);
        assert_eq!(config.batch_size, 50);
        assert!(config.skip_if_warm);
    }

    #[test]
    fn test_parse_timestamp_number() {
        let value = serde_json::json!(1672531200000i64);
        let result = parse_timestamp(&value).unwrap();
        assert_eq!(result, 1672531200000);
    }

    #[test]
    fn test_parse_timestamp_string() {
        let value = serde_json::json!("1672531200000");
        let result = parse_timestamp(&value).unwrap();
        assert_eq!(result, 1672531200000);
    }

    #[test]
    fn test_parse_float_number() {
        let value = serde_json::json!(123.456);
        let result = parse_float(&value).unwrap();
        assert!((result - 123.456).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_float_string() {
        let value = serde_json::json!("123.456");
        let result = parse_float(&value).unwrap();
        assert!((result - 123.456).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_float_null() {
        let value = serde_json::Value::Null;
        let result = parse_float(&value).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_warmup_result() {
        let result = WarmupResult {
            symbol: "BTCUSD".to_string(),
            timeframe: "1m".to_string(),
            candles_processed: 200,
            duration_ms: 150,
            all_indicators_ready: true,
        };

        assert_eq!(result.symbol, "BTCUSD");
        assert_eq!(result.candles_processed, 200);
        assert!(result.all_indicators_ready);
    }

    // === SQL Injection Prevention Tests ===

    #[test]
    fn test_validate_symbol_valid() {
        // Valid symbols
        assert!(validate_symbol("BTCUSDT").is_ok());
        assert!(validate_symbol("BTC_USDT").is_ok());
        assert!(validate_symbol("ETH").is_ok());
        assert!(validate_symbol("BTC/USDT").is_ok());
        assert!(validate_symbol("ETH-PERP").is_ok());

        // Should uppercase
        assert_eq!(validate_symbol("btcusdt").unwrap(), "BTCUSDT");
    }

    #[test]
    fn test_validate_symbol_sql_injection_attempts() {
        // SQL injection attempts should fail
        assert!(validate_symbol("'; DROP TABLE candles_crypto; --").is_err());
        assert!(validate_symbol("BTCUSD' OR '1'='1").is_err());
        assert!(validate_symbol("BTCUSD; DELETE FROM").is_err());
        assert!(validate_symbol("' UNION SELECT * FROM").is_err());
        assert!(validate_symbol("BTCUSD\n--").is_err());
        assert!(validate_symbol("BTC'USDT").is_err());
    }

    #[test]
    fn test_validate_symbol_edge_cases() {
        // Empty and too long
        assert!(validate_symbol("").is_err());
        assert!(validate_symbol("A".repeat(21).as_str()).is_err());

        // Single char is valid
        assert!(validate_symbol("X").is_ok());

        // Cannot start/end with separator
        assert!(validate_symbol("_BTC").is_err());
        assert!(validate_symbol("BTC_").is_err());
    }

    #[test]
    fn test_validate_timeframe_valid() {
        // Standard timeframes
        assert!(validate_timeframe("1m").is_ok());
        assert!(validate_timeframe("5m").is_ok());
        assert!(validate_timeframe("15m").is_ok());
        assert!(validate_timeframe("1h").is_ok());
        assert!(validate_timeframe("4h").is_ok());
        assert!(validate_timeframe("1d").is_ok());
        assert!(validate_timeframe("1w").is_ok());
        assert!(validate_timeframe("1M").is_ok());
    }

    #[test]
    fn test_validate_timeframe_sql_injection_attempts() {
        // SQL injection attempts should fail
        assert!(validate_timeframe("1m'; DROP TABLE --").is_err());
        assert!(validate_timeframe("' OR '1'='1").is_err());
        assert!(validate_timeframe("1h; DELETE FROM").is_err());
        assert!(validate_timeframe("1m\n--comment").is_err());
    }

    #[test]
    fn test_validate_timeframe_invalid() {
        // Invalid formats
        assert!(validate_timeframe("").is_err());
        assert!(validate_timeframe("abc").is_err());
        assert!(validate_timeframe("100x").is_err());
        assert!(validate_timeframe("0m").is_err()); // Cannot start with 0
    }
}
