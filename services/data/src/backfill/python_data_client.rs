//! Python Data Service Client
//!
//! HTTP client that fetches historical OHLCV candles from the Python data
//! service (`fks_ruby`) for use during Janus indicator warmup.
//!
//! ## Why this exists
//!
//! Janus previously warmed up its technical indicators (EMA, RSI, MACD, ATR)
//! by fetching candles directly from:
//!   1. QuestDB — works for crypto ticks already ingested via WebSocket
//!   2. Binance REST API — works for crypto, **not** for CME/futures symbols
//!
//! The Python data service already holds authoritative bar data for all
//! traded instruments (MGC, SIL, MES, MNQ, M2K, MYM) sourced from Massive
//! and stored in Postgres with a Redis cache layer.  This client lets Janus
//! pull from that canonical store instead of bypassing it.
//!
//! ## Endpoint consumed
//!
//! ```text
//! GET /bars/{symbol}/candles
//!     ?interval=1m
//!     &limit=500
//!     &days_back=30
//!     &auto_fill=true
//! ```
//!
//! Response (ascending, oldest-first):
//! ```json
//! {
//!   "symbol": "MGC=F",
//!   "interval": "1m",
//!   "count": 500,
//!   "candles": [
//!     { "timestamp": 1700000000000, "open": 1950.2, "high": 1952.5,
//!       "low": 1949.8, "close": 1951.3, "volume": 42 },
//!     ...
//!   ]
//! }
//! ```
//!
//! ## Additional endpoints consumed
//!
//! | Endpoint                       | Method   | Description                                    |
//! |--------------------------------|----------|------------------------------------------------|
//! | `/bars/symbols`                | `GET`    | List available symbols (lightweight)           |
//! | `/bars/assets`                 | `GET`    | List assets with tickers + bar counts          |
//! | `/bars/status`                 | `GET`    | Bar counts, date ranges, coverage              |
//! | `/bars/{symbol}/gaps`          | `GET`    | Gap report for a symbol                        |
//! | `/bars/{symbol}/fill`          | `POST`   | Trigger incremental fill                       |
//! | `/bars/fill/all`               | `POST`   | Trigger fill for all assets                    |
//! | `/bars/fill/status`            | `GET`    | Poll fill job status                           |
//! | `/api/analysis/data_source`    | `GET`    | Current active data source                     |
//! | `/api/analysis/live_feed`      | `GET`    | Live feed connection status                    |
//! | `/health`                      | `GET`    | Service health check                           |
//!
//! ## Configuration (environment variables)
//!
//! | Variable                     | Default                | Description                                        |
//! |------------------------------|------------------------|----------------------------------------------------|
//! | `PYTHON_DATA_SERVICE_URL`    | `http://fks_ruby:8000` | Base URL of the Python data service                |
//! | `DATA_SERVICE_API_KEY`       | *(empty)*              | Bearer token (matches `API_KEY` in data service)   |
//! | `DATA_CLIENT_TIMEOUT_SECS`   | `30`                   | Per-request HTTP timeout                           |
//! | `DATA_CLIENT_MAX_RETRIES`    | `3`                    | Retries on transient errors                        |
//! | `DATA_CLIENT_RETRY_DELAY_MS` | `500`                  | Base delay between retries (doubles each attempt)  |

use anyhow::{Context, Result, anyhow, bail};
use reqwest::{Client, StatusCode, header};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::actors::indicator::CandleInput;

/// Minimal URL-encoding helper using the `url` crate (already a workspace dep).
/// Matches the inline-module pattern used elsewhere in this codebase.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Python data service HTTP client.
#[derive(Debug, Clone)]
pub struct PythonDataClientConfig {
    /// Base URL of the Python data service, e.g. `http://fks_ruby:8000`
    pub base_url: String,

    /// Optional Bearer token — must match `API_KEY` env var in the data service.
    /// When empty, no `Authorization` header is sent.
    pub api_key: Option<String>,

    /// Per-request HTTP timeout in seconds.
    pub timeout_secs: u64,

    /// Number of retries on network / 5xx errors (not on 4xx).
    pub max_retries: u32,

    /// Base delay between retries in milliseconds (doubles on each attempt).
    pub retry_delay_ms: u64,
}

impl Default for PythonDataClientConfig {
    fn default() -> Self {
        Self {
            base_url: std::env::var("PYTHON_DATA_SERVICE_URL")
                .unwrap_or_else(|_| "http://fks_ruby:8000".to_string()),
            api_key: std::env::var("DATA_SERVICE_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            timeout_secs: std::env::var("DATA_CLIENT_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30),
            max_retries: std::env::var("DATA_CLIENT_MAX_RETRIES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3),
            retry_delay_ms: std::env::var("DATA_CLIENT_RETRY_DELAY_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(500),
        }
    }
}

// ============================================================================
// Wire types — exactly match the Python endpoint's response schema
// ============================================================================

/// A single OHLCV candle as returned by `GET /bars/{symbol}/candles`.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct RawCandle {
    /// Unix timestamp in milliseconds (UTC).
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// Integer volume — Python serialises it as an int, but we accept floats
    /// and strings gracefully via serde_json::Value.
    pub volume: serde_json::Value,
}

/// Top-level response from `GET /bars/{symbol}/candles`.
#[derive(Debug, Deserialize)]
struct CandlesResponse {
    pub symbol: String,
    pub interval: String,
    pub count: usize,
    pub candles: Vec<RawCandle>,
}

/// Asset descriptor from `GET /bars/assets`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AssetInfo {
    pub name: String,
    pub ticker: String,
    #[serde(default)]
    pub symbol: String,
    #[serde(default)]
    pub bar_count: u64,
    #[serde(default)]
    pub latest: Option<String>,
    #[serde(default)]
    pub has_data: bool,
}

/// Response from `GET /bars/symbols`.
#[derive(Debug, Deserialize)]
pub struct SymbolsResponse {
    pub symbols: Vec<String>,
    #[serde(default)]
    pub assets: Vec<AssetInfo>,
}

/// Response from `GET /bars/assets`.
#[derive(Debug, Deserialize)]
pub struct AssetsResponse {
    pub assets: Vec<AssetInfo>,
}

/// Response from `GET /bars/status`.
#[derive(Debug, Deserialize)]
pub struct BarsStatusResponse {
    #[serde(default)]
    pub symbols: Vec<BarStatusEntry>,
}

/// Single entry in the bars status response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BarStatusEntry {
    pub symbol: String,
    #[serde(default)]
    pub bar_count: u64,
    #[serde(default)]
    pub earliest: Option<String>,
    #[serde(default)]
    pub latest: Option<String>,
    #[serde(default)]
    pub is_stale: bool,
}

/// Response from `GET /bars/{symbol}/gaps`.
#[derive(Debug, Deserialize)]
pub struct GapReportResponse {
    pub symbol: String,
    #[serde(default)]
    pub total_bars: u64,
    #[serde(default)]
    pub expected_bars: u64,
    #[serde(default)]
    pub coverage_pct: f64,
    #[serde(default)]
    pub gaps: Vec<GapEntry>,
}

/// A single gap in the gap report.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GapEntry {
    #[serde(default)]
    pub start: String,
    #[serde(default)]
    pub end: String,
    #[serde(default)]
    pub missing_bars: u64,
}

/// Response from `POST /bars/{symbol}/fill` and `POST /bars/fill/all`.
#[derive(Debug, Deserialize)]
pub struct FillJobResponse {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub symbol: Option<String>,
    #[serde(default)]
    pub bars_added: u64,
    #[serde(default)]
    pub filling: bool,
}

/// Response from `GET /api/analysis/data_source`.
#[derive(Debug, Deserialize)]
pub struct DataSourceResponse {
    pub data_source: String,
}

/// Response from `GET /api/analysis/live_feed`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LiveFeedStatus {
    #[serde(default)]
    pub connected: bool,
    #[serde(default)]
    pub status: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

// ============================================================================
// Query parameters
// ============================================================================

/// Parameters for a candle fetch request.
#[derive(Debug, Clone)]
pub struct CandleFetchRequest {
    /// Symbol to fetch, e.g. `"MGC=F"` or `"BTCUSDT"`.
    /// The Python data service normalises futures symbols automatically.
    pub symbol: String,

    /// Candle interval, e.g. `"1m"`, `"5m"`, `"1h"`.
    pub interval: String,

    /// Maximum number of candles to return (1–10 000).
    /// The service returns the most-recent `limit` candles so Janus always
    /// gets the freshest context for indicator warmup.
    pub limit: u32,

    /// How many days back to look when the local store has no data.
    /// The data service will trigger an auto-fill if the data is stale.
    pub days_back: u32,

    /// Whether to ask the data service to auto-fill gaps before responding.
    pub auto_fill: bool,
}

impl Default for CandleFetchRequest {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            interval: "1m".to_string(),
            limit: 500,
            days_back: 30,
            auto_fill: true,
        }
    }
}

// ============================================================================
// Result type
// ============================================================================

/// Result of a successful candle fetch.
#[derive(Debug)]
pub struct FetchCandlesResult {
    /// Canonical symbol as echoed by the data service.
    pub symbol: String,
    /// Interval as echoed by the data service.
    pub interval: String,
    /// Candles in ascending chronological order, ready to feed into
    /// `IndicatorActor`.
    pub candles: Vec<CandleInput>,
}

// ============================================================================
// Client
// ============================================================================

/// HTTP client for the FKS Python data service.
///
/// Create once and reuse — the underlying `reqwest::Client` maintains a
/// connection pool.
pub struct PythonDataClient {
    client: Client,
    config: PythonDataClientConfig,
}

impl PythonDataClient {
    /// Create a new client using the supplied configuration.
    pub fn new(config: PythonDataClientConfig) -> Result<Self> {
        let mut headers = header::HeaderMap::new();

        // Add Bearer auth header if an API key is configured.
        if let Some(ref key) = config.api_key {
            let value = format!("Bearer {}", key);
            let mut header_val = header::HeaderValue::from_str(&value)
                .context("Invalid API key characters in Authorization header")?;
            header_val.set_sensitive(true);
            headers.insert(header::AUTHORIZATION, header_val);
        }

        headers.insert(
            header::ACCEPT,
            header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            header::USER_AGENT,
            header::HeaderValue::from_static("janus-data/1.0"),
        );

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .default_headers(headers)
            .build()
            .context("Failed to build reqwest client for PythonDataClient")?;

        Ok(Self { client, config })
    }

    /// Create a client populated entirely from environment variables.
    ///
    /// See module-level docs for the full list of env vars.
    pub fn from_env() -> Result<Self> {
        Self::new(PythonDataClientConfig::default())
    }

    /// Fetch historical candles for a single symbol.
    ///
    /// Returns `Ok(FetchCandlesResult)` on success.  Returns `Err` only for
    /// hard failures (network unreachable, malformed response, etc.).
    /// An empty candle list (`count == 0`) is considered a soft success —
    /// the caller should fall back to an alternative source in that case.
    pub async fn fetch_candles(&self, request: &CandleFetchRequest) -> Result<FetchCandlesResult> {
        if request.symbol.is_empty() {
            bail!("PythonDataClient: symbol must not be empty");
        }

        // Encode the symbol for use in the URL path (handles `=` in `MGC=F`).
        let encoded_symbol = urlencoding::encode(&request.symbol);
        let url = format!(
            "{}/bars/{}/candles",
            self.config.base_url.trim_end_matches('/'),
            encoded_symbol
        );

        let params = [
            ("interval", request.interval.clone()),
            ("limit", request.limit.to_string()),
            ("days_back", request.days_back.to_string()),
            ("auto_fill", request.auto_fill.to_string()),
        ];

        debug!(
            "PythonDataClient: fetching {} candles for {} from {}",
            request.limit, request.symbol, url
        );

        let response = self
            .fetch_with_retry(&url, &params)
            .await
            .with_context(|| {
                format!(
                    "PythonDataClient: all retries exhausted for {} {}",
                    request.symbol, request.interval
                )
            })?;

        let parsed: CandlesResponse = response
            .json()
            .await
            .context("PythonDataClient: failed to deserialise candles response")?;

        info!(
            "PythonDataClient: received {} candles for {}:{} from data service",
            parsed.count, parsed.symbol, parsed.interval
        );

        let candles = convert_candles(&parsed, &request.symbol, &request.interval);

        Ok(FetchCandlesResult {
            symbol: parsed.symbol,
            interval: parsed.interval,
            candles,
        })
    }

    /// Fetch candles for multiple symbol/timeframe pairs sequentially.
    ///
    /// Failures for individual pairs are logged but do **not** abort the
    /// remaining fetches.  The returned vec has one entry per input pair
    /// (successful entries contain candles; failed entries have an empty vec).
    pub async fn fetch_candles_multi(
        &self,
        requests: &[CandleFetchRequest],
    ) -> Vec<Result<FetchCandlesResult>> {
        let mut results = Vec::with_capacity(requests.len());
        for req in requests {
            results.push(self.fetch_candles(req).await);
        }
        results
    }

    /// Perform a GET request with exponential-backoff retry on transient errors.
    ///
    /// 4xx responses (except 429 Too Many Requests) are returned immediately
    /// without retrying — they indicate a caller bug rather than a transient
    /// server issue.
    async fn fetch_with_retry(
        &self,
        url: &str,
        params: &[(&str, String)],
    ) -> Result<reqwest::Response> {
        let max = self.config.max_retries;
        let base_delay = self.config.retry_delay_ms;

        for attempt in 0..=max {
            let response = self.client.get(url).query(params).send().await;

            match response {
                Ok(resp) => {
                    let status = resp.status();

                    // Success — return immediately.
                    if status.is_success() {
                        return Ok(resp);
                    }

                    // Client errors (except 429) are not retried.
                    if status.is_client_error() && status != StatusCode::TOO_MANY_REQUESTS {
                        let body = resp.text().await.unwrap_or_default();
                        bail!(
                            "PythonDataClient: HTTP {} from {} — {}",
                            status,
                            url,
                            body.chars().take(200).collect::<String>()
                        );
                    }

                    // Server error or 429 — retry with backoff.
                    if attempt < max {
                        let delay = base_delay * 2_u64.pow(attempt);
                        warn!(
                            "PythonDataClient: HTTP {} (attempt {}/{}), retrying in {}ms",
                            status,
                            attempt + 1,
                            max + 1,
                            delay
                        );
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    } else {
                        bail!(
                            "PythonDataClient: HTTP {} after {} attempts for {}",
                            status,
                            max + 1,
                            url
                        );
                    }
                }

                Err(e) => {
                    // Network-level error — retry with backoff.
                    if attempt < max {
                        let delay = base_delay * 2_u64.pow(attempt);
                        warn!(
                            "PythonDataClient: network error (attempt {}/{}): {} — retrying in {}ms",
                            attempt + 1,
                            max + 1,
                            e,
                            delay
                        );
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    } else {
                        return Err(anyhow!(
                            "PythonDataClient: network error after {} attempts: {}",
                            max + 1,
                            e
                        ));
                    }
                }
            }
        }

        // Unreachable: the loop always returns or bails inside.
        unreachable!("fetch_with_retry loop exited without returning")
    }

    /// Check whether the Python data service is reachable and healthy.
    ///
    /// Hits `GET /health` and returns `true` if the response is 200 OK.
    /// Useful for a pre-flight check before starting warmup.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.config.base_url.trim_end_matches('/'));

        match self.client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                debug!("PythonDataClient: health check passed ({})", resp.status());
                true
            }
            Ok(resp) => {
                warn!(
                    "PythonDataClient: health check returned {} for {}",
                    resp.status(),
                    url
                );
                false
            }
            Err(e) => {
                warn!("PythonDataClient: health check failed — {}", e);
                false
            }
        }
    }

    // =========================================================================
    // Asset & symbol discovery
    // =========================================================================

    /// Fetch the list of available symbols from the data service.
    ///
    /// Hits `GET /bars/symbols` — lightweight, no DB queries.
    pub async fn fetch_symbols(&self) -> Result<SymbolsResponse> {
        let url = format!(
            "{}/bars/symbols",
            self.config.base_url.trim_end_matches('/')
        );
        let response = self
            .fetch_with_retry(&url, &[])
            .await
            .context("PythonDataClient: failed to fetch symbols")?;
        response
            .json()
            .await
            .context("PythonDataClient: failed to deserialise symbols response")
    }

    /// Fetch all tracked assets with their tickers and bar counts.
    ///
    /// Hits `GET /bars/assets` — includes DB bar counts so slightly heavier.
    pub async fn fetch_assets(&self) -> Result<AssetsResponse> {
        let url = format!("{}/bars/assets", self.config.base_url.trim_end_matches('/'));
        let response = self
            .fetch_with_retry(&url, &[])
            .await
            .context("PythonDataClient: failed to fetch assets")?;
        response
            .json()
            .await
            .context("PythonDataClient: failed to deserialise assets response")
    }

    // =========================================================================
    // Bar status & gap detection
    // =========================================================================

    /// Fetch bar counts, date ranges, and coverage for all stored symbols.
    ///
    /// Hits `GET /bars/status`.
    pub async fn fetch_bars_status(&self) -> Result<BarsStatusResponse> {
        let url = format!("{}/bars/status", self.config.base_url.trim_end_matches('/'));
        let response = self
            .fetch_with_retry(&url, &[])
            .await
            .context("PythonDataClient: failed to fetch bars status")?;
        response
            .json()
            .await
            .context("PythonDataClient: failed to deserialise bars status response")
    }

    /// Fetch a structured gap report for a specific symbol.
    ///
    /// Hits `GET /bars/{symbol}/gaps`.
    pub async fn fetch_gaps(
        &self,
        symbol: &str,
        days_back: u32,
        interval: &str,
    ) -> Result<GapReportResponse> {
        let encoded = urlencoding::encode(symbol);
        let url = format!(
            "{}/bars/{}/gaps",
            self.config.base_url.trim_end_matches('/'),
            encoded
        );
        let params = [
            ("days_back", days_back.to_string()),
            ("interval", interval.to_string()),
        ];
        let response = self
            .fetch_with_retry(&url, &params)
            .await
            .with_context(|| format!("PythonDataClient: failed to fetch gaps for {}", symbol))?;
        response.json().await.with_context(|| {
            format!(
                "PythonDataClient: failed to deserialise gaps for {}",
                symbol
            )
        })
    }

    // =========================================================================
    // Bar fill triggers
    // =========================================================================

    /// Trigger an incremental fill for a specific symbol.
    ///
    /// Hits `POST /bars/{symbol}/fill`.  For small fills this blocks and
    /// returns the result; for larger fills it returns a job token.
    pub async fn trigger_fill(
        &self,
        symbol: &str,
        days_back: u32,
        interval: &str,
    ) -> Result<FillJobResponse> {
        let encoded = urlencoding::encode(symbol);
        let url = format!(
            "{}/bars/{}/fill",
            self.config.base_url.trim_end_matches('/'),
            encoded
        );
        let body = serde_json::json!({
            "days_back": days_back,
            "interval": interval,
            "chunk_days": 5
        });
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("PythonDataClient: failed to trigger fill for {}", symbol))?;
        if !response.status().is_success() && !response.status().is_redirection() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!(
                "PythonDataClient: fill trigger returned HTTP {} — {}",
                status,
                text
            );
        }
        response.json().await.with_context(|| {
            format!(
                "PythonDataClient: failed to deserialise fill response for {}",
                symbol
            )
        })
    }

    /// Trigger a fill for all assets.
    ///
    /// Hits `POST /bars/fill/all`.  Returns immediately with a job ID.
    pub async fn trigger_fill_all(&self, days_back: u32) -> Result<FillJobResponse> {
        let url = format!(
            "{}/bars/fill/all",
            self.config.base_url.trim_end_matches('/')
        );
        let body = serde_json::json!({
            "days_back": days_back,
            "interval": "1m"
        });
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("PythonDataClient: failed to trigger fill-all")?;
        response
            .json()
            .await
            .context("PythonDataClient: failed to deserialise fill-all response")
    }

    /// Poll the status of a fill job.
    ///
    /// Hits `GET /bars/fill/status?job_id=...`.
    pub async fn poll_fill_status(&self, job_id: Option<&str>) -> Result<FillJobResponse> {
        let url = format!(
            "{}/bars/fill/status",
            self.config.base_url.trim_end_matches('/')
        );
        let params: Vec<(&str, String)> = if let Some(id) = job_id {
            vec![("job_id", id.to_string())]
        } else {
            vec![]
        };
        let response = self
            .fetch_with_retry(&url, &params)
            .await
            .context("PythonDataClient: failed to poll fill status")?;
        response
            .json()
            .await
            .context("PythonDataClient: failed to deserialise fill status")
    }

    // =========================================================================
    // Data source & live feed info
    // =========================================================================

    /// Query the current data source the engine is using.
    ///
    /// Hits `GET /api/analysis/data_source`.
    pub async fn fetch_data_source(&self) -> Result<String> {
        let url = format!(
            "{}/api/analysis/data_source",
            self.config.base_url.trim_end_matches('/')
        );
        let response = self
            .fetch_with_retry(&url, &[])
            .await
            .context("PythonDataClient: failed to fetch data source")?;
        let parsed: DataSourceResponse = response
            .json()
            .await
            .context("PythonDataClient: failed to deserialise data source response")?;
        Ok(parsed.data_source)
    }

    /// Fetch the live feed connection status.
    ///
    /// Hits `GET /api/analysis/live_feed`.
    pub async fn fetch_live_feed_status(&self) -> Result<LiveFeedStatus> {
        let url = format!(
            "{}/api/analysis/live_feed",
            self.config.base_url.trim_end_matches('/')
        );
        let response = self
            .fetch_with_retry(&url, &[])
            .await
            .context("PythonDataClient: failed to fetch live feed status")?;
        response
            .json()
            .await
            .context("PythonDataClient: failed to deserialise live feed status")
    }

    /// Return the base URL this client is configured to talk to.
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }
}

// ============================================================================
// Conversion helpers
// ============================================================================

/// Convert raw candle JSON into `CandleInput` structs for the `IndicatorActor`.
///
/// Timestamps are Unix milliseconds (i64) as expected by `CandleInput.timestamp`.
/// Volume is extracted from `serde_json::Value` — Python serialises it as an
/// integer but we accept floats and strings gracefully.
fn convert_candles(response: &CandlesResponse, symbol: &str, timeframe: &str) -> Vec<CandleInput> {
    let mut candles = Vec::with_capacity(response.candles.len());

    for raw in &response.candles {
        // Sanity-check: skip rows with nonsensical OHLCV values.
        if raw.close <= 0.0 || raw.high < raw.low {
            warn!(
                "PythonDataClient: skipping malformed candle for {} at ts={}",
                symbol, raw.timestamp
            );
            continue;
        }

        let volume = match &raw.volume {
            serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
            serde_json::Value::String(s) => s.parse::<f64>().unwrap_or(0.0),
            _ => 0.0,
        };

        candles.push(CandleInput {
            symbol: symbol.to_string(),
            // Label clearly so logs are easy to trace back to the data service.
            exchange: "fks_ruby".to_string(),
            timeframe: timeframe.to_string(),
            // CandleInput.timestamp is Unix milliseconds (i64).
            timestamp: raw.timestamp,
            open: raw.open,
            high: raw.high,
            low: raw.low,
            close: raw.close,
            volume,
        });
    }

    // The data service returns candles ascending, but sort defensively to
    // guarantee chronological order for the indicator accumulation loop.
    candles.sort_by_key(|c| c.timestamp);

    candles
}

// ============================================================================
// Symbol helpers
// ============================================================================

/// Map a Janus-internal symbol (e.g. `"BTCUSDT"`, `"MGC"`) to the form the
/// Python data service expects.
///
/// The data service's `_normalize_symbol()` handles most cases automatically,
/// but passing the right format avoids an extra normalisation round-trip and
/// makes logs clearer.
///
/// Rules applied here:
/// - CME futures short names (`MGC`, `MES`, `SIL`, etc.) → pass as-is;
///   the data service appends `=F` internally.
/// - Already-suffixed tickers (`MGC=F`) → pass as-is.
/// - Crypto pairs (`BTCUSDT`, `BTC-USDT`) → pass as-is.
/// - Exchange-prefixed symbols (`BINANCE:BTCUSDT`) → strip the prefix.
pub fn to_data_service_symbol(janus_symbol: &str) -> String {
    // Strip exchange prefix if present (e.g. "BINANCE:BTCUSDT" → "BTCUSDT")
    let stripped = if let Some(idx) = janus_symbol.find(':') {
        &janus_symbol[idx + 1..]
    } else {
        janus_symbol
    };

    stripped.to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        // Ensure default config builds without panicking even when env vars
        // are absent (they will be absent in CI).
        let config = PythonDataClientConfig::default();
        assert!(!config.base_url.is_empty());
        assert!(config.timeout_secs > 0);
        assert!(config.max_retries > 0);
        assert!(config.retry_delay_ms > 0);
    }

    #[test]
    fn test_to_data_service_symbol_plain() {
        assert_eq!(to_data_service_symbol("MGC"), "MGC");
        assert_eq!(to_data_service_symbol("MES"), "MES");
        assert_eq!(to_data_service_symbol("BTCUSDT"), "BTCUSDT");
    }

    #[test]
    fn test_to_data_service_symbol_with_suffix() {
        assert_eq!(to_data_service_symbol("MGC=F"), "MGC=F");
        assert_eq!(to_data_service_symbol("MES=F"), "MES=F");
    }

    #[test]
    fn test_to_data_service_symbol_strips_exchange_prefix() {
        assert_eq!(to_data_service_symbol("BINANCE:BTCUSDT"), "BTCUSDT");
        assert_eq!(to_data_service_symbol("KRAKEN:BTC"), "BTC");
    }

    #[test]
    fn test_convert_candles_ascending_sort() {
        // Intentionally out-of-order to verify defensive sort.
        let response = CandlesResponse {
            symbol: "MGC=F".to_string(),
            interval: "1m".to_string(),
            count: 3,
            candles: vec![
                RawCandle {
                    timestamp: 1700000120000,
                    open: 1952.0,
                    high: 1953.0,
                    low: 1951.0,
                    close: 1952.5,
                    volume: serde_json::json!(10),
                },
                RawCandle {
                    timestamp: 1700000000000,
                    open: 1950.0,
                    high: 1952.0,
                    low: 1949.0,
                    close: 1951.5,
                    volume: serde_json::json!(5),
                },
                RawCandle {
                    timestamp: 1700000060000,
                    open: 1951.5,
                    high: 1953.0,
                    low: 1950.5,
                    close: 1952.0,
                    volume: serde_json::json!(8),
                },
            ],
        };

        let candles = convert_candles(&response, "MGC", "1m");
        assert_eq!(candles.len(), 3);
        assert!(candles[0].timestamp < candles[1].timestamp);
        assert!(candles[1].timestamp < candles[2].timestamp);
        assert_eq!(candles[0].timestamp, 1700000000000);
    }

    #[test]
    fn test_convert_candles_skips_malformed() {
        let response = CandlesResponse {
            symbol: "MGC=F".to_string(),
            interval: "1m".to_string(),
            count: 2,
            candles: vec![
                // Valid candle
                RawCandle {
                    timestamp: 1700000000000,
                    open: 1950.0,
                    high: 1952.0,
                    low: 1949.0,
                    close: 1951.5,
                    volume: serde_json::json!(5),
                },
                // Malformed: close <= 0
                RawCandle {
                    timestamp: 1700000060000,
                    open: 0.0,
                    high: 0.0,
                    low: 0.0,
                    close: 0.0,
                    volume: serde_json::json!(0),
                },
            ],
        };

        let candles = convert_candles(&response, "MGC", "1m");
        assert_eq!(candles.len(), 1, "malformed candle should be skipped");
        assert_eq!(candles[0].close, 1951.5);
    }

    #[test]
    fn test_convert_candles_volume_as_float_string() {
        let response = CandlesResponse {
            symbol: "BTC".to_string(),
            interval: "1m".to_string(),
            count: 1,
            candles: vec![RawCandle {
                timestamp: 1700000000000,
                open: 30000.0,
                high: 30100.0,
                low: 29900.0,
                close: 30050.0,
                // Volume as a string — graceful fallback path.
                volume: serde_json::json!("123.45"),
            }],
        };

        let candles = convert_candles(&response, "BTC", "1m");
        assert_eq!(candles.len(), 1);
        assert!((candles[0].volume - 123.45).abs() < f64::EPSILON);
    }

    #[test]
    fn test_candle_fetch_request_defaults() {
        let req = CandleFetchRequest {
            symbol: "MGC".to_string(),
            ..Default::default()
        };
        assert_eq!(req.interval, "1m");
        assert_eq!(req.limit, 500);
        assert_eq!(req.days_back, 30);
        assert!(req.auto_fill);
    }

    #[test]
    fn test_client_builds_from_env() {
        // Should not panic even with missing env vars (uses defaults).
        let client = PythonDataClient::from_env();
        assert!(
            client.is_ok(),
            "PythonDataClient::from_env() should not fail"
        );
    }

    #[test]
    fn test_urlencoding_handles_equals() {
        // The = in MGC=F must be percent-encoded so it doesn't break the URL path.
        let encoded = urlencoding::encode("MGC=F");
        assert!(
            !encoded.contains('='),
            "equals sign should be percent-encoded, got: {}",
            encoded
        );
    }

    #[test]
    fn test_asset_info_deserialise() {
        let json = r#"{"name": "Gold", "ticker": "MGC=F", "symbol": "MGC", "bar_count": 15000, "has_data": true}"#;
        let info: AssetInfo = serde_json::from_str(json).expect("deserialise AssetInfo");
        assert_eq!(info.name, "Gold");
        assert_eq!(info.ticker, "MGC=F");
        assert_eq!(info.bar_count, 15000);
        assert!(info.has_data);
    }

    #[test]
    fn test_symbols_response_deserialise() {
        let json = r#"{"symbols": ["MGC", "MES", "BTCUSDT"], "assets": []}"#;
        let resp: SymbolsResponse =
            serde_json::from_str(json).expect("deserialise SymbolsResponse");
        assert_eq!(resp.symbols.len(), 3);
        assert_eq!(resp.symbols[0], "MGC");
    }

    #[test]
    fn test_gap_report_deserialise() {
        let json = r#"{
            "symbol": "MGC=F",
            "total_bars": 10000,
            "expected_bars": 10500,
            "coverage_pct": 95.2,
            "gaps": [{"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z", "missing_bars": 60}]
        }"#;
        let resp: GapReportResponse =
            serde_json::from_str(json).expect("deserialise GapReportResponse");
        assert_eq!(resp.symbol, "MGC=F");
        assert_eq!(resp.gaps.len(), 1);
        assert_eq!(resp.gaps[0].missing_bars, 60);
    }

    #[test]
    fn test_fill_job_response_deserialise() {
        let json =
            r#"{"status": "complete", "job_id": "abc-123", "bars_added": 500, "filling": false}"#;
        let resp: FillJobResponse =
            serde_json::from_str(json).expect("deserialise FillJobResponse");
        assert_eq!(resp.status, "complete");
        assert_eq!(resp.bars_added, 500);
        assert!(!resp.filling);
    }

    #[test]
    fn test_live_feed_status_deserialise() {
        let json = r#"{"connected": true, "status": "streaming", "symbols": ["MGC", "MES"]}"#;
        let resp: LiveFeedStatus = serde_json::from_str(json).expect("deserialise LiveFeedStatus");
        assert!(resp.connected);
        assert_eq!(resp.status, "streaming");
    }

    #[test]
    fn test_client_base_url() {
        let client = PythonDataClient::from_env().unwrap();
        assert!(!client.base_url().is_empty());
    }
}
