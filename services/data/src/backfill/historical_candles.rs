//! Historical Candle Fetcher
//!
//! Fetches historical kline/candlestick data from Binance REST API for deep
//! indicator warmup. This allows indicators like EMA-200 to be ready immediately
//! on startup without waiting 200+ minutes for real-time candles.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let fetcher = HistoricalCandleFetcher::new();
//! let candles = fetcher.fetch_candles("BTCUSD", "1m", 500).await?;
//! ```
//!
//! ## Supported Timeframes
//!
//! - 1m, 3m, 5m, 15m, 30m (minutes)
//! - 1h, 2h, 4h, 6h, 8h, 12h (hours)
//! - 1d, 3d (days)
//! - 1w (week)
//! - 1M (month)

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use crate::actors::indicator::CandleInput;

/// Binance API base URL
const BINANCE_API_BASE: &str = "https://api.binance.com";

/// Maximum candles per request (Binance limit)
const MAX_CANDLES_PER_REQUEST: usize = 1000;

/// Rate limit delay between requests (ms)
const RATE_LIMIT_DELAY_MS: u64 = 100;

/// Configuration for historical candle fetching
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Base URL for Binance API
    pub api_base: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Delay between paginated requests (ms)
    pub rate_limit_delay_ms: u64,
    /// Maximum retries per request
    pub max_retries: u32,
    /// Retry delay (ms)
    pub retry_delay_ms: u64,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            api_base: BINANCE_API_BASE.to_string(),
            timeout_secs: 30,
            rate_limit_delay_ms: RATE_LIMIT_DELAY_MS,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// Result of a historical fetch operation
#[derive(Debug)]
pub struct FetchResult {
    pub symbol: String,
    pub timeframe: String,
    pub candles_fetched: usize,
    pub oldest_timestamp: i64,
    pub newest_timestamp: i64,
    pub duration_ms: u64,
}

/// Historical candle fetcher for Binance
pub struct HistoricalCandleFetcher {
    client: Client,
    config: FetchConfig,
}

impl HistoricalCandleFetcher {
    /// Create a new fetcher with default configuration
    pub fn new() -> Self {
        Self::with_config(FetchConfig::default())
    }

    /// Create a new fetcher with custom configuration
    pub fn with_config(config: FetchConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Fetch historical candles for a symbol/timeframe
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSD")
    /// * `timeframe` - Candle interval (e.g., "1m", "5m", "1h")
    /// * `limit` - Maximum number of candles to fetch
    ///
    /// # Returns
    /// Vector of CandleInput in chronological order (oldest first)
    pub async fn fetch_candles(
        &self,
        symbol: &str,
        timeframe: &str,
        limit: usize,
    ) -> Result<Vec<CandleInput>> {
        let start = std::time::Instant::now();

        info!(
            "HistoricalFetch: Fetching {} {} candles for {}",
            limit, timeframe, symbol
        );

        // Binance uses different interval notation for some timeframes
        let interval = Self::normalize_interval(timeframe);

        let mut all_candles = Vec::with_capacity(limit);
        let mut end_time: Option<i64> = None;

        // Fetch in batches (Binance limits to 1000 per request)
        while all_candles.len() < limit {
            let batch_limit = std::cmp::min(MAX_CANDLES_PER_REQUEST, limit - all_candles.len());

            let candles = self
                .fetch_batch(symbol, &interval, batch_limit, end_time)
                .await?;

            if candles.is_empty() {
                debug!("HistoricalFetch: No more candles available");
                break;
            }

            // Update end_time for next batch (fetch older candles)
            if let Some(oldest) = candles.first() {
                end_time = Some(oldest.timestamp - 1);
            }

            // Prepend to maintain chronological order
            let mut new_candles = candles;
            new_candles.append(&mut all_candles);
            all_candles = new_candles;

            debug!(
                "HistoricalFetch: Fetched batch, total: {} candles",
                all_candles.len()
            );

            // Rate limiting between requests
            if all_candles.len() < limit {
                tokio::time::sleep(Duration::from_millis(self.config.rate_limit_delay_ms)).await;
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        if all_candles.is_empty() {
            warn!(
                "HistoricalFetch: No candles fetched for {}:{}",
                symbol, timeframe
            );
        } else {
            info!(
                "HistoricalFetch: Fetched {} candles for {}:{} in {}ms (range: {} to {})",
                all_candles.len(),
                symbol,
                timeframe,
                duration_ms,
                all_candles.first().map(|c| c.timestamp).unwrap_or(0),
                all_candles.last().map(|c| c.timestamp).unwrap_or(0)
            );
        }

        Ok(all_candles)
    }

    /// Fetch a batch of candles from Binance
    async fn fetch_batch(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        end_time: Option<i64>,
    ) -> Result<Vec<CandleInput>> {
        let mut retries = 0;

        loop {
            match self
                .fetch_batch_inner(symbol, interval, limit, end_time)
                .await
            {
                Ok(candles) => return Ok(candles),
                Err(e) => {
                    retries += 1;
                    if retries > self.config.max_retries {
                        return Err(e);
                    }

                    warn!(
                        "HistoricalFetch: Retry {}/{} after error: {}",
                        retries, self.config.max_retries, e
                    );

                    tokio::time::sleep(Duration::from_millis(
                        self.config.retry_delay_ms * retries as u64,
                    ))
                    .await;
                }
            }
        }
    }

    /// Inner fetch implementation
    async fn fetch_batch_inner(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        end_time: Option<i64>,
    ) -> Result<Vec<CandleInput>> {
        let url = format!("{}/api/v3/klines", self.config.api_base);

        let mut params = vec![
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.to_string()),
        ];

        if let Some(end) = end_time {
            params.push(("endTime", end.to_string()));
        }

        debug!(
            "HistoricalFetch: Request to {} with params: {:?}",
            url, params
        );

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await
            .context("Failed to send request to Binance")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Binance API error: status={}, body={}",
                status,
                body.chars().take(200).collect::<String>()
            );
        }

        let klines: Vec<BinanceKline> = response
            .json()
            .await
            .context("Failed to parse Binance kline response")?;

        // Convert to CandleInput
        let candles: Vec<CandleInput> = klines
            .into_iter()
            .map(|k| CandleInput {
                symbol: symbol.to_string(),
                exchange: "binance".to_string(),
                timeframe: Self::denormalize_interval(interval),
                timestamp: k.open_time,
                open: k.open.parse().unwrap_or(0.0),
                high: k.high.parse().unwrap_or(0.0),
                low: k.low.parse().unwrap_or(0.0),
                close: k.close.parse().unwrap_or(0.0),
                volume: k.volume.parse().unwrap_or(0.0),
            })
            .collect();

        Ok(candles)
    }

    /// Normalize timeframe to Binance interval format
    fn normalize_interval(timeframe: &str) -> String {
        // Binance uses lowercase intervals
        timeframe.to_lowercase()
    }

    /// Convert Binance interval back to our timeframe format
    fn denormalize_interval(interval: &str) -> String {
        interval.to_string()
    }

    /// Fetch candles for multiple symbols in parallel
    pub async fn fetch_multiple(
        &self,
        pairs: &[(String, String)], // (symbol, timeframe)
        limit: usize,
    ) -> Vec<(String, String, Result<Vec<CandleInput>>)> {
        let mut results = Vec::with_capacity(pairs.len());

        // Process sequentially to respect rate limits
        // Could be parallelized with proper rate limiting
        for (symbol, timeframe) in pairs {
            let result = self.fetch_candles(symbol, timeframe, limit).await;
            results.push((symbol.clone(), timeframe.clone(), result));

            // Rate limiting between different symbol requests
            tokio::time::sleep(Duration::from_millis(self.config.rate_limit_delay_ms)).await;
        }

        results
    }

    /// Get supported timeframes
    pub fn supported_timeframes() -> &'static [&'static str] {
        &[
            "1m", "3m", "5m", "15m", "30m", // Minutes
            "1h", "2h", "4h", "6h", "8h", "12h", // Hours
            "1d", "3d", // Days
            "1w", // Week
            "1M", // Month
        ]
    }

    /// Check if a timeframe is supported
    pub fn is_supported_timeframe(timeframe: &str) -> bool {
        Self::supported_timeframes().contains(&timeframe)
    }
}

impl Default for HistoricalCandleFetcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Binance kline response format
/// [
///   1499040000000,      // Kline open time
///   "0.01634000",       // Open price
///   "0.80000000",       // High price
///   "0.01575800",       // Low price
///   "0.01577100",       // Close price
///   "148976.11427815",  // Volume
///   1499644799999,      // Kline close time
///   "2434.19055334",    // Quote asset volume
///   308,                // Number of trades
///   "1756.87402397",    // Taker buy base asset volume
///   "28.46694368",      // Taker buy quote asset volume
///   "0"                 // Unused field
/// ]
#[derive(Debug)]
struct BinanceKline {
    open_time: i64,
    open: String,
    high: String,
    low: String,
    close: String,
    volume: String,
    _close_time: i64,
    _quote_volume: String,
    _trades: i64,
    _taker_buy_base: String,
    _taker_buy_quote: String,
    _ignore: String,
}

// Custom deserializer for Binance kline array format
impl<'de> Deserialize<'de> for BinanceKline {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let arr: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;

        if arr.len() < 12 {
            return Err(serde::de::Error::custom("Invalid kline array length"));
        }

        Ok(BinanceKline {
            open_time: arr[0].as_i64().unwrap_or(0),
            open: arr[1].as_str().unwrap_or("0").to_string(),
            high: arr[2].as_str().unwrap_or("0").to_string(),
            low: arr[3].as_str().unwrap_or("0").to_string(),
            close: arr[4].as_str().unwrap_or("0").to_string(),
            volume: arr[5].as_str().unwrap_or("0").to_string(),
            _close_time: arr[6].as_i64().unwrap_or(0),
            _quote_volume: arr[7].as_str().unwrap_or("0").to_string(),
            _trades: arr[8].as_i64().unwrap_or(0),
            _taker_buy_base: arr[9].as_str().unwrap_or("0").to_string(),
            _taker_buy_quote: arr[10].as_str().unwrap_or("0").to_string(),
            _ignore: arr[11].as_str().unwrap_or("0").to_string(),
        })
    }
}

/// Deep warmup function that fetches from Binance and feeds to indicator actor
pub async fn deep_warmup(
    indicator_actor: &std::sync::Arc<crate::actors::IndicatorActor>,
    symbols: &[String],
    timeframes: &[String],
    candles_per_pair: usize,
) -> Result<Vec<FetchResult>> {
    let fetcher = HistoricalCandleFetcher::new();
    let mut results = Vec::new();

    for symbol in symbols {
        for timeframe in timeframes {
            let start = std::time::Instant::now();

            info!(
                "DeepWarmup: Fetching {} {} candles for {}",
                candles_per_pair, timeframe, symbol
            );

            match fetcher
                .fetch_candles(symbol, timeframe, candles_per_pair)
                .await
            {
                Ok(candles) => {
                    if candles.is_empty() {
                        warn!(
                            "DeepWarmup: No candles fetched for {}:{}",
                            symbol, timeframe
                        );
                        continue;
                    }

                    let oldest = candles.first().map(|c| c.timestamp).unwrap_or(0);
                    let newest = candles.last().map(|c| c.timestamp).unwrap_or(0);
                    let count = candles.len();

                    // Feed candles to indicator actor
                    let sender = indicator_actor.get_sender();
                    let mut sent = 0;

                    for candle in candles {
                        use crate::actors::indicator::IndicatorMessage;
                        if sender.send(IndicatorMessage::Candle(candle)).is_err() {
                            error!("DeepWarmup: Failed to send candle to indicator actor");
                            break;
                        }
                        sent += 1;

                        // Yield periodically to allow processing
                        if sent % 50 == 0 {
                            tokio::task::yield_now().await;
                        }
                    }

                    let duration_ms = start.elapsed().as_millis() as u64;

                    info!(
                        "DeepWarmup: Sent {} candles for {}:{} in {}ms",
                        sent, symbol, timeframe, duration_ms
                    );

                    results.push(FetchResult {
                        symbol: symbol.clone(),
                        timeframe: timeframe.clone(),
                        candles_fetched: count,
                        oldest_timestamp: oldest,
                        newest_timestamp: newest,
                        duration_ms,
                    });
                }
                Err(e) => {
                    error!(
                        "DeepWarmup: Failed to fetch candles for {}:{}: {}",
                        symbol, timeframe, e
                    );
                }
            }

            // Rate limit between pairs
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fetch_config_default() {
        let config = FetchConfig::default();
        assert_eq!(config.api_base, BINANCE_API_BASE);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_supported_timeframes() {
        let timeframes = HistoricalCandleFetcher::supported_timeframes();
        assert!(timeframes.contains(&"1m"));
        assert!(timeframes.contains(&"5m"));
        assert!(timeframes.contains(&"15m"));
        assert!(timeframes.contains(&"1h"));
        assert!(timeframes.contains(&"4h"));
        assert!(timeframes.contains(&"1d"));
    }

    #[test]
    fn test_is_supported_timeframe() {
        assert!(HistoricalCandleFetcher::is_supported_timeframe("1m"));
        assert!(HistoricalCandleFetcher::is_supported_timeframe("4h"));
        assert!(!HistoricalCandleFetcher::is_supported_timeframe("2m"));
        assert!(!HistoricalCandleFetcher::is_supported_timeframe("invalid"));
    }

    #[test]
    fn test_normalize_interval() {
        assert_eq!(HistoricalCandleFetcher::normalize_interval("1m"), "1m");
        assert_eq!(HistoricalCandleFetcher::normalize_interval("1H"), "1h");
        assert_eq!(HistoricalCandleFetcher::normalize_interval("4H"), "4h");
    }

    #[test]
    fn test_fetch_result() {
        let result = FetchResult {
            symbol: "BTCUSD".to_string(),
            timeframe: "1m".to_string(),
            candles_fetched: 500,
            oldest_timestamp: 1672531200000,
            newest_timestamp: 1672561200000,
            duration_ms: 250,
        };

        assert_eq!(result.symbol, "BTCUSD");
        assert_eq!(result.candles_fetched, 500);
    }
}
