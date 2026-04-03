//! OHLC Data Collector
//!
//! Fetches historical and real-time OHLC candle data from Kraken's REST API
//! and stores it in SQLite for backtesting.
//!
//! # Features
//!
//! - Kraken REST API integration for OHLC data
//! - SQLite storage with efficient upsert operations
//! - Gap detection and backfilling
//! - Rate limiting to respect API limits
//! - Multiple interval support (1m, 5m, 15m, 1h, 4h, 1d)
//!
//! # Kraken OHLC API
//!
//! Endpoint: `GET https://api.kraken.com/0/public/OHLC`
//!
//! Parameters:
//! - `pair`: Asset pair (e.g., "XXBTZUSD")
//! - `interval`: Time interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
//! - `since`: Return data since given timestamp (optional)
//!
//! Response format:
//! ```json
//! {
//!   "error": [],
//!   "result": {
//!     "XXBTZUSD": [
//!       [1616662800, "52000.0", "52100.0", "51900.0", "52050.0", "51975.0", "10.5", 150]
//!     ],
//!     "last": 1616662800
//!   }
//! }
//! ```
//! Array format: [time, open, high, low, close, vwap, volume, count]

use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::{Pool, Row, Sqlite, sqlite::SqlitePoolOptions};
use std::collections::HashMap;

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::{OptimizerServiceConfig, get_kraken_pair};

/// OHLC candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcCandle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: f64,
    pub count: i64,
}

impl OhlcCandle {
    /// Get timestamp as DateTime
    #[allow(dead_code)]
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_opt(self.timestamp, 0).unwrap()
    }
}

/// Collection statistics
#[derive(Debug, Clone, Default)]
pub struct CollectionStats {
    pub total_candles: usize,
    pub new_candles: usize,
    pub assets_collected: Vec<String>,
    pub intervals_collected: Vec<u32>,
    pub duration_secs: f64,
    pub errors: Vec<String>,
}

/// Data gap information
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DataGap {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub missing_candles: usize,
}

/// OHLC Data Collector
pub struct OhlcCollector {
    /// HTTP client for API requests
    client: Client,

    /// SQLite database pool
    db: Pool<Sqlite>,

    /// Configuration
    config: OptimizerServiceConfig,

    /// Rate limiter state (last request time)
    last_request: Arc<RwLock<Option<Instant>>>,
}

impl OhlcCollector {
    /// Create a new OHLC collector
    pub async fn new(config: OptimizerServiceConfig) -> Result<Self> {
        // Ensure database directory exists
        let db_path = config.db_path();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create DB directory: {}", parent.display()))?;
        }

        // Create SQLite connection pool
        let db_url = format!("sqlite:{}?mode=rwc", db_path.display());
        let db = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await
            .with_context(|| format!("Failed to connect to database: {}", db_path.display()))?;

        // Initialize database schema
        Self::init_database(&db).await?;

        // Create HTTP client
        let client = Client::builder()
            .timeout(config.request_timeout)
            .user_agent("JANUS-Optimizer/1.0")
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            db,
            config,
            last_request: Arc::new(RwLock::new(None)),
        })
    }

    /// Initialize database schema
    async fn init_database(db: &Pool<Sqlite>) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS ohlc_candles (
                asset TEXT NOT NULL,
                interval INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                vwap REAL NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (asset, interval, timestamp)
            )
            "#,
        )
        .execute(db)
        .await
        .context("Failed to create ohlc_candles table")?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_ohlc_asset_interval_time
            ON ohlc_candles (asset, interval, timestamp DESC)
            "#,
        )
        .execute(db)
        .await
        .context("Failed to create index")?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS collection_metadata (
                asset TEXT NOT NULL,
                interval INTEGER NOT NULL,
                first_timestamp INTEGER,
                last_timestamp INTEGER,
                candle_count INTEGER DEFAULT 0,
                last_updated INTEGER,
                PRIMARY KEY (asset, interval)
            )
            "#,
        )
        .execute(db)
        .await
        .context("Failed to create metadata table")?;

        Ok(())
    }

    /// Apply rate limiting before making a request
    async fn rate_limit(&self) {
        let min_interval = Duration::from_secs_f64(1.0 / self.config.rate_limit_per_second);

        let mut last = self.last_request.write().await;
        if let Some(last_time) = *last {
            let elapsed = last_time.elapsed();
            if elapsed < min_interval {
                tokio::time::sleep(min_interval - elapsed).await;
            }
        }
        *last = Some(Instant::now());
    }

    /// Fetch OHLC data from Kraken API
    pub async fn fetch_ohlc(
        &self,
        asset: &str,
        interval: u32,
        since: Option<i64>,
    ) -> Result<Vec<OhlcCandle>> {
        self.rate_limit().await;

        let pair = get_kraken_pair(asset);
        let mut url = format!(
            "{}/0/public/OHLC?pair={}&interval={}",
            self.config.kraken_api_url, pair, interval
        );

        if let Some(since_ts) = since {
            url.push_str(&format!("&since={}", since_ts));
        }

        debug!(
            "Fetching OHLC: {} interval={} since={:?}",
            asset, interval, since
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("Failed to fetch OHLC for {}", asset))?;

        if !response.status().is_success() {
            anyhow::bail!("Kraken API error: {} for {}", response.status(), asset);
        }

        let data: KrakenOhlcResponse = response
            .json()
            .await
            .context("Failed to parse Kraken response")?;

        if !data.error.is_empty() {
            anyhow::bail!("Kraken API errors: {:?}", data.error);
        }

        // Find the result for our pair (Kraken may use different key formats)
        let ohlc_data = data
            .result
            .iter()
            .find(|(key, _)| {
                key.as_str() != "last"
                    && (key.contains(&asset.to_uppercase()) || key.as_str() == pair)
            })
            .map(|(_, v)| v)
            .ok_or_else(|| anyhow::anyhow!("No OHLC data found for {}", asset))?;

        let candles: Vec<OhlcCandle> = ohlc_data
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("OHLC data is not an array"))?
            .iter()
            .filter_map(|arr| {
                let arr = arr.as_array()?;
                if arr.len() >= 8 {
                    Some(OhlcCandle {
                        timestamp: arr[0].as_i64()?,
                        open: parse_f64(&arr[1])?,
                        high: parse_f64(&arr[2])?,
                        low: parse_f64(&arr[3])?,
                        close: parse_f64(&arr[4])?,
                        vwap: parse_f64(&arr[5])?,
                        volume: parse_f64(&arr[6])?,
                        count: arr[7].as_i64()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        debug!(
            "Fetched {} candles for {} interval={}",
            candles.len(),
            asset,
            interval
        );
        Ok(candles)
    }

    /// Insert candles into database
    pub async fn insert_candles(
        &self,
        asset: &str,
        interval: u32,
        candles: &[OhlcCandle],
    ) -> Result<usize> {
        if candles.is_empty() {
            return Ok(0);
        }

        let mut inserted = 0;

        for candle in candles {
            let result = sqlx::query(
                r#"
                INSERT OR REPLACE INTO ohlc_candles
                (asset, interval, timestamp, open, high, low, close, volume, vwap, count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
            )
            .bind(asset)
            .bind(interval as i64)
            .bind(candle.timestamp)
            .bind(candle.open)
            .bind(candle.high)
            .bind(candle.low)
            .bind(candle.close)
            .bind(candle.volume)
            .bind(candle.vwap)
            .bind(candle.count)
            .execute(&self.db)
            .await;

            if result.is_ok() {
                inserted += 1;
            }
        }

        // Update metadata
        self.update_metadata(asset, interval).await?;

        debug!(
            "Inserted {} candles for {} interval={}",
            inserted, asset, interval
        );
        Ok(inserted)
    }

    /// Update collection metadata
    async fn update_metadata(&self, asset: &str, interval: u32) -> Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO collection_metadata
            (asset, interval, first_timestamp, last_timestamp, candle_count, last_updated)
            SELECT
                ?,
                ?,
                MIN(timestamp),
                MAX(timestamp),
                COUNT(*),
                ?
            FROM ohlc_candles
            WHERE asset = ? AND interval = ?
            "#,
        )
        .bind(asset)
        .bind(interval as i64)
        .bind(Utc::now().timestamp())
        .bind(asset)
        .bind(interval as i64)
        .execute(&self.db)
        .await?;

        Ok(())
    }

    /// Get candles from database
    pub async fn get_candles(
        &self,
        asset: &str,
        interval: u32,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<OhlcCandle>> {
        let start_ts = start.map(|d| d.timestamp()).unwrap_or(0);
        let end_ts = end.map(|d| d.timestamp()).unwrap_or(i64::MAX);

        let rows = sqlx::query(
            r#"
            SELECT timestamp, open, high, low, close, volume, vwap, count
            FROM ohlc_candles
            WHERE asset = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            "#,
        )
        .bind(asset)
        .bind(interval as i64)
        .bind(start_ts)
        .bind(end_ts)
        .fetch_all(&self.db)
        .await?;

        let candles: Vec<OhlcCandle> = rows
            .iter()
            .map(|row| OhlcCandle {
                timestamp: row.get("timestamp"),
                open: row.get("open"),
                high: row.get("high"),
                low: row.get("low"),
                close: row.get("close"),
                volume: row.get("volume"),
                vwap: row.get("vwap"),
                count: row.get("count"),
            })
            .collect();

        Ok(candles)
    }

    /// Get collection metadata for an asset/interval
    pub async fn get_metadata(
        &self,
        asset: &str,
        interval: u32,
    ) -> Result<Option<CollectionMetadata>> {
        let row = sqlx::query(
            r#"
            SELECT first_timestamp, last_timestamp, candle_count, last_updated
            FROM collection_metadata
            WHERE asset = ? AND interval = ?
            "#,
        )
        .bind(asset)
        .bind(interval as i64)
        .fetch_optional(&self.db)
        .await?;

        Ok(row.map(|r| CollectionMetadata {
            first_timestamp: r.get("first_timestamp"),
            last_timestamp: r.get("last_timestamp"),
            candle_count: r.get::<i64, _>("candle_count") as usize,
            last_updated: r.get("last_updated"),
        }))
    }

    /// Fetch latest candles (update existing data)
    pub async fn fetch_latest(&self, asset: &str, interval: u32) -> Result<usize> {
        let metadata = self.get_metadata(asset, interval).await?;
        let since = metadata.map(|m| m.last_timestamp);

        let candles = self.fetch_ohlc(asset, interval, since).await?;
        self.insert_candles(asset, interval, &candles).await
    }

    /// Fetch historical data going back N days
    pub async fn fetch_historical(&self, asset: &str, interval: u32, days: u32) -> Result<usize> {
        let now = Utc::now();
        let start = now - ChronoDuration::days(days as i64);
        let since = start.timestamp();

        info!(
            "Fetching {} days of historical data for {} interval={}",
            days, asset, interval
        );

        let candles = self.fetch_ohlc(asset, interval, Some(since)).await?;
        self.insert_candles(asset, interval, &candles).await
    }

    /// Collect all data for configured assets
    pub async fn collect_all(&self, fetch_historical: bool) -> Result<CollectionStats> {
        let start = Instant::now();
        let mut stats = CollectionStats::default();

        for asset in &self.config.assets {
            for &interval in &self.config.collection_intervals {
                match if fetch_historical {
                    self.fetch_historical(asset, interval, self.config.historical_days)
                        .await
                } else {
                    self.fetch_latest(asset, interval).await
                } {
                    Ok(count) => {
                        stats.total_candles += count;
                        stats.new_candles += count;
                        if !stats.assets_collected.contains(asset) {
                            stats.assets_collected.push(asset.clone());
                        }
                        if !stats.intervals_collected.contains(&interval) {
                            stats.intervals_collected.push(interval);
                        }
                    }
                    Err(e) => {
                        let err_msg = format!("{} interval={}: {}", asset, interval, e);
                        warn!("Collection error: {}", err_msg);
                        stats.errors.push(err_msg);
                    }
                }
            }
        }

        stats.duration_secs = start.elapsed().as_secs_f64();
        Ok(stats)
    }

    /// Update data (fetch latest for all assets)
    pub async fn update(&self) -> Result<CollectionStats> {
        self.collect_all(false).await
    }

    /// Get candle count for an asset/interval
    pub async fn get_candle_count(&self, asset: &str, interval: u32) -> Result<usize> {
        let row = sqlx::query(
            "SELECT COUNT(*) as count FROM ohlc_candles WHERE asset = ? AND interval = ?",
        )
        .bind(asset)
        .bind(interval as i64)
        .fetch_one(&self.db)
        .await?;

        Ok(row.get::<i64, _>("count") as usize)
    }

    /// Get data date range for an asset/interval
    #[allow(dead_code)]
    pub async fn get_date_range(
        &self,
        asset: &str,
        interval: u32,
    ) -> Result<Option<(DateTime<Utc>, DateTime<Utc>)>> {
        let row = sqlx::query(
            "SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM ohlc_candles WHERE asset = ? AND interval = ?",
        )
        .bind(asset)
        .bind(interval as i64)
        .fetch_one(&self.db)
        .await?;

        let min_ts: Option<i64> = row.get("min_ts");
        let max_ts: Option<i64> = row.get("max_ts");

        match (min_ts, max_ts) {
            (Some(min), Some(max)) => Ok(Some((
                Utc.timestamp_opt(min, 0).unwrap(),
                Utc.timestamp_opt(max, 0).unwrap(),
            ))),
            _ => Ok(None),
        }
    }

    /// Detect gaps in data
    #[allow(dead_code)]
    pub async fn detect_gaps(&self, asset: &str, interval: u32) -> Result<Vec<DataGap>> {
        let candles = self.get_candles(asset, interval, None, None).await?;

        if candles.len() < 2 {
            return Ok(vec![]);
        }

        let expected_gap = (interval as i64) * 60; // interval in seconds
        let mut gaps = Vec::new();

        for window in candles.windows(2) {
            let actual_gap = window[1].timestamp - window[0].timestamp;
            if actual_gap > expected_gap * 2 {
                // More than 2x expected gap
                let missing = (actual_gap / expected_gap) as usize - 1;
                gaps.push(DataGap {
                    start: window[0].datetime(),
                    end: window[1].datetime(),
                    missing_candles: missing,
                });
            }
        }

        Ok(gaps)
    }

    /// Get collection status for all assets
    #[allow(dead_code)]
    pub async fn get_status(&self) -> Result<HashMap<String, HashMap<u32, CollectionMetadata>>> {
        let mut status = HashMap::new();

        for asset in &self.config.assets {
            let mut asset_status = HashMap::new();
            for &interval in &self.config.collection_intervals {
                if let Some(meta) = self.get_metadata(asset, interval).await? {
                    asset_status.insert(interval, meta);
                }
            }
            status.insert(asset.clone(), asset_status);
        }

        Ok(status)
    }

    /// Vacuum database to reclaim space
    #[allow(dead_code)]
    pub async fn vacuum(&self) -> Result<()> {
        sqlx::query("VACUUM").execute(&self.db).await?;
        Ok(())
    }
}

/// Collection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    pub first_timestamp: i64,
    pub last_timestamp: i64,
    pub candle_count: usize,
    pub last_updated: i64,
}

impl CollectionMetadata {
    /// Get first timestamp as DateTime
    #[allow(dead_code)]
    pub fn first_datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_opt(self.first_timestamp, 0).unwrap()
    }

    /// Get last timestamp as DateTime
    #[allow(dead_code)]
    pub fn last_datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_opt(self.last_timestamp, 0).unwrap()
    }

    /// Get duration of data
    #[allow(dead_code)]
    pub fn duration(&self) -> ChronoDuration {
        ChronoDuration::seconds(self.last_timestamp - self.first_timestamp)
    }

    /// Get duration in days
    #[allow(dead_code)]
    pub fn days(&self) -> f64 {
        self.duration().num_seconds() as f64 / 86400.0
    }
}

/// Kraken OHLC API response
#[derive(Debug, Deserialize)]
struct KrakenOhlcResponse {
    error: Vec<String>,
    result: HashMap<String, serde_json::Value>,
}

/// Parse f64 from JSON value (handles both string and number)
fn parse_f64(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::String(s) => s.parse().ok(),
        serde_json::Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_f64() {
        assert_eq!(parse_f64(&serde_json::json!("50000.0")), Some(50000.0));
        assert_eq!(parse_f64(&serde_json::json!(50000.0)), Some(50000.0));
        assert_eq!(parse_f64(&serde_json::json!(50000)), Some(50000.0));
        assert_eq!(parse_f64(&serde_json::json!(null)), None);
    }

    #[test]
    fn test_ohlc_candle_datetime() {
        let candle = OhlcCandle {
            timestamp: 1700000000,
            open: 50000.0,
            high: 51000.0,
            low: 49000.0,
            close: 50500.0,
            volume: 100.0,
            vwap: 50250.0,
            count: 1000,
        };

        let dt = candle.datetime();
        assert_eq!(dt.timestamp(), 1700000000);
    }

    #[test]
    fn test_collection_metadata_days() {
        let meta = CollectionMetadata {
            first_timestamp: 1700000000,
            last_timestamp: 1700086400, // 1 day later
            candle_count: 24,
            last_updated: 1700086400,
        };

        assert!((meta.days() - 1.0).abs() < 0.01);
    }
}
