//! # OHLCV Data Loader
//!
//! Loads OHLCV bar data from CSV and Parquet files, producing `Vec<OhlcvBar>`
//! directly consumable by the `StrategyBacktester`.
//!
//! ## Supported Formats
//!
//! - **CSV**: With or without headers. Supports various timestamp formats.
//! - **Parquet**: Columnar format for high-performance loading of large datasets.
//!
//! ## Column Mapping
//!
//! The loader supports flexible column names via `OhlcvColumnMap`. Common
//! exchange export formats (Kraken, Binance, TradingView, generic) are
//! provided as presets.
//!
//! ## Timestamp Formats
//!
//! - Unix seconds (integer or float)
//! - Unix milliseconds
//! - Unix microseconds / nanoseconds
//! - ISO 8601 / RFC 3339 strings
//! - Custom `chrono` format strings
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_backtest::ohlcv_loader::{OhlcvLoader, OhlcvLoaderConfig};
//!
//! let config = OhlcvLoaderConfig::new("data/kraken_btcusd_15m.csv")
//!     .with_column_map(OhlcvColumnMap::kraken())
//!     .with_symbol("BTCUSD");
//!
//! let loader = OhlcvLoader::new(config);
//! let bars = loader.load()?;
//!
//! println!("Loaded {} bars", bars.len());
//! println!("First: {} @ {:.2}", bars[0].timestamp, bars[0].close);
//! ```

use crate::strategy_backtester::OhlcvBar;
use chrono::{DateTime, NaiveDateTime, Utc};
use polars::prelude::*;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info, warn};

// ============================================================================
// Errors
// ============================================================================

/// Errors that can occur during OHLCV data loading.
#[derive(Error, Debug)]
pub enum OhlcvLoaderError {
    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Missing required column '{0}'. Available columns: {1}")]
    MissingColumn(String, String),

    #[error("Unsupported file format: '{0}'. Use .csv or .parquet")]
    UnsupportedFormat(String),

    #[error("No data loaded (file is empty or all rows filtered out)")]
    NoData,

    #[error("Timestamp parse error at row {row}: {details}")]
    TimestampParse { row: usize, details: String },

    #[error("Invalid price data at row {row}: column '{column}' value is NaN or non-finite")]
    InvalidPrice { row: usize, column: String },

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Date range filter yielded no data: {start} to {end}")]
    NoDataInRange { start: String, end: String },
}

// ============================================================================
// Timestamp Format
// ============================================================================

/// How to interpret the timestamp column.
#[derive(Debug, Clone, Default)]
pub enum TimestampFormat {
    /// Unix timestamp in seconds (integer or float).
    UnixSeconds,
    /// Unix timestamp in milliseconds.
    UnixMillis,
    /// Unix timestamp in microseconds.
    UnixMicros,
    /// Unix timestamp in nanoseconds.
    UnixNanos,
    /// ISO 8601 / RFC 3339 string (e.g. "2024-01-15T12:00:00Z").
    Iso8601,
    /// Custom chrono format string (e.g. "%Y-%m-%d %H:%M:%S").
    Custom(String),
    /// Auto-detect: try numeric first, then ISO 8601.
    #[default]
    Auto,
}

// ============================================================================
// Column Map
// ============================================================================

/// Maps logical column names to actual column names in the data file.
///
/// Different exchanges and data providers use different column headers.
/// This struct lets you specify the mapping once.
#[derive(Debug, Clone)]
pub struct OhlcvColumnMap {
    /// Column name for the timestamp. Default: "timestamp"
    pub timestamp: String,
    /// Column name for the open price. Default: "open"
    pub open: String,
    /// Column name for the high price. Default: "high"
    pub high: String,
    /// Column name for the low price. Default: "low"
    pub low: String,
    /// Column name for the close price. Default: "close"
    pub close: String,
    /// Column name for the volume. Default: "volume"
    pub volume: String,
}

impl Default for OhlcvColumnMap {
    fn default() -> Self {
        Self {
            timestamp: "timestamp".to_string(),
            open: "open".to_string(),
            high: "high".to_string(),
            low: "low".to_string(),
            close: "close".to_string(),
            volume: "volume".to_string(),
        }
    }
}

impl OhlcvColumnMap {
    /// Preset for Kraken OHLCV CSV exports.
    ///
    /// Kraken exports typically have: `timestamp,open,high,low,close,vwap,volume,count`
    /// with Unix timestamps in seconds.
    pub fn kraken() -> Self {
        Self {
            timestamp: "timestamp".to_string(),
            open: "open".to_string(),
            high: "high".to_string(),
            low: "low".to_string(),
            close: "close".to_string(),
            volume: "volume".to_string(),
        }
    }

    /// Preset for Binance kline/candlestick CSV exports.
    ///
    /// Binance exports: `open_time,open,high,low,close,volume,close_time,...`
    /// with Unix timestamps in milliseconds.
    pub fn binance() -> Self {
        Self {
            timestamp: "open_time".to_string(),
            open: "open".to_string(),
            high: "high".to_string(),
            low: "low".to_string(),
            close: "close".to_string(),
            volume: "volume".to_string(),
        }
    }

    /// Preset for TradingView CSV exports.
    ///
    /// TradingView exports: `time,open,high,low,close,Volume`
    pub fn tradingview() -> Self {
        Self {
            timestamp: "time".to_string(),
            open: "open".to_string(),
            high: "high".to_string(),
            low: "low".to_string(),
            close: "close".to_string(),
            volume: "Volume".to_string(),
        }
    }

    /// Preset for headerless CSV files with positional columns.
    ///
    /// Assumes column order: timestamp, open, high, low, close, volume
    /// (columns will be named "column_1", "column_2", etc. by Polars)
    pub fn positional() -> Self {
        Self {
            timestamp: "column_1".to_string(),
            open: "column_2".to_string(),
            high: "column_3".to_string(),
            low: "column_4".to_string(),
            close: "column_5".to_string(),
            volume: "column_6".to_string(),
        }
    }

    /// Create a custom column map.
    pub fn custom(
        timestamp: &str,
        open: &str,
        high: &str,
        low: &str,
        close: &str,
        volume: &str,
    ) -> Self {
        Self {
            timestamp: timestamp.to_string(),
            open: open.to_string(),
            high: high.to_string(),
            low: low.to_string(),
            close: close.to_string(),
            volume: volume.to_string(),
        }
    }

    /// Try to auto-detect the column map from available column names.
    ///
    /// Performs case-insensitive matching and handles common aliases.
    pub fn auto_detect(columns: &[&str]) -> Self {
        let lower: Vec<String> = columns.iter().map(|c| c.to_lowercase()).collect();

        let find = |candidates: &[&str]| -> Option<String> {
            for candidate in candidates {
                for (i, col) in lower.iter().enumerate() {
                    if col == candidate {
                        return Some(columns[i].to_string());
                    }
                }
            }
            None
        };

        Self {
            timestamp: find(&[
                "timestamp",
                "time",
                "date",
                "datetime",
                "open_time",
                "opentime",
                "ts",
                "t",
            ])
            .unwrap_or_else(|| "timestamp".to_string()),
            open: find(&["open", "o", "open_price", "openprice"])
                .unwrap_or_else(|| "open".to_string()),
            high: find(&["high", "h", "high_price", "highprice", "max"])
                .unwrap_or_else(|| "high".to_string()),
            low: find(&["low", "l", "low_price", "lowprice", "min"])
                .unwrap_or_else(|| "low".to_string()),
            close: find(&["close", "c", "close_price", "closeprice", "last"])
                .unwrap_or_else(|| "close".to_string()),
            volume: find(&[
                "volume",
                "vol",
                "v",
                "base_volume",
                "basevolume",
                "qty",
                "quantity",
            ])
            .unwrap_or_else(|| "volume".to_string()),
        }
    }
}

// ============================================================================
// Loader Configuration
// ============================================================================

/// Configuration for the OHLCV data loader.
#[derive(Debug, Clone)]
pub struct OhlcvLoaderConfig {
    /// Path to the data file.
    pub path: PathBuf,

    /// Symbol name (metadata only — included in log messages).
    pub symbol: Option<String>,

    /// Column name mapping.
    pub column_map: OhlcvColumnMap,

    /// How to interpret the timestamp column.
    pub timestamp_format: TimestampFormat,

    /// Whether the CSV file has a header row.
    pub has_header: bool,

    /// Optional start time filter (inclusive).
    pub start_time: Option<DateTime<Utc>>,

    /// Optional end time filter (inclusive).
    pub end_time: Option<DateTime<Utc>>,

    /// Whether to sort bars by timestamp ascending after loading.
    pub sort_ascending: bool,

    /// Whether to drop rows with any NaN/null values in OHLCV columns.
    pub drop_invalid: bool,

    /// Whether to auto-detect column names from headers.
    pub auto_detect_columns: bool,

    /// CSV delimiter character (default: comma).
    pub csv_delimiter: u8,

    /// Maximum number of bars to load (None = unlimited).
    pub max_bars: Option<usize>,

    /// Number of leading bars to skip (useful for skipping warmup data).
    pub skip_bars: usize,
}

impl OhlcvLoaderConfig {
    /// Create a new config pointing to a data file.
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            symbol: None,
            column_map: OhlcvColumnMap::default(),
            timestamp_format: TimestampFormat::Auto,
            has_header: true,
            start_time: None,
            end_time: None,
            sort_ascending: true,
            drop_invalid: true,
            auto_detect_columns: true,
            csv_delimiter: b',',
            max_bars: None,
            skip_bars: 0,
        }
    }

    /// Set the symbol name.
    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Set the column mapping.
    pub fn with_column_map(mut self, map: OhlcvColumnMap) -> Self {
        self.column_map = map;
        self.auto_detect_columns = false;
        self
    }

    /// Set the timestamp format.
    pub fn with_timestamp_format(mut self, format: TimestampFormat) -> Self {
        self.timestamp_format = format;
        self
    }

    /// Set whether the CSV has a header row.
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        if !has_header {
            self.auto_detect_columns = false;
        }
        self
    }

    /// Filter bars to a time range.
    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Set the CSV delimiter.
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.csv_delimiter = delimiter;
        self
    }

    /// Limit the number of bars loaded.
    pub fn with_max_bars(mut self, max: usize) -> Self {
        self.max_bars = Some(max);
        self
    }

    /// Skip leading bars.
    pub fn with_skip_bars(mut self, skip: usize) -> Self {
        self.skip_bars = skip;
        self
    }

    /// Disable sorting.
    pub fn without_sorting(mut self) -> Self {
        self.sort_ascending = false;
        self
    }

    /// Use Kraken preset.
    pub fn kraken(mut self) -> Self {
        self.column_map = OhlcvColumnMap::kraken();
        self.timestamp_format = TimestampFormat::UnixSeconds;
        self.auto_detect_columns = false;
        self
    }

    /// Use Binance preset.
    pub fn binance(mut self) -> Self {
        self.column_map = OhlcvColumnMap::binance();
        self.timestamp_format = TimestampFormat::UnixMillis;
        self.auto_detect_columns = false;
        self
    }

    /// Use TradingView preset.
    pub fn tradingview(mut self) -> Self {
        self.column_map = OhlcvColumnMap::tradingview();
        self.timestamp_format = TimestampFormat::Iso8601;
        self.auto_detect_columns = false;
        self
    }

    /// Use positional columns (no header).
    pub fn positional(mut self) -> Self {
        self.has_header = false;
        self.column_map = OhlcvColumnMap::positional();
        self.auto_detect_columns = false;
        self
    }
}

// ============================================================================
// OHLCV Loader
// ============================================================================

/// Loads OHLCV bar data from CSV or Parquet files.
pub struct OhlcvLoader {
    config: OhlcvLoaderConfig,
}

impl OhlcvLoader {
    /// Create a new loader with the given configuration.
    pub fn new(config: OhlcvLoaderConfig) -> Self {
        Self { config }
    }

    /// Load all OHLCV bars from the configured file.
    ///
    /// Returns bars sorted by timestamp ascending (unless sorting is disabled).
    pub fn load(&self) -> Result<Vec<OhlcvBar>, OhlcvLoaderError> {
        let path = &self.config.path;
        let sym = self.config.symbol.as_deref().unwrap_or("unknown");

        info!(
            "Loading OHLCV data from '{}' for symbol '{}'",
            path.display(),
            sym
        );

        if !path.exists() {
            return Err(OhlcvLoaderError::FileNotFound(path.display().to_string()));
        }

        // Load the raw DataFrame
        let df = self.load_dataframe(path)?;
        debug!(
            "Raw DataFrame: {} rows, {} columns",
            df.height(),
            df.width()
        );

        if df.height() == 0 {
            return Err(OhlcvLoaderError::NoData);
        }

        // Resolve column map (auto-detect if enabled)
        let col_map = if self.config.auto_detect_columns && self.config.has_header {
            let col_names: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
            let detected = OhlcvColumnMap::auto_detect(&col_names);
            debug!(
                "Auto-detected columns: ts={}, o={}, h={}, l={}, c={}, v={}",
                detected.timestamp,
                detected.open,
                detected.high,
                detected.low,
                detected.close,
                detected.volume
            );
            detected
        } else {
            self.config.column_map.clone()
        };

        // Validate required columns exist
        self.validate_columns(&df, &col_map)?;

        // Convert DataFrame rows to OhlcvBar vec
        let mut bars = self.dataframe_to_bars(&df, &col_map)?;

        // Sort by timestamp
        if self.config.sort_ascending {
            bars.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        }

        // Apply time range filter
        bars = self.apply_time_filter(bars)?;

        // Apply skip/limit
        if self.config.skip_bars > 0 {
            if self.config.skip_bars >= bars.len() {
                return Err(OhlcvLoaderError::NoData);
            }
            bars = bars.split_off(self.config.skip_bars);
        }

        if let Some(max) = self.config.max_bars {
            bars.truncate(max);
        }

        if bars.is_empty() {
            return Err(OhlcvLoaderError::NoData);
        }

        info!(
            "Loaded {} OHLCV bars: {} → {} | price range {:.2}–{:.2}",
            bars.len(),
            bars.first().unwrap().timestamp.format("%Y-%m-%d %H:%M"),
            bars.last().unwrap().timestamp.format("%Y-%m-%d %H:%M"),
            bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min),
            bars.iter()
                .map(|b| b.high)
                .fold(f64::NEG_INFINITY, f64::max),
        );

        Ok(bars)
    }

    /// Load the raw Polars DataFrame from file.
    pub fn load_dataframe(&self, path: &Path) -> Result<DataFrame, OhlcvLoaderError> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "csv" | "tsv" | "txt" => self.load_csv(path),
            "parquet" | "pq" => self.load_parquet(path),
            other => Err(OhlcvLoaderError::UnsupportedFormat(other.to_string())),
        }
    }

    fn load_csv(&self, path: &Path) -> Result<DataFrame, OhlcvLoaderError> {
        let parse_options = CsvParseOptions::default()
            .with_separator(self.config.csv_delimiter)
            .with_try_parse_dates(false); // We handle timestamps ourselves

        let df = CsvReadOptions::default()
            .with_has_header(self.config.has_header)
            .with_parse_options(parse_options)
            .try_into_reader_with_file_path(Some(path.to_path_buf()))?
            .finish()?;

        Ok(df)
    }

    fn load_parquet(&self, path: &Path) -> Result<DataFrame, OhlcvLoaderError> {
        let file = std::fs::File::open(path)?;
        let df = ParquetReader::new(file).finish()?;
        Ok(df)
    }

    /// Validate that all required columns exist in the DataFrame.
    fn validate_columns(
        &self,
        df: &DataFrame,
        col_map: &OhlcvColumnMap,
    ) -> Result<(), OhlcvLoaderError> {
        let available: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
        let available_str = available.join(", ");

        let required = [
            (&col_map.timestamp, "timestamp"),
            (&col_map.open, "open"),
            (&col_map.high, "high"),
            (&col_map.low, "low"),
            (&col_map.close, "close"),
            (&col_map.volume, "volume"),
        ];

        for (col_name, logical_name) in &required {
            if !available.contains(&col_name.as_str()) {
                return Err(OhlcvLoaderError::MissingColumn(
                    format!("{} (mapped from '{}')", col_name, logical_name),
                    available_str,
                ));
            }
        }

        Ok(())
    }

    /// Convert a Polars DataFrame into a Vec<OhlcvBar>.
    fn dataframe_to_bars(
        &self,
        df: &DataFrame,
        col_map: &OhlcvColumnMap,
    ) -> Result<Vec<OhlcvBar>, OhlcvLoaderError> {
        let height = df.height();
        let mut bars = Vec::with_capacity(height);

        let ts_col = df.column(&col_map.timestamp)?;
        let open_col = df.column(&col_map.open)?;
        let high_col = df.column(&col_map.high)?;
        let low_col = df.column(&col_map.low)?;
        let close_col = df.column(&col_map.close)?;
        let vol_col = df.column(&col_map.volume)?;

        let mut skipped = 0usize;

        for i in 0..height {
            // Parse timestamp
            let timestamp = match self.parse_timestamp_value(&ts_col.get(i)?, i) {
                Ok(ts) => ts,
                Err(e) => {
                    if self.config.drop_invalid {
                        skipped += 1;
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            };

            // Parse OHLCV values
            let open = match self.parse_f64(&open_col.get(i)?) {
                Some(v) if v.is_finite() => v,
                _ => {
                    if self.config.drop_invalid {
                        skipped += 1;
                        continue;
                    } else {
                        return Err(OhlcvLoaderError::InvalidPrice {
                            row: i,
                            column: col_map.open.clone(),
                        });
                    }
                }
            };

            let high = match self.parse_f64(&high_col.get(i)?) {
                Some(v) if v.is_finite() => v,
                _ => {
                    if self.config.drop_invalid {
                        skipped += 1;
                        continue;
                    } else {
                        return Err(OhlcvLoaderError::InvalidPrice {
                            row: i,
                            column: col_map.high.clone(),
                        });
                    }
                }
            };

            let low = match self.parse_f64(&low_col.get(i)?) {
                Some(v) if v.is_finite() => v,
                _ => {
                    if self.config.drop_invalid {
                        skipped += 1;
                        continue;
                    } else {
                        return Err(OhlcvLoaderError::InvalidPrice {
                            row: i,
                            column: col_map.low.clone(),
                        });
                    }
                }
            };

            let close = match self.parse_f64(&close_col.get(i)?) {
                Some(v) if v.is_finite() => v,
                _ => {
                    if self.config.drop_invalid {
                        skipped += 1;
                        continue;
                    } else {
                        return Err(OhlcvLoaderError::InvalidPrice {
                            row: i,
                            column: col_map.close.clone(),
                        });
                    }
                }
            };

            let volume = self.parse_f64(&vol_col.get(i)?).unwrap_or(0.0).max(0.0);

            bars.push(OhlcvBar::new(timestamp, open, high, low, close, volume));
        }

        if skipped > 0 {
            warn!("Dropped {} invalid rows out of {} total", skipped, height);
        }

        Ok(bars)
    }

    /// Parse a Polars AnyValue into a DateTime<Utc>.
    fn parse_timestamp_value(
        &self,
        value: &AnyValue,
        row: usize,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        match &self.config.timestamp_format {
            TimestampFormat::UnixSeconds => self.parse_unix_seconds(value, row),
            TimestampFormat::UnixMillis => self.parse_unix_millis(value, row),
            TimestampFormat::UnixMicros => self.parse_unix_micros(value, row),
            TimestampFormat::UnixNanos => self.parse_unix_nanos(value, row),
            TimestampFormat::Iso8601 => self.parse_iso8601(value, row),
            TimestampFormat::Custom(fmt) => self.parse_custom(value, row, fmt),
            TimestampFormat::Auto => self.parse_auto(value, row),
        }
    }

    fn parse_unix_seconds(
        &self,
        value: &AnyValue,
        row: usize,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        let secs = match value {
            AnyValue::Int64(v) => *v,
            AnyValue::Int32(v) => *v as i64,
            AnyValue::UInt64(v) => *v as i64,
            AnyValue::UInt32(v) => *v as i64,
            AnyValue::Float64(v) => *v as i64,
            AnyValue::Float32(v) => *v as i64,
            AnyValue::String(s) => s.parse::<f64>().map(|f| f as i64).map_err(|e| {
                OhlcvLoaderError::TimestampParse {
                    row,
                    details: format!("Cannot parse '{}' as unix seconds: {}", s, e),
                }
            })?,
            AnyValue::StringOwned(s) => {
                s.to_string()
                    .parse::<f64>()
                    .map(|f| f as i64)
                    .map_err(|e| OhlcvLoaderError::TimestampParse {
                        row,
                        details: format!("Cannot parse '{}' as unix seconds: {}", s, e),
                    })?
            }
            _ => {
                return Err(OhlcvLoaderError::TimestampParse {
                    row,
                    details: format!("Unexpected type for unix seconds: {:?}", value),
                });
            }
        };

        DateTime::from_timestamp(secs, 0).ok_or_else(|| OhlcvLoaderError::TimestampParse {
            row,
            details: format!("Invalid unix seconds: {}", secs),
        })
    }

    fn parse_unix_millis(
        &self,
        value: &AnyValue,
        row: usize,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        let ms = match value {
            AnyValue::Int64(v) => *v,
            AnyValue::UInt64(v) => *v as i64,
            AnyValue::Float64(v) => *v as i64,
            AnyValue::String(s) => {
                s.parse::<i64>()
                    .map_err(|e| OhlcvLoaderError::TimestampParse {
                        row,
                        details: format!("Cannot parse '{}' as unix millis: {}", s, e),
                    })?
            }
            AnyValue::StringOwned(s) => {
                s.to_string()
                    .parse::<i64>()
                    .map_err(|e| OhlcvLoaderError::TimestampParse {
                        row,
                        details: format!("Cannot parse '{}' as unix millis: {}", s, e),
                    })?
            }
            _ => {
                return Err(OhlcvLoaderError::TimestampParse {
                    row,
                    details: format!("Unexpected type for unix millis: {:?}", value),
                });
            }
        };

        DateTime::from_timestamp_millis(ms).ok_or_else(|| OhlcvLoaderError::TimestampParse {
            row,
            details: format!("Invalid unix millis: {}", ms),
        })
    }

    fn parse_unix_micros(
        &self,
        value: &AnyValue,
        row: usize,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        let us = match value {
            AnyValue::Int64(v) => *v,
            AnyValue::UInt64(v) => *v as i64,
            AnyValue::Float64(v) => *v as i64,
            AnyValue::String(s) => {
                s.parse::<i64>()
                    .map_err(|e| OhlcvLoaderError::TimestampParse {
                        row,
                        details: format!("Cannot parse '{}' as unix micros: {}", s, e),
                    })?
            }
            _ => {
                return Err(OhlcvLoaderError::TimestampParse {
                    row,
                    details: format!("Unexpected type for unix micros: {:?}", value),
                });
            }
        };

        DateTime::from_timestamp_micros(us).ok_or_else(|| OhlcvLoaderError::TimestampParse {
            row,
            details: format!("Invalid unix micros: {}", us),
        })
    }

    fn parse_unix_nanos(
        &self,
        value: &AnyValue,
        row: usize,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        let ns = match value {
            AnyValue::Int64(v) => *v,
            AnyValue::UInt64(v) => *v as i64,
            AnyValue::Float64(v) => *v as i64,
            AnyValue::String(s) => {
                s.parse::<i64>()
                    .map_err(|e| OhlcvLoaderError::TimestampParse {
                        row,
                        details: format!("Cannot parse '{}' as unix nanos: {}", s, e),
                    })?
            }
            _ => {
                return Err(OhlcvLoaderError::TimestampParse {
                    row,
                    details: format!("Unexpected type for unix nanos: {:?}", value),
                });
            }
        };

        let secs = ns / 1_000_000_000;
        let subsec_nanos = (ns % 1_000_000_000) as u32;
        DateTime::from_timestamp(secs, subsec_nanos).ok_or_else(|| {
            OhlcvLoaderError::TimestampParse {
                row,
                details: format!("Invalid unix nanos: {}", ns),
            }
        })
    }

    fn parse_iso8601(
        &self,
        value: &AnyValue,
        row: usize,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        let s = self.any_value_to_string(value);
        self.parse_datetime_string(&s, row)
    }

    fn parse_custom(
        &self,
        value: &AnyValue,
        row: usize,
        fmt: &str,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        let s = self.any_value_to_string(value);
        NaiveDateTime::parse_from_str(&s, fmt)
            .map(|ndt| ndt.and_utc())
            .map_err(|e| OhlcvLoaderError::TimestampParse {
                row,
                details: format!("Cannot parse '{}' with format '{}': {}", s, fmt, e),
            })
    }

    fn parse_auto(&self, value: &AnyValue, row: usize) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        // Handle Polars native datetime types first
        match value {
            AnyValue::Datetime(dt, tu, _) => {
                let ms = match tu {
                    TimeUnit::Nanoseconds => dt / 1_000_000,
                    TimeUnit::Microseconds => dt / 1_000,
                    TimeUnit::Milliseconds => *dt,
                };
                return DateTime::from_timestamp_millis(ms).ok_or_else(|| {
                    OhlcvLoaderError::TimestampParse {
                        row,
                        details: format!("Invalid Polars datetime value: {}", dt),
                    }
                });
            }
            AnyValue::Date(days) => {
                let secs = *days as i64 * 86400;
                return DateTime::from_timestamp(secs, 0).ok_or_else(|| {
                    OhlcvLoaderError::TimestampParse {
                        row,
                        details: format!("Invalid Polars date value: {} days", days),
                    }
                });
            }
            _ => {}
        }

        // Try numeric interpretation
        match value {
            AnyValue::Int64(v) => return self.auto_detect_numeric(*v, row),
            AnyValue::UInt64(v) => return self.auto_detect_numeric(*v as i64, row),
            AnyValue::Int32(v) => return self.auto_detect_numeric(*v as i64, row),
            AnyValue::UInt32(v) => return self.auto_detect_numeric(*v as i64, row),
            AnyValue::Float64(v) => return self.auto_detect_numeric(*v as i64, row),
            AnyValue::Float32(v) => return self.auto_detect_numeric(*v as i64, row),
            _ => {}
        }

        // Try string interpretation
        let s = self.any_value_to_string(value);
        if s.is_empty() {
            return Err(OhlcvLoaderError::TimestampParse {
                row,
                details: "Empty timestamp value".to_string(),
            });
        }

        // Try parsing as a number
        if let Ok(num) = s.parse::<f64>() {
            return self.auto_detect_numeric(num as i64, row);
        }

        // Try parsing as a datetime string
        self.parse_datetime_string(&s, row)
    }

    /// Auto-detect whether a numeric timestamp is seconds, millis, micros, or nanos.
    fn auto_detect_numeric(
        &self,
        value: i64,
        row: usize,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        // Heuristic based on magnitude:
        //   seconds:      ~1_700_000_000       (10 digits, year ~2023)
        //   millis:       ~1_700_000_000_000   (13 digits)
        //   micros:       ~1_700_000_000_000_000 (16 digits)
        //   nanos:        ~1_700_000_000_000_000_000 (19 digits)
        if value > 1_000_000_000_000_000_000 {
            // Nanoseconds
            let secs = value / 1_000_000_000;
            let nsec = (value % 1_000_000_000) as u32;
            DateTime::from_timestamp(secs, nsec)
        } else if value > 1_000_000_000_000_000 {
            // Microseconds
            DateTime::from_timestamp_micros(value)
        } else if value > 1_000_000_000_000 {
            // Milliseconds
            DateTime::from_timestamp_millis(value)
        } else if value > 100_000_000 {
            // Seconds
            DateTime::from_timestamp(value, 0)
        } else {
            None
        }
        .ok_or_else(|| OhlcvLoaderError::TimestampParse {
            row,
            details: format!("Cannot auto-detect timestamp unit for value: {}", value),
        })
    }

    /// Parse a string as a datetime, trying multiple formats.
    fn parse_datetime_string(
        &self,
        s: &str,
        row: usize,
    ) -> Result<DateTime<Utc>, OhlcvLoaderError> {
        // Try RFC 3339 / ISO 8601 with timezone
        if let Ok(dt) = s.parse::<DateTime<Utc>>() {
            return Ok(dt);
        }

        // Try common formats without timezone (assume UTC)
        let formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ];

        for fmt in &formats {
            if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
                return Ok(ndt.and_utc());
            }
        }

        // Try date-only formats
        if let Ok(nd) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            let ndt = nd.and_hms_opt(0, 0, 0).unwrap();
            return Ok(ndt.and_utc());
        }

        Err(OhlcvLoaderError::TimestampParse {
            row,
            details: format!("Cannot parse '{}' as any known datetime format", s),
        })
    }

    /// Extract a string representation from an AnyValue.
    fn any_value_to_string(&self, value: &AnyValue) -> String {
        match value {
            AnyValue::String(s) => s.to_string(),
            AnyValue::StringOwned(s) => s.to_string(),
            AnyValue::Int64(v) => v.to_string(),
            AnyValue::Int32(v) => v.to_string(),
            AnyValue::UInt64(v) => v.to_string(),
            AnyValue::UInt32(v) => v.to_string(),
            AnyValue::Float64(v) => v.to_string(),
            AnyValue::Float32(v) => v.to_string(),
            _ => format!("{:?}", value),
        }
    }

    /// Parse an AnyValue as f64.
    fn parse_f64(&self, value: &AnyValue) -> Option<f64> {
        match value {
            AnyValue::Float64(v) => Some(*v),
            AnyValue::Float32(v) => Some(*v as f64),
            AnyValue::Int64(v) => Some(*v as f64),
            AnyValue::Int32(v) => Some(*v as f64),
            AnyValue::UInt64(v) => Some(*v as f64),
            AnyValue::UInt32(v) => Some(*v as f64),
            AnyValue::String(s) => s.parse::<f64>().ok(),
            AnyValue::StringOwned(s) => s.to_string().parse::<f64>().ok(),
            AnyValue::Null => None,
            _ => None,
        }
    }

    /// Apply start/end time filter to sorted bars.
    fn apply_time_filter(&self, bars: Vec<OhlcvBar>) -> Result<Vec<OhlcvBar>, OhlcvLoaderError> {
        let start = self.config.start_time;
        let end = self.config.end_time;

        if start.is_none() && end.is_none() {
            return Ok(bars);
        }

        let filtered: Vec<OhlcvBar> = bars
            .into_iter()
            .filter(|b| {
                if let Some(s) = start
                    && b.timestamp < s
                {
                    return false;
                }
                if let Some(e) = end
                    && b.timestamp > e
                {
                    return false;
                }
                true
            })
            .collect();

        if filtered.is_empty() {
            let start_str = start
                .map(|s| s.to_rfc3339())
                .unwrap_or_else(|| "unbounded".to_string());
            let end_str = end
                .map(|e| e.to_rfc3339())
                .unwrap_or_else(|| "unbounded".to_string());
            return Err(OhlcvLoaderError::NoDataInRange {
                start: start_str,
                end: end_str,
            });
        }

        Ok(filtered)
    }

    /// Get a summary of the loaded data without converting to OhlcvBar.
    ///
    /// Useful for quick inspection of a file before running a full backtest.
    pub fn inspect(&self) -> Result<DataSummary, OhlcvLoaderError> {
        let path = &self.config.path;

        if !path.exists() {
            return Err(OhlcvLoaderError::FileNotFound(path.display().to_string()));
        }

        let df = self.load_dataframe(path)?;
        let col_names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let row_count = df.height();

        let col_map = if self.config.auto_detect_columns && self.config.has_header {
            let refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
            OhlcvColumnMap::auto_detect(&refs)
        } else {
            self.config.column_map.clone()
        };

        // Try to get price range from close column
        let (min_price, max_price) = if let Ok(close_col) = df.column(&col_map.close) {
            let min = self.series_min_f64(close_col);
            let max = self.series_max_f64(close_col);
            (min, max)
        } else {
            (None, None)
        };

        Ok(DataSummary {
            path: path.display().to_string(),
            format: path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("unknown")
                .to_string(),
            columns: col_names,
            row_count,
            detected_column_map: col_map,
            min_price,
            max_price,
        })
    }

    /// Get the minimum value in a Series as f64.
    fn series_min_f64(&self, series: &Column) -> Option<f64> {
        let height = series.len();
        let mut min = f64::INFINITY;
        for i in 0..height {
            if let Ok(val) = series.get(i)
                && let Some(v) = self.parse_f64(&val)
                && v < min
            {
                min = v;
            }
        }
        if min.is_finite() { Some(min) } else { None }
    }

    /// Get the maximum value in a Series as f64.
    fn series_max_f64(&self, series: &Column) -> Option<f64> {
        let height = series.len();
        let mut max = f64::NEG_INFINITY;
        for i in 0..height {
            if let Ok(val) = series.get(i)
                && let Some(v) = self.parse_f64(&val)
                && v > max
            {
                max = v;
            }
        }
        if max.is_finite() { Some(max) } else { None }
    }
}

// ============================================================================
// Data Summary
// ============================================================================

/// Quick summary of a data file without full conversion.
#[derive(Debug, Clone)]
pub struct DataSummary {
    /// File path.
    pub path: String,
    /// File format (csv, parquet, etc.).
    pub format: String,
    /// Column names found in the file.
    pub columns: Vec<String>,
    /// Total row count.
    pub row_count: usize,
    /// Detected or configured column mapping.
    pub detected_column_map: OhlcvColumnMap,
    /// Minimum close price (if parseable).
    pub min_price: Option<f64>,
    /// Maximum close price (if parseable).
    pub max_price: Option<f64>,
}

impl std::fmt::Display for DataSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Data Summary: {}", self.path)?;
        writeln!(f, "  Format: {}", self.format)?;
        writeln!(f, "  Rows: {}", self.row_count)?;
        writeln!(f, "  Columns: {}", self.columns.join(", "))?;
        writeln!(
            f,
            "  Mapped: ts={}, o={}, h={}, l={}, c={}, v={}",
            self.detected_column_map.timestamp,
            self.detected_column_map.open,
            self.detected_column_map.high,
            self.detected_column_map.low,
            self.detected_column_map.close,
            self.detected_column_map.volume,
        )?;
        if let (Some(min), Some(max)) = (self.min_price, self.max_price) {
            writeln!(f, "  Price range: {:.2} – {:.2}", min, max)?;
        }
        Ok(())
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Load OHLCV bars from a CSV file with default settings.
///
/// This is a shortcut for the most common use case.
pub fn load_ohlcv_csv<P: Into<PathBuf>>(path: P) -> Result<Vec<OhlcvBar>, OhlcvLoaderError> {
    let config = OhlcvLoaderConfig::new(path);
    OhlcvLoader::new(config).load()
}

/// Load OHLCV bars from a Parquet file with default settings.
pub fn load_ohlcv_parquet<P: Into<PathBuf>>(path: P) -> Result<Vec<OhlcvBar>, OhlcvLoaderError> {
    let config = OhlcvLoaderConfig::new(path);
    OhlcvLoader::new(config).load()
}

/// Load OHLCV bars from a Kraken CSV export.
pub fn load_kraken_csv<P: Into<PathBuf>>(
    path: P,
    symbol: &str,
) -> Result<Vec<OhlcvBar>, OhlcvLoaderError> {
    let config = OhlcvLoaderConfig::new(path).with_symbol(symbol).kraken();
    OhlcvLoader::new(config).load()
}

/// Load OHLCV bars from a Binance kline CSV export.
pub fn load_binance_csv<P: Into<PathBuf>>(
    path: P,
    symbol: &str,
) -> Result<Vec<OhlcvBar>, OhlcvLoaderError> {
    let config = OhlcvLoaderConfig::new(path).with_symbol(symbol).binance();
    OhlcvLoader::new(config).load()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    /// Helper: create a CSV file with OHLCV data.
    fn create_test_csv(path: &Path, header: &str, rows: &[&str]) {
        let mut file = std::fs::File::create(path).unwrap();
        writeln!(file, "{}", header).unwrap();
        for row in rows {
            writeln!(file, "{}", row).unwrap();
        }
    }

    #[test]
    fn test_load_basic_csv() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "1700000000,50000.0,50100.0,49900.0,50050.0,100.5",
                "1700000900,50050.0,50200.0,49950.0,50150.0,120.3",
                "1700001800,50150.0,50300.0,50000.0,50250.0,95.7",
            ],
        );

        let config =
            OhlcvLoaderConfig::new(&file).with_timestamp_format(TimestampFormat::UnixSeconds);
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 3);
        assert!((bars[0].open - 50000.0).abs() < 0.01);
        assert!((bars[0].close - 50050.0).abs() < 0.01);
        assert!((bars[2].close - 50250.0).abs() < 0.01);
        assert!(bars[0].timestamp < bars[1].timestamp);
    }

    #[test]
    fn test_load_csv_auto_timestamp() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "1700000000,100.0,110.0,90.0,105.0,1000",
                "1700000060,105.0,115.0,95.0,110.0,1200",
            ],
        );

        let bars = OhlcvLoader::new(OhlcvLoaderConfig::new(&file))
            .load()
            .unwrap();
        assert_eq!(bars.len(), 2);
        // Auto should detect unix seconds
        assert_eq!(bars[0].timestamp.timestamp(), 1700000000);
    }

    #[test]
    fn test_load_csv_millis_timestamp() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "open_time,open,high,low,close,volume",
            &[
                "1700000000000,100.0,110.0,90.0,105.0,1000",
                "1700000060000,105.0,115.0,95.0,110.0,1200",
            ],
        );

        let config = OhlcvLoaderConfig::new(&file)
            .with_column_map(OhlcvColumnMap::binance())
            .with_timestamp_format(TimestampFormat::UnixMillis);
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].timestamp.timestamp(), 1700000000);
    }

    #[test]
    fn test_auto_detect_columns() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        // Use slightly unusual but recognizable column names
        create_test_csv(
            &file,
            "time,Open,High,Low,Close,Vol",
            &[
                "1700000000,100.0,110.0,90.0,105.0,1000",
                "1700000060,105.0,115.0,95.0,110.0,1200",
            ],
        );

        let config = OhlcvLoaderConfig::new(&file);
        let bars = OhlcvLoader::new(config).load().unwrap();
        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn test_iso8601_timestamps() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "2024-01-15T12:00:00Z,100.0,110.0,90.0,105.0,1000",
                "2024-01-15T12:15:00Z,105.0,115.0,95.0,110.0,1200",
            ],
        );

        let config = OhlcvLoaderConfig::new(&file).with_timestamp_format(TimestampFormat::Iso8601);
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 2);
        assert_eq!(
            bars[0].timestamp.format("%Y-%m-%d").to_string(),
            "2024-01-15"
        );
    }

    #[test]
    fn test_time_range_filter() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "1700000000,100.0,110.0,90.0,105.0,1000",
                "1700001000,105.0,115.0,95.0,110.0,1200",
                "1700002000,110.0,120.0,100.0,115.0,1100",
                "1700003000,115.0,125.0,105.0,120.0,1300",
            ],
        );

        let start = DateTime::from_timestamp(1700000500, 0).unwrap();
        let end = DateTime::from_timestamp(1700002500, 0).unwrap();

        let config = OhlcvLoaderConfig::new(&file)
            .with_timestamp_format(TimestampFormat::UnixSeconds)
            .with_time_range(start, end);
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].timestamp.timestamp(), 1700001000);
        assert_eq!(bars[1].timestamp.timestamp(), 1700002000);
    }

    #[test]
    fn test_skip_and_limit() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        let rows: Vec<String> = (0..10)
            .map(|i| {
                format!(
                    "{},{:.1},{:.1},{:.1},{:.1},100",
                    1700000000 + i * 60,
                    100.0 + i as f64,
                    110.0 + i as f64,
                    90.0 + i as f64,
                    105.0 + i as f64,
                )
            })
            .collect();
        let row_refs: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();

        create_test_csv(&file, "timestamp,open,high,low,close,volume", &row_refs);

        let config = OhlcvLoaderConfig::new(&file)
            .with_timestamp_format(TimestampFormat::UnixSeconds)
            .with_skip_bars(3)
            .with_max_bars(4);
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 4);
        // After skipping 3, first bar should be the 4th (index 3)
        assert_eq!(bars[0].timestamp.timestamp(), 1700000000 + 3 * 60);
    }

    #[test]
    fn test_drop_invalid_rows() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "1700000000,100.0,110.0,90.0,105.0,1000",
                "bad_timestamp,105.0,115.0,95.0,110.0,1200",
                "1700002000,110.0,120.0,100.0,115.0,1100",
            ],
        );

        let config =
            OhlcvLoaderConfig::new(&file).with_timestamp_format(TimestampFormat::UnixSeconds);
        let bars = OhlcvLoader::new(config).load().unwrap();

        // The bad row should be dropped
        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn test_file_not_found() {
        let config = OhlcvLoaderConfig::new("/nonexistent/file.csv");
        let result = OhlcvLoader::new(config).load();
        assert!(result.is_err());
        match result.unwrap_err() {
            OhlcvLoaderError::FileNotFound(_) => {}
            e => panic!("Expected FileNotFound, got: {:?}", e),
        }
    }

    #[test]
    fn test_unsupported_format() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.xyz");
        std::fs::write(&file, "data").unwrap();

        let config = OhlcvLoaderConfig::new(&file);
        let result = OhlcvLoader::new(config).load();
        assert!(result.is_err());
        match result.unwrap_err() {
            OhlcvLoaderError::UnsupportedFormat(_) => {}
            e => panic!("Expected UnsupportedFormat, got: {:?}", e),
        }
    }

    #[test]
    fn test_empty_csv() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");
        std::fs::write(&file, "timestamp,open,high,low,close,volume\n").unwrap();

        let config = OhlcvLoaderConfig::new(&file);
        let result = OhlcvLoader::new(config).load();
        assert!(result.is_err());
    }

    #[test]
    fn test_inspect() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "1700000000,100.0,110.0,90.0,105.0,1000",
                "1700000060,105.0,115.0,95.0,110.0,1200",
                "1700000120,110.0,120.0,100.0,115.0,1100",
            ],
        );

        let config = OhlcvLoaderConfig::new(&file);
        let summary = OhlcvLoader::new(config).inspect().unwrap();

        assert_eq!(summary.row_count, 3);
        assert_eq!(summary.format, "csv");
        assert!(summary.columns.contains(&"close".to_string()));
        assert!((summary.min_price.unwrap() - 105.0).abs() < 0.01);
        assert!((summary.max_price.unwrap() - 115.0).abs() < 0.01);

        // Display should not panic
        let display = format!("{}", summary);
        assert!(display.contains("Rows: 3"));
    }

    #[test]
    fn test_custom_delimiter() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.tsv");

        let mut f = std::fs::File::create(&file).unwrap();
        writeln!(f, "timestamp\topen\thigh\tlow\tclose\tvolume").unwrap();
        writeln!(f, "1700000000\t100.0\t110.0\t90.0\t105.0\t1000").unwrap();
        writeln!(f, "1700000060\t105.0\t115.0\t95.0\t110.0\t1200").unwrap();

        let config = OhlcvLoaderConfig::new(&file)
            .with_delimiter(b'\t')
            .with_timestamp_format(TimestampFormat::UnixSeconds);
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn test_kraken_preset() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("kraken.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,vwap,volume,count",
            &[
                "1700000000,50000.0,50100.0,49900.0,50050.0,50025.0,15.5,120",
                "1700000900,50050.0,50200.0,49950.0,50150.0,50100.0,18.2,135",
            ],
        );

        let config = OhlcvLoaderConfig::new(&file).kraken().with_symbol("XBTUSD");
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 2);
        assert!((bars[0].open - 50000.0).abs() < 0.01);
    }

    #[test]
    fn test_sorted_output() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        // Data intentionally out of order
        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "1700002000,110.0,120.0,100.0,115.0,1100",
                "1700000000,100.0,110.0,90.0,105.0,1000",
                "1700001000,105.0,115.0,95.0,110.0,1200",
            ],
        );

        let config =
            OhlcvLoaderConfig::new(&file).with_timestamp_format(TimestampFormat::UnixSeconds);
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 3);
        assert!(bars[0].timestamp < bars[1].timestamp);
        assert!(bars[1].timestamp < bars[2].timestamp);
    }

    #[test]
    fn test_convenience_load_ohlcv_csv() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "1700000000,100.0,110.0,90.0,105.0,1000",
                "1700000060,105.0,115.0,95.0,110.0,1200",
            ],
        );

        let bars = load_ohlcv_csv(&file).unwrap();
        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn test_column_map_presets() {
        // Just verify they construct without panicking
        let _default = OhlcvColumnMap::default();
        let _kraken = OhlcvColumnMap::kraken();
        let _binance = OhlcvColumnMap::binance();
        let _tv = OhlcvColumnMap::tradingview();
        let _pos = OhlcvColumnMap::positional();
        let _custom = OhlcvColumnMap::custom("ts", "o", "h", "l", "c", "v");
    }

    #[test]
    fn test_auto_detect_column_map() {
        let map = OhlcvColumnMap::auto_detect(&["time", "Open", "High", "Low", "Close", "Vol"]);
        assert_eq!(map.timestamp, "time");
        assert_eq!(map.open, "Open");
        assert_eq!(map.high, "High");
        assert_eq!(map.low, "Low");
        assert_eq!(map.close, "Close");
        assert_eq!(map.volume, "Vol");
    }

    #[test]
    fn test_parquet_roundtrip() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.parquet");

        // Create a parquet file with OHLCV data
        let timestamps: Vec<i64> = (0..5).map(|i| 1700000000i64 + i * 900).collect();
        let opens: Vec<f64> = vec![100.0, 105.0, 110.0, 108.0, 112.0];
        let highs: Vec<f64> = vec![110.0, 115.0, 120.0, 118.0, 122.0];
        let lows: Vec<f64> = vec![90.0, 95.0, 100.0, 98.0, 102.0];
        let closes: Vec<f64> = vec![105.0, 110.0, 115.0, 112.0, 118.0];
        let volumes: Vec<f64> = vec![1000.0, 1200.0, 1100.0, 900.0, 1300.0];

        let df = df! {
            "timestamp" => timestamps,
            "open" => opens,
            "high" => highs,
            "low" => lows,
            "close" => closes,
            "volume" => volumes,
        }
        .unwrap();

        let mut out = std::fs::File::create(&file).unwrap();
        ParquetWriter::new(&mut out)
            .finish(&mut df.clone())
            .unwrap();

        // Now load it back
        let config =
            OhlcvLoaderConfig::new(&file).with_timestamp_format(TimestampFormat::UnixSeconds);
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 5);
        assert!((bars[0].open - 100.0).abs() < 0.01);
        assert!((bars[4].close - 118.0).abs() < 0.01);
    }

    #[test]
    fn test_datetime_string_timestamps() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        create_test_csv(
            &file,
            "timestamp,open,high,low,close,volume",
            &[
                "2024-01-15 12:00:00,100.0,110.0,90.0,105.0,1000",
                "2024-01-15 12:15:00,105.0,115.0,95.0,110.0,1200",
            ],
        );

        let config = OhlcvLoaderConfig::new(&file); // Auto-detect
        let bars = OhlcvLoader::new(config).load().unwrap();

        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn test_missing_volume_defaults_to_zero() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.csv");

        // Create CSV where volume column has some nulls (represented as empty)
        let mut f = std::fs::File::create(&file).unwrap();
        writeln!(f, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(f, "1700000000,100.0,110.0,90.0,105.0,1000").unwrap();
        writeln!(f, "1700000060,105.0,115.0,95.0,110.0,").unwrap();

        let config =
            OhlcvLoaderConfig::new(&file).with_timestamp_format(TimestampFormat::UnixSeconds);
        let bars = OhlcvLoader::new(config).load().unwrap();

        // First bar should have volume, second should default to 0
        assert_eq!(bars.len(), 2);
        assert!((bars[0].volume - 1000.0).abs() < 0.01);
        assert!((bars[1].volume - 0.0).abs() < 0.01);
    }
}
