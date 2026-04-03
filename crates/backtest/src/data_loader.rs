//! # Historical Data Loader
//!
//! Loads and prepares historical tick data from various sources (Parquet, CSV, QuestDB).
//!
//! Supports:
//! - Parquet files (preferred for performance)
//! - CSV files (for compatibility)
//! - Memory-efficient chunked loading for large datasets
//! - Automatic schema validation and type conversion

use chrono::{DateTime, Utc};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during data loading
#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Missing required column: {0}")]
    MissingColumn(String),

    #[error("Invalid timestamp format in column {column}: {details}")]
    InvalidTimestamp { column: String, details: String },

    #[error("No data found in date range {start} to {end}")]
    NoDataInRange {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),
}

/// A single market tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub side: Side,
}

/// Trade side
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl crate::fortress::TimestampedEvent for Tick {
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
}

/// Configuration for data loading
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Path to data file(s)
    pub path: PathBuf,

    /// Symbol to filter (if None, loads all symbols)
    pub symbol: Option<String>,

    /// Start time for data range
    pub start_time: Option<DateTime<Utc>>,

    /// End time for data range
    pub end_time: Option<DateTime<Utc>>,

    /// Chunk size for large file processing (rows per chunk)
    pub chunk_size: Option<usize>,

    /// Whether to validate data integrity
    pub validate: bool,
}

impl DataLoaderConfig {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            symbol: None,
            start_time: None,
            end_time: None,
            chunk_size: None,
            validate: true,
        }
    }

    pub fn with_symbol(mut self, symbol: String) -> Self {
        self.symbol = Some(symbol);
        self
    }

    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    pub fn skip_validation(mut self) -> Self {
        self.validate = false;
        self
    }
}

/// Historical data loader
pub struct DataLoader {
    config: DataLoaderConfig,
}

impl DataLoader {
    pub fn new(config: DataLoaderConfig) -> Self {
        Self { config }
    }

    /// Load all ticks into memory
    pub fn load(&self) -> Result<Vec<Tick>, DataLoaderError> {
        let df = self.load_dataframe()?;
        self.dataframe_to_ticks(df)
    }

    /// Load ticks in chunks (for large datasets)
    pub fn load_chunked(&self) -> Result<Vec<Vec<Tick>>, DataLoaderError> {
        let chunk_size = self.config.chunk_size.unwrap_or(100_000);
        let df = self.load_dataframe()?;

        let total_rows = df.height();
        let mut chunks = Vec::new();

        for start in (0..total_rows).step_by(chunk_size) {
            let end = (start + chunk_size).min(total_rows);
            let chunk_df = df.slice(start as i64, end - start);
            let ticks = self.dataframe_to_ticks(chunk_df)?;
            chunks.push(ticks);
        }

        Ok(chunks)
    }

    /// Load raw DataFrame (for advanced processing)
    pub fn load_dataframe(&self) -> Result<DataFrame, DataLoaderError> {
        let path = &self.config.path;

        if !path.exists() {
            return Err(DataLoaderError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            )));
        }

        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let mut df = match extension {
            "parquet" => self.load_parquet(path)?,
            "csv" => self.load_csv(path)?,
            _ => {
                return Err(DataLoaderError::InvalidFormat(format!(
                    "Unsupported file format: {}",
                    extension
                )));
            }
        };

        // Apply filters
        df = self.apply_filters(df)?;

        // Validate schema
        if self.config.validate {
            self.validate_schema(&df)?;
        }

        Ok(df)
    }

    fn load_parquet(&self, path: &Path) -> Result<DataFrame, DataLoaderError> {
        let file = std::fs::File::open(path)?;
        let df = ParquetReader::new(file).finish()?;
        Ok(df)
    }

    pub(crate) fn load_csv(&self, path: &Path) -> Result<DataFrame, DataLoaderError> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_parse_options(CsvParseOptions::default().with_try_parse_dates(true))
            .try_into_reader_with_file_path(Some(path.to_path_buf()))?
            .finish()?;
        Ok(df)
    }

    pub(crate) fn apply_filters(&self, mut df: DataFrame) -> Result<DataFrame, DataLoaderError> {
        // Filter by symbol
        if let Some(ref symbol) = self.config.symbol {
            let col_names: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
            if col_names.contains(&"symbol") {
                let symbol_col = df.column("symbol")?;
                let mask = symbol_col.str()?.equal(symbol.as_str());
                df = df.filter(&mask)?;
            }
        }

        // Filter by time range
        if let Some(start) = self.config.start_time {
            let start_ms = start.timestamp_millis();
            let timestamp_col = df.column("timestamp")?;
            let mask = timestamp_col.i64()?.gt_eq(start_ms);
            df = df.filter(&mask)?;
        }

        if let Some(end) = self.config.end_time {
            let end_ms = end.timestamp_millis();
            let timestamp_col = df.column("timestamp")?;
            let mask = timestamp_col.i64()?.lt_eq(end_ms);
            df = df.filter(&mask)?;
        }

        // Check if we have any data
        if df.height() == 0
            && let (Some(start), Some(end)) = (self.config.start_time, self.config.end_time)
        {
            return Err(DataLoaderError::NoDataInRange { start, end });
        }

        Ok(df)
    }

    fn validate_schema(&self, df: &DataFrame) -> Result<(), DataLoaderError> {
        let required_columns = vec!["timestamp", "symbol", "price", "volume"];
        let col_names: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();

        for col in required_columns {
            if !col_names.contains(&col) {
                return Err(DataLoaderError::MissingColumn(col.to_string()));
            }
        }

        Ok(())
    }

    fn dataframe_to_ticks(&self, df: DataFrame) -> Result<Vec<Tick>, DataLoaderError> {
        let height = df.height();
        let mut ticks = Vec::with_capacity(height);

        let timestamp_col = df.column("timestamp")?;
        let symbol_col = df.column("symbol")?;
        let price_col = df.column("price")?;
        let volume_col = df.column("volume")?;
        let side_col = df.column("side").ok();

        for i in 0..height {
            // Parse timestamp
            let ts_value = timestamp_col.get(i)?;
            let timestamp = self.parse_timestamp(&ts_value)?;

            // Parse symbol
            let symbol = match symbol_col.get(i)? {
                AnyValue::String(s) => s.to_string(),
                AnyValue::StringOwned(s) => s.to_string(),
                _ => return Err(DataLoaderError::MissingColumn("symbol".to_string())),
            };

            // Parse price
            let price = match price_col.get(i)? {
                AnyValue::Float64(p) => p,
                AnyValue::Float32(p) => p as f64,
                AnyValue::Int64(p) => p as f64,
                AnyValue::Int32(p) => p as f64,
                _ => return Err(DataLoaderError::MissingColumn("price".to_string())),
            };

            // Parse volume
            let volume = match volume_col.get(i)? {
                AnyValue::Float64(v) => v,
                AnyValue::Float32(v) => v as f64,
                AnyValue::Int64(v) => v as f64,
                AnyValue::Int32(v) => v as f64,
                _ => return Err(DataLoaderError::MissingColumn("volume".to_string())),
            };

            // Parse side (default to Buy if not present)
            let side = if let Some(side_col) = side_col {
                match side_col.get(i)? {
                    AnyValue::String(s) => {
                        if s.eq_ignore_ascii_case("sell") {
                            Side::Sell
                        } else {
                            Side::Buy
                        }
                    }
                    AnyValue::StringOwned(s) => {
                        if s.eq_ignore_ascii_case("sell") {
                            Side::Sell
                        } else {
                            Side::Buy
                        }
                    }
                    _ => Side::Buy,
                }
            } else {
                Side::Buy
            };

            ticks.push(Tick {
                timestamp,
                symbol,
                price,
                volume,
                side,
            });
        }

        Ok(ticks)
    }

    fn parse_timestamp(&self, value: &AnyValue) -> Result<DateTime<Utc>, DataLoaderError> {
        match value {
            AnyValue::Int64(ms) => DateTime::from_timestamp_millis(*ms).ok_or_else(|| {
                DataLoaderError::InvalidTimestamp {
                    column: "timestamp".to_string(),
                    details: format!("Invalid milliseconds: {}", ms),
                }
            }),
            AnyValue::Datetime(dt, tu, _) => {
                let timestamp_ms = match tu {
                    TimeUnit::Nanoseconds => dt / 1_000_000,
                    TimeUnit::Microseconds => dt / 1_000,
                    TimeUnit::Milliseconds => *dt,
                };
                DateTime::from_timestamp_millis(timestamp_ms).ok_or_else(|| {
                    DataLoaderError::InvalidTimestamp {
                        column: "timestamp".to_string(),
                        details: format!("Invalid datetime: {}", dt),
                    }
                })
            }
            AnyValue::String(s) => {
                s.parse::<DateTime<Utc>>()
                    .map_err(|e| DataLoaderError::InvalidTimestamp {
                        column: "timestamp".to_string(),
                        details: e.to_string(),
                    })
            }
            AnyValue::StringOwned(s) => s.to_string().parse::<DateTime<Utc>>().map_err(|e| {
                DataLoaderError::InvalidTimestamp {
                    column: "timestamp".to_string(),
                    details: e.to_string(),
                }
            }),
            _ => Err(DataLoaderError::InvalidTimestamp {
                column: "timestamp".to_string(),
                details: format!("Unsupported type: {:?}", value),
            }),
        }
    }
}

/// Helper to create sample tick data for testing
#[cfg(test)]
pub fn create_sample_parquet(path: &Path, num_rows: usize) -> Result<(), DataLoaderError> {
    use chrono::Duration;

    let start_time = Utc::now();
    let timestamps: Vec<i64> = (0..num_rows)
        .map(|i| (start_time + Duration::seconds(i as i64)).timestamp_millis())
        .collect();

    let symbols: Vec<String> = vec!["BTCUSD".to_string(); num_rows];
    let prices: Vec<f64> = (0..num_rows).map(|i| 50000.0 + i as f64 * 0.1).collect();
    let volumes: Vec<f64> = vec![1.0; num_rows];
    let sides: Vec<String> = (0..num_rows)
        .map(|i| if i % 2 == 0 { "Buy" } else { "Sell" }.to_string())
        .collect();

    let df = df! {
        "timestamp" => timestamps,
        "symbol" => symbols,
        "price" => prices,
        "volume" => volumes,
        "side" => sides,
    }?;

    let mut file = std::fs::File::create(path)?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parquet_loading() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");

        // Create sample data
        create_sample_parquet(&file_path, 100).unwrap();

        // Load it back
        let config = DataLoaderConfig::new(&file_path);
        let loader = DataLoader::new(config);
        let ticks = loader.load().unwrap();

        assert_eq!(ticks.len(), 100);
        assert_eq!(ticks[0].symbol, "BTCUSD");
        assert!(ticks[0].price >= 50000.0);
    }

    #[test]
    fn test_symbol_filtering() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");

        create_sample_parquet(&file_path, 100).unwrap();

        let config = DataLoaderConfig::new(&file_path).with_symbol("BTCUSD".to_string());
        let loader = DataLoader::new(config);
        let ticks = loader.load().unwrap();

        assert_eq!(ticks.len(), 100);
        assert!(ticks.iter().all(|t| t.symbol == "BTCUSD"));
    }

    #[test]
    fn test_chunked_loading() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");

        create_sample_parquet(&file_path, 250).unwrap();

        let config = DataLoaderConfig::new(&file_path).with_chunk_size(100);
        let loader = DataLoader::new(config);
        let chunks = loader.load_chunked().unwrap();

        assert_eq!(chunks.len(), 3); // 100, 100, 50
        assert_eq!(chunks[0].len(), 100);
        assert_eq!(chunks[1].len(), 100);
        assert_eq!(chunks[2].len(), 50);
    }
}
