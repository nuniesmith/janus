//! Dataset handling for ML training
//!
//! This module provides utilities for:
//! - Loading market data from various sources (Parquet, QuestDB, CSV)
//! - Creating sliding windows for time series
//! - Batching and shuffling data
//! - Train/validation/test splits
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                  Dataset Pipeline                        │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                          │
//! │  Data Source (Parquet/QuestDB/CSV)                      │
//! │         │                                                │
//! │         ▼                                                │
//! │  MarketDataset (load & parse)                           │
//! │         │                                                │
//! │         ▼                                                │
//! │  WindowedDataset (sliding windows)                      │
//! │         │                                                │
//! │         ▼                                                │
//! │  DataLoader (batching & shuffling)                      │
//! │         │                                                │
//! │         ▼                                                │
//! │  Training Loop                                           │
//! │                                                          │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::dataset::{MarketDataset, WindowConfig, DataLoader};
//! use janus_ml::features::TechnicalIndicators;
//!
//! // Load data from parquet files
//! let dataset = MarketDataset::from_parquet("data/*.parquet")?;
//!
//! // Create sliding windows
//! let config = WindowConfig::new(100, 1); // 100 timesteps, predict 1 ahead
//! let windowed = dataset.into_windowed(config)?;
//!
//! // Create data loader with batching
//! let loader = DataLoader::new(windowed)
//!     .batch_size(32)
//!     .shuffle(true)
//!     .num_workers(4);
//!
//! // Iterate over batches
//! for batch in loader {
//!     let (features, targets) = batch?;
//!     // Train model on batch
//! }
//! ```

use std::path::Path;

use burn_core::tensor::{Shape, Tensor, TensorData, backend::Backend};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{MLError, Result};

/// Market data sample containing features and target
#[derive(Debug, Clone)]
pub struct MarketDataSample {
    /// Timestamp of the sample
    pub timestamp: i64,

    /// Feature vector (already extracted)
    pub features: Vec<f64>,

    /// Target value (e.g., future price, direction)
    pub target: f64,

    /// Additional metadata
    pub metadata: SampleMetadata,
}

/// Metadata for a market data sample
#[derive(Debug, Clone, Default)]
pub struct SampleMetadata {
    /// Symbol (e.g., "BTCUSD")
    pub symbol: String,

    /// Exchange (e.g., "binance")
    pub exchange: String,

    /// Original price at this timestep
    pub price: f64,

    /// Volume at this timestep
    pub volume: f64,
}

/// Configuration for sliding window generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Number of timesteps in each window (lookback)
    pub window_size: usize,

    /// Number of timesteps to predict ahead (forecast horizon)
    pub horizon: usize,

    /// Stride for sliding window (default: 1 for maximum overlap)
    pub stride: usize,

    /// Minimum required data points to create a window
    pub min_samples: usize,
}

impl WindowConfig {
    /// Create a new window configuration
    pub fn new(window_size: usize, horizon: usize) -> Self {
        Self {
            window_size,
            horizon,
            stride: 1,
            min_samples: window_size + horizon,
        }
    }

    /// Set the stride for sliding window
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set minimum samples required
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self::new(100, 1) // 100 timestep lookback, predict 1 step ahead
    }
}

/// Market dataset loaded from various sources
#[derive(Debug, Clone)]
pub struct MarketDataset {
    /// All samples in temporal order
    pub samples: Vec<MarketDataSample>,

    /// Dataset metadata
    pub metadata: DatasetMetadata,
}

/// Metadata about the dataset
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    /// Number of samples
    pub num_samples: usize,

    /// Number of features per sample
    pub num_features: usize,

    /// Start timestamp
    pub start_time: DateTime<Utc>,

    /// End timestamp
    pub end_time: DateTime<Utc>,

    /// Symbols included
    pub symbols: Vec<String>,

    /// Exchanges included
    pub exchanges: Vec<String>,

    /// Source path
    pub source: String,
}

impl MarketDataset {
    /// Create a new empty dataset
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            metadata: DatasetMetadata {
                num_samples: 0,
                num_features: 0,
                start_time: Utc::now(),
                end_time: Utc::now(),
                symbols: Vec::new(),
                exchanges: Vec::new(),
                source: String::new(),
            },
        }
    }

    /// Load dataset from Parquet files
    ///
    /// Supports standard OHLCV Parquet files with columns:
    /// - timestamp (int64 milliseconds or microseconds)
    /// - symbol (utf8, optional)
    /// - exchange (utf8, optional)
    /// - open, high, low, close (float64)
    /// - volume (float64)
    ///
    /// Features extracted per row:
    /// - Price returns (close-to-close)
    /// - Range ratio (high-low)/close
    /// - Upper/lower shadow ratios
    /// - Volume (normalized if possible)
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        use arrow::array::{Array, Float64Array, Int64Array, StringArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();
        tracing::info!("Loading dataset from Parquet: {}", path_str);

        // Open Parquet file
        let file = File::open(path_ref).map_err(|e| {
            MLError::DataLoadingError(format!("Failed to open Parquet file: {}", e))
        })?;

        // Create Parquet reader
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            MLError::DataLoadingError(format!("Failed to create Parquet reader: {}", e))
        })?;

        let reader = builder.build().map_err(|e| {
            MLError::DataLoadingError(format!("Failed to build Parquet reader: {}", e))
        })?;

        let mut samples = Vec::new();
        let mut symbols_set = std::collections::HashSet::new();
        let mut exchanges_set = std::collections::HashSet::new();
        let mut min_ts: Option<DateTime<Utc>> = None;
        let mut max_ts: Option<DateTime<Utc>> = None;
        let mut prev_close: Option<f64> = None;

        // Process each record batch
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| MLError::DataLoadingError(format!("Failed to read batch: {}", e)))?;

            let schema = batch.schema();
            let num_rows = batch.num_rows();

            // Find column indices by name (case-insensitive)
            let find_col = |names: &[&str]| -> Option<usize> {
                for name in names {
                    for (i, field) in schema.fields().iter().enumerate() {
                        if field.name().to_lowercase() == *name {
                            return Some(i);
                        }
                    }
                }
                None
            };

            let ts_col = find_col(&["timestamp", "time", "ts", "date"]).ok_or_else(|| {
                MLError::DataLoadingError("No timestamp column found".to_string())
            })?;
            let open_col = find_col(&["open", "o"])
                .ok_or_else(|| MLError::DataLoadingError("No open column found".to_string()))?;
            let high_col = find_col(&["high", "h"])
                .ok_or_else(|| MLError::DataLoadingError("No high column found".to_string()))?;
            let low_col = find_col(&["low", "l"])
                .ok_or_else(|| MLError::DataLoadingError("No low column found".to_string()))?;
            let close_col = find_col(&["close", "c"])
                .ok_or_else(|| MLError::DataLoadingError("No close column found".to_string()))?;
            let volume_col = find_col(&["volume", "vol", "v"])
                .ok_or_else(|| MLError::DataLoadingError("No volume column found".to_string()))?;
            let symbol_col = find_col(&["symbol", "pair", "ticker"]);
            let exchange_col = find_col(&["exchange", "exch"]);

            // Extract arrays
            let ts_array = batch.column(ts_col);
            let open_array = batch
                .column(open_col)
                .as_any()
                .downcast_ref::<Float64Array>();
            let high_array = batch
                .column(high_col)
                .as_any()
                .downcast_ref::<Float64Array>();
            let low_array = batch
                .column(low_col)
                .as_any()
                .downcast_ref::<Float64Array>();
            let close_array = batch
                .column(close_col)
                .as_any()
                .downcast_ref::<Float64Array>();
            let volume_array = batch
                .column(volume_col)
                .as_any()
                .downcast_ref::<Float64Array>();

            // Ensure we have valid arrays
            let (open_arr, high_arr, low_arr, close_arr, volume_arr) =
                match (open_array, high_array, low_array, close_array, volume_array) {
                    (Some(o), Some(h), Some(l), Some(c), Some(v)) => (o, h, l, c, v),
                    _ => {
                        return Err(MLError::DataLoadingError(
                            "OHLCV columns must be Float64".to_string(),
                        ));
                    }
                };

            // Try to get timestamp as Int64 (most common for milliseconds/nanoseconds)
            let ts_int_array = ts_array.as_any().downcast_ref::<Int64Array>();

            // Optional string columns
            let symbol_array =
                symbol_col.and_then(|idx| batch.column(idx).as_any().downcast_ref::<StringArray>());
            let exchange_array = exchange_col
                .and_then(|idx| batch.column(idx).as_any().downcast_ref::<StringArray>());

            for row in 0..num_rows {
                // Parse timestamp
                let timestamp_ms = if let Some(ts_arr) = ts_int_array {
                    if ts_arr.is_null(row) {
                        continue;
                    }
                    let ts_val = ts_arr.value(row);
                    // Heuristic: if > 10^15, assume nanoseconds; if > 10^12, assume microseconds; else milliseconds
                    if ts_val > 1_000_000_000_000_000 {
                        ts_val / 1_000_000 // nanoseconds to milliseconds
                    } else if ts_val > 1_000_000_000_000 {
                        ts_val / 1_000 // microseconds to milliseconds
                    } else {
                        ts_val // already milliseconds
                    }
                } else {
                    // Try to parse as string timestamp
                    continue;
                };

                let timestamp =
                    DateTime::from_timestamp_millis(timestamp_ms).unwrap_or_else(Utc::now);

                // Track time range
                match &min_ts {
                    None => min_ts = Some(timestamp),
                    Some(ts) if timestamp < *ts => min_ts = Some(timestamp),
                    _ => {}
                }
                match &max_ts {
                    None => max_ts = Some(timestamp),
                    Some(ts) if timestamp > *ts => max_ts = Some(timestamp),
                    _ => {}
                }

                // Get OHLCV values
                if open_arr.is_null(row)
                    || high_arr.is_null(row)
                    || low_arr.is_null(row)
                    || close_arr.is_null(row)
                    || volume_arr.is_null(row)
                {
                    continue;
                }

                let open = open_arr.value(row);
                let high = high_arr.value(row);
                let low = low_arr.value(row);
                let close = close_arr.value(row);
                let volume = volume_arr.value(row);

                // Get optional symbol/exchange
                let symbol = symbol_array
                    .and_then(|arr| {
                        if arr.is_null(row) {
                            None
                        } else {
                            Some(arr.value(row).to_string())
                        }
                    })
                    .unwrap_or_else(|| "UNKNOWN".to_string());

                let exchange = exchange_array
                    .and_then(|arr| {
                        if arr.is_null(row) {
                            None
                        } else {
                            Some(arr.value(row).to_string())
                        }
                    })
                    .unwrap_or_else(|| "UNKNOWN".to_string());

                symbols_set.insert(symbol.clone());
                exchanges_set.insert(exchange.clone());

                // Extract features
                let mut features = Vec::with_capacity(6);

                // Feature 1: Price return (if we have previous close)
                let price_return = if let Some(pc) = prev_close {
                    if pc > 0.0 { (close - pc) / pc } else { 0.0 }
                } else {
                    0.0
                };
                features.push(price_return);

                // Feature 2: Range ratio (volatility proxy)
                let range_ratio = if close > 0.0 {
                    (high - low) / close
                } else {
                    0.0
                };
                features.push(range_ratio);

                // Feature 3: Upper shadow ratio
                let body_top = open.max(close);
                let upper_shadow = if high > 0.0 {
                    (high - body_top) / high
                } else {
                    0.0
                };
                features.push(upper_shadow);

                // Feature 4: Lower shadow ratio
                let body_bottom = open.min(close);
                let lower_shadow = if low > 0.0 {
                    (body_bottom - low) / low.max(1e-10)
                } else {
                    0.0
                };
                features.push(lower_shadow);

                // Feature 5: Body ratio (bullish/bearish indicator)
                let body_ratio = if high - low > 0.0 {
                    (close - open) / (high - low)
                } else {
                    0.0
                };
                features.push(body_ratio);

                // Feature 6: Log volume (normalized)
                let log_volume = if volume > 0.0 { volume.ln() } else { 0.0 };
                features.push(log_volume);

                // Target: next close price (will be shifted later)
                let target = close;

                samples.push(MarketDataSample {
                    timestamp: timestamp_ms,
                    features,
                    target,
                    metadata: SampleMetadata {
                        symbol,
                        exchange,
                        price: close,
                        volume,
                    },
                });

                prev_close = Some(close);
            }
        }

        // Shift targets: each sample's target becomes the next sample's close
        if samples.len() > 1 {
            for i in 0..samples.len() - 1 {
                samples[i].target = samples[i + 1].metadata.price;
            }
            // Remove last sample (no target available)
            samples.pop();
        }

        let num_features = samples.first().map(|s| s.features.len()).unwrap_or(0);
        let num_samples = samples.len();
        let symbols: Vec<String> = symbols_set.into_iter().collect();
        let exchanges: Vec<String> = exchanges_set.into_iter().collect();

        tracing::info!(
            "Loaded {} samples with {} features from Parquet",
            num_samples,
            num_features
        );

        // Build metadata first before moving samples
        let metadata = DatasetMetadata {
            num_samples,
            num_features,
            start_time: min_ts.unwrap_or_else(Utc::now),
            end_time: max_ts.unwrap_or_else(Utc::now),
            symbols,
            exchanges,
            source: path_str,
        };

        Ok(Self { samples, metadata })
    }

    /// Load dataset from CSV files
    ///
    /// Expected CSV format:
    /// timestamp,symbol,exchange,open,high,low,close,volume
    ///
    /// The loader will:
    /// 1. Parse each row into OHLCV data
    /// 2. Extract features (price returns, volume ratio, etc.)
    /// 3. Create samples with the next close price as target
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();
        tracing::info!("Loading dataset from CSV: {}", path_str);

        let file = File::open(path_ref)
            .map_err(|e| MLError::DataLoadingError(format!("Failed to open CSV file: {}", e)))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Parse header to identify columns
        let header = lines
            .next()
            .ok_or_else(|| MLError::DataLoadingError("Empty CSV file".to_string()))?
            .map_err(|e| MLError::DataLoadingError(format!("Failed to read header: {}", e)))?;

        let columns: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

        // Find column indices
        let ts_idx = Self::find_column_index(&columns, &["timestamp", "time", "ts", "date"])?;
        let symbol_idx = Self::find_column_index(&columns, &["symbol", "pair", "ticker"]).ok();
        let exchange_idx = Self::find_column_index(&columns, &["exchange", "exch"]).ok();
        let open_idx = Self::find_column_index(&columns, &["open", "o"])?;
        let high_idx = Self::find_column_index(&columns, &["high", "h"])?;
        let low_idx = Self::find_column_index(&columns, &["low", "l"])?;
        let close_idx = Self::find_column_index(&columns, &["close", "c"])?;
        let volume_idx = Self::find_column_index(&columns, &["volume", "vol", "v"])?;

        let mut samples = Vec::new();
        let mut symbols = std::collections::HashSet::new();
        let mut exchanges = std::collections::HashSet::new();
        let mut prev_close: Option<f64> = None;
        let mut min_ts: Option<DateTime<Utc>> = None;
        let mut max_ts: Option<DateTime<Utc>> = None;

        for (line_num, line_result) in lines.enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("Skipping line {}: {}", line_num + 2, e);
                    continue;
                }
            };

            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

            if fields.len() <= volume_idx {
                tracing::warn!("Skipping line {}: insufficient columns", line_num + 2);
                continue;
            }

            // Parse timestamp
            let timestamp = match Self::parse_timestamp(fields[ts_idx]) {
                Ok(ts) => ts,
                Err(e) => {
                    tracing::warn!("Skipping line {}: {}", line_num + 2, e);
                    continue;
                }
            };

            // Track time range
            match &min_ts {
                None => min_ts = Some(timestamp),
                Some(ts) if timestamp < *ts => min_ts = Some(timestamp),
                _ => {}
            }
            match &max_ts {
                None => max_ts = Some(timestamp),
                Some(ts) if timestamp > *ts => max_ts = Some(timestamp),
                _ => {}
            }

            // Parse OHLCV values
            let open: f64 = fields[open_idx].parse().unwrap_or(0.0);
            let high: f64 = fields[high_idx].parse().unwrap_or(0.0);
            let low: f64 = fields[low_idx].parse().unwrap_or(0.0);
            let close: f64 = fields[close_idx].parse().unwrap_or(0.0);
            let volume: f64 = fields[volume_idx].parse().unwrap_or(0.0);

            // Extract symbol and exchange if available
            let symbol = symbol_idx
                .and_then(|idx| fields.get(idx).map(|s| s.to_string()))
                .unwrap_or_else(|| "UNKNOWN".to_string());
            let exchange = exchange_idx
                .and_then(|idx| fields.get(idx).map(|s| s.to_string()))
                .unwrap_or_else(|| "UNKNOWN".to_string());

            symbols.insert(symbol.clone());
            exchanges.insert(exchange.clone());

            // Extract features from OHLCV
            let mut features = Vec::with_capacity(8);

            // Price return (if we have previous close)
            let price_return = prev_close
                .map(|pc| if pc != 0.0 { (close - pc) / pc } else { 0.0 })
                .unwrap_or(0.0);
            features.push(price_return);

            // High-low range normalized by close
            let range = if close != 0.0 {
                (high - low) / close
            } else {
                0.0
            };
            features.push(range);

            // Close position within range (0 = at low, 1 = at high)
            let close_position = if high != low {
                (close - low) / (high - low)
            } else {
                0.5
            };
            features.push(close_position);

            // Open-close body normalized
            let body = if close != 0.0 {
                (close - open) / close
            } else {
                0.0
            };
            features.push(body);

            // Volume (normalized later in windowing)
            features.push(volume);

            // Upper shadow ratio
            let upper_shadow = if high != low {
                (high - close.max(open)) / (high - low)
            } else {
                0.0
            };
            features.push(upper_shadow);

            // Lower shadow ratio
            let lower_shadow = if high != low {
                (close.min(open) - low) / (high - low)
            } else {
                0.0
            };
            features.push(lower_shadow);

            // Typical price (HLC/3)
            let typical_price = (high + low + close) / 3.0;
            features.push(typical_price);

            // Create sample with next close as target (will be updated)
            let sample = MarketDataSample {
                timestamp: timestamp.timestamp_millis(),
                features,
                target: close, // Placeholder - will be updated for prediction
                metadata: SampleMetadata {
                    symbol,
                    exchange,
                    price: close,
                    volume,
                },
            };

            samples.push(sample);
            prev_close = Some(close);
        }

        // Update targets to be next period's close (shift by 1)
        for i in 0..samples.len().saturating_sub(1) {
            samples[i].target = samples[i + 1].metadata.price;
        }

        // Remove last sample since it has no target
        if !samples.is_empty() {
            samples.pop();
        }

        let num_samples = samples.len();
        let num_features = samples.first().map(|s| s.features.len()).unwrap_or(0);

        let metadata = DatasetMetadata {
            num_samples,
            num_features,
            start_time: min_ts.unwrap_or_else(Utc::now),
            end_time: max_ts.unwrap_or_else(Utc::now),
            symbols: symbols.into_iter().collect(),
            exchanges: exchanges.into_iter().collect(),
            source: path_str,
        };

        tracing::info!(
            "Loaded {} samples with {} features from CSV",
            num_samples,
            num_features
        );

        Ok(Self { samples, metadata })
    }

    /// Find column index by name (case-insensitive, supports aliases)
    fn find_column_index(columns: &[&str], names: &[&str]) -> Result<usize> {
        for name in names {
            for (idx, col) in columns.iter().enumerate() {
                if col.eq_ignore_ascii_case(name) {
                    return Ok(idx);
                }
            }
        }
        Err(MLError::DataLoadingError(format!(
            "Column not found: {:?}",
            names
        )))
    }

    /// Parse timestamp from various formats
    fn parse_timestamp(s: &str) -> std::result::Result<DateTime<Utc>, String> {
        // Try parsing as Unix timestamp (seconds)
        if let Ok(ts) = s.parse::<i64>() {
            if ts > 1_000_000_000_000 {
                // Milliseconds
                return chrono::DateTime::from_timestamp_millis(ts)
                    .ok_or_else(|| format!("Invalid timestamp millis: {}", s));
            } else {
                // Seconds
                return chrono::DateTime::from_timestamp(ts, 0)
                    .ok_or_else(|| format!("Invalid timestamp seconds: {}", s));
            }
        }

        // Try parsing as Unix timestamp with decimals
        if let Ok(ts) = s.parse::<f64>() {
            let secs = ts.trunc() as i64;
            let nanos = ((ts.fract()) * 1_000_000_000.0) as u32;
            return chrono::DateTime::from_timestamp(secs, nanos)
                .ok_or_else(|| format!("Invalid timestamp float: {}", s));
        }

        // Try parsing ISO 8601 / RFC 3339 format
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
            return Ok(dt.with_timezone(&Utc));
        }

        // Try common date formats
        let formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
        ];

        for fmt in &formats {
            if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
                return Ok(dt.and_utc());
            }
        }

        Err(format!("Failed to parse timestamp: {}", s))
    }

    /// Create dataset from in-memory samples
    pub fn from_samples(samples: Vec<MarketDataSample>) -> Result<Self> {
        if samples.is_empty() {
            return Err(MLError::InvalidInput(
                "Cannot create dataset from empty samples".to_string(),
            ));
        }

        let num_samples = samples.len();
        let num_features = samples[0].features.len();

        // Extract unique symbols and exchanges
        let mut symbols: Vec<String> = samples.iter().map(|s| s.metadata.symbol.clone()).collect();
        symbols.sort();
        symbols.dedup();

        let mut exchanges: Vec<String> = samples
            .iter()
            .map(|s| s.metadata.exchange.clone())
            .collect();
        exchanges.sort();
        exchanges.dedup();

        // Find time range
        let start_time = DateTime::from_timestamp_micros(samples.first().unwrap().timestamp)
            .unwrap_or_else(Utc::now);

        let end_time = DateTime::from_timestamp_micros(samples.last().unwrap().timestamp)
            .unwrap_or_else(Utc::now);

        let metadata = DatasetMetadata {
            num_samples,
            num_features,
            start_time,
            end_time,
            symbols,
            exchanges,
            source: "in-memory".to_string(),
        };

        Ok(Self { samples, metadata })
    }

    /// Get dataset size
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.metadata.num_features
    }

    /// Split dataset into train/validation/test sets
    ///
    /// # Arguments
    /// * `train_ratio` - Fraction for training (e.g., 0.7)
    /// * `val_ratio` - Fraction for validation (e.g., 0.15)
    /// * `test_ratio` - Fraction for testing (e.g., 0.15)
    ///
    /// Returns (train, validation, test) datasets
    pub fn split(
        self,
        train_ratio: f64,
        val_ratio: f64,
        test_ratio: f64,
    ) -> Result<(Self, Self, Self)> {
        if (train_ratio + val_ratio + test_ratio - 1.0).abs() > 1e-6 {
            return Err(MLError::InvalidInput(
                "Split ratios must sum to 1.0".to_string(),
            ));
        }

        let total = self.samples.len();
        let train_end = (total as f64 * train_ratio) as usize;
        let val_end = train_end + (total as f64 * val_ratio) as usize;

        let mut samples = self.samples;

        let test_samples = samples.split_off(val_end);
        let val_samples = samples.split_off(train_end);
        let train_samples = samples;

        let train = Self::from_samples(train_samples)?;
        let val = Self::from_samples(val_samples)?;
        let test = Self::from_samples(test_samples)?;

        Ok((train, val, test))
    }

    /// Convert to windowed dataset for time series modeling
    pub fn into_windowed(self, config: WindowConfig) -> Result<WindowedDataset> {
        WindowedDataset::from_dataset(self, config)
    }

    /// Filter dataset by symbol
    pub fn filter_by_symbol(&mut self, symbol: &str) {
        self.samples.retain(|s| s.metadata.symbol == symbol);
        self.metadata.num_samples = self.samples.len();
        self.metadata.symbols = vec![symbol.to_string()];
    }

    /// Filter dataset by time range
    pub fn filter_by_time(&mut self, start: DateTime<Utc>, end: DateTime<Utc>) {
        let start_micros = start.timestamp_micros();
        let end_micros = end.timestamp_micros();

        self.samples
            .retain(|s| s.timestamp >= start_micros && s.timestamp <= end_micros);

        self.metadata.num_samples = self.samples.len();
        if let Some(first) = self.samples.first() {
            self.metadata.start_time =
                DateTime::from_timestamp_micros(first.timestamp).unwrap_or_else(Utc::now);
        }
        if let Some(last) = self.samples.last() {
            self.metadata.end_time =
                DateTime::from_timestamp_micros(last.timestamp).unwrap_or_else(Utc::now);
        }
    }
}

impl Default for MarketDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Windowed dataset for time series modeling
///
/// Each window contains (features, target) where:
/// - features: [window_size, num_features] tensor
/// - target: scalar or vector for prediction
#[derive(Debug, Clone)]
pub struct WindowedDataset {
    /// All windows
    pub windows: Vec<Window>,

    /// Window configuration
    pub config: WindowConfig,

    /// Original dataset metadata
    pub metadata: DatasetMetadata,
}

/// A single window sample
#[derive(Debug, Clone)]
pub struct Window {
    /// Input features: [window_size, num_features]
    pub features: Vec<Vec<f64>>,

    /// Target value(s) to predict
    pub target: Vec<f64>,

    /// Timestamp of the last observation in window
    pub timestamp: i64,

    /// Metadata for this window
    pub metadata: SampleMetadata,
}

impl WindowedDataset {
    /// Create windowed dataset from regular dataset
    pub fn from_dataset(dataset: MarketDataset, config: WindowConfig) -> Result<Self> {
        if dataset.samples.len() < config.min_samples {
            return Err(MLError::InvalidInput(format!(
                "Dataset has {} samples but requires at least {}",
                dataset.samples.len(),
                config.min_samples
            )));
        }

        let mut windows = Vec::new();

        // Create sliding windows
        let mut i = 0;
        while i + config.window_size + config.horizon <= dataset.samples.len() {
            // Extract window of features
            let window_samples = &dataset.samples[i..i + config.window_size];
            let features: Vec<Vec<f64>> =
                window_samples.iter().map(|s| s.features.clone()).collect();

            // Extract target(s) from future timesteps
            let target_sample = &dataset.samples[i + config.window_size + config.horizon - 1];
            let target = vec![target_sample.target];

            let window = Window {
                features,
                target,
                timestamp: target_sample.timestamp,
                metadata: target_sample.metadata.clone(),
            };

            windows.push(window);

            i += config.stride;
        }

        tracing::info!(
            "Created {} windows from {} samples (window_size={}, horizon={}, stride={})",
            windows.len(),
            dataset.samples.len(),
            config.window_size,
            config.horizon,
            config.stride
        );

        Ok(Self {
            windows,
            config,
            metadata: dataset.metadata,
        })
    }

    /// Get number of windows
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Get a specific window
    pub fn get(&self, index: usize) -> Option<&Window> {
        self.windows.get(index)
    }

    /// Convert window to tensors for training
    pub fn get_tensor_batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 2>)> {
        if indices.is_empty() {
            return Err(MLError::InvalidInput("Empty batch indices".to_string()));
        }

        let batch_size = indices.len();
        let window_size = self.config.window_size;
        let num_features = self.metadata.num_features;

        // Prepare feature data: [batch_size, window_size, num_features]
        let mut feature_data = Vec::with_capacity(batch_size * window_size * num_features);

        // Prepare target data: [batch_size, 1]
        let mut target_data = Vec::with_capacity(batch_size);

        for &idx in indices {
            if idx >= self.windows.len() {
                return Err(MLError::InvalidInput(format!(
                    "Index {} out of bounds (dataset size: {})",
                    idx,
                    self.windows.len()
                )));
            }

            let window = &self.windows[idx];

            // Add features (flattened)
            for feature_vec in &window.features {
                feature_data.extend_from_slice(feature_vec);
            }

            // Add target
            target_data.push(window.target[0]);
        }

        // Create tensors
        let feature_shape = Shape::new([batch_size, window_size, num_features]);
        let target_shape = Shape::new([batch_size, 1]);

        let features_tensor = Tensor::<B, 3>::from_data(
            TensorData::new(feature_data, feature_shape).convert::<B::FloatElem>(),
            device,
        );

        let targets_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_data, target_shape).convert::<B::FloatElem>(),
            device,
        );

        Ok((features_tensor, targets_tensor))
    }

    /// Split into train/val/test
    pub fn split(
        mut self,
        train_ratio: f64,
        val_ratio: f64,
        test_ratio: f64,
    ) -> Result<(Self, Self, Self)> {
        if (train_ratio + val_ratio + test_ratio - 1.0).abs() > 1e-6 {
            return Err(MLError::InvalidInput(
                "Split ratios must sum to 1.0".to_string(),
            ));
        }

        let total = self.windows.len();
        let train_end = (total as f64 * train_ratio) as usize;
        let val_end = train_end + (total as f64 * val_ratio) as usize;

        let test_windows = self.windows.split_off(val_end);
        let val_windows = self.windows.split_off(train_end);
        let train_windows = self.windows;

        let train = Self {
            windows: train_windows,
            config: self.config.clone(),
            metadata: self.metadata.clone(),
        };

        let val = Self {
            windows: val_windows,
            config: self.config.clone(),
            metadata: self.metadata.clone(),
        };

        let test = Self {
            windows: test_windows,
            config: self.config.clone(),
            metadata: self.metadata.clone(),
        };

        Ok((train, val, test))
    }
}

/// Data loader for batching and iterating over datasets
pub struct DataLoader {
    /// The windowed dataset
    dataset: WindowedDataset,

    /// Batch size
    batch_size: usize,

    /// Whether to shuffle indices
    shuffle: bool,

    /// Random seed for shuffling
    seed: Option<u64>,

    /// Current indices (shuffled or sequential)
    indices: Vec<usize>,

    /// Current position in iteration
    position: usize,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new(dataset: WindowedDataset) -> Self {
        let num_samples = dataset.len();
        let indices: Vec<usize> = (0..num_samples).collect();

        Self {
            dataset,
            batch_size: 32,
            shuffle: false,
            seed: None,
            indices,
            position: 0,
        }
    }

    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable/disable shuffling
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed for shuffling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Reset the data loader (re-shuffle if enabled)
    pub fn reset(&mut self) {
        self.position = 0;

        if self.shuffle {
            // Simple Fisher-Yates shuffle
            use rand::SeedableRng;
            use rand::seq::SliceRandom;

            if let Some(seed) = self.seed {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                self.indices.shuffle(&mut rng);
            } else {
                let mut rng = rand::rng();
                self.indices.shuffle(&mut rng);
            }
        }
    }

    /// Get the next batch
    pub fn next_batch<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Option<Result<(Tensor<B, 3>, Tensor<B, 2>)>> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..end];

        self.position = end;

        Some(self.dataset.get_tensor_batch(batch_indices, device))
    }

    /// Get number of batches per epoch
    pub fn num_batches(&self) -> usize {
        self.dataset.len().div_ceil(self.batch_size)
    }

    /// Get dataset reference
    pub fn dataset(&self) -> &WindowedDataset {
        &self.dataset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    fn create_test_samples(n: usize) -> Vec<MarketDataSample> {
        (0..n)
            .map(|i| MarketDataSample {
                timestamp: (i as i64) * 1_000_000, // 1 second intervals
                features: vec![i as f64, (i * 2) as f64, (i * 3) as f64],
                target: (i + 1) as f64,
                metadata: SampleMetadata {
                    symbol: "BTCUSD".to_string(),
                    exchange: "binance".to_string(),
                    price: 50000.0 + i as f64,
                    volume: 100.0,
                },
            })
            .collect()
    }

    #[test]
    fn test_market_dataset_creation() {
        let samples = create_test_samples(100);
        let dataset = MarketDataset::from_samples(samples).unwrap();

        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.feature_dim(), 3);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_dataset_split() {
        let samples = create_test_samples(100);
        let dataset = MarketDataset::from_samples(samples).unwrap();

        let (train, val, test) = dataset.split(0.7, 0.15, 0.15).unwrap();

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 15);
        assert_eq!(test.len(), 15);
    }

    #[test]
    fn test_window_config() {
        let config = WindowConfig::new(50, 1).with_stride(2).with_min_samples(60);

        assert_eq!(config.window_size, 50);
        assert_eq!(config.horizon, 1);
        assert_eq!(config.stride, 2);
        assert_eq!(config.min_samples, 60);
    }

    #[test]
    fn test_windowed_dataset() {
        let samples = create_test_samples(100);
        let dataset = MarketDataset::from_samples(samples).unwrap();

        let config = WindowConfig::new(10, 1).with_stride(1);
        let windowed = dataset.into_windowed(config).unwrap();

        // With 100 samples, window_size=10, horizon=1, stride=1
        // We get 100 - 10 - 1 + 1 = 90 windows
        assert_eq!(windowed.len(), 90);

        let window = windowed.get(0).unwrap();
        assert_eq!(window.features.len(), 10);
        assert_eq!(window.features[0].len(), 3);
        assert_eq!(window.target.len(), 1);
    }

    #[test]
    fn test_windowed_dataset_with_stride() {
        let samples = create_test_samples(100);
        let dataset = MarketDataset::from_samples(samples).unwrap();

        let config = WindowConfig::new(10, 1).with_stride(5);
        let windowed = dataset.into_windowed(config).unwrap();

        // With stride=5, we get fewer windows
        assert!(windowed.len() < 90);
        assert!(!windowed.is_empty());
    }

    #[test]
    fn test_data_loader() {
        let samples = create_test_samples(100);
        let dataset = MarketDataset::from_samples(samples).unwrap();
        let windowed = dataset.into_windowed(WindowConfig::new(10, 1)).unwrap();

        let mut loader = DataLoader::new(windowed).batch_size(16).shuffle(false);

        assert_eq!(loader.num_batches(), 6); // 90 samples / 16 = 5.625 -> 6 batches

        let device = <CpuBackend as Backend>::Device::default();
        let batch = loader.next_batch::<crate::backend::CpuBackend>(&device);
        assert!(batch.is_some());

        let (features, targets) = batch.unwrap().unwrap();
        let feature_dims = features.shape().dims;
        let target_dims = targets.shape().dims;
        assert_eq!(feature_dims[0], 16); // batch size
        assert_eq!(feature_dims[1], 10); // window size
        assert_eq!(feature_dims[2], 3); // num features
        assert_eq!(target_dims[0], 16); // batch size
        assert_eq!(target_dims[1], 1); // target dimension
    }

    #[test]
    fn test_data_loader_reset() {
        let samples = create_test_samples(50);
        let dataset = MarketDataset::from_samples(samples).unwrap();
        let windowed = dataset.into_windowed(WindowConfig::new(5, 1)).unwrap();

        let mut loader = DataLoader::new(windowed).batch_size(10);

        // Consume all batches
        let device = <crate::backend::CpuBackend as Backend>::Device::default();
        while loader
            .next_batch::<crate::backend::CpuBackend>(&device)
            .is_some()
        {}

        // Reset and try again
        loader.reset();
        let batch = loader.next_batch::<crate::backend::CpuBackend>(&device);
        assert!(batch.is_some());
    }

    #[test]
    fn test_filter_by_symbol() {
        let mut samples = create_test_samples(50);
        samples[25].metadata.symbol = "ETHUSDT".to_string();

        let mut dataset = MarketDataset::from_samples(samples).unwrap();
        dataset.filter_by_symbol("BTCUSD");

        assert_eq!(dataset.len(), 49);
        assert_eq!(dataset.metadata.symbols, vec!["BTCUSD"]);
    }

    #[test]
    fn test_empty_dataset_error() {
        let result = MarketDataset::from_samples(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_split_ratios() {
        let samples = create_test_samples(100);
        let dataset = MarketDataset::from_samples(samples).unwrap();

        let result = dataset.split(0.5, 0.3, 0.3); // Sum > 1.0
        assert!(result.is_err());
    }
}
