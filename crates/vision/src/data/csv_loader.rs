//! CSV loader for OHLCV (Open, High, Low, Close, Volume) market data.
//!
//! This module provides functionality to load historical market data from CSV files
//! and convert it into a format suitable for training vision models.

use chrono::{DateTime, NaiveDateTime, Utc};
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for CSV data loading
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// CSV delimiter (default: comma)
    pub delimiter: u8,
    /// Whether the CSV has a header row
    pub has_header: bool,
    /// Column indices: (timestamp, open, high, low, close, volume)
    pub column_indices: (usize, usize, usize, usize, usize, usize),
    /// Skip invalid rows instead of failing
    pub skip_invalid: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: true,
            column_indices: (0, 1, 2, 3, 4, 5), // Standard OHLCV format
            skip_invalid: true,
        }
    }
}

/// Single OHLCV candle/bar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvCandle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OhlcvCandle {
    /// Create a new OHLCV candle
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Convert to feature vector [open, high, low, close, volume]
    pub fn to_features(&self) -> Vec<f64> {
        vec![self.open, self.high, self.low, self.close, self.volume]
    }

    /// Get OHLC values as a 4-element array
    pub fn ohlc(&self) -> [f64; 4] {
        [self.open, self.high, self.low, self.close]
    }

    /// Validate OHLC relationships
    pub fn is_valid(&self) -> bool {
        // High must be >= all others
        if self.high < self.open || self.high < self.close || self.high < self.low {
            return false;
        }
        // Low must be <= all others
        if self.low > self.open || self.low > self.close || self.low > self.high {
            return false;
        }
        // Volume should be non-negative
        if self.volume < 0.0 {
            return false;
        }
        // Prices should be positive
        if self.open <= 0.0 || self.high <= 0.0 || self.low <= 0.0 || self.close <= 0.0 {
            return false;
        }
        true
    }
}

/// CSV loader for OHLCV data
pub struct CsvLoader {
    config: LoaderConfig,
}

impl CsvLoader {
    /// Create a new CSV loader with the given configuration
    pub fn new(config: LoaderConfig) -> Self {
        Self { config }
    }

    /// Load OHLCV candles from a CSV file
    pub fn load_csv<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Vec<OhlcvCandle>> {
        let mut reader = ReaderBuilder::new()
            .delimiter(self.config.delimiter)
            .has_headers(self.config.has_header)
            .from_path(path)?;

        let mut candles = Vec::new();
        let (ts_idx, o_idx, h_idx, l_idx, c_idx, v_idx) = self.config.column_indices;

        for (line_num, result) in reader.records().enumerate() {
            let record = match result {
                Ok(r) => r,
                Err(e) => {
                    if self.config.skip_invalid {
                        eprintln!("Skipping line {}: {}", line_num + 1, e);
                        continue;
                    } else {
                        return Err(e.into());
                    }
                }
            };

            // Ensure record has enough columns
            let max_idx = ts_idx
                .max(o_idx)
                .max(h_idx)
                .max(l_idx)
                .max(c_idx)
                .max(v_idx);
            if record.len() <= max_idx {
                if self.config.skip_invalid {
                    eprintln!("Skipping line {}: insufficient columns", line_num + 1);
                    continue;
                } else {
                    return Err(anyhow::anyhow!(
                        "Line {}: expected at least {} columns, got {}",
                        line_num + 1,
                        max_idx + 1,
                        record.len()
                    ));
                }
            }

            // Parse timestamp
            let timestamp = match self.parse_timestamp(&record[ts_idx]) {
                Ok(ts) => ts,
                Err(e) => {
                    if self.config.skip_invalid {
                        eprintln!("Skipping line {}: {}", line_num + 1, e);
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            };

            // Parse OHLCV values
            let parse_field = |idx: usize, name: &str| -> anyhow::Result<f64> {
                record[idx].parse::<f64>().map_err(|e| {
                    anyhow::anyhow!(
                        "Line {}: failed to parse {} '{}': {}",
                        line_num + 1,
                        name,
                        &record[idx],
                        e
                    )
                })
            };

            let candle = match (
                parse_field(o_idx, "open"),
                parse_field(h_idx, "high"),
                parse_field(l_idx, "low"),
                parse_field(c_idx, "close"),
                parse_field(v_idx, "volume"),
            ) {
                (Ok(open), Ok(high), Ok(low), Ok(close), Ok(volume)) => {
                    OhlcvCandle::new(timestamp, open, high, low, close, volume)
                }
                _ => {
                    if self.config.skip_invalid {
                        continue;
                    } else {
                        return Err(anyhow::anyhow!(
                            "Line {}: failed to parse OHLCV values",
                            line_num + 1
                        ));
                    }
                }
            };

            // Validate candle
            if !candle.is_valid() {
                if self.config.skip_invalid {
                    eprintln!("Skipping line {}: invalid OHLC relationships", line_num + 1);
                    continue;
                } else {
                    return Err(anyhow::anyhow!(
                        "Line {}: invalid OHLC relationships (high={}, low={}, open={}, close={})",
                        line_num + 1,
                        candle.high,
                        candle.low,
                        candle.open,
                        candle.close
                    ));
                }
            }

            candles.push(candle);
        }

        Ok(candles)
    }

    /// Parse timestamp from string
    /// Supports:
    /// - Unix timestamp in milliseconds (e.g., "1609459200000")
    /// - Unix timestamp in seconds (e.g., "1609459200")
    /// - ISO 8601 / RFC 3339 (e.g., "2024-01-01T00:00:00Z")
    /// - Common datetime format (e.g., "2024-01-01 00:00:00")
    fn parse_timestamp(&self, s: &str) -> anyhow::Result<DateTime<Utc>> {
        let s = s.trim();

        // Try Unix timestamp (milliseconds)
        if let Ok(millis) = s.parse::<i64>() {
            // If value is large, assume milliseconds
            if millis > 1_000_000_000_000 {
                return DateTime::from_timestamp_millis(millis)
                    .ok_or_else(|| anyhow::anyhow!("Invalid timestamp milliseconds: {}", millis));
            }
            // Otherwise assume seconds
            return DateTime::from_timestamp(millis, 0)
                .ok_or_else(|| anyhow::anyhow!("Invalid timestamp seconds: {}", millis));
        }

        // Try RFC 3339 / ISO 8601
        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
            return Ok(dt.to_utc());
        }

        // Try common datetime format: "YYYY-MM-DD HH:MM:SS"
        if let Ok(naive_dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            return Ok(DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc));
        }

        // Try date only: "YYYY-MM-DD"
        if let Ok(naive_dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d") {
            return Ok(DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc));
        }

        Err(anyhow::anyhow!("Unable to parse timestamp: '{}'", s))
    }
}

/// Convenience function to load OHLCV data from a CSV file with default settings.
///
/// This is a shorthand for creating a CsvLoader with default config and loading a file.
///
/// # Example
///
/// ```no_run
/// use vision::load_ohlcv_csv;
///
/// let candles = load_ohlcv_csv("data/BTCUSDT_1h.csv").unwrap();
/// println!("Loaded {} candles", candles.len());
/// ```
pub fn load_ohlcv_csv<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<OhlcvCandle>> {
    let loader = CsvLoader::new(LoaderConfig::default());
    loader.load_csv(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ohlcv_candle_validation() {
        // Valid candle
        let valid = OhlcvCandle::new(Utc::now(), 100.0, 105.0, 95.0, 102.0, 1000.0);
        assert!(valid.is_valid());

        // Invalid: high < close
        let invalid = OhlcvCandle::new(
            Utc::now(),
            100.0,
            99.0, // high < open
            95.0,
            102.0,
            1000.0,
        );
        assert!(!invalid.is_valid());

        // Invalid: low > open
        let invalid = OhlcvCandle::new(
            Utc::now(),
            100.0,
            105.0,
            101.0, // low > open
            102.0,
            1000.0,
        );
        assert!(!invalid.is_valid());

        // Invalid: negative volume
        let invalid = OhlcvCandle::new(Utc::now(), 100.0, 105.0, 95.0, 102.0, -10.0);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_candle_to_features() {
        let candle = OhlcvCandle::new(Utc::now(), 100.0, 105.0, 95.0, 102.0, 1000.0);
        let features = candle.to_features();
        assert_eq!(features, vec![100.0, 105.0, 95.0, 102.0, 1000.0]);
    }

    #[test]
    fn test_timestamp_parsing_unix_millis() {
        let loader = CsvLoader::new(LoaderConfig::default());
        let ts = loader.parse_timestamp("1609459200000").unwrap();
        assert_eq!(ts.timestamp(), 1609459200);
    }

    #[test]
    fn test_timestamp_parsing_unix_seconds() {
        let loader = CsvLoader::new(LoaderConfig::default());
        let ts = loader.parse_timestamp("1609459200").unwrap();
        assert_eq!(ts.timestamp(), 1609459200);
    }

    #[test]
    fn test_timestamp_parsing_iso8601() {
        let loader = CsvLoader::new(LoaderConfig::default());
        let ts = loader.parse_timestamp("2021-01-01T00:00:00Z").unwrap();
        assert_eq!(ts.timestamp(), 1609459200);
    }

    #[test]
    fn test_timestamp_parsing_common_format() {
        let loader = CsvLoader::new(LoaderConfig::default());
        let ts = loader.parse_timestamp("2021-01-01 00:00:00").unwrap();
        assert_eq!(ts.timestamp(), 1609459200);
    }

    #[test]
    fn test_csv_loading() {
        let csv_data = "\
timestamp,open,high,low,close,volume
1609459200000,29000.0,29500.0,28800.0,29200.0,1000.0
1609462800000,29200.0,29400.0,29100.0,29300.0,1100.0
1609466400000,29300.0,29600.0,29200.0,29500.0,1200.0
";

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(csv_data.as_bytes()).unwrap();

        let loader = CsvLoader::new(LoaderConfig::default());
        let candles = loader.load_csv(file.path()).unwrap();

        assert_eq!(candles.len(), 3);
        assert_eq!(candles[0].open, 29000.0);
        assert_eq!(candles[0].high, 29500.0);
        assert_eq!(candles[0].low, 28800.0);
        assert_eq!(candles[0].close, 29200.0);
        assert_eq!(candles[0].volume, 1000.0);

        assert_eq!(candles[2].close, 29500.0);
    }

    #[test]
    fn test_csv_loading_no_header() {
        let csv_data = "\
1609459200000,29000.0,29500.0,28800.0,29200.0,1000.0
1609462800000,29200.0,29400.0,29100.0,29300.0,1100.0
";

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(csv_data.as_bytes()).unwrap();

        let config = LoaderConfig {
            has_header: false,
            ..Default::default()
        };
        let loader = CsvLoader::new(config);
        let candles = loader.load_csv(file.path()).unwrap();

        assert_eq!(candles.len(), 2);
    }

    #[test]
    fn test_csv_loading_skip_invalid() {
        let csv_data = "\
timestamp,open,high,low,close,volume
1609459200000,29000.0,29500.0,28800.0,29200.0,1000.0
invalid_timestamp,29200.0,29400.0,29100.0,29300.0,1100.0
1609466400000,29300.0,29600.0,29200.0,29500.0,1200.0
";

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(csv_data.as_bytes()).unwrap();

        let loader = CsvLoader::new(LoaderConfig::default());
        let candles = loader.load_csv(file.path()).unwrap();

        // Should skip invalid row
        assert_eq!(candles.len(), 2);
        assert_eq!(candles[0].open, 29000.0);
        assert_eq!(candles[1].open, 29300.0);
    }

    #[test]
    fn test_csv_loading_custom_delimiter() {
        let csv_data = "\
timestamp;open;high;low;close;volume
1609459200000;29000.0;29500.0;28800.0;29200.0;1000.0
";

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(csv_data.as_bytes()).unwrap();

        let config = LoaderConfig {
            delimiter: b';',
            ..Default::default()
        };
        let loader = CsvLoader::new(config);
        let candles = loader.load_csv(file.path()).unwrap();

        assert_eq!(candles.len(), 1);
        assert_eq!(candles[0].open, 29000.0);
    }
}
