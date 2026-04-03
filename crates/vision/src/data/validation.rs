//! Data validation utilities for OHLCV candles.
//!
//! This module provides functions to validate OHLCV data quality,
//! detect anomalies, and ensure data integrity.

use crate::data::csv_loader::OhlcvCandle;
#[cfg_attr(not(test), allow(unused_imports))]
use chrono::Duration;
use std::fmt;

/// Validation errors
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Chronological order violation
    ChronologicalOrder { index: usize, message: String },
    /// Duplicate timestamp
    DuplicateTimestamp { index: usize, timestamp: String },
    /// Gap in time series
    TimeGap {
        index: usize,
        gap_seconds: i64,
        expected_seconds: i64,
    },
    /// Invalid OHLC relationship
    InvalidOhlc { index: usize, message: String },
    /// Extreme outlier detected
    Outlier {
        index: usize,
        field: String,
        value: f64,
        mean: f64,
        std_dev: f64,
        z_score: f64,
    },
    /// Negative or zero price
    InvalidPrice {
        index: usize,
        field: String,
        value: f64,
    },
    /// Negative volume
    NegativeVolume { index: usize, volume: f64 },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::ChronologicalOrder { index, message } => {
                write!(
                    f,
                    "Chronological order error at index {}: {}",
                    index, message
                )
            }
            ValidationError::DuplicateTimestamp { index, timestamp } => {
                write!(f, "Duplicate timestamp at index {}: {}", index, timestamp)
            }
            ValidationError::TimeGap {
                index,
                gap_seconds,
                expected_seconds,
            } => {
                write!(
                    f,
                    "Time gap at index {}: {} seconds (expected ~{})",
                    index, gap_seconds, expected_seconds
                )
            }
            ValidationError::InvalidOhlc { index, message } => {
                write!(f, "Invalid OHLC at index {}: {}", index, message)
            }
            ValidationError::Outlier {
                index,
                field,
                value,
                mean,
                std_dev,
                z_score,
            } => {
                write!(
                    f,
                    "Outlier at index {} ({}): value={:.2}, mean={:.2}, std={:.2}, z-score={:.2}",
                    index, field, value, mean, std_dev, z_score
                )
            }
            ValidationError::InvalidPrice {
                index,
                field,
                value,
            } => {
                write!(f, "Invalid price at index {} ({}): {}", index, field, value)
            }
            ValidationError::NegativeVolume { index, volume } => {
                write!(f, "Negative volume at index {}: {}", index, volume)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub total_candles: usize,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationError>,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new(total_candles: usize) -> Self {
        Self {
            total_candles,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Check if validation passed (no errors)
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get total number of issues (errors + warnings)
    pub fn total_issues(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }

    /// Print summary to stdout
    pub fn print_summary(&self) {
        println!("Validation Report:");
        println!("  Total candles: {}", self.total_candles);
        println!("  Errors: {}", self.errors.len());
        println!("  Warnings: {}", self.warnings.len());

        if !self.errors.is_empty() {
            println!("\nErrors:");
            for error in &self.errors {
                println!("  - {}", error);
            }
        }

        if !self.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &self.warnings {
                println!("  - {}", warning);
            }
        }
    }
}

/// Validate OHLCV data
pub fn validate_ohlcv(candles: &[OhlcvCandle]) -> ValidationReport {
    let mut report = ValidationReport::new(candles.len());

    if candles.is_empty() {
        return report;
    }

    // Check chronological order
    validate_chronological_order(candles, &mut report);

    // Check OHLC relationships
    validate_ohlc_relationships(candles, &mut report);

    // Check for price outliers
    validate_price_outliers(candles, &mut report, 6.0); // 6 sigma

    // Check for volume outliers
    validate_volume_outliers(candles, &mut report, 6.0);

    // Check for time gaps
    validate_time_gaps(candles, &mut report);

    report
}

/// Validate chronological order
fn validate_chronological_order(candles: &[OhlcvCandle], report: &mut ValidationReport) {
    for i in 1..candles.len() {
        if candles[i].timestamp <= candles[i - 1].timestamp {
            if candles[i].timestamp == candles[i - 1].timestamp {
                report.errors.push(ValidationError::DuplicateTimestamp {
                    index: i,
                    timestamp: candles[i].timestamp.to_rfc3339(),
                });
            } else {
                report.errors.push(ValidationError::ChronologicalOrder {
                    index: i,
                    message: format!(
                        "Timestamp {} is before previous timestamp {}",
                        candles[i].timestamp.to_rfc3339(),
                        candles[i - 1].timestamp.to_rfc3339()
                    ),
                });
            }
        }
    }
}

/// Validate OHLC relationships
fn validate_ohlc_relationships(candles: &[OhlcvCandle], report: &mut ValidationReport) {
    for (i, candle) in candles.iter().enumerate() {
        // Check prices are positive
        if candle.open <= 0.0 {
            report.errors.push(ValidationError::InvalidPrice {
                index: i,
                field: "open".to_string(),
                value: candle.open,
            });
        }
        if candle.high <= 0.0 {
            report.errors.push(ValidationError::InvalidPrice {
                index: i,
                field: "high".to_string(),
                value: candle.high,
            });
        }
        if candle.low <= 0.0 {
            report.errors.push(ValidationError::InvalidPrice {
                index: i,
                field: "low".to_string(),
                value: candle.low,
            });
        }
        if candle.close <= 0.0 {
            report.errors.push(ValidationError::InvalidPrice {
                index: i,
                field: "close".to_string(),
                value: candle.close,
            });
        }

        // Check volume is non-negative
        if candle.volume < 0.0 {
            report.errors.push(ValidationError::NegativeVolume {
                index: i,
                volume: candle.volume,
            });
        }

        // Check OHLC relationships
        if candle.high < candle.low {
            report.errors.push(ValidationError::InvalidOhlc {
                index: i,
                message: format!("high ({}) < low ({})", candle.high, candle.low),
            });
        }
        if candle.high < candle.open {
            report.errors.push(ValidationError::InvalidOhlc {
                index: i,
                message: format!("high ({}) < open ({})", candle.high, candle.open),
            });
        }
        if candle.high < candle.close {
            report.errors.push(ValidationError::InvalidOhlc {
                index: i,
                message: format!("high ({}) < close ({})", candle.high, candle.close),
            });
        }
        if candle.low > candle.open {
            report.errors.push(ValidationError::InvalidOhlc {
                index: i,
                message: format!("low ({}) > open ({})", candle.low, candle.open),
            });
        }
        if candle.low > candle.close {
            report.errors.push(ValidationError::InvalidOhlc {
                index: i,
                message: format!("low ({}) > close ({})", candle.low, candle.close),
            });
        }
    }
}

/// Validate price outliers using z-score
fn validate_price_outliers(candles: &[OhlcvCandle], report: &mut ValidationReport, threshold: f64) {
    if candles.len() < 10 {
        return; // Not enough data for outlier detection
    }

    // Calculate mean and std for close prices
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let mean = closes.iter().sum::<f64>() / closes.len() as f64;
    let variance = closes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / closes.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return; // No variation
    }

    for (i, candle) in candles.iter().enumerate() {
        let z_score = (candle.close - mean).abs() / std_dev;
        if z_score > threshold {
            report.warnings.push(ValidationError::Outlier {
                index: i,
                field: "close".to_string(),
                value: candle.close,
                mean,
                std_dev,
                z_score,
            });
        }
    }
}

/// Validate volume outliers
fn validate_volume_outliers(
    candles: &[OhlcvCandle],
    report: &mut ValidationReport,
    threshold: f64,
) {
    if candles.len() < 10 {
        return;
    }

    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let variance = volumes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / volumes.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return;
    }

    for (i, candle) in candles.iter().enumerate() {
        let z_score = (candle.volume - mean).abs() / std_dev;
        if z_score > threshold {
            report.warnings.push(ValidationError::Outlier {
                index: i,
                field: "volume".to_string(),
                value: candle.volume,
                mean,
                std_dev,
                z_score,
            });
        }
    }
}

/// Validate time gaps
fn validate_time_gaps(candles: &[OhlcvCandle], report: &mut ValidationReport) {
    if candles.len() < 2 {
        return;
    }

    // Calculate median interval
    let mut intervals: Vec<i64> = candles
        .windows(2)
        .map(|w| (w[1].timestamp - w[0].timestamp).num_seconds())
        .collect();

    if intervals.is_empty() {
        return;
    }

    intervals.sort_unstable();
    let median_interval = intervals[intervals.len() / 2];

    // Check for gaps > 2x median
    for i in 1..candles.len() {
        let gap = (candles[i].timestamp - candles[i - 1].timestamp).num_seconds();
        if gap > median_interval * 2 {
            report.warnings.push(ValidationError::TimeGap {
                index: i,
                gap_seconds: gap,
                expected_seconds: median_interval,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_valid_candle(timestamp_offset: i64, price: f64) -> OhlcvCandle {
        let timestamp = Utc::now() + Duration::seconds(timestamp_offset);
        OhlcvCandle::new(
            timestamp,
            price,
            price + 1.0,
            price - 1.0,
            price + 0.5,
            1000.0,
        )
    }

    #[test]
    fn test_validate_empty() {
        let candles: Vec<OhlcvCandle> = vec![];
        let report = validate_ohlcv(&candles);
        assert!(report.is_valid());
    }

    #[test]
    fn test_validate_valid_data() {
        let candles: Vec<OhlcvCandle> = (0..10)
            .map(|i| create_valid_candle(i * 3600, 100.0 + i as f64))
            .collect();

        let report = validate_ohlcv(&candles);
        assert!(report.is_valid());
    }

    #[test]
    fn test_detect_chronological_error() {
        let candles = vec![
            create_valid_candle(0, 100.0),
            create_valid_candle(3600, 101.0),
            create_valid_candle(1800, 102.0), // Out of order
        ];

        let report = validate_ohlcv(&candles);
        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::ChronologicalOrder { .. }))
        );
    }

    #[test]
    fn test_detect_duplicate_timestamp() {
        let timestamp = Utc::now();
        let candles = vec![
            OhlcvCandle::new(timestamp, 100.0, 105.0, 95.0, 102.0, 1000.0),
            OhlcvCandle::new(timestamp, 101.0, 106.0, 96.0, 103.0, 1100.0), // Duplicate timestamp
        ];

        let report = validate_ohlcv(&candles);
        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::DuplicateTimestamp { .. }))
        );
    }

    #[test]
    fn test_detect_invalid_ohlc() {
        let timestamp = Utc::now();
        let invalid = OhlcvCandle::new(
            timestamp, 100.0, 95.0, // high < open
            90.0, 98.0, 1000.0,
        );

        let report = validate_ohlcv(&vec![invalid]);
        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidOhlc { .. }))
        );
    }

    #[test]
    fn test_detect_negative_price() {
        let timestamp = Utc::now();
        let invalid = OhlcvCandle::new(timestamp, -100.0, 100.0, 90.0, 95.0, 1000.0);

        let report = validate_ohlcv(&vec![invalid]);
        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidPrice { .. }))
        );
    }

    #[test]
    fn test_detect_negative_volume() {
        let timestamp = Utc::now();
        let invalid = OhlcvCandle::new(timestamp, 100.0, 105.0, 95.0, 102.0, -100.0);

        let report = validate_ohlcv(&vec![invalid]);
        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::NegativeVolume { .. }))
        );
    }

    #[test]
    fn test_detect_price_outlier() {
        let mut candles: Vec<OhlcvCandle> = (0..100)
            .map(|i| create_valid_candle(i * 3600, 100.0))
            .collect();

        // Add outlier
        candles.push(create_valid_candle(100 * 3600, 10000.0));

        let report = validate_ohlcv(&candles);
        assert!(report.warnings.len() > 0);
    }

    #[test]
    fn test_detect_time_gap() {
        // Create regular intervals, then a large gap
        let base_time = Utc::now();
        let candles = vec![
            OhlcvCandle::new(base_time, 100.0, 105.0, 95.0, 102.0, 1000.0),
            OhlcvCandle::new(
                base_time + Duration::seconds(3600),
                101.0,
                106.0,
                96.0,
                103.0,
                1000.0,
            ),
            OhlcvCandle::new(
                base_time + Duration::seconds(7200),
                102.0,
                107.0,
                97.0,
                104.0,
                1000.0,
            ),
            OhlcvCandle::new(
                base_time + Duration::seconds(3600 * 10),
                103.0,
                108.0,
                98.0,
                105.0,
                1000.0,
            ), // Large gap
        ];

        let report = validate_ohlcv(&candles);
        assert!(
            report
                .warnings
                .iter()
                .any(|e| matches!(e, ValidationError::TimeGap { .. }))
        );
    }

    #[test]
    fn test_validation_report() {
        let report = ValidationReport::new(100);
        assert!(report.is_valid());
        assert_eq!(report.total_candles, 100);
        assert_eq!(report.total_issues(), 0);
    }
}
