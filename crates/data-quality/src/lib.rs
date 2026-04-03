//! # JANUS Data Quality
//!
//! Comprehensive data quality pipeline for validating, detecting anomalies,
//! and ensuring data integrity across all market data sources.
//!
//! ## Features
//!
//! - **Validators**: Price, volume, timestamp, order book validation
//! - **Anomaly Detection**: Statistical outlier detection, sequence analysis
//! - **Gap Detection**: Integration with existing gap detection system
//! - **Parquet Export**: Long-term storage and analytics
//! - **CNS Metrics**: Comprehensive observability (optional)
//!
//! ## Example
//!
//! ```rust,no_run
//! use janus_data_quality::{DataQualityPipeline, Config, MarketDataState};
//! use janus_core::{MarketDataEvent, TradeEvent, Exchange, Symbol, Side};
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//! use rust_decimal::Decimal;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create shared state for tracking
//!     let state = Arc::new(RwLock::new(MarketDataState::default()));
//!
//!     // Create pipeline from config
//!     let config = Config::default();
//!     let mut pipeline = DataQualityPipeline::from_config(config, state);
//!
//!     // Add validators (optional - from_config adds defaults)
//!     // pipeline.add_validator(...);
//!
//!     // Example trade event
//!     let trade = TradeEvent {
//!         exchange: Exchange::Bybit,
//!         symbol: Symbol::new("BTC", "USDT"),
//!         timestamp: chrono::Utc::now().timestamp_micros(),
//!         received_at: chrono::Utc::now().timestamp_micros(),
//!         price: Decimal::new(50000, 0),
//!         quantity: Decimal::new(1, 1),
//!         side: Side::Buy,
//!         trade_id: "12345".to_string(),
//!         buyer_is_maker: Some(false),
//!     };
//!
//!     let event = MarketDataEvent::Trade(trade);
//!     let result = pipeline.process(&event).await?;
//!
//!     if result.is_valid {
//!         println!("Event is valid with quality score: {:.1}%", result.quality_score * 100.0);
//!     } else {
//!         println!("Validation failed: {:?}", result.errors());
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod anomaly;
pub mod error;
pub mod export;
pub mod metrics;
pub mod pipeline;
pub mod validators;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use janus_core::{Exchange, Symbol};
use serde::{Deserialize, Serialize};

pub use anomaly::{
    AnomalyDetector, AnomalyResult, AnomalySeverity, LatencyAnomalyDetector,
    SequenceAnomalyDetector, StatisticalAnomalyDetector,
};
pub use error::{DataQualityError, Result};
pub use export::{CompressionType, ExportConfig, ExportConfigBuilder, ParquetExporter};
pub use metrics::{
    Counter, Gauge, Histogram, MetricsConfig, MetricsRegistry, MetricsSummary, Timer,
};
pub use pipeline::{DataQualityPipeline, PipelineConfig, PipelineStats, ProcessingResult};
pub use validators::{
    OrderBookValidator, PriceValidator, TimestampValidator, Validator, ValidatorConfig,
    VolumeValidator,
};

/// Configuration for the data quality system
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Validator configuration
    pub validator: ValidatorConfig,

    /// Anomaly detection configuration
    pub anomaly: AnomalyConfig,

    /// Gap detection configuration
    pub gap_detection: GapDetectionConfig,

    /// Export configuration
    pub export: Option<ExportConfig>,

    /// Metrics configuration
    pub metrics: Option<MetricsConfig>,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyConfig {
    /// Enable statistical anomaly detection
    pub statistical_enabled: bool,

    /// Z-score threshold for statistical anomalies
    pub z_score_threshold: f64,

    /// Window size for statistical analysis (number of data points)
    pub window_size: usize,

    /// Enable sequence anomaly detection
    pub sequence_enabled: bool,

    /// Maximum allowed gap in sequence numbers
    pub max_sequence_gap: u64,

    /// Enable latency anomaly detection
    pub latency_enabled: bool,

    /// Maximum acceptable latency (microseconds)
    pub max_latency_us: i64,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            statistical_enabled: true,
            z_score_threshold: 3.0,
            window_size: 100,
            sequence_enabled: true,
            max_sequence_gap: 10,
            latency_enabled: true,
            max_latency_us: 5_000_000, // 5 seconds
        }
    }
}

/// Gap detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapDetectionConfig {
    /// Enable gap detection
    pub enabled: bool,

    /// Minimum gap duration to report (microseconds)
    pub min_gap_duration_us: i64,

    /// Check interval (seconds)
    pub check_interval_secs: u64,

    /// Maximum lookback period (hours)
    pub lookback_hours: u64,
}

impl Default for GapDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_gap_duration_us: 60_000_000, // 60 seconds
            check_interval_secs: 300,        // 5 minutes
            lookback_hours: 24,
        }
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResult {
    /// Whether the data is valid
    pub is_valid: bool,

    /// Validation errors (if any)
    pub errors: Vec<String>,

    /// Warnings (non-fatal issues)
    pub warnings: Vec<String>,

    /// Validator name
    pub validator: String,

    /// Timestamp of validation
    pub validated_at: i64,
}

impl ValidationResult {
    /// Create a successful validation result
    pub fn success(validator: impl Into<String>) -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            validator: validator.into(),
            validated_at: chrono::Utc::now().timestamp_micros(),
        }
    }

    /// Create a failed validation result
    pub fn failure(validator: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            errors: vec![error.into()],
            warnings: Vec::new(),
            validator: validator.into(),
            validated_at: chrono::Utc::now().timestamp_micros(),
        }
    }

    /// Add a warning to the result
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }

    /// Add an error to the result
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.is_valid = false;
        self.errors.push(error.into());
        self
    }
}

/// Statistics tracker for anomaly detection
#[derive(Debug, Clone)]
pub struct Statistics {
    /// Mean value
    pub mean: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Number of samples
    pub count: usize,
}

impl Statistics {
    /// Create empty statistics
    pub fn empty() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            count: 0,
        }
    }

    /// Calculate z-score for a value
    pub fn z_score(&self, value: f64) -> f64 {
        if self.std_dev == 0.0 {
            0.0
        } else {
            (value - self.mean) / self.std_dev
        }
    }

    /// Check if value is an outlier
    pub fn is_outlier(&self, value: f64, threshold: f64) -> bool {
        if self.count < 2 {
            return false;
        }
        self.z_score(value).abs() > threshold
    }
}

/// State tracker for market data
#[derive(Debug, Clone)]
pub struct MarketDataState {
    /// Last sequence number per exchange/symbol
    pub last_sequence: HashMap<(Exchange, Symbol), u64>,

    /// Last timestamp per exchange/symbol
    pub last_timestamp: HashMap<(Exchange, Symbol), i64>,

    /// Price statistics per symbol
    pub price_stats: HashMap<Symbol, RollingWindow>,

    /// Volume statistics per symbol
    pub volume_stats: HashMap<Symbol, RollingWindow>,
}

impl Default for MarketDataState {
    fn default() -> Self {
        Self::new()
    }
}

impl MarketDataState {
    /// Create new market data state
    pub fn new() -> Self {
        Self {
            last_sequence: HashMap::new(),
            last_timestamp: HashMap::new(),
            price_stats: HashMap::new(),
            volume_stats: HashMap::new(),
        }
    }

    /// Update sequence number
    pub fn update_sequence(&mut self, exchange: Exchange, symbol: Symbol, seq: u64) {
        self.last_sequence.insert((exchange, symbol), seq);
    }

    /// Update timestamp
    pub fn update_timestamp(&mut self, exchange: Exchange, symbol: Symbol, ts: i64) {
        self.last_timestamp.insert((exchange, symbol), ts);
    }

    /// Get last sequence number
    pub fn get_last_sequence(&self, exchange: Exchange, symbol: &Symbol) -> Option<u64> {
        self.last_sequence.get(&(exchange, symbol.clone())).copied()
    }

    /// Get last timestamp
    pub fn get_last_timestamp(&self, exchange: Exchange, symbol: &Symbol) -> Option<i64> {
        self.last_timestamp
            .get(&(exchange, symbol.clone()))
            .copied()
    }
}

/// Rolling window for statistics
#[derive(Debug, Clone)]
pub struct RollingWindow {
    /// Window data
    pub data: Vec<f64>,

    /// Maximum window size
    pub max_size: usize,

    /// Current index
    pub index: usize,

    /// Whether window is full
    pub is_full: bool,
}

impl RollingWindow {
    /// Create new rolling window
    pub fn new(max_size: usize) -> Self {
        Self {
            data: Vec::with_capacity(max_size),
            max_size,
            index: 0,
            is_full: false,
        }
    }

    /// Add value to window
    pub fn add(&mut self, value: f64) {
        if self.is_full {
            self.data[self.index] = value;
            self.index = (self.index + 1) % self.max_size;
        } else {
            self.data.push(value);
            if self.data.len() == self.max_size {
                self.is_full = true;
            }
        }
    }

    /// Calculate statistics
    pub fn statistics(&self) -> Statistics {
        if self.data.is_empty() {
            return Statistics::empty();
        }

        let count = self.data.len();
        let sum: f64 = self.data.iter().sum();
        let mean = sum / count as f64;

        let variance: f64 = self
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;

        let std_dev = variance.sqrt();
        let min = self.data.iter().copied().fold(f64::MAX, f64::min);
        let max = self.data.iter().copied().fold(f64::MIN, f64::max);

        Statistics {
            mean,
            std_dev,
            min,
            max,
            count,
        }
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Shared state for the data quality system
pub type SharedState = Arc<RwLock<MarketDataState>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_window() {
        let mut window = RollingWindow::new(5);

        // Add values
        window.add(1.0);
        window.add(2.0);
        window.add(3.0);
        window.add(4.0);
        window.add(5.0);

        assert_eq!(window.len(), 5);
        assert!(window.is_full);

        let stats = window.statistics();
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);

        // Add more values (should replace oldest)
        window.add(6.0);
        assert_eq!(window.len(), 5);
        assert_eq!(window.data, vec![6.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_statistics_z_score() {
        let stats = Statistics {
            mean: 100.0,
            std_dev: 10.0,
            min: 80.0,
            max: 120.0,
            count: 10,
        };

        assert_eq!(stats.z_score(100.0), 0.0);
        assert_eq!(stats.z_score(110.0), 1.0);
        assert_eq!(stats.z_score(90.0), -1.0);
        assert!(stats.is_outlier(135.0, 3.0));
        assert!(!stats.is_outlier(110.0, 3.0));
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult::success("price_validator");
        assert!(result.is_valid);
        assert!(result.errors.is_empty());

        let result = ValidationResult::failure("price_validator", "Price out of range");
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);

        let result = result.with_warning("Near upper limit");
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_market_data_state() {
        let mut state = MarketDataState::new();

        let exchange = Exchange::Coinbase;
        let symbol = Symbol::new("BTC", "USD");

        state.update_sequence(exchange, symbol.clone(), 100);
        state.update_timestamp(exchange, symbol.clone(), 123456789);

        assert_eq!(state.get_last_sequence(exchange, &symbol), Some(100));
        assert_eq!(state.get_last_timestamp(exchange, &symbol), Some(123456789));
    }
}
