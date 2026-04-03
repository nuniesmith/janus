//! Data validators for market data quality checks
//!
//! This module provides validators for different types of market data:
//! - Price validation (range checks, spike detection)
//! - Volume validation (non-negative, reasonable magnitude)
//! - Timestamp validation (monotonicity, clock skew)
//! - Order book validation (bid < ask, valid levels)

use async_trait::async_trait;
use janus_core::{KlineEvent, MarketDataEvent, OrderBookEvent, TradeEvent};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::{Result, ValidationResult};

pub mod orderbook;
pub mod price;
pub mod timestamp;
pub mod volume;

pub use orderbook::OrderBookValidator;
pub use price::PriceValidator;
pub use timestamp::TimestampValidator;
pub use volume::VolumeValidator;

/// Configuration for all validators
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidatorConfig {
    /// Price validator configuration
    pub price: PriceValidatorConfig,

    /// Volume validator configuration
    pub volume: VolumeValidatorConfig,

    /// Timestamp validator configuration
    pub timestamp: TimestampValidatorConfig,

    /// Order book validator configuration
    pub orderbook: OrderBookValidatorConfig,
}

/// Price validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceValidatorConfig {
    /// Enable price validation
    pub enabled: bool,

    /// Minimum allowed price (absolute)
    pub min_price: Option<Decimal>,

    /// Maximum allowed price (absolute)
    pub max_price: Option<Decimal>,

    /// Enable spike detection
    pub spike_detection_enabled: bool,

    /// Maximum allowed price change percentage (0.0 - 1.0)
    pub max_price_change_pct: f64,

    /// Number of decimals to validate
    pub max_decimals: u32,

    /// Enable zero price check
    pub reject_zero_prices: bool,

    /// Enable negative price check
    pub reject_negative_prices: bool,
}

impl Default for PriceValidatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_price: Some(Decimal::new(1, 8)), // 0.00000001
            max_price: Some(Decimal::new(10_000_000, 0)), // 10M
            spike_detection_enabled: true,
            max_price_change_pct: 0.20, // 20%
            max_decimals: 8,
            reject_zero_prices: true,
            reject_negative_prices: true,
        }
    }
}

/// Volume validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeValidatorConfig {
    /// Enable volume validation
    pub enabled: bool,

    /// Minimum allowed volume
    pub min_volume: Option<Decimal>,

    /// Maximum allowed volume
    pub max_volume: Option<Decimal>,

    /// Enable zero volume check
    pub reject_zero_volumes: bool,

    /// Enable negative volume check
    pub reject_negative_volumes: bool,

    /// Maximum volume spike ratio (vs recent average)
    pub max_volume_spike_ratio: f64,
}

impl Default for VolumeValidatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_volume: Some(Decimal::new(1, 10)), // Very small but non-zero
            max_volume: Some(Decimal::new(1_000_000_000, 0)), // 1B
            reject_zero_volumes: false,            // Some exchanges send zero-volume ticks
            reject_negative_volumes: true,
            max_volume_spike_ratio: 100.0, // 100x recent average
        }
    }
}

/// Timestamp validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampValidatorConfig {
    /// Enable timestamp validation
    pub enabled: bool,

    /// Enable monotonicity check (timestamps should increase)
    pub check_monotonicity: bool,

    /// Maximum allowed clock skew (microseconds)
    pub max_clock_skew_us: i64,

    /// Maximum allowed future timestamp (microseconds ahead)
    pub max_future_us: i64,

    /// Maximum allowed past timestamp (microseconds behind)
    pub max_past_us: i64,

    /// Enable duplicate timestamp detection
    pub detect_duplicates: bool,
}

impl Default for TimestampValidatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_monotonicity: true,
            max_clock_skew_us: 10_000_000, // 10 seconds
            max_future_us: 5_000_000,      // 5 seconds
            max_past_us: 300_000_000,      // 5 minutes
            detect_duplicates: true,
        }
    }
}

/// Order book validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookValidatorConfig {
    /// Enable order book validation
    pub enabled: bool,

    /// Check that bid < ask
    pub check_bid_ask_spread: bool,

    /// Minimum allowed spread (percentage)
    pub min_spread_pct: f64,

    /// Maximum allowed spread (percentage)
    pub max_spread_pct: f64,

    /// Check for overlapping levels
    pub check_overlapping_levels: bool,

    /// Check for positive sizes
    pub check_positive_sizes: bool,

    /// Check for sorted levels
    pub check_sorted_levels: bool,

    /// Maximum number of levels to validate
    pub max_levels: usize,
}

impl Default for OrderBookValidatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_bid_ask_spread: true,
            min_spread_pct: 0.0001, // 0.01% (1 bps)
            max_spread_pct: 0.05,   // 5%
            check_overlapping_levels: true,
            check_positive_sizes: true,
            check_sorted_levels: true,
            max_levels: 20,
        }
    }
}

/// Trait for data validators
#[async_trait]
pub trait Validator: Send + Sync {
    /// Name of the validator
    fn name(&self) -> &str;

    /// Validate a market data event
    async fn validate(&self, event: &MarketDataEvent) -> Result<ValidationResult>;

    /// Validate a trade event
    async fn validate_trade(&self, _trade: &TradeEvent) -> Result<ValidationResult> {
        Ok(ValidationResult::success(self.name()))
    }

    /// Validate a ticker event
    async fn validate_ticker(&self, _ticker: &janus_core::TickerEvent) -> Result<ValidationResult> {
        Ok(ValidationResult::success(self.name()))
    }

    /// Validate a kline event
    async fn validate_kline(&self, _kline: &KlineEvent) -> Result<ValidationResult> {
        Ok(ValidationResult::success(self.name()))
    }

    /// Validate an order book event
    async fn validate_orderbook(&self, _orderbook: &OrderBookEvent) -> Result<ValidationResult> {
        Ok(ValidationResult::success(self.name()))
    }
}

/// Composite validator that runs multiple validators
pub struct CompositeValidator {
    validators: Vec<Box<dyn Validator>>,
}

impl CompositeValidator {
    /// Create a new composite validator
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator
    pub fn add_validator(mut self, validator: Box<dyn Validator>) -> Self {
        self.validators.push(validator);
        self
    }

    /// Validate an event with all validators
    pub async fn validate_all(&self, event: &MarketDataEvent) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::with_capacity(self.validators.len());

        for validator in &self.validators {
            let result = validator.validate(event).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Check if all validations passed
    pub fn all_valid(results: &[ValidationResult]) -> bool {
        results.iter().all(|r| r.is_valid)
    }

    /// Get all errors from validation results
    pub fn collect_errors(results: &[ValidationResult]) -> Vec<String> {
        results
            .iter()
            .flat_map(|r| r.errors.iter().cloned())
            .collect()
    }

    /// Get all warnings from validation results
    pub fn collect_warnings(results: &[ValidationResult]) -> Vec<String> {
        results
            .iter()
            .flat_map(|r| r.warnings.iter().cloned())
            .collect()
    }
}

impl Default for CompositeValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_composite_validator() {
        let composite = CompositeValidator::new();
        assert_eq!(composite.validators.len(), 0);
    }

    #[test]
    fn test_validator_config_defaults() {
        let config = ValidatorConfig::default();
        assert!(config.price.enabled);
        assert!(config.volume.enabled);
        assert!(config.timestamp.enabled);
        assert!(config.orderbook.enabled);
    }

    #[test]
    fn test_price_validator_config() {
        let config = PriceValidatorConfig::default();
        assert!(config.spike_detection_enabled);
        assert!(config.reject_zero_prices);
        assert!(config.reject_negative_prices);
        assert_eq!(config.max_price_change_pct, 0.20);
    }

    #[test]
    fn test_validation_result_helpers() {
        let results = vec![
            ValidationResult::success("validator1"),
            ValidationResult::failure("validator2", "error1"),
            ValidationResult::success("validator3").with_warning("warning1"),
        ];

        assert!(!CompositeValidator::all_valid(&results));

        let errors = CompositeValidator::collect_errors(&results);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0], "error1");

        let warnings = CompositeValidator::collect_warnings(&results);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0], "warning1");
    }
}
