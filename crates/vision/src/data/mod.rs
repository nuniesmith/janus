//! Real market data loading and preprocessing for vision models.
//!
//! This module provides utilities for:
//! - Loading OHLCV data from CSV files
//! - Creating time-series sequences for DiffGAF input
//! - Data validation and quality checks
//! - Train/validation/test splitting
//!
//! # Example
//!
//! ```no_run
//! use vision::data::{CsvLoader, LoaderConfig};
//!
//! let loader = CsvLoader::new(LoaderConfig::default());
//! let candles = loader.load_csv("data/BTCUSDT_1h.csv").unwrap();
//! println!("Loaded {} candles", candles.len());
//! ```

pub mod csv_loader;
pub mod dataset;
pub mod validation;

pub use csv_loader::{CsvLoader, LoaderConfig, OhlcvCandle, load_ohlcv_csv};
pub use dataset::{OhlcvDataset, SequenceConfig, TrainValSplit};
pub use validation::{ValidationError, ValidationReport, validate_ohlcv};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all public types are accessible
        let _config = LoaderConfig::default();
        let _seq_config = SequenceConfig::default();
        let _split = TrainValSplit::default();
    }
}
