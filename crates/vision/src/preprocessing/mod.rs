//! Preprocessing utilities for time-series financial data
//!
//! This module provides normalization, scaling, and preprocessing tools
//! for preparing OHLCV data for machine learning models.

pub mod features;
pub mod normalization;
pub mod tensor_conversion;

pub use features::{FeatureConfig, FeatureEngineer};
pub use normalization::{MinMaxScaler, RobustScaler, Scaler, ZScoreScaler};
pub use tensor_conversion::{BatchIterator, TensorConverter, TensorConverterConfig, create_batch};
