//! Signal generation module for trading decisions.
//!
//! This module provides utilities to convert model predictions into
//! actionable trading signals with confidence scoring and filtering.
//!
//! # Overview
//!
//! The signal generation pipeline consists of:
//! 1. **Model Prediction** → Raw logits/probabilities
//! 2. **Confidence Scoring** → Calibrated confidence values
//! 3. **Signal Generation** → TradingSignal objects
//! 4. **Filtering** → Apply risk and validation rules
//!
//! # Example
//!
//! ```rust,ignore
//! use vision::signals::*;
//!
//! // 1. Get model predictions
//! let logits = model.forward(inputs);
//!
//! // 2. Score confidence
//! let scorer = ConfidenceScorer::new(ConfidenceConfig::default());
//! let confidences = scorer.from_logits(&logits);
//! let classes = scorer.get_predicted_classes(&logits);
//!
//! // 3. Generate signals
//! let generator = SignalGenerator::new(GeneratorConfig::default());
//! let signals = generator.generate_batch(&logits, &["BTCUSD"], &device)?;
//!
//! // 4. Filter signals
//! let filter = SignalFilter::new(FilterConfig::default());
//! let passed = filter.get_passed(&signals);
//! ```

pub mod confidence;
pub mod filters;
pub mod generator;
pub mod integration;
pub mod types;

// Re-exports for convenience
pub use confidence::{ConfidenceConfig, ConfidenceScorer, ConfidenceStats};
pub use filters::{FilterConfig, FilterReason, FilterResult, FilterStats, SignalFilter};
pub use generator::{GeneratorConfig, SignalGenerator};
pub use integration::{BatchProcessor, PipelineBuilder, SignalPipeline};
pub use types::{SignalBatch, SignalStrength, SignalType, TradingSignal};
