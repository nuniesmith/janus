//! Data quality pipeline orchestrator
//!
//! This module provides the main pipeline for processing market data through
//! validators and anomaly detectors, producing quality metrics and actionable results.

use std::sync::Arc;
use tokio::sync::RwLock;

use janus_core::MarketDataEvent;

use crate::{
    Config, SharedState, ValidationResult,
    anomaly::{AnomalyDetector, AnomalyResult},
    error::Result,
    validators::Validator,
};

/// Processing result from the data quality pipeline
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Validation results from all validators
    pub validation_results: Vec<ValidationResult>,

    /// Anomaly detection results
    pub anomaly_results: Vec<AnomalyResult>,

    /// Overall quality score (0.0 = all failed, 1.0 = perfect)
    pub quality_score: f64,

    /// Whether the event passed all critical checks
    pub is_valid: bool,

    /// Whether any anomalies were detected
    pub has_anomalies: bool,

    /// Processing latency in microseconds
    pub processing_latency_us: u64,
}

impl ProcessingResult {
    /// Create a new processing result
    pub fn new(
        validation_results: Vec<ValidationResult>,
        anomaly_results: Vec<AnomalyResult>,
        processing_latency_us: u64,
    ) -> Self {
        let is_valid = validation_results.iter().all(|r| r.is_valid);
        let has_anomalies = anomaly_results.iter().any(|r| r.is_anomaly);

        // Calculate quality score
        let total_checks = validation_results.len();
        let passed_checks = validation_results.iter().filter(|r| r.is_valid).count();

        let validation_score = if total_checks > 0 {
            passed_checks as f64 / total_checks as f64
        } else {
            1.0
        };

        // Reduce score if anomalies detected
        let anomaly_penalty = if has_anomalies {
            let high_severity_count = anomaly_results
                .iter()
                .filter(|r| r.is_anomaly && r.severity == crate::anomaly::AnomalySeverity::High)
                .count();
            let medium_severity_count = anomaly_results
                .iter()
                .filter(|r| r.is_anomaly && r.severity == crate::anomaly::AnomalySeverity::Medium)
                .count();

            // High severity: -0.3, Medium: -0.15, Low: -0.05
            (high_severity_count as f64 * 0.3)
                + (medium_severity_count as f64 * 0.15)
                + ((anomaly_results.len() - high_severity_count - medium_severity_count) as f64
                    * 0.05)
        } else {
            0.0
        };

        let quality_score = (validation_score - anomaly_penalty).clamp(0.0, 1.0);

        Self {
            validation_results,
            anomaly_results,
            quality_score,
            is_valid,
            has_anomalies,
            processing_latency_us,
        }
    }

    /// Get all errors from validation results
    pub fn errors(&self) -> Vec<String> {
        self.validation_results
            .iter()
            .flat_map(|r| r.errors.iter().cloned())
            .collect()
    }

    /// Get all warnings from validation results
    pub fn warnings(&self) -> Vec<String> {
        self.validation_results
            .iter()
            .flat_map(|r| r.warnings.iter().cloned())
            .collect()
    }

    /// Get all detected anomalies
    pub fn anomalies(&self) -> Vec<&AnomalyResult> {
        self.anomaly_results
            .iter()
            .filter(|r| r.is_anomaly)
            .collect()
    }

    /// Check if processing should be retried
    pub fn should_retry(&self) -> bool {
        // Retry if we have validation failures but no anomalies
        // (anomalies might indicate genuine market conditions)
        !self.is_valid && !self.has_anomalies
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Quality: {:.1}%, Valid: {}, Anomalies: {}, Errors: {}, Warnings: {}, Latency: {}µs",
            self.quality_score * 100.0,
            self.is_valid,
            self.anomalies().len(),
            self.errors().len(),
            self.warnings().len(),
            self.processing_latency_us
        )
    }
}

/// Data quality pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable validation phase
    pub enable_validation: bool,

    /// Enable anomaly detection phase
    pub enable_anomaly_detection: bool,

    /// Stop processing on first validation failure
    pub fail_fast: bool,

    /// Maximum processing time per event (microseconds)
    pub max_processing_time_us: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            enable_anomaly_detection: true,
            fail_fast: false,
            max_processing_time_us: 10_000, // 10ms
        }
    }
}

/// Main data quality pipeline
pub struct DataQualityPipeline {
    /// Pipeline configuration
    config: PipelineConfig,

    /// Validators to run
    validators: Vec<Arc<dyn Validator>>,

    /// Anomaly detectors to run
    detectors: Vec<Arc<RwLock<dyn AnomalyDetector>>>,

    /// Shared state for tracking
    #[allow(dead_code)]
    state: SharedState,
}

impl DataQualityPipeline {
    /// Create a new data quality pipeline
    pub fn new(config: PipelineConfig, state: SharedState) -> Self {
        Self {
            config,
            validators: Vec::new(),
            detectors: Vec::new(),
            state,
        }
    }

    /// Create a pipeline from full config
    pub fn from_config(config: Config, state: SharedState) -> Self {
        let mut pipeline = Self::new(
            PipelineConfig {
                enable_validation: true,
                enable_anomaly_detection: true,
                fail_fast: false,
                max_processing_time_us: 10_000,
            },
            state.clone(),
        );

        // Add validators
        pipeline.add_validator(Arc::new(crate::validators::PriceValidator::new(
            config.validator.price,
            state.clone(),
        )));
        pipeline.add_validator(Arc::new(crate::validators::VolumeValidator::new(
            config.validator.volume,
            state.clone(),
        )));
        pipeline.add_validator(Arc::new(crate::validators::TimestampValidator::new(
            config.validator.timestamp,
            state.clone(),
        )));
        pipeline.add_validator(Arc::new(crate::validators::OrderBookValidator::new(
            config.validator.orderbook,
            state.clone(),
        )));

        // Add anomaly detectors
        if config.anomaly.statistical_enabled {
            pipeline.add_detector(Arc::new(RwLock::new(
                crate::anomaly::StatisticalAnomalyDetector::new(
                    config.anomaly.z_score_threshold,
                    config.anomaly.window_size,
                ),
            )));
        }

        if config.anomaly.latency_enabled {
            pipeline.add_detector(Arc::new(RwLock::new(
                crate::anomaly::LatencyAnomalyDetector::new(config.anomaly.max_latency_us),
            )));
        }

        if config.anomaly.sequence_enabled {
            pipeline.add_detector(Arc::new(RwLock::new(
                crate::anomaly::SequenceAnomalyDetector::new(config.anomaly.max_sequence_gap),
            )));
        }

        pipeline
    }

    /// Add a validator to the pipeline
    pub fn add_validator(&mut self, validator: Arc<dyn Validator>) {
        self.validators.push(validator);
    }

    /// Add an anomaly detector to the pipeline
    pub fn add_detector(&mut self, detector: Arc<RwLock<dyn AnomalyDetector>>) {
        self.detectors.push(detector);
    }

    /// Process a market data event through the pipeline
    pub async fn process(&self, event: &MarketDataEvent) -> Result<ProcessingResult> {
        let start = std::time::Instant::now();

        // Phase 1: Validation
        let validation_results = if self.config.enable_validation {
            self.validate_event(event).await?
        } else {
            Vec::new()
        };

        // Check if we should continue
        if self.config.fail_fast && validation_results.iter().any(|r| !r.is_valid) {
            let elapsed = start.elapsed().as_micros() as u64;
            return Ok(ProcessingResult::new(
                validation_results,
                Vec::new(),
                elapsed,
            ));
        }

        // Phase 2: Anomaly Detection
        let anomaly_results = if self.config.enable_anomaly_detection {
            self.detect_anomalies(event).await?
        } else {
            Vec::new()
        };

        let elapsed = start.elapsed().as_micros() as u64;

        // Check processing time
        if elapsed > self.config.max_processing_time_us {
            tracing::warn!(
                "Processing took {}µs (max: {}µs)",
                elapsed,
                self.config.max_processing_time_us
            );
        }

        Ok(ProcessingResult::new(
            validation_results,
            anomaly_results,
            elapsed,
        ))
    }

    /// Validate an event using all validators
    async fn validate_event(&self, event: &MarketDataEvent) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::with_capacity(self.validators.len());

        for validator in &self.validators {
            let result = validator.validate(event).await?;
            results.push(result);

            // Fail fast if configured
            if self.config.fail_fast && !results.last().unwrap().is_valid {
                break;
            }
        }

        Ok(results)
    }

    /// Detect anomalies using all detectors
    async fn detect_anomalies(&self, event: &MarketDataEvent) -> Result<Vec<AnomalyResult>> {
        let mut results = Vec::with_capacity(self.detectors.len());

        for detector in &self.detectors {
            let mut detector = detector.write().await;
            let result = detector.detect(event)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Reset all detectors (useful for testing or state cleanup)
    pub async fn reset_detectors(&self) {
        for detector in &self.detectors {
            let mut detector = detector.write().await;
            detector.reset();
        }
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            validator_count: self.validators.len(),
            detector_count: self.detectors.len(),
            validation_enabled: self.config.enable_validation,
            anomaly_detection_enabled: self.config.enable_anomaly_detection,
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Number of active validators
    pub validator_count: usize,

    /// Number of active detectors
    pub detector_count: usize,

    /// Whether validation is enabled
    pub validation_enabled: bool,

    /// Whether anomaly detection is enabled
    pub anomaly_detection_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MarketDataState, anomaly::AnomalySeverity};
    use janus_core::{Exchange, Side, Symbol, TradeEvent};
    use rust_decimal::Decimal;

    fn create_test_state() -> SharedState {
        Arc::new(RwLock::new(MarketDataState::new()))
    }

    fn create_test_trade() -> MarketDataEvent {
        MarketDataEvent::Trade(TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::new(50000, 0),
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: chrono::Utc::now().timestamp_micros(),
            received_at: chrono::Utc::now().timestamp_micros(),
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        })
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let state = create_test_state();
        let config = Config::default();
        let pipeline = DataQualityPipeline::from_config(config, state);

        let stats = pipeline.stats();
        assert_eq!(stats.validator_count, 4); // price, volume, timestamp, orderbook
        assert!(stats.validation_enabled);
        assert!(stats.anomaly_detection_enabled);
    }

    #[tokio::test]
    async fn test_process_valid_event() {
        let state = create_test_state();
        let config = Config::default();
        let pipeline = DataQualityPipeline::from_config(config, state);

        let event = create_test_trade();
        let result = pipeline.process(&event).await.unwrap();

        assert!(result.is_valid);
        assert_eq!(result.errors().len(), 0);
        assert!(result.quality_score > 0.9);
    }

    #[tokio::test]
    async fn test_process_invalid_event() {
        let state = create_test_state();
        let config = Config::default();
        let pipeline = DataQualityPipeline::from_config(config, state);

        // Create invalid trade (zero price)
        let event = MarketDataEvent::Trade(TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::ZERO,
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: chrono::Utc::now().timestamp_micros(),
            received_at: chrono::Utc::now().timestamp_micros(),
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        });

        let result = pipeline.process(&event).await.unwrap();

        assert!(!result.is_valid);
        assert!(!result.errors().is_empty());
        assert!(result.quality_score < 1.0);
    }

    #[tokio::test]
    async fn test_processing_result_quality_score() {
        let valid = ValidationResult::success("test");
        let invalid = ValidationResult::failure("test", "error");

        let result = ProcessingResult::new(vec![valid.clone(), valid.clone()], vec![], 100);
        assert_eq!(result.quality_score, 1.0);

        let result = ProcessingResult::new(vec![valid.clone(), invalid.clone()], vec![], 100);
        assert_eq!(result.quality_score, 0.5);

        let result = ProcessingResult::new(vec![invalid.clone(), invalid.clone()], vec![], 100);
        assert_eq!(result.quality_score, 0.0);
    }

    #[tokio::test]
    async fn test_processing_result_with_anomalies() {
        let valid = ValidationResult::success("test");
        let anomaly = AnomalyResult::detected(
            "test",
            Symbol::new("BTC", "USD"),
            AnomalySeverity::High,
            "test anomaly",
        );

        let result = ProcessingResult::new(vec![valid], vec![anomaly], 100);
        assert!(result.has_anomalies);
        assert_eq!(result.anomalies().len(), 1);
        assert!(result.quality_score < 1.0); // Penalized for anomaly
    }

    #[tokio::test]
    async fn test_fail_fast_mode() {
        let state = create_test_state();
        let mut pipeline = DataQualityPipeline::new(
            PipelineConfig {
                enable_validation: true,
                enable_anomaly_detection: true,
                fail_fast: true,
                max_processing_time_us: 10_000,
            },
            state.clone(),
        );

        pipeline.add_validator(Arc::new(crate::validators::PriceValidator::new(
            crate::validators::PriceValidatorConfig::default(),
            state,
        )));

        // Invalid event
        let event = MarketDataEvent::Trade(TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::ZERO,
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: chrono::Utc::now().timestamp_micros(),
            received_at: chrono::Utc::now().timestamp_micros(),
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        });

        let result = pipeline.process(&event).await.unwrap();
        assert!(!result.is_valid);
        // In fail-fast mode, we should have stopped at first failure
        assert_eq!(result.validation_results.len(), 1);
    }

    #[tokio::test]
    async fn test_processing_result_summary() {
        let valid = ValidationResult::success("test");
        let result = ProcessingResult::new(vec![valid], vec![], 150);

        let summary = result.summary();
        assert!(summary.contains("Quality: 100"));
        assert!(summary.contains("Valid: true"));
        assert!(summary.contains("150µs"));
    }
}
