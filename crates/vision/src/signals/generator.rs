//! Signal generator that converts model outputs to trading signals.
//!
//! This module provides the main SignalGenerator that combines confidence
//! scoring with signal creation to produce actionable trading signals.

use super::confidence::{ConfidenceConfig, ConfidenceScorer};
use super::types::{SignalBatch, SignalType, TradingSignal};
use burn::tensor::{Tensor, backend::Backend};
use chrono::Utc;

/// Configuration for signal generation
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Confidence configuration
    pub confidence: ConfidenceConfig,

    /// Class index mapping to signal types [buy_idx, hold_idx, sell_idx]
    pub class_mapping: [usize; 3],

    /// Default position size suggestion (0.0 to 1.0)
    pub default_position_size: Option<f64>,

    /// Include class probabilities in signal metadata
    pub include_probabilities: bool,

    /// Generate Hold signals (if false, only Buy/Sell/Close)
    pub generate_hold_signals: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            confidence: ConfidenceConfig::default(),
            class_mapping: [0, 1, 2], // buy=0, hold=1, sell=2
            default_position_size: None,
            include_probabilities: true,
            generate_hold_signals: false,
        }
    }
}

impl GeneratorConfig {
    /// Create a config with custom confidence threshold
    pub fn with_threshold(min_confidence: f64) -> Self {
        Self {
            confidence: ConfidenceConfig::with_threshold(min_confidence),
            ..Default::default()
        }
    }

    /// Create a config with default position sizing
    pub fn with_position_size(size: f64) -> Self {
        Self {
            default_position_size: Some(size.clamp(0.0, 1.0)),
            ..Default::default()
        }
    }

    /// Set custom class mapping (useful if model has different class order)
    pub fn with_class_mapping(mut self, buy: usize, hold: usize, sell: usize) -> Self {
        self.class_mapping = [buy, hold, sell];
        self
    }
}

/// Signal generator that converts model predictions to trading signals
pub struct SignalGenerator {
    config: GeneratorConfig,
    scorer: ConfidenceScorer,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: GeneratorConfig) -> Self {
        let scorer = ConfidenceScorer::new(config.confidence.clone());
        Self { config, scorer }
    }

    /// Generate signals from model logits for a batch of assets
    ///
    /// # Arguments
    /// * `logits` - Model output logits [batch_size, num_classes]
    /// * `assets` - Asset identifiers corresponding to each sample
    /// * `device` - Device where tensors are located
    ///
    /// # Returns
    /// SignalBatch containing generated signals
    pub fn generate_batch<B: Backend>(
        &self,
        logits: &Tensor<B, 2>,
        assets: &[&str],
    ) -> SignalBatch {
        let confidences = self.scorer.from_logits(logits);
        let predicted_classes = self.scorer.get_predicted_classes(logits);

        let probabilities = if self.config.include_probabilities {
            Some(self.scorer.get_class_probabilities(logits))
        } else {
            None
        };

        let mut signals = Vec::with_capacity(assets.len());
        let timestamp = Utc::now();

        for (i, asset) in assets.iter().enumerate() {
            let class_idx = predicted_classes[i];
            let confidence = confidences[i];

            let signal_type = self.class_to_signal_type(class_idx);

            // Skip Hold signals if not configured to generate them
            if signal_type == SignalType::Hold && !self.config.generate_hold_signals {
                continue;
            }

            let mut signal = TradingSignal::with_timestamp(
                signal_type,
                confidence,
                asset.to_string(),
                timestamp,
            );

            // Add position size if configured
            if let Some(size) = self.config.default_position_size {
                signal = signal.with_size(size);
            }

            // Add class probabilities if configured
            if let Some(ref probs) = probabilities {
                signal = signal.with_probabilities(probs[i].clone());
            }

            signals.push(signal);
        }

        SignalBatch::new(signals)
    }

    /// Generate a single signal from model logits
    ///
    /// # Arguments
    /// * `logits` - Model output logits [1, num_classes]
    /// * `asset` - Asset identifier
    ///
    /// # Returns
    /// TradingSignal for the asset
    pub fn generate_single<B: Backend>(&self, logits: &Tensor<B, 2>, asset: &str) -> TradingSignal {
        let batch = self.generate_batch(logits, &[asset]);
        batch.signals.into_iter().next().unwrap_or_else(|| {
            // Fallback if Hold signal was filtered out
            TradingSignal::new(SignalType::Hold, 0.0, asset.to_string())
        })
    }

    /// Generate signals from pre-computed probabilities
    ///
    /// # Arguments
    /// * `probabilities` - Softmax probabilities [batch_size, num_classes]
    /// * `assets` - Asset identifiers
    ///
    /// # Returns
    /// SignalBatch containing generated signals
    pub fn generate_from_probabilities<B: Backend>(
        &self,
        probabilities: &Tensor<B, 2>,
        assets: &[&str],
    ) -> SignalBatch {
        let confidences = self.scorer.from_probabilities(probabilities);
        let predicted_classes = self.scorer.get_predicted_classes(probabilities);

        let probs_all = if self.config.include_probabilities {
            Some(self.scorer.get_class_probabilities(probabilities))
        } else {
            None
        };

        let mut signals = Vec::with_capacity(assets.len());
        let timestamp = Utc::now();

        for (i, asset) in assets.iter().enumerate() {
            let class_idx = predicted_classes[i];
            let confidence = confidences[i];

            let signal_type = self.class_to_signal_type(class_idx);

            if signal_type == SignalType::Hold && !self.config.generate_hold_signals {
                continue;
            }

            let mut signal = TradingSignal::with_timestamp(
                signal_type,
                confidence,
                asset.to_string(),
                timestamp,
            );

            if let Some(size) = self.config.default_position_size {
                signal = signal.with_size(size);
            }

            if let Some(ref probs) = probs_all {
                signal = signal.with_probabilities(probs[i].clone());
            }

            signals.push(signal);
        }

        SignalBatch::new(signals)
    }

    /// Convert class index to signal type using configured mapping
    fn class_to_signal_type(&self, class_idx: usize) -> SignalType {
        if class_idx == self.config.class_mapping[0] {
            SignalType::Buy
        } else if class_idx == self.config.class_mapping[1] {
            SignalType::Hold
        } else if class_idx == self.config.class_mapping[2] {
            SignalType::Sell
        } else {
            // Fallback for unexpected class indices
            SignalType::Hold
        }
    }

    /// Get the confidence threshold
    pub fn threshold(&self) -> f64 {
        self.config.confidence.min_threshold
    }

    /// Check if a signal would meet the threshold
    pub fn meets_threshold(&self, confidence: f64) -> bool {
        self.scorer.meets_threshold(confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{Shape, TensorData};

    type TestBackend = NdArray;

    fn create_test_logits(data: Vec<f32>) -> Tensor<TestBackend, 2> {
        let batch_size = data.len() / 3;
        let shape = Shape::new([batch_size, 3]);
        let tensor_data = TensorData::new(data, shape);
        let device = Default::default();
        Tensor::<TestBackend, 2>::from_data(tensor_data.convert::<f32>(), &device)
    }

    #[test]
    fn test_generator_config_default() {
        let config = GeneratorConfig::default();
        assert_eq!(config.confidence.min_threshold, 0.7);
        assert_eq!(config.class_mapping, [0, 1, 2]);
        assert!(!config.generate_hold_signals);
    }

    #[test]
    fn test_generator_config_builders() {
        let config = GeneratorConfig::with_threshold(0.8);
        assert_eq!(config.confidence.min_threshold, 0.8);

        let config = GeneratorConfig::with_position_size(0.25);
        assert_eq!(config.default_position_size, Some(0.25));

        let config = GeneratorConfig::default().with_class_mapping(2, 1, 0);
        assert_eq!(config.class_mapping, [2, 1, 0]);
    }

    #[test]
    fn test_generate_batch() {
        let config = GeneratorConfig::default();
        let generator = SignalGenerator::new(config);

        // Create logits: batch_size=2, num_classes=3
        // First sample: high logit for class 0 (Buy)
        // Second sample: high logit for class 2 (Sell)
        let logits = create_test_logits(vec![3.0, 1.0, 0.5, 0.5, 1.0, 3.0]);
        let assets = vec!["BTCUSD", "ETHUSDT"];

        let batch = generator.generate_batch(&logits, &assets);

        assert_eq!(batch.signals.len(), 2);
        assert_eq!(batch.signals[0].signal_type, SignalType::Buy);
        assert_eq!(batch.signals[0].asset, "BTCUSD");
        assert_eq!(batch.signals[1].signal_type, SignalType::Sell);
        assert_eq!(batch.signals[1].asset, "ETHUSDT");
    }

    #[test]
    fn test_generate_single() {
        let config = GeneratorConfig::default();
        let generator = SignalGenerator::new(config);

        let logits = create_test_logits(vec![3.0, 1.0, 0.5]);
        let signal = generator.generate_single(&logits, "BTCUSD");

        assert_eq!(signal.signal_type, SignalType::Buy);
        assert_eq!(signal.asset, "BTCUSD");
        assert!(signal.confidence > 0.5);
    }

    #[test]
    fn test_generate_with_position_size() {
        let config = GeneratorConfig::with_position_size(0.25);
        let generator = SignalGenerator::new(config);

        let logits = create_test_logits(vec![3.0, 1.0, 0.5]);
        let signal = generator.generate_single(&logits, "BTCUSD");

        assert_eq!(signal.suggested_size, Some(0.25));
    }

    #[test]
    fn test_generate_with_probabilities() {
        let mut config = GeneratorConfig::default();
        config.include_probabilities = true;
        let generator = SignalGenerator::new(config);

        let logits = create_test_logits(vec![3.0, 1.0, 0.5]);
        let signal = generator.generate_single(&logits, "BTCUSD");

        assert!(signal.class_probabilities.is_some());
        let probs = signal.class_probabilities.unwrap();
        assert_eq!(probs.len(), 3);

        // Probabilities should sum to ~1.0
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_generate_hold_signals() {
        let mut config = GeneratorConfig::default();
        config.generate_hold_signals = true;
        let generator = SignalGenerator::new(config);

        // Create logit with high value for class 1 (Hold)
        let logits = create_test_logits(vec![0.5, 3.0, 0.5]);
        let signal = generator.generate_single(&logits, "BTCUSD");

        assert_eq!(signal.signal_type, SignalType::Hold);
    }

    #[test]
    fn test_skip_hold_signals() {
        let mut config = GeneratorConfig::default();
        config.generate_hold_signals = false;
        let generator = SignalGenerator::new(config);

        // Both samples predict Hold (class 1)
        let logits = create_test_logits(vec![0.5, 3.0, 0.5, 0.5, 3.0, 0.5]);
        let assets = vec!["BTCUSD", "ETHUSDT"];

        let batch = generator.generate_batch(&logits, &assets);

        // Should have no signals since Hold signals are skipped
        assert_eq!(batch.signals.len(), 0);
    }

    #[test]
    fn test_custom_class_mapping() {
        let mut config = GeneratorConfig::default();
        // Swap Buy and Sell: sell=0, hold=1, buy=2
        config.class_mapping = [2, 1, 0];
        let generator = SignalGenerator::new(config);

        // High logit for class 0 (now mapped to Sell)
        let logits = create_test_logits(vec![3.0, 1.0, 0.5]);
        let signal = generator.generate_single(&logits, "BTCUSD");

        assert_eq!(signal.signal_type, SignalType::Sell);
    }

    #[test]
    fn test_meets_threshold() {
        let config = GeneratorConfig::with_threshold(0.8);
        let generator = SignalGenerator::new(config);

        assert!(generator.meets_threshold(0.85));
        assert!(generator.meets_threshold(0.8));
        assert!(!generator.meets_threshold(0.75));
    }

    #[test]
    fn test_threshold_getter() {
        let config = GeneratorConfig::with_threshold(0.75);
        let generator = SignalGenerator::new(config);

        assert_eq!(generator.threshold(), 0.75);
    }

    #[test]
    fn test_class_to_signal_type() {
        let config = GeneratorConfig::default();
        let generator = SignalGenerator::new(config);

        assert_eq!(generator.class_to_signal_type(0), SignalType::Buy);
        assert_eq!(generator.class_to_signal_type(1), SignalType::Hold);
        assert_eq!(generator.class_to_signal_type(2), SignalType::Sell);

        // Out of range returns Hold as fallback
        assert_eq!(generator.class_to_signal_type(99), SignalType::Hold);
    }

    #[test]
    fn test_generate_from_probabilities() {
        let config = GeneratorConfig::default();
        let generator = SignalGenerator::new(config);

        // Create probability tensor (already softmaxed)
        // First sample: 70% Buy, 20% Hold, 10% Sell
        // Second sample: 10% Buy, 20% Hold, 70% Sell
        let probs_data = vec![0.7f32, 0.2, 0.1, 0.1, 0.2, 0.7];
        let batch_size = 2;
        let shape = Shape::new([batch_size, 3]);
        let tensor_data = TensorData::new(probs_data, shape);
        let device = Default::default();
        let probs = Tensor::<TestBackend, 2>::from_data(tensor_data.convert::<f32>(), &device);

        let assets = vec!["BTCUSD", "ETHUSDT"];
        let batch = generator.generate_from_probabilities(&probs, &assets);

        assert_eq!(batch.signals.len(), 2);
        assert_eq!(batch.signals[0].signal_type, SignalType::Buy);
        assert_eq!(batch.signals[1].signal_type, SignalType::Sell);
    }
}
