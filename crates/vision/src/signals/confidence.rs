//! Confidence calibration and scoring for trading signals.
//!
//! This module provides utilities to convert raw model outputs (logits/probabilities)
//! into calibrated confidence scores for trading signals.

use burn::tensor::{ElementConversion, Tensor, backend::Backend};

/// Configuration for confidence calibration
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Minimum confidence threshold for actionable signals
    pub min_threshold: f64,

    /// Temperature for softmax calibration (default: 1.0)
    pub temperature: f64,

    /// Whether to apply Platt scaling
    pub use_platt_scaling: bool,

    /// Platt scaling parameters (A, B) where calibrated_prob = 1 / (1 + exp(A * score + B))
    pub platt_params: Option<(f64, f64)>,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            min_threshold: 0.7,
            temperature: 1.0,
            use_platt_scaling: false,
            platt_params: None,
        }
    }
}

impl ConfidenceConfig {
    /// Create a config with custom minimum threshold
    pub fn with_threshold(min_threshold: f64) -> Self {
        Self {
            min_threshold,
            ..Default::default()
        }
    }

    /// Create a config with temperature scaling
    pub fn with_temperature(temperature: f64) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }

    /// Enable Platt scaling with fitted parameters
    pub fn with_platt_scaling(mut self, a: f64, b: f64) -> Self {
        self.use_platt_scaling = true;
        self.platt_params = Some((a, b));
        self
    }
}

/// Confidence scorer for model outputs
pub struct ConfidenceScorer {
    config: ConfidenceConfig,
}

impl ConfidenceScorer {
    /// Create a new confidence scorer
    pub fn new(config: ConfidenceConfig) -> Self {
        Self { config }
    }

    /// Extract confidence from logits tensor
    ///
    /// Applies softmax with optional temperature scaling and returns the maximum probability
    pub fn from_logits<B: Backend>(&self, logits: &Tensor<B, 2>) -> Vec<f64> {
        // Apply temperature scaling
        let scaled_logits = if self.config.temperature != 1.0 {
            logits.clone() / self.config.temperature
        } else {
            logits.clone()
        };

        // Apply softmax
        let probabilities = burn::tensor::activation::softmax(scaled_logits, 1);

        // Extract max probability for each sample
        let probs_data = probabilities.clone().into_data();
        self.extract_max_probabilities(&probs_data)
    }

    /// Extract confidence from probability tensor
    ///
    /// Assumes input is already softmax probabilities
    pub fn from_probabilities<B: Backend>(&self, probabilities: &Tensor<B, 2>) -> Vec<f64> {
        let probs_data = probabilities.clone().into_data();
        self.extract_max_probabilities(&probs_data)
    }

    /// Extract class probabilities for all classes
    pub fn get_class_probabilities<B: Backend>(&self, logits: &Tensor<B, 2>) -> Vec<Vec<f64>> {
        let scaled_logits = if self.config.temperature != 1.0 {
            logits.clone() / self.config.temperature
        } else {
            logits.clone()
        };

        let probabilities = burn::tensor::activation::softmax(scaled_logits, 1);
        let probs_data = probabilities.into_data();

        self.extract_all_probabilities(&probs_data)
    }

    /// Get the predicted class for each sample
    pub fn get_predicted_classes<B: Backend>(&self, logits: &Tensor<B, 2>) -> Vec<usize> {
        let probabilities = burn::tensor::activation::softmax(logits.clone(), 1);
        let probs_data = probabilities.into_data();

        self.extract_predicted_classes(&probs_data)
    }

    /// Apply Platt scaling to a score
    fn apply_platt_scaling(&self, score: f64) -> f64 {
        if let Some((a, b)) = self.config.platt_params {
            1.0 / (1.0 + (a * score + b).exp())
        } else {
            score
        }
    }

    /// Calibrate a confidence score
    pub fn calibrate(&self, raw_confidence: f64) -> f64 {
        if self.config.use_platt_scaling {
            self.apply_platt_scaling(raw_confidence)
        } else {
            raw_confidence
        }
    }

    /// Check if a confidence value meets the minimum threshold
    pub fn meets_threshold(&self, confidence: f64) -> bool {
        confidence >= self.config.min_threshold
    }

    /// Extract maximum probabilities from tensor data
    fn extract_max_probabilities(&self, data: &burn::tensor::TensorData) -> Vec<f64> {
        let shape = &data.shape;
        let batch_size = shape[0];
        let num_classes = shape[1];

        let mut max_probs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut max_prob = 0.0f32;
            for j in 0..num_classes {
                let idx = i * num_classes + j;
                let prob = data.as_slice::<f32>().unwrap()[idx];
                if prob > max_prob {
                    max_prob = prob;
                }
            }
            let confidence = max_prob.elem::<f64>();
            max_probs.push(self.calibrate(confidence));
        }

        max_probs
    }

    /// Extract all class probabilities from tensor data
    fn extract_all_probabilities(&self, data: &burn::tensor::TensorData) -> Vec<Vec<f64>> {
        let shape = &data.shape;
        let batch_size = shape[0];
        let num_classes = shape[1];

        let mut all_probs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut class_probs = Vec::with_capacity(num_classes);
            for j in 0..num_classes {
                let idx = i * num_classes + j;
                let prob = data.as_slice::<f32>().unwrap()[idx];
                class_probs.push(prob.elem::<f64>());
            }
            all_probs.push(class_probs);
        }

        all_probs
    }

    /// Extract predicted class indices from tensor data
    fn extract_predicted_classes(&self, data: &burn::tensor::TensorData) -> Vec<usize> {
        let shape = &data.shape;
        let batch_size = shape[0];
        let num_classes = shape[1];

        let mut predicted = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut max_prob = 0.0f32;
            let mut max_idx = 0;
            for j in 0..num_classes {
                let idx = i * num_classes + j;
                let prob = data.as_slice::<f32>().unwrap()[idx];
                if prob > max_prob {
                    max_prob = prob;
                    max_idx = j;
                }
            }
            predicted.push(max_idx);
        }

        predicted
    }
}

/// Confidence statistics for a batch of signals
#[derive(Debug, Clone)]
pub struct ConfidenceStats {
    /// Mean confidence
    pub mean: f64,

    /// Median confidence
    pub median: f64,

    /// Minimum confidence
    pub min: f64,

    /// Maximum confidence
    pub max: f64,

    /// Standard deviation
    pub std: f64,

    /// Fraction of signals above threshold
    pub above_threshold: f64,
}

impl ConfidenceStats {
    /// Compute statistics from a vector of confidence values
    pub fn from_confidences(confidences: &[f64], threshold: f64) -> Self {
        if confidences.is_empty() {
            return Self {
                mean: 0.0,
                median: 0.0,
                min: 0.0,
                max: 0.0,
                std: 0.0,
                above_threshold: 0.0,
            };
        }

        let mut sorted = confidences.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        let variance =
            confidences.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / confidences.len() as f64;
        let std = variance.sqrt();

        let above_count = confidences.iter().filter(|&&x| x >= threshold).count();
        let above_threshold = above_count as f64 / confidences.len() as f64;

        Self {
            mean,
            median,
            min,
            max,
            std,
            above_threshold,
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Confidence Stats: mean={:.3}, median={:.3}, min={:.3}, max={:.3}, std={:.3}, above_threshold={:.1}%",
            self.mean,
            self.median,
            self.min,
            self.max,
            self.std,
            self.above_threshold * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type TestBackend = NdArray;

    #[test]
    fn test_confidence_config_default() {
        let config = ConfidenceConfig::default();
        assert_eq!(config.min_threshold, 0.7);
        assert_eq!(config.temperature, 1.0);
        assert!(!config.use_platt_scaling);
    }

    #[test]
    fn test_confidence_config_builders() {
        let config = ConfidenceConfig::with_threshold(0.8);
        assert_eq!(config.min_threshold, 0.8);

        let config = ConfidenceConfig::with_temperature(2.0);
        assert_eq!(config.temperature, 2.0);

        let config = ConfidenceConfig::default().with_platt_scaling(1.0, -0.5);
        assert!(config.use_platt_scaling);
        assert_eq!(config.platt_params, Some((1.0, -0.5)));
    }

    #[test]
    fn test_confidence_scorer_from_logits() {
        let config = ConfidenceConfig::default();
        let scorer = ConfidenceScorer::new(config);

        // Create logits: [batch_size=2, num_classes=3]
        let logits_data = vec![2.0f32, 1.0, 0.5, 0.5, 2.5, 1.0];
        let shape = burn::tensor::Shape::new([2, 3]);
        let tensor_data = TensorData::new(logits_data, shape);
        let device = Default::default();
        let logits = Tensor::<TestBackend, 2>::from_data(tensor_data.convert::<f32>(), &device);

        let confidences = scorer.from_logits(&logits);

        assert_eq!(confidences.len(), 2);
        assert!(confidences[0] > 0.5); // First sample should have high confidence for class 0
        assert!(confidences[1] > 0.5); // Second sample should have high confidence for class 1
    }

    #[test]
    fn test_predicted_classes() {
        let config = ConfidenceConfig::default();
        let scorer = ConfidenceScorer::new(config);

        let logits_data = vec![2.0f32, 1.0, 0.5, 0.5, 2.5, 1.0];
        let shape = burn::tensor::Shape::new([2, 3]);
        let tensor_data = TensorData::new(logits_data, shape);
        let device = Default::default();
        let logits = Tensor::<TestBackend, 2>::from_data(tensor_data.convert::<f32>(), &device);

        let classes = scorer.get_predicted_classes(&logits);

        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0], 0); // Max is at index 0
        assert_eq!(classes[1], 1); // Max is at index 1
    }

    #[test]
    fn test_class_probabilities() {
        let config = ConfidenceConfig::default();
        let scorer = ConfidenceScorer::new(config);

        let logits_data = vec![1.0f32, 2.0, 0.5];
        let shape = burn::tensor::Shape::new([1, 3]);
        let tensor_data = TensorData::new(logits_data, shape);
        let device = Default::default();
        let logits = Tensor::<TestBackend, 2>::from_data(tensor_data.convert::<f32>(), &device);

        let probs = scorer.get_class_probabilities(&logits);

        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0].len(), 3);

        // Probabilities should sum to ~1.0
        let sum: f64 = probs[0].iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_meets_threshold() {
        let config = ConfidenceConfig::with_threshold(0.7);
        let scorer = ConfidenceScorer::new(config);

        assert!(scorer.meets_threshold(0.8));
        assert!(scorer.meets_threshold(0.7));
        assert!(!scorer.meets_threshold(0.6));
    }

    #[test]
    fn test_platt_scaling() {
        let config = ConfidenceConfig::default().with_platt_scaling(1.0, -1.0);
        let scorer = ConfidenceScorer::new(config);

        let calibrated = scorer.calibrate(0.5);
        // With A=1.0, B=-1.0: 1 / (1 + exp(1.0 * 0.5 - 1.0)) = 1 / (1 + exp(-0.5))
        assert!(calibrated > 0.5 && calibrated < 0.7);
    }

    #[test]
    fn test_confidence_stats() {
        let confidences = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let stats = ConfidenceStats::from_confidences(&confidences, 0.7);

        assert!((stats.mean - 0.7).abs() < 0.001);
        assert_eq!(stats.median, 0.7);
        assert_eq!(stats.min, 0.5);
        assert_eq!(stats.max, 0.9);
        assert_eq!(stats.above_threshold, 0.6); // 3 out of 5
    }

    #[test]
    fn test_confidence_stats_empty() {
        let confidences: Vec<f64> = vec![];
        let stats = ConfidenceStats::from_confidences(&confidences, 0.7);

        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.median, 0.0);
    }

    #[test]
    fn test_temperature_scaling() {
        let config = ConfidenceConfig::with_temperature(2.0);
        let scorer = ConfidenceScorer::new(config);

        // Higher temperature should make probabilities more uniform
        let logits_data = vec![4.0f32, 2.0, 1.0];
        let shape = burn::tensor::Shape::new([1, 3]);
        let tensor_data = TensorData::new(logits_data, shape);
        let device = Default::default();
        let logits = Tensor::<TestBackend, 2>::from_data(tensor_data.convert::<f32>(), &device);

        let confidences = scorer.from_logits(&logits);

        // With temperature scaling, confidence should be lower than without
        let config_no_temp = ConfidenceConfig::default();
        let scorer_no_temp = ConfidenceScorer::new(config_no_temp);
        let confidences_no_temp = scorer_no_temp.from_logits(&logits);

        assert!(confidences[0] < confidences_no_temp[0]);
    }
}
