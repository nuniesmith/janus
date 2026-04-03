//! Model ensemble system for combining multiple model predictions.
//!
//! This module provides ensemble learning capabilities:
//! - Weighted voting and averaging
//! - Model performance tracking
//! - Dynamic weight adjustment
//! - Ensemble strategies (mean, median, weighted)
//! - Model selection and filtering

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Model prediction with metadata.
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    pub model_id: String,
    pub signal: f64,
    pub confidence: f64,
    pub latency_us: u64,
    pub timestamp: i64,
}

impl ModelPrediction {
    /// Create a new model prediction.
    pub fn new(model_id: String, signal: f64, confidence: f64) -> Self {
        Self {
            model_id,
            signal,
            confidence,
            latency_us: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        }
    }

    /// Create with latency.
    pub fn with_latency(mut self, latency_us: u64) -> Self {
        self.latency_us = latency_us;
        self
    }

    /// Check if prediction is bullish.
    pub fn is_bullish(&self) -> bool {
        self.signal > 0.0
    }

    /// Check if prediction is bearish.
    pub fn is_bearish(&self) -> bool {
        self.signal < 0.0
    }

    /// Get weighted signal (signal * confidence).
    pub fn weighted_signal(&self) -> f64 {
        self.signal * self.confidence
    }
}

/// Model performance statistics.
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    pub model_id: String,
    pub predictions: usize,
    pub correct_predictions: usize,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub avg_confidence: f64,
    pub avg_latency_us: u64,
    pub last_updated: i64,
}

impl ModelPerformance {
    /// Create new performance tracker.
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            predictions: 0,
            correct_predictions: 0,
            total_return: 0.0,
            sharpe_ratio: 0.0,
            avg_confidence: 0.0,
            avg_latency_us: 0,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        }
    }

    /// Get accuracy rate.
    pub fn accuracy(&self) -> f64 {
        if self.predictions == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.predictions as f64
        }
    }

    /// Update with new prediction result.
    pub fn update(&mut self, correct: bool, return_value: f64, confidence: f64, latency_us: u64) {
        self.predictions += 1;
        if correct {
            self.correct_predictions += 1;
        }
        self.total_return += return_value;

        // Update averages
        let n = self.predictions as f64;
        self.avg_confidence = (self.avg_confidence * (n - 1.0) + confidence) / n;
        self.avg_latency_us =
            ((self.avg_latency_us as f64 * (n - 1.0) + latency_us as f64) / n) as u64;
        self.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
    }

    /// Check if model meets minimum quality threshold.
    pub fn meets_quality_threshold(&self, min_accuracy: f64, min_predictions: usize) -> bool {
        self.predictions >= min_predictions && self.accuracy() >= min_accuracy
    }
}

/// Ensemble strategy for combining predictions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnsembleStrategy {
    /// Simple mean of all predictions
    Mean,
    /// Median of all predictions
    Median,
    /// Weighted by confidence
    WeightedByConfidence,
    /// Weighted by historical performance
    WeightedByPerformance,
    /// Weighted by both confidence and performance
    WeightedCombined,
    /// Best performing model only
    BestModel,
}

/// Ensemble configuration.
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    pub strategy: EnsembleStrategy,
    pub min_models: usize,
    pub max_models: usize,
    pub min_model_accuracy: f64,
    pub min_model_predictions: usize,
    pub confidence_weight: f64,
    pub performance_weight: f64,
}

impl EnsembleConfig {
    /// Create default ensemble configuration.
    pub fn default() -> Self {
        Self {
            strategy: EnsembleStrategy::WeightedCombined,
            min_models: 2,
            max_models: 10,
            min_model_accuracy: 0.55,
            min_model_predictions: 100,
            confidence_weight: 0.3,
            performance_weight: 0.7,
        }
    }

    /// Create conservative configuration (higher quality threshold).
    pub fn conservative() -> Self {
        Self {
            strategy: EnsembleStrategy::WeightedByPerformance,
            min_models: 3,
            max_models: 5,
            min_model_accuracy: 0.60,
            min_model_predictions: 200,
            confidence_weight: 0.2,
            performance_weight: 0.8,
        }
    }

    /// Create aggressive configuration (more models, lower threshold).
    pub fn aggressive() -> Self {
        Self {
            strategy: EnsembleStrategy::WeightedCombined,
            min_models: 1,
            max_models: 20,
            min_model_accuracy: 0.50,
            min_model_predictions: 50,
            confidence_weight: 0.5,
            performance_weight: 0.5,
        }
    }
}

/// Ensemble prediction result.
#[derive(Debug, Clone)]
pub struct EnsemblePrediction {
    pub signal: f64,
    pub confidence: f64,
    pub num_models: usize,
    pub agreement: f64,
    pub individual_predictions: Vec<ModelPrediction>,
}

impl EnsemblePrediction {
    /// Check if prediction is bullish.
    pub fn is_bullish(&self) -> bool {
        self.signal > 0.0
    }

    /// Check if prediction is bearish.
    pub fn is_bearish(&self) -> bool {
        self.signal < 0.0
    }

    /// Check if models agree (same direction).
    pub fn has_consensus(&self, threshold: f64) -> bool {
        self.agreement >= threshold
    }

    /// Get bullish/bearish vote counts.
    pub fn vote_counts(&self) -> (usize, usize) {
        let bullish = self
            .individual_predictions
            .iter()
            .filter(|p| p.is_bullish())
            .count();
        let bearish = self
            .individual_predictions
            .iter()
            .filter(|p| p.is_bearish())
            .count();
        (bullish, bearish)
    }
}

/// Model ensemble for combining multiple model predictions.
pub struct ModelEnsemble {
    config: EnsembleConfig,
    performance: Arc<RwLock<HashMap<String, ModelPerformance>>>,
}

impl ModelEnsemble {
    /// Create a new model ensemble.
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            performance: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create an ensemble with default configuration.
    pub fn default() -> Self {
        Self::new(EnsembleConfig::default())
    }

    /// Add or update model performance.
    pub fn update_performance(&self, performance: ModelPerformance) {
        if let Ok(mut perf) = self.performance.write() {
            perf.insert(performance.model_id.clone(), performance);
        }
    }

    /// Record prediction result for a model.
    pub fn record_result(
        &self,
        model_id: &str,
        correct: bool,
        return_value: f64,
        confidence: f64,
        latency_us: u64,
    ) {
        if let Ok(mut perf) = self.performance.write() {
            let entry = perf
                .entry(model_id.to_string())
                .or_insert_with(|| ModelPerformance::new(model_id.to_string()));
            entry.update(correct, return_value, confidence, latency_us);
        }
    }

    /// Combine predictions using the configured strategy.
    pub fn predict(&self, predictions: Vec<ModelPrediction>) -> Option<EnsemblePrediction> {
        if predictions.is_empty() {
            return None;
        }

        // Filter predictions based on model quality
        let filtered = self.filter_predictions(predictions);

        if filtered.len() < self.config.min_models {
            return None;
        }

        let (signal, confidence) = match self.config.strategy {
            EnsembleStrategy::Mean => self.mean_prediction(&filtered),
            EnsembleStrategy::Median => self.median_prediction(&filtered),
            EnsembleStrategy::WeightedByConfidence => {
                self.confidence_weighted_prediction(&filtered)
            }
            EnsembleStrategy::WeightedByPerformance => {
                self.performance_weighted_prediction(&filtered)
            }
            EnsembleStrategy::WeightedCombined => self.combined_weighted_prediction(&filtered),
            EnsembleStrategy::BestModel => self.best_model_prediction(&filtered),
        };

        let agreement = self.calculate_agreement(&filtered);

        Some(EnsemblePrediction {
            signal,
            confidence,
            num_models: filtered.len(),
            agreement,
            individual_predictions: filtered,
        })
    }

    /// Filter predictions based on model performance.
    fn filter_predictions(&self, predictions: Vec<ModelPrediction>) -> Vec<ModelPrediction> {
        if let Ok(perf) = self.performance.read() {
            predictions
                .into_iter()
                .filter(|pred| {
                    if let Some(model_perf) = perf.get(&pred.model_id) {
                        model_perf.meets_quality_threshold(
                            self.config.min_model_accuracy,
                            self.config.min_model_predictions,
                        )
                    } else {
                        // Allow new models with no history
                        true
                    }
                })
                .take(self.config.max_models)
                .collect()
        } else {
            predictions
        }
    }

    /// Simple mean prediction.
    fn mean_prediction(&self, predictions: &[ModelPrediction]) -> (f64, f64) {
        let signal = predictions.iter().map(|p| p.signal).sum::<f64>() / predictions.len() as f64;
        let confidence =
            predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64;
        (signal, confidence)
    }

    /// Median prediction.
    fn median_prediction(&self, predictions: &[ModelPrediction]) -> (f64, f64) {
        let mut signals: Vec<f64> = predictions.iter().map(|p| p.signal).collect();
        let mut confidences: Vec<f64> = predictions.iter().map(|p| p.confidence).collect();

        signals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        confidences.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let signal = signals[signals.len() / 2];
        let confidence = confidences[confidences.len() / 2];

        (signal, confidence)
    }

    /// Confidence-weighted prediction.
    fn confidence_weighted_prediction(&self, predictions: &[ModelPrediction]) -> (f64, f64) {
        let total_confidence: f64 = predictions.iter().map(|p| p.confidence).sum();

        if total_confidence == 0.0 {
            return self.mean_prediction(predictions);
        }

        let weighted_signal: f64 = predictions
            .iter()
            .map(|p| p.signal * p.confidence)
            .sum::<f64>()
            / total_confidence;

        let avg_confidence: f64 = total_confidence / predictions.len() as f64;

        (weighted_signal, avg_confidence)
    }

    /// Performance-weighted prediction.
    fn performance_weighted_prediction(&self, predictions: &[ModelPrediction]) -> (f64, f64) {
        if let Ok(perf) = self.performance.read() {
            let mut weights: Vec<f64> = Vec::new();
            let mut signals: Vec<f64> = Vec::new();
            let mut confidences: Vec<f64> = Vec::new();

            for pred in predictions {
                let weight = if let Some(model_perf) = perf.get(&pred.model_id) {
                    model_perf.accuracy()
                } else {
                    0.5 // Default weight for new models
                };

                weights.push(weight);
                signals.push(pred.signal);
                confidences.push(pred.confidence);
            }

            let total_weight: f64 = weights.iter().sum();
            if total_weight == 0.0 {
                return self.mean_prediction(predictions);
            }

            let weighted_signal: f64 = signals
                .iter()
                .zip(&weights)
                .map(|(s, w)| s * w)
                .sum::<f64>()
                / total_weight;

            let weighted_confidence: f64 = confidences
                .iter()
                .zip(&weights)
                .map(|(c, w)| c * w)
                .sum::<f64>()
                / total_weight;

            (weighted_signal, weighted_confidence)
        } else {
            self.mean_prediction(predictions)
        }
    }

    /// Combined weighted prediction (confidence + performance).
    fn combined_weighted_prediction(&self, predictions: &[ModelPrediction]) -> (f64, f64) {
        if let Ok(perf) = self.performance.read() {
            let cw = self.config.confidence_weight;
            let pw = self.config.performance_weight;

            let mut weights: Vec<f64> = Vec::new();
            let mut signals: Vec<f64> = Vec::new();
            let mut confidences: Vec<f64> = Vec::new();

            for pred in predictions {
                let perf_weight = if let Some(model_perf) = perf.get(&pred.model_id) {
                    model_perf.accuracy()
                } else {
                    0.5
                };

                let combined_weight = cw * pred.confidence + pw * perf_weight;
                weights.push(combined_weight);
                signals.push(pred.signal);
                confidences.push(pred.confidence);
            }

            let total_weight: f64 = weights.iter().sum();
            if total_weight == 0.0 {
                return self.mean_prediction(predictions);
            }

            let weighted_signal: f64 = signals
                .iter()
                .zip(&weights)
                .map(|(s, w)| s * w)
                .sum::<f64>()
                / total_weight;

            let weighted_confidence: f64 = confidences
                .iter()
                .zip(&weights)
                .map(|(c, w)| c * w)
                .sum::<f64>()
                / total_weight;

            (weighted_signal, weighted_confidence)
        } else {
            self.mean_prediction(predictions)
        }
    }

    /// Best model prediction (highest performing).
    fn best_model_prediction(&self, predictions: &[ModelPrediction]) -> (f64, f64) {
        if let Ok(perf) = self.performance.read() {
            let best = predictions.iter().max_by(|a, b| {
                let a_acc = perf.get(&a.model_id).map(|p| p.accuracy()).unwrap_or(0.0);
                let b_acc = perf.get(&b.model_id).map(|p| p.accuracy()).unwrap_or(0.0);
                a_acc.partial_cmp(&b_acc).unwrap()
            });

            if let Some(best_pred) = best {
                return (best_pred.signal, best_pred.confidence);
            }
        }

        self.mean_prediction(predictions)
    }

    /// Calculate agreement among predictions (0.0 to 1.0).
    fn calculate_agreement(&self, predictions: &[ModelPrediction]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        let bullish_count = predictions.iter().filter(|p| p.is_bullish()).count();
        let bearish_count = predictions.iter().filter(|p| p.is_bearish()).count();

        let max_count = bullish_count.max(bearish_count);
        max_count as f64 / predictions.len() as f64
    }

    /// Get model performance statistics.
    pub fn get_performance(&self, model_id: &str) -> Option<ModelPerformance> {
        self.performance.read().ok()?.get(model_id).cloned()
    }

    /// Get all model performances.
    pub fn get_all_performances(&self) -> Vec<ModelPerformance> {
        self.performance
            .read()
            .ok()
            .map(|p| p.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get top N performing models.
    pub fn get_top_models(&self, n: usize) -> Vec<ModelPerformance> {
        let mut performances = self.get_all_performances();
        performances.sort_by(|a, b| b.accuracy().partial_cmp(&a.accuracy()).unwrap());
        performances.into_iter().take(n).collect()
    }

    /// Get ensemble configuration.
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_prediction() {
        let pred = ModelPrediction::new("model1".to_string(), 0.5, 0.8);
        assert!(pred.is_bullish());
        assert!(!pred.is_bearish());
        assert_eq!(pred.weighted_signal(), 0.4);
    }

    #[test]
    fn test_model_performance() {
        let mut perf = ModelPerformance::new("model1".to_string());
        assert_eq!(perf.accuracy(), 0.0);

        perf.update(true, 0.01, 0.8, 1000);
        perf.update(true, 0.02, 0.9, 1200);
        perf.update(false, -0.01, 0.7, 1100);

        assert_eq!(perf.predictions, 3);
        assert_eq!(perf.correct_predictions, 2);
        assert_eq!(perf.accuracy(), 2.0 / 3.0);
    }

    #[test]
    fn test_ensemble_mean() {
        let ensemble = ModelEnsemble::new(EnsembleConfig {
            strategy: EnsembleStrategy::Mean,
            min_models: 1,
            ..EnsembleConfig::default()
        });

        let predictions = vec![
            ModelPrediction::new("m1".to_string(), 0.3, 0.8),
            ModelPrediction::new("m2".to_string(), 0.5, 0.9),
            ModelPrediction::new("m3".to_string(), 0.4, 0.7),
        ];

        let result = ensemble.predict(predictions).unwrap();
        assert!((result.signal - 0.4).abs() < 1e-10);
        assert!((result.confidence - 0.8).abs() < 1e-10);
        assert_eq!(result.num_models, 3);
    }

    #[test]
    fn test_ensemble_weighted() {
        let ensemble = ModelEnsemble::new(EnsembleConfig {
            strategy: EnsembleStrategy::WeightedByConfidence,
            min_models: 1,
            ..EnsembleConfig::default()
        });

        let predictions = vec![
            ModelPrediction::new("m1".to_string(), 0.5, 0.9),
            ModelPrediction::new("m2".to_string(), 0.3, 0.6),
        ];

        let result = ensemble.predict(predictions).unwrap();
        assert!(result.signal > 0.3 && result.signal < 0.5);
    }

    #[test]
    fn test_agreement_calculation() {
        let ensemble = ModelEnsemble::default();

        let predictions = vec![
            ModelPrediction::new("m1".to_string(), 0.3, 0.8),
            ModelPrediction::new("m2".to_string(), 0.5, 0.9),
            ModelPrediction::new("m3".to_string(), -0.2, 0.7),
        ];

        let agreement = ensemble.calculate_agreement(&predictions);
        assert_eq!(agreement, 2.0 / 3.0);
    }

    #[test]
    fn test_filter_predictions() {
        let ensemble = ModelEnsemble::new(EnsembleConfig {
            min_model_accuracy: 0.6,
            min_model_predictions: 10,
            ..EnsembleConfig::default()
        });

        // Add performance for model1 (good) and model2 (bad)
        let mut perf1 = ModelPerformance::new("m1".to_string());
        for _ in 0..10 {
            perf1.update(true, 0.01, 0.8, 1000);
        }
        ensemble.update_performance(perf1);

        let mut perf2 = ModelPerformance::new("m2".to_string());
        for _ in 0..10 {
            perf2.update(false, -0.01, 0.7, 1000);
        }
        ensemble.update_performance(perf2);

        let predictions = vec![
            ModelPrediction::new("m1".to_string(), 0.5, 0.8),
            ModelPrediction::new("m2".to_string(), 0.3, 0.9),
            ModelPrediction::new("m3".to_string(), 0.4, 0.7), // New model
        ];

        let filtered = ensemble.filter_predictions(predictions);
        // m1 passes (100% accuracy), m2 fails (0% accuracy), m3 passes (new)
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_ensemble_prediction_methods() {
        let pred = EnsemblePrediction {
            signal: 0.5,
            confidence: 0.8,
            num_models: 3,
            agreement: 0.9,
            individual_predictions: vec![
                ModelPrediction::new("m1".to_string(), 0.3, 0.8),
                ModelPrediction::new("m2".to_string(), 0.5, 0.9),
                ModelPrediction::new("m3".to_string(), 0.4, 0.7),
            ],
        };

        assert!(pred.is_bullish());
        assert!(pred.has_consensus(0.8));

        let (bullish, bearish) = pred.vote_counts();
        assert_eq!(bullish, 3);
        assert_eq!(bearish, 0);
    }

    #[test]
    fn test_top_models() {
        let ensemble = ModelEnsemble::default();

        let mut perf1 = ModelPerformance::new("m1".to_string());
        perf1.update(true, 0.01, 0.8, 1000);
        perf1.update(true, 0.01, 0.8, 1000);
        ensemble.update_performance(perf1);

        let mut perf2 = ModelPerformance::new("m2".to_string());
        perf2.update(true, 0.01, 0.8, 1000);
        perf2.update(false, -0.01, 0.8, 1000);
        ensemble.update_performance(perf2);

        let top = ensemble.get_top_models(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].model_id, "m1");
    }
}
