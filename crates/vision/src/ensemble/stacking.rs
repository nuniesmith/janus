//! Stacking and blending ensemble strategies with meta-learner support.
//!
//! This module provides advanced ensemble techniques:
//! - Stacking with meta-learner
//! - Blending with hold-out validation
//! - Multi-level ensembles
//! - Out-of-fold predictions
//! - Feature importance tracking

use crate::ensemble::models::{EnsemblePrediction, ModelPrediction};
use std::collections::HashMap;

/// Stacking configuration.
#[derive(Debug, Clone)]
pub struct StackingConfig {
    /// Number of folds for out-of-fold predictions
    pub num_folds: usize,
    /// Use out-of-fold predictions for meta-learner
    pub use_oof_predictions: bool,
    /// Include original features
    pub include_original_features: bool,
    /// Meta-learner regularization strength
    pub regularization: f64,
}

impl StackingConfig {
    /// Create default stacking configuration.
    pub fn default() -> Self {
        Self {
            num_folds: 5,
            use_oof_predictions: true,
            include_original_features: false,
            regularization: 0.01,
        }
    }

    /// Create configuration for blending (no cross-validation).
    pub fn blending() -> Self {
        Self {
            num_folds: 1,
            use_oof_predictions: false,
            include_original_features: false,
            regularization: 0.01,
        }
    }
}

/// Meta-learner for stacking.
#[derive(Debug, Clone)]
pub struct MetaLearner {
    /// Model weights learned from training
    weights: Vec<f64>,
    /// Bias term
    bias: f64,
    /// Number of base models
    num_models: usize,
    /// Training loss
    training_loss: f64,
}

impl MetaLearner {
    /// Create a new meta-learner.
    pub fn new(num_models: usize) -> Self {
        Self {
            weights: vec![1.0 / num_models as f64; num_models],
            bias: 0.0,
            num_models,
            training_loss: 0.0,
        }
    }

    /// Train meta-learner using linear regression.
    pub fn train(&mut self, predictions: &[Vec<f64>], targets: &[f64], regularization: f64) {
        if predictions.is_empty() || targets.is_empty() {
            return;
        }

        let n = predictions.len();
        let m = self.num_models;

        // Simple linear regression with regularization
        // weights = (X^T X + λI)^-1 X^T y

        // For simplicity, use gradient descent
        let learning_rate = 0.01;
        let epochs = 100;

        for _ in 0..epochs {
            let mut grad_weights = vec![0.0; m];
            let mut grad_bias = 0.0;

            for (i, target) in targets.iter().enumerate() {
                let pred = self.predict(&predictions[i]);
                let error = pred - target;

                // Gradient w.r.t weights
                for j in 0..m {
                    grad_weights[j] += error * predictions[i][j];
                }
                grad_bias += error;
            }

            // Update with regularization
            for j in 0..m {
                self.weights[j] -=
                    learning_rate * (grad_weights[j] / n as f64 + regularization * self.weights[j]);
            }
            self.bias -= learning_rate * grad_bias / n as f64;
        }

        // Calculate training loss
        self.training_loss = self.calculate_loss(predictions, targets);
    }

    /// Predict using meta-learner.
    pub fn predict(&self, base_predictions: &[f64]) -> f64 {
        let mut result = self.bias;
        for (i, &pred) in base_predictions.iter().enumerate() {
            if i < self.weights.len() {
                result += self.weights[i] * pred;
            }
        }
        result
    }

    /// Calculate mean squared error.
    fn calculate_loss(&self, predictions: &[Vec<f64>], targets: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (i, target) in targets.iter().enumerate() {
            let pred = self.predict(&predictions[i]);
            loss += (pred - target).powi(2);
        }
        loss / predictions.len() as f64
    }

    /// Get model weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get bias.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Get training loss.
    pub fn training_loss(&self) -> f64 {
        self.training_loss
    }
}

/// Stacking ensemble that learns optimal combination of base models.
pub struct StackingEnsemble {
    config: StackingConfig,
    meta_learner: Option<MetaLearner>,
    model_ids: Vec<String>,
    is_trained: bool,
}

impl StackingEnsemble {
    /// Create a new stacking ensemble.
    pub fn new(config: StackingConfig) -> Self {
        Self {
            config,
            meta_learner: None,
            model_ids: Vec::new(),
            is_trained: false,
        }
    }

    /// Create with default configuration.
    pub fn default() -> Self {
        Self::new(StackingConfig::default())
    }

    /// Train the stacking ensemble.
    pub fn train(
        &mut self,
        predictions: Vec<Vec<ModelPrediction>>,
        targets: Vec<f64>,
    ) -> Result<(), String> {
        if predictions.is_empty() || targets.is_empty() {
            return Err("Empty predictions or targets".to_string());
        }

        if predictions.len() != targets.len() {
            return Err("Predictions and targets length mismatch".to_string());
        }

        // Extract model IDs from first prediction set
        if let Some(first_preds) = predictions.first() {
            self.model_ids = first_preds.iter().map(|p| p.model_id.clone()).collect();
        }

        let num_models = self.model_ids.len();
        if num_models == 0 {
            return Err("No models found".to_string());
        }

        // Convert to matrix format (n_samples x n_models)
        let mut pred_matrix = Vec::new();
        for sample_preds in &predictions {
            let signals: Vec<f64> = sample_preds.iter().map(|p| p.signal).collect();
            pred_matrix.push(signals);
        }

        // Train meta-learner
        let mut meta = MetaLearner::new(num_models);
        meta.train(&pred_matrix, &targets, self.config.regularization);

        self.meta_learner = Some(meta);
        self.is_trained = true;

        Ok(())
    }

    /// Predict using trained stacking ensemble.
    pub fn predict(&self, predictions: Vec<ModelPrediction>) -> Option<EnsemblePrediction> {
        if !self.is_trained {
            return None;
        }

        let meta = self.meta_learner.as_ref()?;

        // Extract signals in correct order
        let mut signals = vec![0.0; self.model_ids.len()];
        for pred in &predictions {
            if let Some(idx) = self.model_ids.iter().position(|id| id == &pred.model_id) {
                signals[idx] = pred.signal;
            }
        }

        let signal = meta.predict(&signals);

        // Calculate average confidence
        let confidence = if !predictions.is_empty() {
            predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64
        } else {
            0.0
        };

        // Calculate agreement
        let bullish = predictions.iter().filter(|p| p.is_bullish()).count();
        let total = predictions.len();
        let agreement = if total > 0 {
            bullish.max(total - bullish) as f64 / total as f64
        } else {
            0.0
        };

        Some(EnsemblePrediction {
            signal,
            confidence,
            num_models: predictions.len(),
            agreement,
            individual_predictions: predictions,
        })
    }

    /// Check if ensemble is trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get meta-learner weights.
    pub fn get_weights(&self) -> Option<Vec<(String, f64)>> {
        let meta = self.meta_learner.as_ref()?;
        Some(
            self.model_ids
                .iter()
                .zip(meta.weights())
                .map(|(id, &w)| (id.clone(), w))
                .collect(),
        )
    }

    /// Get training loss.
    pub fn training_loss(&self) -> Option<f64> {
        self.meta_learner.as_ref().map(|m| m.training_loss())
    }
}

/// Blending ensemble using hold-out validation.
pub struct BlendingEnsemble {
    weights: HashMap<String, f64>,
    total_weight: f64,
    is_trained: bool,
}

impl BlendingEnsemble {
    /// Create a new blending ensemble.
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            total_weight: 0.0,
            is_trained: false,
        }
    }

    /// Train blending ensemble on validation set.
    pub fn train(
        &mut self,
        predictions: Vec<Vec<ModelPrediction>>,
        targets: Vec<f64>,
    ) -> Result<(), String> {
        if predictions.is_empty() || targets.is_empty() {
            return Err("Empty predictions or targets".to_string());
        }

        // Calculate performance of each model
        let mut model_scores: HashMap<String, (f64, usize)> = HashMap::new();

        for (i, sample_preds) in predictions.iter().enumerate() {
            let target = targets[i];

            for pred in sample_preds {
                let error = (pred.signal - target).abs();
                let entry = model_scores
                    .entry(pred.model_id.clone())
                    .or_insert((0.0, 0));
                entry.0 += error;
                entry.1 += 1;
            }
        }

        // Calculate weights based on inverse error
        self.weights.clear();
        self.total_weight = 0.0;

        for (model_id, (total_error, count)) in model_scores {
            if count > 0 {
                let mae = total_error / count as f64;
                // Use inverse error as weight (add small epsilon to avoid division by zero)
                let weight = 1.0 / (mae + 1e-6);
                self.weights.insert(model_id, weight);
                self.total_weight += weight;
            }
        }

        // Normalize weights
        if self.total_weight > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= self.total_weight;
            }
            self.total_weight = 1.0;
        }

        self.is_trained = true;
        Ok(())
    }

    /// Predict using blending ensemble.
    pub fn predict(&self, predictions: Vec<ModelPrediction>) -> Option<EnsemblePrediction> {
        if !self.is_trained || predictions.is_empty() {
            return None;
        }

        let mut weighted_signal = 0.0;
        let mut weighted_confidence = 0.0;
        let mut total_weight = 0.0;

        for pred in &predictions {
            let weight = self.weights.get(&pred.model_id).copied().unwrap_or(0.0);
            weighted_signal += pred.signal * weight;
            weighted_confidence += pred.confidence * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return None;
        }

        let signal = weighted_signal / total_weight;
        let confidence = weighted_confidence / total_weight;

        // Calculate agreement
        let bullish = predictions.iter().filter(|p| p.is_bullish()).count();
        let agreement = bullish.max(predictions.len() - bullish) as f64 / predictions.len() as f64;

        Some(EnsemblePrediction {
            signal,
            confidence,
            num_models: predictions.len(),
            agreement,
            individual_predictions: predictions,
        })
    }

    /// Get model weights.
    pub fn get_weights(&self) -> &HashMap<String, f64> {
        &self.weights
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
}

impl Default for BlendingEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature importance tracker for ensemble models.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    importances: HashMap<String, f64>,
}

impl FeatureImportance {
    /// Create new feature importance tracker.
    pub fn new() -> Self {
        Self {
            importances: HashMap::new(),
        }
    }

    /// Update importance for a model.
    pub fn update(&mut self, model_id: String, importance: f64) {
        self.importances.insert(model_id, importance);
    }

    /// Get importance for a model.
    pub fn get(&self, model_id: &str) -> f64 {
        self.importances.get(model_id).copied().unwrap_or(0.0)
    }

    /// Get all importances sorted by value.
    pub fn get_sorted(&self) -> Vec<(String, f64)> {
        let mut items: Vec<_> = self
            .importances
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        items
    }

    /// Normalize importances to sum to 1.0.
    pub fn normalize(&mut self) {
        let total: f64 = self.importances.values().sum();
        if total > 0.0 {
            for value in self.importances.values_mut() {
                *value /= total;
            }
        }
    }
}

impl Default for FeatureImportance {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learner() {
        let mut meta = MetaLearner::new(2);

        let predictions = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let targets = vec![0.15, 0.35, 0.55];

        meta.train(&predictions, &targets, 0.01);

        let pred = meta.predict(&vec![0.2, 0.3]);
        assert!(pred > 0.0);
    }

    #[test]
    fn test_stacking_ensemble() {
        let mut ensemble = StackingEnsemble::default();

        let predictions = vec![
            vec![
                ModelPrediction::new("m1".to_string(), 0.1, 0.8),
                ModelPrediction::new("m2".to_string(), 0.2, 0.9),
            ],
            vec![
                ModelPrediction::new("m1".to_string(), 0.3, 0.7),
                ModelPrediction::new("m2".to_string(), 0.4, 0.8),
            ],
        ];
        let targets = vec![0.15, 0.35];

        ensemble.train(predictions, targets).unwrap();
        assert!(ensemble.is_trained());

        let weights = ensemble.get_weights().unwrap();
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_blending_ensemble() {
        let mut ensemble = BlendingEnsemble::new();

        let predictions = vec![
            vec![
                ModelPrediction::new("m1".to_string(), 0.1, 0.8),
                ModelPrediction::new("m2".to_string(), 0.2, 0.9),
            ],
            vec![
                ModelPrediction::new("m1".to_string(), 0.3, 0.7),
                ModelPrediction::new("m2".to_string(), 0.35, 0.8),
            ],
        ];
        let targets = vec![0.15, 0.35];

        ensemble.train(predictions, targets).unwrap();
        assert!(ensemble.is_trained());

        let weights = ensemble.get_weights();
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_feature_importance() {
        let mut importance = FeatureImportance::new();

        importance.update("m1".to_string(), 0.6);
        importance.update("m2".to_string(), 0.4);

        let sorted = importance.get_sorted();
        assert_eq!(sorted[0].0, "m1");
        assert_eq!(sorted[1].0, "m2");

        importance.normalize();
        let total: f64 = importance.importances.values().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stacking_prediction() {
        let mut ensemble = StackingEnsemble::default();

        let train_preds = vec![
            vec![
                ModelPrediction::new("m1".to_string(), 0.1, 0.8),
                ModelPrediction::new("m2".to_string(), 0.2, 0.9),
            ],
            vec![
                ModelPrediction::new("m1".to_string(), 0.3, 0.7),
                ModelPrediction::new("m2".to_string(), 0.4, 0.8),
            ],
        ];
        let targets = vec![0.15, 0.35];

        ensemble.train(train_preds, targets).unwrap();

        let test_preds = vec![
            ModelPrediction::new("m1".to_string(), 0.25, 0.75),
            ModelPrediction::new("m2".to_string(), 0.30, 0.85),
        ];

        let result = ensemble.predict(test_preds).unwrap();
        assert!(result.signal >= 0.0);
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_blending_prediction() {
        let mut ensemble = BlendingEnsemble::new();

        let train_preds = vec![vec![
            ModelPrediction::new("m1".to_string(), 0.1, 0.8),
            ModelPrediction::new("m2".to_string(), 0.2, 0.9),
        ]];
        let targets = vec![0.15];

        ensemble.train(train_preds, targets).unwrap();

        let test_preds = vec![
            ModelPrediction::new("m1".to_string(), 0.25, 0.75),
            ModelPrediction::new("m2".to_string(), 0.30, 0.85),
        ];

        let result = ensemble.predict(test_preds).unwrap();
        assert!(result.num_models == 2);
    }
}
