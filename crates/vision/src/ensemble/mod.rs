//! Model ensemble module for combining multiple model predictions.
//!
//! This module provides comprehensive ensemble learning capabilities:
//! - Weighted voting and averaging
//! - Stacking with meta-learner
//! - Blending with hold-out validation
//! - Model performance tracking
//! - Dynamic weight adjustment
//! - Feature importance analysis
//!
//! # Architecture
//!
//! ```text
//! Ensemble System
//! ├── Base Models → Individual Predictions
//! ├── Voting/Averaging → Simple Combination
//! ├── Stacking → Meta-Learner Training
//! ├── Blending → Validation-Based Weights
//! └── Performance Tracking → Weight Updates
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use vision::ensemble::{ModelEnsemble, EnsembleConfig, ModelPrediction};
//!
//! let ensemble = ModelEnsemble::default();
//!
//! let predictions = vec![
//!     ModelPrediction::new("model1".to_string(), 0.5, 0.8),
//!     ModelPrediction::new("model2".to_string(), 0.6, 0.9),
//!     ModelPrediction::new("model3".to_string(), 0.4, 0.7),
//! ];
//!
//! let result = ensemble.predict(predictions).unwrap();
//! println!("Signal: {:.4}", result.signal);
//! println!("Confidence: {:.4}", result.confidence);
//! println!("Agreement: {:.2}%", result.agreement * 100.0);
//! ```

pub mod models;
pub mod stacking;

pub use models::{
    EnsembleConfig, EnsemblePrediction, EnsembleStrategy, ModelEnsemble, ModelPerformance,
    ModelPrediction,
};
pub use stacking::{
    BlendingEnsemble, FeatureImportance, MetaLearner, StackingConfig, StackingEnsemble,
};

use crate::error::Result;
use std::collections::HashMap;

/// Ensemble manager for coordinating multiple ensemble strategies.
pub struct EnsembleManager {
    /// Simple voting ensemble
    voting_ensemble: ModelEnsemble,
    /// Stacking ensemble with meta-learner
    stacking_ensemble: Option<StackingEnsemble>,
    /// Blending ensemble
    blending_ensemble: Option<BlendingEnsemble>,
    /// Active strategy
    active_strategy: ManagerStrategy,
}

/// Strategy selection for ensemble manager.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ManagerStrategy {
    /// Use voting ensemble
    Voting,
    /// Use stacking ensemble
    Stacking,
    /// Use blending ensemble
    Blending,
    /// Automatically select best strategy
    Auto,
}

impl EnsembleManager {
    /// Create a new ensemble manager.
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            voting_ensemble: ModelEnsemble::new(config),
            stacking_ensemble: None,
            blending_ensemble: None,
            active_strategy: ManagerStrategy::Voting,
        }
    }

    /// Create with default configuration.
    pub fn default() -> Self {
        Self::new(EnsembleConfig::default())
    }

    /// Set active strategy.
    pub fn set_strategy(&mut self, strategy: ManagerStrategy) {
        self.active_strategy = strategy;
    }

    /// Train stacking ensemble.
    pub fn train_stacking(
        &mut self,
        predictions: Vec<Vec<ModelPrediction>>,
        targets: Vec<f64>,
    ) -> Result<()> {
        let mut stacking = StackingEnsemble::default();
        stacking
            .train(predictions, targets)
            .map_err(|e| crate::error::VisionError::Other(e))?;
        self.stacking_ensemble = Some(stacking);
        Ok(())
    }

    /// Train blending ensemble.
    pub fn train_blending(
        &mut self,
        predictions: Vec<Vec<ModelPrediction>>,
        targets: Vec<f64>,
    ) -> Result<()> {
        let mut blending = BlendingEnsemble::new();
        blending
            .train(predictions, targets)
            .map_err(|e| crate::error::VisionError::Other(e))?;
        self.blending_ensemble = Some(blending);
        Ok(())
    }

    /// Record model performance.
    pub fn record_performance(
        &self,
        model_id: &str,
        correct: bool,
        return_value: f64,
        confidence: f64,
        latency_us: u64,
    ) {
        self.voting_ensemble
            .record_result(model_id, correct, return_value, confidence, latency_us);
    }

    /// Predict using active strategy.
    pub fn predict(&self, predictions: Vec<ModelPrediction>) -> Option<EnsemblePrediction> {
        match self.active_strategy {
            ManagerStrategy::Voting => self.voting_ensemble.predict(predictions),
            ManagerStrategy::Stacking => {
                if let Some(ref stacking) = self.stacking_ensemble {
                    stacking.predict(predictions)
                } else {
                    self.voting_ensemble.predict(predictions)
                }
            }
            ManagerStrategy::Blending => {
                if let Some(ref blending) = self.blending_ensemble {
                    blending.predict(predictions)
                } else {
                    self.voting_ensemble.predict(predictions)
                }
            }
            ManagerStrategy::Auto => self.auto_predict(predictions),
        }
    }

    /// Automatically select best strategy for prediction.
    fn auto_predict(&self, predictions: Vec<ModelPrediction>) -> Option<EnsemblePrediction> {
        // Try stacking first if available and trained
        if let Some(ref stacking) = self.stacking_ensemble {
            if stacking.is_trained() {
                return stacking.predict(predictions);
            }
        }

        // Try blending next
        if let Some(ref blending) = self.blending_ensemble {
            if blending.is_trained() {
                return blending.predict(predictions);
            }
        }

        // Fall back to voting
        self.voting_ensemble.predict(predictions)
    }

    /// Get model performances.
    pub fn get_performances(&self) -> Vec<ModelPerformance> {
        self.voting_ensemble.get_all_performances()
    }

    /// Get top performing models.
    pub fn get_top_models(&self, n: usize) -> Vec<ModelPerformance> {
        self.voting_ensemble.get_top_models(n)
    }

    /// Get stacking weights if available.
    pub fn get_stacking_weights(&self) -> Option<Vec<(String, f64)>> {
        self.stacking_ensemble.as_ref()?.get_weights()
    }

    /// Get blending weights if available.
    pub fn get_blending_weights(&self) -> Option<HashMap<String, f64>> {
        Some(self.blending_ensemble.as_ref()?.get_weights().clone())
    }

    /// Check if stacking is trained.
    pub fn is_stacking_trained(&self) -> bool {
        self.stacking_ensemble
            .as_ref()
            .map(|s| s.is_trained())
            .unwrap_or(false)
    }

    /// Check if blending is trained.
    pub fn is_blending_trained(&self) -> bool {
        self.blending_ensemble
            .as_ref()
            .map(|b| b.is_trained())
            .unwrap_or(false)
    }

    /// Get voting ensemble.
    pub fn voting_ensemble(&self) -> &ModelEnsemble {
        &self.voting_ensemble
    }

    /// Get active strategy.
    pub fn active_strategy(&self) -> ManagerStrategy {
        self.active_strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_manager_creation() {
        let manager = EnsembleManager::default();
        assert_eq!(manager.active_strategy(), ManagerStrategy::Voting);
    }

    #[test]
    fn test_strategy_switching() {
        let mut manager = EnsembleManager::default();
        manager.set_strategy(ManagerStrategy::Stacking);
        assert_eq!(manager.active_strategy(), ManagerStrategy::Stacking);
    }

    #[test]
    fn test_voting_prediction() {
        let manager = EnsembleManager::default();

        let predictions = vec![
            ModelPrediction::new("m1".to_string(), 0.5, 0.8),
            ModelPrediction::new("m2".to_string(), 0.6, 0.9),
        ];

        let result = manager.predict(predictions);
        assert!(result.is_some());
    }

    #[test]
    fn test_train_stacking() {
        let mut manager = EnsembleManager::default();

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

        manager.train_stacking(train_preds, targets).unwrap();
        assert!(manager.is_stacking_trained());
    }

    #[test]
    fn test_train_blending() {
        let mut manager = EnsembleManager::default();

        let train_preds = vec![vec![
            ModelPrediction::new("m1".to_string(), 0.1, 0.8),
            ModelPrediction::new("m2".to_string(), 0.2, 0.9),
        ]];
        let targets = vec![0.15];

        manager.train_blending(train_preds, targets).unwrap();
        assert!(manager.is_blending_trained());
    }

    #[test]
    fn test_auto_strategy() {
        let mut manager = EnsembleManager::default();
        manager.set_strategy(ManagerStrategy::Auto);

        let predictions = vec![
            ModelPrediction::new("m1".to_string(), 0.5, 0.8),
            ModelPrediction::new("m2".to_string(), 0.6, 0.9),
        ];

        // Should fall back to voting when no trained ensembles
        let result = manager.predict(predictions);
        assert!(result.is_some());
    }

    #[test]
    fn test_performance_tracking() {
        let manager = EnsembleManager::default();

        manager.record_performance("m1", true, 0.01, 0.8, 1000);
        manager.record_performance("m1", true, 0.02, 0.9, 1100);

        let performances = manager.get_performances();
        assert!(!performances.is_empty());
    }

    #[test]
    fn test_top_models() {
        let manager = EnsembleManager::default();

        manager.record_performance("m1", true, 0.01, 0.8, 1000);
        manager.record_performance("m1", true, 0.01, 0.8, 1000);
        manager.record_performance("m2", true, 0.01, 0.8, 1000);
        manager.record_performance("m2", false, -0.01, 0.7, 1000);

        let top = manager.get_top_models(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].model_id, "m1");
    }
}
