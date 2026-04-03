//! # ML Model Inference Module
//!
//! Provides ONNX-based machine learning model inference for trading signal generation.
//!
//! ## Overview
//!
//! This module handles:
//! - ONNX model loading and caching
//! - Feature vector to tensor conversion
//! - Model inference execution
//! - Prediction result parsing
//! - Performance metrics and monitoring
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                  Model Inference Pipeline                │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                           │
//! │  FeatureVector  →  Tensor  →  Model  →  Output  →  Result│
//! │                                                           │
//! │  ┌──────────┐   ┌────────┐   ┌─────┐   ┌──────┐        │
//! │  │ Features │ → │ Tract  │ → │ONNX │ → │Parse │ → Signal│
//! │  │  (f32)   │   │Tensor  │   │Model│   │Result│         │
//! │  └──────────┘   └────────┘   └─────┘   └──────┘        │
//! │                                                           │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use janus_forward::inference::{ModelInference, ModelCache};
//! use janus::features::FeatureVector;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let cache = ModelCache::new();
//! let inference = ModelInference::new(cache);
//!
//! // Load model
//! inference.load_model("signal_classifier", "models/signal.onnx").await?;
//!
//! // Create feature vector
//! let features = FeatureVector::new(
//!     vec!["rsi".to_string(), "macd".to_string()],
//!     vec![45.0, 10.0],
//! );
//!
//! // Run inference
//! let prediction = inference.predict("signal_classifier", &features).await?;
//! println!("Prediction: {:?}", prediction.signal_type);
//! # Ok(())
//! # }
//! ```

use crate::features::FeatureVector;
use crate::signal::SignalType;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use tract_onnx::prelude::*;

/// Model inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Model directory path
    pub model_dir: String,

    /// Enable model caching
    pub enable_cache: bool,

    /// Maximum cached models
    pub max_cached_models: usize,

    /// Inference timeout (milliseconds)
    pub inference_timeout_ms: u64,

    /// Minimum confidence threshold
    pub min_confidence: f64,

    /// Batch size for batch inference
    pub batch_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_dir: "models".to_string(),
            enable_cache: true,
            max_cached_models: 10,
            inference_timeout_ms: 100,
            min_confidence: 0.5,
            batch_size: 32,
        }
    }
}

/// Prediction result from ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Predicted signal type
    pub signal_type: SignalType,

    /// Prediction confidence (0.0 - 1.0)
    pub confidence: f64,

    /// Raw model output scores
    pub scores: Vec<f64>,

    /// Inference latency (microseconds)
    pub latency_us: u64,

    /// Model version/name
    pub model_name: String,
}

impl PredictionResult {
    /// Check if prediction meets confidence threshold
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }

    /// Get prediction class index
    pub fn class_index(&self) -> usize {
        self.scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

/// Type alias for ONNX model plan
type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Model cache for storing loaded ONNX models
pub struct ModelCache {
    models: Arc<RwLock<HashMap<String, OnnxModel>>>,
    config: InferenceConfig,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new() -> Self {
        Self::with_config(InferenceConfig::default())
    }

    /// Create a new model cache with configuration
    pub fn with_config(config: InferenceConfig) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Load and cache a model
    pub async fn load_model(&self, name: &str, path: &Path) -> Result<()> {
        info!("Loading ONNX model '{}' from {:?}", name, path);

        // Load ONNX model using tract
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?
            .into_optimized()
            .map_err(|e| anyhow!("Failed to optimize model: {}", e))?
            .into_runnable()
            .map_err(|e| anyhow!("Failed to create runnable model: {}", e))?;

        // Cache the model
        let mut models = self.models.write().await;

        // Check cache size limit
        if models.len() >= self.config.max_cached_models && !models.contains_key(name) {
            warn!(
                "Model cache full ({}/{}), evicting oldest model",
                models.len(),
                self.config.max_cached_models
            );
            // Simple eviction: remove first entry (could be improved with LRU)
            if let Some(key) = models.keys().next().cloned() {
                models.remove(&key);
            }
        }

        models.insert(name.to_string(), model);
        info!("Model '{}' loaded and cached successfully", name);

        Ok(())
    }

    /// Get a cached model
    pub async fn get_model(
        &self,
        name: &str,
    ) -> Result<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> {
        let models = self.models.read().await;
        models
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Model '{}' not found in cache", name))
    }

    /// Check if model is cached
    pub async fn has_model(&self, name: &str) -> bool {
        let models = self.models.read().await;
        models.contains_key(name)
    }

    /// Clear all cached models
    pub async fn clear(&self) {
        let mut models = self.models.write().await;
        models.clear();
        info!("Model cache cleared");
    }

    /// Get number of cached models
    pub async fn len(&self) -> usize {
        let models = self.models.read().await;
        models.len()
    }

    /// Check if cache is empty
    pub async fn is_empty(&self) -> bool {
        let models = self.models.read().await;
        models.is_empty()
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Model inference engine
pub struct ModelInference {
    cache: Arc<ModelCache>,
    #[allow(dead_code)]
    config: InferenceConfig,
    metrics: Arc<RwLock<ModelMetrics>>,
}

impl ModelInference {
    /// Create a new inference engine
    pub fn new(cache: ModelCache) -> Self {
        Self::with_config(cache, InferenceConfig::default())
    }

    /// Create a new inference engine with configuration
    pub fn with_config(cache: ModelCache, config: InferenceConfig) -> Self {
        Self {
            cache: Arc::new(cache),
            config,
            metrics: Arc::new(RwLock::new(ModelMetrics::default())),
        }
    }

    /// Load a model from file
    pub async fn load_model(&self, name: &str, path: impl AsRef<Path>) -> Result<()> {
        self.cache.load_model(name, path.as_ref()).await
    }

    /// Run inference on a feature vector
    pub async fn predict(
        &self,
        model_name: &str,
        features: &FeatureVector,
    ) -> Result<PredictionResult> {
        let start = std::time::Instant::now();

        // Get model from cache
        let model = self.cache.get_model(model_name).await?;

        // Convert features to tensor
        let input_tensor = self.features_to_tensor(features)?;

        // Run inference
        debug!("Running inference with model '{}'", model_name);
        let output = model
            .run(tvec![input_tensor.into()])
            .map_err(|e| anyhow!("Inference failed: {}", e))?;

        // Parse output
        let prediction = self.parse_output(output, model_name)?;

        // Record metrics
        let latency_us = start.elapsed().as_micros() as u64;
        {
            let mut metrics = self.metrics.write().await;
            metrics.record_inference(latency_us, prediction.confidence);
        }

        Ok(PredictionResult {
            latency_us,
            ..prediction
        })
    }

    /// Run batch inference
    pub async fn predict_batch(
        &self,
        model_name: &str,
        features: &[FeatureVector],
    ) -> Result<Vec<PredictionResult>> {
        let start = std::time::Instant::now();

        if features.is_empty() {
            return Ok(Vec::new());
        }

        // Sequential inference is functionally correct. True batched tensor
        // inference (concatenating features into a single [N, D] tensor) is a
        // throughput optimization for when batch sizes regularly exceed ~16.
        let mut results = Vec::new();
        for feature_vec in features {
            let result = self.predict(model_name, feature_vec).await?;
            results.push(result);
        }

        let latency_us = start.elapsed().as_micros() as u64;
        debug!(
            "Batch inference completed: {} predictions in {}μs",
            results.len(),
            latency_us
        );

        Ok(results)
    }

    /// Convert feature vector to tensor
    fn features_to_tensor(&self, features: &FeatureVector) -> Result<Tensor> {
        let values = features.to_array();
        let shape = &[1, values.len()];

        Tensor::from_shape(shape, &values).map_err(|e| anyhow!("Failed to create tensor: {}", e))
    }

    /// Parse model output to prediction result
    fn parse_output(&self, output: TVec<TValue>, model_name: &str) -> Result<PredictionResult> {
        if output.is_empty() {
            return Err(anyhow!("Model output is empty"));
        }

        // Get first output tensor
        let output_tensor = output[0]
            .to_array_view::<f32>()
            .map_err(|e| anyhow!("Failed to convert output to array: {}", e))?;

        // Get scores from tensor view
        let output_data = &output_tensor;

        // Get scores (assume output is [batch_size, num_classes])
        let scores: Vec<f64> = output_data.iter().map(|&v| v as f64).collect();

        if scores.is_empty() {
            return Err(anyhow!("Model output has no scores"));
        }

        // Apply softmax to get probabilities
        let scores = self.softmax(&scores);

        // Find max score and corresponding class
        let (max_idx, &max_score) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .ok_or_else(|| anyhow!("Failed to find max score"))?;

        // Map class index to signal type
        let signal_type = match max_idx {
            0 => SignalType::Buy,
            1 => SignalType::Sell,
            _ => SignalType::Hold,
        };

        Ok(PredictionResult {
            signal_type,
            confidence: max_score,
            scores,
            latency_us: 0, // Will be set by caller
            model_name: model_name.to_string(),
        })
    }

    /// Apply softmax to scores
    fn softmax(&self, scores: &[f64]) -> Vec<f64> {
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();

        exp_scores.iter().map(|&s| s / sum).collect()
    }

    /// Get inference metrics
    pub async fn metrics(&self) -> ModelMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = ModelMetrics::default();
    }
}

/// Model inference metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Total inferences executed
    pub total_inferences: u64,

    /// Total inference time (microseconds)
    pub total_latency_us: u64,

    /// Minimum latency (microseconds)
    pub min_latency_us: u64,

    /// Maximum latency (microseconds)
    pub max_latency_us: u64,

    /// Average confidence
    pub avg_confidence: f64,

    /// Confidence sum (for calculating average)
    confidence_sum: f64,
}

impl ModelMetrics {
    /// Record an inference
    pub fn record_inference(&mut self, latency_us: u64, confidence: f64) {
        self.total_inferences += 1;
        self.total_latency_us += latency_us;

        if self.min_latency_us == 0 || latency_us < self.min_latency_us {
            self.min_latency_us = latency_us;
        }

        if latency_us > self.max_latency_us {
            self.max_latency_us = latency_us;
        }

        self.confidence_sum += confidence;
        self.avg_confidence = self.confidence_sum / self.total_inferences as f64;
    }

    /// Get average latency (microseconds)
    pub fn avg_latency_us(&self) -> u64 {
        if self.total_inferences == 0 {
            0
        } else {
            self.total_latency_us / self.total_inferences
        }
    }

    /// Get P50 latency estimate (simplified)
    pub fn p50_latency_us(&self) -> u64 {
        self.avg_latency_us()
    }

    /// Get P99 latency estimate (simplified - uses max for now)
    pub fn p99_latency_us(&self) -> u64 {
        self.max_latency_us
    }

    /// Get inferences per second
    pub fn inferences_per_second(&self, duration_secs: f64) -> f64 {
        if duration_secs > 0.0 {
            self.total_inferences as f64 / duration_secs
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.model_dir, "models");
        assert!(config.enable_cache);
        assert_eq!(config.max_cached_models, 10);
    }

    #[test]
    fn test_prediction_result_is_confident() {
        let result = PredictionResult {
            signal_type: SignalType::Buy,
            confidence: 0.8,
            scores: vec![0.8, 0.1, 0.1],
            latency_us: 1000,
            model_name: "test".to_string(),
        };

        assert!(result.is_confident(0.5));
        assert!(result.is_confident(0.8));
        assert!(!result.is_confident(0.9));
    }

    #[test]
    fn test_prediction_result_class_index() {
        let result = PredictionResult {
            signal_type: SignalType::Buy,
            confidence: 0.8,
            scores: vec![0.1, 0.8, 0.1],
            latency_us: 1000,
            model_name: "test".to_string(),
        };

        assert_eq!(result.class_index(), 1);
    }

    #[tokio::test]
    async fn test_model_cache_creation() {
        let cache = ModelCache::new();
        assert!(cache.is_empty().await);
        assert_eq!(cache.len().await, 0);
    }

    #[tokio::test]
    async fn test_model_cache_has_model() {
        let cache = ModelCache::new();
        assert!(!cache.has_model("test_model").await);
    }

    #[tokio::test]
    async fn test_model_cache_clear() {
        let cache = ModelCache::new();
        cache.clear().await;
        assert!(cache.is_empty().await);
    }

    #[test]
    fn test_model_inference_creation() {
        let cache = ModelCache::new();
        let inference = ModelInference::new(cache);
        // Just verify it creates without panicking
        assert_eq!(inference.config.model_dir, "models");
    }

    #[tokio::test]
    async fn test_model_inference_softmax() {
        let cache = ModelCache::new();
        let inference = ModelInference::new(cache);

        let scores = vec![1.0, 2.0, 3.0];
        let result = inference.softmax(&scores);

        // Check that probabilities sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that highest input gets highest probability
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_model_metrics_default() {
        let metrics = ModelMetrics::default();
        assert_eq!(metrics.total_inferences, 0);
        assert_eq!(metrics.total_latency_us, 0);
        assert_eq!(metrics.avg_confidence, 0.0);
    }

    #[test]
    fn test_model_metrics_record_inference() {
        let mut metrics = ModelMetrics::default();

        metrics.record_inference(1000, 0.8);
        assert_eq!(metrics.total_inferences, 1);
        assert_eq!(metrics.total_latency_us, 1000);
        assert_eq!(metrics.min_latency_us, 1000);
        assert_eq!(metrics.max_latency_us, 1000);
        assert_eq!(metrics.avg_confidence, 0.8);

        metrics.record_inference(2000, 0.9);
        assert_eq!(metrics.total_inferences, 2);
        assert_eq!(metrics.total_latency_us, 3000);
        assert_eq!(metrics.min_latency_us, 1000);
        assert_eq!(metrics.max_latency_us, 2000);
        assert!((metrics.avg_confidence - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_model_metrics_avg_latency() {
        let mut metrics = ModelMetrics::default();

        metrics.record_inference(1000, 0.8);
        metrics.record_inference(2000, 0.9);
        metrics.record_inference(3000, 0.7);

        assert_eq!(metrics.avg_latency_us(), 2000);
    }

    #[test]
    fn test_model_metrics_inferences_per_second() {
        let mut metrics = ModelMetrics::default();

        metrics.record_inference(1000, 0.8);
        metrics.record_inference(2000, 0.9);

        let ips = metrics.inferences_per_second(2.0);
        assert_eq!(ips, 1.0);
    }

    #[tokio::test]
    async fn test_inference_metrics_integration() {
        let cache = ModelCache::new();
        let inference = ModelInference::new(cache);

        let metrics = inference.metrics().await;
        assert_eq!(metrics.total_inferences, 0);

        inference.reset_metrics().await;
        let metrics = inference.metrics().await;
        assert_eq!(metrics.total_inferences, 0);
    }

    #[test]
    fn test_features_to_tensor() {
        let cache = ModelCache::new();
        let inference = ModelInference::new(cache);

        let features = FeatureVector::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![1.0, 2.0, 3.0],
        );

        let result = inference.features_to_tensor(&features);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[1, 3]);
    }
}
