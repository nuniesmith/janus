//! # ML Model Inference Module (Burn-native)
//!
//! Loads and runs the Burn `LstmPredictor` checkpoints the **backward** training
//! service produces and uses them for live trading-signal generation.
//!
//! ## Why Burn (and not ONNX)
//!
//! The backward trainer (`services/backward`) trains a double-DQN over an LSTM
//! and saves the inference network with [`janus_ml::models::LstmPredictor::save`]
//! — a `postcard`-encoded `ModelRecord` (`config` + weights). The previous
//! version of this module used `tract-onnx`, which **cannot load those Burn
//! checkpoints** — so the train → checkpoint → reload path was a dead end. This
//! module loads the exact same `LstmPredictor` type via the shared `jflow-ml`
//! crate, so a checkpoint written by `backward` is loadable here by construction.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    Model Inference Pipeline (Burn)                 │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  FeatureVector → Tensor[1,1,N] → LstmPredictor → Q-values → Signal │
//! │     (f32)          (CpuBackend)     forward()      softmax   Buy/…  │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use janus_forward::inference::{ModelInference, ModelCache};
//! use janus_forward::features::FeatureVector;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let cache = ModelCache::new();
//! let inference = ModelInference::new(cache);
//!
//! // Load a checkpoint the backward trainer wrote (latest_model.bin).
//! inference.load_model("signal_classifier", "checkpoints/backward/latest_model.bin").await?;
//!
//! let features = FeatureVector::new(
//!     vec!["rsi".to_string(), "macd".to_string()],
//!     vec![45.0, 10.0],
//! );
//!
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

use janus_ml::backend::{BackendDevice, CpuBackend};
use janus_ml::models::LstmPredictor;
use janus_ml::tensor::{Backend, Tensor, TensorData};

/// A cached, ready-to-run inference network.
///
/// Wrapped in a `Mutex` because Burn's `Param` holds a `!Sync`
/// `OnceCell<Tensor>` (lazy materialisation), so a model can't be shared by
/// shared reference across threads. The mutex makes the cache `Sync`
/// (`Mutex<T>: Sync` whenever `T: Send`) and serialises the interior mutation
/// the forward pass performs. Forward is fast/CPU-bound and the live loop runs
/// one symbol at a time, so contention is a non-issue.
type CachedModel = Arc<std::sync::Mutex<LstmPredictor<CpuBackend>>>;

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

/// Prediction result from the ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Predicted signal type
    pub signal_type: SignalType,

    /// Prediction confidence (0.0 - 1.0)
    pub confidence: f64,

    /// Per-class probabilities (softmax over the model's raw outputs)
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

/// Cache of loaded Burn inference models, keyed by name.
pub struct ModelCache {
    models: Arc<RwLock<HashMap<String, CachedModel>>>,
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

    /// Load and cache a Burn `LstmPredictor` checkpoint.
    ///
    /// `path` must point to a checkpoint written by
    /// [`janus_ml::models::LstmPredictor::save`] (e.g. `backward`'s
    /// `latest_model.bin`). The architecture is read from the checkpoint's
    /// embedded config, so the caller does not need to know it in advance.
    pub async fn load_model(&self, name: &str, path: &Path) -> Result<()> {
        info!("Loading Burn LSTM model '{}' from {:?}", name, path);

        let model = LstmPredictor::<CpuBackend>::load(path, BackendDevice::cpu())
            .map_err(|e| anyhow!("Failed to load Burn model from {:?}: {}", path, e))?;

        let mut models = self.models.write().await;

        // Check cache size limit (simple eviction; could be improved with LRU).
        if models.len() >= self.config.max_cached_models && !models.contains_key(name) {
            warn!(
                "Model cache full ({}/{}), evicting oldest model",
                models.len(),
                self.config.max_cached_models
            );
            if let Some(key) = models.keys().next().cloned() {
                models.remove(&key);
            }
        }

        models.insert(name.to_string(), Arc::new(std::sync::Mutex::new(model)));
        info!("Model '{}' loaded and cached successfully", name);

        Ok(())
    }

    /// Get a cached model (cheap `Arc` clone).
    pub async fn get_model(&self, name: &str) -> Result<CachedModel> {
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

/// Burn inference engine over a [`ModelCache`].
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

    /// Run inference on a feature vector.
    pub async fn predict(
        &self,
        model_name: &str,
        features: &FeatureVector,
    ) -> Result<PredictionResult> {
        let start = std::time::Instant::now();

        let model = self.cache.get_model(model_name).await?;

        debug!("Running Burn inference with model '{}'", model_name);
        let raw = run_forward(&model, features)?;
        let prediction = self.parse_scores(raw, model_name)?;

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

    /// Run batch inference.
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
        // inference (stacking features into one [N, 1, D] tensor) is a
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

    /// Turn raw model outputs (Q-values / logits) into a [`PredictionResult`].
    fn parse_scores(&self, raw: Vec<f64>, model_name: &str) -> Result<PredictionResult> {
        if raw.is_empty() {
            return Err(anyhow!("Model output has no scores"));
        }

        // Softmax the raw outputs into a probability distribution over actions.
        let scores = self.softmax(&raw);

        let (max_idx, &max_score) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .ok_or_else(|| anyhow!("Failed to find max score"))?;

        // Map class index to signal type (matches the trainer's action order:
        // 0 = Buy, 1 = Sell, everything else = Hold).
        let signal_type = match max_idx {
            0 => SignalType::Buy,
            1 => SignalType::Sell,
            _ => SignalType::Hold,
        };

        Ok(PredictionResult {
            signal_type,
            confidence: max_score,
            scores,
            latency_us: 0, // set by the caller
            model_name: model_name.to_string(),
        })
    }

    /// Apply softmax to scores
    fn softmax(&self, scores: &[f64]) -> Vec<f64> {
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();

        if sum == 0.0 {
            return vec![1.0 / scores.len() as f64; scores.len()];
        }
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

/// Feed a [`FeatureVector`] through a cached model and return the raw outputs.
///
/// The feature vector is shaped as a `[1, 1, N]` tensor (batch 1, sequence 1,
/// `N` features) — the same single-step layout the backward trainer uses for
/// its inference forward pass.
fn run_forward(
    model: &std::sync::Mutex<LstmPredictor<CpuBackend>>,
    features: &FeatureVector,
) -> Result<Vec<f64>> {
    let values = features.to_array();
    let n = values.len();
    if n == 0 {
        return Err(anyhow!("empty feature vector"));
    }

    // Serialise access (see `CachedModel`): the guard is held only for this
    // synchronous forward pass and is dropped before `predict` awaits anything.
    let model = model
        .lock()
        .map_err(|e| anyhow!("model mutex poisoned: {e}"))?;

    let expected = model.config().input_size;
    if n != expected {
        return Err(anyhow!(
            "feature count {} does not match model input_size {}",
            n,
            expected
        ));
    }

    let device = <CpuBackend as Backend>::Device::default();
    let input = Tensor::<CpuBackend, 3>::from_data(TensorData::new(values, [1, 1, n]), &device);

    let output = model
        .forward(input)
        .map_err(|e| anyhow!("Inference forward pass failed: {}", e))?;

    let raw: Vec<f32> = output
        .to_data()
        .to_vec()
        .map_err(|e| anyhow!("Failed to read model output tensor: {:?}", e))?;

    Ok(raw.into_iter().map(|v| v as f64).collect())
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
        self.total_latency_us
            .checked_div(self.total_inferences)
            .unwrap_or(0)
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
    use janus_ml::models::LstmConfig;

    /// Build a tiny inference model, save it, and return the temp path.
    fn write_temp_model(input_size: usize, output_size: usize) -> std::path::PathBuf {
        let model = LstmPredictor::<CpuBackend>::new(
            LstmConfig::new(input_size, 8, output_size),
            BackendDevice::cpu(),
        );
        let path = std::env::temp_dir().join(format!(
            "janus_fwd_inference_test_{}_{}.bin",
            std::process::id(),
            // unique-per-call suffix so parallel tests don't collide
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        model.save(&path).expect("save model checkpoint");
        path
    }

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
        assert_eq!(inference.config.model_dir, "models");
    }

    #[tokio::test]
    async fn test_model_inference_softmax() {
        let cache = ModelCache::new();
        let inference = ModelInference::new(cache);

        let scores = vec![1.0, 2.0, 3.0];
        let result = inference.softmax(&scores);

        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    /// End-to-end: a checkpoint written via `LstmPredictor::save` (the same
    /// path the backward trainer uses) loads here and produces a sane signal.
    #[tokio::test]
    async fn test_burn_roundtrip_load_and_predict() {
        let input_size = 4;
        let path = write_temp_model(input_size, 3);

        let cache = ModelCache::new();
        cache
            .load_model("signal_classifier", &path)
            .await
            .expect("load checkpoint");
        assert!(cache.has_model("signal_classifier").await);

        let inference = ModelInference::new(cache);
        let features = FeatureVector::new(
            (0..input_size).map(|i| format!("f{i}")).collect(),
            vec![0.1, 0.2, 0.3, 0.4],
        );

        let pred = inference
            .predict("signal_classifier", &features)
            .await
            .expect("predict");

        assert_eq!(pred.scores.len(), 3, "three action classes");
        assert!((0.0..=1.0).contains(&pred.confidence));
        let prob_sum: f64 = pred.scores.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-6, "softmaxed scores sum to 1");
        assert!(matches!(
            pred.signal_type,
            SignalType::Buy | SignalType::Sell | SignalType::Hold
        ));

        let metrics = inference.metrics().await;
        assert_eq!(metrics.total_inferences, 1);

        let _ = std::fs::remove_file(&path);
    }

    /// A feature vector whose length doesn't match the model's `input_size`
    /// is rejected with a clear error (rather than panicking deep in Burn).
    #[tokio::test]
    async fn test_predict_rejects_feature_size_mismatch() {
        let path = write_temp_model(4, 3);

        let cache = ModelCache::new();
        cache.load_model("m", &path).await.expect("load");
        let inference = ModelInference::new(cache);

        let wrong = FeatureVector::new(
            vec!["a".to_string(), "b".to_string()],
            vec![1.0, 2.0], // 2 features, model expects 4
        );
        let err = inference.predict("m", &wrong).await.unwrap_err();
        assert!(err.to_string().contains("input_size"));

        let _ = std::fs::remove_file(&path);
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
}
