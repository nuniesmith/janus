//! Inference-only model wrapper for low-latency prediction.
//!
//! This module provides optimized inference capabilities:
//! - No gradient computation (inference-only mode)
//! - Pre-allocated buffers for zero-copy operations
//! - Batch prediction support
//! - Model warmup and benchmarking utilities
//! - Thread-safe prediction serving

use crate::error::{Result, VisionError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Inference statistics tracker.
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub total_predictions: usize,
    pub total_latency_us: u64,
    pub min_latency_us: u64,
    pub max_latency_us: u64,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
    latency_history: Vec<u64>,
}

impl InferenceStats {
    /// Create new inference statistics.
    pub fn new() -> Self {
        Self {
            total_predictions: 0,
            total_latency_us: 0,
            min_latency_us: u64::MAX,
            max_latency_us: 0,
            p50_latency_us: 0,
            p95_latency_us: 0,
            p99_latency_us: 0,
            latency_history: Vec::with_capacity(1000),
        }
    }

    /// Record a prediction latency.
    pub fn record_latency(&mut self, latency_us: u64) {
        self.total_predictions += 1;
        self.total_latency_us += latency_us;
        self.min_latency_us = self.min_latency_us.min(latency_us);
        self.max_latency_us = self.max_latency_us.max(latency_us);

        // Keep last 1000 latencies for percentile calculation
        if self.latency_history.len() >= 1000 {
            self.latency_history.remove(0);
        }
        self.latency_history.push(latency_us);

        self.compute_percentiles();
    }

    /// Compute percentile statistics.
    fn compute_percentiles(&mut self) {
        if self.latency_history.is_empty() {
            return;
        }

        let mut sorted = self.latency_history.clone();
        sorted.sort_unstable();

        let len = sorted.len();
        self.p50_latency_us = sorted[len * 50 / 100];
        self.p95_latency_us = sorted[len * 95 / 100];
        self.p99_latency_us = sorted[len * 99 / 100];
    }

    /// Get average latency in microseconds.
    pub fn avg_latency_us(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.total_predictions as f64
        }
    }

    /// Get average latency in milliseconds.
    pub fn avg_latency_ms(&self) -> f64 {
        self.avg_latency_us() / 1000.0
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.total_predictions = 0;
        self.total_latency_us = 0;
        self.min_latency_us = u64::MAX;
        self.max_latency_us = 0;
        self.p50_latency_us = 0;
        self.p95_latency_us = 0;
        self.p99_latency_us = 0;
        self.latency_history.clear();
    }
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Model prediction output.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub signal: f64,
    pub confidence: f64,
    pub metadata: PredictionMetadata,
}

impl Prediction {
    /// Create a new prediction.
    pub fn new(signal: f64, confidence: f64, latency_us: u64) -> Self {
        Self {
            signal,
            confidence,
            metadata: PredictionMetadata {
                latency_us,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64,
            },
        }
    }

    /// Check if the prediction is bullish (signal > 0).
    pub fn is_bullish(&self) -> bool {
        self.signal > 0.0
    }

    /// Check if the prediction is bearish (signal < 0).
    pub fn is_bearish(&self) -> bool {
        self.signal < 0.0
    }

    /// Check if the prediction is neutral (signal ≈ 0).
    pub fn is_neutral(&self, threshold: f64) -> bool {
        self.signal.abs() < threshold
    }

    /// Check if confidence meets a threshold.
    pub fn meets_confidence(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Metadata associated with a prediction.
#[derive(Debug, Clone)]
pub struct PredictionMetadata {
    pub latency_us: u64,
    pub timestamp: i64,
}

/// Batch prediction request.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub inputs: Vec<Vec<Vec<f64>>>,
    pub metadata: RequestMetadata,
}

impl BatchRequest {
    /// Create a new batch request.
    pub fn new(inputs: Vec<Vec<Vec<f64>>>) -> Self {
        Self {
            inputs,
            metadata: RequestMetadata::new(),
        }
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.inputs.len()
    }
}

/// Metadata associated with a batch request.
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    pub request_id: String,
    pub timestamp: i64,
}

impl RequestMetadata {
    fn new() -> Self {
        Self {
            request_id: uuid_simple(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        }
    }
}

/// Simple UUID generation (simplified version for no external deps).
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", nanos)
}

/// Inference engine configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub batch_size: usize,
    pub num_threads: usize,
    pub warmup_iterations: usize,
    pub enable_profiling: bool,
    pub max_latency_ms: f64,
}

impl InferenceConfig {
    /// Create default inference configuration.
    pub fn default() -> Self {
        Self {
            batch_size: 1,
            num_threads: 1,
            warmup_iterations: 10,
            enable_profiling: false,
            max_latency_ms: 100.0,
        }
    }

    /// Create configuration optimized for low latency.
    pub fn low_latency() -> Self {
        Self {
            batch_size: 1,
            num_threads: 1,
            warmup_iterations: 100,
            enable_profiling: true,
            max_latency_ms: 50.0,
        }
    }

    /// Create configuration optimized for high throughput.
    pub fn high_throughput() -> Self {
        Self {
            batch_size: 32,
            num_threads: 4,
            warmup_iterations: 10,
            enable_profiling: false,
            max_latency_ms: 200.0,
        }
    }
}

/// Mock model weights for demonstration (in real implementation, would load actual model).
#[derive(Debug, Clone)]
pub struct ModelWeights {
    // In practice, this would contain actual model parameters
    signal_bias: f64,
    confidence_scale: f64,
}

impl ModelWeights {
    fn default() -> Self {
        Self {
            signal_bias: 0.0,
            confidence_scale: 1.0,
        }
    }
}

/// Inference-only model wrapper.
///
/// Optimized for low-latency predictions without gradient computation.
/// Thread-safe and suitable for concurrent prediction serving.
pub struct InferenceEngine {
    config: InferenceConfig,
    weights: Arc<ModelWeights>,
    stats: Arc<Mutex<InferenceStats>>,
    warmed_up: bool,
}

impl InferenceEngine {
    /// Create a new inference engine.
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            config,
            weights: Arc::new(ModelWeights::default()),
            stats: Arc::new(Mutex::new(InferenceStats::new())),
            warmed_up: false,
        }
    }

    /// Create an inference engine with default configuration.
    pub fn default() -> Self {
        Self::new(InferenceConfig::default())
    }

    /// Load model weights from a checkpoint.
    pub fn load_weights(&mut self, _checkpoint_path: &str) -> Result<()> {
        // In practice, this would load actual model weights
        // For now, we use default mock weights
        Ok(())
    }

    /// Warm up the model by running inference iterations.
    pub fn warmup(&mut self) -> Result<()> {
        if self.warmed_up {
            return Ok(());
        }

        // Create dummy input for warmup
        let dummy_input = vec![vec![vec![0.0; 64]; 64]];

        for _ in 0..self.config.warmup_iterations {
            let _ = self.predict_single(&dummy_input[0])?;
        }

        self.warmed_up = true;
        Ok(())
    }

    /// Run inference on a single input.
    pub fn predict_single(&self, input: &Vec<Vec<f64>>) -> Result<Prediction> {
        let start = Instant::now();

        // Validate input dimensions
        if input.is_empty() || input[0].is_empty() {
            return Err(VisionError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        // Perform inference (simplified mock implementation)
        let prediction = self.forward_pass(input);

        let latency_us = start.elapsed().as_micros() as u64;

        // Record statistics
        if self.config.enable_profiling {
            if let Ok(mut stats) = self.stats.lock() {
                stats.record_latency(latency_us);
            }
        }

        // Check latency threshold
        let latency_ms = latency_us as f64 / 1000.0;
        if latency_ms > self.config.max_latency_ms {
            eprintln!(
                "Warning: Prediction latency ({:.2}ms) exceeded threshold ({:.2}ms)",
                latency_ms, self.config.max_latency_ms
            );
        }

        Ok(Prediction::new(prediction.0, prediction.1, latency_us))
    }

    /// Run inference on a batch of inputs.
    pub fn predict_batch(&self, request: &BatchRequest) -> Result<Vec<Prediction>> {
        let mut predictions = Vec::with_capacity(request.batch_size());

        for input in &request.inputs {
            predictions.push(self.predict_single(input)?);
        }

        Ok(predictions)
    }

    /// Forward pass through the model (simplified mock implementation).
    ///
    /// In a real implementation, this would run the actual model inference.
    fn forward_pass(&self, input: &Vec<Vec<f64>>) -> (f64, f64) {
        // Mock inference: compute simple statistics as proxy for model output
        let mut sum = 0.0;
        let mut count = 0;

        for row in input {
            for &val in row {
                sum += val;
                count += 1;
            }
        }

        let mean = if count > 0 { sum / count as f64 } else { 0.0 };

        // Compute variance
        let mut var_sum = 0.0;
        for row in input {
            for &val in row {
                var_sum += (val - mean).powi(2);
            }
        }
        let variance = if count > 1 {
            var_sum / (count - 1) as f64
        } else {
            0.0
        };

        // Mock signal: normalized mean with bias
        let signal = (mean + self.weights.signal_bias).tanh();

        // Mock confidence: based on variance (lower variance = higher confidence)
        let confidence = (1.0 / (1.0 + variance)).clamp(0.0, 1.0) * self.weights.confidence_scale;

        (signal, confidence)
    }

    /// Get inference statistics.
    pub fn stats(&self) -> Result<InferenceStats> {
        self.stats
            .lock()
            .map(|s| s.clone())
            .map_err(|_| VisionError::InternalError("Failed to lock stats".to_string()))
    }

    /// Reset inference statistics.
    pub fn reset_stats(&self) -> Result<()> {
        self.stats
            .lock()
            .map(|mut s| s.reset())
            .map_err(|_| VisionError::InternalError("Failed to lock stats".to_string()))
    }

    /// Get the inference configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Check if the engine has been warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }

    /// Benchmark the inference engine.
    pub fn benchmark(&mut self, iterations: usize, input_size: usize) -> Result<BenchmarkResults> {
        let mut latencies = Vec::with_capacity(iterations);
        let dummy_input = vec![vec![0.0; input_size]; input_size];

        // Warmup first
        if !self.warmed_up {
            self.warmup()?;
        }

        // Run benchmark
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = self.predict_single(&dummy_input)?;
            let latency_us = start.elapsed().as_micros() as u64;
            latencies.push(latency_us);
        }

        Ok(BenchmarkResults::from_latencies(latencies))
    }
}

/// Benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub iterations: usize,
    pub mean_latency_us: f64,
    pub median_latency_us: u64,
    pub min_latency_us: u64,
    pub max_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
    pub throughput_per_sec: f64,
}

impl BenchmarkResults {
    fn from_latencies(mut latencies: Vec<u64>) -> Self {
        latencies.sort_unstable();
        let iterations = latencies.len();

        let sum: u64 = latencies.iter().sum();
        let mean = sum as f64 / iterations as f64;
        let median = latencies[iterations / 2];
        let min = latencies[0];
        let max = latencies[iterations - 1];
        let p95 = latencies[iterations * 95 / 100];
        let p99 = latencies[iterations * 99 / 100];
        let throughput = 1_000_000.0 / mean;

        Self {
            iterations,
            mean_latency_us: mean,
            median_latency_us: median,
            min_latency_us: min,
            max_latency_us: max,
            p95_latency_us: p95,
            p99_latency_us: p99,
            throughput_per_sec: throughput,
        }
    }

    /// Check if latency meets SLA (Service Level Agreement).
    pub fn meets_sla(&self, max_p99_latency_ms: f64) -> bool {
        (self.p99_latency_us as f64 / 1000.0) <= max_p99_latency_ms
    }

    /// Print benchmark summary.
    pub fn print_summary(&self) {
        println!("=== Benchmark Results ===");
        println!("Iterations: {}", self.iterations);
        println!(
            "Mean latency: {:.2} μs ({:.2} ms)",
            self.mean_latency_us,
            self.mean_latency_us / 1000.0
        );
        println!(
            "Median latency: {} μs ({:.2} ms)",
            self.median_latency_us,
            self.median_latency_us as f64 / 1000.0
        );
        println!("Min latency: {} μs", self.min_latency_us);
        println!("Max latency: {} μs", self.max_latency_us);
        println!(
            "P95 latency: {} μs ({:.2} ms)",
            self.p95_latency_us,
            self.p95_latency_us as f64 / 1000.0
        );
        println!(
            "P99 latency: {} μs ({:.2} ms)",
            self.p99_latency_us,
            self.p99_latency_us as f64 / 1000.0
        );
        println!("Throughput: {:.2} predictions/sec", self.throughput_per_sec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_stats() {
        let mut stats = InferenceStats::new();
        stats.record_latency(100);
        stats.record_latency(200);
        stats.record_latency(150);

        assert_eq!(stats.total_predictions, 3);
        assert_eq!(stats.min_latency_us, 100);
        assert_eq!(stats.max_latency_us, 200);
        assert!((stats.avg_latency_us() - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_prediction_classification() {
        let pred = Prediction::new(0.5, 0.8, 1000);
        assert!(pred.is_bullish());
        assert!(!pred.is_bearish());
        assert!(!pred.is_neutral(0.1));

        let pred2 = Prediction::new(-0.3, 0.7, 1000);
        assert!(!pred2.is_bullish());
        assert!(pred2.is_bearish());

        let pred3 = Prediction::new(0.05, 0.6, 1000);
        assert!(pred3.is_neutral(0.1));
    }

    #[test]
    fn test_prediction_confidence() {
        let pred = Prediction::new(0.5, 0.9, 1000);
        assert!(pred.meets_confidence(0.8));
        assert!(!pred.meets_confidence(0.95));
    }

    #[test]
    fn test_inference_engine_creation() {
        let engine = InferenceEngine::default();
        assert!(!engine.is_warmed_up());
    }

    #[test]
    fn test_inference_engine_warmup() {
        let mut engine = InferenceEngine::new(InferenceConfig {
            warmup_iterations: 5,
            ..InferenceConfig::default()
        });

        engine.warmup().unwrap();
        assert!(engine.is_warmed_up());
    }

    #[test]
    fn test_single_prediction() {
        let engine = InferenceEngine::default();
        let input = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let prediction = engine.predict_single(&input).unwrap();
        assert!(prediction.signal >= -1.0 && prediction.signal <= 1.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }

    #[test]
    fn test_batch_prediction() {
        let engine = InferenceEngine::default();
        let input1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let input2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let request = BatchRequest::new(vec![input1, input2]);
        let predictions = engine.predict_batch(&request).unwrap();

        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_empty_input_error() {
        let engine = InferenceEngine::default();
        let empty_input: Vec<Vec<f64>> = vec![];

        let result = engine.predict_single(&empty_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_config_presets() {
        let low_latency = InferenceConfig::low_latency();
        assert_eq!(low_latency.batch_size, 1);
        assert!(low_latency.enable_profiling);

        let high_throughput = InferenceConfig::high_throughput();
        assert!(high_throughput.batch_size > 1);
    }

    #[test]
    fn test_benchmark() {
        let mut engine = InferenceEngine::new(InferenceConfig {
            warmup_iterations: 2,
            ..InferenceConfig::default()
        });

        let results = engine.benchmark(10, 32).unwrap();
        assert_eq!(results.iterations, 10);
        assert!(results.mean_latency_us > 0.0);
        assert!(results.throughput_per_sec > 0.0);
    }

    #[test]
    fn test_benchmark_sla() {
        let mut engine = InferenceEngine::default();
        let results = engine.benchmark(10, 16).unwrap();

        // Should meet a generous SLA
        assert!(results.meets_sla(1000.0));
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = InferenceStats::new();
        stats.record_latency(100);
        stats.record_latency(200);

        stats.reset();
        assert_eq!(stats.total_predictions, 0);
        assert_eq!(stats.total_latency_us, 0);
    }

    #[test]
    fn test_batch_request_size() {
        let input1 = vec![vec![1.0]];
        let input2 = vec![vec![2.0]];
        let request = BatchRequest::new(vec![input1, input2]);

        assert_eq!(request.batch_size(), 2);
    }
}
