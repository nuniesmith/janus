//! Live pipeline for real-time market data processing and prediction.
//!
//! This module provides a complete end-to-end pipeline for live trading:
//! - Real-time feature computation with streaming updates
//! - Feature caching for low-latency serving
//! - Inference-only model execution
//! - Latency tracking and optimization
//! - Thread-safe concurrent processing
//!
//! # Architecture
//!
//! ```text
//! Market Data → Streaming Buffer → Feature Cache → Inference Engine → Predictions
//!                     ↓                   ↓              ↓
//!                 Incremental         LRU Cache      No Gradients
//!                 Updates             TTL-based      <100ms target
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use vision::live::{LivePipeline, LivePipelineConfig};
//!
//! let config = LivePipelineConfig::low_latency();
//! let mut pipeline = LivePipeline::new(config);
//!
//! // Initialize and warm up
//! pipeline.initialize()?;
//! pipeline.warmup()?;
//!
//! // Process live data
//! let prediction = pipeline.process_tick(market_data)?;
//!
//! if prediction.meets_confidence(0.8) {
//!     println!("Signal: {}, Confidence: {}", prediction.signal, prediction.confidence);
//! }
//! ```

pub mod cache;
pub mod inference;
pub mod latency;
pub mod streaming;

pub use cache::{CachedFeature, FeatureCache, FeatureCacheKey, FeatureType, LRUCache};
pub use inference::{
    BenchmarkResults, InferenceConfig, InferenceEngine, InferenceStats, Prediction,
};
pub use latency::{
    LatencyBudget, LatencyMeasurement, LatencyProfiler, LatencySummary, LatencyTracker,
};
pub use streaming::{
    CircularBuffer, IncrementalGAFComputer, MarketData, MultiTimeframeBuffer, RunningStats,
    StreamingFeatureBuffer,
};

use crate::error::{Result, VisionError};
use std::sync::{Arc, RwLock};

/// Configuration for the live pipeline.
#[derive(Debug, Clone)]
pub struct LivePipelineConfig {
    /// Window size for feature computation
    pub window_size: usize,
    /// Enable feature caching
    pub enable_cache: bool,
    /// Cache capacity
    pub cache_capacity: usize,
    /// Inference configuration
    pub inference_config: InferenceConfig,
    /// Target latency in milliseconds
    pub target_latency_ms: f64,
    /// Enable latency profiling
    pub enable_profiling: bool,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl LivePipelineConfig {
    /// Create default configuration.
    pub fn default() -> Self {
        Self {
            window_size: 60,
            enable_cache: true,
            cache_capacity: 100,
            inference_config: InferenceConfig::default(),
            target_latency_ms: 100.0,
            enable_profiling: false,
            warmup_iterations: 10,
        }
    }

    /// Create configuration optimized for low latency (<50ms).
    pub fn low_latency() -> Self {
        Self {
            window_size: 30,
            enable_cache: true,
            cache_capacity: 200,
            inference_config: InferenceConfig::low_latency(),
            target_latency_ms: 50.0,
            enable_profiling: true,
            warmup_iterations: 100,
        }
    }

    /// Create configuration optimized for high accuracy.
    pub fn high_accuracy() -> Self {
        Self {
            window_size: 120,
            enable_cache: true,
            cache_capacity: 50,
            inference_config: InferenceConfig::default(),
            target_latency_ms: 200.0,
            enable_profiling: false,
            warmup_iterations: 20,
        }
    }

    /// Create configuration for backtesting replay.
    pub fn backtest_replay() -> Self {
        Self {
            window_size: 60,
            enable_cache: false, // Don't cache in backtest
            cache_capacity: 0,
            inference_config: InferenceConfig::high_throughput(),
            target_latency_ms: 500.0,
            enable_profiling: true,
            warmup_iterations: 5,
        }
    }
}

/// Live pipeline for real-time prediction.
///
/// Integrates streaming feature computation, caching, and inference
/// into a single optimized pipeline for live trading.
pub struct LivePipeline {
    config: LivePipelineConfig,
    gaf_computer: IncrementalGAFComputer,
    cache: Option<FeatureCache>,
    inference_engine: InferenceEngine,
    latency_budget: LatencyBudget,
    profiler: Option<LatencyProfiler>,
    initialized: bool,
    prediction_count: usize,
}

impl LivePipeline {
    /// Create a new live pipeline with the given configuration.
    pub fn new(config: LivePipelineConfig) -> Self {
        let gaf_computer = IncrementalGAFComputer::new(config.window_size);
        let cache = if config.enable_cache {
            Some(FeatureCache::with_capacities(
                config.cache_capacity,
                config.cache_capacity,
                config.cache_capacity * 2,
            ))
        } else {
            None
        };
        let inference_engine = InferenceEngine::new(config.inference_config.clone());
        let latency_budget =
            LatencyBudget::from_ms("end_to_end".to_string(), config.target_latency_ms);
        let profiler = if config.enable_profiling {
            Some(LatencyProfiler::new())
        } else {
            None
        };

        Self {
            config,
            gaf_computer,
            cache,
            inference_engine,
            latency_budget,
            profiler,
            initialized: false,
            prediction_count: 0,
        }
    }

    /// Create a live pipeline with default configuration.
    pub fn default() -> Self {
        Self::new(LivePipelineConfig::default())
    }

    /// Initialize the pipeline (load models, etc.).
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // In practice, this would load model weights
        self.inference_engine.load_weights("model.ckpt")?;
        self.initialized = true;
        Ok(())
    }

    /// Warm up the pipeline with dummy data.
    pub fn warmup(&mut self) -> Result<()> {
        if !self.initialized {
            self.initialize()?;
        }

        for i in 0..self.config.warmup_iterations {
            let dummy_data = MarketData::new(
                i as i64,
                100.0 + i as f64 * 0.1,
                101.0 + i as f64 * 0.1,
                99.0 + i as f64 * 0.1,
                100.5 + i as f64 * 0.1,
                1000.0,
            );
            let _ = self.process_market_data(dummy_data)?;
        }

        // Reset statistics after warmup
        self.gaf_computer.reset();
        self.latency_budget.reset();
        if let Some(ref mut profiler) = self.profiler {
            profiler.reset();
        }
        self.prediction_count = 0;

        Ok(())
    }

    /// Process a new market data tick and generate a prediction.
    pub fn process_market_data(&mut self, data: MarketData) -> Result<Option<Prediction>> {
        if !self.initialized {
            return Err(VisionError::InternalError(
                "Pipeline not initialized".to_string(),
            ));
        }

        let mut measurement = LatencyMeasurement::start();

        // Stage 1: Streaming feature update
        if let Some(ref mut profiler) = self.profiler {
            profiler.start_stage("streaming_update");
        }

        let gaf_result = self.gaf_computer.update(data)?;

        if let Some(ref mut profiler) = self.profiler {
            profiler.stop_stage();
        }

        // If buffer not full yet, return None
        if gaf_result.is_none() {
            return Ok(None);
        }

        let gaf = gaf_result.unwrap();

        // Stage 2: Cache lookup (if enabled)
        if let Some(ref mut profiler) = self.profiler {
            profiler.start_stage("cache_lookup");
        }

        // Cache key would be based on actual data in practice
        // For now, we skip caching and use computed GAF directly

        if let Some(ref mut profiler) = self.profiler {
            profiler.stop_stage();
        }

        // Stage 3: Inference
        if let Some(ref mut profiler) = self.profiler {
            profiler.start_stage("inference");
        }

        let prediction = self.inference_engine.predict_single(&gaf)?;

        if let Some(ref mut profiler) = self.profiler {
            profiler.stop_stage();
        }

        // Record end-to-end latency
        let total_latency = measurement.stop();
        self.latency_budget.check(total_latency);
        self.prediction_count += 1;

        Ok(Some(prediction))
    }

    /// Process a tick (alias for process_market_data).
    pub fn process_tick(&mut self, data: MarketData) -> Result<Option<Prediction>> {
        self.process_market_data(data)
    }

    /// Get the number of predictions generated.
    pub fn prediction_count(&self) -> usize {
        self.prediction_count
    }

    /// Check if the pipeline is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Check if the pipeline is ready (buffer full).
    pub fn is_ready(&self) -> bool {
        self.gaf_computer.is_ready()
    }

    /// Get the latency budget.
    pub fn latency_budget(&self) -> &LatencyBudget {
        &self.latency_budget
    }

    /// Get the profiler if enabled.
    pub fn profiler(&self) -> Option<&LatencyProfiler> {
        self.profiler.as_ref()
    }

    /// Get inference statistics.
    pub fn inference_stats(&self) -> Result<InferenceStats> {
        self.inference_engine.stats()
    }

    /// Get cache statistics if caching is enabled.
    pub fn cache_stats(&self) -> Option<cache::CombinedCacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Reset all statistics.
    pub fn reset_stats(&mut self) {
        self.latency_budget.reset();
        if let Some(ref mut profiler) = self.profiler {
            profiler.reset();
        }
        let _ = self.inference_engine.reset_stats();
        self.prediction_count = 0;
    }

    /// Generate a comprehensive performance report.
    pub fn performance_report(&self) {
        println!("=== Live Pipeline Performance Report ===\n");

        println!("Configuration:");
        println!("  Window size: {}", self.config.window_size);
        println!("  Cache enabled: {}", self.config.enable_cache);
        println!("  Target latency: {:.2} ms", self.config.target_latency_ms);
        println!("  Predictions generated: {}\n", self.prediction_count);

        println!("Latency Budget:");
        self.latency_budget.report();
        println!();

        if let Some(ref profiler) = self.profiler {
            profiler.report();
        }

        if let Ok(stats) = self.inference_engine.stats() {
            println!("Inference Statistics:");
            println!("  Total predictions: {}", stats.total_predictions);
            println!("  Avg latency: {:.2} ms", stats.avg_latency_ms());
            println!(
                "  P95 latency: {:.2} ms",
                stats.p95_latency_us as f64 / 1000.0
            );
            println!(
                "  P99 latency: {:.2} ms",
                stats.p99_latency_us as f64 / 1000.0
            );
            println!();
        }

        if let Some(cache_stats) = self.cache_stats() {
            println!("Cache Statistics:");
            println!(
                "  Overall hit rate: {:.2}%",
                cache_stats.overall_hit_rate() * 100.0
            );
            println!("  Total evictions: {}", cache_stats.total_evictions());
        }
    }
}

/// Thread-safe wrapper for LivePipeline.
///
/// Enables concurrent access to the pipeline from multiple threads.
#[derive(Clone)]
pub struct SharedLivePipeline {
    inner: Arc<RwLock<LivePipeline>>,
}

impl SharedLivePipeline {
    /// Create a new shared live pipeline.
    pub fn new(config: LivePipelineConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LivePipeline::new(config))),
        }
    }

    /// Initialize the pipeline.
    pub fn initialize(&self) -> Result<()> {
        self.inner
            .write()
            .map_err(|_| VisionError::InternalError("Lock poisoned".to_string()))?
            .initialize()
    }

    /// Warm up the pipeline.
    pub fn warmup(&self) -> Result<()> {
        self.inner
            .write()
            .map_err(|_| VisionError::InternalError("Lock poisoned".to_string()))?
            .warmup()
    }

    /// Process market data.
    pub fn process_market_data(&self, data: MarketData) -> Result<Option<Prediction>> {
        self.inner
            .write()
            .map_err(|_| VisionError::InternalError("Lock poisoned".to_string()))?
            .process_market_data(data)
    }

    /// Get prediction count.
    pub fn prediction_count(&self) -> Result<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| VisionError::InternalError("Lock poisoned".to_string()))?
            .prediction_count())
    }

    /// Check if initialized.
    pub fn is_initialized(&self) -> Result<bool> {
        Ok(self
            .inner
            .read()
            .map_err(|_| VisionError::InternalError("Lock poisoned".to_string()))?
            .is_initialized())
    }

    /// Generate performance report.
    pub fn performance_report(&self) -> Result<()> {
        self.inner
            .read()
            .map_err(|_| VisionError::InternalError("Lock poisoned".to_string()))?
            .performance_report();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_pipeline_creation() {
        let pipeline = LivePipeline::default();
        assert!(!pipeline.is_initialized());
    }

    #[test]
    fn test_live_pipeline_initialization() {
        let mut pipeline = LivePipeline::default();
        pipeline.initialize().unwrap();
        assert!(pipeline.is_initialized());
    }

    #[test]
    fn test_live_pipeline_warmup() {
        let mut pipeline = LivePipeline::new(LivePipelineConfig {
            warmup_iterations: 5,
            ..LivePipelineConfig::default()
        });
        pipeline.warmup().unwrap();
        assert!(pipeline.is_initialized());
    }

    #[test]
    fn test_process_market_data() {
        let mut pipeline = LivePipeline::default();
        pipeline.initialize().unwrap();

        let data = MarketData::new(0, 100.0, 101.0, 99.0, 100.5, 1000.0);
        let result = pipeline.process_market_data(data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_not_ready() {
        let mut pipeline = LivePipeline::new(LivePipelineConfig {
            window_size: 10,
            ..LivePipelineConfig::default()
        });
        pipeline.initialize().unwrap();

        // First few ticks should return None (buffer not full)
        for i in 0..5 {
            let data = MarketData::new(i, 100.0, 101.0, 99.0, 100.5, 1000.0);
            let result = pipeline.process_market_data(data).unwrap();
            assert!(result.is_none());
        }
    }

    #[test]
    fn test_pipeline_full_flow() {
        let mut pipeline = LivePipeline::new(LivePipelineConfig {
            window_size: 3,
            enable_profiling: true,
            ..LivePipelineConfig::default()
        });
        pipeline.initialize().unwrap();

        // Fill buffer
        for i in 0..3 {
            let data = MarketData::new(i, 100.0 + i as f64, 101.0, 99.0, 100.5, 1000.0);
            let result = pipeline.process_market_data(data).unwrap();

            if i < 2 {
                assert!(result.is_none());
            } else {
                assert!(result.is_some());
                let pred = result.unwrap();
                assert!(pred.signal >= -1.0 && pred.signal <= 1.0);
                assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
            }
        }

        assert_eq!(pipeline.prediction_count(), 1);
    }

    #[test]
    fn test_low_latency_config() {
        let config = LivePipelineConfig::low_latency();
        assert_eq!(config.window_size, 30);
        assert_eq!(config.target_latency_ms, 50.0);
        assert!(config.enable_profiling);
    }

    #[test]
    fn test_high_accuracy_config() {
        let config = LivePipelineConfig::high_accuracy();
        assert_eq!(config.window_size, 120);
        assert_eq!(config.target_latency_ms, 200.0);
    }

    #[test]
    fn test_backtest_replay_config() {
        let config = LivePipelineConfig::backtest_replay();
        assert!(!config.enable_cache);
    }

    #[test]
    fn test_shared_pipeline() {
        let shared = SharedLivePipeline::new(LivePipelineConfig::default());
        shared.initialize().unwrap();

        assert!(shared.is_initialized().unwrap());
    }

    #[test]
    fn test_shared_pipeline_concurrent() {
        use std::thread;

        let shared = SharedLivePipeline::new(LivePipelineConfig {
            window_size: 3,
            ..LivePipelineConfig::default()
        });
        shared.initialize().unwrap();

        let handles: Vec<_> = (0..3)
            .map(|i| {
                let shared_clone = shared.clone();
                thread::spawn(move || {
                    let data = MarketData::new(i, 100.0 + i as f64, 101.0, 99.0, 100.5, 1000.0);
                    shared_clone.process_market_data(data)
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap().unwrap();
        }

        // Just verify the call succeeds
        let _count = shared.prediction_count().unwrap();
    }

    #[test]
    fn test_uninitialized_processing() {
        let mut pipeline = LivePipeline::default();
        let data = MarketData::new(0, 100.0, 101.0, 99.0, 100.5, 1000.0);
        let result = pipeline.process_market_data(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset_stats() {
        let mut pipeline = LivePipeline::new(LivePipelineConfig {
            window_size: 3,
            ..LivePipelineConfig::default()
        });
        pipeline.initialize().unwrap();

        // Generate some predictions
        for i in 0..5 {
            let data = MarketData::new(i, 100.0 + i as f64, 101.0, 99.0, 100.5, 1000.0);
            let _ = pipeline.process_market_data(data);
        }

        assert!(pipeline.prediction_count() > 0);

        pipeline.reset_stats();
        assert_eq!(pipeline.prediction_count(), 0);
    }

    #[test]
    fn test_latency_budget_tracking() {
        let mut pipeline = LivePipeline::new(LivePipelineConfig {
            window_size: 3,
            target_latency_ms: 100.0,
            ..LivePipelineConfig::default()
        });
        pipeline.initialize().unwrap();

        for i in 0..5 {
            let data = MarketData::new(i, 100.0 + i as f64, 101.0, 99.0, 100.5, 1000.0);
            let _ = pipeline.process_market_data(data);
        }

        let budget = pipeline.latency_budget();
        assert!(budget.total_checks() > 0);
    }
}
