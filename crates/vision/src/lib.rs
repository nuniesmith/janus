//! Vision Pipeline for JANUS Trading System
//!
//! This crate provides time-series to image transformation using
//! Differentiable Gramian Angular Fields (DiffGAF).
//!
//! # Architecture
//!
//! ```text
//! Time Series → DiffGAF → 2D Image → Vision Model
//!   [T, F]       →      [F, H, W]    →  Predictions
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use janus_vision::{DiffGAF, DiffGAFConfig};
//! use burn_core::tensor::Tensor;
//!
//! // Create DiffGAF transformer
//! let config = DiffGAFConfig::default();
//! let diffgaf = config.build(&device);
//!
//! // Transform time series to image
//! let time_series = Tensor::from_floats([[1.0, 2.0, 3.0]], &device);
//! let image = diffgaf.forward(time_series); // [B, F, H, W]
//! ```

pub mod adaptive;
pub mod backtest;
pub mod config;
pub mod data;
pub mod diff_gaf;
pub mod diffgaf;
pub mod ensemble;
pub mod error;
pub mod execution;
pub mod gaf;
pub mod live;
pub mod pipeline;
pub mod portfolio;
pub mod preprocessing;
pub mod production;
pub mod risk;
pub mod signals;
pub mod vivit;

#[cfg(feature = "viz")]
pub mod visualization;

// Re-exports
pub use adaptive::{
    AdaptiveSystem, AdaptiveSystemStats, AdaptiveThreshold, CalibrationConfig, CalibrationMetrics,
    CombinedCalibrator, IsotonicCalibration, MarketRegime, MultiLevelThreshold,
    PercentileThreshold, PlattScaling, RegimeAdjuster, RegimeConfig, RegimeDetector,
    RegimeMultipliers, RiskLevel, ThresholdCalibrator, ThresholdConfig, ThresholdStats,
};
pub use backtest::{
    BacktestSimulation, MetricsCalculator, PerformanceMetrics, Position, PositionSizing,
    RiskMetrics, SignalQuality, SimulationConfig, SimulationState, Trade, TradeResult, TradeStats,
    TradeStatus,
};
pub use config::VisionConfig;
pub use data::{
    CsvLoader, LoaderConfig, OhlcvCandle, OhlcvDataset, SequenceConfig, TrainValSplit,
    ValidationError, ValidationReport, load_ohlcv_csv, validate_ohlcv,
};
pub use diffgaf::{
    BestModelTracker, CheckpointMetadata, ClassificationBatch, DiffGAF, DiffGAFConfig, DiffGafLstm,
    DiffGafLstmConfig, GramianLayer, LearnableNorm, PolarEncoder,
};
pub use ensemble::{
    BlendingEnsemble, EnsembleConfig, EnsembleManager, EnsemblePrediction, EnsembleStrategy,
    FeatureImportance, ManagerStrategy, MetaLearner, ModelEnsemble, ModelPerformance,
    ModelPrediction, StackingConfig, StackingEnsemble,
};
pub use error::{Result, VisionError};
pub use execution::{
    ExecutionAnalytics, ExecutionManager, ExecutionRecord, ExecutionReport, ExecutionSlice,
    ExecutionStrategy, OrderId, OrderRequest, OrderState, OrderStatus, Side, TWAPConfig,
    TWAPExecutor, TWAPStatistics, VWAPConfig, VWAPExecutor, VWAPSlice, VWAPStatistics, Venue,
    VenueStats, VolumeProfile,
};
pub use live::{
    BenchmarkResults, CachedFeature, CircularBuffer, FeatureCache, FeatureCacheKey, FeatureType,
    IncrementalGAFComputer, InferenceConfig, InferenceEngine, InferenceStats, LRUCache,
    LatencyBudget, LatencyMeasurement, LatencyProfiler, LatencySummary, LatencyTracker,
    LivePipeline, LivePipelineConfig, MarketData, MultiTimeframeBuffer, Prediction, RunningStats,
    SharedLivePipeline, StreamingFeatureBuffer,
};
pub use pipeline::{ViViTConfig, VisionPipeline, VisionPipelineConfig, VisionPipelineOutput};
pub use portfolio::{
    BlackLittermanConfig, BlackLittermanOptimizer, BlackLittermanResult, CovarianceEstimator,
    MeanVarianceOptimizer, OptimizationObjective, OptimizationResult, PortfolioAnalytics,
    PortfolioConstraints, PortfolioRebalancer, RiskBudget, RiskParityMethod, RiskParityOptimizer,
    RiskParityResult, View,
};
pub use preprocessing::{
    BatchIterator, FeatureConfig, FeatureEngineer, MinMaxScaler, RobustScaler, Scaler,
    TensorConverter, TensorConverterConfig, ZScoreScaler, create_batch,
};
pub use production::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState, CircuitStats,
    ComponentHealth, Counter, ErrorRateTracker, Gauge, HealthCheckConfig, HealthMonitor,
    HealthReport, HealthStatus, Histogram, LivenessProbe, MetricType, MetricsRegistry,
    PipelineMetrics, ProductionConfig, ProductionMonitor, ReadinessProbe, ResourceMetrics,
    RetryConfig, RetryExecutor,
};
pub use risk::{
    CorrelationFilter, CorrelationMatrix, DrawdownMonitor, KellyCalculator, KellyOptimizer,
    OptimalFraction, PortfolioRisk, PositionSizer, RiskConfig, RiskError, RiskLimits, RiskManager,
    RiskMonitor, RiskReport, SizingMethod, ViolationType, VolatilityAdjuster, VolatilityEstimate,
    VolatilityWindow,
};

/// Version of the vision crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
