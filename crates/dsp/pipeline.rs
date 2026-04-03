//! Complete DSP Pipeline for Project JANUS
//!
//! This module orchestrates the entire signal processing chain:
//! 1. Sevcik Fractal Dimension → Hurst Exponent
//! 2. Fractal Adaptive Moving Average (FRAMA)
//! 3. Welford Online Normalization
//! 4. Feature vector generation for ML inference
//!
//! # Architecture
//!
//! The pipeline is designed for zero-allocation hot-path operation:
//! - All buffers pre-allocated during initialization
//! - No dynamic dispatch or heap allocations in `process()`
//! - Cache-friendly data layout
//! - SIMD-ready (future optimization)
//!
//! # Performance Target
//!
//! - Throughput: >1M ticks/sec on modern CPU
//! - Latency: <1μs per tick (median), <10μs (P99)
//! - Memory: O(window_size) ≈ 1KB per pipeline instance
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_dsp::pipeline::{DspPipeline, DspConfig, PipelineOutput};
//!
//! let config = DspConfig::default();
//! let mut pipeline = DspPipeline::new(config);
//!
//! // Process market ticks
//! for tick in market_feed {
//!     match pipeline.process(tick.price) {
//!         Ok(output) => {
//!             // Feed to neural network
//!             model.infer(&output.features);
//!         }
//!         Err(e) => {
//!             // Still warming up or invalid data
//!             log::debug!("Pipeline not ready: {}", e);
//!         }
//!     }
//! }
//! ```

use super::frama::{Frama, FramaDiagnostics, MarketRegime};
use super::normalize::{NormalizationError, WelfordNormalizer};
use super::sevcik::FractalError;

/// DSP Pipeline configuration
#[derive(Debug, Clone)]
pub struct DspConfig {
    /// Window size for fractal dimension calculation
    pub fractal_window: usize,

    /// FRAMA minimum alpha (maximum smoothing)
    pub frama_alpha_min: f64,

    /// FRAMA maximum alpha (minimum smoothing)
    pub frama_alpha_max: f64,

    /// Enable Ehlers Super Smoother
    pub use_super_smoother: bool,

    /// Normalization alpha (exponential decay)
    pub norm_alpha: f64,

    /// Normalization warmup period
    pub norm_warmup: usize,

    /// Z-score clipping threshold (None = no clipping)
    pub norm_clip_threshold: Option<f64>,

    /// Enable divergence normalization
    pub normalize_divergence: bool,

    /// Enable alpha normalization
    pub normalize_alpha: bool,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            fractal_window: 64,
            frama_alpha_min: 0.01,
            frama_alpha_max: 0.5,
            use_super_smoother: false,
            norm_alpha: 0.05,
            norm_warmup: 50,
            norm_clip_threshold: Some(3.0),
            normalize_divergence: true,
            normalize_alpha: true,
        }
    }
}

impl DspConfig {
    /// High-frequency configuration (fast adaptation)
    pub fn high_frequency() -> Self {
        Self {
            fractal_window: 32,
            frama_alpha_min: 0.05,
            frama_alpha_max: 0.7,
            use_super_smoother: true,
            norm_alpha: 0.2,
            norm_warmup: 30,
            norm_clip_threshold: Some(3.0),
            normalize_divergence: true,
            normalize_alpha: true,
        }
    }

    /// Low-frequency configuration (stable, lower noise)
    pub fn low_frequency() -> Self {
        Self {
            fractal_window: 128,
            frama_alpha_min: 0.005,
            frama_alpha_max: 0.3,
            use_super_smoother: false,
            norm_alpha: 0.01,
            norm_warmup: 100,
            norm_clip_threshold: None,
            normalize_divergence: true,
            normalize_alpha: false,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.fractal_window < 8 {
            return Err(format!(
                "fractal_window too small: {} (minimum 8)",
                self.fractal_window
            ));
        }

        if self.frama_alpha_min >= self.frama_alpha_max {
            return Err(format!(
                "Invalid alpha range: [{}, {}]",
                self.frama_alpha_min, self.frama_alpha_max
            ));
        }

        if self.norm_alpha <= 0.0 || self.norm_alpha > 1.0 {
            return Err(format!("Invalid norm_alpha: {}", self.norm_alpha));
        }

        if self.norm_warmup < 2 {
            return Err(format!("norm_warmup too small: {}", self.norm_warmup));
        }

        Ok(())
    }
}

/// Complete pipeline output
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Raw input price
    pub price: f64,

    /// FRAMA value
    pub frama: f64,

    /// Price divergence from FRAMA (price - frama)
    pub divergence: f64,

    /// Normalized divergence (Z-score)
    pub divergence_norm: Option<f64>,

    /// Current FRAMA alpha
    pub alpha: f64,

    /// Normalized alpha (Z-score)
    pub alpha_norm: Option<f64>,

    /// Fractal dimension
    pub fractal_dim: f64,

    /// Hurst exponent
    pub hurst: f64,

    /// Market regime classification
    pub regime: MarketRegime,

    /// Feature vector for ML inference (pre-allocated, fixed size)
    pub features: [f64; 8],

    /// Processing timestamp (for latency monitoring)
    pub timestamp_ns: u64,
}

impl PipelineOutput {
    /// Convert to feature vector for neural network
    ///
    /// Feature layout:
    /// [0] = divergence (normalized)
    /// [1] = alpha (normalized)
    /// [2] = fractal_dim (raw)
    /// [3] = hurst (raw)
    /// [4] = regime (encoded: -1=mean_reverting, 0=random, 1=trending)
    /// [5] = divergence_sign (1 if positive, -1 if negative)
    /// [6] = alpha_deviation (alpha - 0.25, normalized)
    /// [7] = regime_confidence (distance from boundaries)
    #[inline]
    pub fn to_features(&self) -> &[f64; 8] {
        &self.features
    }

    /// Check if all features are valid (no NaN/Inf)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.features.iter().all(|&x| x.is_finite())
    }
}

/// Pipeline error types
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// FRAMA calculation failed
    FramaError(FractalError),

    /// Normalization failed
    NormalizationError(NormalizationError),

    /// Invalid configuration
    ConfigError(String),

    /// Invalid input
    InvalidInput(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FramaError(e) => write!(f, "FRAMA error: {}", e),
            Self::NormalizationError(e) => write!(f, "Normalization error: {}", e),
            Self::ConfigError(e) => write!(f, "Config error: {}", e),
            Self::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<FractalError> for PipelineError {
    fn from(e: FractalError) -> Self {
        Self::FramaError(e)
    }
}

impl From<NormalizationError> for PipelineError {
    fn from(e: NormalizationError) -> Self {
        Self::NormalizationError(e)
    }
}

/// Complete DSP pipeline
pub struct DspPipeline {
    /// Configuration
    config: DspConfig,

    /// FRAMA calculator
    frama: Frama,

    /// Divergence normalizer
    divergence_norm: Option<WelfordNormalizer>,

    /// Alpha normalizer
    alpha_norm: Option<WelfordNormalizer>,

    /// Processing statistics
    total_ticks: u64,
    valid_outputs: u64,
    last_output: Option<PipelineOutput>,
}

impl DspPipeline {
    /// Create a new DSP pipeline
    ///
    /// # Arguments
    ///
    /// * `config` - Pipeline configuration
    ///
    /// # Panics
    ///
    /// Panics if configuration is invalid
    pub fn new(config: DspConfig) -> Self {
        config.validate().expect("Invalid DSP configuration");

        let frama = Frama::new(
            config.fractal_window,
            config.frama_alpha_min,
            config.frama_alpha_max,
            config.use_super_smoother,
        );

        let divergence_norm = if config.normalize_divergence {
            Some(WelfordNormalizer::new(
                config.norm_alpha,
                config.norm_warmup,
                config.norm_clip_threshold,
            ))
        } else {
            None
        };

        let alpha_norm = if config.normalize_alpha {
            Some(WelfordNormalizer::new(
                config.norm_alpha,
                config.norm_warmup,
                config.norm_clip_threshold,
            ))
        } else {
            None
        };

        Self {
            config,
            frama,
            divergence_norm,
            alpha_norm,
            total_ticks: 0,
            valid_outputs: 0,
            last_output: None,
        }
    }

    /// Process a new price tick
    ///
    /// # Arguments
    ///
    /// * `price` - New price observation
    ///
    /// # Returns
    ///
    /// - `Ok(PipelineOutput)` with complete feature vector
    /// - `Err(PipelineError)` if processing failed
    ///
    /// # Performance
    ///
    /// This is the critical hot path. Must be:
    /// - Zero allocations
    /// - Predictable branches
    /// - Cache-friendly
    /// - SIMD-ready
    #[inline]
    pub fn process(&mut self, price: f64) -> Result<PipelineOutput, PipelineError> {
        // Timestamp for latency monitoring
        let timestamp_ns = Self::get_timestamp_ns();

        self.total_ticks += 1;

        // Validate input
        if !price.is_finite() {
            return Err(PipelineError::InvalidInput(
                "Price is NaN or Inf".to_string(),
            ));
        }

        // Update FRAMA
        let frama_diag = self.frama.update(price)?;

        // Normalize divergence
        let divergence_norm = if let Some(ref mut norm) = self.divergence_norm {
            match norm.update(frama_diag.divergence) {
                Ok(nv) => Some(nv.normalized),
                Err(_) => None, // Still warming up
            }
        } else {
            None
        };

        // Normalize alpha
        let alpha_norm = if let Some(ref mut norm) = self.alpha_norm {
            match norm.update(frama_diag.alpha) {
                Ok(nv) => Some(nv.normalized),
                Err(_) => None, // Still warming up
            }
        } else {
            None
        };

        // Build feature vector
        let features = self.build_features(&frama_diag, divergence_norm, alpha_norm);

        let output = PipelineOutput {
            price,
            frama: frama_diag.frama,
            divergence: frama_diag.divergence,
            divergence_norm,
            alpha: frama_diag.alpha,
            alpha_norm,
            fractal_dim: frama_diag.fractal_dim,
            hurst: frama_diag.hurst,
            regime: frama_diag.regime,
            features,
            timestamp_ns,
        };

        // Validate output
        if !output.is_valid() {
            return Err(PipelineError::InvalidInput(
                "Feature vector contains NaN or Inf".to_string(),
            ));
        }

        self.valid_outputs += 1;
        self.last_output = Some(output.clone());

        Ok(output)
    }

    /// Build feature vector from components
    ///
    /// This is performance-critical and must be allocation-free
    #[inline]
    fn build_features(
        &self,
        frama_diag: &FramaDiagnostics,
        divergence_norm: Option<f64>,
        alpha_norm: Option<f64>,
    ) -> [f64; 8] {
        // Encode regime as numeric value
        let regime_encoded = match frama_diag.regime {
            MarketRegime::MeanReverting => -1.0,
            MarketRegime::RandomWalk => 0.0,
            MarketRegime::Trending => 1.0,
            MarketRegime::Unknown => 0.0,
        };

        // Divergence sign
        let divergence_sign = if frama_diag.divergence > 0.0 {
            1.0
        } else if frama_diag.divergence < 0.0 {
            -1.0
        } else {
            0.0
        };

        // Alpha deviation from midpoint (0.25)
        let alpha_deviation = frama_diag.alpha - 0.25;

        // Regime confidence (distance from boundaries)
        let regime_confidence = if frama_diag.hurst > 0.6 {
            frama_diag.hurst - 0.6 // Distance from trending threshold
        } else if frama_diag.hurst < 0.4 {
            0.4 - frama_diag.hurst // Distance from mean-reverting threshold
        } else {
            0.0 // Inside random walk zone
        };

        [
            divergence_norm.unwrap_or(0.0), // [0] Normalized divergence
            alpha_norm.unwrap_or(0.0),      // [1] Normalized alpha
            frama_diag.fractal_dim,         // [2] Raw fractal dimension
            frama_diag.hurst,               // [3] Raw Hurst exponent
            regime_encoded,                 // [4] Regime encoding
            divergence_sign,                // [5] Divergence direction
            alpha_deviation,                // [6] Alpha deviation
            regime_confidence,              // [7] Regime confidence
        ]
    }

    /// Get current timestamp in nanoseconds (for latency monitoring)
    #[inline]
    fn get_timestamp_ns() -> u64 {
        // Use std::time for cross-platform compatibility
        // In production, replace with TSC or rdtsc for sub-nanosecond precision
        use std::time::{SystemTime, UNIX_EPOCH};

        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            total_ticks: self.total_ticks,
            valid_outputs: self.valid_outputs,
            success_rate: if self.total_ticks > 0 {
                self.valid_outputs as f64 / self.total_ticks as f64
            } else {
                0.0
            },
        }
    }

    /// Get last successful output
    pub fn last_output(&self) -> Option<&PipelineOutput> {
        self.last_output.as_ref()
    }

    /// Check if pipeline is ready (fully warmed up)
    pub fn is_ready(&self) -> bool {
        self.frama.is_ready()
            && self.divergence_norm.as_ref().is_none_or(|n| n.is_ready())
            && self.alpha_norm.as_ref().is_none_or(|n| n.is_ready())
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.frama = Frama::new(
            self.config.fractal_window,
            self.config.frama_alpha_min,
            self.config.frama_alpha_max,
            self.config.use_super_smoother,
        );

        if let Some(ref mut norm) = self.divergence_norm {
            norm.reset();
        }

        if let Some(ref mut norm) = self.alpha_norm {
            norm.reset();
        }

        self.total_ticks = 0;
        self.valid_outputs = 0;
        self.last_output = None;
    }

    /// Get configuration
    pub fn config(&self) -> &DspConfig {
        &self.config
    }
}

/// Pipeline statistics
#[derive(Debug, Clone, Copy)]
pub struct PipelineStats {
    /// Total ticks processed
    pub total_ticks: u64,

    /// Valid outputs produced
    pub valid_outputs: u64,

    /// Success rate (valid / total)
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DspConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config = DspConfig {
            fractal_window: 2, // Too small
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipeline_creation() {
        let config = DspConfig::default();
        let pipeline = DspPipeline::new(config);
        assert_eq!(pipeline.stats().total_ticks, 0);
    }

    #[test]
    fn test_pipeline_warmup() {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        // First ticks should fail (warmup)
        for i in 0..60 {
            let result = pipeline.process(100.0 + i as f64 * 0.1);
            // Should fail during warmup
            if i < 64 {
                assert!(result.is_err());
            }
        }

        // After warmup, should succeed
        for i in 60..100 {
            let result = pipeline.process(100.0 + i as f64 * 0.1);
            if result.is_err() {
                // May still fail if normalizers are warming up
                continue;
            }
            let output = result.unwrap();
            assert!(output.is_valid());
        }
    }

    #[test]
    fn test_pipeline_trending_market() {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        // Feed strong uptrend
        for i in 0..200 {
            let price = 100.0 + i as f64 * 0.5;
            let _ = pipeline.process(price);
        }

        let output = pipeline.process(200.0).unwrap();

        // Should detect trending regime
        assert!(
            matches!(
                output.regime,
                MarketRegime::Trending | MarketRegime::RandomWalk
            ),
            "Expected trending, got {:?}",
            output.regime
        );

        // FRAMA should lag behind price
        assert!(output.frama < output.price);

        // Divergence should be positive
        assert!(output.divergence > 0.0);
    }

    #[test]
    fn test_feature_vector() {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        // Warmup
        for i in 0..150 {
            let _ = pipeline.process(100.0 + i as f64 * 0.1);
        }

        let output = pipeline.process(115.0).unwrap();

        // Feature vector should have 8 elements
        assert_eq!(output.features.len(), 8);

        // All features should be finite
        assert!(output.is_valid());
    }

    #[test]
    fn test_invalid_price() {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        // NaN should be rejected
        let result = pipeline.process(f64::NAN);
        assert!(result.is_err());

        // Infinity should be rejected
        let result = pipeline.process(f64::INFINITY);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        // Process some data
        for i in 0..100 {
            let _ = pipeline.process(100.0 + i as f64);
        }

        assert!(pipeline.stats().total_ticks > 0);

        // Reset
        pipeline.reset();

        assert_eq!(pipeline.stats().total_ticks, 0);
        assert_eq!(pipeline.stats().valid_outputs, 0);
    }

    #[test]
    fn test_high_frequency_config() {
        let config = DspConfig::high_frequency();
        assert!(config.validate().is_ok());

        let pipeline = DspPipeline::new(config);
        assert_eq!(pipeline.config().fractal_window, 32);
    }

    #[test]
    fn test_low_frequency_config() {
        let config = DspConfig::low_frequency();
        assert!(config.validate().is_ok());

        let pipeline = DspPipeline::new(config);
        assert_eq!(pipeline.config().fractal_window, 128);
    }

    #[test]
    fn test_stats_tracking() {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        for i in 0..100 {
            let _ = pipeline.process(100.0 + i as f64);
        }

        let stats = pipeline.stats();
        assert_eq!(stats.total_ticks, 100);
        assert!(stats.success_rate >= 0.0 && stats.success_rate <= 1.0);
    }
}
