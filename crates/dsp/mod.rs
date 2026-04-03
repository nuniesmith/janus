//! Project JANUS - DSP Layer
//!
//! This module implements the complete Digital Signal Processing pipeline for
//! high-frequency trading, featuring:
//!
//! 1. **Sevcik Fractal Dimension** - Streaming O(1) market roughness estimation
//! 2. **Fractal Adaptive Moving Average (FRAMA)** - Regime-aware smoothing
//! 3. **Welford Online Normalization** - Non-stationary Z-score calculation
//! 4. **Complete Pipeline** - Orchestration with zero-allocation hot path
//!
//! # Architecture
//!
//! The DSP layer sits between raw market data and the neural inference core,
//! transforming chaotic price feeds into structured feature vectors:
//!
//! ```text
//! Market Feed → DSP Pipeline → Feature Vector → LTN Inference → Trading Signal
//!     (ticks)      (FRAMA)        (8D tensor)     (Burn-rs)        (order)
//! ```
//!
//! # Performance Characteristics
//!
//! - **Throughput**: >1M ticks/sec (target), measured at 25-50K in Python prototype
//! - **Latency**: <1μs median, <10μs P99 per tick (Rust target)
//! - **Memory**: O(window_size) ≈ 1KB per pipeline instance
//! - **Allocations**: Zero after warmup (critical for HFT)
//!
//! # Design Principles
//!
//! 1. **Causality**: All indicators are strictly causal (no look-ahead bias)
//! 2. **Streaming**: Incremental updates, no batch reprocessing
//! 3. **Regime-Aware**: Adapts to market conditions (trending vs. mean-reverting)
//! 4. **Robustness**: Handles outliers, flat lines, and regime shifts gracefully
//! 5. **Zero-Copy**: Pre-allocated buffers, cache-friendly layout
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use janus_dsp::pipeline::{DspPipeline, DspConfig};
//!
//! // Create pipeline with default configuration
//! let config = DspConfig::default();
//! let mut pipeline = DspPipeline::new(config);
//!
//! // Process market ticks
//! loop {
//!     let tick = market_feed.recv()?;
//!
//!     match pipeline.process(tick.price) {
//!         Ok(output) => {
//!             // Feature vector ready for ML inference
//!             println!("FRAMA: {:.2}, Regime: {:?}, Hurst: {:.3}",
//!                      output.frama, output.regime, output.hurst);
//!
//!             // Feed to neural network
//!             let signal = model.infer(&output.features)?;
//!
//!             if signal.confidence > threshold {
//!                 execute_trade(signal);
//!             }
//!         }
//!         Err(e) => {
//!             // Pipeline warming up or invalid data
//!             log::debug!("DSP not ready: {}", e);
//!         }
//!     }
//! }
//! ```
//!
//! # Configuration Examples
//!
//! ## High-Frequency Trading (sub-second)
//!
//! ```rust,ignore
//! use janus_dsp::pipeline::DspConfig;
//!
//! let config = DspConfig::high_frequency();
//! // - fractal_window: 32 (fast response)
//! // - frama_alpha_max: 0.7 (aggressive tracking)
//! // - norm_alpha: 0.2 (quick regime adaptation)
//! ```
//!
//! ## Low-Frequency Trading (multi-minute)
//!
//! ```rust,ignore
//! use janus_dsp::pipeline::DspConfig;
//!
//! let config = DspConfig::low_frequency();
//! // - fractal_window: 128 (stable estimation)
//! // - frama_alpha_min: 0.005 (heavy smoothing)
//! // - norm_alpha: 0.01 (slow regime adaptation)
//! ```
//!
//! # Feature Vector Layout
//!
//! The pipeline outputs an 8-dimensional feature vector optimized for neural networks:
//!
//! | Index | Feature              | Range       | Description                          |
//! |-------|----------------------|-------------|--------------------------------------|
//! | 0     | divergence_norm      | ±3σ         | Normalized price-FRAMA divergence    |
//! | 1     | alpha_norm           | ±3σ         | Normalized FRAMA smoothing factor    |
//! | 2     | fractal_dim          | [1.0, 2.0]  | Raw Sevcik fractal dimension         |
//! | 3     | hurst                | [0.0, 1.0]  | Hurst exponent (persistence)         |
//! | 4     | regime               | {-1, 0, 1}  | Market regime encoding               |
//! | 5     | divergence_sign      | {-1, 0, 1}  | Directional signal                   |
//! | 6     | alpha_deviation      | ±0.25       | Alpha deviation from midpoint        |
//! | 7     | regime_confidence    | [0.0, 0.6]  | Distance from regime boundaries      |
//!
//! # Mathematical Background
//!
//! ## Sevcik Fractal Dimension
//!
//! The Sevcik approximation estimates the fractal dimension D of a time series:
//!
//! ```text
//! D = 1 + (ln(L) - ln(2)) / ln(2(N-1))
//! ```
//!
//! Where:
//! - L = Euclidean curve length in normalized [0,1]×[0,1] space
//! - N = Window size (typically 64)
//! - D ∈ [1, 2]: 1 = smooth trend, 2 = random noise
//!
//! ## Hurst Exponent
//!
//! Derived from fractal dimension: H = 2 - D
//!
//! - H > 0.6: Trending (persistent)
//! - H = 0.5: Random walk (efficient market)
//! - H < 0.4: Mean-reverting (anti-persistent)
//!
//! ## FRAMA Adaptation
//!
//! The smoothing factor adapts based on fractal dimension:
//!
//! ```text
//! α = exp(-4.6 × (D - 1))
//! α_clamped = clamp(α, α_min, α_max)
//! ```
//!
//! - Low D (trending) → High α → Fast tracking
//! - High D (noisy) → Low α → Heavy smoothing
//!
//! ## Welford Normalization
//!
//! Exponentially weighted mean and variance:
//!
//! ```text
//! μ[t] = μ[t-1] + α × (x[t] - μ[t-1])
//! σ²[t] = (1-α) × σ²[t-1] + α × (x[t] - μ[t-1]) × (x[t] - μ[t])
//! z[t] = (x[t] - μ[t]) / σ[t]
//! ```
//!
//! # Implementation Notes
//!
//! ## Zero-Allocation Hot Path
//!
//! The `DspPipeline::process()` method is the critical hot path and must not
//! allocate memory after warmup:
//!
//! - All buffers pre-allocated in `new()`
//! - Fixed-size arrays (no Vec resizing)
//! - Monotonic deques with fixed capacity
//! - No dynamic dispatch (trait objects avoided)
//!
//! ## Cache-Friendly Layout
//!
//! Data structures optimized for CPU cache:
//!
//! - Small working sets fit in L1 cache (~32KB)
//! - Sequential access patterns (prefetcher-friendly)
//! - Avoid false sharing (future: add `#[repr(align(64))]`)
//!
//! ## SIMD Opportunities (Future)
//!
//! The following operations are SIMD-ready:
//!
//! - Euclidean distance calculation in Sevcik
//! - Exponential smoothing updates
//! - Feature vector construction
//! - Z-score normalization
//!
//! # Testing & Validation
//!
//! ## Unit Tests
//!
//! Each module has comprehensive unit tests covering:
//!
//! - Warmup behavior
//! - Numerical accuracy vs. Python prototype
//! - Edge cases (NaN, Inf, flat lines)
//! - Regime detection
//! - Boundary conditions
//!
//! ## Benchmarks
//!
//! Performance benchmarks in `benches/bench_dsp.rs`:
//!
//! ```bash
//! cargo bench --bench bench_dsp
//! ```
//!
//! Target metrics:
//! - Sevcik update: <100ns
//! - FRAMA update: <200ns
//! - Full pipeline: <1μs
//!
//! ## Integration Tests
//!
//! End-to-end validation in `tests/integration.rs`:
//!
//! - Synthetic trending markets
//! - Synthetic mean-reverting markets
//! - Real market data replay
//! - Regime change detection
//!
//! # Production Checklist
//!
//! Before deploying to production:
//!
//! - [ ] Verify zero allocations with allocation profiler
//! - [ ] Measure P99 latency under load (target <10μs)
//! - [ ] Test with extreme market events (flash crash, halt)
//! - [ ] Validate against Python prototype (numerical equivalence)
//! - [ ] Run chaos tests (random seeds, pathological inputs)
//! - [ ] Profile cache misses and branch mispredictions
//! - [ ] Test on production hardware (same CPU model)
//! - [ ] Benchmark with realistic market data (full day replay)
//!
//! # References
//!
//! - Sevcik, C. (2010). "A procedure to estimate the fractal dimension of waveforms". arXiv:1003.5266
//! - Ehlers, J. F. (2001). "Rocket Science for Traders". Wiley.
//! - Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products". Technometrics, 4(3), 419-420.
//! - Hurst, H. E. (1951). "Long-term storage capacity of reservoirs". Transactions of the American Society of Civil Engineers, 116, 770-799.
//!
//! # Module Organization
//!
//! ```text
//! dsp/
//! ├── mod.rs           ← You are here
//! ├── sevcik.rs        ← Fractal dimension calculator
//! ├── frama.rs         ← Adaptive moving average
//! ├── normalize.rs     ← Online normalization
//! ├── pipeline.rs      ← Complete orchestration
//! ├── benches/         ← Performance benchmarks
//! └── tests/           ← Integration tests
//! ```

// Re-export public API
pub mod frama;
pub mod normalize;
pub mod pipeline;
pub mod sevcik;

// Convenience re-exports for common usage
pub use frama::{Frama, FramaDiagnostics, MarketRegime};
pub use normalize::{NormalizationError, NormalizedValue, WelfordNormalizer};
pub use pipeline::{DspConfig, DspPipeline, PipelineError, PipelineOutput, PipelineStats};
pub use sevcik::{FractalError, FractalResult, SevcikFractalDimension};

/// DSP module version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Recommended default window size for fractal dimension
pub const DEFAULT_FRACTAL_WINDOW: usize = 64;

/// Recommended default FRAMA alpha range
pub const DEFAULT_FRAMA_ALPHA_MIN: f64 = 0.01;
pub const DEFAULT_FRAMA_ALPHA_MAX: f64 = 0.5;

/// Recommended default normalization parameters
pub const DEFAULT_NORM_ALPHA: f64 = 0.05;
pub const DEFAULT_NORM_WARMUP: usize = 50;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_api() {
        // Verify all public types are accessible
        let config = DspConfig::default();
        let pipeline = DspPipeline::new(config);
        assert_eq!(pipeline.stats().total_ticks, 0);
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_FRACTAL_WINDOW, 64);
        assert_eq!(DEFAULT_FRAMA_ALPHA_MIN, 0.01);
        assert_eq!(DEFAULT_FRAMA_ALPHA_MAX, 0.5);
    }
}
