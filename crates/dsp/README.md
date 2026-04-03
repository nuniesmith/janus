# Project JANUS - DSP Layer

**Production-grade Digital Signal Processing for High-Frequency Trading**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

---

## Overview

The DSP layer transforms chaotic market tick data into structured, regime-aware feature vectors optimized for neural inference. This module is the critical preprocessing stage between raw market feeds and the Logic Tensor Network (LTN) core.

**Key Innovation**: Fractal-adaptive filtering that dynamically adjusts smoothing based on real-time market roughness estimation (Sevcik fractal dimension → FRAMA).

## Performance Characteristics

| Metric | Target | Python Prototype | Rust Implementation |
|--------|--------|------------------|---------------------|
| **Throughput** | >1M ticks/sec | 25K-50K ticks/sec | **>250K ticks/sec** (10x min) |
| **Latency (P50)** | <1μs | ~40μs | **<1μs** (target) |
| **Latency (P99)** | <10μs | ~200μs | **<10μs** (target) |
| **Memory** | <1KB/pipeline | ~5KB | **~1KB** |
| **Allocations** | Zero (hot path) | N/A | ✅ **Zero after warmup** |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Market Data Feed                          │
│                   (Tick-by-tick prices)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Sevcik Fractal Dimension                       │
│   • Streaming O(1) complexity                                   │
│   • Hurst exponent derivation (H = 2 - D)                       │
│   • Regime detection (trending/mean-reverting/random)           │
└────────────────────────────┬────────────────────────────────────┘
                             │ D, H
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         Fractal Adaptive Moving Average (FRAMA)                 │
│   • α = exp(-4.6 × (D - 1)), clamped to [0.01, 0.5]            │
│   • Optional Ehlers Super Smoother (2-pole Butterworth)         │
│   • Adaptive smoothing: trending → fast, noisy → heavy          │
└────────────────────────────┬────────────────────────────────────┘
                             │ FRAMA value, divergence
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            Welford Online Normalization                         │
│   • Exponentially weighted mean & variance                      │
│   • Z-score calculation with outlier clipping                   │
│   • Adapts to non-stationary markets                            │
└────────────────────────────┬────────────────────────────────────┘
                             │ Normalized features
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Vector (8D)                            │
│   [0] divergence_norm  [1] alpha_norm      [2] fractal_dim     │
│   [3] hurst            [4] regime          [5] divergence_sign │
│   [6] alpha_deviation  [7] regime_confidence                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    Logic Tensor Network
                     (Neural Inference)
```

## Quick Start

### Basic Usage

```rust
use janus::dsp::pipeline::{DspPipeline, DspConfig};

// Create pipeline with default configuration
let config = DspConfig::default();
let mut pipeline = DspPipeline::new(config);

// Process market ticks
for tick in market_feed {
    match pipeline.process(tick.price) {
        Ok(output) => {
            println!("FRAMA: {:.2}", output.frama);
            println!("Hurst: {:.3} → {:?}", output.hurst, output.regime);
            println!("Features: {:?}", output.features);
            
            // Feed to ML model
            let signal = model.infer(&output.features)?;
            
            if signal.confidence > 0.8 {
                execute_trade(signal);
            }
        }
        Err(e) => {
            // Pipeline still warming up (need ~64+ ticks)
            log::debug!("DSP not ready: {}", e);
        }
    }
}
```

### Configuration Presets

#### High-Frequency Trading (Sub-second)

```rust
let config = DspConfig::high_frequency();
// - fractal_window: 32 (fast response)
// - frama_alpha_max: 0.7 (aggressive tracking)
// - use_super_smoother: true (reduce lag)
// - norm_alpha: 0.2 (quick adaptation)
```

#### Low-Frequency Trading (Multi-minute)

```rust
let config = DspConfig::low_frequency();
// - fractal_window: 128 (stable estimation)
// - frama_alpha_min: 0.005 (heavy smoothing)
// - use_super_smoother: false (less processing)
// - norm_alpha: 0.01 (slow adaptation)
```

#### Custom Configuration

```rust
let config = DspConfig {
    fractal_window: 64,
    frama_alpha_min: 0.01,
    frama_alpha_max: 0.5,
    use_super_smoother: false,
    norm_alpha: 0.05,
    norm_warmup: 50,
    norm_clip_threshold: Some(3.0),  // Clip outliers at ±3σ
    normalize_divergence: true,
    normalize_alpha: true,
};

let mut pipeline = DspPipeline::new(config);
```

## Feature Vector

The pipeline outputs an 8-dimensional feature vector optimized for neural networks:

| Index | Feature | Type | Range | Description |
|-------|---------|------|-------|-------------|
| **0** | `divergence_norm` | Z-score | ±3σ | Price-FRAMA divergence (normalized) |
| **1** | `alpha_norm` | Z-score | ±3σ | FRAMA smoothing factor (normalized) |
| **2** | `fractal_dim` | Raw | [1.0, 2.0] | Sevcik fractal dimension (1=smooth, 2=noise) |
| **3** | `hurst` | Raw | [0.0, 1.0] | Hurst exponent (persistence measure) |
| **4** | `regime` | Categorical | {-1, 0, 1} | Market regime (-1=mean-rev, 0=random, 1=trend) |
| **5** | `divergence_sign` | Categorical | {-1, 0, 1} | Directional signal |
| **6** | `alpha_deviation` | Raw | ±0.25 | Alpha deviation from midpoint (0.25) |
| **7** | `regime_confidence` | Raw | [0.0, 0.6] | Distance from regime boundaries |

### Feature Interpretation

- **divergence_norm > 0**: Price above FRAMA → potential overbought
- **divergence_norm < 0**: Price below FRAMA → potential oversold
- **hurst > 0.6**: Trending market (persistent)
- **hurst < 0.4**: Mean-reverting market (anti-persistent)
- **alpha_deviation > 0**: High volatility/noise → heavy smoothing
- **regime_confidence > 0.2**: Strong regime signal

## Mathematical Background

### Sevcik Fractal Dimension

Approximates the fractal dimension D of a time series:

```
D = 1 + (ln(L) - ln(2)) / ln(2(N-1))
```

Where:
- **L** = Euclidean curve length in normalized [0,1]×[0,1] space
- **N** = Window size (typically 64)
- **D ∈ [1, 2]**: 1 = perfectly smooth trend, 2 = space-filling noise

**Complexity**: O(1) amortized per update (using monotonic deques for min/max tracking)

### Hurst Exponent

Derived from fractal dimension: **H = 2 - D**

Market regimes:
- **H > 0.6**: Trending (persistent, momentum)
- **H ≈ 0.5**: Random walk (efficient market hypothesis)
- **H < 0.4**: Mean-reverting (anti-persistent, oscillating)

### FRAMA Adaptation

The smoothing factor adapts based on fractal dimension:

```
α_raw = exp(-4.6 × (D - 1))
α = clamp(α_raw, α_min, α_max)
```

**Critical clamping** prevents pathological behavior:
- Low D (trending) → High α → Fast tracking
- High D (noisy) → Low α → Heavy smoothing

Update rule:
```
FRAMA[t] = α × Price[t] + (1 - α) × FRAMA[t-1]
```

### Welford Normalization

Exponentially weighted mean and variance (numerically stable):

```
μ[t] = μ[t-1] + α × (x[t] - μ[t-1])
σ²[t] = (1-α) × σ²[t-1] + α × (x[t] - μ[t-1]) × (x[t] - μ[t])
z[t] = (x[t] - μ[t]) / σ[t]
```

Advantages:
- No catastrophic cancellation
- Adapts to regime changes
- O(1) update complexity

## Benchmarks

Run performance benchmarks:

```bash
cargo bench --bench bench_dsp
```

### Expected Results (on modern CPU)

```
test bench_sevcik_update_warm         ... bench:      85 ns/iter
test bench_frama_update_warm          ... bench:     180 ns/iter
test bench_pipeline_process_warm      ... bench:     450 ns/iter  ← CRITICAL
test bench_pipeline_throughput_100k   ... bench: 45,000,000 ns/iter (2.2M ticks/sec)
```

**Production target**: <1μs per tick (1M ticks/sec)

## Testing

### Unit Tests

```bash
cargo test --lib dsp
```

Each module has comprehensive unit tests:
- ✅ Warmup behavior
- ✅ Numerical accuracy
- ✅ Edge cases (NaN, Inf, flat lines)
- ✅ Regime detection
- ✅ Boundary conditions

### Integration Tests

```bash
cargo test --test integration
```

End-to-end validation with synthetic markets:
- ✅ Uptrend detection
- ✅ Downtrend detection
- ✅ Mean-reverting oscillations
- ✅ Random walk handling
- ✅ Spike/outlier robustness
- ✅ Regime change adaptation

## Production Deployment

### Pre-deployment Checklist

- [ ] **Verify zero allocations**: Run with allocation profiler
- [ ] **Measure P99 latency**: Target <10μs under realistic load
- [ ] **Test extreme events**: Flash crash, trading halt, gaps
- [ ] **Cross-validate**: Compare against Python prototype (numerical equivalence within 1e-6)
- [ ] **Chaos testing**: Random seeds, pathological inputs (NaN, Inf, flat lines)
- [ ] **Profile cache misses**: Use `perf` or Intel VTune
- [ ] **Hardware validation**: Test on production CPU (same model)
- [ ] **Full-day replay**: Benchmark with complete market data

### Monitoring Metrics

Key metrics to track in production:

```rust
let stats = pipeline.stats();

// Throughput
metrics.gauge("dsp.total_ticks", stats.total_ticks);
metrics.gauge("dsp.valid_outputs", stats.valid_outputs);
metrics.gauge("dsp.success_rate", stats.success_rate);

// Latency (measure per-tick processing time)
metrics.histogram("dsp.latency_ns", latency_ns);

// Feature distribution (detect regime shifts)
metrics.histogram("dsp.hurst", output.hurst);
metrics.histogram("dsp.fractal_dim", output.fractal_dim);

// Regime tracking
metrics.counter(&format!("dsp.regime.{:?}", output.regime), 1);
```

### Alert Thresholds

```yaml
alerts:
  - name: dsp_success_rate_low
    condition: dsp.success_rate < 0.9
    severity: warning
  
  - name: dsp_latency_high_p99
    condition: dsp.latency_ns.p99 > 10000  # 10μs
    severity: critical
  
  - name: dsp_invalid_features
    condition: rate(dsp.invalid_outputs) > 0.01
    severity: critical
```

## Module Structure

```
dsp/
├── mod.rs              # Public API and module documentation
├── sevcik.rs           # Fractal dimension calculator (O(1) streaming)
├── frama.rs            # Adaptive moving average
├── normalize.rs        # Welford online normalization
├── pipeline.rs         # Complete orchestration
├── README.md           # This file
├── benches/
│   └── bench_dsp.rs    # Performance benchmarks
└── tests/
    └── integration.rs  # End-to-end tests
```

## API Reference

### Core Types

- **`DspPipeline`**: Main orchestrator
- **`DspConfig`**: Configuration builder
- **`PipelineOutput`**: Output with 8D feature vector
- **`SevcikFractalDimension`**: Fractal calculator
- **`Frama`**: Adaptive moving average
- **`WelfordNormalizer`**: Online normalization
- **`MarketRegime`**: Enum `{Trending, RandomWalk, MeanReverting, Unknown}`

### Error Types

- **`PipelineError`**: Top-level pipeline errors
- **`FractalError`**: Insufficient data, invalid range, invalid dimension
- **`NormalizationError`**: Insufficient data, invalid value, invalid variance

## References

1. **Sevcik, C. (2010)**. "A procedure to estimate the fractal dimension of waveforms". *arXiv:1003.5266*

2. **Ehlers, J. F. (2001)**. "Rocket Science for Traders". *Wiley*.

3. **Welford, B. P. (1962)**. "Note on a method for calculating corrected sums of squares and products". *Technometrics, 4(3)*, 419-420.

4. **Hurst, H. E. (1951)**. "Long-term storage capacity of reservoirs". *Transactions of the American Society of Civil Engineers, 116*, 770-799.

## Contributing

See [IMPLEMENTATION_ROADMAP.md](../../../research/IMPLEMENTATION_ROADMAP.md) for Phase 2 development tasks.

## License

MIT License - see [LICENSE](../../LICENSE)

---

**Status**: ✅ Phase 2 Complete (Rust port from Python prototype)

**Next Phase**: Logic Tensor Network (LTN) integration with Burn-rs