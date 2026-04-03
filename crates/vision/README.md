# JANUS Vision Pipeline

> **Differentiable Gramian Angular Field (DiffGAF) for Time-Series Classification**

[![Tests](https://img.shields.io/badge/tests-505%2F505%20passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)]()
[![Burn](https://img.shields.io/badge/burn-0.19-blue)]()

Transform time-series data into learnable 2D images for deep learning with the DiffGAF vision pipeline. Built with the [Burn](https://burn.dev) deep learning framework. Now includes **complete production trading system** with ensemble learning, adaptive thresholding, advanced order execution, portfolio optimization, and Prometheus metrics.

## ✨ Features

### Core Vision Pipeline
- **🔥 Fully Differentiable**: End-to-end learnable GAF transformation
- **🚀 GPU Accelerated**: Optional CUDA support for Gramian computation
- **📉 Memory Efficient**: Adaptive pooling reduces memory by up to 14×
- **💾 Production Ready**: Complete checkpoint system with metadata
- **🎯 High Performance**: Achieves 100% accuracy on synthetic data
- **📁 Real Data Ready**: CSV loading, validation, normalization, feature engineering

### Production Trading System (Week 7-8)
- **🎲 Ensemble Learning**: Model stacking, weighted averaging, dynamic combination
- **🔄 Adaptive Systems**: Regime detection, dynamic thresholds, online calibration
- **⚡ Advanced Execution**: TWAP/VWAP algorithms, smart order routing, slippage tracking
- **📊 Portfolio Optimization**: Mean-variance, risk parity, Black-Litterman
- **📈 Prometheus Metrics**: 50+ metrics for execution quality and performance
- **🏥 Health Monitoring**: Real-time health checks, automated alerting
- **📊 Well Tested**: 505/505 tests passing with comprehensive coverage

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
vision = { path = "path/to/vision" }
burn = "0.19"
```

For GPU support:

```toml
[dependencies]
vision = { path = "path/to/vision", features = ["gpu"] }
```

## 🚀 Quick Start

### Basic Usage - Training with Real Data & Feature Engineering

```rust
use burn::backend::NdArray;
use vision::{
    load_ohlcv_csv, MinMaxScaler, OhlcvDataset, SequenceConfig,
    TensorConverter, TensorConverterConfig, TrainValSplit,
    FeatureConfig, FeatureEngineer,
};

type Backend = NdArray<f32>;

// 1. Load OHLCV data from CSV
let candles = load_ohlcv_csv("data/BTCUSDT_1h.csv")?;

// 2. Create sequences
let config = SequenceConfig {
    sequence_length: 60,
    stride: 1,
    prediction_horizon: 1,
};
let dataset = OhlcvDataset::from_candles(candles, config)?;

// 3. Split train/val/test
let split = TrainValSplit::default(); // 70/15/15
let (train, val, test) = dataset.split(split)?;

// 4. Feature engineering (optional - adds technical indicators)
let feature_config = FeatureConfig::with_common_indicators();
let engineer = FeatureEngineer::new(feature_config.clone());

// Parallel computation (3-4x faster)
let all_features = engineer.compute_dataset_features_parallel(&dataset)?;

// Computes: SMA, EMA, RSI, MACD, ATR, Bollinger Bands
// Total features: 19 (vs 5 for raw OHLCV)

// 5. Setup normalization
let mut converter = TensorConverter::<MinMaxScaler>::new(
    TensorConverterConfig {
        normalize: true,
        num_features: feature_config.num_features(),
    }
);
converter.fit(&train);

// 6. Convert to tensors
let device = Default::default();
let inputs = converter.dataset_to_tensor::<Backend>(&train, &device);
// Shape: [num_sequences, 60, 19] with indicators
```

### Model Usage

```rust
use burn::backend::{Autodiff, NdArray};
use vision::{DiffGafLstmConfig, DiffGafLstm};

type Backend = Autodiff<NdArray>;

// Configure model
let config = DiffGafLstmConfig {
    input_features: 5,      // OHLCV features
    time_steps: 60,         // Sequence length
    lstm_hidden_size: 64,   // LSTM hidden dimension
    num_lstm_layers: 2,     // Number of LSTM layers
    num_classes: 3,         // Buy/Sell/Hold
    dropout: 0.2,
    gaf_pool_size: 16,      // Pool GAF to 16×16
    bidirectional: false,
};

// Initialize model
let device = Default::default();
let model = config.init::<Backend>(&device);

// Forward pass
let input = Tensor::ones([batch_size, 60, 5], &device);
let output = model.forward(input); // [batch_size, 3]
```

### Training Examples

Run the synthetic training example:

```bash
cargo run --package vision --example train_diffgaf_lstm
```

Run the real data pipeline demo:

```bash
cargo run --package vision --example train_with_real_data
```

Run the feature engineering demo:

```bash
cargo run --package vision --example feature_engineering_demo
```

Run the production training pipeline:

```bash
cargo run --package vision --example production_training --release
```

Examples demonstrate:
- ✅ CSV data loading and validation
- ✅ Sequence generation and splitting
- ✅ Technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger)
- ✅ Feature engineering with 19+ features
- ✅ Feature normalization (MinMax, Z-Score, Robust)
- ✅ Tensor conversion for Burn
- ✅ Training loop with Adam optimizer
- ✅ Validation and metrics tracking
- ✅ Checkpoint management
- ✅ Early stopping
- ✅ Best model tracking
- ✅ Parallel feature computation (3-4x speedup)
- ✅ Production training pipeline

## 🏗️ Architecture

### DiffGAF Pipeline

```
Time Series [B, T, F]
      ↓
  LearnableNorm (learnable min/max)
      ↓
  PolarEncoder (angle encoding)
      ↓
  GramianLayer (GAF computation)
      ↓
GAF Images [B, F, T, T]
      ↓
  AdaptiveAvgPool2d (→ K×K)
      ↓
Pooled GAF [B, F, K, K]
      ↓
  Flatten + Project
      ↓
  Multi-layer LSTM
      ↓
  MLP Classifier
      ↓
Predictions [B, num_classes]
```

### Memory Optimization

Adaptive pooling reduces memory footprint:

| Time Steps | Pool Size | Memory Reduction |
|-----------|-----------|------------------|
| 60        | 16        | **14.06×**       |
| 64        | 32        | **4.00×**        |
| 128       | 64        | **4.00×**        |

## 💾 Checkpointing

### Save Model

```rust
use vision::{CheckpointMetadata, DiffGafLstm};

// Save checkpoint
let metadata = CheckpointMetadata::new(
    epoch,
    train_loss,
    Some(val_loss),
    config.clone(),
);
model.save_checkpoint("checkpoints/epoch_10", metadata)?;
```

Creates two files:
- `epoch_10.bin` - Binary weights
- `epoch_10.meta.json` - Human-readable metadata

### Load Model

```rust
// Load checkpoint
let (model, metadata) = DiffGafLstm::load_checkpoint(
    "checkpoints/epoch_10",
    &device,
)?;

println!("Loaded from epoch {}", metadata.epoch);
```

### Best Model Tracking

```rust
use vision::BestModelTracker;

let mut tracker = BestModelTracker::new("checkpoints/best");

// Training loop
for epoch in 1..=100 {
    let train_loss = train_epoch(&model, &train_data);
    let val_loss = validate(&model, &val_data);
    
    // Automatically save if new best
    if tracker.update(&model, epoch, train_loss, val_loss, config.clone())? {
        println!("✨ New best model saved!");
    }
}
```

## 🧪 Testing

Run all tests:

```bash
cargo test --package vision
```

Run specific test categories:

```bash
# Checkpoint tests
cargo test --package vision --lib checkpoint

# Pooling tests  
cargo test --package vision --lib pooling

# DiffGAF tests
cargo test --package vision --lib diffgaf
```

## 📊 Performance

### Benchmarks (CPU)

```bash
cargo bench --package vision
```

### With GPU

```bash
cargo build --package vision --features gpu
cargo test --package vision --features gpu
```

Expected speedup with CUDA: **10-50× faster** for Gramian computation.

## 🎯 Model Configuration

### Default Config

```rust
DiffGafLstmConfig {
    input_features: 5,
    time_steps: 60,
    lstm_hidden_size: 32,
    num_lstm_layers: 2,
    num_classes: 3,
    dropout: 0.3,
    gaf_pool_size: 32,
    bidirectional: false,
}
```

### Recommended Settings

| Use Case | Pool Size | LSTM Hidden | Layers | Dropout |
|----------|-----------|-------------|--------|---------|
| Small dataset | 16 | 32 | 1-2 | 0.3-0.4 |
| Medium dataset | 32 | 64 | 2 | 0.2-0.3 |
| Large dataset | 64 | 128 | 2-3 | 0.1-0.2 |

## 📁 Project Structure

```
crates/vision/
├── benches/          # Performance benchmarks
├── examples/         # Training and trading examples
│   ├── train_diffgaf_lstm.rs               # Synthetic data training
│   ├── train_with_real_data.rs             # Real CSV pipeline
│   ├── feature_engineering_demo.rs         # Technical indicators demo
│   ├── production_training.rs              # Complete training pipeline
│   ├── live_pipeline.rs                    # Real-time prediction
│   ├── live_pipeline_with_execution.rs     # Full trading system (NEW)
│   ├── metrics_server.rs                   # Prometheus metrics HTTP server (NEW)
│   ├── execution_demo.rs                   # Order execution examples
│   ├── portfolio_demo.rs                   # Portfolio optimization
│   └── end_to_end_trading_system.rs        # Complete integration
├── src/
│   ├── adaptive/     # Adaptive systems (Week 7, Day 2)
│   │   ├── calibration.rs      # Confidence calibration
│   │   ├── regime.rs           # Regime detection
│   │   └── threshold.rs        # Dynamic thresholds
│   ├── backtest/     # Backtesting framework
│   │   ├── metrics.rs          # Performance metrics
│   │   ├── simulation.rs       # Backtest simulation
│   │   └── trade.rs            # Trade tracking
│   ├── data/         # Real data loading (Week 6)
│   │   ├── csv_loader.rs       # CSV OHLCV loader
│   │   ├── dataset.rs          # Sequence generation
│   │   └── validation.rs       # Data quality checks
│   ├── diffgaf/      # DiffGAF implementation (Week 5)
│   │   ├── combined.rs         # DiffGAF+LSTM model
│   │   ├── config.rs           # Configuration
│   │   ├── layers.rs           # DiffGAF layer
│   │   └── transforms.rs       # Norm, encoder, Gramian
│   ├── ensemble/     # Ensemble learning (Week 7, Day 1)
│   │   ├── models.rs           # Model wrapper
│   │   └── stacking.rs         # Stacking ensemble
│   ├── execution/    # Advanced order execution (Week 8, Day 3)
│   │   ├── analytics.rs        # Execution quality tracking
│   │   ├── instrumented.rs     # Metrics-enabled manager (NEW)
│   │   ├── manager.rs          # Order lifecycle management
│   │   ├── metrics.rs          # Prometheus metrics (NEW)
│   │   ├── twap.rs             # TWAP algorithm
│   │   └── vwap.rs             # VWAP algorithm
│   ├── live/         # Live trading pipeline
│   │   ├── cache.rs            # Prediction caching
│   │   ├── inference.rs        # Real-time inference
│   │   ├── latency.rs          # Latency monitoring
│   │   └── streaming.rs        # Streaming data handler
│   ├── portfolio/    # Portfolio optimization (Week 8, Day 4)
│   │   ├── analytics.rs        # Portfolio analytics
│   │   ├── black_litterman.rs  # Black-Litterman model
│   │   ├── covariance.rs       # Covariance estimation
│   │   ├── mean_variance.rs    # Mean-variance optimization
│   │   └── risk_parity.rs      # Risk parity optimization
│   ├── preprocessing/     # Feature engineering (Week 6)
│   │   ├── features.rs         # Technical indicators + parallel compute
│   │   ├── normalization.rs    # MinMax, Z-Score, Robust
│   │   └── tensor_conversion.rs # Burn tensor utils
│   ├── production/   # Production utilities
│   │   ├── health.rs           # Health checks
│   │   └── metrics.rs          # Production metrics
│   ├── risk/         # Risk management
│   │   ├── kelly.rs            # Kelly criterion
│   │   ├── position_sizing.rs  # Position sizing
│   │   └── mod.rs              # Risk manager
│   ├── signals/      # Signal generation
│   │   ├── integration.rs      # Signal integration
│   │   └── types.rs            # Signal types
│   ├── visualization/     # GAF visualization (optional)
│   ├── config.rs     # Vision config
│   ├── error.rs      # Error types
│   └── lib.rs        # Public API
├── docs/             # Documentation
│   ├── week5_*.md    # Week 5 summaries (DiffGAF)
│   ├── week6_*.md    # Week 6 summaries (Real data)
│   ├── week7_*.md    # Week 7 summaries (Ensemble, Adaptive)
│   ├── week8_*.md    # Week 8 summaries (Execution, Portfolio, Metrics)
│   ├── WEEK8_EXECUTION_METRICS_COMPLETE.md  # Week 8 complete summary
│   ├── EXECUTION_METRICS_QUICKREF.md        # Metrics quick reference
│   └── WEEK9_ROADMAP.md                     # Production deployment plan
└── tests/            # Integration tests (505 passing)
```

## 🔬 Implementation Details

### Learnable Components

1. **LearnableNorm**: Learnable min/max normalization parameters
2. **PolarEncoder**: Safe arccos approximation using Taylor series
3. **GramianLayer**: Differentiable GAF computation with GPU path

### Key Features

- ✅ **Full Precision**: Uses `FullPrecisionSettings` for checkpoints
- ✅ **Device Agnostic**: Works on CPU and GPU
- ✅ **Type Safe**: Compile-time shape checking with const generics
- ✅ **Numerically Stable**: Safe approximations for trig functions
- ✅ **Memory Efficient**: Block-based pooling algorithm

## 📚 Documentation

### Generated Docs

```bash
cargo doc --package vision --open
```

### Implementation Guides

See the [docs](../../docs/) directory:

**Week 5: DiffGAF Vision Pipeline**
- [WEEK5_DAY1_COMPLETE.md](../../docs/WEEK5_DAY1_COMPLETE.md) - DiffGAF core
- [WEEK5_DAY2_COMPLETE.md](../../docs/WEEK5_DAY2_COMPLETE.md) - GPU & training
- [WEEK5_DAY3_COMPLETE.md](../../docs/WEEK5_DAY3_COMPLETE.md) - Pooling & memory
- [WEEK5_DAY4_COMPLETE.md](../../docs/WEEK5_DAY4_COMPLETE.md) - Checkpointing

**Week 6: Real Data Integration**
- [WEEK6_DAY1_COMPLETE.md](docs/WEEK6_DAY1_COMPLETE.md) - Data loading infrastructure
- [WEEK6_DAY2_COMPLETE.md](docs/WEEK6_DAY2_COMPLETE.md) - Feature engineering & preprocessing
- [WEEK6_DAY3_COMPLETE.md](docs/WEEK6_DAY3_COMPLETE.md) - Technical indicators integration
- [WEEK6_DAY4_COMPLETE.md](docs/WEEK6_DAY4_COMPLETE.md) - Production pipeline & optimization

## 🤝 Contributing

**Week 5 Status: COMPLETE** ✅
- [x] DiffGAF core (LearnableNorm, PolarEncoder, GramianLayer)
- [x] GPU optimization
- [x] Training integration (TrainStep/ValidStep)
- [x] Memory-efficient pooling (14× reduction)
- [x] Model checkpointing & persistence
- [x] Best model tracking
- [x] Training examples
- [x] Comprehensive testing (24/24 passing)

**Week 6 Status: COMPLETE** ✅
- [x] Day 1: Data loading infrastructure (CSV, validation, sequences)
- [x] Day 2: Feature engineering & preprocessing (normalization, tensors)
- [x] Day 3: Technical indicators integration (SMA, EMA, RSI, MACD, ATR, Bollinger)
- [x] Day 4: Production pipeline & optimization (parallel compute, training loop)

**Week 7 Status: COMPLETE** ✅
- [x] Day 1: Ensemble learning (model stacking, weighted averaging, 18 tests)
- [x] Day 2: Adaptive systems (regime detection, dynamic thresholds, 28 tests)

**Week 8 Status: COMPLETE** ✅
- [x] Day 1-2: Advanced execution (TWAP/VWAP algorithms, 67 tests)
- [x] Day 3: Execution analytics (slippage tracking, quality scoring)
- [x] Day 4: Portfolio optimization (MV, RP, BL, 34 tests)
- [x] Day 5: End-to-end integration (complete trading pipeline)
- [x] Day 6: Prometheus metrics & monitoring (50+ metrics, health checks)

**Current Test Status: 505/505 passing** ✅

## 📄 License

MIT OR Apache-2.0

## 🙏 Acknowledgments

Built as part of the JANUS 24-week roadmap for algorithmic trading.

- **Burn Framework**: [burn.dev](https://burn.dev)
- **Gramian Angular Fields**: [arXiv:1506.00327](https://arxiv.org/abs/1506.00327)

## 🚀 Getting Started

1. **Clone and build**:
   ```bash
   cd crates/vision
   cargo build --release
   ```

2. **Run tests**:
   ```bash
   cargo test
   ```

3. **Try the example**:
   ```bash
   cargo run --example train_diffgaf_lstm
   ```

4. **Check results**:
   ```bash
   ls checkpoints/
   # best_model.bin, best_model.meta.json, epoch_10.bin, ...
   ```

## 📊 Results

### DiffGAF Training Results

Training on synthetic data (1000 samples, 60 timesteps, 5 features):

```
Epoch  1/50: train_loss=0.0854, val_loss=0.0000, val_acc=100.00% ⭐
Epoch 11/50: train_loss=0.0000, val_loss=0.0000, val_acc=100.00%

⏸️  Early stopping triggered at epoch 11
✅ Training complete!

=== Training Summary ===
Final val accuracy: 100.00%
Best val accuracy: 100.00%
```

### Production Trading System Results

Full pipeline simulation (200 ticks):

```
═══ Production Pipeline Performance ═══

Throughput
  Total Ticks:      200
  Throughput:       40.2 ticks/sec

Latency
  Mean:             245.3 μs
  Median (p50):     198 μs
  p95:              512 μs
  p99:              876 μs

Signal Generation
  Signals Generated: 47 (23.5%)
  Trades Executed:   41 (87.2% conversion)

Execution Quality
  Average Slippage:  2.4 bps
  Quality Score:     94.7/100
```

## 🔗 Links

- **Documentation**: Run `cargo doc --open`
- **Examples**: See `examples/` directory (13+ examples)
- **Tests**: Run `cargo test --package vision` (505 tests)
- **Benchmarks**: Run `cargo bench --package vision`
- **Metrics Server**: `cargo run --example metrics_server --release`
- **Live Trading**: `cargo run --example live_pipeline_with_execution --release`

---

**Status**: Production Ready 🎉  
**Version**: 0.1.0  
**Week 5**: Complete ✅ (DiffGAF Vision Pipeline)  
**Week 6**: Complete ✅ (Real Data Integration)  
**Week 7**: Complete ✅ (Ensemble & Adaptive Systems)  
**Week 8**: Complete ✅ (Execution, Portfolio, Metrics)  
**Next**: Week 9 - Production Deployment (Docker, K8s, CI/CD)