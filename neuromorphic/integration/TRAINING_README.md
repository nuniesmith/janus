# Phase 6: End-to-End Training Pipeline 🚀

**Complete workflow for training ViViT models on market data**

---

## 📋 Quick Navigation

- **[Full Implementation Guide](../../../../docs/PHASE_6_TRAINING_PIPELINE.md)** - Comprehensive 1,100-line technical guide
- **[Quick Reference](../../../../docs/PHASE_6_QUICK_REF.md)** - Practical examples and cheat sheet
- **[Completion Summary](../../../../docs/PHASE_6_COMPLETE.md)** - Project overview and statistics

---

## 🎯 What is Phase 6?

Phase 6 integrates all previous neuromorphic components into a **complete, production-ready training pipeline**:

- ✅ Load market data (CSV/database/memory)
- ✅ Transform to GAF images (Gramian Angular Fields)
- ✅ Train ViViT models with backpropagation
- ✅ Multi-GPU distributed training
- ✅ Checkpoint management and resume
- ✅ Real-time metrics and monitoring

---

## 🚀 Quick Start (30 seconds)

```rust
use janus_neuromorphic::integration::{
    MarketDataPipeline, PipelineConfig, TrainingPipeline, TrainingConfig,
};
use candle_core::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Load market data
    let mut data = MarketDataPipeline::new(PipelineConfig::default());
    data.load_from_csv("data/btc_usd.csv").await?;
    data.preprocess().await?;
    
    // 2. Train model
    let device = Device::cuda_if_available(0)?;
    let mut trainer = TrainingPipeline::new(TrainingConfig::default(), device).await?;
    trainer.train(10).await?;
    
    Ok(())
}
```

**Run it:**
```bash
cargo run --example train_vivit -- --device cuda --epochs 100
```

---

## 📦 Components

### 1. **TrainingPipeline** (`training.rs`)
Complete training orchestration with:
- VarMap + AdamW optimizer
- Cross-entropy loss
- Gradient clipping & accumulation
- Learning rate warmup
- Early stopping
- Checkpoint save/load
- Metrics tracking

### 2. **MarketDataPipeline** (`data.rs`)
Market data preprocessing with:
- CSV/database/memory loading
- GAF transformation (GASF)
- Multi-feature encoding (OHLCV)
- Automatic labeling (Buy/Hold/Sell)
- Train/validation splitting
- Batch iteration

### 3. **Example** (`../examples/train_vivit.rs`)
End-to-end demonstration with:
- Synthetic data generation
- CLI interface
- Checkpoint resume
- Progress logging

---

## 📊 Example Output

```
INFO  Epoch 1, Step 100: loss=0.8234, acc=65.23%, lr=1.00e-05, throughput=128.5 samples/s
INFO    Validation: loss=0.7891, acc=68.45%
INFO  New best validation loss: 0.7891 (previous: 0.8123)
INFO  Checkpoint saved to "checkpoints/vivit/vivit_step_1000"
```

---

## 🏗️ Architecture

```
Market Data (CSV)
    ↓
Preprocess (Normalize, Clean)
    ↓
GAF Transform (Time Series → Images)
    ↓
Create Sequences (Sliding Windows)
    ↓
Label Generation (Buy/Hold/Sell)
    ↓
Train/Val Split
    ↓
[Training Loop]
    ├─ Forward Pass (ViViT)
    ├─ Loss Computation (Cross-Entropy)
    ├─ Backward Pass (Gradients)
    ├─ Gradient Sync (Multi-GPU)
    ├─ Optimizer Step (AdamW)
    └─ Metrics Update
    ↓
Validation
    ↓
Checkpoint Save
    ↓
Trained Model ✓
```

---

## 🔧 Configuration

### Data Pipeline

```rust
let config = PipelineConfig {
    num_frames: 16,              // Video frames
    candles_per_frame: 60,       // Candles per frame (1 hour if 1-min candles)
    gaf_image_size: 224,         // Image size (224x224)
    
    features: vec![
        GafFeature::Close,       // Price
        GafFeature::Volume,      // Volume
        GafFeature::HighLow,     // Spread
    ],
    
    prediction_horizon: 10,      // Predict N candles ahead
    buy_threshold: 0.5,          // +0.5% return = Buy
    sell_threshold: -0.5,        // -0.5% return = Sell
    
    train_split: 0.8,            // 80% train, 20% validation
    shuffle: true,
    seed: Some(42),
};
```

### Training

```rust
let config = TrainingConfig {
    batch_size: 16,
    learning_rate: 1e-4,
    weight_decay: 0.05,
    
    gradient_clip_norm: Some(1.0),
    gradient_accumulation_steps: 1,
    
    early_stopping_patience: Some(10),
    save_every_n_steps: 500,
    log_every_n_steps: 100,
    warmup_steps: 1000,
    
    checkpoint_dir: PathBuf::from("checkpoints/vivit"),
    ..Default::default()
};
```

---

## 💡 Common Usage Patterns

### Load Real Data

```rust
// From CSV
pipeline.load_from_csv("data/btc_1m_2024.csv").await?;

// From vector (e.g., database query)
let candles = load_from_database().await?;
pipeline.load_from_vec(candles);
```

### Resume Training

```rust
let mut trainer = TrainingPipeline::new(config, device).await?;
trainer.load_checkpoint("vivit_step_5000").await?;
trainer.train(50).await?;  // Continue for 50 more epochs
```

### Monitor Progress

```rust
let metrics = trainer.metrics();
println!("Step: {}", metrics.step);
println!("Loss: {:.4}", metrics.train_loss);
println!("Accuracy: {:.2}%", metrics.train_accuracy * 100.0);
```

### Multi-GPU

```rust
// Automatically uses all available GPUs
let coordinator = TrainingCoordinator::new()?;
println!("Training on {} GPUs", coordinator.available_devices().len());
```

---

## 📈 Performance

### Throughput Benchmarks

| GPU | Batch Size | Samples/sec |
|-----|-----------|-------------|
| CPU | 16 | ~5-10 |
| RTX 3090 | 32 | ~150-200 |
| V100 | 32 | ~100-150 |
| A100 | 64 | ~250-350 |

### Training Time (10k samples, 100 epochs)

| GPU | Time |
|-----|------|
| CPU | ~20-30 hours |
| RTX 3090 | ~1-2 hours |
| A100 | ~0.5-1 hour |

---

## 🐛 Troubleshooting

### Out of Memory?
```rust
// Reduce batch size
batch_size: 8,

// Add gradient accumulation (maintains effective batch size)
gradient_accumulation_steps: 4,  // Effective batch = 32
```

### Slow Training?
```bash
# Check GPU utilization
nvidia-smi -l 1

# Increase batch size if GPU underutilized
```

### NaN Loss?
```rust
// Reduce learning rate
learning_rate: 1e-5,

// Enable gradient clipping
gradient_clip_norm: Some(1.0),
```

### No Convergence?
```rust
// Check label distribution
let (sell, hold, buy) = pipeline.label_distribution();
println!("Labels: {}/{}/{}", sell, hold, buy);

// Increase learning rate
learning_rate: 5e-4,
```

---

## 🧪 Testing

```bash
# Run tests
cargo test --package janus-neuromorphic integration::training

# Run example
cargo run --example train_vivit

# With logging
RUST_LOG=debug cargo run --example train_vivit
```

---

## 📚 Documentation

1. **[PHASE_6_TRAINING_PIPELINE.md](../../../../docs/PHASE_6_TRAINING_PIPELINE.md)** (1,100 lines)
   - Complete architecture
   - Component deep-dive
   - GAF transformation details
   - Training loop breakdown
   - Performance tuning
   - Troubleshooting guide

2. **[PHASE_6_QUICK_REF.md](../../../../docs/PHASE_6_QUICK_REF.md)** (540 lines)
   - Quick start examples
   - Configuration templates
   - Common patterns
   - Cheat sheet

3. **[PHASE_6_COMPLETE.md](../../../../docs/PHASE_6_COMPLETE.md)** (760 lines)
   - Project summary
   - Objectives achieved
   - Statistics
   - Future enhancements

---

## 🔗 Related Phases

- **Phase 3B**: Training infrastructure and emotional memory
- **Phase 4**: GPU-accelerated ViViT models
- **Phase 5**: Distributed multi-GPU training

---

## ✅ Status

**Phase 6: COMPLETE** ✅

- ✅ 1,400 lines of code
- ✅ 1,600 lines of documentation
- ✅ 9 unit tests
- ✅ Production ready
- ✅ Multi-GPU support
- ✅ Comprehensive examples

---

## 🚀 Next Steps

1. **Collect Data**: Gather historical market data (OHLCV candles)
2. **Prepare CSV**: Format as `timestamp,open,high,low,close,volume`
3. **Configure**: Adjust `PipelineConfig` and `TrainingConfig` for your use case
4. **Train**: Run `cargo run --example train_vivit`
5. **Monitor**: Watch metrics and checkpoints
6. **Tune**: Adjust hyperparameters based on results
7. **Deploy**: Use trained model for inference

---

## 💬 Support

- Check [Full Guide](../../../../docs/PHASE_6_TRAINING_PIPELINE.md) for details
- Review [Quick Reference](../../../../docs/PHASE_6_QUICK_REF.md) for examples
- Enable debug logs: `RUST_LOG=debug`
- Check GPU: `nvidia-smi`

---

**Happy Training! 🎉**