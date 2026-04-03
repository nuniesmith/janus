# DiffGAF Vision - Quick Reference Card

**Week 6: Real Data Integration**

## 🚀 Quick Start (30 seconds)

```rust
use vision::*;

// 1. Load data
let candles = load_ohlcv_csv("data.csv")?;

// 2. Create sequences
let dataset = OhlcvDataset::from_candles(candles, SequenceConfig {
    sequence_length: 60,
    stride: 1,
    prediction_horizon: 1,
})?;

// 3. Engineer features (parallel)
let engineer = FeatureEngineer::new(FeatureConfig::with_common_indicators());
let features = engineer.compute_dataset_features_parallel(&dataset)?;

// 4. Normalize
let mut converter = TensorConverter::<ZScoreScaler>::new(TensorConverterConfig {
    normalize: true,
    num_features: engineer.get_num_features(),
});
converter.fit_with_features(&features);

// 5. Train
let (inputs, targets) = create_batch_with_features(&features, &labels, &converter, &device);
let output = model.forward(inputs);
```

---

## 📊 Data Loading

### Load CSV
```rust
let candles = load_ohlcv_csv("data/BTCUSD.csv")?;
```

**CSV Format**:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,100.0,105.0,99.0,103.0,1000000
```

### Validate Data
```rust
let report = dataset.validate()?;
println!("{}", report.summary());
```

---

## 🔧 Feature Engineering

### Preset Configs
```rust
// Minimal (6 features)
FeatureConfig::minimal()

// Common indicators (19 features)
FeatureConfig::with_common_indicators()

// Custom
FeatureConfig {
    include_ohlcv: true,
    include_returns: true,
    sma_periods: vec![10, 20, 50],
    ema_periods: vec![12, 26],
    rsi_period: Some(14),
    macd_config: Some((12, 26, 9)),
    atr_period: Some(14),
    bollinger_bands: Some((20, 2.0)),
    ..Default::default()
}
```

### Compute Features

**Sequential**:
```rust
let features = engineer.compute_dataset_features(&dataset)?;
```

**Parallel** (3-4x faster):
```rust
let features = engineer.compute_dataset_features_parallel(&dataset)?;
```

---

## 📐 Normalization

### Available Scalers

**MinMaxScaler**: Range [0, 1] or custom
```rust
TensorConverter::<MinMaxScaler>::new(config)
```

**ZScoreScaler**: Mean=0, Std=1
```rust
TensorConverter::<ZScoreScaler>::new(config)
```

**RobustScaler**: Outlier-resistant
```rust
TensorConverter::<RobustScaler>::new(config)
```

### Workflow
```rust
// 1. Create converter
let mut converter = TensorConverter::<ZScoreScaler>::new(config);

// 2. Fit on training data ONLY
converter.fit_with_features(&train_features);

// 3. Transform train and validation
let train_tensors = converter.features_batch_to_tensor(&train_features, &device);
let val_tensors = converter.features_batch_to_tensor(&val_features, &device);
```

---

## 🎯 Model Training

### Initialize Model
```rust
let model_config = DiffGafLstmConfig {
    input_features: 19,        // From engineer.get_num_features()
    time_steps: 60,            // Sequence length
    lstm_hidden_size: 128,
    num_lstm_layers: 2,
    num_classes: 3,            // buy/sell/hold
    dropout: 0.3,
    gaf_pool_size: 32,
    bidirectional: false,
};

let model = model_config.init::<MyBackend>(&device);
```

### Training Loop
```rust
let mut optimizer = AdamConfig::new().init();

for epoch in 0..epochs {
    // Train
    let (model, train_loss) = train_epoch(
        model,
        &mut optimizer,
        &train_features,
        &train_labels,
        &converter,
        batch_size,
        &device,
    );
    
    // Validate
    let val_loss = validate_epoch(
        &model.valid(),
        &val_features,
        &val_labels,
        &converter,
        batch_size,
        &device,
    );
    
    println!("Epoch {}: train={:.4}, val={:.4}", epoch, train_loss, val_loss);
}
```

---

## 🏃 Running Examples

### Basic Pipeline
```bash
cargo run --package vision --example train_with_real_data
```

### Feature Engineering Demo
```bash
cargo run --package vision --example feature_engineering_demo
```

### Production Training
```bash
cargo run --package vision --example production_training --release
```

### Tests
```bash
cargo test --package vision --lib
```

---

## 📦 Batch Processing

### Create Batches
```rust
let (inputs, targets) = create_batch_with_features(
    &batch_features,  // [batch_size, seq_len, num_features]
    &batch_labels,    // [batch_size]
    &converter,
    &device,
);
// inputs: Tensor<B, 3>
// targets: Tensor<B, 1>
```

### Batch Iterator
```rust
let mut iter = BatchIterator::new(&dataset, batch_size);

while let Some((indices, is_last)) = iter.next_batch() {
    // Process batch
    if is_last { break; }
}
```

---

## ⚡ Performance Tips

### 1. Use Parallel Computation
```rust
// ✅ Fast
engineer.compute_dataset_features_parallel(&dataset)?;

// ❌ Slow
engineer.compute_dataset_features(&dataset)?;
```

### 2. Precompute Features
```rust
// ✅ Fast - compute once
let all_features = engineer.compute_dataset_features_parallel(&dataset)?;
for epoch in 0..100 {
    for batch in batches(&all_features) { /* train */ }
}

// ❌ Slow - recompute every epoch
for epoch in 0..100 {
    let features = engineer.compute_dataset_features_parallel(&dataset)?;
    /* train */
}
```

### 3. Use Release Mode
```bash
# ✅ 10-50x faster
cargo run --release

# ❌ Slow for training
cargo run
```

### 4. Choose Minimal Features
```rust
// Fast: 6 features, small memory
FeatureConfig::minimal()

// Slower: 19 features, larger memory
FeatureConfig::with_common_indicators()
```

---

## 🐛 Common Issues

### "Transform before fit"
**Error**: Panic when calling `features_to_tensor` without fitting.

**Fix**: Always call `fit_with_features` first:
```rust
converter.fit_with_features(&train_features);  // Must call first!
let tensors = converter.features_to_tensor(&test_features, &device);
```

### NaN in Features
**Cause**: Indicators need warmup period (e.g., SMA-20 needs 20 points).

**Fix**: Use longer sequences or check `num_features()`:
```rust
if dataset.sequences[0].len() < 60 {
    eprintln!("Warning: sequence too short for indicators");
}
```

### Out of Memory
**Cause**: Too many features or large batch size.

**Fix**:
1. Reduce batch size
2. Use fewer features
3. Reduce sequence length

### Slow Training
**Cause**: Debug mode or sequential computation.

**Fix**:
1. Use `--release` flag
2. Use parallel feature computation
3. Precompute features outside training loop

---

## 📊 Feature Counts

| Config | Features | Description |
|--------|----------|-------------|
| Raw OHLCV | 5 | open, high, low, close, volume |
| + Returns | 6 | + price change |
| + Log Returns | 7 | + log price change |
| + Volume Change | 8 | + volume delta |
| Minimal | 6 | OHLCV + returns |
| Common Indicators | 19 | + SMAs, EMAs, RSI, MACD, ATR, BB |
| Extended | 25+ | Custom configuration |

---

## 🔗 Links

- **Full Docs**: `cargo doc --package vision --open`
- **Day 1**: [WEEK6_DAY1_COMPLETE.md](./WEEK6_DAY1_COMPLETE.md)
- **Day 2**: [WEEK6_DAY2_COMPLETE.md](./WEEK6_DAY2_COMPLETE.md)
- **Day 3**: [WEEK6_DAY3_COMPLETE.md](./WEEK6_DAY3_COMPLETE.md)
- **Day 4**: [WEEK6_DAY4_COMPLETE.md](./WEEK6_DAY4_COMPLETE.md)
- **Summary**: [WEEK6_SUMMARY.md](./WEEK6_SUMMARY.md)

---

## 💡 Cheat Sheet

```rust
// Complete pipeline in ~30 lines
use vision::*;

let candles = load_ohlcv_csv("data.csv")?;
let dataset = OhlcvDataset::from_candles(candles, SequenceConfig::default())?;

let config = FeatureConfig::with_common_indicators();
let engineer = FeatureEngineer::new(config);
let all_features = engineer.compute_dataset_features_parallel(&dataset)?;

let (train_f, val_f) = split_features(&all_features, 0.8);
let (train_l, val_l) = split_labels(&dataset.labels, 0.8);

let mut converter = TensorConverter::<ZScoreScaler>::new(TensorConverterConfig {
    normalize: true,
    num_features: engineer.get_num_features(),
});
converter.fit_with_features(&train_f);

let model_config = DiffGafLstmConfig {
    input_features: engineer.get_num_features(),
    time_steps: 60,
    lstm_hidden_size: 128,
    num_classes: 3,
    ..Default::default()
};

let model = model_config.init(&device);
let mut optimizer = AdamConfig::new().init();

// Train!
for epoch in 0..50 {
    let (model, loss) = train_epoch(model, &mut optimizer, ...);
    println!("Epoch {}: loss={:.4}", epoch, loss);
}
```

---

**Status**: Production Ready 🎉  
**Tests**: 88/88 passing ✅  
**Week**: 6/24 Complete ✅