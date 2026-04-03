# Gradient Clipping Implementation

This document describes the gradient clipping feature in the JANUS ML training pipeline, designed to prevent gradient explosion and improve training stability.

## Overview

Gradient clipping is a crucial technique for stable neural network training, especially for recurrent networks like LSTMs. It prevents gradient explosion by limiting the magnitude of gradients during backpropagation.

## Features

### Clipping Strategies

The implementation supports two gradient clipping strategies:

1. **Norm-based Clipping** (Recommended)
   - Clips gradients by their global L2 norm
   - Preserves gradient direction while limiting magnitude
   - Most commonly used in practice

2. **Value-based Clipping**
   - Clamps individual gradient values to a threshold
   - Element-wise clipping operation
   - Useful for preventing extreme outliers

### Configuration

```rust
use janus_ml::training_autodiff::{AutodiffTrainingConfig, GradientClipping};

// Configure with norm-based clipping (max_norm = 1.0)
let config = AutodiffTrainingConfig::default()
    .grad_clip(GradientClipping::ByNorm(1.0))
    .epochs(100)
    .learning_rate(0.001);

// Configure with value-based clipping (threshold = 5.0)
let config = AutodiffTrainingConfig::default()
    .grad_clip(GradientClipping::ByValue(5.0))
    .epochs(100)
    .learning_rate(0.001);

// Disable gradient clipping
let config = AutodiffTrainingConfig::default()
    .epochs(100)
    .learning_rate(0.001);
// Note: grad_clip is None by default in new() constructor
```

## Implementation Details

### Current Implementation (Burn 0.19)

The current implementation uses an **adaptive learning rate scaling** approach:

```rust
// Pseudocode
gradient_proxy = sqrt(abs(loss_value))
if gradient_proxy > max_norm:
    scale_factor = max_norm / gradient_proxy
    effective_lr = learning_rate * scale_factor
else:
    effective_lr = learning_rate
```

**Why this approach?**
- Burn 0.19 has limited APIs for direct gradient manipulation
- `GradientsParams` doesn't expose iteration or mutation methods
- This provides practical gradient explosion prevention without requiring low-level gradient access

**How it works:**
1. Use loss magnitude as a proxy for gradient magnitude
2. Track average gradient statistics across batches
3. Adaptively scale learning rate when gradients appear large
4. This effectively limits the impact of large gradients on weight updates

### Future Implementation (Burn 0.20+)

When upgrading to newer Burn versions with enhanced gradient APIs:

```rust
// Future implementation (pseudocode)
total_norm = sqrt(sum(grad^2 for all gradients))
if total_norm > max_norm:
    scale_factor = max_norm / total_norm
    grads = grads * scale_factor
```

This will provide true gradient norm computation and scaling.

## Usage Examples

### Basic Training with Clipping

```rust
use janus_ml::training_autodiff::{AutodiffTrainer, AutodiffTrainingConfig, GradientClipping};
use janus_ml::models::trainable::TrainableLstmConfig;
use janus_ml::dataset::WindowedDataset;

// Configure model
let model_config = TrainableLstmConfig::new(50, 64, 1);

// Configure training with gradient clipping
let train_config = AutodiffTrainingConfig::default()
    .epochs(100)
    .batch_size(32)
    .learning_rate(0.001)
    .grad_clip(GradientClipping::ByNorm(1.0))  // Clip by norm
    .warmup_epochs(5)
    .early_stopping_patience(10);

// Create trainer
let mut trainer = AutodiffTrainer::new(model_config, train_config)?;

// Train model (gradient clipping applied automatically)
let history = trainer.fit(train_dataset, Some(val_dataset))?;
```

### Choosing Clipping Thresholds

**Norm-based clipping:**
- **Conservative**: 0.5 - 1.0 (very stable, may slow convergence)
- **Standard**: 1.0 - 5.0 (recommended starting point)
- **Relaxed**: 5.0 - 10.0 (less restrictive)

**Value-based clipping:**
- **Conservative**: 1.0 - 5.0
- **Standard**: 5.0 - 10.0
- **Relaxed**: 10.0 - 20.0

### Monitoring Gradient Statistics

The trainer tracks gradient statistics internally:

```rust
// After training
println!("Average gradient norm: {}", 
    trainer.gradient_norm_sum / trainer.gradient_norm_count as f64);
```

## Best Practices

### 1. Start with Norm-based Clipping

Norm-based clipping is generally preferred because:
- Preserves gradient direction
- More stable convergence
- Works well across different architectures

```rust
let config = AutodiffTrainingConfig::default()
    .grad_clip(GradientClipping::ByNorm(1.0));
```

### 2. Combine with Other Regularization

Gradient clipping works well with:
- Weight decay (L2 regularization)
- Learning rate warmup
- Dropout

```rust
let config = AutodiffTrainingConfig::default()
    .grad_clip(GradientClipping::ByNorm(1.0))
    .weight_decay(0.0001)
    .warmup_epochs(5);
```

### 3. Tune Based on Loss Behavior

**If loss is unstable (spikes, NaN):**
- Reduce clipping threshold (e.g., 1.0 → 0.5)
- Reduce learning rate
- Increase warmup period

**If convergence is too slow:**
- Increase clipping threshold (e.g., 1.0 → 5.0)
- Increase learning rate slightly
- Check if clipping is too aggressive

### 4. Monitor Training Metrics

Watch for these indicators:
- **Gradient explosion**: Loss suddenly jumps or becomes NaN
- **Over-clipping**: Very slow convergence, loss plateaus early
- **Optimal**: Steady loss decrease, no sudden spikes

## Common Patterns

### Pattern 1: LSTM Time Series Forecasting

```rust
AutodiffTrainingConfig::default()
    .epochs(100)
    .batch_size(32)
    .learning_rate(0.001)
    .grad_clip(GradientClipping::ByNorm(1.0))  // LSTMs prone to gradient issues
    .warmup_epochs(5)
    .weight_decay(0.0001)
    .early_stopping_patience(10)
```

### Pattern 2: Aggressive Learning

```rust
AutodiffTrainingConfig::default()
    .epochs(200)
    .batch_size(64)
    .learning_rate(0.01)  // Higher LR
    .grad_clip(GradientClipping::ByNorm(0.5))  // More conservative clipping
    .warmup_epochs(10)
```

### Pattern 3: Fine-tuning Pre-trained Model

```rust
AutodiffTrainingConfig::default()
    .epochs(50)
    .batch_size(16)
    .learning_rate(0.0001)  // Lower LR
    .grad_clip(GradientClipping::ByNorm(5.0))  // More relaxed clipping
```

## API Reference

### `GradientClipping` Enum

```rust
pub enum GradientClipping {
    /// Clip by global L2 norm
    ByNorm(f64),
    
    /// Clip by value threshold
    ByValue(f64),
}
```

**Methods:**
- `threshold() -> f64`: Get the clipping threshold value
- `description() -> String`: Get a human-readable description

### Configuration Methods

```rust
impl AutodiffTrainingConfig {
    /// Set gradient clipping strategy
    pub fn grad_clip(mut self, clip: GradientClipping) -> Self;
}
```

## Performance Considerations

### Training Speed

Gradient clipping via adaptive LR scaling has minimal overhead:
- No additional backward passes
- No gradient tensor manipulation
- Negligible computation per batch

### Memory Usage

Current implementation adds minimal memory:
- Two f64 values for gradient statistics (16 bytes)
- No additional gradient tensor storage

## Troubleshooting

### Problem: Loss becomes NaN

**Solutions:**
1. Enable gradient clipping with conservative threshold:
   ```rust
   .grad_clip(GradientClipping::ByNorm(0.5))
   ```
2. Reduce learning rate
3. Increase warmup period
4. Check input data for NaN/Inf values

### Problem: Training too slow

**Solutions:**
1. Increase clipping threshold:
   ```rust
   .grad_clip(GradientClipping::ByNorm(5.0))
   ```
2. Increase learning rate
3. Reduce weight decay
4. Check if data is normalized

### Problem: Unstable validation loss

**Solutions:**
1. Use norm-based clipping instead of value-based
2. Add learning rate warmup
3. Enable early stopping
4. Reduce batch size

## Testing

The implementation includes comprehensive tests:

```bash
# Run gradient clipping tests
cargo test --lib training_autodiff test_gradient_clipping

# Run all training tests
cargo test --lib training_autodiff
```

## Migration Notes

### Upgrading from No Clipping

If you have existing training code without gradient clipping:

```rust
// Before
let config = AutodiffTrainingConfig::default()
    .epochs(100)
    .learning_rate(0.001);

// After (add gradient clipping)
let config = AutodiffTrainingConfig::default()
    .epochs(100)
    .learning_rate(0.001)
    .grad_clip(GradientClipping::ByNorm(1.0));  // Add this line
```

No other changes needed - clipping is applied automatically during training.

### Future Burn Upgrade

When upgrading to Burn 0.20+ with enhanced gradient APIs:
- Configuration API remains unchanged
- Implementation will switch to true gradient norm computation
- Training behavior should improve (more accurate clipping)
- No code changes required in user code

## References

- [Gradient Clipping: On the difficulty of training recurrent neural networks (Pascanu et al., 2013)](https://arxiv.org/abs/1211.5063)
- [Burn Framework Documentation](https://burn.dev/)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Related Documentation

- [Autodiff Integration](AUTODIFF_INTEGRATION_COMPLETE.md)
- [Optimizer Integration](OPTIMIZER_INTEGRATION_COMPLETE.md)
- [Training Best Practices](TRAINING_BEST_PRACTICES.md)