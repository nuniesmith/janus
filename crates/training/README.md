# JANUS Training Infrastructure

**Phase 4: Training Pipeline & Hardware Acceleration**

This crate provides the training infrastructure for Project JANUS, enabling end-to-end neural network training with logical constraints, experience replay, and advanced optimization strategies.

## Overview

The training crate integrates with the Vision (ViViT + DiffGAF) and Logic (LTN) crates to provide:

1. **Training Loop**: End-to-end coordinator with checkpointing and callbacks
2. **Optimizers**: Adam, AdamW, SGD with configurable parameters
3. **Learning Rate Schedulers**: Warmup, cosine annealing, step decay, exponential decay
4. **Experience Replay**: Prioritized replay buffer with Sharp-Wave Ripple (SWR) sampling
5. **Gradient Management**: Gradient clipping and monitoring utilities

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Infrastructure                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Optimizer   │  │   Scheduler  │  │    Replay    │     │
│  │ (AdamW/Adam) │  │ (Warmup/Cos) │  │    Buffer    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                   ┌────────▼────────┐                       │
│                   │ Training Loop   │                       │
│                   │  (Vision+LTN)   │                       │
│                   └────────┬────────┘                       │
│                            │                                │
│                   ┌────────▼────────┐                       │
│                   │  Backprop + GPU │                       │
│                   │   RTX 3080      │                       │
│                   └─────────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Modules

### 0. Training Loop (`loop.rs`)

**NEW in Phase 4**: End-to-end training coordinator that orchestrates all components.

**Features:**
- Vision + LTN integration
- Prioritized replay buffer sampling
- Automatic checkpointing and model versioning
- Validation and early stopping
- Custom callbacks for metrics and logging
- Gradient clipping and monitoring
- Learning rate scheduling
- Wake/Sleep coordination support

**Example:**

```rust
use training::{
    TrainingLoop, TrainingConfig, OptimizerConfig,
    LRSchedulerConfig, ReplayBufferConfig, DefaultCallback
};

// Configure all components
let train_config = TrainingConfig::default()
    .device(Device::cuda_if_available(0)?);
    
let opt_config = OptimizerConfig::adamw()
    .learning_rate(1e-4)
    .weight_decay(0.01)
    .build();

let sched_config = LRSchedulerConfig::warmup_cosine()
    .warmup_steps(1000)
    .total_steps(100_000)
    .build();

let replay_config = ReplayBufferConfig::default();

// Create training loop
let mut training = TrainingLoop::new(
    train_config,
    opt_config,
    sched_config,
    replay_config,
)?;

// Add callbacks
training.add_callback(Box::new(DefaultCallback));

// Define loss functions
let task_loss_fn = |batch, var_map, device| {
    // Forward pass through vision pipeline
    let embeddings = vision.forward(&batch_states)?;
    // Compute MSE or classification loss
    Ok(mse_loss)
};

let logic_loss_fn = |batch, var_map, device| {
    // Create LTN grounding from predictions
    let mut grounding = Grounding::new();
    grounding.set("predictions", predictions);
    // Compute satisfaction loss
    ltn.satisfaction_loss(&grounding)
};

// Run training
let final_metrics = training.run(
    task_loss_fn,
    logic_loss_fn,
    Some(&validation_data),  // Optional validation set
    Some(val_task_loss_fn),
    Some(val_logic_loss_fn),
)?;
```

**Checkpointing:**

```rust
// Automatic checkpointing every N steps
let config = TrainingConfig {
    checkpoint_every: 5000,
    checkpoint_dir: PathBuf::from("checkpoints"),
    max_checkpoints: 10,  // Keep only 10 latest
    ..Default::default()
};

// Manual checkpoint
let checkpoint_path = training.save_checkpoint(&metrics)?;

// Load checkpoint
training.load_checkpoint(&checkpoint_path)?;
```

**Custom Callbacks:**

```rust
use training::TrainingCallback;

struct PrometheusCallback {
    metrics_registry: Registry,
}

impl TrainingCallback for PrometheusCallback {
    fn on_step_end(&mut self, metrics: &StepMetrics) -> Result<()> {
        // Push metrics to Prometheus
        self.metrics_registry.register_metric("loss", metrics.total_loss);
        Ok(())
    }
    
    fn on_validation_end(&mut self, metrics: &ValidationMetrics) -> Result<()> {
        // Log validation results
        println!("Validation loss: {}", metrics.val_total_loss);
        Ok(())
    }
}

training.add_callback(Box::new(PrometheusCallback { /* ... */ }));
```

### 1. Optimizer (`optimizer.rs`)

Provides configuration and wrappers for neural network optimizers.

**Supported Optimizers:**
- **AdamW** (recommended): Adam with decoupled weight decay
- **Adam**: Adaptive moment estimation
- **SGD**: Stochastic gradient descent (configuration only, use AdamW wrapper)

**Features:**
- Gradient clipping by global norm
- Learning rate adjustment
- Configurable hyperparameters (β₁, β₂, weight decay, etc.)

**Example:**

```rust
use training::{OptimizerConfig, OptimizerWrapper};
use candle_nn::VarMap;

// Create optimizer configuration
let config = OptimizerConfig::adamw()
    .learning_rate(1e-4)
    .weight_decay(0.01)
    .beta1(0.9)
    .beta2(0.999)
    .max_grad_norm(Some(1.0))
    .build();

// Build optimizer
let var_map = VarMap::new();
let mut optimizer = OptimizerWrapper::from_config(&config, var_map)?;

// Training step
optimizer.step(&grads)?;
optimizer.set_learning_rate(new_lr);
```

### 2. Learning Rate Scheduler (`scheduler.rs`)

Advanced learning rate scheduling strategies for improved training.

**Supported Schedulers:**
- **WarmupCosine** (recommended): Linear warmup followed by cosine annealing
- **CosineAnnealing**: Smooth decay using cosine function
- **StepLR**: Step-wise decay every N steps
- **LinearWarmup**: Linear increase to target LR
- **ExponentialDecay**: Exponential decay over time
- **Constant**: Fixed learning rate

**Example:**

```rust
use training::{LRScheduler, LRSchedulerConfig};

// Warmup + Cosine (recommended for transformers)
let scheduler = LRSchedulerConfig::warmup_cosine()
    .warmup_steps(1000)
    .total_steps(100_000)
    .min_lr(1e-6)
    .build();

let mut lr_sched = LRScheduler::from_config(scheduler, 1e-4);

// Training loop
for step in 0..100_000 {
    let lr = lr_sched.get_lr(step);
    optimizer.set_learning_rate(lr);
    // ... training ...
}
```

**Warmup + Cosine Formula:**

```
             ⎧  t/T_warmup * lr_max                    (t < T_warmup)
lr(t) =      ⎨
             ⎩  lr_min + 0.5(lr_max - lr_min)(1 + cos(π(t-T_warmup)/(T-T_warmup)))
```

### 3. Prioritized Replay Buffer (`replay.rs`)

Experience replay buffer with prioritized sampling for reinforcement learning and continual learning.

**Features:**
- **Prioritized Sampling**: Sample important experiences more frequently
- **Importance Sampling**: Correct for bias with IS weights
- **SWR-style Replay**: Sharp-Wave Ripple sampling for memory consolidation
- **Circular Buffer**: Fixed capacity with automatic overwriting
- **Priority Updates**: Adjust based on TD errors

**Example:**

```rust
use training::{PrioritizedReplayBuffer, ReplayBufferConfig, create_experience};

// Create buffer
let config = ReplayBufferConfig {
    capacity: 100_000,
    alpha: 0.6,      // Prioritization strength
    beta: 0.4,       // IS correction (anneals to 1.0)
    ..Default::default()
};

let mut buffer = PrioritizedReplayBuffer::new(config);

// Add experiences
let exp = create_experience(state, action, reward, next_state, done);
buffer.add(exp);

// Sample batch
let batch = buffer.sample(32)?;

// Train and update priorities based on TD errors
let td_errors = compute_td_errors(&batch.experiences);
buffer.update_priorities(&batch.indices, &td_errors);

// Apply importance sampling weights to loss
let weighted_loss = loss * batch.is_weights;
```

**SWR Sampling:**

```rust
// Sample with recency bias (mimics hippocampal replay)
let batch = buffer.sample_swr(32, 0.5)?;
```

## Integration Example

### Complete Training with TrainingLoop (Recommended)

```rust
use training::{
    TrainingLoop, TrainingConfig, OptimizerConfig,
    LRSchedulerConfig, ReplayBufferConfig, create_experience
};
use vision::{VisionPipeline, VisionPipelineConfig};
use logic::{DiffLTN, Grounding, RuleBuilder, TNormType};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

// 1. Setup device and configuration
let device = Device::cuda_if_available(0)?;

let train_config = TrainingConfig {
    num_steps: 100_000,
    batch_size: 32,
    logic_weight: 0.5,
    checkpoint_every: 5_000,
    validate_every: 1_000,
    ..Default::default()
}.device(device.clone());

let opt_config = OptimizerConfig::adamw()
    .learning_rate(1e-4)
    .weight_decay(0.01)
    .build();

let sched_config = LRSchedulerConfig::warmup_cosine()
    .warmup_steps(1000)
    .total_steps(100_000)
    .build();

let replay_config = ReplayBufferConfig::default();

// 2. Create training loop
let mut training = TrainingLoop::new(
    train_config,
    opt_config,
    sched_config,
    replay_config,
)?;

// 3. Build models using training loop's VarMap
let vb = VarBuilder::from_varmap(training.var_map(), DType::F32, &device);

let vision_config = VisionPipelineConfig::default();
let vision = VisionPipeline::from_vb(vision_config, vb.pp("vision"))?;

// 4. Define LTN rules
let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);
ltn.add_rule(
    RuleBuilder::new("confidence_action")
        .implies("high_confidence", "action_allowed")
        .weight(2.0)
        .build()
);

// 5. Populate replay buffer with experiences
for i in 0..1000 {
    let state = Tensor::randn(0f32, 1.0, (128, 5), &device)?;
    let action = Tensor::randn(0f32, 1.0, (10,), &device)?;
    let reward = compute_reward(i);
    let next_state = Tensor::randn(0f32, 1.0, (128, 5), &device)?;
    let done = i % 100 == 0;
    
    let exp = create_experience(state, action, reward, next_state, done, None)?;
    training.add_experience(exp);
}

// 6. Define loss functions
let task_loss_fn = |batch, var_map, device| {
    // Extract states and run vision pipeline
    let embeddings = vision.forward(&batch_states)?;
    let predictions = mlp.forward(&embeddings)?;
    
    // Compute task loss (e.g., MSE)
    let task_loss = mse_loss(&predictions, &targets)?;
    Ok(task_loss)
};

let logic_loss_fn = |batch, var_map, device| {
    // Create grounding from predictions
    let mut grounding = Grounding::new();
    grounding.set("high_confidence", confidence_pred);
    grounding.set("action_allowed", action_pred);
    
    // Compute LTN satisfaction loss
    ltn.satisfaction_loss(&grounding)
};

// 7. Run training with validation
let val_data = load_validation_data()?;

let final_metrics = training.run(
    task_loss_fn,
    logic_loss_fn,
    Some(&val_data),
    Some(val_task_loss_fn),
    Some(val_logic_loss_fn),
)?;

println!("Training complete! Final loss: {}", final_metrics.total_loss);
```

### Manual Training Loop (Advanced)

For fine-grained control, you can implement your own loop:

```rust
use training::{OptimizerConfig, LRSchedulerConfig, PrioritizedReplayBuffer};
use vision::VisionPipeline;
use logic::{DiffLTN, Grounding};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;

// Setup
let device = Device::cuda_if_available(0)?;
let var_map = VarMap::new();

// Vision pipeline
let vision = VisionPipeline::new(vision_config, vb.pp("vision"))?;

// Logic constraints
let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);
ltn.add_rules(trading_rules);

// Optimizer
let opt_config = OptimizerConfig::adamw()
    .learning_rate(1e-4)
    .weight_decay(0.01)
    .build();
let mut optimizer = OptimizerWrapper::from_config(&opt_config, var_map)?;

// LR Scheduler
let sched_config = LRSchedulerConfig::warmup_cosine()
    .warmup_steps(1000)
    .total_steps(100_000)
    .min_lr(1e-6)
    .build();
let lr_sched = LRScheduler::from_config(sched_config, 1e-4);

// Replay buffer
let mut buffer = PrioritizedReplayBuffer::new(
    ReplayBufferConfig::default()
);

// Training loop
for step in 0..100_000 {
    // Update learning rate
    let lr = lr_sched.get_lr(step);
    optimizer.set_learning_rate(lr);
    
    // Sample batch
    let batch = buffer.sample(32)?;
    
    // Forward pass
    let states: Vec<Tensor> = batch.experiences.iter()
        .map(|e| tensor_from_state(&e.state))
        .collect();
    let input = Tensor::stack(&states, 0)?;
    
    let embeddings = vision.forward(&input)?;
    let predictions = classifier.forward(&embeddings)?;
    
    // Compute task loss
    let task_loss = cross_entropy(&predictions, &labels)?;
    
    // Compute logical loss
    let mut grounding = Grounding::new();
    grounding.set("predictions", predictions.clone());
    let logic_loss = ltn.satisfaction_loss(&grounding)?;
    
    // Total loss
    let total_loss = (task_loss + 0.1 * logic_loss)?;
    
    // Backward pass
    let grads = total_loss.backward()?;
    
    // Optimizer step
    optimizer.step(&grads)?;
    
    // Update priorities
    let td_errors = compute_td_errors(&batch);
    buffer.update_priorities(&batch.indices, &td_errors);
    
    // Logging
    if step % 100 == 0 {
        println!("Step {}: Loss = {:.4}, LR = {:.6}", 
                 step, total_loss.to_scalar::<f32>()?, lr);
    }
}
```

## Wake/Sleep Cycle

For systems with limited VRAM (e.g., RTX 3080), coordinate training with inference:

### Wake Phase (Inference)
- Forward service runs on GPU
- Collect experiences
- Store in replay buffer

### Sleep Phase (Training)
- Forward service pauses/offloads
- Backward service loads model
- Train on GPU with batched replay
- Save checkpoints

**Implementation:**

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

struct ModelState {
    on_gpu: bool,
    training: bool,
}

let state = Arc::new(RwLock::new(ModelState {
    on_gpu: true,
    training: false,
}));

// Wake: Inference
async fn wake_phase(state: Arc<RwLock<ModelState>>) {
    let mut s = state.write().await;
    s.on_gpu = true;
    s.training = false;
    // Run inference, collect experiences
}

// Sleep: Training
async fn sleep_phase(state: Arc<RwLock<ModelState>>) {
    let mut s = state.write().await;
    s.training = true;
    // Train model, update weights
}

// Schedule
tokio::spawn(async move {
    loop {
        wake_phase(state.clone()).await;
        sleep(Duration::from_hours(16)).await;
        
        sleep_phase(state.clone()).await;
        sleep(Duration::from_hours(8)).await;
    }
});
```

## GPU Acceleration

### Enable CUDA

```toml
[dependencies]
training = { path = "../training", features = ["cuda"] }
```

```rust
use candle_core::Device;

// Use GPU if available
let device = Device::cuda_if_available(0)?;

// Or explicitly select CUDA
let device = Device::new_cuda(0)?;
```

### Mixed Precision (Future)

```rust
// Use F16 for forward pass, F32 for gradients
let embeddings = vision.forward_f16(&input)?;
let loss = compute_loss_f32(&embeddings)?;
```

## Monitoring & Checkpointing

### Training Metrics

```rust
use training::replay::ReplayBufferStats;

// Buffer statistics
let stats = buffer.stats();
println!("Buffer: {}/{}, Avg Priority: {:.2}", 
         stats.size, stats.capacity, stats.avg_priority);

// Optimizer state
println!("LR: {:.6}", optimizer.learning_rate());
```

### Checkpointing

```rust
// Save model
var_map.save("checkpoints/model_step_10000.safetensors")?;

// Load model
let loaded_map = VarMap::load("checkpoints/model_step_10000.safetensors")?;
```

## Performance Tips

1. **Batch Size**: Start with 32-64 for ViViT, adjust based on VRAM
2. **Gradient Accumulation**: For larger effective batch sizes
3. **Learning Rate**: 1e-4 for AdamW is a good starting point
4. **Warmup**: Use 5-10% of total steps for warmup
5. **Weight Decay**: 0.01 for AdamW prevents overfitting
6. **Replay Buffer**: 100k capacity for good diversity
7. **SWR Sampling**: Use during consolidation for better retention

## Examples

Run the complete Vision + LTN training example:

```bash
# CPU training
cargo run --example vision_ltn_training

# GPU training (CUDA)
cargo run --example vision_ltn_training --features cuda

# GPU training (Metal - Apple Silicon)
cargo run --example vision_ltn_training --features metal
```

## Testing

```bash
# Run all tests
cd src/janus && cargo test --package training

# Run specific module tests
cargo test --package training optimizer::tests
cargo test --package training replay::tests
cargo test --package training scheduler::tests
cargo test --package training loop::tests

# With GPU support
cargo test --package training --features cuda
```

## Features

```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
vision = ["dep:vision"]  # Optional vision pipeline integration (macOS only)
```

### Feature Descriptions

- **`cpu`** (default) - CPU-only training, works on all platforms
- **`cuda`** - Enable NVIDIA GPU acceleration via CUDA
- **`metal`** - Enable Apple GPU acceleration via Metal (macOS only)
- **`vision`** - Enable vision pipeline integration with DiffGAF + ViViT (macOS only)

### Platform Compatibility

| Feature | Linux | macOS | Windows |
|---------|-------|-------|---------|
| `cpu` | ✅ | ✅ | ✅ |
| `cuda` | ✅ | ❌ | ✅ |
| `metal` | ❌ | ✅ | ❌ |
| `vision` | ❌ | ✅ | ❌ |

**Note:** The `vision` feature requires macOS-specific dependencies (`font-kit`, `core-graphics`) and is automatically excluded on other platforms. Examples that require this feature (e.g., `vision_ltn_training`) will be skipped when building without it.

### Building with Features

```bash
# Default (CPU only, all platforms)
cargo build

# With CUDA (Linux/Windows)
cargo build --features cuda

# With vision (macOS only)
cargo build --features vision

# With vision and Metal (macOS only)
cargo build --features vision,metal

# Run vision example (macOS only)
cargo run --example vision_ltn_training --features vision
```

## References

- [Adam Optimizer](https://arxiv.org/abs/1412.6980) - Kingma & Ba, 2014
- [Decoupled Weight Decay](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter, 2017
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - Schaul et al., 2015
- [Cosine Annealing](https://arxiv.org/abs/1608.03983) - Loshchilov & Hutter, 2016
- [Sharp-Wave Ripples](https://www.nature.com/articles/nrn3962) - Buzsáki, 2015

## License

MIT License - see repository root for details.