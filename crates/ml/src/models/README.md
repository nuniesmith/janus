# JANUS ML Models

Neural network model implementations for the JANUS trading system.

## Overview

This module provides production-ready ML models for financial time series prediction and signal classification:

- **LSTM Predictor**: Time series forecasting for price movements
- **MLP Classifier**: Multi-class signal classification (buy/sell/hold)
- **Model Trait**: Unified interface for all model types

## Quick Start

### LSTM Price Predictor

```rust
use janus_ml::models::{LstmConfig, LstmPredictor, Model};
use janus_ml::backend::BackendDevice;

// Configure LSTM
let config = LstmConfig::new(50, 64, 1)  // input_size, hidden_size, output_size
    .with_num_layers(2)
    .with_dropout(0.2);

// Create model
let device = BackendDevice::cpu();
let model = LstmPredictor::new(config, device);

// Forward pass
let predictions = model.forward(input_tensor)?;

// Save/load
model.save("model.bin")?;
let loaded = LstmPredictor::load("model.bin", device)?;
```

### MLP Signal Classifier

```rust
use janus_ml::models::{MlpConfig, MlpClassifier, Model};

// Configure MLP
let config = MlpConfig::new(50, vec![128, 64], 3)  // input, hidden layers, num_classes
    .with_dropout(0.3)
    .with_batch_norm(true);

let model = MlpClassifier::new(config, BackendDevice::cpu());

// Get class probabilities
let probs = model.forward_with_softmax(input)?;

// Or get predicted classes
let predictions = model.predict(input)?;
```

## Architecture

### LSTM Predictor

```
Input [batch, seq_len, features]
  ↓
LSTM Layer 1 (hidden_size)
  ↓
Dropout
  ↓
LSTM Layer N
  ↓
Extract last timestep
  ↓
Linear projection
  ↓
Output [batch, output_size]
```

**Use Cases:**
- Price prediction
- Return forecasting
- Volatility prediction
- Trend strength estimation

**Configuration:**
- `input_size`: Number of input features (e.g., 50)
- `hidden_size`: LSTM hidden dimension (e.g., 64, 128)
- `output_size`: Prediction dimension (typically 1)
- `num_layers`: Stack depth (default: 2)
- `dropout`: Regularization (default: 0.2)
- `bidirectional`: Process sequence both directions (default: false)

### MLP Classifier

```
Input [batch, seq_len, features]
  ↓
Extract last timestep
  ↓
Linear → BatchNorm → ReLU → Dropout
  ↓ (repeat for each hidden layer)
Linear → BatchNorm → ReLU → Dropout
  ↓
Output Linear (logits)
  ↓
Softmax (optional)
  ↓
Class Probabilities [batch, num_classes]
```

**Use Cases:**
- Trading signal classification (buy/sell/hold)
- Regime detection (bull/bear/neutral)
- Entry/exit signals
- Risk level classification

**Configuration:**
- `input_size`: Number of input features
- `hidden_sizes`: Hidden layer dimensions (e.g., `vec![128, 64]`)
- `output_size`: Number of classes (e.g., 3 for buy/sell/hold)
- `dropout`: Regularization (default: 0.3)
- `batch_norm`: Enable batch normalization (default: true)
- `activation`: Activation function (default: ReLU)

## Model Trait

All models implement the `Model` trait:

```rust
pub trait Model: Sized {
    type Config: Clone + Serialize + for<'de> Deserialize<'de>;
    
    fn new(config: Self::Config, device: BackendDevice) -> Self;
    fn forward(&self, input: Tensor<CpuBackend, 3>) -> Result<Tensor<CpuBackend, 2>>;
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;
    fn load<P: AsRef<Path>>(path: P, device: BackendDevice) -> Result<Self>;
    fn metadata(&self) -> &ModelMetadata;
    fn config(&self) -> &Self::Config;
    fn device(&self) -> &BackendDevice;
}
```

**Benefits:**
- Consistent API across model types
- Easy model swapping
- Standardized persistence
- Metadata tracking

## Input/Output Shapes

### LSTM
- **Input**: `[batch_size, sequence_length, input_size]`
- **Output**: `[batch_size, output_size]`

Example: Predict next price from 20 timesteps of 50 features
- Input: `[32, 20, 50]` (batch=32)
- Output: `[32, 1]` (single prediction per sample)

### MLP
- **Input**: `[batch_size, sequence_length, input_size]`
- **Output**: `[batch_size, num_classes]`

Example: Classify signal from 10 timesteps of 50 features into 3 classes
- Input: `[32, 10, 50]`
- Output: `[32, 3]` (probabilities for buy/sell/hold)

## Model Metadata

Every model tracks:
- **Name**: Model identifier
- **Version**: Semantic versioning
- **Type**: "lstm", "mlp", etc.
- **Trained At**: Timestamp
- **Input/Output Sizes**: Dimensions
- **Backend**: "cpu" or "gpu"
- **Metrics**: Training/validation metrics
- **Hyperparameters**: Full configuration

Access via `model.metadata()`.

## Persistence

Models can be saved and loaded:

```rust
// Save
model.save("trained_model.bin")?;

// Load
let model = LstmPredictor::load("trained_model.bin", device)?;
```

**Format:**
- Bincode v2 serialization
- Version checking
- Metadata + config + weights
- Backward compatible

**Note:** Current implementation saves config and metadata. Full weight serialization 
is pending (returns freshly initialized weights on load). This will be completed 
during training infrastructure implementation.

## Examples

### Multi-layer LSTM for Complex Patterns

```rust
let config = LstmConfig::new(100, 128, 1)
    .with_num_layers(3)        // Deep network
    .with_dropout(0.3)         // Higher dropout for regularization
    .with_bidirectional(false); // Forward-only (faster inference)

let model = LstmPredictor::new(config, device);
```

### Lightweight MLP for Fast Inference

```rust
let config = MlpConfig::new(30, vec![64], 3)  // Single hidden layer
    .with_dropout(0.2)
    .with_batch_norm(false);  // Disable for speed

let model = MlpClassifier::new(config, device);
```

### Ensemble Prediction

```rust
// Load multiple models
let lstm1 = LstmPredictor::load("model1.bin", device)?;
let lstm2 = LstmPredictor::load("model2.bin", device)?;

// Get predictions
let pred1 = lstm1.forward(input.clone())?;
let pred2 = lstm2.forward(input)?;

// Combine (simple average)
let ensemble = (pred1 + pred2) / 2.0;
```

## Testing

Run model tests:

```bash
cargo test --package janus-ml --lib models
```

**Test Coverage:**
- Configuration builders
- Model creation
- Forward pass validation
- Shape checking
- Save/load roundtrip
- Metadata tracking
- Model trait conformance

## Performance Considerations

### LSTM
- **Memory**: O(layers * batch * seq_len * hidden_size)
- **Compute**: O(layers * seq_len * hidden_size²)
- **Recommendation**: Use hidden_size 64-128 for most tasks

### MLP
- **Memory**: O(batch * max(hidden_sizes))
- **Compute**: O(sum of layer transitions)
- **Recommendation**: 2-3 hidden layers, sizes decreasing (e.g., 128→64→32)

### Backend Selection

```rust
// CPU (default)
let device = BackendDevice::cpu();

// GPU (if available and feature enabled)
#[cfg(feature = "gpu")]
let device = BackendDevice::gpu()?;

// Auto-detect
let device = BackendDevice::auto();
```

## Known Limitations

1. **Bidirectional LSTM**: Dimension handling needs investigation (test ignored)
2. **Weight Serialization**: Currently placeholder (full implementation pending)
3. **Activation Functions**: MLP uses hardcoded ReLU (dispatch not implemented)

See `WEEK4_DAY3_COMPLETE.md` for detailed status and roadmap.

## Integration with JANUS

Models are designed to work with:
- **Feature Extraction**: `janus_ml::features`
- **Data Quality**: `janus_data_quality` pipeline
- **Training Loop**: (upcoming) `janus_ml::training`
- **Inference Engine**: (upcoming) `janus_ml::inference`

Complete pipeline:
```
Market Data → Data Quality → Feature Extraction → Model → Prediction
```

## Contributing

When adding new models:

1. Implement the `Model` trait
2. Create `YourModelConfig` with builder pattern
3. Add comprehensive tests
4. Document architecture and use cases
5. Update this README

## License

MIT - See main JANUS LICENSE file

## References

- Burn ML Framework: https://github.com/tracel-ai/burn
- JANUS Architecture: `../../docs/ARCHITECTURE.md`
- Week 4 Progress: `../../docs/WEEK4_DAY3_COMPLETE.md`
