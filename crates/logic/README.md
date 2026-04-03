# Janus Logic Crate

**Neuro-Symbolic Reasoning Layer for Project JANUS**

This crate provides the logic and reasoning capabilities for Project JANUS, implementing both traditional (inference-time) and differentiable (training-time) Logic Tensor Networks (LTN).

## Overview

The logic crate bridges neural networks with symbolic reasoning, enabling:

1. **Logical Constraints as Losses**: Convert logical rules into differentiable loss functions
2. **Grounding**: Map neural network outputs to logical variables
3. **Rule Satisfaction**: Measure and optimize how well neural outputs satisfy logical rules
4. **Inference-Time Gating**: Use fuzzy logic to gate neural network outputs against rules

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Neuro-Symbolic System                    │
├─────────────────────────────────────────────────────────────┤
│  Neural Network (ViViT)  →  Embeddings  →  Predictions      │
│           ↓                                    ↓             │
│      Grounding          ←─────────────────────┘             │
│           ↓                                                  │
│      Predicates         (learnable/threshold)               │
│           ↓                                                  │
│      Diff T-Norms       (AND, OR, IMPLIES, NOT)             │
│           ↓                                                  │
│      Logical Rules      (formulas, quantifiers)             │
│           ↓                                                  │
│   Satisfaction Loss     →  Backpropagation  →  Training     │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### Differentiable LTN (Candle-based)

| Module | Description |
|--------|-------------|
| `diff_tnorm` | Differentiable fuzzy T-norms (Łukasiewicz, Product, Gödel) |
| `predicates` | Learnable and threshold-based logical predicates |
| `ltn` | Full LTN framework with rule composition and loss functions |

### Traditional Logic (ndarray-based)

| Module | Description |
|--------|-------------|
| `tnorm` | Traditional fuzzy logic operations for inference |
| `constraints` | Rule-based constraint checking |
| `signal` | Signal validation and filtering |
| `position` | Position-related logic |
| `risk` | Risk assessment logic |
| `risk_engine` | Full risk evaluation engine |

## Quick Start

### Training with Logical Constraints

```rust
use logic::{DiffLTN, RuleBuilder, Grounding, TNormType};
use candle_core::{Device, Tensor};

// Create LTN with Łukasiewicz t-norm (best gradient flow)
let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

// Add logical rules
ltn.add_rule(
    RuleBuilder::new("high_conf_action")
        .implies("confidence_high", "action_allowed")
        .weight(1.0)
        .build()
);

ltn.add_rule(
    RuleBuilder::new("risk_bounds")
        .requires("position_valid")
        .weight(2.0)
        .build()
);

// Create grounding from neural network outputs
let mut grounding = Grounding::new();
grounding.set("confidence_high", confidence_tensor);
grounding.set("action_allowed", action_tensor);
grounding.set("position_valid", position_tensor);

// Compute differentiable satisfaction loss
let logic_loss = ltn.satisfaction_loss(&grounding)?;

// Combine with task loss for training
let total_loss = (task_loss + logic_weight * logic_loss)?;
// Backpropagate...
```

### Using Predicates

```rust
use logic::predicates::{ThresholdPredicate, TradingPredicates, LearnablePredicate};
use candle_core::{Device, Tensor};

let device = Device::Cpu;

// Threshold predicates (no parameters)
let high_conf = ThresholdPredicate::greater_than(0.8);
let in_range = ThresholdPredicate::in_range(0.0, 100.0);

// Pre-built trading predicates
let pos_valid = TradingPredicates::position_size_valid(10000.0);
let risk_ok = TradingPredicates::acceptable_risk(0.05);

// Learnable predicate (trainable parameters)
let config = LearnablePredicateConfig::simple(256, "signal_quality");
let (learnable_pred, var_map) = LearnablePredicate::new_random(config, &device)?;

// Evaluate
let values = Tensor::new(&[0.3f32, 0.7, 0.9], &device)?;
let truth_values = high_conf.evaluate(&values)?;
```

### T-Norm Operations

```rust
use logic::{lukasiewicz, product, godel, DiffTNorm};
use candle_core::{Device, Tensor};

let device = Device::Cpu;
let a = Tensor::new(&[0.8f32, 0.6, 0.9], &device)?;
let b = Tensor::new(&[0.7f32, 0.5, 0.3], &device)?;

// Choose t-norm family
let tnorm = lukasiewicz(); // Best for LTN training
// let tnorm = product();  // Probabilistic interpretation
// let tnorm = godel(0.1); // Classical min/max (smoothed)

// Fuzzy operations
let and_result = tnorm.and(&a, &b)?;      // Conjunction
let or_result = tnorm.or(&a, &b)?;        // Disjunction
let implies = tnorm.implies(&a, &b)?;     // Implication
let not_a = tnorm.not(&a)?;               // Negation

// Quantifiers
let forall = tnorm.forall(&a, None)?;     // Universal (all must be true)
let exists = tnorm.exists(&a, None)?;     // Existential (at least one true)
```

### Building Complex Formulas

```rust
use logic::{Formula, DiffLTN, Grounding};

// (high_confidence ∧ low_risk) → action_allowed
let formula = Formula::var("high_confidence")
    .and(Formula::var("low_risk"))
    .implies(Formula::var("action_allowed"));

// ¬(buy ∧ sell) - mutual exclusion
let mutual_exclusion = Formula::var("buy")
    .and(Formula::var("sell"))
    .not();

// ∀x: position_valid(x)
let all_valid = Formula::var("position_valid").forall(None);

// Complex: (A ∧ B) → (C ∨ D)
let complex = Formula::var("A")
    .and(Formula::var("B"))
    .implies(Formula::var("C").or(Formula::var("D")));
```

## T-Norm Families

### Łukasiewicz (Recommended for Training)

- **AND**: `max(0, a + b - 1)`
- **OR**: `min(1, a + b)`
- **IMPLIES**: `min(1, 1 - a + b)`
- **ForAll**: Mean (smooth gradient)

Best choice for LTN training due to excellent gradient flow.

### Product (Probabilistic)

- **AND**: `a * b`
- **OR**: `a + b - a*b`
- **IMPLIES**: `min(1, b / (a + ε))`
- **ForAll**: Geometric mean

Natural probabilistic interpretation, smooth everywhere.

### Gödel (Classical, Smoothed)

- **AND**: `soft_min(a, b)`
- **OR**: `soft_max(a, b)`
- **IMPLIES**: Smooth approximation

Classical min/max logic with soft approximations for differentiability.
Use temperature parameter to control sharpness (lower = sharper).

## Pre-built Rule Sets

```rust
use logic::RuleSets;

// Trading rules for risk management
let trading_rules = RuleSets::trading_rules();

// Consistency rules for multi-output networks
let consistency_rules = RuleSets::consistency_rules();

// Add to LTN
let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);
ltn.add_rules(trading_rules);
ltn.add_rules(consistency_rules);
```

## Satisfaction Aggregation

Configure how multiple rule satisfactions are combined:

```rust
use logic::{DiffLTN, DiffLTNConfig, SatisfactionAggregation};

let config = DiffLTNConfig {
    aggregation: SatisfactionAggregation::Mean,    // Weighted average (default)
    // aggregation: SatisfactionAggregation::Product, // Geometric mean (stricter)
    // aggregation: SatisfactionAggregation::Min,     // Soft minimum (strictest)
    normalize_weights: true,
    ..Default::default()
};

let ltn = DiffLTN::with_config(config);
```

## Monitoring & Debugging

```rust
// Get detailed satisfaction report
let report = ltn.satisfaction_details(&grounding)?;

println!("Overall satisfaction: {:.2}", report.overall_satisfaction);
println!("Overall loss: {:.2}", report.overall_loss);
println!("Active rules: {}", report.num_active_rules);

for rule in &report.rule_satisfactions {
    println!("  {}: {:.2} (weight: {:.1})", 
             rule.name, rule.satisfaction, rule.weight);
}
```

## Integration with Vision Pipeline

```rust
use vision::VisionPipeline;
use logic::{DiffLTN, Grounding, ThresholdPredicate};

// Vision pipeline produces embeddings
let embeddings = vision_pipeline.forward(&input)?;

// Classification head produces predictions
let predictions = classifier.forward(&embeddings)?;
let confidence = predictions.softmax(1)?;

// Create grounding
let mut grounding = Grounding::new();

// Apply predicates to predictions
let high_conf_pred = ThresholdPredicate::greater_than(0.8);
grounding.set("confidence_high", high_conf_pred.evaluate(&confidence.max(1)?.0)?);
grounding.set("action_taken", predictions.argmax(1)?.to_dtype(DType::F32)?);

// Compute logical loss
let logic_loss = ltn.satisfaction_loss(&grounding)?;

// Total loss for training
let task_loss = cross_entropy(&predictions, &labels)?;
let total_loss = (task_loss + 0.1 * logic_loss)?;
```

## Features

```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
```

## Testing

```bash
# Run all logic tests
cd src/janus && cargo test --package logic

# Run with verbose output
cd src/janus && cargo test --package logic -- --nocapture

# Run specific test
cd src/janus && cargo test --package logic diff_tnorm::tests::test_lukasiewicz_and
```

## Performance Considerations

1. **Batch Operations**: All operations support batched tensors for GPU efficiency
2. **Gradient Flow**: Łukasiewicz t-norm provides the smoothest gradients
3. **Numerical Stability**: Gödel t-norm uses log-sum-exp trick internally
4. **Memory**: Grounding holds tensor references, consider cloning for long-lived structures

## References

- [Logic Tensor Networks](https://arxiv.org/abs/2012.13635) - Badreddine et al.
- [Fuzzy Sets and Fuzzy Logic](https://link.springer.com/book/10.1007/978-94-015-8449-4) - Klir & Yuan
- [Neural-Symbolic Integration](https://arxiv.org/abs/1905.06088) - Garcez et al.

## License

MIT License - see repository root for details.