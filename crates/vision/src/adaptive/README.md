# Adaptive Thresholding & Dynamic Calibration

This module provides adaptive mechanisms for trading decision-making that automatically adjust to changing market conditions.

## Overview

The adaptive module consists of four main components:

1. **Regime Detection** - Identifies market regimes (trending, ranging, volatile)
2. **Dynamic Thresholds** - Adjusts confidence thresholds based on performance
3. **Confidence Calibration** - Calibrates model outputs to actual probabilities
4. **Adaptive System** - Unified interface integrating all components

## Quick Start

```rust
use vision::adaptive::AdaptiveSystem;

// Initialize the adaptive system
let mut system = AdaptiveSystem::new();

// In your trading loop
loop {
    // 1. Update market data
    let regime = system.update_market(price, volatility);
    
    // 2. Get model prediction
    let raw_confidence = model.predict(&features);
    
    // 3. Process through adaptive system
    let (calibrated_confidence, adjusted_threshold, should_trade) = 
        system.process_prediction(raw_confidence);
    
    // 4. Make trading decision
    if should_trade {
        let position_size = system.get_adjusted_position_size(1000.0);
        let profit = execute_trade(position_size);
        
        // 5. Update with outcome for learning
        system.update_trade(raw_confidence, profit, volatility);
    }
}
```

## Components

### 1. Regime Detection (`regime.rs`)

Detects current market regime to adapt strategy parameters.

**Regime Types:**
- `BullTrending` - Strong uptrend
- `BearTrending` - Strong downtrend
- `RangingCalm` - Sideways with low volatility
- `RangingVolatile` - Sideways with high volatility
- `BullVolatile` - Uptrend with high volatility
- `BearVolatile` - Downtrend with high volatility
- `Unknown` - Insufficient data

**Example:**
```rust
use vision::adaptive::regime::{RegimeDetector, RegimeConfig};

let config = RegimeConfig {
    trend_window: 20,
    volatility_window: 20,
    adx_threshold: 25.0,
    volatility_threshold: 0.7,
    min_data_points: 20,
};

let mut detector = RegimeDetector::new(config);

for price in prices {
    let regime = detector.update(price);
    match regime {
        MarketRegime::BullTrending => { /* Trade aggressively long */ },
        MarketRegime::RangingVolatile => { /* Reduce size or avoid */ },
        _ => { /* Normal trading */ }
    }
}
```

### 2. Dynamic Thresholds (`threshold.rs`)

Automatically adjusts decision thresholds based on performance feedback.

**Features:**
- Performance-based learning (targets specific win rate)
- Volatility-adjusted thresholds
- Multi-level thresholds (conservative/moderate/aggressive)
- Bounded adjustments to prevent extremes

**Example:**
```rust
use vision::adaptive::threshold::{ThresholdCalibrator, ThresholdConfig};

let config = ThresholdConfig {
    base_confidence: 0.7,
    target_win_rate: 0.55,
    learning_rate: 0.01,
    max_adjustment: 0.3,
    ..Default::default()
};

let mut calibrator = ThresholdCalibrator::new(config);

// Update with trade outcomes
calibrator.update(confidence, profit);

// Get adjusted threshold
let threshold = calibrator.current_threshold();

// With volatility adjustment
calibrator.update_volatility(current_volatility);
let vol_adjusted = calibrator.volatility_adjusted_threshold();
```

### 3. Confidence Calibration (`calibration.rs`)

Transforms raw model outputs into calibrated probabilities.

**Methods:**
- **Platt Scaling** - Logistic regression calibration
- **Isotonic Regression** - Non-parametric monotonic calibration
- **Combined** - Weighted ensemble of both methods

**Example:**
```rust
use vision::adaptive::calibration::{CombinedCalibrator, CalibrationConfig};

let config = CalibrationConfig {
    num_bins: 10,
    min_samples_per_bin: 10,
    lookback_window: 1000,
    regularization: 0.01,
};

let mut calibrator = CombinedCalibrator::new(config);

// Train with historical data
for (predicted, actual) in historical_predictions {
    calibrator.add_sample(predicted, actual);
}

// Calibrate new predictions
let calibrated = calibrator.calibrate(raw_prediction);

// Check quality
let metrics = calibrator.get_metrics();
println!("Brier Score: {:.4}", metrics.brier_score); // Lower is better
println!("Log Loss: {:.4}", metrics.log_loss);       // Lower is better
```

### 4. Adaptive System (`mod.rs`)

Unified interface that integrates all components.

**Example:**
```rust
use vision::adaptive::{AdaptiveSystem, RegimeConfig, ThresholdConfig, CalibrationConfig};

// Create with custom configs
let system = AdaptiveSystem::with_configs(
    RegimeConfig::default(),
    ThresholdConfig {
        base_confidence: 0.65,
        target_win_rate: 0.55,
        ..Default::default()
    },
    CalibrationConfig::default(),
);

// Or use defaults
let mut system = AdaptiveSystem::new();

// Get comprehensive statistics
let stats = system.get_stats();
println!("Regime: {:?}", stats.regime);
println!("Win Rate: {:.1}%", stats.threshold_stats.win_rate * 100.0);
println!("Brier Score: {:.4}", stats.calibration_metrics.brier_score);
```

## Configuration Presets

### Conservative Trading
```rust
RegimeConfig {
    trend_window: 30,
    volatility_window: 30,
    adx_threshold: 30.0,
    volatility_threshold: 0.8,
    min_data_points: 30,
}

ThresholdConfig {
    base_confidence: 0.75,
    target_win_rate: 0.60,
    learning_rate: 0.005,
    max_adjustment: 0.2,
    ..Default::default()
}
```

### Aggressive Trading
```rust
RegimeConfig {
    trend_window: 10,
    volatility_window: 10,
    adx_threshold: 20.0,
    volatility_threshold: 0.6,
    min_data_points: 15,
}

ThresholdConfig {
    base_confidence: 0.55,
    target_win_rate: 0.50,
    learning_rate: 0.02,
    max_adjustment: 0.4,
    ..Default::default()
}
```

## Common Patterns

### Pattern 1: Regime-Specific Strategy
```rust
let regime = system.current_regime();

match regime {
    MarketRegime::BullTrending | MarketRegime::BearTrending => {
        system.threshold.set_risk_level(RiskLevel::Aggressive);
    }
    MarketRegime::RangingVolatile => {
        system.threshold.set_risk_level(RiskLevel::Conservative);
    }
    _ => {
        system.threshold.set_risk_level(RiskLevel::Moderate);
    }
}
```

### Pattern 2: Multi-Timeframe Confirmation
```rust
let mut systems = vec![
    AdaptiveSystem::new(), // 1-hour
    AdaptiveSystem::new(), // 4-hour  
    AdaptiveSystem::new(), // Daily
];

// Trade only if all timeframes agree
let all_trending = systems.iter()
    .all(|s| s.current_regime().is_trending());

if all_trending {
    execute_trade();
}
```

### Pattern 3: Calibration Quality Check
```rust
let (calibrated, threshold, should_trade) = system.process_prediction(raw);

if should_trade {
    let metrics = system.calibrator.get_metrics();
    
    // Only trade if calibration is good
    if metrics.brier_score < 0.25 && calibrated > 0.75 {
        execute_trade();
    }
}
```

## Monitoring

### Health Checks
```rust
let stats = system.get_stats();

// Check calibration quality
if stats.calibration_metrics.brier_score > 0.3 {
    eprintln!("⚠️ Poor calibration quality");
}

// Check sufficient data
if stats.calibration_metrics.sample_count < 50 {
    eprintln!("⚠️ Insufficient calibration data");
}

// Check threshold bounds
if stats.threshold_stats.current_threshold < 0.4 {
    eprintln!("⚠️ Very low threshold - market may be difficult");
}
```

### Performance Tracking
```rust
let threshold_stats = system.threshold.get_stats();

println!("Trades: {}", threshold_stats.sample_count);
println!("Win Rate: {:.1}%", threshold_stats.win_rate * 100.0);
println!("Avg Profit: ${:.2}", threshold_stats.avg_profit);
println!("Avg Win: ${:.2}", threshold_stats.avg_winning_profit);
println!("Avg Loss: ${:.2}", threshold_stats.avg_losing_profit);
```

## Integration

### With Backtesting
```rust
use vision::backtest::BacktestSimulation;

let mut backtest = BacktestSimulation::new(config);
let mut adaptive = AdaptiveSystem::new();

for candle in data {
    let regime = adaptive.update_market(candle.close, volatility);
    let (cal, thresh, trade) = adaptive.process_prediction(signal.confidence);
    
    if trade {
        let size = adaptive.get_adjusted_position_size(1000.0);
        backtest.execute_signal(&signal, size);
    }
}
```

### With Risk Management
```rust
use vision::risk::RiskManager;

let regime_size = adaptive.get_adjusted_position_size(base_size);
let final_size = risk_manager.calculate_position_size(
    regime_size,
    entry_price,
    stop_loss,
);
```

### With Live Pipeline
```rust
use vision::live::LivePipeline;

let mut pipeline = LivePipeline::new(config);
let mut adaptive = AdaptiveSystem::new();

loop {
    let data = pipeline.get_latest_data();
    let regime = adaptive.update_market(data.price, data.volatility);
    let prediction = pipeline.infer(&data);
    let (cal, thresh, trade) = adaptive.process_prediction(prediction.confidence);
    
    if trade {
        execute_order();
        adaptive.update_trade(prediction.confidence, realized_pnl, data.volatility);
    }
}
```

## Performance

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Regime Detection | O(n) | O(window) |
| Threshold Update | O(1) | O(lookback) |
| Platt Scaling | O(m) | O(m) |
| Isotonic Calibration | O(m log m) | O(bins) |
| Complete System | O(n + m) | O(lookback) |

Where n = window size (20-30), m = calibration samples (100-1000)

## Best Practices

1. **Warm-up Period**: Don't trade for first 20-50 bars
2. **Monitor Calibration**: Keep Brier score < 0.25
3. **Regime Transitions**: Be cautious when regime changes
4. **Data Quality**: Ensure clean price/volatility data
5. **Logging**: Track all decisions for post-analysis
6. **Periodic Reset**: Reset if market structure changes significantly
7. **Conservative Defaults**: Start with higher thresholds
8. **Gradual Learning**: Use low learning rates initially

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Regime flips too often | Increase `trend_window` and `volatility_window` |
| Poor calibration | Collect more samples, increase `num_bins` |
| Threshold won't converge | Lower `learning_rate`, increase `min_samples` |
| Too many trades in volatile markets | Increase multipliers for volatile regimes |
| Not enough trades | Lower `base_confidence`, check calibration |
| Threshold drifts | Check if `target_win_rate` is realistic |

## Examples

See `examples/adaptive_thresholding.rs` for comprehensive examples including:
- Regime detection scenarios
- Threshold calibration with different win rates
- Confidence calibration
- Complete adaptive system
- Realistic multi-phase trading scenario

Run with:
```bash
cargo run --example adaptive_thresholding --release
```

## Testing

Run tests:
```bash
# All adaptive tests
cargo test --lib adaptive

# Specific module
cargo test --lib adaptive::regime::tests
cargo test --lib adaptive::threshold::tests
cargo test --lib adaptive::calibration::tests
```

## Documentation

- **Full Documentation**: `docs/WEEK8_DAY2_ADAPTIVE_THRESHOLDING.md`
- **Quick Reference**: `docs/ADAPTIVE_QUICK_REFERENCE.md`
- **API Docs**: Run `cargo doc --open`

## License

Part of the JANUS vision crate.