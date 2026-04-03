# Adaptive Thresholding Quick Reference

## Installation

```rust
use vision::adaptive::{
    AdaptiveSystem,
    RegimeDetector, RegimeConfig, RegimeAdjuster, MarketRegime,
    ThresholdCalibrator, ThresholdConfig, AdaptiveThreshold,
    CombinedCalibrator, CalibrationConfig,
};
```

## Quick Start

### 1. Complete Adaptive System (Recommended)

```rust
// Initialize
let mut system = AdaptiveSystem::new();

// In your trading loop
loop {
    // Update market data
    let regime = system.update_market(price, volatility);
    
    // Get model prediction
    let raw_confidence = model.predict(&features);
    
    // Process through adaptive system
    let (calibrated, threshold, should_trade) = 
        system.process_prediction(raw_confidence);
    
    if should_trade {
        let size = system.get_adjusted_position_size(1000.0);
        let profit = execute_trade(size);
        
        // Update with outcome
        system.update_trade(raw_confidence, profit, volatility);
    }
}
```

### 2. Regime Detection Only

```rust
let mut detector = RegimeDetector::new(RegimeConfig::default());

let regime = detector.update(price);

match regime {
    MarketRegime::BullTrending => { /* Strong uptrend */ },
    MarketRegime::BearTrending => { /* Strong downtrend */ },
    MarketRegime::RangingCalm => { /* Sideways, low vol */ },
    MarketRegime::RangingVolatile => { /* Choppy */ },
    MarketRegime::BullVolatile => { /* Uptrend + high vol */ },
    MarketRegime::BearVolatile => { /* Downtrend + high vol */ },
    MarketRegime::Unknown => { /* Insufficient data */ },
}
```

### 3. Dynamic Thresholds Only

```rust
let mut calibrator = ThresholdCalibrator::new(ThresholdConfig::default());

// Update with trades
calibrator.update(confidence, profit);

// Get current threshold
let threshold = calibrator.current_threshold();

// With volatility adjustment
calibrator.update_volatility(vol);
let vol_adjusted = calibrator.volatility_adjusted_threshold();
```

### 4. Confidence Calibration Only

```rust
let mut calibrator = CombinedCalibrator::new(CalibrationConfig::default());

// Train
calibrator.add_sample(predicted_confidence, actual_outcome);

// Calibrate
let calibrated = calibrator.calibrate(raw_prediction);
```

## Configuration Presets

### Conservative Trading

```rust
let regime_config = RegimeConfig {
    trend_window: 30,
    volatility_window: 30,
    adx_threshold: 30.0,
    volatility_threshold: 0.8,
    min_data_points: 30,
};

let threshold_config = ThresholdConfig {
    base_confidence: 0.75,
    target_win_rate: 0.60,
    learning_rate: 0.005,
    max_adjustment: 0.2,
    ..Default::default()
};
```

### Aggressive Trading

```rust
let regime_config = RegimeConfig {
    trend_window: 10,
    volatility_window: 10,
    adx_threshold: 20.0,
    volatility_threshold: 0.6,
    min_data_points: 15,
};

let threshold_config = ThresholdConfig {
    base_confidence: 0.55,
    target_win_rate: 0.50,
    learning_rate: 0.02,
    max_adjustment: 0.4,
    ..Default::default()
};
```

### High-Frequency (Fast Adaptation)

```rust
let regime_config = RegimeConfig {
    trend_window: 5,
    volatility_window: 5,
    adx_threshold: 20.0,
    volatility_threshold: 0.6,
    min_data_points: 10,
};

let threshold_config = ThresholdConfig {
    base_confidence: 0.65,
    lookback_window: 50,
    min_samples: 10,
    learning_rate: 0.05,
    ..Default::default()
};
```

## Common Patterns

### Pattern 1: Regime-Specific Strategy

```rust
let regime = system.current_regime();

match regime {
    MarketRegime::BullTrending | MarketRegime::BearTrending => {
        // Aggressive in trends
        system.threshold.set_risk_level(RiskLevel::Aggressive);
    }
    MarketRegime::RangingVolatile => {
        // Very conservative in choppy markets
        system.threshold.set_risk_level(RiskLevel::Conservative);
    }
    _ => {
        system.threshold.set_risk_level(RiskLevel::Moderate);
    }
}
```

### Pattern 2: Multi-Timeframe Confirmation

```rust
let mut hourly = AdaptiveSystem::new();
let mut daily = AdaptiveSystem::new();

// Update both
let regime_1h = hourly.update_market(price_1h, vol_1h);
let regime_1d = daily.update_market(price_1d, vol_1d);

// Trade only if both agree on trend
if regime_1h.is_trending() && regime_1d.is_trending() {
    if regime_1h.is_bullish() == regime_1d.is_bullish() {
        // Strong trend across timeframes
        trade();
    }
}
```

### Pattern 3: Confidence Filtering

```rust
let (calibrated, threshold, should_trade) = 
    system.process_prediction(raw_confidence);

// Additional safety checks
if should_trade {
    // Only trade if very high confidence
    if calibrated > 0.8 {
        execute_trade();
    }
    // Or check calibration quality
    let metrics = system.calibrator.get_metrics();
    if metrics.brier_score < 0.2 && calibrated > threshold {
        execute_trade();
    }
}
```

### Pattern 4: Volatility Scaling

```rust
let base_size = 1000.0;

// Scale by regime
let regime_size = system.get_adjusted_position_size(base_size);

// Further scale by volatility
let current_vol = calculate_volatility();
let avg_vol = 0.02; // Historical average
let vol_scalar = (avg_vol / current_vol).min(2.0).max(0.5);
let final_size = regime_size * vol_scalar;
```

## Monitoring & Diagnostics

### System Health Check

```rust
let stats = system.get_stats();

println!("Regime: {:?}", stats.regime);
println!("Threshold: {:.3}", stats.threshold_stats.current_threshold);
println!("Win Rate: {:.1}%", stats.threshold_stats.win_rate * 100.0);
println!("Brier Score: {:.4}", stats.calibration_metrics.brier_score);

// Warnings
if stats.threshold_stats.current_threshold < 0.4 {
    eprintln!("⚠️ Very low threshold - market may be difficult");
}

if stats.calibration_metrics.brier_score > 0.3 {
    eprintln!("⚠️ Poor calibration quality");
}

if stats.calibration_metrics.sample_count < 50 {
    eprintln!("⚠️ Insufficient calibration data");
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

let profit_factor = if threshold_stats.avg_losing_profit != 0.0 {
    threshold_stats.avg_winning_profit / -threshold_stats.avg_losing_profit
} else {
    f64::INFINITY
};
println!("Profit Factor: {:.2}", profit_factor);
```

## Integration Examples

### With Backtesting

```rust
use vision::backtest::BacktestSimulation;

let mut backtest = BacktestSimulation::new(config);
let mut adaptive = AdaptiveSystem::new();

for candle in historical_data {
    let regime = adaptive.update_market(candle.close, volatility);
    let signal = model.predict(&candle);
    
    let (calibrated, threshold, should_trade) = 
        adaptive.process_prediction(signal.confidence);
    
    if should_trade {
        let size = adaptive.get_adjusted_position_size(1000.0);
        backtest.execute_signal(&signal, size);
    }
}
```

### With Risk Management

```rust
use vision::risk::RiskManager;

let mut risk = RiskManager::new(risk_config);
let mut adaptive = AdaptiveSystem::new();

// Combine regime and risk-based sizing
let regime_size = adaptive.get_adjusted_position_size(1000.0);
let final_size = risk.calculate_position_size(
    regime_size,
    entry_price,
    stop_loss,
);
```

### With Live Pipeline

```rust
use vision::live::LivePipeline;

let pipeline = LivePipeline::new(config);
let mut adaptive = AdaptiveSystem::new();

loop {
    let data = pipeline.get_latest_data();
    let regime = adaptive.update_market(data.price, data.volatility);
    
    let prediction = pipeline.infer(&data);
    let (calibrated, threshold, should_trade) = 
        adaptive.process_prediction(prediction.confidence);
    
    if should_trade {
        execute_order(calibrated, regime);
    }
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Regime flips too often | Increase `trend_window` and `volatility_window` |
| Poor calibration (high Brier) | Collect more samples, increase `num_bins` |
| Threshold won't converge | Lower `learning_rate`, increase `min_samples` |
| Too many trades in volatile markets | Increase regime multipliers for volatile regimes |
| Not enough trades | Lower `base_confidence`, check calibration |
| Threshold drifts constantly | Check if `target_win_rate` is realistic |

## Best Practices

1. **Warm-up Period**: Don't trade for first 20-50 bars
2. **Monitor Calibration**: Check Brier score < 0.25
3. **Regime Transitions**: Be cautious when regime changes
4. **Data Quality**: Ensure clean price/volatility data
5. **Logging**: Track all decisions for analysis
6. **Periodic Reset**: Reset calibration if market structure changes
7. **Conservative Defaults**: Start with higher thresholds
8. **Gradual Adjustment**: Use low learning rates initially

## API Reference

### AdaptiveSystem

- `new()` → AdaptiveSystem
- `update_market(price, volatility)` → MarketRegime
- `update_trade(confidence, profit, volatility)` → void
- `process_prediction(raw_confidence)` → (f64, f64, bool)
- `get_adjusted_position_size(base_size)` → f64
- `current_regime()` → MarketRegime
- `get_stats()` → AdaptiveSystemStats
- `reset()` → void

### RegimeDetector

- `new(config)` → RegimeDetector
- `update(price)` → MarketRegime
- `current_regime()` → MarketRegime
- `reset()` → void

### ThresholdCalibrator

- `new(config)` → ThresholdCalibrator
- `update(confidence, profit)` → void
- `update_volatility(volatility)` → void
- `current_threshold()` → f64
- `volatility_adjusted_threshold()` → f64
- `statistics()` → ThresholdStats
- `reset()` → void

### CombinedCalibrator

- `new(config)` → CombinedCalibrator
- `add_sample(predicted, actual)` → void
- `calibrate(predicted)` → f64
- `set_platt_weight(weight)` → void
- `get_metrics()` → CalibrationMetrics
- `is_fitted()` → bool
- `reset()` → void

## Performance Notes

- Regime detection: O(n) where n = window size
- Threshold calibration: O(1) per update
- Platt scaling: O(m) where m = sample count
- Isotonic calibration: O(m log m) per fit
- Memory: ~O(lookback_window) per component

## Further Reading

- Full documentation: `WEEK8_DAY2_ADAPTIVE_THRESHOLDING.md`
- Example code: `examples/adaptive_thresholding.rs`
- Tests: `src/adaptive/*/tests.rs`
