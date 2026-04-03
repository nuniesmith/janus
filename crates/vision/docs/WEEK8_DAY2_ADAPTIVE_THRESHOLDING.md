# Week 8 Day 2: Adaptive Thresholding & Dynamic Calibration

## Overview

Day 2 of Week 8 implements adaptive thresholding and dynamic calibration systems that automatically adjust trading decision thresholds based on market conditions, performance feedback, and regime detection. This enables the trading system to adapt to changing market dynamics without manual intervention.

## Architecture

```
Market Data → Regime Detection → Regime-Adjusted Thresholds
                    ↓
Model Predictions → Confidence Calibration → Calibrated Predictions
                    ↓
        Performance Feedback → Threshold Learning
                    ↓
            Trading Decisions
```

## Components

### 1. Market Regime Detection (`regime.rs`)

Identifies different market regimes to adapt strategy parameters:

#### Regime Types
- **BullTrending**: Strong uptrend with directional movement
- **BearTrending**: Strong downtrend with directional movement
- **RangingCalm**: Sideways market with low volatility
- **RangingVolatile**: Sideways market with high volatility
- **BullVolatile**: Uptrend with high volatility
- **BearVolatile**: Downtrend with high volatility
- **Unknown**: Insufficient data or transitional state

#### Key Features

**RegimeDetector**
```rust
use vision::adaptive::regime::{RegimeDetector, RegimeConfig};

let config = RegimeConfig {
    trend_window: 20,           // Lookback for trend detection
    volatility_window: 20,       // Lookback for volatility
    adx_threshold: 25.0,         // ADX threshold for trending
    volatility_threshold: 0.7,   // Volatility percentile threshold
    min_data_points: 20,         // Min data before detection
};

let mut detector = RegimeDetector::new(config);

// Update with new price
let regime = detector.update(100.5);
println!("Current regime: {:?}", regime);
```

**RegimeAdjuster**
```rust
use vision::adaptive::regime::{RegimeAdjuster, MarketRegime};

let adjuster = RegimeAdjuster::default();

// Adjust confidence threshold based on regime
let base_threshold = 0.7;
let adjusted = adjuster.adjust_confidence_threshold(
    base_threshold,
    MarketRegime::BullTrending
);
// Returns 0.63 (lower in strong trends)

// Adjust position size based on regime
let base_size = 1000.0;
let adjusted_size = adjuster.adjust_position_size(
    base_size,
    MarketRegime::RangingVolatile
);
// Returns 500.0 (smaller in volatile ranging)
```

#### Detection Methods

1. **Trend Detection**: Uses simplified ADX (Average Directional Index)
   - Calculates directional movement (up vs down)
   - Measures trend strength
   - Threshold typically 25 (values above indicate trending)

2. **Volatility Detection**: Compares current to historical volatility
   - Rolling window standard deviation of returns
   - Percentile-based classification
   - Identifies volatile vs calm periods

3. **Trend Direction**: Linear regression slope
   - Positive slope → bullish
   - Negative slope → bearish
   - Near-zero → ranging

### 2. Dynamic Threshold Calibration (`threshold.rs`)

Automatically adjusts decision thresholds based on performance feedback:

#### ThresholdCalibrator

```rust
use vision::adaptive::threshold::{ThresholdCalibrator, ThresholdConfig};

let config = ThresholdConfig {
    base_confidence: 0.7,        // Starting threshold
    lookback_window: 100,        // Performance history window
    min_samples: 20,             // Min trades before adjusting
    max_adjustment: 0.3,         // Max deviation from base
    target_win_rate: 0.55,       // Desired win rate
    learning_rate: 0.01,         // Adjustment speed
};

let mut calibrator = ThresholdCalibrator::new(config);

// Update with trade outcomes
calibrator.update(0.8, 100.0);  // confidence=0.8, profit=100
calibrator.update(0.6, -50.0);  // confidence=0.6, profit=-50

// Get adjusted threshold
let threshold = calibrator.current_threshold();

// Get statistics
let stats = calibrator.statistics();
println!("Win rate: {:.1}%", stats.win_rate * 100.0);
println!("Avg profit: ${:.2}", stats.avg_profit);
```

#### Adjustment Logic

The calibrator uses gradient descent to optimize thresholds:

1. **Calculate Win Rate**: Percentage of profitable trades
2. **Calculate Error**: `error = target_win_rate - current_win_rate`
3. **Adjust Threshold**: 
   - If win rate too low → **lower threshold** (allow more trades)
   - If win rate too high → **raise threshold** (be more selective)
4. **Apply Learning Rate**: Small incremental adjustments
5. **Clamp to Bounds**: Prevent extreme values

#### Volatility-Adjusted Thresholds

```rust
// Add volatility data
calibrator.update_volatility(0.02); // Normal volatility
calibrator.update_volatility(0.05); // High volatility

// Get volatility-adjusted threshold
let vol_adjusted = calibrator.volatility_adjusted_threshold();
// Higher volatility → higher threshold (more conservative)
```

#### Multi-Level Thresholds

```rust
use vision::adaptive::threshold::{MultiLevelThreshold, RiskLevel};

let mut thresholds = MultiLevelThreshold::new(0.7);
// Creates: conservative=0.85, moderate=0.70, aggressive=0.55

let threshold = thresholds.get_threshold(RiskLevel::Conservative);
// Returns 0.85

// Adjust all levels for volatility
thresholds.adjust_for_volatility(1.5); // 50% higher volatility
```

#### AdaptiveThreshold

Combines calibration, multi-level thresholds, and percentile tracking:

```rust
use vision::adaptive::threshold::{AdaptiveThreshold, ThresholdConfig, RiskLevel};

let mut adaptive = AdaptiveThreshold::new(ThresholdConfig::default());

// Set risk level
adaptive.set_risk_level(RiskLevel::Moderate);

// Update with outcomes
adaptive.update(0.75, 50.0, 0.02); // confidence, profit, volatility

// Check if should trade
if adaptive.should_trade(0.80) {
    println!("Confidence exceeds threshold - trade!");
}
```

### 3. Confidence Calibration (`calibration.rs`)

Transforms raw model outputs into calibrated probabilities:

#### Why Calibration?

Neural networks often produce **overconfident** or **underconfident** predictions:
- A model saying 0.9 confidence might only be correct 70% of the time
- Calibration aligns predicted confidence with actual success rate

#### Platt Scaling (Logistic Calibration)

Fits a logistic regression on model outputs:

```rust
use vision::adaptive::calibration::{PlattScaling, CalibrationConfig};

let config = CalibrationConfig {
    num_bins: 10,
    min_samples_per_bin: 10,
    lookback_window: 1000,
    regularization: 0.01,
};

let mut platt = PlattScaling::new(config);

// Add calibration samples (predicted_confidence, actual_outcome)
for _ in 0..100 {
    platt.add_sample(0.9, true);   // High confidence, success
    platt.add_sample(0.3, false);  // Low confidence, failure
}

// Calibrate new prediction
let calibrated = platt.calibrate(0.75);
println!("Calibrated probability: {:.3}", calibrated);
```

**How it works:**
- Fits parameters `A` and `B` using gradient descent
- Calibrated probability: `sigmoid(A * predicted + B)`
- Regularization prevents overfitting

#### Isotonic Regression (Non-parametric)

Monotonic step function calibration:

```rust
use vision::adaptive::calibration::IsotonicCalibration;

let mut isotonic = IsotonicCalibration::new(config);

// Add samples
for i in 0..100 {
    let confidence = i as f64 / 100.0;
    let outcome = confidence > 0.5; // Simple relationship
    isotonic.add_sample(confidence, outcome);
}

// Calibrate
let calibrated = isotonic.calibrate(0.65);

// Get calibration curve
let curve = isotonic.get_calibration_curve();
for (pred, actual) in curve {
    println!("Predicted: {:.2} → Actual: {:.2}", pred, actual);
}
```

**How it works:**
- Divides predictions into bins
- Calculates actual success rate per bin
- Enforces monotonicity (higher confidence → higher probability)
- Uses linear interpolation between bins

#### Combined Calibrator

Weighted combination of Platt scaling and isotonic regression:

```rust
use vision::adaptive::calibration::CombinedCalibrator;

let mut calibrator = CombinedCalibrator::new(config);

// Add samples
calibrator.add_sample(0.8, true);
calibrator.add_sample(0.4, false);

// Set weighting (0.5 = equal weight)
calibrator.set_platt_weight(0.6); // 60% Platt, 40% isotonic

// Calibrate
let calibrated = calibrator.calibrate(0.7);

// Get quality metrics
let metrics = calibrator.get_metrics();
println!("Brier score: {:.4}", metrics.brier_score); // Lower is better
println!("Log loss: {:.4}", metrics.log_loss);       // Lower is better
```

### 4. Complete Adaptive System (`mod.rs`)

Integrates all components into a unified system:

```rust
use vision::adaptive::{AdaptiveSystem, RegimeConfig, ThresholdConfig, CalibrationConfig};

// Create with default configs
let mut system = AdaptiveSystem::new();

// Or with custom configs
let system = AdaptiveSystem::with_configs(
    RegimeConfig::default(),
    ThresholdConfig::default(),
    CalibrationConfig::default(),
);

// Trading loop
loop {
    // 1. Update market data
    let regime = system.update_market(price, volatility);
    
    // 2. Get model prediction
    let raw_confidence = model.predict(&features);
    
    // 3. Process through adaptive system
    let (calibrated, threshold, should_trade) = 
        system.process_prediction(raw_confidence);
    
    if should_trade {
        // 4. Get regime-adjusted position size
        let base_size = 1000.0;
        let adjusted_size = system.get_adjusted_position_size(base_size);
        
        // 5. Execute trade
        let profit = execute_trade(adjusted_size);
        
        // 6. Update with outcome
        system.update_trade(raw_confidence, profit, volatility);
    }
}

// Get statistics
let stats = system.get_stats();
println!("Regime: {:?}", stats.regime);
println!("Win rate: {:.1}%", stats.threshold_stats.win_rate * 100.0);
println!("Brier score: {:.4}", stats.calibration_metrics.brier_score);
```

## Configuration Guidelines

### Regime Detection

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| `trend_window` | 30 | 20 | 10 |
| `volatility_window` | 30 | 20 | 10 |
| `adx_threshold` | 30 | 25 | 20 |
| `volatility_threshold` | 0.8 | 0.7 | 0.6 |
| `min_data_points` | 30 | 20 | 15 |

### Threshold Calibration

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| `base_confidence` | 0.75 | 0.65 | 0.55 |
| `target_win_rate` | 0.60 | 0.55 | 0.50 |
| `learning_rate` | 0.005 | 0.01 | 0.02 |
| `max_adjustment` | 0.2 | 0.3 | 0.4 |

### Calibration

| Parameter | Small Dataset | Medium Dataset | Large Dataset |
|-----------|---------------|----------------|---------------|
| `num_bins` | 5 | 10 | 20 |
| `min_samples_per_bin` | 5 | 10 | 20 |
| `lookback_window` | 500 | 1000 | 2000 |
| `regularization` | 0.05 | 0.01 | 0.001 |

## Performance Metrics

### Calibration Quality

**Brier Score**: Measures calibration accuracy
- Formula: `mean((predicted - actual)^2)`
- Range: [0, 1], lower is better
- Perfect calibration: 0.0
- Random guessing: ~0.25

**Log Loss (Cross-Entropy)**
- Formula: `-mean(y*log(p) + (1-y)*log(1-p))`
- Range: [0, ∞], lower is better
- Perfect calibration: 0.0
- Penalizes confident wrong predictions heavily

### Threshold Performance

**Win Rate**: Percentage of profitable trades
- Track over rolling window
- Compare to target win rate
- Should stabilize near target

**Average Profit**: Mean profit per trade
- Higher win rate trades should be more profitable
- Balance win rate vs profit per trade

**Threshold Stability**: Variance in threshold over time
- Low variance = stable market conditions
- High variance = changing conditions or insufficient data

## Usage Patterns

### Pattern 1: Conservative Trading

```rust
let mut system = AdaptiveSystem::new();

// Set conservative parameters
let regime_adjuster = RegimeAdjuster::default();

system.threshold.set_risk_level(RiskLevel::Conservative);

// Only trade in favorable regimes
if system.current_regime().is_trending() {
    let (calibrated, threshold, should_trade) = 
        system.process_prediction(confidence);
    
    if should_trade && calibrated > 0.8 {
        // Very high confidence required
        trade(size);
    }
}
```

### Pattern 2: Regime-Specific Strategies

```rust
match system.current_regime() {
    MarketRegime::BullTrending => {
        // Aggressive in uptrends
        system.threshold.set_risk_level(RiskLevel::Aggressive);
    }
    MarketRegime::RangingVolatile => {
        // Very conservative in choppy markets
        system.threshold.set_risk_level(RiskLevel::Conservative);
    }
    _ => {
        // Moderate otherwise
        system.threshold.set_risk_level(RiskLevel::Moderate);
    }
}
```

### Pattern 3: Multi-Timeframe Analysis

```rust
let mut systems = vec![
    AdaptiveSystem::new(), // 1-hour
    AdaptiveSystem::new(), // 4-hour
    AdaptiveSystem::new(), // Daily
];

// Update all timeframes
for (i, system) in systems.iter_mut().enumerate() {
    let regime = system.update_market(prices[i], vols[i]);
    
    // Trade only if all timeframes agree
    if regimes.iter().all(|r| r.is_trending()) {
        // Strong trend across all timeframes
        let (cal, thresh, trade) = system.process_prediction(conf);
        if trade { execute(); }
    }
}
```

## Testing

The module includes comprehensive tests:

```bash
# Run adaptive tests
cargo test --lib adaptive

# Run specific test suites
cargo test --lib adaptive::regime::tests
cargo test --lib adaptive::threshold::tests
cargo test --lib adaptive::calibration::tests

# Run example
cargo run --example adaptive_thresholding --release
```

### Test Coverage

- ✅ Regime classification logic
- ✅ Regime detector with synthetic trends/ranges
- ✅ Threshold calibration convergence
- ✅ Volatility adjustment
- ✅ Platt scaling calibration
- ✅ Isotonic regression monotonicity
- ✅ Combined calibrator weighting
- ✅ Calibration metrics calculation
- ✅ Complete adaptive system integration
- ✅ Reset and edge cases

## Integration with Existing Components

### With Backtesting

```rust
use vision::backtest::BacktestSimulation;
use vision::adaptive::AdaptiveSystem;

let mut backtest = BacktestSimulation::new(config);
let mut adaptive = AdaptiveSystem::new();

for candle in historical_data {
    // Update adaptive system
    let regime = adaptive.update_market(candle.close, volatility);
    
    // Get model signal
    let signal = model.predict(&candle);
    
    // Process through adaptive system
    let (calibrated, threshold, should_trade) = 
        adaptive.process_prediction(signal.confidence);
    
    if should_trade {
        let adjusted_size = adaptive.get_adjusted_position_size(1000.0);
        backtest.execute_signal(&signal, adjusted_size);
    }
}
```

### With Risk Management

```rust
use vision::risk::RiskManager;
use vision::adaptive::AdaptiveSystem;

let mut risk_manager = RiskManager::new(risk_config);
let mut adaptive = AdaptiveSystem::new();

// Combine regime-based and risk-based sizing
let regime = adaptive.current_regime();
let regime_size = adaptive.get_adjusted_position_size(1000.0);

let final_size = risk_manager.calculate_position_size(
    regime_size,
    entry_price,
    stop_loss,
);
```

### With Live Pipeline

```rust
use vision::live::LivePipeline;
use vision::adaptive::AdaptiveSystem;

let pipeline = LivePipeline::new(config);
let mut adaptive = AdaptiveSystem::new();

loop {
    let market_data = pipeline.get_latest_data();
    
    // Update regime
    let regime = adaptive.update_market(
        market_data.price,
        market_data.volatility,
    );
    
    // Get prediction
    let prediction = pipeline.infer(&market_data);
    
    // Adaptive decision
    let (calibrated, threshold, should_trade) = 
        adaptive.process_prediction(prediction.confidence);
    
    if should_trade {
        execute_trade(calibrated, regime);
        
        // Later: update with outcome
        adaptive.update_trade(
            prediction.confidence,
            realized_profit,
            market_data.volatility,
        );
    }
}
```

## Best Practices

### 1. Warm-up Period

Allow sufficient data before trusting adaptive decisions:

```rust
let mut adaptive = AdaptiveSystem::new();
let mut warmup_count = 0;
const WARMUP_PERIOD: usize = 50;

loop {
    adaptive.update_market(price, vol);
    
    if warmup_count < WARMUP_PERIOD {
        warmup_count += 1;
        continue; // Don't trade during warmup
    }
    
    // Now safe to use adaptive decisions
    let (cal, thresh, trade) = adaptive.process_prediction(conf);
}
```

### 2. Monitor Calibration Quality

```rust
let metrics = adaptive.calibrator.get_metrics();

if metrics.brier_score > 0.3 {
    eprintln!("Warning: Poor calibration quality");
    // Consider recalibrating or using higher thresholds
}

if metrics.sample_count < 100 {
    eprintln!("Warning: Insufficient calibration samples");
    // Use conservative defaults
}
```

### 3. Regime Transition Handling

```rust
let mut prev_regime = MarketRegime::Unknown;

loop {
    let regime = adaptive.update_market(price, vol);
    
    if regime != prev_regime && regime != MarketRegime::Unknown {
        println!("Regime changed: {:?} → {:?}", prev_regime, regime);
        
        // Optional: flatten positions during transitions
        close_all_positions();
        
        // Wait for confirmation
        thread::sleep(Duration::from_secs(60));
    }
    
    prev_regime = regime;
}
```

### 4. Logging and Monitoring

```rust
use tracing::{info, warn};

let stats = adaptive.get_stats();

info!(
    regime = ?stats.regime,
    threshold = stats.threshold_stats.current_threshold,
    win_rate = stats.threshold_stats.win_rate,
    brier_score = stats.calibration_metrics.brier_score,
    "Adaptive system status"
);

if stats.threshold_stats.current_threshold < 0.5 {
    warn!("Threshold very low - market may be difficult");
}
```

## Common Issues and Solutions

### Issue: Regime Flipping

**Problem**: Regime changes too frequently

**Solution**:
- Increase `trend_window` and `volatility_window`
- Add hysteresis (require regime to persist N periods)
- Use longer timeframes for regime detection

### Issue: Poor Calibration

**Problem**: High Brier score or log loss

**Solutions**:
- Collect more calibration samples
- Increase `num_bins` or `min_samples_per_bin`
- Check for data quality issues
- Use Platt scaling only (set `platt_weight = 1.0`)

### Issue: Threshold Not Converging

**Problem**: Threshold oscillates or drifts

**Solutions**:
- Reduce `learning_rate`
- Increase `min_samples` before adjusting
- Check `target_win_rate` is realistic
- Ensure sufficient trade diversity

### Issue: Overtrading in Volatile Markets

**Problem**: Too many trades during high volatility

**Solutions**:
```rust
let adjuster = RegimeAdjuster::default()
    .with_confidence_multipliers(RegimeMultipliers {
        ranging_volatile: 1.8,  // Much higher threshold
        bull_volatile: 1.3,
        bear_volatile: 1.3,
        ..Default::default()
    });
```

## Roadmap

### Completed ✅
- Market regime detection (trend, volatility, direction)
- Dynamic threshold calibration with learning
- Platt scaling confidence calibration
- Isotonic regression calibration
- Combined calibrator with weighting
- Complete adaptive system integration
- Comprehensive testing and examples

### Future Enhancements 🚀
- Hidden Markov Models for regime detection
- Bayesian threshold updates
- Temperature scaling for calibration
- Multi-model ensemble calibration
- Regime transition prediction
- Automated parameter tuning
- Real-time calibration metrics dashboard

## References

- **ADX**: Wilder, J. W. (1978). New Concepts in Technical Trading Systems
- **Platt Scaling**: Platt, J. (1999). Probabilistic Outputs for Support Vector Machines
- **Isotonic Regression**: Zadrozny & Elkan (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates
- **Calibration**: Guo et al. (2017). On Calibration of Modern Neural Networks

## Summary

Week 8 Day 2 delivers a comprehensive adaptive system that:

1. **Detects Market Regimes**: Automatically identifies trending, ranging, and volatile conditions
2. **Adjusts Thresholds**: Dynamically calibrates decision thresholds based on performance
3. **Calibrates Confidence**: Transforms model outputs into accurate probabilities
4. **Adapts Position Sizing**: Adjusts risk based on market regime
5. **Learns from Feedback**: Continuously improves from trading outcomes

This system enables the JANUS trading platform to automatically adapt to changing market conditions, improving robustness and reducing the need for manual parameter tuning.

**Next**: Week 8 Day 3 will implement Advanced Order Execution with TWAP/VWAP algorithms and smart order routing.