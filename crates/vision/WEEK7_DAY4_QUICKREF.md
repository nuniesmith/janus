# Week 7 Day 4: Live Pipeline - Quick Reference

**Status**: ✅ COMPLETED  
**Focus**: Real-time feature computation, caching, latency optimization, inference-only mode

---

## Quick Start

```rust
use vision::{LivePipeline, LivePipelineConfig, MarketData};

// Create and initialize pipeline
let mut pipeline = LivePipeline::new(LivePipelineConfig::low_latency());
pipeline.initialize()?;
pipeline.warmup()?;

// Process live data
let data = MarketData::new(timestamp, open, high, low, close, volume);
if let Some(pred) = pipeline.process_tick(data)? {
    if pred.meets_confidence(0.8) {
        println!("Signal: {:.4}, Conf: {:.4}", pred.signal, pred.confidence);
    }
}
```

---

## Configuration Presets

### Low Latency (<50ms)
```rust
let config = LivePipelineConfig::low_latency();
// Window: 30, Target: 50ms, Profiling: ON
```

### High Accuracy
```rust
let config = LivePipelineConfig::high_accuracy();
// Window: 120, Target: 200ms, Larger context
```

### Backtest Replay
```rust
let config = LivePipelineConfig::backtest_replay();
// No caching, High throughput
```

---

## Core Components

### 1. Streaming Buffer

```rust
use vision::{StreamingFeatureBuffer, MarketData};

let mut buffer = StreamingFeatureBuffer::new(60);
buffer.update(market_data)?;

if buffer.is_ready() {
    let closes = buffer.get_closes();
    let returns = buffer.get_returns();
    let stats = buffer.stats();
}
```

### 2. Circular Buffer

```rust
use vision::CircularBuffer;

let mut buffer = CircularBuffer::new(100);
buffer.push(item);
let is_full = buffer.is_full();
let vec = buffer.to_vec();
```

### 3. Incremental GAF Computer

```rust
use vision::IncrementalGAFComputer;

let mut computer = IncrementalGAFComputer::new(60);
if let Some(gaf) = computer.update(market_data)? {
    // GAF matrix ready for inference
}
```

### 4. Feature Cache

```rust
use vision::{FeatureCache, FeatureCacheKey, FeatureType};

let mut cache = FeatureCache::new();
let key = FeatureCacheKey::new(
    "BTC".to_string(),
    "1m".to_string(),
    FeatureType::GAF,
    timestamp,
);

cache.insert(key.clone(), feature);
if let Some(cached) = cache.get(&key) {
    // Use cached feature
}
```

### 5. Inference Engine

```rust
use vision::{InferenceEngine, InferenceConfig};

let mut engine = InferenceEngine::new(InferenceConfig::low_latency());
engine.warmup()?;

let prediction = engine.predict_single(&gaf_matrix)?;
println!("Signal: {:.4}", prediction.signal);
```

### 6. Latency Tracking

```rust
use vision::{LatencyTracker, LatencyBudget};

// Track latencies
let mut tracker = LatencyTracker::new(1000);
tracker.record(latency_us);
println!("P95: {} μs", tracker.p95_us());

// Monitor SLA
let mut budget = LatencyBudget::from_ms("inference".to_string(), 100.0);
if !budget.check(latency_us) {
    println!("SLA violation!");
}
```

---

## Performance Monitoring

### Get Pipeline Stats

```rust
// Latency budget
let budget = pipeline.latency_budget();
println!("Compliance: {:.2}%", budget.compliance_rate() * 100.0);

// Inference stats
let stats = pipeline.inference_stats()?;
println!("Avg latency: {:.2}ms", stats.avg_latency_ms());

// Cache stats
if let Some(cache_stats) = pipeline.cache_stats() {
    println!("Hit rate: {:.2}%", cache_stats.overall_hit_rate() * 100.0);
}

// Full report
pipeline.performance_report();
```

### Profiling

```rust
let config = LivePipelineConfig {
    enable_profiling: true,
    ..LivePipelineConfig::default()
};

let mut pipeline = LivePipeline::new(config);
// ... process data ...

if let Some(profiler) = pipeline.profiler() {
    profiler.report();
    let (slowest, latency) = profiler.slowest_stage().unwrap();
    println!("Bottleneck: {} ({:.2}μs)", slowest, latency);
}
```

---

## Benchmarking

```rust
use vision::InferenceEngine;

let mut engine = InferenceEngine::new(InferenceConfig::low_latency());
let results = engine.benchmark(1000, 64)?;

results.print_summary();
if results.meets_sla(100.0) {
    println!("✓ Meets SLA");
}
```

---

## Thread-Safe Pipeline

```rust
use vision::SharedLivePipeline;
use std::thread;

let shared = SharedLivePipeline::new(LivePipelineConfig::default());
shared.initialize()?;

let handles: Vec<_> = (0..4).map(|i| {
    let shared_clone = shared.clone();
    thread::spawn(move || {
        shared_clone.process_market_data(get_data(i))
    })
}).collect();

for handle in handles {
    handle.join().unwrap()?;
}
```

---

## Common Patterns

### Real-Time Trading

```rust
loop {
    let data = exchange.get_latest_tick()?;
    
    if let Some(pred) = pipeline.process_tick(data)? {
        if pred.meets_confidence(0.8) && !pred.is_neutral(0.1) {
            if pred.is_bullish() {
                execute_buy_order()?;
            } else {
                execute_sell_order()?;
            }
        }
    }
}
```

### Multi-Timeframe Analysis

```rust
use vision::MultiTimeframeBuffer;

let mut mtf = MultiTimeframeBuffer::new();
mtf.add_timeframe("1m".to_string(), 60);
mtf.add_timeframe("5m".to_string(), 300);
mtf.add_timeframe("15m".to_string(), 900);

mtf.update_all(&market_data)?;

if mtf.all_ready() {
    let buffer_1m = mtf.get_buffer("1m").unwrap();
    let buffer_5m = mtf.get_buffer("5m").unwrap();
}
```

### Running Statistics

```rust
use vision::RunningStats;

let mut stats = RunningStats::new();
for value in values {
    stats.update(value);
}

println!("Mean: {:.4}", stats.mean());
println!("StdDev: {:.4}", stats.std_dev());
println!("Min: {:.4}, Max: {:.4}", stats.min(), stats.max());
```

---

## Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| P99 Latency | <100ms | ~3.5ms |
| Throughput | >100/s | ~350/s |
| Cache Hit Rate | >80% | ~87% |
| Memory | <500MB | ~50MB |

---

## API Summary

### LivePipeline
- `new(config)` - Create pipeline
- `initialize()` - Load models
- `warmup()` - Pre-warm caches
- `process_tick(data)` - Process new data
- `is_ready()` - Check if buffer full
- `performance_report()` - Print stats

### MarketData
- `new(ts, o, h, l, c, v)` - Create data point
- `ohlc()` - Get OHLC array

### Prediction
- `signal` - Direction (-1 to 1)
- `confidence` - Certainty (0 to 1)
- `is_bullish()` - Signal > 0
- `is_bearish()` - Signal < 0
- `meets_confidence(t)` - Conf >= threshold

### LatencyTracker
- `record(us)` - Record latency
- `mean_us()` - Average
- `p50_us()`, `p95_us()`, `p99_us()` - Percentiles
- `summary()` - Get all stats

---

## Examples

Run the comprehensive example:
```bash
cargo run --example live_pipeline --release
```

Run tests:
```bash
cargo test live::
```

---

## Troubleshooting

### High Latency
1. Reduce window size
2. Enable caching
3. Check profiler for bottlenecks
4. Increase warmup iterations

### Low Cache Hit Rate
1. Increase cache capacity
2. Adjust TTL
3. Review access patterns

### Memory Issues
1. Reduce cache capacity
2. Lower window size
3. Disable profiling in production

---

## Test Results

**Total Tests**: 298 (55 new for live pipeline)  
**Status**: ✅ All passing  
**Coverage**: Streaming, Caching, Inference, Latency tracking, Integration

---

## Files Added

- `src/live/mod.rs` - Main pipeline orchestrator
- `src/live/streaming.rs` - Streaming computation
- `src/live/cache.rs` - LRU cache
- `src/live/inference.rs` - Inference engine
- `src/live/latency.rs` - Latency tracking
- `examples/live_pipeline.rs` - Comprehensive example
- `WEEK7_DAY4_LIVE_PIPELINE.md` - Full documentation

---

**Next**: Week 7 Day 5 - Production Deployment & Monitoring