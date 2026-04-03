//! Live Pipeline Example
//!
//! Demonstrates real-time market data processing and prediction using the live pipeline.
//!
//! This example shows:
//! - Setting up a live pipeline with different configurations
//! - Processing streaming market data
//! - Generating predictions in real-time
//! - Performance monitoring and latency tracking
//! - Cache statistics and optimization
//!
//! Run with:
//! ```bash
//! cargo run --example live_pipeline --release
//! ```

use vision::{LivePipeline, LivePipelineConfig, MarketData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Live Pipeline Example ===\n");

    // Example 1: Basic live pipeline
    println!("Example 1: Basic Live Pipeline");
    println!("------------------------------");
    basic_pipeline()?;
    println!();

    // Example 2: Low-latency pipeline
    println!("Example 2: Low-Latency Pipeline");
    println!("--------------------------------");
    low_latency_pipeline()?;
    println!();

    // Example 3: High-accuracy pipeline
    println!("Example 3: High-Accuracy Pipeline");
    println!("----------------------------------");
    high_accuracy_pipeline()?;
    println!();

    // Example 4: Simulated live trading
    println!("Example 4: Simulated Live Trading");
    println!("----------------------------------");
    simulated_live_trading()?;
    println!();

    // Example 5: Performance benchmarking
    println!("Example 5: Performance Benchmarking");
    println!("------------------------------------");
    benchmark_pipeline()?;
    println!();

    Ok(())
}

/// Basic live pipeline example
fn basic_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = LivePipeline::default();

    // Initialize and warm up
    pipeline.initialize()?;
    pipeline.warmup()?;

    println!("Pipeline initialized and warmed up");
    println!("Window size: {}", 60);
    println!("Processing 10 ticks...\n");

    // Simulate market data
    for i in 0..10 {
        let data = MarketData::new(
            i as i64,
            100.0 + i as f64 * 0.5,
            101.0 + i as f64 * 0.5,
            99.0 + i as f64 * 0.5,
            100.25 + i as f64 * 0.5,
            1000.0 + i as f64 * 100.0,
        );

        if let Some(prediction) = pipeline.process_tick(data)? {
            println!(
                "Tick {}: Signal={:.4}, Confidence={:.4}, Latency={} μs",
                i, prediction.signal, prediction.confidence, prediction.metadata.latency_us
            );

            if prediction.meets_confidence(0.7) {
                if prediction.is_bullish() {
                    println!("  → 📈 BULLISH signal (high confidence)");
                } else if prediction.is_bearish() {
                    println!("  → 📉 BEARISH signal (high confidence)");
                }
            }
        }
    }

    println!("\nTotal predictions: {}", pipeline.prediction_count());

    Ok(())
}

/// Low-latency pipeline optimized for speed
fn low_latency_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let config = LivePipelineConfig::low_latency();
    let mut pipeline = LivePipeline::new(config);

    pipeline.initialize()?;
    pipeline.warmup()?;

    println!("Low-latency pipeline configured");
    println!("Target: <50ms end-to-end latency");
    println!("Window size: 30\n");

    // Process ticks
    for i in 0..20 {
        let data = MarketData::new(
            i as i64,
            100.0 + (i as f64 * 0.1).sin() * 2.0,
            102.0 + (i as f64 * 0.1).sin() * 2.0,
            98.0 + (i as f64 * 0.1).sin() * 2.0,
            100.5 + (i as f64 * 0.1).sin() * 2.0,
            1000.0,
        );

        if let Some(prediction) = pipeline.process_tick(data)? {
            let latency_ms = prediction.metadata.latency_us as f64 / 1000.0;
            let status = if latency_ms < 50.0 { "✓" } else { "✗" };
            println!(
                "Tick {}: Latency={:.2}ms {}, Signal={:.4}",
                i, latency_ms, status, prediction.signal
            );
        }
    }

    println!();
    let budget = pipeline.latency_budget();
    println!("Latency Budget:");
    println!("  Target: {:.2}ms", budget.budget_us() as f64 / 1000.0);
    println!(
        "  Violations: {}/{}",
        budget.violations(),
        budget.total_checks()
    );
    println!("  Compliance: {:.2}%", budget.compliance_rate() * 100.0);

    Ok(())
}

/// High-accuracy pipeline with larger window
fn high_accuracy_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let config = LivePipelineConfig::high_accuracy();
    let mut pipeline = LivePipeline::new(config);

    pipeline.initialize()?;
    pipeline.warmup()?;

    println!("High-accuracy pipeline configured");
    println!("Window size: 120 (larger context)");
    println!("Processing trend data...\n");

    // Simulate trending market
    let mut high_conf_count = 0;
    for i in 0..30 {
        let trend = i as f64 * 0.2;
        let noise = (i as f64 * 0.5).sin() * 0.5;
        let price = 100.0 + trend + noise;

        let data = MarketData::new(
            i as i64,
            price - 0.5,
            price + 1.0,
            price - 1.0,
            price,
            1000.0,
        );

        if let Some(prediction) = pipeline.process_tick(data)? {
            if prediction.meets_confidence(0.8) {
                high_conf_count += 1;
                println!(
                    "Tick {}: HIGH CONFIDENCE - Signal={:.4}, Conf={:.4}",
                    i, prediction.signal, prediction.confidence
                );
            }
        }
    }

    println!("\nHigh confidence signals: {}", high_conf_count);
    println!("Total predictions: {}", pipeline.prediction_count());

    Ok(())
}

/// Simulated live trading scenario
fn simulated_live_trading() -> Result<(), Box<dyn std::error::Error>> {
    let config = LivePipelineConfig {
        window_size: 60,
        enable_cache: true,
        cache_capacity: 200,
        enable_profiling: true,
        ..LivePipelineConfig::default()
    };

    let mut pipeline = LivePipeline::new(config);
    pipeline.initialize()?;
    pipeline.warmup()?;

    println!("Simulating live trading session");
    println!("Duration: 100 ticks\n");

    let mut signals_generated = 0;
    let mut bullish_signals = 0;
    let mut bearish_signals = 0;
    let confidence_threshold = 0.75;

    // Simulate realistic price movement
    let mut price = 100.0;
    for i in 0..100 {
        // Random walk with slight upward bias
        let change = ((i * 7) % 11) as f64 / 20.0 - 0.2;
        price += change;
        price = price.max(90.0).min(110.0);

        let data = MarketData::new(
            i as i64,
            price - 0.1,
            price + 0.5,
            price - 0.5,
            price,
            1000.0 + i as f64 * 10.0,
        );

        if let Some(prediction) = pipeline.process_tick(data)? {
            if prediction.meets_confidence(confidence_threshold) {
                signals_generated += 1;

                if prediction.is_bullish() {
                    bullish_signals += 1;
                    if signals_generated <= 5 {
                        println!(
                            "Tick {}: 📈 BUY signal - Price={:.2}, Conf={:.2}",
                            i, price, prediction.confidence
                        );
                    }
                } else if prediction.is_bearish() {
                    bearish_signals += 1;
                    if signals_generated <= 5 {
                        println!(
                            "Tick {}: 📉 SELL signal - Price={:.2}, Conf={:.2}",
                            i, price, prediction.confidence
                        );
                    }
                }
            }
        }
    }

    println!("\n=== Trading Session Summary ===");
    println!("Total ticks processed: 100");
    println!(
        "Signals generated (conf >= {:.0}%): {}",
        confidence_threshold * 100.0,
        signals_generated
    );
    println!("  Bullish: {}", bullish_signals);
    println!("  Bearish: {}", bearish_signals);
    println!("Signal rate: {:.1}%", signals_generated as f64);

    // Performance report
    println!();
    pipeline.performance_report();

    Ok(())
}

/// Benchmark pipeline performance
fn benchmark_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing different pipeline configurations\n");

    // Benchmark 1: Default config
    println!("1. Default Configuration:");
    let latencies1 = run_benchmark(LivePipelineConfig::default(), 50)?;
    print_benchmark_stats("Default", &latencies1);
    println!();

    // Benchmark 2: Low-latency config
    println!("2. Low-Latency Configuration:");
    let latencies2 = run_benchmark(LivePipelineConfig::low_latency(), 50)?;
    print_benchmark_stats("Low-Latency", &latencies2);
    println!();

    // Benchmark 3: High-accuracy config
    println!("3. High-Accuracy Configuration:");
    let latencies3 = run_benchmark(LivePipelineConfig::high_accuracy(), 50)?;
    print_benchmark_stats("High-Accuracy", &latencies3);
    println!();

    // Comparison
    println!("=== Configuration Comparison ===");
    println!("Default:       {:.2}ms median latency", median(&latencies1));
    println!(
        "Low-Latency:   {:.2}ms median latency ({:.1}x faster)",
        median(&latencies2),
        median(&latencies1) / median(&latencies2)
    );
    println!("High-Accuracy: {:.2}ms median latency", median(&latencies3));

    Ok(())
}

/// Run a benchmark with a given configuration
fn run_benchmark(
    config: LivePipelineConfig,
    iterations: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut pipeline = LivePipeline::new(config);
    pipeline.initialize()?;
    pipeline.warmup()?;

    let mut latencies = Vec::new();

    for i in 0..iterations {
        let data = MarketData::new(i as i64, 100.0 + i as f64 * 0.1, 101.0, 99.0, 100.5, 1000.0);

        if let Some(prediction) = pipeline.process_tick(data)? {
            latencies.push(prediction.metadata.latency_us as f64 / 1000.0);
        }
    }

    Ok(latencies)
}

/// Print benchmark statistics
fn print_benchmark_stats(name: &str, latencies: &[f64]) {
    if latencies.is_empty() {
        println!("No latency data");
        return;
    }

    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let min = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let med = median(latencies);

    println!("  Mean: {:.2}ms", mean);
    println!("  Median: {:.2}ms", med);
    println!("  Min: {:.2}ms", min);
    println!("  Max: {:.2}ms", max);
    println!("  Throughput: {:.0} predictions/sec", 1000.0 / mean);
}

/// Calculate median of a vector
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}
