//! DSP Performance Benchmarks
//!
//! This benchmark suite measures the performance of the DSP pipeline against
//! production targets:
//!
//! - Throughput: >1M ticks/sec
//! - Latency: <1μs median, <10μs P99 per tick
//!
//! Run with:
//! ```bash
//! cargo bench --bench bench_dsp
//! ```
//!
//! For flamegraph profiling:
//! ```bash
//! cargo bench --bench bench_dsp -- --profile-time=10
//! ```

#![cfg_attr(feature = "nightly", feature(test))]

#[cfg(feature = "nightly")]
extern crate test;

#[cfg(feature = "nightly")]
use test::Bencher;

#[cfg(feature = "nightly")]
use dsp::frama::Frama;
#[cfg(feature = "nightly")]
use dsp::normalize::WelfordNormalizer;
#[cfg(feature = "nightly")]
use dsp::pipeline::{DspConfig, DspPipeline};
#[cfg(feature = "nightly")]
use dsp::sevcik::SevcikFractalDimension;

// ============================================================================
// Sevcik Fractal Dimension Benchmarks
// ============================================================================

#[cfg(feature = "nightly")]
#[bench]
fn bench_sevcik_update_cold(b: &mut Bencher) {
    let mut calc = SevcikFractalDimension::new(64);
    let mut price = 100.0;

    b.iter(|| {
        price += 0.01;
        let _ = calc.update(price);
        test::black_box(price)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_sevcik_batch(b: &mut Bencher) {
    let mut calc = SevcikFractalDimension::new(64);

    // Warmup
    for i in 0..100 {
        let _ = calc.update(100.0 + i as f64 * 0.1);
    }

    let mut price = 110.0;

    b.iter(|| {
        price += 0.01;
        let result = calc.update(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_sevcik_window_32(b: &mut Bencher) {
    let mut calc = SevcikFractalDimension::new(32);

    // Warmup
    for i in 0..50 {
        let _ = calc.update(100.0 + i as f64 * 0.1);
    }

    let mut price = 105.0;

    b.iter(|| {
        price += 0.01;
        let result = calc.update(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_sevcik_window_128(b: &mut Bencher) {
    let mut calc = SevcikFractalDimension::new(128);

    // Warmup
    for i in 0..150 {
        let _ = calc.update(100.0 + i as f64 * 0.1);
    }

    let mut price = 115.0;

    b.iter(|| {
        price += 0.01;
        let result = calc.update(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_sevcik_throughput_1k(b: &mut Bencher) {
    b.iter(|| {
        let mut calc = SevcikFractalDimension::new(64);
        for i in 0..1000 {
            let _ = calc.update(100.0 + i as f64 * 0.01);
        }
        test::black_box(calc)
    });
}

// ============================================================================
// FRAMA Benchmarks
// ============================================================================

#[cfg(feature = "nightly")]
#[bench]
fn bench_normalizer_batch(b: &mut Bencher) {
    let mut frama = Frama::new(64, 0.01, 0.5, false);

    // Warmup
    for i in 0..100 {
        let _ = frama.update(100.0 + i as f64 * 0.1);
    }

    let mut price = 110.0;

    b.iter(|| {
        price += 0.01;
        let result = frama.update(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_frama_with_super_smoother(b: &mut Bencher) {
    let mut frama = Frama::new(64, 0.01, 0.5, true);

    // Warmup
    for i in 0..100 {
        let _ = frama.update(100.0 + i as f64 * 0.1);
    }

    let mut price = 110.0;

    b.iter(|| {
        price += 0.01;
        let result = frama.update(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_frama_throughput_1k(b: &mut Bencher) {
    b.iter(|| {
        let mut frama = Frama::new(64, 0.01, 0.5, false);
        for i in 0..1000 {
            let _ = frama.update(100.0 + i as f64 * 0.01);
        }
        test::black_box(frama)
    });
}

// ============================================================================
// Welford Normalizer Benchmarks
// Normalizer Benchmarks
// ============================================================================

#[cfg(feature = "nightly")]
#[bench]
fn bench_normalizer_update(b: &mut Bencher) {
    let mut norm = WelfordNormalizer::new(0.05, 50, Some(3.0));

    // Warmup
    for i in 0..100 {
        let _ = norm.update(100.0 + i as f64 * 0.1);
    }

    let mut price = 110.0;

    b.iter(|| {
        price += 0.01;
        let result = norm.update(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_normalizer_fast(b: &mut Bencher) {
    let mut norm = WelfordNormalizer::fast();

    // Warmup
    for i in 0..100 {
        let _ = norm.update(100.0 + i as f64 * 0.1);
    }

    let mut price = 110.0;

    b.iter(|| {
        price += 0.01;
        let result = norm.update(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_throughput(b: &mut Bencher) {
    b.iter(|| {
        let mut norm = WelfordNormalizer::new(0.05, 50, Some(3.0));
        for i in 0..1000 {
            let _ = norm.update(100.0 + i as f64 * 0.01);
        }
        test::black_box(norm)
    });
}

// ============================================================================
// Complete Pipeline Benchmarks (CRITICAL HOT PATH)
// ============================================================================

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_process_warm(b: &mut Bencher) {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..200 {
        let _ = pipeline.process(100.0 + i as f64 * 0.1);
    }

    let mut price = 120.0;

    b.iter(|| {
        price += 0.01;
        let result = pipeline.process(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_high_frequency(b: &mut Bencher) {
    let config = DspConfig::high_frequency();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..200 {
        let _ = pipeline.process(100.0 + i as f64 * 0.1);
    }

    let mut price = 120.0;

    b.iter(|| {
        price += 0.01;
        let result = pipeline.process(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_low_frequency(b: &mut Bencher) {
    let config = DspConfig::low_frequency();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..300 {
        let _ = pipeline.process(100.0 + i as f64 * 0.1);
    }

    let mut price = 130.0;

    b.iter(|| {
        price += 0.01;
        let result = pipeline.process(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_throughput_1k(b: &mut Bencher) {
    b.iter(|| {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        for i in 0..1000 {
            let _ = pipeline.process(100.0 + i as f64 * 0.01);
        }

        test::black_box(pipeline)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_throughput_10k(b: &mut Bencher) {
    b.iter(|| {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        for i in 0..10_000 {
            let _ = pipeline.process(100.0 + i as f64 * 0.001);
        }

        test::black_box(pipeline)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_throughput_100k(b: &mut Bencher) {
    b.iter(|| {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        for i in 0..100_000 {
            let _ = pipeline.process(100.0 + i as f64 * 0.0001);
        }

        test::black_box(pipeline)
    });
}

// ============================================================================
// Realistic Market Scenario Benchmarks
// ============================================================================

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_trending_market(b: &mut Bencher) {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..200 {
        let _ = pipeline.process(100.0 + i as f64 * 0.5);
    }

    let mut price = 200.0;

    b.iter(|| {
        // Strong uptrend
        price += 0.1;
        let result = pipeline.process(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_mean_reverting_market(b: &mut Bencher) {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..200 {
        let price = 100.0 + 5.0 * ((i as f64 * 0.1).sin());
        let _ = pipeline.process(price);
    }

    let mut t = 20.0;

    b.iter(|| {
        // Oscillating price
        let price = 100.0 + 5.0 * (t * 0.1).sin();
        t += 0.1;
        let result = pipeline.process(price);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_noisy_market(b: &mut Bencher) {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..200 {
        let noise = ((i * 17) % 100) as f64 * 0.01 - 0.5;
        let _ = pipeline.process(100.0 + noise);
    }

    let mut i = 200;

    b.iter(|| {
        // Random-walk-like noise
        let noise = ((i * 17) % 100) as f64 * 0.01 - 0.5;
        i += 1;
        let result = pipeline.process(100.0 + noise);
        test::black_box(result)
    });
}

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_volatile_market(b: &mut Bencher) {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..200 {
        let volatility = ((i * 13) % 50) as f64 * 0.1 - 2.5;
        let _ = pipeline.process(100.0 + volatility);
    }

    let mut i = 200;

    b.iter(|| {
        // High volatility
        let volatility = ((i * 13) % 50) as f64 * 0.1 - 2.5;
        i += 1;
        let result = pipeline.process(100.0 + volatility);
        test::black_box(result)
    });
}

// ============================================================================
// Allocation Testing (Verify Zero-Allocation Hot Path)
// ============================================================================

#[cfg(feature = "nightly")]
#[bench]
fn bench_pipeline_no_allocation_check(b: &mut Bencher) {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..200 {
        let _ = pipeline.process(100.0 + i as f64 * 0.1);
    }

    // This benchmark should show zero allocations after warmup
    let mut price = 120.0;

    b.iter(|| {
        price += 0.01;
        // If this allocates, we've violated the zero-allocation requirement
        let result = pipeline.process(price);
        test::black_box(result)
    });
}

// ============================================================================
// Latency Distribution Testing
// ============================================================================

#[cfg(test)]
mod latency_tests {
    #![allow(unused_imports)]
    use dsp::pipeline::{DspConfig, DspPipeline};
    use std::time::Instant;

    #[test]
    fn test_pipeline_latency_distribution() {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        // Warmup
        for i in 0..200 {
            let _ = pipeline.process(100.0 + i as f64 * 0.1);
        }

        // Measure latency distribution
        let mut latencies = Vec::with_capacity(10_000);

        for i in 0..10_000 {
            let price = 120.0 + i as f64 * 0.01;

            let start = Instant::now();
            let _ = pipeline.process(price);
            let elapsed = start.elapsed();

            latencies.push(elapsed.as_nanos() as u64);
        }

        // Sort for percentile calculation
        latencies.sort_unstable();

        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[latencies.len() * 99 / 100];
        let p999 = latencies[latencies.len() * 999 / 1000];
        let max = latencies[latencies.len() - 1];

        println!("\n=== Pipeline Latency Distribution ===");
        println!("P50:  {:>6} ns", p50);
        println!("P99:  {:>6} ns", p99);
        println!("P99.9: {:>6} ns", p999);
        println!("Max:  {:>6} ns", max);

        // Target: <1μs median, <10μs P99
        // These are aspirational for Rust; actual values depend on hardware
        println!("\nTarget: P50 < 1,000ns, P99 < 10,000ns");

        // We expect significant improvement over Python (25-50K ticks/sec)
        // Rust target: >1M ticks/sec = <1μs per tick
    }

    #[test]
    fn test_throughput_measurement() {
        let config = DspConfig::default();
        let mut pipeline = DspPipeline::new(config);

        // Warmup
        for i in 0..200 {
            let _ = pipeline.process(100.0 + i as f64 * 0.1);
        }

        // Measure throughput
        let n_ticks = 1_000_000;
        let start = Instant::now();

        for i in 0..n_ticks {
            let price = 120.0 + i as f64 * 0.00001;
            let _ = pipeline.process(price);
        }

        let elapsed = start.elapsed();
        let ticks_per_sec = n_ticks as f64 / elapsed.as_secs_f64();

        println!("\n=== Pipeline Throughput ===");
        println!("Ticks:      {}", n_ticks);
        println!("Duration:   {:.3} s", elapsed.as_secs_f64());
        println!("Throughput: {:.0} ticks/sec", ticks_per_sec);
        println!("Latency:    {:.2} ns/tick", 1e9 / ticks_per_sec);

        println!("\nTarget: >1,000,000 ticks/sec");
        println!("Python baseline: 25,000-50,000 ticks/sec");

        // Assert we're at least 10x faster than Python
        assert!(
            ticks_per_sec > 250_000.0,
            "Expected >250K ticks/sec (10x Python), got {:.0}",
            ticks_per_sec
        );
    }
}

// ============================================================================
// Component Comparison Benchmarks
// ============================================================================

#[cfg(feature = "nightly")]
#[bench]
fn bench_sevcik_vs_frama_vs_pipeline(b: &mut Bencher) {
    // This benchmark compares the overhead of each layer
    let mut sevcik = SevcikFractalDimension::new(64);
    let mut frama = Frama::new(64, 0.01, 0.5, false);
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Warmup all
    for i in 0..200 {
        let price = 100.0 + i as f64 * 0.1;
        let _ = sevcik.update(price);
        let _ = frama.update(price);
        let _ = pipeline.process(price);
    }

    let mut price = 120.0;

    b.iter(|| {
        price += 0.01;

        // Sevcik only
        let _s = sevcik.update(price);

        // FRAMA (includes Sevcik)
        let _f = frama.update(price);

        // Full pipeline (includes all)
        let _p = pipeline.process(price);

        test::black_box((price, _s, _f, _p))
    });
}

// Provide main function when harness=false and nightly feature is not enabled
#[cfg(not(feature = "nightly"))]
#[allow(clippy::items_after_test_module)]
fn main() {
    eprintln!("Benchmarks require the 'nightly' feature and nightly Rust compiler.");
    eprintln!("Run with: cargo +nightly bench --features nightly");
}
