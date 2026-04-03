//! Benchmark harness for parallel optimization performance testing
//!
//! This benchmark suite measures the performance of walk-forward optimization
//! with different configurations:
//! - Sequential vs parallel execution
//! - Different worker counts
//! - Different parameter combination counts
//! - Different strategy evaluation complexities
//!
//! Run with:
//! ```bash
//! cargo bench --bench optimization_benchmark
//! ```
//!
//! For detailed HTML reports:
//! ```bash
//! cargo bench --bench optimization_benchmark -- --save-baseline baseline
//! ```

use chrono::{DateTime, Duration, Utc};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use janus_execution::sim::{
    OptimizationConfig, OptimizationDirection, OptimizationError, OptimizationMetric,
    OptimizationRunResult, ParameterRange, ParameterSet, StrategyEvaluator,
    WalkForwardBacktestRunner, WalkForwardConfig,
};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

// ============================================================================
// Mock Strategy Evaluators
// ============================================================================

/// A mock strategy evaluator with configurable computation time
///
/// Used to simulate different strategy evaluation complexities
#[derive(Clone)]
struct MockStrategyEvaluator {
    /// Base computation time in microseconds
    base_compute_us: u64,
    /// Evaluation counter for verification
    eval_count: std::sync::Arc<AtomicU64>,
}

impl MockStrategyEvaluator {
    fn new(base_compute_us: u64) -> Self {
        Self {
            base_compute_us,
            eval_count: std::sync::Arc::new(AtomicU64::new(0)),
        }
    }

    #[allow(dead_code)]
    fn eval_count(&self) -> u64 {
        self.eval_count.load(Ordering::Relaxed)
    }

    fn reset_count(&self) {
        self.eval_count.store(0, Ordering::Relaxed)
    }

    /// Simulate CPU-bound work
    fn do_work(&self, params: &ParameterSet) {
        // Use params to vary the work slightly (prevents optimization)
        let multiplier = params.get("period").and_then(|v| v.as_int()).unwrap_or(1) as u64;

        let iterations = self
            .base_compute_us
            .saturating_mul(multiplier.max(1) / 10 + 1);

        // CPU-intensive busy loop
        let mut sum: u64 = 0;
        for i in 0..iterations {
            sum = sum.wrapping_add(i);
            sum = sum.wrapping_mul(17);
            sum ^= sum >> 3;
        }
        black_box(sum);
    }
}

impl StrategyEvaluator for MockStrategyEvaluator {
    fn evaluate(
        &self,
        params: &ParameterSet,
        _start_time: DateTime<Utc>,
        _end_time: DateTime<Utc>,
    ) -> Result<OptimizationRunResult, OptimizationError> {
        self.eval_count.fetch_add(1, Ordering::Relaxed);

        // Simulate computation
        self.do_work(params);

        // Generate deterministic results based on params
        let period = params.get("period").and_then(|v| v.as_int()).unwrap_or(20);
        let threshold = params
            .get("threshold")
            .and_then(|v| v.as_float())
            .unwrap_or(0.5);

        // Simulate a strategy where middle values perform best
        let period_score = 1.0 - ((period as f64 - 30.0) / 20.0).powi(2);
        let threshold_score = 1.0 - ((threshold - 0.6) / 0.4).powi(2);
        let sharpe = period_score * 0.5 + threshold_score * 0.5 + 0.5;

        Ok(OptimizationRunResult {
            parameters: params.clone(),
            metrics: HashMap::new(),
            metric_value: sharpe,
            total_trades: 100,
            win_rate: 0.55 + threshold_score * 0.1,
            max_drawdown: 0.15 - period_score * 0.05,
            sharpe_ratio: Some(sharpe),
            profit_factor: Some(1.5 + period_score * 0.5),
            run_duration_ms: self.base_compute_us / 1000,
        })
    }
}

/// A more expensive mock evaluator that simulates complex strategy computations
struct HeavyMockStrategyEvaluator {
    inner: MockStrategyEvaluator,
    /// Additional memory allocation to simulate real strategies
    _buffer_size: usize,
}

impl HeavyMockStrategyEvaluator {
    fn new(base_compute_us: u64, buffer_size: usize) -> Self {
        Self {
            inner: MockStrategyEvaluator::new(base_compute_us),
            _buffer_size: buffer_size,
        }
    }

    #[allow(dead_code)]
    fn eval_count(&self) -> u64 {
        self.inner.eval_count()
    }

    #[allow(dead_code)]
    fn reset_count(&self) {
        self.inner.reset_count()
    }
}

impl StrategyEvaluator for HeavyMockStrategyEvaluator {
    fn evaluate(
        &self,
        params: &ParameterSet,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<OptimizationRunResult, OptimizationError> {
        // Allocate and use a buffer to simulate memory-intensive operations
        let buffer: Vec<f64> = vec![0.0; self._buffer_size];
        black_box(&buffer);

        self.inner.evaluate(params, start_time, end_time)
    }
}

// ============================================================================
// Benchmark Helpers
// ============================================================================

/// Create test time range
fn test_time_range() -> (DateTime<Utc>, DateTime<Utc>) {
    let end = Utc::now();
    let start = end - Duration::days(365);
    (start, end)
}

/// Create optimization config with given parameter counts
fn create_optimization_config(
    period_count: usize,
    threshold_count: usize,
    parallel: bool,
    max_workers: usize,
) -> OptimizationConfig {
    let period_step = if period_count > 1 {
        (50 - 10) / (period_count - 1) as i64
    } else {
        1
    };

    let threshold_step = if threshold_count > 1 {
        (1.0 - 0.1) / (threshold_count - 1) as f64
    } else {
        0.1
    };

    OptimizationConfig::new()
        .with_parameter(ParameterRange::int("period", 10, 50, period_step.max(1)))
        .with_parameter(ParameterRange::float(
            "threshold",
            0.1,
            1.0,
            threshold_step.max(0.01),
        ))
        .with_metric(OptimizationMetric::SharpeRatio)
        .with_direction(OptimizationDirection::Maximize)
        .with_parallel(parallel)
        .with_max_workers(max_workers)
}

/// Create walk-forward config
fn create_walk_forward_config(
    num_windows: usize,
    optimization: OptimizationConfig,
) -> WalkForwardConfig {
    WalkForwardConfig::new(num_windows)
        .with_in_sample_pct(0.7)
        .with_min_trades(10)
        .rolling()
        .with_optimization(optimization)
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Benchmark sequential vs parallel optimization
fn bench_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_vs_parallel");
    group.sample_size(20);

    let (start, end) = test_time_range();
    let strategy = MockStrategyEvaluator::new(100); // 100us base compute time

    // Test with different combination counts
    for &combinations in &[25, 100, 400] {
        let side = (combinations as f64).sqrt() as usize;

        // Sequential
        {
            let config = create_optimization_config(side, side, false, 1);
            let wf_config = create_walk_forward_config(3, config);
            let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

            group.throughput(Throughput::Elements(combinations as u64));
            group.bench_with_input(
                BenchmarkId::new("sequential", combinations),
                &combinations,
                |b, _| {
                    b.iter(|| {
                        strategy.reset_count();
                        runner.run_sync(&strategy).unwrap()
                    })
                },
            );
        }

        // Parallel with default workers
        {
            let config = create_optimization_config(side, side, true, 0);
            let wf_config = create_walk_forward_config(3, config);
            let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

            group.bench_with_input(
                BenchmarkId::new("parallel_auto", combinations),
                &combinations,
                |b, _| {
                    b.iter(|| {
                        strategy.reset_count();
                        runner.run_sync(&strategy).unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark different worker counts
fn bench_worker_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_counts");
    group.sample_size(15);

    let (start, end) = test_time_range();
    let strategy = MockStrategyEvaluator::new(200); // 200us base compute time

    let combinations = 400; // 20x20 grid
    let side = 20;
    let available_cpus = num_cpus();

    // Test different worker counts
    let worker_counts: Vec<usize> = vec![1, 2, 4, 8, 16]
        .into_iter()
        .filter(|&w| w <= available_cpus * 2)
        .collect();

    for &workers in &worker_counts {
        let config = create_optimization_config(side, side, workers > 1, workers);
        let wf_config = create_walk_forward_config(3, config);
        let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

        group.throughput(Throughput::Elements(combinations as u64));
        group.bench_with_input(BenchmarkId::new("workers", workers), &workers, |b, _| {
            b.iter(|| {
                strategy.reset_count();
                runner.run_sync(&strategy).unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark scaling with combination count
fn bench_combination_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("combination_scaling");
    group.sample_size(10);

    let (start, end) = test_time_range();
    let strategy = MockStrategyEvaluator::new(50); // Fast evaluations to test scaling

    // Test scaling: 25, 100, 225, 400, 625 combinations
    for &side in &[5, 10, 15, 20, 25] {
        let combinations = side * side;

        // Parallel
        let config = create_optimization_config(side, side, true, 0);
        let wf_config = create_walk_forward_config(3, config);
        let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

        group.throughput(Throughput::Elements(combinations as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel", combinations),
            &combinations,
            |b, _| {
                b.iter(|| {
                    strategy.reset_count();
                    runner.run_sync(&strategy).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark different evaluation complexities
fn bench_evaluation_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation_complexity");
    group.sample_size(10);

    let (start, end) = test_time_range();
    let _combinations = 100; // Fixed 10x10 grid
    let side = 10;

    // Test different compute times (in microseconds)
    for &compute_us in &[10, 50, 100, 500, 1000] {
        let strategy = MockStrategyEvaluator::new(compute_us);

        // Sequential
        {
            let config = create_optimization_config(side, side, false, 1);
            let wf_config = create_walk_forward_config(2, config);
            let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

            group.bench_with_input(
                BenchmarkId::new("sequential_us", compute_us),
                &compute_us,
                |b, _| {
                    b.iter(|| {
                        strategy.reset_count();
                        runner.run_sync(&strategy).unwrap()
                    })
                },
            );
        }

        // Parallel
        {
            let config = create_optimization_config(side, side, true, 0);
            let wf_config = create_walk_forward_config(2, config);
            let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

            group.bench_with_input(
                BenchmarkId::new("parallel_us", compute_us),
                &compute_us,
                |b, _| {
                    b.iter(|| {
                        strategy.reset_count();
                        runner.run_sync(&strategy).unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark walk-forward window counts
fn bench_window_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_counts");
    group.sample_size(10);

    let (start, end) = test_time_range();
    let strategy = MockStrategyEvaluator::new(50);
    let combinations = 64; // 8x8 grid
    let side = 8;

    for &windows in &[2, 3, 5, 7, 10] {
        let config = create_optimization_config(side, side, true, 0);
        let wf_config = create_walk_forward_config(windows, config);
        let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

        // Each window runs IS optimization + OOS evaluation
        let total_evals = windows * combinations * 2;
        group.throughput(Throughput::Elements(total_evals as u64));
        group.bench_with_input(BenchmarkId::new("windows", windows), &windows, |b, _| {
            b.iter(|| {
                strategy.reset_count();
                runner.run_sync(&strategy).unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark memory-intensive evaluations
fn bench_memory_intensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_intensive");
    group.sample_size(10);

    let (start, end) = test_time_range();
    let combinations = 64; // 8x8 grid
    let side = 8;

    // Test different buffer sizes (simulating price history, indicators, etc.)
    for &buffer_kb in &[1, 10, 100, 1000] {
        let buffer_size = buffer_kb * 1024 / 8; // Convert KB to f64 count
        let strategy = HeavyMockStrategyEvaluator::new(50, buffer_size);

        let config = create_optimization_config(side, side, true, 0);
        let wf_config = create_walk_forward_config(3, config);
        let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

        group.throughput(Throughput::Bytes((buffer_kb * 1024 * combinations) as u64));
        group.bench_with_input(
            BenchmarkId::new("buffer_kb", buffer_kb),
            &buffer_kb,
            |b, _| {
                b.iter(|| {
                    strategy.reset_count();
                    runner.run_sync(&strategy).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark speedup calculation (measures actual speedup vs sequential)
fn bench_speedup_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_measurement");
    group.sample_size(10);

    let (start, end) = test_time_range();
    let strategy = MockStrategyEvaluator::new(500); // 500us to make parallel benefits visible
    let combinations = 400;
    let side = 20;

    // Sequential baseline
    let seq_config = create_optimization_config(side, side, false, 1);
    let seq_wf_config = create_walk_forward_config(3, seq_config);
    let seq_runner = WalkForwardBacktestRunner::new(seq_wf_config, start, end).unwrap();

    group.throughput(Throughput::Elements(combinations as u64));
    group.bench_function("baseline_sequential", |b| {
        b.iter(|| {
            strategy.reset_count();
            seq_runner.run_sync(&strategy).unwrap()
        })
    });

    // Parallel with all CPUs
    let par_config = create_optimization_config(side, side, true, 0);
    let par_wf_config = create_walk_forward_config(3, par_config);
    let par_runner = WalkForwardBacktestRunner::new(par_wf_config, start, end).unwrap();

    group.bench_function("parallel_all_cpus", |b| {
        b.iter(|| {
            strategy.reset_count();
            par_runner.run_sync(&strategy).unwrap()
        })
    });

    group.finish();
}

// ============================================================================
// Utility Functions
// ============================================================================

fn num_cpus() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = optimization_benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_sequential_vs_parallel,
        bench_worker_counts,
        bench_combination_scaling,
        bench_evaluation_complexity,
        bench_window_counts,
        bench_memory_intensive,
        bench_speedup_measurement
);

criterion_main!(optimization_benches);

// ============================================================================
// Manual Benchmark Runner (for quick testing without criterion)
// ============================================================================

#[cfg(test)]
#[allow(unused_imports)]
mod manual_benchmarks {
    use super::{
        MockStrategyEvaluator, create_optimization_config, create_walk_forward_config, num_cpus,
        test_time_range,
    };
    use std::time::Instant;

    /// Quick manual benchmark for development
    #[test]
    #[ignore] // Run with: cargo test --release manual_speedup_test -- --ignored --nocapture
    fn manual_speedup_test() {
        let (start, end) = test_time_range();
        let strategy = MockStrategyEvaluator::new(200);
        let combinations = 400;
        let side = 20;
        let iterations = 5;

        println!("\n=== Manual Speedup Test ===");
        println!("Combinations: {}", combinations);
        println!("CPUs available: {}", num_cpus());
        println!("Iterations: {}\n", iterations);

        // Sequential
        let seq_config = create_optimization_config(side, side, false, 1);
        let seq_wf_config = create_walk_forward_config(3, seq_config);
        let seq_runner = WalkForwardBacktestRunner::new(seq_wf_config, start, end).unwrap();

        let mut seq_times = Vec::new();
        for i in 0..iterations {
            strategy.reset_count();
            let start_time = Instant::now();
            let _result = seq_runner.run_sync(&strategy).unwrap();
            let elapsed = start_time.elapsed();
            seq_times.push(elapsed.as_millis());
            println!(
                "Sequential run {}: {}ms ({} evals)",
                i + 1,
                elapsed.as_millis(),
                strategy.eval_count()
            );
        }

        // Parallel
        let par_config = create_optimization_config(side, side, true, 0);
        let par_wf_config = create_walk_forward_config(3, par_config);
        let par_runner = WalkForwardBacktestRunner::new(par_wf_config, start, end).unwrap();

        let mut par_times = Vec::new();
        for i in 0..iterations {
            strategy.reset_count();
            let start_time = Instant::now();
            let _result = par_runner.run_sync(&strategy).unwrap();
            let elapsed = start_time.elapsed();
            par_times.push(elapsed.as_millis());
            println!(
                "Parallel run {}: {}ms ({} evals)",
                i + 1,
                elapsed.as_millis(),
                strategy.eval_count()
            );
        }

        let avg_seq: f64 = seq_times.iter().sum::<u128>() as f64 / iterations as f64;
        let avg_par: f64 = par_times.iter().sum::<u128>() as f64 / iterations as f64;
        let speedup = avg_seq / avg_par;

        println!("\n=== Results ===");
        println!("Avg Sequential: {:.1}ms", avg_seq);
        println!("Avg Parallel:   {:.1}ms", avg_par);
        println!("Speedup:        {:.2}x", speedup);
        println!(
            "Efficiency:     {:.1}%",
            (speedup / num_cpus() as f64) * 100.0
        );
    }

    /// Test worker scaling
    #[test]
    #[ignore]
    fn manual_worker_scaling_test() {
        let (start, end) = test_time_range();
        let strategy = MockStrategyEvaluator::new(200);
        let combinations = 400;
        let side = 20;
        let iterations = 3;
        let max_cpus = num_cpus();

        println!("\n=== Worker Scaling Test ===");
        println!("Combinations: {}", combinations);
        println!("Max CPUs: {}\n", max_cpus);

        let worker_counts: Vec<usize> = (1..=max_cpus.min(16)).collect();
        let mut results: Vec<(usize, f64)> = Vec::new();

        for workers in worker_counts {
            let config = create_optimization_config(side, side, workers > 1, workers);
            let wf_config = create_walk_forward_config(3, config);
            let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

            let mut times = Vec::new();
            for _ in 0..iterations {
                strategy.reset_count();
                let start_time = Instant::now();
                let _result = runner.run_sync(&strategy).unwrap();
                times.push(start_time.elapsed().as_millis());
            }

            let avg: f64 = times.iter().sum::<u128>() as f64 / iterations as f64;
            results.push((workers, avg));
            println!("Workers {:2}: {:.1}ms", workers, avg);
        }

        // Calculate and display speedups
        if let Some(&(_, baseline)) = results.first() {
            println!("\n=== Speedups vs Single Worker ===");
            for (workers, time) in &results {
                let speedup = baseline / time;
                let efficiency = speedup / *workers as f64;
                println!(
                    "Workers {:2}: {:.2}x speedup, {:.1}% efficiency",
                    workers,
                    speedup,
                    efficiency * 100.0
                );
            }
        }
    }
}
