//! Standalone Benchmark Example for Optimization Performance Testing
//!
//! This example provides a quick way to measure parallel optimization speedup
//! without requiring the full criterion benchmark setup.
//!
//! Run with:
//! ```bash
//! cargo run --release --example benchmark_optimization
//! ```
//!
//! Or with custom parameters:
//! ```bash
//! cargo run --release --example benchmark_optimization -- --combinations 400 --workers 8
//! ```

use chrono::{DateTime, Duration, Utc};
use janus_execution::sim::{
    OptimizationConfig, OptimizationDirection, OptimizationError, OptimizationMetric,
    OptimizationRunResult, ParameterRange, ParameterSet, StrategyEvaluator,
    WalkForwardBacktestRunner, WalkForwardConfig,
};
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Instant;

// ============================================================================
// Mock Strategy Evaluator
// ============================================================================

/// A mock strategy evaluator with configurable computation time
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

    fn eval_count(&self) -> u64 {
        self.eval_count.load(Ordering::Relaxed)
    }

    fn reset_count(&self) {
        self.eval_count.store(0, Ordering::Relaxed);
    }

    /// Simulate CPU-bound work
    fn do_work(&self, params: &ParameterSet) {
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
        std::hint::black_box(sum);
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
        self.do_work(params);

        let period = params.get("period").and_then(|v| v.as_int()).unwrap_or(20);
        let threshold = params
            .get("threshold")
            .and_then(|v| v.as_float())
            .unwrap_or(0.5);

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

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchConfig {
    combinations: usize,
    windows: usize,
    compute_us: u64,
    iterations: usize,
    max_workers: usize,
    run_scaling_test: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            combinations: 400,
            windows: 3,
            compute_us: 200,
            iterations: 5,
            max_workers: 0, // 0 = auto (use all CPUs)
            run_scaling_test: false,
        }
    }
}

fn parse_args() -> BenchConfig {
    let args: Vec<String> = env::args().collect();
    let mut config = BenchConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--combinations" | "-c" => {
                if i + 1 < args.len() {
                    config.combinations = args[i + 1].parse().unwrap_or(config.combinations);
                    i += 1;
                }
            }
            "--windows" | "-w" => {
                if i + 1 < args.len() {
                    config.windows = args[i + 1].parse().unwrap_or(config.windows);
                    i += 1;
                }
            }
            "--compute" | "-u" => {
                if i + 1 < args.len() {
                    config.compute_us = args[i + 1].parse().unwrap_or(config.compute_us);
                    i += 1;
                }
            }
            "--iterations" | "-i" => {
                if i + 1 < args.len() {
                    config.iterations = args[i + 1].parse().unwrap_or(config.iterations);
                    i += 1;
                }
            }
            "--workers" | "-j" => {
                if i + 1 < args.len() {
                    config.max_workers = args[i + 1].parse().unwrap_or(config.max_workers);
                    i += 1;
                }
            }
            "--scaling" | "-s" => {
                config.run_scaling_test = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    config
}

fn print_help() {
    println!(
        r#"
Optimization Benchmark

Usage: benchmark_optimization [OPTIONS]

Options:
  -c, --combinations <N>  Number of parameter combinations (default: 400)
  -w, --windows <N>       Number of walk-forward windows (default: 3)
  -u, --compute <US>      Simulated compute time per eval in microseconds (default: 200)
  -i, --iterations <N>    Number of benchmark iterations (default: 5)
  -j, --workers <N>       Max worker threads (0 = auto) (default: 0)
  -s, --scaling           Run worker scaling test
  -h, --help              Show this help message

Examples:
  # Quick benchmark with defaults
  cargo run --release --example benchmark_optimization

  # Custom configuration
  cargo run --release --example benchmark_optimization -- -c 900 -w 5 -u 500

  # Run worker scaling analysis
  cargo run --release --example benchmark_optimization -- --scaling
"#
    );
}

// ============================================================================
// Helper Functions
// ============================================================================

fn num_cpus() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn test_time_range() -> (DateTime<Utc>, DateTime<Utc>) {
    let end = Utc::now();
    let start = end - Duration::days(365);
    (start, end)
}

fn create_optimization_config(
    combinations: usize,
    parallel: bool,
    max_workers: usize,
) -> OptimizationConfig {
    // Calculate grid dimensions for desired combination count
    let side = (combinations as f64).sqrt().ceil() as usize;
    let period_count = side;
    let threshold_count = side;

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
// Benchmark Tests
// ============================================================================

fn run_speedup_benchmark(config: &BenchConfig) {
    let (start, end) = test_time_range();
    let strategy = MockStrategyEvaluator::new(config.compute_us);

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           Parallel Optimization Speedup Benchmark              ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Configuration:                                                 ║");
    println!(
        "║   Combinations:    {:>6}                                      ║",
        config.combinations
    );
    println!(
        "║   Windows:         {:>6}                                      ║",
        config.windows
    );
    println!(
        "║   Compute time:    {:>6} μs                                   ║",
        config.compute_us
    );
    println!(
        "║   Iterations:      {:>6}                                      ║",
        config.iterations
    );
    println!(
        "║   Available CPUs:  {:>6}                                      ║",
        num_cpus()
    );
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Sequential benchmark
    println!("Running sequential benchmark...");
    let seq_config = create_optimization_config(config.combinations, false, 1);
    let seq_wf_config = create_walk_forward_config(config.windows, seq_config);
    let seq_runner = WalkForwardBacktestRunner::new(seq_wf_config, start, end).unwrap();

    let mut seq_times = Vec::new();
    for i in 0..config.iterations {
        strategy.reset_count();
        let start_time = Instant::now();
        let _result = seq_runner.run_sync(&strategy).unwrap();
        let elapsed = start_time.elapsed();
        seq_times.push(elapsed.as_millis());
        println!(
            "  Iteration {}: {:>6}ms ({} evaluations)",
            i + 1,
            elapsed.as_millis(),
            strategy.eval_count()
        );
    }

    // Parallel benchmark
    println!("\nRunning parallel benchmark...");
    let workers = if config.max_workers == 0 {
        num_cpus()
    } else {
        config.max_workers
    };
    let par_config = create_optimization_config(config.combinations, true, workers);
    let par_wf_config = create_walk_forward_config(config.windows, par_config);
    let par_runner = WalkForwardBacktestRunner::new(par_wf_config, start, end).unwrap();

    let mut par_times = Vec::new();
    for i in 0..config.iterations {
        strategy.reset_count();
        let start_time = Instant::now();
        let _result = par_runner.run_sync(&strategy).unwrap();
        let elapsed = start_time.elapsed();
        par_times.push(elapsed.as_millis());
        println!(
            "  Iteration {}: {:>6}ms ({} evaluations, {} workers)",
            i + 1,
            elapsed.as_millis(),
            strategy.eval_count(),
            workers
        );
    }

    // Calculate statistics
    let avg_seq: f64 = seq_times.iter().sum::<u128>() as f64 / config.iterations as f64;
    let avg_par: f64 = par_times.iter().sum::<u128>() as f64 / config.iterations as f64;
    let speedup = avg_seq / avg_par;
    let efficiency = (speedup / workers as f64) * 100.0;

    let min_seq = seq_times.iter().min().unwrap_or(&0);
    let max_seq = seq_times.iter().max().unwrap_or(&0);
    let min_par = par_times.iter().min().unwrap_or(&0);
    let max_par = par_times.iter().max().unwrap_or(&0);

    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                         Results                                ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Sequential:  avg {:>7.1}ms  (min {:>5}ms, max {:>5}ms)        ║",
        avg_seq, min_seq, max_seq
    );
    println!(
        "║ Parallel:    avg {:>7.1}ms  (min {:>5}ms, max {:>5}ms)        ║",
        avg_par, min_par, max_par
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Speedup:         {:>6.2}x                                      ║",
        speedup
    );
    println!(
        "║ Efficiency:      {:>6.1}%  (speedup / workers)                 ║",
        efficiency
    );
    println!(
        "║ Time saved:      {:>6.1}ms per run                             ║",
        avg_seq - avg_par
    );
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Interpretation
    println!();
    if speedup >= workers as f64 * 0.7 {
        println!("✅ Excellent parallelization efficiency (≥70%)");
    } else if speedup >= workers as f64 * 0.5 {
        println!("⚠️  Good parallelization efficiency (50-70%)");
    } else {
        println!("❌ Poor parallelization efficiency (<50%)");
        println!("   Consider increasing compute_us or combinations for better scaling");
    }
}

fn run_worker_scaling_benchmark(config: &BenchConfig) {
    let (start, end) = test_time_range();
    let strategy = MockStrategyEvaluator::new(config.compute_us);
    let max_cpus = num_cpus();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              Worker Scaling Analysis                           ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Combinations: {:>6}   Compute: {:>6} μs                      ║",
        config.combinations, config.compute_us
    );
    println!(
        "║ Windows: {:>6}        Max CPUs: {:>6}                        ║",
        config.windows, max_cpus
    );
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Test range of worker counts
    let worker_counts: Vec<usize> = (1..=max_cpus.min(16)).collect();
    let mut results: Vec<(usize, f64)> = Vec::new();

    for workers in &worker_counts {
        let opt_config = create_optimization_config(config.combinations, *workers > 1, *workers);
        let wf_config = create_walk_forward_config(config.windows, opt_config);
        let runner = WalkForwardBacktestRunner::new(wf_config, start, end).unwrap();

        let mut times = Vec::new();
        for _ in 0..config.iterations.min(3) {
            strategy.reset_count();
            let start_time = Instant::now();
            let _result = runner.run_sync(&strategy).unwrap();
            times.push(start_time.elapsed().as_millis());
        }

        let avg: f64 = times.iter().sum::<u128>() as f64 / times.len() as f64;
        results.push((*workers, avg));
        print!("  Workers {:>2}: {:>7.1}ms", workers, avg);

        // Print progress bar
        let max_time = results.first().map(|(_, t)| *t).unwrap_or(avg);
        let bar_len = ((avg / max_time) * 30.0) as usize;
        print!("  [");
        for i in 0..30 {
            if i < 30 - bar_len {
                print!("█");
            } else {
                print!("░");
            }
        }
        println!("]");
    }

    // Print speedup table
    if let Some(&(_, baseline)) = results.first() {
        println!();
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                    Scaling Analysis                            ║");
        println!("╠════════╦════════════╦════════════╦═════════════════════════════╣");
        println!("║Workers ║   Time(ms) ║   Speedup  ║  Efficiency                 ║");
        println!("╠════════╬════════════╬════════════╬═════════════════════════════╣");

        for (workers, time) in &results {
            let speedup = baseline / time;
            let efficiency = (speedup / *workers as f64) * 100.0;
            let eff_bar_len = (efficiency / 100.0 * 20.0) as usize;

            print!("║  {:>4}  ║ {:>9.1} ║ {:>9.2}x ║ ", workers, time, speedup);
            for i in 0..20 {
                if i < eff_bar_len {
                    print!("█");
                } else {
                    print!("░");
                }
            }
            println!("{:>4.0}% ║", efficiency);
        }
        println!("╚════════╩════════════╩════════════╩═════════════════════════════╝");

        // Find optimal worker count
        let optimal = results
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        println!();
        println!(
            "🎯 Optimal worker count: {} ({}ms, {:.2}x speedup)",
            optimal.0,
            optimal.1,
            baseline / optimal.1
        );

        // Check for diminishing returns
        let half_cpus = max_cpus / 2;
        if let Some((_, half_time)) = results.iter().find(|(w, _)| *w == half_cpus)
            && let Some((_, full_time)) = results.iter().find(|(w, _)| *w == max_cpus)
        {
            let marginal_gain = (half_time - full_time) / half_time * 100.0;
            if marginal_gain < 20.0 {
                println!(
                    "💡 Tip: Using {} workers provides similar performance to {} workers",
                    half_cpus, max_cpus
                );
                println!(
                    "   with lower resource usage (only {:.1}% gain from doubling)",
                    marginal_gain
                );
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let config = parse_args();

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("       FKS Execution - Parallel Optimization Benchmark");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    if config.run_scaling_test {
        run_worker_scaling_benchmark(&config);
    } else {
        run_speedup_benchmark(&config);
    }

    println!();
}
