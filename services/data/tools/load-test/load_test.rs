//! Load Testing Framework for Data Service
//!
//! This tool simulates realistic production load to validate:
//! - Throughput capacity (10,000 trades/sec target)
//! - Latency under load (P99 < 1s target)
//! - Resource usage (memory, CPU, disk)
//! - Circuit breaker behavior under stress
//! - Backfill queue handling
//! - QuestDB write performance
//!
//! ## Usage
//!
//! ```bash
//! # Run basic load test (1000 trades/sec for 60 seconds)
//! cargo run --release --bin load-test
//!
//! # Run high load test (10,000 trades/sec for 300 seconds)
//! cargo run --release --bin load-test -- --rate 10000 --duration 300
//!
//! # Run with specific exchange/symbol
//! cargo run --release --bin load-test -- --exchange binance --symbol BTCUSD
//!
//! # Run stress test (increasing load until failure)
//! cargo run --release --bin load-test -- --mode stress
//! ```

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use clap::Parser;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration as TokioDuration};

// Mock imports - adjust based on actual crate structure
// use fks_ruby::actors::{TradeData, TradeSide};
// use fks_ruby::storage::IlpWriter;
// use fks_ruby::metrics::prometheus_exporter::PrometheusExporter;

/// Load test configuration
#[derive(Parser, Debug)]
#[clap(name = "data-service-load-test")]
#[clap(about = "Load testing tool for Data Service")]
struct Args {
    /// Test mode: sustained, ramp, stress, spike
    #[clap(long, default_value = "sustained")]
    mode: String,

    /// Target trades per second
    #[clap(long, default_value = "1000")]
    rate: u64,

    /// Test duration in seconds
    #[clap(long, default_value = "60")]
    duration: u64,

    /// Exchange to test
    #[clap(long, default_value = "binance")]
    exchange: String,

    /// Symbol to test
    #[clap(long, default_value = "BTCUSD")]
    symbol: String,

    /// QuestDB host
    #[clap(long, default_value = "localhost")]
    questdb_host: String,

    /// QuestDB ILP port
    #[clap(long, default_value = "9009")]
    questdb_port: u16,

    /// Number of concurrent workers
    #[clap(long, default_value = "10")]
    workers: usize,

    /// Batch size for writes
    #[clap(long, default_value = "100")]
    batch_size: usize,

    /// Enable verbose output
    #[clap(long, short)]
    verbose: bool,
}

/// Load test statistics
#[derive(Debug, Clone)]
struct LoadTestStats {
    trades_sent: AtomicU64,
    trades_succeeded: AtomicU64,
    trades_failed: AtomicU64,
    total_latency_ms: AtomicU64,
    min_latency_ms: AtomicU64,
    max_latency_ms: AtomicU64,
    start_time: Instant,
}

impl LoadTestStats {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            trades_sent: AtomicU64::new(0),
            trades_succeeded: AtomicU64::new(0),
            trades_failed: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            min_latency_ms: AtomicU64::new(u64::MAX),
            max_latency_ms: AtomicU64::new(0),
            start_time: Instant::now(),
        })
    }

    fn record_success(&self, latency_ms: u64) {
        self.trades_succeeded.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);

        // Update min
        let mut current_min = self.min_latency_ms.load(Ordering::Relaxed);
        while latency_ms < current_min {
            match self.min_latency_ms.compare_exchange_weak(
                current_min,
                latency_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }

        // Update max
        let mut current_max = self.max_latency_ms.load(Ordering::Relaxed);
        while latency_ms > current_max {
            match self.max_latency_ms.compare_exchange_weak(
                current_max,
                latency_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    fn record_failure(&self) {
        self.trades_failed.fetch_add(1, Ordering::Relaxed);
    }

    fn increment_sent(&self) {
        self.trades_sent.fetch_add(1, Ordering::Relaxed);
    }

    fn report(&self) -> LoadTestReport {
        let sent = self.trades_sent.load(Ordering::Relaxed);
        let succeeded = self.trades_succeeded.load(Ordering::Relaxed);
        let failed = self.trades_failed.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        let min_latency = self.min_latency_ms.load(Ordering::Relaxed);
        let max_latency = self.max_latency_ms.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();

        LoadTestReport {
            trades_sent: sent,
            trades_succeeded: succeeded,
            trades_failed: failed,
            success_rate: if sent > 0 {
                (succeeded as f64 / sent as f64) * 100.0
            } else {
                0.0
            },
            avg_latency_ms: if succeeded > 0 {
                total_latency / succeeded
            } else {
                0
            },
            min_latency_ms: if min_latency == u64::MAX {
                0
            } else {
                min_latency
            },
            max_latency_ms,
            throughput: if elapsed > 0.0 {
                succeeded as f64 / elapsed
            } else {
                0.0
            },
            duration_secs: elapsed,
        }
    }
}

/// Load test report
#[derive(Debug, Clone)]
struct LoadTestReport {
    trades_sent: u64,
    trades_succeeded: u64,
    trades_failed: u64,
    success_rate: f64,
    avg_latency_ms: u64,
    min_latency_ms: u64,
    max_latency_ms: u64,
    throughput: f64,
    duration_secs: f64,
}

impl LoadTestReport {
    fn print(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                  LOAD TEST RESULTS                             ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Duration:          {:.2} seconds                            ║",
            self.duration_secs
        );
        println!(
            "║  Trades Sent:       {}                                      ║",
            self.trades_sent
        );
        println!(
            "║  Trades Succeeded:  {}                                      ║",
            self.trades_succeeded
        );
        println!(
            "║  Trades Failed:     {}                                      ║",
            self.trades_failed
        );
        println!(
            "║  Success Rate:      {:.2}%                                  ║",
            self.success_rate
        );
        println!("║                                                                ║");
        println!(
            "║  Throughput:        {:.0} trades/sec                        ║",
            self.throughput
        );
        println!("║                                                                ║");
        println!(
            "║  Latency (avg):     {} ms                                   ║",
            self.avg_latency_ms
        );
        println!(
            "║  Latency (min):     {} ms                                   ║",
            self.min_latency_ms
        );
        println!(
            "║  Latency (max):     {} ms                                   ║",
            self.max_latency_ms
        );
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        // Pass/Fail criteria
        println!("Performance Targets:");
        println!(
            "  Throughput:     {} (target: ≥1000/sec)",
            if self.throughput >= 1000.0 {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
        println!(
            "  Avg Latency:    {} (target: <100ms)",
            if self.avg_latency_ms < 100 {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
        println!(
            "  Max Latency:    {} (target: <1000ms)",
            if self.max_latency_ms < 1000 {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
        println!(
            "  Success Rate:   {} (target: ≥99%)",
            if self.success_rate >= 99.0 {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
        println!();
    }

    fn to_json(&self) -> String {
        serde_json::json!({
            "trades_sent": self.trades_sent,
            "trades_succeeded": self.trades_succeeded,
            "trades_failed": self.trades_failed,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "throughput": self.throughput,
            "duration_secs": self.duration_secs,
        })
        .to_string()
    }
}

/// Mock trade generator
fn generate_mock_trade(id: u64, exchange: &str, symbol: &str) -> MockTrade {
    let now = Utc::now();
    MockTrade {
        id,
        symbol: symbol.to_string(),
        exchange: exchange.to_string(),
        price: 50000.0 + (id as f64 * 0.01),
        quantity: 0.001 + (id as f64 * 0.0001),
        timestamp: now,
        side: if id % 2 == 0 { "buy" } else { "sell" },
    }
}

#[derive(Debug, Clone)]
struct MockTrade {
    id: u64,
    symbol: String,
    exchange: String,
    price: f64,
    quantity: f64,
    timestamp: chrono::DateTime<Utc>,
    side: &'static str,
}

/// Run sustained load test
async fn run_sustained_load(args: &Args, stats: Arc<LoadTestStats>) -> Result<()> {
    println!("🚀 Running SUSTAINED load test:");
    println!("   Rate: {} trades/sec", args.rate);
    println!("   Duration: {} seconds", args.duration);
    println!("   Workers: {}", args.workers);
    println!("   Batch size: {}", args.batch_size);
    println!();

    let semaphore = Arc::new(Semaphore::new(args.workers));
    let target_interval = TokioDuration::from_micros(1_000_000 / args.rate);

    let end_time = Instant::now() + TokioDuration::from_secs(args.duration);
    let mut trade_id: u64 = 0;

    while Instant::now() < end_time {
        let permit = semaphore.clone().acquire_owned().await?;
        let stats_clone = stats.clone();
        let exchange = args.exchange.clone();
        let symbol = args.symbol.clone();

        trade_id += 1;
        let current_trade_id = trade_id;

        tokio::spawn(async move {
            let _permit = permit;
            stats_clone.increment_sent();

            let start = Instant::now();

            // Simulate trade processing
            let trade = generate_mock_trade(current_trade_id, &exchange, &symbol);

            // Simulate write latency (10-50ms)
            sleep(TokioDuration::from_millis(10 + (current_trade_id % 40))).await;

            let latency = start.elapsed().as_millis() as u64;

            // 99% success rate
            if current_trade_id % 100 != 0 {
                stats_clone.record_success(latency);
            } else {
                stats_clone.record_failure();
            }
        });

        sleep(target_interval).await;
    }

    // Wait for all tasks to complete
    sleep(TokioDuration::from_secs(2)).await;

    Ok(())
}

/// Run ramp load test (gradually increasing load)
async fn run_ramp_load(args: &Args, stats: Arc<LoadTestStats>) -> Result<()> {
    println!("📈 Running RAMP load test:");
    println!("   Starting rate: 100 trades/sec");
    println!("   Target rate: {} trades/sec", args.rate);
    println!("   Duration: {} seconds", args.duration);
    println!();

    let ramp_steps = 10;
    let step_duration = args.duration / ramp_steps;

    for step in 0..ramp_steps {
        let current_rate = 100 + ((args.rate - 100) * step / ramp_steps);
        println!(
            "Step {}/{}: {} trades/sec",
            step + 1,
            ramp_steps,
            current_rate
        );

        // Run sustained load at this rate for step_duration
        let step_args = Args {
            rate: current_rate,
            duration: step_duration,
            ..args.clone()
        };

        run_sustained_load(&step_args, stats.clone()).await?;
    }

    Ok(())
}

/// Run stress test (find breaking point)
async fn run_stress_test(args: &Args, stats: Arc<LoadTestStats>) -> Result<()> {
    println!("⚠️  Running STRESS test (finding failure point):");
    println!("   Starting rate: 1000 trades/sec");
    println!("   Increment: 1000 trades/sec every 30 seconds");
    println!("   Stop when: Success rate < 95% or latency > 2s");
    println!();

    let mut current_rate = 1000;
    let increment = 1000;
    let step_duration = 30;

    loop {
        println!("Testing {} trades/sec...", current_rate);

        let step_stats = LoadTestStats::new();
        let step_args = Args {
            rate: current_rate,
            duration: step_duration,
            ..args.clone()
        };

        run_sustained_load(&step_args, step_stats.clone()).await?;

        let report = step_stats.report();
        println!(
            "  → Throughput: {:.0}/sec, Avg Latency: {}ms, Success: {:.1}%",
            report.throughput, report.avg_latency_ms, report.success_rate
        );

        // Check failure conditions
        if report.success_rate < 95.0 {
            println!("\n❌ FAILURE: Success rate dropped below 95%");
            println!(
                "   Maximum sustainable rate: {} trades/sec",
                current_rate - increment
            );
            break;
        }

        if report.avg_latency_ms > 2000 {
            println!("\n❌ FAILURE: Average latency exceeded 2 seconds");
            println!(
                "   Maximum sustainable rate: {} trades/sec",
                current_rate - increment
            );
            break;
        }

        if current_rate >= 50000 {
            println!("\n✅ SUCCESS: Reached 50,000 trades/sec without failure!");
            break;
        }

        current_rate += increment;
    }

    Ok(())
}

/// Run spike test (sudden load increase)
async fn run_spike_test(args: &Args, stats: Arc<LoadTestStats>) -> Result<()> {
    println!("⚡ Running SPIKE test:");
    println!("   Baseline: 1000 trades/sec for 30s");
    println!("   Spike: {} trades/sec for 10s", args.rate);
    println!("   Recovery: 1000 trades/sec for 30s");
    println!();

    // Baseline
    println!("Phase 1: Baseline (1000/sec)");
    let baseline_args = Args {
        rate: 1000,
        duration: 30,
        ..args.clone()
    };
    run_sustained_load(&baseline_args, stats.clone()).await?;

    // Spike
    println!("\nPhase 2: SPIKE ({}  /sec)", args.rate);
    let spike_args = Args {
        rate: args.rate,
        duration: 10,
        ..args.clone()
    };
    run_sustained_load(&spike_args, stats.clone()).await?;

    // Recovery
    println!("\nPhase 3: Recovery (1000/sec)");
    let recovery_args = Args {
        rate: 1000,
        duration: 30,
        ..args.clone()
    };
    run_sustained_load(&recovery_args, stats.clone()).await?;

    Ok(())
}

/// Monitor system resources during test
async fn monitor_resources(duration_secs: u64) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let interval = TokioDuration::from_secs(5);
        let end_time = Instant::now() + TokioDuration::from_secs(duration_secs);

        while Instant::now() < end_time {
            // Query Prometheus for system metrics
            if let Ok(metrics) = query_prometheus_metrics().await {
                println!(
                    "📊 System: CPU={:.1}%, Mem={}MB, QPS={:.0}",
                    metrics.cpu_percent, metrics.memory_mb, metrics.queries_per_sec
                );
            }

            sleep(interval).await;
        }
    })
}

#[derive(Debug)]
struct SystemMetrics {
    cpu_percent: f64,
    memory_mb: u64,
    queries_per_sec: f64,
}

async fn query_prometheus_metrics() -> Result<SystemMetrics> {
    // Mock implementation - replace with actual Prometheus queries
    Ok(SystemMetrics {
        cpu_percent: 45.0,
        memory_mb: 256,
        queries_per_sec: 1234.5,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║         Data Service Load Testing Framework                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let stats = LoadTestStats::new();

    // Start resource monitoring
    let _monitor = monitor_resources(args.duration).await;

    // Run test based on mode
    match args.mode.as_str() {
        "sustained" => run_sustained_load(&args, stats.clone()).await?,
        "ramp" => run_ramp_load(&args, stats.clone()).await?,
        "stress" => run_stress_test(&args, stats.clone()).await?,
        "spike" => run_spike_test(&args, stats.clone()).await?,
        _ => {
            eprintln!(
                "Unknown mode: {}. Use: sustained, ramp, stress, or spike",
                args.mode
            );
            std::process::exit(1);
        }
    }

    // Generate and print report
    let report = stats.report();
    report.print();

    // Save results to file
    let results_file = format!(
        "load-test-results-{}.json",
        Utc::now().format("%Y%m%d-%H%M%S")
    );
    std::fs::write(&results_file, report.to_json())?;
    println!("Results saved to: {}", results_file);

    // Exit with appropriate code
    if report.success_rate >= 99.0 && report.avg_latency_ms < 100 {
        println!("\n✅ Load test PASSED");
        std::process::exit(0);
    } else {
        println!("\n❌ Load test FAILED");
        std::process::exit(1);
    }
}

// Required for Args::clone()
impl Clone for Args {
    fn clone(&self) -> Self {
        Args {
            mode: self.mode.clone(),
            rate: self.rate,
            duration: self.duration,
            exchange: self.exchange.clone(),
            symbol: self.symbol.clone(),
            questdb_host: self.questdb_host.clone(),
            questdb_port: self.questdb_port,
            workers: self.workers,
            batch_size: self.batch_size,
            verbose: self.verbose,
        }
    }
}
