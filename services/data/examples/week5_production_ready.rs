//! Week 5 Production-Ready Backfill Example
//!
//! This example demonstrates all Week 5 features:
//! - QuestDB batch writes via ILP
//! - Circuit breaker integration
//! - Post-backfill verification
//! - Structured logging with correlation IDs
//! - Prometheus metrics
//! - Error handling and resilience
//!
//! ## Prerequisites
//!
//! Start the integration environment:
//! ```bash
//! docker-compose -f docker-compose.integration.yml up -d
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo run --example week5_production_ready
//! ```
//!
//! ## What This Example Does
//!
//! 1. Creates an ILP writer connected to QuestDB
//! 2. Sets up a circuit breaker for resilience
//! 3. Executes a backfill with correlation ID tracking
//! 4. Writes trades to QuestDB in batches
//! 5. Verifies the data was written correctly
//! 6. Records metrics throughout the process
//! 7. Handles errors gracefully

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use janus_data::actors::{TradeData, TradeSide};
use janus_data::backfill::executor::{BackfillExecutor, BackfillRequest};
use janus_data::logging::*;
use janus_data::metrics::prometheus_exporter::PrometheusExporter;
use janus_data::storage::IlpWriter;
use janus_rate_limiter::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

// Configuration
const QUESTDB_HOST: &str = "localhost";
const QUESTDB_ILP_PORT: u16 = 9009;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("week5_production_ready=debug,fks_ruby=debug")
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .compact()
        .init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Week 5 Production-Ready Backfill Example                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Initialize components
    println!("📦 Step 1: Initializing components...");
    let components = initialize_components().await?;
    println!("✅ Components initialized");
    println!();

    // Step 2: Demonstrate circuit breaker
    println!("🔌 Step 2: Testing circuit breaker...");
    demonstrate_circuit_breaker(&components.circuit_breaker).await?;
    println!("✅ Circuit breaker tested");
    println!();

    // Step 3: Write batch data to QuestDB
    println!("💾 Step 3: Writing test data to QuestDB...");
    write_test_data(&components).await?;
    println!("✅ Test data written");
    println!();

    // Step 4: Execute production backfill
    println!("⚙️  Step 4: Executing production backfill...");
    execute_production_backfill(&components).await?;
    println!("✅ Backfill completed");
    println!();

    // Step 5: Verify data and check metrics
    println!("🔍 Step 5: Verifying data and metrics...");
    verify_and_report(&components).await?;
    println!("✅ Verification complete");
    println!();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  All Week 5 features demonstrated successfully! 🎉       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("📊 View metrics at: http://localhost:9091/metrics");
    println!("📈 View Grafana at: http://localhost:3000 (admin/admin)");
    println!("🗄️  View QuestDB at: http://localhost:9000");

    Ok(())
}

/// Components needed for production backfill
struct Components {
    ilp_writer: Arc<Mutex<IlpWriter>>,
    circuit_breaker: Arc<CircuitBreaker>,
    metrics: Arc<PrometheusExporter>,
    executor: BackfillExecutor,
}

/// Initialize all components
async fn initialize_components() -> Result<Components> {
    // Create ILP writer
    println!(
        "  → Connecting to QuestDB at {}:{}",
        QUESTDB_HOST, QUESTDB_ILP_PORT
    );
    let ilp_writer = IlpWriter::new(
        QUESTDB_HOST,
        QUESTDB_ILP_PORT,
        1000, // buffer size
        100,  // flush interval ms
    )
    .await
    .context("Failed to create ILP writer. Is QuestDB running?")?;

    let ilp_writer = Arc::new(Mutex::new(ilp_writer));
    println!("  → QuestDB connected");

    // Create circuit breaker
    println!("  → Setting up circuit breaker");
    let cb_config = CircuitBreakerConfig {
        failure_threshold: 5,
        success_threshold: 2,
        timeout: std::time::Duration::from_secs(60),
    };
    let circuit_breaker = CircuitBreaker::new(cb_config);
    println!("  → Circuit breaker configured (5 failures → OPEN)");

    // Create metrics exporter
    println!("  → Initializing Prometheus metrics");
    let metrics = Arc::new(PrometheusExporter::new());
    println!("  → Metrics exporter ready");

    // Create backfill executor
    println!("  → Creating backfill executor");
    let executor = BackfillExecutor::new_with_writer(Some(ilp_writer.clone()))?;
    println!("  → Executor ready");

    Ok(Components {
        ilp_writer,
        circuit_breaker,
        metrics,
        executor,
    })
}

/// Demonstrate circuit breaker functionality
async fn demonstrate_circuit_breaker(breaker: &Arc<CircuitBreaker>) -> Result<()> {
    println!("  → Testing circuit breaker state machine");

    // Initial state should be Closed
    assert_eq!(breaker.state(), CircuitState::Closed);
    println!("  → Initial state: {:?}", breaker.state());

    // Simulate successful calls
    for i in 1..=3 {
        breaker
            .call(|| async { Ok::<(), anyhow::Error>(()) })
            .await?;
        println!("  → Success {}/3 - State: {:?}", i, breaker.state());
    }

    // Circuit should still be closed
    assert_eq!(breaker.state(), CircuitState::Closed);
    println!("  → Circuit remains CLOSED after successes");

    // Simulate rate limit failures
    println!("  → Simulating rate limit failures (429 errors)...");
    for i in 1..=5 {
        let _ = breaker
            .call(|| async { Err::<(), _>(anyhow::anyhow!("429 Rate limit exceeded")) })
            .await;
        println!(
            "  → Failure {}/5 - State: {:?}, Failures: {}",
            i,
            breaker.state(),
            breaker.failure_count()
        );
    }

    // Circuit should be open now
    assert_eq!(breaker.state(), CircuitState::Open);
    println!("  → ⚠️  Circuit is now OPEN (failing fast)");

    // Calls should fail fast
    let result = breaker.call(|| async { Ok::<(), anyhow::Error>(()) }).await;
    assert!(result.is_err());
    println!("  → Subsequent calls fail immediately (no API request)");

    // Reset for remaining examples
    breaker.force_close();
    println!("  → Circuit reset to CLOSED for remaining examples");

    Ok(())
}

/// Write test data to QuestDB using batch writes
async fn write_test_data(components: &Components) -> Result<()> {
    let correlation_id = CorrelationId::new();
    println!("  → Correlation ID: {}", correlation_id);

    // Generate test trades
    let base_time = Utc::now();
    let mut trades = Vec::new();

    println!("  → Generating 500 test trades...");
    for i in 0..500 {
        trades.push(TradeData {
            symbol: "TEST-USDT".to_string(),
            exchange: "test".to_string(),
            side: if i % 2 == 0 {
                TradeSide::Buy
            } else {
                TradeSide::Sell
            },
            price: 100.0 + (i as f64 * 0.1),
            amount: 1.0,
            exchange_ts: (base_time - Duration::minutes(10)).timestamp_millis() + (i as i64 * 1000),
            receipt_ts: base_time.timestamp_millis(),
            trade_id: format!("test-{}", i),
        });
    }

    // Write in batches
    let start = Instant::now();
    let mut writer = components.ilp_writer.lock().await;

    println!("  → Writing batch to QuestDB...");
    let count = writer.write_trade_batch(&trades).await?;

    let duration = start.elapsed();
    println!(
        "  → ✅ Wrote {} trades in {:?} ({:.0} trades/sec)",
        count,
        duration,
        count as f64 / duration.as_secs_f64()
    );

    // Check buffer state
    let buffer_state = writer.buffer_state();
    println!(
        "  → Buffer: {}/{} lines ({:.1}% utilized)",
        buffer_state.lines_buffered, buffer_state.max_lines, buffer_state.utilization_pct
    );

    // Log the write
    log_questdb_write(
        &correlation_id,
        "test",
        "TEST-USDT",
        count,
        duration.as_millis() as u64,
    );

    // Record metrics
    components.metrics.record_questdb_write("trades");
    components
        .metrics
        .record_questdb_write_latency("trades", duration.as_secs_f64());

    Ok(())
}

/// Execute a production backfill with all features
async fn execute_production_backfill(components: &Components) -> Result<()> {
    let correlation_id = CorrelationId::new();
    println!("  → Correlation ID: {}", correlation_id);

    // Create backfill request for a small time window
    let end_time = Utc::now();
    let start_time = end_time - Duration::minutes(5);

    let request = BackfillRequest {
        correlation_id,
        exchange: "binance".to_string(),
        symbol: "BTCUSD".to_string(),
        start_time,
        end_time,
        estimated_trades: 100,
    };

    println!("  → Exchange: {}", request.exchange);
    println!("  → Symbol: {}", request.symbol);
    println!("  → Time range: {} to {}", start_time, end_time);
    println!("  → Estimated trades: {}", request.estimated_trades);
    println!();

    // Check circuit breaker before executing
    let cb_state = components.circuit_breaker.state();
    log_circuit_breaker_checked(&correlation_id, &request.exchange, cb_state);
    println!("  → Circuit breaker state: {:?}", cb_state);

    // Execute backfill (wrapped in circuit breaker inside executor)
    println!("  → Starting backfill execution...");
    let start = Instant::now();

    let result = components
        .executor
        .execute_backfill(request.clone())
        .await?;

    let duration = start.elapsed();

    // Report results
    println!();
    if result.success {
        println!("  → ✅ Backfill succeeded!");
        println!("     - Trades filled: {}", result.trades_filled);
        println!("     - Duration: {:?}", duration);
        println!(
            "     - Throughput: {:.0} trades/sec",
            result.trades_filled as f64 / duration.as_secs_f64()
        );

        // Check verification
        if let Some(verification) = result.verification {
            println!("     - Verification: {}", verification.verified);
            println!("     - Rows found: {}", verification.rows_found);

            log_verification_completed(
                &correlation_id,
                &request.exchange,
                &request.symbol,
                verification.verified,
                verification.rows_found,
            );
        } else {
            println!("     - Verification: (not performed)");
        }
    } else {
        println!("  → ❌ Backfill failed!");
        if let Some(error) = result.error {
            println!("     - Error: {}", error);
        }
    }

    Ok(())
}

/// Verify data and report metrics
async fn verify_and_report(components: &Components) -> Result<()> {
    // Get ILP writer stats
    let writer = components.ilp_writer.lock().await;
    let stats = writer.stats();
    let buffer_state = writer.buffer_state();

    println!("  → ILP Writer Statistics:");
    println!("     - Lines written: {}", stats.lines_written);
    println!("     - Flushes completed: {}", stats.flushes_completed);
    println!("     - Flush errors: {}", stats.flush_errors);
    println!(
        "     - Current buffer: {} lines",
        buffer_state.lines_buffered
    );
    println!(
        "     - Buffer utilization: {:.1}%",
        buffer_state.utilization_pct
    );

    // Check if writer is healthy
    let is_healthy = writer.is_healthy();
    println!(
        "     - Health status: {}",
        if is_healthy {
            "✅ HEALTHY"
        } else {
            "❌ UNHEALTHY"
        }
    );

    drop(writer); // Release lock

    // Export Prometheus metrics
    println!();
    println!("  → Exporting Prometheus metrics...");
    let metrics_output = components.metrics.export()?;

    // Show sample metrics
    println!("     - Sample metrics:");
    for line in metrics_output.lines().take(10) {
        if !line.starts_with('#') && !line.is_empty() {
            println!("       {}", line);
        }
    }
    println!(
        "       ... ({} total lines)",
        metrics_output.lines().count()
    );

    // Verify QuestDB connection
    println!();
    println!("  → Verifying QuestDB connectivity...");
    let can_connect =
        tokio::net::TcpStream::connect(format!("{}:{}", QUESTDB_HOST, QUESTDB_ILP_PORT))
            .await
            .is_ok();
    println!(
        "     - QuestDB ILP: {}",
        if can_connect {
            "✅ Connected"
        } else {
            "❌ Disconnected"
        }
    );

    // Circuit breaker status
    println!();
    println!("  → Circuit Breaker Status:");
    println!("     - State: {:?}", components.circuit_breaker.state());
    println!(
        "     - Failure count: {}",
        components.circuit_breaker.failure_count()
    );
    println!(
        "     - Success count: {}",
        components.circuit_breaker.success_count()
    );

    Ok(())
}
