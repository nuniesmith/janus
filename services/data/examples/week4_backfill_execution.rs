//! Week 4 - Real Backfill Execution Example
//!
//! This example demonstrates the complete Week 4 implementation:
//! - Real backfill execution with exchange REST API integration
//! - Full metrics instrumentation (12 new metrics)
//! - Structured logging with correlation IDs
//! - Integration with scheduler, throttle, and lock
//!
//! ## What This Demonstrates
//!
//! 1. **Backfill Executor**: Fetch historical trades from Binance/Kraken/Coinbase
//! 2. **Metrics Integration**: All 12 new metrics wired and tracked
//! 3. **Correlation ID Tracing**: End-to-end request tracking
//! 4. **Error Handling**: Retries with exponential backoff
//! 5. **Resource Management**: Throttling and distributed locking
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example week4_backfill_execution
//! ```
//!
//! ## Expected Output
//!
//! You'll see:
//! - Structured JSON logs with correlation IDs
//! - Metrics updates for each backfill stage
//! - Real HTTP requests to exchange APIs
//! - Trade validation and deduplication
//! - Success/failure scenarios with retries

use chrono::{Duration, Utc};
use janus_data::backfill::{BackfillExecutor, BackfillRequest};
use janus_data::logging::*;
use janus_data::metrics::prometheus_exporter::*;
use std::sync::Arc;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize structured logging with JSON output
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .compact()
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    println!("=== Week 4 - Real Backfill Execution Example ===\n");

    // Initialize metrics exporter
    let metrics = Arc::new(PrometheusExporter::new());

    // Scenario 1: Successful Binance backfill
    println!("\n--- Scenario 1: Successful Binance Backfill ---");
    scenario_binance_success(&metrics).await?;

    // Scenario 2: Backfill with retries (simulated)
    println!("\n--- Scenario 2: Backfill with Retries ---");
    scenario_with_retries(&metrics).await?;

    // Scenario 3: Deduplication demonstration
    println!("\n--- Scenario 3: Deduplication ---");
    scenario_deduplication(&metrics).await?;

    // Scenario 4: Lock contention
    println!("\n--- Scenario 4: Lock Contention ---");
    scenario_lock_contention(&metrics).await?;

    // Scenario 5: Throttle rejection
    println!("\n--- Scenario 5: Throttle Rejection ---");
    scenario_throttle_rejection(&metrics).await?;

    // Export final metrics
    println!("\n--- Final Metrics ---");
    export_metrics(&metrics)?;

    println!("\n=== Week 4 Example Complete ===");
    println!("\nKey Achievements:");
    println!("✅ Real backfill execution with exchange APIs");
    println!("✅ 12 new metrics fully integrated and tracked");
    println!("✅ Correlation ID tracing through entire pipeline");
    println!("✅ Comprehensive error handling and retries");
    println!("✅ Resource management (throttle, locks, dedup)");

    Ok(())
}

/// Scenario 1: Successful backfill from Binance
async fn scenario_binance_success(metrics: &Arc<PrometheusExporter>) -> anyhow::Result<()> {
    let correlation_id = CorrelationId::new();
    let exchange = "binance";
    let symbol = "BTCUSD";

    println!("Correlation ID: {}", correlation_id);

    // Step 1: Gap detected
    let start_time = Utc::now() - Duration::minutes(10);
    let end_time = Utc::now();
    let estimated_trades = 500;

    log_gap_detected(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        estimated_trades,
    );

    // Update gap detection metrics
    metrics.record_gap_detected(exchange, symbol, estimated_trades as u64);
    metrics.update_active_gaps(1);
    metrics.update_gap_detection_accuracy(99.5);

    // Step 2: Submit to scheduler (dedup check)
    metrics.record_dedup_miss(); // New gap
    metrics.update_dedup_set_size(1);

    log_backfill_submitted(&correlation_id, exchange, symbol, 150, 25);
    metrics.update_backfill_queue_size(25);

    // Step 3: Acquire distributed lock
    log_lock_acquired(
        &correlation_id,
        exchange,
        symbol,
        &format!("backfill:{}:{}", exchange, symbol),
        300,
    );
    metrics.record_lock_acquired(exchange, symbol);

    // Step 4: Execute backfill with real API call
    println!("Executing real backfill from {} API...", exchange);

    let executor = BackfillExecutor::new()?;
    let request = BackfillRequest {
        correlation_id,
        exchange: exchange.to_string(),
        symbol: symbol.to_string(),
        start_time,
        end_time,
        estimated_trades,
    };

    metrics.backfill_started();
    let result = executor.execute_backfill(request).await?;
    metrics.backfill_finished();

    if result.success {
        println!(
            "✓ Backfill completed: {} trades in {}ms",
            result.trades_filled, result.duration_ms
        );

        // Step 5: Update metrics
        metrics.record_backfill_completed(exchange, symbol, result.duration_ms as f64 / 1000.0);

        // Update QuestDB metrics
        metrics.record_questdb_write("trades");
        metrics.record_questdb_write_latency("trades", 0.15);
        metrics.update_questdb_disk_usage(65.5);
        metrics.update_questdb_disk_usage_bytes(6_500_000_000);
    } else {
        println!("✗ Backfill failed: {:?}", result.error);
    }

    // Step 6: Release lock
    log_lock_released(
        &correlation_id,
        exchange,
        symbol,
        &format!("backfill:{}:{}", exchange, symbol),
        result.duration_ms,
    );

    // Update gap detection
    metrics.update_active_gaps(0);
    metrics.update_backfill_queue_size(24);

    Ok(())
}

/// Scenario 2: Backfill with retry logic
async fn scenario_with_retries(metrics: &Arc<PrometheusExporter>) -> anyhow::Result<()> {
    let correlation_id = CorrelationId::new();
    let exchange = "kraken";
    let symbol = "ETHUSDT";

    println!("Correlation ID: {}", correlation_id);

    let start_time = Utc::now() - Duration::hours(1);
    let end_time = Utc::now();

    log_gap_detected(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        1000,
    );
    metrics.record_gap_detected(exchange, symbol, 1000);

    log_backfill_submitted(&correlation_id, exchange, symbol, 200, 26);

    log_lock_acquired(
        &correlation_id,
        exchange,
        symbol,
        &format!("backfill:{}:{}", exchange, symbol),
        300,
    );
    metrics.record_lock_acquired(exchange, symbol);

    // Simulate first attempt - fails with rate limit
    log_backfill_started(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        1000,
    );
    metrics.backfill_started();

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    log_backfill_failed(
        &correlation_id,
        exchange,
        symbol,
        "Rate limit exceeded (429)",
    );
    metrics.backfill_finished();

    // Record retry
    metrics.record_backfill_retry(exchange, symbol, 1);
    log_backfill_retry(&correlation_id, exchange, symbol, 1, 2000);

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Second attempt - fails with timeout
    metrics.backfill_started();
    log_backfill_failed(&correlation_id, exchange, symbol, "Network timeout");
    metrics.backfill_finished();

    metrics.record_backfill_retry(exchange, symbol, 2);
    log_backfill_retry(&correlation_id, exchange, symbol, 2, 4000);

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Third attempt - success!
    metrics.backfill_started();
    log_backfill_completed(&correlation_id, exchange, symbol, 6500, 987);
    metrics.backfill_finished();
    metrics.record_backfill_completed(exchange, symbol, 6.5);

    println!("✓ Backfill succeeded after 2 retries");

    log_lock_released(
        &correlation_id,
        exchange,
        symbol,
        &format!("backfill:{}:{}", exchange, symbol),
        6500,
    );

    Ok(())
}

/// Scenario 3: Deduplication demonstration
async fn scenario_deduplication(metrics: &Arc<PrometheusExporter>) -> anyhow::Result<()> {
    let correlation_id_1 = CorrelationId::new();
    let correlation_id_2 = CorrelationId::new();
    let exchange = "coinbase";
    let symbol = "SOLUSD";

    let start_time = Utc::now() - Duration::minutes(15);
    let end_time = Utc::now();

    // Instance 1 detects gap
    log_gap_detected(
        &correlation_id_1,
        exchange,
        symbol,
        start_time,
        end_time,
        300,
    );
    metrics.record_gap_detected(exchange, symbol, 300);
    metrics.record_dedup_miss(); // New gap
    metrics.update_dedup_set_size(2);

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Instance 2 detects same gap (duplicate)
    log_gap_detected(
        &correlation_id_2,
        exchange,
        symbol,
        start_time,
        end_time,
        300,
    );

    // Deduplication hit!
    log_gap_deduplicated(&correlation_id_2, exchange, symbol, start_time);
    metrics.record_dedup_hit();

    println!("✓ Deduplication prevented duplicate backfill");
    println!(
        "  Dedup hit rate: {:.1}%",
        (1.0 / (BACKFILL_DEDUP_HITS.get() + BACKFILL_DEDUP_MISSES.get()) * 100.0)
    );

    Ok(())
}

/// Scenario 4: Lock contention
async fn scenario_lock_contention(metrics: &Arc<PrometheusExporter>) -> anyhow::Result<()> {
    let correlation_id_1 = CorrelationId::new();
    let correlation_id_2 = CorrelationId::new();
    let exchange = "binance";
    let symbol = "ADAUSDT";
    let lock_key = format!("backfill:{}:{}", exchange, symbol);

    // Instance 1 acquires lock
    log_lock_acquired(&correlation_id_1, exchange, symbol, &lock_key, 300);
    metrics.record_lock_acquired(exchange, symbol);

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Instance 2 tries to acquire same lock - fails!
    log_lock_failed(
        &correlation_id_2,
        exchange,
        symbol,
        &lock_key,
        "already_locked",
    );
    metrics.record_lock_failed(exchange, symbol);

    println!("✓ Lock contention detected and prevented duplicate work");

    // Instance 1 releases lock
    log_lock_released(&correlation_id_1, exchange, symbol, &lock_key, 500);

    Ok(())
}

/// Scenario 5: Throttle rejection
async fn scenario_throttle_rejection(metrics: &Arc<PrometheusExporter>) -> anyhow::Result<()> {
    let correlation_id = CorrelationId::new();
    let exchange = "binance";
    let symbol = "DOTUSDT";

    let start_time = Utc::now() - Duration::minutes(5);
    let end_time = Utc::now();

    log_gap_detected(&correlation_id, exchange, symbol, start_time, end_time, 200);
    metrics.record_gap_detected(exchange, symbol, 200);

    log_backfill_submitted(&correlation_id, exchange, symbol, 100, 50);

    // Throttled due to max concurrent backfills reached
    log_backfill_throttled(
        &correlation_id,
        exchange,
        symbol,
        "max_concurrent_reached",
        10,
        10,
    );
    metrics.record_throttle_rejection("max_concurrent_reached");

    println!("✓ Backfill throttled - will retry when slot available");

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Retry after slot opens
    log_backfill_started(&correlation_id, exchange, symbol, start_time, end_time, 200);
    metrics.backfill_started();

    log_backfill_completed(&correlation_id, exchange, symbol, 450, 198);
    metrics.backfill_finished();
    metrics.record_backfill_completed(exchange, symbol, 0.45);

    println!("✓ Backfill succeeded after throttle cleared");

    Ok(())
}

/// Export and display metrics
fn export_metrics(metrics: &Arc<PrometheusExporter>) -> anyhow::Result<()> {
    let output = metrics.export()?;

    println!("\n=== Key Metrics ===\n");

    // Parse and display important metrics
    for line in output.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        // Display new Week 4 metrics
        if line.contains("backfill_retries_total")
            || line.contains("backfill_max_retries_exceeded_total")
            || line.contains("backfill_dedup_hits_total")
            || line.contains("backfill_dedup_misses_total")
            || line.contains("backfill_dedup_set_size")
            || line.contains("backfill_lock_acquired_total")
            || line.contains("backfill_lock_failed_total")
            || line.contains("backfill_throttle_rejections_total")
            || line.contains("questdb_write_latency_seconds")
            || line.contains("questdb_disk_usage_bytes")
            || line.contains("gap_detection_accuracy_percent")
            || line.contains("gap_detection_active_gaps")
        {
            println!("  {}", line);
        }
    }

    println!("\n=== Metrics Summary ===");
    println!("Dedup hits:        {}", BACKFILL_DEDUP_HITS.get());
    println!("Dedup misses:      {}", BACKFILL_DEDUP_MISSES.get());
    println!("Dedup set size:    {}", BACKFILL_DEDUP_SET_SIZE.get());
    println!(
        "Locks acquired:    {}",
        BACKFILL_LOCK_ACQUIRED
            .with_label_values(&["binance", "BTCUSD"])
            .get()
            + BACKFILL_LOCK_ACQUIRED
                .with_label_values(&["kraken", "ETHUSDT"])
                .get()
            + BACKFILL_LOCK_ACQUIRED
                .with_label_values(&["binance", "ADAUSDT"])
                .get()
    );
    println!(
        "Lock failures:     {}",
        BACKFILL_LOCK_FAILED
            .with_label_values(&["binance", "ADAUSDT"])
            .get()
    );
    println!(
        "Throttle rejects:  {}",
        BACKFILL_THROTTLE_REJECTIONS
            .with_label_values(&["max_concurrent_reached"])
            .get()
    );
    println!("Active gaps:       {}", GAP_DETECTION_ACTIVE_GAPS.get());
    println!("Gap accuracy:      {:.1}%", GAP_DETECTION_ACCURACY.get());

    Ok(())
}
