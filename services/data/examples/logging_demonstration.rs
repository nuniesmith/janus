//! Logging Demonstration with Correlation IDs
//!
//! This example demonstrates the structured logging module with correlation ID
//! support for end-to-end request tracing through the backfill orchestration pipeline.
//!
//! ## What This Demonstrates
//!
//! 1. **Correlation ID Creation**: Generate unique IDs for request tracking
//! 2. **End-to-End Tracing**: Follow a gap from detection → scheduler → backfill → QuestDB
//! 3. **Structured Logging**: JSON-formatted logs with consistent fields
//! 4. **Error Scenarios**: Logging failures, retries, and throttling
//! 5. **Log Aggregation**: How logs can be searched by correlation_id
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example logging_demonstration
//! ```
//!
//! ## Expected Output
//!
//! You should see JSON-formatted log entries that can be:
//! - Parsed by log aggregation systems (Loki, ELK)
//! - Searched by correlation_id
//! - Filtered by component, exchange, symbol
//! - Analyzed for performance metrics

use chrono::{Duration, Utc};
use janus_data::logging::*;
use std::thread;
use std::time::Duration as StdDuration;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() {
    // Initialize tracing with JSON formatting
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .compact()
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    println!("=== Logging Demonstration with Correlation IDs ===\n");

    // Scenario 1: Successful Gap Detection and Backfill
    println!("\n--- Scenario 1: Successful Backfill ---");
    demonstrate_successful_backfill().await;

    // Scenario 2: Backfill with Retries
    println!("\n--- Scenario 2: Backfill with Retries ---");
    demonstrate_backfill_with_retries().await;

    // Scenario 3: Throttled Backfill
    println!("\n--- Scenario 3: Throttled Backfill ---");
    demonstrate_throttled_backfill().await;

    // Scenario 4: Lock Contention
    println!("\n--- Scenario 4: Lock Contention ---");
    demonstrate_lock_contention().await;

    // Scenario 5: Multiple Concurrent Backfills
    println!("\n--- Scenario 5: Multiple Concurrent Backfills ---");
    demonstrate_concurrent_backfills().await;

    println!("\n=== Demonstration Complete ===");
    println!("\nKey Takeaways:");
    println!("1. Each request has a unique correlation_id for end-to-end tracing");
    println!("2. Logs are JSON-formatted for machine parsing");
    println!("3. Consistent field names enable easy filtering and aggregation");
    println!("4. Errors include context for debugging");
    println!("5. Search logs by correlation_id to trace full request lifecycle");
}

/// Demonstrate successful gap detection → backfill → completion flow
async fn demonstrate_successful_backfill() {
    let correlation_id = CorrelationId::new();
    let exchange = "binance";
    let symbol = "BTCUSD";

    // Step 1: Gap Detection
    let start_time = Utc::now() - Duration::minutes(10);
    let end_time = Utc::now();
    let estimated_trades = 5000;

    log_gap_detected(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        estimated_trades,
    );

    thread::sleep(StdDuration::from_millis(100));

    // Step 2: Submit to Scheduler
    let priority = 150;
    let queue_size = 25;

    log_backfill_submitted(&correlation_id, exchange, symbol, priority, queue_size);

    thread::sleep(StdDuration::from_millis(100));

    // Step 3: Acquire Lock
    let lock_key = format!("backfill:{}:{}", exchange, symbol);
    let ttl_seconds = 300;

    log_lock_acquired(&correlation_id, exchange, symbol, &lock_key, ttl_seconds);

    thread::sleep(StdDuration::from_millis(100));

    // Step 4: Start Backfill
    log_backfill_started(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        estimated_trades,
    );

    thread::sleep(StdDuration::from_millis(500));

    // Step 5: Write to QuestDB
    let trades_count = 4987;
    let write_duration_ms = 250;

    log_questdb_write(
        &correlation_id,
        exchange,
        symbol,
        trades_count,
        write_duration_ms,
    );

    thread::sleep(StdDuration::from_millis(100));

    // Step 6: Complete Backfill
    let total_duration_ms = 850;

    log_backfill_completed(
        &correlation_id,
        exchange,
        symbol,
        total_duration_ms,
        trades_count,
    );

    thread::sleep(StdDuration::from_millis(100));

    // Step 7: Release Lock
    log_lock_released(
        &correlation_id,
        exchange,
        symbol,
        &lock_key,
        total_duration_ms,
    );
}

/// Demonstrate backfill with retries due to temporary failures
async fn demonstrate_backfill_with_retries() {
    let correlation_id = CorrelationId::new();
    let exchange = "kraken";
    let symbol = "ETHUSDT";

    // Gap detected
    let start_time = Utc::now() - Duration::hours(1);
    let end_time = Utc::now();
    let estimated_trades = 10000;

    log_gap_detected(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        estimated_trades,
    );

    thread::sleep(StdDuration::from_millis(100));

    log_backfill_submitted(&correlation_id, exchange, symbol, 200, 30);

    thread::sleep(StdDuration::from_millis(100));

    log_lock_acquired(
        &correlation_id,
        exchange,
        symbol,
        &format!("backfill:{}:{}", exchange, symbol),
        300,
    );

    thread::sleep(StdDuration::from_millis(100));

    log_backfill_started(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        estimated_trades,
    );

    thread::sleep(StdDuration::from_millis(200));

    // First failure: Rate limit
    log_backfill_failed(
        &correlation_id,
        exchange,
        symbol,
        "Rate limit exceeded (429)",
    );

    thread::sleep(StdDuration::from_millis(100));

    // Retry 1
    let backoff_ms = 2000;
    log_backfill_retry(&correlation_id, exchange, symbol, 1, backoff_ms);

    thread::sleep(StdDuration::from_millis(300));

    // Second attempt - also fails
    log_backfill_failed(&correlation_id, exchange, symbol, "Network timeout");

    thread::sleep(StdDuration::from_millis(100));

    // Retry 2 with exponential backoff
    let backoff_ms = 4000;
    log_backfill_retry(&correlation_id, exchange, symbol, 2, backoff_ms);

    thread::sleep(StdDuration::from_millis(300));

    // Third attempt - success!
    log_questdb_write(&correlation_id, exchange, symbol, 9876, 300);

    thread::sleep(StdDuration::from_millis(100));

    log_backfill_completed(&correlation_id, exchange, symbol, 7500, 9876);

    thread::sleep(StdDuration::from_millis(100));

    log_lock_released(
        &correlation_id,
        exchange,
        symbol,
        &format!("backfill:{}:{}", exchange, symbol),
        7500,
    );
}

/// Demonstrate throttled backfill due to resource constraints
async fn demonstrate_throttled_backfill() {
    let correlation_id = CorrelationId::new();
    let exchange = "coinbase";
    let symbol = "SOLUSD";

    let start_time = Utc::now() - Duration::minutes(5);
    let end_time = Utc::now();

    log_gap_detected(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        2000,
    );

    thread::sleep(StdDuration::from_millis(100));

    log_backfill_submitted(&correlation_id, exchange, symbol, 100, 45);

    thread::sleep(StdDuration::from_millis(100));

    // Throttled due to too many concurrent backfills
    log_backfill_throttled(
        &correlation_id,
        exchange,
        symbol,
        "max_concurrent_reached",
        10,
        10,
    );

    thread::sleep(StdDuration::from_millis(500));

    // Retry after slot opens up
    log_backfill_started(
        &correlation_id,
        exchange,
        symbol,
        start_time,
        end_time,
        2000,
    );

    thread::sleep(StdDuration::from_millis(300));

    log_questdb_write(&correlation_id, exchange, symbol, 1987, 150);

    thread::sleep(StdDuration::from_millis(100));

    log_backfill_completed(&correlation_id, exchange, symbol, 650, 1987);
}

/// Demonstrate lock contention when multiple instances try same gap
async fn demonstrate_lock_contention() {
    let correlation_id_1 = CorrelationId::new();
    let correlation_id_2 = CorrelationId::new();
    let exchange = "binance";
    let symbol = "ADAUSDT";
    let lock_key = format!("backfill:{}:{}", exchange, symbol);

    let start_time = Utc::now() - Duration::minutes(15);
    let end_time = Utc::now();

    // Instance 1 detects gap
    log_gap_detected(
        &correlation_id_1,
        exchange,
        symbol,
        start_time,
        end_time,
        3000,
    );

    thread::sleep(StdDuration::from_millis(50));

    // Instance 2 also detects same gap
    log_gap_detected(
        &correlation_id_2,
        exchange,
        symbol,
        start_time,
        end_time,
        3000,
    );

    thread::sleep(StdDuration::from_millis(100));

    // Instance 1 acquires lock
    log_lock_acquired(&correlation_id_1, exchange, symbol, &lock_key, 300);

    thread::sleep(StdDuration::from_millis(50));

    // Instance 2 fails to acquire lock (already held)
    log_lock_failed(
        &correlation_id_2,
        exchange,
        symbol,
        &lock_key,
        "already_locked",
    );

    thread::sleep(StdDuration::from_millis(100));

    // Instance 2 gap is deduplicated
    log_gap_deduplicated(&correlation_id_2, exchange, symbol, start_time);

    thread::sleep(StdDuration::from_millis(200));

    // Instance 1 continues with backfill
    log_backfill_started(
        &correlation_id_1,
        exchange,
        symbol,
        start_time,
        end_time,
        3000,
    );

    thread::sleep(StdDuration::from_millis(300));

    log_questdb_write(&correlation_id_1, exchange, symbol, 2998, 200);

    thread::sleep(StdDuration::from_millis(100));

    log_backfill_completed(&correlation_id_1, exchange, symbol, 650, 2998);

    thread::sleep(StdDuration::from_millis(100));

    log_lock_released(&correlation_id_1, exchange, symbol, &lock_key, 650);
}

/// Demonstrate multiple concurrent backfills for different symbols
async fn demonstrate_concurrent_backfills() {
    println!("Processing 3 gaps concurrently...");

    let gaps = vec![
        ("binance", "BTCUSD", 5000),
        ("kraken", "ETHUSDT", 8000),
        ("coinbase", "SOLUSD", 3000),
    ];

    let mut handles = vec![];

    for (exchange, symbol, estimated_trades) in gaps {
        let handle = tokio::spawn(async move {
            let correlation_id = CorrelationId::new();
            let start_time = Utc::now() - Duration::minutes(20);
            let end_time = Utc::now();

            log_gap_detected(
                &correlation_id,
                exchange,
                symbol,
                start_time,
                end_time,
                estimated_trades,
            );

            thread::sleep(StdDuration::from_millis(100));

            log_backfill_submitted(&correlation_id, exchange, symbol, 150, 35);

            thread::sleep(StdDuration::from_millis(100));

            log_lock_acquired(
                &correlation_id,
                exchange,
                symbol,
                &format!("backfill:{}:{}", exchange, symbol),
                300,
            );

            thread::sleep(StdDuration::from_millis(100));

            log_backfill_started(
                &correlation_id,
                exchange,
                symbol,
                start_time,
                end_time,
                estimated_trades,
            );

            thread::sleep(StdDuration::from_millis(400));

            log_questdb_write(&correlation_id, exchange, symbol, estimated_trades, 250);

            thread::sleep(StdDuration::from_millis(100));

            log_backfill_completed(&correlation_id, exchange, symbol, 850, estimated_trades);

            thread::sleep(StdDuration::from_millis(100));

            log_lock_released(
                &correlation_id,
                exchange,
                symbol,
                &format!("backfill:{}:{}", exchange, symbol),
                850,
            );
        });

        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        handle.await.unwrap();
    }

    println!("All concurrent backfills completed!");
}
