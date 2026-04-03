//! Integration Tests for Data Service
//!
//! These tests verify infrastructure integration with real services:
//! - QuestDB writes via ILP
//! - Redis distributed locks and dedup
//! - Circuit breaker integration
//! - Prometheus metrics
//!
//! Exchange-specific integration tests focus on Kraken (see `janus-execution`
//! integration tests) since we have API keys for Kraken in CI.
//!
//! ## Prerequisites
//!
//! Run the integration test environment:
//! ```bash
//! docker-compose -f docker-compose.integration.yml up -d
//! ```
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --test integration_test -- --test-threads=1
//! ```

use anyhow::Result;
use chrono::Utc;
use janus_data::actors::{TradeData, TradeSide};
use janus_data::storage::ilp::IlpWriter;
use redis::Commands;

// Test configuration
const QUESTDB_HOST: &str = "localhost";
const QUESTDB_ILP_PORT: u16 = 9009;
const REDIS_URL: &str = "redis://localhost:6379";

/// Helper to check if integration test environment is available
async fn is_integration_env_available() -> bool {
    // Try to connect to QuestDB
    let questdb_available =
        tokio::net::TcpStream::connect(format!("{}:{}", QUESTDB_HOST, QUESTDB_ILP_PORT))
            .await
            .is_ok();

    // Try to connect to Redis
    let redis_available = redis::Client::open(REDIS_URL)
        .ok()
        .and_then(|client| client.get_connection().ok())
        .is_some();

    questdb_available && redis_available
}

/// Create a test ILP writer connected to QuestDB
async fn create_test_ilp_writer() -> Result<IlpWriter> {
    IlpWriter::new(QUESTDB_HOST, QUESTDB_ILP_PORT, 1000, 100).await
}

/// Create a test Redis client
fn create_test_redis_client() -> Result<redis::Client> {
    Ok(redis::Client::open(REDIS_URL)?)
}

#[tokio::test]
#[ignore] // Run with: cargo test --test integration_test -- --ignored
async fn test_questdb_ilp_connection() -> Result<()> {
    if !is_integration_env_available().await {
        eprintln!("Skipping test: Integration environment not available");
        eprintln!("Run: docker-compose -f docker-compose.integration.yml up -d");
        return Ok(());
    }

    // Create ILP writer
    let mut writer = create_test_ilp_writer().await?;

    // Write a test trade
    let trade = TradeData {
        symbol: "BTC-USDT".to_string(),
        exchange: "test".to_string(),
        side: TradeSide::Buy,
        price: 50000.0,
        amount: 0.001,
        exchange_ts: Utc::now().timestamp_millis(),
        receipt_ts: Utc::now().timestamp_millis(),
        trade_id: "test-trade-1".to_string(),
    };

    writer.write_trade(&trade).await?;
    writer.flush().await?;

    // Check stats
    let stats = writer.stats();
    assert!(stats.lines_written > 0);
    assert_eq!(stats.flush_errors, 0);

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_questdb_batch_write() -> Result<()> {
    if !is_integration_env_available().await {
        eprintln!("Skipping test: Integration environment not available");
        return Ok(());
    }

    let mut writer = create_test_ilp_writer().await?;

    // Create a batch of trades
    let mut trades = Vec::new();
    let base_ts = Utc::now().timestamp_millis();

    for i in 0..100 {
        trades.push(TradeData {
            symbol: "ETH-USDT".to_string(),
            exchange: "test".to_string(),
            side: if i % 2 == 0 {
                TradeSide::Buy
            } else {
                TradeSide::Sell
            },
            price: 3000.0 + (i as f64 * 0.1),
            amount: 0.1,
            exchange_ts: base_ts + (i as i64 * 1000),
            receipt_ts: base_ts + (i as i64 * 1000) + 50,
            trade_id: format!("test-batch-{}", i),
        });
    }

    // Write batch
    let count = writer.write_trade_batch(&trades).await?;
    assert_eq!(count, 100);

    // Verify stats
    let stats = writer.stats();
    assert!(stats.lines_written >= 100);
    assert_eq!(stats.flush_errors, 0);

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_redis_distributed_lock() -> Result<()> {
    if !is_integration_env_available().await {
        eprintln!("Skipping test: Integration environment not available");
        return Ok(());
    }

    let client = create_test_redis_client()?;
    let mut conn = client.get_connection()?;

    // Acquire lock
    let lock_key = "test:lock:btc-usdt";
    let lock_value = "test-worker-1";
    let ttl_seconds = 60;

    // SET NX with expiry
    let acquired: bool = redis::cmd("SET")
        .arg(lock_key)
        .arg(lock_value)
        .arg("NX")
        .arg("EX")
        .arg(ttl_seconds)
        .query(&mut conn)?;

    assert!(acquired, "Should acquire lock on first attempt");

    // Try to acquire again (should fail)
    let acquired_again: bool = redis::cmd("SET")
        .arg(lock_key)
        .arg("test-worker-2")
        .arg("NX")
        .arg("EX")
        .arg(ttl_seconds)
        .query(&mut conn)?;

    assert!(!acquired_again, "Should not acquire lock when already held");

    // Release lock
    let _: () = conn.del(lock_key)?;

    // Should be able to acquire again
    let acquired_after_release: bool = redis::cmd("SET")
        .arg(lock_key)
        .arg("test-worker-3")
        .arg("NX")
        .arg("EX")
        .arg(ttl_seconds)
        .query(&mut conn)?;

    assert!(acquired_after_release, "Should acquire lock after release");

    // Cleanup
    let _: () = conn.del(lock_key)?;

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_redis_deduplication() -> Result<()> {
    if !is_integration_env_available().await {
        eprintln!("Skipping test: Integration environment not available");
        return Ok(());
    }

    let client = create_test_redis_client()?;
    let mut conn = client.get_connection()?;

    let dedup_key = "test:dedup:gaps";

    // Add items to dedup set
    let item1 = "binance:BTC-USDT:1672531200000";
    let item2 = "binance:ETH-USDT:1672531200000";

    let added1: i64 = conn.sadd(dedup_key, item1)?;
    assert_eq!(added1, 1, "First item should be added");

    let added2: i64 = conn.sadd(dedup_key, item2)?;
    assert_eq!(added2, 1, "Second item should be added");

    // Try to add duplicate
    let added_dup: i64 = conn.sadd(dedup_key, item1)?;
    assert_eq!(added_dup, 0, "Duplicate should not be added");

    // Check membership
    let is_member: bool = conn.sismember(dedup_key, item1)?;
    assert!(is_member, "Item should be in set");

    // Get set size
    let size: i64 = conn.scard(dedup_key)?;
    assert_eq!(size, 2, "Set should have 2 items");

    // Cleanup
    let _: () = conn.del(dedup_key)?;

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_circuit_breaker_integration() -> Result<()> {
    use janus_rate_limiter::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use std::time::Duration as StdDuration;

    let config = CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout: StdDuration::from_secs(1),
    };

    let breaker = CircuitBreaker::new(config);

    // Simulate failures to open circuit
    for _ in 0..3 {
        let _ = breaker
            .call(|| async { Err::<(), _>(anyhow::anyhow!("429 Rate limit exceeded")) })
            .await;
    }

    // Circuit should be open
    use janus_rate_limiter::circuit_breaker::CircuitState;
    assert_eq!(breaker.state(), CircuitState::Open);

    // Calls should fail fast
    let result = breaker.call(|| async { Ok::<(), anyhow::Error>(()) }).await;
    assert!(result.is_err());

    // Wait for timeout
    tokio::time::sleep(StdDuration::from_millis(1100)).await;

    // Should transition to HalfOpen and allow requests
    let _ = breaker.call(|| async { Ok::<(), anyhow::Error>(()) }).await;

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_concurrent_backfills_with_locks() -> Result<()> {
    if !is_integration_env_available().await {
        eprintln!("Skipping test: Integration environment not available");
        return Ok(());
    }

    // Test that concurrent backfills for the same symbol are properly serialized
    // via distributed locks

    let client = create_test_redis_client()?;
    let mut conn = client.get_connection()?;

    let lock_key = "backfill:lock:binance:BTCUSD";

    // Simulate two workers trying to acquire lock
    let worker1_acquired: bool = redis::cmd("SET")
        .arg(lock_key)
        .arg("worker-1")
        .arg("NX")
        .arg("EX")
        .arg(300)
        .query(&mut conn)?;

    assert!(worker1_acquired, "Worker 1 should acquire lock");

    // Worker 2 tries to acquire
    let worker2_acquired: bool = redis::cmd("SET")
        .arg(lock_key)
        .arg("worker-2")
        .arg("NX")
        .arg("EX")
        .arg(300)
        .query(&mut conn)?;

    assert!(
        !worker2_acquired,
        "Worker 2 should not acquire lock while worker 1 holds it"
    );

    // Worker 1 releases lock
    let _: () = conn.del(lock_key)?;

    // Now worker 2 can acquire
    let worker2_retry: bool = redis::cmd("SET")
        .arg(lock_key)
        .arg("worker-2")
        .arg("NX")
        .arg("EX")
        .arg(300)
        .query(&mut conn)?;

    assert!(worker2_retry, "Worker 2 should acquire lock after release");

    // Cleanup
    let _: () = conn.del(lock_key)?;

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_ilp_writer_buffer_management() -> Result<()> {
    if !is_integration_env_available().await {
        eprintln!("Skipping test: Integration environment not available");
        return Ok(());
    }

    let mut writer = create_test_ilp_writer().await?;

    // Write trades until buffer is full
    let base_ts = Utc::now().timestamp_millis();

    for i in 0..500 {
        let trade = TradeData {
            symbol: "TEST-USDT".to_string(),
            exchange: "test".to_string(),
            side: TradeSide::Buy,
            price: 1.0,
            amount: 1.0,
            exchange_ts: base_ts + i,
            receipt_ts: base_ts + i + 10,
            trade_id: format!("buffer-test-{}", i),
        };

        writer.write_trade(&trade).await?;

        // Check buffer state
        let buffer_state = writer.buffer_state();
        assert!(buffer_state.utilization_pct <= 100.0);
    }

    // Final flush
    writer.flush().await?;

    // Verify all trades were written
    let stats = writer.stats();
    assert!(stats.lines_written >= 500);
    assert_eq!(stats.flush_errors, 0);

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_metrics_integration() -> Result<()> {
    use janus_data::metrics::prometheus_exporter::PrometheusExporter;

    let exporter = PrometheusExporter::new();

    // Record various metrics
    exporter.record_trade_ingested("binance", "BTCUSD", 150.5);
    exporter.record_gap_detected("binance", "ETHUSDT", 100);
    exporter.update_data_completeness(99.95);
    exporter.backfill_started();
    exporter.record_backfill_completed("binance", "BTCUSD", 5.2);
    exporter.backfill_finished();
    exporter.record_questdb_write("trades");
    exporter.record_questdb_write_latency("trades", 0.05);

    // Export metrics
    let metrics_output = exporter.export().unwrap();

    // Verify metrics are present
    assert!(metrics_output.contains("data_completeness_percent"));
    assert!(metrics_output.contains("gaps_detected_total"));
    assert!(metrics_output.contains("trades_ingested_total"));
    assert!(metrics_output.contains("backfills_running"));
    assert!(metrics_output.contains("questdb_writes_total"));

    println!("Sample metrics output:\n{}", metrics_output);

    Ok(())
}
