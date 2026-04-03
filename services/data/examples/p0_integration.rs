//! P0 Integration Example
//!
//! Demonstrates how all P0 critical items work together in production:
//! 1. API Key Security (Docker Secrets)
//! 2. Backfill Locking (Distributed locks)
//! 3. Circuit Breaker (Rate limit protection)
//! 4. Backfill Throttling (Resource management)
//! 5. Prometheus Metrics (Observability)
//!
//! ## Running this example
//!
//! ```bash
//! # Start Redis (for backfill locking)
//! docker run -d -p 6379:6379 redis:latest
//!
//! # Run the example
//! cargo run --example p0_integration
//! ```
//!
//! ## What this demonstrates
//!
//! - Loading API keys from Docker Secrets files (or env vars in dev)
//! - Using circuit breaker to wrap API calls
//! - Distributed locking for backfill operations
//! - Throttling backfills with resource checks
//! - Exporting Prometheus metrics
//! - End-to-end production-ready workflow

use anyhow::Result;
use janus_data::backfill::{
    BackfillLock, BackfillThrottle, LockConfig, LockMetrics, ThrottleConfig,
};
use janus_data::config::ExchangeCredentials;
use janus_data::metrics::prometheus_exporter::PrometheusExporter;
use janus_rate_limiter::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use prometheus::Registry;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("p0_integration=debug,fks_ruby=debug,janus_rate_limiter=debug")
        .init();

    info!("=== P0 Integration Example ===");
    info!("Demonstrating all P0 critical items working together\n");

    // ========================================================================
    // P0 Item 1: API Key Security
    // ========================================================================
    info!("1️⃣  API Key Security (Docker Secrets)");

    // In production: keys loaded from /run/secrets/*
    // In development: fallback to environment variables
    let credentials = load_credentials_securely()?;
    info!("   ✅ API keys loaded securely");
    info!(
        "   📝 Binance: {}",
        if credentials.binance.is_some() {
            "Configured"
        } else {
            "Not configured"
        }
    );
    info!(
        "   📝 Bybit: {}\n",
        if credentials.bybit.is_some() {
            "Configured"
        } else {
            "Not configured"
        }
    );

    // ========================================================================
    // P0 Item 3: Circuit Breaker
    // ========================================================================
    info!("3️⃣  Circuit Breaker (Rate Limit Protection)");

    let circuit_config = CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout: Duration::from_secs(5),
    };
    let circuit_breaker = CircuitBreaker::new(circuit_config);
    info!("   ✅ Circuit breaker initialized");
    info!("   📝 Failure threshold: 3 consecutive 429s");
    info!("   📝 Recovery timeout: 5 seconds");

    // Simulate some API calls
    info!("   🔄 Simulating API calls...");
    for i in 1..=5 {
        let result = circuit_breaker
            .call(|| async {
                sleep(Duration::from_millis(10)).await;
                if i <= 3 {
                    // Simulate rate limit errors
                    Err(anyhow::anyhow!("429 Rate limit exceeded"))
                } else {
                    // Simulate success
                    Ok(format!("API response {}", i))
                }
            })
            .await;

        match result {
            Ok(response) => info!("   ✅ Call {}: {}", i, response),
            Err(e) => warn!("   ⚠️  Call {}: {}", i, e),
        }
    }

    info!("   📊 Circuit state: {:?}\n", circuit_breaker.state());

    // ========================================================================
    // P0 Item 2: Backfill Locking
    // ========================================================================
    info!("2️⃣  Distributed Backfill Locking");

    // Connect to Redis
    let redis_client = redis::Client::open("redis://127.0.0.1:6379")?;
    let lock_config = LockConfig::default();
    let registry = Registry::new();
    let lock_metrics = Arc::new(LockMetrics::new(&registry)?);
    let backfill_lock = BackfillLock::new(redis_client.clone(), lock_config, lock_metrics);

    info!("   ✅ Backfill lock initialized");
    info!("   📝 Lock TTL: 300 seconds");
    info!("   📝 Key prefix: backfill:");

    // Simulate concurrent backfill attempts
    info!("   🔄 Simulating concurrent backfill attempts...");

    let gap_id = "binance:BTCUSD:2025-12-30:12:00:00";

    // First attempt - should succeed
    match backfill_lock.acquire(gap_id).await? {
        Some(guard) => {
            info!("   ✅ Lock acquired: {}", guard.lock_id());

            // Simulate backfill work
            sleep(Duration::from_millis(100)).await;

            // Lock auto-releases on drop
            drop(guard);
            info!("   ✅ Lock released");
        }
        None => {
            warn!("   ⚠️  Lock already held by another instance");
        }
    }

    info!("");

    // ========================================================================
    // P0 Item 4: Backfill Throttling
    // ========================================================================
    info!("4️⃣  Backfill Throttling & Resource Management");

    let throttle_config = ThrottleConfig {
        max_concurrent: 2,
        max_disk_usage: 0.90,
        alert_disk_usage: 0.80,
        max_ooo_rows: 100_000, // Lower for demo
        batch_size: 10_000,
        questdb_data_dir: "/tmp".to_string(), // Use /tmp for demo
    };
    let throttle = Arc::new(BackfillThrottle::new(throttle_config));

    info!("   ✅ Backfill throttle initialized");
    info!("   📝 Max concurrent: 2 backfills");
    info!("   📝 Max disk usage: 90%");
    info!("   📝 Alert threshold: 80%");

    // Start disk monitor
    let monitor_handle = throttle.clone().start_disk_monitor();
    info!("   ✅ Disk monitor started");

    // Simulate multiple backfill requests
    info!("   🔄 Simulating backfill requests...");

    let mut handles = vec![];
    for i in 1..=4 {
        let throttle = throttle.clone();
        let exporter = PrometheusExporter::new();

        let handle = tokio::spawn(async move {
            let gap_size = 5_000;
            let start = Instant::now();

            match throttle
                .execute_backfill(gap_size, || async {
                    info!("      🔄 Backfill {} executing...", i);
                    sleep(Duration::from_millis(200)).await;
                    exporter.record_backfill_completed("binance", "BTCUSD", 0.2);
                    Ok::<_, anyhow::Error>(())
                })
                .await
            {
                Ok(_) => {
                    info!("      ✅ Backfill {} completed in {:?}", i, start.elapsed());
                }
                Err(e) => {
                    warn!("      ⚠️  Backfill {} failed: {}", i, e);
                }
            }
        });

        handles.push(handle);
        sleep(Duration::from_millis(50)).await; // Stagger starts
    }

    // Wait for all backfills
    for handle in handles {
        let _ = handle.await;
    }

    info!("   📊 Throttle stats:");
    info!("      Running: {}", throttle.running_count());
    info!("      Available slots: {}\n", throttle.available_slots());

    // Stop disk monitor
    monitor_handle.abort();

    // ========================================================================
    // P0 Item 5: Prometheus Metrics
    // ========================================================================
    info!("5️⃣  Prometheus Metrics Export");

    let exporter = PrometheusExporter::new();
    info!("   ✅ Prometheus exporter initialized");

    // Record various metrics
    exporter.record_trade_ingested("binance", "BTCUSD", 125.5);
    exporter.record_trade_ingested("binance", "ETHUSDT", 89.3);
    exporter.record_gap_detected("binance", "BTCUSD", 42);
    exporter.update_data_completeness(99.95);
    exporter.update_websocket_status("binance", true);
    exporter.update_circuit_breaker_state("binance", circuit_breaker.state() as u8 as i64);
    exporter.record_rate_limit_request("binance", true);

    info!("   ✅ Metrics recorded:");
    info!("      - 2 trades ingested");
    info!("      - 1 gap detected (42 trades)");
    info!("      - Data completeness: 99.95%");
    info!(
        "      - Circuit breaker state: {:?}",
        circuit_breaker.state()
    );

    // Export metrics
    let metrics_output = exporter.export()?;

    info!("\n   📊 Prometheus Metrics Export:");
    info!("   ════════════════════════════════════════════");

    // Show key metrics
    for line in metrics_output.lines() {
        if line.starts_with("# HELP") || line.starts_with("# TYPE") {
            continue;
        }
        if !line.trim().is_empty() && !line.starts_with('#') {
            info!("   {}", line);
        }
    }
    info!("   ════════════════════════════════════════════\n");

    // ========================================================================
    // Summary
    // ========================================================================
    info!("✅ P0 Integration Example Complete!");
    info!("");
    info!("Summary of P0 Items:");
    info!("  1️⃣  API Key Security: ✅ Keys loaded securely");
    info!("  2️⃣  Backfill Locking: ✅ Distributed locks working");
    info!("  3️⃣  Circuit Breaker: ✅ Rate limit protection active");
    info!("  4️⃣  Backfill Throttling: ✅ Resource management enabled");
    info!("  5️⃣  Prometheus Metrics: ✅ All SLIs exported");
    info!("");
    info!("Production Readiness: 🟢 71% (5/7 P0 items complete)");
    info!("Remaining: Grafana Dashboards + Alertmanager Rules");

    Ok(())
}

/// Load API credentials securely from Docker Secrets or environment variables
fn load_credentials_securely() -> Result<ExchangeCredentials> {
    // This simulates the secure credential loading from config.rs
    // In production, this reads from /run/secrets/*
    // In development, falls back to environment variables

    Ok(ExchangeCredentials {
        binance: Some(janus_data::config::ApiKeyPair {
            api_key: "***REDACTED***".to_string(),
            api_secret: "***REDACTED***".to_string(),
        }),
        bybit: Some(janus_data::config::ApiKeyPair {
            api_key: "***REDACTED***".to_string(),
            api_secret: "***REDACTED***".to_string(),
        }),
        kucoin: None,
        alphavantage: None,
        coinmarketcap: None,
    })
}
