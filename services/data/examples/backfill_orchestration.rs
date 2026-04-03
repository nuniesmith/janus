//! Backfill Orchestration Example
//!
//! This example demonstrates the complete backfill orchestration system:
//! - Gap detection integration
//! - Priority-based scheduling
//! - Distributed locking
//! - Resource throttling
//! - Automatic retry with exponential backoff
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example backfill_orchestration
//! ```
//!
//! ## Prerequisites
//!
//! - Redis running on localhost:6379
//! - QuestDB running on localhost:9009 (optional, for real backfills)

use anyhow::Result;
use chrono::{Duration, Utc};
use janus_data::backfill::{
    BackfillLock, BackfillScheduler, BackfillThrottle, GapInfo, GapIntegrationConfig,
    GapIntegrationManager, LockConfig, LockMetrics, SchedulerConfig, ThrottleConfig,
};
use prometheus::Registry;
use std::sync::Arc;
use tokio::time::sleep;
use tracing::{Level, info};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("=== Backfill Orchestration Example ===\n");

    // ========================================================================
    // 1. Setup Infrastructure
    // ========================================================================
    info!("📦 Setting up infrastructure components...");

    // Redis client for distributed locking
    let redis_client = redis::Client::open("redis://127.0.0.1:6379")?;
    info!("   ✅ Connected to Redis");

    // Prometheus registry for metrics
    let registry = Registry::new();
    info!("   ✅ Prometheus registry created");

    // ========================================================================
    // 2. Initialize Backfill Components
    // ========================================================================
    info!("\n🔧 Initializing backfill components...");

    // Throttle (resource management)
    let throttle_config = ThrottleConfig {
        max_concurrent: 2,                    // Max 2 concurrent backfills
        max_disk_usage: 0.90,                 // Stop at 90% disk
        max_ooo_rows: 2_000_000,              // QuestDB OOO limit
        questdb_data_dir: "/tmp".to_string(), // For example purposes
        ..Default::default()
    };
    let throttle = Arc::new(BackfillThrottle::new(throttle_config));
    info!("   ✅ Throttle configured (max 2 concurrent)");

    // Distributed lock (coordination)
    let lock_config = LockConfig::default();
    let lock_metrics = Arc::new(LockMetrics::new(&registry)?);
    let lock = Arc::new(BackfillLock::new(redis_client, lock_config, lock_metrics));
    info!("   ✅ Distributed lock initialized");

    // Scheduler (priority queue)
    let scheduler_config = SchedulerConfig {
        poll_interval_ms: 500,       // Check queue every 500ms
        max_retries: 3,              // Retry up to 3 times
        initial_retry_delay_secs: 5, // Start with 5s delay
        max_retry_delay_secs: 60,    // Max 60s delay
        backoff_multiplier: 2.0,     // Double delay each retry
        ..Default::default()
    };
    let scheduler = Arc::new(BackfillScheduler::new(
        scheduler_config,
        Arc::clone(&throttle),
        Arc::clone(&lock),
    ));
    info!("   ✅ Scheduler initialized");

    // Gap integration (automatic gap submission)
    let gap_config = GapIntegrationConfig {
        min_gap_duration_secs: 10, // Ignore gaps < 10 seconds
        min_gap_trades: 10,        // Ignore gaps < 10 trades
        max_gap_trades: 100_000,   // Don't backfill > 100K trades
        auto_submit: true,         // Auto-submit to scheduler
        ..Default::default()
    };
    let gap_manager = Arc::new(GapIntegrationManager::new(
        gap_config,
        Arc::clone(&scheduler),
    ));
    info!("   ✅ Gap integration manager created");

    // ========================================================================
    // 3. Simulate Gap Detection
    // ========================================================================
    info!("\n🔍 Simulating gap detection...");

    let now = Utc::now();

    // Simulate various types of gaps being detected
    let gaps = vec![
        // Critical: Large recent gap on Binance BTC
        (
            "binance",
            "BTCUSD",
            now - Duration::minutes(30),
            now - Duration::minutes(20),
            5000_u64,
            "Critical - Recent large gap",
        ),
        // High: Older gap on Binance ETH
        (
            "binance",
            "ETHUSDT",
            now - Duration::hours(2),
            now - Duration::hours(1),
            3000_u64,
            "High - Older gap",
        ),
        // Medium: Moderate gap on Bybit
        (
            "bybit",
            "BTCUSD",
            now - Duration::minutes(45),
            now - Duration::minutes(35),
            1000_u64,
            "Medium - Bybit gap",
        ),
        // Low: Small gap on KuCoin
        (
            "kucoin",
            "BTCUSD",
            now - Duration::minutes(15),
            now - Duration::minutes(10),
            200_u64,
            "Low - Small KuCoin gap",
        ),
        // Filtered: Too small (should be filtered)
        (
            "binance",
            "DOGEUSDT",
            now - Duration::seconds(5),
            now,
            5_u64,
            "Should be filtered - too small",
        ),
    ];

    info!("   Submitting {} detected gaps...", gaps.len());
    for (exchange, symbol, start, end, trades, description) in gaps {
        info!(
            "   📊 {}: {} {} ({} trades)",
            description, exchange, symbol, trades
        );

        gap_manager
            .handle_gap(exchange.to_string(), symbol.to_string(), start, end, trades)
            .await;
    }

    // Give a moment for processing
    sleep(std::time::Duration::from_millis(100)).await;

    // ========================================================================
    // 4. Display Queue Status
    // ========================================================================
    info!("\n📊 Queue Status:");
    let stats = scheduler.stats().await;
    info!("   Total queued: {}", stats.total_queued);
    info!("   Ready to process: {}", stats.ready_count);
    info!("   Waiting for retry: {}", stats.waiting_count);

    let integration_stats = gap_manager.get_stats().await;
    info!("\n📈 Gap Integration Statistics:");
    info!("   Total detected: {}", integration_stats.total_detected);
    info!("   Submitted: {}", integration_stats.submitted);
    info!(
        "   Filtered (too small): {}",
        integration_stats.filtered_too_small
    );
    info!(
        "   Filtered (too large): {}",
        integration_stats.filtered_too_large
    );
    info!("   Duplicates: {}", integration_stats.duplicates);
    info!(
        "   Submission rate: {:.1}%",
        integration_stats.submission_rate() * 100.0
    );

    // ========================================================================
    // 5. Process Backfills (Simulation)
    // ========================================================================
    info!("\n⚙️  Processing backfills (simulated)...");
    info!("   (In production, scheduler.run().await would process the queue)");
    info!("   (For this example, we'll just show what would happen)\n");

    // Demonstrate manual gap submission
    info!("📤 Demonstrating manual gap submission:");
    let manual_gap = GapInfo {
        exchange: "binance".to_string(),
        symbol: "ADAUSDT".to_string(),
        start_time: now - Duration::minutes(20),
        end_time: now - Duration::minutes(10),
        estimated_trades: 800,
    };
    scheduler.submit_gap(manual_gap).await;
    info!("   ✅ Manually submitted ADA gap");

    let final_queue_size = scheduler.queue_size().await;
    info!("   Final queue size: {}", final_queue_size);

    // ========================================================================
    // 6. Demonstrate Priority Ordering
    // ========================================================================
    info!("\n🎯 Priority Ordering:");
    info!("   Gaps are processed in priority order based on:");
    info!("   - Age (older gaps = higher priority)");
    info!("   - Size (larger gaps = higher priority)");
    info!("   - Exchange criticality (Binance > Bybit > KuCoin)");
    info!("\n   Expected processing order:");
    info!("   1. Binance ETHUSDT (oldest, high priority exchange)");
    info!("   2. Binance BTCUSD (recent but large)");
    info!("   3. Bybit BTCUSD (moderate age/size)");
    info!("   4. Binance ADAUSDT (recent manual submission)");
    info!("   5. KuCoin BTCUSD (small gap, low priority exchange)");

    // ========================================================================
    // 7. Demonstrate Resource Management
    // ========================================================================
    info!("\n🛡️  Resource Management:");
    info!("   ✅ Max 2 concurrent backfills enforced by throttle");
    info!("   ✅ Distributed locking prevents duplicate work");
    info!("   ✅ Disk usage monitoring (stops at 90%)");
    info!("   ✅ QuestDB OOO protection (max 2M rows)");
    info!("   ✅ Automatic retry with exponential backoff");

    // ========================================================================
    // 8. Cleanup and Summary
    // ========================================================================
    info!("\n✅ Example Complete!");
    info!("\n📚 Key Takeaways:");
    info!("   1. Gap detection automatically feeds the scheduler");
    info!("   2. Priority queue ensures critical gaps are processed first");
    info!("   3. Throttling prevents resource exhaustion");
    info!("   4. Distributed locking prevents duplicate backfills");
    info!("   5. Automatic retry handles transient failures");
    info!("\n🚀 Production Deployment:");
    info!("   - Run scheduler.run().await in a dedicated task");
    info!("   - Connect gap detection system to gap_manager");
    info!("   - Monitor via Prometheus metrics");
    info!("   - Configure alerts for queue depth and failures");

    Ok(())
}
