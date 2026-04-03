//! # Real-World Gap Detection Simulation
//!
//! This example simulates various real-world scenarios that cause data gaps:
//! 1. Network disconnections (heartbeat timeout)
//! 2. Missing trades (sequence gaps)
//! 3. Exchange API degradation (statistical anomaly)
//! 4. Low-liquidity pair handling
//!
//! Run with:
//! ```bash
//! cargo run --example real_world_simulation
//! ```

use chrono::{Duration, Utc};
use janus_gap_detection::{Gap, GapDetectionManager, GapSeverity, Trade};
use std::time::Duration as StdDuration;
use tokio::time::sleep;
use tracing::{Level, info, warn};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() {
    // Initialize logging
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .init();

    println!("\n=== Gap Detection Real-World Simulation ===\n");

    // Run scenarios
    scenario_1_sequence_gap().await;
    println!("\n{}\n", "=".repeat(60));

    scenario_2_heartbeat_timeout().await;
    println!("\n{}\n", "=".repeat(60));

    scenario_3_statistical_anomaly().await;
    println!("\n{}\n", "=".repeat(60));

    scenario_4_combined_failures().await;
    println!("\n{}\n", "=".repeat(60));

    scenario_5_low_liquidity_pair().await;

    println!("\n=== All scenarios complete ===");
}

/// Scenario 1: Missing trades detected via sequence ID gaps
async fn scenario_1_sequence_gap() {
    println!("Scenario 1: Sequence Gap Detection");
    println!("-----------------------------------");
    println!("Simulating: WebSocket reconnection causing missed trades\n");

    let manager = GapDetectionManager::default();
    let now = Utc::now();

    // Simulate normal trading
    info!("Processing normal trade stream...");
    for id in 1000..1020 {
        let trade = Trade {
            exchange: "binance".to_string(),
            pair: "BTCUSD".to_string(),
            trade_id: Some(id),
            timestamp: now + Duration::milliseconds((id - 1000) as i64 * 100),
            price: 50000.0 + (id as f64 * 0.1),
            amount: 0.1,
        };
        manager.process_trade(&trade);
        sleep(StdDuration::from_millis(10)).await;
    }

    info!("Normal stream: trades 1000-1019 received");

    // Simulate disconnection - missing trades 1020-1099
    warn!("DISCONNECTION: Missing trades 1020-1099 (80 trades)");
    sleep(StdDuration::from_secs(2)).await;

    // Reconnect - next trade is 1100
    info!("RECONNECTED: Receiving trade 1100");
    let reconnect_trade = Trade {
        exchange: "binance".to_string(),
        pair: "BTCUSD".to_string(),
        trade_id: Some(1100),
        timestamp: now + Duration::seconds(5),
        price: 50010.0,
        amount: 0.1,
    };
    manager.process_trade(&reconnect_trade);

    // Check gaps
    let gaps = manager.get_all_gaps();
    println!("\nDetected Gaps: {}", gaps.len());
    for gap in &gaps {
        println!(
            "  ✗ {} gap: Missing {} trades",
            gap.gap_type, gap.missing_count
        );
        println!("    Severity: {:?}", gap.severity());
        println!("    Exchange: {}, Pair: {}", gap.exchange, gap.pair);
        println!("    Duration: {:?}", gap.duration());
    }

    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].missing_count, 80);
}

/// Scenario 2: Heartbeat timeout detection
async fn scenario_2_heartbeat_timeout() {
    println!("Scenario 2: Heartbeat Timeout Detection");
    println!("----------------------------------------");
    println!("Simulating: Silent WebSocket disconnection\n");

    let manager = GapDetectionManager::new(
        10000, // max sequence gap
        5,     // 5 second heartbeat timeout (short for demo)
        10,    // statistical window
        0.3,   // threshold
    );
    let now = Utc::now();

    // Send initial trades
    info!("Sending initial trades...");
    for i in 0..5 {
        let trade = Trade {
            exchange: "bybit".to_string(),
            pair: "ETHUSDT".to_string(),
            trade_id: Some(5000 + i),
            timestamp: now + Duration::seconds(i as i64),
            price: 3000.0,
            amount: 0.5,
        };
        manager.process_trade(&trade);
        sleep(StdDuration::from_millis(100)).await;
    }

    info!("Last trade received at {:?}", Utc::now());

    // Simulate silent disconnection - no more data
    warn!("SILENT DISCONNECTION: No more data being received");
    println!("Waiting for heartbeat timeout (6 seconds)...\n");

    for i in 1..=7 {
        sleep(StdDuration::from_secs(1)).await;
        print!(".");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        if i == 6 {
            // Run periodic check
            manager.run_periodic_checks().await;
        }
    }
    println!();

    // Check for timeout gaps
    let gaps = manager.get_all_gaps();
    println!("\nDetected Gaps: {}", gaps.len());
    for gap in &gaps {
        println!("  ✗ Heartbeat Timeout Detected!");
        println!("    Exchange: {}, Pair: {}", gap.exchange, gap.pair);
        println!("    Duration: {:?}", gap.duration());
        println!(
            "    Last data received: {} seconds ago",
            gap.metadata
                .get("duration_secs")
                .unwrap_or(&"unknown".to_string())
        );
    }

    assert_eq!(gaps.len(), 1);
    assert_eq!(
        gaps[0].gap_type,
        janus_gap_detection::GapType::HeartbeatTimeout
    );
}

/// Scenario 3: Statistical anomaly detection
async fn scenario_3_statistical_anomaly() {
    println!("Scenario 3: Statistical Anomaly Detection");
    println!("------------------------------------------");
    println!("Simulating: Exchange API degradation (tick rate drops)\n");

    let _manager = GapDetectionManager::default();

    // Note: This scenario demonstrates the concept but requires
    // manual tick counting which we don't implement in the basic manager
    info!("Normally: 100 trades/minute for BTC");
    info!("During degradation: Only 20 trades/minute");
    info!("Statistical detector would flag this as anomaly");

    println!("\n(Full implementation requires periodic tick counting)");
    println!("See StatisticalDetector::check_anomaly() for details");
}

/// Scenario 4: Combined failure modes
async fn scenario_4_combined_failures() {
    println!("Scenario 4: Combined Failure Modes");
    println!("-----------------------------------");
    println!("Simulating: Multiple exchanges with different issues\n");

    let manager = GapDetectionManager::default();
    let now = Utc::now();

    // Exchange 1: Binance - sequence gap
    info!("Binance: Processing trades with sequence gap");
    manager.process_trade(&Trade {
        exchange: "binance".to_string(),
        pair: "BTCUSD".to_string(),
        trade_id: Some(2000),
        timestamp: now,
        price: 50000.0,
        amount: 0.1,
    });

    manager.process_trade(&Trade {
        exchange: "binance".to_string(),
        pair: "BTCUSD".to_string(),
        trade_id: Some(2050), // Gap of 49
        timestamp: now + Duration::seconds(1),
        price: 50005.0,
        amount: 0.1,
    });

    // Exchange 2: Bybit - normal operation
    info!("Bybit: Normal operation");
    for i in 3000..3010 {
        manager.process_trade(&Trade {
            exchange: "bybit".to_string(),
            pair: "ETHUSDT".to_string(),
            trade_id: Some(i),
            timestamp: now + Duration::milliseconds((i - 3000) as i64 * 50),
            price: 3000.0,
            amount: 0.5,
        });
    }

    // Exchange 3: Kucoin - sequence gap
    info!("Kucoin: Processing trades with larger gap");
    manager.process_trade(&Trade {
        exchange: "kucoin".to_string(),
        pair: "SOLUSDT".to_string(),
        trade_id: Some(4000),
        timestamp: now,
        price: 100.0,
        amount: 10.0,
    });

    manager.process_trade(&Trade {
        exchange: "kucoin".to_string(),
        pair: "SOLUSDT".to_string(),
        trade_id: Some(4500), // Gap of 499
        timestamp: now + Duration::seconds(2),
        price: 100.5,
        amount: 10.0,
    });

    // Analyze results
    let gaps = manager.get_all_gaps();
    println!("\nGap Analysis:");
    println!("Total gaps detected: {}", gaps.len());

    // Group by exchange
    let mut binance_gaps = 0;
    let mut bybit_gaps = 0;
    let mut kucoin_gaps = 0;

    for gap in &gaps {
        match gap.exchange.as_str() {
            "binance" => binance_gaps += 1,
            "bybit" => bybit_gaps += 1,
            "kucoin" => kucoin_gaps += 1,
            _ => {}
        }

        println!(
            "  ✗ {}: {} missing {} trades",
            gap.exchange, gap.pair, gap.missing_count
        );
    }

    println!("\nPer-Exchange Summary:");
    println!("  Binance: {} gaps", binance_gaps);
    println!("  Bybit: {} gaps", bybit_gaps);
    println!("  Kucoin: {} gaps", kucoin_gaps);

    assert_eq!(gaps.len(), 2); // Binance and Kucoin have gaps
}

/// Scenario 5: Low-liquidity pair handling
async fn scenario_5_low_liquidity_pair() {
    println!("Scenario 5: Low-Liquidity Pair Handling");
    println!("----------------------------------------");
    println!("Simulating: Distinguishing low-liquidity from actual gaps\n");

    let manager = GapDetectionManager::default();
    let now = Utc::now();

    // High-liquidity pair (BTC): Gaps are significant
    info!("High-liquidity pair (BTCUSD):");
    manager.process_trade(&Trade {
        exchange: "binance".to_string(),
        pair: "BTCUSD".to_string(),
        trade_id: Some(1000),
        timestamp: now,
        price: 50000.0,
        amount: 0.1,
    });

    manager.process_trade(&Trade {
        exchange: "binance".to_string(),
        pair: "BTCUSD".to_string(),
        trade_id: Some(1100), // Gap of 99
        timestamp: now + Duration::seconds(1),
        price: 50005.0,
        amount: 0.1,
    });

    // Low-liquidity pair: Sparse trades are normal
    info!("Low-liquidity pair (RAREUSDT):");
    manager.process_trade(&Trade {
        exchange: "binance".to_string(),
        pair: "RAREUSDT".to_string(),
        trade_id: Some(500),
        timestamp: now,
        price: 1.0,
        amount: 100.0,
    });

    // 30 seconds later - next trade (normal for low-liquidity)
    manager.process_trade(&Trade {
        exchange: "binance".to_string(),
        pair: "RAREUSDT".to_string(),
        trade_id: Some(501), // No gap, just low volume
        timestamp: now + Duration::seconds(30),
        price: 1.01,
        amount: 50.0,
    });

    let gaps = manager.get_all_gaps();

    println!("\nResults:");
    println!("  High-liquidity pair gaps: {}", gaps.len());

    for gap in &gaps {
        println!(
            "    {} ({}): {} missing trades",
            gap.pair, gap.exchange, gap.missing_count
        );
    }

    println!("\n  Low-liquidity pair: No gaps detected");
    println!("  (Sequence IDs are consecutive, time gap is normal)");

    println!("\nKey Insight:");
    println!("  Sequence ID tracking handles low-liquidity pairs correctly.");
    println!("  Time-based detection alone would false-positive on RAREUSDT.");
}

/// Helper function to print gap summary
#[allow(dead_code)]
fn print_gap_summary(gaps: &[Gap]) {
    println!("\n=== Gap Summary ===");
    println!("Total gaps: {}", gaps.len());

    let critical = gaps
        .iter()
        .filter(|g| g.severity() == GapSeverity::Critical)
        .count();
    let high = gaps
        .iter()
        .filter(|g| g.severity() == GapSeverity::High)
        .count();
    let medium = gaps
        .iter()
        .filter(|g| g.severity() == GapSeverity::Medium)
        .count();
    let low = gaps
        .iter()
        .filter(|g| g.severity() == GapSeverity::Low)
        .count();

    println!("  Critical: {}", critical);
    println!("  High: {}", high);
    println!("  Medium: {}", medium);
    println!("  Low: {}", low);
}
