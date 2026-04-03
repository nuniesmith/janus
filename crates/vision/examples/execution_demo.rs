//! Execution System Demonstration
//!
//! This example demonstrates the complete execution system including:
//! - TWAP (Time-Weighted Average Price) execution
//! - VWAP (Volume-Weighted Average Price) execution
//! - Execution analytics and quality metrics
//! - Order lifecycle management
//! - Smart routing

use std::time::Duration;
use vision::execution::{
    analytics::{ExecutionAnalytics, ExecutionRecord, Side},
    manager::{ExecutionManager, ExecutionStrategy, OrderRequest},
    twap::{TWAPConfig, TWAPExecutor},
    vwap::{VWAPConfig, VWAPExecutor, VolumeProfile},
};

fn main() {
    println!("=== Advanced Order Execution Demo ===\n");

    // Example 1: TWAP Execution
    twap_execution_example();

    // Example 2: VWAP Execution
    vwap_execution_example();

    // Example 3: Execution Analytics
    execution_analytics_example();

    // Example 4: Complete Order Management
    order_management_example();

    // Example 5: Realistic Trading Scenario
    realistic_trading_scenario();
}

fn twap_execution_example() {
    println!("--- Example 1: TWAP Execution ---");
    println!("Goal: Execute 10,000 shares over 5 minutes using 10 slices\n");

    let config = TWAPConfig {
        total_quantity: 10000.0,
        duration: Duration::from_secs(5), // 5 seconds for demo
        num_slices: 10,
        min_slice_size: 100.0,
        randomize_timing: false,
        randomize_size: false,
        timing_randomness: 0.0,
        size_randomness: 0.0,
    };

    let mut executor = TWAPExecutor::new(config);
    executor.start();

    println!("Executing {} slices:", executor.slices().len());
    let start = std::time::Instant::now();

    let mut executed_count = 0;
    while !executor.is_complete() {
        if let Some(slice) = executor.next_slice() {
            executed_count += 1;
            let execution_price = 100.0 + (executed_count as f64 * 0.05); // Simulated price drift
            let slice_id = slice.slice_id;
            let quantity = slice.quantity;

            println!(
                "  Slice {}: {} shares @ ${:.2}",
                slice_id + 1,
                quantity,
                execution_price
            );

            executor.mark_executed(quantity, execution_price);
        }

        std::thread::sleep(Duration::from_millis(500));
    }

    let stats = executor.statistics();
    let elapsed = start.elapsed();

    println!("\nTWAP Execution Complete:");
    println!("  Time Elapsed: {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Slices Executed: {}/{}",
        stats.executed_slices, stats.total_slices
    );
    println!(
        "  Total Filled: {:.0}/{:.0}",
        stats.executed_quantity, stats.total_quantity
    );
    println!("  Fill Rate: {:.1}%", stats.fill_rate * 100.0);
    println!(
        "  Average Price: ${:.2}",
        stats.average_execution_price.unwrap_or(0.0)
    );
    println!();
}

fn vwap_execution_example() {
    println!("--- Example 2: VWAP Execution ---");
    println!("Goal: Execute 15,000 shares following U-shaped volume profile\n");

    // Create U-shaped intraday volume profile (higher at open and close)
    let profile = VolumeProfile::intraday_u_shape(8);

    println!("Volume Profile Distribution:");
    for (i, pct) in profile.percentages.iter().enumerate() {
        let bar_length = (pct * 100.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("  Period {}: {:.1}% {}", i + 1, pct * 100.0, bar);
    }
    println!();

    let config = VWAPConfig {
        total_quantity: 15000.0,
        duration: Duration::from_secs(8), // 8 seconds for demo
        volume_profile: profile,
        min_slice_size: 100.0,
        participation_rate: 0.2, // Target 20% of market volume
        adaptive: true,
    };

    let mut executor = VWAPExecutor::new(config);
    executor.start();

    println!("Executing {} slices:", executor.slices().len());

    let mut period = 0;
    while !executor.is_complete() {
        if let Some(slice) = executor.next_slice() {
            period += 1;
            let quantity = slice.quantity;
            let execution_price = 100.0 + ((period as f64 * 0.3).sin() * 0.5); // Oscillating price
            let market_volume = quantity * 5.0; // Simulated market volume

            println!(
                "  Period {}: {:.0} shares @ ${:.2} (Market Vol: {:.0})",
                period, quantity, execution_price, market_volume
            );

            executor.mark_executed(quantity, execution_price, Some(market_volume));
        }

        std::thread::sleep(Duration::from_secs(1));
    }

    let stats = executor.statistics();

    println!("\nVWAP Execution Complete:");
    println!(
        "  Slices Executed: {}/{}",
        stats.executed_slices, stats.total_slices
    );
    println!(
        "  Total Filled: {:.0}/{:.0}",
        stats.executed_quantity, stats.total_quantity
    );
    println!("  VWAP Price: ${:.2}", stats.vwap_price.unwrap_or(0.0));
    println!(
        "  Target Participation: {:.1}%",
        stats.target_participation * 100.0
    );
    if let Some(actual) = stats.actual_participation {
        println!("  Actual Participation: {:.1}%", actual * 100.0);
    }
    println!("  Total Market Volume: {:.0}", stats.total_market_volume);
    println!();
}

fn execution_analytics_example() {
    println!("--- Example 3: Execution Analytics ---");
    println!("Analyzing execution quality across multiple orders\n");

    let mut analytics = ExecutionAnalytics::new();

    // Simulate multiple executions across different venues
    let executions = vec![
        ("ORD001", 1000.0, 100.5, 100.0, Side::Buy, "NYSE"),
        ("ORD001", 1000.0, 100.7, 100.0, Side::Buy, "NYSE"),
        ("ORD002", 500.0, 99.8, 100.0, Side::Sell, "NASDAQ"),
        ("ORD002", 500.0, 99.6, 100.0, Side::Sell, "NASDAQ"),
        ("ORD003", 2000.0, 100.3, 100.0, Side::Buy, "BATS"),
        ("ORD004", 1500.0, 100.1, 100.0, Side::Buy, "IEX"),
        ("ORD005", 800.0, 99.9, 100.0, Side::Sell, "DARK"),
    ];

    println!("Recording {} executions...", executions.len());
    for (order_id, qty, exec_price, bench_price, side, venue) in executions {
        analytics.record_execution(ExecutionRecord {
            order_id: order_id.to_string(),
            quantity: qty,
            execution_price: exec_price,
            benchmark_price: bench_price,
            timestamp: std::time::Instant::now(),
            side,
            venue: venue.to_string(),
        });

        let slippage = ((exec_price - bench_price) / bench_price * 10000.0).abs();
        println!(
            "  {} {} {:.0} @ ${:.2} on {} (Slippage: {:.1} bps)",
            order_id,
            match side {
                Side::Buy => "BUY ",
                Side::Sell => "SELL",
            },
            qty,
            exec_price,
            venue,
            slippage
        );
    }

    println!("\n=== Execution Analytics Report ===");
    let report = analytics.generate_report();

    println!("Overall Statistics:");
    println!("  Total Executions: {}", report.total_executions);
    println!("    Buy: {}", report.buy_executions);
    println!("    Sell: {}", report.sell_executions);
    println!("  Total Quantity: {:.0}", report.total_quantity);

    println!("\nSlippage Analysis:");
    println!("  Average: {:.2} bps", report.average_slippage_bps);
    println!("  VWAP: {:.2} bps", report.vwap_slippage_bps);
    println!("  Min: {:.2} bps", report.min_slippage_bps);
    println!("  Max: {:.2} bps", report.max_slippage_bps);

    println!("\nCost Analysis:");
    println!("  Total Cost: ${:.2}", report.total_cost);
    println!(
        "  Implementation Shortfall: {:.4}%",
        report.implementation_shortfall
    );

    println!("\nExecution Quality:");
    println!("  Quality Score: {:.1}/100", report.quality_score);
    if report.quality_score >= 80.0 {
        println!("  Rating: Excellent ⭐⭐⭐");
    } else if report.quality_score >= 60.0 {
        println!("  Rating: Good ⭐⭐");
    } else {
        println!("  Rating: Fair ⭐");
    }

    println!("\nVenue Statistics:");
    for (venue, stats) in &report.venue_stats {
        println!("  {}:", venue);
        println!("    Executions: {}", stats.execution_count);
        println!("    Quantity: {:.0}", stats.total_quantity);
        println!("    Avg Slippage: {:.2} bps", stats.average_slippage_bps);
        println!("    Cost: ${:.2}", stats.total_cost);
    }

    println!();
}

fn order_management_example() {
    println!("--- Example 4: Complete Order Management ---");
    println!("Managing multiple orders with different strategies\n");

    let mut manager = ExecutionManager::new();

    // Set benchmark prices
    manager.set_benchmark_price("AAPL".to_string(), 150.0);
    manager.set_benchmark_price("GOOGL".to_string(), 2800.0);

    // Submit different order types
    println!("Submitting orders...");

    let order1 = manager.submit_order(OrderRequest {
        symbol: "AAPL".to_string(),
        quantity: 1000.0,
        side: Side::Buy,
        strategy: ExecutionStrategy::Market,
        limit_price: None,
        venues: None,
    });
    println!("  {} - Market order for 1000 AAPL", order1);

    let order2 = manager.submit_order(OrderRequest {
        symbol: "AAPL".to_string(),
        quantity: 5000.0,
        side: Side::Buy,
        strategy: ExecutionStrategy::TWAP {
            duration: Duration::from_secs(3),
            num_slices: 3,
        },
        limit_price: Some(151.0),
        venues: None,
    });
    println!("  {} - TWAP order for 5000 AAPL", order2);

    let order3 = manager.submit_order(OrderRequest {
        symbol: "GOOGL".to_string(),
        quantity: 500.0,
        side: Side::Sell,
        strategy: ExecutionStrategy::VWAP {
            duration: Duration::from_secs(3),
            volume_profile: VolumeProfile::uniform(3),
        },
        limit_price: None,
        venues: None,
    });
    println!("  {} - VWAP order for 500 GOOGL", order3);

    println!("\nProcessing orders...");

    for i in 0..4 {
        std::thread::sleep(Duration::from_secs(1));
        manager.process();

        println!("\n  === Cycle {} ===", i + 1);
        for status in manager.active_orders() {
            println!(
                "  {}: {:?} - {:.0}/{:.0} filled ({:.0}%)",
                status.order_id,
                status.state,
                status.filled_quantity,
                status.total_quantity,
                status.fill_rate() * 100.0
            );
        }
    }

    println!("\n=== Final Order Status ===");
    for order_id in [&order1, &order2, &order3] {
        if let Some(status) = manager.get_order_status(order_id) {
            println!("\n{} ({}):", status.order_id, status.symbol);
            println!("  State: {:?}", status.state);
            println!(
                "  Filled: {:.0}/{:.0}",
                status.filled_quantity, status.total_quantity
            );
            println!("  Fill Rate: {:.1}%", status.fill_rate() * 100.0);
            if let Some(avg_price) = status.average_price {
                println!("  Avg Price: ${:.2}", avg_price);
            }
            println!("  Fills: {}", status.fills.len());
        }
    }

    println!("\n=== Execution Performance ===");
    let report = manager.execution_report();
    println!("Total Executions: {}", report.total_executions);
    println!("Average Slippage: {:.2} bps", report.average_slippage_bps);
    println!("Quality Score: {:.1}/100", report.quality_score);

    println!();
}

fn realistic_trading_scenario() {
    println!("--- Example 5: Realistic Trading Scenario ---");
    println!("Day trading with adaptive execution strategies\n");

    let mut manager = ExecutionManager::new();
    manager.set_benchmark_price("STOCK".to_string(), 100.0);

    let scenarios = vec![
        (
            "Morning Open",
            20000.0,
            ExecutionStrategy::VWAP {
                duration: Duration::from_secs(2),
                volume_profile: VolumeProfile::reverse_j_shape(4), // High volume at open
            },
        ),
        (
            "Mid-Day",
            10000.0,
            ExecutionStrategy::TWAP {
                duration: Duration::from_secs(2),
                num_slices: 4,
            },
        ),
        (
            "Afternoon Rally",
            15000.0,
            ExecutionStrategy::VWAP {
                duration: Duration::from_secs(2),
                volume_profile: VolumeProfile::uniform(4),
            },
        ),
        (
            "Close",
            25000.0,
            ExecutionStrategy::VWAP {
                duration: Duration::from_secs(2),
                volume_profile: VolumeProfile::intraday_u_shape(4), // High volume at close
            },
        ),
    ];

    let mut total_quantity = 0.0;
    let session_start = std::time::Instant::now();

    for (period, quantity, strategy) in scenarios {
        println!("=== {} Session ===", period);
        println!("Target: {:.0} shares", quantity);
        total_quantity += quantity;

        let order_id = manager.submit_order(OrderRequest {
            symbol: "STOCK".to_string(),
            quantity,
            side: Side::Buy,
            strategy,
            limit_price: None,
            venues: None,
        });

        // Execute the session
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(500));
            manager.process();
        }

        if let Some(status) = manager.get_order_status(&order_id) {
            println!(
                "  Result: {:.0}/{:.0} filled ({:.1}%)",
                status.filled_quantity,
                status.total_quantity,
                status.fill_rate() * 100.0
            );
            if let Some(avg_price) = status.average_price {
                println!("  Avg Price: ${:.2}", avg_price);
            }
        }

        println!();
    }

    let session_duration = session_start.elapsed();

    println!("=== End of Day Summary ===");
    println!("Session Duration: {:.1}s", session_duration.as_secs_f64());
    println!("Target Total: {:.0} shares", total_quantity);

    let report = manager.execution_report();
    println!("\nExecution Statistics:");
    println!("  Total Executions: {}", report.total_executions);
    println!("  Total Quantity: {:.0}", report.total_quantity);
    println!("  Avg Slippage: {:.2} bps", report.average_slippage_bps);
    println!("  VWAP Slippage: {:.2} bps", report.vwap_slippage_bps);
    println!("  Total Cost: ${:.2}", report.total_cost);
    println!(
        "  Implementation Shortfall: {:.4}%",
        report.implementation_shortfall
    );
    println!("  Quality Score: {:.1}/100", report.quality_score);

    if !report.venue_stats.is_empty() {
        println!("\nVenue Breakdown:");
        for (venue, stats) in &report.venue_stats {
            println!(
                "  {}: {} fills, {:.0} shares, ${:.2} cost",
                venue, stats.execution_count, stats.total_quantity, stats.total_cost
            );
        }
    }

    println!("\n✓ Trading session complete!");
}
