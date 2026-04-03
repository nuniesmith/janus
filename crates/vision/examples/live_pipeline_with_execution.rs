//! Live Trading Pipeline with Execution and Metrics
//!
//! This example demonstrates a complete end-to-end live trading system that integrates:
//! - Real-time market data processing
//! - Model predictions with ensemble methods
//! - Adaptive thresholding and regime detection
//! - Risk management and position sizing
//! - Portfolio optimization
//! - Advanced order execution (TWAP/VWAP)
//! - Prometheus metrics and monitoring
//!
//! This is the production-ready pipeline that ties all components together
//! with full observability.
//!
//! Run with:
//! ```bash
//! cargo run --example live_pipeline_with_execution --release
//! ```

use vision::{
    LivePipeline, LivePipelineConfig, MarketData,
    adaptive::AdaptiveSystem,
    execution::{ExecutionStrategy, InstrumentedExecutionManager, OrderRequest, Side},
};

use std::collections::VecDeque;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Live Trading Pipeline with Execution & Metrics Demo        ║");
    println!("║   Full End-to-End System Integration                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Run different scenarios
    println!("Scenario 1: Basic Live Trading with Metrics");
    println!("════════════════════════════════════════════\n");
    basic_live_trading_with_metrics()?;

    println!("\n\nScenario 2: Multi-Asset Portfolio with Smart Execution");
    println!("═══════════════════════════════════════════════════════\n");
    multi_asset_portfolio_execution()?;

    println!("\n\nScenario 3: Adaptive System with Risk Management");
    println!("═════════════════════════════════════════════════════\n");
    adaptive_risk_managed_execution()?;

    println!("\n\nScenario 4: Full Production Pipeline Simulation");
    println!("════════════════════════════════════════════════════\n");
    full_production_pipeline()?;

    println!("\n\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Demo Complete - Check Metrics Export                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Scenario 1: Basic live trading with metrics collection
fn basic_live_trading_with_metrics() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize components
    let mut pipeline = LivePipeline::default();
    let mut exec_manager = InstrumentedExecutionManager::new();

    pipeline.initialize()?;
    pipeline.warmup()?;

    println!("✓ Pipeline initialized with metrics collection");
    println!("✓ Execution manager ready\n");

    // Simulate live trading session
    let mut position: f64 = 0.0;
    let confidence_threshold = 0.75;
    let position_size = 100.0;

    println!("Starting live trading simulation (50 ticks)...\n");

    for i in 0..50 {
        // Simulate market data with some trend
        let trend = (i as f64 / 10.0).sin() * 2.0;
        let noise = ((i * 7) % 11) as f64 / 10.0 - 0.5;
        let price = 100.0 + trend + noise;

        let data = MarketData::new(
            i as i64,
            price - 0.1,
            price + 0.5,
            price - 0.5,
            price,
            1000.0 + i as f64 * 50.0,
        );

        // Get prediction
        if let Some(prediction) = pipeline.process_tick(data)? {
            // Check if signal meets threshold
            if prediction.meets_confidence(confidence_threshold) {
                let side = if prediction.is_bullish() {
                    Side::Buy
                } else if prediction.is_bearish() {
                    Side::Sell
                } else {
                    continue;
                };

                // Submit order with TWAP execution
                let order_id = exec_manager.submit_order(OrderRequest {
                    symbol: "ASSET".to_string(),
                    quantity: position_size,
                    side: side.clone(),
                    strategy: ExecutionStrategy::TWAP {
                        duration: Duration::from_secs(5),
                        num_slices: 5,
                    },
                    limit_price: Some(
                        price
                            * if matches!(side, Side::Buy) {
                                1.01
                            } else {
                                0.99
                            },
                    ),
                    venues: None,
                });

                println!(
                    "Tick {:3}: {:?} signal @ {:.2} (conf={:.2}%) - Order {}",
                    i,
                    side,
                    price,
                    prediction.confidence * 100.0,
                    order_id
                );

                // Update position tracking
                match side {
                    Side::Buy => position += position_size,
                    Side::Sell => position -= position_size,
                }
            }
        }

        // Process executions
        exec_manager.process();

        // Sleep to simulate real-time
        std::thread::sleep(Duration::from_millis(100));
    }

    // Print results
    println!("\n═══ Session Summary ═══");
    println!("Final Position: {:.0} units", position);
    exec_manager.print_status();

    // Export metrics
    println!("\n═══ Prometheus Metrics Sample ═══");
    let metrics = exec_manager.export_metrics()?;
    let lines: Vec<_> = metrics.lines().take(10).collect();
    for line in lines {
        println!("{}", line);
    }
    println!("... ({} total lines)", metrics.lines().count());

    Ok(())
}

/// Scenario 2: Multi-asset portfolio with smart execution
fn multi_asset_portfolio_execution() -> Result<(), Box<dyn std::error::Error>> {
    let assets = vec!["ASSET_A", "ASSET_B", "ASSET_C"];
    let mut pipelines: Vec<LivePipeline> = assets
        .iter()
        .map(|_| {
            let mut p = LivePipeline::default();
            p.initialize().ok();
            p.warmup().ok();
            p
        })
        .collect();

    let mut exec_manager = InstrumentedExecutionManager::new();

    println!("✓ Initialized {} asset pipelines", assets.len());
    println!("✓ Smart execution ready\n");

    // Track signals and prices
    let mut signals = vec![0.0; assets.len()];
    let mut prices = vec![100.0, 150.0, 200.0];

    println!("Processing multi-asset signals...\n");

    for tick in 0..30 {
        // Generate market data for each asset
        for (i, _asset) in assets.iter().enumerate() {
            let phase = i as f64 * 1.5;
            let trend = (tick as f64 / 5.0 + phase).sin() * 3.0;
            let price = prices[i] + trend;
            prices[i] = price;

            let data = MarketData::new(
                tick as i64,
                price - 0.2,
                price + 0.5,
                price - 0.5,
                price,
                1000.0,
            );

            if let Some(prediction) = pipelines[i].process_tick(data).ok().flatten() {
                signals[i] = prediction.signal;
            }
        }

        // Rebalance portfolio every 10 ticks
        if tick % 10 == 0 && tick > 0 {
            println!("Tick {}: Portfolio Rebalancing", tick);

            // Simple equal-weight strategy for demo
            let target_weight = 1.0 / assets.len() as f64;

            println!("  Target Weights (equal weight):");
            for (i, asset) in assets.iter().enumerate() {
                println!("    {}: {:.1}%", asset, target_weight * 100.0);

                // Execute rebalancing trades with signal-based sizing
                let signal_strength = signals[i].abs();
                if signal_strength > 0.5 {
                    let quantity = target_weight * 10000.0 * signal_strength; // Scale by signal

                    let side = if signals[i] > 0.0 {
                        Side::Buy
                    } else {
                        Side::Sell
                    };

                    let order_id = exec_manager.submit_order(OrderRequest {
                        symbol: asset.to_string(),
                        quantity,
                        side,
                        strategy: ExecutionStrategy::VWAP {
                            duration: Duration::from_secs(3),
                            volume_profile: vision::execution::VolumeProfile::uniform(3),
                        },
                        limit_price: None,
                        venues: None,
                    });
                    println!(
                        "    → Submitted order {} ({} {:.0} units)",
                        order_id,
                        if signals[i] > 0.0 { "BUY" } else { "SELL" },
                        quantity
                    );
                }
            }
        }

        exec_manager.process();
        std::thread::sleep(Duration::from_millis(50));
    }

    println!("\n═══ Multi-Asset Execution Summary ═══");
    exec_manager.print_status();

    Ok(())
}

/// Scenario 3: Adaptive system with risk management
fn adaptive_risk_managed_execution() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize components
    let mut pipeline = LivePipeline::new(LivePipelineConfig::high_accuracy());
    let mut adaptive = AdaptiveSystem::new();
    let mut exec_manager = InstrumentedExecutionManager::new();

    pipeline.initialize()?;
    pipeline.warmup()?;

    println!("✓ High-accuracy pipeline initialized");
    println!("✓ Adaptive system with regime detection");
    println!("✓ Execution manager active\n");

    let mut equity_curve = VecDeque::new();
    let mut current_equity = 100000.0; // $100k starting capital
    let mut position: f64 = 0.0;
    let mut position_price: f64 = 0.0;

    println!("Starting adaptive risk-managed trading...\n");

    for tick in 0..100 {
        // Simulate market with regime changes
        let regime_shift = if tick > 50 { 0.5 } else { -0.5 };
        let volatility = if tick > 50 { 2.0 } else { 0.5 };
        let trend = (tick as f64 / 10.0).sin() * 2.0 + regime_shift;
        let noise = ((tick * 13) % 17) as f64 / 10.0 * volatility;
        let price = 100.0 + trend + noise;

        let data = MarketData::new(
            tick as i64,
            price - 0.2,
            price + 1.0,
            price - 1.0,
            price,
            1000.0 + tick as f64 * 100.0,
        );

        // Get prediction
        if let Some(prediction) = pipeline.process_tick(data)? {
            // Process through adaptive system
            let (calibrated_conf, threshold, should_trade) =
                adaptive.process_prediction(prediction.confidence);

            if tick % 20 == 0 {
                println!(
                    "Tick {:3}: Regime={:?}, Threshold={:.2}%, Conf={:.2}%",
                    tick,
                    adaptive.current_regime(),
                    threshold * 100.0,
                    calibrated_conf * 100.0
                );
            }

            // Trade if adaptive system says so
            if should_trade {
                // Get regime-adjusted position size
                let base_size = adaptive.get_adjusted_position_size(1000.0);

                // Simple risk limit: cap at 1500 units
                let position_size = base_size.min(1500.0);

                let side = if prediction.is_bullish() {
                    Side::Buy
                } else {
                    Side::Sell
                };

                // Submit with smart routing based on size
                let strategy = if position_size > 500.0 {
                    // Large order - use VWAP
                    ExecutionStrategy::VWAP {
                        duration: Duration::from_secs(5),
                        volume_profile: vision::execution::VolumeProfile::intraday_u_shape(5),
                    }
                } else {
                    // Small order - use market
                    ExecutionStrategy::Market
                };

                let order_id = exec_manager.submit_order(OrderRequest {
                    symbol: "ASSET".to_string(),
                    quantity: position_size,
                    side: side.clone(),
                    strategy,
                    limit_price: None,
                    venues: None,
                });

                println!(
                    "  → {:?} {:.0} @ {:.2} (order: {})",
                    side, position_size, price, order_id
                );

                // Update position
                match side {
                    Side::Buy => {
                        position += position_size;
                        position_price = price;
                    }
                    Side::Sell => {
                        // Close position if we have one
                        if position > 0.0 {
                            let pnl = (price - position_price) * position;
                            current_equity += pnl;
                            println!("  → Closed position: P&L = ${:.2}", pnl);
                            position = 0.0;
                        }
                    }
                }
            }
        }

        // Update equity curve
        let mark_to_market = if position > 0.0 {
            current_equity + (price - position_price) * position
        } else {
            current_equity
        };
        equity_curve.push_back(mark_to_market);
        if equity_curve.len() > 100 {
            equity_curve.pop_front();
        }

        // Process executions
        exec_manager.process();
        std::thread::sleep(Duration::from_millis(50));
    }

    // Calculate performance metrics
    let final_equity = equity_curve.back().copied().unwrap_or(current_equity);
    let returns: Vec<f64> = equity_curve
        .iter()
        .zip(equity_curve.iter().skip(1))
        .map(|(a, b)| (b - a) / a)
        .collect();

    let total_return = (final_equity - 100000.0) / 100000.0;
    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility = {
        let variance = returns
            .iter()
            .map(|r| (r - avg_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        variance.sqrt()
    };
    let sharpe = if volatility > 0.0 {
        avg_return / volatility * (252.0_f64).sqrt()
    } else {
        0.0
    };

    println!("\n═══ Performance Summary ═══");
    println!("Starting Equity: $100,000.00");
    println!("Final Equity:    ${:.2}", final_equity);
    println!("Total Return:    {:.2}%", total_return * 100.0);
    println!("Sharpe Ratio:    {:.2}", sharpe);
    println!(
        "Volatility:      {:.2}%",
        volatility * 100.0 * (252.0_f64).sqrt()
    );

    println!("\n═══ Execution Analytics ═══");
    exec_manager.print_status();

    Ok(())
}

/// Scenario 4: Full production pipeline
fn full_production_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing full production pipeline...\n");

    // All components
    let mut pipeline = LivePipeline::new(LivePipelineConfig::low_latency());
    let mut adaptive = AdaptiveSystem::new();
    let mut exec_manager = InstrumentedExecutionManager::new();

    pipeline.initialize()?;
    pipeline.warmup()?;

    println!("✓ Low-latency pipeline (target: <50ms)");
    println!("✓ Adaptive thresholding + regime detection");
    println!("✓ Instrumented execution with metrics\n");

    // Performance tracking
    let start_time = Instant::now();
    let mut tick_count = 0;
    let mut signal_count = 0;
    let mut trade_count = 0;
    let mut latencies = Vec::new();

    println!("Running production simulation (200 ticks)...\n");

    for tick in 0..200 {
        let tick_start = Instant::now();

        // Market data with realistic features
        let time_of_day = (tick % 100) as f64 / 100.0;
        let intraday_pattern = (time_of_day * std::f64::consts::PI * 2.0).sin();
        let trend = (tick as f64 / 20.0).cos() * 5.0;
        let microstructure_noise = ((tick * 31) % 7) as f64 / 10.0 - 0.35;
        let price = 100.0 + trend + intraday_pattern + microstructure_noise;

        let data = MarketData::new(
            tick as i64,
            price - 0.1,
            price + 0.2,
            price - 0.2,
            price,
            1000.0 * (1.0 + intraday_pattern.abs()),
        );

        // Process
        if let Some(prediction) = pipeline.process_tick(data)? {
            tick_count += 1;

            let (calibrated_conf, _threshold, should_trade) =
                adaptive.process_prediction(prediction.confidence);

            if should_trade {
                signal_count += 1;

                // Get position size
                let position_size = adaptive.get_adjusted_position_size(500.0);

                let side = if prediction.is_bullish() {
                    Side::Buy
                } else {
                    Side::Sell
                };

                // Smart order routing based on urgency and size
                let strategy = match (position_size > 300.0, calibrated_conf > 0.9) {
                    (true, true) => {
                        // Large, high confidence - aggressive TWAP
                        ExecutionStrategy::TWAP {
                            duration: Duration::from_secs(2),
                            num_slices: 4,
                        }
                    }
                    (true, false) => {
                        // Large, lower confidence - patient VWAP
                        ExecutionStrategy::VWAP {
                            duration: Duration::from_secs(5),
                            volume_profile: vision::execution::VolumeProfile::uniform(5),
                        }
                    }
                    (false, _) => ExecutionStrategy::Market,
                };

                exec_manager.submit_order(OrderRequest {
                    symbol: "ASSET".to_string(),
                    quantity: position_size,
                    side,
                    strategy,
                    limit_price: None,
                    venues: None,
                });

                trade_count += 1;
            }

            // Track latency
            latencies.push(prediction.metadata.latency_us);
        }

        exec_manager.process();

        // Record tick latency
        let tick_latency = tick_start.elapsed().as_micros();
        if tick % 50 == 0 {
            println!(
                "Tick {:3}: Processed in {} μs (signals: {}, trades: {})",
                tick, tick_latency, signal_count, trade_count
            );
        }

        std::thread::sleep(Duration::from_millis(20));
    }

    let elapsed = start_time.elapsed();

    // Final report
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║              Production Pipeline Performance                  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("═══ Throughput ═══");
    println!("Total Ticks:      {}", tick_count);
    println!("Total Time:       {:.2}s", elapsed.as_secs_f64());
    println!(
        "Throughput:       {:.1} ticks/sec",
        tick_count as f64 / elapsed.as_secs_f64()
    );

    println!("\n═══ Latency ═══");
    if !latencies.is_empty() {
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "Mean:             {:.1} μs",
            latencies.iter().sum::<u64>() as f64 / latencies.len() as f64
        );
        println!("Median (p50):     {} μs", latencies[latencies.len() / 2]);
        println!(
            "p95:              {} μs",
            latencies[(latencies.len() * 95) / 100]
        );
        println!(
            "p99:              {} μs",
            latencies[(latencies.len() * 99) / 100]
        );
        println!("Max:              {} μs", latencies.last().unwrap());
    }

    println!("\n═══ Signal Generation ═══");
    println!("Signals Generated: {}", signal_count);
    println!(
        "Signal Rate:       {:.1}%",
        signal_count as f64 / tick_count as f64 * 100.0
    );
    println!("Trades Executed:   {}", trade_count);
    println!(
        "Execution Rate:    {:.1}%",
        trade_count as f64 / signal_count.max(1) as f64 * 100.0
    );

    println!("\n═══ Execution Quality ═══");
    exec_manager.print_status();

    // Export final metrics
    println!("\n═══ Prometheus Metrics Export ═══");
    let metrics_text = exec_manager.export_metrics()?;
    println!("Total metrics lines: {}", metrics_text.lines().count());
    println!("\nSample metrics:");
    for line in metrics_text.lines().take(15) {
        if !line.is_empty() && !line.starts_with('#') {
            println!("  {}", line);
        }
    }

    println!("\n✓ Production pipeline simulation complete!");
    println!("✓ Metrics available for Prometheus scraping");
    println!("✓ System healthy: {}", exec_manager.is_healthy());

    Ok(())
}
