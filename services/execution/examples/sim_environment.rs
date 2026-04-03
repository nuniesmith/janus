//! FKS Simulation Environment Example
//!
//! This example demonstrates how to use the unified simulation environment for:
//! - Backtesting with historical data
//! - Forward testing (paper trading) with live data
//! - Strategy optimization with walk-forward analysis
//!
//! # Usage
//!
//! ```bash
//! # Run backtest mode
//! cargo run --release --example sim_environment -- backtest
//!
//! # Run forward test (paper trading) with live data
//! cargo run --release --example sim_environment -- forward
//!
//! # Run optimization
//! cargo run --release --example sim_environment -- optimize
//!
//! # Run with custom settings
//! SIM_SYMBOLS=BTC/USDT,ETH/USDT SIM_INITIAL_BALANCE=50000 \
//!     cargo run --release --example sim_environment -- forward
//! ```

use chrono::{Duration, Utc};
use janus_execution::exchanges::{
    MarketDataAggregator, MarketDataProvider, binance::BinanceProvider, bybit::BybitProvider,
    kraken::KrakenProvider,
};
use janus_execution::sim::{
    config::SimConfig,
    data_feed::{MarketEvent, TickData},
    environment::{Signal, SimEnvironment, Strategy},
    live_feed_bridge::LiveFeedBridgeConfig,
    optimization::{
        OptimizationConfig, OptimizationDirection, OptimizationMetric, ParameterRange,
        WalkForwardAnalysis, WalkForwardConfig,
    },
    replay::{ReplayConfig, ReplayEngine},
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use tokio::time::interval;
use tracing::{Level, debug, info, warn};
use tracing_subscriber::FmtSubscriber;

/// Example EMA Crossover Strategy
///
/// Generates buy signals when fast EMA crosses above slow EMA,
/// and sell signals when fast EMA crosses below slow EMA.
struct EMACrossoverStrategy {
    name: String,
    fast_period: usize,
    slow_period: usize,
    // State
    prices: HashMap<String, Vec<f64>>,
    fast_ema: HashMap<String, Option<f64>>,
    slow_ema: HashMap<String, Option<f64>>,
    position_size: Decimal,
}

impl EMACrossoverStrategy {
    fn new(fast_period: usize, slow_period: usize, position_size: Decimal) -> Self {
        Self {
            name: format!("EMA_{}_{}", fast_period, slow_period),
            fast_period,
            slow_period,
            prices: HashMap::new(),
            fast_ema: HashMap::new(),
            slow_ema: HashMap::new(),
            position_size,
        }
    }

    fn update_ema(&self, prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period {
            return None;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[..period].iter().sum::<f64>() / period as f64;

        for price in &prices[period..] {
            ema = (price - ema) * multiplier + ema;
        }

        Some(ema)
    }
}

impl Strategy for EMACrossoverStrategy {
    fn on_event(&mut self, event: &MarketEvent) -> Vec<Signal> {
        let tick = match event {
            MarketEvent::Tick(t) => t,
            _ => return vec![Signal::None],
        };

        // Get mid price
        let mid_price = tick.mid_price();
        let price_f64 = mid_price.to_string().parse::<f64>().unwrap_or(0.0);

        if price_f64 <= 0.0 {
            return vec![Signal::None];
        }

        // Update price history
        {
            let prices = self.prices.entry(tick.symbol.clone()).or_default();
            prices.push(price_f64);

            // Keep only necessary history
            let max_history = self.slow_period * 2;
            if prices.len() > max_history {
                prices.drain(0..prices.len() - max_history);
            }
        }

        // Calculate EMAs (clone prices to avoid borrow conflict)
        let prev_fast = self.fast_ema.get(&tick.symbol).copied().flatten();
        let prev_slow = self.slow_ema.get(&tick.symbol).copied().flatten();

        let prices_clone = self.prices.get(&tick.symbol).cloned().unwrap_or_default();
        let new_fast = self.update_ema(&prices_clone, self.fast_period);
        let new_slow = self.update_ema(&prices_clone, self.slow_period);

        // Store new EMAs
        self.fast_ema.insert(tick.symbol.clone(), new_fast);
        self.slow_ema.insert(tick.symbol.clone(), new_slow);

        // Check for crossover
        if let (Some(pf), Some(ps), Some(nf), Some(ns)) = (prev_fast, prev_slow, new_fast, new_slow)
        {
            // Bullish crossover: fast crosses above slow
            if pf <= ps && nf > ns {
                info!(
                    "🟢 BUY signal for {} - Fast EMA ({:.2}) crossed above Slow EMA ({:.2})",
                    tick.symbol, nf, ns
                );
                return vec![Signal::Buy {
                    symbol: tick.symbol.clone(),
                    size: self.position_size,
                    price: Some(tick.ask_price),
                    stop_loss: Some(mid_price * dec!(0.98)), // 2% stop loss
                    take_profit: Some(mid_price * dec!(1.04)), // 4% take profit
                }];
            }
            // Bearish crossover: fast crosses below slow
            if pf >= ps && nf < ns {
                info!(
                    "🔴 SELL signal for {} - Fast EMA ({:.2}) crossed below Slow EMA ({:.2})",
                    tick.symbol, nf, ns
                );
                return vec![Signal::Sell {
                    symbol: tick.symbol.clone(),
                    size: self.position_size,
                    price: Some(tick.bid_price),
                    stop_loss: Some(mid_price * dec!(1.02)), // 2% stop loss
                    take_profit: Some(mid_price * dec!(0.96)), // 4% take profit
                }];
            }
        }

        vec![Signal::None]
    }

    fn on_start(&mut self) {
        info!("Starting {} strategy", self.name);
    }

    fn on_stop(&mut self) {
        info!("Stopping {} strategy", self.name);
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.prices.clear();
        self.fast_ema.clear();
        self.slow_ema.clear();
    }
}

/// Run backtest mode
async fn run_backtest() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              FKS Simulation - Backtest Mode                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Create backtest configuration
    // In a real scenario, you would load from a parquet or CSV file
    let config = SimConfig::backtest()
        .with_initial_balance(10_000.0)
        .with_symbols(vec!["BTC/USDT".to_string(), "ETH/USDT".to_string()])
        .with_slippage_bps(5.0)
        .with_commission_bps(6.0)
        .with_verbose(true)
        .build_unchecked(); // Use build_unchecked since we don't have a real data file

    info!("Configuration: {:?}", config.mode);
    info!("Initial balance: ${}", config.initial_balance);
    info!("Symbols: {:?}", config.symbols);
    info!(
        "Execution: slippage={} bps, commission={} bps",
        config.execution.slippage_bps, config.execution.commission_bps
    );

    // Create environment
    let mut env = SimEnvironment::new(config).await?;
    env.initialize().await?;

    // Create strategy
    let mut strategy = EMACrossoverStrategy::new(8, 21, dec!(0.1));

    // Generate some synthetic test data for demonstration
    info!("\nGenerating synthetic market data for backtest...");

    let base_time = Utc::now() - Duration::hours(24);
    let mut btc_price = 50000.0_f64;
    let mut eth_price = 3000.0_f64;

    // Simulate 24 hours of 1-minute data
    for i in 0..1440 {
        let timestamp = base_time + Duration::minutes(i);

        // Add some price movement (random walk with trend)
        let btc_change = (rand_simple(i as u64) - 0.5) * 100.0;
        let eth_change = (rand_simple(i as u64 + 1000) - 0.5) * 10.0;

        btc_price = (btc_price + btc_change).clamp(45000.0, 55000.0);
        eth_price = (eth_price + eth_change).clamp(2800.0, 3200.0);

        // Create ticks
        let btc_tick = TickData::new(
            "BTC/USDT",
            "kraken",
            Decimal::try_from(btc_price - 5.0).unwrap(),
            Decimal::try_from(btc_price + 5.0).unwrap(),
            dec!(1.0),
            dec!(1.0),
            timestamp,
        );

        let eth_tick = TickData::new(
            "ETH/USDT",
            "kraken",
            Decimal::try_from(eth_price - 0.5).unwrap(),
            Decimal::try_from(eth_price + 0.5).unwrap(),
            dec!(10.0),
            dec!(10.0),
            timestamp,
        );

        // Publish events
        env.publish_event(MarketEvent::Tick(btc_tick));
        env.publish_event(MarketEvent::Tick(eth_tick));
    }

    // Run backtest
    info!("\nRunning backtest...");
    let result = env.run_backtest(&mut strategy).await?;

    // Print results
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    BACKTEST RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Strategy:        {}", result.strategy_name);
    println!("Duration:        {} seconds", result.duration_seconds);
    println!(
        "Initial Balance: ${:.2}",
        result.initial_balance.to_string().parse::<f64>().unwrap()
    );
    println!(
        "Final Equity:    ${:.2}",
        result.final_equity.to_string().parse::<f64>().unwrap()
    );
    println!(
        "Total Return:    {:.2}%",
        result.total_return_pct.to_string().parse::<f64>().unwrap()
    );
    println!("Total Trades:    {}", result.total_trades);
    println!("Win Rate:        {:.1}%", result.win_rate);
    println!(
        "Max Drawdown:    {:.2}%",
        result.max_drawdown_pct.to_string().parse::<f64>().unwrap() * 100.0
    );
    println!(
        "Sharpe Ratio:    {}",
        result
            .sharpe_ratio
            .map(|s| format!("{:.2}", s))
            .unwrap_or_else(|| "N/A".to_string())
    );
    println!("Events Processed: {}", result.events_processed);
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Run forward test (paper trading) mode
async fn run_forward_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           FKS Simulation - Forward Test Mode                  ║");
    println!("║                 (Paper Trading with Live Data)                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Get settings from environment
    let symbols: Vec<String> = env::var("SIM_SYMBOLS")
        .unwrap_or_else(|_| "BTC/USDT,ETH/USDT".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let initial_balance: f64 = env::var("SIM_INITIAL_BALANCE")
        .unwrap_or_else(|_| "10000".to_string())
        .parse()
        .unwrap_or(10000.0);

    let duration_secs: u64 = env::var("SIM_DURATION_SECS")
        .unwrap_or_else(|_| "60".to_string())
        .parse()
        .unwrap_or(60);

    // Create forward test configuration
    let config = SimConfig::forward_test()
        .with_exchanges(vec![
            "kraken".to_string(),
            "binance".to_string(),
            "bybit".to_string(),
        ])
        .with_symbols(symbols.clone())
        .with_initial_balance(initial_balance)
        .with_slippage_bps(5.0)
        .with_commission_bps(6.0)
        .with_record_data(true)
        .with_questdb("localhost", 9009)
        .with_verbose(true)
        .build_unchecked();

    info!("Configuration:");
    info!("  Mode: {:?}", config.mode);
    info!("  Symbols: {:?}", symbols);
    info!("  Initial Balance: ${}", initial_balance);
    info!("  Duration: {} seconds", duration_secs);
    info!("  Recording: {}", config.recording.enabled);

    // Create environment
    let mut env = SimEnvironment::new(config).await?;
    env.initialize().await?;

    // Create strategy
    let mut strategy = EMACrossoverStrategy::new(8, 21, dec!(0.1));

    info!("\nStarting forward test...");
    info!("Press Ctrl+C to stop early\n");

    // In a real forward test, we would connect to live exchange WebSocket feeds
    // For this demo, we'll simulate live data
    let data_feed = env.data_feed();

    // Spawn a task to generate simulated "live" data
    let symbols_clone = symbols.clone();
    let feed_task = tokio::spawn(async move {
        let mut prices: HashMap<String, f64> = HashMap::new();
        prices.insert("BTC/USDT".to_string(), 50000.0);
        prices.insert("ETH/USDT".to_string(), 3000.0);

        let mut tick_interval = interval(std::time::Duration::from_millis(500));
        let mut tick_count = 0u64;

        loop {
            tick_interval.tick().await;
            tick_count += 1;

            for symbol in &symbols_clone {
                if let Some(price) = prices.get_mut(symbol) {
                    // Simulate price movement
                    let change = (rand_simple(tick_count + symbol.len() as u64) - 0.5) * 10.0;
                    *price = (*price + change).max(100.0);

                    let spread = *price * 0.0001; // 1 bps spread

                    let tick = TickData::new(
                        symbol,
                        "kraken",
                        Decimal::try_from(*price - spread).unwrap(),
                        Decimal::try_from(*price + spread).unwrap(),
                        dec!(1.0),
                        dec!(1.0),
                        Utc::now(),
                    );

                    data_feed.read().publish(MarketEvent::Tick(tick));
                }
            }

            if tick_count.is_multiple_of(10) {
                debug!("Generated {} ticks", tick_count);
            }
        }
    });

    // Run forward test
    let duration = std::time::Duration::from_secs(duration_secs);
    let result = env.run_forward_test(&mut strategy, duration).await?;

    // Stop the feed task
    feed_task.abort();

    // Print results
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                  FORWARD TEST RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Strategy:        {}", result.strategy_name);
    println!(
        "Duration:        {} seconds ({:.1} minutes)",
        result.duration_seconds,
        result.duration_seconds as f64 / 60.0
    );
    println!(
        "Initial Balance: ${:.2}",
        result.initial_balance.to_string().parse::<f64>().unwrap()
    );
    println!(
        "Final Equity:    ${:.2}",
        result.final_equity.to_string().parse::<f64>().unwrap()
    );
    println!(
        "Total Return:    {:.2}%",
        result.total_return_pct.to_string().parse::<f64>().unwrap()
    );
    println!("Total Trades:    {}", result.total_trades);
    println!("Events Processed: {}", result.events_processed);

    if !result.final_positions.is_empty() {
        println!("\nOpen Positions:");
        for pos in &result.final_positions {
            println!(
                "  {} {}: size={}, entry={}, current={}, unrealized_pnl={}",
                if pos.size > Decimal::ZERO {
                    "LONG"
                } else {
                    "SHORT"
                },
                pos.symbol,
                pos.size,
                pos.entry_price,
                pos.current_price,
                pos.unrealized_pnl
            );
        }
    }

    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Run optimization mode
async fn run_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           FKS Simulation - Optimization Mode                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Define parameter ranges
    let opt_config = OptimizationConfig::new()
        .with_parameter(ParameterRange::int("fast_period", 5, 15, 2)) // 5, 7, 9, 11, 13, 15
        .with_parameter(ParameterRange::int("slow_period", 15, 30, 5)) // 15, 20, 25, 30
        .with_metric(OptimizationMetric::SharpeRatio)
        .with_direction(OptimizationDirection::Maximize)
        .with_top_n(5)
        .with_verbose(true);

    let total_combinations = opt_config.total_combinations();
    info!(
        "Optimization Configuration: {} total parameter combinations",
        total_combinations
    );

    // Generate all parameter combinations
    let combinations = opt_config.generate_combinations();

    println!("\nParameter Combinations to Test:");
    for (i, combo) in combinations.iter().enumerate() {
        let fast = combo
            .get("fast_period")
            .and_then(|v| v.as_int())
            .unwrap_or(0);
        let slow = combo
            .get("slow_period")
            .and_then(|v| v.as_int())
            .unwrap_or(0);
        println!("  {}: fast_period={}, slow_period={}", i + 1, fast, slow);
    }

    println!("\nRunning optimization (this is a simplified demo)...");

    // In a real optimization, we would:
    // 1. Load historical data
    // 2. Run backtest for each parameter combination
    // 3. Track results and find the best

    // For demo, we'll simulate optimization results
    let mut results: Vec<(i64, i64, f64)> = Vec::new();

    for combo in &combinations {
        let fast = combo
            .get("fast_period")
            .and_then(|v| v.as_int())
            .unwrap_or(0);
        let slow = combo
            .get("slow_period")
            .and_then(|v| v.as_int())
            .unwrap_or(0);

        // Simulate a Sharpe ratio (in reality, this comes from backtest)
        // Assume closer to 8/21 is better (just for demo)
        let fast_diff = (fast - 8).abs() as f64;
        let slow_diff = (slow - 21).abs() as f64;
        let simulated_sharpe = 2.0 - (fast_diff * 0.1 + slow_diff * 0.05);

        results.push((fast, slow, simulated_sharpe));

        debug!(
            "Tested fast={}, slow={} -> Sharpe={:.4}",
            fast, slow, simulated_sharpe
        );
    }

    // Sort by Sharpe ratio (descending)
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                  OPTIMIZATION RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("\nTop 5 Parameter Combinations:");
    println!("  Rank | Fast Period | Slow Period | Sharpe Ratio");
    println!("  -----|-------------|-------------|-------------");

    for (i, (fast, slow, sharpe)) in results.iter().take(5).enumerate() {
        println!("  {:4} | {:11} | {:11} | {:.4}", i + 1, fast, slow, sharpe);
    }

    let (best_fast, best_slow, best_sharpe) = results[0];
    println!(
        "\n✅ Best Parameters: fast_period={}, slow_period={}",
        best_fast, best_slow
    );
    println!("   Expected Sharpe Ratio: {:.4}", best_sharpe);

    // Demonstrate Walk-Forward Analysis setup
    println!("\n--- Walk-Forward Analysis Demo ---");

    let wf_config = WalkForwardConfig::new(4)
        .with_in_sample_pct(0.7)
        .with_min_trades(30)
        .rolling()
        .with_optimization(opt_config);

    let data_start = Utc::now() - Duration::days(365);
    let data_end = Utc::now();

    match WalkForwardAnalysis::new(wf_config, data_start, data_end) {
        Ok(wf_analysis) => {
            println!("\nWalk-Forward Windows:");
            for window in wf_analysis.windows() {
                println!(
                    "  Window {}: IS [{} to {}] -> OOS [{} to {}]",
                    window.index + 1,
                    window.is_start.format("%Y-%m-%d"),
                    window.is_end.format("%Y-%m-%d"),
                    window.oos_start.format("%Y-%m-%d"),
                    window.oos_end.format("%Y-%m-%d"),
                );
            }
        }
        Err(e) => {
            warn!("Could not create walk-forward analysis: {}", e);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Run replay demo
async fn run_replay_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║             FKS Simulation - Replay Engine Demo               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Create replay configuration
    let config = ReplayConfig::fast()
        .with_symbols(vec!["BTC/USDT".to_string()])
        .with_verbose(true);

    let mut engine = ReplayEngine::new(config);

    // Generate some test events
    info!("Generating test data...");

    let base_time = Utc::now() - Duration::hours(1);
    let events: Vec<MarketEvent> = (0..1000)
        .map(|i| {
            let price = 50000.0 + (i as f64 * 0.5);
            MarketEvent::Tick(TickData::new(
                "BTC/USDT",
                "kraken",
                Decimal::try_from(price - 5.0).unwrap(),
                Decimal::try_from(price + 5.0).unwrap(),
                dec!(1.0),
                dec!(1.0),
                base_time + Duration::milliseconds(i * 100),
            ))
        })
        .collect();

    engine.load_events(events)?;

    let stats = engine.stats();
    println!("\nLoaded Data Statistics:");
    println!("  Total Events: {}", stats.total_events);
    println!("  Symbols: {:?}", stats.symbols);
    println!("  Exchanges: {:?}", stats.exchanges);
    println!(
        "  Time Range: {} to {}",
        stats
            .data_start_time
            .map(|t| t.to_string())
            .unwrap_or_default(),
        stats
            .data_end_time
            .map(|t| t.to_string())
            .unwrap_or_default()
    );

    // Subscribe to events
    let mut rx = engine.subscribe();

    // Start replay
    info!("\nStarting replay at maximum speed...");

    engine.start().await?;

    // Count received events
    let mut received = 0;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, MarketEvent::EndOfData) {
            break;
        }
        received += 1;
    }

    let final_stats = engine.stats();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                     REPLAY RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Events Replayed: {}", final_stats.events_replayed);
    println!("Events Received: {}", received);
    println!("Elapsed Time:    {:.3}s", final_stats.elapsed_seconds);
    println!(
        "Replay Rate:     {:.0} events/sec",
        final_stats.events_per_second
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Simple pseudo-random number generator for demo purposes
fn rand_simple(seed: u64) -> f64 {
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 16) & 0x7fff) as f64 / 32768.0
}

/// Run forward test with real live exchange data via LiveFeedBridge
async fn run_live_forward_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           FKS Simulation - Live Forward Test Mode             ║");
    println!("║           (Paper Trading with REAL Exchange Data)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Get settings from environment
    let symbols: Vec<String> = env::var("SIM_SYMBOLS")
        .unwrap_or_else(|_| "BTC/USDT,ETH/USDT".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let initial_balance: f64 = env::var("SIM_INITIAL_BALANCE")
        .unwrap_or_else(|_| "10000".to_string())
        .parse()
        .unwrap_or(10000.0);

    let duration_secs: u64 = env::var("SIM_DURATION_SECS")
        .unwrap_or_else(|_| "60".to_string())
        .parse()
        .unwrap_or(60);

    // Create forward test configuration
    let config = SimConfig::forward_test()
        .with_exchanges(vec![
            "kraken".to_string(),
            "binance".to_string(),
            "bybit".to_string(),
        ])
        .with_symbols(symbols.clone())
        .with_initial_balance(initial_balance)
        .with_slippage_bps(5.0)
        .with_commission_bps(6.0)
        .with_record_data(false) // Don't record for this demo
        .with_verbose(true)
        .build_unchecked();

    info!("Configuration:");
    info!("  Mode: {:?} (LIVE DATA)", config.mode);
    info!("  Symbols: {:?}", symbols);
    info!("  Initial Balance: ${}", initial_balance);
    info!("  Duration: {} seconds", duration_secs);

    // Create environment
    let mut env = SimEnvironment::new(config).await?;
    env.initialize().await?;

    // Create market data aggregator and connect to exchanges
    let mut aggregator = MarketDataAggregator::new();
    let mut connected_exchanges: Vec<&str> = Vec::new();

    // Kraken provider
    info!("Connecting to Kraken...");
    let kraken_provider = Arc::new(KrakenProvider::new());
    match kraken_provider.connect().await {
        Ok(_) => {
            info!("✅ Kraken connected");
            let symbols_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
            match kraken_provider.subscribe_ticker(&symbols_refs).await {
                Err(e) => {
                    warn!("Failed to subscribe to Kraken tickers: {}", e);
                }
                _ => {
                    connected_exchanges.push("Kraken");
                    aggregator.add_provider(kraken_provider);
                }
            }
        }
        Err(e) => {
            warn!("❌ Failed to connect to Kraken: {}", e);
        }
    }

    // Binance provider (subscribe BEFORE connect for combined streams)
    info!("Connecting to Binance...");
    let binance_provider = Arc::new(BinanceProvider::new());
    let symbols_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
    if let Err(e) = binance_provider.subscribe_ticker(&symbols_refs).await {
        warn!("Failed to queue Binance ticker subscriptions: {}", e);
    }
    match binance_provider.connect().await {
        Ok(_) => {
            info!("✅ Binance connected");
            connected_exchanges.push("Binance");
            aggregator.add_provider(binance_provider);
        }
        Err(e) => {
            warn!("❌ Failed to connect to Binance: {}", e);
        }
    }

    // Bybit provider (Linear perpetuals for bid/ask data)
    info!("Connecting to Bybit...");
    let bybit_provider = Arc::new(BybitProvider::linear(false));
    match bybit_provider.connect().await {
        Ok(_) => {
            info!("✅ Bybit connected");
            let symbols_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
            let _ = bybit_provider.subscribe_ticker(&symbols_refs).await;
            connected_exchanges.push("Bybit");
            aggregator.add_provider(bybit_provider);
        }
        Err(e) => {
            warn!("❌ Failed to connect to Bybit: {}", e);
        }
    }

    if connected_exchanges.is_empty() {
        warn!("No exchanges connected! Cannot run live forward test.");
        return Err("No exchanges connected".into());
    }

    info!(
        "Connected to {} exchange(s): {:?}",
        connected_exchanges.len(),
        connected_exchanges
    );

    // Connect the SimEnvironment to live data via LiveFeedBridge
    let bridge_config = LiveFeedBridgeConfig::default()
        .with_normalize_symbols(true)
        .with_filter_invalid_ticks(true)
        .with_tick_debounce_ms(50); // Debounce rapid ticks

    let bridge = env
        .connect_live_data(&aggregator, Some(bridge_config))
        .await?;
    info!("LiveFeedBridge connected to SimEnvironment");

    // Create strategy
    let mut strategy = EMACrossoverStrategy::new(8, 21, dec!(0.1));

    info!("\nStarting live forward test...");
    info!("Press Ctrl+C to stop early\n");

    // Run forward test with real live data
    let duration = std::time::Duration::from_secs(duration_secs);
    let result = env.run_forward_test(&mut strategy, duration).await?;

    // Stop the bridge
    bridge.stop().await;

    // Get bridge statistics
    let bridge_stats = bridge.stats();

    // Disconnect exchanges
    aggregator.disconnect_all().await?;

    // Print results
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("               LIVE FORWARD TEST RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Strategy:        {}", result.strategy_name);
    println!(
        "Duration:        {} seconds ({:.1} minutes)",
        result.duration_seconds,
        result.duration_seconds as f64 / 60.0
    );
    println!(
        "Initial Balance: ${:.2}",
        result.initial_balance.to_string().parse::<f64>().unwrap()
    );
    println!(
        "Final Equity:    ${:.2}",
        result.final_equity.to_string().parse::<f64>().unwrap()
    );
    println!(
        "Total Return:    {:.2}%",
        result.total_return_pct.to_string().parse::<f64>().unwrap()
    );
    println!("Total Trades:    {}", result.total_trades);
    println!("Events Processed: {}", result.events_processed);

    println!("\n--- LiveFeedBridge Statistics ---");
    println!("Events Received:  {}", bridge_stats.events_received);
    println!("Events Published: {}", bridge_stats.events_published);
    println!("Events Dropped:   {}", bridge_stats.events_dropped);
    println!("Success Rate:     {:.2}%", bridge_stats.success_rate());
    let rate = bridge_stats.events_per_second();
    if rate > 0.0 {
        println!("Events/Second:    {:.2}", rate);
    }

    if !result.final_positions.is_empty() {
        println!("\nOpen Positions:");
        for pos in &result.final_positions {
            println!(
                "  {} {}: size={}, entry={}, current={}, unrealized_pnl={}",
                if pos.size > Decimal::ZERO {
                    "LONG"
                } else {
                    "SHORT"
                },
                pos.symbol,
                pos.size,
                pos.entry_price,
                pos.current_price,
                pos.unrealized_pnl
            );
        }
    }

    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

fn print_usage() {
    println!("\nUsage: sim_environment <mode>");
    println!("\nModes:");
    println!("  backtest  - Run backtest with synthetic historical data");
    println!("  forward   - Run forward test (paper trading) with simulated live data");
    println!("  live      - Run forward test with REAL live exchange data");
    println!("  optimize  - Run strategy parameter optimization");
    println!("  replay    - Demo the replay engine");
    println!("  all       - Run all modes (excluding live)");
    println!("\nEnvironment Variables:");
    println!("  SIM_SYMBOLS           - Comma-separated symbols (default: BTC/USDT,ETH/USDT)");
    println!("  SIM_INITIAL_BALANCE   - Starting balance (default: 10000)");
    println!("  SIM_DURATION_SECS     - Forward test duration (default: 60)");
    println!();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         FKS Unified Simulation Environment                    ║");
    println!("║     Backtesting · Forward Testing · Optimization              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    match mode {
        "backtest" | "bt" => {
            run_backtest().await?;
        }
        "forward" | "ft" | "paper" => {
            run_forward_test().await?;
        }
        "live" | "real" => {
            run_live_forward_test().await?;
        }
        "optimize" | "opt" => {
            run_optimization().await?;
        }
        "replay" => {
            run_replay_demo().await?;
        }
        "all" => {
            run_backtest().await?;
            run_forward_test().await?;
            run_optimization().await?;
            run_replay_demo().await?;
            // Note: live mode is excluded from 'all' to avoid requiring network access
        }
        _ => {
            print_usage();
        }
    }

    println!("✅ Simulation completed successfully\n");

    Ok(())
}
