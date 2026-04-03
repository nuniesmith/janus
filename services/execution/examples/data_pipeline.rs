//! FKS Data Pipeline Example
//!
//! Demonstrates the full end-to-end data pipeline:
//! 1. Connect to live exchanges (Kraken, Binance, Bybit)
//! 2. Record market data to QuestDB via DataRecorder
//! 3. Load recorded data from QuestDB via ReplayEngine
//! 4. Run a backtest with the replayed data
//!
//! # Prerequisites
//!
//! - QuestDB running on localhost:9009 (ILP) and localhost:9000 (HTTP)
//! - Network access to exchanges (or use simulated mode)
//!
//! # Usage
//!
//! ```bash
//! # Record live data for 60 seconds
//! cargo run --release --example data_pipeline -- record
//!
//! # Replay recorded data
//! cargo run --release --example data_pipeline -- replay
//!
//! # Run full pipeline (record then replay)
//! cargo run --release --example data_pipeline -- full
//!
//! # Use simulated data (no exchange connection needed)
//! cargo run --release --example data_pipeline -- simulate
//! ```
//!
//! # Environment Variables
//!
//! - `QUESTDB_HOST` - QuestDB host (default: localhost)
//! - `QUESTDB_ILP_PORT` - QuestDB ILP port (default: 9009)
//! - `QUESTDB_HTTP_PORT` - QuestDB HTTP port (default: 9000)
//! - `RECORD_DURATION_SECS` - Recording duration (default: 60)
//! - `SYMBOLS` - Comma-separated symbols (default: BTC/USDT,ETH/USDT)

use chrono::{Duration, Utc};
use janus_execution::exchanges::{
    MarketDataAggregator, MarketDataProvider, binance::BinanceProvider, bybit::BybitProvider,
    kraken::KrakenProvider,
};
use janus_execution::sim::{
    config::SimConfig,
    data_feed::{AggregatedDataFeed, MarketEvent, TickData},
    data_recorder::{DataRecorder, RecorderConfig},
    environment::{Signal, SimEnvironment, Strategy},
    live_feed_bridge::{LiveFeedBridge, LiveFeedBridgeConfig},
    replay::{ReplayConfig, ReplayEngine, ReplaySpeed},
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tokio::time::{interval, sleep};
use tracing::{Level, info, warn};
use tracing_subscriber::FmtSubscriber;

/// Simple momentum strategy for demonstration
struct MomentumStrategy {
    name: String,
    /// Price history per symbol
    prices: HashMap<String, Vec<f64>>,
    /// Lookback period
    lookback: usize,
    /// Position size
    position_size: Decimal,
    /// Current positions
    positions: HashMap<String, bool>,
}

impl MomentumStrategy {
    fn new(lookback: usize, position_size: Decimal) -> Self {
        Self {
            name: format!("Momentum_{}", lookback),
            prices: HashMap::new(),
            lookback,
            position_size,
            positions: HashMap::new(),
        }
    }
}

impl Strategy for MomentumStrategy {
    fn on_event(&mut self, event: &MarketEvent) -> Vec<Signal> {
        let tick = match event {
            MarketEvent::Tick(t) => t,
            _ => return vec![Signal::None],
        };

        let mid_price = tick.mid_price();
        let price_f64 = mid_price.to_string().parse::<f64>().unwrap_or(0.0);

        if price_f64 <= 0.0 {
            return vec![Signal::None];
        }

        let prices = self.prices.entry(tick.symbol.clone()).or_default();
        prices.push(price_f64);

        // Keep only lookback + 1 prices
        while prices.len() > self.lookback + 1 {
            prices.remove(0);
        }

        // Need enough history
        if prices.len() < self.lookback {
            return vec![Signal::None];
        }

        // Calculate momentum (current price vs lookback periods ago)
        let current = prices.last().unwrap();
        let past = prices[0];
        let momentum = (current - past) / past;

        let has_position = *self.positions.get(&tick.symbol).unwrap_or(&false);

        // Simple momentum rules
        if momentum > 0.001 && !has_position {
            // Positive momentum, buy
            self.positions.insert(tick.symbol.clone(), true);
            return vec![Signal::Buy {
                symbol: tick.symbol.clone(),
                size: self.position_size,
                price: Some(mid_price),
                stop_loss: None,
                take_profit: None,
            }];
        } else if momentum < -0.001 && has_position {
            // Negative momentum, close position
            self.positions.insert(tick.symbol.clone(), false);
            return vec![Signal::Close {
                symbol: tick.symbol.clone(),
            }];
        }

        vec![Signal::None]
    }

    fn on_start(&mut self) {
        info!("Strategy {} started", self.name);
    }

    fn on_stop(&mut self) {
        info!("Strategy {} stopped", self.name);
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.prices.clear();
        self.positions.clear();
    }
}

/// Configuration from environment
struct PipelineConfig {
    questdb_host: String,
    questdb_ilp_port: u16,
    questdb_http_port: u16,
    record_duration_secs: u64,
    symbols: Vec<String>,
    use_simulated: bool,
}

impl PipelineConfig {
    fn from_env() -> Self {
        Self {
            questdb_host: env::var("QUESTDB_HOST").unwrap_or_else(|_| "localhost".to_string()),
            questdb_ilp_port: env::var("QUESTDB_ILP_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9009),
            questdb_http_port: env::var("QUESTDB_HTTP_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9000),
            record_duration_secs: env::var("RECORD_DURATION_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60),
            symbols: env::var("SYMBOLS")
                .unwrap_or_else(|_| "BTC/USDT,ETH/USDT".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            use_simulated: false,
        }
    }

    fn with_simulated(mut self) -> Self {
        self.use_simulated = true;
        self
    }
}

/// Record live market data to QuestDB
async fn run_record(config: &PipelineConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              FKS Data Pipeline - RECORD Mode                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    info!("Configuration:");
    info!(
        "  QuestDB: {}:{}",
        config.questdb_host, config.questdb_ilp_port
    );
    info!("  Duration: {} seconds", config.record_duration_secs);
    info!("  Symbols: {:?}", config.symbols);
    info!("  Simulated: {}", config.use_simulated);

    // Create data feed
    let data_feed = Arc::new(AggregatedDataFeed::new("pipeline_feed"));

    // Create and start recorder
    let recorder_config = RecorderConfig::new(&config.questdb_host, config.questdb_ilp_port)
        .with_batch_size(100)
        .with_flush_interval(StdDuration::from_secs(1))
        .with_table_prefix("fks_pipeline")
        .with_record_ticks(true)
        .with_record_trades(true);

    let mut recorder = DataRecorder::new(recorder_config);
    recorder.start().await?;
    recorder.subscribe_to_feed(&data_feed).await?;

    info!("DataRecorder started and subscribed to feed");

    if config.use_simulated {
        // Spawn simulated data generator
        let feed_clone = data_feed.clone();
        let symbols = config.symbols.clone();
        let duration = config.record_duration_secs;

        tokio::spawn(async move {
            generate_simulated_data(feed_clone, symbols, duration).await;
        });
    } else {
        // Connect to real exchanges
        let mut aggregator = MarketDataAggregator::new();
        connect_exchanges(&mut aggregator, &config.symbols).await?;

        // Create bridge to forward exchange events to our data feed
        let bridge_config = LiveFeedBridgeConfig::default()
            .with_normalize_symbols(true)
            .with_filter_invalid_ticks(true);

        let bridge = Arc::new(LiveFeedBridge::with_config(
            Arc::new(AggregatedDataFeed::with_sender(data_feed.sender())),
            bridge_config,
        ));
        bridge.connect_to_aggregator(&aggregator).await?;

        info!("Connected to exchanges via LiveFeedBridge");
    }

    // Record for specified duration
    info!(
        "\nRecording data for {} seconds...\n",
        config.record_duration_secs
    );

    let start = std::time::Instant::now();
    let mut last_report = start;
    let report_interval = StdDuration::from_secs(10);

    while start.elapsed() < StdDuration::from_secs(config.record_duration_secs) {
        sleep(StdDuration::from_secs(1)).await;

        if last_report.elapsed() >= report_interval {
            let stats = recorder.stats();
            info!(
                "Recording progress: {} events recorded, {} events/sec, buffer: {:.1}%",
                stats.events_recorded,
                stats.events_per_second() as u64,
                stats.buffer_utilization_pct()
            );
            last_report = std::time::Instant::now();
        }
    }

    // Stop and report
    recorder.flush().await?;
    recorder.stop().await?;

    let stats = recorder.stats();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    RECORDING COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Events Recorded:   {}", stats.events_recorded);
    println!("  - Ticks:         {}", stats.ticks_recorded);
    println!("  - Trades:        {}", stats.trades_recorded);
    println!("Events Dropped:    {}", stats.events_dropped);
    println!("Write Errors:      {}", stats.write_errors);
    println!("Bytes Written:     {} KB", stats.bytes_written / 1024);
    println!("Flush Count:       {}", stats.flush_count);
    println!(
        "Health:            {}",
        if stats.is_healthy() {
            "✅ Healthy"
        } else {
            "❌ Unhealthy"
        }
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Replay recorded data from QuestDB
async fn run_replay(config: &PipelineConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              FKS Data Pipeline - REPLAY Mode                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    info!("Configuration:");
    info!(
        "  QuestDB HTTP: {}:{}",
        config.questdb_host, config.questdb_http_port
    );
    info!("  Symbols: {:?}", config.symbols);

    // Calculate time range (last hour)
    let end_time = Utc::now();
    let start_time = end_time - Duration::hours(1);

    info!("Time range: {} to {}", start_time, end_time);

    // Create replay engine
    let replay_config = ReplayConfig::default()
        .with_speed(ReplaySpeed::Maximum)
        .with_time_range(Some(start_time), Some(end_time))
        .with_symbols(config.symbols.clone())
        .with_verbose(true);

    let mut engine = ReplayEngine::new(replay_config);

    // Load data from QuestDB
    info!("\nLoading data from QuestDB...");

    let count = engine
        .load_from_questdb(
            &config.questdb_host,
            config.questdb_http_port,
            "fks_pipeline_ticks",
            Some(start_time),
            Some(end_time),
            Some(config.symbols.clone()),
        )
        .await?;

    if count == 0 {
        warn!("No data found in QuestDB for the specified time range.");
        warn!("Make sure to run 'record' mode first to capture data.");
        return Ok(());
    }

    info!("Loaded {} events from QuestDB", count);

    // Subscribe to replay events
    let mut rx = engine.subscribe();

    // Start replay in background
    let engine = Arc::new(tokio::sync::Mutex::new(engine));
    let engine_clone = engine.clone();

    let replay_task = tokio::spawn(async move {
        let mut eng = engine_clone.lock().await;
        eng.start().await
    });

    // Process replayed events
    let mut event_count = 0u64;
    let mut symbols_seen: HashMap<String, u64> = HashMap::new();

    loop {
        match rx.recv().await {
            Ok(MarketEvent::EndOfData) => {
                info!("Replay complete - EndOfData received");
                break;
            }
            Ok(event) => {
                event_count += 1;
                if let Some(symbol) = event.symbol() {
                    *symbols_seen.entry(symbol.to_string()).or_insert(0) += 1;
                }

                if event_count.is_multiple_of(1000) {
                    info!("Processed {} events", event_count);
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                warn!("Lagged {} events", n);
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                break;
            }
        }
    }

    // Wait for replay task
    let _ = replay_task.await;

    // Report
    let stats = engine.lock().await.stats();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                     REPLAY COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Total Events:      {}", stats.total_events);
    println!("Events Replayed:   {}", stats.events_replayed);
    println!("Duration:          {:.2}s", stats.elapsed_seconds);
    println!(
        "Rate:              {:.0} events/sec",
        stats.events_per_second
    );
    println!("\nEvents by Symbol:");
    for (symbol, count) in &symbols_seen {
        println!("  {}: {}", symbol, count);
    }
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Run a backtest using replayed data
async fn run_backtest(config: &PipelineConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              FKS Data Pipeline - BACKTEST Mode                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Calculate time range (last hour)
    let end_time = Utc::now();
    let start_time = end_time - Duration::hours(1);

    info!("Time range: {} to {}", start_time, end_time);

    // Create sim environment
    let sim_config = SimConfig::backtest()
        .with_initial_balance(10000.0)
        .with_symbols(config.symbols.clone())
        .with_slippage_bps(5.0)
        .with_commission_bps(6.0)
        .build_unchecked();

    let mut env = SimEnvironment::new(sim_config).await?;
    env.initialize().await?;

    // Load data from QuestDB
    info!("\nLoading data from QuestDB for backtest...");

    let replay_config = ReplayConfig::default()
        .with_speed(ReplaySpeed::Maximum)
        .with_time_range(Some(start_time), Some(end_time))
        .with_symbols(config.symbols.clone());

    let mut replay_engine = ReplayEngine::new(replay_config);

    let count = replay_engine
        .load_from_questdb(
            &config.questdb_host,
            config.questdb_http_port,
            "fks_pipeline_ticks",
            Some(start_time),
            Some(end_time),
            Some(config.symbols.clone()),
        )
        .await?;

    if count == 0 {
        warn!("No data found. Run 'record' mode first.");
        return Ok(());
    }

    info!("Loaded {} events for backtest", count);

    // Create strategy
    let mut strategy = MomentumStrategy::new(20, dec!(0.1));

    // Subscribe to replay events and forward to sim environment
    let mut rx = replay_engine.subscribe();
    let data_feed = env.data_feed();

    // Start replay
    let replay_engine = Arc::new(tokio::sync::Mutex::new(replay_engine));
    let replay_clone = replay_engine.clone();

    let replay_task = tokio::spawn(async move {
        let mut eng = replay_clone.lock().await;
        let _ = eng.start().await;
    });

    // Process events through sim environment
    info!("\nRunning backtest...\n");

    let mut events_processed = 0u64;

    loop {
        match rx.recv().await {
            Ok(MarketEvent::EndOfData) => {
                break;
            }
            Ok(event) => {
                // Forward to sim environment's data feed
                data_feed.read().publish(event.clone());

                // Process through strategy
                let signals = strategy.on_event(&event);
                for signal in signals {
                    if signal.is_actionable() {
                        // In a real backtest, SimEnvironment.run_backtest handles this
                        // For this demo, we just log the signals
                        if events_processed.is_multiple_of(100) {
                            match &signal {
                                Signal::Buy { symbol, size, .. } => {
                                    info!("Signal: BUY {} size={}", symbol, size);
                                }
                                Signal::Sell { symbol, size, .. } => {
                                    info!("Signal: SELL {} size={}", symbol, size);
                                }
                                Signal::Close { symbol } => {
                                    info!("Signal: CLOSE {}", symbol);
                                }
                                _ => {}
                            }
                        }
                    }
                }

                events_processed += 1;
                if events_processed.is_multiple_of(5000) {
                    info!("Processed {} events", events_processed);
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                warn!("Lagged {} events", n);
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                break;
            }
        }
    }

    let _ = replay_task.await;

    // Report
    let replay_stats = replay_engine.lock().await.stats();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    BACKTEST COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Strategy:          {}", strategy.name());
    println!("Events Processed:  {}", events_processed);
    println!("Duration:          {:.2}s", replay_stats.elapsed_seconds);
    println!(
        "Rate:              {:.0} events/sec",
        replay_stats.events_per_second
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Run full pipeline: record then replay
async fn run_full_pipeline(config: &PipelineConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              FKS Data Pipeline - FULL Mode                    ║");
    println!("║          (Record → Replay → Backtest)                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Record
    info!("=== STEP 1: Recording Live Data ===\n");
    run_record(config).await?;

    // Brief pause
    info!("Waiting 2 seconds before replay...\n");
    sleep(StdDuration::from_secs(2)).await;

    // Step 2: Replay
    info!("=== STEP 2: Replaying Recorded Data ===\n");
    run_replay(config).await?;

    // Brief pause
    info!("Waiting 2 seconds before backtest...\n");
    sleep(StdDuration::from_secs(2)).await;

    // Step 3: Backtest
    info!("=== STEP 3: Running Backtest ===\n");
    run_backtest(config).await?;

    println!("\n✅ Full pipeline completed successfully!\n");

    Ok(())
}

/// Generate simulated market data
async fn generate_simulated_data(
    feed: Arc<AggregatedDataFeed>,
    symbols: Vec<String>,
    duration_secs: u64,
) {
    let mut prices: HashMap<String, f64> = HashMap::new();
    prices.insert("BTC/USDT".to_string(), 50000.0);
    prices.insert("ETH/USDT".to_string(), 3000.0);
    prices.insert("SOL/USDT".to_string(), 100.0);

    // Default prices for unknown symbols
    for symbol in &symbols {
        prices.entry(symbol.clone()).or_insert(1000.0);
    }

    let start = std::time::Instant::now();
    let mut tick_interval = interval(StdDuration::from_millis(100));
    let mut tick_count = 0u64;

    while start.elapsed() < StdDuration::from_secs(duration_secs) {
        tick_interval.tick().await;
        tick_count += 1;

        for symbol in &symbols {
            if let Some(price) = prices.get_mut(symbol) {
                // Simulate price movement (random walk)
                let change = ((tick_count as f64 * 17.0 + symbol.len() as f64 * 31.0).sin()
                    * 0.001
                    + (tick_count as f64 * 7.0).cos() * 0.0005)
                    * *price;
                *price = (*price + change).max(1.0);

                let spread = *price * 0.0001; // 1 bps spread

                let tick = TickData::new(
                    symbol,
                    "simulated",
                    Decimal::try_from(*price - spread).unwrap_or(dec!(0)),
                    Decimal::try_from(*price + spread).unwrap_or(dec!(0)),
                    dec!(1.0),
                    dec!(1.0),
                    Utc::now(),
                );

                feed.publish(MarketEvent::Tick(tick));
            }
        }
    }

    info!(
        "Simulated data generation complete: {} ticks",
        tick_count * symbols.len() as u64
    );
}

/// Connect to real exchanges
async fn connect_exchanges(
    aggregator: &mut MarketDataAggregator,
    symbols: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let symbols_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();

    // Kraken
    info!("Connecting to Kraken...");
    let kraken = Arc::new(KrakenProvider::new());
    match kraken.connect().await {
        Ok(_) => {
            if kraken.subscribe_ticker(&symbols_refs).await.is_ok() {
                info!("✅ Kraken connected");
                aggregator.add_provider(kraken);
            }
        }
        Err(e) => warn!("❌ Kraken connection failed: {}", e),
    }

    // Binance (subscribe before connect for combined streams)
    info!("Connecting to Binance...");
    let binance = Arc::new(BinanceProvider::new());
    let _ = binance.subscribe_ticker(&symbols_refs).await;
    match binance.connect().await {
        Ok(_) => {
            info!("✅ Binance connected");
            aggregator.add_provider(binance);
        }
        Err(e) => warn!("❌ Binance connection failed: {}", e),
    }

    // Bybit (Linear for bid/ask data)
    info!("Connecting to Bybit...");
    let bybit = Arc::new(BybitProvider::linear(false));
    match bybit.connect().await {
        Ok(_) => {
            let _ = bybit.subscribe_ticker(&symbols_refs).await;
            info!("✅ Bybit connected");
            aggregator.add_provider(bybit);
        }
        Err(e) => warn!("❌ Bybit connection failed: {}", e),
    }

    Ok(())
}

fn print_usage() {
    println!("\nUsage: data_pipeline <mode>");
    println!("\nModes:");
    println!("  record    - Record live market data to QuestDB");
    println!("  replay    - Replay recorded data from QuestDB");
    println!("  backtest  - Run backtest with replayed data");
    println!("  full      - Run full pipeline (record → replay → backtest)");
    println!("  simulate  - Use simulated data (no exchange connection)");
    println!("\nEnvironment Variables:");
    println!("  QUESTDB_HOST         - QuestDB host (default: localhost)");
    println!("  QUESTDB_ILP_PORT     - QuestDB ILP port (default: 9009)");
    println!("  QUESTDB_HTTP_PORT    - QuestDB HTTP port (default: 9000)");
    println!("  RECORD_DURATION_SECS - Recording duration (default: 60)");
    println!("  SYMBOLS              - Comma-separated symbols (default: BTC/USDT,ETH/USDT)");
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

    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    let config = PipelineConfig::from_env();

    match mode {
        "record" => run_record(&config).await?,
        "replay" => run_replay(&config).await?,
        "backtest" => run_backtest(&config).await?,
        "full" => run_full_pipeline(&config).await?,
        "simulate" => {
            let config = config.with_simulated();
            run_full_pipeline(&config).await?
        }
        _ => print_usage(),
    }

    Ok(())
}
