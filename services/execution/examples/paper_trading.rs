//! Paper Trading Example with Arbitrage Monitoring
//!
//! This example demonstrates how to run a paper trading session using Kraken
//! as the execution venue while monitoring arbitrage opportunities across
//! Bybit and Binance.
//!
//! # Features
//!
//! - Connects to Kraken, Bybit, and Binance WebSocket feeds
//! - Monitors cross-exchange price spreads
//! - Detects arbitrage opportunities
//! - Provides execution recommendations for Kraken
//! - Tracks execution quality metrics
//! - **Exposes Prometheus metrics on HTTP endpoint for monitoring**
//!
//! # Environment Variables
//!
//! Required:
//! - `KRAKEN_API_KEY`: Kraken API key (optional for market data only)
//! - `KRAKEN_API_SECRET`: Kraken API secret (optional for market data only)
//!
//! Optional:
//! - `KRAKEN_DRY_RUN`: Set to "true" for paper trading (default: true)
//! - `PAPER_TRADING_SYMBOLS`: Comma-separated list of symbols (default: BTC/USDT,ETH/USDT)
//! - `PAPER_TRADING_DURATION_SECS`: How long to run (default: 300 = 5 minutes)
//! - `MIN_SPREAD_PCT`: Minimum spread to report (default: 0.1)
//! - `USE_SIMULATED_DATA`: Set to "true" to use simulated data instead of real feeds
//! - `METRICS_PORT`: Port for Prometheus metrics endpoint (default: 8080)
//!
//! # Usage
//!
//! ```bash
//! # Run with real market data
//! cargo run --example paper_trading
//!
//! # Run with simulated data (no network required)
//! USE_SIMULATED_DATA=true cargo run --example paper_trading
//!
//! # Run for 10 minutes with specific symbols
//! PAPER_TRADING_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT \
//! PAPER_TRADING_DURATION_SECS=600 \
//! cargo run --example paper_trading
//!
//! # Scrape metrics during the run
//! curl http://localhost:8080/metrics
//! ```

use axum::{Router, routing::get};
use chrono::Utc;
use janus_execution::exchanges::provider::{MarketDataAggregator, MarketDataProvider};
use janus_execution::exchanges::{BinanceProvider, BybitProvider, KrakenProvider};
use janus_execution::execution::arbitrage::{
    ArbitrageConfig, ArbitrageEvent, ArbitrageMonitor, Exchange, ExchangePrice,
};
use janus_execution::execution::arbitrage_bridge::ArbitrageBridgeBuilder;
use janus_execution::execution::best_execution::{
    BestExecutionAnalyzer, ExecutionConfig, ExecutionRecommendation,
};
use janus_execution::execution::metrics::unified_prometheus_metrics;
use janus_execution::sim::sim_prometheus_metrics;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Paper trading statistics
#[derive(Debug, Default)]
struct PaperTradingStats {
    /// Total ticker updates received
    ticker_updates: AtomicU64,
    /// Arbitrage opportunities detected
    opportunities_detected: AtomicU64,
    /// Execute now recommendations
    execute_now_count: AtomicU64,
    /// Wait recommendations
    wait_count: AtomicU64,
    /// Avoid recommendations
    avoid_count: AtomicU64,
    /// Best spread seen (in basis points * 100 for atomic storage)
    best_spread_bps_x100: AtomicU64,
    /// Price updates per exchange
    exchange_updates: RwLock<HashMap<String, u64>>,
}

impl PaperTradingStats {
    fn new() -> Self {
        Self::default()
    }

    fn record_ticker(&self) {
        self.ticker_updates.fetch_add(1, Ordering::Relaxed);
    }

    fn record_opportunity(&self, spread_bps: Decimal) {
        self.opportunities_detected.fetch_add(1, Ordering::Relaxed);

        // Track best spread (store as bps * 100 to preserve some decimal precision)
        let spread_x100 = (spread_bps * Decimal::from(100))
            .to_string()
            .parse::<u64>()
            .unwrap_or(0);

        self.best_spread_bps_x100
            .fetch_max(spread_x100, Ordering::Relaxed);
    }

    fn record_recommendation(&self, rec: &ExecutionRecommendation) {
        match rec {
            ExecutionRecommendation::ExecuteNow => {
                self.execute_now_count.fetch_add(1, Ordering::Relaxed);
            }
            ExecutionRecommendation::Wait => {
                self.wait_count.fetch_add(1, Ordering::Relaxed);
            }
            ExecutionRecommendation::Avoid => {
                self.avoid_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    async fn record_exchange_update(&self, exchange: &str) {
        let mut updates = self.exchange_updates.write().await;
        *updates.entry(exchange.to_string()).or_insert(0) += 1;
    }

    fn summary(&self) -> String {
        let best_spread = self.best_spread_bps_x100.load(Ordering::Relaxed) as f64 / 100.0;

        format!(
            "Paper Trading Summary:\n\
             ─────────────────────────────────────\n\
             Total ticker updates:     {}\n\
             Opportunities detected:   {}\n\
             Best spread (bps):        {:.2}\n\
             Execute now suggestions:  {}\n\
             Wait suggestions:         {}\n\
             Avoid suggestions:        {}",
            self.ticker_updates.load(Ordering::Relaxed),
            self.opportunities_detected.load(Ordering::Relaxed),
            best_spread,
            self.execute_now_count.load(Ordering::Relaxed),
            self.wait_count.load(Ordering::Relaxed),
            self.avoid_count.load(Ordering::Relaxed),
        )
    }
}

/// Configuration for the paper trading session
#[derive(Debug, Clone)]
struct PaperTradingConfig {
    /// Symbols to monitor
    symbols: Vec<String>,
    /// Duration to run
    duration: Duration,
    /// Minimum spread to report (as percentage)
    min_spread_pct: Decimal,
    /// Whether this is a dry run (no real orders)
    dry_run: bool,
    /// Analysis interval
    analysis_interval: Duration,
    /// Use simulated data instead of real feeds
    use_simulated_data: bool,
    /// Port for metrics HTTP server
    metrics_port: u16,
}

impl PaperTradingConfig {
    fn from_env() -> Self {
        let symbols: Vec<String> = std::env::var("PAPER_TRADING_SYMBOLS")
            .unwrap_or_else(|_| "BTC/USDT,ETH/USDT".to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let duration_secs: u64 = std::env::var("PAPER_TRADING_DURATION_SECS")
            .unwrap_or_else(|_| "300".to_string())
            .parse()
            .unwrap_or(300);

        let min_spread_pct: Decimal = std::env::var("MIN_SPREAD_PCT")
            .unwrap_or_else(|_| "0.1".to_string())
            .parse()
            .unwrap_or_else(|_| Decimal::new(1, 1));

        let dry_run = std::env::var("KRAKEN_DRY_RUN")
            .unwrap_or_else(|_| "true".to_string())
            .to_lowercase()
            == "true";

        let use_simulated_data = std::env::var("USE_SIMULATED_DATA")
            .unwrap_or_else(|_| "false".to_string())
            .to_lowercase()
            == "true";

        let metrics_port: u16 = std::env::var("METRICS_PORT")
            .unwrap_or_else(|_| "8080".to_string())
            .parse()
            .unwrap_or(8080);

        Self {
            symbols,
            duration: Duration::from_secs(duration_secs),
            min_spread_pct,
            dry_run,
            analysis_interval: Duration::from_secs(10),
            use_simulated_data,
            metrics_port,
        }
    }
}

/// Metrics HTTP handler - returns Prometheus-format metrics
async fn metrics_handler() -> String {
    let now = chrono::Utc::now().timestamp_millis();
    let mut output = String::new();

    // Service info
    output.push_str("# HELP paper_trading_info Paper trading service information\n");
    output.push_str("# TYPE paper_trading_info gauge\n");
    output.push_str(&format!(
        "paper_trading_info{{version=\"{}\"}} 1\n",
        env!("CARGO_PKG_VERSION")
    ));
    output.push('\n');

    output.push_str("# HELP paper_trading_timestamp_ms Current timestamp in milliseconds\n");
    output.push_str("# TYPE paper_trading_timestamp_ms gauge\n");
    output.push_str(&format!("paper_trading_timestamp_ms {}\n", now));
    output.push('\n');

    // Unified execution and exchange metrics
    output.push_str(&unified_prometheus_metrics());
    output.push('\n');

    // Simulation metrics (DataRecorder, LiveFeedBridge, Replay)
    let sim_metrics = sim_prometheus_metrics();
    if !sim_metrics.is_empty() {
        output.push_str(
            "# ============================================================================\n",
        );
        output.push_str("# Simulation Metrics (Recorder, Bridge, Replay)\n");
        output.push_str(
            "# ============================================================================\n",
        );
        output.push_str(&sim_metrics);
    }

    output
}

/// Health check handler
async fn health_handler() -> &'static str {
    "OK"
}

/// Start the metrics HTTP server with retry and port fallback
async fn start_metrics_server(port: u16) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/", get(health_handler));

    // Try the requested port first, then try alternative ports
    let ports_to_try = [port, port + 1, port + 2, port + 10, 0]; // 0 = OS assigns free port

    let mut listener = None;
    let mut bound_addr = None;

    for try_port in ports_to_try {
        let addr = SocketAddr::from(([0, 0, 0, 0], try_port));
        match TcpListener::bind(addr).await {
            Ok(l) => {
                bound_addr = Some(l.local_addr().unwrap_or(addr));
                listener = Some(l);
                if try_port != port && try_port != 0 {
                    warn!(
                        "Port {} was in use, metrics server bound to port {} instead",
                        port,
                        bound_addr.unwrap().port()
                    );
                }
                break;
            }
            Err(e) => {
                if try_port == 0 {
                    // Even OS-assigned port failed, give up
                    error!("Failed to bind metrics server to any port: {}", e);
                    return Err(e.into());
                }
                debug!("Port {} unavailable: {}, trying next...", try_port, e);
            }
        }
    }

    let listener = listener.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::AddrInUse, "All ports unavailable")
    })?;

    let addr = bound_addr.unwrap();
    info!("📊 Metrics server listening on http://{}/metrics", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    init_logging();

    let config = PaperTradingConfig::from_env();

    info!("╔══════════════════════════════════════════════════════════╗");
    info!("║       FKS Paper Trading with Arbitrage Monitoring        ║");
    info!("╚══════════════════════════════════════════════════════════╝");
    info!("");
    info!("Configuration:");
    info!("  Symbols:        {:?}", config.symbols);
    info!("  Duration:       {} seconds", config.duration.as_secs());
    info!("  Min spread:     {}%", config.min_spread_pct);
    info!("  Dry run:        {}", config.dry_run);
    info!("  Simulated data: {}", config.use_simulated_data);
    info!("  Metrics port:   {}", config.metrics_port);
    info!("");

    // Start metrics HTTP server in background
    let metrics_port = config.metrics_port;
    let metrics_handle = tokio::spawn(async move {
        if let Err(e) = start_metrics_server(metrics_port).await {
            error!("Metrics server error: {}", e);
        }
    });

    if !config.dry_run {
        warn!("⚠️  DRY RUN IS DISABLED - REAL ORDERS MAY BE PLACED!");
        warn!("    Set KRAKEN_DRY_RUN=true to enable paper trading mode");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    // Initialize components
    info!("Initializing components...");

    // Create arbitrage monitor
    let arb_config = ArbitrageConfig {
        min_spread_pct: config.min_spread_pct,
        symbols: config.symbols.clone(),
        ..Default::default()
    };
    let monitor = Arc::new(ArbitrageMonitor::new(arb_config));

    // Create best execution analyzer
    let exec_config = ExecutionConfig::default();
    let analyzer = Arc::new(BestExecutionAnalyzer::new(exec_config));

    // Create market data aggregator
    let mut aggregator = MarketDataAggregator::new();

    // Statistics tracker
    let stats = Arc::new(PaperTradingStats::new());

    // Track which providers were connected
    let mut connected_providers: Vec<&str> = Vec::new();

    if config.use_simulated_data {
        info!("Using simulated data mode - no real exchange connections");
    } else {
        // Create and connect providers
        info!("Connecting to exchanges...");

        // Kraken provider
        let kraken_provider = Arc::new(KrakenProvider::new());
        match kraken_provider.connect().await {
            Ok(_) => {
                info!("✅ Kraken connected");
                let symbols_refs: Vec<&str> = config.symbols.iter().map(|s| s.as_str()).collect();
                match kraken_provider.subscribe_ticker(&symbols_refs).await {
                    Err(e) => {
                        warn!("Failed to subscribe to Kraken tickers: {}", e);
                    }
                    _ => {
                        connected_providers.push("Kraken");
                        aggregator.add_provider(kraken_provider);
                    }
                }
            }
            Err(e) => {
                warn!("❌ Failed to connect to Kraken: {}", e);
            }
        }

        // Binance provider
        // Note: Binance uses combined streams via URL, so subscribe BEFORE connect
        let binance_provider = Arc::new(BinanceProvider::new());
        let symbols_refs: Vec<&str> = config.symbols.iter().map(|s| s.as_str()).collect();
        // Queue subscriptions first (will be included in connection URL)
        if let Err(e) = binance_provider.subscribe_ticker(&symbols_refs).await {
            warn!("Failed to queue Binance ticker subscriptions: {}", e);
        }
        // Now connect (subscriptions will be applied via URL params)
        match binance_provider.connect().await {
            Ok(_) => {
                info!("✅ Binance connected");
                connected_providers.push("Binance");
                aggregator.add_provider(binance_provider);
            }
            Err(e) => {
                warn!("❌ Failed to connect to Binance: {}", e);
            }
        }

        // Bybit provider (Linear perpetuals - includes bid/ask in ticker data)
        // Note: Spot ticker doesn't include bid/ask, only Linear/Inverse do
        let bybit_provider = Arc::new(BybitProvider::linear(false));
        match bybit_provider.connect().await {
            Ok(_) => {
                info!("✅ Bybit connected");
                let symbols_refs: Vec<&str> = config.symbols.iter().map(|s| s.as_str()).collect();
                let _ = bybit_provider.subscribe_ticker(&symbols_refs).await;
                connected_providers.push("Bybit");
                aggregator.add_provider(bybit_provider);
            }
            Err(e) => {
                warn!("❌ Failed to connect to Bybit: {}", e);
            }
        }

        if connected_providers.is_empty() {
            warn!("No exchanges connected! Falling back to simulated data mode.");
        } else {
            info!(
                "Connected to {} exchange(s): {:?}",
                connected_providers.len(),
                connected_providers
            );
        }
    }

    let aggregator = Arc::new(aggregator);

    // Create the bridge (connects aggregator -> monitor -> analyzer)
    let bridge = ArbitrageBridgeBuilder::new()
        .aggregator(aggregator.clone())
        .monitor(monitor.clone())
        .best_execution(analyzer.clone())
        .build()?;

    // Subscribe to arbitrage events
    let mut event_rx = monitor.subscribe();

    // Start the bridge
    bridge.start().await?;
    info!("Arbitrage bridge started");

    // Start the monitor
    monitor.start().await?;
    info!("Arbitrage monitor started");

    // Spawn event handler
    let stats_clone = stats.clone();
    let event_handle = tokio::spawn(async move {
        while let Ok(event) = event_rx.recv().await {
            match event {
                ArbitrageEvent::Opportunity(opp) => {
                    stats_clone.record_opportunity(opp.spread_pct);

                    info!(
                        "🎯 ARBITRAGE OPPORTUNITY: {} - Buy on {} @ {}, Sell on {} @ {} | Spread: {:.2} bps | Est. Profit: {}",
                        opp.symbol,
                        opp.buy_exchange,
                        opp.buy_price,
                        opp.sell_exchange,
                        opp.sell_price,
                        opp.spread_pct * Decimal::from(100),
                        opp.estimated_profit,
                    );
                }
                ArbitrageEvent::SpreadAlert {
                    symbol,
                    spread_pct,
                    message,
                } => {
                    warn!(
                        "⚠️  SPREAD ALERT: {} - {:.2}% - {}",
                        symbol, spread_pct, message
                    );
                }
                ArbitrageEvent::Recommendation(rec) => {
                    info!(
                        "📊 Kraken Recommendation: {} - {:?} | Deviation: {:.2}% | Strength: {:.2}",
                        rec.symbol, rec.action, rec.deviation_pct, rec.strength
                    );
                }
                ArbitrageEvent::PriceUpdate(comparison) => {
                    // Just count these, don't log every one
                    if let Some(spread_pct) = comparison.cross_spread_pct
                        && spread_pct > Decimal::new(5, 2)
                    {
                        // > 0.05%
                        info!(
                            "💹 Price update: {} - Cross spread: {:.2} bps",
                            comparison.symbol,
                            spread_pct * Decimal::from(100)
                        );
                    }
                }
                ArbitrageEvent::Started => {
                    info!("✅ Arbitrage monitor started");
                }
                ArbitrageEvent::Stopped => {
                    info!("🛑 Arbitrage monitor stopped");
                }
                ArbitrageEvent::Error(msg) => {
                    error!("❌ Arbitrage error: {}", msg);
                }
            }
        }
    });

    // Spawn analysis task
    let analyzer_clone = analyzer.clone();
    let stats_clone = stats.clone();
    let symbols = config.symbols.clone();
    let analysis_interval = config.analysis_interval;

    let analysis_handle = tokio::spawn(async move {
        let mut ticker = interval(analysis_interval);

        loop {
            ticker.tick().await;

            for symbol in &symbols {
                // Analyze buy opportunity
                if let Some(analysis) = analyzer_clone.analyze_buy(symbol, Decimal::new(1, 2)).await
                {
                    stats_clone.record_recommendation(&analysis.recommendation);

                    let emoji = match analysis.recommendation {
                        ExecutionRecommendation::ExecuteNow => "🟢",
                        ExecutionRecommendation::Acceptable => "🟡",
                        ExecutionRecommendation::Wait => "🟠",
                        ExecutionRecommendation::UseLimitOrder => "📝",
                        ExecutionRecommendation::Avoid => "🔴",
                    };

                    info!(
                        "{} {} BUY Analysis: Score={:.0}/100, Rec={}, Price={}, Est.Cost={:.1}bps",
                        emoji,
                        symbol,
                        analysis.score,
                        analysis.recommendation,
                        analysis.kraken_price,
                        analysis.estimated_cost_bps,
                    );

                    if !analysis.reasons.is_empty() {
                        for reason in &analysis.reasons {
                            info!("   └─ {}", reason);
                        }
                    }
                }
            }
        }
    });

    // Spawn simulated price feed if using simulated data or if no real providers connected
    let price_feed_handle = if config.use_simulated_data || connected_providers.is_empty() {
        let monitor_clone = monitor.clone();
        let stats_clone = stats.clone();
        let symbols = config.symbols.clone();

        Some(tokio::spawn(async move {
            let mut ticker = interval(Duration::from_millis(500));
            let mut iteration = 0u64;

            info!("📡 Starting simulated price feed...");

            loop {
                ticker.tick().await;
                iteration += 1;

                for symbol in &symbols {
                    // Simulate price data from each exchange with slight variations
                    let base_price = match symbol.as_str() {
                        "BTC/USDT" => Decimal::from(67500),
                        "ETH/USDT" => Decimal::from(3500),
                        "SOL/USDT" => Decimal::from(180),
                        _ => Decimal::from(100),
                    };

                    // Add some variation to simulate real market conditions
                    let variation = Decimal::new((iteration % 100) as i64 - 50, 2); // -0.50 to +0.50

                    // Kraken price (baseline)
                    let kraken_mid = base_price + variation;
                    let kraken_spread = base_price * Decimal::new(1, 3); // 0.1% spread
                    let kraken_price = ExchangePrice::new(
                        Exchange::Kraken,
                        symbol.clone(),
                        kraken_mid - kraken_spread / Decimal::from(2),
                        kraken_mid + kraken_spread / Decimal::from(2),
                        Decimal::from(10),
                        Decimal::from(10),
                    );

                    // Bybit price (slightly different)
                    let bybit_offset = Decimal::new((iteration % 50) as i64 - 25, 2);
                    let bybit_mid = base_price + variation + bybit_offset;
                    let bybit_spread = base_price * Decimal::new(8, 4); // 0.08% spread
                    let bybit_price = ExchangePrice::new(
                        Exchange::Bybit,
                        symbol.clone(),
                        bybit_mid - bybit_spread / Decimal::from(2),
                        bybit_mid + bybit_spread / Decimal::from(2),
                        Decimal::from(15),
                        Decimal::from(15),
                    );

                    // Binance price (slightly different again)
                    let binance_offset = Decimal::new((iteration % 30) as i64 - 15, 2);
                    let binance_mid = base_price + variation + binance_offset;
                    let binance_spread = base_price * Decimal::new(5, 4); // 0.05% spread
                    let binance_price = ExchangePrice::new(
                        Exchange::Binance,
                        symbol.clone(),
                        binance_mid - binance_spread / Decimal::from(2),
                        binance_mid + binance_spread / Decimal::from(2),
                        Decimal::from(20),
                        Decimal::from(20),
                    );

                    // Update monitor with prices
                    monitor_clone.update_price(kraken_price).await;
                    monitor_clone.update_price(bybit_price).await;
                    monitor_clone.update_price(binance_price).await;

                    stats_clone.record_ticker();
                    stats_clone.record_exchange_update("Kraken").await;
                    stats_clone.record_exchange_update("Bybit").await;
                    stats_clone.record_exchange_update("Binance").await;
                }
            }
        }))
    } else {
        None
    };

    // Run for specified duration
    info!("");
    info!("📈 Paper trading session started at {}", Utc::now());
    info!(
        "   Session will run for {} seconds",
        config.duration.as_secs()
    );
    info!("   Press Ctrl+C to stop early");
    info!("");

    // Wait for duration or Ctrl+C
    tokio::select! {
        _ = tokio::time::sleep(config.duration) => {
            info!("Session duration completed");
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received Ctrl+C, shutting down...");
        }
    }

    // Cleanup
    info!("");
    info!("Shutting down...");

    // Stop components
    monitor.stop().await;
    bridge.stop().await;

    // Disconnect providers
    if !config.use_simulated_data {
        aggregator.disconnect_all().await.ok();
    }

    // Abort background tasks
    event_handle.abort();
    analysis_handle.abort();
    metrics_handle.abort();
    if let Some(handle) = price_feed_handle {
        handle.abort();
    }

    // Print final statistics
    info!("");
    info!("{}", stats.summary());
    info!("");

    // Print exchange update counts
    let exchange_updates = stats.exchange_updates.read().await;
    if !exchange_updates.is_empty() {
        info!("Exchange Update Counts:");
        for (exchange, count) in exchange_updates.iter() {
            info!("  {}: {}", exchange, count);
        }
        info!("");
    }

    // Print monitor summary
    let monitor_summary = monitor.summary().await;
    info!("Arbitrage Monitor Summary:");
    info!(
        "  Monitored symbols:         {}",
        monitor_summary.monitored_symbols
    );
    info!(
        "  Active opportunities:      {}",
        monitor_summary.active_opportunities
    );
    info!(
        "  Actionable opportunities:  {}",
        monitor_summary.actionable_opportunities
    );
    if let Some(best) = monitor_summary.best_opportunity {
        info!("  Best opportunity:");
        info!("    Symbol:       {}", best.symbol);
        info!(
            "    Buy on:       {} @ {}",
            best.buy_exchange, best.buy_price
        );
        info!(
            "    Sell on:      {} @ {}",
            best.sell_exchange, best.sell_price
        );
        info!(
            "    Spread:       {:.2} bps",
            best.spread_pct * Decimal::from(100)
        );
        info!("    Est. profit:  {}", best.estimated_profit);
    }

    info!("");
    info!("Paper trading session ended at {}", Utc::now());
    info!("═══════════════════════════════════════════════════════════");

    Ok(())
}

/// Initialize logging
fn init_logging() {
    use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info,fks_execution=debug"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(false)
                .with_level(true)
                .with_ansi(true)
                .compact(),
        )
        .init();
}
