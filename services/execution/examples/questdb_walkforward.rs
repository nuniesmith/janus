//! QuestDB Walk-Forward Backtest Example
//!
//! This example demonstrates how to run walk-forward parameter optimization
//! using market data recorded in QuestDB.
//!
//! ## Prerequisites
//!
//! 1. QuestDB running on localhost:9000 (or set QUESTDB_HOST/QUESTDB_PORT)
//! 2. Market data recorded in `fks_ticks` and `fks_trades` tables
//!    (use the data_pipeline example to record data first)
//!
//! ## Usage
//!
//! ```bash
//! # Run with default settings (requires QuestDB with data)
//! cargo run --example questdb_walkforward
//!
//! # Specify custom QuestDB host and symbol
//! QUESTDB_HOST=192.168.1.100 QUESTDB_PORT=9000 \
//! SYMBOL=ETH/USDT \
//! cargo run --example questdb_walkforward
//!
//! # Run with verbose output
//! RUST_LOG=info cargo run --example questdb_walkforward
//! ```
//!
//! ## Walk-Forward Analysis
//!
//! Walk-forward optimization helps detect overfitting by:
//! 1. Dividing data into rolling windows
//! 2. Optimizing parameters on in-sample (IS) data
//! 3. Testing best parameters on out-of-sample (OOS) data
//! 4. Measuring "efficiency" = OOS performance / IS performance
//!
//! A robust strategy should show efficiency > 50-70%.

use chrono::{Duration, Utc};
use janus_execution::sim::data_feed::{TickData, TradeData};
use janus_execution::sim::optimization::{OptimizationMetric, ParameterRange};
use janus_execution::sim::questdb_backtest::{
    BacktestState, EventBasedStrategy, QuestDBBacktestConfig, QuestDBWalkForwardRunner, SignalType,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

// ============================================================================
// Moving Average Crossover Strategy
// ============================================================================

/// Simple moving average crossover strategy
///
/// Goes long when fast MA crosses above slow MA, exits when it crosses below.
#[derive(Clone)]
struct MACrossoverStrategy {
    /// Fast moving average period
    fast_period: usize,
    /// Slow moving average period
    slow_period: usize,
    /// Position size per trade
    position_size: Decimal,
    /// Symbol to trade
    symbol: String,
    /// Previous fast MA value (for crossover detection)
    prev_fast_ma: Option<Decimal>,
    /// Previous slow MA value (for crossover detection)
    prev_slow_ma: Option<Decimal>,
}

impl MACrossoverStrategy {
    fn new(fast_period: usize, slow_period: usize, symbol: &str) -> Self {
        Self {
            fast_period,
            slow_period,
            position_size: dec!(0.1),
            symbol: symbol.to_string(),
            prev_fast_ma: None,
            prev_slow_ma: None,
        }
    }

    fn check_crossover(&mut self, state: &BacktestState) -> Option<SignalType> {
        // Calculate current MAs
        let fast_ma = state.sma(&self.symbol, self.fast_period)?;
        let slow_ma = state.sma(&self.symbol, self.slow_period)?;

        let signal =
            if let (Some(prev_fast), Some(prev_slow)) = (self.prev_fast_ma, self.prev_slow_ma) {
                // Check for crossover
                if prev_fast <= prev_slow && fast_ma > slow_ma {
                    // Bullish crossover - go long
                    let current_pos = state
                        .positions
                        .get(&self.symbol)
                        .map(|p| p.size)
                        .unwrap_or_default();

                    if current_pos <= Decimal::ZERO {
                        Some(SignalType::Buy {
                            size: self.position_size,
                        })
                    } else {
                        None
                    }
                } else if prev_fast >= prev_slow && fast_ma < slow_ma {
                    // Bearish crossover - exit long
                    let current_pos = state
                        .positions
                        .get(&self.symbol)
                        .map(|p| p.size)
                        .unwrap_or_default();

                    if current_pos > Decimal::ZERO {
                        Some(SignalType::Sell { size: current_pos })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

        // Update previous values
        self.prev_fast_ma = Some(fast_ma);
        self.prev_slow_ma = Some(slow_ma);

        signal
    }
}

impl EventBasedStrategy for MACrossoverStrategy {
    fn on_tick(&mut self, tick: &TickData, state: &mut BacktestState) -> Option<SignalType> {
        // Only process our symbol
        if tick.symbol != self.symbol {
            return None;
        }

        self.check_crossover(state)
    }

    fn on_trade(&mut self, trade: &TradeData, state: &mut BacktestState) -> Option<SignalType> {
        // Only process our symbol
        if trade.symbol != self.symbol {
            return None;
        }

        self.check_crossover(state)
    }

    fn with_params(
        &self,
        params: &HashMap<String, janus_execution::sim::optimization::ParameterValue>,
    ) -> Box<dyn EventBasedStrategy> {
        let fast = params
            .get("fast_period")
            .and_then(|v| v.as_int())
            .unwrap_or(self.fast_period as i64) as usize;

        let slow = params
            .get("slow_period")
            .and_then(|v| v.as_int())
            .unwrap_or(self.slow_period as i64) as usize;

        Box::new(MACrossoverStrategy::new(fast, slow, &self.symbol))
    }

    fn name(&self) -> &str {
        "MA Crossover"
    }
}

// ============================================================================
// RSI Mean Reversion Strategy
// ============================================================================

/// RSI-based mean reversion strategy
///
/// Buys when RSI drops below oversold threshold, sells when above overbought.
#[derive(Clone)]
struct RSIMeanReversionStrategy {
    /// RSI period
    rsi_period: usize,
    /// Oversold threshold (buy signal)
    oversold: Decimal,
    /// Overbought threshold (sell signal)
    overbought: Decimal,
    /// Position size
    position_size: Decimal,
    /// Symbol to trade
    symbol: String,
    /// Price changes for RSI calculation
    price_changes: Vec<Decimal>,
    /// Last price
    last_price: Option<Decimal>,
}

impl RSIMeanReversionStrategy {
    fn new(rsi_period: usize, oversold: Decimal, overbought: Decimal, symbol: &str) -> Self {
        Self {
            rsi_period,
            oversold,
            overbought,
            position_size: dec!(0.1),
            symbol: symbol.to_string(),
            price_changes: Vec::new(),
            last_price: None,
        }
    }

    fn calculate_rsi(&self) -> Option<Decimal> {
        if self.price_changes.len() < self.rsi_period {
            return None;
        }

        let recent: Vec<_> = self
            .price_changes
            .iter()
            .rev()
            .take(self.rsi_period)
            .cloned()
            .collect();

        let gains: Decimal = recent.iter().filter(|c| **c > Decimal::ZERO).cloned().sum();
        let losses: Decimal = recent
            .iter()
            .filter(|c| **c < Decimal::ZERO)
            .map(|c| c.abs())
            .sum();

        let period = Decimal::from(self.rsi_period);
        let avg_gain = gains / period;
        let avg_loss = losses / period;

        if avg_loss == Decimal::ZERO {
            return Some(dec!(100));
        }

        let rs = avg_gain / avg_loss;
        let rsi = dec!(100) - (dec!(100) / (Decimal::ONE + rs));

        Some(rsi)
    }

    fn process_price(&mut self, price: Decimal, state: &BacktestState) -> Option<SignalType> {
        // Calculate price change
        if let Some(last) = self.last_price {
            let change = price - last;
            self.price_changes.push(change);
            if self.price_changes.len() > self.rsi_period * 2 {
                self.price_changes.remove(0);
            }
        }
        self.last_price = Some(price);

        // Calculate RSI and generate signal
        let rsi = self.calculate_rsi()?;
        let current_pos = state
            .positions
            .get(&self.symbol)
            .map(|p| p.size)
            .unwrap_or_default();

        if rsi < self.oversold && current_pos <= Decimal::ZERO {
            // Oversold - buy
            Some(SignalType::Buy {
                size: self.position_size,
            })
        } else if rsi > self.overbought && current_pos > Decimal::ZERO {
            // Overbought - sell
            Some(SignalType::Sell { size: current_pos })
        } else {
            None
        }
    }
}

impl EventBasedStrategy for RSIMeanReversionStrategy {
    fn on_tick(&mut self, tick: &TickData, state: &mut BacktestState) -> Option<SignalType> {
        if tick.symbol != self.symbol {
            return None;
        }
        self.process_price(tick.mid_price(), state)
    }

    fn on_trade(&mut self, trade: &TradeData, state: &mut BacktestState) -> Option<SignalType> {
        if trade.symbol != self.symbol {
            return None;
        }
        self.process_price(trade.price, state)
    }

    fn with_params(
        &self,
        params: &HashMap<String, janus_execution::sim::optimization::ParameterValue>,
    ) -> Box<dyn EventBasedStrategy> {
        let period = params
            .get("rsi_period")
            .and_then(|v| v.as_int())
            .unwrap_or(self.rsi_period as i64) as usize;

        let oversold = params
            .get("oversold")
            .and_then(|v| v.as_float())
            .map(|f| Decimal::try_from(f).unwrap_or(self.oversold))
            .unwrap_or(self.oversold);

        let overbought = params
            .get("overbought")
            .and_then(|v| v.as_float())
            .map(|f| Decimal::try_from(f).unwrap_or(self.overbought))
            .unwrap_or(self.overbought);

        Box::new(RSIMeanReversionStrategy::new(
            period,
            oversold,
            overbought,
            &self.symbol,
        ))
    }

    fn name(&self) -> &str {
        "RSI Mean Reversion"
    }
}

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    questdb_host: String,
    questdb_port: u16,
    symbol: String,
    ticks_table: String,
    trades_table: String,
    initial_balance: Decimal,
    num_windows: usize,
    in_sample_pct: f64,
}

impl Config {
    fn from_env() -> Self {
        Self {
            questdb_host: std::env::var("QUESTDB_HOST").unwrap_or_else(|_| "localhost".to_string()),
            questdb_port: std::env::var("QUESTDB_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9000),
            symbol: std::env::var("SYMBOL").unwrap_or_else(|_| "BTC/USDT".to_string()),
            ticks_table: std::env::var("TICKS_TABLE").unwrap_or_else(|_| "fks_ticks".to_string()),
            trades_table: std::env::var("TRADES_TABLE")
                .unwrap_or_else(|_| "fks_trades".to_string()),
            initial_balance: Decimal::from(10_000),
            num_windows: std::env::var("NUM_WINDOWS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            in_sample_pct: std::env::var("IN_SAMPLE_PCT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7),
        }
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().compact())
        .init();

    let config = Config::from_env();

    info!("╔══════════════════════════════════════════════════════════╗");
    info!("║       QuestDB Walk-Forward Backtest Example              ║");
    info!("╚══════════════════════════════════════════════════════════╝");
    info!("");
    info!("Configuration:");
    info!(
        "  QuestDB:        {}:{}",
        config.questdb_host, config.questdb_port
    );
    info!("  Symbol:         {}", config.symbol);
    info!("  Ticks table:    {}", config.ticks_table);
    info!("  Trades table:   {}", config.trades_table);
    info!("  Initial balance: ${}", config.initial_balance);
    info!("  Walk-forward windows: {}", config.num_windows);
    info!("  In-sample %:    {:.0}%", config.in_sample_pct * 100.0);
    info!("");

    // Build QuestDB configuration
    let questdb_config = QuestDBBacktestConfig::new(&config.questdb_host, config.questdb_port)
        .with_ticks_table(&config.ticks_table)
        .with_trades_table(&config.trades_table)
        .with_symbols(vec![config.symbol.clone()])
        .with_initial_balance(config.initial_balance)
        .with_commission_bps(dec!(0.10)) // 0.10% commission
        .with_slippage_bps(dec!(0.05)); // 0.05% slippage

    // ========================================================================
    // Strategy 1: MA Crossover
    // ========================================================================

    info!("═══════════════════════════════════════════════════════════");
    info!("Strategy 1: Moving Average Crossover");
    info!("═══════════════════════════════════════════════════════════");

    let ma_strategy = MACrossoverStrategy::new(10, 30, &config.symbol);

    let mut ma_runner = match QuestDBWalkForwardRunner::builder()
        .questdb_config(questdb_config.clone())
        .windows(config.num_windows)
        .in_sample_pct(config.in_sample_pct)
        .rolling()
        .parameter(ParameterRange::int("fast_period", 5, 20, 5)) // 5, 10, 15, 20
        .parameter(ParameterRange::int("slow_period", 20, 50, 10)) // 20, 30, 40, 50
        .metric(OptimizationMetric::SharpeRatio)
        .verbose(true)
        .build()
    {
        Ok(runner) => runner,
        Err(e) => {
            error!("Failed to build MA runner: {}", e);
            return Err(e.into());
        }
    };

    info!("Discovering data time range from QuestDB...");

    match ma_runner.discover_time_range().await {
        Ok((start, end)) => {
            let duration = end - start;
            info!(
                "Data range: {} to {} ({} days)",
                start.format("%Y-%m-%d %H:%M:%S"),
                end.format("%Y-%m-%d %H:%M:%S"),
                duration.num_days()
            );
        }
        Err(e) => {
            warn!("Could not discover time range: {}", e);
            warn!("Make sure QuestDB is running and contains data in the specified tables.");
            warn!("");
            warn!("To record data, run:");
            warn!("  cargo run --example data_pipeline");
            warn!("");

            // Use simulated data range for demo purposes
            info!("Using simulated time range for demonstration...");
            let end = Utc::now();
            let start = end - Duration::days(30);
            ma_runner = ma_runner.with_time_range(start, end);
        }
    }

    info!("");
    info!("Running MA Crossover walk-forward optimization...");
    info!("Parameter space: fast_period=[5,10,15,20], slow_period=[20,30,40,50]");
    info!("Total combinations: 16");
    info!("");

    match ma_runner.run(&ma_strategy).await {
        Ok(result) => {
            print_walk_forward_results("MA Crossover", &result);
        }
        Err(e) => {
            error!("MA Crossover walk-forward failed: {}", e);
            info!("This may be due to missing data in QuestDB.");
        }
    }

    // ========================================================================
    // Strategy 2: RSI Mean Reversion
    // ========================================================================

    info!("");
    info!("═══════════════════════════════════════════════════════════");
    info!("Strategy 2: RSI Mean Reversion");
    info!("═══════════════════════════════════════════════════════════");

    let rsi_strategy = RSIMeanReversionStrategy::new(14, dec!(30), dec!(70), &config.symbol);

    let mut rsi_runner = match QuestDBWalkForwardRunner::builder()
        .questdb_config(questdb_config)
        .windows(config.num_windows)
        .in_sample_pct(config.in_sample_pct)
        .rolling()
        .parameter(ParameterRange::int("rsi_period", 7, 21, 7)) // 7, 14, 21
        .parameter(ParameterRange::float("oversold", 20.0, 35.0, 5.0)) // 20, 25, 30, 35
        .parameter(ParameterRange::float("overbought", 65.0, 80.0, 5.0)) // 65, 70, 75, 80
        .metric(OptimizationMetric::SharpeRatio)
        .verbose(true)
        .build()
    {
        Ok(runner) => runner,
        Err(e) => {
            error!("Failed to build RSI runner: {}", e);
            return Err(e.into());
        }
    };

    // Use same time range discovery
    if let Err(e) = rsi_runner.discover_time_range().await {
        warn!("Using simulated time range: {}", e);
        let end = Utc::now();
        let start = end - Duration::days(30);
        rsi_runner = rsi_runner.with_time_range(start, end);
    }

    info!("");
    info!("Running RSI Mean Reversion walk-forward optimization...");
    info!("Parameter space: rsi_period=[7,14,21], oversold=[20-35], overbought=[65-80]");
    info!("Total combinations: 48");
    info!("");

    match rsi_runner.run(&rsi_strategy).await {
        Ok(result) => {
            print_walk_forward_results("RSI Mean Reversion", &result);
        }
        Err(e) => {
            error!("RSI Mean Reversion walk-forward failed: {}", e);
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================

    info!("");
    info!("═══════════════════════════════════════════════════════════");
    info!("Walk-Forward Analysis Complete");
    info!("═══════════════════════════════════════════════════════════");
    info!("");
    info!("Key Metrics to Evaluate:");
    info!("  • Efficiency > 70%: Strategy likely robust");
    info!("  • Efficiency 50-70%: Marginally robust, needs more testing");
    info!("  • Efficiency < 50%: Likely overfit, avoid live trading");
    info!("");
    info!("  • Parameter Stability > 0.7: Parameters are stable across windows");
    info!("  • Parameter Stability < 0.5: Parameters vary too much (overfit risk)");
    info!("");
    info!("Next Steps:");
    info!("  1. Record more data: cargo run --example data_pipeline");
    info!("  2. Test with different symbols and time periods");
    info!("  3. Add more sophisticated strategies");
    info!("  4. Forward test winning strategies in paper trading mode");

    Ok(())
}

fn print_walk_forward_results(
    strategy_name: &str,
    result: &janus_execution::sim::optimization::WalkForwardResult,
) {
    info!("");
    info!("┌─────────────────────────────────────────────────────────┐");
    info!("│ {} Results", strategy_name);
    info!("├─────────────────────────────────────────────────────────┤");
    info!(
        "│ Avg In-Sample Metric:     {:>10.4}                   │",
        result.avg_in_sample_metric
    );
    info!(
        "│ Avg Out-of-Sample Metric: {:>10.4}                   │",
        result.avg_out_of_sample_metric
    );
    info!(
        "│ Efficiency:               {:>10.1}%                  │",
        result.efficiency * 100.0
    );

    // Calculate average parameter stability
    let avg_stability: f64 = if result.parameter_stability.is_empty() {
        0.0
    } else {
        result.parameter_stability.values().sum::<f64>() / result.parameter_stability.len() as f64
    };
    info!(
        "│ Parameter Stability:      {:>10.4}                   │",
        avg_stability
    );
    info!(
        "│ Duration:                 {:>10}ms                 │",
        result.total_duration_ms
    );
    info!("└─────────────────────────────────────────────────────────┘");

    // Print window details
    info!("");
    info!("Window-by-Window Results:");
    info!("─────────────────────────────────────────────────────────────");

    for (i, window) in result.windows.iter().enumerate() {
        let is_metric = window
            .in_sample_result
            .as_ref()
            .map(|r| r.metric_value)
            .unwrap_or(0.0);
        let oos_metric = window
            .out_of_sample_result
            .as_ref()
            .map(|r| r.metric_value)
            .unwrap_or(0.0);

        let params_str = window
            .best_params
            .as_ref()
            .map(|p| {
                p.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "N/A".to_string());

        let efficiency = if is_metric != 0.0 {
            (oos_metric / is_metric) * 100.0
        } else {
            0.0
        };

        info!(
            "Window {}: IS={:>7.4}, OOS={:>7.4}, Eff={:>5.1}%, Params: {}",
            i + 1,
            is_metric,
            oos_metric,
            efficiency,
            params_str
        );
    }

    // Robustness check
    info!("");
    if result.is_robust(0.7) {
        info!("✅ Strategy PASSES walk-forward validation (efficiency >= 70%)");
    } else if result.is_robust(0.5) {
        warn!("⚠️  Strategy is MARGINALLY robust (50% <= efficiency < 70%)");
        warn!("    Consider more testing before live trading");
    } else {
        error!("❌ Strategy FAILS walk-forward validation (efficiency < 50%)");
        error!("    High risk of overfitting - do not trade live");
    }
}
