//! Walk-Forward Backtest Example
//!
//! This example demonstrates how to use the walk-forward optimization framework
//! to validate a trading strategy's robustness using recorded market data.
//!
//! Walk-forward analysis divides historical data into multiple windows:
//! - Each window has an in-sample (training) period and out-of-sample (validation) period
//! - Parameters are optimized on in-sample data
//! - Best parameters are validated on out-of-sample data
//! - This helps prevent overfitting and tests parameter stability
//!
//! # Usage
//!
//! ```bash
//! cargo run --example walk_forward_backtest
//! ```
//!
//! # With QuestDB Data
//!
//! ```bash
//! cargo run --example walk_forward_backtest -- --questdb http://localhost:9000
//! ```

use chrono::{DateTime, Datelike, Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;

use janus_execution::sim::{
    OptimizationError, OptimizationMetric, OptimizationRunResult, ParameterRange, ParameterSet,
    StrategyEvaluator, WalkForwardBacktestBuilder, WalkForwardBacktestRunner, WalkForwardConfig,
};

// ============================================================================
// Example Moving Average Crossover Strategy
// ============================================================================

/// A simple moving average crossover strategy for demonstration
///
/// Parameters:
/// - fast_period: Fast MA period (5-50)
/// - slow_period: Slow MA period (20-200)
/// - use_ema: Whether to use EMA instead of SMA
struct MovingAverageCrossoverStrategy {
    /// Simulated price data (in a real scenario, this would come from QuestDB)
    price_data: Arc<Vec<PricePoint>>,
}

#[derive(Clone)]
struct PricePoint {
    timestamp: DateTime<Utc>,
    close: f64,
}

impl MovingAverageCrossoverStrategy {
    fn new(price_data: Vec<PricePoint>) -> Self {
        Self {
            price_data: Arc::new(price_data),
        }
    }

    /// Calculate simple moving average
    fn sma(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![];
        }

        let mut result = Vec::with_capacity(prices.len() - period + 1);
        let mut sum: f64 = prices[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..prices.len() {
            sum = sum - prices[i - period] + prices[i];
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate exponential moving average
    fn ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(prices.len());

        // Start with SMA for first value
        let initial_sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
        result.push(initial_sma);

        for price in prices.iter().skip(period) {
            let ema = (price - result.last().unwrap()) * multiplier + result.last().unwrap();
            result.push(ema);
        }

        result
    }

    /// Run backtest with given parameters
    fn backtest(
        &self,
        fast_period: i64,
        slow_period: i64,
        use_ema: bool,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> BacktestResult {
        // Filter data to time range
        let data: Vec<&PricePoint> = self
            .price_data
            .iter()
            .filter(|p| p.timestamp >= start_time && p.timestamp <= end_time)
            .collect();

        if data.len() < slow_period as usize + 10 {
            return BacktestResult::default();
        }

        let prices: Vec<f64> = data.iter().map(|p| p.close).collect();

        // Calculate moving averages
        let (fast_ma, slow_ma) = if use_ema {
            (
                self.ema(&prices, fast_period as usize),
                self.ema(&prices, slow_period as usize),
            )
        } else {
            (
                self.sma(&prices, fast_period as usize),
                self.sma(&prices, slow_period as usize),
            )
        };

        // Align MAs (slow MA starts later)
        let offset = slow_period as usize - fast_period as usize;
        if fast_ma.len() <= offset {
            return BacktestResult::default();
        }

        let fast_ma_aligned = &fast_ma[offset..];
        let min_len = fast_ma_aligned.len().min(slow_ma.len());

        // Generate signals and calculate PnL
        let mut position = 0; // -1, 0, or 1
        let mut trades = Vec::new();
        let mut entry_price = 0.0;

        for i in 1..min_len {
            let fast_prev = fast_ma_aligned[i - 1];
            let fast_curr = fast_ma_aligned[i];
            let slow_prev = slow_ma[i - 1];
            let slow_curr = slow_ma[i];

            // Crossover detection
            let crossed_above = fast_prev <= slow_prev && fast_curr > slow_curr;
            let crossed_below = fast_prev >= slow_prev && fast_curr < slow_curr;

            let price_idx = slow_period as usize + i;
            if price_idx >= prices.len() {
                break;
            }
            let current_price = prices[price_idx];

            if crossed_above && position <= 0 {
                // Close short, open long
                if position < 0 {
                    trades.push(entry_price - current_price);
                }
                position = 1;
                entry_price = current_price;
            } else if crossed_below && position >= 0 {
                // Close long, open short
                if position > 0 {
                    trades.push(current_price - entry_price);
                }
                position = -1;
                entry_price = current_price;
            }
        }

        // Close any open position
        if position != 0 && !prices.is_empty() {
            let final_price = *prices.last().unwrap();
            if position > 0 {
                trades.push(final_price - entry_price);
            } else {
                trades.push(entry_price - final_price);
            }
        }

        BacktestResult::from_trades(&trades)
    }
}

/// Backtest result with computed metrics
#[derive(Default)]
struct BacktestResult {
    total_trades: u64,
    win_rate: f64,
    total_return: f64,
    max_drawdown: f64,
    sharpe_ratio: f64,
    profit_factor: f64,
}

impl BacktestResult {
    fn from_trades(trades: &[f64]) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        let wins: Vec<&f64> = trades.iter().filter(|&&t| t > 0.0).collect();
        let losses: Vec<&f64> = trades.iter().filter(|&&t| t < 0.0).collect();

        let total_return: f64 = trades.iter().sum();
        let win_rate = wins.len() as f64 / trades.len() as f64 * 100.0;

        let gross_profit: f64 = wins.iter().copied().sum();
        let gross_loss: f64 = losses.iter().copied().map(|l| l.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Calculate max drawdown from equity curve
        let mut equity: f64 = 0.0;
        let mut peak: f64 = 0.0;
        let mut max_dd: f64 = 0.0;
        for pnl in trades {
            equity += pnl;
            peak = peak.max(equity);
            let dd = peak - equity;
            max_dd = max_dd.max(dd);
        }

        // Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
        let mean_return = total_return / trades.len() as f64;
        let variance: f64 = trades
            .iter()
            .map(|t| (t - mean_return).powi(2))
            .sum::<f64>()
            / trades.len() as f64;
        let std_dev = variance.sqrt();
        let sharpe = if std_dev > 0.0 {
            mean_return / std_dev * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        };

        Self {
            total_trades: trades.len() as u64,
            win_rate,
            total_return,
            max_drawdown: max_dd,
            sharpe_ratio: sharpe,
            profit_factor,
        }
    }
}

impl StrategyEvaluator for MovingAverageCrossoverStrategy {
    fn evaluate(
        &self,
        params: &ParameterSet,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<OptimizationRunResult, OptimizationError> {
        // Extract parameters
        let fast_period = params
            .get("fast_period")
            .and_then(|v| v.as_int())
            .unwrap_or(10);

        let slow_period = params
            .get("slow_period")
            .and_then(|v| v.as_int())
            .unwrap_or(50);

        let use_ema = params
            .get("use_ema")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Validate parameters
        if fast_period >= slow_period {
            return Err(OptimizationError::InvalidRange(
                "fast_period must be less than slow_period".to_string(),
            ));
        }

        // Run backtest
        let result = self.backtest(fast_period, slow_period, use_ema, start_time, end_time);

        // Build optimization result
        let mut metrics = HashMap::new();
        metrics.insert("total_return".to_string(), result.total_return);
        metrics.insert("win_rate".to_string(), result.win_rate);
        metrics.insert("max_drawdown".to_string(), result.max_drawdown);
        metrics.insert("profit_factor".to_string(), result.profit_factor);

        Ok(OptimizationRunResult {
            parameters: params.clone(),
            metrics,
            metric_value: result.sharpe_ratio, // Optimizing for Sharpe ratio
            total_trades: result.total_trades,
            win_rate: result.win_rate,
            max_drawdown: result.max_drawdown,
            sharpe_ratio: Some(result.sharpe_ratio),
            profit_factor: Some(result.profit_factor),
            run_duration_ms: 0,
        })
    }
}

// ============================================================================
// Simulated Data Generation
// ============================================================================

/// Generate simulated price data for testing
fn generate_simulated_data(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    initial_price: f64,
) -> Vec<PricePoint> {
    let mut data = Vec::new();
    let mut price = initial_price;
    let mut current = start;

    // Simple random walk with trend and mean reversion
    let trend = 0.0001; // Slight upward trend
    let volatility = 0.02;
    let mean_reversion = 0.01;

    // Simple pseudo-random number generator state
    let mut rng_state: u64 = current.timestamp() as u64;

    while current <= end {
        // Skip weekends for more realistic data
        let weekday = current.weekday().num_days_from_monday();
        if weekday < 5 {
            data.push(PricePoint {
                timestamp: current,
                close: price,
            });

            // Update price with random walk (simple LCG random)
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let random = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0;
            let reversion = (initial_price - price) * mean_reversion;
            price *= 1.0 + trend + random * volatility + reversion;
            price = price.max(initial_price * 0.5).min(initial_price * 2.0);
        }

        current += Duration::hours(1);
    }

    data
}

// ============================================================================
// Main Example
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Walk-Forward Backtest Optimization Example          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Define data time range (1 year of data)
    let data_end = Utc::now();
    let data_start = data_end - Duration::days(365);

    println!(
        "📊 Data Range: {} to {}",
        data_start.format("%Y-%m-%d"),
        data_end.format("%Y-%m-%d")
    );
    println!();

    // Generate simulated price data
    println!("🔄 Generating simulated price data...");
    let price_data = generate_simulated_data(data_start, data_end, 100.0);
    println!("   Generated {} price points", price_data.len());
    println!();

    // Create strategy
    let strategy = MovingAverageCrossoverStrategy::new(price_data);

    // ========================================================================
    // Method 1: Using WalkForwardBacktestBuilder (Recommended)
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Method 1: Using WalkForwardBacktestBuilder");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let runner = WalkForwardBacktestBuilder::new()
        .windows(5) // 5 walk-forward windows
        .in_sample_pct(0.7) // 70% in-sample, 30% out-of-sample
        .min_trades(10) // Minimum 10 trades per window
        .rolling() // Use rolling windows (not anchored)
        .parameter(ParameterRange::int("fast_period", 5, 25, 5)) // Fast MA: 5, 10, 15, 20, 25
        .parameter(ParameterRange::int("slow_period", 30, 100, 10)) // Slow MA: 30, 40, ..., 100
        .parameter(ParameterRange::boolean("use_ema")) // Test both SMA and EMA
        .metric(OptimizationMetric::SharpeRatio)
        .verbose(true)
        .build(data_start, data_end)?;

    println!("📈 Running walk-forward optimization...");
    println!("   Windows: {}", runner.analysis().windows().len());
    println!(
        "   Parameter combinations: {}",
        runner.analysis().config().optimization.total_combinations()
    );
    println!();

    let result = runner.run(&strategy).await?;

    // Print results
    print_walk_forward_results(&result);

    // ========================================================================
    // Method 2: Manual Configuration
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Method 2: Manual Configuration with Anchored Windows");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let opt_config = janus_execution::sim::OptimizationConfig::new()
        .with_parameter(ParameterRange::int("fast_period", 10, 30, 10))
        .with_parameter(ParameterRange::int("slow_period", 50, 100, 25))
        .with_metric(OptimizationMetric::SharpeRatio)
        .with_direction(janus_execution::sim::OptimizationDirection::Maximize);

    let wf_config = WalkForwardConfig::new(4)
        .with_in_sample_pct(0.8)
        .with_min_trades(5)
        .anchored() // Expanding windows from start
        .with_optimization(opt_config);

    let manual_runner = WalkForwardBacktestRunner::new(wf_config, data_start, data_end)?;

    println!("📈 Running anchored walk-forward optimization...");
    let manual_result = manual_runner.run_sync(&strategy)?;

    print_walk_forward_results(&manual_result);

    // ========================================================================
    // Robustness Check
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Robustness Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let efficiency_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9];
    println!("Efficiency Threshold Analysis (Rolling Method):");
    for threshold in efficiency_thresholds {
        let passes = result.is_robust(threshold);
        let status = if passes { "✅ PASS" } else { "❌ FAIL" };
        println!(
            "  {:.0}% threshold: {} (actual: {:.1}%)",
            threshold * 100.0,
            status,
            result.efficiency * 100.0
        );
    }

    println!();
    println!("Efficiency Threshold Analysis (Anchored Method):");
    for threshold in efficiency_thresholds {
        let passes = manual_result.is_robust(threshold);
        let status = if passes { "✅ PASS" } else { "❌ FAIL" };
        println!(
            "  {:.0}% threshold: {} (actual: {:.1}%)",
            threshold * 100.0,
            status,
            manual_result.efficiency * 100.0
        );
    }

    // ========================================================================
    // Parameter Stability Analysis
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Parameter Stability (Rolling Method)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    if result.parameter_stability.is_empty() {
        println!("  No parameter stability data available.");
    } else {
        for (param, stability) in &result.parameter_stability {
            let bar_len = (stability * 20.0) as usize;
            let bar = "█".repeat(bar_len);
            let empty = "░".repeat(20 - bar_len);
            println!("  {:<15} [{bar}{empty}] {:.1}%", param, stability * 100.0);
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Done! Walk-forward analysis complete.");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn print_walk_forward_results(result: &janus_execution::sim::WalkForwardResult) {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│                    Walk-Forward Results                      │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!(
        "│ Windows Processed:      {:>6}                              │",
        result.windows.len()
    );
    println!(
        "│ Avg In-Sample Metric:   {:>10.4}                        │",
        result.avg_in_sample_metric
    );
    println!(
        "│ Avg Out-of-Sample Metric: {:>8.4}                        │",
        result.avg_out_of_sample_metric
    );
    println!(
        "│ Walk-Forward Efficiency: {:>8.1}%                        │",
        result.efficiency * 100.0
    );
    println!(
        "│ Total Duration:         {:>6} ms                         │",
        result.total_duration_ms
    );
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // Print window details
    println!("Window-by-Window Results:");
    println!(
        "┌────────┬────────────────────────┬────────────────────────┬─────────────┬─────────────┐"
    );
    println!(
        "│ Window │      In-Sample         │     Out-of-Sample      │   IS Metric │  OOS Metric │"
    );
    println!(
        "├────────┼────────────────────────┼────────────────────────┼─────────────┼─────────────┤"
    );

    for window in &result.windows {
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

        println!(
            "│ {:>6} │ {:>22} │ {:>22} │ {:>11.4} │ {:>11.4} │",
            window.index,
            window.is_start.format("%Y-%m-%d"),
            window.oos_start.format("%Y-%m-%d"),
            is_metric,
            oos_metric
        );
    }

    println!(
        "└────────┴────────────────────────┴────────────────────────┴─────────────┴─────────────┘"
    );

    // Print best parameters from each window
    println!();
    println!("Best Parameters per Window:");
    for window in &result.windows {
        if let Some(ref params) = window.best_params {
            let params_str: Vec<String> =
                params.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
            println!("  Window {}: {}", window.index, params_str.join(", "));
        }
    }
}
