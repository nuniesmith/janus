//! Backtesting example for trading signals.
//!
//! This example demonstrates how to:
//! 1. Generate trading signals from a trained model
//! 2. Run a backtest simulation on historical data
//! 3. Calculate performance metrics (Sharpe ratio, win rate, drawdown)
//! 4. Analyze signal quality and accuracy
//!
//! # Usage
//!
//! ```bash
//! cargo run --example backtest_signals --release
//! ```

use chrono::{Duration, Utc};
use std::collections::HashMap;
use vision::backtest::{BacktestSimulation, PerformanceMetrics, PositionSizing, SimulationConfig};
use vision::signals::{SignalType, TradingSignal};

fn main() {
    println!("=== Trading Signal Backtesting Demo ===\n");

    // Run different backtesting scenarios
    println!("Scenario 1: Basic Backtesting");
    println!("─────────────────────────────────");
    run_basic_backtest();

    println!("\n\nScenario 2: Advanced Strategy with Stop-Loss");
    println!("──────────────────────────────────────────────");
    run_advanced_backtest();

    println!("\n\nScenario 3: Multiple Position Sizing Strategies");
    println!("─────────────────────────────────────────────────");
    compare_position_sizing();

    println!("\n\nScenario 4: Signal Quality Analysis");
    println!("────────────────────────────────────");
    analyze_signal_quality();

    println!("\n✅ All backtesting scenarios complete!");
}

/// Basic backtesting with default configuration
fn run_basic_backtest() {
    // Create simulation with default config
    let config = SimulationConfig {
        initial_capital: 10000.0,
        commission_rate: 0.001,
        slippage_rate: 0.0005,
        min_confidence: 0.6,
        max_positions: 1,
        ..Default::default()
    };

    let mut sim = BacktestSimulation::new(config);

    // Generate some test signals
    let signals = generate_test_signals();
    let prices = generate_test_prices();

    println!("Initial Capital: ${:.2}", sim.state().capital);
    println!("Processing {} signals...\n", signals.len());

    // Process each signal
    for (i, signal) in signals.iter().enumerate() {
        if let Some(&price) = prices.get(&signal.asset) {
            sim.process_signal(signal, price * (1.0 + (i as f64 * 0.01))); // Simulate price movement
        }
    }

    // Close all positions at final prices
    sim.close_all_positions(&prices);

    // Calculate and display metrics
    let metrics = sim.calculate_metrics();
    display_metrics(&metrics);

    println!("\n{}", sim.summary());
}

/// Advanced backtesting with stop-loss and take-profit
fn run_advanced_backtest() {
    let config = SimulationConfig {
        initial_capital: 10000.0,
        commission_rate: 0.001,
        slippage_rate: 0.0005,
        position_size: PositionSizing::FixedPercent(0.2), // 20% per trade
        min_confidence: 0.7,
        max_positions: 2,
        use_stop_loss: true,
        stop_loss_pct: 0.02, // 2% stop-loss
        use_take_profit: true,
        take_profit_pct: 0.05, // 5% take-profit
        allow_shorts: true,
        max_hold_hours: 48.0, // Max 48 hours
        ..Default::default()
    };

    let mut sim = BacktestSimulation::new(config);

    // Generate signals with varying confidence
    let signals = generate_varied_signals();
    let mut prices = generate_test_prices();

    println!("Strategy Settings:");
    println!("  • Stop-Loss: 2%");
    println!("  • Take-Profit: 5%");
    println!("  • Position Size: 20% of capital");
    println!("  • Max Hold: 48 hours");
    println!("  • Min Confidence: 70%\n");

    // Simulate price action
    for (i, signal) in signals.iter().enumerate() {
        let base_price = prices.get(&signal.asset).copied().unwrap_or(50000.0);

        // Simulate price volatility
        let price_multiplier = 1.0 + ((i as f64 * 0.005) - 0.01);
        let current_price = base_price * price_multiplier;

        prices.insert(signal.asset.clone(), current_price);

        sim.process_signal(signal, current_price);
        sim.update_positions(&prices);
    }

    // Close all positions
    sim.close_all_positions(&prices);

    let metrics = sim.calculate_metrics();
    display_metrics(&metrics);

    println!("\nRisk Metrics:");
    println!("  • Max Drawdown: {:.2}%", metrics.max_drawdown_pct);
    println!("  • Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
    println!("  • Sortino Ratio: {:.2}", metrics.sortino_ratio);
    println!("  • Calmar Ratio: {:.2}", metrics.calmar_ratio);
}

/// Compare different position sizing strategies
fn compare_position_sizing() {
    let strategies = vec![
        ("Fixed 10%", PositionSizing::FixedPercent(0.1)),
        ("Fixed 25%", PositionSizing::FixedPercent(0.25)),
        ("Fixed $1000", PositionSizing::FixedDollar(1000.0)),
        (
            "Kelly Criterion",
            PositionSizing::Kelly {
                win_rate: 0.6,
                avg_win_loss_ratio: 1.5,
            },
        ),
        (
            "Risk-Based 2%",
            PositionSizing::RiskBased {
                risk_percent: 0.02,
                stop_loss_percent: 0.02,
            },
        ),
    ];

    let signals = generate_test_signals();
    let prices = generate_test_prices();

    println!(
        "{:<20} {:>12} {:>10} {:>10} {:>10}",
        "Strategy", "Final ($)", "Return %", "Sharpe", "Win Rate"
    );
    println!("{}", "─".repeat(64));

    for (name, sizing) in strategies {
        let config = SimulationConfig {
            initial_capital: 10000.0,
            position_size: sizing,
            ..Default::default()
        };

        let mut sim = BacktestSimulation::new(config);

        // Run backtest
        for (i, signal) in signals.iter().enumerate() {
            if let Some(&price) = prices.get(&signal.asset) {
                sim.process_signal(signal, price * (1.0 + (i as f64 * 0.01)));
            }
        }
        sim.close_all_positions(&prices);

        let metrics = sim.calculate_metrics();
        let final_capital = 10000.0 + metrics.total_pnl;

        println!(
            "{:<20} {:>12.2} {:>9.2}% {:>10.2} {:>9.2}%",
            name,
            final_capital,
            metrics.total_return_pct,
            metrics.sharpe_ratio,
            metrics.win_rate * 100.0
        );
    }
}

/// Analyze signal quality and prediction accuracy
fn analyze_signal_quality() {
    let config = SimulationConfig {
        initial_capital: 10000.0,
        min_confidence: 0.5, // Accept all signals for analysis
        ..Default::default()
    };

    let mut sim = BacktestSimulation::new(config);

    // Generate signals with known outcomes
    let signals = generate_calibrated_signals();
    let prices = generate_test_prices();

    for (i, signal) in signals.iter().enumerate() {
        if let Some(&price) = prices.get(&signal.asset) {
            sim.process_signal(signal, price * (1.0 + (i as f64 * 0.01)));
        }
    }
    sim.close_all_positions(&prices);

    let quality = sim.calculate_signal_quality();

    println!("Signal Quality Metrics:");
    println!("─────────────────────────────────────");
    println!("Total Signals: {}", quality.total_signals);
    println!("Signals Traded: {}", quality.signals_traded);
    println!("Conversion Rate: {:.2}%", quality.signal_trade_rate * 100.0);
    println!("\nConfidence Analysis:");
    println!(
        "  • Avg Winning Confidence: {:.2}%",
        quality.avg_winning_confidence * 100.0
    );
    println!(
        "  • Avg Losing Confidence: {:.2}%",
        quality.avg_losing_confidence * 100.0
    );
    println!("  • Calibration Error: {:.4}", quality.calibration_error);
    println!(
        "\nHigh Confidence (>80%) Accuracy: {:.2}%",
        quality.high_confidence_accuracy * 100.0
    );
    println!(
        "Confidence-Return Correlation: {:.3}",
        quality.confidence_correlation
    );

    // Quality assessment
    if quality.avg_winning_confidence > quality.avg_losing_confidence + 0.1 {
        println!("\n✓ Good signal quality: Winners have higher confidence");
    } else {
        println!("\n⚠ Warning: Confidence not well calibrated");
    }
}

/// Display performance metrics in a nice format
fn display_metrics(metrics: &PerformanceMetrics) {
    println!("Performance Metrics:");
    println!("─────────────────────────────────────");
    println!("Total Trades: {}", metrics.total_trades);
    println!(
        "Winning Trades: {} ({:.2}%)",
        metrics.winning_trades,
        metrics.win_rate * 100.0
    );
    println!("Losing Trades: {}", metrics.losing_trades);

    println!("\nReturns:");
    println!("  • Total Return: {:.2}%", metrics.total_return_pct);
    println!(
        "  • Annualized Return: {:.2}%",
        metrics.annualized_return_pct
    );
    println!(
        "  • Average Return/Trade: {:.2}%",
        metrics.avg_return_per_trade
    );
    println!(
        "  • Median Return/Trade: {:.2}%",
        metrics.median_return_per_trade
    );

    println!("\nProfit/Loss:");
    println!("  • Total P&L: ${:.2}", metrics.total_pnl);
    println!("  • Gross Profit: ${:.2}", metrics.gross_profit);
    println!("  • Gross Loss: ${:.2}", metrics.gross_loss);
    println!("  • Profit Factor: {:.2}", metrics.profit_factor);

    println!("\nTrade Statistics:");
    println!("  • Average Win: ${:.2}", metrics.avg_win);
    println!("  • Average Loss: ${:.2}", metrics.avg_loss);
    println!("  • Largest Win: ${:.2}", metrics.largest_win);
    println!("  • Largest Loss: ${:.2}", metrics.largest_loss);
    println!("  • Win/Loss Ratio: {:.2}", metrics.avg_win_loss_ratio);

    println!("\nRisk Metrics:");
    println!("  • Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
    println!("  • Max Drawdown: {:.2}%", metrics.max_drawdown_pct);
    println!("  • Expectancy: ${:.2}", metrics.expectancy);

    println!("\nStreaks:");
    println!("  • Max Consecutive Wins: {}", metrics.max_consecutive_wins);
    println!(
        "  • Max Consecutive Losses: {}",
        metrics.max_consecutive_losses
    );

    println!("\nQuality Score: {:.1}/100", metrics.quality_score());

    if metrics.is_profitable() {
        println!("\n✓ Strategy is PROFITABLE");
    } else {
        println!("\n✗ Strategy is NOT PROFITABLE");
    }
}

/// Generate test trading signals
fn generate_test_signals() -> Vec<TradingSignal> {
    let mut signals = Vec::new();
    let assets = vec!["BTCUSD", "ETHUSDT", "SOLUSDT"];
    let signal_types = vec![SignalType::Buy, SignalType::Sell, SignalType::Close];

    for i in 0..15 {
        let asset = assets[i % assets.len()];
        let signal_type = signal_types[i % signal_types.len()];
        let confidence = 0.6 + (i as f64 % 4.0) * 0.1; // Vary between 0.6 and 0.9

        let mut signal = TradingSignal::new(signal_type, confidence, asset.to_string());
        signal.timestamp = Utc::now() - Duration::hours((15 - i) as i64);

        signals.push(signal);
    }

    signals
}

/// Generate signals with varying confidence levels
fn generate_varied_signals() -> Vec<TradingSignal> {
    let mut signals = Vec::new();
    let confidences = vec![0.95, 0.85, 0.75, 0.65, 0.90, 0.70, 0.80, 0.60, 0.85, 0.75];

    for (i, &conf) in confidences.iter().enumerate() {
        let signal_type = if i % 3 == 0 {
            SignalType::Buy
        } else if i % 3 == 1 {
            SignalType::Sell
        } else {
            SignalType::Close
        };

        let asset = if i % 2 == 0 { "BTCUSD" } else { "ETHUSDT" };

        let mut signal = TradingSignal::new(signal_type, conf, asset.to_string());
        signal.timestamp = Utc::now() - Duration::hours((10 - i) as i64);

        signals.push(signal);
    }

    signals
}

/// Generate signals with calibrated confidence (for quality analysis)
fn generate_calibrated_signals() -> Vec<TradingSignal> {
    let mut signals = Vec::new();

    // High confidence signals (should mostly win)
    for _i in 0..5 {
        let signal = TradingSignal::new(SignalType::Buy, 0.9, "BTCUSD".to_string());
        signals.push(signal);
    }

    // Medium confidence signals (mixed results)
    for _i in 0..5 {
        let signal = TradingSignal::new(SignalType::Buy, 0.7, "ETHUSDT".to_string());
        signals.push(signal);
    }

    // Low confidence signals (should mostly lose)
    for _i in 0..5 {
        let signal = TradingSignal::new(SignalType::Buy, 0.55, "SOLUSDT".to_string());
        signals.push(signal);
    }

    // Add some close signals
    for asset in ["BTCUSD", "ETHUSDT", "SOLUSDT"] {
        let signal = TradingSignal::new(SignalType::Close, 0.8, asset.to_string());
        signals.push(signal);
    }

    signals
}

/// Generate test price data
fn generate_test_prices() -> HashMap<String, f64> {
    let mut prices = HashMap::new();
    prices.insert("BTCUSD".to_string(), 50000.0);
    prices.insert("ETHUSDT".to_string(), 3000.0);
    prices.insert("SOLUSDT".to_string(), 100.0);
    prices
}
