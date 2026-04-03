//! Portfolio Optimization Integration Example
//!
//! Demonstrates how to integrate portfolio optimization with:
//! - Backtesting simulation
//! - Risk management
//! - Execution analytics
//! - Live pipeline (conceptual)

use vision::portfolio::{
    MeanVarianceOptimizer, OptimizationObjective, PortfolioAnalytics, RiskParityMethod,
    RiskParityOptimizer,
};

fn main() {
    println!("{}", "=".repeat(80));
    println!("Portfolio Optimization Integration Example");
    println!("{}", "=".repeat(80));
    println!();

    // Part 1: Backtest with Portfolio Optimization
    println!("\n{}", "─".repeat(80));
    println!("PART 1: BACKTESTING WITH PORTFOLIO REBALANCING");
    println!("{}", "─".repeat(80));
    demo_backtest_integration();

    // Part 2: Risk Management Integration
    println!("\n{}", "─".repeat(80));
    println!("PART 2: RISK-MANAGED PORTFOLIO OPTIMIZATION");
    println!("{}", "─".repeat(80));
    demo_risk_management_integration();

    // Part 3: Multi-Strategy Portfolio
    println!("\n{}", "─".repeat(80));
    println!("PART 3: MULTI-STRATEGY PORTFOLIO ALLOCATION");
    println!("{}", "─".repeat(80));
    demo_multi_strategy_portfolio();

    // Part 4: Regime-Based Portfolio
    println!("\n{}", "─".repeat(80));
    println!("PART 4: REGIME-DEPENDENT OPTIMIZATION");
    println!("{}", "─".repeat(80));
    demo_regime_based_portfolio();

    println!("\n{}", "=".repeat(80));
    println!("Integration Demo Complete!");
    println!("{}", "=".repeat(80));
}

/// Demonstrate backtesting with periodic portfolio rebalancing
fn demo_backtest_integration() {
    println!("\nSimulating multi-asset portfolio with monthly rebalancing...\n");

    // Portfolio universe
    let symbols = vec!["SPY", "TLT", "GLD", "VNQ"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    // Simulated historical data (252 trading days)
    let historical_returns = generate_historical_returns(&symbols, 252);
    let prices = generate_price_series(&historical_returns, 100.0);

    // Backtest parameters
    let initial_capital = 100_000.0;
    let rebalance_frequency = 21; // Monthly (21 trading days)
    let risk_free_rate = 0.02;

    println!("Initial Capital: ${:.0}", initial_capital);
    println!("Rebalance Frequency: {} days", rebalance_frequency);
    println!("Universe: {:?}", symbols);
    println!();

    // Storage for backtest results
    let mut portfolio_values = vec![initial_capital];
    let mut current_weights = vec![0.25; symbols.len()]; // Start equal-weighted
    let mut total_turnover = 0.0;
    let mut rebalance_count = 0;

    // Run backtest
    for day in 0..prices[0].len() {
        // Calculate portfolio value
        let returns_today: Vec<f64> = prices
            .iter()
            .map(|price_series| {
                if day > 0 {
                    (price_series[day] - price_series[day - 1]) / price_series[day - 1]
                } else {
                    0.0
                }
            })
            .collect();

        let portfolio_return = current_weights
            .iter()
            .zip(returns_today.iter())
            .map(|(w, r)| w * r)
            .sum::<f64>();

        let portfolio_value = portfolio_values.last().unwrap() * (1.0 + portfolio_return);
        portfolio_values.push(portfolio_value);

        // Rebalance periodically
        if day > 0 && day % rebalance_frequency == 0 {
            // Estimate expected returns (simple: use trailing 60-day returns)
            let lookback = 60.min(day);
            let expected_returns: Vec<f64> = (0..symbols.len())
                .map(|i| {
                    let returns = &prices[i][day - lookback..day];
                    let mean_return = returns
                        .windows(2)
                        .map(|w| (w[1] - w[0]) / w[0])
                        .sum::<f64>()
                        / (returns.len() - 1) as f64;
                    mean_return * 252.0 // Annualize
                })
                .collect();

            // Estimate covariance (simple: use trailing 60-day)
            let covariance = estimate_covariance(&prices, day, lookback);

            // Optimize portfolio
            let optimizer =
                MeanVarianceOptimizer::new(expected_returns, covariance, symbols.clone())
                    .unwrap()
                    .with_risk_free_rate(risk_free_rate);

            let result = optimizer
                .optimize(OptimizationObjective::MaxSharpe)
                .unwrap();

            // Calculate turnover
            let turnover = PortfolioAnalytics::turnover(&current_weights, &result.weights);
            total_turnover += turnover;

            // Update weights
            current_weights = result.weights.clone();
            rebalance_count += 1;

            if rebalance_count <= 3 {
                println!("Rebalance #{} (Day {}):", rebalance_count, day);
                println!("  Expected Return: {:.2}%", result.expected_return * 100.0);
                println!("  Expected Volatility: {:.2}%", result.volatility * 100.0);
                println!("  Sharpe Ratio: {:.4}", result.sharpe_ratio.unwrap_or(0.0));
                println!("  Turnover: {:.2}%", turnover * 100.0);
                println!("  Portfolio Value: ${:.0}", portfolio_value);
                println!();
            }
        }
    }

    // Calculate backtest statistics
    let final_value = portfolio_values.last().unwrap();
    let total_return = (final_value - initial_capital) / initial_capital;
    let annualized_return = (1.0 + total_return).powf(252.0 / portfolio_values.len() as f64) - 1.0;
    let max_dd = PortfolioAnalytics::max_drawdown(&portfolio_values);
    let avg_turnover = total_turnover / rebalance_count as f64;

    println!("Backtest Results:");
    println!("─────────────────────────────────────────────");
    println!("Final Portfolio Value: ${:.0}", final_value);
    println!("Total Return: {:.2}%", total_return * 100.0);
    println!("Annualized Return: {:.2}%", annualized_return * 100.0);
    println!("Max Drawdown: {:.2}%", max_dd * 100.0);
    println!("Number of Rebalances: {}", rebalance_count);
    println!(
        "Average Turnover per Rebalance: {:.2}%",
        avg_turnover * 100.0
    );
    println!(
        "Total Transaction Cost (5bps): ${:.0}",
        total_turnover * final_value * 0.0005
    );
}

/// Demonstrate risk-managed portfolio optimization
fn demo_risk_management_integration() {
    println!("\nOptimizing portfolio with risk constraints...\n");

    let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let returns = vec![0.12, 0.15, 0.10, 0.18, 0.25];
    let covariance = vec![
        vec![0.0400, 0.0100, 0.0150, 0.0120, 0.0180],
        vec![0.0100, 0.0900, 0.0200, 0.0150, 0.0250],
        vec![0.0150, 0.0200, 0.0625, 0.0180, 0.0220],
        vec![0.0120, 0.0150, 0.0180, 0.1600, 0.0300],
        vec![0.0180, 0.0250, 0.0220, 0.0300, 0.3600],
    ];

    // Risk limits (simple thresholds for demonstration)
    let max_volatility = 0.20;
    let max_position_size = 0.30;

    // Initial optimization (unconstrained)
    println!("1. Unconstrained Optimization:");
    let optimizer =
        MeanVarianceOptimizer::new(returns.clone(), covariance.clone(), symbols.clone())
            .unwrap()
            .with_risk_free_rate(0.02);

    let unconstrained = optimizer
        .optimize(OptimizationObjective::MaxSharpe)
        .unwrap();

    print_portfolio_summary(&unconstrained, "Unconstrained");

    // Check risk limits
    let risk_check = unconstrained.volatility <= max_volatility
        && unconstrained
            .weights
            .iter()
            .all(|&w| w <= max_position_size);
    println!("Risk Check: {}", if risk_check { "PASS" } else { "FAIL" });

    if !risk_check {
        println!("\n2. Risk-Constrained Optimization:");

        // Apply position size constraints
        use vision::portfolio::PortfolioConstraints;

        let constraints = PortfolioConstraints::default().with_weight_bounds(
            vec![0.0; 5],
            vec![0.30; 5], // Max 30% per position
        );

        let constrained_optimizer =
            MeanVarianceOptimizer::new(returns.clone(), covariance.clone(), symbols.clone())
                .unwrap()
                .with_risk_free_rate(0.02)
                .with_constraints(constraints);

        let constrained = constrained_optimizer
            .optimize(OptimizationObjective::MaxSharpe)
            .unwrap();

        print_portfolio_summary(&constrained, "Risk-Constrained");

        let risk_check2 = constrained.volatility <= max_volatility
            && constrained.weights.iter().all(|&w| w <= max_position_size);
        println!("Risk Check: {}", if risk_check2 { "PASS" } else { "FAIL" });

        if !risk_check2 && constrained.volatility > max_volatility {
            println!("\n3. Volatility-Constrained Optimization (Risk Parity):");

            // Use risk parity instead
            let rp_optimizer = RiskParityOptimizer::new(covariance, symbols.clone())
                .unwrap()
                .with_method(RiskParityMethod::EqualRiskContribution);

            let rp_result = rp_optimizer.optimize().unwrap();

            println!("Portfolio Volatility: {:.2}%", rp_result.volatility * 100.0);
            println!(
                "Max Weight: {:.2}%",
                rp_result
                    .weights
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    * 100.0
            );

            let vol_check = rp_result.volatility <= 0.20;
            let conc_check = rp_result.weights.iter().all(|&w| w <= 0.30);

            println!(
                "Volatility Check: {}",
                if vol_check { "PASS" } else { "FAIL" }
            );
            println!(
                "Concentration Check: {}",
                if conc_check { "PASS" } else { "FAIL" }
            );
        }
    }
}

/// Demonstrate multi-strategy portfolio allocation
fn demo_multi_strategy_portfolio() {
    println!("\nAllocating capital across multiple trading strategies...\n");

    // Define strategies with their characteristics
    let strategies = vec![
        ("Momentum", 0.15, 0.20),   // 15% return, 20% vol
        ("MeanRevert", 0.12, 0.15), // 12% return, 15% vol
        ("Breakout", 0.18, 0.25),   // 18% return, 25% vol
        ("Pairs", 0.10, 0.12),      // 10% return, 12% vol
    ];

    let symbols: Vec<String> = strategies
        .iter()
        .map(|(name, _, _)| name.to_string())
        .collect();
    let returns: Vec<f64> = strategies.iter().map(|(_, ret, _)| *ret).collect();

    // Correlation between strategies (lower is better for diversification)
    let covariance = vec![
        vec![0.0400, 0.0050, 0.0100, 0.0020],
        vec![0.0050, 0.0225, 0.0080, 0.0030],
        vec![0.0100, 0.0080, 0.0625, 0.0050],
        vec![0.0020, 0.0030, 0.0050, 0.0144],
    ];

    println!("Available Strategies:");
    for (_i, (name, ret, vol)) in strategies.iter().enumerate() {
        println!(
            "  {}: Return {:.1}%, Vol {:.1}%",
            name,
            ret * 100.0,
            vol * 100.0
        );
    }
    println!();

    // Compare allocation methods
    println!("1. Maximum Sharpe Allocation:");
    let optimizer =
        MeanVarianceOptimizer::new(returns.clone(), covariance.clone(), symbols.clone())
            .unwrap()
            .with_risk_free_rate(0.02);

    let max_sharpe = optimizer
        .optimize(OptimizationObjective::MaxSharpe)
        .unwrap();

    for (strategy, weight) in max_sharpe.symbols.iter().zip(max_sharpe.weights.iter()) {
        if *weight > 0.01 {
            println!("  {}: {:.1}%", strategy, weight * 100.0);
        }
    }
    println!(
        "  Portfolio: {:.2}% return, {:.2}% vol, {:.4} Sharpe",
        max_sharpe.expected_return * 100.0,
        max_sharpe.volatility * 100.0,
        max_sharpe.sharpe_ratio.unwrap()
    );

    println!("\n2. Risk Parity Allocation:");
    let rp_optimizer = RiskParityOptimizer::new(covariance.clone(), symbols.clone())
        .unwrap()
        .with_method(RiskParityMethod::EqualRiskContribution);

    let rp_result = rp_optimizer.optimize().unwrap();

    for (strategy, weight) in rp_result.symbols.iter().zip(rp_result.weights.iter()) {
        println!("  {}: {:.1}%", strategy, weight * 100.0);
    }

    let rp_return = PortfolioAnalytics::portfolio_return(&rp_result.weights, &returns);
    let rp_sharpe = (rp_return - 0.02) / rp_result.volatility;

    println!(
        "  Portfolio: {:.2}% return, {:.2}% vol, {:.4} Sharpe",
        rp_return * 100.0,
        rp_result.volatility * 100.0,
        rp_sharpe
    );

    println!("\n3. Equal Weight Allocation:");
    let equal_weights = vec![0.25; 4];
    let eq_return = PortfolioAnalytics::portfolio_return(&equal_weights, &returns);
    let eq_vol = PortfolioAnalytics::portfolio_volatility(&equal_weights, &covariance);
    let eq_sharpe = (eq_return - 0.02) / eq_vol;

    for (strategy, weight) in symbols.iter().zip(equal_weights.iter()) {
        println!("  {}: {:.1}%", strategy, weight * 100.0);
    }
    println!(
        "  Portfolio: {:.2}% return, {:.2}% vol, {:.4} Sharpe",
        eq_return * 100.0,
        eq_vol * 100.0,
        eq_sharpe
    );
}

/// Demonstrate regime-based portfolio optimization
fn demo_regime_based_portfolio() {
    println!("\nAdjusting portfolio based on market regime...\n");

    let symbols = vec!["SPY", "TLT", "GLD"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    // Define market regimes
    let regimes = vec![
        ("Bull Market", vec![0.15, 0.03, 0.05]),
        ("Bear Market", vec![-0.10, 0.08, 0.12]),
        ("High Volatility", vec![0.05, 0.06, 0.10]),
    ];

    // Covariance changes by regime
    let covariances = vec![
        // Bull: low correlation
        vec![
            vec![0.0400, 0.0050, 0.0030],
            vec![0.0050, 0.0100, 0.0020],
            vec![0.0030, 0.0020, 0.0225],
        ],
        // Bear: high correlation
        vec![
            vec![0.0625, 0.0200, 0.0150],
            vec![0.0200, 0.0144, 0.0080],
            vec![0.0150, 0.0080, 0.0400],
        ],
        // High Vol: very high correlation
        vec![
            vec![0.1600, 0.0400, 0.0300],
            vec![0.0400, 0.0225, 0.0150],
            vec![0.0300, 0.0150, 0.0900],
        ],
    ];

    for (i, (regime_name, expected_returns)) in regimes.iter().enumerate() {
        println!("{}:", regime_name);
        println!("─────────────────────────────────────");

        let optimizer = MeanVarianceOptimizer::new(
            expected_returns.clone(),
            covariances[i].clone(),
            symbols.clone(),
        )
        .unwrap()
        .with_risk_free_rate(0.02);

        let result = optimizer
            .optimize(OptimizationObjective::MaxSharpe)
            .unwrap();

        for (symbol, weight) in result.symbols.iter().zip(result.weights.iter()) {
            if *weight > 0.01 {
                println!("  {}: {:.1}%", symbol, weight * 100.0);
            }
        }

        println!(
            "  Return: {:.2}%, Vol: {:.2}%, Sharpe: {:.4}",
            result.expected_return * 100.0,
            result.volatility * 100.0,
            result.sharpe_ratio.unwrap()
        );
        println!();
    }

    println!("Key Insights:");
    println!("- Bull Market: Higher equity allocation");
    println!("- Bear Market: Flight to safety (bonds, gold)");
    println!("- High Volatility: More balanced, risk parity approach");
}

// Helper functions

fn generate_historical_returns(symbols: &[String], days: usize) -> Vec<Vec<f64>> {
    // Simple random walk for demonstration
    symbols
        .iter()
        .enumerate()
        .map(|(i, _)| {
            (0..days)
                .map(|d| {
                    let base_return = 0.0001 * (i as f64 + 1.0);
                    let noise = 0.01 * ((d as f64 * 0.1 + i as f64).sin());
                    base_return + noise
                })
                .collect()
        })
        .collect()
}

fn generate_price_series(returns: &[Vec<f64>], initial_price: f64) -> Vec<Vec<f64>> {
    returns
        .iter()
        .map(|asset_returns| {
            let mut prices = vec![initial_price];
            for &ret in asset_returns {
                let last_price = prices.last().unwrap();
                prices.push(last_price * (1.0 + ret));
            }
            prices
        })
        .collect()
}

fn estimate_covariance(prices: &[Vec<f64>], current_day: usize, lookback: usize) -> Vec<Vec<f64>> {
    let n = prices.len();
    let start = current_day.saturating_sub(lookback);

    let mut returns_matrix = Vec::new();
    for price_series in prices {
        let returns: Vec<f64> = price_series[start..current_day]
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        returns_matrix.push(returns);
    }

    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mean_i = returns_matrix[i].iter().sum::<f64>() / returns_matrix[i].len() as f64;
            let mean_j = returns_matrix[j].iter().sum::<f64>() / returns_matrix[j].len() as f64;

            let covariance: f64 = returns_matrix[i]
                .iter()
                .zip(returns_matrix[j].iter())
                .map(|(ri, rj)| (ri - mean_i) * (rj - mean_j))
                .sum::<f64>()
                / (returns_matrix[i].len() - 1) as f64;

            cov[i][j] = covariance * 252.0; // Annualize
        }
    }

    cov
}

fn print_portfolio_summary(result: &vision::portfolio::OptimizationResult, label: &str) {
    println!("{}:", label);
    println!("  Return: {:.2}%", result.expected_return * 100.0);
    println!("  Volatility: {:.2}%", result.volatility * 100.0);
    println!("  Sharpe: {:.4}", result.sharpe_ratio.unwrap_or(0.0));

    let max_weight = result
        .weights
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("  Max Position: {:.2}%", max_weight * 100.0);
    println!();
}
