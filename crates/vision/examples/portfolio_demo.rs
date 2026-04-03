//! Portfolio Optimization Demo
//!
//! Demonstrates all portfolio optimization methods:
//! - Mean-Variance Optimization (Markowitz)
//! - Risk Parity / Equal Risk Contribution
//! - Black-Litterman Model
//! - Portfolio Analytics and Rebalancing

use vision::portfolio::{
    BlackLittermanConfig, BlackLittermanOptimizer, BlackLittermanResult, CovarianceEstimator,
    MeanVarianceOptimizer, OptimizationObjective, OptimizationResult, PortfolioAnalytics,
    PortfolioConstraints, PortfolioRebalancer, RiskBudget, RiskParityMethod, RiskParityOptimizer,
    RiskParityResult, View,
};

fn main() {
    println!("{}", "=".repeat(80));
    println!("JANUS Vision - Portfolio Optimization Demo");
    println!("{}", "=".repeat(80));
    println!();

    // Define test portfolio
    let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    // Expected annual returns (%)
    let returns = vec![0.12, 0.15, 0.10, 0.18, 0.25];

    // Covariance matrix (annualized)
    let covariance = vec![
        vec![0.0400, 0.0100, 0.0150, 0.0120, 0.0180],
        vec![0.0100, 0.0900, 0.0200, 0.0150, 0.0250],
        vec![0.0150, 0.0200, 0.0625, 0.0180, 0.0220],
        vec![0.0120, 0.0150, 0.0180, 0.1600, 0.0300],
        vec![0.0180, 0.0250, 0.0220, 0.0300, 0.3600],
    ];

    // Market capitalization weights
    let market_weights = vec![0.30, 0.25, 0.20, 0.15, 0.10];

    let risk_free_rate = 0.02; // 2% risk-free rate

    // ========================================================================
    // Part 1: Mean-Variance Optimization
    // ========================================================================
    println!("\n{}", "─".repeat(80));
    println!("PART 1: MEAN-VARIANCE OPTIMIZATION (MARKOWITZ)");
    println!("{}", "─".repeat(80));

    demo_mean_variance(&symbols, &returns, &covariance, risk_free_rate);

    // ========================================================================
    // Part 2: Risk Parity
    // ========================================================================
    println!("\n{}", "─".repeat(80));
    println!("PART 2: RISK PARITY OPTIMIZATION");
    println!("{}", "─".repeat(80));

    demo_risk_parity(&symbols, &covariance);

    // ========================================================================
    // Part 3: Black-Litterman
    // ========================================================================
    println!("\n{}", "─".repeat(80));
    println!("PART 3: BLACK-LITTERMAN MODEL");
    println!("{}", "─".repeat(80));

    demo_black_litterman(&symbols, &covariance, &market_weights, risk_free_rate);

    // ========================================================================
    // Part 4: Portfolio Analytics
    // ========================================================================
    println!("\n{}", "─".repeat(80));
    println!("PART 4: PORTFOLIO ANALYTICS");
    println!("{}", "─".repeat(80));

    demo_analytics(&symbols, &returns, &covariance, risk_free_rate);

    // ========================================================================
    // Part 5: Rebalancing
    // ========================================================================
    println!("\n{}", "─".repeat(80));
    println!("PART 5: PORTFOLIO REBALANCING");
    println!("{}", "─".repeat(80));

    demo_rebalancing();

    // ========================================================================
    // Part 6: Efficient Frontier
    // ========================================================================
    println!("\n{}", "─".repeat(80));
    println!("PART 6: EFFICIENT FRONTIER");
    println!("{}", "─".repeat(80));

    demo_efficient_frontier(&symbols, &returns, &covariance, risk_free_rate);

    println!("\n{}", "=".repeat(80));
    println!("Demo Complete!");
    println!("{}", "=".repeat(80));
}

fn demo_mean_variance(
    symbols: &[String],
    returns: &[f64],
    covariance: &[Vec<f64>],
    risk_free_rate: f64,
) {
    let optimizer =
        MeanVarianceOptimizer::new(returns.to_vec(), covariance.to_vec(), symbols.to_vec())
            .unwrap()
            .with_risk_free_rate(risk_free_rate);

    println!("\n1.1 Maximum Sharpe Ratio Portfolio");
    println!("{}", "-".repeat(40));

    let max_sharpe = optimizer
        .optimize(OptimizationObjective::MaxSharpe)
        .unwrap();

    print_optimization_result(&max_sharpe);

    println!("\n1.2 Minimum Variance Portfolio");
    println!("{}", "-".repeat(40));

    let min_var = optimizer
        .optimize(OptimizationObjective::MinVarianceUnconstrained)
        .unwrap();

    print_optimization_result(&min_var);

    println!("\n1.3 Target Return Portfolio (12%)");
    println!("{}", "-".repeat(40));

    let target_return = optimizer
        .optimize(OptimizationObjective::MinVariance {
            target_return: 0.12,
        })
        .unwrap();

    print_optimization_result(&target_return);

    println!("\n1.4 Long-Only Constrained Portfolio");
    println!("{}", "-".repeat(40));

    let constrained = optimizer
        .with_constraints(PortfolioConstraints::long_only())
        .optimize(OptimizationObjective::MaxSharpe)
        .unwrap();

    print_optimization_result(&constrained);

    // Compare portfolios
    println!("\n1.5 Portfolio Comparison");
    println!("{}", "-".repeat(40));
    println!(
        "Turnover (Max Sharpe → Min Variance): {:.2}%",
        max_sharpe.turnover(&min_var) * 100.0
    );
}

fn demo_risk_parity(symbols: &[String], covariance: &[Vec<f64>]) {
    println!("\n2.1 Equal Risk Contribution");
    println!("{}", "-".repeat(40));

    let optimizer = RiskParityOptimizer::new(covariance.to_vec(), symbols.to_vec())
        .unwrap()
        .with_method(RiskParityMethod::EqualRiskContribution);

    let result = optimizer.optimize().unwrap();
    print_risk_parity_result(&result);

    println!("\n2.2 Inverse Volatility Weighting");
    println!("{}", "-".repeat(40));

    let inv_vol = RiskParityOptimizer::new(covariance.to_vec(), symbols.to_vec())
        .unwrap()
        .with_method(RiskParityMethod::InverseVolatility)
        .optimize()
        .unwrap();

    print_risk_parity_result(&inv_vol);

    println!("\n2.3 Custom Risk Budgeting");
    println!("{}", "-".repeat(40));

    // Allocate more risk to tech stocks, less to others
    let budget = RiskBudget::custom(
        symbols.to_vec(),
        vec![0.25, 0.25, 0.20, 0.20, 0.10], // AAPL, GOOGL, MSFT, AMZN, TSLA
    )
    .unwrap();

    let risk_budget = RiskParityOptimizer::new(covariance.to_vec(), symbols.to_vec())
        .unwrap()
        .with_method(RiskParityMethod::RiskBudgeting)
        .with_risk_budget(budget)
        .optimize()
        .unwrap();

    print_risk_parity_result(&risk_budget);

    // Diversification ratio
    let div_ratio = RiskParityOptimizer::new(covariance.to_vec(), symbols.to_vec())
        .unwrap()
        .diversification_ratio(&result.weights);

    println!("\nDiversification Ratio: {:.4}", div_ratio);
    println!("(Higher is better, >1 means diversification benefit)");
}

fn demo_black_litterman(
    symbols: &[String],
    covariance: &[Vec<f64>],
    market_weights: &[f64],
    risk_free_rate: f64,
) {
    println!("\n3.1 Market Equilibrium (No Views)");
    println!("{}", "-".repeat(40));

    let optimizer = BlackLittermanOptimizer::new(symbols.to_vec(), covariance.to_vec())
        .unwrap()
        .with_market_cap_weights(market_weights.to_vec())
        .unwrap()
        .with_risk_free_rate(risk_free_rate);

    let no_views = optimizer.optimize().unwrap();
    print_black_litterman_result(&no_views);

    println!("\n3.2 Absolute View: AAPL will return 15%");
    println!("{}", "-".repeat(40));

    let with_view = BlackLittermanOptimizer::new(symbols.to_vec(), covariance.to_vec())
        .unwrap()
        .with_market_cap_weights(market_weights.to_vec())
        .unwrap()
        .with_risk_free_rate(risk_free_rate)
        .add_view(View::absolute("AAPL".to_string(), 0.15, 0.80));

    let result = with_view.optimize().unwrap();
    print_black_litterman_result(&result);

    println!("\n3.3 Relative View: GOOGL will outperform MSFT by 5%");
    println!("{}", "-".repeat(40));

    let relative_view = BlackLittermanOptimizer::new(symbols.to_vec(), covariance.to_vec())
        .unwrap()
        .with_market_cap_weights(market_weights.to_vec())
        .unwrap()
        .with_risk_free_rate(risk_free_rate)
        .add_view(View::relative(
            "GOOGL".to_string(),
            "MSFT".to_string(),
            0.05,
            0.70,
        ));

    let result = relative_view.optimize().unwrap();
    print_black_litterman_result(&result);

    println!("\n3.4 Multiple Views");
    println!("{}", "-".repeat(40));

    let multi_view = BlackLittermanOptimizer::new(symbols.to_vec(), covariance.to_vec())
        .unwrap()
        .with_market_cap_weights(market_weights.to_vec())
        .unwrap()
        .with_risk_free_rate(risk_free_rate)
        .add_view(View::absolute("AAPL".to_string(), 0.14, 0.75))
        .add_view(View::relative(
            "AMZN".to_string(),
            "MSFT".to_string(),
            0.08,
            0.65,
        ))
        .add_view(View::absolute("TSLA".to_string(), 0.22, 0.50));

    let result = multi_view.optimize().unwrap();
    print_black_litterman_result(&result);

    // Show return adjustments
    println!("\nReturn Adjustments from Views:");
    println!(
        "{:<8} {:>10} {:>10} {:>10}",
        "Symbol", "Prior", "Posterior", "Δ"
    );
    println!("{}", "-".repeat(42));
    for (symbol, prior, posterior, delta) in result.return_adjustments() {
        println!(
            "{:<8} {:>9.2}% {:>9.2}% {:>9.2}%",
            symbol,
            prior * 100.0,
            posterior * 100.0,
            delta * 100.0
        );
    }

    println!("\n3.5 Confidence Impact Analysis");
    println!("{}", "-".repeat(40));

    // High confidence
    let high_conf = BlackLittermanOptimizer::new(symbols.to_vec(), covariance.to_vec())
        .unwrap()
        .with_market_cap_weights(market_weights.to_vec())
        .unwrap()
        .add_view(View::absolute("TSLA".to_string(), 0.30, 0.95))
        .optimize()
        .unwrap();

    // Low confidence
    let low_conf = BlackLittermanOptimizer::new(symbols.to_vec(), covariance.to_vec())
        .unwrap()
        .with_market_cap_weights(market_weights.to_vec())
        .unwrap()
        .add_view(View::absolute("TSLA".to_string(), 0.30, 0.20))
        .optimize()
        .unwrap();

    println!(
        "TSLA Expected Return with High Confidence (95%): {:.2}%",
        high_conf.expected_return("TSLA").unwrap() * 100.0
    );
    println!(
        "TSLA Expected Return with Low Confidence (20%): {:.2}%",
        low_conf.expected_return("TSLA").unwrap() * 100.0
    );
    println!(
        "Prior (Equilibrium) Return: {:.2}%",
        no_views.expected_return("TSLA").unwrap() * 100.0
    );
}

fn demo_analytics(
    symbols: &[String],
    returns: &[f64],
    covariance: &[Vec<f64>],
    risk_free_rate: f64,
) {
    // Create a sample portfolio
    let weights = vec![0.25, 0.20, 0.20, 0.20, 0.15];

    println!("\nSample Portfolio Weights:");
    for (symbol, &weight) in symbols.iter().zip(weights.iter()) {
        println!("  {}: {:.1}%", symbol, weight * 100.0);
    }

    println!("\n4.1 Risk-Return Metrics");
    println!("{}", "-".repeat(40));

    let port_return = PortfolioAnalytics::portfolio_return(&weights, returns);
    let port_vol = PortfolioAnalytics::portfolio_volatility(&weights, covariance);
    let sharpe = PortfolioAnalytics::sharpe_ratio(&weights, returns, covariance, risk_free_rate);

    println!("Expected Return: {:.2}%", port_return * 100.0);
    println!("Volatility: {:.2}%", port_vol * 100.0);
    println!("Sharpe Ratio: {:.4}", sharpe);

    println!("\n4.2 Risk Decomposition");
    println!("{}", "-".repeat(40));

    // Calculate marginal contribution to risk
    let variance = PortfolioAnalytics::portfolio_variance(&weights, covariance);
    println!("Portfolio Variance: {:.6}", variance);
    println!("Portfolio Std Dev: {:.4}", variance.sqrt());

    println!("\n4.3 Tracking Error Analysis");
    println!("{}", "-".repeat(40));

    let benchmark_weights = vec![0.30, 0.25, 0.20, 0.15, 0.10]; // Market cap weights
    let tracking_error =
        PortfolioAnalytics::tracking_error(&weights, &benchmark_weights, covariance);
    let info_ratio =
        PortfolioAnalytics::information_ratio(&weights, &benchmark_weights, returns, covariance);

    println!("Tracking Error: {:.2}%", tracking_error * 100.0);
    println!("Information Ratio: {:.4}", info_ratio);

    println!("\n4.4 Downside Risk Metrics");
    println!("{}", "-".repeat(40));

    // Simulate return history
    let return_history = vec![
        -0.08, -0.03, 0.02, 0.05, 0.08, -0.02, 0.04, 0.06, -0.05, 0.03, 0.07, -0.01, 0.04, 0.02,
        -0.04, 0.05, 0.03, 0.06, -0.03, 0.04,
    ];

    let sortino = PortfolioAnalytics::sortino_ratio(&return_history, risk_free_rate, 0.0);
    let var_95 = PortfolioAnalytics::var_historical(&return_history, 0.95);
    let cvar_95 = PortfolioAnalytics::cvar_historical(&return_history, 0.95);
    let max_dd = PortfolioAnalytics::max_drawdown(&cumulative_returns(&return_history));

    println!("Sortino Ratio: {:.4}", sortino);
    println!("VaR (95%): {:.2}%", var_95 * 100.0);
    println!("CVaR (95%): {:.2}%", cvar_95 * 100.0);
    println!("Max Drawdown: {:.2}%", max_dd * 100.0);

    println!("\n4.5 Concentration Metrics");
    println!("{}", "-".repeat(40));

    let herfindahl = weights.iter().map(|w| w * w).sum::<f64>();
    let effective_n = 1.0 / herfindahl;

    println!("Herfindahl Index: {:.4}", herfindahl);
    println!("Effective Number of Assets: {:.2}", effective_n);
}

fn demo_rebalancing() {
    let current_weights = vec![0.35, 0.20, 0.18, 0.17, 0.10];
    let target_weights = vec![0.25, 0.20, 0.20, 0.20, 0.15];
    let portfolio_value = 1_000_000.0;

    println!("\nCurrent vs Target Allocation:");
    println!(
        "{:<8} {:>12} {:>12} {:>12}",
        "Asset", "Current", "Target", "Drift"
    );
    println!("{}", "-".repeat(48));

    for i in 0..5 {
        let drift = current_weights[i] - target_weights[i];
        println!(
            "Asset {:1} {:>11.1}% {:>11.1}% {:>11.1}%",
            i + 1,
            current_weights[i] * 100.0,
            target_weights[i] * 100.0,
            drift * 100.0
        );
    }

    println!("\n5.1 Full Rebalance");
    println!("{}", "-".repeat(40));

    let trades =
        PortfolioRebalancer::calculate_trades(&current_weights, &target_weights, portfolio_value);

    println!("{:<8} {:>15}", "Asset", "Trade Amount");
    println!("{}", "-".repeat(25));
    for (i, &trade) in trades.iter().enumerate() {
        let sign = if trade >= 0.0 { "+" } else { "" };
        println!("Asset {:1} {:>14}{:.0}", i + 1, sign, trade);
    }

    let total_turnover = PortfolioAnalytics::turnover(&current_weights, &target_weights);
    println!("\nTotal Turnover: {:.2}%", total_turnover * 100.0);

    println!("\n5.2 Threshold Rebalancing (2%)");
    println!("{}", "-".repeat(40));

    let threshold_weights =
        PortfolioRebalancer::threshold_rebalance(&current_weights, &target_weights, 0.02);

    println!(
        "{:<8} {:>12} {:>12} {:>10}",
        "Asset", "Current", "New", "Changed"
    );
    println!("{}", "-".repeat(44));
    for i in 0..5 {
        let changed = if (current_weights[i] - threshold_weights[i]).abs() > 1e-6 {
            "Yes"
        } else {
            "No"
        };
        println!(
            "Asset {:1} {:>11.1}% {:>11.1}% {:>10}",
            i + 1,
            current_weights[i] * 100.0,
            threshold_weights[i] * 100.0,
            changed
        );
    }

    println!("\n5.3 Optimal Rebalancing Frequency");
    println!("{}", "-".repeat(40));

    let te_cost = 0.0001; // Cost per day of tracking error
    let txn_cost_bps = 5.0; // 5 bps transaction cost
    let expected_turnover = 0.15; // 15% expected turnover per rebalance

    let optimal_days = PortfolioRebalancer::optimal_rebalancing_frequency(
        te_cost,
        txn_cost_bps,
        expected_turnover,
    );

    println!("Optimal rebalancing interval: {:.0} days", optimal_days);
    println!(
        "Approximately {} rebalances per year",
        (252.0 / optimal_days) as usize
    );
}

fn demo_efficient_frontier(
    symbols: &[String],
    returns: &[f64],
    covariance: &[Vec<f64>],
    risk_free_rate: f64,
) {
    let optimizer =
        MeanVarianceOptimizer::new(returns.to_vec(), covariance.to_vec(), symbols.to_vec())
            .unwrap()
            .with_risk_free_rate(risk_free_rate);

    println!("\nGenerating Efficient Frontier (10 points)...\n");

    let frontier = optimizer.efficient_frontier(10);

    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Return", "Volatility", "Sharpe", "Weight Dist"
    );
    println!("{}", "-".repeat(52));

    for result in frontier.iter() {
        let sharpe = result.sharpe_ratio.unwrap_or(0.0);
        let top_weight = result
            .weights
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        println!(
            "{:>11.2}% {:>11.2}% {:>12.4} {:>12.2}%",
            result.expected_return * 100.0,
            result.volatility * 100.0,
            sharpe,
            top_weight * 100.0
        );
    }

    // Find the maximum Sharpe ratio point
    let max_sharpe_point = frontier
        .iter()
        .max_by(|a, b| {
            a.sharpe_ratio
                .unwrap_or(0.0)
                .partial_cmp(&b.sharpe_ratio.unwrap_or(0.0))
                .unwrap()
        })
        .unwrap();

    println!("\nOptimal Portfolio (Max Sharpe):");
    println!("{}", "-".repeat(40));
    println!("Return: {:.2}%", max_sharpe_point.expected_return * 100.0);
    println!("Volatility: {:.2}%", max_sharpe_point.volatility * 100.0);
    println!(
        "Sharpe Ratio: {:.4}",
        max_sharpe_point.sharpe_ratio.unwrap()
    );

    println!("\nWeights:");
    for (symbol, weight) in max_sharpe_point
        .symbols
        .iter()
        .zip(max_sharpe_point.weights.iter())
    {
        if *weight > 0.01 {
            println!("  {}: {:.2}%", symbol, weight * 100.0);
        }
    }
}

// Helper functions

fn print_optimization_result(result: &OptimizationResult) {
    println!("Expected Return: {:.2}%", result.expected_return * 100.0);
    println!("Volatility: {:.2}%", result.volatility * 100.0);
    if let Some(sharpe) = result.sharpe_ratio {
        println!("Sharpe Ratio: {:.4}", sharpe);
    }
    println!("Converged: {}", result.converged);
    println!("\nWeights:");

    let top_holdings = result.top_holdings(10);
    for (symbol, weight) in top_holdings {
        if weight.abs() > 0.001 {
            println!("  {}: {:.2}%", symbol, weight * 100.0);
        }
    }
}

fn print_risk_parity_result(result: &RiskParityResult) {
    println!("Portfolio Volatility: {:.2}%", result.volatility * 100.0);
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Final Error: {:.6}", result.error);

    println!("\nWeights and Risk Contributions:");
    println!("{:<8} {:>12} {:>18}", "Symbol", "Weight", "Risk Contrib");
    println!("{}", "-".repeat(40));

    for i in 0..result.symbols.len() {
        println!(
            "{:<8} {:>11.2}% {:>17.2}%",
            result.symbols[i],
            result.weights[i] * 100.0,
            result.risk_contribution_pct[i] * 100.0
        );
    }

    println!(
        "\nMax deviation from equal risk: {:.2}%",
        result.max_risk_deviation() * 100.0
    );
}

fn print_black_litterman_result(result: &BlackLittermanResult) {
    println!("Portfolio Return: {:.2}%", result.portfolio_return * 100.0);
    println!(
        "Portfolio Volatility: {:.2}%",
        result.portfolio_volatility * 100.0
    );
    println!("Sharpe Ratio: {:.4}", result.sharpe_ratio);

    println!("\nExpected Returns:");
    println!("{:<8} {:>15}", "Symbol", "E[Return]");
    println!("{}", "-".repeat(25));
    for (symbol, &ret) in result.symbols.iter().zip(result.expected_returns.iter()) {
        println!("{:<8} {:>14.2}%", symbol, ret * 100.0);
    }

    println!("\nOptimal Weights:");
    println!("{:<8} {:>15}", "Symbol", "Weight");
    println!("{}", "-".repeat(25));
    for (symbol, &weight) in result.symbols.iter().zip(result.weights.iter()) {
        if weight > 0.01 {
            println!("{:<8} {:>14.2}%", symbol, weight * 100.0);
        }
    }
}

fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
    let mut cumulative = vec![100.0];
    for &ret in returns {
        let last = cumulative.last().unwrap();
        cumulative.push(last * (1.0 + ret));
    }
    cumulative
}
