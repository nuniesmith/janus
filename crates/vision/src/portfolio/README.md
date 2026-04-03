# Portfolio Optimization Module

This module provides comprehensive portfolio optimization tools for quantitative trading, including modern portfolio theory implementations, risk parity, and Bayesian approaches.

## Features

### 1. Mean-Variance Optimization (Markowitz)
Classic portfolio optimization based on expected returns and covariance:
- **Maximum Sharpe Ratio**: Find the portfolio with the best risk-adjusted returns
- **Minimum Variance**: Construct the lowest-risk portfolio
- **Target Return**: Minimize risk for a specific return target
- **Efficient Frontier**: Generate the full set of optimal portfolios
- **Constraints**: Long-only, leverage limits, position size bounds

### 2. Risk Parity
Equal risk contribution approach for more balanced portfolios:
- **Equal Risk Contribution (ERC)**: Each asset contributes equally to portfolio risk
- **Inverse Volatility**: Simple volatility-weighted allocation
- **Risk Budgeting**: Custom risk allocation across assets
- **Diversification Ratio**: Measure portfolio diversification benefit

### 3. Black-Litterman Model
Bayesian framework combining market equilibrium with investor views:
- **Market Equilibrium**: Calculate implied returns from market cap weights
- **Absolute Views**: Express beliefs about specific asset returns
- **Relative Views**: Express beliefs about relative performance
- **Confidence Levels**: Weight views by conviction level
- **Posterior Distribution**: Blend prior (market) and views optimally

### 4. Portfolio Analytics
Comprehensive risk and performance metrics:
- **Risk-Return Metrics**: Sharpe ratio, Sortino ratio, Information ratio
- **Risk Decomposition**: Variance, volatility, beta, tracking error
- **Downside Risk**: VaR, CVaR, maximum drawdown
- **Concentration**: Herfindahl index, effective number of assets

### 5. Rebalancing Utilities
Tools for portfolio maintenance:
- **Trade Calculation**: Compute trades needed to reach target weights
- **Threshold Rebalancing**: Only rebalance when drift exceeds threshold
- **Optimal Frequency**: Balance tracking error vs transaction costs

### 6. Covariance Estimation
Advanced covariance matrix estimation:
- **Sample Covariance**: Standard historical estimation
- **Exponential Weighting**: Recent data weighted more heavily (RiskMetrics)
- **Ledoit-Wolf Shrinkage**: Reduce estimation error via shrinkage

## Quick Start

### Mean-Variance Optimization

```rust
use vision::portfolio::{MeanVarianceOptimizer, OptimizationObjective};

// Define assets
let returns = vec![0.10, 0.12, 0.08];
let covariance = vec![
    vec![0.04, 0.01, 0.02],
    vec![0.01, 0.09, 0.01],
    vec![0.02, 0.01, 0.16],
];
let symbols = vec!["AAPL", "GOOGL", "MSFT"]
    .iter()
    .map(|s| s.to_string())
    .collect();

// Create optimizer
let optimizer = MeanVarianceOptimizer::new(returns, covariance, symbols)
    .unwrap()
    .with_risk_free_rate(0.02);

// Maximize Sharpe ratio
let result = optimizer.optimize(OptimizationObjective::MaxSharpe).unwrap();

println!("Expected Return: {:.2}%", result.expected_return * 100.0);
println!("Volatility: {:.2}%", result.volatility * 100.0);
println!("Sharpe Ratio: {:.4}", result.sharpe_ratio.unwrap());

for (symbol, weight) in result.symbols.iter().zip(result.weights.iter()) {
    println!("{}: {:.2}%", symbol, weight * 100.0);
}
```

### Risk Parity

```rust
use vision::portfolio::{RiskParityOptimizer, RiskParityMethod};

let optimizer = RiskParityOptimizer::new(covariance, symbols)
    .unwrap()
    .with_method(RiskParityMethod::EqualRiskContribution);

let result = optimizer.optimize().unwrap();

for i in 0..result.symbols.len() {
    println!(
        "{}: {:.2}% weight, {:.2}% risk contribution",
        result.symbols[i],
        result.weights[i] * 100.0,
        result.risk_contribution_pct[i] * 100.0
    );
}
```

### Black-Litterman

```rust
use vision::portfolio::{BlackLittermanOptimizer, View};

// Start with market equilibrium
let market_weights = vec![0.4, 0.3, 0.3];

let optimizer = BlackLittermanOptimizer::new(symbols, covariance)
    .unwrap()
    .with_market_cap_weights(market_weights)
    .unwrap()
    .with_risk_free_rate(0.02);

// Add investor views
let with_views = optimizer
    .add_view(View::absolute("AAPL".to_string(), 0.15, 0.80))
    .add_view(View::relative(
        "GOOGL".to_string(),
        "MSFT".to_string(),
        0.05,
        0.70,
    ));

let result = with_views.optimize().unwrap();

println!("Portfolio Return: {:.2}%", result.portfolio_return * 100.0);
println!("Portfolio Volatility: {:.2}%", result.portfolio_volatility * 100.0);
println!("Sharpe Ratio: {:.4}", result.sharpe_ratio);

// See how views adjusted the returns
for (symbol, prior, posterior, delta) in result.return_adjustments() {
    println!(
        "{}: {:.2}% → {:.2}% (Δ {:.2}%)",
        symbol,
        prior * 100.0,
        posterior * 100.0,
        delta * 100.0
    );
}
```

## Advanced Usage

### Constraints

```rust
use vision::portfolio::PortfolioConstraints;

// Long-only portfolio
let constraints = PortfolioConstraints::long_only();

// Long-short with 130/30 strategy
let constraints = PortfolioConstraints::long_short(1.6);

// Custom position limits
let constraints = PortfolioConstraints::default()
    .with_weight_bounds(
        vec![0.0, 0.0, 0.0],      // Min weights
        vec![0.3, 0.3, 0.4],      // Max weights (concentration limits)
    );

let optimizer = MeanVarianceOptimizer::new(returns, covariance, symbols)
    .unwrap()
    .with_constraints(constraints);
```

### Efficient Frontier

```rust
let optimizer = MeanVarianceOptimizer::new(returns, covariance, symbols)
    .unwrap()
    .with_risk_free_rate(0.02);

let frontier = optimizer.efficient_frontier(20);

for point in frontier {
    println!(
        "Return: {:.2}%, Volatility: {:.2}%, Sharpe: {:.4}",
        point.expected_return * 100.0,
        point.volatility * 100.0,
        point.sharpe_ratio.unwrap_or(0.0)
    );
}
```

### Custom Risk Budgets

```rust
use vision::portfolio::RiskBudget;

// Allocate 50% risk to equities, 30% to bonds, 20% to alternatives
let budget = RiskBudget::custom(
    symbols.clone(),
    vec![0.50, 0.30, 0.20],
).unwrap();

let optimizer = RiskParityOptimizer::new(covariance, symbols)
    .unwrap()
    .with_method(RiskParityMethod::RiskBudgeting)
    .with_risk_budget(budget);

let result = optimizer.optimize().unwrap();
```

### Portfolio Analytics

```rust
use vision::portfolio::PortfolioAnalytics;

let weights = vec![0.4, 0.3, 0.3];

// Basic metrics
let return_val = PortfolioAnalytics::portfolio_return(&weights, &returns);
let volatility = PortfolioAnalytics::portfolio_volatility(&weights, &covariance);
let sharpe = PortfolioAnalytics::sharpe_ratio(&weights, &returns, &covariance, 0.02);

// Tracking error vs benchmark
let benchmark = vec![0.5, 0.3, 0.2];
let tracking_error = PortfolioAnalytics::tracking_error(&weights, &benchmark, &covariance);
let info_ratio = PortfolioAnalytics::information_ratio(
    &weights,
    &benchmark,
    &returns,
    &covariance,
);

// Downside risk
let return_history = vec![0.02, -0.01, 0.03, -0.02, 0.01];
let var_95 = PortfolioAnalytics::var_historical(&return_history, 0.95);
let cvar_95 = PortfolioAnalytics::cvar_historical(&return_history, 0.95);

println!("VaR (95%): {:.2}%", var_95 * 100.0);
println!("CVaR (95%): {:.2}%", cvar_95 * 100.0);
```

### Rebalancing

```rust
use vision::portfolio::PortfolioRebalancer;

let current = vec![0.5, 0.3, 0.2];
let target = vec![0.4, 0.3, 0.3];
let portfolio_value = 1_000_000.0;

// Calculate exact trades
let trades = PortfolioRebalancer::calculate_trades(&current, &target, portfolio_value);

// Only rebalance if drift exceeds 5%
let new_weights = PortfolioRebalancer::threshold_rebalance(&current, &target, 0.05);

// Determine optimal rebalancing frequency
let te_cost = 0.0001;           // Daily tracking error cost
let txn_cost = 5.0;             // 5 bps transaction cost
let expected_turnover = 0.20;   // 20% expected turnover

let optimal_days = PortfolioRebalancer::optimal_rebalancing_frequency(
    te_cost,
    txn_cost,
    expected_turnover,
);

println!("Optimal rebalancing: every {} days", optimal_days as u32);
```

### Covariance Estimation

```rust
use vision::portfolio::CovarianceEstimator;

// Historical returns: [asset][time]
let returns = vec![
    vec![0.01, 0.02, -0.01, 0.03],
    vec![0.02, 0.01, 0.00, 0.02],
    vec![-0.01, 0.03, 0.02, 0.01],
];

// Sample covariance
let cov = CovarianceEstimator::from_returns(&returns);

// Exponentially weighted (λ = 0.94 for RiskMetrics)
let ewma_cov = CovarianceEstimator::exponential_weighted(&returns, 0.94);

// Ledoit-Wolf shrinkage (20% shrinkage)
let shrunk_cov = CovarianceEstimator::ledoit_wolf_shrinkage(&cov, 0.2);
```

## Integration with Other Modules

### With Backtesting

```rust
use vision::backtest::BacktestSimulation;
use vision::portfolio::MeanVarianceOptimizer;

// In your trading strategy
fn rebalance_portfolio(&mut self, current_prices: &[f64]) {
    // Estimate expected returns (e.g., from ML model)
    let expected_returns = self.forecast_returns(current_prices);
    
    // Estimate covariance (e.g., rolling window)
    let covariance = self.estimate_covariance();
    
    // Optimize
    let optimizer = MeanVarianceOptimizer::new(
        expected_returns,
        covariance,
        self.symbols.clone(),
    ).unwrap();
    
    let result = optimizer.optimize(OptimizationObjective::MaxSharpe).unwrap();
    
    // Update positions
    self.target_weights = result.weights;
}
```

### With Risk Management

```rust
use vision::risk::RiskManager;
use vision::portfolio::{PortfolioAnalytics, OptimizationResult};

fn check_portfolio_risk(result: &OptimizationResult, risk_manager: &RiskManager) -> bool {
    // Check volatility limit
    if result.volatility > 0.20 {
        return false;
    }
    
    // Check concentration
    let max_weight = result.weights.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    
    if *max_weight > 0.30 {
        return false;
    }
    
    // Check with risk manager
    risk_manager.validate_weights(&result.weights)
}
```

### With Execution

```rust
use vision::execution::{ExecutionManager, OrderRequest, Side};
use vision::portfolio::PortfolioRebalancer;

fn execute_rebalance(
    current: &[f64],
    target: &[f64],
    prices: &[f64],
    portfolio_value: f64,
    exec_manager: &mut ExecutionManager,
) {
    let trades = PortfolioRebalancer::calculate_trades(current, target, portfolio_value);
    
    for (i, trade_value) in trades.iter().enumerate() {
        if trade_value.abs() < 1000.0 {
            continue; // Skip tiny trades
        }
        
        let quantity = trade_value.abs() / prices[i];
        let side = if *trade_value > 0.0 { Side::Buy } else { Side::Sell };
        
        let order = OrderRequest::market(
            format!("SYMBOL_{}", i),
            side,
            quantity,
        );
        
        exec_manager.submit_order(order).unwrap();
    }
}
```

## Performance Considerations

### Optimization Convergence

The optimizers use iterative algorithms. If convergence is slow:

1. **Check data quality**: Ensure covariance matrix is positive definite
2. **Adjust learning rates**: Default parameters work for most cases
3. **Set reasonable constraints**: Overly tight constraints can prevent convergence
4. **Use good initial guesses**: Risk parity starts with inverse volatility weights

### Numerical Stability

For improved numerical stability:

1. **Regularize covariance**: Add small constant to diagonal
2. **Use shrinkage**: Ledoit-Wolf shrinkage reduces estimation error
3. **Scale returns**: Work in percentage rather than decimal returns
4. **Check condition number**: High condition numbers indicate ill-conditioned problems

### Computational Complexity

- **Mean-Variance**: O(n³) for matrix inversion, O(n²×iter) for optimization
- **Risk Parity**: O(n²×iter) where iter typically 50-200
- **Black-Litterman**: O(n³ + k³) where k = number of views
- **Efficient Frontier**: O(n³ + points × n²×iter)

For large portfolios (n > 100), consider:
- Sparse covariance matrices
- Factor models instead of full covariance
- Approximate optimization methods

## Mathematical Background

### Mean-Variance Optimization

Minimize portfolio variance subject to target return:

```
min  w^T Σ w
s.t. μ^T w = r_target
     1^T w = 1
     w >= 0  (if long-only)
```

Where:
- w: portfolio weights
- Σ: covariance matrix
- μ: expected returns
- r_target: target return

### Risk Parity

Each asset contributes equally to portfolio risk:

```
RUSTCODE_i = w_i × (Σw)_i / σ_p = 1/n

Where RUSTCODE_i is the risk contribution of asset i
```

Solved iteratively using gradient descent.

### Black-Litterman

Posterior expected returns combining prior (equilibrium) and views:

```
E[R] = [(τΣ)^-1 + P^T Ω^-1 P]^-1 [(τΣ)^-1 Π + P^T Ω^-1 Q]
```

Where:
- Π: equilibrium returns (δ × Σ × w_mkt)
- P: pick matrix (view specification)
- Q: view returns
- Ω: view uncertainty
- τ: scaling factor (typically 0.01-0.05)

## Testing

Run the portfolio optimization tests:

```bash
cargo test --lib portfolio
```

Run the comprehensive demo:

```bash
cargo run --example portfolio_demo --release
```

## References

1. **Mean-Variance**: Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*.
2. **Risk Parity**: Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios".
3. **Black-Litterman**: Black, F. & Litterman, R. (1992). "Global Portfolio Optimization". *Financial Analysts Journal*.
4. **Shrinkage**: Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix".

## License

MIT OR Apache-2.0