# Portfolio Optimization Quick Reference

## 🚀 Quick Start

### Mean-Variance (Max Sharpe)
```rust
use vision::portfolio::{MeanVarianceOptimizer, OptimizationObjective};

let optimizer = MeanVarianceOptimizer::new(returns, covariance, symbols)?
    .with_risk_free_rate(0.02);
let result = optimizer.optimize(OptimizationObjective::MaxSharpe)?;
```

### Risk Parity
```rust
use vision::portfolio::{RiskParityOptimizer, RiskParityMethod};

let optimizer = RiskParityOptimizer::new(covariance, symbols)?
    .with_method(RiskParityMethod::EqualRiskContribution);
let result = optimizer.optimize()?;
```

### Black-Litterman
```rust
use vision::portfolio::{BlackLittermanOptimizer, View};

let optimizer = BlackLittermanOptimizer::new(symbols, covariance)?
    .with_market_cap_weights(market_weights)?
    .add_view(View::absolute("AAPL".to_string(), 0.15, 0.80));
let result = optimizer.optimize()?;
```

---

## 📊 Optimization Objectives

| Objective | Code | Use Case |
|-----------|------|----------|
| Max Sharpe | `OptimizationObjective::MaxSharpe` | Best risk-adjusted return |
| Min Variance | `OptimizationObjective::MinVarianceUnconstrained` | Lowest risk |
| Target Return | `OptimizationObjective::MinVariance { target_return: 0.12 }` | Minimize risk for 12% return |
| Target Risk | `OptimizationObjective::MaxReturn { target_risk: 0.15 }` | Max return at 15% volatility |

---

## 🎯 Risk Parity Methods

| Method | Code | Description |
|--------|------|-------------|
| Equal Risk | `RiskParityMethod::EqualRiskContribution` | Each asset contributes equally |
| Inverse Vol | `RiskParityMethod::InverseVolatility` | Weight by 1/volatility |
| Risk Budget | `RiskParityMethod::RiskBudgeting` | Custom risk allocation |

```rust
// Custom risk budgets: 50% to stocks, 30% to bonds, 20% to commodities
let budget = RiskBudget::custom(symbols, vec![0.5, 0.3, 0.2])?;
```

---

## 💡 Black-Litterman Views

### Absolute View
"I believe AAPL will return 15% with 80% confidence"
```rust
View::absolute("AAPL".to_string(), 0.15, 0.80)
```

### Relative View
"I believe GOOGL will outperform MSFT by 5% with 70% confidence"
```rust
View::relative("GOOGL".to_string(), "MSFT".to_string(), 0.05, 0.70)
```

### Multiple Views
```rust
optimizer
    .add_view(View::absolute("AAPL".to_string(), 0.15, 0.80))
    .add_view(View::relative("GOOGL".to_string(), "MSFT".to_string(), 0.05, 0.70))
    .add_view(View::absolute("TSLA".to_string(), 0.25, 0.60))
```

---

## 🔒 Constraints

### Long-Only
```rust
let constraints = PortfolioConstraints::long_only();
```

### Long-Short (130/30 strategy)
```rust
let constraints = PortfolioConstraints::long_short(1.6); // 160% gross exposure
```

### Position Limits
```rust
let constraints = PortfolioConstraints::default()
    .with_weight_bounds(
        vec![0.0, 0.0, 0.0],     // Min: no shorting
        vec![0.3, 0.3, 0.4],     // Max: concentration limits
    );
```

---

## 📈 Portfolio Analytics

### Basic Metrics
```rust
use vision::portfolio::PortfolioAnalytics;

let return_val = PortfolioAnalytics::portfolio_return(&weights, &returns);
let volatility = PortfolioAnalytics::portfolio_volatility(&weights, &covariance);
let sharpe = PortfolioAnalytics::sharpe_ratio(&weights, &returns, &cov, 0.02);
```

### Tracking Metrics
```rust
let tracking_error = PortfolioAnalytics::tracking_error(&weights, &benchmark, &cov);
let info_ratio = PortfolioAnalytics::information_ratio(&weights, &bmk, &ret, &cov);
let turnover = PortfolioAnalytics::turnover(&old_weights, &new_weights);
```

### Downside Risk
```rust
let var_95 = PortfolioAnalytics::var_historical(&returns, 0.95);
let cvar_95 = PortfolioAnalytics::cvar_historical(&returns, 0.95);
let max_dd = PortfolioAnalytics::max_drawdown(&portfolio_values);
let sortino = PortfolioAnalytics::sortino_ratio(&returns, rf_rate, target);
```

---

## 🔄 Rebalancing

### Calculate Trades
```rust
use vision::portfolio::PortfolioRebalancer;

let trades = PortfolioRebalancer::calculate_trades(
    &current_weights,
    &target_weights,
    portfolio_value
);
```

### Threshold Rebalancing
Only rebalance if drift > 5%:
```rust
let new_weights = PortfolioRebalancer::threshold_rebalance(
    &current,
    &target,
    0.05  // 5% threshold
);
```

### Optimal Frequency
```rust
let days = PortfolioRebalancer::optimal_rebalancing_frequency(
    0.0001,  // Tracking error cost per day
    5.0,     // Transaction cost (bps)
    0.20     // Expected turnover per rebalance
);
```

---

## 📊 Covariance Estimation

### Sample Covariance
```rust
use vision::portfolio::CovarianceEstimator;

let cov = CovarianceEstimator::from_returns(&returns);
```

### Exponentially Weighted (RiskMetrics)
```rust
let ewma_cov = CovarianceEstimator::exponential_weighted(&returns, 0.94);
```

### Ledoit-Wolf Shrinkage
```rust
let shrunk_cov = CovarianceEstimator::ledoit_wolf_shrinkage(&sample_cov, 0.2);
```

---

## 🎨 Efficient Frontier

```rust
let frontier = optimizer.efficient_frontier(20); // 20 points

for point in frontier {
    println!("Return: {:.2}%, Vol: {:.2}%, Sharpe: {:.4}",
        point.expected_return * 100.0,
        point.volatility * 100.0,
        point.sharpe_ratio.unwrap_or(0.0)
    );
}
```

---

## 🔍 Result Inspection

### Optimization Result
```rust
println!("Return: {:.2}%", result.expected_return * 100.0);
println!("Volatility: {:.2}%", result.volatility * 100.0);
println!("Sharpe: {:.4}", result.sharpe_ratio.unwrap());

// Top holdings
for (symbol, weight) in result.top_holdings(5) {
    println!("{}: {:.2}%", symbol, weight * 100.0);
}

// Specific asset
if let Some(weight) = result.weight("AAPL") {
    println!("AAPL allocation: {:.2}%", weight * 100.0);
}
```

### Risk Parity Result
```rust
println!("Volatility: {:.2}%", result.volatility * 100.0);
println!("Converged: {}", result.converged);
println!("Max deviation: {:.2}%", result.max_risk_deviation() * 100.0);

for i in 0..result.symbols.len() {
    println!("{}: {:.1}% weight, {:.1}% risk",
        result.symbols[i],
        result.weights[i] * 100.0,
        result.risk_contribution_pct[i] * 100.0
    );
}
```

### Black-Litterman Result
```rust
println!("Portfolio Return: {:.2}%", result.portfolio_return * 100.0);
println!("Sharpe: {:.4}", result.sharpe_ratio);

// See how views adjusted returns
for (symbol, prior, posterior, delta) in result.return_adjustments() {
    println!("{}: {:.2}% → {:.2}% (Δ {:.2}%)",
        symbol, prior * 100.0, posterior * 100.0, delta * 100.0
    );
}
```

---

## ⚙️ Configuration

### Black-Litterman Config
```rust
use vision::portfolio::BlackLittermanConfig;

let config = BlackLittermanConfig {
    risk_aversion: 2.5,              // Typical: 2.5-3.5
    tau: 0.025,                       // Typical: 0.01-0.05
    use_market_cap_weights: true,
};

let optimizer = optimizer.with_config(config);
```

---

## 📋 Common Patterns

### 1. Simple Max Sharpe Portfolio
```rust
let result = MeanVarianceOptimizer::new(returns, cov, symbols)?
    .with_risk_free_rate(0.02)
    .optimize(OptimizationObjective::MaxSharpe)?;
```

### 2. Risk Parity with Analytics
```rust
let result = RiskParityOptimizer::new(cov, symbols)?
    .optimize()?;

let div_ratio = RiskParityOptimizer::new(cov, symbols)?
    .diversification_ratio(&result.weights);

println!("Diversification: {:.2}", div_ratio);
```

### 3. Black-Litterman with Market Views
```rust
let result = BlackLittermanOptimizer::new(symbols, cov)?
    .with_market_cap_weights(market_weights)?
    .add_view(View::absolute("AAPL".to_string(), 0.15, 0.80))
    .add_view(View::relative("TSLA".to_string(), "GM".to_string(), 0.10, 0.70))
    .optimize()?;
```

### 4. Rebalance with Threshold
```rust
let current = get_current_weights();
let target = optimizer.optimize(objective)?.weights;
let threshold = 0.05;

let new_weights = PortfolioRebalancer::threshold_rebalance(
    &current, &target, threshold
);

if new_weights != current {
    execute_rebalance(&current, &new_weights, portfolio_value);
}
```

---

## 🧪 Testing

Run tests:
```bash
cargo test --lib portfolio
```

Run demo:
```bash
cargo run --example portfolio_demo --release
```

---

## 📚 See Also

- **README.md**: Comprehensive guide with examples
- **portfolio_demo.rs**: Full working examples
- **week8_day4_summary.md**: Implementation details

---

## 🎓 Key Concepts

| Concept | Description |
|---------|-------------|
| **Sharpe Ratio** | (Return - RiskFree) / Volatility |
| **Risk Parity** | Each asset contributes equally to total risk |
| **Efficient Frontier** | Set of optimal portfolios for each risk level |
| **Tracking Error** | Volatility of active returns vs benchmark |
| **VaR** | Maximum loss at confidence level |
| **CVaR** | Expected loss beyond VaR threshold |

---

## ⚡ Performance Tips

1. **Cache covariance**: Don't recalculate if data unchanged
2. **Use shrinkage**: Improves stability for small samples
3. **Regularize**: Add small constant to covariance diagonal
4. **Warm start**: Use previous solution as initial guess
5. **Simplify**: Reduce assets if convergence slow

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Singular matrix | Use shrinkage or regularization |
| Slow convergence | Check covariance conditioning |
| Extreme weights | Add position limits |
| Negative weights | Use long-only constraints |
| Unstable solutions | Increase tau in Black-Litterman |