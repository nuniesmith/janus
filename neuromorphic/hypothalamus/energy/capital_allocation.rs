//! Capital allocation logic
//!
//! Part of the Hypothalamus region
//! Component: energy
//!
//! This module implements capital allocation across multiple strategies using
//! various allocation methodologies including equal weight, risk parity, and
//! momentum-based approaches.

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Allocation strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationStrategy {
    /// Equal weight across all strategies
    EqualWeight,
    /// Risk parity - allocate inversely proportional to volatility
    RiskParity,
    /// Momentum-based allocation - favor recent performers
    Momentum,
    /// Mean-variance optimization (simplified)
    MeanVariance,
    /// Custom weights provided by user
    Custom,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::EqualWeight
    }
}

/// Strategy performance metrics for allocation decisions
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Strategy identifier
    pub strategy_id: String,
    /// Recent returns (e.g., last 30 days annualized)
    pub returns: f64,
    /// Volatility (annualized standard deviation)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Current allocation (0.0 - 1.0)
    pub current_allocation: f64,
    /// Is strategy active
    pub is_active: bool,
}

impl StrategyMetrics {
    pub fn new(strategy_id: &str) -> Self {
        Self {
            strategy_id: strategy_id.to_string(),
            returns: 0.0,
            volatility: 0.15, // Default 15% vol
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            current_allocation: 0.0,
            is_active: true,
        }
    }
}

/// Allocation recommendation
#[derive(Debug, Clone)]
pub struct AllocationRecommendation {
    /// Strategy ID
    pub strategy_id: String,
    /// Recommended allocation (0.0 - 1.0)
    pub target_allocation: f64,
    /// Current allocation
    pub current_allocation: f64,
    /// Change required
    pub allocation_change: f64,
    /// Reason for the recommendation
    pub reason: String,
}

/// Configuration for capital allocation
#[derive(Debug, Clone)]
pub struct CapitalAllocationConfig {
    /// Allocation strategy to use
    pub strategy: AllocationStrategy,
    /// Minimum allocation per strategy (0.0 - 1.0)
    pub min_allocation: f64,
    /// Maximum allocation per strategy (0.0 - 1.0)
    pub max_allocation: f64,
    /// Rebalance threshold (trigger rebalance if drift exceeds this)
    pub rebalance_threshold: f64,
    /// Risk-free rate for Sharpe calculations
    pub risk_free_rate: f64,
    /// Momentum lookback period in days
    pub momentum_lookback_days: u32,
    /// Custom weights (only used with AllocationStrategy::Custom)
    pub custom_weights: HashMap<String, f64>,
    /// Maximum total exposure (can be > 1.0 for leverage)
    pub max_total_exposure: f64,
    /// Minimum cash reserve (0.0 - 1.0)
    pub min_cash_reserve: f64,
}

impl Default for CapitalAllocationConfig {
    fn default() -> Self {
        Self {
            strategy: AllocationStrategy::EqualWeight,
            min_allocation: 0.05,
            max_allocation: 0.40,
            rebalance_threshold: 0.05,
            risk_free_rate: 0.05,
            momentum_lookback_days: 30,
            custom_weights: HashMap::new(),
            max_total_exposure: 1.0,
            min_cash_reserve: 0.05,
        }
    }
}

/// Capital allocation logic
pub struct CapitalAllocation {
    /// Configuration
    config: CapitalAllocationConfig,
    /// Strategy metrics keyed by strategy_id
    strategies: HashMap<String, StrategyMetrics>,
    /// Current target allocations
    target_allocations: HashMap<String, f64>,
    /// Total capital available
    total_capital: f64,
    /// Last rebalance timestamp (epoch millis)
    last_rebalance_time: u64,
}

impl Default for CapitalAllocation {
    fn default() -> Self {
        Self::new()
    }
}

impl CapitalAllocation {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(CapitalAllocationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CapitalAllocationConfig) -> Self {
        Self {
            config,
            strategies: HashMap::new(),
            target_allocations: HashMap::new(),
            total_capital: 0.0,
            last_rebalance_time: 0,
        }
    }

    /// Set total capital available for allocation
    pub fn set_total_capital(&mut self, capital: f64) {
        self.total_capital = capital;
    }

    /// Register a strategy for allocation
    pub fn register_strategy(&mut self, metrics: StrategyMetrics) {
        self.strategies.insert(metrics.strategy_id.clone(), metrics);
    }

    /// Update metrics for a strategy
    pub fn update_strategy_metrics(&mut self, metrics: StrategyMetrics) -> Result<()> {
        if !self.strategies.contains_key(&metrics.strategy_id) {
            return Err(Error::NotFound(format!(
                "Strategy {} not registered",
                metrics.strategy_id
            )));
        }
        self.strategies.insert(metrics.strategy_id.clone(), metrics);
        Ok(())
    }

    /// Set strategy active/inactive
    pub fn set_strategy_active(&mut self, strategy_id: &str, active: bool) -> Result<()> {
        if let Some(metrics) = self.strategies.get_mut(strategy_id) {
            metrics.is_active = active;
            Ok(())
        } else {
            Err(Error::NotFound(format!(
                "Strategy {} not registered",
                strategy_id
            )))
        }
    }

    /// Calculate allocations based on the configured strategy
    pub fn calculate_allocations(&mut self) -> Result<Vec<AllocationRecommendation>> {
        let active_strategies: Vec<&StrategyMetrics> =
            self.strategies.values().filter(|s| s.is_active).collect();

        if active_strategies.is_empty() {
            return Ok(vec![]);
        }

        let raw_allocations = match self.config.strategy {
            AllocationStrategy::EqualWeight => self.calculate_equal_weight(&active_strategies),
            AllocationStrategy::RiskParity => self.calculate_risk_parity(&active_strategies),
            AllocationStrategy::Momentum => self.calculate_momentum(&active_strategies),
            AllocationStrategy::MeanVariance => self.calculate_mean_variance(&active_strategies),
            AllocationStrategy::Custom => self.calculate_custom(&active_strategies),
        };

        // Apply constraints
        let constrained = self.apply_constraints(raw_allocations);

        // Generate recommendations
        let recommendations = self.generate_recommendations(constrained);

        // Store target allocations
        for rec in &recommendations {
            self.target_allocations
                .insert(rec.strategy_id.clone(), rec.target_allocation);
        }

        Ok(recommendations)
    }

    /// Equal weight allocation
    fn calculate_equal_weight(&self, strategies: &[&StrategyMetrics]) -> HashMap<String, f64> {
        let weight = 1.0 / strategies.len() as f64;
        strategies
            .iter()
            .map(|s| (s.strategy_id.clone(), weight))
            .collect()
    }

    /// Risk parity allocation - inversely proportional to volatility
    fn calculate_risk_parity(&self, strategies: &[&StrategyMetrics]) -> HashMap<String, f64> {
        let total_inv_vol: f64 = strategies
            .iter()
            .map(|s| 1.0 / s.volatility.max(0.01))
            .sum();

        strategies
            .iter()
            .map(|s| {
                let inv_vol = 1.0 / s.volatility.max(0.01);
                (s.strategy_id.clone(), inv_vol / total_inv_vol)
            })
            .collect()
    }

    /// Momentum-based allocation - favor recent performers
    fn calculate_momentum(&self, strategies: &[&StrategyMetrics]) -> HashMap<String, f64> {
        // Use risk-adjusted returns (Sharpe)
        let sharpes: Vec<f64> = strategies.iter().map(|s| s.sharpe_ratio).collect();

        // Shift to positive values if any negative
        let min_sharpe = sharpes.iter().cloned().fold(f64::INFINITY, f64::min);
        let shift = if min_sharpe < 0.0 {
            -min_sharpe + 0.1
        } else {
            0.0
        };

        let shifted_sharpes: Vec<f64> = sharpes.iter().map(|s| s + shift).collect();
        let total: f64 = shifted_sharpes.iter().sum();

        if total <= 0.0 {
            // Fall back to equal weight
            return self.calculate_equal_weight(strategies);
        }

        strategies
            .iter()
            .zip(shifted_sharpes.iter())
            .map(|(s, &sharpe)| (s.strategy_id.clone(), sharpe / total))
            .collect()
    }

    /// Simplified mean-variance optimization
    fn calculate_mean_variance(&self, strategies: &[&StrategyMetrics]) -> HashMap<String, f64> {
        // Simplified: use returns/volatility ratio
        let ratios: Vec<f64> = strategies
            .iter()
            .map(|s| s.returns / s.volatility.max(0.01))
            .collect();

        let min_ratio = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        let shift = if min_ratio < 0.0 {
            -min_ratio + 0.1
        } else {
            0.0
        };

        let shifted: Vec<f64> = ratios.iter().map(|r| r + shift).collect();
        let total: f64 = shifted.iter().sum();

        if total <= 0.0 {
            return self.calculate_equal_weight(strategies);
        }

        strategies
            .iter()
            .zip(shifted.iter())
            .map(|(s, &ratio)| (s.strategy_id.clone(), ratio / total))
            .collect()
    }

    /// Custom weights allocation
    fn calculate_custom(&self, strategies: &[&StrategyMetrics]) -> HashMap<String, f64> {
        let mut allocations = HashMap::new();
        let mut total = 0.0;

        for s in strategies {
            let weight = self
                .config
                .custom_weights
                .get(&s.strategy_id)
                .copied()
                .unwrap_or(0.0);
            allocations.insert(s.strategy_id.clone(), weight);
            total += weight;
        }

        // Normalize if not summing to 1.0
        if total > 0.0 && (total - 1.0).abs() > 0.001 {
            for (_, v) in allocations.iter_mut() {
                *v /= total;
            }
        }

        allocations
    }

    /// Apply min/max and other constraints
    fn apply_constraints(&self, mut allocations: HashMap<String, f64>) -> HashMap<String, f64> {
        let available_exposure = self.config.max_total_exposure - self.config.min_cash_reserve;

        // First pass: apply min/max constraints
        let mut total = 0.0;
        for (_, allocation) in allocations.iter_mut() {
            *allocation = allocation.clamp(self.config.min_allocation, self.config.max_allocation);
            total += *allocation;
        }

        // Normalize to fit within available exposure
        if total > 0.0 {
            let scale = (available_exposure / total).min(1.0);
            for (_, allocation) in allocations.iter_mut() {
                *allocation *= scale;
            }
        }

        allocations
    }

    /// Generate allocation recommendations
    fn generate_recommendations(
        &self,
        allocations: HashMap<String, f64>,
    ) -> Vec<AllocationRecommendation> {
        allocations
            .into_iter()
            .map(|(strategy_id, target)| {
                let current = self
                    .strategies
                    .get(&strategy_id)
                    .map(|s| s.current_allocation)
                    .unwrap_or(0.0);
                let change = target - current;

                let reason = if change.abs() < 0.001 {
                    "No change needed".to_string()
                } else if change > 0.0 {
                    format!("Increase allocation by {:.1}%", change * 100.0)
                } else {
                    format!("Decrease allocation by {:.1}%", -change * 100.0)
                };

                AllocationRecommendation {
                    strategy_id,
                    target_allocation: target,
                    current_allocation: current,
                    allocation_change: change,
                    reason,
                }
            })
            .collect()
    }

    /// Check if rebalancing is needed
    pub fn needs_rebalance(&self) -> bool {
        for (strategy_id, target) in &self.target_allocations {
            if let Some(metrics) = self.strategies.get(strategy_id) {
                let drift = (metrics.current_allocation - target).abs();
                if drift > self.config.rebalance_threshold {
                    return true;
                }
            }
        }
        false
    }

    /// Get current allocations
    pub fn get_current_allocations(&self) -> HashMap<String, f64> {
        self.strategies
            .iter()
            .map(|(id, m)| (id.clone(), m.current_allocation))
            .collect()
    }

    /// Get target allocations
    pub fn get_target_allocations(&self) -> &HashMap<String, f64> {
        &self.target_allocations
    }

    /// Calculate dollar amounts for each strategy
    pub fn get_dollar_allocations(&self) -> HashMap<String, f64> {
        self.target_allocations
            .iter()
            .map(|(id, pct)| (id.clone(), pct * self.total_capital))
            .collect()
    }

    /// Get allocation for a specific strategy
    pub fn get_strategy_allocation(&self, strategy_id: &str) -> Option<f64> {
        self.target_allocations.get(strategy_id).copied()
    }

    /// Main processing function
    pub fn process(&mut self) -> Result<()> {
        if self.strategies.is_empty() {
            return Ok(());
        }

        // Recalculate allocations
        let _ = self.calculate_allocations()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_strategies() -> Vec<StrategyMetrics> {
        vec![
            StrategyMetrics {
                strategy_id: "momentum".to_string(),
                returns: 0.15,
                volatility: 0.20,
                sharpe_ratio: 0.75,
                max_drawdown: 0.10,
                current_allocation: 0.30,
                is_active: true,
            },
            StrategyMetrics {
                strategy_id: "mean_reversion".to_string(),
                returns: 0.10,
                volatility: 0.10,
                sharpe_ratio: 1.0,
                max_drawdown: 0.05,
                current_allocation: 0.30,
                is_active: true,
            },
            StrategyMetrics {
                strategy_id: "trend".to_string(),
                returns: 0.20,
                volatility: 0.25,
                sharpe_ratio: 0.80,
                max_drawdown: 0.15,
                current_allocation: 0.40,
                is_active: true,
            },
        ]
    }

    #[test]
    fn test_basic() {
        let instance = CapitalAllocation::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_equal_weight_allocation() {
        let mut allocator = CapitalAllocation::new();
        allocator.set_total_capital(100_000.0);

        for strategy in create_test_strategies() {
            allocator.register_strategy(strategy);
        }

        let recommendations = allocator.calculate_allocations().unwrap();

        // Should have 3 recommendations
        assert_eq!(recommendations.len(), 3);

        // Sum should be close to available exposure (1.0 - min_cash_reserve)
        let total: f64 = recommendations.iter().map(|r| r.target_allocation).sum();
        assert!(total <= allocator.config.max_total_exposure);
    }

    #[test]
    fn test_risk_parity_allocation() {
        let config = CapitalAllocationConfig {
            strategy: AllocationStrategy::RiskParity,
            ..Default::default()
        };
        let mut allocator = CapitalAllocation::with_config(config);
        allocator.set_total_capital(100_000.0);

        for strategy in create_test_strategies() {
            allocator.register_strategy(strategy);
        }

        let recommendations = allocator.calculate_allocations().unwrap();

        // Lower vol strategy should have higher allocation
        let mean_rev = recommendations
            .iter()
            .find(|r| r.strategy_id == "mean_reversion")
            .unwrap();
        let trend = recommendations
            .iter()
            .find(|r| r.strategy_id == "trend")
            .unwrap();

        assert!(mean_rev.target_allocation > trend.target_allocation);
    }

    #[test]
    fn test_momentum_allocation() {
        let config = CapitalAllocationConfig {
            strategy: AllocationStrategy::Momentum,
            ..Default::default()
        };
        let mut allocator = CapitalAllocation::with_config(config);
        allocator.set_total_capital(100_000.0);

        for strategy in create_test_strategies() {
            allocator.register_strategy(strategy);
        }

        let recommendations = allocator.calculate_allocations().unwrap();

        // Higher Sharpe should have higher allocation
        let mean_rev = recommendations
            .iter()
            .find(|r| r.strategy_id == "mean_reversion")
            .unwrap();
        let momentum = recommendations
            .iter()
            .find(|r| r.strategy_id == "momentum")
            .unwrap();

        // mean_reversion has Sharpe 1.0, momentum has 0.75
        assert!(mean_rev.target_allocation > momentum.target_allocation);
    }

    #[test]
    fn test_constraints_applied() {
        let config = CapitalAllocationConfig {
            strategy: AllocationStrategy::EqualWeight,
            min_allocation: 0.20,
            max_allocation: 0.50,
            ..Default::default()
        };
        let mut allocator = CapitalAllocation::with_config(config);
        allocator.set_total_capital(100_000.0);

        for strategy in create_test_strategies() {
            allocator.register_strategy(strategy);
        }

        let recommendations = allocator.calculate_allocations().unwrap();

        for rec in &recommendations {
            assert!(rec.target_allocation >= 0.0);
            assert!(rec.target_allocation <= 0.50);
        }
    }

    #[test]
    fn test_needs_rebalance() {
        let mut allocator = CapitalAllocation::new();
        allocator.set_total_capital(100_000.0);

        let mut strategy = StrategyMetrics::new("test");
        strategy.current_allocation = 0.30;
        allocator.register_strategy(strategy);

        let _ = allocator.calculate_allocations().unwrap();

        // Equal weight with one strategy = ~0.95 (after cash reserve)
        // Current is 0.30, so drift is > threshold
        assert!(allocator.needs_rebalance());
    }

    #[test]
    fn test_inactive_strategy_excluded() {
        let mut allocator = CapitalAllocation::new();

        let mut strategies = create_test_strategies();
        strategies[0].is_active = false;

        for strategy in strategies {
            allocator.register_strategy(strategy);
        }

        let recommendations = allocator.calculate_allocations().unwrap();

        // Should only have 2 active strategies
        assert_eq!(recommendations.len(), 2);
        assert!(!recommendations.iter().any(|r| r.strategy_id == "momentum"));
    }

    #[test]
    fn test_dollar_allocations() {
        let mut allocator = CapitalAllocation::new();
        allocator.set_total_capital(100_000.0);

        for strategy in create_test_strategies() {
            allocator.register_strategy(strategy);
        }

        let _ = allocator.calculate_allocations().unwrap();
        let dollar_amounts = allocator.get_dollar_allocations();

        let total_dollars: f64 = dollar_amounts.values().sum();
        assert!(total_dollars <= 100_000.0);
        assert!(total_dollars > 0.0);
    }

    #[test]
    fn test_custom_weights() {
        let mut custom_weights = HashMap::new();
        custom_weights.insert("momentum".to_string(), 0.50);
        custom_weights.insert("mean_reversion".to_string(), 0.30);
        custom_weights.insert("trend".to_string(), 0.20);

        let config = CapitalAllocationConfig {
            strategy: AllocationStrategy::Custom,
            custom_weights,
            ..Default::default()
        };
        let mut allocator = CapitalAllocation::with_config(config);
        allocator.set_total_capital(100_000.0);

        for strategy in create_test_strategies() {
            allocator.register_strategy(strategy);
        }

        let recommendations = allocator.calculate_allocations().unwrap();
        let momentum = recommendations
            .iter()
            .find(|r| r.strategy_id == "momentum")
            .unwrap();

        // Should respect custom weights (proportionally after constraints)
        assert!(momentum.target_allocation > 0.0);
    }

    #[test]
    fn test_update_strategy_metrics() {
        let mut allocator = CapitalAllocation::new();

        let strategy = StrategyMetrics::new("test");
        allocator.register_strategy(strategy);

        let mut updated = StrategyMetrics::new("test");
        updated.returns = 0.25;
        assert!(allocator.update_strategy_metrics(updated).is_ok());

        // Non-existent strategy should fail
        let unknown = StrategyMetrics::new("unknown");
        assert!(allocator.update_strategy_metrics(unknown).is_err());
    }

    #[test]
    fn test_set_strategy_active() {
        let mut allocator = CapitalAllocation::new();

        let strategy = StrategyMetrics::new("test");
        allocator.register_strategy(strategy);

        assert!(allocator.set_strategy_active("test", false).is_ok());
        assert!(allocator.set_strategy_active("unknown", false).is_err());
    }
}
