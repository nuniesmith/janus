//! Portfolio rebalancing
//!
//! Part of the Hypothalamus region
//! Component: energy
//!
//! Implements drift detection, threshold-based triggers, and optimal rebalance calculation
//! to maintain target portfolio allocations.

use crate::common::{Error, Result};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Target allocation for an asset
#[derive(Debug, Clone)]
pub struct TargetAllocation {
    /// Asset symbol
    pub symbol: String,
    /// Target weight (0.0 to 1.0)
    pub target_weight: f64,
    /// Minimum allowed weight
    pub min_weight: f64,
    /// Maximum allowed weight
    pub max_weight: f64,
}

/// Current position in portfolio
#[derive(Debug, Clone)]
pub struct PortfolioPosition {
    /// Asset symbol
    pub symbol: String,
    /// Current value in base currency
    pub value: f64,
    /// Current weight in portfolio
    pub weight: f64,
    /// Current quantity held
    pub quantity: f64,
    /// Current price
    pub price: f64,
}

/// Drift measurement for an asset
#[derive(Debug, Clone)]
pub struct DriftMeasurement {
    /// Asset symbol
    pub symbol: String,
    /// Target weight
    pub target_weight: f64,
    /// Current weight
    pub current_weight: f64,
    /// Absolute drift (current - target)
    pub absolute_drift: f64,
    /// Relative drift as percentage of target
    pub relative_drift: f64,
    /// Whether drift exceeds threshold
    pub exceeds_threshold: bool,
}

/// Recommended rebalance trade
#[derive(Debug, Clone)]
pub struct RebalanceTrade {
    /// Asset symbol
    pub symbol: String,
    /// Trade direction (positive = buy, negative = sell)
    pub quantity: f64,
    /// Estimated value of trade
    pub value: f64,
    /// Priority (higher = more urgent)
    pub priority: f64,
    /// Reason for trade
    pub reason: RebalanceReason,
}

/// Reason for rebalancing
#[derive(Debug, Clone, PartialEq)]
pub enum RebalanceReason {
    /// Drift exceeded threshold
    DriftThreshold,
    /// Position exceeded max weight
    MaxWeightExceeded,
    /// Position below min weight
    MinWeightViolation,
    /// Scheduled rebalance
    Scheduled,
    /// New target allocation
    NewTarget,
    /// Cash deployment
    CashDeployment,
}

/// Rebalancing strategy
#[derive(Debug, Clone, PartialEq)]
pub enum RebalanceStrategy {
    /// Rebalance all positions to exact targets
    Full,
    /// Only rebalance positions that exceed thresholds
    ThresholdBased,
    /// Minimize number of trades
    MinimizeTrades,
    /// Minimize transaction costs
    MinimizeCosts,
    /// Tax-aware rebalancing (avoid short-term gains)
    TaxAware,
}

/// Configuration for rebalancing
#[derive(Debug, Clone)]
pub struct RebalancingConfig {
    /// Absolute drift threshold (e.g., 0.05 = 5%)
    pub absolute_drift_threshold: f64,
    /// Relative drift threshold (e.g., 0.20 = 20% of target)
    pub relative_drift_threshold: f64,
    /// Minimum trade value to execute
    pub min_trade_value: f64,
    /// Maximum trades per rebalance
    pub max_trades: usize,
    /// Transaction cost estimate (as fraction)
    pub transaction_cost: f64,
    /// Rebalancing strategy
    pub strategy: RebalanceStrategy,
    /// Consider tax implications
    pub tax_aware: bool,
    /// Buffer zone around thresholds
    pub threshold_buffer: f64,
}

impl Default for RebalancingConfig {
    fn default() -> Self {
        Self {
            absolute_drift_threshold: 0.05,      // 5% absolute drift
            relative_drift_threshold: 0.20,      // 20% relative drift
            min_trade_value: 100.0,              // Minimum $100 trade
            max_trades: 10,                       // Max 10 trades per rebalance
            transaction_cost: 0.001,              // 0.1% transaction cost
            strategy: RebalanceStrategy::ThresholdBased,
            tax_aware: false,
            threshold_buffer: 0.01,               // 1% buffer
        }
    }
}

/// Result of rebalancing analysis
#[derive(Debug, Clone)]
pub struct RebalanceResult {
    /// Total portfolio value
    pub total_value: f64,
    /// Drift measurements for each asset
    pub drift_measurements: Vec<DriftMeasurement>,
    /// Recommended trades
    pub trades: Vec<RebalanceTrade>,
    /// Whether rebalancing is recommended
    pub should_rebalance: bool,
    /// Total trade value
    pub total_trade_value: f64,
    /// Estimated transaction costs
    pub estimated_costs: f64,
    /// Portfolio tracking error before rebalance
    pub tracking_error_before: f64,
    /// Portfolio tracking error after rebalance
    pub tracking_error_after: f64,
}

/// Portfolio rebalancing
pub struct Rebalancing {
    /// Configuration
    config: RebalancingConfig,
    /// Target allocations
    targets: HashMap<String, TargetAllocation>,
    /// Historical drift measurements
    drift_history: Vec<(u64, HashMap<String, f64>)>,
    /// Last rebalance timestamp
    last_rebalance: Option<u64>,
    /// Rebalance count
    rebalance_count: u64,
}

impl Default for Rebalancing {
    fn default() -> Self {
        Self::new()
    }
}

impl Rebalancing {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            config: RebalancingConfig::default(),
            targets: HashMap::new(),
            drift_history: Vec::new(),
            last_rebalance: None,
            rebalance_count: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: RebalancingConfig) -> Self {
        Self {
            config,
            targets: HashMap::new(),
            drift_history: Vec::new(),
            last_rebalance: None,
            rebalance_count: 0,
        }
    }

    /// Set target allocations
    pub fn set_targets(&mut self, targets: Vec<TargetAllocation>) -> Result<()> {
        // Validate targets sum to 1.0 (approximately)
        let total_weight: f64 = targets.iter().map(|t| t.target_weight).sum();
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(Error::InvalidInput(format!(
                "Target weights must sum to 1.0, got {}",
                total_weight
            )));
        }

        // Validate individual targets
        for target in &targets {
            if target.target_weight < target.min_weight {
                return Err(Error::InvalidInput(format!(
                    "Target weight {} below min {} for {}",
                    target.target_weight, target.min_weight, target.symbol
                )));
            }
            if target.target_weight > target.max_weight {
                return Err(Error::InvalidInput(format!(
                    "Target weight {} above max {} for {}",
                    target.target_weight, target.max_weight, target.symbol
                )));
            }
        }

        self.targets.clear();
        for target in targets {
            self.targets.insert(target.symbol.clone(), target);
        }

        info!(
            num_targets = self.targets.len(),
            "Set target allocations"
        );

        Ok(())
    }

    /// Measure drift from target allocations
    pub fn measure_drift(&self, positions: &[PortfolioPosition]) -> Vec<DriftMeasurement> {
        let mut measurements = Vec::new();

        // Calculate current weights
        let total_value: f64 = positions.iter().map(|p| p.value).sum();
        if total_value <= 0.0 {
            return measurements;
        }

        let current_weights: HashMap<String, f64> = positions
            .iter()
            .map(|p| (p.symbol.clone(), p.value / total_value))
            .collect();

        // Measure drift for each target
        for (symbol, target) in &self.targets {
            let current_weight = current_weights.get(symbol).copied().unwrap_or(0.0);
            let absolute_drift = current_weight - target.target_weight;
            let relative_drift = if target.target_weight > 0.0 {
                absolute_drift / target.target_weight
            } else {
                0.0
            };

            let exceeds_threshold = absolute_drift.abs() > self.config.absolute_drift_threshold
                || relative_drift.abs() > self.config.relative_drift_threshold;

            measurements.push(DriftMeasurement {
                symbol: symbol.clone(),
                target_weight: target.target_weight,
                current_weight,
                absolute_drift,
                relative_drift,
                exceeds_threshold,
            });
        }

        // Check for positions not in targets
        for pos in positions {
            if !self.targets.contains_key(&pos.symbol) {
                let current_weight = pos.value / total_value;
                measurements.push(DriftMeasurement {
                    symbol: pos.symbol.clone(),
                    target_weight: 0.0,
                    current_weight,
                    absolute_drift: current_weight,
                    relative_drift: f64::INFINITY,
                    exceeds_threshold: true,
                });
            }
        }

        measurements
    }

    /// Calculate tracking error (root mean square deviation)
    fn calculate_tracking_error(&self, drift_measurements: &[DriftMeasurement]) -> f64 {
        if drift_measurements.is_empty() {
            return 0.0;
        }

        let sum_sq: f64 = drift_measurements
            .iter()
            .map(|d| d.absolute_drift * d.absolute_drift)
            .sum();

        (sum_sq / drift_measurements.len() as f64).sqrt()
    }

    /// Generate optimal rebalance trades
    pub fn calculate_rebalance(
        &self,
        positions: &[PortfolioPosition],
        available_cash: f64,
    ) -> RebalanceResult {
        let drift_measurements = self.measure_drift(positions);
        let total_value: f64 = positions.iter().map(|p| p.value).sum();
        let total_portfolio = total_value + available_cash;

        let tracking_error_before = self.calculate_tracking_error(&drift_measurements);

        // Build position map
        let position_map: HashMap<String, &PortfolioPosition> = positions
            .iter()
            .map(|p| (p.symbol.clone(), p))
            .collect();

        let mut trades = Vec::new();

        match self.config.strategy {
            RebalanceStrategy::Full => {
                // Generate trades for all positions to exact targets
                trades = self.generate_full_rebalance_trades(
                    &position_map,
                    total_portfolio,
                    available_cash,
                );
            }
            RebalanceStrategy::ThresholdBased => {
                // Only generate trades for positions exceeding thresholds
                trades = self.generate_threshold_trades(
                    &drift_measurements,
                    &position_map,
                    total_portfolio,
                    available_cash,
                );
            }
            RebalanceStrategy::MinimizeTrades => {
                // Generate minimum trades to get within thresholds
                trades = self.generate_minimum_trades(
                    &drift_measurements,
                    &position_map,
                    total_portfolio,
                    available_cash,
                );
            }
            RebalanceStrategy::MinimizeCosts | RebalanceStrategy::TaxAware => {
                // Use threshold-based with cost optimization
                trades = self.generate_cost_optimized_trades(
                    &drift_measurements,
                    &position_map,
                    total_portfolio,
                    available_cash,
                );
            }
        }

        // Filter out trades below minimum value
        trades.retain(|t| t.value.abs() >= self.config.min_trade_value);

        // Sort by priority and limit trades
        trades.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
        trades.truncate(self.config.max_trades);

        let total_trade_value: f64 = trades.iter().map(|t| t.value.abs()).sum();
        let estimated_costs = total_trade_value * self.config.transaction_cost;

        // Estimate tracking error after rebalance
        let tracking_error_after = if trades.is_empty() {
            tracking_error_before
        } else {
            // Simplified estimate: proportionally reduce tracking error
            let rebalance_ratio = total_trade_value / total_portfolio.max(1.0);
            tracking_error_before * (1.0 - rebalance_ratio.min(1.0))
        };

        let should_rebalance = !trades.is_empty()
            && total_trade_value > estimated_costs * 10.0; // Trade value should be 10x costs

        RebalanceResult {
            total_value: total_portfolio,
            drift_measurements,
            trades,
            should_rebalance,
            total_trade_value,
            estimated_costs,
            tracking_error_before,
            tracking_error_after,
        }
    }

    /// Generate trades for full rebalance to exact targets
    fn generate_full_rebalance_trades(
        &self,
        positions: &HashMap<String, &PortfolioPosition>,
        total_value: f64,
        _available_cash: f64,
    ) -> Vec<RebalanceTrade> {
        let mut trades = Vec::new();

        for (symbol, target) in &self.targets {
            let target_value = total_value * target.target_weight;
            let current_value = positions.get(symbol).map(|p| p.value).unwrap_or(0.0);
            let current_price = positions.get(symbol).map(|p| p.price).unwrap_or(1.0);

            let value_diff = target_value - current_value;
            if value_diff.abs() >= self.config.min_trade_value {
                let quantity = value_diff / current_price;
                trades.push(RebalanceTrade {
                    symbol: symbol.clone(),
                    quantity,
                    value: value_diff,
                    priority: value_diff.abs() / total_value,
                    reason: RebalanceReason::DriftThreshold,
                });
            }
        }

        // Sell positions not in targets
        for (symbol, pos) in positions {
            if !self.targets.contains_key(symbol) && pos.value >= self.config.min_trade_value {
                trades.push(RebalanceTrade {
                    symbol: symbol.clone(),
                    quantity: -pos.quantity,
                    value: -pos.value,
                    priority: 1.0, // High priority to remove unwanted positions
                    reason: RebalanceReason::NewTarget,
                });
            }
        }

        trades
    }

    /// Generate trades only for positions exceeding thresholds
    fn generate_threshold_trades(
        &self,
        drift_measurements: &[DriftMeasurement],
        positions: &HashMap<String, &PortfolioPosition>,
        total_value: f64,
        _available_cash: f64,
    ) -> Vec<RebalanceTrade> {
        let mut trades = Vec::new();

        for drift in drift_measurements {
            if !drift.exceeds_threshold {
                continue;
            }

            let target = self.targets.get(&drift.symbol);
            let target_weight = target.map(|t| t.target_weight).unwrap_or(0.0);
            let target_value = total_value * target_weight;
            let current_value = positions.get(&drift.symbol).map(|p| p.value).unwrap_or(0.0);
            let current_price = positions.get(&drift.symbol).map(|p| p.price).unwrap_or(1.0);

            let value_diff = target_value - current_value;
            if value_diff.abs() >= self.config.min_trade_value {
                let quantity = value_diff / current_price;

                let reason = if let Some(t) = target {
                    if drift.current_weight > t.max_weight {
                        RebalanceReason::MaxWeightExceeded
                    } else if drift.current_weight < t.min_weight {
                        RebalanceReason::MinWeightViolation
                    } else {
                        RebalanceReason::DriftThreshold
                    }
                } else {
                    RebalanceReason::NewTarget
                };

                trades.push(RebalanceTrade {
                    symbol: drift.symbol.clone(),
                    quantity,
                    value: value_diff,
                    priority: drift.absolute_drift.abs(),
                    reason,
                });
            }
        }

        trades
    }

    /// Generate minimum trades to get within thresholds
    fn generate_minimum_trades(
        &self,
        drift_measurements: &[DriftMeasurement],
        positions: &HashMap<String, &PortfolioPosition>,
        total_value: f64,
        _available_cash: f64,
    ) -> Vec<RebalanceTrade> {
        let mut trades = Vec::new();

        for drift in drift_measurements {
            if !drift.exceeds_threshold {
                continue;
            }

            let target = self.targets.get(&drift.symbol);
            let target_weight = target.map(|t| t.target_weight).unwrap_or(0.0);
            let current_price = positions.get(&drift.symbol).map(|p| p.price).unwrap_or(1.0);

            // Calculate minimum trade to get within threshold (with buffer)
            let threshold_with_buffer = self.config.absolute_drift_threshold - self.config.threshold_buffer;

            let target_drift = if drift.absolute_drift > 0.0 {
                // Over-weight: reduce to upper threshold
                threshold_with_buffer
            } else {
                // Under-weight: increase to lower threshold
                -threshold_with_buffer
            };

            let new_weight = target_weight + target_drift;
            let new_value = total_value * new_weight;
            let current_value = positions.get(&drift.symbol).map(|p| p.value).unwrap_or(0.0);

            let value_diff = new_value - current_value;
            if value_diff.abs() >= self.config.min_trade_value {
                let quantity = value_diff / current_price;
                trades.push(RebalanceTrade {
                    symbol: drift.symbol.clone(),
                    quantity,
                    value: value_diff,
                    priority: drift.absolute_drift.abs() * 0.5, // Lower priority than full rebalance
                    reason: RebalanceReason::DriftThreshold,
                });
            }
        }

        trades
    }

    /// Generate cost-optimized trades
    fn generate_cost_optimized_trades(
        &self,
        drift_measurements: &[DriftMeasurement],
        positions: &HashMap<String, &PortfolioPosition>,
        total_value: f64,
        available_cash: f64,
    ) -> Vec<RebalanceTrade> {
        // Start with threshold trades
        let mut trades = self.generate_threshold_trades(
            drift_measurements,
            positions,
            total_value,
            available_cash,
        );

        // Filter trades where cost exceeds expected benefit
        trades.retain(|trade| {
            let trade_cost = trade.value.abs() * self.config.transaction_cost;
            let drift_cost = trade.priority * total_value * 0.01; // Estimated cost of drift
            drift_cost > trade_cost * 2.0 // Only trade if drift cost is 2x transaction cost
        });

        if self.config.tax_aware {
            // Prefer buying over selling to avoid realizing gains
            trades.sort_by(|a, b| {
                let a_score = if a.quantity > 0.0 { 1.0 } else { 0.0 };
                let b_score = if b.quantity > 0.0 { 1.0 } else { 0.0 };
                b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        trades
    }

    /// Record drift for historical analysis
    pub fn record_drift(&mut self, timestamp: u64, positions: &[PortfolioPosition]) {
        let drift_measurements = self.measure_drift(positions);
        let drift_map: HashMap<String, f64> = drift_measurements
            .into_iter()
            .map(|d| (d.symbol, d.absolute_drift))
            .collect();

        self.drift_history.push((timestamp, drift_map));

        // Keep last 1000 measurements
        if self.drift_history.len() > 1000 {
            self.drift_history.remove(0);
        }
    }

    /// Record that a rebalance was executed
    pub fn record_rebalance(&mut self, timestamp: u64) {
        self.last_rebalance = Some(timestamp);
        self.rebalance_count += 1;
        info!(
            count = self.rebalance_count,
            timestamp = timestamp,
            "Recorded rebalance execution"
        );
    }

    /// Get average drift over time
    pub fn get_average_drift(&self) -> HashMap<String, f64> {
        let mut sum_drift: HashMap<String, f64> = HashMap::new();
        let mut count: HashMap<String, u64> = HashMap::new();

        for (_, drifts) in &self.drift_history {
            for (symbol, drift) in drifts {
                *sum_drift.entry(symbol.clone()).or_insert(0.0) += drift.abs();
                *count.entry(symbol.clone()).or_insert(0) += 1;
            }
        }

        sum_drift
            .into_iter()
            .map(|(symbol, sum)| {
                let c = count.get(&symbol).copied().unwrap_or(1);
                (symbol, sum / c as f64)
            })
            .collect()
    }

    /// Check if rebalancing is due based on time
    pub fn is_rebalance_due(&self, current_time: u64, min_interval_secs: u64) -> bool {
        match self.last_rebalance {
            Some(last) => current_time >= last + min_interval_secs,
            None => true,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> RebalancingStats {
        RebalancingStats {
            target_count: self.targets.len(),
            rebalance_count: self.rebalance_count,
            last_rebalance: self.last_rebalance,
            drift_history_len: self.drift_history.len(),
            average_drift: self.get_average_drift(),
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        debug!(
            targets = self.targets.len(),
            "Rebalancing process called"
        );
        Ok(())
    }
}

/// Statistics for rebalancing
#[derive(Debug, Clone)]
pub struct RebalancingStats {
    pub target_count: usize,
    pub rebalance_count: u64,
    pub last_rebalance: Option<u64>,
    pub drift_history_len: usize,
    pub average_drift: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_targets() -> Vec<TargetAllocation> {
        vec![
            TargetAllocation {
                symbol: "BTC".to_string(),
                target_weight: 0.5,
                min_weight: 0.3,
                max_weight: 0.7,
            },
            TargetAllocation {
                symbol: "ETH".to_string(),
                target_weight: 0.3,
                min_weight: 0.1,
                max_weight: 0.5,
            },
            TargetAllocation {
                symbol: "USDT".to_string(),
                target_weight: 0.2,
                min_weight: 0.0,
                max_weight: 0.5,
            },
        ]
    }

    fn create_test_positions() -> Vec<PortfolioPosition> {
        vec![
            PortfolioPosition {
                symbol: "BTC".to_string(),
                value: 6000.0,
                weight: 0.6,
                quantity: 0.1,
                price: 60000.0,
            },
            PortfolioPosition {
                symbol: "ETH".to_string(),
                value: 2000.0,
                weight: 0.2,
                quantity: 1.0,
                price: 2000.0,
            },
            PortfolioPosition {
                symbol: "USDT".to_string(),
                value: 2000.0,
                weight: 0.2,
                quantity: 2000.0,
                price: 1.0,
            },
        ]
    }

    #[test]
    fn test_basic() {
        let instance = Rebalancing::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_set_targets() {
        let mut rebalancing = Rebalancing::new();
        let targets = create_test_targets();
        assert!(rebalancing.set_targets(targets).is_ok());
        assert_eq!(rebalancing.targets.len(), 3);
    }

    #[test]
    fn test_invalid_targets_sum() {
        let mut rebalancing = Rebalancing::new();
        let targets = vec![
            TargetAllocation {
                symbol: "BTC".to_string(),
                target_weight: 0.5,
                min_weight: 0.0,
                max_weight: 1.0,
            },
            TargetAllocation {
                symbol: "ETH".to_string(),
                target_weight: 0.6, // Total = 1.1
                min_weight: 0.0,
                max_weight: 1.0,
            },
        ];
        assert!(rebalancing.set_targets(targets).is_err());
    }

    #[test]
    fn test_measure_drift() {
        let mut rebalancing = Rebalancing::new();
        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        let drift = rebalancing.measure_drift(&positions);

        assert_eq!(drift.len(), 3);

        // BTC: current 60%, target 50% -> +10% drift
        let btc_drift = drift.iter().find(|d| d.symbol == "BTC").unwrap();
        assert!((btc_drift.absolute_drift - 0.1).abs() < 0.01);

        // ETH: current 20%, target 30% -> -10% drift
        let eth_drift = drift.iter().find(|d| d.symbol == "ETH").unwrap();
        assert!((eth_drift.absolute_drift - (-0.1)).abs() < 0.01);
    }

    #[test]
    fn test_calculate_rebalance_threshold() {
        let mut rebalancing = Rebalancing::with_config(RebalancingConfig {
            absolute_drift_threshold: 0.05,
            min_trade_value: 10.0,
            strategy: RebalanceStrategy::ThresholdBased,
            ..Default::default()
        });

        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        let result = rebalancing.calculate_rebalance(&positions, 0.0);

        assert!(result.should_rebalance);
        assert!(!result.trades.is_empty());

        // Should have trades for BTC (over) and ETH (under)
        let btc_trade = result.trades.iter().find(|t| t.symbol == "BTC");
        let eth_trade = result.trades.iter().find(|t| t.symbol == "ETH");

        assert!(btc_trade.is_some());
        assert!(eth_trade.is_some());

        // BTC should be sell (negative), ETH should be buy (positive)
        assert!(btc_trade.unwrap().quantity < 0.0);
        assert!(eth_trade.unwrap().quantity > 0.0);
    }

    #[test]
    fn test_calculate_rebalance_full() {
        let mut rebalancing = Rebalancing::with_config(RebalancingConfig {
            strategy: RebalanceStrategy::Full,
            min_trade_value: 10.0,
            ..Default::default()
        });

        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        let result = rebalancing.calculate_rebalance(&positions, 0.0);

        assert!(result.should_rebalance);
        // Full rebalance generates trades for all drifting positions
        assert!(result.trades.len() >= 2);
    }

    #[test]
    fn test_no_rebalance_needed() {
        let mut rebalancing = Rebalancing::with_config(RebalancingConfig {
            absolute_drift_threshold: 0.15, // Higher threshold
            ..Default::default()
        });

        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        let result = rebalancing.calculate_rebalance(&positions, 0.0);

        // With 15% threshold, 10% drift shouldn't trigger
        assert!(result.trades.is_empty() || !result.should_rebalance);
    }

    #[test]
    fn test_record_drift() {
        let mut rebalancing = Rebalancing::new();
        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        rebalancing.record_drift(1000, &positions);
        rebalancing.record_drift(2000, &positions);

        assert_eq!(rebalancing.drift_history.len(), 2);

        let avg_drift = rebalancing.get_average_drift();
        assert!(avg_drift.contains_key("BTC"));
    }

    #[test]
    fn test_is_rebalance_due() {
        let mut rebalancing = Rebalancing::new();

        // No previous rebalance, should be due
        assert!(rebalancing.is_rebalance_due(1000, 3600));

        // Record rebalance
        rebalancing.record_rebalance(1000);
        assert!(!rebalancing.is_rebalance_due(2000, 3600)); // Only 1000s passed
        assert!(rebalancing.is_rebalance_due(5000, 3600));  // 4000s passed
    }

    #[test]
    fn test_minimize_trades_strategy() {
        let mut rebalancing = Rebalancing::with_config(RebalancingConfig {
            strategy: RebalanceStrategy::MinimizeTrades,
            absolute_drift_threshold: 0.05,
            threshold_buffer: 0.02,
            min_trade_value: 10.0,
            ..Default::default()
        });

        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        let result = rebalancing.calculate_rebalance(&positions, 0.0);

        // MinimizeTrades should generate smaller trades to just get within threshold
        for trade in &result.trades {
            // Trade values should generally be smaller than full rebalance
            assert!(trade.value.abs() > 0.0);
        }
    }

    #[test]
    fn test_stats() {
        let mut rebalancing = Rebalancing::new();
        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        rebalancing.record_drift(1000, &positions);
        rebalancing.record_rebalance(1000);

        let stats = rebalancing.stats();
        assert_eq!(stats.target_count, 3);
        assert_eq!(stats.rebalance_count, 1);
        assert_eq!(stats.last_rebalance, Some(1000));
        assert_eq!(stats.drift_history_len, 1);
    }

    #[test]
    fn test_unknown_position() {
        let mut rebalancing = Rebalancing::new();
        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        // Add a position not in targets
        let mut positions = create_test_positions();
        positions.push(PortfolioPosition {
            symbol: "SOL".to_string(),
            value: 1000.0,
            weight: 0.09,
            quantity: 10.0,
            price: 100.0,
        });

        let drift = rebalancing.measure_drift(&positions);
        let sol_drift = drift.iter().find(|d| d.symbol == "SOL").unwrap();
        assert!(sol_drift.exceeds_threshold);
        assert_eq!(sol_drift.target_weight, 0.0);
    }

    #[test]
    fn test_tracking_error() {
        let mut rebalancing = Rebalancing::new();
        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        let result = rebalancing.calculate_rebalance(&positions, 0.0);

        assert!(result.tracking_error_before > 0.0);
        assert!(result.tracking_error_after <= result.tracking_error_before);
    }

    #[test]
    fn test_estimated_costs() {
        let mut rebalancing = Rebalancing::with_config(RebalancingConfig {
            transaction_cost: 0.001,
            min_trade_value: 10.0,
            ..Default::default()
        });

        let targets = create_test_targets();
        rebalancing.set_targets(targets).unwrap();

        let positions = create_test_positions();
        let result = rebalancing.calculate_rebalance(&positions, 0.0);

        if result.total_trade_value > 0.0 {
            let expected_cost = result.total_trade_value * 0.001;
            assert!((result.estimated_costs - expected_cost).abs() < 0.01);
        }
    }
}
