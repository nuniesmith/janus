//! Homeostasis Controller
//!
//! Maintains portfolio balance and stability by detecting deviations
//! from target setpoints and applying corrective actions.
//!
//! Inspired by biological homeostasis - the tendency to maintain
//! internal stability despite external changes.

use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Portfolio metrics that are tracked
#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub total_value: f64,
    pub cash_balance: f64,
    pub position_value: f64,
    pub leverage: f64,
    pub drawdown: f64,
    pub volatility: f64,
}

impl PortfolioMetrics {
    pub fn new() -> Self {
        Self {
            total_value: 0.0,
            cash_balance: 0.0,
            position_value: 0.0,
            leverage: 1.0,
            drawdown: 0.0,
            volatility: 0.0,
        }
    }

    /// Calculate leverage ratio
    pub fn calculate_leverage(&self) -> f64 {
        if self.total_value > 0.0 {
            self.position_value.abs() / self.total_value
        } else {
            0.0
        }
    }

    /// Calculate cash ratio
    pub fn cash_ratio(&self) -> f64 {
        if self.total_value > 0.0 {
            self.cash_balance / self.total_value
        } else {
            0.0
        }
    }
}

impl Default for PortfolioMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Corrective action recommendation
#[derive(Debug, Clone, PartialEq)]
pub enum CorrectiveAction {
    /// Reduce position sizes by given percentage
    ReducePositions(f64),
    /// Increase position sizes by given percentage
    IncreasePositions(f64),
    /// Rebalance to target allocation
    Rebalance,
    /// Raise cash by closing positions
    RaiseCash(f64),
    /// Deploy cash by opening positions
    DeployCash(f64),
    /// Reduce leverage to target
    Deleverage(f64),
    /// No action needed
    None,
}

/// Homeostasis Controller
///
/// Monitors portfolio state and recommends corrective actions to maintain
/// stability and target metrics.
pub struct HomeostasisController {
    /// Target portfolio values
    setpoints: HashMap<String, f64>,
    /// Tolerance before triggering corrections (as fraction of setpoint)
    tolerance: f64,
    /// Current portfolio metrics
    current_metrics: PortfolioMetrics,
    /// Historical peak value (for drawdown calculation)
    peak_value: f64,
    /// Risk scaling factor (reduced during adverse conditions)
    risk_scale: f64,
    /// Maximum allowed drawdown before emergency action
    max_drawdown: f64,
}

impl HomeostasisController {
    /// Create a new homeostasis controller
    ///
    /// # Arguments
    /// * `target_portfolio_value` - Target total portfolio value
    /// * `max_drawdown` - Maximum allowed drawdown (0.0-1.0)
    /// * `tolerance` - Deviation tolerance before correction (0.0-1.0)
    pub fn new(target_portfolio_value: f64, max_drawdown: f64, tolerance: f64) -> Self {
        let mut setpoints = HashMap::new();
        setpoints.insert("portfolio_value".to_string(), target_portfolio_value);
        setpoints.insert("cash_ratio".to_string(), 0.1); // 10% cash
        setpoints.insert("max_leverage".to_string(), 2.0); // 2x max
        setpoints.insert("target_volatility".to_string(), 0.15); // 15% annualized

        Self {
            setpoints,
            tolerance,
            current_metrics: PortfolioMetrics::new(),
            peak_value: target_portfolio_value,
            risk_scale: 1.0,
            max_drawdown,
        }
    }

    /// Update current portfolio metrics
    pub fn update_metrics(&mut self, metrics: PortfolioMetrics) {
        // Update peak value for drawdown calculation
        if metrics.total_value > self.peak_value {
            self.peak_value = metrics.total_value;
        }

        // Calculate drawdown
        let mut updated_metrics = metrics;
        updated_metrics.drawdown = if self.peak_value > 0.0 {
            (self.peak_value - updated_metrics.total_value) / self.peak_value
        } else {
            0.0
        };

        // Update leverage
        updated_metrics.leverage = updated_metrics.calculate_leverage();

        self.current_metrics = updated_metrics;

        // Adjust risk scale based on drawdown
        self.adjust_risk_scale();

        debug!(
            "Portfolio metrics updated - Value: {:.2}, Drawdown: {:.2}%, Leverage: {:.2}x, Risk Scale: {:.2}",
            self.current_metrics.total_value,
            self.current_metrics.drawdown * 100.0,
            self.current_metrics.leverage,
            self.risk_scale
        );
    }

    /// Adjust risk scale based on current drawdown
    pub fn adjust_risk_scale(&mut self) {
        let drawdown = self.current_metrics.drawdown;

        if drawdown >= self.max_drawdown * 0.9 {
            // Critical drawdown - minimal risk
            self.risk_scale = 0.1;
            warn!(
                "Critical drawdown: {:.2}% - Risk scale reduced to 0.1",
                drawdown * 100.0
            );
        } else if drawdown >= self.max_drawdown * 0.7 {
            // High drawdown - reduce risk significantly
            self.risk_scale = 0.25;
            warn!(
                "High drawdown: {:.2}% - Risk scale reduced to 0.25",
                drawdown * 100.0
            );
        } else if drawdown >= self.max_drawdown * 0.5 {
            // Moderate drawdown - reduce risk
            self.risk_scale = 0.5;
            info!(
                "Moderate drawdown: {:.2}% - Risk scale reduced to 0.5",
                drawdown * 100.0
            );
        } else if drawdown >= self.max_drawdown * 0.3 {
            // Minor drawdown - slightly reduce risk
            self.risk_scale = 0.75;
        } else {
            // Normal conditions - full risk
            self.risk_scale = 1.0;
        }
    }

    /// Check for deviations and recommend corrective actions
    pub fn check_and_correct(&self) -> Vec<CorrectiveAction> {
        let mut actions = Vec::new();

        // Check drawdown
        if self.current_metrics.drawdown > self.max_drawdown {
            warn!(
                "Maximum drawdown exceeded: {:.2}% > {:.2}%",
                self.current_metrics.drawdown * 100.0,
                self.max_drawdown * 100.0
            );
            actions.push(CorrectiveAction::ReducePositions(50.0));
            return actions; // Emergency - return immediately
        }

        // Check leverage
        if let Some(&max_leverage) = self.setpoints.get("max_leverage") {
            if self.current_metrics.leverage > max_leverage * (1.0 + self.tolerance) {
                let reduction = ((self.current_metrics.leverage - max_leverage)
                    / self.current_metrics.leverage)
                    * 100.0;
                warn!(
                    "Leverage too high: {:.2}x > {:.2}x - Reducing by {:.1}%",
                    self.current_metrics.leverage, max_leverage, reduction
                );
                actions.push(CorrectiveAction::Deleverage(max_leverage));
            }
        }

        // Check cash ratio
        if let Some(&target_cash_ratio) = self.setpoints.get("cash_ratio") {
            let current_cash_ratio = self.current_metrics.cash_ratio();
            let deviation = (current_cash_ratio - target_cash_ratio).abs();

            if deviation > target_cash_ratio * self.tolerance {
                if current_cash_ratio < target_cash_ratio {
                    let cash_needed =
                        (target_cash_ratio - current_cash_ratio) * self.current_metrics.total_value;
                    info!(
                        "Cash ratio low: {:.2}% < {:.2}% - Need to raise ${:.2}",
                        current_cash_ratio * 100.0,
                        target_cash_ratio * 100.0,
                        cash_needed
                    );
                    actions.push(CorrectiveAction::RaiseCash(cash_needed));
                } else {
                    let cash_to_deploy =
                        (current_cash_ratio - target_cash_ratio) * self.current_metrics.total_value;
                    info!(
                        "Cash ratio high: {:.2}% > {:.2}% - Can deploy ${:.2}",
                        current_cash_ratio * 100.0,
                        target_cash_ratio * 100.0,
                        cash_to_deploy
                    );
                    actions.push(CorrectiveAction::DeployCash(cash_to_deploy));
                }
            }
        }

        // If no specific actions, check for general rebalancing need
        if actions.is_empty() {
            let total_deviation = self.calculate_total_deviation();
            if total_deviation > self.tolerance {
                actions.push(CorrectiveAction::Rebalance);
            }
        }

        if actions.is_empty() {
            actions.push(CorrectiveAction::None);
        }

        actions
    }

    /// Calculate total deviation from all setpoints
    fn calculate_total_deviation(&self) -> f64 {
        let mut total_deviation = 0.0;
        let mut count = 0;

        // Cash ratio deviation
        if let Some(&target) = self.setpoints.get("cash_ratio") {
            let current = self.current_metrics.cash_ratio();
            total_deviation += ((current - target) / target).abs();
            count += 1;
        }

        // Leverage deviation
        if let Some(&target) = self.setpoints.get("max_leverage") {
            let current = self.current_metrics.leverage;
            if current > target {
                total_deviation += ((current - target) / target).abs();
                count += 1;
            }
        }

        if count > 0 {
            total_deviation / count as f64
        } else {
            0.0
        }
    }

    /// Get current risk scaling factor
    ///
    /// This should be used to multiply position sizes:
    /// - 1.0 = normal risk
    /// - 0.5 = half risk
    /// - 0.1 = minimal risk (emergency)
    pub fn get_risk_scale(&self) -> f64 {
        self.risk_scale
    }

    /// Get current portfolio metrics
    pub fn get_metrics(&self) -> &PortfolioMetrics {
        &self.current_metrics
    }

    /// Set a custom setpoint
    pub fn set_setpoint(&mut self, key: &str, value: f64) {
        self.setpoints.insert(key.to_string(), value);
    }

    /// Get a setpoint value
    pub fn get_setpoint(&self, key: &str) -> Option<f64> {
        self.setpoints.get(key).copied()
    }

    /// Reset peak value (e.g., after a new equity high)
    pub fn reset_peak(&mut self) {
        self.peak_value = self.current_metrics.total_value;
        info!("Peak value reset to {:.2}", self.peak_value);
    }

    /// Check if portfolio is in healthy state
    pub fn is_healthy(&self) -> bool {
        self.current_metrics.drawdown < self.max_drawdown * 0.5
            && self.current_metrics.leverage
                <= self.setpoints.get("max_leverage").copied().unwrap_or(2.0)
            && self.risk_scale >= 0.75
    }

    /// Get health score (0.0-1.0)
    pub fn health_score(&self) -> f64 {
        let mut score = 1.0;

        // Penalize drawdown
        score -= self.current_metrics.drawdown / self.max_drawdown * 0.4;

        // Penalize excessive leverage
        if let Some(&max_lev) = self.setpoints.get("max_leverage") {
            if self.current_metrics.leverage > max_lev {
                score -= (self.current_metrics.leverage - max_lev) / max_lev * 0.3;
            }
        }

        // Penalize cash ratio deviation
        if let Some(&target_cash) = self.setpoints.get("cash_ratio") {
            let cash_dev = (self.current_metrics.cash_ratio() - target_cash).abs();
            score -= cash_dev * 0.3;
        }

        score.clamp(0.0, 1.0)
    }
}

impl Default for HomeostasisController {
    fn default() -> Self {
        Self::new(100_000.0, 0.2, 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let controller = HomeostasisController::new(100_000.0, 0.2, 0.1);
        assert_eq!(controller.get_risk_scale(), 1.0);
        assert_eq!(controller.peak_value, 100_000.0);
    }

    #[test]
    fn test_metrics_update() {
        let mut controller = HomeostasisController::new(100_000.0, 0.2, 0.1);

        let metrics = PortfolioMetrics {
            total_value: 90_000.0,
            cash_balance: 10_000.0,
            position_value: 80_000.0,
            leverage: 0.0,
            drawdown: 0.0,
            volatility: 0.15,
        };

        controller.update_metrics(metrics);
        assert_eq!(controller.current_metrics.drawdown, 0.1); // 10% drawdown
    }

    #[test]
    fn test_risk_scaling() {
        // Test 1: No drawdown -> risk_scale = 1.0
        let mut controller = HomeostasisController::new(100_000.0, 0.2, 0.1);
        controller.peak_value = 100_000.0;
        controller.current_metrics.drawdown = 0.0;
        controller.adjust_risk_scale();
        assert_eq!(controller.get_risk_scale(), 1.0);

        // Test 2: Small drawdown (5% = 0.25 * max) -> risk_scale = 1.0
        controller.current_metrics.drawdown = 0.05;
        controller.adjust_risk_scale();
        assert_eq!(controller.get_risk_scale(), 1.0);

        // Test 3: Minor drawdown (6% = 0.3 * max) -> risk_scale = 0.75
        controller.current_metrics.drawdown = 0.06;
        controller.adjust_risk_scale();
        assert_eq!(controller.get_risk_scale(), 0.75);

        // Test 4: Moderate drawdown (11% = 0.55 * max) -> risk_scale = 0.5
        controller.current_metrics.drawdown = 0.11;
        controller.adjust_risk_scale();
        assert_eq!(controller.get_risk_scale(), 0.5);

        // Test 5: High drawdown (15% = 0.75 * max) -> risk_scale = 0.25
        controller.current_metrics.drawdown = 0.15;
        controller.adjust_risk_scale();
        assert_eq!(controller.get_risk_scale(), 0.25);

        // Test 6: Critical drawdown (19% = 0.95 * max) -> risk_scale = 0.1
        controller.current_metrics.drawdown = 0.19;
        controller.adjust_risk_scale();
        assert_eq!(controller.get_risk_scale(), 0.1);
    }

    #[test]
    fn test_leverage_check() {
        let mut controller = HomeostasisController::new(100_000.0, 0.2, 0.1);

        let metrics = PortfolioMetrics {
            total_value: 100_000.0,
            cash_balance: 0.0,
            position_value: 250_000.0, // 2.5x leverage
            leverage: 0.0,
            drawdown: 0.0,
            volatility: 0.15,
        };
        controller.update_metrics(metrics);

        let actions = controller.check_and_correct();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, CorrectiveAction::Deleverage(_)))
        );
    }

    #[test]
    fn test_cash_ratio_low() {
        let mut controller = HomeostasisController::new(100_000.0, 0.2, 0.1);

        let metrics = PortfolioMetrics {
            total_value: 100_000.0,
            cash_balance: 2_000.0, // Only 2% cash (target is 10%)
            position_value: 98_000.0,
            leverage: 0.98,
            drawdown: 0.0,
            volatility: 0.15,
        };
        controller.update_metrics(metrics);

        let actions = controller.check_and_correct();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, CorrectiveAction::RaiseCash(_)))
        );
    }

    #[test]
    fn test_emergency_drawdown() {
        let mut controller = HomeostasisController::new(100_000.0, 0.2, 0.1);

        let metrics = PortfolioMetrics {
            total_value: 75_000.0, // 25% drawdown (exceeds 20% max)
            cash_balance: 10_000.0,
            position_value: 65_000.0,
            leverage: 0.0,
            drawdown: 0.0,
            volatility: 0.15,
        };
        controller.update_metrics(metrics);

        let actions = controller.check_and_correct();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, CorrectiveAction::ReducePositions(_)))
        );
    }

    #[test]
    fn test_health_score() {
        let mut controller = HomeostasisController::new(100_000.0, 0.2, 0.1);

        // Healthy portfolio
        let metrics = PortfolioMetrics {
            total_value: 100_000.0,
            cash_balance: 10_000.0,
            position_value: 90_000.0,
            leverage: 0.9,
            drawdown: 0.0,
            volatility: 0.15,
        };
        controller.update_metrics(metrics);
        let score = controller.health_score();
        assert!(score > 0.9);

        // Unhealthy portfolio
        let metrics = PortfolioMetrics {
            total_value: 80_000.0,
            cash_balance: 1_000.0,
            position_value: 79_000.0,
            leverage: 0.0,
            drawdown: 0.0,
            volatility: 0.15,
        };
        controller.update_metrics(metrics);
        let score = controller.health_score();
        assert!(score < 0.7);
    }

    #[test]
    fn test_is_healthy() {
        let mut controller = HomeostasisController::new(100_000.0, 0.2, 0.1);

        let metrics = PortfolioMetrics {
            total_value: 100_000.0,
            cash_balance: 10_000.0,
            position_value: 90_000.0,
            leverage: 0.9,
            drawdown: 0.0,
            volatility: 0.15,
        };
        controller.update_metrics(metrics);
        assert!(controller.is_healthy());
    }
}
