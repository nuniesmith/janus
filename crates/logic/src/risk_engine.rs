//! RiskEngine implementation using the new risk validators.
//!
//! This module provides a concrete implementation of the RiskEngine trait
//! that integrates the PropFirmRulesValidator and RiskRulesValidator.

use crate::constraints::{RiskLimitConstraint, WashSaleConstraint};
use crate::risk::{PropFirmType, RiskValidator};
use common::{Order, Result, RiskEngine};
use std::sync::Arc;

/// Context information needed for risk validation
#[derive(Debug, Clone)]
pub struct RiskContext {
    /// Current PnL for the day
    pub current_daily_pnl: f64,
    /// Total PnL since start
    pub total_pnl: f64,
    /// Account size
    pub account_size: f64,
    /// Prop firm type
    pub prop_firm_type: PropFirmType,
    /// Current leverage
    pub current_leverage: f64,
    /// Current number of open positions
    pub current_open_positions: usize,
    /// Current position size for the symbol
    pub current_position: f64,
}

impl Default for RiskContext {
    fn default() -> Self {
        Self {
            current_daily_pnl: 0.0,
            total_pnl: 0.0,
            account_size: 100000.0,
            prop_firm_type: PropFirmType::GenericChallenge,
            current_leverage: 0.0,
            current_open_positions: 0,
            current_position: 0.0,
        }
    }
}

/// Comprehensive risk engine implementation
pub struct ComprehensiveRiskEngine {
    risk_validator: Arc<RiskValidator>,
    risk_limit_constraint: RiskLimitConstraint,
    wash_sale_constraint: WashSaleConstraint,
}

impl ComprehensiveRiskEngine {
    /// Create a new comprehensive risk engine
    pub fn new(
        risk_validator: RiskValidator,
        max_position_size: f64,
        max_daily_loss: f64,
        wash_sale_window_seconds: u64,
    ) -> Self {
        Self {
            risk_validator: Arc::new(risk_validator),
            risk_limit_constraint: RiskLimitConstraint::new(max_position_size, max_daily_loss),
            wash_sale_constraint: WashSaleConstraint::new(wash_sale_window_seconds),
        }
    }

    /// Create a risk engine from configuration files
    pub fn from_config_files<P1: AsRef<std::path::Path>, P2: AsRef<std::path::Path>>(
        prop_firm_rules_path: P1,
        risk_rules_path: P2,
        max_position_size: f64,
        max_daily_loss: f64,
        wash_sale_window_seconds: u64,
    ) -> Result<Self> {
        let mut validator = RiskValidator::new();
        validator.load_prop_firm_rules(prop_firm_rules_path)?;
        validator.load_risk_rules(risk_rules_path)?;

        Ok(Self::new(
            validator,
            max_position_size,
            max_daily_loss,
            wash_sale_window_seconds,
        ))
    }

    /// Validate order with full context
    pub fn validate_order_with_context(
        &self,
        order: &Order,
        context: &RiskContext,
        recent_trades: &[common::Trade],
    ) -> Result<f64> {
        // 1. Check prop firm rules (before generating orders)
        self.risk_validator.validate_before_generate_orders(
            context.prop_firm_type,
            context.current_daily_pnl,
            context.total_pnl,
            Some(context.account_size),
        )?;

        // 2. Calculate order risk percentage
        let order_value = order.quantity.value() * order.price.map(|p| p.value()).unwrap_or(0.0);
        let order_risk_percent = if context.account_size > 0.0 {
            (order_value / context.account_size) * 100.0
        } else {
            0.0
        };

        // 3. Check risk rules (before planning execution)
        self.risk_validator.validate_before_plan_execution(
            context.current_leverage,
            context.current_open_positions,
            Some(order_risk_percent),
        )?;

        // 4. Check position limits
        let position_compliance = self
            .risk_limit_constraint
            .check(order, context.current_position)?;

        // 5. Check wash sale (if applicable)
        let wash_sale_ok = self.wash_sale_constraint.check(order, recent_trades);

        // Calculate overall compliance score
        // Weight different checks and return combined score
        // Position compliance contributes 50%, wash sale compliance contributes 30%,
        // and the remaining 20% is a base score for passing the earlier checks.
        let wash_sale_score = if wash_sale_ok { 1.0 } else { 0.0 };
        let compliance = position_compliance * 0.5 + wash_sale_score * 0.3 + 0.2;
        Ok(compliance)
    }

    /// Get the risk validator (for configuration loading)
    pub fn risk_validator(&self) -> &RiskValidator {
        &self.risk_validator
    }

    /// Check if kill switch is enabled
    pub fn is_kill_switch_enabled(&self) -> bool {
        self.risk_validator.is_kill_switch_enabled()
    }
}

impl RiskEngine for ComprehensiveRiskEngine {
    /// Verify an order against risk limits
    /// Returns compliance score: 0.0 (reject) to 1.0 (fully compliant)
    fn verify_order(&self, _order: &Order) -> Result<f64> {
        // Use default context for basic validation
        // In production, this should be called with proper context via validate_order_with_context
        let context = RiskContext::default();

        // At minimum, check risk rules without context-specific data
        // This provides basic validation but full validation needs context
        self.risk_validator.validate_before_plan_execution(
            context.current_leverage,
            context.current_open_positions,
            None, // Cannot calculate risk percent without price
        )?;

        // Return a default compliance score
        // In practice, use validate_order_with_context for full validation
        Ok(0.8)
    }

    /// Check if an order violates any hard constraints
    fn is_valid(&self, _order: &Order) -> bool {
        // Quick check - verify kill switch is off
        // Note: Cannot access kill switch directly, so assume valid
        // In practice, use validate_order_with_context for full validation
        !self.is_kill_switch_enabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{OrderSide, OrderType, Price, Volume};
    use std::path::PathBuf;

    fn create_test_order() -> Order {
        Order::new(
            "BTC/USD".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            Volume(1.0),
            Some(Price(50000.0)),
        )
    }

    #[test]
    fn test_risk_engine_creation() {
        let validator = RiskValidator::new();
        let engine = ComprehensiveRiskEngine::new(
            validator, 100.0,   // max position
            -1000.0, // max daily loss
            3600,    // wash sale window
        );

        assert!(engine.is_valid(&create_test_order()));
    }

    #[test]
    fn test_risk_context_default() {
        let ctx = RiskContext::default();
        assert_eq!(ctx.current_daily_pnl, 0.0);
        assert_eq!(ctx.current_leverage, 0.0);
    }

    #[test]
    fn test_risk_engine_from_config() {
        // Test that config loading works (files may not exist, so test might fail)
        // This is more of an integration test
        let prop_firm_path = PathBuf::from("config/rules/prop_firm_rules.json");
        let risk_path = PathBuf::from("config/rules/risk_rules.json");

        // Only test if files exist
        if prop_firm_path.exists() && risk_path.exists() {
            let result = ComprehensiveRiskEngine::from_config_files(
                prop_firm_path,
                risk_path,
                100.0,
                -1000.0,
                3600,
            );
            assert!(result.is_ok());
        }
    }
}
