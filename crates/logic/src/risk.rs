//! Risk validation and prop firm rules enforcement.
//!
//! This module implements comprehensive risk validation including:
//! - Prop firm trading rules (daily drawdown, total loss limits)
//! - Risk rules (leverage, position limits, kill switch)
//! - Per-trade risk validation

use common::{JanusError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Prop firm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropFirmType {
    GenericChallenge,
    HyroTrader,
    TakeProfitTrader,
}

impl PropFirmType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PropFirmType::GenericChallenge => "generic_challenge",
            PropFirmType::HyroTrader => "hyrotrader",
            PropFirmType::TakeProfitTrader => "takeprofittrader",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "hyrotrader" => PropFirmType::HyroTrader,
            "takeprofittrader" => PropFirmType::TakeProfitTrader,
            _ => PropFirmType::GenericChallenge,
        }
    }
}

/// Prop firm constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropFirmConstraints {
    pub max_daily_drawdown_limit: f64,
    pub max_total_loss_limit: f64,
    pub account_size: f64,
}

/// Prop firm rules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropFirmRules {
    pub account_size: f64,
    #[serde(default)]
    pub constraints: PropFirmConstraints,
}

/// Prop firm rules validator
pub struct PropFirmRulesValidator {
    rules: HashMap<String, PropFirmRules>,
}

impl PropFirmRulesValidator {
    /// Create a new prop firm rules validator
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Load rules from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let contents = std::fs::read_to_string(path.as_ref())
            .map_err(|e| JanusError::Internal(format!("Failed to read rules file: {}", e)))?;

        let rules: HashMap<String, PropFirmRules> = serde_json::from_str(&contents)
            .map_err(|e| JanusError::Internal(format!("Failed to parse rules JSON: {}", e)))?;

        self.rules = rules;
        Ok(())
    }

    /// Add rules for a specific prop firm type (useful for testing)
    pub fn add_rules(&mut self, prop_firm_type: PropFirmType, rules: PropFirmRules) {
        self.rules
            .insert(prop_firm_type.as_str().to_string(), rules);
    }

    /// Get rules for a specific prop firm type
    pub fn get_rules(&self, prop_firm_type: PropFirmType) -> Option<&PropFirmRules> {
        self.rules.get(prop_firm_type.as_str())
    }

    /// Validate daily drawdown limit
    pub fn validate_daily_drawdown(
        &self,
        prop_firm_type: PropFirmType,
        current_pnl: f64,
        account_size: Option<f64>,
    ) -> Result<()> {
        let rules = self.get_rules(prop_firm_type).ok_or_else(|| {
            JanusError::RiskViolation(format!(
                "Prop firm rules not loaded for type: {}",
                prop_firm_type.as_str()
            ))
        })?;

        let _account = account_size.unwrap_or(rules.account_size);
        let max_drawdown = rules.constraints.max_daily_drawdown_limit;

        // Drawdown limits are positive values, but losses are negative
        // Check if the loss (negative PnL) exceeds the limit
        if current_pnl < -max_drawdown {
            return Err(JanusError::RiskViolation(format!(
                "Daily drawdown limit exceeded: {:.2} < -{:.2} (prop firm: {})",
                current_pnl,
                max_drawdown,
                prop_firm_type.as_str()
            )));
        }

        Ok(())
    }

    /// Validate total loss limit
    pub fn validate_total_loss(
        &self,
        prop_firm_type: PropFirmType,
        total_pnl: f64,
        account_size: Option<f64>,
    ) -> Result<()> {
        let rules = self.get_rules(prop_firm_type).ok_or_else(|| {
            JanusError::RiskViolation(format!(
                "Prop firm rules not loaded for type: {}",
                prop_firm_type.as_str()
            ))
        })?;

        let _account = account_size.unwrap_or(rules.account_size);
        let max_loss = rules.constraints.max_total_loss_limit;

        // Loss limits are positive values, but losses are negative
        // Check if the loss (negative PnL) exceeds the limit
        if total_pnl < -max_loss {
            return Err(JanusError::RiskViolation(format!(
                "Total loss limit exceeded: {:.2} < -{:.2} (prop firm: {})",
                total_pnl,
                max_loss,
                prop_firm_type.as_str()
            )));
        }

        Ok(())
    }

    /// Validate order against all prop firm rules
    pub fn validate_order(
        &self,
        prop_firm_type: PropFirmType,
        current_daily_pnl: f64,
        total_pnl: f64,
        account_size: Option<f64>,
    ) -> Result<()> {
        self.validate_daily_drawdown(prop_firm_type, current_daily_pnl, account_size)?;
        self.validate_total_loss(prop_firm_type, total_pnl, account_size)?;
        Ok(())
    }
}

impl Default for PropFirmRulesValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Risk rules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskRules {
    #[serde(default = "default_global_safety")]
    pub global_safety: GlobalSafety,
    #[serde(default = "default_per_trade_risk")]
    pub per_trade_risk: PerTradeRisk,
}

fn default_global_safety() -> GlobalSafety {
    GlobalSafety {
        max_leverage: 10.0,
        max_open_positions: 3,
        kill_switch_enabled: false,
    }
}

fn default_per_trade_risk() -> PerTradeRisk {
    PerTradeRisk {
        max_risk_per_trade_percent: 1.0,
    }
}

/// Global safety settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSafety {
    pub max_leverage: f64,
    pub max_open_positions: usize,
    pub kill_switch_enabled: bool,
}

/// Per-trade risk settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerTradeRisk {
    pub max_risk_per_trade_percent: f64,
}

/// Risk rules validator
pub struct RiskRulesValidator {
    rules: Option<RiskRules>,
}

impl RiskRulesValidator {
    /// Create a new risk rules validator
    pub fn new() -> Self {
        Self { rules: None }
    }

    /// Load rules from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let contents = std::fs::read_to_string(path.as_ref())
            .map_err(|e| JanusError::Internal(format!("Failed to read rules file: {}", e)))?;

        let rules: RiskRules = serde_json::from_str(&contents)
            .map_err(|e| JanusError::Internal(format!("Failed to parse rules JSON: {}", e)))?;

        self.rules = Some(rules);
        Ok(())
    }

    /// Set rules manually (useful for testing)
    pub fn set_rules(&mut self, rules: RiskRules) {
        self.rules = Some(rules);
    }

    /// Get max leverage from rules
    pub fn get_max_leverage(&self) -> Result<f64> {
        self.rules
            .as_ref()
            .map(|r| r.global_safety.max_leverage)
            .ok_or_else(|| JanusError::Internal("Risk rules not loaded".to_string()))
    }

    /// Get max open positions from rules
    pub fn get_max_open_positions(&self) -> Result<usize> {
        self.rules
            .as_ref()
            .map(|r| r.global_safety.max_open_positions)
            .ok_or_else(|| JanusError::Internal("Risk rules not loaded".to_string()))
    }

    /// Check if kill switch is enabled
    pub fn is_kill_switch_enabled(&self) -> bool {
        self.rules
            .as_ref()
            .map(|r| r.global_safety.kill_switch_enabled)
            .unwrap_or(false)
    }

    /// Get max risk per trade percentage
    pub fn get_max_risk_per_trade(&self) -> Result<f64> {
        self.rules
            .as_ref()
            .map(|r| r.per_trade_risk.max_risk_per_trade_percent)
            .ok_or_else(|| JanusError::Internal("Risk rules not loaded".to_string()))
    }

    /// Validate leverage
    pub fn validate_leverage(&self, current_leverage: f64) -> Result<()> {
        if self.rules.is_none() {
            return Err(JanusError::Internal(
                "Risk rules not loaded - cannot validate leverage".to_string(),
            ));
        }

        let max_leverage = self.get_max_leverage()?;

        if current_leverage > max_leverage {
            return Err(JanusError::RiskViolation(format!(
                "Max leverage limit exceeded: {} > {}",
                current_leverage, max_leverage
            )));
        }

        Ok(())
    }

    /// Validate number of open positions
    pub fn validate_open_positions(&self, current_open_positions: usize) -> Result<()> {
        if self.rules.is_none() {
            return Err(JanusError::Internal(
                "Risk rules not loaded - cannot validate open positions".to_string(),
            ));
        }

        let max_positions = self.get_max_open_positions()?;

        if current_open_positions >= max_positions {
            return Err(JanusError::RiskViolation(format!(
                "Max open positions limit exceeded: {} >= {}",
                current_open_positions, max_positions
            )));
        }

        Ok(())
    }

    /// Validate kill switch
    pub fn validate_kill_switch(&self) -> Result<()> {
        if self.rules.is_none() {
            return Err(JanusError::Internal(
                "Risk rules not loaded - cannot validate kill switch".to_string(),
            ));
        }

        if self.is_kill_switch_enabled() {
            return Err(JanusError::RiskViolation(
                "Kill switch is enabled - all trading is blocked".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate per-trade risk
    pub fn validate_per_trade_risk(&self, order_risk_percent: f64) -> Result<()> {
        let max_risk = self.get_max_risk_per_trade()?;

        if order_risk_percent > max_risk {
            return Err(JanusError::RiskViolation(format!(
                "Per-trade risk limit exceeded: {:.2}% > {:.2}%",
                order_risk_percent, max_risk
            )));
        }

        Ok(())
    }

    /// Validate order against all risk rules
    pub fn validate_order(
        &self,
        current_leverage: f64,
        current_open_positions: usize,
        order_risk_percent: Option<f64>,
    ) -> Result<()> {
        self.validate_kill_switch()?;
        self.validate_leverage(current_leverage)?;
        self.validate_open_positions(current_open_positions)?;

        if let Some(risk_pct) = order_risk_percent {
            self.validate_per_trade_risk(risk_pct)?;
        }

        Ok(())
    }
}

impl Default for RiskRulesValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined risk validator that validates both prop firm and risk rules
pub struct RiskValidator {
    prop_firm_validator: PropFirmRulesValidator,
    risk_validator: RiskRulesValidator,
}

impl RiskValidator {
    /// Create a new combined risk validator
    pub fn new() -> Self {
        Self {
            prop_firm_validator: PropFirmRulesValidator::new(),
            risk_validator: RiskRulesValidator::new(),
        }
    }

    /// Add prop firm rules manually (useful for testing)
    pub fn add_prop_firm_rules(&mut self, prop_firm_type: PropFirmType, rules: PropFirmRules) {
        self.prop_firm_validator.add_rules(prop_firm_type, rules);
    }

    /// Load prop firm rules from file
    pub fn load_prop_firm_rules<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.prop_firm_validator.load_from_file(path)
    }

    /// Load risk rules from file
    pub fn load_risk_rules<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.risk_validator.load_from_file(path)
    }

    /// Set risk rules manually (useful for testing)
    pub fn set_risk_rules(&mut self, rules: RiskRules) {
        self.risk_validator.set_rules(rules);
    }

    /// Validate before generating orders (prop firm rules check)
    pub fn validate_before_generate_orders(
        &self,
        prop_firm_type: PropFirmType,
        current_daily_pnl: f64,
        total_pnl: f64,
        account_size: Option<f64>,
    ) -> Result<()> {
        self.prop_firm_validator.validate_order(
            prop_firm_type,
            current_daily_pnl,
            total_pnl,
            account_size,
        )
    }

    /// Validate before planning execution (risk rules check)
    pub fn validate_before_plan_execution(
        &self,
        current_leverage: f64,
        current_open_positions: usize,
        order_risk_percent: Option<f64>,
    ) -> Result<()> {
        self.risk_validator.validate_order(
            current_leverage,
            current_open_positions,
            order_risk_percent,
        )
    }

    /// Check if kill switch is enabled
    pub fn is_kill_switch_enabled(&self) -> bool {
        self.risk_validator.is_kill_switch_enabled()
    }
}

impl Default for RiskValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prop_firm_type_conversion() {
        assert_eq!(PropFirmType::GenericChallenge.as_str(), "generic_challenge");
        assert_eq!(
            PropFirmType::from_str("hyrotrader"),
            PropFirmType::HyroTrader
        );
    }

    #[test]
    fn test_risk_rules_validator_defaults() {
        let validator = RiskRulesValidator::new();
        assert!(!validator.is_kill_switch_enabled());
    }

    #[test]
    fn test_prop_firm_validator_empty() {
        let validator = PropFirmRulesValidator::new();
        assert!(
            validator
                .get_rules(PropFirmType::GenericChallenge)
                .is_none()
        );
    }
}
