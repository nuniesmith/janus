//! HyroTrader Prop Firm Compliance Module
//!
//! This module handles all compliance checking for HyroTrader prop firm rules,
//! including risk management, stop-loss validation, drawdown tracking, and
//! prohibited practice detection.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HyroTrader account configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropFirmAccount {
    pub account_size: f64,
    pub challenge_type: ChallengeType,
    pub current_balance: f64,
    pub starting_balance: f64,
    pub daily_starting_balance: f64,
    pub peak_balance: f64,
    pub trading_days: i32,
    pub current_phase: i32,
}

/// Challenge type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeType {
    OneStep,
    TwoStep,
    Funded,
}

/// Rule violation severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Critical,
    AccountFailure,
}

/// Rule violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleViolation {
    pub rule: String,
    pub severity: ViolationSeverity,
    pub description: String,
    pub timestamp: DateTime<Utc>,
    pub value: Option<f64>,
    pub limit: Option<f64>,
}

/// Compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    pub compliant: bool,
    pub violations: Vec<RuleViolation>,
    pub warnings: Vec<String>,
    pub account_status: AccountStatus,
}

/// Account status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountStatus {
    Active,
    SoftBreach,
    Failed,
    Passed,
}

/// HyroTrader compliance checker
pub struct PropFirmValidator {
    pub account: PropFirmAccount,
    pub rules: PropFirmRules,
    pub soft_breach_count: i32,
    pub stop_loss_violations: HashMap<String, DateTime<Utc>>,
}

/// Which prop firm's rule set applies.
///
/// The validator is parameterized over this so one implementation covers every
/// firm; [`PropFirmRules::for_firm`] returns the matching preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropFirm {
    HyroTrader,
    TakeProfitTrader,
    /// Permissive baseline for an unconfigured / generic challenge.
    Generic,
}

/// Prop firm rules configuration.
///
/// Firm-agnostic: construct via [`PropFirmRules::for_firm`] or a named preset
/// (e.g. [`PropFirmRules::hyrotrader`]). `Default` is the HyroTrader preset, so
/// `PropFirmValidator::new` stays backward-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropFirmRules {
    // Risk limits
    pub daily_drawdown_percent: f64,
    pub max_drawdown_percent: f64,
    pub max_risk_per_trade_percent: f64,
    pub max_exposure_percent: f64,
    pub cumulative_position_limit_multiplier: f64,

    // Stop loss requirements
    pub stop_loss_required: bool,
    pub stop_loss_setup_time_minutes: i64,
    pub stop_loss_grace_period_minutes: i64,
    pub soft_breach_allowed: bool,
    pub soft_breach_count_limit: i32,

    // Trading requirements
    pub minimum_trade_value_percent: f64,
    pub minimum_pnl_percent: f64,
    pub minimum_trading_days: i32,

    // Profit targets
    pub profit_target_percent: f64,
    pub phase_2_profit_target_percent: Option<f64>,

    /// Symbols this firm prohibits, matched as a substring of the traded
    /// symbol. Empty = no symbol restrictions.
    #[serde(default)]
    pub prohibited_symbols: Vec<String>,
}

impl PropFirmRules {
    /// Rules for a given prop firm.
    pub fn for_firm(firm: PropFirm) -> Self {
        match firm {
            PropFirm::HyroTrader => Self::hyrotrader(),
            PropFirm::TakeProfitTrader => Self::take_profit_trader(),
            PropFirm::Generic => Self::generic(),
        }
    }

    /// HyroTrader rule set (the historical default).
    pub fn hyrotrader() -> Self {
        Self {
            daily_drawdown_percent: 5.0,
            max_drawdown_percent: 10.0,
            max_risk_per_trade_percent: 3.0,
            max_exposure_percent: 25.0,
            cumulative_position_limit_multiplier: 2.0,
            stop_loss_required: true,
            stop_loss_setup_time_minutes: 5,
            stop_loss_grace_period_minutes: 60,
            soft_breach_allowed: true,
            soft_breach_count_limit: 1,
            minimum_trade_value_percent: 5.0,
            minimum_pnl_percent: 1.0,
            minimum_trading_days: 10,
            profit_target_percent: 10.0,
            phase_2_profit_target_percent: Some(5.0),
            prohibited_symbols: vec!["EUR/USD".to_string(), "USDC".to_string()],
        }
    }

    /// TakeProfitTrader rule set. These values are reasonable starting points;
    /// reconcile against the firm's current rulebook before relying on them for
    /// enforcement.
    pub fn take_profit_trader() -> Self {
        Self {
            daily_drawdown_percent: 4.0,
            max_drawdown_percent: 6.0,
            max_risk_per_trade_percent: 2.0,
            profit_target_percent: 8.0,
            phase_2_profit_target_percent: None,
            prohibited_symbols: Vec::new(),
            ..Self::hyrotrader()
        }
    }

    /// Permissive baseline for a generic / unconfigured challenge: no symbol
    /// restrictions and stop-loss not mandatory.
    pub fn generic() -> Self {
        Self {
            stop_loss_required: false,
            prohibited_symbols: Vec::new(),
            ..Self::hyrotrader()
        }
    }
}

impl Default for PropFirmRules {
    fn default() -> Self {
        Self::hyrotrader()
    }
}

impl PropFirmValidator {
    /// Create a validator using the HyroTrader rule set (back-compat default).
    pub fn new(account_size: f64, challenge_type: ChallengeType) -> Self {
        Self::with_rules(account_size, challenge_type, PropFirmRules::default())
    }

    /// Create a validator for a specific prop firm's rule set.
    pub fn for_firm(account_size: f64, challenge_type: ChallengeType, firm: PropFirm) -> Self {
        Self::with_rules(account_size, challenge_type, PropFirmRules::for_firm(firm))
    }

    /// Create a validator with an explicit rule set.
    pub fn with_rules(
        account_size: f64,
        challenge_type: ChallengeType,
        rules: PropFirmRules,
    ) -> Self {
        let account = PropFirmAccount {
            account_size,
            challenge_type,
            current_balance: account_size,
            starting_balance: account_size,
            daily_starting_balance: account_size,
            peak_balance: account_size,
            trading_days: 0,
            current_phase: 1,
        };

        Self {
            account,
            rules,
            soft_breach_count: 0,
            stop_loss_violations: HashMap::new(),
        }
    }

    /// Validate a trade before execution
    pub fn validate_trade(
        &self,
        symbol: &str,
        entry_price: f64,
        stop_loss: f64,
        position_size: f64,
        leverage: i32,
    ) -> ComplianceResult {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();

        // Check stop loss is set
        if self.rules.stop_loss_required && stop_loss == 0.0 {
            violations.push(RuleViolation {
                rule: "stop_loss_required".to_string(),
                severity: ViolationSeverity::AccountFailure,
                description: "Stop loss must be set within 5 minutes of trade entry".to_string(),
                timestamp: Utc::now(),
                value: None,
                limit: None,
            });
        }

        // Calculate risk amount
        let risk_distance_percent = ((entry_price - stop_loss).abs() / entry_price) * 100.0;
        let risk_amount = position_size * (risk_distance_percent / 100.0) * leverage as f64;
        let risk_percent = (risk_amount / self.account.starting_balance) * 100.0;

        // Check risk per trade limit
        if risk_percent > self.rules.max_risk_per_trade_percent {
            violations.push(RuleViolation {
                rule: "max_risk_per_trade".to_string(),
                severity: ViolationSeverity::Critical,
                description: format!(
                    "Risk per trade ({:.2}%) exceeds maximum allowed ({:.2}%)",
                    risk_percent, self.rules.max_risk_per_trade_percent
                ),
                timestamp: Utc::now(),
                value: Some(risk_percent),
                limit: Some(self.rules.max_risk_per_trade_percent),
            });
        }

        // Check position size relative to account
        let position_percent = (position_size / self.account.starting_balance) * 100.0;
        if position_percent < self.rules.minimum_trade_value_percent {
            warnings.push(format!(
                "Position size ({:.2}%) is below minimum ({:.2}%) - trade may not count toward minimum days",
                position_percent, self.rules.minimum_trade_value_percent
            ));
        }

        // Check prohibited symbols (per the active firm's rule set)
        if self
            .rules
            .prohibited_symbols
            .iter()
            .any(|p| symbol.contains(p.as_str()))
        {
            violations.push(RuleViolation {
                rule: "prohibited_symbol".to_string(),
                severity: ViolationSeverity::AccountFailure,
                description: format!("Trading {} is prohibited", symbol),
                timestamp: Utc::now(),
                value: None,
                limit: None,
            });
        }

        let compliant = violations.is_empty()
            || violations.iter().all(|v| {
                matches!(
                    v.severity,
                    ViolationSeverity::Info | ViolationSeverity::Warning
                )
            });

        let account_status = if violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::AccountFailure)
        {
            AccountStatus::Failed
        } else {
            AccountStatus::Active
        };

        ComplianceResult {
            compliant,
            violations,
            warnings,
            account_status,
        }
    }

    /// Check daily drawdown limit
    pub fn check_daily_drawdown(&self, current_balance: f64) -> Option<RuleViolation> {
        let daily_loss = self.account.daily_starting_balance - current_balance;
        let daily_loss_percent = (daily_loss / self.account.starting_balance) * 100.0;

        if daily_loss_percent > self.rules.daily_drawdown_percent {
            Some(RuleViolation {
                rule: "daily_drawdown".to_string(),
                severity: ViolationSeverity::AccountFailure,
                description: format!(
                    "Daily drawdown ({:.2}%) exceeds limit ({:.2}%)",
                    daily_loss_percent, self.rules.daily_drawdown_percent
                ),
                timestamp: Utc::now(),
                value: Some(daily_loss_percent),
                limit: Some(self.rules.daily_drawdown_percent),
            })
        } else {
            None
        }
    }

    /// Check maximum drawdown limit
    pub fn check_max_drawdown(&self, current_balance: f64) -> Option<RuleViolation> {
        let max_loss = self.account.peak_balance - current_balance;
        let max_loss_percent = (max_loss / self.account.starting_balance) * 100.0;

        if max_loss_percent > self.rules.max_drawdown_percent {
            Some(RuleViolation {
                rule: "max_drawdown".to_string(),
                severity: ViolationSeverity::AccountFailure,
                description: format!(
                    "Maximum drawdown ({:.2}%) exceeds limit ({:.2}%)",
                    max_loss_percent, self.rules.max_drawdown_percent
                ),
                timestamp: Utc::now(),
                value: Some(max_loss_percent),
                limit: Some(self.rules.max_drawdown_percent),
            })
        } else {
            None
        }
    }

    /// Check if minimum trading days requirement is met
    pub fn check_minimum_trading_days(&self) -> bool {
        self.account.trading_days >= self.rules.minimum_trading_days
    }

    /// Check if profit target is met
    pub fn check_profit_target(&self) -> bool {
        let profit_percent = ((self.account.current_balance - self.account.starting_balance)
            / self.account.starting_balance)
            * 100.0;

        let target = match self.account.challenge_type {
            ChallengeType::OneStep => self.rules.profit_target_percent,
            ChallengeType::TwoStep => {
                if self.account.current_phase == 1 {
                    self.rules.profit_target_percent
                } else {
                    self.rules.phase_2_profit_target_percent.unwrap_or(5.0)
                }
            }
            ChallengeType::Funded => 0.0, // No target for funded accounts
        };

        profit_percent >= target
    }

    /// Update account balance
    pub fn update_balance(&mut self, new_balance: f64) {
        self.account.current_balance = new_balance;
        if new_balance > self.account.peak_balance {
            self.account.peak_balance = new_balance;
        }
    }

    /// Reset daily starting balance (call at UTC midnight)
    pub fn reset_daily_balance(&mut self) {
        self.account.daily_starting_balance = self.account.current_balance;
    }

    /// Increment trading day count
    pub fn increment_trading_day(&mut self) {
        self.account.trading_days += 1;
    }

    /// Check stop loss timing violation
    pub fn check_stop_loss_timing(
        &mut self,
        trade_id: &str,
        entry_time: DateTime<Utc>,
        stop_loss_set_time: Option<DateTime<Utc>>,
    ) -> Option<RuleViolation> {
        if !self.rules.stop_loss_required {
            return None;
        }

        let stop_loss_time = match stop_loss_set_time {
            Some(t) => t,
            None => {
                // No stop loss set - check if grace period applies
                if self.soft_breach_count < self.rules.soft_breach_count_limit {
                    self.soft_breach_count += 1;
                    self.stop_loss_violations
                        .insert(trade_id.to_string(), Utc::now());

                    return Some(RuleViolation {
                        rule: "stop_loss_timing".to_string(),
                        severity: ViolationSeverity::Warning,
                        description: format!(
                            "Soft breach: Stop loss not set. You have {} hour to correct this.",
                            self.rules.stop_loss_grace_period_minutes / 60
                        ),
                        timestamp: Utc::now(),
                        value: None,
                        limit: None,
                    });
                } else {
                    return Some(RuleViolation {
                        rule: "stop_loss_timing".to_string(),
                        severity: ViolationSeverity::AccountFailure,
                        description: "Stop loss not set and grace period exhausted".to_string(),
                        timestamp: Utc::now(),
                        value: None,
                        limit: None,
                    });
                }
            }
        };

        let time_diff = stop_loss_time.signed_duration_since(entry_time);
        let max_duration = Duration::minutes(self.rules.stop_loss_setup_time_minutes);

        if time_diff > max_duration {
            Some(RuleViolation {
                rule: "stop_loss_timing".to_string(),
                severity: ViolationSeverity::Critical,
                description: format!(
                    "Stop loss set {} minutes after entry, exceeds {} minute limit",
                    time_diff.num_minutes(),
                    self.rules.stop_loss_setup_time_minutes
                ),
                timestamp: Utc::now(),
                value: Some(time_diff.num_minutes() as f64),
                limit: Some(self.rules.stop_loss_setup_time_minutes as f64),
            })
        } else {
            None
        }
    }

    /// Get account summary
    pub fn get_account_summary(&self) -> AccountSummary {
        let profit = self.account.current_balance - self.account.starting_balance;
        let profit_percent = (profit / self.account.starting_balance) * 100.0;
        let daily_pnl = self.account.current_balance - self.account.daily_starting_balance;
        let daily_pnl_percent = (daily_pnl / self.account.starting_balance) * 100.0;
        let max_dd = self.account.peak_balance - self.account.current_balance;
        let max_dd_percent = (max_dd / self.account.starting_balance) * 100.0;

        let status = if !self.check_minimum_trading_days() {
            "Need more trading days".to_string()
        } else if !self.check_profit_target() {
            "Need more profit".to_string()
        } else {
            "Challenge passed!".to_string()
        };

        AccountSummary {
            account_size: self.account.account_size,
            current_balance: self.account.current_balance,
            profit,
            profit_percent,
            daily_pnl,
            daily_pnl_percent,
            max_drawdown: max_dd,
            max_drawdown_percent: max_dd_percent,
            trading_days: self.account.trading_days,
            target_trading_days: self.rules.minimum_trading_days,
            profit_target: self.rules.profit_target_percent,
            status,
        }
    }
}

/// Account summary for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSummary {
    pub account_size: f64,
    pub current_balance: f64,
    pub profit: f64,
    pub profit_percent: f64,
    pub daily_pnl: f64,
    pub daily_pnl_percent: f64,
    pub max_drawdown: f64,
    pub max_drawdown_percent: f64,
    pub trading_days: i32,
    pub target_trading_days: i32,
    pub profit_target: f64,
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_calculation() {
        let validator = PropFirmValidator::new(5000.0, ChallengeType::OneStep);

        // Test compliant trade: 3% risk
        let result = validator.validate_trade("BTC/USDT", 50000.0, 48500.0, 500.0, 1);
        assert!(result.compliant);

        // Test non-compliant trade: >3% risk
        // 20% stop distance * $1000 position * 1x leverage = $200 risk = 4% of $5000 account
        let result = validator.validate_trade("BTC/USDT", 50000.0, 40000.0, 1000.0, 1);
        assert!(!result.compliant);
    }

    #[test]
    fn test_daily_drawdown() {
        let validator = PropFirmValidator::new(5000.0, ChallengeType::OneStep);

        // 4% loss - should be OK
        let violation = validator.check_daily_drawdown(4800.0);
        assert!(violation.is_none());

        // 6% loss - should violate
        let violation = validator.check_daily_drawdown(4700.0);
        assert!(violation.is_some());
    }

    #[test]
    fn test_profit_target() {
        let mut validator = PropFirmValidator::new(5000.0, ChallengeType::OneStep);

        validator.update_balance(5400.0);
        assert!(!validator.check_profit_target()); // 8% < 10%

        validator.update_balance(5500.0);
        assert!(validator.check_profit_target()); // 10% = 10%
    }

    #[test]
    fn test_new_defaults_to_hyrotrader_preset() {
        // `new` must stay behaviourally identical to the pre-generalization
        // hardcoded HyroTrader rules (the live path relies on this).
        let v = PropFirmValidator::new(10000.0, ChallengeType::OneStep);
        assert_eq!(v.rules.daily_drawdown_percent, 5.0);
        assert_eq!(v.rules.max_drawdown_percent, 10.0);
        assert_eq!(v.rules.max_risk_per_trade_percent, 3.0);
        assert!(v.rules.stop_loss_required);
        assert!(v.rules.prohibited_symbols.iter().any(|s| s == "EUR/USD"));
        assert!(v.rules.prohibited_symbols.iter().any(|s| s == "USDC"));
    }

    #[test]
    fn test_prohibited_symbols_are_firm_specific() {
        // A low-risk, stop-loss-set trade on a USDC pair: only the prohibited
        // -symbol rule can fail it, isolating the behaviour under test.
        let hyro = PropFirmValidator::new(5000.0, ChallengeType::OneStep);
        let r = hyro.validate_trade("BTC/USDC", 50000.0, 48500.0, 500.0, 1);
        assert!(!r.compliant);
        assert!(r.violations.iter().any(|v| v.rule == "prohibited_symbol"));

        // The generic firm imposes no symbol restrictions.
        let generic =
            PropFirmValidator::for_firm(5000.0, ChallengeType::OneStep, PropFirm::Generic);
        let r = generic.validate_trade("BTC/USDC", 50000.0, 48500.0, 500.0, 1);
        assert!(r.compliant);
    }

    #[test]
    fn test_with_rules_uses_supplied_rules() {
        let mut rules = PropFirmRules::generic();
        rules.max_risk_per_trade_percent = 1.0;
        let v = PropFirmValidator::with_rules(5000.0, ChallengeType::Funded, rules);
        // ~4% risk trade (4% stop distance on a full-balance position)
        // exceeds the custom 1% per-trade cap.
        let r = v.validate_trade("BTC/USDT", 50000.0, 48000.0, 5000.0, 1);
        assert!(!r.compliant);
    }
}
