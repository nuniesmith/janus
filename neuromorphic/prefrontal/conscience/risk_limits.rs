//! Risk limits enforcement — trading constraint guardrails
//!
//! Part of the Prefrontal region
//! Component: conscience
//!
//! Enforces hard and soft risk limits across multiple dimensions:
//! - Maximum position size (per-instrument and portfolio-wide)
//! - Maximum drawdown (from peak equity)
//! - Maximum daily loss
//! - Maximum gross/net exposure
//! - Maximum number of concurrent positions
//! - Maximum correlation exposure
//!
//! Each limit can be configured as a hard limit (reject) or soft limit (warn).
//! The module tracks current state and returns violation signals that upstream
//! modules (kill switch, circuit breakers) can act on.

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Whether a limit is enforced as hard (reject) or soft (warn only)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitMode {
    /// Hard limit — violations cause order rejection
    Hard,
    /// Soft limit — violations produce warnings but allow the action
    Soft,
    /// Disabled — limit is not checked
    Disabled,
}

impl Default for LimitMode {
    fn default() -> Self {
        LimitMode::Hard
    }
}

/// Configuration for a single limit dimension
#[derive(Debug, Clone)]
pub struct LimitRule {
    /// The threshold value
    pub threshold: f64,
    /// Warning threshold (triggers a soft warning before the hard limit)
    pub warning_threshold: f64,
    /// Enforcement mode
    pub mode: LimitMode,
    /// Human-readable label for this rule
    pub label: String,
}

impl LimitRule {
    /// Create a new hard limit
    pub fn hard(label: impl Into<String>, threshold: f64) -> Self {
        Self {
            threshold,
            warning_threshold: threshold * 0.8,
            mode: LimitMode::Hard,
            label: label.into(),
        }
    }

    /// Create a new soft limit
    pub fn soft(label: impl Into<String>, threshold: f64) -> Self {
        Self {
            threshold,
            warning_threshold: threshold * 0.8,
            mode: LimitMode::Soft,
            label: label.into(),
        }
    }

    /// Create a disabled limit
    pub fn disabled(label: impl Into<String>) -> Self {
        Self {
            threshold: f64::MAX,
            warning_threshold: f64::MAX,
            mode: LimitMode::Disabled,
            label: label.into(),
        }
    }

    /// Set a custom warning threshold
    pub fn with_warning(mut self, warning: f64) -> Self {
        self.warning_threshold = warning;
        self
    }
}

/// Configuration for all risk limits
#[derive(Debug, Clone)]
pub struct RiskLimitsConfig {
    /// Maximum position size per instrument (in base units)
    pub max_position_size: LimitRule,
    /// Maximum total portfolio position size (sum of absolute values)
    pub max_portfolio_size: LimitRule,
    /// Maximum drawdown from peak equity as a fraction (e.g., 0.10 = 10%)
    pub max_drawdown: LimitRule,
    /// Maximum daily loss as a fraction of starting equity (e.g., 0.02 = 2%)
    pub max_daily_loss: LimitRule,
    /// Maximum gross exposure as a multiple of equity (e.g., 3.0 = 3x leverage)
    pub max_gross_exposure: LimitRule,
    /// Maximum net exposure as a fraction of equity (e.g., 1.0 = 100% net long/short)
    pub max_net_exposure: LimitRule,
    /// Maximum number of concurrent open positions
    pub max_open_positions: LimitRule,
    /// Maximum single-instrument exposure as fraction of portfolio (concentration limit)
    pub max_concentration: LimitRule,
    /// Maximum loss per single trade as fraction of equity
    pub max_single_trade_loss: LimitRule,
}

impl Default for RiskLimitsConfig {
    fn default() -> Self {
        Self {
            max_position_size: LimitRule::hard("max_position_size", 100_000.0),
            max_portfolio_size: LimitRule::hard("max_portfolio_size", 500_000.0),
            max_drawdown: LimitRule::hard("max_drawdown", 0.10),
            max_daily_loss: LimitRule::hard("max_daily_loss", 0.02),
            max_gross_exposure: LimitRule::hard("max_gross_exposure", 3.0),
            max_net_exposure: LimitRule::soft("max_net_exposure", 1.0),
            max_open_positions: LimitRule::hard("max_open_positions", 20.0),
            max_concentration: LimitRule::soft("max_concentration", 0.25),
            max_single_trade_loss: LimitRule::hard("max_single_trade_loss", 0.01),
        }
    }
}

/// Severity of a limit check result
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// All clear — well within limits
    Ok,
    /// Approaching the warning threshold
    Warning,
    /// At or beyond the hard limit
    Breach,
}

/// Result of checking a single limit
#[derive(Debug, Clone)]
pub struct LimitCheckResult {
    /// Name of the limit that was checked
    pub limit_name: String,
    /// Current value being checked
    pub current_value: f64,
    /// The threshold for this limit
    pub threshold: f64,
    /// Warning threshold
    pub warning_threshold: f64,
    /// How close to the limit (0.0 = at zero, 1.0 = at threshold)
    pub utilization: f64,
    /// Severity of the result
    pub severity: Severity,
    /// Enforcement mode of the limit
    pub mode: LimitMode,
    /// Whether this check would block the proposed action
    pub blocks_action: bool,
}

/// Summary of all limit checks
#[derive(Debug, Clone)]
pub struct LimitCheckSummary {
    /// Individual limit check results
    pub checks: Vec<LimitCheckResult>,
    /// Overall severity (worst across all checks)
    pub worst_severity: Severity,
    /// Whether any hard limit was breached (action should be blocked)
    pub action_blocked: bool,
    /// Number of warnings
    pub warning_count: usize,
    /// Number of breaches
    pub breach_count: usize,
    /// The names of breached limits
    pub breached_limits: Vec<String>,
}

/// Current portfolio state snapshot, provided by the caller
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    /// Current equity
    pub equity: f64,
    /// Peak equity (for drawdown calculation)
    pub peak_equity: f64,
    /// Equity at start of day
    pub day_start_equity: f64,
    /// Current realized + unrealized PnL for the day
    pub daily_pnl: f64,
    /// Per-instrument position sizes (instrument_id → signed position size in base units)
    pub positions: HashMap<String, f64>,
    /// Per-instrument mark-to-market values (instrument_id → notional value)
    pub notional_values: HashMap<String, f64>,
}

impl PortfolioState {
    /// Total absolute (gross) position size across all instruments
    pub fn gross_position(&self) -> f64 {
        self.positions.values().map(|p| p.abs()).sum()
    }

    /// Net position (sum of signed positions — positive = net long)
    pub fn net_position(&self) -> f64 {
        self.positions.values().sum()
    }

    /// Gross exposure as a multiple of equity
    pub fn gross_exposure(&self) -> f64 {
        if self.equity <= 0.0 {
            return 0.0;
        }
        let gross_notional: f64 = self.notional_values.values().map(|v| v.abs()).sum();
        gross_notional / self.equity
    }

    /// Net exposure as a fraction of equity
    pub fn net_exposure(&self) -> f64 {
        if self.equity <= 0.0 {
            return 0.0;
        }
        let net_notional: f64 = self.notional_values.values().sum();
        net_notional.abs() / self.equity
    }

    /// Current drawdown from peak
    pub fn drawdown(&self) -> f64 {
        if self.peak_equity <= 0.0 {
            return 0.0;
        }
        ((self.peak_equity - self.equity) / self.peak_equity).max(0.0)
    }

    /// Daily loss as a fraction of start-of-day equity
    pub fn daily_loss_fraction(&self) -> f64 {
        if self.day_start_equity <= 0.0 {
            return 0.0;
        }
        // Positive daily_pnl = profit; we want the loss fraction (positive when losing)
        ((-self.daily_pnl) / self.day_start_equity).max(0.0)
    }

    /// Number of open positions
    pub fn open_position_count(&self) -> usize {
        self.positions.values().filter(|p| p.abs() > 1e-12).count()
    }

    /// Maximum single-instrument concentration (notional / equity)
    pub fn max_concentration(&self) -> f64 {
        if self.equity <= 0.0 {
            return 0.0;
        }
        self.notional_values
            .values()
            .map(|v| v.abs() / self.equity)
            .fold(0.0_f64, f64::max)
    }

    /// Get the largest absolute position size across all instruments
    pub fn max_position_size(&self) -> f64 {
        self.positions
            .values()
            .map(|p| p.abs())
            .fold(0.0_f64, f64::max)
    }
}

/// Proposed order for pre-trade limit checking
#[derive(Debug, Clone)]
pub struct ProposedOrder {
    /// Instrument identifier
    pub instrument_id: String,
    /// Signed quantity (positive = buy, negative = sell)
    pub quantity: f64,
    /// Estimated notional value of the order
    pub notional_value: f64,
    /// Estimated worst-case loss for this trade (used for single-trade-loss check)
    pub estimated_max_loss: f64,
}

/// Risk limits statistics
#[derive(Debug, Clone, Default)]
pub struct RiskLimitsStats {
    /// Total number of checks performed
    pub total_checks: u64,
    /// Number of times action was blocked
    pub blocks: u64,
    /// Number of warnings issued
    pub warnings: u64,
    /// Per-limit breach counts
    pub breach_counts: HashMap<String, u64>,
    /// Per-limit warning counts
    pub warning_counts: HashMap<String, u64>,
    /// Maximum drawdown observed
    pub max_drawdown_observed: f64,
    /// Maximum daily loss observed (fraction)
    pub max_daily_loss_observed: f64,
    /// Maximum gross exposure observed
    pub max_gross_exposure_observed: f64,
}

impl RiskLimitsStats {
    /// Block rate (fraction of checks that resulted in a block)
    pub fn block_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.blocks as f64 / self.total_checks as f64
    }

    /// Warning rate
    pub fn warning_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.warnings as f64 / self.total_checks as f64
    }
}

/// Risk limits enforcer
///
/// Checks proposed orders and portfolio state against configured risk limits.
/// Returns detailed check results indicating whether actions should be
/// allowed, warned about, or blocked.
///
/// Usage:
/// ```ignore
/// let limits = RiskLimits::new();
/// let state = PortfolioState { equity: 100_000.0, .. };
/// let order = ProposedOrder { instrument_id: "BTCUSD".into(), quantity: 10.0, .. };
///
/// let result = limits.check_order(&state, &order);
/// if result.action_blocked {
///     // reject the order
/// }
/// ```
pub struct RiskLimits {
    config: RiskLimitsConfig,
    stats: RiskLimitsStats,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskLimits {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(RiskLimitsConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: RiskLimitsConfig) -> Self {
        Self {
            config,
            stats: RiskLimitsStats::default(),
        }
    }

    /// Validate internal configuration
    pub fn process(&self) -> Result<()> {
        if self.config.max_drawdown.threshold < 0.0 || self.config.max_drawdown.threshold > 1.0 {
            return Err(Error::Configuration(
                "RiskLimits: max_drawdown threshold must be in [0, 1]".into(),
            ));
        }
        if self.config.max_daily_loss.threshold < 0.0 || self.config.max_daily_loss.threshold > 1.0
        {
            return Err(Error::Configuration(
                "RiskLimits: max_daily_loss threshold must be in [0, 1]".into(),
            ));
        }
        if self.config.max_gross_exposure.threshold < 0.0 {
            return Err(Error::Configuration(
                "RiskLimits: max_gross_exposure must be non-negative".into(),
            ));
        }
        Ok(())
    }

    /// Check the current portfolio state against all limits (no proposed order)
    pub fn check_portfolio(&mut self, state: &PortfolioState) -> LimitCheckSummary {
        let mut checks = Vec::new();

        // Drawdown check
        checks.push(self.check_single(&self.config.max_drawdown, state.drawdown()));

        // Daily loss check
        checks.push(self.check_single(&self.config.max_daily_loss, state.daily_loss_fraction()));

        // Gross exposure check
        checks.push(self.check_single(&self.config.max_gross_exposure, state.gross_exposure()));

        // Net exposure check
        checks.push(self.check_single(&self.config.max_net_exposure, state.net_exposure()));

        // Open positions count
        checks.push(self.check_single(
            &self.config.max_open_positions,
            state.open_position_count() as f64,
        ));

        // Portfolio size
        checks.push(self.check_single(&self.config.max_portfolio_size, state.gross_position()));

        // Concentration
        checks.push(self.check_single(&self.config.max_concentration, state.max_concentration()));

        // Position size (max single)
        checks.push(self.check_single(&self.config.max_position_size, state.max_position_size()));

        self.build_summary(checks, state)
    }

    /// Check a proposed order against all limits, given the current state
    ///
    /// This performs a "what-if" analysis: it simulates the portfolio state
    /// after the order would be filled and checks whether any limits would
    /// be breached.
    pub fn check_order(
        &mut self,
        state: &PortfolioState,
        order: &ProposedOrder,
    ) -> LimitCheckSummary {
        let mut checks = Vec::new();

        // Current checks (independent of order)
        checks.push(self.check_single(&self.config.max_drawdown, state.drawdown()));
        checks.push(self.check_single(&self.config.max_daily_loss, state.daily_loss_fraction()));

        // Post-trade position size for this instrument
        let current_pos = state
            .positions
            .get(&order.instrument_id)
            .copied()
            .unwrap_or(0.0);
        let post_trade_pos = (current_pos + order.quantity).abs();
        checks.push(self.check_single(&self.config.max_position_size, post_trade_pos));

        // Post-trade portfolio size
        let current_gross = state.gross_position();
        // Subtract the old position contribution, add the new
        let post_gross = current_gross - current_pos.abs() + post_trade_pos;
        checks.push(self.check_single(&self.config.max_portfolio_size, post_gross));

        // Post-trade gross exposure
        let current_gross_notional: f64 = state.notional_values.values().map(|v| v.abs()).sum();
        let current_notional = state
            .notional_values
            .get(&order.instrument_id)
            .copied()
            .unwrap_or(0.0);
        let post_notional = current_notional + order.notional_value;
        let post_gross_notional =
            current_gross_notional - current_notional.abs() + post_notional.abs();
        let post_gross_exp = if state.equity > 0.0 {
            post_gross_notional / state.equity
        } else {
            0.0
        };
        checks.push(self.check_single(&self.config.max_gross_exposure, post_gross_exp));

        // Post-trade net exposure
        let current_net_notional: f64 = state.notional_values.values().sum();
        let post_net_notional = current_net_notional - current_notional + post_notional;
        let post_net_exp = if state.equity > 0.0 {
            post_net_notional.abs() / state.equity
        } else {
            0.0
        };
        checks.push(self.check_single(&self.config.max_net_exposure, post_net_exp));

        // Post-trade open position count
        let mut post_count = state.open_position_count();
        let was_open = current_pos.abs() > 1e-12;
        let will_be_open = post_trade_pos > 1e-12;
        if !was_open && will_be_open {
            post_count += 1;
        } else if was_open && !will_be_open {
            post_count = post_count.saturating_sub(1);
        }
        checks.push(self.check_single(&self.config.max_open_positions, post_count as f64));

        // Post-trade concentration
        let post_concentration = if state.equity > 0.0 {
            post_notional.abs() / state.equity
        } else {
            0.0
        };
        checks.push(self.check_single(&self.config.max_concentration, post_concentration));

        // Single trade loss check
        let single_trade_loss = if state.equity > 0.0 {
            order.estimated_max_loss.abs() / state.equity
        } else {
            0.0
        };
        checks.push(self.check_single(&self.config.max_single_trade_loss, single_trade_loss));

        self.build_summary(checks, state)
    }

    /// Check whether adding a position of the given size would breach limits
    /// (simplified interface for quick gate checks)
    pub fn can_trade(
        &mut self,
        state: &PortfolioState,
        instrument_id: &str,
        quantity: f64,
        notional: f64,
    ) -> bool {
        let order = ProposedOrder {
            instrument_id: instrument_id.to_string(),
            quantity,
            notional_value: notional,
            estimated_max_loss: notional.abs() * 0.02, // Default 2% max loss estimate
        };
        let summary = self.check_order(state, &order);
        !summary.action_blocked
    }

    /// Get the current statistics
    pub fn stats(&self) -> &RiskLimitsStats {
        &self.stats
    }

    /// Get the current configuration
    pub fn config(&self) -> &RiskLimitsConfig {
        &self.config
    }

    /// Update configuration at runtime
    pub fn set_config(&mut self, config: RiskLimitsConfig) {
        self.config = config;
    }

    /// Update a specific limit threshold
    pub fn set_limit(&mut self, limit_name: &str, threshold: f64) -> Result<()> {
        match limit_name {
            "max_position_size" => self.config.max_position_size.threshold = threshold,
            "max_portfolio_size" => self.config.max_portfolio_size.threshold = threshold,
            "max_drawdown" => self.config.max_drawdown.threshold = threshold,
            "max_daily_loss" => self.config.max_daily_loss.threshold = threshold,
            "max_gross_exposure" => self.config.max_gross_exposure.threshold = threshold,
            "max_net_exposure" => self.config.max_net_exposure.threshold = threshold,
            "max_open_positions" => self.config.max_open_positions.threshold = threshold,
            "max_concentration" => self.config.max_concentration.threshold = threshold,
            "max_single_trade_loss" => self.config.max_single_trade_loss.threshold = threshold,
            other => {
                return Err(Error::Configuration(format!(
                    "Unknown limit name: {}",
                    other
                )));
            }
        }
        Ok(())
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = RiskLimitsStats::default();
    }

    /// Get the utilization of a specific limit (current value / threshold)
    pub fn utilization(&self, state: &PortfolioState, limit_name: &str) -> f64 {
        match limit_name {
            "max_drawdown" => state.drawdown() / self.config.max_drawdown.threshold,
            "max_daily_loss" => state.daily_loss_fraction() / self.config.max_daily_loss.threshold,
            "max_gross_exposure" => {
                state.gross_exposure() / self.config.max_gross_exposure.threshold
            }
            "max_net_exposure" => state.net_exposure() / self.config.max_net_exposure.threshold,
            "max_open_positions" => {
                state.open_position_count() as f64 / self.config.max_open_positions.threshold
            }
            "max_portfolio_size" => {
                state.gross_position() / self.config.max_portfolio_size.threshold
            }
            "max_concentration" => {
                state.max_concentration() / self.config.max_concentration.threshold
            }
            "max_position_size" => {
                state.max_position_size() / self.config.max_position_size.threshold
            }
            _ => 0.0,
        }
    }

    /// Get a summary of all limit utilizations
    pub fn all_utilizations(&self, state: &PortfolioState) -> Vec<(String, f64)> {
        vec![
            (
                "max_drawdown".into(),
                self.utilization(state, "max_drawdown"),
            ),
            (
                "max_daily_loss".into(),
                self.utilization(state, "max_daily_loss"),
            ),
            (
                "max_gross_exposure".into(),
                self.utilization(state, "max_gross_exposure"),
            ),
            (
                "max_net_exposure".into(),
                self.utilization(state, "max_net_exposure"),
            ),
            (
                "max_open_positions".into(),
                self.utilization(state, "max_open_positions"),
            ),
            (
                "max_portfolio_size".into(),
                self.utilization(state, "max_portfolio_size"),
            ),
            (
                "max_concentration".into(),
                self.utilization(state, "max_concentration"),
            ),
            (
                "max_position_size".into(),
                self.utilization(state, "max_position_size"),
            ),
        ]
    }

    // ── internal ──

    fn check_single(&self, rule: &LimitRule, current_value: f64) -> LimitCheckResult {
        if rule.mode == LimitMode::Disabled {
            return LimitCheckResult {
                limit_name: rule.label.clone(),
                current_value,
                threshold: rule.threshold,
                warning_threshold: rule.warning_threshold,
                utilization: 0.0,
                severity: Severity::Ok,
                mode: rule.mode,
                blocks_action: false,
            };
        }

        let utilization = if rule.threshold > 0.0 {
            current_value / rule.threshold
        } else {
            0.0
        };

        let severity = if current_value >= rule.threshold {
            Severity::Breach
        } else if current_value >= rule.warning_threshold {
            Severity::Warning
        } else {
            Severity::Ok
        };

        let blocks_action = severity == Severity::Breach && rule.mode == LimitMode::Hard;

        LimitCheckResult {
            limit_name: rule.label.clone(),
            current_value,
            threshold: rule.threshold,
            warning_threshold: rule.warning_threshold,
            utilization,
            severity,
            mode: rule.mode,
            blocks_action,
        }
    }

    fn build_summary(
        &mut self,
        checks: Vec<LimitCheckResult>,
        state: &PortfolioState,
    ) -> LimitCheckSummary {
        let worst_severity = checks
            .iter()
            .map(|c| c.severity)
            .max()
            .unwrap_or(Severity::Ok);

        let action_blocked = checks.iter().any(|c| c.blocks_action);

        let warning_count = checks
            .iter()
            .filter(|c| c.severity == Severity::Warning)
            .count();

        let breach_count = checks
            .iter()
            .filter(|c| c.severity == Severity::Breach)
            .count();

        let breached_limits: Vec<String> = checks
            .iter()
            .filter(|c| c.severity == Severity::Breach)
            .map(|c| c.limit_name.clone())
            .collect();

        // Update stats
        self.stats.total_checks += 1;
        if action_blocked {
            self.stats.blocks += 1;
        }
        if warning_count > 0 {
            self.stats.warnings += 1;
        }
        for name in &breached_limits {
            *self.stats.breach_counts.entry(name.clone()).or_insert(0) += 1;
        }
        for check in &checks {
            if check.severity == Severity::Warning {
                *self
                    .stats
                    .warning_counts
                    .entry(check.limit_name.clone())
                    .or_insert(0) += 1;
            }
        }

        // Track peak observations
        let dd = state.drawdown();
        if dd > self.stats.max_drawdown_observed {
            self.stats.max_drawdown_observed = dd;
        }
        let dl = state.daily_loss_fraction();
        if dl > self.stats.max_daily_loss_observed {
            self.stats.max_daily_loss_observed = dl;
        }
        let ge = state.gross_exposure();
        if ge > self.stats.max_gross_exposure_observed {
            self.stats.max_gross_exposure_observed = ge;
        }

        LimitCheckSummary {
            checks,
            worst_severity,
            action_blocked,
            warning_count,
            breach_count,
            breached_limits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// State that intentionally breaches some default limits (concentration, drawdown warning).
    /// Use for tests that expect breaches or don't care about severity.
    fn base_state() -> PortfolioState {
        let mut positions = HashMap::new();
        positions.insert("BTCUSD".into(), 5000.0);
        positions.insert("ETHUSD".into(), -3000.0);

        let mut notional = HashMap::new();
        notional.insert("BTCUSD".into(), 50_000.0);
        notional.insert("ETHUSD".into(), -30_000.0);

        PortfolioState {
            equity: 100_000.0,
            peak_equity: 110_000.0,
            day_start_equity: 102_000.0,
            daily_pnl: -1_000.0,
            positions,
            notional_values: notional,
        }
    }

    /// State that is well within ALL default limits — no warnings or breaches.
    fn clean_state() -> PortfolioState {
        let mut positions = HashMap::new();
        positions.insert("BTCUSD".into(), 1000.0);
        positions.insert("ETHUSD".into(), -800.0);

        let mut notional = HashMap::new();
        notional.insert("BTCUSD".into(), 10_000.0);
        notional.insert("ETHUSD".into(), -8_000.0);

        PortfolioState {
            equity: 100_000.0,
            peak_equity: 100_500.0, // tiny drawdown ~0.5%
            day_start_equity: 100_000.0,
            daily_pnl: 0.0, // no loss
            positions,
            notional_values: notional, // max concentration = 10%, gross exposure = 0.18
        }
    }

    #[test]
    fn test_basic() {
        let instance = RiskLimits::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_portfolio_state_calculations() {
        let state = base_state();

        assert!((state.gross_position() - 8000.0).abs() < 1e-12);
        assert!((state.net_position() - 2000.0).abs() < 1e-12);
        assert_eq!(state.open_position_count(), 2);

        // Gross exposure: (50000 + 30000) / 100000 = 0.8
        assert!((state.gross_exposure() - 0.8).abs() < 1e-12);

        // Net exposure: |50000 + (-30000)| / 100000 = 0.2
        assert!((state.net_exposure() - 0.2).abs() < 1e-12);

        // Drawdown: (110000 - 100000) / 110000 ≈ 0.0909
        let expected_dd = 10_000.0 / 110_000.0;
        assert!((state.drawdown() - expected_dd).abs() < 1e-6);

        // Daily loss: 1000 / 102000 ≈ 0.0098
        let expected_dl = 1_000.0 / 102_000.0;
        assert!((state.daily_loss_fraction() - expected_dl).abs() < 1e-6);

        // Concentration: max(50000, 30000) / 100000 = 0.5
        assert!((state.max_concentration() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_portfolio_check_no_breach() {
        let mut limits = RiskLimits::new();
        let state = clean_state();
        let summary = limits.check_portfolio(&state);

        assert!(!summary.action_blocked);
        assert_eq!(summary.worst_severity, Severity::Ok);
    }

    #[test]
    fn test_drawdown_breach() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::hard("max_drawdown", 0.05), // 5% max
            ..Default::default()
        });

        let state = base_state(); // ~9.09% drawdown
        let summary = limits.check_portfolio(&state);

        assert!(summary.action_blocked);
        assert!(
            summary
                .breached_limits
                .contains(&"max_drawdown".to_string())
        );
    }

    #[test]
    fn test_drawdown_warning() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::hard("max_drawdown", 0.12).with_warning(0.08), // Warning at 8%, breach at 12%
            // Use permissive limits for everything else so only drawdown triggers
            max_concentration: LimitRule::disabled("max_concentration"),
            ..Default::default()
        });

        let state = base_state(); // ~9.09% drawdown → warning
        let summary = limits.check_portfolio(&state);

        assert!(!summary.action_blocked);
        assert!(summary.warning_count > 0);
        assert_eq!(summary.worst_severity, Severity::Warning);
    }

    #[test]
    fn test_soft_limit_does_not_block() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::soft("max_drawdown", 0.05), // 5% soft limit
            ..Default::default()
        });

        let state = base_state(); // ~9.09% drawdown > 5% limit
        let summary = limits.check_portfolio(&state);

        // Soft limit is breached but should NOT block
        assert!(!summary.action_blocked);
        assert!(summary.breach_count > 0);
    }

    #[test]
    fn test_disabled_limit_ignored() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::disabled("max_drawdown"),
            ..Default::default()
        });

        // Even with huge drawdown, disabled limit should be Ok
        let mut state = base_state();
        state.peak_equity = 500_000.0; // Massive drawdown
        let summary = limits.check_portfolio(&state);

        let dd_check = summary
            .checks
            .iter()
            .find(|c| c.limit_name == "max_drawdown");
        assert!(dd_check.is_some());
        assert_eq!(dd_check.unwrap().severity, Severity::Ok);
    }

    #[test]
    fn test_order_position_size_breach() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_position_size: LimitRule::hard("max_position_size", 10_000.0),
            ..Default::default()
        });

        let state = base_state(); // BTCUSD = 5000
        let order = ProposedOrder {
            instrument_id: "BTCUSD".into(),
            quantity: 6000.0, // Would make position 11000 > 10000
            notional_value: 60_000.0,
            estimated_max_loss: 600.0,
        };

        let summary = limits.check_order(&state, &order);
        assert!(summary.action_blocked);
        assert!(
            summary
                .breached_limits
                .contains(&"max_position_size".to_string())
        );
    }

    #[test]
    fn test_order_within_limits() {
        let mut limits = RiskLimits::new();
        let state = clean_state();
        let order = ProposedOrder {
            instrument_id: "BTCUSD".into(),
            quantity: 100.0,
            notional_value: 1_000.0,
            estimated_max_loss: 20.0,
        };

        let summary = limits.check_order(&state, &order);
        assert!(!summary.action_blocked);
    }

    #[test]
    fn test_new_position_increments_count() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_open_positions: LimitRule::hard("max_open_positions", 2.0),
            ..Default::default()
        });

        let state = base_state(); // 2 positions
        let order = ProposedOrder {
            instrument_id: "SOLUSD".into(), // New instrument
            quantity: 100.0,
            notional_value: 1_000.0,
            estimated_max_loss: 20.0,
        };

        let summary = limits.check_order(&state, &order);
        // Post-trade would be 3 positions > 2 limit
        assert!(summary.action_blocked);
    }

    #[test]
    fn test_closing_position_decrements_count() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_open_positions: LimitRule::hard("max_open_positions", 2.0),
            ..Default::default()
        });

        let state = base_state(); // 2 positions: BTCUSD=5000, ETHUSD=-3000
        let order = ProposedOrder {
            instrument_id: "ETHUSD".into(),
            quantity: 3000.0, // Close the short
            notional_value: 30_000.0,
            estimated_max_loss: 600.0,
        };

        let summary = limits.check_order(&state, &order);
        // Post-trade would be 1 position ≤ 2 limit
        assert!(!summary.action_blocked);
    }

    #[test]
    fn test_daily_loss_breach() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_daily_loss: LimitRule::hard("max_daily_loss", 0.005), // 0.5%
            ..Default::default()
        });

        let state = base_state(); // daily loss ≈ 0.98% > 0.5%
        let summary = limits.check_portfolio(&state);

        assert!(summary.action_blocked);
        assert!(
            summary
                .breached_limits
                .contains(&"max_daily_loss".to_string())
        );
    }

    #[test]
    fn test_gross_exposure_breach() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_gross_exposure: LimitRule::hard("max_gross_exposure", 0.5),
            ..Default::default()
        });

        let state = base_state(); // gross exposure = 0.8 > 0.5
        let summary = limits.check_portfolio(&state);

        assert!(summary.action_blocked);
    }

    #[test]
    fn test_concentration_breach() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_concentration: LimitRule::hard("max_concentration", 0.40),
            ..Default::default()
        });

        let state = base_state(); // BTCUSD concentration = 50000/100000 = 0.5 > 0.4
        let summary = limits.check_portfolio(&state);

        assert!(summary.action_blocked);
    }

    #[test]
    fn test_single_trade_loss_breach() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_single_trade_loss: LimitRule::hard("max_single_trade_loss", 0.005), // 0.5%
            ..Default::default()
        });

        let state = base_state();
        let order = ProposedOrder {
            instrument_id: "BTCUSD".into(),
            quantity: 100.0,
            notional_value: 10_000.0,
            estimated_max_loss: 1_000.0, // 1% of equity > 0.5%
        };

        let summary = limits.check_order(&state, &order);
        assert!(summary.action_blocked);
    }

    #[test]
    fn test_can_trade_shortcut() {
        let mut limits = RiskLimits::new();
        let state = clean_state();

        assert!(limits.can_trade(&state, "BTCUSD", 100.0, 1_000.0));
    }

    #[test]
    fn test_can_trade_blocked() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_position_size: LimitRule::hard("max_position_size", 1.0),
            ..Default::default()
        });
        let state = base_state();

        assert!(!limits.can_trade(&state, "BTCUSD", 100.0, 1_000.0));
    }

    #[test]
    fn test_stats_tracking() {
        let mut limits = RiskLimits::new();
        let state = base_state();

        for _ in 0..5 {
            limits.check_portfolio(&state);
        }

        assert_eq!(limits.stats().total_checks, 5);
    }

    #[test]
    fn test_stats_block_counting() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::hard("max_drawdown", 0.01),
            ..Default::default()
        });
        let state = base_state();

        limits.check_portfolio(&state);
        limits.check_portfolio(&state);

        assert_eq!(limits.stats().blocks, 2);
        assert!(limits.stats().block_rate() > 0.0);
    }

    #[test]
    fn test_set_limit() {
        let mut limits = RiskLimits::new();
        assert!(limits.set_limit("max_drawdown", 0.15).is_ok());
        assert!((limits.config().max_drawdown.threshold - 0.15).abs() < 1e-12);
    }

    #[test]
    fn test_set_limit_unknown() {
        let mut limits = RiskLimits::new();
        assert!(limits.set_limit("nonexistent", 1.0).is_err());
    }

    #[test]
    fn test_utilization() {
        let limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::hard("max_drawdown", 0.20),
            ..Default::default()
        });
        let state = base_state(); // ~9.09% drawdown

        let util = limits.utilization(&state, "max_drawdown");
        let expected = state.drawdown() / 0.20;
        assert!((util - expected).abs() < 1e-6);
    }

    #[test]
    fn test_all_utilizations() {
        let limits = RiskLimits::new();
        let state = base_state();

        let utils = limits.all_utilizations(&state);
        assert_eq!(utils.len(), 8);
        for (name, val) in &utils {
            assert!(!name.is_empty());
            assert!(*val >= 0.0, "{} had negative utilization: {}", name, val);
        }
    }

    #[test]
    fn test_reset_stats() {
        let mut limits = RiskLimits::new();
        let state = base_state();

        limits.check_portfolio(&state);
        assert!(limits.stats().total_checks > 0);

        limits.reset_stats();
        assert_eq!(limits.stats().total_checks, 0);
    }

    #[test]
    fn test_multiple_breaches() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::hard("max_drawdown", 0.05),
            max_daily_loss: LimitRule::hard("max_daily_loss", 0.005),
            max_gross_exposure: LimitRule::hard("max_gross_exposure", 0.5),
            ..Default::default()
        });

        let state = base_state();
        let summary = limits.check_portfolio(&state);

        // Should have multiple breaches
        assert!(summary.breach_count >= 3);
        assert!(summary.action_blocked);
    }

    #[test]
    fn test_zero_equity_state() {
        let mut limits = RiskLimits::new();
        let state = PortfolioState {
            equity: 0.0,
            ..Default::default()
        };

        let summary = limits.check_portfolio(&state);
        // Should not panic, should handle gracefully
        assert_eq!(summary.worst_severity, Severity::Ok);
    }

    #[test]
    fn test_empty_portfolio() {
        let mut limits = RiskLimits::new();
        let state = PortfolioState {
            equity: 100_000.0,
            peak_equity: 100_000.0,
            day_start_equity: 100_000.0,
            daily_pnl: 0.0,
            positions: HashMap::new(),
            notional_values: HashMap::new(),
        };

        let summary = limits.check_portfolio(&state);
        assert!(!summary.action_blocked);
        assert_eq!(summary.worst_severity, Severity::Ok);
    }

    #[test]
    fn test_config_validation_bad_drawdown() {
        let limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::hard("max_drawdown", 1.5), // > 1.0
            ..Default::default()
        });
        assert!(limits.process().is_err());
    }

    #[test]
    fn test_config_validation_bad_daily_loss() {
        let limits = RiskLimits::with_config(RiskLimitsConfig {
            max_daily_loss: LimitRule::hard("max_daily_loss", -0.1),
            ..Default::default()
        });
        assert!(limits.process().is_err());
    }

    #[test]
    fn test_config_validation_bad_gross_exposure() {
        let limits = RiskLimits::with_config(RiskLimitsConfig {
            max_gross_exposure: LimitRule::hard("max_gross_exposure", -1.0),
            ..Default::default()
        });
        assert!(limits.process().is_err());
    }

    #[test]
    fn test_breach_count_in_stats() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_drawdown: LimitRule::hard("max_drawdown", 0.05),
            ..Default::default()
        });

        let state = base_state();
        limits.check_portfolio(&state);
        limits.check_portfolio(&state);

        assert_eq!(
            *limits
                .stats()
                .breach_counts
                .get("max_drawdown")
                .unwrap_or(&0),
            2
        );
    }

    #[test]
    fn test_warning_threshold_customization() {
        let rule = LimitRule::hard("test", 100.0).with_warning(90.0);
        assert!((rule.warning_threshold - 90.0).abs() < 1e-12);
        assert!((rule.threshold - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_limit_rule_defaults() {
        let hard = LimitRule::hard("test_hard", 50.0);
        assert_eq!(hard.mode, LimitMode::Hard);
        assert!((hard.warning_threshold - 40.0).abs() < 1e-12); // 80% of 50

        let soft = LimitRule::soft("test_soft", 50.0);
        assert_eq!(soft.mode, LimitMode::Soft);

        let disabled = LimitRule::disabled("test_disabled");
        assert_eq!(disabled.mode, LimitMode::Disabled);
    }

    #[test]
    fn test_partial_fill_rate_and_adverse_rate_default_zero() {
        let stats = RiskLimitsStats::default();
        assert!((stats.block_rate()).abs() < 1e-12);
        assert!((stats.warning_rate()).abs() < 1e-12);
    }

    #[test]
    fn test_order_reducing_position_allowed() {
        let mut limits = RiskLimits::with_config(RiskLimitsConfig {
            max_position_size: LimitRule::hard("max_position_size", 6000.0),
            ..Default::default()
        });

        let state = base_state(); // BTCUSD = 5000
        // Reducing position should be allowed even though current position is near limit
        let order = ProposedOrder {
            instrument_id: "BTCUSD".into(),
            quantity: -2000.0, // Reduce from 5000 to 3000
            notional_value: -20_000.0,
            estimated_max_loss: 200.0,
        };

        let summary = limits.check_order(&state, &order);
        // Position would go from 5000 to 3000, which is within 6000 limit
        let pos_check = summary
            .checks
            .iter()
            .find(|c| c.limit_name == "max_position_size");
        assert!(pos_check.is_some());
        assert_eq!(pos_check.unwrap().severity, Severity::Ok);
    }

    #[test]
    fn test_max_observations_tracked() {
        let mut limits = RiskLimits::new();

        // State with increasing drawdown
        let mut state = base_state();
        state.peak_equity = 200_000.0; // large drawdown
        limits.check_portfolio(&state);

        assert!(limits.stats().max_drawdown_observed > 0.0);
    }
}
