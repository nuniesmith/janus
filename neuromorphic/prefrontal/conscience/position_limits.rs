//! Position size limits enforcement
//!
//! Part of the Prefrontal region (conscience/compliance)
//! Component: conscience
//!
//! Implements multi-level position limit enforcement including symbol-level,
//! sector-level, and portfolio-wide limits. Supports dynamic limit adjustment
//! based on volatility, correlation, and risk metrics.

use crate::common::Result;
use std::collections::HashMap;
use tracing::{debug, info};

/// Limit type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitType {
    /// Hard limit - never exceeded
    Hard,
    /// Soft limit - warning only
    Soft,
    /// Dynamic limit - adjusted based on conditions
    Dynamic,
}

/// Limit violation severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    /// No violation
    None,
    /// Approaching limit (warning zone)
    Warning,
    /// At or exceeding soft limit
    SoftBreach,
    /// At or exceeding hard limit
    HardBreach,
}

/// Symbol-level position limit configuration
#[derive(Debug, Clone)]
pub struct SymbolLimit {
    /// Symbol identifier
    pub symbol: String,
    /// Maximum notional value
    pub max_notional: f64,
    /// Maximum quantity/contracts
    pub max_quantity: f64,
    /// Maximum percentage of portfolio
    pub max_portfolio_pct: f64,
    /// Limit type
    pub limit_type: LimitType,
    /// Warning threshold (percentage of limit)
    pub warning_threshold: f64,
}

impl Default for SymbolLimit {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            max_notional: 100_000.0,
            max_quantity: f64::INFINITY,
            max_portfolio_pct: 0.20, // 20% max per symbol
            limit_type: LimitType::Hard,
            warning_threshold: 0.80,
        }
    }
}

/// Sector-level position limit configuration
#[derive(Debug, Clone)]
pub struct SectorLimit {
    /// Sector identifier
    pub sector: String,
    /// Symbols in this sector
    pub symbols: Vec<String>,
    /// Maximum notional value for sector
    pub max_notional: f64,
    /// Maximum percentage of portfolio
    pub max_portfolio_pct: f64,
    /// Maximum correlation-weighted exposure
    pub max_correlated_exposure: f64,
    /// Limit type
    pub limit_type: LimitType,
    /// Warning threshold
    pub warning_threshold: f64,
}

impl Default for SectorLimit {
    fn default() -> Self {
        Self {
            sector: String::new(),
            symbols: Vec::new(),
            max_notional: 500_000.0,
            max_portfolio_pct: 0.40, // 40% max per sector
            max_correlated_exposure: 600_000.0,
            limit_type: LimitType::Hard,
            warning_threshold: 0.80,
        }
    }
}

/// Portfolio-wide limit configuration
#[derive(Debug, Clone)]
pub struct PortfolioLimit {
    /// Maximum total notional exposure
    pub max_total_notional: f64,
    /// Maximum gross exposure (long + short)
    pub max_gross_exposure: f64,
    /// Maximum net exposure (long - short)
    pub max_net_exposure: f64,
    /// Maximum number of positions
    pub max_positions: usize,
    /// Maximum single-day addition
    pub max_daily_addition: f64,
    /// Maximum concentration (HHI-based)
    pub max_concentration: f64,
    /// Warning threshold
    pub warning_threshold: f64,
}

impl Default for PortfolioLimit {
    fn default() -> Self {
        Self {
            max_total_notional: 1_000_000.0,
            max_gross_exposure: 1_500_000.0,
            max_net_exposure: 500_000.0,
            max_positions: 50,
            max_daily_addition: 200_000.0,
            max_concentration: 0.25, // Max HHI
            warning_threshold: 0.80,
        }
    }
}

/// Current position state
#[derive(Debug, Clone)]
pub struct PositionState {
    /// Symbol
    pub symbol: String,
    /// Sector assignment
    pub sector: Option<String>,
    /// Current notional value
    pub notional: f64,
    /// Current quantity
    pub quantity: f64,
    /// Is long position
    pub is_long: bool,
    /// Today's addition to this position
    pub daily_addition: f64,
}

/// Limit check result
#[derive(Debug, Clone)]
pub struct LimitCheckResult {
    /// Overall pass/fail
    pub passed: bool,
    /// Highest severity violation
    pub max_severity: ViolationSeverity,
    /// Individual violations
    pub violations: Vec<LimitViolation>,
    /// Warnings (approaching limits)
    pub warnings: Vec<LimitWarning>,
    /// Maximum allowed for the proposed change
    pub max_allowed: Option<f64>,
}

impl LimitCheckResult {
    pub fn pass() -> Self {
        Self {
            passed: true,
            max_severity: ViolationSeverity::None,
            violations: Vec::new(),
            warnings: Vec::new(),
            max_allowed: None,
        }
    }

    pub fn fail(violation: LimitViolation) -> Self {
        let severity = violation.severity;
        Self {
            passed: false,
            max_severity: severity,
            violations: vec![violation],
            warnings: Vec::new(),
            max_allowed: None,
        }
    }
}

/// Limit violation details
#[derive(Debug, Clone)]
pub struct LimitViolation {
    /// Type of limit violated
    pub limit_category: LimitCategory,
    /// Severity of violation
    pub severity: ViolationSeverity,
    /// Limit value
    pub limit: f64,
    /// Current/proposed value
    pub value: f64,
    /// Utilization percentage
    pub utilization: f64,
    /// Human-readable message
    pub message: String,
}

/// Limit warning details
#[derive(Debug, Clone)]
pub struct LimitWarning {
    /// Type of limit approaching
    pub limit_category: LimitCategory,
    /// Limit value
    pub limit: f64,
    /// Current value
    pub value: f64,
    /// Utilization percentage
    pub utilization: f64,
    /// Human-readable message
    pub message: String,
}

/// Category of limit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitCategory {
    SymbolNotional,
    SymbolQuantity,
    SymbolPortfolioPct,
    SectorNotional,
    SectorPortfolioPct,
    SectorCorrelation,
    PortfolioTotalNotional,
    PortfolioGrossExposure,
    PortfolioNetExposure,
    PortfolioPositionCount,
    PortfolioDailyAddition,
    PortfolioConcentration,
}

impl std::fmt::Display for LimitCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LimitCategory::SymbolNotional => write!(f, "Symbol Notional"),
            LimitCategory::SymbolQuantity => write!(f, "Symbol Quantity"),
            LimitCategory::SymbolPortfolioPct => write!(f, "Symbol Portfolio %"),
            LimitCategory::SectorNotional => write!(f, "Sector Notional"),
            LimitCategory::SectorPortfolioPct => write!(f, "Sector Portfolio %"),
            LimitCategory::SectorCorrelation => write!(f, "Sector Correlation"),
            LimitCategory::PortfolioTotalNotional => write!(f, "Portfolio Total Notional"),
            LimitCategory::PortfolioGrossExposure => write!(f, "Portfolio Gross Exposure"),
            LimitCategory::PortfolioNetExposure => write!(f, "Portfolio Net Exposure"),
            LimitCategory::PortfolioPositionCount => write!(f, "Portfolio Position Count"),
            LimitCategory::PortfolioDailyAddition => write!(f, "Portfolio Daily Addition"),
            LimitCategory::PortfolioConcentration => write!(f, "Portfolio Concentration"),
        }
    }
}

/// Configuration for position limits
#[derive(Debug, Clone)]
pub struct PositionLimitsConfig {
    /// Default symbol limits (applied if no specific limit)
    pub default_symbol_limit: SymbolLimit,
    /// Symbol-specific limits
    pub symbol_limits: HashMap<String, SymbolLimit>,
    /// Sector limits
    pub sector_limits: HashMap<String, SectorLimit>,
    /// Portfolio-wide limits
    pub portfolio_limits: PortfolioLimit,
    /// Enable dynamic limit adjustment
    pub enable_dynamic_adjustment: bool,
    /// Volatility multiplier for dynamic limits
    pub volatility_multiplier: f64,
    /// Total portfolio capital for percentage calculations
    pub portfolio_capital: f64,
}

impl Default for PositionLimitsConfig {
    fn default() -> Self {
        Self {
            default_symbol_limit: SymbolLimit::default(),
            symbol_limits: HashMap::new(),
            sector_limits: HashMap::new(),
            portfolio_limits: PortfolioLimit::default(),
            enable_dynamic_adjustment: true,
            volatility_multiplier: 1.5,
            portfolio_capital: 1_000_000.0,
        }
    }
}

/// Position limits enforcement
pub struct PositionLimits {
    /// Configuration
    config: PositionLimitsConfig,
    /// Current positions
    positions: HashMap<String, PositionState>,
    /// Symbol to sector mapping
    symbol_sectors: HashMap<String, String>,
    /// Current volatility by symbol
    volatilities: HashMap<String, f64>,
    /// Today's total additions
    daily_additions: f64,
    /// Violation history count
    violation_count: u64,
    /// Warning count
    warning_count: u64,
}

impl Default for PositionLimits {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionLimits {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_config(PositionLimitsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PositionLimitsConfig) -> Self {
        // Build symbol-sector mapping
        let mut symbol_sectors = HashMap::new();
        for (sector, limit) in &config.sector_limits {
            for symbol in &limit.symbols {
                symbol_sectors.insert(symbol.clone(), sector.clone());
            }
        }

        Self {
            config,
            positions: HashMap::new(),
            symbol_sectors,
            volatilities: HashMap::new(),
            daily_additions: 0.0,
            violation_count: 0,
            warning_count: 0,
        }
    }

    /// Set portfolio capital for percentage calculations
    pub fn set_portfolio_capital(&mut self, capital: f64) {
        self.config.portfolio_capital = capital;
    }

    /// Update current position
    pub fn update_position(&mut self, position: PositionState) {
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Remove a position
    pub fn remove_position(&mut self, symbol: &str) {
        self.positions.remove(symbol);
    }

    /// Update volatility for a symbol
    pub fn update_volatility(&mut self, symbol: &str, volatility: f64) {
        self.volatilities.insert(symbol.to_string(), volatility);
    }

    /// Set symbol limit
    pub fn set_symbol_limit(&mut self, limit: SymbolLimit) {
        self.config
            .symbol_limits
            .insert(limit.symbol.clone(), limit);
    }

    /// Set sector limit
    pub fn set_sector_limit(&mut self, limit: SectorLimit) {
        // Update symbol-sector mapping
        for symbol in &limit.symbols {
            self.symbol_sectors
                .insert(symbol.clone(), limit.sector.clone());
        }
        self.config
            .sector_limits
            .insert(limit.sector.clone(), limit);
    }

    /// Reset daily additions (call at start of day)
    pub fn reset_daily(&mut self) {
        self.daily_additions = 0.0;
        for pos in self.positions.values_mut() {
            pos.daily_addition = 0.0;
        }
        info!("Daily position limits reset");
    }

    /// Check if a proposed position change is allowed
    pub fn check_position_change(
        &mut self,
        symbol: &str,
        proposed_notional: f64,
        proposed_quantity: f64,
        is_long: bool,
    ) -> LimitCheckResult {
        let mut result = LimitCheckResult::pass();

        // Get current position state
        let current = self.positions.get(symbol).cloned();
        let current_notional = current.as_ref().map(|p| p.notional).unwrap_or(0.0);
        let change_amount = proposed_notional - current_notional;

        // Check symbol-level limits
        self.check_symbol_limits(symbol, proposed_notional, proposed_quantity, &mut result);

        // Check sector-level limits
        if let Some(sector) = self.symbol_sectors.get(symbol).cloned() {
            self.check_sector_limits(&sector, symbol, change_amount, &mut result);
        }

        // Check portfolio-level limits
        self.check_portfolio_limits(symbol, change_amount, is_long, &mut result);

        // Update severity
        if !result.violations.is_empty() {
            result.passed = false;
            result.max_severity = result
                .violations
                .iter()
                .map(|v| v.severity)
                .max()
                .unwrap_or(ViolationSeverity::None);
            self.violation_count += 1;
        }

        if !result.warnings.is_empty() {
            self.warning_count += 1;
        }

        result
    }

    /// Check symbol-level limits
    fn check_symbol_limits(
        &self,
        symbol: &str,
        proposed_notional: f64,
        proposed_quantity: f64,
        result: &mut LimitCheckResult,
    ) {
        let limit = self.get_symbol_limit(symbol);
        let effective_limit = self.apply_volatility_adjustment(symbol, &limit);

        // Check notional limit
        if proposed_notional > effective_limit.max_notional {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::SymbolNotional,
                severity: match limit.limit_type {
                    LimitType::Hard => ViolationSeverity::HardBreach,
                    LimitType::Soft | LimitType::Dynamic => ViolationSeverity::SoftBreach,
                },
                limit: effective_limit.max_notional,
                value: proposed_notional,
                utilization: proposed_notional / effective_limit.max_notional,
                message: format!(
                    "Symbol {} notional ${:.0} exceeds limit ${:.0}",
                    symbol, proposed_notional, effective_limit.max_notional
                ),
            });
            result.max_allowed = Some(effective_limit.max_notional);
        } else if proposed_notional / effective_limit.max_notional
            > effective_limit.warning_threshold
        {
            result.warnings.push(LimitWarning {
                limit_category: LimitCategory::SymbolNotional,
                limit: effective_limit.max_notional,
                value: proposed_notional,
                utilization: proposed_notional / effective_limit.max_notional,
                message: format!(
                    "Symbol {} notional ${:.0} at {:.0}% of limit",
                    symbol,
                    proposed_notional,
                    (proposed_notional / effective_limit.max_notional) * 100.0
                ),
            });
        }

        // Check quantity limit
        if proposed_quantity > effective_limit.max_quantity {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::SymbolQuantity,
                severity: ViolationSeverity::HardBreach,
                limit: effective_limit.max_quantity,
                value: proposed_quantity,
                utilization: proposed_quantity / effective_limit.max_quantity,
                message: format!(
                    "Symbol {} quantity {:.2} exceeds limit {:.2}",
                    symbol, proposed_quantity, effective_limit.max_quantity
                ),
            });
        }

        // Check portfolio percentage limit
        if self.config.portfolio_capital > 0.0 {
            let portfolio_pct = proposed_notional / self.config.portfolio_capital;
            if portfolio_pct > effective_limit.max_portfolio_pct {
                result.violations.push(LimitViolation {
                    limit_category: LimitCategory::SymbolPortfolioPct,
                    severity: ViolationSeverity::HardBreach,
                    limit: effective_limit.max_portfolio_pct,
                    value: portfolio_pct,
                    utilization: portfolio_pct / effective_limit.max_portfolio_pct,
                    message: format!(
                        "Symbol {} at {:.1}% of portfolio exceeds {:.1}% limit",
                        symbol,
                        portfolio_pct * 100.0,
                        effective_limit.max_portfolio_pct * 100.0
                    ),
                });
            }
        }
    }

    /// Check sector-level limits
    fn check_sector_limits(
        &self,
        sector: &str,
        changing_symbol: &str,
        change_amount: f64,
        result: &mut LimitCheckResult,
    ) {
        let limit = match self.config.sector_limits.get(sector) {
            Some(l) => l,
            None => return,
        };

        // Calculate current sector exposure
        let mut sector_notional = 0.0;
        for symbol in &limit.symbols {
            if let Some(pos) = self.positions.get(symbol) {
                sector_notional += pos.notional;
            }
        }
        let proposed_sector = sector_notional + change_amount;

        // Check notional limit
        if proposed_sector > limit.max_notional {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::SectorNotional,
                severity: match limit.limit_type {
                    LimitType::Hard => ViolationSeverity::HardBreach,
                    _ => ViolationSeverity::SoftBreach,
                },
                limit: limit.max_notional,
                value: proposed_sector,
                utilization: proposed_sector / limit.max_notional,
                message: format!(
                    "Sector {} notional ${:.0} exceeds limit ${:.0}",
                    sector, proposed_sector, limit.max_notional
                ),
            });
        } else if proposed_sector / limit.max_notional > limit.warning_threshold {
            result.warnings.push(LimitWarning {
                limit_category: LimitCategory::SectorNotional,
                limit: limit.max_notional,
                value: proposed_sector,
                utilization: proposed_sector / limit.max_notional,
                message: format!(
                    "Sector {} notional ${:.0} at {:.0}% of limit",
                    sector,
                    proposed_sector,
                    (proposed_sector / limit.max_notional) * 100.0
                ),
            });
        }

        // Check portfolio percentage
        if self.config.portfolio_capital > 0.0 {
            let sector_pct = proposed_sector / self.config.portfolio_capital;
            if sector_pct > limit.max_portfolio_pct {
                result.violations.push(LimitViolation {
                    limit_category: LimitCategory::SectorPortfolioPct,
                    severity: ViolationSeverity::HardBreach,
                    limit: limit.max_portfolio_pct,
                    value: sector_pct,
                    utilization: sector_pct / limit.max_portfolio_pct,
                    message: format!(
                        "Sector {} at {:.1}% of portfolio exceeds {:.1}% limit",
                        sector,
                        sector_pct * 100.0,
                        limit.max_portfolio_pct * 100.0
                    ),
                });
            }
        }

        debug!(
            sector = sector,
            changing_symbol = changing_symbol,
            sector_notional = proposed_sector,
            "Checked sector limits"
        );
    }

    /// Check portfolio-level limits
    fn check_portfolio_limits(
        &self,
        changing_symbol: &str,
        change_amount: f64,
        is_long: bool,
        result: &mut LimitCheckResult,
    ) {
        let limits = &self.config.portfolio_limits;

        // Calculate current totals
        let mut total_long = 0.0;
        let mut total_short = 0.0;
        let mut position_count = self.positions.len();

        for pos in self.positions.values() {
            if pos.is_long {
                total_long += pos.notional;
            } else {
                total_short += pos.notional;
            }
        }

        // Adjust for change
        if !self.positions.contains_key(changing_symbol) && change_amount > 0.0 {
            position_count += 1;
        }
        if is_long {
            total_long += change_amount;
        } else {
            total_short += change_amount;
        }

        let gross_exposure = total_long + total_short;
        let net_exposure = (total_long - total_short).abs();
        let total_notional = gross_exposure;

        // Check total notional
        if total_notional > limits.max_total_notional {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::PortfolioTotalNotional,
                severity: ViolationSeverity::HardBreach,
                limit: limits.max_total_notional,
                value: total_notional,
                utilization: total_notional / limits.max_total_notional,
                message: format!(
                    "Portfolio notional ${:.0} exceeds limit ${:.0}",
                    total_notional, limits.max_total_notional
                ),
            });
        } else if total_notional / limits.max_total_notional > limits.warning_threshold {
            result.warnings.push(LimitWarning {
                limit_category: LimitCategory::PortfolioTotalNotional,
                limit: limits.max_total_notional,
                value: total_notional,
                utilization: total_notional / limits.max_total_notional,
                message: format!(
                    "Portfolio notional ${:.0} at {:.0}% of limit",
                    total_notional,
                    (total_notional / limits.max_total_notional) * 100.0
                ),
            });
        }

        // Check gross exposure
        if gross_exposure > limits.max_gross_exposure {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::PortfolioGrossExposure,
                severity: ViolationSeverity::HardBreach,
                limit: limits.max_gross_exposure,
                value: gross_exposure,
                utilization: gross_exposure / limits.max_gross_exposure,
                message: format!(
                    "Gross exposure ${:.0} exceeds limit ${:.0}",
                    gross_exposure, limits.max_gross_exposure
                ),
            });
        }

        // Check net exposure
        if net_exposure > limits.max_net_exposure {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::PortfolioNetExposure,
                severity: ViolationSeverity::SoftBreach,
                limit: limits.max_net_exposure,
                value: net_exposure,
                utilization: net_exposure / limits.max_net_exposure,
                message: format!(
                    "Net exposure ${:.0} exceeds limit ${:.0}",
                    net_exposure, limits.max_net_exposure
                ),
            });
        }

        // Check position count
        if position_count > limits.max_positions {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::PortfolioPositionCount,
                severity: ViolationSeverity::HardBreach,
                limit: limits.max_positions as f64,
                value: position_count as f64,
                utilization: position_count as f64 / limits.max_positions as f64,
                message: format!(
                    "Position count {} exceeds limit {}",
                    position_count, limits.max_positions
                ),
            });
        }

        // Check daily addition
        let proposed_daily = self.daily_additions + change_amount.max(0.0);
        if proposed_daily > limits.max_daily_addition {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::PortfolioDailyAddition,
                severity: ViolationSeverity::SoftBreach,
                limit: limits.max_daily_addition,
                value: proposed_daily,
                utilization: proposed_daily / limits.max_daily_addition,
                message: format!(
                    "Daily additions ${:.0} exceeds limit ${:.0}",
                    proposed_daily, limits.max_daily_addition
                ),
            });
        }

        // Check concentration (HHI)
        let concentration =
            self.calculate_hhi_after_change(changing_symbol, change_amount, is_long);
        if concentration > limits.max_concentration {
            result.violations.push(LimitViolation {
                limit_category: LimitCategory::PortfolioConcentration,
                severity: ViolationSeverity::SoftBreach,
                limit: limits.max_concentration,
                value: concentration,
                utilization: concentration / limits.max_concentration,
                message: format!(
                    "Portfolio concentration {:.2} exceeds limit {:.2}",
                    concentration, limits.max_concentration
                ),
            });
        }
    }

    /// Get symbol limit (specific or default)
    fn get_symbol_limit(&self, symbol: &str) -> SymbolLimit {
        self.config
            .symbol_limits
            .get(symbol)
            .cloned()
            .unwrap_or_else(|| {
                let mut default = self.config.default_symbol_limit.clone();
                default.symbol = symbol.to_string();
                default
            })
    }

    /// Apply volatility adjustment to limits
    fn apply_volatility_adjustment(&self, symbol: &str, limit: &SymbolLimit) -> SymbolLimit {
        if !self.config.enable_dynamic_adjustment || limit.limit_type != LimitType::Dynamic {
            return limit.clone();
        }

        let vol = self.volatilities.get(symbol).copied().unwrap_or(0.2);
        let base_vol = 0.2; // 20% annualized as base
        let vol_ratio = base_vol / vol.max(0.01);
        let adjustment = vol_ratio.clamp(0.5, 2.0);

        let mut adjusted = limit.clone();
        adjusted.max_notional *= adjustment;
        adjusted.max_portfolio_pct *= adjustment;
        adjusted
    }

    /// Calculate HHI (Herfindahl-Hirschman Index) after proposed change
    fn calculate_hhi_after_change(&self, symbol: &str, change_amount: f64, _is_long: bool) -> f64 {
        let mut total = 0.0;
        let mut position_values: Vec<f64> = Vec::new();

        for (sym, pos) in &self.positions {
            let value = if sym == symbol {
                pos.notional + change_amount
            } else {
                pos.notional
            };
            if value > 0.0 {
                position_values.push(value);
                total += value;
            }
        }

        // Add new position if not existing
        if !self.positions.contains_key(symbol) && change_amount > 0.0 {
            position_values.push(change_amount);
            total += change_amount;
        }

        if total <= 0.0 {
            return 0.0;
        }

        // Calculate HHI as sum of squared weights
        position_values.iter().map(|v| (v / total).powi(2)).sum()
    }

    /// Get current utilization summary
    pub fn utilization_summary(&self) -> UtilizationSummary {
        let mut total_long = 0.0;
        let mut total_short = 0.0;
        let mut sector_exposures: HashMap<String, f64> = HashMap::new();

        for pos in self.positions.values() {
            if pos.is_long {
                total_long += pos.notional;
            } else {
                total_short += pos.notional;
            }

            if let Some(sector) = &pos.sector {
                *sector_exposures.entry(sector.clone()).or_insert(0.0) += pos.notional;
            }
        }

        let gross = total_long + total_short;
        let net = (total_long - total_short).abs();

        UtilizationSummary {
            total_long,
            total_short,
            gross_exposure: gross,
            net_exposure: net,
            position_count: self.positions.len(),
            daily_additions: self.daily_additions,
            gross_utilization: gross / self.config.portfolio_limits.max_gross_exposure,
            net_utilization: net / self.config.portfolio_limits.max_net_exposure,
            position_utilization: self.positions.len() as f64
                / self.config.portfolio_limits.max_positions as f64,
            concentration_hhi: self.calculate_current_hhi(),
            sector_exposures,
        }
    }

    /// Calculate current HHI
    fn calculate_current_hhi(&self) -> f64 {
        let total: f64 = self.positions.values().map(|p| p.notional).sum();
        if total <= 0.0 {
            return 0.0;
        }

        self.positions
            .values()
            .map(|p| (p.notional / total).powi(2))
            .sum()
    }

    /// Get statistics
    pub fn stats(&self) -> PositionLimitsStats {
        let summary = self.utilization_summary();
        PositionLimitsStats {
            position_count: self.positions.len(),
            total_exposure: summary.gross_exposure,
            gross_utilization_pct: summary.gross_utilization * 100.0,
            net_utilization_pct: summary.net_utilization * 100.0,
            concentration_hhi: summary.concentration_hhi,
            violation_count: self.violation_count,
            warning_count: self.warning_count,
            daily_additions: self.daily_additions,
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        debug!(
            positions = self.positions.len(),
            violations = self.violation_count,
            warnings = self.warning_count,
            "Position limits process called"
        );
        Ok(())
    }
}

/// Utilization summary
#[derive(Debug, Clone)]
pub struct UtilizationSummary {
    pub total_long: f64,
    pub total_short: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub position_count: usize,
    pub daily_additions: f64,
    pub gross_utilization: f64,
    pub net_utilization: f64,
    pub position_utilization: f64,
    pub concentration_hhi: f64,
    pub sector_exposures: HashMap<String, f64>,
}

/// Position limits statistics
#[derive(Debug, Clone)]
pub struct PositionLimitsStats {
    pub position_count: usize,
    pub total_exposure: f64,
    pub gross_utilization_pct: f64,
    pub net_utilization_pct: f64,
    pub concentration_hhi: f64,
    pub violation_count: u64,
    pub warning_count: u64,
    pub daily_additions: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_limits() -> PositionLimits {
        let mut config = PositionLimitsConfig::default();
        config.portfolio_capital = 1_000_000.0;
        config.portfolio_limits.max_total_notional = 800_000.0;
        config.portfolio_limits.max_gross_exposure = 1_000_000.0;
        config.portfolio_limits.max_net_exposure = 400_000.0;
        config.portfolio_limits.max_positions = 10;
        // Allow high concentration for single-position tests
        config.portfolio_limits.max_concentration = 1.0;

        let mut limits = PositionLimits::with_config(config);

        // Add a sector
        limits.set_sector_limit(SectorLimit {
            sector: "crypto".to_string(),
            symbols: vec!["BTC".to_string(), "ETH".to_string()],
            max_notional: 300_000.0,
            max_portfolio_pct: 0.35,
            ..Default::default()
        });

        limits
    }

    #[test]
    fn test_basic() {
        let instance = PositionLimits::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_symbol_limit_pass() {
        let mut limits = create_test_limits();

        let result = limits.check_position_change("AAPL", 50_000.0, 100.0, true);
        if !result.passed {
            eprintln!("Violations: {:?}", result.violations);
            eprintln!("Warnings: {:?}", result.warnings);
            eprintln!("Max severity: {:?}", result.max_severity);
        }
        assert!(
            result.passed,
            "Expected pass but got violations: {:?}",
            result.violations
        );
        assert_eq!(result.max_severity, ViolationSeverity::None);
    }

    #[test]
    fn test_symbol_limit_fail() {
        let mut limits = create_test_limits();
        limits.set_symbol_limit(SymbolLimit {
            symbol: "AAPL".to_string(),
            max_notional: 30_000.0,
            max_quantity: 50.0,
            max_portfolio_pct: 0.05,
            limit_type: LimitType::Hard,
            warning_threshold: 0.80,
        });

        let result = limits.check_position_change("AAPL", 50_000.0, 100.0, true);
        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.limit_category == LimitCategory::SymbolNotional)
        );
    }

    #[test]
    fn test_sector_limit() {
        let mut limits = create_test_limits();

        // Add positions to crypto sector
        limits.update_position(PositionState {
            symbol: "BTC".to_string(),
            sector: Some("crypto".to_string()),
            notional: 200_000.0,
            quantity: 4.0,
            is_long: true,
            daily_addition: 200_000.0,
        });

        // Try to add more - should fail sector limit
        let result = limits.check_position_change("ETH", 150_000.0, 50.0, true);
        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.limit_category == LimitCategory::SectorNotional)
        );
    }

    #[test]
    fn test_portfolio_position_count() {
        let mut limits = create_test_limits();

        // Add max positions
        for i in 0..10 {
            limits.update_position(PositionState {
                symbol: format!("SYM{}", i),
                sector: None,
                notional: 10_000.0,
                quantity: 100.0,
                is_long: true,
                daily_addition: 0.0,
            });
        }

        // Try to add one more
        let result = limits.check_position_change("NEW", 10_000.0, 100.0, true);
        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.limit_category == LimitCategory::PortfolioPositionCount)
        );
    }

    #[test]
    fn test_portfolio_gross_exposure() {
        let mut limits = create_test_limits();

        limits.update_position(PositionState {
            symbol: "LONG1".to_string(),
            sector: None,
            notional: 500_000.0,
            quantity: 1000.0,
            is_long: true,
            daily_addition: 0.0,
        });

        // Adding 600k would exceed 1M gross limit
        let result = limits.check_position_change("LONG2", 600_000.0, 1000.0, true);
        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.limit_category == LimitCategory::PortfolioGrossExposure)
        );
    }

    #[test]
    fn test_warning_threshold() {
        let mut limits = create_test_limits();
        limits.set_symbol_limit(SymbolLimit {
            symbol: "AAPL".to_string(),
            max_notional: 100_000.0,
            max_quantity: f64::INFINITY,
            max_portfolio_pct: 0.20,
            limit_type: LimitType::Hard,
            warning_threshold: 0.80,
        });

        // 85% of limit should trigger warning
        let result = limits.check_position_change("AAPL", 85_000.0, 100.0, true);
        assert!(result.passed);
        assert!(!result.warnings.is_empty());
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.limit_category == LimitCategory::SymbolNotional)
        );
    }

    #[test]
    fn test_net_exposure() {
        let mut limits = create_test_limits();

        // Increase default symbol limit to allow larger positions for this test
        limits.config.default_symbol_limit.max_notional = 500_000.0;

        // Add a large long position
        limits.update_position(PositionState {
            symbol: "LONG".to_string(),
            sector: None,
            notional: 300_000.0,
            quantity: 1000.0,
            is_long: true,
            daily_addition: 0.0,
        });

        // Large short position should be ok (reduces net)
        let result = limits.check_position_change("SHORT", 200_000.0, 1000.0, false);
        if !result.passed {
            eprintln!("Violations: {:?}", result.violations);
            eprintln!("Warnings: {:?}", result.warnings);
        }
        assert!(
            result.passed,
            "Expected pass but got violations: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_concentration_hhi() {
        let mut limits = create_test_limits();
        limits.config.portfolio_limits.max_concentration = 0.30;

        // Single large position = high concentration
        let result = limits.check_position_change("SINGLE", 500_000.0, 1000.0, true);
        // Single position has HHI of 1.0, which exceeds 0.30
        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.limit_category == LimitCategory::PortfolioConcentration)
        );
    }

    #[test]
    fn test_utilization_summary() {
        let mut limits = create_test_limits();

        limits.update_position(PositionState {
            symbol: "BTC".to_string(),
            sector: Some("crypto".to_string()),
            notional: 100_000.0,
            quantity: 2.0,
            is_long: true,
            daily_addition: 50_000.0,
        });

        limits.update_position(PositionState {
            symbol: "AAPL".to_string(),
            sector: Some("tech".to_string()),
            notional: 50_000.0,
            quantity: 300.0,
            is_long: true,
            daily_addition: 25_000.0,
        });

        let summary = limits.utilization_summary();
        assert_eq!(summary.position_count, 2);
        assert!((summary.total_long - 150_000.0).abs() < 0.01);
        assert!((summary.gross_exposure - 150_000.0).abs() < 0.01);
    }

    #[test]
    fn test_daily_reset() {
        let mut limits = create_test_limits();

        limits.update_position(PositionState {
            symbol: "TEST".to_string(),
            sector: None,
            notional: 50_000.0,
            quantity: 100.0,
            is_long: true,
            daily_addition: 50_000.0,
        });
        limits.daily_additions = 50_000.0;

        limits.reset_daily();

        assert_eq!(limits.daily_additions, 0.0);
        assert_eq!(limits.positions.get("TEST").unwrap().daily_addition, 0.0);
    }

    #[test]
    fn test_volatility_adjustment() {
        let mut limits = PositionLimits::with_config(PositionLimitsConfig {
            enable_dynamic_adjustment: true,
            volatility_multiplier: 1.5,
            ..Default::default()
        });

        limits.set_symbol_limit(SymbolLimit {
            symbol: "VOLATILE".to_string(),
            max_notional: 100_000.0,
            limit_type: LimitType::Dynamic,
            ..Default::default()
        });

        // High volatility should reduce limits
        limits.update_volatility("VOLATILE", 0.5); // 50% vol vs 20% base

        let limit = limits.get_symbol_limit("VOLATILE");
        let adjusted = limits.apply_volatility_adjustment("VOLATILE", &limit);

        // 0.2 / 0.5 = 0.4 adjustment factor
        assert!(adjusted.max_notional < limit.max_notional);
    }

    #[test]
    fn test_remove_position() {
        let mut limits = create_test_limits();

        limits.update_position(PositionState {
            symbol: "TEST".to_string(),
            sector: None,
            notional: 50_000.0,
            quantity: 100.0,
            is_long: true,
            daily_addition: 0.0,
        });

        assert_eq!(limits.positions.len(), 1);

        limits.remove_position("TEST");
        assert_eq!(limits.positions.len(), 0);
    }

    #[test]
    fn test_stats() {
        let mut limits = create_test_limits();

        limits.update_position(PositionState {
            symbol: "TEST".to_string(),
            sector: None,
            notional: 100_000.0,
            quantity: 100.0,
            is_long: true,
            daily_addition: 0.0,
        });

        // Trigger a violation
        let _ = limits.check_position_change("BIG", 2_000_000.0, 1000.0, true);

        let stats = limits.stats();
        assert_eq!(stats.position_count, 1);
        assert!(stats.violation_count > 0);
    }
}
