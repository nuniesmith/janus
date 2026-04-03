//! Prop Firm Rules — Trading rule compliance enforcement
//!
//! Part of the Prefrontal region
//! Component: conscience
//!
//! Enforces prop firm trading constraints that are common across funded
//! trader programs (FTMO, TopStep, The5ers, etc.). These rules run as
//! pre-trade and post-trade checks to prevent violations that would
//! cause account termination.
//!
//! Dimensions enforced:
//! - **Daily loss limit**: Maximum loss allowed in a single trading day
//! - **Maximum drawdown**: Trailing or static max drawdown from peak equity
//! - **Profit target**: Minimum profit required to pass evaluation phases
//! - **Trading hours**: Restrict trading to allowed market hours
//! - **Minimum trading days**: Require activity across N distinct days
//! - **Consistency rule**: No single day's profit exceeds X% of total
//! - **News restriction**: Block trading around high-impact news events
//! - **Weekend/overnight holding**: Restrict positions over weekends
//! - **Maximum position size**: Per-instrument lot/contract limits
//! - **Maximum concurrent positions**: Limit open positions simultaneously
//!
//! Features:
//! - Three enforcement modes per rule: Hard (reject), Soft (warn), Disabled
//! - Pre-trade "can I place this order?" gate check
//! - Post-trade "did this violate anything?" audit check
//! - Warning thresholds (configurable, default 80% of limit)
//! - Daily reset logic for daily-scoped rules
//! - Running statistics and violation history
//! - Windowed diagnostics

use crate::common::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Enforcement mode for a single rule
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleMode {
    /// Hard rule — violations cause order rejection
    Hard,
    /// Soft rule — violations produce warnings but allow the action
    Soft,
    /// Disabled — rule is not checked
    Disabled,
}

impl Default for RuleMode {
    fn default() -> Self {
        RuleMode::Hard
    }
}

/// Drawdown calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawdownType {
    /// Trailing drawdown from the highest equity ever reached
    Trailing,
    /// Static drawdown from the initial account balance
    Static,
    /// End-of-day drawdown (only computed at daily close)
    EndOfDay,
}

impl Default for DrawdownType {
    fn default() -> Self {
        DrawdownType::Trailing
    }
}

/// A single rule definition with its threshold and mode
#[derive(Debug, Clone)]
pub struct RuleDefinition {
    /// The limit value (interpretation depends on the rule)
    pub limit: f64,
    /// Warning threshold as a fraction of the limit (e.g. 0.8 = warn at 80%)
    pub warning_threshold: f64,
    /// Enforcement mode
    pub mode: RuleMode,
    /// Human-readable label
    pub label: String,
}

impl RuleDefinition {
    pub fn hard(limit: f64, label: &str) -> Self {
        Self {
            limit,
            warning_threshold: 0.8,
            mode: RuleMode::Hard,
            label: label.to_string(),
        }
    }

    pub fn soft(limit: f64, label: &str) -> Self {
        Self {
            limit,
            warning_threshold: 0.8,
            mode: RuleMode::Soft,
            label: label.to_string(),
        }
    }

    pub fn disabled() -> Self {
        Self {
            limit: 0.0,
            warning_threshold: 0.8,
            mode: RuleMode::Disabled,
            label: String::new(),
        }
    }

    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// The absolute value at which a warning is triggered
    pub fn warning_value(&self) -> f64 {
        self.limit * self.warning_threshold
    }

    /// Whether this rule is active (not disabled)
    pub fn is_active(&self) -> bool {
        self.mode != RuleMode::Disabled
    }
}

/// Hour range (24-hour format, UTC)
#[derive(Debug, Clone)]
pub struct TradingHours {
    /// Start hour (inclusive), 0-23
    pub start_hour: u8,
    /// Start minute, 0-59
    pub start_minute: u8,
    /// End hour (exclusive), 0-23
    pub end_hour: u8,
    /// End minute, 0-59
    pub end_minute: u8,
}

impl TradingHours {
    pub fn new(start_hour: u8, start_minute: u8, end_hour: u8, end_minute: u8) -> Self {
        Self {
            start_hour: start_hour.min(23),
            start_minute: start_minute.min(59),
            end_hour: end_hour.min(23),
            end_minute: end_minute.min(59),
        }
    }

    /// Total start time in minutes from midnight
    pub fn start_minutes(&self) -> u32 {
        self.start_hour as u32 * 60 + self.start_minute as u32
    }

    /// Total end time in minutes from midnight
    pub fn end_minutes(&self) -> u32 {
        self.end_hour as u32 * 60 + self.end_minute as u32
    }

    /// Check if a given time (hour, minute) falls within the trading window
    pub fn is_within(&self, hour: u8, minute: u8) -> bool {
        let time = hour as u32 * 60 + minute as u32;
        let start = self.start_minutes();
        let end = self.end_minutes();

        if start <= end {
            time >= start && time < end
        } else {
            // Wraps past midnight
            time >= start || time < end
        }
    }
}

impl Default for TradingHours {
    fn default() -> Self {
        // Default: 24-hour trading (00:00 to 23:59)
        Self::new(0, 0, 23, 59)
    }
}

/// Configuration for `PropFirmRules`
#[derive(Debug, Clone)]
pub struct PropFirmRulesConfig {
    /// Initial account balance
    pub initial_balance: f64,
    /// Daily loss limit (absolute value, e.g. 500.0 means max $500 daily loss)
    pub daily_loss: RuleDefinition,
    /// Maximum drawdown (absolute value from peak or initial balance)
    pub max_drawdown: RuleDefinition,
    /// Drawdown calculation method
    pub drawdown_type: DrawdownType,
    /// Profit target (for evaluation phases)
    pub profit_target: RuleDefinition,
    /// Maximum position size per instrument (in lots/contracts)
    pub max_position_size: RuleDefinition,
    /// Maximum concurrent open positions
    pub max_concurrent_positions: RuleDefinition,
    /// Consistency rule: max fraction of total profit from a single day
    pub consistency_max_day_fraction: RuleDefinition,
    /// Minimum trading days required (for evaluation phases)
    pub min_trading_days: RuleDefinition,
    /// Allowed trading hours (None = 24/7)
    pub trading_hours: Option<TradingHours>,
    /// Trading hours enforcement mode
    pub trading_hours_mode: RuleMode,
    /// Whether to block trading around news events
    pub news_restriction_enabled: bool,
    /// Minutes before/after news to restrict trading
    pub news_blackout_minutes: u32,
    /// Whether overnight/weekend holding is restricted
    pub no_overnight_positions: bool,
    /// Overnight restriction enforcement mode
    pub overnight_mode: RuleMode,
    /// Sliding window for recent check results
    pub window_size: usize,
}

impl Default for PropFirmRulesConfig {
    fn default() -> Self {
        Self {
            initial_balance: 100_000.0,
            daily_loss: RuleDefinition::hard(5_000.0, "Daily loss limit"),
            max_drawdown: RuleDefinition::hard(10_000.0, "Maximum drawdown"),
            drawdown_type: DrawdownType::Trailing,
            profit_target: RuleDefinition::soft(10_000.0, "Profit target"),
            max_position_size: RuleDefinition::hard(10.0, "Max position size"),
            max_concurrent_positions: RuleDefinition::hard(5.0, "Max concurrent positions"),
            consistency_max_day_fraction: RuleDefinition::soft(0.40, "Consistency rule"),
            min_trading_days: RuleDefinition::soft(10.0, "Minimum trading days"),
            trading_hours: None,
            trading_hours_mode: RuleMode::Hard,
            news_restriction_enabled: false,
            news_blackout_minutes: 5,
            no_overnight_positions: false,
            overnight_mode: RuleMode::Soft,
            window_size: 200,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// Current account state for rule evaluation
#[derive(Debug, Clone)]
pub struct AccountState {
    /// Current equity (balance + unrealised P&L)
    pub equity: f64,
    /// Today's realised P&L
    pub daily_pnl: f64,
    /// Total realised profit since start
    pub total_profit: f64,
    /// Peak equity ever reached
    pub peak_equity: f64,
    /// Number of currently open positions
    pub open_positions: usize,
    /// Current hour (UTC, 0-23) for trading hours check
    pub current_hour: u8,
    /// Current minute (UTC, 0-59)
    pub current_minute: u8,
    /// Whether a high-impact news event is imminent or recent
    pub news_active: bool,
    /// Whether the market is in weekend/overnight session
    pub is_overnight: bool,
    /// Per-day profit map (day_index → profit)
    pub daily_profits: HashMap<u32, f64>,
    /// Number of distinct trading days so far
    pub trading_days: u32,
    /// Per-instrument position sizes (symbol → lots)
    pub position_sizes: HashMap<String, f64>,
}

impl Default for AccountState {
    fn default() -> Self {
        Self {
            equity: 100_000.0,
            daily_pnl: 0.0,
            total_profit: 0.0,
            peak_equity: 100_000.0,
            open_positions: 0,
            current_hour: 12,
            current_minute: 0,
            news_active: false,
            is_overnight: false,
            daily_profits: HashMap::new(),
            trading_days: 0,
            position_sizes: HashMap::new(),
        }
    }
}

/// A proposed order to check against rules
#[derive(Debug, Clone)]
pub struct ProposedOrder {
    /// Symbol/instrument
    pub symbol: String,
    /// Position size in lots/contracts
    pub size: f64,
    /// Whether this opens a new position (vs adding to existing)
    pub opens_new_position: bool,
}

/// Severity of a rule check finding
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// No issue
    Pass,
    /// Approaching limit (within warning threshold)
    Warning,
    /// Soft limit breached (allowed but flagged)
    SoftBreach,
    /// Hard limit breached (must reject)
    HardBreach,
}

/// A single rule violation or warning
#[derive(Debug, Clone)]
pub struct RuleViolation {
    /// Which rule was violated
    pub rule_name: String,
    /// Severity of the violation
    pub severity: Severity,
    /// Current value that triggered the violation
    pub current_value: f64,
    /// The limit value
    pub limit_value: f64,
    /// Utilisation ratio (current / limit)
    pub utilisation: f64,
    /// Human-readable message
    pub message: String,
}

/// Result of a comprehensive rule check
#[derive(Debug, Clone)]
pub struct RuleCheckResult {
    /// Whether trading is allowed (no hard breaches)
    pub allowed: bool,
    /// Maximum severity found across all rules
    pub max_severity: Severity,
    /// All violations and warnings found
    pub findings: Vec<RuleViolation>,
    /// Number of hard breaches
    pub hard_breaches: usize,
    /// Number of soft breaches
    pub soft_breaches: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Overall risk utilisation (0..1, max across all dimensions)
    pub max_utilisation: f64,
}

impl RuleCheckResult {
    fn new() -> Self {
        Self {
            allowed: true,
            max_severity: Severity::Pass,
            findings: Vec::new(),
            hard_breaches: 0,
            soft_breaches: 0,
            warnings: 0,
            max_utilisation: 0.0,
        }
    }

    fn add_finding(&mut self, finding: RuleViolation) {
        if finding.severity > self.max_severity {
            self.max_severity = finding.severity;
        }
        match finding.severity {
            Severity::HardBreach => {
                self.hard_breaches += 1;
                self.allowed = false;
            }
            Severity::SoftBreach => self.soft_breaches += 1,
            Severity::Warning => self.warnings += 1,
            Severity::Pass => {}
        }
        if finding.utilisation > self.max_utilisation {
            self.max_utilisation = finding.utilisation;
        }
        self.findings.push(finding);
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the prop firm rules engine
#[derive(Debug, Clone, Default)]
pub struct PropFirmRulesStats {
    /// Total rule checks performed
    pub total_checks: u64,
    /// Number of checks that resulted in a block (hard breach)
    pub blocked_count: u64,
    /// Number of checks with warnings
    pub warning_count: u64,
    /// Number of checks with soft breaches
    pub soft_breach_count: u64,
    /// Peak daily loss observed
    pub peak_daily_loss: f64,
    /// Peak drawdown observed
    pub peak_drawdown: f64,
    /// Peak position count observed
    pub peak_position_count: usize,
    /// Distinct trading days recorded
    pub trading_days_recorded: u32,
    /// Per-rule violation counts
    pub violation_counts: HashMap<String, u64>,
    /// Number of order checks performed
    pub order_checks: u64,
    /// Number of orders blocked
    pub orders_blocked: u64,
}

impl PropFirmRulesStats {
    /// Block rate (fraction of checks that resulted in a block)
    pub fn block_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.blocked_count as f64 / self.total_checks as f64
    }

    /// Warning rate
    pub fn warning_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.warning_count as f64 / self.total_checks as f64
    }

    /// Order block rate
    pub fn order_block_rate(&self) -> f64 {
        if self.order_checks == 0 {
            return 0.0;
        }
        self.orders_blocked as f64 / self.order_checks as f64
    }
}

// ---------------------------------------------------------------------------
// Internal record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CheckRecord {
    allowed: bool,
    max_severity: Severity,
    max_utilisation: f64,
    finding_count: usize,
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Prop firm trading rule compliance engine
pub struct PropFirmRules {
    config: PropFirmRulesConfig,

    /// Peak equity for drawdown tracking
    peak_equity: f64,

    /// Current daily loss tracking
    current_daily_loss: f64,

    /// Set of days with trading activity
    active_days: HashSet<u32>,

    /// Per-day profit tracking for consistency rule
    day_profits: HashMap<u32, f64>,

    /// Current day index
    current_day: u32,

    /// Sliding window of recent check results
    recent: VecDeque<CheckRecord>,

    /// Running statistics
    stats: PropFirmRulesStats,
}

impl Default for PropFirmRules {
    fn default() -> Self {
        Self::new()
    }
}

impl PropFirmRules {
    /// Create with default configuration
    pub fn new() -> Self {
        Self::with_config(PropFirmRulesConfig::default()).unwrap()
    }

    /// Create with a specific configuration
    pub fn with_config(config: PropFirmRulesConfig) -> Result<Self> {
        if config.initial_balance <= 0.0 {
            return Err(Error::InvalidInput("initial_balance must be > 0".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.daily_loss.is_active() && config.daily_loss.limit <= 0.0 {
            return Err(Error::InvalidInput(
                "daily_loss limit must be > 0 when active".into(),
            ));
        }
        if config.max_drawdown.is_active() && config.max_drawdown.limit <= 0.0 {
            return Err(Error::InvalidInput(
                "max_drawdown limit must be > 0 when active".into(),
            ));
        }

        let peak = config.initial_balance;

        Ok(Self {
            config,
            peak_equity: peak,
            current_daily_loss: 0.0,
            active_days: HashSet::new(),
            day_profits: HashMap::new(),
            current_day: 0,
            recent: VecDeque::new(),
            stats: PropFirmRulesStats::default(),
        })
    }

    /// Convenience factory
    pub fn process(config: PropFirmRulesConfig) -> Result<Self> {
        Self::with_config(config)
    }

    // -----------------------------------------------------------------------
    // Core checks
    // -----------------------------------------------------------------------

    /// Perform a comprehensive rule check against the current account state.
    /// This is the main entry point for periodic (e.g. per-tick) compliance monitoring.
    pub fn check(&mut self, state: &AccountState) -> RuleCheckResult {
        let mut result = RuleCheckResult::new();

        // Update peak equity
        if state.peak_equity > self.peak_equity {
            self.peak_equity = state.peak_equity;
        }

        // --- Daily loss ---
        if self.config.daily_loss.is_active() {
            let daily_loss = (-state.daily_pnl).max(0.0);
            self.current_daily_loss = daily_loss;

            if daily_loss > self.stats.peak_daily_loss {
                self.stats.peak_daily_loss = daily_loss;
            }

            let utilisation = if self.config.daily_loss.limit > 0.0 {
                daily_loss / self.config.daily_loss.limit
            } else {
                0.0
            };

            if daily_loss >= self.config.daily_loss.limit {
                let severity = match self.config.daily_loss.mode {
                    RuleMode::Hard => Severity::HardBreach,
                    RuleMode::Soft => Severity::SoftBreach,
                    RuleMode::Disabled => Severity::Pass,
                };
                result.add_finding(RuleViolation {
                    rule_name: "daily_loss".to_string(),
                    severity,
                    current_value: daily_loss,
                    limit_value: self.config.daily_loss.limit,
                    utilisation,
                    message: format!(
                        "Daily loss {:.2} exceeds limit {:.2}",
                        daily_loss, self.config.daily_loss.limit
                    ),
                });
                self.increment_violation("daily_loss");
            } else if daily_loss >= self.config.daily_loss.warning_value() {
                result.add_finding(RuleViolation {
                    rule_name: "daily_loss".to_string(),
                    severity: Severity::Warning,
                    current_value: daily_loss,
                    limit_value: self.config.daily_loss.limit,
                    utilisation,
                    message: format!(
                        "Daily loss {:.2} approaching limit {:.2} ({:.0}%)",
                        daily_loss,
                        self.config.daily_loss.limit,
                        utilisation * 100.0
                    ),
                });
            }
        }

        // --- Max drawdown ---
        if self.config.max_drawdown.is_active() {
            let drawdown = match self.config.drawdown_type {
                DrawdownType::Trailing => self.peak_equity - state.equity,
                DrawdownType::Static => self.config.initial_balance - state.equity,
                DrawdownType::EndOfDay => self.peak_equity - state.equity, // simplified
            };
            let drawdown = drawdown.max(0.0);

            if drawdown > self.stats.peak_drawdown {
                self.stats.peak_drawdown = drawdown;
            }

            let utilisation = if self.config.max_drawdown.limit > 0.0 {
                drawdown / self.config.max_drawdown.limit
            } else {
                0.0
            };

            if drawdown >= self.config.max_drawdown.limit {
                let severity = match self.config.max_drawdown.mode {
                    RuleMode::Hard => Severity::HardBreach,
                    RuleMode::Soft => Severity::SoftBreach,
                    RuleMode::Disabled => Severity::Pass,
                };
                result.add_finding(RuleViolation {
                    rule_name: "max_drawdown".to_string(),
                    severity,
                    current_value: drawdown,
                    limit_value: self.config.max_drawdown.limit,
                    utilisation,
                    message: format!(
                        "Drawdown {:.2} exceeds limit {:.2}",
                        drawdown, self.config.max_drawdown.limit
                    ),
                });
                self.increment_violation("max_drawdown");
            } else if drawdown >= self.config.max_drawdown.warning_value() {
                result.add_finding(RuleViolation {
                    rule_name: "max_drawdown".to_string(),
                    severity: Severity::Warning,
                    current_value: drawdown,
                    limit_value: self.config.max_drawdown.limit,
                    utilisation,
                    message: format!(
                        "Drawdown {:.2} approaching limit {:.2} ({:.0}%)",
                        drawdown,
                        self.config.max_drawdown.limit,
                        utilisation * 100.0
                    ),
                });
            }
        }

        // --- Trading hours ---
        if let Some(ref hours) = self.config.trading_hours {
            if self.config.trading_hours_mode != RuleMode::Disabled {
                let within = hours.is_within(state.current_hour, state.current_minute);
                if !within && state.open_positions > 0 {
                    let severity = match self.config.trading_hours_mode {
                        RuleMode::Hard => Severity::HardBreach,
                        RuleMode::Soft => Severity::SoftBreach,
                        RuleMode::Disabled => Severity::Pass,
                    };
                    result.add_finding(RuleViolation {
                        rule_name: "trading_hours".to_string(),
                        severity,
                        current_value: state.current_hour as f64,
                        limit_value: 0.0,
                        utilisation: 1.0,
                        message: format!(
                            "Trading outside allowed hours ({:02}:{:02})",
                            state.current_hour, state.current_minute
                        ),
                    });
                    self.increment_violation("trading_hours");
                }
            }
        }

        // --- News restriction ---
        if self.config.news_restriction_enabled && state.news_active && state.open_positions > 0 {
            result.add_finding(RuleViolation {
                rule_name: "news_restriction".to_string(),
                severity: Severity::Warning,
                current_value: 1.0,
                limit_value: 0.0,
                utilisation: 1.0,
                message: "Trading during news blackout period".to_string(),
            });
        }

        // --- Overnight holding ---
        if self.config.no_overnight_positions
            && self.config.overnight_mode != RuleMode::Disabled
            && state.is_overnight
            && state.open_positions > 0
        {
            let severity = match self.config.overnight_mode {
                RuleMode::Hard => Severity::HardBreach,
                RuleMode::Soft => Severity::SoftBreach,
                RuleMode::Disabled => Severity::Pass,
            };
            result.add_finding(RuleViolation {
                rule_name: "overnight_holding".to_string(),
                severity,
                current_value: state.open_positions as f64,
                limit_value: 0.0,
                utilisation: 1.0,
                message: format!("Holding {} positions overnight", state.open_positions),
            });
            self.increment_violation("overnight_holding");
        }

        // --- Concurrent positions ---
        if self.config.max_concurrent_positions.is_active() {
            let max = self.config.max_concurrent_positions.limit;
            let count = state.open_positions as f64;

            if state.open_positions > self.stats.peak_position_count {
                self.stats.peak_position_count = state.open_positions;
            }

            let utilisation = if max > 0.0 { count / max } else { 0.0 };

            if count >= max {
                let severity = match self.config.max_concurrent_positions.mode {
                    RuleMode::Hard => Severity::HardBreach,
                    RuleMode::Soft => Severity::SoftBreach,
                    RuleMode::Disabled => Severity::Pass,
                };
                result.add_finding(RuleViolation {
                    rule_name: "max_concurrent_positions".to_string(),
                    severity,
                    current_value: count,
                    limit_value: max,
                    utilisation,
                    message: format!(
                        "{} open positions exceeds limit {}",
                        state.open_positions, max as u32
                    ),
                });
                self.increment_violation("max_concurrent_positions");
            } else if count >= max * self.config.max_concurrent_positions.warning_threshold {
                result.add_finding(RuleViolation {
                    rule_name: "max_concurrent_positions".to_string(),
                    severity: Severity::Warning,
                    current_value: count,
                    limit_value: max,
                    utilisation,
                    message: format!(
                        "{} open positions approaching limit {}",
                        state.open_positions, max as u32
                    ),
                });
            }
        }

        // --- Consistency rule ---
        if self.config.consistency_max_day_fraction.is_active() && state.total_profit > 0.0 {
            let max_fraction = self.config.consistency_max_day_fraction.limit;
            for &day_profit in state.daily_profits.values() {
                if day_profit > 0.0 {
                    let fraction = day_profit / state.total_profit;
                    if fraction > max_fraction {
                        let severity = match self.config.consistency_max_day_fraction.mode {
                            RuleMode::Hard => Severity::HardBreach,
                            RuleMode::Soft => Severity::SoftBreach,
                            RuleMode::Disabled => Severity::Pass,
                        };
                        result.add_finding(RuleViolation {
                            rule_name: "consistency".to_string(),
                            severity,
                            current_value: fraction,
                            limit_value: max_fraction,
                            utilisation: fraction / max_fraction,
                            message: format!(
                                "Single day accounts for {:.0}% of total profit (limit: {:.0}%)",
                                fraction * 100.0,
                                max_fraction * 100.0
                            ),
                        });
                        self.increment_violation("consistency");
                        break; // Only report once
                    }
                }
            }
        }

        // --- Update stats ---
        self.stats.total_checks += 1;
        if !result.allowed {
            self.stats.blocked_count += 1;
        }
        if result.warnings > 0 {
            self.stats.warning_count += 1;
        }
        if result.soft_breaches > 0 {
            self.stats.soft_breach_count += 1;
        }

        // Window
        let record = CheckRecord {
            allowed: result.allowed,
            max_severity: result.max_severity,
            max_utilisation: result.max_utilisation,
            finding_count: result.findings.len(),
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        result
    }

    /// Pre-trade order check: can this specific order be placed?
    pub fn check_order(&mut self, state: &AccountState, order: &ProposedOrder) -> RuleCheckResult {
        let mut result = self.check(state);

        self.stats.order_checks += 1;

        // --- Position size check ---
        if self.config.max_position_size.is_active() {
            let max_size = self.config.max_position_size.limit;
            let current_size = state
                .position_sizes
                .get(&order.symbol)
                .copied()
                .unwrap_or(0.0);
            let new_size = current_size + order.size;

            let utilisation = if max_size > 0.0 {
                new_size / max_size
            } else {
                0.0
            };

            if new_size > max_size {
                let severity = match self.config.max_position_size.mode {
                    RuleMode::Hard => Severity::HardBreach,
                    RuleMode::Soft => Severity::SoftBreach,
                    RuleMode::Disabled => Severity::Pass,
                };
                result.add_finding(RuleViolation {
                    rule_name: "max_position_size".to_string(),
                    severity,
                    current_value: new_size,
                    limit_value: max_size,
                    utilisation,
                    message: format!(
                        "Position size {:.2} for {} would exceed limit {:.2}",
                        new_size, order.symbol, max_size
                    ),
                });
                self.increment_violation("max_position_size");
            } else if new_size >= max_size * self.config.max_position_size.warning_threshold {
                result.add_finding(RuleViolation {
                    rule_name: "max_position_size".to_string(),
                    severity: Severity::Warning,
                    current_value: new_size,
                    limit_value: max_size,
                    utilisation,
                    message: format!(
                        "Position size {:.2} for {} approaching limit {:.2}",
                        new_size, order.symbol, max_size
                    ),
                });
            }
        }

        // --- New position count check ---
        if order.opens_new_position && self.config.max_concurrent_positions.is_active() {
            let max = self.config.max_concurrent_positions.limit;
            let new_count = (state.open_positions + 1) as f64;

            if new_count > max {
                let severity = match self.config.max_concurrent_positions.mode {
                    RuleMode::Hard => Severity::HardBreach,
                    RuleMode::Soft => Severity::SoftBreach,
                    RuleMode::Disabled => Severity::Pass,
                };
                result.add_finding(RuleViolation {
                    rule_name: "max_concurrent_positions_order".to_string(),
                    severity,
                    current_value: new_count,
                    limit_value: max,
                    utilisation: new_count / max,
                    message: format!(
                        "Opening new position would result in {} positions (limit: {})",
                        new_count as u32, max as u32
                    ),
                });
                self.increment_violation("max_concurrent_positions_order");
            }
        }

        // --- Trading hours check for new orders ---
        if let Some(ref hours) = self.config.trading_hours {
            if self.config.trading_hours_mode != RuleMode::Disabled {
                if !hours.is_within(state.current_hour, state.current_minute) {
                    let severity = match self.config.trading_hours_mode {
                        RuleMode::Hard => Severity::HardBreach,
                        RuleMode::Soft => Severity::SoftBreach,
                        RuleMode::Disabled => Severity::Pass,
                    };
                    result.add_finding(RuleViolation {
                        rule_name: "trading_hours_order".to_string(),
                        severity,
                        current_value: state.current_hour as f64,
                        limit_value: 0.0,
                        utilisation: 1.0,
                        message: "Cannot place orders outside trading hours".to_string(),
                    });
                }
            }
        }

        // --- News restriction for new orders ---
        if self.config.news_restriction_enabled && state.news_active {
            result.add_finding(RuleViolation {
                rule_name: "news_restriction_order".to_string(),
                severity: Severity::Warning,
                current_value: 1.0,
                limit_value: 0.0,
                utilisation: 1.0,
                message: "Placing order during news blackout period".to_string(),
            });
        }

        if !result.allowed {
            self.stats.orders_blocked += 1;
        }

        result
    }

    /// Quick gate check: is trading currently allowed?
    pub fn can_trade(&mut self, state: &AccountState) -> bool {
        let result = self.check(state);
        result.allowed
    }

    // -----------------------------------------------------------------------
    // Daily management
    // -----------------------------------------------------------------------

    /// Reset daily-scoped rules (call at start of each trading day)
    pub fn reset_daily(&mut self, day_index: u32) {
        self.current_day = day_index;
        self.current_daily_loss = 0.0;
        self.active_days.insert(day_index);
        self.stats.trading_days_recorded = self.active_days.len() as u32;
    }

    /// Record a day's profit for consistency tracking
    pub fn record_day_profit(&mut self, day_index: u32, profit: f64) {
        self.day_profits.insert(day_index, profit);
        self.active_days.insert(day_index);
        self.stats.trading_days_recorded = self.active_days.len() as u32;
    }

    /// Update peak equity
    pub fn update_peak_equity(&mut self, equity: f64) {
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
    }

    // -----------------------------------------------------------------------
    // Utilisation / status queries
    // -----------------------------------------------------------------------

    /// Get utilisation for each active rule (0..1)
    pub fn utilisation(&self, state: &AccountState) -> HashMap<String, f64> {
        let mut utils = HashMap::new();

        if self.config.daily_loss.is_active() && self.config.daily_loss.limit > 0.0 {
            let daily_loss = (-state.daily_pnl).max(0.0);
            utils.insert(
                "daily_loss".to_string(),
                daily_loss / self.config.daily_loss.limit,
            );
        }

        if self.config.max_drawdown.is_active() && self.config.max_drawdown.limit > 0.0 {
            let drawdown = match self.config.drawdown_type {
                DrawdownType::Trailing => (self.peak_equity - state.equity).max(0.0),
                DrawdownType::Static => (self.config.initial_balance - state.equity).max(0.0),
                DrawdownType::EndOfDay => (self.peak_equity - state.equity).max(0.0),
            };
            utils.insert(
                "max_drawdown".to_string(),
                drawdown / self.config.max_drawdown.limit,
            );
        }

        if self.config.max_concurrent_positions.is_active()
            && self.config.max_concurrent_positions.limit > 0.0
        {
            utils.insert(
                "max_concurrent_positions".to_string(),
                state.open_positions as f64 / self.config.max_concurrent_positions.limit,
            );
        }

        if self.config.profit_target.is_active() && self.config.profit_target.limit > 0.0 {
            utils.insert(
                "profit_target".to_string(),
                state.total_profit / self.config.profit_target.limit,
            );
        }

        if self.config.min_trading_days.is_active() && self.config.min_trading_days.limit > 0.0 {
            utils.insert(
                "min_trading_days".to_string(),
                state.trading_days as f64 / self.config.min_trading_days.limit,
            );
        }

        utils
    }

    /// Whether the profit target has been reached
    pub fn profit_target_reached(&self, state: &AccountState) -> bool {
        if !self.config.profit_target.is_active() {
            return false;
        }
        state.total_profit >= self.config.profit_target.limit
    }

    /// Whether minimum trading days requirement is met
    pub fn min_trading_days_met(&self) -> bool {
        if !self.config.min_trading_days.is_active() {
            return true;
        }
        self.active_days.len() as f64 >= self.config.min_trading_days.limit
    }

    /// Remaining daily loss budget
    pub fn remaining_daily_loss_budget(&self, state: &AccountState) -> f64 {
        if !self.config.daily_loss.is_active() {
            return f64::MAX;
        }
        let daily_loss = (-state.daily_pnl).max(0.0);
        (self.config.daily_loss.limit - daily_loss).max(0.0)
    }

    /// Remaining drawdown budget
    pub fn remaining_drawdown_budget(&self, state: &AccountState) -> f64 {
        if !self.config.max_drawdown.is_active() {
            return f64::MAX;
        }
        let drawdown = match self.config.drawdown_type {
            DrawdownType::Trailing => (self.peak_equity - state.equity).max(0.0),
            DrawdownType::Static => (self.config.initial_balance - state.equity).max(0.0),
            DrawdownType::EndOfDay => (self.peak_equity - state.equity).max(0.0),
        };
        (self.config.max_drawdown.limit - drawdown).max(0.0)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn increment_violation(&mut self, rule_name: &str) {
        *self
            .stats
            .violation_counts
            .entry(rule_name.to_string())
            .or_insert(0) += 1;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Access running statistics
    pub fn stats(&self) -> &PropFirmRulesStats {
        &self.stats
    }

    /// Access the configuration
    pub fn config(&self) -> &PropFirmRulesConfig {
        &self.config
    }

    /// Current peak equity
    pub fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    /// Number of distinct active trading days
    pub fn active_days_count(&self) -> u32 {
        self.active_days.len() as u32
    }

    /// Window count
    pub fn window_count(&self) -> usize {
        self.recent.len()
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Windowed pass rate (fraction of recent checks that were allowed)
    pub fn windowed_pass_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let passed = self.recent.iter().filter(|r| r.allowed).count();
        passed as f64 / self.recent.len() as f64
    }

    /// Windowed mean utilisation
    pub fn windowed_mean_utilisation(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.max_utilisation).sum();
        sum / self.recent.len() as f64
    }

    /// Whether compliance is deteriorating (pass rate declining)
    pub fn is_compliance_deteriorating(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_pass =
            self.recent.iter().take(half).filter(|r| r.allowed).count() as f64 / half as f64;
        let second_half_len = self.recent.len() - half;
        let second_pass = self.recent.iter().skip(half).filter(|r| r.allowed).count() as f64
            / second_half_len as f64;
        second_pass < first_pass * 0.8
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all state, keeping configuration
    pub fn reset(&mut self) {
        self.peak_equity = self.config.initial_balance;
        self.current_daily_loss = 0.0;
        self.active_days.clear();
        self.day_profits.clear();
        self.current_day = 0;
        self.recent.clear();
        self.stats = PropFirmRulesStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_state() -> AccountState {
        AccountState::default()
    }

    fn state_with_loss(daily_pnl: f64) -> AccountState {
        AccountState {
            daily_pnl,
            equity: 100_000.0 + daily_pnl,
            peak_equity: 100_000.0,
            ..Default::default()
        }
    }

    fn state_with_drawdown(equity: f64) -> AccountState {
        AccountState {
            equity,
            peak_equity: 100_000.0,
            ..Default::default()
        }
    }

    fn default_order() -> ProposedOrder {
        ProposedOrder {
            symbol: "EURUSD".to_string(),
            size: 1.0,
            opens_new_position: true,
        }
    }

    #[test]
    fn test_basic() {
        let instance = PropFirmRules::new();
        assert_eq!(instance.peak_equity(), 100_000.0);
    }

    #[test]
    fn test_default_config() {
        let pf = PropFirmRules::new();
        assert_eq!(pf.config().initial_balance, 100_000.0);
        assert!(pf.config().daily_loss.is_active());
    }

    // -- Clean state passes --

    #[test]
    fn test_clean_state_passes() {
        let mut pf = PropFirmRules::new();
        let result = pf.check(&default_state());
        assert!(result.allowed);
        assert_eq!(result.max_severity, Severity::Pass);
        assert_eq!(result.hard_breaches, 0);
    }

    // -- Daily loss tests --

    #[test]
    fn test_daily_loss_breach() {
        let mut pf = PropFirmRules::new();
        let state = state_with_loss(-6000.0); // exceeds 5000 limit
        let result = pf.check(&state);
        assert!(!result.allowed);
        assert_eq!(result.max_severity, Severity::HardBreach);
    }

    #[test]
    fn test_daily_loss_warning() {
        let mut pf = PropFirmRules::new();
        // Warning at 80% of 5000 = 4000
        let state = state_with_loss(-4200.0);
        let result = pf.check(&state);
        assert!(result.allowed); // warnings don't block
        assert!(result.warnings > 0);
    }

    #[test]
    fn test_daily_loss_within_limit() {
        let mut pf = PropFirmRules::new();
        let state = state_with_loss(-1000.0);
        let result = pf.check(&state);
        assert!(result.allowed);
        assert_eq!(result.findings.len(), 0);
    }

    #[test]
    fn test_daily_loss_positive_pnl_no_issue() {
        let mut pf = PropFirmRules::new();
        let state = AccountState {
            daily_pnl: 3000.0,
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(result.allowed);
    }

    #[test]
    fn test_daily_loss_soft_mode() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            daily_loss: RuleDefinition::soft(5000.0, "Daily loss"),
            ..Default::default()
        })
        .unwrap();

        let state = state_with_loss(-6000.0);
        let result = pf.check(&state);
        assert!(result.allowed); // soft mode doesn't block
        assert!(result.soft_breaches > 0);
    }

    #[test]
    fn test_daily_loss_disabled() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            daily_loss: RuleDefinition::disabled(),
            max_drawdown: RuleDefinition::disabled(),
            ..Default::default()
        })
        .unwrap();

        let state = state_with_loss(-99999.0);
        let result = pf.check(&state);
        assert!(result.allowed);
    }

    // -- Drawdown tests --

    #[test]
    fn test_trailing_drawdown_breach() {
        let mut pf = PropFirmRules::new();
        let state = state_with_drawdown(89_000.0); // 11K drawdown > 10K limit
        let result = pf.check(&state);
        assert!(!result.allowed);
    }

    #[test]
    fn test_trailing_drawdown_warning() {
        let mut pf = PropFirmRules::new();
        // Warning at 80% of 10000 = 8000
        let state = state_with_drawdown(91_500.0); // 8500 drawdown
        let result = pf.check(&state);
        assert!(result.allowed);
        assert!(result.warnings > 0);
    }

    #[test]
    fn test_trailing_drawdown_within_limit() {
        let mut pf = PropFirmRules::new();
        let state = state_with_drawdown(95_000.0); // 5K drawdown
        let result = pf.check(&state);
        assert!(result.allowed);
    }

    #[test]
    fn test_static_drawdown() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            drawdown_type: DrawdownType::Static,
            max_drawdown: RuleDefinition::hard(10_000.0, "Max DD"),
            initial_balance: 100_000.0,
            ..Default::default()
        })
        .unwrap();

        let state = AccountState {
            equity: 88_000.0,
            peak_equity: 105_000.0, // peak doesn't matter for static
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(!result.allowed); // 12K static drawdown > 10K limit
    }

    // -- Concurrent positions --

    #[test]
    fn test_concurrent_positions_breach() {
        let mut pf = PropFirmRules::new();
        let state = AccountState {
            open_positions: 6, // > limit of 5
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(!result.allowed);
    }

    #[test]
    fn test_concurrent_positions_within_limit() {
        let mut pf = PropFirmRules::new();
        let state = AccountState {
            open_positions: 3,
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(result.allowed);
    }

    // -- Trading hours --

    #[test]
    fn test_trading_hours_within() {
        let hours = TradingHours::new(8, 0, 16, 0);
        assert!(hours.is_within(12, 0));
        assert!(hours.is_within(8, 0));
        assert!(!hours.is_within(7, 59));
        assert!(!hours.is_within(16, 0));
    }

    #[test]
    fn test_trading_hours_wrap_midnight() {
        let hours = TradingHours::new(22, 0, 6, 0);
        assert!(hours.is_within(23, 0));
        assert!(hours.is_within(2, 0));
        assert!(!hours.is_within(12, 0));
    }

    #[test]
    fn test_trading_hours_default_all_day() {
        let hours = TradingHours::default();
        assert!(hours.is_within(0, 0));
        assert!(hours.is_within(12, 0));
        assert!(hours.is_within(23, 58));
    }

    #[test]
    fn test_trading_hours_block() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            trading_hours: Some(TradingHours::new(8, 0, 16, 0)),
            trading_hours_mode: RuleMode::Hard,
            ..Default::default()
        })
        .unwrap();

        let state = AccountState {
            current_hour: 20,
            current_minute: 0,
            open_positions: 1,
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(!result.allowed);
    }

    #[test]
    fn test_trading_hours_allows_within() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            trading_hours: Some(TradingHours::new(8, 0, 16, 0)),
            trading_hours_mode: RuleMode::Hard,
            ..Default::default()
        })
        .unwrap();

        let state = AccountState {
            current_hour: 12,
            current_minute: 0,
            open_positions: 1,
            ..Default::default()
        };
        let result = pf.check(&state);
        // Should not be blocked for trading hours
        let hours_violations: Vec<_> = result
            .findings
            .iter()
            .filter(|f| f.rule_name == "trading_hours")
            .collect();
        assert!(hours_violations.is_empty());
    }

    // -- Overnight --

    #[test]
    fn test_overnight_holding_blocked() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            no_overnight_positions: true,
            overnight_mode: RuleMode::Hard,
            ..Default::default()
        })
        .unwrap();

        let state = AccountState {
            is_overnight: true,
            open_positions: 2,
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(!result.allowed);
    }

    #[test]
    fn test_overnight_no_positions_ok() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            no_overnight_positions: true,
            overnight_mode: RuleMode::Hard,
            ..Default::default()
        })
        .unwrap();

        let state = AccountState {
            is_overnight: true,
            open_positions: 0,
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(result.allowed);
    }

    // -- News restriction --

    #[test]
    fn test_news_restriction_warning() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            news_restriction_enabled: true,
            ..Default::default()
        })
        .unwrap();

        let state = AccountState {
            news_active: true,
            open_positions: 1,
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(result.warnings > 0);
    }

    // -- Consistency rule --

    #[test]
    fn test_consistency_breach() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            consistency_max_day_fraction: RuleDefinition::hard(0.30, "Consistency"),
            ..Default::default()
        })
        .unwrap();

        let mut daily_profits = HashMap::new();
        daily_profits.insert(1, 8000.0); // 80% of total
        daily_profits.insert(2, 1000.0);
        daily_profits.insert(3, 1000.0);

        let state = AccountState {
            total_profit: 10_000.0,
            daily_profits,
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(!result.allowed);
    }

    #[test]
    fn test_consistency_passes() {
        let mut pf = PropFirmRules::new();

        let mut daily_profits = HashMap::new();
        daily_profits.insert(1, 3000.0); // 30% < 40% limit
        daily_profits.insert(2, 3500.0);
        daily_profits.insert(3, 3500.0);

        let state = AccountState {
            total_profit: 10_000.0,
            daily_profits,
            ..Default::default()
        };
        let result = pf.check(&state);
        // No consistency violation
        let consistency_findings: Vec<_> = result
            .findings
            .iter()
            .filter(|f| f.rule_name == "consistency")
            .collect();
        assert!(consistency_findings.is_empty());
    }

    // -- Order checks --

    #[test]
    fn test_check_order_position_size_breach() {
        let mut pf = PropFirmRules::new();
        let mut sizes = HashMap::new();
        sizes.insert("EURUSD".to_string(), 9.0);

        let state = AccountState {
            position_sizes: sizes,
            ..Default::default()
        };
        let order = ProposedOrder {
            symbol: "EURUSD".to_string(),
            size: 2.0, // 9 + 2 = 11 > 10 limit
            opens_new_position: false,
        };
        let result = pf.check_order(&state, &order);
        assert!(!result.allowed);
    }

    #[test]
    fn test_check_order_within_size_limit() {
        let mut pf = PropFirmRules::new();
        let state = AccountState::default();
        let order = ProposedOrder {
            symbol: "EURUSD".to_string(),
            size: 5.0,
            opens_new_position: true,
        };
        let result = pf.check_order(&state, &order);
        // Should pass since 5 < 10 and 1 < 5 positions
        assert!(result.allowed);
    }

    #[test]
    fn test_check_order_new_position_exceeds_count() {
        let mut pf = PropFirmRules::new();
        let state = AccountState {
            open_positions: 5, // already at limit
            ..Default::default()
        };
        let order = ProposedOrder {
            symbol: "GBPUSD".to_string(),
            size: 1.0,
            opens_new_position: true,
        };
        let result = pf.check_order(&state, &order);
        assert!(!result.allowed);
    }

    #[test]
    fn test_check_order_outside_trading_hours() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            trading_hours: Some(TradingHours::new(8, 0, 16, 0)),
            trading_hours_mode: RuleMode::Hard,
            ..Default::default()
        })
        .unwrap();

        let state = AccountState {
            current_hour: 20,
            ..Default::default()
        };
        let result = pf.check_order(&state, &default_order());
        assert!(!result.allowed);
    }

    // -- Can trade --

    #[test]
    fn test_can_trade_normal() {
        let mut pf = PropFirmRules::new();
        assert!(pf.can_trade(&default_state()));
    }

    #[test]
    fn test_can_trade_blocked() {
        let mut pf = PropFirmRules::new();
        assert!(!pf.can_trade(&state_with_loss(-6000.0)));
    }

    // -- Daily management --

    #[test]
    fn test_reset_daily() {
        let mut pf = PropFirmRules::new();
        pf.reset_daily(1);
        pf.reset_daily(2);
        assert_eq!(pf.active_days_count(), 2);
    }

    #[test]
    fn test_record_day_profit() {
        let mut pf = PropFirmRules::new();
        pf.record_day_profit(1, 500.0);
        pf.record_day_profit(2, 300.0);
        assert_eq!(pf.active_days_count(), 2);
    }

    #[test]
    fn test_update_peak_equity() {
        let mut pf = PropFirmRules::new();
        pf.update_peak_equity(110_000.0);
        assert!((pf.peak_equity() - 110_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_peak_equity_only_increases() {
        let mut pf = PropFirmRules::new();
        pf.update_peak_equity(110_000.0);
        pf.update_peak_equity(105_000.0);
        assert!((pf.peak_equity() - 110_000.0).abs() < 1e-10);
    }

    // -- Profit target --

    #[test]
    fn test_profit_target_reached() {
        let pf = PropFirmRules::new();
        let state = AccountState {
            total_profit: 12_000.0, // > 10K target
            ..Default::default()
        };
        assert!(pf.profit_target_reached(&state));
    }

    #[test]
    fn test_profit_target_not_reached() {
        let pf = PropFirmRules::new();
        let state = AccountState {
            total_profit: 5_000.0,
            ..Default::default()
        };
        assert!(!pf.profit_target_reached(&state));
    }

    // -- Min trading days --

    #[test]
    fn test_min_trading_days_met() {
        let mut pf = PropFirmRules::new();
        for i in 0..10 {
            pf.reset_daily(i);
        }
        assert!(pf.min_trading_days_met());
    }

    #[test]
    fn test_min_trading_days_not_met() {
        let mut pf = PropFirmRules::new();
        for i in 0..5 {
            pf.reset_daily(i);
        }
        assert!(!pf.min_trading_days_met());
    }

    // -- Budget queries --

    #[test]
    fn test_remaining_daily_loss_budget() {
        let pf = PropFirmRules::new();
        let state = state_with_loss(-2000.0);
        let budget = pf.remaining_daily_loss_budget(&state);
        assert!((budget - 3000.0).abs() < 1e-10); // 5000 - 2000
    }

    #[test]
    fn test_remaining_drawdown_budget() {
        let pf = PropFirmRules::new();
        let state = state_with_drawdown(96_000.0); // 4K drawdown
        let budget = pf.remaining_drawdown_budget(&state);
        assert!((budget - 6000.0).abs() < 1e-10); // 10000 - 4000
    }

    // -- Utilisation --

    #[test]
    fn test_utilisation() {
        let pf = PropFirmRules::new();
        let state = state_with_loss(-2500.0);
        let utils = pf.utilisation(&state);
        let daily = utils.get("daily_loss").unwrap();
        assert!((daily - 0.5).abs() < 1e-10); // 2500 / 5000
    }

    // -- Stats --

    #[test]
    fn test_stats_tracking() {
        let mut pf = PropFirmRules::new();
        pf.check(&default_state());
        pf.check(&state_with_loss(-6000.0));
        pf.check(&default_state());

        assert_eq!(pf.stats().total_checks, 3);
        assert_eq!(pf.stats().blocked_count, 1);
    }

    #[test]
    fn test_stats_block_rate() {
        let mut pf = PropFirmRules::new();
        pf.check(&default_state());
        pf.check(&state_with_loss(-6000.0));

        assert!((pf.stats().block_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_violation_counts() {
        let mut pf = PropFirmRules::new();
        pf.check(&state_with_loss(-6000.0));
        pf.check(&state_with_loss(-6000.0));

        let count = pf.stats().violation_counts.get("daily_loss").unwrap();
        assert_eq!(*count, 2);
    }

    #[test]
    fn test_stats_peak_daily_loss() {
        let mut pf = PropFirmRules::new();
        pf.check(&state_with_loss(-2000.0));
        pf.check(&state_with_loss(-4000.0));
        pf.check(&state_with_loss(-1000.0));

        assert!((pf.stats().peak_daily_loss - 4000.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_peak_drawdown() {
        let mut pf = PropFirmRules::new();
        pf.check(&state_with_drawdown(97_000.0)); // 3K DD
        pf.check(&state_with_drawdown(94_000.0)); // 6K DD
        pf.check(&state_with_drawdown(96_000.0)); // 4K DD

        assert!((pf.stats().peak_drawdown - 6000.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_order_block_rate() {
        let mut pf = PropFirmRules::new();
        pf.check_order(&default_state(), &default_order());
        pf.check_order(&state_with_loss(-6000.0), &default_order());

        assert!((pf.stats().order_block_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = PropFirmRulesStats::default();
        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.block_rate(), 0.0);
        assert_eq!(stats.warning_rate(), 0.0);
        assert_eq!(stats.order_block_rate(), 0.0);
    }

    // -- Window tests --

    #[test]
    fn test_windowed_pass_rate() {
        let mut pf = PropFirmRules::new();
        pf.check(&default_state()); // pass
        pf.check(&state_with_loss(-6000.0)); // fail

        assert!((pf.windowed_pass_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_utilisation() {
        let mut pf = PropFirmRules::new();
        pf.check(&default_state());
        assert!(pf.windowed_mean_utilisation() >= 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let pf = PropFirmRules::new();
        assert_eq!(pf.windowed_pass_rate(), 0.0);
        assert_eq!(pf.windowed_mean_utilisation(), 0.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            window_size: 3,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..5 {
            pf.check(&default_state());
        }
        assert_eq!(pf.window_count(), 3);
    }

    // -- Compliance deteriorating --

    #[test]
    fn test_compliance_deteriorating() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        // First half: all pass
        for _ in 0..10 {
            pf.check(&default_state());
        }
        // Second half: all fail
        for _ in 0..10 {
            pf.check(&state_with_loss(-6000.0));
        }

        assert!(pf.is_compliance_deteriorating());
    }

    #[test]
    fn test_not_deteriorating_consistent() {
        let mut pf = PropFirmRules::with_config(PropFirmRulesConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..20 {
            pf.check(&default_state());
        }

        assert!(!pf.is_compliance_deteriorating());
    }

    #[test]
    fn test_not_deteriorating_insufficient_data() {
        let mut pf = PropFirmRules::new();
        pf.check(&default_state());
        assert!(!pf.is_compliance_deteriorating());
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut pf = PropFirmRules::new();
        pf.check(&state_with_loss(-6000.0));
        pf.reset_daily(1);
        pf.record_day_profit(1, 500.0);

        pf.reset();
        assert_eq!(pf.peak_equity(), 100_000.0);
        assert_eq!(pf.active_days_count(), 0);
        assert_eq!(pf.window_count(), 0);
        assert_eq!(pf.stats().total_checks, 0);
    }

    // -- Config validation --

    #[test]
    fn test_invalid_config_zero_balance() {
        let result = PropFirmRules::with_config(PropFirmRulesConfig {
            initial_balance: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let result = PropFirmRules::with_config(PropFirmRulesConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_daily_loss_when_active() {
        let result = PropFirmRules::with_config(PropFirmRulesConfig {
            daily_loss: RuleDefinition::hard(0.0, "test"),
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_drawdown_when_active() {
        let result = PropFirmRules::with_config(PropFirmRulesConfig {
            max_drawdown: RuleDefinition::hard(0.0, "test"),
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- Process convenience --

    #[test]
    fn test_process_returns_instance() {
        let pf = PropFirmRules::process(PropFirmRulesConfig::default());
        assert!(pf.is_ok());
    }

    #[test]
    fn test_process_rejects_bad_config() {
        let result = PropFirmRules::process(PropFirmRulesConfig {
            initial_balance: -1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- RuleDefinition --

    #[test]
    fn test_rule_definition_hard() {
        let rule = RuleDefinition::hard(1000.0, "Test");
        assert!(rule.is_active());
        assert_eq!(rule.mode, RuleMode::Hard);
        assert!((rule.warning_value() - 800.0).abs() < 1e-10);
    }

    #[test]
    fn test_rule_definition_soft() {
        let rule = RuleDefinition::soft(1000.0, "Test");
        assert!(rule.is_active());
        assert_eq!(rule.mode, RuleMode::Soft);
    }

    #[test]
    fn test_rule_definition_disabled() {
        let rule = RuleDefinition::disabled();
        assert!(!rule.is_active());
    }

    #[test]
    fn test_rule_definition_custom_warning() {
        let rule = RuleDefinition::hard(1000.0, "Test").with_warning_threshold(0.9);
        assert!((rule.warning_value() - 900.0).abs() < 1e-10);
    }

    // -- Severity ordering --

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Pass < Severity::Warning);
        assert!(Severity::Warning < Severity::SoftBreach);
        assert!(Severity::SoftBreach < Severity::HardBreach);
    }

    // -- Multiple violations --

    #[test]
    fn test_multiple_violations() {
        let mut pf = PropFirmRules::new();
        let state = AccountState {
            daily_pnl: -6000.0,
            equity: 88_000.0,
            peak_equity: 100_000.0,
            open_positions: 6,
            ..Default::default()
        };
        let result = pf.check(&state);
        assert!(!result.allowed);
        assert!(result.hard_breaches >= 3); // daily_loss + drawdown + positions
    }

    // -- RuleCheckResult --

    #[test]
    fn test_rule_check_result_new() {
        let result = RuleCheckResult::new();
        assert!(result.allowed);
        assert_eq!(result.max_severity, Severity::Pass);
        assert_eq!(result.hard_breaches, 0);
        assert_eq!(result.soft_breaches, 0);
        assert_eq!(result.warnings, 0);
    }

    // -- TradingHours --

    #[test]
    fn test_trading_hours_minutes() {
        let hours = TradingHours::new(8, 30, 16, 45);
        assert_eq!(hours.start_minutes(), 510);
        assert_eq!(hours.end_minutes(), 1005);
    }

    #[test]
    fn test_trading_hours_boundary() {
        let hours = TradingHours::new(9, 0, 17, 0);
        assert!(hours.is_within(9, 0)); // start inclusive
        assert!(!hours.is_within(17, 0)); // end exclusive
        assert!(hours.is_within(16, 59));
    }

    // -- DrawdownType --

    #[test]
    fn test_drawdown_type_default() {
        assert_eq!(DrawdownType::default(), DrawdownType::Trailing);
    }

    // -- Position size warning --

    #[test]
    fn test_position_size_warning() {
        let mut pf = PropFirmRules::new();
        let mut sizes = HashMap::new();
        sizes.insert("EURUSD".to_string(), 7.5); // 75% of 10

        let state = AccountState {
            position_sizes: sizes,
            ..Default::default()
        };
        let order = ProposedOrder {
            symbol: "EURUSD".to_string(),
            size: 1.0, // total 8.5 > 80% warning
            opens_new_position: false,
        };
        let result = pf.check_order(&state, &order);
        assert!(result.allowed); // warning, not breach
        assert!(
            result.warnings > 0
                || result
                    .findings
                    .iter()
                    .any(|f| f.rule_name == "max_position_size")
        );
    }
}
