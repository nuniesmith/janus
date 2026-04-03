//! Risk limits and violation tracking.
//!
//! This module provides mechanisms to enforce portfolio-level risk limits
//! and track violations.

use serde::{Deserialize, Serialize};

/// Type of risk limit violation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Maximum drawdown exceeded
    MaxDrawdown,

    /// Maximum portfolio risk exceeded
    MaxPortfolioRisk,

    /// Maximum position size exceeded
    MaxPositionSize,

    /// Maximum number of positions exceeded
    MaxPositions,

    /// Maximum asset exposure exceeded
    MaxAssetExposure,

    /// Minimum confidence not met
    MinConfidence,
}

impl std::fmt::Display for ViolationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationType::MaxDrawdown => write!(f, "MAX_DRAWDOWN"),
            ViolationType::MaxPortfolioRisk => write!(f, "MAX_PORTFOLIO_RISK"),
            ViolationType::MaxPositionSize => write!(f, "MAX_POSITION_SIZE"),
            ViolationType::MaxPositions => write!(f, "MAX_POSITIONS"),
            ViolationType::MaxAssetExposure => write!(f, "MAX_ASSET_EXPOSURE"),
            ViolationType::MinConfidence => write!(f, "MIN_CONFIDENCE"),
        }
    }
}

/// Risk limit violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// Type of violation
    pub violation_type: ViolationType,

    /// Current value that caused the violation
    pub current_value: f64,

    /// Limit that was exceeded
    pub limit: f64,

    /// Timestamp of violation
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Additional context
    pub context: String,
}

impl Violation {
    /// Create a new violation
    pub fn new(
        violation_type: ViolationType,
        current_value: f64,
        limit: f64,
        context: String,
    ) -> Self {
        Self {
            violation_type,
            current_value,
            limit,
            timestamp: chrono::Utc::now(),
            context,
        }
    }

    /// Get severity as percentage over limit
    pub fn severity(&self) -> f64 {
        if self.limit == 0.0 {
            return 0.0;
        }
        ((self.current_value - self.limit) / self.limit).abs()
    }

    /// Check if this is a critical violation (>20% over limit)
    pub fn is_critical(&self) -> bool {
        self.severity() > 0.2
    }
}

/// Portfolio-level risk limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum drawdown allowed (fraction)
    pub max_drawdown: f64,

    /// Maximum portfolio risk (fraction of capital)
    pub max_portfolio_risk: f64,

    /// Maximum position size (fraction of capital)
    pub max_position_size: f64,

    /// Maximum number of concurrent positions
    pub max_positions: usize,

    /// Maximum exposure per asset (fraction of capital)
    pub max_asset_exposure: f64,

    /// Minimum signal confidence
    pub min_confidence: f64,

    /// Stop trading on violation
    pub stop_on_violation: bool,

    /// Violations recorded
    violations: Vec<Violation>,

    /// Trading stopped flag
    trading_stopped: bool,
}

impl RiskLimits {
    /// Create new risk limits
    pub fn new(
        max_drawdown: f64,
        max_portfolio_risk: f64,
        max_position_size: f64,
        max_positions: usize,
    ) -> Self {
        Self {
            max_drawdown,
            max_portfolio_risk,
            max_position_size,
            max_positions,
            max_asset_exposure: max_position_size, // Default to same as position size
            min_confidence: 0.6,
            stop_on_violation: true,
            violations: Vec::new(),
            trading_stopped: false,
        }
    }

    /// Check if a drawdown exceeds the limit
    pub fn check_drawdown(&mut self, current_dd: f64) -> bool {
        if current_dd > self.max_drawdown {
            self.record_violation(Violation::new(
                ViolationType::MaxDrawdown,
                current_dd,
                self.max_drawdown,
                format!(
                    "Drawdown {:.2}% exceeds limit {:.2}%",
                    current_dd * 100.0,
                    self.max_drawdown * 100.0
                ),
            ));
            return false;
        }
        true
    }

    /// Check if portfolio risk exceeds the limit
    pub fn check_portfolio_risk(&mut self, current_risk: f64) -> bool {
        if current_risk > self.max_portfolio_risk {
            self.record_violation(Violation::new(
                ViolationType::MaxPortfolioRisk,
                current_risk,
                self.max_portfolio_risk,
                format!(
                    "Portfolio risk {:.2}% exceeds limit {:.2}%",
                    current_risk * 100.0,
                    self.max_portfolio_risk * 100.0
                ),
            ));
            return false;
        }
        true
    }

    /// Check if position size exceeds the limit
    pub fn check_position_size(&mut self, size_fraction: f64) -> bool {
        if size_fraction > self.max_position_size {
            self.record_violation(Violation::new(
                ViolationType::MaxPositionSize,
                size_fraction,
                self.max_position_size,
                format!(
                    "Position size {:.2}% exceeds limit {:.2}%",
                    size_fraction * 100.0,
                    self.max_position_size * 100.0
                ),
            ));
            return false;
        }
        true
    }

    /// Check if number of positions exceeds the limit
    pub fn check_num_positions(&mut self, num_positions: usize) -> bool {
        if num_positions > self.max_positions {
            self.record_violation(Violation::new(
                ViolationType::MaxPositions,
                num_positions as f64,
                self.max_positions as f64,
                format!(
                    "Number of positions {} exceeds limit {}",
                    num_positions, self.max_positions
                ),
            ));
            return false;
        }
        true
    }

    /// Check if confidence meets the minimum
    pub fn check_confidence(&mut self, confidence: f64) -> bool {
        if confidence < self.min_confidence {
            self.record_violation(Violation::new(
                ViolationType::MinConfidence,
                confidence,
                self.min_confidence,
                format!(
                    "Confidence {:.2}% below minimum {:.2}%",
                    confidence * 100.0,
                    self.min_confidence * 100.0
                ),
            ));
            return false;
        }
        true
    }

    /// Record a violation
    fn record_violation(&mut self, violation: Violation) {
        if self.stop_on_violation {
            self.trading_stopped = true;
        }
        self.violations.push(violation);
    }

    /// Get all violations
    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }

    /// Get recent violations (last N)
    pub fn recent_violations(&self, n: usize) -> Vec<&Violation> {
        let start = self.violations.len().saturating_sub(n);
        self.violations[start..].iter().collect()
    }

    /// Check if trading is stopped
    pub fn is_trading_stopped(&self) -> bool {
        self.trading_stopped
    }

    /// Resume trading after violation
    pub fn resume_trading(&mut self) {
        self.trading_stopped = false;
    }

    /// Clear all violations
    pub fn clear_violations(&mut self) {
        self.violations.clear();
        self.trading_stopped = false;
    }

    /// Get violation count by type
    pub fn violation_count(&self, violation_type: ViolationType) -> usize {
        self.violations
            .iter()
            .filter(|v| v.violation_type == violation_type)
            .count()
    }

    /// Get total violation count
    pub fn total_violations(&self) -> usize {
        self.violations.len()
    }

    /// Check if any limits would be violated
    pub fn would_violate(&self, drawdown: f64, portfolio_risk: f64, num_positions: usize) -> bool {
        drawdown > self.max_drawdown
            || portfolio_risk > self.max_portfolio_risk
            || num_positions > self.max_positions
    }
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self::new(0.15, 0.20, 0.10, 5)
    }
}

/// Drawdown monitor for tracking equity peaks and valleys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownMonitor {
    /// Peak equity
    peak_equity: f64,

    /// Current equity
    current_equity: f64,

    /// Maximum drawdown limit (fraction)
    max_drawdown_limit: f64,

    /// Current drawdown (fraction)
    current_drawdown: f64,

    /// Historical max drawdown
    historical_max_drawdown: f64,

    /// Trading stopped due to drawdown
    stopped: bool,

    /// Drawdown history (for analysis)
    drawdown_history: Vec<f64>,
}

impl DrawdownMonitor {
    /// Create a new drawdown monitor
    pub fn new(max_drawdown_limit: f64) -> Self {
        Self {
            peak_equity: 0.0,
            current_equity: 0.0,
            max_drawdown_limit,
            current_drawdown: 0.0,
            historical_max_drawdown: 0.0,
            stopped: false,
            drawdown_history: Vec::new(),
        }
    }

    /// Update with new equity values
    pub fn update(&mut self, current_equity: f64, peak_equity: f64) {
        self.current_equity = current_equity;
        self.peak_equity = peak_equity.max(self.peak_equity);

        // Calculate current drawdown
        if self.peak_equity > 0.0 {
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity;
        } else {
            self.current_drawdown = 0.0;
        }

        // Update historical max
        if self.current_drawdown > self.historical_max_drawdown {
            self.historical_max_drawdown = self.current_drawdown;
        }

        // Record in history
        self.drawdown_history.push(self.current_drawdown);

        // Check if we should stop
        if self.current_drawdown > self.max_drawdown_limit {
            self.stopped = true;
        }
    }

    /// Get current drawdown percentage
    pub fn current_drawdown_pct(&self) -> f64 {
        self.current_drawdown
    }

    /// Get current drawdown in dollars
    pub fn current_drawdown_dollars(&self) -> f64 {
        self.peak_equity - self.current_equity
    }

    /// Check if trading should be stopped
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Get scaling factor based on current drawdown
    /// Returns 1.0 at no drawdown, decreases as drawdown increases
    pub fn get_scaling_factor(&self) -> f64 {
        if self.current_drawdown >= self.max_drawdown_limit {
            return 0.0; // Stop trading
        }

        // Linear scaling from 1.0 to 0.5 as we approach the limit
        let ratio = self.current_drawdown / self.max_drawdown_limit;
        1.0 - (ratio * 0.5)
    }

    /// Reset the monitor
    pub fn reset(&mut self) {
        self.peak_equity = 0.0;
        self.current_equity = 0.0;
        self.current_drawdown = 0.0;
        self.stopped = false;
        self.drawdown_history.clear();
    }

    /// Resume trading after stop
    pub fn resume(&mut self) {
        self.stopped = false;
    }

    /// Get drawdown history
    pub fn history(&self) -> &[f64] {
        &self.drawdown_history
    }

    /// Get historical maximum drawdown
    pub fn historical_max(&self) -> f64 {
        self.historical_max_drawdown
    }

    /// Get average drawdown
    pub fn average_drawdown(&self) -> f64 {
        if self.drawdown_history.is_empty() {
            return 0.0;
        }
        self.drawdown_history.iter().sum::<f64>() / self.drawdown_history.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_limits_creation() {
        let limits = RiskLimits::new(0.15, 0.20, 0.10, 5);
        assert_eq!(limits.max_drawdown, 0.15);
        assert_eq!(limits.max_portfolio_risk, 0.20);
        assert_eq!(limits.max_position_size, 0.10);
        assert_eq!(limits.max_positions, 5);
    }

    #[test]
    fn test_check_drawdown() {
        let mut limits = RiskLimits::new(0.15, 0.20, 0.10, 5);

        assert!(limits.check_drawdown(0.10)); // Within limit
        assert!(!limits.check_drawdown(0.20)); // Exceeds limit
        assert_eq!(limits.total_violations(), 1);
    }

    #[test]
    fn test_check_portfolio_risk() {
        let mut limits = RiskLimits::new(0.15, 0.20, 0.10, 5);

        assert!(limits.check_portfolio_risk(0.15)); // Within limit
        assert!(!limits.check_portfolio_risk(0.25)); // Exceeds limit
    }

    #[test]
    fn test_check_position_size() {
        let mut limits = RiskLimits::new(0.15, 0.20, 0.10, 5);

        assert!(limits.check_position_size(0.08)); // Within limit
        assert!(!limits.check_position_size(0.12)); // Exceeds limit
    }

    #[test]
    fn test_check_num_positions() {
        let mut limits = RiskLimits::new(0.15, 0.20, 0.10, 5);

        assert!(limits.check_num_positions(4)); // Within limit
        assert!(!limits.check_num_positions(6)); // Exceeds limit
    }

    #[test]
    fn test_trading_stopped() {
        let mut limits = RiskLimits::new(0.15, 0.20, 0.10, 5);

        assert!(!limits.is_trading_stopped());
        limits.check_drawdown(0.20); // Trigger violation
        assert!(limits.is_trading_stopped());

        limits.resume_trading();
        assert!(!limits.is_trading_stopped());
    }

    #[test]
    fn test_violation_count() {
        let mut limits = RiskLimits::new(0.15, 0.20, 0.10, 5);

        limits.check_drawdown(0.20);
        limits.check_drawdown(0.22);
        limits.check_portfolio_risk(0.25);

        assert_eq!(limits.violation_count(ViolationType::MaxDrawdown), 2);
        assert_eq!(limits.violation_count(ViolationType::MaxPortfolioRisk), 1);
        assert_eq!(limits.total_violations(), 3);
    }

    #[test]
    fn test_drawdown_monitor_creation() {
        let monitor = DrawdownMonitor::new(0.15);
        assert_eq!(monitor.max_drawdown_limit, 0.15);
        assert!(!monitor.is_stopped());
    }

    #[test]
    fn test_drawdown_monitor_update() {
        let mut monitor = DrawdownMonitor::new(0.15);

        monitor.update(10000.0, 10000.0);
        assert_eq!(monitor.current_drawdown_pct(), 0.0);

        monitor.update(9000.0, 10000.0);
        assert_eq!(monitor.current_drawdown_pct(), 0.10); // 10% drawdown

        monitor.update(8500.0, 10000.0);
        assert_eq!(monitor.current_drawdown_pct(), 0.15); // 15% drawdown
    }

    #[test]
    fn test_drawdown_monitor_stop() {
        let mut monitor = DrawdownMonitor::new(0.15);

        monitor.update(10000.0, 10000.0);
        assert!(!monitor.is_stopped());

        monitor.update(8000.0, 10000.0); // 20% drawdown
        assert!(monitor.is_stopped());
    }

    #[test]
    fn test_drawdown_scaling_factor() {
        let mut monitor = DrawdownMonitor::new(0.20);

        monitor.update(10000.0, 10000.0);
        assert_eq!(monitor.get_scaling_factor(), 1.0); // No drawdown

        monitor.update(9000.0, 10000.0); // 10% drawdown (50% of limit)
        assert!((monitor.get_scaling_factor() - 0.75).abs() < 0.01);

        monitor.update(8000.0, 10000.0); // 20% drawdown (at limit)
        assert_eq!(monitor.get_scaling_factor(), 0.0); // Should stop
    }

    #[test]
    fn test_violation_severity() {
        let violation = Violation::new(ViolationType::MaxDrawdown, 0.20, 0.15, "Test".to_string());

        assert!((violation.severity() - 0.333).abs() < 0.01); // 33% over limit
        assert!(violation.is_critical());
    }

    #[test]
    fn test_violation_display() {
        assert_eq!(ViolationType::MaxDrawdown.to_string(), "MAX_DRAWDOWN");
        assert_eq!(ViolationType::MaxPositions.to_string(), "MAX_POSITIONS");
    }
}
