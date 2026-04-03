//! Portfolio risk monitoring and reporting.
//!
//! This module provides real-time monitoring of portfolio risk metrics
//! and generates comprehensive risk reports.

use crate::backtest::Position;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Portfolio risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioRisk {
    /// Total portfolio value at risk
    pub portfolio_var: f64,

    /// Total exposure (sum of all position values)
    pub total_exposure: f64,

    /// Net exposure (long - short)
    pub net_exposure: f64,

    /// Gross exposure (long + short)
    pub gross_exposure: f64,

    /// Number of open positions
    pub num_positions: usize,

    /// Exposure by asset
    pub asset_exposures: HashMap<String, f64>,

    /// Largest position exposure
    pub max_position_exposure: f64,

    /// Average position size
    pub avg_position_size: f64,

    /// Portfolio concentration (Herfindahl index)
    pub concentration: f64,
}

impl Default for PortfolioRisk {
    fn default() -> Self {
        Self {
            portfolio_var: 0.0,
            total_exposure: 0.0,
            net_exposure: 0.0,
            gross_exposure: 0.0,
            num_positions: 0,
            asset_exposures: HashMap::new(),
            max_position_exposure: 0.0,
            avg_position_size: 0.0,
            concentration: 0.0,
        }
    }
}

impl PortfolioRisk {
    /// Calculate portfolio concentration (0 = diversified, 1 = concentrated)
    pub fn calculate_concentration(&self) -> f64 {
        if self.asset_exposures.is_empty() {
            return 0.0;
        }

        let total = self.total_exposure;
        if total == 0.0 {
            return 0.0;
        }

        // Herfindahl-Hirschman Index
        let hhi: f64 = self
            .asset_exposures
            .values()
            .map(|&exposure| {
                let share = exposure / total;
                share * share
            })
            .sum();

        hhi
    }

    /// Check if portfolio is well-diversified (HHI < 0.15)
    pub fn is_diversified(&self) -> bool {
        self.concentration < 0.15
    }

    /// Get exposure as percentage of total
    pub fn exposure_percentage(&self, capital: f64) -> f64 {
        if capital == 0.0 {
            return 0.0;
        }
        (self.total_exposure / capital) * 100.0
    }
}

/// Risk report with detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskReport {
    /// Timestamp of report
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Portfolio risk metrics
    pub portfolio_risk: PortfolioRisk,

    /// Current capital
    pub capital: f64,

    /// Portfolio risk as percentage of capital
    pub risk_percentage: f64,

    /// Warnings
    pub warnings: Vec<String>,

    /// Risk level (Low, Medium, High, Critical)
    pub risk_level: RiskLevel,
}

impl RiskReport {
    /// Create a new risk report
    pub fn new(portfolio_risk: PortfolioRisk, capital: f64) -> Self {
        let risk_percentage = if capital > 0.0 {
            (portfolio_risk.total_exposure / capital) * 100.0
        } else {
            0.0
        };

        let risk_level = Self::calculate_risk_level(risk_percentage, &portfolio_risk);
        let warnings = Self::generate_warnings(&portfolio_risk, capital);

        Self {
            timestamp: chrono::Utc::now(),
            portfolio_risk,
            capital,
            risk_percentage,
            warnings,
            risk_level,
        }
    }

    /// Calculate risk level
    fn calculate_risk_level(risk_percentage: f64, portfolio: &PortfolioRisk) -> RiskLevel {
        // Check multiple factors
        let exposure_high = risk_percentage > 50.0;
        let concentrated = !portfolio.is_diversified();
        let many_positions = portfolio.num_positions > 10;

        if exposure_high && concentrated {
            RiskLevel::Critical
        } else if exposure_high || (concentrated && many_positions) {
            RiskLevel::High
        } else if risk_percentage > 25.0 || concentrated {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Generate warnings based on risk metrics
    fn generate_warnings(portfolio: &PortfolioRisk, capital: f64) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check exposure
        let exposure_pct = if capital > 0.0 {
            (portfolio.total_exposure / capital) * 100.0
        } else {
            0.0
        };

        if exposure_pct > 80.0 {
            warnings.push(format!(
                "High portfolio exposure: {:.1}% of capital",
                exposure_pct
            ));
        }

        // Check concentration
        if !portfolio.is_diversified() {
            warnings.push(format!(
                "Portfolio is concentrated (HHI: {:.3})",
                portfolio.concentration
            ));
        }

        // Check max position size
        let max_position_pct = if capital > 0.0 {
            (portfolio.max_position_exposure / capital) * 100.0
        } else {
            0.0
        };

        if max_position_pct > 20.0 {
            warnings.push(format!(
                "Large single position: {:.1}% of capital",
                max_position_pct
            ));
        }

        // Check number of positions
        if portfolio.num_positions > 15 {
            warnings.push(format!(
                "Many open positions: {} (may be hard to manage)",
                portfolio.num_positions
            ));
        }

        warnings
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Risk Report [{:?}]:\n\
             Capital: ${:.2}\n\
             Total Exposure: ${:.2} ({:.1}%)\n\
             Positions: {}\n\
             Concentration: {:.3}\n\
             Warnings: {}",
            self.risk_level,
            self.capital,
            self.portfolio_risk.total_exposure,
            self.risk_percentage,
            self.portfolio_risk.num_positions,
            self.portfolio_risk.concentration,
            self.warnings.len()
        )
    }
}

/// Risk level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk - well within limits
    Low,
    /// Medium risk - approaching some limits
    Medium,
    /// High risk - near or at limits
    High,
    /// Critical risk - limits exceeded
    Critical,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "LOW"),
            RiskLevel::Medium => write!(f, "MEDIUM"),
            RiskLevel::High => write!(f, "HIGH"),
            RiskLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Portfolio risk monitor
pub struct RiskMonitor {
    /// Maximum portfolio risk allowed
    #[allow(dead_code)]
    max_portfolio_risk: f64,

    /// Maximum number of positions
    max_positions: usize,

    /// Current positions (for tracking)
    current_positions: HashMap<String, Position>,

    /// Risk history
    risk_history: Vec<PortfolioRisk>,
}

impl RiskMonitor {
    /// Create a new risk monitor
    pub fn new(max_portfolio_risk: f64, max_positions: usize) -> Self {
        Self {
            max_portfolio_risk,
            max_positions,
            current_positions: HashMap::new(),
            risk_history: Vec::new(),
        }
    }

    /// Update positions
    pub fn update_positions(&mut self, positions: &HashMap<String, Position>) {
        self.current_positions = positions.clone();
        let risk = self.calculate_current_risk();
        self.risk_history.push(risk);
    }

    /// Calculate current portfolio risk
    pub fn calculate_current_risk(&self) -> PortfolioRisk {
        let mut total_exposure = 0.0;
        let mut net_exposure = 0.0;
        let mut gross_exposure = 0.0;
        let mut asset_exposures = HashMap::new();
        let mut max_position_exposure = 0.0;

        for (asset, position) in &self.current_positions {
            let position_value = position.entry_value();

            total_exposure += position_value;
            asset_exposures.insert(asset.clone(), position_value);

            // Track long/short exposure
            match position.side {
                crate::signals::SignalType::Buy => {
                    net_exposure += position_value;
                    gross_exposure += position_value;
                }
                crate::signals::SignalType::Sell => {
                    net_exposure -= position_value;
                    gross_exposure += position_value;
                }
                _ => {}
            }

            if position_value > max_position_exposure {
                max_position_exposure = position_value;
            }
        }

        let num_positions = self.current_positions.len();
        let avg_position_size = if num_positions > 0 {
            total_exposure / num_positions as f64
        } else {
            0.0
        };

        let mut risk = PortfolioRisk {
            portfolio_var: total_exposure * 0.02, // Simplified VAR estimate
            total_exposure,
            net_exposure,
            gross_exposure,
            num_positions,
            asset_exposures,
            max_position_exposure,
            avg_position_size,
            concentration: 0.0,
        };

        risk.concentration = risk.calculate_concentration();

        risk
    }

    /// Check if risk limits are exceeded
    pub fn is_limit_exceeded(&self) -> bool {
        let risk = self.calculate_current_risk();
        risk.num_positions > self.max_positions
    }

    /// Check if we can add another position
    pub fn can_add_position(&self) -> bool {
        self.current_positions.len() < self.max_positions
    }

    /// Generate risk report
    pub fn generate_report(&self) -> RiskReport {
        let risk = self.calculate_current_risk();
        let capital = self.estimate_capital();
        RiskReport::new(risk, capital)
    }

    /// Estimate total capital (simplified)
    fn estimate_capital(&self) -> f64 {
        // Sum of all position values as proxy for capital
        // In real usage, this would come from account balance
        self.current_positions
            .values()
            .map(|p| p.entry_value())
            .sum::<f64>()
            * 10.0 // Assume 10% deployment
    }

    /// Get risk history
    pub fn history(&self) -> &[PortfolioRisk] {
        &self.risk_history
    }

    /// Get current number of positions
    pub fn num_positions(&self) -> usize {
        self.current_positions.len()
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.risk_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::{SignalType, TradingSignal};

    fn create_test_position(asset: &str, side: SignalType, entry: f64, size: f64) -> Position {
        let signal = TradingSignal::new(side, 0.8, asset.to_string());
        Position::new(&signal, entry, size)
    }

    #[test]
    fn test_portfolio_risk_default() {
        let risk = PortfolioRisk::default();
        assert_eq!(risk.num_positions, 0);
        assert_eq!(risk.total_exposure, 0.0);
    }

    #[test]
    fn test_concentration_calculation() {
        let mut risk = PortfolioRisk::default();
        risk.total_exposure = 100.0;
        risk.asset_exposures.insert("A".to_string(), 50.0);
        risk.asset_exposures.insert("B".to_string(), 30.0);
        risk.asset_exposures.insert("C".to_string(), 20.0);

        let concentration = risk.calculate_concentration();
        // HHI = (0.5^2 + 0.3^2 + 0.2^2) = 0.25 + 0.09 + 0.04 = 0.38
        assert!((concentration - 0.38).abs() < 0.01);
    }

    #[test]
    fn test_diversification_check() {
        let mut risk = PortfolioRisk::default();

        // Well diversified
        risk.concentration = 0.10;
        assert!(risk.is_diversified());

        // Concentrated
        risk.concentration = 0.50;
        assert!(!risk.is_diversified());
    }

    #[test]
    fn test_risk_monitor_creation() {
        let monitor = RiskMonitor::new(0.20, 5);
        assert_eq!(monitor.max_portfolio_risk, 0.20);
        assert_eq!(monitor.max_positions, 5);
    }

    #[test]
    fn test_can_add_position() {
        let mut monitor = RiskMonitor::new(0.20, 2);
        assert!(monitor.can_add_position());

        let mut positions = HashMap::new();
        positions.insert(
            "BTC".to_string(),
            create_test_position("BTC", SignalType::Buy, 50000.0, 0.1),
        );
        positions.insert(
            "ETH".to_string(),
            create_test_position("ETH", SignalType::Buy, 3000.0, 1.0),
        );

        monitor.update_positions(&positions);
        assert!(!monitor.can_add_position()); // At limit
    }

    #[test]
    fn test_calculate_current_risk() {
        let monitor = RiskMonitor::new(0.20, 5);
        let mut positions = HashMap::new();

        positions.insert(
            "BTC".to_string(),
            create_test_position("BTC", SignalType::Buy, 50000.0, 0.1),
        );

        let mut temp_monitor = monitor;
        temp_monitor.update_positions(&positions);

        let risk = temp_monitor.calculate_current_risk();
        assert_eq!(risk.num_positions, 1);
        assert!(risk.total_exposure > 0.0);
    }

    #[test]
    fn test_risk_level() {
        assert_eq!(RiskLevel::Low.to_string(), "LOW");
        assert_eq!(RiskLevel::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_risk_report_creation() {
        let risk = PortfolioRisk::default();
        let report = RiskReport::new(risk, 10000.0);

        assert_eq!(report.capital, 10000.0);
        assert!(matches!(report.risk_level, RiskLevel::Low));
    }

    #[test]
    fn test_risk_report_warnings() {
        let mut risk = PortfolioRisk::default();
        risk.total_exposure = 9000.0; // 90% of capital
        risk.concentration = 0.5; // High concentration
        risk.num_positions = 20;

        let report = RiskReport::new(risk, 10000.0);
        assert!(!report.warnings.is_empty());
        assert!(report.warnings.len() >= 2); // Should have multiple warnings
    }
}
