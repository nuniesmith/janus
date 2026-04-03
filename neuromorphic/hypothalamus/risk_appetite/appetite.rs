//! Risk Appetite - Dynamic Risk Adjustment
//!
//! Adjusts risk-taking based on recent performance and drawdown.

pub struct RiskAppetite {
    pub current_drawdown: f64,
    pub recent_pnl: Vec<f64>,
    pub max_lookback: usize,
}

impl RiskAppetite {
    pub fn new(max_lookback: usize) -> Self {
        Self {
            current_drawdown: 0.0,
            recent_pnl: Vec::new(),
            max_lookback,
        }
    }

    /// Update with latest P&L
    pub fn update(&mut self, pnl: f64, portfolio_value: f64) {
        self.recent_pnl.push(pnl);

        if self.recent_pnl.len() > self.max_lookback {
            self.recent_pnl.remove(0);
        }

        // Update drawdown
        self.current_drawdown = self.calculate_drawdown(portfolio_value);
    }

    fn calculate_drawdown(&self, current_value: f64) -> f64 {
        if self.recent_pnl.is_empty() {
            return 0.0;
        }

        // Simple max drawdown calculation
        let mut peak = current_value;
        let mut max_dd: f64 = 0.0;

        let mut running_value = current_value;
        for pnl in self.recent_pnl.iter().rev() {
            running_value -= pnl;
            peak = peak.max(running_value);

            if peak > 0.0 {
                let dd = (peak - running_value) / peak;
                max_dd = max_dd.max(dd);
            }
        }

        max_dd
    }

    /// Calculate risk appetite multiplier
    /// Returns value in [0, 2] where:
    /// - < 1.0 = reduce risk
    /// - 1.0 = normal risk
    /// - > 1.0 = increase risk (carefully!)
    pub fn appetite_multiplier(&self) -> f64 {
        // Severe drawdown: cut risk dramatically
        if self.current_drawdown > 0.20 {
            return 0.25; // 25% of normal risk
        }

        // Moderate drawdown: reduce risk
        if self.current_drawdown > 0.10 {
            return 0.5; // 50% of normal risk
        }

        // Small drawdown: slight reduction
        if self.current_drawdown > 0.05 {
            return 0.75;
        }

        // Check recent performance
        if !self.recent_pnl.is_empty() {
            let avg_recent: f64 =
                self.recent_pnl.iter().sum::<f64>() / self.recent_pnl.len() as f64;

            // Winning streak: can slightly increase risk
            if avg_recent > 0.0 && self.current_drawdown < 0.02 {
                return 1.1; // 110% of normal risk (be careful!)
            }
        }

        1.0 // Normal risk
    }

    /// Check if should stop trading
    pub fn should_stop_trading(&self) -> bool {
        self.current_drawdown > 0.25 // Stop at 25% drawdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_appetite_normal() {
        let appetite = RiskAppetite::new(20);
        let multiplier = appetite.appetite_multiplier();

        assert_eq!(multiplier, 1.0);
    }

    #[test]
    fn test_risk_appetite_drawdown() {
        let mut appetite = RiskAppetite::new(20);
        appetite.current_drawdown = 0.15;

        let multiplier = appetite.appetite_multiplier();
        assert!(multiplier < 1.0);
    }

    #[test]
    fn test_should_stop() {
        let mut appetite = RiskAppetite::new(20);
        appetite.current_drawdown = 0.3;

        assert!(appetite.should_stop_trading());
    }
}
