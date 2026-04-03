//! Kelly Criterion for optimal position sizing.
//!
//! The Kelly Criterion is a formula for position sizing that maximizes
//! long-term growth of capital by balancing risk and reward.
//!
//! # Formula
//!
//! Kelly% = (p * b - q) / b
//!
//! Where:
//! - p = probability of winning
//! - q = probability of losing (1 - p)
//! - b = ratio of win amount to loss amount
//!
//! # Safety
//!
//! Full Kelly can be very aggressive. Common practice is to use fractional Kelly
//! (e.g., half-Kelly = 0.5) to reduce volatility.

use serde::{Deserialize, Serialize};

/// Optimal Kelly fraction calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalFraction {
    /// Full Kelly fraction
    pub full_kelly: f64,

    /// Adjusted Kelly fraction (with safety multiplier)
    pub adjusted_kelly: f64,

    /// Recommended position size as fraction of capital
    pub position_size: f64,

    /// Expected growth rate
    pub expected_growth: f64,

    /// Risk of ruin (probability of losing all capital)
    pub risk_of_ruin: f64,
}

/// Kelly Criterion calculator
pub struct KellyCalculator {
    /// Fraction of Kelly to use (e.g., 0.5 for half-Kelly)
    kelly_fraction: f64,
}

impl KellyCalculator {
    /// Create a new Kelly calculator
    pub fn new(kelly_fraction: f64) -> Self {
        Self {
            kelly_fraction: kelly_fraction.clamp(0.0, 1.0),
        }
    }

    /// Calculate Kelly fraction from win rate and win/loss ratio
    pub fn calculate_fraction(&self, win_rate: f64, win_loss_ratio: f64) -> f64 {
        if win_rate <= 0.0 || win_rate >= 1.0 {
            return 0.0;
        }

        let p = win_rate;
        let q = 1.0 - p;
        let b = win_loss_ratio;

        // Kelly formula: (p * b - q) / b
        let full_kelly = ((p * b) - q) / b;

        // Apply safety fraction and clamp
        let adjusted = (full_kelly * self.kelly_fraction).clamp(0.0, 0.25);

        adjusted
    }

    /// Calculate position size from capital and signal confidence
    pub fn calculate_size(&self, capital: f64, price: f64, confidence: f64) -> f64 {
        // Use confidence as proxy for win rate
        // Assume 1.5:1 win/loss ratio as default
        let kelly_fraction = self.calculate_fraction(confidence, 1.5);

        // Calculate position value
        let position_value = capital * kelly_fraction;

        // Convert to shares/contracts
        if price > 0.0 {
            position_value / price
        } else {
            0.0
        }
    }

    /// Get the Kelly fraction multiplier
    pub fn kelly_fraction(&self) -> f64 {
        self.kelly_fraction
    }

    /// Set the Kelly fraction multiplier
    pub fn set_kelly_fraction(&mut self, fraction: f64) {
        self.kelly_fraction = fraction.clamp(0.0, 1.0);
    }
}

impl Default for KellyCalculator {
    fn default() -> Self {
        Self::new(0.5) // Half-Kelly by default
    }
}

/// Kelly optimizer for finding optimal fractions
pub struct KellyOptimizer {
    /// Historical win rate
    win_rate: f64,

    /// Historical win/loss ratio
    win_loss_ratio: f64,

    /// Number of simulations for optimization
    num_simulations: usize,
}

impl KellyOptimizer {
    /// Create a new Kelly optimizer
    pub fn new(win_rate: f64, win_loss_ratio: f64) -> Self {
        Self {
            win_rate: win_rate.clamp(0.0, 1.0),
            win_loss_ratio: win_loss_ratio.max(0.0),
            num_simulations: 10000,
        }
    }

    /// Calculate optimal Kelly fraction
    pub fn optimize(&self) -> OptimalFraction {
        let p = self.win_rate;
        let q = 1.0 - p;
        let b = self.win_loss_ratio;

        // Full Kelly calculation
        let full_kelly = if b > 0.0 { ((p * b) - q) / b } else { 0.0 };

        // Clamp to reasonable range
        let full_kelly = full_kelly.clamp(0.0, 1.0);

        // Half-Kelly is often recommended for safety
        let adjusted_kelly = (full_kelly * 0.5).clamp(0.0, 0.25);

        // Calculate expected growth rate
        let expected_growth = if full_kelly > 0.0 {
            p * (1.0 + full_kelly * b).ln() + q * (1.0 - full_kelly).ln()
        } else {
            0.0
        };

        // Estimate risk of ruin (simplified)
        let risk_of_ruin = self.estimate_risk_of_ruin(full_kelly);

        OptimalFraction {
            full_kelly,
            adjusted_kelly,
            position_size: adjusted_kelly,
            expected_growth,
            risk_of_ruin,
        }
    }

    /// Find optimal fraction given drawdown tolerance
    pub fn optimize_for_drawdown(&self, max_drawdown: f64) -> OptimalFraction {
        let mut optimal = self.optimize();

        // Reduce Kelly fraction if it would cause excessive drawdown
        // Rule of thumb: max drawdown ≈ 1 / (2 * Kelly^2)
        let kelly_for_dd = (1.0 / (2.0 * max_drawdown)).sqrt();

        if kelly_for_dd < optimal.adjusted_kelly {
            optimal.adjusted_kelly = kelly_for_dd;
            optimal.position_size = kelly_for_dd;
        }

        optimal
    }

    /// Estimate risk of ruin using simplified formula
    fn estimate_risk_of_ruin(&self, kelly_fraction: f64) -> f64 {
        if kelly_fraction <= 0.0 || self.win_rate >= 1.0 {
            return 0.0;
        }

        let p = self.win_rate;
        let q = 1.0 - p;
        let b = self.win_loss_ratio;

        // Simplified risk of ruin calculation
        // More aggressive sizing = higher risk
        let risk_factor = kelly_fraction / (p * b);
        let risk = (q / p).powf(risk_factor);

        risk.clamp(0.0, 1.0)
    }

    /// Simulate growth over multiple periods
    pub fn simulate_growth(&self, kelly_fraction: f64, periods: usize) -> Vec<f64> {
        let mut equity = vec![1.0]; // Start with 1.0 (100%)

        for _ in 0..periods {
            let current = equity.last().unwrap();

            // Simulate win or loss based on win rate
            // Simple pseudo-random number generation for simulation
            let random_val = (current * 12345.6789) % 1.0;
            let outcome = if random_val < self.win_rate {
                // Win
                current * (1.0 + kelly_fraction * self.win_loss_ratio)
            } else {
                // Loss
                current * (1.0 - kelly_fraction)
            };

            equity.push(outcome);
        }

        equity
    }

    /// Find optimal fraction through grid search
    pub fn grid_search(&self, min_fraction: f64, max_fraction: f64, steps: usize) -> f64 {
        let step_size = (max_fraction - min_fraction) / steps as f64;
        let mut best_fraction = 0.0;
        let mut best_growth = f64::NEG_INFINITY;

        for i in 0..=steps {
            let fraction = min_fraction + i as f64 * step_size;

            // Simulate multiple times and average
            let mut total_growth = 0.0;
            for _ in 0..100 {
                let equity = self.simulate_growth(fraction, 100);
                let final_equity = equity.last().unwrap();
                total_growth += final_equity.ln();
            }
            let avg_growth = total_growth / 100.0;

            if avg_growth > best_growth {
                best_growth = avg_growth;
                best_fraction = fraction;
            }
        }

        best_fraction
    }

    /// Set number of simulations
    pub fn set_simulations(&mut self, num: usize) {
        self.num_simulations = num;
    }

    /// Update historical statistics
    pub fn update_statistics(&mut self, win_rate: f64, win_loss_ratio: f64) {
        self.win_rate = win_rate.clamp(0.0, 1.0);
        self.win_loss_ratio = win_loss_ratio.max(0.0);
    }
}

/// Calculate Kelly fraction from trade history
pub fn calculate_from_history(
    winning_trades: usize,
    losing_trades: usize,
    avg_win: f64,
    avg_loss: f64,
) -> f64 {
    let total_trades = winning_trades + losing_trades;
    if total_trades == 0 || avg_loss == 0.0 {
        return 0.0;
    }

    let win_rate = winning_trades as f64 / total_trades as f64;
    let win_loss_ratio = avg_win / avg_loss;

    let p = win_rate;
    let q = 1.0 - p;
    let b = win_loss_ratio;

    let kelly = ((p * b) - q) / b;

    // Return half-Kelly for safety
    (kelly * 0.5).clamp(0.0, 0.25)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_calculator_creation() {
        let calc = KellyCalculator::new(0.5);
        assert_eq!(calc.kelly_fraction(), 0.5);
    }

    #[test]
    fn test_kelly_fraction_calculation() {
        let calc = KellyCalculator::new(1.0); // Full Kelly

        // 60% win rate, 1.5:1 win/loss ratio
        let fraction = calc.calculate_fraction(0.6, 1.5);

        // Kelly = (0.6 * 1.5 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.333...
        // But capped at 0.25
        assert_eq!(fraction, 0.25);
    }

    #[test]
    fn test_half_kelly() {
        let calc = KellyCalculator::new(0.5); // Half-Kelly

        let fraction = calc.calculate_fraction(0.6, 1.5);

        // Should be half of full Kelly
        assert!((fraction - 0.167).abs() < 0.01);
    }

    #[test]
    fn test_kelly_size_calculation() {
        let calc = KellyCalculator::new(0.5);

        let size = calc.calculate_size(10000.0, 50000.0, 0.6);

        // With 60% confidence (win rate), 1.5:1 ratio, half-Kelly
        // Kelly = ~0.167, so position = 10000 * 0.167 = 1670
        // Size = 1670 / 50000 = 0.0334
        assert!(size > 0.0);
        assert!(size < 0.1);
    }

    #[test]
    fn test_kelly_optimizer_creation() {
        let optimizer = KellyOptimizer::new(0.6, 1.5);
        assert_eq!(optimizer.win_rate, 0.6);
        assert_eq!(optimizer.win_loss_ratio, 1.5);
    }

    #[test]
    fn test_optimize() {
        let optimizer = KellyOptimizer::new(0.6, 1.5);
        let optimal = optimizer.optimize();

        assert!(optimal.full_kelly > 0.0);
        assert!(optimal.adjusted_kelly < optimal.full_kelly);
        assert!(optimal.adjusted_kelly <= 0.25); // Capped at 25%
        assert!(optimal.risk_of_ruin >= 0.0);
        assert!(optimal.risk_of_ruin <= 1.0);
    }

    #[test]
    fn test_optimize_for_drawdown() {
        let optimizer = KellyOptimizer::new(0.6, 1.5);
        let optimal = optimizer.optimize_for_drawdown(0.2); // 20% max drawdown

        assert!(optimal.adjusted_kelly > 0.0);
        assert!(optimal.adjusted_kelly <= 0.25);
    }

    #[test]
    fn test_no_edge() {
        let calc = KellyCalculator::new(1.0);

        // 50% win rate, 1:1 ratio = no edge
        let fraction = calc.calculate_fraction(0.5, 1.0);

        // Kelly should be 0 (no bet)
        assert_eq!(fraction, 0.0);
    }

    #[test]
    fn test_negative_edge() {
        let calc = KellyCalculator::new(1.0);

        // 40% win rate, 1:1 ratio = negative edge
        let fraction = calc.calculate_fraction(0.4, 1.0);

        // Kelly should be 0 (don't bet)
        assert_eq!(fraction, 0.0);
    }

    #[test]
    fn test_calculate_from_history() {
        let kelly = calculate_from_history(
            60,    // 60 wins
            40,    // 40 losses
            100.0, // avg win $100
            50.0,  // avg loss $50
        );

        // Win rate = 0.6, win/loss ratio = 2.0
        // Full Kelly = (0.6 * 2.0 - 0.4) / 2.0 = 0.4
        // Half Kelly = 0.2
        assert!((kelly - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_simulate_growth() {
        let optimizer = KellyOptimizer::new(0.6, 1.5);
        let equity = optimizer.simulate_growth(0.1, 100);

        assert_eq!(equity.len(), 101); // Initial + 100 periods
        assert_eq!(equity[0], 1.0); // Starts at 100%
    }

    #[test]
    fn test_kelly_fraction_bounds() {
        let mut calc = KellyCalculator::new(0.5);

        // Test clamping
        calc.set_kelly_fraction(2.0); // Too high
        assert_eq!(calc.kelly_fraction(), 1.0);

        calc.set_kelly_fraction(-0.5); // Too low
        assert_eq!(calc.kelly_fraction(), 0.0);
    }

    #[test]
    fn test_default_kelly_calculator() {
        let calc = KellyCalculator::default();
        assert_eq!(calc.kelly_fraction(), 0.5); // Half-Kelly default
    }
}
