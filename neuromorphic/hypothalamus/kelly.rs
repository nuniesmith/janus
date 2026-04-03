//! Kelly Criterion Position Sizing
//!
//! Optimal bet sizing to maximize long-term growth.

pub struct KellyCriterion {
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub fraction: f64, // Kelly fraction (usually 0.25 for half-Kelly)
}

impl KellyCriterion {
    pub fn new(win_rate: f64, avg_win: f64, avg_loss: f64) -> Self {
        Self {
            win_rate,
            avg_win,
            avg_loss,
            fraction: 0.25, // Default to quarter-Kelly (conservative)
        }
    }

    /// Calculate optimal fraction of capital to risk
    pub fn optimal_fraction(&self) -> f64 {
        if self.avg_loss == 0.0 {
            return 0.0;
        }

        let p = self.win_rate;
        let b = self.avg_win / self.avg_loss;

        // Kelly formula: f = (bp - q) / b
        // where q = 1 - p
        let kelly = (p * b - (1.0 - p)) / b;

        // Apply fraction (e.g., half-Kelly or quarter-Kelly)
        (kelly * self.fraction).clamp(0.0, 1.0)
    }

    /// Update statistics with new trade
    pub fn update(&mut self, won: bool, amount: f64) {
        // Simple exponential moving average
        let alpha = 0.1;

        if won {
            self.avg_win = self.avg_win * (1.0 - alpha) + amount * alpha;
            self.win_rate = self.win_rate * (1.0 - alpha) + alpha;
        } else {
            self.avg_loss = self.avg_loss * (1.0 - alpha) + amount * alpha;
            self.win_rate *= 1.0 - alpha;
        }
    }

    /// Calculate position size in dollars
    pub fn position_size(&self, capital: f64) -> f64 {
        capital * self.optimal_fraction()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_criterion() {
        let kelly = KellyCriterion::new(0.55, 100.0, 100.0);
        let fraction = kelly.optimal_fraction();

        assert!(fraction > 0.0);
        assert!(fraction <= 1.0);
    }

    #[test]
    fn test_position_size() {
        let kelly = KellyCriterion::new(0.6, 100.0, 100.0);
        let size = kelly.position_size(10000.0);

        assert!(size > 0.0);
        assert!(size <= 10000.0);
    }

    #[test]
    fn test_update() {
        let mut kelly = KellyCriterion::new(0.5, 100.0, 100.0);

        kelly.update(true, 150.0);
        assert!(kelly.avg_win > 100.0);

        kelly.update(false, 80.0);
        // Win rate should adjust
    }
}
