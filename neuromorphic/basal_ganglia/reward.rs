//! Reward Processing
//!
//! Computes rewards from trading outcomes for reinforcement learning.

pub struct RewardCalculator {
    profit_weight: f32,
    risk_penalty: f32,
    sharpe_weight: f32,
}

impl RewardCalculator {
    pub fn new(profit_weight: f32, risk_penalty: f32, sharpe_weight: f32) -> Self {
        Self {
            profit_weight,
            risk_penalty,
            sharpe_weight,
        }
    }

    /// Calculate reward from trading outcome
    pub fn calculate_reward(&self, pnl: f32, risk: f32, sharpe: f32) -> f32 {
        self.profit_weight * pnl - self.risk_penalty * risk + self.sharpe_weight * sharpe
    }

    /// Calculate reward from position change
    pub fn position_reward(&self, entry_price: f32, exit_price: f32, quantity: f32) -> f32 {
        // Simple reward based on profit/loss
        (exit_price - entry_price) * quantity
    }

    /// Calculate penalty for violating constraints
    pub fn constraint_penalty(&self, violations: &[ConstraintViolation]) -> f32 {
        -violations.iter().map(|v| v.severity).sum::<f32>() // Negative reward for violations
    }
}

#[derive(Debug)]
pub struct ConstraintViolation {
    pub constraint_name: String,
    pub severity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_calculation() {
        let calc = RewardCalculator::new(1.0, 0.5, 0.2);

        // Positive PnL, low risk
        let reward = calc.calculate_reward(100.0, 10.0, 1.5);
        assert!(reward > 0.0);

        // Negative PnL
        let reward = calc.calculate_reward(-50.0, 10.0, 0.5);
        assert!(reward < 0.0);
    }

    #[test]
    fn test_position_reward() {
        let calc = RewardCalculator::new(1.0, 0.5, 0.2);

        // Profitable trade
        let reward = calc.position_reward(100.0, 110.0, 10.0);
        assert_eq!(reward, 100.0);

        // Loss
        let reward = calc.position_reward(100.0, 95.0, 10.0);
        assert_eq!(reward, -50.0);
    }
}
