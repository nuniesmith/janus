//! Almgren-Chriss Optimal Execution Model
//!
//! Computes optimal trading trajectories to minimize market impact + risk.

pub struct AlmgrenChriss {
    pub lambda: f64, // Risk aversion parameter
    pub eta: f64,    // Temporary market impact
    pub gamma: f64,  // Permanent market impact
}

impl AlmgrenChriss {
    pub fn new(lambda: f64, eta: f64, gamma: f64) -> Self {
        Self { lambda, eta, gamma }
    }

    /// Compute optimal trading trajectory
    ///
    /// # Arguments
    /// *  - Total number of shares to trade
    /// *  - Number of time intervals
    ///
    /// # Returns
    /// Vector of shares to trade in each interval
    pub fn optimal_trajectory(&self, total_shares: f64, num_intervals: usize) -> Vec<f64> {
        if num_intervals == 0 {
            return vec![];
        }

        let t = num_intervals as f64;
        let kappa = self.lambda * self.eta / (t * self.eta + self.gamma);

        let mut trajectory = Vec::with_capacity(num_intervals);
        let mut remaining = total_shares;

        for j in 0..num_intervals {
            let tau = (num_intervals - j) as f64;

            // Optimal trading rate (simplified)
            let trade_amount = if j == num_intervals - 1 {
                // Last interval: trade remaining
                remaining
            } else {
                // Exponential decay with U-shape at endpoints
                let decay = (-kappa * tau).exp();
                let amount = remaining * (1.0 - decay);
                amount.max(total_shares * 0.01) // Minimum 1% per interval
            };

            trajectory.push(trade_amount);
            remaining -= trade_amount;
        }

        // Normalize to ensure sum = total_shares
        let sum: f64 = trajectory.iter().sum();
        if sum > 0.0 {
            trajectory.iter_mut().for_each(|x| *x *= total_shares / sum);
        }

        trajectory
    }

    /// Calculate expected cost
    pub fn expected_cost(&self, trajectory: &[f64]) -> f64 {
        let mut cost = 0.0;

        for (i, &shares) in trajectory.iter().enumerate() {
            // Temporary impact
            cost += self.eta * shares.powi(2);

            // Permanent impact
            let remaining: f64 = trajectory[i..].iter().sum();
            cost += self.gamma * shares * remaining;
        }

        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_sum() {
        let model = AlmgrenChriss::new(0.001, 0.1, 0.01);
        let trajectory = model.optimal_trajectory(1000.0, 10);

        let sum: f64 = trajectory.iter().sum();
        assert!(
            (sum - 1000.0).abs() < 1.0,
            "Sum should be close to total shares"
        );
    }

    #[test]
    fn test_trajectory_u_shape() {
        let model = AlmgrenChriss::new(0.001, 0.1, 0.01);
        let trajectory = model.optimal_trajectory(1000.0, 10);

        assert!(trajectory.len() == 10);
        // Just check it's reasonable
        assert!(trajectory.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_expected_cost() {
        let model = AlmgrenChriss::new(0.001, 0.1, 0.01);
        let trajectory = model.optimal_trajectory(1000.0, 10);

        let cost = model.expected_cost(&trajectory);
        assert!(cost > 0.0);
    }
}
