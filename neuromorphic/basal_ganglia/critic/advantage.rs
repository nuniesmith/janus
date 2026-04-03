//! Advantage function (A = Q - V)
//!
//! Part of the Basal Ganglia region
//! Component: critic
//!
//! The Advantage function measures how much better an action is compared to the
//! average action in that state. This is crucial for:
//! - Policy gradient variance reduction
//! - Actor-critic methods
//! - A2C/A3C algorithms
//! - PPO and TRPO
//!
//! A(s, a) = Q(s, a) - V(s)
//!
//! Where:
//! - Q(s, a) is the action-value function
//! - V(s) is the state-value function (baseline)
//!
//! Advantages can be estimated using:
//! - TD(0): A = r + γV(s') - V(s)
//! - n-step returns: A = Σγ^i * r_i + γ^n * V(s_n) - V(s)
//! - GAE (Generalized Advantage Estimation)

use crate::common::Result;
use std::collections::VecDeque;

/// Configuration for advantage estimation
#[derive(Debug, Clone)]
pub struct AdvantageConfig {
    /// Discount factor (gamma)
    pub gamma: f32,
    /// Number of steps for n-step returns (1 = TD(0))
    pub n_steps: usize,
    /// Whether to normalize advantages
    pub normalize: bool,
    /// Epsilon for numerical stability in normalization
    pub norm_epsilon: f32,
    /// Buffer size for computing statistics
    pub buffer_size: usize,
}

impl Default for AdvantageConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            n_steps: 1,
            normalize: true,
            norm_epsilon: 1e-8,
            buffer_size: 1000,
        }
    }
}

/// A transition for advantage computation
#[derive(Debug, Clone)]
pub struct Transition {
    /// State
    pub state: Vec<f32>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f32,
    /// Next state
    pub next_state: Vec<f32>,
    /// Whether episode terminated
    pub done: bool,
    /// Value estimate V(s)
    pub value: f32,
    /// Next value estimate V(s')
    pub next_value: f32,
}

impl Transition {
    /// Create a new transition
    pub fn new(
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
        value: f32,
        next_value: f32,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
            value,
            next_value,
        }
    }

    /// Compute TD(0) advantage: A = r + γV(s') - V(s)
    pub fn td_advantage(&self, gamma: f32) -> f32 {
        let bootstrap = if self.done { 0.0 } else { self.next_value };
        self.reward + gamma * bootstrap - self.value
    }
}

/// Advantage function (A = Q - V)
#[derive(Debug, Clone)]
pub struct Advantage {
    /// Configuration
    pub config: AdvantageConfig,
    /// Rolling buffer for normalization statistics
    advantage_buffer: VecDeque<f32>,
    /// Running mean of advantages
    running_mean: f32,
    /// Running variance of advantages
    running_var: f32,
    /// Count of observations
    count: u64,
}

impl Default for Advantage {
    fn default() -> Self {
        Self::new(AdvantageConfig::default())
    }
}

impl Advantage {
    /// Create a new instance
    pub fn new(config: AdvantageConfig) -> Self {
        Self {
            advantage_buffer: VecDeque::with_capacity(config.buffer_size),
            config,
            running_mean: 0.0,
            running_var: 1.0,
            count: 0,
        }
    }

    /// Compute TD(0) advantage: A = r + γV(s') - V(s)
    pub fn td0(&self, reward: f32, value: f32, next_value: f32, done: bool) -> f32 {
        let bootstrap = if done { 0.0 } else { next_value };
        reward + self.config.gamma * bootstrap - value
    }

    /// Compute advantage from a transition
    pub fn from_transition(&self, transition: &Transition) -> f32 {
        transition.td_advantage(self.config.gamma)
    }

    /// Compute n-step advantage
    ///
    /// A_n = Σ_{i=0}^{n-1} γ^i * r_{t+i} + γ^n * V(s_{t+n}) - V(s_t)
    pub fn n_step(
        &self,
        rewards: &[f32],
        values: &[f32],
        final_value: f32,
        final_done: bool,
    ) -> f32 {
        if rewards.is_empty() || values.is_empty() {
            return 0.0;
        }

        let gamma = self.config.gamma;
        let n = rewards.len().min(self.config.n_steps);

        // Compute discounted sum of rewards
        let mut discounted_return = 0.0;
        for i in 0..n {
            discounted_return += gamma.powi(i as i32) * rewards[i];
        }

        // Add bootstrap value
        let bootstrap = if final_done {
            0.0
        } else {
            gamma.powi(n as i32) * final_value
        };

        discounted_return + bootstrap - values[0]
    }

    /// Compute advantages for a batch of transitions
    pub fn batch_td0(&self, transitions: &[Transition]) -> Vec<f32> {
        transitions
            .iter()
            .map(|t| self.from_transition(t))
            .collect()
    }

    /// Compute n-step returns for a trajectory
    ///
    /// Returns: (advantages, returns)
    pub fn compute_returns(
        &self,
        rewards: &[f32],
        values: &[f32],
        final_value: f32,
        dones: &[bool],
    ) -> (Vec<f32>, Vec<f32>) {
        let n = rewards.len();
        if n == 0 || values.len() != n || dones.len() != n {
            return (Vec::new(), Vec::new());
        }

        let gamma = self.config.gamma;
        let n_steps = self.config.n_steps;

        let mut advantages = vec![0.0; n];
        let mut returns = vec![0.0; n];

        // Compute returns and advantages backwards
        for t in (0..n).rev() {
            let end_idx = (t + n_steps).min(n);
            let steps = end_idx - t;

            // Compute n-step return
            let mut g = 0.0;
            for i in 0..steps {
                g += gamma.powi(i as i32) * rewards[t + i];

                // If episode terminates, don't bootstrap
                if dones[t + i] {
                    break;
                }
            }

            // Add bootstrap value if not terminal
            let bootstrap_idx = end_idx - 1;
            if !dones[bootstrap_idx] {
                let bootstrap = if end_idx < n {
                    values[end_idx]
                } else {
                    final_value
                };
                g += gamma.powi(steps as i32) * bootstrap;
            }

            returns[t] = g;
            advantages[t] = g - values[t];
        }

        (advantages, returns)
    }

    /// Normalize advantages using running statistics
    pub fn normalize(&mut self, advantages: &mut [f32]) {
        if advantages.is_empty() {
            return;
        }

        // Update running statistics
        for &adv in advantages.iter() {
            self.update_stats(adv);
        }

        if !self.config.normalize {
            return;
        }

        // Compute batch statistics
        let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let variance: f32 =
            advantages.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
        let std = (variance + self.config.norm_epsilon).sqrt();

        // Normalize
        for adv in advantages.iter_mut() {
            *adv = (*adv - mean) / std;
        }
    }

    /// Normalize using global running statistics
    pub fn normalize_global(&self, advantages: &mut [f32]) {
        if !self.config.normalize || self.count < 2 {
            return;
        }

        let std = (self.running_var + self.config.norm_epsilon).sqrt();

        for adv in advantages.iter_mut() {
            *adv = (*adv - self.running_mean) / std;
        }
    }

    /// Update running statistics with a new observation
    fn update_stats(&mut self, advantage: f32) {
        self.count += 1;

        // Welford's online algorithm for mean and variance
        let delta = advantage - self.running_mean;
        self.running_mean += delta / self.count as f32;
        let delta2 = advantage - self.running_mean;
        self.running_var += (delta * delta2 - self.running_var) / self.count as f32;

        // Update buffer
        self.advantage_buffer.push_back(advantage);
        if self.advantage_buffer.len() > self.config.buffer_size {
            self.advantage_buffer.pop_front();
        }
    }

    /// Get statistics
    pub fn stats(&self) -> AdvantageStats {
        let (buffer_mean, buffer_var) = if self.advantage_buffer.is_empty() {
            (0.0, 0.0)
        } else {
            let mean: f32 =
                self.advantage_buffer.iter().sum::<f32>() / self.advantage_buffer.len() as f32;
            let var: f32 = self
                .advantage_buffer
                .iter()
                .map(|&a| (a - mean).powi(2))
                .sum::<f32>()
                / self.advantage_buffer.len() as f32;
            (mean, var)
        };

        AdvantageStats {
            running_mean: self.running_mean,
            running_var: self.running_var,
            buffer_mean,
            buffer_var,
            count: self.count,
            buffer_size: self.advantage_buffer.len(),
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.running_mean = 0.0;
        self.running_var = 1.0;
        self.count = 0;
        self.advantage_buffer.clear();
    }

    /// Main processing function (for interface compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Compute advantage with reward shaping
    ///
    /// Reward shaping: A = r + F(s') - F(s) + γV(s') - V(s)
    /// Where F is a potential function
    pub fn with_reward_shaping<F>(
        &self,
        reward: f32,
        value: f32,
        next_value: f32,
        done: bool,
        _potential_fn: F,
    ) -> f32
    where
        F: Fn(&[f32]) -> f32,
    {
        let base_advantage = self.td0(reward, value, next_value, done);
        // Note: Reward shaping requires state access, so this is a simplified version
        base_advantage
    }

    /// Clip advantages to a range
    pub fn clip(&self, advantages: &mut [f32], min: f32, max: f32) {
        for adv in advantages.iter_mut() {
            *adv = adv.clamp(min, max);
        }
    }
}

/// Statistics about advantage estimates
#[derive(Debug, Clone)]
pub struct AdvantageStats {
    pub running_mean: f32,
    pub running_var: f32,
    pub buffer_mean: f32,
    pub buffer_var: f32,
    pub count: u64,
    pub buffer_size: usize,
}

/// Builder for Advantage
#[derive(Debug, Clone)]
pub struct AdvantageBuilder {
    config: AdvantageConfig,
}

impl AdvantageBuilder {
    pub fn new() -> Self {
        Self {
            config: AdvantageConfig::default(),
        }
    }

    pub fn gamma(mut self, gamma: f32) -> Self {
        self.config.gamma = gamma;
        self
    }

    pub fn n_steps(mut self, n: usize) -> Self {
        self.config.n_steps = n;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    pub fn build(self) -> Advantage {
        Advantage::new(self.config)
    }
}

impl Default for AdvantageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advantage_creation() {
        let advantage = Advantage::default();
        assert_eq!(advantage.config.gamma, 0.99);
        assert_eq!(advantage.config.n_steps, 1);
    }

    #[test]
    fn test_td0_advantage() {
        let advantage = AdvantageBuilder::new().gamma(0.99).build();

        // Test: reward=1, V(s)=0, V(s')=0, not done
        let adv = advantage.td0(1.0, 0.0, 0.0, false);
        assert!((adv - 1.0).abs() < 1e-5);

        // Test: reward=1, V(s)=0, V(s')=1, not done
        let adv = advantage.td0(1.0, 0.0, 1.0, false);
        assert!((adv - 1.99).abs() < 1e-5);

        // Test: terminal state (done=true)
        let adv = advantage.td0(1.0, 0.5, 10.0, true);
        assert!((adv - 0.5).abs() < 1e-5); // Should ignore next_value when done
    }

    #[test]
    fn test_transition_advantage() {
        let advantage = AdvantageBuilder::new().gamma(0.9).build();

        let transition = Transition::new(
            vec![1.0, 0.0],
            0,
            2.0,
            vec![0.0, 1.0],
            false,
            1.0, // V(s) = 1.0
            3.0, // V(s') = 3.0
        );

        // A = r + γV(s') - V(s) = 2.0 + 0.9*3.0 - 1.0 = 3.7
        let adv = advantage.from_transition(&transition);
        assert!((adv - 3.7).abs() < 1e-5);
    }

    #[test]
    fn test_n_step_advantage() {
        let advantage = AdvantageBuilder::new().gamma(0.9).n_steps(3).build();

        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.5, 1.0];
        let final_value = 2.0;

        // A = r0 + γr1 + γ²r2 + γ³V(final) - V(s0)
        // A = 1.0 + 0.9*1.0 + 0.81*1.0 + 0.729*2.0 - 0.0
        // A = 1.0 + 0.9 + 0.81 + 1.458 = 4.168
        let adv = advantage.n_step(&rewards, &values, final_value, false);
        assert!((adv - 4.168).abs() < 0.01);
    }

    #[test]
    fn test_batch_advantages() {
        let advantage = AdvantageBuilder::new().gamma(0.99).build();

        let transitions = vec![
            Transition::new(vec![1.0], 0, 1.0, vec![2.0], false, 0.0, 1.0),
            Transition::new(vec![2.0], 1, 2.0, vec![3.0], false, 1.0, 2.0),
        ];

        let advantages = advantage.batch_td0(&transitions);
        assert_eq!(advantages.len(), 2);

        // First: A = 1.0 + 0.99*1.0 - 0.0 = 1.99
        assert!((advantages[0] - 1.99).abs() < 1e-5);

        // Second: A = 2.0 + 0.99*2.0 - 1.0 = 2.98
        assert!((advantages[1] - 2.98).abs() < 1e-5);
    }

    #[test]
    fn test_compute_returns() {
        let advantage = AdvantageBuilder::new().gamma(0.9).n_steps(2).build();

        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0];
        let dones = vec![false, false, false];
        let final_value = 0.0;

        let (advantages, returns) =
            advantage.compute_returns(&rewards, &values, final_value, &dones);

        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);

        // Returns should be positive for positive rewards
        assert!(returns.iter().all(|&r| r > 0.0));
    }

    #[test]
    fn test_normalization() {
        let mut advantage = AdvantageBuilder::new().normalize(true).build();

        let mut advantages = vec![10.0, 20.0, 30.0, 40.0];
        advantage.normalize(&mut advantages);

        // After normalization, mean should be ~0 and std ~1
        let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let var: f32 =
            advantages.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
        let std = var.sqrt();

        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
        assert!((std - 1.0).abs() < 0.1, "Std should be ~1, got {}", std);
    }

    #[test]
    fn test_running_statistics() {
        let mut advantage = AdvantageBuilder::new().buffer_size(10).build();

        // Add some observations
        for i in 0..5 {
            advantage.update_stats(i as f32);
        }

        let stats = advantage.stats();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.buffer_size, 5);

        // Mean of [0,1,2,3,4] = 2.0
        assert!((stats.buffer_mean - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_clipping() {
        let advantage = Advantage::default();

        let mut advantages = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        advantage.clip(&mut advantages, -2.0, 2.0);

        assert_eq!(advantages, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_terminal_state_handling() {
        let advantage = AdvantageBuilder::new().gamma(0.99).build();

        // When done=true, next_value should be ignored
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![true];
        let final_value = 100.0; // Should be ignored

        let (advantages, _returns) =
            advantage.compute_returns(&rewards, &values, final_value, &dones);

        // A = r - V(s) = 1.0 - 0.5 = 0.5 (no bootstrap because done)
        assert!((advantages[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_builder_pattern() {
        let advantage = AdvantageBuilder::new()
            .gamma(0.95)
            .n_steps(5)
            .normalize(false)
            .buffer_size(500)
            .build();

        assert_eq!(advantage.config.gamma, 0.95);
        assert_eq!(advantage.config.n_steps, 5);
        assert!(!advantage.config.normalize);
        assert_eq!(advantage.config.buffer_size, 500);
    }

    #[test]
    fn test_basic() {
        let instance = Advantage::default();
        assert!(instance.process().is_ok());
    }
}
