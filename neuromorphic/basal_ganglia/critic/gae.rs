//! Generalized Advantage Estimation (GAE)
//!
//! Part of the Basal Ganglia region
//! Component: critic
//!
//! GAE provides a family of policy gradient estimators that trade off bias and variance.
//! This is essential for stable reinforcement learning in complex environments.
//!
//! GAE(γ, λ) = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
//!
//! Where:
//! - δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD residual
//! - γ is the discount factor
//! - λ is the GAE parameter controlling bias-variance tradeoff
//!
//! Special cases:
//! - λ = 0: GAE(γ, 0) = δ_t (one-step TD, high bias, low variance)
//! - λ = 1: GAE(γ, 1) = Σγ^l * r_{t+l} - V(s_t) (Monte Carlo, low bias, high variance)

use crate::common::Result;

/// Configuration for GAE
#[derive(Debug, Clone)]
pub struct GaeConfig {
    /// Discount factor (gamma)
    pub gamma: f32,
    /// GAE lambda parameter (bias-variance tradeoff)
    /// λ = 0: High bias, low variance (TD(0))
    /// λ = 1: Low bias, high variance (Monte Carlo)
    pub lambda: f32,
    /// Whether to normalize advantages
    pub normalize: bool,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Use proper handling of episode boundaries
    pub handle_dones: bool,
    /// Maximum trajectory length for computation
    pub max_trajectory_length: usize,
}

impl Default for GaeConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            normalize: true,
            epsilon: 1e-8,
            handle_dones: true,
            max_trajectory_length: 2048,
        }
    }
}

/// Trajectory data for GAE computation
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// States
    pub states: Vec<Vec<f32>>,
    /// Actions taken
    pub actions: Vec<usize>,
    /// Rewards received
    pub rewards: Vec<f32>,
    /// Value estimates V(s_t)
    pub values: Vec<f32>,
    /// Log probabilities of actions
    pub log_probs: Vec<f32>,
    /// Episode done flags
    pub dones: Vec<bool>,
    /// Final value estimate (for bootstrapping)
    pub final_value: f32,
}

impl Trajectory {
    /// Create a new empty trajectory
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            log_probs: Vec::new(),
            dones: Vec::new(),
            final_value: 0.0,
        }
    }

    /// Create trajectory with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            log_probs: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            final_value: 0.0,
        }
    }

    /// Add a transition to the trajectory
    pub fn add(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        value: f32,
        log_prob: f32,
        done: bool,
    ) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.values.push(value);
        self.log_probs.push(log_prob);
        self.dones.push(done);
    }

    /// Set the final value for bootstrapping
    pub fn set_final_value(&mut self, value: f32) {
        self.final_value = value;
    }

    /// Get trajectory length
    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    /// Check if trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.rewards.is_empty()
    }

    /// Clear the trajectory
    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.values.clear();
        self.log_probs.clear();
        self.dones.clear();
        self.final_value = 0.0;
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Generalized Advantage Estimation
#[derive(Debug, Clone)]
pub struct Gae {
    /// Configuration
    pub config: GaeConfig,
    /// Statistics tracking
    stats: GaeStats,
}

/// Statistics for GAE computation
#[derive(Debug, Clone, Default)]
pub struct GaeStats {
    /// Number of trajectories processed
    pub trajectories_processed: u64,
    /// Total timesteps processed
    pub timesteps_processed: u64,
    /// Running mean of advantages
    pub advantage_mean: f32,
    /// Running variance of advantages
    pub advantage_var: f32,
    /// Running mean of returns
    pub return_mean: f32,
    /// Running variance of returns
    pub return_var: f32,
    /// Count for running statistics
    pub count: u64,
}

impl Default for Gae {
    fn default() -> Self {
        Self::new(GaeConfig::default())
    }
}

impl Gae {
    /// Create a new GAE instance
    pub fn new(config: GaeConfig) -> Self {
        Self {
            config,
            stats: GaeStats::default(),
        }
    }

    /// Compute GAE advantages and returns for a trajectory
    ///
    /// Returns: (advantages, returns)
    pub fn compute(&mut self, trajectory: &Trajectory) -> (Vec<f32>, Vec<f32>) {
        let n = trajectory.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let gamma = self.config.gamma;
        let lambda = self.config.lambda;

        let mut advantages = vec![0.0; n];
        let mut returns = vec![0.0; n];

        // Compute GAE backwards
        let mut gae = 0.0;

        for t in (0..n).rev() {
            // Get next value (bootstrap or value of next state)
            let next_value = if t == n - 1 {
                if trajectory.dones[t] && self.config.handle_dones {
                    0.0
                } else {
                    trajectory.final_value
                }
            } else if trajectory.dones[t] && self.config.handle_dones {
                0.0
            } else {
                trajectory.values[t + 1]
            };

            // Compute TD residual: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            let delta = trajectory.rewards[t] + gamma * next_value - trajectory.values[t];

            // GAE: A_t = δ_t + (γλ)A_{t+1}
            // Reset GAE if episode ended
            if trajectory.dones[t] && self.config.handle_dones {
                gae = delta;
            } else {
                gae = delta + gamma * lambda * gae;
            }

            advantages[t] = gae;
            // Return = Advantage + Value
            returns[t] = gae + trajectory.values[t];
        }

        // Update statistics
        self.update_stats(&advantages, &returns);

        // Normalize advantages if configured
        if self.config.normalize && n > 1 {
            self.normalize_advantages(&mut advantages);
        }

        (advantages, returns)
    }

    /// Compute advantages using rewards and values arrays directly
    pub fn compute_from_arrays(
        &mut self,
        rewards: &[f32],
        values: &[f32],
        dones: &[bool],
        final_value: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = rewards.len();
        if n == 0 || values.len() != n || dones.len() != n {
            return (Vec::new(), Vec::new());
        }

        let gamma = self.config.gamma;
        let lambda = self.config.lambda;

        let mut advantages = vec![0.0; n];
        let mut returns = vec![0.0; n];

        let mut gae = 0.0;

        for t in (0..n).rev() {
            let next_value = if t == n - 1 {
                if dones[t] && self.config.handle_dones {
                    0.0
                } else {
                    final_value
                }
            } else if dones[t] && self.config.handle_dones {
                0.0
            } else {
                values[t + 1]
            };

            let delta = rewards[t] + gamma * next_value - values[t];

            if dones[t] && self.config.handle_dones {
                gae = delta;
            } else {
                gae = delta + gamma * lambda * gae;
            }

            advantages[t] = gae;
            returns[t] = gae + values[t];
        }

        self.update_stats(&advantages, &returns);

        if self.config.normalize && n > 1 {
            self.normalize_advantages(&mut advantages);
        }

        (advantages, returns)
    }

    /// Compute TD(λ) returns (alternative formulation)
    ///
    /// G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t^{(n)}
    ///
    /// Where G_t^{(n)} is the n-step return
    pub fn compute_td_lambda_returns(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[bool],
        final_value: f32,
    ) -> Vec<f32> {
        let n = rewards.len();
        if n == 0 {
            return Vec::new();
        }

        let gamma = self.config.gamma;
        let lambda = self.config.lambda;

        let mut returns = vec![0.0; n];

        // Compute backwards
        let mut g = if dones[n - 1] { 0.0 } else { final_value };

        for t in (0..n).rev() {
            if dones[t] && self.config.handle_dones && t < n - 1 {
                g = 0.0;
            }

            // G_t = r_t + γ * [(1-λ)V(s_{t+1}) + λG_{t+1}]
            let next_value = if t == n - 1 {
                if dones[t] { 0.0 } else { final_value }
            } else {
                values[t + 1]
            };

            g = rewards[t] + gamma * ((1.0 - lambda) * next_value + lambda * g);
            returns[t] = g;
        }

        returns
    }

    /// Normalize advantages
    fn normalize_advantages(&self, advantages: &mut [f32]) {
        if advantages.len() < 2 {
            return;
        }

        let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let variance: f32 =
            advantages.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
        let std = (variance + self.config.epsilon).sqrt();

        for adv in advantages.iter_mut() {
            *adv = (*adv - mean) / std;
        }
    }

    /// Update running statistics
    fn update_stats(&mut self, advantages: &[f32], returns: &[f32]) {
        self.stats.trajectories_processed += 1;
        self.stats.timesteps_processed += advantages.len() as u64;

        for (&adv, &ret) in advantages.iter().zip(returns.iter()) {
            self.stats.count += 1;
            let n = self.stats.count as f32;

            // Welford's algorithm for mean and variance
            let delta_adv = adv - self.stats.advantage_mean;
            self.stats.advantage_mean += delta_adv / n;
            let delta_adv2 = adv - self.stats.advantage_mean;
            self.stats.advantage_var += (delta_adv * delta_adv2 - self.stats.advantage_var) / n;

            let delta_ret = ret - self.stats.return_mean;
            self.stats.return_mean += delta_ret / n;
            let delta_ret2 = ret - self.stats.return_mean;
            self.stats.return_var += (delta_ret * delta_ret2 - self.stats.return_var) / n;
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &GaeStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = GaeStats::default();
    }

    /// Compute explained variance: 1 - Var(returns - values) / Var(returns)
    pub fn explained_variance(&self, values: &[f32], returns: &[f32]) -> f32 {
        if values.len() != returns.len() || values.is_empty() {
            return 0.0;
        }

        let return_mean: f32 = returns.iter().sum::<f32>() / returns.len() as f32;
        let return_var: f32 = returns
            .iter()
            .map(|&r| (r - return_mean).powi(2))
            .sum::<f32>()
            / returns.len() as f32;

        if return_var < self.config.epsilon {
            return 1.0; // Perfect prediction when returns are constant
        }

        let residuals: Vec<f32> = returns
            .iter()
            .zip(values.iter())
            .map(|(&r, &v)| r - v)
            .collect();
        let residual_mean: f32 = residuals.iter().sum::<f32>() / residuals.len() as f32;
        let residual_var: f32 = residuals
            .iter()
            .map(|&r| (r - residual_mean).powi(2))
            .sum::<f32>()
            / residuals.len() as f32;

        1.0 - residual_var / return_var
    }

    /// Main processing function (for interface compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Builder for GAE
#[derive(Debug, Clone)]
pub struct GaeBuilder {
    config: GaeConfig,
}

impl GaeBuilder {
    pub fn new() -> Self {
        Self {
            config: GaeConfig::default(),
        }
    }

    pub fn gamma(mut self, gamma: f32) -> Self {
        self.config.gamma = gamma;
        self
    }

    pub fn lambda(mut self, lambda: f32) -> Self {
        self.config.lambda = lambda;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    pub fn handle_dones(mut self, handle: bool) -> Self {
        self.config.handle_dones = handle;
        self
    }

    pub fn max_trajectory_length(mut self, length: usize) -> Self {
        self.config.max_trajectory_length = length;
        self
    }

    pub fn build(self) -> Gae {
        Gae::new(self.config)
    }
}

impl Default for GaeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Presets for common GAE configurations
pub struct GaePresets;

impl GaePresets {
    /// PPO default configuration
    pub fn ppo() -> Gae {
        GaeBuilder::new()
            .gamma(0.99)
            .lambda(0.95)
            .normalize(true)
            .build()
    }

    /// A2C default configuration
    pub fn a2c() -> Gae {
        GaeBuilder::new()
            .gamma(0.99)
            .lambda(1.0) // No GAE, just returns
            .normalize(true)
            .build()
    }

    /// Low variance configuration (more TD-like)
    pub fn low_variance() -> Gae {
        GaeBuilder::new()
            .gamma(0.99)
            .lambda(0.0) // Pure TD(0)
            .normalize(true)
            .build()
    }

    /// Low bias configuration (more MC-like)
    pub fn low_bias() -> Gae {
        GaeBuilder::new()
            .gamma(0.99)
            .lambda(0.99)
            .normalize(true)
            .build()
    }

    /// Short horizon tasks (e.g., trading with quick feedback)
    pub fn short_horizon() -> Gae {
        GaeBuilder::new()
            .gamma(0.95)
            .lambda(0.9)
            .normalize(true)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gae_creation() {
        let gae = Gae::default();
        assert_eq!(gae.config.gamma, 0.99);
        assert_eq!(gae.config.lambda, 0.95);
    }

    #[test]
    fn test_trajectory_operations() {
        let mut trajectory = Trajectory::new();
        assert!(trajectory.is_empty());

        trajectory.add(vec![1.0, 2.0], 0, 1.0, 0.5, -0.5, false);
        trajectory.add(vec![2.0, 3.0], 1, 2.0, 1.0, -1.0, false);
        trajectory.add(vec![3.0, 4.0], 0, 3.0, 1.5, -0.8, true);

        assert_eq!(trajectory.len(), 3);
        assert!(!trajectory.is_empty());

        trajectory.set_final_value(2.0);
        assert_eq!(trajectory.final_value, 2.0);

        trajectory.clear();
        assert!(trajectory.is_empty());
    }

    #[test]
    fn test_gae_computation() {
        let mut gae = GaeBuilder::new()
            .gamma(0.99)
            .lambda(0.95)
            .normalize(false)
            .build();

        let mut trajectory = Trajectory::new();
        trajectory.add(vec![1.0], 0, 1.0, 0.0, -0.5, false);
        trajectory.add(vec![2.0], 1, 1.0, 0.5, -0.5, false);
        trajectory.add(vec![3.0], 0, 1.0, 1.0, -0.5, true);
        trajectory.set_final_value(0.0);

        let (advantages, returns) = gae.compute(&trajectory);

        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);

        // Check that advantages are finite
        assert!(advantages.iter().all(|&a| a.is_finite()));
        assert!(returns.iter().all(|&r| r.is_finite()));
    }

    #[test]
    fn test_gae_lambda_zero() {
        // λ = 0 should give TD(0) advantages
        let mut gae = GaeBuilder::new()
            .gamma(0.9)
            .lambda(0.0)
            .normalize(false)
            .build();

        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.5, 1.0];
        let dones = vec![false, false, false];
        let final_value = 1.5;

        let (advantages, _) = gae.compute_from_arrays(&rewards, &values, &dones, final_value);

        // TD(0): A_t = r_t + γV(s_{t+1}) - V(s_t)
        // A_0 = 1.0 + 0.9*0.5 - 0.0 = 1.45
        // A_1 = 1.0 + 0.9*1.0 - 0.5 = 1.4
        // A_2 = 1.0 + 0.9*1.5 - 1.0 = 1.35

        assert!((advantages[0] - 1.45).abs() < 0.01);
        assert!((advantages[1] - 1.4).abs() < 0.01);
        assert!((advantages[2] - 1.35).abs() < 0.01);
    }

    #[test]
    fn test_gae_lambda_one() {
        // λ = 1 should give Monte Carlo-like returns
        let mut gae = GaeBuilder::new()
            .gamma(0.9)
            .lambda(1.0)
            .normalize(false)
            .build();

        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0];
        let dones = vec![false, false, true];
        let final_value = 0.0;

        let (_advantages, returns) =
            gae.compute_from_arrays(&rewards, &values, &dones, final_value);

        // With λ=1 and all V=0, GAE reduces to discounted returns
        // G_2 = 1.0 (terminal)
        // G_1 = 1.0 + 0.9*1.0 = 1.9
        // G_0 = 1.0 + 0.9*1.9 = 2.71

        assert!((returns[2] - 1.0).abs() < 0.01);
        assert!((returns[1] - 1.9).abs() < 0.01);
        assert!((returns[0] - 2.71).abs() < 0.01);
    }

    #[test]
    fn test_episode_boundary_handling() {
        let mut gae = GaeBuilder::new()
            .gamma(0.99)
            .lambda(0.95)
            .handle_dones(true)
            .normalize(false)
            .build();

        // Episode ends at t=1
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let dones = vec![false, true, false]; // Episode ends at t=1
        let final_value = 1.0;

        let (advantages, _) = gae.compute_from_arrays(&rewards, &values, &dones, final_value);

        // At t=1 (done=true), GAE should reset
        // This prevents information from t=2 leaking back to t=0
        assert!(advantages.iter().all(|&a| a.is_finite()));
    }

    #[test]
    fn test_normalization() {
        let mut gae = GaeBuilder::new()
            .gamma(0.99)
            .lambda(0.95)
            .normalize(true)
            .build();

        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec![0.5, 1.0, 1.5, 2.0, 2.5];
        let dones = vec![false, false, false, false, false];

        let (advantages, _) = gae.compute_from_arrays(&rewards, &values, &dones, 3.0);

        // After normalization, mean should be ~0 and std ~1
        let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let var: f32 =
            advantages.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
        let std = var.sqrt();

        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
        assert!((std - 1.0).abs() < 0.1, "Std should be ~1, got {}", std);
    }

    #[test]
    fn test_explained_variance() {
        let gae = Gae::default();

        // Perfect predictions
        let values = vec![1.0, 2.0, 3.0];
        let returns = vec![1.0, 2.0, 3.0];
        let ev = gae.explained_variance(&values, &returns);
        assert!(
            (ev - 1.0).abs() < 0.01,
            "Perfect prediction should give EV~1"
        );

        // Random predictions
        let values = vec![1.0, 1.0, 1.0];
        let returns = vec![1.0, 5.0, 10.0];
        let ev = gae.explained_variance(&values, &returns);
        assert!(ev < 1.0, "Imperfect predictions should give EV<1");
    }

    #[test]
    fn test_td_lambda_returns() {
        let gae = GaeBuilder::new()
            .gamma(0.9)
            .lambda(0.8)
            .normalize(false)
            .build();

        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.5, 1.0];
        let dones = vec![false, false, false];

        let returns = gae.compute_td_lambda_returns(&rewards, &values, &dones, 1.5);

        assert_eq!(returns.len(), 3);
        assert!(returns.iter().all(|&r| r.is_finite()));

        // Returns should be positive for positive rewards
        assert!(returns.iter().all(|&r| r > 0.0));
    }

    #[test]
    fn test_statistics_tracking() {
        let mut gae = GaeBuilder::new().normalize(false).build();

        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.0, 0.5, 1.0];
        let dones = vec![false, false, false];

        gae.compute_from_arrays(&rewards, &values, &dones, 1.5);

        let stats = gae.stats();
        assert_eq!(stats.trajectories_processed, 1);
        assert_eq!(stats.timesteps_processed, 3);
        assert!(stats.count > 0);
    }

    #[test]
    fn test_presets() {
        let ppo = GaePresets::ppo();
        assert_eq!(ppo.config.gamma, 0.99);
        assert_eq!(ppo.config.lambda, 0.95);

        let a2c = GaePresets::a2c();
        assert_eq!(a2c.config.lambda, 1.0);

        let low_var = GaePresets::low_variance();
        assert_eq!(low_var.config.lambda, 0.0);

        let short = GaePresets::short_horizon();
        assert_eq!(short.config.gamma, 0.95);
    }

    #[test]
    fn test_builder_pattern() {
        let gae = GaeBuilder::new()
            .gamma(0.95)
            .lambda(0.9)
            .normalize(false)
            .handle_dones(false)
            .max_trajectory_length(1024)
            .build();

        assert_eq!(gae.config.gamma, 0.95);
        assert_eq!(gae.config.lambda, 0.9);
        assert!(!gae.config.normalize);
        assert!(!gae.config.handle_dones);
        assert_eq!(gae.config.max_trajectory_length, 1024);
    }

    #[test]
    fn test_empty_trajectory() {
        let mut gae = Gae::default();

        let (advantages, returns) = gae.compute(&Trajectory::new());
        assert!(advantages.is_empty());
        assert!(returns.is_empty());
    }

    #[test]
    fn test_single_step_trajectory() {
        let mut gae = GaeBuilder::new()
            .gamma(0.99)
            .lambda(0.95)
            .normalize(false)
            .build();

        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![true];

        let (advantages, _returns) = gae.compute_from_arrays(&rewards, &values, &dones, 0.0);

        // For terminal state: A = r - V(s) = 1.0 - 0.5 = 0.5
        assert_eq!(advantages.len(), 1);
        assert!((advantages[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_basic() {
        let instance = Gae::default();
        assert!(instance.process().is_ok());
    }
}
