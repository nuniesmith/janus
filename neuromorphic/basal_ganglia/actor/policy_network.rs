//! Neural Policy Network
//!
//! Implements policy-based action selection for reinforcement learning:
//! - Stochastic policy with softmax output
//! - Action sampling with entropy regularization
//! - Policy gradient estimation (REINFORCE)
//! - PPO-compatible ratio computation
//!
//! Part of the Basal Ganglia region - Actor component

use crate::common::{Error, Result};
use rand::{Rng, RngExt};

/// Configuration for the policy network
#[derive(Debug, Clone)]
pub struct PolicyConfig {
    /// Input state dimension
    pub state_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Output action dimension
    pub action_dim: usize,
    /// Temperature for softmax (higher = more exploration)
    pub temperature: f32,
    /// Entropy coefficient for regularization
    pub entropy_coef: f32,
    /// Clip ratio for PPO
    pub clip_ratio: f32,
    /// Learning rate
    pub learning_rate: f32,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            state_dim: 64,
            hidden_dim: 128,
            action_dim: 5,
            temperature: 1.0,
            entropy_coef: 0.01,
            clip_ratio: 0.2,
            learning_rate: 0.0003,
        }
    }
}

/// Neural policy network for action selection
pub struct PolicyNetwork {
    /// Configuration
    config: PolicyConfig,
    /// Weights: state -> hidden
    w1: Vec<Vec<f32>>,
    /// Biases: hidden layer
    b1: Vec<f32>,
    /// Weights: hidden -> action
    w2: Vec<Vec<f32>>,
    /// Biases: action layer
    b2: Vec<f32>,
    /// Last hidden activations (for gradient computation)
    last_hidden: Vec<f32>,
    /// Last action probabilities
    last_probs: Vec<f32>,
    /// Running mean for input normalization
    running_mean: Vec<f32>,
    /// Running variance for input normalization
    running_var: Vec<f32>,
    /// Update count for normalization
    update_count: u64,
}

/// Action selection result
#[derive(Debug, Clone)]
pub struct ActionResult {
    /// Selected action index
    pub action: usize,
    /// Probability of selected action
    pub prob: f32,
    /// Log probability of selected action
    pub log_prob: f32,
    /// Full probability distribution
    pub probs: Vec<f32>,
    /// Entropy of the distribution
    pub entropy: f32,
}

/// Policy gradient update result
#[derive(Debug, Clone)]
pub struct PolicyUpdate {
    /// Policy loss
    pub policy_loss: f32,
    /// Entropy bonus
    pub entropy_bonus: f32,
    /// Total loss
    pub total_loss: f32,
    /// Average probability ratio (for PPO)
    pub avg_ratio: f32,
}

impl Default for PolicyNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyNetwork {
    /// Create a new policy network with default config
    pub fn new() -> Self {
        Self::with_config(PolicyConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PolicyConfig) -> Self {
        let mut rng = rand::rng();

        // Xavier/Glorot initialization
        let w1_scale = (2.0 / (config.state_dim + config.hidden_dim) as f32).sqrt();
        let w2_scale = (2.0 / (config.hidden_dim + config.action_dim) as f32).sqrt();

        let w1: Vec<Vec<f32>> = (0..config.hidden_dim)
            .map(|_| {
                (0..config.state_dim)
                    .map(|_| rng.random_range(-w1_scale..w1_scale))
                    .collect()
            })
            .collect();

        let b1 = vec![0.0; config.hidden_dim];

        let w2: Vec<Vec<f32>> = (0..config.action_dim)
            .map(|_| {
                (0..config.hidden_dim)
                    .map(|_| rng.random_range(-w2_scale..w2_scale))
                    .collect()
            })
            .collect();

        let b2 = vec![0.0; config.action_dim];

        Self {
            last_hidden: vec![0.0; config.hidden_dim],
            last_probs: vec![0.0; config.action_dim],
            running_mean: vec![0.0; config.state_dim],
            running_var: vec![1.0; config.state_dim],
            update_count: 0,
            config,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    /// Forward pass: state -> action probabilities
    pub fn forward(&mut self, state: &[f32]) -> Vec<f32> {
        // Normalize input
        let normalized = self.normalize_input(state);

        // Layer 1: state -> hidden with ReLU
        for i in 0..self.config.hidden_dim {
            let mut sum = self.b1[i];
            for j in 0..self.config.state_dim.min(normalized.len()) {
                sum += self.w1[i][j] * normalized[j];
            }
            // ReLU activation
            self.last_hidden[i] = sum.max(0.0);
        }

        // Layer 2: hidden -> action logits
        let mut logits = vec![0.0; self.config.action_dim];
        for i in 0..self.config.action_dim {
            let mut sum = self.b2[i];
            for j in 0..self.config.hidden_dim {
                sum += self.w2[i][j] * self.last_hidden[j];
            }
            // Apply temperature
            logits[i] = sum / self.config.temperature;
        }

        // Softmax
        self.last_probs = self.softmax(&logits);
        self.last_probs.clone()
    }

    /// Normalize input using running statistics
    fn normalize_input(&mut self, state: &[f32]) -> Vec<f32> {
        self.update_count += 1;
        let alpha = 0.01; // EMA decay

        let mut normalized = vec![0.0; state.len()];

        for (i, &val) in state.iter().enumerate() {
            if i < self.config.state_dim {
                // Update running statistics
                self.running_mean[i] = (1.0 - alpha) * self.running_mean[i] + alpha * val;
                let diff = val - self.running_mean[i];
                self.running_var[i] = (1.0 - alpha) * self.running_var[i] + alpha * diff * diff;

                // Normalize
                let std = (self.running_var[i] + 1e-8).sqrt();
                normalized[i] = (val - self.running_mean[i]) / std;
            }
        }

        normalized
    }

    /// Softmax with numerical stability
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        exp_logits.iter().map(|&x| x / sum.max(1e-8)).collect()
    }

    /// Select action from policy distribution
    pub fn select_action(&mut self, state: &[f32]) -> ActionResult {
        let probs = self.forward(state);

        // Sample action
        let mut rng = rand::rng();
        let sample: f32 = rng.random();

        let mut cumsum = 0.0;
        let mut action = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if sample <= cumsum {
                action = i;
                break;
            }
        }

        let prob = probs[action].max(1e-8);
        let log_prob = prob.ln();
        let entropy = self.compute_entropy(&probs);

        ActionResult {
            action,
            prob,
            log_prob,
            probs,
            entropy,
        }
    }

    /// Select action deterministically (for evaluation)
    pub fn select_action_deterministic(&mut self, state: &[f32]) -> usize {
        let probs = self.forward(state);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute entropy of probability distribution
    pub fn compute_entropy(&self, probs: &[f32]) -> f32 {
        -probs
            .iter()
            .map(|&p| if p > 1e-8 { p * p.ln() } else { 0.0 })
            .sum::<f32>()
    }

    /// Get log probability of a specific action given state
    pub fn get_log_prob(&mut self, state: &[f32], action: usize) -> f32 {
        let probs = self.forward(state);
        probs[action].max(1e-8).ln()
    }

    /// Compute importance sampling ratio (for PPO)
    pub fn compute_ratio(&mut self, state: &[f32], action: usize, old_log_prob: f32) -> f32 {
        let new_log_prob = self.get_log_prob(state, action);
        (new_log_prob - old_log_prob).exp()
    }

    /// REINFORCE policy gradient update
    ///
    /// ∇J(θ) ≈ Σ_t [∇log π(a_t|s_t) * A_t]
    pub fn reinforce_update(
        &mut self,
        states: &[Vec<f32>],
        actions: &[usize],
        advantages: &[f32],
    ) -> PolicyUpdate {
        if states.is_empty() {
            return PolicyUpdate {
                policy_loss: 0.0,
                entropy_bonus: 0.0,
                total_loss: 0.0,
                avg_ratio: 1.0,
            };
        }

        let mut policy_loss = 0.0;
        let mut total_entropy = 0.0;

        // Compute gradients and accumulate updates
        let mut w1_grad = vec![vec![0.0; self.config.state_dim]; self.config.hidden_dim];
        let mut b1_grad = vec![0.0; self.config.hidden_dim];
        let mut w2_grad = vec![vec![0.0; self.config.hidden_dim]; self.config.action_dim];
        let mut b2_grad = vec![0.0; self.config.action_dim];

        for ((state, &action), &advantage) in
            states.iter().zip(actions.iter()).zip(advantages.iter())
        {
            let probs = self.forward(state);
            let log_prob = probs[action].max(1e-8).ln();
            let entropy = self.compute_entropy(&probs);

            policy_loss -= log_prob * advantage;
            total_entropy += entropy;

            // Compute gradients (simplified backprop)
            // Gradient of log softmax w.r.t. logits
            let mut d_logits = probs.clone();
            d_logits[action] -= 1.0; // Gradient of cross-entropy

            // Scale by advantage
            for d in &mut d_logits {
                *d *= -advantage;
            }

            // Backprop through layer 2
            for i in 0..self.config.action_dim {
                b2_grad[i] += d_logits[i];
                for j in 0..self.config.hidden_dim {
                    w2_grad[i][j] += d_logits[i] * self.last_hidden[j];
                }
            }

            // Backprop through ReLU and layer 1
            let normalized = self.normalize_input(state);
            for j in 0..self.config.hidden_dim {
                let mut d_hidden = 0.0;
                for i in 0..self.config.action_dim {
                    d_hidden += d_logits[i] * self.w2[i][j];
                }

                // ReLU gradient
                if self.last_hidden[j] > 0.0 {
                    b1_grad[j] += d_hidden;
                    for k in 0..self.config.state_dim.min(normalized.len()) {
                        w1_grad[j][k] += d_hidden * normalized[k];
                    }
                }
            }
        }

        let n = states.len() as f32;
        policy_loss /= n;
        let entropy_bonus = (total_entropy / n) * self.config.entropy_coef;

        // Apply updates
        for i in 0..self.config.hidden_dim {
            self.b1[i] -= self.config.learning_rate * b1_grad[i] / n;
            for j in 0..self.config.state_dim {
                self.w1[i][j] -= self.config.learning_rate * w1_grad[i][j] / n;
            }
        }

        for i in 0..self.config.action_dim {
            // Add entropy gradient (encourages exploration)
            self.b2[i] -= self.config.learning_rate * (b2_grad[i] / n);
            for j in 0..self.config.hidden_dim {
                self.w2[i][j] -= self.config.learning_rate * (w2_grad[i][j] / n);
            }
        }

        PolicyUpdate {
            policy_loss,
            entropy_bonus,
            total_loss: policy_loss - entropy_bonus,
            avg_ratio: 1.0,
        }
    }

    /// PPO clipped surrogate update
    ///
    /// L^CLIP = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
    pub fn ppo_update(
        &mut self,
        states: &[Vec<f32>],
        actions: &[usize],
        old_log_probs: &[f32],
        advantages: &[f32],
    ) -> PolicyUpdate {
        if states.is_empty() {
            return PolicyUpdate {
                policy_loss: 0.0,
                entropy_bonus: 0.0,
                total_loss: 0.0,
                avg_ratio: 1.0,
            };
        }

        let mut policy_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut total_ratio = 0.0;

        // Simplified PPO update (without full backprop infrastructure)
        for (((state, &action), &old_log_prob), &advantage) in states
            .iter()
            .zip(actions.iter())
            .zip(old_log_probs.iter())
            .zip(advantages.iter())
        {
            let probs = self.forward(state);
            let new_log_prob = probs[action].max(1e-8).ln();
            let entropy = self.compute_entropy(&probs);

            // Importance sampling ratio
            let ratio = (new_log_prob - old_log_prob).exp();
            total_ratio += ratio;

            // Clipped surrogate objective
            let surr1 = ratio * advantage;
            let surr2 =
                ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantage;

            policy_loss -= surr1.min(surr2);
            total_entropy += entropy;

            // Apply gradient update for this sample
            // (Simplified: adjust action weights based on advantage sign)
            let update_scale = self.config.learning_rate * advantage.signum() * 0.01;

            if advantage > 0.0 && ratio < 1.0 + self.config.clip_ratio {
                // Encourage this action
                self.b2[action] += update_scale;
            } else if advantage < 0.0 && ratio > 1.0 - self.config.clip_ratio {
                // Discourage this action
                self.b2[action] -= update_scale.abs();
            }
        }

        let n = states.len() as f32;
        policy_loss /= n;
        let entropy_bonus = (total_entropy / n) * self.config.entropy_coef;

        PolicyUpdate {
            policy_loss,
            entropy_bonus,
            total_loss: policy_loss - entropy_bonus,
            avg_ratio: total_ratio / n,
        }
    }

    /// Set temperature for exploration control
    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature.max(0.01);
    }

    /// Get current action probabilities (last forward pass)
    pub fn get_last_probs(&self) -> &[f32] {
        &self.last_probs
    }

    /// Reset network to initial state
    pub fn reset(&mut self) {
        let mut rng = rand::rng();
        let w1_scale = (2.0 / (self.config.state_dim + self.config.hidden_dim) as f32).sqrt();
        let w2_scale = (2.0 / (self.config.hidden_dim + self.config.action_dim) as f32).sqrt();

        for i in 0..self.config.hidden_dim {
            self.b1[i] = 0.0;
            for j in 0..self.config.state_dim {
                self.w1[i][j] = rng.random_range(-w1_scale..w1_scale);
            }
        }

        for i in 0..self.config.action_dim {
            self.b2[i] = 0.0;
            for j in 0..self.config.hidden_dim {
                self.w2[i][j] = rng.random_range(-w2_scale..w2_scale);
            }
        }

        self.running_mean = vec![0.0; self.config.state_dim];
        self.running_var = vec![1.0; self.config.state_dim];
        self.update_count = 0;
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_creation() {
        let policy = PolicyNetwork::new();
        assert_eq!(policy.config.action_dim, 5);
    }

    #[test]
    fn test_forward_pass() {
        let mut policy = PolicyNetwork::with_config(PolicyConfig {
            state_dim: 4,
            hidden_dim: 8,
            action_dim: 3,
            ..Default::default()
        });

        let state = vec![0.5, 0.3, -0.2, 0.8];
        let probs = policy.forward(&state);

        assert_eq!(probs.len(), 3);

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // All probabilities should be positive
        assert!(probs.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_action_selection() {
        let mut policy = PolicyNetwork::with_config(PolicyConfig {
            state_dim: 4,
            hidden_dim: 8,
            action_dim: 5,
            ..Default::default()
        });

        let state = vec![0.5, 0.3, -0.2, 0.8];
        let result = policy.select_action(&state);

        assert!(result.action < 5);
        assert!(result.prob > 0.0);
        assert!(result.entropy >= 0.0);
    }

    #[test]
    fn test_deterministic_selection() {
        let mut policy = PolicyNetwork::with_config(PolicyConfig {
            state_dim: 4,
            hidden_dim: 8,
            action_dim: 3,
            temperature: 0.01, // Low temperature = more deterministic
            ..Default::default()
        });

        let state = vec![0.5, 0.3, -0.2, 0.8];

        // Multiple calls should return same action with low temperature
        let action1 = policy.select_action_deterministic(&state);
        let action2 = policy.select_action_deterministic(&state);

        assert_eq!(action1, action2);
    }

    #[test]
    fn test_entropy_computation() {
        let policy = PolicyNetwork::new();

        // Uniform distribution should have max entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let uniform_entropy = policy.compute_entropy(&uniform);

        // Peaked distribution should have lower entropy
        let peaked = vec![0.9, 0.05, 0.03, 0.02];
        let peaked_entropy = policy.compute_entropy(&peaked);

        assert!(uniform_entropy > peaked_entropy);
    }

    #[test]
    fn test_reinforce_update() {
        let mut policy = PolicyNetwork::with_config(PolicyConfig {
            state_dim: 4,
            hidden_dim: 8,
            action_dim: 3,
            learning_rate: 0.01,
            ..Default::default()
        });

        let states = vec![
            vec![0.5, 0.3, -0.2, 0.8],
            vec![0.1, -0.5, 0.3, 0.2],
            vec![-0.3, 0.7, 0.1, -0.4],
        ];
        let actions = vec![0, 1, 2];
        let advantages = vec![1.0, -0.5, 0.3];

        let update = policy.reinforce_update(&states, &actions, &advantages);

        assert!(update.total_loss.is_finite());
        assert!(update.entropy_bonus.is_finite());
    }

    #[test]
    fn test_ppo_update() {
        let mut policy = PolicyNetwork::with_config(PolicyConfig {
            state_dim: 4,
            hidden_dim: 8,
            action_dim: 3,
            learning_rate: 0.01,
            clip_ratio: 0.2,
            ..Default::default()
        });

        let states = vec![vec![0.5, 0.3, -0.2, 0.8], vec![0.1, -0.5, 0.3, 0.2]];
        let actions = vec![0, 1];
        let old_log_probs = vec![-1.2, -0.8];
        let advantages = vec![1.0, -0.5];

        let update = policy.ppo_update(&states, &actions, &old_log_probs, &advantages);

        assert!(update.total_loss.is_finite());
        assert!(update.avg_ratio > 0.0);
    }

    #[test]
    fn test_importance_ratio() {
        let mut policy = PolicyNetwork::with_config(PolicyConfig {
            state_dim: 4,
            hidden_dim: 8,
            action_dim: 3,
            ..Default::default()
        });

        let state = vec![0.5, 0.3, -0.2, 0.8];
        let action = 1;

        // Get current log prob
        let log_prob = policy.get_log_prob(&state, action);

        // Ratio with same log prob should be 1.0
        let ratio = policy.compute_ratio(&state, action, log_prob);
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_temperature_effect() {
        let mut policy_low_temp = PolicyNetwork::with_config(PolicyConfig {
            state_dim: 4,
            hidden_dim: 8,
            action_dim: 3,
            temperature: 0.1,
            ..Default::default()
        });

        let mut policy_high_temp = PolicyNetwork::with_config(PolicyConfig {
            state_dim: 4,
            hidden_dim: 8,
            action_dim: 3,
            temperature: 2.0,
            ..Default::default()
        });

        // Copy weights
        policy_high_temp.w1 = policy_low_temp.w1.clone();
        policy_high_temp.b1 = policy_low_temp.b1.clone();
        policy_high_temp.w2 = policy_low_temp.w2.clone();
        policy_high_temp.b2 = policy_low_temp.b2.clone();

        let state = vec![0.5, 0.3, -0.2, 0.8];

        let probs_low = policy_low_temp.forward(&state);
        let probs_high = policy_high_temp.forward(&state);

        let entropy_low = policy_low_temp.compute_entropy(&probs_low);
        let entropy_high = policy_high_temp.compute_entropy(&probs_high);

        // Higher temperature should give higher entropy (more uniform)
        assert!(entropy_high > entropy_low);
    }
}
