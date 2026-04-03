//! Actor-Critic Network
//!
//! Implements the actor-critic architecture for reinforcement learning.
//! Uses simple feedforward neural networks with ReLU activations.

use crate::common::Result;
use rand::RngExt;

/// Simple feedforward neural network for policy
pub struct PolicyNetwork {
    pub state_dim: usize,
    pub hidden_dim: usize,
    pub action_dim: usize,
    // Weights: state -> hidden
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    // Weights: hidden -> action
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
}

impl PolicyNetwork {
    pub fn new(state_dim: usize, hidden_dim: usize, action_dim: usize) -> Self {
        let mut rng = rand::rng();

        // Xavier/Glorot initialization
        let w1_scale = (2.0 / (state_dim + hidden_dim) as f32).sqrt();
        let w2_scale = (2.0 / (hidden_dim + action_dim) as f32).sqrt();

        let w1 = (0..hidden_dim)
            .map(|_| {
                (0..state_dim)
                    .map(|_| rng.random_range(-w1_scale..w1_scale))
                    .collect()
            })
            .collect();

        let b1 = vec![0.0; hidden_dim];

        let w2 = (0..action_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.random_range(-w2_scale..w2_scale))
                    .collect()
            })
            .collect();

        let b2 = vec![0.0; action_dim];

        Self {
            state_dim,
            hidden_dim,
            action_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Forward pass: state -> action probabilities
    pub fn forward(&self, state: &[f32]) -> Vec<f32> {
        // Layer 1: state -> hidden
        let mut hidden = vec![0.0; self.hidden_dim];
        for (i, h) in hidden.iter_mut().enumerate().take(self.hidden_dim) {
            let mut sum = self.b1[i];
            for (j, &s) in state
                .iter()
                .enumerate()
                .take(self.state_dim.min(state.len()))
            {
                sum += self.w1[i][j] * s;
            }
            // ReLU activation
            *h = sum.max(0.0);
        }

        // Layer 2: hidden -> action logits
        let mut logits = vec![0.0; self.action_dim];
        for (i, logit) in logits.iter_mut().enumerate().take(self.action_dim) {
            let mut sum = self.b2[i];
            for (j, &h) in hidden.iter().enumerate().take(self.hidden_dim) {
                sum += self.w2[i][j] * h;
            }
            *logit = sum;
        }

        // Softmax to get probabilities
        self.softmax(&logits)
    }

    /// Softmax activation
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        exp_logits.iter().map(|&x| x / sum).collect()
    }

    /// Sample action from policy
    pub fn sample_action(&self, state: &[f32]) -> usize {
        let probs = self.forward(state);

        let mut rng = rand::rng();
        let sample: f32 = rng.random();

        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if sample <= cumsum {
                return i;
            }
        }

        probs.len() - 1
    }
}

/// Simple feedforward neural network for value estimation
pub struct ValueNetwork {
    pub state_dim: usize,
    pub hidden_dim: usize,
    // Weights: state -> hidden
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    // Weights: hidden -> value (single output)
    pub w2: Vec<f32>,
    pub b2: f32,
}

impl ValueNetwork {
    pub fn new(state_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();

        // Xavier/Glorot initialization
        let w1_scale = (2.0 / (state_dim + hidden_dim) as f32).sqrt();
        let w2_scale = (2.0 / (hidden_dim + 1) as f32).sqrt();

        let w1 = (0..hidden_dim)
            .map(|_| {
                (0..state_dim)
                    .map(|_| rng.random_range(-w1_scale..w1_scale))
                    .collect()
            })
            .collect();

        let b1 = vec![0.0; hidden_dim];

        let w2 = (0..hidden_dim)
            .map(|_| rng.random_range(-w2_scale..w2_scale))
            .collect();

        let b2 = 0.0;

        Self {
            state_dim,
            hidden_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Forward pass: state -> value estimate
    pub fn forward(&self, state: &[f32]) -> f32 {
        // Layer 1: state -> hidden
        let mut hidden = vec![0.0; self.hidden_dim];
        for (i, h) in hidden.iter_mut().enumerate().take(self.hidden_dim) {
            let mut sum = self.b1[i];
            for (j, &s) in state
                .iter()
                .enumerate()
                .take(self.state_dim.min(state.len()))
            {
                sum += self.w1[i][j] * s;
            }
            // ReLU activation
            *h = sum.max(0.0);
        }

        // Layer 2: hidden -> value
        let mut value = self.b2;
        for (i, &h) in hidden.iter().enumerate().take(self.hidden_dim) {
            value += self.w2[i] * h;
        }

        value
    }
}

pub struct ActorCritic {
    pub actor: PolicyNetwork,
    pub critic: ValueNetwork,
    pub gamma: f32, // Discount factor
}

impl ActorCritic {
    pub fn new(state_dim: usize, hidden_dim: usize, action_dim: usize, gamma: f32) -> Self {
        Self {
            actor: PolicyNetwork::new(state_dim, hidden_dim, action_dim),
            critic: ValueNetwork::new(state_dim, hidden_dim),
            gamma,
        }
    }

    /// Select action using current policy
    pub fn select_action(&self, state: &[f32]) -> usize {
        self.actor.sample_action(state)
    }

    /// Compute advantage: A(s,a) = Q(s,a) - V(s)
    pub fn compute_advantage(&self, state: &[f32], reward: f32, next_state: &[f32]) -> f32 {
        let v_s = self.critic.forward(state);
        let v_next = self.critic.forward(next_state);

        // TD(0) advantage
        reward + self.gamma * v_next - v_s
    }

    /// Update networks using simple gradient descent
    /// Note: This is a simplified implementation. For production, use a proper
    /// deep learning framework like candle or tch-rs.
    pub async fn update(&mut self, experiences: &[Experience]) -> Result<()> {
        if experiences.is_empty() {
            return Ok(());
        }

        let learning_rate = 0.001;

        for exp in experiences {
            // Compute advantage
            let advantage = self.compute_advantage(&exp.state, exp.reward, &exp.next_state);

            // Policy gradient update (simplified)
            // In practice, this would involve proper backpropagation
            let _action_probs = self.actor.forward(&exp.state);
            let _log_prob = _action_probs[exp.action].max(1e-8).ln();

            // Actor update: maximize advantage-weighted log probability
            // Simplified: just adjust towards taken action when advantage is positive
            if advantage > 0.0 {
                // Strengthen this action
                for i in 0..self.actor.action_dim {
                    if i == exp.action {
                        // Increase probability of this action slightly
                        self.actor.b2[i] += learning_rate * advantage * 0.1;
                    }
                }
            }

            // Critic update: minimize TD error
            let td_error = advantage;
            let _value_estimate = self.critic.forward(&exp.state);

            // Simplified gradient update for critic
            // Adjust biases based on TD error
            for i in 0..self.critic.hidden_dim {
                self.critic.b1[i] -= learning_rate * td_error * 0.01;
            }
            self.critic.b2 -= learning_rate * td_error;
        }

        Ok(())
    }
}

// Placeholder for experience type
#[derive(Clone)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_critic_creation() {
        let ac = ActorCritic::new(10, 64, 5, 0.99);
        assert_eq!(ac.actor.action_dim, 5);
        assert_eq!(ac.gamma, 0.99);
    }

    #[test]
    fn test_action_selection() {
        let ac = ActorCritic::new(10, 64, 5, 0.99);
        let state = vec![0.5; 10];

        let action = ac.select_action(&state);
        assert!(action < 5);
    }
}
