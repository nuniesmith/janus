//! Temporal Difference Learning
//!
//! Implements various TD learning algorithms for value function estimation:
//! - TD(0): Single-step temporal difference
//! - TD(λ): Eligibility traces for multi-step credit assignment
//! - N-step returns: Fixed horizon bootstrapping
//!
//! Part of the Basal Ganglia region - Critic component

use crate::common::Result;

/// Configuration for TD learning
#[derive(Debug, Clone)]
pub struct TdConfig {
    /// Learning rate (alpha)
    pub learning_rate: f32,
    /// Discount factor (gamma)
    pub discount_factor: f32,
    /// Eligibility trace decay (lambda) for TD(λ)
    pub trace_decay: f32,
    /// N-step horizon for n-step returns
    pub n_steps: usize,
    /// Initial value estimate
    pub initial_value: f32,
}

impl Default for TdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            discount_factor: 0.99,
            trace_decay: 0.9,
            n_steps: 5,
            initial_value: 0.0,
        }
    }
}

/// Temporal difference learning for value estimation
pub struct TdLearning {
    /// Configuration
    config: TdConfig,
    /// Value estimates (state -> value)
    values: Vec<f32>,
    /// Eligibility traces for TD(λ)
    eligibility_traces: Vec<f32>,
    /// State dimension
    state_dim: usize,
    /// Running TD error for monitoring
    running_td_error: f32,
    /// EMA decay for running stats
    ema_decay: f32,
    /// Experience buffer for n-step returns
    experience_buffer: Vec<TdExperience>,
}

/// Single experience tuple for TD learning
#[derive(Debug, Clone)]
pub struct TdExperience {
    /// State representation (flattened)
    pub state: Vec<f32>,
    /// Reward received
    pub reward: f32,
    /// Next state (None if terminal)
    pub next_state: Option<Vec<f32>>,
    /// Whether this was a terminal state
    pub done: bool,
}

/// TD update result with metrics
#[derive(Debug, Clone)]
pub struct TdUpdateResult {
    /// TD error (delta)
    pub td_error: f32,
    /// Updated value estimate
    pub new_value: f32,
    /// Previous value estimate
    pub old_value: f32,
}

impl Default for TdLearning {
    fn default() -> Self {
        Self::new()
    }
}

impl TdLearning {
    /// Create a new TD learning instance with default config
    pub fn new() -> Self {
        Self::with_config(TdConfig::default(), 64)
    }

    /// Create with custom configuration
    pub fn with_config(config: TdConfig, state_dim: usize) -> Self {
        Self {
            values: vec![config.initial_value; state_dim],
            eligibility_traces: vec![0.0; state_dim],
            state_dim,
            running_td_error: 0.0,
            ema_decay: 0.99,
            experience_buffer: Vec::with_capacity(config.n_steps),
            config,
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &TdConfig {
        &self.config
    }

    /// Update learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    /// TD(0) update: single-step temporal difference
    ///
    /// V(s) <- V(s) + α * [r + γ * V(s') - V(s)]
    pub fn td0_update(&mut self, experience: &TdExperience) -> TdUpdateResult {
        let state_idx = self.state_to_index(&experience.state);
        let old_value = self.values[state_idx];

        // Compute TD target
        let next_value = if experience.done {
            0.0
        } else if let Some(ref next_state) = experience.next_state {
            let next_idx = self.state_to_index(next_state);
            self.values[next_idx]
        } else {
            0.0
        };

        let td_target = experience.reward + self.config.discount_factor * next_value;
        let td_error = td_target - old_value;

        // Update value
        let new_value = old_value + self.config.learning_rate * td_error;
        self.values[state_idx] = new_value;

        // Update running TD error
        self.running_td_error =
            self.ema_decay * self.running_td_error + (1.0 - self.ema_decay) * td_error.abs();

        TdUpdateResult {
            td_error,
            new_value,
            old_value,
        }
    }

    /// TD(λ) update with eligibility traces
    ///
    /// Uses backward view with accumulating traces:
    /// e(s) <- γλe(s) + 1 (for visited state)
    /// V(s) <- V(s) + α * δ * e(s) (for all states)
    pub fn td_lambda_update(&mut self, experience: &TdExperience) -> TdUpdateResult {
        let state_idx = self.state_to_index(&experience.state);
        let old_value = self.values[state_idx];

        // Compute TD error
        let next_value = if experience.done {
            0.0
        } else if let Some(ref next_state) = experience.next_state {
            let next_idx = self.state_to_index(next_state);
            self.values[next_idx]
        } else {
            0.0
        };

        let td_error = experience.reward + self.config.discount_factor * next_value - old_value;

        // Decay all traces and increment current state's trace
        for i in 0..self.state_dim {
            self.eligibility_traces[i] *= self.config.discount_factor * self.config.trace_decay;
        }
        self.eligibility_traces[state_idx] += 1.0;

        // Update all values using eligibility traces
        for i in 0..self.state_dim {
            self.values[i] += self.config.learning_rate * td_error * self.eligibility_traces[i];
        }

        // Reset traces on terminal state
        if experience.done {
            self.reset_traces();
        }

        let new_value = self.values[state_idx];

        // Update running TD error
        self.running_td_error =
            self.ema_decay * self.running_td_error + (1.0 - self.ema_decay) * td_error.abs();

        TdUpdateResult {
            td_error,
            new_value,
            old_value,
        }
    }

    /// N-step return update
    ///
    /// G_t:t+n = R_t+1 + γR_t+2 + ... + γ^(n-1)R_t+n + γ^n V(S_t+n)
    /// V(S_t) <- V(S_t) + α * [G_t:t+n - V(S_t)]
    pub fn n_step_update(&mut self, experience: TdExperience) -> Option<TdUpdateResult> {
        self.experience_buffer.push(experience);

        // Only update when we have enough experiences
        if self.experience_buffer.len() < self.config.n_steps {
            return None;
        }

        // Compute n-step return
        let mut n_step_return = 0.0;
        let mut gamma_power = 1.0;

        for exp in self.experience_buffer.iter().take(self.config.n_steps) {
            n_step_return += gamma_power * exp.reward;
            gamma_power *= self.config.discount_factor;
        }

        // Add bootstrapped value for non-terminal states
        if let Some(last_exp) = self.experience_buffer.get(self.config.n_steps - 1) {
            if !last_exp.done {
                if let Some(ref next_state) = last_exp.next_state {
                    let next_idx = self.state_to_index(next_state);
                    n_step_return += gamma_power * self.values[next_idx];
                }
            }
        }

        // Update the first state in the buffer
        let first_state = &self.experience_buffer[0].state;
        let state_idx = self.state_to_index(first_state);
        let old_value = self.values[state_idx];

        let td_error = n_step_return - old_value;
        let new_value = old_value + self.config.learning_rate * td_error;
        self.values[state_idx] = new_value;

        // Remove the first experience (sliding window)
        self.experience_buffer.remove(0);

        // Update running TD error
        self.running_td_error =
            self.ema_decay * self.running_td_error + (1.0 - self.ema_decay) * td_error.abs();

        Some(TdUpdateResult {
            td_error,
            new_value,
            old_value,
        })
    }

    /// Flush remaining experiences at episode end
    pub fn flush_n_step_buffer(&mut self) -> Vec<TdUpdateResult> {
        let mut results = Vec::new();

        while !self.experience_buffer.is_empty() {
            // Compute return with remaining experiences
            let mut n_step_return = 0.0;
            let mut gamma_power = 1.0;

            for exp in &self.experience_buffer {
                n_step_return += gamma_power * exp.reward;
                gamma_power *= self.config.discount_factor;

                if exp.done {
                    break;
                }
            }

            // Add bootstrapped value if not terminal
            if let Some(last_exp) = self.experience_buffer.last() {
                if !last_exp.done {
                    if let Some(ref next_state) = last_exp.next_state {
                        let next_idx = self.state_to_index(next_state);
                        n_step_return += gamma_power * self.values[next_idx];
                    }
                }
            }

            // Update first state
            let first_state = &self.experience_buffer[0].state;
            let state_idx = self.state_to_index(first_state);
            let old_value = self.values[state_idx];

            let td_error = n_step_return - old_value;
            let new_value = old_value + self.config.learning_rate * td_error;
            self.values[state_idx] = new_value;

            results.push(TdUpdateResult {
                td_error,
                new_value,
                old_value,
            });

            self.experience_buffer.remove(0);
        }

        results
    }

    /// Get value estimate for a state
    pub fn get_value(&self, state: &[f32]) -> f32 {
        let idx = self.state_to_index(state);
        self.values[idx]
    }

    /// Get running average TD error (for monitoring)
    pub fn get_running_td_error(&self) -> f32 {
        self.running_td_error
    }

    /// Reset eligibility traces (typically at episode end)
    pub fn reset_traces(&mut self) {
        for trace in &mut self.eligibility_traces {
            *trace = 0.0;
        }
    }

    /// Reset all state (values and traces)
    pub fn reset(&mut self) {
        for v in &mut self.values {
            *v = self.config.initial_value;
        }
        self.reset_traces();
        self.running_td_error = 0.0;
        self.experience_buffer.clear();
    }

    /// Convert state vector to index using simple hashing
    /// For production, consider using function approximation instead of tabular
    fn state_to_index(&self, state: &[f32]) -> usize {
        // Simple hash-based indexing
        let mut hash: u64 = 0;
        for (i, &val) in state.iter().enumerate() {
            // Discretize and hash
            let discretized = (val * 1000.0) as i64;
            hash = hash.wrapping_add(discretized as u64 * (i as u64 + 1));
        }
        (hash as usize) % self.state_dim
    }

    /// Main processing function for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Compute TD error without updating (for external use)
    pub fn compute_td_error(&self, state: &[f32], reward: f32, next_state: Option<&[f32]>) -> f32 {
        let state_idx = self.state_to_index(state);
        let current_value = self.values[state_idx];

        let next_value = next_state
            .map(|ns| {
                let next_idx = self.state_to_index(ns);
                self.values[next_idx]
            })
            .unwrap_or(0.0);

        reward + self.config.discount_factor * next_value - current_value
    }
}

/// Generalized Advantage Estimation (GAE)
/// Combines TD(λ) with advantage estimation for actor-critic methods
pub struct GaeEstimator {
    /// Discount factor
    gamma: f32,
    /// GAE lambda parameter
    lambda: f32,
}

impl GaeEstimator {
    pub fn new(gamma: f32, lambda: f32) -> Self {
        Self { gamma, lambda }
    }

    /// Compute GAE advantages from a trajectory
    ///
    /// A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    /// where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    pub fn compute_advantages(
        &self,
        rewards: &[f32],
        values: &[f32],
        next_value: f32,
        dones: &[bool],
    ) -> Vec<f32> {
        let n = rewards.len();
        let mut advantages = vec![0.0; n];
        let mut last_gae = 0.0;

        // Compute backwards
        for t in (0..n).rev() {
            let next_val = if t == n - 1 {
                next_value
            } else {
                values[t + 1]
            };

            let mask = if dones[t] { 0.0 } else { 1.0 };
            let delta = rewards[t] + self.gamma * next_val * mask - values[t];
            last_gae = delta + self.gamma * self.lambda * mask * last_gae;
            advantages[t] = last_gae;
        }

        advantages
    }

    /// Compute returns (advantages + values)
    pub fn compute_returns(&self, advantages: &[f32], values: &[f32]) -> Vec<f32> {
        advantages
            .iter()
            .zip(values.iter())
            .map(|(a, v)| a + v)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_td0_update() {
        let mut td = TdLearning::with_config(
            TdConfig {
                learning_rate: 0.1,
                discount_factor: 0.9,
                ..Default::default()
            },
            64,
        );

        let exp = TdExperience {
            state: vec![0.5; 4],
            reward: 1.0,
            next_state: Some(vec![0.6; 4]),
            done: false,
        };

        let result = td.td0_update(&exp);

        // Value should have changed
        assert!(result.new_value != result.old_value);
        assert!(result.td_error.abs() > 0.0);
    }

    #[test]
    fn test_td_lambda_update() {
        let mut td = TdLearning::with_config(
            TdConfig {
                learning_rate: 0.1,
                discount_factor: 0.9,
                trace_decay: 0.8,
                ..Default::default()
            },
            64,
        );

        // Simulate a short episode
        for i in 0..5 {
            let exp = TdExperience {
                state: vec![i as f32 * 0.1; 4],
                reward: 1.0,
                next_state: Some(vec![(i + 1) as f32 * 0.1; 4]),
                done: i == 4,
            };

            td.td_lambda_update(&exp);
        }

        // Traces should be reset after terminal state
        assert!(td.eligibility_traces.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn test_n_step_returns() {
        let mut td = TdLearning::with_config(
            TdConfig {
                learning_rate: 0.1,
                discount_factor: 0.9,
                n_steps: 3,
                ..Default::default()
            },
            64,
        );

        // First two updates return None (buffer not full)
        for i in 0..2 {
            let exp = TdExperience {
                state: vec![i as f32 * 0.1; 4],
                reward: 1.0,
                next_state: Some(vec![(i + 1) as f32 * 0.1; 4]),
                done: false,
            };
            assert!(td.n_step_update(exp).is_none());
        }

        // Third update should produce a result
        let exp = TdExperience {
            state: vec![0.2; 4],
            reward: 1.0,
            next_state: Some(vec![0.3; 4]),
            done: false,
        };
        let result = td.n_step_update(exp);
        assert!(result.is_some());
    }

    #[test]
    fn test_gae_estimation() {
        let gae = GaeEstimator::new(0.99, 0.95);

        let rewards = vec![1.0, 0.5, 2.0];
        let values = vec![1.0, 1.5, 2.0];
        let next_value = 1.8;
        let dones = vec![false, false, false];

        let advantages = gae.compute_advantages(&rewards, &values, next_value, &dones);

        assert_eq!(advantages.len(), 3);
        // Advantages should be computed backwards
        assert!(advantages[0].abs() > 0.0);
    }

    #[test]
    fn test_terminal_state() {
        let mut td = TdLearning::with_config(
            TdConfig {
                learning_rate: 0.1,
                discount_factor: 0.9,
                ..Default::default()
            },
            64,
        );

        let exp = TdExperience {
            state: vec![0.5; 4],
            reward: 10.0,
            next_state: None,
            done: true,
        };

        let result = td.td0_update(&exp);

        // For terminal state, TD target should just be the reward
        assert!((result.td_error - (10.0 - result.old_value)).abs() < 0.001);
    }

    #[test]
    fn test_value_retrieval() {
        let mut td = TdLearning::with_config(
            TdConfig {
                initial_value: 5.0,
                ..Default::default()
            },
            64,
        );

        let state = vec![0.5; 4];
        let initial_val = td.get_value(&state);
        assert_eq!(initial_val, 5.0);

        // Update and check again
        let exp = TdExperience {
            state: state.clone(),
            reward: 10.0,
            next_state: Some(vec![0.6; 4]),
            done: false,
        };
        td.td0_update(&exp);

        let updated_val = td.get_value(&state);
        assert_ne!(updated_val, initial_val);
    }
}
