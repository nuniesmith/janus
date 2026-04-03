//! Prioritized Experience Replay Buffer with Sharp-Wave Ripple (SWR) Sampling.
//!
//! This module implements an experience replay buffer that stores transitions
//! (state, action, reward, next_state) with prioritization for efficient learning.
//!
//! # Features
//!
//! - **Prioritized Sampling**: Sample important experiences more frequently
//! - **SWR-style Replay**: Simulate hippocampal sharp-wave ripples during consolidation
//! - **Circular Buffer**: Fixed-size buffer with automatic overwriting
//! - **Importance Sampling**: Correct for bias introduced by prioritization
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │              Prioritized Replay Buffer                   │
//! ├──────────────────────────────────────────────────────────┤
//! │                                                           │
//! │  Experiences: [(s, a, r, s'), ...]  [Circular Buffer]    │
//! │       ↓                                                   │
//! │  Priorities: [p₁, p₂, ..., pₙ]     [TD-Error Based]      │
//! │       ↓                                                   │
//! │  Sampling: P(i) ∝ pᵢᵅ              [Stochastic]          │
//! │       ↓                                                   │
//! │  IS Weights: wᵢ = (N·P(i))⁻ᵝ       [Bias Correction]     │
//! │                                                           │
//! └──────────────────────────────────────────────────────────┘
//! ```

use anyhow::Result;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for the replay buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBufferConfig {
    /// Maximum capacity of the buffer
    pub capacity: usize,

    /// Alpha parameter for prioritization (0 = uniform, 1 = full prioritization)
    pub alpha: f32,

    /// Beta parameter for importance sampling (0 = no correction, 1 = full correction)
    pub beta: f32,

    /// Beta annealing: increase beta over time to full correction
    pub beta_anneal_steps: usize,

    /// Epsilon for numerical stability in priority calculation
    pub priority_epsilon: f32,

    /// Minimum priority to avoid zero probabilities
    pub min_priority: f32,
}

impl Default for ReplayBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 100_000,
            alpha: 0.6,
            beta: 0.4,
            beta_anneal_steps: 1_000_000,
            priority_epsilon: 1e-6,
            min_priority: 0.01,
        }
    }
}

/// A single experience/transition in the replay buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience<S, A> {
    /// Current state
    pub state: S,

    /// Action taken
    pub action: A,

    /// Reward received
    pub reward: f32,

    /// Next state
    pub next_state: S,

    /// Whether the episode terminated
    pub done: bool,

    /// Additional metadata (e.g., timestamp, episode ID)
    pub metadata: ExperienceMetadata,
}

/// Metadata associated with an experience.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExperienceMetadata {
    /// Episode ID
    pub episode_id: u64,

    /// Step within episode
    pub step: usize,

    /// Timestamp when experience was collected
    pub timestamp: i64,

    /// Optional: importance for SWR replay
    pub swr_importance: Option<f32>,
}

/// Sampled batch from the replay buffer.
#[derive(Debug, Clone)]
pub struct ReplayBatch<S, A> {
    /// Batch of experiences
    pub experiences: Vec<Experience<S, A>>,

    /// Indices in the buffer (for priority updates)
    pub indices: Vec<usize>,

    /// Importance sampling weights
    pub is_weights: Vec<f32>,
}

/// Prioritized Experience Replay Buffer.
///
/// Stores experiences with associated priorities and samples them
/// according to their importance for training.
pub struct PrioritizedReplayBuffer<S, A> {
    /// Configuration
    config: ReplayBufferConfig,

    /// Circular buffer of experiences
    buffer: VecDeque<Experience<S, A>>,

    /// Priorities for each experience
    priorities: VecDeque<f32>,

    /// Current buffer size
    size: usize,

    /// Total number of experiences added (for annealing)
    total_added: usize,

    /// Maximum priority seen so far (for new experiences)
    max_priority: f32,

    /// Random number generator
    rng: ThreadRng,
}

impl<S, A> PrioritizedReplayBuffer<S, A>
where
    S: Clone,
    A: Clone,
{
    /// Create a new prioritized replay buffer.
    pub fn new(config: ReplayBufferConfig) -> Self {
        Self {
            buffer: VecDeque::with_capacity(config.capacity),
            priorities: VecDeque::with_capacity(config.capacity),
            size: 0,
            total_added: 0,
            max_priority: config.min_priority,
            config,
            rng: rand::rng(),
        }
    }

    /// Add an experience to the buffer.
    ///
    /// New experiences are assigned the maximum priority to ensure
    /// they get sampled at least once.
    pub fn add(&mut self, experience: Experience<S, A>) {
        let priority = self.max_priority.max(self.config.min_priority);

        if self.buffer.len() >= self.config.capacity {
            // Buffer is full, remove oldest
            self.buffer.pop_front();
            self.priorities.pop_front();
        } else {
            self.size += 1;
        }

        self.buffer.push_back(experience);
        self.priorities.push_back(priority);
        self.total_added += 1;
    }

    /// Add multiple experiences at once.
    pub fn add_batch(&mut self, experiences: Vec<Experience<S, A>>) {
        for exp in experiences {
            self.add(exp);
        }
    }

    /// Sample a batch of experiences using prioritized sampling.
    ///
    /// Returns the experiences, their indices, and importance sampling weights.
    pub fn sample(&mut self, batch_size: usize) -> Result<ReplayBatch<S, A>> {
        if self.size == 0 {
            anyhow::bail!("Cannot sample from empty buffer");
        }

        if batch_size > self.size {
            anyhow::bail!(
                "Batch size {} exceeds buffer size {}",
                batch_size,
                self.size
            );
        }

        // Compute sampling probabilities
        let probabilities = self.compute_sampling_probabilities();

        // Sample indices using weighted random sampling
        let mut indices = Vec::with_capacity(batch_size);
        let mut experiences = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Sample index based on probabilities
            let idx = self.weighted_sample(&probabilities);
            indices.push(idx);
            experiences.push(self.buffer[idx].clone());
        }

        // Compute importance sampling weights
        let is_weights = self.compute_is_weights(&indices, &probabilities);

        Ok(ReplayBatch {
            experiences,
            indices,
            is_weights,
        })
    }

    /// Sample with SWR-style replay: prioritize recent high-reward experiences.
    ///
    /// Simulates hippocampal sharp-wave ripples by biasing sampling towards
    /// recent, important experiences during "sleep" consolidation.
    pub fn sample_swr(
        &mut self,
        batch_size: usize,
        recency_bias: f32,
    ) -> Result<ReplayBatch<S, A>> {
        if self.size == 0 {
            anyhow::bail!("Cannot sample from empty buffer");
        }

        // Compute base probabilities
        let mut probabilities = self.compute_sampling_probabilities();

        // Apply recency bias (exponential decay)
        for (i, prob) in probabilities.iter_mut().enumerate() {
            let age = self.size - i - 1;
            let recency_weight = (-recency_bias * age as f32 / self.size as f32).exp();
            *prob *= recency_weight;
        }

        // Apply SWR importance if available
        for (i, exp) in self.buffer.iter().enumerate() {
            if let Some(swr_importance) = exp.metadata.swr_importance {
                probabilities[i] *= swr_importance;
            }
        }

        // Renormalize
        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            for prob in &mut probabilities {
                *prob /= sum;
            }
        }

        // Sample using modified distribution
        let mut indices = Vec::with_capacity(batch_size);
        let mut experiences = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = self.weighted_sample(&probabilities);
            indices.push(idx);
            experiences.push(self.buffer[idx].clone());
        }

        let is_weights = self.compute_is_weights(&indices, &probabilities);

        Ok(ReplayBatch {
            experiences,
            indices,
            is_weights,
        })
    }

    /// Update priorities for a batch of experiences based on TD errors.
    ///
    /// Called after training to adjust priorities based on how surprising
    /// each transition was (high TD error = high priority).
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        assert_eq!(
            indices.len(),
            td_errors.len(),
            "Indices and TD errors must have same length"
        );

        for (&idx, &td_error) in indices.iter().zip(td_errors.iter()) {
            if idx < self.size {
                // Priority = |TD error| + epsilon
                let priority = (td_error.abs() + self.config.priority_epsilon)
                    .powf(self.config.alpha)
                    .max(self.config.min_priority);

                self.priorities[idx] = priority;
                self.max_priority = self.max_priority.max(priority);
            }
        }
    }

    /// Get the current size of the buffer.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Check if the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.size >= self.config.capacity
    }

    /// Get the current beta value (annealed over time).
    pub fn current_beta(&self) -> f32 {
        let progress = (self.total_added as f32 / self.config.beta_anneal_steps as f32).min(1.0);
        self.config.beta + (1.0 - self.config.beta) * progress
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.priorities.clear();
        self.size = 0;
        self.max_priority = self.config.min_priority;
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> ReplayBufferStats {
        let avg_priority = if self.size > 0 {
            self.priorities.iter().sum::<f32>() / self.size as f32
        } else {
            0.0
        };

        let min_priority = self.priorities.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        ReplayBufferStats {
            size: self.size,
            capacity: self.config.capacity,
            total_added: self.total_added,
            avg_priority,
            min_priority,
            max_priority: self.max_priority,
            current_beta: self.current_beta(),
        }
    }

    // =========================================================================
    // Private helper methods
    // =========================================================================

    /// Compute sampling probabilities from priorities.
    fn compute_sampling_probabilities(&self) -> Vec<f32> {
        let mut probabilities: Vec<f32> = self
            .priorities
            .iter()
            .map(|&p| p.max(self.config.min_priority))
            .collect();

        // Normalize to sum to 1
        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            for prob in &mut probabilities {
                *prob /= sum;
            }
        } else {
            // Fallback to uniform if all priorities are zero
            let uniform = 1.0 / self.size as f32;
            probabilities.fill(uniform);
        }

        probabilities
    }

    /// Compute importance sampling weights to correct for bias.
    fn compute_is_weights(&self, indices: &[usize], probabilities: &[f32]) -> Vec<f32> {
        let beta = self.current_beta();
        let n = self.size as f32;

        // Find minimum probability for normalization
        let min_prob = probabilities.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        // Compute weights: (N * P(i))^(-beta)
        let max_weight = (n * min_prob).powf(-beta);

        indices
            .iter()
            .map(|&idx| {
                let prob = probabilities[idx];
                let weight = (n * prob).powf(-beta);
                // Normalize by max weight to keep weights in reasonable range
                weight / max_weight * prob / max_weight
            })
            .collect()
    }

    /// Sample a weighted random index from probabilities.
    fn weighted_sample(&mut self, probabilities: &[f32]) -> usize {
        let total: f32 = probabilities.iter().sum();
        let mut threshold = self.rng.random::<f32>() * total;

        for (idx, &prob) in probabilities.iter().enumerate() {
            threshold -= prob;
            if threshold <= 0.0 {
                return idx;
            }
        }

        // Fallback to last index if rounding errors occur
        probabilities.len().saturating_sub(1)
    }
}

/// Statistics about the replay buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBufferStats {
    /// Current size
    pub size: usize,

    /// Maximum capacity
    pub capacity: usize,

    /// Total experiences added (including overwrites)
    pub total_added: usize,

    /// Average priority
    pub avg_priority: f32,

    /// Minimum priority
    pub min_priority: f32,

    /// Maximum priority
    pub max_priority: f32,

    /// Current beta value (for IS weighting)
    pub current_beta: f32,
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Create a simple experience for testing or basic use cases.
pub fn create_experience<S, A>(
    state: S,
    action: A,
    reward: f32,
    next_state: S,
    done: bool,
) -> Experience<S, A> {
    Experience {
        state,
        action,
        reward,
        next_state,
        done,
        metadata: ExperienceMetadata::default(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_creation() {
        let config = ReplayBufferConfig::default();
        let buffer: PrioritizedReplayBuffer<Vec<f32>, usize> = PrioritizedReplayBuffer::new(config);

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_add_experience() {
        let config = ReplayBufferConfig {
            capacity: 10,
            ..Default::default()
        };
        let mut buffer = PrioritizedReplayBuffer::new(config);

        let exp = create_experience(vec![1.0], 0, 1.0, vec![2.0], false);
        buffer.add(exp);

        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_buffer_capacity() {
        let config = ReplayBufferConfig {
            capacity: 5,
            ..Default::default()
        };
        let mut buffer = PrioritizedReplayBuffer::new(config);

        // Add more than capacity
        for i in 0..10 {
            let exp = create_experience(vec![i as f32], i, 1.0, vec![i as f32 + 1.0], false);
            buffer.add(exp);
        }

        // Should only keep last 5
        assert_eq!(buffer.len(), 5);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_sample_batch() {
        let config = ReplayBufferConfig {
            capacity: 100,
            ..Default::default()
        };
        let mut buffer = PrioritizedReplayBuffer::new(config);

        // Add experiences
        for i in 0..50 {
            let exp = create_experience(vec![i as f32], i, 1.0, vec![i as f32 + 1.0], false);
            buffer.add(exp);
        }

        // Sample batch
        let batch = buffer.sample(10).unwrap();
        assert_eq!(batch.experiences.len(), 10);
        assert_eq!(batch.indices.len(), 10);
        assert_eq!(batch.is_weights.len(), 10);

        // All IS weights should be positive
        for weight in &batch.is_weights {
            assert!(*weight > 0.0);
        }
    }

    #[test]
    fn test_priority_update() {
        let config = ReplayBufferConfig {
            capacity: 100,
            alpha: 0.6,
            ..Default::default()
        };
        let mut buffer = PrioritizedReplayBuffer::new(config);

        // Add experiences
        for i in 0..10 {
            let exp = create_experience(vec![i as f32], i, 1.0, vec![i as f32 + 1.0], false);
            buffer.add(exp);
        }

        let initial_stats = buffer.stats();

        // Update priorities with varying TD errors
        let indices = vec![0, 1, 2];
        let td_errors = vec![10.0, 0.1, 5.0]; // High, low, medium
        buffer.update_priorities(&indices, &td_errors);

        let updated_stats = buffer.stats();

        // Max priority should have increased
        assert!(updated_stats.max_priority > initial_stats.max_priority);
    }

    #[test]
    fn test_swr_sampling() {
        let config = ReplayBufferConfig {
            capacity: 100,
            ..Default::default()
        };
        let mut buffer = PrioritizedReplayBuffer::new(config);

        // Add experiences with varying SWR importance
        for i in 0..50 {
            let mut exp = create_experience(vec![i as f32], i, 1.0, vec![i as f32 + 1.0], false);
            exp.metadata.swr_importance = Some(if i % 10 == 0 { 2.0 } else { 1.0 });
            buffer.add(exp);
        }

        // Sample with SWR bias
        let batch = buffer.sample_swr(10, 0.5).unwrap();
        assert_eq!(batch.experiences.len(), 10);
    }

    #[test]
    fn test_beta_annealing() {
        let config = ReplayBufferConfig {
            capacity: 100,
            beta: 0.4,
            beta_anneal_steps: 100,
            ..Default::default()
        };
        let mut buffer = PrioritizedReplayBuffer::new(config);

        let initial_beta = buffer.current_beta();
        assert!((initial_beta - 0.4).abs() < 1e-5);

        // Add experiences to increase total_added
        for i in 0..100 {
            let exp = create_experience(vec![i as f32], i, 1.0, vec![i as f32 + 1.0], false);
            buffer.add(exp);
        }

        let final_beta = buffer.current_beta();
        assert!(final_beta > initial_beta);
        assert!((final_beta - 1.0).abs() < 1e-5); // Should reach 1.0 after annealing
    }

    #[test]
    fn test_buffer_stats() {
        let config = ReplayBufferConfig {
            capacity: 10,
            ..Default::default()
        };
        let mut buffer = PrioritizedReplayBuffer::new(config);

        for i in 0..5 {
            let exp = create_experience(vec![i as f32], i, 1.0, vec![i as f32 + 1.0], false);
            buffer.add(exp);
        }

        let stats = buffer.stats();
        assert_eq!(stats.size, 5);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.total_added, 5);
        assert!(stats.avg_priority > 0.0);
    }

    #[test]
    fn test_clear_buffer() {
        let config = ReplayBufferConfig {
            capacity: 10,
            ..Default::default()
        };
        let mut buffer = PrioritizedReplayBuffer::new(config);

        for i in 0..5 {
            let exp = create_experience(vec![i as f32], i, 1.0, vec![i as f32 + 1.0], false);
            buffer.add(exp);
        }

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }
}
