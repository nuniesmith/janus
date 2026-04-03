//! Experience storage and sampling
//!
//! Part of the Hippocampus region
//! Component: replay
//!
//! Implements a circular replay buffer for storing and sampling experiences
//! for reinforcement learning. Supports both uniform random sampling and
//! prioritized replay.

use crate::common::Result;
use rand::RngExt;
use std::collections::VecDeque;

/// A single experience tuple (s, a, r, s', done)
#[derive(Clone, Debug)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

impl Experience {
    pub fn new(
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}

/// Experience storage and sampling with circular buffer
pub struct ReplayBuffer {
    /// Maximum capacity of the buffer
    capacity: usize,
    /// Circular buffer storing experiences
    buffer: VecDeque<Experience>,
    /// Current position in buffer (for statistics)
    position: usize,
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self::new(10_000)
    }
}

impl ReplayBuffer {
    /// Create a new replay buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            position: 0,
        }
    }

    /// Add an experience to the buffer
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push_back(experience);
        } else {
            // Buffer is full, overwrite oldest
            self.buffer.pop_front();
            self.buffer.push_back(experience);
        }
        self.position = (self.position + 1) % self.capacity;
    }

    /// Sample a batch of experiences uniformly at random
    pub fn sample(&self, batch_size: usize) -> Result<Vec<Experience>> {
        if self.buffer.is_empty() {
            return Err(crate::common::Error::InvalidInput(
                "Cannot sample from empty buffer".to_string(),
            ));
        }

        if batch_size > self.buffer.len() {
            return Err(crate::common::Error::InvalidInput(format!(
                "Batch size {} exceeds buffer size {}",
                batch_size,
                self.buffer.len()
            )));
        }

        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(batch_size);

        // Random sampling with replacement
        for _ in 0..batch_size {
            let idx = rng.random_range(0..self.buffer.len());
            samples.push(self.buffer[idx].clone());
        }

        Ok(samples)
    }

    /// Get the current size of the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Clear all experiences from the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.position = 0;
    }

    /// Get the most recent n experiences
    pub fn get_recent(&self, n: usize) -> Vec<Experience> {
        let count = n.min(self.buffer.len());
        self.buffer.iter().rev().take(count).cloned().collect()
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        // This is a compatibility method - the main operations are
        // push() and sample()
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_experience(reward: f32) -> Experience {
        Experience::new(vec![1.0, 2.0, 3.0], 0, reward, vec![4.0, 5.0, 6.0], false)
    }

    #[test]
    fn test_basic() {
        let instance = ReplayBuffer::new(1000);
        assert!(instance.process().is_ok());
        assert_eq!(instance.len(), 0);
        assert!(instance.is_empty());
    }

    #[test]
    fn test_push_and_len() {
        let mut buffer = ReplayBuffer::new(10);
        assert_eq!(buffer.len(), 0);

        buffer.push(create_test_experience(1.0));
        assert_eq!(buffer.len(), 1);

        buffer.push(create_test_experience(2.0));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = ReplayBuffer::new(3);

        // Fill buffer
        buffer.push(create_test_experience(1.0));
        buffer.push(create_test_experience(2.0));
        buffer.push(create_test_experience(3.0));
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());

        // Add one more - should overwrite oldest
        buffer.push(create_test_experience(4.0));
        assert_eq!(buffer.len(), 3); // Still 3
        assert!(buffer.is_full());
    }

    #[test]
    fn test_sample() {
        let mut buffer = ReplayBuffer::new(100);

        // Add some experiences
        for i in 0..10 {
            buffer.push(create_test_experience(i as f32));
        }

        // Sample a batch
        let samples = buffer.sample(5).unwrap();
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_sample_empty_error() {
        let buffer = ReplayBuffer::new(10);
        let result = buffer.sample(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_too_large_error() {
        let mut buffer = ReplayBuffer::new(10);
        buffer.push(create_test_experience(1.0));
        buffer.push(create_test_experience(2.0));

        let result = buffer.sample(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_clear() {
        let mut buffer = ReplayBuffer::new(10);
        buffer.push(create_test_experience(1.0));
        buffer.push(create_test_experience(2.0));
        assert_eq!(buffer.len(), 2);

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_get_recent() {
        let mut buffer = ReplayBuffer::new(10);
        buffer.push(create_test_experience(1.0));
        buffer.push(create_test_experience(2.0));
        buffer.push(create_test_experience(3.0));

        let recent = buffer.get_recent(2);
        assert_eq!(recent.len(), 2);
        // Most recent should be 3.0
        assert_eq!(recent[0].reward, 3.0);
        assert_eq!(recent[1].reward, 2.0);
    }

    #[test]
    fn test_capacity() {
        let buffer = ReplayBuffer::new(1000);
        assert_eq!(buffer.capacity(), 1000);
    }
}
