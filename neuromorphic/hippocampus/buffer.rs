//! Episodic Memory Buffer
//!
//! Stores trading experiences (state, action, reward, next_state) for replay.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

#[derive(Debug)]
pub struct EpisodicBuffer {
    capacity: usize,
    buffer: VecDeque<Experience>,
}

impl EpisodicBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    pub fn store(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();

        let mut experiences: Vec<_> = self.buffer.iter().cloned().collect();
        experiences.shuffle(&mut rng);
        experiences.truncate(batch_size.min(self.buffer.len()));
        experiences
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_store_and_sample() {
        let mut buffer = EpisodicBuffer::new(10);

        // Store experiences
        for i in 0..5 {
            buffer.store(Experience {
                state: vec![i as f32],
                action: vec![0.0],
                reward: 1.0,
                next_state: vec![(i + 1) as f32],
                done: false,
            });
        }

        assert_eq!(buffer.len(), 5);

        // Sample
        let batch = buffer.sample(3);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_buffer_capacity() {
        let mut buffer = EpisodicBuffer::new(3);

        for i in 0..5 {
            buffer.store(Experience {
                state: vec![i as f32],
                action: vec![0.0],
                reward: 1.0,
                next_state: vec![(i + 1) as f32],
                done: false,
            });
        }

        // Should only keep last 3
        assert_eq!(buffer.len(), 3);
    }
}
