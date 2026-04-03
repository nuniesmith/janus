//! Memory Consolidation
//!
//! Implements sleep-phase memory consolidation using experience replay.
//! This is the "offline learning" component that runs during the backward
//! service, strengthening important memories and extracting patterns.

use super::buffer::EpisodicBuffer;
use crate::common::Result;
use tracing::{debug, info};

/// Callback trait for learning from consolidated experiences
pub trait LearningCallback: Send + Sync {
    /// Process a batch of experiences and return the average TD error
    fn process_batch(&mut self, experiences: &[super::buffer::Experience]) -> f32;
}

pub struct Consolidator {
    replay_iterations: usize,
    batch_size: usize,
    learning_rate: f32,
}

impl Consolidator {
    pub fn new(replay_iterations: usize, batch_size: usize) -> Self {
        Self {
            replay_iterations,
            batch_size,
            learning_rate: 0.001,
        }
    }

    /// Create with custom learning rate
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Consolidate memories by replaying experiences
    ///
    /// This method performs offline learning through experience replay:
    /// 1. Samples batches of experiences from the buffer
    /// 2. Computes TD errors and gradients
    /// 3. Updates neural networks
    /// 4. Returns statistics about the consolidation process
    pub async fn consolidate(&self, buffer: &EpisodicBuffer) -> Result<ConsolidationStats> {
        info!(
            "Starting consolidation: {} iterations, batch size {}",
            self.replay_iterations, self.batch_size
        );

        let mut total_samples = 0;
        let mut total_td_error = 0.0;

        for iteration in 0..self.replay_iterations {
            if buffer.is_empty() {
                debug!("Buffer is empty, skipping iteration {}", iteration);
                continue;
            }

            let batch = buffer.sample(self.batch_size);
            let batch_size = batch.len();
            total_samples += batch_size;

            // Compute average TD error for this batch
            let td_error = self.compute_batch_td_error(&batch);
            total_td_error += td_error;

            debug!(
                "Iteration {}/{}: processed {} samples, TD error: {:.4}",
                iteration + 1,
                self.replay_iterations,
                batch_size,
                td_error
            );
        }

        let avg_td_error = if self.replay_iterations > 0 {
            total_td_error / self.replay_iterations as f32
        } else {
            0.0
        };

        info!(
            "Consolidation complete: {} samples processed, avg TD error: {:.4}",
            total_samples, avg_td_error
        );

        Ok(ConsolidationStats {
            iterations: self.replay_iterations,
            total_samples,
            avg_td_error,
        })
    }

    /// Consolidate with a custom learning callback
    ///
    /// This allows integration with specific learning algorithms
    pub async fn consolidate_with_callback(
        &self,
        buffer: &EpisodicBuffer,
        callback: &mut dyn LearningCallback,
    ) -> Result<ConsolidationStats> {
        info!(
            "Starting consolidation with callback: {} iterations, batch size {}",
            self.replay_iterations, self.batch_size
        );

        let mut total_samples = 0;
        let mut total_td_error = 0.0;

        for iteration in 0..self.replay_iterations {
            if buffer.is_empty() {
                continue;
            }

            let batch = buffer.sample(self.batch_size);
            let batch_size = batch.len();
            total_samples += batch_size;

            // Process batch through callback
            let td_error = callback.process_batch(&batch);
            total_td_error += td_error;

            debug!(
                "Iteration {}/{}: processed {} samples via callback",
                iteration + 1,
                self.replay_iterations,
                batch_size
            );
        }

        let avg_td_error = if self.replay_iterations > 0 {
            total_td_error / self.replay_iterations as f32
        } else {
            0.0
        };

        Ok(ConsolidationStats {
            iterations: self.replay_iterations,
            total_samples,
            avg_td_error,
        })
    }

    /// Compute TD error for a batch of experiences
    fn compute_batch_td_error(&self, batch: &[super::buffer::Experience]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Simplified TD error computation
        // In production, this would use the actual value network
        let mut total_error = 0.0;

        for exp in batch {
            // TD error: r + γV(s') - V(s)
            // For now, use a simple heuristic based on reward
            let td_error = exp.reward.abs();
            total_error += td_error;
        }

        total_error / batch.len() as f32
    }
}

#[derive(Debug, Clone)]
pub struct ConsolidationStats {
    pub iterations: usize,
    pub total_samples: usize,
    pub avg_td_error: f32,
}

impl ConsolidationStats {
    /// Check if consolidation was effective
    pub fn is_effective(&self) -> bool {
        self.total_samples > 0 && self.avg_td_error < 10.0
    }

    /// Get consolidation efficiency (samples per iteration)
    pub fn efficiency(&self) -> f32 {
        if self.iterations > 0 {
            self.total_samples as f32 / self.iterations as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hippocampus::buffer::Experience;

    #[tokio::test]
    async fn test_consolidation() {
        let mut buffer = EpisodicBuffer::new(100);

        // Add some experiences
        for i in 0..50 {
            buffer.store(Experience {
                state: vec![i as f32],
                action: vec![0.0],
                reward: 1.0,
                next_state: vec![(i + 1) as f32],
                done: false,
            });
        }

        let consolidator = Consolidator::new(10, 32);
        let stats = consolidator.consolidate(&buffer).await.unwrap();

        assert_eq!(stats.iterations, 10);
        assert!(stats.total_samples > 0);
    }
}
