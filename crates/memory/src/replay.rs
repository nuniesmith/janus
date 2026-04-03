//! Prioritized Experience Replay (PER) implementation.
//!
//! Uses a Sum-Tree data structure for O(log N) sampling and updates.

use common::{Experience, JanusError, Result};

/// Sum-Tree for Prioritized Experience Replay
///
/// This is a flat Vec-based implementation for O(log N) operations.
/// The tree is stored as a complete binary tree in a Vec.
pub struct SumTree {
    capacity: usize,
    tree: Vec<f64>,        // Sum tree
    data: Vec<Experience>, // Experience data
    write_index: usize,
    size: usize,
    alpha: f64, // Priority exponent (0 = uniform, 1 = fully prioritized)
    beta: f64,  // Importance sampling exponent
}

impl SumTree {
    /// Create a new SumTree with given capacity
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        // For a complete binary tree with capacity leaf nodes,
        // we need capacity - 1 internal nodes + capacity leaf nodes = 2 * capacity - 1 total
        let tree_size = 2 * capacity - 1;

        Self {
            capacity,
            tree: vec![0.0; tree_size],
            data: Vec::with_capacity(capacity),
            write_index: 0,
            size: 0,
            alpha,
            beta,
        }
    }

    /// Add an experience with priority
    pub fn add(&mut self, experience: Experience, priority: f64) {
        let idx = self.write_index;

        // Update data
        if idx < self.data.len() {
            self.data[idx] = experience;
        } else {
            self.data.push(experience);
        }

        // Update priority (TD error raised to alpha)
        let priority = priority.max(1e-6).powf(self.alpha);

        // Update tree
        self.update_tree(idx, priority);

        // Advance write index
        self.write_index = (self.write_index + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// Sample a batch of experiences
    pub fn sample(&self, batch_size: usize) -> Result<Vec<(Experience, usize, f64)>> {
        if self.size == 0 {
            return Err(JanusError::Memory(
                "Cannot sample from empty buffer".to_string(),
            ));
        }

        let total_priority = self.tree[0];
        let segment_size = total_priority / (batch_size as f64);

        let mut samples = Vec::with_capacity(batch_size);

        use rand::RngExt;
        let mut rng = rand::rng();

        for i in 0..batch_size {
            let value = (i as f64 + rng.random::<f64>()) * segment_size;
            let (idx, priority) = self.retrieve(value, 0);

            if idx < self.size {
                let experience = self.data[idx].clone();
                let prob = priority / total_priority;
                let weight = (self.size as f64 * prob).powf(-self.beta);

                samples.push((experience, idx, weight));
            }
        }

        Ok(samples)
    }

    /// Update priority for an experience
    pub fn update_priority(&mut self, idx: usize, priority: f64) {
        let priority = priority.max(1e-6).powf(self.alpha);
        self.update_tree(idx, priority);
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Update tree node (recursive)
    fn update_tree(&mut self, idx: usize, priority: f64) {
        let tree_idx = self.capacity - 1 + idx;
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;
        self.propagate_change(tree_idx, change);
    }

    /// Propagate change up the tree
    fn propagate_change(&mut self, mut idx: usize, change: f64) {
        while idx != 0 {
            idx = (idx - 1) / 2;
            self.tree[idx] += change;
        }
    }

    /// Retrieve experience index and priority from tree
    fn retrieve(&self, value: f64, idx: usize) -> (usize, f64) {
        let left = 2 * idx + 1;
        let right = left + 1;

        if left >= self.tree.len() {
            // Leaf node
            let data_idx = idx - (self.capacity - 1);
            return (data_idx, self.tree[idx]);
        }

        if value <= self.tree[left] {
            self.retrieve(value, left)
        } else {
            self.retrieve(value - self.tree[left], right)
        }
    }
}

// Add rand dependency for sampling
#[cfg(test)]
mod tests {
    use super::*;
    use common::{Action, ActionType, State, StateMetadata};

    fn create_test_experience() -> Experience {
        let metadata = StateMetadata::new("BTCUSD".to_string());
        let state = State::from_flat_gaf(vec![0.5_f32, 0.3_f32, 0.8_f32], vec![], metadata.clone());
        let next_state = State::from_flat_gaf(vec![0.6_f32, 0.4_f32, 0.9_f32], vec![], metadata);
        let action = Action::new(ActionType::Buy, "BTCUSD".to_string(), 1.0);

        Experience::new(state, action, 0.1, next_state, false)
    }

    #[test]
    fn test_sum_tree_add() {
        let mut tree = SumTree::new(100, 0.6, 0.4);
        let exp = create_test_experience();
        tree.add(exp, 1.0);
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_sum_tree_sample() {
        let mut tree = SumTree::new(100, 0.6, 0.4);
        for _ in 0..10 {
            let exp = create_test_experience();
            tree.add(exp, 1.0);
        }
        let samples = tree.sample(5).unwrap();
        assert_eq!(samples.len(), 5);
    }
}
