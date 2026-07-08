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

    /// Read-only view of the most recently written experiences, returned in
    /// insertion order (oldest-first, newest-last).
    ///
    /// Returns at most `min(n, len())` items and correctly honours the ring
    /// wrap once the buffer is full. Used to greedily evaluate a freshly
    /// trained model on a recent slice of the buffer.
    pub fn recent(&self, n: usize) -> Vec<&Experience> {
        let take = n.min(self.size);
        let mut out = Vec::with_capacity(take);
        // Walk from the oldest of the last `take` writes to the newest,
        // mapping each position through the ring so wraparound is respected.
        for i in (0..take).rev() {
            let idx = (self.write_index + self.capacity - 1 - i) % self.capacity;
            out.push(&self.data[idx]);
        }
        out
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

    /// Build an experience whose `reward` encodes its identity, so `recent`
    /// ordering can be asserted precisely.
    fn exp_with_reward(reward: f32) -> Experience {
        let mut exp = create_test_experience();
        exp.reward = reward;
        exp
    }

    #[test]
    fn test_recent_returns_newest_last_no_wrap() {
        let mut tree = SumTree::new(100, 0.6, 0.4);
        for r in 0..5 {
            tree.add(exp_with_reward(r as f32), 1.0);
        }
        // Oldest-first, newest-last over the last 3 writes.
        let last3: Vec<f32> = tree.recent(3).iter().map(|e| e.reward).collect();
        assert_eq!(last3, vec![2.0, 3.0, 4.0]);
        // Requesting more than present clamps to size.
        assert_eq!(tree.recent(100).len(), 5);
        // Zero-length slice is empty.
        assert!(tree.recent(0).is_empty());
    }

    #[test]
    fn test_recent_honours_ring_wrap() {
        // Capacity 3, six writes → the buffer holds only the last three
        // (rewards 3, 4, 5) and `recent` must return them in insertion order.
        let mut tree = SumTree::new(3, 0.6, 0.4);
        for r in 0..6 {
            tree.add(exp_with_reward(r as f32), 1.0);
        }
        assert_eq!(tree.len(), 3);
        let all: Vec<f32> = tree.recent(3).iter().map(|e| e.reward).collect();
        assert_eq!(all, vec![3.0, 4.0, 5.0]);
        let last2: Vec<f32> = tree.recent(2).iter().map(|e| e.reward).collect();
        assert_eq!(last2, vec![4.0, 5.0]);
    }
}
