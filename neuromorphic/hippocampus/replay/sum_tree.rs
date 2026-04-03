//! Sum Tree Data Structure for Efficient Prioritized Sampling
//!
//! Part of the Hippocampus region
//! Component: replay
//!
//! A sum tree is a binary tree where each parent node's value is the sum of its
//! children's values. This allows O(log n) sampling proportional to priorities
//! and O(log n) priority updates.
//!
//! Used in Prioritized Experience Replay (PER) to efficiently sample experiences
//! proportional to their TD-error priorities.

use crate::common::{Error, Result};

/// Sum Tree for efficient priority-based sampling
///
/// The tree is stored as a flat array where:
/// - Leaf nodes (priorities) are at indices [capacity-1, 2*capacity-1)
/// - Internal nodes store sums at indices [0, capacity-1)
/// - Parent of node i is at (i-1)/2
/// - Children of node i are at 2*i+1 and 2*i+2
#[derive(Debug, Clone)]
pub struct SumTree {
    /// Capacity (number of leaf nodes / experiences)
    capacity: usize,
    /// Tree storage: internal nodes + leaf nodes
    /// Size is 2*capacity - 1 for a complete binary tree
    tree: Vec<f64>,
    /// Data indices stored at leaf positions
    data_pointer: usize,
    /// Minimum priority value (for importance sampling correction)
    min_priority: f64,
    /// Maximum priority seen (for normalization)
    max_priority: f64,
    /// Small constant to ensure non-zero priorities
    epsilon: f64,
}

impl Default for SumTree {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl SumTree {
    /// Create a new sum tree with given capacity
    pub fn new(capacity: usize) -> Self {
        // Tree size is 2 * capacity - 1 for a complete binary tree
        // Leaves at [capacity-1, 2*capacity-1)
        let tree_size = 2 * capacity - 1;
        Self {
            capacity,
            tree: vec![0.0; tree_size],
            data_pointer: 0,
            min_priority: f64::MAX,
            max_priority: 0.0,
            epsilon: 1e-6,
        }
    }

    /// Create with custom epsilon value
    pub fn with_epsilon(capacity: usize, epsilon: f64) -> Self {
        let mut tree = Self::new(capacity);
        tree.epsilon = epsilon;
        tree
    }

    /// Get the total priority sum (root of tree)
    pub fn total(&self) -> f64 {
        self.tree[0]
    }

    /// Get the minimum priority stored
    pub fn min(&self) -> f64 {
        if self.min_priority == f64::MAX {
            0.0
        } else {
            self.min_priority
        }
    }

    /// Get the maximum priority stored
    pub fn max(&self) -> f64 {
        self.max_priority
    }

    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the leaf start index
    fn leaf_start(&self) -> usize {
        self.capacity - 1
    }

    /// Add a new priority at the next available slot
    /// Returns the data index where the priority was stored
    pub fn add(&mut self, priority: f64) -> usize {
        let priority = priority.max(self.epsilon);
        let idx = self.data_pointer;
        self.update(idx, priority);
        self.data_pointer = (self.data_pointer + 1) % self.capacity;
        idx
    }

    /// Update priority at a specific data index
    pub fn update(&mut self, data_idx: usize, priority: f64) {
        if data_idx >= self.capacity {
            return;
        }

        let priority = priority.max(self.epsilon);

        // Update min/max tracking
        if priority < self.min_priority {
            self.min_priority = priority;
        }
        if priority > self.max_priority {
            self.max_priority = priority;
        }

        // Calculate tree index (leaf node position)
        // Leaves are at indices [capacity-1, 2*capacity-1)
        let tree_idx = data_idx + self.leaf_start();

        // Calculate change in priority
        let change = priority - self.tree[tree_idx];

        // Update the leaf
        self.tree[tree_idx] = priority;

        // Propagate change up the tree
        self.propagate(tree_idx, change);
    }

    /// Propagate a change up the tree from a leaf node
    fn propagate(&mut self, mut tree_idx: usize, change: f64) {
        while tree_idx > 0 {
            tree_idx = (tree_idx - 1) / 2;
            self.tree[tree_idx] += change;
        }
    }

    /// Sample a data index proportional to priorities
    ///
    /// Given a value in [0, total), returns the data index of the
    /// leaf node whose cumulative priority range contains that value.
    pub fn sample(&self, value: f64) -> usize {
        let mut value = value.clamp(0.0, self.total() - self.epsilon);
        let mut idx = 0;
        let leaf_start = self.leaf_start();

        // Traverse down the tree until we reach a leaf node
        while idx < leaf_start {
            let left = 2 * idx + 1;
            let right = left + 1;

            if left >= self.tree.len() {
                break;
            }

            if value <= self.tree[left] {
                idx = left;
            } else {
                value -= self.tree[left];
                if right < self.tree.len() {
                    idx = right;
                } else {
                    break;
                }
            }
        }

        // Convert tree index to data index
        // Leaves are at [leaf_start, 2*capacity-1), so subtract leaf_start
        idx.saturating_sub(leaf_start)
    }

    /// Sample multiple data indices with stratified sampling
    ///
    /// Divides the total priority range into n equal segments and
    /// samples one index from each segment. This provides more diverse
    /// samples than pure random sampling.
    pub fn sample_batch(&self, batch_size: usize) -> Vec<usize> {
        if self.total() <= 0.0 {
            return vec![0; batch_size];
        }

        let segment_size = self.total() / batch_size as f64;
        let mut indices = Vec::with_capacity(batch_size);
        let mut rng = rand::rng();

        for i in 0..batch_size {
            let low = segment_size * i as f64;
            let high = segment_size * (i + 1) as f64;
            let value = low + rand::RngExt::random::<f64>(&mut rng) * (high - low);
            indices.push(self.sample(value));
        }

        indices
    }

    /// Get the priority at a specific data index
    pub fn get_priority(&self, data_idx: usize) -> f64 {
        if data_idx >= self.capacity {
            0.0
        } else {
            self.tree[data_idx + self.leaf_start()]
        }
    }

    /// Get all priorities as a slice
    pub fn priorities(&self) -> &[f64] {
        &self.tree[self.leaf_start()..]
    }

    /// Calculate importance sampling weight for a data index
    ///
    /// Weight = (1/N * 1/P(i))^beta where P(i) = priority(i) / total
    /// Normalized by max weight to keep values in [0, 1]
    pub fn importance_weight(&self, data_idx: usize, beta: f64, n: usize) -> f64 {
        let priority = self.get_priority(data_idx);
        let total = self.total();

        if total <= 0.0 || priority <= 0.0 || n == 0 {
            return 1.0;
        }

        // P(i) = priority / total
        let prob = priority / total;

        // Weight = (N * P(i))^(-beta)
        let weight = (n as f64 * prob).powf(-beta);

        // Normalize by max weight (occurs at min priority)
        let min_prob = self.min() / total;
        let max_weight = (n as f64 * min_prob).powf(-beta);

        if max_weight > 0.0 {
            weight / max_weight
        } else {
            1.0
        }
    }

    /// Reset the tree to empty state
    pub fn clear(&mut self) {
        self.tree.fill(0.0);
        self.data_pointer = 0;
        self.min_priority = f64::MAX;
        self.max_priority = 0.0;
    }

    /// Check if tree has any non-zero priorities
    pub fn is_empty(&self) -> bool {
        self.total() <= self.epsilon
    }

    /// Get number of non-zero priority entries
    pub fn count_nonzero(&self) -> usize {
        self.tree[self.leaf_start()..]
            .iter()
            .filter(|&&p| p > self.epsilon)
            .count()
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Validate tree invariants (for debugging)
    pub fn validate(&self) -> Result<()> {
        let leaf_start = self.leaf_start();

        // Check that each internal node equals sum of children
        for i in 0..leaf_start {
            let left = 2 * i + 1;
            let right = left + 1;

            if right < self.tree.len() {
                let expected = self.tree[left] + self.tree[right];
                let actual = self.tree[i];
                let diff = (expected - actual).abs();

                if diff > 1e-10 {
                    return Err(Error::InvalidState(format!(
                        "Tree invariant violated at index {}: expected {}, got {}",
                        i, expected, actual
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = SumTree::new(16);
        assert!(instance.process().is_ok());
        assert_eq!(instance.total(), 0.0);
        assert!(instance.is_empty());
    }

    #[test]
    fn test_add_and_total() {
        let mut tree = SumTree::new(8);

        tree.add(1.0);
        assert!((tree.total() - 1.0).abs() < 1e-10);

        tree.add(2.0);
        assert!((tree.total() - 3.0).abs() < 1e-10);

        tree.add(3.0);
        assert!((tree.total() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_update() {
        let mut tree = SumTree::new(4);

        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);
        assert!((tree.total() - 6.0).abs() < 1e-10);

        // Update index 1 from 2.0 to 5.0
        tree.update(1, 5.0);
        assert!((tree.total() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_priority() {
        let mut tree = SumTree::new(4);

        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);

        assert!((tree.get_priority(0) - 1.0).abs() < 1e-10);
        assert!((tree.get_priority(1) - 2.0).abs() < 1e-10);
        assert!((tree.get_priority(2) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_proportional() {
        let mut tree = SumTree::new(4);

        // Add priorities: 1, 2, 3, 4 (total = 10)
        tree.add(1.0); // idx 0: [0, 1)
        tree.add(2.0); // idx 1: [1, 3)
        tree.add(3.0); // idx 2: [3, 6)
        tree.add(4.0); // idx 3: [6, 10)

        // Sample from different ranges
        assert_eq!(tree.sample(0.5), 0); // In [0, 1)
        assert_eq!(tree.sample(2.0), 1); // In [1, 3)
        assert_eq!(tree.sample(4.0), 2); // In [3, 6)
        assert_eq!(tree.sample(8.0), 3); // In [6, 10)
    }

    #[test]
    fn test_sample_batch() {
        let mut tree = SumTree::new(16);

        for i in 1..=10 {
            tree.add(i as f64);
        }

        let batch = tree.sample_batch(5);
        assert_eq!(batch.len(), 5);

        // All indices should be valid
        for &idx in &batch {
            assert!(idx < tree.capacity());
        }
    }

    #[test]
    fn test_min_max_tracking() {
        let mut tree = SumTree::new(8);

        tree.add(5.0);
        assert!((tree.min() - 5.0).abs() < 1e-10);
        assert!((tree.max() - 5.0).abs() < 1e-10);

        tree.add(2.0);
        assert!((tree.min() - 2.0).abs() < 1e-10);
        assert!((tree.max() - 5.0).abs() < 1e-10);

        tree.add(10.0);
        assert!((tree.min() - 2.0).abs() < 1e-10);
        assert!((tree.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_importance_weight() {
        let mut tree = SumTree::new(4);

        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);

        // With beta=1, weights should correct for sampling probability
        let w0 = tree.importance_weight(0, 1.0, 3);
        let w1 = tree.importance_weight(1, 1.0, 3);
        let w2 = tree.importance_weight(2, 1.0, 3);

        // Lower priority should have higher weight
        assert!(w0 > w1);
        assert!(w1 > w2);
    }

    #[test]
    fn test_clear() {
        let mut tree = SumTree::new(4);

        tree.add(1.0);
        tree.add(2.0);
        assert!((tree.total() - 3.0).abs() < 1e-10);

        tree.clear();
        assert!(tree.is_empty());
        assert_eq!(tree.total(), 0.0);
    }

    #[test]
    fn test_circular_overwrite() {
        let mut tree = SumTree::new(3);

        // Fill the tree
        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);
        assert!((tree.total() - 6.0).abs() < 1e-10);

        // Add more, should overwrite
        tree.add(10.0); // Overwrites index 0
        // Now: 10, 2, 3
        assert!((tree.total() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate() {
        let mut tree = SumTree::new(8);

        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);
        tree.add(4.0);

        // Tree should be valid after operations
        assert!(tree.validate().is_ok());
    }

    #[test]
    fn test_count_nonzero() {
        let mut tree = SumTree::new(8);

        assert_eq!(tree.count_nonzero(), 0);

        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);

        assert_eq!(tree.count_nonzero(), 3);
    }

    #[test]
    fn test_epsilon_prevents_zero() {
        let mut tree = SumTree::with_epsilon(4, 1e-5);

        // Adding 0 should be clamped to epsilon
        tree.add(0.0);
        assert!(tree.get_priority(0) > 0.0);
    }

    #[test]
    fn test_capacity() {
        let tree = SumTree::new(1024);
        assert_eq!(tree.capacity(), 1024);
    }

    #[test]
    fn test_priorities_slice() {
        let mut tree = SumTree::new(4);

        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);

        let priorities = tree.priorities();
        assert_eq!(priorities.len(), 4);
        assert!((priorities[0] - 1.0).abs() < 1e-10);
        assert!((priorities[1] - 2.0).abs() < 1e-10);
        assert!((priorities[2] - 3.0).abs() < 1e-10);
    }
}
