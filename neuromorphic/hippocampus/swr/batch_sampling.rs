//! Batch experience sampling
//!
//! Part of the Hippocampus region
//! Component: swr
//!
//! This module implements sophisticated batch sampling strategies for
//! experience replay, including prioritized sampling, stratified batching,
//! and importance sampling correction.

use crate::common::{Error, Result};
use rand::RngExt;
use std::collections::HashMap;

/// Sampling strategy for batch creation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Uniform random sampling
    Uniform,
    /// Prioritized sampling based on TD error or other priority
    Prioritized,
    /// Stratified sampling across categories
    Stratified,
    /// Recency-weighted sampling (more recent = higher probability)
    RecencyWeighted,
    /// Diverse sampling (maximize batch diversity)
    Diverse,
    /// Curriculum sampling (easy to hard)
    Curriculum,
    /// Hindsight sampling (focus on goal-related experiences)
    Hindsight,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::Prioritized
    }
}

/// Experience entry for sampling
#[derive(Debug, Clone)]
pub struct SampleExperience {
    /// Unique identifier
    pub id: u64,
    /// State representation
    pub state: Vec<f32>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f32,
    /// Next state
    pub next_state: Vec<f32>,
    /// Whether episode ended
    pub done: bool,
    /// Priority for sampling
    pub priority: f64,
    /// Category/stratum for stratified sampling
    pub category: Option<String>,
    /// Timestamp of experience
    pub timestamp: u64,
    /// Difficulty score (for curriculum)
    pub difficulty: f64,
    /// Times this experience was sampled
    pub sample_count: usize,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl SampleExperience {
    pub fn new(
        id: u64,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) -> Self {
        Self {
            id,
            state,
            action,
            reward,
            next_state,
            done,
            priority: 1.0,
            category: None,
            timestamp: 0,
            difficulty: 0.5,
            sample_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority;
        self
    }

    /// Set category
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = Some(category.to_string());
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set difficulty
    pub fn with_difficulty(mut self, difficulty: f64) -> Self {
        self.difficulty = difficulty.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: f64) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }
}

/// Sampled batch with importance weights
#[derive(Debug, Clone)]
pub struct SampledBatch {
    /// Experiences in the batch
    pub experiences: Vec<SampleExperience>,
    /// Importance sampling weights (for bias correction)
    pub weights: Vec<f64>,
    /// Indices in original buffer
    pub indices: Vec<usize>,
    /// Sampling strategy used
    pub strategy: SamplingStrategy,
    /// Average priority of batch
    pub avg_priority: f64,
    /// Category distribution in batch
    pub category_distribution: HashMap<String, usize>,
}

impl SampledBatch {
    pub fn new(strategy: SamplingStrategy) -> Self {
        Self {
            experiences: Vec::new(),
            weights: Vec::new(),
            indices: Vec::new(),
            strategy,
            avg_priority: 0.0,
            category_distribution: HashMap::new(),
        }
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }

    /// Get states as matrix (batch_size x state_dim)
    pub fn states(&self) -> Vec<&Vec<f32>> {
        self.experiences.iter().map(|e| &e.state).collect()
    }

    /// Get actions
    pub fn actions(&self) -> Vec<usize> {
        self.experiences.iter().map(|e| e.action).collect()
    }

    /// Get rewards
    pub fn rewards(&self) -> Vec<f32> {
        self.experiences.iter().map(|e| e.reward).collect()
    }

    /// Get next states
    pub fn next_states(&self) -> Vec<&Vec<f32>> {
        self.experiences.iter().map(|e| &e.next_state).collect()
    }

    /// Get done flags
    pub fn dones(&self) -> Vec<bool> {
        self.experiences.iter().map(|e| e.done).collect()
    }
}

/// Configuration for batch sampling
#[derive(Debug, Clone)]
pub struct BatchSamplingConfig {
    /// Default batch size
    pub batch_size: usize,
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Priority exponent (alpha) for prioritized sampling
    pub priority_alpha: f64,
    /// Importance sampling exponent (beta) for bias correction
    pub importance_beta: f64,
    /// Beta annealing rate (increase beta over time)
    pub beta_annealing_rate: f64,
    /// Maximum beta value
    pub max_beta: f64,
    /// Recency decay rate (for recency-weighted sampling)
    pub recency_decay: f64,
    /// Minimum priority (prevents zero probability)
    pub min_priority: f64,
    /// Whether to sample with replacement
    pub with_replacement: bool,
    /// Number of strata for stratified sampling
    pub num_strata: usize,
    /// Diversity threshold (minimum pairwise distance)
    pub diversity_threshold: f64,
    /// Curriculum progress (0-1, affects difficulty weighting)
    pub curriculum_progress: f64,
}

impl Default for BatchSamplingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            strategy: SamplingStrategy::Prioritized,
            priority_alpha: 0.6,
            importance_beta: 0.4,
            beta_annealing_rate: 0.001,
            max_beta: 1.0,
            recency_decay: 0.99,
            min_priority: 1e-6,
            with_replacement: false,
            num_strata: 4,
            diversity_threshold: 0.1,
            curriculum_progress: 0.0,
        }
    }
}

/// Sum tree for efficient prioritized sampling
#[derive(Debug, Clone)]
pub struct SumTree {
    /// Tree storage (priorities)
    tree: Vec<f64>,
    /// Capacity (number of leaf nodes)
    capacity: usize,
    /// Current write position
    position: usize,
    /// Number of entries
    size: usize,
}

impl SumTree {
    pub fn new(capacity: usize) -> Self {
        // Tree size is 2 * capacity - 1 for a complete binary tree
        let tree_size = 2 * capacity - 1;
        Self {
            tree: vec![0.0; tree_size],
            capacity,
            position: 0,
            size: 0,
        }
    }

    /// Add a priority value
    pub fn add(&mut self, priority: f64) {
        let tree_index = self.position + self.capacity - 1;
        self.update(tree_index, priority);

        self.position = (self.position + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// Update priority at tree index
    pub fn update(&mut self, tree_index: usize, priority: f64) {
        let change = priority - self.tree[tree_index];
        self.tree[tree_index] = priority;

        // Propagate change up the tree
        let mut idx = tree_index;
        while idx > 0 {
            idx = (idx - 1) / 2;
            self.tree[idx] += change;
        }
    }

    /// Update priority at data index
    pub fn update_at(&mut self, data_index: usize, priority: f64) {
        let tree_index = data_index + self.capacity - 1;
        self.update(tree_index, priority);
    }

    /// Get sample by priority value (returns data index)
    pub fn get(&self, value: f64) -> usize {
        if self.size == 0 {
            return 0;
        }

        let mut value = value;
        let mut idx = 0;
        let leaf_start = self.capacity - 1;

        // Traverse down to leaf level
        while idx < leaf_start {
            let left = 2 * idx + 1;
            let right = left + 1;

            // Bounds check for safety
            if left >= self.tree.len() {
                break;
            }

            if value <= self.tree[left] {
                idx = left;
            } else {
                value -= self.tree[left];
                idx = right.min(self.tree.len() - 1);
            }
        }

        // Convert tree index to data index, clamping to valid range
        idx.saturating_sub(leaf_start)
            .min(self.size.saturating_sub(1))
    }

    /// Get total priority sum
    pub fn total(&self) -> f64 {
        self.tree[0]
    }

    /// Get minimum priority among leaves
    pub fn min(&self) -> f64 {
        let leaf_start = self.capacity - 1;
        self.tree[leaf_start..leaf_start + self.size]
            .iter()
            .cloned()
            .filter(|&x| x > 0.0)
            .fold(f64::INFINITY, f64::min)
    }

    /// Get priority at data index
    pub fn get_priority(&self, data_index: usize) -> f64 {
        self.tree[data_index + self.capacity - 1]
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// Batch experience sampling
pub struct BatchSampling {
    /// Configuration
    config: BatchSamplingConfig,
    /// Experience buffer
    buffer: Vec<SampleExperience>,
    /// Priority sum tree for efficient sampling
    sum_tree: SumTree,
    /// Maximum buffer size
    max_size: usize,
    /// Current position in circular buffer
    position: usize,
    /// Category indices for stratified sampling
    category_indices: HashMap<String, Vec<usize>>,
    /// Current importance sampling beta
    current_beta: f64,
    /// Total samples drawn
    total_samples: usize,
    /// Experience counter
    experience_counter: u64,
}

impl Default for BatchSampling {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchSampling {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Create with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            config: BatchSamplingConfig::default(),
            buffer: Vec::with_capacity(capacity),
            sum_tree: SumTree::new(capacity),
            max_size: capacity,
            position: 0,
            category_indices: HashMap::new(),
            current_beta: BatchSamplingConfig::default().importance_beta,
            total_samples: 0,
            experience_counter: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BatchSamplingConfig, capacity: usize) -> Self {
        let current_beta = config.importance_beta;
        Self {
            config,
            buffer: Vec::with_capacity(capacity),
            sum_tree: SumTree::new(capacity),
            max_size: capacity,
            position: 0,
            category_indices: HashMap::new(),
            current_beta,
            total_samples: 0,
            experience_counter: 0,
        }
    }

    /// Add an experience to the buffer
    pub fn add(&mut self, mut experience: SampleExperience) {
        self.experience_counter += 1;
        experience.id = self.experience_counter;

        // Apply priority transformation
        let priority =
            (experience.priority + self.config.min_priority).powf(self.config.priority_alpha);

        if self.buffer.len() < self.max_size {
            // Update category indices
            if let Some(ref cat) = experience.category {
                self.category_indices
                    .entry(cat.clone())
                    .or_insert_with(Vec::new)
                    .push(self.buffer.len());
            }

            self.buffer.push(experience);
            self.sum_tree.add(priority);
        } else {
            // Remove old category index
            if let Some(ref old_cat) = self.buffer[self.position].category {
                if let Some(indices) = self.category_indices.get_mut(old_cat) {
                    indices.retain(|&i| i != self.position);
                }
            }

            // Add new category index
            if let Some(ref cat) = experience.category {
                self.category_indices
                    .entry(cat.clone())
                    .or_insert_with(Vec::new)
                    .push(self.position);
            }

            self.buffer[self.position] = experience;
            self.sum_tree.update_at(self.position, priority);
            self.position = (self.position + 1) % self.max_size;
        }
    }

    /// Update priority for an experience
    pub fn update_priority(&mut self, index: usize, priority: f64) {
        if index < self.buffer.len() {
            self.buffer[index].priority = priority;
            let transformed =
                (priority + self.config.min_priority).powf(self.config.priority_alpha);
            self.sum_tree.update_at(index, transformed);
        }
    }

    /// Sample a batch using configured strategy
    pub fn sample(&mut self, batch_size: Option<usize>) -> Result<SampledBatch> {
        let size = batch_size.unwrap_or(self.config.batch_size);

        if self.buffer.len() < size {
            return Err(Error::InvalidInput(format!(
                "Buffer size {} is less than batch size {}",
                self.buffer.len(),
                size
            )));
        }

        match self.config.strategy {
            SamplingStrategy::Uniform => self.sample_uniform(size),
            SamplingStrategy::Prioritized => self.sample_prioritized(size),
            SamplingStrategy::Stratified => self.sample_stratified(size),
            SamplingStrategy::RecencyWeighted => self.sample_recency_weighted(size),
            SamplingStrategy::Diverse => self.sample_diverse(size),
            SamplingStrategy::Curriculum => self.sample_curriculum(size),
            SamplingStrategy::Hindsight => self.sample_hindsight(size),
        }
    }

    /// Uniform random sampling
    fn sample_uniform(&mut self, batch_size: usize) -> Result<SampledBatch> {
        let mut batch = SampledBatch::new(SamplingStrategy::Uniform);
        let mut rng = rand::rng();
        let mut sampled_indices: Vec<usize> = Vec::with_capacity(batch_size);

        if self.config.with_replacement {
            for _ in 0..batch_size {
                let idx = rng.random_range(0..self.buffer.len());
                sampled_indices.push(idx);
            }
        } else {
            // Fisher-Yates style sampling
            let mut available: Vec<usize> = (0..self.buffer.len()).collect();
            for _ in 0..batch_size {
                let i = rng.random_range(0..available.len());
                sampled_indices.push(available.swap_remove(i));
            }
        }

        for idx in &sampled_indices {
            let mut exp = self.buffer[*idx].clone();
            exp.sample_count += 1;

            if let Some(ref cat) = exp.category {
                *batch.category_distribution.entry(cat.clone()).or_insert(0) += 1;
            }

            batch.experiences.push(exp);
            batch.weights.push(1.0); // Uniform weights
            batch.indices.push(*idx);
        }

        batch.avg_priority =
            batch.experiences.iter().map(|e| e.priority).sum::<f64>() / batch_size as f64;

        self.total_samples += batch_size;
        Ok(batch)
    }

    /// Prioritized sampling with importance weights
    fn sample_prioritized(&mut self, batch_size: usize) -> Result<SampledBatch> {
        let mut batch = SampledBatch::new(SamplingStrategy::Prioritized);
        let mut rng = rand::rng();
        let mut sampled_indices: Vec<usize> = Vec::with_capacity(batch_size);

        let total_priority = self.sum_tree.total();
        if total_priority <= 0.0 {
            return self.sample_uniform(batch_size);
        }

        let segment_size = total_priority / batch_size as f64;
        let min_priority = self.sum_tree.min();

        // Calculate max weight for normalization
        let max_weight =
            (self.buffer.len() as f64 * min_priority / total_priority).powf(-self.current_beta);

        for i in 0..batch_size {
            // Sample from segment
            let low = segment_size * i as f64;
            let high = segment_size * (i + 1) as f64;
            let value = rng.random_range(low..high);

            let idx = self.sum_tree.get(value);
            sampled_indices.push(idx);

            // Calculate importance weight
            let priority = self.sum_tree.get_priority(idx);
            let prob = priority / total_priority;
            let weight = (self.buffer.len() as f64 * prob).powf(-self.current_beta);
            let normalized_weight = weight / max_weight;

            batch.weights.push(normalized_weight);
        }

        for idx in &sampled_indices {
            let mut exp = self.buffer[*idx].clone();
            exp.sample_count += 1;

            if let Some(ref cat) = exp.category {
                *batch.category_distribution.entry(cat.clone()).or_insert(0) += 1;
            }

            batch.experiences.push(exp);
            batch.indices.push(*idx);
        }

        batch.avg_priority =
            batch.experiences.iter().map(|e| e.priority).sum::<f64>() / batch_size as f64;

        // Anneal beta
        self.current_beta =
            (self.current_beta + self.config.beta_annealing_rate).min(self.config.max_beta);

        self.total_samples += batch_size;
        Ok(batch)
    }

    /// Stratified sampling across categories
    fn sample_stratified(&mut self, batch_size: usize) -> Result<SampledBatch> {
        let mut batch = SampledBatch::new(SamplingStrategy::Stratified);
        let mut rng = rand::rng();

        // Calculate samples per stratum
        let categories: Vec<String> = self.category_indices.keys().cloned().collect();

        if categories.is_empty() {
            // Fall back to uniform if no categories
            return self.sample_uniform(batch_size);
        }

        let samples_per_category = batch_size / categories.len();
        let remainder = batch_size % categories.len();

        for (i, cat) in categories.iter().enumerate() {
            let indices = match self.category_indices.get(cat) {
                Some(idx) if !idx.is_empty() => idx,
                _ => continue,
            };

            let count = samples_per_category + if i < remainder { 1 } else { 0 };
            let count = count.min(indices.len());

            // Sample from this category
            let mut category_samples: Vec<usize> = Vec::new();
            for _ in 0..count {
                let idx = indices[rng.random_range(0..indices.len())];
                category_samples.push(idx);
            }

            for idx in category_samples {
                let mut exp = self.buffer[idx].clone();
                exp.sample_count += 1;

                *batch.category_distribution.entry(cat.clone()).or_insert(0) += 1;

                batch.experiences.push(exp);
                batch.weights.push(1.0);
                batch.indices.push(idx);
            }
        }

        batch.avg_priority = if !batch.experiences.is_empty() {
            batch.experiences.iter().map(|e| e.priority).sum::<f64>()
                / batch.experiences.len() as f64
        } else {
            0.0
        };

        self.total_samples += batch.experiences.len();
        Ok(batch)
    }

    /// Recency-weighted sampling
    fn sample_recency_weighted(&mut self, batch_size: usize) -> Result<SampledBatch> {
        let mut batch = SampledBatch::new(SamplingStrategy::RecencyWeighted);
        let mut rng = rand::rng();

        // Calculate recency weights
        let max_timestamp = self.buffer.iter().map(|e| e.timestamp).max().unwrap_or(0);

        let weights: Vec<f64> = self
            .buffer
            .iter()
            .map(|e| {
                let age = max_timestamp.saturating_sub(e.timestamp);
                self.config.recency_decay.powf(age as f64)
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();

        if total_weight <= 0.0 {
            return self.sample_uniform(batch_size);
        }

        // Sample according to weights
        for _ in 0..batch_size {
            let target = rng.random::<f64>() * total_weight;
            let mut cumsum = 0.0;
            let mut selected_idx = 0;

            for (idx, &w) in weights.iter().enumerate() {
                cumsum += w;
                if cumsum >= target {
                    selected_idx = idx;
                    break;
                }
            }

            let mut exp = self.buffer[selected_idx].clone();
            exp.sample_count += 1;

            if let Some(ref cat) = exp.category {
                *batch.category_distribution.entry(cat.clone()).or_insert(0) += 1;
            }

            let importance_weight = (self.buffer.len() as f64 * weights[selected_idx]
                / total_weight)
                .powf(-self.current_beta);

            batch.experiences.push(exp);
            batch.weights.push(importance_weight);
            batch.indices.push(selected_idx);
        }

        // Normalize weights
        let max_weight = batch.weights.iter().cloned().fold(0.0, f64::max);
        if max_weight > 0.0 {
            for w in &mut batch.weights {
                *w /= max_weight;
            }
        }

        batch.avg_priority =
            batch.experiences.iter().map(|e| e.priority).sum::<f64>() / batch_size as f64;

        self.total_samples += batch_size;
        Ok(batch)
    }

    /// Diverse sampling (maximize diversity in batch)
    fn sample_diverse(&mut self, batch_size: usize) -> Result<SampledBatch> {
        let mut batch = SampledBatch::new(SamplingStrategy::Diverse);
        let mut rng = rand::rng();
        let mut selected: Vec<usize> = Vec::with_capacity(batch_size);

        // Start with a random sample
        let first_idx = rng.random_range(0..self.buffer.len());
        selected.push(first_idx);

        // Greedily select most diverse samples
        while selected.len() < batch_size && selected.len() < self.buffer.len() {
            let mut best_idx = 0;
            let mut best_min_dist = -1.0f64;

            for (idx, exp) in self.buffer.iter().enumerate() {
                if selected.contains(&idx) {
                    continue;
                }

                // Calculate minimum distance to selected samples
                let min_dist = selected
                    .iter()
                    .map(|&sel_idx| self.state_distance(&exp.state, &self.buffer[sel_idx].state))
                    .fold(f64::INFINITY, f64::min);

                if min_dist > best_min_dist {
                    best_min_dist = min_dist;
                    best_idx = idx;
                }
            }

            if best_min_dist >= self.config.diversity_threshold || selected.len() < batch_size / 2 {
                selected.push(best_idx);
            } else {
                // Fall back to random if diversity threshold not met
                let remaining: Vec<usize> = (0..self.buffer.len())
                    .filter(|i| !selected.contains(i))
                    .collect();
                if !remaining.is_empty() {
                    selected.push(remaining[rng.random_range(0..remaining.len())]);
                }
            }
        }

        for idx in &selected {
            let mut exp = self.buffer[*idx].clone();
            exp.sample_count += 1;

            if let Some(ref cat) = exp.category {
                *batch.category_distribution.entry(cat.clone()).or_insert(0) += 1;
            }

            batch.experiences.push(exp);
            batch.weights.push(1.0);
            batch.indices.push(*idx);
        }

        batch.avg_priority = batch.experiences.iter().map(|e| e.priority).sum::<f64>()
            / batch.experiences.len() as f64;

        self.total_samples += selected.len();
        Ok(batch)
    }

    /// Curriculum sampling (easy to hard based on progress)
    fn sample_curriculum(&mut self, batch_size: usize) -> Result<SampledBatch> {
        let mut batch = SampledBatch::new(SamplingStrategy::Curriculum);
        let mut rng = rand::rng();

        // Calculate difficulty-based weights
        let progress = self.config.curriculum_progress;

        let weights: Vec<f64> = self
            .buffer
            .iter()
            .map(|e| {
                // At progress=0, prefer easy (low difficulty)
                // At progress=1, sample uniformly
                let target_difficulty = progress;
                let distance = (e.difficulty - target_difficulty).abs();
                (-distance * 2.0).exp()
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();

        if total_weight <= 0.0 {
            return self.sample_uniform(batch_size);
        }

        for _ in 0..batch_size {
            let target = rng.random::<f64>() * total_weight;
            let mut cumsum = 0.0;
            let mut selected_idx = 0;

            for (idx, &w) in weights.iter().enumerate() {
                cumsum += w;
                if cumsum >= target {
                    selected_idx = idx;
                    break;
                }
            }

            let mut exp = self.buffer[selected_idx].clone();
            exp.sample_count += 1;

            if let Some(ref cat) = exp.category {
                *batch.category_distribution.entry(cat.clone()).or_insert(0) += 1;
            }

            batch.experiences.push(exp);
            batch.weights.push(1.0);
            batch.indices.push(selected_idx);
        }

        batch.avg_priority =
            batch.experiences.iter().map(|e| e.priority).sum::<f64>() / batch_size as f64;

        self.total_samples += batch_size;
        Ok(batch)
    }

    /// Hindsight sampling (focus on goal-achieving experiences)
    fn sample_hindsight(&mut self, batch_size: usize) -> Result<SampledBatch> {
        let mut batch = SampledBatch::new(SamplingStrategy::Hindsight);
        let mut rng = rand::rng();

        // Weight by reward magnitude (success-focused)
        let weights: Vec<f64> = self
            .buffer
            .iter()
            .map(|e| {
                // Higher weight for successful experiences
                let base = if e.reward > 0.0 { 2.0 } else { 1.0 };
                base * (1.0 + e.reward.abs() as f64)
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();

        if total_weight <= 0.0 {
            return self.sample_uniform(batch_size);
        }

        for _ in 0..batch_size {
            let target = rng.random::<f64>() * total_weight;
            let mut cumsum = 0.0;
            let mut selected_idx = 0;

            for (idx, &w) in weights.iter().enumerate() {
                cumsum += w;
                if cumsum >= target {
                    selected_idx = idx;
                    break;
                }
            }

            let mut exp = self.buffer[selected_idx].clone();
            exp.sample_count += 1;

            if let Some(ref cat) = exp.category {
                *batch.category_distribution.entry(cat.clone()).or_insert(0) += 1;
            }

            batch.experiences.push(exp);
            batch.weights.push(1.0);
            batch.indices.push(selected_idx);
        }

        batch.avg_priority =
            batch.experiences.iter().map(|e| e.priority).sum::<f64>() / batch_size as f64;

        self.total_samples += batch_size;
        Ok(batch)
    }

    /// Calculate distance between two states (for diversity sampling)
    fn state_distance(&self, s1: &[f32], s2: &[f32]) -> f64 {
        if s1.len() != s2.len() {
            return f64::INFINITY;
        }

        let sum_sq: f64 = s1
            .iter()
            .zip(s2.iter())
            .map(|(a, b)| (*a - *b).powi(2) as f64)
            .sum();

        sum_sq.sqrt()
    }

    /// Set sampling strategy
    pub fn set_strategy(&mut self, strategy: SamplingStrategy) {
        self.config.strategy = strategy;
    }

    /// Get current strategy
    pub fn strategy(&self) -> SamplingStrategy {
        self.config.strategy
    }

    /// Set curriculum progress
    pub fn set_curriculum_progress(&mut self, progress: f64) {
        self.config.curriculum_progress = progress.clamp(0.0, 1.0);
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Get total samples drawn
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get current beta value
    pub fn current_beta(&self) -> f64 {
        self.current_beta
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.sum_tree = SumTree::new(self.max_size);
        self.position = 0;
        self.category_indices.clear();
    }

    /// Get experience by index
    pub fn get(&self, index: usize) -> Option<&SampleExperience> {
        self.buffer.get(index)
    }

    /// Get categories
    pub fn categories(&self) -> Vec<&String> {
        self.category_indices.keys().collect()
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via sample methods
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_experience(id: u64, reward: f32, priority: f64) -> SampleExperience {
        SampleExperience::new(
            id,
            vec![1.0, 2.0, 3.0],
            0,
            reward,
            vec![2.0, 3.0, 4.0],
            false,
        )
        .with_priority(priority)
    }

    #[test]
    fn test_basic() {
        let instance = BatchSampling::new();
        assert!(instance.process().is_ok());
        assert!(instance.is_empty());
    }

    #[test]
    fn test_add_experiences() {
        let mut sampler = BatchSampling::with_capacity(100);

        for i in 0..10 {
            sampler.add(create_test_experience(i, i as f32, 1.0));
        }

        assert_eq!(sampler.len(), 10);
    }

    #[test]
    fn test_uniform_sampling() {
        let mut sampler = BatchSampling::with_capacity(100);
        sampler.set_strategy(SamplingStrategy::Uniform);

        for i in 0..50 {
            sampler.add(create_test_experience(i, i as f32, 1.0));
        }

        let batch = sampler.sample(Some(10)).unwrap();
        assert_eq!(batch.len(), 10);
        assert_eq!(batch.strategy, SamplingStrategy::Uniform);
    }

    #[test]
    fn test_prioritized_sampling() {
        let mut sampler = BatchSampling::with_capacity(100);
        sampler.set_strategy(SamplingStrategy::Prioritized);

        // Add experiences with varying priorities
        for i in 0..50 {
            let priority = if i < 10 { 10.0 } else { 1.0 };
            sampler.add(create_test_experience(i, i as f32, priority));
        }

        let batch = sampler.sample(Some(10)).unwrap();
        assert_eq!(batch.len(), 10);

        // Importance weights should be present
        assert!(!batch.weights.is_empty());
    }

    #[test]
    fn test_stratified_sampling() {
        let mut sampler = BatchSampling::with_capacity(100);
        sampler.set_strategy(SamplingStrategy::Stratified);

        // Add experiences in different categories
        for i in 0..30 {
            let category = if i % 3 == 0 {
                "A"
            } else if i % 3 == 1 {
                "B"
            } else {
                "C"
            };
            let exp = create_test_experience(i, i as f32, 1.0).with_category(category);
            sampler.add(exp);
        }

        let batch = sampler.sample(Some(9)).unwrap();

        // Should have samples from multiple categories
        assert!(batch.category_distribution.len() > 1);
    }

    #[test]
    fn test_recency_weighted_sampling() {
        let mut sampler = BatchSampling::with_capacity(100);
        sampler.set_strategy(SamplingStrategy::RecencyWeighted);

        for i in 0..50 {
            let exp = create_test_experience(i, i as f32, 1.0).with_timestamp(i * 100);
            sampler.add(exp);
        }

        let batch = sampler.sample(Some(10)).unwrap();
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_diverse_sampling() {
        let mut sampler = BatchSampling::with_capacity(100);
        sampler.set_strategy(SamplingStrategy::Diverse);

        // Add experiences with different states
        for i in 0..50 {
            let exp = SampleExperience::new(
                i as u64,
                vec![i as f32, (i * 2) as f32, (i * 3) as f32],
                0,
                i as f32,
                vec![1.0, 2.0, 3.0],
                false,
            );
            sampler.add(exp);
        }

        let batch = sampler.sample(Some(10)).unwrap();
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_curriculum_sampling() {
        let mut sampler = BatchSampling::with_capacity(100);
        sampler.set_strategy(SamplingStrategy::Curriculum);

        // Add experiences with different difficulties
        for i in 0..50 {
            let difficulty = i as f64 / 50.0;
            let exp = create_test_experience(i, i as f32, 1.0).with_difficulty(difficulty);
            sampler.add(exp);
        }

        // Early in curriculum (prefer easy)
        sampler.set_curriculum_progress(0.0);
        let batch = sampler.sample(Some(10)).unwrap();
        assert_eq!(batch.len(), 10);

        // Late in curriculum (more uniform)
        sampler.set_curriculum_progress(1.0);
        let batch = sampler.sample(Some(10)).unwrap();
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_hindsight_sampling() {
        let mut sampler = BatchSampling::with_capacity(100);
        sampler.set_strategy(SamplingStrategy::Hindsight);

        // Add experiences with different rewards
        for i in 0..50 {
            let reward = if i % 5 == 0 { 10.0 } else { -1.0 };
            sampler.add(create_test_experience(i, reward, 1.0));
        }

        let batch = sampler.sample(Some(10)).unwrap();
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_batch_accessors() {
        let mut sampler = BatchSampling::with_capacity(100);
        sampler.set_strategy(SamplingStrategy::Uniform);

        for i in 0..20 {
            sampler.add(create_test_experience(i, i as f32, 1.0));
        }

        let batch = sampler.sample(Some(5)).unwrap();

        assert_eq!(batch.states().len(), 5);
        assert_eq!(batch.actions().len(), 5);
        assert_eq!(batch.rewards().len(), 5);
        assert_eq!(batch.next_states().len(), 5);
        assert_eq!(batch.dones().len(), 5);
    }

    #[test]
    fn test_priority_update() {
        let mut sampler = BatchSampling::with_capacity(100);

        for i in 0..10 {
            sampler.add(create_test_experience(i, i as f32, 1.0));
        }

        sampler.update_priority(5, 10.0);

        let exp = sampler.get(5).unwrap();
        assert_eq!(exp.priority, 10.0);
    }

    #[test]
    fn test_sum_tree() {
        let mut tree = SumTree::new(8);

        // Add some priorities
        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);
        tree.add(4.0);

        assert_eq!(tree.len(), 4);
        assert!((tree.total() - 10.0).abs() < 0.001);
        assert!((tree.min() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_beta_annealing() {
        let mut config = BatchSamplingConfig::default();
        config.importance_beta = 0.4;
        config.beta_annealing_rate = 0.1;
        config.max_beta = 1.0;

        let mut sampler = BatchSampling::with_config(config, 100);
        sampler.set_strategy(SamplingStrategy::Prioritized);

        for i in 0..50 {
            sampler.add(create_test_experience(i, i as f32, 1.0 + i as f64));
        }

        let initial_beta = sampler.current_beta();

        // Sample multiple times to anneal beta
        for _ in 0..5 {
            let _ = sampler.sample(Some(10));
        }

        assert!(sampler.current_beta() > initial_beta);
    }

    #[test]
    fn test_circular_buffer() {
        let mut sampler = BatchSampling::with_capacity(10);

        // Add more than capacity
        for i in 0..20 {
            sampler.add(create_test_experience(i, i as f32, 1.0));
        }

        assert_eq!(sampler.len(), 10);
    }

    #[test]
    fn test_insufficient_samples() {
        let mut sampler = BatchSampling::with_capacity(100);

        for i in 0..5 {
            sampler.add(create_test_experience(i, i as f32, 1.0));
        }

        let result = sampler.sample(Some(10));
        assert!(result.is_err());
    }

    #[test]
    fn test_experience_builders() {
        let exp = SampleExperience::new(1, vec![1.0], 0, 1.0, vec![2.0], false)
            .with_priority(5.0)
            .with_category("test")
            .with_timestamp(12345)
            .with_difficulty(0.7)
            .with_metadata("key", 42.0);

        assert_eq!(exp.priority, 5.0);
        assert_eq!(exp.category, Some("test".to_string()));
        assert_eq!(exp.timestamp, 12345);
        assert!((exp.difficulty - 0.7).abs() < 0.001);
        assert_eq!(exp.metadata.get("key"), Some(&42.0));
    }

    #[test]
    fn test_clear() {
        let mut sampler = BatchSampling::with_capacity(100);

        for i in 0..20 {
            sampler.add(create_test_experience(i, i as f32, 1.0));
        }

        assert_eq!(sampler.len(), 20);

        sampler.clear();

        assert!(sampler.is_empty());
    }

    #[test]
    fn test_categories() {
        let mut sampler = BatchSampling::with_capacity(100);

        for i in 0..30 {
            let cat = format!("cat_{}", i % 3);
            let exp = create_test_experience(i, i as f32, 1.0).with_category(&cat);
            sampler.add(exp);
        }

        let categories = sampler.categories();
        assert_eq!(categories.len(), 3);
    }
}
