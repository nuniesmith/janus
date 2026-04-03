//! Emotional Memory - Fear-associated memories with emotional valence
//!
//! Stores and retrieves memories based on their emotional significance.
//! Memories with strong emotional valence (fear, loss) are stored with
//! higher priority and persist longer.
//!
//! Key features:
//! - Emotional tagging of market events
//! - Valence-weighted memory retrieval
//! - Decay based on emotional intensity
//! - Consolidation of similar emotional experiences

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

/// Emotional valence (positive to negative)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Valence {
    /// Strongly positive (large gain)
    StrongPositive,
    /// Mildly positive (small gain)
    MildPositive,
    /// Neutral
    Neutral,
    /// Mildly negative (small loss)
    MildNegative,
    /// Strongly negative (large loss)
    StrongNegative,
}

impl Valence {
    /// Convert to numeric value (-1.0 to 1.0)
    pub fn to_value(&self) -> f64 {
        match self {
            Self::StrongPositive => 1.0,
            Self::MildPositive => 0.5,
            Self::Neutral => 0.0,
            Self::MildNegative => -0.5,
            Self::StrongNegative => -1.0,
        }
    }

    /// Create from numeric value
    pub fn from_value(value: f64) -> Self {
        if value > 0.7 {
            Self::StrongPositive
        } else if value > 0.2 {
            Self::MildPositive
        } else if value > -0.2 {
            Self::Neutral
        } else if value > -0.7 {
            Self::MildNegative
        } else {
            Self::StrongNegative
        }
    }

    /// Get memory persistence multiplier (negative emotions persist longer)
    pub fn persistence_multiplier(&self) -> f64 {
        match self {
            Self::StrongNegative => 2.0,
            Self::MildNegative => 1.5,
            Self::Neutral => 1.0,
            Self::MildPositive => 0.8,
            Self::StrongPositive => 0.7,
        }
    }

    /// Is this a negative valence?
    pub fn is_negative(&self) -> bool {
        matches!(self, Self::MildNegative | Self::StrongNegative)
    }

    /// Is this a fear-related valence?
    pub fn is_fear(&self) -> bool {
        self.is_negative()
    }
}

/// Configuration for emotional memory system
#[derive(Debug, Clone)]
pub struct EmotionalMemoryConfig {
    /// Maximum number of memories to store
    pub capacity: usize,
    /// Base decay rate per time unit
    pub base_decay_rate: f64,
    /// Threshold for emotional significance (0.0 - 1.0)
    pub significance_threshold: f64,
    /// Weight for emotional intensity in retrieval
    pub emotion_weight: f64,
    /// Weight for recency in retrieval
    pub recency_weight: f64,
    /// Weight for similarity in retrieval
    pub similarity_weight: f64,
    /// Consolidation threshold (how similar memories must be to merge)
    pub consolidation_threshold: f64,
    /// Maximum age for memories (milliseconds, 0 = unlimited)
    pub max_age_ms: i64,
}

impl Default for EmotionalMemoryConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            base_decay_rate: 0.99,
            significance_threshold: 0.2,
            emotion_weight: 0.4,
            recency_weight: 0.3,
            similarity_weight: 0.3,
            consolidation_threshold: 0.85,
            max_age_ms: 7 * 24 * 60 * 60 * 1000, // 7 days
        }
    }
}

/// An emotional memory entry
#[derive(Debug, Clone)]
pub struct EmotionalMemoryEntry {
    /// Unique identifier
    pub id: u64,
    /// Feature vector representing the market state
    pub features: Vec<f64>,
    /// Emotional valence
    pub valence: Valence,
    /// Emotional intensity (0.0 - 1.0)
    pub intensity: f64,
    /// Outcome value (e.g., P&L)
    pub outcome: f64,
    /// Context/category label
    pub context: String,
    /// Creation timestamp
    pub created_at: i64,
    /// Last access timestamp
    pub last_accessed: i64,
    /// Access count
    pub access_count: u32,
    /// Consolidation count (how many memories merged into this)
    pub consolidation_count: u32,
    /// Current strength (decays over time)
    pub strength: f64,
}

impl EmotionalMemoryEntry {
    /// Create a new emotional memory
    pub fn new(
        id: u64,
        features: Vec<f64>,
        valence: Valence,
        intensity: f64,
        outcome: f64,
        context: String,
    ) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            id,
            features,
            valence,
            intensity: intensity.clamp(0.0, 1.0),
            outcome,
            context,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            consolidation_count: 1,
            strength: 1.0,
        }
    }

    /// Calculate similarity to a feature vector
    pub fn similarity(&self, other: &[f64]) -> f64 {
        if self.features.len() != other.len() || self.features.is_empty() {
            return 0.0;
        }

        // Cosine similarity
        let dot: f64 = self
            .features
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f64 = self.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = other.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate age in milliseconds
    pub fn age_ms(&self) -> i64 {
        chrono::Utc::now().timestamp_millis() - self.created_at
    }

    /// Calculate emotional significance score
    pub fn significance(&self) -> f64 {
        self.intensity * self.strength * self.valence.persistence_multiplier()
    }

    /// Mark as accessed
    pub fn access(&mut self) {
        self.last_accessed = chrono::Utc::now().timestamp_millis();
        self.access_count += 1;
        // Accessing strengthens the memory slightly
        self.strength = (self.strength * 1.05).min(1.0);
    }

    /// Merge with another similar memory
    pub fn merge(&mut self, other: &EmotionalMemoryEntry) {
        // Average the features
        for (a, b) in self.features.iter_mut().zip(other.features.iter()) {
            *a = (*a + *b) / 2.0;
        }

        // Take stronger emotional response
        if other.intensity > self.intensity {
            self.intensity = other.intensity;
            self.valence = other.valence;
        }

        // Combine outcomes (weighted by consolidation count)
        let total_count = self.consolidation_count + other.consolidation_count;
        self.outcome = (self.outcome * self.consolidation_count as f64
            + other.outcome * other.consolidation_count as f64)
            / total_count as f64;

        self.consolidation_count = total_count;

        // Strengthen the memory
        self.strength = (self.strength + other.strength * 0.5).min(1.0);
        self.last_accessed = chrono::Utc::now().timestamp_millis();
    }
}

/// Result of a memory retrieval query
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Retrieved memories with their relevance scores
    pub memories: Vec<(EmotionalMemoryEntry, f64)>,
    /// Aggregate emotional valence
    pub aggregate_valence: f64,
    /// Aggregate emotional intensity
    pub aggregate_intensity: f64,
    /// Number of fear-related memories
    pub fear_count: usize,
    /// Average outcome of similar situations
    pub expected_outcome: f64,
}

/// Statistics about the emotional memory system
#[derive(Debug, Clone, Default)]
pub struct MemoryStatistics {
    pub total_memories: usize,
    pub fear_memories: usize,
    pub positive_memories: usize,
    pub neutral_memories: usize,
    pub average_intensity: f64,
    pub average_strength: f64,
    pub total_accesses: u64,
    pub oldest_memory_age_ms: i64,
    pub contexts: HashMap<String, usize>,
}

/// Fear-associated memories with emotional valence
pub struct EmotionalMemory {
    config: EmotionalMemoryConfig,
    /// Stored memories
    memories: Vec<EmotionalMemoryEntry>,
    /// Next memory ID
    next_id: u64,
    /// Recent retrievals for caching
    retrieval_cache: VecDeque<(Vec<f64>, RetrievalResult)>,
    /// Cache size
    cache_size: usize,
    /// Total encoding count
    encoding_count: u64,
    /// Total retrieval count
    retrieval_count: u64,
}

impl Default for EmotionalMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionalMemory {
    /// Create a new emotional memory system
    pub fn new() -> Self {
        Self::with_config(EmotionalMemoryConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EmotionalMemoryConfig) -> Self {
        Self {
            memories: Vec::with_capacity(config.capacity),
            next_id: 1,
            retrieval_cache: VecDeque::with_capacity(100),
            cache_size: 100,
            encoding_count: 0,
            retrieval_count: 0,
            config,
        }
    }

    /// Encode a new emotional memory
    pub fn encode(
        &mut self,
        features: Vec<f64>,
        valence: Valence,
        intensity: f64,
        outcome: f64,
        context: String,
    ) -> u64 {
        // Check if this is emotionally significant enough to store
        if intensity < self.config.significance_threshold && valence == Valence::Neutral {
            return 0; // Not significant enough
        }

        // Try to consolidate with existing similar memory
        if let Some(idx) = self.find_similar_memory(&features) {
            let new_memory =
                EmotionalMemoryEntry::new(0, features, valence, intensity, outcome, context);
            self.memories[idx].merge(&new_memory);
            self.encoding_count += 1;
            self.invalidate_cache();
            return self.memories[idx].id;
        }

        // Create new memory
        let id = self.next_id;
        self.next_id += 1;

        let memory = EmotionalMemoryEntry::new(id, features, valence, intensity, outcome, context);

        // Manage capacity
        if self.memories.len() >= self.config.capacity {
            self.evict_weakest();
        }

        self.memories.push(memory);
        self.encoding_count += 1;
        self.invalidate_cache();

        id
    }

    /// Retrieve memories similar to given features
    pub fn retrieve(&mut self, features: &[f64], limit: usize) -> RetrievalResult {
        // Apply decay before retrieval
        self.apply_decay();

        // Check cache
        if let Some(cached) = self.check_cache(features) {
            return cached;
        }

        self.retrieval_count += 1;

        // Score all memories
        let mut scored: Vec<(usize, f64)> = self
            .memories
            .iter()
            .enumerate()
            .map(|(idx, mem)| {
                let similarity = mem.similarity(features);
                let recency = self.calculate_recency_score(mem);
                let emotion = mem.significance();

                let score = similarity * self.config.similarity_weight
                    + recency * self.config.recency_weight
                    + emotion * self.config.emotion_weight;

                (idx, score)
            })
            .filter(|(_, score)| *score > 0.1)
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(limit);

        // Mark as accessed and collect results
        let mut memories = Vec::with_capacity(scored.len());
        let mut total_valence = 0.0;
        let mut total_intensity = 0.0;
        let mut total_outcome = 0.0;
        let mut fear_count = 0;
        let mut weight_sum = 0.0;

        for (idx, score) in scored {
            self.memories[idx].access();
            let mem = self.memories[idx].clone();

            if mem.valence.is_fear() {
                fear_count += 1;
            }

            total_valence += mem.valence.to_value() * score;
            total_intensity += mem.intensity * score;
            total_outcome += mem.outcome * score;
            weight_sum += score;

            memories.push((mem, score));
        }

        // Normalize aggregates
        let aggregate_valence = if weight_sum > 0.0 {
            total_valence / weight_sum
        } else {
            0.0
        };

        let aggregate_intensity = if weight_sum > 0.0 {
            total_intensity / weight_sum
        } else {
            0.0
        };

        let expected_outcome = if weight_sum > 0.0 {
            total_outcome / weight_sum
        } else {
            0.0
        };

        let result = RetrievalResult {
            memories,
            aggregate_valence,
            aggregate_intensity,
            fear_count,
            expected_outcome,
        };

        // Cache the result
        self.cache_result(features.to_vec(), result.clone());

        result
    }

    /// Retrieve only fear-related memories
    pub fn retrieve_fears(&mut self, features: &[f64], limit: usize) -> RetrievalResult {
        self.apply_decay();

        let mut scored: Vec<(usize, f64)> = self
            .memories
            .iter()
            .enumerate()
            .filter(|(_, mem)| mem.valence.is_fear())
            .map(|(idx, mem)| {
                let similarity = mem.similarity(features);
                let score = similarity * mem.significance();
                (idx, score)
            })
            .filter(|(_, score)| *score > 0.05)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(limit);

        let mut memories = Vec::with_capacity(scored.len());
        let mut total_intensity = 0.0;
        let mut total_outcome = 0.0;
        let mut weight_sum = 0.0;

        for (idx, score) in &scored {
            self.memories[*idx].access();
            let mem = self.memories[*idx].clone();

            total_intensity += mem.intensity * score;
            total_outcome += mem.outcome * score;
            weight_sum += score;

            memories.push((mem, *score));
        }

        let aggregate_intensity = if weight_sum > 0.0 {
            total_intensity / weight_sum
        } else {
            0.0
        };

        let expected_outcome = if weight_sum > 0.0 {
            total_outcome / weight_sum
        } else {
            0.0
        };

        RetrievalResult {
            memories,
            aggregate_valence: -1.0, // All fears are negative
            aggregate_intensity,
            fear_count: scored.len(),
            expected_outcome,
        }
    }

    /// Find memory most similar to features
    fn find_similar_memory(&self, features: &[f64]) -> Option<usize> {
        self.memories
            .iter()
            .enumerate()
            .map(|(idx, mem)| (idx, mem.similarity(features)))
            .filter(|(_, sim)| *sim >= self.config.consolidation_threshold)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(idx, _)| idx)
    }

    /// Calculate recency score (0.0 to 1.0)
    fn calculate_recency_score(&self, memory: &EmotionalMemoryEntry) -> f64 {
        let age_ms = chrono::Utc::now().timestamp_millis() - memory.last_accessed;
        let max_age = self.config.max_age_ms as f64;

        if max_age > 0.0 {
            (1.0 - (age_ms as f64 / max_age)).max(0.0)
        } else {
            // Exponential decay with half-life of 24 hours
            let half_life_ms = 24.0 * 60.0 * 60.0 * 1000.0;
            0.5_f64.powf(age_ms as f64 / half_life_ms)
        }
    }

    /// Apply decay to all memories
    fn apply_decay(&mut self) {
        let now = chrono::Utc::now().timestamp_millis();

        for memory in &mut self.memories {
            let age_hours = (now - memory.last_accessed) as f64 / (60.0 * 60.0 * 1000.0);
            let decay = self.config.base_decay_rate.powf(age_hours);
            memory.strength *= decay * memory.valence.persistence_multiplier();
            memory.strength = memory.strength.clamp(0.01, 1.0);
        }

        // Remove expired or very weak memories
        if self.config.max_age_ms > 0 {
            self.memories
                .retain(|m| m.age_ms() < self.config.max_age_ms && m.strength > 0.05);
        } else {
            self.memories.retain(|m| m.strength > 0.05);
        }
    }

    /// Evict the weakest memory
    fn evict_weakest(&mut self) {
        if let Some((idx, _)) = self.memories.iter().enumerate().min_by(|(_, a), (_, b)| {
            let score_a = a.strength * a.significance();
            let score_b = b.strength * b.significance();
            score_a.partial_cmp(&score_b).unwrap()
        }) {
            self.memories.remove(idx);
        }
    }

    /// Check retrieval cache
    fn check_cache(&self, features: &[f64]) -> Option<RetrievalResult> {
        for (cached_features, result) in &self.retrieval_cache {
            if cached_features.len() == features.len() {
                let diff: f64 = cached_features
                    .iter()
                    .zip(features.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                if diff < 0.01 {
                    return Some(result.clone());
                }
            }
        }
        None
    }

    /// Cache a retrieval result
    fn cache_result(&mut self, features: Vec<f64>, result: RetrievalResult) {
        if self.retrieval_cache.len() >= self.cache_size {
            self.retrieval_cache.pop_front();
        }
        self.retrieval_cache.push_back((features, result));
    }

    /// Invalidate the cache
    fn invalidate_cache(&mut self) {
        self.retrieval_cache.clear();
    }

    /// Get total memory count
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Get fear memory count
    pub fn fear_count(&self) -> usize {
        self.memories.iter().filter(|m| m.valence.is_fear()).count()
    }

    /// Get statistics about the memory system
    pub fn statistics(&self) -> MemoryStatistics {
        let mut contexts: HashMap<String, usize> = HashMap::new();
        let mut total_intensity = 0.0;
        let mut total_strength = 0.0;
        let mut total_accesses = 0u64;
        let mut oldest_age = 0i64;
        let mut fear_count = 0;
        let mut positive_count = 0;
        let mut neutral_count = 0;

        for mem in &self.memories {
            *contexts.entry(mem.context.clone()).or_insert(0) += 1;
            total_intensity += mem.intensity;
            total_strength += mem.strength;
            total_accesses += mem.access_count as u64;

            let age = mem.age_ms();
            if age > oldest_age {
                oldest_age = age;
            }

            match mem.valence {
                Valence::StrongNegative | Valence::MildNegative => fear_count += 1,
                Valence::StrongPositive | Valence::MildPositive => positive_count += 1,
                Valence::Neutral => neutral_count += 1,
            }
        }

        let n = self.memories.len() as f64;

        MemoryStatistics {
            total_memories: self.memories.len(),
            fear_memories: fear_count,
            positive_memories: positive_count,
            neutral_memories: neutral_count,
            average_intensity: if n > 0.0 { total_intensity / n } else { 0.0 },
            average_strength: if n > 0.0 { total_strength / n } else { 0.0 },
            total_accesses,
            oldest_memory_age_ms: oldest_age,
            contexts,
        }
    }

    /// Reset the memory system
    pub fn reset(&mut self) {
        self.memories.clear();
        self.retrieval_cache.clear();
        self.encoding_count = 0;
        self.retrieval_count = 0;
    }

    /// Main processing function (compatibility with neuromorphic interface)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_creation() {
        let memory = EmotionalMemory::new();
        assert_eq!(memory.memory_count(), 0);
    }

    #[test]
    fn test_encoding() {
        let mut memory = EmotionalMemory::new();

        let id = memory.encode(
            vec![1.0, 0.5, 0.3],
            Valence::StrongNegative,
            0.8,
            -0.05,
            "loss".to_string(),
        );

        assert!(id > 0);
        assert_eq!(memory.memory_count(), 1);
        assert_eq!(memory.fear_count(), 1);
    }

    #[test]
    fn test_retrieval() {
        let mut memory = EmotionalMemory::new();

        // Encode a fear memory
        memory.encode(
            vec![1.0, 0.5, 0.3],
            Valence::StrongNegative,
            0.8,
            -0.05,
            "loss".to_string(),
        );

        // Retrieve similar
        let result = memory.retrieve(&[0.9, 0.5, 0.3], 10);

        assert!(!result.memories.is_empty());
        assert!(result.aggregate_valence < 0.0);
    }

    #[test]
    fn test_fear_retrieval() {
        let mut memory = EmotionalMemory::new();

        // Encode fear and positive memories
        memory.encode(
            vec![1.0, 0.0, 0.0],
            Valence::StrongNegative,
            0.8,
            -0.05,
            "loss".to_string(),
        );

        memory.encode(
            vec![0.0, 1.0, 0.0],
            Valence::StrongPositive,
            0.8,
            0.05,
            "gain".to_string(),
        );

        // Retrieve only fears
        let result = memory.retrieve_fears(&[1.0, 0.0, 0.0], 10);

        assert_eq!(result.fear_count, 1);
        assert!(result.aggregate_valence < 0.0);
    }

    #[test]
    fn test_consolidation() {
        let mut memory = EmotionalMemory::with_config(EmotionalMemoryConfig {
            consolidation_threshold: 0.9,
            ..Default::default()
        });

        // Encode similar memories
        memory.encode(
            vec![1.0, 0.5, 0.3],
            Valence::MildNegative,
            0.5,
            -0.02,
            "loss".to_string(),
        );

        memory.encode(
            vec![1.0, 0.5, 0.3],
            Valence::MildNegative,
            0.6,
            -0.03,
            "loss".to_string(),
        );

        // Should be consolidated into one
        assert_eq!(memory.memory_count(), 1);

        let stats = memory.statistics();
        assert!(stats.average_intensity > 0.5);
    }

    #[test]
    fn test_valence() {
        assert!(Valence::StrongNegative.is_fear());
        assert!(Valence::MildNegative.is_fear());
        assert!(!Valence::Neutral.is_fear());
        assert!(!Valence::MildPositive.is_fear());

        assert!((Valence::StrongNegative.to_value() - (-1.0)).abs() < 0.001);
        assert!((Valence::Neutral.to_value() - 0.0).abs() < 0.001);
        assert!((Valence::StrongPositive.to_value() - 1.0).abs() < 0.001);

        assert_eq!(Valence::from_value(-0.9), Valence::StrongNegative);
        assert_eq!(Valence::from_value(0.0), Valence::Neutral);
        assert_eq!(Valence::from_value(0.9), Valence::StrongPositive);
    }

    #[test]
    fn test_significance_threshold() {
        let mut memory = EmotionalMemory::with_config(EmotionalMemoryConfig {
            significance_threshold: 0.5,
            ..Default::default()
        });

        // Low intensity neutral should not be stored
        let id = memory.encode(
            vec![1.0, 0.5, 0.3],
            Valence::Neutral,
            0.1,
            0.0,
            "boring".to_string(),
        );

        assert_eq!(id, 0);
        assert_eq!(memory.memory_count(), 0);
    }

    #[test]
    fn test_memory_similarity() {
        let memory = EmotionalMemoryEntry::new(
            1,
            vec![1.0, 0.0, 0.0],
            Valence::Neutral,
            0.5,
            0.0,
            "test".to_string(),
        );

        // Identical
        let sim = memory.similarity(&[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal
        let sim = memory.similarity(&[0.0, 1.0, 0.0]);
        assert!(sim.abs() < 0.001);

        // Opposite
        let sim = memory.similarity(&[-1.0, 0.0, 0.0]);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_memory_merge() {
        let mut mem1 = EmotionalMemoryEntry::new(
            1,
            vec![1.0, 0.0],
            Valence::MildNegative,
            0.5,
            -0.02,
            "test".to_string(),
        );

        let mem2 = EmotionalMemoryEntry::new(
            2,
            vec![0.8, 0.2],
            Valence::StrongNegative,
            0.8,
            -0.05,
            "test".to_string(),
        );

        mem1.merge(&mem2);

        // Should take stronger emotion
        assert_eq!(mem1.valence, Valence::StrongNegative);
        assert!(mem1.intensity > 0.5);
        assert_eq!(mem1.consolidation_count, 2);
    }

    #[test]
    fn test_statistics() {
        let mut memory = EmotionalMemory::new();

        memory.encode(
            vec![1.0, 0.0, 0.0],
            Valence::StrongNegative,
            0.8,
            -0.05,
            "crash".to_string(),
        );

        memory.encode(
            vec![0.0, 1.0, 0.0],
            Valence::StrongPositive,
            0.7,
            0.05,
            "rally".to_string(),
        );

        let stats = memory.statistics();

        assert_eq!(stats.total_memories, 2);
        assert_eq!(stats.fear_memories, 1);
        assert_eq!(stats.positive_memories, 1);
        assert_eq!(stats.contexts.len(), 2);
    }

    #[test]
    fn test_reset() {
        let mut memory = EmotionalMemory::new();

        memory.encode(
            vec![1.0, 0.5, 0.3],
            Valence::StrongNegative,
            0.8,
            -0.05,
            "loss".to_string(),
        );

        assert!(memory.memory_count() > 0);

        memory.reset();

        assert_eq!(memory.memory_count(), 0);
    }

    #[test]
    fn test_persistence_multiplier() {
        // Negative emotions persist longer
        assert!(
            Valence::StrongNegative.persistence_multiplier()
                > Valence::Neutral.persistence_multiplier()
        );
        assert!(
            Valence::Neutral.persistence_multiplier()
                > Valence::StrongPositive.persistence_multiplier()
        );
    }

    #[test]
    fn test_process() {
        let memory = EmotionalMemory::new();
        assert!(memory.process().is_ok());
    }
}
