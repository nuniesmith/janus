//! Emotional Memory Tagging
//!
//! Bridges hippocampus (episodic memory) with amygdala (emotional responses).
//! Tags memories with emotional significance for fear conditioning and learning.

use crate::amygdala::{FearNetwork, MarketConditions, ThreatSignature};
use crate::hippocampus::memory::{
    MemoryEntry, MemoryType, MockVectorDB, VectorDB, VectorStorage, VectorStorageBackend,
};
use common::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

/// Emotional valence of a memory
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EmotionalValence {
    /// Positive outcome (profit, avoided loss)
    Positive,
    /// Negative outcome (loss, missed opportunity)
    Negative,
    /// Neutral (no significant outcome)
    Neutral,
}

/// Emotional arousal level (intensity)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EmotionalArousal {
    /// Low arousal (calm, routine)
    Low,
    /// Medium arousal (alert, engaged)
    Medium,
    /// High arousal (fear, excitement, panic)
    High,
}

/// Emotional tag attached to a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalTag {
    /// Emotional valence (positive/negative/neutral)
    pub valence: EmotionalValence,

    /// Arousal level (intensity)
    pub arousal: EmotionalArousal,

    /// Fear score (0.0 to 1.0)
    pub fear_score: f32,

    /// Threat signature if detected
    pub threat_signature: Option<String>,

    /// Outcome value (PnL, reward, etc.)
    pub outcome_value: f32,

    /// Timestamp when tagged
    pub timestamp: i64,
}

impl EmotionalTag {
    /// Create a new emotional tag
    pub fn new(
        valence: EmotionalValence,
        arousal: EmotionalArousal,
        fear_score: f32,
        outcome_value: f32,
    ) -> Self {
        Self {
            valence,
            arousal,
            fear_score: fear_score.clamp(0.0, 1.0),
            threat_signature: None,
            outcome_value,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Create tag from market conditions and fear network
    pub fn from_market_conditions(
        conditions: &MarketConditions,
        fear_network: &FearNetwork,
        outcome_value: f32,
    ) -> Self {
        // Detect threat
        let threat = fear_network.detect_threat(conditions);

        let fear_score = threat.as_ref().map(|t| t.danger_score).unwrap_or(0.0);

        let arousal = if fear_score > 0.7 {
            EmotionalArousal::High
        } else if fear_score > 0.3 {
            EmotionalArousal::Medium
        } else {
            EmotionalArousal::Low
        };

        let valence = if outcome_value > 0.0 {
            EmotionalValence::Positive
        } else if outcome_value < 0.0 {
            EmotionalValence::Negative
        } else {
            EmotionalValence::Neutral
        };

        let mut tag = Self::new(valence, arousal, fear_score, outcome_value);
        tag.threat_signature = threat.map(|t| t.name.clone());

        tag
    }

    /// Calculate memory consolidation priority (0.0 to 1.0)
    /// High arousal + strong valence = higher priority
    pub fn consolidation_priority(&self) -> f32 {
        let arousal_score = match self.arousal {
            EmotionalArousal::Low => 0.3,
            EmotionalArousal::Medium => 0.6,
            EmotionalArousal::High => 1.0,
        };

        let valence_score = match self.valence {
            EmotionalValence::Neutral => 0.2,
            EmotionalValence::Positive | EmotionalValence::Negative => 0.8,
        };

        let fear_contribution = self.fear_score * 0.5;

        ((arousal_score + valence_score) / 2.0 + fear_contribution).clamp(0.0, 1.0)
    }

    /// Check if this memory should be strongly consolidated (stored long-term)
    pub fn should_consolidate(&self) -> bool {
        self.consolidation_priority() > 0.6
    }
}

/// Emotional Memory System - bridges hippocampus and amygdala
pub struct EmotionalMemory {
    /// Vector database for persistent storage (supports both real and mock backends)
    storage: VectorStorageBackend,

    /// Fear network for threat detection
    fear_network: FearNetwork,

    /// Recent emotional tags cache
    recent_tags: HashMap<String, EmotionalTag>,

    /// Max cache size
    max_cache_size: usize,
}

impl EmotionalMemory {
    /// Create a new emotional memory system with real Qdrant backend
    pub async fn new(qdrant_url: &str, fear_threshold: f32) -> Result<Self> {
        info!("Initializing Emotional Memory System");

        let vector_db = VectorDB::new(qdrant_url).await?;
        vector_db.initialize().await?;

        let fear_network = FearNetwork::new(fear_threshold);

        Ok(Self {
            storage: VectorStorageBackend::Real(vector_db),
            fear_network,
            recent_tags: HashMap::new(),
            max_cache_size: 1000,
        })
    }

    /// Create a new emotional memory system with mock backend (for testing)
    pub fn new_mock(fear_threshold: f32) -> Self {
        info!("Initializing Emotional Memory System (mock mode)");

        let mock_db = MockVectorDB::new();
        let fear_network = FearNetwork::new(fear_threshold);

        Self {
            storage: VectorStorageBackend::Mock(mock_db),
            fear_network,
            recent_tags: HashMap::new(),
            max_cache_size: 1000,
        }
    }

    /// Create with a custom storage backend
    pub fn with_backend(backend: VectorStorageBackend, fear_threshold: f32) -> Self {
        let fear_network = FearNetwork::new(fear_threshold);

        Self {
            storage: backend,
            fear_network,
            recent_tags: HashMap::new(),
            max_cache_size: 1000,
        }
    }

    /// Tag a memory with emotional significance
    pub async fn tag_memory(
        &mut self,
        entry: MemoryEntry,
        conditions: &MarketConditions,
        outcome_value: f32,
    ) -> Result<String> {
        debug!("Tagging memory {} with emotional significance", entry.id);

        // Create emotional tag
        let tag =
            EmotionalTag::from_market_conditions(conditions, &self.fear_network, outcome_value);

        // Store in cache
        self.recent_tags.insert(entry.id.clone(), tag.clone());

        // Prune cache if needed
        if self.recent_tags.len() > self.max_cache_size {
            self.prune_cache();
        }

        // Create emotional memory entry with its own UUID.
        // Both Embedding and Emotional map to the same Qdrant collection,
        // so re-using the original ID would overwrite the base entry.
        let mut emotional_entry = entry.clone();
        emotional_entry.id = Uuid::new_v4().to_string();
        emotional_entry.memory_type = MemoryType::Emotional;

        // Add emotional metadata to payload
        emotional_entry.payload.insert(
            "valence".to_string(),
            serde_json::json!(format!("{:?}", tag.valence)),
        );
        emotional_entry.payload.insert(
            "arousal".to_string(),
            serde_json::json!(format!("{:?}", tag.arousal)),
        );
        emotional_entry
            .payload
            .insert("fear_score".to_string(), serde_json::json!(tag.fear_score));
        emotional_entry.payload.insert(
            "outcome_value".to_string(),
            serde_json::json!(tag.outcome_value),
        );
        emotional_entry.payload.insert(
            "consolidation_priority".to_string(),
            serde_json::json!(tag.consolidation_priority()),
        );

        if let Some(ref signature) = tag.threat_signature {
            emotional_entry
                .payload
                .insert("threat_signature".to_string(), serde_json::json!(signature));
        }

        // Store both original and emotional memory
        let original_id = self.storage.store(entry).await?;
        let emotional_id = self.storage.store(emotional_entry).await?;

        info!(
            "Memory {} tagged with {:?} valence, {:?} arousal, fear={:.2}",
            original_id, tag.valence, tag.arousal, tag.fear_score
        );

        Ok(emotional_id)
    }

    /// Retrieve memories similar to a fear response
    pub async fn recall_fear_memories(
        &self,
        query_vector: Vec<f32>,
        min_fear_score: f32,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, EmotionalTag)>> {
        debug!(
            "Recalling fear memories with min_fear_score={:.2}",
            min_fear_score
        );

        let mut filters = HashMap::new();
        filters.insert("fear_score".to_string(), serde_json::json!(min_fear_score));

        let search_query = crate::hippocampus::memory::SearchQuery {
            vector: query_vector,
            limit,
            score_threshold: Some(0.7),
            filters: Some(filters),
            collection: crate::hippocampus::memory::EMBEDDINGS_COLLECTION.to_string(),
        };

        let results = self.storage.search(search_query).await?;

        let mut memories = Vec::new();
        for result in results {
            if let Some(tag) = self.extract_emotional_tag(&result.entry) {
                if tag.fear_score >= min_fear_score {
                    memories.push((result.entry, tag));
                }
            }
        }

        info!("Recalled {} fear memories", memories.len());
        Ok(memories)
    }

    /// Retrieve high-priority memories for consolidation
    pub async fn get_consolidation_candidates(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        debug!("Getting consolidation candidates");

        // Search for high-priority emotional memories
        let search_query = crate::hippocampus::memory::SearchQuery {
            vector: vec![0.0; 768], // Placeholder - will be replaced with proper query
            limit: limit * 2,       // Get more than needed to filter
            score_threshold: None,
            filters: None,
            collection: crate::hippocampus::memory::EMBEDDINGS_COLLECTION.to_string(),
        };

        let results = self.storage.search(search_query).await?;

        let mut candidates: Vec<(MemoryEntry, f32)> = results
            .into_iter()
            .filter_map(|result| {
                let priority = result
                    .entry
                    .payload
                    .get("consolidation_priority")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;

                if priority > 0.6 {
                    Some((result.entry, priority))
                } else {
                    None
                }
            })
            .collect();

        // Sort by priority descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let candidates: Vec<MemoryEntry> = candidates
            .into_iter()
            .take(limit)
            .map(|(entry, _)| entry)
            .collect();

        info!("Found {} consolidation candidates", candidates.len());
        Ok(candidates)
    }

    /// Add a custom fear memory (learned threat pattern)
    pub fn add_fear_pattern(&mut self, signature: ThreatSignature) {
        info!("Adding fear pattern: {}", signature.name);
        self.fear_network.add_fear_memory(signature);
    }

    /// Extract emotional tag from memory entry payload
    fn extract_emotional_tag(&self, entry: &MemoryEntry) -> Option<EmotionalTag> {
        // Check cache first
        if let Some(tag) = self.recent_tags.get(&entry.id) {
            return Some(tag.clone());
        }

        // Extract from payload
        let valence = entry
            .payload
            .get("valence")
            .and_then(|v| v.as_str())
            .and_then(|s| match s {
                "Positive" => Some(EmotionalValence::Positive),
                "Negative" => Some(EmotionalValence::Negative),
                "Neutral" => Some(EmotionalValence::Neutral),
                _ => None,
            })?;

        let arousal = entry
            .payload
            .get("arousal")
            .and_then(|v| v.as_str())
            .and_then(|s| match s {
                "Low" => Some(EmotionalArousal::Low),
                "Medium" => Some(EmotionalArousal::Medium),
                "High" => Some(EmotionalArousal::High),
                _ => None,
            })?;

        let fear_score = entry
            .payload
            .get("fear_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let outcome_value = entry
            .payload
            .get("outcome_value")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let threat_signature = entry
            .payload
            .get("threat_signature")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mut tag = EmotionalTag::new(valence, arousal, fear_score, outcome_value);
        tag.threat_signature = threat_signature;
        tag.timestamp = entry.timestamp;

        Some(tag)
    }

    /// Prune least important tags from cache
    fn prune_cache(&mut self) {
        if self.recent_tags.len() <= self.max_cache_size {
            return;
        }

        let target_size = (self.max_cache_size * 3) / 4; // Prune to 75%

        // Calculate priorities
        let mut entries: Vec<(String, f32)> = self
            .recent_tags
            .iter()
            .map(|(id, tag)| (id.clone(), tag.consolidation_priority()))
            .collect();

        // Sort by priority ascending (remove lowest)
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_remove = entries.len().saturating_sub(target_size);
        for (id, _) in entries.iter().take(to_remove) {
            self.recent_tags.remove(id);
        }

        debug!("Pruned cache from {} to {}", entries.len(), target_size);
    }

    /// Prune old emotional memories from database
    pub async fn prune_old_memories(&self, older_than_seconds: i64) -> Result<usize> {
        info!(
            "Pruning emotional memories older than {} seconds",
            older_than_seconds
        );

        let pruned = self
            .storage
            .prune_old_memories(
                crate::hippocampus::memory::EMBEDDINGS_COLLECTION,
                older_than_seconds,
            )
            .await?;

        info!("Pruned {} old emotional memories", pruned);
        Ok(pruned)
    }

    /// Get statistics about emotional memories
    pub async fn get_stats(&self) -> Result<EmotionalMemoryStats> {
        let db_stats = self
            .storage
            .get_stats(crate::hippocampus::memory::EMBEDDINGS_COLLECTION)
            .await?;

        Ok(EmotionalMemoryStats {
            total_memories: db_stats.points_count,
            cached_tags: self.recent_tags.len(),
            fear_patterns: self.fear_network_size(),
        })
    }

    fn fear_network_size(&self) -> usize {
        // This would require exposing the fear_memories from FearNetwork
        // For now, return a placeholder
        0
    }
}

/// Statistics about the emotional memory system
#[derive(Debug, Clone)]
pub struct EmotionalMemoryStats {
    pub total_memories: u64,
    pub cached_tags: usize,
    pub fear_patterns: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_emotional_tag_creation() {
        let tag = EmotionalTag::new(
            EmotionalValence::Negative,
            EmotionalArousal::High,
            0.85,
            -1000.0,
        );

        assert_eq!(tag.valence, EmotionalValence::Negative);
        assert_eq!(tag.arousal, EmotionalArousal::High);
        assert_eq!(tag.fear_score, 0.85);
        assert_eq!(tag.outcome_value, -1000.0);
    }

    #[test]
    fn test_consolidation_priority() {
        // High arousal + negative valence + high fear = high priority
        let high_priority = EmotionalTag::new(
            EmotionalValence::Negative,
            EmotionalArousal::High,
            0.9,
            -5000.0,
        );
        assert!(high_priority.consolidation_priority() > 0.8);
        assert!(high_priority.should_consolidate());

        // Low arousal + neutral valence + no fear = low priority
        let low_priority =
            EmotionalTag::new(EmotionalValence::Neutral, EmotionalArousal::Low, 0.0, 0.0);
        assert!(low_priority.consolidation_priority() < 0.5);
        assert!(!low_priority.should_consolidate());
    }

    #[test]
    fn test_tag_from_market_conditions() {
        let fear_network = FearNetwork::new(0.7);

        let dangerous_conditions = MarketConditions {
            volatility: 0.08,
            avg_volatility: 0.02,
            volume: 10000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.06,
        };

        let tag =
            EmotionalTag::from_market_conditions(&dangerous_conditions, &fear_network, -2000.0);

        assert_eq!(tag.valence, EmotionalValence::Negative);
        assert_eq!(tag.arousal, EmotionalArousal::High);
        assert!(tag.fear_score > 0.7);
        assert!(tag.threat_signature.is_some());
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_emotional_memory_system() {
        let mut em = EmotionalMemory::new("http://localhost:6334", 0.7)
            .await
            .unwrap();

        let conditions = MarketConditions {
            volatility: 0.05,
            avg_volatility: 0.02,
            volume: 5000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.03,
        };

        let mut payload = HashMap::new();
        payload.insert("symbol".to_string(), serde_json::json!("AAPL"));

        let entry = MemoryEntry {
            id: Uuid::new_v4().to_string(),
            vector: vec![0.5; 768],
            payload,
            timestamp: chrono::Utc::now().timestamp(),
            memory_type: MemoryType::Episode,
        };

        let id = em.tag_memory(entry, &conditions, -500.0).await.unwrap();
        assert!(!id.is_empty());
    }

    #[tokio::test]
    async fn test_emotional_memory_with_mock() {
        let mut em = EmotionalMemory::new_mock(0.7);

        let conditions = MarketConditions {
            volatility: 0.05,
            avg_volatility: 0.02,
            volume: 5000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.03,
        };

        let mut payload = HashMap::new();
        payload.insert("symbol".to_string(), serde_json::json!("AAPL"));

        let entry = MemoryEntry {
            id: "test-mock-emotional".to_string(),
            vector: vec![0.5; 768],
            payload,
            timestamp: chrono::Utc::now().timestamp(),
            memory_type: MemoryType::Episode,
        };

        let id = em.tag_memory(entry, &conditions, -500.0).await.unwrap();
        assert!(!id.is_empty());
        // tag_memory returns the emotional entry's ID, which is now a
        // separate UUID to avoid overwriting the original in Qdrant.
        assert_ne!(id, "test-mock-emotional");
    }

    #[tokio::test]
    async fn test_emotional_tagging_high_fear() {
        let mut em = EmotionalMemory::new_mock(0.5);

        // High volatility, large loss - should trigger high fear
        let dangerous_conditions = MarketConditions {
            volatility: 0.10,
            avg_volatility: 0.02,
            volume: 20000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.08,
        };

        let entry = MemoryEntry {
            id: "high-fear-test".to_string(),
            vector: vec![0.5; 768],
            payload: HashMap::new(),
            timestamp: chrono::Utc::now().timestamp(),
            memory_type: MemoryType::Episode,
        };

        let id = em
            .tag_memory(entry, &dangerous_conditions, -5000.0)
            .await
            .unwrap();
        // tag_memory returns the emotional entry's own UUID, not the
        // original entry ID.
        assert!(!id.is_empty());
        assert_ne!(id, "high-fear-test");
    }

    #[tokio::test]
    async fn test_add_fear_pattern() {
        let mut em = EmotionalMemory::new_mock(0.7);

        let signature = ThreatSignature {
            name: "test_pattern".to_string(),
            volatility_spike: true,
            volume_spike: true,
            price_gap: false,
            danger_score: 0.9,
            learned_weight: 0.0,
            detection_count: 0,
            true_positives: 0,
            false_alarms: 0,
            avg_loss_on_threat: 0.0,
            avg_loss_on_false_alarm: 0.0,
            last_updated: chrono::Utc::now().timestamp(),
        };

        em.add_fear_pattern(signature);
        // Pattern should be added without error
    }
}
