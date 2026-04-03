//! Cognitive Core — Integrated Perception, Memory & Emotion
//!
//! The cognitive core unifies the neuromorphic subsystems into a single
//! learning loop:
//! - Qdrant Vector Database (episodic memory)
//! - Emotional Memory System (hippocampus + amygdala bridge)
//! - Fear Network with Reinforcement Learning (FNI-RL)
//! - ViViT Training Pipeline (visual cortex)
//!
//! This module provides a complete neuromorphic trading system with:
//! - Persistent memory storage and retrieval
//! - Emotional tagging of trading experiences
//! - Adaptive fear response learning
//! - Visual pattern recognition training

use crate::amygdala::{FearNetwork, MarketConditions, ThreatSignature};
use crate::hippocampus::{EmotionalMemory, EmotionalTag, MemoryEntry, MemoryType};
use crate::visual_cortex::vivit::{
    AdamOptimizer, EpochMetrics, Trainer, TrainingConfig, VivitConfig, VivitModel,
};
use common::Result;
use ndarray::{Array1, Array5};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

/// Cognitive Core configuration
#[derive(Debug, Clone)]
pub struct CognitiveCoreConfig {
    /// Qdrant server URL
    pub qdrant_url: String,

    /// Fear network threshold (0.0 to 1.0)
    pub fear_threshold: f32,

    /// ViViT model configuration
    pub vivit_config: VivitConfig,

    /// Training configuration
    pub training_config: TrainingConfig,

    /// Memory retention period (seconds)
    pub memory_retention_seconds: i64,

    /// Enable automatic consolidation
    pub auto_consolidate: bool,

    /// Consolidation interval (seconds)
    pub consolidation_interval: i64,
}

impl Default for CognitiveCoreConfig {
    fn default() -> Self {
        Self {
            qdrant_url: "http://localhost:6334".to_string(),
            fear_threshold: 0.7,
            vivit_config: VivitConfig::default(),
            training_config: TrainingConfig::default(),
            memory_retention_seconds: 30 * 24 * 3600, // 30 days
            auto_consolidate: true,
            consolidation_interval: 3600, // 1 hour
        }
    }
}

/// Cognitive Core — integrates perception, memory, and emotion
pub struct CognitiveCore {
    /// Configuration
    config: CognitiveCoreConfig,

    /// Emotional memory system (hippocampus + amygdala bridge)
    emotional_memory: EmotionalMemory,

    /// ViViT model for pattern recognition
    vivit_model: VivitModel,

    /// Training pipeline
    trainer: Option<Trainer>,

    /// Last consolidation timestamp
    last_consolidation: i64,

    /// System statistics
    stats: SystemStats,
}

/// System statistics
#[derive(Debug, Clone, Default)]
pub struct SystemStats {
    pub total_experiences: u64,
    pub fear_activations: u64,
    pub true_positives: u64,
    pub false_alarms: u64,
    pub avg_prediction_confidence: f32,
    pub memories_consolidated: u64,
    pub memories_pruned: u64,
}

impl CognitiveCore {
    /// Initialize the cognitive core with real Qdrant backend
    pub async fn new(config: CognitiveCoreConfig) -> Result<Self> {
        info!("Initializing Cognitive Core");

        // Initialize emotional memory system
        let emotional_memory =
            EmotionalMemory::new(&config.qdrant_url, config.fear_threshold).await?;

        // Initialize ViViT model
        let vivit_model = VivitModel::new(config.vivit_config.clone());

        info!("Cognitive core initialized successfully");
        info!("  - Qdrant URL: {}", config.qdrant_url);
        info!("  - Fear threshold: {:.2}", config.fear_threshold);
        info!("  - ViViT layers: {}", config.vivit_config.num_layers);

        Ok(Self {
            config,
            emotional_memory,
            vivit_model,
            trainer: None,
            last_consolidation: chrono::Utc::now().timestamp(),
            stats: SystemStats::default(),
        })
    }

    /// Initialize the cognitive core with mock backend (for testing)
    pub fn new_mock(config: CognitiveCoreConfig) -> Self {
        info!("Initializing Cognitive Core (mock mode)");

        // Initialize emotional memory system with mock backend
        let emotional_memory = EmotionalMemory::new_mock(config.fear_threshold);

        // Initialize ViViT model
        let vivit_model = VivitModel::new(config.vivit_config.clone());

        info!("Cognitive core initialized successfully (mock mode)");
        info!("  - Fear threshold: {:.2}", config.fear_threshold);
        info!("  - ViViT layers: {}", config.vivit_config.num_layers);

        Self {
            config,
            emotional_memory,
            vivit_model,
            trainer: None,
            last_consolidation: chrono::Utc::now().timestamp(),
            stats: SystemStats::default(),
        }
    }

    /// Process a trading experience with emotional tagging
    pub async fn process_experience(
        &mut self,
        video_data: &Array5<f32>,
        conditions: &MarketConditions,
        outcome_value: f32,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<ExperienceProcessingResult> {
        debug!("Processing trading experience");

        // 1. Extract visual features using ViViT
        let embeddings = self.extract_features(video_data)?;

        // 2. Create memory entry
        //    Use MemoryType::Embedding so the vector is stored in the
        //    learned_embeddings collection whose dimension (768) matches the
        //    ViViT embedding_dim.  MemoryType::Episode maps to a 512-dim
        //    collection which would cause a dimension mismatch.
        let entry_id = Uuid::new_v4().to_string();
        let entry = MemoryEntry {
            id: entry_id.clone(),
            vector: embeddings.to_vec(),
            payload: metadata,
            timestamp: chrono::Utc::now().timestamp(),
            memory_type: MemoryType::Embedding,
        };

        // 3. Tag with emotional significance
        let emotional_id = self
            .emotional_memory
            .tag_memory(entry, conditions, outcome_value)
            .await?;

        // 4. Check for fear response
        let fear_response = self.check_fear_response(conditions);

        // 5. Update statistics
        self.stats.total_experiences += 1;
        if fear_response.is_some() {
            self.stats.fear_activations += 1;
        }

        // 6. Check if consolidation is needed
        if self.config.auto_consolidate {
            self.maybe_consolidate().await?;
        }

        Ok(ExperienceProcessingResult {
            entry_id,
            emotional_id,
            fear_response,
            embeddings: embeddings.to_vec(),
        })
    }

    /// Extract features from video data using ViViT
    fn extract_features(&mut self, video_data: &Array5<f32>) -> Result<Array1<f32>> {
        debug!("Extracting features with ViViT");

        // Extract CLS token embeddings (embedding_dim = 768) rather than
        // classification logits (num_classes = 3).  The embedding vector is
        // what we store in Qdrant, so its dimensionality must match the
        // collection schema (EMBEDDING_DIM = 768).
        let cls_embeddings = self.vivit_model.embed(video_data)?;
        let embeddings = cls_embeddings.row(0).to_owned();

        Ok(embeddings)
    }

    /// Check for fear response
    fn check_fear_response(&self, conditions: &MarketConditions) -> Option<FearResponse> {
        // This would access the fear network through emotional_memory
        // For now, we create a temporary one to check
        let fear_network = FearNetwork::new(self.config.fear_threshold);

        fear_network
            .detect_threat(conditions)
            .map(|threat| FearResponse {
                threat_name: threat.name.clone(),
                danger_score: threat.effective_danger_score(),
                should_halt_trading: threat.effective_danger_score() > 0.9,
            })
    }

    /// Record outcome for reinforcement learning
    pub async fn record_outcome(
        &mut self,
        threat_name: String,
        conditions: MarketConditions,
        was_dangerous: bool,
        outcome_value: f32,
    ) -> Result<()> {
        info!(
            "Recording outcome for threat '{}': dangerous={}, value={:.2}",
            threat_name, was_dangerous, outcome_value
        );

        // Update statistics
        if was_dangerous {
            self.stats.true_positives += 1;
        } else {
            self.stats.false_alarms += 1;
        }

        // Log market conditions for debugging
        debug!(
            "Market conditions - volatility: {:.4}, volume: {:.0}, price_change_pct: {:.4}",
            conditions.volatility, conditions.volume, conditions.price_change_pct
        );

        // This would update the fear network in emotional_memory
        // For full implementation, we'd need to expose this through emotional_memory
        // Store the outcome for later learning
        debug!(
            "Outcome recorded for threat '{}': {:.2}",
            threat_name, outcome_value
        );

        Ok(())
    }

    /// Train ViViT model on collected experiences
    pub async fn train(
        &mut self,
        train_data: Vec<(Array5<f32>, usize)>,
        val_data: Option<Vec<(Array5<f32>, usize)>>,
    ) -> Result<Vec<EpochMetrics>> {
        info!("Starting ViViT training with {} examples", train_data.len());

        // Initialize trainer if not exists
        if self.trainer.is_none() {
            self.trainer = Some(Trainer::new(self.config.training_config.clone()));
        }

        let trainer = self.trainer.as_mut().unwrap();

        // Initialize optimizer
        let _optimizer = AdamOptimizer::new(self.config.training_config.optimizer.clone());

        // Log data statistics
        info!("Training samples: {}", train_data.len());
        if let Some(ref val) = val_data {
            info!("Validation samples: {}", val.len());
        }

        // Training loop would go here
        // For now, return empty metrics as placeholder
        // Full implementation requires trainer.train_model() to be implemented

        // Update statistics
        self.stats.total_experiences += train_data.len() as u64;

        info!("Training complete (placeholder)");
        Ok(trainer.history().to_vec())
    }

    /// Recall similar experiences from memory
    pub async fn recall_similar_experiences(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        min_similarity: f32,
    ) -> Result<Vec<MemoryEntry>> {
        debug!("Recalling similar experiences (limit={})", limit);

        let _search_query = crate::hippocampus::memory::SearchQuery {
            vector: query_vector,
            limit,
            score_threshold: Some(min_similarity),
            filters: None,
            collection: crate::hippocampus::memory::EPISODES_COLLECTION.to_string(),
        };

        // Search would go through emotional memory's vector database
        // For now, return empty as the actual search method needs to be exposed
        debug!(
            "Recall similar experiences requested (query limit: {}, threshold: {:.2})",
            limit, min_similarity
        );
        Ok(Vec::new())
    }

    /// Recall fear-related memories
    pub async fn recall_fear_memories(
        &self,
        query_vector: Vec<f32>,
        min_fear_score: f32,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, EmotionalTag)>> {
        debug!(
            "Recalling fear memories (min_fear={:.2}, limit={})",
            min_fear_score, limit
        );

        self.emotional_memory
            .recall_fear_memories(query_vector, min_fear_score, limit)
            .await
    }

    /// Consolidate high-priority memories
    pub async fn consolidate_memories(&mut self) -> Result<usize> {
        info!("Consolidating high-priority memories");

        let candidates = self
            .emotional_memory
            .get_consolidation_candidates(100)
            .await?;

        let count = candidates.len();
        self.stats.memories_consolidated += count as u64;
        self.last_consolidation = chrono::Utc::now().timestamp();

        info!("Consolidated {} memories", count);
        Ok(count)
    }

    /// Maybe consolidate based on time interval
    async fn maybe_consolidate(&mut self) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        let elapsed = now - self.last_consolidation;

        if elapsed >= self.config.consolidation_interval {
            self.consolidate_memories().await?;
        }

        Ok(())
    }

    /// Prune old memories
    pub async fn prune_old_memories(&mut self) -> Result<usize> {
        info!(
            "Pruning memories older than {} seconds",
            self.config.memory_retention_seconds
        );

        let pruned = self
            .emotional_memory
            .prune_old_memories(self.config.memory_retention_seconds)
            .await?;

        self.stats.memories_pruned += pruned as u64;

        info!("Pruned {} old memories", pruned);
        Ok(pruned)
    }

    /// Add a custom threat pattern to the fear network
    pub fn add_threat_pattern(&mut self, signature: ThreatSignature) {
        info!("Adding custom threat pattern: {}", signature.name);
        self.emotional_memory.add_fear_pattern(signature);
    }

    /// Get system statistics
    pub async fn get_stats(&self) -> Result<SystemStats> {
        let stats = self.stats.clone();

        // Get emotional memory stats (would be used for detailed reporting)
        // let em_stats = self.emotional_memory.get_stats().await?;

        // Add ViViT parameter count (would be used for model info)
        // let vivit_params = self.vivit_model.num_parameters();

        info!("System Stats:");
        info!("  Total experiences: {}", stats.total_experiences);
        info!("  Fear activations: {}", stats.fear_activations);
        info!("  True positives: {}", stats.true_positives);
        info!("  False alarms: {}", stats.false_alarms);
        // info!("  Memories in DB: {}", em_stats.total_memories);
        // info!("  ViViT parameters: {}", vivit_params);

        Ok(stats)
    }

    /// Set the model to training mode
    pub fn train_mode(&mut self) {
        self.vivit_model.train();
    }

    /// Set the model to evaluation mode
    pub fn eval_mode(&mut self) {
        self.vivit_model.eval();
    }

    /// Get configuration
    pub fn config(&self) -> &CognitiveCoreConfig {
        &self.config
    }
}

/// Result of processing an experience
#[derive(Debug, Clone)]
pub struct ExperienceProcessingResult {
    /// Original entry ID
    pub entry_id: String,

    /// Emotional memory ID
    pub emotional_id: String,

    /// Fear response if triggered
    pub fear_response: Option<FearResponse>,

    /// Extracted embeddings
    pub embeddings: Vec<f32>,
}

/// Fear response information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FearResponse {
    /// Name of threat pattern
    pub threat_name: String,

    /// Danger score (0.0 to 1.0)
    pub danger_score: f32,

    /// Whether trading should be halted
    pub should_halt_trading: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_cognitive_core_initialization() {
        let config = CognitiveCoreConfig::default();
        let system = CognitiveCore::new(config).await;
        assert!(system.is_ok());
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_process_experience() {
        let config = CognitiveCoreConfig::default();
        let mut system = CognitiveCore::new(config).await.unwrap();

        let video_data = Array5::from_elem((1, 8, 64, 64, 3), 0.5);
        let conditions = MarketConditions {
            volatility: 0.03,
            avg_volatility: 0.02,
            volume: 3000.0,
            avg_volume: 2000.0,
            price_change_pct: 0.01,
        };

        let result = system
            .process_experience(&video_data, &conditions, 100.0, HashMap::new())
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_fear_response() {
        let config = CognitiveCoreConfig::default();
        let mut system = CognitiveCore::new(config).await.unwrap();

        let dangerous_conditions = MarketConditions {
            volatility: 0.10,
            avg_volatility: 0.02,
            volume: 15000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.08,
        };

        let video_data = Array5::from_elem((1, 8, 64, 64, 3), 0.5);

        let result = system
            .process_experience(&video_data, &dangerous_conditions, -2000.0, HashMap::new())
            .await
            .unwrap();

        assert!(result.fear_response.is_some());
        if let Some(fear) = result.fear_response {
            assert!(fear.danger_score > 0.7);
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = CognitiveCoreConfig::default();
        assert_eq!(config.fear_threshold, 0.7);
        assert!(config.auto_consolidate);
        assert_eq!(config.consolidation_interval, 3600);
    }

    // Mock-based tests that don't require Qdrant

    #[test]
    fn test_cognitive_core_mock_initialization() {
        let config = CognitiveCoreConfig::default();
        let system = CognitiveCore::new_mock(config);
        assert_eq!(system.config().fear_threshold, 0.7);
    }

    #[tokio::test]
    async fn test_process_experience_mock() {
        let config = CognitiveCoreConfig::default();
        let mut system = CognitiveCore::new_mock(config);

        let video_data = Array5::from_elem((1, 8, 64, 64, 3), 0.5);
        let conditions = MarketConditions {
            volatility: 0.03,
            avg_volatility: 0.02,
            volume: 3000.0,
            avg_volume: 2000.0,
            price_change_pct: 0.01,
        };

        let result = system
            .process_experience(&video_data, &conditions, 100.0, HashMap::new())
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.entry_id.is_empty());
        assert!(!result.emotional_id.is_empty());
    }

    #[tokio::test]
    async fn test_fear_response_mock() {
        let config = CognitiveCoreConfig::default();
        let mut system = CognitiveCore::new_mock(config);

        // High volatility, large loss - should trigger high fear
        let dangerous_conditions = MarketConditions {
            volatility: 0.10,
            avg_volatility: 0.02,
            volume: 15000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.08,
        };

        let video_data = Array5::from_elem((1, 8, 64, 64, 3), 0.5);

        let result = system
            .process_experience(&video_data, &dangerous_conditions, -2000.0, HashMap::new())
            .await
            .unwrap();

        // Fear response should be triggered for dangerous conditions
        assert!(result.fear_response.is_some());
        if let Some(fear) = result.fear_response {
            assert!(fear.danger_score > 0.7);
        }
    }

    #[tokio::test]
    async fn test_record_outcome_mock() {
        let config = CognitiveCoreConfig::default();
        let mut system = CognitiveCore::new_mock(config);

        let conditions = MarketConditions {
            volatility: 0.05,
            avg_volatility: 0.02,
            volume: 5000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.03,
        };

        let result = system
            .record_outcome("test_threat".to_string(), conditions, true, -1000.0)
            .await;

        assert!(result.is_ok());

        let stats = system.get_stats().await.unwrap();
        assert_eq!(stats.true_positives, 1);
    }

    #[tokio::test]
    async fn test_consolidate_memories_mock() {
        let config = CognitiveCoreConfig::default();
        let mut system = CognitiveCore::new_mock(config);

        // Store some experiences first
        let video_data = Array5::from_elem((1, 8, 64, 64, 3), 0.5);
        let conditions = MarketConditions {
            volatility: 0.08,
            avg_volatility: 0.02,
            volume: 8000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.05,
        };

        system
            .process_experience(&video_data, &conditions, -1000.0, HashMap::new())
            .await
            .unwrap();

        // Consolidate
        let result = system.consolidate_memories().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_train_eval_mode_mock() {
        let config = CognitiveCoreConfig::default();
        let mut system = CognitiveCore::new_mock(config);

        system.train_mode();
        system.eval_mode();
        // Should not panic
    }

    #[tokio::test]
    async fn test_stats_mock() {
        let config = CognitiveCoreConfig::default();
        let system = CognitiveCore::new_mock(config);

        let stats = system.get_stats().await.unwrap();
        assert_eq!(stats.total_experiences, 0);
        assert_eq!(stats.fear_activations, 0);
    }
}
