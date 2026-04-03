//! End-to-End Sentiment Pipeline Orchestrator
//!
//! Connects the BERT sentiment analyzer, the SentimentFusion engine, and the
//! Qdrant storage bridge into a single coherent pipeline.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Sentiment Pipeline                               │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
//! │  │  Raw Text  │──▶│ BertSentiment   │──▶│  SentimentFusion     │  │
//! │  │  (batch)   │   │ Analyzer        │   │  (multi-source       │  │
//! │  └────────────┘   │                 │   │   weighted fusion)   │  │
//! │                    │ • tokenize      │   │                      │  │
//! │                    │ • BERT forward  │   │ • ingest readings    │  │
//! │                    │ • softmax       │   │ • staleness decay    │  │
//! │                    │ • [CLS] embed   │   │ • conflict detect    │  │
//! │                    └────────┬────────┘   └──────────┬───────────┘  │
//! │                             │                       │              │
//! │                             ▼                       ▼              │
//! │                    ┌─────────────────┐   ┌──────────────────────┐  │
//! │                    │ SentimentQdrant │   │  FusedSentiment      │  │
//! │                    │ Bridge          │   │  • score / smoothed  │  │
//! │                    │                 │   │  • confidence        │  │
//! │                    │ • store [CLS]   │   │  • conflict metric   │  │
//! │                    │ • similarity    │   └──────────────────────┘  │
//! │                    │   search        │                             │
//! │                    └─────────────────┘                             │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_neuromorphic::thalamus::sources::sentiment_pipeline::*;
//! use janus_neuromorphic::thalamus::sources::bert_sentiment::BertSentimentConfig;
//! use janus_neuromorphic::thalamus::fusion::sentiment_fusion::SentimentFusionConfig;
//!
//! let config = SentimentPipelineConfig::default();
//! let mut pipeline = SentimentPipeline::new(config);
//!
//! // Attach a Qdrant bridge (optional — pipeline works without it)
//! pipeline.attach_qdrant(bridge);
//!
//! // Process a batch of headlines
//! let output = pipeline.process_texts(&[
//!     "Bitcoin surges past $100k",
//!     "SEC announces new crypto regulations",
//! ], None).await?;
//!
//! println!("Fused score: {:?}", output.fused);
//! println!("Stored {} embeddings", output.embeddings_stored);
//! ```

use crate::common::{Error, Result};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::bert_sentiment::{
    BatchSentimentResult, BertModel, BertModelConfig, BertSentimentAnalyzer, BertSentimentConfig,
    BertSentimentStats, SentimentResult,
};
use super::sentiment_qdrant_bridge::{
    BridgeStats, RetrievedSentiment, SentimentQdrantBridge, SimilarityQuery, StorageContext,
};
use crate::thalamus::fusion::sentiment_fusion::{
    FusedSentiment, SentimentFusion, SentimentFusionConfig, SentimentFusionStats, SentimentReading,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the end-to-end sentiment pipeline.
#[derive(Debug, Clone)]
pub struct SentimentPipelineConfig {
    /// BERT sentiment analyzer configuration.
    pub bert_config: BertSentimentConfig,

    /// SentimentFusion configuration.
    pub fusion_config: SentimentFusionConfig,

    /// Source identifier used when feeding BERT results into SentimentFusion.
    /// Each unique source_id is tracked independently by the fusion engine.
    pub source_id: String,

    /// Whether to automatically store embeddings in Qdrant after analysis.
    /// Requires a Qdrant bridge to be attached.
    pub auto_store: bool,

    /// Whether to run fusion after each `process_texts` call.
    pub auto_fuse: bool,

    /// Minimum confidence threshold for feeding results into SentimentFusion.
    /// Results below this threshold are still returned but not fused.
    pub fusion_min_confidence: f64,

    /// Whether to log pipeline events via `tracing`.
    pub enable_tracing: bool,
}

impl Default for SentimentPipelineConfig {
    fn default() -> Self {
        Self {
            bert_config: BertSentimentConfig::default(),
            fusion_config: SentimentFusionConfig::default(),
            source_id: "bert_finbert".to_string(),
            auto_store: true,
            auto_fuse: true,
            fusion_min_confidence: 0.3,
            enable_tracing: true,
        }
    }
}

impl SentimentPipelineConfig {
    /// Create a pipeline config for FinBERT with default fusion settings.
    pub fn finbert() -> Self {
        Self {
            bert_config: BertSentimentConfig::finbert(),
            source_id: "bert_finbert".to_string(),
            ..Default::default()
        }
    }

    /// Create a pipeline config for DistilBERT.
    pub fn distilbert() -> Self {
        Self {
            bert_config: BertSentimentConfig::distilbert(),
            source_id: "bert_distilbert".to_string(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline Output
// ---------------------------------------------------------------------------

/// Output from a single `process_texts` call.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Per-text BERT sentiment results.
    pub batch_result: BatchSentimentResult,

    /// Fused sentiment (if auto_fuse is enabled and enough sources exist).
    pub fused: Option<FusedSentiment>,

    /// Number of embeddings stored in Qdrant (0 if no bridge attached).
    pub embeddings_stored: usize,

    /// Number of results fed into SentimentFusion.
    pub fusion_ingested: usize,

    /// Number of results skipped for fusion (low confidence).
    pub fusion_skipped: usize,

    /// Total pipeline latency in milliseconds.
    pub latency_ms: f64,
}

/// Aggregated pipeline statistics.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total number of `process_texts` calls.
    pub total_calls: u64,

    /// Total texts processed across all calls.
    pub total_texts: u64,

    /// Total embeddings stored in Qdrant.
    pub total_embeddings_stored: u64,

    /// Total results ingested into SentimentFusion.
    pub total_fusion_ingested: u64,

    /// Total results skipped for fusion.
    pub total_fusion_skipped: u64,

    /// Total number of fused outputs produced.
    pub total_fusions: u64,

    /// Cumulative pipeline latency in milliseconds.
    pub cumulative_latency_ms: f64,

    /// Number of Qdrant store errors (non-fatal).
    pub qdrant_errors: u64,
}

impl PipelineStats {
    /// Average latency per `process_texts` call in milliseconds.
    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.cumulative_latency_ms / self.total_calls as f64
        }
    }

    /// Average texts per call.
    pub fn avg_texts_per_call(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.total_texts as f64 / self.total_calls as f64
        }
    }

    /// Fusion rate: fraction of results that were ingested into fusion.
    pub fn fusion_rate(&self) -> f64 {
        let total = self.total_fusion_ingested + self.total_fusion_skipped;
        if total == 0 {
            1.0
        } else {
            self.total_fusion_ingested as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// SentimentPipeline
// ---------------------------------------------------------------------------

/// End-to-end sentiment pipeline: BERT → SentimentFusion → Qdrant.
///
/// Orchestrates the full lifecycle of text sentiment analysis:
/// 1. BERT inference (tokenize → forward → softmax → [CLS] embedding)
/// 2. Feed results into SentimentFusion as source readings
/// 3. Store [CLS] embeddings in Qdrant for similarity retrieval
/// 4. Produce fused sentiment output
pub struct SentimentPipeline {
    /// Pipeline configuration.
    config: SentimentPipelineConfig,

    /// BERT sentiment analyzer.
    analyzer: BertSentimentAnalyzer,

    /// Multi-source sentiment fusion engine.
    fusion: SentimentFusion,

    /// Optional Qdrant storage bridge.
    qdrant_bridge: Option<Arc<SentimentQdrantBridge>>,

    /// Pipeline statistics.
    stats: PipelineStats,
}

impl SentimentPipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// The BERT model is **not** loaded at construction time. Call
    /// `load_model()` before processing texts, or use `initialize()` for
    /// random-weight testing.
    pub fn new(config: SentimentPipelineConfig) -> Self {
        let analyzer = BertSentimentAnalyzer::from_config(config.bert_config.clone());
        let fusion = SentimentFusion::with_config(config.fusion_config.clone());

        Self {
            config,
            analyzer,
            fusion,
            qdrant_bridge: None,
            stats: PipelineStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SentimentPipelineConfig::default())
    }

    /// Create a FinBERT pipeline.
    pub fn finbert() -> Self {
        Self::new(SentimentPipelineConfig::finbert())
    }

    /// Create a DistilBERT pipeline.
    pub fn distilbert() -> Self {
        Self::new(SentimentPipelineConfig::distilbert())
    }

    // -----------------------------------------------------------------------
    // Setup
    // -----------------------------------------------------------------------

    /// Attach a Qdrant bridge for embedding storage.
    ///
    /// Without a bridge, the pipeline still runs BERT and fusion but does not
    /// persist embeddings.
    pub fn attach_qdrant(&mut self, bridge: SentimentQdrantBridge) {
        self.qdrant_bridge = Some(Arc::new(bridge));
    }

    /// Attach a shared Qdrant bridge.
    pub fn attach_shared_qdrant(&mut self, bridge: Arc<SentimentQdrantBridge>) {
        self.qdrant_bridge = Some(bridge);
    }

    /// Detach the Qdrant bridge.
    pub fn detach_qdrant(&mut self) -> Option<Arc<SentimentQdrantBridge>> {
        self.qdrant_bridge.take()
    }

    /// Load the BERT model weights from the configured directory.
    ///
    /// Must be called before `process_texts()`. See
    /// [`BertSentimentAnalyzer::load_model`] for details on model resolution.
    pub fn load_model(&mut self) -> Result<()> {
        self.analyzer.load_model()?;
        if self.config.enable_tracing {
            tracing::info!(
                variant = %self.config.bert_config.model_variant,
                params = self.analyzer.num_parameters(),
                "Sentiment pipeline: BERT model loaded"
            );
        }
        Ok(())
    }

    /// Initialize the BERT model with random weights (for testing).
    pub fn initialize_model(&mut self) -> Result<()> {
        // Access the underlying model through the analyzer and initialize it
        // This requires the analyzer to expose model initialization, which it
        // does through load_model. For testing, we use a small model.
        // We'll initialize directly on the analyzer's internal model.
        //
        // Since BertSentimentAnalyzer doesn't expose initialize() directly,
        // we need to construct a fresh analyzer with an initialized model.
        // For now, we document that tests should call load_model with a test
        // model directory or use the analyzer's existing test patterns.
        Err(Error::Configuration(
            "Use load_model() with a model directory, or use \
             BertSentimentAnalyzer directly for random-weight testing"
                .into(),
        ))
    }

    /// Check if the pipeline is ready for inference.
    pub fn is_ready(&self) -> bool {
        self.analyzer.is_ready()
    }

    /// Check if a Qdrant bridge is attached.
    pub fn has_qdrant(&self) -> bool {
        self.qdrant_bridge.is_some()
    }

    // -----------------------------------------------------------------------
    // Core pipeline
    // -----------------------------------------------------------------------

    /// Process a batch of texts through the full pipeline.
    ///
    /// Steps:
    /// 1. Run BERT sentiment analysis on the texts
    /// 2. Feed each result into SentimentFusion (if confidence >= threshold)
    /// 3. Optionally fuse to produce a combined score
    /// 4. Optionally store [CLS] embeddings in Qdrant
    ///
    /// Returns a [`PipelineOutput`] containing all results.
    pub async fn process_texts(
        &mut self,
        texts: &[&str],
        context: Option<&StorageContext>,
    ) -> Result<PipelineOutput> {
        let start = std::time::Instant::now();

        // Step 1: BERT inference
        let batch_result = self.analyzer.analyze_batch(texts)?;

        // Step 2: Feed into SentimentFusion
        let (fusion_ingested, fusion_skipped) = self.ingest_into_fusion(&batch_result);

        // Step 3: Fuse
        let fused = if self.config.auto_fuse {
            let now = Self::now_secs();
            self.fusion.fuse(now)
        } else {
            None
        };

        // Step 4: Store in Qdrant
        let embeddings_stored = if self.config.auto_store {
            self.store_embeddings(&batch_result, context).await
        } else {
            0
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update stats
        self.stats.total_calls += 1;
        self.stats.total_texts += texts.len() as u64;
        self.stats.total_embeddings_stored += embeddings_stored as u64;
        self.stats.total_fusion_ingested += fusion_ingested as u64;
        self.stats.total_fusion_skipped += fusion_skipped as u64;
        self.stats.cumulative_latency_ms += latency_ms;
        if fused.is_some() {
            self.stats.total_fusions += 1;
        }

        if self.config.enable_tracing {
            tracing::debug!(
                texts = texts.len(),
                mean_score = batch_result.mean_score,
                fused = fused.is_some(),
                stored = embeddings_stored,
                latency_ms = latency_ms,
                "Sentiment pipeline: batch processed"
            );
        }

        Ok(PipelineOutput {
            batch_result,
            fused,
            embeddings_stored,
            fusion_ingested,
            fusion_skipped,
            latency_ms,
        })
    }

    /// Analyze a single text through the pipeline.
    ///
    /// Convenience wrapper around `process_texts` for single-text use.
    pub async fn process_text(
        &mut self,
        text: &str,
        context: Option<&StorageContext>,
    ) -> Result<PipelineOutput> {
        self.process_texts(&[text], context).await
    }

    /// Feed external sentiment results into the fusion engine.
    ///
    /// This allows other sentiment sources (not just BERT) to contribute to
    /// the fused score. Each source should use a unique `source_id`.
    pub fn ingest_external(
        &mut self,
        source_id: &str,
        score: f64,
        confidence: f64,
        timestamp: f64,
    ) -> Result<()> {
        let reading = SentimentReading {
            source_id: source_id.to_string(),
            score,
            confidence,
            timestamp,
        };
        self.fusion.ingest(&reading)
    }

    /// Manually trigger fusion without processing new texts.
    pub fn fuse_now(&mut self) -> Option<FusedSentiment> {
        let now = Self::now_secs();
        self.fusion.fuse(now)
    }

    /// Record an outcome observation for accuracy tracking in SentimentFusion.
    ///
    /// `actual_direction` should be positive for bullish outcomes, negative
    /// for bearish outcomes.
    pub fn record_outcome(&mut self, actual_direction: f64, timestamp: f64) {
        self.fusion.record_outcome(actual_direction, timestamp);
    }

    // -----------------------------------------------------------------------
    // Qdrant queries
    // -----------------------------------------------------------------------

    /// Search for similar sentiment embeddings in Qdrant.
    ///
    /// Returns an error if no Qdrant bridge is attached.
    pub async fn search_similar(&self, query: &SimilarityQuery) -> Result<Vec<RetrievedSentiment>> {
        let bridge = self
            .qdrant_bridge
            .as_ref()
            .ok_or_else(|| Error::Configuration("No Qdrant bridge attached".into()))?;
        bridge.search_similar(query).await
    }

    /// Search for similar sentiment in a specific market regime.
    pub async fn search_by_regime(
        &self,
        embedding: &[f32],
        regime: &str,
        limit: u64,
    ) -> Result<Vec<RetrievedSentiment>> {
        let bridge = self
            .qdrant_bridge
            .as_ref()
            .ok_or_else(|| Error::Configuration("No Qdrant bridge attached".into()))?;
        bridge.search_by_regime(embedding, regime, limit).await
    }

    /// Find sentiment embeddings similar to a given result.
    pub async fn find_similar_context(
        &self,
        result: &SentimentResult,
        limit: u64,
    ) -> Result<Vec<RetrievedSentiment>> {
        let bridge = self
            .qdrant_bridge
            .as_ref()
            .ok_or_else(|| Error::Configuration("No Qdrant bridge attached".into()))?;
        bridge.search_similar_to_result(result, limit).await
    }

    /// Get the count of stored sentiment embeddings.
    pub async fn stored_count(&self) -> Result<u64> {
        match &self.qdrant_bridge {
            Some(bridge) => bridge.count().await,
            None => Ok(0),
        }
    }

    // -----------------------------------------------------------------------
    // Model hot-reload
    // -----------------------------------------------------------------------

    /// Hot-reload BERT model weights from a new safetensors file.
    ///
    /// This replaces the current model weights without restarting the
    /// pipeline. The tokenizer and fusion state are preserved. During reload,
    /// the pipeline is temporarily unable to process texts (is_ready returns
    /// false until reload completes).
    ///
    /// # Errors
    ///
    /// Returns an error if the weights file cannot be loaded. In that case,
    /// the pipeline remains in an unloaded state and `load_model()` or another
    /// `hot_reload_weights()` call is required to restore inference capability.
    pub fn hot_reload_weights(&mut self, weights_path: &str) -> Result<()> {
        if self.config.enable_tracing {
            tracing::info!(
                path = weights_path,
                "Sentiment pipeline: hot-reloading BERT weights"
            );
        }

        // Build a new BertModel with the same config and device
        let model_config = self.analyzer.config().model_variant.clone();
        let bert_config = match &model_config {
            super::bert_sentiment::BertModelVariant::FinBert => BertModelConfig::finbert(),
            super::bert_sentiment::BertModelVariant::DistilBert => BertModelConfig::distilbert(),
            super::bert_sentiment::BertModelVariant::Custom { num_labels, .. } => BertModelConfig {
                num_labels: *num_labels,
                ..Default::default()
            },
        };

        let mut new_model = BertModel::new(bert_config, self.config.bert_config.device);
        new_model.load_weights(weights_path)?;

        // Swap the model inside the analyzer. Since BertSentimentAnalyzer
        // doesn't expose a set_model method, we reload through the analyzer's
        // own mechanism. We'll build a new analyzer preserving the tokenizer
        // state by reloading from the same config.
        //
        // The most robust approach: create a new analyzer from the same config,
        // load it, and swap. The fusion state and stats are preserved because
        // they live on the pipeline, not the analyzer.
        let mut new_analyzer = BertSentimentAnalyzer::from_config(self.config.bert_config.clone());

        // We need to re-resolve the model dir for the tokenizer, but use the
        // new weights path. The simplest approach is to create a config with
        // the parent directory of the weights file as model_dir.
        let model_dir = std::path::Path::new(weights_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string());

        if let Some(dir) = model_dir {
            let mut reload_config = self.config.bert_config.clone();
            reload_config.model_dir = Some(dir);
            new_analyzer = BertSentimentAnalyzer::from_config(reload_config);
        }

        new_analyzer
            .load_model()
            .map_err(|e| Error::Configuration(format!("Hot-reload failed: {}", e)))?;

        self.analyzer = new_analyzer;

        if self.config.enable_tracing {
            tracing::info!(
                params = self.analyzer.num_parameters(),
                "Sentiment pipeline: BERT weights hot-reloaded successfully"
            );
        }

        Ok(())
    }

    /// Hot-reload weights from the default model directory.
    ///
    /// Re-loads from the same directory that was used during initial
    /// `load_model()`. Useful for picking up updated model files.
    pub fn hot_reload(&mut self) -> Result<()> {
        if self.config.enable_tracing {
            tracing::info!("Sentiment pipeline: hot-reloading from default model directory");
        }

        let mut new_analyzer = BertSentimentAnalyzer::from_config(self.config.bert_config.clone());
        new_analyzer.load_model().map_err(|e| {
            Error::Configuration(format!("Hot-reload from default dir failed: {}", e))
        })?;

        self.analyzer = new_analyzer;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get the pipeline configuration.
    pub fn config(&self) -> &SentimentPipelineConfig {
        &self.config
    }

    /// Get the BERT analyzer (read-only access).
    pub fn analyzer(&self) -> &BertSentimentAnalyzer {
        &self.analyzer
    }

    /// Get the BERT analyzer (mutable access for direct operations).
    pub fn analyzer_mut(&mut self) -> &mut BertSentimentAnalyzer {
        &mut self.analyzer
    }

    /// Get the SentimentFusion engine (read-only).
    pub fn fusion(&self) -> &SentimentFusion {
        &self.fusion
    }

    /// Get the SentimentFusion engine (mutable).
    pub fn fusion_mut(&mut self) -> &mut SentimentFusion {
        &mut self.fusion
    }

    /// Get the Qdrant bridge (if attached).
    pub fn qdrant_bridge(&self) -> Option<&Arc<SentimentQdrantBridge>> {
        self.qdrant_bridge.as_ref()
    }

    /// Get pipeline statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get BERT analyzer statistics.
    pub fn bert_stats(&self) -> &BertSentimentStats {
        self.analyzer.stats()
    }

    /// Get fusion statistics.
    pub fn fusion_stats(&self) -> &SentimentFusionStats {
        self.fusion.stats()
    }

    /// Get Qdrant bridge statistics (if attached).
    pub async fn qdrant_stats(&self) -> Option<BridgeStats> {
        match &self.qdrant_bridge {
            Some(bridge) => Some(bridge.stats().await),
            None => None,
        }
    }

    /// Get the EMA-smoothed sentiment score from the BERT analyzer.
    pub fn smoothed_score(&self) -> f64 {
        self.analyzer.smoothed_score()
    }

    /// Get the EMA-smoothed confidence from the BERT analyzer.
    pub fn smoothed_confidence(&self) -> f64 {
        self.analyzer.smoothed_confidence()
    }

    /// Get the windowed mean sentiment score.
    pub fn windowed_mean_score(&self) -> f64 {
        self.analyzer.windowed_mean_score()
    }

    /// Check if sentiment is improving (positive trend in recent window).
    pub fn is_sentiment_improving(&self) -> Option<bool> {
        self.analyzer.is_sentiment_improving()
    }

    /// Get the number of active sources in the fusion engine.
    pub fn fusion_source_count(&self) -> usize {
        self.fusion.source_count()
    }

    /// Get the approximate number of model parameters.
    pub fn num_parameters(&self) -> usize {
        self.analyzer.num_parameters()
    }

    /// Reset all pipeline state (analyzer, fusion, stats).
    pub fn reset(&mut self) {
        self.analyzer.reset();
        self.fusion.reset();
        self.stats = PipelineStats::default();
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Feed BERT results into the SentimentFusion engine.
    ///
    /// Returns (ingested_count, skipped_count).
    fn ingest_into_fusion(&mut self, batch: &BatchSentimentResult) -> (usize, usize) {
        let now = Self::now_secs();
        let mut ingested = 0usize;
        let mut skipped = 0usize;

        for result in &batch.results {
            // Skip low-confidence results from fusion
            if result.confidence < self.config.fusion_min_confidence {
                skipped += 1;
                continue;
            }

            let reading = SentimentReading {
                source_id: self.config.source_id.clone(),
                score: result.score,
                confidence: result.confidence,
                timestamp: now,
            };

            match self.fusion.ingest(&reading) {
                Ok(()) => ingested += 1,
                Err(e) => {
                    if self.config.enable_tracing {
                        tracing::warn!(
                            error = %e,
                            score = result.score,
                            "Failed to ingest sentiment reading into fusion"
                        );
                    }
                    skipped += 1;
                }
            }
        }

        (ingested, skipped)
    }

    /// Store batch embeddings in Qdrant (best-effort, non-fatal on error).
    async fn store_embeddings(
        &mut self,
        batch: &BatchSentimentResult,
        context: Option<&StorageContext>,
    ) -> usize {
        let bridge = match &self.qdrant_bridge {
            Some(b) => b.clone(),
            None => return 0,
        };

        match bridge.store_batch(batch, context).await {
            Ok(count) => count,
            Err(e) => {
                self.stats.qdrant_errors += 1;
                if self.config.enable_tracing {
                    tracing::warn!(
                        error = %e,
                        "Sentiment pipeline: failed to store embeddings in Qdrant"
                    );
                }
                0
            }
        }
    }

    /// Current time as seconds since Unix epoch.
    fn now_secs() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use memory::qdrant_client::QdrantProductionClient;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_pipeline() -> SentimentPipeline {
        SentimentPipeline::new(SentimentPipelineConfig::default())
    }

    fn pipeline_with_qdrant() -> SentimentPipeline {
        let mut pipeline = default_pipeline();
        let client = QdrantProductionClient::mock();
        let bridge = SentimentQdrantBridge::with_defaults(client);
        pipeline.attach_qdrant(bridge);
        pipeline
    }

    // -----------------------------------------------------------------------
    // Config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let config = SentimentPipelineConfig::default();
        assert_eq!(config.source_id, "bert_finbert");
        assert!(config.auto_store);
        assert!(config.auto_fuse);
        assert!((config.fusion_min_confidence - 0.3).abs() < 1e-6);
        assert!(config.enable_tracing);
    }

    #[test]
    fn test_finbert_config() {
        let config = SentimentPipelineConfig::finbert();
        assert_eq!(config.source_id, "bert_finbert");
    }

    #[test]
    fn test_distilbert_config() {
        let config = SentimentPipelineConfig::distilbert();
        assert_eq!(config.source_id, "bert_distilbert");
    }

    // -----------------------------------------------------------------------
    // Construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_construction() {
        let pipeline = default_pipeline();
        assert!(!pipeline.is_ready());
        assert!(!pipeline.has_qdrant());
        assert_eq!(pipeline.fusion_source_count(), 0);
    }

    #[test]
    fn test_pipeline_with_defaults() {
        let pipeline = SentimentPipeline::with_defaults();
        assert!(!pipeline.is_ready());
    }

    #[test]
    fn test_finbert_pipeline() {
        let pipeline = SentimentPipeline::finbert();
        assert_eq!(pipeline.config().source_id, "bert_finbert");
    }

    #[test]
    fn test_distilbert_pipeline() {
        let pipeline = SentimentPipeline::distilbert();
        assert_eq!(pipeline.config().source_id, "bert_distilbert");
    }

    // -----------------------------------------------------------------------
    // Qdrant bridge attachment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_attach_qdrant() {
        let mut pipeline = default_pipeline();
        assert!(!pipeline.has_qdrant());

        let client = QdrantProductionClient::mock();
        let bridge = SentimentQdrantBridge::with_defaults(client);
        pipeline.attach_qdrant(bridge);
        assert!(pipeline.has_qdrant());
    }

    #[test]
    fn test_detach_qdrant() {
        let mut pipeline = pipeline_with_qdrant();
        assert!(pipeline.has_qdrant());

        let bridge = pipeline.detach_qdrant();
        assert!(bridge.is_some());
        assert!(!pipeline.has_qdrant());
    }

    #[test]
    fn test_attach_shared_qdrant() {
        let mut pipeline = default_pipeline();
        let client = QdrantProductionClient::mock();
        let bridge = Arc::new(SentimentQdrantBridge::with_defaults(client));
        pipeline.attach_shared_qdrant(bridge.clone());
        assert!(pipeline.has_qdrant());
    }

    // -----------------------------------------------------------------------
    // Fusion ingestion tests (using direct fusion manipulation)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ingest_external() {
        let mut pipeline = default_pipeline();
        let now = SentimentPipeline::now_secs();

        let result = pipeline.ingest_external("twitter_sentiment", 0.6, 0.85, now);
        assert!(result.is_ok());
        assert_eq!(pipeline.fusion_source_count(), 1);
    }

    #[test]
    fn test_ingest_external_invalid_score() {
        let mut pipeline = default_pipeline();
        let now = SentimentPipeline::now_secs();

        let result = pipeline.ingest_external("bad_source", 2.0, 0.5, now);
        assert!(result.is_err());
    }

    #[test]
    fn test_fuse_now_no_sources() {
        let mut pipeline = default_pipeline();
        // With min_sources=1 (default) and no ingested data, fuse returns None
        let fused = pipeline.fuse_now();
        assert!(fused.is_none());
    }

    #[test]
    fn test_fuse_now_with_source() {
        let mut pipeline = default_pipeline();
        let now = SentimentPipeline::now_secs();

        pipeline.ingest_external("source_a", 0.7, 0.9, now).unwrap();
        let fused = pipeline.fuse_now();
        assert!(fused.is_some());

        let f = fused.unwrap();
        assert!((f.score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_multiple_sources() {
        let mut pipeline = default_pipeline();
        let now = SentimentPipeline::now_secs();

        pipeline.ingest_external("source_a", 0.8, 0.9, now).unwrap();
        pipeline.ingest_external("source_b", 0.4, 0.8, now).unwrap();

        let fused = pipeline.fuse_now();
        assert!(fused.is_some());

        let f = fused.unwrap();
        // Should be between 0.4 and 0.8
        assert!(f.score > 0.4 && f.score < 0.8);
        assert_eq!(f.active_sources, 2);
    }

    // -----------------------------------------------------------------------
    // Record outcome tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_outcome() {
        let mut pipeline = default_pipeline();
        let now = SentimentPipeline::now_secs();

        pipeline.ingest_external("source_a", 0.5, 0.9, now).unwrap();
        // Record a positive outcome
        pipeline.record_outcome(1.0, now);
        // Should not panic
    }

    // -----------------------------------------------------------------------
    // Ingest into fusion (internal) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ingest_into_fusion_filters_low_confidence() {
        let config = SentimentPipelineConfig {
            fusion_min_confidence: 0.8,
            ..Default::default()
        };
        let mut pipeline = SentimentPipeline::new(config);

        let batch = make_batch_result(vec![
            ("high conf", 0.5, 0.9),
            ("low conf", 0.3, 0.2),
            ("medium conf", 0.7, 0.85),
        ]);

        let (ingested, skipped) = pipeline.ingest_into_fusion(&batch);
        assert_eq!(ingested, 2);
        assert_eq!(skipped, 1);
    }

    #[test]
    fn test_ingest_into_fusion_all_pass() {
        let mut pipeline = default_pipeline(); // min_confidence=0.3

        let batch = make_batch_result(vec![("a", 0.5, 0.9), ("b", 0.3, 0.8), ("c", -0.2, 0.7)]);

        let (ingested, skipped) = pipeline.ingest_into_fusion(&batch);
        assert_eq!(ingested, 3);
        assert_eq!(skipped, 0);
    }

    #[test]
    fn test_ingest_into_fusion_empty_batch() {
        let mut pipeline = default_pipeline();
        let batch = make_batch_result(vec![]);

        let (ingested, skipped) = pipeline.ingest_into_fusion(&batch);
        assert_eq!(ingested, 0);
        assert_eq!(skipped, 0);
    }

    // -----------------------------------------------------------------------
    // Stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_defaults() {
        let pipeline = default_pipeline();
        let stats = pipeline.stats();
        assert_eq!(stats.total_calls, 0);
        assert_eq!(stats.total_texts, 0);
        assert_eq!(stats.avg_latency_ms(), 0.0);
        assert_eq!(stats.avg_texts_per_call(), 0.0);
        assert_eq!(stats.fusion_rate(), 1.0);
    }

    #[test]
    fn test_pipeline_stats_avg_latency() {
        let stats = PipelineStats {
            total_calls: 4,
            cumulative_latency_ms: 200.0,
            ..Default::default()
        };
        assert!((stats.avg_latency_ms() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_stats_avg_texts() {
        let stats = PipelineStats {
            total_calls: 3,
            total_texts: 15,
            ..Default::default()
        };
        assert!((stats.avg_texts_per_call() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_stats_fusion_rate() {
        let stats = PipelineStats {
            total_fusion_ingested: 8,
            total_fusion_skipped: 2,
            ..Default::default()
        };
        assert!((stats.fusion_rate() - 0.8).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_accessors() {
        let pipeline = default_pipeline();
        assert!(!pipeline.is_ready());
        assert_eq!(pipeline.smoothed_score(), 0.0);
        assert_eq!(pipeline.smoothed_confidence(), 0.0);
        assert_eq!(pipeline.windowed_mean_score(), 0.0);
        assert_eq!(pipeline.is_sentiment_improving(), None);
        assert_eq!(pipeline.fusion_source_count(), 0);
        assert!(pipeline.num_parameters() > 0); // BERT base has millions of params
    }

    #[test]
    fn test_reset() {
        let mut pipeline = default_pipeline();
        let now = SentimentPipeline::now_secs();
        pipeline.ingest_external("src", 0.5, 0.9, now).unwrap();
        assert_eq!(pipeline.fusion_source_count(), 1);

        pipeline.reset();
        assert_eq!(pipeline.fusion_source_count(), 0);
        assert_eq!(pipeline.stats().total_calls, 0);
    }

    #[test]
    fn test_analyzer_access() {
        let pipeline = default_pipeline();
        let _ = pipeline.analyzer();
        let _ = pipeline.bert_stats();
        let _ = pipeline.fusion_stats();
    }

    #[test]
    fn test_analyzer_mut_access() {
        let mut pipeline = default_pipeline();
        let _ = pipeline.analyzer_mut();
        let _ = pipeline.fusion_mut();
    }

    // -----------------------------------------------------------------------
    // Qdrant query tests (without model — just bridge)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_search_without_bridge() {
        let pipeline = default_pipeline();
        let query = SimilarityQuery::new(vec![0.0; 768]);

        let result = pipeline.search_similar(&query).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stored_count_without_bridge() {
        let pipeline = default_pipeline();
        let count = pipeline.stored_count().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_stored_count_with_bridge() {
        let pipeline = pipeline_with_qdrant();
        let count = pipeline.stored_count().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_qdrant_stats_without_bridge() {
        let pipeline = default_pipeline();
        let stats = pipeline.qdrant_stats().await;
        assert!(stats.is_none());
    }

    #[tokio::test]
    async fn test_qdrant_stats_with_bridge() {
        let pipeline = pipeline_with_qdrant();
        let stats = pipeline.qdrant_stats().await;
        assert!(stats.is_some());
    }

    #[tokio::test]
    async fn test_search_by_regime_without_bridge() {
        let pipeline = default_pipeline();
        let result = pipeline.search_by_regime(&[0.0; 768], "bull", 5).await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Process texts without model (should fail gracefully)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_texts_without_model() {
        let mut pipeline = default_pipeline();
        let result = pipeline.process_texts(&["Hello world"], None).await;
        assert!(result.is_err()); // Model not loaded
    }

    #[tokio::test]
    async fn test_process_texts_empty() {
        let mut pipeline = default_pipeline();
        // Empty batch should succeed even without a loaded model
        // because analyze_batch handles empty input as a special case
        let result = pipeline.process_texts(&[], None).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.batch_result.count, 0);
        assert_eq!(output.fusion_ingested, 0);
        assert_eq!(output.embeddings_stored, 0);
    }

    // -----------------------------------------------------------------------
    // Pipeline output tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_output_defaults() {
        let output = PipelineOutput {
            batch_result: make_batch_result(vec![]),
            fused: None,
            embeddings_stored: 0,
            fusion_ingested: 0,
            fusion_skipped: 0,
            latency_ms: 0.0,
        };

        assert!(output.fused.is_none());
        assert_eq!(output.embeddings_stored, 0);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_batch_result(items: Vec<(&str, f64, f64)>) -> BatchSentimentResult {
        let results: Vec<SentimentResult> = items
            .iter()
            .map(|(text, score, confidence)| SentimentResult {
                text: text.to_string(),
                score: *score,
                confidence: *confidence,
                label: if *score > 0.1 {
                    "positive".to_string()
                } else if *score < -0.1 {
                    "negative".to_string()
                } else {
                    "neutral".to_string()
                },
                label_index: 0,
                class_probabilities: vec![0.7, 0.2, 0.1],
                embedding: Some(vec![0.1; 768]),
                low_confidence: *confidence < 0.5,
                num_tokens: 5,
                was_truncated: false,
            })
            .collect();

        let count = results.len();
        let mean_score = if count > 0 {
            results.iter().map(|r| r.score).sum::<f64>() / count as f64
        } else {
            0.0
        };
        let mean_confidence = if count > 0 {
            results.iter().map(|r| r.confidence).sum::<f64>() / count as f64
        } else {
            0.0
        };

        BatchSentimentResult {
            results,
            mean_score,
            median_score: mean_score,
            std_score: 0.0,
            positive_fraction: 0.0,
            negative_fraction: 0.0,
            neutral_fraction: 0.0,
            mean_confidence,
            low_confidence_count: 0,
            count,
        }
    }
}
