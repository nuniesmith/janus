//! Sentiment–Qdrant Storage Bridge
//!
//! Bridges the BERT sentiment pipeline to the Qdrant vector database,
//! storing [CLS] embeddings with rich metadata for regime-aware similarity
//! retrieval.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Sentiment → Qdrant Bridge                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────────┐      ┌────────────────────────────┐  │
//! │  │ BertSentiment    │      │   Qdrant                   │  │
//! │  │ Analyzer         │─────▶│   sentiment_embeddings     │  │
//! │  │                  │      │   collection               │  │
//! │  │ • SentimentResult│      │                            │  │
//! │  │ • [CLS] embedding│      │   • vector: 768-d          │  │
//! │  │ • score          │      │   • payload: score, label, │  │
//! │  │ • confidence     │      │     confidence, text,      │  │
//! │  └──────────────────┘      │     timestamp, regime      │  │
//! │                             └────────────────────────────┘  │
//! │                                                              │
//! │  Search modes:                                               │
//! │  • Similarity: find semantically similar past sentiment     │
//! │  • Regime-aware: filter by market regime + similarity       │
//! │  • Temporal: filter by time range + similarity              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_neuromorphic::thalamus::sources::sentiment_qdrant_bridge::*;
//!
//! // Create bridge with mock Qdrant client
//! let client = QdrantProductionClient::mock();
//! let bridge = SentimentQdrantBridge::new(client, SentimentQdrantConfig::default());
//!
//! // Store a sentiment result with embedding
//! bridge.store_sentiment(&result, None).await?;
//!
//! // Search for similar sentiment contexts
//! let query = SimilarityQuery::new(embedding_vec)
//!     .with_limit(10)
//!     .with_regime("bull_trend");
//! let similar = bridge.search_similar(&query).await?;
//! ```

use crate::common::{Error, Result};
use memory::qdrant_client::{
    CollectionSpec, DistanceMetric, QdrantProductionClient, ScoredPoint, UpsertPoint, collections,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use super::bert_sentiment::{BatchSentimentResult, SentimentResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Sentiment–Qdrant bridge.
#[derive(Debug, Clone)]
pub struct SentimentQdrantConfig {
    /// Qdrant collection name for sentiment embeddings.
    /// Defaults to `collections::SENTIMENT_EMBEDDINGS`.
    pub collection_name: String,

    /// Expected embedding dimensionality. Must match the BERT model's
    /// hidden_size (typically 768 for base models, 256 for tiny).
    pub embedding_dim: usize,

    /// Maximum number of points to upsert in a single batch call.
    /// Larger batches are chunked automatically.
    pub max_batch_size: usize,

    /// Whether to store the original text in the payload.
    /// Disable for privacy or storage constraints.
    pub store_text: bool,

    /// Maximum text length to store in payload (characters).
    /// Longer texts are truncated with "…".
    pub max_text_length: usize,

    /// Whether to skip storage for low-confidence results.
    pub skip_low_confidence: bool,

    /// Score threshold for similarity searches (minimum cosine similarity).
    pub default_score_threshold: Option<f64>,

    /// Default number of results returned by similarity search.
    pub default_search_limit: u64,
}

impl Default for SentimentQdrantConfig {
    fn default() -> Self {
        Self {
            collection_name: collections::SENTIMENT_EMBEDDINGS.to_string(),
            embedding_dim: 768,
            max_batch_size: 100,
            store_text: true,
            max_text_length: 512,
            skip_low_confidence: false,
            default_score_threshold: Some(0.7),
            default_search_limit: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Query Types
// ---------------------------------------------------------------------------

/// A query for similarity search against stored sentiment embeddings.
#[derive(Debug, Clone)]
pub struct SimilarityQuery {
    /// The query embedding vector.
    pub vector: Vec<f64>,

    /// Maximum number of results.
    pub limit: u64,

    /// Minimum cosine similarity score.
    pub score_threshold: Option<f64>,

    /// Optional regime filter (e.g. "bull_trend", "high_volatility").
    pub regime: Option<String>,

    /// Optional source filter (e.g. "finbert", "cryptopanic").
    pub source: Option<String>,

    /// Optional minimum timestamp (seconds since epoch).
    pub min_timestamp: Option<f64>,

    /// Optional maximum timestamp (seconds since epoch).
    pub max_timestamp: Option<f64>,

    /// Optional sentiment score range filter [min, max].
    pub score_range: Option<(f64, f64)>,
}

impl SimilarityQuery {
    /// Create a new similarity query from an embedding vector.
    pub fn new(vector: Vec<f64>) -> Self {
        Self {
            vector,
            limit: 10,
            score_threshold: Some(0.7),
            regime: None,
            source: None,
            min_timestamp: None,
            max_timestamp: None,
            score_range: None,
        }
    }

    /// Create from an f32 embedding (e.g. directly from BERT output).
    pub fn from_f32(vector: &[f32]) -> Self {
        Self::new(vector.iter().map(|&v| v as f64).collect())
    }

    /// Set the maximum number of results.
    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = limit;
        self
    }

    /// Set the minimum similarity score.
    pub fn with_score_threshold(mut self, threshold: f64) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    /// Filter by market regime.
    pub fn with_regime(mut self, regime: &str) -> Self {
        self.regime = Some(regime.to_string());
        self
    }

    /// Filter by source identifier.
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    /// Filter by time range.
    pub fn with_time_range(mut self, min_ts: f64, max_ts: f64) -> Self {
        self.min_timestamp = Some(min_ts);
        self.max_timestamp = Some(max_ts);
        self
    }

    /// Filter by sentiment score range.
    pub fn with_score_range(mut self, min: f64, max: f64) -> Self {
        self.score_range = Some((min, max));
        self
    }
}

/// A sentiment point retrieved from Qdrant.
#[derive(Debug, Clone)]
pub struct RetrievedSentiment {
    /// Qdrant point ID.
    pub id: String,

    /// Cosine similarity score from the search.
    pub similarity: f64,

    /// Original sentiment score [-1, 1].
    pub sentiment_score: f64,

    /// Prediction confidence.
    pub confidence: f64,

    /// Sentiment label.
    pub label: String,

    /// Source identifier.
    pub source: String,

    /// Timestamp (seconds since epoch).
    pub timestamp: f64,

    /// Market regime at time of storage.
    pub regime: Option<String>,

    /// Original text (if stored).
    pub text: Option<String>,

    /// The embedding vector (if returned by Qdrant).
    pub embedding: Option<Vec<f64>>,
}

impl RetrievedSentiment {
    /// Parse a `ScoredPoint` from Qdrant into a `RetrievedSentiment`.
    fn from_scored_point(sp: &ScoredPoint) -> Self {
        let get_str = |key: &str| -> String {
            sp.payload
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let get_f64 =
            |key: &str| -> f64 { sp.payload.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0) };

        Self {
            id: sp.id.clone(),
            similarity: sp.score,
            sentiment_score: get_f64("sentiment_score"),
            confidence: get_f64("confidence"),
            label: get_str("label"),
            source: get_str("source"),
            timestamp: get_f64("timestamp"),
            regime: sp
                .payload
                .get("regime")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            text: sp
                .payload
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            embedding: sp.vector.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Bridge Statistics
// ---------------------------------------------------------------------------

/// Statistics for the Sentiment–Qdrant bridge.
#[derive(Debug, Clone, Default)]
pub struct BridgeStats {
    /// Total number of individual embeddings stored.
    pub embeddings_stored: u64,

    /// Total number of store_sentiment calls.
    pub store_calls: u64,

    /// Total number of batch store calls.
    pub batch_store_calls: u64,

    /// Total number of search calls.
    pub search_calls: u64,

    /// Total number of results returned across all searches.
    pub search_results_returned: u64,

    /// Number of embeddings skipped (no embedding, low confidence, etc.).
    pub skipped_count: u64,

    /// Number of storage errors (non-fatal, logged and continued).
    pub storage_errors: u64,

    /// Number of search errors.
    pub search_errors: u64,

    /// Cumulative storage latency in milliseconds.
    pub cumulative_store_latency_ms: f64,

    /// Cumulative search latency in milliseconds.
    pub cumulative_search_latency_ms: f64,
}

impl BridgeStats {
    /// Average store latency in milliseconds.
    pub fn avg_store_latency_ms(&self) -> f64 {
        if self.store_calls + self.batch_store_calls == 0 {
            return 0.0;
        }
        self.cumulative_store_latency_ms / (self.store_calls + self.batch_store_calls) as f64
    }

    /// Average search latency in milliseconds.
    pub fn avg_search_latency_ms(&self) -> f64 {
        if self.search_calls == 0 {
            return 0.0;
        }
        self.cumulative_search_latency_ms / self.search_calls as f64
    }

    /// Success rate for store operations.
    pub fn store_success_rate(&self) -> f64 {
        let total = self.store_calls + self.batch_store_calls;
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.storage_errors as f64 / total as f64)
    }
}

// ---------------------------------------------------------------------------
// Metadata context for enriching stored points
// ---------------------------------------------------------------------------

/// Optional metadata to attach to a stored sentiment embedding.
#[derive(Debug, Clone, Default)]
pub struct StorageContext {
    /// Market regime label (e.g. "bull_trend", "high_volatility").
    pub regime: Option<String>,

    /// Source identifier (e.g. "finbert", "cryptopanic", "newsapi").
    pub source: Option<String>,

    /// Trading symbol / asset (e.g. "BTCUSDT").
    pub symbol: Option<String>,

    /// Override timestamp (seconds since epoch). If `None`, uses current time.
    pub timestamp: Option<f64>,

    /// Arbitrary key-value tags.
    pub tags: HashMap<String, String>,
}

impl StorageContext {
    /// Create a new context with a regime label.
    pub fn with_regime(regime: &str) -> Self {
        Self {
            regime: Some(regime.to_string()),
            ..Default::default()
        }
    }

    /// Set the source identifier.
    pub fn source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    /// Set the symbol.
    pub fn symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Set a custom timestamp.
    pub fn timestamp(mut self, ts: f64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Add a tag.
    pub fn tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// SentimentQdrantBridge
// ---------------------------------------------------------------------------

/// Bridge between the BERT sentiment pipeline and Qdrant vector storage.
///
/// Stores [CLS] embedding vectors in the `sentiment_embeddings` collection
/// with payload metadata (score, confidence, label, text, regime, timestamp).
/// Supports similarity search for regime-aware context retrieval.
pub struct SentimentQdrantBridge {
    /// Qdrant client (production or mock).
    client: Arc<QdrantProductionClient>,

    /// Bridge configuration.
    config: SentimentQdrantConfig,

    /// Accumulated statistics.
    stats: Arc<RwLock<BridgeStats>>,

    /// Whether the collection has been ensured (lazy init).
    collection_ensured: Arc<RwLock<bool>>,
}

impl SentimentQdrantBridge {
    /// Create a new bridge wrapping an existing Qdrant client.
    pub fn new(client: QdrantProductionClient, config: SentimentQdrantConfig) -> Self {
        Self {
            client: Arc::new(client),
            config,
            stats: Arc::new(RwLock::new(BridgeStats::default())),
            collection_ensured: Arc::new(RwLock::new(false)),
        }
    }

    /// Create with an already-`Arc`'d client (shared across components).
    pub fn with_shared_client(
        client: Arc<QdrantProductionClient>,
        config: SentimentQdrantConfig,
    ) -> Self {
        Self {
            client,
            config,
            stats: Arc::new(RwLock::new(BridgeStats::default())),
            collection_ensured: Arc::new(RwLock::new(false)),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(client: QdrantProductionClient) -> Self {
        Self::new(client, SentimentQdrantConfig::default())
    }

    // -----------------------------------------------------------------------
    // Collection management
    // -----------------------------------------------------------------------

    /// Ensure the sentiment embeddings collection exists.
    ///
    /// This is called lazily on the first store/search operation, but can
    /// be called explicitly during initialization.
    pub async fn ensure_collection(&self) -> Result<()> {
        {
            let ensured = self.collection_ensured.read().await;
            if *ensured {
                return Ok(());
            }
        }

        let spec = CollectionSpec::new(
            &self.config.collection_name,
            self.config.embedding_dim as u64,
            DistanceMetric::Cosine,
        )
        .with_payload_index("source")
        .with_payload_index("timestamp")
        .with_payload_index("sentiment_score");

        self.client.ensure_collection(&spec).await.map_err(|e| {
            Error::Memory(format!(
                "Failed to ensure sentiment collection '{}': {}",
                self.config.collection_name, e
            ))
        })?;

        let mut ensured = self.collection_ensured.write().await;
        *ensured = true;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Storage
    // -----------------------------------------------------------------------

    /// Store a single sentiment result's embedding in Qdrant.
    ///
    /// Returns `Ok(true)` if the embedding was stored, `Ok(false)` if it was
    /// skipped (no embedding, low confidence, etc.), or `Err` on failure.
    pub async fn store_sentiment(
        &self,
        result: &SentimentResult,
        context: Option<&StorageContext>,
    ) -> Result<bool> {
        self.ensure_collection().await?;

        let start = std::time::Instant::now();
        let mut stats = self.stats.write().await;
        stats.store_calls += 1;
        drop(stats);

        // Check if we have an embedding to store
        let embedding = match &result.embedding {
            Some(emb) if !emb.is_empty() => emb,
            _ => {
                let mut stats = self.stats.write().await;
                stats.skipped_count += 1;
                return Ok(false);
            }
        };

        // Optionally skip low-confidence results
        if self.config.skip_low_confidence && result.low_confidence {
            let mut stats = self.stats.write().await;
            stats.skipped_count += 1;
            return Ok(false);
        }

        let point = self.build_upsert_point(result, embedding, context);

        let store_result = self
            .client
            .upsert(&self.config.collection_name, &[point])
            .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut stats = self.stats.write().await;
        stats.cumulative_store_latency_ms += elapsed_ms;

        match store_result {
            Ok(()) => {
                stats.embeddings_stored += 1;
                Ok(true)
            }
            Err(e) => {
                stats.storage_errors += 1;
                Err(Error::Memory(format!(
                    "Failed to store sentiment embedding: {}",
                    e
                )))
            }
        }
    }

    /// Store embeddings from a batch sentiment result.
    ///
    /// Returns the number of embeddings actually stored (some may be skipped).
    pub async fn store_batch(
        &self,
        batch: &BatchSentimentResult,
        context: Option<&StorageContext>,
    ) -> Result<usize> {
        self.ensure_collection().await?;

        let start = std::time::Instant::now();
        {
            let mut stats = self.stats.write().await;
            stats.batch_store_calls += 1;
        }

        // Collect points that have embeddings
        let mut points: Vec<UpsertPoint> = Vec::with_capacity(batch.results.len());
        let mut skipped = 0u64;

        for result in &batch.results {
            let embedding = match &result.embedding {
                Some(emb) if !emb.is_empty() => emb,
                _ => {
                    skipped += 1;
                    continue;
                }
            };

            if self.config.skip_low_confidence && result.low_confidence {
                skipped += 1;
                continue;
            }

            points.push(self.build_upsert_point(result, embedding, context));
        }

        if points.is_empty() {
            let mut stats = self.stats.write().await;
            stats.skipped_count += skipped;
            return Ok(0);
        }

        let stored_count = points.len();

        // Chunk and upsert
        let mut errors = 0u64;
        for chunk in points.chunks(self.config.max_batch_size) {
            if let Err(e) = self
                .client
                .upsert(&self.config.collection_name, chunk)
                .await
            {
                tracing::warn!(
                    error = %e,
                    chunk_size = chunk.len(),
                    "Failed to upsert sentiment embedding chunk"
                );
                errors += 1;
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut stats = self.stats.write().await;
        stats.cumulative_store_latency_ms += elapsed_ms;
        stats.embeddings_stored += stored_count as u64;
        stats.skipped_count += skipped;
        stats.storage_errors += errors;

        if errors > 0 {
            tracing::warn!(
                errors,
                stored = stored_count,
                "Some sentiment embedding chunks failed to upsert"
            );
        }

        Ok(stored_count)
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search for similar sentiment embeddings.
    ///
    /// Returns results sorted by cosine similarity (highest first).
    pub async fn search_similar(&self, query: &SimilarityQuery) -> Result<Vec<RetrievedSentiment>> {
        self.ensure_collection().await?;

        let start = std::time::Instant::now();
        {
            let mut stats = self.stats.write().await;
            stats.search_calls += 1;
        }

        let threshold = query
            .score_threshold
            .or(self.config.default_score_threshold);

        let results = self
            .client
            .search(
                &self.config.collection_name,
                &query.vector,
                query.limit,
                threshold,
            )
            .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        match results {
            Ok(scored_points) => {
                // Apply client-side filters that Qdrant mock may not support
                let filtered = self.apply_query_filters(&scored_points, query);

                let mut stats = self.stats.write().await;
                stats.cumulative_search_latency_ms += elapsed_ms;
                stats.search_results_returned += filtered.len() as u64;

                Ok(filtered)
            }
            Err(e) => {
                let mut stats = self.stats.write().await;
                stats.search_errors += 1;
                stats.cumulative_search_latency_ms += elapsed_ms;
                Err(Error::Memory(format!(
                    "Sentiment similarity search failed: {}",
                    e
                )))
            }
        }
    }

    /// Search for sentiment embeddings similar to a given `SentimentResult`.
    ///
    /// Convenience method that extracts the embedding from the result.
    pub async fn search_similar_to_result(
        &self,
        result: &SentimentResult,
        limit: u64,
    ) -> Result<Vec<RetrievedSentiment>> {
        let embedding = result
            .embedding
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("SentimentResult has no embedding".into()))?;

        let query = SimilarityQuery::from_f32(embedding).with_limit(limit);
        self.search_similar(&query).await
    }

    /// Search for sentiment embeddings within a specific market regime.
    pub async fn search_by_regime(
        &self,
        embedding: &[f32],
        regime: &str,
        limit: u64,
    ) -> Result<Vec<RetrievedSentiment>> {
        let query = SimilarityQuery::from_f32(embedding)
            .with_limit(limit)
            .with_regime(regime);
        self.search_similar(&query).await
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Get the number of stored embeddings.
    pub async fn count(&self) -> Result<u64> {
        self.ensure_collection().await?;
        self.client
            .count(&self.config.collection_name)
            .await
            .map_err(|e| Error::Memory(format!("Failed to count sentiment embeddings: {}", e)))
    }

    /// Clear all stored sentiment embeddings.
    pub async fn clear(&self) -> Result<()> {
        self.ensure_collection().await?;
        self.client
            .clear_collection(&self.config.collection_name)
            .await
            .map_err(|e| Error::Memory(format!("Failed to clear sentiment collection: {}", e)))
    }

    /// Delete a specific embedding by ID.
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.client
            .delete(&self.config.collection_name, &[id.to_string()])
            .await
            .map(|_| ())
            .map_err(|e| Error::Memory(format!("Failed to delete sentiment embedding: {}", e)))
    }

    /// Get bridge statistics.
    pub async fn stats(&self) -> BridgeStats {
        self.stats.read().await.clone()
    }

    /// Get the underlying Qdrant client (for advanced operations).
    pub fn client(&self) -> &Arc<QdrantProductionClient> {
        &self.client
    }

    /// Get the bridge configuration.
    pub fn config(&self) -> &SentimentQdrantConfig {
        &self.config
    }

    /// Reset bridge statistics.
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = BridgeStats::default();
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Build an `UpsertPoint` from a `SentimentResult` and its embedding.
    fn build_upsert_point(
        &self,
        result: &SentimentResult,
        embedding: &[f32],
        context: Option<&StorageContext>,
    ) -> UpsertPoint {
        let vector: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();
        let now_secs = Self::now_secs();

        let timestamp = context.and_then(|c| c.timestamp).unwrap_or(now_secs);

        let mut point = UpsertPoint::new(vector)
            .with_float("sentiment_score", result.score)
            .with_float("confidence", result.confidence)
            .with_string("label", &result.label)
            .with_integer("label_index", result.label_index as i64)
            .with_float("timestamp", timestamp)
            .with_integer("num_tokens", result.num_tokens as i64)
            .with_bool("was_truncated", result.was_truncated)
            .with_bool("low_confidence", result.low_confidence);

        // Store class probabilities as a comma-separated string
        if !result.class_probabilities.is_empty() {
            let probs_str: String = result
                .class_probabilities
                .iter()
                .map(|p| format!("{:.6}", p))
                .collect::<Vec<_>>()
                .join(",");
            point = point.with_string("class_probabilities", &probs_str);
        }

        // Optionally store text
        if self.config.store_text && !result.text.is_empty() {
            let text = if result.text.len() > self.config.max_text_length {
                let truncated: String = result
                    .text
                    .chars()
                    .take(self.config.max_text_length)
                    .collect();
                format!("{}…", truncated)
            } else {
                result.text.clone()
            };
            point = point.with_string("text", &text);
        }

        // Apply context metadata
        if let Some(ctx) = context {
            if let Some(ref regime) = ctx.regime {
                point = point.with_string("regime", regime);
            }
            if let Some(ref source) = ctx.source {
                point = point.with_string("source", source);
            }
            if let Some(ref symbol) = ctx.symbol {
                point = point.with_string("symbol", symbol);
            }
            for (key, value) in &ctx.tags {
                point = point.with_string(key, value);
            }
        }

        point
    }

    /// Apply query filters client-side (for fields that Qdrant mock
    /// doesn't natively filter on).
    fn apply_query_filters(
        &self,
        points: &[ScoredPoint],
        query: &SimilarityQuery,
    ) -> Vec<RetrievedSentiment> {
        points
            .iter()
            .map(RetrievedSentiment::from_scored_point)
            .filter(|r| {
                // Regime filter
                if let Some(ref regime) = query.regime {
                    match &r.regime {
                        Some(r_regime) if r_regime == regime => {}
                        Some(_) => return false,
                        None => return false,
                    }
                }
                // Source filter
                if let Some(ref source) = query.source {
                    if r.source != *source {
                        return false;
                    }
                }
                // Timestamp filters
                if let Some(min_ts) = query.min_timestamp {
                    if r.timestamp < min_ts {
                        return false;
                    }
                }
                if let Some(max_ts) = query.max_timestamp {
                    if r.timestamp > max_ts {
                        return false;
                    }
                }
                // Score range filter
                if let Some((min_score, max_score)) = query.score_range {
                    if r.sentiment_score < min_score || r.sentiment_score > max_score {
                        return false;
                    }
                }
                true
            })
            .collect()
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
    use memory::qdrant_client::PayloadValue;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn mock_bridge() -> SentimentQdrantBridge {
        let client = QdrantProductionClient::mock();
        SentimentQdrantBridge::with_defaults(client)
    }

    fn mock_bridge_with_config(config: SentimentQdrantConfig) -> SentimentQdrantBridge {
        let client = QdrantProductionClient::mock();
        SentimentQdrantBridge::new(client, config)
    }

    fn dummy_embedding(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32) * 0.01).collect()
    }

    fn make_result(text: &str, score: f64, confidence: f64) -> SentimentResult {
        SentimentResult {
            text: text.to_string(),
            score,
            confidence,
            label: if score > 0.1 {
                "positive".to_string()
            } else if score < -0.1 {
                "negative".to_string()
            } else {
                "neutral".to_string()
            },
            label_index: if score > 0.1 {
                0
            } else if score < -0.1 {
                2
            } else {
                1
            },
            class_probabilities: vec![0.7, 0.2, 0.1],
            embedding: Some(dummy_embedding(768)),
            low_confidence: confidence < 0.5,
            num_tokens: 10,
            was_truncated: false,
        }
    }

    fn make_result_no_embedding(text: &str) -> SentimentResult {
        SentimentResult {
            text: text.to_string(),
            score: 0.5,
            confidence: 0.8,
            label: "positive".to_string(),
            label_index: 0,
            class_probabilities: vec![0.8, 0.1, 0.1],
            embedding: None,
            low_confidence: false,
            num_tokens: 5,
            was_truncated: false,
        }
    }

    fn make_batch(count: usize) -> BatchSentimentResult {
        let results: Vec<SentimentResult> = (0..count)
            .map(|i| make_result(&format!("text_{}", i), 0.5 - (i as f64 * 0.2), 0.9))
            .collect();

        let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
        let mean_score = scores.iter().sum::<f64>() / scores.len().max(1) as f64;

        BatchSentimentResult {
            count,
            mean_score,
            median_score: mean_score,
            std_score: 0.1,
            positive_fraction: 0.5,
            negative_fraction: 0.3,
            neutral_fraction: 0.2,
            mean_confidence: 0.9,
            low_confidence_count: 0,
            results,
        }
    }

    // -----------------------------------------------------------------------
    // Config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let config = SentimentQdrantConfig::default();
        assert_eq!(config.collection_name, "sentiment_embeddings");
        assert_eq!(config.embedding_dim, 768);
        assert_eq!(config.max_batch_size, 100);
        assert!(config.store_text);
        assert_eq!(config.max_text_length, 512);
        assert!(!config.skip_low_confidence);
        assert_eq!(config.default_score_threshold, Some(0.7));
        assert_eq!(config.default_search_limit, 10);
    }

    // -----------------------------------------------------------------------
    // Query tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_similarity_query_builder() {
        let query = SimilarityQuery::new(vec![1.0, 2.0, 3.0])
            .with_limit(20)
            .with_score_threshold(0.5)
            .with_regime("bull")
            .with_source("finbert")
            .with_time_range(100.0, 200.0)
            .with_score_range(-0.5, 0.5);

        assert_eq!(query.vector.len(), 3);
        assert_eq!(query.limit, 20);
        assert_eq!(query.score_threshold, Some(0.5));
        assert_eq!(query.regime, Some("bull".to_string()));
        assert_eq!(query.source, Some("finbert".to_string()));
        assert_eq!(query.min_timestamp, Some(100.0));
        assert_eq!(query.max_timestamp, Some(200.0));
        assert_eq!(query.score_range, Some((-0.5, 0.5)));
    }

    #[test]
    fn test_similarity_query_from_f32() {
        let f32_vec: Vec<f32> = vec![1.0, 2.0, 3.0];
        let query = SimilarityQuery::from_f32(&f32_vec);
        assert_eq!(query.vector.len(), 3);
        assert!((query.vector[0] - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // StorageContext tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_storage_context_builder() {
        let ctx = StorageContext::with_regime("bull_trend")
            .source("finbert")
            .symbol("BTCUSDT")
            .timestamp(12345.0)
            .tag("model_version", "v1");

        assert_eq!(ctx.regime, Some("bull_trend".to_string()));
        assert_eq!(ctx.source, Some("finbert".to_string()));
        assert_eq!(ctx.symbol, Some("BTCUSDT".to_string()));
        assert_eq!(ctx.timestamp, Some(12345.0));
        assert_eq!(ctx.tags.get("model_version"), Some(&"v1".to_string()));
    }

    // -----------------------------------------------------------------------
    // Bridge stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bridge_stats_defaults() {
        let stats = BridgeStats::default();
        assert_eq!(stats.embeddings_stored, 0);
        assert_eq!(stats.store_calls, 0);
        assert_eq!(stats.search_calls, 0);
        assert_eq!(stats.avg_store_latency_ms(), 0.0);
        assert_eq!(stats.avg_search_latency_ms(), 0.0);
        assert_eq!(stats.store_success_rate(), 1.0);
    }

    #[test]
    fn test_bridge_stats_avg_latency() {
        let stats = BridgeStats {
            store_calls: 5,
            cumulative_store_latency_ms: 100.0,
            search_calls: 10,
            cumulative_search_latency_ms: 200.0,
            ..Default::default()
        };
        assert!((stats.avg_store_latency_ms() - 20.0).abs() < 1e-6);
        assert!((stats.avg_search_latency_ms() - 20.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // UpsertPoint builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_upsert_point_basic() {
        let bridge = mock_bridge();
        let result = make_result("Bitcoin surges", 0.8, 0.95);
        let embedding = dummy_embedding(768);

        let point = bridge.build_upsert_point(&result, &embedding, None);

        assert!(!point.id.is_empty());
        assert_eq!(point.vector.len(), 768);

        // Verify payload
        assert!(point.payload.contains_key("sentiment_score"));
        assert!(point.payload.contains_key("confidence"));
        assert!(point.payload.contains_key("label"));
        assert!(point.payload.contains_key("timestamp"));
        assert!(point.payload.contains_key("text"));
    }

    #[test]
    fn test_build_upsert_point_with_context() {
        let bridge = mock_bridge();
        let result = make_result("Market crash", -0.9, 0.99);
        let embedding = dummy_embedding(768);
        let ctx = StorageContext::with_regime("bear_panic")
            .source("finbert")
            .symbol("ETHUSDT")
            .tag("batch_id", "abc123");

        let point = bridge.build_upsert_point(&result, &embedding, Some(&ctx));

        assert_eq!(
            point.payload.get("regime").and_then(|v| v.as_str()),
            Some("bear_panic")
        );
        assert_eq!(
            point.payload.get("source").and_then(|v| v.as_str()),
            Some("finbert")
        );
        assert_eq!(
            point.payload.get("symbol").and_then(|v| v.as_str()),
            Some("ETHUSDT")
        );
        assert_eq!(
            point.payload.get("batch_id").and_then(|v| v.as_str()),
            Some("abc123")
        );
    }

    #[test]
    fn test_build_upsert_point_truncates_long_text() {
        let config = SentimentQdrantConfig {
            max_text_length: 10,
            ..Default::default()
        };
        let bridge = mock_bridge_with_config(config);
        let result = make_result(
            "This is a very long text that should be truncated",
            0.5,
            0.8,
        );
        let embedding = dummy_embedding(768);

        let point = bridge.build_upsert_point(&result, &embedding, None);

        let stored_text = point.payload.get("text").and_then(|v| v.as_str()).unwrap();
        // 10 chars + "…"
        assert!(stored_text.len() <= 15); // UTF-8 "…" is 3 bytes
        assert!(stored_text.ends_with('…'));
    }

    #[test]
    fn test_build_upsert_point_no_text_when_disabled() {
        let config = SentimentQdrantConfig {
            store_text: false,
            ..Default::default()
        };
        let bridge = mock_bridge_with_config(config);
        let result = make_result("Secret text", 0.5, 0.8);
        let embedding = dummy_embedding(768);

        let point = bridge.build_upsert_point(&result, &embedding, None);
        assert!(!point.payload.contains_key("text"));
    }

    // -----------------------------------------------------------------------
    // Async storage tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_store_sentiment_basic() {
        let bridge = mock_bridge();
        let result = make_result("BTC up", 0.7, 0.9);

        let stored = bridge.store_sentiment(&result, None).await.unwrap();
        assert!(stored);

        let stats = bridge.stats().await;
        assert_eq!(stats.store_calls, 1);
        assert_eq!(stats.embeddings_stored, 1);
        assert_eq!(stats.skipped_count, 0);
    }

    #[tokio::test]
    async fn test_store_sentiment_skips_no_embedding() {
        let bridge = mock_bridge();
        let result = make_result_no_embedding("No embedding");

        let stored = bridge.store_sentiment(&result, None).await.unwrap();
        assert!(!stored);

        let stats = bridge.stats().await;
        assert_eq!(stats.store_calls, 1);
        assert_eq!(stats.embeddings_stored, 0);
        assert_eq!(stats.skipped_count, 1);
    }

    #[tokio::test]
    async fn test_store_sentiment_skips_low_confidence() {
        let config = SentimentQdrantConfig {
            skip_low_confidence: true,
            ..Default::default()
        };
        let bridge = mock_bridge_with_config(config);
        let result = make_result("Meh", 0.1, 0.3); // confidence 0.3 => low_confidence=true

        let stored = bridge.store_sentiment(&result, None).await.unwrap();
        assert!(!stored);

        let stats = bridge.stats().await;
        assert_eq!(stats.skipped_count, 1);
    }

    #[tokio::test]
    async fn test_store_sentiment_with_context() {
        let bridge = mock_bridge();
        let result = make_result("Bullish!", 0.9, 0.95);
        let ctx = StorageContext::with_regime("bull").source("finbert");

        let stored = bridge.store_sentiment(&result, Some(&ctx)).await.unwrap();
        assert!(stored);

        let stats = bridge.stats().await;
        assert_eq!(stats.embeddings_stored, 1);
    }

    #[tokio::test]
    async fn test_store_batch() {
        let bridge = mock_bridge();
        let batch = make_batch(5);

        let stored = bridge.store_batch(&batch, None).await.unwrap();
        assert_eq!(stored, 5);

        let stats = bridge.stats().await;
        assert_eq!(stats.batch_store_calls, 1);
        assert_eq!(stats.embeddings_stored, 5);
    }

    #[tokio::test]
    async fn test_store_batch_with_context() {
        let bridge = mock_bridge();
        let batch = make_batch(3);
        let ctx = StorageContext::with_regime("sideways").source("distilbert");

        let stored = bridge.store_batch(&batch, Some(&ctx)).await.unwrap();
        assert_eq!(stored, 3);
    }

    #[tokio::test]
    async fn test_store_batch_empty() {
        let bridge = mock_bridge();
        let batch = make_batch(0);

        let stored = bridge.store_batch(&batch, None).await.unwrap();
        assert_eq!(stored, 0);
    }

    #[tokio::test]
    async fn test_store_batch_skips_no_embeddings() {
        let bridge = mock_bridge();
        let mut batch = make_batch(3);
        // Remove embedding from the second result
        batch.results[1].embedding = None;

        let stored = bridge.store_batch(&batch, None).await.unwrap();
        assert_eq!(stored, 2);

        let stats = bridge.stats().await;
        assert_eq!(stats.skipped_count, 1);
    }

    // -----------------------------------------------------------------------
    // Search tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_search_similar_empty_collection() {
        let bridge = mock_bridge();
        let query = SimilarityQuery::new(vec![0.0; 768]);

        let results = bridge.search_similar(&query).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_after_store() {
        let bridge = mock_bridge();
        let result = make_result("Fed hikes rates", -0.6, 0.9);
        bridge.store_sentiment(&result, None).await.unwrap();

        // Search with the same embedding (should match)
        let embedding: Vec<f64> = result
            .embedding
            .as_ref()
            .unwrap()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let query = SimilarityQuery::new(embedding)
            .with_limit(5)
            .with_score_threshold(0.0); // low threshold to ensure match

        let results = bridge.search_similar(&query).await.unwrap();
        // With mock client, we should get at least 1 result
        assert!(!results.is_empty());

        let stats = bridge.stats().await;
        assert_eq!(stats.search_calls, 1);
        assert!(stats.search_results_returned > 0);
    }

    #[tokio::test]
    async fn test_search_similar_to_result() {
        let bridge = mock_bridge();
        let result = make_result("Crypto boom", 0.8, 0.95);
        bridge.store_sentiment(&result, None).await.unwrap();

        let similar = bridge.search_similar_to_result(&result, 5).await.unwrap();
        // Mock client uses cosine similarity; same embedding should match itself
        assert!(!similar.is_empty());
    }

    #[tokio::test]
    async fn test_search_similar_to_result_no_embedding() {
        let bridge = mock_bridge();
        let result = make_result_no_embedding("No embedding");

        let err = bridge.search_similar_to_result(&result, 5).await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_search_by_regime() {
        let bridge = mock_bridge();
        let result = make_result("Bull run!", 0.9, 0.95);
        let ctx = StorageContext::with_regime("bull_trend");
        bridge.store_sentiment(&result, Some(&ctx)).await.unwrap();

        let embedding = dummy_embedding(768);
        let results = bridge
            .search_by_regime(&embedding, "bull_trend", 5)
            .await
            .unwrap();

        // All returned results should have the "bull_trend" regime
        for r in &results {
            assert_eq!(r.regime.as_deref(), Some("bull_trend"));
        }
    }

    // -----------------------------------------------------------------------
    // Client-side filter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_query_filters_regime() {
        let bridge = mock_bridge();
        let points = vec![
            make_scored_point("1", 0.9, Some("bull")),
            make_scored_point("2", 0.8, Some("bear")),
            make_scored_point("3", 0.7, None),
        ];

        let query = SimilarityQuery::new(vec![]).with_regime("bull");
        let filtered = bridge.apply_query_filters(&points, &query);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "1");
    }

    #[test]
    fn test_apply_query_filters_score_range() {
        let bridge = mock_bridge();
        let points = vec![
            make_scored_point_with_score("1", 0.9, 0.8),
            make_scored_point_with_score("2", 0.8, -0.5),
            make_scored_point_with_score("3", 0.7, 0.0),
        ];

        let query = SimilarityQuery::new(vec![]).with_score_range(-0.1, 0.5);
        let filtered = bridge.apply_query_filters(&points, &query);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "3");
    }

    #[test]
    fn test_apply_query_filters_no_filters() {
        let bridge = mock_bridge();
        let points = vec![
            make_scored_point("1", 0.9, None),
            make_scored_point("2", 0.8, None),
        ];

        let query = SimilarityQuery::new(vec![]);
        let filtered = bridge.apply_query_filters(&points, &query);

        assert_eq!(filtered.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Maintenance tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_count_empty() {
        let bridge = mock_bridge();
        let count = bridge.count().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_count_after_store() {
        let bridge = mock_bridge();
        let result = make_result("test", 0.5, 0.8);
        bridge.store_sentiment(&result, None).await.unwrap();

        let count = bridge.count().await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_clear_collection() {
        let bridge = mock_bridge();
        let result = make_result("test", 0.5, 0.8);
        bridge.store_sentiment(&result, None).await.unwrap();

        bridge.clear().await.unwrap();
        let count = bridge.count().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_reset_stats() {
        let bridge = mock_bridge();
        let result = make_result("test", 0.5, 0.8);
        bridge.store_sentiment(&result, None).await.unwrap();

        let stats = bridge.stats().await;
        assert!(stats.store_calls > 0);

        bridge.reset_stats().await;
        let stats = bridge.stats().await;
        assert_eq!(stats.store_calls, 0);
        assert_eq!(stats.embeddings_stored, 0);
    }

    // -----------------------------------------------------------------------
    // RetrievedSentiment parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_retrieved_sentiment_from_scored_point() {
        let point = make_scored_point("test-id", 0.95, Some("bull"));
        let retrieved = RetrievedSentiment::from_scored_point(&point);

        assert_eq!(retrieved.id, "test-id");
        assert!((retrieved.similarity - 0.95).abs() < 1e-6);
        assert_eq!(retrieved.regime, Some("bull".to_string()));
    }

    #[test]
    fn test_retrieved_sentiment_missing_fields() {
        let point = ScoredPoint {
            id: "empty".to_string(),
            score: 0.5,
            vector: None,
            payload: HashMap::new(),
        };

        let retrieved = RetrievedSentiment::from_scored_point(&point);
        assert_eq!(retrieved.id, "empty");
        assert_eq!(retrieved.sentiment_score, 0.0);
        assert_eq!(retrieved.label, "");
        assert_eq!(retrieved.regime, None);
        assert_eq!(retrieved.text, None);
    }

    // -----------------------------------------------------------------------
    // Helpers for test ScoredPoints
    // -----------------------------------------------------------------------

    fn make_scored_point(id: &str, score: f64, regime: Option<&str>) -> ScoredPoint {
        let mut payload = HashMap::new();
        payload.insert("sentiment_score".to_string(), PayloadValue::Float(0.5));
        payload.insert("confidence".to_string(), PayloadValue::Float(0.9));
        payload.insert(
            "label".to_string(),
            PayloadValue::String("positive".to_string()),
        );
        payload.insert(
            "source".to_string(),
            PayloadValue::String("finbert".to_string()),
        );
        payload.insert("timestamp".to_string(), PayloadValue::Float(1700000000.0));
        if let Some(r) = regime {
            payload.insert("regime".to_string(), PayloadValue::String(r.to_string()));
        }

        ScoredPoint {
            id: id.to_string(),
            score,
            vector: None,
            payload,
        }
    }

    fn make_scored_point_with_score(
        id: &str,
        similarity: f64,
        sentiment_score: f64,
    ) -> ScoredPoint {
        let mut payload = HashMap::new();
        payload.insert(
            "sentiment_score".to_string(),
            PayloadValue::Float(sentiment_score),
        );
        payload.insert("confidence".to_string(), PayloadValue::Float(0.9));
        payload.insert(
            "label".to_string(),
            PayloadValue::String("test".to_string()),
        );
        payload.insert(
            "source".to_string(),
            PayloadValue::String("test".to_string()),
        );
        payload.insert("timestamp".to_string(), PayloadValue::Float(1700000000.0));

        ScoredPoint {
            id: id.to_string(),
            score: similarity,
            vector: None,
            payload,
        }
    }
}
