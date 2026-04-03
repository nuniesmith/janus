//! Vector Database Integration (Qdrant)
//!
//! Provides persistent vector storage for episodic memories, market patterns,
//! and learned representations. Integrates with Qdrant vector database for
//! efficient similarity search and retrieval.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Vector Database (Qdrant)                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                               │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
//! │  │  Episodes    │  │   Patterns   │  │ Embeddings   │      │
//! │  │  Collection  │  │  Collection  │  │  Collection  │      │
//! │  └──────────────┘  └──────────────┘  └──────────────┘      │
//! │                                                               │
//! │  ┌────────────────────────────────────────────────────┐     │
//! │  │            Similarity Search Engine                 │     │
//! │  │  • Cosine Similarity                                │     │
//! │  │  • Euclidean Distance                               │     │
//! │  │  • Dot Product                                      │     │
//! │  └────────────────────────────────────────────────────┘     │
//! │                                                               │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use common::{JanusError, Result};
use qdrant_client::{
    Qdrant,
    qdrant::{
        Condition, CreateCollection, Distance, FieldCondition, Filter, PointId, PointStruct,
        PointsIdsList, PointsSelector, Range, SearchPoints, Value, VectorParams, VectorsConfig,
        vectors_config::Config,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Vector dimensionality for different embedding types
pub const EPISODE_DIM: u64 = 512;
pub const PATTERN_DIM: u64 = 256;
pub const EMBEDDING_DIM: u64 = 768;

/// Collection names
pub const EPISODES_COLLECTION: &str = "trading_episodes";
pub const PATTERNS_COLLECTION: &str = "market_patterns";
pub const EMBEDDINGS_COLLECTION: &str = "learned_embeddings";

/// Vector database client for managing episodic and semantic memories
pub struct VectorDB {
    /// Qdrant client
    client: Qdrant,

    /// Default collection for episodes
    episodes_collection: String,

    /// Default collection for patterns
    patterns_collection: String,

    /// Default collection for embeddings
    embeddings_collection: String,

    /// Connection status
    connected: bool,
}

/// Memory entry stored in vector database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: String,

    /// Vector embedding
    pub vector: Vec<f32>,

    /// Metadata payload
    pub payload: HashMap<String, serde_json::Value>,

    /// Timestamp (Unix epoch)
    pub timestamp: i64,

    /// Memory type
    pub memory_type: MemoryType,
}

/// Type of memory stored
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Trading episode (state, action, reward sequence)
    Episode,

    /// Market pattern (recognized chart pattern)
    Pattern,

    /// Learned embedding (from ViViT or other models)
    Embedding,

    /// Emotional memory (fear, greed associations)
    Emotional,
}

/// Search query for vector similarity
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// Query vector
    pub vector: Vec<f32>,

    /// Number of results to return
    pub limit: usize,

    /// Minimum similarity score (0.0 to 1.0)
    pub score_threshold: Option<f32>,

    /// Metadata filters
    pub filters: Option<HashMap<String, serde_json::Value>>,

    /// Collection to search
    pub collection: String,
}

/// Search result from vector database
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Memory entry
    pub entry: MemoryEntry,

    /// Similarity score
    pub score: f32,
}

impl VectorDB {
    /// Create a new vector database client
    ///
    /// # Arguments
    ///
    /// * `url` - Qdrant server URL (e.g., "http://localhost:6334")
    ///
    /// # Example
    ///
    /// ```no_run
    /// use janus_neuromorphic::hippocampus::memory::VectorDB;
    ///
    /// # async fn example() -> common::Result<()> {
    /// let db = VectorDB::new("http://localhost:6334").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(url: &str) -> Result<Self> {
        info!("Connecting to Qdrant at {}", url);

        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| JanusError::Database(format!("Failed to connect to Qdrant: {}", e)))?;

        let mut db = Self {
            client,
            episodes_collection: EPISODES_COLLECTION.to_string(),
            patterns_collection: PATTERNS_COLLECTION.to_string(),
            embeddings_collection: EMBEDDINGS_COLLECTION.to_string(),
            connected: false,
        };

        // Test connection
        db.test_connection().await?;
        db.connected = true;

        info!("VectorDB connected successfully to Qdrant");
        Ok(db)
    }

    /// Initialize all collections with proper schemas
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing VectorDB collections");

        // Create episodes collection
        self.create_collection_if_not_exists(
            &self.episodes_collection,
            EPISODE_DIM,
            Distance::Cosine,
        )
        .await?;

        // Create patterns collection
        self.create_collection_if_not_exists(
            &self.patterns_collection,
            PATTERN_DIM,
            Distance::Cosine,
        )
        .await?;

        // Create embeddings collection
        self.create_collection_if_not_exists(
            &self.embeddings_collection,
            EMBEDDING_DIM,
            Distance::Cosine,
        )
        .await?;

        info!("VectorDB collections initialized");
        Ok(())
    }

    /// Create a collection if it doesn't exist
    async fn create_collection_if_not_exists(
        &self,
        name: &str,
        vector_size: u64,
        distance: Distance,
    ) -> Result<()> {
        // Check if collection exists
        let collections = self
            .client
            .list_collections()
            .await
            .map_err(|e| JanusError::Database(format!("Failed to list collections: {}", e)))?;

        let exists = collections.collections.iter().any(|c| c.name == name);

        if exists {
            info!("Collection '{}' already exists", name);
            return Ok(());
        }

        info!(
            "Creating collection '{}' with vector size {} and distance {:?}",
            name, vector_size, distance
        );

        self.client
            .create_collection(CreateCollection {
                collection_name: name.to_string(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: vector_size,
                        distance: distance.into(),
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await
            .map_err(|e| JanusError::Internal(format!("Failed to create collection: {}", e)))?;

        info!("Collection '{}' created successfully", name);
        Ok(())
    }

    /// Test connection to Qdrant
    async fn test_connection(&self) -> Result<()> {
        self.client
            .list_collections()
            .await
            .map_err(|e| JanusError::Internal(format!("Qdrant connection test failed: {}", e)))?;
        Ok(())
    }

    /// Store a memory entry
    ///
    /// # Arguments
    ///
    /// * `entry` - Memory entry to store
    ///
    /// # Returns
    ///
    /// The ID of the stored entry
    pub async fn store(&self, entry: MemoryEntry) -> Result<String> {
        let collection = self.get_collection_for_type(entry.memory_type);

        debug!("Storing entry {} in collection '{}'", entry.id, collection);

        // Convert payload to Qdrant format
        let mut payload: HashMap<String, Value> = HashMap::new();

        // Add timestamp as indexed field
        payload.insert("timestamp".to_string(), Value::from(entry.timestamp));

        // Add memory type
        payload.insert(
            "memory_type".to_string(),
            Value::from(format!("{:?}", entry.memory_type)),
        );

        // Add custom payload fields
        for (key, value) in &entry.payload {
            payload.insert(key.clone(), json_to_qdrant_value(value));
        }

        // Create point
        let point = PointStruct::new(entry.id.clone(), entry.vector.clone(), payload);

        // Upsert point
        self.client
            .upsert_points(qdrant_client::qdrant::UpsertPoints {
                collection_name: collection.to_string(),
                points: vec![point],
                ..Default::default()
            })
            .await
            .map_err(|e| JanusError::Internal(format!("Failed to store entry: {}", e)))?;

        debug!("Entry {} stored successfully", entry.id);
        Ok(entry.id)
    }

    /// Store multiple memory entries in batch
    pub async fn store_batch(&self, entries: Vec<MemoryEntry>) -> Result<Vec<String>> {
        if entries.is_empty() {
            return Ok(vec![]);
        }

        info!("Batch storing {} entries", entries.len());

        // Group entries by collection
        let mut by_collection: HashMap<String, Vec<PointStruct>> = HashMap::new();

        for entry in &entries {
            let collection = self.get_collection_for_type(entry.memory_type).to_string();

            let mut payload: HashMap<String, Value> = HashMap::new();
            payload.insert("timestamp".to_string(), Value::from(entry.timestamp));
            payload.insert(
                "memory_type".to_string(),
                Value::from(format!("{:?}", entry.memory_type)),
            );

            for (key, value) in &entry.payload {
                payload.insert(key.clone(), json_to_qdrant_value(value));
            }

            let point = PointStruct::new(entry.id.clone(), entry.vector.clone(), payload);

            by_collection.entry(collection).or_default().push(point);
        }

        // Upsert to each collection
        for (collection, points) in by_collection {
            self.client
                .upsert_points(qdrant_client::qdrant::UpsertPoints {
                    collection_name: collection.to_string(),
                    points,
                    ..Default::default()
                })
                .await
                .map_err(|e| {
                    JanusError::Internal(format!("Failed to batch store entries: {}", e))
                })?;
        }

        let ids: Vec<String> = entries.iter().map(|e| e.id.clone()).collect();
        info!("Batch stored {} entries successfully", ids.len());
        Ok(ids)
    }

    /// Search for similar vectors
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        debug!(
            "Searching collection '{}' with limit {}",
            query.collection, query.limit
        );

        let mut search_request = SearchPoints {
            collection_name: query.collection.clone(),
            vector: query.vector.clone(),
            limit: query.limit as u64,
            with_payload: Some(true.into()),
            score_threshold: query.score_threshold,
            ..Default::default()
        };

        // Add filters if provided
        if let Some(filters) = &query.filters {
            if !filters.is_empty() {
                let mut conditions = Vec::new();

                for (key, value) in filters {
                    conditions.push(Condition {
                        condition_one_of: Some(
                            qdrant_client::qdrant::condition::ConditionOneOf::Field(
                                FieldCondition {
                                    key: key.clone(),
                                    r#match: Some(qdrant_client::qdrant::Match {
                                        match_value: Some(json_to_match_value(value)),
                                    }),
                                    ..Default::default()
                                },
                            ),
                        ),
                    });
                }

                search_request.filter = Some(Filter {
                    must: conditions,
                    ..Default::default()
                });
            }
        }

        let search_result = self
            .client
            .search_points(search_request)
            .await
            .map_err(|e| JanusError::Internal(format!("Search failed: {}", e)))?;

        let mut results = Vec::new();

        for scored_point in search_result.result {
            let id = match scored_point.id {
                Some(PointId {
                    point_id_options: Some(opts),
                }) => match opts {
                    qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid) => uuid,
                    qdrant_client::qdrant::point_id::PointIdOptions::Num(num) => num.to_string(),
                },
                _ => continue,
            };

            #[allow(deprecated)]
            let vector = scored_point
                .vectors
                .and_then(|v| v.vectors_options)
                .and_then(|opts| match opts {
                    qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(vec) => {
                        // NOTE: Uses deprecated vectors_output API — functionally correct.
                        // Will be updated when qdrant_client releases the replacement accessor.
                        Some(vec.data)
                    }
                    _ => None,
                })
                .unwrap_or_default();

            let mut payload_map = HashMap::new();
            let mut timestamp = 0i64;
            let mut memory_type = MemoryType::Episode;

            for (key, value) in &scored_point.payload {
                if key == "timestamp" {
                    if let Some(kind) = &value.kind {
                        timestamp = extract_i64_from_value(kind);
                    }
                } else if key == "memory_type" {
                    if let Some(kind) = &value.kind {
                        memory_type = parse_memory_type(kind);
                    }
                } else {
                    payload_map.insert(key.clone(), qdrant_value_to_json(value));
                }
            }

            let entry = MemoryEntry {
                id,
                vector,
                payload: payload_map,
                timestamp,
                memory_type,
            };

            results.push(SearchResult {
                entry,
                score: scored_point.score,
            });
        }

        debug!("Search returned {} results", results.len());
        Ok(results)
    }

    /// Retrieve a memory entry by ID
    pub async fn get(&self, id: &str, collection: &str) -> Result<Option<MemoryEntry>> {
        debug!("Getting entry {} from collection '{}'", id, collection);

        let points = self
            .client
            .get_points(qdrant_client::qdrant::GetPoints {
                collection_name: collection.to_string(),
                ids: vec![PointId::from(id.to_string())],
                with_payload: Some(true.into()),
                with_vectors: Some(true.into()),
                ..Default::default()
            })
            .await
            .map_err(|e| JanusError::Internal(format!("Failed to get entry: {}", e)))?;

        if points.result.is_empty() {
            return Ok(None);
        }

        let point = &points.result[0];

        #[allow(deprecated)]
        let vector = point
            .vectors
            .as_ref()
            .and_then(|v| v.vectors_options.as_ref())
            .and_then(|opts| match opts {
                qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(vec) => {
                    // NOTE: Uses deprecated vectors_output API — functionally correct.
                    // Will be updated when qdrant_client releases the replacement accessor.
                    Some(vec.data.clone())
                }
                _ => None,
            })
            .unwrap_or_default();

        let mut payload_map: HashMap<String, serde_json::Value> = HashMap::new();
        let mut timestamp = 0i64;
        let mut memory_type = MemoryType::Episode;

        for (key, value) in &point.payload {
            if key == "timestamp" {
                if let Some(kind) = &value.kind {
                    timestamp = extract_i64_from_value(kind);
                }
            } else if key == "memory_type" {
                if let Some(kind) = &value.kind {
                    memory_type = parse_memory_type(kind);
                }
            } else {
                payload_map.insert(key.clone(), qdrant_value_to_json(value));
            }
        }

        Ok(Some(MemoryEntry {
            id: id.to_string(),
            vector,
            payload: payload_map,
            timestamp,
            memory_type,
        }))
    }

    /// Delete a memory entry by ID
    pub async fn delete(&self, id: &str, collection: &str) -> Result<()> {
        debug!("Deleting entry {} from collection '{}'", id, collection);

        self.client
            .delete_points(qdrant_client::qdrant::DeletePoints {
                collection_name: collection.to_string(),
                points: Some(PointsSelector {
                    points_selector_one_of: Some(
                        qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Points(
                            PointsIdsList {
                                ids: vec![PointId::from(id.to_string())],
                            },
                        ),
                    ),
                }),
                ..Default::default()
            })
            .await
            .map_err(|e| JanusError::Database(format!("Failed to delete point: {}", e)))?;

        debug!("Entry {} deleted successfully", id);
        Ok(())
    }

    /// Delete multiple entries by filter
    pub async fn delete_by_filter(
        &self,
        collection: &str,
        filters: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        debug!("Deleting from collection '{}' by filter", collection);

        if filters.is_empty() {
            warn!("Delete by filter called with empty filters - skipping");
            return Ok(());
        }

        let mut conditions = Vec::new();

        for (key, value) in filters {
            conditions.push(Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: key.clone(),
                        r#match: Some(qdrant_client::qdrant::Match {
                            match_value: Some(json_to_match_value(&value)),
                        }),
                        ..Default::default()
                    },
                )),
            });
        }

        let filter = Filter {
            must: conditions,
            ..Default::default()
        };

        self.client
            .delete_points(qdrant_client::qdrant::DeletePoints {
                collection_name: collection.to_string(),
                points: Some(PointsSelector {
                    points_selector_one_of: Some(
                        qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Filter(filter),
                    ),
                }),
                ..Default::default()
            })
            .await
            .map_err(|e| JanusError::Internal(format!("Failed to delete by filter: {}", e)))?;

        info!("Deleted entries by filter from '{}'", collection);
        Ok(())
    }

    /// Get collection statistics
    pub async fn get_stats(&self, collection: &str) -> Result<CollectionStats> {
        debug!("Getting stats for collection '{}'", collection);

        let info =
            self.client.collection_info(collection).await.map_err(|e| {
                JanusError::Internal(format!("Failed to get collection info: {}", e))
            })?;

        let stats = CollectionStats {
            points_count: info
                .result
                .as_ref()
                .and_then(|r| r.points_count)
                .unwrap_or(0),
            segments_count: info
                .result
                .as_ref()
                .map(|r| r.segments_count as usize)
                .unwrap_or(0),
            vectors_count: info
                .result
                .as_ref()
                .and_then(|r| r.points_count)
                .unwrap_or(0),
        };

        Ok(stats)
    }

    /// Get the appropriate collection name for a memory type
    fn get_collection_for_type(&self, memory_type: MemoryType) -> &str {
        match memory_type {
            MemoryType::Episode => &self.episodes_collection,
            MemoryType::Pattern => &self.patterns_collection,
            MemoryType::Embedding | MemoryType::Emotional => &self.embeddings_collection,
        }
    }

    /// Check if connected to Qdrant
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Prune old memories based on timestamp
    pub async fn prune_old_memories(
        &self,
        collection: &str,
        older_than_seconds: i64,
    ) -> Result<usize> {
        let cutoff_timestamp = chrono::Utc::now().timestamp() - older_than_seconds;

        info!(
            "Pruning memories older than {} seconds from '{}' (cutoff: {})",
            older_than_seconds, collection, cutoff_timestamp
        );

        let filter = Filter {
            must: vec![Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: "timestamp".to_string(),
                        range: Some(Range {
                            lt: Some(cutoff_timestamp as f64),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )),
            }],
            ..Default::default()
        };

        // Get count before deletion
        let before_stats = self.get_stats(collection).await?;

        self.client
            .delete_points(qdrant_client::qdrant::DeletePoints {
                collection_name: collection.to_string(),
                points: Some(PointsSelector {
                    points_selector_one_of: Some(
                        qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Filter(
                            filter.clone(),
                        ),
                    ),
                }),
                ..Default::default()
            })
            .await
            .map_err(|e| JanusError::Internal(format!("Failed to delete old entries: {}", e)))?;

        // Get count after deletion
        let after_stats = self.get_stats(collection).await?;
        let pruned = before_stats
            .points_count
            .saturating_sub(after_stats.points_count);

        info!("Pruned {} old memories from '{}'", pruned, collection);
        Ok(pruned as usize)
    }
}

/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStats {
    pub points_count: u64,
    pub segments_count: usize,
    pub vectors_count: u64,
}

// Helper functions for type conversion

fn json_to_qdrant_value(value: &serde_json::Value) -> Value {
    use qdrant_client::qdrant::value::Kind;

    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Kind::IntegerValue(i)
            } else if let Some(f) = n.as_f64() {
                Kind::DoubleValue(f)
            } else {
                Kind::NullValue(0)
            }
        }
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        _ => Kind::StringValue(value.to_string()),
    };

    Value { kind: Some(kind) }
}

fn qdrant_value_to_json(value: &Value) -> serde_json::Value {
    use qdrant_client::qdrant::value::Kind;

    match &value.kind {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::IntegerValue(i)) => serde_json::json!(i),
        Some(Kind::DoubleValue(f)) => serde_json::json!(f),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        _ => serde_json::Value::Null,
    }
}

fn json_to_match_value(value: &serde_json::Value) -> qdrant_client::qdrant::r#match::MatchValue {
    use qdrant_client::qdrant::r#match::MatchValue;

    match value {
        serde_json::Value::Bool(b) => MatchValue::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                MatchValue::Integer(i)
            } else {
                MatchValue::Text(value.to_string())
            }
        }
        serde_json::Value::String(s) => MatchValue::Text(s.clone()),
        _ => MatchValue::Text(value.to_string()),
    }
}

fn extract_i64_from_value(kind: &qdrant_client::qdrant::value::Kind) -> i64 {
    use qdrant_client::qdrant::value::Kind;

    match kind {
        Kind::IntegerValue(i) => *i,
        Kind::DoubleValue(f) => *f as i64,
        _ => 0,
    }
}

fn parse_memory_type(kind: &qdrant_client::qdrant::value::Kind) -> MemoryType {
    use qdrant_client::qdrant::value::Kind;

    match kind {
        Kind::StringValue(s) => match s.as_str() {
            "Episode" => MemoryType::Episode,
            "Pattern" => MemoryType::Pattern,
            "Embedding" => MemoryType::Embedding,
            "Emotional" => MemoryType::Emotional,
            _ => MemoryType::Episode,
        },
        _ => MemoryType::Episode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    // These tests require a running Qdrant instance
    // Run: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_connection() {
        let db = VectorDB::new("http://localhost:6334").await;
        assert!(db.is_ok());
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_initialize_collections() {
        let db = VectorDB::new("http://localhost:6334").await.unwrap();
        let result = db.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_store_and_retrieve() {
        let db = VectorDB::new("http://localhost:6334").await.unwrap();
        db.initialize().await.unwrap();

        let mut payload = HashMap::new();
        payload.insert(
            "symbol".to_string(),
            serde_json::Value::String("AAPL".to_string()),
        );

        let test_id = Uuid::new_v4().to_string();

        let entry = MemoryEntry {
            id: test_id.clone(),
            vector: vec![0.1; EPISODE_DIM as usize],
            payload,
            timestamp: chrono::Utc::now().timestamp(),
            memory_type: MemoryType::Episode,
        };

        let id = db.store(entry.clone()).await.unwrap();
        assert_eq!(id, test_id);

        let retrieved = db.get(&id, EPISODES_COLLECTION).await.unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_search() {
        let db = VectorDB::new("http://localhost:6334").await.unwrap();
        db.initialize().await.unwrap();

        // Store a test entry
        let mut payload = HashMap::new();
        payload.insert(
            "symbol".to_string(),
            serde_json::Value::String("AAPL".to_string()),
        );

        let entry = MemoryEntry {
            id: Uuid::new_v4().to_string(),
            vector: vec![0.5; EPISODE_DIM as usize],
            payload,
            timestamp: chrono::Utc::now().timestamp(),
            memory_type: MemoryType::Episode,
        };

        db.store(entry).await.unwrap();

        // Search for similar vectors
        let query = SearchQuery {
            vector: vec![0.5; EPISODE_DIM as usize],
            limit: 10,
            score_threshold: Some(0.8),
            filters: None,
            collection: EPISODES_COLLECTION.to_string(),
        };

        let results = db.search(query).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_batch_store() {
        let db = VectorDB::new("http://localhost:6334").await.unwrap();
        db.initialize().await.unwrap();

        let entries: Vec<MemoryEntry> = (0..10)
            .map(|i| {
                let mut payload = HashMap::new();
                payload.insert("index".to_string(), serde_json::json!(i));

                MemoryEntry {
                    id: Uuid::new_v4().to_string(),
                    vector: vec![i as f32 / 10.0; EPISODE_DIM as usize],
                    payload,
                    timestamp: chrono::Utc::now().timestamp(),
                    memory_type: MemoryType::Episode,
                }
            })
            .collect();

        let ids = db.store_batch(entries).await.unwrap();
        assert_eq!(ids.len(), 10);
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_delete() {
        let db = VectorDB::new("http://localhost:6334").await.unwrap();
        db.initialize().await.unwrap();

        let mut payload = HashMap::new();
        payload.insert(
            "test".to_string(),
            serde_json::Value::String("delete_me".to_string()),
        );

        let test_id = Uuid::new_v4().to_string();

        let entry = MemoryEntry {
            id: test_id.clone(),
            vector: vec![0.1; EPISODE_DIM as usize],
            payload,
            timestamp: chrono::Utc::now().timestamp(),
            memory_type: MemoryType::Episode,
        };

        db.store(entry).await.unwrap();

        let result = db.delete(&test_id, EPISODES_COLLECTION).await;
        assert!(result.is_ok());

        let retrieved = db.get(&test_id, EPISODES_COLLECTION).await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_get_stats() {
        let db = VectorDB::new("http://localhost:6334").await.unwrap();
        db.initialize().await.unwrap();

        let stats = db.get_stats(EPISODES_COLLECTION).await.unwrap();
        // Just verify we got stats back successfully
        let _ = stats.points_count;
    }

    #[tokio::test]
    #[ignore = "Requires Qdrant instance running on localhost:6334"]
    async fn test_prune_old_memories() {
        let db = VectorDB::new("http://localhost:6334").await.unwrap();
        db.initialize().await.unwrap();

        // Store entry with old timestamp
        let mut payload = HashMap::new();
        payload.insert("test".to_string(), serde_json::json!("old_entry"));

        let old_entry = MemoryEntry {
            id: Uuid::new_v4().to_string(),
            vector: vec![0.1; EPISODE_DIM as usize],
            payload,
            timestamp: chrono::Utc::now().timestamp() - 10000, // 10000 seconds ago
            memory_type: MemoryType::Episode,
        };

        db.store(old_entry).await.unwrap();

        // Prune memories older than 5000 seconds
        let pruned = db
            .prune_old_memories(EPISODES_COLLECTION, 5000)
            .await
            .unwrap();
        assert!(pruned >= 1);
    }
}
