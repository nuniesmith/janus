//! Mock Vector Database for Testing
//!
//! Provides an in-memory vector database implementation that mimics the
//! VectorDB interface without requiring a Qdrant server. This is useful
//! for unit tests that need vector storage functionality.

use super::{
    CollectionStats, EMBEDDINGS_COLLECTION, EPISODES_COLLECTION, MemoryEntry, MemoryType,
    PATTERNS_COLLECTION, SearchQuery, SearchResult,
};
use common::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::debug;

/// In-memory mock implementation of vector database for testing
///
/// This struct provides the same interface as VectorDB but stores
/// all data in memory, making it suitable for unit tests without
/// requiring external infrastructure.
#[derive(Clone)]
pub struct MockVectorDB {
    /// In-memory storage for each collection
    collections: Arc<RwLock<HashMap<String, HashMap<String, MemoryEntry>>>>,

    /// Simulated connection status
    connected: bool,
}

impl Default for MockVectorDB {
    fn default() -> Self {
        Self::new()
    }
}

impl MockVectorDB {
    /// Create a new mock vector database
    pub fn new() -> Self {
        let mut collections = HashMap::new();
        collections.insert(EPISODES_COLLECTION.to_string(), HashMap::new());
        collections.insert(PATTERNS_COLLECTION.to_string(), HashMap::new());
        collections.insert(EMBEDDINGS_COLLECTION.to_string(), HashMap::new());

        Self {
            collections: Arc::new(RwLock::new(collections)),
            connected: true,
        }
    }

    /// Initialize collections (no-op for mock, collections are pre-created)
    pub async fn initialize(&self) -> Result<()> {
        debug!("MockVectorDB: initialize (no-op)");
        Ok(())
    }

    /// Store a memory entry
    pub async fn store(&self, entry: MemoryEntry) -> Result<String> {
        let collection = self.get_collection_for_type(entry.memory_type);
        let id = entry.id.clone();

        debug!("MockVectorDB: storing entry {} in {}", id, collection);

        let mut collections = self.collections.write().unwrap();
        let coll = collections.entry(collection).or_insert_with(HashMap::new);
        coll.insert(id.clone(), entry);

        Ok(id)
    }

    /// Store multiple memory entries
    pub async fn store_batch(&self, entries: Vec<MemoryEntry>) -> Result<Vec<String>> {
        let mut ids = Vec::with_capacity(entries.len());
        for entry in entries {
            let id = self.store(entry).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Search for similar vectors
    ///
    /// Uses cosine similarity for ranking results
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        debug!(
            "MockVectorDB: searching in {} with limit {}",
            query.collection, query.limit
        );

        let collections = self.collections.read().unwrap();
        let coll = match collections.get(&query.collection) {
            Some(c) => c,
            None => return Ok(vec![]),
        };

        let mut results: Vec<SearchResult> = coll
            .values()
            .filter_map(|entry| {
                // Calculate cosine similarity
                let score = cosine_similarity(&query.vector, &entry.vector);

                // Apply score threshold if specified
                if let Some(threshold) = query.score_threshold {
                    if score < threshold {
                        return None;
                    }
                }

                // Apply filters if specified
                if let Some(ref filters) = query.filters {
                    for (key, value) in filters {
                        if let Some(entry_value) = entry.payload.get(key) {
                            if entry_value != value {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                }

                Some(SearchResult {
                    entry: entry.clone(),
                    score,
                })
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(query.limit);

        Ok(results)
    }

    /// Get a specific entry by ID
    pub async fn get(&self, collection: &str, id: &str) -> Result<Option<MemoryEntry>> {
        debug!("MockVectorDB: getting {} from {}", id, collection);

        let collections = self.collections.read().unwrap();
        let entry = collections
            .get(collection)
            .and_then(|coll| coll.get(id))
            .cloned();

        Ok(entry)
    }

    /// Delete an entry by ID
    pub async fn delete(&self, collection: &str, id: &str) -> Result<bool> {
        debug!("MockVectorDB: deleting {} from {}", id, collection);

        let mut collections = self.collections.write().unwrap();
        let deleted = collections
            .get_mut(collection)
            .map(|coll| coll.remove(id).is_some())
            .unwrap_or(false);

        Ok(deleted)
    }

    /// Delete entries matching a filter
    pub async fn delete_by_filter(
        &self,
        collection: &str,
        filters: HashMap<String, serde_json::Value>,
    ) -> Result<usize> {
        debug!("MockVectorDB: deleting by filter from {}", collection);

        let mut collections = self.collections.write().unwrap();
        let coll = match collections.get_mut(collection) {
            Some(c) => c,
            None => return Ok(0),
        };

        let to_remove: Vec<String> = coll
            .iter()
            .filter(|(_, entry)| {
                filters
                    .iter()
                    .all(|(key, value)| entry.payload.get(key).map(|v| v == value).unwrap_or(false))
            })
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            coll.remove(&id);
        }

        Ok(count)
    }

    /// Get collection statistics
    pub async fn get_stats(&self, collection: &str) -> Result<CollectionStats> {
        let collections = self.collections.read().unwrap();
        let count = collections
            .get(collection)
            .map(|c| c.len() as u64)
            .unwrap_or(0);

        Ok(CollectionStats {
            points_count: count,
            segments_count: 1,
            vectors_count: count,
        })
    }

    /// Get the collection name for a memory type
    fn get_collection_for_type(&self, memory_type: MemoryType) -> String {
        match memory_type {
            MemoryType::Episode | MemoryType::Emotional => EPISODES_COLLECTION.to_string(),
            MemoryType::Pattern => PATTERNS_COLLECTION.to_string(),
            MemoryType::Embedding => EMBEDDINGS_COLLECTION.to_string(),
        }
    }

    /// Check if connected (always true for mock)
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Prune old memories by timestamp
    pub async fn prune_old_memories(
        &self,
        collection: &str,
        max_age_seconds: i64,
    ) -> Result<usize> {
        let current_time = chrono::Utc::now().timestamp();
        let cutoff = current_time - max_age_seconds;

        debug!(
            "MockVectorDB: pruning entries older than {} from {}",
            cutoff, collection
        );

        let mut collections = self.collections.write().unwrap();
        let coll = match collections.get_mut(collection) {
            Some(c) => c,
            None => return Ok(0),
        };

        let to_remove: Vec<String> = coll
            .iter()
            .filter(|(_, entry)| entry.timestamp < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            coll.remove(&id);
        }

        Ok(count)
    }

    /// Clear all data from a collection (test utility)
    pub fn clear_collection(&self, collection: &str) {
        let mut collections = self.collections.write().unwrap();
        if let Some(coll) = collections.get_mut(collection) {
            coll.clear();
        }
    }

    /// Clear all data from all collections (test utility)
    pub fn clear_all(&self) {
        let mut collections = self.collections.write().unwrap();
        for coll in collections.values_mut() {
            coll.clear();
        }
    }

    /// Get total count across all collections (test utility)
    pub fn total_entries(&self) -> usize {
        let collections = self.collections.read().unwrap();
        collections.values().map(|c| c.len()).sum()
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_entry(id: &str, vector: Vec<f32>, memory_type: MemoryType) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            vector,
            payload: HashMap::new(),
            timestamp: chrono::Utc::now().timestamp(),
            memory_type,
        }
    }

    #[tokio::test]
    async fn test_mock_store_and_get() {
        let db = MockVectorDB::new();

        let entry = create_test_entry("test1", vec![1.0, 0.0, 0.0], MemoryType::Episode);
        let id = db.store(entry.clone()).await.unwrap();

        assert_eq!(id, "test1");

        let retrieved = db.get(EPISODES_COLLECTION, "test1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test1");
    }

    #[tokio::test]
    async fn test_mock_search() {
        let db = MockVectorDB::new();

        // Store some entries
        db.store(create_test_entry(
            "v1",
            vec![1.0, 0.0, 0.0],
            MemoryType::Episode,
        ))
        .await
        .unwrap();
        db.store(create_test_entry(
            "v2",
            vec![0.9, 0.1, 0.0],
            MemoryType::Episode,
        ))
        .await
        .unwrap();
        db.store(create_test_entry(
            "v3",
            vec![0.0, 1.0, 0.0],
            MemoryType::Episode,
        ))
        .await
        .unwrap();

        // Search for similar vectors
        let query = SearchQuery {
            vector: vec![1.0, 0.0, 0.0],
            limit: 2,
            score_threshold: Some(0.5),
            filters: None,
            collection: EPISODES_COLLECTION.to_string(),
        };

        let results = db.search(query).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entry.id, "v1"); // Exact match should be first
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn test_mock_delete() {
        let db = MockVectorDB::new();

        db.store(create_test_entry(
            "del1",
            vec![1.0, 0.0, 0.0],
            MemoryType::Episode,
        ))
        .await
        .unwrap();

        assert!(db.get(EPISODES_COLLECTION, "del1").await.unwrap().is_some());

        let deleted = db.delete(EPISODES_COLLECTION, "del1").await.unwrap();
        assert!(deleted);

        assert!(db.get(EPISODES_COLLECTION, "del1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_mock_stats() {
        let db = MockVectorDB::new();

        db.store(create_test_entry("s1", vec![1.0], MemoryType::Episode))
            .await
            .unwrap();
        db.store(create_test_entry("s2", vec![1.0], MemoryType::Episode))
            .await
            .unwrap();

        let stats = db.get_stats(EPISODES_COLLECTION).await.unwrap();
        assert_eq!(stats.points_count, 2);
    }

    #[tokio::test]
    async fn test_mock_batch_store() {
        let db = MockVectorDB::new();

        let entries = vec![
            create_test_entry("b1", vec![1.0], MemoryType::Episode),
            create_test_entry("b2", vec![2.0], MemoryType::Episode),
            create_test_entry("b3", vec![3.0], MemoryType::Episode),
        ];

        let ids = db.store_batch(entries).await.unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(db.total_entries(), 3);
    }

    #[tokio::test]
    async fn test_cosine_similarity() {
        // Identical vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.0001);

        // Orthogonal vectors
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 0.0001);

        // Opposite vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_clear_all() {
        let db = MockVectorDB::new();

        db.store(create_test_entry("c1", vec![1.0], MemoryType::Episode))
            .await
            .unwrap();
        db.store(create_test_entry("c2", vec![1.0], MemoryType::Pattern))
            .await
            .unwrap();

        assert!(db.total_entries() > 0);

        db.clear_all();
        assert_eq!(db.total_entries(), 0);
    }

    #[tokio::test]
    async fn test_prune_old_memories() {
        let db = MockVectorDB::new();

        // Create an old entry
        let mut old_entry = create_test_entry("old", vec![1.0], MemoryType::Episode);
        old_entry.timestamp = chrono::Utc::now().timestamp() - 1000;
        db.store(old_entry).await.unwrap();

        // Create a new entry
        db.store(create_test_entry("new", vec![1.0], MemoryType::Episode))
            .await
            .unwrap();

        // Prune entries older than 500 seconds
        let pruned = db
            .prune_old_memories(EPISODES_COLLECTION, 500)
            .await
            .unwrap();

        assert_eq!(pruned, 1);
        assert!(db.get(EPISODES_COLLECTION, "old").await.unwrap().is_none());
        assert!(db.get(EPISODES_COLLECTION, "new").await.unwrap().is_some());
    }
}
