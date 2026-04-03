//! Vector Storage Trait
//!
//! Provides a trait abstraction for vector database operations, allowing
//! the use of either a real Qdrant-backed implementation or an in-memory
//! mock for testing.

use super::{CollectionStats, MemoryEntry, SearchQuery, SearchResult};
use async_trait::async_trait;
use common::Result;
use std::collections::HashMap;

/// Trait for vector storage operations
///
/// This trait abstracts the vector database interface, allowing components
/// to work with either a real Qdrant database or an in-memory mock.
///
/// # Example
///
/// ```ignore
/// use janus_neuromorphic::hippocampus::memory::{VectorStorage, MockVectorDB};
///
/// async fn example<S: VectorStorage>(storage: &S) {
///     let entry = MemoryEntry { /* ... */ };
///     storage.store(entry).await.unwrap();
/// }
/// ```
#[async_trait]
pub trait VectorStorage: Send + Sync {
    /// Initialize the storage (create collections, etc.)
    async fn initialize(&self) -> Result<()>;

    /// Store a memory entry
    ///
    /// # Arguments
    ///
    /// * `entry` - The memory entry to store
    ///
    /// # Returns
    ///
    /// The ID of the stored entry
    async fn store(&self, entry: MemoryEntry) -> Result<String>;

    /// Store multiple memory entries
    ///
    /// # Arguments
    ///
    /// * `entries` - Vector of memory entries to store
    ///
    /// # Returns
    ///
    /// Vector of IDs for the stored entries
    async fn store_batch(&self, entries: Vec<MemoryEntry>) -> Result<Vec<String>>;

    /// Search for similar vectors
    ///
    /// # Arguments
    ///
    /// * `query` - Search query containing the vector and parameters
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by similarity score
    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>>;

    /// Get a specific entry by ID
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection to search in
    /// * `id` - The ID of the entry to retrieve
    ///
    /// # Returns
    ///
    /// The memory entry if found, None otherwise
    async fn get(&self, collection: &str, id: &str) -> Result<Option<MemoryEntry>>;

    /// Delete an entry by ID
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection to delete from
    /// * `id` - The ID of the entry to delete
    async fn delete(&self, collection: &str, id: &str) -> Result<()>;

    /// Delete entries matching a filter
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection to delete from
    /// * `filters` - Key-value pairs for filtering entries
    async fn delete_by_filter(
        &self,
        collection: &str,
        filters: HashMap<String, serde_json::Value>,
    ) -> Result<()>;

    /// Get collection statistics
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection to get stats for
    ///
    /// # Returns
    ///
    /// Statistics about the collection
    async fn get_stats(&self, collection: &str) -> Result<CollectionStats>;

    /// Check if connected to the storage backend
    fn is_connected(&self) -> bool;

    /// Prune old memories by timestamp
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection to prune
    /// * `max_age_seconds` - Maximum age of memories to keep
    ///
    /// # Returns
    ///
    /// The number of entries pruned
    async fn prune_old_memories(&self, collection: &str, max_age_seconds: i64) -> Result<usize>;
}

/// Wrapper enum for dynamic dispatch between storage implementations
///
/// This enum allows runtime selection of storage backend without
/// requiring trait objects everywhere.
pub enum VectorStorageBackend {
    /// Real Qdrant-backed storage
    Real(super::VectorDB),
    /// In-memory mock storage for testing
    Mock(super::MockVectorDB),
}

#[async_trait]
impl VectorStorage for VectorStorageBackend {
    async fn initialize(&self) -> Result<()> {
        match self {
            VectorStorageBackend::Real(db) => db.initialize().await,
            VectorStorageBackend::Mock(db) => db.initialize().await,
        }
    }

    async fn store(&self, entry: MemoryEntry) -> Result<String> {
        match self {
            VectorStorageBackend::Real(db) => db.store(entry).await,
            VectorStorageBackend::Mock(db) => db.store(entry).await,
        }
    }

    async fn store_batch(&self, entries: Vec<MemoryEntry>) -> Result<Vec<String>> {
        match self {
            VectorStorageBackend::Real(db) => db.store_batch(entries).await,
            VectorStorageBackend::Mock(db) => db.store_batch(entries).await,
        }
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        match self {
            VectorStorageBackend::Real(db) => db.search(query).await,
            VectorStorageBackend::Mock(db) => db.search(query).await,
        }
    }

    async fn get(&self, collection: &str, id: &str) -> Result<Option<MemoryEntry>> {
        match self {
            VectorStorageBackend::Real(db) => db.get(id, collection).await,
            VectorStorageBackend::Mock(db) => db.get(collection, id).await,
        }
    }

    async fn delete(&self, collection: &str, id: &str) -> Result<()> {
        match self {
            VectorStorageBackend::Real(db) => db.delete(id, collection).await,
            VectorStorageBackend::Mock(db) => {
                db.delete(collection, id).await?;
                Ok(())
            }
        }
    }

    async fn delete_by_filter(
        &self,
        collection: &str,
        filters: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        match self {
            VectorStorageBackend::Real(db) => db.delete_by_filter(collection, filters).await,
            VectorStorageBackend::Mock(db) => {
                db.delete_by_filter(collection, filters).await?;
                Ok(())
            }
        }
    }

    async fn get_stats(&self, collection: &str) -> Result<CollectionStats> {
        match self {
            VectorStorageBackend::Real(db) => db.get_stats(collection).await,
            VectorStorageBackend::Mock(db) => db.get_stats(collection).await,
        }
    }

    fn is_connected(&self) -> bool {
        match self {
            VectorStorageBackend::Real(db) => db.is_connected(),
            VectorStorageBackend::Mock(db) => db.is_connected(),
        }
    }

    async fn prune_old_memories(&self, collection: &str, max_age_seconds: i64) -> Result<usize> {
        match self {
            VectorStorageBackend::Real(db) => {
                db.prune_old_memories(collection, max_age_seconds).await
            }
            VectorStorageBackend::Mock(db) => {
                db.prune_old_memories(collection, max_age_seconds).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hippocampus::memory::{EPISODES_COLLECTION, MemoryType, MockVectorDB};

    #[tokio::test]
    async fn test_storage_trait_with_mock() {
        let mock = MockVectorDB::new();
        let backend = VectorStorageBackend::Mock(mock);

        // Test through trait
        backend.initialize().await.unwrap();
        assert!(backend.is_connected());

        let entry = MemoryEntry {
            id: "trait-test".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            payload: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now().timestamp(),
            memory_type: MemoryType::Episode,
        };

        let id = backend.store(entry).await.unwrap();
        assert_eq!(id, "trait-test");

        let retrieved = backend
            .get(EPISODES_COLLECTION, "trait-test")
            .await
            .unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_storage_search_through_trait() {
        let mock = MockVectorDB::new();
        let backend = VectorStorageBackend::Mock(mock);

        // Store entries
        for i in 0..3 {
            let entry = MemoryEntry {
                id: format!("search-{}", i),
                vector: vec![i as f32, 0.0, 0.0],
                payload: std::collections::HashMap::new(),
                timestamp: chrono::Utc::now().timestamp(),
                memory_type: MemoryType::Episode,
            };
            backend.store(entry).await.unwrap();
        }

        // Search
        let query = SearchQuery {
            vector: vec![1.0, 0.0, 0.0],
            limit: 2,
            score_threshold: None,
            filters: None,
            collection: EPISODES_COLLECTION.to_string(),
        };

        let results = backend.search(query).await.unwrap();
        assert!(!results.is_empty());
    }
}
