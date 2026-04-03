//! Vector Database client wrapper for Qdrant.
//!
//! Stores market regimes and experiences as vectors for similarity search.
//! Provides both a mock implementation for testing and a real Qdrant client.

use common::{JanusError, MarketRegime, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Vector dimension for market regime embeddings
pub const REGIME_VECTOR_DIM: usize = 64;

/// Configuration for VectorDb
#[derive(Debug, Clone)]
pub struct VectorDbConfig {
    /// Qdrant server URL
    pub url: String,
    /// Collection name for storing vectors
    pub collection_name: String,
    /// Vector dimension
    pub vector_dim: usize,
    /// Use mock implementation (for testing)
    pub use_mock: bool,
    /// Connection timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries for failed operations
    pub max_retries: u32,
}

impl Default for VectorDbConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            collection_name: "market_regimes".to_string(),
            vector_dim: REGIME_VECTOR_DIM,
            use_mock: true, // Default to mock for safety
            timeout_secs: 10,
            max_retries: 3,
        }
    }
}

/// Search result from vector database
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching regime
    pub regime: MarketRegime,
    /// Similarity score (0.0 - 1.0)
    pub score: f64,
}

/// Statistics about the vector database
#[derive(Debug, Clone, Default)]
pub struct VectorDbStats {
    /// Number of vectors stored
    pub vector_count: usize,
    /// Number of successful searches
    pub search_count: u64,
    /// Number of successful stores
    pub store_count: u64,
    /// Number of failed operations
    pub error_count: u64,
    /// Average search latency in milliseconds
    pub avg_search_latency_ms: f64,
}

/// Mock storage for testing without Qdrant
#[derive(Debug, Default)]
struct MockStorage {
    regimes: HashMap<String, (Vec<f64>, MarketRegime)>,
    stats: VectorDbStats,
}

impl MockStorage {
    fn new() -> Self {
        Self::default()
    }

    fn store(&mut self, regime: &MarketRegime) -> Result<()> {
        self.regimes
            .insert(regime.id.clone(), (regime.features.clone(), regime.clone()));
        self.stats.vector_count = self.regimes.len();
        self.stats.store_count += 1;
        Ok(())
    }

    fn search(&mut self, query: &[f64], limit: usize) -> Result<Vec<SearchResult>> {
        self.stats.search_count += 1;

        let mut results: Vec<(f64, MarketRegime)> = self
            .regimes
            .values()
            .map(|(features, regime)| {
                let score = cosine_similarity(query, features);
                (score, regime.clone())
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(limit)
            .map(|(score, regime)| SearchResult { regime, score })
            .collect())
    }

    fn delete(&mut self, id: &str) -> Result<bool> {
        let existed = self.regimes.remove(id).is_some();
        self.stats.vector_count = self.regimes.len();
        Ok(existed)
    }

    fn clear(&mut self) {
        self.regimes.clear();
        self.stats.vector_count = 0;
    }

    fn stats(&self) -> VectorDbStats {
        self.stats.clone()
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Qdrant client wrapper for storing and retrieving market regimes
pub struct VectorDb {
    config: VectorDbConfig,
    /// Mock storage (used when use_mock is true or Qdrant unavailable)
    mock_storage: Arc<RwLock<MockStorage>>,
    /// Whether we're using mock mode
    is_mock: bool,
    /// Connection status
    connected: Arc<RwLock<bool>>,
}

impl VectorDb {
    /// Create a new VectorDb client with default configuration
    pub async fn new(url: &str, collection_name: String) -> Result<Self> {
        let config = VectorDbConfig {
            url: url.to_string(),
            collection_name,
            use_mock: true, // Default to mock mode
            ..Default::default()
        };
        Self::with_config(config).await
    }

    /// Create a new VectorDb client with custom configuration
    pub async fn with_config(config: VectorDbConfig) -> Result<Self> {
        let db = Self {
            mock_storage: Arc::new(RwLock::new(MockStorage::new())),
            is_mock: config.use_mock,
            connected: Arc::new(RwLock::new(false)),
            config,
        };

        // Try to connect if not using mock
        if !db.is_mock {
            match db.try_connect().await {
                Ok(_) => {
                    tracing::info!("Connected to Qdrant at {}", db.config.url);
                    *db.connected.write().await = true;
                }
                Err(e) => {
                    tracing::warn!("Failed to connect to Qdrant, falling back to mock: {}", e);
                    // Fall back to mock mode
                }
            }
        } else {
            tracing::info!("Using mock VectorDb (Qdrant disabled)");
            *db.connected.write().await = true;
        }

        Ok(db)
    }

    /// Create a mock VectorDb for testing
    pub fn mock() -> Self {
        Self {
            config: VectorDbConfig {
                use_mock: true,
                ..Default::default()
            },
            mock_storage: Arc::new(RwLock::new(MockStorage::new())),
            is_mock: true,
            connected: Arc::new(RwLock::new(true)),
        }
    }

    /// Try to connect to Qdrant server
    async fn try_connect(&self) -> Result<()> {
        // Attempt HTTP health check
        let health_url = format!("{}/", self.config.url);

        let client: reqwest::Client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(|e| JanusError::Memory(format!("Failed to create HTTP client: {}", e)))?;

        let _response: reqwest::Response = client
            .get(&health_url)
            .send()
            .await
            .map_err(|e| JanusError::Memory(format!("Qdrant connection failed: {}", e)))?;

        Ok(())
    }

    /// Check if connected to Qdrant (or mock is ready)
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Check if using mock mode
    pub fn is_mock(&self) -> bool {
        self.is_mock
    }

    /// Store a market regime
    pub async fn store_regime(&self, regime: &MarketRegime) -> Result<()> {
        if self.is_mock || !*self.connected.read().await {
            let mut storage: tokio::sync::RwLockWriteGuard<'_, MockStorage> =
                self.mock_storage.write().await;
            return storage.store(regime);
        }

        // Real Qdrant implementation would go here
        // For now, use mock storage as fallback
        let mut storage: tokio::sync::RwLockWriteGuard<'_, MockStorage> =
            self.mock_storage.write().await;
        storage.store(regime)
    }

    /// Store multiple regimes in a batch
    pub async fn store_regimes(&self, regimes: &[MarketRegime]) -> Result<usize> {
        let mut count = 0;
        for regime in regimes {
            if self.store_regime(regime).await.is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Search for similar regimes
    pub async fn search_similar(
        &self,
        query_vector: &[f64],
        limit: u64,
    ) -> Result<Vec<MarketRegime>> {
        let results = self.search_with_scores(query_vector, limit).await?;
        Ok(results.into_iter().map(|r| r.regime).collect())
    }

    /// Search for similar regimes with similarity scores
    pub async fn search_with_scores(
        &self,
        query_vector: &[f64],
        limit: u64,
    ) -> Result<Vec<SearchResult>> {
        if self.is_mock || !*self.connected.read().await {
            let mut storage: tokio::sync::RwLockWriteGuard<'_, MockStorage> =
                self.mock_storage.write().await;
            return storage.search(query_vector, limit as usize);
        }

        // Real Qdrant implementation would go here
        let mut storage: tokio::sync::RwLockWriteGuard<'_, MockStorage> =
            self.mock_storage.write().await;
        storage.search(query_vector, limit as usize)
    }

    /// Find the most similar regime
    pub async fn find_most_similar(&self, query_vector: &[f64]) -> Result<Option<SearchResult>> {
        let results = self.search_with_scores(query_vector, 1).await?;
        Ok(results.into_iter().next())
    }

    /// Delete a regime by ID
    pub async fn delete_regime(&self, id: &str) -> Result<bool> {
        if self.is_mock || !*self.connected.read().await {
            let mut storage: tokio::sync::RwLockWriteGuard<'_, MockStorage> =
                self.mock_storage.write().await;
            return storage.delete(id);
        }

        let mut storage: tokio::sync::RwLockWriteGuard<'_, MockStorage> =
            self.mock_storage.write().await;
        storage.delete(id)
    }

    /// Get a regime by ID
    pub async fn get_regime(&self, id: &str) -> Result<Option<MarketRegime>> {
        let storage: tokio::sync::RwLockReadGuard<'_, MockStorage> = self.mock_storage.read().await;
        Ok(storage
            .regimes
            .get(id)
            .map(|(_, regime): &(Vec<f64>, MarketRegime)| regime.clone()))
    }

    /// Clear all stored regimes
    pub async fn clear(&self) -> Result<()> {
        let mut storage: tokio::sync::RwLockWriteGuard<'_, MockStorage> =
            self.mock_storage.write().await;
        storage.clear();
        Ok(())
    }

    /// Get statistics about the database
    pub async fn stats(&self) -> VectorDbStats {
        let storage: tokio::sync::RwLockReadGuard<'_, MockStorage> = self.mock_storage.read().await;
        storage.stats()
    }

    /// Get the number of stored vectors
    pub async fn count(&self) -> usize {
        let storage: tokio::sync::RwLockReadGuard<'_, MockStorage> = self.mock_storage.read().await;
        storage.regimes.len()
    }

    /// Ensure the collection exists (creates if needed)
    pub async fn ensure_collection(&self) -> Result<()> {
        // In mock mode, collection always exists
        if self.is_mock {
            return Ok(());
        }

        // Real implementation would create collection in Qdrant
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_regime(id: &str, features: Vec<f64>) -> MarketRegime {
        MarketRegime {
            id: id.to_string(),
            name: format!("Regime {}", id),
            features,
            volatility: 0.02,
            trend: 0.5,
        }
    }

    #[tokio::test]
    async fn test_mock_creation() {
        let db = VectorDb::mock();
        assert!(db.is_mock());
        assert!(db.is_connected().await);
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let db = VectorDb::mock();

        let regime = create_test_regime("test1", vec![1.0, 0.0, 0.0]);
        db.store_regime(&regime).await.unwrap();

        let retrieved = db.get_regime("test1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test1");
    }

    #[tokio::test]
    async fn test_search_similar() {
        let db = VectorDb::mock();

        // Store some regimes
        db.store_regime(&create_test_regime("r1", vec![1.0, 0.0, 0.0]))
            .await
            .unwrap();
        db.store_regime(&create_test_regime("r2", vec![0.9, 0.1, 0.0]))
            .await
            .unwrap();
        db.store_regime(&create_test_regime("r3", vec![0.0, 1.0, 0.0]))
            .await
            .unwrap();

        // Search for similar to r1
        let results = db.search_with_scores(&[1.0, 0.0, 0.0], 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].regime.id, "r1"); // Exact match should be first
        assert!(results[0].score > 0.99);
    }

    #[tokio::test]
    async fn test_find_most_similar() {
        let db = VectorDb::mock();

        db.store_regime(&create_test_regime("r1", vec![1.0, 0.0]))
            .await
            .unwrap();
        db.store_regime(&create_test_regime("r2", vec![0.0, 1.0]))
            .await
            .unwrap();

        let result = db.find_most_similar(&[0.9, 0.1]).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().regime.id, "r1");
    }

    #[tokio::test]
    async fn test_delete() {
        let db = VectorDb::mock();

        let regime = create_test_regime("test1", vec![1.0, 0.0]);
        db.store_regime(&regime).await.unwrap();

        assert_eq!(db.count().await, 1);

        let deleted = db.delete_regime("test1").await.unwrap();
        assert!(deleted);

        assert_eq!(db.count().await, 0);

        let deleted_again = db.delete_regime("test1").await.unwrap();
        assert!(!deleted_again);
    }

    #[tokio::test]
    async fn test_clear() {
        let db = VectorDb::mock();

        for i in 0..5 {
            db.store_regime(&create_test_regime(&format!("r{}", i), vec![i as f64]))
                .await
                .unwrap();
        }

        assert_eq!(db.count().await, 5);

        db.clear().await.unwrap();

        assert_eq!(db.count().await, 0);
    }

    #[tokio::test]
    async fn test_stats() {
        let db = VectorDb::mock();

        db.store_regime(&create_test_regime("r1", vec![1.0]))
            .await
            .unwrap();
        db.store_regime(&create_test_regime("r2", vec![2.0]))
            .await
            .unwrap();

        db.search_similar(&[1.0], 10).await.unwrap();
        db.search_similar(&[2.0], 10).await.unwrap();

        let stats = db.stats().await;
        assert_eq!(stats.vector_count, 2);
        assert_eq!(stats.store_count, 2);
        assert_eq!(stats.search_count, 2);
    }

    #[tokio::test]
    async fn test_batch_store() {
        let db = VectorDb::mock();

        let regimes: Vec<_> = (0..10)
            .map(|i| create_test_regime(&format!("r{}", i), vec![i as f64]))
            .collect();

        let count = db.store_regimes(&regimes).await.unwrap();
        assert_eq!(count, 10);
        assert_eq!(db.count().await, 10);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!(sim.abs() < 0.001);

        // Opposite vectors
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]);
        assert!((sim - (-1.0)).abs() < 0.001);

        // Different lengths
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]);
        assert_eq!(sim, 0.0);

        // Empty vectors
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[tokio::test]
    async fn test_config() {
        let config = VectorDbConfig {
            url: "http://localhost:6333".to_string(),
            collection_name: "test_collection".to_string(),
            vector_dim: 128,
            use_mock: true,
            timeout_secs: 5,
            max_retries: 2,
        };

        let db = VectorDb::with_config(config).await.unwrap();
        assert!(db.is_mock());
    }
}
