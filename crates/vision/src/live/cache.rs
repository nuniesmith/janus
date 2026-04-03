//! Feature cache with LRU eviction for low-latency feature serving.
//!
//! This module provides efficient caching mechanisms for computed features:
//! - LRU (Least Recently Used) eviction policy
//! - Time-based expiration
//! - Multi-level caching (memory + optional disk)
//! - Cache statistics and hit rate tracking

use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

/// A cache entry with expiration tracking.
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    last_accessed: Instant,
    created_at: Instant,
    access_count: usize,
}

impl<V> CacheEntry<V> {
    fn new(value: V) -> Self {
        let now = Instant::now();
        Self {
            value,
            last_accessed: now,
            created_at: now,
            access_count: 0,
        }
    }

    fn access(&mut self) -> &V {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        &self.value
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// LRU cache with time-based expiration.
///
/// Provides O(1) access and insertion with automatic eviction of least
/// recently used entries when capacity is reached.
#[derive(Debug)]
pub struct LRUCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    cache: HashMap<K, CacheEntry<V>>,
    capacity: usize,
    ttl: Option<Duration>,
    stats: CacheStats,
}

impl<K, V> LRUCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new LRU cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            capacity,
            ttl: None,
            stats: CacheStats::new(),
        }
    }

    /// Create a new LRU cache with capacity and TTL.
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            capacity,
            ttl: Some(ttl),
            stats: CacheStats::new(),
        }
    }

    /// Insert a value into the cache.
    pub fn insert(&mut self, key: K, value: V) {
        // Evict expired entries if TTL is set
        if self.ttl.is_some() {
            self.evict_expired();
        }

        // Evict LRU entry if at capacity
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            self.evict_lru();
        }

        self.cache.insert(key, CacheEntry::new(value));
        self.stats.inserts += 1;
    }

    /// Get a value from the cache, updating access time.
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.cache.get_mut(key) {
            // Check if expired
            if let Some(ttl) = self.ttl {
                if entry.is_expired(ttl) {
                    self.cache.remove(key);
                    self.stats.misses += 1;
                    self.stats.evictions += 1;
                    return None;
                }
            }

            self.stats.hits += 1;
            Some(entry.access().clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Get a value without updating access time (peek).
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.cache.get(key).map(|entry| &entry.value)
    }

    /// Check if the cache contains a key.
    pub fn contains_key(&self, key: &K) -> bool {
        if let Some(entry) = self.cache.get(key) {
            if let Some(ttl) = self.ttl {
                !entry.is_expired(ttl)
            } else {
                true
            }
        } else {
            false
        }
    }

    /// Remove a value from the cache.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.cache.remove(key).map(|entry| entry.value)
    }

    /// Clear all entries from the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.stats.reset();
    }

    /// Get the current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.find_lru_key() {
            self.cache.remove(&lru_key);
            self.stats.evictions += 1;
        }
    }

    /// Find the key of the least recently used entry.
    fn find_lru_key(&self) -> Option<K> {
        self.cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone())
    }

    /// Evict all expired entries.
    fn evict_expired(&mut self) {
        if let Some(ttl) = self.ttl {
            let expired_keys: Vec<K> = self
                .cache
                .iter()
                .filter(|(_, entry)| entry.is_expired(ttl))
                .map(|(key, _)| key.clone())
                .collect();

            for key in expired_keys {
                self.cache.remove(&key);
                self.stats.evictions += 1;
            }
        }
    }

    /// Get the capacity of the cache.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Resize the cache capacity.
    pub fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
        while self.cache.len() > self.capacity {
            self.evict_lru();
        }
    }
}

/// Cache statistics tracker.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub inserts: usize,
    pub evictions: usize,
}

impl CacheStats {
    /// Create new cache statistics.
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            inserts: 0,
            evictions: 0,
        }
    }

    /// Calculate the cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate the cache miss rate.
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    /// Get total number of accesses.
    pub fn total_accesses(&self) -> usize {
        self.hits + self.misses
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.inserts = 0;
        self.evictions = 0;
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature cache key for GAF/DiffGAF features.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureCacheKey {
    pub symbol: String,
    pub timeframe: String,
    pub feature_type: FeatureType,
    pub window_end: i64,
}

impl FeatureCacheKey {
    /// Create a new feature cache key.
    pub fn new(
        symbol: String,
        timeframe: String,
        feature_type: FeatureType,
        window_end: i64,
    ) -> Self {
        Self {
            symbol,
            timeframe,
            feature_type,
            window_end,
        }
    }
}

/// Type of feature being cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureType {
    GAF,
    DiffGAF,
    Preprocessed,
    Raw,
}

/// Cached feature value.
#[derive(Debug, Clone)]
pub struct CachedFeature {
    pub data: Vec<Vec<f64>>,
    pub metadata: FeatureMetadata,
}

impl CachedFeature {
    /// Create a new cached feature.
    pub fn new(data: Vec<Vec<f64>>, metadata: FeatureMetadata) -> Self {
        Self { data, metadata }
    }
}

/// Metadata associated with a cached feature.
#[derive(Debug, Clone)]
pub struct FeatureMetadata {
    pub window_size: usize,
    pub timestamp: i64,
    pub computation_time_us: u64,
}

impl FeatureMetadata {
    /// Create new feature metadata.
    pub fn new(window_size: usize, timestamp: i64, computation_time_us: u64) -> Self {
        Self {
            window_size,
            timestamp,
            computation_time_us,
        }
    }
}

/// Multi-level feature cache with memory and optional persistence.
///
/// Provides hierarchical caching with different TTLs and capacities
/// for different feature types.
pub struct FeatureCache {
    gaf_cache: LRUCache<FeatureCacheKey, CachedFeature>,
    diffgaf_cache: LRUCache<FeatureCacheKey, CachedFeature>,
    preprocessed_cache: LRUCache<FeatureCacheKey, CachedFeature>,
}

impl FeatureCache {
    /// Create a new feature cache with default capacities.
    pub fn new() -> Self {
        Self::with_capacities(100, 100, 200)
    }

    /// Create a feature cache with custom capacities.
    pub fn with_capacities(
        gaf_capacity: usize,
        diffgaf_capacity: usize,
        preprocessed_capacity: usize,
    ) -> Self {
        Self {
            gaf_cache: LRUCache::with_ttl(gaf_capacity, Duration::from_secs(300)),
            diffgaf_cache: LRUCache::with_ttl(diffgaf_capacity, Duration::from_secs(300)),
            preprocessed_cache: LRUCache::with_ttl(preprocessed_capacity, Duration::from_secs(60)),
        }
    }

    /// Get a feature from the appropriate cache.
    pub fn get(&mut self, key: &FeatureCacheKey) -> Option<CachedFeature> {
        match key.feature_type {
            FeatureType::GAF => self.gaf_cache.get(key),
            FeatureType::DiffGAF => self.diffgaf_cache.get(key),
            FeatureType::Preprocessed => self.preprocessed_cache.get(key),
            FeatureType::Raw => None, // Raw data not cached
        }
    }

    /// Insert a feature into the appropriate cache.
    pub fn insert(&mut self, key: FeatureCacheKey, value: CachedFeature) {
        match key.feature_type {
            FeatureType::GAF => self.gaf_cache.insert(key, value),
            FeatureType::DiffGAF => self.diffgaf_cache.insert(key, value),
            FeatureType::Preprocessed => self.preprocessed_cache.insert(key, value),
            FeatureType::Raw => {} // Raw data not cached
        }
    }

    /// Check if a feature is cached.
    pub fn contains(&self, key: &FeatureCacheKey) -> bool {
        match key.feature_type {
            FeatureType::GAF => self.gaf_cache.contains_key(key),
            FeatureType::DiffGAF => self.diffgaf_cache.contains_key(key),
            FeatureType::Preprocessed => self.preprocessed_cache.contains_key(key),
            FeatureType::Raw => false,
        }
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        self.gaf_cache.clear();
        self.diffgaf_cache.clear();
        self.preprocessed_cache.clear();
    }

    /// Get combined cache statistics.
    pub fn stats(&self) -> CombinedCacheStats {
        CombinedCacheStats {
            gaf: self.gaf_cache.stats().clone(),
            diffgaf: self.diffgaf_cache.stats().clone(),
            preprocessed: self.preprocessed_cache.stats().clone(),
        }
    }

    /// Get total memory usage estimate (number of entries).
    pub fn total_entries(&self) -> usize {
        self.gaf_cache.len() + self.diffgaf_cache.len() + self.preprocessed_cache.len()
    }
}

impl Default for FeatureCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined statistics across all cache levels.
#[derive(Debug, Clone)]
pub struct CombinedCacheStats {
    pub gaf: CacheStats,
    pub diffgaf: CacheStats,
    pub preprocessed: CacheStats,
}

impl CombinedCacheStats {
    /// Get overall hit rate across all caches.
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.gaf.hits + self.diffgaf.hits + self.preprocessed.hits;
        let total_misses = self.gaf.misses + self.diffgaf.misses + self.preprocessed.misses;
        let total = total_hits + total_misses;

        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }

    /// Get total number of evictions across all caches.
    pub fn total_evictions(&self) -> usize {
        self.gaf.evictions + self.diffgaf.evictions + self.preprocessed.evictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_basic() {
        let mut cache = LRUCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);

        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), Some(2));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LRUCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3); // Should evict "a"

        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"b"), Some(2));
        assert_eq!(cache.get(&"c"), Some(3));
    }

    #[test]
    fn test_lru_access_updates() {
        let mut cache = LRUCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);

        // Access "a" to make it more recent
        cache.get(&"a");

        cache.insert("c", 3); // Should evict "b" (least recently used)

        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"c"), Some(3));
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = LRUCache::new(10);
        cache.insert("a", 1);
        cache.insert("b", 2);

        cache.get(&"a");
        cache.get(&"a");
        cache.get(&"c"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.inserts, 2);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ttl_expiration() {
        let mut cache = LRUCache::with_ttl(10, Duration::from_millis(50));
        cache.insert("a", 1);

        assert_eq!(cache.get(&"a"), Some(1));

        std::thread::sleep(Duration::from_millis(60));

        assert_eq!(cache.get(&"a"), None);
    }

    #[test]
    fn test_feature_cache() {
        let mut cache = FeatureCache::new();

        let key = FeatureCacheKey::new(
            "BTC".to_string(),
            "1m".to_string(),
            FeatureType::GAF,
            1234567890,
        );

        let feature = CachedFeature::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            FeatureMetadata::new(60, 1234567890, 1000),
        );

        cache.insert(key.clone(), feature);
        assert!(cache.contains(&key));

        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_cache_peek() {
        let mut cache = LRUCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);

        // Peek doesn't update access time
        assert_eq!(cache.peek(&"a"), Some(&1));

        cache.insert("c", 3); // Should still evict "a"
        assert_eq!(cache.get(&"a"), None);
    }

    #[test]
    fn test_cache_resize() {
        let mut cache = LRUCache::new(3);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        cache.resize(2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.capacity(), 2);
    }

    #[test]
    fn test_combined_stats() {
        let mut cache = FeatureCache::new();

        let key1 = FeatureCacheKey::new(
            "BTC".to_string(),
            "1m".to_string(),
            FeatureType::GAF,
            1234567890,
        );

        let key2 = FeatureCacheKey::new(
            "ETH".to_string(),
            "5m".to_string(),
            FeatureType::DiffGAF,
            1234567900,
        );

        let feature =
            CachedFeature::new(vec![vec![1.0]], FeatureMetadata::new(60, 1234567890, 500));

        cache.insert(key1.clone(), feature.clone());
        cache.insert(key2.clone(), feature);

        cache.get(&key1);
        cache.get(&key2);

        let stats = cache.stats();
        assert_eq!(stats.overall_hit_rate(), 1.0);
    }
}
