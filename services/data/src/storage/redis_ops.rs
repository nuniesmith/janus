//! Redis operations for state management, rate limiting, and deduplication
//!
//! This module provides Redis-backed utilities for:
//! - Rate limiting using token bucket algorithm
//! - Metric deduplication (hash-based)
//! - Health status storage
//! - Configuration management
//! - Active exchange tracking
//!
//! ## Key Patterns:
//! - Key namespacing: `{prefix}:{category}:{identifier}`
//! - TTL management: Automatic expiration for rate limits and temporary state
//! - Connection pooling: Multiplexed async connections

#![allow(dead_code)]

use anyhow::{Context, Result};
use redis::{AsyncCommands, Client, aio::ConnectionManager};
use tracing::{debug, info};

use crate::actors::MetricData;

/// Redis manager for state and cache operations
pub struct RedisManager {
    client: Client,
    connection: ConnectionManager,
    key_prefix: String,
}

impl RedisManager {
    /// Create a new Redis manager
    pub async fn new(url: &str, key_prefix: &str) -> Result<Self> {
        info!("RedisManager: Connecting to Redis at {}", url);

        let client = Client::open(url).context("Failed to create Redis client")?;

        let connection = ConnectionManager::new(client.clone())
            .await
            .context("Failed to establish Redis connection")?;

        info!("RedisManager: Connected to Redis successfully");

        Ok(Self {
            client,
            connection,
            key_prefix: key_prefix.to_string(),
        })
    }

    /// Build a namespaced key
    fn build_key(&self, category: &str, identifier: &str) -> String {
        format!("{}:{}:{}", self.key_prefix, category, identifier)
    }

    /// Check if a metric has been seen before (deduplication)
    pub async fn is_metric_duplicate(&self, metric: &MetricData) -> Result<bool> {
        let hash = self.compute_metric_hash(metric);
        let key = self.build_key(
            "metric_hash",
            &format!("{}:{}", metric.metric_type, metric.asset),
        );

        let mut conn = self.connection.clone();

        // Check if hash exists
        let exists: bool = conn.get(&key).await.unwrap_or(false);

        if exists {
            let stored_hash: String = conn.get(&key).await.unwrap_or_default();
            Ok(stored_hash == hash)
        } else {
            Ok(false)
        }
    }

    /// Mark a metric as seen
    pub async fn mark_metric_seen(&self, metric: &MetricData) -> Result<()> {
        let hash = self.compute_metric_hash(metric);
        let key = self.build_key(
            "metric_hash",
            &format!("{}:{}", metric.metric_type, metric.asset),
        );

        let mut conn = self.connection.clone();

        // Store hash with 24 hour TTL
        let ttl_secs = 86400; // 24 hours
        let _: () = conn.set_ex(&key, hash, ttl_secs).await?;

        debug!("RedisManager: Marked metric as seen: {}", key);

        Ok(())
    }

    /// Ping Redis to check connectivity
    pub async fn ping(&self) -> Result<()> {
        let mut conn = self.connection.clone();
        let _: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .context("Redis PING failed")?;
        Ok(())
    }

    /// Compute a hash for a metric (for deduplication)
    fn compute_metric_hash(&self, metric: &MetricData) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        metric.metric_type.hash(&mut hasher);
        metric.asset.hash(&mut hasher);
        metric.value.to_bits().hash(&mut hasher);
        metric.timestamp.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Store health status for a component
    pub async fn store_health_status(&self, component: &str, status: &str) -> Result<()> {
        let key = self.build_key("health", component);
        let mut conn = self.connection.clone();

        // Store with timestamp
        let timestamp = chrono::Utc::now().timestamp_millis();
        let value = format!("{}:{}", status, timestamp);

        let _: () = conn.set_ex(&key, value, 300).await?; // 5 minute TTL

        debug!(
            "RedisManager: Stored health status for {}: {}",
            component, status
        );

        Ok(())
    }

    /// Get health status for a component
    pub async fn get_health_status(&self, component: &str) -> Result<Option<(String, i64)>> {
        let key = self.build_key("health", component);
        let mut conn = self.connection.clone();

        let value: Option<String> = conn.get(&key).await?;

        if let Some(val) = value {
            let parts: Vec<&str> = val.split(':').collect();
            if parts.len() == 2 {
                let status = parts[0].to_string();
                let timestamp = parts[1].parse::<i64>().unwrap_or(0);
                return Ok(Some((status, timestamp)));
            }
        }

        Ok(None)
    }

    /// Check rate limit using token bucket algorithm
    ///
    /// Returns true if request is allowed, false if rate limit exceeded
    pub async fn check_rate_limit(
        &self,
        key: &str,
        max_requests: u32,
        window_secs: u64,
    ) -> Result<bool> {
        let now = chrono::Utc::now().timestamp();
        let window_start = now - (now % window_secs as i64);
        let rate_limit_key = self.build_key("ratelimit", &format!("{}:{}", key, window_start));

        let mut conn = self.connection.clone();

        // Increment counter
        let count: u32 = conn.incr(&rate_limit_key, 1).await?;

        // Set expiration on first increment
        if count == 1 {
            let ttl = (window_secs * 2) as i64; // 2x window for safety
            let _: () = conn.expire(&rate_limit_key, ttl).await?;
        }

        // Check if limit exceeded
        if count > max_requests {
            debug!(
                "RedisManager: Rate limit exceeded for {} ({}/{})",
                key, count, max_requests
            );
            Ok(false)
        } else {
            debug!(
                "RedisManager: Rate limit OK for {} ({}/{})",
                key, count, max_requests
            );
            Ok(true)
        }
    }

    /// Decrement rate limit counter (for token bucket with refill)
    pub async fn decrement_rate_limit(&self, key: &str) -> Result<i32> {
        let rate_limit_key = self.build_key("ratelimit", key);
        let mut conn = self.connection.clone();

        let count: i32 = conn.decr(&rate_limit_key, 1).await?;

        Ok(count)
    }

    /// Get the current active exchange from Redis
    pub async fn get_active_exchange(&self, asset: &str) -> Result<Option<String>> {
        let key = self.build_key("config", &format!("active_exchange:{}", asset));
        let mut conn = self.connection.clone();

        let exchange: Option<String> = conn.get(&key).await?;

        Ok(exchange)
    }

    /// Set the active exchange in Redis
    pub async fn set_active_exchange(&self, asset: &str, exchange: &str) -> Result<()> {
        let key = self.build_key("config", &format!("active_exchange:{}", asset));
        let mut conn = self.connection.clone();

        let _: () = conn.set(&key, exchange).await?;

        info!(
            "RedisManager: Set active exchange for {} to {}",
            asset, exchange
        );

        Ok(())
    }

    /// Store a heartbeat timestamp
    pub async fn heartbeat(&self, component: &str) -> Result<()> {
        let key = self.build_key("heartbeat", component);
        let mut conn = self.connection.clone();

        let timestamp = chrono::Utc::now().timestamp_millis();
        let _: () = conn.set_ex(&key, timestamp, 60).await?; // 1 minute TTL

        Ok(())
    }

    /// Check if a component is alive based on heartbeat
    pub async fn is_alive(&self, component: &str, max_age_secs: i64) -> Result<bool> {
        let key = self.build_key("heartbeat", component);
        let mut conn = self.connection.clone();

        let timestamp: Option<i64> = conn.get(&key).await?;

        if let Some(ts) = timestamp {
            let now = chrono::Utc::now().timestamp_millis();
            let age_secs = (now - ts) / 1000;

            Ok(age_secs <= max_age_secs)
        } else {
            Ok(false)
        }
    }

    /// Store gap detection info (for backfilling)
    pub async fn store_gap(&self, symbol: &str, start_ts: i64, end_ts: i64) -> Result<()> {
        let key = self.build_key("gaps", symbol);
        let mut conn = self.connection.clone();

        let gap_info = format!("{}:{}", start_ts, end_ts);

        // Add to a Redis list (LPUSH)
        let _: () = conn.lpush(&key, gap_info).await?;

        // Trim to keep only last 100 gaps
        let _: () = conn.ltrim(&key, 0, 99).await?;

        info!(
            "RedisManager: Stored gap for {} ({} - {})",
            symbol, start_ts, end_ts
        );

        Ok(())
    }

    /// Get pending gaps for backfilling
    pub async fn get_gaps(&self, symbol: &str) -> Result<Vec<(i64, i64)>> {
        let key = self.build_key("gaps", symbol);
        let mut conn = self.connection.clone();

        let gaps: Vec<String> = conn.lrange(&key, 0, -1).await?;

        let mut result = Vec::new();
        for gap in gaps {
            let parts: Vec<&str> = gap.split(':').collect();
            if parts.len() == 2
                && let (Ok(start), Ok(end)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>())
            {
                result.push((start, end));
            }
        }

        Ok(result)
    }

    /// Clear gaps for a symbol
    pub async fn clear_gaps(&self, symbol: &str) -> Result<()> {
        let key = self.build_key("gaps", symbol);
        let mut conn = self.connection.clone();

        let _: () = conn.del(&key).await?;

        info!("RedisManager: Cleared gaps for {}", symbol);

        Ok(())
    }

    /// Increment a counter (useful for metrics)
    pub async fn increment_counter(&self, counter_name: &str) -> Result<u64> {
        let key = self.build_key("counter", counter_name);
        let mut conn = self.connection.clone();

        let count: u64 = conn.incr(&key, 1).await?;

        Ok(count)
    }

    /// Get a counter value
    pub async fn get_counter(&self, counter_name: &str) -> Result<u64> {
        let key = self.build_key("counter", counter_name);
        let mut conn = self.connection.clone();

        let count: u64 = conn.get(&key).await.unwrap_or(0);

        Ok(count)
    }

    /// Reset a counter
    pub async fn reset_counter(&self, counter_name: &str) -> Result<()> {
        let key = self.build_key("counter", counter_name);
        let mut conn = self.connection.clone();

        let _: () = conn.set(&key, 0).await?;

        Ok(())
    }

    /// Get Redis info (for debugging)
    pub async fn info(&self) -> Result<String> {
        let mut conn = self.connection.clone();

        let info: String = redis::cmd("INFO")
            .arg("server")
            .query_async(&mut conn)
            .await
            .context("Failed to get Redis info")?;

        Ok(info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "Requires Redis connection on localhost"]
    async fn test_build_key() {
        let manager = RedisManager {
            client: Client::open("redis://localhost").unwrap(),
            connection: ConnectionManager::new(Client::open("redis://localhost").unwrap())
                .await
                .unwrap(),
            key_prefix: "test_factory".to_string(),
        };

        let key = manager.build_key("health", "binance_ws");
        assert_eq!(key, "test_factory:health:binance_ws");
    }

    #[tokio::test]
    #[ignore = "Requires Redis connection on localhost"]
    async fn test_compute_metric_hash() {
        let manager = RedisManager {
            client: Client::open("redis://localhost").unwrap(),
            connection: ConnectionManager::new(Client::open("redis://localhost").unwrap())
                .await
                .unwrap(),
            key_prefix: "test".to_string(),
        };

        let metric1 = MetricData {
            metric_type: "fear_greed".to_string(),
            asset: "BTC".to_string(),
            source: "alternative_me".to_string(),
            value: 45.0,
            meta: None,
            timestamp: 1672531200000,
        };

        let metric2 = MetricData {
            metric_type: "fear_greed".to_string(),
            asset: "BTC".to_string(),
            source: "alternative_me".to_string(),
            value: 45.0,
            meta: None,
            timestamp: 1672531200000,
        };

        let metric3 = MetricData {
            metric_type: "fear_greed".to_string(),
            asset: "BTC".to_string(),
            source: "alternative_me".to_string(),
            value: 50.0, // Different value
            meta: None,
            timestamp: 1672531200000,
        };

        let hash1 = manager.compute_metric_hash(&metric1);
        let hash2 = manager.compute_metric_hash(&metric2);
        let hash3 = manager.compute_metric_hash(&metric3);

        // Same metrics should produce same hash
        assert_eq!(hash1, hash2);

        // Different metrics should produce different hash
        assert_ne!(hash1, hash3);
    }
}
