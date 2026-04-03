//! Aggregator Service Module
//!
//! This module provides the aggregator service that combines all external data
//! sources (news, weather, celestial) into a unified feed for the neuromorphic
//! system.
//!
//! ## Features
//!
//! - Unified data feed from multiple sources
//! - Configurable polling intervals per source
//! - Data fusion and normalization
//! - Event streaming via channels
//! - Health monitoring and metrics
//! - Graceful degradation when sources fail
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    AGGREGATOR SERVICE                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │    News      │  │   Weather    │  │  Celestial   │          │
//! │  │   Sources    │  │   Sources    │  │   Sources    │          │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
//! │         │                  │                  │                  │
//! │         └──────────────────┼──────────────────┘                  │
//! │                            │                                     │
//! │                   ┌────────▼────────┐                           │
//! │                   │   Data Fusion   │                           │
//! │                   │     Engine      │                           │
//! │                   └────────┬────────┘                           │
//! │                            │                                     │
//! │                   ┌────────▼────────┐                           │
//! │                   │  Unified Feed   │                           │
//! │                   │   (broadcast)   │                           │
//! │                   └─────────────────┘                           │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use super::bridge::DataSourceBridge;
use super::cache::{CacheKey, RedisCache};
use super::config::{AggregatorConfig, ExternalDataConfig};
use super::{ExternalDataPoint, MoonPhase};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{RwLock, broadcast};
use tokio::time::{Instant, interval};
use tracing::{debug, error, info, warn};

/// Aggregator service error types
#[derive(Debug, thiserror::Error)]
pub enum AggregatorError {
    #[error("Service not running")]
    NotRunning,

    #[error("Service already running")]
    AlreadyRunning,

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Data source error: {0}")]
    SourceError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Channel error: {0}")]
    ChannelError(String),
}

pub type Result<T> = std::result::Result<T, AggregatorError>;

/// Unified data feed item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedDataFeed {
    /// Feed ID
    pub id: String,

    /// Timestamp when the feed was created
    pub timestamp: DateTime<Utc>,

    /// Aggregated external data
    pub data: ExternalDataPoint,

    /// Computed composite scores
    pub scores: CompositeScores,

    /// Source health status
    pub source_health: HashMap<String, bool>,

    /// Processing latency in milliseconds
    pub latency_ms: u64,

    /// Sequence number
    pub sequence: u64,
}

/// Composite scores computed from aggregated data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompositeScores {
    /// Overall market sentiment (-1.0 to 1.0)
    pub market_sentiment: f64,

    /// Sentiment confidence (0.0 to 1.0)
    pub sentiment_confidence: f64,

    /// Environmental factor score (-1.0 to 1.0)
    pub environmental_factor: f64,

    /// Celestial influence score (-1.0 to 1.0)
    pub celestial_influence: f64,

    /// Combined risk indicator (0.0 to 1.0)
    pub risk_indicator: f64,

    /// Volatility prediction (0.0 to 1.0)
    pub volatility_prediction: f64,

    /// Trading opportunity score (0.0 to 1.0)
    pub opportunity_score: f64,
}

impl CompositeScores {
    /// Create new composite scores from data point
    pub fn from_data(data: &ExternalDataPoint) -> Self {
        let mut scores = Self::default();

        // Calculate market sentiment from news
        if let Some(ref news) = data.news {
            scores.market_sentiment = news.score;
            scores.sentiment_confidence = news.confidence;

            // Higher sentiment volatility indicates more market uncertainty
            scores.volatility_prediction = (1.0 - news.confidence) * 0.5;
        }

        // Calculate environmental factor from weather
        if let Some(ref weather) = data.weather {
            // Extreme weather can affect market behavior
            let temp_factor = if weather.temperature_c > 35.0 || weather.temperature_c < -10.0 {
                -0.2 // Extreme temps slightly negative
            } else {
                0.0
            };

            let humidity_factor = if weather.humidity > 90.0 {
                -0.1 // High humidity slightly negative
            } else {
                0.0
            };

            scores.environmental_factor = temp_factor + humidity_factor;
        }

        // Calculate celestial influence
        if let Some(ref celestial) = data.celestial {
            // Moon phase influence (full moon = higher volatility expectation)
            let moon_factor = match celestial.moon_phase {
                MoonPhase::FullMoon => 0.3,
                MoonPhase::NewMoon => 0.1,
                MoonPhase::FirstQuarter | MoonPhase::LastQuarter => 0.2,
                _ => 0.0,
            };

            // Space weather influence
            let space_factor = {
                let sw = &celestial.space_weather;
                // High Kp index indicates geomagnetic storm
                if sw.kp_index > 5.0 {
                    0.2 // Increased volatility
                } else {
                    0.0
                }
            };

            scores.celestial_influence = moon_factor + space_factor;
            scores.volatility_prediction += moon_factor * 0.2;
        }

        // Calculate combined risk indicator
        scores.risk_indicator = Self::calculate_risk(
            scores.market_sentiment,
            scores.sentiment_confidence,
            scores.volatility_prediction,
        );

        // Calculate opportunity score
        scores.opportunity_score = Self::calculate_opportunity(
            scores.market_sentiment,
            scores.sentiment_confidence,
            scores.risk_indicator,
        );

        scores
    }

    /// Calculate risk indicator
    fn calculate_risk(sentiment: f64, confidence: f64, volatility: f64) -> f64 {
        let base_risk = 0.5;

        // Lower confidence increases risk
        let confidence_risk = (1.0 - confidence) * 0.3;

        // Higher volatility increases risk
        let volatility_risk = volatility * 0.4;

        // Extreme sentiment (positive or negative) increases risk
        let sentiment_risk = sentiment.abs() * 0.2;

        (base_risk + confidence_risk + volatility_risk + sentiment_risk).min(1.0)
    }

    /// Calculate opportunity score
    fn calculate_opportunity(sentiment: f64, confidence: f64, risk: f64) -> f64 {
        // Strong positive sentiment with high confidence = good opportunity
        let sentiment_opportunity = if sentiment > 0.0 {
            sentiment * confidence
        } else {
            0.0
        };

        // Lower risk = better opportunity
        let risk_adjusted = (1.0 - risk) * 0.5;

        (sentiment_opportunity + risk_adjusted).min(1.0)
    }
}

/// Aggregator service state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceState {
    Stopped,
    Starting,
    Running,
    Stopping,
    Error,
}

/// Aggregator service metrics
#[derive(Debug, Default)]
pub struct AggregatorMetrics {
    /// Total aggregations performed
    pub aggregations: AtomicU64,

    /// Successful aggregations
    pub successful: AtomicU64,

    /// Failed aggregations
    pub failed: AtomicU64,

    /// Total feeds published
    pub feeds_published: AtomicU64,

    /// Average processing latency (cumulative)
    pub total_latency_ms: AtomicU64,

    /// Current sequence number
    pub sequence: AtomicU64,
}

impl AggregatorMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_aggregation(&self, success: bool, latency_ms: u64) {
        self.aggregations.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successful.fetch_add(1, Ordering::Relaxed);
            self.total_latency_ms
                .fetch_add(latency_ms, Ordering::Relaxed);
        } else {
            self.failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_feed_published(&self) {
        self.feeds_published.fetch_add(1, Ordering::Relaxed);
    }

    pub fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    pub fn average_latency_ms(&self) -> f64 {
        let total = self.total_latency_ms.load(Ordering::Relaxed);
        let count = self.successful.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.aggregations.load(Ordering::Relaxed);
        let success = self.successful.load(Ordering::Relaxed);
        if total == 0 {
            1.0
        } else {
            success as f64 / total as f64
        }
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> AggregatorMetricsSnapshot {
        AggregatorMetricsSnapshot {
            aggregations: self.aggregations.load(Ordering::Relaxed),
            successful: self.successful.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            feeds_published: self.feeds_published.load(Ordering::Relaxed),
            average_latency_ms: self.average_latency_ms(),
            success_rate: self.success_rate(),
            sequence: self.sequence.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of aggregator metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorMetricsSnapshot {
    pub aggregations: u64,
    pub successful: u64,
    pub failed: u64,
    pub feeds_published: u64,
    pub average_latency_ms: f64,
    pub success_rate: f64,
    pub sequence: u64,
}

/// Aggregator service configuration
#[derive(Debug, Clone)]
pub struct AggregatorServiceConfig {
    /// Base aggregator config
    pub aggregator: AggregatorConfig,

    /// Feed buffer size
    pub feed_buffer_size: usize,

    /// Enable caching
    pub enable_caching: bool,

    /// Cache TTL for feeds
    pub feed_cache_ttl: Duration,

    /// Minimum interval between identical feeds
    pub dedup_interval: Duration,

    /// Enable composite score calculation
    pub enable_scoring: bool,
}

impl Default for AggregatorServiceConfig {
    fn default() -> Self {
        Self {
            aggregator: AggregatorConfig::default(),
            feed_buffer_size: 1000,
            enable_caching: true,
            feed_cache_ttl: Duration::from_secs(60),
            dedup_interval: Duration::from_secs(5),
            enable_scoring: true,
        }
    }
}

/// Aggregator service that combines all data sources
pub struct AggregatorService {
    /// Service configuration
    config: AggregatorServiceConfig,

    /// Data source bridge
    bridge: Option<Arc<DataSourceBridge>>,

    /// Redis cache
    cache: Option<Arc<RedisCache>>,

    /// Service state
    state: RwLock<ServiceState>,

    /// Service metrics
    metrics: Arc<AggregatorMetrics>,

    /// Feed broadcaster
    feed_tx: broadcast::Sender<UnifiedDataFeed>,

    /// Running flag
    running: AtomicBool,

    /// Last feed timestamp
    last_feed: RwLock<Option<DateTime<Utc>>>,

    /// Source health status
    source_health: RwLock<HashMap<String, bool>>,
}

impl AggregatorService {
    /// Create a new aggregator service
    pub fn new(config: AggregatorServiceConfig) -> Self {
        let (feed_tx, _) = broadcast::channel(config.feed_buffer_size);

        Self {
            config,
            bridge: None,
            cache: None,
            state: RwLock::new(ServiceState::Stopped),
            metrics: Arc::new(AggregatorMetrics::new()),
            feed_tx,
            running: AtomicBool::new(false),
            last_feed: RwLock::new(None),
            source_health: RwLock::new(HashMap::new()),
        }
    }

    /// Set the data source bridge
    pub fn with_bridge(mut self, bridge: Arc<DataSourceBridge>) -> Self {
        self.bridge = Some(bridge);
        self
    }

    /// Set the cache
    pub fn with_cache(mut self, cache: Arc<RedisCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Subscribe to the unified feed
    pub fn subscribe(&self) -> broadcast::Receiver<UnifiedDataFeed> {
        self.feed_tx.subscribe()
    }

    /// Get service metrics
    pub fn metrics(&self) -> Arc<AggregatorMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Get current service state
    pub async fn state(&self) -> ServiceState {
        *self.state.read().await
    }

    /// Check if service is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the latest feed from cache
    pub async fn get_latest_feed(&self) -> Option<UnifiedDataFeed> {
        if let Some(ref cache) = self.cache {
            match cache
                .get::<UnifiedDataFeed>(CacheKey::custom("feed:latest"))
                .await
            {
                Ok(feed) => feed,
                Err(e) => {
                    warn!("Failed to get latest feed from cache: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }

    /// Get source health status
    pub async fn source_health(&self) -> HashMap<String, bool> {
        self.source_health.read().await.clone()
    }

    /// Aggregate data from all sources
    async fn aggregate(&self) -> Result<ExternalDataPoint> {
        let Some(ref bridge) = self.bridge else {
            return Err(AggregatorError::ConfigError(
                "No data source bridge configured".to_string(),
            ));
        };

        // Fetch all data from bridge
        let data = bridge.fetch_all().await;

        // Update source health
        let health = bridge.health_check().await;
        *self.source_health.write().await = health;

        Ok(data)
    }

    /// Create unified feed from data point
    fn create_feed(&self, data: ExternalDataPoint, latency_ms: u64) -> UnifiedDataFeed {
        let scores = if self.config.enable_scoring {
            CompositeScores::from_data(&data)
        } else {
            CompositeScores::default()
        };

        UnifiedDataFeed {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            data,
            scores,
            source_health: HashMap::new(), // Will be populated from source_health
            latency_ms,
            sequence: self.metrics.next_sequence(),
        }
    }

    /// Publish feed to subscribers and cache
    async fn publish_feed(&self, mut feed: UnifiedDataFeed) -> Result<()> {
        // Add source health to feed
        feed.source_health = self.source_health.read().await.clone();

        // Cache the feed
        if self.config.enable_caching {
            if let Some(ref cache) = self.cache {
                let cache_key = CacheKey::custom("feed:latest");
                if let Err(e) = cache
                    .set(cache_key, &feed, Some(self.config.feed_cache_ttl))
                    .await
                {
                    warn!("Failed to cache feed: {}", e);
                }

                // Also cache by sequence for history
                let history_key = CacheKey::custom(&format!("feed:{}", feed.sequence));
                if let Err(e) = cache
                    .set(history_key, &feed, Some(Duration::from_secs(3600)))
                    .await
                {
                    warn!("Failed to cache feed history: {}", e);
                }
            }
        }

        // Broadcast to subscribers
        match self.feed_tx.send(feed) {
            Ok(count) => {
                debug!("Published feed to {} subscribers", count);
                self.metrics.record_feed_published();
            }
            Err(_) => {
                debug!("No subscribers for feed");
            }
        }

        // Update last feed timestamp
        *self.last_feed.write().await = Some(Utc::now());

        Ok(())
    }

    /// Check if we should skip this aggregation (deduplication)
    async fn should_skip(&self) -> bool {
        if let Some(last) = *self.last_feed.read().await {
            let elapsed = Utc::now().signed_duration_since(last);
            if let Ok(duration) = elapsed.to_std() {
                return duration < self.config.dedup_interval;
            }
        }
        false
    }

    /// Run one aggregation cycle
    pub async fn run_cycle(&self) -> Result<()> {
        // Check deduplication
        if self.should_skip().await {
            debug!("Skipping aggregation cycle (dedup)");
            return Ok(());
        }

        let start = Instant::now();

        // Aggregate data
        let data = self.aggregate().await?;

        let latency = start.elapsed().as_millis() as u64;
        self.metrics.record_aggregation(true, latency);

        // Create and publish feed
        let feed = self.create_feed(data, latency);
        self.publish_feed(feed).await?;

        Ok(())
    }

    /// Start the aggregator service
    pub async fn start(&self) -> Result<()> {
        // Check state
        {
            let state = self.state.read().await;
            if *state == ServiceState::Running {
                return Err(AggregatorError::AlreadyRunning);
            }
        }

        // Validate configuration
        if self.bridge.is_none() {
            return Err(AggregatorError::ConfigError(
                "No data source bridge configured".to_string(),
            ));
        }

        // Update state
        *self.state.write().await = ServiceState::Starting;
        info!("Starting aggregator service");

        self.running.store(true, Ordering::SeqCst);
        *self.state.write().await = ServiceState::Running;

        let poll_interval = self.config.aggregator.poll_duration();
        let mut ticker = interval(poll_interval);

        while self.running.load(Ordering::SeqCst) {
            ticker.tick().await;

            match self.run_cycle().await {
                Ok(()) => {
                    debug!("Aggregation cycle completed");
                }
                Err(e) => {
                    error!("Aggregation cycle failed: {}", e);
                    self.metrics.record_aggregation(false, 0);
                }
            }
        }

        *self.state.write().await = ServiceState::Stopped;
        info!("Aggregator service stopped");

        Ok(())
    }

    /// Stop the aggregator service
    pub async fn stop(&self) -> Result<()> {
        if !self.is_running() {
            return Err(AggregatorError::NotRunning);
        }

        info!("Stopping aggregator service");
        *self.state.write().await = ServiceState::Stopping;
        self.running.store(false, Ordering::SeqCst);

        Ok(())
    }

    /// Perform health check
    pub async fn health_check(&self) -> bool {
        // Check service state
        let state = *self.state.read().await;
        if state != ServiceState::Running {
            return false;
        }

        // Check bridge health
        if let Some(ref bridge) = self.bridge {
            let health = bridge.health_check().await;
            let healthy_count = health.values().filter(|&&v| v).count();
            let total = health.len();

            // Consider healthy if at least 50% of sources are healthy
            if total > 0 && healthy_count < total / 2 {
                return false;
            }
        }

        // Check metrics
        let success_rate = self.metrics.success_rate();
        if success_rate < 0.5 {
            return false;
        }

        true
    }
}

/// Builder for AggregatorService
pub struct AggregatorServiceBuilder {
    config: AggregatorServiceConfig,
    bridge: Option<Arc<DataSourceBridge>>,
    cache: Option<Arc<RedisCache>>,
}

impl AggregatorServiceBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: AggregatorServiceConfig::default(),
            bridge: None,
            cache: None,
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: AggregatorServiceConfig) -> Self {
        self.config = config;
        self
    }

    /// Set poll interval
    pub fn poll_interval(mut self, seconds: u64) -> Self {
        self.config.aggregator.poll_interval = seconds;
        self
    }

    /// Set feed buffer size
    pub fn feed_buffer_size(mut self, size: usize) -> Self {
        self.config.feed_buffer_size = size;
        self
    }

    /// Enable caching
    pub fn enable_caching(mut self, enabled: bool) -> Self {
        self.config.enable_caching = enabled;
        self
    }

    /// Enable scoring
    pub fn enable_scoring(mut self, enabled: bool) -> Self {
        self.config.enable_scoring = enabled;
        self
    }

    /// Set the data source bridge
    pub fn bridge(mut self, bridge: Arc<DataSourceBridge>) -> Self {
        self.bridge = Some(bridge);
        self
    }

    /// Set the cache
    pub fn cache(mut self, cache: Arc<RedisCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Build the aggregator service
    pub fn build(self) -> Result<AggregatorService> {
        let mut service = AggregatorService::new(self.config);

        if let Some(bridge) = self.bridge {
            service = service.with_bridge(bridge);
        }

        if let Some(cache) = self.cache {
            service = service.with_cache(cache);
        }

        Ok(service)
    }
}

impl Default for AggregatorServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create aggregator from external data config
pub fn create_aggregator_from_config(
    config: &ExternalDataConfig,
    bridge: Arc<DataSourceBridge>,
    cache: Option<Arc<RedisCache>>,
) -> Result<AggregatorService> {
    let service_config = AggregatorServiceConfig {
        aggregator: config.aggregator.clone(),
        enable_caching: config.redis.enabled,
        ..Default::default()
    };

    let mut builder = AggregatorServiceBuilder::new()
        .config(service_config)
        .bridge(bridge);

    if let Some(cache) = cache {
        builder = builder.cache(cache);
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thalamus::sources::news::NewsSentiment;

    #[test]
    fn test_composite_scores_default() {
        let scores = CompositeScores::default();
        assert_eq!(scores.market_sentiment, 0.0);
        assert_eq!(scores.sentiment_confidence, 0.0);
    }

    #[test]
    fn test_composite_scores_from_data() {
        let mut data = ExternalDataPoint::new();

        // Add positive sentiment
        let sentiment = NewsSentiment {
            score: 0.5,
            confidence: 0.8,
            source_count: 10,
            keywords: vec!["bitcoin".to_string()],
            timestamp: Utc::now(),
        };
        data = data.with_news(sentiment);

        let scores = CompositeScores::from_data(&data);

        assert_eq!(scores.market_sentiment, 0.5);
        assert_eq!(scores.sentiment_confidence, 0.8);
        assert!(scores.opportunity_score > 0.0);
    }

    #[test]
    fn test_composite_scores_risk_calculation() {
        // High confidence, low volatility = lower risk
        let risk = CompositeScores::calculate_risk(0.5, 0.9, 0.1);
        assert!(risk < 0.7);

        // Low confidence, high volatility = higher risk
        let risk = CompositeScores::calculate_risk(0.5, 0.3, 0.8);
        assert!(risk > 0.7);
    }

    #[test]
    fn test_aggregator_metrics() {
        let metrics = AggregatorMetrics::new();

        metrics.record_aggregation(true, 100);
        metrics.record_aggregation(true, 200);
        metrics.record_aggregation(false, 0);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.aggregations, 3);
        assert_eq!(snapshot.successful, 2);
        assert_eq!(snapshot.failed, 1);
        assert!((snapshot.average_latency_ms - 150.0).abs() < 0.01);
        assert!((snapshot.success_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_aggregator_metrics_sequence() {
        let metrics = AggregatorMetrics::new();

        assert_eq!(metrics.next_sequence(), 0);
        assert_eq!(metrics.next_sequence(), 1);
        assert_eq!(metrics.next_sequence(), 2);
    }

    #[test]
    fn test_service_config_default() {
        let config = AggregatorServiceConfig::default();
        assert_eq!(config.feed_buffer_size, 1000);
        assert!(config.enable_caching);
        assert!(config.enable_scoring);
    }

    #[tokio::test]
    async fn test_aggregator_service_creation() {
        let service = AggregatorService::new(AggregatorServiceConfig::default());

        assert_eq!(service.state().await, ServiceState::Stopped);
        assert!(!service.is_running());
    }

    #[tokio::test]
    async fn test_aggregator_service_builder() {
        let service = AggregatorServiceBuilder::new()
            .poll_interval(30)
            .feed_buffer_size(500)
            .enable_caching(false)
            .build()
            .unwrap();

        assert_eq!(service.config.aggregator.poll_interval, 30);
        assert_eq!(service.config.feed_buffer_size, 500);
        assert!(!service.config.enable_caching);
    }

    #[tokio::test]
    async fn test_unified_feed_creation() {
        let service = AggregatorService::new(AggregatorServiceConfig::default());
        let data = ExternalDataPoint::new();

        let feed = service.create_feed(data, 100);

        assert!(!feed.id.is_empty());
        assert_eq!(feed.latency_ms, 100);
        assert_eq!(feed.sequence, 0);
    }

    #[tokio::test]
    async fn test_service_state_transitions() {
        let service = AggregatorService::new(AggregatorServiceConfig::default());

        // Initial state
        assert_eq!(service.state().await, ServiceState::Stopped);

        // Try to start without bridge - should fail
        let result = service.start().await;
        assert!(result.is_err());
    }
}
