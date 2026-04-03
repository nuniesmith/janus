//! External Data Sources Module
//!
//! Part of Thalamus region - Sensory Relay for External Data
//!
//! This module handles ingestion and routing of external data feeds including:
//! - News and sentiment data
//! - Weather data (relevant for energy/agriculture markets)
//! - Celestial data (moon phases, space weather, orbital mechanics)
//! - Alternative data sources
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  External Data Sources                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │    News      │  │   Weather    │  │  Celestial   │     │
//! │  │  Sentiment   │  │    Data      │  │    Data      │     │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
//! │         │                  │                  │             │
//! │         └──────────────────┼──────────────────┘             │
//! │                            │                                │
//! │                   ┌────────▼────────┐                       │
//! │                   │  Data Router    │                       │
//! │                   │  (Thalamus)     │                       │
//! │                   └────────┬────────┘                       │
//! │                            │                                │
//! │         ┌──────────────────┴──────────────────┐            │
//! │         │                                      │            │
//! │    ┌────▼────┐                          ┌─────▼─────┐      │
//! │    │ Forward │                          │ Backward  │      │
//! │    │ Service │                          │  Service  │      │
//! │    └─────────┘                          └───────────┘      │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Modules
//!
//! - `news` - News article and sentiment data types
//! - `weather` - Weather data types and sources
//! - `celestial` - Moon phase, space weather, and celestial data
//! - `clients` - API client implementations for external data sources
//! - `config` - Configuration management for API keys and settings
//! - `cache` - Redis cache layer for API response caching
//! - `bridge` - Data source bridge connecting API clients to service bridges
//! - `aggregator` - Aggregator service combining all data sources

pub mod bert_sentiment;
pub mod celestial;
pub mod chronos;
pub mod clients;
pub mod news;
pub mod weather;

// New modules for service integration
pub mod aggregator;
pub mod bridge;
pub mod cache;
pub mod config;

// Sentiment → Qdrant bridge and end-to-end pipeline
pub mod sentiment_pipeline;
pub mod sentiment_qdrant_bridge;

// Integration tests using wiremock
#[cfg(test)]
mod integration_tests;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-exports - Data types
pub use bert_sentiment::{
    BatchSentimentResult, BertModel, BertModelConfig, BertModelVariant, BertSentimentAnalyzer,
    BertSentimentConfig, BertSentimentStats, BertTokenizer, InferenceDevice, ModelOutput,
    SentimentResult, Token, TokenizedInput,
};

// Re-export Sentiment–Qdrant bridge types
pub use sentiment_qdrant_bridge::{
    BridgeStats as SentimentBridgeStats, RetrievedSentiment, SentimentQdrantBridge,
    SentimentQdrantConfig, SimilarityQuery, StorageContext,
};

// Re-export Sentiment pipeline types
pub use celestial::{CelestialData, CelestialSource, MoonPhase, SpaceWeather};
pub use chronos::{
    BinningMethod, ChronosConfig, ChronosError, ChronosForecast, ChronosInference,
    ChronosTokenizer, EngineState, ForecastCombiner, HorizonSummary, InferenceStats, SpecialTokens,
    TokenizerConfig,
};
pub use news::{NewsArticle, NewsSentiment, NewsSource};
pub use sentiment_pipeline::{
    PipelineOutput, PipelineStats, SentimentPipeline, SentimentPipelineConfig,
};
pub use weather::{WeatherData, WeatherSource};

// Re-export API clients
pub use clients::{
    ApiClient, ApiClientConfig, CryptoCompareClient, CryptoPanicClient, NewsApiClient,
    OpenWeatherMapClient, RateLimiter, SpaceWeatherClient,
};

// Re-export configuration types
pub use config::{
    AggregatorConfig, ApiKeyConfig, ConfigBuilder, ConfigError, DataSourceConfig,
    ExternalDataConfig, NewsSourceConfig, RedisConfig, WeatherSourceConfig,
};

// Re-export cache types
pub use cache::{CacheEntry, CacheError, CacheKey, CacheStats, CacheStatsSnapshot, RedisCache};

// Re-export bridge types
pub use bridge::{
    BridgeError, BridgeStats, CircuitBreaker, CircuitState, DataSourceBridge,
    DataSourceBridgeBuilder, ExternalDataEvent,
};

// Re-export aggregator types
pub use aggregator::{
    AggregatorError, AggregatorMetrics, AggregatorMetricsSnapshot, AggregatorService,
    AggregatorServiceBuilder, AggregatorServiceConfig, CompositeScores, ServiceState,
    UnifiedDataFeed,
};

/// Common trait for all external data sources
#[async_trait]
pub trait DataSource: Send + Sync {
    /// The type of data this source produces
    type Data: Send + Sync;

    /// Get the source name/identifier
    fn name(&self) -> &str;

    /// Fetch the latest data from the source
    async fn fetch_latest(&self) -> crate::common::Result<Self::Data>;

    /// Check if the source is healthy/available
    async fn health_check(&self) -> bool;

    /// Get the last update timestamp
    fn last_update(&self) -> Option<DateTime<Utc>>;
}

/// Unified external data point combining all sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalDataPoint {
    /// Timestamp of the data point
    pub timestamp: DateTime<Utc>,

    /// News/sentiment data if available
    pub news: Option<NewsSentiment>,

    /// Weather data if available
    pub weather: Option<WeatherData>,

    /// Celestial data if available
    pub celestial: Option<CelestialData>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ExternalDataPoint {
    /// Create a new empty external data point
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            news: None,
            weather: None,
            celestial: None,
            metadata: HashMap::new(),
        }
    }

    /// Create with timestamp
    pub fn with_timestamp(timestamp: DateTime<Utc>) -> Self {
        Self {
            timestamp,
            news: None,
            weather: None,
            celestial: None,
            metadata: HashMap::new(),
        }
    }

    /// Add news sentiment
    pub fn with_news(mut self, news: NewsSentiment) -> Self {
        self.news = Some(news);
        self
    }

    /// Add weather data
    pub fn with_weather(mut self, weather: WeatherData) -> Self {
        self.weather = Some(weather);
        self
    }

    /// Add celestial data
    pub fn with_celestial(mut self, celestial: CelestialData) -> Self {
        self.celestial = Some(celestial);
        self
    }

    /// Add metadata entry
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if any data is present
    pub fn has_data(&self) -> bool {
        self.news.is_some() || self.weather.is_some() || self.celestial.is_some()
    }

    /// Get data completeness score (0.0 to 1.0)
    pub fn completeness(&self) -> f64 {
        let mut count = 0.0;
        if self.news.is_some() {
            count += 1.0;
        }
        if self.weather.is_some() {
            count += 1.0;
        }
        if self.celestial.is_some() {
            count += 1.0;
        }
        count / 3.0
    }
}

impl Default for ExternalDataPoint {
    fn default() -> Self {
        Self::new()
    }
}

/// External data aggregator - collects from all sources
pub struct ExternalDataAggregator {
    news_sources: Vec<Box<dyn DataSource<Data = NewsSentiment>>>,
    weather_sources: Vec<Box<dyn DataSource<Data = WeatherData>>>,
    celestial_sources: Vec<Box<dyn DataSource<Data = CelestialData>>>,
}

impl ExternalDataAggregator {
    /// Create a new aggregator
    pub fn new() -> Self {
        Self {
            news_sources: Vec::new(),
            weather_sources: Vec::new(),
            celestial_sources: Vec::new(),
        }
    }

    /// Add a news source
    pub fn add_news_source(&mut self, source: Box<dyn DataSource<Data = NewsSentiment>>) {
        self.news_sources.push(source);
    }

    /// Add a weather source
    pub fn add_weather_source(&mut self, source: Box<dyn DataSource<Data = WeatherData>>) {
        self.weather_sources.push(source);
    }

    /// Add a celestial source
    pub fn add_celestial_source(&mut self, source: Box<dyn DataSource<Data = CelestialData>>) {
        self.celestial_sources.push(source);
    }

    /// Get number of news sources
    pub fn news_source_count(&self) -> usize {
        self.news_sources.len()
    }

    /// Get number of weather sources
    pub fn weather_source_count(&self) -> usize {
        self.weather_sources.len()
    }

    /// Get number of celestial sources
    pub fn celestial_source_count(&self) -> usize {
        self.celestial_sources.len()
    }

    /// Fetch all external data and aggregate.
    ///
    /// News sentiment is aggregated across **all** responding sources using a
    /// confidence-weighted average so that higher-confidence sources contribute
    /// more to the final score.  Keywords are merged and deduplicated.
    pub async fn fetch_all(&self) -> crate::common::Result<ExternalDataPoint> {
        let mut data = ExternalDataPoint::new();

        // --- News sentiment: aggregate across all available sources -----------
        let mut sentiments: Vec<NewsSentiment> = Vec::new();
        for source in &self.news_sources {
            if let Ok(sentiment) = source.fetch_latest().await {
                sentiments.push(sentiment);
            }
        }

        if !sentiments.is_empty() {
            data.news = Some(Self::aggregate_news_sentiments(&sentiments));
        }

        // --- Weather data: use first responding source -----------------------
        for source in &self.weather_sources {
            if let Ok(weather) = source.fetch_latest().await {
                data.weather = Some(weather);
                break;
            }
        }

        // --- Celestial data: use first responding source ---------------------
        for source in &self.celestial_sources {
            if let Ok(celestial) = source.fetch_latest().await {
                data.celestial = Some(celestial);
                break;
            }
        }

        Ok(data)
    }

    /// Aggregate multiple [`NewsSentiment`] values into one using a
    /// confidence-weighted average for the score and a simple average for
    /// confidence.  Keywords are merged and deduplicated.
    fn aggregate_news_sentiments(sentiments: &[NewsSentiment]) -> NewsSentiment {
        debug_assert!(!sentiments.is_empty());

        // Fast path: single source needs no aggregation
        if sentiments.len() == 1 {
            return sentiments[0].clone();
        }

        let total_confidence: f64 = sentiments.iter().map(|s| s.confidence).sum();

        // Confidence-weighted average score.  If every source has zero
        // confidence we fall back to a simple arithmetic mean.
        let aggregated_score = if total_confidence > 0.0 {
            sentiments
                .iter()
                .map(|s| s.score * s.confidence)
                .sum::<f64>()
                / total_confidence
        } else {
            sentiments.iter().map(|s| s.score).sum::<f64>() / sentiments.len() as f64
        };

        // Average confidence across sources
        let aggregated_confidence = total_confidence / sentiments.len() as f64;

        // Merge and deduplicate keywords, preserving insertion order
        let mut seen = std::collections::HashSet::new();
        let mut merged_keywords = Vec::new();
        for s in sentiments {
            for kw in &s.keywords {
                if seen.insert(kw.clone()) {
                    merged_keywords.push(kw.clone());
                }
            }
        }

        // Use the most recent timestamp
        let latest_ts = sentiments
            .iter()
            .map(|s| s.timestamp)
            .max()
            .unwrap_or_else(Utc::now);

        NewsSentiment {
            score: aggregated_score,
            confidence: aggregated_confidence,
            source_count: sentiments.len(),
            keywords: merged_keywords,
            timestamp: latest_ts,
        }
    }

    /// Get health status of all sources
    pub async fn health_status(&self) -> HashMap<String, bool> {
        let mut status = HashMap::new();

        for source in &self.news_sources {
            status.insert(source.name().to_string(), source.health_check().await);
        }

        for source in &self.weather_sources {
            status.insert(source.name().to_string(), source.health_check().await);
        }

        for source in &self.celestial_sources {
            status.insert(source.name().to_string(), source.health_check().await);
        }

        status
    }

    /// Clear all sources
    pub fn clear(&mut self) {
        self.news_sources.clear();
        self.weather_sources.clear();
        self.celestial_sources.clear();
    }
}

impl Default for ExternalDataAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_data_point_creation() {
        let data = ExternalDataPoint::new();
        assert!(!data.has_data());
        assert_eq!(data.completeness(), 0.0);
    }

    #[test]
    fn test_external_data_point_builder() {
        let sentiment = NewsSentiment {
            score: 0.5,
            confidence: 0.8,
            source_count: 5,
            keywords: vec!["bitcoin".to_string()],
            timestamp: Utc::now(),
        };

        let data = ExternalDataPoint::new()
            .with_news(sentiment)
            .with_metadata("source", "test");

        assert!(data.has_data());
        assert!(data.news.is_some());
        assert_eq!(data.metadata.get("source"), Some(&"test".to_string()));
        assert!((data.completeness() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_external_data_point_completeness() {
        let sentiment = NewsSentiment {
            score: 0.5,
            confidence: 0.8,
            source_count: 5,
            keywords: vec![],
            timestamp: Utc::now(),
        };

        let weather = WeatherData::new("New York".to_string(), 20.0);

        let celestial = CelestialData::new(MoonPhase::FullMoon);

        let data = ExternalDataPoint::new()
            .with_news(sentiment)
            .with_weather(weather)
            .with_celestial(celestial);

        assert_eq!(data.completeness(), 1.0);
    }

    #[test]
    fn test_aggregator_creation() {
        let aggregator = ExternalDataAggregator::new();
        assert_eq!(aggregator.news_source_count(), 0);
        assert_eq!(aggregator.weather_source_count(), 0);
        assert_eq!(aggregator.celestial_source_count(), 0);
    }
}
