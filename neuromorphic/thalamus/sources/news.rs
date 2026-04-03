//! News and Sentiment Data Sources
//!
//! Part of Thalamus region - External Data Sources
//!
//! This module handles news and sentiment data ingestion from various sources:
//! - Financial news APIs (Bloomberg, Reuters, etc.)
//! - Social media sentiment (Twitter, Reddit, etc.)
//! - Crypto-specific sources (CoinDesk, CryptoCompare, etc.)
//! - Alternative data providers

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::DataSource;

/// Aggregated news sentiment score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsSentiment {
    /// Sentiment score from -1.0 (bearish) to 1.0 (bullish)
    pub score: f64,

    /// Confidence level of the sentiment (0.0 to 1.0)
    pub confidence: f64,

    /// Number of sources contributing to this sentiment
    pub source_count: usize,

    /// Keywords/topics extracted from news
    pub keywords: Vec<String>,

    /// Timestamp of aggregation
    pub timestamp: DateTime<Utc>,
}

impl NewsSentiment {
    /// Create a neutral sentiment
    pub fn neutral() -> Self {
        Self {
            score: 0.0,
            confidence: 0.0,
            source_count: 0,
            keywords: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    /// Check if sentiment is bullish
    pub fn is_bullish(&self) -> bool {
        self.score > 0.2 && self.confidence > 0.5
    }

    /// Check if sentiment is bearish
    pub fn is_bearish(&self) -> bool {
        self.score < -0.2 && self.confidence > 0.5
    }

    /// Check if sentiment is neutral
    pub fn is_neutral(&self) -> bool {
        self.score.abs() <= 0.2 || self.confidence <= 0.5
    }
}

impl Default for NewsSentiment {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Individual news article
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsArticle {
    /// Unique identifier
    pub id: String,

    /// Article title
    pub title: String,

    /// Article summary/description
    pub summary: Option<String>,

    /// Full article content (if available)
    pub content: Option<String>,

    /// Source name (e.g., "Reuters", "Bloomberg")
    pub source: String,

    /// Article URL
    pub url: String,

    /// Publication timestamp
    pub published_at: DateTime<Utc>,

    /// Related symbols/tickers
    pub symbols: Vec<String>,

    /// Sentiment score for this article (-1.0 to 1.0)
    pub sentiment: Option<f64>,

    /// Article categories/tags
    pub categories: Vec<String>,

    /// Author (if available)
    pub author: Option<String>,
}

impl NewsArticle {
    /// Create a new article with minimal required fields
    pub fn new(id: String, title: String, source: String, url: String) -> Self {
        Self {
            id,
            title,
            summary: None,
            content: None,
            source,
            url,
            published_at: Utc::now(),
            symbols: Vec::new(),
            sentiment: None,
            categories: Vec::new(),
            author: None,
        }
    }
}

/// Configuration for news sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsSourceConfig {
    /// API key for the news source
    pub api_key: Option<String>,

    /// Base URL for the API
    pub base_url: String,

    /// Symbols to track
    pub symbols: Vec<String>,

    /// Rate limit (requests per minute)
    pub rate_limit: u32,

    /// Enable sentiment analysis
    pub enable_sentiment: bool,

    /// Maximum articles to fetch per request
    pub max_articles: usize,

    /// Language filter
    pub language: String,
}

impl Default for NewsSourceConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: String::new(),
            symbols: vec!["BTC".to_string(), "ETH".to_string()],
            rate_limit: 60,
            enable_sentiment: true,
            max_articles: 100,
            language: "en".to_string(),
        }
    }
}

/// News source types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NewsSourceType {
    /// Traditional financial news (Bloomberg, Reuters)
    Financial,
    /// Crypto-specific news (CoinDesk, CryptoCompare)
    Crypto,
    /// Social media (Twitter, Reddit)
    Social,
    /// Alternative data providers
    Alternative,
    /// RSS feeds
    Rss,
    /// Custom webhook
    Webhook,
}

/// Abstract news source that can be implemented for different providers
pub struct NewsSource {
    /// Source name
    name: String,

    /// Source type
    source_type: NewsSourceType,

    /// Configuration
    config: NewsSourceConfig,

    /// Last update timestamp (reserved for future caching)
    #[allow(dead_code)]
    last_update: Option<DateTime<Utc>>,

    /// Cached articles
    cached_articles: Vec<NewsArticle>,

    /// Cached sentiment (reserved for future caching)
    #[allow(dead_code)]
    cached_sentiment: Option<NewsSentiment>,
}

impl NewsSource {
    /// Create a new news source
    pub fn new(name: String, source_type: NewsSourceType, config: NewsSourceConfig) -> Self {
        Self {
            name,
            source_type,
            config,
            last_update: None,
            cached_articles: Vec::new(),
            cached_sentiment: None,
        }
    }

    /// Get cached articles
    pub fn cached_articles(&self) -> &[NewsArticle] {
        &self.cached_articles
    }

    /// Get source type
    pub fn source_type(&self) -> NewsSourceType {
        self.source_type
    }

    /// Get configuration
    pub fn config(&self) -> &NewsSourceConfig {
        &self.config
    }

    /// Calculate aggregate sentiment from articles
    fn calculate_sentiment(&self, articles: &[NewsArticle]) -> NewsSentiment {
        if articles.is_empty() {
            return NewsSentiment::neutral();
        }

        let sentiments: Vec<f64> = articles.iter().filter_map(|a| a.sentiment).collect();

        if sentiments.is_empty() {
            return NewsSentiment::neutral();
        }

        let avg_score = sentiments.iter().sum::<f64>() / sentiments.len() as f64;

        // Calculate confidence based on agreement between sources
        let variance = sentiments
            .iter()
            .map(|s| (s - avg_score).powi(2))
            .sum::<f64>()
            / sentiments.len() as f64;
        let confidence = 1.0 - variance.sqrt().min(1.0);

        // Extract keywords from articles
        let mut keyword_counts: HashMap<String, usize> = HashMap::new();
        for article in articles {
            for category in &article.categories {
                *keyword_counts.entry(category.clone()).or_insert(0) += 1;
            }
        }
        let mut keywords: Vec<(String, usize)> = keyword_counts.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));
        let top_keywords: Vec<String> = keywords.into_iter().take(10).map(|(k, _)| k).collect();

        NewsSentiment {
            score: avg_score,
            confidence,
            source_count: sentiments.len(),
            keywords: top_keywords,
            timestamp: Utc::now(),
        }
    }

    /// Fetch articles (placeholder - implement for each provider)
    async fn fetch_articles(&self) -> crate::common::Result<Vec<NewsArticle>> {
        // This would be implemented differently for each news provider
        // For now, return empty to allow compilation
        Ok(Vec::new())
    }
}

#[async_trait]
impl DataSource for NewsSource {
    type Data = NewsSentiment;

    fn name(&self) -> &str {
        &self.name
    }

    async fn fetch_latest(&self) -> crate::common::Result<Self::Data> {
        let articles = self.fetch_articles().await?;
        Ok(self.calculate_sentiment(&articles))
    }

    async fn health_check(&self) -> bool {
        // Basic health check - verify API connectivity
        // In a real implementation, this would make a test API call
        self.config.api_key.is_some() || !self.config.base_url.is_empty()
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        self.last_update
    }
}

/// Social media sentiment source (Twitter, Reddit, etc.)
pub struct SocialSentimentSource {
    inner: NewsSource,
    /// Social platform name
    platform: String,
    /// Tracked hashtags/subreddits
    tracked_topics: Vec<String>,
}

impl SocialSentimentSource {
    /// Create a new social sentiment source
    pub fn new(platform: String, config: NewsSourceConfig) -> Self {
        let name = format!("social_{}", platform.to_lowercase());
        Self {
            inner: NewsSource::new(name, NewsSourceType::Social, config),
            platform,
            tracked_topics: Vec::new(),
        }
    }

    /// Add topics to track
    pub fn track_topics(&mut self, topics: Vec<String>) {
        self.tracked_topics.extend(topics);
    }

    /// Get platform name
    pub fn platform(&self) -> &str {
        &self.platform
    }
}

#[async_trait]
impl DataSource for SocialSentimentSource {
    type Data = NewsSentiment;

    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn fetch_latest(&self) -> crate::common::Result<Self::Data> {
        self.inner.fetch_latest().await
    }

    async fn health_check(&self) -> bool {
        self.inner.health_check().await
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        self.inner.last_update()
    }
}

/// Crypto news aggregator that combines multiple crypto-specific sources
pub struct CryptoNewsAggregator {
    sources: Vec<NewsSource>,
    last_update: Option<DateTime<Utc>>,
}

impl CryptoNewsAggregator {
    /// Create a new crypto news aggregator
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            last_update: None,
        }
    }

    /// Add a news source
    pub fn add_source(&mut self, source: NewsSource) {
        self.sources.push(source);
    }

    /// Get number of sources
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

impl Default for CryptoNewsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataSource for CryptoNewsAggregator {
    type Data = NewsSentiment;

    fn name(&self) -> &str {
        "crypto_news_aggregator"
    }

    async fn fetch_latest(&self) -> crate::common::Result<Self::Data> {
        let mut all_sentiments: Vec<NewsSentiment> = Vec::new();

        for source in &self.sources {
            if let Ok(sentiment) = source.fetch_latest().await {
                all_sentiments.push(sentiment);
            }
        }

        if all_sentiments.is_empty() {
            return Ok(NewsSentiment::neutral());
        }

        // Aggregate sentiments from all sources
        let total_sources: usize = all_sentiments.iter().map(|s| s.source_count).sum();
        let weighted_score: f64 = all_sentiments
            .iter()
            .map(|s| s.score * s.source_count as f64)
            .sum::<f64>()
            / total_sources as f64;

        let avg_confidence: f64 =
            all_sentiments.iter().map(|s| s.confidence).sum::<f64>() / all_sentiments.len() as f64;

        // Merge keywords
        let mut all_keywords: Vec<String> = all_sentiments
            .into_iter()
            .flat_map(|s| s.keywords)
            .collect();
        all_keywords.sort();
        all_keywords.dedup();
        all_keywords.truncate(20);

        Ok(NewsSentiment {
            score: weighted_score,
            confidence: avg_confidence,
            source_count: total_sources,
            keywords: all_keywords,
            timestamp: Utc::now(),
        })
    }

    async fn health_check(&self) -> bool {
        // At least one source should be healthy
        for source in &self.sources {
            if source.health_check().await {
                return true;
            }
        }
        false
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        self.last_update
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_creation() {
        let sentiment = NewsSentiment::neutral();
        assert_eq!(sentiment.score, 0.0);
        assert!(sentiment.is_neutral());
    }

    #[test]
    fn test_sentiment_bullish() {
        let sentiment = NewsSentiment {
            score: 0.5,
            confidence: 0.8,
            source_count: 10,
            keywords: vec!["bullish".to_string()],
            timestamp: Utc::now(),
        };
        assert!(sentiment.is_bullish());
        assert!(!sentiment.is_bearish());
    }

    #[test]
    fn test_sentiment_bearish() {
        let sentiment = NewsSentiment {
            score: -0.5,
            confidence: 0.8,
            source_count: 10,
            keywords: vec!["bearish".to_string()],
            timestamp: Utc::now(),
        };
        assert!(sentiment.is_bearish());
        assert!(!sentiment.is_bullish());
    }

    #[test]
    fn test_news_article_creation() {
        let article = NewsArticle::new(
            "123".to_string(),
            "Test Article".to_string(),
            "TestSource".to_string(),
            "https://example.com".to_string(),
        );
        assert_eq!(article.id, "123");
        assert_eq!(article.title, "Test Article");
    }

    #[test]
    fn test_news_source_config_default() {
        let config = NewsSourceConfig::default();
        assert!(config.symbols.contains(&"BTC".to_string()));
        assert!(config.enable_sentiment);
    }

    #[test]
    fn test_crypto_news_aggregator() {
        let aggregator = CryptoNewsAggregator::new();
        assert_eq!(aggregator.source_count(), 0);
    }
}
