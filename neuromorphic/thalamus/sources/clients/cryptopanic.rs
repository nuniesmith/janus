//! CryptoPanic News API Client
//!
//! Provides access to cryptocurrency news aggregated by CryptoPanic.
//! API Documentation: https://cryptopanic.com/developers/api/
//!
//! ## Features
//!
//! - Aggregated crypto news from multiple sources
//! - News filtering by coin/token
//! - Sentiment voting data (bullish/bearish)
//! - News importance/impact ratings

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use tracing::{debug, error, warn};

use super::{ApiClient, ApiClientConfig, RateLimiter};
use crate::common::{Error, Result};
use crate::thalamus::sources::DataSource;
use crate::thalamus::sources::news::{NewsArticle, NewsSentiment};

/// CryptoPanic API base URL
const CRYPTOPANIC_BASE_URL: &str = "https://cryptopanic.com/api/v1";

/// News filter types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewsFilter {
    /// Rising news (trending)
    Rising,
    /// Hot news (most activity)
    Hot,
    /// Bullish news only
    Bullish,
    /// Bearish news only
    Bearish,
    /// Important news (high impact)
    Important,
    /// Saved news (requires auth)
    Saved,
    /// Latest news (default)
    Latest,
}

impl NewsFilter {
    fn as_str(&self) -> &str {
        match self {
            NewsFilter::Rising => "rising",
            NewsFilter::Hot => "hot",
            NewsFilter::Bullish => "bullish",
            NewsFilter::Bearish => "bearish",
            NewsFilter::Important => "important",
            NewsFilter::Saved => "saved",
            NewsFilter::Latest => "latest",
        }
    }
}

/// News kind (type of content)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewsKind {
    /// News articles
    News,
    /// Media (videos, podcasts)
    Media,
    /// All types
    All,
}

impl NewsKind {
    fn as_str(&self) -> &str {
        match self {
            NewsKind::News => "news",
            NewsKind::Media => "media",
            NewsKind::All => "all",
        }
    }
}

/// CryptoPanic API client
pub struct CryptoPanicClient {
    /// HTTP client
    client: reqwest::Client,
    /// API configuration
    config: ApiClientConfig,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Cached articles
    cached_articles: tokio::sync::RwLock<Vec<NewsArticle>>,
    /// Last update timestamp
    last_update: tokio::sync::RwLock<Option<DateTime<Utc>>>,
}

impl CryptoPanicClient {
    /// Create a new CryptoPanic client
    /// Note: API key is required for CryptoPanic
    pub fn new(api_key: String) -> Result<Self> {
        let config = ApiClientConfig {
            api_key: Some(api_key),
            base_url: CRYPTOPANIC_BASE_URL.to_string(),
            ..Default::default()
        };

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            rate_limiter: RateLimiter::new(5.0), // 5 requests per second
            cached_articles: tokio::sync::RwLock::new(Vec::new()),
            last_update: tokio::sync::RwLock::new(None),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: ApiClientConfig) -> Result<Self> {
        if config.api_key.is_none() {
            return Err(Error::Other("CryptoPanic requires an API key".to_string()));
        }

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            rate_limiter: RateLimiter::new(5.0),
            cached_articles: tokio::sync::RwLock::new(Vec::new()),
            last_update: tokio::sync::RwLock::new(None),
        })
    }

    /// Fetch news with optional filters
    pub async fn fetch_news(
        &self,
        filter: Option<NewsFilter>,
        kind: Option<NewsKind>,
        currencies: Option<&[&str]>,
        regions: Option<&[&str]>,
    ) -> Result<Vec<NewsArticle>> {
        self.rate_limiter.wait().await;

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Other("API key required".to_string()))?;

        let mut url = format!("{}/posts/?auth_token={}", self.config.base_url, api_key);

        // Add filter
        if let Some(f) = filter {
            url.push_str(&format!("&filter={}", f.as_str()));
        }

        // Add kind
        if let Some(k) = kind {
            url.push_str(&format!("&kind={}", k.as_str()));
        }

        // Add currencies filter
        if let Some(coins) = currencies {
            let coins_str = coins.join(",");
            url.push_str(&format!("&currencies={}", coins_str));
        }

        // Add regions filter
        if let Some(regs) = regions {
            let regions_str = regs.join(",");
            url.push_str(&format!("&regions={}", regions_str));
        }

        // Always request public data
        url.push_str("&public=true");

        debug!("Fetching CryptoPanic news");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("CryptoPanic API error: {} - {}", status, text);
            return Err(Error::Other(format!("API error: {} - {}", status, text)));
        }

        let api_response: CryptoPanicResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        // Convert to our NewsArticle format
        let articles: Vec<NewsArticle> = api_response
            .results
            .into_iter()
            .map(|item| self.convert_article(item))
            .collect();

        // Update cache
        {
            let mut cache = self.cached_articles.write().await;
            *cache = articles.clone();
            let mut last = self.last_update.write().await;
            *last = Some(Utc::now());
        }

        debug!("Fetched {} articles from CryptoPanic", articles.len());
        Ok(articles)
    }

    /// Fetch bullish news only
    pub async fn fetch_bullish_news(
        &self,
        currencies: Option<&[&str]>,
    ) -> Result<Vec<NewsArticle>> {
        self.fetch_news(
            Some(NewsFilter::Bullish),
            Some(NewsKind::News),
            currencies,
            None,
        )
        .await
    }

    /// Fetch bearish news only
    pub async fn fetch_bearish_news(
        &self,
        currencies: Option<&[&str]>,
    ) -> Result<Vec<NewsArticle>> {
        self.fetch_news(
            Some(NewsFilter::Bearish),
            Some(NewsKind::News),
            currencies,
            None,
        )
        .await
    }

    /// Fetch important/high-impact news
    pub async fn fetch_important_news(
        &self,
        currencies: Option<&[&str]>,
    ) -> Result<Vec<NewsArticle>> {
        self.fetch_news(
            Some(NewsFilter::Important),
            Some(NewsKind::News),
            currencies,
            None,
        )
        .await
    }

    /// Fetch trending news
    pub async fn fetch_trending_news(
        &self,
        currencies: Option<&[&str]>,
    ) -> Result<Vec<NewsArticle>> {
        self.fetch_news(
            Some(NewsFilter::Hot),
            Some(NewsKind::News),
            currencies,
            None,
        )
        .await
    }

    /// Convert CryptoPanic post to our NewsArticle format
    fn convert_article(&self, item: CryptoPanicPost) -> NewsArticle {
        // Parse timestamp
        let published_at = DateTime::parse_from_rfc3339(&item.published_at)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        // Calculate sentiment from votes
        let sentiment = if let Some(votes) = &item.votes {
            let positive = votes.positive as f64;
            let negative = votes.negative as f64;
            let total = positive + negative;

            if total > 0.0 {
                Some((positive - negative) / total)
            } else {
                None
            }
        } else {
            None
        };

        // Extract symbols from currencies
        let symbols: Vec<String> = item
            .currencies
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.code)
            .collect();

        // Build categories from metadata
        let mut categories = Vec::new();
        if let Some(kind) = &item.kind {
            categories.push(kind.clone());
        }
        if let Some(domain) = &item.domain {
            categories.push(domain.clone());
        }

        NewsArticle {
            id: item.id.to_string(),
            title: item.title,
            summary: None,
            content: None,
            source: item
                .source
                .map(|s| s.title)
                .unwrap_or_else(|| "CryptoPanic".to_string()),
            url: item.url,
            published_at,
            symbols,
            sentiment,
            categories,
            author: None,
        }
    }

    /// Calculate aggregate sentiment from cached articles
    pub async fn calculate_sentiment(&self) -> NewsSentiment {
        let articles = self.cached_articles.read().await;

        if articles.is_empty() {
            return NewsSentiment::neutral();
        }

        let sentiments: Vec<f64> = articles.iter().filter_map(|a| a.sentiment).collect();

        if sentiments.is_empty() {
            return NewsSentiment::neutral();
        }

        let avg_score = sentiments.iter().sum::<f64>() / sentiments.len() as f64;

        // Calculate confidence based on sample size and agreement
        let variance = sentiments
            .iter()
            .map(|s| (s - avg_score).powi(2))
            .sum::<f64>()
            / sentiments.len() as f64;
        let std_dev = variance.sqrt();

        let size_factor = (sentiments.len() as f64 / 50.0).min(1.0);
        let agreement_factor = 1.0 - std_dev.min(1.0);
        let confidence = (size_factor + agreement_factor) / 2.0;

        // Extract top symbols as keywords
        let mut symbol_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for article in articles.iter() {
            for symbol in &article.symbols {
                *symbol_counts.entry(symbol.clone()).or_insert(0) += 1;
            }
        }
        let mut keywords: Vec<(String, usize)> = symbol_counts.into_iter().collect();
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

    /// Get fear and greed indicator based on bullish/bearish ratio
    pub async fn fear_greed_indicator(&self) -> f64 {
        let articles = self.cached_articles.read().await;

        if articles.is_empty() {
            return 0.5; // Neutral
        }

        let sentiments: Vec<f64> = articles.iter().filter_map(|a| a.sentiment).collect();

        if sentiments.is_empty() {
            return 0.5;
        }

        // Calculate ratio of bullish to bearish
        let bullish = sentiments.iter().filter(|&&s| s > 0.2).count();
        let bearish = sentiments.iter().filter(|&&s| s < -0.2).count();
        let total = bullish + bearish;

        if total == 0 {
            return 0.5;
        }

        // Return 0 = extreme fear, 1 = extreme greed
        bullish as f64 / total as f64
    }

    /// Get cached articles
    pub async fn cached_articles(&self) -> Vec<NewsArticle> {
        self.cached_articles.read().await.clone()
    }

    /// Get last update timestamp
    pub async fn last_update_time(&self) -> Option<DateTime<Utc>> {
        *self.last_update.read().await
    }
}

#[async_trait]
impl ApiClient for CryptoPanicClient {
    fn name(&self) -> &str {
        "cryptopanic"
    }

    fn is_configured(&self) -> bool {
        self.config.api_key.is_some()
    }

    async fn health_check(&self) -> bool {
        if !self.is_configured() {
            return false;
        }

        // Try fetching a small amount of data
        match self.fetch_news(None, None, None, None).await {
            Ok(_) => true,
            Err(e) => {
                warn!("CryptoPanic health check failed: {}", e);
                false
            }
        }
    }
}

#[async_trait]
impl DataSource for CryptoPanicClient {
    type Data = NewsSentiment;

    fn name(&self) -> &str {
        "cryptopanic"
    }

    async fn fetch_latest(&self) -> Result<Self::Data> {
        // Fetch fresh news
        self.fetch_news(None, Some(NewsKind::News), None, None)
            .await?;
        // Calculate sentiment
        Ok(self.calculate_sentiment().await)
    }

    async fn health_check(&self) -> bool {
        <Self as ApiClient>::health_check(self).await
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        None
    }
}

// ============================================================================
// API Response Types
// ============================================================================

/// CryptoPanic API response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CryptoPanicResponse {
    /// Number of results
    count: Option<i32>,
    /// Next page URL
    next: Option<String>,
    /// Previous page URL
    previous: Option<String>,
    /// Results (posts)
    results: Vec<CryptoPanicPost>,
}

/// Individual post from CryptoPanic
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CryptoPanicPost {
    /// Post ID
    id: i64,
    /// Post kind (news, media)
    kind: Option<String>,
    /// Domain of the source
    domain: Option<String>,
    /// Source information
    source: Option<CryptoPanicSource>,
    /// Post title
    title: String,
    /// Published timestamp (ISO 8601)
    published_at: String,
    /// Post URL
    url: String,
    /// Related currencies
    currencies: Option<Vec<CryptoPanicCurrency>>,
    /// Vote data
    votes: Option<CryptoPanicVotes>,
    /// Metadata
    metadata: Option<CryptoPanicMetadata>,
}

/// Source information
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CryptoPanicSource {
    /// Source title
    title: String,
    /// Source region
    region: Option<String>,
    /// Source domain
    domain: Option<String>,
    /// Source path
    path: Option<String>,
}

/// Currency information
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CryptoPanicCurrency {
    /// Currency code (e.g., "BTC")
    code: String,
    /// Currency title (e.g., "Bitcoin")
    title: String,
    /// Currency slug
    slug: String,
    /// Currency URL on CryptoPanic
    url: String,
}

/// Vote information
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CryptoPanicVotes {
    /// Negative votes (bearish)
    negative: i32,
    /// Positive votes (bullish)
    positive: i32,
    /// Important votes
    important: i32,
    /// Liked votes
    liked: i32,
    /// Disliked votes
    disliked: i32,
    /// LOL votes
    lol: i32,
    /// Toxic votes
    toxic: i32,
    /// Saved count
    saved: i32,
    /// Comments count
    comments: i32,
}

/// Metadata for a post
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CryptoPanicMetadata {
    /// Description/summary
    description: Option<String>,
    /// Image URL
    image: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_requires_api_key() {
        // CryptoPanic requires an API key
        let config = ApiClientConfig {
            api_key: None,
            base_url: CRYPTOPANIC_BASE_URL.to_string(),
            ..Default::default()
        };

        let result = CryptoPanicClient::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_client_creation_with_key() {
        let client = CryptoPanicClient::new("test_api_key".to_string());
        assert!(client.is_ok());

        let client = client.unwrap();
        assert!(client.is_configured());
    }

    #[test]
    fn test_news_filter() {
        assert_eq!(NewsFilter::Bullish.as_str(), "bullish");
        assert_eq!(NewsFilter::Bearish.as_str(), "bearish");
        assert_eq!(NewsFilter::Hot.as_str(), "hot");
        assert_eq!(NewsFilter::Important.as_str(), "important");
    }

    #[test]
    fn test_news_kind() {
        assert_eq!(NewsKind::News.as_str(), "news");
        assert_eq!(NewsKind::Media.as_str(), "media");
        assert_eq!(NewsKind::All.as_str(), "all");
    }

    #[tokio::test]
    async fn test_calculate_sentiment_empty() {
        let client = CryptoPanicClient::new("test_key".to_string()).unwrap();
        let sentiment = client.calculate_sentiment().await;

        assert_eq!(sentiment.score, 0.0);
        assert_eq!(sentiment.confidence, 0.0);
        assert!(sentiment.is_neutral());
    }

    #[tokio::test]
    async fn test_fear_greed_indicator_neutral() {
        let client = CryptoPanicClient::new("test_key".to_string()).unwrap();
        let indicator = client.fear_greed_indicator().await;

        // Empty cache should return neutral (0.5)
        assert_eq!(indicator, 0.5);
    }
}
