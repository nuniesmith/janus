//! CryptoCompare News API Client
//!
//! Provides access to cryptocurrency news and social data from CryptoCompare.
//! API Documentation: https://min-api.cryptocompare.com/documentation
//!
//! ## Features
//!
//! - Latest crypto news articles
//! - News by category (blockchain, trading, mining, etc.)
//! - News by coin/token
//! - Social stats for coins

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use super::{ApiClient, ApiClientConfig, RateLimiter};
use crate::common::{Error, Result};
use crate::thalamus::sources::DataSource;
use crate::thalamus::sources::news::{NewsArticle, NewsSentiment};

/// CryptoCompare API base URL
const CRYPTOCOMPARE_BASE_URL: &str = "https://min-api.cryptocompare.com";

/// CryptoCompare news API client
pub struct CryptoCompareClient {
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

impl CryptoCompareClient {
    /// Create a new CryptoCompare client
    pub fn new(api_key: Option<String>) -> Result<Self> {
        let config = ApiClientConfig {
            api_key,
            base_url: CRYPTOCOMPARE_BASE_URL.to_string(),
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
            rate_limiter: RateLimiter::new(10.0), // 10 requests per second
            cached_articles: tokio::sync::RwLock::new(Vec::new()),
            last_update: tokio::sync::RwLock::new(None),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: ApiClientConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            rate_limiter: RateLimiter::new(10.0),
            cached_articles: tokio::sync::RwLock::new(Vec::new()),
            last_update: tokio::sync::RwLock::new(None),
        })
    }

    /// Fetch latest news articles
    pub async fn fetch_news(&self, categories: Option<&[&str]>) -> Result<Vec<NewsArticle>> {
        self.rate_limiter.wait().await;

        let mut url = format!("{}/data/v2/news/?lang=EN", self.config.base_url);

        // Add categories filter if specified
        if let Some(cats) = categories {
            let cat_str = cats.join(",");
            url.push_str(&format!("&categories={}", cat_str));
        }

        // Add API key if available
        if let Some(ref key) = self.config.api_key {
            url.push_str(&format!("&api_key={}", key));
        }

        debug!("Fetching CryptoCompare news from: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("CryptoCompare API error: {} - {}", status, text);
            return Err(Error::Other(format!("API error: {} - {}", status, text)));
        }

        let api_response: CryptoCompareNewsResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        if api_response.response != "Success" {
            warn!("CryptoCompare API returned non-success response");
            return Ok(Vec::new());
        }

        // Convert to our NewsArticle format
        let articles: Vec<NewsArticle> = api_response
            .data
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

        debug!("Fetched {} articles from CryptoCompare", articles.len());
        Ok(articles)
    }

    /// Fetch news for specific coins
    pub async fn fetch_news_for_coins(&self, coins: &[&str]) -> Result<Vec<NewsArticle>> {
        self.rate_limiter.wait().await;

        let coins_str = coins.join(",");
        let mut url = format!(
            "{}/data/v2/news/?lang=EN&categories={}",
            self.config.base_url, coins_str
        );

        if let Some(ref key) = self.config.api_key {
            url.push_str(&format!("&api_key={}", key));
        }

        debug!("Fetching CryptoCompare news for coins: {:?}", coins);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Other(format!("API error: {}", response.status())));
        }

        let api_response: CryptoCompareNewsResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        let articles: Vec<NewsArticle> = api_response
            .data
            .into_iter()
            .map(|item| self.convert_article(item))
            .collect();

        Ok(articles)
    }

    /// Fetch social stats for a coin
    pub async fn fetch_social_stats(&self, coin_id: u32) -> Result<CryptoCompareSocialStats> {
        self.rate_limiter.wait().await;

        let mut url = format!(
            "{}/data/social/coin/latest?coinId={}",
            self.config.base_url, coin_id
        );

        if let Some(ref key) = self.config.api_key {
            url.push_str(&format!("&api_key={}", key));
        }

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Other(format!("API error: {}", response.status())));
        }

        let api_response: CryptoCompareSocialResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        Ok(api_response.data)
    }

    /// Convert CryptoCompare news item to our NewsArticle format
    fn convert_article(&self, item: CryptoCompareNewsItem) -> NewsArticle {
        // Parse timestamp
        let published_at = Utc
            .timestamp_opt(item.published_on, 0)
            .single()
            .unwrap_or_else(Utc::now);

        // Extract sentiment from upvotes/downvotes if available
        let sentiment = if item.upvotes > 0 || item.downvotes > 0 {
            let total = (item.upvotes + item.downvotes) as f64;
            if total > 0.0 {
                Some((item.upvotes as f64 - item.downvotes as f64) / total)
            } else {
                None
            }
        } else {
            None
        };

        // Extract symbols from categories
        let symbols: Vec<String> = item
            .categories
            .split('|')
            .filter(|c| c.len() <= 5) // Likely ticker symbols
            .map(|s| s.to_uppercase())
            .collect();

        // Extract categories
        let categories: Vec<String> = item
            .categories
            .split('|')
            .filter(|c| c.len() > 5) // Likely category names
            .map(|s| s.to_string())
            .collect();

        NewsArticle {
            id: item.id.to_string(),
            title: item.title,
            summary: Some(item.body),
            content: None,
            source: item
                .source_info
                .map(|s| s.name)
                .unwrap_or_else(|| "CryptoCompare".to_string()),
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

        // Higher sample size and lower variance = higher confidence
        let size_factor = (sentiments.len() as f64 / 50.0).min(1.0);
        let agreement_factor = 1.0 - std_dev.min(1.0);
        let confidence = (size_factor + agreement_factor) / 2.0;

        // Extract keywords from article categories
        let mut keyword_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for article in articles.iter() {
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

    /// Get cached articles
    pub async fn cached_articles(&self) -> Vec<NewsArticle> {
        self.cached_articles.read().await.clone()
    }

    /// Get last update timestamp
    pub async fn last_update(&self) -> Option<DateTime<Utc>> {
        *self.last_update.read().await
    }
}

#[async_trait]
impl ApiClient for CryptoCompareClient {
    fn name(&self) -> &str {
        "cryptocompare"
    }

    fn is_configured(&self) -> bool {
        // CryptoCompare works without API key (with rate limits)
        true
    }

    async fn health_check(&self) -> bool {
        // Try a simple request to check connectivity
        let url = format!("{}/data/v2/news/?lang=EN", self.config.base_url);
        match self.client.get(&url).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }
}

#[async_trait]
impl DataSource for CryptoCompareClient {
    type Data = NewsSentiment;

    fn name(&self) -> &str {
        "cryptocompare"
    }

    async fn fetch_latest(&self) -> Result<Self::Data> {
        // Fetch fresh news
        self.fetch_news(None).await?;
        // Calculate sentiment
        Ok(self.calculate_sentiment().await)
    }

    async fn health_check(&self) -> bool {
        <Self as ApiClient>::health_check(self).await
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        // Use blocking read since this isn't async
        // In practice, you might want to cache this differently
        None
    }
}

// ============================================================================
// API Response Types
// ============================================================================

/// CryptoCompare news API response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
#[allow(dead_code)]
struct CryptoCompareNewsResponse {
    #[serde(rename = "Type")]
    type_: i32,
    #[serde(rename = "Message")]
    message: String,
    #[serde(rename = "Response")]
    response: String,
    #[serde(rename = "Data")]
    data: Vec<CryptoCompareNewsItem>,
}

/// Individual news item from CryptoCompare
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CryptoCompareNewsItem {
    id: String,
    guid: Option<String>,
    published_on: i64,
    imageurl: Option<String>,
    title: String,
    url: String,
    body: String,
    tags: Option<String>,
    categories: String,
    upvotes: i32,
    downvotes: i32,
    lang: String,
    source_info: Option<CryptoCompareSourceInfo>,
}

/// Source information for news item
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CryptoCompareSourceInfo {
    name: String,
    img: Option<String>,
    lang: Option<String>,
}

/// CryptoCompare social stats response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
#[allow(dead_code)]
struct CryptoCompareSocialResponse {
    #[serde(rename = "Response")]
    response: String,
    #[serde(rename = "Data")]
    data: CryptoCompareSocialStats,
}

/// Social stats for a coin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoCompareSocialStats {
    /// General info
    #[serde(rename = "General")]
    pub general: Option<CryptoCompareSocialGeneral>,
    /// Twitter stats
    #[serde(rename = "Twitter")]
    pub twitter: Option<CryptoCompareTwitterStats>,
    /// Reddit stats
    #[serde(rename = "Reddit")]
    pub reddit: Option<CryptoCompareRedditStats>,
    /// Code repository stats
    #[serde(rename = "CodeRepository")]
    pub code_repository: Option<CryptoCompareCodeStats>,
}

/// General social info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoCompareSocialGeneral {
    #[serde(rename = "Points")]
    pub points: Option<i64>,
    #[serde(rename = "Name")]
    pub name: Option<String>,
    #[serde(rename = "CoinName")]
    pub coin_name: Option<String>,
}

/// Twitter stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoCompareTwitterStats {
    pub followers: Option<i64>,
    pub statuses: Option<i64>,
    pub link: Option<String>,
}

/// Reddit stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoCompareRedditStats {
    pub subscribers: Option<i64>,
    pub active_users: Option<i64>,
    pub posts_per_day: Option<f64>,
    pub comments_per_day: Option<f64>,
    pub link: Option<String>,
}

/// Code repository stats (GitHub)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoCompareCodeStats {
    pub stars: Option<i64>,
    pub forks: Option<i64>,
    pub subscribers: Option<i64>,
    pub contributors: Option<i64>,
    pub link: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = CryptoCompareClient::new(None);
        assert!(client.is_ok());

        let client = client.unwrap();
        assert!(client.is_configured());
    }

    #[test]
    fn test_client_with_api_key() {
        let client = CryptoCompareClient::new(Some("test_key".to_string())).unwrap();
        assert!(client.is_configured());
        assert_eq!(client.config.api_key, Some("test_key".to_string()));
    }

    #[tokio::test]
    async fn test_calculate_sentiment_empty() {
        let client = CryptoCompareClient::new(None).unwrap();
        let sentiment = client.calculate_sentiment().await;

        assert_eq!(sentiment.score, 0.0);
        assert_eq!(sentiment.confidence, 0.0);
        assert!(sentiment.is_neutral());
    }
}
