//! NewsAPI.org Client
//!
//! Provides access to news from thousands of sources via NewsAPI.org.
//! API Documentation: https://newsapi.org/docs
//!
//! ## Features
//!
//! - Top headlines from various categories
//! - Everything endpoint for historical news search
//! - Source filtering
//! - Keyword search
//!
//! ## Rate Limits
//!
//! - Free tier: 100 requests/day
//! - Developer: 500 requests/day
//! - Business: Unlimited

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use tracing::{debug, error, warn};

use super::{ApiClient, ApiClientConfig, RateLimiter};
use crate::common::{Error, Result};
use crate::thalamus::sources::DataSource;
use crate::thalamus::sources::news::{NewsArticle, NewsSentiment};

/// NewsAPI base URL
const NEWSAPI_BASE_URL: &str = "https://newsapi.org/v2";

/// News category for top headlines
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewsCategory {
    Business,
    Entertainment,
    General,
    Health,
    Science,
    Sports,
    Technology,
}

impl NewsCategory {
    fn as_str(&self) -> &str {
        match self {
            NewsCategory::Business => "business",
            NewsCategory::Entertainment => "entertainment",
            NewsCategory::General => "general",
            NewsCategory::Health => "health",
            NewsCategory::Science => "science",
            NewsCategory::Sports => "sports",
            NewsCategory::Technology => "technology",
        }
    }
}

/// Sort order for search results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    /// Most recent first
    PublishedAt,
    /// Most relevant first
    Relevancy,
    /// Most popular first
    Popularity,
}

impl SortBy {
    fn as_str(&self) -> &str {
        match self {
            SortBy::PublishedAt => "publishedAt",
            SortBy::Relevancy => "relevancy",
            SortBy::Popularity => "popularity",
        }
    }
}

/// Country code for news filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Country {
    US,
    GB,
    CA,
    AU,
    DE,
    FR,
    JP,
    CN,
    IN,
    BR,
}

impl Country {
    fn as_str(&self) -> &str {
        match self {
            Country::US => "us",
            Country::GB => "gb",
            Country::CA => "ca",
            Country::AU => "au",
            Country::DE => "de",
            Country::FR => "fr",
            Country::JP => "jp",
            Country::CN => "cn",
            Country::IN => "in",
            Country::BR => "br",
        }
    }
}

/// Language for news filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
    Italian,
    Portuguese,
    Russian,
    Chinese,
    Japanese,
    Arabic,
}

impl Language {
    fn as_str(&self) -> &str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Russian => "ru",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Arabic => "ar",
        }
    }
}

/// NewsAPI.org client
pub struct NewsApiClient {
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

impl NewsApiClient {
    /// Create a new NewsAPI client
    /// Note: API key is required for NewsAPI
    pub fn new(api_key: String) -> Result<Self> {
        let config = ApiClientConfig {
            api_key: Some(api_key),
            base_url: NEWSAPI_BASE_URL.to_string(),
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
            // NewsAPI free tier: ~1 request per 14 minutes (100/day)
            // Be conservative with rate limiting
            rate_limiter: RateLimiter::new(0.1), // 1 request per 10 seconds
            cached_articles: tokio::sync::RwLock::new(Vec::new()),
            last_update: tokio::sync::RwLock::new(None),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: ApiClientConfig) -> Result<Self> {
        if config.api_key.is_none() {
            return Err(Error::Other("NewsAPI requires an API key".to_string()));
        }

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            rate_limiter: RateLimiter::new(0.1),
            cached_articles: tokio::sync::RwLock::new(Vec::new()),
            last_update: tokio::sync::RwLock::new(None),
        })
    }

    /// Fetch top headlines
    pub async fn fetch_top_headlines(
        &self,
        category: Option<NewsCategory>,
        country: Option<Country>,
        query: Option<&str>,
        page_size: Option<u32>,
    ) -> Result<Vec<NewsArticle>> {
        self.rate_limiter.wait().await;

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Other("API key required".to_string()))?;

        let mut url = format!("{}/top-headlines?apiKey={}", self.config.base_url, api_key);

        // Add category filter
        if let Some(cat) = category {
            url.push_str(&format!("&category={}", cat.as_str()));
        }

        // Add country filter
        if let Some(c) = country {
            url.push_str(&format!("&country={}", c.as_str()));
        }

        // Add query
        if let Some(q) = query {
            url.push_str(&format!("&q={}", urlencoding::encode(q)));
        }

        // Add page size (max 100)
        let size = page_size.unwrap_or(50).min(100);
        url.push_str(&format!("&pageSize={}", size));

        debug!("Fetching NewsAPI top headlines");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("NewsAPI error: {} - {}", status, text);
            return Err(Error::Other(format!("API error: {} - {}", status, text)));
        }

        let api_response: NewsApiResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        if api_response.status != "ok" {
            let msg = api_response
                .message
                .unwrap_or_else(|| "Unknown error".to_string());
            warn!("NewsAPI returned error: {}", msg);
            return Err(Error::Other(format!("API error: {}", msg)));
        }

        // Convert to our NewsArticle format
        let articles: Vec<NewsArticle> = api_response
            .articles
            .unwrap_or_default()
            .into_iter()
            .map(|item| self.convert_article(item))
            .collect();

        // Update cache
        self.update_cache(articles.clone()).await;

        debug!("Fetched {} articles from NewsAPI", articles.len());
        Ok(articles)
    }

    /// Search for news articles
    pub async fn search_everything(
        &self,
        query: &str,
        language: Option<Language>,
        sort_by: Option<SortBy>,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        page_size: Option<u32>,
    ) -> Result<Vec<NewsArticle>> {
        self.rate_limiter.wait().await;

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Other("API key required".to_string()))?;

        let mut url = format!(
            "{}/everything?apiKey={}&q={}",
            self.config.base_url,
            api_key,
            urlencoding::encode(query)
        );

        // Add language filter
        if let Some(lang) = language {
            url.push_str(&format!("&language={}", lang.as_str()));
        }

        // Add sort order
        if let Some(sort) = sort_by {
            url.push_str(&format!("&sortBy={}", sort.as_str()));
        }

        // Add date range
        if let Some(from_date) = from {
            url.push_str(&format!("&from={}", from_date.format("%Y-%m-%d")));
        }
        if let Some(to_date) = to {
            url.push_str(&format!("&to={}", to_date.format("%Y-%m-%d")));
        }

        // Add page size (max 100)
        let size = page_size.unwrap_or(50).min(100);
        url.push_str(&format!("&pageSize={}", size));

        debug!("Searching NewsAPI for: {}", query);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("NewsAPI error: {} - {}", status, text);
            return Err(Error::Other(format!("API error: {} - {}", status, text)));
        }

        let api_response: NewsApiResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        if api_response.status != "ok" {
            let msg = api_response
                .message
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(Error::Other(format!("API error: {}", msg)));
        }

        let articles: Vec<NewsArticle> = api_response
            .articles
            .unwrap_or_default()
            .into_iter()
            .map(|item| self.convert_article(item))
            .collect();

        // Update cache
        self.update_cache(articles.clone()).await;

        debug!("Found {} articles from NewsAPI search", articles.len());
        Ok(articles)
    }

    /// Fetch crypto/finance news
    pub async fn fetch_crypto_news(&self) -> Result<Vec<NewsArticle>> {
        self.search_everything(
            "cryptocurrency OR bitcoin OR ethereum OR crypto trading",
            Some(Language::English),
            Some(SortBy::PublishedAt),
            None,
            None,
            Some(50),
        )
        .await
    }

    /// Fetch financial market news
    pub async fn fetch_finance_news(&self) -> Result<Vec<NewsArticle>> {
        self.fetch_top_headlines(
            Some(NewsCategory::Business),
            Some(Country::US),
            Some("market OR stocks OR trading OR finance"),
            Some(50),
        )
        .await
    }

    /// Convert NewsAPI article to our format
    fn convert_article(&self, item: NewsApiArticle) -> NewsArticle {
        // Parse timestamp
        let published_at = item
            .published_at
            .as_ref()
            .and_then(|ts| DateTime::parse_from_rfc3339(ts).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        // Generate unique ID from URL
        let id = item
            .url
            .as_ref()
            .map(|u| {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                u.hash(&mut hasher);
                format!("newsapi_{}", hasher.finish())
            })
            .unwrap_or_else(|| format!("newsapi_{}", uuid::Uuid::new_v4()));

        // Extract source name
        let source = item
            .source
            .map(|s| s.name.unwrap_or_else(|| "Unknown".to_string()))
            .unwrap_or_else(|| "NewsAPI".to_string());

        // Simple sentiment analysis based on title keywords
        let sentiment = self.analyze_title_sentiment(item.title.as_deref().unwrap_or(""));

        // Extract potential symbols from title and description
        let symbols = self.extract_symbols(
            item.title.as_deref().unwrap_or(""),
            item.description.as_deref().unwrap_or(""),
        );

        NewsArticle {
            id,
            title: item.title.unwrap_or_else(|| "No title".to_string()),
            summary: item.description,
            content: item.content,
            source,
            url: item.url.unwrap_or_default(),
            published_at,
            symbols,
            sentiment,
            categories: vec!["financial_news".to_string()],
            author: item.author,
        }
    }

    /// Simple sentiment analysis based on title keywords
    fn analyze_title_sentiment(&self, title: &str) -> Option<f64> {
        let title_lower = title.to_lowercase();

        // Positive keywords
        let positive_words = [
            "surge",
            "soar",
            "rally",
            "gain",
            "rise",
            "bull",
            "bullish",
            "up",
            "growth",
            "profit",
            "positive",
            "optimistic",
            "breakthrough",
            "success",
            "record high",
            "all-time high",
            "ath",
            "moon",
            "pump",
            "green",
            "boost",
            "recover",
        ];

        // Negative keywords
        let negative_words = [
            "crash",
            "plunge",
            "drop",
            "fall",
            "bear",
            "bearish",
            "down",
            "loss",
            "negative",
            "pessimistic",
            "fear",
            "panic",
            "sell-off",
            "dump",
            "red",
            "decline",
            "slump",
            "tumble",
            "sink",
            "collapse",
            "crisis",
        ];

        let positive_count = positive_words
            .iter()
            .filter(|w| title_lower.contains(*w))
            .count();
        let negative_count = negative_words
            .iter()
            .filter(|w| title_lower.contains(*w))
            .count();

        if positive_count == 0 && negative_count == 0 {
            return None;
        }

        let total = (positive_count + negative_count) as f64;
        Some((positive_count as f64 - negative_count as f64) / total)
    }

    /// Extract potential crypto/stock symbols from text
    fn extract_symbols(&self, title: &str, description: &str) -> Vec<String> {
        let mut symbols = Vec::new();
        let text = format!("{} {}", title, description).to_uppercase();

        // Common crypto symbols to look for
        let crypto_symbols = [
            "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "MATIC", "LINK", "AVAX", "ATOM",
            "UNI", "LTC", "BCH", "ALGO", "NEAR", "FTM", "SAND", "MANA", "APE", "SHIB", "CRO",
        ];

        // Check for each symbol with word boundaries
        for symbol in crypto_symbols {
            // Check for exact match or with common suffixes
            if text.contains(&format!(" {} ", symbol))
                || text.contains(&format!("${}", symbol))
                || text.starts_with(&format!("{} ", symbol))
                || text.ends_with(&format!(" {}", symbol))
            {
                symbols.push(symbol.to_string());
            }
        }

        // Also look for "Bitcoin", "Ethereum" etc.
        let name_to_symbol = [
            ("BITCOIN", "BTC"),
            ("ETHEREUM", "ETH"),
            ("SOLANA", "SOL"),
            ("RIPPLE", "XRP"),
            ("CARDANO", "ADA"),
            ("DOGECOIN", "DOGE"),
            ("POLKADOT", "DOT"),
            ("POLYGON", "MATIC"),
            ("CHAINLINK", "LINK"),
            ("AVALANCHE", "AVAX"),
        ];

        for (name, symbol) in name_to_symbol {
            if text.contains(name) && !symbols.contains(&symbol.to_string()) {
                symbols.push(symbol.to_string());
            }
        }

        symbols
    }

    /// Update cache with new articles
    async fn update_cache(&self, articles: Vec<NewsArticle>) {
        let mut cache = self.cached_articles.write().await;
        *cache = articles;
        let mut last = self.last_update.write().await;
        *last = Some(Utc::now());
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
impl ApiClient for NewsApiClient {
    fn name(&self) -> &str {
        "newsapi"
    }

    fn is_configured(&self) -> bool {
        self.config.api_key.is_some()
    }

    async fn health_check(&self) -> bool {
        if !self.is_configured() {
            return false;
        }

        // Try fetching a small amount of data
        match self
            .fetch_top_headlines(
                Some(NewsCategory::Business),
                Some(Country::US),
                None,
                Some(1),
            )
            .await
        {
            Ok(_) => true,
            Err(e) => {
                warn!("NewsAPI health check failed: {}", e);
                false
            }
        }
    }
}

#[async_trait]
impl DataSource for NewsApiClient {
    type Data = NewsSentiment;

    fn name(&self) -> &str {
        "newsapi"
    }

    async fn fetch_latest(&self) -> Result<Self::Data> {
        // Fetch financial news
        self.fetch_finance_news().await?;
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

/// NewsAPI response structure
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NewsApiResponse {
    /// Response status ("ok" or "error")
    status: String,
    /// Total results count
    #[serde(rename = "totalResults")]
    total_results: Option<i32>,
    /// Error code (if error)
    code: Option<String>,
    /// Error message (if error)
    message: Option<String>,
    /// Articles list
    articles: Option<Vec<NewsApiArticle>>,
}

/// Individual article from NewsAPI
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NewsApiArticle {
    /// Source information
    source: Option<NewsApiSource>,
    /// Article author
    author: Option<String>,
    /// Article title
    title: Option<String>,
    /// Article description/summary
    description: Option<String>,
    /// Article URL
    url: Option<String>,
    /// Image URL
    #[serde(rename = "urlToImage")]
    url_to_image: Option<String>,
    /// Published timestamp (ISO 8601)
    #[serde(rename = "publishedAt")]
    published_at: Option<String>,
    /// Article content (truncated in free tier)
    content: Option<String>,
}

/// Source information
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NewsApiSource {
    /// Source ID
    id: Option<String>,
    /// Source name
    name: Option<String>,
}

/// URL encoding helper
mod urlencoding {
    pub fn encode(s: &str) -> String {
        url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_requires_api_key() {
        let config = ApiClientConfig {
            api_key: None,
            base_url: NEWSAPI_BASE_URL.to_string(),
            ..Default::default()
        };

        let result = NewsApiClient::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_client_creation_with_key() {
        let client = NewsApiClient::new("test_api_key".to_string());
        assert!(client.is_ok());

        let client = client.unwrap();
        assert!(client.is_configured());
    }

    #[test]
    fn test_news_category() {
        assert_eq!(NewsCategory::Business.as_str(), "business");
        assert_eq!(NewsCategory::Technology.as_str(), "technology");
    }

    #[test]
    fn test_sort_by() {
        assert_eq!(SortBy::PublishedAt.as_str(), "publishedAt");
        assert_eq!(SortBy::Relevancy.as_str(), "relevancy");
    }

    #[test]
    fn test_country() {
        assert_eq!(Country::US.as_str(), "us");
        assert_eq!(Country::GB.as_str(), "gb");
    }

    #[test]
    fn test_sentiment_analysis() {
        let client = NewsApiClient::new("test_key".to_string()).unwrap();

        // Positive sentiment
        let sentiment = client.analyze_title_sentiment("Bitcoin surges to new record high");
        assert!(sentiment.is_some());
        assert!(sentiment.unwrap() > 0.0);

        // Negative sentiment
        let sentiment = client.analyze_title_sentiment("Crypto market crashes amid panic selling");
        assert!(sentiment.is_some());
        assert!(sentiment.unwrap() < 0.0);

        // Neutral
        let sentiment = client.analyze_title_sentiment("SEC announces new crypto regulations");
        assert!(sentiment.is_none());
    }

    #[test]
    fn test_symbol_extraction() {
        let client = NewsApiClient::new("test_key".to_string()).unwrap();

        let symbols = client.extract_symbols(
            "Bitcoin and Ethereum lead crypto rally",
            "BTC price surges as ETH follows",
        );

        assert!(symbols.contains(&"BTC".to_string()));
        assert!(symbols.contains(&"ETH".to_string()));
    }

    #[tokio::test]
    async fn test_calculate_sentiment_empty() {
        let client = NewsApiClient::new("test_key".to_string()).unwrap();
        let sentiment = client.calculate_sentiment().await;

        assert_eq!(sentiment.score, 0.0);
        assert_eq!(sentiment.confidence, 0.0);
        assert!(sentiment.is_neutral());
    }
}
