//! Data Source Bridge Module
//!
//! This module provides the bridge that connects external data API clients
//! (news, weather, celestial) to the neuromorphic service bridge system.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    DATA SOURCE BRIDGE                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │  NewsAPI     │  │OpenWeatherMap│  │ SpaceWeather │          │
//! │  │  Client      │  │   Client     │  │   Client     │          │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
//! │         │                  │                  │                  │
//! │         └──────────────────┼──────────────────┘                  │
//! │                            │                                     │
//! │                   ┌────────▼────────┐                           │
//! │                   │  DataSource     │                           │
//! │                   │  Bridge         │                           │
//! │                   └────────┬────────┘                           │
//! │                            │                                     │
//! │         ┌──────────────────┼──────────────────┐                 │
//! │         │                  │                  │                 │
//! │    ┌────▼────┐       ┌─────▼────┐      ┌─────▼────┐           │
//! │    │ Forward │       │   Data   │      │   CNS    │           │
//! │    │ Bridge  │       │  Bridge  │      │  Bridge  │           │
//! │    └─────────┘       └──────────┘      └──────────┘           │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - Polls external data sources on configurable intervals
//! - Transforms external data into bridge messages
//! - Handles failures gracefully with circuit breaker pattern
//! - Reports health metrics to CNS
//! - Caches responses in Redis for deduplication

use super::cache::{CacheKey, RedisCache};
use super::config::ExternalDataConfig;
use super::{
    ApiClient, CelestialData, CryptoCompareClient, CryptoPanicClient, ExternalDataPoint,
    NewsApiClient, NewsSentiment, OpenWeatherMapClient, SpaceWeatherClient, WeatherData,
};
use crate::integration::service_bridges::{
    BridgeManager, HealthPayload, NeuromorphicSignal, RiskLevel, SignalType,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{RwLock, broadcast};
use tokio::time::{Instant, interval};
use tracing::{debug, error, info, warn};

/// Data source bridge error types
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Bridge not connected: {0}")]
    NotConnected(String),

    #[error("Data source error: {0}")]
    DataSourceError(String),

    #[error("Bridge message send failed: {0}")]
    SendFailed(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Circuit breaker open for source: {0}")]
    CircuitBreakerOpen(String),

    #[error("Cache error: {0}")]
    CacheError(String),
}

pub type Result<T> = std::result::Result<T, BridgeError>;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker for a data source
#[derive(Debug)]
pub struct CircuitBreaker {
    state: RwLock<CircuitState>,
    failure_count: AtomicU32,
    last_failure: RwLock<Option<Instant>>,
    threshold: u32,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU32::new(0),
            last_failure: RwLock::new(None),
            threshold,
            reset_timeout,
        }
    }

    /// Check if the circuit allows requests
    pub async fn allow_request(&self) -> bool {
        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if reset timeout has elapsed
                if let Some(last_failure) = *self.last_failure.read().await {
                    if last_failure.elapsed() >= self.reset_timeout {
                        // Transition to half-open
                        *self.state.write().await = CircuitState::HalfOpen;
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful request
    pub async fn record_success(&self) {
        let mut state = self.state.write().await;
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Closed;
        }
        self.failure_count.store(0, Ordering::SeqCst);
    }

    /// Record a failed request
    pub async fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        *self.last_failure.write().await = Some(Instant::now());

        if count >= self.threshold {
            let mut state = self.state.write().await;
            if *state != CircuitState::Open {
                warn!("Circuit breaker opened after {} failures", self.threshold);
                *state = CircuitState::Open;
            }
        }
    }

    /// Get current state
    pub async fn state(&self) -> CircuitState {
        *self.state.read().await
    }

    /// Reset the circuit breaker
    pub async fn reset(&self) {
        *self.state.write().await = CircuitState::Closed;
        self.failure_count.store(0, Ordering::SeqCst);
        *self.last_failure.write().await = None;
    }
}

/// External data event for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalDataEvent {
    /// Event ID
    pub id: String,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Source of the data
    pub source: String,

    /// The aggregated data point
    pub data: ExternalDataPoint,

    /// Processing latency in milliseconds
    pub latency_ms: u64,
}

/// Data source bridge statistics
#[derive(Debug, Default)]
pub struct BridgeStats {
    /// Total data fetches
    pub fetches: AtomicU64,

    /// Successful fetches
    pub successes: AtomicU64,

    /// Failed fetches
    pub failures: AtomicU64,

    /// Cache hits
    pub cache_hits: AtomicU64,

    /// Messages sent to bridges
    pub messages_sent: AtomicU64,

    /// Average latency (cumulative for averaging)
    pub total_latency_ms: AtomicU64,
}

impl BridgeStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_fetch(&self, success: bool, latency_ms: u64) {
        self.fetches.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successes.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_message_sent(&self) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
    }

    pub fn average_latency_ms(&self) -> f64 {
        let total = self.total_latency_ms.load(Ordering::Relaxed);
        let count = self.successes.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.fetches.load(Ordering::Relaxed);
        let success = self.successes.load(Ordering::Relaxed);
        if total == 0 {
            0.0 // No fetches means no success rate data
        } else {
            success as f64 / total as f64
        }
    }
}

/// Data source bridge - connects external data clients to service bridges
pub struct DataSourceBridge {
    /// Bridge configuration
    config: ExternalDataConfig,

    /// Service bridge manager
    bridge_manager: Arc<BridgeManager>,

    /// Redis cache
    cache: Option<Arc<RedisCache>>,

    /// News API client
    newsapi_client: Option<NewsApiClient>,

    /// CryptoPanic client
    cryptopanic_client: Option<CryptoPanicClient>,

    /// CryptoCompare client
    cryptocompare_client: Option<CryptoCompareClient>,

    /// OpenWeatherMap client
    weather_client: Option<OpenWeatherMapClient>,

    /// Space weather client
    spaceweather_client: Option<SpaceWeatherClient>,

    /// Circuit breakers for each source
    circuit_breakers: HashMap<String, CircuitBreaker>,

    /// Bridge statistics
    stats: Arc<BridgeStats>,

    /// Event broadcaster
    event_tx: broadcast::Sender<ExternalDataEvent>,

    /// Running flag
    running: AtomicBool,

    /// Last fetch time per source (reserved for rate limiting)
    #[allow(dead_code)]
    last_fetch: RwLock<HashMap<String, Instant>>,
}

impl DataSourceBridge {
    /// Create a new data source bridge
    pub fn new(config: ExternalDataConfig, bridge_manager: Arc<BridgeManager>) -> Self {
        let (event_tx, _) = broadcast::channel(1000);

        let threshold = config.aggregator.circuit_breaker_threshold;
        let reset_timeout = Duration::from_secs(config.aggregator.circuit_breaker_reset_timeout);

        let mut circuit_breakers = HashMap::new();
        circuit_breakers.insert(
            "newsapi".to_string(),
            CircuitBreaker::new(threshold, reset_timeout),
        );
        circuit_breakers.insert(
            "cryptopanic".to_string(),
            CircuitBreaker::new(threshold, reset_timeout),
        );
        circuit_breakers.insert(
            "cryptocompare".to_string(),
            CircuitBreaker::new(threshold, reset_timeout),
        );
        circuit_breakers.insert(
            "openweathermap".to_string(),
            CircuitBreaker::new(threshold, reset_timeout),
        );
        circuit_breakers.insert(
            "spaceweather".to_string(),
            CircuitBreaker::new(threshold, reset_timeout),
        );

        Self {
            config,
            bridge_manager,
            cache: None,
            newsapi_client: None,
            cryptopanic_client: None,
            cryptocompare_client: None,
            weather_client: None,
            spaceweather_client: None,
            circuit_breakers,
            stats: Arc::new(BridgeStats::new()),
            event_tx,
            running: AtomicBool::new(false),
            last_fetch: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize API clients based on configuration
    pub fn initialize_clients(&mut self) -> Result<()> {
        let api_keys = &self.config.api_keys;
        let _aggregator = &self.config.aggregator;

        // Initialize NewsAPI client
        if let Some(key) = &api_keys.newsapi {
            info!("Initializing NewsAPI client");
            match NewsApiClient::new(key.clone()) {
                Ok(client) => self.newsapi_client = Some(client),
                Err(e) => warn!("Failed to initialize NewsAPI client: {}", e),
            }
        }

        // Initialize CryptoPanic client
        if let Some(key) = &api_keys.cryptopanic {
            info!("Initializing CryptoPanic client");
            match CryptoPanicClient::new(key.clone()) {
                Ok(client) => self.cryptopanic_client = Some(client),
                Err(e) => warn!("Failed to initialize CryptoPanic client: {}", e),
            }
        }

        // Initialize CryptoCompare client
        if let Some(key) = &api_keys.cryptocompare {
            info!("Initializing CryptoCompare client");
            match CryptoCompareClient::new(Some(key.clone())) {
                Ok(client) => self.cryptocompare_client = Some(client),
                Err(e) => warn!("Failed to initialize CryptoCompare client: {}", e),
            }
        }

        // Initialize OpenWeatherMap client
        if let Some(key) = &api_keys.openweathermap {
            info!("Initializing OpenWeatherMap client");
            match OpenWeatherMapClient::new(key.clone()) {
                Ok(client) => self.weather_client = Some(client),
                Err(e) => warn!("Failed to initialize OpenWeatherMap client: {}", e),
            }
        }

        // Initialize Space Weather client (no API key required)
        info!("Initializing Space Weather client");
        self.spaceweather_client = Some(SpaceWeatherClient::default());

        Ok(())
    }

    /// Set the Redis cache
    pub fn with_cache(mut self, cache: Arc<RedisCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Subscribe to external data events
    pub fn subscribe(&self) -> broadcast::Receiver<ExternalDataEvent> {
        self.event_tx.subscribe()
    }

    /// Get bridge statistics
    pub fn stats(&self) -> Arc<BridgeStats> {
        Arc::clone(&self.stats)
    }

    /// Check if a source's circuit breaker allows requests
    async fn check_circuit_breaker(&self, source: &str) -> bool {
        if !self.config.aggregator.circuit_breaker_enabled {
            return true;
        }

        if let Some(cb) = self.circuit_breakers.get(source) {
            cb.allow_request().await
        } else {
            true
        }
    }

    /// Record circuit breaker result
    async fn record_circuit_result(&self, source: &str, success: bool) {
        if let Some(cb) = self.circuit_breakers.get(source) {
            if success {
                cb.record_success().await;
            } else {
                cb.record_failure().await;
            }
        }
    }

    /// Fetch news sentiment from configured sources
    async fn fetch_news_sentiment(&self) -> Option<NewsSentiment> {
        let mut sentiment: Option<NewsSentiment> = None;

        // Try NewsAPI first
        if let Some(client) = &self.newsapi_client {
            if self.check_circuit_breaker("newsapi").await {
                let start = Instant::now();
                let s = client.calculate_sentiment().await;
                let latency = start.elapsed().as_millis() as u64;
                self.stats.record_fetch(true, latency);
                self.record_circuit_result("newsapi", true).await;
                sentiment = Some(s);
                debug!("Fetched news sentiment from NewsAPI");
            }
        }

        // Try CryptoPanic as fallback
        if sentiment.is_none() {
            if let Some(client) = &self.cryptopanic_client {
                if self.check_circuit_breaker("cryptopanic").await {
                    let start = Instant::now();
                    // First fetch news to populate cache, then calculate sentiment
                    match client.fetch_news(None, None, None, None).await {
                        Ok(_) => {
                            let s = client.calculate_sentiment().await;
                            let latency = start.elapsed().as_millis() as u64;
                            self.stats.record_fetch(true, latency);
                            self.record_circuit_result("cryptopanic", true).await;
                            sentiment = Some(s);
                            debug!("Fetched news sentiment from CryptoPanic");
                        }
                        Err(e) => {
                            self.stats.record_fetch(false, 0);
                            self.record_circuit_result("cryptopanic", false).await;
                            warn!("Failed to fetch from CryptoPanic: {}", e);
                        }
                    }
                }
            }
        }

        // Try CryptoCompare as final fallback
        if sentiment.is_none() {
            if let Some(client) = &self.cryptocompare_client {
                if self.check_circuit_breaker("cryptocompare").await {
                    let start = Instant::now();
                    // First fetch news to populate cache, then calculate sentiment
                    match client.fetch_news(None).await {
                        Ok(_) => {
                            let s = client.calculate_sentiment().await;
                            let latency = start.elapsed().as_millis() as u64;
                            self.stats.record_fetch(true, latency);
                            self.record_circuit_result("cryptocompare", true).await;
                            sentiment = Some(s);
                            debug!("Fetched news sentiment from CryptoCompare");
                        }
                        Err(e) => {
                            self.stats.record_fetch(false, 0);
                            self.record_circuit_result("cryptocompare", false).await;
                            warn!("Failed to fetch from CryptoCompare: {}", e);
                        }
                    }
                }
            }
        }

        sentiment
    }

    /// Fetch weather data
    async fn fetch_weather(&self) -> Option<WeatherData> {
        if let Some(client) = &self.weather_client {
            if self.check_circuit_breaker("openweathermap").await {
                let start = Instant::now();
                // Fetch for default location (New York)
                match client.fetch_weather_by_city("New York").await {
                    Ok(w) => {
                        let latency = start.elapsed().as_millis() as u64;
                        self.stats.record_fetch(true, latency);
                        self.record_circuit_result("openweathermap", true).await;
                        debug!("Fetched weather data");
                        return Some(w);
                    }
                    Err(e) => {
                        self.stats.record_fetch(false, 0);
                        self.record_circuit_result("openweathermap", false).await;
                        warn!("Failed to fetch weather: {}", e);
                    }
                }
            }
        }
        None
    }

    /// Fetch celestial data
    async fn fetch_celestial(&self) -> Option<CelestialData> {
        if let Some(client) = &self.spaceweather_client {
            if self.check_circuit_breaker("spaceweather").await {
                let start = Instant::now();
                match client.fetch_celestial_data().await {
                    Ok(c) => {
                        let latency = start.elapsed().as_millis() as u64;
                        self.stats.record_fetch(true, latency);
                        self.record_circuit_result("spaceweather", true).await;
                        debug!("Fetched celestial data");
                        return Some(c);
                    }
                    Err(e) => {
                        self.stats.record_fetch(false, 0);
                        self.record_circuit_result("spaceweather", false).await;
                        warn!("Failed to fetch celestial data: {}", e);
                    }
                }
            }
        }
        None
    }

    /// Fetch all external data and aggregate
    pub async fn fetch_all(&self) -> ExternalDataPoint {
        let start = Instant::now();
        let mut data = ExternalDataPoint::new();

        // Fetch in parallel if configured
        if self.config.aggregator.parallel_fetch {
            let (news, weather, celestial) = tokio::join!(
                self.fetch_news_sentiment(),
                self.fetch_weather(),
                self.fetch_celestial()
            );

            if let Some(n) = news {
                data = data.with_news(n);
            }
            if let Some(w) = weather {
                data = data.with_weather(w);
            }
            if let Some(c) = celestial {
                data = data.with_celestial(c);
            }
        } else {
            // Sequential fetching
            if let Some(news) = self.fetch_news_sentiment().await {
                data = data.with_news(news);
            }
            if let Some(weather) = self.fetch_weather().await {
                data = data.with_weather(weather);
            }
            if let Some(celestial) = self.fetch_celestial().await {
                data = data.with_celestial(celestial);
            }
        }

        let latency = start.elapsed().as_millis() as u64;
        debug!("Aggregated external data in {}ms", latency);

        data
    }

    /// Send aggregated data to service bridges
    async fn send_to_bridges(&self, data: &ExternalDataPoint) -> Result<()> {
        // Send health update to CNS bridge
        let health = HealthPayload::healthy("thalamus-external-data")
            .with_metric(
                "news_available",
                if data.news.is_some() { 1.0 } else { 0.0 },
            )
            .with_metric(
                "weather_available",
                if data.weather.is_some() { 1.0 } else { 0.0 },
            )
            .with_metric(
                "celestial_available",
                if data.celestial.is_some() { 1.0 } else { 0.0 },
            )
            .with_metric("success_rate", self.stats.success_rate())
            .with_metric("avg_latency_ms", self.stats.average_latency_ms());

        let cns_bridge = self.bridge_manager.cns();
        if let Err(e) = cns_bridge.read().await.send_health_update(health).await {
            warn!("Failed to send health update to CNS: {}", e);
        } else {
            self.stats.record_message_sent();
        }

        // If we have news sentiment with high confidence, generate a signal
        if let Some(ref news) = data.news {
            if news.confidence > 0.7 {
                let signal = self.create_sentiment_signal(news);
                let forward_bridge = self.bridge_manager.forward();
                let send_result = forward_bridge.read().await.send_signal(signal).await;
                if let Err(e) = send_result {
                    warn!("Failed to send signal to forward bridge: {}", e);
                } else {
                    self.stats.record_message_sent();
                }
            }
        }

        Ok(())
    }

    /// Create a trading signal from sentiment data
    fn create_sentiment_signal(&self, sentiment: &NewsSentiment) -> NeuromorphicSignal {
        let signal_type = if sentiment.score > 0.3 {
            SignalType::Buy
        } else if sentiment.score < -0.3 {
            SignalType::Sell
        } else {
            SignalType::Hold
        };

        let risk_level = if sentiment.confidence > 0.8 {
            RiskLevel::Low
        } else if sentiment.confidence > 0.5 {
            RiskLevel::Medium
        } else {
            RiskLevel::High
        };

        NeuromorphicSignal::new(
            "BTC".to_string(), // Default symbol
            signal_type,
            sentiment.confidence,
        )
        .with_risk(risk_level, sentiment.confidence)
    }

    /// Broadcast external data event
    fn broadcast_event(&self, data: ExternalDataPoint, latency_ms: u64) {
        let event = ExternalDataEvent {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            source: "aggregator".to_string(),
            data,
            latency_ms,
        };

        if let Err(e) = self.event_tx.send(event) {
            debug!("No subscribers for external data event: {}", e);
        }
    }

    /// Start the data source bridge polling loop
    pub async fn start(&self) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(BridgeError::ConfigError(
                "Bridge already running".to_string(),
            ));
        }

        self.running.store(true, Ordering::SeqCst);
        info!("Starting data source bridge");

        // Connect the bridge manager
        if let Err(e) = self.bridge_manager.connect_all().await {
            warn!("Failed to connect all bridges: {}", e);
        }

        let poll_interval = self.config.aggregator.poll_duration();
        let mut ticker = interval(poll_interval);

        while self.running.load(Ordering::SeqCst) {
            ticker.tick().await;

            let start = Instant::now();

            // Fetch and aggregate data
            let data = self.fetch_all().await;

            // Send to bridges
            if let Err(e) = self.send_to_bridges(&data).await {
                error!("Failed to send data to bridges: {}", e);
            }

            // Cache the aggregated data
            if let Some(ref cache) = self.cache {
                let cache_key = CacheKey::aggregated("latest");
                if let Err(e) = cache.set(cache_key, &data, None).await {
                    warn!("Failed to cache aggregated data: {}", e);
                }
            }

            // Broadcast event
            let latency = start.elapsed().as_millis() as u64;
            self.broadcast_event(data, latency);
        }

        info!("Data source bridge stopped");
        Ok(())
    }

    /// Stop the data source bridge
    pub fn stop(&self) {
        info!("Stopping data source bridge");
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the bridge is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the latest cached data
    pub async fn get_cached_data(&self) -> Option<ExternalDataPoint> {
        if let Some(ref cache) = self.cache {
            match cache
                .get::<ExternalDataPoint>(CacheKey::aggregated("latest"))
                .await
            {
                Ok(data) => data,
                Err(e) => {
                    warn!("Failed to get cached data: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }

    /// Health check for all components
    pub async fn health_check(&self) -> HashMap<String, bool> {
        let mut status = HashMap::new();

        // Check API clients
        if let Some(ref client) = self.newsapi_client {
            status.insert("newsapi".to_string(), client.health_check().await);
        }
        if let Some(ref client) = self.cryptopanic_client {
            status.insert("cryptopanic".to_string(), client.health_check().await);
        }
        if let Some(ref client) = self.cryptocompare_client {
            status.insert("cryptocompare".to_string(), client.health_check().await);
        }
        if let Some(ref client) = self.weather_client {
            status.insert("openweathermap".to_string(), client.health_check().await);
        }
        if let Some(ref client) = self.spaceweather_client {
            status.insert("spaceweather".to_string(), client.health_check().await);
        }

        // Check cache
        if let Some(ref cache) = self.cache {
            status.insert("cache".to_string(), cache.health_check().await);
        }

        // Check circuit breakers
        for (name, cb) in &self.circuit_breakers {
            let cb_healthy = cb.state().await != CircuitState::Open;
            status.insert(format!("{}_circuit", name), cb_healthy);
        }

        // Check bridge manager
        let bridge_health = self.bridge_manager.health_check_all().await;
        for (name, healthy) in bridge_health {
            status.insert(format!("bridge_{}", name), healthy);
        }

        status
    }

    /// Reset all circuit breakers
    pub async fn reset_circuit_breakers(&self) {
        for (name, cb) in &self.circuit_breakers {
            cb.reset().await;
            info!("Reset circuit breaker for {}", name);
        }
    }

    /// Get circuit breaker states
    pub async fn circuit_breaker_states(&self) -> HashMap<String, CircuitState> {
        let mut states = HashMap::new();
        for (name, cb) in &self.circuit_breakers {
            states.insert(name.clone(), cb.state().await);
        }
        states
    }
}

/// Builder for DataSourceBridge
pub struct DataSourceBridgeBuilder {
    config: Option<ExternalDataConfig>,
    bridge_manager: Option<Arc<BridgeManager>>,
    cache: Option<Arc<RedisCache>>,
}

impl DataSourceBridgeBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: None,
            bridge_manager: None,
            cache: None,
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: ExternalDataConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the bridge manager
    pub fn bridge_manager(mut self, bridge_manager: Arc<BridgeManager>) -> Self {
        self.bridge_manager = Some(bridge_manager);
        self
    }

    /// Set the cache
    pub fn cache(mut self, cache: Arc<RedisCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Build the data source bridge
    pub fn build(self) -> Result<DataSourceBridge> {
        let config = self
            .config
            .ok_or_else(|| BridgeError::ConfigError("Configuration required".to_string()))?;

        let bridge_manager = self
            .bridge_manager
            .ok_or_else(|| BridgeError::ConfigError("Bridge manager required".to_string()))?;

        let mut bridge = DataSourceBridge::new(config, bridge_manager);

        if let Some(cache) = self.cache {
            bridge = bridge.with_cache(cache);
        }

        bridge.initialize_clients()?;

        Ok(bridge)
    }
}

impl Default for DataSourceBridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));
        assert!(cb.allow_request().await);
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));

        // Record failures up to threshold
        for _ in 0..3 {
            cb.record_failure().await;
        }

        assert_eq!(cb.state().await, CircuitState::Open);
        assert!(!cb.allow_request().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reset() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));

        for _ in 0..3 {
            cb.record_failure().await;
        }

        assert_eq!(cb.state().await, CircuitState::Open);

        cb.reset().await;

        assert_eq!(cb.state().await, CircuitState::Closed);
        assert!(cb.allow_request().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_success_resets_count() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));

        cb.record_failure().await;
        cb.record_failure().await;
        cb.record_success().await;
        cb.record_failure().await;
        cb.record_failure().await;

        // Should still be closed because success reset the count
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[test]
    fn test_bridge_stats() {
        let stats = BridgeStats::new();

        stats.record_fetch(true, 100);
        stats.record_fetch(true, 200);
        stats.record_fetch(false, 0);

        assert_eq!(stats.fetches.load(Ordering::Relaxed), 3);
        assert_eq!(stats.successes.load(Ordering::Relaxed), 2);
        assert_eq!(stats.failures.load(Ordering::Relaxed), 1);

        assert!((stats.success_rate() - 0.666).abs() < 0.01);
        assert!((stats.average_latency_ms() - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_external_data_event() {
        let data = ExternalDataPoint::new();
        let event = ExternalDataEvent {
            id: "test-id".to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            data,
            latency_ms: 100,
        };

        assert_eq!(event.source, "test");
        assert_eq!(event.latency_ms, 100);
    }
}
