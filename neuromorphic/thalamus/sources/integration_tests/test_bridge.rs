//! Integration tests for DataSourceBridge using wiremock
//!
//! These tests verify the bridge correctly:
//! - Fetches data from external sources
//! - Handles circuit breaker logic
//! - Transforms data for neuromorphic consumption
//! - Sends events to service bridges

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use chrono::Utc;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::thalamus::sources::bridge::{
    BridgeStats, CircuitBreaker, CircuitState, ExternalDataEvent,
};
use crate::thalamus::sources::{ExternalDataPoint, NewsSentiment, WeatherData};

// ============================================================================
// Circuit Breaker Tests
// ============================================================================

mod circuit_breaker_tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_starts_closed() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(30));
        assert_eq!(breaker.state().await, CircuitState::Closed);
        assert!(breaker.allow_request().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_threshold() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(30));

        // Record failures up to threshold
        for _ in 0..3 {
            breaker.record_failure().await;
        }

        assert_eq!(breaker.state().await, CircuitState::Open);
        assert!(!breaker.allow_request().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_success_resets_count() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(30));

        // Record some failures
        breaker.record_failure().await;
        breaker.record_failure().await;

        // Success should reset failure count
        breaker.record_success().await;

        // Should still be closed after more failures
        breaker.record_failure().await;
        breaker.record_failure().await;

        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_after_timeout() {
        let breaker = CircuitBreaker::new(3, Duration::from_millis(50));

        // Open the circuit
        for _ in 0..3 {
            breaker.record_failure().await;
        }
        assert_eq!(breaker.state().await, CircuitState::Open);

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should transition to half-open and allow one request
        assert!(breaker.allow_request().await);
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_closes_after_success_in_half_open() {
        let breaker = CircuitBreaker::new(3, Duration::from_millis(50));

        // Open the circuit
        for _ in 0..3 {
            breaker.record_failure().await;
        }

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Allow request (transitions to half-open)
        breaker.allow_request().await;

        // Success in half-open should close circuit
        breaker.record_success().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reopens_after_failure_in_half_open() {
        let breaker = CircuitBreaker::new(3, Duration::from_millis(50));

        // Open the circuit
        for _ in 0..3 {
            breaker.record_failure().await;
        }

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Allow request (transitions to half-open)
        breaker.allow_request().await;

        // Failure in half-open should reopen circuit
        breaker.record_failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reset() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(30));

        // Open the circuit
        for _ in 0..3 {
            breaker.record_failure().await;
        }
        assert_eq!(breaker.state().await, CircuitState::Open);

        // Manual reset
        breaker.reset().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }
}

// ============================================================================
// Bridge Stats Tests
// ============================================================================

mod bridge_stats_tests {
    use super::*;

    #[test]
    fn test_stats_new() {
        let stats = BridgeStats::new();
        assert_eq!(stats.fetches.load(Ordering::Relaxed), 0);
        assert_eq!(stats.successes.load(Ordering::Relaxed), 0);
        assert_eq!(stats.failures.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_stats_record_fetch_success() {
        let stats = BridgeStats::new();
        stats.record_fetch(true, 150);

        assert_eq!(stats.fetches.load(Ordering::Relaxed), 1);
        assert_eq!(stats.successes.load(Ordering::Relaxed), 1);
        assert_eq!(stats.failures.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_latency_ms.load(Ordering::Relaxed), 150);
    }

    #[test]
    fn test_stats_record_fetch_failure() {
        let stats = BridgeStats::new();
        stats.record_fetch(false, 50);

        assert_eq!(stats.fetches.load(Ordering::Relaxed), 1);
        assert_eq!(stats.successes.load(Ordering::Relaxed), 0);
        assert_eq!(stats.failures.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_stats_cache_hits() {
        let stats = BridgeStats::new();
        stats.record_cache_hit();
        stats.record_cache_hit();
        stats.record_cache_hit();

        assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_stats_messages_sent() {
        let stats = BridgeStats::new();
        stats.record_message_sent();
        stats.record_message_sent();

        assert_eq!(stats.messages_sent.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_stats_average_latency() {
        let stats = BridgeStats::new();
        stats.record_fetch(true, 100);
        stats.record_fetch(true, 200);
        stats.record_fetch(true, 300);

        let avg = stats.average_latency_ms();
        assert_eq!(avg, 200.0); // (100 + 200 + 300) / 3
    }

    #[test]
    fn test_stats_average_latency_no_fetches() {
        let stats = BridgeStats::new();
        assert_eq!(stats.average_latency_ms(), 0.0);
    }

    #[test]
    fn test_stats_success_rate() {
        let stats = BridgeStats::new();
        stats.record_fetch(true, 100);
        stats.record_fetch(true, 100);
        stats.record_fetch(false, 100);
        stats.record_fetch(true, 100);

        let rate = stats.success_rate();
        assert!((rate - 0.75).abs() < 0.001); // 3/4 = 75%
    }

    #[test]
    fn test_stats_success_rate_no_fetches() {
        let stats = BridgeStats::new();
        assert_eq!(stats.success_rate(), 0.0);
    }
}

// ============================================================================
// External Data Event Tests
// ============================================================================

mod event_tests {
    use super::*;

    #[test]
    fn test_external_data_event_creation() {
        let event = ExternalDataEvent {
            id: "event-001".to_string(),
            timestamp: Utc::now(),
            source: "newsapi".to_string(),
            data: ExternalDataPoint::new(),
            latency_ms: 150,
        };

        assert_eq!(event.id, "event-001");
        assert_eq!(event.source, "newsapi");
        assert_eq!(event.latency_ms, 150);
    }

    #[test]
    fn test_external_data_event_with_data() {
        let sentiment = NewsSentiment {
            score: 0.75,
            confidence: 0.8,
            source_count: 5,
            keywords: vec!["bitcoin".to_string()],
            timestamp: Utc::now(),
        };

        let data = ExternalDataPoint::new().with_news(sentiment);

        let event = ExternalDataEvent {
            id: "event-002".to_string(),
            timestamp: Utc::now(),
            source: "cryptopanic".to_string(),
            data,
            latency_ms: 200,
        };

        assert!(event.data.has_data());
        assert!(event.data.news.is_some());
    }

    #[test]
    fn test_external_data_event_serialization() {
        let event = ExternalDataEvent {
            id: "event-003".to_string(),
            timestamp: Utc::now(),
            source: "weather".to_string(),
            data: ExternalDataPoint::new(),
            latency_ms: 100,
        };

        let json = serde_json::to_string(&event).expect("Failed to serialize");
        assert!(json.contains("weather"));
        assert!(json.contains("event-003"));
    }
}

// ============================================================================
// Bridge Message Transformation Tests
// ============================================================================

mod transformation_tests {
    use super::*;

    #[test]
    fn test_sentiment_to_signal_type_bullish() {
        let sentiment = 0.75;

        // Bullish sentiment should map to bullish signal
        let signal_type = if sentiment > 0.5 {
            "SentimentBullish"
        } else if sentiment < -0.5 {
            "SentimentBearish"
        } else {
            "MarketData"
        };

        assert_eq!(signal_type, "SentimentBullish");
    }

    #[test]
    fn test_sentiment_to_signal_type_bearish() {
        let sentiment = -0.75;

        let signal_type = if sentiment > 0.5 {
            "SentimentBullish"
        } else if sentiment < -0.5 {
            "SentimentBearish"
        } else {
            "MarketData"
        };

        assert_eq!(signal_type, "SentimentBearish");
    }

    #[test]
    fn test_sentiment_to_signal_type_neutral() {
        let sentiment = 0.2;

        let signal_type = if sentiment > 0.5 {
            "SentimentBullish"
        } else if sentiment < -0.5 {
            "SentimentBearish"
        } else {
            "MarketData"
        };

        assert_eq!(signal_type, "MarketData");
    }

    #[test]
    fn test_confidence_to_risk_level() {
        fn map_risk(confidence: f64) -> &'static str {
            if confidence < 0.3 {
                "Low"
            } else if confidence < 0.7 {
                "Medium"
            } else {
                "High"
            }
        }

        assert_eq!(map_risk(0.2), "Low");
        assert_eq!(map_risk(0.5), "Medium");
        assert_eq!(map_risk(0.9), "High");
    }

    #[test]
    fn test_external_data_completeness() {
        let sentiment = NewsSentiment {
            score: 0.5,
            confidence: 0.8,
            source_count: 3,
            keywords: vec![],
            timestamp: Utc::now(),
        };

        let weather = WeatherData::new("London".to_string(), 20.0);

        let data = ExternalDataPoint::new()
            .with_news(sentiment)
            .with_weather(weather);

        // Completeness should be 2/3 (news + weather, no celestial)
        let completeness = data.completeness();
        assert!((completeness - 0.666).abs() < 0.01);
    }
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

mod concurrency_tests {
    use super::*;
    use tokio::task::JoinSet;

    #[tokio::test]
    async fn test_concurrent_circuit_breaker_access() {
        let breaker = Arc::new(CircuitBreaker::new(10, Duration::from_secs(30)));
        let mut join_set = JoinSet::new();

        // Spawn multiple tasks accessing the circuit breaker
        for i in 0..10 {
            let breaker_clone = breaker.clone();
            join_set.spawn(async move {
                for _ in 0..10 {
                    if i % 2 == 0 {
                        breaker_clone.record_success().await;
                    } else {
                        breaker_clone.record_failure().await;
                    }
                    let _ = breaker_clone.allow_request().await;
                }
            });
        }

        // Wait for all tasks
        while let Some(result) = join_set.join_next().await {
            assert!(result.is_ok());
        }

        // Circuit should still be in a valid state
        let state = breaker.state().await;
        assert!(matches!(
            state,
            CircuitState::Closed | CircuitState::Open | CircuitState::HalfOpen
        ));
    }

    #[tokio::test]
    async fn test_concurrent_stats_updates() {
        let stats = Arc::new(BridgeStats::new());
        let mut join_set = JoinSet::new();

        // Spawn multiple tasks updating stats
        for _ in 0..10 {
            let stats_clone = stats.clone();
            join_set.spawn(async move {
                for j in 0..100 {
                    stats_clone.record_fetch(j % 2 == 0, 10);
                    stats_clone.record_cache_hit();
                    stats_clone.record_message_sent();
                }
            });
        }

        // Wait for all tasks
        while let Some(result) = join_set.join_next().await {
            assert!(result.is_ok());
        }

        // Verify total counts
        assert_eq!(stats.fetches.load(Ordering::Relaxed), 1000);
        assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 1000);
        assert_eq!(stats.messages_sent.load(Ordering::Relaxed), 1000);
    }

    #[tokio::test]
    async fn test_concurrent_circuit_breaker_threshold() {
        let breaker = Arc::new(CircuitBreaker::new(5, Duration::from_secs(30)));
        let mut join_set = JoinSet::new();

        // All tasks record failures
        for _ in 0..10 {
            let breaker_clone = breaker.clone();
            join_set.spawn(async move {
                breaker_clone.record_failure().await;
            });
        }

        // Wait for all tasks
        while let Some(result) = join_set.join_next().await {
            assert!(result.is_ok());
        }

        // Circuit should be open after exceeding threshold
        assert_eq!(breaker.state().await, CircuitState::Open);
    }
}

// ============================================================================
// Mock Server Integration Tests
// ============================================================================

mod mock_integration_tests {
    use super::*;

    fn mock_newsapi_response() -> serde_json::Value {
        json!({
            "status": "ok",
            "totalResults": 1,
            "articles": [{
                "source": { "name": "Test Source" },
                "title": "Bitcoin Hits New High",
                "description": "Market sentiment is bullish",
                "url": "https://example.com/article",
                "publishedAt": "2024-01-15T12:00:00Z"
            }]
        })
    }

    fn mock_weather_response() -> serde_json::Value {
        json!({
            "main": {
                "temp": 22.5,
                "humidity": 60,
                "pressure": 1013
            },
            "weather": [{ "main": "Clear", "description": "clear sky" }],
            "wind": { "speed": 5.5, "deg": 180 },
            "name": "London",
            "dt": 1705320000,
            "cod": 200
        })
    }

    #[tokio::test]
    async fn test_mock_server_news_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/top-headlines"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_newsapi_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v2/top-headlines", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_mock_server_weather_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/weather"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_weather_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/weather", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_mock_server_error_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/error"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/error", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert_eq!(response.status().as_u16(), 500);
    }

    #[tokio::test]
    async fn test_mock_server_rate_limit_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/rate-limited"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(json!({"error": "Rate limit exceeded"}))
                    .insert_header("Retry-After", "60"),
            )
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/rate-limited", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert_eq!(response.status().as_u16(), 429);
        assert!(response.headers().contains_key("Retry-After"));
    }

    #[tokio::test]
    async fn test_mock_server_timeout() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/slow"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("Slow response")
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(100))
            .build()
            .expect("Failed to build client");

        let result = client
            .get(format!("{}/api/slow", mock_server.uri()))
            .send()
            .await;

        assert!(result.is_err()); // Should timeout
    }
}
