//! Wiremock-based integration tests for external API clients
//!
//! These tests use wiremock to mock external API responses, enabling:
//! - Deterministic CI testing without real API keys
//! - Testing error handling and edge cases
//! - Testing rate limiting and retry behavior
//! - No network dependency or secret leakage
//!
//! Note: These tests verify HTTP behavior patterns that the actual API
//! clients should handle. They test the mock server infrastructure and
//! common HTTP patterns used by external APIs.

use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// NewsAPI-style Mock Tests
// ============================================================================

mod newsapi_mock_tests {
    use super::*;

    fn mock_news_response() -> serde_json::Value {
        json!({
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "source": { "id": "bloomberg", "name": "Bloomberg" },
                    "author": "Test Author",
                    "title": "Bitcoin Surges Past $100K as Institutional Interest Grows",
                    "description": "Bitcoin reached new highs amid increased adoption.",
                    "url": "https://example.com/article1",
                    "urlToImage": "https://example.com/image1.jpg",
                    "publishedAt": "2024-01-15T10:30:00Z",
                    "content": "Full article content here..."
                },
                {
                    "source": { "id": null, "name": "Reuters" },
                    "author": null,
                    "title": "Fed Signals Potential Rate Cuts in 2024",
                    "description": "Federal Reserve hints at monetary policy shift.",
                    "url": "https://example.com/article2",
                    "urlToImage": null,
                    "publishedAt": "2024-01-15T09:00:00Z",
                    "content": null
                }
            ]
        })
    }

    #[tokio::test]
    async fn test_news_endpoint_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/top-headlines"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_news_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v2/top-headlines", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");
        assert_eq!(body["status"], "ok");
        assert_eq!(body["articles"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_news_endpoint_with_api_key() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/top-headlines"))
            .and(query_param("apiKey", "test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_news_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "{}/v2/top-headlines?apiKey=test-api-key",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_news_endpoint_with_country_filter() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/top-headlines"))
            .and(query_param("country", "us"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_news_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v2/top-headlines?country=us", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_news_endpoint_unauthorized() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/top-headlines"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "status": "error",
                "code": "apiKeyInvalid",
                "message": "Your API key is invalid or incorrect."
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v2/top-headlines", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert_eq!(response.status().as_u16(), 401);
    }

    #[tokio::test]
    async fn test_news_endpoint_rate_limited() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/top-headlines"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(json!({
                        "status": "error",
                        "code": "rateLimited",
                        "message": "You have been rate limited."
                    }))
                    .insert_header("Retry-After", "60"),
            )
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v2/top-headlines", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert_eq!(response.status().as_u16(), 429);
        assert!(response.headers().contains_key("retry-after"));
    }

    #[tokio::test]
    async fn test_news_endpoint_empty_results() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/top-headlines"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "ok",
                "totalResults": 0,
                "articles": []
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v2/top-headlines", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse");
        assert_eq!(body["totalResults"], 0);
        assert!(body["articles"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_news_endpoint_search() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/everything"))
            .and(query_param("q", "bitcoin"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_news_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v2/everything?q=bitcoin", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }
}

// ============================================================================
// CryptoPanic-style Mock Tests
// ============================================================================

mod cryptopanic_mock_tests {
    use super::*;

    fn mock_crypto_news_response() -> serde_json::Value {
        json!({
            "count": 2,
            "next": null,
            "previous": null,
            "results": [
                {
                    "id": 12345678,
                    "kind": "news",
                    "domain": "coindesk.com",
                    "source": {
                        "title": "CoinDesk",
                        "region": "en",
                        "domain": "coindesk.com"
                    },
                    "title": "Ethereum ETF Approval Expected Soon",
                    "published_at": "2024-01-15T12:00:00Z",
                    "url": "https://coindesk.com/article1",
                    "currencies": [
                        { "code": "ETH", "title": "Ethereum", "slug": "ethereum" }
                    ],
                    "votes": {
                        "negative": 5,
                        "positive": 42,
                        "important": 15
                    }
                },
                {
                    "id": 12345679,
                    "kind": "news",
                    "domain": "decrypt.co",
                    "source": {
                        "title": "Decrypt",
                        "region": "en",
                        "domain": "decrypt.co"
                    },
                    "title": "DeFi Protocol Sees Record Volume",
                    "published_at": "2024-01-15T11:30:00Z",
                    "url": "https://decrypt.co/article2",
                    "currencies": [
                        { "code": "BTC", "title": "Bitcoin", "slug": "bitcoin" }
                    ],
                    "votes": {
                        "negative": 3,
                        "positive": 28,
                        "important": 10
                    }
                }
            ]
        })
    }

    #[tokio::test]
    async fn test_crypto_news_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/posts/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_crypto_news_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v1/posts/", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");
        assert_eq!(body["count"], 2);
        assert_eq!(body["results"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_crypto_news_with_filter() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/posts/"))
            .and(query_param("filter", "bullish"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "count": 1,
                "results": [{
                    "id": 1,
                    "title": "Bitcoin to the Moon!",
                    "published_at": "2024-01-15T12:00:00Z",
                    "url": "https://example.com",
                    "votes": { "positive": 100, "negative": 2, "important": 50 }
                }]
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v1/posts/?filter=bullish", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse");
        assert_eq!(body["count"], 1);
    }

    #[tokio::test]
    async fn test_crypto_news_by_currency() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/posts/"))
            .and(query_param("currencies", "BTC,ETH"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_crypto_news_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "{}/v1/posts/?currencies=BTC,ETH",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_crypto_news_unauthorized() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/posts/"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": "Invalid auth token"
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/v1/posts/", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert_eq!(response.status().as_u16(), 401);
    }
}

// ============================================================================
// OpenWeatherMap-style Mock Tests
// ============================================================================

mod openweathermap_mock_tests {
    use super::*;

    fn mock_current_weather_response() -> serde_json::Value {
        json!({
            "coord": { "lon": -0.13, "lat": 51.51 },
            "weather": [
                {
                    "id": 800,
                    "main": "Clear",
                    "description": "clear sky",
                    "icon": "01d"
                }
            ],
            "main": {
                "temp": 22.5,
                "feels_like": 21.8,
                "temp_min": 20.0,
                "temp_max": 25.0,
                "pressure": 1013,
                "humidity": 45
            },
            "visibility": 10000,
            "wind": {
                "speed": 3.5,
                "deg": 180,
                "gust": 5.2
            },
            "clouds": { "all": 5 },
            "dt": 1705320000,
            "sys": {
                "country": "GB",
                "sunrise": 1705325400,
                "sunset": 1705361400
            },
            "timezone": 0,
            "name": "London",
            "cod": 200
        })
    }

    fn mock_forecast_response() -> serde_json::Value {
        json!({
            "cod": "200",
            "message": 0,
            "cnt": 2,
            "list": [
                {
                    "dt": 1705320000,
                    "main": {
                        "temp": 22.5,
                        "humidity": 45
                    },
                    "weather": [{ "main": "Clear", "description": "clear sky" }],
                    "wind": { "speed": 3.5 },
                    "dt_txt": "2024-01-15 12:00:00"
                },
                {
                    "dt": 1705330800,
                    "main": {
                        "temp": 24.0,
                        "humidity": 40
                    },
                    "weather": [{ "main": "Clouds", "description": "few clouds" }],
                    "wind": { "speed": 4.0 },
                    "dt_txt": "2024-01-15 15:00:00"
                }
            ],
            "city": {
                "name": "London",
                "country": "GB"
            }
        })
    }

    #[tokio::test]
    async fn test_weather_by_city() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/weather"))
            .and(query_param("q", "London"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_current_weather_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/weather?q=London&appid=test", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse");
        assert_eq!(body["main"]["temp"], 22.5);
        assert_eq!(body["name"], "London");
    }

    #[tokio::test]
    async fn test_weather_by_coords() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/weather"))
            .and(query_param("lat", "51.51"))
            .and(query_param("lon", "-0.13"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_current_weather_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/weather?lat=51.51&lon=-0.13", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_forecast_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/forecast"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_forecast_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/forecast?q=London", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse");
        assert_eq!(body["cnt"], 2);
    }

    #[tokio::test]
    async fn test_weather_city_not_found() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/weather"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                "cod": "404",
                "message": "city not found"
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/weather?q=NonExistentCity", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert_eq!(response.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_weather_server_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/weather"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/weather?q=London", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert_eq!(response.status().as_u16(), 500);
    }
}

// ============================================================================
// Space Weather (NOAA) Mock Tests
// ============================================================================

mod spaceweather_mock_tests {
    use super::*;

    fn mock_solar_wind_response() -> serde_json::Value {
        json!([
            {
                "time_tag": "2024-01-15 12:00:00.000",
                "density": 5.2,
                "speed": 420.5,
                "temperature": 125000
            },
            {
                "time_tag": "2024-01-15 11:55:00.000",
                "density": 5.1,
                "speed": 418.0,
                "temperature": 123000
            }
        ])
    }

    fn mock_geomagnetic_response() -> serde_json::Value {
        json!([
            ["time_tag", "Kp", "a_running", "station_count"],
            ["2024-01-15 12:00:00.000", "3", "15", "8"],
            ["2024-01-15 09:00:00.000", "2", "10", "8"]
        ])
    }

    fn mock_xray_flux_response() -> serde_json::Value {
        json!([
            {
                "time_tag": "2024-01-15 12:00:00.000",
                "satellite": 16,
                "current_class": "B",
                "current_ratio": 2.5,
                "current_int_xrlong": 2.5e-7
            }
        ])
    }

    #[tokio::test]
    async fn test_solar_wind_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/products/solar-wind/plasma-5-minute.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_solar_wind_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "{}/products/solar-wind/plasma-5-minute.json",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse");
        assert!(body.as_array().is_some());
        assert!(!body.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_geomagnetic_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/products/noaa-planetary-k-index.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_geomagnetic_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "{}/products/noaa-planetary-k-index.json",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_xray_flux_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/products/goes-primary-xr-flux-6hour.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_xray_flux_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "{}/products/goes-primary-xr-flux-6hour.json",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_space_weather_partial_failure() {
        let mock_server = MockServer::start().await;

        // Solar wind succeeds
        Mock::given(method("GET"))
            .and(path("/products/solar-wind/plasma-5-minute.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_solar_wind_response()))
            .mount(&mock_server)
            .await;

        // Geomagnetic fails
        Mock::given(method("GET"))
            .and(path("/products/noaa-planetary-k-index.json"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();

        let solar = client
            .get(format!(
                "{}/products/solar-wind/plasma-5-minute.json",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");
        assert!(solar.status().is_success());

        let geo = client
            .get(format!(
                "{}/products/noaa-planetary-k-index.json",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");
        assert_eq!(geo.status().as_u16(), 500);
    }
}

// ============================================================================
// CryptoCompare-style Mock Tests
// ============================================================================

mod cryptocompare_mock_tests {
    use super::*;

    fn mock_cc_news_response() -> serde_json::Value {
        json!({
            "Type": 100,
            "Message": "News list successfully returned",
            "Data": [
                {
                    "id": "1234567",
                    "published_on": 1705320000,
                    "title": "Bitcoin Mining Difficulty Reaches All-Time High",
                    "url": "https://example.com/article/1",
                    "body": "Bitcoin mining difficulty has increased...",
                    "tags": "BTC|Mining|Network",
                    "categories": "Mining|Network",
                    "source": "CryptoNews"
                },
                {
                    "id": "1234568",
                    "published_on": 1705316400,
                    "title": "Ethereum Layer 2 Solutions See Record Growth",
                    "url": "https://example.com/article/2",
                    "body": "Layer 2 networks on Ethereum...",
                    "tags": "ETH|L2|Scaling",
                    "categories": "Technology|Scaling",
                    "source": "BlockchainDaily"
                }
            ]
        })
    }

    fn mock_price_response() -> serde_json::Value {
        json!({
            "USD": 45000.50,
            "EUR": 41500.25
        })
    }

    #[tokio::test]
    async fn test_cc_news_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/data/v2/news/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_cc_news_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/data/v2/news/", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse");
        assert_eq!(body["Type"], 100);
        assert_eq!(body["Data"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_cc_news_by_category() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/data/v2/news/"))
            .and(query_param("categories", "Mining"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_cc_news_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "{}/data/v2/news/?categories=Mining",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
    }

    #[tokio::test]
    async fn test_cc_price_endpoint() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/data/price"))
            .and(query_param("fsym", "BTC"))
            .and(query_param("tsyms", "USD,EUR"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_price_response()))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "{}/data/price?fsym=BTC&tsyms=USD,EUR",
                mock_server.uri()
            ))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await.expect("Failed to parse");
        assert_eq!(body["USD"], 45000.50);
        assert_eq!(body["EUR"], 41500.25);
    }

    #[tokio::test]
    async fn test_cc_rate_limited() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/data/v2/news/"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(json!({
                        "Response": "Error",
                        "Message": "Rate limit exceeded"
                    }))
                    .insert_header("X-RateLimit-Remaining", "0"),
            )
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/data/v2/news/", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert_eq!(response.status().as_u16(), 429);
    }
}

// ============================================================================
// Common HTTP Pattern Tests
// ============================================================================

mod common_http_tests {
    use super::*;

    #[tokio::test]
    async fn test_malformed_json_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/data"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not valid json"))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/data", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");

        assert!(response.status().is_success());
        let result: Result<serde_json::Value, _> = response.json().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/slow"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"data": "slow"}))
                    .set_delay(Duration::from_secs(10)),
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

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_simulation() {
        let mock_server = MockServer::start().await;
        let request_count = Arc::new(AtomicU32::new(0));
        let count_clone = request_count.clone();

        Mock::given(method("GET"))
            .and(path("/api/flaky"))
            .respond_with(move |_req: &wiremock::Request| {
                let count = count_clone.fetch_add(1, Ordering::SeqCst);
                if count < 2 {
                    ResponseTemplate::new(503).set_body_string("Service Unavailable")
                } else {
                    ResponseTemplate::new(200).set_body_json(json!({"status": "ok"}))
                }
            })
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();

        // First two requests should fail
        let r1 = client
            .get(format!("{}/api/flaky", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");
        assert_eq!(r1.status().as_u16(), 503);

        let r2 = client
            .get(format!("{}/api/flaky", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");
        assert_eq!(r2.status().as_u16(), 503);

        // Third request should succeed
        let r3 = client
            .get(format!("{}/api/flaky", mock_server.uri()))
            .send()
            .await
            .expect("Request failed");
        assert!(r3.status().is_success());
    }

    #[tokio::test]
    async fn test_concurrent_requests() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/data"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/api/data", mock_server.uri());

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let client = client.clone();
                let url = url.clone();
                tokio::spawn(async move { client.get(&url).send().await })
            })
            .collect();

        for handle in handles {
            let result = handle.await.expect("Task panicked");
            let response = result.expect("Request failed");
            assert!(response.status().is_success());
        }
    }

    #[tokio::test]
    async fn test_request_counting() {
        let mock_server = MockServer::start().await;
        let request_count = Arc::new(AtomicU32::new(0));
        let count_clone = request_count.clone();

        Mock::given(method("GET"))
            .and(path("/api/counted"))
            .respond_with(move |_req: &wiremock::Request| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200).set_body_json(json!({"counted": true}))
            })
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();

        for _ in 0..5 {
            let _ = client
                .get(format!("{}/api/counted", mock_server.uri()))
                .send()
                .await;
        }

        assert_eq!(request_count.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn test_various_http_status_codes() {
        let mock_server = MockServer::start().await;

        let status_codes = [200, 201, 400, 401, 403, 404, 500, 502, 503];

        for code in status_codes {
            Mock::given(method("GET"))
                .and(path(format!("/status/{}", code)))
                .respond_with(ResponseTemplate::new(code))
                .mount(&mock_server)
                .await;
        }

        let client = reqwest::Client::new();

        for code in status_codes {
            let response = client
                .get(format!("{}/status/{}", mock_server.uri(), code))
                .send()
                .await
                .expect("Request failed");

            assert_eq!(response.status().as_u16(), code);
        }
    }
}
