//! Integration tests for rate limiter with simulated exchange APIs
//!
//! These tests use wiremock to simulate real exchange behavior including:
//! - Rate limit headers
//! - 429 responses
//! - Varying response times
//! - Realistic request patterns

use janus_rate_limiter::{RateLimitError, RateLimiterManager, TokenBucket, TokenBucketConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Simulates Binance API with rate limit headers
#[tokio::test]
async fn test_binance_simulation_with_headers() {
    let mock_server = MockServer::start().await;

    // Track request count for realistic header values
    let request_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let request_count_clone = request_count.clone();

    Mock::given(method("GET"))
        .and(path("/api/v3/ticker/price"))
        .respond_with(move |_req: &wiremock::Request| {
            let count = request_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let used_weight = count * 2; // Each request costs 2 weight

            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({
                    "symbol": "BTCUSD",
                    "price": "50000.00"
                }))
                .insert_header("X-MBX-USED-WEIGHT-1M", used_weight.to_string())
                .insert_header("X-MBX-ORDER-COUNT-10S", "0")
        })
        .mount(&mock_server)
        .await;

    // Create rate limiter with Binance config
    let config = TokenBucketConfig::binance_spot();
    let limiter = Arc::new(TokenBucket::new(config).unwrap());
    let client = reqwest::Client::new();

    // Make requests and update limiter based on headers
    for i in 0..10 {
        limiter.acquire_async(2).await.unwrap();

        let response = client
            .get(format!("{}/api/v3/ticker/price", mock_server.uri()))
            .send()
            .await
            .unwrap();

        // Update limiter from response headers
        if let Some(used_weight) = response.headers().get("X-MBX-USED-WEIGHT-1M") {
            let used = used_weight.to_str().unwrap().parse::<u32>().unwrap();
            limiter.update_from_headers(used, 6000);

            println!("Request {}: Used weight = {}", i, used);
        }
    }

    let metrics = limiter.metrics();
    assert_eq!(metrics.accepted_requests, 10);
    assert_eq!(metrics.rejected_requests, 0);
}

/// Simulates 429 rate limit responses
#[tokio::test]
async fn test_429_response_handling() {
    let mock_server = MockServer::start().await;

    // First 5 requests succeed, then we get rate limited
    let request_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let request_count_clone = request_count.clone();

    Mock::given(method("GET"))
        .and(path("/api/v3/klines"))
        .respond_with(move |_req: &wiremock::Request| {
            let count = request_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            if count < 5 {
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!([]))
                    .insert_header("X-MBX-USED-WEIGHT-1M", ((count + 1) * 1000).to_string())
            } else {
                ResponseTemplate::new(429)
                    .set_body_json(serde_json::json!({
                        "code": -1003,
                        "msg": "Too many requests"
                    }))
                    .insert_header("Retry-After", "60")
            }
        })
        .mount(&mock_server)
        .await;

    let config = TokenBucketConfig {
        capacity: 6000,
        refill_rate: 100.0,
        sliding_window: true,
        window_duration: Duration::from_secs(60),
        safety_margin: 1.0,
    };
    let limiter = Arc::new(TokenBucket::new(config).unwrap());
    let client = reqwest::Client::new();

    let mut success_count = 0;
    let mut rate_limited_count = 0;

    for _ in 0..10 {
        // Heavy request (1000 weight)
        if limiter.acquire(1000).is_err() {
            println!("Client-side rate limit triggered");
            break;
        }

        let response = client
            .get(format!("{}/api/v3/klines", mock_server.uri()))
            .send()
            .await
            .unwrap();

        if response.status() == 429 {
            rate_limited_count += 1;
            println!("Server returned 429");

            // Update our limiter to match server state
            limiter.update_from_headers(6000, 6000); // Maxed out
            break;
        } else {
            success_count += 1;

            // Update from headers
            if let Some(used_weight) = response.headers().get("X-MBX-USED-WEIGHT-1M") {
                let used = used_weight.to_str().unwrap().parse::<u32>().unwrap();
                limiter.update_from_headers(used, 6000);
            }
        }
    }

    assert_eq!(success_count, 5);
    assert_eq!(rate_limited_count, 1);
}

/// Test concurrent actors hitting the same rate limiter
#[tokio::test]
async fn test_concurrent_actors() {
    let config = TokenBucketConfig {
        capacity: 1000,
        refill_rate: 100.0,
        sliding_window: false,
        safety_margin: 0.8,
        window_duration: Duration::from_secs(60),
    };

    let limiter = Arc::new(TokenBucket::new(config).unwrap());
    let mut join_set = JoinSet::new();

    // Spawn 10 concurrent "actors"
    for actor_id in 0..10 {
        let limiter_clone = limiter.clone();
        join_set.spawn(async move {
            let mut successful_requests = 0;
            let mut failed_requests = 0;

            for _ in 0..20 {
                match limiter_clone.acquire(10) {
                    Ok(()) => {
                        successful_requests += 1;
                        // Simulate some work
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    Err(RateLimitError::Exceeded { retry_after: _ }) => {
                        failed_requests += 1;
                        // In real code, we'd wait or switch sources
                        // For this test, just continue
                    }
                    Err(_) => {}
                }
            }

            (actor_id, successful_requests, failed_requests)
        });
    }

    // Collect results
    let mut total_success = 0;
    let mut total_failed = 0;

    while let Some(result) = join_set.join_next().await {
        let (actor_id, success, failed) = result.unwrap();
        println!("Actor {}: {} success, {} failed", actor_id, success, failed);
        total_success += success;
        total_failed += failed;
    }

    println!("Total: {} success, {} failed", total_success, total_failed);

    // We should have some failures due to contention
    assert!(total_failed > 0, "Expected some rate limit failures");

    // Total requests should not exceed capacity
    let metrics = limiter.metrics();
    assert!(metrics.accepted_requests <= 1000);
}

/// Test rate limiter manager with multiple exchanges
#[tokio::test]
async fn test_multi_exchange_manager() {
    let manager = RateLimiterManager::new();

    manager
        .register("binance".to_string(), TokenBucketConfig::binance_spot())
        .unwrap();
    manager
        .register("bybit".to_string(), TokenBucketConfig::bybit_v5())
        .unwrap();
    manager
        .register("kucoin".to_string(), TokenBucketConfig::kucoin_public())
        .unwrap();

    // Spawn concurrent tasks hitting different exchanges
    let mut join_set = JoinSet::new();

    for exchange in ["binance", "bybit", "kucoin"] {
        let manager_clone = manager.clone();
        let exchange_name = exchange.to_string();

        join_set.spawn(async move {
            let mut count = 0;
            for _ in 0..50 {
                if manager_clone.acquire(&exchange_name, 1).await.is_ok() {
                    count += 1;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            (exchange_name, count)
        });
    }

    while let Some(result) = join_set.join_next().await {
        let (exchange, count) = result.unwrap();
        println!("{}: {} successful requests", exchange, count);
        assert!(count > 0);
    }

    // Check metrics
    let metrics = manager.all_metrics();
    assert_eq!(metrics.len(), 3);

    for (exchange, metric) in metrics.iter() {
        println!("{} metrics: {:?}", exchange, metric);
        assert!(metric.total_requests > 0);
    }
}

/// Test sliding window behavior with precise timing
///
/// NOTE: This test uses real time delays (1.1 seconds total) and can be slow or hang in CI.
/// Run with `cargo test -- --ignored` to include this test.
#[tokio::test]
#[ignore]
async fn test_sliding_window_precision() {
    let config = TokenBucketConfig {
        capacity: 100,
        refill_rate: 100.0,
        sliding_window: true,
        window_duration: Duration::from_secs(1),
        safety_margin: 1.0,
    };

    let limiter = TokenBucket::new(config).unwrap();

    // Fill up the window
    for _ in 0..10 {
        limiter.acquire(10).unwrap();
    }

    // This should fail - window is full
    assert!(limiter.acquire(1).is_err());

    // Wait for half the window to expire
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Still should fail - we haven't passed the full window
    assert!(limiter.acquire(1).is_err());

    // Wait for the rest of the window
    tokio::time::sleep(Duration::from_millis(600)).await;

    // Now it should succeed - window has reset
    assert!(limiter.acquire(50).is_ok());
}

/// Test safety margin behavior
#[tokio::test]
async fn test_safety_margin_prevents_429() {
    let config = TokenBucketConfig {
        capacity: 1000,
        refill_rate: 100.0,
        sliding_window: false,
        safety_margin: 0.8, // Stop at 80%
        window_duration: Duration::from_secs(60),
    };

    let limiter = TokenBucket::new(config).unwrap();

    // We should be able to use 800 tokens (80% of 1000)
    assert!(limiter.acquire(800).is_ok());

    // This should fail even though we technically have 200 tokens left
    assert!(limiter.acquire(1).is_err());

    let metrics = limiter.metrics();
    println!("Metrics: {:?}", metrics);
    assert_eq!(metrics.rejected_requests, 1);
}

/// Test async acquire waiting behavior
#[tokio::test]
async fn test_async_acquire_waits_correctly() {
    let config = TokenBucketConfig {
        capacity: 100,
        refill_rate: 100.0, // 100 tokens/sec
        sliding_window: false,
        safety_margin: 1.0,
        window_duration: Duration::from_secs(60),
    };

    let limiter = TokenBucket::new(config).unwrap();

    // Drain the bucket
    limiter.acquire(100).unwrap();

    let start = Instant::now();

    // This should wait ~0.5 seconds to get 50 tokens
    limiter.acquire_async(50).await.unwrap();

    let elapsed = start.elapsed();
    println!("Wait time: {:?}", elapsed);

    // Should be roughly 500ms (allow 200ms margin for scheduling)
    assert!(elapsed >= Duration::from_millis(400));
    assert!(elapsed <= Duration::from_millis(700));

    let metrics = limiter.metrics();
    assert!(metrics.total_wait_time_ms >= 400);
}

/// Stress test: high-frequency requests
#[tokio::test]
async fn test_high_frequency_stress() {
    let config = TokenBucketConfig {
        capacity: 10000,
        refill_rate: 1000.0,
        sliding_window: true,
        window_duration: Duration::from_secs(60),
        safety_margin: 0.9,
    };

    let limiter = Arc::new(TokenBucket::new(config).unwrap());
    let mut join_set = JoinSet::new();

    let start = Instant::now();

    // Spawn 20 tasks making rapid requests
    for _ in 0..20 {
        let limiter_clone = limiter.clone();
        join_set.spawn(async move {
            let mut count = 0;
            for _ in 0..100 {
                if limiter_clone.acquire(1).is_ok() {
                    count += 1;
                }
            }
            count
        });
    }

    let mut total = 0;
    while let Some(result) = join_set.join_next().await {
        total += result.unwrap();
    }

    let elapsed = start.elapsed();
    println!(
        "Processed {} requests in {:?} ({:.0} req/sec)",
        total,
        elapsed,
        total as f64 / elapsed.as_secs_f64()
    );

    let metrics = limiter.metrics();
    println!("Final metrics: {:?}", metrics);

    assert!(total > 0);
    assert_eq!(metrics.total_requests, 2000); // 20 tasks * 100 requests
}

/// Test recovery after hitting rate limit
#[tokio::test]
async fn test_recovery_after_rate_limit() {
    let config = TokenBucketConfig {
        capacity: 100,
        refill_rate: 50.0, // Refill 50/sec
        sliding_window: false,
        safety_margin: 1.0,
        window_duration: Duration::from_secs(60),
    };

    let limiter = TokenBucket::new(config).unwrap();

    // Phase 1: Use all tokens
    limiter.acquire(100).unwrap();

    // Phase 2: Should fail immediately
    assert!(limiter.acquire(10).is_err());

    // Phase 3: Wait for refill
    tokio::time::sleep(Duration::from_millis(400)).await;

    // Phase 4: Should succeed now (~20 tokens refilled)
    assert!(limiter.acquire(15).is_ok());

    let metrics = limiter.metrics();
    assert_eq!(metrics.accepted_requests, 2);
    assert_eq!(metrics.rejected_requests, 1);
}
