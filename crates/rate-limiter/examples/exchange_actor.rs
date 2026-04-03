//! # Exchange Actor Example
//!
//! This example demonstrates a realistic exchange actor pattern using the rate limiter.
//! It simulates connecting to multiple exchanges, making various API calls, and handling
//! rate limits gracefully.
//!
//! ## Run with:
//! ```bash
//! cargo run --example exchange_actor
//! ```

use janus_rate_limiter::{RateLimiterManager, TokenBucket, TokenBucketConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

// ============================================================================
// Simulated Exchange API Client
// ============================================================================

#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
enum ApiRequest {
    GetPrice {
        #[allow(dead_code)]
        symbol: String,
        weight: u32,
    },
    GetKlines {
        #[allow(dead_code)]
        symbol: String,
        #[allow(dead_code)]
        interval: String,
        weight: u32,
    },
    GetOrderBook {
        #[allow(dead_code)]
        symbol: String,
        #[allow(dead_code)]
        depth: u32,
        weight: u32,
    },
}

#[derive(Debug)]
struct ApiResponse {
    #[allow(dead_code)]
    request: ApiRequest,
    success: bool,
    latency_ms: u64,
}

/// Simulated exchange actor that respects rate limits
struct ExchangeActor {
    exchange_name: String,
    rate_limiter: Arc<TokenBucket>,
    request_rx: mpsc::Receiver<ApiRequest>,
    response_tx: mpsc::Sender<ApiResponse>,
}

impl ExchangeActor {
    fn new(
        exchange_name: String,
        rate_limiter: Arc<TokenBucket>,
        request_rx: mpsc::Receiver<ApiRequest>,
        response_tx: mpsc::Sender<ApiResponse>,
    ) -> Self {
        Self {
            exchange_name,
            rate_limiter,
            request_rx,
            response_tx,
        }
    }

    async fn run(mut self) {
        info!(exchange = %self.exchange_name, "Exchange actor started");

        while let Some(request) = self.request_rx.recv().await {
            let weight = self.get_request_weight(&request);

            debug!(
                exchange = %self.exchange_name,
                request = ?request,
                weight = weight,
                "Processing request"
            );

            // Attempt to acquire rate limit tokens
            let acquire_start = Instant::now();
            match self.rate_limiter.acquire_async(weight).await {
                Ok(()) => {
                    let wait_time = acquire_start.elapsed();
                    if wait_time > Duration::from_millis(10) {
                        warn!(
                            exchange = %self.exchange_name,
                            wait_ms = wait_time.as_millis(),
                            "Had to wait for rate limit"
                        );
                    }

                    // Simulate API call
                    let response = self.make_api_call(request).await;

                    // Send response back
                    let _ = self.response_tx.send(response).await;
                }
                Err(e) => {
                    error!(
                        exchange = %self.exchange_name,
                        error = %e,
                        "Rate limit error"
                    );

                    // Send failure response
                    let _ = self
                        .response_tx
                        .send(ApiResponse {
                            request,
                            success: false,
                            latency_ms: 0,
                        })
                        .await;
                }
            }
        }

        info!(exchange = %self.exchange_name, "Exchange actor stopped");
    }

    fn get_request_weight(&self, request: &ApiRequest) -> u32 {
        match request {
            ApiRequest::GetPrice { weight, .. } => *weight,
            ApiRequest::GetKlines { weight, .. } => *weight,
            ApiRequest::GetOrderBook { weight, .. } => *weight,
        }
    }

    async fn make_api_call(&self, request: ApiRequest) -> ApiResponse {
        let start = Instant::now();

        // Simulate network latency (50-200ms)
        let latency = 50 + (rand::random::<u64>() % 150);
        sleep(Duration::from_millis(latency)).await;

        // Simulate occasional failures (5%)
        let success = rand::random::<f64>() > 0.05;

        if !success {
            warn!(
                exchange = %self.exchange_name,
                request = ?request,
                "API call failed"
            );
        }

        ApiResponse {
            request,
            success,
            latency_ms: start.elapsed().as_millis() as u64,
        }
    }
}

// ============================================================================
// Request Generator (Simulates Trading Strategy)
// ============================================================================

async fn generate_requests(exchange: &str, tx: mpsc::Sender<ApiRequest>, duration: Duration) {
    info!(exchange = exchange, "Starting request generator");

    let start = Instant::now();
    let mut request_count = 0;

    while start.elapsed() < duration {
        let request = match rand::random::<u32>() % 10 {
            // 50% - Lightweight price checks (weight 1)
            0..=4 => ApiRequest::GetPrice {
                symbol: "BTCUSD".to_string(),
                weight: 1,
            },
            // 30% - Kline requests (weight 5)
            5..=7 => ApiRequest::GetKlines {
                symbol: "ETHUSDT".to_string(),
                interval: "1m".to_string(),
                weight: 5,
            },
            // 20% - Order book requests (weight 10)
            _ => ApiRequest::GetOrderBook {
                symbol: "SOLUSDT".to_string(),
                depth: 100,
                weight: 10,
            },
        };

        if tx.send(request).await.is_err() {
            break;
        }

        request_count += 1;

        // Variable request frequency (10-100ms between requests)
        let interval = 10 + (rand::random::<u64>() % 90);
        sleep(Duration::from_millis(interval)).await;
    }

    info!(
        exchange = exchange,
        request_count = request_count,
        "Request generator finished"
    );
}

// ============================================================================
// Metrics Collector
// ============================================================================

async fn collect_metrics(
    rate_limiter: Arc<TokenBucket>,
    exchange_name: String,
    interval: Duration,
    duration: Duration,
) {
    let start = Instant::now();

    while start.elapsed() < duration {
        sleep(interval).await;

        let metrics = rate_limiter.metrics();
        info!(
            exchange = exchange_name,
            total_requests = metrics.total_requests,
            accepted = metrics.accepted_requests,
            rejected = metrics.rejected_requests,
            current_tokens = metrics.current_tokens as u32,
            window_usage = metrics.window_usage,
            total_wait_ms = metrics.total_wait_time_ms,
            "Rate limiter metrics"
        );
    }
}

// ============================================================================
// Main Scenarios
// ============================================================================

async fn scenario_1_single_exchange() {
    println!("\n=== Scenario 1: Single Exchange (Binance) ===\n");

    let (request_tx, request_rx) = mpsc::channel(1000);
    let (response_tx, mut response_rx) = mpsc::channel(1000);

    let config = TokenBucketConfig::binance_spot();
    let rate_limiter = Arc::new(TokenBucket::new(config).unwrap());
    let actor = ExchangeActor::new(
        "binance".to_string(),
        rate_limiter.clone(),
        request_rx,
        response_tx,
    );

    // Start actor
    let actor_handle = tokio::spawn(actor.run());

    // Start metrics collector
    let metrics_handle = tokio::spawn(collect_metrics(
        rate_limiter.clone(),
        "binance".to_string(),
        Duration::from_secs(2),
        Duration::from_secs(10),
    ));

    // Generate requests for 10 seconds
    let generator_handle = tokio::spawn(generate_requests(
        "binance",
        request_tx.clone(),
        Duration::from_secs(10),
    ));

    // Collect responses
    let success_count;
    let failure_count;
    let total_latency;

    let response_handle = tokio::spawn(async move {
        let mut success = 0;
        let mut failure = 0;
        let mut latency = 0;

        while let Some(response) = response_rx.recv().await {
            if response.success {
                success += 1;
                latency += response.latency_ms;
            } else {
                failure += 1;
            }
        }

        (success, failure, latency)
    });

    // Wait for generator to finish
    generator_handle.await.unwrap();

    // Close the request channel to stop the actor
    drop(request_tx);

    // Wait for actor to finish processing
    actor_handle.await.unwrap();

    // Get final stats
    (success_count, failure_count, total_latency) = response_handle.await.unwrap();

    metrics_handle.abort();

    println!("\nScenario 1 Results:");
    println!("  Successful requests: {}", success_count);
    println!("  Failed requests: {}", failure_count);
    println!(
        "  Average latency: {} ms",
        if success_count > 0 {
            total_latency / success_count as u64
        } else {
            0
        }
    );
}

async fn scenario_2_multi_exchange() {
    println!("\n=== Scenario 2: Multi-Exchange with Manager ===\n");

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

    let exchanges = vec!["binance", "bybit", "kucoin"];
    let mut handles = vec![];

    for exchange in &exchanges {
        let manager_clone = manager.clone();
        let exchange_name = exchange.to_string();

        let handle = tokio::spawn(async move {
            let mut success = 0;
            let mut rate_limited = 0;

            for _ in 0..50 {
                match manager_clone.acquire(&exchange_name, 1).await {
                    Ok(()) => {
                        success += 1;
                        // Simulate API call
                        sleep(Duration::from_millis(20)).await;
                    }
                    Err(_) => {
                        rate_limited += 1;
                    }
                }
                sleep(Duration::from_millis(50)).await;
            }

            (exchange_name, success, rate_limited)
        });

        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        let (exchange, success, rate_limited) = handle.await.unwrap();
        println!(
            "  {}: {} successful, {} rate limited",
            exchange, success, rate_limited
        );
    }

    // Print final metrics
    println!("\nFinal Metrics:");
    for (exchange, metrics) in manager.all_metrics() {
        println!("  {}: {:?}", exchange, metrics);
    }
}

async fn scenario_3_burst_handling() {
    println!("\n=== Scenario 3: Burst Handling Test ===\n");

    let config = TokenBucketConfig {
        capacity: 100,
        refill_rate: 10.0, // Slow refill
        sliding_window: true,
        window_duration: Duration::from_secs(5),
        safety_margin: 0.9,
    };

    let limiter = Arc::new(TokenBucket::new(config).unwrap());

    // Phase 1: Burst of requests
    println!("Phase 1: Sending burst of 100 requests...");
    let start = Instant::now();
    let mut accepted = 0;
    let mut rejected = 0;

    for _ in 0..100 {
        match limiter.acquire(1) {
            Ok(()) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    println!(
        "  Accepted: {}, Rejected: {} (took {:?})",
        accepted,
        rejected,
        start.elapsed()
    );

    // Phase 2: Wait and retry
    println!("\nPhase 2: Waiting 2 seconds for refill...");
    sleep(Duration::from_secs(2)).await;

    let start = Instant::now();
    let mut second_wave = 0;

    for _ in 0..30 {
        if limiter.acquire(1).is_ok() {
            second_wave += 1;
        }
    }

    println!(
        "  Accepted in second wave: {} (took {:?})",
        second_wave,
        start.elapsed()
    );

    // Phase 3: Full window expiration
    println!("\nPhase 3: Waiting for full window to expire (5 seconds)...");
    sleep(Duration::from_secs(5)).await;

    let start = Instant::now();
    let mut third_wave = 0;

    for _ in 0..90 {
        if limiter.acquire(1).is_ok() {
            third_wave += 1;
        }
    }

    println!(
        "  Accepted in third wave: {} (took {:?})",
        third_wave,
        start.elapsed()
    );

    let metrics = limiter.metrics();
    println!("\nFinal metrics: {:?}", metrics);
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("rate_limiter_spike=debug".parse().unwrap()),
        )
        .with_target(false)
        .with_thread_ids(true)
        .init();

    println!("=== Rate Limiter Exchange Actor Examples ===");
    println!("This demonstrates realistic exchange interaction patterns\n");

    // Run scenarios
    scenario_1_single_exchange().await;

    tokio::time::sleep(Duration::from_secs(2)).await;

    scenario_2_multi_exchange().await;

    tokio::time::sleep(Duration::from_secs(2)).await;

    scenario_3_burst_handling().await;

    println!("\n=== All scenarios complete ===");
}
