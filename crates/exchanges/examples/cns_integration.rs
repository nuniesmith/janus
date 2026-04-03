//! CNS Integration Example
//!
//! Demonstrates how to integrate CNSReporter with exchange adapters
//! to send health and performance metrics to the JANUS CNS Prometheus registry.
//!
//! Run with:
//! ```bash
//! cargo run --example cns_integration --features cns-metrics
//! ```

use janus_core::MarketDataEvent;
use janus_exchanges::{
    CNSReporter,
    adapters::coinbase::CoinbaseAdapter,
    health::{ExchangeHealthStatus, HealthChecker},
};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== JANUS Exchange CNS Integration Example ===\n");

    // Create exchange adapter
    let adapter = CoinbaseAdapter::new();

    // Create CNS reporter for this exchange
    let reporter = CNSReporter::new("coinbase");

    // Create health checker
    let health_checker = HealthChecker::new();

    println!("Exchange: {}", reporter.exchange());
    println!("WebSocket URL: {}\n", adapter.ws_url());

    // Simulate receiving and parsing messages
    println!("Simulating message processing...\n");

    // Example 1: Trade message
    let trade_msg = r#"{
        "channel": "matches",
        "client_id": "",
        "timestamp": "2023-01-01T00:00:00.000000Z",
        "sequence_num": 0,
        "events": [{
            "type": "match",
            "trades": [{
                "trade_id": "12345",
                "product_id": "BTC-USD",
                "price": "50000.00",
                "size": "0.001",
                "side": "BUY",
                "time": "2023-01-01T00:00:00.000000Z"
            }]
        }]
    }"#;

    let start = Instant::now();
    let events = adapter.parse_message(trade_msg)?;
    let latency = start.elapsed();

    // Report to CNS
    reporter.record_message("matches", "BTC-USD");
    reporter.record_latency("matches", latency);

    // Update health checker
    health_checker
        .record_message("coinbase", Some(latency.as_secs_f64() * 1000.0))
        .await;

    println!("✓ Processed trade message:");
    println!("  - Events: {}", events.len());
    println!("  - Latency: {:?}", latency);
    println!("  - Metrics recorded: message_total, latency_seconds\n");

    // Example 2: Order book message
    let book_msg = r#"{
        "channel": "level2",
        "client_id": "",
        "timestamp": "2023-01-01T00:00:00.000000Z",
        "sequence_num": 1000,
        "events": [{
            "type": "snapshot",
            "product_id": "BTC-USD",
            "updates": [
                {
                    "side": "bid",
                    "event_time": "2023-01-01T00:00:00.000000Z",
                    "price_level": "50000.00",
                    "new_quantity": "1.5"
                },
                {
                    "side": "offer",
                    "event_time": "2023-01-01T00:00:00.000000Z",
                    "price_level": "50001.00",
                    "new_quantity": "1.0"
                }
            ]
        }]
    }"#;

    let start = Instant::now();
    let events = adapter.parse_message(book_msg)?;
    let latency = start.elapsed();

    reporter.record_message("level2", "BTC-USD");
    reporter.record_latency("level2", latency);
    health_checker
        .record_message("coinbase", Some(latency.as_secs_f64() * 1000.0))
        .await;

    println!("✓ Processed order book snapshot:");
    println!("  - Events: {}", events.len());
    println!("  - Latency: {:?}", latency);
    if let Some(MarketDataEvent::OrderBook(book)) = events.first() {
        println!("  - Bids: {}, Asks: {}", book.bids.len(), book.asks.len());
    }
    println!("  - Metrics recorded: message_total, latency_seconds\n");

    // Example 3: Parse error
    let invalid_msg = r#"{"invalid": "json structure"#;

    match adapter.parse_message(invalid_msg) {
        Ok(_) => println!("Unexpectedly parsed invalid JSON"),
        Err(e) => {
            println!("✓ Caught parse error: {}", e);
            reporter.record_parse_error("invalid_json");
            health_checker.record_error("coinbase", e.to_string()).await;
            println!("  - Metrics recorded: parse_errors_total\n");
        }
    }

    // Update health status based on connection quality
    let health = health_checker.get_health("coinbase").await.unwrap();
    let status = match health.status {
        janus_exchanges::health::HealthStatus::Healthy => ExchangeHealthStatus::Healthy,
        janus_exchanges::health::HealthStatus::Degraded => ExchangeHealthStatus::Degraded,
        janus_exchanges::health::HealthStatus::Down => ExchangeHealthStatus::Down,
        janus_exchanges::health::HealthStatus::Unknown => ExchangeHealthStatus::Unknown,
    };

    reporter.update_health(status);

    println!("=== Health Status ===");
    println!("Status: {}", health.status);
    println!("Health Score: {:.2}", health.health_score());
    println!("Total Messages: {}", health.total_messages);
    println!("Parse Errors: {}", health.parse_errors);
    println!("Avg Latency: {:.2}ms", health.avg_latency_ms);
    println!("Error Rate: {:.2} per 1000 msgs", health.error_rate());
    println!("Metrics recorded: health_status\n");

    // Simulate multiple exchanges
    println!("=== Multi-Exchange Example ===\n");

    let exchanges = vec!["binance", "kraken", "okx"];
    for exchange in &exchanges {
        let reporter = CNSReporter::new(exchange);

        // Simulate some messages
        for i in 0..10 {
            reporter.record_message("trades", "BTC-USDT");
            reporter.record_latency("trades", Duration::from_millis(3 + i));
        }

        reporter.update_health(ExchangeHealthStatus::Healthy);
        println!("✓ Reported metrics for {}", exchange);
    }

    println!("\n=== CNS Metrics Summary ===");
    println!("The following Prometheus metrics have been recorded:");
    println!(
        "
  • janus_exchange_message_total{{exchange, channel, symbol}}
    - Total messages received from exchanges
    - Labels: exchange name, channel type, trading symbol

  • janus_exchange_message_parse_errors_total{{exchange, reason}}
    - Parse errors by exchange and error reason
    - Labels: exchange name, error reason

  • janus_exchange_health_status{{exchange}}
    - Current health status (1.0=Healthy, 0.5=Degraded, 0.0=Down)
    - Labels: exchange name

  • janus_exchange_latency_seconds{{exchange, channel}}
    - Message processing latency distribution
    - Labels: exchange name, channel type
    "
    );

    println!("\n=== Grafana Dashboard Queries ===");
    println!("Use these PromQL queries in Grafana:\n");

    println!("1. Message Rate by Exchange:");
    println!("   rate(janus_exchange_message_total[1m])\n");

    println!("2. Parse Error Rate:");
    println!("   rate(janus_exchange_message_parse_errors_total[5m])\n");

    println!("3. P95 Latency by Exchange:");
    println!("   histogram_quantile(0.95, rate(janus_exchange_latency_seconds_bucket[5m]))\n");

    println!("4. Exchange Health Status:");
    println!("   janus_exchange_health_status\n");

    println!("5. Messages per Symbol (top 10):");
    println!("   topk(10, sum by (symbol) (rate(janus_exchange_message_total[1m])))\n");

    #[cfg(feature = "cns-metrics")]
    {
        println!("\n=== Metrics Export ===");
        println!("CNS metrics feature is ENABLED.");
        println!("Metrics are being sent to Prometheus registry.");
        println!("\nTo view metrics:");
        println!("1. Start Prometheus server");
        println!("2. Visit http://localhost:9090");
        println!("3. Query metrics with namespace 'janus' and subsystem 'exchange'");
    }

    #[cfg(not(feature = "cns-metrics"))]
    {
        println!("\n⚠️  CNS metrics feature is DISABLED.");
        println!("To enable metrics reporting:");
        println!("  cargo run --example cns_integration --features cns-metrics");
    }

    println!("\n✅ CNS integration example complete!");

    Ok(())
}
