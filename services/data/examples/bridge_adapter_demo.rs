//! Bridge Adapter Demo
//!
//! This example demonstrates how to use the new bridge adapters to connect
//! to Coinbase, Kraken, and OKX exchanges with CNS metrics and health monitoring.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example bridge_adapter_demo
//! ```
//!
//! ## What This Demonstrates
//!
//! 1. Creating bridge adapters for new exchanges
//! 2. Building WebSocket configurations
//! 3. Parsing messages with automatic CNS metric recording
//! 4. Health status tracking
//! 5. Converting MarketDataEvent to DataMessage

use janus_data::actors::{DataMessage, ExchangeConnector};
use janus_data::connectors::bridge::{CoinbaseBridge, KrakenBridge, OkxBridge};

fn main() {
    println!("=== JANUS Bridge Adapter Demo ===\n");

    // 1. Create bridge adapters
    println!("1. Creating bridge adapters...");
    let coinbase = CoinbaseBridge::new("wss://advanced-trade-ws.coinbase.com".to_string());
    let kraken = KrakenBridge::new("wss://ws.kraken.com/v2".to_string());
    let okx = OkxBridge::new("wss://ws.okx.com:8443/ws/v5/public".to_string());

    println!("   ✓ Coinbase: {}", coinbase.exchange_name());
    println!("   ✓ Kraken: {}", kraken.exchange_name());
    println!("   ✓ OKX: {}\n", okx.exchange_name());

    // 2. Build WebSocket configurations
    println!("2. Building WebSocket configurations for BTC...");
    let coinbase_config = coinbase.build_ws_config("btcusdt");
    let kraken_config = kraken.build_ws_config("btcusdt");
    let okx_config = okx.build_ws_config("btcusdt");

    println!("   Coinbase:");
    println!("     - Symbol: {}", coinbase_config.symbol);
    println!("     - URL: {}", coinbase_config.url);
    println!(
        "     - Ping interval: {}s",
        coinbase_config.ping_interval_secs
    );

    println!("   Kraken:");
    println!("     - Symbol: {}", kraken_config.symbol);
    println!("     - URL: {}", kraken_config.url);

    println!("   OKX:");
    println!("     - Symbol: {}", okx_config.symbol);
    println!("     - URL: {}\n", okx_config.url);

    // 3. Parse sample messages
    println!("3. Parsing sample messages...\n");

    // Coinbase trade message
    let coinbase_msg = r#"{
        "channel": "matches",
        "client_id": "",
        "timestamp": "2024-01-01T00:00:00.000000Z",
        "sequence_num": 0,
        "events": [{
            "type": "match",
            "trades": [{
                "trade_id": "12345",
                "product_id": "BTC-USD",
                "price": "50000.00",
                "size": "0.001",
                "side": "BUY",
                "time": "2024-01-01T00:00:00.000000Z"
            }]
        }]
    }"#;

    println!("   Parsing Coinbase trade message...");
    match coinbase.parse_message(coinbase_msg) {
        Ok(messages) => {
            println!("   ✓ Parsed {} message(s)", messages.len());
            for msg in messages {
                match msg {
                    DataMessage::Trade(trade) => {
                        println!(
                            "     - Trade: {} {} @ {} on {}",
                            trade.symbol, trade.side, trade.price, trade.exchange
                        );
                        println!("       Amount: {}, ID: {}", trade.amount, trade.trade_id);
                    }
                    _ => println!("     - Other message type"),
                }
            }
        }
        Err(e) => println!("   ✗ Parse error: {}", e),
    }

    // Coinbase subscription confirmation
    let coinbase_sub = r#"{
        "channel": "subscriptions",
        "client_id": "",
        "timestamp": "2024-01-01T00:00:00.000000Z",
        "sequence_num": 0,
        "events": [{
            "type": "subscriptions",
            "subscriptions": {
                "ticker": ["BTC-USD"]
            }
        }]
    }"#;

    println!("\n   Parsing Coinbase subscription confirmation...");
    match coinbase.parse_message(coinbase_sub) {
        Ok(messages) => {
            println!(
                "   ✓ Parsed {} message(s) (subscription confirmations are ignored)",
                messages.len()
            );
        }
        Err(e) => println!("   ✗ Parse error: {}", e),
    }

    // Kraken subscription confirmation
    let kraken_sub = r#"{
        "method": "subscribe",
        "result": {
            "channel": "trade",
            "snapshot": false,
            "symbol": "BTC/USD"
        },
        "success": true,
        "time_in": "2024-01-01T00:00:00.000000Z",
        "time_out": "2024-01-01T00:00:00.000000Z"
    }"#;

    println!("\n   Parsing Kraken subscription confirmation...");
    match kraken.parse_message(kraken_sub) {
        Ok(messages) => {
            println!(
                "   ✓ Parsed {} message(s) (subscription confirmations are ignored)",
                messages.len()
            );
        }
        Err(e) => println!("   ✗ Parse error: {}", e),
    }

    // OKX subscription confirmation
    let okx_sub = r#"{
        "event": "subscribe",
        "arg": {
            "channel": "trades",
            "instId": "BTC-USDT"
        },
        "connId": "test"
    }"#;

    println!("\n   Parsing OKX subscription confirmation...");
    match okx.parse_message(okx_sub) {
        Ok(messages) => {
            println!(
                "   ✓ Parsed {} message(s) (subscription confirmations are ignored)",
                messages.len()
            );
        }
        Err(e) => println!("   ✗ Parse error: {}", e),
    }

    // 4. Health monitoring
    println!("\n4. Health Monitoring...");
    println!("   Each bridge adapter has an independent HealthChecker:");
    println!("   - Tracks message counts, latency, errors");
    println!("   - Updates health status (Healthy/Degraded/Down)");
    println!("   - Exposes metrics via CNS/Prometheus");

    let coinbase_health = coinbase.health_checker();
    let kraken_health = kraken.health_checker();
    let okx_health = okx.health_checker();

    println!("\n   Health checkers are independent instances:");
    println!("   - Coinbase HealthChecker: {:p}", &*coinbase_health);
    println!("   - Kraken HealthChecker: {:p}", &*kraken_health);
    println!("   - OKX HealthChecker: {:p}", &*okx_health);

    // 5. CNS Metrics
    println!("\n5. CNS Metrics Integration...");
    println!("   Each bridge automatically records Prometheus metrics:");
    println!("   - janus_exchange_message_total{{exchange=\"coinbase\", ...}}");
    println!("   - janus_exchange_message_parse_errors_total{{exchange=\"coinbase\", ...}}");
    println!("   - janus_exchange_latency_seconds{{exchange=\"coinbase\", ...}}");
    println!("   - janus_exchange_health_status{{exchange=\"coinbase\"}}");
    println!("\n   Metrics are recorded on every parse_message() call.");

    // 6. Symbol formatting
    println!("\n6. Symbol Formatting...");
    println!("   Each exchange uses different symbol formats:");

    let symbols = vec!["btcusdt", "ethusd", "solusdt"];

    for symbol in symbols {
        let cb_config = coinbase.build_ws_config(symbol);
        let kr_config = kraken.build_ws_config(symbol);
        let ox_config = okx.build_ws_config(symbol);

        println!("\n   Input: '{}'", symbol);
        println!("     - Coinbase: {}", cb_config.symbol);
        println!("     - Kraken: {}", kr_config.symbol);
        println!("     - OKX: {}", ox_config.symbol);
    }

    // 7. Error handling
    println!("\n7. Error Handling...");
    println!("   Invalid JSON triggers parse errors and CNS error metrics:");

    match coinbase.parse_message("invalid json") {
        Ok(_) => println!("   ✗ Should have failed!"),
        Err(e) => {
            println!("   ✓ Error caught: {}", e);
            println!("   ✓ Parse error metric recorded");
            println!("   ✓ Health status updated");
        }
    }

    // Summary
    println!("\n=== Summary ===");
    println!("Bridge adapters provide:");
    println!("  ✓ Unified ExchangeConnector interface");
    println!("  ✓ Automatic MarketDataEvent → DataMessage conversion");
    println!("  ✓ CNS metrics recording (messages, errors, latency)");
    println!("  ✓ Health status tracking");
    println!("  ✓ Exchange-specific symbol formatting");
    println!("  ✓ WebSocket configuration generation");
    println!("\nNext steps:");
    println!("  1. Use in ConnectorManager for live connections");
    println!("  2. Query health via health_check() API");
    println!("  3. Monitor metrics in Grafana/Prometheus");
    println!("  4. Set up alerts for health degradation");
    println!("\n=== Demo Complete ===\n");
}
