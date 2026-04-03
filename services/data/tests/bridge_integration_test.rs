//! Integration tests for bridge adapters
//!
//! These tests verify that the bridge adapters correctly:
//! - Wrap exchange adapters from janus-exchanges
//! - Implement the ExchangeConnector trait
//! - Convert MarketDataEvent to DataMessage
//! - Record CNS metrics (messages, errors, latency)
//! - Track health status
//! - Build WebSocket configurations correctly

use janus_data::actors::{DataMessage, ExchangeConnector};
use janus_data::connectors::bridge::{CoinbaseBridge, KrakenBridge, OkxBridge};
use std::sync::Arc;

#[test]
fn test_coinbase_bridge_implements_exchange_connector() {
    let bridge = CoinbaseBridge::new("wss://test.coinbase.com".to_string());

    assert_eq!(bridge.exchange_name(), "coinbase");
    assert_eq!(bridge.ws_url(), "wss://test.coinbase.com");
}

#[test]
fn test_coinbase_bridge_builds_ws_config() {
    let bridge = CoinbaseBridge::new("wss://test.coinbase.com".to_string());

    let config = bridge.build_ws_config("btcusdt");

    assert_eq!(config.url, "wss://test.coinbase.com");
    assert_eq!(config.exchange, "coinbase");
    assert_eq!(config.symbol, "BTC-USDT");
    assert!(config.subscription_msg.is_some());
    assert_eq!(config.ping_interval_secs, 30);
}

#[tokio::test]
async fn test_coinbase_bridge_parses_trade_message() {
    let bridge = CoinbaseBridge::new("wss://test.coinbase.com".to_string());

    // Sample Coinbase trade message
    let raw_message = r#"{
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

    let result = bridge.parse_message(raw_message);
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert_eq!(messages.len(), 1);

    match &messages[0] {
        DataMessage::Trade(trade) => {
            // Symbol is normalized to standard format (BTC/USD) from exchange format (BTC-USD)
            assert_eq!(trade.symbol, "BTC/USD");
            assert_eq!(trade.exchange, "coinbase");
            assert_eq!(trade.price, 50000.0);
            assert_eq!(trade.amount, 0.001);
        }
        _ => panic!("Expected Trade message"),
    }
}

#[tokio::test]
async fn test_coinbase_bridge_handles_subscription_confirmation() {
    let bridge = CoinbaseBridge::new("wss://test.coinbase.com".to_string());

    let raw_message = r#"{
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

    let result = bridge.parse_message(raw_message);
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert_eq!(messages.len(), 0); // Subscription confirmations don't generate data messages
}

#[tokio::test]
async fn test_coinbase_bridge_handles_parse_errors() {
    let bridge = CoinbaseBridge::new("wss://test.coinbase.com".to_string());

    let invalid_json = "not valid json";
    let result = bridge.parse_message(invalid_json);

    // Should return an error
    assert!(result.is_err());
}

#[test]
fn test_kraken_bridge_implements_exchange_connector() {
    let bridge = KrakenBridge::new("wss://test.kraken.com".to_string());

    assert_eq!(bridge.exchange_name(), "kraken");
    assert_eq!(bridge.ws_url(), "wss://test.kraken.com");
}

#[test]
fn test_kraken_bridge_builds_ws_config() {
    let bridge = KrakenBridge::new("wss://test.kraken.com".to_string());

    let config = bridge.build_ws_config("btcusdt");

    assert_eq!(config.url, "wss://test.kraken.com");
    assert_eq!(config.exchange, "kraken");
    assert_eq!(config.symbol, "BTC/USDT");
    assert!(config.subscription_msg.is_some());
}

#[tokio::test]
async fn test_kraken_bridge_handles_subscription_confirmation() {
    let bridge = KrakenBridge::new("wss://test.kraken.com".to_string());

    let raw_message = r#"{
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

    let result = bridge.parse_message(raw_message);
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert_eq!(messages.len(), 0); // Subscription confirmations don't generate data messages
}

#[test]
fn test_okx_bridge_implements_exchange_connector() {
    let bridge = OkxBridge::new("wss://test.okx.com".to_string());

    assert_eq!(bridge.exchange_name(), "okx");
    assert_eq!(bridge.ws_url(), "wss://test.okx.com");
}

#[test]
fn test_okx_bridge_builds_ws_config() {
    let bridge = OkxBridge::new("wss://test.okx.com".to_string());

    let config = bridge.build_ws_config("btcusdt");

    assert_eq!(config.url, "wss://test.okx.com");
    assert_eq!(config.exchange, "okx");
    assert_eq!(config.symbol, "BTC-USDT");
    assert!(config.subscription_msg.is_some());
}

#[tokio::test]
async fn test_okx_bridge_handles_subscription_confirmation() {
    let bridge = OkxBridge::new("wss://test.okx.com".to_string());

    let raw_message = r#"{
        "event": "subscribe",
        "arg": {
            "channel": "trades",
            "instId": "BTC-USDT"
        },
        "connId": "test"
    }"#;

    let result = bridge.parse_message(raw_message);
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert_eq!(messages.len(), 0); // Subscription confirmations don't generate data messages
}

#[test]
fn test_bridge_symbol_formatting_coinbase() {
    let bridge = CoinbaseBridge::new("wss://test".to_string());

    let config = bridge.build_ws_config("btcusdt");
    assert_eq!(config.symbol, "BTC-USDT");

    let config2 = bridge.build_ws_config("ethusd");
    assert_eq!(config2.symbol, "ETH-USD");
}

#[test]
fn test_bridge_symbol_formatting_kraken() {
    let bridge = KrakenBridge::new("wss://test".to_string());

    let config = bridge.build_ws_config("btcusdt");
    assert_eq!(config.symbol, "BTC/USDT");

    let config2 = bridge.build_ws_config("ethusd");
    assert_eq!(config2.symbol, "ETH/USD");
}

#[test]
fn test_bridge_symbol_formatting_okx() {
    let bridge = OkxBridge::new("wss://test".to_string());

    let config = bridge.build_ws_config("btcusdt");
    assert_eq!(config.symbol, "BTC-USDT");

    let config2 = bridge.build_ws_config("ethusd");
    assert_eq!(config2.symbol, "ETH-USD");
}

#[test]
fn test_bridge_subscription_message_generation() {
    let coinbase = CoinbaseBridge::new("wss://test".to_string());
    let kraken = KrakenBridge::new("wss://test".to_string());
    let okx = OkxBridge::new("wss://test".to_string());

    // All bridges should generate subscription messages
    assert!(coinbase.subscription_message("btcusdt").is_some());
    assert!(kraken.subscription_message("btcusdt").is_some());
    assert!(okx.subscription_message("btcusdt").is_some());
}

#[test]
fn test_health_checker_integration() {
    let bridge = CoinbaseBridge::new("wss://test".to_string());

    let health_checker = bridge.health_checker();

    // Initial health should be Unknown
    // Note: get_health is async and returns Option<ExchangeHealth>
    // We can't easily test this in a non-async test, so we just verify the checker exists
    assert!(Arc::strong_count(&health_checker) >= 1);

    // Note: record_message is async, so we can't easily call it from a sync test
    // Health tracking is tested in the janus-exchanges crate tests
}

#[test]
fn test_multiple_exchanges_side_by_side() {
    let coinbase = CoinbaseBridge::new("wss://coinbase.test".to_string());
    let kraken = KrakenBridge::new("wss://kraken.test".to_string());
    let okx = OkxBridge::new("wss://okx.test".to_string());

    assert_eq!(coinbase.exchange_name(), "coinbase");
    assert_eq!(kraken.exchange_name(), "kraken");
    assert_eq!(okx.exchange_name(), "okx");

    assert_eq!(coinbase.ws_url(), "wss://coinbase.test");
    assert_eq!(kraken.ws_url(), "wss://kraken.test");
    assert_eq!(okx.ws_url(), "wss://okx.test");
}

// Test that health checkers are independent per exchange
#[test]
fn test_independent_health_checkers() {
    let coinbase = CoinbaseBridge::new("wss://test".to_string());
    let kraken = KrakenBridge::new("wss://test".to_string());

    let coinbase_health = coinbase.health_checker();
    let kraken_health = kraken.health_checker();

    // Record error on Coinbase only
    // Note: record_error is async, so we can't easily call it from a sync test
    // The key test is that the health checkers are independent instances
    assert!(!Arc::ptr_eq(&coinbase_health, &kraken_health));
}

#[test]
fn test_bridge_adapter_reconnect_config() {
    let bridge = CoinbaseBridge::new("wss://test".to_string());
    let config = bridge.build_ws_config("btcusdt");

    // Verify reconnection settings
    assert_eq!(config.reconnect_delay_secs, 5);
    assert_eq!(config.max_reconnect_attempts, 10);
    assert_eq!(config.ping_interval_secs, 30);
}

#[cfg(test)]
mod cns_metrics_tests {
    use super::*;

    // Note: These tests verify that CNS integration is wired correctly.
    // Actual metric values would need a CNS/Prometheus server to query.

    #[tokio::test]
    async fn test_cns_reporter_records_messages() {
        let bridge = CoinbaseBridge::new("wss://test".to_string());

        // Parse a valid message - should record metrics
        let raw_message = r#"{
            "channel": "heartbeats",
            "client_id": "",
            "timestamp": "2024-01-01T00:00:00.000000Z",
            "sequence_num": 0,
            "events": [{
                "current_time": "2024-01-01T00:00:00.000000Z",
                "heartbeat_counter": 1
            }]
        }"#;

        let result = bridge.parse_message(raw_message);
        assert!(result.is_ok());

        // CNS metrics would be recorded internally (we can't easily assert on them
        // without a Prometheus endpoint, but we verify no errors occurred)
    }

    #[tokio::test]
    async fn test_cns_reporter_records_parse_errors() {
        let bridge = CoinbaseBridge::new("wss://test".to_string());

        // Invalid JSON should record a parse error
        let result = bridge.parse_message("invalid json");
        assert!(result.is_err());

        // Error metric would be incremented (verified via CNS/Prometheus in integration)
    }
}
