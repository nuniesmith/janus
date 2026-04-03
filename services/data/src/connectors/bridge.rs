//! Bridge Adapters for JANUS Exchanges
//!
//! This module provides bridge adapters that wrap the new `janus-exchanges` crate
//! adapters to implement the legacy `ExchangeConnector` trait used by the data service.
//!
//! This allows a smooth migration path:
//! 1. Use the new unified exchange adapters from `janus-exchanges`
//! 2. Convert their `MarketDataEvent` output to legacy `DataMessage` format
//! 3. Wire CNS metrics and health monitoring
//! 4. Eventually migrate the entire pipeline to use `MarketDataEvent` end-to-end
//!
//! ## Architecture:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    WebSocketActor                            │
//! │                (expects ExchangeConnector)                   │
//! └───────────────────────┬─────────────────────────────────────┘
//!                         │
//!                         │ parse_message()
//!                         ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   BridgeAdapter                              │
//! │  (implements ExchangeConnector, wraps exchange adapter)      │
//! ├─────────────────────────────────────────────────────────────┤
//! │  • parse_message() → Vec<DataMessage>                       │
//! │  • Record CNS metrics (messages, latency, errors)           │
//! │  • Update health status                                      │
//! └───────────────────────┬─────────────────────────────────────┘
//!                         │
//!                         ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │            janus-exchanges Adapter                           │
//! │         (Coinbase, Kraken, OKX, etc.)                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │  • parse_message() → Vec<MarketDataEvent>                   │
//! └───────────────────────┬─────────────────────────────────────┘
//!                         │
//!                         ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  MarketDataEvent                             │
//! │         (Trade, OrderBook, Ticker, etc.)                     │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use anyhow::Result;
use janus_core::{MarketDataEvent, Side, Symbol};
use janus_exchanges::adapters::{
    coinbase::CoinbaseAdapter, kraken::KrakenAdapter, okx::OkxAdapter,
};
use janus_exchanges::{CNSReporter, HealthChecker};
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

use crate::actors::{
    CandleData, DataMessage, ExchangeConnector, TradeData, TradeSide, WebSocketConfig,
};

/// Bridge adapter for Coinbase
pub struct CoinbaseBridge {
    adapter: CoinbaseAdapter,
    cns_reporter: CNSReporter,
    health_checker: Arc<HealthChecker>,
    ws_url: String,
}

impl CoinbaseBridge {
    /// Create a new Coinbase bridge adapter
    pub fn new(ws_url: String) -> Self {
        let adapter = CoinbaseAdapter::with_url(ws_url.clone());
        let cns_reporter = CNSReporter::new("coinbase");
        let health_checker = Arc::new(HealthChecker::new());

        Self {
            adapter,
            cns_reporter,
            health_checker,
            ws_url,
        }
    }

    /// Get health checker reference
    pub fn health_checker(&self) -> Arc<HealthChecker> {
        Arc::clone(&self.health_checker)
    }
}

impl ExchangeConnector for CoinbaseBridge {
    fn exchange_name(&self) -> &str {
        "coinbase"
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        // Parse symbol (e.g., "btcusdt" -> "BTC-USD")
        let formatted_symbol = if symbol.to_lowercase().ends_with("usdt") {
            let base = symbol.strip_suffix("usdt").unwrap_or(symbol);
            format!("{}-USDT", base.to_uppercase())
        } else if symbol.to_lowercase().ends_with("usd") {
            let base = symbol.strip_suffix("usd").unwrap_or(symbol);
            format!("{}-USD", base.to_uppercase())
        } else {
            symbol.to_uppercase()
        };

        let symbol_obj =
            Symbol::from_exchange_format(&formatted_symbol, janus_core::Exchange::Coinbase)
                .unwrap_or_else(|| Symbol::new("BTC", "USD"));

        let subscription_msg = self.adapter.default_subscribe(&symbol_obj);

        WebSocketConfig {
            url: self.ws_url.clone(),
            exchange: "coinbase".to_string(),
            symbol: formatted_symbol,
            subscription_msg: Some(subscription_msg),
            ping_interval_secs: 30,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }

    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>> {
        let start = Instant::now();

        // Parse using the adapter
        let events = match self.adapter.parse_message(raw) {
            Ok(events) => {
                if !events.is_empty() {
                    self.cns_reporter
                        .record_message("market_data", &format!("{} events", events.len()));
                }
                events
            }
            Err(e) => {
                self.cns_reporter
                    .record_parse_error(&format!("parse_error: {}", e));
                // Note: health_checker methods are async, so we spawn a task
                let hc = Arc::clone(&self.health_checker);
                let err_msg = e.to_string();
                tokio::spawn(async move {
                    hc.record_error("coinbase", err_msg).await;
                });
                return Err(e);
            }
        };

        // Record latency
        let latency = start.elapsed();
        self.cns_reporter.record_latency("market_data", latency);

        // Convert MarketDataEvent -> DataMessage
        let mut messages = Vec::new();

        for event in events {
            match convert_market_data_event(event, "coinbase") {
                Some(msg) => messages.push(msg),
                None => {
                    debug!("Coinbase: Skipped unsupported event type");
                }
            }
        }

        // Update health (async, spawn a task)
        let hc = Arc::clone(&self.health_checker);
        tokio::spawn(async move {
            hc.record_message("coinbase", None).await;
        });

        Ok(messages)
    }

    fn ws_url(&self) -> &str {
        &self.ws_url
    }

    fn subscription_message(&self, symbol: &str) -> Option<String> {
        let config = self.build_ws_config(symbol);
        config.subscription_msg
    }
}

/// Bridge adapter for Kraken
pub struct KrakenBridge {
    adapter: KrakenAdapter,
    cns_reporter: CNSReporter,
    health_checker: Arc<HealthChecker>,
    ws_url: String,
}

impl KrakenBridge {
    /// Create a new Kraken bridge adapter
    pub fn new(ws_url: String) -> Self {
        let adapter = KrakenAdapter::with_url(ws_url.clone());
        let cns_reporter = CNSReporter::new("kraken");
        let health_checker = Arc::new(HealthChecker::new());

        Self {
            adapter,
            cns_reporter,
            health_checker,
            ws_url,
        }
    }

    /// Get health checker reference
    pub fn health_checker(&self) -> Arc<HealthChecker> {
        Arc::clone(&self.health_checker)
    }
}

impl ExchangeConnector for KrakenBridge {
    fn exchange_name(&self) -> &str {
        "kraken"
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        // Parse symbol (e.g., "btcusdt" -> "BTC/USDT")
        let formatted_symbol = if symbol.to_lowercase().ends_with("usdt") {
            let base = symbol.strip_suffix("usdt").unwrap_or(symbol);
            format!("{}/USDT", base.to_uppercase())
        } else if symbol.to_lowercase().ends_with("usd") {
            let base = symbol.strip_suffix("usd").unwrap_or(symbol);
            format!("{}/USD", base.to_uppercase())
        } else {
            symbol.to_uppercase()
        };

        let symbol_obj =
            Symbol::from_exchange_format(&formatted_symbol, janus_core::Exchange::Kraken)
                .unwrap_or_else(|| Symbol::new("BTC", "USD"));

        let subscription_msg = self.adapter.default_subscribe(&symbol_obj);

        WebSocketConfig {
            url: self.ws_url.clone(),
            exchange: "kraken".to_string(),
            symbol: formatted_symbol,
            subscription_msg: Some(subscription_msg),
            ping_interval_secs: 30,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }

    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>> {
        let start = Instant::now();

        // Parse using the adapter
        let events = match self.adapter.parse_message(raw) {
            Ok(events) => {
                if !events.is_empty() {
                    self.cns_reporter
                        .record_message("market_data", &format!("{} events", events.len()));
                }
                events
            }
            Err(e) => {
                self.cns_reporter
                    .record_parse_error(&format!("parse_error: {}", e));
                let hc = Arc::clone(&self.health_checker);
                let err_msg = e.to_string();
                tokio::spawn(async move {
                    hc.record_error("kraken", err_msg).await;
                });
                return Err(e);
            }
        };

        // Record latency
        let latency = start.elapsed();
        self.cns_reporter.record_latency("market_data", latency);

        // Convert MarketDataEvent -> DataMessage
        let mut messages = Vec::new();

        for event in events {
            match convert_market_data_event(event, "kraken") {
                Some(msg) => messages.push(msg),
                None => {
                    debug!("Kraken: Skipped unsupported event type");
                }
            }
        }

        // Update health (async, spawn a task)
        let hc = Arc::clone(&self.health_checker);
        tokio::spawn(async move {
            hc.record_message("kraken", None).await;
        });

        Ok(messages)
    }

    fn ws_url(&self) -> &str {
        &self.ws_url
    }

    fn subscription_message(&self, symbol: &str) -> Option<String> {
        let config = self.build_ws_config(symbol);
        config.subscription_msg
    }
}

/// Bridge adapter for OKX
pub struct OkxBridge {
    adapter: OkxAdapter,
    cns_reporter: CNSReporter,
    health_checker: Arc<HealthChecker>,
    ws_url: String,
}

impl OkxBridge {
    /// Create a new OKX bridge adapter
    pub fn new(ws_url: String) -> Self {
        let adapter = OkxAdapter::with_url(ws_url.clone());
        let cns_reporter = CNSReporter::new("okx");
        let health_checker = Arc::new(HealthChecker::new());

        Self {
            adapter,
            cns_reporter,
            health_checker,
            ws_url,
        }
    }

    /// Get health checker reference
    pub fn health_checker(&self) -> Arc<HealthChecker> {
        Arc::clone(&self.health_checker)
    }
}

impl ExchangeConnector for OkxBridge {
    fn exchange_name(&self) -> &str {
        "okx"
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        // Parse symbol (e.g., "btcusdt" -> "BTC-USDT")
        let formatted_symbol = if symbol.to_lowercase().ends_with("usdt") {
            let base = symbol.strip_suffix("usdt").unwrap_or(symbol);
            format!("{}-USDT", base.to_uppercase())
        } else if symbol.to_lowercase().ends_with("usd") {
            let base = symbol.strip_suffix("usd").unwrap_or(symbol);
            format!("{}-USD", base.to_uppercase())
        } else {
            symbol.to_uppercase()
        };

        let symbol_obj = Symbol::from_exchange_format(&formatted_symbol, janus_core::Exchange::Okx)
            .unwrap_or_else(|| Symbol::new("BTC", "USD"));

        let subscription_msg = self.adapter.default_subscribe(&symbol_obj);

        WebSocketConfig {
            url: self.ws_url.clone(),
            exchange: "okx".to_string(),
            symbol: formatted_symbol,
            subscription_msg: Some(subscription_msg),
            ping_interval_secs: 30,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }

    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>> {
        let start = Instant::now();

        // Parse using the adapter
        let events = match self.adapter.parse_message(raw) {
            Ok(events) => {
                if !events.is_empty() {
                    self.cns_reporter
                        .record_message("market_data", &format!("{} events", events.len()));
                }
                events
            }
            Err(e) => {
                self.cns_reporter
                    .record_parse_error(&format!("parse_error: {}", e));
                let hc = Arc::clone(&self.health_checker);
                let err_msg = e.to_string();
                tokio::spawn(async move {
                    hc.record_error("okx", err_msg).await;
                });
                return Err(e);
            }
        };

        // Record latency
        let latency = start.elapsed();
        self.cns_reporter.record_latency("market_data", latency);

        // Convert MarketDataEvent -> DataMessage
        let mut messages = Vec::new();

        for event in events {
            match convert_market_data_event(event, "okx") {
                Some(msg) => messages.push(msg),
                None => {
                    debug!("OKX: Skipped unsupported event type");
                }
            }
        }

        // Update health (async, spawn a task)
        let hc = Arc::clone(&self.health_checker);
        tokio::spawn(async move {
            hc.record_message("okx", None).await;
        });

        Ok(messages)
    }

    fn ws_url(&self) -> &str {
        &self.ws_url
    }

    fn subscription_message(&self, symbol: &str) -> Option<String> {
        let config = self.build_ws_config(symbol);
        config.subscription_msg
    }
}

/// Convert MarketDataEvent to legacy DataMessage
fn convert_market_data_event(event: MarketDataEvent, exchange: &str) -> Option<DataMessage> {
    match event {
        MarketDataEvent::Trade(trade) => {
            let side = match trade.side {
                Side::Buy => TradeSide::Buy,
                Side::Sell => TradeSide::Sell,
            };

            Some(DataMessage::Trade(TradeData {
                symbol: trade.symbol.to_string(),
                exchange: exchange.to_string(),
                side,
                price: trade.price.to_string().parse().unwrap_or(0.0),
                amount: trade.quantity.to_string().parse().unwrap_or(0.0),
                exchange_ts: trade.timestamp / 1000, // Convert µs to ms
                receipt_ts: chrono::Utc::now().timestamp_millis(),
                trade_id: trade.trade_id,
            }))
        }

        MarketDataEvent::Kline(kline) => {
            // Only emit closed candles
            if !kline.is_closed {
                return None;
            }

            Some(DataMessage::Candle(CandleData {
                symbol: kline.symbol.to_string(),
                exchange: exchange.to_string(),
                open_time: kline.open_time / 1000, // Convert µs to ms
                close_time: kline.close_time / 1000,
                open: kline.open.to_string().parse().unwrap_or(0.0),
                high: kline.high.to_string().parse().unwrap_or(0.0),
                low: kline.low.to_string().parse().unwrap_or(0.0),
                close: kline.close.to_string().parse().unwrap_or(0.0),
                volume: kline.volume.to_string().parse().unwrap_or(0.0),
                interval: kline.interval.clone(),
            }))
        }

        MarketDataEvent::OrderBook(_) => {
            // Order book events are not supported in the legacy DataMessage format
            // These would be handled separately in a production system
            None
        }

        MarketDataEvent::Ticker(_) => {
            // Ticker events are not supported in the legacy DataMessage format
            None
        }

        MarketDataEvent::Liquidation(_) => {
            // Liquidation events are not supported in the legacy DataMessage format
            None
        }

        MarketDataEvent::FundingRate(_) => {
            // Funding rate events are not supported in the legacy DataMessage format
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coinbase_bridge_creation() {
        let bridge = CoinbaseBridge::new("wss://test.coinbase.com".to_string());
        assert_eq!(bridge.exchange_name(), "coinbase");
        assert_eq!(bridge.ws_url(), "wss://test.coinbase.com");
    }

    #[test]
    fn test_kraken_bridge_creation() {
        let bridge = KrakenBridge::new("wss://test.kraken.com".to_string());
        assert_eq!(bridge.exchange_name(), "kraken");
        assert_eq!(bridge.ws_url(), "wss://test.kraken.com");
    }

    #[test]
    fn test_okx_bridge_creation() {
        let bridge = OkxBridge::new("wss://test.okx.com".to_string());
        assert_eq!(bridge.exchange_name(), "okx");
        assert_eq!(bridge.ws_url(), "wss://test.okx.com");
    }

    #[test]
    fn test_symbol_formatting_coinbase() {
        let bridge = CoinbaseBridge::new("wss://test".to_string());
        let config = bridge.build_ws_config("btcusdt");
        assert_eq!(config.symbol, "BTC-USDT");
    }

    #[test]
    fn test_symbol_formatting_kraken() {
        let bridge = KrakenBridge::new("wss://test".to_string());
        let config = bridge.build_ws_config("btcusdt");
        assert_eq!(config.symbol, "BTC/USDT");
    }

    #[test]
    fn test_symbol_formatting_okx() {
        let bridge = OkxBridge::new("wss://test".to_string());
        let config = bridge.build_ws_config("btcusdt");
        assert_eq!(config.symbol, "BTC-USDT");
    }
}
