//! Bridge adapters wiring the published `exchange-apiws` connectors into the
//! data service's legacy `ExchangeConnector` trait.
//!
//! Each bridge (`CoinbaseBridge` / `KrakenBridge` / `OkxBridge`) wraps the
//! corresponding `exchange_apiws` connector for market-data parsing, layers
//! janus-side CNS metrics + health monitoring on top, and converts the unified
//! `exchange_apiws::DataMessage` into the data service's local `DataMessage`.
//!
//! Previously these wrapped `jflow-exchanges`'s in-house adapters; that
//! duplicate parsing has been retired in favour of the published crate
//! (`jflow-exchanges` is now used only for `CNSReporter` + `HealthChecker`).
//!
//! ```text
//!   WebSocketActor ──(parse_message)──► <Exchange>Bridge
//!                                          │  CNS metrics + health
//!                                          ▼
//!                          exchange_apiws::<Exchange>Connector
//!                                          │  Vec<exchange_apiws::DataMessage>
//!                                          ▼
//!                          convert_ea_message → Vec<DataMessage>
//! ```

use anyhow::Result;
// Market-data parsing now comes from the published `exchange-apiws` connectors;
// CNS metrics + health monitoring remain janus-side (jflow-exchanges).
use exchange_apiws::actors::{DataMessage as EaDataMessage, ExchangeConnector as EaConnector};
use exchange_apiws::{
    CoinbaseChannel, CoinbaseConnector, KrakenConnector, OkxChannel, OkxConnector,
};
use janus_exchanges::{CNSReporter, HealthChecker};
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

use crate::actors::{
    CandleData, DataMessage, ExchangeConnector, TradeData, TradeSide, WebSocketConfig,
};

/// Format a janus symbol (e.g. `"btcusdt"`) into an exchange product id using
/// `sep` between base and quote (`"-"` for Coinbase/OKX, `"/"` for Kraken).
fn format_product(symbol: &str, sep: &str) -> String {
    let lower = symbol.to_lowercase();
    if let Some(base) = lower.strip_suffix("usdt") {
        format!("{}{sep}USDT", base.to_uppercase())
    } else if let Some(base) = lower.strip_suffix("usd") {
        format!("{}{sep}USD", base.to_uppercase())
    } else {
        symbol.to_uppercase()
    }
}

/// Bridge adapter for Coinbase
pub struct CoinbaseBridge {
    cns_reporter: CNSReporter,
    health_checker: Arc<HealthChecker>,
    ws_url: String,
}

impl CoinbaseBridge {
    /// Create a new Coinbase bridge adapter
    pub fn new(ws_url: String) -> Self {
        let cns_reporter = CNSReporter::new("coinbase");
        let health_checker = Arc::new(HealthChecker::new());

        Self {
            cns_reporter,
            health_checker,
            ws_url,
        }
    }

    /// Get health checker reference
    pub fn health_checker(&self) -> Arc<HealthChecker> {
        Arc::clone(&self.health_checker)
    }

    /// Build a published exchange-apiws connector for `product` (trades + ticker).
    fn connector(&self, product: &str) -> CoinbaseConnector {
        CoinbaseConnector::with_url(
            self.ws_url.clone(),
            vec![product.to_string()],
            vec![CoinbaseChannel::MarketTrades],
        )
    }
}

impl ExchangeConnector for CoinbaseBridge {
    fn exchange_name(&self) -> &str {
        "coinbase"
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        let formatted_symbol = format_product(symbol, "-");
        let subscription_msg = self
            .connector(&formatted_symbol)
            .subscription_message(&formatted_symbol);

        WebSocketConfig {
            url: self.ws_url.clone(),
            exchange: "coinbase".to_string(),
            symbol: formatted_symbol,
            subscription_msg,
            ping_interval_secs: 30,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }

    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>> {
        let start = Instant::now();
        let messages = parse_via_connector(
            &self.connector(""),
            raw,
            "coinbase",
            &self.cns_reporter,
            &self.health_checker,
            start,
        )?;
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
    cns_reporter: CNSReporter,
    health_checker: Arc<HealthChecker>,
    ws_url: String,
}

impl KrakenBridge {
    /// Create a new Kraken bridge adapter
    pub fn new(ws_url: String) -> Self {
        let cns_reporter = CNSReporter::new("kraken");
        let health_checker = Arc::new(HealthChecker::new());

        Self {
            cns_reporter,
            health_checker,
            ws_url,
        }
    }

    /// Get health checker reference
    pub fn health_checker(&self) -> Arc<HealthChecker> {
        Arc::clone(&self.health_checker)
    }

    fn connector(&self) -> KrakenConnector {
        KrakenConnector::with_url(self.ws_url.clone())
    }
}

impl ExchangeConnector for KrakenBridge {
    fn exchange_name(&self) -> &str {
        "kraken"
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        let formatted_symbol = format_product(symbol, "/");
        // Kraken v2 subscribes by pair list via a static frame builder.
        let subscription_msg = KrakenConnector::trade_subscription(&[&formatted_symbol]);

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
        let messages = parse_via_connector(
            &self.connector(),
            raw,
            "kraken",
            &self.cns_reporter,
            &self.health_checker,
            start,
        )?;
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
    cns_reporter: CNSReporter,
    health_checker: Arc<HealthChecker>,
    ws_url: String,
}

impl OkxBridge {
    /// Create a new OKX bridge adapter
    pub fn new(ws_url: String) -> Self {
        let cns_reporter = CNSReporter::new("okx");
        let health_checker = Arc::new(HealthChecker::new());

        Self {
            cns_reporter,
            health_checker,
            ws_url,
        }
    }

    /// Get health checker reference
    pub fn health_checker(&self) -> Arc<HealthChecker> {
        Arc::clone(&self.health_checker)
    }

    fn connector(&self, inst_id: &str) -> OkxConnector {
        OkxConnector::with_url(self.ws_url.clone(), vec![OkxChannel::trades(inst_id)])
    }
}

impl ExchangeConnector for OkxBridge {
    fn exchange_name(&self) -> &str {
        "okx"
    }

    fn build_ws_config(&self, symbol: &str) -> WebSocketConfig {
        let formatted_symbol = format_product(symbol, "-");
        let subscription_msg = self
            .connector(&formatted_symbol)
            .subscription_message(&formatted_symbol);

        WebSocketConfig {
            url: self.ws_url.clone(),
            exchange: "okx".to_string(),
            symbol: formatted_symbol,
            subscription_msg,
            ping_interval_secs: 30,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 10,
        }
    }

    fn parse_message(&self, raw: &str) -> Result<Vec<DataMessage>> {
        let start = Instant::now();
        let messages = parse_via_connector(
            &self.connector(""),
            raw,
            "okx",
            &self.cns_reporter,
            &self.health_checker,
            start,
        )?;
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

/// Shared parse path: run an exchange-apiws connector's `parse_message`, record
/// CNS metrics + health, and convert the unified `DataMessage` into janus's
/// local `DataMessage`. Used by all three bridges.
fn parse_via_connector(
    connector: &dyn EaConnector,
    raw: &str,
    exchange: &str,
    cns_reporter: &CNSReporter,
    health_checker: &Arc<HealthChecker>,
    start: Instant,
) -> Result<Vec<DataMessage>> {
    let events = match connector.parse_message(raw) {
        Ok(events) => {
            if !events.is_empty() {
                cns_reporter.record_message("market_data", &format!("{} events", events.len()));
            }
            events
        }
        Err(e) => {
            cns_reporter.record_parse_error(&format!("parse_error: {e}"));
            let hc = Arc::clone(health_checker);
            let ex = exchange.to_string();
            let err_msg = e.to_string();
            tokio::spawn(async move {
                hc.record_error(&ex, err_msg).await;
            });
            return Err(anyhow::anyhow!("{e}"));
        }
    };

    cns_reporter.record_latency("market_data", start.elapsed());

    let mut messages = Vec::new();
    for event in events {
        match convert_ea_message(event, exchange) {
            Some(msg) => messages.push(msg),
            None => debug!("{exchange}: skipped unsupported event type"),
        }
    }

    let hc = Arc::clone(health_checker);
    let ex = exchange.to_string();
    tokio::spawn(async move {
        hc.record_message(&ex, None).await;
    });

    Ok(messages)
}

/// Convert an exchange-apiws `DataMessage` into janus's legacy `DataMessage`.
/// Trades + candles map across; ticker / order-book frames are dropped (the
/// legacy format the data pipeline consumes doesn't carry them — same as the
/// previous adapter behaviour).
fn convert_ea_message(event: EaDataMessage, exchange: &str) -> Option<DataMessage> {
    match event {
        EaDataMessage::Trade(trade) => {
            let side = match trade.side {
                exchange_apiws::actors::TradeSide::Buy => TradeSide::Buy,
                exchange_apiws::actors::TradeSide::Sell => TradeSide::Sell,
            };
            Some(DataMessage::Trade(TradeData {
                symbol: trade.symbol,
                exchange: exchange.to_string(),
                side,
                price: trade.price,
                amount: trade.amount,
                exchange_ts: trade.exchange_ts,
                receipt_ts: trade.receipt_ts,
                trade_id: trade.trade_id,
            }))
        }
        EaDataMessage::Candle(c) => {
            // Match the legacy behaviour: only emit fully-closed bars.
            if !c.is_closed {
                return None;
            }
            Some(DataMessage::Candle(CandleData {
                symbol: c.symbol,
                exchange: exchange.to_string(),
                // exchange-apiws candles carry only the open timestamp; the
                // legacy close_time is approximated by the receipt time.
                open_time: c.open_ts,
                close_time: c.receipt_ts,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
                volume: c.volume,
                interval: c.interval,
            }))
        }
        // Ticker / OrderBook / FundingRate / private events are not part of the
        // legacy DataMessage surface the data pipeline consumes.
        _ => None,
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
