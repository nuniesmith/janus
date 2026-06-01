//! Compatibility adapter: the Bybit API the forward event loop expects,
//! implemented on top of the published `exchange-apiws` crate.
//!
//! janus historically depended on an in-house `jflow-bybit-client`. That crate
//! has been retired in favour of `exchange-apiws` (which now ships a signed
//! Bybit v5 surface). This module re-exposes the exact types/methods the event
//! loop uses — `BybitTick`, `BybitTrade`, `WsMessage`, `BybitWebSocket`,
//! `BybitRestClient`, `OrderRequest`, `OrderSide`, `OrderType`,
//! `BybitCredentials` — so the swap needs no churn in the 4k-line event loop.
//!
//! - **REST / order entry** wraps [`exchange_apiws::BybitPrivateClient`].
//! - **Public market feed** drives [`exchange_apiws::BybitConnector`] through
//!   `run_feed`, translating `DataMessage::{OrderBook,Ticker,Trade}` into the
//!   janus-shaped `WsMessage` stream.

use std::sync::Arc;

use anyhow::Result;
use exchange_apiws::actors::{DataMessage, ExchangeConnector, TradeSide};
use exchange_apiws::bybit::BybitConnector;
use exchange_apiws::ws::{WsRunnerConfig, run_feed};
use exchange_apiws::{BybitCategory, BybitOrderRequest, BybitPrivateClient};
use tokio::sync::{mpsc, watch};

// ── Re-exported order-entry enums (janus names → exchange-apiws types) ────────

/// Order side — alias of the exchange-apiws type so existing
/// `OrderSide::Buy` / `OrderSide::Sell` call sites keep working.
pub use exchange_apiws::BybitOrderSide as OrderSide;
/// Order type — alias of the exchange-apiws type.
pub use exchange_apiws::BybitOrderType as OrderType;

// ── Credentials (janus constructs these positionally) ─────────────────────────

/// Bybit credentials with janus's historical `new(key, secret)` shape.
#[derive(Clone)]
pub struct BybitCredentials {
    key: String,
    secret: String,
}

impl BybitCredentials {
    /// Construct from key + secret (matches the old jflow-bybit-client API).
    pub fn new(key: String, secret: String) -> Self {
        Self { key, secret }
    }
}

// ── Market-data types (janus field shapes) ────────────────────────────────────

/// Best-bid/ask tick, in the shape `process_tick` consumes.
#[derive(Debug, Clone)]
pub struct BybitTick {
    pub symbol: String,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub timestamp: u64,
}

/// A public trade print.
#[derive(Debug, Clone)]
pub struct BybitTrade {
    pub symbol: String,
    pub side: String,
    pub size: f64,
    pub price: f64,
    pub timestamp: u64,
}

/// Stream messages delivered to the event loop. Only the variants the loop
/// matches on are populated; the private-feed variants remain for source
/// compatibility with the old match arms.
#[derive(Debug, Clone)]
pub enum WsMessage {
    Tick(BybitTick),
    Trade(BybitTrade),
    #[allow(dead_code)]
    Kline(serde_json::Value),
    OrderUpdate(serde_json::Value),
    PositionUpdate(serde_json::Value),
    ExecutionUpdate(serde_json::Value),
}

// ── Order request (janus literal shape) ───────────────────────────────────────

/// New-order request preserving the field set janus's event loop builds,
/// including the legacy `close_on_trigger` (ignored by exchange-apiws, which
/// has no such field).
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub category: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub qty: String,
    pub price: Option<String>,
    pub time_in_force: Option<String>,
    pub order_link_id: Option<String>,
    pub reduce_only: Option<bool>,
    pub close_on_trigger: Option<bool>,
}

impl OrderRequest {
    fn into_exchange(self) -> BybitOrderRequest {
        let tif = self.time_in_force.as_deref().map(|s| match s {
            "IOC" => exchange_apiws::BybitTimeInForce::IOC,
            "FOK" => exchange_apiws::BybitTimeInForce::FOK,
            "PostOnly" => exchange_apiws::BybitTimeInForce::PostOnly,
            _ => exchange_apiws::BybitTimeInForce::GTC,
        });
        BybitOrderRequest {
            category: self.category,
            symbol: self.symbol,
            side: self.side,
            order_type: self.order_type,
            qty: self.qty,
            price: self.price,
            time_in_force: tif,
            order_link_id: self.order_link_id,
            reduce_only: self.reduce_only,
        }
    }
}

/// Order-placement response (janus reads `.order_id`).
pub struct OrderResponse {
    pub order_id: String,
    #[allow(dead_code)]
    pub order_link_id: String,
}

// ── REST client (signed) ──────────────────────────────────────────────────────

/// Signed Bybit REST client, wrapping [`BybitPrivateClient`].
pub struct BybitRestClient {
    inner: BybitPrivateClient,
}

impl BybitRestClient {
    /// Build a signed client (mainnet unless `testnet`).
    pub fn new(creds: BybitCredentials, testnet: bool) -> Self {
        let inner = BybitPrivateClient::new(
            exchange_apiws::BybitCredentials::new(creds.key, creds.secret),
            testnet,
        )
        .expect("bybit private client");
        Self { inner }
    }

    /// Place an order. Mirrors the old `place_order(order)` (by value).
    pub async fn place_order(&self, order: OrderRequest) -> Result<OrderResponse> {
        let ack = self.inner.place_order(&order.into_exchange()).await?;
        Ok(OrderResponse {
            order_id: ack.order_id,
            order_link_id: ack.order_link_id,
        })
    }

    /// Cancel an order by id (category fixed to linear, as janus trades perps).
    #[allow(dead_code)]
    pub async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<()> {
        self.inner
            .cancel_order(BybitCategory::Linear, symbol, order_id)
            .await?;
        Ok(())
    }

    /// Positions for the linear category (raw JSON, as before).
    #[allow(dead_code)]
    pub async fn get_positions(&self, symbol: Option<&str>) -> Result<serde_json::Value> {
        Ok(self
            .inner
            .get_positions(BybitCategory::Linear, symbol)
            .await?)
    }

    /// Wallet balance for an account type (raw JSON, as before).
    #[allow(dead_code)]
    pub async fn get_balance(&self, account_type: &str) -> Result<serde_json::Value> {
        Ok(self.inner.get_wallet_balance(account_type).await?)
    }
}

// ── Public WebSocket feed ─────────────────────────────────────────────────────

/// Public Bybit market feed presenting the old `(ws, rx)` + `connect(subs)`
/// surface, backed by `exchange-apiws`'s `BybitConnector` + `run_feed`.
pub struct BybitWebSocket {
    testnet: bool,
    tx: mpsc::UnboundedSender<WsMessage>,
}

impl BybitWebSocket {
    /// Create a public feed. Returns the driver plus the receiver the event
    /// loop pulls `WsMessage`s from.
    pub fn new_public(testnet: bool) -> (Self, mpsc::UnboundedReceiver<WsMessage>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { testnet, tx }, rx)
    }

    /// Connect and stream until the feed terminates. `subscriptions` are janus's
    /// topic strings (e.g. `"orderbook.50.BTCUSDT"`); the symbol is parsed from
    /// the first one to build the connector.
    pub async fn connect(self, subscriptions: Vec<String>) -> Result<()> {
        // Translate `exchange-apiws` DataMessages into janus WsMessages.
        let (data_tx, mut data_rx) = mpsc::channel::<DataMessage>(1024);
        let category = BybitCategory::Linear;
        let bybit = BybitConnector::new(category, subscriptions.clone());
        // `ws_url` comes from the ExchangeConnector trait; read it before the
        // connector is moved into the Arc handed to `run_feed`.
        let ws_url = if self.testnet {
            "wss://stream-testnet.bybit.com/v5/linear".to_string()
        } else {
            bybit.ws_url().to_string()
        };
        let connector = Arc::new(bybit);
        let (_sd_tx, sd_rx) = watch::channel(false);

        // Forward-translate task.
        let out = self.tx.clone();
        tokio::spawn(async move {
            while let Some(msg) = data_rx.recv().await {
                match msg {
                    DataMessage::OrderBook(ob) => {
                        let bid = ob.bids.first().copied().unwrap_or([0.0, 0.0]);
                        let ask = ob.asks.first().copied().unwrap_or([0.0, 0.0]);
                        let tick = BybitTick {
                            symbol: ob.symbol,
                            bid_price: bid[0],
                            ask_price: ask[0],
                            bid_size: bid[1],
                            ask_size: ask[1],
                            last_price: (bid[0] + ask[0]) / 2.0,
                            timestamp: ob.exchange_ts.max(0) as u64,
                        };
                        let _ = out.send(WsMessage::Tick(tick));
                    }
                    DataMessage::Ticker(t) => {
                        let tick = BybitTick {
                            symbol: t.symbol,
                            bid_price: t.best_bid,
                            ask_price: t.best_ask,
                            bid_size: 0.0,
                            ask_size: 0.0,
                            last_price: t.price,
                            timestamp: t.exchange_ts.max(0) as u64,
                        };
                        let _ = out.send(WsMessage::Tick(tick));
                    }
                    DataMessage::Trade(tr) => {
                        let _ = out.send(WsMessage::Trade(BybitTrade {
                            symbol: tr.symbol,
                            side: match tr.side {
                                TradeSide::Buy => "Buy".to_string(),
                                TradeSide::Sell => "Sell".to_string(),
                            },
                            size: tr.amount,
                            price: tr.price,
                            timestamp: tr.exchange_ts.max(0) as u64,
                        }));
                    }
                    _ => {}
                }
            }
        });

        run_feed(
            ws_url,
            subscriptions,
            connector,
            data_tx,
            WsRunnerConfig::default(),
            sd_rx,
        )
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_request_maps_tif_strings() {
        let mk = |tif: &str| {
            OrderRequest {
                category: "linear".into(),
                symbol: "BTCUSDT".into(),
                side: OrderSide::Buy,
                order_type: OrderType::Limit,
                qty: "0.1".into(),
                price: Some("100".into()),
                time_in_force: Some(tif.into()),
                order_link_id: None,
                reduce_only: Some(false),
                close_on_trigger: Some(false),
            }
            .into_exchange()
        };
        assert_eq!(
            mk("IOC").time_in_force,
            Some(exchange_apiws::BybitTimeInForce::IOC)
        );
        assert_eq!(
            mk("GTC").time_in_force,
            Some(exchange_apiws::BybitTimeInForce::GTC)
        );
        // Unknown → GTC.
        assert_eq!(
            mk("weird").time_in_force,
            Some(exchange_apiws::BybitTimeInForce::GTC)
        );
    }
}
