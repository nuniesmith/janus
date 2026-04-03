//! OKX Exchange Adapter
//!
//! Implements WebSocket connection to OKX for real-time market data.
//!
//! ## API Documentation:
//! - WebSocket API: https://www.okx.com/docs-v5/en/#websocket-api
//! - Public Channels: https://www.okx.com/docs-v5/en/#websocket-api-public-channel
//!
//! ## WebSocket Endpoint:
//! - Production Public: wss://ws.okx.com:8443/ws/v5/public
//! - Production Private: wss://ws.okx.com:8443/ws/v5/private
//! - AWS Public: wss://wsaws.okx.com:8443/ws/v5/public
//!
//! ## Supported Channels:
//! - `trades`: Real-time trade executions
//! - `tickers`: Ticker channel (24hr statistics)
//! - `books`: Order book channel (books for incremental, books5/books-l2-tbt for snapshot)
//! - `bbo-tbt`: Best bid/offer tick-by-tick
//! - `candle1m`: 1-minute candlestick data
//! - `funding-rate`: Funding rate for perpetual swaps
//! - `liquidation-orders`: Liquidation orders
//!
//! ## Symbol Format:
//! OKX uses "BTC-USDT" format for spot, "BTC-USDT-SWAP" for perpetual futures.
//!
//! ## Message Format (Trade):
//! ```json
//! {
//!   "arg": {
//!     "channel": "trades",
//!     "instId": "BTC-USDT"
//!   },
//!   "data": [{
//!     "instId": "BTC-USDT",
//!     "tradeId": "123456789",
//!     "px": "50000.0",
//!     "sz": "0.001",
//!     "side": "buy",
//!     "ts": "1609459200123"
//!   }]
//! }
//! ```
//!
//! ## Message Format (Ticker):
//! ```json
//! {
//!   "arg": {
//!     "channel": "tickers",
//!     "instId": "BTC-USDT"
//!   },
//!   "data": [{
//!     "instId": "BTC-USDT",
//!     "last": "50000.0",
//!     "lastSz": "0.001",
//!     "askPx": "50010.0",
//!     "askSz": "1.5",
//!     "bidPx": "49990.0",
//!     "bidSz": "2.0",
//!     "open24h": "49000.0",
//!     "high24h": "51000.0",
//!     "low24h": "48000.0",
//!     "volCcy24h": "1000000.0",
//!     "vol24h": "20.5",
//!     "ts": "1609459200123"
//!   }]
//! }
//! ```

use anyhow::{Context, Result};
use janus_core::{
    Exchange, FundingRateEvent, LiquidationEvent, MarketDataEvent, Side, Symbol, TradeEvent,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;
use tracing::{debug, trace, warn};

/// OKX exchange adapter
pub struct OkxAdapter {
    ws_url: String,
    inst_type: OkxInstType,
}

impl OkxAdapter {
    /// Create a new OKX adapter for spot markets
    pub fn new() -> Self {
        Self {
            ws_url: "wss://ws.okx.com:8443/ws/v5/public".to_string(),
            inst_type: OkxInstType::Spot,
        }
    }

    /// Create adapter for specific instrument type
    pub fn with_inst_type(inst_type: OkxInstType) -> Self {
        Self {
            ws_url: "wss://ws.okx.com:8443/ws/v5/public".to_string(),
            inst_type,
        }
    }

    /// Create adapter with custom WebSocket URL (for testing)
    pub fn with_url(ws_url: String) -> Self {
        Self {
            ws_url,
            inst_type: OkxInstType::Spot,
        }
    }

    /// Get the WebSocket URL
    pub fn ws_url(&self) -> &str {
        &self.ws_url
    }

    /// Build subscription message for a symbol
    pub fn subscribe_message(&self, symbol: &Symbol, channels: &[OkxChannel]) -> String {
        let inst_id = self.format_inst_id(symbol);

        let args: Vec<OkxSubscriptionArg> = channels
            .iter()
            .map(|channel| OkxSubscriptionArg {
                channel: channel.to_string(),
                inst_id: inst_id.clone(),
            })
            .collect();

        let msg = OkxSubscribeMessage {
            op: "subscribe".to_string(),
            args,
        };

        serde_json::to_string(&msg).unwrap_or_else(|_| "{}".to_string())
    }

    /// Build default subscription message (trades + tickers)
    pub fn default_subscribe(&self, symbol: &Symbol) -> String {
        self.subscribe_message(symbol, &[OkxChannel::Trades, OkxChannel::Tickers])
    }

    /// Parse WebSocket message into market data events
    pub fn parse_message(&self, raw: &str) -> Result<Vec<MarketDataEvent>> {
        trace!("OKX: Parsing message: {}", raw);

        // Try to parse as JSON
        let value: Value =
            serde_json::from_str(raw).context("Failed to parse OKX message as JSON")?;

        // OKX messages have different structures:
        // 1. Data messages: {"arg": {...}, "data": [...]}
        // 2. Event messages: {"event": "...", ...}

        if let Some(event) = value.get("event").and_then(|e| e.as_str()) {
            return self.parse_event_message(event, &value);
        }

        if value.get("arg").is_some() && value.get("data").is_some() {
            return self.parse_data_message(&value);
        }

        warn!("OKX: Unknown message format");
        Ok(vec![])
    }

    /// Parse event messages (subscription confirmations, errors, etc.)
    fn parse_event_message(&self, event: &str, value: &Value) -> Result<Vec<MarketDataEvent>> {
        match event {
            "subscribe" => {
                let arg = value.get("arg");
                debug!("OKX: Subscription confirmed: {:?}", arg);
                Ok(vec![])
            }
            "error" => {
                let msg = value
                    .get("msg")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown error");
                warn!("OKX: Error: {}", msg);
                Ok(vec![])
            }
            _ => {
                debug!("OKX: Unknown event: {}", event);
                Ok(vec![])
            }
        }
    }

    /// Parse data messages
    fn parse_data_message(&self, value: &Value) -> Result<Vec<MarketDataEvent>> {
        let arg = value.get("arg").context("Missing 'arg' field")?;
        let channel = arg
            .get("channel")
            .and_then(|c| c.as_str())
            .context("Missing channel in arg")?;

        let data = value
            .get("data")
            .and_then(|d| d.as_array())
            .context("Missing or invalid 'data' field")?;

        match channel {
            "trades" => self.parse_trades(data),
            "tickers" => self.parse_tickers(data),
            "books" | "books5" | "books-l2-tbt" => self.parse_books(data),
            "bbo-tbt" => self.parse_bbo(data),
            "funding-rate" => self.parse_funding_rate(data),
            "liquidation-orders" => self.parse_liquidations(data),
            _ if channel.starts_with("candle") => self.parse_candles(data),
            _ => {
                debug!("OKX: Unknown channel: {}", channel);
                Ok(vec![])
            }
        }
    }

    /// Parse trade data
    fn parse_trades(&self, data: &[Value]) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();

        for item in data {
            let inst_id = item
                .get("instId")
                .and_then(|i| i.as_str())
                .context("Missing instId in trade")?;

            let symbol = self.parse_inst_id(inst_id)?;

            let price = item
                .get("px")
                .and_then(|p| p.as_str())
                .and_then(|s| Decimal::from_str(s).ok())
                .context("Failed to parse trade price")?;

            let quantity = item
                .get("sz")
                .and_then(|s| s.as_str())
                .and_then(|s| Decimal::from_str(s).ok())
                .context("Failed to parse trade size")?;

            let side_str = item
                .get("side")
                .and_then(|s| s.as_str())
                .context("Missing trade side")?;

            let side = match side_str {
                "buy" => Side::Buy,
                "sell" => Side::Sell,
                _ => {
                    warn!("OKX: Unknown side: {}", side_str);
                    continue;
                }
            };

            let timestamp_str = item
                .get("ts")
                .and_then(|t| t.as_str())
                .context("Missing trade timestamp")?;

            let timestamp_ms: i64 = timestamp_str
                .parse()
                .context("Failed to parse trade timestamp")?;

            let timestamp_micros = timestamp_ms * 1000;

            let trade_id = item
                .get("tradeId")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();

            let event = TradeEvent::new(
                Exchange::Okx,
                symbol,
                timestamp_micros,
                price,
                quantity,
                side,
                trade_id,
            );

            events.push(MarketDataEvent::Trade(event));
        }

        Ok(events)
    }

    /// Parse ticker data
    fn parse_tickers(&self, data: &[Value]) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();

        for item in data {
            let inst_id = item
                .get("instId")
                .and_then(|i| i.as_str())
                .context("Missing instId in ticker")?;

            let symbol = self.parse_inst_id(inst_id)?;

            let last_price = item
                .get("last")
                .and_then(|p| p.as_str())
                .and_then(|s| Decimal::from_str(s).ok())
                .unwrap_or(Decimal::ZERO);

            let best_bid = item
                .get("bidPx")
                .and_then(|p| p.as_str())
                .and_then(|s| Decimal::from_str(s).ok());

            let best_ask = item
                .get("askPx")
                .and_then(|p| p.as_str())
                .and_then(|s| Decimal::from_str(s).ok());

            let volume_24h = item
                .get("vol24h")
                .and_then(|v| v.as_str())
                .and_then(|s| Decimal::from_str(s).ok())
                .unwrap_or(Decimal::ZERO);

            let quote_volume_24h = item
                .get("volCcy24h")
                .and_then(|v| v.as_str())
                .and_then(|s| Decimal::from_str(s).ok())
                .unwrap_or(Decimal::ZERO);

            let high_24h = item
                .get("high24h")
                .and_then(|h| h.as_str())
                .and_then(|s| Decimal::from_str(s).ok());

            let low_24h = item
                .get("low24h")
                .and_then(|l| l.as_str())
                .and_then(|s| Decimal::from_str(s).ok());

            let open_24h = item
                .get("open24h")
                .and_then(|o| o.as_str())
                .and_then(|s| Decimal::from_str(s).ok());

            // Calculate price change percentage
            let price_change_pct_24h = if let Some(open) = open_24h {
                if open > Decimal::ZERO {
                    Some(((last_price - open) / open) * Decimal::from(100))
                } else {
                    None
                }
            } else {
                None
            };

            let timestamp_str = item.get("ts").and_then(|t| t.as_str()).unwrap_or("0");

            let timestamp_ms: i64 = timestamp_str.parse().unwrap_or(0);
            let timestamp_micros = timestamp_ms * 1000;

            let ticker_event = janus_core::TickerEvent {
                exchange: Exchange::Okx,
                symbol,
                timestamp: timestamp_micros,
                last_price,
                best_bid,
                best_ask,
                volume_24h,
                quote_volume_24h,
                price_change_24h: None,
                price_change_pct_24h,
                high_24h,
                low_24h,
            };

            events.push(MarketDataEvent::Ticker(ticker_event));
        }

        Ok(events)
    }

    /// Parse order book data
    fn parse_books(&self, data: &[Value]) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();

        for item in data {
            let inst_id = item
                .get("instId")
                .and_then(|i| i.as_str())
                .context("Missing instId in book")?;

            let symbol = self.parse_inst_id(inst_id)?;

            // Parse bids
            let mut bids = Vec::new();
            if let Some(bids_array) = item.get("bids").and_then(|b| b.as_array()) {
                for bid_array in bids_array {
                    if let Some(bid) = bid_array.as_array()
                        && bid.len() >= 2
                        && let (Some(price_str), Some(volume_str)) =
                            (bid[0].as_str(), bid[1].as_str())
                        && let (Ok(price), Ok(volume)) =
                            (Decimal::from_str(price_str), Decimal::from_str(volume_str))
                        && volume > Decimal::ZERO
                    {
                        bids.push(janus_core::PriceLevel::new(price, volume));
                    }
                }
            }

            // Parse asks
            let mut asks = Vec::new();
            if let Some(asks_array) = item.get("asks").and_then(|a| a.as_array()) {
                for ask_array in asks_array {
                    if let Some(ask) = ask_array.as_array()
                        && ask.len() >= 2
                        && let (Some(price_str), Some(volume_str)) =
                            (ask[0].as_str(), ask[1].as_str())
                        && let (Ok(price), Ok(volume)) =
                            (Decimal::from_str(price_str), Decimal::from_str(volume_str))
                        && volume > Decimal::ZERO
                    {
                        asks.push(janus_core::PriceLevel::new(price, volume));
                    }
                }
            }

            // Sort bids descending (highest first), asks ascending (lowest first)
            bids.sort_by(|a, b| b.price.cmp(&a.price));
            asks.sort_by(|a, b| a.price.cmp(&b.price));

            // Determine if this is a snapshot or update
            // OKX sends "action" field: "snapshot" or "update"
            let action = item.get("action").and_then(|a| a.as_str()).unwrap_or("");
            let is_snapshot = action == "snapshot" || action.is_empty();

            let timestamp_str = item.get("ts").and_then(|t| t.as_str()).unwrap_or("0");
            let timestamp_ms: i64 = timestamp_str.parse().unwrap_or(0);
            let timestamp_micros = timestamp_ms * 1000;

            // OKX provides sequence number (seqId)
            let sequence = item.get("seqId").and_then(|s| s.as_u64()).unwrap_or(0);

            let orderbook_event = janus_core::OrderBookEvent {
                exchange: Exchange::Okx,
                symbol,
                timestamp: timestamp_micros,
                sequence,
                is_snapshot,
                bids,
                asks,
            };

            events.push(MarketDataEvent::OrderBook(orderbook_event));
        }

        Ok(events)
    }

    /// Parse best bid/offer data (BBO top of book)
    fn parse_bbo(&self, data: &[Value]) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();

        for item in data {
            let inst_id = item
                .get("instId")
                .and_then(|i| i.as_str())
                .context("Missing instId in BBO")?;

            let symbol = self.parse_inst_id(inst_id)?;

            let mut bids = Vec::new();
            let mut asks = Vec::new();

            // BBO format: single best bid and ask
            if let Some(bids_array) = item.get("bids").and_then(|b| b.as_array())
                && let Some(bid) = bids_array.first().and_then(|b| b.as_array())
                && bid.len() >= 2
                && let (Some(price_str), Some(volume_str)) = (bid[0].as_str(), bid[1].as_str())
                && let (Ok(price), Ok(volume)) =
                    (Decimal::from_str(price_str), Decimal::from_str(volume_str))
                && volume > Decimal::ZERO
            {
                bids.push(janus_core::PriceLevel::new(price, volume));
            }

            if let Some(asks_array) = item.get("asks").and_then(|a| a.as_array())
                && let Some(ask) = asks_array.first().and_then(|a| a.as_array())
                && ask.len() >= 2
                && let (Some(price_str), Some(volume_str)) = (ask[0].as_str(), ask[1].as_str())
                && let (Ok(price), Ok(volume)) =
                    (Decimal::from_str(price_str), Decimal::from_str(volume_str))
                && volume > Decimal::ZERO
            {
                asks.push(janus_core::PriceLevel::new(price, volume));
            }

            let timestamp_str = item.get("ts").and_then(|t| t.as_str()).unwrap_or("0");
            let timestamp_ms: i64 = timestamp_str.parse().unwrap_or(0);
            let timestamp_micros = timestamp_ms * 1000;

            let sequence = item.get("seqId").and_then(|s| s.as_u64()).unwrap_or(0);

            let orderbook_event = janus_core::OrderBookEvent {
                exchange: Exchange::Okx,
                symbol,
                timestamp: timestamp_micros,
                sequence,
                is_snapshot: false, // BBO is always an update
                bids,
                asks,
            };

            events.push(MarketDataEvent::OrderBook(orderbook_event));
        }

        Ok(events)
    }

    /// Parse funding rate data
    fn parse_funding_rate(&self, data: &[Value]) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();

        for item in data {
            let inst_id = item
                .get("instId")
                .and_then(|i| i.as_str())
                .context("Missing instId in funding rate")?;

            let symbol = self.parse_inst_id(inst_id)?;

            let rate = item
                .get("fundingRate")
                .and_then(|r| r.as_str())
                .and_then(|s| Decimal::from_str(s).ok())
                .context("Failed to parse funding rate")?;

            let next_funding_time = item
                .get("nextFundingTime")
                .and_then(|t| t.as_str())
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);

            let timestamp_str = item
                .get("fundingTime")
                .and_then(|t| t.as_str())
                .unwrap_or("0");

            let timestamp_ms: i64 = timestamp_str.parse().unwrap_or(0);
            let timestamp_micros = timestamp_ms * 1000;

            let event = FundingRateEvent {
                exchange: Exchange::Okx,
                symbol,
                timestamp: timestamp_micros,
                rate,
                next_funding_time,
            };

            events.push(MarketDataEvent::FundingRate(event));
        }

        Ok(events)
    }

    /// Parse liquidation data
    fn parse_liquidations(&self, data: &[Value]) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();

        for item in data {
            let inst_id = item
                .get("instId")
                .and_then(|i| i.as_str())
                .context("Missing instId in liquidation")?;

            let symbol = self.parse_inst_id(inst_id)?;

            let price = item
                .get("bkPx")
                .and_then(|p| p.as_str())
                .and_then(|s| Decimal::from_str(s).ok())
                .context("Failed to parse liquidation price")?;

            let quantity = item
                .get("sz")
                .and_then(|s| s.as_str())
                .and_then(|s| Decimal::from_str(s).ok())
                .context("Failed to parse liquidation size")?;

            let side_str = item
                .get("side")
                .and_then(|s| s.as_str())
                .context("Missing liquidation side")?;

            let side = match side_str {
                "buy" => Side::Buy,
                "sell" => Side::Sell,
                _ => continue,
            };

            let timestamp_str = item.get("ts").and_then(|t| t.as_str()).unwrap_or("0");

            let timestamp_ms: i64 = timestamp_str.parse().unwrap_or(0);
            let timestamp_micros = timestamp_ms * 1000;

            let event = LiquidationEvent {
                exchange: Exchange::Okx,
                symbol,
                timestamp: timestamp_micros,
                side,
                price,
                quantity,
                order_id: None,
            };

            events.push(MarketDataEvent::Liquidation(event));
        }

        Ok(events)
    }

    /// Parse candlestick data
    ///
    /// OKX candle message format (data array element):
    /// ```json
    /// {
    ///   "instId": "BTC-USDT",
    ///   "ts": "1597026383085",      // Unix timestamp in milliseconds
    ///   "o": "50000.0",              // Open price
    ///   "h": "51000.0",              // Highest price
    ///   "l": "49000.0",              // Lowest price
    ///   "c": "50500.0",              // Close price
    ///   "vol": "100.5",              // Volume in base currency
    ///   "volCcy": "5025000.0",       // Volume in quote currency
    ///   "volCcyQuote": "5025000.0",  // Volume in quote currency (alternative field)
    ///   "confirm": "0"               // "0" = incomplete, "1" = complete candle
    /// }
    /// ```
    fn parse_candles(&self, data: &[Value]) -> Result<Vec<MarketDataEvent>> {
        let mut events = Vec::new();

        for item in data {
            // Parse instrument ID
            let inst_id = item["instId"]
                .as_str()
                .context("Missing instId in candle data")?;
            let symbol = self.parse_inst_id(inst_id)?;

            // Parse timestamp (milliseconds)
            let ts_str = item["ts"].as_str().context("Missing ts in candle data")?;
            let ts_millis: i64 = ts_str.parse().context("Failed to parse candle timestamp")?;
            let open_time_micros = ts_millis * 1000; // Convert to microseconds

            // Parse OHLC prices
            let open = Decimal::from_str(item["o"].as_str().context("Missing open price")?)
                .context("Failed to parse open price")?;

            let high = Decimal::from_str(item["h"].as_str().context("Missing high price")?)
                .context("Failed to parse high price")?;

            let low = Decimal::from_str(item["l"].as_str().context("Missing low price")?)
                .context("Failed to parse low price")?;

            let close = Decimal::from_str(item["c"].as_str().context("Missing close price")?)
                .context("Failed to parse close price")?;

            // Parse volume
            let volume = Decimal::from_str(item["vol"].as_str().context("Missing volume")?)
                .context("Failed to parse volume")?;

            // Parse quote volume (try both field names)
            let quote_volume = item["volCcy"]
                .as_str()
                .or_else(|| item["volCcyQuote"].as_str())
                .and_then(|s| Decimal::from_str(s).ok());

            // Check if candle is closed/confirmed
            let is_closed = item["confirm"].as_str().map(|s| s == "1").unwrap_or(false);

            // Determine interval from subscription context
            // OKX uses channel names like "candle1m", "candle5m", "candle1H"
            // This would need to come from the subscription, default to "1m"
            let interval = "1m".to_string();

            // Calculate close time (for 1m candle, add 60 seconds)
            // In practice, this should be derived from the interval
            let close_time_micros = open_time_micros + 60_000_000; // 60 seconds in microseconds

            let kline_event = janus_core::KlineEvent {
                exchange: Exchange::Okx,
                symbol: symbol.clone(),
                interval,
                open_time: open_time_micros,
                close_time: close_time_micros,
                open,
                high,
                low,
                close,
                volume,
                quote_volume,
                trades: None, // OKX doesn't provide trade count in candle data
                is_closed,
            };

            trace!(
                "OKX Candle: {} O:{} H:{} L:{} C:{} V:{} closed:{}",
                inst_id, open, high, low, close, volume, is_closed
            );

            events.push(MarketDataEvent::Kline(kline_event));
        }

        Ok(events)
    }

    /// Format symbol to OKX instrument ID
    fn format_inst_id(&self, symbol: &Symbol) -> String {
        match self.inst_type {
            OkxInstType::Spot => {
                format!(
                    "{}-{}",
                    symbol.base.to_uppercase(),
                    symbol.quote.to_uppercase()
                )
            }
            OkxInstType::Swap => {
                format!(
                    "{}-{}-SWAP",
                    symbol.base.to_uppercase(),
                    symbol.quote.to_uppercase()
                )
            }
            OkxInstType::Futures => {
                // Futures require date suffix (e.g., BTC-USDT-230630)
                // This is a placeholder - real implementation would need expiry date
                format!(
                    "{}-{}-FUTURES",
                    symbol.base.to_uppercase(),
                    symbol.quote.to_uppercase()
                )
            }
        }
    }

    /// Parse OKX instrument ID to Symbol
    fn parse_inst_id(&self, inst_id: &str) -> Result<Symbol> {
        let parts: Vec<&str> = inst_id.split('-').collect();

        if parts.len() < 2 {
            return Err(anyhow::anyhow!("Invalid OKX instrument ID: {}", inst_id));
        }

        Ok(Symbol::new(parts[0], parts[1]))
    }
}

impl Default for OkxAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// OKX instrument type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OkxInstType {
    /// Spot market
    Spot,
    /// Perpetual swap
    Swap,
    /// Dated futures
    Futures,
}

/// OKX WebSocket channels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OkxChannel {
    Trades,
    Tickers,
    Books,
    Books5,
    BboTbt,
    Candle1m,
    Candle5m,
    Candle15m,
    Candle1h,
    FundingRate,
    LiquidationOrders,
}

impl std::fmt::Display for OkxChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OkxChannel::Trades => write!(f, "trades"),
            OkxChannel::Tickers => write!(f, "tickers"),
            OkxChannel::Books => write!(f, "books"),
            OkxChannel::Books5 => write!(f, "books5"),
            OkxChannel::BboTbt => write!(f, "bbo-tbt"),
            OkxChannel::Candle1m => write!(f, "candle1m"),
            OkxChannel::Candle5m => write!(f, "candle5m"),
            OkxChannel::Candle15m => write!(f, "candle15m"),
            OkxChannel::Candle1h => write!(f, "candle1H"),
            OkxChannel::FundingRate => write!(f, "funding-rate"),
            OkxChannel::LiquidationOrders => write!(f, "liquidation-orders"),
        }
    }
}

/// OKX subscription message
#[derive(Debug, Serialize, Deserialize)]
struct OkxSubscribeMessage {
    op: String,
    args: Vec<OkxSubscriptionArg>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OkxSubscriptionArg {
    channel: String,
    #[serde(rename = "instId")]
    inst_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_inst_id() {
        let adapter = OkxAdapter::new();
        let symbol = Symbol::new("BTC", "USDT");
        assert_eq!(adapter.format_inst_id(&symbol), "BTC-USDT");

        let adapter = OkxAdapter::with_inst_type(OkxInstType::Swap);
        assert_eq!(adapter.format_inst_id(&symbol), "BTC-USDT-SWAP");
    }

    #[test]
    fn test_parse_inst_id() {
        let adapter = OkxAdapter::new();

        let symbol = adapter.parse_inst_id("BTC-USDT").unwrap();
        assert_eq!(symbol.base, "BTC");
        assert_eq!(symbol.quote, "USDT");

        let symbol = adapter.parse_inst_id("BTC-USDT-SWAP").unwrap();
        assert_eq!(symbol.base, "BTC");
        assert_eq!(symbol.quote, "USDT");
    }

    #[test]
    fn test_subscribe_message() {
        let adapter = OkxAdapter::new();
        let symbol = Symbol::new("BTC", "USDT");
        let msg = adapter.default_subscribe(&symbol);

        assert!(msg.contains("subscribe"));
        assert!(msg.contains("BTC-USDT"));
        assert!(msg.contains("trades"));
        assert!(msg.contains("tickers"));
    }

    #[test]
    fn test_parse_trade() {
        let adapter = OkxAdapter::new();
        let raw = r#"{
            "arg": {
                "channel": "trades",
                "instId": "BTC-USDT"
            },
            "data": [{
                "instId": "BTC-USDT",
                "tradeId": "123456789",
                "px": "50000.0",
                "sz": "0.001",
                "side": "buy",
                "ts": "1609459200123"
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            MarketDataEvent::Trade(trade) => {
                assert_eq!(trade.exchange, Exchange::Okx);
                assert_eq!(trade.symbol.base, "BTC");
                assert_eq!(trade.symbol.quote, "USDT");
                assert_eq!(trade.price, Decimal::from_str("50000.0").unwrap());
                assert_eq!(trade.quantity, Decimal::from_str("0.001").unwrap());
                assert_eq!(trade.side, Side::Buy);
                assert_eq!(trade.trade_id, "123456789");
            }
            _ => panic!("Expected TradeEvent"),
        }
    }

    #[test]
    fn test_parse_book_snapshot() {
        let adapter = OkxAdapter::new();
        let raw = r#"{
            "arg": {
                "channel": "books",
                "instId": "BTC-USDT"
            },
            "data": [{
                "asks": [
                    ["50001.0", "1.5", "0", "2"],
                    ["50002.0", "2.0", "0", "1"],
                    ["50003.0", "0.5", "0", "1"]
                ],
                "bids": [
                    ["50000.0", "2.5", "0", "3"],
                    ["49999.0", "1.0", "0", "2"],
                    ["49998.0", "3.0", "0", "1"]
                ],
                "ts": "1609459200123",
                "checksum": 123456789,
                "seqId": 1000,
                "action": "snapshot",
                "instId": "BTC-USDT"
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        if let MarketDataEvent::OrderBook(book) = &events[0] {
            assert_eq!(book.exchange, Exchange::Okx);
            assert_eq!(book.symbol.base, "BTC");
            assert_eq!(book.symbol.quote, "USDT");
            assert!(book.is_snapshot);
            assert_eq!(book.sequence, 1000);
            assert_eq!(book.bids.len(), 3);
            assert_eq!(book.asks.len(), 3);

            // Verify bids are sorted descending
            assert!(book.bids[0].price > book.bids[1].price);
            assert_eq!(book.bids[0].price, Decimal::from_str("50000.0").unwrap());
            assert_eq!(book.bids[0].quantity, Decimal::from_str("2.5").unwrap());

            // Verify asks are sorted ascending
            assert!(book.asks[0].price < book.asks[1].price);
            assert_eq!(book.asks[0].price, Decimal::from_str("50001.0").unwrap());
            assert_eq!(book.asks[0].quantity, Decimal::from_str("1.5").unwrap());

            // Test spread calculation
            let spread = book.spread().unwrap();
            assert_eq!(spread, Decimal::from_str("1.0").unwrap());
        } else {
            panic!("Expected OrderBook event");
        }
    }

    #[test]
    fn test_parse_book_update() {
        let adapter = OkxAdapter::new();
        let raw = r#"{
            "arg": {
                "channel": "books",
                "instId": "ETH-USDT"
            },
            "data": [{
                "asks": [
                    ["3005.0", "5.5", "0", "1"]
                ],
                "bids": [
                    ["2995.0", "2.2", "0", "1"],
                    ["2994.0", "0.0", "0", "0"]
                ],
                "ts": "1609459201123",
                "checksum": 987654321,
                "seqId": 1001,
                "action": "update",
                "instId": "ETH-USDT"
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        if let MarketDataEvent::OrderBook(book) = &events[0] {
            assert_eq!(book.symbol.base, "ETH");
            assert_eq!(book.symbol.quote, "USDT");
            assert!(!book.is_snapshot);
            assert_eq!(book.sequence, 1001);
            // Zero quantity updates should be filtered out
            assert_eq!(book.bids.len(), 1);
            assert_eq!(book.asks.len(), 1);
            assert_eq!(book.bids[0].price, Decimal::from_str("2995.0").unwrap());
            assert_eq!(book.bids[0].quantity, Decimal::from_str("2.2").unwrap());
        } else {
            panic!("Expected OrderBook event");
        }
    }

    #[test]
    fn test_parse_bbo() {
        let adapter = OkxAdapter::new();
        let raw = r#"{
            "arg": {
                "channel": "bbo-tbt",
                "instId": "BTC-USDT"
            },
            "data": [{
                "asks": [
                    ["50001.0", "1.5", "0", "1"]
                ],
                "bids": [
                    ["50000.0", "2.5", "0", "1"]
                ],
                "ts": "1609459200123",
                "seqId": 500,
                "instId": "BTC-USDT"
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        if let MarketDataEvent::OrderBook(book) = &events[0] {
            assert_eq!(book.exchange, Exchange::Okx);
            assert_eq!(book.symbol.base, "BTC");
            assert!(!book.is_snapshot);
            assert_eq!(book.sequence, 500);
            // BBO should have only 1 bid and 1 ask
            assert_eq!(book.bids.len(), 1);
            assert_eq!(book.asks.len(), 1);
            assert_eq!(book.bids[0].price, Decimal::from_str("50000.0").unwrap());
            assert_eq!(book.asks[0].price, Decimal::from_str("50001.0").unwrap());

            // Test mid price
            let mid = book.mid_price().unwrap();
            assert_eq!(mid, Decimal::from_str("50000.5").unwrap());
        } else {
            panic!("Expected OrderBook event");
        }
    }

    #[test]
    fn test_parse_ticker() {
        let adapter = OkxAdapter::new();
        let raw = r#"{
            "arg": {
                "channel": "tickers",
                "instId": "BTC-USDT"
            },
            "data": [{
                "instId": "BTC-USDT",
                "last": "50000.0",
                "lastSz": "0.001",
                "askPx": "50010.0",
                "askSz": "1.5",
                "bidPx": "49990.0",
                "bidSz": "2.0",
                "open24h": "49000.0",
                "high24h": "51000.0",
                "low24h": "48000.0",
                "volCcy24h": "1000000.0",
                "vol24h": "20.5",
                "ts": "1609459200123"
            }]
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            MarketDataEvent::Ticker(ticker) => {
                assert_eq!(ticker.exchange, Exchange::Okx);
                assert_eq!(ticker.symbol.base, "BTC");
                assert_eq!(ticker.last_price, Decimal::from_str("50000.0").unwrap());
                assert_eq!(ticker.volume_24h, Decimal::from_str("20.5").unwrap());
            }
            _ => panic!("Expected TickerEvent"),
        }
    }

    #[test]
    fn test_parse_subscription_confirmation() {
        let adapter = OkxAdapter::new();
        let raw = r#"{
            "event": "subscribe",
            "arg": {
                "channel": "trades",
                "instId": "BTC-USDT"
            }
        }"#;

        let events = adapter.parse_message(raw).unwrap();
        assert_eq!(events.len(), 0);
    }
}
