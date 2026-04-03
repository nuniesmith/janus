//! Binance Exchange Integration
//!
//! This module provides integration with Binance exchange including:
//! - REST API for order execution and queries
//! - WebSocket API for real-time market data updates
//! - Authentication and request signing
//!
//! # WebSocket URLs
//!
//! - **Spot (Public)**: `wss://stream.binance.com:9443/ws` (FREE - No API Key)
//! - **Spot (Combined)**: `wss://stream.binance.com:9443/stream` (FREE - Multiple streams)
//! - **Testnet**: `wss://testnet.binance.vision/ws` (For testing)
//!
//! # REST API URLs
//!
//! - **Mainnet**: `https://api.binance.com`
//! - **Testnet**: `https://testnet.binance.vision`
//!
//! # Supported Streams
//!
//! | Stream | Description | Auth Required |
//! |--------|-------------|---------------|
//! | `<symbol>@trade` | Real-time trades | No |
//! | `<symbol>@ticker` | 24hr ticker | No |
//! | `<symbol>@kline_<interval>` | Candlestick data | No |
//! | `<symbol>@depth<levels>` | Order book | No |
//! | `<symbol>@aggTrade` | Aggregate trades | No |
//!
//! # Symbol Format
//!
//! Binance uses lowercase concatenated format (e.g., `btcusdt`, `ethusdt`).
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::exchanges::binance::{BinanceWebSocket, BinanceWsConfig, BinanceExchange};
//!
//! // Create REST client for trading (requires API keys)
//! let exchange = BinanceExchange::new(api_key, api_secret, false);
//! let order_id = exchange.place_order(&order).await?;
//!
//! // Create WebSocket client for market data (FREE!)
//! let config = BinanceWsConfig::default();
//! let ws = BinanceWebSocket::new(config);
//!
//! // Subscribe to market data
//! ws.connect().await?;
//! ws.subscribe_ticker(&["BTC/USDT", "ETH/USDT"]).await?;
//!
//! // Receive events
//! let mut rx = ws.subscribe_events();
//! while let Ok(event) = rx.recv().await {
//!     println!("Received: {:?}", event);
//! }
//! ```

pub mod private_websocket;
pub mod provider;
pub mod rest;
pub mod websocket;

// REST API exports
pub use rest::{
    BinanceAccountInfo, BinanceBalance, BinanceExchange, BinanceFill, BinanceOrderResponse,
    BinanceOrderStatus,
};

// Private WebSocket exports (User Data Stream)
pub use private_websocket::{
    AccountBalanceUpdate, BinancePrivateWebSocket, BinancePrivateWsConfig, OrderUpdateEvent,
    PrivateWsEvent,
};

// Public WebSocket exports (Market Data)
pub use websocket::{
    BinanceCandle, BinanceEvent, BinanceTicker, BinanceTrade, BinanceWebSocket, BinanceWsConfig,
};

// Provider export
pub use provider::BinanceProvider;

use crate::exchanges::{ExchangeId, TradingPair};

/// Convert a normalized symbol to Binance format
///
/// Binance uses lowercase concatenated format (e.g., `btcusdt`)
pub fn to_binance_symbol(normalized: &str) -> String {
    TradingPair::from_normalized(normalized)
        .map(|pair| pair.to_binance())
        .unwrap_or_else(|| normalized.to_lowercase().replace("/", ""))
}

/// Convert a Binance symbol to normalized format
pub fn from_binance_symbol(binance_symbol: &str) -> String {
    // Binance symbols are like BTCUSD - need to insert the /
    let upper = binance_symbol.to_uppercase();

    // Common quote currencies to check
    let quotes = ["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"];

    for quote in quotes {
        if upper.ends_with(quote) {
            let base = &upper[..upper.len() - quote.len()];
            if !base.is_empty() {
                return format!("{}/{}", base, quote);
            }
        }
    }

    // Fallback - return as is
    upper
}

/// Create a trade stream name
pub fn trade_stream(symbol: &str) -> String {
    format!("{}@trade", to_binance_symbol(symbol))
}

/// Create a ticker stream name
pub fn ticker_stream(symbol: &str) -> String {
    format!("{}@ticker", to_binance_symbol(symbol))
}

/// Create a kline/candlestick stream name
pub fn kline_stream(symbol: &str, interval: &str) -> String {
    format!("{}@kline_{}", to_binance_symbol(symbol), interval)
}

/// Create a depth (order book) stream name
pub fn depth_stream(symbol: &str, levels: u32) -> String {
    format!("{}@depth{}", to_binance_symbol(symbol), levels)
}

/// Create an aggregate trade stream name
pub fn agg_trade_stream(symbol: &str) -> String {
    format!("{}@aggTrade", to_binance_symbol(symbol))
}

/// Binance exchange identifier
pub const EXCHANGE_ID: ExchangeId = ExchangeId::Binance;

/// Binance spot WebSocket URL
pub const WS_SPOT_URL: &str = "wss://stream.binance.com:9443/ws";

/// Binance combined stream URL
pub const WS_COMBINED_URL: &str = "wss://stream.binance.com:9443/stream";

/// Binance testnet WebSocket URL
pub const WS_TESTNET_URL: &str = "wss://testnet.binance.vision/ws";

/// Binance REST API URL
pub const REST_API_URL: &str = "https://api.binance.com";

/// Binance testnet REST API URL
pub const REST_TESTNET_URL: &str = "https://testnet.binance.vision";

/// Convert interval minutes to Binance interval string
pub fn interval_to_binance(minutes: u32) -> &'static str {
    match minutes {
        1 => "1m",
        3 => "3m",
        5 => "5m",
        15 => "15m",
        30 => "30m",
        60 => "1h",
        120 => "2h",
        240 => "4h",
        360 => "6h",
        480 => "8h",
        720 => "12h",
        1440 => "1d",
        4320 => "3d",
        10080 => "1w",
        43200 => "1M",
        _ => "1m",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_binance_symbol() {
        assert_eq!(to_binance_symbol("BTC/USDT"), "btcusdt");
        assert_eq!(to_binance_symbol("ETH/USDT"), "ethusdt");
        assert_eq!(to_binance_symbol("SOL/USDT"), "solusdt");
    }

    #[test]
    fn test_from_binance_symbol() {
        assert_eq!(from_binance_symbol("BTCUSDT"), "BTC/USDT");
        assert_eq!(from_binance_symbol("ETHUSDT"), "ETH/USDT");
        assert_eq!(from_binance_symbol("btcusdt"), "BTC/USDT");
    }

    #[test]
    fn test_stream_names() {
        assert_eq!(trade_stream("BTC/USDT"), "btcusdt@trade");
        assert_eq!(ticker_stream("ETH/USDT"), "ethusdt@ticker");
        assert_eq!(kline_stream("BTC/USDT", "5m"), "btcusdt@kline_5m");
        assert_eq!(depth_stream("BTC/USDT", 20), "btcusdt@depth20");
        assert_eq!(agg_trade_stream("BTC/USDT"), "btcusdt@aggTrade");
    }

    #[test]
    fn test_interval_to_binance() {
        assert_eq!(interval_to_binance(1), "1m");
        assert_eq!(interval_to_binance(5), "5m");
        assert_eq!(interval_to_binance(15), "15m");
        assert_eq!(interval_to_binance(60), "1h");
        assert_eq!(interval_to_binance(240), "4h");
        assert_eq!(interval_to_binance(1440), "1d");
    }

    #[test]
    fn test_constants() {
        assert_eq!(WS_SPOT_URL, "wss://stream.binance.com:9443/ws");
        assert_eq!(WS_TESTNET_URL, "wss://testnet.binance.vision/ws");
        assert_eq!(REST_API_URL, "https://api.binance.com");
        assert_eq!(REST_TESTNET_URL, "https://testnet.binance.vision");
    }
}
