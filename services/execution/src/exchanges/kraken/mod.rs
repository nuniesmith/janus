//! Kraken Exchange Integration
//!
//! This module provides WebSocket and REST API clients for Kraken exchange.
//! Kraken offers FREE market data via WebSocket with no API key required.
//!
//! # WebSocket URLs
//!
//! - **Public (Market Data)**: `wss://ws.kraken.com/v2` (FREE - No API Key)
//! - **Private (Trading)**: `wss://ws-auth.kraken.com/v2` (Requires API Key)
//!
//! # Supported Channels
//!
//! | Channel | Description | Auth Required |
//! |---------|-------------|---------------|
//! | ticker  | Real-time price updates | No |
//! | trade   | Real-time trades | No |
//! | ohlc    | Candlestick data | No |
//! | book    | Order book | No |
//! | executions | Order fills | Yes |
//!
//! # Symbol Format
//!
//! Kraken uses `BASE/QUOTE` format (e.g., `BTC/USD`, `ETH/USD`).
//! Note: Kraken primarily uses USD, not USDT for major pairs.
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::exchanges::kraken::{KrakenWebSocket, KrakenWsConfig};
//!
//! let config = KrakenWsConfig::default();
//! let ws = KrakenWebSocket::new(config);
//!
//! // Subscribe to market data (FREE!)
//! ws.connect().await?;
//! ws.subscribe_ticker(&["BTC/USD", "ETH/USD"]).await?;
//!
//! // Receive events
//! let mut rx = ws.subscribe_events();
//! while let Ok(event) = rx.recv().await {
//!     println!("Received: {:?}", event);
//! }
//! ```

pub mod fill_tracker;
pub mod private_websocket;
pub mod provider;
pub mod rest;
pub mod websocket;

// Re-exports
pub use fill_tracker::{FillTrackerConfig, KrakenFillTracker, ReconciliationResult};
pub use private_websocket::{
    BalanceUpdateEvent, ExecutionEvent, ExecutionType, KrakenPrivateWebSocket,
    KrakenPrivateWsConfig, OrderStatusEvent, PrivateWsEvent,
};
pub use provider::KrakenProvider;
pub use rest::{KrakenBalance, KrakenOrderResult, KrakenRestClient, KrakenRestConfig};
pub use websocket::{
    KrakenCandle, KrakenEvent, KrakenTicker, KrakenTrade, KrakenWebSocket, KrakenWsConfig,
};

use crate::exchanges::{ExchangeId, TradingPair};

/// Convert a normalized symbol to Kraken format
///
/// Kraken uses `BASE/QUOTE` format with USD instead of USDT
pub fn to_kraken_symbol(normalized: &str) -> String {
    TradingPair::from_normalized(normalized)
        .map(|pair| pair.to_kraken())
        .unwrap_or_else(|| normalized.to_string())
}

/// Convert a Kraken symbol to normalized format
pub fn from_kraken_symbol(kraken_symbol: &str) -> String {
    // Kraken already uses BASE/QUOTE format
    // But we need to normalize USD -> USDT for consistency
    if kraken_symbol.ends_with("/USD") {
        kraken_symbol.replace("/USD", "/USDT")
    } else {
        kraken_symbol.to_string()
    }
}

/// Kraken exchange identifier
pub const EXCHANGE_ID: ExchangeId = ExchangeId::Kraken;

/// Kraken public WebSocket URL (v2)
pub const WS_PUBLIC_URL: &str = "wss://ws.kraken.com/v2";

/// Kraken private WebSocket URL (v2)
pub const WS_PRIVATE_URL: &str = "wss://ws-auth.kraken.com/v2";

/// Kraken REST API URL
pub const REST_API_URL: &str = "https://api.kraken.com";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_kraken_symbol() {
        assert_eq!(to_kraken_symbol("BTC/USDT"), "BTC/USD");
        assert_eq!(to_kraken_symbol("ETH/USDT"), "ETH/USD");
        assert_eq!(to_kraken_symbol("SOL/USDT"), "SOL/USD");
    }

    #[test]
    fn test_from_kraken_symbol() {
        assert_eq!(from_kraken_symbol("BTC/USD"), "BTC/USDT");
        assert_eq!(from_kraken_symbol("ETH/USD"), "ETH/USDT");
        assert_eq!(from_kraken_symbol("XRP/EUR"), "XRP/EUR");
    }

    #[test]
    fn test_constants() {
        assert_eq!(WS_PUBLIC_URL, "wss://ws.kraken.com/v2");
        assert_eq!(REST_API_URL, "https://api.kraken.com");
    }
}
