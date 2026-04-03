//! # JANUS Exchanges
//!
//! Unified exchange adapters for cryptocurrency market data ingestion.
//!
//! This crate provides a consistent interface for connecting to multiple cryptocurrency
//! exchanges via WebSocket, normalizing their disparate message formats into JANUS's
//! unified `MarketDataEvent` types.
//!
//! ## Features
//!
//! - **Unified API**: All exchanges expose the same interface
//! - **Type Safety**: Exchange-specific messages converted to type-safe Rust structs
//! - **Extensible**: Easy to add new exchanges
//! - **Feature Flags**: Enable only the exchanges you need
//!
//! ## Supported Exchanges
//!
//! | Exchange | Feature Flag | Markets | Notes |
//! |----------|--------------|---------|-------|
//! | Binance  | `binance`    | Spot, Futures | High volume, multi-timeframe klines |
//! | Bybit    | `bybit`      | Spot, Derivatives | V5 unified API |
//! | Coinbase | `coinbase`   | Spot | Advanced Trade API |
//! | Kraken   | `kraken`     | Spot, Futures | Professional trading |
//! | OKX      | `okx`        | Spot, Swap, Futures | Global derivatives |
//! | Kucoin   | `kucoin`     | Spot | Token-based auth |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use janus_exchanges::adapters::coinbase::CoinbaseAdapter;
//! use janus_core::{Symbol, MarketDataEvent};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create adapter
//! let adapter = CoinbaseAdapter::new();
//!
//! // Build subscription
//! let symbol = Symbol::new("BTC", "USD");
//! let subscribe_msg = adapter.default_subscribe(&symbol);
//!
//! // Connect to WebSocket (using tokio-tungstenite)
//! // let (ws_stream, _) = tokio_tungstenite::connect_async(adapter.ws_url()).await?;
//!
//! // Parse incoming messages
//! let raw_message = r#"{"channel":"matches","events":[...]}"#;
//! let events = adapter.parse_message(raw_message)?;
//!
//! for event in events {
//!     match event {
//!         MarketDataEvent::Trade(trade) => {
//!             println!("Trade: {} @ {}", trade.quantity, trade.price);
//!         }
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   Exchange Adapters                      │
//! ├──────────┬──────────┬──────────┬──────────┬──────────────┤
//! │ Binance  │  Bybit   │ Coinbase │  Kraken  │  OKX  Kucoin │
//! └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬────┬───┘
//!      │          │          │          │          │    │
//!      └──────────┴──────────┴──────────┴──────────┴────┘
//!                           │
//!                    Unified Events
//!                           │
//!      ┌────────────────────┴────────────────────┐
//!      │         MarketDataEvent                 │
//!      ├─────────────────────────────────────────┤
//!      │ • Trade                                 │
//!      │ • OrderBook                             │
//!      │ • Ticker                                │
//!      │ • Liquidation                           │
//!      │ • FundingRate                           │
//!      │ • Kline                                 │
//!      └─────────────────────────────────────────┘
//! ```
//!
//! ## Adding a New Exchange
//!
//! To add support for a new exchange:
//!
//! 1. Create `src/adapters/your_exchange.rs`
//! 2. Implement the adapter struct with:
//!    - `new()` - Constructor
//!    - `ws_url()` - WebSocket endpoint
//!    - `subscribe_message()` - Build subscription
//!    - `parse_message()` - Parse to `MarketDataEvent`
//! 3. Add feature flag in `Cargo.toml`
//! 4. Export from `src/adapters/mod.rs`
//! 5. Add tests
//!
//! ## Error Handling
//!
//! All parsing methods return `Result<Vec<MarketDataEvent>, anyhow::Error>`.
//! Empty vectors are returned for:
//! - Control messages (heartbeats, subscription confirmations)
//! - Unknown/unsupported message types
//! - Malformed data that can be safely ignored
//!
//! Errors are returned only for critical parsing failures that indicate
//! a broken connection or API change.

pub mod adapters;
pub mod cns;
pub mod health;
pub mod normalizer;
pub mod types;

// Re-export commonly used types
pub use adapters::*;
pub use cns::CNSReporter;
pub use health::{ExchangeHealth, HealthChecker};
pub use normalizer::PriceNormalizer;
pub use types::{ConnectionConfig, SubscriptionRequest};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
