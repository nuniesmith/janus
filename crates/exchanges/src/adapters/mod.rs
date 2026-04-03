//! Exchange Adapters Module
//!
//! This module provides unified adapters for all supported cryptocurrency exchanges.
//! Each adapter implements exchange-specific WebSocket protocols and converts
//! exchange-native messages into JANUS's unified `MarketDataEvent` format.
//!
//! ## Supported Exchanges:
//! - **Coinbase**: Coinbase Advanced Trade API
//! - **Kraken**: Professional trading platform
//! - **OKX**: Global exchange with derivatives support
//!
//! ## Planned Exchanges:
//! - **Binance**: High-volume spot and futures markets (to be migrated)
//! - **Bybit**: V5 unified API for spot and derivatives (to be migrated)
//! - **Kucoin**: Token-based WebSocket authentication (to be migrated)
//!
//! ## Design Pattern:
//! Each adapter provides:
//! - `new()` - Create adapter with default settings
//! - `ws_url()` - Get WebSocket endpoint URL
//! - `subscribe_message()` - Build subscription message for symbol/channel
//! - `parse_message()` - Parse raw WebSocket message into `MarketDataEvent`s
//!
//! ## Example:
//! ```rust,ignore
//! use janus_exchanges::adapters::coinbase::CoinbaseAdapter;
//! use janus_core::Symbol;
//!
//! let adapter = CoinbaseAdapter::new();
//! let symbol = Symbol::new("BTC", "USD");
//! let subscribe_msg = adapter.default_subscribe(&symbol);
//!
//! // Later, when receiving WebSocket messages:
//! let events = adapter.parse_message(&raw_message)?;
//! for event in events {
//!     // Process unified MarketDataEvent
//! }
//! ```

#[cfg(feature = "coinbase")]
pub mod coinbase;

#[cfg(feature = "kraken")]
pub mod kraken;

#[cfg(feature = "okx")]
pub mod okx;

// Re-exports for convenience
#[cfg(feature = "coinbase")]
pub use coinbase::{CoinbaseAdapter, CoinbaseChannel};

#[cfg(feature = "kraken")]
pub use kraken::{KrakenAdapter, KrakenChannel};

#[cfg(feature = "okx")]
pub use okx::{OkxAdapter, OkxChannel, OkxInstType};
