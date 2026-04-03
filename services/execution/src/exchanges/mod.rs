//! Exchange abstraction layer for the FKS Execution Service
//!
//! This module provides a unified interface for interacting with different
//! cryptocurrency exchanges. Each exchange implements the `Exchange` trait
//! for trading operations and `MarketDataProvider` for streaming market data.
//!
//! # Exchange Priority
//!
//! | Priority | Exchange | Role                                | Market Data | Trading |
//! |----------|----------|-------------------------------------|-------------|---------|
//! | 1st      | **Kraken** | **Primary** — default data & REST | ✅ WS + REST | ✅ Full |
//! | 2nd      | Bybit    | Backup / alternate                  | ✅ WS + REST | ✅ Full |
//! | 3rd      | Binance  | Tertiary / additional liquidity     | ✅ WS + REST | ✅ Full |
//!
//! Kraken is the default exchange for data websockets and REST APIs.
//! Bybit serves as the backup/alternate. Binance provides additional
//! liquidity and cross-exchange price comparison.
//!
//! Use [`market_data::ExchangeId::primary()`] and
//! [`market_data::ExchangeId::all_by_priority()`] to obtain the canonical
//! ordering when initialising providers.
//!
//! # Free Market Data (No API Key Required)
//!
//! All three exchanges offer free WebSocket market data:
//! - **Kraken** (primary): `wss://ws.kraken.com/v2`
//! - **Bybit** (backup): `wss://stream.bybit.com/v5/public/spot`
//! - **Binance** (tertiary): `wss://stream.binance.com:9443/ws`
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    MarketDataAggregator                      │
//! │  (Aggregates data from all exchanges, finds best prices)    │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!        ┌─────────────────────┼─────────────────────┐
//!        ▼                     ▼                     ▼
//! ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
//! │  ⭐ Kraken  │       │   Bybit     │       │  Binance    │
//! │  (Primary)  │       │  (Backup)   │       │ (Tertiary)  │
//! └─────────────┘       └─────────────┘       └─────────────┘
//!        │                     │                     │
//!        ▼                     ▼                     ▼
//! ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
//! │  WebSocket  │       │  WebSocket  │       │  WebSocket  │
//! │   Client    │       │   Client    │       │   Client    │
//! └─────────────┘       └─────────────┘       └─────────────┘
//! ```
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use janus_execution::exchanges::{
//!     MarketDataAggregator, MarketDataProvider, ExchangeId,
//!     BybitProvider, KrakenProvider, BinanceProvider,
//! };
//!
//! // Create providers (no API keys needed for market data!)
//! // Add in priority order: Kraken (primary) → Bybit (backup) → Binance
//! let kraken = Arc::new(KrakenProvider::new());
//! let bybit = Arc::new(BybitProvider::new());
//! let binance = Arc::new(BinanceProvider::new());
//!
//! // Create aggregator — add providers in priority order
//! let mut aggregator = MarketDataAggregator::new();
//! aggregator.add_provider(kraken);   // primary
//! aggregator.add_provider(bybit);    // backup
//! aggregator.add_provider(binance);  // tertiary
//!
//! // Connect all
//! aggregator.connect_all().await?;
//!
//! // Subscribe to tickers
//! aggregator.subscribe_ticker_all(&["BTC/USDT", "ETH/USDT"]).await?;
//!
//! // Get best price across all exchanges
//! if let Some(best) = aggregator.get_best_price("BTC/USDT") {
//!     println!("Best bid: {} on {}", best.best_bid, best.best_bid_exchange);
//!     println!("Best ask: {} on {}", best.best_ask, best.best_ask_exchange);
//! }
//! ```

use crate::error::Result;
use crate::types::{Fill, Order, OrderStatusEnum, Position};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// ============================================================================
// Submodules
// ============================================================================

pub mod binance;
pub mod bybit;
pub mod kraken;
pub mod market_data;
pub mod metrics;
pub mod provider;
pub mod rate_limit;
pub mod router;

// ============================================================================
// Re-exports
// ============================================================================

// Market Data Types
pub use market_data::{
    BestPrice, Candle, ExchangeId, MarketDataEvent, OrderBook, OrderBookLevel, Subscription,
    SubscriptionChannel, SymbolNormalizer, Ticker, Trade, TradeSide, TradingPair,
};

// Provider Trait & Types
pub use provider::{
    ConnectionHealth, MarketDataAggregator, MarketDataProvider, ProviderConfig, ReconnectConfig,
    SubscriptionConfig,
};

// Exchange-specific re-exports
pub use binance::{
    BinanceCandle, BinanceEvent, BinanceProvider, BinanceTicker, BinanceTrade, BinanceWebSocket,
    BinanceWsConfig,
};
pub use bybit::{BybitProvider, BybitProviderConfig};
pub use kraken::{
    KrakenCandle, KrakenEvent, KrakenProvider, KrakenTicker, KrakenTrade, KrakenWebSocket,
    KrakenWsConfig,
};

// Metrics
pub use metrics::{
    ExchangeHealthStatus, ExchangeMetrics, ExchangeMetricsRegistry, HealthSummary, global_registry,
    prometheus_metrics, register_exchange,
};

/// Exchange trait that all exchange adapters must implement
#[async_trait]
pub trait Exchange: Send + Sync {
    /// Get the exchange name
    fn name(&self) -> &str;

    /// Check if the exchange is in testnet mode
    fn is_testnet(&self) -> bool;

    /// Place a new order on the exchange
    ///
    /// # Arguments
    /// * `order` - The order to place
    ///
    /// # Returns
    /// * `Ok(String)` - The exchange's order ID on success
    /// * `Err(ExecutionError)` - On failure
    async fn place_order(&self, order: &Order) -> Result<String>;

    /// Cancel an existing order
    ///
    /// # Arguments
    /// * `order_id` - The exchange order ID to cancel
    ///
    /// # Returns
    /// * `Ok(())` - On successful cancellation
    /// * `Err(ExecutionError)` - On failure
    async fn cancel_order(&self, order_id: &str) -> Result<()>;

    /// Cancel all orders for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Optional symbol filter. If None, cancels all orders
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` - List of cancelled order IDs
    /// * `Err(ExecutionError)` - On failure
    async fn cancel_all_orders(&self, symbol: Option<&str>) -> Result<Vec<String>>;

    /// Get the status of an order
    ///
    /// # Arguments
    /// * `order_id` - The exchange order ID
    ///
    /// # Returns
    /// * `Ok(OrderStatusResponse)` - Order status details
    /// * `Err(ExecutionError)` - On failure
    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatusResponse>;

    /// Get all active orders
    ///
    /// # Arguments
    /// * `symbol` - Optional symbol filter
    ///
    /// # Returns
    /// * `Ok(Vec<OrderStatusResponse>)` - List of active orders
    /// * `Err(ExecutionError)` - On failure
    async fn get_active_orders(&self, symbol: Option<&str>) -> Result<Vec<OrderStatusResponse>>;

    /// Get account balance information
    ///
    /// # Returns
    /// * `Ok(Balance)` - Account balance details
    /// * `Err(ExecutionError)` - On failure
    async fn get_balance(&self) -> Result<Balance>;

    /// Get current positions
    ///
    /// # Arguments
    /// * `symbol` - Optional symbol filter
    ///
    /// # Returns
    /// * `Ok(Vec<Position>)` - List of positions
    /// * `Err(ExecutionError)` - On failure
    async fn get_positions(&self, symbol: Option<&str>) -> Result<Vec<Position>>;

    /// Subscribe to order updates via WebSocket
    ///
    /// # Returns
    /// * `Ok(OrderUpdateReceiver)` - Receiver for order updates
    /// * `Err(ExecutionError)` - On failure
    async fn subscribe_order_updates(&self) -> Result<OrderUpdateReceiver>;

    /// Subscribe to position updates via WebSocket
    ///
    /// # Returns
    /// * `Ok(PositionUpdateReceiver)` - Receiver for position updates
    /// * `Err(ExecutionError)` - On failure
    async fn subscribe_position_updates(&self) -> Result<PositionUpdateReceiver>;

    /// Health check for the exchange connection
    ///
    /// # Returns
    /// * `Ok(())` - If exchange is healthy
    /// * `Err(ExecutionError)` - If exchange is unhealthy
    async fn health_check(&self) -> Result<()>;
}

/// Order status response from exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderStatusResponse {
    /// Exchange order ID
    pub order_id: String,

    /// Client order ID (if provided)
    pub client_order_id: Option<String>,

    /// Trading symbol
    pub symbol: String,

    /// Order status
    pub status: OrderStatusEnum,

    /// Order quantity
    pub quantity: Decimal,

    /// Filled quantity
    pub filled_quantity: Decimal,

    /// Remaining quantity
    pub remaining_quantity: Decimal,

    /// Order price (for limit orders)
    pub price: Option<Decimal>,

    /// Average fill price
    pub average_fill_price: Option<Decimal>,

    /// Order creation time
    pub created_at: DateTime<Utc>,

    /// Last update time
    pub updated_at: DateTime<Utc>,

    /// List of fills
    pub fills: Vec<Fill>,
}

/// Account balance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    /// Total balance
    pub total: Decimal,

    /// Available balance for trading
    pub available: Decimal,

    /// Balance in use (margin, orders)
    pub used: Decimal,

    /// Asset/currency (e.g., "USDT", "BTC")
    pub currency: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Order update from WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderUpdate {
    /// Exchange order ID
    pub order_id: String,

    /// Order status
    pub status: OrderStatusEnum,

    /// Symbol
    pub symbol: String,

    /// Filled quantity
    pub filled_quantity: Decimal,

    /// Average fill price
    pub average_fill_price: Option<Decimal>,

    /// Update timestamp
    pub timestamp: DateTime<Utc>,

    /// Fill details (if this is a fill update)
    pub fill: Option<Fill>,
}

/// Position update from WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    /// Position details
    pub position: Position,

    /// Update timestamp
    pub timestamp: DateTime<Utc>,
}

/// Receiver type for order updates
pub type OrderUpdateReceiver = tokio::sync::mpsc::UnboundedReceiver<OrderUpdate>;

/// Receiver type for position updates
pub type PositionUpdateReceiver = tokio::sync::mpsc::UnboundedReceiver<PositionUpdate>;

/// Exchange capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExchangeCapabilities {
    /// Supports WebSocket streaming
    pub websocket_support: bool,

    /// Supports limit orders
    pub limit_orders: bool,

    /// Supports market orders
    pub market_orders: bool,

    /// Supports stop orders
    pub stop_orders: bool,

    /// Supports margin trading
    pub margin_trading: bool,

    /// Supports futures trading
    pub futures_trading: bool,
}

impl Default for ExchangeCapabilities {
    fn default() -> Self {
        Self {
            websocket_support: true,
            limit_orders: true,
            market_orders: true,
            stop_orders: true,
            margin_trading: false,
            futures_trading: false,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities_default() {
        let caps = ExchangeCapabilities::default();
        assert!(caps.websocket_support);
        assert!(caps.limit_orders);
        assert!(caps.market_orders);
    }

    #[test]
    fn test_balance_serialization() {
        let balance = Balance {
            total: Decimal::from(10000),
            available: Decimal::from(5000),
            used: Decimal::from(5000),
            currency: "USDT".to_string(),
            timestamp: Utc::now(),
        };

        let json = serde_json::to_string(&balance).unwrap();
        let deserialized: Balance = serde_json::from_str(&json).unwrap();

        assert_eq!(balance.total, deserialized.total);
        assert_eq!(balance.currency, deserialized.currency);
    }

    #[test]
    fn test_exchange_id_urls() {
        assert_eq!(
            ExchangeId::Bybit.market_data_ws_url(),
            "wss://stream.bybit.com/v5/public/spot"
        );
        assert_eq!(
            ExchangeId::Kraken.market_data_ws_url(),
            "wss://ws.kraken.com/v2"
        );
        assert_eq!(
            ExchangeId::Binance.market_data_ws_url(),
            "wss://stream.binance.com:9443/ws"
        );
    }

    #[test]
    fn test_trading_pair_conversion() {
        let pair = TradingPair::new("BTC", "USDT");
        assert_eq!(pair.to_bybit(), "BTCUSDT");
        assert_eq!(pair.to_kraken(), "BTC/USD");
        assert_eq!(pair.to_binance(), "btcusdt");
    }
}
