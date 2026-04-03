//! Bybit Exchange Integration
//!
//! This module provides integration with Bybit exchange including:
//! - REST API for order execution and queries
//! - WebSocket API for real-time updates (private channels)
//! - Public WebSocket API for market data streaming
//! - Fill Tracker for real-time execution monitoring
//! - Market Data Provider for unified data access
//! - Authentication and request signing

pub mod fill_tracker;
pub mod provider;
pub mod public_websocket;
pub mod rest;
pub mod websocket;

pub use fill_tracker::{
    BybitFillTracker, FillCallback as BybitFillCallback,
    FillTrackerConfig as BybitFillTrackerConfig, OrderStatusCallback as BybitOrderStatusCallback,
    ReconciliationResult as BybitReconciliationResult, StatusMismatch as BybitStatusMismatch,
};
pub use rest::BybitExchange;
pub use websocket::{
    BybitEvent, BybitWebSocket, BybitWsConfig, OrderUpdate, PositionUpdate, WalletUpdate,
};

// Public WebSocket exports
pub use public_websocket::{
    BybitCandle, BybitCategory, BybitPublicEvent, BybitPublicWebSocket, BybitPublicWsConfig,
    BybitTicker, BybitTrade,
};

// Provider exports
pub use provider::{
    BybitProvider, BybitProviderConfig, WS_PUBLIC_LINEAR_MAINNET, WS_PUBLIC_LINEAR_TESTNET,
    WS_PUBLIC_SPOT_MAINNET, WS_PUBLIC_SPOT_TESTNET,
};
