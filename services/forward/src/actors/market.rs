//! Market data ingestion actor.
//!
//! # Status
//!
//! Market data ingestion is handled directly by the `EventLoop` via the
//! Bybit/Kraken WebSocket connection (`janus_bybit_client::BybitWebSocket`).
//! This actor is retained as a structural placeholder for a future refactor
//! that decouples market data ingestion into its own actor with:
//!
//! - Multi-exchange support (Kraken REST + Bybit WS)
//! - Automatic reconnection with exponential backoff
//! - Tick normalization across exchange formats
//! - Configurable symbol routing
//!
//! For the current production path, see `event_loop.rs` which connects
//! directly to the exchange WebSocket and feeds ticks into the strategy
//! pipeline.

use anyhow::Result;
use common::Tick;
use tokio::sync::mpsc;
use tracing::{debug, info};

/// Market data actor
///
/// Provides a channel-based interface for receiving normalized ticks.
/// Currently a placeholder — the `EventLoop` handles market data directly.
#[allow(dead_code)]
pub struct MarketActor {
    symbol: String,
    tx: mpsc::Sender<Tick>,
}

impl MarketActor {
    /// Create a new market actor for the given symbol.
    ///
    /// `tx` is the channel sender that ticks will be forwarded to once
    /// the actor's `run` loop is connected to an exchange WebSocket.
    pub fn new(symbol: String, tx: mpsc::Sender<Tick>) -> Self {
        Self { symbol, tx }
    }

    /// Run the market data ingestion loop.
    ///
    /// Currently waits indefinitely — market data flows through the
    /// `EventLoop`'s direct WebSocket connection instead. This method
    /// exists so the actor can be spawned without panicking; it will
    /// be replaced with a real WebSocket consumer when multi-exchange
    /// support is implemented.
    pub async fn run(&mut self) -> Result<()> {
        info!(
            "Market actor started for symbol: {} (data flows via EventLoop WebSocket)",
            self.symbol
        );

        // Sleep indefinitely — the EventLoop handles market data directly.
        // When this actor is activated, this will be replaced with a
        // WebSocket connection loop that sends ticks via self.tx.
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            debug!(
                "Market actor idle for {} — data handled by EventLoop",
                self.symbol
            );
        }
    }
}
