//! # JANUS Limit Order Book (LOB) Simulator
//!
//! High-fidelity limit order book simulation for backtesting and market
//! microstructure analysis. Replaces the simple BPS-slippage model in
//! `janus-backtest` and `janus-execution` sim with realistic order book
//! dynamics.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    LOB Simulator                                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
//! │  │ L2 Replay    │───▶│ Order Book   │───▶│ Matching        │   │
//! │  │ (Historical  │    │ (Sorted bid/ │    │ Engine          │   │
//! │  │  snapshots)  │    │  ask levels) │    │ (Price-time     │   │
//! │  └──────────────┘    └──────────────┘    │  priority)      │   │
//! │                                           └────────┬────────┘   │
//! │                                                    │            │
//! │  ┌─────────────────────────────────────────────────▼────────┐  │
//! │  │                   Execution Models                        │  │
//! │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │  │
//! │  │  │ Fill     │ │ Market   │ │ Latency  │ │ Queue      │  │  │
//! │  │  │ Model    │ │ Impact   │ │ Model    │ │ Position   │  │  │
//! │  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                              │                                   │
//! │                              ▼                                   │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │  Integration Layer                                        │   │
//! │  │  • LobSimulator orchestrator                              │   │
//! │  │  • SimEnvironment adapter (execution service)             │   │
//! │  │  • ReplayEngine adapter (backtest crate)                  │   │
//! │  │  • TemporalFortress compatible (zero-lookahead)           │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Modules
//!
//! - [`orderbook`] — Core order book data structure (sorted price levels, depth)
//! - [`matching_engine`] — Price-time priority matching with partial fills
//! - [`order_types`] — Order type definitions (market, limit, stop, iceberg, etc.)
//! - [`market_impact`] — Market impact models (linear, square-root, Almgren-Chriss)
//! - [`fill_model`] — Fill probability and queue position estimation
//! - [`latency`] — Network and exchange latency simulation
//! - [`l2_replay`] — Historical L2/L3 order book data replay
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_lob::*;
//!
//! // Create an order book
//! let mut book = OrderBook::new("BTC/USDT");
//!
//! // Add resting orders (from L2 snapshot)
//! book.add_level(Side::Bid, PriceLevel::new(dec!(67000.0), dec!(1.5)));
//! book.add_level(Side::Bid, PriceLevel::new(dec!(66999.0), dec!(3.2)));
//! book.add_level(Side::Ask, PriceLevel::new(dec!(67001.0), dec!(0.8)));
//! book.add_level(Side::Ask, PriceLevel::new(dec!(67002.0), dec!(2.1)));
//!
//! // Create a matching engine with market impact
//! let impact_model = MarketImpactModel::linear(0.1); // 10bps per unit
//! let latency_model = LatencyModel::normal(Duration::from_millis(5), Duration::from_millis(2));
//! let mut engine = MatchingEngine::new(impact_model, latency_model);
//!
//! // Submit a market order
//! let order = Order::market(Side::Buy, dec!(0.5));
//! let fills = engine.submit(&mut book, order)?;
//!
//! for fill in &fills {
//!     println!("Filled {} @ {} (impact: {} bps)",
//!         fill.quantity, fill.price, fill.impact_bps);
//! }
//! ```

pub mod fill_model;
pub mod l2_replay;
pub mod latency;
pub mod market_impact;
pub mod matching_engine;
pub mod order_types;
pub mod orderbook;

// Re-export primary types
pub use fill_model::{FillModel, FillProbability, QueuePosition};
pub use l2_replay::{L2Event, L2Replay, L2ReplayConfig, L2ReplayError};
pub use latency::{LatencyEstimate, LatencyModel, LatencyStats};
pub use market_impact::{ImpactEstimate, MarketImpactModel};
pub use matching_engine::{
    Fill, MatchResult, MatchingEngine, MatchingEngineConfig, MatchingEngineError,
    MatchingEngineStats,
};
pub use order_types::{Order, OrderId, OrderStatus, OrderType, Side, TimeInForce};
pub use orderbook::{
    BookSide, OrderBook, OrderBookDeltaBatch, OrderBookError, OrderBookSnapshot, PriceLevel,
};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// LobSimulator — Top-level orchestrator
// ---------------------------------------------------------------------------

/// Top-level LOB simulation orchestrator.
///
/// Wires together `OrderBook`, `MatchingEngine`, `MarketImpactModel`,
/// `LatencyModel`, and `FillModel` into a single simulation entry point.
/// This is the primary interface for backtest and execution sim layers.
///
/// # Example
///
/// ```rust,ignore
/// use janus_lob::*;
/// use rust_decimal_macros::dec;
///
/// let mut sim = LobSimulator::new("BTC/USDT");
/// sim.apply_snapshot(snapshot);
/// let result = sim.submit_order(Order::market(Side::Buy, dec!(1.0)))?;
/// println!("Filled: {} @ avg {}", result.total_filled, result.avg_fill_price.unwrap());
/// ```
pub struct LobSimulator {
    /// The order book being simulated.
    book: OrderBook,
    /// The matching engine for order execution.
    engine: MatchingEngine,
    /// Fill model for queue position and fill probability estimation.
    fill_model: FillModel,
}

/// Configuration for the LOB simulator.
#[derive(Debug, Clone)]
pub struct LobSimulatorConfig {
    /// Symbol to simulate.
    pub symbol: String,
    /// Matching engine configuration.
    pub engine_config: MatchingEngineConfig,
    /// Queue position model for fill estimation.
    pub queue_model: fill_model::QueuePositionModel,
    /// Maximum depth per side (None = unlimited).
    pub max_depth: Option<usize>,
}

impl Default for LobSimulatorConfig {
    fn default() -> Self {
        Self {
            symbol: "SIM".to_string(),
            engine_config: MatchingEngineConfig::default(),
            queue_model: fill_model::QueuePositionModel::BackOfQueue,
            max_depth: None,
        }
    }
}

impl LobSimulator {
    /// Create a new LOB simulator with default configuration for the given symbol.
    pub fn new(symbol: impl Into<String>) -> Self {
        let sym = symbol.into();
        Self {
            book: OrderBook::new(&sym),
            engine: MatchingEngine::new(MatchingEngineConfig::default()),
            fill_model: FillModel::new(fill_model::QueuePositionModel::BackOfQueue),
        }
    }

    /// Create a new LOB simulator with custom configuration.
    pub fn with_config(config: LobSimulatorConfig) -> Self {
        let mut book = OrderBook::new(&config.symbol);
        if let Some(max) = config.max_depth {
            book = book.with_max_depth(max);
        }
        Self {
            book,
            engine: MatchingEngine::new(config.engine_config),
            fill_model: FillModel::new(config.queue_model),
        }
    }

    // ── Book management ────────────────────────────────────────────────

    /// Apply an L2 snapshot to the order book.
    pub fn apply_snapshot(&mut self, snapshot: OrderBookSnapshot) -> Result<(), OrderBookError> {
        self.book.apply_snapshot(snapshot)
    }

    /// Apply an L2 delta to the order book.
    pub fn apply_delta(&mut self, delta: orderbook::OrderBookDelta) -> Result<(), OrderBookError> {
        self.book.apply_delta(delta)
    }

    /// Apply a batch of L2 deltas atomically.
    pub fn apply_delta_batch(&mut self, batch: OrderBookDeltaBatch) -> Result<(), OrderBookError> {
        self.book.apply_delta_batch(batch)
    }

    // ── Order submission ───────────────────────────────────────────────

    /// Submit an order to the matching engine against the current book.
    pub fn submit_order(&mut self, order: Order) -> Result<MatchResult, MatchingEngineError> {
        self.engine.submit(&mut self.book, order)
    }

    /// Check and trigger any pending stop orders.
    ///
    /// `last_trade_price` is used to determine whether stop orders should
    /// be triggered (buy stops trigger at or above, sell stops at or below).
    pub fn check_stops(
        &mut self,
        last_trade_price: rust_decimal::Decimal,
    ) -> Vec<Result<MatchResult, MatchingEngineError>> {
        self.engine.check_stops(&mut self.book, last_trade_price)
    }

    /// Cancel a pending stop order by ID.
    pub fn cancel_stop(&mut self, order_id: &OrderId) -> Option<Order> {
        self.engine.cancel_stop(order_id)
    }

    /// Cancel all pending stop orders.
    pub fn cancel_all_stops(&mut self) -> Vec<Order> {
        self.engine.cancel_all_stops()
    }

    // ── Fill probability ───────────────────────────────────────────────

    /// Estimate fill probability for a resting limit order at the given
    /// price level.
    pub fn estimate_fill_probability(
        &mut self,
        our_quantity: rust_decimal::Decimal,
        price_level_quantity: rust_decimal::Decimal,
        expected_volume: rust_decimal::Decimal,
    ) -> FillProbability {
        self.fill_model.estimate_fill_probability(
            our_quantity,
            price_level_quantity,
            None,
            expected_volume,
        )
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Get a reference to the order book.
    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    /// Get a mutable reference to the order book.
    pub fn book_mut(&mut self) -> &mut OrderBook {
        &mut self.book
    }

    /// Get a reference to the matching engine.
    pub fn engine(&self) -> &MatchingEngine {
        &self.engine
    }

    /// Get engine statistics.
    pub fn engine_stats(&self) -> &MatchingEngineStats {
        self.engine.stats()
    }

    /// Get fill model statistics.
    pub fn fill_model_stats(&self) -> &fill_model::FillModelStats {
        self.fill_model.stats()
    }

    /// Get the symbol being simulated.
    pub fn symbol(&self) -> &str {
        self.book.symbol()
    }

    /// Halt the matching engine.
    pub fn halt(&mut self, reason: &str) {
        self.engine.halt(reason);
    }

    /// Resume the matching engine.
    pub fn resume(&mut self) {
        self.engine.resume();
    }

    /// Check if the matching engine is halted.
    pub fn is_halted(&self) -> bool {
        self.engine.is_halted()
    }

    /// Reset the simulator (book + engine stats).
    pub fn reset(&mut self) {
        self.book.reset();
        self.engine.stats_mut().reset();
    }
}

impl std::fmt::Debug for LobSimulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LobSimulator")
            .field("symbol", &self.book.symbol())
            .field("bid_levels", &self.book.bids().len())
            .field("ask_levels", &self.book.asks().len())
            .field("engine_halted", &self.engine.is_halted())
            .field("pending_stops", &self.engine.pending_stops())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_crate_version() {
        assert!(!VERSION.is_empty());
    }

    // ── LobSimulator unit tests ────────────────────────────────────────

    #[test]
    fn test_lob_simulator_new() {
        let sim = LobSimulator::new("BTC/USDT");
        assert_eq!(sim.symbol(), "BTC/USDT");
        assert!(sim.book().is_empty());
        assert!(!sim.is_halted());
    }

    #[test]
    fn test_lob_simulator_with_config() {
        let config = LobSimulatorConfig {
            symbol: "ETH/USDT".to_string(),
            engine_config: MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
            queue_model: fill_model::QueuePositionModel::Uniform,
            max_depth: Some(50),
        };
        let sim = LobSimulator::with_config(config);
        assert_eq!(sim.symbol(), "ETH/USDT");
    }

    #[test]
    fn test_lob_simulator_snapshot_and_submit() {
        let mut sim = LobSimulator::with_config(LobSimulatorConfig {
            symbol: "BTC/USDT".to_string(),
            engine_config: MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
            ..Default::default()
        });

        let snapshot = OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![
                PriceLevel::new(dec!(67000.0), dec!(1.5)),
                PriceLevel::new(dec!(66999.0), dec!(3.2)),
            ],
            asks: vec![
                PriceLevel::new(dec!(67001.0), dec!(0.8)),
                PriceLevel::new(dec!(67002.0), dec!(2.1)),
            ],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        };

        sim.apply_snapshot(snapshot).unwrap();
        assert!(!sim.book().is_empty());
        assert_eq!(sim.book().bids().len(), 2);
        assert_eq!(sim.book().asks().len(), 2);

        // Submit a market buy
        let order = Order::market(Side::Buy, dec!(0.5)).with_symbol("BTC/USDT");
        let result = sim.submit_order(order).unwrap();
        assert!(result.has_fills());
        assert_eq!(result.total_filled, dec!(0.5));
        assert_eq!(result.fills[0].price, dec!(67001.0));
    }

    #[test]
    fn test_lob_simulator_halt_resume() {
        let mut sim = LobSimulator::new("TEST");
        assert!(!sim.is_halted());

        sim.halt("test");
        assert!(sim.is_halted());

        sim.resume();
        assert!(!sim.is_halted());
    }

    #[test]
    fn test_lob_simulator_reset() {
        let mut sim = LobSimulator::with_config(LobSimulatorConfig {
            symbol: "BTC/USDT".to_string(),
            engine_config: MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
            ..Default::default()
        });

        let snapshot = OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![PriceLevel::new(dec!(67000.0), dec!(1.5))],
            asks: vec![PriceLevel::new(dec!(67001.0), dec!(0.8))],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        };
        sim.apply_snapshot(snapshot).unwrap();
        assert!(!sim.book().is_empty());

        sim.reset();
        assert!(sim.book().is_empty());
    }

    #[test]
    fn test_lob_simulator_debug() {
        let sim = LobSimulator::new("BTC/USDT");
        let debug = format!("{:?}", sim);
        assert!(debug.contains("LobSimulator"));
        assert!(debug.contains("BTC/USDT"));
    }

    // ── Integration test: snapshot → matching engine → fill validation ─

    #[test]
    fn test_integration_snapshot_to_fills() {
        let mut sim = LobSimulator::with_config(LobSimulatorConfig {
            symbol: "BTC/USDT".to_string(),
            engine_config: MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
            ..Default::default()
        });

        // Build a realistic book
        let snapshot = OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![
                PriceLevel::new(dec!(67000.0), dec!(1.5)),
                PriceLevel::new(dec!(66999.0), dec!(3.2)),
                PriceLevel::new(dec!(66998.5), dec!(0.75)),
                PriceLevel::new(dec!(66997.0), dec!(5.0)),
            ],
            asks: vec![
                PriceLevel::new(dec!(67001.0), dec!(0.8)),
                PriceLevel::new(dec!(67002.0), dec!(2.1)),
                PriceLevel::new(dec!(67003.0), dec!(1.3)),
                PriceLevel::new(dec!(67005.0), dec!(4.0)),
            ],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        };
        sim.apply_snapshot(snapshot).unwrap();

        // Submit market buy that sweeps multiple levels
        let order = Order::market(Side::Buy, dec!(3.0)).with_symbol("BTC/USDT");
        let result = sim.submit_order(order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Filled);
        assert_eq!(result.total_filled, dec!(3.0));
        // Asks: 0.8 @ 67001, 2.1 @ 67002, 1.3 @ 67003, 4.0 @ 67005
        // Buy 3.0: 0.8 @ 67001 (exhausted) + 2.1 @ 67002 (exhausted) + 0.1 @ 67003 = 3.0
        assert_eq!(result.fills.len(), 3);

        // Validate fill prices
        assert_eq!(result.fills[0].price, dec!(67001.0));
        assert_eq!(result.fills[0].quantity, dec!(0.8));
        assert_eq!(result.fills[1].price, dec!(67002.0));
        assert_eq!(result.fills[1].quantity, dec!(2.1));
        assert_eq!(result.fills[2].price, dec!(67003.0));
        assert_eq!(result.fills[2].quantity, dec!(0.1));

        // Validate VWAP
        let expected_vwap =
            (dec!(0.8) * dec!(67001.0) + dec!(2.1) * dec!(67002.0) + dec!(0.1) * dec!(67003.0))
                / dec!(3.0);
        let actual_vwap = result.avg_fill_price.unwrap();
        let diff: f64 = (actual_vwap - expected_vwap)
            .abs()
            .try_into()
            .unwrap_or(1.0);
        assert!(diff < 0.01);

        // Book should reflect consumed liquidity
        assert!(sim.book().asks().get_level(dec!(67001.0)).is_none()); // Fully consumed
        assert!(sim.book().asks().get_level(dec!(67002.0)).is_none()); // Fully consumed
        // 67003 should have 1.3 - 0.1 = 1.2 remaining
        let remaining_67003 = sim.book().asks().get_level(dec!(67003.0)).unwrap();
        assert_eq!(remaining_67003.quantity, dec!(1.2));
    }

    // ── Integration test: delta batch → book state ─────────────────────

    #[test]
    fn test_integration_delta_batch_consistency() {
        let mut sim = LobSimulator::with_config(LobSimulatorConfig {
            symbol: "BTC/USDT".to_string(),
            engine_config: MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
            ..Default::default()
        });

        // Initial snapshot
        let snapshot = OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![PriceLevel::new(dec!(50000.0), dec!(1.0))],
            asks: vec![PriceLevel::new(dec!(50001.0), dec!(1.0))],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        };
        sim.apply_snapshot(snapshot).unwrap();

        // Apply a batch of deltas
        let mut batch = OrderBookDeltaBatch::new("BTC/USDT", 2);
        batch.add(Side::Buy, dec!(49999.0), dec!(2.0));
        batch.add(Side::Buy, dec!(50000.0), dec!(3.0)); // Update existing
        batch.add(Side::Sell, dec!(50001.0), dec!(0.0)); // Remove
        batch.add(Side::Sell, dec!(50002.0), dec!(1.5)); // New level

        sim.apply_delta_batch(batch).unwrap();

        assert_eq!(sim.book().bids().len(), 2);
        assert_eq!(
            sim.book().bids().get_level(dec!(50000.0)).unwrap().quantity,
            dec!(3.0)
        );
        assert_eq!(
            sim.book().bids().get_level(dec!(49999.0)).unwrap().quantity,
            dec!(2.0)
        );
        assert!(sim.book().asks().get_level(dec!(50001.0)).is_none());
        assert_eq!(
            sim.book().asks().get_level(dec!(50002.0)).unwrap().quantity,
            dec!(1.5)
        );
    }

    // ── Integration test: stop order lifecycle ─────────────────────────

    #[test]
    fn test_integration_stop_order_lifecycle() {
        let mut sim = LobSimulator::with_config(LobSimulatorConfig {
            symbol: "BTC/USDT".to_string(),
            engine_config: MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
            ..Default::default()
        });

        let snapshot = OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![
                PriceLevel::new(dec!(50000.0), dec!(1.0)),
                PriceLevel::new(dec!(49999.0), dec!(2.0)),
                PriceLevel::new(dec!(49998.0), dec!(3.0)),
            ],
            asks: vec![
                PriceLevel::new(dec!(50001.0), dec!(1.0)),
                PriceLevel::new(dec!(50002.0), dec!(2.0)),
            ],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        };
        sim.apply_snapshot(snapshot).unwrap();

        // Place a sell stop
        let stop = Order::stop_market(Side::Sell, dec!(49999.0), dec!(0.5)).with_symbol("BTC/USDT");
        sim.submit_order(stop).unwrap();
        assert_eq!(sim.engine().pending_stops(), 1);

        // Cancel the stop
        let cancelled = sim.cancel_all_stops();
        assert_eq!(cancelled.len(), 1);
        assert_eq!(sim.engine().pending_stops(), 0);
    }

    // ── Stress test ────────────────────────────────────────────────────

    #[test]
    fn test_stress_rapid_order_submission() {
        let mut sim = LobSimulator::with_config(LobSimulatorConfig {
            symbol: "BTC/USDT".to_string(),
            engine_config: MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
            ..Default::default()
        });

        // Build a deep book (1000 levels per side)
        let bids: Vec<PriceLevel> = (0..1000)
            .map(|i| {
                PriceLevel::new(
                    dec!(50000.0) - rust_decimal::Decimal::from(i) * dec!(0.5),
                    dec!(1.0),
                )
            })
            .collect();
        let asks: Vec<PriceLevel> = (0..1000)
            .map(|i| {
                PriceLevel::new(
                    dec!(50001.0) + rust_decimal::Decimal::from(i) * dec!(0.5),
                    dec!(1.0),
                )
            })
            .collect();

        sim.apply_snapshot(OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids,
            asks,
            timestamp: chrono::Utc::now(),
            sequence: 1,
        })
        .unwrap();

        // Submit 1000 small market orders rapidly
        let mut total_filled = rust_decimal::Decimal::ZERO;
        for i in 0..1000 {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            let order = Order::market(side, dec!(0.01)).with_symbol("BTC/USDT");
            if let Ok(result) = sim.submit_order(order) {
                total_filled += result.total_filled;
            }
        }

        // Should have filled all or nearly all orders
        assert!(
            total_filled > dec!(9.0),
            "Expected substantial fills, got {}",
            total_filled
        );
        assert!(
            sim.engine_stats().orders_submitted >= 1000,
            "Expected 1000+ submissions"
        );
    }
}
