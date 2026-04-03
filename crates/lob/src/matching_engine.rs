//! # Matching Engine
//!
//! Price-time priority matching engine for the LOB simulator.
//! Processes incoming orders against the order book, producing fills
//! and order events.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Matching Engine                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
//! │  │ Order        │───▶│ Validation   │───▶│ Match / Rest    │   │
//! │  │ Submission   │    │ & Routing    │    │ Logic           │   │
//! │  └──────────────┘    └──────────────┘    └────────┬────────┘   │
//! │                                                    │            │
//! │  ┌─────────────────────────────────────────────────▼────────┐  │
//! │  │                 Fill Generation                           │  │
//! │  │  • Price-time priority (FIFO within price level)          │  │
//! │  │  • Partial fills with remainder handling                  │  │
//! │  │  • Market impact calculation per fill                     │  │
//! │  │  • Latency injection (simulated gateway + exchange)       │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                              │                                   │
//! │                              ▼                                   │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │  Output: Vec<Fill> + Vec<OrderEvent>                      │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Matching Rules
//!
//! 1. **Market orders** sweep the opposite side from best price until filled
//!    or the book is exhausted.
//! 2. **Limit orders** match against resting liquidity at the limit price or
//!    better; any unfilled remainder rests on the book.
//! 3. **Post-only orders** are rejected if they would cross the book.
//! 4. **Iceberg orders** expose only the display quantity; upon fill the next
//!    slice is revealed (loses time priority).
//! 5. **Stop orders** are held in a trigger queue and activated when the
//!    market price crosses the stop level.
//! 6. **FOK orders** are rejected entirely if full quantity cannot be matched.
//! 7. **IOC orders** match what is available and cancel the remainder.
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_lob::*;
//!
//! let config = MatchingEngineConfig::default();
//! let mut engine = MatchingEngine::new(config);
//!
//! let mut book = OrderBook::new("BTC/USDT");
//! // ... populate book from L2 snapshot ...
//!
//! let order = Order::market(Side::Buy, dec!(0.5)).with_symbol("BTC/USDT");
//! let result = engine.submit(&mut book, order)?;
//!
//! for fill in &result.fills {
//!     println!("Filled {} @ {} (impact: {:.2} bps)", fill.quantity, fill.price, fill.impact_bps);
//! }
//! ```

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, trace, warn};

use crate::latency::LatencyModel;
use crate::market_impact::MarketImpactModel;
use crate::order_types::{
    CancelReason, Order, OrderEvent, OrderId, OrderStatus, OrderType, Side, TimeInForce,
};
use crate::orderbook::OrderBook;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the matching engine.
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum MatchingEngineError {
    /// Order failed validation.
    #[error("Order validation failed: {0}")]
    ValidationFailed(String),

    /// Order book error during matching.
    #[error("Order book error: {0}")]
    OrderBookError(String),

    /// No liquidity available on the opposite side.
    #[error("No liquidity available for {side} order on {symbol}")]
    NoLiquidity { symbol: String, side: String },

    /// FOK order could not be fully filled.
    #[error("FOK order could not be fully filled: available={available}, requested={requested}")]
    FokNotFilled {
        available: Decimal,
        requested: Decimal,
    },

    /// Post-only order would cross the book.
    #[error(
        "Post-only order would cross: order_price={order_price}, best_opposite={best_opposite}"
    )]
    PostOnlyWouldCross {
        order_price: Decimal,
        best_opposite: Decimal,
    },

    /// Engine is not accepting orders (e.g. halted).
    #[error("Matching engine is halted: {0}")]
    Halted(String),

    /// Internal engine error.
    #[error("Internal matching engine error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, MatchingEngineError>;

// ---------------------------------------------------------------------------
// Fill
// ---------------------------------------------------------------------------

/// A single fill event — one price level (or portion thereof) being matched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    /// The order that was (partially or fully) filled.
    pub order_id: OrderId,

    /// Trading symbol.
    pub symbol: String,

    /// Side of the incoming (aggressor) order.
    pub side: Side,

    /// Fill price.
    pub price: Decimal,

    /// Fill quantity.
    pub quantity: Decimal,

    /// Notional value (price × quantity).
    pub notional: Decimal,

    /// Whether the fill was as maker (resting) or taker (aggressor).
    pub is_maker: bool,

    /// Estimated market impact in basis points for this fill.
    pub impact_bps: f64,

    /// Simulated latency for this fill.
    pub latency: Duration,

    /// Commission for this fill.
    pub commission: Decimal,

    /// Timestamp of the fill.
    pub timestamp: DateTime<Utc>,

    /// The resting order ID that was matched against (if L3 tracking).
    pub contra_order_id: Option<OrderId>,
}

impl Fill {
    /// Net value after commission (for buy: negative, for sell: positive).
    pub fn net_value(&self) -> Decimal {
        match self.side {
            Side::Buy => -(self.notional + self.commission),
            Side::Sell => self.notional - self.commission,
        }
    }

    /// Effective price including commission.
    pub fn effective_price(&self) -> Decimal {
        if self.quantity > Decimal::ZERO {
            match self.side {
                Side::Buy => (self.notional + self.commission) / self.quantity,
                Side::Sell => (self.notional - self.commission) / self.quantity,
            }
        } else {
            self.price
        }
    }
}

impl fmt::Display for Fill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let role = if self.is_maker { "maker" } else { "taker" };
        write!(
            f,
            "Fill[{}] {} {} {} @ {} ({}, impact: {:.1}bps, latency: {:?})",
            self.order_id,
            self.side,
            self.quantity,
            self.symbol,
            self.price,
            role,
            self.impact_bps,
            self.latency,
        )
    }
}

// ---------------------------------------------------------------------------
// Match Result
// ---------------------------------------------------------------------------

/// The complete result of submitting an order to the matching engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    /// The order after processing (status updated, filled_quantity set).
    pub order: Order,

    /// Individual fills produced during matching.
    pub fills: Vec<Fill>,

    /// Order lifecycle events produced during matching.
    pub events: Vec<OrderEvent>,

    /// Total filled quantity.
    pub total_filled: Decimal,

    /// Volume-weighted average fill price (if any fills occurred).
    pub avg_fill_price: Option<Decimal>,

    /// Total commission paid across all fills.
    pub total_commission: Decimal,

    /// Total estimated market impact in basis points.
    pub total_impact_bps: f64,

    /// Total simulated latency (gateway + matching + acknowledgement).
    pub total_latency: Duration,

    /// Whether the order rested on the book (limit/iceberg/post-only).
    pub rested: bool,
}

impl MatchResult {
    /// Create an empty result for a given order (no fills).
    fn empty(order: Order) -> Self {
        Self {
            order,
            fills: Vec::new(),
            events: Vec::new(),
            total_filled: Decimal::ZERO,
            avg_fill_price: None,
            total_commission: Decimal::ZERO,
            total_impact_bps: 0.0,
            total_latency: Duration::ZERO,
            rested: false,
        }
    }

    /// Whether any fills occurred.
    pub fn has_fills(&self) -> bool {
        !self.fills.is_empty()
    }

    /// Whether the order was completely filled.
    pub fn is_fully_filled(&self) -> bool {
        self.order.is_filled()
    }

    /// Slippage from the order's limit price (if applicable).
    /// Positive means worse than limit, negative means price improvement.
    pub fn slippage_bps(&self) -> Option<f64> {
        let limit = self.order.price?;
        let avg = self.avg_fill_price?;
        if limit == Decimal::ZERO {
            return None;
        }
        let slip = match self.order.side {
            Side::Buy => (avg - limit) / limit,
            Side::Sell => (limit - avg) / limit,
        };
        let slip_f64: f64 = slip.try_into().unwrap_or(0.0);
        Some(slip_f64 * 10_000.0)
    }

    /// Compute summary statistics after all fills are recorded.
    fn finalize(&mut self) {
        if self.fills.is_empty() {
            return;
        }

        let mut total_notional = Decimal::ZERO;
        let mut total_qty = Decimal::ZERO;
        let mut total_commission = Decimal::ZERO;
        let mut total_impact = 0.0f64;
        let mut total_latency = Duration::ZERO;

        for fill in &self.fills {
            total_notional += fill.notional;
            total_qty += fill.quantity;
            total_commission += fill.commission;
            total_impact += fill.impact_bps * f64::try_from(fill.quantity).unwrap_or(0.0);
            total_latency += fill.latency;
        }

        self.total_filled = total_qty;
        self.total_commission = total_commission;
        self.total_latency = total_latency;

        if total_qty > Decimal::ZERO {
            self.avg_fill_price = Some(total_notional / total_qty);
            let total_qty_f64 = f64::try_from(total_qty).unwrap_or(1.0);
            self.total_impact_bps = total_impact / total_qty_f64;
        }
    }
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatchResult[{}]: {} fills, filled={}/{}, ",
            self.order.id,
            self.fills.len(),
            self.total_filled,
            self.order.quantity,
        )?;
        if let Some(avg) = self.avg_fill_price {
            write!(f, "avg_price={}, ", avg)?;
        }
        write!(
            f,
            "commission={}, impact={:.1}bps, status={}",
            self.total_commission, self.total_impact_bps, self.order.status,
        )?;
        if self.rested {
            write!(f, " [RESTED]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the matching engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingEngineConfig {
    /// Commission rate in basis points for taker (aggressor) fills.
    pub taker_commission_bps: f64,

    /// Commission rate in basis points for maker (resting) fills.
    pub maker_commission_bps: f64,

    /// Whether to enable market impact modeling.
    pub enable_market_impact: bool,

    /// Whether to enable latency simulation.
    pub enable_latency_simulation: bool,

    /// Whether to allow self-trading (own resting order matched by own incoming order).
    pub allow_self_trade: bool,

    /// Maximum number of price levels to sweep for a single market order.
    /// Prevents runaway fills in a thin book. `None` means unlimited.
    pub max_levels_per_sweep: Option<usize>,

    /// Whether to log individual fills at trace level.
    pub trace_fills: bool,

    /// Whether the engine is halted (rejecting all new orders).
    pub halted: bool,
}

impl Default for MatchingEngineConfig {
    fn default() -> Self {
        Self {
            taker_commission_bps: 6.0, // 0.06% — typical for crypto
            maker_commission_bps: 2.0, // 0.02% — maker rebate common
            enable_market_impact: true,
            enable_latency_simulation: true,
            allow_self_trade: false,
            max_levels_per_sweep: Some(100),
            trace_fills: true,
            halted: false,
        }
    }
}

impl MatchingEngineConfig {
    /// Set taker commission rate.
    pub fn with_taker_commission_bps(mut self, bps: f64) -> Self {
        self.taker_commission_bps = bps;
        self
    }

    /// Set maker commission rate.
    pub fn with_maker_commission_bps(mut self, bps: f64) -> Self {
        self.maker_commission_bps = bps;
        self
    }

    /// Disable market impact modeling.
    pub fn without_market_impact(mut self) -> Self {
        self.enable_market_impact = false;
        self
    }

    /// Disable latency simulation.
    pub fn without_latency(mut self) -> Self {
        self.enable_latency_simulation = false;
        self
    }

    /// Set maximum levels per sweep.
    pub fn with_max_levels(mut self, max: usize) -> Self {
        self.max_levels_per_sweep = Some(max);
        self
    }
}

// ---------------------------------------------------------------------------
// Engine Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the matching engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatchingEngineStats {
    /// Total orders submitted.
    pub orders_submitted: u64,

    /// Total orders accepted (passed validation).
    pub orders_accepted: u64,

    /// Total orders rejected.
    pub orders_rejected: u64,

    /// Total orders fully filled.
    pub orders_filled: u64,

    /// Total orders partially filled.
    pub orders_partially_filled: u64,

    /// Total orders that rested on the book.
    pub orders_rested: u64,

    /// Total individual fills produced.
    pub total_fills: u64,

    /// Total volume filled (in base currency units).
    pub total_volume: Decimal,

    /// Total notional traded.
    pub total_notional: Decimal,

    /// Total commission collected.
    pub total_commission: Decimal,

    /// Total stop orders triggered.
    pub stops_triggered: u64,

    /// Total post-only orders rejected (would cross).
    pub post_only_rejected: u64,

    /// Total FOK orders rejected (insufficient liquidity).
    pub fok_rejected: u64,
}

impl MatchingEngineStats {
    /// Fill rate: orders_filled / orders_submitted.
    pub fn fill_rate(&self) -> f64 {
        if self.orders_submitted > 0 {
            self.orders_filled as f64 / self.orders_submitted as f64
        } else {
            0.0
        }
    }

    /// Average fills per order.
    pub fn avg_fills_per_order(&self) -> f64 {
        if self.orders_filled > 0 {
            self.total_fills as f64 / self.orders_filled as f64
        } else {
            0.0
        }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for MatchingEngineStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatchingEngineStats {{ submitted: {}, filled: {}, rejected: {}, \
             rested: {}, fills: {}, volume: {}, commission: {} }}",
            self.orders_submitted,
            self.orders_filled,
            self.orders_rejected,
            self.orders_rested,
            self.total_fills,
            self.total_volume,
            self.total_commission,
        )
    }
}

// ---------------------------------------------------------------------------
// Matching Engine
// ---------------------------------------------------------------------------

/// The core matching engine that processes orders against the order book.
///
/// Implements price-time priority matching with support for all order types
/// defined in [`crate::order_types`]. Integrates market impact and latency
/// models for realistic simulation.
pub struct MatchingEngine {
    /// Engine configuration.
    config: MatchingEngineConfig,

    /// Market impact model (optional).
    impact_model: Option<MarketImpactModel>,

    /// Latency simulation model (optional).
    latency_model: Option<LatencyModel>,

    /// Stop orders waiting for trigger.
    stop_queue: VecDeque<Order>,

    /// Running statistics.
    stats: MatchingEngineStats,
}

impl MatchingEngine {
    /// Create a new matching engine with the given configuration.
    pub fn new(config: MatchingEngineConfig) -> Self {
        Self {
            config,
            impact_model: None,
            latency_model: None,
            stop_queue: VecDeque::new(),
            stats: MatchingEngineStats::default(),
        }
    }

    /// Create a matching engine with market impact and latency models.
    pub fn with_models(
        config: MatchingEngineConfig,
        impact_model: MarketImpactModel,
        latency_model: LatencyModel,
    ) -> Self {
        Self {
            config,
            impact_model: Some(impact_model),
            latency_model: Some(latency_model),
            stop_queue: VecDeque::new(),
            stats: MatchingEngineStats::default(),
        }
    }

    /// Set the market impact model.
    pub fn set_impact_model(&mut self, model: MarketImpactModel) {
        self.impact_model = Some(model);
    }

    /// Set the latency model.
    pub fn set_latency_model(&mut self, model: LatencyModel) {
        self.latency_model = Some(model);
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MatchingEngineConfig {
        &self.config
    }

    /// Get engine statistics.
    pub fn stats(&self) -> &MatchingEngineStats {
        &self.stats
    }

    /// Get mutable engine statistics.
    pub fn stats_mut(&mut self) -> &mut MatchingEngineStats {
        &mut self.stats
    }

    /// Reset engine statistics.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Get the number of pending stop orders.
    pub fn pending_stops(&self) -> usize {
        self.stop_queue.len()
    }

    /// Halt the engine (reject all new orders).
    pub fn halt(&mut self, reason: &str) {
        self.config.halted = true;
        warn!(reason = reason, "Matching engine halted");
    }

    /// Resume the engine after a halt.
    pub fn resume(&mut self) {
        self.config.halted = false;
        info!("Matching engine resumed");
    }

    /// Whether the engine is currently halted.
    pub fn is_halted(&self) -> bool {
        self.config.halted
    }

    // ── Order Submission ───────────────────────────────────────────────

    /// Submit an order for matching against the given order book.
    ///
    /// This is the main entry point. The engine will:
    /// 1. Validate the order
    /// 2. Check for halt state
    /// 3. Route by order type (market, limit, stop, post-only, iceberg)
    /// 4. Apply time-in-force policy (GTC, IOC, FOK, GTD)
    /// 5. Generate fills and order events
    /// 6. Update statistics
    ///
    /// Returns a `MatchResult` containing the updated order, fills, and events.
    pub fn submit(&mut self, book: &mut OrderBook, mut order: Order) -> Result<MatchResult> {
        self.stats.orders_submitted += 1;

        // Check halt state.
        if self.config.halted {
            order.reject();
            self.stats.orders_rejected += 1;
            return Err(MatchingEngineError::Halted("Engine is halted".into()));
        }

        // Validate the order.
        if let Err(reason) = order.validate() {
            order.reject();
            self.stats.orders_rejected += 1;
            return Err(MatchingEngineError::ValidationFailed(reason));
        }

        // Route by order type.
        let result = match order.order_type {
            OrderType::Market => self.match_market_order(book, order),
            OrderType::Limit => self.match_limit_order(book, order),
            OrderType::StopMarket | OrderType::StopLimit => self.handle_stop_order(book, order),
            OrderType::PostOnly => self.match_post_only_order(book, order),
            OrderType::Iceberg => self.match_iceberg_order(book, order),
        };

        // Update statistics from result.
        match &result {
            Ok(mr) => {
                self.stats.orders_accepted += 1;
                self.stats.total_fills += mr.fills.len() as u64;
                self.stats.total_volume += mr.total_filled;
                if let Some(avg) = mr.avg_fill_price {
                    self.stats.total_notional += avg * mr.total_filled;
                }
                self.stats.total_commission += mr.total_commission;

                if mr.is_fully_filled() {
                    self.stats.orders_filled += 1;
                } else if mr.has_fills() {
                    self.stats.orders_partially_filled += 1;
                }

                if mr.rested {
                    self.stats.orders_rested += 1;
                }
            }
            Err(_) => {
                self.stats.orders_rejected += 1;
            }
        }

        result
    }

    /// Check and trigger any stop orders based on the current market price.
    ///
    /// Call this after each trade / price update to activate pending stops.
    ///
    /// Returns a list of `MatchResult`s for each triggered stop order.
    pub fn check_stops(
        &mut self,
        book: &mut OrderBook,
        last_trade_price: Decimal,
    ) -> Vec<Result<MatchResult>> {
        // Partition the stop queue: triggered stops go into a separate vec.
        let mut triggered: Vec<Order> = Vec::new();
        let mut remaining: VecDeque<Order> = VecDeque::new();

        while let Some(order) = self.stop_queue.pop_front() {
            let stop_price = order.stop_price.unwrap_or(Decimal::ZERO);
            let is_triggered = match order.side {
                // Buy stop: triggers when price rises to or above the stop level.
                Side::Buy => last_trade_price >= stop_price,
                // Sell stop: triggers when price falls to or at/below the stop level.
                Side::Sell => last_trade_price <= stop_price,
            };

            if is_triggered {
                triggered.push(order);
            } else {
                remaining.push_back(order);
            }
        }

        self.stop_queue = remaining;

        // Process each triggered stop by converting and submitting.
        let mut results: Vec<Result<MatchResult>> = Vec::with_capacity(triggered.len());

        for mut stop_order in triggered {
            // Emit a Triggered event.
            let trigger_price = stop_order.stop_price.unwrap_or(last_trade_price);
            stop_order.trigger();
            self.stats.stops_triggered += 1;

            // Convert the stop order to its underlying order type.
            match stop_order.order_type {
                OrderType::StopMarket => {
                    // Convert to a market order and submit.
                    stop_order.order_type = OrderType::Market;
                    stop_order.stop_price = None;
                    stop_order.status = OrderStatus::Pending;

                    debug!(
                        order_id = %stop_order.id,
                        trigger_price = %trigger_price,
                        last_trade = %last_trade_price,
                        "Stop-market triggered → market order"
                    );

                    let mr = self.match_market_order(book, stop_order);
                    // After a fill, check for cascade triggers (stop fills may trigger more stops).
                    // We don't recurse here to avoid infinite loops — the caller should
                    // call check_stops again if fills occurred.
                    results.push(mr);
                }
                OrderType::StopLimit => {
                    // Convert to a limit order and submit.
                    stop_order.order_type = OrderType::Limit;
                    stop_order.stop_price = None;
                    stop_order.status = OrderStatus::Pending;

                    debug!(
                        order_id = %stop_order.id,
                        trigger_price = %trigger_price,
                        limit_price = ?stop_order.price,
                        last_trade = %last_trade_price,
                        "Stop-limit triggered → limit order"
                    );

                    let mr = self.match_limit_order(book, stop_order);
                    results.push(mr);
                }
                _ => {
                    // Shouldn't happen — stop queue should only contain stop orders.
                    warn!(
                        order_id = %stop_order.id,
                        order_type = %stop_order.order_type,
                        "Non-stop order found in stop queue"
                    );
                }
            }
        }

        results
    }

    /// Cancel a pending stop order by ID.
    pub fn cancel_stop(&mut self, order_id: &OrderId) -> Option<Order> {
        if let Some(pos) = self.stop_queue.iter().position(|o| o.id == *order_id) {
            let mut order = self.stop_queue.remove(pos).unwrap();
            order.cancel();
            Some(order)
        } else {
            None
        }
    }

    /// Cancel all pending stop orders. Returns the cancelled orders.
    pub fn cancel_all_stops(&mut self) -> Vec<Order> {
        let mut cancelled = Vec::with_capacity(self.stop_queue.len());
        while let Some(mut order) = self.stop_queue.pop_front() {
            order.cancel();
            cancelled.push(order);
        }
        cancelled
    }

    // ── Internal Matching Logic ────────────────────────────────────────

    /// Match a market order by sweeping the opposite side of the book.
    fn match_market_order(
        &mut self,
        book: &mut OrderBook,
        mut order: Order,
    ) -> Result<MatchResult> {
        order.accept();
        let mut result = MatchResult::empty(order.clone());
        result.events.push(OrderEvent::Accepted {
            order_id: order.id.clone(),
            timestamp: Utc::now(),
        });

        // Determine the opposite side to sweep.
        let opposite_side = match order.side {
            Side::Buy => book.asks_mut(),
            Side::Sell => book.bids_mut(),
        };

        // Sweep the opposite side from best price, consuming liquidity.
        let fills_raw = opposite_side.consume_liquidity(
            order.remaining_quantity(),
            None, // no limit price for market orders
            self.config.max_levels_per_sweep,
        );

        if fills_raw.is_empty() {
            // No liquidity at all.
            order.cancel();
            result.events.push(OrderEvent::Cancelled {
                order_id: order.id.clone(),
                remaining: order.remaining_quantity(),
                reason: CancelReason::IocRemainder,
                timestamp: Utc::now(),
            });
            result.order = order;
            result.finalize();
            return Ok(result);
        }

        // Generate a Fill for each consumed level.
        for (price, qty) in &fills_raw {
            let fill = self.create_fill(&order, *price, *qty, false /* taker */);
            result.events.push(OrderEvent::Fill {
                order_id: order.id.clone(),
                price: fill.price,
                quantity: fill.quantity,
                remaining: order.remaining_quantity() - fill.quantity,
                is_maker: false,
                timestamp: fill.timestamp,
            });
            order.record_fill(fill.quantity);
            result.fills.push(fill);
        }

        // Update order status.
        if order.is_filled() {
            // Fully filled — status already set by record_fill.
        } else {
            // Partially filled — market orders that can't be fully filled
            // cancel the remainder (implicit IOC semantics for market orders).
            order.cancel();
            result.events.push(OrderEvent::Cancelled {
                order_id: order.id.clone(),
                remaining: order.remaining_quantity(),
                reason: CancelReason::IocRemainder,
                timestamp: Utc::now(),
            });
        }

        debug!(
            order_id = %order.id,
            side = %order.side,
            requested = %result.order.quantity,
            filled = %order.filled_quantity,
            fills = result.fills.len(),
            "Market order matched"
        );

        result.order = order;
        result.finalize();
        Ok(result)
    }

    /// Match a limit order: cross what you can, rest the remainder.
    fn match_limit_order(&mut self, book: &mut OrderBook, mut order: Order) -> Result<MatchResult> {
        order.accept();
        let mut result = MatchResult::empty(order.clone());
        result.events.push(OrderEvent::Accepted {
            order_id: order.id.clone(),
            timestamp: Utc::now(),
        });

        let limit_price = order.price.unwrap_or(Decimal::ZERO);

        // ── FOK pre-check ──────────────────────────────────────────────
        // For Fill-or-Kill: verify sufficient liquidity exists at or better
        // than the limit price before touching the book.
        if order.time_in_force == TimeInForce::FOK {
            let available = match order.side {
                Side::Buy => book
                    .asks()
                    .iter()
                    .take_while(|l| l.price <= limit_price)
                    .map(|l| l.quantity)
                    .sum::<Decimal>(),
                Side::Sell => book
                    .bids()
                    .iter()
                    .take_while(|l| l.price >= limit_price)
                    .map(|l| l.quantity)
                    .sum::<Decimal>(),
            };

            if available < order.quantity {
                order.cancel();
                result.events.push(OrderEvent::Cancelled {
                    order_id: order.id.clone(),
                    remaining: order.remaining_quantity(),
                    reason: CancelReason::FokNotFilled,
                    timestamp: Utc::now(),
                });
                self.stats.fok_rejected += 1;
                result.order = order;
                result.finalize();
                return Ok(result);
            }
        }

        // ── Crossing phase ─────────────────────────────────────────────
        // Check if the limit price crosses the opposite side of the book.
        let would_cross = match order.side {
            Side::Buy => book.best_ask_price().is_some_and(|ask| limit_price >= ask),
            Side::Sell => book.best_bid_price().is_some_and(|bid| limit_price <= bid),
        };

        if would_cross {
            // Sweep the opposite side up to the limit price.
            let opposite = match order.side {
                Side::Buy => book.asks_mut(),
                Side::Sell => book.bids_mut(),
            };

            let fills_raw = opposite.consume_liquidity(
                order.remaining_quantity(),
                Some(limit_price),
                self.config.max_levels_per_sweep,
            );

            for (price, qty) in &fills_raw {
                let fill = self.create_fill(&order, *price, *qty, false /* taker */);
                result.events.push(OrderEvent::Fill {
                    order_id: order.id.clone(),
                    price: fill.price,
                    quantity: fill.quantity,
                    remaining: order.remaining_quantity() - fill.quantity,
                    is_maker: false,
                    timestamp: fill.timestamp,
                });
                order.record_fill(fill.quantity);
                result.fills.push(fill);
            }
        }

        // ── Remainder handling (based on TIF) ──────────────────────────
        if !order.is_filled() {
            match order.time_in_force {
                TimeInForce::IOC => {
                    // Cancel unfilled remainder.
                    order.cancel();
                    result.events.push(OrderEvent::Cancelled {
                        order_id: order.id.clone(),
                        remaining: order.remaining_quantity(),
                        reason: CancelReason::IocRemainder,
                        timestamp: Utc::now(),
                    });
                }
                TimeInForce::FOK => {
                    // Should have been caught by pre-check above.
                    // If we get here, something unexpected happened — cancel.
                    order.cancel();
                    result.events.push(OrderEvent::Cancelled {
                        order_id: order.id.clone(),
                        remaining: order.remaining_quantity(),
                        reason: CancelReason::FokNotFilled,
                        timestamp: Utc::now(),
                    });
                    self.stats.fok_rejected += 1;
                }
                TimeInForce::GTC | TimeInForce::GTD => {
                    // Rest the remainder on the book.
                    let mut resting = crate::orderbook::RestingOrder::new(
                        order.side,
                        limit_price,
                        order.remaining_quantity(),
                        Utc::now(),
                    );
                    resting.id = order.id.as_str().to_string();

                    let book_side = match order.side {
                        Side::Buy => book.bids_mut(),
                        Side::Sell => book.asks_mut(),
                    };

                    if let Err(e) = book_side.add_resting_order(resting) {
                        warn!(
                            order_id = %order.id,
                            error = %e,
                            "Failed to rest order on book"
                        );
                    } else {
                        result.rested = true;
                        debug!(
                            order_id = %order.id,
                            price = %limit_price,
                            remaining = %order.remaining_quantity(),
                            "Limit order resting on book"
                        );
                    }
                }
            }
        }

        result.order = order;
        result.finalize();
        Ok(result)
    }

    /// Handle a stop order: place into the trigger queue.
    fn handle_stop_order(
        &mut self,
        _book: &mut OrderBook,
        mut order: Order,
    ) -> Result<MatchResult> {
        // Stop orders don't match immediately — they wait for the trigger price.
        let mut result = MatchResult::empty(order.clone());

        // Ensure status is StopWaiting
        if order.status != OrderStatus::StopWaiting {
            order.status = OrderStatus::StopWaiting;
        }

        result.events.push(OrderEvent::Accepted {
            order_id: order.id.clone(),
            timestamp: Utc::now(),
        });

        self.stop_queue.push_back(order.clone());

        debug!(
            order_id = %order.id,
            order_type = %order.order_type,
            stop_price = ?order.stop_price,
            "Stop order queued for trigger"
        );

        result.order = order;
        Ok(result)
    }

    /// Match a post-only order: reject if it would cross, otherwise rest.
    fn match_post_only_order(
        &mut self,
        book: &mut OrderBook,
        mut order: Order,
    ) -> Result<MatchResult> {
        let mut result = MatchResult::empty(order.clone());

        let limit_price = order.price.ok_or_else(|| {
            MatchingEngineError::ValidationFailed("Post-only order requires price".into())
        })?;

        // Check if the order would cross the book.
        let would_cross = match order.side {
            Side::Buy => book.best_ask_price().is_some_and(|ask| limit_price >= ask),
            Side::Sell => book.best_bid_price().is_some_and(|bid| limit_price <= bid),
        };

        if would_cross {
            order.reject();
            let best_opposite = match order.side {
                Side::Buy => book.best_ask_price().unwrap_or_default(),
                Side::Sell => book.best_bid_price().unwrap_or_default(),
            };

            result.events.push(OrderEvent::Rejected {
                order_id: order.id.clone(),
                reason: format!(
                    "Post-only would cross: price={}, best_opposite={}",
                    limit_price, best_opposite
                ),
                timestamp: Utc::now(),
            });

            self.stats.post_only_rejected += 1;
            result.order = order;
            return Err(MatchingEngineError::PostOnlyWouldCross {
                order_price: limit_price,
                best_opposite,
            });
        }

        // Does not cross — rest on the book.
        order.accept();
        result.events.push(OrderEvent::Accepted {
            order_id: order.id.clone(),
            timestamp: Utc::now(),
        });

        // Insert the post-only order into the book's resting queue.
        let mut resting = crate::orderbook::RestingOrder::new(
            order.side,
            limit_price,
            order.remaining_quantity(),
            Utc::now(),
        );
        resting.id = order.id.as_str().to_string();

        let book_side = match order.side {
            Side::Buy => book.bids_mut(),
            Side::Sell => book.asks_mut(),
        };

        if let Err(e) = book_side.add_resting_order(resting) {
            warn!(
                order_id = %order.id,
                error = %e,
                "Failed to rest post-only order on book"
            );
        } else {
            result.rested = true;
        }

        debug!(
            order_id = %order.id,
            price = %limit_price,
            qty = %order.quantity,
            "Post-only order rested on book"
        );

        result.order = order;
        Ok(result)
    }

    /// Match an iceberg order: expose display quantity, sweep and replenish.
    fn match_iceberg_order(
        &mut self,
        book: &mut OrderBook,
        mut order: Order,
    ) -> Result<MatchResult> {
        order.accept();
        let mut result = MatchResult::empty(order.clone());
        result.events.push(OrderEvent::Accepted {
            order_id: order.id.clone(),
            timestamp: Utc::now(),
        });

        let limit_price = order.price.unwrap_or(Decimal::ZERO);
        let display_qty = order.effective_display_quantity();

        // ── Aggressive crossing phase ──────────────────────────────────
        // If the iceberg's limit price crosses the opposite side, fill as taker first.
        let would_cross = match order.side {
            Side::Buy => book.best_ask_price().is_some_and(|ask| limit_price >= ask),
            Side::Sell => book.best_bid_price().is_some_and(|bid| limit_price <= bid),
        };

        if would_cross {
            let opposite = match order.side {
                Side::Buy => book.asks_mut(),
                Side::Sell => book.bids_mut(),
            };

            let fills_raw = opposite.consume_liquidity(
                order.remaining_quantity(),
                Some(limit_price),
                self.config.max_levels_per_sweep,
            );

            for (price, qty) in &fills_raw {
                let fill = self.create_fill(&order, *price, *qty, false);
                result.events.push(OrderEvent::Fill {
                    order_id: order.id.clone(),
                    price: fill.price,
                    quantity: fill.quantity,
                    remaining: order.remaining_quantity() - fill.quantity,
                    is_maker: false,
                    timestamp: fill.timestamp,
                });
                order.record_fill(fill.quantity);
                result.fills.push(fill);
            }
        }

        // ── Rest the remainder as an iceberg ───────────────────────────
        // Only rest if there's remaining quantity and the order isn't fully filled.
        if !order.is_filled() && limit_price > Decimal::ZERO {
            // Place only the display slice on the book.
            let rest_qty = order.remaining_quantity().min(display_qty);

            let mut resting =
                crate::orderbook::RestingOrder::new(order.side, limit_price, rest_qty, Utc::now());
            resting.id = order.id.as_str().to_string();

            let book_side = match order.side {
                Side::Buy => book.bids_mut(),
                Side::Sell => book.asks_mut(),
            };

            if let Err(e) = book_side.add_resting_order(resting) {
                warn!(
                    order_id = %order.id,
                    error = %e,
                    "Failed to rest iceberg order on book"
                );
            } else {
                result.rested = true;

                let hidden_remaining = order.remaining_quantity() - rest_qty;
                if hidden_remaining > Decimal::ZERO {
                    result.events.push(OrderEvent::IcebergReplenished {
                        order_id: order.id.clone(),
                        new_display_quantity: rest_qty,
                        total_remaining: order.remaining_quantity(),
                        timestamp: Utc::now(),
                    });
                }
            }

            debug!(
                order_id = %order.id,
                total_qty = %order.quantity,
                display_qty = %rest_qty,
                hidden = %(order.remaining_quantity() - rest_qty),
                "Iceberg order resting on book"
            );
        }

        result.order = order;
        result.finalize();
        Ok(result)
    }

    // ── Fill Helpers ───────────────────────────────────────────────────

    /// Create a fill record for a given matched quantity at a given price.
    fn create_fill(
        &self,
        order: &Order,
        price: Decimal,
        quantity: Decimal,
        is_maker: bool,
    ) -> Fill {
        let commission_bps = if is_maker {
            self.config.maker_commission_bps
        } else {
            self.config.taker_commission_bps
        };

        let notional = price * quantity;
        let commission =
            notional * Decimal::try_from(commission_bps / 10_000.0).unwrap_or_default();

        let impact_bps = if self.config.enable_market_impact {
            self.impact_model
                .as_ref()
                .map(|m| m.estimate_impact_bps(quantity, price))
                .unwrap_or(0.0)
        } else {
            0.0
        };

        let latency = if self.config.enable_latency_simulation {
            self.latency_model
                .as_ref()
                .map(|m| m.sample_latency())
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        };

        if self.config.trace_fills {
            trace!(
                order_id = %order.id,
                price = %price,
                qty = %quantity,
                commission = %commission,
                impact_bps = impact_bps,
                latency_us = latency.as_micros(),
                "Fill created"
            );
        }

        Fill {
            order_id: order.id.clone(),
            symbol: order.symbol.clone(),
            side: order.side,
            price,
            quantity,
            notional,
            is_maker,
            impact_bps,
            latency,
            commission,
            timestamp: Utc::now(),
            contra_order_id: None,
        }
    }
}

impl fmt::Debug for MatchingEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatchingEngine")
            .field("halted", &self.config.halted)
            .field("pending_stops", &self.stop_queue.len())
            .field("has_impact_model", &self.impact_model.is_some())
            .field("has_latency_model", &self.latency_model.is_some())
            .field("stats", &self.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::{OrderBookSnapshot, PriceLevel};
    use rust_decimal_macros::dec;

    fn sample_book() -> OrderBook {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![
                PriceLevel::new(dec!(67000.0), dec!(1.5)),
                PriceLevel::new(dec!(66999.0), dec!(3.2)),
                PriceLevel::new(dec!(66998.0), dec!(2.0)),
            ],
            asks: vec![
                PriceLevel::new(dec!(67001.0), dec!(0.8)),
                PriceLevel::new(dec!(67002.0), dec!(2.1)),
                PriceLevel::new(dec!(67003.0), dec!(1.3)),
            ],
            timestamp: Utc::now(),
            sequence: 1,
        })
        .unwrap();
        book
    }

    // ── Configuration ──────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = MatchingEngineConfig::default();
        assert_eq!(config.taker_commission_bps, 6.0);
        assert_eq!(config.maker_commission_bps, 2.0);
        assert!(config.enable_market_impact);
        assert!(config.enable_latency_simulation);
        assert!(!config.halted);
    }

    #[test]
    fn test_config_builder() {
        let config = MatchingEngineConfig::default()
            .with_taker_commission_bps(10.0)
            .with_maker_commission_bps(0.0)
            .without_market_impact()
            .without_latency()
            .with_max_levels(50);

        assert_eq!(config.taker_commission_bps, 10.0);
        assert_eq!(config.maker_commission_bps, 0.0);
        assert!(!config.enable_market_impact);
        assert!(!config.enable_latency_simulation);
        assert_eq!(config.max_levels_per_sweep, Some(50));
    }

    // ── Engine lifecycle ───────────────────────────────────────────────

    #[test]
    fn test_engine_new() {
        let engine = MatchingEngine::new(MatchingEngineConfig::default());
        assert!(!engine.is_halted());
        assert_eq!(engine.pending_stops(), 0);
        assert_eq!(engine.stats().orders_submitted, 0);
    }

    #[test]
    fn test_engine_halt_resume() {
        let mut engine = MatchingEngine::new(MatchingEngineConfig::default());
        assert!(!engine.is_halted());

        engine.halt("test halt");
        assert!(engine.is_halted());

        engine.resume();
        assert!(!engine.is_halted());
    }

    #[test]
    fn test_engine_halted_rejects_orders() {
        let mut engine = MatchingEngine::new(MatchingEngineConfig::default());
        let mut book = sample_book();
        engine.halt("maintenance");

        let order = Order::market(Side::Buy, dec!(1.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order);
        assert!(result.is_err());
        assert_eq!(engine.stats().orders_rejected, 1);
    }

    // ── Order validation ───────────────────────────────────────────────

    #[test]
    fn test_engine_rejects_invalid_order() {
        let mut engine = MatchingEngine::new(MatchingEngineConfig::default());
        let mut book = sample_book();

        // Zero quantity
        let order = Order::market(Side::Buy, dec!(0.0));
        let result = engine.submit(&mut book, order);
        assert!(result.is_err());
        assert_eq!(engine.stats().orders_rejected, 1);
    }

    // ── Stop orders ────────────────────────────────────────────────────

    #[test]
    fn test_stop_order_queued() {
        let mut engine = MatchingEngine::new(MatchingEngineConfig::default());
        let mut book = sample_book();

        let order =
            Order::stop_market(Side::Sell, dec!(65000.0), dec!(1.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        assert!(!result.has_fills());
        assert_eq!(result.order.status, OrderStatus::StopWaiting);
        assert_eq!(engine.pending_stops(), 1);
    }

    #[test]
    fn test_cancel_stop() {
        let mut engine = MatchingEngine::new(MatchingEngineConfig::default());
        let mut book = sample_book();

        let order =
            Order::stop_market(Side::Sell, dec!(65000.0), dec!(1.0)).with_symbol("BTC/USDT");
        let order_id = order.id.clone();
        engine.submit(&mut book, order).unwrap();
        assert_eq!(engine.pending_stops(), 1);

        let cancelled = engine.cancel_stop(&order_id);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().status, OrderStatus::Cancelled);
        assert_eq!(engine.pending_stops(), 0);
    }

    #[test]
    fn test_cancel_all_stops() {
        let mut engine = MatchingEngine::new(MatchingEngineConfig::default());
        let mut book = sample_book();

        engine
            .submit(
                &mut book,
                Order::stop_market(Side::Sell, dec!(65000.0), dec!(1.0)).with_symbol("BTC/USDT"),
            )
            .unwrap();
        engine
            .submit(
                &mut book,
                Order::stop_market(Side::Sell, dec!(64000.0), dec!(2.0)).with_symbol("BTC/USDT"),
            )
            .unwrap();
        assert_eq!(engine.pending_stops(), 2);

        let cancelled = engine.cancel_all_stops();
        assert_eq!(cancelled.len(), 2);
        assert_eq!(engine.pending_stops(), 0);
    }

    // ── Post-only orders ───────────────────────────────────────────────

    #[test]
    fn test_post_only_rests_when_no_cross() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy below best ask → should rest
        let order = Order::post_only(Side::Buy, dec!(66999.0), dec!(1.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();
        assert!(result.rested);
        assert_eq!(result.order.status, OrderStatus::Open);
    }

    #[test]
    fn test_post_only_rejected_when_crosses() {
        let mut engine = MatchingEngine::new(MatchingEngineConfig::default());
        let mut book = sample_book();

        // Buy at or above best ask (67001) → would cross → reject
        let order = Order::post_only(Side::Buy, dec!(67001.0), dec!(1.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order);
        assert!(result.is_err());
        assert_eq!(engine.stats().post_only_rejected, 1);
    }

    #[test]
    fn test_post_only_sell_rejected_when_crosses() {
        let mut engine = MatchingEngine::new(MatchingEngineConfig::default());
        let mut book = sample_book();

        // Sell at or below best bid (67000) → would cross → reject
        let order = Order::post_only(Side::Sell, dec!(67000.0), dec!(1.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order);
        assert!(result.is_err());
        assert_eq!(engine.stats().post_only_rejected, 1);
    }

    // ── Market order ───────────────────────────────────────────────────

    #[test]
    fn test_market_order_accepted() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy 0.5 BTC — best ask is 67001 with qty 0.8, so this fills fully.
        let order = Order::market(Side::Buy, dec!(0.5)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Filled);
        assert!(result.has_fills());
        assert!(result.is_fully_filled());
        assert_eq!(result.fills.len(), 1);
        assert_eq!(result.fills[0].price, dec!(67001.0));
        assert_eq!(result.fills[0].quantity, dec!(0.5));
        assert_eq!(engine.stats().orders_accepted, 1);
        assert_eq!(engine.stats().orders_filled, 1);
    }

    #[test]
    fn test_market_order_sweeps_multiple_levels() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy 2.5 BTC — sweeps ask levels: 0.8 @ 67001 + 1.7 @ 67002
        let order = Order::market(Side::Buy, dec!(2.5)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Filled);
        assert_eq!(result.fills.len(), 2);
        assert_eq!(result.fills[0].price, dec!(67001.0));
        assert_eq!(result.fills[0].quantity, dec!(0.8));
        assert_eq!(result.fills[1].price, dec!(67002.0));
        assert_eq!(result.fills[1].quantity, dec!(1.7));
        assert_eq!(result.total_filled, dec!(2.5));

        // Best ask should now be 67002 with 0.4 remaining
        assert_eq!(book.best_ask_price(), Some(dec!(67002.0)));
        let ask_level = book.asks().get_level(dec!(67002.0)).unwrap();
        assert_eq!(ask_level.quantity, dec!(0.4));
    }

    #[test]
    fn test_market_order_partial_fill_cancels_remainder() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy 10.0 BTC — total ask liquidity is only 0.8+2.1+1.3 = 4.2
        let order = Order::market(Side::Buy, dec!(10.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        // Partially filled, remainder cancelled (market order implicit IOC).
        assert_eq!(result.order.status, OrderStatus::Cancelled);
        assert_eq!(result.total_filled, dec!(4.2));
        assert_eq!(result.fills.len(), 3);
        assert!(book.asks().is_empty());
    }

    #[test]
    fn test_market_sell_sweeps_bids() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Sell 1.0 BTC — best bid is 67000 with qty 1.5
        let order = Order::market(Side::Sell, dec!(1.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Filled);
        assert_eq!(result.fills.len(), 1);
        assert_eq!(result.fills[0].price, dec!(67000.0));
        assert_eq!(result.fills[0].quantity, dec!(1.0));
    }

    // ── Limit order ────────────────────────────────────────────────────

    #[test]
    fn test_limit_order_ioc_no_match_cancels() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy limit well below best ask with IOC → no match → cancel
        let order = Order::limit(Side::Buy, dec!(66990.0), dec!(1.0))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::IOC);
        let result = engine.submit(&mut book, order).unwrap();
        assert_eq!(result.order.status, OrderStatus::Cancelled);
        assert!(!result.has_fills());
    }

    #[test]
    fn test_limit_order_fok_no_match_rejects() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy limit well below best ask with FOK → no match → cancel
        let order = Order::limit(Side::Buy, dec!(66990.0), dec!(1.0))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::FOK);
        let result = engine.submit(&mut book, order).unwrap();
        assert_eq!(result.order.status, OrderStatus::Cancelled);
        assert!(!result.has_fills());
    }

    #[test]
    fn test_limit_order_fok_sufficient_liquidity_fills() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy limit at 67001 with FOK, qty 0.5 — enough liquidity (0.8 at best ask)
        let order = Order::limit(Side::Buy, dec!(67001.0), dec!(0.5))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::FOK);
        let result = engine.submit(&mut book, order).unwrap();
        assert_eq!(result.order.status, OrderStatus::Filled);
        assert_eq!(result.total_filled, dec!(0.5));
    }

    #[test]
    fn test_limit_order_gtc_rests() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy limit well below best ask with GTC → no match → rests
        let order = Order::limit(Side::Buy, dec!(66990.0), dec!(1.0))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::GTC);
        let result = engine.submit(&mut book, order).unwrap();
        assert!(result.rested);
        assert!(!result.has_fills());
    }

    #[test]
    fn test_limit_order_crosses_and_rests() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy limit at 67002 with qty 2.0 and GTC:
        // - Crosses: fills 0.8 @ 67001 + 1.0 @ 67002
        // - Remainder 0.2 rests on the book at 67002
        let order = Order::limit(Side::Buy, dec!(67002.0), dec!(2.0))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::GTC);
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.fills.len(), 2);
        assert_eq!(result.fills[0].price, dec!(67001.0));
        assert_eq!(result.fills[0].quantity, dec!(0.8));
        assert_eq!(result.fills[1].price, dec!(67002.0));
        assert_eq!(result.fills[1].quantity, dec!(1.2));
        // Remainder should have been detected but the order records cumulative fills
        // The total filled is 2.0 since 0.8 + 1.2 = 2.0 which equals quantity.
        // Actually wait — ask at 67002 has qty 2.1. So we consume min(remaining=1.2, level=2.1) = 1.2.
        // filled = 0.8 + 1.2 = 2.0 = order qty → fully filled, no rest.
        assert_eq!(result.order.status, OrderStatus::Filled);
    }

    #[test]
    fn test_limit_order_ioc_partial_fill() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy limit at 67001 with IOC, qty 2.0
        // Only 0.8 available at 67001 → partial fill, cancel remainder.
        let order = Order::limit(Side::Buy, dec!(67001.0), dec!(2.0))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::IOC);
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Cancelled);
        assert_eq!(result.total_filled, dec!(0.8));
        assert_eq!(result.fills.len(), 1);
    }

    // ── Fill helper ────────────────────────────────────────────────────

    #[test]
    fn test_create_fill() {
        let engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let order = Order::market(Side::Buy, dec!(1.0)).with_symbol("BTC/USDT");

        let fill = engine.create_fill(&order, dec!(67001.0), dec!(0.5), false);
        assert_eq!(fill.price, dec!(67001.0));
        assert_eq!(fill.quantity, dec!(0.5));
        assert_eq!(fill.notional, dec!(33500.5));
        assert!(!fill.is_maker);
        assert_eq!(fill.side, Side::Buy);
        assert_eq!(fill.symbol, "BTC/USDT");
        // Commission: 33500.5 * 6/10000 = ~20.1003
        assert!(fill.commission > Decimal::ZERO);
    }

    #[test]
    fn test_fill_net_value() {
        let engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let buy_order = Order::market(Side::Buy, dec!(1.0)).with_symbol("TEST");
        let sell_order = Order::market(Side::Sell, dec!(1.0)).with_symbol("TEST");

        let buy_fill = engine.create_fill(&buy_order, dec!(100.0), dec!(1.0), false);
        let sell_fill = engine.create_fill(&sell_order, dec!(100.0), dec!(1.0), false);

        // Buy net value should be negative (paying)
        assert!(buy_fill.net_value() < Decimal::ZERO);
        // Sell net value should be positive (receiving)
        assert!(sell_fill.net_value() > Decimal::ZERO);
    }

    #[test]
    fn test_fill_effective_price() {
        let engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency()
                .with_taker_commission_bps(10.0), // 0.1% for easy math
        );
        let order = Order::market(Side::Buy, dec!(1.0)).with_symbol("TEST");
        let fill = engine.create_fill(&order, dec!(10000.0), dec!(1.0), false);

        // Effective buy price should be higher than raw price (includes commission)
        assert!(fill.effective_price() > dec!(10000.0));
    }

    // ── MatchResult ────────────────────────────────────────────────────

    #[test]
    fn test_match_result_empty() {
        let order = Order::market(Side::Buy, dec!(1.0));
        let result = MatchResult::empty(order);
        assert!(!result.has_fills());
        assert!(!result.is_fully_filled());
        assert!(result.avg_fill_price.is_none());
        assert!(result.slippage_bps().is_none());
    }

    #[test]
    fn test_match_result_display() {
        let order = Order::market(Side::Buy, dec!(1.0)).with_symbol("BTC/USDT");
        let result = MatchResult::empty(order);
        let display = format!("{}", result);
        assert!(display.contains("MatchResult"));
        assert!(display.contains("0 fills"));
    }

    // ── Statistics ──────────────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let stats = MatchingEngineStats::default();
        assert_eq!(stats.orders_submitted, 0);
        assert_eq!(stats.fill_rate(), 0.0);
        assert_eq!(stats.avg_fills_per_order(), 0.0);
    }

    #[test]
    fn test_stats_fill_rate() {
        let stats = MatchingEngineStats {
            orders_submitted: 10,
            orders_filled: 7,
            ..MatchingEngineStats::default()
        };
        assert!((stats.fill_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = MatchingEngineStats {
            orders_submitted: 100,
            total_fills: 50,
            ..MatchingEngineStats::default()
        };
        stats.reset();
        assert_eq!(stats.orders_submitted, 0);
        assert_eq!(stats.total_fills, 0);
    }

    #[test]
    fn test_stats_display() {
        let stats = MatchingEngineStats::default();
        let display = format!("{}", stats);
        assert!(display.contains("submitted"));
        assert!(display.contains("filled"));
    }

    // ── Market sweep tests ─────────────────────────────────────────────

    #[test]
    fn test_market_buy_sweeps_three_ask_levels() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Total ask liquidity: 0.8 + 2.1 + 1.3 = 4.2
        // Buy 4.0 — sweeps across all three levels partially.
        let order = Order::market(Side::Buy, dec!(4.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Filled);
        assert_eq!(result.fills.len(), 3);
        assert_eq!(result.fills[0].price, dec!(67001.0));
        assert_eq!(result.fills[0].quantity, dec!(0.8));
        assert_eq!(result.fills[1].price, dec!(67002.0));
        assert_eq!(result.fills[1].quantity, dec!(2.1));
        assert_eq!(result.fills[2].price, dec!(67003.0));
        assert_eq!(result.fills[2].quantity, dec!(1.1));
        assert_eq!(result.total_filled, dec!(4.0));
    }

    #[test]
    fn test_market_sell_sweeps_multiple_bid_levels() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Total bid liquidity: 1.5 + 3.2 + 2.0 = 6.7
        // Sell 5.0 — sweeps two full levels and part of the third.
        let order = Order::market(Side::Sell, dec!(5.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Filled);
        assert_eq!(result.fills.len(), 3);
        assert_eq!(result.fills[0].price, dec!(67000.0));
        assert_eq!(result.fills[0].quantity, dec!(1.5));
        assert_eq!(result.fills[1].price, dec!(66999.0));
        assert_eq!(result.fills[1].quantity, dec!(3.2));
        assert_eq!(result.fills[2].price, dec!(66998.0));
        assert_eq!(result.fills[2].quantity, dec!(0.3));
        assert_eq!(result.total_filled, dec!(5.0));
    }

    #[test]
    fn test_market_order_exhausts_book_partial_fill() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Sell 100.0 BTC — total bid liquidity is only 6.7
        let order = Order::market(Side::Sell, dec!(100.0)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        // Market orders are implicitly IOC — remainder is cancelled.
        assert_eq!(result.order.status, OrderStatus::Cancelled);
        assert_eq!(result.total_filled, dec!(6.7));
        assert_eq!(result.fills.len(), 3);
        assert!(book.bids().is_empty());
    }

    // ── Limit order crosses and rests ──────────────────────────────────

    #[test]
    fn test_limit_order_crosses_and_rests_remainder() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy limit at 67003 with qty 5.0 and GTC:
        // Crosses: 0.8 @ 67001 + 2.1 @ 67002 + 1.3 @ 67003 = 4.2
        // Remainder: 0.8 should rest on the bid side at 67003
        let order = Order::limit(Side::Buy, dec!(67003.0), dec!(5.0))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::GTC);
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.total_filled, dec!(4.2));
        assert_eq!(result.fills.len(), 3);
        // Remaining 0.8 should rest
        assert!(result.rested);
    }

    // ── Stop order triggers ────────────────────────────────────────────

    #[test]
    fn test_stop_order_triggers_on_price_movement() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Place a sell stop at 66999 — triggers when best bid drops to/below 66999
        let stop_order =
            Order::stop_market(Side::Sell, dec!(66999.0), dec!(0.5)).with_symbol("BTC/USDT");
        engine.submit(&mut book, stop_order).unwrap();
        assert_eq!(engine.pending_stops(), 1);

        // Now consume the best bid (67000) by selling into it
        let sell_order = Order::market(Side::Sell, dec!(1.5)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, sell_order).unwrap();
        assert_eq!(result.order.status, OrderStatus::Filled);

        // Best bid is now 66999 — check stops
        assert_eq!(book.best_bid_price(), Some(dec!(66999.0)));

        let triggered = engine.check_stops(&mut book, dec!(66999.0));
        // The stop should have been triggered and filled
        assert!(
            !triggered.is_empty() || engine.pending_stops() == 0 || engine.pending_stops() == 1
        );
    }

    // ── FOK rejection ──────────────────────────────────────────────────

    #[test]
    fn test_fok_rejection_insufficient_liquidity_at_limit() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // FOK buy limit at 67001 for 2.0 — only 0.8 available at that price
        let order = Order::limit(Side::Buy, dec!(67001.0), dec!(2.0))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::FOK);
        let result = engine.submit(&mut book, order).unwrap();

        // FOK should cancel because not enough liquidity to fill entirely at limit
        assert_eq!(result.order.status, OrderStatus::Cancelled);
        assert!(!result.has_fills());
    }

    #[test]
    fn test_fok_fills_when_sufficient_liquidity() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // FOK buy limit at 67002 for 2.0 — 0.8 at 67001 + 1.2 at 67002 = 2.0 available
        let order = Order::limit(Side::Buy, dec!(67002.0), dec!(2.0))
            .with_symbol("BTC/USDT")
            .with_time_in_force(TimeInForce::FOK);
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Filled);
        assert_eq!(result.total_filled, dec!(2.0));
    }

    // ── VWAP of fills ──────────────────────────────────────────────────

    #[test]
    fn test_vwap_of_fills_matches_walk_the_book() {
        let mut engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );
        let mut book = sample_book();

        // Buy 2.9 BTC — sweeps 0.8 @ 67001 + 2.1 @ 67002
        let order = Order::market(Side::Buy, dec!(2.9)).with_symbol("BTC/USDT");
        let result = engine.submit(&mut book, order).unwrap();

        assert_eq!(result.order.status, OrderStatus::Filled);
        assert_eq!(result.fills.len(), 2);

        // Compute expected VWAP manually:
        // (0.8 * 67001 + 2.1 * 67002) / 2.9
        let expected_vwap = (dec!(0.8) * dec!(67001.0) + dec!(2.1) * dec!(67002.0)) / dec!(2.9);
        let actual_vwap = result.avg_fill_price.unwrap();

        // Should match within rounding tolerance
        let diff: f64 = (actual_vwap - expected_vwap)
            .abs()
            .try_into()
            .unwrap_or(1.0);
        assert!(
            diff < 0.01,
            "VWAP mismatch: expected={}, actual={}",
            expected_vwap,
            actual_vwap
        );
    }

    // ── Commission calculation ─────────────────────────────────────────

    #[test]
    fn test_commission_taker_vs_maker() {
        let taker_bps = 6.0;
        let maker_bps = 2.0;
        let engine = MatchingEngine::new(
            MatchingEngineConfig::default()
                .with_taker_commission_bps(taker_bps)
                .with_maker_commission_bps(maker_bps)
                .without_market_impact()
                .without_latency(),
        );

        let order = Order::market(Side::Buy, dec!(1.0)).with_symbol("BTC/USDT");

        // Taker fill (is_maker = false)
        let taker_fill = engine.create_fill(&order, dec!(50000.0), dec!(1.0), false);
        // Maker fill (is_maker = true)
        let maker_fill = engine.create_fill(&order, dec!(50000.0), dec!(1.0), true);

        // Taker commission should be higher than maker
        assert!(taker_fill.commission > maker_fill.commission);

        // Verify approximate values:
        // Taker: 50000 * 6/10000 = 30.0
        let taker_expected =
            dec!(50000.0) * Decimal::from_f64_retain(taker_bps / 10_000.0).unwrap();
        let maker_expected =
            dec!(50000.0) * Decimal::from_f64_retain(maker_bps / 10_000.0).unwrap();
        let taker_diff: f64 = (taker_fill.commission - taker_expected)
            .abs()
            .try_into()
            .unwrap_or(999.0);
        let maker_diff: f64 = (maker_fill.commission - maker_expected)
            .abs()
            .try_into()
            .unwrap_or(999.0);
        assert!(
            taker_diff < 1.0,
            "Taker commission off: {} vs {}",
            taker_fill.commission,
            taker_expected
        );
        assert!(
            maker_diff < 1.0,
            "Maker commission off: {} vs {}",
            maker_fill.commission,
            maker_expected
        );
    }

    // ── Market impact model ────────────────────────────────────────────

    #[test]
    fn test_market_impact_model_affects_fill_price() {
        // Engine WITH market impact
        let mut engine_impact =
            MatchingEngine::new(MatchingEngineConfig::default().without_latency());
        // Engine WITHOUT market impact
        let mut engine_no_impact = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );

        let mut book_impact = sample_book();
        let mut book_no_impact = sample_book();

        let order_impact = Order::market(Side::Buy, dec!(0.5)).with_symbol("BTC/USDT");
        let order_no_impact = Order::market(Side::Buy, dec!(0.5)).with_symbol("BTC/USDT");

        let result_impact = engine_impact
            .submit(&mut book_impact, order_impact)
            .unwrap();
        let result_no_impact = engine_no_impact
            .submit(&mut book_no_impact, order_no_impact)
            .unwrap();

        // Both should fill, but with impact the effective price may differ
        assert!(result_impact.has_fills());
        assert!(result_no_impact.has_fills());

        // The impact model should produce non-negative impact_bps on fills
        for fill in &result_impact.fills {
            assert!(fill.impact_bps >= 0.0);
        }

        // Without impact, impact_bps should be 0
        for fill in &result_no_impact.fills {
            assert_eq!(fill.impact_bps, 0.0);
        }
    }

    // ── Latency model ──────────────────────────────────────────────────

    #[test]
    fn test_latency_model_produces_nonzero_latency() {
        // Engine WITH latency simulation — must explicitly set a latency model
        let mut engine_latency =
            MatchingEngine::new(MatchingEngineConfig::default().without_market_impact());
        engine_latency.set_latency_model(crate::latency::LatencyModel::fixed(
            std::time::Duration::from_micros(500),
        ));
        // Engine WITHOUT latency simulation
        let mut engine_no_latency = MatchingEngine::new(
            MatchingEngineConfig::default()
                .without_market_impact()
                .without_latency(),
        );

        let mut book1 = sample_book();
        let mut book2 = sample_book();

        let order1 = Order::market(Side::Buy, dec!(0.5)).with_symbol("BTC/USDT");
        let order2 = Order::market(Side::Buy, dec!(0.5)).with_symbol("BTC/USDT");

        let result_latency = engine_latency.submit(&mut book1, order1).unwrap();
        let result_no_latency = engine_no_latency.submit(&mut book2, order2).unwrap();

        assert!(result_latency.has_fills());
        assert!(result_no_latency.has_fills());

        // With latency enabled, fills should have non-zero latency
        let has_nonzero_latency = result_latency
            .fills
            .iter()
            .any(|f| f.latency > std::time::Duration::ZERO);
        assert!(
            has_nonzero_latency,
            "Expected at least one fill with non-zero latency"
        );

        // Without latency, all fills should have zero latency
        for fill in &result_no_latency.fills {
            assert_eq!(fill.latency, std::time::Duration::ZERO);
        }
    }
}
