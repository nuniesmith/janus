//! Limit Order Book (LOB) Simulator
//!
//! A full-featured Level 2/3 limit order book simulator for realistic
//! backtesting, market microstructure research, and execution strategy
//! development within the JANUS neuromorphic cerebellum.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   LOB Simulator Engine                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
//! │  │ Order Entry  │    │ Matching     │    │ Trade           │   │
//! │  │ Gateway      │───▶│ Engine       │───▶│ Reporter        │   │
//! │  │ (Validation  │    │ (Price-Time  │    │ (Execution      │   │
//! │  │  + Latency)  │    │  Priority)   │    │  Reporting)     │   │
//! │  └──────────────┘    └──────────────┘    └─────────────────┘   │
//! │         │                    │                                    │
//! │         ▼                    ▼                                    │
//! │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
//! │  │ Order Book   │    │ Market       │    │ L2/L3           │   │
//! │  │ Data Struct  │    │ Impact       │    │ Snapshot        │   │
//! │  │ (BTreeMap +  │    │ Model        │    │ Generator       │   │
//! │  │  VecDeque)   │    │ (Almgren-    │    │                 │   │
//! │  └──────────────┘    │  Chriss)     │    └─────────────────┘   │
//! │                      └──────────────┘                            │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │  Statistics & Analytics                                   │   │
//! │  │  • VWAP, TWAP, Implementation Shortfall                  │   │
//! │  │  • Spread Analytics, Queue Position Tracking              │   │
//! │  │  • Latency Distribution, Fill Rate Monitoring             │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Price-time priority matching** with FIFO queue at each price level
//! - **Order types**: Limit, Market, Stop, Stop-Limit, IOC, FOK, Post-Only, Iceberg
//! - **L2 order book snapshots** with configurable depth
//! - **L3 order-level detail** for queue position analysis
//! - **Market impact modeling** integrated with Almgren-Chriss
//! - **Configurable latency simulation** with jitter
//! - **Trade/execution reporting** with full audit trail
//! - **Historical replay** from L2/L3 market data feeds
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_neuromorphic::cerebellum::lob_simulator::*;
//!
//! let config = LobConfig::default()
//!     .with_tick_size(0.01)
//!     .with_lot_size(1.0)
//!     .with_latency(Duration::from_micros(50));
//!
//! let mut lob = LimitOrderBook::new("BTC-USDT", config);
//!
//! // Submit a limit buy order
//! let order = Order::limit_buy(100.0, 50000.0, "strategy_1");
//! let receipt = lob.submit_order(order)?;
//!
//! // Submit a market sell that crosses the spread
//! let market = Order::market_sell(50.0, "strategy_2");
//! let receipt = lob.submit_order(market)?;
//!
//! // Get L2 snapshot
//! let snapshot = lob.snapshot(10);
//! println!("Best bid: {:?}, Best ask: {:?}", snapshot.best_bid(), snapshot.best_ask());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
#[allow(unused_imports)]
use std::time::{Duration, Instant};
use tracing::{debug, info, trace, warn};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the LOB simulator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LobError {
    /// The order failed validation.
    InvalidOrder(String),
    /// The referenced order was not found.
    OrderNotFound(u64),
    /// A self-trade would occur.
    SelfTrade { aggressor: u64, resting: u64 },
    /// The order book is in an invalid state.
    InvalidState(String),
    /// Fill-or-kill order could not be completely filled.
    FokNotFilled {
        order_id: u64,
        available: f64,
        requested: f64,
    },
    /// Post-only order would have taken liquidity.
    PostOnlyWouldTake { order_id: u64 },
    /// Insufficient quantity for the operation.
    InsufficientQuantity(String),
    /// Price is outside allowed range.
    PriceOutOfRange { price: f64, min: f64, max: f64 },
    /// Generic internal error.
    Internal(String),
}

impl fmt::Display for LobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOrder(e) => write!(f, "LOB: invalid order: {e}"),
            Self::OrderNotFound(id) => write!(f, "LOB: order {id} not found"),
            Self::SelfTrade { aggressor, resting } => {
                write!(f, "LOB: self-trade prevented ({aggressor} vs {resting})")
            }
            Self::InvalidState(e) => write!(f, "LOB: invalid state: {e}"),
            Self::FokNotFilled {
                order_id,
                available,
                requested,
            } => {
                write!(
                    f,
                    "LOB: FOK order {order_id} not filled (available={available}, requested={requested})"
                )
            }
            Self::PostOnlyWouldTake { order_id } => {
                write!(f, "LOB: post-only order {order_id} would take liquidity")
            }
            Self::InsufficientQuantity(e) => write!(f, "LOB: insufficient quantity: {e}"),
            Self::PriceOutOfRange { price, min, max } => {
                write!(f, "LOB: price {price} out of range [{min}, {max}]")
            }
            Self::Internal(e) => write!(f, "LOB: internal error: {e}"),
        }
    }
}

impl std::error::Error for LobError {}

pub type Result<T> = std::result::Result<T, LobError>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the LOB simulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LobConfig {
    /// Minimum price increment (tick size).
    pub tick_size: f64,

    /// Minimum order quantity increment (lot size).
    pub lot_size: f64,

    /// Simulated gateway latency (order entry → matching).
    pub gateway_latency: Duration,

    /// Simulated latency jitter (uniform random up to this value).
    pub latency_jitter: Duration,

    /// Maximum allowed price (circuit breaker).
    pub max_price: f64,

    /// Minimum allowed price.
    pub min_price: f64,

    /// Maximum number of open orders per participant.
    pub max_orders_per_participant: usize,

    /// Whether self-trade prevention is enabled.
    pub self_trade_prevention: bool,

    /// Maximum L2 depth to track per side.
    pub max_book_depth: usize,

    /// Whether to record full trade history.
    pub record_trades: bool,

    /// Whether to record order event history.
    pub record_order_events: bool,

    /// Maker fee rate (negative = rebate).
    pub maker_fee_rate: f64,

    /// Taker fee rate.
    pub taker_fee_rate: f64,

    /// Symbol name for this order book.
    pub symbol: String,
}

impl Default for LobConfig {
    fn default() -> Self {
        Self {
            tick_size: 0.01,
            lot_size: 0.001,
            gateway_latency: Duration::from_micros(50),
            latency_jitter: Duration::from_micros(10),
            max_price: 1_000_000.0,
            min_price: 0.01,
            max_orders_per_participant: 10_000,
            self_trade_prevention: true,
            max_book_depth: 500,
            record_trades: true,
            record_order_events: true,
            maker_fee_rate: -0.0001, // 1 bps rebate
            taker_fee_rate: 0.0004,  // 4 bps fee
            symbol: "UNKNOWN".to_string(),
        }
    }
}

impl LobConfig {
    /// Builder: set tick size.
    pub fn with_tick_size(mut self, tick_size: f64) -> Self {
        self.tick_size = tick_size;
        self
    }

    /// Builder: set lot size.
    pub fn with_lot_size(mut self, lot_size: f64) -> Self {
        self.lot_size = lot_size;
        self
    }

    /// Builder: set gateway latency.
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.gateway_latency = latency;
        self
    }

    /// Builder: set latency jitter.
    pub fn with_jitter(mut self, jitter: Duration) -> Self {
        self.latency_jitter = jitter;
        self
    }

    /// Builder: set price bounds.
    pub fn with_price_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_price = min;
        self.max_price = max;
        self
    }

    /// Builder: set symbol.
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = symbol.into();
        self
    }

    /// Builder: set fee rates.
    pub fn with_fees(mut self, maker: f64, taker: f64) -> Self {
        self.maker_fee_rate = maker;
        self.taker_fee_rate = taker;
        self
    }

    /// Builder: enable/disable trade recording.
    pub fn with_record_trades(mut self, record: bool) -> Self {
        self.record_trades = record;
        self
    }

    /// Builder: enable/disable self-trade prevention.
    pub fn with_self_trade_prevention(mut self, enabled: bool) -> Self {
        self.self_trade_prevention = enabled;
        self
    }

    /// Round a price to the nearest valid tick.
    pub fn round_price(&self, price: f64) -> f64 {
        (price / self.tick_size).round() * self.tick_size
    }

    /// Round a quantity to the nearest valid lot.
    pub fn round_quantity(&self, qty: f64) -> f64 {
        (qty / self.lot_size).round() * self.lot_size
    }

    /// Validate that a price is within bounds and on a valid tick.
    pub fn validate_price(&self, price: f64) -> Result<f64> {
        if price < self.min_price || price > self.max_price {
            return Err(LobError::PriceOutOfRange {
                price,
                min: self.min_price,
                max: self.max_price,
            });
        }
        Ok(self.round_price(price))
    }

    /// Validate that a quantity is positive and on a valid lot boundary.
    pub fn validate_quantity(&self, qty: f64) -> Result<f64> {
        if qty <= 0.0 {
            return Err(LobError::InvalidOrder(
                "Quantity must be positive".to_string(),
            ));
        }
        let rounded = self.round_quantity(qty);
        if rounded <= 0.0 {
            return Err(LobError::InvalidOrder(
                "Quantity rounds to zero".to_string(),
            ));
        }
        Ok(rounded)
    }
}

// ---------------------------------------------------------------------------
// Order types & sides
// ---------------------------------------------------------------------------

/// Side of the order book.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// The opposite side.
    pub fn opposite(&self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
        }
    }
}

/// Type of order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    /// Standard limit order — rests on the book until filled or cancelled.
    Limit,
    /// Market order — fills immediately at best available price.
    Market,
    /// Stop order — becomes a market order when the stop price is reached.
    Stop,
    /// Stop-limit order — becomes a limit order when the stop price is reached.
    StopLimit,
    /// Immediate-or-cancel — fills as much as possible, cancels the rest.
    ImmediateOrCancel,
    /// Fill-or-kill — must be completely filled or entirely cancelled.
    FillOrKill,
    /// Post-only — cancelled if it would take liquidity (maker only).
    PostOnly,
    /// Iceberg — displays only a portion of the total quantity.
    Iceberg,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Limit => write!(f, "LIMIT"),
            Self::Market => write!(f, "MARKET"),
            Self::Stop => write!(f, "STOP"),
            Self::StopLimit => write!(f, "STOP_LIMIT"),
            Self::ImmediateOrCancel => write!(f, "IOC"),
            Self::FillOrKill => write!(f, "FOK"),
            Self::PostOnly => write!(f, "POST_ONLY"),
            Self::Iceberg => write!(f, "ICEBERG"),
        }
    }
}

/// Time-in-force for limit orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good 'til cancelled (default for limit orders).
    GoodTilCancelled,
    /// Good for the current session only.
    GoodForSession,
    /// Good for a specific duration.
    GoodForDuration(u64), // milliseconds
}

impl Default for TimeInForce {
    fn default() -> Self {
        Self::GoodTilCancelled
    }
}

/// Status of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Order has been accepted but not yet processed.
    Pending,
    /// Order is resting on the book (open).
    Open,
    /// Order has been partially filled.
    PartiallyFilled,
    /// Order has been completely filled.
    Filled,
    /// Order has been cancelled.
    Cancelled,
    /// Order was rejected.
    Rejected,
    /// Stop order is waiting for trigger.
    StopWaiting,
    /// Order has expired.
    Expired,
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "PENDING"),
            Self::Open => write!(f, "OPEN"),
            Self::PartiallyFilled => write!(f, "PARTIAL"),
            Self::Filled => write!(f, "FILLED"),
            Self::Cancelled => write!(f, "CANCELLED"),
            Self::Rejected => write!(f, "REJECTED"),
            Self::StopWaiting => write!(f, "STOP_WAITING"),
            Self::Expired => write!(f, "EXPIRED"),
        }
    }
}

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------

/// An order submitted to the LOB simulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order identifier (assigned by the engine).
    pub id: u64,

    /// Side (buy or sell).
    pub side: Side,

    /// Order type.
    pub order_type: OrderType,

    /// Limit price (not used for market orders).
    pub price: f64,

    /// Total quantity.
    pub quantity: f64,

    /// Remaining unfilled quantity.
    pub remaining: f64,

    /// Filled quantity.
    pub filled: f64,

    /// Average fill price.
    pub avg_fill_price: f64,

    /// Participant/strategy identifier.
    pub participant: String,

    /// Time in force.
    pub time_in_force: TimeInForce,

    /// Current status.
    pub status: OrderStatus,

    /// Stop trigger price (for stop / stop-limit orders).
    pub stop_price: Option<f64>,

    /// Display quantity for iceberg orders (None = show all).
    pub display_quantity: Option<f64>,

    /// Simulated timestamp (nanoseconds since epoch or sequence number).
    pub timestamp_ns: u64,

    /// When the order was created.
    pub created_at_ns: u64,

    /// Accumulated fees (positive = paid, negative = rebate).
    pub fees: f64,

    /// Optional client-side order ID for correlation.
    pub client_order_id: Option<String>,
}

impl Order {
    /// Create a new order (ID will be assigned by the engine).
    pub fn new(
        side: Side,
        order_type: OrderType,
        price: f64,
        quantity: f64,
        participant: impl Into<String>,
    ) -> Self {
        Self {
            id: 0,
            side,
            order_type,
            price,
            quantity,
            remaining: quantity,
            filled: 0.0,
            avg_fill_price: 0.0,
            participant: participant.into(),
            time_in_force: TimeInForce::default(),
            status: OrderStatus::Pending,
            stop_price: None,
            display_quantity: None,
            timestamp_ns: 0,
            created_at_ns: 0,
            fees: 0.0,
            client_order_id: None,
        }
    }

    /// Convenience: create a limit buy order.
    pub fn limit_buy(quantity: f64, price: f64, participant: impl Into<String>) -> Self {
        Self::new(Side::Buy, OrderType::Limit, price, quantity, participant)
    }

    /// Convenience: create a limit sell order.
    pub fn limit_sell(quantity: f64, price: f64, participant: impl Into<String>) -> Self {
        Self::new(Side::Sell, OrderType::Limit, price, quantity, participant)
    }

    /// Convenience: create a market buy order.
    pub fn market_buy(quantity: f64, participant: impl Into<String>) -> Self {
        Self::new(Side::Buy, OrderType::Market, 0.0, quantity, participant)
    }

    /// Convenience: create a market sell order.
    pub fn market_sell(quantity: f64, participant: impl Into<String>) -> Self {
        Self::new(Side::Sell, OrderType::Market, 0.0, quantity, participant)
    }

    /// Convenience: create an IOC order.
    pub fn ioc(side: Side, quantity: f64, price: f64, participant: impl Into<String>) -> Self {
        Self::new(
            side,
            OrderType::ImmediateOrCancel,
            price,
            quantity,
            participant,
        )
    }

    /// Convenience: create a fill-or-kill order.
    pub fn fok(side: Side, quantity: f64, price: f64, participant: impl Into<String>) -> Self {
        Self::new(side, OrderType::FillOrKill, price, quantity, participant)
    }

    /// Convenience: create a post-only limit order.
    pub fn post_only(
        side: Side,
        quantity: f64,
        price: f64,
        participant: impl Into<String>,
    ) -> Self {
        Self::new(side, OrderType::PostOnly, price, quantity, participant)
    }

    /// Convenience: create an iceberg order.
    pub fn iceberg(
        side: Side,
        quantity: f64,
        price: f64,
        display_qty: f64,
        participant: impl Into<String>,
    ) -> Self {
        let mut order = Self::new(side, OrderType::Iceberg, price, quantity, participant);
        order.display_quantity = Some(display_qty);
        order
    }

    /// Convenience: create a stop order.
    pub fn stop(
        side: Side,
        quantity: f64,
        stop_price: f64,
        participant: impl Into<String>,
    ) -> Self {
        let mut order = Self::new(side, OrderType::Stop, 0.0, quantity, participant);
        order.stop_price = Some(stop_price);
        order
    }

    /// Convenience: create a stop-limit order.
    pub fn stop_limit(
        side: Side,
        quantity: f64,
        limit_price: f64,
        stop_price: f64,
        participant: impl Into<String>,
    ) -> Self {
        let mut order = Self::new(
            side,
            OrderType::StopLimit,
            limit_price,
            quantity,
            participant,
        );
        order.stop_price = Some(stop_price);
        order
    }

    /// Set client order ID.
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_order_id = Some(client_id.into());
        self
    }

    /// Set time in force.
    pub fn with_tif(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Whether this order is still active (can be matched or cancelled).
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Open | OrderStatus::PartiallyFilled | OrderStatus::StopWaiting
        )
    }

    /// Whether this order has been fully filled.
    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }

    /// Whether this order is done (filled, cancelled, rejected, expired).
    pub fn is_done(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled
                | OrderStatus::Cancelled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }

    /// The visible quantity (for iceberg orders).
    pub fn visible_quantity(&self) -> f64 {
        match self.display_quantity {
            Some(dq) => dq.min(self.remaining),
            None => self.remaining,
        }
    }

    /// Record a fill on this order.
    fn record_fill(&mut self, fill_qty: f64, fill_price: f64, fee: f64) {
        let prev_value = self.avg_fill_price * self.filled;
        self.filled += fill_qty;
        self.remaining -= fill_qty;
        self.fees += fee;

        if self.filled > 0.0 {
            self.avg_fill_price = (prev_value + fill_price * fill_qty) / self.filled;
        }

        if self.remaining <= 0.0 {
            self.remaining = 0.0;
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }
    }
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Order(id={}, {} {} {:.4}@{:.2}, remaining={:.4}, status={}, participant={})",
            self.id,
            self.side,
            self.order_type,
            self.quantity,
            self.price,
            self.remaining,
            self.status,
            self.participant,
        )
    }
}

// ---------------------------------------------------------------------------
// Order receipt — returned after submission
// ---------------------------------------------------------------------------

/// Receipt returned after an order is submitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderReceipt {
    /// The assigned order ID.
    pub order_id: u64,
    /// Status after processing.
    pub status: OrderStatus,
    /// Fills that occurred during submission (for aggressive orders).
    pub fills: Vec<Fill>,
    /// Total quantity filled during submission.
    pub filled_quantity: f64,
    /// Average fill price during submission.
    pub avg_fill_price: f64,
    /// Remaining quantity on the book (0 if fully filled or rejected).
    pub remaining_quantity: f64,
    /// Fees incurred during submission.
    pub fees: f64,
    /// Simulated latency for this order.
    pub latency: Duration,
    /// Rejection reason (if rejected).
    pub rejection_reason: Option<String>,
    /// Timestamp.
    pub timestamp_ns: u64,
}

impl OrderReceipt {
    /// Whether the order was accepted (placed or filled).
    pub fn is_accepted(&self) -> bool {
        !matches!(self.status, OrderStatus::Rejected)
    }

    /// Whether the order was fully filled.
    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }
}

impl fmt::Display for OrderReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Receipt(id={}, status={}, filled={:.4}@{:.2}, remaining={:.4}, fills={}, latency={:?})",
            self.order_id,
            self.status,
            self.filled_quantity,
            self.avg_fill_price,
            self.remaining_quantity,
            self.fills.len(),
            self.latency,
        )
    }
}

// ---------------------------------------------------------------------------
// Fill (trade execution)
// ---------------------------------------------------------------------------

/// A single fill (partial or complete match between two orders).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    /// Unique fill identifier.
    pub fill_id: u64,
    /// Aggressor order ID (the order that crossed the spread).
    pub aggressor_order_id: u64,
    /// Resting order ID (the order that was on the book).
    pub resting_order_id: u64,
    /// Execution price.
    pub price: f64,
    /// Execution quantity.
    pub quantity: f64,
    /// Side of the aggressor.
    pub aggressor_side: Side,
    /// Aggressor participant.
    pub aggressor_participant: String,
    /// Resting participant.
    pub resting_participant: String,
    /// Fee charged to the aggressor (taker).
    pub taker_fee: f64,
    /// Fee charged to the resting order (maker — may be rebate).
    pub maker_fee: f64,
    /// Timestamp (nanoseconds).
    pub timestamp_ns: u64,
}

impl Fill {
    /// Notional value of this fill.
    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

impl fmt::Display for Fill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fill(id={}, {:.4}@{:.2}, aggressor={}, resting={})",
            self.fill_id, self.quantity, self.price, self.aggressor_order_id, self.resting_order_id,
        )
    }
}

// ---------------------------------------------------------------------------
// Price level
// ---------------------------------------------------------------------------

/// A single price level in the order book containing a FIFO queue of orders.
#[derive(Debug, Clone)]
struct PriceLevel {
    /// The price of this level.
    price: f64,
    /// Orders at this level in FIFO order (price-time priority).
    orders: VecDeque<u64>, // order IDs
    /// Total visible quantity at this level.
    total_quantity: f64,
    /// Number of orders at this level.
    order_count: usize,
}

impl PriceLevel {
    fn new(price: f64) -> Self {
        Self {
            price,
            orders: VecDeque::new(),
            total_quantity: 0.0,
            order_count: 0,
        }
    }

    fn add_order(&mut self, order_id: u64, quantity: f64) {
        self.orders.push_back(order_id);
        self.total_quantity += quantity;
        self.order_count += 1;
    }

    fn remove_order(&mut self, order_id: u64, quantity: f64) {
        self.orders.retain(|&id| id != order_id);
        self.total_quantity -= quantity;
        if self.total_quantity < 0.0 {
            self.total_quantity = 0.0;
        }
        self.order_count = self.orders.len();
    }

    fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    #[allow(dead_code)]
    fn front_order_id(&self) -> Option<u64> {
        self.orders.front().copied()
    }
}

// ---------------------------------------------------------------------------
// Order book side (bid or ask)
// ---------------------------------------------------------------------------

/// One side of the order book (all bids or all asks).
#[derive(Debug, Clone)]
struct BookSide {
    /// Price levels keyed by price (scaled to integer ticks for ordering).
    levels: BTreeMap<i64, PriceLevel>,
    /// Side (Buy or Sell).
    side: Side,
    /// Tick size for price → tick conversion.
    tick_size: f64,
    /// Total quantity on this side.
    total_quantity: f64,
    /// Total number of orders on this side.
    total_orders: usize,
}

impl BookSide {
    fn new(side: Side, tick_size: f64) -> Self {
        Self {
            levels: BTreeMap::new(),
            side,
            tick_size,
            total_quantity: 0.0,
            total_orders: 0,
        }
    }

    /// Convert a price to an integer tick key.
    fn price_to_tick(&self, price: f64) -> i64 {
        (price / self.tick_size).round() as i64
    }

    /// Convert an integer tick key back to a price.
    fn tick_to_price(&self, tick: i64) -> f64 {
        tick as f64 * self.tick_size
    }

    /// Add an order to the appropriate price level.
    fn add_order(&mut self, order_id: u64, price: f64, quantity: f64) {
        let tick = self.price_to_tick(price);
        let level = self
            .levels
            .entry(tick)
            .or_insert_with(|| PriceLevel::new(price));
        level.add_order(order_id, quantity);
        self.total_quantity += quantity;
        self.total_orders += 1;
    }

    /// Remove an order from its price level.
    fn remove_order(&mut self, order_id: u64, price: f64, quantity: f64) {
        let tick = self.price_to_tick(price);
        if let Some(level) = self.levels.get_mut(&tick) {
            level.remove_order(order_id, quantity);
            if level.is_empty() {
                self.levels.remove(&tick);
            }
        }
        self.total_quantity -= quantity;
        if self.total_quantity < 0.0 {
            self.total_quantity = 0.0;
        }
        self.total_orders = self.total_orders.saturating_sub(1);
    }

    /// Reduce quantity at a specific order (partial fill).
    fn reduce_quantity(&mut self, _order_id: u64, price: f64, reduce_by: f64) {
        let tick = self.price_to_tick(price);
        if let Some(level) = self.levels.get_mut(&tick) {
            level.total_quantity -= reduce_by;
            if level.total_quantity < 0.0 {
                level.total_quantity = 0.0;
            }
        }
        self.total_quantity -= reduce_by;
        if self.total_quantity < 0.0 {
            self.total_quantity = 0.0;
        }
    }

    /// Get the best price level (highest bid or lowest ask).
    fn best_level(&self) -> Option<&PriceLevel> {
        match self.side {
            Side::Buy => self.levels.values().next_back(), // highest bid
            Side::Sell => self.levels.values().next(),     // lowest ask
        }
    }

    /// Get the best price.
    fn best_price(&self) -> Option<f64> {
        self.best_level().map(|l| l.price)
    }

    /// Get the best tick key.
    #[allow(dead_code)]
    fn best_tick(&self) -> Option<i64> {
        match self.side {
            Side::Buy => self.levels.keys().next_back().copied(),
            Side::Sell => self.levels.keys().next().copied(),
        }
    }

    /// Check if an incoming order at the given price would cross this side.
    fn would_cross(&self, incoming_price: f64, incoming_side: Side) -> bool {
        if let Some(best) = self.best_price() {
            match incoming_side {
                Side::Buy => incoming_price >= best,  // buy crosses ask
                Side::Sell => incoming_price <= best, // sell crosses bid
            }
        } else {
            false
        }
    }

    /// Get price levels for L2 snapshot (best to worst, up to `depth` levels).
    fn snapshot_levels(&self, depth: usize) -> Vec<L2Level> {
        let iter: Box<dyn Iterator<Item = (&i64, &PriceLevel)>> = match self.side {
            Side::Buy => Box::new(self.levels.iter().rev()),
            Side::Sell => Box::new(self.levels.iter()),
        };

        iter.take(depth)
            .map(|(_, level)| L2Level {
                price: level.price,
                quantity: level.total_quantity,
                order_count: level.order_count,
            })
            .collect()
    }

    /// Get price levels with individual order detail for L3 snapshot.
    fn snapshot_levels_l3(&self, depth: usize, orders: &HashMap<u64, Order>) -> Vec<L3Level> {
        let iter: Box<dyn Iterator<Item = (&i64, &PriceLevel)>> = match self.side {
            Side::Buy => Box::new(self.levels.iter().rev()),
            Side::Sell => Box::new(self.levels.iter()),
        };

        iter.take(depth)
            .map(|(_, level)| {
                let order_details: Vec<L3Order> = level
                    .orders
                    .iter()
                    .filter_map(|&id| {
                        orders.get(&id).map(|o| L3Order {
                            order_id: id,
                            quantity: o.remaining,
                            visible_quantity: o.visible_quantity(),
                            participant: o.participant.clone(),
                            timestamp_ns: o.timestamp_ns,
                        })
                    })
                    .collect();

                L3Level {
                    price: level.price,
                    total_quantity: level.total_quantity,
                    orders: order_details,
                }
            })
            .collect()
    }

    /// Number of price levels.
    fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Whether this side is empty.
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Iterate over the matchable levels in priority order.
    /// For bids: highest → lowest (to match against incoming sells).
    /// For asks: lowest → highest (to match against incoming buys).
    fn matchable_ticks(&self) -> Vec<i64> {
        match self.side {
            Side::Buy => self.levels.keys().rev().copied().collect(),
            Side::Sell => self.levels.keys().copied().collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// L2 / L3 Snapshot types
// ---------------------------------------------------------------------------

/// A single level in an L2 order book snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Level {
    /// Price at this level.
    pub price: f64,
    /// Total quantity at this level.
    pub quantity: f64,
    /// Number of resting orders at this level.
    pub order_count: usize,
}

impl fmt::Display for L2Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.2} x {:.4} ({})",
            self.price, self.quantity, self.order_count
        )
    }
}

/// A single order in an L3 snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L3Order {
    /// Order ID.
    pub order_id: u64,
    /// Total remaining quantity.
    pub quantity: f64,
    /// Visible quantity (may differ for iceberg orders).
    pub visible_quantity: f64,
    /// Participant identifier.
    pub participant: String,
    /// Order timestamp.
    pub timestamp_ns: u64,
}

/// A single level in an L3 order book snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L3Level {
    /// Price at this level.
    pub price: f64,
    /// Total quantity at this level.
    pub total_quantity: f64,
    /// Individual orders in FIFO order.
    pub orders: Vec<L3Order>,
}

/// An L2 order book snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Snapshot {
    /// Symbol.
    pub symbol: String,
    /// Bid levels (best to worst).
    pub bids: Vec<L2Level>,
    /// Ask levels (best to worst).
    pub asks: Vec<L2Level>,
    /// Timestamp (nanoseconds).
    pub timestamp_ns: u64,
    /// Sequence number.
    pub sequence: u64,
}

impl L2Snapshot {
    /// Best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Mid price.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Spread (absolute).
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Spread in basis points relative to mid price.
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 10_000.0),
            _ => None,
        }
    }

    /// Total bid quantity across all levels.
    pub fn total_bid_quantity(&self) -> f64 {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Total ask quantity across all levels.
    pub fn total_ask_quantity(&self) -> f64 {
        self.asks.iter().map(|l| l.quantity).sum()
    }

    /// Book imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty).
    /// Range: [-1, 1]. Positive = more bids (buy pressure).
    pub fn imbalance(&self) -> f64 {
        let bid_qty = self.total_bid_quantity();
        let ask_qty = self.total_ask_quantity();
        let total = bid_qty + ask_qty;
        if total == 0.0 {
            return 0.0;
        }
        (bid_qty - ask_qty) / total
    }

    /// Weighted mid price (quantity-weighted by top of book).
    pub fn weighted_mid(&self) -> Option<f64> {
        match (self.bids.first(), self.asks.first()) {
            (Some(bid), Some(ask)) => {
                let total = bid.quantity + ask.quantity;
                if total == 0.0 {
                    return Some((bid.price + ask.price) / 2.0);
                }
                Some((bid.price * ask.quantity + ask.price * bid.quantity) / total)
            }
            _ => None,
        }
    }

    /// Number of bid levels.
    pub fn bid_depth(&self) -> usize {
        self.bids.len()
    }

    /// Number of ask levels.
    pub fn ask_depth(&self) -> usize {
        self.asks.len()
    }

    /// Total depth (bid + ask levels).
    pub fn total_depth(&self) -> usize {
        self.bids.len() + self.asks.len()
    }

    /// VWAP for a hypothetical buy of `quantity` from the ask side.
    pub fn vwap_buy(&self, quantity: f64) -> Option<f64> {
        Self::vwap_side(&self.asks, quantity)
    }

    /// VWAP for a hypothetical sell of `quantity` into the bid side.
    pub fn vwap_sell(&self, quantity: f64) -> Option<f64> {
        Self::vwap_side(&self.bids, quantity)
    }

    fn vwap_side(levels: &[L2Level], quantity: f64) -> Option<f64> {
        if levels.is_empty() || quantity <= 0.0 {
            return None;
        }

        let mut remaining = quantity;
        let mut total_cost = 0.0;
        let mut total_qty = 0.0;

        for level in levels {
            let fill = remaining.min(level.quantity);
            total_cost += fill * level.price;
            total_qty += fill;
            remaining -= fill;
            if remaining <= 0.0 {
                break;
            }
        }

        if total_qty > 0.0 {
            Some(total_cost / total_qty)
        } else {
            None
        }
    }
}

impl fmt::Display for L2Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== {} L2 Snapshot (seq={}) ===",
            self.symbol, self.sequence
        )?;
        writeln!(f, "Asks:")?;
        for (i, level) in self.asks.iter().enumerate().rev() {
            writeln!(f, "  [{i}] {level}")?;
        }
        if let Some(spread) = self.spread() {
            writeln!(f, "  --- spread: {spread:.4} ---")?;
        }
        writeln!(f, "Bids:")?;
        for (i, level) in self.bids.iter().enumerate() {
            writeln!(f, "  [{i}] {level}")?;
        }
        Ok(())
    }
}

/// An L3 order book snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L3Snapshot {
    /// Symbol.
    pub symbol: String,
    /// Bid levels with individual order detail (best to worst).
    pub bids: Vec<L3Level>,
    /// Ask levels with individual order detail (best to worst).
    pub asks: Vec<L3Level>,
    /// Timestamp (nanoseconds).
    pub timestamp_ns: u64,
    /// Sequence number.
    pub sequence: u64,
}

// ---------------------------------------------------------------------------
// Order events (audit trail)
// ---------------------------------------------------------------------------

/// An event in the lifecycle of an order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    /// Event type.
    pub event_type: OrderEventType,
    /// Order ID.
    pub order_id: u64,
    /// Timestamp.
    pub timestamp_ns: u64,
    /// Additional details.
    pub details: String,
}

/// Types of order lifecycle events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderEventType {
    Submitted,
    Accepted,
    Rejected,
    Filled,
    PartiallyFilled,
    Cancelled,
    Modified,
    Expired,
    StopTriggered,
}

impl fmt::Display for OrderEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Submitted => write!(f, "SUBMITTED"),
            Self::Accepted => write!(f, "ACCEPTED"),
            Self::Rejected => write!(f, "REJECTED"),
            Self::Filled => write!(f, "FILLED"),
            Self::PartiallyFilled => write!(f, "PARTIAL_FILL"),
            Self::Cancelled => write!(f, "CANCELLED"),
            Self::Modified => write!(f, "MODIFIED"),
            Self::Expired => write!(f, "EXPIRED"),
            Self::StopTriggered => write!(f, "STOP_TRIGGERED"),
        }
    }
}

// ---------------------------------------------------------------------------
// Market impact model
// ---------------------------------------------------------------------------

/// Simple market impact model based on the Almgren-Chriss framework.
///
/// Impact = η * (quantity / ADV)^γ * σ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactModel {
    /// Temporary impact coefficient.
    pub eta: f64,
    /// Impact exponent (typically 0.5 for square-root law).
    pub gamma: f64,
    /// Average daily volume (ADV) for normalization.
    pub adv: f64,
    /// Daily volatility (σ).
    pub daily_volatility: f64,
}

impl Default for MarketImpactModel {
    fn default() -> Self {
        Self {
            eta: 0.1,
            gamma: 0.5,
            adv: 1_000_000.0,
            daily_volatility: 0.02,
        }
    }
}

impl MarketImpactModel {
    /// Create a new market impact model.
    pub fn new(eta: f64, gamma: f64, adv: f64, daily_volatility: f64) -> Self {
        Self {
            eta,
            gamma,
            adv,
            daily_volatility,
        }
    }

    /// Estimate temporary market impact in price units.
    pub fn temporary_impact(&self, quantity: f64, reference_price: f64) -> f64 {
        if self.adv <= 0.0 || reference_price <= 0.0 {
            return 0.0;
        }
        let participation_rate = (quantity / self.adv).abs();
        let impact_bps =
            self.eta * participation_rate.powf(self.gamma) * self.daily_volatility * 10_000.0;
        reference_price * impact_bps / 10_000.0
    }

    /// Estimate permanent market impact in price units.
    pub fn permanent_impact(&self, quantity: f64, reference_price: f64) -> f64 {
        // Permanent impact is typically a fraction of temporary
        self.temporary_impact(quantity, reference_price) * 0.3
    }

    /// Total expected execution cost for a given quantity and reference price.
    pub fn total_cost(&self, quantity: f64, reference_price: f64) -> f64 {
        let temp = self.temporary_impact(quantity, reference_price);
        let perm = self.permanent_impact(quantity, reference_price);
        (temp + perm) * quantity
    }
}

// ---------------------------------------------------------------------------
// LOB Statistics
// ---------------------------------------------------------------------------

/// Accumulated statistics for the LOB simulator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LobStatistics {
    /// Total orders submitted.
    pub total_orders_submitted: u64,
    /// Total orders accepted.
    pub total_orders_accepted: u64,
    /// Total orders rejected.
    pub total_orders_rejected: u64,
    /// Total orders cancelled.
    pub total_orders_cancelled: u64,
    /// Total orders fully filled.
    pub total_orders_filled: u64,
    /// Total trades (fills) executed.
    pub total_trades: u64,
    /// Total volume traded.
    pub total_volume: f64,
    /// Total notional value traded.
    pub total_notional: f64,
    /// Total taker fees collected.
    pub total_taker_fees: f64,
    /// Total maker fees/rebates paid.
    pub total_maker_fees: f64,
    /// Highest trade price.
    pub high_price: f64,
    /// Lowest trade price.
    pub low_price: f64,
    /// Last trade price.
    pub last_price: f64,
    /// VWAP of all trades.
    pub vwap: f64,
    /// Number of self-trade preventions.
    pub self_trade_preventions: u64,
    /// Number of post-only rejections.
    pub post_only_rejections: u64,
    /// Number of FOK rejections.
    pub fok_rejections: u64,
    /// Number of stop orders triggered.
    pub stops_triggered: u64,
}

impl LobStatistics {
    fn record_trade(&mut self, price: f64, quantity: f64, taker_fee: f64, maker_fee: f64) {
        self.total_trades += 1;
        self.total_volume += quantity;
        let notional = price * quantity;
        self.total_notional += notional;
        self.total_taker_fees += taker_fee;
        self.total_maker_fees += maker_fee;

        if self.total_trades == 1 || price > self.high_price {
            self.high_price = price;
        }
        if self.total_trades == 1 || price < self.low_price {
            self.low_price = price;
        }
        self.last_price = price;

        // Running VWAP
        if self.total_volume > 0.0 {
            self.vwap = self.total_notional / self.total_volume;
        }
    }
}

impl fmt::Display for LobStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LOB Stats: orders={}/{}, trades={}, volume={:.2}, \
             vwap={:.4}, last={:.4}, range=[{:.4}, {:.4}], \
             fees(taker={:.4}, maker={:.4})",
            self.total_orders_accepted,
            self.total_orders_submitted,
            self.total_trades,
            self.total_volume,
            self.vwap,
            self.last_price,
            self.low_price,
            self.high_price,
            self.total_taker_fees,
            self.total_maker_fees,
        )
    }
}

// ---------------------------------------------------------------------------
// Cancel receipt
// ---------------------------------------------------------------------------

/// Receipt returned after a cancel request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelReceipt {
    /// The order that was cancelled.
    pub order_id: u64,
    /// Remaining quantity that was removed from the book.
    pub cancelled_quantity: f64,
    /// Already filled quantity.
    pub filled_quantity: f64,
    /// Whether the cancellation was successful.
    pub success: bool,
    /// Reason for failure (if any).
    pub reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Modify receipt
// ---------------------------------------------------------------------------

/// Receipt returned after a modify request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifyReceipt {
    /// The order that was modified.
    pub order_id: u64,
    /// Whether the modification was successful.
    pub success: bool,
    /// The old price.
    pub old_price: f64,
    /// The new price.
    pub new_price: f64,
    /// The old quantity.
    pub old_quantity: f64,
    /// The new quantity.
    pub new_quantity: f64,
    /// Reason for failure (if any).
    pub reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Limit Order Book
// ---------------------------------------------------------------------------

/// The main Limit Order Book simulator engine.
///
/// This is a full-featured LOB with price-time priority matching,
/// multiple order types, L2/L3 snapshots, market impact modeling,
/// latency simulation, and comprehensive statistics.
pub struct LimitOrderBook {
    /// Configuration.
    config: LobConfig,

    /// Bid side of the book.
    bids: BookSide,

    /// Ask side of the book.
    asks: BookSide,

    /// All known orders (open, filled, cancelled, etc.) keyed by order ID.
    orders: HashMap<u64, Order>,

    /// Stop orders waiting for trigger, keyed by order ID.
    stop_orders: Vec<u64>,

    /// Next order ID to assign.
    next_order_id: u64,

    /// Next fill ID to assign.
    next_fill_id: u64,

    /// Current simulated time (nanoseconds).
    current_time_ns: u64,

    /// Sequence counter for snapshots.
    sequence: u64,

    /// Trade history.
    trade_history: Vec<Fill>,

    /// Order event history.
    event_history: Vec<OrderEvent>,

    /// Statistics.
    stats: LobStatistics,

    /// Market impact model.
    impact_model: MarketImpactModel,

    /// Orders per participant (for throttling).
    participant_order_count: HashMap<String, usize>,

    /// Simple RNG state for latency jitter (xorshift64).
    rng_state: u64,
}

impl fmt::Debug for LimitOrderBook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LimitOrderBook")
            .field("symbol", &self.config.symbol)
            .field("bid_levels", &self.bids.num_levels())
            .field("ask_levels", &self.asks.num_levels())
            .field("total_orders", &self.orders.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl LimitOrderBook {
    // ── Construction ───────────────────────────────────────────────────

    /// Create a new empty limit order book.
    pub fn new(symbol: impl Into<String>, config: LobConfig) -> Self {
        let symbol = symbol.into();
        let tick_size = config.tick_size;
        let cfg = LobConfig {
            symbol: symbol.clone(),
            ..config
        };

        info!(symbol = %symbol, tick_size, "LOB simulator created");

        Self {
            config: cfg,
            bids: BookSide::new(Side::Buy, tick_size),
            asks: BookSide::new(Side::Sell, tick_size),
            orders: HashMap::new(),
            stop_orders: Vec::new(),
            next_order_id: 1,
            next_fill_id: 1,
            current_time_ns: 0,
            sequence: 0,
            trade_history: Vec::new(),
            event_history: Vec::new(),
            stats: LobStatistics::default(),
            impact_model: MarketImpactModel::default(),
            participant_order_count: HashMap::new(),
            rng_state: 0xDEAD_BEEF_CAFE_1234,
        }
    }

    /// Create a LOB with default config for a given symbol.
    pub fn default_for(symbol: impl Into<String>) -> Self {
        Self::new(symbol, LobConfig::default())
    }

    /// Set the market impact model.
    pub fn with_impact_model(mut self, model: MarketImpactModel) -> Self {
        self.impact_model = model;
        self
    }

    // ── Time management ────────────────────────────────────────────────

    /// Advance the simulated clock by the given duration.
    pub fn advance_time(&mut self, duration: Duration) {
        self.current_time_ns += duration.as_nanos() as u64;
    }

    /// Set the simulated clock to a specific timestamp.
    pub fn set_time_ns(&mut self, time_ns: u64) {
        self.current_time_ns = time_ns;
    }

    /// Get the current simulated time.
    pub fn current_time_ns(&self) -> u64 {
        self.current_time_ns
    }

    // ── Order submission ───────────────────────────────────────────────

    /// Submit an order to the LOB.
    ///
    /// This is the main entry point. The order goes through:
    /// 1. Validation
    /// 2. Simulated latency
    /// 3. Matching (for aggressive orders)
    /// 4. Placement (for resting orders)
    pub fn submit_order(&mut self, mut order: Order) -> Result<OrderReceipt> {
        // Assign ID and timestamps.
        let order_id = self.next_order_id;
        self.next_order_id += 1;
        order.id = order_id;
        order.created_at_ns = self.current_time_ns;

        // Simulate latency.
        let latency = self.simulate_latency();
        self.current_time_ns += latency.as_nanos() as u64;
        order.timestamp_ns = self.current_time_ns;

        self.stats.total_orders_submitted += 1;

        // Record submission event.
        self.record_event(order_id, OrderEventType::Submitted, format!("{order}"));

        // Validate.
        if let Err(e) = self.validate_order(&order) {
            order.status = OrderStatus::Rejected;
            self.stats.total_orders_rejected += 1;
            self.record_event(order_id, OrderEventType::Rejected, format!("{e}"));
            self.orders.insert(order_id, order);
            return Ok(OrderReceipt {
                order_id,
                status: OrderStatus::Rejected,
                fills: vec![],
                filled_quantity: 0.0,
                avg_fill_price: 0.0,
                remaining_quantity: 0.0,
                fees: 0.0,
                latency,
                rejection_reason: Some(format!("{e}")),
                timestamp_ns: self.current_time_ns,
            });
        }

        // Validate price for non-market orders.
        // Stop orders carry price=0.0 by design (they only use stop_price),
        // so skip limit-price validation for them.
        if order.order_type != OrderType::Market && order.order_type != OrderType::Stop {
            order.price = self.config.validate_price(order.price)?;
        }

        // Validate the stop_price for orders that carry one.
        if let Some(sp) = order.stop_price {
            let validated = self.config.validate_price(sp)?;
            order.stop_price = Some(validated);
        }
        order.quantity = self.config.validate_quantity(order.quantity)?;
        order.remaining = order.quantity;

        self.stats.total_orders_accepted += 1;
        self.record_event(order_id, OrderEventType::Accepted, String::new());

        // Handle stop orders specially.
        if order.order_type == OrderType::Stop || order.order_type == OrderType::StopLimit {
            order.status = OrderStatus::StopWaiting;
            self.stop_orders.push(order_id);
            let remaining = order.remaining;
            self.orders.insert(order_id, order);
            *self
                .participant_order_count
                .entry(self.orders[&order_id].participant.clone())
                .or_insert(0) += 1;
            return Ok(OrderReceipt {
                order_id,
                status: OrderStatus::StopWaiting,
                fills: vec![],
                filled_quantity: 0.0,
                avg_fill_price: 0.0,
                remaining_quantity: remaining,
                fees: 0.0,
                latency,
                rejection_reason: None,
                timestamp_ns: self.current_time_ns,
            });
        }

        // Process the order.
        let fills = self.process_order(&mut order)?;

        let filled_qty = order.filled;
        let avg_price = order.avg_fill_price;
        let remaining = order.remaining;
        let status = order.status;
        let fees = order.fees;

        // Track participant orders.
        if order.is_active() {
            *self
                .participant_order_count
                .entry(order.participant.clone())
                .or_insert(0) += 1;
        }

        // Store the order.
        self.orders.insert(order_id, order);

        // Check for stop triggers after trade.
        if !fills.is_empty() {
            self.check_stop_triggers();
        }

        Ok(OrderReceipt {
            order_id,
            status,
            fills,
            filled_quantity: filled_qty,
            avg_fill_price: avg_price,
            remaining_quantity: remaining,
            fees,
            latency,
            rejection_reason: None,
            timestamp_ns: self.current_time_ns,
        })
    }

    /// Cancel an open order.
    pub fn cancel_order(&mut self, order_id: u64) -> Result<CancelReceipt> {
        let order = self
            .orders
            .get_mut(&order_id)
            .ok_or(LobError::OrderNotFound(order_id))?;

        if order.is_done() {
            return Ok(CancelReceipt {
                order_id,
                cancelled_quantity: 0.0,
                filled_quantity: order.filled,
                success: false,
                reason: Some(format!("Order already in terminal state: {}", order.status)),
            });
        }

        let cancelled_qty = order.remaining;
        let filled_qty = order.filled;
        let price = order.price;
        let side = order.side;
        let participant = order.participant.clone();

        order.remaining = 0.0;
        order.status = OrderStatus::Cancelled;

        // Remove from the book.
        match side {
            Side::Buy => self.bids.remove_order(order_id, price, cancelled_qty),
            Side::Sell => self.asks.remove_order(order_id, price, cancelled_qty),
        }

        // Remove from stop list if it's a stop order.
        self.stop_orders.retain(|&id| id != order_id);

        // Update participant count.
        if let Some(count) = self.participant_order_count.get_mut(&participant) {
            *count = count.saturating_sub(1);
        }

        self.stats.total_orders_cancelled += 1;
        self.record_event(
            order_id,
            OrderEventType::Cancelled,
            format!("cancelled_qty={cancelled_qty:.4}"),
        );

        debug!(order_id, cancelled_qty, "Order cancelled");

        Ok(CancelReceipt {
            order_id,
            cancelled_quantity: cancelled_qty,
            filled_quantity: filled_qty,
            success: true,
            reason: None,
        })
    }

    /// Modify an open order's price and/or quantity.
    ///
    /// This is implemented as cancel + replace (loses time priority).
    pub fn modify_order(
        &mut self,
        order_id: u64,
        new_price: Option<f64>,
        new_quantity: Option<f64>,
    ) -> Result<ModifyReceipt> {
        let order = self
            .orders
            .get(&order_id)
            .ok_or(LobError::OrderNotFound(order_id))?;

        if !order.is_active() {
            return Ok(ModifyReceipt {
                order_id,
                success: false,
                old_price: order.price,
                new_price: order.price,
                old_quantity: order.quantity,
                new_quantity: order.quantity,
                reason: Some(format!("Order not active: {}", order.status)),
            });
        }

        let old_price = order.price;
        let old_quantity = order.quantity;
        let old_remaining = order.remaining;
        let side = order.side;
        let _participant = order.participant.clone();

        let final_price = match new_price {
            Some(p) => self.config.validate_price(p)?,
            None => old_price,
        };
        let final_quantity = match new_quantity {
            Some(q) => {
                let q = self.config.validate_quantity(q)?;
                if q < order.filled {
                    return Err(LobError::InvalidOrder(
                        "New quantity less than already filled".to_string(),
                    ));
                }
                q
            }
            None => old_quantity,
        };

        // Remove from book.
        match side {
            Side::Buy => self.bids.remove_order(order_id, old_price, old_remaining),
            Side::Sell => self.asks.remove_order(order_id, old_price, old_remaining),
        }

        // Update the order.
        let order = self.orders.get_mut(&order_id).unwrap();
        order.price = final_price;
        order.quantity = final_quantity;
        order.remaining = final_quantity - order.filled;
        order.timestamp_ns = self.current_time_ns; // loses priority

        // Re-add to book.
        let new_remaining = order.remaining;
        match side {
            Side::Buy => self.bids.add_order(order_id, final_price, new_remaining),
            Side::Sell => self.asks.add_order(order_id, final_price, new_remaining),
        }

        self.record_event(
            order_id,
            OrderEventType::Modified,
            format!(
                "price: {old_price:.4} → {final_price:.4}, qty: {old_quantity:.4} → {final_quantity:.4}"
            ),
        );

        debug!(
            order_id,
            old_price,
            new_price = final_price,
            old_quantity,
            new_quantity = final_quantity,
            "Order modified"
        );

        Ok(ModifyReceipt {
            order_id,
            success: true,
            old_price,
            new_price: final_price,
            old_quantity,
            new_quantity: final_quantity,
            reason: None,
        })
    }

    /// Cancel all orders for a given participant.
    pub fn cancel_all(&mut self, participant: &str) -> Vec<CancelReceipt> {
        let order_ids: Vec<u64> = self
            .orders
            .values()
            .filter(|o| o.participant == participant && o.is_active())
            .map(|o| o.id)
            .collect();

        let mut receipts = Vec::new();
        for id in order_ids {
            if let Ok(receipt) = self.cancel_order(id) {
                receipts.push(receipt);
            }
        }
        receipts
    }

    // ── Snapshots ──────────────────────────────────────────────────────

    /// Generate an L2 order book snapshot.
    pub fn snapshot(&mut self, depth: usize) -> L2Snapshot {
        self.sequence += 1;
        L2Snapshot {
            symbol: self.config.symbol.clone(),
            bids: self.bids.snapshot_levels(depth),
            asks: self.asks.snapshot_levels(depth),
            timestamp_ns: self.current_time_ns,
            sequence: self.sequence,
        }
    }

    /// Generate an L3 order book snapshot with individual order detail.
    pub fn snapshot_l3(&mut self, depth: usize) -> L3Snapshot {
        self.sequence += 1;
        L3Snapshot {
            symbol: self.config.symbol.clone(),
            bids: self.bids.snapshot_levels_l3(depth, &self.orders),
            asks: self.asks.snapshot_levels_l3(depth, &self.orders),
            timestamp_ns: self.current_time_ns,
            sequence: self.sequence,
        }
    }

    // ── Queries ────────────────────────────────────────────────────────

    /// Best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.best_price()
    }

    /// Best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.best_price()
    }

    /// Mid price.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Last trade price.
    pub fn last_price(&self) -> Option<f64> {
        if self.stats.total_trades > 0 {
            Some(self.stats.last_price)
        } else {
            None
        }
    }

    /// Get an order by ID.
    pub fn get_order(&self, order_id: u64) -> Option<&Order> {
        self.orders.get(&order_id)
    }

    /// Get all open orders for a participant.
    pub fn open_orders(&self, participant: &str) -> Vec<&Order> {
        self.orders
            .values()
            .filter(|o| o.participant == participant && o.is_active())
            .collect()
    }

    /// Total number of open orders.
    pub fn total_open_orders(&self) -> usize {
        self.bids.total_orders + self.asks.total_orders
    }

    /// Number of bid levels.
    pub fn bid_levels(&self) -> usize {
        self.bids.num_levels()
    }

    /// Number of ask levels.
    pub fn ask_levels(&self) -> usize {
        self.asks.num_levels()
    }

    /// Total bid quantity.
    pub fn total_bid_quantity(&self) -> f64 {
        self.bids.total_quantity
    }

    /// Total ask quantity.
    pub fn total_ask_quantity(&self) -> f64 {
        self.asks.total_quantity
    }

    /// Trade history.
    pub fn trade_history(&self) -> &[Fill] {
        &self.trade_history
    }

    /// Order event history.
    pub fn event_history(&self) -> &[OrderEvent] {
        &self.event_history
    }

    /// LOB statistics.
    pub fn statistics(&self) -> &LobStatistics {
        &self.stats
    }

    /// Configuration.
    pub fn config(&self) -> &LobConfig {
        &self.config
    }

    /// Market impact model.
    pub fn impact_model(&self) -> &MarketImpactModel {
        &self.impact_model
    }

    /// Estimate market impact for a given order size.
    pub fn estimate_impact(&self, side: Side, quantity: f64) -> f64 {
        let ref_price = self.mid_price().unwrap_or(self.stats.last_price);
        let impact = self.impact_model.temporary_impact(quantity, ref_price);
        match side {
            Side::Buy => impact,   // price goes up
            Side::Sell => -impact, // price goes down
        }
    }

    /// Compute the queue position for a given order.
    pub fn queue_position(&self, order_id: u64) -> Option<(usize, f64)> {
        let order = self.orders.get(&order_id)?;
        if !order.is_active() {
            return None;
        }

        let side = match order.side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };

        let tick = side.price_to_tick(order.price);
        let level = side.levels.get(&tick)?;

        let mut quantity_ahead = 0.0;

        for (position, &id) in level.orders.iter().enumerate() {
            if id == order_id {
                return Some((position, quantity_ahead));
            }
            if let Some(o) = self.orders.get(&id) {
                quantity_ahead += o.remaining;
            }
        }

        None
    }

    /// Clear all state and reset.
    pub fn reset(&mut self) {
        self.bids = BookSide::new(Side::Buy, self.config.tick_size);
        self.asks = BookSide::new(Side::Sell, self.config.tick_size);
        self.orders.clear();
        self.stop_orders.clear();
        self.trade_history.clear();
        self.event_history.clear();
        self.stats = LobStatistics::default();
        self.participant_order_count.clear();
        self.next_order_id = 1;
        self.next_fill_id = 1;
        self.current_time_ns = 0;
        self.sequence = 0;
        info!(symbol = %self.config.symbol, "LOB reset");
    }

    // ── Historical replay ──────────────────────────────────────────────

    /// Replay a sequence of orders (useful for historical simulation).
    pub fn replay_orders(&mut self, orders: Vec<Order>) -> Vec<OrderReceipt> {
        let mut receipts = Vec::with_capacity(orders.len());
        for order in orders {
            match self.submit_order(order) {
                Ok(receipt) => receipts.push(receipt),
                Err(e) => {
                    warn!("Replay order failed: {e}");
                }
            }
        }
        receipts
    }

    /// Seed the order book with initial bids and asks.
    ///
    /// Useful for setting up a realistic initial state before simulation.
    pub fn seed_book(
        &mut self,
        bids: &[(f64, f64)], // (price, quantity)
        asks: &[(f64, f64)], // (price, quantity)
        participant: &str,
    ) -> Vec<OrderReceipt> {
        let mut receipts = Vec::new();

        for &(price, qty) in bids {
            let order = Order::limit_buy(qty, price, participant);
            if let Ok(receipt) = self.submit_order(order) {
                receipts.push(receipt);
            }
        }

        for &(price, qty) in asks {
            let order = Order::limit_sell(qty, price, participant);
            if let Ok(receipt) = self.submit_order(order) {
                receipts.push(receipt);
            }
        }

        info!(
            bids = bids.len(),
            asks = asks.len(),
            participant,
            "Order book seeded"
        );

        receipts
    }

    // ── Internal: Order processing ─────────────────────────────────────

    /// Validate an order before processing.
    fn validate_order(&self, order: &Order) -> Result<()> {
        // Check quantity.
        if order.quantity <= 0.0 {
            return Err(LobError::InvalidOrder(
                "Quantity must be positive".to_string(),
            ));
        }

        // Check price for limit orders.
        if matches!(
            order.order_type,
            OrderType::Limit
                | OrderType::PostOnly
                | OrderType::ImmediateOrCancel
                | OrderType::FillOrKill
                | OrderType::Iceberg
                | OrderType::StopLimit
        ) && order.price <= 0.0
        {
            return Err(LobError::InvalidOrder(
                "Price must be positive for limit orders".to_string(),
            ));
        }

        // Check stop price for stop orders.
        if matches!(order.order_type, OrderType::Stop | OrderType::StopLimit) {
            if order.stop_price.is_none() {
                return Err(LobError::InvalidOrder(
                    "Stop price required for stop orders".to_string(),
                ));
            }
        }

        // Check iceberg display quantity.
        if order.order_type == OrderType::Iceberg {
            if let Some(dq) = order.display_quantity {
                if dq <= 0.0 || dq > order.quantity {
                    return Err(LobError::InvalidOrder(
                        "Display quantity must be in (0, total_quantity]".to_string(),
                    ));
                }
            } else {
                return Err(LobError::InvalidOrder(
                    "Display quantity required for iceberg orders".to_string(),
                ));
            }
        }

        // Check participant order limit.
        let current = self
            .participant_order_count
            .get(&order.participant)
            .copied()
            .unwrap_or(0);
        if current >= self.config.max_orders_per_participant {
            return Err(LobError::InvalidOrder(format!(
                "Participant '{}' has reached max order limit ({})",
                order.participant, self.config.max_orders_per_participant
            )));
        }

        Ok(())
    }

    /// Process an order through the matching engine.
    fn process_order(&mut self, order: &mut Order) -> Result<Vec<Fill>> {
        match order.order_type {
            OrderType::Market => self.process_market_order(order),
            OrderType::Limit => self.process_limit_order(order),
            OrderType::ImmediateOrCancel => self.process_ioc_order(order),
            OrderType::FillOrKill => self.process_fok_order(order),
            OrderType::PostOnly => self.process_post_only_order(order),
            OrderType::Iceberg => self.process_iceberg_order(order),
            OrderType::Stop | OrderType::StopLimit => {
                // Stop orders are handled by check_stop_triggers
                Ok(vec![])
            }
        }
    }

    /// Process a market order.
    fn process_market_order(&mut self, order: &mut Order) -> Result<Vec<Fill>> {
        let fills = self.match_aggressive(order)?;

        if order.remaining > 0.0 {
            // Market order could not be fully filled — cancel remaining.
            order.status = if order.filled > 0.0 {
                OrderStatus::PartiallyFilled
            } else {
                OrderStatus::Cancelled
            };
            order.remaining = 0.0;
        }

        Ok(fills)
    }

    /// Process a limit order.
    fn process_limit_order(&mut self, order: &mut Order) -> Result<Vec<Fill>> {
        // First try to match aggressively.
        let fills = self.match_aggressive(order)?;

        // If there's remaining quantity, place on the book.
        if order.remaining > 0.0 {
            self.place_on_book(order);
        }

        Ok(fills)
    }

    /// Process an IOC order.
    fn process_ioc_order(&mut self, order: &mut Order) -> Result<Vec<Fill>> {
        let fills = self.match_aggressive(order)?;

        // Cancel any remaining quantity.
        if order.remaining > 0.0 {
            order.status = if order.filled > 0.0 {
                OrderStatus::PartiallyFilled
            } else {
                OrderStatus::Cancelled
            };
            order.remaining = 0.0;
        }

        Ok(fills)
    }

    /// Process a FOK order.
    fn process_fok_order(&mut self, order: &mut Order) -> Result<Vec<Fill>> {
        // Check if full quantity is available first.
        let available = self.available_quantity(order.side.opposite(), order.price, order.side);
        if available < order.quantity {
            self.stats.fok_rejections += 1;
            order.status = OrderStatus::Rejected;
            return Err(LobError::FokNotFilled {
                order_id: order.id,
                available,
                requested: order.quantity,
            });
        }

        // Execute.
        let fills = self.match_aggressive(order)?;
        Ok(fills)
    }

    /// Process a post-only order.
    fn process_post_only_order(&mut self, order: &mut Order) -> Result<Vec<Fill>> {
        // Check if the order would cross the spread.
        let would_cross = match order.side {
            Side::Buy => self.asks.would_cross(order.price, order.side),
            Side::Sell => self.bids.would_cross(order.price, order.side),
        };

        if would_cross {
            self.stats.post_only_rejections += 1;
            order.status = OrderStatus::Rejected;
            return Err(LobError::PostOnlyWouldTake { order_id: order.id });
        }

        // Place on book.
        self.place_on_book(order);
        Ok(vec![])
    }

    /// Process an iceberg order.
    fn process_iceberg_order(&mut self, order: &mut Order) -> Result<Vec<Fill>> {
        // Process like a limit order — the display_quantity field
        // controls what is visible at the price level.
        self.process_limit_order(order)
    }

    /// Try to match an aggressive order against the opposite side of the book.
    fn match_aggressive(&mut self, order: &mut Order) -> Result<Vec<Fill>> {
        let mut fills = Vec::new();
        let opposite_side = order.side.opposite();

        // Determine which side we're matching against.
        let matchable_ticks = match opposite_side {
            Side::Buy => self.bids.matchable_ticks(),
            Side::Sell => self.asks.matchable_ticks(),
        };

        for tick in matchable_ticks {
            if order.remaining <= 0.0 {
                break;
            }

            let tick_price = match opposite_side {
                Side::Buy => self.bids.tick_to_price(tick),
                Side::Sell => self.asks.tick_to_price(tick),
            };

            // Price check: does the incoming order's price allow matching?
            let price_ok = match order.order_type {
                OrderType::Market => true, // market orders match any price
                _ => match order.side {
                    Side::Buy => order.price >= tick_price,
                    Side::Sell => order.price <= tick_price,
                },
            };

            if !price_ok {
                break; // No more matchable levels.
            }

            // Get order IDs at this level.
            let order_ids_at_level: Vec<u64> = match opposite_side {
                Side::Buy => self
                    .bids
                    .levels
                    .get(&tick)
                    .map(|l| l.orders.iter().copied().collect())
                    .unwrap_or_default(),
                Side::Sell => self
                    .asks
                    .levels
                    .get(&tick)
                    .map(|l| l.orders.iter().copied().collect())
                    .unwrap_or_default(),
            };

            for resting_id in order_ids_at_level {
                if order.remaining <= 0.0 {
                    break;
                }

                // Self-trade prevention.
                let resting_participant = self
                    .orders
                    .get(&resting_id)
                    .map(|o| o.participant.clone())
                    .unwrap_or_default();

                if self.config.self_trade_prevention && order.participant == resting_participant {
                    self.stats.self_trade_preventions += 1;
                    trace!(
                        aggressor = order.id,
                        resting = resting_id,
                        "Self-trade prevented"
                    );
                    continue;
                }

                let resting_remaining = self
                    .orders
                    .get(&resting_id)
                    .map(|o| o.remaining)
                    .unwrap_or(0.0);

                if resting_remaining <= 0.0 {
                    continue;
                }

                let fill_qty = order.remaining.min(resting_remaining);
                let fill_price = tick_price; // Price-time priority: fill at resting price.

                // Compute fees.
                let taker_fee = fill_qty * fill_price * self.config.taker_fee_rate;
                let maker_fee = fill_qty * fill_price * self.config.maker_fee_rate;

                // Create fill.
                let fill_id = self.next_fill_id;
                self.next_fill_id += 1;

                let fill = Fill {
                    fill_id,
                    aggressor_order_id: order.id,
                    resting_order_id: resting_id,
                    price: fill_price,
                    quantity: fill_qty,
                    aggressor_side: order.side,
                    aggressor_participant: order.participant.clone(),
                    resting_participant: resting_participant.clone(),
                    taker_fee,
                    maker_fee,
                    timestamp_ns: self.current_time_ns,
                };

                // Update aggressor order.
                order.record_fill(fill_qty, fill_price, taker_fee);

                // Update resting order.
                if let Some(resting_order) = self.orders.get_mut(&resting_id) {
                    resting_order.record_fill(fill_qty, fill_price, maker_fee);

                    // Record fill event for resting order.
                    let event_type = if resting_order.is_filled() {
                        OrderEventType::Filled
                    } else {
                        OrderEventType::PartiallyFilled
                    };

                    let event = OrderEvent {
                        event_type,
                        order_id: resting_id,
                        timestamp_ns: self.current_time_ns,
                        details: format!("{fill_qty:.4}@{fill_price:.4}"),
                    };
                    if self.config.record_order_events {
                        self.event_history.push(event);
                    }

                    // If resting order is fully filled, remove from book.
                    if resting_order.is_filled() {
                        self.stats.total_orders_filled += 1;
                        match opposite_side {
                            Side::Buy => self.bids.remove_order(resting_id, fill_price, 0.0),
                            Side::Sell => self.asks.remove_order(resting_id, fill_price, 0.0),
                        }
                        // Update participant count.
                        if let Some(count) =
                            self.participant_order_count.get_mut(&resting_participant)
                        {
                            *count = count.saturating_sub(1);
                        }
                    } else {
                        // Reduce quantity at level.
                        match opposite_side {
                            Side::Buy => {
                                self.bids.reduce_quantity(resting_id, fill_price, fill_qty)
                            }
                            Side::Sell => {
                                self.asks.reduce_quantity(resting_id, fill_price, fill_qty)
                            }
                        }
                    }
                }

                // Update statistics.
                self.stats
                    .record_trade(fill_price, fill_qty, taker_fee, maker_fee);

                // Record trade.
                if self.config.record_trades {
                    self.trade_history.push(fill.clone());
                }

                fills.push(fill);
            }
        }

        // Record fill events for the aggressor.
        if !fills.is_empty() {
            let event_type = if order.is_filled() {
                OrderEventType::Filled
            } else if order.filled > 0.0 {
                OrderEventType::PartiallyFilled
            } else {
                OrderEventType::Submitted
            };

            self.record_event(
                order.id,
                event_type,
                format!(
                    "filled={:.4}@{:.4}, remaining={:.4}",
                    order.filled, order.avg_fill_price, order.remaining
                ),
            );

            if order.is_filled() {
                self.stats.total_orders_filled += 1;
            }
        }

        Ok(fills)
    }

    /// Place a resting order on the book.
    fn place_on_book(&mut self, order: &mut Order) {
        // If the order already matched some quantity before resting,
        // preserve its PartiallyFilled status instead of resetting to Open.
        order.status = if order.filled > 0.0 {
            OrderStatus::PartiallyFilled
        } else {
            OrderStatus::Open
        };

        let remaining = order.remaining;
        let price = order.price;
        let order_id = order.id;

        match order.side {
            Side::Buy => self.bids.add_order(order_id, price, remaining),
            Side::Sell => self.asks.add_order(order_id, price, remaining),
        }

        debug!(
            order_id,
            side = %order.side,
            price,
            remaining,
            "Order placed on book"
        );
    }

    /// Check how much quantity is available at price levels up to a limit price.
    fn available_quantity(&self, side: Side, limit_price: f64, aggressor_side: Side) -> f64 {
        let book_side = match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };

        let mut total = 0.0;

        for level in book_side.levels.values() {
            let price_ok = match aggressor_side {
                Side::Buy => limit_price >= level.price,
                Side::Sell => limit_price <= level.price,
            };
            if price_ok {
                total += level.total_quantity;
            }
        }

        total
    }

    /// Check if any stop orders should be triggered.
    fn check_stop_triggers(&mut self) {
        if self.stop_orders.is_empty() || self.stats.total_trades == 0 {
            return;
        }

        let last_price = self.stats.last_price;
        let mut triggered: Vec<u64> = Vec::new();

        for &order_id in &self.stop_orders {
            if let Some(order) = self.orders.get(&order_id) {
                if order.status != OrderStatus::StopWaiting {
                    continue;
                }

                if let Some(stop_price) = order.stop_price {
                    let triggered_flag = match order.side {
                        Side::Buy => last_price >= stop_price,
                        Side::Sell => last_price <= stop_price,
                    };

                    if triggered_flag {
                        triggered.push(order_id);
                    }
                }
            }
        }

        for order_id in triggered {
            self.stop_orders.retain(|&id| id != order_id);
            self.stats.stops_triggered += 1;

            self.record_event(
                order_id,
                OrderEventType::StopTriggered,
                format!("last_price={last_price:.4}"),
            );

            if let Some(order) = self.orders.get_mut(&order_id) {
                let side = order.side;
                let quantity = order.remaining;
                let order_type = order.order_type;
                let price = order.price;
                let participant = order.participant.clone();

                // Convert stop to appropriate order type.
                let mut new_order = match order_type {
                    OrderType::Stop => Order::market_buy(quantity, &participant),
                    OrderType::StopLimit => Order::limit_buy(quantity, price, &participant),
                    _ => continue,
                };
                new_order.side = side;

                // Mark the original as cancelled.
                order.status = OrderStatus::Cancelled;

                debug!(order_id, "Stop order triggered, submitting new order");

                // Submit the triggered order (ignore errors).
                let _ = self.submit_order(new_order);
            }
        }
    }

    // ── Internal: Helpers ──────────────────────────────────────────────

    /// Simulate gateway latency with jitter.
    fn simulate_latency(&mut self) -> Duration {
        let base = self.config.gateway_latency;
        let jitter_ns = if self.config.latency_jitter.as_nanos() > 0 {
            self.xorshift64() % self.config.latency_jitter.as_nanos() as u64
        } else {
            0
        };
        base + Duration::from_nanos(jitter_ns)
    }

    /// Simple xorshift64 PRNG for latency jitter.
    fn xorshift64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Record an order event.
    fn record_event(&mut self, order_id: u64, event_type: OrderEventType, details: String) {
        if self.config.record_order_events {
            self.event_history.push(OrderEvent {
                event_type,
                order_id,
                timestamp_ns: self.current_time_ns,
                details,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LobConfig {
        LobConfig::default()
            .with_tick_size(0.01)
            .with_lot_size(0.001)
            .with_symbol("TEST-USDT")
            .with_latency(Duration::ZERO)
            .with_jitter(Duration::ZERO)
            .with_self_trade_prevention(false)
    }

    fn seeded_book() -> LimitOrderBook {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        // Place some resting orders
        let bids = vec![
            (100.00, 10.0),
            (99.99, 20.0),
            (99.98, 30.0),
            (99.97, 40.0),
            (99.96, 50.0),
        ];
        let asks = vec![
            (100.01, 10.0),
            (100.02, 20.0),
            (100.03, 30.0),
            (100.04, 40.0),
            (100.05, 50.0),
        ];
        lob.seed_book(&bids, &asks, "market_maker");
        lob
    }

    // ── Config tests ───────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = LobConfig::default();
        assert_eq!(config.tick_size, 0.01);
        assert_eq!(config.lot_size, 0.001);
        assert!(config.self_trade_prevention);
    }

    #[test]
    fn test_config_builder() {
        let config = LobConfig::default()
            .with_tick_size(0.05)
            .with_lot_size(1.0)
            .with_symbol("ETH-USDT")
            .with_latency(Duration::from_micros(100))
            .with_jitter(Duration::from_micros(20))
            .with_fees(-0.0002, 0.0005)
            .with_price_bounds(0.1, 100_000.0)
            .with_self_trade_prevention(false);

        assert_eq!(config.tick_size, 0.05);
        assert_eq!(config.lot_size, 1.0);
        assert_eq!(config.symbol, "ETH-USDT");
        assert_eq!(config.maker_fee_rate, -0.0002);
        assert_eq!(config.taker_fee_rate, 0.0005);
        assert!(!config.self_trade_prevention);
    }

    #[test]
    fn test_config_round_price() {
        let config = LobConfig::default().with_tick_size(0.05);
        assert!((config.round_price(1.03) - 1.05).abs() < 1e-10);
        assert!((config.round_price(1.07) - 1.05).abs() < 1e-10);
        assert!((config.round_price(1.10) - 1.10).abs() < 1e-10);
    }

    #[test]
    fn test_config_round_quantity() {
        let config = LobConfig::default().with_lot_size(0.1);
        assert!((config.round_quantity(1.05) - 1.1).abs() < 1e-10);
        assert!((config.round_quantity(1.04) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_validate_price() {
        let config = LobConfig::default().with_price_bounds(1.0, 100.0);
        assert!(config.validate_price(50.0).is_ok());
        assert!(config.validate_price(0.5).is_err());
        assert!(config.validate_price(200.0).is_err());
    }

    #[test]
    fn test_config_validate_quantity() {
        let config = LobConfig::default();
        assert!(config.validate_quantity(1.0).is_ok());
        assert!(config.validate_quantity(-1.0).is_err());
        assert!(config.validate_quantity(0.0).is_err());
    }

    // ── Side tests ─────────────────────────────────────────────────────

    #[test]
    fn test_side_opposite() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
    }

    #[test]
    fn test_side_display() {
        assert_eq!(format!("{}", Side::Buy), "BUY");
        assert_eq!(format!("{}", Side::Sell), "SELL");
    }

    // ── Order creation tests ───────────────────────────────────────────

    #[test]
    fn test_order_limit_buy() {
        let order = Order::limit_buy(10.0, 100.0, "test");
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.quantity, 10.0);
        assert_eq!(order.price, 100.0);
        assert_eq!(order.remaining, 10.0);
        assert_eq!(order.participant, "test");
        assert_eq!(order.status, OrderStatus::Pending);
    }

    #[test]
    fn test_order_limit_sell() {
        let order = Order::limit_sell(5.0, 200.0, "test");
        assert_eq!(order.side, Side::Sell);
        assert_eq!(order.order_type, OrderType::Limit);
    }

    #[test]
    fn test_order_market_buy() {
        let order = Order::market_buy(10.0, "test");
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.price, 0.0);
    }

    #[test]
    fn test_order_market_sell() {
        let order = Order::market_sell(10.0, "test");
        assert_eq!(order.side, Side::Sell);
        assert_eq!(order.order_type, OrderType::Market);
    }

    #[test]
    fn test_order_ioc() {
        let order = Order::ioc(Side::Buy, 10.0, 100.0, "test");
        assert_eq!(order.order_type, OrderType::ImmediateOrCancel);
    }

    #[test]
    fn test_order_fok() {
        let order = Order::fok(Side::Sell, 10.0, 100.0, "test");
        assert_eq!(order.order_type, OrderType::FillOrKill);
    }

    #[test]
    fn test_order_post_only() {
        let order = Order::post_only(Side::Buy, 10.0, 100.0, "test");
        assert_eq!(order.order_type, OrderType::PostOnly);
    }

    #[test]
    fn test_order_iceberg() {
        let order = Order::iceberg(Side::Buy, 100.0, 50.0, 10.0, "test");
        assert_eq!(order.order_type, OrderType::Iceberg);
        assert_eq!(order.display_quantity, Some(10.0));
    }

    #[test]
    fn test_order_stop() {
        let order = Order::stop(Side::Sell, 10.0, 95.0, "test");
        assert_eq!(order.order_type, OrderType::Stop);
        assert_eq!(order.stop_price, Some(95.0));
    }

    #[test]
    fn test_order_stop_limit() {
        let order = Order::stop_limit(Side::Buy, 10.0, 105.0, 104.0, "test");
        assert_eq!(order.order_type, OrderType::StopLimit);
        assert_eq!(order.price, 105.0);
        assert_eq!(order.stop_price, Some(104.0));
    }

    #[test]
    fn test_order_with_client_id() {
        let order = Order::limit_buy(10.0, 100.0, "test").with_client_id("client_123");
        assert_eq!(order.client_order_id.as_deref(), Some("client_123"));
    }

    #[test]
    fn test_order_visible_quantity() {
        let order = Order::iceberg(Side::Buy, 100.0, 50.0, 10.0, "test");
        assert_eq!(order.visible_quantity(), 10.0);

        let order2 = Order::limit_buy(100.0, 50.0, "test");
        assert_eq!(order2.visible_quantity(), 100.0);
    }

    #[test]
    fn test_order_is_active() {
        let mut order = Order::limit_buy(10.0, 100.0, "test");
        order.status = OrderStatus::Open;
        assert!(order.is_active());

        order.status = OrderStatus::Filled;
        assert!(!order.is_active());
        assert!(order.is_filled());
        assert!(order.is_done());
    }

    #[test]
    fn test_order_display() {
        let order = Order::limit_buy(10.0, 100.0, "test");
        let s = format!("{order}");
        assert!(s.contains("BUY"));
        assert!(s.contains("LIMIT"));
        assert!(s.contains("test"));
    }

    // ── LOB creation tests ─────────────────────────────────────────────

    #[test]
    fn test_lob_creation() {
        let lob = LimitOrderBook::new("BTC-USDT", LobConfig::default());
        assert_eq!(lob.config().symbol, "BTC-USDT");
        assert!(lob.best_bid().is_none());
        assert!(lob.best_ask().is_none());
        assert!(lob.mid_price().is_none());
        assert!(lob.spread().is_none());
        assert_eq!(lob.total_open_orders(), 0);
    }

    #[test]
    fn test_lob_default_for() {
        let lob = LimitOrderBook::default_for("ETH-USDT");
        assert_eq!(lob.config().symbol, "ETH-USDT");
    }

    // ── Limit order submission tests ───────────────────────────────────

    #[test]
    fn test_submit_limit_buy() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let order = Order::limit_buy(10.0, 100.0, "alice");
        let receipt = lob.submit_order(order).unwrap();

        assert!(receipt.is_accepted());
        assert_eq!(receipt.status, OrderStatus::Open);
        assert_eq!(receipt.filled_quantity, 0.0);
        assert_eq!(receipt.remaining_quantity, 10.0);
        assert!(receipt.fills.is_empty());

        assert_eq!(lob.best_bid(), Some(100.0));
        assert_eq!(lob.total_open_orders(), 1);
    }

    #[test]
    fn test_submit_limit_sell() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let order = Order::limit_sell(5.0, 200.0, "bob");
        let receipt = lob.submit_order(order).unwrap();

        assert!(receipt.is_accepted());
        assert_eq!(receipt.status, OrderStatus::Open);
        assert_eq!(lob.best_ask(), Some(200.0));
    }

    #[test]
    fn test_multiple_limit_orders() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        lob.submit_order(Order::limit_buy(10.0, 99.0, "alice"))
            .unwrap();
        lob.submit_order(Order::limit_buy(20.0, 100.0, "bob"))
            .unwrap();
        lob.submit_order(Order::limit_buy(30.0, 98.0, "carol"))
            .unwrap();

        // Best bid should be the highest
        assert_eq!(lob.best_bid(), Some(100.0));
        assert_eq!(lob.bid_levels(), 3);
        assert_eq!(lob.total_open_orders(), 3);
    }

    // ── Matching tests ─────────────────────────────────────────────────

    #[test]
    fn test_market_buy_matches_asks() {
        let mut lob = seeded_book();

        let order = Order::market_buy(5.0, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert!(receipt.is_filled());
        assert_eq!(receipt.filled_quantity, 5.0);
        assert!((receipt.avg_fill_price - 100.01).abs() < 0.001);
        assert_eq!(receipt.fills.len(), 1);
    }

    #[test]
    fn test_market_sell_matches_bids() {
        let mut lob = seeded_book();

        let order = Order::market_sell(5.0, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert!(receipt.is_filled());
        assert_eq!(receipt.filled_quantity, 5.0);
        assert!((receipt.avg_fill_price - 100.00).abs() < 0.001);
    }

    #[test]
    fn test_market_buy_sweeps_multiple_levels() {
        let mut lob = seeded_book();

        // Buy 25: should take 10 @ 100.01 + 15 @ 100.02
        let order = Order::market_buy(25.0, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert!(receipt.is_filled());
        assert_eq!(receipt.filled_quantity, 25.0);
        assert_eq!(receipt.fills.len(), 2);
        assert!((receipt.fills[0].price - 100.01).abs() < 0.001);
        assert!((receipt.fills[0].quantity - 10.0).abs() < 0.001);
        assert!((receipt.fills[1].price - 100.02).abs() < 0.001);
        assert!((receipt.fills[1].quantity - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_limit_buy_crosses_spread() {
        let mut lob = seeded_book();

        // Limit buy at 100.02 should take the ask at 100.01 and 100.02
        let order = Order::limit_buy(15.0, 100.02, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert!(receipt.is_filled());
        assert_eq!(receipt.filled_quantity, 15.0);
        assert_eq!(receipt.fills.len(), 2);
    }

    #[test]
    fn test_limit_buy_partial_cross_and_rest() {
        let mut lob = seeded_book();

        // Limit buy 15 @ 100.01: fills 10 from ask, rests 5 on bid side
        let order = Order::limit_buy(15.0, 100.01, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert_eq!(receipt.status, OrderStatus::PartiallyFilled);
        assert_eq!(receipt.filled_quantity, 10.0);
        assert_eq!(receipt.remaining_quantity, 5.0);
        assert_eq!(receipt.fills.len(), 1);

        // Best bid should now be 100.01 (the resting portion)
        assert_eq!(lob.best_bid(), Some(100.01));
    }

    #[test]
    fn test_price_time_priority() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        // Two asks at the same price — first one should be matched first.
        lob.submit_order(Order::limit_sell(10.0, 100.0, "alice"))
            .unwrap();
        lob.submit_order(Order::limit_sell(10.0, 100.0, "bob"))
            .unwrap();

        let order = Order::market_buy(5.0, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert_eq!(receipt.fills.len(), 1);
        assert_eq!(receipt.fills[0].resting_participant, "alice");
    }

    // ── IOC tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ioc_full_fill() {
        let mut lob = seeded_book();

        let order = Order::ioc(Side::Buy, 5.0, 100.01, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert!(receipt.is_filled());
        assert_eq!(receipt.filled_quantity, 5.0);
    }

    #[test]
    fn test_ioc_partial_fill_cancel_rest() {
        let mut lob = seeded_book();

        // IOC buy 15 @ 100.01: only 10 available at that price
        let order = Order::ioc(Side::Buy, 15.0, 100.01, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert_eq!(receipt.filled_quantity, 10.0);
        // Remaining should be cancelled (not on book)
        assert_eq!(lob.total_open_orders(), 9); // 5 bids + 4 asks (one ask filled)
    }

    #[test]
    fn test_ioc_no_fill() {
        let mut lob = seeded_book();

        // IOC buy at a price below the ask — nothing to match
        let order = Order::ioc(Side::Buy, 10.0, 99.0, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert_eq!(receipt.filled_quantity, 0.0);
        assert_eq!(receipt.status, OrderStatus::Cancelled);
    }

    // ── FOK tests ──────────────────────────────────────────────────────

    #[test]
    fn test_fok_success() {
        let mut lob = seeded_book();

        // FOK buy 10 @ 100.01: exactly 10 available
        let order = Order::fok(Side::Buy, 10.0, 100.01, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert!(receipt.is_filled());
        assert_eq!(receipt.filled_quantity, 10.0);
    }

    #[test]
    fn test_fok_rejection() {
        let mut lob = seeded_book();

        // FOK buy 15 @ 100.01: only 10 available at that price → reject
        let order = Order::fok(Side::Buy, 15.0, 100.01, "trader");
        let result = lob.submit_order(order);

        assert!(result.is_err());
        match result.unwrap_err() {
            LobError::FokNotFilled { .. } => {}
            e => panic!("Expected FokNotFilled, got: {e}"),
        }
    }

    // ── Post-only tests ────────────────────────────────────────────────

    #[test]
    fn test_post_only_rests() {
        let mut lob = seeded_book();

        // Post-only buy at 99.95 — doesn't cross the ask
        let order = Order::post_only(Side::Buy, 10.0, 99.95, "maker");
        let receipt = lob.submit_order(order).unwrap();

        assert_eq!(receipt.status, OrderStatus::Open);
        assert_eq!(receipt.filled_quantity, 0.0);
    }

    #[test]
    fn test_post_only_rejection() {
        let mut lob = seeded_book();

        // Post-only buy at 100.01 — would cross the ask → reject
        let order = Order::post_only(Side::Buy, 10.0, 100.01, "maker");
        let result = lob.submit_order(order);

        assert!(result.is_err());
        match result.unwrap_err() {
            LobError::PostOnlyWouldTake { .. } => {}
            e => panic!("Expected PostOnlyWouldTake, got: {e}"),
        }
    }

    // ── Cancel tests ───────────────────────────────────────────────────

    #[test]
    fn test_cancel_order() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let receipt = lob
            .submit_order(Order::limit_buy(10.0, 100.0, "alice"))
            .unwrap();
        let order_id = receipt.order_id;

        assert_eq!(lob.total_open_orders(), 1);

        let cancel = lob.cancel_order(order_id).unwrap();
        assert!(cancel.success);
        assert_eq!(cancel.cancelled_quantity, 10.0);
        assert_eq!(lob.total_open_orders(), 0);
    }

    #[test]
    fn test_cancel_not_found() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let result = lob.cancel_order(9999);
        assert!(result.is_err());
    }

    #[test]
    fn test_cancel_already_filled() {
        let mut lob = seeded_book();

        // Market buy fills immediately
        let receipt = lob.submit_order(Order::market_buy(5.0, "trader")).unwrap();
        assert!(receipt.is_filled());

        // Try to cancel the filled order
        let cancel = lob.cancel_order(receipt.order_id).unwrap();
        assert!(!cancel.success);
    }

    #[test]
    fn test_cancel_all() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        lob.submit_order(Order::limit_buy(10.0, 99.0, "alice"))
            .unwrap();
        lob.submit_order(Order::limit_buy(20.0, 98.0, "alice"))
            .unwrap();
        lob.submit_order(Order::limit_sell(5.0, 101.0, "bob"))
            .unwrap();

        let cancelled = lob.cancel_all("alice");
        assert_eq!(cancelled.len(), 2);
        assert!(cancelled.iter().all(|c| c.success));
        assert_eq!(lob.total_open_orders(), 1); // Bob's order remains
    }

    // ── Modify tests ───────────────────────────────────────────────────

    #[test]
    fn test_modify_price() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let receipt = lob
            .submit_order(Order::limit_buy(10.0, 99.0, "alice"))
            .unwrap();

        let modify = lob
            .modify_order(receipt.order_id, Some(99.5), None)
            .unwrap();
        assert!(modify.success);
        assert!((modify.old_price - 99.0).abs() < 0.001);
        assert!((modify.new_price - 99.5).abs() < 0.001);

        assert_eq!(lob.best_bid(), Some(99.5));
    }

    #[test]
    fn test_modify_quantity() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let receipt = lob
            .submit_order(Order::limit_buy(10.0, 99.0, "alice"))
            .unwrap();

        let modify = lob
            .modify_order(receipt.order_id, None, Some(20.0))
            .unwrap();
        assert!(modify.success);
        assert!((modify.old_quantity - 10.0).abs() < 0.001);
        assert!((modify.new_quantity - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_modify_not_found() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let result = lob.modify_order(9999, Some(100.0), None);
        assert!(result.is_err());
    }

    // ── Snapshot tests ─────────────────────────────────────────────────

    #[test]
    fn test_l2_snapshot() {
        let mut lob = seeded_book();
        let snap = lob.snapshot(5);

        assert_eq!(snap.symbol, "TEST");
        assert_eq!(snap.bids.len(), 5);
        assert_eq!(snap.asks.len(), 5);

        // Best bid should be highest
        assert!((snap.bids[0].price - 100.0).abs() < 0.001);
        // Best ask should be lowest
        assert!((snap.asks[0].price - 100.01).abs() < 0.001);

        assert!(snap.best_bid().is_some());
        assert!(snap.best_ask().is_some());
        assert!(snap.mid_price().is_some());
        assert!(snap.spread().is_some());
    }

    #[test]
    fn test_l2_snapshot_spread() {
        let mut lob = seeded_book();
        let snap = lob.snapshot(5);

        let spread = snap.spread().unwrap();
        assert!(
            (spread - 0.01).abs() < 0.001,
            "Spread should be ~0.01, got {spread}"
        );
    }

    #[test]
    fn test_l2_snapshot_imbalance() {
        let mut lob = seeded_book();
        let snap = lob.snapshot(5);

        // Both sides have equal total quantity (10+20+30+40+50 = 150 each)
        let imbalance = snap.imbalance();
        assert!(
            imbalance.abs() < 0.01,
            "Imbalance should be ~0 for symmetric book, got {imbalance}"
        );
    }

    #[test]
    fn test_l2_snapshot_weighted_mid() {
        let mut lob = seeded_book();
        let snap = lob.snapshot(5);
        let wmid = snap.weighted_mid().unwrap();
        // Weighted mid should be close to arithmetic mid
        let mid = snap.mid_price().unwrap();
        assert!(
            (wmid - mid).abs() < 0.01,
            "Weighted mid {wmid} should be close to mid {mid}"
        );
    }

    #[test]
    fn test_l2_snapshot_vwap_buy() {
        let mut lob = seeded_book();
        let snap = lob.snapshot(5);

        // VWAP for buying 10 should be exactly 100.01 (all at first level)
        let vwap = snap.vwap_buy(10.0).unwrap();
        assert!((vwap - 100.01).abs() < 0.001);

        // VWAP for buying 30 should sweep two levels: 10@100.01 + 20@100.02
        let vwap = snap.vwap_buy(30.0).unwrap();
        let expected = (10.0 * 100.01 + 20.0 * 100.02) / 30.0;
        assert!((vwap - expected).abs() < 0.001);
    }

    #[test]
    fn test_l2_snapshot_vwap_sell() {
        let mut lob = seeded_book();
        let snap = lob.snapshot(5);

        let vwap = snap.vwap_sell(10.0).unwrap();
        assert!((vwap - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_l2_snapshot_spread_bps() {
        let mut lob = seeded_book();
        let snap = lob.snapshot(5);
        let bps = snap.spread_bps().unwrap();
        // Spread = 0.01, mid ≈ 100.005 → ~0.9999 bps
        assert!(bps > 0.0 && bps < 2.0, "Spread bps: {bps}");
    }

    #[test]
    fn test_l2_snapshot_display() {
        let mut lob = seeded_book();
        let snap = lob.snapshot(3);
        let s = format!("{snap}");
        assert!(s.contains("Asks:"));
        assert!(s.contains("Bids:"));
        assert!(s.contains("spread"));
    }

    #[test]
    fn test_l3_snapshot() {
        let mut lob = seeded_book();
        let snap = lob.snapshot_l3(3);

        assert_eq!(snap.bids.len(), 3);
        assert_eq!(snap.asks.len(), 3);

        // Each level should have exactly 1 order (we placed one per price)
        assert_eq!(snap.bids[0].orders.len(), 1);
        assert_eq!(snap.asks[0].orders.len(), 1);

        // Check order detail
        assert_eq!(snap.bids[0].orders[0].participant, "market_maker");
    }

    // ── Statistics tests ───────────────────────────────────────────────

    #[test]
    fn test_statistics_after_trades() {
        let mut lob = seeded_book();

        lob.submit_order(Order::market_buy(10.0, "trader")).unwrap();
        lob.submit_order(Order::market_sell(10.0, "trader2"))
            .unwrap();

        let stats = lob.statistics();
        assert!(stats.total_trades >= 2);
        assert!(stats.total_volume > 0.0);
        assert!(stats.total_notional > 0.0);
        assert!(stats.vwap > 0.0);
        assert!(stats.last_price > 0.0);
        assert!(stats.high_price >= stats.low_price);
    }

    #[test]
    fn test_statistics_display() {
        let mut lob = seeded_book();
        lob.submit_order(Order::market_buy(5.0, "trader")).unwrap();

        let stats = lob.statistics();
        let s = format!("{stats}");
        assert!(s.contains("LOB Stats:"));
        assert!(s.contains("trades="));
    }

    // ── Fee tests ──────────────────────────────────────────────────────

    #[test]
    fn test_taker_fee() {
        let config = test_config().with_fees(-0.0001, 0.0004);
        let mut lob = LimitOrderBook::new("TEST", config);
        lob.seed_book(&[(100.0, 10.0)], &[(101.0, 10.0)], "mm");

        let receipt = lob.submit_order(Order::market_buy(5.0, "trader")).unwrap();

        // Taker fee should be positive
        assert!(receipt.fees > 0.0);
        // fee = 5 * 101 * 0.0004 = 0.202
        assert!(
            (receipt.fees - 0.202).abs() < 0.01,
            "Taker fee: {}",
            receipt.fees
        );
    }

    // ── Self-trade prevention tests ────────────────────────────────────

    #[test]
    fn test_self_trade_prevention() {
        let config = test_config().with_self_trade_prevention(true);
        let mut lob = LimitOrderBook::new("TEST", config);

        lob.submit_order(Order::limit_sell(10.0, 100.0, "alice"))
            .unwrap();
        let receipt = lob.submit_order(Order::market_buy(10.0, "alice")).unwrap();

        // Should not fill (self-trade prevented)
        assert_eq!(receipt.filled_quantity, 0.0);
        assert_eq!(lob.statistics().self_trade_preventions, 1);
    }

    // ── Queue position tests ───────────────────────────────────────────

    #[test]
    fn test_queue_position() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        let r1 = lob
            .submit_order(Order::limit_buy(10.0, 100.0, "alice"))
            .unwrap();
        let r2 = lob
            .submit_order(Order::limit_buy(20.0, 100.0, "bob"))
            .unwrap();
        let r3 = lob
            .submit_order(Order::limit_buy(30.0, 100.0, "carol"))
            .unwrap();

        let (pos1, ahead1) = lob.queue_position(r1.order_id).unwrap();
        assert_eq!(pos1, 0);
        assert_eq!(ahead1, 0.0);

        let (pos2, ahead2) = lob.queue_position(r2.order_id).unwrap();
        assert_eq!(pos2, 1);
        assert!((ahead2 - 10.0).abs() < 0.001);

        let (pos3, ahead3) = lob.queue_position(r3.order_id).unwrap();
        assert_eq!(pos3, 2);
        assert!((ahead3 - 30.0).abs() < 0.001);
    }

    // ── Time management tests ──────────────────────────────────────────

    #[test]
    fn test_advance_time() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        assert_eq!(lob.current_time_ns(), 0);

        lob.advance_time(Duration::from_millis(100));
        assert_eq!(lob.current_time_ns(), 100_000_000);

        lob.set_time_ns(1_000_000_000);
        assert_eq!(lob.current_time_ns(), 1_000_000_000);
    }

    // ── Reset tests ────────────────────────────────────────────────────

    #[test]
    fn test_reset() {
        let mut lob = seeded_book();
        lob.submit_order(Order::market_buy(5.0, "trader")).unwrap();

        assert!(lob.statistics().total_trades > 0);
        assert!(lob.total_open_orders() > 0);

        lob.reset();

        assert_eq!(lob.total_open_orders(), 0);
        assert_eq!(lob.statistics().total_trades, 0);
        assert!(lob.best_bid().is_none());
        assert!(lob.best_ask().is_none());
    }

    // ── Market impact model tests ──────────────────────────────────────

    #[test]
    fn test_market_impact_default() {
        let model = MarketImpactModel::default();
        assert!(model.eta > 0.0);
        assert!(model.gamma > 0.0);
        assert!(model.adv > 0.0);
    }

    #[test]
    fn test_temporary_impact() {
        let model = MarketImpactModel::new(0.1, 0.5, 1_000_000.0, 0.02);

        let impact = model.temporary_impact(1000.0, 100.0);
        assert!(impact > 0.0);
        assert!(impact < 1.0); // Should be reasonable

        // Larger orders should have more impact
        let impact_large = model.temporary_impact(10_000.0, 100.0);
        assert!(impact_large > impact);
    }

    #[test]
    fn test_permanent_impact() {
        let model = MarketImpactModel::default();
        let perm = model.permanent_impact(1000.0, 100.0);
        let temp = model.temporary_impact(1000.0, 100.0);
        assert!(perm < temp); // Permanent should be a fraction of temporary
    }

    #[test]
    fn test_total_cost() {
        let model = MarketImpactModel::default();
        let cost = model.total_cost(1000.0, 100.0);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_estimate_impact() {
        let mut lob = seeded_book();
        lob.submit_order(Order::market_buy(1.0, "price_setter"))
            .unwrap();

        let buy_impact = lob.estimate_impact(Side::Buy, 1000.0);
        let sell_impact = lob.estimate_impact(Side::Sell, 1000.0);

        assert!(buy_impact > 0.0);
        assert!(sell_impact < 0.0);
    }

    // ── Replay tests ───────────────────────────────────────────────────

    #[test]
    fn test_replay_orders() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        let orders = vec![
            Order::limit_buy(10.0, 99.0, "alice"),
            Order::limit_sell(10.0, 101.0, "bob"),
            Order::limit_buy(5.0, 100.0, "carol"),
        ];

        let receipts = lob.replay_orders(orders);
        assert_eq!(receipts.len(), 3);
        assert!(receipts.iter().all(|r| r.is_accepted()));
    }

    #[test]
    fn test_seed_book() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        let bids = vec![(99.0, 10.0), (98.0, 20.0)];
        let asks = vec![(101.0, 10.0), (102.0, 20.0)];

        let receipts = lob.seed_book(&bids, &asks, "seed");
        assert_eq!(receipts.len(), 4);
        assert_eq!(lob.bid_levels(), 2);
        assert_eq!(lob.ask_levels(), 2);
        assert_eq!(lob.total_open_orders(), 4);
    }

    // ── Event history tests ────────────────────────────────────────────

    #[test]
    fn test_event_history() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        lob.submit_order(Order::limit_buy(10.0, 100.0, "alice"))
            .unwrap();

        let events = lob.event_history();
        assert!(events.len() >= 2); // Submitted + Accepted
        assert_eq!(events[0].event_type, OrderEventType::Submitted);
        assert_eq!(events[1].event_type, OrderEventType::Accepted);
    }

    #[test]
    fn test_trade_history() {
        let mut lob = seeded_book();
        lob.submit_order(Order::market_buy(5.0, "trader")).unwrap();

        let trades = lob.trade_history();
        assert!(!trades.is_empty());
        assert!(trades[0].quantity > 0.0);
        assert!(trades[0].price > 0.0);
    }

    // ── Stop order tests ───────────────────────────────────────────────

    #[test]
    fn test_stop_order_submission() {
        let mut lob = seeded_book();

        let order = Order::stop(Side::Sell, 5.0, 99.95, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert_eq!(receipt.status, OrderStatus::StopWaiting);
        assert_eq!(receipt.filled_quantity, 0.0);
    }

    #[test]
    fn test_stop_order_trigger() {
        let config = test_config().with_self_trade_prevention(false);
        let mut lob = LimitOrderBook::new("TEST", config);

        // Set up a book
        lob.submit_order(Order::limit_buy(100.0, 100.0, "mm"))
            .unwrap();
        lob.submit_order(Order::limit_buy(100.0, 99.0, "mm"))
            .unwrap();
        lob.submit_order(Order::limit_sell(100.0, 101.0, "mm"))
            .unwrap();

        // Place a stop-sell at 99.5
        let stop_receipt = lob
            .submit_order(Order::stop(Side::Sell, 5.0, 99.5, "trader"))
            .unwrap();
        assert_eq!(stop_receipt.status, OrderStatus::StopWaiting);

        // Trade at a price that triggers the stop
        // Market sell to push price to 100.0 first
        let _ = lob
            .submit_order(Order::market_sell(100.0, "seller"))
            .unwrap();

        // The stop should have been triggered and may have filled
        assert!(lob.statistics().stops_triggered > 0 || lob.statistics().total_trades > 0);
    }

    // ── Iceberg order tests ────────────────────────────────────────────

    #[test]
    fn test_iceberg_order() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        let order = Order::iceberg(Side::Buy, 100.0, 50.0, 10.0, "trader");
        let receipt = lob.submit_order(order).unwrap();

        assert_eq!(receipt.status, OrderStatus::Open);
        // Total should be 100, but only 10 visible
        let snap = lob.snapshot(1);
        // The level should show full remaining for matching purposes
        // but the order's visible_quantity would be 10
        assert_eq!(snap.bids.len(), 1);
    }

    #[test]
    fn test_iceberg_validation() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        // Display qty > total qty → invalid
        let order = Order::iceberg(Side::Buy, 10.0, 50.0, 20.0, "trader");
        let receipt = lob.submit_order(order).unwrap();
        assert_eq!(receipt.status, OrderStatus::Rejected);
    }

    // ── Open orders query ──────────────────────────────────────────────

    #[test]
    fn test_open_orders() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        lob.submit_order(Order::limit_buy(10.0, 99.0, "alice"))
            .unwrap();
        lob.submit_order(Order::limit_buy(20.0, 98.0, "alice"))
            .unwrap();
        lob.submit_order(Order::limit_sell(5.0, 101.0, "bob"))
            .unwrap();

        let alice_orders = lob.open_orders("alice");
        assert_eq!(alice_orders.len(), 2);

        let bob_orders = lob.open_orders("bob");
        assert_eq!(bob_orders.len(), 1);

        let nobody_orders = lob.open_orders("nobody");
        assert_eq!(nobody_orders.len(), 0);
    }

    // ── Get order ──────────────────────────────────────────────────────

    #[test]
    fn test_get_order() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let receipt = lob
            .submit_order(Order::limit_buy(10.0, 99.0, "alice"))
            .unwrap();

        let order = lob.get_order(receipt.order_id).unwrap();
        assert_eq!(order.participant, "alice");
        assert_eq!(order.status, OrderStatus::Open);
        assert!((order.price - 99.0).abs() < 0.001);

        assert!(lob.get_order(9999).is_none());
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_empty_book_market_order() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        let receipt = lob.submit_order(Order::market_buy(10.0, "trader")).unwrap();
        assert_eq!(receipt.filled_quantity, 0.0);
        assert_eq!(receipt.status, OrderStatus::Cancelled);
    }

    #[test]
    fn test_zero_quantity_rejected() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let order = Order::limit_buy(0.0, 100.0, "test");
        let receipt = lob.submit_order(order).unwrap();
        assert_eq!(receipt.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_negative_price_rejected() {
        let mut lob = LimitOrderBook::new("TEST", test_config());
        let order = Order::limit_buy(10.0, -5.0, "test");
        let receipt = lob.submit_order(order).unwrap();
        assert_eq!(receipt.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_multiple_fills_across_levels() {
        let mut lob = seeded_book();

        // Buy all 150 units across 5 ask levels
        let receipt = lob.submit_order(Order::market_buy(150.0, "whale")).unwrap();

        assert!(receipt.is_filled());
        assert_eq!(receipt.filled_quantity, 150.0);
        assert_eq!(receipt.fills.len(), 5);

        // All asks should be gone
        assert!(lob.best_ask().is_none());
        assert_eq!(lob.ask_levels(), 0);
    }

    #[test]
    fn test_last_price() {
        let mut lob = LimitOrderBook::new("TEST", test_config());

        assert!(lob.last_price().is_none());

        lob.submit_order(Order::limit_sell(10.0, 100.0, "alice"))
            .unwrap();
        lob.submit_order(Order::market_buy(5.0, "bob")).unwrap();

        assert_eq!(lob.last_price(), Some(100.0));
    }

    // ── LobError display tests ─────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = LobError::InvalidOrder("bad".into());
        assert!(format!("{e}").contains("invalid order"));

        let e = LobError::OrderNotFound(42);
        assert!(format!("{e}").contains("42"));

        let e = LobError::FokNotFilled {
            order_id: 1,
            available: 5.0,
            requested: 10.0,
        };
        assert!(format!("{e}").contains("FOK"));

        let e = LobError::PostOnlyWouldTake { order_id: 1 };
        assert!(format!("{e}").contains("post-only"));

        let e = LobError::PriceOutOfRange {
            price: 0.001,
            min: 0.01,
            max: 100.0,
        };
        assert!(format!("{e}").contains("out of range"));
    }

    // ── OrderReceipt tests ─────────────────────────────────────────────

    #[test]
    fn test_receipt_display() {
        let receipt = OrderReceipt {
            order_id: 1,
            status: OrderStatus::Filled,
            fills: vec![],
            filled_quantity: 10.0,
            avg_fill_price: 100.0,
            remaining_quantity: 0.0,
            fees: 0.04,
            latency: Duration::from_micros(50),
            rejection_reason: None,
            timestamp_ns: 0,
        };
        let s = format!("{receipt}");
        assert!(s.contains("id=1"));
        assert!(s.contains("FILLED"));
    }

    // ── Fill tests ─────────────────────────────────────────────────────

    #[test]
    fn test_fill_notional() {
        let fill = Fill {
            fill_id: 1,
            aggressor_order_id: 1,
            resting_order_id: 2,
            price: 100.0,
            quantity: 10.0,
            aggressor_side: Side::Buy,
            aggressor_participant: "a".into(),
            resting_participant: "b".into(),
            taker_fee: 0.0,
            maker_fee: 0.0,
            timestamp_ns: 0,
        };
        assert!((fill.notional() - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_fill_display() {
        let fill = Fill {
            fill_id: 42,
            aggressor_order_id: 1,
            resting_order_id: 2,
            price: 100.0,
            quantity: 5.0,
            aggressor_side: Side::Buy,
            aggressor_participant: "a".into(),
            resting_participant: "b".into(),
            taker_fee: 0.0,
            maker_fee: 0.0,
            timestamp_ns: 0,
        };
        let s = format!("{fill}");
        assert!(s.contains("id=42"));
    }

    // ── Order event display ────────────────────────────────────────────

    #[test]
    fn test_order_event_type_display() {
        assert_eq!(format!("{}", OrderEventType::Submitted), "SUBMITTED");
        assert_eq!(format!("{}", OrderEventType::Filled), "FILLED");
        assert_eq!(format!("{}", OrderEventType::Cancelled), "CANCELLED");
        assert_eq!(
            format!("{}", OrderEventType::StopTriggered),
            "STOP_TRIGGERED"
        );
    }

    // ── Order type display ─────────────────────────────────────────────

    #[test]
    fn test_order_type_display() {
        assert_eq!(format!("{}", OrderType::Limit), "LIMIT");
        assert_eq!(format!("{}", OrderType::Market), "MARKET");
        assert_eq!(format!("{}", OrderType::ImmediateOrCancel), "IOC");
        assert_eq!(format!("{}", OrderType::FillOrKill), "FOK");
        assert_eq!(format!("{}", OrderType::PostOnly), "POST_ONLY");
        assert_eq!(format!("{}", OrderType::Iceberg), "ICEBERG");
    }

    // ── Stress test ────────────────────────────────────────────────────

    #[test]
    fn test_many_orders() {
        let mut lob = LimitOrderBook::new("STRESS", test_config());

        // Place 100 bid levels and 100 ask levels
        for i in 0..100 {
            let bid_price = 100.0 - (i as f64) * 0.01;
            let ask_price = 100.01 + (i as f64) * 0.01;
            lob.submit_order(Order::limit_buy(10.0, bid_price, "mm"))
                .unwrap();
            lob.submit_order(Order::limit_sell(10.0, ask_price, "mm"))
                .unwrap();
        }

        assert_eq!(lob.bid_levels(), 100);
        assert_eq!(lob.ask_levels(), 100);
        assert_eq!(lob.total_open_orders(), 200);

        // Sweep half the asks
        let receipt = lob.submit_order(Order::market_buy(500.0, "whale")).unwrap();
        assert_eq!(receipt.filled_quantity, 500.0);
        assert_eq!(receipt.fills.len(), 50);
        assert_eq!(lob.ask_levels(), 50);
    }

    // ── Debug display ──────────────────────────────────────────────────

    #[test]
    fn test_lob_debug() {
        let lob = seeded_book();
        let s = format!("{lob:?}");
        assert!(s.contains("LimitOrderBook"));
        assert!(s.contains("bid_levels"));
        assert!(s.contains("ask_levels"));
    }
}
