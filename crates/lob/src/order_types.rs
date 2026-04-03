//! # Order Types
//!
//! Defines all order types, sides, time-in-force policies, and order status
//! tracking for the LOB simulator.
//!
//! # Supported Order Types
//!
//! | Type        | Description                                              |
//! |-------------|----------------------------------------------------------|
//! | Market      | Execute immediately at best available price               |
//! | Limit       | Execute at specified price or better                      |
//! | StopMarket  | Becomes market order when stop price is reached           |
//! | StopLimit   | Becomes limit order when stop price is reached            |
//! | Iceberg     | Limit order with hidden quantity (display qty < total)    |
//! | PostOnly    | Limit order that is rejected if it would cross the book   |
//!
//! # Time-in-Force
//!
//! | Policy | Description                                                |
//! |--------|------------------------------------------------------------|
//! | GTC    | Good 'Til Cancelled — rests on book until filled/cancelled |
//! | IOC    | Immediate or Cancel — fill what you can, cancel the rest   |
//! | FOK    | Fill or Kill — fill entirely or reject completely          |
//! | GTD    | Good 'Til Date — rests until a specific expiry time        |
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_lob::order_types::*;
//! use rust_decimal_macros::dec;
//!
//! // Market buy
//! let market = Order::market(Side::Buy, dec!(0.5));
//!
//! // Limit sell
//! let limit = Order::limit(Side::Sell, dec!(68000.0), dec!(1.0));
//!
//! // Stop-limit with IOC
//! let stop = Order::stop_limit(Side::Sell, dec!(65000.0), dec!(64900.0), dec!(2.0))
//!     .with_time_in_force(TimeInForce::IOC);
//!
//! // Iceberg order (show 0.1, total 5.0)
//! let iceberg = Order::iceberg(Side::Buy, dec!(67000.0), dec!(5.0), dec!(0.1));
//! ```

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Order ID
// ---------------------------------------------------------------------------

/// Unique order identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub String);

impl OrderId {
    /// Generate a new random order ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create an order ID from an existing string.
    pub fn from_string(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the inner string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Short form for display (first 8 chars).
    pub fn short(&self) -> &str {
        if self.0.len() >= 8 {
            &self.0[..8]
        } else {
            &self.0
        }
    }
}

impl Default for OrderId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short())
    }
}

impl From<String> for OrderId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for OrderId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Side
// ---------------------------------------------------------------------------

/// Order side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    /// Buy (bid) side.
    Buy,
    /// Sell (ask) side.
    Sell,
}

impl Side {
    /// Get the opposite side.
    pub fn opposite(&self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    /// Whether this is the buy side.
    pub fn is_buy(&self) -> bool {
        matches!(self, Side::Buy)
    }

    /// Whether this is the sell side.
    pub fn is_sell(&self) -> bool {
        matches!(self, Side::Sell)
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

// ---------------------------------------------------------------------------
// Order Type
// ---------------------------------------------------------------------------

/// The type of order determining matching behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    /// Market order — execute immediately at best available price.
    /// No price limit; takes liquidity from the opposite side of the book.
    Market,

    /// Limit order — execute at the specified price or better.
    /// If not immediately filled, rests on the book.
    Limit,

    /// Stop-market order — becomes a market order when the stop price is
    /// breached. Used for stop-loss or breakout entry.
    StopMarket,

    /// Stop-limit order — becomes a limit order when the stop price is
    /// breached. Provides price protection but may not fill.
    StopLimit,

    /// Iceberg order — a limit order that only shows a portion of the
    /// total quantity (display qty). As the displayed portion fills, the
    /// next slice is revealed. Used to hide large order size.
    Iceberg,

    /// Post-only order — a limit order that is guaranteed to add
    /// liquidity. If it would cross the book and take liquidity, it is
    /// rejected instead. Useful for maker rebates.
    PostOnly,
}

impl OrderType {
    /// Whether this order type requires a limit price.
    pub fn requires_price(&self) -> bool {
        matches!(
            self,
            OrderType::Limit | OrderType::StopLimit | OrderType::Iceberg | OrderType::PostOnly
        )
    }

    /// Whether this order type requires a stop price.
    pub fn requires_stop_price(&self) -> bool {
        matches!(self, OrderType::StopMarket | OrderType::StopLimit)
    }

    /// Whether this order type can rest on the book.
    pub fn can_rest(&self) -> bool {
        matches!(
            self,
            OrderType::Limit | OrderType::Iceberg | OrderType::PostOnly
        )
    }

    /// Whether this order type takes liquidity aggressively.
    pub fn is_aggressive(&self) -> bool {
        matches!(self, OrderType::Market)
    }
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "MARKET"),
            OrderType::Limit => write!(f, "LIMIT"),
            OrderType::StopMarket => write!(f, "STOP_MARKET"),
            OrderType::StopLimit => write!(f, "STOP_LIMIT"),
            OrderType::Iceberg => write!(f, "ICEBERG"),
            OrderType::PostOnly => write!(f, "POST_ONLY"),
        }
    }
}

// ---------------------------------------------------------------------------
// Time in Force
// ---------------------------------------------------------------------------

/// Time-in-force policy controlling how long an order remains active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
#[derive(Default)]
pub enum TimeInForce {
    /// Good 'Til Cancelled — the order rests on the book until explicitly
    /// cancelled or fully filled. Default for limit orders.
    #[default]
    GTC,

    /// Immediate or Cancel — fill whatever quantity is available immediately,
    /// then cancel any remaining unfilled portion.
    IOC,

    /// Fill or Kill — the entire order must be filled immediately or it is
    /// rejected completely. No partial fills.
    FOK,

    /// Good 'Til Date — the order rests on the book until a specified
    /// expiry timestamp, after which it is automatically cancelled.
    GTD,
}

impl TimeInForce {
    /// Whether partial fills are allowed under this policy.
    pub fn allows_partial_fill(&self) -> bool {
        match self {
            TimeInForce::GTC => true,
            TimeInForce::IOC => true,
            TimeInForce::FOK => false,
            TimeInForce::GTD => true,
        }
    }

    /// Whether the order can rest on the book (survive past the initial match attempt).
    pub fn can_rest(&self) -> bool {
        match self {
            TimeInForce::GTC => true,
            TimeInForce::IOC => false,
            TimeInForce::FOK => false,
            TimeInForce::GTD => true,
        }
    }
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeInForce::GTC => write!(f, "GTC"),
            TimeInForce::IOC => write!(f, "IOC"),
            TimeInForce::FOK => write!(f, "FOK"),
            TimeInForce::GTD => write!(f, "GTD"),
        }
    }
}

// ---------------------------------------------------------------------------
// Order Status
// ---------------------------------------------------------------------------

/// Current status of an order through its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderStatus {
    /// Order has been created but not yet submitted.
    Pending,

    /// Order has been submitted and accepted by the matching engine.
    /// For limit/iceberg/post-only, it may be resting on the book.
    Open,

    /// Order has been partially filled (some quantity remains).
    PartiallyFilled,

    /// Order has been completely filled (no remaining quantity).
    Filled,

    /// Order has been cancelled (by user or by system, e.g. IOC remainder).
    Cancelled,

    /// Order was rejected (e.g. post-only would cross, invalid parameters).
    Rejected,

    /// Order has expired (GTD past its expiry time).
    Expired,

    /// Stop order is waiting for the trigger price to be reached.
    /// Once triggered it transitions to Open (stop-limit) or immediately
    /// fills (stop-market).
    StopWaiting,
}

impl OrderStatus {
    /// Whether the order is in a terminal state (no further transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderStatus::Filled
                | OrderStatus::Cancelled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }

    /// Whether the order is active (could still be matched).
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            OrderStatus::Open | OrderStatus::PartiallyFilled | OrderStatus::StopWaiting
        )
    }
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderStatus::Pending => write!(f, "PENDING"),
            OrderStatus::Open => write!(f, "OPEN"),
            OrderStatus::PartiallyFilled => write!(f, "PARTIAL"),
            OrderStatus::Filled => write!(f, "FILLED"),
            OrderStatus::Cancelled => write!(f, "CANCELLED"),
            OrderStatus::Rejected => write!(f, "REJECTED"),
            OrderStatus::Expired => write!(f, "EXPIRED"),
            OrderStatus::StopWaiting => write!(f, "STOP_WAITING"),
        }
    }
}

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------

/// A complete order with all parameters needed for matching engine processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order identifier.
    pub id: OrderId,

    /// Trading symbol (e.g. "BTC/USDT").
    pub symbol: String,

    /// Buy or sell.
    pub side: Side,

    /// Order type (market, limit, stop, etc.).
    pub order_type: OrderType,

    /// Time-in-force policy.
    pub time_in_force: TimeInForce,

    /// Total quantity to fill.
    pub quantity: Decimal,

    /// Filled quantity so far.
    pub filled_quantity: Decimal,

    /// Limit price (required for Limit, StopLimit, Iceberg, PostOnly).
    pub price: Option<Decimal>,

    /// Stop/trigger price (required for StopMarket, StopLimit).
    pub stop_price: Option<Decimal>,

    /// Display quantity for iceberg orders (how much is visible on the book).
    pub display_quantity: Option<Decimal>,

    /// Expiry time for GTD orders.
    pub expire_at: Option<DateTime<Utc>>,

    /// Current status.
    pub status: OrderStatus,

    /// When the order was created.
    pub created_at: DateTime<Utc>,

    /// When the order was last updated.
    pub updated_at: DateTime<Utc>,

    /// Client-provided tag for tracking (e.g. strategy name).
    pub client_tag: Option<String>,

    /// Whether this order is simulated (not sent to a real exchange).
    pub simulated: bool,
}

impl Order {
    // ── Constructors ───────────────────────────────────────────────────

    /// Create a market order.
    pub fn market(side: Side, quantity: Decimal) -> Self {
        Self {
            id: OrderId::new(),
            symbol: String::new(),
            side,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::IOC, // Market orders are implicitly IOC
            quantity,
            filled_quantity: Decimal::ZERO,
            price: None,
            stop_price: None,
            display_quantity: None,
            expire_at: None,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            client_tag: None,
            simulated: true,
        }
    }

    /// Create a limit order (default GTC).
    pub fn limit(side: Side, price: Decimal, quantity: Decimal) -> Self {
        Self {
            id: OrderId::new(),
            symbol: String::new(),
            side,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            quantity,
            filled_quantity: Decimal::ZERO,
            price: Some(price),
            stop_price: None,
            display_quantity: None,
            expire_at: None,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            client_tag: None,
            simulated: true,
        }
    }

    /// Create a stop-market order.
    pub fn stop_market(side: Side, stop_price: Decimal, quantity: Decimal) -> Self {
        Self {
            id: OrderId::new(),
            symbol: String::new(),
            side,
            order_type: OrderType::StopMarket,
            time_in_force: TimeInForce::GTC,
            quantity,
            filled_quantity: Decimal::ZERO,
            price: None,
            stop_price: Some(stop_price),
            display_quantity: None,
            expire_at: None,
            status: OrderStatus::StopWaiting,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            client_tag: None,
            simulated: true,
        }
    }

    /// Create a stop-limit order.
    pub fn stop_limit(
        side: Side,
        stop_price: Decimal,
        limit_price: Decimal,
        quantity: Decimal,
    ) -> Self {
        Self {
            id: OrderId::new(),
            symbol: String::new(),
            side,
            order_type: OrderType::StopLimit,
            time_in_force: TimeInForce::GTC,
            quantity,
            filled_quantity: Decimal::ZERO,
            price: Some(limit_price),
            stop_price: Some(stop_price),
            display_quantity: None,
            expire_at: None,
            status: OrderStatus::StopWaiting,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            client_tag: None,
            simulated: true,
        }
    }

    /// Create an iceberg order with visible display quantity.
    pub fn iceberg(
        side: Side,
        price: Decimal,
        total_quantity: Decimal,
        display_quantity: Decimal,
    ) -> Self {
        Self {
            id: OrderId::new(),
            symbol: String::new(),
            side,
            order_type: OrderType::Iceberg,
            time_in_force: TimeInForce::GTC,
            quantity: total_quantity,
            filled_quantity: Decimal::ZERO,
            price: Some(price),
            stop_price: None,
            display_quantity: Some(display_quantity),
            expire_at: None,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            client_tag: None,
            simulated: true,
        }
    }

    /// Create a post-only limit order (rejected if it would cross the book).
    pub fn post_only(side: Side, price: Decimal, quantity: Decimal) -> Self {
        Self {
            id: OrderId::new(),
            symbol: String::new(),
            side,
            order_type: OrderType::PostOnly,
            time_in_force: TimeInForce::GTC,
            quantity,
            filled_quantity: Decimal::ZERO,
            price: Some(price),
            stop_price: None,
            display_quantity: None,
            expire_at: None,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            client_tag: None,
            simulated: true,
        }
    }

    // ── Builder methods ────────────────────────────────────────────────

    /// Set the trading symbol.
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = symbol.into();
        self
    }

    /// Set the time-in-force policy.
    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Set the client tag.
    pub fn with_client_tag(mut self, tag: impl Into<String>) -> Self {
        self.client_tag = Some(tag.into());
        self
    }

    /// Set the order ID (for replaying historical orders).
    pub fn with_id(mut self, id: OrderId) -> Self {
        self.id = id;
        self
    }

    /// Set the expiry time (for GTD orders).
    pub fn with_expire_at(mut self, expire_at: DateTime<Utc>) -> Self {
        self.expire_at = Some(expire_at);
        self.time_in_force = TimeInForce::GTD;
        self
    }

    /// Mark this order as non-simulated (real exchange order).
    pub fn live(mut self) -> Self {
        self.simulated = false;
        self
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Remaining (unfilled) quantity.
    pub fn remaining_quantity(&self) -> Decimal {
        self.quantity - self.filled_quantity
    }

    /// Whether the order has been completely filled.
    pub fn is_filled(&self) -> bool {
        self.filled_quantity >= self.quantity
    }

    /// Whether the order is still active (can be matched or is waiting for trigger).
    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    /// Whether the order is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    /// Whether this is a stop order that hasn't been triggered yet.
    pub fn is_stop_waiting(&self) -> bool {
        self.status == OrderStatus::StopWaiting
    }

    /// Fill ratio as a percentage (0.0 – 100.0).
    pub fn fill_pct(&self) -> f64 {
        if self.quantity > Decimal::ZERO {
            let ratio = self.filled_quantity / self.quantity;
            // Convert Decimal to f64 for percentage
            ratio.try_into().unwrap_or(0.0) * 100.0
        } else {
            0.0
        }
    }

    /// The effective display quantity (for iceberg orders, or full quantity otherwise).
    pub fn effective_display_quantity(&self) -> Decimal {
        self.display_quantity
            .unwrap_or(self.remaining_quantity())
            .min(self.remaining_quantity())
    }

    // ── Mutations (called by matching engine) ──────────────────────────

    /// Record a fill of `qty` at `fill_price`.
    ///
    /// Updates filled_quantity and status. Returns the filled amount.
    pub fn record_fill(&mut self, qty: Decimal) -> Decimal {
        let actual = qty.min(self.remaining_quantity());
        self.filled_quantity += actual;
        self.updated_at = Utc::now();

        if self.is_filled() {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }

        actual
    }

    /// Cancel this order.
    pub fn cancel(&mut self) {
        if !self.status.is_terminal() {
            self.status = OrderStatus::Cancelled;
            self.updated_at = Utc::now();
        }
    }

    /// Reject this order with a reason.
    pub fn reject(&mut self) {
        if self.status == OrderStatus::Pending {
            self.status = OrderStatus::Rejected;
            self.updated_at = Utc::now();
        }
    }

    /// Expire this order (for GTD orders past their expiry).
    pub fn expire(&mut self) {
        if !self.status.is_terminal() {
            self.status = OrderStatus::Expired;
            self.updated_at = Utc::now();
        }
    }

    /// Trigger a stop order (transition from StopWaiting to Open/Pending).
    pub fn trigger(&mut self) {
        if self.status == OrderStatus::StopWaiting {
            self.status = OrderStatus::Open;
            self.updated_at = Utc::now();
        }
    }

    /// Accept a pending order (transition from Pending to Open).
    pub fn accept(&mut self) {
        if self.status == OrderStatus::Pending {
            self.status = OrderStatus::Open;
            self.updated_at = Utc::now();
        }
    }

    // ── Validation ─────────────────────────────────────────────────────

    /// Validate that the order parameters are consistent.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.quantity <= Decimal::ZERO {
            return Err("Quantity must be positive".into());
        }

        if self.order_type.requires_price() && self.price.is_none() {
            return Err(format!("{} order requires a price", self.order_type));
        }

        if self.order_type.requires_stop_price() && self.stop_price.is_none() {
            return Err(format!("{} order requires a stop_price", self.order_type));
        }

        if let Some(price) = self.price
            && price <= Decimal::ZERO
        {
            return Err("Price must be positive".into());
        }

        if let Some(stop) = self.stop_price
            && stop <= Decimal::ZERO
        {
            return Err("Stop price must be positive".into());
        }

        if let Some(display) = self.display_quantity {
            if display <= Decimal::ZERO {
                return Err("Display quantity must be positive".into());
            }
            if display > self.quantity {
                return Err("Display quantity cannot exceed total quantity".into());
            }
        }

        if self.time_in_force == TimeInForce::GTD && self.expire_at.is_none() {
            return Err("GTD orders require an expire_at timestamp".into());
        }

        // Note: Stop-price direction validation (buy stop above market, sell stop
        // below market) requires the current market price, which is not available
        // at order construction time. Use `validate_against_market(price)` instead.
        //
        // Post-only price validation (buy below best ask, sell above best bid) is
        // enforced at matching time by the MatchingEngine, not at construction.

        Ok(())
    }

    /// Reduce the order quantity (amendment that preserves queue position).
    ///
    /// Returns `Ok(())` if the reduction is valid, or an error message if not.
    /// Increasing quantity is not allowed via this method — it would require
    /// a cancel-replace which loses queue position.
    pub fn reduce_quantity(&mut self, new_qty: Decimal) -> std::result::Result<(), String> {
        if new_qty <= Decimal::ZERO {
            return Err("New quantity must be positive; use cancel() to remove the order".into());
        }
        if new_qty > self.quantity {
            return Err(format!(
                "Cannot increase quantity via reduce_quantity ({} > {}). Use cancel-replace instead.",
                new_qty, self.quantity
            ));
        }
        if new_qty < self.filled_quantity {
            return Err(format!(
                "New quantity ({}) cannot be less than already filled quantity ({})",
                new_qty, self.filled_quantity
            ));
        }
        self.quantity = new_qty;
        self.updated_at = Utc::now();
        // If the reduction causes the order to become fully filled, update status.
        if self.filled_quantity >= self.quantity {
            self.status = OrderStatus::Filled;
        }
        Ok(())
    }

    /// Amend the limit price of a resting order (cancel-replace semantics).
    ///
    /// This always results in loss of queue position. The caller (matching
    /// engine) is responsible for removing the order from its old price level
    /// and re-inserting at the new price.
    ///
    /// Returns `Ok(())` if the amendment is valid.
    pub fn amend_price(&mut self, new_price: Decimal) -> std::result::Result<(), String> {
        if new_price <= Decimal::ZERO {
            return Err("Price must be positive".into());
        }
        if !self.order_type.requires_price() {
            return Err(format!(
                "{} orders do not have an amendable price",
                self.order_type
            ));
        }
        if self.status.is_terminal() {
            return Err("Cannot amend a terminal order".into());
        }
        self.price = Some(new_price);
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Convert this order into a `RestingOrder` for insertion into the book.
    ///
    /// Only applicable to order types that can rest on the book (Limit,
    /// PostOnly, Iceberg). Returns `None` for market and stop orders that
    /// haven't been triggered.
    pub fn to_resting_order(&self) -> Option<crate::orderbook::RestingOrder> {
        match self.order_type {
            OrderType::Limit | OrderType::PostOnly | OrderType::Iceberg => {
                let mut resting = crate::orderbook::RestingOrder::new(
                    self.side,
                    self.price?,
                    self.remaining_quantity(),
                    self.created_at,
                );
                // Preserve the original order ID so we can correlate later.
                resting.id = self.id.to_string();
                Some(resting)
            }
            // StopLimit can rest after being triggered
            OrderType::StopLimit if self.status == OrderStatus::Open => {
                let mut resting = crate::orderbook::RestingOrder::new(
                    self.side,
                    self.price?,
                    self.remaining_quantity(),
                    self.created_at,
                );
                resting.id = self.id.to_string();
                Some(resting)
            }
            _ => None,
        }
    }

    /// Validate stop-price direction against the current market price.
    ///
    /// - Buy stop: `stop_price` should be ≥ `market_price` (triggers on upward move).
    /// - Sell stop: `stop_price` should be ≤ `market_price` (triggers on downward move).
    ///
    /// Returns `Ok(())` if valid or if the order has no stop price.
    pub fn validate_against_market(
        &self,
        market_price: Decimal,
    ) -> std::result::Result<(), String> {
        if let Some(stop) = self.stop_price {
            match self.side {
                Side::Buy => {
                    if stop < market_price {
                        return Err(format!(
                            "Buy stop price ({}) should be at or above market price ({})",
                            stop, market_price
                        ));
                    }
                }
                Side::Sell => {
                    if stop > market_price {
                        return Err(format!(
                            "Sell stop price ({}) should be at or below market price ({})",
                            stop, market_price
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} {} {} @ ",
            self.id, self.order_type, self.side, self.quantity,
        )?;

        match (self.price, self.stop_price) {
            (Some(p), Some(s)) => write!(f, "{} (stop {})", p, s)?,
            (Some(p), None) => write!(f, "{}", p)?,
            (None, Some(s)) => write!(f, "MKT (stop {})", s)?,
            (None, None) => write!(f, "MKT")?,
        }

        write!(f, " [{}] {}", self.time_in_force, self.status)?;

        if self.filled_quantity > Decimal::ZERO {
            write!(f, " (filled {}/{})", self.filled_quantity, self.quantity)?;
        }

        if let Some(ref tag) = self.client_tag {
            write!(f, " tag={}", tag)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Order Event (for matching engine output)
// ---------------------------------------------------------------------------

/// An event produced by the matching engine during order processing.
///
/// These events form the audit trail of order lifecycle transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderEvent {
    /// Order was accepted and is now active.
    Accepted {
        order_id: OrderId,
        timestamp: DateTime<Utc>,
    },

    /// Order was rejected (e.g. post-only would cross, invalid params).
    Rejected {
        order_id: OrderId,
        reason: String,
        timestamp: DateTime<Utc>,
    },

    /// Order (or a portion) was filled.
    Fill {
        order_id: OrderId,
        price: Decimal,
        quantity: Decimal,
        remaining: Decimal,
        is_maker: bool,
        timestamp: DateTime<Utc>,
    },

    /// Order was cancelled (by user, IOC remainder, or system).
    Cancelled {
        order_id: OrderId,
        remaining: Decimal,
        reason: CancelReason,
        timestamp: DateTime<Utc>,
    },

    /// Stop order was triggered.
    Triggered {
        order_id: OrderId,
        trigger_price: Decimal,
        timestamp: DateTime<Utc>,
    },

    /// Order expired (GTD past expiry).
    Expired {
        order_id: OrderId,
        timestamp: DateTime<Utc>,
    },

    /// Iceberg slice was replenished (new visible tranche).
    IcebergReplenished {
        order_id: OrderId,
        new_display_quantity: Decimal,
        total_remaining: Decimal,
        timestamp: DateTime<Utc>,
    },
}

impl OrderEvent {
    /// Get the order ID associated with this event.
    pub fn order_id(&self) -> &OrderId {
        match self {
            OrderEvent::Accepted { order_id, .. }
            | OrderEvent::Rejected { order_id, .. }
            | OrderEvent::Fill { order_id, .. }
            | OrderEvent::Cancelled { order_id, .. }
            | OrderEvent::Triggered { order_id, .. }
            | OrderEvent::Expired { order_id, .. }
            | OrderEvent::IcebergReplenished { order_id, .. } => order_id,
        }
    }

    /// Get the timestamp of this event.
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            OrderEvent::Accepted { timestamp, .. }
            | OrderEvent::Rejected { timestamp, .. }
            | OrderEvent::Fill { timestamp, .. }
            | OrderEvent::Cancelled { timestamp, .. }
            | OrderEvent::Triggered { timestamp, .. }
            | OrderEvent::Expired { timestamp, .. }
            | OrderEvent::IcebergReplenished { timestamp, .. } => *timestamp,
        }
    }

    /// Convert this event into a FIX-style execution report string.
    ///
    /// Produces a pipe-delimited summary compatible with FIX tag semantics:
    /// `ExecType | OrdStatus | OrderID | Side | Price | Qty | Text`
    ///
    /// This is a simplified representation; a full FIX adapter would produce
    /// proper tag=value pairs.
    pub fn to_execution_report(&self) -> String {
        match self {
            OrderEvent::Accepted {
                order_id,
                timestamp,
            } => {
                format!(
                    "ExecType=New|OrdStatus=New|OrderID={}|TransactTime={}",
                    order_id, timestamp
                )
            }
            OrderEvent::Rejected {
                order_id,
                reason,
                timestamp,
            } => {
                format!(
                    "ExecType=Rejected|OrdStatus=Rejected|OrderID={}|Text={}|TransactTime={}",
                    order_id, reason, timestamp
                )
            }
            OrderEvent::Fill {
                order_id,
                price,
                quantity,
                remaining,
                is_maker,
                timestamp,
            } => {
                let ord_status = if *remaining <= Decimal::ZERO {
                    "Filled"
                } else {
                    "PartiallyFilled"
                };
                format!(
                    "ExecType=Trade|OrdStatus={}|OrderID={}|LastPx={}|LastQty={}|LeavesQty={}|IsMaker={}|TransactTime={}",
                    ord_status, order_id, price, quantity, remaining, is_maker, timestamp
                )
            }
            OrderEvent::Cancelled {
                order_id,
                remaining,
                reason,
                timestamp,
            } => {
                format!(
                    "ExecType=Cancelled|OrdStatus=Cancelled|OrderID={}|LeavesQty={}|Text={:?}|TransactTime={}",
                    order_id, remaining, reason, timestamp
                )
            }
            OrderEvent::Triggered {
                order_id,
                trigger_price,
                timestamp,
            } => {
                format!(
                    "ExecType=Triggered|OrdStatus=Open|OrderID={}|TriggerPx={}|TransactTime={}",
                    order_id, trigger_price, timestamp
                )
            }
            OrderEvent::Expired {
                order_id,
                timestamp,
            } => {
                format!(
                    "ExecType=Expired|OrdStatus=Expired|OrderID={}|TransactTime={}",
                    order_id, timestamp
                )
            }
            OrderEvent::IcebergReplenished {
                order_id,
                new_display_quantity,
                total_remaining,
                timestamp,
            } => {
                format!(
                    "ExecType=Restated|OrdStatus=Open|OrderID={}|DisplayQty={}|LeavesQty={}|TransactTime={}",
                    order_id, new_display_quantity, total_remaining, timestamp
                )
            }
        }
    }
}

impl fmt::Display for OrderEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderEvent::Accepted { order_id, .. } => {
                write!(f, "ACCEPTED {}", order_id)
            }
            OrderEvent::Rejected {
                order_id, reason, ..
            } => write!(f, "REJECTED {} — {}", order_id, reason),
            OrderEvent::Fill {
                order_id,
                price,
                quantity,
                remaining,
                is_maker,
                ..
            } => {
                let role = if *is_maker { "maker" } else { "taker" };
                write!(
                    f,
                    "FILL {} {} @ {} ({}, remaining: {})",
                    order_id, quantity, price, role, remaining
                )
            }
            OrderEvent::Cancelled {
                order_id,
                remaining,
                reason,
                ..
            } => write!(
                f,
                "CANCELLED {} (remaining: {}, reason: {})",
                order_id, remaining, reason
            ),
            OrderEvent::Triggered {
                order_id,
                trigger_price,
                ..
            } => write!(f, "TRIGGERED {} @ {}", order_id, trigger_price),
            OrderEvent::Expired { order_id, .. } => {
                write!(f, "EXPIRED {}", order_id)
            }
            OrderEvent::IcebergReplenished {
                order_id,
                new_display_quantity,
                total_remaining,
                ..
            } => write!(
                f,
                "ICEBERG_REPLENISH {} display={}, total_remaining={}",
                order_id, new_display_quantity, total_remaining
            ),
        }
    }
}

/// Reason for order cancellation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelReason {
    /// Cancelled by the user/strategy.
    UserRequested,
    /// IOC order — unfilled remainder cancelled.
    IocRemainder,
    /// FOK order — could not fill entirely.
    FokNotFilled,
    /// Post-only order would have crossed the book.
    PostOnlyWouldCross,
    /// GTD order expired.
    Expired,
    /// System-initiated (e.g. risk limit breach, kill switch).
    System(String),
}

impl fmt::Display for CancelReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CancelReason::UserRequested => write!(f, "user_requested"),
            CancelReason::IocRemainder => write!(f, "ioc_remainder"),
            CancelReason::FokNotFilled => write!(f, "fok_not_filled"),
            CancelReason::PostOnlyWouldCross => write!(f, "post_only_would_cross"),
            CancelReason::Expired => write!(f, "expired"),
            CancelReason::System(reason) => write!(f, "system: {}", reason),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // ── OrderId ────────────────────────────────────────────────────────

    #[test]
    fn test_order_id_new() {
        let id1 = OrderId::new();
        let id2 = OrderId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_order_id_from_string() {
        let id = OrderId::from_string("test-123");
        assert_eq!(id.as_str(), "test-123");
    }

    #[test]
    fn test_order_id_short() {
        let id = OrderId::from_string("abcdefghijklmnop");
        assert_eq!(id.short(), "abcdefgh");
    }

    #[test]
    fn test_order_id_display() {
        let id = OrderId::from_string("abc");
        assert_eq!(format!("{}", id), "abc");
    }

    // ── Side ───────────────────────────────────────────────────────────

    #[test]
    fn test_side_opposite() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
    }

    #[test]
    fn test_side_is_buy_sell() {
        assert!(Side::Buy.is_buy());
        assert!(!Side::Buy.is_sell());
        assert!(Side::Sell.is_sell());
        assert!(!Side::Sell.is_buy());
    }

    #[test]
    fn test_side_display() {
        assert_eq!(format!("{}", Side::Buy), "BUY");
        assert_eq!(format!("{}", Side::Sell), "SELL");
    }

    // ── OrderType ──────────────────────────────────────────────────────

    #[test]
    fn test_order_type_requires_price() {
        assert!(!OrderType::Market.requires_price());
        assert!(OrderType::Limit.requires_price());
        assert!(!OrderType::StopMarket.requires_price());
        assert!(OrderType::StopLimit.requires_price());
        assert!(OrderType::Iceberg.requires_price());
        assert!(OrderType::PostOnly.requires_price());
    }

    #[test]
    fn test_order_type_requires_stop() {
        assert!(!OrderType::Market.requires_stop_price());
        assert!(!OrderType::Limit.requires_stop_price());
        assert!(OrderType::StopMarket.requires_stop_price());
        assert!(OrderType::StopLimit.requires_stop_price());
    }

    #[test]
    fn test_order_type_can_rest() {
        assert!(!OrderType::Market.can_rest());
        assert!(OrderType::Limit.can_rest());
        assert!(OrderType::Iceberg.can_rest());
        assert!(OrderType::PostOnly.can_rest());
    }

    #[test]
    fn test_order_type_display() {
        assert_eq!(format!("{}", OrderType::Market), "MARKET");
        assert_eq!(format!("{}", OrderType::Limit), "LIMIT");
        assert_eq!(format!("{}", OrderType::StopMarket), "STOP_MARKET");
        assert_eq!(format!("{}", OrderType::StopLimit), "STOP_LIMIT");
        assert_eq!(format!("{}", OrderType::Iceberg), "ICEBERG");
        assert_eq!(format!("{}", OrderType::PostOnly), "POST_ONLY");
    }

    // ── TimeInForce ────────────────────────────────────────────────────

    #[test]
    fn test_tif_default() {
        assert_eq!(TimeInForce::default(), TimeInForce::GTC);
    }

    #[test]
    fn test_tif_allows_partial_fill() {
        assert!(TimeInForce::GTC.allows_partial_fill());
        assert!(TimeInForce::IOC.allows_partial_fill());
        assert!(!TimeInForce::FOK.allows_partial_fill());
        assert!(TimeInForce::GTD.allows_partial_fill());
    }

    #[test]
    fn test_tif_can_rest() {
        assert!(TimeInForce::GTC.can_rest());
        assert!(!TimeInForce::IOC.can_rest());
        assert!(!TimeInForce::FOK.can_rest());
        assert!(TimeInForce::GTD.can_rest());
    }

    // ── OrderStatus ────────────────────────────────────────────────────

    #[test]
    fn test_status_is_terminal() {
        assert!(!OrderStatus::Pending.is_terminal());
        assert!(!OrderStatus::Open.is_terminal());
        assert!(!OrderStatus::PartiallyFilled.is_terminal());
        assert!(OrderStatus::Filled.is_terminal());
        assert!(OrderStatus::Cancelled.is_terminal());
        assert!(OrderStatus::Rejected.is_terminal());
        assert!(OrderStatus::Expired.is_terminal());
        assert!(!OrderStatus::StopWaiting.is_terminal());
    }

    #[test]
    fn test_status_is_active() {
        assert!(!OrderStatus::Pending.is_active());
        assert!(OrderStatus::Open.is_active());
        assert!(OrderStatus::PartiallyFilled.is_active());
        assert!(!OrderStatus::Filled.is_active());
        assert!(!OrderStatus::Cancelled.is_active());
        assert!(OrderStatus::StopWaiting.is_active());
    }

    // ── Order constructors ─────────────────────────────────────────────

    #[test]
    fn test_market_order() {
        let order = Order::market(Side::Buy, dec!(1.0));
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.time_in_force, TimeInForce::IOC);
        assert_eq!(order.quantity, dec!(1.0));
        assert_eq!(order.filled_quantity, Decimal::ZERO);
        assert!(order.price.is_none());
        assert!(order.stop_price.is_none());
        assert_eq!(order.status, OrderStatus::Pending);
        assert!(order.simulated);
    }

    #[test]
    fn test_limit_order() {
        let order = Order::limit(Side::Sell, dec!(50000.0), dec!(2.5));
        assert_eq!(order.side, Side::Sell);
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.time_in_force, TimeInForce::GTC);
        assert_eq!(order.price, Some(dec!(50000.0)));
        assert_eq!(order.quantity, dec!(2.5));
    }

    #[test]
    fn test_stop_market_order() {
        let order = Order::stop_market(Side::Sell, dec!(45000.0), dec!(1.0));
        assert_eq!(order.order_type, OrderType::StopMarket);
        assert_eq!(order.stop_price, Some(dec!(45000.0)));
        assert!(order.price.is_none());
        assert_eq!(order.status, OrderStatus::StopWaiting);
    }

    #[test]
    fn test_stop_limit_order() {
        let order = Order::stop_limit(Side::Sell, dec!(45000.0), dec!(44900.0), dec!(1.0));
        assert_eq!(order.order_type, OrderType::StopLimit);
        assert_eq!(order.stop_price, Some(dec!(45000.0)));
        assert_eq!(order.price, Some(dec!(44900.0)));
        assert_eq!(order.status, OrderStatus::StopWaiting);
    }

    #[test]
    fn test_iceberg_order() {
        let order = Order::iceberg(Side::Buy, dec!(50000.0), dec!(10.0), dec!(1.0));
        assert_eq!(order.order_type, OrderType::Iceberg);
        assert_eq!(order.quantity, dec!(10.0));
        assert_eq!(order.display_quantity, Some(dec!(1.0)));
    }

    #[test]
    fn test_post_only_order() {
        let order = Order::post_only(Side::Buy, dec!(49999.0), dec!(1.0));
        assert_eq!(order.order_type, OrderType::PostOnly);
        assert_eq!(order.price, Some(dec!(49999.0)));
    }

    // ── Order builders ─────────────────────────────────────────────────

    #[test]
    fn test_order_with_symbol() {
        let order = Order::market(Side::Buy, dec!(1.0)).with_symbol("BTC/USDT");
        assert_eq!(order.symbol, "BTC/USDT");
    }

    #[test]
    fn test_order_with_tif() {
        let order =
            Order::limit(Side::Buy, dec!(100.0), dec!(1.0)).with_time_in_force(TimeInForce::IOC);
        assert_eq!(order.time_in_force, TimeInForce::IOC);
    }

    #[test]
    fn test_order_with_client_tag() {
        let order = Order::market(Side::Buy, dec!(1.0)).with_client_tag("ema_flip");
        assert_eq!(order.client_tag, Some("ema_flip".to_string()));
    }

    #[test]
    fn test_order_live() {
        let order = Order::market(Side::Buy, dec!(1.0)).live();
        assert!(!order.simulated);
    }

    // ── Order state ────────────────────────────────────────────────────

    #[test]
    fn test_remaining_quantity() {
        let mut order = Order::market(Side::Buy, dec!(5.0));
        assert_eq!(order.remaining_quantity(), dec!(5.0));

        order.filled_quantity = dec!(2.0);
        assert_eq!(order.remaining_quantity(), dec!(3.0));
    }

    #[test]
    fn test_is_filled() {
        let mut order = Order::market(Side::Buy, dec!(5.0));
        assert!(!order.is_filled());

        order.filled_quantity = dec!(5.0);
        assert!(order.is_filled());
    }

    #[test]
    fn test_fill_pct() {
        let mut order = Order::market(Side::Buy, dec!(10.0));
        assert_eq!(order.fill_pct(), 0.0);

        order.filled_quantity = dec!(5.0);
        assert!((order.fill_pct() - 50.0).abs() < 0.01);

        order.filled_quantity = dec!(10.0);
        assert!((order.fill_pct() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_effective_display_quantity() {
        let order = Order::iceberg(Side::Buy, dec!(100.0), dec!(10.0), dec!(2.0));
        assert_eq!(order.effective_display_quantity(), dec!(2.0));

        // Non-iceberg uses full remaining.
        let order = Order::limit(Side::Buy, dec!(100.0), dec!(5.0));
        assert_eq!(order.effective_display_quantity(), dec!(5.0));
    }

    // ── Order mutations ────────────────────────────────────────────────

    #[test]
    fn test_record_fill() {
        let mut order = Order::limit(Side::Buy, dec!(100.0), dec!(5.0));
        order.accept();

        let filled = order.record_fill(dec!(2.0));
        assert_eq!(filled, dec!(2.0));
        assert_eq!(order.filled_quantity, dec!(2.0));
        assert_eq!(order.status, OrderStatus::PartiallyFilled);

        let filled = order.record_fill(dec!(3.0));
        assert_eq!(filled, dec!(3.0));
        assert_eq!(order.status, OrderStatus::Filled);
    }

    #[test]
    fn test_record_fill_overfill_clamped() {
        let mut order = Order::limit(Side::Buy, dec!(100.0), dec!(2.0));
        order.accept();

        let filled = order.record_fill(dec!(5.0));
        assert_eq!(filled, dec!(2.0)); // Clamped to remaining.
        assert_eq!(order.status, OrderStatus::Filled);
    }

    #[test]
    fn test_cancel() {
        let mut order = Order::limit(Side::Buy, dec!(100.0), dec!(5.0));
        order.accept();
        order.cancel();
        assert_eq!(order.status, OrderStatus::Cancelled);
    }

    #[test]
    fn test_cancel_terminal_is_noop() {
        let mut order = Order::limit(Side::Buy, dec!(100.0), dec!(5.0));
        order.accept();
        order.record_fill(dec!(5.0));
        assert_eq!(order.status, OrderStatus::Filled);
        order.cancel(); // No-op.
        assert_eq!(order.status, OrderStatus::Filled);
    }

    #[test]
    fn test_reject() {
        let mut order = Order::post_only(Side::Buy, dec!(100.0), dec!(1.0));
        order.reject();
        assert_eq!(order.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_expire() {
        let mut order = Order::limit(Side::Buy, dec!(100.0), dec!(1.0));
        order.accept();
        order.expire();
        assert_eq!(order.status, OrderStatus::Expired);
    }

    #[test]
    fn test_trigger_stop() {
        let mut order = Order::stop_market(Side::Sell, dec!(45000.0), dec!(1.0));
        assert_eq!(order.status, OrderStatus::StopWaiting);
        order.trigger();
        assert_eq!(order.status, OrderStatus::Open);
    }

    #[test]
    fn test_accept() {
        let mut order = Order::limit(Side::Buy, dec!(100.0), dec!(1.0));
        assert_eq!(order.status, OrderStatus::Pending);
        order.accept();
        assert_eq!(order.status, OrderStatus::Open);
    }

    // ── Order validation ───────────────────────────────────────────────

    #[test]
    fn test_validate_valid_orders() {
        assert!(Order::market(Side::Buy, dec!(1.0)).validate().is_ok());
        assert!(
            Order::limit(Side::Buy, dec!(100.0), dec!(1.0))
                .validate()
                .is_ok()
        );
        assert!(
            Order::stop_market(Side::Sell, dec!(50.0), dec!(1.0))
                .validate()
                .is_ok()
        );
        assert!(
            Order::stop_limit(Side::Sell, dec!(50.0), dec!(49.0), dec!(1.0))
                .validate()
                .is_ok()
        );
        assert!(
            Order::iceberg(Side::Buy, dec!(100.0), dec!(10.0), dec!(1.0))
                .validate()
                .is_ok()
        );
        assert!(
            Order::post_only(Side::Buy, dec!(100.0), dec!(1.0))
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn test_validate_zero_quantity() {
        let order = Order::market(Side::Buy, dec!(0.0));
        assert!(order.validate().is_err());
    }

    #[test]
    fn test_validate_negative_quantity() {
        let order = Order::market(Side::Buy, dec!(-1.0));
        assert!(order.validate().is_err());
    }

    #[test]
    fn test_validate_limit_no_price() {
        let mut order = Order::limit(Side::Buy, dec!(100.0), dec!(1.0));
        order.price = None;
        assert!(order.validate().is_err());
    }

    #[test]
    fn test_validate_stop_no_stop_price() {
        let mut order = Order::stop_market(Side::Sell, dec!(50.0), dec!(1.0));
        order.stop_price = None;
        assert!(order.validate().is_err());
    }

    #[test]
    fn test_validate_iceberg_display_exceeds_total() {
        let order = Order::iceberg(Side::Buy, dec!(100.0), dec!(5.0), dec!(10.0));
        assert!(order.validate().is_err());
    }

    #[test]
    fn test_validate_gtd_no_expiry() {
        let order =
            Order::limit(Side::Buy, dec!(100.0), dec!(1.0)).with_time_in_force(TimeInForce::GTD);
        assert!(order.validate().is_err());
    }

    // ── Order display ──────────────────────────────────────────────────

    #[test]
    fn test_order_display_market() {
        let order = Order::market(Side::Buy, dec!(1.0));
        let display = format!("{}", order);
        assert!(display.contains("MARKET"));
        assert!(display.contains("BUY"));
        assert!(display.contains("MKT"));
    }

    #[test]
    fn test_order_display_limit() {
        let order = Order::limit(Side::Sell, dec!(50000.0), dec!(2.5));
        let display = format!("{}", order);
        assert!(display.contains("LIMIT"));
        assert!(display.contains("SELL"));
        assert!(display.contains("50000"));
    }

    #[test]
    fn test_order_display_with_fill() {
        let mut order = Order::limit(Side::Buy, dec!(100.0), dec!(5.0));
        order.accept();
        order.record_fill(dec!(2.0));
        let display = format!("{}", order);
        assert!(display.contains("filled 2"));
    }

    #[test]
    fn test_order_display_with_tag() {
        let order = Order::market(Side::Buy, dec!(1.0)).with_client_tag("my_strategy");
        let display = format!("{}", order);
        assert!(display.contains("tag=my_strategy"));
    }

    // ── OrderEvent ─────────────────────────────────────────────────────

    #[test]
    fn test_order_event_order_id() {
        let id = OrderId::from_string("test-id");
        let event = OrderEvent::Accepted {
            order_id: id.clone(),
            timestamp: Utc::now(),
        };
        assert_eq!(event.order_id(), &id);
    }

    #[test]
    fn test_order_event_display() {
        let id = OrderId::from_string("test-1234");
        let event = OrderEvent::Fill {
            order_id: id,
            price: dec!(50000.0),
            quantity: dec!(1.0),
            remaining: dec!(0.0),
            is_maker: false,
            timestamp: Utc::now(),
        };
        let display = format!("{}", event);
        assert!(display.contains("FILL"));
        assert!(display.contains("50000"));
        assert!(display.contains("taker"));
    }

    // ── CancelReason ───────────────────────────────────────────────────

    #[test]
    fn test_cancel_reason_display() {
        assert_eq!(format!("{}", CancelReason::UserRequested), "user_requested");
        assert_eq!(format!("{}", CancelReason::IocRemainder), "ioc_remainder");
        assert_eq!(format!("{}", CancelReason::FokNotFilled), "fok_not_filled");
        assert_eq!(
            format!("{}", CancelReason::System("risk breach".into())),
            "system: risk breach"
        );
    }
}
