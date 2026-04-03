//! # Order Book Data Structure
//!
//! Core limit order book implementation using sorted price levels with
//! price-time priority. Supports L2 (aggregated price levels) and L3
//! (individual orders) representations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        OrderBook                                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────────┐    ┌─────────────────────┐            │
//! │  │     Bids (Buy)      │    │    Asks (Sell)       │            │
//! │  │  (desc by price)    │    │  (asc by price)      │            │
//! │  ├─────────────────────┤    ├─────────────────────┤            │
//! │  │ 67000.0 │ 1.50 BTC │    │ 67001.0 │ 0.80 BTC │            │
//! │  │ 66999.0 │ 3.20 BTC │    │ 67002.0 │ 2.10 BTC │            │
//! │  │ 66998.5 │ 0.75 BTC │    │ 67003.0 │ 1.30 BTC │            │
//! │  │   ...   │   ...    │    │   ...   │   ...    │            │
//! │  └─────────────────────┘    └─────────────────────┘            │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │ Per-Level Queue (L3 mode)                                 │   │
//! │  │  Order_1 → Order_2 → Order_3  (FIFO within price level)  │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_lob::orderbook::*;
//! use janus_lob::order_types::Side;
//! use rust_decimal_macros::dec;
//!
//! let mut book = OrderBook::new("BTC/USDT");
//!
//! // Apply L2 snapshot
//! book.apply_snapshot(OrderBookSnapshot {
//!     symbol: "BTC/USDT".into(),
//!     bids: vec![
//!         PriceLevel::new(dec!(67000.0), dec!(1.5)),
//!         PriceLevel::new(dec!(66999.0), dec!(3.2)),
//!     ],
//!     asks: vec![
//!         PriceLevel::new(dec!(67001.0), dec!(0.8)),
//!         PriceLevel::new(dec!(67002.0), dec!(2.1)),
//!     ],
//!     timestamp: Utc::now(),
//!     sequence: 1,
//! })?;
//!
//! assert_eq!(book.best_bid().unwrap().price, dec!(67000.0));
//! assert_eq!(book.best_ask().unwrap().price, dec!(67001.0));
//! assert_eq!(book.spread().unwrap(), dec!(1.0));
//! assert_eq!(book.mid_price().unwrap(), dec!(67000.5));
//! ```

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use thiserror::Error;
use tracing::{debug, trace, warn};
use uuid::Uuid;

use crate::order_types::Side;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by order book operations.
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum OrderBookError {
    /// Price level not found at the given price.
    #[error("Price level not found: {price} on {side} side")]
    LevelNotFound { price: Decimal, side: String },

    /// Invalid price (zero or negative).
    #[error("Invalid price: {0}")]
    InvalidPrice(Decimal),

    /// Invalid quantity (negative).
    #[error("Invalid quantity: {0}")]
    InvalidQuantity(Decimal),

    /// Crossed book (best bid ≥ best ask) — indicates a data error.
    #[error("Crossed book: best_bid={bid} >= best_ask={ask}")]
    CrossedBook { bid: Decimal, ask: Decimal },

    /// Snapshot sequence out of order.
    #[error("Stale snapshot: received seq {received}, expected > {expected}")]
    StaleSnapshot { received: u64, expected: u64 },

    /// Order not found in L3 queue.
    #[error("Order not found: {0}")]
    OrderNotFound(String),

    /// Book is empty on the requested side.
    #[error("Empty book side: {0}")]
    EmptySide(String),

    /// Internal consistency error.
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, OrderBookError>;

// ---------------------------------------------------------------------------
// Price Level
// ---------------------------------------------------------------------------

/// A single price level in the order book (L2 aggregated view).
///
/// Represents the total quantity available at a specific price point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PriceLevel {
    /// Price at this level.
    pub price: Decimal,
    /// Total aggregated quantity at this price.
    pub quantity: Decimal,
    /// Number of individual orders at this level (if known from L3 data).
    pub order_count: u32,
    /// Individual order queue (L3 mode only, FIFO order).
    #[serde(skip)]
    pub orders: Vec<RestingOrder>,
    /// Last update timestamp.
    pub last_updated: DateTime<Utc>,
}

impl PriceLevel {
    /// Create a new price level with aggregated quantity.
    pub fn new(price: Decimal, quantity: Decimal) -> Self {
        Self {
            price,
            quantity,
            order_count: 1,
            orders: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    /// Create a price level with a known order count.
    pub fn with_order_count(price: Decimal, quantity: Decimal, order_count: u32) -> Self {
        Self {
            price,
            quantity,
            order_count,
            orders: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    /// Check whether this level has been fully consumed.
    pub fn is_empty(&self) -> bool {
        self.quantity <= Decimal::ZERO
    }

    /// Notional value of this level (price × quantity).
    pub fn notional(&self) -> Decimal {
        self.price * self.quantity
    }

    /// Consume liquidity from this level.
    ///
    /// Removes up to `qty` from the level's available quantity. If L3 orders
    /// are present, fills them in FIFO order. Returns `(filled_qty, remaining_level_qty)`.
    pub fn consume(&mut self, qty: Decimal) -> (Decimal, Decimal) {
        if qty <= Decimal::ZERO || self.quantity <= Decimal::ZERO {
            return (Decimal::ZERO, self.quantity);
        }

        let filled = qty.min(self.quantity);

        if !self.orders.is_empty() {
            // L3 mode: consume from FIFO order queue.
            let mut remaining_to_fill = filled;
            let mut i = 0;
            while i < self.orders.len() && remaining_to_fill > Decimal::ZERO {
                let take = remaining_to_fill.min(self.orders[i].remaining_quantity);
                self.orders[i].remaining_quantity -= take;
                remaining_to_fill -= take;
                if self.orders[i].is_filled() {
                    i += 1; // will be drained below
                } else {
                    i += 1;
                }
            }
            // Remove fully filled orders from the front.
            self.orders.retain(|o| !o.is_filled());
            self.order_count = self.orders.len() as u32;
        } else if self.order_count > 0 {
            // L2 mode: just decrement the aggregated order count proportionally.
            let ratio = filled / self.quantity;
            let removed = (Decimal::from(self.order_count) * ratio)
                .round()
                .to_string()
                .parse::<u32>()
                .unwrap_or(1)
                .max(if filled >= self.quantity {
                    self.order_count
                } else {
                    0
                });
            self.order_count = self.order_count.saturating_sub(removed);
        }

        self.quantity -= filled;
        self.last_updated = Utc::now();

        (filled, self.quantity)
    }

    /// Add a resting order to this level's L3 queue.
    ///
    /// The order is appended to the back of the FIFO queue (loses time
    /// priority relative to existing orders). The aggregated quantity is
    /// updated accordingly.
    pub fn add_order(&mut self, order: RestingOrder) {
        self.quantity += order.remaining_quantity;
        self.orders.push(order);
        self.order_count = self.orders.len() as u32;
        self.last_updated = Utc::now();
    }

    /// Cancel (remove) a specific order from the L3 queue by ID.
    ///
    /// Returns the removed order if found. The aggregated quantity is
    /// adjusted to reflect the cancellation.
    pub fn cancel_order(&mut self, order_id: &str) -> Option<RestingOrder> {
        if let Some(pos) = self.orders.iter().position(|o| o.id == order_id) {
            let removed = self.orders.remove(pos);
            self.quantity -= removed.remaining_quantity;
            self.order_count = self.orders.len() as u32;
            self.last_updated = Utc::now();
            Some(removed)
        } else {
            None
        }
    }

    /// Modify an order's quantity in the L3 queue.
    ///
    /// - If `new_qty < current_qty`: reduce in-place, preserving queue position.
    /// - If `new_qty > current_qty`: move to back of queue (per exchange rules)
    ///   and increase quantity.
    /// - If `new_qty == 0`: equivalent to cancellation.
    ///
    /// Returns `true` if the order was found and modified.
    pub fn modify_order(&mut self, order_id: &str, new_qty: Decimal) -> bool {
        let pos = match self.orders.iter().position(|o| o.id == order_id) {
            Some(p) => p,
            None => return false,
        };

        if new_qty <= Decimal::ZERO {
            // Cancel.
            let removed = self.orders.remove(pos);
            self.quantity -= removed.remaining_quantity;
            self.order_count = self.orders.len() as u32;
            self.last_updated = Utc::now();
            return true;
        }

        let old_qty = self.orders[pos].remaining_quantity;
        let delta = new_qty - old_qty;

        if new_qty <= old_qty {
            // Reduce in-place — preserves queue position.
            self.orders[pos].remaining_quantity = new_qty;
            self.quantity += delta; // delta is negative or zero
        } else {
            // Increase — must move to back of queue (loses priority).
            let mut order = self.orders.remove(pos);
            order.remaining_quantity = new_qty;
            order.timestamp = Utc::now();
            self.quantity += delta;
            self.orders.push(order);
        }

        self.order_count = self.orders.len() as u32;
        self.last_updated = Utc::now();
        true
    }
}

impl fmt::Display for PriceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} @ {} ({} orders)",
            self.quantity, self.price, self.order_count
        )
    }
}

// ---------------------------------------------------------------------------
// Resting Order (L3)
// ---------------------------------------------------------------------------

/// An individual resting order in the book (L3 representation).
///
/// Orders within a price level are maintained in FIFO order for
/// price-time priority matching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RestingOrder {
    /// Unique order identifier.
    pub id: String,
    /// Side (bid or ask).
    pub side: Side,
    /// Limit price.
    pub price: Decimal,
    /// Remaining quantity.
    pub remaining_quantity: Decimal,
    /// Original quantity at submission.
    pub original_quantity: Decimal,
    /// Timestamp when the order was placed.
    pub timestamp: DateTime<Utc>,
    /// Whether this is our own order (for queue position tracking).
    pub is_own: bool,
}

impl RestingOrder {
    /// Create a new resting order.
    pub fn new(side: Side, price: Decimal, quantity: Decimal, timestamp: DateTime<Utc>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            side,
            price,
            remaining_quantity: quantity,
            original_quantity: quantity,
            timestamp,
            is_own: false,
        }
    }

    /// Create a resting order marked as our own (for queue position tracking).
    pub fn own(side: Side, price: Decimal, quantity: Decimal, timestamp: DateTime<Utc>) -> Self {
        let mut order = Self::new(side, price, quantity, timestamp);
        order.is_own = true;
        order
    }

    /// Check whether this order has been fully filled.
    pub fn is_filled(&self) -> bool {
        self.remaining_quantity <= Decimal::ZERO
    }

    /// Fraction of the original order that has been filled.
    pub fn fill_ratio(&self) -> Decimal {
        if self.original_quantity > Decimal::ZERO {
            (self.original_quantity - self.remaining_quantity) / self.original_quantity
        } else {
            Decimal::ZERO
        }
    }

    /// Apply a partial fill to this resting order.
    ///
    /// Reduces `remaining_quantity` by `qty` (clamped to available) and returns
    /// the actual filled quantity. If the fill exhausts the order, the returned
    /// quantity equals the previous `remaining_quantity`.
    pub fn partial_fill(&mut self, qty: Decimal) -> Decimal {
        if qty <= Decimal::ZERO {
            return Decimal::ZERO;
        }
        let filled = qty.min(self.remaining_quantity);
        self.remaining_quantity -= filled;
        filled
    }
}

impl fmt::Display for RestingOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} {} @ {} (remaining: {})",
            &self.id[..8],
            self.side,
            self.original_quantity,
            self.price,
            self.remaining_quantity,
        )
    }
}

// ---------------------------------------------------------------------------
// Book Side
// ---------------------------------------------------------------------------

/// One side of the order book (bids or asks).
///
/// Uses a `BTreeMap<Decimal, PriceLevel>` for O(log n) insertion/lookup
/// with sorted iteration. Bids are iterated in descending order (highest
/// first), asks in ascending order (lowest first).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookSide {
    /// The side (bid or ask).
    pub side: Side,
    /// Price levels keyed by price, sorted by the BTreeMap's natural ordering.
    /// For bids we negate the key internally so that iteration order is
    /// descending by real price. For asks the natural ascending order is correct.
    levels: BTreeMap<PriceSortKey, PriceLevel>,
    /// Total depth (sum of all quantities across all levels).
    total_quantity: Decimal,
    /// Number of price levels.
    level_count: usize,
}

/// Internal sort key that encodes price ordering.
///
/// Asks sort ascending (natural Decimal order).
/// Bids sort descending — we negate the price so BTreeMap's ascending
/// iteration yields highest-price-first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct PriceSortKey(i128);

impl PriceSortKey {
    /// Create a sort key for asks (ascending — best ask is lowest price).
    fn ask(price: Decimal) -> Self {
        Self(price.mantissa())
    }

    /// Create a sort key for bids (descending — best bid is highest price).
    fn bid(price: Decimal) -> Self {
        Self(-price.mantissa())
    }

    fn for_side(price: Decimal, side: Side) -> Self {
        match side {
            Side::Buy => Self::bid(price),
            Side::Sell => Self::ask(price),
        }
    }
}

impl BookSide {
    /// Create a new empty book side.
    pub fn new(side: Side) -> Self {
        Self {
            side,
            levels: BTreeMap::new(),
            total_quantity: Decimal::ZERO,
            level_count: 0,
        }
    }

    /// Get the best price level (best bid = highest price, best ask = lowest price).
    pub fn best(&self) -> Option<&PriceLevel> {
        self.levels.values().next()
    }

    /// Get the best price (best bid = highest, best ask = lowest).
    pub fn best_price(&self) -> Option<Decimal> {
        self.best().map(|l| l.price)
    }

    /// Get the number of price levels.
    pub fn len(&self) -> usize {
        self.level_count
    }

    /// Check whether this side is empty.
    pub fn is_empty(&self) -> bool {
        self.level_count == 0
    }

    /// Total depth across all levels.
    pub fn total_depth(&self) -> Decimal {
        self.total_quantity
    }

    /// Total notional value across all levels.
    pub fn total_notional(&self) -> Decimal {
        self.levels.values().map(PriceLevel::notional).sum()
    }

    /// Insert or update a price level.
    ///
    /// If `quantity` is zero the level is removed (exchange convention for
    /// L2 delta updates).
    pub fn set_level(&mut self, price: Decimal, quantity: Decimal) -> Result<()> {
        if price <= Decimal::ZERO {
            return Err(OrderBookError::InvalidPrice(price));
        }
        if quantity < Decimal::ZERO {
            return Err(OrderBookError::InvalidQuantity(quantity));
        }

        let key = PriceSortKey::for_side(price, self.side);

        if quantity.is_zero() {
            // Remove the level.
            if let Some(removed) = self.levels.remove(&key) {
                self.total_quantity -= removed.quantity;
                self.level_count -= 1;
                trace!(price = %price, side = %self.side, "Removed price level");
            }
        } else if let Some(existing) = self.levels.get_mut(&key) {
            // Update existing level.
            self.total_quantity -= existing.quantity;
            existing.quantity = quantity;
            existing.last_updated = Utc::now();
            self.total_quantity += quantity;
            trace!(price = %price, qty = %quantity, side = %self.side, "Updated price level");
        } else {
            // Insert new level.
            let level = PriceLevel::new(price, quantity);
            self.levels.insert(key, level);
            self.total_quantity += quantity;
            self.level_count += 1;
            trace!(price = %price, qty = %quantity, side = %self.side, "Inserted price level");
        }

        Ok(())
    }

    /// Get the price level at a specific price.
    pub fn get_level(&self, price: Decimal) -> Option<&PriceLevel> {
        let key = PriceSortKey::for_side(price, self.side);
        self.levels.get(&key)
    }

    /// Get a mutable reference to the price level at a specific price.
    pub fn get_level_mut(&mut self, price: Decimal) -> Option<&mut PriceLevel> {
        let key = PriceSortKey::for_side(price, self.side);
        self.levels.get_mut(&key)
    }

    /// Remove a price level at a specific price.
    pub fn remove_level(&mut self, price: Decimal) -> Option<PriceLevel> {
        let key = PriceSortKey::for_side(price, self.side);
        if let Some(removed) = self.levels.remove(&key) {
            self.total_quantity -= removed.quantity;
            self.level_count -= 1;
            Some(removed)
        } else {
            None
        }
    }

    /// Iterate over price levels in best-to-worst order.
    pub fn iter(&self) -> impl Iterator<Item = &PriceLevel> {
        self.levels.values()
    }

    /// Get the top `n` price levels (best-to-worst).
    pub fn top_n(&self, n: usize) -> Vec<&PriceLevel> {
        self.levels.values().take(n).collect()
    }

    /// Clear all levels.
    pub fn clear(&mut self) {
        self.levels.clear();
        self.total_quantity = Decimal::ZERO;
        self.level_count = 0;
    }

    /// Compute the volume-weighted average price (VWAP) over the top `depth`
    /// units of liquidity.
    ///
    /// This answers the question: "If I market-swept `depth` units, what would
    /// my average fill price be?"
    pub fn vwap_for_depth(&self, depth: Decimal) -> Option<Decimal> {
        if depth <= Decimal::ZERO || self.is_empty() {
            return None;
        }

        let mut remaining = depth;
        let mut notional_sum = Decimal::ZERO;
        let mut filled_qty = Decimal::ZERO;

        for level in self.iter() {
            let take = remaining.min(level.quantity);
            notional_sum += take * level.price;
            filled_qty += take;
            remaining -= take;
            if remaining <= Decimal::ZERO {
                break;
            }
        }

        if filled_qty > Decimal::ZERO {
            Some(notional_sum / filled_qty)
        } else {
            None
        }
    }

    /// Consume liquidity from the book side, sweeping from best price.
    ///
    /// Walks the book from the best price level towards worse prices, consuming
    /// up to `qty` total. Returns a list of `(price, filled_qty)` pairs for
    /// each level that was (partially or fully) consumed. Fully consumed levels
    /// are removed from the book.
    ///
    /// An optional `limit_price` constrains the sweep:
    /// - For asks (buy sweep): only consume levels with price ≤ limit_price.
    /// - For bids (sell sweep): only consume levels with price ≥ limit_price.
    ///
    /// This is the core function that powers market order and crossing limit
    /// order matching.
    pub fn consume_liquidity(
        &mut self,
        qty: Decimal,
        limit_price: Option<Decimal>,
        max_levels: Option<usize>,
    ) -> Vec<(Decimal, Decimal)> {
        if qty <= Decimal::ZERO || self.is_empty() {
            return Vec::new();
        }

        let mut remaining = qty;
        let mut fills: Vec<(Decimal, Decimal)> = Vec::new();
        let mut levels_consumed = 0usize;
        let max = max_levels.unwrap_or(usize::MAX);

        // Collect the keys we need to visit in best-to-worst order.
        let keys: Vec<PriceSortKey> = self.levels.keys().copied().collect();

        for key in keys {
            if remaining <= Decimal::ZERO || levels_consumed >= max {
                break;
            }

            let level_price = match self.levels.get(&key) {
                Some(level) => level.price,
                None => continue,
            };

            // Check limit price constraint.
            if let Some(limit) = limit_price {
                match self.side {
                    Side::Sell => {
                        // Sweeping asks (buy side) — only consume if ask ≤ limit.
                        if level_price > limit {
                            break; // All subsequent levels are worse.
                        }
                    }
                    Side::Buy => {
                        // Sweeping bids (sell side) — only consume if bid ≥ limit.
                        if level_price < limit {
                            break;
                        }
                    }
                }
            }

            let level = self.levels.get_mut(&key).unwrap();
            let (filled, level_remaining) = level.consume(remaining);

            if filled > Decimal::ZERO {
                fills.push((level_price, filled));
                remaining -= filled;
                self.total_quantity -= filled;
                levels_consumed += 1;
            }

            // Remove the level if fully consumed.
            if level_remaining <= Decimal::ZERO {
                self.levels.remove(&key);
                self.level_count -= 1;
            }
        }

        fills
    }

    /// Total quantity available within `bps` basis points of the best price.
    ///
    /// Useful for assessing available liquidity before placing an order.
    pub fn depth_at_bps(&self, bps: u32) -> Decimal {
        let best = match self.best_price() {
            Some(p) => p,
            None => return Decimal::ZERO,
        };

        let bps_frac = Decimal::from(bps) / Decimal::from(10_000);
        let threshold = best * bps_frac;

        let mut total = Decimal::ZERO;
        for level in self.iter() {
            let distance = match self.side {
                Side::Buy => best - level.price,  // bids: best is highest
                Side::Sell => level.price - best, // asks: best is lowest
            };
            if distance > threshold {
                break; // Levels are sorted best-to-worst, so we can stop.
            }
            total += level.quantity;
        }
        total
    }

    /// Compute the order book imbalance ratio against the opposite side.
    ///
    /// Returns `(self_depth - other_depth) / (self_depth + other_depth)` using
    /// the top `n_levels` from each side. Result is in `[-1, 1]`:
    /// - Positive: this side has more depth (bullish if this is bid side).
    /// - Negative: the other side has more depth.
    pub fn imbalance(&self, other: &BookSide, n_levels: usize) -> f64 {
        let self_depth: f64 = self
            .iter()
            .take(n_levels)
            .map(|l| -> f64 { l.quantity.try_into().unwrap_or(0.0) })
            .sum();

        let other_depth: f64 = other
            .iter()
            .take(n_levels)
            .map(|l| -> f64 { l.quantity.try_into().unwrap_or(0.0) })
            .sum();

        let total = self_depth + other_depth;
        if total <= 0.0 {
            0.0
        } else {
            (self_depth - other_depth) / total
        }
    }

    /// Add a resting order to the book at the given price.
    ///
    /// If a level at the price already exists, the order is appended to its
    /// L3 queue. Otherwise, a new level is created. The aggregated quantity
    /// and level count are kept consistent.
    pub fn add_resting_order(&mut self, order: RestingOrder) -> crate::orderbook::Result<()> {
        let price = order.price;
        if price <= Decimal::ZERO {
            return Err(OrderBookError::InvalidPrice(price));
        }

        let key = PriceSortKey::for_side(price, self.side);

        if let Some(level) = self.levels.get_mut(&key) {
            self.total_quantity += order.remaining_quantity;
            level.add_order(order);
        } else {
            let qty = order.remaining_quantity;
            let mut level = PriceLevel::new(price, Decimal::ZERO);
            level.quantity = Decimal::ZERO; // will be set by add_order
            level.order_count = 0;
            level.add_order(order);
            self.levels.insert(key, level);
            self.level_count += 1;
            self.total_quantity += qty;
        }

        Ok(())
    }

    /// Cancel a resting order from the book by order ID.
    ///
    /// Searches all levels for the order. If found, removes it and adjusts
    /// the level quantity. If the level becomes empty, it is removed entirely.
    ///
    /// Returns the cancelled order if found.
    pub fn cancel_resting_order(&mut self, order_id: &str) -> Option<RestingOrder> {
        // We need to find which level the order is in.
        let mut found_key: Option<PriceSortKey> = None;
        let mut result: Option<RestingOrder> = None;

        for (&key, level) in self.levels.iter_mut() {
            if let Some(removed) = level.cancel_order(order_id) {
                self.total_quantity -= removed.remaining_quantity;
                result = Some(removed);
                if level.is_empty() && level.orders.is_empty() {
                    found_key = Some(key);
                }
                break;
            }
        }

        // Remove empty level outside the mutable borrow.
        if let Some(key) = found_key {
            self.levels.remove(&key);
            self.level_count -= 1;
        }

        result
    }

    /// Modify a resting order's quantity by order ID.
    ///
    /// See [`PriceLevel::modify_order`] for semantics.
    /// Returns `true` if the order was found and modified.
    pub fn modify_resting_order(&mut self, order_id: &str, new_qty: Decimal) -> bool {
        for level in self.levels.values_mut() {
            let old_qty: Decimal = level
                .orders
                .iter()
                .find(|o| o.id == order_id)
                .map(|o| o.remaining_quantity)
                .unwrap_or(Decimal::ZERO);

            if old_qty > Decimal::ZERO || level.orders.iter().any(|o| o.id == order_id) {
                let delta = new_qty - old_qty;
                if level.modify_order(order_id, new_qty) {
                    self.total_quantity += delta;
                    return true;
                }
            }
        }
        false
    }
}

impl fmt::Display for BookSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} side ({} levels, depth: {}):",
            self.side, self.level_count, self.total_quantity
        )?;
        for (i, level) in self.iter().enumerate().take(5) {
            writeln!(f, "  [{}] {}", i + 1, level)?;
        }
        if self.level_count > 5 {
            writeln!(f, "  ... and {} more levels", self.level_count - 5)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Order Book Snapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of the order book, typically received from an
/// exchange L2 data feed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    /// Trading symbol.
    pub symbol: String,
    /// Bid price levels (should be sorted descending by price).
    pub bids: Vec<PriceLevel>,
    /// Ask price levels (should be sorted ascending by price).
    pub asks: Vec<PriceLevel>,
    /// Snapshot timestamp (exchange time).
    pub timestamp: DateTime<Utc>,
    /// Sequence number for ordering.
    pub sequence: u64,
}

impl OrderBookSnapshot {
    /// Create an empty snapshot.
    pub fn empty(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp: Utc::now(),
            sequence: 0,
        }
    }

    /// Total number of price levels in the snapshot.
    pub fn total_levels(&self) -> usize {
        self.bids.len() + self.asks.len()
    }

    /// Check whether the snapshot is empty.
    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// Mid price from the snapshot (if both sides present).
    pub fn mid_price(&self) -> Option<Decimal> {
        let best_bid = self.bids.first().map(|l| l.price)?;
        let best_ask = self.asks.first().map(|l| l.price)?;
        Some((best_bid + best_ask) / Decimal::from(2))
    }

    /// Spread from the snapshot.
    pub fn spread(&self) -> Option<Decimal> {
        let best_bid = self.bids.first().map(|l| l.price)?;
        let best_ask = self.asks.first().map(|l| l.price)?;
        Some(best_ask - best_bid)
    }
}

impl fmt::Display for OrderBookSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Snapshot[{}] {} bids, {} asks, seq={}",
            self.symbol,
            self.bids.len(),
            self.asks.len(),
            self.sequence,
        )?;
        if let Some(mid) = self.mid_price() {
            write!(f, ", mid={}", mid)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Order Book Delta
// ---------------------------------------------------------------------------

/// An incremental update to the order book (L2 delta message from exchange).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookDelta {
    /// Trading symbol.
    pub symbol: String,
    /// Side being updated.
    pub side: Side,
    /// Price of the level being updated.
    pub price: Decimal,
    /// New quantity at this level (0 = remove level).
    pub quantity: Decimal,
    /// Timestamp of the update.
    pub timestamp: DateTime<Utc>,
    /// Sequence number.
    pub sequence: u64,
}

/// A batch of incremental order book deltas to be applied atomically.
///
/// Many exchanges send bid and ask updates together in a single WebSocket
/// message. Applying them as a batch ensures the book is never observed in
/// an intermediate (potentially crossed) state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookDeltaBatch {
    /// Trading symbol.
    pub symbol: String,
    /// The individual deltas in this batch.
    pub deltas: Vec<OrderBookDelta>,
    /// Sequence number for the entire batch.
    pub sequence: u64,
    /// Timestamp of the batch.
    pub timestamp: DateTime<Utc>,
}

impl OrderBookDeltaBatch {
    /// Create a new empty batch.
    pub fn new(symbol: impl Into<String>, sequence: u64) -> Self {
        Self {
            symbol: symbol.into(),
            deltas: Vec::new(),
            sequence,
            timestamp: Utc::now(),
        }
    }

    /// Add a delta to the batch.
    pub fn add(&mut self, side: Side, price: Decimal, quantity: Decimal) {
        self.deltas.push(OrderBookDelta {
            symbol: self.symbol.clone(),
            side,
            price,
            quantity,
            timestamp: self.timestamp,
            sequence: self.sequence,
        });
    }

    /// Number of deltas in the batch.
    pub fn len(&self) -> usize {
        self.deltas.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Order Book Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the order book.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderBookStats {
    /// Total number of snapshots applied.
    pub snapshots_applied: u64,
    /// Total number of deltas applied.
    pub deltas_applied: u64,
    /// Number of crossed-book events detected (data errors).
    pub crossed_book_events: u64,
    /// Number of stale/out-of-order updates dropped.
    pub stale_updates_dropped: u64,
    /// Last update timestamp.
    pub last_update: Option<DateTime<Utc>>,
    /// Last sequence number processed.
    pub last_sequence: u64,
}

impl OrderBookStats {
    /// Total updates processed.
    pub fn total_updates(&self) -> u64 {
        self.snapshots_applied + self.deltas_applied
    }
}

impl fmt::Display for OrderBookStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OrderBookStats {{ snapshots: {}, deltas: {}, crossed: {}, stale: {} }}",
            self.snapshots_applied,
            self.deltas_applied,
            self.crossed_book_events,
            self.stale_updates_dropped,
        )
    }
}

// ---------------------------------------------------------------------------
// Order Book
// ---------------------------------------------------------------------------

/// A full limit order book for a single trading symbol.
///
/// Maintains sorted bid (descending) and ask (ascending) sides with
/// O(log n) insert/update/delete via `BTreeMap`.
///
/// Supports two modes:
/// - **L2 mode** (default): Aggregated price levels, quantity per price.
/// - **L3 mode**: Individual orders tracked within each price level (FIFO queue).
///
/// The book validates data integrity (no crossed book, no negative prices)
/// and maintains running statistics.
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Trading symbol.
    symbol: String,
    /// Bid side (buy orders, best = highest price).
    bids: BookSide,
    /// Ask side (sell orders, best = lowest price).
    asks: BookSide,
    /// Whether L3 order tracking is enabled.
    l3_enabled: bool,
    /// Running statistics.
    stats: OrderBookStats,
    /// Maximum depth (number of levels) to retain per side.
    /// `None` means unlimited.
    max_depth: Option<usize>,
}

impl OrderBook {
    /// Create a new empty order book for the given symbol.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bids: BookSide::new(Side::Buy),
            asks: BookSide::new(Side::Sell),
            l3_enabled: false,
            stats: OrderBookStats::default(),
            max_depth: None,
        }
    }

    /// Create a new order book with L3 order tracking enabled.
    pub fn with_l3(symbol: impl Into<String>) -> Self {
        let mut book = Self::new(symbol);
        book.l3_enabled = true;
        book
    }

    /// Create a new order book with a maximum depth per side.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Get the trading symbol.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get the bid side.
    pub fn bids(&self) -> &BookSide {
        &self.bids
    }

    /// Get the ask side.
    pub fn asks(&self) -> &BookSide {
        &self.asks
    }

    /// Get mutable bid side.
    pub fn bids_mut(&mut self) -> &mut BookSide {
        &mut self.bids
    }

    /// Get mutable ask side.
    pub fn asks_mut(&mut self) -> &mut BookSide {
        &mut self.asks
    }

    /// Get the running statistics.
    pub fn stats(&self) -> &OrderBookStats {
        &self.stats
    }

    /// Whether L3 order tracking is enabled.
    pub fn is_l3(&self) -> bool {
        self.l3_enabled
    }

    // ── Best prices ────────────────────────────────────────────────────

    /// Best bid (highest buy price).
    pub fn best_bid(&self) -> Option<&PriceLevel> {
        self.bids.best()
    }

    /// Best ask (lowest sell price).
    pub fn best_ask(&self) -> Option<&PriceLevel> {
        self.asks.best()
    }

    /// Best bid price.
    pub fn best_bid_price(&self) -> Option<Decimal> {
        self.bids.best_price()
    }

    /// Best ask price.
    pub fn best_ask_price(&self) -> Option<Decimal> {
        self.asks.best_price()
    }

    /// Mid price: (best_bid + best_ask) / 2.
    pub fn mid_price(&self) -> Option<Decimal> {
        let bid = self.best_bid_price()?;
        let ask = self.best_ask_price()?;
        Some((bid + ask) / Decimal::from(2))
    }

    /// Spread: best_ask - best_bid.
    pub fn spread(&self) -> Option<Decimal> {
        let bid = self.best_bid_price()?;
        let ask = self.best_ask_price()?;
        Some(ask - bid)
    }

    /// Spread in basis points relative to mid price.
    pub fn spread_bps(&self) -> Option<Decimal> {
        let spread = self.spread()?;
        let mid = self.mid_price()?;
        if mid > Decimal::ZERO {
            Some(spread / mid * Decimal::from(10_000))
        } else {
            None
        }
    }

    /// Whether the book is crossed (best_bid ≥ best_ask).
    /// This indicates a data integrity problem.
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid_price(), self.best_ask_price()) {
            (Some(bid), Some(ask)) => bid >= ask,
            _ => false,
        }
    }

    /// Whether the book is empty on both sides.
    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// Total number of price levels across both sides.
    pub fn total_levels(&self) -> usize {
        self.bids.len() + self.asks.len()
    }

    // ── Mutations ──────────────────────────────────────────────────────

    /// Apply a full L2 snapshot, replacing the current book state.
    pub fn apply_snapshot(&mut self, snapshot: OrderBookSnapshot) -> Result<()> {
        // Stale check.
        if snapshot.sequence > 0
            && snapshot.sequence <= self.stats.last_sequence
            && self.stats.snapshots_applied > 0
        {
            self.stats.stale_updates_dropped += 1;
            warn!(
                symbol = %self.symbol,
                received = snapshot.sequence,
                expected = self.stats.last_sequence + 1,
                "Stale snapshot dropped"
            );
            return Err(OrderBookError::StaleSnapshot {
                received: snapshot.sequence,
                expected: self.stats.last_sequence,
            });
        }

        self.bids.clear();
        self.asks.clear();

        for level in &snapshot.bids {
            self.bids.set_level(level.price, level.quantity)?;
        }
        for level in &snapshot.asks {
            self.asks.set_level(level.price, level.quantity)?;
        }

        // Optionally truncate to max_depth.
        if let Some(max) = self.max_depth {
            self.truncate_depth(max);
        }

        self.stats.snapshots_applied += 1;
        self.stats.last_update = Some(snapshot.timestamp);
        self.stats.last_sequence = snapshot.sequence;

        // Warn on crossed book.
        if self.is_crossed() {
            self.stats.crossed_book_events += 1;
            let bid = self.best_bid_price().unwrap_or_default();
            let ask = self.best_ask_price().unwrap_or_default();
            warn!(
                symbol = %self.symbol,
                bid = %bid,
                ask = %ask,
                "Crossed book after snapshot"
            );
        }

        debug!(
            symbol = %self.symbol,
            bids = self.bids.len(),
            asks = self.asks.len(),
            seq = snapshot.sequence,
            "Applied order book snapshot"
        );

        Ok(())
    }

    /// Apply an incremental L2 delta update.
    pub fn apply_delta(&mut self, delta: OrderBookDelta) -> Result<()> {
        // Stale check.
        if delta.sequence > 0 && delta.sequence <= self.stats.last_sequence {
            self.stats.stale_updates_dropped += 1;
            return Err(OrderBookError::StaleSnapshot {
                received: delta.sequence,
                expected: self.stats.last_sequence,
            });
        }

        let side = match delta.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        side.set_level(delta.price, delta.quantity)?;

        self.stats.deltas_applied += 1;
        self.stats.last_update = Some(delta.timestamp);
        if delta.sequence > 0 {
            self.stats.last_sequence = delta.sequence;
        }

        // Check for crossed book after update.
        if self.is_crossed() {
            self.stats.crossed_book_events += 1;
        }

        Ok(())
    }

    /// Set a price level directly on a specific side.
    pub fn set_level(&mut self, side: Side, price: Decimal, quantity: Decimal) -> Result<()> {
        match side {
            Side::Buy => self.bids.set_level(price, quantity),
            Side::Sell => self.asks.set_level(price, quantity),
        }
    }

    /// Clear the entire book.
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    /// Reset the book and statistics.
    pub fn reset(&mut self) {
        self.clear();
        self.stats = OrderBookStats::default();
    }

    // ── Depth analysis ─────────────────────────────────────────────────

    /// Get the VWAP for buying `quantity` units (sweeping the ask side).
    pub fn buy_vwap(&self, quantity: Decimal) -> Option<Decimal> {
        self.asks.vwap_for_depth(quantity)
    }

    /// Get the VWAP for selling `quantity` units (sweeping the bid side).
    pub fn sell_vwap(&self, quantity: Decimal) -> Option<Decimal> {
        self.bids.vwap_for_depth(quantity)
    }

    /// Take a snapshot of the current book state.
    pub fn snapshot(&self) -> OrderBookSnapshot {
        OrderBookSnapshot {
            symbol: self.symbol.clone(),
            bids: self.bids.iter().cloned().collect(),
            asks: self.asks.iter().cloned().collect(),
            timestamp: self.stats.last_update.unwrap_or_else(Utc::now),
            sequence: self.stats.last_sequence,
        }
    }

    /// Get the top `n` levels from each side.
    pub fn top_of_book(&self, n: usize) -> (Vec<&PriceLevel>, Vec<&PriceLevel>) {
        (self.bids.top_n(n), self.asks.top_n(n))
    }

    // ── Internal helpers ───────────────────────────────────────────────

    /// Truncate both sides to at most `max` levels.
    fn truncate_depth(&mut self, max: usize) {
        // BookSide uses a BTreeMap so we can't easily truncate in place.
        // For now we rebuild if over limit.

        if self.bids.len() > max {
            let keep: Vec<(Decimal, Decimal)> = self
                .bids
                .iter()
                .take(max)
                .map(|l| (l.price, l.quantity))
                .collect();
            self.bids.clear();
            for (p, q) in keep {
                let _ = self.bids.set_level(p, q);
            }
        }
        if self.asks.len() > max {
            let keep: Vec<(Decimal, Decimal)> = self
                .asks
                .iter()
                .take(max)
                .map(|l| (l.price, l.quantity))
                .collect();
            self.asks.clear();
            for (p, q) in keep {
                let _ = self.asks.set_level(p, q);
            }
        }
    }

    /// Compute the order book imbalance over the top `depth` levels.
    ///
    /// Returns `(bid_volume - ask_volume) / (bid_volume + ask_volume)`:
    /// - Positive: more bid depth (bullish pressure).
    /// - Negative: more ask depth (bearish pressure).
    /// - Zero: balanced or empty book.
    pub fn order_book_imbalance(&self, depth: usize) -> f64 {
        self.bids.imbalance(&self.asks, depth)
    }

    /// Volume-weighted mid price.
    ///
    /// Computed as `(bid_price × ask_qty + ask_price × bid_qty) / (bid_qty + ask_qty)`.
    /// More responsive to order book imbalance than the simple mid price — when bid
    /// depth is much larger than ask depth, the weighted mid shifts towards the ask
    /// (reflecting higher buying pressure).
    pub fn weighted_mid_price(&self) -> Option<Decimal> {
        let best_bid = self.best_bid()?;
        let best_ask = self.best_ask()?;
        let total = best_bid.quantity + best_ask.quantity;
        if total <= Decimal::ZERO {
            return None;
        }
        Some((best_bid.price * best_ask.quantity + best_ask.price * best_bid.quantity) / total)
    }

    /// Compute the cost of executing a market order of size `qty` against the
    /// current book, expressed in basis points relative to the mid price.
    ///
    /// For a buy, this sweeps the ask side; for a sell, the bid side. The
    /// impact cost is defined as `(vwap − mid) / mid × 10_000` for buys and
    /// `(mid − vwap) / mid × 10_000` for sells.
    ///
    /// Returns `None` if the mid price is unavailable or the book lacks
    /// liquidity on the relevant side.
    pub fn impact_cost(&self, side: Side, qty: Decimal) -> Option<Decimal> {
        let mid = self.mid_price()?;
        if mid <= Decimal::ZERO || qty <= Decimal::ZERO {
            return None;
        }
        let vwap = match side {
            Side::Buy => self.asks.vwap_for_depth(qty)?,
            Side::Sell => self.bids.vwap_for_depth(qty)?,
        };
        let impact = match side {
            Side::Buy => (vwap - mid) / mid * Decimal::from(10_000),
            Side::Sell => (mid - vwap) / mid * Decimal::from(10_000),
        };
        Some(impact)
    }

    /// Apply a batch of deltas atomically.
    ///
    /// All deltas in the batch are applied before any crossed-book or
    /// statistics checks, ensuring the book is never observed in an
    /// intermediate state.
    pub fn apply_delta_batch(&mut self, batch: OrderBookDeltaBatch) -> Result<()> {
        // Stale check on the batch sequence.
        if batch.sequence > 0 && batch.sequence <= self.stats.last_sequence {
            self.stats.stale_updates_dropped += 1;
            return Err(OrderBookError::StaleSnapshot {
                received: batch.sequence,
                expected: self.stats.last_sequence,
            });
        }

        for delta in &batch.deltas {
            let side = match delta.side {
                Side::Buy => &mut self.bids,
                Side::Sell => &mut self.asks,
            };
            side.set_level(delta.price, delta.quantity)?;
        }

        self.stats.deltas_applied += batch.deltas.len() as u64;
        self.stats.last_update = Some(batch.timestamp);
        if batch.sequence > 0 {
            self.stats.last_sequence = batch.sequence;
        }

        if self.is_crossed() {
            self.stats.crossed_book_events += 1;
        }

        Ok(())
    }
}

impl fmt::Display for OrderBook {
    /// Pretty-print the order book in a market-depth ladder format.
    ///
    /// Asks are displayed in descending order (worst to best from top),
    /// followed by a separator with mid/spread info, then bids in
    /// descending order (best to worst from the separator).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max_display = 10;
        writeln!(f, "╔══════════════════════════════════════════╗")?;
        writeln!(f, "║  OrderBook: {:^28} ║", self.symbol)?;
        writeln!(f, "╠══════════════════════════════════════════╣")?;

        // Collect ask levels (up to max_display), then reverse so worst is on top.
        let ask_levels: Vec<&PriceLevel> = self.asks.iter().take(max_display).collect();
        if ask_levels.is_empty() {
            writeln!(f, "║  (no asks)                               ║")?;
        } else {
            for level in ask_levels.iter().rev() {
                writeln!(
                    f,
                    "║  ASK  {:>12}  × {:>12}        ║",
                    level.price, level.quantity
                )?;
            }
        }

        // Separator with mid/spread
        writeln!(f, "╠──────────────────────────────────────────╣")?;
        if let (Some(mid), Some(spread)) = (self.mid_price(), self.spread()) {
            writeln!(
                f,
                "║  Mid: {}  Spread: {}{}",
                mid,
                spread,
                if let Some(bps) = self.spread_bps() {
                    format!(" ({:.1} bps)", bps)
                } else {
                    String::new()
                }
            )?;
        }
        writeln!(f, "╠──────────────────────────────────────────╣")?;

        // Bid levels (best to worst, top to bottom).
        let bid_levels: Vec<&PriceLevel> = self.bids.iter().take(max_display).collect();
        if bid_levels.is_empty() {
            writeln!(f, "║  (no bids)                               ║")?;
        } else {
            for level in &bid_levels {
                writeln!(
                    f,
                    "║  BID  {:>12}  × {:>12}        ║",
                    level.price, level.quantity
                )?;
            }
        }

        writeln!(f, "╠══════════════════════════════════════════╣")?;
        writeln!(
            f,
            "║  Bids: {} levels ({})  Asks: {} levels ({}) ║",
            self.bids.len(),
            self.bids.total_depth(),
            self.asks.len(),
            self.asks.total_depth(),
        )?;
        writeln!(f, "║  {}", self.stats)?;
        write!(f, "╚══════════════════════════════════════════╝")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn sample_snapshot() -> OrderBookSnapshot {
        OrderBookSnapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![
                PriceLevel::new(dec!(67000.0), dec!(1.5)),
                PriceLevel::new(dec!(66999.0), dec!(3.2)),
                PriceLevel::new(dec!(66998.5), dec!(0.75)),
            ],
            asks: vec![
                PriceLevel::new(dec!(67001.0), dec!(0.8)),
                PriceLevel::new(dec!(67002.0), dec!(2.1)),
                PriceLevel::new(dec!(67003.0), dec!(1.3)),
            ],
            timestamp: Utc::now(),
            sequence: 1,
        }
    }

    // ── PriceLevel tests ───────────────────────────────────────────────

    #[test]
    fn test_price_level_new() {
        let level = PriceLevel::new(dec!(100.0), dec!(5.0));
        assert_eq!(level.price, dec!(100.0));
        assert_eq!(level.quantity, dec!(5.0));
        assert_eq!(level.order_count, 1);
        assert!(!level.is_empty());
    }

    #[test]
    fn test_price_level_notional() {
        let level = PriceLevel::new(dec!(50000.0), dec!(2.0));
        assert_eq!(level.notional(), dec!(100000.0));
    }

    #[test]
    fn test_price_level_empty() {
        let level = PriceLevel::new(dec!(100.0), dec!(0.0));
        assert!(level.is_empty());
    }

    #[test]
    fn test_price_level_display() {
        let level = PriceLevel::with_order_count(dec!(100.0), dec!(5.0), 3);
        let display = format!("{}", level);
        assert!(display.contains("100"));
        assert!(display.contains("5"));
        assert!(display.contains("3 orders"));
    }

    // ── RestingOrder tests ─────────────────────────────────────────────

    #[test]
    fn test_resting_order_new() {
        let order = RestingOrder::new(Side::Buy, dec!(100.0), dec!(5.0), Utc::now());
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.price, dec!(100.0));
        assert_eq!(order.remaining_quantity, dec!(5.0));
        assert_eq!(order.original_quantity, dec!(5.0));
        assert!(!order.is_own);
        assert!(!order.is_filled());
    }

    #[test]
    fn test_resting_order_own() {
        let order = RestingOrder::own(Side::Sell, dec!(200.0), dec!(1.0), Utc::now());
        assert!(order.is_own);
    }

    #[test]
    fn test_resting_order_fill_ratio() {
        let mut order = RestingOrder::new(Side::Buy, dec!(100.0), dec!(10.0), Utc::now());
        assert_eq!(order.fill_ratio(), dec!(0.0));
        order.remaining_quantity = dec!(3.0);
        assert_eq!(order.fill_ratio(), dec!(0.7));
    }

    #[test]
    fn test_resting_order_is_filled() {
        let mut order = RestingOrder::new(Side::Buy, dec!(100.0), dec!(5.0), Utc::now());
        assert!(!order.is_filled());
        order.remaining_quantity = dec!(0.0);
        assert!(order.is_filled());
    }

    // ── BookSide tests ─────────────────────────────────────────────────

    #[test]
    fn test_book_side_new() {
        let side = BookSide::new(Side::Buy);
        assert!(side.is_empty());
        assert_eq!(side.len(), 0);
        assert_eq!(side.total_depth(), Decimal::ZERO);
        assert!(side.best().is_none());
    }

    #[test]
    fn test_book_side_set_level() {
        let mut side = BookSide::new(Side::Buy);
        side.set_level(dec!(100.0), dec!(5.0)).unwrap();
        assert_eq!(side.len(), 1);
        assert_eq!(side.total_depth(), dec!(5.0));
        assert_eq!(side.best_price(), Some(dec!(100.0)));
    }

    #[test]
    fn test_book_side_bid_ordering() {
        let mut bids = BookSide::new(Side::Buy);
        bids.set_level(dec!(99.0), dec!(1.0)).unwrap();
        bids.set_level(dec!(101.0), dec!(2.0)).unwrap();
        bids.set_level(dec!(100.0), dec!(3.0)).unwrap();

        // Best bid should be highest price.
        assert_eq!(bids.best_price(), Some(dec!(101.0)));

        let prices: Vec<Decimal> = bids.iter().map(|l| l.price).collect();
        assert_eq!(prices, vec![dec!(101.0), dec!(100.0), dec!(99.0)]);
    }

    #[test]
    fn test_book_side_ask_ordering() {
        let mut asks = BookSide::new(Side::Sell);
        asks.set_level(dec!(101.0), dec!(2.0)).unwrap();
        asks.set_level(dec!(99.0), dec!(1.0)).unwrap();
        asks.set_level(dec!(100.0), dec!(3.0)).unwrap();

        // Best ask should be lowest price.
        assert_eq!(asks.best_price(), Some(dec!(99.0)));

        let prices: Vec<Decimal> = asks.iter().map(|l| l.price).collect();
        assert_eq!(prices, vec![dec!(99.0), dec!(100.0), dec!(101.0)]);
    }

    #[test]
    fn test_book_side_update_level() {
        let mut side = BookSide::new(Side::Buy);
        side.set_level(dec!(100.0), dec!(5.0)).unwrap();
        assert_eq!(side.total_depth(), dec!(5.0));

        // Update quantity.
        side.set_level(dec!(100.0), dec!(8.0)).unwrap();
        assert_eq!(side.len(), 1); // Still one level.
        assert_eq!(side.total_depth(), dec!(8.0));
    }

    #[test]
    fn test_book_side_remove_level_via_zero_qty() {
        let mut side = BookSide::new(Side::Buy);
        side.set_level(dec!(100.0), dec!(5.0)).unwrap();
        assert_eq!(side.len(), 1);

        // Setting quantity to 0 removes the level.
        side.set_level(dec!(100.0), dec!(0.0)).unwrap();
        assert_eq!(side.len(), 0);
        assert!(side.is_empty());
    }

    #[test]
    fn test_book_side_remove_level() {
        let mut side = BookSide::new(Side::Buy);
        side.set_level(dec!(100.0), dec!(5.0)).unwrap();
        let removed = side.remove_level(dec!(100.0));
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().quantity, dec!(5.0));
        assert!(side.is_empty());
    }

    #[test]
    fn test_book_side_invalid_price() {
        let mut side = BookSide::new(Side::Buy);
        assert!(side.set_level(dec!(0.0), dec!(5.0)).is_err());
        assert!(side.set_level(dec!(-1.0), dec!(5.0)).is_err());
    }

    #[test]
    fn test_book_side_invalid_quantity() {
        let mut side = BookSide::new(Side::Buy);
        assert!(side.set_level(dec!(100.0), dec!(-1.0)).is_err());
    }

    #[test]
    fn test_book_side_vwap() {
        let mut asks = BookSide::new(Side::Sell);
        asks.set_level(dec!(100.0), dec!(2.0)).unwrap();
        asks.set_level(dec!(101.0), dec!(3.0)).unwrap();

        // Buy 2 units → all at 100.
        assert_eq!(asks.vwap_for_depth(dec!(2.0)), Some(dec!(100.0)));

        // Buy 5 units → 2 @ 100 + 3 @ 101 = 503 / 5 = 100.6.
        assert_eq!(asks.vwap_for_depth(dec!(5.0)), Some(dec!(100.6)));

        // Buy 1 unit → all at 100.
        assert_eq!(asks.vwap_for_depth(dec!(1.0)), Some(dec!(100.0)));
    }

    #[test]
    fn test_book_side_vwap_empty() {
        let side = BookSide::new(Side::Sell);
        assert_eq!(side.vwap_for_depth(dec!(1.0)), None);
    }

    #[test]
    fn test_book_side_top_n() {
        let mut bids = BookSide::new(Side::Buy);
        bids.set_level(dec!(100.0), dec!(1.0)).unwrap();
        bids.set_level(dec!(99.0), dec!(2.0)).unwrap();
        bids.set_level(dec!(98.0), dec!(3.0)).unwrap();

        let top2 = bids.top_n(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].price, dec!(100.0));
        assert_eq!(top2[1].price, dec!(99.0));
    }

    #[test]
    fn test_book_side_total_notional() {
        let mut asks = BookSide::new(Side::Sell);
        asks.set_level(dec!(100.0), dec!(2.0)).unwrap();
        asks.set_level(dec!(200.0), dec!(1.0)).unwrap();
        // 100*2 + 200*1 = 400
        assert_eq!(asks.total_notional(), dec!(400.0));
    }

    #[test]
    fn test_book_side_clear() {
        let mut side = BookSide::new(Side::Buy);
        side.set_level(dec!(100.0), dec!(5.0)).unwrap();
        side.set_level(dec!(99.0), dec!(3.0)).unwrap();
        side.clear();
        assert!(side.is_empty());
        assert_eq!(side.total_depth(), Decimal::ZERO);
    }

    // ── OrderBookSnapshot tests ────────────────────────────────────────

    #[test]
    fn test_snapshot_empty() {
        let snap = OrderBookSnapshot::empty("BTC/USDT");
        assert!(snap.is_empty());
        assert_eq!(snap.total_levels(), 0);
        assert!(snap.mid_price().is_none());
    }

    #[test]
    fn test_snapshot_mid_price() {
        let snap = sample_snapshot();
        let mid = snap.mid_price().unwrap();
        // (67000 + 67001) / 2 = 67000.5
        assert_eq!(mid, dec!(67000.5));
    }

    #[test]
    fn test_snapshot_spread() {
        let snap = sample_snapshot();
        let spread = snap.spread().unwrap();
        assert_eq!(spread, dec!(1.0));
    }

    #[test]
    fn test_snapshot_display() {
        let snap = sample_snapshot();
        let display = format!("{}", snap);
        assert!(display.contains("BTC/USDT"));
        assert!(display.contains("3 bids"));
        assert!(display.contains("3 asks"));
    }

    // ── OrderBook tests ────────────────────────────────────────────────

    #[test]
    fn test_orderbook_new() {
        let book = OrderBook::new("BTC/USDT");
        assert_eq!(book.symbol(), "BTC/USDT");
        assert!(book.is_empty());
        assert!(!book.is_l3());
        assert!(!book.is_crossed());
    }

    #[test]
    fn test_orderbook_with_l3() {
        let book = OrderBook::with_l3("ETH/USDT");
        assert!(book.is_l3());
    }

    #[test]
    fn test_orderbook_apply_snapshot() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        assert_eq!(book.bids().len(), 3);
        assert_eq!(book.asks().len(), 3);
        assert_eq!(book.total_levels(), 6);
        assert_eq!(book.best_bid_price(), Some(dec!(67000.0)));
        assert_eq!(book.best_ask_price(), Some(dec!(67001.0)));
        assert_eq!(book.mid_price(), Some(dec!(67000.5)));
        assert_eq!(book.spread(), Some(dec!(1.0)));
        assert!(!book.is_crossed());
        assert_eq!(book.stats().snapshots_applied, 1);
    }

    #[test]
    fn test_orderbook_apply_delta() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        // Update best ask quantity.
        let delta = OrderBookDelta {
            symbol: "BTC/USDT".into(),
            side: Side::Sell,
            price: dec!(67001.0),
            quantity: dec!(5.0),
            timestamp: Utc::now(),
            sequence: 2,
        };
        book.apply_delta(delta).unwrap();

        let ask = book.best_ask().unwrap();
        assert_eq!(ask.quantity, dec!(5.0));
        assert_eq!(book.stats().deltas_applied, 1);
    }

    #[test]
    fn test_orderbook_delta_remove_level() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();
        assert_eq!(book.asks().len(), 3);

        // Remove a level by setting qty to 0.
        let delta = OrderBookDelta {
            symbol: "BTC/USDT".into(),
            side: Side::Sell,
            price: dec!(67003.0),
            quantity: dec!(0.0),
            timestamp: Utc::now(),
            sequence: 2,
        };
        book.apply_delta(delta).unwrap();
        assert_eq!(book.asks().len(), 2);
    }

    #[test]
    fn test_orderbook_stale_snapshot() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        // Same sequence should fail.
        let result = book.apply_snapshot(sample_snapshot());
        assert!(result.is_err());
        assert_eq!(book.stats().stale_updates_dropped, 1);
    }

    #[test]
    fn test_orderbook_stale_delta() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        let delta = OrderBookDelta {
            symbol: "BTC/USDT".into(),
            side: Side::Buy,
            price: dec!(66999.0),
            quantity: dec!(10.0),
            timestamp: Utc::now(),
            sequence: 1, // Same as snapshot.
        };
        assert!(book.apply_delta(delta).is_err());
    }

    #[test]
    fn test_orderbook_spread_bps() {
        let mut book = OrderBook::new("TEST");
        book.set_level(Side::Buy, dec!(100.0), dec!(1.0)).unwrap();
        book.set_level(Side::Sell, dec!(100.10), dec!(1.0)).unwrap();

        let bps = book.spread_bps().unwrap();
        // spread = 0.10, mid = 100.05, bps = 0.10/100.05 * 10000 ≈ 9.995
        assert!(bps > dec!(9.0) && bps < dec!(11.0));
    }

    #[test]
    fn test_orderbook_buy_vwap() {
        let mut book = OrderBook::new("TEST");
        book.set_level(Side::Sell, dec!(100.0), dec!(2.0)).unwrap();
        book.set_level(Side::Sell, dec!(101.0), dec!(3.0)).unwrap();

        let vwap = book.buy_vwap(dec!(5.0)).unwrap();
        // 2@100 + 3@101 = 503 / 5 = 100.6
        assert_eq!(vwap, dec!(100.6));
    }

    #[test]
    fn test_orderbook_sell_vwap() {
        let mut book = OrderBook::new("TEST");
        book.set_level(Side::Buy, dec!(100.0), dec!(3.0)).unwrap();
        book.set_level(Side::Buy, dec!(99.0), dec!(2.0)).unwrap();

        let vwap = book.sell_vwap(dec!(5.0)).unwrap();
        // 3@100 + 2@99 = 498 / 5 = 99.6
        assert_eq!(vwap, dec!(99.6));
    }

    #[test]
    fn test_orderbook_snapshot_roundtrip() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        let snap = book.snapshot();
        assert_eq!(snap.bids.len(), 3);
        assert_eq!(snap.asks.len(), 3);
        assert_eq!(snap.symbol, "BTC/USDT");
    }

    #[test]
    fn test_orderbook_top_of_book() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        let (top_bids, top_asks) = book.top_of_book(2);
        assert_eq!(top_bids.len(), 2);
        assert_eq!(top_asks.len(), 2);
        assert_eq!(top_bids[0].price, dec!(67000.0));
        assert_eq!(top_asks[0].price, dec!(67001.0));
    }

    #[test]
    fn test_orderbook_max_depth() {
        let mut book = OrderBook::new("BTC/USDT").with_max_depth(2);
        book.apply_snapshot(sample_snapshot()).unwrap();

        // Should be truncated to 2 levels per side.
        assert_eq!(book.bids().len(), 2);
        assert_eq!(book.asks().len(), 2);
    }

    #[test]
    fn test_orderbook_clear() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();
        book.clear();
        assert!(book.is_empty());
        // Stats should be preserved after clear.
        assert_eq!(book.stats().snapshots_applied, 1);
    }

    #[test]
    fn test_orderbook_reset() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();
        book.reset();
        assert!(book.is_empty());
        // Stats should also be reset.
        assert_eq!(book.stats().snapshots_applied, 0);
    }

    #[test]
    fn test_orderbook_display() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();
        let display = format!("{}", book);
        assert!(display.contains("BTC/USDT"));
        assert!(display.contains("Mid"));
        assert!(display.contains("Spread"));
    }

    // ── Crossed book detection ─────────────────────────────────────────

    #[test]
    fn test_crossed_book_detection() {
        let mut book = OrderBook::new("TEST");
        // Manually set levels to create a crossed book (bid >= ask).
        book.set_level(Side::Buy, dec!(100.0), dec!(1.0)).unwrap();
        book.set_level(Side::Sell, dec!(99.0), dec!(1.0)).unwrap();
        assert!(book.is_crossed());
        assert_eq!(book.stats().crossed_book_events, 0); // Direct set_level doesn't track
    }

    #[test]
    fn test_crossed_book_via_snapshot() {
        let mut book = OrderBook::new("TEST");
        let snap = OrderBookSnapshot {
            symbol: "TEST".into(),
            bids: vec![PriceLevel::new(dec!(100.0), dec!(1.0))],
            asks: vec![PriceLevel::new(dec!(99.0), dec!(1.0))],
            timestamp: Utc::now(),
            sequence: 1,
        };
        book.apply_snapshot(snap).unwrap();
        assert!(book.is_crossed());
        assert_eq!(book.stats().crossed_book_events, 1);
    }

    // ── Partial fill ───────────────────────────────────────────────────

    #[test]
    fn test_resting_order_partial_fill() {
        let mut order = RestingOrder::new(Side::Buy, dec!(100.0), dec!(5.0), Utc::now());
        let filled = order.partial_fill(dec!(2.0));
        assert_eq!(filled, dec!(2.0));
        assert_eq!(order.remaining_quantity, dec!(3.0));
        assert!(!order.is_filled());
    }

    #[test]
    fn test_resting_order_partial_fill_clamps_to_remaining() {
        let mut order = RestingOrder::new(Side::Sell, dec!(50.0), dec!(3.0), Utc::now());
        let filled = order.partial_fill(dec!(10.0));
        assert_eq!(filled, dec!(3.0));
        assert_eq!(order.remaining_quantity, dec!(0.0));
        assert!(order.is_filled());
    }

    #[test]
    fn test_resting_order_partial_fill_zero() {
        let mut order = RestingOrder::new(Side::Buy, dec!(100.0), dec!(5.0), Utc::now());
        let filled = order.partial_fill(dec!(0.0));
        assert_eq!(filled, dec!(0.0));
        assert_eq!(order.remaining_quantity, dec!(5.0));
    }

    // ── Weighted mid price ─────────────────────────────────────────────

    #[test]
    fn test_weighted_mid_price_balanced() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();
        // Best bid: 67000 qty 1.5, best ask: 67001 qty 0.8
        // wmid = (67000 * 0.8 + 67001 * 1.5) / (1.5 + 0.8)
        //      = (53600 + 100501.5) / 2.3
        //      = 154101.5 / 2.3 ≈ 67000.6521739...
        let wmid = book.weighted_mid_price().unwrap();
        let mid = book.mid_price().unwrap();
        // Weighted mid should be closer to ask when bid qty > ask qty
        assert!(wmid > mid);
    }

    #[test]
    fn test_weighted_mid_price_empty() {
        let book = OrderBook::new("TEST");
        assert!(book.weighted_mid_price().is_none());
    }

    // ── Impact cost ────────────────────────────────────────────────────

    #[test]
    fn test_impact_cost_buy() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();
        let mid = book.mid_price().unwrap();
        let cost = book.impact_cost(Side::Buy, dec!(0.5)).unwrap();
        // Buying 0.5 at best ask 67001, mid is 67000.5
        // VWAP = 67001, impact = (67001 - 67000.5) / 67000.5 * 10000
        assert!(cost > Decimal::ZERO);
        let _ = mid; // suppress unused warning
    }

    #[test]
    fn test_impact_cost_sell() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();
        let cost = book.impact_cost(Side::Sell, dec!(0.5)).unwrap();
        // Selling 0.5 at best bid 67000, mid is 67000.5
        // VWAP = 67000, impact = (67000.5 - 67000) / 67000.5 * 10000
        assert!(cost > Decimal::ZERO);
    }

    #[test]
    fn test_impact_cost_empty_book() {
        let book = OrderBook::new("TEST");
        assert!(book.impact_cost(Side::Buy, dec!(1.0)).is_none());
    }

    // ── Delta batch ────────────────────────────────────────────────────

    #[test]
    fn test_delta_batch_application() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        let mut batch = OrderBookDeltaBatch::new("BTC/USDT", 2);
        batch.add(Side::Buy, dec!(67000.0), dec!(2.0)); // Update bid qty
        batch.add(Side::Sell, dec!(67001.0), dec!(1.5)); // Update ask qty
        batch.add(Side::Sell, dec!(67004.0), dec!(0.5)); // Add new ask level

        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());

        book.apply_delta_batch(batch).unwrap();

        assert_eq!(
            book.bids().get_level(dec!(67000.0)).unwrap().quantity,
            dec!(2.0)
        );
        assert_eq!(
            book.asks().get_level(dec!(67001.0)).unwrap().quantity,
            dec!(1.5)
        );
        assert_eq!(
            book.asks().get_level(dec!(67004.0)).unwrap().quantity,
            dec!(0.5)
        );
        assert_eq!(book.stats().deltas_applied, 3);
        assert_eq!(book.stats().last_sequence, 2);
    }

    #[test]
    fn test_delta_batch_stale_rejected() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        // Sequence 1 already consumed by snapshot, so batch seq=1 is stale.
        let batch = OrderBookDeltaBatch::new("BTC/USDT", 1);
        assert!(book.apply_delta_batch(batch).is_err());
        assert_eq!(book.stats().stale_updates_dropped, 1);
    }

    #[test]
    fn test_delta_batch_remove_level() {
        let mut book = OrderBook::new("BTC/USDT");
        book.apply_snapshot(sample_snapshot()).unwrap();

        let mut batch = OrderBookDeltaBatch::new("BTC/USDT", 2);
        // qty=0 removes the level
        batch.add(Side::Sell, dec!(67001.0), dec!(0.0));
        book.apply_delta_batch(batch).unwrap();

        assert!(book.asks().get_level(dec!(67001.0)).is_none());
        assert_eq!(book.best_ask_price(), Some(dec!(67002.0)));
    }
}
