//! # L2 Replay Engine
//!
//! Replays historical L2 (and optionally L3) order book data through the LOB
//! simulator. Ingests snapshots and delta streams, applying them to the order
//! book in chronological order while enforcing the TemporalFortress
//! zero-lookahead invariant.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      L2 Replay Engine                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
//! │  │ L2Event      │───▶│ Time-sorted  │───▶│ OrderBook       │   │
//! │  │ Stream       │    │ Event Queue  │    │ Application     │   │
//! │  └──────────────┘    └──────────────┘    └────────┬────────┘   │
//! │                                                    │            │
//! │  ┌─────────────────────────────────────────────────▼────────┐  │
//! │  │  TemporalFortress Guard                                   │  │
//! │  │  • Monotonic timestamp enforcement                        │  │
//! │  │  • No future data leakage                                 │  │
//! │  │  • Sequence number validation                             │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                              │                                   │
//! │                              ▼                                   │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │  Callbacks: on_snapshot, on_delta, on_trade, on_error     │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_lob::l2_replay::*;
//! use janus_lob::orderbook::*;
//! use chrono::Utc;
//!
//! // Build a replay from events
//! let config = L2ReplayConfig::default();
//! let mut replay = L2Replay::new("BTC/USDT", config);
//!
//! // Feed events (from file, database, or streaming source)
//! replay.push_event(L2Event::Snapshot { ... });
//! replay.push_event(L2Event::Delta { ... });
//!
//! // Step through events one at a time
//! while let Some(result) = replay.step() {
//!     match result {
//!         Ok(event) => println!("Applied: {}", event),
//!         Err(e) => eprintln!("Error: {}", e),
//!     }
//! }
//!
//! // Or replay all events at once
//! let stats = replay.replay_all()?;
//! println!("Replayed {} events in {:?}", stats.events_applied, stats.duration);
//! ```

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::fmt;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, trace, warn};

use crate::order_types::Side;
use crate::orderbook::{OrderBook, OrderBookDelta, OrderBookSnapshot, PriceLevel};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the L2 replay engine.
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum L2ReplayError {
    /// Event timestamp is in the future relative to the replay clock.
    #[error("Future event detected: event_time={event_time}, replay_clock={replay_clock}")]
    FutureEvent {
        event_time: DateTime<Utc>,
        replay_clock: DateTime<Utc>,
    },

    /// Event timestamp is out of order (non-monotonic).
    #[error("Out-of-order event: event_time={event_time}, last_time={last_time}")]
    OutOfOrder {
        event_time: DateTime<Utc>,
        last_time: DateTime<Utc>,
    },

    /// Sequence number gap detected.
    #[error("Sequence gap: expected={expected}, received={received}")]
    SequenceGap { expected: u64, received: u64 },

    /// Order book error during event application.
    #[error("OrderBook error: {0}")]
    BookError(String),

    /// No events in the replay buffer.
    #[error("Replay buffer is empty")]
    EmptyBuffer,

    /// Replay has already completed.
    #[error("Replay already completed")]
    AlreadyCompleted,

    /// Invalid event data.
    #[error("Invalid event: {0}")]
    InvalidEvent(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, L2ReplayError>;

// ---------------------------------------------------------------------------
// L2 Events
// ---------------------------------------------------------------------------

/// An L2 order book event for replay.
///
/// Events are time-stamped and sequenced. The replay engine processes them
/// in chronological order, applying each to the order book state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum L2Event {
    /// Full order book snapshot — replaces the entire book state.
    Snapshot {
        /// Trading symbol.
        symbol: String,
        /// Bid price levels.
        bids: Vec<L2Level>,
        /// Ask price levels.
        asks: Vec<L2Level>,
        /// Event timestamp.
        timestamp: DateTime<Utc>,
        /// Sequence number (monotonically increasing).
        sequence: u64,
    },

    /// Incremental delta update to a single price level.
    Delta {
        /// Trading symbol.
        symbol: String,
        /// Side of the book being updated.
        side: Side,
        /// Price of the level being updated.
        price: Decimal,
        /// New quantity at this price (0 = level removed).
        quantity: Decimal,
        /// Event timestamp.
        timestamp: DateTime<Utc>,
        /// Sequence number.
        sequence: u64,
    },

    /// Trade event — a trade occurred at this price and quantity.
    ///
    /// Trades are informational and don't directly modify the book,
    /// but they are used for:
    /// - Triggering stop orders
    /// - Updating last trade price
    /// - Volume tracking
    /// - Market impact calibration
    Trade {
        /// Trading symbol.
        symbol: String,
        /// Trade price.
        price: Decimal,
        /// Trade quantity.
        quantity: Decimal,
        /// Aggressor side (the side that initiated the trade).
        aggressor_side: Side,
        /// Event timestamp.
        timestamp: DateTime<Utc>,
        /// Sequence number.
        sequence: u64,
    },

    /// Book reset — clear the entire book (e.g., exchange maintenance).
    Reset {
        /// Trading symbol.
        symbol: String,
        /// Reason for the reset.
        reason: String,
        /// Event timestamp.
        timestamp: DateTime<Utc>,
        /// Sequence number.
        sequence: u64,
    },
}

impl L2Event {
    /// Get the timestamp of this event.
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            L2Event::Snapshot { timestamp, .. } => *timestamp,
            L2Event::Delta { timestamp, .. } => *timestamp,
            L2Event::Trade { timestamp, .. } => *timestamp,
            L2Event::Reset { timestamp, .. } => *timestamp,
        }
    }

    /// Get the sequence number of this event.
    pub fn sequence(&self) -> u64 {
        match self {
            L2Event::Snapshot { sequence, .. } => *sequence,
            L2Event::Delta { sequence, .. } => *sequence,
            L2Event::Trade { sequence, .. } => *sequence,
            L2Event::Reset { sequence, .. } => *sequence,
        }
    }

    /// Get the symbol of this event.
    pub fn symbol(&self) -> &str {
        match self {
            L2Event::Snapshot { symbol, .. } => symbol,
            L2Event::Delta { symbol, .. } => symbol,
            L2Event::Trade { symbol, .. } => symbol,
            L2Event::Reset { symbol, .. } => symbol,
        }
    }

    /// Get the event type as a string label.
    pub fn event_type(&self) -> &'static str {
        match self {
            L2Event::Snapshot { .. } => "Snapshot",
            L2Event::Delta { .. } => "Delta",
            L2Event::Trade { .. } => "Trade",
            L2Event::Reset { .. } => "Reset",
        }
    }

    /// Whether this event modifies the order book state.
    pub fn modifies_book(&self) -> bool {
        matches!(
            self,
            L2Event::Snapshot { .. } | L2Event::Delta { .. } | L2Event::Reset { .. }
        )
    }
}

impl fmt::Display for L2Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            L2Event::Snapshot {
                symbol,
                bids,
                asks,
                timestamp,
                sequence,
            } => {
                write!(
                    f,
                    "Snapshot({}, {} bids, {} asks, seq={}, t={})",
                    symbol,
                    bids.len(),
                    asks.len(),
                    sequence,
                    timestamp.format("%H:%M:%S%.3f"),
                )
            }
            L2Event::Delta {
                symbol,
                side,
                price,
                quantity,
                timestamp,
                sequence,
            } => {
                write!(
                    f,
                    "Delta({}, {} {} @ {}, seq={}, t={})",
                    symbol,
                    side,
                    quantity,
                    price,
                    sequence,
                    timestamp.format("%H:%M:%S%.3f"),
                )
            }
            L2Event::Trade {
                symbol,
                price,
                quantity,
                aggressor_side,
                timestamp,
                sequence,
            } => {
                write!(
                    f,
                    "Trade({}, {} {} @ {}, seq={}, t={})",
                    symbol,
                    aggressor_side,
                    quantity,
                    price,
                    sequence,
                    timestamp.format("%H:%M:%S%.3f"),
                )
            }
            L2Event::Reset {
                symbol,
                reason,
                timestamp,
                sequence,
            } => {
                write!(
                    f,
                    "Reset({}, reason='{}', seq={}, t={})",
                    symbol,
                    reason,
                    sequence,
                    timestamp.format("%H:%M:%S%.3f"),
                )
            }
        }
    }
}

/// A single price level in an L2 event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Level {
    /// Price.
    pub price: Decimal,
    /// Quantity at this price.
    pub quantity: Decimal,
    /// Number of orders at this level (if known).
    pub order_count: Option<u32>,
}

impl L2Level {
    /// Create a new L2 level.
    pub fn new(price: Decimal, quantity: Decimal) -> Self {
        Self {
            price,
            quantity,
            order_count: None,
        }
    }

    /// Create a new L2 level with order count.
    pub fn with_order_count(price: Decimal, quantity: Decimal, count: u32) -> Self {
        Self {
            price,
            quantity,
            order_count: Some(count),
        }
    }

    /// Convert to a `PriceLevel` for the order book.
    pub fn to_price_level(&self) -> PriceLevel {
        if let Some(count) = self.order_count {
            PriceLevel::with_order_count(self.price, self.quantity, count)
        } else {
            PriceLevel::new(self.price, self.quantity)
        }
    }
}

impl fmt::Display for L2Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.order_count {
            Some(count) => write!(f, "{} @ {} ({} orders)", self.quantity, self.price, count),
            None => write!(f, "{} @ {}", self.quantity, self.price),
        }
    }
}

// ---------------------------------------------------------------------------
// Priority queue wrapper for time-ordered event processing
// ---------------------------------------------------------------------------

/// Wrapper around `L2Event` that implements reverse-chronological ordering
/// for use in a `BinaryHeap` (min-heap by timestamp).
#[derive(Debug, Clone)]
struct TimedEvent {
    event: L2Event,
    insertion_order: u64,
}

impl PartialEq for TimedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.event.timestamp() == other.event.timestamp()
            && self.insertion_order == other.insertion_order
    }
}

impl Eq for TimedEvent {}

impl PartialOrd for TimedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimedEvent {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse order so BinaryHeap gives us earliest-first.
        other
            .event
            .timestamp()
            .cmp(&self.event.timestamp())
            .then_with(|| other.insertion_order.cmp(&self.insertion_order))
    }
}

// ---------------------------------------------------------------------------
// Replay Configuration
// ---------------------------------------------------------------------------

/// Configuration for the L2 replay engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2ReplayConfig {
    /// Whether to enforce strictly monotonic timestamps.
    /// If `true`, out-of-order events are rejected.
    /// If `false`, out-of-order events are reordered (using the priority queue).
    pub strict_ordering: bool,

    /// Whether to enforce monotonic sequence numbers.
    /// If `true`, sequence gaps cause an error.
    /// If `false`, sequence gaps are logged as warnings.
    pub strict_sequencing: bool,

    /// Whether to validate that event symbols match the replay symbol.
    pub validate_symbol: bool,

    /// Maximum number of events to buffer in the priority queue.
    /// Events beyond this limit are dropped.
    pub max_buffer_size: usize,

    /// Whether to skip events that would cause a crossed book.
    /// If `true`, deltas that cross the book are silently dropped.
    /// If `false`, they are applied (may indicate real data).
    pub skip_crossed_updates: bool,

    /// Whether to track and emit trade events.
    /// If `false`, trade events are silently consumed without callbacks.
    pub track_trades: bool,

    /// Maximum allowed timestamp gap between consecutive events.
    /// Events with a gap larger than this trigger a warning.
    pub max_gap: Option<Duration>,

    /// Whether to log every applied event (very verbose).
    pub trace_events: bool,

    /// Optional start time filter: skip events before this time.
    pub start_time: Option<DateTime<Utc>>,

    /// Optional end time filter: stop replay after this time.
    pub end_time: Option<DateTime<Utc>>,
}

impl Default for L2ReplayConfig {
    fn default() -> Self {
        Self {
            strict_ordering: true,
            strict_sequencing: false,
            validate_symbol: true,
            max_buffer_size: 1_000_000,
            skip_crossed_updates: false,
            track_trades: true,
            max_gap: None,
            trace_events: false,
            start_time: None,
            end_time: None,
        }
    }
}

impl L2ReplayConfig {
    /// Set strict ordering mode.
    pub fn with_strict_ordering(mut self, strict: bool) -> Self {
        self.strict_ordering = strict;
        self
    }

    /// Set strict sequencing mode.
    pub fn with_strict_sequencing(mut self, strict: bool) -> Self {
        self.strict_sequencing = strict;
        self
    }

    /// Set symbol validation.
    pub fn with_symbol_validation(mut self, validate: bool) -> Self {
        self.validate_symbol = validate;
        self
    }

    /// Set maximum buffer size.
    pub fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Set whether to skip crossed updates.
    pub fn with_skip_crossed(mut self, skip: bool) -> Self {
        self.skip_crossed_updates = skip;
        self
    }

    /// Set trade tracking.
    pub fn with_trade_tracking(mut self, track: bool) -> Self {
        self.track_trades = track;
        self
    }

    /// Set maximum allowed gap between events.
    pub fn with_max_gap(mut self, gap: Duration) -> Self {
        self.max_gap = Some(gap);
        self
    }

    /// Set event tracing.
    pub fn with_trace_events(mut self, trace: bool) -> Self {
        self.trace_events = trace;
        self
    }

    /// Set start time filter.
    pub fn with_start_time(mut self, start: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self
    }

    /// Set end time filter.
    pub fn with_end_time(mut self, end: DateTime<Utc>) -> Self {
        self.end_time = Some(end);
        self
    }
}

// ---------------------------------------------------------------------------
// Replay Statistics
// ---------------------------------------------------------------------------

/// Statistics collected during an L2 replay.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct L2ReplayStats {
    /// Total events pushed into the replay buffer.
    pub events_pushed: u64,

    /// Total events applied to the order book.
    pub events_applied: u64,

    /// Number of snapshot events applied.
    pub snapshots_applied: u64,

    /// Number of delta events applied.
    pub deltas_applied: u64,

    /// Number of trade events processed.
    pub trades_processed: u64,

    /// Number of reset events processed.
    pub resets_processed: u64,

    /// Number of events skipped (filtered, out-of-order, etc.).
    pub events_skipped: u64,

    /// Number of out-of-order events detected.
    pub out_of_order_events: u64,

    /// Number of sequence gaps detected.
    pub sequence_gaps: u64,

    /// Number of crossed book events detected after applying updates.
    pub crossed_book_events: u64,

    /// Number of events dropped due to buffer overflow.
    pub events_dropped: u64,

    /// Timestamp of the first event.
    pub first_event_time: Option<DateTime<Utc>>,

    /// Timestamp of the last event applied.
    pub last_event_time: Option<DateTime<Utc>>,

    /// Last sequence number seen.
    pub last_sequence: u64,

    /// Total volume from trade events.
    pub total_trade_volume: Decimal,

    /// Total notional from trade events.
    pub total_trade_notional: Decimal,

    /// Wall-clock duration of the replay.
    pub duration: Option<Duration>,
}

impl L2ReplayStats {
    /// Total book-modifying events applied (snapshots + deltas + resets).
    pub fn book_events(&self) -> u64 {
        self.snapshots_applied + self.deltas_applied + self.resets_processed
    }

    /// Event application rate (events per second of wall-clock time).
    pub fn events_per_second(&self) -> f64 {
        match self.duration {
            Some(d) if d.as_secs_f64() > 0.0 => self.events_applied as f64 / d.as_secs_f64(),
            _ => 0.0,
        }
    }

    /// Simulated time span covered by the replay.
    pub fn simulated_duration(&self) -> Option<chrono::Duration> {
        match (self.first_event_time, self.last_event_time) {
            (Some(first), Some(last)) => Some(last - first),
            _ => None,
        }
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for L2ReplayStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "L2Replay(applied={}, snapshots={}, deltas={}, trades={}, \
             skipped={}, gaps={}, crossed={}, {:.0} ev/s)",
            self.events_applied,
            self.snapshots_applied,
            self.deltas_applied,
            self.trades_processed,
            self.events_skipped,
            self.sequence_gaps,
            self.crossed_book_events,
            self.events_per_second(),
        )
    }
}

// ---------------------------------------------------------------------------
// Replay State
// ---------------------------------------------------------------------------

/// Current state of the replay engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayState {
    /// Engine is idle, waiting for events to be pushed.
    Idle,
    /// Engine is actively replaying events.
    Running,
    /// Replay has been paused.
    Paused,
    /// Replay has completed (all events processed).
    Completed,
    /// Replay encountered a fatal error and stopped.
    Error,
}

impl fmt::Display for ReplayState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Running => write!(f, "Running"),
            Self::Paused => write!(f, "Paused"),
            Self::Completed => write!(f, "Completed"),
            Self::Error => write!(f, "Error"),
        }
    }
}

// ---------------------------------------------------------------------------
// L2 Replay Engine
// ---------------------------------------------------------------------------

/// L2 order book replay engine.
///
/// Processes a stream of L2 events (snapshots, deltas, trades) and applies
/// them to an internal `OrderBook`, enforcing temporal ordering and the
/// zero-lookahead invariant required by TemporalFortress.
///
/// # Zero-Lookahead Invariant
///
/// The replay engine guarantees that at any point during replay, the order
/// book reflects only information that would have been available at the
/// current replay timestamp. No future data is leaked into the book state.
/// This is critical for backtesting integrity.
pub struct L2Replay {
    /// Trading symbol this replay is for.
    symbol: String,

    /// Configuration.
    config: L2ReplayConfig,

    /// The order book being replayed into.
    book: OrderBook,

    /// Priority queue of pending events (earliest first).
    event_queue: BinaryHeap<TimedEvent>,

    /// Counter for maintaining insertion order (tie-breaking).
    insertion_counter: u64,

    /// Current replay clock (timestamp of last applied event).
    replay_clock: Option<DateTime<Utc>>,

    /// Last sequence number applied.
    last_sequence: u64,

    /// Current state of the replay engine.
    state: ReplayState,

    /// Running statistics.
    stats: L2ReplayStats,

    /// Last trade price (for stop order triggering).
    last_trade_price: Option<Decimal>,

    /// Last trade side.
    last_trade_side: Option<Side>,
}

impl L2Replay {
    /// Create a new replay engine for the given symbol.
    pub fn new(symbol: impl Into<String>, config: L2ReplayConfig) -> Self {
        let sym = symbol.into();
        Self {
            book: OrderBook::new(&sym),
            symbol: sym,
            config,
            event_queue: BinaryHeap::new(),
            insertion_counter: 0,
            replay_clock: None,
            last_sequence: 0,
            state: ReplayState::Idle,
            stats: L2ReplayStats::default(),
            last_trade_price: None,
            last_trade_side: None,
        }
    }

    /// Create a new replay engine with an existing order book.
    pub fn with_book(book: OrderBook, config: L2ReplayConfig) -> Self {
        let symbol = book.symbol().to_string();
        Self {
            book,
            symbol,
            config,
            event_queue: BinaryHeap::new(),
            insertion_counter: 0,
            replay_clock: None,
            last_sequence: 0,
            state: ReplayState::Idle,
            stats: L2ReplayStats::default(),
            last_trade_price: None,
            last_trade_side: None,
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Get the trading symbol.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get a reference to the current order book state.
    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    /// Get a mutable reference to the order book.
    pub fn book_mut(&mut self) -> &mut OrderBook {
        &mut self.book
    }

    /// Get the current replay clock (timestamp of last applied event).
    pub fn replay_clock(&self) -> Option<DateTime<Utc>> {
        self.replay_clock
    }

    /// Get the current replay state.
    pub fn state(&self) -> ReplayState {
        self.state
    }

    /// Get the replay statistics.
    pub fn stats(&self) -> &L2ReplayStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &L2ReplayConfig {
        &self.config
    }

    /// Get the last trade price.
    pub fn last_trade_price(&self) -> Option<Decimal> {
        self.last_trade_price
    }

    /// Get the last trade side.
    pub fn last_trade_side(&self) -> Option<Side> {
        self.last_trade_side
    }

    /// Number of events in the pending queue.
    pub fn pending_events(&self) -> usize {
        self.event_queue.len()
    }

    /// Whether the replay is complete (no more events to process).
    pub fn is_complete(&self) -> bool {
        self.state == ReplayState::Completed || self.event_queue.is_empty()
    }

    // ── Event Ingestion ────────────────────────────────────────────────

    /// Push a single event into the replay buffer.
    ///
    /// Events are internally sorted by timestamp. Returns an error if the
    /// buffer is full or the event fails validation.
    pub fn push_event(&mut self, event: L2Event) -> Result<()> {
        // Check buffer capacity.
        if self.event_queue.len() >= self.config.max_buffer_size {
            self.stats.events_dropped += 1;
            return Err(L2ReplayError::InvalidEvent(format!(
                "Buffer full ({} events), dropping event seq={}",
                self.config.max_buffer_size,
                event.sequence(),
            )));
        }

        // Validate symbol if configured.
        if self.config.validate_symbol && event.symbol() != self.symbol {
            return Err(L2ReplayError::InvalidEvent(format!(
                "Symbol mismatch: expected '{}', got '{}'",
                self.symbol,
                event.symbol(),
            )));
        }

        // Apply time filter.
        if let Some(start) = self.config.start_time
            && event.timestamp() < start
        {
            self.stats.events_skipped += 1;
            return Ok(());
        }
        if let Some(end) = self.config.end_time
            && event.timestamp() > end
        {
            self.stats.events_skipped += 1;
            return Ok(());
        }

        // Track first event time.
        if self.stats.first_event_time.is_none() {
            self.stats.first_event_time = Some(event.timestamp());
        }

        self.stats.events_pushed += 1;

        let timed = TimedEvent {
            event,
            insertion_order: self.insertion_counter,
        };
        self.insertion_counter += 1;

        self.event_queue.push(timed);

        if self.state == ReplayState::Idle {
            self.state = ReplayState::Running;
        }

        Ok(())
    }

    /// Push multiple events into the replay buffer.
    pub fn push_events(&mut self, events: Vec<L2Event>) -> Result<usize> {
        let mut pushed = 0;
        for event in events {
            match self.push_event(event) {
                Ok(()) => pushed += 1,
                Err(L2ReplayError::InvalidEvent(_)) => {
                    // Continue with remaining events.
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(pushed)
    }

    // ── Replay Execution ───────────────────────────────────────────────

    /// Process the next event in chronological order.
    ///
    /// Returns the event that was applied, or `None` if the queue is empty.
    /// This is the primary step function for incremental replay.
    pub fn step(&mut self) -> Option<Result<L2Event>> {
        if self.state == ReplayState::Completed || self.state == ReplayState::Paused {
            return None;
        }

        let timed = self.event_queue.pop()?;
        let event = timed.event;

        Some(self.apply_event(event))
    }

    /// Advance the replay to the given timestamp.
    ///
    /// Processes all events up to and including `until`. Returns the number
    /// of events applied. This is useful for clock-driven replay where you
    /// advance time in fixed increments.
    pub fn advance_to(&mut self, until: DateTime<Utc>) -> Result<u64> {
        let mut applied = 0u64;

        while let Some(timed) = self.event_queue.peek() {
            let next_time = timed.event.timestamp();

            if next_time > until {
                break;
            }

            if self.state == ReplayState::Paused || self.state == ReplayState::Completed {
                break;
            }

            let timed = self.event_queue.pop().unwrap();
            match self.apply_event(timed.event) {
                Ok(_) => applied += 1,
                Err(e) => {
                    // Log and continue for non-fatal errors.
                    warn!(error = %e, "Error during advance_to, skipping event");
                    self.stats.events_skipped += 1;
                }
            }
        }

        Ok(applied)
    }

    /// Replay all remaining events.
    ///
    /// Returns the final replay statistics. This processes every event in
    /// the queue sequentially.
    pub fn replay_all(&mut self) -> Result<L2ReplayStats> {
        if self.state == ReplayState::Completed {
            return Err(L2ReplayError::AlreadyCompleted);
        }

        let start = Instant::now();
        self.state = ReplayState::Running;

        while let Some(timed) = self.event_queue.pop() {
            match self.apply_event(timed.event) {
                Ok(_) => {}
                Err(e) => {
                    warn!(error = %e, "Error during replay_all, skipping event");
                    self.stats.events_skipped += 1;
                }
            }
        }

        self.stats.duration = Some(start.elapsed());
        self.state = ReplayState::Completed;

        info!(
            symbol = %self.symbol,
            events = self.stats.events_applied,
            duration_ms = self.stats.duration.unwrap_or_default().as_millis(),
            events_per_sec = self.stats.events_per_second(),
            "L2 replay completed"
        );

        Ok(self.stats.clone())
    }

    /// Pause the replay. Call `resume()` to continue.
    pub fn pause(&mut self) {
        if self.state == ReplayState::Running {
            self.state = ReplayState::Paused;
            debug!(symbol = %self.symbol, "L2 replay paused");
        }
    }

    /// Resume a paused replay.
    pub fn resume(&mut self) {
        if self.state == ReplayState::Paused {
            self.state = ReplayState::Running;
            debug!(symbol = %self.symbol, "L2 replay resumed");
        }
    }

    /// Reset the replay engine, clearing all state and the book.
    pub fn reset(&mut self) {
        self.book.reset();
        self.event_queue.clear();
        self.insertion_counter = 0;
        self.replay_clock = None;
        self.last_sequence = 0;
        self.state = ReplayState::Idle;
        self.stats.reset();
        self.last_trade_price = None;
        self.last_trade_side = None;
        debug!(symbol = %self.symbol, "L2 replay reset");
    }

    // ── Internal Event Application ─────────────────────────────────────

    /// Apply a single event to the order book.
    ///
    /// This is the core function that enforces temporal ordering and
    /// applies the event to the book state.
    fn apply_event(&mut self, event: L2Event) -> Result<L2Event> {
        let event_time = event.timestamp();
        let event_seq = event.sequence();

        // ── TemporalFortress: enforce monotonic time ───────────────────
        if let Some(clock) = self.replay_clock
            && event_time < clock
            && self.config.strict_ordering
        {
            self.stats.out_of_order_events += 1;
            return Err(L2ReplayError::OutOfOrder {
                event_time,
                last_time: clock,
            });
        }

        // ── Sequence validation ────────────────────────────────────────
        if event_seq > 0 && self.last_sequence > 0 && event_seq != self.last_sequence + 1 {
            self.stats.sequence_gaps += 1;
            if self.config.strict_sequencing {
                return Err(L2ReplayError::SequenceGap {
                    expected: self.last_sequence + 1,
                    received: event_seq,
                });
            } else {
                trace!(
                    expected = self.last_sequence + 1,
                    received = event_seq,
                    "Sequence gap detected (non-strict mode)"
                );
            }
        }

        // ── Gap detection ──────────────────────────────────────────────
        if let (Some(max_gap), Some(clock)) = (self.config.max_gap, self.replay_clock) {
            let gap = event_time - clock;
            if let Ok(std_gap) = gap.to_std()
                && std_gap > max_gap
            {
                warn!(
                    symbol = %self.symbol,
                    gap_secs = std_gap.as_secs_f64(),
                    max_gap_secs = max_gap.as_secs_f64(),
                    "Large time gap between events"
                );
            }
        }

        // ── Apply the event ────────────────────────────────────────────
        match &event {
            L2Event::Snapshot {
                bids,
                asks,
                timestamp,
                sequence,
                ..
            } => {
                let snapshot = OrderBookSnapshot {
                    symbol: self.symbol.clone(),
                    bids: bids.iter().map(|l| l.to_price_level()).collect(),
                    asks: asks.iter().map(|l| l.to_price_level()).collect(),
                    timestamp: *timestamp,
                    sequence: *sequence,
                };

                // Reset sequence tracking on snapshot (snapshots reset state).
                self.book.clear();
                self.book
                    .apply_snapshot(snapshot)
                    .map_err(|e| L2ReplayError::BookError(e.to_string()))?;

                self.stats.snapshots_applied += 1;

                if self.config.trace_events {
                    trace!(
                        symbol = %self.symbol,
                        bids = bids.len(),
                        asks = asks.len(),
                        seq = sequence,
                        "Applied snapshot"
                    );
                }
            }

            L2Event::Delta {
                side,
                price,
                quantity,
                timestamp,
                sequence,
                ..
            } => {
                // Optionally skip deltas that would cross the book.
                if self.config.skip_crossed_updates {
                    let would_cross = match side {
                        Side::Buy => self
                            .book
                            .best_ask_price()
                            .is_some_and(|ask| *price >= ask && !quantity.is_zero()),
                        Side::Sell => self
                            .book
                            .best_bid_price()
                            .is_some_and(|bid| *price <= bid && !quantity.is_zero()),
                    };

                    if would_cross {
                        self.stats.crossed_book_events += 1;
                        self.stats.events_skipped += 1;
                        trace!(
                            symbol = %self.symbol,
                            side = %side,
                            price = %price,
                            qty = %quantity,
                            "Skipping crossed delta"
                        );
                        // Still update clock and sequence.
                        self.replay_clock = Some(event_time);
                        if event_seq > 0 {
                            self.last_sequence = event_seq;
                            self.stats.last_sequence = event_seq;
                        }
                        self.stats.events_applied += 1;
                        self.stats.last_event_time = Some(event_time);
                        return Ok(event);
                    }
                }

                let delta = OrderBookDelta {
                    symbol: self.symbol.clone(),
                    side: *side,
                    price: *price,
                    quantity: *quantity,
                    timestamp: *timestamp,
                    sequence: *sequence,
                };

                self.book
                    .apply_delta(delta)
                    .map_err(|e| L2ReplayError::BookError(e.to_string()))?;

                // Check for crossed book after delta.
                if self.book.is_crossed() {
                    self.stats.crossed_book_events += 1;
                }

                self.stats.deltas_applied += 1;

                if self.config.trace_events {
                    trace!(
                        symbol = %self.symbol,
                        side = %side,
                        price = %price,
                        qty = %quantity,
                        seq = sequence,
                        "Applied delta"
                    );
                }
            }

            L2Event::Trade {
                price,
                quantity,
                aggressor_side,
                ..
            } => {
                self.last_trade_price = Some(*price);
                self.last_trade_side = Some(*aggressor_side);

                if self.config.track_trades {
                    self.stats.total_trade_volume += quantity;
                    self.stats.total_trade_notional += price * quantity;
                    self.stats.trades_processed += 1;
                }

                if self.config.trace_events {
                    trace!(
                        symbol = %self.symbol,
                        price = %price,
                        qty = %quantity,
                        side = %aggressor_side,
                        "Processed trade"
                    );
                }
            }

            L2Event::Reset { reason, .. } => {
                self.book.reset();
                self.stats.resets_processed += 1;

                debug!(
                    symbol = %self.symbol,
                    reason = %reason,
                    "Book reset during replay"
                );
            }
        }

        // ── Update replay clock and sequence ───────────────────────────
        self.replay_clock = Some(event_time);
        if event_seq > 0 {
            self.last_sequence = event_seq;
            self.stats.last_sequence = event_seq;
        }
        self.stats.events_applied += 1;
        self.stats.last_event_time = Some(event_time);

        Ok(event)
    }
}

impl fmt::Display for L2Replay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "L2Replay({}, state={}, pending={}, applied={}, book_levels={})",
            self.symbol,
            self.state,
            self.event_queue.len(),
            self.stats.events_applied,
            self.book.total_levels(),
        )
    }
}

impl fmt::Debug for L2Replay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("L2Replay")
            .field("symbol", &self.symbol)
            .field("state", &self.state)
            .field("pending_events", &self.event_queue.len())
            .field("replay_clock", &self.replay_clock)
            .field("last_sequence", &self.last_sequence)
            .field("stats", &self.stats)
            .finish()
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    // ── Helpers ────────────────────────────────────────────────────────

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + secs, 0).unwrap()
    }

    fn make_snapshot(seq: u64, secs: i64) -> L2Event {
        L2Event::Snapshot {
            symbol: "BTC/USDT".into(),
            bids: vec![
                L2Level::new(dec!(67000), dec!(1.5)),
                L2Level::new(dec!(66999), dec!(3.2)),
            ],
            asks: vec![
                L2Level::new(dec!(67001), dec!(0.8)),
                L2Level::new(dec!(67002), dec!(2.1)),
            ],
            timestamp: ts(secs),
            sequence: seq,
        }
    }

    fn make_delta(seq: u64, secs: i64, side: Side, price: Decimal, qty: Decimal) -> L2Event {
        L2Event::Delta {
            symbol: "BTC/USDT".into(),
            side,
            price,
            quantity: qty,
            timestamp: ts(secs),
            sequence: seq,
        }
    }

    fn make_trade(seq: u64, secs: i64, price: Decimal, qty: Decimal, side: Side) -> L2Event {
        L2Event::Trade {
            symbol: "BTC/USDT".into(),
            price,
            quantity: qty,
            aggressor_side: side,
            timestamp: ts(secs),
            sequence: seq,
        }
    }

    fn make_reset(seq: u64, secs: i64) -> L2Event {
        L2Event::Reset {
            symbol: "BTC/USDT".into(),
            reason: "test reset".into(),
            timestamp: ts(secs),
            sequence: seq,
        }
    }

    // ── L2Event ────────────────────────────────────────────────────────

    #[test]
    fn test_event_timestamp() {
        let event = make_snapshot(1, 0);
        assert_eq!(event.timestamp(), ts(0));
    }

    #[test]
    fn test_event_sequence() {
        let event = make_snapshot(42, 0);
        assert_eq!(event.sequence(), 42);
    }

    #[test]
    fn test_event_symbol() {
        let event = make_snapshot(1, 0);
        assert_eq!(event.symbol(), "BTC/USDT");
    }

    #[test]
    fn test_event_type_label() {
        assert_eq!(make_snapshot(1, 0).event_type(), "Snapshot");
        assert_eq!(
            make_delta(1, 0, Side::Buy, dec!(100), dec!(1)).event_type(),
            "Delta"
        );
        assert_eq!(
            make_trade(1, 0, dec!(100), dec!(1), Side::Buy).event_type(),
            "Trade"
        );
        assert_eq!(make_reset(1, 0).event_type(), "Reset");
    }

    #[test]
    fn test_event_modifies_book() {
        assert!(make_snapshot(1, 0).modifies_book());
        assert!(make_delta(1, 0, Side::Buy, dec!(100), dec!(1)).modifies_book());
        assert!(!make_trade(1, 0, dec!(100), dec!(1), Side::Buy).modifies_book());
        assert!(make_reset(1, 0).modifies_book());
    }

    #[test]
    fn test_event_display_snapshot() {
        let s = format!("{}", make_snapshot(1, 0));
        assert!(s.contains("Snapshot"));
        assert!(s.contains("BTC/USDT"));
        assert!(s.contains("2 bids"));
        assert!(s.contains("2 asks"));
    }

    #[test]
    fn test_event_display_delta() {
        let event = make_delta(5, 10, Side::Sell, dec!(67003), dec!(1.5));
        let s = format!("{}", event);
        assert!(s.contains("Delta"));
        assert!(s.contains("SELL"));
        assert!(s.contains("67003"));
    }

    #[test]
    fn test_event_display_trade() {
        let event = make_trade(5, 10, dec!(67001), dec!(0.3), Side::Buy);
        let s = format!("{}", event);
        assert!(s.contains("Trade"));
        assert!(s.contains("BUY"));
        assert!(s.contains("67001"));
    }

    #[test]
    fn test_event_display_reset() {
        let s = format!("{}", make_reset(1, 0));
        assert!(s.contains("Reset"));
        assert!(s.contains("test reset"));
    }

    // ── L2Level ────────────────────────────────────────────────────────

    #[test]
    fn test_l2_level_new() {
        let level = L2Level::new(dec!(67000), dec!(1.5));
        assert_eq!(level.price, dec!(67000));
        assert_eq!(level.quantity, dec!(1.5));
        assert!(level.order_count.is_none());
    }

    #[test]
    fn test_l2_level_with_order_count() {
        let level = L2Level::with_order_count(dec!(67000), dec!(1.5), 5);
        assert_eq!(level.order_count, Some(5));
    }

    #[test]
    fn test_l2_level_to_price_level() {
        let level = L2Level::new(dec!(67000), dec!(1.5));
        let pl = level.to_price_level();
        assert_eq!(pl.price, dec!(67000));
        assert_eq!(pl.quantity, dec!(1.5));
    }

    #[test]
    fn test_l2_level_to_price_level_with_count() {
        let level = L2Level::with_order_count(dec!(67000), dec!(1.5), 10);
        let pl = level.to_price_level();
        assert_eq!(pl.order_count, 10);
    }

    #[test]
    fn test_l2_level_display() {
        let level = L2Level::new(dec!(67000), dec!(1.5));
        let s = format!("{}", level);
        assert!(s.contains("67000"));
        assert!(s.contains("1.5"));
    }

    #[test]
    fn test_l2_level_display_with_count() {
        let level = L2Level::with_order_count(dec!(67000), dec!(1.5), 3);
        let s = format!("{}", level);
        assert!(s.contains("3 orders"));
    }

    // ── L2ReplayConfig ─────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = L2ReplayConfig::default();
        assert!(config.strict_ordering);
        assert!(!config.strict_sequencing);
        assert!(config.validate_symbol);
        assert!(config.track_trades);
        assert!(!config.trace_events);
        assert!(config.start_time.is_none());
        assert!(config.end_time.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = L2ReplayConfig::default()
            .with_strict_ordering(false)
            .with_strict_sequencing(true)
            .with_symbol_validation(false)
            .with_max_buffer_size(100)
            .with_skip_crossed(true)
            .with_trade_tracking(false)
            .with_max_gap(Duration::from_secs(60))
            .with_trace_events(true)
            .with_start_time(ts(0))
            .with_end_time(ts(100));

        assert!(!config.strict_ordering);
        assert!(config.strict_sequencing);
        assert!(!config.validate_symbol);
        assert_eq!(config.max_buffer_size, 100);
        assert!(config.skip_crossed_updates);
        assert!(!config.track_trades);
        assert_eq!(config.max_gap, Some(Duration::from_secs(60)));
        assert!(config.trace_events);
        assert_eq!(config.start_time, Some(ts(0)));
        assert_eq!(config.end_time, Some(ts(100)));
    }

    // ── L2ReplayStats ──────────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let stats = L2ReplayStats::default();
        assert_eq!(stats.events_applied, 0);
        assert_eq!(stats.book_events(), 0);
        assert_eq!(stats.events_per_second(), 0.0);
    }

    #[test]
    fn test_stats_book_events() {
        let stats = L2ReplayStats {
            snapshots_applied: 2,
            deltas_applied: 10,
            resets_processed: 1,
            ..Default::default()
        };
        assert_eq!(stats.book_events(), 13);
    }

    #[test]
    fn test_stats_events_per_second() {
        let stats = L2ReplayStats {
            events_applied: 1000,
            duration: Some(Duration::from_secs(2)),
            ..Default::default()
        };
        assert!((stats.events_per_second() - 500.0).abs() < 1.0);
    }

    #[test]
    fn test_stats_simulated_duration() {
        let stats = L2ReplayStats {
            first_event_time: Some(ts(0)),
            last_event_time: Some(ts(100)),
            ..Default::default()
        };
        let dur = stats.simulated_duration().unwrap();
        assert_eq!(dur.num_seconds(), 100);
    }

    #[test]
    fn test_stats_simulated_duration_none() {
        let stats = L2ReplayStats::default();
        assert!(stats.simulated_duration().is_none());
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = L2ReplayStats {
            events_applied: 100,
            snapshots_applied: 5,
            deltas_applied: 90,
            ..Default::default()
        };
        stats.reset();
        assert_eq!(stats.events_applied, 0);
    }

    #[test]
    fn test_stats_display() {
        let stats = L2ReplayStats {
            events_applied: 100,
            snapshots_applied: 2,
            deltas_applied: 90,
            trades_processed: 8,
            ..Default::default()
        };
        let s = format!("{}", stats);
        assert!(s.contains("applied=100"));
        assert!(s.contains("snapshots=2"));
    }

    // ── ReplayState ────────────────────────────────────────────────────

    #[test]
    fn test_replay_state_display() {
        assert_eq!(format!("{}", ReplayState::Idle), "Idle");
        assert_eq!(format!("{}", ReplayState::Running), "Running");
        assert_eq!(format!("{}", ReplayState::Paused), "Paused");
        assert_eq!(format!("{}", ReplayState::Completed), "Completed");
        assert_eq!(format!("{}", ReplayState::Error), "Error");
    }

    // ── L2Replay Basic ─────────────────────────────────────────────────

    #[test]
    fn test_replay_new() {
        let replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        assert_eq!(replay.symbol(), "BTC/USDT");
        assert_eq!(replay.state(), ReplayState::Idle);
        assert_eq!(replay.pending_events(), 0);
        assert!(replay.is_complete());
        assert!(replay.replay_clock().is_none());
        assert!(replay.last_trade_price().is_none());
    }

    #[test]
    fn test_replay_with_book() {
        let book = OrderBook::new("ETH/USDT");
        let replay = L2Replay::with_book(book, L2ReplayConfig::default());
        assert_eq!(replay.symbol(), "ETH/USDT");
    }

    #[test]
    fn test_replay_push_event() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();
        assert_eq!(replay.pending_events(), 1);
        assert_eq!(replay.stats().events_pushed, 1);
        assert_eq!(replay.state(), ReplayState::Running);
    }

    #[test]
    fn test_replay_push_wrong_symbol() {
        let mut replay = L2Replay::new("ETH/USDT", L2ReplayConfig::default());
        let result = replay.push_event(make_snapshot(1, 0)); // BTC/USDT
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_push_wrong_symbol_no_validation() {
        let config = L2ReplayConfig::default().with_symbol_validation(false);
        let mut replay = L2Replay::new("ETH/USDT", config);
        let result = replay.push_event(make_snapshot(1, 0)); // BTC/USDT
        assert!(result.is_ok());
    }

    #[test]
    fn test_replay_push_buffer_overflow() {
        let config = L2ReplayConfig::default()
            .with_max_buffer_size(2)
            .with_symbol_validation(false);
        let mut replay = L2Replay::new("BTC/USDT", config);

        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay.push_event(make_snapshot(2, 1)).unwrap();
        let result = replay.push_event(make_snapshot(3, 2));
        assert!(result.is_err());
        assert_eq!(replay.stats().events_dropped, 1);
    }

    #[test]
    fn test_replay_push_events_batch() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        let events = vec![
            make_snapshot(1, 0),
            make_delta(2, 1, Side::Buy, dec!(67001), dec!(2.0)),
            make_trade(3, 2, dec!(67001), dec!(0.5), Side::Buy),
        ];
        let pushed = replay.push_events(events).unwrap();
        assert_eq!(pushed, 3);
        assert_eq!(replay.pending_events(), 3);
    }

    // ── L2Replay Step ──────────────────────────────────────────────────

    #[test]
    fn test_replay_step_snapshot() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();

        let result = replay.step().unwrap();
        assert!(result.is_ok());
        let event = result.unwrap();
        assert_eq!(event.event_type(), "Snapshot");

        // Book should now have levels.
        assert_eq!(replay.book().bids().len(), 2);
        assert_eq!(replay.book().asks().len(), 2);
        assert_eq!(replay.book().best_bid_price(), Some(dec!(67000)));
        assert_eq!(replay.book().best_ask_price(), Some(dec!(67001)));
    }

    #[test]
    fn test_replay_step_delta() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_delta(2, 1, Side::Buy, dec!(67000), dec!(2.5)))
            .unwrap();

        replay.step().unwrap().unwrap(); // Snapshot
        replay.step().unwrap().unwrap(); // Delta

        // Bid at 67000 should be updated.
        let bid = replay.book().bids().get_level(dec!(67000)).unwrap();
        assert_eq!(bid.quantity, dec!(2.5));
    }

    #[test]
    fn test_replay_step_trade() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_trade(2, 1, dec!(67001), dec!(0.3), Side::Buy))
            .unwrap();

        replay.step().unwrap().unwrap(); // Snapshot
        replay.step().unwrap().unwrap(); // Trade

        assert_eq!(replay.last_trade_price(), Some(dec!(67001)));
        assert_eq!(replay.last_trade_side(), Some(Side::Buy));
        assert_eq!(replay.stats().trades_processed, 1);
    }

    #[test]
    fn test_replay_step_reset() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay.push_event(make_reset(2, 1)).unwrap();

        replay.step().unwrap().unwrap(); // Snapshot
        assert!(!replay.book().is_empty());

        replay.step().unwrap().unwrap(); // Reset
        assert!(replay.book().is_empty());
        assert_eq!(replay.stats().resets_processed, 1);
    }

    #[test]
    fn test_replay_step_empty() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        assert!(replay.step().is_none());
    }

    #[test]
    fn test_replay_step_completed() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay.replay_all().unwrap();
        assert!(replay.step().is_none());
    }

    // ── L2Replay Temporal Ordering ─────────────────────────────────────

    #[test]
    fn test_replay_chronological_order() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());

        // Push events out of order — they should be replayed in order.
        replay
            .push_event(make_delta(3, 3, Side::Buy, dec!(67000), dec!(3.0)))
            .unwrap();
        replay.push_event(make_snapshot(1, 1)).unwrap();
        replay
            .push_event(make_delta(2, 2, Side::Buy, dec!(67000), dec!(2.0)))
            .unwrap();

        // Step through — should get snapshot first, then deltas in order.
        let e1 = replay.step().unwrap().unwrap();
        assert_eq!(e1.sequence(), 1);

        let e2 = replay.step().unwrap().unwrap();
        assert_eq!(e2.sequence(), 2);

        let e3 = replay.step().unwrap().unwrap();
        assert_eq!(e3.sequence(), 3);
    }

    #[test]
    fn test_replay_strict_ordering_rejects_out_of_order() {
        let config = L2ReplayConfig::default().with_strict_ordering(true);
        let mut replay = L2Replay::new("BTC/USDT", config);

        // Manually apply events to set clock, then try to go backward.
        replay.push_event(make_snapshot(1, 10)).unwrap();
        replay.step().unwrap().unwrap();

        // Now push an event with an earlier timestamp.
        replay
            .push_event(make_delta(2, 5, Side::Buy, dec!(67000), dec!(1.0)))
            .unwrap();

        let result = replay.step().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_non_strict_allows_out_of_order() {
        let config = L2ReplayConfig::default().with_strict_ordering(false);
        let mut replay = L2Replay::new("BTC/USDT", config);

        replay.push_event(make_snapshot(1, 10)).unwrap();
        replay.step().unwrap().unwrap();

        replay
            .push_event(make_delta(2, 5, Side::Buy, dec!(67000), dec!(1.0)))
            .unwrap();

        let result = replay.step().unwrap();
        assert!(result.is_ok());
    }

    // ── L2Replay Sequence Validation ───────────────────────────────────

    #[test]
    fn test_replay_strict_sequencing_rejects_gap() {
        let config = L2ReplayConfig::default().with_strict_sequencing(true);
        let mut replay = L2Replay::new("BTC/USDT", config);

        replay.push_event(make_snapshot(1, 0)).unwrap();
        // Sequence gap: 1 → 5 (expected 2).
        replay
            .push_event(make_delta(5, 1, Side::Buy, dec!(67000), dec!(1.0)))
            .unwrap();

        replay.step().unwrap().unwrap(); // seq=1
        let result = replay.step().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_replay_non_strict_sequencing_warns_on_gap() {
        let config = L2ReplayConfig::default().with_strict_sequencing(false);
        let mut replay = L2Replay::new("BTC/USDT", config);

        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_delta(5, 1, Side::Buy, dec!(67000), dec!(1.0)))
            .unwrap();

        replay.step().unwrap().unwrap(); // seq=1
        let result = replay.step().unwrap();
        assert!(result.is_ok()); // Non-strict: proceed with warning.
        assert_eq!(replay.stats().sequence_gaps, 1);
    }

    // ── L2Replay replay_all ────────────────────────────────────────────

    #[test]
    fn test_replay_all() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());

        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_delta(2, 1, Side::Buy, dec!(67000), dec!(2.0)))
            .unwrap();
        replay
            .push_event(make_delta(3, 2, Side::Sell, dec!(67001), dec!(1.5)))
            .unwrap();
        replay
            .push_event(make_trade(4, 3, dec!(67001), dec!(0.5), Side::Buy))
            .unwrap();

        let stats = replay.replay_all().unwrap();

        assert_eq!(stats.events_applied, 4);
        assert_eq!(stats.snapshots_applied, 1);
        assert_eq!(stats.deltas_applied, 2);
        assert_eq!(stats.trades_processed, 1);
        assert!(stats.duration.is_some());
        assert_eq!(replay.state(), ReplayState::Completed);
    }

    #[test]
    fn test_replay_all_already_completed() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay.replay_all().unwrap();

        let result = replay.replay_all();
        assert!(result.is_err());
    }

    // ── L2Replay advance_to ────────────────────────────────────────────

    #[test]
    fn test_replay_advance_to() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());

        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_delta(2, 5, Side::Buy, dec!(67000), dec!(2.0)))
            .unwrap();
        replay
            .push_event(make_delta(3, 10, Side::Sell, dec!(67001), dec!(1.5)))
            .unwrap();
        replay
            .push_event(make_delta(4, 15, Side::Buy, dec!(66998), dec!(0.5)))
            .unwrap();

        // Advance to t=7 — should process snapshot (t=0) and first delta (t=5).
        let applied = replay.advance_to(ts(7)).unwrap();
        assert_eq!(applied, 2);
        assert_eq!(replay.pending_events(), 2);

        // Advance to t=12 — should process second delta (t=10).
        let applied = replay.advance_to(ts(12)).unwrap();
        assert_eq!(applied, 1);
        assert_eq!(replay.pending_events(), 1);
    }

    // ── L2Replay Pause/Resume ──────────────────────────────────────────

    #[test]
    fn test_replay_pause_resume() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_delta(2, 1, Side::Buy, dec!(67000), dec!(2.0)))
            .unwrap();

        replay.step().unwrap().unwrap();
        replay.pause();
        assert_eq!(replay.state(), ReplayState::Paused);

        // Step should return None while paused.
        assert!(replay.step().is_none());

        replay.resume();
        assert_eq!(replay.state(), ReplayState::Running);

        // Should be able to step again.
        let result = replay.step().unwrap();
        assert!(result.is_ok());
    }

    // ── L2Replay Reset ─────────────────────────────────────────────────

    #[test]
    fn test_replay_reset() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay.step().unwrap().unwrap();

        assert!(!replay.book().is_empty());
        assert!(replay.replay_clock().is_some());

        replay.reset();

        assert!(replay.book().is_empty());
        assert!(replay.replay_clock().is_none());
        assert_eq!(replay.state(), ReplayState::Idle);
        assert_eq!(replay.stats().events_applied, 0);
        assert_eq!(replay.pending_events(), 0);
    }

    // ── L2Replay Time Filtering ────────────────────────────────────────

    #[test]
    fn test_replay_start_time_filter() {
        let config = L2ReplayConfig::default().with_start_time(ts(5));
        let mut replay = L2Replay::new("BTC/USDT", config);

        // Event at t=0 should be filtered out.
        replay.push_event(make_snapshot(1, 0)).unwrap();
        assert_eq!(replay.pending_events(), 0);
        assert_eq!(replay.stats().events_skipped, 1);

        // Event at t=10 should pass.
        replay.push_event(make_snapshot(2, 10)).unwrap();
        assert_eq!(replay.pending_events(), 1);
    }

    #[test]
    fn test_replay_end_time_filter() {
        let config = L2ReplayConfig::default().with_end_time(ts(5));
        let mut replay = L2Replay::new("BTC/USDT", config);

        // Event at t=0 should pass.
        replay.push_event(make_snapshot(1, 0)).unwrap();
        assert_eq!(replay.pending_events(), 1);

        // Event at t=10 should be filtered out.
        replay
            .push_event(make_delta(2, 10, Side::Buy, dec!(67000), dec!(1.0)))
            .unwrap();
        assert_eq!(replay.pending_events(), 1);
        assert_eq!(replay.stats().events_skipped, 1);
    }

    // ── L2Replay Skip Crossed ──────────────────────────────────────────

    #[test]
    fn test_replay_skip_crossed_delta() {
        let config = L2ReplayConfig::default().with_skip_crossed(true);
        let mut replay = L2Replay::new("BTC/USDT", config);

        replay.push_event(make_snapshot(1, 0)).unwrap();
        // This delta would cross the book: bid at 67002 >= best ask 67001.
        replay
            .push_event(make_delta(2, 1, Side::Buy, dec!(67002), dec!(1.0)))
            .unwrap();

        replay.step().unwrap().unwrap(); // Snapshot
        replay.step().unwrap().unwrap(); // Delta (skipped internally)

        assert_eq!(replay.stats().crossed_book_events, 1);
        // The book should NOT have a bid at 67002 because it was skipped.
        assert!(replay.book().bids().get_level(dec!(67002)).is_none());
    }

    // ── L2Replay Trade Volume Tracking ─────────────────────────────────

    #[test]
    fn test_replay_trade_volume_tracking() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());

        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_trade(2, 1, dec!(67001), dec!(0.5), Side::Buy))
            .unwrap();
        replay
            .push_event(make_trade(3, 2, dec!(67000), dec!(1.0), Side::Sell))
            .unwrap();

        replay.replay_all().unwrap();

        assert_eq!(replay.stats().total_trade_volume, dec!(1.5));
        // Notional: 0.5*67001 + 1.0*67000 = 33500.5 + 67000 = 100500.5
        let expected_notional = dec!(0.5) * dec!(67001) + dec!(1.0) * dec!(67000);
        assert_eq!(replay.stats().total_trade_notional, expected_notional);
    }

    #[test]
    fn test_replay_trade_tracking_disabled() {
        let config = L2ReplayConfig::default().with_trade_tracking(false);
        let mut replay = L2Replay::new("BTC/USDT", config);

        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_trade(2, 1, dec!(67001), dec!(0.5), Side::Buy))
            .unwrap();

        replay.replay_all().unwrap();

        // Trades still update last price, but volume is not tracked.
        assert_eq!(replay.last_trade_price(), Some(dec!(67001)));
        assert_eq!(replay.stats().trades_processed, 0);
        assert_eq!(replay.stats().total_trade_volume, dec!(0));
    }

    // ── L2Replay Display ───────────────────────────────────────────────

    #[test]
    fn test_replay_display() {
        let replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        let s = format!("{}", replay);
        assert!(s.contains("BTC/USDT"));
        assert!(s.contains("Idle"));
    }

    #[test]
    fn test_replay_debug() {
        let replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());
        let s = format!("{:?}", replay);
        assert!(s.contains("L2Replay"));
        assert!(s.contains("BTC/USDT"));
    }

    // ── L2Replay Delta Removes Level ───────────────────────────────────

    #[test]
    fn test_replay_delta_removes_level() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());

        replay.push_event(make_snapshot(1, 0)).unwrap();
        // Remove the bid at 66999 by setting qty to 0.
        replay
            .push_event(make_delta(2, 1, Side::Buy, dec!(66999), dec!(0)))
            .unwrap();

        replay.replay_all().unwrap();

        assert_eq!(replay.book().bids().len(), 1);
        assert!(replay.book().bids().get_level(dec!(66999)).is_none());
    }

    // ── L2Replay Full Scenario ─────────────────────────────────────────

    #[test]
    fn test_replay_full_scenario() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());

        // 1. Initial snapshot.
        replay.push_event(make_snapshot(1, 0)).unwrap();

        // 2. Some deltas.
        replay
            .push_event(make_delta(2, 1, Side::Buy, dec!(67000), dec!(2.5)))
            .unwrap();
        replay
            .push_event(make_delta(3, 2, Side::Sell, dec!(67003), dec!(1.0)))
            .unwrap();

        // 3. A trade.
        replay
            .push_event(make_trade(4, 3, dec!(67001), dec!(0.3), Side::Buy))
            .unwrap();

        // 4. Remove a level.
        replay
            .push_event(make_delta(5, 4, Side::Sell, dec!(67001), dec!(0)))
            .unwrap();

        // 5. Another snapshot (replaces everything).
        replay
            .push_event(L2Event::Snapshot {
                symbol: "BTC/USDT".into(),
                bids: vec![L2Level::new(dec!(67005), dec!(1.0))],
                asks: vec![L2Level::new(dec!(67006), dec!(0.5))],
                timestamp: ts(10),
                sequence: 6,
            })
            .unwrap();

        let stats = replay.replay_all().unwrap();

        assert_eq!(stats.events_applied, 6);
        assert_eq!(stats.snapshots_applied, 2);
        assert_eq!(stats.deltas_applied, 3);
        assert_eq!(stats.trades_processed, 1);

        // Final book state should reflect the second snapshot.
        assert_eq!(replay.book().bids().len(), 1);
        assert_eq!(replay.book().asks().len(), 1);
        assert_eq!(replay.book().best_bid_price(), Some(dec!(67005)));
        assert_eq!(replay.book().best_ask_price(), Some(dec!(67006)));
    }

    // ── TimedEvent Ordering ────────────────────────────────────────────

    #[test]
    fn test_timed_event_ordering() {
        let mut heap = BinaryHeap::new();

        heap.push(TimedEvent {
            event: make_snapshot(3, 30),
            insertion_order: 2,
        });
        heap.push(TimedEvent {
            event: make_snapshot(1, 10),
            insertion_order: 0,
        });
        heap.push(TimedEvent {
            event: make_snapshot(2, 20),
            insertion_order: 1,
        });

        // Should come out in chronological order (earliest first).
        let first = heap.pop().unwrap();
        assert_eq!(first.event.sequence(), 1);

        let second = heap.pop().unwrap();
        assert_eq!(second.event.sequence(), 2);

        let third = heap.pop().unwrap();
        assert_eq!(third.event.sequence(), 3);
    }

    #[test]
    fn test_timed_event_same_timestamp_preserves_insertion_order() {
        let mut heap = BinaryHeap::new();

        heap.push(TimedEvent {
            event: make_delta(1, 10, Side::Buy, dec!(100), dec!(1)),
            insertion_order: 0,
        });
        heap.push(TimedEvent {
            event: make_delta(2, 10, Side::Sell, dec!(101), dec!(1)),
            insertion_order: 1,
        });
        heap.push(TimedEvent {
            event: make_delta(3, 10, Side::Buy, dec!(99), dec!(1)),
            insertion_order: 2,
        });

        // Same timestamp → should come out in insertion order.
        let first = heap.pop().unwrap();
        assert_eq!(first.insertion_order, 0);

        let second = heap.pop().unwrap();
        assert_eq!(second.insertion_order, 1);

        let third = heap.pop().unwrap();
        assert_eq!(third.insertion_order, 2);
    }

    // ── L2Replay Replay Clock ──────────────────────────────────────────

    #[test]
    fn test_replay_clock_advances() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());

        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_delta(2, 5, Side::Buy, dec!(67000), dec!(2.0)))
            .unwrap();

        replay.step().unwrap().unwrap();
        assert_eq!(replay.replay_clock(), Some(ts(0)));

        replay.step().unwrap().unwrap();
        assert_eq!(replay.replay_clock(), Some(ts(5)));
    }

    // ── L2Replay Sequence Tracking ─────────────────────────────────────

    #[test]
    fn test_replay_last_sequence_tracking() {
        let mut replay = L2Replay::new("BTC/USDT", L2ReplayConfig::default());

        replay.push_event(make_snapshot(1, 0)).unwrap();
        replay
            .push_event(make_delta(2, 1, Side::Buy, dec!(67000), dec!(2.0)))
            .unwrap();
        replay
            .push_event(make_delta(3, 2, Side::Sell, dec!(67001), dec!(1.5)))
            .unwrap();

        replay.replay_all().unwrap();

        assert_eq!(replay.stats().last_sequence, 3);
    }

    // ── L2Replay Edge: Zero Sequence ───────────────────────────────────

    #[test]
    fn test_replay_zero_sequence_no_gap_check() {
        let config = L2ReplayConfig::default().with_strict_sequencing(true);
        let mut replay = L2Replay::new("BTC/USDT", config);

        // Sequence 0 should bypass gap checking.
        replay
            .push_event(L2Event::Snapshot {
                symbol: "BTC/USDT".into(),
                bids: vec![L2Level::new(dec!(100), dec!(1))],
                asks: vec![L2Level::new(dec!(101), dec!(1))],
                timestamp: ts(0),
                sequence: 0,
            })
            .unwrap();

        replay
            .push_event(L2Event::Delta {
                symbol: "BTC/USDT".into(),
                side: Side::Buy,
                price: dec!(100),
                quantity: dec!(2),
                timestamp: ts(1),
                sequence: 0,
            })
            .unwrap();

        replay.step().unwrap().unwrap();
        let result = replay.step().unwrap();
        assert!(result.is_ok()); // No sequence gap error for seq=0.
        assert_eq!(replay.stats().sequence_gaps, 0);
    }
}
