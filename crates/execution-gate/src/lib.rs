//! Execution gate — centralized trade-entry filtering for JANUS.
//!
//! A chain-of-responsibility that runs a prospective trade entry through a
//! series of gates in priority order, stopping at the first BLOCK. This is a
//! faithful Rust port of the Python trading system's
//! `services/execution_gate.py` (the `nuniesmith/ruby` repo), brought into janus
//! as part of the Rust migration (see `fks/docs/architecture/RUST_MIGRATION.md`
//! §6 Phase 3 / §12-D, and the janus `TODO.md`).
//!
//! # Gate order
//!
//! 1. [`GateVerdict::BlockCircuitBreaker`] — consecutive-loss breaker open
//! 2. [`GateVerdict::BlockRisk`] — risk engine forbids trading
//! 3. [`GateVerdict::BlockVolFilter`] — volatility outside the entry band
//! 4. [`GateVerdict::BlockQuality`] — signal quality too low *(first entries only)*
//! 5. [`GateVerdict::BlockAoDivergence`] — Awesome-Oscillator sign disagrees *(first entries only)*
//! 6. [`GateVerdict::BlockFeeViability`] — fees too large vs. take-profit
//! 7. [`GateVerdict::BlockCnnDisagree`] — CNN disagrees with the side
//! 8. [`GateVerdict::BlockCnnConfidence`] — CNN confidence below threshold
//! 9. [`GateVerdict::BlockCorrelation`] — too many correlated open positions
//!
//! # Design
//!
//! Evaluation ([`ExecutionGate::evaluate`]) is **pure and synchronous**: it takes
//! a [`GateContext`] of inputs plus the current unix time and returns a
//! `(GateVerdict, reason)`. The three stateful collaborators — the
//! [`ConsecutiveLossBreaker`], [`CorrelationGuard`], and [`AdaptiveThreshold`] —
//! are owned by the gate and updated out-of-band by the trading loop
//! ([`ExecutionGate::record_trade_outcome`], [`ExecutionGate::update_correlation`]).
//! This keeps the whole crate dependency-free and trivially unit-testable, which
//! matters for a safety-critical component.
//!
//! Persistence (Redis-backed breaker / threshold state and dashboard block
//! counters) is intentionally **not** in this crate; it belongs in a thin
//! adapter at the wiring layer. See the janus `TODO.md` for that follow-up and
//! for the plan to wire this gate into `services/forward`'s live signal loop
//! behind a config flag (advisory first, then enforcing — mirroring how the
//! prop-firm / risk-manager enforcement landed).
//!
//! # Example
//!
//! ```
//! use janus_execution_gate::{ExecutionGate, GateContext, GateVerdict, Side};
//!
//! let mut gate = ExecutionGate::default();
//! let ctx = GateContext {
//!     can_trade: true,
//!     quality: 72.0,
//!     ao: 1.5,
//!     tp_pct: 0.01,
//!     ..Default::default()
//! };
//! let now = 1_700_000_000; // unix seconds
//! let (verdict, _reason) = gate.evaluate("btc", Side::Buy, false, &ctx, now);
//! assert_eq!(verdict, GateVerdict::Pass);
//! ```

pub mod adaptive_threshold;
pub mod circuit_breaker;
pub mod context;
pub mod correlation;
pub mod gate;
pub mod verdict;

pub use adaptive_threshold::AdaptiveThreshold;
pub use circuit_breaker::{BreakerSnapshotEntry, ConsecutiveLossBreaker};
pub use context::{CnnVote, GateContext, Side};
pub use correlation::CorrelationGuard;
pub use gate::{ExecutionGate, GateMetrics, GateMetricsSnapshot};
pub use verdict::GateVerdict;
